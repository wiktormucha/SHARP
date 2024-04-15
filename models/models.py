from models.backbones import BackboneModel
import numpy as np
from torch import nn
import torch.nn as nn
import numpy as np
import numpy as np
import torch
from utils.general_utils import heatmaps_to_coordinates_tensor

CAM_INTRS = np.array([[636.6593017578125, 0.00000000e+00, 635.283881879317],
                      [0.00000000e+00, 636.251953125, 366.8740353496978],
                      [0.00000000e+00, 0.00000000e+00, 1.0]])


def project_points_2D_to_3D(xyz: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Projects 2D coordinates into 3D space.

    Args:
        xyz (torch.Tensor): 2D keypoints and estimated depth in batches
        K (torch.Tensor): camera intrinsic

    Returns:
        torch.Tensor: 3D keypoints in batches
    """
    xy = xyz[:, :, :-1]
    z = xyz[:, :, -1:]
    ones = torch.ones((xy.shape[0], xy.shape[1], 1),
                      device=xy.device, dtype=xy.dtype)
    xy = torch.cat((xy, ones), dim=-1)
    K_inv = torch.inverse(K).to(xy.device).to(xy.dtype)
    xyz = torch.matmul(K_inv.unsqueeze(
        0), xy.transpose(-2, -1)).transpose(-2, -1)
    xyz *= z
    return xyz


class DeconvolutionLayer2(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2, padding=0, last=False) -> None:

        super().__init__()

        self.last = last

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:

        out = self.deconv(x)

        if not self.last:
            out = self.norm(out)

        return out


class SimpleHead50(nn.Module):
    def __init__(self, in_channels=1280, out_channels=21) -> None:
        super().__init__()

        self.deconv1 = DeconvolutionLayer2(
            in_channels=in_channels, out_channels=256)
        self.deconv2 = DeconvolutionLayer2(in_channels=256, out_channels=256)
        self.deconv3 = DeconvolutionLayer2(in_channels=256, out_channels=256)
        self.deconv4 = DeconvolutionLayer2(in_channels=256, out_channels=256)
        self.deconv5 = DeconvolutionLayer2(
            in_channels=256, out_channels=256, last=True)

        self.final = torch.nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv5(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x


class CustomEgocentric3D_zsep(nn.Module):
    def __init__(self, handness_in: int = 81920, handness_out: int = 2, *args, **kwargs) -> None:
        """Initilise the model
        Args:
            handness_in (int, optional): _description_. Defaults to 81920 for 512 input image. 11520 for 224
            handness_out (int, optional): _description_. Defaults to 2, binary classification.
        """
        dim = 81920
        # dim = 11520
        super().__init__(*args, **kwargs)

        self.backbone = BackboneModel()

        self.left_hand = nn.Linear(
            in_features=dim, out_features=handness_out)
        self.right_hand = nn.Linear(
            in_features=dim, out_features=handness_out)
        self.pooling = nn.MaxPool2d(2)

        self.left_pose = SimpleHead50()
        self.right_pose = SimpleHead50()

        self.z_estimation_l = nn.Linear(in_features=dim, out_features=21)
        self.z_estimation_r = nn.Linear(in_features=dim, out_features=21)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): input tensor shape B,3,512,512

        Returns:
            torch.Tensor: handness_lef, handness_right, left pose, right pose
        """

        ret_dict = {}
        features = self.backbone(x)
        flatten = torch.flatten(self.pooling(features), 1)

        left_2D_pose = self.left_pose(features)
        right_2D_pose = self.right_pose(features)
        depth_estimation_l = self.z_estimation_l(flatten)
        depth_estimation_r = self.z_estimation_r(flatten)

        heatmaps = torch.cat((left_2D_pose, right_2D_pose), 1)

        ret_dict['heatmaps'] = heatmaps

        ret_dict['left_handness'] = self.left_hand(flatten)
        ret_dict['right_handness'] = self.right_hand(flatten)
        depth_estimation = torch.cat(
            (depth_estimation_l, depth_estimation_r), 1)
        ret_dict['z'] = depth_estimation

        if not self.training:

            kpts2d_img = heatmaps_to_coordinates_tensor(
                heatmaps=heatmaps, num_kpts=42, img_size=left_2D_pose.shape[-1])

            kpts2d_img = kpts2d_img * \
                torch.tensor([1280.0, 720.0]).to(kpts2d_img.device)

            kpts25d = torch.cat((kpts2d_img, depth_estimation.reshape(
                depth_estimation.shape[0], 42, 1)), 2)
            kpts3d = project_points_2D_to_3D(
                xyz=kpts25d, K=torch.tensor(CAM_INTRS))

            ret_dict['kpts_3d_cam'] = kpts3d
            ret_dict['kpts25d'] = kpts25d
            ret_dict['kpts2d_img'] = kpts2d_img

        return ret_dict


def count_parameters(model: nn.Module) -> int:
    """
    Counts parameters for training in a given model.

    Args:
        model (nn.Module): Input model

    Returns:
        int: No. of trainable parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(model_cfg, device='cpu', parameter_info=True):

    ModelClass = globals()[model_cfg.model_type]
    model = ModelClass()

    model = model.to(device)

    print(f'Model created on device: {device}')

    # If loading weights from checkpoin
    if model_cfg.load_model:
        model.load_state_dict(torch.load(
            model_cfg.load_model_path, map_location=torch.device(device)))
        print("Model's checkpoint loaded")

    if parameter_info:
        print('Number of parameters to learn:', count_parameters(model))

    return model
