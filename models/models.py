from models.backbones import BackboneModel
import torch.nn as nn
import torch
from utils.general_utils import heatmaps_to_coordinates_tensor, project_points_2D_to_3D
from datasets.h2o import CAM_INTRS
import numpy as np

def get_positional_embeddings(sequence_lenght, d, device, freq=10000):
    result = torch.ones(sequence_lenght, d)
    for i in range(sequence_lenght):
        for j in range(d):
            result[i][j] = np.sin(i / (freq ** (j / d)) if j %
                                  2 == 0 else np.cos(i / (freq ** ((j - 1) / d))))

    return result.to(device)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, seq_length, num_heads, dropout, dropout_att):

        super(TransformerEncoder, self).__init__()
        print(f'Embedding dimmension {embed_dim}')
        self.MultHeaAtten = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_att)

        self.layer_norm_1 = nn.LayerNorm((seq_length + 1, embed_dim))

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU())

        self.layer_norm_2 = nn.LayerNorm((seq_length + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Encoder Layer

        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        """
        # Calculating self attention
        attn_output, attn_output_weights = self.MultHeaAtten(
            query=x, key=x, value=x, key_padding_mask=None)
        # apply layer normalization on sum of the input and the attention output to get the
        # output of the multi-head attention layer (~1 line)
        out1 = self.layer_norm_1(attn_output + x)
        # pass the output of the multi-head attention layer through a ffn (~1 line)
        # (batch_size, input_seq_len, fully_connected_dim)
        ffn_output = self.ffn(out1)

        # apply layer normalization on sum of the output from multi-head attention and ffn output to get the
        # output of the encoder layer (~1 line)
        enc_out = self.layer_norm_2(out1 + ffn_output)
        enc_out = self.dropout(enc_out)

        return enc_out


class TransfomerKeypoints_loop_NoLinear(nn.Module):
    def __init__(self, model_cfg, input_dim=126, out_dim=37, hidden_d=126, device=0) -> None:
        super().__init__()

        input_dim = model_cfg.input_dim
        # hidden_layers =
        out_dim = model_cfg.out_dim
        dropout = model_cfg.dropout
        dropout_att = model_cfg.dropout_att
        dropout = model_cfg.dropout
        num_heads = model_cfg.trans_num_heads
        self.num_layers = model_cfg.trans_num_layers
        self.hidden_layers = model_cfg.hidden_layers
        self.seq_length = model_cfg.seq_length
        self.device = device

        self.input_norm = nn.LayerNorm(135)

        # 1) Linear mapper
        # self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(input_dim, self.hidden_layers)
        # self.linear_mapper = nn.Sequential(nn.Linear(input_dim, self.hidden_layers),
        #                                    nn.Dropout(0.1),
        #                                    )
        # 2) Learnable classifiation token
        # self.class_token = nn.Parameter(torch.rand(1, self.hidden_layers))
        self.class_token = nn.Parameter(torch.rand(1, input_dim))

        # self.encoder_layers = [TransformerEncoder(
        #     embed_dim=self.hidden_layers, seq_length=self.seq_length, num_heads=num_heads, dropout=dropout, dropout_att=dropout_att).to(device) for _ in range(self.num_layers)]
        self.encoder_layers = nn.ModuleList()
        # for _ in range(self.num_layers):
        #     self.encoder_layers.append(TransformerEncoder(
        #         embed_dim=self.hidden_layers, seq_length=self.seq_length, num_heads=num_heads, dropout=dropout, dropout_att=dropout_att).to(device))
        for _ in range(self.num_layers):
            self.encoder_layers.append(TransformerEncoder(
                embed_dim=input_dim, seq_length=self.seq_length, num_heads=num_heads, dropout=dropout, dropout_att=dropout_att).to(device))
        # 6) CLassification MLP
        self.mlp = nn.Sequential(
            nn.Dropout(model_cfg.dropout_mlp),
            nn.LayerNorm(136),
            nn.Linear(136, out_dim),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):

        batch_s, _, _ = x.shape
        # 1 Lineralize
        # print(x.shape)
        obj_label = x[:, :, -1]
        obj_label = obj_label[:, 0].reshape(batch_s, 1)
        # print(f'obj label: {obj_label.shape}')
        # print(f'obj label1: {obj_label[0]}')
        # print(f'obj label2: {obj_label[1]}')
        tokens = self.linear_mapper(x)
        # tokens = self.input_norm(x)
        # print(x.shape)

        # Adding classification token to the tokens
        tokens = torch.stack(
            [torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        # tokens += get_positional_embeddings(self.seq_length+1,
        #                                     self.hidden_layers, device=self.device).repeat(batch_s, 1, 1)
        tokens += get_positional_embeddings(self.seq_length+1,
                                            135, device=self.device).repeat(batch_s, 1, 1)
        # Encoder block
        x = tokens

        # for i in range(self.num_layers):
        #     x = self.encoder_layers[i](x)
        # Encoder block
        for blk in self.encoder_layers:
            x = blk(x)

        out = x[:, 0]
        # return x

        # obj = obj.reshape(batch_s, 1)

        # Merging with object label
        out = torch.cat((out, obj_label), dim=1)
        # print(f'out: {out.shape}')
        out = self.mlp(out)

        return out


class DeconvolutionLayer(nn.Module):

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


class Upsampler(nn.Module):
    def __init__(self, in_channels=1280, out_channels=21) -> None:
        super().__init__()

        self.deconv1 = DeconvolutionLayer(
            in_channels=in_channels, out_channels=256)
        self.deconv2 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv3 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv4 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv5 = DeconvolutionLayer(
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


class EffHandEgoNet3D(nn.Module):
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

        self.left_pose = Upsampler()
        self.right_pose = Upsampler()

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
