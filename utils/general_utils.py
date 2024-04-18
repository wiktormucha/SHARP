import numpy as np
import cv2 as cv2
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random

COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g"},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c"},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b"},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m"},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r"},
}


def vector_to_heatmaps(keypoints: np.array, scale_factor: int = 1, out_size: int = 128, n_keypoints: int = 21) -> np.array:
    """
    Creates 2D heatmaps from keypoint locations for a single image.

    Args:
        keypoints (np.array): array of size N_KEYPOINTS x 2
        scale_factor (int, optional): Factor to scale keypoints (factor = 1 when keypoints are org size). Defaults to 1.
        out_size (int, optional): Size of output heatmap. Defaults to MODEL_IMG_SIZE.

    Returns:
        np.array: Heatmap
    """
    heatmaps = np.zeros([n_keypoints, out_size, out_size])
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x * scale_factor), int(y * scale_factor)
        if (0 <= x < out_size) and (0 <= y < out_size):
            heatmaps[k, int(y), int(x)] = 1

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


def blur_heatmaps(heatmaps: np.array) -> np.array:
    """
    Blurs heatmaps using GaussinaBlur of defined size

    Args:
        heatmaps (np.array): Input heatmap

    Returns:
        np.array: Output heatmap
    """
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / \
                heatmaps_blurred[k].max()
    return heatmaps_blurred


def project_points_3D_to_2D(xyz: np.array, K: np.array) -> np.array:
    """
    Projects 3D coordinates into 2D space. Taken from FreiHAND dataset repository.

    Args:
        xyz (np.array): 3D keypoints
        K (np.array): camera intrinsic

    Returns:
        np.array: 2D keypoints
    """
    uv = np.matmul(xyz, K.T)
    return uv[:, :2] / uv[:, -1:]


# def project_points_2D_to_3D(xy: np.array, z: np.array, K: list) -> np.array:
#     """
#     Projects 2D coordinates into 3D space.

#     Args:
#         xy (np.array): 2D keypoints
#         z (np.array): estimated depth
#         K (list): camera intrinsic

#     Returns:
#         np.array: 3D keypoints
#     """
#     xy = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1)
#     K_inv = np.linalg.inv(np.array(K))
#     xyz = np.matmul(K_inv, xy.T).T
#     xyz *= z.reshape(21, 1)
#     return xyz

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


def heatmaps_to_coordinates_tensor(heatmaps: torch.Tensor, num_kpts, img_size: int) -> torch.Tensor:
    """
    Transforms heatmaps to 2d keypoints

    Args:
        heatmaps (torch.Tensor): Input heatmap

    Returns:
        torch.Tensor: Output points
    """
    batch_size = heatmaps.shape[0]
    sums = heatmaps.sum(dim=-1).sum(dim=-1)

    sums = sums.unsqueeze(-1).unsqueeze(-1)
    normalized = heatmaps / sums

    x_prob = normalized.sum(dim=2)
    y_prob = normalized.sum(dim=3)

    arr = torch.arange(0, img_size, device=heatmaps.device).float().repeat(
        batch_size, num_kpts, 1)

    x = (arr * x_prob).sum(dim=2)
    y = (arr * y_prob).sum(dim=2)

    keypoints = torch.stack([x, y], dim=-1)

    return keypoints / img_size


def make_optimiser(model, training_cfg):

    optimiser = optim.SGD(model.parameters(
    ), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay, momentum=training_cfg.momentum)

    # optimiser = optim.AdamW(
    #     model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)

    return optimiser


def define_optimizer(model, optimizer_cfg):

    if optimizer_cfg.type == 'SGD':
        optimizer = optim.SGD(model.parameters(
        ), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay, momentum=optimizer_cfg.momentum)
    elif optimizer_cfg.type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
    return optimizer


def draw_keypoints_on_single_hand(pts, linewidth=1, colormap=None):

    pts_draw = pts.copy()

    if colormap == None:
        colormap = COLORMAP

    for finger, params in colormap.items():
        plt.plot(
            pts_draw[params["ids"], 0],
            pts_draw[params["ids"], 1],
            params["color"], linewidth=linewidth
        )


def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def freeze_seeds(seed_num=42):
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)


def set_max_cpu_threads(max_threads=16):
    torch.set_num_threads(max_threads)
