import torch.nn as nn
import numpy as np
import torch


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) loss function for semantic segmentation.

    Args:
        epsilon (float): A small value to avoid division by zero.

    Attributes:
        epsilon (float): A small value to avoid division by zero.

    Methods:
        _op_sum(x): Computes the sum of elements in the input tensor along the last two dimensions.
        forward(y_pred, y_true): Computes the IoU loss between the predicted and ground truth heatmaps.

    """

    def __init__(self, epsilon=1e-6):
        super(IoULoss, self).__init__()
        self.epsilon = epsilon

    def _op_sum(self, x):
        """
        Computes the sum of elements in the input tensor along the last two dimensions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sum of elements along the last two dimensions.

        """
        return x.sum(-1).sum(-1)

    def forward(self, y_pred: np.array, y_true: np.array) -> float:
        """
        Computes the IoU loss between the predicted and ground truth heatmaps.

        Args:
            y_pred (np.array): Predicted heatmap.
            y_true (np.array): Ground truth heatmap.

        Returns:
            float: IoU loss.

        """
        inter = self._op_sum(y_true * y_pred)
        union = (
            self._op_sum(y_true ** 2)
            + self._op_sum(y_pred ** 2)
            - self._op_sum(y_true * y_pred)
        )
        iou = (inter + self.epsilon) / (union + self.epsilon)
        iou = torch.mean(iou)

        return 1 - iou


class EffHandEgoNetLoss3D(nn.Module):
    """
    This class defines the loss function for the EffHandEgoNet model.

    Args:
    - None

    Returns:
    - None
    """

    def __init__(self):
        """
        Initializes the loss function.

        Args:
        - None

        Returns:
        - None
        """
        super(EffHandEgoNetLoss3D, self).__init__()

        self.handness = nn.CrossEntropyLoss()
        self.iou = IoULoss()
        self.depth_loss = nn.L1Loss()
        self.batch_activation = nn.Sigmoid()

    def forward(self, results, targets):

        # Handness loss
        handness_loss_left = self.handness(
            results['left_handness'], targets["handness"][0])
        handness_loss_right = self.handness(
            results['right_handness'], targets["handness"][1])
        handness_loss = 0.5*(handness_loss_left + handness_loss_right)

        # Substruct non existing
        batch_left = torch.argmax(
            self.batch_activation(results['left_handness']), dim=1)
        batch_right = torch.argmax(
            self.batch_activation(results['right_handness']), dim=1)
        left_heatmap_pred = results['heatmaps'][:, :21, :, :][batch_left != 0]
        left_heatmap_gt = targets['heatmaps'][:, :21, :, :][batch_left != 0]
        right_heatmap_pred = results['heatmaps'][:,
                                                 21:, :, :][batch_right != 0]
        right_heatmap_gt = targets['heatmaps'][:, 21:, :, :][batch_right != 0]

        # IOU loss fro 2D pose
        iou_left = self.iou(left_heatmap_pred, left_heatmap_gt)
        iou_right = self.iou(right_heatmap_pred, right_heatmap_gt)
        iou_loss = 0.5*(iou_left + iou_right)

        z_gt = targets['pose_3d_gt'][:, :, 2:].reshape(-1, 42)
        z_pred = results["z"].reshape(-1, 42)
        z_pred_left = z_pred[:, :21][batch_left != 0]
        z_pred_right = z_pred[:, 21:][batch_right != 0]
        z_gt_left = z_gt[:, :21][batch_left != 0]
        z_gt_right = z_gt[:, 21:][batch_right != 0]

        z_loss_left = self.depth_loss(z_pred_left, z_gt_left) / 1000
        z_loss_right = self.depth_loss(z_pred_right, z_gt_right) / 1000

        z_loss = 0.5*(z_loss_left + z_loss_right)

        loss_hands = iou_loss.clone()
        loss_hands.add_(0.3 * z_loss)

        final_loss = handness_loss.clone()
        final_loss.mul_(0.02).add_(0.98 * loss_hands)

        return final_loss, z_loss
