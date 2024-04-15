import torch
import torch.nn as nn
import torchvision
from torchvision.models import EfficientNet_V2_S_Weights


class BackboneModel(nn.Module):
    """
    Efficientnet backbone without last max pooling. Outputs dimmensions of Bx4x4x1280.
    """

    def __init__(self):
        """
        Init
        """
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights)
        self.backbone = torch.nn.Sequential(
            *(list(self.backbone.children())[:-2]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        return self.backbone(x)
