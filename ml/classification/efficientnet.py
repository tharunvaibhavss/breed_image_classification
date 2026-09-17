"""EfficientNet-B0 Breed Classifier PyTorch Module.

Implements transfer learning architecture for 6 Indian cattle and buffalo breeds.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class BreedClassifier(nn.Module):
    """EfficientNet-B0 PyTorch Breed Classification Model."""

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        """Initialize BreedClassifier.

        Args:
            num_classes: Number of output breed classes (default 6).
            pretrained: If True, loads ImageNet pretrained backbone weights.
            dropout_rate: Dropout probability for classifier head.
        """
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.backbone = efficientnet_b0(weights=weights)
            except Exception:
                # Fallback to uninitialized backbone if offline
                self.backbone = efficientnet_b0(weights=None)
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Retrieve input feature dimension of the classifier head
        in_features = self.backbone.classifier[1].in_features

        # Replace classification head for 6 breed output logits
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)
