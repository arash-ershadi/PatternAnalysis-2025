import torch
import torch.nn as nn
import timm

class Classifier(nn.Module):
    """
    Enhanced ConvNeXt-based classifier for Alzheimer's classification.

    - Uses pretrained weights from the timm library
    - Proper feature extraction with adaptive pooling
    - Dropout for regularization

    Args:
        num_classes: Number of output classes (2 for AD/NC)
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability for regularization
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super().__init__()

        # Load pretrained ConvNeXt backbone (without classifier)
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling to add our own
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features

        # Add adaptive pooling to ensure fixed size features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Add dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Final classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # Initialize classifier weights
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, H, W] (grayscale)

        Returns:
            torch.Tensor: Logits of shape [B, num_classes]
        """
        # Convert grayscale to 3-channel by repeating
        # ConvNeXt expects 3 channels (RGB)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Extract features using backbone
        features = self.backbone.forward_features(x)

        # Global average pooling
        features = self.global_pool(features)
        features = torch.flatten(features, 1)

        # Apply dropout
        features = self.dropout(features)

        # Classification
        logits = self.classifier(features)

        return logits
