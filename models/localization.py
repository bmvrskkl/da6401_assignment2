import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout


class LocalizationModel(nn.Module):

    def __init__(self, pretrained_vgg: VGG11):
        super().__init__()

        self.encoder = nn.Sequential(
            pretrained_vgg.block1,
            pretrained_vgg.block2,
            pretrained_vgg.block3,
            pretrained_vgg.block4,
            pretrained_vgg.block5,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.ReLU(inplace=True),   # pixel coords must be non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return self.regressor(feat)
