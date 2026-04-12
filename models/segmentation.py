import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg11 import VGG11


class DoubleConv(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DiceCELoss(nn.Module):

    def __init__(self, num_classes: int = 3, alpha: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, logits, targets):
        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        inter   = (probs * one_hot).sum(dim=(0, 2, 3))
        total   = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice    = (2.0 * inter + self.eps) / (total + self.eps)
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        return self.alpha * self.ce(logits, targets) + \
               (1 - self.alpha) * self.dice_loss(logits, targets)


class UNetVGG11(nn.Module):
    
    NUM_SEG_CLASSES = 3

    def __init__(self, pretrained_vgg: VGG11):
        super().__init__()

        # Encoder (from pretrained VGG11)
        self.enc1 = pretrained_vgg.block1   # [B,  64, 112, 112]
        self.enc2 = pretrained_vgg.block2   # [B, 128,  56,  56]
        self.enc3 = pretrained_vgg.block3   # [B, 256,  28,  28]
        self.enc4 = pretrained_vgg.block4   # [B, 512,  14,  14]
        self.enc5 = pretrained_vgg.block5   # [B, 512,   7,   7]

        # Decoder (symmetric, learnable transposed convolutions)
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   #  7->14
        self.dec5 = DoubleConv(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # 14->28
        self.dec4 = DoubleConv(256 + 256, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # 28->56
        self.dec3 = DoubleConv(128 + 128, 128)

        self.up2  = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)   # 56->112
        self.dec2 = DoubleConv(64 + 64, 64)

        self.up1  = nn.ConvTranspose2d(64,   32, kernel_size=2, stride=2)   # 112->224
        self.dec1 = DoubleConv(32, 32)

        self.out_conv = nn.Conv2d(32, self.NUM_SEG_CLASSES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)    # [B,  64, 112, 112]
        s2 = self.enc2(s1)   # [B, 128,  56,  56]
        s3 = self.enc3(s2)   # [B, 256,  28,  28]
        s4 = self.enc4(s3)   # [B, 512,  14,  14]
        s5 = self.enc5(s4)   # [B, 512,   7,   7]

        # Decoder with skip connections
        d = self.dec5(torch.cat([self.up5(s5), s4], dim=1))
        d = self.dec4(torch.cat([self.up4(d),  s3], dim=1))
        d = self.dec3(torch.cat([self.up3(d),  s2], dim=1))
        d = self.dec2(torch.cat([self.up2(d),  s1], dim=1))
        d = self.dec1(self.up1(d))

        return self.out_conv(d)   # [B, 3, 224, 224]
