import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout
from models.segmentation import DoubleConv


def conv_bn_relu(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):

    IMG_SIZE = 224   

    def __init__(
        self,
        num_classes: int = 37,
        dropout_p: float = 0.5,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path:  str = "checkpoints/localizer.pth",
        unet_path:       str = "checkpoints/unet.pth",
    ):
        super().__init__()

        import gdown

        gdown.download(id="111IT-1TqXz6Brx0q6W7rE_jJ6eqa7VF-", output=classifier_path, quiet=False)
        gdown.download(id="16Rbhi4BAJnDNWFxOcH7nGC07q32hhOFv", output=localizer_path, quiet=False)
        gdown.download(id="19J9VS3Tgg63_FSZund9I9rPcyunzQX86", output=unet_path, quiet=False)

        # ── Shared encoder (VGG11 blocks) ────────────────────────────────────
        self.block1 = nn.Sequential(conv_bn_relu(3, 64),    nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(conv_bn_relu(64, 128),  nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(conv_bn_relu(128, 256), conv_bn_relu(256, 256), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(conv_bn_relu(256, 512), conv_bn_relu(512, 512), nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(conv_bn_relu(512, 512), conv_bn_relu(512, 512), nn.MaxPool2d(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ── Head 1: Classification ───────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.BatchNorm1d(4096), nn.ReLU(True), CustomDropout(dropout_p),
            nn.Linear(4096, 4096),         nn.BatchNorm1d(4096), nn.ReLU(True), CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )

        # ── Head 2: Bounding Box Regression ─────────────────────────────────
        self.loc_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024), nn.BatchNorm1d(1024), nn.ReLU(True), CustomDropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(True),
            nn.Linear(256, 4),   nn.ReLU(True),   # pixel coords >= 0
        )

        # ── Head 3: U-Net Segmentation Decoder ──────────────────────────────
        self.up5  = nn.ConvTranspose2d(512, 512, 2, stride=2); self.dec5 = DoubleConv(512 + 512, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec4 = DoubleConv(256 + 256, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = DoubleConv(128 + 128, 128)
        self.up2  = nn.ConvTranspose2d(128,  64, 2, stride=2); self.dec2 = DoubleConv(64 + 64, 64)
        self.up1  = nn.ConvTranspose2d(64,   32, 2, stride=2); self.dec1 = DoubleConv(32, 32)
        self.seg_out = nn.Conv2d(32, 3, kernel_size=1)

        # ── Load pretrained weights into backbone + heads ────────────────────
        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load classifier -> shared encoder + cls_head
        cls_ckpt = torch.load(classifier_path, map_location=device)
        enc_keys = {k: v for k, v in cls_ckpt.items()
                    if k.startswith(("block1","block2","block3","block4","block5"))}
        cls_head_keys = {k.replace("classifier.", ""): v
                         for k, v in cls_ckpt.items() if k.startswith("classifier")}
        self.load_state_dict(enc_keys, strict=False)
        self.cls_head.load_state_dict(cls_head_keys, strict=False)

        # Load localizer -> loc_head
        loc_ckpt = torch.load(localizer_path, map_location=device)
        loc_head_keys = {k.replace("regressor.", ""): v
                         for k, v in loc_ckpt.items() if k.startswith("regressor")}
        self.loc_head.load_state_dict(loc_head_keys, strict=False)

        # Load unet -> decoder heads
        seg_ckpt = torch.load(unet_path, map_location=device)
        seg_keys = {k: v for k, v in seg_ckpt.items()
                    if k.startswith(("up","dec","seg_out"))}
        self.load_state_dict(seg_keys, strict=False)

    def forward(self, x: torch.Tensor):
        # Shared encoder
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        s5 = self.block5(s4)

        pooled = torch.flatten(self.avgpool(s5), 1)

        # Task heads
        cls_logits = self.cls_head(pooled)
        bbox_pred  = self.loc_head(pooled)

        d = self.dec5(torch.cat([self.up5(s5), s4], 1))
        d = self.dec4(torch.cat([self.up4(d),  s3], 1))
        d = self.dec3(torch.cat([self.up3(d),  s2], 1))
        d = self.dec2(torch.cat([self.up2(d),  s1], 1))
        d = self.dec1(self.up1(d))
        seg_logits = self.seg_out(d)

        return cls_logits, bbox_pred, seg_logits
