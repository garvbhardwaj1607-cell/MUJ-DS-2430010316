"""Models.

LeafNet            : small from-scratch CNN. Optional components used in the CO5 ablation
                       se=True            -> squeeze-and-excitation channel attention after each stage   (component A)
                       spatial_att=True   -> spatial attention gate on stage-3 features, which can be
                                             supervised with the leaf mask (loss in train.py)           (component C)
                     (component B, background randomisation, is a training-time augmentation in train.py)
ResNetSmall        : small residual CNN, reference architecture (trained from scratch, same split/budget)
MobileNetSmall     : depthwise-separable inverted-residual CNN, reference architecture
TinyViT            : small vision transformer (patch 12, 64 tokens), reference architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(i, o, k=3, s=1, g=1, act=True):
    layers = [nn.Conv2d(i, o, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(o)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean((2, 3)))
        return x * w[:, :, None, None]


class Stage(nn.Module):
    def __init__(self, i, o, se):
        super().__init__()
        self.body = nn.Sequential(conv_bn(i, o, 3, 2), conv_bn(o, o, 3, 1))
        self.se = SE(o) if se else nn.Identity()

    def forward(self, x):
        return self.se(self.body(x))


class LeafNet(nn.Module):
    def __init__(self, num_classes=38, w=24, se=False, spatial_att=False, dropout=0.3):
        super().__init__()
        self.stem = conv_bn(3, w, 3, 2)               # 96 -> 48
        self.s1 = Stage(w, 2 * w, se)                 # 24
        self.s2 = Stage(2 * w, 4 * w, se)             # 12
        self.use_att = spatial_att
        self.att = nn.Conv2d(4 * w, 1, 1) if spatial_att else None
        self.s3 = Stage(4 * w, 8 * w, se)             # 6
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(8 * w, num_classes)
        self.cam_layer = self.s3                      # Grad-CAM target
        self.last_att_logits = None

    def forward(self, x):
        x = self.s2(self.s1(self.stem(x)))
        if self.use_att:
            a = self.att(x)                           # (B,1,12,12) logits
            self.last_att_logits = a
            x = x * torch.sigmoid(a)
        x = self.s3(x)
        return self.fc(self.drop(x.mean((2, 3))))


class ResBlock(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.c1, self.c2 = conv_bn(i, o, 3, s), conv_bn(o, o, 3, 1, act=False)
        self.sc = nn.Identity() if (i == o and s == 1) else conv_bn(i, o, 1, s, act=False)

    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + self.sc(x))


class ResNetSmall(nn.Module):
    def __init__(self, num_classes=38, w=24):
        super().__init__()
        self.stem = conv_bn(3, w, 3, 2)
        self.layers = nn.Sequential(ResBlock(w, 2 * w, 2), ResBlock(2 * w, 2 * w, 1),
                                    ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 4 * w, 1),
                                    ResBlock(4 * w, 8 * w, 2), ResBlock(8 * w, 8 * w, 1))
        self.drop, self.fc = nn.Dropout(0.3), nn.Linear(8 * w, num_classes)
        self.cam_layer = self.layers[-1]

    def forward(self, x):
        return self.fc(self.drop(self.layers(self.stem(x)).mean((2, 3))))


class InvRes(nn.Module):
    def __init__(self, i, o, s, e=2):
        super().__init__()
        m = i * e
        self.use_res = s == 1 and i == o
        self.net = nn.Sequential(conv_bn(i, m, 1), conv_bn(m, m, 3, s, g=m), conv_bn(m, o, 1, act=False))

    def forward(self, x):
        return x + self.net(x) if self.use_res else self.net(x)


class MobileNetSmall(nn.Module):
    def __init__(self, num_classes=38, w=24):
        super().__init__()
        self.stem = conv_bn(3, w, 3, 2)
        self.layers = nn.Sequential(InvRes(w, 2 * w, 2), InvRes(2 * w, 2 * w, 1),
                                    InvRes(2 * w, 4 * w, 2), InvRes(4 * w, 4 * w, 1),
                                    InvRes(4 * w, 8 * w, 2), InvRes(8 * w, 8 * w, 1))
        self.head = conv_bn(8 * w, 16 * w, 1)
        self.drop, self.fc = nn.Dropout(0.3), nn.Linear(16 * w, num_classes)
        self.cam_layer = self.head

    def forward(self, x):
        return self.fc(self.drop(self.head(self.layers(self.stem(x))).mean((2, 3))))


class TinyViT(nn.Module):
    def __init__(self, num_classes=38, img=96, patch=12, dim=96, depth=4, heads=4):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, patch, patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm, self.fc = nn.LayerNorm(dim), nn.Linear(dim, num_classes)
        self.cam_layer = None

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.fc(self.norm(self.enc(t)[:, 0]))


class PretrainedTimm(nn.Module):
    """ImageNet-pretrained backbone (timm architecture, weights from the official timm GitHub release, loaded from
    models/pretrained/<name>.pth) with a new 38-way head. Fine-tuned end to end on the same split as every other model."""

    def __init__(self, name, num_classes=38, load=True):
        super().__init__()
        import os, timm
        self.net = timm.create_model(name, pretrained=False, num_classes=1000)
        path = os.path.join("models", "pretrained", f"{name}.pth")
        if load:
            sd = torch.load(path, map_location="cpu")
            self.net.load_state_dict(sd)                   # strict: fails loudly if weights do not match the arch
        self.net.reset_classifier(num_classes)
        self.cam_layer = self.net.blocks[-1]               # last convolutional stage (Grad-CAM target)

    def forward(self, x):
        return self.net(x)


VARIANTS = {
    # ---- CO4 references: ImageNet-pretrained, fine-tuned on the same split
    "ref_pre_mnv3": dict(model=lambda load=True: PretrainedTimm("mobilenetv3_large_100", load=load), bg_aug=False, mask_loss=False),
    "ref_pre_effb0": dict(model=lambda load=True: PretrainedTimm("efficientnet_b0", load=load), bg_aug=False, mask_loss=False),
    # ---- component B (background randomisation) applied on top of the pretrained backbone
    "pre_mnv3_bgaug": dict(model=lambda load=True: PretrainedTimm("mobilenetv3_large_100", load=load), bg_aug=True, mask_loss=False),
    # ---- CO5 ablation (same LeafNet backbone, components switched on one by one)
    "baseline": dict(model=lambda: LeafNet(), bg_aug=False, mask_loss=False),
    "A_se": dict(model=lambda: LeafNet(se=True), bg_aug=False, mask_loss=False),
    "AB_se_bgaug": dict(model=lambda: LeafNet(se=True), bg_aug=True, mask_loss=False),
    "ABC_proposed": dict(model=lambda: LeafNet(se=True, spatial_att=True), bg_aug=True, mask_loss=True),
    # ---- CO4 reference architectures (from scratch, same split and training budget)
    "ref_resnet": dict(model=lambda: ResNetSmall(), bg_aug=False, mask_loss=False),
    "ref_mobilenet": dict(model=lambda: MobileNetSmall(), bg_aug=False, mask_loss=False),
    "ref_tinyvit": dict(model=lambda: TinyViT(), bg_aug=False, mask_loss=False),
}


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
