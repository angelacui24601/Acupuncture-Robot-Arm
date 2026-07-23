import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """轻量级多任务基线：共享卷积编码器 + 分割头 + 关键点头。"""

    def __init__(self, opts):
        super().__init__()
        self.image_size = int(opts.image_size)
        self.num_classes = int(opts.num_classes)
        self.num_keypoints = 16

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_classes, kernel_size=1),
        )

        self.kpt_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_keypoints * 3),
        )

    def forward(self, img, _acupoint):
        feat = self.encoder(img)

        seg_logits = self.seg_head(feat)
        seg_logits = F.interpolate(
            seg_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        b = img.size(0)
        points = self.kpt_head(feat).view(b, self.num_keypoints, 3)
        return seg_logits, points
