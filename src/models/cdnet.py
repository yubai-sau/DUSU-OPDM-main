import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import signed_sqrt_l2
from .kan import KAN
class SiameseEncoder(nn.Module):
    def __init__(self, in_ch, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(base),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(base),
            nn.MaxPool2d(2, ceil_mode=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.net(x)
        v = self.pool(f).flatten(1)
        return v

class CDnet(nn.Module):
    def __init__(self, P, base=64):
        super().__init__()
        self.enc = SiameseEncoder(P, base=base)
        D = base
        # self.classifier = nn.Linear(D*D, 2)
        # 例如：输入层(D*D) -> 隐藏层(64) -> 输出层(2)
        self.classifier = KAN([D * D, 64, 2])
        # out = self.Cross_fc_out(out)

    def forward(self, A1, A2):
        v1 = self.enc(A1)
        v2 = self.enc(A2)
        outer = torch.bmm(v1.unsqueeze(2), v2.unsqueeze(1))
        feat = outer.flatten(1)
        feat = signed_sqrt_l2(feat)
        logits = self.classifier(feat)
        return logits
