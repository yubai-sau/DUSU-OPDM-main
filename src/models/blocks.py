import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch//r), 1)
        self.fc2 = nn.Conv2d(max(1, ch//r), ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class ProxConv(nn.Module):
    """A light proximal refiner (conv-residual + squeeze-excitation)."""
    def __init__(self, ch, depth=3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(ch)]
        self.body = nn.Sequential(*layers)
        self.se = SEBlock(ch)

    def forward(self, x):
        out = self.body(x)
        out = self.se(out)
        return x + out

def signed_sqrt_l2(x, eps=1e-6):
    x = torch.sign(x) * torch.sqrt(torch.clamp(torch.abs(x), min=eps))
    x = F.normalize(x.flatten(1), dim=1)
    return x
