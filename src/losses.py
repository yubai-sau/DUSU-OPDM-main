import torch
import torch.nn.functional as F
import torch.nn as nn

def l1_recon(Y, Yhat):
    return F.l1_loss(Yhat, Y)

def tv_loss(A):
    dh = torch.abs(A[:,:,1:,:] - A[:,:,:-1,:]).mean()
    dw = torch.abs(A[:,:,:,1:] - A[:,:,:,:-1]).mean()
    return dh + dw

class JointLoss(nn.Module):
    def __init__(self, lambda_rec=0.5, lambda_tv=0.005, lambda_temp=0.2, use_temp=True):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_rec = float(lambda_rec)
        self.lambda_tv = float(lambda_tv)
        self.lambda_temp = float(lambda_temp)
        self.use_temp = bool(use_temp)

    def forward(self, logits, y, Y1, Y1h, Y2, Y2h, A1, A2, temp_mask=None):
        Lce = self.ce(logits, y) if logits is not None else 0.0
        Lrec = l1_recon(Y1, Y1h) + l1_recon(Y2, Y2h)
        Ltv = tv_loss(A1) + tv_loss(A2)
        Ltemp = 0.0
        if self.use_temp and (temp_mask is not None):
            mask = temp_mask.float().view(-1,1,1,1)
            diff = torch.abs(A1 - A2).mean(dim=1, keepdim=True)
            sim = (1.0 - mask) * diff.mean()
            dis = mask * (1.0 - diff).mean()
            Ltemp = sim + dis
        return Lce + self.lambda_rec*Lrec + self.lambda_tv*Ltv + self.lambda_temp*Ltemp
