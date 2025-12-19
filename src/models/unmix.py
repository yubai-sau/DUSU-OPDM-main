import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ProxConv

class UnmixNet(nn.Module):
    '''
    Global endmembers E in R^{C x P}; abundance maps A in R^{P x H x W}.
    K-stage unfolded gradient + proximal refinement + softmax simplex projection.
    '''
    def __init__(self, C, P=8, K=4):
        super().__init__()
        self.C, self.P, self.K = C, P, K
        self.E = nn.Parameter(torch.randn(C, P) * 0.01)
        self.alpha_A = nn.Parameter(torch.tensor(0.5))
        self.proxA = nn.ModuleList([ProxConv(P, depth=3) for _ in range(K)])

    def project_simplex(self, A, temp=1.0):
        return F.softmax(A / temp, dim=1)

    def forward_once(self, Y):
        A = F.softmax(F.conv2d(Y, self.E.t().unsqueeze(-1).unsqueeze(-1)), dim=1)  # (B,P,H,W)
        E = self.E
        for k in range(self.K):
            EA = torch.einsum('cp,bphw->bchw', E, A)
            resid = EA - Y
            gradA = torch.einsum('cp,bchw->bphw', E, resid)
            A = A - self.alpha_A * gradA
            A = self.proxA[k](A)
            A = self.project_simplex(A, temp=1.0)
        Yhat = torch.einsum('cp,bphw->bchw', E, A)
        return A, Yhat, E

    def forward(self, Y1, Y2, share_E=True):
        A1, Y1h, E1 = self.forward_once(Y1)
        if share_E:
            A2, Y2h, _ = self.forward_once(Y2)
            E2 = E1
        else:
            A2, Y2h, _ = self.forward_once(Y2)
            E2 = E1
        return (A1, Y1h, E1), (A2, Y2h, E2)
