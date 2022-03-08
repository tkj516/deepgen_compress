import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

class CodeBP(nn.Module):

    def __init__(self, H, device='cuda:0'):

        super(CodeBP, self).__init__()

        # Get the shape of the parity check matrix
        self.K, self.N = H.shape
        self.H = H

        # Set an epsilon for high precision calculations
        self.epsilon = 1e-15

        # Initialize the messages to all 0
        # Set them as parameters for gradient backpropagation
        self.Hxs = nn.Parameter(torch.zeros(self.K, self.N))
        self.M_out = nn.Parameter(torch.zeros(self.N, 2))

        # Set the device
        self.device = device

    def reset(self):

        self.Hxs = nn.Parameter(torch.zeros(self.K, self.N).to(self.device))
        self.M_out = nn.Parameter(torch.zeros(self.N, 2).to(self.device))

    def forward(self, ps, x, Min):
        """
        Min : N x 2
        ps  : N x 2
        x   : K x 2
        Mout: N x 2
        """

        infcap = 100
        eps = 0.0001

        if Min is None:
            Min = 0.5 * torch.ones(self.N, 2).to(self.device)

        ps = ps / torch.sum(ps, dim=1, keepdim=True)
        Min = Min / torch.sum(Min, dim=1, keepdim=True)

        dope_msg = torch.log(1 - ps[:, 1:]) - torch.log(ps[:, 1:])
        phi = torch.log(1 - Min[:, 1:]) - torch.log(Min[:, 1:])
        prior = (dope_msg + phi).reshape(1, -1) + torch.sum(self.Hxs.data, dim=0, keepdim=True)
        # prior = (dope_msg + phi).reshape(1, -1) + torch.sum(self.Hxs.data * self.H, dim=0, keepdim=True)

        # Hsx = self.H @ torch.diag(torch.clamp(prior.reshape(-1, ), min=-infcap, max=infcap)) - self.Hxs.data + eps * self.H
        Hsx = torch.clamp(prior, min=-infcap, max=infcap) * self.H - self.Hxs.data + eps * self.H
        print(Hsx)
        if torch.any(torch.isnan(Hsx)):
            print("Hsx is NaN")
            exit(0)

        tanhm = torch.tanh(Hsx / 2)
        zero_mask = tanhm.clone() == 0
        tanhm_mod = tanhm.clone()
        tanhm_mod[zero_mask] = 1
        prodtanhm = torch.prod(tanhm_mod, dim=-1, keepdim=True)

        prodtanhmdivt = prodtanhm / tanhm
        prodtanhmdivt[zero_mask] = 0
        # multbyx = 2 * torch.atanh(torch.diag(1 - 2 * x.reshape(-1, )) @ prodtanhmdivt)
        multbyx = 2 * torch.atanh((1 - 2 * x) * prodtanhmdivt)

        self.Hxs.data = torch.clamp(multbyx, min=-1, max=1)
        if torch.any(torch.isnan(self.Hxs.data)):
            print("Hxs is NaN")
            exit(0)

        z_prob = (1 - torch.tanh((torch.sum(self.Hxs, dim=0, keepdim=True).T + dope_msg) / 2)) / 2
        if torch.any(torch.isnan(z_prob)):
            print("zprob is NaN")
            exit(0)

        self.M_out.data = torch.cat([1 - z_prob, z_prob], dim=1)