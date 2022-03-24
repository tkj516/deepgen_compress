import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

class GridBP(nn.Module):

    def __init__(self, a=0.7, s=0.51, s0=1, mu0=0, n=1024, device='cuda:0'):

        super(GridBP, self).__init__()

        # Store model parameters
        self.a = a
        self.s = s
        self.s0 = s0
        self.mu0 = mu0
        self.n = n
        
        # Define the node and edge potentials
        # Register the potentials as buffers to indicate non-trainable members of the module
        self.h0 = 0  # self.m0 / self.s0
        self.hn = 0

        # Initialize the messages
        # Set them as parameters for gradient backpropagation
        self.hf = nn.Parameter(torch.zeros(n-1, 1))
        self.Lf = nn.Parameter(torch.ones(n-1, 1))
        self.hb = nn.Parameter(torch.zeros(n-1, 1))
        self.Lb = nn.Parameter(torch.ones(n-1, 1))

        # Node and edge potentials
        Li = torch.cat([torch.ones(1, 1) / s0, torch.ones(n-1, 1) / s], dim=0) + \
            torch.cat([a**2 * torch.ones(n-1, 1) / s, torch.zeros(1, 1)], dim=0)
        hi = torch.zeros(n, 1)
        self.register_buffer('Li', Li)
        self.register_buffer('hi', hi)

        self.Lij = -a / s

        # Store the device name
        self.device = device
        self.register_buffer('zero', torch.zeros(1, 1))
        self.register_buffer('one', torch.ones(1, 1))

    def forward(self, mu_in, var_in):
        """
        Min : h x w x 2
        Mout: h x w x 2
        """

        Lin = 1 / var_in
        hin = Lin * mu_in

        Lpass = Lin + self.Li
        hpass = hin + self.hi

        hf = -self.Lij / (Lpass[:-1, :] + torch.cat([self.zero, self.Lf[:-1, :]], dim=0)) * (hpass[:-1, :] + torch.cat([self.zero, self.hf[:-1, :]], dim=0))
        Lf = -self.Lij**2 / (Lpass[:-1, :] + torch.cat([self.zero, self.Lf[:-1, :]], dim=0))

        hb = -self.Lij / (Lpass[1:, :] + torch.cat([self.Lb[1:, :], self.zero], dim=0)) * (hpass[1:, :] + torch.cat([self.hb[1:, :], self.zero], dim=0))
        Lb = -self.Lij**2 / (Lpass[1:, :] + torch.cat([self.Lb[1:, :], self.zero], dim=0))

        Li_hat = self.Li + torch.cat([self.zero, Lf], dim=0) + torch.cat([Lb, self.zero], dim=0)
        hi_hat = self.hi + torch.cat([self.zero, hf], dim=0) + torch.cat([hb, self.zero], dim=0)

        self.Lf.data = Lf
        self.hf.data = hf
        self.Lb.data = Lb
        self.hb.data = hb

        s2s_prod_var = 1 / Li_hat
        s2s_prod_mean = s2s_prod_var * hi_hat

        return s2s_prod_mean, s2s_prod_var

    def reset_messages(self):

        # Reset the message parameters
        self.hf = nn.Parameter(torch.zeros(self.n-1, 1))
        self.Lf = nn.Parameter(torch.ones(self.n-1, 1))
        self.hb = nn.Parameter(torch.zeros(self.n-1, 1))
        self.Lb = nn.Parameter(torch.ones(self.n-1, 1))

    def bp(self, mu_in, var_in, num_iter=1):

        for i in tqdm(range(num_iter)):
            self.forward(mu_in, var_in)


    def source_to_quant(
        self,
        quant_mean,
        quant_var,
        min,
        max,
        width,
    ):

        bins = torch.arange(min + 1, max + 1).reshape(-1, 1, 1).to(self.device)
        inv_width = 1 / width

        def Phi(x):

            phi = lambda x: 0.5 * torch.erfc(-x / np.sqrt(2))

            return phi(x)

        def cdf(u):
            return Phi(-(inv_width * quant_mean - u) / (inv_width * torch.sqrt(quant_var)))  # (num_bins - 1, n, 1)
            
        probs = torch.cat([torch.zeros(1, self.n, 1).to(self.device), cdf(bins), torch.ones(1, self.n, 1).to(self.device)], dim=0)  # (num_bins + 1, n, 1)
        probs = probs[1:, ...] - probs[:-1, ...]  # (num_bins, n, 1)
        
        return probs



