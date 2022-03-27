import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from spn_code_decoding.dgcspn import DgcSpn

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
        self.hf = nn.Parameter(torch.zeros(n-1, 1).double())
        self.Lf = nn.Parameter(torch.ones(n-1, 1).double())
        self.hb = nn.Parameter(torch.zeros(n-1, 1).double())
        self.Lb = nn.Parameter(torch.ones(n-1, 1).double())

        # Node and edge potentials
        Li = torch.cat([torch.ones(1, 1).double() / s0, torch.ones(n-1, 1).double() / s], dim=0) + \
            torch.cat([a**2 * torch.ones(n-1, 1).double() / s, torch.zeros(1, 1).double()], dim=0)
        hi = torch.zeros(n, 1).double()
        self.register_buffer('Li', Li)
        self.register_buffer('hi', hi)

        self.Lij = -a / s

        # Store the device name
        self.device = device
        self.register_buffer('zero', torch.zeros(1, 1).double())
        self.register_buffer('one', torch.ones(1, 1).double())

    def forward(self, mu_in, var_in):
        """
        Min : h x w x 2
        Mout: h x w x 2
        """

        mu_in = mu_in.double()
        var_in = var_in.double()

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

        return s2s_prod_mean.float(), s2s_prod_var.float()

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

# Define a class for DataParallel that can access attributes
class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class SPNBP(nn.Module):

    def __init__(self, in_size=(1, 32, 32)):

        super(SPNBP, self).__init__()

        # Specify the device
        self.device = torch.device('cuda:0')

        # Store some parameters
        self.h = in_size[1]
        self.w = in_size[2]
        
        # Load the model
        self.model = DgcSpn(
                        in_size,
                        dequantize=False,
                        logit=None,
                        out_classes=1,
                        n_batch=64,
                        sum_channels=64,
                        depthwise=True,
                        n_pooling=0,
                        optimize_scale=True,
                        in_dropout=None,
                        sum_dropout=None,
                        quantiles_loc=None,
                        uniform_loc=None,
                        rand_state=np.random.RandomState(42),
                        leaf_distribution='gaussian',
                    ).to(self.device)

        self.model = MyDataParallel(self.model, device_ids=[0, 1])
        self.model.to(self.device)

        # Restore the checkpoint and set to evaluation mode
        model_checkpoint = torch.load('/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/spn_code_decoding/dgcspn/gauss-markov/generative/model_2022-03-24_13:20:25.pt')
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        del model_checkpoint

        # Save the base layer meand and variance
        # The SPN base distribution leaves are gaussian, so compute the new means and variances
        # of the scaled gaussian by incorporating external messages
        self.base_var = torch.exp(2 * self.model.base_layer.log_scale)  # (num_components, 1, h, w)
        self.base_mean = self.model.base_layer.mean
        self.mixture_probs = torch.ones(self.base_mean.shape[0], self.h * self.w, 1).to(self.device)

    def compute_prior_bp(self, q2s_prod_mean, q2s_prod_var):

        # Get the shape of the base means and variances
        num_components, _, h, w = self.model.base_layer.mean.shape

        # Reshape the external means and variances
        q2s_prod_mean = q2s_prod_mean.reshape(1, 1, h, w)
        q2s_prod_var = q2s_prod_var.reshape(1, 1, h, w)

        # The leaves of the SPN are Gaussian distributions.  To marginalize them out with
        # external beliefs we need to compute the expected value of the external beliefs
        # w.r.t. the leaf distributions
        z = -0.5 * torch.log(2 * np.pi * (self.base_var + q2s_prod_var)) - 0.5 * ((self.base_mean - q2s_prod_mean)**2 / (self.base_var + q2s_prod_var))
        z = z.squeeze(1).unsqueeze(0)  # Remove in_channels dimension and add batch dimension

        y = z
        for layer in self.model.layers:
            y = layer(y)

        y = self.model.root_layer(y)

        # Compute gradients to compute the mixture component probabilities. Normalize them
        # since they are discrete probabilities
        (z_grad, ) = torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y))  # (1, num_components, h, w)
        z_grad /= torch.sum(z_grad, dim=1, keepdim=True)
        z_grad = z_grad.reshape(1, num_components, h * w, 1)  # (1, num_components, n, 1)
        
        self.mixture_probs = z_grad.squeeze(0)

    def compute_quant_to_alphabet(self, Q, Q0, b):

    # % computes the Q to U messages (cf Section 5.2.3)
    # %
    # % arguments:
    # %  s2q_mean:    m*n sparse matrix; mean of the S to Q Gaussian messages
    # %  s2q_var:     m*n sparse matrix; variance of the S to Q Gaussian messages
    # %  Q:           m*n sparse matrix; the quantization matrix
    # %  Q0:          m*1 vector; the offset vector
    # %  b:           scalar; the size of the translator output
    # %
    # % returns:
    # %  u_prob:      m*(2^b) matrix; the Q to U messages (== U to T messages), 
    # %               j-th row represents p(u_a == [j-th index])

    # %% define helping variables
    # num_bins = 2^b;
    # m = size(Q,1);
    # bin_internal = -num_bins/2+1:num_bins/2-1;

    # QQ = Q.^2;
    # Qm = Q.*s2q_mean;
    # Qms = sum(Qm, 2);
    # QQv = QQ.*s2q_var;

    # %% calculate u_prob (Eq. 5.49)
    # QQvsrt = sum(QQv,2).^0.5; 
    # basec = (-Qms - Q0) ./ QQvsrt; % first 2 terms in Eq. 5.48
    # addc = bsxfun(@rdivide, bin_internal, QQvsrt); % last term in Eq. 5.48
    # cc = bsxfun(@plus, basec, addc); % value in Eq. 5.48
    # u_prob = diff([zeros(m,1), normcdf(cc), ones(m,1)], 1, 2); %Eq. 5.49

        # Get the shape of the base means and variances
        num_components, _, h, w = self.model.base_layer.mean.shape
        n = h * w

        # Transform the means and variances to shape num_components x m x n
        s2q_mean = torch.zeros(num_components, n, n).to(self.device)
        s2q_var = torch.zeros(num_components, n, n).to(self.device)
        s2q_mean[:, torch.eye(n) == 1] = self.base_mean.squeeze().reshape(num_components, n)
        s2q_var[:, torch.eye(n) == 1] = self.base_var.squeeze().reshape(num_components, n)

        num_bins = 2 ** b
        m = Q.shape[0]
        bin_internal = torch.arange(-num_bins/2 + 1, num_bins/2).reshape(1, 1, -1).to(self.device)

        QQ = Q ** 2  # m x n
        Qm = Q * s2q_mean  # num_components x m x n
        Qms = torch.sum(Qm, -1, keepdim=True)  # num_components x m x 1
        QQv = QQ * s2q_var  # num_components x m x n

        QQvsrt = torch.sqrt(torch.sum(QQv, -1, keepdim=True))  # num_components x m x 1

        basec = (-Qms - Q0) / QQvsrt  # num_components x m x 1
        addc = bin_internal / QQvsrt  # 1 x 1 x (num_bins - 1)
        cc = basec + addc  # num_components x m x (num_bins - 1)

        def Phi(x):

            phi = lambda x: 0.5 * torch.erfc(-x / np.sqrt(2))

            return phi(x)
        
        probs = torch.cat([torch.zeros(num_components, m, 1).to(self.device), Phi(cc), torch.ones(num_components, m, 1).to(self.device)], dim=-1)  # num_components x m x (num_bins + 1)
        probs = probs[..., 1:] - probs[..., :-1]  # num_components x m x num_bins

        # Multiply with the mixture probabilities.  
        probs = torch.sum(self.mixture_probs * probs, dim=0)  # m x num_bins

        return probs    

    def marginalize(self, q2s_prod_mean, q2s_prod_var):

        # Get the shape of the base means and variances
        num_components, _, h, w = self.model.base_layer.mean.shape

        # The SPN base distribution leaves are gaussian, so compute the new means and variances
        # of the scaled gaussian by incorporating external messages
        base_var = self.base_var.reshape(num_components, h * w, 1)
        base_mean = self.base_mean.reshape(num_components, h * w, 1)  # num_components x n x 1

        # Get scaling factor 
        scaling_factor = torch.exp(-0.5 * (base_mean - q2s_prod_mean)**2 / (base_var + q2s_prod_var)) / torch.sqrt(2 * np.pi * (base_var + q2s_prod_var))  # num_components x n x 1
        # Multiply scaling factor with mixture probs and then normalize
        scale_mul_mixture_probs = scaling_factor * self.mixture_probs.reshape(-1, h * w, 1)
        scale_mul_mixture_probs_norm = scale_mul_mixture_probs / torch.sum(scale_mul_mixture_probs, dim=0, keepdim=True)

        # Compute mean by multiplying by mean of each mixture and summing
        # base_mul_exp = (base_mean * q2s_prod_var + q2s_prod_mean * base_var) / (base_var + q2s_prod_var)
        base_mul_exp = base_mean
        s_mean = torch.sum(scale_mul_mixture_probs_norm * base_mul_exp, dim=0)

        return s_mean



