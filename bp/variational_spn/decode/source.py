import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from variational_spn.models.factorized_prior import VAE
from spn_code_decoding.dgcspn import DgcSpn

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

    def __init__(self, 
                 args,
                 mean,
                 log_scale, 
                 in_size=(1, 32, 32), 
                 checkpoint='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/spn_code_decoding/dgcspn/gauss-markov/generative/model_2022-03-24_13:20:25.pt'):

        super(SPNBP, self).__init__()

        # Specify the device
        self.device = torch.device('cuda:0')

        # Store some parameters
        self.h = in_size[1]
        self.w = in_size[2]
        
        # Load the model
        self.model = VAE(
            network_channels=args.network_channels,
            compression_channels=args.compression_channels,
            num_base_distributions=args.n_batches,
            in_size=in_size,
            sum_channels=args.sum_channels,
        ).to(self.device)

        # TODO: Extract SPN and discard the rest

        self.model = MyDataParallel(self.model, device_ids=[0, 1])
        self.model.to(self.device)

        # Restore the checkpoint and set to evaluation mode
        model_checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        del model_checkpoint

        # Save the base layer meand and variance
        # The SPN base distribution leaves are gaussian, so compute the new means and variances
        # of the scaled gaussian by incorporating external messages
        self.base_var = torch.exp(2 * log_scale).double()  # (num_components, 1, h, w)
        self.base_mean = mean.double()
        self.mixture_probs = torch.ones(self.base_mean.shape[0], self.h * self.w, 1).double().to(self.device)

    def compute_prior_bp(self, q2s_prod_mean, q2s_prod_var):

        q2s_prod_mean = q2s_prod_mean.double()
        q2s_prod_var = q2s_prod_var.double()
        
        # Get the shape of the base means and variances
        num_components, _, h, w = self.model.base_layer.mean.shape

        # Reshape the external means and variances
        q2s_prod_mean = q2s_prod_mean.reshape(1, 1, h, w)
        q2s_prod_var = q2s_prod_var.reshape(1, 1, h, w)

        # The leaves of the SPN are Gaussian distributions.  To marginalize them out with
        # external beliefs we need to compute the expected value of the external beliefs
        # w.r.t. the leaf distributions
        z = -0.5 * torch.log(2 * np.pi * (self.base_var + q2s_prod_var)) - 0.5 * ((self.base_mean - q2s_prod_mean)**2 / (self.base_var + q2s_prod_var))
        z = z.squeeze(1).unsqueeze(0).float()  # Remove in_channels dimension and add batch dimension

        y = z
        for layer in self.model.layers:
            y = layer(y)

        y = self.model.root_layer(y)

        # Compute gradients to compute the mixture component probabilities. Normalize them
        # since they are discrete probabilities
        (z_grad, ) = torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y))  # (1, num_components, h, w)
        # z_grad *= torch.exp(z)
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

        Q = Q.double()
        Q0 = Q0.double()

        # Get the shape of the base means and variances
        num_components, _, h, w = self.model.base_layer.mean.shape
        n = h * w

        # Transform the means and variances to shape num_components x m x n
        s2q_mean = torch.zeros(num_components, n, n).double().to(self.device)
        s2q_var = torch.zeros(num_components, n, n).double().to(self.device)
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

        return probs.float()    

    def marginalize(self, q2s_prod_mean, q2s_prod_var):

        q2s_prod_mean = q2s_prod_mean.double()
        q2s_prod_var = q2s_prod_var.double()

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
        base_mul_exp = (base_mean * q2s_prod_var + q2s_prod_mean * base_var) / (base_var + q2s_prod_var)
        # base_mul_exp = base_mean
        s_mean = torch.sum(scale_mul_mixture_probs_norm * base_mul_exp, dim=0)

        return s_mean.float()



