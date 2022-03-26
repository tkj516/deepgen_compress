import torch
import numpy as np
from torch_parallel.code_bp_torch_v3 import CodeBP
from gauss_markov.source import GridBP, SPNBP

device = torch.device('cuda:0')

def bin_to_gray(x, bits=8):
    """Convert integer to graycode.

    Args:
        x (int): Integer
        bits (int, optional): Number of bits per alphabet. Defaults to 8.

    Returns:
        str: Graycode in string format.
    """

    out = x >> 1 ^ x
    out_str = bin(out).split('b')[-1].zfill(bits)

    return out, out_str

def convert_to_graycode(s, bits=8):
    """Convert integer samples to graycode.

    Args:
        s (np.ndarray): Integer samples.
        bits (int, optional): Number of bits per alphabet. Defaults to 8.

    Returns:
        np.ndarray: Graycoded samples
    """

    if not isinstance(s, np.ndarray):
        s = s.cpu().numpy()

    out = ''.join([bin_to_gray(int(sample), bits)[1] for sample in s.flatten()])  # [::-1]
    out = np.array([float(sample) for sample in out]).reshape(-1, 1)

    return torch.FloatTensor(out)


def compute_alphabet_to_binary(u_prob, z_prob, b, binmx):

# function [ z_prob ] = compute_alphabet_to_binary( u_prob, z_prob, b, binmx )
# % computes the T to Z messages (cf. Section 4.4.2, Equation 4.36),
# % 
# % arguments:
# %  u_prob:  m*(2^b) matrix; the U to T messages, 
# %           j-th row represents p(u_a == [j-th index])
# %  z_prob:  mb*1 vector; the Z to T messages, represents p(z_i == 1)
# %  b:       scalar; the size of the translator output
# %  binmx:   1 * (2^b) * b 3D-matrix; represents choice of translator
# %           (cf Section 4.2)
# %
# % returns:
# %  z_prob:  mb*1 vector: the T to Z messages, represents p(z_i == 1)

# %% define helping variables
# m = size(z_prob,1) / b;
# ep = .0001; %small value

# %% pre-processing
# z_prob_reshaped = permute(reshape(z_prob, b, m), [2,3,1]) + ep; % m * 1 * b 3D-matrix [add .0001 to avoid div by 0]
# u_prob_pre = abs(bsxfun(@plus, z_prob_reshaped, binmx) - 1); % m * (2^b) * b 3D-matrix

# %% take product along 3rd axis (in log domain)
# u_prob_pre_log = log(u_prob_pre);
# logprodmx = sum(u_prob_pre_log, 3) + log(u_prob); % m * (2^b) 2D-matrix
# prodmx = exp(bsxfun(@minus, logprodmx, u_prob_pre_log)); % m * (2^b) * b 3D-matrix

# %% sum and normalized (Eq. 4.36)
# z_prob_out_pre = sum(bsxfun(@times, prodmx, binmx), 2) ./ sum(prodmx, 2); % m * 1 * b 3D-matrix
# z_prob = reshape(permute(z_prob_out_pre, [3,1,2]), [m*b, 1]); 

    m = z_prob.shape[0] // b
    ep = 0.0001

    z_prob_reshaped = z_prob.reshape(m, 1, b) + ep  # TODO:  Fixed from MATLAB version (don't use FORTRAN ordering)
    u_prob_pre = torch.abs(z_prob_reshaped + binmx - 1) 

    u_prob_pre_log = torch.log(u_prob_pre)
    logprodmx = torch.sum(u_prob_pre_log, dim=2) + torch.log(u_prob)
    prodmx = torch.exp(logprodmx.unsqueeze(-1) - u_prob_pre_log)

    z_prob_out_pre = torch.sum(prodmx * binmx, dim=1, keepdim=True) / torch.sum(prodmx, dim=1, keepdim=True)
    z_prob = z_prob_out_pre.reshape(m*b, 1)  # TODO:  Fixed from MATLAB version (don't use FORTRAN ordering)

    return z_prob


def compute_binary_to_alphabet(z_prob, b, binmx):


# function [ u_prob ] = compute_binary_to_alphabet( z_prob, b, binmx )
# % computss the T to U messages (cf Section 4.4.2, Equation 4.35)
# %
# % arguments:
# %  z_prob:  mb*1 vector; the Z to T messages, represents p(z_i == 1)
# %  b:       scalar; the size of the translator output
# %  binmx:   1 * (2^b) * b 3D-matrix; represents choice of translator
# %           (cf Section 4.2)
# % returns:
# %  u_prob:  m*(2^b) matrix; the T to U messages, 
# %           j-th row represents p(u_a == [j-th index])

# %% define helping variables
# m = size(z_prob,1) / b;

# %% pre-processing
# z_prob_reshaped = permute(reshape(z_prob, b, m), [2,3,1]); % m * 1 * b 3D-matrix
# u_prob_pre = abs(bsxfun(@plus, z_prob_reshaped, binmx) - 1); % m * (2^b) * b 3D-matrix

# %% take product along 3rd axis (Eq. 4.35)
# u_prob = exp(sum(log(u_prob_pre), 3));     

    m = z_prob.shape[0] // b

    z_prob_reshaped = z_prob.reshape(m, 1, b)  # TODO: Changed from MATLAB version to circumvent FORTRAN indexing
    u_prob_pre = torch.abs(z_prob_reshaped + binmx - 1) 

    u_prob = torch.exp(torch.sum(torch.log(u_prob_pre), dim=2))

    return u_prob

def compute_quant_to_alphabet(s2q_prod_mean, s2q_prod_var, Q, Q0, b):

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

    num_bins = 2 ** b
    m = Q.shape[0]
    bin_internal = torch.arange(-num_bins/2 + 1, num_bins/2).reshape(1, -1).to(device)

    q = Q[0, 0]
    
    QQ = q ** 2
    Qms = q * s2q_prod_mean  # m x 1
    QQv = QQ * s2q_prod_var  # m x 1

    QQvsrt = torch.sqrt(QQv)  # m x 1
    basec = (-Qms - Q0) / QQvsrt  # m x 1
    addc = bin_internal / QQvsrt  # 1 x (num_bins - 1)
    cc = basec + addc  # m x (num_bins - 1)

    def Phi(x):

        phi = lambda x: 0.5 * torch.erfc(-x / np.sqrt(2))

        return phi(x)
    
    probs = torch.cat([torch.zeros(m, 1).to(device), Phi(cc), torch.ones(m, 1).to(device)], dim=1)  # m x (num_bins + 1)
    probs = probs[..., 1:] - probs[..., :-1]  # m x num_bins

    return probs


def compute_quant_to_source(Q, Q0, u_prob, b):

# function [ q2s_mean, q2s_var ] = compute_quant_to_source( s2q_mean, s2q_var, Q, Q0, u_prob, b )
# % computes the Q to S messages (cf Section 5.2.1)
# %
# % arguments:
# %  s2q_mean:    m*n sparse matrix; mean of the S to Q Gaussian messages
# %  s2q_var:     m*n sparse matrix; variance of the S to Q Gaussian messages
# %  Q:           m*n sparse matrix; the quantization matrix
# %  Q0:          m*1 vector; the offset vector
# %  u_prob:      m*(2^b) matrix; the U to Q messages (== T to U messages), 
# %               j-th row represents p(u_a == [j-th index])
# %  b:           scalar; the size of the translator output
# %
# % returns:
# %  q2s_mean:    m*n sparse matrix; mean of the Q to S Gaussian messages
# %  q2s_var:     m*n sparse matrix; variance of the Q to S Gaussian messages

# num_bins = 2^b;
# bin = -num_bins/2:num_bins/2-1;

# %% define helping variables
# qpos = spones(Q); %0-1 representation of sparsity of Q
# QQ = Q.^2;
# Qm = Q.*s2q_mean;
# Qms = sum(Qm, 2);
# QQv = QQ.*s2q_var;
# Qinv = spfun(@(x) 1./x, Q);
# QQinv = Qinv.^2;
# binmean = sum(bsxfun(@times, u_prob, bin),2);
# binvar = sum(bsxfun(@times, u_prob, bin.^2),2) - binmean.^2;

# %% calculate Q to S messages (Eq. 5.28 and 5.30)
# q2s_mean = - Qinv .* (diag(sparse(Qms)) * qpos - Qm) + diag(sparse(- Q0 + binmean + .5)) * Qinv; 
# q2s_var = QQinv .* (diag(sparse(sum(QQv, 2))) * qpos - QQv) + diag(sparse(binvar)) * QQinv + convert_slab(Qinv);

    q = Q[0, 0]  # 1 / width
    
    num_bins = 2 ** b
    bin = torch.arange(-num_bins / 2, num_bins/2).reshape(1, -1).to(device)

    Qinv = 1 / q
    QQinv = Qinv ** 2
    binmean = torch.sum(u_prob * bin, dim=1, keepdim=True)  # m x 1
    binvar = torch.sum(u_prob * (bin ** 2), 1, keepdim=True) - (binmean ** 2)  # m x 1

    q2s_mean = Qinv * (-Q0 + binmean + 0.5)  # m x 1
    q2s_var = QQinv * binvar + QQinv / 12  # m x 1

    return q2s_mean, q2s_var


def marginalize(s2q_prod_mean, s2q_prod_var, q2s_prod_mean, q2s_prod_var):
   
# function [ s_mean, s_var ] = marginalize( s2s_prod_mean, s2s_prod_var, q2s_mean, q2s_var)
# % computes the marginal of the source values (cf Section 6.2.3)
# %
# % arguments:
# %  s2s_prod_mean:   n*1 vector; mean of product of intra-source messages
# %  s2s_prod_var:	n*1 vector, variance of product of intra-source messages
# %  q2s_mean:        m*n sparse matrix; mean of the Q to S Gaussian messages
# %  q2s_var:         m*n sparse matrix; variance of the Q to S Gaussian messages
# %
# % returns:
# %  s_mean:          n*1 vector: mean of the marginal (MMSE estimate)
# %  s_var:           n*1 vector: variance of the marginal

# varinv = spfun(@(x) 1./x, q2s_var);
# s_var = 1./ (1./s2s_prod_var + sum(varinv,1)'); % Eq. 6.33

# mdivv = q2s_mean.*varinv;
# s_mean = (s2s_prod_mean./s2s_prod_var + sum(mdivv, 1)') .* s_var; % Eq. 6.34   

    s_var = 1 / (1/s2q_prod_var + 1/q2s_prod_var)

    s_mean = (s2q_prod_mean/s2q_prod_var + q2s_prod_mean/q2s_prod_var) * s_var

    return s_mean, s_var


def decode(x, dope, H, Q, Q0, b, iters, convg):

    m, n = Q.shape
    kb = x.shape[0]

    num_bins = 2 ** b
    binmx = convert_to_graycode(np.arange(0, num_bins), bits=b).reshape(1, num_bins, b).to(device)

    s2q_prod_mean = torch.zeros(m, 1).to(device)
    s2q_prod_var = torch.ones(m, 1).to(device)
    z_prob = 0.5 * torch.ones(m * b, 1).to(device)
    
    s_hat_prev = torch.zeros(n, 1).to(device)

    code = CodeBP(H).to(device)
    source = GridBP(a=0.7, s=0.51, s0=1, mu0=0, n=n).to(device)

    for i in range(iters):

        u_prob = compute_quant_to_alphabet(s2q_prod_mean, s2q_prod_var, Q, Q0, b)
        if torch.any(torch.isnan(u_prob)):
            print("a")
        z_prob = compute_alphabet_to_binary(u_prob, z_prob, b, binmx)
        if torch.any(torch.isnan(z_prob)):
            print("b")
        code(torch.cat([1-dope, dope], dim=1), x, torch.cat([1-z_prob, z_prob], dim=1))  # Appropriate concatentation and slicing
        z_prob = code.M_out[:, 1:]
        if torch.any(torch.isnan(z_prob)):
            print("c")
        u_prob = compute_binary_to_alphabet(z_prob, b, binmx)
        if torch.any(torch.isnan(u_prob)):
            print("d")
        q2s_prod_mean, q2s_prod_var = compute_quant_to_source(Q, Q0, u_prob, b)
        if torch.any(torch.isnan(q2s_prod_mean + q2s_prod_var)):
            print("e")
        s2q_prod_mean, s2q_prod_var = source(q2s_prod_mean, q2s_prod_var)
        if torch.any(torch.isnan(s2q_prod_mean + s2q_prod_var)):
            print("g")

        # Check convergence
        s_hat, _ = marginalize(s2q_prod_mean, s2q_prod_var, q2s_prod_mean, q2s_prod_var)
        s_diff = torch.max(torch.abs(s_hat_prev - s_hat))
        print(f"{i}: {s_diff}")

        if s_diff < convg:
            break
        s_hat_prev = s_hat

    return s_hat