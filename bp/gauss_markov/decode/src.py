import torch
import numpy as np

device = torch.device('cuda:0')

def spinv(x):

    xinv = 1 / x
    xinv[x == 0] = 0

    return xinv

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

    z_prob_reshaped = z_prob.reshape(b, m, 1).permute(1, 2, 0) + ep
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

def compute_quant_to_alphabet(s2q_mean, s2q_var, Q, Q0, b):

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

    device = s2q_mean.device

    num_bins = 2 ** b
    m = Q.shape[0]
    bin_internal = torch.arange(-num_bins/2 + 1, num_bins/2).reshape(1, -1).to(device)

    QQ = Q ** 2  # m x n
    Qm = Q * s2q_mean  # m x n
    Qms = torch.sum(Qm, 1, keepdim=True)  # m x 1
    QQv = QQ * s2q_var  # m x n

    QQvsrt = torch.sqrt(torch.sum(QQv, 1, keepdim=True))  # m x 1
    basec = (-Qms - Q0) / QQvsrt  # m x 1
    addc = bin_internal / QQvsrt  # 1 x (num_bins - 1)
    cc = basec + addc  # m x (num_bins - 1)

    def Phi(x):

        phi = lambda x: 0.5 * torch.erfc(-x / np.sqrt(2))

        return phi(x)
    
    probs = torch.cat([torch.zeros(m, 1).to(device), Phi(cc), torch.ones(m, 1).to(device)], dim=1)  # m x (num_bins + 1)
    probs = probs[:, 1:] - probs[..., :-1]  # m x num_bins

    return probs


def compute_quant_to_source(s2q_mean, s2q_var, Q, Q0, u_prob, b):

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

    num_bins = 2 ** b
    bin = torch.arange(-num_bins / 2, num_bins/2).reshape(1, -1).to(device)

    qpos = 1 - (Q == 0).float()
    QQ = Q ** 2
    Qm = Q * s2q_mean
    Qms = torch.sum(Qm, 1, keepdim=True)  # m x 1
    QQv = QQ * s2q_var
    Qinv = spinv(Q)
    QQinv = Qinv ** 2
    binmean = torch.sum(u_prob * bin, dim=1)  # 1 x num_bins
    binvar = torch.sum(u_prob * (bin ** 2), 1) - (binmean ** 2)  # 1 x num_bins

    q2s_mean = -Qinv * (Qms * qpos - Qm) + Qinv * (-Q0 + binmean + 0.5)
    q2s_var = QQinv * (torch.sum(QQv, dim=1, keepdim=True) * qpos - QQv) + QQinv * binvar + QQinv / 12

    return q2s_mean, q2s_var


def compute_source_to_prior(q2s_mean, q2s_var):

# function [ q2s_prod_mean, q2s_prod_var ] = compute_source_to_prior( q2s_mean, q2s_var )
# % takes the product of the Q to S messages (cf Section 6.2.2)
# %
# % arguments:
# %  q2s_mean:        m*n sparse matrix, mean of the Q to S Gaussian messages
# %  q2s_var:         m*n sparse matrix, variance of the Q to S Gaussian messages
# %
# % returns:
# %  q2s_prod_mean:   n*1 vector, mean of the product of the Q to S messages
# %  q2s_prod_var:	n*1 vector, variance of the product of the Q to S messages

# varinv = spfun(@(x) 1./x, q2s_var);
# q2s_prod_var = full(1./ sum(varinv,1)');

# mdivv = q2s_mean.*varinv;
# q2s_prod_mean = full(sum(mdivv, 1)') .* q2s_prod_var;

    varinv = spinv(q2s_var)
    q2s_prod_var = 1 / torch.sum(varinv, dim=0).reshape(-1, 1)

    mdivv = q2s_mean * varinv
    q2s_prod_mean = torch.sum(mdivv, 0).reshape(-1, 1) * q2s_prod_var

    return q2s_prod_mean, q2s_prod_var

def compute_source_to_quant(s2s_prod_mean, s2s_prod_var, q2s_mean, q2s_var, Q):

# % computes the S to Q messages (cf Section 5.2.2)
# %
# % arguments:
# %  s2s_prod_mean:   n*1 vector; mean of product of intra-source messages
# %  s2s_prod_var:	n*1 vector; variance of product of intra-source messages
# %  q2s_mean:        m*n sparse matrix; mean of Q to S Gaussian messages
# %  q2s_var:         m*n sparse matrix; variance of Q to S Gaussian messages
# %  Q:               m*n sparse matrix; the quantization matrix
# %
# % reutrns:
# %  s2q_mean:        m*n sparse matrix, mean of the S to Q Gaussian messages
# %  s2q_var:         m*n sparse matrix, variance of the S to Q Gaussian messages

# qpos = spones(Q); %0-1 representation of sparsity of Q

# varinv = spfun(@(x) 1./x, q2s_var);
# s2q_var = spfun(@(x) 1./x, bsxfun(@times, qpos, 1./s2s_prod_var' + sum(varinv,1)) - varinv);

# mdivv = q2s_mean.*varinv;
# s2q_mean = (bsxfun(@times, qpos, (s2s_prod_mean./s2s_prod_var)' + sum(mdivv, 1)) - mdivv) .* s2q_var;

    qpos = 1 - (Q == 0).float()

    varinv = spinv(q2s_var)
    s2q_var = spinv(qpos * (1/s2s_prod_var.reshape(1, -1) + torch.sum(varinv, 0)) - varinv)

    mdivv = q2s_mean * varinv
    s2q_mean = qpos * ((s2s_prod_mean/s2s_prod_var).reshape(1, -1) + torch.sum(mdivv, 0) - mdivv) * s2q_var

    return s2q_mean, s2q_var


def marginalize(s2s_prod_mean, s2s_prod_var, q2s_mean, q2s_var):
   
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

    varinv = spinv(q2s_var)
    s_var = 1 / (1 / s2s_prod_var + torch.sum(varinv, 0).reshape(-1, 1))

    mdivv = q2s_mean * varinv
    s_mean = (s2s_prod_mean/s2s_prod_var + torch.sum(mdivv, 0).reshape(-1, 1)) * s_var

    return s_mean, s_var