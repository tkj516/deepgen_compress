import torch
import numpy as np
from .src import *
from torch_parallel.code_bp_torch_v2 import CodeBP
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


def decode(x, dope, H, Q, Q0, b, iters, convg):

    m, n = Q.shape
    kb = x.shape[0]

    num_bins = 2 ** b
    binmx = convert_to_graycode(np.arange(0, num_bins), bits=b).reshape(1, num_bins, b).to(device)

    s2q_mean = torch.zeros(m, n).to(device)
    s2q_var = torch.ones(Q.shape).to(device)
    z_prob = 0.5 * torch.ones(m * b, 1).to(device)
    
    s_hat_prev = torch.zeros(n, 1).to(device)

    code = CodeBP(H).to(device)
    source = GridBP(a=0.7, s=0.51, s0=1, mu0=0, n=n).to(device)

    for i in range(iters):

        u_prob = compute_quant_to_alphabet(s2q_mean, s2q_var, Q, Q0, b)
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
        q2s_mean, q2s_var = compute_quant_to_source(s2q_mean, s2q_var, Q, Q0, u_prob, b)
        if torch.any(torch.isnan(q2s_mean + q2s_var)):
            print("e")
        q2s_prod_mean, q2s_prod_var = compute_source_to_prior(q2s_mean, q2s_var)
        if torch.any(torch.isnan(q2s_prod_mean + q2s_prod_var)):
            print("f")
        s2s_prod_mean, s2s_prod_var = source(q2s_prod_mean, q2s_prod_var)
        if torch.any(torch.isnan(s2s_prod_mean + s2s_prod_var)):
            print("g")
        s2q_mean, s2q_var = compute_source_to_quant(s2s_prod_mean, s2s_prod_var, q2s_mean, q2s_var, Q)
        if torch.any(torch.isnan(s2q_mean + s2q_var)):
            print("h")

        # Check convergence
        s_hat, _ = marginalize(s2s_prod_mean, s2s_prod_var, q2s_mean, q2s_var)
        s_diff = torch.max(torch.abs(s_hat_prev - s_hat))
        print(f"{i}: {s_diff}")

        if s_diff < convg:
            break
        s_hat_prev = s_hat

    return s_hat


def decode_SPN(x, dope, H, Q, Q0, b, iters, convg, checkpoint=None):

    m, n = Q.shape
    kb = x.shape[0]

    num_bins = 2 ** b
    binmx = convert_to_graycode(np.arange(0, num_bins), bits=b).reshape(1, num_bins, b).to(device)

    s2q_mean = torch.zeros(m, n).to(device)
    s2q_var = torch.ones(Q.shape).to(device)
    z_prob = 0.5 * torch.ones(m * b, 1).to(device)
    
    s_hat_prev = torch.zeros(n, 1).to(device)

    code = CodeBP(H).to(device)
    source = SPNBP().to(device)

    for i in range(iters):

        u_prob = source.compute_quant_to_alphabet(Q, Q0, b)
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
        q2s_mean, q2s_var = compute_quant_to_source(s2q_mean, s2q_var, Q, Q0, u_prob, b)
        if torch.any(torch.isnan(q2s_mean + q2s_var)):
            print("e")
        q2s_prod_mean, q2s_prod_var = compute_source_to_prior(q2s_mean, q2s_var)
        if torch.any(torch.isnan(q2s_prod_mean + q2s_prod_var)):
            print("f")
        source.compute_prior_bp(q2s_prod_mean, q2s_prod_var)

        # Check convergence
        s_hat = source.marginalize(q2s_prod_mean, q2s_prod_var)
        s_diff = torch.max(torch.abs(s_hat_prev - s_hat))
        print(f"{i}: {s_diff}")

        if s_diff < convg:
            break
        s_hat_prev = s_hat

    return s_hat


    