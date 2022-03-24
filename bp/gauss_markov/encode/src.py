import torch
import numpy as np

device = torch.device('cuda:0')

def generate_markov(mu_0=0, 
                    s_0=1, 
                    s=0.51, 
                    a=0.7, 
                    n=1024):

    samp = np.zeros(n)
    samp[0] = np.random.normal(mu_0, np.sqrt(s_0))
    for i in range(1, n):
        samp[i] = a * samp[i-1] + np.random.normal(0, np.sqrt(s))

    return torch.FloatTensor(samp).reshape(-1, 1).to(device)

def generate_q(n, m, w):
    # function [ Q, Q0 ] = generate_q( n, m, q_option )
    # % generate the quantization matrix
    # %
    # % arguments:
    # %  n:           scalar; width of the quantization matrix
    # %  m:           scalar; height of the quantization matrix
    # %  q_option:    struct; contains information about generating Q
    # %
    # % returns:
    # %  Q:           m*n sparse matrix; the quantization matrix
    # %  Q0:          m*1 vector; the offset vector

    Q = torch.eye(n) / w
    Q0 = torch.zeros(m, 1)

    return Q.to(device), Q0.to(device)

def quantize_slice(s, Q, Q0, b):
    # function [ u ] = quantize_slice( s, Q, Q0, b )
    # % quantizes the source
    # %
    # % arguments:
    # %  s:   n*1 vector; sample to be quantized
    # %  Q:   m*n sparse matrix; the quantization matrix
    # %  Q0:  m*1 vector; the offset vector
    # %  b:   scalar; the size of the translator output
    # %
    # % returns:
    # %  u:   m*1 vector; the quantized sequence

    # num_bins = 2^b;

    # u = min(max(floor(Q*s + Q0),-num_bins/2),num_bins/2-1);

    num_bins = 2 ** b
    u = torch.clamp(torch.floor(torch.matmul(Q, s) + Q0), min=-num_bins/2, max=num_bins/2 - 1)

    return u      

# All unit tests pass
def bitget(x, i):

    if not isinstance(x, int):
        x = int(x)

    return x >> i & 1

# All unit tests pass
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

    return torch.FloatTensor(out).to(device)

def translate(u, b):

    num_bins = 2 ** b
    u_adj = u + num_bins//2
    z = convert_to_graycode(u_adj, bits=b)

    return z

def hash_ldpc(z, H):

    return torch.matmul(H, z) % 2
