import torch
from _typeshed import OpenTextMode
import numpy as np
from scipy.linalg.decomp import eig
from scipy.stats import norm
from scipy.linalg import toeplitz
from numpy.linalg import eig
from scipy.optimize import fsolve
from numpy.matlib import repmat

#############################################################
# Bit and binary functions
#############################################################

def bitget(x, i):

    return x >> i & 1

def bin_to_gray(x, bits=8):

    out = x >> 1 ^ x
    out = bin(out).split('b')[-1].zfill(bits)

    return out

def gray_to_bin(x):

    mask = x >> 1
    out = x
    while not mask == 0:
        out = out ^ mask
        mask = mask >> 1

    return out

def convert_to_graycode(s, bits=8):
    
    in_size = s.shape
    out = ''.join([bin_to_gray(sample) for sample in s.flatten()])
    out = np.array([float(sample) for sample in out]).reshape(in_size)

    return out

#############################################################
# Markov chain functions
#############################################################

def entropy(p):
    if len(p) == 1:
        if p[0] in [0, 1]:
            return 0
        else:
            return np.array([p, 1-p])

    try:
        if np.abs(np.sum(p) - 1) <= 1e-15:
            temp = p * np.log2(p)
            temp[np.isnan(p)] = 0.0
            h = -np.sum(temp)
            return h
    except Exception as e:
        print(e)

def stochastic_matrix_entropy(
    bleed,
    M,
):
    """
    Finds the entropy of a transition matrix epot
    epot(i, j) = P(j->i) or P(i|j)
    epot * state_dist = new_state_dist
    """

    if bleed <= 0:
        return 0

    # Start with a transition matrix
    prow = (norm.cdf(np.arange(0.5, M - 1 + 0.5 + 1), loc=0, scale=bleed)
                - norm.cdf(np.arange(-0.5, M - 1 - 0.5 + 1), loc=0, scale=bleed))
    epot = toeplitz(prow)
    # Normalize the transition matrix
    epot /= np.sum(epot, axis=0, keepdims=True)

    # Find stationary distribution
    sta, d = eig(epot)
    ind = np.argmax(d)
    sta = sta[:, ind].reshape(-1, 1) / np.sum(sta[:, ind])

    # Compute entropy
    hcond = [entropy(epot[:, i]) for i in range(M)]
    h = hcond @ sta / np.log2(M)

    return h

def generate_transition_matrix(
    M,
    hf=0.01,
):

    bleed_src = fsolve(lambda x: stochastic_matrix_entropy(x, M) - hf, M / 2)
    # Start with a transition matrix
    prow = (norm.cdf(np.arange(0.5, M - 1 + 0.5 + 1), loc=0, scale=bleed_src)
                - norm.cdf(np.arange(-0.5, M - 1 - 0.5 + 1), loc=0, scale=bleed_src))
    epot = toeplitz(prow)
    # Normalize the transition matrix
    epot /= np.sum(epot, axis=0, keepdims=True)

    # Find stationary distribution
    sta, d = eig(epot)
    ind = np.argmax(d)
    sta = sta[:, ind].reshape(-1, 1) / np.sum(sta[:, ind])

    return bleed_src, epot, sta

def generate_sample(
    epot, 
    sta,
    N, 
    bits,
):
    """Generate a sample from the Markov chain.

    Args:
        epot (np.ndarray): Edge potentials
        sta (np.ndarray): Stationary distribution
        N (int): Length of Markov chain
        bits (int): Number of bits per alphabet

    Returns:
        (np.ndarray, np.ndarray): (Integer sample, graycoded sample)
    """
    # Flatten matrix for ease of processing
    sta = sta.flatten()

    s = []
    find = lambda x: np.argwhere(x)[0]
    s.append(
        find(
            np.random.rand() < np.cumsum(sta)
        ) - 1
    )
    for _ in range(1, N):
        s.append(
            find(
                np.random.rand() < np.cumsum(epot[:, s[-1]])
                ) - 1
        )
    s = s.reshape(-1, 1)

    # Get a graycode representation
    graycoded_s = convert_to_graycode(s, bits)

    return s, graycoded_s

def msg_graycode_to_int(
    M_in, 
    height,
    width,
    bits
):
    """Convert a graycode message to a multi-alphabet message.

    Args:
        M_in (torch.Tensor): Input message over Markov chain nodes (N * bits, 2)
        height (int): Height of chain.      
        width (int): Width of chain.
        bits (int): Number of bits per alphabet.

    Returns:
        torch.Tensor: Converted output message (1 x N x M)
    """

    device = M_in.device
    M_in = M_in.detach().cpu().numpy()
    assert M_in.shape[0] // (height * width) == bits

    M = 2 ** bits

    temp = M_in.reshape(bits, height, width, 2)
    M_out = np.ones((height, width, M))

    for i in range(bits):
        index_mask = np.array([bitget(a, i) for a in range(M)])
        M_out[..., index_mask == 0] *= repmat(temp[i, ..., 0], 1, M // 2)
        M_out[..., index_mask == 1] *= repmat(temp[i, ..., 1], 1, M // 2)

    M_out[:, :, [gray_to_bin(i) for i in range(M)]] = M_out

    if width == 1:
        M_out = M_out.squeeze()

    return torch.Tensor(M_out).to(device)

def msg_int_to_graycode(
    M_in,
):
    """Convert a multi-alphabet message to a graycode message.

    Args:
        M_in (torch.Tensor): Input message over Markov chain nodes (1 x N x M)

    Returns:
        torch.Tensor: Converted output message (N * bits x 2)
    """
    M_in = M_in.reshape(-1, M_in.shape[-1])
    device = M_in.device
    M_in = M_in.detach().cpu().numpy()

    M = M_in.shape[-1]
    bits = np.log2(M)
    N = M_in.shape[0]
    M_out = np.zeros((N * bits, 2))

    graycodes = convert_to_graycode(np.arange(0, M))

    for i in range(int(bits)):
        index_mask = np.array([bitget(a, i) for a in graycodes])
        temp1 = np.sum(M_in[:, index_mask == 0], axis=1)
        temp2 = np.sum(M_in[:, index_mask == 1], axis=1)
        M_out[i::bits, 0] = temp1
        M_out[i::bits, 1] = temp2

    return torch.Tensor(M_out).to(device)







    