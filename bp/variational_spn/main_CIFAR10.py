import sys
sys.path.append('..')

import os
import time
import torch
import torchvision
from PIL import Image
from variational_spn.encode.src import *
from ldpc_generate import pyldpc_generate
import numpy as np
from variational_spn.decode.decode import decode_SPN
from scipy.io import loadmat
from gauss_markov.utils import GaussMarkovDataset
from spnflow.torch.transforms import Reshape
from torchvision.datasets import CIFAR10

in_size = (1, 32, 32)
transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                lambda x: torch.tensor(np.array(x)),
                Reshape(in_size),
                lambda x: x.float(),
                lambda x: x / 256,
                lambda x: 2 * x - 1,
            ])
dataset = CIFAR10('../../../CIFAR10', train=False, transform=transform)

def generate_sample():

    idx = np.random.randint(0, len(dataset))
    s, _ = dataset[idx]

    return s.reshape(-1, 1).to(device)

device = torch.device('cuda:0')

# Parameters
n = 1024  # 20
m = n

b = 5
w = 0.1

total_rate = 4.6  # 3.7

rand_dope = 0
rand_dope_rate = 0.34
lat_dope_bits = np.array([b-1]).reshape(-1, 1)

max_iter = 100
convg_epsilon = 0.01

# Generate quantizer matrix
Q, Q0 = generate_q(n, m, w)

# Generate LDPC
mb = m * b
if rand_dope:
    num_dope_bits = int(np.ceil(rand_dope_rate * mb))
    dope_indices = np.sort(np.random.choice(mb, num_dope_bits, replace=False)).reshape(-1, 1)
else:
    dope_indices = (np.arange(0, mb, b).reshape(1, -1) + lat_dope_bits).reshape(1, -1)
    num_dope_bits = dope_indices.shape[1]
kb = total_rate * n - num_dope_bits

H = pyldpc_generate.generate(kb, mb, 3.0, 2, 123)
H = torch.FloatTensor(np.array(H.todense())).to(device)

# Generate source sequence
s = generate_sample()
u = quantize_slice(s, Q, Q0, b)
z = translate(u, b)
x = hash_ldpc(z, H)

# Doping
dope = 0.5 * np.ones((mb, 1))
dope[dope_indices] = (z.cpu().numpy()[dope_indices] == 1)
dope = torch.FloatTensor(dope).to(device)

# Decode
s_hat = decode_SPN(x, dope, H, Q, Q0, b, max_iter, convg_epsilon, checkpoint="/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/spn_code_decoding/dgcspn/cifar10/generative/model_2022-03-29_23:57:39.pt")
s_hat = s_hat.detach().cpu().numpy()

# Print results
s_mse = np.sum((s.cpu().numpy() - s_hat)**2) / s.shape[0]
sqnr = -10 * np.log10(s_mse)

u_hat = quantize_slice(torch.FloatTensor(s_hat).to(device), Q, Q0, b)
z_hat = translate(u_hat, b)
x_hat = hash_ldpc(z_hat, H)

z_err = np.sum(1 - ((z - z_hat) == 0).float().cpu().numpy()) / mb

print(f"b ={b}, w = {w}, d = {num_dope_bits/mb}, k = {kb/mb}, R = {(num_dope_bits+kb)/n}, SQNR = {sqnr}, z_err = {z_err}")

if z_err == 0:
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(f'cifar_results_images_gray_lossy/{timestamp}')

    orig = (s.reshape(32, 32).cpu().numpy().repeat(16, axis=-1).repeat(16, axis=-2) + 1) / 2
    recon = (s_hat.reshape(32, 32).repeat(16, axis=-1).repeat(16, axis=-2) + 1)/2
    im = Image.fromarray((255 * orig).astype('uint8')).convert('RGB')
    im.save(f'cifar_results_images_gray_lossy/{timestamp}/orig.png')
    im = Image.fromarray((255 * recon).astype('uint8')).convert('RGB')
    im.save(f'cifar_results_images_gray_lossy/{timestamp}/recon.png')