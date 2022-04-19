import sys
sys.path.append('..')

import os
import time
import json
import torchvision
from PIL import Image
from encode.src import *
from ldpc_generate import pyldpc_generate
import numpy as np
from decode.decode import decode_SPN
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

max_iter = 60
convg_epsilon = 0.01

# Map to cache LDPC matrices for reuse
LDPC = {}

# List to store all the results
results = []

for i in range(10):

    # Choose sample
    s, _ = dataset[i]
    s = s.reshape(-1, 1).to(device)

    for b in [3, 5, 7, 8]:
        for w in [0.01, 0.1, 0.2, 0.4, 0.6]:

            print(f"{(i, b, w)}")

            # Set the doping bits
            lat_dope_bits = np.array([b-1]).reshape(-1, 1)

            # Generate quantizer matrix
            Q, Q0 = generate_q(n, m, w)

            min_rate = best_rate = 0
            max_rate = b
            mse_opt = 0
            sqnr_opt = 0

            while min_rate < max_rate:

                # Get the total rate
                total_rate = round((min_rate + max_rate) / 2, 2)

                print(f" {(total_rate, min_rate, max_rate)}")

                # Setup doping
                mb = m * b
                dope_indices = (np.arange(0, mb, b).reshape(1, -1) + lat_dope_bits).reshape(1, -1)
                num_dope_bits = dope_indices.shape[1]
                kb = total_rate * n - num_dope_bits

                # Load the appropriate size LDPC matrix
                if (kb, mb) in LDPC.keys():
                    H = LDPC[(kb, mb)]
                    H = torch.FloatTensor(np.array(H.todense())).to(device)
                else:
                    H = pyldpc_generate.generate(kb, mb, 3.0, 2, 123)
                    LDPC[(kb, mb)] = H
                    H = torch.FloatTensor(np.array(H.todense())).to(device)

                # Generate source sequence
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
                z_err = np.sum(1 - ((z - z_hat) == 0).float().cpu().numpy()) / mb

                if z_err == 0:
                    max_rate = total_rate - 0.1
                    best_rate = total_rate
                    mse_opt = s_mse
                    sqnr_opt = sqnr
                else:
                    min_rate = total_rate + 0.1

            results.append((i, b, w, mse_opt, sqnr_opt, best_rate))

with open('temp.json', 'w') as file:
    json.dump(results, file, indent=4)

filepath = os.path.join('demo_gray_lossy', 'cifar-10', 'results.json')
os.makedirs(os.path.join('demo_gray_lossy', 'cifar-10'), exist_ok=True)
with open(filepath, 'w') as file:
    json.dump(results, file, indent=4)



