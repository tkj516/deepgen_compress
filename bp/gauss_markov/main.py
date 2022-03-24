
import sys
sys.path.append('..')

from encode.src import *
from ldpc_generate import pyldpc_generate
import numpy as np
from decode.decode import decode
from scipy.io import loadmat

device = torch.device('cuda:0')

# Parameters
n = 20
m = n

b = 4
w = 0.6

total_rate = 3.7

rand_dope = 0
rand_dope_rate = 0.34
lat_dope_bits = np.array([0]).reshape(-1, 1)

max_iter = 150
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

# H = pyldpc_generate.generate(kb, mb, 3.0, 2, 123)
print(kb, mb)
# H = torch.FloatTensor(np.array(H.todense())).to(device)
H = torch.FloatTensor(loadmat('/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/gauss_markov/H.mat')['H_f']).to(device)

# Generate source sequence
# s = generate_markov(
#     mu_0=0,
#     s_0=1,
#     s=0.51,
#     a=0.7,
#     n=n,
# )
s = torch.FloatTensor(loadmat('/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/gauss_markov/s.mat')['s']).to(device)
u = quantize_slice(s, Q, Q0, b)
z = translate(u, b)
x = hash_ldpc(z, H)

# Doping
dope = 0.5 * np.ones((mb, 1))
dope[dope_indices] = (z.cpu().numpy()[dope_indices] == 1)
dope = torch.FloatTensor(dope).to(device)

# Decode
s_hat = decode(x, dope, H, Q, Q0, b, max_iter, convg_epsilon)
s_hat = s_hat.detach().cpu().numpy()

# Print results
s_mse = np.sum((s.cpu().numpy() - s_hat)**2) / s.shape[0]
sqnr = -10 * np.log10(s_mse)

u_hat = quantize_slice(s, Q, Q0, b)
z_hat = translate(u_hat, b)
x_hat = hash_ldpc(z_hat, H)

z_err = np.sum(1 - ((z - z_hat) == 0).float().cpu().numpy()) / mb

print(f"b ={b}, w = {w}, d = {num_dope_bits/mb}, k = {kb/mb}, R = {(num_dope_bits+kb)/n}, SQNR = {sqnr}, z_err = {z_err}")