import sys
sys.path.append('..')

import numpy as np
from scipy import integrate, interpolate
from torch_parallel.grid_gibbs import GibbsSampler
import matplotlib.pyplot as plt
from ldpc_generate import pyldpc_generate
from tqdm import tqdm

def ising_entropy(stays=None):

    logZN_left = []
    logZN_right = []
    logZN = []

    if stays is None:
        stays = np.arange(0.5, 0.98, 0.02)
    thetas = np.arctanh(2 * (stays - 0.5))

    thetas_left = thetas - 0.001
    thetas_right = thetas + 0.001

    for i in range(len(thetas)):
        t = np.tanh(thetas_left[i])
        logZN_left.append(logZN_inf_lattice(t))
        t = np.tanh(thetas_right[i])
        logZN_right.append(logZN_inf_lattice(t))
        t = np.tanh(thetas[i])
        logZN.append(logZN_inf_lattice(t))

    # Convert to numpy arrays
    logZN_left = np.array(logZN_left)
    logZN_right = np.array(logZN_right)
    logZN = np.array(logZN)

    hs = -thetas * (logZN_right - logZN_left) / (thetas_right - thetas_left) + logZN
    hs /= np.log(2)
    hs[np.isnan(hs)] = 0

    return hs, stays

def logZN_inf_lattice(t):

    f = lambda qy, qx: np.log((1+t**2)**2 - 2*t*(1-t**2)*(np.cos(qx)+np.cos(qy)))

    integral, _ = integrate.dblquad(f, -np.pi, np.pi, -np.pi, np.pi)

    logZN = np.log(2/(1-t**2)) + (0.5/(2*np.pi)**2) * integral

    return logZN

def finite_entropy(stays, N=100):

    SE, DE, Ns, y = count_edges(stays, N=N)

    thetas = np.arctanh(2 * (np.array(stays) - 0.5))
    newthetas = np.arange(thetas[0], thetas[-1], 0.02)

    f = interpolate.interp1d(thetas, y)
    dlogZN_dtheta = f(newthetas)
    
    logZN = []
    for theta in newthetas:
        integral, _ = integrate.quad(f, 0, theta)
        logZN.append(integral)

    HN = (np.array(logZN) - newthetas * dlogZN_dtheta) / np.log(2) + 1
    newstays = np.tanh(newthetas) / 2 + 0.5

    return HN, newstays

def count_edges(stays, N=100):

    SE = {}
    DE = {}
    Ns = {}
    y = []

    for stay in stays:
        gibbs_sampler = GibbsSampler(28, 28, 0.5, stay)
        Ns[stay] = N
        se = 0
        de = 0

        for i in range(N):
            gibbs_sampler.sampler(1000)
            sample = gibbs_sampler.samp
            hedges = sample[:, :-1] - sample[:, 1:]
            vedges = sample[:-1, :] - sample[1:, :]
            gibbs_sampler.reset()

            se = se + (hedges == 0).astype(float).sum() + (vedges == 0).astype(float).sum()
        de = 2*28*27*N - se

        se /= N # 2*28*27
        de /= N # 2*28*27

        SE[stay] = se
        DE[stay] = de

        y.append((se - de)/28**2)

    return SE, DE, Ns, y

def compute_sep_threshold(sep_prot, doperate=0.04):

    # Get the base decoding rate by subtracting doping rate
    base_rates = np.array(sep_prot) - doperate

    sep_threshold = []

    x = np.arange(0.01, 1, 0.01)

    for i in tqdm(range(len(base_rates))):

        r = base_rates[i]

        # Create LDPC matrix for this rate and compute
        # mean row sum (rho_bar)
        lambda_bar = 3
        H = pyldpc_generate.generate(int(r*784), 784, 3.0, 2, 123)
        k, n = H.shape[0], H.shape[1]
        rho_bar = H.sum(-1).mean()

        # Implement degree polynomial as shown in Ex 5.2 in YZ Thesis
        # TODO: Add derivation to notes
        frac = rho_bar - int(rho_bar)
        P_prime = lambda x: int(rho_bar)*(k - int(k*frac))*x**(int(rho_bar) - 1) + (int(rho_bar)+1)*int(k*frac)*x**int(rho_bar)
        f = lambda eps, x: eps*(1 - (P_prime(1-x)/P_prime(1.0)))**2

        maximum = 1.0
        eps = r
        
        while maximum >= 0:

            eps -= 0.005
            maximum = np.max(f(eps, x) - x)

        # Add doping back when appending
        sep_threshold.append(eps + doperate)

    return sep_threshold

if __name__ == "__main__":

    hs, stays = ising_entropy()

    hf, newstays = finite_entropy(np.arange(0.5, 1.0, 0.04), N=50)

    q = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    sep_prot_pgm = [1.19649, 1.18292, 1.14193, 1.03935, 0.71966, 0.33407, 0.20538, 0.16562, 0.1470, 0.14261]
    sep_prot_spn = [1.19490, 1.18673, 1.14894, 1.06529, 0.74532, 0.34625, 0.22401, 0.17977, 0.16053, 0.15176]

    sep_threshold_pgm = compute_sep_threshold(sep_prot_pgm)
    sep_threshold_spn = compute_sep_threshold(sep_prot_spn)

    fig, ax = plt.subplots()
    ax.plot(stays, hs, 'b-', label='$H_\infty$')
    ax.plot(newstays, hf, 'r-', label='$H_{h, w}$')
    ax.plot(q, sep_prot_pgm, '+-', label='sep-prot PGM')
    ax.plot(q, sep_prot_spn, 'x-', label='sep-prot SPN')
    ax.plot(q, sep_threshold_pgm, 'o-', label='sep-thresh PGM')
    ax.plot(q, sep_threshold_spn, '^-', label='sep-thresh SPN')
    ax.set_ylabel('output bits per input bits')
    ax.set_xlabel('q')
    ax.legend()

    plt.grid()
    plt.show()
