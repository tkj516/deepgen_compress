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
        stays = np.arange(0.5, 0.95, 0.02)
    thetas = np.arctanh(2 * (stays - 0.5))

    thetas_left = thetas - 0.001
    thetas_right = thetas + 0.001

    for i in tqdm(range(len(thetas))):
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
    for theta in tqdm(newthetas):
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
    gzip = [1.030612244897959, 1.030612244897959, 1.030612244897959, 1.0305918367346938, 1.0167551020408163, 0.6528571428571429, 0.4486734693877551, 0.3719387755102041, 0.3307244897959184, 0.30307142857142855]
    bz2 = [1.6972755102040817, 1.694704081632653, 1.6805102040816324, 1.6221122448979592, 1.362877551020408, 0.905530612244898, 0.677704081632653, 0.5849285714285715, 0.5389285714285713, 0.5144591836734694]
    jbig2 = [0.99, 0.9907755102040817, 0.9906122448979593, 0.9824285714285713, 0.8459285714285715, 0.43851020408163266, 0.26753061224489794, 0.20369387755102042, 0.17539795918367346, 0.1604795918367347]

    sep_threshold_pgm = compute_sep_threshold(sep_prot_pgm)
    sep_threshold_spn = compute_sep_threshold(sep_prot_spn)

    # Interpolate the values for plotting
    q_interp = np.arange(0.50, 0.95, 0.02)
    sep_prot_pgm_interp = interpolate.interp1d(q, sep_prot_pgm)(q_interp)
    sep_prot_spn_interp = interpolate.interp1d(q, sep_prot_spn)(q_interp)
    sep_threshold_pgm_interp = interpolate.interp1d(q, sep_threshold_pgm)(q_interp)
    sep_threshold_spn_interp = interpolate.interp1d(q, sep_threshold_spn)(q_interp)
    gzip_interp = interpolate.interp1d(q, gzip)(q_interp)
    bz2_interp = interpolate.interp1d(q, bz2)(q_interp)
    jbig2_interp = interpolate.interp1d(q, jbig2)(q_interp)

    fig, ax = plt.subplots()
    ax.plot(stays, hs, 'k-', label='$H_\infty$')
    ax.plot(newstays, hf, 'g-', label='$H_{h, w}$')
    ax.plot(q_interp, sep_prot_pgm_interp, 'm+--', label='PGM-SEP-prot')
    ax.plot(q_interp, sep_prot_spn_interp, 'x--', label='SPN-SEP-prot')
    ax.plot(q_interp, sep_threshold_pgm_interp, 'bo-', label='PGM-SEP-thresh', markerfacecolor="None", ms=5)
    ax.plot(q_interp, sep_threshold_spn_interp, '^-', label='SPN-SEP-thresh', markerfacecolor="None", ms=5)
    ax.plot(q_interp, gzip_interp, '*-', label='GZIP', markerfacecolor="None", ms=5)
    ax.plot(q_interp, bz2_interp, 'D-', label='BZ2', markerfacecolor="None", ms=5)
    ax.plot(q_interp, jbig2_interp, 's-', label='JBIG2', markerfacecolor="None", ms=5)
    ax.set_ylabel('output bits per input bits')
    ax.set_xlabel('q')
    ax.set_xlim(None, 0.96)
    ax.legend()

    plt.grid(True, linestyle='--')
    # plt.show()

    plt.savefig('ising_comparison.png')
