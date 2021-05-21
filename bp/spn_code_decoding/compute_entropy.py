import sys
sys.path.append('..')

import numpy as np
from scipy import integrate, interpolate
from torch_parallel.grid_gibbs import GibbsSampler
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    hs, stays = ising_entropy()

    hf, newstays = finite_entropy(np.arange(0.5, 1.0, 0.04), N=50)

    q = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    pgm = [1.04, 1.04, 1.04, 1.01186, 0.71966, 0.33407, 0.20538, 0.16562, 0.1470, 0.14261]
    spn = [1.04, 1.04, 1.04, 1.02601, 0.74532, 0.34625, 0.22401, 0.17977, 0.16053, 0.15176]

    fig, ax = plt.subplots()
    ax.plot(stays, hs, 'o-', label='H_inf')
    ax.plot(newstays, hf, '^-', label='H_f')
    ax.plot(q, pgm, '+-', label='pgm')
    ax.plot(q, spn, 'x-', label='spn')
    ax.set_ylabel('output bits per input bits')
    ax.set_xlabel('q')
    ax.legend()

    plt.grid()
    plt.show()