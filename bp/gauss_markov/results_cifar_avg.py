import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def get_rd_stats(results, idx):

    rd = defaultdict(lambda: [])
    for x  in tqdm(results):
        i, b, w, mse_opt, sqnr_opt, best_rate = x
        if i == idx and not sqnr_opt == 0:
            rd[(b, w)] += [[best_rate, sqnr_opt]]

    print(rd)

    # Get some metrics like lowest rate, highest rate and average rate
    metrics = {}
    metrics['lowest_rate'] = {}
    metrics['highest_rate'] = {}
    metrics['average_rate'] = {}

    for num_bins in [3, 5, 7, 8]:
        for width in [0.01, 0.1, 0.2, 0.4, 0.6]:

            if (num_bins, width) not in rd.keys():
                continue
            print(f"Processing {(num_bins, width)} ...")
            rates = np.array(rd[(num_bins, width)])[:, 0]
            sqnrs = np.array(rd[(num_bins, width)])[:, 1]

            lowest_rate_idx = np.argmin(rates)
            highest_rate_idx = np.argmax(rates)

            metrics['lowest_rate'][f"{num_bins}_{width}"] = [rates[lowest_rate_idx], sqnrs[lowest_rate_idx]]
            metrics['highest_rate'][f"{num_bins}_{width}"] = [rates[highest_rate_idx], sqnrs[highest_rate_idx]]
            metrics['average_rate'][f"{num_bins}_{width}"] = [np.mean(rates), np.mean(sqnrs)]

    return metrics

if __name__ == "__main__":

    filepath = "/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/gauss_markov/demo_gray_lossy/cifar-10/results.json"
    with open(filepath, "r") as f:
        results = json.load(f)

    metrics = get_rd_stats(results, idx=0)
    ar = np.array(list(metrics['average_rate'].values()))
    print(ar)

    fig, ax = plt.subplots()
    idx = [2, 1, 5, 4]
    s = np.arange(0, 30, 0.1)
    ax.plot(ar[idx, 1], ar[idx, 0], 'm+--', label='Average Rate PGM')
    ax.set_ylabel('Bits per pixel')
    ax.set_xlabel('SQNR (dB)')
    ax.set_xlim([10, 32])
    ax.legend()

    plt.grid(True, linestyle='--')
    # plt.show()

    plt.savefig('cifar10_comparison.png')