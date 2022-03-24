import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def get_rd_stats(results):

    rd = defaultdict(lambda: [])
    for x  in tqdm(results):
        num_bins, width, mse, sqnr, r_code = x
        rd[(num_bins, width)] += [[r_code * np.log2(num_bins) / 100 + 1, sqnr]]

    # Get some metrics like lowest rate, highest rate and average rate
    metrics = {}
    metrics['lowest_rate'] = {}
    metrics['highest_rate'] = {}
    metrics['average_rate'] = {}

    for num_bins in [2, 4, 8, 16, 32, 64, 128]:
        for width in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            print(f"Processing {(num_bins, width)} ...")
            rates = np.array(rd[(num_bins, width)])[:, 0]
            sqnrs = np.array(rd[(num_bins, width)])[:, 1]

            lowest_rate_idx = np.argmin(rates)
            highest_rate_idx = np.argmax(rates)

            metrics['lowest_rate'][f"{num_bins}_{width}"] = [rates[lowest_rate_idx], sqnrs[lowest_rate_idx]]
            metrics['highest_rate'][f"{num_bins}_{width}"] = [rates[highest_rate_idx], sqnrs[highest_rate_idx]]
            metrics['average_rate'][f"{num_bins}_{width}"] = [np.mean(rates), np.mean(sqnrs)]

    filepath = os.path.join('/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/spn_code_decoding/demo_gray_lossy', 'gauss-markov', 'spn' + '_' + 'test', 'metrics' + '.json')
    os.makedirs(os.path.join('/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/spn_code_decoding/demo_gray_lossy', 'gauss-markov', 'spn' + '_' + 'test'), exist_ok=True)
    with open(filepath, 'w') as file:
        json.dump(metrics, file, indent=4)

    return metrics

if __name__ == "__main__":

    filepath = "/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/spn_code_decoding/demo_gray_lossy/gauss-markov/spn_test/gauss_markov_07_071_0_1.json"
    with open(filepath, "r") as f:
        results = json.load(f)

    metrics = get_rd_stats(results['rates'])

    lr = np.array(list(metrics['lowest_rate'].values()))
    hr = np.array(list(metrics['highest_rate'].values()))
    ar = np.array(list(metrics['average_rate'].values()))

    fig, ax = plt.subplots()
    idx = [9, 15, 21, 38, 37, 36]
    s = np.arange(0, 30, 0.1)
    ax.plot(lr[idx, 1], lr[idx, 0], 'kx-', label='Lowest Rate')
    ax.plot(hr[idx, 1], hr[idx, 0], 'go-', label='Highest Rate$')
    ax.plot(ar[idx, 1], ar[idx, 0], 'm+--', label='Average Rate')
    ax.plot(s, 0.166 * s - 0.486, 'y--', label='Rate-Distortion Bound')
    ax.plot(s, 0.166 * s - 0.255, 'g--', label='ECUS')
    ax.set_ylabel('Bits per pixel')
    ax.set_xlabel('SQNR (dB)')
    ax.legend()

    plt.grid(True, linestyle='--')
    # plt.show()

    plt.savefig('gauss_markov_comparison.png')