import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import savgol_filter

def get_rd_stats(results):

    rd = defaultdict(lambda: [0, 0])
    cd = defaultdict(lambda: 0)
    for x  in tqdm(results):
        i, b, w, mse_opt, sqnr_opt, best_rate = x
        if not sqnr_opt == 0:
            rd[(b, w)] = [best_rate + rd[(b, w)][0], sqnr_opt + rd[(b, w)][1]]
            cd[(b, w)] += 1
            
    for key in rd.keys():
        rd[key][0] /= cd[key]
        rd[key][1] /= cd[key]
    
    print(rd)

    # Get some metrics like lowest rate, highest rate and average rate
    metrics = {}
    metrics['average_rate'] = rd

    return metrics

def get_rd_stats_jpeg(results):

    rd = defaultdict(lambda: [0, 0])
    cd = defaultdict(lambda: 0)
    for x  in tqdm(results):
        i, bpp, actual_bpp, mse, sqnr, = x
        rd[bpp] = [actual_bpp + rd[bpp][0], sqnr + rd[bpp][1]]
        cd[bpp] += 1
            
    for key in rd.keys():
        rd[key][0] /= cd[key]
        rd[key][1] /= cd[key]
    
    print(rd)

    # Get some metrics like lowest rate, highest rate and average rate
    metrics = {}
    metrics['average_rate'] = rd

    return metrics

if __name__ == "__main__":

    filepath = "/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/gauss_markov/demo_gray_lossy/cifar-10/results.json"
    with open(filepath, "r") as f:
        results = json.load(f)

    metrics = get_rd_stats(results)
    ar = np.array(list(metrics['average_rate'].values()))
    ar = ar[ar[:, 0].argsort()]
    print(ar)

    filepath = "/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/gauss_markov/demo_gray_lossy/cifar-10/results_jpeg.json"
    with open(filepath, "r") as f:
        results = json.load(f)

    metrics = get_rd_stats_jpeg(results)
    jpg = np.array(list(metrics['average_rate'].values()))
    jpg= jpg[jpg[:, 0].argsort()]
    print(jpg)

    fig, ax = plt.subplots()
    idx = [0, 4, 11, 14]

    idx_jpeg = list(range(jpg.shape[0]))
    idx_jpeg.pop(-2)
    poly = np.polyfit(jpg[idx_jpeg, 1], jpg[idx_jpeg, 0], 5)
    y_hat = np.poly1d(poly)(jpg[idx_jpeg, 1])
    print(y_hat)

    ax.plot(ar[idx, 1], ar[idx, 0], 'r+--', label='Average Rate SPN')
    ax.scatter(jpg[idx_jpeg, 1], jpg[idx_jpeg, 0], marker='x', label='JPEG')
    ax.plot(jpg[idx_jpeg, 1], y_hat, 'b-.', label='JPEG-smoothened')
    ax.set_ylabel('Bits per pixel')
    ax.set_xlabel('SQNR (dB)')
    ax.set_xlim([14, 32])
    ax.legend()

    plt.grid(True, linestyle='--')
    # plt.show()

    plt.savefig('cifar10_comparison_v2.png')