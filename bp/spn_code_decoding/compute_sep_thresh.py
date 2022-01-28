import sys
sys.path.append('..')

import os
import json
import argparse
import numpy as np
from ldpc_generate import pyldpc_generate
from tqdm import tqdm

def compute_sep_threshold(sep_prot, doperate=0.04, binary=False):

    # Get the base decoding rate by subtracting doping rate
    base_rates = np.array(sep_prot) - doperate

    sep_threshold = []

    x = np.arange(0.01, 1, 0.01)

    bits = 1 if binary else 8

    for i in tqdm(range(len(base_rates))):

        r = base_rates[i] / 100.0

        # Create LDPC matrix for this rate and compute
        # mean row sum (rho_bar)
        lambda_bar = 3
        H = pyldpc_generate.generate(int(r*784*bits), 784*bits, 3.0, 2, 123)
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
    parser = argparse.ArgumentParser(description='Sep Threshold Arguments')
    parser.add_argument('--json_file', type=str, required=True, help='Path to json file.')
    parser.add_argument('--binary', action='store_true', default=False, help="Is data binary")
    args = parser.parse_args()

    with open(args.json_file) as f:
        results = json.load(f)
    rates = results['rates']

    rates_thresh = compute_sep_threshold(rates, doperate=0.00)
    avg_rate_thresh = np.average(rates_thresh)
    min_rate_thresh = np.min(rates_thresh)
    max_rate_thresh = np.max(rates_thresh)

    results_thresh = {}
    results_thresh['rates'] = rates_thresh
    results_thresh['avg_rate'] = avg_rate_thresh
    results_thresh['min_rate'] = min_rate_thresh
    results_thresh['max_rate'] = max_rate_thresh

    print(f'Avg Rate: {avg_rate_thresh}, Min Rate: {min_rate_thresh}, Max Rate: {max_rate_thresh}')

    if args.binary:
        with open(args.json_file.split('.')[0] + '_bin_thresh.json', 'w') as file:
            json.dump(results_thresh, file, indent=4)
    else:
        with open(args.json_file.split('.')[0] + '_thresh.json', 'w') as file:
                json.dump(results_thresh, file, indent=4)