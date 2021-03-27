import torch
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

class CodeBP():

    def __init__(self, H):

        # Get the shape of the parity check matrix
        self.K, self.N = H.shape
        self.H = H

        # Set an epsilon for high precision calculations
        self.epsilon = 1e-15

        # Build the adjaceny list for variable and check nodes
        self.neighbors_of_factor = self.neighbors_of_variable = []
        self.build_adjacency_list()

        # Initialize the messages to all 0
        self.Hsx = np.zeros((self.K, self.N))
        self.Hxs = np.zeros((self.K, self.N))
        self.M_out = np.zeros((self.N, 2))

    def build_adjacency_list(self):

        print("[Building adjacency list ...]")

        # Intialize the lists
        self.neighbors_of_factor = [[] for _ in range(self.K)]
        self.neighbors_of_variable = [[] for _ in range(self.N)]

        # Create a sparse matrix
        sp_H = csr_matrix(self.H)
        rows, cols = sp_H.nonzero()
        
        # Build the factor adjaceny list
        for a, i in tqdm(zip(rows, cols)):
            self.neighbors_of_variable[i].append(a)
            self.neighbors_of_factor[a].append(i)

    def update(self, ps, x, Min):
        """
        Min : N x 2
        ps  : N x 2
        Mout: N x 2
        """

        if Min is None:
            Min = np.ones((self.N, 2))/2

        # Calculate the log likelihood of the messages
        log_Min = 0.5 * (np.log(Min[:, 0]) - np.log(Min[:, 1]))
        log_ps = 0.5 * (np.log(ps[:, 0]) - np.log(ps[:, 1]))

        # Node to factor messages
        for i in range(self.N):
            sure_neighbors = []
            factor_neighbors = self.neighbors_of_variable[i]
            self.Hsx[factor_neighbors, i] = log_Min[i] + log_ps[i]
            ll_sum = 0

            for k in factor_neighbors:
                if np.isinf(self.Hxs[k, i]):
                    sure_neighbors.append(k)
                else:
                    ll_sum += self.Hxs[k, i]

            sure_sum = np.sum(self.Hxs[sure_neighbors, i])
            if len(sure_neighbors) == 0:
                self.Hsx[factor_neighbors, i] += ll_sum - self.Hxs[factor_neighbors, i]
            elif len(sure_neighbors) == 1:
                for k in factor_neighbors:
                    if np.isinf(self.Hxs[k, i]):
                        self.Hsx[k, i] += ll_sum
                    else:
                        self.Hsx[k, i] = sure_sum
            else:
                if np.isnan(sure_sum):
                    self.Hsx[factor_neighbors, i] = 0
                else:
                    self.Hsx[factor_neighbors, i] += sure_sum

        # Factor to node messages
        for a in range(self.K):
            variable_neighbors = self.neighbors_of_factor[a]
            
            # Multiply all the likelihoods
            # Initialize diff product as -1 or 1
            ll_diff_prod = 1 - 2*x[a]
            null_neighbors = []
            for j in variable_neighbors:
                ll_diff = np.tanh(self.Hsx[a, j])
                if ll_diff == 0:
                    null_neighbors.append(j)
                else:
                    ll_diff_prod *= ll_diff
            if len(null_neighbors) > 1:
                for j in variable_neighbors:
                    self.Hxs[a, j] = 0
            elif len(null_neighbors) == 1:
                for j in variable_neighbors:
                    if null_neighbors[0] != j:
                        self.Hxs[a, j] = 0
                    else:
                        self.Hxs[a, j] = np.arctanh(ll_diff_prod)
            else:
                for j in variable_neighbors:
                    temp = ll_diff_prod / np.tanh(self.Hsx[a, j])
                    self.Hxs[a, j] = np.arctanh(temp)
            
        # Compute outgoing super edge message
        for i in range(self.N):
            factor_neighbors = self.neighbors_of_variable[i]

            M_out_diff = np.tanh(np.sum(self.Hxs[factor_neighbors, i], 0))
            if np.isnan(M_out_diff):
                M_out_diff = np.tanh(np.sum(self.Hxs[factor_neighbors[not np.isinf(self.Hxs[factor_neighbors, i])], i]))
            self.M_out[i, :] = np.array([0.5 + M_out_diff/2, 0.5 - M_out_diff/2])






