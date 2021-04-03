import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

class CodeBP(nn.Module):

    def __init__(self, H, device='cuda:0'):

        super(CodeBP, self).__init__()

        # Get the shape of the parity check matrix
        self.K, self.N = H.shape
        self.H = H

        # Get the maximum number of variables neighbors and factor neighbors
        self.max_factor_neighbors = int(np.max(np.sum(H.cpu().numpy(), 0)))
        self.max_variable_neighbors = int(np.max(np.sum(H.cpu().numpy(), 1)))

        # Set an epsilon for high precision calculations
        self.epsilon = 1e-15

        # Build the adjaceny list for variable and check nodes
        self.neighbors_of_factor = self.neighbors_of_variable = []
        self.build_adjacency_list()

        # Initialize the messages to all 0
        # Set them as parameters for gradient backpropagation
        self.Hsx = nn.Parameter(torch.zeros(self.K, self.N))
        self.Hxs = nn.Parameter(torch.zeros(self.K, self.N))
        self.M_out = nn.Parameter(torch.zeros(self.N, 2))

        # Set the device
        self.device = device

    def build_adjacency_list(self):

        print("[Building adjacency list ...]")

        # Intialize the lists
        self.neighbors_of_factor = [[] for _ in range(self.K)]
        self.neighbors_of_variable = [[] for _ in range(self.N)]

        # Create a sparse matrix
        sp_H = csr_matrix(self.H.cpu().numpy())
        rows, cols = sp_H.nonzero()
        
        # Build the factor adjaceny list
        for a, i in tqdm(zip(rows, cols)):
            self.neighbors_of_variable[i].append(a)
            self.neighbors_of_factor[a].append(i)

    def forward(self, ps, x, Min):
        """
        Min : N x 2
        ps  : N x 2
        x   : K x 2
        Mout: N x 2
        """

        if Min is None:
            Min = 0.5 * torch.ones(self.N, 2).to(self.device)

        # Calculate the log likelihood of the messages
        log_Min = 0.5 * (torch.log(Min[:, 0]) - torch.log(Min[:, 1]))
        log_ps = 0.5 * (torch.log(ps[:, 0]) - torch.log(ps[:, 1]))

        #####################################################################################################
        # Node to factor messages
        #####################################################################################################

        grid = torch.LongTensor([[i]*self.max_factor_neighbors for i in range(self.N)]).to(self.device)
        factor_neighbors = torch.LongTensor(self.neighbors_of_variable).to(self.device)

        # Update messages first using external beliefs
        self.Hsx.data[factor_neighbors, grid] = log_Min.unsqueeze(-1) + log_ps.unsqueeze(-1)

        # Check if any of the messages from node to factor were infinity
        inf_check = torch.isinf(self.Hxs[factor_neighbors, grid])
        # Number of sure neighbors is the number of infinities
        sure_neighbors = torch.sum(inf_check.float(), -1, keepdim=True)
        # Calculate log-likelihood discarding sure neidhbors
        ll_sum = torch.sum(self.Hxs[factor_neighbors, grid].masked_fill(inf_check, 0), -1, keepdim=True)
        # Sure sum is nan if there are conflicting messages
        sure_sum = torch.sum(self.Hxs[factor_neighbors, grid].masked_fill(~inf_check, 0), -1, keepdim=True)

        # Zero sure neighbors
        check = torch.nonzero((sure_neighbors == 0).float())
        self.Hsx.data[factor_neighbors[check[:,0], :], grid[check[:,0], :]] += ll_sum[check[:,0]] - \
                                                        self.Hxs.data[factor_neighbors[check[:,0], :], grid[check[:,0], :]]
        # One sure neighbor
        check1 = torch.nonzero(torch.logical_and((sure_neighbors == 1), inf_check).float())
        check2 = torch.nonzero(torch.logical_and((sure_neighbors == 1), ~inf_check).float())
        if check1.shape[0] > 0:
            # Update the sure neighbor using incoming messages
            self.Hsx.data[factor_neighbors[check1[:,0], check1[:,1]], grid[check1[:,0], check1[:,1]]] += ll_sum[check1[:,0], 0]
        if check2.shape[0] > 0:
            # Other neighbors should be set to the sure neighbor's value
            self.Hsx.data[factor_neighbors[check2[:,0], check2[:,1]], grid[check2[:,0], check2[:,1]]] = sure_sum[check2[:,0], 0]
        # Multiple sure neighbors
        nan_check = torch.isnan(sure_sum)
        check1 = torch.nonzero(torch.logical_and((sure_neighbors > 1), nan_check).float())
        check2 = torch.nonzero(torch.logical_and((sure_neighbors > 1), ~nan_check).float())
        if check1.shape[0] > 0:
            # If nan then conflicting messages so set to 0
            self.Hsx.data[factor_neighbors[check1[:,0], :], grid[check1[:,0], :]] = 0
        if check2.shape[0] > 0:
            self.Hsx.data[factor_neighbors[check2[:,0], :], grid[check2[:,0], :]] += sure_sum[check2[:,0]]

        #####################################################################################################
        # Factor to node messages
        #####################################################################################################
        
        grid = torch.LongTensor([[i]*self.max_variable_neighbors for i in range(self.K)]).to(self.device)
        variable_neighbors = torch.LongTensor(self.neighbors_of_factor).to(self.device)

        ll_diff_prod = 1 - 2*x
        # Calculate all the log-likelihood terms that will be multiplied at each factor
        ll_diff = torch.tanh(self.Hsx.data[grid, variable_neighbors])
        # Check if any of the terms are zero
        zero_check = (ll_diff == 0)
        # Now calculate the log-likelihood product for each check variable
        ll_diff_prod *= torch.prod(ll_diff.masked_fill(zero_check, 1), -1, keepdim=True)
        # Caculates the number of null neighbors at each factor
        null_neighbors = torch.sum(zero_check.float(), -1, keepdim=True)

        # Greater than one null neighbor
        check = torch.nonzero((null_neighbors > 1).float())
        if check.shape[0] > 0:
            self.Hxs.data[grid[check[:,0], :], variable_neighbors[check[:,0], :]] = 0
        # One null neighbor
        check1 = torch.nonzero(torch.logical_and(null_neighbors == 1, ~zero_check).float())
        check2 = torch.nonzero(torch.logical_and(null_neighbors == 1, zero_check).float())
        if check1.shape[0] > 0:
            self.Hxs.data[grid[check1[:,0], check1[:,1]], variable_neighbors[check1[:,0], check1[:,1]]] = 0
        if check2.shape[0] > 0:
            self.Hxs.data[grid[check2[:,0], check2[:,1]], variable_neighbors[check2[:,0], check2[:,1]]] = \
                                                                torch.atanh(ll_diff_prod[check2[:,0], 0])
        # Zero null neighbors
        check = torch.nonzero((null_neighbors == 0).float())
        if check.shape[0] > 0:
            self.Hxs.data[grid[check[:,0], :], variable_neighbors[check[:,0], :]] = torch.atanh(ll_diff_prod[check[:,0]] / ll_diff[check[:,0], :])

        #####################################################################################################
        # Outgoing super edge messages
        #####################################################################################################

        grid = torch.LongTensor([[i]*3 for i in range(self.N)]).to(self.device)
        # Sum up all the incoming messages at each node and apply tanh to get likelihood
        M_out_diff = torch.tanh(torch.sum(self.Hxs.data[factor_neighbors, grid], -1, keepdim=True))
        nan_check = torch.logical_and(torch.isnan(M_out_diff), torch.isinf(self.Hxs[factor_neighbors, grid]))
        nan_check_idx = torch.nonzero(torch.sum(nan_check.float(), -1, keepdim=True))
        # Resolve nans by calculating meaningful messages
        if nan_check_idx.shape[0] > 0:
            M_out_diff[nan_check_idx[:,0], 0] = \
                torch.tanh(torch.sum(self.Hxs.data[factor_neighbors, grid].masked_fill(nan_check, 0), -1, keepdim=True))[nan_check_idx[:,0],0]
        # Convert likelihoods back to probability
        self.M_out.data = 0.5 + torch.cat([M_out_diff, -M_out_diff], -1)/2

#################################################################################################################
# ITERATIVE MESSAGE PASSING ARCHIVE
#################################################################################################################


        # Node to factor messages
        # for i in range(self.N):
        #     sure_neighbors = []
        #     factor_neighbors = self.neighbors_of_variable[i]
        #     self.Hsx[factor_neighbors, i] = log_Min[i] + log_ps[i]
        #     ll_sum = 0

        #     for k in factor_neighbors:
        #         if torch.isinf(self.Hxs[k, i]):
        #             sure_neighbors.append(k)
        #         else:
        #             ll_sum += self.Hxs[k, i]

        #     sure_sum = torch.sum(self.Hxs[sure_neighbors, i])
        #     if len(sure_neighbors) == 0:
        #         self.Hsx[factor_neighbors, i] += ll_sum - self.Hxs[factor_neighbors, i]
        #     elif len(sure_neighbors) == 1:
        #         for k in factor_neighbors:
        #             if torch.isinf(self.Hxs[k, i]):
        #                 self.Hsx[k, i] += ll_sum
        #             else:
        #                 self.Hsx[k, i] = sure_sum
        #     else:
        #         if torch.isnan(sure_sum):
        #             self.Hsx[factor_neighbors, i] = 0
        #         else:
        #             self.Hsx[factor_neighbors, i] += sure_sum

        # # Factor to node messages
        # for a in range(self.K):
        #     variable_neighbors = self.neighbors_of_factor[a]
            
        #     # Multiply all the likelihoods
        #     # Initialize diff product as -1 or 1
        #     ll_diff_prod = 1 - 2*x[a]
        #     null_neighbors = []
        #     for j in variable_neighbors:
        #         ll_diff = torch.tanh(self.Hsx[a, j])
        #         if ll_diff == 0:
        #             null_neighbors.append(j)
        #         else:
        #             ll_diff_prod *= ll_diff
        #     if len(null_neighbors) > 1:
        #         for j in variable_neighbors:
        #             self.Hxs[a, j] = 0
        #     elif len(null_neighbors) == 1:
        #         for j in variable_neighbors:
        #             if null_neighbors[0] != j:
        #                 self.Hxs[a, j] = 0
        #             else:
        #                 self.Hxs[a, j] = torch.atanh(ll_diff_prod)
        #     else:
        #         for j in variable_neighbors:
        #             temp = ll_diff_prod / torch.tanh(self.Hsx[a, j])
        #             self.Hxs[a, j] = torch.atanh(temp)

        # # Compute outgoing super edge message
        # for i in range(self.N):
        #     factor_neighbors = self.neighbors_of_variable[i]

        #     M_out_diff = torch.tanh(torch.sum(self.Hxs[factor_neighbors, i], 0))
        #     if torch.isnan(M_out_diff):
        #         M_out_diff = torch.tanh(torch.sum(self.Hxs[factor_neighbors[not torch.isinf(self.Hxs[factor_neighbors, i])], i]))
        #     self.M_out[i, :] = torch.tensor(np.array([0.5 + M_out_diff/2, 0.5 - M_out_diff/2]))






