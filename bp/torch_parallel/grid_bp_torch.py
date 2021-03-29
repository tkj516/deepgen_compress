import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class GridBP(nn.Module):

    def __init__(self, h, w, p=0.5, stay=0.9, alpha=0.8, device='cuda:0'):

        super(GridBP, self).__init__()

        # Get the height and width of the image
        self.h = h
        self.w = w
        self.N = h * w 
        
        # Define the node and edge potentials
        # Register the potentials as buffers to indicate non-trainable members of the module
        self.epot = np.array([[stay, 1-stay],
                              [1-stay, stay]]).reshape(1, 1, 2, 2)
        self.register_buffer('npot', torch.tensor(np.tile(np.array([1-p, p]), (h*w, 1)).reshape(h, w, 2)))
        self.register_buffer('epot_v', torch.tensor(np.tile(self.epot, (h-1, w, 1, 1))))
        self.register_buffer('epot_h', torch.tensor(np.tile(self.epot, (h, w-1, 1, 1))))

        # Initialize the messages
        # Set them as parameters for gradient backpropagation
        self.Md = nn.Parameter(torch.ones(self.h-1, self.w, 2)/2)
        self.Mu = nn.Parameter(torch.ones(self.h-1, self.w, 2)/2)
        self.Mr = nn.Parameter(torch.ones(self.h, self.w-1, 2)/2)
        self.Ml = nn.Parameter(torch.ones(self.h, self.w-1, 2)/2)
        self.Mout = nn.Parameter(torch.ones(self.h, self.w, 2)/2)

        # Initialize the learning rate
        self.alpha = alpha

        # Store the device name
        self.device = device

    def forward(self, Min):
        """
        Min : h x w x 2
        Mout: h x w x 2
        """
       
        # Resolve contradictory signals
        S = torch.sum(self.npot * Min, -1)
        check_zero = torch.nonzero((S == 0).float())
        Min[check_zero[:,0], check_zero[:,0], 0][S == 0] = 0.5
        Min[check_zero[:,0], check_zero[:,0], 1][S == 0] = 0.5

        # Top to bottom
        Md1  = self.npot[:-1, ...] * Min[:-1, ...]
        Md1[1:, ...] *= self.Md[:-1, ...]
        Md1[:, 1:, :] *= self.Mr[:-1, ...]
        Md1[:, :-1, :] *= self.Ml[:-1, ...]
        Md1[..., 0], Md1[..., 1] = torch.sum(self.epot_v[..., 0, :]*Md1, -1), torch.sum(self.epot_v[..., 1, :]*Md1, -1)

        # Bottom to top
        Mu1  = self.npot[1:, ...] * Min[1:, ...]
        Mu1[:-1, ...] *= self.Mu[1:, ...]
        Mu1[:, 1:, :] *= self.Mr[1:, ...]
        Mu1[:, :-1, :] *= self.Ml[1:, ...]
        Mu1[..., 0], Mu1[..., 1] = torch.sum(self.epot_v[..., 0]*Mu1, -1), torch.sum(self.epot_v[..., 1]*Mu1, -1)

        # Left to right
        Mr1  = self.npot[:, :-1, :] * Min[:, :-1, :]
        Mr1[:-1, ...] *= self.Mu[:, :-1, :]
        Mr1[:, 1:, :] *= self.Mr[:, :-1, :]
        Mr1[1:, ...] *= self.Md[:, :-1, :]
        Mr1[..., 0], Mr1[..., 1] = torch.sum(self.epot_h[..., 0, :]*Mr1, -1), torch.sum(self.epot_h[..., 1, :]*Mr1, -1)

        # Right to left
        Ml1  = self.npot[:, 1:, :] * Min[:, 1:, :]
        Ml1[:-1, ...] *= self.Mu[:, 1:, :]
        Ml1[:, :-1, :] *= self.Ml[:, 1:, :]
        Ml1[1:, :, :] *= self.Md[:, 1:, :]
        Ml1[..., 0], Ml1[..., 1] = torch.sum(self.epot_h[..., 0]*Ml1, -1), torch.sum(self.epot_h[..., 1]*Ml1, -1)

        # Normalize the messages
        Md1 /= torch.sum(Md1, -1).reshape(self.h-1, self.w, 1)
        Mu1 /= torch.sum(Mu1, -1).reshape(self.h-1, self.w, 1)
        Mr1 /= torch.sum(Mr1, -1).reshape(self.h, self.w-1, 1)
        Ml1 /= torch.sum(Ml1, -1).reshape(self.h, self.w-1, 1)

        # Update the messages
        # Since the messages are parameters access using .data
        self.Md.data = self.alpha*Md1 + (1-self.alpha)*self.Md.data
        self.Mu.data = self.alpha*Mu1 + (1-self.alpha)*self.Mu.data
        self.Mr.data = self.alpha*Mr1 + (1-self.alpha)*self.Mr.data
        self.Ml.data = self.alpha*Ml1 + (1-self.alpha)*self.Ml.data

        # Compute Mout
        self.Mout.data = torch.ones(self.h, self.w, 2).to(self.device)
        self.Mout[1:, :, :] *= self.Md
        self.Mout[:-1, ...] *= self.Mu
        self.Mout[:, 1:, :] *= self.Mr
        self.Mout[:, :-1, :] *= self.Ml
        self.Mout.data /= torch.sum(self.Mout.data, -1).unsqueeze(-1)

    def reset_messages(self):

        # Reset the message parameters
        self.Md = nn.Parameter(torch.ones(self.h-1, self.w, 2)/2).to(self.device)
        self.Mu = nn.Parameter(torch.ones(self.h-1, self.w, 2)/2).to(self.device)
        self.Mr = nn.Parameter(torch.ones(self.h, self.w-1, 2)/2).to(self.device)
        self.Ml = nn.Parameter(torch.ones(self.h, self.w-1, 2)/2).to(self.device)
        self.Mout = nn.Parameter(torch.ones(self.h, self.w, 2)/2).to(self.device)

    def bp(self, Min, num_iter=1):

        for i in tqdm(range(num_iter)):
            self.forward(Min)

def test_grid_bp():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    grid_bp = GridBP(128, 128).to(device)
    Min = torch.rand(128, 128, 2).to(device)
    grid_bp.bp(Min)

if __name__ == "__main__":
    test_grid_bp()



