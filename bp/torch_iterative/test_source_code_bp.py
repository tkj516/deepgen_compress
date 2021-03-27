import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm

from code_bp import CodeBP
from grid_bp import GridBP
from grid_gibbs import GibbsSampler

class SourceCodeBP():

    def __init__(self,
                 H,
                 h=128,
                 w=128,
                 p=0.5, 
                 stay=0.9,
                 alpha=0.8,
                 doperate=0.04):

        # Store the parameters
        self.h = h
        self.w = w
        self.p = p
        self.stay = stay
        self.alpha = alpha
        self.doperate = doperate

        # Store the parity check matrix
        self.H = H
        self.K, self.N = self.H.shape

        # Setup the Gibbs sampler
        self.sampler = GibbsSampler(self.h, self.w, self.p, self.stay)
        print("[Setup the sampler ...]")

        # Setup the source graph
        self.source = GridBP(self.h, self.w, self.p, self.stay, self.alpha)
        print("[Setup the source graph ...]")

        # Setup the code graph
        self.code = CodeBP(self.H)
        print("[Setup the code graph ...]")

        # Store a matrix for doping probabilities
        self.ps = torch.FloatTensor(np.tile(np.array([1-p, p]), (h*w, 1)))

        # Input image
        self.samp = None

        # Encoded image
        self.x = None

        # Initialize the messages
        self.M_to_code = None
        self.M_to_grid = None
        self.B = None

    def doping(self):

        indices = np.random.randint(self.N, size=int(self.N*self.doperate)+1)
        self.ps[indices, 0], self.ps[indices, 1] = (self.samp[indices, 0] == 0).float(), (self.samp[indices, 0] == 1).float()
        # Update the node potential after doping
        self.source.npot = self.ps.reshape(self.h, self.w, 2)

    def generate_sample(self):

        self.sampler.sampler(1000)
        self.samp = torch.FloatTensor(self.sampler.samp.reshape(-1, 1))

    def encode(self):

        self.x = (self.H @ self.samp) % 2

    def decode_step(self):

        self.code.update(self.ps, self.x, self.M_to_code)
        self.M_from_code = self.code.M_out
        # Reshape to send to grid
        self.M_to_grid = self.M_from_code.reshape(self.h, self.w, 2)

        self.source.update(self.M_to_grid)
        self.M_from_grid = self.source.Mout
        # Reshape to send to code
        self.M_to_code = self.M_from_grid.reshape(self.N, 2)

    def decode(self, num_iter=1):

        B_old = torch.tensor(float('nan') * np.ones((self.h, self.w)))
        for i in tqdm(range(num_iter)):

            self.decode_step()

            # Calculate the belief
            self.B = self.M_from_grid * self.M_to_grid * self.source.npot
            self.B /= torch.sum(self.B, -1).unsqueeze(-1)

            if torch.sum(torch.abs(self.B[..., 1] - B_old)).item() < 0.5:
                break
            B_old = self.B[..., 1]
            errs = torch.sum(torch.abs((self.B[..., 1] > 0.5).float() - self.samp.reshape(self.h, self.w))).item()
            print(errs)

def test_source_code_bp():

    # Load the LDPC matrix
    H = torch.FloatTensor(loadmat('H.mat')['Hf'])

    source_code_bp = SourceCodeBP(H, h=64, w=64)

    print("[Generating the sample ...]")
    source_code_bp.generate_sample()

    print("[Encoding the sample ...]")
    source_code_bp.encode()

    print("[Doping ...]")
    source_code_bp.doping()

    print("[Decoding ...]")
    source_code_bp.decode(num_iter=100)

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(source_code_bp.samp.cpu().numpy().reshape(64, 64))
    ax[1].imshow(np.argmax(source_code_bp.B.cpu().numpy(), axis=-1))
    plt.show()

if __name__ == "__main__":

    test_source_code_bp()

