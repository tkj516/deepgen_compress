import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import time
import argparse

from code_bp_torch_v2 import CodeBP
from grid_bp_torch import GridBP
from grid_gibbs import GibbsSampler

parser = argparse.ArgumentParser(description='Belief propagation training arguments')
parser.add_argument('--height', type=int, default=128, help="Height of image")
parser.add_argument('--width', type=int, default=128, help="Width of image")
parser.add_argument('--ldpc_mat', type=str, default='../H.mat', help="Path to LDPC matrix")
parser.add_argument('--load_image', action='store_true', help="Load an image for testing")
parser.add_argument('--image', type=str, default='S_128.mat', help="Path to loadable image")
parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the code on")
parser.add_argument('--num_iters', type=int, default=100, help="Number of bp iterations")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

print(f"[Using device {args.device}]")

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
        self.source = GridBP(self.h, self.w, self.p, self.stay, self.alpha, device).to(device)
        print("[Setup the source graph ...]")

        # Setup the code graph
        self.code = CodeBP(self.H, device).to(device)
        print("[Setup the code graph ...]")

        # Store a matrix for doping probabilities
        self.ps = torch.FloatTensor(np.tile(np.array([1-p, p]), (h*w, 1))).to(device)

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
        self.source.npot.data = self.ps.reshape(self.h, self.w, 2)

    def generate_sample(self):

        self.sampler.sampler(1000)
        self.samp = torch.FloatTensor(self.sampler.samp.reshape(-1, 1)).to(device)

    def encode(self):

        self.x = (self.H @ self.samp) % 2

    def decode_step(self):

        # Perform one step of code graph belief propagation
        self.code(self.ps, self.x, self.M_to_code)
        self.M_from_code = self.code.M_out
        # Reshape to send to grid
        self.M_to_grid = self.M_from_code.reshape(self.h, self.w, 2)

        # Perform one step of source graph belief propagation
        self.source(self.M_to_grid)
        self.M_from_grid = self.source.Mout
        # Reshape to send to code
        self.M_to_code = self.M_from_grid.reshape(self.N, 2)

    def decode(self, num_iter=1):

        # Set the initial beliefs to all nans
        B_old = torch.tensor(float('nan') * np.ones((self.h, self.w))).to(device)
        start = time.time()

        # Perform multiple iterations of belief propagation
        for i in range(num_iter):

            # Perform a step of message passing/decoding
            self.decode_step()

            # Calculate the belief
            self.B = self.M_from_grid * self.M_to_grid * self.source.npot
            self.B /= torch.sum(self.B, -1).unsqueeze(-1)

            # Termination condition to end belief propagation
            if torch.sum(torch.abs(self.B[..., 1] - B_old)).item() < 0.5:
                break
            B_old = self.B[..., 1]

            # Compute the number of errors and print some information
            errs = torch.sum(torch.abs((self.B[..., 1] > 0.5).float() - self.samp.reshape(self.h, self.w))).item()
            print(f'Iteration {i}: {errs} errors')

        end = time.time()
        print(f'Total time taken for decoding is {end - start}s')

def test_source_code_bp():

    h = args.height
    w = args.width

    # Load the LDPC matrix
    H = torch.FloatTensor(loadmat(args.ldpc_mat)['Hf']).to(device)

    # Intialize the source-code decoding graph
    source_code_bp = SourceCodeBP(H, h=h, w=w)

    # Either load a sample image or generate one using Gibb's sampling
    print("[Generating the sample ...]")
    if args.load_image:
        source_code_bp.samp = torch.FloatTensor(loadmat(args.image)['Sb']).reshape(-1, 1).to(device)
    else:
        source_code_bp.generate_sample()
    
    # Encode the sample using the LDPC matrix
    print("[Encoding the sample ...]")
    source_code_bp.encode()

    # Dope it to update our initial beliefs
    print("[Doping ...]")
    source_code_bp.doping()

    # Decode the code using belief propagation
    print("[Decoding ...]")
    source_code_bp.decode(num_iter=args.num_iters)

    # Visualize the decoded image
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(source_code_bp.samp.cpu().numpy().reshape(h, w))
    ax[0].set_title("Source Image")
    ax[1].imshow(np.argmax(source_code_bp.B.detach().cpu().numpy(), axis=-1))
    ax[1].set_title("Reconstructed Image")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    test_source_code_bp()

