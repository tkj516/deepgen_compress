import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import time
import argparse
from datetime import datetime
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from functools import partial
import matplotlib.pyplot as plt
import os

from torch_parallel.code_bp_torch_v2 import CodeBP
from torch_parallel.grid_bp_torch import GridBP
from torch_parallel.grid_gibbs import GibbsSampler
from pixel_models.pixelcnnpp import *
from pixel_models.pixelcnn import *

parser = argparse.ArgumentParser(description='Belief propagation training arguments')
parser.add_argument('--h', type=int, default=28, help="Height of the images")
parser.add_argument('--w', type=int, default=28, help="Width of the images")
parser.add_argument('--n_bits', type=int, default=1, help="Number of itensity levels in PixelCNN")
parser.add_argument('--ldpc_mat', type=str, default='H_28.mat', help="Path to LDPC matrix")
parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the code on")
parser.add_argument('--restore_file_0', type=str, default='/fs/data/tejasj/Masters_Thesis/pixel_models/results/pixelcnn/2021-04-15_05-19-01/checkpoint.pt', help="Directory with checkpoint for orig orientation model")
parser.add_argument('--restore_file_180', type=str, default='/fs/data/tejasj/Masters_Thesis/pixel_models/results/pixelcnn/2021-04-15_16-11-09/checkpoint.pt', help="Directory with checkpoint for rotated orientation model")
parser.add_argument('--arch', type=str, default='pixelcnn', help="Type of architecture")
parser.add_argument('--num_iter', type=int, default=100, help="Number of bp iterations")
parser.add_argument('--data_path', default='datasets/ising_28_05_09_75000', help='Location of datasets.')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class IsingDataset(Dataset):
    """Dataset of Gibbs sampled Ising images"""

    def __init__(self, 
                root_dir='../deepgen_compress/bp/datasets/ising_28_05_09_75000',
                phase='train', 
                transform=None):
        # Root directory for the data
        self.root_dir = root_dir
        # Choose the phase
        self.phase = phase

        # Choose the number of files
        if self.phase == 'train':
            start_idx = 0
            end_idx = 70000
        else:
            start_idx = 70000
            end_idx = 75000

        # Read the training files from the mat file
        self.files = sorted(os.listdir(self.root_dir))[start_idx:end_idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = os.path.join(self.root_dir, self.files[idx])
        image = np.load(image)

        sample = torch.FloatTensor(image).unsqueeze(0)

        return sample, torch.tensor([0])

class Source():

    def __init__(self,
                 arch='pixelcnn',
                 image_dims=(1, 28, 28),
                 n_channels=128,
                 n_res_layers=5,
                 n_logistic_mix=10,
                 n_bits=1,
                 n_out_conv_channels=1024,
                 kernel_size=5,
                 norm_layer=True,
                 n_cond_classes=None):
        
        # Define the pixelcnn architecture that is being used
        self.arch = arch

        # Load the model
        if self.arch == 'pixelcnnpp':
            self.model = PixelCNNpp(image_dims, n_channels, n_res_layers, n_logistic_mix,
                                        n_cond_classes).to(device)
        elif self.arch == 'pixelcnn':
            n_res_layers = 12
            self.model_0 = MyDataParallel(PixelCNN(image_dims, n_bits, n_channels, n_out_conv_channels, kernel_size,
                                        n_res_layers, n_cond_classes, norm_layer)).to(device)
            self.model_180 = MyDataParallel(PixelCNN(image_dims, n_bits, n_channels, n_out_conv_channels, kernel_size,
                                        n_res_layers, n_cond_classes, norm_layer)).to(device)

            # Restore the checkpoints and set to evaluation mode
            model_checkpoint = torch.load(args.restore_file_0, map_location=device)
            self.model_0.load_state_dict(model_checkpoint['state_dict'])
            self.model_0.eval()
            model_checkpoint = torch.load(args.restore_file_180, map_location=device)
            self.model_180.load_state_dict(model_checkpoint['state_dict'])
            self.model_180.eval()           
        else:
            pass

        # Define the transform being used when computing source graph inputs
        self.transform = T.Compose([#T.ToTensor(),                                            # tensor in [0,1]
                                    lambda x: x.mul(255).div(2**(8-n_bits)).floor(),    # lower bits
                                    partial(self.preprocess)])                # to model space [-1,1]

        self.n_bits = n_bits

    def preprocess(self, x):
        # 1. convert data to float
        # 2. normalize to [0,1] given quantization
        # 3. shift to [-1,1]
        return x.float().div(2**self.n_bits - 1).mul(2).add(-1)

    def convert_logits_to_prob(self, l):

        # The input logits is of shape (B, 2**n_bits, C, H, W), convert to probs
        out = F.softmax(l.squeeze(2), 1)

        return out

    def discretized_mix_logistic_loss(self, l, x):
        """ log likelihood for mixture of discretized logistics
        Args
            l -- model output tensor of shape (B, 10*n_mix, H, W), where for each n_mix there are
                    3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
            x -- data tensor of shape (B, C, H, W) with values in model space [-1, 1]
        """
        # shapes
        B, C, H, W = x.shape
        n_mix = l.shape[1] // (1 + 3*C)

        # unpack params of mixture of logistics
        logits = l[:, :n_mix, :, :]                         # (B, n_mix, H, W)
        l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
        means, logscales, coeffs = l.split(n_mix, 1)        # (B, n_mix, C, H, W)
        logscales = logscales.clamp(min=-7)
        coeffs = coeffs.tanh()

        # adjust means of channels based on preceding subpixel (cf PixelCNN++ eq 3)
        x  = x.unsqueeze(1).expand_as(means)
        if C!=1:
            m1 = means[:, :, 0, :, :]
            m2 = means[:, :, 1, :, :] + coeffs[:, :, 0, :, :] * x[:, :, 0, :, :]
            m3 = means[:, :, 2, :, :] + coeffs[:, :, 1, :, :] * x[:, :, 0, :, :] + coeffs[:, :, 2, :, :] * x[:, :, 1, :, :]
            means = torch.stack([m1, m2, m3], 2)  # out (B, n_mix, C, H, W)

        # log prob components
        scales = torch.exp(-logscales)
        plus = scales * (x - means + 1/(2**self.n_bits-1))
        minus = scales * (x - means - 1/(2**self.n_bits-1))

        # partition the logistic pdf and cdf for x in [<-0.999, mid, >0.999]
        # 1. x<-0.999 ie edge case of 0 before scaling
        cdf_minus = torch.sigmoid(minus)
        log_one_minus_cdf_minus = - F.softplus(minus)
        # 2. x>0.999 ie edge case of 255 before scaling
        cdf_plus = torch.sigmoid(plus)
        log_cdf_plus = plus - F.softplus(plus)
        # 3. x in [-.999, .999] is log(cdf_plus - cdf_minus)

        # compute log probs:
        # 1. for x < -0.999, return log_cdf_plus
        # 2. for x > 0.999,  return log_one_minus_cdf_minus
        # 3. x otherwise,    return cdf_plus - cdf_minus
        log_probs = torch.where(x < -0.999, log_cdf_plus,
                                torch.where(x > 0.999, log_one_minus_cdf_minus,
                                            torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))))
        log_probs = log_probs.sum(2) + F.log_softmax(logits, 1) # log_probs sum over channels (cf eq 3), softmax over n_mix components (cf eq 1)

        # marginalize over n_mix components and return log likelihood over whole image
        return log_probs.logsumexp(1)  # out (B, H, W)

    def message(self, x):

        # TODO: Fix this after training rotated model
        if self.arch == 'pixelcnnpp':
            x_t = self.transform(x)
            # First dimension contains true probs and second contains probs of inverted values
            out = self.model(torch.cat([x_t, -1*x_t], 0), None)
            self.ll = self.discretized_mix_logistic_loss(out, torch.cat([x_t, -1*x_t], 0))
            self.prob = torch.exp(self.ll)

            b, h, w = self.ll.shape
            message = torch.zeros(1, 2, h, w).to(device)
            message[:,0,:,:] = torch.where(x==0, self.prob[0, ...], self.prob[1, ...])
            message[:,1,:,:] = torch.where(x==1, self.prob[0, ...], self.prob[1, ...])
            message /= torch.sum(message, 1, keepdim=True)

        elif self.arch == 'pixelcnn':

            # Set the inputs -- original and rotated orientation
            x_t = self.transform(x)
            x_t_rot = x_t.rot90(2, [2, 3])
            # Get the logits and convert to probabilities
            out_0 = self.convert_logits_to_prob(self.model_0(x_t, None))
            out_180 = self.convert_logits_to_prob(self.model_180(x_t_rot, None).rot90(2, [-2, -1]))
            # Multiply probabilities and normalize
            message = out_0 * out_180
            message /= torch.sum(message, 1, keepdim=True)

        else:
            message = None

        return message

class SourceCodeBP():

    def __init__(self,
                 H,
                 h=28,
                 w=28,
                 p=0.5, 
                 stay=0.9,
                 alpha=0.8,
                 doperate=0.04,
                 n_bits=2,
                 arch='pixelcnn'):

        # Store the parameters
        self.h = h
        self.w = w
        self.p = p
        self.stay = stay
        self.alpha = alpha
        self.doperate = doperate
        self.n_bits=n_bits
        self.arch = arch

        # Store the parity check matrix
        self.H = H
        self.K, self.N = self.H.shape

        # Setup the Ising dataset
        self.dataset = IsingDataset(args.data_path, phase='train')
        print("[Setup the sampler ...]")

        # Setup the source graph
        self.source = Source(arch=self.arch, image_dims=(1, self.h, self.w), n_bits=self.n_bits)
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
        self.M_from_code = None
        self.M_from_grid = None
        self.B = None

    def doping(self):

        indices = np.random.randint(self.N, size=int(self.N*self.doperate)+1)
        # Dope the very first sample also
        self.ps[0, 0], self.ps[0, 1] = (self.samp[0, 0] == 0).float(), (self.samp[0, 0] == 1).float()
        self.ps[indices, 0], self.ps[indices, 1] = (self.samp[indices, 0] == 0).float(), (self.samp[indices, 0] == 1).float()
        # Update the node potential after doping
        self.npot = self.ps.reshape(self.h, self.w, 2)

    def preprocess(self, x):
        # 1. convert data to float
        # 2. normalize to [0,1] given quantization
        # 3. shift to [-1,1]
        return x.float().div(2**self.n_bits - 1).mul(2).add(-1)

    def generate_sample(self):

        idx = np.random.randint(0, len(self.dataset))
        self.samp, _ = self.dataset[idx]
        self.samp = torch.FloatTensor(self.samp.reshape(-1, 1)).to(device)

        # test = self.samp.reshape(1, 1, self.h, self.w)
        # test = torch.randint(high=2, size=(1, 1, self.h, self.w), dtype=torch.float32).to(device)
        # test_t = self.source.transform(test)
        # prob = F.softmax(self.source.model_0(test_t, None).squeeze(2))
        # print(test_t[0, :, :4, :4])
        # print(prob[0, :, :4, :4])
        # exit(0)

    def encode(self):

        self.x = (self.H @ self.samp) % 2

    def decode_step(self):

        # Perform one step of code graph belief propagation
        self.code(self.ps, self.x, self.M_to_code)
        self.M_from_code = self.code.M_out
        # Reshape to send to grid
        self.M_to_grid = self.M_from_code.reshape(self.h, self.w, 2)
        print(self.M_to_grid)

        # Perform one step of source graph belief propagation
        # Extract the last channel of the code message
        belief = self.M_to_grid * self.npot
        belief /= torch.sum(belief, -1, keepdim=True)
        source_input = (belief[:,:,1].reshape(1, 1, self.h, self.w) > 0.5).float()
        self.M_from_grid = self.source.message(source_input)
        # Permute this output
        self.M_from_grid = self.M_from_grid.squeeze(0).permute(1, 2, 0)
        # Reshape to send to code
        self.M_to_code = self.M_from_grid.reshape(-1, 2)

    def decode(self, num_iter=1):

        # Set the initial beliefs to all nans
        B_old = torch.tensor(float('nan') * np.ones((self.h, self.w))).to(device)
        start = time.time()

        # Perform multiple iterations of belief propagation
        for i in range(num_iter):

            # Perform a step of message passing/decoding
            self.decode_step()

            # Calculate the belief
            if self.M_from_grid is None:
                self.B = self.M_to_grid * self.npot
            else:
                self.B = self.M_from_grid * self.M_to_grid * self.npot
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

    h = args.h
    w = args.w
    n_bits = args.n_bits

    # Load the LDPC matrix
    H = torch.FloatTensor(loadmat(args.ldpc_mat)['Hf']).to(device)

    # Intialize the source-code decoding graph
    source_code_bp = SourceCodeBP(H, h=h, w=w, arch=args.arch, n_bits=n_bits)

    # Either load a sample image or generate one using Gibb's sampling
    print("[Generating the sample ...]")
    source_code_bp.generate_sample()
    
    # Encode the sample using the LDPC matrix
    print("[Encoding the sample ...]")
    source_code_bp.encode()

    # Dope it to update our initial beliefs
    print("[Doping ...]")
    source_code_bp.doping()

    # Decode the code using belief propagation
    print("[Decoding ...]")
    source_code_bp.decode(num_iter=args.num_iter)

    # Visualize the decoded image
    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(source_code_bp.samp.cpu().numpy().reshape(28, 28))
    ax[0].set_title("Source Image")
    ax[1].imshow((source_code_bp.npot.cpu()[..., 1] > 0.5).float().numpy())
    ax[1].set_title("Doping samples")
    ax[2].imshow((source_code_bp.B.cpu()[..., 1] > 0.5).float().numpy())
    ax[2].set_title("Reconstructed Image")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_source_code_bp()
