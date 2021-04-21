import torch
import torch.nn as nn
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

from pixel_models.pixelcnnpp import *
from pixel_models.pixelcnn import *

parser = argparse.ArgumentParser(description='Belief propagation training arguments')
parser.add_argument('--ldpc_mat', type=str, default='H_28.mat', help="Path to LDPC matrix")
parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the code on")
parser.add_argument('--restore_file', type=str, default='/fs/data/tejasj/Masters_Thesis/pixel_models/results/pixelcnnpp/2021-04-03_02-38-57/checkpoint.pt', help="Directory with checkpoint")
parser.add_argument('--arch', type=str, default='pixelcnnpp', help="Type of architecture")
parser.add_argument('--num_iter', type=int, default=100, help="Number of bp iterations")
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

class Decoder(nn.Module):

    def __init__(self, 
                 H, 
                 arch='pixelcnnpp',
                 image_dims=(1, 28, 28),
                 n_channels=128,
                 n_res_layers=5,
                 n_logistic_mix=10,
                 n_bits=1,
                 n_out_conv_channels=1024,
                 kernel_size=5,
                 norm_layer=True,
                 n_cond_classes=None):

        super(Decoder, self).__init__()

        self.H = H
        self.n_bits = n_bits
        self.arch = arch
        self.c, self.h, self.w = image_dims

        # Define a parameter for the input image
        self.input = torch.randn(1, 100).to(device)

        self.massager = nn.Sequential(nn.Linear(100, 512),
                                      nn.Tanh(),
                                      nn.Linear(512, 100))
        self.massager.train()

        # Define the pixelcnn architecture that is being used
        self.arch = arch

        self.source = torch.hub.load("Lornatang/GAN-PyTorch", "mnist", pretrained=True, progress=True, verbose=False)
        self.source.eval()
        self.source = self.source.to(device)

        # Define a few loss functions
        self.L1loss = nn.L1Loss()
        self.cosine_similarity = nn.CosineSimilarity(dim=0)
        self.MSEloss = nn.MSELoss()
        self.softmarginloss = nn.SoftMarginLoss()

        # Define normal distribution
        self.normal = torch.distributions.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

        # Define lagrange multipliers
        # self.multipliers = torch.ones(1, self.H.shape[0]).to(device)

    def forward(self):

        # Normalize the input in the correct range for the source model
        self.massaged_input = self.massager(self.input)
        self.normalized_input = self.source(self.massaged_input)

    def smooth_modulus(self, x, y=2, epsilon=1e-4):

        theta = x.mul(np.pi).div(y)
        return 1 - (y * torch.atan((torch.cos(theta)*torch.sin(theta)) / (torch.sin(theta)**2 + epsilon**2))) / np.pi

    def calculate_loss(self, targets, doped_values, doped_indices):

        # Threshold the input
        input = (self.normalized_input + 1) / 2

        # Get the loss from the pixelcnn
        # logistic_loss = discretized_mix_logistic_loss(self.logits, thresholded_input, self.n_bits)

        # Get the encoding loss using the LDPC matrix
        encodings = self.H @ input.reshape(-1, 1)
        encodings = self.smooth_modulus(encodings)

        # Enforce that massaged input is normal
        nll = -1*torch.sum(self.normal.log_prob(self.massaged_input))

        # Apply similarity loss
        # similarity_loss = -1*self.cosine_similarity(encodings, targets)
        # similarity_loss = self.MSEloss(encodings, targets.detach())
        similarity_loss = self.softmarginloss(encodings, 2*targets.detach() - 1)

        # doping_loss = self.MSEloss(input.reshape(-1, 1)[doped_indices, 0], doped_values)
        doping_loss = self.softmarginloss(input.reshape(-1, 1)[doped_indices, 0], 2*doped_values - 1)

        # similarity_loss = torch.clamp(self.multipliers, min = 0) @ (encodings - targets.detach())

        return 100*similarity_loss + nll + 50*doping_loss

def test_source_code_decode():

    h = 28
    w = 28
    n_bits = 1

    # Load the LDPC matrix
    H = torch.FloatTensor(loadmat(args.ldpc_mat)['Hf']).to(device)

    # Intialize the decoder
    decoder = Decoder(H).to(device)

    # Setup an optimizer for the input image
    optimizer = torch.optim.Adam(params=list(decoder.massager.parameters()), lr=5e-5, betas=(0.9, 0.999))

    # Either load a sample image or generate one using Gibb's sampling
    print("[Generating the sample ...]")
    # Setup the transforms
    transform = T.Compose([T.ToTensor(),                                            # tensor in [0,1]
                           lambda x: x.mul(255).div(2**(8-n_bits)).floor()])    # lower bits
    target_transform = None

    # Setup the MNIST dataset
    dataset = MNIST('../../MNIST', train=True, transform=transform, target_transform=target_transform)
    
    idx = np.random.randint(0, len(dataset))
    samp, _ = dataset[idx]
    samp = torch.FloatTensor(samp.reshape(-1, 1)).to(device)
    
    # Encode the sample using the LDPC matrix
    print("[Encoding the sample ...]")
    targets = (H @ samp) % 2

    # Perform doping
    doped_indices = np.random.randint(h*w, size=int(h*w*0.04)+1)
    # Dope the very first sample also
    doped_values = samp[doped_indices, 0]

    # Decode the code using belief propagation
    print("[Decoding ...]")
    for i in range(args.num_iter):
        decoder()

        optimizer.zero_grad()
        s_loss = decoder.calculate_loss(targets, doped_values, doped_indices)
        loss = s_loss
        loss.backward()
        optimizer.step()

        print(f'Iteration {i}:- Similarity Loss: {s_loss.item()}')

    # Visualize the decoded image
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(samp.cpu().numpy().reshape(h, w))
    ax[0].set_title("Source Image")
    ax[1].imshow((decoder.normalized_input.reshape(h, w).detach().cpu() > 0).float().numpy())
    ax[1].set_title("Reconstructed Image")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_source_code_decode()
