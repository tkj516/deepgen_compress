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
        self.input = nn.Parameter(torch.zeros(1, 100).to(device))

        # Define the pixelcnn architecture that is being used
        self.arch = arch

        # Load the model
        # if self.arch == 'pixelcnnpp':
        #     self.source = PixelCNNpp(image_dims, n_channels, n_res_layers, n_logistic_mix,
        #                                 n_cond_classes).to(device)
        # elif self.arch == 'pixelcnn':
        #     n_res_layers = 12
        #     self.source = MyDataParallel(PixelCNN(image_dims, n_bits, n_channels, n_out_conv_channels, kernel_size,
        #                                 n_res_layers, n_cond_classes, norm_layer)).to(device)
        # else:
        #     pass

        # # Restore the checkpoint and set to evaluation mode
        # source_checkpoint = torch.load(args.restore_file, map_location=device)
        # self.source.load_state_dict(source_checkpoint['state_dict'])
        # self.source.eval()

        self.source = torch.hub.load("Lornatang/GAN-PyTorch", "mnist", pretrained=True, progress=True, verbose=False)
        self.source.eval()
        self.source = self.source.to(device)

        # Define a few loss functions
        self.L1loss = nn.L1Loss()
        self.cosine_similarity = nn.CosineSimilarity(dim=0)

    def forward(self):

        # Normalize the input in the correct range for the source model
        self.normalized_input = self.source(torch.sigmoid(self.input))

    def calculate_loss(self, targets):

        # Threshold the input
        thresholded_input = self.normalized_input

        # Get the loss from the pixelcnn
        # logistic_loss = discretized_mix_logistic_loss(self.logits, thresholded_input, self.n_bits)

        # Get the encoding loss using the LDPC matrix
        encodings = self.H @ ((thresholded_input + 1) / 2).reshape(-1, 1)
        encodings = (torch.cos(np.pi*encodings) + 1) / 2

        # Apply similarity loss
        similarity_loss = -1*self.cosine_similarity(encodings, targets)

        return similarity_loss

def test_source_code_decode():

    h = 28
    w = 28
    n_bits = 1

    # Load the LDPC matrix
    H = torch.FloatTensor(loadmat(args.ldpc_mat)['Hf']).to(device)

    # Intialize the decoder
    decoder = Decoder(H)

    # Setup an optimizer for the input image
    optimizer = torch.optim.Adam(params=[decoder.input], lr=1e-3, betas=(0.5, 0.999))

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

    # Decode the code using belief propagation
    print("[Decoding ...]")
    for i in range(args.num_iter):
        decoder()

        optimizer.zero_grad()
        s_loss = decoder.calculate_loss(targets)
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
