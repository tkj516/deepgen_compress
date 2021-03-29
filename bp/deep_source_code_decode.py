'''
In this version we will use multiple vector quantizers to encode 
different aspects about the image.
'''

import time
import argparse
import sys
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from datetime import datetime
from PIL import Image
from skimage import io
from skimage.transform import resize
from collections import OrderedDict
from tensorboardX import SummaryWriter
from scipy.io import loadmat
import matplotlib.pyplot as plt

from pix2pix_models.deep_source_code_models import *

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='Training Options for Ising Model Decoding')

checkpoint_string = './checkpoints_' + sys.argv[0].split('.')[0]

# basic parameters
parser.add_argument('--dataroot', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)', default='/fs/data/tejasj/mrf/mrf_128')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default=checkpoint_string, help='models are saved here')
parser.add_argument('--isTrain', type=bool, default=True, help='true if training the model')
# model parameters
parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
parser.add_argument('--gan_arch', type=str, default='isinggan', help='choose the type of decoder GAN architecture')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
# dataset parameters
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--crop_size', type=int, default=128, help='then crop to this size')
parser.add_argument('--display_winsize', type=int, default=128, help='display window size for both visdom and HTML')
# additional parameters
parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
# visdom and HTML visualization parameters
parser.add_argument('--display_freq', type=int, default=5120, help='frequency of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=640, help='frequency of showing training results on console')
# network saving and loading parameters
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--continue_checkpoint', type=str, default='.', help='continue checkpoint')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
# training parameters
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
parser.add_argument('--lambda_diff', type=float, default=10.0, help='weight for L1 loss')
# More parameters
parser.add_argument('--h', type=int, default=128, help="Height of image")
parser.add_argument('--w', type=int, default=128, help="Width of image")
parser.add_argument('--p', type=float, default=0.5, help="Node probabilities")
parser.add_argument('--stay', type=float, default=0.8, help="Stay probabilities")
parser.add_argument('--ldpc_mat', type=str, default='H.mat', help="Path to LDPC matrix")
parser.add_argument('--load_image', action='store_true', help="Load an image for testing")
parser.add_argument('--image', type=str, default='S_128.mat', help="Path to loadable image")
parser.add_argument('--doperate', type=float, default=0.04, help="Doping rate for model")
parser.add_argument('--checkpoint', type=str, default='./12-03-2021_01_45_26/checkpoint.pth', help="Pre-trained model checkpoint")

args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu_ids) if torch.cuda.is_available() else "cpu")

class MRFDataset(Dataset):
    """Dataset of Gibbs sampled MRF images"""

    def __init__(self, 
                root_dir='/fs/data/tejasj/mrf/mrf_128',
                phase='train', 
                transform=None):
        # Root directory for the data
        self.root_dir = root_dir
        # Choose the phase
        self.phase = phase

        # Choose the number of files
        if self.phase == 'train':
            start_idx = 0
            end_idx = 25088
        else:
            start_idx = 25088
            end_idx = 25600

        # Read the training files from the mat file
        self.files = sorted(os.listdir(self.root_dir))[start_idx:end_idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = os.path.join(self.root_dir, self.files[idx])
        image = loadmat(image)['d_Sb']

        sample = torch.FloatTensor(image).unsqueeze(0)

        return {'sample': sample}

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

writer = SummaryWriter('runs/' + sys.argv[0].split('.')[0] + '/' + dt_string)

def test_source_code_bp():

    # Intialize the source-code decoding network
    source_code_bp = Pix2PixModel(args)

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
    source_code_bp.decode(num_iter=100)

    # Visualize the decoded image
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(source_code_bp.samp.cpu().numpy().reshape(h, w))
    ax[0].set_title("Source Image")
    ax[1].imshow(np.argmax(source_code_bp.B.detach().cpu().numpy(), axis=-1))
    ax[1].set_title("Reconstructed Image")
    plt.tight_layout()
    plt.show()

def main():
    # Set up training parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    dataroot = args.dataroot
    checkpoint_path = args.checkpoints_dir

    # Get the total number of training iterations
    total_iters = 0 
    start_epoch = 0
    best_loss = 1000000000

    # Get the model
    model = Pix2PixModel(args)

    # Get dataset and dataloader
    dataset = MRFDataset(root_dir=dataroot)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4)

    # Before the training starts setup the checkpoint folder
    checkpoint_directory = os.path.join(checkpoint_path, dt_string)
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    # Load checkpoint if continuing to train
    if args.continue_train == True:
        if args.continue_checkpoint is None:
            print("Please provide valid checkpoint")
            exit(0)
        model.load(args.continue_checkpoint)

    print("[Training begins]")

    for epoch in range(start_epoch, n_epochs):

        epoch_iter = 0
        epoch_loss_G = 0
        epoch_loss_D = 0

        for i, sample in enumerate(dataloader):

            epoch_iter += batch_size
            total_iters += batch_size

            # Project the image
            b, c, h, w = sample['sample'].shape
            encoding = torch.matmul(model.ldpc_mat, sample['sample'].reshape(b, w*h, c).to(model.device)).squeeze(-1)
            encoding = encoding % 2

            # Perform optimization
            model.set_input(sample, encoding)
            model.optimize_parameters()

            # Get the reconstruction from the model
            output = model.fake_B

            # Store the losses
            loss_G = model.loss_G
            loss_D = model.loss_D
            # Calculate the epoch losses
            epoch_loss_G += loss_G.item() * batch_size
            epoch_loss_D += loss_D.item() * batch_size

            # Create a nice plot to display the loss convergence
            if epoch == 0:
                writer.add_scalar(f'data/{epoch}/loss_G', loss_G, i)
                writer.add_scalar(f'data/{epoch}/loss_D', loss_D, i)
            
            # Print some stats
            if epoch_iter % args.print_freq == 0:
                print(f'Epoch: {epoch}, Iteration: {i}:- Total Loss: {loss_G + loss_D}, G Loss: {loss_G}, D Loss: {loss_D}')

            # Populate tensorboard with results
            if epoch_iter % args.display_freq == 0:
                detached_out = torch.argmax(output.detach().cpu(), dim=1, keepdim=True)
                detached_encoding = encoding.detach().cpu()
                output_encoding = (torch.matmul(model.ldpc_mat.detach().cpu(), detached_out.reshape(b, w*h, c).float()).squeeze(-1)) % 2
                visuals = OrderedDict()

                # Print the first image in each batch
                j = 0
                
                visuals['encoded ' + str(i*batch_size + j)] = detached_encoding[j, ...].reshape(1, h//4, w//4)
                visuals['original ' + str(i*batch_size + j)] = sample['sample'][j, ...].reshape(1, h, w)
                visuals['output_recon_image ' + str(i*batch_size + j)] = detached_out[j, ...].reshape(1, h, w)
                visuals['output_encoded ' + str(i*batch_size + j)] = output_encoding[j, ...].reshape(1, h//4, w//4)

                writer.add_image(f'{epoch}/{i}/encoded', visuals['encoded ' + str(i*batch_size + j)], i)
                writer.add_image(f'{epoch}/{i}/original', visuals['original ' + str(i*batch_size + j)], i)
                writer.add_image(f'{epoch}/{i}/output_recon_image', visuals['output_recon_image ' + str(i*batch_size + j)], i)
                writer.add_image(f'{epoch}/{i}/output_encoded', visuals['output_encoded ' + str(i*batch_size + j)], i)

        # Update the overall loss plot every epoch
        writer.add_scalar('data/loss_G', epoch_loss_G/len(dataset), epoch)
        writer.add_scalar('data/loss_D', epoch_loss_D/len(dataset), epoch)
        
        # Update the checkpoint if generator loss has decreased
        checkpoint_file = os.path.join(checkpoint_directory, 'checkpoint.pth')
        model.save(checkpoint_file)
        if epoch_loss_G < best_loss:
            checkpoint_file = os.path.join(checkpoint_directory, 'best_checkpoint.pth')
            model.save(checkpoint_file)
            best_loss = epoch_loss_G

        # Update the learning rate
        model.update_learning_rate()

if __name__ == "__main__":
    test_source_code_bp()
