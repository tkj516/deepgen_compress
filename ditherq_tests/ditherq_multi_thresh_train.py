# In this version I use a PatchGAN to train on thresholded images
# A batch size larger than 1 can be specified and it has the capability
# to load and store checkpoints

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

from ditherq_tests.pix2pix_models.pix2pix_model_ditherq import *

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='Training options for UNet')

checkpoint_string = './checkpoints_' + sys.argv[0].split('.')[0]

# basic parameters
parser.add_argument('--dataroot', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)', default='/fs/data/tejasj/image_decompression/CelebA/celeba/img_align_celeba')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default=checkpoint_string, help='models are saved here')
parser.add_argument('--isTrain', type=bool, default=True, help='true if training the model')
# noise parameters
parser.add_argument('--delta', type=float, default=0.75, help='width of uniform noise')
# model parameters
parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
parser.add_argument('--input_nc', type=int, default=7, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
parser.add_argument('--netG', type=str, default='mtgan', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--dither', action='store_true', help='Perform dithering')
parser.add_argument('--thresholds', type=str, default='-0.25,-0.10,0.0,0.10,0.25,0.50,0.75', help='threshold levels to use for training')
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
parser.add_argument('--display_freq', type=int, default=1280, help='frequency of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=640, help='frequency of showing training results on console')
# network saving and loading parameters
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--continue_checkpoint', type=str, default='.', help='continue checkpoint')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
# training parameters
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu_ids) if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    """Image dataset for CelebA Dataset."""

    def __init__(self, 
                root_dir='/fs/data/tejasj/image_decompression/CelebA/celeba/img_align_celeba',
                phase='train', 
                transform=None):
        # Root directory for the data
        self.root_dir = root_dir
        # The transform that will applied to the data
        self.transform = transform
        # Choose the phase
        self.phase = phase

        # Choose the number of files
        if self.phase == 'train':
            start_idx = 0
            end_idx = 25600
        else:
            start_idx = 25600
            end_idx = 30000

        # Read the training files from the mat file
        self.files = sorted(os.listdir(self.root_dir))[start_idx:end_idx]

        # Generate uniform random noise of size 1 x 128 x 128 on the interval [-delta, delta]
        self.noise = args.delta * (2 * torch.rand(1, 128, 128) - 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(image)

        sample = self.transform(image)
        # Add the noise to it
        noisy_sample =  sample + self.noise

        return {'noisy_sample': noisy_sample, 'sample': sample}

def image_transform():
    normalize = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),                                   
                ])
    return normalize


now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

writer = SummaryWriter('runs/' + sys.argv[0].split('.')[0] + '/' + dt_string)

def main():
    # Set up training parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    dataroot = args.dataroot
    checkpoint_path = args.checkpoints_dir

    total_iters = 0 # the total number of training iterations
    start_epoch = 0

    # Get the model
    model = Pix2PixModel(args)

    # Get dataset and dataloader
    imagetransform = image_transform()
    dataset = ImageDataset(dataroot, 
                           transform=imagetransform)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4)

    # Before the training starts setup the checkpoint folder
    checkpoint_directory = os.path.join(checkpoint_path, dt_string)
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    # Load checkpoint
    if args.continue_train == True:
        if args.continue_checkpoint is None:
            print("Please provide valid checkpoint")
            exit(0)
        model.load(args.continue_checkpoint)

    # Save the noise in the checkpoint directory
    torch.save(dataset.noise, os.path.join(checkpoint_directory, 'noise.pth'))

    for epoch in range(start_epoch, n_epochs):

        epoch_iter = 0
        epoch_loss_G = 0
        epoch_loss_D = 0
        best_loss = 1000000

        for i, sample in enumerate(dataloader):

            epoch_iter += batch_size
            total_iters += batch_size
            
            # Perform optimization
            model.set_input(sample)
            model.optimize_parameters()

            # Store various quanitites of interest
            output = model.fake_B
            dithered = model.real_A
            loss_G = model.loss_G
            loss_D = model.loss_D

            # Calculate the epoch losses
            epoch_loss_G += loss_G.item() * batch_size
            epoch_loss_D += loss_D.item() * batch_size

            # Create a nice plot to display the loss convergence
            if epoch == 0:
                writer.add_scalar(f'data/{epoch}/loss_G', loss_G, i)
                writer.add_scalar(f'data/{epoch}/loss_D', loss_D, i)
            
            if epoch_iter % args.print_freq == 0:
                print(f'Epoch: {epoch}, Iteration: {i}:- Total Loss: {loss_G + loss_D}, G Loss: {loss_G}, D Loss: {loss_D}')

            if total_iters % args.display_freq == 0:
                detached_out = output.detach().cpu()
                dithered_out = dithered.detach().cpu()
                visuals = OrderedDict()

                # Print the first image in each batch
                j = 0
                
                visuals['noisy' + str(i*batch_size + j)] = sample['noisy_sample'][j, ...]
                visuals['original' + str(i*batch_size + j)] = sample['sample'][j, ...]
                visuals['output_recon_image' + str(i*batch_size + j)] = detached_out[j, ...]
                for k in range(dithered_out.shape[1]):
                    visuals[f'dithered ({model.thresholds[k]})' + str(i*batch_size + j)] = dithered_out[j, k, ...].unsqueeze(0)


                writer.add_image(f'{epoch}/{i}/noisy', visuals['noisy' + str(i*batch_size + j)], i)
                writer.add_image(f'{epoch}/{i}/original', visuals['original' + str(i*batch_size + j)], i)
                writer.add_image(f'{epoch}/{i}/output_recon_image', visuals['output_recon_image' + str(i*batch_size + j)], i)
                for k in range(dithered_out.shape[1]):
                    writer.add_image(f'{epoch}/{i}/dithered {model.thresholds[k]}', visuals[f'dithered ({model.thresholds[k]})' + str(i*batch_size + j)], i)

        # Update the overall loss plot every epoch
        writer.add_scalar('data/loss_G', epoch_loss_G/len(dataset), epoch)
        writer.add_scalar('data/loss_D', epoch_loss_D/len(dataset), epoch)
        
        # Update the checkpoint if generator loss has decreased
        if epoch_loss_G/len(dataset) < best_loss:
            checkpoint_file = os.path.join(checkpoint_directory, 'checkpoint.pth')
            model.save(checkpoint_file)
            best_loss = epoch_loss_G/len(dataset)

        # Update the learning rate
        model.update_learning_rate()

if __name__ == "__main__":
    main()
