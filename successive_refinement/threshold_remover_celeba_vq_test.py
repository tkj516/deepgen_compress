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

from vq_tests.pix2pix_models.pix2pix_model_vq import *

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='Training options for UNet')

checkpoint_string = './checkpoints_' + sys.argv[0].split('.')[0]

# basic parameters
parser.add_argument('--dataroot', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)', default='/fs/data/tejasj/image_decompression/CelebA/celeba/img_align_celeba')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default=checkpoint_string, help='models are saved here')
parser.add_argument('--isTrain', type=bool, default=False, help='true if training the model')
# noise parameters
parser.add_argument('--delta', type=float, default=1.0, help='width of uniform noise')
# model parameters
parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
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
parser.add_argument('--display_freq', type=int, default=64, help='frequency of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=640, help='frequency of showing training results on console')
# network saving and loading parameters
parser.add_argument('--checkpoint', type=str, default='.', help='checkpoint')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
# training parameters
parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu_ids) if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    """Image dataset for CelebA Dataset."""

    def __init__(self, 
                root_dir='/fs/data/tejasj/image_decompression/CelebA/celeba/img_align_celeba',
                phase='test', 
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(image)

        sample = self.transform(image)

        return {'sample': sample}

# Create a thresholding transform
class SingleCrossingThreshold(object):
    
    def __init__(self, threshold=0.5):
        
        self.threshold = threshold
        
    def __call__(self, im):
        
        im_n = im.numpy()
        im_n_c = np.copy(im_n)
        im_n[im_n_c >= self.threshold] = 1.0
        im_n[im_n_c < self.threshold] = 0.0
        
        out = torch.Tensor(im_n)
        
        return out      

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
    # Set up testing parameters
    batch_size = args.batch_size
    dataroot = args.dataroot

    total_iters = 0

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

    # Load the checkpoint
    model.load(args.checkpoint)

    for i, sample in enumerate(dataloader):

        total_iters += batch_size
        
        # Perform optimization
        model.set_input(sample)
        model.test()

        # Store various quanitites of interest
        output = model.fake_B
        quantized = model.real_A       

        if total_iters % args.display_freq == 0:
            detached_out = output.detach().cpu()
            detached_quantized = quantized.detach().cpu()
            visuals = OrderedDict()

            # Print the first image in each batch
            j = 0
            
            visuals['quantized' + str(i*batch_size + j)] = detached_quantized[j, ...]
            visuals['original' + str(i*batch_size + j)] = sample['sample'][j, ...]
            visuals['output_recon_image' + str(i*batch_size + j)] = detached_out[j, ...]

            writer.add_image(f'{i}/quantized', visuals['quantized' + str(i*batch_size + j)], i)
            writer.add_image(f'{i}/original', visuals['original' + str(i*batch_size + j)], i)
            writer.add_image(f'{i}/output_recon_image', visuals['output_recon_image' + str(i*batch_size + j)], i)

if __name__ == "__main__":
    main()
