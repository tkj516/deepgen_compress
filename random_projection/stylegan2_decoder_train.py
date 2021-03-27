import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
import sys
from os import path as osp
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image
from skimage import io
from skimage.transform import resize
from collections import OrderedDict
from tensorboardX import SummaryWriter
from scipy.io import loadmat

from random_projection.models import create_model
from basicsr.utils.dist_util import get_dist_info, init_dist
from random_projection.utils.options import dict2str, parse
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)


def parse_options(checkpoint_directory, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', 
                        type=str, 
                        default='/fs/data/tejasj/BasicSR/random_projection/options/train_StyleGAN2_MRF_128.yml', 
                        help='Path to option YAML file.')
    parser.add_argument('--launcher', 
                        choices=['none', 'pytorch', 'slurm'], 
                        default='none', 
                        help='job launcher')
    parser.add_argument('--local_rank', 
                        type=int, 
                        default=0)
    args = parser.parse_args()
    opt = parse(args.opt, checkpoint_directory, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    return opt

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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(image)

        sample = self.transform(image)

        return {'sample': sample}

def image_transform():
    normalize = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),                                   
                ])
    return normalize

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

def main():

    checkpoint_path = './checkpoints_' + sys.argv[0].split('.')[0]
    # Before the training starts setup the checkpoint folder
    checkpoint_directory = os.path.join(checkpoint_path, dt_string)
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    # Parse options, set distributed setting, set ramdom seed
    opt = parse_options(checkpoint_directory, is_train=True)

    # Get some training variables
    batch_size = opt['datasets']['train']['batch_size_per_gpu']
    n_epochs = opt['train']['n_epochs']
    total_iters = 0
    
    # Get some visualization variables
    print_freq = batch_size * 20 
    display_freq = batch_size * 100

    # Load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # Get the MRF dataset
    dataset = MRFDataset()
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=opt['datasets']['train']['num_worker_per_gpu'])
    
    print(f"[Starting slope for slope annealing is {model.net_g.m}]")
    print("[Training begins]")

    for epoch in range(start_epoch, n_epochs):

        epoch_iter = 0
        epoch_loss_G = 0
        epoch_loss_D = 0
        best_loss = 1000000

        for i, sample in enumerate(dataloader):

            current_iter += 1

            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # Project the image
            b, c, h, w = sample['sample'].shape
            encoding = torch.matmul(model.ldpc_mat, sample['sample'].reshape(b, w*h, c).to(model.device)).squeeze(-1)
            encoding = encoding % 2
            
            # Perform the training
            model.feed_data(sample)
            model.optimize_parameters(current_iter, encoder_output=encoding)

            # Store various quanitites of interest
            output = model.fake_img
            loss_G = model.log_dict['l_g'] + 100*model.log_dict['l_g_l1']
            loss_D = model.log_dict['l_d']

            # Calculate the epoch losses
            epoch_loss_G += loss_G * batch_size
            epoch_loss_D += loss_D * batch_size

            # Create a nice plot to display the loss convergence
            if epoch == 0:
                writer.add_scalar(f'data/{epoch}/loss_G', loss_G, i)
                writer.add_scalar(f'data/{epoch}/loss_D', loss_D, i)
            
            if epoch_iter % print_freq == 0:
                print(f'Epoch: {epoch}, Iteration: {i}:- Total Loss: {loss_G + loss_D}, G Loss: {loss_G}, D Loss: {loss_D}')

            if total_iters % display_freq == 0:
                detached_out = output.detach().cpu()
                visuals = OrderedDict()

                # Print the first image in each batch
                j = 0
                
                visuals['original' + str(i*batch_size + j)] = sample['sample'][j, ...]
                visuals['output_recon_image' + str(i*batch_size + j)] = detached_out[j, ...]

                writer.add_image(f'{epoch}/{i}/original', visuals['original' + str(i*batch_size + j)], i)
                writer.add_image(f'{epoch}/{i}/output_recon_image', visuals['output_recon_image' + str(i*batch_size + j)], i)

            epoch_iter += batch_size
            total_iters += batch_size

        # Update the overall loss plot every epoch
        writer.add_scalar('data/loss_G', epoch_loss_G/len(dataset), epoch)
        writer.add_scalar('data/loss_D', epoch_loss_D/len(dataset), epoch)
        
        # Update the checkpoint if generator loss has decreased
        if epoch_loss_G/len(dataset) < best_loss:
            # save models and training states
            model.save(epoch, 0)
            best_loss = epoch_loss_G/len(dataset)

        # At the end of every epoch update the slop m for annealing
        model.net_g.m *= 1.1


if __name__ == '__main__':
    main()
