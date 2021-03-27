import torch
import numpy as np
from torchinfo import summary 
import argpase

# Import stylegan2
from basicsr.models.archs.stylegan2_arch import *

parser = argparse.ArgumentParser(description='Summary options for StyleGAN2')
parser.add_argument('--gpu_id', type=int, default=1, help='Device to run the script on')
args = parser.parse_args()

if __name__ == "__main__":

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    net = StyleGAN2Generator(out_size=128)