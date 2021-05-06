import os
import torch
import numpy as np
from torch.utils.data import Dataset

class IsingDataset(Dataset):
    """Dataset of Gibbs sampled Ising images"""

    def __init__(self, 
                root_dir='/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/datasets/ising_28_05_09_75000',
                phase='train', 
                transform=None):
        # Root directory for the data
        self.root_dir = root_dir
        # Choose the phase
        self.phase = phase

        # Read the training files from the mat file
        self.files = sorted(os.listdir(self.root_dir))

        # Choose the number of files
        if self.phase == 'train':
            start_idx = 0
            end_idx = int(0.9*len(self.files))
        else:
            start_idx = int(0.9*len(self.files))
            end_idx = len(self.files)

        self.files = sorted(os.listdir(self.root_dir))[start_idx:end_idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = os.path.join(self.root_dir, self.files[idx])
        image = np.load(image)

        sample = torch.FloatTensor(image).unsqueeze(0)

        return sample, torch.tensor([0])