import os
import csv
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

def csv_to_numpy(filepath, sep=',', dtype=np.uint8):
    with open(filepath, "r") as file:
        reader = csv.reader(file, delimiter=sep)
        dataset = np.array(list(reader)).astype(dtype)
        return dataset

def load_binary_dataset(root, name):
    directory = os.path.join(root, 'binary', name)
    data_train = csv_to_numpy(os.path.join(directory, name + '.train.data'))
    data_valid = csv_to_numpy(os.path.join(directory, name + '.valid.data'))
    data_test = csv_to_numpy(os.path.join(directory, name + '.test.data'))
    return data_train, data_valid, data_test

class UCIDNADataset(Dataset):
    """Dataset of binarized UCI DNA sequences"""

    def __init__(self, 
                root='/fs/data/tejasj/Masters_Thesis/spnflow/datasets',
                name='dna',
                phase='train'):

        # Choose the phase
        self.phase = phase

        # Get the train, val and test datasets
        data_train, data_valid, data_test = load_binary_dataset(root, name)

        # Choose the number of files
        if self.phase == 'train':
            self.files = data_train
        elif self.phase == 'val':
            self.files = data_valid
        else:
            self.files = data_test

    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, idx):

        sample = torch.FloatTensor(self.files[idx:idx+1])

        return sample, torch.tensor([0])