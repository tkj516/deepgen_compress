import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

# Create grid of Ising images
filepath = '/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_{{{rate:.2f}}}_200000'
savepath = 'ising_samples'
os.makedirs(savepath, exist_ok=True)
images = []
for rate in [0.5, 0.6, 0.7, 0.8, 0.9]:
    idx = np.random.randint(0, 200000)
    image_path = os.path.join(filepath.format(rate=rate), f'{str(idx).zfill(6)}.npy')
    im = np.load(image_path).repeat(4, axis=-1).repeat(4, axis=-2)
    im = Image.fromarray(255*im).convert('RGB')
    im.save(os.path.join(savepath, f'{rate}.png'))

# Create grid of MNIST images
savepath = 'mnist_samples'
os.makedirs(savepath, exist_ok=True)
dataset = MNIST('../../../MNIST', train=False)
images = []
for number in [2, 4, 7, 8]:
    num = -1
    while not number == num: 
        idx = np.random.randint(0, len(dataset))
        im, num = dataset[idx]
    im  = im.resize((112, 112), resample=Image.NEAREST)
    im.save(os.path.join(savepath, f'{number}.png'))

# Create grid of Fashion MNIST images
savepath = 'fashion_mnist_samples'
os.makedirs(savepath, exist_ok=True)
dataset = FashionMNIST('../../../FashionMNIST', train=False)
images = []
for number in range(10):
    num = -1
    while not number == num: 
        idx = np.random.randint(0, len(dataset))
        im, num = dataset[idx]
    im  = im.resize((112, 112), resample=Image.NEAREST)
    im.save(os.path.join(savepath, f'{number}.png'))

# Create grid of CIFAR10 images
savepath = 'cifar10_samples'
os.makedirs(savepath, exist_ok=True)
dataset = CIFAR10('../../../CIFAR10', train=False, transform=torchvision.transforms.Grayscale())
images = []
for number in range(10):
    num = -1
    while not number == num: 
        idx = np.random.randint(0, len(dataset))
        im, num = dataset[idx]
    im  = im.resize((128, 128), resample=Image.NEAREST)
    im.save(os.path.join(savepath, f'{number}.png'))