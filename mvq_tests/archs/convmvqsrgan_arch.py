'''
This file defines the architecture for the MVQSRGAN.  
MVQSRGAN = Multiple Vector Quantizer Super-Resolution GAN

The network takes an input an image and downsamples the resolution
and quantizes the image pixels to a single-bit using multiple vector 
quantizers.  The quantized inputs are then fed to an SRGAN for image
reconstruction.
'''

import torch
import torch.nn as nn
from mvq_tests.archs.utils import BlockMaker
from mvq_tests.archs.vector_quantizer import VectorQuantizerEMA
from basicsr.models.archs.rrdbnet_arch import RRDBNet

from torchinfo import summary

class ConvMVQSRGAN(nn.Module):

    def __init__(self, 
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_block=16,
                 block_size=4,
                 num_embeddings=2, 
                 embedding_dim=16, 
                 commitment_cost=4, 
                 decay=0.99,
                 kernel_sizes=[3, 5, 7],
                 num_kernels=[3, 3, 3]):

        super(ConvMVQSRGAN, self).__init__()
        
        # Make sure that the number of vector quantizers
        # is the same as the num of kernels being used
        assert(num_in_ch == sum(num_kernels))

        self.block_size = block_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Number of vector quantizers
        self.num_vq = num_in_ch + 1  # Plus one since the original image is also quantized

        # Initialize the different kernels and the number
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels

        # Initialize the convolution kernels
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, 
                                              out_channels=num_kernels[i], 
                                              kernel_size=kernel_sizes[i], 
                                              padding=(kernel_sizes[i]-1)//2) for i in range(len(num_kernels))])

        # Initialize the block maker
        self.block_maker = BlockMaker(block_size=self.block_size)

        # Intialize the vector quantizers            
        self.vq = nn.ModuleList([VectorQuantizerEMA(num_embeddings=self.num_embeddings,
                                                    embedding_dim=self.embedding_dim,
                                                    commitment_cost=self.commitment_cost,
                                                    decay=self.decay) for _ in range(self.num_vq)])

        # Initialize the RRDBNet
        self.rrdbnet = RRDBNet(num_in_ch=self.num_vq,
                               num_out_ch=num_out_ch,
                               num_feat=num_feat,
                               num_block=num_block)

    def encode(self, x):

        # Save the input size for the future
        b, c, h, w = x.shape

        # Filter the image multiple times
        filtered_images = []
        for i in range(len(self.convs)):
            filtered_images.append(self.convs[i](x))
        # Prepend the original image 
        filtered_images = torch.cat([x] + filtered_images, dim=1)

        # Now vector quantization happens
        commitment_loss = 0
        encodings = []
        for i in range(self.num_vq):

            # Pass the input through the block maker to get the image
            # blocks of size b x (s**2) x h//s x w//s
            unfolded = self.block_maker(filtered_images[:, i, ...].unsqueeze(1))

            loss, _, _, encoding_indices = self.vq[i](unfolded)
            # Reshape the encoding indices back to b x c x h//s x w//s
            encoded = encoding_indices.reshape(b, c, h//self.block_size, w//self.block_size).type(torch.cuda.FloatTensor).to(x.device)

            commitment_loss += loss
            encodings.append(encoded)

        # Concatenate the encodings into a single tensor
        encodings = torch.cat(encodings, dim=1)

        return commitment_loss, encodings, filtered_images

    def decode(self, encodings):

        # Now pass through the RRDBNet
        out = self.rrdbnet(encodings)

        return out

    def forward(self, x):

        # Return multiple encodings of the image using vector quantizers
        commitment_loss, encodings, filtered_images = self.encode(x)

        # Use the encodings to get a reconstruction
        out = self.decode(encodings)

        return commitment_loss, encodings, out, filtered_images

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    mvqsrgan = ConvMVQSRGAN(num_in_ch=9).to(device)

    summary(mvqsrgan, (2, 1, 128, 128))



