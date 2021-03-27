'''
This file defines the architecture for the VQSRGAN.  
VQSRGAN = Vector Quantized Super-Resolution GAN

The network takes an input an image and downsamples the resolution
and quantizes the image pixels to a single-bit using a vector 
quantizer.  The quantized input is then fed to an SRGAN for image
reconstruction.
'''

import torch
import torch.nn as nn
from vq_tests.utils import BlockMaker
from vq_tests.vector_quantizer import VectorQuantizerEMA
from basicsr.models.archs.rrdbnet_arch import RRDBNet

from torchinfo import summary

class VQSRGAN(nn.Module):

    def __init__(self, 
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_block=16,
                 block_size=4,
                 num_embeddings=2, 
                 embedding_dim=16, 
                 commitment_cost=4, 
                 decay=0.99):

        super(VQSRGAN, self).__init__()

        self.block_size = block_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Initialize the block maker
        self.block_maker = BlockMaker(block_size=self.block_size)

        # Intialize the vector quantizer
        self.vq = VectorQuantizerEMA(num_embeddings=self.num_embeddings,
                                     embedding_dim=self.embedding_dim,
                                     commitment_cost=self.commitment_cost,
                                     decay=self.decay)

        # Initialize the RRDBNet
        self.edsr = RRDBNet(num_in_ch=num_in_ch,
                            num_out_ch=num_out_ch,
                            num_feat=num_feat,
                            num_block=num_block)

    def forward(self, x):

        # Save the input size for the future
        b, c, h, w = x.shape

        # Pass the input through the block maker to get the image
        # blocks of size b x (s**2) x h//s x w//s
        unfolded = self.block_maker(x)

        # Now vector quantization happens
        commitment_loss, _, _, encoding_indices = self.vq(unfolded)

        # Reshape the encoding indices back to b x c x h//s x w//s
        encoded = encoding_indices.reshape(b, c, h//self.block_size, w//self.block_size).type(torch.cuda.FloatTensor).to(x.device)

        # Now pass through the RRDBNet
        out = self.edsr(encoded)

        return commitment_loss, encoded, out

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vqsrgan = VQSRGAN().to(device)

    summary(vqsrgan, (2, 1, 128, 128))



