import torch
import torch.nn as nn

class BlockMaker(nn.Module):
    '''
    This module takes in an image tensor of shape b x 1 x h x w
    and a block size parameter s.  It then reshapes the tensor
    into shape b x (s**2) x h//s x w//s
    '''

    def __init__(self, block_size=4):

        super(BlockMaker, self).__init__()

        self.block_size = block_size
        self.fold_params = dict(kernel_size=self.block_size, stride=self.block_size)
        self.unfold = nn.Unfold(**self.fold_params)

    def forward(self, x):

        # Save the shape for the future
        b, _, h, w = x.shape

        # Take the input and unfold into the b x (s**2) x (h//s*w//s)
        unfolded = self.unfold(x)
        # Reshape it into size b x (s**2) x h//s x w//s
        unfolded = unfolded.reshape(b, -1, h//self.block_size, w//self.block_size)

        return unfolded

        



