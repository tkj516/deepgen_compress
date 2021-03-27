import torch
import torch.nn as nn
import sys

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

class TensorShifter(nn.Module):
    '''
    This module takes in a tensor and shifts it up and down in the 
    spatial dimension by a specified amount and pads the unknown values
    with zeros
    '''

    def __init__(self, shift=(0, 0)):

        super(TensorShifter, self).__init__()

        self.shift = shift

    def forward(self, x):

        shifted_tensor = torch.zeros(x.size()).type(type(x)).to(x.device)
        shifted_tensor[..., :x.shape[-2]+self.shift[0], :x.shape[-1]+self.shift[1]] = x[..., -1*self.shift[0]:, -1*self.shift[1]:]

        return shifted_tensor

if __name__ == "__main__":

    print("Test the Tensor Shifter...")

    x = torch.rand([5, 5])
    shifter = TensorShifter(shift=(int(sys.argv[1]), int(sys.argv[2])))
    y = shifter(x)

    print(f"Before...{x}")
    print(f"After...{y}")





