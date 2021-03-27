import torch
import torch.nn as nn

from ditherq_tests.utils.compression_utils import Thresholder
from ditherq_tests.pix2pix_models.networks_ditherq import ResnetGenerator

class MTGAN(nn.Module):

    def __init__(self,
                 input_nc=1,
                 output_nc=1,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks=9,
                 threshold_levels=[-0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 0.75]):

        super(MTGAN, self).__init__()

        self.threshold_levels = threshold_levels

        # Initialize the residual generator
        self.gen = ResnetGenerator(input_nc=len(self.threshold_levels), 
                                   output_nc=output_nc, 
                                   ngf=ngf, 
                                   norm_layer=norm_layer, 
                                   use_dropout=use_dropout, 
                                   n_blocks=9)

        # Initialize the thresholders
        self.thresholders = nn.ModuleList([Thresholder(threshold=threshold_levels[i]) for i in range(len(self.threshold_levels))])

    def forward(self, x):

        # Pass the image through the thresholders
        x_t = []
        for i in range(self.threshold_levels):
            x_t.append(self.thresholders[i](x))
        x_t = torch.cat(x_t, dim=1)

        # Pass through the generator
        out = self.gen(x_t)

        return out, x_t

