import torch
import torch.nn as nn
import numpy as np

class Thresholder(nn.Module):

    def __init__(self, threshold=0.5):
        
        super(Thresholder, self).__init__()
        self.threshold = threshold
        
    def forward(self, im):
        
        threshold = self.threshold * torch.ones(im.size()).to(im.device)
        out = (im > threshold).type(torch.cuda.FloatTensor).to(im.device)
        
        return out      
