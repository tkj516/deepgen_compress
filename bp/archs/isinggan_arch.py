import torch
import torch.nn as nn


class IsingGenerator(nn.Module):

    def __init__(self,
                 encoding_shape=7**2,
                 output_dim=128,
                 output_nc=1):

        super(IsingGenerator, self).__init__()

        # Store some parameters for future use
        self.output_dim = output_dim
        self.output_nc = output_nc
        
        # Massage the encoding into noise that the Generator can use
        self.massager = nn.Sequential(nn.Linear(encoding_shape, encoding_shape),
                                      nn.BatchNorm1d(encoding_shape),
                                      nn.LeakyReLU())

        # Outputs a segmentation mask
        self.body = nn.Sequential(nn.ConvTranspose2d(1, 128, 5, stride=1, padding=2),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(),
                                  nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(),
                                  nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),
                                  )

    def forward(self, encoding):

        out = self.massager(encoding)
        
        b, _ = out.shape
        out = out.reshape(b, self.output_nc, self.output_dim//4, self.output_dim//4)

        # Output is like a two channel segmentation mask before softmax
        out = self.body(out)

        return out

class IsingDiscriminator(nn.Module):

    def __init__(self):

        super(IsingDiscriminator, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(1, 16, 5, stride=2, padding=2),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(0.3),
                                    nn.Conv2d(16, 32, 5, stride=2, padding=2),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.3),
                                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                    nn.LeakyReLU(),
                                    nn.Flatten(),
                                    nn.Linear(7*7*32, 1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 1))

    def forward(self, x):

        out = self.model(x)
        
        return out

