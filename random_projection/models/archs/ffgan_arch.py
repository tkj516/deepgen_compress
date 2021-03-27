import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self,
                 encoding_size=32**2,
                 output_shape=128**2):

        super(Generator, self).__init__()

        self.linear1 = nn.Linear(encoding_size, encoding_size*4)
        self.linear2 = nn.Linear(encoding_size*4, encoding_size*16)
        self.linear3 = nn.Linear(encoding_size*16, output_shape)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU()

        self.m = 1.0

    def forward(self, x):

        out = self.lrelu(self.linear1(x))
        out = self.lrelu(self.linear2(out))
        out = self.linear3(out)

        # Apply DBN
        # out = self.sigmoid(self.m*out) + (torch.heaviside(self.sigmoid(out) - 0.5, torch.tensor([0.0]).to(x.device)) - self.sigmoid(self.m*out)).detach()

        return self.sigmoid(out)

class Discriminator(nn.Module):

    def __init__(self, 
                 input_shape=128**2,
                 output_shape=1):

        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(input_shape, input_shape//4)
        self.linear2 = nn.Linear(input_shape//4, input_shape//16)
        self.linear3 = nn.Linear(input_shape//16, 256)
        self.linear4 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.sigmoid(self.linear4(out))

        return out





