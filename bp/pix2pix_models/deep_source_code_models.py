import torch
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np
import time

from .deep_source_code_base import BaseModel
from . import networks_projection as networks
from torch_parallel.code_bp_torch import CodeBP
from torch_parallel.grid_gibbs import GibbsSampler

class DeepSourceCode(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    """

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
                  
        # if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        #     self.netD = networks.define_D(netD='isinggan', gpu_ids=self.gpu_ids)

        # if self.isTrain:
        #     # define loss functions
        #     self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        #     self.criterionCE = torch.nn.CrossEntropyLoss()
        #     self.criterionL1 = torch.nn.L1Loss()
        #     # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #     self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #     self.optimizers.append(self.optimizer_G)
        #     self.optimizers.append(self.optimizer_D)
        #     # Setup the schedulers
        #     self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        # # Store the parameters
        # self.h = h
        # self.w = w
        # self.p = p
        # self.stay = stay
        # self.alpha = alpha
        # self.doperate = doperate

        # Store the parity check matrix
        self.H = torch.FloatTensor(loadmat(self.ldpc_mat)['H']).to(self.device)
        self.K, self.N = self.H.shape

        # Setup the Gibbs sampler
        self.sampler = GibbsSampler(self.h, self.w, self.p, self.stay)
        print("[Setup the sampler ...]")

        # Setup the source neural network
        self.source = networks.define_D(netD='isinggan', gpu_ids=self.gpu_ids)
        self.load(self.checkpoint)
        print("[Setup the source graph ...]")

        # Setup the code graph
        self.code = CodeBP(self.H).to(self.device)
        print("[Setup the code graph ...]")

        # Store a matrix for doping probabilities
        self.ps = torch.FloatTensor(np.tile(np.array([1-p, p]), (self.h*self.w, 1))).to(self.device)

        # Input image
        self.samp = None

        # Encoded image
        self.x = None

        # Initialize the messages
        self.M_to_code = None
        self.M_to_grid = None
        self.B = None

    # def set_input(self, input, encoding):
        
    #     # Input Gibb's sampled image
    #     b, c, h, w = input['sample'].shape
    #     self.real_B = input['sample'].to(self.device)
    #     # Get the encoding obtained after multiplying with LDPC matrix
    #     self.real_A = encoding.to(self.device)

    def doping(self):

        indices = np.random.randint(self.N, size=int(self.N*self.doperate)+1)
        self.ps[indices, 0], self.ps[indices, 1] = (self.samp[indices, 0] == 0).float(), (self.samp[indices, 0] == 1).float()
        # Update the node potential after doping
        self.source.npot.data = self.ps.reshape(self.h, self.w, 2)

    def generate_sample(self):

        self.sampler.sampler(1000)
        self.samp = torch.FloatTensor(self.sampler.samp.reshape(-1, 1)).to(self.device)

    def encode(self):

        self.x = (self.H @ self.samp) % 2

    def decode_step(self):

        # Perform one step of code graph belief propagation
        self.code(self.ps, self.x, self.M_to_code)
        self.M_from_code = self.code.M_out
        # Reshape to send to grid
        self.M_to_grid = self.M_from_code.reshape(self.h, self.w, 2)

        # Modify the message for input to the source network
        intermediate_B = self.M_to_grid * self.source.npot
        source_input = intermediate_B[..., 1].reshape(1, 1, self.h, self.w)

        # Perform one step of source graph belief propagation
        self.from_grid = self.source(source_input)
        # Reshape to send to code
        self.M_to_code = self.from_grid * torch.ones((self.N, 2)).to(self.device)

    def decode(self, num_iter=1):

        # Set the initial beliefs to all nans
        B_old = torch.tensor(float('nan') * np.ones((self.h, self.w))).to(self.device)
        start = time.time()

        # Perform multiple iterations of belief propagation
        for i in range(num_iter):

            # Perform a step of message passing/decoding
            self.decode_step()

            # Calculate the belief
            self.B = self.M_from_grid * self.M_to_grid * self.source.npot
            self.B /= torch.sum(self.B, -1).unsqueeze(-1)

            # Termination condition to end belief propagation
            if torch.sum(torch.abs(self.B[..., 1] - B_old)).item() < 0.5:
                break
            B_old = self.B[..., 1]

            # Compute the number of errors and print some information
            errs = torch.sum(torch.abs((self.B[..., 1] > 0.5).float() - self.samp.reshape(self.h, self.w))).item()
            print(f'Iteration {i}: {errs} errors')

        end = time.time()
        print(f'Total time taken for decoding is {end - start}s')

    # def forward(self):
    #     """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
    #     # Get the fake logits segmentation mask
    #     self.fake_B = self.netG(self.real_A)  # G(A)
    #     # Obtain an (approx binary) image from this
    #     self.fake_B_img = F.softmax(self.fake_B, dim=1)
    #     self.fake_B_img = 10000*(self.fake_B_img[:, 1, ...] - self.fake_B_img[:, 0, ...]).unsqueeze(1)
    #     self.fake_B_img = torch.sigmoid(self.fake_B_img)

    # def backward_D(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B_img -- use the approx binary image here
    #     pred_fake = self.netD(self.fake_B_img.detach(), self.real_A.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Real
    #     pred_real = self.netD(self.real_B, self.real_A)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     self.loss_D.backward()

    # def backward_G(self):
    #     """Calculate GAN and L1 loss for the generator"""
    #     # First, G(A) should fake the discriminator
    #     pred_fake = self.netD(self.fake_B_img, self.real_A)
    #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)
    #     # Second, G(A) = B -- use the segmentation map here
    #     self.loss_G_CE = self.criterionCE(self.fake_B, self.real_B.squeeze(1).detach().long().to(self.fake_B_img.device)) * self.opt.lambda_L1
    #     # Third: Apply the difference loss between compressed images -- use the approx binary image here
    #     b, c, w, h = self.fake_B_img.shape
    #     pred_projection = torch.sin(torch.matmul(self.ldpc_mat, self.fake_B_img.reshape(b, w*h, 1)) * np.pi / 2.0)
    #     true_projection = torch.sin(torch.matmul(self.ldpc_mat, self.real_B.reshape(b, w*h, 1)) * np.pi / 2.0)
    #     self.loss_G_diff = self.criterionL1(pred_projection, true_projection) * self.opt.lambda_diff
    #     # combine loss and calculate gradients
    #     self.loss_G = self.loss_G_GAN + self.loss_G_CE + self.loss_G_diff
    #     self.loss_G.backward()

    # def optimize_parameters(self):
    #     self.forward()                   # compute fake images: G(A)
    #     # update D
    #     self.set_requires_grad(self.netD, True)  # enable backprop for D
    #     self.optimizer_D.zero_grad()     # set D's gradients to zero
    #     self.backward_D()                # calculate gradients for D
    #     self.optimizer_D.step()          # update D's weights
    #     # update G
    #     self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
    #     self.optimizer_G.zero_grad()        # set G's gradients to zero
    #     self.backward_G()                   # calculate graidents for G
    #     self.optimizer_G.step()             # udpate G's weights

    # def save(self, checkpoint_file):

    #     # Store the parameters in the checkpoint file
    #     torch.save({'netG_state_dict': self.netG.state_dict(),
    #                 'netD_state_dict': self.netD.state_dict(),
    #                 'optimizer_G_state_dict': self.optimizer_G.state_dict(),
    #                 'optimizer_D_state_dict': self.optimizer_D.state_dict()
    #                 }, checkpoint_file)

    # def load(self, checkpoint_file):

    #     cp = torch.load(checkpoint_file)
    #     self.netG.load_state_dict(cp['netG_state_dict'])
    #     if self.isTrain:
    #         self.optimizer_G.load_state_dict(cp['optimizer_G_state_dict'])
    #         self.netD.load_state_dict(cp['netD_state_dict'])
    #         self.optimizer_D.load_state_dict(cp['optimizer_D_state_dict'])

    # def test(self):

    #     with torch.no_grad():
    #         self.forward()

    def load(self, checkpoint_file):

        cp = torch.load(checkpoint_file)
        self.netD.load_state_dict(cp['netD_state_dict'])
        self.netD.eval()