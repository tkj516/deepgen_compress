import torch
import torch.nn.functional as F
from .base_model_projection import BaseModel
from . import networks_projection as networks
from scipy.io import loadmat
import numpy as np

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
                  
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(netG='isinggan', gpu_ids=self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(netD='isinggan', gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # Setup the schedulers
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        
        # Input Gibb's sampled image
        self.real_B = input['sample'].to(self.device)
        self.b, self.c, self.h, self.w = self.real_B.shape

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        # Get the fake logits segmentation mask
        # input = torch.rand(self.b, self.c*self.h//4*self.w//4).to(self.device)
        input = torch.randn(selfb, 100)
        self.fake_B = self.netG(input)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B_img -- use the approx binary image here
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B -- use the segmentation map here
        self.loss_G_BCE = self.criterionBCE(self.fake_B, self.real_B.detach().to(self.fake_B.device)) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_BCE
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def save(self, checkpoint_file):

        # Store the parameters in the checkpoint file
        torch.save({'netG_state_dict': self.netG.state_dict(),
                    'netD_state_dict': self.netD.state_dict(),
                    'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict()
                    }, checkpoint_file)

    def load(self, checkpoint_file):

        cp = torch.load(checkpoint_file)
        self.netG.load_state_dict(cp['netG_state_dict'])
        if self.isTrain:
            self.optimizer_G.load_state_dict(cp['optimizer_G_state_dict'])
            self.netD.load_state_dict(cp['netD_state_dict'])
            self.optimizer_D.load_state_dict(cp['optimizer_D_state_dict'])

    def test(self):

        with torch.no_grad():
            self.forward()

