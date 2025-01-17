import torch
from .base_model_vq import BaseModel
from . import networks_vq as networks


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
        self.generate = networks.define_G(opt.input_nc, opt.output_nc, netG='vqsrgan', init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.refine = networks.define_G(opt.input_nc, opt.output_nc, netG='unet_128', init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        # define relativistic discriminator
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc, netD='relativistic', init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_generate = torch.optim.Adam(self.generate.parameters(), lr=opt.lr_generate, betas=(opt.beta1, 0.999))
            self.optimizer_refine = torch.optim.Adam(self.refine.parameters(), lr=opt.lr_refine, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_generate)
            self.optimizers.append(self.optimizer_refine)
            self.optimizers.append(self.optimizer_D)
            # Setup the scheduler
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """

        self.gt = input['sample'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # Note that in the vector quantized case the real_A is generated within the network
        # It is the encoded image

        self.commitment_loss, self.real_A, self.fake_B = self.generate(self.gt)  # G(A)
        self.refined_B = self.refine(self.fake_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        #real
        fake_d_pred = self.netD(self.fake_B).detach()
        real_d_pred = self.netD(self.gt)
        l_d_real = self.criterionGAN(
            real_d_pred - torch.mean(fake_d_pred), True
        ) * 0.5
        l_d_real.backward()
        #fake
        fake_d_pred = self.netD(self.fake_B.detach())
        l_d_fake = self.criterionGAN(
            fake_d_pred - torch.mean(real_d_pred.detach()), False
        ) * 0.5
        l_d_fake.backward()
        self.loss_D = l_d_real + l_d_fake

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        real_d_pred = self.netD(self.gt).detach()
        fake_g_pred = self.netD(self.fake_B)
        l_g_real = self.criterionGAN(
            real_d_pred - torch.mean(fake_g_pred), False
        )
        l_g_fake = self.criterionGAN(
            fake_g_pred - torch.mean(real_d_pred), True
        )
        self.loss_G_GAN = (l_g_real + l_g_fake) / 2
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.gt) * self.opt.lambda_L1
        # Add a refinement loss
        self.refinement_loss = self.criterionL1(self.refined_B, self.gt) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.commitment_loss + self.refinement_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_generate.zero_grad()        # set G's gradients to zero
        self.optimizer_refine.zero_grad()
        self.backward_G()                   # calculate graidents for G
        self.optimizer_generate.step()             # udpate G's weights
        self.optimizer_refine.step()
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

    def save(self, checkpoint_file):

        # Store the parameters in the checkpoint file
        torch.save({'generate_state_dict': self.generate.state_dict(),
                    'refine_state_dict': self.refine.state_dict(),
                    'netD_state_dict': self.netD.state_dict(),
                    'optimizer_generate_state_dict': self.optimizer_generate.state_dict(),
                    'optimizer_refine_state_dict': self.optimizer_refine.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict()
                    }, checkpoint_file)

    def load(self, checkpoint_file):

        cp = torch.load(checkpoint_file)
        self.generate.load_state_dict(cp['netG_state_dict'])
        self.refine.load_state_dict(cp['refine_state_dict'])
        if self.isTrain:
            self.optimizer_generate.load_state_dict(cp['optimizer_generate_state_dict'])
            self.optimizer_refine.load_state_dict(cp['optimizer_refine_state_dict'])
            self.netD.load_state_dict(cp['netD_state_dict'])
            self.optimizer_D.load_state_dict(cp['optimizer_D_state_dict'])

    def test(self):

        with torch.no_grad():
            self.forward()

