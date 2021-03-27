import cv2
import importlib
import math
import numpy as np
import random
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from scipy.io import loadmat

# TODO: Change to random_projection
from random_projection.models.archs import define_network
from random_projection.models.base_model import BaseModel

from basicsr.models.losses.losses import g_path_regularize, r1_penalty
from basicsr.utils import imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')


class StyleGAN2Model(BaseModel):
    """StyleGAN2 model."""

    def __init__(self, opt):
        super(StyleGAN2Model, self).__init__(opt)

        # define network net_g
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g',
                                                   True), param_key)

        # latent dimension: self.num_style_feat
        self.num_style_feat = opt['network_g']['num_style_feat']
        num_val_samples = self.opt['val'].get('num_val_samples', 16)
        self.fixed_sample = torch.randn(
            num_val_samples, self.num_style_feat, device=self.device)

        if self.is_train:
            self.init_training_settings()

        # TODO: Load the LDPC matrix
        self.ldpc_mat = torch.FloatTensor(loadmat('/fs/data/tejasj/mrf/ldpc.mat')['H']).to(self.device)

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network net_d
        self.net_d = define_network(deepcopy(self.opt['network_d'])) 
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path'].get('strict_load_d', True))

        # define network net_g with Exponential Moving Average (EMA)
        # net_g_ema only used for testing on one GPU and saving, do not need to
        # wrap with DistributedDataParallel
        self.net_g_ema = define_network(deepcopy(self.opt['network_g'])).to(
            self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path,
                              self.opt['path'].get('strict_load_g',
                                                   True), 'params_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        self.net_g.train()
        self.net_d.train()
        self.net_g_ema.eval()

        # define losses
        # gan loss (wgan)
        cri_gan_cls = getattr(loss_module, train_opt['gan_opt'].pop('type'))
        self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)
        # TODO: Done, Added L1 reconstruction loss
        self.l1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        # regularization weights
        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.path_reg_weight = train_opt['path_reg_weight']  # for generator

        self.net_g_reg_every = train_opt['net_g_reg_every']
        self.net_d_reg_every = train_opt['net_d_reg_every']
        self.mixing_prob = train_opt['mixing_prob']  # TODO: Done, Set this to 0 in the options so that no mixing occurs

        self.mean_path_length = 0

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        net_g_reg_ratio = self.net_g_reg_every / (self.net_g_reg_every + 1)
        if self.opt['network_g']['type'] == 'StyleGAN2GeneratorC':
            normal_params = []
            style_mlp_params = []
            modulation_conv_params = []
            for name, param in self.net_g.named_parameters():
                if 'modulation' in name:
                    normal_params.append(param)
                elif 'style_mlp' in name:
                    style_mlp_params.append(param)
                elif 'modulated_conv' in name:
                    modulation_conv_params.append(param)
                else:
                    normal_params.append(param)
            optim_params_g = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': style_mlp_params,
                    'lr': train_opt['optim_g']['lr'] * 0.01
                },
                {
                    'params': modulation_conv_params,
                    'lr': train_opt['optim_g']['lr'] / 3
                }
            ]
        else:
            normal_params = []
            for name, param in self.net_g.named_parameters():
                normal_params.append(param)
            optim_params_g = [{  # add normal params first
                'params': normal_params,
                'lr': train_opt['optim_g']['lr']
            }]

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params_g,
                lr=train_opt['optim_g']['lr'] * net_g_reg_ratio,
                betas=(0**net_g_reg_ratio, 0.99**net_g_reg_ratio))
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        if self.opt['network_d']['type'] == 'StyleGAN2DiscriminatorC':
            normal_params = []
            linear_params = []
            for name, param in self.net_d.named_parameters():
                if 'final_linear' in name:
                    linear_params.append(param)
                else:
                    normal_params.append(param)
            optim_params_d = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_d']['lr']
                },
                {
                    'params': linear_params,
                    'lr': train_opt['optim_d']['lr'] * (1 / math.sqrt(512))
                }
            ]
        else:
            normal_params = []
            for name, param in self.net_d.named_parameters():
                normal_params.append(param)
            optim_params_d = [{  # add normal params first
                'params': normal_params,
                'lr': train_opt['optim_d']['lr']
            }]

        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(
                optim_params_d,
                lr=train_opt['optim_d']['lr'] * net_d_reg_ratio,
                betas=(0**net_d_reg_ratio, 0.99**net_d_reg_ratio))
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay)

    def feed_data(self, data):
        self.real_img = data['sample'].to(self.device)

    def make_noise(self, batch, num_noise):
        if num_noise == 1:
            noises = torch.randn(
                batch, self.num_style_feat, device=self.device)
        else:
            noises = torch.randn(
                num_noise, batch, self.num_style_feat,
                device=self.device).unbind(0)
        return noises
        
    # TODO: Done, If the encoder output is used, use this as the latent vector
    def mixing_noise(self, batch, prob, encoder_output=None):

        if not encoder_output is None:
                return [encoder_output.to(self.device)]
        if random.random() < prob:
            return self.make_noise(batch, 2)
        else:
            return [self.make_noise(batch, 1)]

    def optimize_parameters(self, current_iter, encoder_output=None):
        loss_dict = OrderedDict()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        batch = self.real_img.size(0)
        noise = self.mixing_noise(batch, self.mixing_prob, encoder_output)
        fake_img, _ = self.net_g(noise)
        fake_pred = self.net_d(fake_img.detach())

        real_pred = self.net_d(self.real_img)
        # wgan loss with softplus (logistic loss) for discriminator
        l_d = self.cri_gan(
            real_pred, True, is_disc=True) + self.cri_gan(
                fake_pred, False, is_disc=True)
        loss_dict['l_d'] = l_d
        # In wgan, real_score should be positive and fake_score should be
        # negative
        loss_dict['real_score'] = real_pred.detach().mean()
        loss_dict['fake_score'] = fake_pred.detach().mean()
        l_d.backward()

        # if current_iter % self.net_d_reg_every == 0:
        #     self.real_img.requires_grad = True
        #     real_pred = self.net_d(self.real_img)
        #     l_d_r1 = r1_penalty(real_pred, self.real_img)
        #     l_d_r1 = (
        #         self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every +
        #         0 * real_pred[0])
        #     # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
        #     # error will arise: RuntimeError: Expected to have finished
        #     # reduction in the prior iteration before starting a new one.
        #     # This error indicates that your module has parameters that were
        #     # not used in producing loss.
        #     loss_dict['l_d_r1'] = l_d_r1.detach().mean()
        #     l_d_r1.backward()

        self.optimizer_d.step()

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        noise = self.mixing_noise(batch, self.mixing_prob, encoder_output)
        fake_img, _ = self.net_g(noise)
        fake_pred = self.net_d(fake_img)

        # wgan loss with softplus (non-saturating loss) for generator
        l_g = self.cri_gan(fake_pred, True, is_disc=False)
        loss_dict['l_g'] = l_g
        # l_g.backward()

        # TODO: Done, Add an L1 loss here
        l_g_l1 = self.l1loss(fake_img, self.real_img)
        loss_dict['l_g_l1'] = l_g_l1

        # TODO: Apply the difference loss between compressed images
        b, c, w, h = fake_img.shape
        pred_projection = torch.matmul(self.ldpc_mat, fake_img.reshape(b, w*h, c))
        true_projection = torch.matmul(self.ldpc_mat, self.real_img.reshape(b, w*h, c))
        l_g_diff = self.mseloss(pred_projection, true_projection)
        loss_dict['l_g_diff'] = l_g_diff

        total_l_g = l_g + 10*l_g_l1 + l_g_diff
        total_l_g.backward()

        # if current_iter % self.net_g_reg_every == 0:
        #     path_batch_size = max(
        #         1, batch // self.opt['train']['path_batch_shrink'])
        #     noise = self.mixing_noise(path_batch_size, self.mixing_prob, encoder_output)
        #     fake_img, latents = self.net_g(noise, return_latents=True)
        #     l_g_path, path_lengths, self.mean_path_length = g_path_regularize(
        #         fake_img, latents, self.mean_path_length)

        #     l_g_path = (
        #         self.path_reg_weight * self.net_g_reg_every * l_g_path +
        #         0 * fake_img[0, 0, 0, 0])
        #     # TODO:  why do we need to add 0 * fake_img[0, 0, 0, 0]
        #     l_g_path.backward()
        #     loss_dict['l_g_path'] = l_g_path.detach().mean()
        #     loss_dict['path_length'] = path_lengths

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA
        self.model_ema(decay=0.5**(32 / (10 * 1000)))

        # Save the fake image
        self.fake_img = fake_img

    def test(self):
        with torch.no_grad():
            self.net_g_ema.eval()
            self.output, _ = self.net_g_ema([self.fixed_sample])

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger,
                                    save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        assert dataloader is None, 'Validation dataloader should be None.'
        self.test()
        result = tensor2img(self.output, min_max=(-1, 1))
        if self.opt['is_train']:
            save_img_path = osp.join(self.opt['path']['visualization'],
                                     'train', f'train_{current_iter}.png')
        else:
            save_img_path = osp.join(self.opt['path']['visualization'], 'test',
                                     f'test_{self.opt["name"]}.png')
        imwrite(result, save_img_path)
        # add sample images to tb_logger
        result = (result / 255.).astype(np.float32)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        if tb_logger is not None:
            tb_logger.add_image(
                'samples', result, global_step=current_iter, dataformats='HWC')

    def save(self, epoch, current_iter):
        self.save_network([self.net_g, self.net_g_ema],
                          'net_g',
                          current_iter,
                          param_key=['params', 'params_ema'])
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
