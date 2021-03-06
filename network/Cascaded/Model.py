import pdb

import numpy as np
import torch
import os

from torch import optim

import torch.nn.functional as F
#import network.DuRN_Pure_Conv_3_fg_dual as pure
import network.Cascaded_Refinement as pure
from options import opt
from torch_template.model_zoo import model_zoo
# from network.pyramid_ppp import Pyramid_Net
from torch_template.network.base_model import BaseModel
# import network.DuRN_Pure_Conv_3_coarse_3 as coarse_3
from torch_template.network.metrics import ssim, L1_loss
from torch_template.network.weights_init import init_weights
from torch_template.utils.torch_utils import ExponentialMovingAverage, print_network
from torch_template.loss import VGG16Loss


vgg16loss = VGG16Loss(opt.device)

models = {
    'default': pure.cleaner(),
    'pure': pure.cleaner(),
    'FFA': model_zoo['FFA'],
    'Nested': model_zoo['NestedUNet'],
}


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()

        # self.cleaner = Pyramid_Net(3, 256).cuda(device=opt.device)
        self.cleaner = models[opt.model].cuda(device=opt.device)
        if opt.init:
            init_weights(self.cleaner, opt.init)

        print_network(self.cleaner)

        params = {
            'dim': 64,
            'norm': 'none',
            'activ': 'lrelu',  # activation function [relu/lrelu/prelu/selu/tanh]
            'n_layer': 4,  # number of layers in D
            'gan_type': 'lsgan',  # GAN loss [lsgan/nsgan]
            'num_scales': 3,  # number of scales
            'pad_type': 'reflect'
        }

        # self.discriminitor = MsImageDis(input_dim=3, params=params).cuda(device=opt.device)
        #
        # print(self.discriminitor)

        self.g_optimizer = optim.Adam(self.cleaner.parameters(), lr=opt.lr)
        # self.d_optimizer = optim.Adam(cleaner.parameters(), lr=opt.lr)

        # load networks
        if opt.load:
            pretrained_path = opt.load
            self.load_network(self.cleaner, 'G', opt.which_epoch, pretrained_path)
            # if self.training:
            #     self.load_network(self.discriminitor, 'D', opt.which_epoch, pretrained_path)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update_G(self, x, y):

        # L1 & SSIM loss
        some_loss = 0
        y_downsample_1 = F.interpolate(y, scale_factor=0.5,  mode='bilinear', align_corners=True)
        y_downsample_2 = F.interpolate(y, scale_factor=0.25, mode='bilinear', align_corners=True)

        cleaned, R_, feature1, feature2 = self.cleaner(x)  # feature1 is 64, feature2 is 128
        re_construct = cleaned + R_

        ssim_loss_r = -ssim(cleaned, y)

        percpetual_loss = vgg16loss(cleaned, y) + \
                          vgg16loss(feature1, y_downsample_2) * 0.5 + \
                          vgg16loss(feature2, y_downsample_1) * 0.3


        l1_loss = L1_loss(cleaned, y)

        re_construct_loss = L1_loss(re_construct, x)

        loss = ssim_loss_r * 1.1 + l1_loss * 0.75 + re_construct_loss * 0.75 + percpetual_loss * 0.5

        # GAN loss
        # loss_gen_adv = self.discriminitor.calc_gen_loss(input_fake=cleaned)
        self.avg_meters.update({'ssim': -ssim_loss_r.item(), 'L1': l1_loss.item(), 'reconstruct': re_construct_loss.item(),
                                'percpetual_loss': percpetual_loss.item()})

        #loss_gen = loss + loss_gen_adv * 1.
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return cleaned, re_construct

    def update_D(self, x, y):
        self.d_optimizer.zero_grad()
        # encode
        cleaned = cleaner(x)
        # h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)

        # D loss
        loss_dis = self.discriminitor.calc_dis_loss(input_fake=cleaned, input_real=y)
        self.avg_meters.update({'dis': loss_dis})

        loss_dis = loss_dis * 1.  # weights
        loss_dis.backward()
        self.d_optimizer.step()
        return cleaned

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, x):
        return self.cleaner(x)

    def inference(self, x, image=None):
        pass

    def save(self, which_epoch):
        self.save_network(self.cleaner, 'G', which_epoch)
        # self.save_network(self.discriminitor, 'D', which_epoch)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
