"""
    Unofficial implementation of 'Single Image Reflection Removal through Cascaded Refinement'

    Paper:
        https://arxiv.org/pdf/1911.06634.pdf

    Author: xuhaoyu@tju.edu.cn
"""
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from options import opt

#  Refinement iterations
iters = 4   # Refinement iterations
####################################

class cleaner(nn.Module):
    def __init__(self):
        super(cleaner, self).__init__()
        for i in range(iters):
            self.add_module('G_T_%d' % (i + 1), G_Module(in_channels=9))
            self.add_module('G_R_%d' % (i + 1), G_Module(in_channels=9))

        self.conv_feature1 = nn.Conv2d(128, 3, kernel_size=3, stride=2, padding=1)
        self.conv_feature2 = nn.Conv2d(64, 3, kernel_size=3, stride=2, padding=1)

    def forward(self, I):
        batch_size, h, w = I.size(0), I.size(2) // 4, I.size(3) // 4
        T_ = Variable(torch.zeros_like(I).cuda(device=opt.device))
        R_ = Variable(torch.zeros_like(I).cuda(device=opt.device))

        x = torch.cat((T_, I, R_), 1)

        c1 = Variable(torch.zeros(batch_size, 256, h, w)).cuda(device=opt.device)
        h1 = Variable(torch.zeros(batch_size, 256, h, w)).cuda(device=opt.device)
        c2 = Variable(torch.zeros(batch_size, 256, h, w)).cuda(device=opt.device)
        h2 = Variable(torch.zeros(batch_size, 256, h, w)).cuda(device=opt.device)

        for i in range(iters):
            T_, c1, h1, feature1, feature2 = self._modules['G_T_%d' % (i + 1)](x, c1, h1)
            R_, c2, h2, _, _ = self._modules['G_R_%d' % (i + 1)](x, c2, h2)
            x = torch.cat((T_, I, R_), 1)

        feature1 = self.conv_feature1(feature1)
        feature2 = self.conv_feature2(feature2)

        return T_, R_, feature1, feature2


class G_Module(nn.Module):
    def __init__(self, in_channels=9, out_channels=3):
        super(G_Module, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, 1, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )

        """
            LSTM
        """
        self.conv_i = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, 1, 1),
            nn.Sigmoid()
        )
        """
            LSTM
        """

        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
            nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
            nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8),
            nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, input, c, h):
        """

            :param input:
            :param c: c(t-1)
            :param h: h(t-1)
            :return:
        """
        x = self.conv1(input)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        #################
        #      LSTM
        #################

        x = torch.cat((x, h), 1)

        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * F.tanh(c)
        #################
        #      LSTM
        #################
        x = self.diconv1(h)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.deconv1(x)
        x = x + res2

        feature1 = x

        x = self.conv9(x)
        x = self.deconv2(x)
        x = x + res1
        feature2 = x
        x = self.conv10(x)
        out = self.output(x)

        return out, c, h, feature1, feature2
