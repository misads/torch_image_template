# encoding=utf-8
"""
Misc PyTorch utils

Author: xuhaoyu@tju.edu.cn

update 12.7

Usage:
    `from torch_utils import *`
    `func_name()`  # to call functions in this file
"""
from datetime import datetime
import os

import torch
from tensorboardX import SummaryWriter


##############################
#    Abstract Meters class
##############################
class Meters(object):
    def __init__(self):
        pass

    def update(self, new_dic):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def items(self):
        return self.dic.items()


class AverageMeters(Meters):
    """
        Example:
        avg_meters = AverageMeters()
        for i in range(100):
            avg_meters.update({'f': i})
            print(str(avg_meters))
    """

    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


class ExponentialMovingAverage(Meters):
    """
        Example:
        ema_meters = ExponentialMovingAverage(0.98)
        for i in range(100):
            ema_meters.update({'f': i})
            print(str(ema_meters))
    """

    def __init__(self, decay=0.9, dic=None, total_num=None):
        self.decay = decay
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        decay = self.decay
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = (1 - decay) * new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] = decay * self.dic[key] + (1 - decay) * new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key]  # / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


def load_ckpt(model, ckpt_path):
    """
        Example:
            class Model(nn.Module):
                ....

            model = Model().cuda()
            load_ckpt(model, 'model.pt')

        :param model: object of a subclass of nn.Module
        :param ckpt_path: *.pt file to load
        :return:
    """
    model.load_state_dict(torch.load(ckpt_path))


def save_ckpt(model, ckpt_path):
    """
        Example:
            class Model(nn.Module):
                ....

            model = Model().cuda()
            save_ckpt(model, 'model.pt')

        :param model: object of a subclass of nn.Module
        :param ckpt_path: *.pt file to save
        :return:
    """

    torch.save(model.state_dict(), ckpt_path)


"""
    TensorBoard
    Example:
        writer = create_summary_writer(os.path.join(self.basedir, 'logs'))
        write_meters_loss(writer, 'train', avg_meters, iteration)
        write_loss(writer, 'train', 'F1', 0.78, iteration)
        write_image(writer, 'train', 'input', img, iteration)
        # shell
        tensorboard --logdir {base_path}/logs

"""


def create_summary_writer(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    return writer


def write_loss(writer: SummaryWriter, prefix, loss_name: str, value: float, iteration):
    """
        Example:
            write_loss(writer, 'train', 'F1', 0.78, iteration)
        :param writer: writer created by create_summary_writer()
        :param prefix: e.g. for '/train/loss1' is 'train'
        :param loss_name:
        :param value:
        :param iteration:
    :return:
    """
    writer.add_scalar(
        os.path.join(prefix, loss_name), value, iteration)


def write_image(writer: SummaryWriter, prefix, image_name: str, img, iteration, dataformats='CHW'):
    """
        Example:
            write_image(writer, 'train', 'input', img, iteration)
        :param writer: writer created by create_summary_writer()
        :param prefix:
        :param image_name:
        :param img: image Tensor, should be channel first. Specific size of [C, H, W].
        :param iteration:
        :param dataformats: 'CHW' or 'HWC' or 'NCHW'''
    :return:
    """
    writer.add_image(
        os.path.join(prefix, image_name), img, iteration, dataformats=dataformats)


def write_meters_loss(writer: SummaryWriter, prefix, avg_meters: Meters, iteration):
    """
        Example:
            writer = create_summary_writer(os.path.join(self.basedir, 'logs'))
            ema_meters = ExponentialMovingAverage(0.98)
            for i in range(100):
                ema_meters.update({'f1': i, 'f2': i*0.5})
                write_meters_loss(writer, 'train', ema_meters, i)
        :param writer:
        :param prefix:
        :param avg_meters: avg_meters param should be a Meters subclass
        :param iteration:
        :return:
    """
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(
            os.path.join(prefix, key), meter, iteration)


