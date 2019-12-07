###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import pdb

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, fns=None, repeat=1):
    """
        :param dir:
        :param fns: To specify file name list
        :param repeat:
        :return:
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    for i in range(repeat):
                        images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)
                for i in range(repeat):
                    images.append(path)
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        try:
            path = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.return_paths:
                return img, path
            else:
                return img
        except:
            print(index)
            pdb.set_trace()

    def __len__(self):
        return len(self.imgs)


# def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
#                            height=256, width=256, num_workers=4, crop=True):
#     transform_list = [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5))]
#     transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
#     transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
#     transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
#     transform = transforms.Compose(transform_list)
#     dataset = ImageFilelist(root, file_list, transform=transform)
#     loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
#     return loader

def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True, normalization=False):
    transform_list = [transforms.ToTensor()]

    transform_list = transform_list + [transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))] if normalization else transform_list
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if train and crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader
