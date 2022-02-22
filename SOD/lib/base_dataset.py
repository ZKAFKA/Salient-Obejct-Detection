#!/usr/bin/python
# -*- encoding: utf-8 -*-
import random
import os
import os.path as osp
import json
from skimage.transform import resize
import skimage
import torch
import torchvision
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.transforms import ToTensor
# from lib.sampler import RepeatedDistSampler


def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        return None
    else:
        ext = allfiles[0].split('.')[-1]
        filelist = [
            fname.replace(''.join(['.', ext]), '') for fname in allfiles
        ]
        return ext, filelist


class BaseDataset(Dataset):
    def __init__(self, dataDir, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        if not os.path.isdir(os.path.join(dataDir, 'images')):
            raise ValueError(
                'Please put your images in folder \'Images\' and masks in \'Masks\'')
        self.mode = mode
        self.trans_func = trans_func
        self.dataDir = dataDir
        _, self.imgList = fold_files(os.path.join(dataDir, 'images'))
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        # imgName = self.imgList[idx]
        # img = os.path.join(self.dataDir, 'images', imgName + '.jpg')
        # gt = os.path.join(self.dataDir, 'masks', imgName + '.png')
        #
        # img, gt = cv2.imread(img)[:, :, ::-1].copy(), cv2.imread(gt).copy()
        #
        #
        # img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        # gt = torch.from_numpy(np.transpose(gt, (2, 0, 1)))
        #
        # if not self.trans_func is None:
        #     img = self.trans_func(img)
        #     gt = self.trans_func(gt)
        #
        # return img.detach(), gt.detach()
        imgName = self.imgList[idx]

        img = skimage.img_as_float(
            io.imread(os.path.join(self.dataDir, 'images', imgName + '.jpg')))
        gt = skimage.img_as_float(
            io.imread(os.path.join(self.dataDir, 'masks', imgName + '.png'),
                      as_gray=True))
        imgsize = gt.shape
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            img = np.repeat(img, 3, 2)
        img = resize(img, (1024, 1024),
                     mode='reflect',
                     anti_aliasing=False)
        if self.mode == 'train':
            gt = resize(gt, (1024, 1024),
                        mode='reflect',
                        anti_aliasing=False)
        # Normalize image
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        gt = gt[np.newaxis, ::]
        img = torch.from_numpy(img)
        gt = torch.from_numpy(gt)
        if not self.trans_func is None:
            img = self.trans_func(img)

        print(img.type)
        print(gt.type)
        if self.mode == "train":
            sample = {'img': img, 'gt': gt}
        else:
            sample = {'img': img, 'gt': gt, 'h': imgsize[0], 'w': imgsize[1]}
        return sample
class Augment(object):
    """
    Augment image as well as target(image like array, not box)
    augmentation include Crop Pad and Filp
    """
    def __init__(self, size_h=15, size_w=15, padding=None, p_flip=None):
        super(Augment, self).__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.padding = padding
        self.p_flip = p_flip

    def get_params(self, img):
        im_sz = img.shape[:2]
        row1 = random.randrange(self.size_h)
        row2 = -random.randrange(
            self.size_h) - 1  # minus 1 to avoid row1==row2==0
        col1 = random.randrange(self.size_w)
        col2 = -random.randrange(self.size_w) - 1
        if row1 - row2 >= im_sz[0] or col1 - col2 >= im_sz[1]:
            raise ValueError(
                "Image size too small, please choose smaller crop size")
        padding = None
        if self.padding is not None:
            padding = random.randint(0, self.padding)
        flip_method = None
        if self.p_flip is not None and random.random() < self.p_flip:
            if random.random() < 0.5:
                flip_method = 'lr'
            else:
                flip_method = 'ud'
        return row1, row2, col1, col2, flip_method, padding

    def transform(self,
                  img,
                  row1,
                  row2,
                  col1,
                  col2,
                  flip_method,
                  padding=None):
        """img should be 2 or 3 dimensional numpy array"""
        img = img[row1:row2,
                  col1:col2, :] if len(img.shape) == 3 else img[row1:row2,
                                                                col1:col2]
        if padding is not None:  # TODO: not working yet, fix it later
            pad = transforms.Pad(padding)
            topil = transforms.ToPILImage()
            img = pad(topil(img))
            img = np.array(img)
        if flip_method is not None:
            if flip_method == 'lr':
                img = np.fliplr(img)
            else:
                img = np.flipud(img)
        return img

    def __call__(self, img, target):
        """img and target should have the same spatial size"""
        paras = self.get_params(img)
        img = self.transform(img, *paras)
        target = self.transform(target, *paras)
        return img, target

class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.trans_func = T.Compose([
            T.RandomResizedCrop(size=cropsize,scale=scales),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb

class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['img'], im_lb['gt']
        return dict(im=im, lb=lb)


# if __name__ == "__main__":
#     from tqdm import tqdm
#     from torch.utils.data import DataLoader
#     ds = CityScapes('./data/', mode='val')
#     dl= DataLoader(ds,
#                     batch_size = 4,
#                     shuffle = True,
#                     num_workers = 4,
#                     drop_last = True)
#     for imgs, label in dl:
#         print(len(imgs))
#         for el in imgs:
#             print(el.size())
#         break
