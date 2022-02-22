#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import logging
import argparse
import math
from tabulate import tabulate
import time
from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.banet import model
from configs import cfg
from lib.logger import setup_logger
from lib.cityscapes_cv2 import get_data_loader

parser = argparse.ArgumentParser(description='PyTorch SOD FOR CSNet')
parser.add_argument(
    "--config",
    default="configs/csnet-L-x2_train.yml",
    metavar="FILE",
    help="path to config file",
    type=str,
)

class MscEvalV0(object):

    def __init__(self, scales=(0.5, ), flip=False, ignore_label=255):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, val_loader, n_classes):
        ## evaluate
        time_list=[]
        hist = torch.zeros(n_classes, n_classes, dtype=torch.float32).cpu().detach()
        # for i, (imgs, label) in diter:
        for i, [imgs, label, h, w] in enumerate(val_loader):
            label = torch.stack(label)
            N, _, H, W = label.shape
            label = label.squeeze(1).cuda().float()
            size_t = label.size()[-2:]
            size = size_t[0]
            probs = torch.zeros(
                    (N, n_classes, H, W), dtype=torch.float32).cuda().detach()
            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda().float()
                start = time.time()
                logits = net(im_sc)
                end = time.time()
                time_cost = end-start
                time_list.append(time_cost)
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc)[0]
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            label = label.cpu()
            preds = preds.cpu()
            bin_input = (label[keep] * n_classes + preds[keep]).cpu().int()
            hist += torch.bincount(
                bin_input,
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        fps = (len(time_list)/np.sum(time_list))
        print('fps : ',fps)
        print('sum',np.sum(time_list))
        print('len',len(time_list))
        return miou.item()



class MscEvalCrop(object):

    def __init__(
        self,
        cropsize=1024,
        cropstride=2./3,
        flip=True,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        lb_ignore=255,
    ):
        self.scales = scales
        self.ignore_label = lb_ignore
        self.flip = flip
        #self.distributed = dist.is_initialized()

        self.cropsize = cropsize if isinstance(cropsize, (list, tuple)) else (cropsize, cropsize)
        self.cropstride = cropstride


    def pad_tensor(self, inten):
        N, C, H, W = inten.size()
        cropH, cropW = self.cropsize
        if cropH < H and cropW < W: return inten, [0, H, 0, W]
        padH, padW = max(cropH, H), max(cropW, W)
        outten = torch.zeros(N, C, padH, padW).cuda()
        outten.requires_grad_(False)
        marginH, marginW = padH - H, padW - W
        hst, hed = marginH // 2, marginH // 2 + H
        wst, wed = marginW // 2, marginW // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, net, crop):
        prob = net(crop)[0].softmax(dim=1)
        if self.flip:
            crop = torch.flip(crop, dims=(3,))
            prob += net(crop)[0].flip(dims=(3,)).softmax(dim=1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, net, im, n_classes):
        cropH, cropW = self.cropsize
        stride_rate = self.cropstride
        im, indices = self.pad_tensor(im)
        N, C, H, W = im.size()

        strdH = math.ceil(cropH * stride_rate)
        strdW = math.ceil(cropW * stride_rate)
        n_h = math.ceil((H - cropH) / strdH) + 1
        n_w = math.ceil((W - cropW) / strdW) + 1
        prob = torch.zeros(N, n_classes, H, W).cuda()
        prob.requires_grad_(False)
        for i in range(n_h):
            for j in range(n_w):
                stH, stW = strdH * i, strdW * j
                endH, endW = min(H, stH + cropH), min(W, stW + cropW)
                stH, stW = endH - cropH, endW - cropW
                chip = im[:, :, stH:endH, stW:endW]
                prob[:, :, stH:endH, stW:endW] += self.eval_chip(net, chip)
        hst, hed, wst, wed = indices
        prob = prob[:, :, hst:hed, wst:wed]
        return prob


    def scale_crop_eval(self, net, im, scale, n_classes):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(net, im, n_classes)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    @torch.no_grad()
    def __call__(self, net, dl, n_classes):
        #dloader = dl if self.distributed and not dist.get_rank() == 0 else tqdm(dl)
        dloader = tqdm(dl)
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        hist.requires_grad_(False)
        for i, (imgs, label) in enumerate(dloader):
            imgs = imgs.cuda()
            label = label.squeeze(1).cuda()
            N, H, W = label.shape
            probs = torch.zeros((N, n_classes, H, W)).cuda()
            probs.requires_grad_(False)
            for sc in self.scales:
                probs += self.scale_crop_eval(net, imgs, sc, n_classes)
            torch.cuda.empty_cache()
            preds = torch.argmax(probs, dim=1)

            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()

import argparse
parse = argparse.ArgumentParser()
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument('--name',dest = 'store_name',type=str)
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='model_final.pth',)
    parse.add_argument('--epoch-to-train', dest = 'epoch_to_train',type=int)
    return parse.parse_args()

args = parse_args()

@torch.no_grad()
def eval_model(net, val_loader, it):
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()
    
    if (it+1)!=cfg.SOLVER.MAX_EPOCHS:
        single_scale = MscEvalV0((1., ), False)
        mIOU = single_scale(net, val_loader, 19)
        heads.append('single_scale')
        mious.append(mIOU)
        logger.info('single mIOU is: %s\n', mIOU)
        return heads, mious,mIOU
    
    single_scale = MscEvalV0((1., ), False)
    mIOUss = single_scale(net, val_loader, 19)
    heads.append('single_scale')
    mious.append(mIOUss)
    logger.info('single mIOU is: %s\n', mIOUss)
 
    single_crop = MscEvalCrop(
        cropsize=1024,
        cropstride=2. / 3,
        flip=False,
        scales=[1., ],
        lb_ignore=255,
    )
    '''mIOU = single_crop(net, dl, 19)
    heads.append('single_scale_crop')
    mious.append(mIOU)
    logger.info('single scale crop mIOU is: %s\n', mIOU)

    ms_flip = MscEvalV0((0.5, 0.75, 1, 1.25, 1.5, 1.75), True)
    mIOU = ms_flip(net, dl, 19)
    heads.append('ms_flip')
    mious.append(mIOU)
    logger.info('ms flip mIOU is: %s\n', mIOU)

    ms_flip_crop = MscEvalCrop(
        cropsize=1024,
        cropstride=2. / 3,
        flip=True,
        scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
        lb_ignore=255,
    )
    mIOU = ms_flip_crop(net, dl, 19)
    heads.append('ms_flip_crop')
    mious.append(mIOU)
    logger.info('ms crop mIOU is: %s\n', mIOU)'''
    return heads, mious,mIOUss


def evaluate(cfg, weight_pth):
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = model_factory[args.model](19)
    #  net = BiSeNetV2(19)
    net.load_state_dict(torch.load(weight_pth))
    net.cuda()
    ## evaluator
    heads, mious = eval_model(net, 2, cfg.im_root, cfg.val_im_anns)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))


def main():
    args = parse_args()
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-eval'.format('BANet', cfg.respth))
    evaluate(cfg, args.weight_pth)


if __name__ == "__main__":
    main()
