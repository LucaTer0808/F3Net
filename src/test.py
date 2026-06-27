#!/usr/bin/python3
#coding=utf-8

import os
import sys
import time
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net  import F3Net


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot='./out/model-32', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image)
                out = out2u

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()
    
    def save(self):
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                head  = '/scratch/tmp/lterfehr/models/F3Net/eval/maps/F3Net/dataset'
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        images_processed = len(self.loader.dataset)
        time_per_image = total_time / images_processed if images_processed > 0 else 0

        with open(head + '/evaluation.txt', 'w') as f:
            f.write(f'Total time: {total_time:.4f} seconds\n')
            f.write(f'Images processed: {images_processed}\n')
            f.write(f'Time per image: {time_per_image:.4f} seconds\n')


if __name__=='__main__':
    t = Test(dataset, F3Net, '../data/dataset')
    t.save()

