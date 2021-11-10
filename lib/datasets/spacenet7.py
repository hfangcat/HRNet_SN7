# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Heng Fang (hfang@kth.se)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image
import random
import json

from .base_dataset import BaseDataset


class Spacenet7(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=2,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=512,
                 crop_size=(512, 512),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(Spacenet7, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {
                    'img': image_path,
                    'label': label_path,
                    'name': name
                }
                files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        # Read data from month.json (TCR)
        a_file = open("/Midgard/home/hfang/temporal_CD/HRNet_SN7/lib/datasets/month.json", "r")
        dic = json.load(a_file)
        a_file.close()

        item = self.files[index]
        name = item["name"]
        # image_path = os.path.join(self.root, 'ade20k', item['img'])
        # label_path = os.path.join(self.root, 'ade20k', item['label'])
        image_path = os.path.join(self.root, item['img'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        size = image.shape

        # Get another input to the second branch (TCR)
        aoi = item['img'].split('/')[0]
        month = item['img'].split('_mosaic')[0].split('monthly_')[-1]
        month_TCR = random.choice([j for j in dic[aoi] if j != month])
        name_TCR = name.replace(month, month_TCR)
        image_TCR_path = os.path.join(self.root, item['img'].replace(month, month_TCR))
        image_TCR = cv2.imread(
            image_TCR_path,
            cv2.IMREAD_COLOR
        )
        size_TCR = image_TCR.shape

        if 'test' in self.list_path:
            image = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label_path = os.path.join(self.root, item['label'])
        label = np.array(
            Image.open(label_path).convert('P')
        )

        label[label != 0] = 1

        # Get another label to the second branch (TCR)
        label_TCR_path = os.path.join(self.root, item['label'].replace(month, month_TCR))
        label_TCR = np.array(
            Image.open(label_TCR_path).convert('P')
        )
        label_TCR[label_TCR != 0] = 1

        if 'val' in self.list_path:
            image, label = self.resize_short_length(
                image,
                label=label,
                short_length=self.base_size,
                fit_stride=8
            )

            image, label = self.rand_crop(image, label)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            # preprocess of image and label of TCR branch (TCR)
            image_TCR, label_TCR = self.resize_short_length(
                image_TCR,
                label=label_TCR,
                short_length=self.base_size,
                fit_stride=8
            )

            image_TCR, label_TCR = self.rand_crop(image_TCR, label_TCR)
            image_TCR = self.input_transform(image_TCR)
            image_TCR = image_TCR.transpose((2, 0, 1))

            # return image.copy(), label.copy(), np.array(size), name
            # return inputs to the two branch (TCR)
            return image.copy(), label.copy(), np.array(size), name, image_TCR.copy(), label_TCR.copy(), np.array(size_TCR), name_TCR

        image, label = self.resize_short_length(image, label, short_length=self.base_size)
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        # return image.copy(), label.copy(), np.array(size), name

        # return images and labels of the two branch (TCR)
        image_TCR, label_TCR = self.resize_short_length(image_TCR, label_TCR, short_length=self.base_size)
        image_TCR, label_TCR = self.gen_sample(image_TCR, label_TCR, self.multi_scale, self.flip)
        return image.copy(), label.copy(), np.array(size), name, image_TCR.copy(), label_TCR.copy(), np.array(size_TCR), name_TCR


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(preds.cpu())
        for i in range(preds.shape[0]):
            # fetch the probability of the building class after the softmax layer
            pred = preds[i][1]
            pred = np.squeeze(pred)
            sv_path_npy = os.path.join(sv_path, 'npy')
            if not os.path.exists(sv_path_npy):
                    os.mkdir(sv_path_npy)
            np.save(os.path.join(sv_path_npy, name[i]+'.npy'), pred)

            pred = np.asarray(pred * 255.0, dtype=np.uint8)
            save_img = Image.fromarray(pred)
            sv_path_png = os.path.join(sv_path, 'png')
            if not os.path.exists(sv_path_png):
                    os.mkdir(sv_path_png)
            save_img.save(os.path.join(sv_path_png, name[i]+'.png'))

            