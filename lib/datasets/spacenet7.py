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

    # Function "rand_crop_tcr" for the image/image_TCR and label/label_TCR
    # Utilized in the validation set
    def rand_crop_tcr(self, image, label, image_TCR, label_TCR):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))

        h_TCR, w_TCR = image_TCR.shape[:-1]
        image_TCR = self.pad_image(image_TCR, h_TCR, w_TCR, self.crop_size, (0.0, 0.0, 0.0))
        label_TCR = self.pad_image(label_TCR, h_TCR, w_TCR, self.crop_size, (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        image_TCR = image_TCR[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label_TCR = label_TCR[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label, image_TCR, label_TCR

    # Function "multi_scale_aug_tcr" for the image/image_TCR and label/label_TCR
    # Utilized in the gen_sample function -> training set
    def multi_scale_aug_tcr(self, image, image_TCR, label=None, label_TCR=None, rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        image_TCR = cv2.resize(image_TCR, (new_w, new_h), 
                           interpolation=cv2.INTER_LINEAR)

        if label is not None and label_TCR is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            label_TCR = cv2.resize(label_TCR, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image, image_TCR

        if rand_crop:
            image, label, image_TCR, label_TCR = self.rand_crop_tcr(image, label, image_TCR, label_TCR)

        return image, label, image_TCR, label_TCR

    # Function "gen_sample_TCR" for the image/image_TCR and label/label_TCR
    # Utilized in the training set
    def gen_sample_tcr(self, image, label, image_TCR, label_TCR, multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label, image_TCR, label_TCR = self.multi_scale_aug_tcr(image, image_TCR, label, label_TCR, rand_scale=rand_scale)

        image = self.random_brightness(image)
        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        image_TCR = self.random_brightness(image_TCR)
        image_TCR = self.input_transform(image_TCR)
        label_TCR = self.label_transform(label_TCR)

        image_TCR = image_TCR.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            image_TCR = image_TCR[:, :, ::flip]
            label_TCR = label_TCR[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )
            label_TCR = cv2.resize(
                label_TCR,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label, image_TCR, label_TCR


    def __getitem__(self, index):
        # Read data from month.json (TCR)
        # a_file = open("/Midgard/home/hfang/temporal_CD/HRNet_SN7/lib/datasets/month.json", "r")
        a_file = open("/proj/berzelius-2021-54/users/HRNet_SN7/lib/datasets/month.json", "r")
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

        if 'test' in self.list_path:
            image = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

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

            # preprocess of image and label of TCR branch (TCR)
            image_TCR, label_TCR = self.resize_short_length(
                image_TCR,
                label=label_TCR,
                short_length=self.base_size,
                fit_stride=8
            )

            # rand_crop -> rand_crop_tcr
            # keep the same transformation for the image/image_TCR and label/label_TCR
            image, label, image_TCR, label_TCR = self.rand_crop_tcr(image, label, image_TCR, label_TCR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            image_TCR = self.input_transform(image_TCR)
            image_TCR = image_TCR.transpose((2, 0, 1))

            # return image.copy(), label.copy(), np.array(size), name
            # return inputs to the two branch (TCR)
            return image.copy(), label.copy(), np.array(size), name, image_TCR.copy(), label_TCR.copy(), np.array(size_TCR), name_TCR

        image, label = self.resize_short_length(image, label, short_length=self.base_size)

        # return image.copy(), label.copy(), np.array(size), name

        # return images and labels of the two branch (TCR)
        image_TCR, label_TCR = self.resize_short_length(image_TCR, label_TCR, short_length=self.base_size)

        # gen_sample -> gen_sample_tcr
        # keep the same transformation for the image/image_TCR and label/label_TCR (flip + rand_crop + multi_scale_aug)
        image, label, image_TCR, label_TCR = self.gen_sample_tcr(image, label, image_TCR, label_TCR, self.multi_scale, self.flip)
        
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


    def inference(self, config, model, image, flip=False):
        size = image.size()
        # seg_hrnet.py -> forward() -> return x_tcr, x
        _, pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            # seg_hrnet.py -> forward() -> return x_tcr, x
            _, flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

            