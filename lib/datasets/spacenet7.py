# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Heng Fang (hfang@kth.se)
# ------------------------------------------------------------------------------

from configparser import Interpolation
import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image
import random
import json

from .base_dataset import BaseDataset

# flag for tcr versions
# flag = 0 -> tcr version 1 (add one tcr branch)
# flag = 1 -> tcr version 2 (add cloud information)
flag = 1

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

    """
    Add cloud information
    """
    # Function "rand_crop_tcr_w_cloud" for the image/image_TCR, label/label_TCR, and cloud/cloud_TCR
    # Utilized in the validation set
    def rand_crop_tcr_w_cloud(self, image, label, cloud, image_TCR, label_TCR, cloud_TCR):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))
        # pad cloud information
        cloud = self.pad_image(cloud, h, w, self.crop_size, (0,))

        h_TCR, w_TCR = image_TCR.shape[:-1]
        image_TCR = self.pad_image(image_TCR, h_TCR, w_TCR, self.crop_size, (0.0, 0.0, 0.0))
        label_TCR = self.pad_image(label_TCR, h_TCR, w_TCR, self.crop_size, (self.ignore_label,))
        # pad cloud_TCR information
        cloud_TCR = self.pad_image(cloud_TCR, h_TCR, w_TCR, self.crop_size, (0,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        # crop cloud information
        cloud = cloud[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        image_TCR = image_TCR[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label_TCR = label_TCR[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        # crop cloud_TCR information
        cloud_TCR = cloud_TCR[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label, cloud, image_TCR, label_TCR, cloud_TCR

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

    """
    Add cloud information
    """
    # Function "multi_scale_aug_tcr_w_cloud" for the image/image_TCR, label/label_TCR, and cloud/cloud_TCR
    # Utilized in the gen_sample function -> training set
    def multi_scale_aug_tcr_w_cloud(self, image, image_TCR, cloud, cloud_TCR, label=None, label_TCR=None, rand_scale=1, rand_crop=True):
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

        # resize cloud and cloud_TCR
        cloud = cv2.resize(cloud, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        cloud_TCR = cv2.resize(cloud_TCR, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if label is not None and label_TCR is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            label_TCR = cv2.resize(label_TCR, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image, image_TCR, cloud, cloud_TCR

        if rand_crop:
            image, label, cloud, image_TCR, label_TCR, cloud_TCR = self.rand_crop_tcr_w_cloud(image, label, cloud, image_TCR, label_TCR, cloud_TCR)

        return image, label, cloud, image_TCR, label_TCR, cloud_TCR

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

    """
    Add cloud information
    """
    # Function "gen_sample_TCR_w_cloud" for the image/image_TCR, label/label_TCR, and cloud/cloud_TCR
    # Utilized in the training set
    def gen_sample_tcr_w_cloud(self, image, label, cloud, image_TCR, label_TCR, cloud_TCR, multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            # multi scale augmentation for image/image_TCR, label/label_TCR, and cloud/cloud_TCR
            image, label, cloud, image_TCR, label_TCR, cloud_TCR = self.multi_scale_aug_tcr_w_cloud(image, image_TCR, cloud, cloud_TCR, label, label_TCR, rand_scale=rand_scale)

        image = self.random_brightness(image)
        image = self.input_transform(image)
        label = self.label_transform(label)
        # label transform for cloud information
        cloud = self.label_transform(cloud)

        image = image.transpose((2, 0, 1))

        image_TCR = self.random_brightness(image_TCR)
        image_TCR = self.input_transform(image_TCR)
        label_TCR = self.label_transform(label_TCR)
        # label transform for cloud_TCR information
        cloud_TCR = self.label_transform(cloud_TCR)

        image_TCR = image_TCR.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            # flip cloud information
            cloud = cloud[:, ::flip]
            image_TCR = image_TCR[:, :, ::flip]
            label_TCR = label_TCR[:, ::flip]
            # flip cloud_TCR information
            cloud_TCR = cloud_TCR[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )
            # downsample cloud information
            cloud = cv2.resize(
                cloud,
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
            # downsample cloud_TCR information
            cloud_TCR = cv2.resize(
                cloud_TCR,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label, cloud, image_TCR, label_TCR, cloud_TCR

    """
    Add cloud information
    """
    # Function resize_short_length_w_cloud for resize image/label/cloud
    def resize_short_length_w_cloud(self, image, cloud, label=None, short_length=None, fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = np.int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = np.int(h * short_length / w + 0.5)        
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        # resize cloud information
        cloud = cv2.resize(cloud, (new_w, new_h),
                           interpolation=cv2.INTER_NEAREST)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=tuple(x * 255 for x in self.mean[::-1])
            )
            # make boarder for cloud information
            cloud = cv2.copyMakeBorder(
                cloud, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=self.ignore_label
            )

        if label is not None:
            label = cv2.resize(
                label, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w, 
                    cv2.BORDER_CONSTANT, value=self.ignore_label
                )
            if return_padding:
                return image, label, cloud, (pad_h, pad_w)
            else:
                return image, label, cloud
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image

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
        # read cloud information (alpha channel)
        # cv2.IMREAD_UNCHANGED -> load an image as such including alpha channel
        # [:, :, 3] -> if the alpha channel is 0 -> fully transparent -> cloud
        # 1 -> cloud, 0 -> no cloud
        if flag == 1:
            cloud = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:,:,3]==0
            cloud = np.array(cloud).astype(int)
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
        # read cloud_TCR information
        if flag == 1:
            cloud_TCR = cv2.imread(image_TCR_path, cv2.IMREAD_UNCHANGED)[:,:,3]==0
            cloud_TCR = np.array(cloud_TCR).astype(int)
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
            if flag == 0:
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
            elif flag == 1:
                # resize_short_length image/label/cloud
                image, label, cloud = self.resize_short_length_w_cloud(
                    image,
                    cloud, 
                    label=label,
                    short_length=self.base_size,
                    fit_stride=8
                )
                image_TCR, label_TCR, cloud_TCR = self.resize_short_length_w_cloud(
                    image_TCR,
                    cloud_TCR,
                    label=label_TCR,
                    short_length=self.base_size,
                    fit_stride=8
                )

            if flag == 0:
                # rand_crop -> rand_crop_tcr
                # keep the same transformation for the image/image_TCR and label/label_TCR
                image, label, image_TCR, label_TCR = self.rand_crop_tcr(image, label, image_TCR, label_TCR)
            elif flag == 1:
                # rand_crop_tcr -> rand_crop_tcr_w_cloud
                # keep the same transformation for the image/image_TCR, label/label_TCR, and cloud/cloud_TCR
                image, label, cloud, image_TCR, label_TCR, cloud_TCR = self.rand_crop_tcr_w_cloud(image, label, cloud, image_TCR, label_TCR, cloud_TCR)

            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            image_TCR = self.input_transform(image_TCR)
            image_TCR = image_TCR.transpose((2, 0, 1))

            if flag == 0:
                # return image.copy(), label.copy(), np.array(size), name
                # return inputs to the two branch (TCR)
                return image.copy(), label.copy(), np.array(size), name, image_TCR.copy(), label_TCR.copy(), np.array(size_TCR), name_TCR
            elif flag == 1:
                # return inputs to the two branch (TCR) with cloud information
                return image.copy(), label.copy(), cloud.copy(), np.array(size), name, image_TCR.copy(), label_TCR.copy(), cloud_TCR.copy(), np.array(size_TCR), name_TCR

        if flag == 0:
            image, label = self.resize_short_length(image, label, short_length=self.base_size)

            # return image.copy(), label.copy(), np.array(size), name

            # return images and labels of the two branch (TCR)
            image_TCR, label_TCR = self.resize_short_length(image_TCR, label_TCR, short_length=self.base_size)
        elif flag == 1:
            # resize_short_length image/label/cloud
            image, label, cloud = self.resize_short_length_w_cloud(image, cloud, label, short_length=self.base_size)
            image_TCR, label_TCR, cloud_TCR = self.resize_short_length_w_cloud(image_TCR, cloud_TCR, label_TCR, short_length=self.base_size)

        if flag == 0:
            # gen_sample -> gen_sample_tcr
            # keep the same transformation for the image/image_TCR and label/label_TCR (flip + rand_crop + multi_scale_aug)
            image, label, image_TCR, label_TCR = self.gen_sample_tcr(image, label, image_TCR, label_TCR, self.multi_scale, self.flip)
            
            return image.copy(), label.copy(), np.array(size), name, image_TCR.copy(), label_TCR.copy(), np.array(size_TCR), name_TCR
        elif flag == 1:
            # gen_sample_tcr -> gen_sample_tcr_w_cloud
            # keep the same transforamtion for the image/image_TCR, label/label_TCR, and cloud/cloud_TCR (flip + rand_crop + multi_scale_aug)
            image, label, cloud, image_TCR, label_TCR, cloud_TCR = self.gen_sample_tcr_w_cloud(image, label, cloud, image_TCR, label_TCR, cloud_TCR, self.multi_scale, self.flip)

            return image.copy(), label.copy(), cloud.copy(), np.array(size), name, image_TCR.copy(), label_TCR.copy(), cloud_TCR.copy(), np.array(size_TCR), name_TCR


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

            