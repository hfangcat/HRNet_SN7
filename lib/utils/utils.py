# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
from torch.nn import functional as F
from config import config

import numpy as np

import torch
import torch.nn as nn

class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce 
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    # def forward(self, inputs, labels, *args, **kwargs):
    #     outputs = self.model(inputs, *args, **kwargs)
    #     loss = self.loss(outputs, labels)
    #     return torch.unsqueeze(loss,0), outputs

    """
    forward function for TCR
    """
    def forward(self, inputs1, inputs2, labels1, labels2, *args, **kwargs):
        score_tcr1, outputs1 = self.model(inputs1, *args, **kwargs)
        score_tcr2, outputs2 = self.model(inputs2, *args, **kwargs)
        loss1 = self.loss(outputs1, labels1)
        loss2 = self.loss(outputs2, labels2)
        # score_tcr1, score_tcr2 -> (bs, 336, 128, 128)
        ph, pw = score_tcr1.size(2), score_tcr1.size(3)
        labels1 = labels1.unsqueeze(1)
        labels2 = labels2.unsqueeze(1)
        # label1, label2 -> (bs, 1, 512, 512)
        h, w = labels1.size(2), labels1.size(3)
        
        if ph != h or pw != w:
            score_tcr1 = F.interpolate(input=score_tcr1, size=
                (h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            score_tcr2 = F.interpolate(input=score_tcr2, size=
                (h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        # loss3 (bs, 336, 512, 512) = | | (bs, 336, 512, 512) * (2| | - 1) (bs, 1, 512, 512)
        loss3 = torch.mean((torch.abs(score_tcr1 - score_tcr2)) * (2 * torch.abs(labels1 - labels2) - 1))

        ce_loss = loss1 + loss2
        
        return torch.unsqueeze(ce_loss,0), torch.unsqueeze(loss3,0), outputs1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr