import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
from torchsummaryX_mod import summary

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    print(get_model_summary(model.cuda(), dump_input.cuda()))
    # print(model)
    summary(model.cuda(), dump_input.cuda())

    # Load pretrained weights
    model_state_file = '/local_storage/users/hfang/HRNet_SN7/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth'        
    
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # # Save torch .pth model (whole model)
    # torch.save(model, 'HRNet.pth')

    # # Export to .onnx model
    # model.eval()
    # batch_size = 1

    # dump_input = dump_input.cuda()
    # export_onnx_file = "HRNet.onnx"
    # torch.onnx.export(model,
    #                     dump_input,
    #                     export_onnx_file,
    #                     opset_version=11,
    #                     do_constant_folding=True,
    #                     input_names=["input"],
    #                     output_names=["output"],
    #                     dynamic_axes={"input":{0:"batch_size"},
    #                                     "output":{0:"batch_size"}})


if __name__ == '__main__':
    main()
