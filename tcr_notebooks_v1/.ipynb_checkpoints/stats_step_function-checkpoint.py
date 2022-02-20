# import necessary packages
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing
import pandas as pd
import numpy as np
import skimage.io
import tqdm
import glob
import math
import gdal
import time
import sys
import os

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib
# matplotlib.use('Agg') # non-interactive
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

import solaris as sol
from solaris.utils.core import _check_gdf_load
from solaris.raster.image import create_multiband_geotiff 

# import from data_postproc_funcs
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from sn7_baseline_postproc_funcs import map_wrapper, multithread_polys, \
        calculate_iou, track_footprint_identifiers, \
        sn7_convert_geojsons_to_csv

import argparse

parser = argparse.ArgumentParser(description='Specify prediction directory in command line')
parser.add_argument('pred_top_dir', type=str)

args = parser.parse_args()

# test
# pred_top_dir = '/Midgard/home/hfang/temporal_CD/Berzelius_HRNet_SN7/berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_1_train'
pred_top_dir = args.pred_top_dir
print(pred_top_dir)
im_top_dir = '/local_storage/datasets/sn7_winner_split/test_public'

def right_shift_array(a, shift):
    # we can also call it roll with padding
    tmp = np.zeros_like(a)
    tmp = np.roll(a, shift)
    tmp[:shift] = 0
    return tmp

# get the list of aoi
aois = sorted([f for f in os.listdir(os.path.join(im_top_dir))
                   if os.path.isdir(os.path.join(im_top_dir, f))
                  and f != 'list'])

all_0_list = np.zeros(len(aois))
all_1_list = np.zeros(len(aois))
others_list = np.zeros(len(aois))

for i, aoi in enumerate(aois):
    print(i, "aoi:", aoi)
    
    # directory paths (predictions & input images & ground truth)
    pred_dir = os.path.join(pred_top_dir, 'grouped', aoi, 'masks')
    im_dir = os.path.join(im_top_dir, aoi, 'images_masked')
    gt_dir = os.path.join(im_top_dir, aoi, 'masks')

    im_list = sorted([z for z in os.listdir(im_dir) if z.endswith('.tif')])
    
    # lists of frames (# of frames * 3072 * 3072)
    num_frames = len(im_list)
    gt_image_concat = np.zeros((num_frames, 3072, 3072))
    mask_image_concat = np.zeros((num_frames, 3072, 3072))
    cloud_concat = np.zeros((num_frames, 3072, 3072), dtype=bool)
    
    # frame loop
    for j, f in enumerate(im_list):        
        sample_mask_name = f
        # file paths
        sample_mask_path = os.path.join(pred_dir, sample_mask_name).replace('.tif', '.npy')
        sample_im_path = os.path.join(im_dir, sample_mask_name)
        sample_gt_path = os.path.join(gt_dir, sample_mask_name).replace('.tif', '_Buildings.tif')

        image = skimage.io.imread(sample_im_path)
        mask_image = np.load(sample_mask_path)
        gt_image = skimage.io.imread(sample_gt_path)

        norm = (mask_image - np.min(mask_image)) / (np.max(mask_image) - np.min(mask_image))

        gt_image = gt_image / 255

        tmp = np.zeros((1024, 1024))
        tmp[:gt_image.shape[0],:gt_image.shape[1]] = gt_image
        gt_image = np.repeat(tmp, 3, axis=0)
        gt_image = np.repeat(gt_image, 3, axis=1)
        
        # get cloud information
        cloud = image[:,:,3]==0
        cloud = np.repeat(cloud, 3, axis=0)
        cloud = np.repeat(cloud, 3, axis=1)
        
#         figsize=(24, 12)
#         name = sample_im_path.split('/')[-1].split('.')[0].split('global_monthly_')[-1]
#         fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
#         _ = ax0.imshow(image)
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         # _ = ax0.set_title(name)
#         _ = ax1.imshow(cloud)
#         ax1.set_xticks([])
#         ax1.set_yticks([])
#         # _ = ax1.set_title(name)
#         _ = fig.suptitle(name)
#         plt.tight_layout()
        
        gt_image_concat[j] = gt_image
        mask_image_concat[j] = mask_image
        cloud_concat[j] = cloud

    print(gt_image.shape)
    print(mask_image.shape)
    print(cloud.shape)
    
    all_0 = 0
    all_1 = 0
    others = 0
    
    for p in range(3072):
#         print(p)
        for q in range(3072):
            gt_seq = gt_image_concat[:, p, q]
            cloud_seq = cloud_concat[:, p, q]
            mask_seq = mask_image_concat[:, p, q]
            
            tmp_list = []
            
#             cloud_seq[2] = True
#             cloud_seq[21] = True
#             gt_seq[4] = True
#             gt_seq[20] = True
#             print(gt_seq)
#             print(cloud_seq)
#             print(mask_seq)
            
            # delete pixels in cloud masks
            for tmp in range(num_frames):
                if cloud_seq[tmp]:
                    tmp_list.append(tmp)
                    
            gt_seq = np.delete(gt_seq, tmp_list)
            mask_seq = np.delete(mask_seq, tmp_list)
            
#             print(gt_seq)
#             print(cloud_seq)
#             print(mask_seq)
                    
            num_frames_wo_cloud = len(gt_seq)
            
            ref_0 = np.zeros(num_frames_wo_cloud, dtype=bool)
            ref_1 = np.ones(num_frames_wo_cloud, dtype=bool)
            ref_01 = np.zeros_like(ref_1, dtype=bool)

#             print(ref_0)
#             print(ref_1)
#             print(ref_01)

            gt_seq = np.array(gt_seq, dtype=bool)
            
            for tmp in range(1, num_frames_wo_cloud):
                ref_01 = right_shift_array(ref_1, tmp)
#                 print(ref_01)
                
                if all(gt_seq ^ ref_01 == ref_01):
                    others += 1
                    
            if all(gt_seq ^ ref_0 == ref_0):
                all_0 += 1
            elif all(gt_seq ^ ref_1 == ref_1):
                all_1 += 1
                
    all_0_list[i] = all_0
    all_1_list[i] = all_1
    others_list[i] = others
    
print(all_0_list)
print(all_1_list)
print(others_list)