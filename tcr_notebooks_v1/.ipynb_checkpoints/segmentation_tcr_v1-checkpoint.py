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

def dice_coef(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    return (2. * np.sum(intersection)) / (np.sum(y_true) + np.sum(y_pred))

def iou_coef(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def auc_coef(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def ap_coef(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return average_precision_score(y_true, y_pred)

def evaluate(pred_top_dir, im_top_dir, outfile, mode):
    df = pd.DataFrame()
    
    aois = sorted([f for f in os.listdir(os.path.join(im_top_dir))
                   if os.path.isdir(os.path.join(im_top_dir, f))
                  and f != 'list'])
    
    print(aois)

    auc_multi_aoi = 0.
    ap_multi_aoi = 0.
    dice_multi_aoi = 0.
    iou_multi_aoi = 0.

    for i, aoi in enumerate(aois):
        print(i, "aoi:", aoi)

        pred_dir = os.path.join(pred_top_dir, 'grouped', aoi, 'masks')
        im_dir = os.path.join(im_top_dir, aoi, 'images_masked')
        gt_dir = os.path.join(im_top_dir, aoi, 'masks')

        im_list = sorted([z for z in os.listdir(im_dir) if z.endswith('.tif')])

        auc_one_aoi = 0.
        ap_one_aoi = 0.
        dice_one_aoi = 0.
        iou_one_aoi = 0.

        for j, f in enumerate(im_list):        
            sample_mask_name = f
            if mode == 'baseline':
                sample_mask_path = os.path.join(pred_dir, sample_mask_name)
            elif mode == 'winner':
                sample_mask_path = os.path.join(pred_dir, sample_mask_name).replace('.tif', '.npy')
            sample_im_path = os.path.join(im_dir, sample_mask_name)
            sample_gt_path = os.path.join(gt_dir, sample_mask_name).replace('.tif', '_Buildings.tif')

            image = skimage.io.imread(sample_im_path)
            if mode == 'baseline':
                mask_image = skimage.io.imread(sample_mask_path)
            elif mode == 'winner':
                mask_image = np.load(sample_mask_path)
            gt_image = skimage.io.imread(sample_gt_path)
    #         print(mask_image)
    #         print("mask_image.shape:", mask_image.shape)
    #         print("min, max, mean mask image:", np.min(mask_image), np.max(mask_image), np.mean(mask_image))
    #         print(gt_image)
    #         print("gt_image.shape:", gt_image.shape)
    #         print("min, max, mean gt image:", np.min(gt_image), np.max(gt_image), np.mean(gt_image))

            norm = (mask_image - np.min(mask_image)) / (np.max(mask_image) - np.min(mask_image))

    #         print(norm)
    #         print("mask_image.shape:", norm.shape)
    #         print("min, max, mean mask image:", np.min(norm), np.max(norm), np.mean(norm))

            gt_image = gt_image / 255
            
            if mode == 'winner':
                tmp = np.zeros((1024, 1024))
                tmp[:gt_image.shape[0],:gt_image.shape[1]] = gt_image
                gt_image = np.repeat(tmp, 3, axis=0)
                gt_image = np.repeat(gt_image, 3, axis=1)
            
            auc = auc_coef(gt_image, norm)
            ap = ap_coef(gt_image, norm)
            
            auc_one_aoi += auc
            ap_one_aoi += ap
                
            norm = np.where(norm > 0.5, 1, 0)

            dice = dice_coef(gt_image, norm)
            iou = iou_coef(gt_image, norm)
            dice_one_aoi += dice
            iou_one_aoi += iou

        auc_one_aoi = auc_one_aoi / len(im_list)
        ap_one_aoi = ap_one_aoi / len(im_list)
        dice_one_aoi = dice_one_aoi / len(im_list)
        iou_one_aoi = iou_one_aoi / len(im_list)

        print("AUC: ", auc_one_aoi)
        print("AP: ", ap_one_aoi)
        print("Dice: ", dice_one_aoi)
        print("IOU: ", iou_one_aoi)
        
        row1 = {'AOI': aoi, 'AUC': auc_one_aoi, 'AP': ap_one_aoi, 'Dice': dice_one_aoi, 'IOU': iou_one_aoi}
        df = df.append(row1, ignore_index=True)
        
        auc_multi_aoi += auc_one_aoi
        ap_multi_aoi += ap_one_aoi
        dice_multi_aoi += dice_one_aoi
        iou_multi_aoi += iou_one_aoi
    
    df.to_csv(outfile, index=False, header=True)

    auc_multi_aoi = auc_multi_aoi / len(aois)
    ap_multi_aoi = ap_multi_aoi / len(aois)
    dice_multi_aoi = dice_multi_aoi / len(aois)
    iou_multi_aoi = iou_multi_aoi / len(aois)

    print("Average AUC: ", auc_multi_aoi)
    print("Average AP: ", ap_multi_aoi)
    print("Average Dice: ", dice_multi_aoi)
    print("Average IOU: ", iou_multi_aoi)
    
def group(raw_name, grouped_name):
    im_list = sorted([z for z in os.listdir(os.path.join(raw_name)) if z.endswith('.npy')])
    df = pd.DataFrame({'image': im_list})
    roots = [z.split('mosaic_')[-1].split('.npy')[0] for z in df['image'].values]
    df['root'] = roots
    # copy files
    for idx, row in df.iterrows():
        in_path_tmp = os.path.join(raw_name, row['image'])
        out_dir_tmp = os.path.join(grouped_name, row['root'], 'masks')
        os.makedirs(out_dir_tmp, exist_ok=True)
        cmd = 'cp ' + in_path_tmp + ' ' + out_dir_tmp
#         print("cmd:", cmd)
        os.system(cmd)   
    
def dice_iou_threshold(pred_top_dir, im_top_dir, outfile, mode):
    df = pd.DataFrame()
    
    aois = sorted([f for f in os.listdir(os.path.join(im_top_dir))
                   if os.path.isdir(os.path.join(im_top_dir, f))
                  and f != 'list'])

    for theta in range(11):
        threshold = theta / 10
        print(threshold)
        
        dice_multi_aoi = 0.
        iou_multi_aoi = 0.

        for i, aoi in enumerate(aois):
            print(i, "aoi:", aoi)

            pred_dir = os.path.join(pred_top_dir, 'grouped', aoi, 'masks')
            im_dir = os.path.join(im_top_dir, aoi, 'images_masked')
            gt_dir = os.path.join(im_top_dir, aoi, 'masks')

            im_list = sorted([z for z in os.listdir(im_dir) if z.endswith('.tif')])

            dice_one_aoi = 0.
            iou_one_aoi = 0.

            for j, f in enumerate(im_list):        
                sample_mask_name = f
                if mode == 'baseline':
                    sample_mask_path = os.path.join(pred_dir, sample_mask_name)
                elif mode == 'winner':
                    sample_mask_path = os.path.join(pred_dir, sample_mask_name).replace('.tif', '.npy')
                sample_im_path = os.path.join(im_dir, sample_mask_name)
                sample_gt_path = os.path.join(gt_dir, sample_mask_name).replace('.tif', '_Buildings.tif')

                image = skimage.io.imread(sample_im_path)
                if mode == 'baseline':
                    mask_image = skimage.io.imread(sample_mask_path)
                elif mode == 'winner':
                    mask_image = np.load(sample_mask_path)
                gt_image = skimage.io.imread(sample_gt_path)
        #         print(mask_image)
        #         print("mask_image.shape:", mask_image.shape)
        #         print("min, max, mean mask image:", np.min(mask_image), np.max(mask_image), np.mean(mask_image))
        #         print(gt_image)
        #         print("gt_image.shape:", gt_image.shape)
        #         print("min, max, mean gt image:", np.min(gt_image), np.max(gt_image), np.mean(gt_image))

                norm = (mask_image - np.min(mask_image)) / (np.max(mask_image) - np.min(mask_image))

        #         print(norm)
        #         print("mask_image.shape:", norm.shape)
        #         print("min, max, mean mask image:", np.min(norm), np.max(norm), np.mean(norm))

                gt_image = gt_image / 255

                if mode == 'winner':
                    tmp = np.zeros((1024, 1024))
                    tmp[:gt_image.shape[0],:gt_image.shape[1]] = gt_image
                    gt_image = np.repeat(tmp, 3, axis=0)
                    gt_image = np.repeat(gt_image, 3, axis=1)


                norm = np.where(norm > threshold, 1, 0)

                dice = dice_coef(gt_image, norm)
                iou = iou_coef(gt_image, norm)
                dice_one_aoi += dice
                iou_one_aoi += iou

            dice_one_aoi = dice_one_aoi / len(im_list)
            iou_one_aoi = iou_one_aoi / len(im_list)

            print("Dice: ", dice_one_aoi)
            print("IOU: ", iou_one_aoi)

            row1 = {'AOI': aoi, 'Dice': dice_one_aoi, 'IOU': iou_one_aoi, 'threshold': threshold}
            df = df.append(row1, ignore_index=True)

            dice_multi_aoi += dice_one_aoi
            iou_multi_aoi += iou_one_aoi

        dice_multi_aoi = dice_multi_aoi / len(aois)
        iou_multi_aoi = iou_multi_aoi / len(aois)

        print("Average Dice: ", dice_multi_aoi)
        print("Average IOU: ", iou_multi_aoi)
        
    df.to_csv(outfile, index=False, header=True)

raw_name = '/Midgard/home/hfang/temporal_CD/Berzelius_HRNet_SN7/berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_05_train/npy_compose'
grouped_name = '/Midgard/home/hfang/temporal_CD/Berzelius_HRNet_SN7/berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_05_train/grouped'
group(raw_name, grouped_name)

# Set prediction and image (ground truth) directories (edit appropriately)
pred_top_dir = '/Midgard/home/hfang/temporal_CD/Berzelius_HRNet_SN7/berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_05_train'
im_top_dir = '/local_storage/datasets/sn7_winner_split/test_public'

evaluate(pred_top_dir, im_top_dir, 'segmentation_tcr_v1_alpha_05.csv', mode='winner')

# Set prediction and image (ground truth) directories (edit appropriately)
pred_top_dir = '/Midgard/home/hfang/temporal_CD/Berzelius_HRNet_SN7/berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_05_train'
im_top_dir = '/local_storage/datasets/sn7_winner_split/test_public'

dice_iou_threshold(pred_top_dir, im_top_dir, 'segmentation_infer_threshold_tcr_v1_alpha_05.csv', mode='winner')