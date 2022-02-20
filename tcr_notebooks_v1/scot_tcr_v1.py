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
import geopandas as gpd
from shapely import wkt

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib
# matplotlib.use('Agg') # non-interactive

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
import solaris.eval.scot as scot

ground_truth = pd.read_csv('/Midgard/home/hfang/temporal_CD/HRNet_SN7_csv/ground_truth.csv')
aois = [z.split('mosaic_')[-1] for z in ground_truth['filename'].values]
ground_truth['aoi'] = aois
timesteps = [z.split('monthly_')[-1].split('_mosaic')[0] for z in ground_truth['filename'].values]
ground_truth['timestep'] = timesteps
ground_truth = ground_truth.drop('filename', 1)

# ground_truth['geometry'] = ground_truth['geometry'].apply(wkt.loads)
# gdf = gpd.GeoDataFrame(ground_truth, geometry = 'geometry')
# print(gdf)

# pred = pd.read_csv('/Midgard/home/hfang/temporal_CD/HRNet_SN7/submission_tcr_v1_alpha_05_finetune1.csv')
# pred = pd.read_csv('/Midgard/home/hfang/temporal_CD/HRNet_SN7/submission_tcr_v1_alpha_09_finetune1.csv')
pred = pd.read_csv('/Midgard/home/hfang/temporal_CD/HRNet_SN7/submission_tcr_v1_alpha_1_finetune1.csv')
aois = [z.split('mosaic_')[-1] for z in pred['filename'].values]
pred['aoi'] = aois
timesteps = [z.split('monthly_')[-1].split('_mosaic')[0] for z in pred['filename'].values]
pred['timestep'] = timesteps
pred = pred.drop('filename', 1)

# pred['geometry'] = pred['geometry'].apply(wkt.loads)
# pdf = gpd.GeoDataFrame(pred, geometry = 'geometry')
# print(pdf)

# scot.scot_multi_aoi(gdf, pdf, verbose=True)

aois = sorted(list(ground_truth.aoi.drop_duplicates()))
print(aois)
print(len(aois))

# Record the track score, change score and combined score (to csv)
df = pd.DataFrame()

cumulative_score = 0.
all_stats = {}

for i, aoi in enumerate(aois):
    print()
    print('%i / %i: AOI %s' % (i + 1, len(aois), aoi))
    grnd_df_one_aoi = ground_truth.loc[ground_truth.aoi == aoi].copy()
    prop_df_one_aoi = pred.loc[pred.aoi == aoi].copy()
    
    grnd_df_one_aoi['geometry'] = grnd_df_one_aoi['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(grnd_df_one_aoi, geometry = 'geometry')
    # print(gdf)
    
    prop_df_one_aoi['geometry'] = prop_df_one_aoi['geometry'].apply(wkt.loads)
    pdf = gpd.GeoDataFrame(prop_df_one_aoi, geometry = 'geometry')
    # print(pdf)

    score_one_aoi, stats_one_aoi = scot.scot_one_aoi(gdf, pdf, threshold=0.25, base_reward=100., beta=2., stats=True, verbose=True)
    cumulative_score += score_one_aoi
    all_stats[aoi] = stats_one_aoi
    
    # Record the score of each AOI
    row1 = {'AOI': aoi, 'Track Score': stats_one_aoi[4], 'Change Score': stats_one_aoi[8], 'Combined Score': score_one_aoi}
    df = df.append(row1, ignore_index=True)

# df.to_csv('scot_alpha_05_finetune1.csv', index=False, header=True)
# df.to_csv('scot_alpha_09_finetune1.csv', index=False, header=True)
df.to_csv('scot_alpha_1_finetune1.csv', index=False, header=True)

# Return combined SCOT metric score
score = cumulative_score / len(aois)
print('Overall score: %f' % score)
print(score)
print(all_stats)