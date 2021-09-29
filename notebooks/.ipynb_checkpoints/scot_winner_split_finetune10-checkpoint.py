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

ground_truth['geometry'] = ground_truth['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(ground_truth, geometry = 'geometry')
print(gdf)

pred = pd.read_csv('/Midgard/home/hfang/temporal_CD/HRNet_SN7_csv/submission_hrnet_finetune10.csv')
aois = [z.split('mosaic_')[-1] for z in pred['filename'].values]
pred['aoi'] = aois
timesteps = [z.split('monthly_')[-1].split('_mosaic')[0] for z in pred['filename'].values]
pred['timestep'] = timesteps
pred = pred.drop('filename', 1)

pred['geometry'] = pred['geometry'].apply(wkt.loads)
pdf = gpd.GeoDataFrame(pred, geometry = 'geometry')
print(pdf)

scot.scot_multi_aoi(gdf, pdf, verbose=True)