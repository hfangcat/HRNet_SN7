#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 3-00:00:00
#SBATCH -A berzelius-2021-54

nvidia-smi
module load Anaconda/2021.05-nsc1
conda activate /proj/berzelius-2021-54/users/HRNet-Semantic-Segmentation/mycondaenv/

PYTHON="/proj/berzelius-2021-54/users/HRNet-Semantic-Segmentation/mycondaenv/bin/python"
GPU_NUM=1
# CONFIG="berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_1_train"
# CONFIG="berzelius_seg_hrnet_tcr_version_2_w48_512x512_sgd_lr1e-3_wd1e-4_bs_48_epoch70_alpha_01_theta1_08_theta2_01_theta3_01_train"
CONFIG="berzelius_seg_hrnet_tcr_version_4_w48_512x512_adam_lr1e-4_bs_48_epoch200_alpha_02_theta1_085_theta2_005_theta3_01_train"

$PYTHON tools/test.py --cfg experiments/spacenet7/$CONFIG.yaml