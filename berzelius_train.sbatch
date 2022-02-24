#!/bin/bash
#SBATCH --gpus 8
#SBATCH -t 18:00:00

nvidia-smi
module load Anaconda/2021.05-nsc1
conda activate /proj/berzelius-2021-54/users/HRNet-Semantic-Segmentation/mycondaenv/

PYTHON="/proj/berzelius-2021-54/users/HRNet-Semantic-Segmentation/mycondaenv/bin/python"
GPU_NUM=8
# CONFIG="berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_1_train"
CONFIG="berzelius_seg_hrnet_tcr_version_2_w48_512x512_sgd_lr1e-3_wd1e-4_bs_48_epoch70_alpha_05_train"

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        tools/train.py \
        --cfg experiments/spacenet7/$CONFIG.yaml
        2>&1 | tee local_log_spacenet7_alpha_05_version_2.txt