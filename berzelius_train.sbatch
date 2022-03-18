#!/bin/bash
#SBATCH --gpus 8
#SBATCH -t 3-00:00:00
#SBATCH -A berzelius-2022-50

nvidia-smi
module load Anaconda/2021.05-nsc1
conda activate /proj/berzelius-2021-54/users/HRNet-Semantic-Segmentation/mycondaenv/

PYTHON="/proj/berzelius-2021-54/users/HRNet-Semantic-Segmentation/mycondaenv/bin/python"
GPU_NUM=8
# CONFIG="berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_1_train"
# CONFIG="berzelius_seg_hrnet_tcr_version_2_w48_512x512_sgd_lr1e-3_wd1e-4_bs_48_epoch70_alpha_05_train"
# CONFIG="berzelius_seg_hrnet_tcr_version_2_w48_512x512_sgd_lr1e-3_wd1e-4_bs_48_epoch70_alpha_01_theta1_08_theta2_01_theta3_01_train"
# CONFIG="berzelius_seg_hrnet_tcr_version_3_w48_512x512_sgd_lr1e-3_wd1e-4_bs_48_epoch140_alpha_01_theta1_08_theta2_01_theta3_01_train"
# CONFIG="berzelius_seg_hrnet_tcr_version_4_w48_512x512_sgd_lr1e-3_wd1e-4_bs_48_epoch200_alpha_01_theta1_08_theta2_01_theta3_01_train"
# CONFIG="berzelius_seg_hrnet_tcr_version_4_w48_512x512_sgd_lr1e-3_wd1e-4_bs_48_epoch200_alpha_01_theta1_1_theta2_0_theta3_0_train"
# CONFIG="berzelius_seg_hrnet_tcr_version_4_w48_512x512_adam_lr1e-3_bs_48_epoch200_alpha_02_theta1_085_theta2_005_theta3_01_train"
# CONFIG="berzelius_seg_hrnet_tcr_version_4_w48_512x512_adam_lr1e-2_bs_48_epoch200_alpha_02_theta1_085_theta2_005_theta3_01_train"
CONFIG="berzelius_seg_hrnet_tcr_version_4_w48_512x512_adam_lr1e-4_bs_48_epoch200_alpha_02_theta1_085_theta2_005_theta3_01_train"

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        tools/train.py \
        --cfg experiments/spacenet7/$CONFIG.yaml
        2>&1 | tee local_log_spacenet7_alpha_01_version_4_param1.txt