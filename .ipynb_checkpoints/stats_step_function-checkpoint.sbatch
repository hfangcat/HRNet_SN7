#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain=balrog
#SBATCH --mail-type FAIL
#SBATCH --mail-user hfang@kth.se
#SBATCH --output /Midgard/home/hfang/temporal_CD/HRNet_SN7/slurm_logs/%J_slurm.out
#SBATCH --error  /Midgard/home/hfang/temporal_CD/HRNet_SN7/slurm_logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate solaris
cd tcr_notebooks_v1
python stats_step_function.py /Midgard/home/hfang/temporal_CD/Berzelius_HRNet_SN7/berzelius_seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_64_epoch70_alpha_1_train