#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --constrain=belegost
#SBATCH --mail-type FAIL
#SBATCH --mail-user hfang@kth.se
#SBATCH --output /Midgard/home/hfang/temporal_CD/HRNet_SN7/slurm_logs/%J_slurm.out
#SBATCH --error  /Midgard/home/hfang/temporal_CD/HRNet_SN7/slurm_logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate solaris
cd notebooks
python scot_winner_split_finetune10.py