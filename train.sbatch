#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 4
#SBATCH --constrain=balrog
#SBATCH --mail-type FAIL
#SBATCH --mail-user hfang@kth.se
#SBATCH --output /Midgard/home/hfang/temporal_CD/HRNet_SN7/slurm_logs/%J_hrnet_tcr_train.out
#SBATCH --error  /Midgard/home/hfang/temporal_CD/HRNet_SN7/slurm_logs/%J_hrnet_tcr_train.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate solaris
./run_local.sh