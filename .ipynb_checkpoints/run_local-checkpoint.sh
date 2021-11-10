PYTHON="/Midgard/home/hfang/miniconda3/envs/solaris/bin/python"
GPU_NUM=1
# CONFIG="seg_hrnet_w48_512x512_sgd_lr1e-2_wd4e-5_bs_16_epoch70"
# CONFIG="seg_hrnet_w48_512x512_sgd_lr1e-3_wd1e-4_bs_16_epoch200_train"
CONFIG="seg_hrnet_tcr_w48_512x512_sgd_lr1e-3_wd1e-4_bs_16_epoch200_train"

# $PYTHON -m pip install -r requirements.txt

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        tools/train.py \
        --cfg experiments/spacenet7/$CONFIG.yaml \
        2>&1 | tee local_spacenet7.txt

# $PYTHON tools/test.py --cfg experiments/spacenet7/$CONFIG.yaml