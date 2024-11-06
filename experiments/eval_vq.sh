#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/eval_vq.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask
python eval_t2m_vq.py \
    --gpu_id 0 \
    --name 2024-01-08-15-24-24_momask_rerun_lr2e-4_bestFID_resetEvryIter \
    --dataset_name t2m \
    --ext rvq_nq6

sleep 500
# --commit 0.01 \

# MoMask pretrain
# --gamma 0.05 \
# --lr 2e-4 \
# --num_quantizers 6 \
