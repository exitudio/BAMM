#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/eval_res_trans_kit.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask

python eval_t2m_trans_res.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw_k \
    --dataset_name kit \
    --name 2024-02-19-22-08-55_0_kit \
    --which_epoch net_best_fid.tar \
    --gpu_id 2 \
    --cond_scale 2 \
    --time_steps 10 \
    --ext 100_rebuttal_NoRes1Iter

sleep 500
# --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
# --commit 0.01 \

# MoMask pretrain
# --gamma 0.05 \
# --lr 2e-4 \
# --num_quantizers 6 \
