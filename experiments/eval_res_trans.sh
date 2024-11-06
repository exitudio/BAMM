#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/eval_res_trans.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask

python eval_t2m_trans_res.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
    --dataset_name t2m \
    --name 2024-05-14-08-45-09_101_8192x32 \
    --which_epoch latest.tar \
    --gpu_id 2 \
    --cond_scale 4 \
    --time_steps 10 \
    --ext 1_latest_GPT_full

sleep 500
# --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
# --commit 0.01 \

# MoMask pretrain
# --gamma 0.05 \
# --lr 2e-4 \
# --num_quantizers 6 \
# --name 2024-05-14-08-48-38_101_8192x32_latest \