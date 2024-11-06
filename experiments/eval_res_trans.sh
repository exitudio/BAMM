#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/BAMM/experiments/eval_res_trans.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/BAMM
conda activate momask

python eval_t2m_trans_res.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
    --name 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd \
    --gpu_id 1 \
    --ext LOG_NAME

sleep 500
# --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
# --commit 0.01 \

# MoMask pretrain
# --gamma 0.05 \
# --lr 2e-4 \
# --num_quantizers 6 \
# --name 2024-05-14-08-48-38_101_8192x32_latest \