#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/train_res_trans.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask

python train_res_transformer.py \
    --name 1_GPT_2iter  \
    --gpu_id 2 \
    --dataset_name t2m \
    --batch_size 64 \
    --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
    --trans 2024-02-10-11-15-28_6_GPT_ref-end-3_cond.5rand0-.5_lossAllToken_noPredEnd_addTxtCondFIXED_eval2Iter \
    --checkpoints_dir ./log/res \
    --cond_drop_prob 0.2 \
    --share_weight

sleep 500
