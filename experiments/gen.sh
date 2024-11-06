#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/BAMM/experiments/gen.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/BAMM
conda activate momask

python gen_t2m.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
    --name 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd \
    --text_prompt "the person crouches and walks forward." \
    --motion_length -1 \
    --repeat_times 2 \
    --gpu_id 1 \
    --ext generation_name_nopredlen


sleep 500
# --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
# --commit 0.01 \

# MoMask pretrain
# --gamma 0.05 \
# --lr 2e-4 \
# --num_quantizers 6 \
# --name 2024-05-14-08-48-38_101_8192x32_latest \