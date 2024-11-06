#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/train_trans_kit.sh
# screen -L -Logfile .5GPT.5cond.5 -S temp /home/epinyoan/git/momask-codes/experiments/train_trans_kit.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask
name='3_.5GPT.5cond0-.5Batch__evalEveryOther2iter'
python train_t2m_transformer.py \
    --name ${name} \
    --gpu_id 1 \
    --dataset_name kit \
    --milestones 50000 80000 \
    --batch_size 64 \
    --max_epoch 2000 \
    --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2_k \
    --trans official



sleep 500
