#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/train_trans.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask
name='101_8192x32_latest'
python train_t2m_transformer.py \
    --name ${name} \
    --gpu_id 2 \
    --dataset_name t2m \
    --milestones 50000 80000 \
    --batch_size 512 \
    --max_epoch 2000 \
    --vq_name 2024-05-13-20-41-19_100_8192x32_reset5 \
    --trans official



sleep 500
