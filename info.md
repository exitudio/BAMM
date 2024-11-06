```
conda activate momask
python train_vq.py --name rvq_name_50ep --gpu_id 4 --dataset_name t2m --batch_size 512 --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2
```

python train_vq.py --name mmm_lfq_ent.1 --gpu_id 1 --dataset_name t2m --batch_size 512 --num_quantizers 6  --max_epoch 100 --quantize_dropout_prob 0.2