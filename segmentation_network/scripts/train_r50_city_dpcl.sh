#!/usr/bin/env bash
    # Example on Cityscapes  
  python train.py \
  --dataset cityscapes \
  --covstat_val_dataset cityscapes \
  --val_dataset bdd100k  \
  --arch network.deepv3_ori_prototype_dec2.DeepR50V3PlusD \
  --lr 0.005 \
  --max_iter 50000 \
  --bs_mult 8 \
  --warmup_end_epoch 9 \
  --random_select_num 50 \
  --num_big_momentum 200 \
  --marginal_param 1. \
  --hard_pixel_prop 0.333 \
  --moving_param 0.99 \
  --pretrained_autoencoder ./pretrain_ae/city/city_pretrain_ae.pth \
  --date 0101 \
  --exp r50_city_DPCL \
  --ckpt ./logs/ \
  --tb_path ./logs/
