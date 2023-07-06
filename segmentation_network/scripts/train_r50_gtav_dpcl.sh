#!/usr/bin/env bash
    # Example on Cityscapes  
  python train.py \
  --dataset gtav \
  --covstat_val_dataset gtav \
  --val_dataset bdd100k cityscapes mapillary \
  --arch network.deepv3_ori_prototype_dec2.DeepR50V3PlusD \
  --lr 0.001 \
  --max_iter 40000 \
  --bs_mult 8 \
  --warmup_end_epoch 9 \
  --random_select_num 30 \
  --num_big_momentum 100 \
  --marginal_param 0.5 \
  --hard_pixel_prop 0.5 \
  --moving_param 0.999 \
  --pretrained_autoencoder ./pretrain_ae/gtav/gtav_pretrain_ae.pth \
  --date 0101 \
  --exp r50_gtav_DPCL \
  --ckpt ./logs/ \
  --tb_path ./logs/
