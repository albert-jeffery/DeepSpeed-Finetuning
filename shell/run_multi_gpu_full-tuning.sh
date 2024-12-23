#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6 --master_port 12345 train.py \
          --model_name_or_path THUDM/chatglm3-6b \
          --src_len 512 \
          --tgt_len 128 \
          --data_path ./data/test.jsonl \
          --train_micro_batch_size_per_gpu 1\
          --gradient_accumulation_steps 16 \
          --max_lr 1e-4 \
          --initial_lr 1e-6 \
          --min_lr 1e-8 \
          --weight_decay 0.01 \
          --adam_beta1 0.9 \
          --adam_beta2 0.999\
          --epochs 1 \
          --output_dir ./checkpoints \
          --finetune_method full-tuning \
          --ds_config_path ./config/dsconfig.json \
          --offload_device cpu \
          --nvme_path ./mnt/nvme