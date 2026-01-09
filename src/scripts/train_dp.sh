#!/bin/bash
# train_dp.sh

DATA_PATH="../../data/grasp_soda/202601061638-Aboslute"
REPO_ID="202601061638-Aboslute"

# python third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
python src/lerobot/scripts/lerobot_train.py \
    --policy.type=diffusion \
    --dataset.repo_id=$REPO_ID \
    --dataset.root=$DATA_PATH \
    --dataset.image_transforms.enable=true \
    --output_dir=outputs/train/grasp_soda/dp_$(date +%Y%m%d_%H%M%S) \
    --batch_size=64 \
    --num_workers=8 \
    --steps=100000 \
    --eval_freq=0 \
    --save_freq=10000 \
    --wandb.enable=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \