#!/bin/bash
SEEDS="0 1"
for SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=3 python baselines/gail/run_mujoco.py --env_id Humanoid-v1 --expert_path "dataset/humanoid.npz" --num_epoch 2000 --reward_coeff 1.0 --seed $SEED
done