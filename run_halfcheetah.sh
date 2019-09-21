#!/bin/bash
SEEDS="0 1"
for SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=0 python baselines/gail/run_mujoco.py --env_id HalfCheetah-v1 --expert_path "dataset/half_cheetah.npz" --num_epoch 1000 --seed $SEED
done