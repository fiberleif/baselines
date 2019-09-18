#!/bin/bash
SEEDS="0 1 2 3 4"
for SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=0 python baselines/gail/run_mujoco.py --env_id Hopper-v1 --expert_path "dataset/hopper.npz" --num_epoch 1000 --seed $SEED
done
