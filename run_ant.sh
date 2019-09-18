#!/bin/bash
SEEDS="0 1 2 3 4"
for SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=2 python baselines/gail/run_mujoco.py --env_id Ant-v1 --expert_path "dataset/ant.npz" --num_epoch 2000 --seed $SEED
done