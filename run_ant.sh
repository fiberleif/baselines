#!/bin/bash
SEEDS="0 1"
for SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=2 python baselines/gail/run_mujoco.py --env_id Ant-v1 --expert_path "dataset/ant.npz" --num_epoch 2000 --traj_limitation 11 --reward_coeff 10.0 --seed $SEED
done