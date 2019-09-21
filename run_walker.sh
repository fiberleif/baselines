#!/bin/bash
SEEDS="0 1"
for SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=0 python baselines/gail/run_mujoco.py --env_id Walker2d-v1 --expert_path "dataset/walker.npz" --num_epoch 1000 --traj_limitation 11 --reward_coeff 10.0 --seed $SEED
done
