#!/bin/bash

SEEDS="0 1 2 3 4"
for SEED in $SEEDS
do
	python baselines/gail/run_mujoco.py --env_id Humanoid-v1 --seed $SEED --num_epochs 4000 --timesteps_per_batch 5000 --evaluation_freq 4 --expert_path dataset/humanoid.npz --no-gaussian_fixed_var
done