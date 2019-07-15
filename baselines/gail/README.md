# Generative Adversarial Imitation Learning (GAIL)

- Original paper: https://arxiv.org/abs/1606.03476


## Guide on Running BC/GAIL Baselines

### Step 1: Prepare Expert Data

 ```bash
mkdir dataset
cp ~/demo_dataset/* dataset/
```

### Step 2: Run Behavior Clone
```bash
CUDA_VISIBLE_DEVICES=0 python baselines/gail/behavior_clone.py --env_id Hopper-v1
```

Tips: if you want to run BC in other tasks, you only need to change the argument of env_id: ```--env_id```.

See help (`-h`) for more options.

### Step 3: Run GAIL
```bash
CUDA_VISIBLE_DEVICES=0 python baselines/gail/run_mujoco.py --env_id Hopper-v1
```

Tips: if you want to run GAIL in other tasks, you only need to change the argument of env_id: ```--env_id```.

See help (`-h`) for more options.

### Step 4: Script for Building BC & GAIL Baselines
```bash
python bc_mujoco_run.py --gpu_id 0 # run BC baseline
python gail_mujoco_run.py --gpu_id 1 # run GAIL baseline  
```
Tips: we can use ```-g``` to replace ```--gpu_id``` here for convenience.

## Maintainers

- Yuan-Hong Liao, andrewliao11_at_gmail_dot_com
- Ryan Julian, ryanjulian_at_gmail_dot_com

## Others

Thanks to the open source:

- @openai/imitation
- @carpedm20/deep-rl-tensorflow
