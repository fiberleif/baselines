# Generative Adversarial Imitation Learning (GAIL)

- Original paper: https://arxiv.org/abs/1606.03476


## Guide on running BC/GAIL Baselines

### Step 1: Set expert data (demonstrations)

 ```bash
mkdir dataset
cp -r ~/demo_dataset dataset/
```

### Step 2: Run Behavior Clone
```bash
python baselines/gail/behavior_clone.py --env_id Hopper-v1
```
tips: use ```CUDA_VISIBLE_DEVICES=```

### Step 2: Run GAIL

Run with single thread:

```bash
python -m baselines.gail.run_mujoco
```

Run with multiple threads:

```bash
mpirun -np 16 python -m baselines.gail.run_mujoco
```

See help (`-h`) for more options.

#### In case you want to run Behavior Cloning (BC)

```bash
python -m baselines.gail.behavior_clone
```

See help (`-h`) for more options.


## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/openai/baselines/pulls.

## Maintainers

- Yuan-Hong Liao, andrewliao11_at_gmail_dot_com
- Ryan Julian, ryanjulian_at_gmail_dot_com

## Others

Thanks to the open source:

- @openai/imitation
- @carpedm20/deep-rl-tensorflow
