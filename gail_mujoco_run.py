import os
import argparse
from itertools import product

# TRAJ_LIMITATIONS = {
#     "Hopper-v1": [4, 11, 18, 25],
#     "HalfCheetah-v1": [4, 11, 18, 25],
#     "Walker2d-v1": [4, 11, 18, 25],
#     "Ant-v1": [4, 11, 18, 25],
#     "Humanoid-v1": [80, 160, 240],
# }

TRAJ_LIMITATIONS = {
    "Hopper-v1": [4],
    "HalfCheetah-v1": [4],
    "Walker2d-v1": [4],
    "Ant-v1": [4],
    "Humanoid-v1": [80],
}


TIMESTEPS_PER_BATCHS = {
    "Hopper-v1": [1000],
    "HalfCheetah-v1": [1000],
    "Walker2d-v1": [1000],
    "Ant-v1": [5000],
    "Humanoid-v1": [5000],
}


NUM_EPOCHS = {
    "Hopper-v1": [2000],
    "HalfCheetah-v1": [4000],
    "Walker2d-v1": [2000],
    "Ant-v1": [2000],
    "Humanoid-v1": [3000],
}


def run_job(hyper_keys, hyper_values, gpu_id):
    BASH = "CUDA_VISIBLE_DEVICES={} python baselines/gail/run_mujoco.py".format(gpu_id)
    for key, value in zip(hyper_keys, hyper_values):
        if 'obs_normalize' in key or 'pretrained' in key:
            if value:
                BASH += " --{0}".format(key)
            else:
                BASH += " --no-{0}".format(key)
        else:
            BASH += " --{0} {1}".format(key, value)
    os.system(BASH)


if __name__ == "__main__":
    # User Guide
    # step 0: modify BASH (from line 20 to line 22) to your BASH
    # step 1: modify grid-search argument information (from line 43 to line 45).
    # !!! Kind remind that the key of hyper_dict you modify should be aligned with the argument name in your code :)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", type=int, default=0, help="which gpu card you want to use.")
    parser.add_argument("-e", "--env_id", type=str, default="Hopper-v1", help="which env you want to run.")
    args = parser.parse_args()

    algo_hyper_dict = {}
    # algo_hyper_dict['env_id'] = ["Hopper-v1", "HalfCheetah-v1", "Walker2d-v1", "Ant-v1"]
    # algo_hyper_dict['env_id'] = ["Humanoid-v1",]
    # algo_hyper_dict['traj_limitation'] = [4, 11, 18, 25]
    # algo_hyper_dict['traj_limitation'] = [25]
    # algo_hyper_dict['traj_limitation'] = [80, 160, 240]

    algo_hyper_dict['env_id'] = [args.env_id]
    algo_hyper_dict['traj_limitation'] = TRAJ_LIMITATIONS[args.env_id]
    algo_hyper_dict['subsample_freq'] = [20]
    algo_hyper_dict['num_epoch'] = NUM_EPOCHS[args.env_id]
    algo_hyper_dict['timesteps_per_batch'] = TIMESTEPS_PER_BATCHS[args.env_id]
    algo_hyper_dict['obs_normalize'] = [True]
    algo_hyper_dict['seed'] = [0, 1, 2, 3, 4]
    algo_hyper_keys = list(algo_hyper_dict.keys())

    for variant in product(*algo_hyper_dict.values()):
        jobname_prefix = "run_bc"
        jobname = jobname_prefix
        for index, value in enumerate(variant):
            jobname += "_{0}-{1}".format(algo_hyper_keys[index], value)

        # run job
        run_job(algo_hyper_keys, variant, args.gpu_id)


