import os
import argparse
from itertools import product

TIMESTEPS_PER_BATCHS = {
   "Hopper-v1": 1000,
    "HalfCheetah-v1": 1000,
    "Walker2d-v1": 1000,
    "Ant-v1": 5000,
    "Humanoid-v1": 5000,
}


NUM_EPOCHS = {
    "Hopper-v1": 2000,
    "HalfCheetah-v1": 4000,
    "Walker2d-v1": 2000,
    "Ant-v1": 2000,
    "Humanoid-v1": 3000,
}


def run_job(hyper_keys, hyper_values, gpu_id):
    BASH = "CUDA_VISIBLE_DEVICES={} python baselines/gail/run_mujoco.py".format(gpu_id)
    for key, value in zip(hyper_keys, hyper_values):
        BASH += " --{0} {1}".format(key, value)
    BASH += "--num_epochs {}".format(NUM_EPOCHS[hyper_values[0]])
    BASH += "--timesteps_per_batch".format(TIMESTEPS_PER_BATCHS[hyper_values[0]])
    os.system(BASH)


if __name__ == "__main__":
    # User Guide
    # step 0: modify BASH (from line 20 to line 22) to your BASH
    # step 1: modify grid-search argument information (from line 43 to line 45).
    # !!! Kind remind that the key of hyper_dict you modify should be aligned with the argument name in your code :)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", type=int, default=0, help="which gpu card you want to use.")
    args = parser.parse_args()

    algo_hyper_dict = {}
    algo_hyper_dict['env_id'] = ["Hopper-v1", "HalfCheetah-v1", "Walker2d-v1", "Ant-v1"]
    # algo_hyper_dict['env_id'] = ["Humanoid-v1",]
    # algo_hyper_dict['traj_limitation'] = [4, 11, 18, 25]
    algo_hyper_dict['traj_limitation'] = [25]
    # algo_hyper_dict['traj_limitation'] = [80, 160, 240,]
    algo_hyper_dict['subsample_freq'] = [1, 20]
    # algo_hyper_dict['seed'] = [0, 1, 2]
    algo_hyper_keys = list(algo_hyper_dict.keys())

    for variant in product(*algo_hyper_dict.values()):
        jobname_prefix = "run_bc"
        jobname = jobname_prefix
        for index, value in enumerate(variant):
            jobname += "_{0}-{1}".format(algo_hyper_keys[index], value)

        # run job
        run_job(algo_hyper_keys, variant, args.gpu_id)


