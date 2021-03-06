'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
import numpy as np
import gym
import os
from mpi4py import MPI
from tqdm import tqdm
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    # Environment Configuration
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--max_path_length', help='Max path length', type=int, default=1000)
    # parser.add_argument('--delay_freq', help='Delay frequency', type=int, default=10)
    parser.add_argument('--expert_path', type=str, default='dataset/hopper.npz')
    # Task Configuration
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate Configuration
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    # ------------------------------------------------------------------------------------------------------------------
    # Train Configuration
    # Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--subsample_freq', type=int, default=20)
    # Optimization Configuration
    parser.add_argument('--timesteps_per_batch', help='number of timesteps in each batch', type=int, default=1000)
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=1)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=5)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    boolean_flag(parser, 'gaussian_fixed_var', default=False, help='use the fixed var for each state')
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    boolean_flag(parser, 'obs_normalize', default=False, help='whether to perform obs normalization in the policy')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Training Configuration
    parser.add_argument('--num_epochs', help='Number of training epochs', type=int, default=2e3)
    parser.add_argument('--evaluation_freq', help='Number of updates to evaluate', type=int, default=10)
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    # ------------------------------------------------------------------------------------------------------------------
    return parser.parse_args()


def get_task_name(args):
    task_name = "run_gail_env_" + args.env_id
    if args.pretrained:
        task_name += "_with_pretrained"
    if args.obs_normalize:
        task_name += "_with_obs_normalize"
    if args.traj_limitation != np.inf:
        task_name += "_traj_limitation_" + str(args.traj_limitation)
        task_name += "_subsample_freq_" + str(args.subsample_freq)
    task_name = task_name + "_g_step_" + str(args.g_step) + "_d_step_" + str(args.d_step) + \
        "_policy_entcoeff_" + str(args.policy_entcoeff) \
        + "_adversary_entcoeff_" + str(args.adversary_entcoeff) \
        + "_timesteps_per_batch_" + str(args.timesteps_per_batch) + "_gaussian_fixed_var_" + str(args.gaussian_fixed_var)
    task_name += "_seed_" + str(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    # env = DelayRewardWrapper(env, args.delay_freq, args.max_path_length)
    eval_env = gym.make(args.env_id)

    logger.configure(os.path.join("log", "GAIL", args.env_id, "subsample_{}".format(args.subsample_freq),
                                  "traj_{}".format(args.traj_limitation), "seed_{}".format(args.seed)))

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2,
                                    gaussian_fixed_var=args.gaussian_fixed_var, obs_normalize=args.obs_normalize)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, "GAIL", task_name)

    if args.task == 'train':
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, data_subsample_freq=args.subsample_freq)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff, obs_normalize=args.obs_normalize)
        train(env,
              eval_env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              args.num_epochs,
              args.evaluation_freq,
              args.timesteps_per_batch,
              task_name,
              )
    elif args.task == 'evaluate':
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=args.timesteps_per_batch,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, eval_env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, num_epochs, evaluation_freq, timesteps_per_batch,
          task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    if algo == 'trpo':
        from baselines.gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, eval_env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=timesteps_per_batch,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3,
                       num_epochs=num_epochs,
                       evaluation_freq=evaluation_freq,
                       task_name=task_name)
    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    # U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    # U.load_state(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))

    output_infos = {"avg_return": np.mean(ret_list),
                    "std_return": np.std(ret_list),
                    "max_return": np.max(ret_list),
                    "min_return": np.min(ret_list),}

    return output_infos


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    args.num_epochs = int(args.num_epochs)
    args.expert_path = 'dataset/{}.npz'.format(args.env_id).lower().replace("-v1", "")  # set expert path
    main(args)
