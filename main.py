import argparse
import os
import time

import numpy as np
import scipy.io as sio
import torch

import utils
from algorithms import FL_MADDPG
from env import env_CF_MIMO_noGroup


def whiten(State, L):
    for l in range(L):
        State[l] = (State[l] - np.mean(State[l])) / np.std(State[l])
    return State


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Choose the type of the experiment
    parser.add_argument('--experiment_type', default='custom',
                        choices=['custom', 'power', 'ris_elements', 'learning_rate', 'decay'],
                        help='Choose one of the experiment types to reproduce the learning curves given in the paper')

    # Training-specific parameters
    parser.add_argument("--policy", default="FL_MADDPG", help='Algorithm (default: FL_MADDPG)')
    parser.add_argument("--env", default="CF_MIMO_noGroup", help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=47, type=int, help='Seed number for PyTorch and NumPy (default: 47)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--start_time_steps", default=0, type=int, metavar='N',
                        help='Number of exploration time steps sampling random actions (default: 0)')
    parser.add_argument("--buffer_size", default=10000, type=int,
                        help='Size of the experience replay buffer (default: 100000)')
    parser.add_argument("--batch_size", default=1024, metavar='N', help='Batch size (default: 16)')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    # Environment-specific parameters
    parser.add_argument("--num_APs", default=64, type=int, metavar='N', help='Number of APs')
    parser.add_argument("--num_antennas", default=64, type=int, metavar='N', help='Number of antennas in the BS')
    parser.add_argument("--num_RIS_elements", default=64, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=64, type=int, metavar='N', help='Number of users')
    parser.add_argument("--area_size", default=100, type=int, metavar='N', help='Size of simulation area')
    parser.add_argument("--power_limit", default=64, type=float, metavar='N',
                        help='Transmission power for the constrained optimization in dB')
    parser.add_argument("--num_time_steps_per_eps", default=100, type=int, metavar='N',
                        help='Maximum number of steps per episode (default: 10000)')
    parser.add_argument("--num_eps", default=64, type=int, metavar='N',
                        help='Maximum number of episodes (default: 5000)')
    parser.add_argument("--awgn_var", default=64, type=float, metavar='G',
                        help='Variance of the additive white Gaussian noise (default: 1e-9)')
    parser.add_argument("--channel_est_error", default=False, type=bool,
                        help='Noisy channel estimate? (default: False)')

    # Algorithm-specific parameters
    parser.add_argument("--exploration_noise", default=0.0, metavar='G', help='Std of Gaussian exploration noise')
    parser.add_argument("--discount", default=0.99, metavar='G', help='Discount factor for reward (default: 0.99)')
    parser.add_argument("--tau", default=1e-2, type=float, metavar='G',
                        help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
    parser.add_argument("--lr", default=1e-2, type=float, metavar='G',
                        help='Learning rate for the networks (default: 0.001)')
    parser.add_argument("--decay", default=1e-5, type=float, metavar='G',
                        help='Decay rate for the networks (default: 0.00001)')

    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    file_name = f"noGrouping_{args.policy}_{args.experiment_type}_{args.num_APs}_{args.num_antennas}_{args.num_RIS_elements}_{args.num_users}_{args.power_limit}_{args.lr}_{args.decay}"

    if not os.path.exists(f"./Learning Curves/{args.experiment_type}"):
        os.makedirs(f"./Learning Curves/{args.experiment_type}")

    if not os.path.exists(f"./Learning Data/{args.experiment_type}"):
        os.makedirs(f"./Learning Data/{args.experiment_type}")

    if args.save_model and not os.path.exists("./Models"):
        os.makedirs("./Models")

    env = env_CF_MIMO_noGroup.CF_MIMO_noGroup(args.num_APs, args.num_antennas, args.num_RIS_elements, args.num_users, args.area_size,
                                    args.awgn_var, args.power_limit)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_fuzzy = 4

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "L": args.num_APs,
        "M": args.num_antennas,
        "R": 1,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "power_limit": args.power_limit,
        "num_fuzzy": num_fuzzy,
        "max_action": max_action,
        "actor_lr": args.lr,
        "critic_lr": args.lr,
        "actor_decay": args.decay,
        "critic_decay": args.decay,
        "device": device,
        "discount": args.discount,
        "tau": args.tau
    }

    # # Initialize the algorithm
    # the algorithm is arg.policy
    algo = args.policy
    if algo == "FL_MADDPG":
        agent = FL_MADDPG.FL_MADDPG(**kwargs)

    # replay_buffer = utils.ExperienceReplayBuffer(args.num_APs, state_dim, action_dim, max_size=args.buffer_size)
    replay_buffer = utils.ExperienceReplayBuffer(num_fuzzy, state_dim, action_dim, max_size=args.buffer_size)
    # Initialize the instant rewards recording array
    instant_rewards = []
    instant_rewardsSE = []
    instant_rewardsEE = []
    instant_time = []

    max_reward = 0
    episode_num = 0

    for eps in range(int(args.num_eps)):
        episode_start_time = time.time()

        # replay_buffer = utils.ExperienceReplayBuffer(num_fuzzy, state_dim, action_dim, max_size=args.buffer_size)

        state = env.reset()
        state = whiten(state, args.num_APs)
        episode_reward = 0
        episode_rewardSE = 0
        episode_rewardEE = 0
        episode_opt_reward = 0
        episode_time_steps = 0

        eps_time = []

        eps_rewards = []
        eps_reward_sum = []
        eps_rewardSE = []
        eps_rewardEE = []
        a_SE = 0
        a_EE = 0

        obs_fuzzy = []
        temp = []
        while len(temp) < num_fuzzy:
            i = np.random.randint(0, int(args.num_APs) - 1)
            if i not in temp:
                temp.append(i)
                obs_fuzzy.append(state[i])
        membership = utils.updatemembership(state[0: int(args.num_APs)], obs_fuzzy)

        obs_fuzzy = np.array(obs_fuzzy)
        for t in range(int(args.num_time_steps_per_eps)):
            # Choose action from the policy

            fuzzy_action = agent.select_action(np.array(obs_fuzzy), num_fuzzy)
            action_n = np.array(utils.getagentaction(fuzzy_action, membership))

            # action_l l

            # Take the selected action

            # print(a_EE)
            next_state, reward, _ = env.step(action_n)
            new_obs_fuzzy = np.array(utils.getfuzzyobs(next_state[0: int(args.num_APs)], membership))

            fuzzy_reward = np.array(utils.getfuzzyreward(reward[0:int(args.num_APs)], membership))
            new_reward_n = fuzzy_reward
            # print(done)
            # Store data in the experience replay buffer
            # print(state.shape)
            # print(len(obs_fuzzy))
            replay_buffer.add(obs_fuzzy, fuzzy_action, new_obs_fuzzy, new_reward_n)


            # Train the agent
            agent.update_parameters(num_fuzzy, replay_buffer, args.batch_size)
            # print(replay_buffer.state.shape)
            for l in range(args.num_APs):
                eps_rewards.append(reward[l])
                episode_reward += reward[l]
            # print(episode_reward.item())
            eps_reward_sum.append(episode_reward.item())

            a_EE,a_SE = env._compute_EE_()

            episode_time = time.time() - episode_start_time
            eps_time.append(episode_time)

            print(
                f"Time step: {t + 1} Episode Num: {eps + 1} Training Time: {episode_time :.2f}s Reward: {episode_reward.item():.3f}")
            # 局部 reward _compute_EE_print(f"Time step: {t + 1} Episode Num: {episode_num + 1} Reward: {reward.sum():.3f}")(self, Phi)
            state = next_state
            obs_fuzzy = new_obs_fuzzy
            membership = utils.updatemembership(state[0: int(args.num_APs)], obs_fuzzy)


            episode_reward = 0
            episode_rewardEE = a_EE
            episode_rewardSE = a_SE
            episode_rewardSE += episode_rewardSE
            episode_rewardEE += episode_rewardEE

            print("----------- episode_rewardSE ------------")
            print(episode_rewardSE)
            print("----------- episode_rewardEE ------------")
            print(episode_rewardEE)
            state = whiten(state, args.num_APs)
            # print(replay_buffer.state.shape)
            eps_rewardSE.append(episode_rewardSE)
            eps_rewardEE.append(episode_rewardEE)

            # replay_buffer = utils.ExperienceReplayBuffer(num_fuzzy, state_dim, action_dim, max_size=args.buffer_size)
            # print(replay_buffer.state.shape)

            episode_time_steps += 1

            if t == args.num_time_steps_per_eps - 1.0:
                print(
                    f"\nTotal T: {t + 1} Episode Num: {eps + 1} Episode T: {episode_time_steps} Training T: {episode_time :.2f}s ")

                # Reset the environment
                episode_start_time = time.time()

                state  = env.reset()
                episode_reward = 0
                episode_rewardSE = 0
                episode_rewardEE = 0
                episode_opt_reward = 0
                episode_time_steps = 0
                # episode_num += 1

                state = whiten(state, args.num_APs)

                instant_rewards.append(eps_rewards)
                instant_rewardsSE.append(eps_rewardSE)
                instant_rewardsEE.append(eps_rewardEE)
                instant_time.append(eps_time)

                # np.save(f"./Learning Curves/{args.experiment_type}/{file_name}_episode_{episode_num + 1}", instant_rewards)
                sio.savemat(f"./Learning Data/{args.experiment_type}/{file_name}_episode_{episode_num + 1}_reward.mat",
                            {'instant_rewards': instant_rewards})
                sio.savemat(f"./Learning Data/{args.experiment_type}/{file_name}_episode_{episode_num + 1}_SE.mat",
                            {'instant_rewardsSE': instant_rewardsSE})
                sio.savemat(f"./Learning Data/{args.experiment_type}/{file_name}_episode_{episode_num + 1}_EE.mat",
                            {'instant_rewardsEE': instant_rewardsEE})
                sio.savemat(f"./Learning Data/{args.experiment_type}/{file_name}_episode_{episode_num + 1}_time.mat",
                            {'instant_time': instant_time})


    # FL_MADDPG.FL_MADDPG.save("model")