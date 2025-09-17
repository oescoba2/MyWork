from numpy.typing import ArrayLike, NDArray
from torch import optim
from tqdm import tqdm
from utils import get_epsilon
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn

def QLearning(env:gym.Env, α=0.1, γ=0.6, ε=0.2, N=70_000, decay=False) -> tuple[NDArray, dict[str:ArrayLike], ArrayLike]:
    """Use Q-learning algorithm to estimate q*(s,a) values of a given 
    environment.

    Parameters:
        - env (gym.Env) : the discrete gym environment object.
        - α (float) : learning rate. Default is 0.1.
        - γ (float) : discount factor. Default is 0.6.
        - ε (float) : epsilon value for epsilon-greedy algo. Default is 0.2.
        - N (int) : number of episodes to train for. Default is 70_000.
        - decay (bool) : whether to decay epsilon according to epsilon_decay.
                         Default is False.
    Returns:
        - q_table (NDArray) : the Q(s,a) approximation values stored in
                              nxm array.
        - training_stats (dict) : a dict whose keys are strings and values
                                  are arrays storing the rewards, timesteps,
                                  and elapsed_time of each episode experienced
                                  during the training.
        - episodes (ArrayLike) : an array of ints containing all the episode
                                 numbers.
    """

    # Make Q-table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    training_stats = {"Reward" : np.zeros((N, )),
                      "Time steps" : np.zeros((N, ))}
    episodes = np.arange(start=1, stop=N+1, dtype=int)
    pbar = tqdm(range(1, N+1), desc='Training')

    # Train for N episodes
    for i in episodes:

        # Get epsilon value
        if decay:
            ε = get_epsilon(episode_num=i, num_episodes_decay=N)

        # Reset env and get initial state; Initialize penalties, reward, done
        curr_state, info = env.reset()
        reward = 0
        done = False
        trunc = False

        # Keep going until the terminal state is reached
        while not done and not trunc:

            # Employ epsilon-greedy algo
            if random.uniform(0, 1) < ε: 
                curr_action = env.action_space.sample()
            else:                             
                curr_action = (q_table[curr_state]).argmax()

            # Take action and get new state and reward
            next_state, reward, done, trunc, info = env.step(curr_action)

            # Calculate new qvalue
            old_value = q_table[curr_state, curr_action]
            next_max = (q_table[next_state]).max()
            new_value = (1 - α) * old_value + α * (reward + γ * next_max)
            q_table[curr_state, curr_action] = new_value

            # Get next observation
            curr_state = next_state

        training_stats["Reward"][i-1] = info['episode']['r']
        training_stats['Time steps'][i-1] = info['episode']['l']
        pbar.set_postfix(episodic_reward=info["episode"]['r'], timesteps=info["episode"]['l'])
        pbar.update()

    env.close()

    return q_table, training_stats, episodes
