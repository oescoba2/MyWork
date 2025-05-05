from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from networks import PolicyNetwork
from numpy.typing import ArrayLike
from torch import nn
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import os
import re
import time
import torch
import warnings

class GAE():
    """Define a class to compute generalized advantage estimates (GAE).
    
    Attributes:
        - gamma : discount factor.
        - lamb : variance control factor.

    Hidden Methods:
        - __init__ : the class constructor.
        - __call__ : method to allow a class instance to be used as a function.
                  
    """

    def __init__(self, gamma:float=0.99, lamb:float=0.95) -> None:
        """Define the hyperparameters of GAE.
        
        Parameters:
            - gamma (float) : the discount factor to use with future rewards. 
                              Defaulted to 0.99.
            - lamb (float) : the variance controlling weight used to balance 
                             the bias-variance tradeoff when averaging n-step
                             advantage estimates. Defaulted to 0.95.
        
        Returns:
            - None
        """

        if gamma < 0 and gamma > 1:
            raise ValueError(f"Expected a discount factor between 0 and 1. Got {gamma}")
        if lamb < 0 and lamb > 1:
            raise ValueError(f"Expected a variance controlling weight between 0 and 1. Got {lamb}")
        
        self.γ = gamma
        self.λ = lamb

    def __call__(self, rewards:ArrayLike, state_vals:ArrayLike, dones:ArrayLike) -> ArrayLike:
        """Compute the advantage estimates, A(s, a), using GAE.

        Parameters:
            - rewards (ArrayLike; (T, )) : an ArrayLike object containing the rewards
                                           obtained in the trajectory. Must be of shape
                                           of shape (T, )
            - state_vals (ArrayLike; (T+1, )) : an ArrayLike object containing the values
                                                of the states visted in a trajectory. Must
                                                be of shape (T+1, ).
            - dones (ArrayLike; (T, )) : an ArrayLike of boolean values denoting whether the
                                         visited state was a terminal state or not. Must be 
                                         of shape (T, ).

        Returns:
            - A (ArrayLike; (T, )) : the computed advantages of actions in the given states
                                     of the trajectory. 
        """

        T = len(rewards)
        A = []
        last_adv = 0

        if len(state_vals) != (T+1):
            raise ValueError(f'Expected state_vals to have shape ({T+1},) but got shape ({len(state_vals)},)')
        if len(dones) != T:
            raise ValueError(f"Expected done arrat to have shape ({T+1},) but got shape ({len(dones)},)")

        # Compute A_t = δ_t + γ * λ * A_{t+1}
        for t in range(T-1, -1, -1):

            δ = rewards[t] + self.γ * state_vals[t + 1] * (1.0 - dones[t]) - state_vals[t]
            last_adv = δ + self.γ * self.λ * (1.0 - dones[t]) * last_adv
            A += [last_adv]

        return np.array(A[::-1])

def get_env(env_name:str, vid_dir:str, vid_name_prefix:str='humanoid-train-video', recording_freq:int=10_000) -> gym.Env:
    """Create chosen robotic environment that allows the recording of the environment 
    at the specified recording frequency. If given env name is not able to created,
    default env create is 'Humanoid-v5'.
    
    Parameters:
        - env_name (str) : the gymnasium environment id string.
        - vid_dir (str) : path to directory where videos should be stored.
        - vid_name_prefix (str) : the prefix to give the video. Defaulted to 
                                  "humanoid-test-video".
        - recording_freq (int) : the frequency at which to record an episode. 
                                 Defaulted to 5_000 for every 5_000 episodes record 
                                 a video.
    
    Returns:
        - env (gym.Env) : the created environment to simulate.                   
    """

    try:
        env = gym.make(env_name, render_mode='rgb_array')

    except Exception:
        print("Could not make environment. Defaulting to 'Humanoid-v5'.")
        env = gym.make("Humanoid-v5", render_mode='rgb_array')
    
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder=vid_dir, name_prefix=vid_name_prefix, episode_trigger=lambda x: (x % recording_freq) == 0)
    return env

def get_weights(dir:str, device:torch.device) -> dict[str:dict]|None:
    """Find the most recent model file, for both actor and critic, based on episode 
    number and return the weights. This returns None if there are no files from 
    where weights can be extracted.
    
    Parameters:
        - dir (str) : the directory to search for the weights files.
        - device (torch.device) : the device tag indicating where to load
                                  the tensor.
        
    Returns:
        - (dict) : a dict with str:dict key-valua pairs containing the weights
                   of the actor and critic networks. Actor parameters are accessed by 
                   'actor' and critic by 'critic'. If no weights are found, function
                   returns None.
    """
    
    files = os.listdir(dir)
    actor_files = [(f, int(re.search(r"_epoch(\d+)", f).group(1))) for f in files if f.startswith("actor_weights") and re.search(r"_epoch(\d+)", f)]
    critic_files = [(f, int(re.search(r"_epoch(\d+)", f).group(1))) for f in files if f.startswith("critic_weights") and re.search(r"_epoch(\d+)", f)]

    if not actor_files or not critic_files:
        return None
    
    # Find latest matching episode
    latest_actor = max(actor_files, key=lambda x: x[1])
    latest_critic = max(critic_files, key=lambda x: x[1])

    if latest_actor[1] != latest_critic[1]:
        warnings.warn(f"Actor and critic weights are from different episodes!\nActor episode: {latest_actor[1]}\nCritic episode: {latest_critic[1]}")
        time.sleep(1.5)

    actor_path = os.path.join(dir, latest_actor[0])
    critic_path = os.path.join(dir, latest_critic[0])
    actor_weights = torch.load(actor_path, map_location=device)
    critic_weights = torch.load(critic_path, map_location=device)

    return {'actor':actor_weights, 'critic':critic_weights}

@torch.no_grad()
def test_agent(env_name:str, vid_dir:str, weights_dir:str, pol_net:PolicyNetwork, device:torch.device,
               use_best_action:bool=False, num_test_episodes:int=2, test_recording_freq:int=1, 
               vid_prefix:str='humanoid-test-video') -> None:
    """Test the policy network on a given environment. User can specify 
    whether to use a stochastic or deterministic policy. See parameters 
    below.

    Parameters:
        - env_name (str) : the str id of the environment to test the agent.
        - vid_dir (str) : a str denoting the path of the directory to store
                          the recorded videos of the episodes.
        - weight_dir (str): a str denoting the path of the directory to look
                            into.
        - pol_net (PolicyNetwork) : the policy class used to create the neural
                                    network for the policy. The object should
                                    not be an instance but rather the class.
        - device (torch.device): the device tag indicating where to load
                                 the tensor.
        - use_best_action (bool) : whether to keep using the learned stochastic
                            policy for testing. If False, the mean of each
                            distribution over the action space will be used,
                            so that we are using a deterministic policy.
                            Using a stochastic policy serves more to test general
                            policy robustness; whereas, a deterministic policy 
                            will test peak agent performance. Default is False.
        - num_test_episodes (int) : the total number of episodes to test the
                                    agent on. Default is 2.
        - test_recording_freq (int) : the number of episodes that must pass 
                                      before the next episode is recorded.
                                      Default is 1.
        - vid_prefix (str) : a str to prepend the video name with. Default is 
                             'humanoid-test-video'.

    Returns:
        - None
    """

    env = get_env(env_name, vid_dir, vid_name_prefix=vid_prefix, recording_freq=test_recording_freq)
    agent = pol_net(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]).to(device)

    weights = get_weights(dir=weights_dir, device=device)
    if weights is None:
        raise ValueError("No files containing the actor and critic weights could be found!")
    agent.load_state_dict(weights['actor'])
    agent.eval()

    pbar = tqdm(iterable=range(1, num_test_episodes+1), desc='Testing Agent')

    for _ in range(num_test_episodes):
        episodic_reward = 0
        state, info = env.reset()
        terminated = False

        while not terminated:
            state_tensor = ((torch.tensor(state, dtype=torch.float32)).unsqueeze(0)).to(device)

            if use_best_action:
                μ, _ = agent(state_tensor)
                action = μ.squeeze(0).cpu().numpy()
            else:
                action, _ = agent.act(state_tensor)
                action = action.squeeze(0).cpu().numpy()

            state, reward, terminated, trunc, info = env.step(action)
            episodic_reward += reward

        pbar.set_postfix(episodic_reward=episodic_reward)
        pbar.update()

    agent.train()
    env.close()