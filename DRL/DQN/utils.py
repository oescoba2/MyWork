from agent import QNetwork
from collections import deque, namedtuple
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics, RecordVideo
from numpy.typing import ArrayLike
from torch import nn
from tqdm import tqdm
from warnings import warn

import ale_py
import gymnasium as gym
import numpy as np
import os
import pickle
import time
import torch
import random
import re

Experience = namedtuple('Experience', field_names=['state', 'action', 'next_state', 'reward', 'terminated'])

class UniformReplay:
    """This class is intended to manage a memory buffer for storing 
    experiences from a reinforcement learning environment.

    Attributes:
        - buffer_min_size (int) : the minimum size the buffer must
                                  be for sampling.

    Methods:
        - add() : Adds an experience to the memory buffer.
        - initialize() : Initializes the memory to a given size.
        - sample() : Samples a batch of experiences from the memory 
                     buffer.
        - save() : Saves the replay buffer for later use.

    Hidden Methods:
       _ __init__() : the constructor of the class. Creates the
                     capacity attribute.
       - __len__() : allows the use of the len() function.
    """

    def __init__(self, capacity:int, min_size:int) -> None:
        """Create the memory buffer as an attribute.

        Parameters:
            - capacity (int) : The maximum number of experiences to store
                               in the memory buffer.
            - min_size (int) : the desired size the memory buffer should
                               be before random sampling is available
        
        Returns:
            None
        """

        self.buffer = deque(maxlen=capacity)  # Create empty deque
        self.buffer_min_size = min_size

    def add(self, *args) -> None:
        """Save an experience e_t to memory.
        
        Parameters:
            - *args : any number of arguments. These will be unpacked when
                      placed into namedtuple Experience.

        Returns:
            None
        """

        self.buffer.append(Experience(*args))  # Add to deque

    def initialize(self, env:gym.Env) -> None:
        """Initialize the memory buffer to some specified size. It 
        raises an exception should the given 'env' not be of the 
        right type. Message is printed out once buffer is initialized 
        to desired size.

        Parameters:
            - env (gym.Env) : the environment being simulated

        Returns:
            None
        """

        # Check input
        if 'gymnasium' not in str(type(env)):
            raise TypeError("The given argument for 'env' is not a gymnasium " +\
                            f"environment type object. The given type is {type(env)}")
        
        curr_size = len(self.buffer)
        
        print("Initializing memory buffer...")
        while curr_size < self.buffer_min_size:
            curr_obs, info = env.reset()
            terminated = False

            while not terminated:
                action = env.action_space.sample()
                next_obs, reward, terminated, trunc, info = env.step(action)

                self.buffer.append(Experience(*(curr_obs, action, next_obs, reward, terminated)))
                curr_size = len(self.buffer)

                if curr_size >= self.buffer_min_size:
                    break
                else:
                    curr_obs = next_obs
        env.close()
        print("Memory buffer initialized.")

    def sample(self, batch_size:int) -> list[ArrayLike, ...]:
        """Randomly sample a batch of experiences 
        e_t = (s_t, a_t, s_{t+1}, r_{t+1}, terminated) from memory.

        Parameters:
            - batch_size (int) : The number of experiences to sample.

        Returns:
            - (list[ArrayLike]) : a list of 5 ArrayLike objects 
                                  containing the states, actions, 
                                  next states, rewards, and 
                                  terminations for each experience 
                                  sampled (in that order).
        """

        if len(self.buffer) == 0:
            raise Exception("Replay buffer has not being initialized yet. Cannot sample with 0 experiences.")
        
        if len(self.buffer) <= batch_size:
            raise Exception('Cannot sample more experiences than what is store in replay buffer.')
         
        experiences = random.sample(self.buffer, batch_size)                      
        experience_tuple = Experience(*zip(*experiences))
        sampled_experiences = [np.array(experience_tuple[i]) for i in range(5)]

        return sampled_experiences

    def save(self, drl_directory:str, episode:int) -> None:
        """Save the replay buffer to a file using pickle.

        Parameters:
            - drl_directory (str): The directory on where to store
                                   the pickle file.
            - 
        """

        filename = 'replay_buffer_episode' + str(episode) + '.pickle'
        filepath = os.path.join(drl_directory, filename)

        with open(filepath, 'wb') as f:

            # Serialize the buffer and its minimum size
            pickle.dump({
                'buffer': list(self.buffer),                                    # Convert deque to list
                'min_size': self.buffer_min_size,
                'capacity': self.buffer.maxlen
            }, f)

    def __len__(self) -> int:
        """Enables the use of len() function.
        
        Parameters:
            None
            
        Returns:
            - (int): the length of the deque
        """

        return len(self.buffer)

def get_env(vid_dir:str, game:str='BreakoutNoFrameskip-v4', vid_name_prefix:str='Qnet-train-video', recording_freq:int=2_500) -> gym.Env|None:
  """Make and preprocess the environment (i.e. game emulator) as
  outlined by the DeepMind 'Human-level control through deep
  reinforcement learning' paper. This includes the phi-function
  outlined in the paper as well as giving the ability to record
  videos and stats of episodes during the training phase. It
  raises an exception should the game not be supported.
      
  Parameters:
    - vid_dir (str): The directory to save the videos to.
    - game (str): the game to emulate. Defaulted to 'Breakout
                  NoFrameskip-v4'.
    - vid_name_prefix (str): The prefix to use for the video
                             files. Defaulted to 'drl-train-video'.
    - recording_freq (int): The episode frequency at which to
                            record videos. Defaulted to 2_500
                            (i.e. record every 2_500 episodes)

  Returns:
    - env (gym.Env): The preprocessed environment.
  """
  
  if "NoFrameskip" in game:
    env = gym.make(game, render_mode='rgb_array')
          
  else:
    try:  
      env = gym.make(game, render_mode='rgb_array', frameskip=1)
          
    except Exception:
      raise Exception("The given game is not supported. Try another game.")

  # Preprocess
  env = RecordEpisodeStatistics(env)
  env = RecordVideo(env, video_folder=vid_dir, name_prefix=vid_name_prefix,
                    episode_trigger=lambda x: (x % recording_freq) == 0)
  env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, 
                            grayscale_obs=True)
  env = FrameStackObservation(env, stack_size=4)

  return env

def get_epsilon(episode_num:int, num_episodes_decay:int=int(6e5), epsilon_start:float=1.0, epsilon_min:float=0.1) -> float:
    """Linearly decay the epsilon for a specified number of 
    episodes. After that, set epsilon to a constant value.
    
    Parameters:
        - episode_num (int): the number of the episode the
                             agent is in.
        - num_episodes_decay (int): the number of episodes
                                    on which epsilon will
                                    decay to a specifc value.
                                    Defaulted to 600_000.
        _ epsilon_start (float): the starting value of 
                                 epsilon. Defaulted to 1.0.
        - epsilon_min (float): the minimum value epsilon
                               can have for the remainder
                               episodes left. This is also
                               the ending value for epsilon.
                               Defaulted to 0.1. 
    Returns:
        - (float): the value for epsilon for the given epi-
                   sode number.
    """

    slope = (epsilon_start - epsilon_min) / num_episodes_decay

    return max(epsilon_start - slope * episode_num, epsilon_min)

def get_weights_buffer(dir:str, device:torch.device, get_buffer:bool=True) -> dict|None:
    """Find the most recent trained model weights based on episode 
    number as well as the replay buffer, if specified. This returns
    None if there are no files from where weights can be extracted 
    as well as the buffer.
    
    Parameters:
        - dir (str) : the directory to search for the weights
                      and replay buffer files.
        - device (torch.device): the device tag indicating where
                                 to load to load the tensor of
                                 weights.
        - get_buffer (bool) : whether to find and return the replay
                              buffer. Defaulted to False.
        
    Returns:
        - (dict) : When get_buffer is True, the return is a 
                   dictionary whose keys are 'weights' and 'buffer'
                   and the values are the Q-network weights and
                   the replay buffer; otherwise None is returned
                   if any of these files is not able to be found. 
                   When get_buffer is False, the return is an 
                   ordered dictionary of the Q-network weights; 
                   otherwise the return is None if no weights 
                   file can be found.
    """
    
    if get_buffer:
        files = os.listdir(dir)
        weights_files = [(f, int(re.search(r"_episode(\d+)", f).group(1))) for f in files if f.startswith("Qnet_weights") and re.search(r"_episode(\d+)", f)]
        buffer_files = [(f, int(re.search(r"_episode(\d+)", f).group(1))) for f in files if f.startswith("replay_buffer") and re.search(r"_episode(\d+)", f)]
        
        if not weights_files or not buffer_files:
            return None
        
        latest_weight = max(weights_files, key=lambda x: x[1])
        latest_buffer = max(buffer_files, key=lambda x: x[1])

        if latest_weight[1] != latest_buffer[1]:
            warn(f"Saved weights and replay buffer are from different episodes!\nWeights episode: {latest_weight[1]}\nBuffer episode: {latest_buffer[1]}")
            time.sleep(1.5)

        weight_path = os.path.join(dir, latest_weight[0])
        weights = torch.load(weight_path, map_location=device)

        buffer_path = os.path.join(dir, latest_buffer[0])
        with open(buffer_path, 'rb') as f:
            data = pickle.load(f)
        buffer = UniformReplay(capacity=data['capacity'], min_size=data['min_size'])
        buffer.buffer = deque(data['buffer'], maxlen=data['capacity'])

        return {'weights':weights, 'buffer':buffer}

    else: 
        files = os.listdir(dir)
        weights_files = [(f, int(re.search(r"_episode(\d+)", f).group(1))) for f in files if f.startswith("Qnet_weights") and re.search(r"_episode(\d+)", f)]

        if not weights_files:
            return None
        
        latest_weight = max(weights_files, key=lambda x: x[1])[0]
        weight_path = os.path.join(dir, latest_weight)
        weights = torch.load(weight_path, map_location=device)

        return weights

def test_agent(net:QNetwork, game:str, device:torch.device, weights_dir:str, vid_dir:str, uses_dueling:bool=False, 
               num_test_episodes:int=2, vid_prefix:str='Qnet-test-video', test_recording_freq:int=1) -> None:
    """Test a trained agent by running episodes in the environment 
    using the learned policy.
    
    Parameters:
        - net (QNetwork) : the network/agent class to evaluate.
                           Note that this is the class and not
                           the object/instance.
        - game (str) : The str id of the  environment to test the 
                       agent on.
        - device (torch.device) : the device tag indicating where
                                  to load to load the tensor.
        - weights_dir (str) : the directory where to search for 
                              the network weights.
        - vid_dir (str) : the directory to store the created videos.
        - uses_dueling (bool) : whether the given network uses
                                dueling architecture. Defaulted
                                to False.
        - num_test_episodes (int) : Number of episodes to run for 
                                    testing. Defaulted to 100.
        - vid_prefix (str) : the prefix to give to the created 
                             videos. Default is 'Qnet-test-video'.
        - test_recording_freq (str) : the number of episodes that 
                                      must pass before the next 
                                      episode is recorded. Defaulted
                                      to 1.
                                   
    Returns:
        - None
    """

    # Create model
    agent = net(env.observation_space, env.action_space, dueling=uses_dueling).to(device)
    weights = get_weights_buffer(weights_dir, device, get_buffer=False)  

    if weights is not None:                                      
        agent.load_state_dict(weights)
    else:
        raise FileNotFoundError("No file for most recent Q-network weights was found.")
    
    agent.eval()                                                                                  # Set the network to evaluation mode
    env = get_env(vid_dir, game=game, vid_name_prefix=vid_prefix, recording_freq=test_recording_freq)
    pbar = tqdm(range(1, num_test_episodes+1), desc='Testing')

    for _ in range(num_test_episodes):

        state, info = env.reset()
        terminated = False

        while not terminated:

            # Choose the action with the highest Q-value
            Qvals = agent(torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device))
            action = torch.argmax(Qvals, dim=1).cpu().numpy().item()

            # Take the chosen action in the environment
            next_state, reward, terminated, trunc, info = env.step(action)
            state = next_state

        pbar.set_postfix(episodic_reward=info["episode"]['r'])
        pbar.update()
    
    env.close()
    agent.train()