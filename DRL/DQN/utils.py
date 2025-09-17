from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics, RecordVideo
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm
import ale_py
import gymnasium as gym
import numpy as np
import os

def get_env(game:str, atari:bool=False, record:bool=True, record_freq:int=2_500, 
            vid_dir:str="./vids", vid_name_prefix:str='RL-vid') -> gym.Env:
    """Create the specified game into a gymnasium environment.

    Parameters:
        - game (str) : the name of the game that the agent should learn
                       to play.
        - atari (bool) : a bool indicating whether the given str name 
                         is an atari game. Default is False which 
                         indicates that a normal gym env should be 
                         created and not an atari env.
        - record (bool) : a bool indicating whether or not to record
                          videos of the environment. Default is True.
        - record_freq (int) : the number of epsiodes between successive
                              videos being taken of the environment.
                              Default is 2_500 which means record every
                              2_500 episodes. Only used when record is
                              True.
        - vid_dir (str) : a str specifying the location of the directory
                          where the created videos should be stored in.
                          Default is './vids'. Only used when record is
                          True.
        - vid_name_prefix (str) : a str indicating the prefix to use in
                                  naming the videos. Default is 'RL-vid'.
                                  Only used when record is True.
    Returns: 
        - env (gym.Env) : the created gymnasium environment.
    """

    env = gym.make(game, render_mode='rgb_array') if not atari else gym.make(game, render_mode='rgb_array', frameskip=1)
    env = RecordEpisodeStatistics(env)

    if record:
        env = RecordVideo(env, video_folder=vid_dir, name_prefix=vid_name_prefix,
                          episode_trigger=lambda x: (x % record_freq) == 0)
    
    if atari:
        env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, 
                                 grayscale_obs=True)
        env = FrameStackObservation(env, stack_size=4)

    return env

def get_epsilon(episode_num:int, num_episodes_decay:int=int(6e5), epsilon_start:float=1.0, 
                epsilon_min:float=0.1) -> float:
    """Linearly decay the epsilon for a specified number of episodes.
    After that, set epsilon to a constant value.
    
    Parameters:
        - episode_num (int): the number of the episode the agent is
                             in.
        - num_episodes_decay (int): the number of episodes on which
                                    epsilon will decay to a specifc 
                                    value. Default is 600_000.
        _ epsilon_start (float): the starting value of epsilon. 
                                 Defaulted to 1.0.
        - epsilon_min (float): the minimum value epsilon can have
                               for the remainder episodes left.
                               This is also the ending value for 
                               epsilon. Defaulted to 0.1. 
    Returns:
        - (float): the value for epsilon for the given episode
                   number.
    """

    slope = (epsilon_start - epsilon_min) / num_episodes_decay

    return max(epsilon_start - slope * episode_num, epsilon_min)

def make_dirs(dirs:list[str]=['vids', 'weights', 'buffers']) -> None:
    """Create the given directories.
    
    Parameters:
        - dirs (list) : a list of strings containing the names of
                        the directories to create. Default is 
                        ['vids', 'weights', 'buffers']
    Returns:
        - None
    """
    
    cwd = os.getcwd()

    for dir in dirs:
        dir_path = os.path.join(cwd, dir)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def plot_stats(content:dict[str:ArrayLike], xvals:ArrayLike, moving_avg:bool=False,
               figsize:tuple[int]=(6,4), dpi:int=400, xlabel:str='Episode',
               figtitle:str='Results') -> None:
    """Plot the given dictionary using one line plot and gridlines
    versus a constant array of x-axis values.

    Parameters:
        - content (dict) : a dict containing all the data to plot
                           on the y-axis. Each entry in the 
                           dictionary will be a separate plot.
                           a separate line in the plot. The keys
                           are the string labels to use for the 
                           y-axis. The values are ArrayLike
                           containg the y-axis values.
        - xvals (ArrayLike) : the values to plot in the x-axis.
        - moving_avg (bool) : whether or not to plot the moving
                              average of the given y-axis values
                              found in content. Default is False.
        - figsize (tuple) : a tuple of two integers denoting the
                            (width, height) measurements, in 
                            inches, of the created plot. Default
                            is (6, 4).
        - dpi (int) : an int denoting the dots-per-inch (resolution)
                      to use in creating the plot. Default is 
                      400.
        - xlabel (str) : the name to give the x-axis. Default is 
                         'Episode'.
        - figtitle (str) : the name to give the generate plot. 
                           Default is 'Results'.

    Returns: 
        - None
    """

    xscale = [min(xvals), max(xvals)]
    num_rows = len(content)
    fig, ax = plt.subplots(num_rows, ncols=1, figsize=figsize, dpi=dpi)

    for i, (label, yvals) in enumerate(content.items()):
        
        ax[i].grid(alpha=0.3)
        ax[i].set_xlim(xscale[0], xscale[1])
        ax[i].set_xlabel(xlabel)
        yscale = [min(0, min(yvals)), max(yvals)+1]
        ax[i].set_ylim(yscale[0], yscale[1])
        ax[i].set_ylabel(label)

        if moving_avg:
            ax[i].plot(xvals, yvals, color='blue')
            avgs = np.cumsum(yvals) / xvals
            ax[i].plot(xvals, avgs, color='red', label='Average')
            ax[i].legend()
        else:
            ax[i].plot(xvals, yvals)

    fig.suptitle(figtitle)
    plt.tight_layout()
    plt.show()

def test_agent(game:str, qnet:object=None, qtable:NDArray=None, N:int=2,
               vid_dir:str="./vids", 
               vid_prefix:str='Qnet-test-video', test_recording_freq:int=1) -> tuple[dict[str:ArrayLike], ArrayLike]:
    """Test the trained agent on a given game.

    Parameters:

    Returns:

    """

    if (qnet is None) and (qtable is None):
        raise ValueError("Either qnet or qtable must be given. Both cannot be None.")
    
    # Test Q-learning agent
    elif (qtable is not None) and (qnet is None):
        env = get_env(game=game, record_freq=test_recording_freq, vid_dir=vid_dir, vid_name_prefix=vid_prefix)
        pbar = tqdm(range(1, N+1), desc='Testing')
        episodes = np.arange(start=1, stop=N+1, dtype=int)
        testing_stats = {"Reward" : np.zeros((N, )),
                         "Time steps" : np.zeros((N, ))}

        for i in episodes:
            
            state, info = env.reset()
            done = False
            trunc = False

            while not done and not trunc:
                act = (qtable[state]).argmax()
                state, reward, done, trunc, info = env.step(act)

            testing_stats["Reward"][i-1] = info['episode']['r']
            testing_stats['Time steps'][i-1] = info['episode']['l']
            pbar.set_postfix(episodic_reward=info["episode"]['r'], timesteps=info["episode"]['l'])
            pbar.update()
    
        env.close()

        return testing_stats, episodes
    
    elif (qnet is not None) and (qtable is None):
        pass
