import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import time
import gym
import torch

class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.buffer = [None] * capacity

        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def put(self, obs, action, reward, next_obs, done):
        """Put a tuple of (obs, action, rewards, next_obs, done) into the replay buffer.
        The max length specified by capacity should never be exceeded.
        The oldest elements inside the replay buffer should be overwritten first.
        """
        self.buffer[self.ptr] = (obs, action, reward, next_obs, done)

        self.size = min(self.size + 1, self.capacity)
        self.ptr = (self.ptr + 1) % self.capacity

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer.
        Should return 5 lists of, each for every attribute stored (i.e. obs_lst, action_lst, ....)
        """
        return zip(*random.sample(self.buffer[:self.size], batch_size))

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        return self.size

def rolling_window(a, window, step_size):
    """Create a rolling window view of a numpy array."""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


ax = None
fig = None


def episode_reward_plot(rewards, frame_idx, window_size=5, step_size=1, updating=True):
    """Plot episode rewards rolling window mean, min-max range and standard deviation.

    Parameters
    ----------
    rewards : list
        List of episode rewards.
    frame_idx : int
        Current frame index.
    window_size : int
        Rolling window size.
    step_size: int
        Step size between windows.
    updating: bool
        You can try to set updating to True, which hinders matplotlib to create a new window for every plot.
        Doesn't work with my Pycharm SciView currently.
    """
    global ax
    global fig

    if len(rewards) < window_size + 1:
        return

    plt.ion()
    rewards_rolling = rolling_window(np.array(rewards), window_size, step_size)
    mean = np.mean(rewards_rolling, axis=1)
    std = np.std(rewards_rolling, axis=1)
    min = np.min(rewards_rolling, axis=1)
    max = np.max(rewards_rolling, axis=1)
    x = np.arange(math.floor(window_size / 2), len(rewards) - math.floor(window_size / 2), step_size)

    if ax is None or not updating:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(x, mean, color='blue')
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, facecolor='blue')
    ax.fill_between(x, min, max, alpha=0.1, facecolor='red')
    if updating:
        plt.ion()
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.pause(0.01)
        # plt.show()
        ax.cla()
    else:
        plt.show(block=False)

class RandomAgent:
    def __init__(self, env):
        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')
        self.num_actions = env.action_space.n

    def predict(self, obs):
        return np.random.randint(0, self.num_actions)


def visualize_agent(env, agent, num_episodes=1, num_timesteps=None):
    """ Visualize an agent performing insinde a Gym environment. If you pass 'num_timesteps', it will """
    with torch.no_grad():
        obs = env.reset()
        eps_counter = 0
        for episode in range(num_episodes):
            timestep = 0
            while True:
                timeout = (num_timesteps and timestep == num_timesteps)
                env.render()

                if len(obs.shape) == 4:
                    obs = np.squeeze(obs, axis=3)
                elif len(obs.shape) != 1:
                    # Change from HWC to CHW format
                    obs = np.transpose(obs, (2, 0, 1))
                action = agent.predict(obs)
                obs, reward, done, _ = env.step(action)
                if done or timeout:
                    if timeout:
                        return
                    obs = env.reset()
                    break
                # 30 FPS
                time.sleep(0.033)
                timestep += 1