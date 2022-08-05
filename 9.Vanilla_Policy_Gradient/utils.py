import numpy as np
import matplotlib.pyplot as plt
import math


def rolling_window(a, window, step_size):
    """Create a rolling window view of a numpy array.

    Parameters
    ----------
    a : numpy.array
    window : int
    step_size : int
    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


fig, ax = None, None
def episode_reward_plot(rewards, frame_idx, window_size=5, step_size=1):
    """Plot episode rewards rolling window mean, min-max range and standard deviation.


    Parameters
    ----------
    rewards : list
        List of episode rewards.
    frame_idx : int
        Current frame index.
    window_size : int
    step_size: int
    """

    global fig
    global ax
    plt.ion()
    rewards_rolling = rolling_window(np.array(rewards), window_size, step_size)
    mean = np.mean(rewards_rolling, axis=1)
    std = np.std(rewards_rolling, axis=1)
    min = np.min(rewards_rolling, axis=1)
    max = np.max(rewards_rolling, axis=1)
    x = np.arange(math.floor(window_size/2), len(rewards) - math.floor(window_size/2) , step_size)

    if fig is None:
        fig, ax = plt.subplots(figsize=(20,10))    
    ax.set_title('Frame %s. Reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    ax.plot(x, mean, color='blue')
    ax.fill_between(x, mean-std, mean+std, alpha=0.3, facecolor='blue')
    ax.fill_between(x, min, max, alpha=0.1, facecolor='red')
    plt.pause(0.0000001)
    plt.cla()
    plt.draw()
