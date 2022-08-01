"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from copy import deepcopy

import copy


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)

class GridWorldEnv(gym.Env):
    """
    Description:
        An agent moves inside of a small maze.
    Source:
        This environment is a variation on common grid world environments.
    Observation:
        Type: Box(1)
        Num     Observation            Min                     Max
        0       y position             0                       self.map.shape[0] - 1
        1       x position             0                       self.map.shape[1] - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed into one of the squares marked "s"
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    metadata = {
        'render.modes': ['ascii'],
    }

    def __init__(self, map_name='standard'):
        if map_name == 'standard':
            self.map = [
                list("s   "),
                list("    "),
                list("    "),
                list("gt g"),
            ]
            self.action_space = spaces.Discrete(4)
            self.grid_size = 4
            self.observation_space = spaces.Box(0, self.grid_size, shape=(2,), dtype=np.float32)
            self.start_position = [0, 0]
            self.agent_position = deepcopy(self.start_position)
            self.trap_reward = -1.0

        elif map_name == 'cliffwalking':
            # TODO: Implement the Cliff Walking environment
            raise NotImplementedError
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if action == 0:
            self.agent_position[0] -= 1
        if action == 1:
            self.agent_position[1] += 1
        if action == 2:
            self.agent_position[0] += 1
        if action == 3:
            self.agent_position[1] -= 1

        self.agent_position[0] = clamp(self.agent_position[0], 0, self.grid_size - 1)
        self.agent_position[1] = clamp(self.agent_position[1], 0, self.grid_size - 1)

        reward = 0.0
        done = False

        if self.map[self.agent_position[0]][self.agent_position[1]] == "t":
            reward = self.trap_reward
            done = True

        if self.map[self.agent_position[0]][self.agent_position[1]] == "g":
            reward = 1
            done = True
        
        return self.observe(), reward, done, {}

    def reset(self):
        self.agent_position = deepcopy(self.start_position)
        return self.observe()

    def observe(self):
        return np.array(self.agent_position)

    def render(self, mode='ascii'):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print("".join(["|", " "]  + row + [" ", "|"]))
        print("--------")
        return None

    def close(self):
        pass
