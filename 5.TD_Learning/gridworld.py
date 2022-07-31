import copy

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    This environment is a variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       x position             0                       self.map.shape[0] - 1
        1       y position             0                       self.map.shape[1] - 1
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
        The agent is placed at [0, 0]
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    metadata = {
        'render.modes': ['ascii'],
    }

    def __init__(self):
        self.map = [
            list("s   "),
            list("    "),
            list("    "),
            list("gt g"),
        ]

        # TODO: Define your action_space and observation_space here
        self.action_space = gym.spaces.discrete.Discrete(4,)
        self.observation_space = gym.spaces.box.Box(low = np.array([0 , 0]) ,
                                                    high = np.array((len(self.map[0]) , len(self.map))),
                                                    dtype = np.int32)

        self.agent_position = np.array([0 , 0])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # TODO: Write your implementation here

        x_pos , y_pos = self.agent_position
        reward = 0
        done = False
        info = {}

        if (action == 0) and (y_pos > 0):
            y_pos -= 1

        elif (action == 1) and (x_pos < len(self.map[0]) - 1):
            x_pos += 1

        elif (action == 2) and (y_pos < len(self.map) - 1):
            y_pos += 1

        elif (action == 3) and (x_pos > 0):
            x_pos -= 1

        self.agent_position = np.array([x_pos , y_pos])

        if self.map[x_pos][y_pos] == 'g':
            reward = 1
            done = True

        elif self.map[x_pos][y_pos] == 't':
            reward = -1
            done = True

        return self.agent_position , reward , done , info


    def reset(self):
        # TODO: Write your implementation here

        self.agent_position = np.array([0 , 0])
        return np.array([0 , 0])

    def render(self, mode='ascii'):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print("".join(["|", " "] + row + [" ", "|"]))
        print("--------")
        return None

    def close(self):
        pass
