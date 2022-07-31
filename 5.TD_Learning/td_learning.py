import numpy as np

from gridworld import GridWorldEnv
from util import value_function_policy_plot, estimate_policy_array_from_samples


class TDAgent:
    def __init__(self, env, discount_factor, learning_rate):
        self.g = discount_factor
        self.lr = learning_rate

        self.grid_height = int(env.observation_space.high[0])
        self.grid_width = int(env.observation_space.high[1])

        self.num_actions = env.action_space.n

        # V[x, y] is value for grid position x, y
        self.V = np.zeros([self.grid_height, self.grid_width], dtype=np.float32)
        # policy[x, y, z] is prob of action z when in grid position x, y
        self.policy = np.ones([self.grid_height, self.grid_width, self.num_actions],
                              dtype=np.float32) / self.num_actions

        # Uniform random actions in all states, except:
        self.policy[1, 1] = 0
        self.policy[1, 1, 1] = 1

        self.policy[2, 1] = 0
        self.policy[2, 1, 1] = 1

        self.policy[1, 0] = 0
        self.policy[1, 0, 1] = 1

        self.policy[2, 0] = 0
        self.policy[2, 0, 1] = 1

        self.env = env

    def action(self, s):
        # This is quite slow, but whatever
        action = np.random.choice(np.arange(self.num_actions), p=self.policy[s[0], s[1]])
        return action

    def learn(self, n_timesteps=50000):
        s = self.env.reset()
        s_ = None

        for i in range(n_timesteps):
            # TODO: implement the learn loop
            while True:
                action = self.action(s)
                s_ , reward , done , _ = self.env.step(action)
                self.V[s[0] , s[1]] += self.lr * (reward + self.g * self.V[s_[0] , s_[1]] - self.V[s[0] , s[1]])
                s = s_
                if done:
                    s = self.env.reset()
                    break


if __name__ == "__main__":
    # Create Agent and environment
    env = GridWorldEnv()
    td_agent = TDAgent(env, discount_factor=0.9, learning_rate=0.01)

    # Learn the Value function for 10000 steps.
    td_agent.learn(n_timesteps=10000)

    # Visualize V
    V = td_agent.V.copy()  # .reshape(4, 4)
    policy = estimate_policy_array_from_samples(td_agent)
    env_map = td_agent.env.map.copy()
    value_function_policy_plot(V, policy, env_map)
