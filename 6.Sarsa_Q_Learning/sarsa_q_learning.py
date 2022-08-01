import numpy as np
import random
from helper import action_value_plot, test_agent

from gym_gridworld import GridWorldEnv

class SARSAQBaseAgent:
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        self.g = discount_factor
        self.lr = learning_rate
        self.eps = epsilon
        self.env = env

        # TODO: define a Q-function member variable self.Q
        # Remark: Use this kind of Q-value member variable for visualization tools to work, i.e. of shape [grid_height, grid_width, num_actions]
        # Q[y, x, z] is value of action z for grid position y, x
        self.Q = np.zeros([4, 4, 4], dtype=np.float32)

    def action(self, s, epsilon=0.0):
        # TODO: implement epsilon-greedy action selection

        random_num = np.random.uniform(0 , 1)

        if random_num <= epsilon:
            action = self.env.action_space.sample()

        else:
            max_q_actions = np.argwhere(self.Q[s[0] , s[1]] == np.amax(self.Q[s[0] , s[1]])).flatten()
            action = np.random.choice(max_q_actions)

        return action

class SARSAAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=50000):
        # TODO: implement training loop


        for i in range(n_timesteps):
            state = self.env.reset()
            action = self.action(state, self.eps)
            while True:
                next_state , reward , done , _ = self.env.step(action)
                next_state_action = self.action(next_state , self.eps)
                self.update_Q(state , action , reward , next_state , next_state_action)

                if done:
                    break
                state = next_state
                action = next_state_action

    def update_Q(self, s, a, r, s_, a_):
        # TODO: implement Q-value update rule
        self.Q[s[0] , s[1] , a] += self.lr * (r + self.g * self.Q[s_[0] , s[1] , a_] - self.Q[s[0] , s[1] , a])


class QLearningAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(QLearningAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=50000):
        # TODO: implement training loop

        for i in range(n_timesteps):
            state = self.env.reset()
            while True:
                action = self.action(state , epsilon=self.eps)
                next_state , reward , done , _ = self.env.step(action)
                self.update_Q(state , action , reward , next_state)

                if done:
                    break

                state = next_state



    def update_Q(self, s, a, r, s_):
        # TODO: implement Q-value update rule
        max_next_a_val = np.max(self.Q[s_[0] , s_[1]])
        self.Q[s[0] , s[1] , a] += self.lr * (r + self.g * max_next_a_val - self.Q[s[0] , s[1] , a])


if __name__ == "__main__":
    # Create environment
    env = GridWorldEnv()

    discount_factor = 0.9
    learning_rate = 0.1
    epsilon = 0.4
    n_timesteps = 200000

    # Train SARSA agent
    sarsa_agent = SARSAAgent(env, discount_factor, learning_rate, epsilon)
    sarsa_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(sarsa_agent)
    # Uncomment to do a test run
    print("Testing SARSA agent...")
    test_agent(sarsa_agent, env, epsilon)

    # Train Q-Learning agent
    # qlearning_agent = QLearningAgent(env, discount_factor, learning_rate, epsilon)
    # qlearning_agent.learn(n_timesteps=n_timesteps)
    # action_value_plot(qlearning_agent)
    # # Uncomment to do a test run
    # print("Testing Q-Learning agent...")
    # test_agent(qlearning_agent, env, 0.0)
