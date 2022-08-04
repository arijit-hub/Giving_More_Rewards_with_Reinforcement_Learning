import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from helper import visualize_agent, episode_reward_plot

class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        # TODO

        self.capacity = capacity
        self.replay_buffer = []

    def put(self, obs, action, reward, next_obs, done):
        """Put a tuple of (obs, action, rewards, next_obs, done) into the replay buffer.
        The max length specified by capacity should never be exceeded. 
        The oldest elements inside the replay buffer should be overwritten first.
        """
        # TODO

        if len(self.replay_buffer) == self.capacity:
            self.replay_buffer = self.replay_buffer[1:]

        self.replay_buffer.append((obs, action, reward, next_obs, done))


    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer."""
        # TODO

        mini_batch = random.sample(self.replay_buffer , batch_size)

        return zip(*mini_batch)


    
    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        # TODO

        return len(self.replay_buffer)


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super(DQNNetwork, self).__init__()
        # TODO: Implement the network structure

        hid_state = 128
        self.net = nn.Sequential(nn.Linear(num_obs , hid_state) ,
                                 nn.ReLU() ,
                                 nn.Linear(hid_state , num_actions))
        
    def forward(self, x):
        # TODO: Implement the forward function, which returns the output net(x)

        return self.net(x)

class DQN():
    """The DQN method."""

    def __init__(self, env, replay_size=10000, batch_size=32, gamma=0.99, sync_after=5, lr=0.001):
        """ Initializes the DQN method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.
        sync_after: int
            Timesteps after which the target network should be synchronized with the main network.
        lr: float
            Adam optimizer learning rate.        
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')
        
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize DQN network
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim)
        # TODO: Initialize DQN target network, load parameters from DQN network
        self.target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.target_net.load_state_dict(self.dqn_net.state_dict())

        # Set up optimizer, only needed for DQN network
        self.optim_dqn = optim.Adam(self.dqn_net.parameters(), lr=lr)
    

    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []

        obs = self.env.reset()
        for timestep in range(1, timesteps + 1):
            epsilon = epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)
            
            next_obs, reward, done, _ = env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if done:
                obs = env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_msbe_loss()

                self.optim_dqn.zero_grad()
                loss.backward()
                self.optim_dqn.step()

            # TODO: Sync the target network
            if timestep % self.sync_after == 0:
                self.target_net.load_state_dict(self.dqn_net.state_dict())

            if timestep % 500 == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
    
    def predict(self, state, epsilon=0.0):
        """Predict the best action based on state. With probability epsilon take random action
        
        Returns
        -------
        int
            The action to be taken.
        """

        # TODO: Implement epsilon-greedy action selection
        rand_num = random.random()

        if rand_num <= epsilon:
            action = random.randrange(self.act_dim)

        else:
            state_tensor = torch.tensor(np.array([state]))
            q_val = self.dqn_net(state_tensor)
            #print(q_val.shape)
            action = int(torch.argmax(q_val , dim = 1).item())

        return action


    def compute_msbe_loss(self):
        """Compute the MSBE loss between self.dqn_net predictions and expected Q-values.
        
        Returns
        -------
        float
            The MSE between Q-value prediction and expected Q-values.
        """
        # TODO: Implement MSBE calculation

        obs_list, action_list, reward_list, next_obs_list, done_list = self.replay_buffer.get(self.batch_size)

        obs_tensor = torch.tensor(np.array(obs_list))
        action_tensor = torch.LongTensor(action_list).unsqueeze(1)
        reward_tensor = torch.tensor(np.array(reward_list)).unsqueeze(1)
        next_obs_tensor = torch.tensor(np.array(next_obs_list))
        done_tensor = torch.tensor(np.array(done_list))


        pred_q_val = torch.gather(self.dqn_net(obs_tensor) ,
                                dim = 1 ,
                                index = action_tensor)

        next_state_q_val = self.target_net(next_obs_tensor)
        #next_state_q_val[done_tensor] = 0.0

        max_next_state_q_val = torch.gather(next_state_q_val ,
                                            dim = 1 ,
                                            index = torch.max(next_state_q_val ,
                                            dim = 1 ,
                                            keepdim = True)[1])
        max_next_state_q_val[done_tensor] = 0.0

        expected_q_val = reward_tensor + self.gamma * max_next_state_q_val

        loss_val = F.mse_loss(pred_q_val.float() , expected_q_val.float())

        return loss_val


def epsilon_by_timestep(timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000):
    """Linearily decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps"""
    # TODO: Implement epsilon decay function

    if timestep == 0:
        return epsilon_start
    elif timestep >= frames_decay:
        return epsilon_final
    elif timestep > 0 and timestep < frames_decay:
        range = epsilon_start - epsilon_final
        each_update = range / frames_decay
        return epsilon_start - timestep * each_update

if __name__ == '__main__':
    # Create gym environment
    env_id = "CartPole-v1"
    env = gym.make(env_id)

    # Plot epsilon rate over time
    plt.plot([epsilon_by_timestep(i) for i in range(50000)])
    plt.show()

    # Train the DQN agent
    dqn = DQN(env)
    dqn.learn(30000)

    # Visualize the agent
    visualize_agent(env, dqn)