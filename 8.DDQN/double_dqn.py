import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from helper import visualize_agent, episode_reward_plot, RandomAgent


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
        if len(obs.shape) == 2:
            # This is for grayscale images
            # Else, PyTorch will think we pass n-dimensional images into the networks, due to the way we stack them inside the calc_msbe_loss func
            obs = np.expand_dims(obs, 0)
            next_obs = np.expand_dims(next_obs, 0)

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


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super(DQNNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)


class DQNNetworkVisual(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, c_in, num_actions):
        super(DQNNetworkVisual, self).__init__()
        # TODO: Implement here
        self.net = nn.Sequential(nn.Conv2d(c_in , 32 , (8 , 8) , 4) ,
                                 nn.ReLU(),
                                 nn.Conv2d(32 , 64 , (4 , 4) , 2),
                                 nn.ReLU(),
                                 nn.Conv2d(64 , 64 , (3 , 3) , 1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(3136 , 512),
                                 nn.ReLU(),
                                 nn.Linear(512 , 256),
                                 nn.ReLU(),
                                 nn.Linear(256 , num_actions)
                                 )

    def forward(self, x):
        # TODO: Implement here
        return self.net(x)



class DoubleDQN():
    """The DQN method."""

    def __init__(self, env, replay_size=10000, batch_size=32,
                 gamma=0.99, sync_after=1000, lr=0.00001,
                 learning_start=10000, train_freq=4, frame_stack=4,
                 save_freq=1000, resume=False, vis_while_train=10000):
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
        self.env = env
        self.act_dim = env.action_space.n
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.save_freq = save_freq
        self.vis_while_train = vis_while_train
        self.learning_starts = learning_start
        self.train_freq = train_freq
        self.frame_stack = frame_stack

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'

        if len(env.observation_space.shape) >= 2:
            # We assume a visual observation space if WxHxC input.
            # See, e.g. https://www.gymlibrary.ml/environments/atari/pong/ for information on input dimensions and action space.
            network = DQNNetworkVisual
            in_param = env.observation_space.shape[-1] * self.frame_stack
        else:
            # In case of vector obs
            network = DQNNetwork
            in_param = env.observation_space.shape[0]

        # Initialize DQN network
        self.dqn_net = network(in_param, self.act_dim).to(self.device)

        # Load the model from 'agent.pth' if set
        if resume:
            self.dqn_net.load_state_dict(torch.load('agent.pth'))

        # Initialize DQN target network, load parameters from DQN network
        self.dqn_target_net = network(in_param, self.act_dim).to(self.device)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

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
        # This is weird shape augmentation necessary for allowing vector, 1D grayscale and n-dim framstacked observations
        if len(obs.shape) == 4:
            obs = np.squeeze(obs, axis=3)
        elif len(obs.shape) != 1:
            # Change from HWC to CHW format
            obs = np.transpose(obs, (2, 0, 1))

        for timestep in tqdm(range(1, timesteps + 1)):
            epsilon = epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            # Change from HWC to CHW format
            if len(next_obs.shape) == 4:
                next_obs = np.squeeze(next_obs, axis=3)
            elif len(next_obs.shape) != 1:
                # Change from HWC to CHW format
                next_obs = np.transpose(next_obs, (2, 0, 1))

            self.replay_buffer.put(obs, action, reward, next_obs, done)
            obs = next_obs

            episode_rewards.append(reward)

            if done:
                obs = env.reset()
                if len(obs.shape) == 4:
                    obs = np.squeeze(obs, axis=3)
                elif len(obs.shape) != 1:
                    # Change from HWC to CHW format
                    obs = np.transpose(obs, (2, 0, 1))

                all_rewards.append(sum(episode_rewards))
                episode_rewards = []

            if self.batch_size < len(self.replay_buffer) and len(self.replay_buffer) >= self.learning_starts and timestep % self.train_freq == 0:
                # TODO: implement here

                # Calculate loss
                loss = self.compute_msbe_loss()

                # Optimize
                self.optim_dqn.zero_grad()
                loss.backward()
                self.optim_dqn.step()

            if timestep % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

            # Save model checkpoint
            if timestep % self.save_freq == 0:
                torch.save(self.dqn_net.state_dict(), 'agent.pth')
                plt.savefig("rewards.pdf")

            if timestep > self.learning_starts:
                # Plot smoothed reward plot
                if timestep % 1000 == 0:
                    num_rewards = len(all_rewards)
                    sub_sample = max(1, int(100 / num_rewards))
                    episode_reward_plot(all_rewards[::sub_sample], timestep, window_size=7, step_size=1)
                # Visualize agent during training
                if self.vis_while_train and timestep % self.vis_while_train == 0:
                    print("Visualizing...")
                    visualize_agent(env, self, num_timesteps=100)

    def predict(self, state, epsilon=0.001):
        """Predict the best action based on state. With probability epsilon take random action
        
        Returns
        -------
        int
            The action to be taken.
        """

        if random.random() > epsilon:
            # The unsqueezing is to get our data into batch form
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value = self.dqn_net.forward(state)
            action = q_value.squeeze().cpu().argmax().item()
        else:
            action = random.randrange(self.act_dim)
        return action

    def compute_msbe_loss(self):
        """Compute the MSBE loss between self.dqn_net predictions and expected Q-values.
        
        Returns
        -------
        float
            The MSE between Q-value prediction and expected Q-values.
        """
        obs, actions, rewards, next_obs, done = self.replay_buffer.get(self.batch_size)

        # Convert to Tensors
        obs = torch.stack([torch.Tensor(ob) for ob in obs]).to(self.device)
        next_obs = torch.stack([torch.Tensor(next_ob) for next_ob in next_obs]).to(self.device)
        rewards = torch.Tensor(rewards)
        # Will have 1.0 if done and 0.0 if not done
        done = torch.Tensor(done)

        # Compute q_values and next_q_values
        q_values = self.dqn_net(obs).cpu()

        #########################################
        ###################TODO##################
        #########################################
        # Hint: You will have to hinder Pytorch to compute gradients in the following, e.g. through torch.no_grad()
        next_q_values = None #
        max_actions = None # torch.max(self.dqn_net(next_obs) , dim = 1)[1]

        with torch.no_grad():
            # TODO: implement here
            next_q_values = self.dqn_net(next_obs).cpu()
            next_q_values_target = self.dqn_target_net(next_obs).cpu()
            max_actions = np.argmax(next_q_values , axis = 1)
        #########################################

        # Has to be torch.LongTensor in order to being able to use as index for torch.gather()
        actions = torch.LongTensor(actions)
        # Select Q-values of actions actually taken
        #print(q_values.shape, actions.shape)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Calculate max over next Q-values

        #########################################
        ########This block has changed###########
        #########################################
        # The following line is the Q-value target used in the original DQN.
        # next_q_values = next_q_values.max(1)[0]
        # In DoubleDQN, we the changed target:
        next_q_values = next_q_values_target.gather(1, max_actions.unsqueeze(1)).squeeze(1)
        #########################################

        # The target we want to update our network towards
        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - done)

        # Calculate loss
        loss = F.mse_loss(q_values, expected_q_values)
        return loss


def epsilon_by_timestep(timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=1000000):
    """Linearily decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps"""
    return max(epsilon_final, epsilon_start - (timestep / frames_decay) * (epsilon_start - epsilon_final))

def create_atari_env(env_id="PongNoFrameskip-v4"):
    config = dict(
        mode=0,
        difficulty=0,
        repeat_action_probability=0.0,
        obs_type='image',
    )
    env = gym.make(env_id, **config)
    env = AtariPreprocessing(env, scale_obs=True, grayscale_newaxis=True, terminal_on_life_loss=True)
    # Normally, you should use this, but this introduces computational complexity
    env = FrameStack(env, 5)
    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eat. Pray. Double DQN.")
    parser.add_argument("--test", action='store_true', help="Load the agent from 'agent.pth' and visualize it.")
    args = parser.parse_args()

    if args.test:
        env = create_atari_env()
        dqn = DoubleDQN(env, resume=True)
        visualize_agent(env, dqn)

    # Plot epsilon rate over time
    # plt.plot([epsilon_by_timestep(i) for i in range(50000)])
    # plt.show()

    #########################################
    ###############Cartpole##################
    #########################################
    # Train the DQN agent on CartPole-v2 as in the last exercise
    # env_id = "CartPole-v1"
    # env = gym.make(env_id)
    # dqn = DoubleDQN(env)
    # dqn.learn(10000)
    # visualize_agent(env, dqn)

    #########################################
    ################Atari####################
    #########################################
    # Now on to Atari
    env = create_atari_env()
    dqn = DoubleDQN(env)
    dqn.learn(10000000)