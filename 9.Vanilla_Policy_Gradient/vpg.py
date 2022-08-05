import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

from utils import episode_reward_plot


def compute_returns(rewards, next_value, discount):
    """ Compute returns based on episode rewards.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, bootstrapped value otherwise.
    discount : float
        Discount factor.

    Returns
    -------
    list of float
        Episode returns.
    """

    # TODO

    total_return = 0.0
    return_list = []
    for reward in reversed(rewards):
        total_return = reward + discount * total_return
        return_list.append(total_return)

    return return_list[::-1]

class TransitionMemory():
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma):
        # TODO
        self.gamma = gamma
        #self.transition_list = []
        self.each_episode = []
        self.episodes_rewards = []
        self.traj_start = 0

    def put(self, obs, action, reward, logp):
        """Put a transition into the memory."""
        # TODO
        self.each_episode.append((obs , action , reward , logp))

    def get(self):
        """Get all stored transition attributes in the form of lists."""
        # TODO
        transitions = list(zip(*self.each_episode))
        assert len(transitions[0]) == len(transitions[3])
        return transitions

    def clear(self):
        """Reset the transition memory."""
        # TODO
        #self.transition_list = []
        self.each_episode = []

        self.traj_start = 0

    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).
        
        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """
        # TODO
        self.episodes_rewards.extend(compute_returns(list(self.get())[2] ,
                                                     next_value ,
                                                     self.gamma))
        #self.transition_list.append(self.each_episode)
        self.traj_start = len(self.get()[0])




class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()
        # TODO
        self.net = nn.Sequential(nn.Linear(num_observations , 128) ,
                                 nn.ReLU() ,
                                 nn.Linear(128 , 256),
                                 nn.ReLU(),
                                 nn.Linear(256 , num_actions),
                                 )

    def forward(self, obs):
        # TODO
        obs_tensor = torch.tensor(obs)
        return self.net(obs_tensor)


class VPG():
    """The vanilla policy gradient (VPG) approach."""

    def __init__(self, env, episodes_update=5, gamma=0.99, lr=0.01):
        """ Constructor.
        
        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        episodes_update : int
            Number episodes to collect for every optimization step.
        gamma : float, optional
            Discount factor.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continous actions not implemented!')
        
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.memory = TransitionMemory(gamma)
        self.episodes_update = episodes_update

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)
        

    def learn(self, total_timesteps):
        """Train the VPG agent.
        
        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """
        # TODO
        obs = self.env.reset()
        # For plotting
        overall_rewards = []
        episode_rewards = []

        num_episodes = 0

        for timestep in range(1, total_timesteps + 1):
            # Do one step
            action, logp = self.predict(obs, train_returns=True)
            next_obs, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)

            # TODO put into transition buffer
            self.memory.put(obs , action , reward , logp)
            # Update current obs
            obs = next_obs

            if done:
                # TODO reset environment
                obs = self.env.reset()
                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []

                # TODO finish trajectory
                self.memory.finish_trajectory(0.0)
                num_episodes += 1

                if num_episodes == self.episodes_update:
                    # TODO optimize the actor
                    self.optim_actor.zero_grad()
                    _ , _ , rew_list , logp_list = self.memory.get()
                    loss = self.calc_actor_loss(logp_list , rew_list)
                    loss.backward()
                    self.optim_actor.step()
                    self.memory.clear()
                    num_episodes = 0

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)

    def calc_actor_loss(self, logp_lst, return_lst):
        """Calculate actor "loss" for one batch of transitions."""
        # TODO
        #print(logp_lst.shape)
        # print(return_lst.shape)
        loss = -torch.mean(torch.stack(logp_lst) - torch.Tensor(return_lst))

        return loss

        

    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.
        
        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.
        """

        # TODO
        probs = self.actor_net(obs)
        action_dis = torch.distributions.categorical.Categorical(logits = probs)
        action = action_dis.sample()
        #print(action)
        #print(action.shape)
        if train_returns:
            # Return action, logp
            logp = action_dis.log_prob(action)
            return action.item() , logp
        else:
            # Return action
            return action.item()


if __name__ == '__main__':
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    vpg = VPG(env)
    vpg.learn(100000)