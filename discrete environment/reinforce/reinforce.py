import gym
import numpy as np

from network import Network

class Agent:
    def __init__(self, n_features, n_actions, use_baseline=False):
        self.N_FEATURES = n_features
        self.N_ACTIONS = n_actions
        
        self.GAMMA = 0.99
        
        # Set to True to use baseline
        # The baseline is the mean value of the episode rewards
        self.USE_BASELINE = use_baseline
        
        # Initialize the network
        self.net = Network(self.N_FEATURES, self.N_ACTIONS)
        
        self._clear_memory()

    def take_action(self, state):
        action = self.net.act(state)

        return action

    def train_network(self):     
        advantage = self._calculate_advantage()

        self.net.train(self.episode_states, self.episode_actions, advantage)
            
        self._clear_memory()
    
    def _calculate_advantage(self):
        if self.USE_BASELINE:
            baseline = np.mean(self.episode_rewards)
        else:
            baseline = 0
            
        discounted_rewards = self._calculate_discounted_rewards()
        
        # Advantage = Discounted rewards - Baseline
        advantage = [dr - baseline for dr in discounted_rewards]
        
        mean = np.mean(advantage)
        std = np.std(advantage) + 1e-8
        advantage_norm = [(a - mean) / std for a in advantage]

        return advantage_norm

    def _calculate_discounted_rewards(self):
        discounted_rewards = []
        d_r = 0
            
        for r in self.episode_rewards[: : -1]:
            d_r = r + self.GAMMA * d_r
            discounted_rewards.insert(0, d_r)
            
        return discounted_rewards

    def _clear_memory(self):
        self.episode_states, self.episode_actions, self.episode_rewards = [],[],[]

    def store_in_memory(self, state, action, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def save_model(self):
        self.net.save_model()

    def load_model(self):
        self.net.load_model()
