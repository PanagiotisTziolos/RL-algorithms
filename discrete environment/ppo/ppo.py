import numpy as np

from network import Actor, Critic

class Agent:
    def __init__(self, n_features, n_actions):
        self.N_FEATURES = n_features
        self.N_ACTIONS = n_actions
        
        self.GAMMA = 0.99
        
        # Number of times that the networks are trained
        # on the same data set
        self.EPOCHS = 5
        
        # Initialize the networks
        self.actor = Actor(n_features, n_actions)
        self.critic = Critic(n_features, n_actions)
        
        self._clear_memory()

    def take_action(self, state):
        action = self.actor.act(state)
        
        return action

    def train_network(self):
        old_probs = self.actor(self.episode_states)

        discounted_rewards = self._normalize(self._calculate_discounted_rewards())
        advantage = self._calculate_advantage(discounted_rewards)

        for _ in range(self.EPOCHS):
            self.actor.train(self.episode_states, self.episode_actions, advantage, old_probs)
            self.critic.train(self.episode_states, discounted_rewards)
            
        self._clear_memory()
    
    def _calculate_advantage(self, discounted_rewards):
        values = self.critic(self.episode_states)
        
        advantage = [dr - v for dr, v in zip(discounted_rewards, values)]

        return advantage

    def _calculate_discounted_rewards(self):
        discounted_rewards = []
        d_r = 0
            
        for r in self.episode_rewards[: : -1]:
            d_r = r + self.GAMMA * d_r
            discounted_rewards.insert(0, d_r)
            
        return discounted_rewards
        
    def _normalize(self, array):
        mean = np.mean(array)
        # Add 1e-8 so that the result is not 0
        std = np.std(array) + 1e-8
        array_norm = [(element - mean) / std for element in array]
        
        return array_norm

    def _clear_memory(self):
        self.episode_states, self.episode_actions, self.episode_rewards = [],[],[]

    def store_in_memory(self, state, action, reward, ns, done):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.done = done

    def save_model(self):
        self.actor.save_model()
        self.critic.save_model()

    def load_model(self):
        self.actor.load_model()
        self.critic.load_model()
