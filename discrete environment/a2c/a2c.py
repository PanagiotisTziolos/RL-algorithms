import numpy as np

from network import Actor, Critic

class Agent:
    def __init__(self, n_features, n_actions):
        self.N_FEATURES = n_features
        self.N_ACTIONS = n_actions
        self.GAMMA = 0.99
        
        self.actor = Actor(n_features, n_actions)
        self.critic = Critic(n_features, n_actions)
        
        self._clear_memory()

    def take_action(self, state):
        action = self.actor.act(state)
        
        return action

    def train_network(self):    
        discounted_rewards = self._normalize(self._calculate_discounted_rewards())
        advantage = self._normalize(self._calculate_advantage(discounted_rewards))

        self.actor.train(self.episode_states, self.episode_actions, advantage)
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
        std = np.std(array) + 1e-8
        array = np.array(array)
        
        array_norm = (array - mean) / std

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
