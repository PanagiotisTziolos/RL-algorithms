import numpy as np
import random

from network import Network

class AgentDQN:
    def __init__(self, n_features, n_actions):
        self.N_FEATURES = n_features
        self.N_ACTIONS = n_actions
        
        self.GAMMA = 0.99
        self.EPS = 0.99
        self.EPS_DECAY = 0.0001
        
        # Number of data that is required to be in the memory 
        # in order to start the training
        self.START_TRAINING = 400
        
        # Number of data that are used for the training
        self.BATCH_SIZE = 64
        
        # Number of times that the q network has been trained.
        # A counter that helps with the update of the target network
        self.N_TRAINS = 0
        
        # Target network's weights are updated every 1000 steps of training
        self.UPDATE_FREQ = 1000 
        
        # Initialize the networks
        self.net = Network(self.N_FEATURES, self.N_ACTIONS)
        self.target_net = Network(self.N_FEATURES, self.N_ACTIONS)
        
        # Copy the q network's weights to the target network
        self.target_net.set_dict(self.net.get_dict())
        
        self._clear_memory()

    def take_action(self, state):
        action_values = self.net(state)
        
        if random.random() < self.EPS:
            action = random.randint(0, self.N_ACTIONS-1)
        else:
            action = np.argmax(action_values)
            
        return action

    def train_network(self):
        if len(self.episode_states) >= self.START_TRAINING:
            self.N_TRAINS += 1
            
            s, a, r, ns, d = self._get_batch()
                  
            target = self._calculate_target(r, ns, d)

            self.net.train(s, a, target)
            
            # Update the target's weights
            if self.N_TRAINS % self.UPDATE_FREQ == 0:  self.target_net.set_dict(self.net.get_dict())
            
            # Decay the ε value
            if self.EPS > 0.001 : self.EPS -= self.EPS_DECAY
    
    def _calculate_target(self, reward, n_states, done):
        action_values = self.target_net(n_states)
        max_values = [max(action_values[i]) for i in range(len(action_values))]
        
        target = [r + self.GAMMA * m_v * (1 - d) for r, m_v, d in zip(reward, max_values, done)]

        return target

    def _clear_memory(self):
        self.episode_states, self.episode_actions, self.episode_rewards = [],[],[]
        self.episode_n_states, self.episode_done = [], []

    def store_in_memory(self, state, action, reward, ns, done):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_n_states.append(ns)
        self.episode_done.append(done)
        
    def _get_batch(self):
        indices = np.random.randint(len(self.episode_states), size=self.BATCH_SIZE)
        
        s, a, r, ns, d = [], [], [], [], []
        
        for index in indices:
            s.append(self.episode_states[index])
            a.append(self.episode_actions[index])
            r.append(self.episode_rewards[index])
            ns.append(self.episode_n_states[index])
            d.append(self.episode_done[index])
            
        return s, a, r, ns, d

    def save_model(self):
        self.net.save_model()

    def load_model(self):
        self.net.load_model()
        
