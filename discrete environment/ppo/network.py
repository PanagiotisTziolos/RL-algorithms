import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Actor, self).__init__()
             
        self.N_FEATURES = n_features
        self.N_ACTIONS = n_actions
        
        self.HIDDEN_UNITS = 64
        self.LEARNING_RATE = 0.001
        
        # ε parameter that defines the threshold
        # [1-ε, 1+ε] for the clipping value
        self.EPS = 0.2

        self.net = nn.Sequential(
            nn.Linear(self.N_FEATURES, self.HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_UNITS, self.HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_UNITS, self.N_ACTIONS)
        )

        self.optimizer = torch.optim.Adam(self.parameters() ,lr=self.LEARNING_RATE)
    
    def forward(self, x):
        with torch.no_grad():
            action_logits = self.net(torch.tensor(x, dtype=torch.float32))
        dist = Categorical(logits=action_logits)
        action_probs = dist.probs
        
        return action_probs
        
    def act(self, x):
        with torch.no_grad():
            action_logits = self.net(torch.tensor(x, dtype=torch.float32))
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        
        return action.item()
        
    def train(self, states, actions, advantage, old_probs):
        loss = self.calculate_loss(states, actions, advantage, old_probs)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def calculate_loss(self, states, actions, advantage, old_probs):
        states = torch.tensor(states, dtype=torch.float32)
        actions = F.one_hot(torch.tensor(actions), self.N_ACTIONS)
        advantage = torch.tensor(advantage, dtype=torch.float32)

        actions_logits = self.net(states)
        dist = Categorical(logits=actions_logits) 
        probs = torch.sum(dist.probs * actions, dim=-1)
        
        old_dist = Categorical(old_probs)
        
        # Add 1e-8 so that the result is not 0
        old_probs = torch.sum(old_dist.probs * actions, dim=-1) + 1e-8 
        
        ratio = probs / old_probs
        
        sur1 = ratio * advantage
        sur2 = torch.clamp(ratio, 1 - self.EPS, 1 + self.EPS) * advantage

        loss = -torch.mean(torch.min(sur1, sur2))
        
        return loss
    
    def save_model(self):
        torch.save(self.state_dict(), "actor_weights")

    def load_model(self):
        self.load_state_dict(torch.load("actor_weights"))
        

class Critic(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Critic, self).__init__()
             
        self.N_FEATURES = n_features
        self.N_ACTIONS = n_actions
        
        self.HIDDEN_UNITS = 64
        self.LEARNING_RATE = 0.01

        self.net = nn.Sequential(
            nn.Linear(self.N_FEATURES, self.HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_UNITS, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters() ,lr=self.LEARNING_RATE)
    
    def forward(self, x):
        with torch.no_grad():
            action_values = self.net(torch.tensor(x, dtype=torch.float32))

        return action_values.squeeze().tolist()
    
    def train(self, states, rewards):
        loss = self.calculate_loss(states, rewards)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def calculate_loss(self, states, target):
        states = torch.tensor(states, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        actions_values = self.net(states).squeeze()
        
        loss = F.mse_loss(actions_values, target)
        
        return loss
    
    def save_model(self):
        torch.save(self.state_dict(), "critic_weights")

    def load_model(self):
        self.load_state_dict(torch.load("critic_weights"))

