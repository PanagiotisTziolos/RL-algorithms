import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.distributions import Categorical


class Network(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Network, self).__init__()
		     
        self.N_FEATURES = n_features
        self.N_ACTIONS = n_actions
		
        self.HIDDEN_UNITS = 64
        self.LEARNING_RATE = 0.001
        
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
        
        return action_probs.tolist()
        
    def act(self, x):
        with torch.no_grad():
            action_logits = self.net(torch.tensor(x, dtype=torch.float32))
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        
        return action.item()
	
    def train(self, states, actions, advantage):
        loss = self.calculate_episode_loss(states, actions, advantage)
		
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
		
    def calculate_episode_loss(self, states, actions, advantage):
        states = torch.tensor(states, dtype=torch.float32)
        actions = F.one_hot(torch.tensor(actions), self.N_ACTIONS)
        advantage = torch.tensor(advantage, dtype=torch.float32)
		
        actions_logits = self.net(states)
        actions_probs = torch.softmax(actions_logits, dim=-1)
        logp = torch.log(actions_probs + 1e-8) 
        logp = torch.sum(logp * actions, dim=-1)
		
        loss = -torch.sum(advantage * logp)
		
        return loss
	
    def save_model(self):
        torch.save(self.state_dict(), "reinforce_weights")

    def load_model(self):
        self.load_state_dict(torch.load("reinforce_weights"))