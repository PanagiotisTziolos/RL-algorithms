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
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
    
    def forward(self, x):
        return self.net(torch.tensor(x, dtype=torch.float32)).detach().tolist()
    
    def train(self, states, actions, target):   
        states = torch.tensor(states, dtype=torch.float32)  
        actions = F.one_hot(torch.tensor(actions), self.N_ACTIONS)
        target = torch.tensor(target, dtype=torch.float32)  
        
        y_pred = self.net(states)
        q = torch.sum(y_pred * actions, dim=-1)
        
        loss = F.mse_loss(q, target)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def set_dict(self, weights):
        self.net.load_state_dict(weights)
        
    def get_dict(self):
        return self.net.state_dict()
        
    def save_model(self):
        torch.save(self.state_dict(), "dqn_weights")

    def load_model(self):
        self.load_state_dict(torch.load("dqn_weights"))
        
