import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as func 


class Critic(nn.Module): 
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, num_agents, num_actions, name, checkpoint): 
        super(Critic, self).__init__()
        self.checkpoint_file = os.path.join(checkpoint, name)
        self.fc1 = nn.Linear(input_dims + num_agents*num_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q_function = nn.Linear(fc2_dims, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:1') 
        self.to(self.device)

    # Function to feed forward for critic network and return its q value 
    def forward(self, state, action): 
        x = func.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = func.relu(self.fc2(x))
        q_value = self.q_function(x) 

        return q_value 

    def save_checkpoint(self): 
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Actor(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, num_actions, name, checkpoint): 
        super(Actor, self).__init__()
        self.checkpoint = os.path.join(checkpoint, name)
        self.fc1 = nn.Linear(input_dims, fc1_dims) 
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.policy_pi = nn.Linear(fc2_dims, num_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:1')
        self.to(self.device)

    # Function to feed forward for actor network to output softmax actions
    def forward(self, state): 
        x = func.relu(self.fc1(state))
        x = func.relu(self.fc2(x))
        pi = func.softmax(self.policy_pi(x), dim=1)

        return pi 

    def save_checkpoint(self): 
        torch.save(self.state_dict(), self.checkpoint)

    def load_checkpoint(self): 
        self.load_state_dict(torch.load(self.checkpoint))

    