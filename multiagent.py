import torch 
import torch.nn.functional as func
from agent import Agent


class MultiAgent(): 
    def __init__(self, actor_obs_space, total_actor_dims, num_agents, num_actions, env='simple', 
            alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, checkpoint='tmp/multiagent/'):
        self.agents = [] 
        self.num_agents = num_agents
        self.num_actions = num_actions
        checkpoint_dir += env 
        for i in range(self.num_agents): 
            self.agents.append(Agent(actor_obs_space[i], total_actor_dims, num_actions, num_agents, i, alpha=alpha, beta=beta, checkpoint=checkpoint_dir))

    def save_checkpoint(self): 
        print('saving checkpoint...')
        for agent in self.agents: 
            agent.save_models()

    def load_checkpoint(self): 
        print('loading checkpoint...')
        for agent in self.agents: 
            agent.load_models()

    # Iterate through agents and return list of actions by each agent 
    def choose_action(self, raw_obs): 
        actions = [] 
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[i])
            actions.append(action)
        
        return actions 

    
    def learn(self, memory): 
        if not memory.can_be_sampled(): 
            return 
        
        states, states_, rewards, terminal, actor_states, actor_states_, actions = memory.sample_memory() 

        device = self.agents[0].actor.device 
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device) 
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device) 
        terminal = torch.tensor(terminal, dtype=bool).to(device) 

        agents_actions_ = []
        agents_mu_actions_ = [] 
        agents_actions = [] 

        for i, agent in enumerate(self.agents): 

            new_states = torch.tensor(actor_states_[i], dtype=torch.float).to(device) 
            pi_ = agent.target_actor.forward(new_states)
            agents_actions_.append(pi_)
            mu_states = torch.tensor(actor_states[i], dtype=torch.float).to(device) 
            pi = agent.actor.forward(mu_states)
            agents_mu_actions_.append(pi)
            agents_actions.append(actions[i])

        new_actions = torch.cat([act for act in agents_actions_], dim=1)
        mu = torch.cat([act for act in agents_mu_actions_], dim=1)
        prev_actions = torch.cat([act for act in agents_actions], dim=1)

        for i, agent in enumerate(self.agents): 
            q_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            q_value_[terminal[:,0]] = 0.0 
            q_value = agent.critic.forward(states, prev_actions).flatten()

            # Use the y value to calculate the mse loss for theta(parameters of the actor mu network) 
            y = rewards[:,i] + agent.gamma * q_value_  
            critic_loss = func.mse_loss(y, q_value) 
            agent.critic.optimizer.zero_grad() 
            critic_loss.backward(retain_graph=True) 
            agent.critic.optimizer.step() 

            # Update actor network using the sampled critic(actor(states)) and take grident wrt theta(parameters of actor network)
            actor_loss = agent.critic.forward(states, mu).flatten() 
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad() 
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step() 

            agent.update_parameters()

        










    

