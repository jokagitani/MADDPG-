import torch 
from networks import Critic, Actor 


class Agent(): 
    def __init__(self, actor_obs_space, total_actor_dims, num_actions, num_agents, agent_index, checkpoint, 
    alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.95, tau=0.01): 
        self.gamma = gamma
        self.tau = tau 
        self.num_actions = num_actions
        self.agent_name = f"agent {agent_index}"
        self.actor = Actor(alpha, actor_obs_space, fc1, fc2, num_actions, name=self.agent_name+'_actor', checkpoint=checkpoint)
        self.critic = Critic(beta, total_actor_dims, fc1, fc2, num_agents, num_actions, name=self.agent_name+'_critic', checkpoint=checkpoint)
        self.target_actor = Actor(alpha, actor_obs_space, fc1, fc2, num_actions, name=self.agent_name+'_target_actor', checkpoint=checkpoint)
        self.target_critic = Critic(beta, total_actor_dims, fc1, fc2, num_actions, name=self.agent_name+'_target_critic', checkpoint=checkpoint)
        self.update_parameters(tau=1)

    # Function to produce output actions with random noise from the actor network 
    def choose_action(self, observation): 
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = torch.rand(self.num_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0]

    # Update parameters of networks using its target networks
    def update_parameters(self, tau=None): 
        if not tau: 
            tau = self.tau 

        target_actor_parameters = dict(self.target_actor.named_parameters())
        actor_parameters = dict(self.actor.named_parameters()) 

        for name in actor_parameters: 
            actor_parameters[name] = tau * actor_parameters[name].clone() + (1-tau) * target_actor_parameters[name].clone()

        self.target_actor.load_state_dict(actor_parameters)

        target_critic_parameters = dict(self.target_critic.named_parameters())
        critic_parameters = dict(self.critic.named_parameters()) 

        for name in critic_parameters: 
            critic_parameters[name] = tau * critic_parameters[name].clone() + (1-tau) * target_critic_parameters[name].clone()

        self.target_critic.load_state_dict(critic_parameters)

    def save_models(self): 
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint() 

    def load_models(self): 
        self.actor.load_checkpoint() 
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

        



            

    