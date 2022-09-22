import numpy as np 


class ReplayMemory(): 
    def __init__(self, max_size, critic_dims, actor_dims, num_actions, num_agents, batch_size): 
        self.memory_size = max_size
        self.memory_counter = 0 
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.batch_size = batch_size 
        self.actor_dims = actor_dims
        self.state_memory = np.zeros((self.memory_size, critic_dims))
        self.new_state_memory = np.zeros((self.memory_size, critic_dims))
        self.reward_memory = np.zeros((self.memory_size, num_agents))
        self.terminal_memory = np.zeros((self.memory_size, num_agents), dtype=bool)
        self.init_actors_memory()

    # Function to initialize and store the memory of each individual actors 
    def init_actors_memory(self): 
        self.actor_state_memory = []
        self.actor_new_state_memory = [] 
        self.actor_action_memory = [] 
        for i in range(self.num_agents): 
            self.actor_state_memory.append(np.zeros((self.memory_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.memory_size, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.memory_size, self.num_actions)))

    # Function to store memory of each episode 
    def store_transitions(self, raw_obs, state, action, reward, raw_obs_, state_, done): 
        index = self.memory_counter % self.memory_size 
        for i in range(self.num_agents):
            self.actor_state_memory[i][index] = raw_obs[i]
            self.actor_new_state_memory[i][index] = raw_obs_[i]
            self.actor_action_memory[i][index] = action[i]

        self.state_memory[index] = state 
        self.new_state_memory[index] = state_ 
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done 
        self.memory_counter += 1 

    # Function to sample batch of training data
    def sample_memory(self): 
        index = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(index, self.batch_size, replace=False)
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        actor_states = []
        actor_states_ = []
        actions = [] 

        for i in range(self.num_agents): 
            actor_states.append(self.actor_state_memory[i][batch])
            actor_states_.append(self.actor_new_state_memory[i][batch])
            actions.append(self.actor_action_memory[i][batch])

        return states, states_, rewards, terminal, actor_states, actor_states_, actions

    def can_be_sampled(self): 
        return self.memory_counter > self.batch_size 

    
        

