from make_env import make_env
from multiagent import MultiAgent
from replay import ReplayMemory
from utils import obs_list_to_state_vector
import numpy as np

if __name__ == '__main__': 

    MAX_MEMORY_SIZE = 1000000
    BATCH_SIZE = 1024
    PRINT_INTERVAL = 500
    NUM_GAMES = 30000
    MAX_STEPS = 25 

    scene = 'simple'
    env = make_env(scene)
    num_agents = env.n 
    actor_observation_dims = []
    for i in range(num_agents): 
        actor_observation_dims.append(env.observation_space[0].shape[0])
    total_observation_dims = sum(actor_observation_dims)
    num_actions = env.action_space[0].n

    multi_agent = MultiAgent(actor_observation_dims, total_observation_dims, num_agents, num_actions)
    memory = ReplayMemory(MAX_MEMORY_SIZE, total_observation_dims, actor_observation_dims, num_agents, BATCH_SIZE)

    total_steps = 0 
    score_history = []
    eval = False 
    best_score = 0 
    
    if eval:
        multi_agent.load_checkpoint() 

    for i in range(NUM_GAMES): 
        obs = env.reset() 
        score = 0 
        done = [False] * num_agents
        episode_step = 0 
        # Check if any of the agents have terminated 
        while not any(done): 
            if eval: 
                env.render() 
            actions = multi_agent.choose_action(obs)
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step > MAX_STEPS: 
                done = [True] * num_agents
            
            memory.store_transitions(obs, state, actions, reward, obs_, state_, done)
            
            if total_steps % 100 == 0 and not eval: 
                multi_agent.learn(memory)
            
            obs = obs_ 
            score += sum(reward)
            total_steps += 1
            episode_step += 1 

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if not eval: 
            if avg_score > best_score: 
                multi_agent.save_checkpoint() 
                best_score = avg_score 

        if i % PRINT_INTERVAL == 0 and i > 0 :
            print(f'episode {i} with score of {avg_score}')


