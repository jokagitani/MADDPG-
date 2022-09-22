import numpy as np 

# Convert list of 3 np array of observation space to single np array of 28 elem 
def obs_list_to_state_vector(observation): 
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state 
