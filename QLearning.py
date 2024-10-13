import gym
import random
import numpy as np

enviroment  = gym.make("FrozenLake-v1",is_slippery = False,render_mode = "ansi")
enviroment.reset()

nb_states = enviroment.observation_space.n
nb_actions = enviroment.action_space.n
qtable = np.zeros((nb_states,nb_actions))

print("QTable: ")
print(qtable)

action = enviroment.action_space.sample()

new_state, reward, done, info, _ = enviroment.step(action)

#%%
import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
enviroment  = gym.make("FrozenLake-v1",is_slippery = False,render_mode = "ansi")
enviroment.reset()

nb_states = enviroment.observation_space.n
nb_actions = enviroment.action_space.n
qtable = np.zeros((nb_states,nb_actions))

print("QTable: ")
print(qtable)

episodes = 10000
alpha = 0.5
gamma = 0.6

outcomes = []

#training
for _ in tqdm(range(episodes)):
    state, _ = enviroment.reset()
    done = False
    outcomes.append("Failure")
    
    while not done:
        
        if np.argmax(qtable[state])>0:
            action = np.argmax(qtable[state])
        else:
            action = enviroment.action_space.sample()

        new_state, reward, done, info, _ = enviroment.step(action)
        
        qtable[state,action] = qtable[state,action] + alpha *(reward+gamma*np.max(qtable[new_state])-qtable[state,action])  

        state = new_state

        if reward:
            outcomes[-1] = "Success"

print("QTable After Training: ")
print(qtable)

plt.bar(range(episodes),outcomes)







 
