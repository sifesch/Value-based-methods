from unityagents import UnityEnvironment
import numpy as np
from utils import create_training_plot

env = UnityEnvironment(file_name="Banana_Linux/Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent

agent = Agent(state_size=37, action_size=4, seed=0)

def dqn(n_episodes: int =300, max_t: int = 1000, eps_start: float =1.0, eps_end: float = 0.01, eps_decay: float = 0.995, trial_number: str = '03') -> list:
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # Reset env, extract the data  and extract the state.
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations[0]  
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            # Retrive step result, extract next state and the reward
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0] 
            done = env_info.local_done[0] 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'results/checkpoint_{trial_number}.pth')
            break
    return scores

if __name__ == '__main__':
    ####### Define the Trial Number and the Parameters for Training the DQN Agent #######
    trial_number = '03'
    n_episodes = 1800
    max_t = 1200
    eps_start = 1.0
    eps_end = 0.02
    eps_decay = 0.9

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]            # get the current state 
    scores = dqn(n_episodes=n_episodes, max_t = max_t, eps_start=eps_start, eps_end = eps_end, eps_decay = eps_decay)

    # Save results
    np.save(f"results/scores_trial_{trial_number}.npy", np.array(scores))
    create_training_plot(scores = scores, trial_num= trial_number)

    env.close()