from config import *
import time
import random
import numpy as np
from policy import Policy

def cos_sin_to_theta(cos, sin):
    return np.arctan2(cos, sin)

def bin(x, n_bins, lo = -1, hi = 1):
    lc = (hi - lo) / n_bins
    bin_no = int((x - lo) / lc)
    assert x >= lo and x < hi

    return bin_no

"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The methods "registered_reset_train" and "compute_action_train" are invoked here. 
Complete these functions to train your agent and save the state.

- Test Phase
The methods "registered_reset_test" and "compute_action_test" are invoked here. 
The final scoring is based on your agent's performance in this phase. 
Use the state saved in train phase here. 

"""


class Agent:
    def get_state_a(self, obs):
        new_obs = [cos_sin_to_theta(obs[0], obs[1]),
                   cos_sin_to_theta(obs[2], obs[3]),
                   obs[4], obs[5]
                  ]
        
        return tuple(
            bin(
                new_obs[i],
                self.config['nbins'][i],
                self.config['low'][i], 
                self.config['high'][i]
            ) for i in range(len(new_obs))
        )
    
    def get_state_t(self, obs):
        return (obs, )
    
    def get_state_kbc(self, obs):
        obs = list(obs) + [""]
        ind = obs.index("")
        return (ind, 0 if ind == 0 else obs[ind-1]+1)

    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]
        self.actions = []
        self.states = []
        self.rewards = []
        self.gma = 1.0
        self.n_step = 0
        if self.env_name == 'acrobot':
            self.eps = 0.5
            self.eta = 0.8
            self.alpha = 5e-6
            self.whiten = False
            self.get_state = self.get_state_a
            self.policy = Policy(np.zeros((*self.config['nbins'], self.config['n_actions'])), self.config['n_actions'], self.alpha)
            self.Q = np.random.rand(*self.config['nbins'], self.config['n_actions'])/1000
            self.choice = 1
        elif self.env_name == 'taxi':
            self.eps = 0.5
            self.eta = 0.6
            self.alpha = 1e-1
            self.whiten = True
            self.get_state = self.get_state_t
            self.policy = Policy(np.random.rand(*self.config['state_space'], self.config['n_actions'])/1000, self.config['n_actions'], self.alpha)
            self.Q = np.random.rand(*self.config['state_space'], self.config['n_actions'])/1000
            self.choice = 0
        else:
            self.eps = 0.3
            self.eta = 0.1
            self.alpha = 5e-6
            self.whiten = False
            self.get_state = self.get_state_kbc
            self.policy = Policy(np.random.rand(*self.config['state_space'], self.config['n_actions'])/1000, self.config['n_actions'], self.alpha)
            self.Q = np.random.rand(*self.config['state_space'], self.config['n_actions'])/1000
            self.choice = 0


    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        state = self.get_state(obs)
        self.states = [state]
        if self.choice:
            action, _ = self.policy.act(state)
            self.rewards = []
            self.G = []
        else:
            if random.uniform(0, 1) < self.eps:
                action = random.randint(0, self.config['n_actions']-1)
            else:
                action = np.argmax(self.Q[state])

        self.actions = [action]
        return action

    def compute_action_train(self, obs, reward, done, info):
        """
        Use this function in the train phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        state = self.get_state(obs)
        self.states.append(state)
        if self.choice:
            action, _ = self.policy.act(state)
        else:
            self.Q[self.states[-2] + (self.actions[-1],)] = (1-self.eta)*self.Q[self.states[-2] + (self.actions[-1],)] + \
                    self.eta*(reward + self.gma*np.max(self.Q[state]))

            if random.uniform(0, 1) < self.eps:
                action = random.randint(0, self.config['n_actions']-1)
            else:
                action = np.argmax(self.Q[state])

        self.actions.append(action)
        self.rewards.append(reward)

        if done and self.choice:
            self.G = [self.rewards[-1]]
            for i in range(len(self.rewards)-2, -1, -1):
                self.G.append(self.G[-1]*self.gma + self.rewards[i])
            self.G = np.array(self.G[::-1])

            if self.whiten:
                G_bar = np.mean(self.G)
                G_sigma = np.std(self.G)
                if G_sigma > 0:
                    self.G = (self.G - G_bar) / G_sigma

            for i in range(len(self.rewards)):
                self.policy.update(self.states[i], self.actions[i], self.G[i])

        return action

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        state = self.get_state(obs)
        if self.choice:
            action, _ = self.policy.act(state)
        else:
            action = np.argmax(self.Q[state])

        return action

    def compute_action_test(self, obs, reward, done, info):
        """
        Use this function in the test phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        state = self.get_state(obs)
        if self.choice:
            action, _ = self.policy.act(state)
        else:
            action = np.argmax(self.Q[state])

        return action
