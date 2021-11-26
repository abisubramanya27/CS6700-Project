from config import *
import math
import time
import random
import numpy as np

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
    def convert_a(self, state):
        a1 = (state[0]+1)*self.nbins//2
        if a1 == self.nbins:
            a1 -= 1
        a2 = (state[2]+1)*self.nbins//2
        if a2 == self.nbins:
            a2 -= 1
        a3 = (state[4]+1)*self.nbins//26
        if a3 == self.nbins:
            a3 -= 1
        a4 = (state[5]+1)*self.nbins//57
        if a4 == self.nbins:
            a4 -= 1
        return int(a1*(self.nbins ** 3) + a2*(self.nbins ** 2) + a3 *(self.nbins) + a4) 
    def convert_kbc(self, state):
        if "" in state: 
            return state.index("")
        else:
            return len(state)

    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]
        self.n_episodes = 0
        self.prev_action = None
        self.prev_obs = None
        self.epsilon = 0.3
        self.gma = 0.9   # earlier 0.6
        self.eta = 0.2  # earlier 0.1
        if self.config[0]:
            self.Q = np.zeros((self.config[1], self.config[2]))
            self.n_obs_space = self.config[1]
            self.n_action_space = self.config[2]
        else:
            self.nbins = 5
            self.Q = np.zeros((self.nbins ** 4+1, 3))
            self.n_action_space = self.config[2]

    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        if not self.config[0]:
            obs = self.convert_a(obs)
        if self.config[0] >= 2:
            obs = self.convert_kbc(obs)
        #print(obs)
        if random.uniform(0, 1) < self.epsilon:
            self.prev_action = random.randint(0, self.n_action_space-1)
        else:
            self.prev_action = np.argmax(self.Q[obs,:])

        self.prev_obs = obs
        self.n_episodes += 1
        return self.prev_action

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
        if not self.config[0]:
            obs = self.convert_a(obs)
        if self.config[0] >= 2:
            obs = self.convert_kbc(obs)
        #print(obs, self.prev_action)
        self.Q[self.prev_obs,self.prev_action] = self.Q[self.prev_obs,self.prev_action] + \
            self.eta*(reward + self.gma*np.max(self.Q[obs,:]) - \
                self.Q[self.prev_obs,self.prev_action])

        if random.uniform(0, 1) < self.epsilon:
            self.prev_action = random.randint(0, self.n_action_space-1)
        else:
            self.prev_action = np.argmax(self.Q[obs,:])
        
        self.prev_obs = obs
        return self.prev_action

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        if not self.config[0]:
            obs = self.convert_a(obs)
        if self.config[0] >= 2:
            obs = self.convert_kbc(obs)
        action = np.argmax(self.Q[obs,:])
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

        if not self.config[0]:
            obs = self.convert_a(obs)
        if self.config[0] >= 2:
            obs = self.convert_kbc(obs)
        action = np.argmax(self.Q[obs,:])
        return action
