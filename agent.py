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
    def argmax(self, lst):
        #return np.argmax(np.array(lst))
        return lst.index(max(lst))

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
            l1 = state.index("")
            if l1 == 0:
                return 0
            else:
                if state[l1-1] == 1:
                    return l1 + len(state)
                else:
                    return l1
        else:
            if state[-1] == 1:
                return 2*len(state)
            else:
                return len(state)
    
    def convert(self, state):
        #print(state)
        if self.config[0] == 0:
            return self.convert_a(state)
        elif self.config[0] == 1:
            return state
        else:
            return self.convert_kbc(state)

    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]
        self.n_episodes = 0
        self.prev_action = None
        self.prev_obs = None
        epsilon_dict = {
                0: 0.7,
                1: 0.5,
                2: 0.5,
                3: 0.3,
                4: 0.3
                }
        self.epsilon = epsilon_dict[self.config[0]]
        gamma_dict = {
                0: 0.9,
                1: 1,
                2: 1,
                3: 0.9,
                4: 0.9
                }
        self.gma = gamma_dict[self.config[0]]   
        eta_dict = {
                0: 0.6,
                1: 0.6,
                2: 0.4,
                3: 0.2,
                4: 0.2
                }
        self.eta = eta_dict[self.config[0]]   
        if self.config[0]:
            self.Q = np.zeros((self.config[1], self.config[2]))
            #self.Q = [[0]*self.config[2]]*self.config[1]
            self.n_obs_space = self.config[1]
            self.n_action_space = self.config[2]
        else:
            self.nbins = 4
            self.Q = np.zeros((self.nbins ** 4+1, 3))
            #self.Q = [[0]*3]*(self.nbins ** 4+1)
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

        obs = self.convert(obs)
        #print(obs)
        if random.uniform(0, 1) < self.epsilon:
            self.prev_action = random.randint(0, self.n_action_space-1)
        else:
            self.prev_action = np.argmax(self.Q[obs,:])
            #self.prev_action = self.argmax(self.Q[obs])

        self.prev_obs = obs
        self.n_episodes += 1
        return int(self.prev_action)

    def compute_action_train(self, obs, reward, done, info):
        """
        Use this function in the train phase
        This function is called at all subsequent st]ps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        obs = self.convert(obs)
        #print(obs, self.prev_action)
        self.Q[self.prev_obs,self.prev_action] = self.Q[self.prev_obs,self.prev_action] + self.eta*(reward + self.gma*np.max(self.Q[obs,:]) - self.Q[self.prev_obs,self.prev_action])

        #self.Q[self.prev_obs][self.prev_action] = self.Q[self.prev_obs][self.prev_action] + self.eta*(reward + self.gma*max(self.Q[obs]) - self.Q[self.prev_obs][self.prev_action])

        if random.uniform(0, 1) < self.epsilon:
            self.prev_action = random.randint(0, self.n_action_space-1)
        else:
            self.prev_action = np.argmax(self.Q[obs,:])
            #self.prev_action = self.argmax(self.Q[obs])
        
        self.prev_obs = obs
        return int(self.prev_action)

    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        obs = self.convert(obs)
        #print(self.Q)
        action = np.argmax(self.Q[obs,:])
        #action = self.argmax(self.Q[obs])
        return int(action)

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

        obs = self.convert(obs)
        action = np.argmax(self.Q[obs,:])
        #action = self.argmax(self.Q[obs])

        return int(action)
