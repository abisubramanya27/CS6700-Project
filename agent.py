from config import *
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
    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]
        self.n_episodes = 0
        self.prev_action = None
        self.prev_obs = None
        self.epsilon = 0.1
        self.gma = 0.9   # earlier 0.6
        if self.config[0]:
            self.Q = np.zeros((self.config[1], self.config[2]))
            self.n_obs_space = self.config[1]
            self.n_action_space = self.config[2]
            self.eta = 0.2  # earlier 0.1
        else:
            raise NotImplementedError
        pass

    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        if self.config[0]:
            if random.uniform(0, 1) < self.epsilon:
                self.prev_action = random.randint(0, self.n_action_space-1)
            else:
                self.prev_action = np.argmax(self.Q[obs,:])

            self.prev_obs = obs
            self.n_episodes += 1
        else:
            raise NotImplementedError
        
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

        if self.config[0]:
            self.Q[self.prev_obs,self.prev_action] = self.Q[self.prev_obs,self.prev_action] + \
                    self.eta*(reward + self.gma*np.max(self.Q[obs,:]) - \
                        self.Q[self.prev_obs,self.prev_action])

            if random.uniform(0, 1) < self.epsilon:
                self.prev_action = random.randint(0, self.n_action_space-1)
            else:
                self.prev_action = np.argmax(self.Q[obs,:])
            
            self.prev_obs = obs
        else:
            raise NotImplementedError

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

        if self.config[0]:
            action = np.argmax(self.Q[obs,:])
        else:
            raise NotImplementedError

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

        if self.config[0]:
            action = np.argmax(self.Q[obs,:])
        else:
            raise NotImplementedError
        return action
