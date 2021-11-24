from tensorflow.keras import keras
from keras.layers import Dense

class ActorCritic(keras.model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=1024):
        super(ActorCritic, self).__init__()
        self.fc1_dims = 1024
        self.fc2_dims = 1024
        self.n_actions = n_actions
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(self.n_actions, activation='softmax')
    
    def call(self, state):
        op = self.fc1(state)
        op = self.fc2(op)

        v = self.v(op)
        pi = self.pi(op)

        return v, pi


        

