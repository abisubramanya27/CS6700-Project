import numpy as np

def encode_vector(index, dim):
    # Encode index as one-hot vector
    vector_encoded = np.zeros((dim, ))
    vector_encoded[index] = 1

    return vector_encoded

class LinearPolicy:

    def __init__(self, W, b, n_actions):
        # Initialize paramters W, b; learning rate alpha and discount factor gma
        self.W = W
        self.b = b
        self.n_actions = n_actions

    def probs(self, state):
        # Returns P_{w,b}(action | state)
        probs = np.zeros((self.n_actions, ))
        state = np.array(state)
        probs = self.W @ state + self.b

        probs = np.exp(probs - np.max(probs))
        return probs / np.sum(probs)

    def act(self, state):
        # sample an action in proportion to probabilities
        probs = self.probs(state)
        action = np.random.choice(list(range(self.n_actions)), p=probs)

        return action, probs[action]

    def grad_log_p(self, state, action):
        # calculate grad-log-probs
        e = encode_vector(action, self.n_actions)
        x = np.array(state)

        probs = self.probs(state)
        grad_log_p = (
            (e - probs).reshape((self.n_actions,1)) @ x.reshape((1,*x.shape)),
            (e - probs)
        )

        return grad_log_p

    def update(self, state, action, G, grad_log_p, alpha):
        # update W, b for given state, action, G
        self.W += alpha * G * grad_log_p[0]
        self.b += alpha * G * grad_log_p[1]
