import numpy as np

def encode_vector(index, dim):
    # Encode index as one-hot vector
    vector_encoded = np.zeros((dim, ))
    vector_encoded[index] = 1

    return vector_encoded

class Policy:

    def __init__(self, theta, n_actions, alpha):
        # Initialize paramters theta, learning rate alpha and discount factor gma
        self.theta = theta
        self.alpha = alpha
        self.n_actions = n_actions

    def probs(self, state):
        # Returns P_{theta}(action | state)
        probs = np.zeros((self.n_actions, ))
        for act in range(self.n_actions):
            act_vector = encode_vector(act, self.n_actions)
            probs[act] = np.dot(self.theta[state], act_vector)

        probs = np.exp(probs - np.max(probs))
        return probs / np.sum(probs)

    def act(self, state):
        # sample an action in proportion to probabilities
        probs = self.probs(state)
        action = np.random.choice(list(range(self.n_actions)), p=probs)

        return action, probs[action]

    def grad_log_p(self, state, action):
        # calculate grad-log-probs
        phi = encode_vector(action, self.n_actions)
        weighted_phi = np.zeros_like(phi)

        probs = self.probs(state)
        
        for act in range(self.n_actions):
            act_vector = encode_vector(act, self.n_actions)
            weighted_phi += act_vector * probs[act]

        grad_log_p = phi - weighted_phi

        return grad_log_p

    def update(self, state, action, G, grad_log_p):
        # update theta for given state, action, G 
        self.theta[state] += self.alpha * G * self.grad_log_p(state, action)
