import random
import numpy as np
from sklearn.neighbors import KDTree


class ReplayBuffer:
    def __init__(self, max_length):
        self.max_length = max_length
        self.memory = list()

    def __len__(self):
        return len(self.memory)

    def enqueue(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.max_length:
            del self.memory[0]

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)


class DifferentiableNeuralDictionary:
    def __init__(self, size, key_size, tau=.5, k=50):
        self.tau = tau
        self.k = k
        self.max_size = size
        self.key_size = key_size
        self.current_size = 0

        # memory
        self.embeddings = np.zeros((size, key_size))
        self.q_values = np.zeros(size)
        self.tree = None
        self.weights = np.zeros(size)

        # lru memory
        self.lru = np.zeros(size)
        self.tm = .0

    def write(self, key, value):
        # TODO Key already exists
        # index = (self.embeddings == key).all(axis=1).nonzero()
        distance = self.tree.query(key, k=1, return_distance=False)
        if distance < self.tau:
            if self.current_size < self.max_size:
                self.embeddings[self.current_size + 1] = key
                self.q_values[self.current_size + 1] = value
                self.current_size += 1
            else:
                index = np.argmin(self.lru)
                self.embeddings[index] = key
                self.q_values[index] = value
                self.lru[index] = self.tm

            if self.is_queryable():
                self.rebuild_tree()

            self.tm += .01

    def lookup(self, key):
        if not self.is_queryable():
            raise Exception('DND is not Queryable')

        distances, indices = self.tree.query(key, k=self.k,
                                             return_distance=True)
        q_values = self.q_values[indices]

        self.lru[distances] = self.tm
        self.tm += .01

        return distances, indices, q_values

    def is_queryable(self):
        return self.current_size > self.k

    def rebuild_tree(self):
        self.tree = KDTree(self.embeddings[:self.current_size])

    def attend(self, key):
        if not self.is_queryable():
            raise Exception('DND is not Queryable')
        _, indices, q_values = self.lookup(key)
        # TODO Check if there is a better way (maybe use kernel)
        distances, all_indices = self.tree.query(key, self.current_size,
                                                 return_distance=True)
        self.weights[all_indices] = distances / np.sum(distances)
        return np.sum(self.weights[indices].T * q_values[indices])

    def tabular_update(self, g_n, gamma):
        self.q_values[:self.current_size] = \
            self.q_values[:self.current_size] + gamma * self.weights[:self.current_size].T * \
            (g_n - self.q_values[:self.current_size])


class Q_Memory:
    def __init__(self, actions, mem_size, key_size, k, tau):
        self.memory = dict()
        self.tau = tau
        for action in actions:
            self.memory[action] = DifferentiableNeuralDictionary(size=mem_size,
                                                                 key_size=key_size,
                                                                 tau=tau,
                                                                 k=k)
    def query(self, state, action):
        pass

    def insert(self, state, action, value):
        self.memory[action].write(state, value)



if __name__ == '__main__':
    pass
