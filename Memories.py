import random
import numpy as np
from sklearn.neighbors import KDTree
import tensorflow as tf


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
        if not self.is_queryable():
            self.embeddings[self.current_size] = key
            self.q_values[self.current_size] = value
            self.lru[self.current_size] = self.tm
            self.current_size += 1
            self.tm += .01
            if self.is_queryable():
                self.rebuild_tree()
            return

        distance = self.tree.query(key, k=1, return_distance=False)

        if distance < self.tau:
            if self.current_size < self.max_size:
                self.embeddings[self.current_size] = key
                self.q_values[self.current_size] = value
                self.lru[self.current_size] = self.tm
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


class NeuralEpisodicControl:
    # TODO refactor __init__
    def __init__(self, optimizer, q_network, loss_fn, key_size, dnd_size, batch_size=64, buffer_size=50000, k=50,
                 **kwargs):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.q_network = q_network
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.key_size = key_size
        self.gamma = kwargs['gamma']
        self.tau = kwargs['tau']

        self.memory = dict()

        with open('allmoves.txt', 'r') as f:
            data = f.read()
            actions = data.split('\n')

        for action in actions:
            self.memory[action] = DifferentiableNeuralDictionary(size=dnd_size,
                                                                 key_size=self.key_size,
                                                                 tau=self.tau,
                                                                 k=k)

    def get_attention(self, batch_state, batch_action):
        attention = np.array(len(batch_action))
        keys = self.q_network(batch_state)
        if keys.shape[0] == 1:
            for i, action in enumerate(batch_action):
                attention[i] = self.memory[action].attend(keys[0])
        else:
            for i, action in enumerate(batch_action):
                attention[i] = self.memory[action].attend(keys[i])

        return attention

    def insert(self, state, action, value):
        key = self.q_network(state)
        self.memory[action].write(key, value)

    def tabular_update(self, first_action, g_n):
        self.memory[first_action].tabular_update(g_n, self.gamma)

    def learn(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        batch_initial_state, batch_action, batch_reward, batch_next_state = self.get_transitions()

        with tf.GradientTape() as tape:
            predicted_q_values = self.get_attention(batch_initial_state, batch_action)

            loss_val = self.loss_fn(predicted_q_values, batch_reward)

        grads = tape.gradient(loss_val, self.q_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))

    def get_transitions(self):
        transitions = self.replay_buffer.sample(self.batch_size)
        states, f_actions, q_values, actions = list(zip(*transitions))
        return np.stack(states), list(f_actions), np.array(q_values), list(actions)


if __name__ == '__main__':
    pass
