import random
import numpy as np
from Memories import Q_Memory, ReplayBuffer
from tqdm import tqdm
from keras.optimizers import Adam
from keras.losses import MSE
import tensorflow as tf


class NeuralEpisodicControl:
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.q_network = kwargs['q_network']
        self.replay_buffer = ReplayBuffer(50000)
        self.memory = Q_Memory(kwargs['actions'],
                               mem_size=kwargs['dnd_size'],
                               key_size=kwargs['key_size'],
                               k=kwargs['k'],
                               tau=kwargs['tau'])

        # params
        self.n_steps = kwargs['n_steps']
        self.gamma = kwargs['gamma']
        self.alpha = kwargs['alpha']
        self.dnd_size = kwargs['dnd_size']
        self.tau = kwargs['tau']

        # Epsilon Decay
        self.eps_start = kwargs['eps_start']
        self.eps_decay = kwargs['eps_decay']
        self.eps_end = kwargs['eps_end']

        self.batch_size = kwargs['batch_size']

        self.optimizer = Adam(lr=1e-3)
        self.loss_fn = MSE

        # stats
        self.rewards = []

    def train(self, episodes=1000):
        done = False
        for i in tqdm(range(episodes), total=episodes):
            player = random.choice([True, False])
            observation = self.env.reset(player, not player)
            if not player:
                action = random.choice(self.env.get_legal_moves())
                observation, _, done = self.env.step(action)

            self.rewards.append(self.play_episode(observation, i, player, done))

    def test(self):
        pass

    def play_episode(self, state, episode, player, done):
        steps = 0
        cumulative_reward = 0
        while True:
            # n_steps loop
            g_n = 0
            for n in range(1, self.n_steps + 1):

                if player:
                    # agent turn

                    action = self.select_action(state, episode)
                    # save first state/action
                    if n == 0:
                        first_action = action
                        first_state = state

                    next_state, reward, done = self.env.step(action, player)
                    player = not player

                # opponent turn
                if not done:
                    opponent_action = self.select_action(state, episode)
                    next_state, reward, done = self.env.step(opponent_action, player)
                    player = not player

                cumulative_reward += reward
                steps += 1

                # cumulative discounted reward
                g_n += (self.gamma ** n) * reward
                state = next_state

                if done:
                    break

            actions = self.env.get_legal_moves()
            if actions:
                # Calculate Bellman Target for action u (first action)
                attention = self.memory.get_attention(next_state, actions)
                bellman_target = g_n + (self.alpha ** n) * np.min(attention)
            else:
                # if no actions query returns 0
                bellman_target = g_n

            self.memory.insert(state, bellman_target)

            # Add to replay memory
            self.replay_buffer.enqueue((first_state, first_action, bellman_target, state))

            # Tabular Update
            self.memory.tabular_update(first_action, g_n)

            self.learn()

            if done:
                break

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch_initial_state, batch_action, batch_reward, batch_next_state = self.get_transitions()

        with tf.GradientTape() as tape:
            predicted_q_values = self.memory.get_attention(batch_initial_state, batch_action)

            loss_val = self.loss_fn(predicted_q_values, batch_reward)

        grads = tape.gradient(loss_val, self.q_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.q_network.trainable_weights))




    def select_action(self):
        pass

    def get_transitions(self):
        transitions = self.replay_buffer.sample(self.batch_size)
        states, f_actions, q_values, actions = list(zip(*transitions))
        return np.stack(states), list(f_actions), np.array(q_values), list(actions)


if __name__ == '__main__':
    pass
