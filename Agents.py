import random
import math
import numpy as np
from Memories import NeuralEpisodicControl
from tqdm import tqdm
from keras.optimizers import Adam
from keras.losses import MSE


class NECAgent:
    # TODO refactor __init__
    def __init__(self, env, q_network, optimizer, loss_fn, key_size=48, n_steps=1,
                 dnd_size=100, gamma=.9, alpha=.5, tau=.5,
                 **kwargs):

        self.current_step = 0
        self.env = env

        # params
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.dnd_size = dnd_size

        # Epsilon Decay
        self.eps_start = .3
        self.eps_decay = 1000
        self.eps_end = .05

        self.batch_size = 64

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.memory = NeuralEpisodicControl(optimizer,
                                            q_network,
                                            loss_fn,
                                            key_size,
                                            dnd_size,
                                            tau=tau,
                                            gamma=self.gamma)

        # stats
        self.rewards = []

    def train(self, episodes=1000):
        self.current_step = 0
        done = False
        for _ in tqdm(range(episodes), total=episodes):
            player = random.choice([True, False])
            observation = self.env.reset(player, not player)
            if not player:
                action = random.choice(self.env.get_legal_moves())

                observation, _, done = self.env.step(action, player)
                player = not player

            self.rewards.append(self.play_episode(observation, player, done))

    def test(self):
        pass

    def play_episode(self, state, player, done):
        steps = 0
        cumulative_reward = 0
        while True:
            # n_steps loop
            g_n = 0
            n = 1
            for n in range(1, self.n_steps + 1):

                if player:
                    # agent turn

                    action = self.select_action(state)
                    # save first state/action
                    if n == 1:
                        first_action = action
                        first_state = state

                    next_state, reward, done = self.env.step(action, player)
                    player = not player

                # opponent turn
                if not done:
                    opponent_action = self.select_action(state)
                    next_state, reward, done = self.env.step(opponent_action, player)
                    player = not player

                cumulative_reward += reward
                steps += 1
                self.current_step += 1

                # cumulative discounted reward
                g_n += (self.gamma ** n) * reward
                state = next_state

                if done:
                    break

            # End N-step Q-learning

            actions = self.env.get_legal_moves()
            if actions:
                # Calculate Bellman Target for action u (first action)
                attention = self.memory.get_attention(np.expand_dims(next_state,axis=0), actions)
                bellman_target = g_n + (self.alpha ** n) * np.min(attention)
            else:
                # if no actions query returns 0
                bellman_target = g_n

            self.memory.insert(np.expand_dims(state, axis=0), first_action, bellman_target)

            # Add to replay memory
            self.memory.replay_buffer.enqueue((first_state, first_action, bellman_target, state))

            # Tabular Update
            self.memory.tabular_update(first_action, g_n)

            self.memory.learn()

            if done:
                break

    def select_action(self, state):
        actions = self.env.get_legal_moves()
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.current_step / self.eps_decay)
        if random.random() > eps:
            indices = np.argmax(self.memory.get_attention(np.expand_dims(state, axis=0), actions))
            return random.choice([a for i, a in enumerate(actions) if i in indices])
        else:
            return random.choice(actions)


if __name__ == '__main__':
    pass
