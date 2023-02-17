import random

from Memories import DifferentiableNeuralDictionary, ReplayBuffer
from tqdm import tqdm


class NeuralEpisodicControl:
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.q_network = kwargs['q_network']
        self.replay_buffer = ReplayBuffer(50000)
        self.memory = {}

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
            for n in range(self.n_steps):

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

            # Calculate Bellman Target for action u (first action)
            bellman_target = g_n + (self.alpha ** n) * min(self.memory.query(next_state, action))

            # Tabular Update

            if done:
                break

    def learn(self):
        pass

    def select_action(self):
        pass


if __name__ == '__main__':
    pass
