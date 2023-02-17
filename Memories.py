import random


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
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
