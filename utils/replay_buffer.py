
import numpy as np
import random
from typing import List



class ReplayBuffer:
    def __init__(self, observation_space, buffer_size=512, batch_size=32):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.episode_mem = {}
        self.observation_space = observation_space
        self.pointer = 0

    # def init_episode(self, episode_num: int):
    #     state_mem = np.zeros((self.buffer_size, *(self.observation_space.shape)), dtype=np.float32)
    #     action_mem = np.zeros((self.buffer_size), dtype=np.int32)
    #     reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
    #     next_state_mem = np.zeros((self.buffer_size, *(self.observation_space.shape)), dtype=np.float32)
    #     done_mem = np.zeros((self.buffer_size), dtype=np.bool)
    #     pointer = 0
    #     self.episode_mem[episode_num] = (state_mem, action_mem, reward_mem, next_state_mem, done_mem, pointer)

    # def init_episode(self, episode_num: int):
    #     state_mem = []
    #     action_mem = []
    #     reward_mem = []
    #     next_state_mem = []
    #     done_mem = []
    #     pointer = 0
    #     self.episode_mem[episode_num] = (state_mem, action_mem, reward_mem, next_state_mem, done_mem, pointer)

    def add_exp(self, state: List, action: List, reward: List, next_state: List, done: List):
        self.episode_mem[self.pointer] = (len(state),
                                          np.array(state),
                                          np.array(action),
                                          np.array(reward),
                                          np.array(next_state),
                                          np.array(done))
        self.pointer = (self.pointer + 1) % self.buffer_size

    def sample_exp(self):
        loss_key_list = [(self.episode_mem[key][0], key) for key in list(self.episode_mem.keys())]
        if len(loss_key_list) == 0:
            return []
        loss_list, key_list = zip(*loss_key_list)
        probabilities = np.array(loss_list) / np.sum(loss_list)
        probabilities[-1] = 1 - np.sum(probabilities[:-1])
        # assert np.sum(probabilities) == 1
        key = np.random.choice(key_list, p=probabilities, size=min(len(key_list), self.batch_size), replace=False)
        # episode_num = 0
        out = []
        for k in key:
            loss, state, action, reward, next_state, done = self.episode_mem[k]

            middle = round(len(state) / 2)
            act_batch = state[:middle], action[:middle], reward[:middle], next_state[:middle], done[:middle]
            train_batch = state[middle:], action[middle:], reward[middle:], next_state[middle:], done[middle:]
            out.append((act_batch, train_batch, k))
        return out

    def update_loss(self, episode_key, loss):
        _, state, action, reward, next_state, done = self.episode_mem[episode_key]
        self.episode_mem[episode_key] = loss, state, action, reward, next_state, done

    def is_filled(self):
        return len(self.episode_mem) == self.buffer_size