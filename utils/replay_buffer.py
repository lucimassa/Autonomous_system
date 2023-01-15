
import numpy as np
import random
from typing import List
from utils.const import IMG_STACK_COUNT, BUFFER_SIZE




class ReplayBuffer:
    def __init__(self, observation_space, n_step, train_batch_len, batch_size=32):
        self.buffer_size = BUFFER_SIZE
        self.batch_size = batch_size
        self.episode_mem = {}
        self.observation_space = observation_space
        self.pointer = 0
        self.n_step = n_step
        self.train_batch_len = train_batch_len
        self.mu = 0.9
        self.priority_exponent = 0.9
        self.img_stack_count = IMG_STACK_COUNT

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

    def add_exp(self, state: List, action: List, reward: List, done: List):
        # given self.act_batch_len as the size of act batch, the train batch must be at least of size n_step
        if len(action) > self.n_step:
            self.episode_mem[self.pointer] = (1,
                                              self.compress_state(np.array(state)),
                                              np.array(action),
                                              np.array(reward),
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
            loss, state, action, reward, done = self.episode_mem[k]
            state = self.extract_state(state)
            next_state = state[1:]
            state = state[:-1]
            state = self.stack_states(state)
            next_state = self.stack_states(next_state)
            state, action, reward, next_state, done = self.n_step_fix(state, action, reward, next_state, done)

            if len(state) <= self.n_step + 1:
                out.append((None, (state, action, reward, next_state, done), k))
            else:
                pivot = min(self.train_batch_len, len(state) - (self.n_step + 2))
                act_batch = state[:pivot], action[:pivot], reward[:pivot], next_state[:pivot], done[:pivot]
                train_batch = state[pivot:], action[pivot:], reward[pivot:], next_state[pivot:], done[pivot:]

                out.append((act_batch, train_batch, k))
        return out

    def n_step_fix(self, states, actions, rewards, next_states, dones):

        n_step_states = states
        n_step_actions = actions
        n_step_rewards = rewards
        # n_step_next_states = np.zeros(next_states.shape)
        n_step_next_states = np.full(next_states.shape, next_states[-1])
        n_step_next_states[:-(self.n_step - 1)] = next_states[(self.n_step - 1):]
        n_step_dones = np.full(dones.shape, dones[-1])          # propagate the last value of done over the last values
        n_step_dones[:-(self.n_step - 1)] = dones[(self.n_step - 1):]
        return n_step_states, n_step_actions, n_step_rewards, n_step_next_states, n_step_dones

    def update_loss(self, episode_key, predicted, expected):
        _, state, action, reward, done = self.episode_mem[episode_key]
        self.episode_mem[episode_key] = self.__calculate_loss(predicted, expected), state, action, reward, done

    def __calculate_loss(self, predicted, expected):
        error = np.abs(expected - predicted)
        loss = self.mu * np.max(error) + (1 - self.mu) * np.mean(error)
        return loss ** self.priority_exponent

    def is_filled(self):
        return len(self.episode_mem) == self.buffer_size

    def get_size(self):
        return len(self.episode_mem)

    def compress_state(self, state):
        min = 0
        max = 255
        range = max - min
        return np.rint(state * range + min)

    def extract_state(self, state):
        min = 0
        max = 255
        range = max - min
        return (state - min) / range if range > 0 else state - min

    def stack_states(self, states):
        stacks = np.full((self.img_stack_count, *states.shape), states[0])
        stacks[0, :] = states
        for i in range(1, self.img_stack_count):
            stacks[i, i:] = states[:-i]
        return np.concatenate(stacks, axis=-1)


