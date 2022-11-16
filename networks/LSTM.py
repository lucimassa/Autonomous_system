import random
import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import tensorflow.python.keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.python.keras.layers import InputLayer, LSTM, Input
import tensorflow as tf
import time
import random
from typing import List


# class exp_replay():
#     def __init__(self, observation_space, buffer_size=1000000):
#         self.buffer_size = buffer_size
#         self.state_mem = np.zeros((self.buffer_size, *(observation_space.shape)), dtype=np.float32)
#         self.action_mem = np.zeros((self.buffer_size), dtype=np.int32)
#         self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
#         self.next_state_mem = np.zeros((self.buffer_size, *(observation_space.shape)), dtype=np.float32)
#         self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
#         self.pointer = 0
#
#     def add_exp(self, state, action, reward, next_state, done):
#         idx = self.pointer % self.buffer_size
#         self.state_mem[idx] = state
#         self.action_mem[idx] = action
#         self.reward_mem[idx] = reward
#         self.next_state_mem[idx] = next_state
#         self.done_mem[idx] = 1 - int(done)
#         self.pointer += 1
#
#     def sample_exp(self, batch_size= 64):
#         max_mem = min(self.pointer, self.buffer_size)
#         batch = np.random.choice(max_mem, batch_size, replace=False)
#         states = self.state_mem[batch]
#         actions = self.action_mem[batch]
#         rewards = self.reward_mem[batch]
#         next_states = self.next_state_mem[batch]
#         dones = self.done_mem[batch]
#         return states, actions, rewards, next_states, dones


class exp_replay():
    def __init__(self, observation_space, buffer_size=10):
        self.buffer_size = buffer_size
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

    def init_episode(self, episode_num: int):
        state_mem = []
        action_mem = []
        reward_mem = []
        next_state_mem = []
        done_mem = []
        pointer = 0
        self.episode_mem[episode_num] = (state_mem, action_mem, reward_mem, next_state_mem, done_mem, pointer)

    def add_exp(self, state: List, action: List, reward: List, next_state: List, done: List):
        self.episode_mem[self.pointer] = (state, action, reward, next_state, done)

        episode_num = episode_num % self.buffer_size
        # reset next episode
        self.init_episode((episode_num + 1) % self.buffer_size)

        state_mem, action_mem, reward_mem, next_state_mem, done_mem, pointer = self.episode_mem[episode_num]
        # idx = pointer % self.buffer_size
        state_mem.append(state)
        action_mem.append(action)
        reward_mem.append(reward)
        next_state_mem.append(next_state)
        done_mem.append(done)
        pointer += 1
        self.episode_mem[episode_num] = (state_mem, action_mem, reward_mem, next_state_mem, done_mem, pointer)

    # def sample_exp(self, batch_size=64):
    #     max_mem = min(self.pointer, self.buffer_size)
    #     batch = np.random.choice(max_mem, batch_size, replace=False)
    #     states = self.state_mem[batch]
    #     actions = self.action_mem[batch]
    #     rewards = self.reward_mem[batch]
    #     next_states = self.next_state_mem[batch]
    #     dones = self.done_mem[batch]
    #     return states, actions, rewards, next_states, dones

    def sample_exp(self, batch_size=64):
        min_seq_len = 10
        available_episod_num_list = [key for key in list(self.episode_mem.keys()) if self.episode_mem[key][-1] >= min_seq_len]
        if len(available_episod_num_list) == 0:
            return None, None
        episode_num = random.choice(available_episod_num_list)
        # episode_num = 0
        _, _, _, _, _, pointer = self.episode_mem[episode_num]
        if pointer < batch_size:
            batch_size = pointer
        max_mem = pointer - batch_size
        start = random.randint(0, max_mem)
        end = start + batch_size
        middle = start + round(batch_size / 2)
        batch_act = np.arange(start, middle)
        batch_traint = np.arange(middle, end)

        return self.get_info(batch_act, episode_num), self.get_info(batch_traint, episode_num)

    def get_info(self, batch, episode_num: int):
        state_mem, action_mem, reward_mem, next_state_mem, done_mem, pointer = self.episode_mem[episode_num]
        states = np.array(state_mem)[batch]
        actions = np.array(action_mem)[batch]
        rewards = np.array(reward_mem)[batch]
        next_states = np.array(next_state_mem)[batch]
        dones = np.array(done_mem)[batch]
        return states, actions, rewards, next_states, dones



class LSTMBasedNet(tensorflow.keras.Model):

    LSTM_1_UNITS = 128
    LSTM_2_UNITS = 128

    def __init__(self, action_space_size, batch_size=1, use_lstm_states=True):
        super(LSTMBasedNet, self).__init__()
        self.lstm1 = LSTM(self.LSTM_1_UNITS, activation="tanh", return_sequences=True, return_state=True, name="first_LSTM")
        self.lstm2 = LSTM(self.LSTM_2_UNITS, activation="tanh", return_sequences=True, return_state=True, name="second_LSTM")
        self.v = LSTM(1, activation=None, return_sequences=True, return_state=True, name="value_LSTM")
        self.a = LSTM(action_space_size, activation=None, return_sequences=True, return_state=True, name="advantage_LSTM")
        # self.v = Dense(1, activation=None, name="value_dense")
        # self.a = Dense(action_space_size, activation=None, name="advantage_dense")
        self.action_space_size = action_space_size
        self.batch_size = batch_size
        self.use_lstm_states = use_lstm_states
        self.state_1 = None
        self.state_2 = None
        self.state_v = None
        self.state_a = None
        self.reset_lstm_states()
        self.d1_old = tf.keras.layers.Dense(128, activation='relu')
        self.d2_old = tf.keras.layers.Dense(128, activation='relu')
        self.v_old = tf.keras.layers.Dense(1, activation=None)
        self.a_old = tf.keras.layers.Dense(action_space_size, activation=None)

    def call(self, inputs, training=None, mask=None):
        if not self.use_lstm_states:
            self.reset_lstm_states()
        x, state_h_1, state_c_1 = self.lstm1(inputs, initial_state=self.state_1)
        x, state_h_2, state_c_2 = self.lstm2(x, initial_state=self.state_2)
        v, state_h_v, state_c_v = self.v(x, initial_state=self.state_v)
        a, state_h_a, state_c_a = self.a(x, initial_state=self.state_a)
        # v = self.v(x)
        # a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=-1, keepdims=True))
        # self.lstm_states.write(0,[state_h_1, state_h_2])
        self.state_1 = [state_h_1, state_c_1]
        self.state_2 = [state_h_2, state_c_2]
        self.state_v = [state_h_v, state_c_v]
        self.state_a = [state_h_a, state_c_a]
        return Q

    # def call(self, inputs, training=None, mask=None):
    #     x = self.d1_old(inputs)
    #     x = self.d2_old(x)
    #     v = self.v_old(x)
    #     a = self.a_old(x)
    #     Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
    #     return Q

    def advantage(self, state):
        if not self.use_lstm_states:
            self.reset_lstm_states()
        x, state_h_1, state_c_1 = self.lstm1(state, initial_state=self.state_1)
        x, state_h_2, state_c_2 = self.lstm2(x, initial_state=self.state_2)
        a, state_h_a, state_c_a = self.a(x, initial_state=self.state_a)
        # a = self.a(x)
        self.state_1 = [state_h_1, state_c_1]
        self.state_2 = [state_h_2, state_c_2]
        self.state_a = [state_h_a, state_c_a]
        return a

    def value_advantage(self, state):
        if not self.use_lstm_states:
            self.reset_lstm_states()
        x, state_h_1, state_c_1 = self.lstm1(state, initial_state=self.state_1)
        x, state_h_2, state_c_2 = self.lstm2(x, initial_state=self.state_2)
        v, state_h_v, state_c_v = self.v(x, initial_state=self.state_v)
        a, state_h_a, state_c_a = self.a(x, initial_state=self.state_a)
        # a = self.a(x)
        self.state_1 = [state_h_1, state_c_1]
        self.state_2 = [state_h_2, state_c_2]
        self.state_a = [state_h_a, state_c_a]
        return v, a

    # def advantage(self, state):
    #     x = self.d1_old(state)
    #     x = self.d2_old(x)
    #     a = self.a_old(x)
    #     return a

    def reset_lstm_states(self):
        self.state_1 = [tf.zeros((self.batch_size, self.LSTM_1_UNITS)), tf.zeros((self.batch_size, self.LSTM_1_UNITS))]
        self.state_2 = [tf.zeros((self.batch_size, self.LSTM_2_UNITS)), tf.zeros((self.batch_size, self.LSTM_2_UNITS))]
        self.state_v = [tf.zeros((self.batch_size, 1)), tf.zeros((self.batch_size, 1))]
        self.state_a = [tf.zeros((self.batch_size, self.action_space_size)), tf.zeros((self.batch_size, self.action_space_size))]

    def get_lstm_states(self):
        return self.state_1, self.state_2, self.state_v, self.state_a

    def set_lstm_states(self, states):
        self.state_1, self.state_2, self.state_v, self.state_a = states


class LSTMAgent:
    def __init__(self, state_size, action_size, observation_space, replace=50):
        self.state_size = state_size
        self.action_size = action_size
        self.seq_length = 64
        self.memory = exp_replay(observation_space)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.replace = replace
        self.trainstep = 0
        self.learning_rate = 0.001 #0.001
        self._build_model(action_size)
        self.net_chckpoint = None
        self.history = {}
        self.history["loss"] = []
        self.debug = True

    def _build_model(self, action_size):
        # Neural Net for Deep-Q learning Model
        self.q_net = LSTMBasedNet(action_size)
        self.target_net = LSTMBasedNet(action_size)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # note: try using Huber loss instad
        self.q_net.compile(loss='mse', optimizer=opt, run_eagerly=True)
        self.target_net.compile(loss='mse', optimizer=opt, run_eagerly=True)
        self.q_net.predict(tf.zeros([1, 1, self.state_size]), verbose=0)
        self.target_net.predict(tf.zeros([1, 1, self.state_size]), verbose=0)

    def memorize(self, state, action, reward, next_state, done, episode_num):
        self.memory.add_exp(state, action, reward, next_state, done, episode_num)

    def act(self, state, test=False):
        if np.random.rand() <= self.epsilon and not test:
            return random.randrange(self.action_size)
        # state = tf.expand_dims(state, axis=0)
        act_values = self.q_net.advantage(np.array([state]))
        print(np.argmax(act_values))
        return np.argmax(act_values)  # returns action

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        return self.epsilon

    def train(self):

        if self.trainstep % self.replace == 0:
            self.update_target()

        self.save_net_states()

        # first run the RNN to update the state of the net (without training the weights)
        self.reset_net_states()
        batch_act, batch_train = self.memory.sample_exp(self.seq_length)
        if batch_act is None or batch_train is None:
            return
        states, actions, rewards, next_states, dones = batch_act
        states = tf.expand_dims(states, axis=0)
        next_states = tf.expand_dims(next_states, axis=0)
        _ = self.q_net.predict(states, verbose=0)
        _ = self.target_net.predict(next_states, verbose=0)

        states, actions, rewards, next_states, dones = batch_train

        states = tf.expand_dims(states, axis=0)             # add batch size as 1
        next_states = tf.expand_dims(next_states, axis=0)
        target = self.q_net.predict(states, verbose=0)
        next_state_val = self.target_net.predict(next_states, verbose=0)
        max_action = np.argmax(self.q_net.predict(next_states, verbose=0), axis=-1)
        batch_index = np.arange(states.shape[0], dtype=np.int32)
        seq_index = np.arange(states.shape[1], dtype=np.int32)
        q_target = np.copy(target)  # optional
        q_target[0, batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, seq_index, max_action] * np.logical_not(dones)
        # q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * np.logical_not(dones)

        value_prev, adv_prev = None, None
        if self.debug:
            lstm_states = self.q_net.get_lstm_states()
            print(f"expected: {q_target}")
            print(f"predicted: {self.q_net.predict(states, verbose=0)}")
            self.q_net.set_lstm_states(lstm_states)
            value_prev, adv_prev = self.q_net.value_advantage(states)
            self.q_net.set_lstm_states(lstm_states)
        history = self.q_net.fit(states, q_target, epochs=1, verbose=0)
        loss = history.history["loss"]
        if self.debug:
            lstm_states = self.q_net.get_lstm_states()
            print(f"loss: {loss}")
            value, adv = self.q_net.value_advantage(states)
            self.q_net.set_lstm_states(lstm_states)
            print(f"value: {value_prev} to {value}")
            print(f"adv: {adv_prev} to {adv}")
            print(f"predicted_after: {self.q_net.predict(states, verbose=0)}\n\n")
            self.q_net.set_lstm_states(lstm_states)
        self.history["loss"].append(loss)
        # if np.all(dones):
        #     self.q_net.fit(states, q_target, epochs=10, verbose=2)
        self.update_epsilon()
        self.trainstep += 1

        self.restore_net_states()

    def see_advancements(self):
        batch1 = self.memory.get_info([0], 0)
        batch2 = self.memory.get_info([3], 2)
        batch3 = self.memory.get_info([5], 4)
        for batch, batch_name in zip([batch1, batch2, batch3], ["batch1", "batch2", "batch3"]):
            print(batch_name)
            states, actions, rewards, next_states, dones = batch
            states = tf.expand_dims(states, axis=0)
            next_states = tf.expand_dims(next_states, axis=0)
            val = self.q_net.predict(states, verbose=0)
            val_next = self.q_net.predict(next_states, verbose=0)
            val_target = self.target_net.predict(states, verbose=0)
            print(f"actions: {actions},\nval: {val},\nval_next: {val_next},\nval_target: {val_target}")




    # def train(self):
    #     if self.memory.pointer < self.seq_length:
    #         return
    #
    #     if self.trainstep % self.replace == 0:
    #         self.update_target()
    #
    #     self.save_net_states()
    #
    #     # first run the RNN to update the state of the net (without training the weights)
    #     self.reset_net_states()
    #     batch_act, batch_train = self.memory.sample_exp(self.seq_length)
    #     states, actions, rewards, next_states, dones = batch_act
    #     # states = tf.expand_dims(states, axis=0)
    #     _ = self.q_net.predict(states)
    #
    #     # states = tf.expand_dims(states, axis=0)             # add batch size as 1
    #     # next_states = tf.expand_dims(next_states, axis=0)
    #     target = self.q_net.predict(states)
    #     next_state_val = np.squeeze(self.target_net.predict(next_states))
    #     max_action = np.argmax(tf.squeeze(self.q_net.predict(next_states)), axis=1)
    #     batch_index = np.arange(states.shape[-2], dtype=np.int32)
    #     q_target = np.copy(target)  # optional
    #     # q_target[0, batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones
    #     q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones
    #     self.q_net.fit(states, q_target, epochs=1, verbose=0)
    #
    #     self.restore_net_states()

    def load(self, name: str):
        self.q_net.load_weights(name + "_q_net.h5")
        self.target_net.load_weights(name + "_target_net.h5")

    def save(self, name):
        self.q_net.save_weights(name + "_q_net.h5")
        self.target_net.save_weights(name + "_target_net.h5")

    def reset_net_states(self):
        self.q_net.reset_lstm_states()
        self.target_net.reset_lstm_states()

    def save_net_states(self):
        self.net_chckpoint = self.q_net.get_lstm_states(), self.target_net.get_lstm_states()

    def restore_net_states(self):
        q_net_state, target_net_state = self.net_chckpoint
        self.q_net.set_lstm_states(q_net_state)
        self.target_net.set_lstm_states(target_net_state)


def lstm_tutorial():
    from tensorflow.python.keras import Sequential
    anction_space_dim = 50
    model = Sequential()
    model.add(InputLayer(input_shape=(1, anction_space_dim), batch_size=1))
    model.add(LSTM(128, input_shape=(32, anction_space_dim), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='sgd')

    data = np.random.random((1, 1, anction_space_dim)).astype(np.float32)
    pred = model.predict(data)
    print(pred)


def test_lstm():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agentoo7 = LSTMAgent(state_size, action_size, env.observation_space)
    try:
        agentoo7.load("agent_64")
    except:
        print("unable to load agent")
    EPISODES = 20
    for s in range(EPISODES):
        print(f"EPISODE: {s}")
        done = False
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        agentoo7.reset_net_states()
        while not done:
            # env.render()
            action = agentoo7.act(state)
            next_state, reward, done, _, _ = env.step(action)

            state = np.reshape(state, [state_size])
            agentoo7.memorize(state, action, reward, next_state, done, s)
            next_state = np.reshape(next_state, [1, state_size])
            for i in range(10):
                agentoo7.train()
            # if s > 4:
            #     agentoo7.see_advancements()
            state = next_state
            episode_reward += reward

            if done:
                print(f"total reward after {s} episode is {episode_reward} and epsilon is {agentoo7.epsilon}")
        agentoo7.save("agent_64")


    losses = agentoo7.history["loss"]
    plt.plot(np.arange(0, len(losses), 1), losses)
    plt.show()
    input("Press Enter to continue...")
    env = gym.make('CartPole-v1', render_mode="human")
    total_reward = 0
    for s in range(5):
        done = False
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        agentoo7.reset_net_states()
        while not done:
            env.render()
            action = agentoo7.act(state, test=True)
            print(f"action: {action}")
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            episode_reward += reward
            total_reward += reward

            if done:
                print(f"total reward after {s} episode is {episode_reward} and epsilon is {agentoo7.epsilon}")

    print(total_reward)

