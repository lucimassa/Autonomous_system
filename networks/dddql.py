import random
import gym
import numpy as np
from collections import deque

import tensorflow.python.keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import time

EPISODES = 50


class DDDQN(tensorflow.keras.Model):
    def __init__(self, action_space_size):
        super(DDDQN, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(action_space_size, activation=None)

    def call(self, inputs, training=None, mask=None):
        x = self.d1(inputs)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        return a


class exp_replay():
    def __init__(self, observation_space, buffer_size=1000000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, *(observation_space.shape)), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size), dtype=np.int32)
        self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, *(observation_space.shape)), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
        self.pointer = 0

    def add_exp(self, state, action, reward, next_state, done):
        idx = self.pointer % self.buffer_size
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size= 64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones


class DDQNAgent:
    def __init__(self, state_size, action_size, observation_space, replace=100):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 32
        self.memory = exp_replay(observation_space)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.replace = replace
        self.trainstep = 0
        self.learning_rate = 0.001
        self._build_model(action_size)

    def _build_model(self, action_size):
        # Neural Net for Deep-Q learning Model
        self.q_net = DDDQN(action_size)
        self.target_net = DDDQN(action_size)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)

    def memorize(self, state, action, reward, next_state, done):
          self.memory.add_exp(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.q_net.advantage(np.array([state]))
        return np.argmax(act_values)  # returns action

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        return self.epsilon

    def train(self):
        if self.memory.pointer < self.batch_size:
            return

        if self.trainstep % self.replace == 0:
            self.update_target()
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
        target = self.q_net.predict(states, verbose=0)
        next_state_val = self.target_net.predict(next_states, verbose=0)
        max_action = np.argmax(self.q_net.predict(next_states, verbose=0), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)  # optional
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones
        self.q_net.fit(states, q_target, epochs=1, verbose=2)
        self.update_epsilon()
        self.trainstep += 1

    def load(self, name: str):
        self.q_net.load_weights(name + "_q_net.h5")
        self.target_net.load_weights(name + "_target_net.h5")

    def save(self, name):
        self.q_net.save_weights(name + "_q_net.h5")
        self.target_net.save_weights(name + "_target_net.h5")


def test_dddql():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agentoo7 = DDQNAgent(state_size, action_size, env.observation_space)
    EPISODES = 50
    for s in range(EPISODES):
        print(f"EPISODE: {s}")
        done = False
        state, _ = env.reset()
        episode_reward = 0
        while not done:
            # env.render()
            action = agentoo7.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done else -10
            agentoo7.memorize(state, action, reward, next_state, done)
            agentoo7.train()
            state = next_state
            episode_reward += reward

            if done:
                print(f"total reward after {s} episode is {episode_reward} and epsilon is {agentoo7.epsilon}")

    input("Press Enter to continue...")
    env = gym.make('CartPole-v1', render_mode="human")
    total_reward = 0
    for s in range(5):
        done = False
        state, _ = env.reset()
        episode_reward = 0
        while not done:
            env.render()
            action = agentoo7.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done else -10
            # agentoo7.memorize(state, action, reward, next_state, done)
            # agentoo7.train()
            state = next_state
            episode_reward += reward
            total_reward += reward

            if done:
                print(f"total reward after {s} episode is {episode_reward} and epsilon is {agentoo7.epsilon}")

    print(total_reward)

