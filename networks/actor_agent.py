import copy
import numpy as np
import random
import tensorflow as tf

from utils.replay_buffer import ReplayBuffer
from networks.lstm_based_dddql import LSTMBasedNet


class ActorAgent:
    SEQ_LEN = 80
    MIN_LEN = 30

    def __init__(self, env, replay_buffer: ReplayBuffer = None, test=False):
        copy.deepcopy(env)
        self.env = env
        self.env.reset()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.replay_buffer = replay_buffer
        self._build_model()

        self.epsilon = 0 if test else 1  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.is_initialized = False

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        self.agent = LSTMBasedNet(action_space_size=self.action_size, batch_size=1)
        opt = tf.keras.optimizers.Adam()

        # should not be needed to specify the loss function,
        #   as this net will never train but just copy another trained network
        loss_func = tf.keras.losses.Huber()
        self.agent.compile(loss=loss_func, optimizer=opt, run_eagerly=True)
        self.agent.predict(tf.zeros([1, 1, self.state_size]), verbose=0)

    def _act(self, state, test=False):
        if np.random.rand() <= self.epsilon and not test and False:
            return random.randrange(self.action_size)
        # state = tf.expand_dims(state, axis=0)
        act_values = self.agent.advantage(np.array([state]))
        # print(np.argmax(act_values))
        return np.argmax(act_values)  # returns action

    def act(self, episodes=500, test=False):
        for ep in range(episodes):
            done = False
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            episode_reward = 0
            # self.agent.predict(np.array([state]))
            self.agent.reset_lstm_states()
            state_list = []
            action_list = []
            reward_list = []
            next_state_list = []
            done_list = []
            saved_something = False
            total_reward = 0
            while not done:
                # env.render()
                action = self._act(state, test=test)
                next_state, reward, done, _, _ = self.env.step(action)
                if test:
                    self.env.render()
                total_reward += reward

                state = np.reshape(state, [self.state_size])
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                next_state_list.append(next_state)
                done_list.append(done)
                next_state = np.reshape(next_state, [1, self.state_size])

                mem_len = len(state_list)
                if self.replay_buffer is not None:
                    if (done and mem_len >= self.MIN_LEN) or mem_len >= self.SEQ_LEN:
                    # if mem_len >= self.MIN_LEN:
                        self.replay_buffer.add_exp(state_list, action_list, reward_list, next_state_list, done_list)
                        middle = round(mem_len / 2)
                        state_list = state_list[-middle:]
                        action_list = action_list[-middle:]
                        reward_list = reward_list[-middle:]
                        next_state_list = next_state_list[-middle:]
                        done_list = done_list[-middle:]
                        saved_something = True

                state = next_state
                episode_reward += reward
            if saved_something or test:
                print(f"total_trial: {ep}")
                return total_reward

    def update(self, trained_agent: LSTMBasedNet):
        self.agent.set_weights(trained_agent.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        return self.epsilon



