
import numpy as np
import time

import tensorflow as tf
import pickle
from typing import List
from utils.replay_buffer import ReplayBuffer
from networks.lstm_based_dddql import LSTMBasedNet
from pynverse import inversefunc

from utils.preprocessing import get_state_size
from utils.const import OPTIMIZER, DICOUNT_RATE, SAVE_AGENT_NAME


class LearnerAgent:
    def __init__(self, state_size, action_size, n_step, seq_len, replay_buffer: ReplayBuffer):
        self.state_size = get_state_size(state_size)
        self.action_size = action_size
        self.seq_len = seq_len
        self.replay_buffer = replay_buffer
        self.gamma = DICOUNT_RATE  # discount rate
        self.replace = 50
        self.trainstep = 0
        self.momentum = 0.9
        self._build_model(action_size)
        self.net_chckpoint = None
        self.history = {}
        self.history["loss"] = []
        self.debug = False
        self.n_step = n_step
        self.epsilon = .001

        # for debug only
        self.replaced_times = 0

    def _build_model(self, action_size):
        self.q_net = LSTMBasedNet(state_size=self.state_size, action_space_size=action_size)
        self.target_net = LSTMBasedNet(state_size=self.state_size, action_space_size=action_size)
        opt = OPTIMIZER
        loss_func = tf.keras.losses.mse
        self.q_net.compile(loss=loss_func, optimizer=opt, run_eagerly=True)
        self.target_net.compile(loss=loss_func, optimizer=opt, run_eagerly=True)
        self.q_net.predict(tf.zeros([1, 1] + self.state_size), verbose=0)
        self.target_net.predict(tf.zeros([1, 1] + self.state_size), verbose=0)
        self.q_net.summary()

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def train(self, epochs, batch_size=16):

        self.trainstep += 1
        if self.trainstep % self.replace == 0:
            self.replaced_times += 1
            # print(f"replacing: {self.replaced_times}")
            self.update_target()
        batch = []
        for batch_act, batch_train, episode_key in self.replay_buffer.sample_exp(batch_size=batch_size):

            start = time.time()
            # first run the RNN to update the state of the net (without training the weights)
            self.reset_net_states()
            # print(f"episode_key: {episode_key}")
            if batch_train is None:
                # print("Not enouth data to train.")
                return

            q_net_lstm_states_next = self.q_net.get_lstm_states()
            if batch_act is not None:
                states, actions, rewards, next_states, dones = batch_act
                states = tf.expand_dims(states, axis=0)
                next_states = tf.expand_dims(next_states, axis=0)
                _ = self.q_net.predict(next_states, verbose=0)
                q_net_lstm_states_next = self.q_net.get_lstm_states()
                self.reset_net_states()
                _ = self.q_net.predict(states, verbose=0)
                _ = self.target_net.predict(next_states, verbose=0)
                # print("non_empty_batch_act")
            else:
                print("empty_batch_act")

            states, actions, rewards, next_states, dones = batch_train
            n_step_states = states
            n_step_actions = actions
            n_step_rewards = self.get_n_step_reward(rewards)
            n_step_next_states = next_states
            n_step_dones = dones
            if not dones[-1]:
                n_step_states = n_step_states[:-(self.n_step - 1)]
                n_step_actions = n_step_actions[:-(self.n_step - 1)]
                n_step_rewards = n_step_rewards[:-(self.n_step - 1)]
                n_step_next_states = n_step_next_states[:-(self.n_step - 1)]
                n_step_dones = n_step_dones[:-(self.n_step - 1)]

            if len(n_step_states) == 0:
                # print("sequnce too short")
                continue

            n_step_states = tf.expand_dims(n_step_states, axis=0)             # add batch size as 1
            n_step_next_states = tf.expand_dims(n_step_next_states, axis=0)

            target = self.predict_q_net(n_step_states)
            next_state_val = self.predict_target_net(n_step_next_states)
            max_action = np.argmax(self.predict_q_net(n_step_next_states, lstm_states=q_net_lstm_states_next), axis=-1)
            batch_index = np.arange(n_step_states.shape[0], dtype=np.int32)
            seq_index = np.arange(n_step_states.shape[1], dtype=np.int32)
            q_target = np.copy(target)  # optional

            q_target[0, seq_index, n_step_actions] = self.h(n_step_rewards + (self.gamma ** self.n_step) * \
                                                     self.h_inverse(next_state_val[batch_index, seq_index, max_action] * \
                                                     np.logical_not(n_step_dones)))
            # q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * np.logical_not(dones)

            value_prev, adv_prev = None, None
            if self.debug:
                print(f"expected: {q_target}")
                predicted = self.predict_q_net(n_step_states)
                print(f"predicted: {predicted}")
                print(f"next_values_qnet: {self.predict_q_net(n_step_next_states, lstm_states=q_net_lstm_states_next)}")
                print(f"max_action: {max_action}")
                print(f"n_step_actions: {n_step_actions}")
                print(f"n_step_rewards: {n_step_rewards}")
                print(f"next_valus_target: {next_state_val}")
                print(f"episode_key: {episode_key}")
                print(f"replay buffer priorities:: {[(key, self.replay_buffer.episode_mem[key][0]) for key in self.replay_buffer.episode_mem.keys()]}")
                value_prev, adv_prev = self.get_val_adv(n_step_states)
                print(f"value: {value_prev}")
                print(f"adv: {adv_prev}")

            batch.append((self.q_net.get_lstm_states(), n_step_states, q_target, episode_key))

            loss = None
        start = time.time()
        self.fit_q_net_batch(batch, epochs)
        # print(f"fit_time: {time.time() - start}")

    # def h(self, x: np.ndarray):
    #     return np.sign(x) * (np.sqrt(np.absolute(x) + 1) - 1) + self.epsilon * x

    def h(self, x: np.ndarray):
        return x

    def h_inverse(self, x):
        return inversefunc(self.h, y_values=x)

    def get_n_step_reward(self, reward):
        n_step_reward = reward
        for i in range(1, self.n_step, 1):
            to_add = np.zeros(reward.shape)
            to_add[:-i] = reward[i:]
            n_step_reward = n_step_reward + (self.gamma ** i) * to_add
        return n_step_reward

    def fit_q_net_batch(self, batch, epochs):
        """
        in practice just train all elements of the batch in sequence
        """
        for ep in range(epochs):
            losses = []
            for lstm_states, n_step_states, q_target, episode_key in batch:
                self.q_net.set_lstm_states(lstm_states)
                loss = self.fit_q_net(n_step_states, q_target, epochs=1)
                if ep == epochs - 1:
                    # replace loss with better function
                    pred = self.predict_q_net(n_step_states)
                    self.replay_buffer.update_loss(episode_key, pred, q_target)
                    if loss is not None:
                        self.history["loss"].append(loss)
                losses.append(np.mean(loss))
            print(f"loss: {np.mean(losses)}")

    def fit_q_net(self, n_step_states, q_target, epochs=1, change_rnn_states=False):
        """
        fit q_net and restores the states that the LSTM had before fitting, if change_rnn_states is false
        """
        history = None
        loss = None
        for ep in range(epochs):
            if change_rnn_states:
                loss = self.q_net.train_on_batch(n_step_states, q_target)
            else:
                q_net_states = self.q_net.get_lstm_states()
                loss = self.q_net.train_on_batch(n_step_states, q_target)
                self.q_net.set_lstm_states(q_net_states)
        return loss

    def predict_q_net(self, states, lstm_states=None):
        """
        predict q_net and restores the states that the LSTM had before fitting, if change_rnn_states is false
        """
        return self._predict(self.q_net, states, lstm_states)

    def get_val_adv(self, states):
        q_net_states = self.q_net.get_lstm_states()
        value_prev, adv_prev, _ = self.q_net.value_advantage(states)
        self.q_net.set_lstm_states(q_net_states)
        return value_prev, adv_prev

    def predict_target_net(self, states, lstm_states=None):
        """
        predict target_net and restores the states that the LSTM had before fitting, if change_rnn_states is false
        """
        return self._predict(self.target_net, states, lstm_states)

    def _predict(self, network: LSTMBasedNet, states, lstm_states):
        """
        predict "network" and restores the states that the LSTM had before fitting, if change_rnn_states is false
        """
        q_net_states = network.get_lstm_states()
        if lstm_states is not None:
            network.set_lstm_states(lstm_states)
        out = network.predict(states, verbose=0)
        network.set_lstm_states(q_net_states)

        return out

    def save(self, name):
        """
        saves to file the weights and information needed to restore the training (optimizer's variables)
        """
        self.q_net.save_weights(f'{name}_weights.h5')
        weight_values = self.q_net.optimizer.get_weights()
        with open(f'{name}_optimizer.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

    def load(self, name: str):
        """
        load all what is needed to restore the training
        """
        self.q_net.make_train_function()
        with open(f'{name}_optimizer.pkl', 'rb') as f:
            weight_values = pickle.load(f)

        grad_vars = self.q_net.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        self.q_net.optimizer.apply_gradients(zip(zero_grads, grad_vars))
        self.q_net.optimizer.set_weights(weight_values)

        self.q_net.load_weights(f'{name}_weights.h5')
        self.target_net.set_weights(self.q_net.get_weights())

    def reset_net_states(self):
        self.q_net.reset_lstm_states()
        self.target_net.reset_lstm_states()

    def save_net_states(self):
        self.net_chckpoint = self.q_net.get_lstm_states(), self.target_net.get_lstm_states()

    def restore_net_states(self):
        q_net_state, target_net_state = self.net_chckpoint
        self.q_net.set_lstm_states(q_net_state)
        self.target_net.set_lstm_states(target_net_state)