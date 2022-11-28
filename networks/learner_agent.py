
import numpy as np

import tensorflow as tf
from typing import List
from utils.replay_buffer import ReplayBuffer
from networks.lstm_based_dddql import LSTMBasedNet


class LearnerAgent:
    def __init__(self, state_size, action_size, replay_buffer: ReplayBuffer):
        self.state_size = state_size
        self.action_size = action_size
        self.seq_length = 64
        self.replay_buffer = replay_buffer
        self.gamma = 0.95  # discount rate
        self.replace = 50
        self.trainstep = 0
        self.learning_rate = 0.001
        self._build_model(action_size)
        self.net_chckpoint = None
        self.history = {}
        self.history["loss"] = []
        self.debug = True
        self.n_step = 5

    def _build_model(self, action_size):
        # Neural Net for Deep-Q learning Model
        self.q_net = LSTMBasedNet(action_size)
        self.target_net = LSTMBasedNet(action_size)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss_func = tf.keras.losses.Huber()
        # note: try using Huber loss instad
        self.q_net.compile(loss=loss_func, optimizer=opt, run_eagerly=True)
        self.target_net.compile(loss=loss_func, optimizer=opt, run_eagerly=True)
        self.q_net.predict(tf.zeros([1, 1, self.state_size]), verbose=0)
        self.target_net.predict(tf.zeros([1, 1, self.state_size]), verbose=0)
        self.q_net.summary()

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def train(self, times):

        for t in range(times):
            if self.trainstep % self.replace == 0:
                self.update_target()

            self.save_net_states()

            # first run the RNN to update the state of the net (without training the weights)
            self.reset_net_states()
            batch_act, batch_train, episode_key = self.replay_buffer.sample_exp(self.gamma, self.n_step)
            if batch_act is None or batch_train is None:
                print("Not enouth data to train.")
                return
            states, actions, rewards, next_states, dones = batch_act
            states = tf.expand_dims(states, axis=0)
            next_states = tf.expand_dims(next_states, axis=0)
            _ = self.q_net.predict(states, verbose=0)
            _ = self.target_net.predict(next_states, verbose=0)

            states, actions, rewards, next_states, dones = batch_train

            states = tf.expand_dims(states, axis=0)             # add batch size as 1
            next_states = tf.expand_dims(next_states, axis=0)
            target = self.predict_q_net(states)
            next_state_val = self.predict_target_net(next_states)
            max_action = np.argmax(self.predict_q_net(next_states), axis=-1)
            batch_index = np.arange(states.shape[0], dtype=np.int32)
            seq_index = np.arange(states.shape[1], dtype=np.int32)
            q_target = np.copy(target)  # optional

            q_target[0, seq_index, actions] = rewards + self.gamma * next_state_val[batch_index, seq_index, max_action] * np.logical_not(dones)
            # q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * np.logical_not(dones)

            value_prev, adv_prev = None, None
            if self.debug:
                print(f"expected: {q_target}")
                print(f"predicted: {self.predict_q_net(states)}")
                print(f"next_values_qnet: {max_action}")
                print(f"next_valus_target: {next_state_val}")
                print(f"episode_key: {episode_key}")
                print(f"replay buffer priorities:: {[(key, self.replay_buffer.episode_mem[key][0]) for key in self.replay_buffer.episode_mem.keys()]}")
                q_net_states = self.q_net.get_lstm_states()
                value_prev, adv_prev = self.q_net.value_advantage(states)
                self.q_net.set_lstm_states(q_net_states)
            q_net_states = self.q_net.get_lstm_states()
            history = self.q_net.fit(states, q_target, epochs=1, verbose=0)
            self.q_net.set_lstm_states(q_net_states)
            loss = history.history["loss"]
            assert len(loss) == 1
            self.replay_buffer.update_loss(episode_key, loss[0])
            if self.debug:
                print(f"loss: {loss}")
                value, adv = self.q_net.value_advantage(states)
                self.q_net.set_lstm_states(q_net_states)
                print(f"value: {value_prev} to {value}")
                print(f"adv: {adv_prev} to {adv}")
                print(f"predicted_after: {self.predict_q_net(states)}\n\n")
            self.history["loss"].append(loss)
            # if np.all(dones):
            #     self.q_net.fit(states, q_target, epochs=10, verbose=2)
            self.trainstep += 1

            self.restore_net_states()

    def predict_q_net(self, states, change_rnn_states=False):
        return self._predict(self.target_net, states, change_rnn_states)

    def predict_target_net(self, states, change_rnn_states=False):
        return self._predict(self.q_net, states, change_rnn_states)

    def _predict(self, network: LSTMBasedNet, states, change_rnn_states=False):
        if change_rnn_states:
            return network.predict(states, verbose=0)
        else:
            q_net_states = network.get_lstm_states()
            out = network.predict(states, verbose=0)
            network.set_lstm_states(q_net_states)

        return out

    def load(self, name: str):
        weights = np.load(name + ".npy", allow_pickle=True)
        self.q_net.set_weights(weights)
        self.target_net.set_weights(weights)

    def save(self, name):
        weights = np.array(self.q_net.get_weights(), dtype="object")
        np.save(name + ".npy", weights, allow_pickle=True)

    def reset_net_states(self):
        self.q_net.reset_lstm_states()
        self.target_net.reset_lstm_states()

    def save_net_states(self):
        self.net_chckpoint = self.q_net.get_lstm_states(), self.target_net.get_lstm_states()

    def restore_net_states(self):
        q_net_state, target_net_state = self.net_chckpoint
        self.q_net.set_lstm_states(q_net_state)
        self.target_net.set_lstm_states(target_net_state)