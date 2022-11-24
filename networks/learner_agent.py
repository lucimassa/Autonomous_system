
import numpy as np

import tensorflow as tf
from typing import List
from utils.replay_buffer import ReplayBuffer
from networks.lstm_based_dddql import LSTMBasedNet
from tensorflow.python.keras.models import load_model, save_model


class LearnerAgent:
    def __init__(self, state_size, action_size, replay_buffer: ReplayBuffer):
        self.state_size = state_size
        self.action_size = action_size
        self.seq_length = 64
        self.replay_buffer = replay_buffer
        self.gamma = 0.95  # discount rate
        self.replace = 20
        self.trainstep = 0
        self.learning_rate = 0.001
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

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def train(self, times):

        for t in range(times):
            if self.trainstep % self.replace == 0:
                self.update_target()

            self.save_net_states()

            # first run the RNN to update the state of the net (without training the weights)
            self.reset_net_states()
            batch_act, batch_train, episode_key = self.replay_buffer.sample_exp()
            if batch_act is None or batch_train is None:
                print("Not enouth data to train.")
                return
            states, actions, rewards, next_states, dones = batch_act
            states = tf.expand_dims(states, axis=0)
            next_states = tf.expand_dims(next_states, axis=0)
            _ = self.q_net.predict(states, verbose=0)
            _ = self.target_net.predict(next_states, verbose=0)

            states, actions, rewards, next_states, dones = batch_train
            q_net_states = self.q_net.get_lstm_states()
            target_net_states = self.target_net.get_lstm_states()

            states = tf.expand_dims(states, axis=0)             # add batch size as 1
            next_states = tf.expand_dims(next_states, axis=0)
            target = self.q_net.predict(states, verbose=0)
            self.q_net.set_lstm_states(q_net_states)
            next_state_val = self.target_net.predict(next_states, verbose=0)
            max_action = np.argmax(self.q_net.predict(next_states, verbose=0), axis=-1)
            self.q_net.set_lstm_states(q_net_states)
            batch_index = np.arange(states.shape[0], dtype=np.int32)
            seq_index = np.arange(states.shape[1], dtype=np.int32)
            q_target = np.copy(target)  # optional
            q_target[0, seq_index, actions] = rewards + self.gamma * next_state_val[batch_index, seq_index, max_action] * np.logical_not(dones)
            # q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * np.logical_not(dones)

            value_prev, adv_prev = None, None
            if self.debug:
                print(f"expected: {q_target}")
                print(f"predicted: {self.q_net.predict(states, verbose=0)}")
                print(f"next_values_qnet: {max_action}")
                print(f"next_valus_target: {next_state_val}")
                print(f"episode_key: {episode_key}")
                print(f"replay buffer priorities:: {[(key, self.replay_buffer.episode_mem[key][0]) for key in self.replay_buffer.episode_mem.keys()]}")
                self.q_net.set_lstm_states(q_net_states)
                value_prev, adv_prev = self.q_net.value_advantage(states)
                self.q_net.set_lstm_states(q_net_states)
            old_q_net_states = q_net_states
            history = self.q_net.fit(states, q_target, epochs=1, verbose=0)
            q_net_states = self.q_net.get_lstm_states()
            loss = history.history["loss"]
            assert len(loss) == 1
            self.replay_buffer.update_loss(episode_key, loss[0])
            if self.debug:
                print(f"loss: {loss}")
                self.q_net.set_lstm_states(old_q_net_states)
                value, adv = self.q_net.value_advantage(states)
                self.q_net.set_lstm_states(old_q_net_states)
                print(f"value: {value_prev} to {value}")
                print(f"adv: {adv_prev} to {adv}")
                print(f"predicted_after: {self.q_net.predict(states, verbose=0)}\n\n")
                self.q_net.set_lstm_states(old_q_net_states)
            self.history["loss"].append(loss)
            # if np.all(dones):
            #     self.q_net.fit(states, q_target, epochs=10, verbose=2)
            self.trainstep += 1

            self.restore_net_states()

    # def see_advancements(self):
    #     batch1 = self.memory.get_info([0], 0)
    #     batch2 = self.memory.get_info([3], 2)
    #     batch3 = self.memory.get_info([5], 4)
    #     for batch, batch_name in zip([batch1, batch2, batch3], ["batch1", "batch2", "batch3"]):
    #         print(batch_name)
    #         states, actions, rewards, next_states, dones = batch
    #         states = tf.expand_dims(states, axis=0)
    #         next_states = tf.expand_dims(next_states, axis=0)
    #         val = self.q_net.predict(states, verbose=0)
    #         val_next = self.q_net.predict(next_states, verbose=0)
    #         val_target = self.target_net.predict(states, verbose=0)
    #         print(f"actions: {actions},\nval: {val},\nval_next: {val_next},\nval_target: {val_target}")

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