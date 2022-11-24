
import numpy as np

import tensorflow.python.keras
from tensorflow.python.keras.layers import LSTM
import tensorflow as tf


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

