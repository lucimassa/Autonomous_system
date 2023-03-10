
import numpy as np

import tensorflow as tf
# from tensorflow.python.keras.layers import LSTM, Conv1D
# from tensorflow.python.keras import Model
from typing import List


class LSTMBasedNet(tf.keras.Model):

    LSTM_1_UNITS = 128
    LSTM_2_UNITS = 128
    CONV_1_UNITS = 32
    CONV_2_UNITS = 64
    CONV_3_UNITS = 64
    # CONV_4_UNITS = 64
    CONV_5_UNITS = 64
    # DENSE_1_UNITS = 128
    # DENSE_2_UNITS = 128
    DENSE_3_UNITS = 64

    def __init__(self, state_size: List, action_space_size, batch_size=1, use_lstm_states=True):
        super(LSTMBasedNet, self).__init__()
        activation = "relu"
        # activation = tf.keras.layers.LeakyReLU(alpha=0.01)
        kr = None       # 'l2'
        self.conv1 = tf.keras.layers.Conv3D(filters=self.CONV_1_UNITS, kernel_size=[1, 8, 8],
                                            strides=(1, 4, 4), activation=activation,
                                            kernel_regularizer=kr,
                                            name="conv1")
        self.conv2 = tf.keras.layers.Conv3D(filters=self.CONV_2_UNITS, kernel_size=[1, 4, 4],
                                            strides=(1, 2, 2), activation=activation,
                                            kernel_regularizer=kr,
                                            name="conv2")
        self.conv3 = tf.keras.layers.Conv3D(filters=self.CONV_3_UNITS, kernel_size=[1, 3, 3],
                                            strides=(1, 1, 1), activation=activation,
                                            kernel_regularizer=kr,
                                            name="conv3")

        # self.conv4 = tf.keras.layers.Conv3D(filters=self.CONV_4_UNITS, kernel_size=[1, 4, 4],
        #                                     strides=(1, 2, 2), activation=activation,
        #                                     kernel_regularizer=kr,
        #                                     name="conv4")
        self.conv5 = tf.keras.layers.Conv3D(filters=self.CONV_5_UNITS, kernel_size=[1, 3, 3],
                                            strides=(1, 1, 1), activation=activation,
                                            kernel_regularizer=kr,
                                            name="conv5")
        # self.conv_tmp = tf.keras.layers.Conv3D(filters=3, kernel_size=[1, 1, 1],
        #                                        strides=(1, 1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.01),
        #                                        kernel_regularizer='l2',
        #                                        name="conv3")
        self.max_pooling = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))
        # self.dense1 = tf.keras.layers.Conv1D(filters=self.DENSE_1_UNITS, kernel_size=1,
        #                                      activation=activation,
        #                                      kernel_regularizer=kr,
        #                                      name="dense1")
        self.lstm1 = tf.keras.layers.LSTM(self.LSTM_1_UNITS, activation=activation,
                                          return_sequences=True,
                                          return_state=True, name="first_LSTM")
        self.lstm2 = tf.keras.layers.LSTM(self.LSTM_2_UNITS, activation="tanh", return_sequences=True,
                                          return_state=True, name="second_LSTM")
        # self.dense2 = tf.keras.layers.Conv1D(filters=self.DENSE_2_UNITS, kernel_size=1,
        #                                      activation=activation, name="dense2")
        self.dense3 = tf.keras.layers.Conv1D(filters=self.DENSE_3_UNITS, kernel_size=1,
                                             activation=activation, name="dense3")
        self.v = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation=None,
                                        kernel_regularizer=kr, name="value_conv")
        self.a = tf.keras.layers.Conv1D(filters=action_space_size, kernel_size=1, activation=None,
                                        kernel_regularizer=kr, name="advantage_conv")
        # self.v = tf.keras.layers.LSTM(1, activation=None, return_sequences=True, return_state=True, name="value_LSTM")
        # self.a = tf.keras.layers.LSTM(action_space_size, activation=None, return_sequences=True, return_state=True, name="advantage_LSTM")
        # self.v = Dense(1, activation=None, name="value_dense")
        # self.a = Dense(action_space_size, activation=None, name="advantage_dense")
        self.action_space_size = action_space_size
        self.batch_size = batch_size
        self.use_lstm_states = use_lstm_states
        self.state_1 = None
        self.state_2 = None
        self.reset_lstm_states()

    def call(self, obs_and_states, training=None, mask=None):
        obs, lstm_states = self.unpack_input(obs_and_states)
        x = self.core_net(obs, lstm_states)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=-1, keepdims=True))
        return Q

    def advantage(self, obs_and_states):
        obs, lstm_states = self.unpack_input(obs_and_states)
        x = self.core_net(obs, lstm_states)
        a = self.a(x)
        return a

    def value_advantage(self, obs_and_states):
        obs, lstm_states = self.unpack_input(obs_and_states)
        x = self.core_net(obs, lstm_states)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=-1, keepdims=True))
        return v, a, (Q, tf.math.reduce_mean(a, axis=-1, keepdims=True))

    def unpack_input(self, obs_and_states):
        if isinstance(obs_and_states, tuple):
            obs, lstm_states = obs_and_states
        else:
            obs = obs_and_states
            lstm_states = None
        return obs, lstm_states

    def core_net(self, obs, lstm_states):
        if lstm_states is not None:
            self.set_lstm_states(lstm_states)
        if not self.use_lstm_states:
            self.reset_lstm_states()
        x = obs

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.conv5(x)
        # x = self.max_pooling(x)
        shape = x.shape
        x = tf.reshape(x, [shape[0], shape[1], np.prod(shape[2:])])
        # x = self.dense1(x)
        x, state_h_1, state_c_1 = self.lstm1(x, initial_state=self.state_1)
        # x, state_h_2, state_c_2 = self.lstm2(x, initial_state=self.state_2)
        # x = self.dense2(x)
        x = self.dense3(x)
        self.state_1 = [state_h_1, state_c_1]
        # self.state_2 = [state_h_2, state_c_2]
        return x

    def reset_lstm_states(self):
        self.state_1 = [tf.zeros((self.batch_size, self.LSTM_1_UNITS)), tf.zeros((self.batch_size, self.LSTM_1_UNITS))]
        self.state_2 = [tf.zeros((self.batch_size, self.LSTM_2_UNITS)), tf.zeros((self.batch_size, self.LSTM_2_UNITS))]

    def get_lstm_states(self):
        return self.state_1, self.state_2

    def set_lstm_states(self, states):
        self.state_1, self.state_2 = states

    def debug_func(self, obs):

        x = obs

        x = self.conv1(x)
        # print(f"shape_obs: {obs.shape}")
        # print(f"shape_conv1: {x.shape}")
        # x = tf.keras.layers.Dropout(0.2)(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x1 = self.max_pooling(x)

        # x1 = self.conv_tmp(x)

        shape = x1.shape
        x2 = tf.reshape(x1, [shape[0], shape[1], np.prod(shape[2:])])
        x3 = self.dense1(x2)
        x4, state_h_1, state_c_1 = self.lstm1(x3, initial_state=self.state_1)
        x5, state_h_2, state_c_2 = self.lstm2(x4, initial_state=self.state_2)
        # x = tf.keras.layers.Dropout(0.2)(x)
        x6 = self.dense2(x5)

        # w1 = self.conv_tmp.get_weights()
        w2 = self.dense1.get_weights()
        w3 = self.lstm1.get_weights()
        w4 = self.lstm2.get_weights()
        w5 = self.dense2.get_weights()

        v = self.v(x6)
        a = self.a(x6)
        return x1, x2, x3, x4, x5, x6, w2, w3, w4, w5, v, a

