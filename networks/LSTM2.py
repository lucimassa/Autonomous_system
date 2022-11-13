import random
import gym
import numpy as np
from collections import deque

import tensorflow.python.keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.python.keras.layers import InputLayer, LSTM, Input
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense


class LSTMBasedNet:
    LSTM_UNITS = 128

    def __init__(self, seq_len, action_space_size):
        self.action_space_size = action_space_size
        self.batch_size = 1
        self.seq_len = seq_len
        self.lstm1 = LSTM(self.LSTM_UNITS, return_sequences=True, return_state=True, name="first_LSTM")
        self.lstm2 = LSTM(self.LSTM_UNITS, return_sequences=True, return_state=True, name="second_LSTM")
        self.v = LSTM(1, return_sequences=True, return_state=True, name="value_LSTM")
        self.a = LSTM(action_space_size, return_sequences=True, return_state=True, name="advantage_LSTM")

    def build_model(self):
        inp = tf.keras.Input(shape=(self.seq_len, self.action_space_size), batch_size=self.batch_size)
        inp = tf.keras.Input()
        x = Dense(..., name="dense1")(inp)
        y = Dense(..., name="dense2")(x)
        z = Dense(..., name="dense3")(x)
        model1 = tf.keras.Model(
            inputs=inp,
            outputs=[y, z]
        )


def create_dataset(n_X, look_back):
   dataX, dataY = [], []
   for i in range(len(n_X)-look_back):
      a = n_X[i:(i+look_back), ]
      dataX.append(a)
      dataY.append(n_X[i + look_back, ])
   return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    x = np.arange(1, 50, 0.1)
    # y = 0.4 * x + 30
    y = np.sin(x)
    # plt.plot(x, y)
    # plt.show()
    trainx, testx = x[0:int(0.8*(len(x)))], x[int(0.8*(len(x))):]
    trainy, testy = y[0:int(0.8*(len(y)))], y[int(0.8*(len(y))):]
    train = np.array(list(zip(trainx,trainy)))
    test = np.array(list(zip(testx,testy)))
    plt.plot(trainx, trainy)
    plt.plot(testx, testy)
    plt.show()

    look_back = 1
    # trainx, trainy = create_dataset(train, look_back)
    # testx, testy = create_dataset(test, look_back)

    trainx = np.reshape(trainx, (trainx.shape[0], 1, 1))
    testx = np.reshape(testx, (testx.shape[0], 1, 1))
    trainy = np.reshape(trainy, (trainy.shape[0], 1, 1))
    testy = np.reshape(testy, (testy.shape[0], 1, 1))

    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(trainx.shape[1], 1)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(1, return_sequences=True))
    # model.add(Dense(256))
    # model.add(Dense(128))
    # model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.load_weights('LSTMBasic1.h5')
    model.fit(trainx, trainy, epochs=2000, batch_size=10, verbose=2, shuffle=False)
    model.save_weights('LSTMBasic1.h5')

    model.load_weights('LSTMBasic1.h5')
    predict = model.predict(testx)
    plt.plot(np.squeeze(testx), np.squeeze(testy))
    plt.plot(np.squeeze(testx), np.squeeze(predict))
    plt.show()
    predict = model.predict(trainx)
    plt.plot(np.squeeze(trainx), np.squeeze(trainy))
    plt.plot(np.squeeze(trainx), np.squeeze(predict))
    plt.show()

    print("finished")

