
from networks.LSTM import *
import tensorflow as tf
import time


if __name__ == '__main__':
    start = time.time()
    gpus = tf.config.list_logical_devices('GPU')
    with tf.device("/cpu:0"):
        test_actor_learner_agents()