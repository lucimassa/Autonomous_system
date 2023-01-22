import gym
from networks.LSTM_trainer_tester import *
import tensorflow as tf
import time


def test_environment():
    env = gym.make("LunarLander-v2", render_mode="human")
    env.action_space.seed(42)

    observation, info = env.reset(seed=42)

    for i in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


def test_tensorflow_GPU():
    tf.debugging.set_log_device_placement(True)
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test_environment()
    start = time.time()
    gpus = tf.config.list_logical_devices('GPU')
    with tf.device(gpus[0].name):
    # with tf.device("/cpu:0"):

        # test_gym()x
        train_actor_learner_agents()
    print(f"time usd: {time.time() - start}")




