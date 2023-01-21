import gym
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.python.keras
from keras.layers import Dense
from tensorflow.python.keras.layers import InputLayer, LSTM
import tensorflow as tf
import random
from typing import List

from networks.actor_agent import ActorAgent
from networks.learner_agent import LearnerAgent
from utils.replay_buffer import ReplayBuffer
import statistics
from utils.const import ACT_SEQ_SIZE, BATCH_SIZE, GAME_NAME, MIN_BUFFER_SIZE, SAVE_AGENT_NAME, N_STEP, TOTAL_TRAIN_STEPS
from utils.file_manager import save_learner, load_learner, restore_results, load_confs, save_confs


# game_name = "CartPole-v1"


def lstm_tutorial():
    from tensorflow.python.keras import Sequential
    anction_space_dim = 50
    model = Sequential()
    model.add(InputLayer(input_shape=(1, anction_space_dim), batch_size=1))
    model.add(LSTM(128, input_shape=(32, anction_space_dim), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='sgd')

    data = np.random.random((1, 1, anction_space_dim)).astype(np.float32)
    pred = model.predict(data)
    print(pred)


def train_actor_learner_agents():
    starting_epsilon = 1
    env = gym.make(GAME_NAME)
    state_size = list(env.observation_space.shape)
    action_size = env.action_space.n
    replay_buffer = ReplayBuffer(env.observation_space, batch_size=BATCH_SIZE, n_step=N_STEP, train_batch_len=round(ACT_SEQ_SIZE / 2))
    actor = ActorAgent(env, ACT_SEQ_SIZE, replay_buffer, epsilon=starting_epsilon)
    learner = LearnerAgent(state_size, action_size, N_STEP, ACT_SEQ_SIZE, replay_buffer)

    train_step = 0
    # 26: 0.0026550

    try:
        # learner = load_learner()
        learner.load(SAVE_AGENT_NAME)
        actor.update(learner.q_net)
        train_step = load_confs()
        actor.update_epsilon(train_step)
        print("weights loaded")
    except IOError:
        learner = LearnerAgent(state_size, action_size, N_STEP, ACT_SEQ_SIZE, replay_buffer)
        print("unable to load weights")
    print(learner.q_net.get_weights())

    has_trained_once = False
    # for prep_ep in range(batch_size):
    #     actor.act()
    act_num = 0
    try:
        rewards_list = restore_results()
        actor.results = rewards_list
        print(f"reward_list: {len(rewards_list)}")
    except:
        print("unable to load results")
        rewards_list = []
    test_num = 5
    t = 0
    while train_step < TOTAL_TRAIN_STEPS:
        print(f"train_step: {train_step}")
        print(f"epsilon: {actor.epsilon}")
        print(f"buffer_filled: {replay_buffer.get_size()}")
        # for _ in range(5):
        #     actor.act()
        tot_reward, saved_chunks = actor.act()
        # rewards_list.append(tot_reward)
        if replay_buffer.get_size() < MIN_BUFFER_SIZE:
            actor.epsilon = starting_epsilon
            continue
        train_batch_size = saved_chunks*2
        learner.train(epochs=1, batch_size=train_batch_size)

        if t % 5 == 0:
            print("saving learner")
            # save_learner(learner)
            learner.save(SAVE_AGENT_NAME)
            actor.save_scores(SAVE_AGENT_NAME)
            actor.update(learner.q_net)
            save_confs(train_step)
        actor.update_epsilon(train_step)
        if t % 10 == 0:
            for i, (loss, e1, e2, e3, e4) in enumerate(replay_buffer.episode_mem.values()):
                print(f"{i}-len(e): {len(e1)}, loss: {loss}")
            mean_reward = actor.evaluate()
            rewards_list.append(mean_reward)
            print(f"reward list length: {len(rewards_list)}")
            print(f"mean_reward: {mean_reward}")
            # print("plotting")
            # plt.plot(rewards_list)
            # plt.draw()
        train_step += train_batch_size
        t += 1
    plt.plot(rewards_list)
    plt.show()


def test_actor_learner_agents():

    env = gym.make(GAME_NAME, render_mode="human")
    state_size = list(env.observation_space.shape)
    action_size = env.action_space.n
    replay_buffer = ReplayBuffer(env.observation_space, n_step=5, train_batch_len=80)
    actor = ActorAgent(env, 80, None)
    learner = LearnerAgent(state_size, action_size, 5, 80, replay_buffer)
    EPISODES = 5

    try:
        # learner = load_learner()
        learner.load(SAVE_AGENT_NAME)
        actor.update(learner.q_net)
        print("weights loaded")
    except FileNotFoundError:
        learner = LearnerAgent(state_size, action_size, 5, 80, replay_buffer)
        print("unable to load weights")
        return

    has_trained_once = False
    for ep in range(EPISODES):
        print(f"EPISODE: {ep}")
        reward = actor.act(test=True, render=True)
        print(f"total reward: {reward}")


def test_gym():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()

        if terminated or truncated:
            observation, info = env.reset()

    env.close()



