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
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    replay_buffer = ReplayBuffer(env.observation_space)
    actor = ActorAgent(env, replay_buffer)
    learner = LearnerAgent(state_size, action_size, replay_buffer)
    EPISODES = 200
    try:
        learner.load("agent_64")
        print("weights loaded")
    except IOError:
        print("unable to load weights")
    print(learner.q_net.get_weights())

    has_trained_once = False
    for ep in range(EPISODES):
        print(f"EPISODE: {ep}")
        print(f"epsilon: {actor.epsilon}")
        actor.act()
        learner.train(5)

        if ep % 5 == 0:
            learner.save("agent_64")
            actor.update(learner.q_net)
            if ep > 0:
                print(f"average loss:  {statistics.mean([statistics.mean(e) for e in learner.history['loss']])}")
        actor.update_epsilon()
        if ep % 10 == 0:
            for loss, e1, e2, e3, e4, e5 in replay_buffer.episode_mem.values():
                print(f"len(e): {len(e1)}")

    print(learner.q_net.get_weights())
    plt.plot([statistics.mean(e) for e in learner.history['loss']])
    plt.show()


def test_actor_learner_agents():
    env = gym.make('CartPole-v1', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    replay_buffer = ReplayBuffer(env.observation_space)
    actor = ActorAgent(env, None)
    learner = LearnerAgent(state_size, action_size, replay_buffer)
    EPISODES = 5
    try:
        learner.load("agent_64")
        actor.update(learner.q_net)
        print("weights loaded")
    except FileNotFoundError:
        print("unable to load weights")
        return

    has_trained_once = False
    for ep in range(EPISODES):
        print(f"EPISODE: {ep}")
        reward = actor.act(test=True)
        print(f"total reward: {reward}")




# def test_lstm():
#     env = gym.make('CartPole-v1')
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     agentoo7 = LSTMAgent(state_size, action_size, env.observation_space)
#     try:
#         agentoo7.load("agent_64")
#     except:
#         print("unable to load agent")
#     EPISODES = 20
#     SEQ_LEN = 50
#     for s in range(EPISODES):
#         print(f"EPISODE: {s}")
#         done = False
#         state, _ = env.reset()
#         state = np.reshape(state, [1, state_size])
#         episode_reward = 0
#         agentoo7.reset_net_states()
#         state_list = []
#         action_list = []
#         reward_list = []
#         next_state_list = []
#         done_list = []
#         while not done:
#             # env.render()
#             action = agentoo7.act(state)
#             next_state, reward, done, _, _ = env.step(action)
#
#             state = np.reshape(state, [state_size])
#             state_list.append(state)
#             action_list.append(action)
#             reward_list.append(reward)
#             next_state_list.append(next_state)
#             done_list.append(done)
#             next_state = np.reshape(next_state, [1, state_size])
#
#             mem_len = len(state)
#             if mem_len >= SEQ_LEN:
#                 agentoo7.memorize(state_list, action_list, reward_list, next_state_list, done_list)
#                 middle = round(mem_len / 2)
#                 state_list = state_list[-middle:]
#                 action_list = action_list[-middle:]
#                 reward_list = reward_list[-middle:]
#                 next_state_list = next_state_list[-middle:]
#                 done_list = done_list[-middle:]
#
#             agentoo7.train()
#             agentoo7.update_epsilon()
#             # if s > 4:
#             #     agentoo7.see_advancements()
#             state = next_state
#             episode_reward += reward
#
#             if done:
#                 print(f"total reward after {s} episode is {episode_reward} and epsilon is {agentoo7.epsilon}")
#         agentoo7.save("agent_64")
#
#
#     losses = agentoo7.history["loss"]
#     plt.plot(np.arange(0, len(losses), 1), losses)
#     plt.show()
#     input("Press Enter to continue...")
#     env = gym.make('CartPole-v1', render_mode="human")
#     total_reward = 0
#     for s in range(5):
#         done = False
#         state, _ = env.reset()
#         state = np.reshape(state, [1, state_size])
#         episode_reward = 0
#         agentoo7.reset_net_states()
#         while not done:
#             env.render()
#             action = agentoo7.act(state, test=True)
#             print(f"action: {action}")
#             next_state, reward, done, _, _ = env.step(action)
#             next_state = np.reshape(next_state, [1, state_size])
#             state = next_state
#             episode_reward += reward
#             total_reward += reward
#
#             if done:
#                 print(f"total reward after {s} episode is {episode_reward} and epsilon is {agentoo7.epsilon}")
#
#     print(total_reward)

