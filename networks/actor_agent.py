import copy
import numpy as np
import random
import tensorflow as tf

from utils.replay_buffer import ReplayBuffer
from networks.lstm_based_dddql import LSTMBasedNet

from utils.preprocessing import get_state_size
from utils.const import IMG_STACK_COUNT


class ActorAgent:

    def __init__(self, env, seq_len, replay_buffer: ReplayBuffer = None, test=False, epsilon=None):
        self.env = copy.deepcopy(env)
        self.seq_len = seq_len
        self.env.reset()
        self.state_size = get_state_size(list(env.observation_space.shape))
        self.action_size = env.action_space.n
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(0, 5, 0)
        self._build_model()

        self.epsilon = 0 if test else epsilon if epsilon is not None else 1  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.is_initialized = False

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        self.agent = LSTMBasedNet(state_size=self.state_size, action_space_size=self.action_size, batch_size=1)
        opt = tf.keras.optimizers.Adam()

        # should not be needed to specify the loss function,
        #   as this net will never train but just copy another trained network
        # loss_func = tf.keras.losses.Huber()
        loss_func = tf.keras.losses.mse
        self.agent.compile(loss=loss_func, optimizer=opt, run_eagerly=True)
        self.agent.predict(tf.zeros([1, 1] + self.state_size), verbose=0)

    def _act(self, state, test=False):
        if np.random.rand() <= self.epsilon and not test:
            return random.randrange(self.action_size)
        # state = tf.expand_dims(state, axis=0)
        act_values = self.agent.advantage(np.array([state]))
        # print(np.argmax(act_values))
        return np.argmax(act_values)  # returns action

    def act(self, trials=1000, test=False):
        for ep in range(trials):
            done = False
            state, _ = self.env.reset()
            state = state.astype(float)
            state = self.normalize_state(state)
            state = state.astype('float16')
            state_queue = [state] * IMG_STACK_COUNT
            sk_state = self.stack_states(state_queue)
            sk_state = np.reshape(sk_state, [1] + list(sk_state.shape))
            episode_reward = 0
            # self.agent.predict(np.array([state]))
            self.agent.reset_lstm_states()
            state_list = []
            action_list = []
            reward_list = []
            next_state_list = []
            done_list = []
            saved_something = False
            total_reward = 0
            debug_first_time = True
            while not done:
                action = self._act(sk_state, test=test)
                next_state, reward, done, _, _ = self.env.step(action)
                if reward != 0:
                    print(f"reward: {reward}")
                next_state = self.normalize_state(next_state)
                # print(f"reward: {reward}")
                next_state = next_state.astype('float16')
                state_queue = self.add_to_queue(state_queue, next_state)
                sk_next_state = self.stack_states(state_queue)
                if test:
                    print(f"action: {action}")
                    print("rendering")
                    self.env.render()
                total_reward += reward
                sk_state = np.reshape(sk_state, list(sk_state.shape)[1:])
                state_list.append(sk_state)
                action_list.append(action)
                reward_list.append(reward)
                next_state_list.append(sk_next_state)
                done_list.append(done)
                sk_next_state = np.reshape(sk_next_state, [1] + list(sk_next_state.shape))

                mem_len = len(state_list)
                if self.replay_buffer is not None:
                    if done or mem_len >= self.seq_len:
                        self.replay_buffer.add_exp(state_list, action_list, reward_list, next_state_list, done_list)
                        middle = round(mem_len / 2)
                        state_list = state_list[-middle:]
                        action_list = action_list[-middle:]
                        reward_list = reward_list[-middle:]
                        next_state_list = next_state_list[-middle:]
                        done_list = done_list[-middle:]
                        saved_something = True

                sk_state = sk_next_state
                episode_reward += reward
            if saved_something or test:
                print(f"action_list: {action_list}")
                print(f"total_trial: {ep}")
                return total_reward

    def update(self, trained_agent: LSTMBasedNet):
        self.agent.set_weights(trained_agent.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        return self.epsilon

    def normalize_state(self, state):
        min = 0
        max = 255
        range = max - min
        return (state - min) / range if range > 0 else state - min

    def add_to_queue(self, queue, element):
        queue.pop(0)
        queue.append(element)
        return queue

    def stack_states(self, states):
        return np.concatenate(states.copy(), axis=-1)





