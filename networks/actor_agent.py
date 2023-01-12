import copy
import numpy as np
import random
import tensorflow as tf

from utils.replay_buffer import ReplayBuffer
from networks.lstm_based_dddql import LSTMBasedNet

from utils.preprocessing import get_state_size
from utils.const import IMG_STACK_COUNT, OPTIMIZER, MAX_ACT_ITERATIONS, EPSILON_DECAY, EPS_1_PROB, RAND_EPS_PROB


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
        self.epsilon_decay = EPSILON_DECAY
        self.is_initialized = False

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        self.agent = LSTMBasedNet(state_size=self.state_size, action_space_size=self.action_size, batch_size=1)
        opt = OPTIMIZER

        # should not be needed to specify the loss function,
        #   as this net will never train but just copy another trained network
        # loss_func = tf.keras.losses.Huber()
        loss_func = tf.keras.losses.mse
        self.agent.compile(loss=loss_func, optimizer=opt, run_eagerly=True)
        self.agent.predict(tf.zeros([1, 1] + self.state_size), verbose=0)

    def _act(self, state, epsilon, test=False):
        if np.random.rand() <= epsilon and not test:
            return random.randrange(self.action_size), False
        # state = tf.expand_dims(state, axis=0)
        act_values = self.agent.advantage(np.array([state]))
        # print(np.argmax(act_values))
        return np.argmax(act_values), True  # returns action

    def act(self, test=False, render=False):
        choice = np.random.rand()
        if choice < EPS_1_PROB:
            episode_epsilon = 1
            print("completely random episode")
        elif choice < EPS_1_PROB + RAND_EPS_PROB:
            episode_epsilon = random.uniform(self.epsilon, 1)
            print(f"random epilon: {episode_epsilon}")
        else:
            episode_epsilon = self.epsilon
            print(f"real epsilon: {episode_epsilon}")
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []
        done = False
        state, _ = self.env.reset()
        state = state.astype(float)
        state = self.normalize_state(state).astype('float16')
        state_queue = [state] * IMG_STACK_COUNT
        state_list.append(state)
        episode_reward = 0
        # self.agent.predict(np.array([state]))
        self.agent.reset_lstm_states()
        saved_something = False
        total_reward = 0
        debug_first_time = True
        time_step = 0
        debug_actions_made = []
        while not done and time_step < MAX_ACT_ITERATIONS:
            sk_state = self.stack_states(state_queue)
            sk_state = np.reshape(sk_state, [1] + list(sk_state.shape))
            action, is_not_rand = self._act(sk_state, episode_epsilon, test=test)
            if is_not_rand:
                debug_actions_made.append(action)
            state, reward, done, _, _ = self.env.step(action)
            if reward != 0:
                time_step = 0
                print(f"reward: {reward}")
            state = self.normalize_state(state).astype('float16')
            # print(f"reward: {reward}")
            state_queue = self.add_to_queue(state_queue, state)
            if render:
                print(f"action: {action}")
                print("rendering")
                self.env.render()
            total_reward += reward
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)
            # sk_next_state = np.reshape(sk_next_state, [1] + list(sk_next_state.shape))

            mem_len = len(action_list)

            if self.replay_buffer is not None and not test:
                if done or time_step == round(self.seq_len / 2 - 1) or mem_len >= self.seq_len:
                    self.replay_buffer.add_exp(state_list, action_list, reward_list, done_list)
                    middle = round(mem_len / 2)
                    state_list = state_list[middle:]
                    action_list = action_list[middle:]
                    reward_list = reward_list[middle:]
                    done_list = done_list[middle:]
                    saved_something = True

            episode_reward += reward
            time_step += 1
        print(f"action_list: {debug_actions_made}")
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





