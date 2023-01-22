import tensorflow as tf
import os


AGENT_NAME = "agent_117"
SAVE_AGENT_NAME = os.path.join("checkpoints", AGENT_NAME)
N_STEP = 5
TOTAL_TRAIN_STEPS = 3000 * 16
TRAIN_STEPS_BEFORE_MIN_EPS = 2000 * 16
ACT_SEQ_SIZE = 40
BATCH_SIZE = 16
GAME_NAME = "ALE/Breakout-v5"
IMG_STACK_COUNT = 4         # how many raw images are stacked together to form on state
BUFFER_SIZE = 1000
MIN_BUFFER_SIZE = 512
LEARNING_RATE = 1e-4
MAX_ACT_ITERATIONS = 1000
EPSILON_DECAY = 0.9975      # not used anymore
# OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
OPTIMIZER = tf.keras.optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)
DICOUNT_RATE = 0.95

# epilon greedy
EPS_1_PROB = 0          # .1
RAND_EPS_PROB = 0       # .2
