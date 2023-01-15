import tensorflow as tf

SAVE_AGENT_NAME = "agent_115"

ACT_SEQ_SIZE = 40
BATCH_SIZE = 16
GAME_NAME = "ALE/Breakout-v5"
IMG_STACK_COUNT = 4         # how many raw images are stacked together to form on state
BUFFER_SIZE = 1000          # 2048
MIN_BUFFER_SIZE = 512         # 512
LEARNING_RATE = 1e-4
MAX_ACT_ITERATIONS = 1000
EPSILON_DECAY = 0.99
# OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
OPTIMIZER = tf.keras.optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)
DICOUNT_RATE = 0.95

# epilon greedy
EPS_1_PROB = .1          # .2
RAND_EPS_PROB = .2       # .2
