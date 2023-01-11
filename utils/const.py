import tensorflow as tf

ACT_SEQ_SIZE = 40
BATCH_SIZE = 32
GAME_NAME = "ALE/Breakout-v5"
IMG_STACK_COUNT = 4         # how many raw images are stacked together to form on state
BUFFER_SIZE = 514
MIN_BUFFER_SIZE = 256
LEARNING_RATE = 0.001
MAX_ACT_ITERATIONS = 1000
EPSILON_DECAY = 0.991
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# OPTIMIZER = tf.keras.optimizers.RMSprop(lr=5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)
