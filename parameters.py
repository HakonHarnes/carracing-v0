
# Optimization parameters 
REDUCED_INPUT = False 
GRAYSCALE = True
ACTION_REPEAT = 8
FRAME_STACK = 4

# Training parameters
LOAD_PARAMETERS = False
LEARNING_RATE = 0.001
MAX_EPISODES = 3000
BUFFER_SIZE = 6000
BATCH_SIZE = 128
MAX_STEPS = 2000
EPSILON = 0.1
GAMMA = 0.99
EPOCHS = 8
SEED = 28

# Testing parameters
LOAD_PARAMETER_EPISODE = 0
RENDER_EPISODES = 10

# Logging
LOG_INTERVAL = 1
PLOT_INTERVAL = 10
SAVE_INTERVAL = 200
VIDEO_INTERVAL = 200