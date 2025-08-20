import tensorflow as tf

# Game configuration
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60  # Set to 60 for smooth visualization, 0 for maximum speed

# Car configuration
CAR_WIDTH = 20
CAR_HEIGHT = 10
CAR_MAX_SPEED = 4  # Slightly reduced for better control
CAR_ACCELERATION = 0.15  # Slightly reduced
CAR_ROTATION_SPEED = 2  # Slightly reduced

# Sensor configuration
SENSOR_COUNT = 5
SENSOR_LENGTH = 80  # Reduced for better sensitivity

# RL configuration
STATE_SIZE = SENSOR_COUNT + 2  # 5 sensors + speed + angle
ACTION_SIZE = 3  # left, right, forward
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 20000
TARGET_UPDATE_FREQUENCY = 5

# Training optimization
FAST_TRAINING = True
MAX_STEPS_PER_EPISODE = 500  # Increased for longer episodes
TRAINING_EPISODES = 200
EPISODES_PER_PRINT = 20

# GPU Configuration
USE_GPU = False
GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) > 0

if GPU_AVAILABLE:
    USE_GPU = True
    print("NVIDIA GPU detected! Using GPU for training.")
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    print("No NVIDIA GPU detected. Using CPU for training.")

