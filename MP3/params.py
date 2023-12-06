"""
To store all the hyperparameters and constants
"""
from torch import cuda

# Random seed
SEED = 1
DEVICE = "cuda" if cuda.is_available() else "cpu"

# Hyperparameters
N_EPOCHS = 2
LR = 0.01
MOMENTUM = 0.5

LOG_INTERVAL = 10

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
