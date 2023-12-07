"""
To store all the hyperparameters and constants
"""
from torch import cuda
import neptune

# Random seed
SEED = 1
DEVICE = "cuda" if cuda.is_available() else "cpu"
NEPTUNE_API = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZjM3ODdiMy00ZmNhLTRkODUtODYyMi1mMGM5MzM5OTk4MWUifQ=="


# Images
N_CHANNEL = 1
IMG_SIZE = 28

# Hyperparameters
N_EPOCHS = 1
LR = 0.01
MOMENTUM = 0.5

LOG_INTERVAL = 10

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000

# Neptune run
run: neptune.Run
