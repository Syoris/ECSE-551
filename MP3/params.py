"""
To store all the hyperparameters and constants
"""
from torch import cuda

# Random seed
SEED = 1
DEVICE = "cuda" if cuda.is_available() else "cpu"

# Hyperparameters
n_epochs = 2
learning_rate = 0.01
momentum = 0.5
log_interval = 10
