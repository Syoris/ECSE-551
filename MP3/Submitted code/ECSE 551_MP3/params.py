"""
To store all the hyperparameters and constants
"""
from torch import cuda

# Random seed
SEED = 1
DEVICE = "cuda" if cuda.is_available() else "cpu"
NUM_WORKERS = 0  # os.cpu_count()
print(f'DEVICE: {DEVICE}')

NEPTUNE_API = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZjM3ODdiMy00ZmNhLTRkODUtODYyMi1mMGM5MzM5OTk4MWUifQ=="

N_CHANNEL = 1  # Images

LOG_INTERVAL = 10
