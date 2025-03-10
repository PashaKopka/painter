import os

import torch

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
IMAGE_SIZE = 256
RANDOM_SEED = 42
NUM_OF_GENERATOR_RESNET_BLOCKS = 9
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = "checkpoints"
LOAD_CHECKPOINTS = True
