import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

BATCH_SIZE = 8
NUM_WORKERS = 8

IMAGE_SIZE = 256

NUM_OF_GENERATOR_RESNET_BLOCKS = 9
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = "checkpoints"
LOAD_CHECKPOINTS = True
