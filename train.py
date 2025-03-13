import os

base_dir = os.getcwd()

input_dir = os.path.join(base_dir, "data/")

photos_path = os.path.join(input_dir, "monet/photo_jpg")
monet_path = os.path.join(input_dir, "monet/monet_jpg")

import random
import re
from itertools import chain

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import config
from model.dataset import get_loaders
from model.discriminator import Discriminator
from model.generator import Generator
from model.model import CycleGAN, train

BATCH_SIZE = config.BATCH_SIZE
IMG_SIZE = config.IMAGE_SIZE
DEVICE = config.DEVICE
NUM_WORKERS = config.NUM_WORKERS

torch.manual_seed(config.RANDOM_SEED)
torch.cuda.manual_seed_all(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
val_transform = test_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_loader, val_loader = get_loaders(
    photos_path,
    monet_path,
    train_transform,
    val_transform,
    batch_size=BATCH_SIZE,
    train_split=0.9,
    num_workers=NUM_WORKERS,
)


def lr_lambda(epoch):
    if epoch < 100:
        return 1.0
    else:
        return 1.0 - (epoch - 100) / 100


generator_X = Generator()
generator_Y = Generator()
discriminator_X = Discriminator()
discriminator_Y = Discriminator()

gen_params = chain(generator_X.parameters(), generator_Y.parameters())
disc_params = chain(discriminator_X.parameters(), discriminator_Y.parameters())

gen_opt = optim.Adam(gen_params, lr=0.0002, betas=(0.5, 0.999))
disc_opt = optim.Adam(disc_params, lr=0.0002, betas=(0.5, 0.999))

scheduler_gen = lr_scheduler.LambdaLR(gen_opt, lr_lambda)
scheduler_disc = lr_scheduler.LambdaLR(disc_opt, lr_lambda)

writer = SummaryWriter("runs/CycleGAN")

model = CycleGAN(
    generator_X=generator_X,
    generator_Y=generator_Y,
    discriminator_X=discriminator_X,
    discriminator_Y=discriminator_Y,
    gen_optimizer=gen_opt,
    disc_optimizer=disc_opt,
    tensorboard_writer=writer,
    discriminator_loss_factor=0.5,
    lambda_cycle=10,
    lambda_identity=0.5,
).to(DEVICE)


def get_latest_checkpoint(checkpoint_dir, pattern="cyclegan_epoch_(\d+).pth"):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(pattern, f)]
    if not checkpoint_files or not config.LOAD_CHECKPOINTS:
        return None  # No checkpoint found

    # Extract epoch numbers and find the latest one
    latest_checkpoint = max(
        checkpoint_files, key=lambda f: int(re.search(pattern, f).group(1))
    )
    return os.path.join(checkpoint_dir, latest_checkpoint)


def train_with_lambda_cycle(
    model: CycleGAN,
    lambda_cycle_values,
    checkpoint_dir,
    start_epoch=0,
    checkpoint_path=None,
    **kwargs,
):
    """Handles training across multiple lambda_cycle values with automatic checkpoint loading."""

    # Load latest checkpoint if none is specified
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)

    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = model.load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint: {checkpoint_path} (Epoch {start_epoch})")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting from scratch.")

    # Iterate through lambda_cycle values and train
    for lambda_cycle, next_epoch in lambda_cycle_values:
        if start_epoch >= next_epoch:
            print(f"Skipping lambda_cycle={lambda_cycle} (already trained)")
            continue
        model.lambda_cycle = lambda_cycle
        print(f"\nStarting training with lambda_cycle={lambda_cycle}")

        train(
            model=model,
            checkpoint_dir=checkpoint_dir,
            start_epoch=start_epoch,
            num_epochs=next_epoch,
            **kwargs,
        )

        start_epoch = next_epoch  # Update for next phase


lambda_cycle_values = [
    (10, 180),
    (5, 200),
]

train_with_lambda_cycle(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    save_every=5,
    lambda_cycle_values=lambda_cycle_values,
    device=DEVICE.type,
    scheduler_gen=scheduler_gen,
    scheduler_disc=scheduler_disc,
    checkpoint_dir=config.CHECKPOINT_DIR,
    writer=writer,
)

writer.close()
