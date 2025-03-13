import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from model.config import SAVE_CHECKPOINTS
from model.discriminator import Discriminator
from model.generator import Generator
from model.image_buffer import ImageBuffer


class CycleGAN(nn.Module):

    def __init__(
        self,
        generator_X: Generator,
        generator_Y: Generator,
        discriminator_X: Discriminator,
        discriminator_Y: Discriminator,
        gen_optimizer: optim.Optimizer,
        disc_optimizer: optim.Optimizer,
        tensorboard_writer: torch.utils.tensorboard.SummaryWriter,
        discriminator_loss_factor: float = 0.5,
        lambda_cycle: float = 10,
        lambda_identity: float = 0.5,
    ):
        super().__init__()
        """
        Naming conventions:
        - Generators: The letter indicates the target domain.
        - generator_X: Converts images from domain Y to domain X (Y → X)
        - generator_Y: Converts images from domain X to domain Y (X → Y)

        - Discriminators: The letter indicates the domain of the input image.
        - discriminator_X: Evaluates images in domain X (real or generated)
        - discriminator_Y: Evaluates images in domain Y (real or generated)
        """
        # Generators
        self.generator_X = generator_X
        self.generator_Y = generator_Y

        # Discriminators
        self.discriminator_X = discriminator_X
        self.discriminator_Y = discriminator_Y

        # Image Buffers
        self.X_buffer = ImageBuffer(50)
        self.Y_buffer = ImageBuffer(50)

        # Optimizers
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.gen_scaler = torch.amp.GradScaler()
        self.disc_scaler = torch.amp.GradScaler()

        self.disc_loss_factor = discriminator_loss_factor
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_cycle * lambda_identity

        self.gen_loss = 0
        self.disc_loss = 0

        self.writer = tensorboard_writer

    def _train_discriminator(
        self, real_x: torch.Tensor, real_y: torch.Tensor, device: torch.device
    ):
        with torch.amp.autocast(device):
            # Fake generated images to domain Y
            fake_y = self.generator_Y(real_x)

            # Getting predictions for real and fake images
            disc_y_real = self.discriminator_Y(real_y)
            disc_y_fake = self.discriminator_Y(fake_y.detach())

            # Calculating losses. Real images should be classified as 1 and fake as 0
            disc_y_real_loss = self.mse(disc_y_real, torch.ones_like(disc_y_real))
            disc_y_fake_loss = self.mse(disc_y_fake, torch.zeros_like(disc_y_fake))

            disc_y_loss = disc_y_real_loss + disc_y_fake_loss

            # Fake generated images to domain X
            fake_x = self.generator_X(real_y)

            # Getting predictions for real and fake images
            disc_x_real = self.discriminator_X(real_x)
            disc_x_fake = self.discriminator_X(fake_x.detach())

            # Calculating losses. Real images should be classified as 1 and fake as 0
            disc_x_real_loss = self.mse(disc_x_real, torch.ones_like(disc_x_real))
            disc_x_fake_loss = self.mse(disc_x_fake, torch.zeros_like(disc_x_fake))

            disc_x_loss = disc_x_real_loss + disc_x_fake_loss

            # Total discriminator loss
            total_disc_loss = (disc_y_loss + disc_x_loss) * self.disc_loss_factor

        return total_disc_loss, fake_x, fake_y

    def _train_generator(
        self,
        real_x: torch.Tensor,
        real_y: torch.Tensor,
        fake_x: torch.Tensor,
        fake_y: torch.Tensor,
        device: torch.device,
    ):
        with torch.amp.autocast(device):
            # Getting predictions for fake images
            disc_y_fake = self.discriminator_Y(fake_y)
            disc_x_fake = self.discriminator_X(fake_x)

            # Adversiral losses: Generators should fool discriminators
            gen_y_loss = self.mse(disc_y_fake, torch.ones_like(disc_y_fake))
            gen_x_loss = self.mse(disc_x_fake, torch.ones_like(disc_x_fake))

            # Total adversarial loss
            total_adv_loss = gen_y_loss + gen_x_loss

            # Cycle consistency loss: Images should be reconstructed to the original domain
            cycle_y = self.generator_Y(fake_x)
            cycle_x = self.generator_X(fake_y)

            cycle_y_loss = self.l1(real_y, cycle_y) * self.lambda_cycle
            cycle_x_loss = self.l1(real_x, cycle_x) * self.lambda_cycle

            # Identity mapping loss: Images should not be changed if they are already in the target domain
            identity_y = self.generator_Y(real_y)
            identity_x = self.generator_X(real_x)

            identity_y_loss = self.l1(real_y, identity_y) * self.lambda_identity
            identity_x_loss = self.l1(real_x, identity_x) * self.lambda_identity

            # add everything together
            total_gen_loss = (
                total_adv_loss
                + cycle_y_loss
                + cycle_x_loss
                + identity_y_loss
                + identity_x_loss
            )

        return total_gen_loss

    def backward(self):
        self.disc_optimizer.zero_grad()
        self.disc_scaler.scale(self.disc_loss).backward()
        self.disc_scaler.step(self.disc_optimizer)
        self.disc_scaler.update()

        self.gen_optimizer.zero_grad()
        self.gen_scaler.scale(self.gen_loss).backward()
        self.gen_scaler.step(self.gen_optimizer)
        self.gen_scaler.update()

    def forward(self, real_x, real_y, device):
        disc_loss, fake_x, fake_y = self._train_discriminator(real_x, real_y, device)
        gen_loss = self._train_generator(real_x, real_y, fake_x, fake_y, device)

        self.disc_loss = disc_loss
        self.gen_loss = gen_loss

        return {
            "gen_loss": gen_loss.item(),
            "disc_loss": disc_loss.item(),
            "fake_x": fake_x,
            "fake_y": fake_y,
        }

    def save_checkpoint(self, epoch, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "generator_X": self.generator_X.state_dict(),
            "generator_Y": self.generator_Y.state_dict(),
            "discriminator_X": self.discriminator_X.state_dict(),
            "discriminator_Y": self.discriminator_Y.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{checkpoint_dir}/cyclegan_epoch_{epoch + 1}.pth")

    def load_checkpoint(self, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.generator_X.load_state_dict(checkpoint["generator_X"])
        self.generator_Y.load_state_dict(checkpoint["generator_Y"])
        self.discriminator_X.load_state_dict(checkpoint["discriminator_X"])
        self.discriminator_Y.load_state_dict(checkpoint["discriminator_Y"])

        self.gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])

        print(
            f"Checkpoint loaded from {checkpoint_path}, starting from epoch {checkpoint['epoch'] + 1}"
        )
        return checkpoint["epoch"] + 1


def _show_images_in_tensorboard(images, name, writer, step):
    img_grid = vutils.make_grid(images, normalize=True)
    writer.add_image(name, img_grid, step)


def train(
    model: CycleGAN,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    checkpoint_dir: str,
    writer: torch.utils.tensorboard.SummaryWriter,
    num_epochs: int = 100,
    save_every: int = 10,
    start_epoch: int = 0,
    scheduler_gen: torch.optim.lr_scheduler._LRScheduler = None,
    scheduler_disc: torch.optim.lr_scheduler._LRScheduler = None,
):
    # Create a profiler
    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/cyclegan_trace"),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )
    prof.start()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        start_time = time.time()

        loop = tqdm(train_loader, leave=True)

        for idx, (real_X, real_Y) in enumerate(loop):
            real_X = real_X.to(device)
            real_Y = real_Y.to(device)

            train_results = model(real_X, real_Y, device)
            model.backward()

            loop.set_postfix(
                {
                    "Gen Loss": train_results["gen_loss"],
                    "Disc Loss": train_results["disc_loss"],
                }
            )
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

            # add to tensorboard
            writer.add_scalar("Loss/Gen/Train", train_results["gen_loss"], epoch)
            writer.add_scalar("Loss/Disc/Train", train_results["disc_loss"], epoch)

            prof.step()

        # Show images in tensorboard every 5 epochs
        if epoch % 5 == 0:
            _show_images_in_tensorboard(
                train_results["fake_x"], "Generated_X", writer, epoch
            )
            _show_images_in_tensorboard(
                train_results["fake_y"], "Generated_Y", writer, epoch
            )

        model.eval()
        val_gen_loss, val_disc_loss = 0, 0
        with torch.no_grad():
            for real_X, real_Y in val_loader:
                real_X = real_X.to(device)
                real_Y = real_Y.to(device)

                val_losses = model(real_X, real_Y, device)

                # save losses for visualization
                val_gen_loss += val_losses["gen_loss"]
                val_disc_loss += val_losses["disc_loss"]

                # add to tensorboard
                writer.add_scalar("Loss/Gen/Val", val_losses["gen_loss"], epoch)
                writer.add_scalar("Loss/Disc/Val", val_losses["disc_loss"], epoch)

            val_gen_loss /= len(val_loader)
            val_disc_loss /= len(val_loader)

        time_diff = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{num_epochs}] - {time_diff:.2f}s. "
            f"- Train Gen Loss: {train_results['gen_loss']:.4f},"
            f" Train Disc Loss: {train_results['disc_loss']:.4f},"
            f" Val Gen Loss: {val_gen_loss:.4f},"
            f" Val Disc Loss: {val_disc_loss:.4f}"
        )

        if scheduler_gen is not None:
            scheduler_gen.step(epoch=epoch)
            writer.add_scalar("LR/Gen", scheduler_gen.get_last_lr()[0], epoch)

        if scheduler_disc is not None:
            scheduler_disc.step(epoch=epoch)
            writer.add_scalar("LR/Disc", scheduler_disc.get_last_lr()[0], epoch)

        if (epoch + 1) % save_every == 0 and SAVE_CHECKPOINTS:
            model.save_checkpoint(epoch, checkpoint_dir)

    prof.stop()
