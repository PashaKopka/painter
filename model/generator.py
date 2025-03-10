import torch
from torch import nn

from model.config import NUM_OF_GENERATOR_RESNET_BLOCKS


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_downsampling: bool = True,
        use_activation: bool = True,
        **kwargs,
    ):
        super().__init__()
        layers = []

        if is_downsampling:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    padding_mode="reflect",
                    **kwargs,
                )
            )
        else:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    **kwargs,
                )
            )

        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True) if use_activation else nn.Identity())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ResnetBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            ConvBlock(
                channels,
                channels,
                use_activation=False,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):

    def __init__(
        self,
        img_channels: int = 3,
        num_of_downsampling: int = 2,
        num_resnet_blocks=NUM_OF_GENERATOR_RESNET_BLOCKS,
    ):
        super().__init__()

        num_features = 64

        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.down_blocks = nn.ModuleList()
        for _ in range(num_of_downsampling):
            self.down_blocks.append(
                ConvBlock(
                    num_features,
                    num_features * 2,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            num_features *= 2

        # ResNet blocks
        self.resnet_blocks = nn.Sequential(
            *[ResnetBlock(num_features) for _ in range(num_resnet_blocks)]
        )

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for _ in range(num_of_downsampling):
            self.up_blocks.append(
                ConvBlock(
                    num_features,
                    num_features // 2,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            num_features //= 2

        self.final = nn.Conv2d(
            num_features,
            img_channels,
            kernel_size=7,
            padding=3,
            stride=1,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.resnet_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.final(x)
        return torch.tanh(x)


# TODO change relu on leaky_relu?
