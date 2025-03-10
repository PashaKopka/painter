import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_instance_norm: bool = True,
        relu_slope: float = 0.2,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            )
        ]
        if use_instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(relu_slope))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_of_blocks=3,
    ):
        super().__init__()

        num_features = 64

        self.initial = ConvBlock(
            in_channels,
            num_features,
            stride=2,
            use_instance_norm=False,
        )

        self.layers = nn.ModuleList()
        for i in range(num_of_blocks):
            stride = 1 if i == num_of_blocks else 2

            self.layers.append(
                ConvBlock(
                    in_channels=num_features,
                    out_channels=num_features * 2,
                    stride=stride,
                    use_instance_norm=True,
                )
            )

            num_features *= 2

        self.final = nn.Conv2d(
            num_features,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return torch.sigmoid(x)
