
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + residual

        return out


class ResidualRestorationNet(nn.Module):

    def __init__(self, num_blocks=8):
        super().__init__()

        self.input_conv = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        residual = x

        out = self.input_conv(x)
        out = self.blocks(out)
        out = self.output_conv(out)

        return out + residual


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ResidualRestorationNet().to(device)

print("Model ready")
print("Device:", device)

# IMPORTANT:
# The model will be saved after actual training:
#
# torch.save(
#     model.state_dict(),
#     "/content/residual_model.pth"
# )
