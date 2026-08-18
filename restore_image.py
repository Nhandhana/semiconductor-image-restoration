
import torch
import torch.nn as nn
import numpy as np
import os


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

        return out + residual


class ResidualRestorationNet(nn.Module):

    def __init__(self, num_blocks=8):
        super().__init__()

        self.input_conv = nn.Conv2d(
            1, 64,
            kernel_size=3,
            padding=1
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Conv2d(
            64, 1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        residual = x

        out = self.input_conv(x)
        out = self.blocks(out)
        out = self.output_conv(out)

        return out + residual


def restore_image(
    model_path,
    input_path,
    output_path
):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = ResidualRestorationNet().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    image = np.load(input_path)

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    if image.ndim == 2:
        image = image.unsqueeze(0)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        restored = model(image)

    restored = restored.squeeze().cpu().numpy()

    np.save(
        output_path,
        restored
    )

    print(
        "Restored image saved:",
        output_path
    )


print("Restoration script created successfully")
