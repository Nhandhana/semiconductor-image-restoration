import os
import argparse
import numpy as np
import torch

from model import KLABaseline


def main():

    parser = argparse.ArgumentParser(
        description="Restore a single degraded image"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained .pth model"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .npy image"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output .npy image"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = KLABaseline().to(device)

    model.load_state_dict(
        torch.load(
            args.model,
            map_location=device
        )
    )

    model.eval()

    image = np.load(
        args.input
    )

    print(
        "Input shape:",
        image.shape
    )

    image = torch.from_numpy(
        image
    ).float()

    if image.ndim == 2:
        image = image.unsqueeze(0)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        restored = model(
            image
        )

    restored = (
        restored
        .squeeze()
        .cpu()
        .numpy()
    )

    os.makedirs(
        os.path.dirname(
            args.output
        ) or ".",
        exist_ok=True
    )

    np.save(
        args.output,
        restored
    )

    print(
        "Output shape:",
        restored.shape
    )

    print(
        "Restored image saved:",
        args.output
    )


if __name__ == "__main__":
    main()
