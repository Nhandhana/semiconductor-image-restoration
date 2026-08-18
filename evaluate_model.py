import os
import sys
import time
import argparse
import numpy as np
import torch

from model import KLABaseline


def calculate_psnr(predicted, target):

    mse = torch.mean(
        (predicted - target) ** 2
    )

    if mse.item() == 0:
        return float("inf")

    return (
        10 * torch.log10(
            1.0 / mse
        )
    ).item()


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate KLA restoration model"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing test .npy images"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where restored images will be saved"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained .pth model"
    )

    parser.add_argument(
        "--gt_dir",
        default=None,
        help="Optional GT directory for PSNR calculation"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = KLABaseline().to(device)

    state_dict = torch.load(
        args.model,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith(".npy")
    ])

    print(
        "Input images:",
        len(files)
    )

    total_time = 0.0
    psnr_values = []

    for filename in files:

        input_path = os.path.join(
            args.input_dir,
            filename
        )

        output_path = os.path.join(
            args.output_dir,
            filename
        )

        image = np.load(
            input_path
        )

        image = torch.from_numpy(
            image
        ).float()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        image = image.unsqueeze(0)

        image = image.to(device)

        start = time.perf_counter()

        with torch.no_grad():

            restored = model(
                image
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        total_time += elapsed

        restored = (
            restored
            .squeeze()
            .cpu()
            .numpy()
        )

        np.save(
            output_path,
            restored
        )

        # Optional PSNR
        if args.gt_dir is not None:

            gt_path = os.path.join(
                args.gt_dir,
                filename
            )

            if os.path.exists(
                gt_path
            ):

                gt = np.load(
                    gt_path
                )

                gt = torch.from_numpy(
                    gt
                ).float()

                restored_tensor = torch.from_numpy(
                    restored
                ).float()

                psnr = calculate_psnr(
                    restored_tensor,
                    gt
                )

                psnr_values.append(
                    psnr
                )

    print()
    print("Evaluation complete.")
    print(
        "Restored images:",
        len(files)
    )

    if len(files) > 0:

        average_time = (
            total_time / len(files)
        )

        print(
            "Average inference time:",
            average_time,
            "seconds/image"
        )

    if len(psnr_values) > 0:

        print(
            "Average PSNR:",
            sum(psnr_values)
            / len(psnr_values),
            "dB"
        )


if __name__ == "__main__":
    main()
