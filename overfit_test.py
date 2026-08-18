import os
import sys

sys.path.append("src")

import torch
import torch.nn as nn

from dataset import KLADataset
from model import KLABaseline


GT_DIR = "../train/GT"
NOISY_DIR = "../train/NoisyLR"


# Get filenames
files = []

for name in os.listdir(GT_DIR):

    if name.endswith(".npy"):
        files.append(name)

files.sort()


# Use only ONE image
files = [files[0]]


# Dataset
dataset = KLADataset(
    GT_DIR,
    NOISY_DIR,
    files
)


noisy, gt = dataset[0]

noisy = noisy.unsqueeze(0)
gt = gt.unsqueeze(0)


print("Input:", noisy.shape)
print("Target:", gt.shape)


# Model
model = KLABaseline()


# Loss
criterion = nn.L1Loss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Train on one image
for epoch in range(1, 501):

    optimizer.zero_grad()

    prediction = model(noisy)

    loss = criterion(prediction, gt)

    loss.backward()

    optimizer.step()

    if epoch == 1 or epoch % 50 == 0:

        print(
            "Epoch:",
            epoch,
            "Loss:",
            loss.item()
        )