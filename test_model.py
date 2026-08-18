import sys

sys.path.append("src")

import torch

from model import KLABaseline


model = KLABaseline()


input_image = torch.randn(1, 1, 128, 128)


output = model(input_image)


print("Input shape:", input_image.shape)
print("Output shape:", output.shape)