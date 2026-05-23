import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


def get_inputs():
    return [
        torch.randn(1024, 1024, dtype=torch.float16),
        torch.randn(1024, 1024, dtype=torch.float16),
    ]


def get_init_inputs():
    return []