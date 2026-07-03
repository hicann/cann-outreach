import torch


def compute(inputs, attrs):
    x = inputs[0]
    return torch.abs(x)
