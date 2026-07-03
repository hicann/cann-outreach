import torch


def compute(inputs, attrs):
    a, b = inputs[0], inputs[1]
    return torch.matmul(a.float(), b.float()).to(a.dtype)
