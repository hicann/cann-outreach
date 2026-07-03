import torch


def compute(inputs, attrs):
    a, b = inputs[0], inputs[1]
    if a.dim() == 3:
        return torch.bmm(a, b)
    return torch.matmul(a, b)
