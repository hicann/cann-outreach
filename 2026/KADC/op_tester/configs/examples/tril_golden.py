import torch


def compute(inputs, attrs):
    x = inputs[0]
    diagonal = int(attrs.get("diagonal", 0))
    return torch.tril(x, diagonal=diagonal)
