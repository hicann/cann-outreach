import torch


def compute(inputs, attrs):
    x = inputs[0]
    dim = attrs.get("dim", -1)
    return torch.softmax(x.float(), dim=dim).to(x.dtype)
