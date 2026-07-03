import torch


def compute(inputs, attrs):
    x = inputs[0]
    dim = attrs.get("dim")
    keepdim = bool(attrs.get("keepdim", False))
    if dim is None:
        return x.sum()
    return x.sum(dim=dim, keepdim=keepdim)
