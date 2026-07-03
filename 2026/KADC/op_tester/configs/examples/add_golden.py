import torch


def compute(inputs, attrs):
    x, y = inputs[0], inputs[1]
    return x + y
