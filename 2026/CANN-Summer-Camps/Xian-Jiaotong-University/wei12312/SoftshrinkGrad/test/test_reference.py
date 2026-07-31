#!/usr/bin/env python3
"""CPU reference and contract tests for SoftshrinkGrad."""

import itertools
import math
import unittest


def _product(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result


def _broadcast_shape(left, right):
    rank = max(len(left), len(right))
    left = (1,) * (rank - len(left)) + tuple(left)
    right = (1,) * (rank - len(right)) + tuple(right)
    output = []
    for lhs, rhs in zip(left, right):
        if lhs == rhs:
            output.append(lhs)
        elif lhs == 1:
            output.append(rhs)
        elif rhs == 1:
            output.append(lhs)
        else:
            raise ValueError("input shapes are not broadcastable")
    return tuple(output)


def _flat_index(output_index, input_shape, output_shape):
    padded = (1,) * (len(output_shape) - len(input_shape)) + tuple(input_shape)
    source_index = [0 if dim == 1 else index
                    for index, dim in zip(output_index, padded)]
    flat = 0
    stride = 1
    for index, dim in zip(reversed(source_index), reversed(padded)):
        flat += index * stride
        stride *= dim
    return flat


def softshrink_grad(grad_output, grad_shape, self_data, self_shape, lambd=0.5):
    if lambd < 0:
        raise ValueError("lambd must be greater than or equal to zero")
    if len(grad_shape) > 8 or len(self_shape) > 8:
        raise ValueError("rank must be in [0, 8]")
    if len(grad_output) != _product(grad_shape):
        raise ValueError("grad_output size does not match grad_shape")
    if len(self_data) != _product(self_shape):
        raise ValueError("self size does not match self_shape")

    output_shape = _broadcast_shape(grad_shape, self_shape)
    if any(dim == 0 for dim in output_shape):
        return [], output_shape

    output = []
    ranges = [range(dim) for dim in output_shape]
    for index in itertools.product(*ranges):
        grad_value = grad_output[_flat_index(index, grad_shape, output_shape)]
        self_value = self_data[_flat_index(index, self_shape, output_shape)]
        output.append(grad_value if abs(self_value) > lambd else 0.0)
    return output, output_shape


class SoftshrinkGradReferenceTest(unittest.TestCase):
    def test_same_shape_and_boundaries(self):
        result, shape = softshrink_grad(
            [1, 2, 3, 4, 5, 6, 7],
            (7,),
            [-1.0, -0.5, -0.49, 0.0, 0.5, 0.51, 2.0],
            (7,),
            0.5,
        )
        self.assertEqual(shape, (7,))
        self.assertEqual(result, [1, 0.0, 0.0, 0.0, 0.0, 6, 7])

    def test_broadcast(self):
        result, shape = softshrink_grad(
            [1, 2, 3],
            (1, 3),
            [-1.0, 0.0],
            (2, 1),
            0.5,
        )
        self.assertEqual(shape, (2, 3))
        self.assertEqual(result, [1, 2, 3, 0.0, 0.0, 0.0])

    def test_scalar(self):
        result, shape = softshrink_grad([3.0], (), [-0.75], (), 0.5)
        self.assertEqual(shape, ())
        self.assertEqual(result, [3.0])

    def test_empty_tensor(self):
        result, shape = softshrink_grad([], (0, 4), [], (0, 4), 0.5)
        self.assertEqual(shape, (0, 4))
        self.assertEqual(result, [])

    def test_empty_tensor_broadcast(self):
        result, shape = softshrink_grad([], (0, 4), [1.0] * 4, (1, 4), 0.5)
        self.assertEqual(shape, (0, 4))
        self.assertEqual(result, [])

    def test_nan_is_masked(self):
        result, _ = softshrink_grad([2.0], (1,), [math.nan], (1,), 0.5)
        self.assertEqual(result, [0.0])

    def test_negative_lambda_is_rejected(self):
        with self.assertRaises(ValueError):
            softshrink_grad([1.0], (1,), [1.0], (1,), -0.1)

    def test_incompatible_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            softshrink_grad([1.0, 2.0], (2,), [1.0, 2.0, 3.0], (3,), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
