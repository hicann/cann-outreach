# SoftShrinkGrad

`SoftShrinkGrad` computes the backward gradient of the SoftShrink activation:

```text
output_y[i] = input_grad[i],  input_x[i] > lambd or input_x[i] < -lambd
              0,              otherwise
```

`lambd` is an optional non-negative float attribute whose default value is `0.5`.

## Inputs and output

| Name | Direction | Data type | Format | Description |
| --- | --- | --- | --- | --- |
| `input_grad` | input | `float16`, `float32`, `bfloat16` | ND | Upstream gradient |
| `input_x` | input | Same as `input_grad` | ND | Input of the forward SoftShrink operation |
| `output_y` | output | Same as the inputs | ND | Gradient with respect to `input_x` |

The two inputs and the output have the same shape and data type. The Ascend C implementation supports
Ascend 910B-series devices.
