# Lesson2 SubCustomTemplate

## Overview

This directory implements the Lesson2 custom Ascend C operator `SubCustomTemplate`.

The operator computes element-wise vector subtraction:

```text
z = x - y
```

## Operator Specification

- Operator name: `SubCustomTemplate`
- Input tensors: `x`, `y`
- Output tensor: `z`
- Data types: `float16`, `float32`
- Shape: `(8, 2048)`
- Format: `ND`

## Files

- `sub_custom_template.json`: operator prototype file.
- `custom_op/`: custom operator project generated from the prototype and completed with Ascend C kernel code.
- `custom_op/op_kernel/sub_custom_template.cpp`: Ascend C kernel implementation.
- `custom_op/op_host/sub_custom_template.cpp`: operator definition, shape inference, dtype inference, and tiling.
- `test/main.cpp`: ACL test case for both `float16` and `float32`.
- `run.sh`: builds, installs, and tests the custom operator.

## Build and Run

Run the following commands in a CANN environment:

```bash
cd 2026/CANN-Summer-Camps/Xian-Jiaotong-University/gcw_hNbuFSEs/lesson2
bash run.sh
```

The script performs these steps:

1. Loads the CANN environment.
2. Builds the custom operator package.
3. Installs the generated `.run` package.
4. Loads the custom operator environment.
5. Builds the ACL test case.
6. Runs `float16` and `float32` verification.

## Verification

The operator was verified in a CANN 8.5 environment. Key output:

```text
float16 result first 10:
-2.2500 -2.0000 -1.7500 -1.5000 -1.2500 -1.0000 -0.7500 1.2500 1.5000 1.7500
float16 test pass. maxDiff=0.00000000
float32 result first 10:
-2.2500 -2.0000 -1.7500 -1.5000 -1.2500 -1.0000 -0.7500 1.2500 1.5000 1.7500
float32 test pass. maxDiff=0.00000000
```
