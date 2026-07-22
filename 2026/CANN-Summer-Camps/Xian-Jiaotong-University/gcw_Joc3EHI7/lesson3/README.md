# Lesson3 AddCustomTemplate Performance Optimization

## Overview

This directory implements the Lesson3 performance optimization task for the custom Ascend C operator `AddCustomTemplate`.

The operator computes element-wise vector addition:

```text
z = x + y
```

## Task Target

- Operator name: `AddCustomTemplate`
- Input tensors: `x`, `y`
- Output tensor: `z`
- Data type: `float`
- Shape: `[45, 20480]`
- Format: `ND`
- Performance target: task duration less than `80us`

## Optimization

The host tiling config uses `blockDim = 40` to split the input across AI Core blocks. For each block, `tileNum` is set to `4` so each tile is large enough to reduce loop and data copy overhead while keeping UB memory usage within a reasonable range.

For this input shape:

```text
totalLength = 45 * 20480 = 921600
blockLength = 921600 / 40 = 23040
tileLength = 23040 / 4 = 5760
```

## Files

- `custom_op/`: custom operator project.
- `custom_op/op_host/add_custom_template.cpp`: operator definition, shape inference, dtype inference, and tiling configuration.
- `custom_op/op_kernel/add_custom_template.cpp`: Ascend C kernel implementation.
- `custom_op/op_kernel/add_custom_template_tiling.h`: tiling data definition.
- `test/main.cpp`: ACL test case for `[45, 20480]` float inputs.
- `run.sh`: builds, installs, tests, and profiles the custom operator.

## Build and Run

Run the following commands in a CANN environment:

```bash
cd 2026/CANN-Summer-Camps/Xian-Jiaotong-University/gcw_hNbuFSEs/lesson3
bash run.sh
```

The script performs these steps:

1. Loads the CANN environment.
2. Builds the custom operator package.
3. Installs the generated `.run` package.
4. Builds the ACL test case.
5. Runs the correctness test with `msprof`.
6. Prints the operator performance result from `OpBasicInfo.csv`.

## Verification

The operator was verified in a CANN environment. Key output:

```text
result is:
3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0
test pass
Task Duration(us): 13.620000
Performance target: PASS (<80us)
```
