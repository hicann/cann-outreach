# AscendC TanhCustom

This repository contains an AscendC custom `TanhCustom` operator experiment and an ACLNN invocation sample.

## Contents

- `TanhCustom/`: custom operator source, tiling, build scripts, and packaging config.
- `AclNNInvocation/`: ACLNN sample used to generate input/golden data, run the operator, and verify output.

## Kernel Implementation

`TanhCustom/op_kernel/tanh_custom.cpp` implements `tanh(x)` with AscendC vector APIs:

```text
tanh(x) = 2 / (1 + exp(-2x)) - 1
```

The tested input/output dtype is `float16`.

## Build And Run

Build and package the custom operator:

```bash
cd TanhCustom
bash build.sh
```

Install the generated operator package:

```bash
cd TanhCustom/build_out
chmod +x custom_opp_openEuler_aarch64.run
./custom_opp_openEuler_aarch64.run --quiet
```

Run the ACLNN invocation and precision check:

```bash
cd AclNNInvocation
bash run.sh
```

## Verified Result

The experiment was verified on Ascend 910B with CANN 9.0.0. `AclNNInvocation/run.sh` completed successfully:

```text
INFO: acl executable run success!
test pass

#####################################
INFO: you have passed the Precision!
#####################################
```
