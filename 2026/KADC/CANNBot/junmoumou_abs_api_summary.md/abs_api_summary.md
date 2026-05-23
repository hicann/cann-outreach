# Abs API 资源汇总

## Abs API 文档（4 类）

| API 类型 | 文档路径 | 支持产品 | 支持数据类型 |
|---------|---------|---------|-------------|
| **SIMD Memory 矢量** | `asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Abs.md` | A3/A2/A200I/AI Core/训练/950PR | half, float, int16, int32, int8, int64（按产品不同） |
| **SIMD Reg 矢量** | `asc-devkit/docs/api/SIMD-API/基础API/Reg矢量计算/基础算术/Abs-15.md` | 仅 950PR/950DT | int8, int16, int32, int64, half, float, complex32, complex64 |
| **C API Reg 矢量** | `asc-devkit/docs/api/c_api/reg/reg_vector/asc_abs.md` | 仅 950PR/950DT | int8, int16, half, int32, float |
| **C API 矢量计算** | `asc-devkit/docs/api/c_api/vector_compute/asc_abs.md` | A3/A2 | half, float |

### SIMD Memory Abs（最常用）

```cpp
// 前n个数据计算
AscendC::Abs(dstLocal, srcLocal, count);

// 高维切分 - mask连续模式
AscendC::Abs(dstLocal, srcLocal, mask, repeatTime, {1, 1, 8, 8});

// 复数取模
AscendC::Abs(dstLocal_float, srcLocal_complex64, count);
```

**约束**：地址需 32 字节对齐；950PR 上 int8/int64 仅支持前n个数据接口。

### SIMD Reg Abs（950PR/950DT 专用）

```cpp
AscendC::Reg::Abs(dstReg, srcReg, mask);
```

**约束**：int8 的 -128 取绝对值会被截断为 -128（非饱和截断）。

## 示例代码

- **Reg Abs 示例**：`asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/03_reg_vector_compute/abs/abs.asc`
  - 演示 `AbsVF` 函数使用 Reg API + `asc_vf_call` 调用
  - 适用于 Relu/Exp/Sqrt/Ln/Log/Neg 等同类单目运算

## 其他 abs 相关 API

| 文档 | 说明 |
|------|------|
| `asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/复合计算/AbsSub(ISASI).md` | Abs + Sub 复合运算 |
| `asc-devkit/docs/api/SIMD-API/基础API/Reg矢量计算/复合计算/AbsSub.md` | Reg 版 AbsSub |
| `asc-devkit/docs/api/SIMT-API/数学函数/half类型/half类型算术函数/__habs.md` | SIMT half abs |
| `asc-devkit/docs/api/SIMT-API/数学函数/bfloat16类型/bfloat16类型算术函数/__habs-150.md` | SIMT bfloat16 abs |
| `asc-devkit/docs/api/Utils-API/C++标准库/数学函数/abs1.md` | C++ std abs |