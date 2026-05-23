# sparse_gemm 算子审查报告

> **审查时间**: 2026-05-22  
> **审查轮次**: 第 1 轮  
> **审查结论**: ❌ **FAIL**  
> **总分**: 31/100

---

## 1. 审查结论

| 项目 | 结果 |
|------|------|
| **最终判定** | ❌ FAIL |
| **总分** | 31/100 |
| **必须修复问题数** | 8 |
| **建议修复问题数** | 3 |

**失败原因**：存在多个必须修复问题，包括 CMake 配置错误、架构合规性问题、硬编码参数、同步策略错误、精度问题（输出 nan）。

---

## 2. 评分详情

### 维度 1：编译验证（10 分）

| 检查项 | 分值 | 得分 | 状态 | 说明 |
|--------|------|------|------|------|
| 1.1 独立编译成功 | 7 | 0 | ❌ | CMake 配置缺少 tiling_api 库链接 |
| 1.2 无代码级警告 | 3 | 0 | ❌ | 编译失败，无法验证 |
| **小计** | **10** | **0** | | |

### 维度 2：架构合规（15 分）

| 检查项 | 分值 | 得分 | 状态 | 说明 |
|--------|------|------|------|------|
| 2.1 TPipe/TQue 模式 | 3 | 0 | ❌ | 未使用 TPipe/TQue，使用 HardEvent 同步 |
| 2.2 入口属性正确 | 3 | 0 | ❌ | 使用 `__global__ __cube__` 而非 `__global__ __aicore__` |
| 2.3 定义顺序正确 | 3 | 3 | ✅ | Kernel 类定义在入口函数之前 |
| 2.4 内存管理配对 | 3 | 0 | ❌ | 使用静态地址分配，未使用 AllocTensor/FreeTensor |
| 2.5 数据流完整 | 3 | 0 | ❌ | 数据流不完整，缺少正确的流水线同步 |
| **小计** | **15** | **3** | | |

### 维度 3：编码规范（15 分）

| 检查项 | 分值 | 得分 | 状态 | 说明 |
|--------|------|------|------|------|
| 3.1 矢量 API | 4 | 4 | ✅ | 使用了正确的 API（LoadDataWithSparse、MmadWithSparse） |
| 3.2 API 约束满足 | 4 | 0 | ❌ | 硬编码 baseM=128, baseN=128, baseK=64 |
| 3.3 数据对齐 | 4 | 4 | ✅ | 数据大小计算考虑了对齐 |
| 3.4 命名规范 | 3 | 3 | ✅ | 命名符合规范 |
| **小计** | **15** | **11** | | |

### 维度 4：性能优化（20 分）

| 检查项 | 分值 | 得分 | 状态 | 说明 |
|--------|------|------|------|------|
| 4.1 动态硬件参数 | 4 | 0 | ❌ | 硬编码 maxCoreNum=20, baseM/N/K |
| 4.2 多核并行 | 4 | 2 | ⚠️ | 实现了 M×N 切分，但核数计算逻辑有问题 |
| 4.3 流水线/双缓冲 | 4 | 0 | ❌ | 未使用双缓冲，SetFlag/WaitFlag 紧挨着使用 |
| 4.4 同步策略 | 4 | 0 | ❌ | 同步策略错误（详见依赖分析） |
| 4.5 计算效率与上板性能 | 4 | 0 | ❌ | 输出 nan，无法验证性能 |
| **小计** | **20** | **2** | | |

### 维度 5：测试覆盖（15 分）

| 检查项 | 分值 | 得分 | 状态 | 说明 |
|--------|------|------|------|------|
| 5.1 测试数据生成 | 4 | 2 | ⚠️ | gen_data.py 存在，但 golden 计算逻辑有问题 |
| 5.2 结果验证脚本 | 4 | 4 | ✅ | verify_result.py 存在且功能完整 |
| 5.3 Level 0 覆盖 | 4 | 0 | ❌ | 测试失败（输出 nan） |
| 5.4 精度标准明确 | 3 | 3 | ✅ | 明确了 atol/rtol 标准 |
| **小计** | **15** | **9** | | |

### 维度 6：精度验证（10 分）

| 检查项 | 分值 | 得分 | 状态 | 说明 |
|--------|------|------|------|------|
| 6.1 FP32 全用例 PASS | 4 | 0 | ❌ | 输出 nan，精度验证失败 |
| 6.2 FP16 全用例 PASS | 3 | 0 | ❌ | 不适用（当前仅支持 int8→float32） |
| 6.3 BF16 全用例 PASS | 3 | 0 | ❌ | 不适用（当前仅支持 int8→float32） |
| **小计** | **10** | **0** | | |

### 维度 7：文档（15 分）

| 检查项 | 分值 | 得分 | 状态 | 说明 |
|--------|------|------|------|------|
| 7.1 README.md 存在 | 3 | 3 | ✅ | README.md 存在 |
| 7.2 数学公式 | 3 | 3 | ✅ | 包含数学公式 C = A * B_sparse^T |
| 7.3 编译运行指南 | 3 | 3 | ✅ | 包含编译和运行指南 |
| 7.4 API 映射/约束 | 3 | 3 | ✅ | DESIGN.md 中包含 API 映射表 |
| 7.5 已知限制 | 3 | 3 | ✅ | README.md 中包含注意事项 |
| **小计** | **15** | **15** | | |

---

## 3. 必须修复问题清单

### 🔴 问题 1：CMake 配置缺少 tiling_api 库链接

**严重级别**: 阻塞  
**位置**: `CMakeLists.txt`  
**问题**: CMake 配置验证失败，缺少 `target_link_libraries(... PRIVATE tiling_api)`  
**影响**: 无法编译通过  
**修复建议**:
```cmake
# 在 CMakeLists.txt 末尾添加
target_link_libraries(sparse_gemm PRIVATE tiling_api register platform m dl)
```

### 🔴 问题 2：入口属性错误

**严重级别**: 阻塞  
**位置**: `kernel/sparse_gemm_kernel.asc:240`  
**问题**: 使用 `__global__ __cube__` 而非 `__global__ __aicore__`  
**影响**: 不符合 Ascend C 编程规范  
**修复建议**:
```cpp
// 修改前
extern "C" __global__ __cube__ void sparse_gemm_custom(...)

// 修改后
extern "C" __global__ __aicore__ void sparse_gemm_custom(...)
```

### 🔴 问题 3：未使用 TPipe/TQue 模式

**严重级别**: 高  
**位置**: `kernel/sparse_gemm_kernel.asc`  
**问题**: 代码中声明了 `TPipe pipe` 但未使用，而是使用 HardEvent 同步  
**影响**: 不符合架构规范，无法实现流水线并行  
**修复建议**:
1. 使用 TQue 管理数据搬运和计算
2. 使用 EnQue/DeQue 替代 HardEvent
3. 参考 `asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/` 示例

### 🔴 问题 4：硬编码硬件参数

**严重级别**: 阻塞  
**位置**: `kernel/sparse_gemm_kernel.asc:22-25`, `host/sparse_gemm.asc:41-43,56`  
**问题**: 硬编码了以下参数：
- `CUBE_BLOCK = 16`
- `C0_SIZE = 32`
- `baseM = 128`
- `baseN = 128`
- `baseK = 64`
- `maxCoreNum = 20`  
**影响**: 无法适配不同硬件，违反动态获取原则  
**修复建议**:
1. 从 Tiling 数据动态获取 baseM/baseN/baseK
2. 使用 `GetBlockNum()` 动态获取核数
3. 参考 `asc-devkit/examples/` 中的动态参数获取方式

### 🔴 问题 5：同步策略错误

**严重级别**: 阻塞  
**位置**: `kernel/sparse_gemm_kernel.asc:116-134`  
**问题**: SetFlag/WaitFlag 紧挨着使用，没有实际的流水线效果：
```cpp
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);  // 立即等待，无意义
```  
**影响**: 无法实现搬运/计算重叠，性能严重下降  
**修复建议**:
1. 使用 TQue + EnQue/DeQue 实现流水线
2. 或使用双缓冲 + 正确的同步点
3. 参考审查手册中的同步策略最佳实践

### 🔴 问题 6：gen_data.py golden 计算逻辑错误

**严重级别**: 阻塞  
**位置**: `scripts/gen_data.py:51-81`  
**问题**: golden 计算逻辑简化处理，没有正确处理稀疏格式：
```python
# 错误：直接截取 A 的前 k/2 列
a_batch_sliced = a_batch[:, :k//2]  # (m, k/2)
```  
**影响**: golden 数据不正确，无法验证精度  
**修复建议**:
1. 根据索引矩阵恢复完整的 B 矩阵
2. 正确实现稀疏矩阵乘法的 golden 计算
3. 参考 2:4 稀疏格式的数学定义

### 🔴 问题 7：输出 nan 值

**严重级别**: 阻塞  
**位置**: `kernel/sparse_gemm_kernel.asc`  
**问题**: PLAN.md 中记录输出 nan 值，可能原因：
- 数据搬运错误
- 计算参数错误
- 内存越界
- 类型转换错误  
**影响**: 算子功能完全不可用  
**修复建议**:
1. 使用 CPU_DEBUG 模式逐步调试
2. 检查数据搬运的偏移和大小计算
3. 验证 MmadWithSparse 的参数是否正确

### 🔴 问题 8：静态地址分配

**严重级别**: 高  
**位置**: `kernel/sparse_gemm_kernel.asc:211-216`  
**问题**: 使用静态地址分配：
```cpp
static constexpr uint32_t a1Addr = 0;
static constexpr uint32_t a2Addr = 0;  // A1 和 A2 共享同一块 L1
static constexpr uint32_t b1Addr = 128 * 1024;
static constexpr uint32_t b2Addr = 0;   // B1 和 B2 共享同一块 L1
static constexpr uint32_t idxB1Addr = 256 * 1024;
static constexpr uint32_t cAddr = 0;    // CO1 地址
```  
**影响**: 内存管理不灵活，可能导致冲突  
**修复建议**:
1. 使用 TPipe/TQue 管理内存
2. 使用 AllocTensor/FreeTensor 动态分配
3. 参考内存管理最佳实践

---

## 4. 建议修复问题清单

### 🟡 建议 1：添加 L2 cache bypass 优化

**位置**: `kernel/sparse_gemm_kernel.asc`  
**建议**: 对于大数据量搬运，考虑使用 L2 cache bypass 优化  
**参考**: `asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/`

### 🟡 建议 2：完善尾块处理逻辑

**位置**: `kernel/sparse_gemm_kernel.asc`  
**建议**: 当前尾块处理逻辑不完整，需要添加 SetTail 处理  
**参考**: DESIGN.md 中的尾块分支设计

### 🟡 建议 3：添加更多测试用例

**位置**: `scripts/gen_data.py`  
**建议**: 当前仅有 T1 用例，建议添加 T2（全零稀疏矩阵）和 T3（单元素稀疏矩阵）用例  
**参考**: PLAN.md 中的测试计划

---

## 5. 同步策略逐项依赖分析

| 行号 | 前操作 | 前 Pipe | 后操作 | 后 Pipe | 依赖类型 | 判定 |
|------|--------|---------|--------|---------|---------|------|
| 116-117 | DataCopy (MTE2) | MTE2 | LoadData (MTE1) | MTE1 | MTE2→MTE1 跨 pipe | 必要，但应使用 TQue |
| 126-127 | LoadData (MTE1) | MTE1 | MmadWithSparse (M) | M | MTE1→M 跨 pipe | 必要，但应使用 TQue |
| 133-134 | MmadWithSparse (M) | M | Fixpipe (MTE3) | MTE3 | M→MTE3 跨 pipe | 必要，但应使用 TQue |

**冗余率**: 0%（所有同步都是必要的）  
**问题**: 同步方式不正确，应使用 TQue/EnQue/DeQue 替代 HardEvent

---

## 6. 设计合规检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 技术路线一致 | ⚠️ | DESIGN.md 选择基础 API 路线，代码基本一致 |
| API 调用正确 | ⚠️ | API 选择正确，但参数传递有问题 |
| 数据流一致 | ❌ | 代码实现与设计文档的数据流不完全一致 |
| Buffer 规划一致 | ❌ | 代码使用静态地址，未按设计文档的 Buffer 规划实现 |

---

## 7. 测试覆盖评估

| 测试级别 | 状态 | 说明 |
|---------|------|------|
| Level 0 (8-16 元素) | ❌ | 未实现，当前最小测试为 4096×4096 |
| Level 1 (1K 元素) | ❌ | 未实现 |
| Level 2 (极值/边界) | ❌ | 未实现 |
| Level 3 (大数据量) | ⚠️ | 有大数据量测试，但输出 nan |

---

## 8. 精度验证结果

**独立运行结果**: 未执行（编译失败）  
**Developer 自报结果**: PLAN.md 记录输出 nan  
**判定**: ❌ 精度验证失败

---

## 9. 硬件参数检查

```bash
# Grep 检查结果
grep -n "blockDim\s*=\s*[0-9]" operators/sparse_gemm/**/*.asc → 未发现
grep -n "blockIdx\s*=\s*[0-9]" operators/sparse_gemm/**/*.asc → 未发现
```

**判定**: ✅ 通过（无硬编码核数/核索引）

---

## 10. 修复优先级建议

### P0（阻塞，必须立即修复）

1. **CMake 配置**：添加 tiling_api 库链接
2. **入口属性**：修改为 `__global__ __aicore__`
3. **输出 nan**：调试并修复数据搬运/计算逻辑

### P1（高优先级）

4. **硬编码参数**：改为动态获取
5. **同步策略**：使用 TQue/EnQue/DeQue
6. **内存管理**：使用 TPipe 管理内存

### P2（中优先级）

7. **golden 计算**：修复稀疏格式处理逻辑
8. **测试覆盖**：添加 Level 0-2 测试用例

---

## 11. 参考资源

| 资源 | 路径 | 用途 |
|------|------|------|
| 官方示例 | `asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/` | API 使用参考 |
| 审查手册 | `workflows/references/review-checklist.md` | 代码审查标准 |
| 精度调试 | `/ascendc-precision-debug` Skill | 精度问题诊断 |
| API 最佳实践 | `/ascendc-api-best-practices` Skill | API 使用规范 |

---

## 12. 总结

本次审查发现 **8 个必须修复问题** 和 **3 个建议修复问题**。主要问题集中在：

1. **编译配置错误**：CMake 缺少必要的库链接
2. **架构合规性**：未使用 TPipe/TQue 模式，入口属性错误
3. **硬编码问题**：多个硬件参数写死，无法适配不同环境
4. **同步策略**：SetFlag/WaitFlag 使用方式错误，无法实现流水线
5. **精度问题**：输出 nan，算子功能完全不可用

**建议**：Developer 需要参考官方示例 `mmad_with_sparse` 重新实现算子，重点关注：
1. TPipe/TQue 的正确使用
2. 动态硬件参数获取
3. 正确的同步策略
4. 稀疏格式的正确处理

---

**审查人**: CANNBot Reviewer  
**审查时间**: 2026-05-22  
**下次审查**: 待 Developer 修复后提交
