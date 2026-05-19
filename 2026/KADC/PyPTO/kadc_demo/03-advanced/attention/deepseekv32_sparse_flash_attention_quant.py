#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3/V2 Sparse Flash Attention (量化版) 端到端测试

本文件是 DeepSeek MLA 稀疏注意力算子的集成测试入口，职责包括：
  1. 生成测试数据 (golden data)：构造 Query / KV Cache / topk_indices / block_table 等输入，
     支持 INT8 量化和 BF16 两种 Key 格式。
  2. 计算 golden 参考输出：使用纯 PyTorch 实现标准 softmax 和 Flash Attention 两种算法。
  3. 调用 NPU 端 PyPTO kernel 并与 golden 对比验证精度。

支持两种推理模式：
  - Decode (s1=1/2)：标准 softmax，对应 kernel sparse_flash_attention_quant_d / _d_950
  - Prefill (s1=256)：Flash online softmax，对应 kernel sparse_flash_attention_quant_p

支持两种芯片：
  - Ascend 910B
  - Ascend 950（TileShape 不同）

依赖文件：
  - sparse_flash_attention_quant_impl.py : PyPTO kernel 实现（JIT 编译）
  - utils/compare.py                     : 精度对比工具（atol/rtol + NaN/Inf 检测）
"""

# ---- 标准库 ----
import os
import math
import logging
from dataclasses import dataclass

# ---- PyTorch & NPU ----
import torch
import torch_npu

# ---- 测试框架 ----
import pytest
import numpy as np

# ---- PyPTO 算子框架 ----
import pypto

# ---- 本算子实现与工具 ----
from sparse_flash_attention_quant_impl \
    import sparse_flash_attention_quant_d, sparse_flash_attention_quant_p,\
           sparse_flash_attention_quant_d_950, SaTileShapeConfig
from utils.compare import compare


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成，与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def compute_attention(input_data, params, s2_tile):
    """
    Flash Attention golden 参考实现（PyTorch）
    使用 online softmax 算法，跨 s2 tile 增量更新 oi/li/mi 三个运行状态，
    仅在最后一个 s2 tile 时做最终归一化，减少中间精度损失。
    对应 PyPTO kernel 中的 sparse_flash_attention_quant_compute_flash（prefill 模式）。

    计算流程（每个 s2 tile）：
        C1: S_ij = Q × K^T           (FP32 matmul)
        V1: online softmax            (exp(S - max), 累积 oi/li/mi)
        C2: O_partial = softmax × V   (FP32 matmul, 在 V1 内完成)
    """
    q, kn, kr, kn_scales, topk_indices, block_table, actual_seq = input_data
    block_size, scalar, topk, d_v, is_kn_quant = params

    # 维度: q=(B,S1,N_Q,D_Q), kn=(total_kv,D_K), kr=(total_kv,D_V)
    # D_Q = kv_lora_rank + qk_rope_dim = 576, D_K = kv_lora_rank = 512, D_V = qk_rope_dim = 64
    b, s1, n1, dq = q.shape
    _, dk = kn.shape
    _, dv = kr.shape

    # topk_indices: (B*S1, topk) — 每个 query token 的 top-k 索引
    if topk_indices.ndim > 2:
        topk_indices = topk_indices.reshape(b * s1, topk)

    atten_out_shape = [b, s1, n1, d_v]
    input_dtype = q.dtype
    kn_dtype = kn.dtype

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([b, s1, n1], dtype=input_dtype)

    # ---- LOOP_L0: batch 维度 ----
    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        # ---- LOOP_L1: s1 (query seq) 维度 ----
        for s1_idx in range(s1):
            # 当前 token 实际可关注的 KV 数量（因果 mask + topk 截断）
            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
            # s2 tile 数量（向上取整）
            bn_per_batch = math.ceil(cur_seq / s2_tile)

            # qi: 当前 (b, s1) 的 query, shape=(N_Q, D_Q)
            qi = q[b_idx, s1_idx, :, :] # (n1, dk)

            # ---- LOOP_L4: s2 (KV seq tile) 维度 ----
            for s2_idx in range(bn_per_batch):
                # 最后一个 tile 可能不足 s2_tile
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                s2_end = s2_start + s2_tile_cur

                # 取出当前 tile 的 topk_indices
                topk_indices_tmp = topk_indices[b_idx * s1 + s1_idx, s2_start:s2_end]

                # 为当前 tile 的 KV 分配临时空间
                slc_kn = torch.zeros([s2_tile_cur, dk], dtype=kn_dtype)
                slc_kr = torch.zeros([s2_tile_cur, dv], dtype=input_dtype)
                slc_kn_scales = torch.zeros([s2_tile_cur, 4], dtype=torch.float32)

                # ---- Gather 阶段: topk_index → KV Cache 物理偏移 ----
                # PagedAttention 映射: topk_index → block_idx → block_table[b, block_idx] → 物理块号
                # 物理偏移 = 物理块号 * block_size + 块内偏移
                offset = torch.zeros([s2_tile_cur], dtype=torch.int32)
                for cur_s2_idx in range(s2_tile_cur):
                    s2_idx_tmp = s2_start + cur_s2_idx
                    topk_index = topk_indices_tmp[s2_idx_tmp]
                    block_idx_in_batch = topk_index // block_size      # 逻辑块索引
                    slc_block_idx = block_table[b_idx, block_idx_in_batch]  # 物理块号
                    tail = topk_index % block_size                     # 块内偏移
                    offset[cur_s2_idx] = slc_block_idx * block_size + tail  # 全局物理偏移

                # ---- 按 offset 从 2D KV Cache 中 Gather 出对应的 kn/kr/scales ----
                for cur_s2_idx in range(s2_tile_cur):
                    slc_idx = offset[cur_s2_idx]
                    slc_kn[cur_s2_idx, :] = kn[slc_idx, :]
                    slc_kr[cur_s2_idx, :] = kr[slc_idx, :]
                    slc_kn_scales[cur_s2_idx, :] = kn_scales[slc_idx, :]

                # ---- 反量化: INT8 Key → BF16 ----
                # kn 按 128 元素一组量化，每组 1 个 scale，共 512/128=4 组
                # 反量化: kn_bf16 = int8_kn * scale (逐组)
                if is_kn_quant:
                    kn_bs = slc_kn.reshape(-1, 128).to(torch.float)       # (s2*4, 128) INT8→FP32
                    kn_scales_tmp = slc_kn_scales.reshape(-1, 1)           # (s2*4, 1) scale
                    kn_tmp = kn_bs * kn_scales_tmp                         # (s2*4, 128) 反量化
                    kn_tmp = kn_tmp.reshape(-1, 512).to(input_dtype)       # (s2, 512) → BF16
                else:
                    kn_tmp = slc_kn
                kr_tmp = slc_kr
                # V = Key 的 nope 部分 (kv_lora_rank=512)
                vj = kn_tmp

                # 拼接 Key: [nope(512) | rope(64)] = 576 维
                kj_view = torch.cat([kn_tmp, kr_tmp], dim=-1)

                # ---- C1: Q × K^T, shape=(N_Q, s2_tile_cur) ----
                sij = torch.matmul(qi.to(torch.float32), kj_view.transpose(1, 0).to(torch.float32)).to(torch.float32)

                # ---- V1: Online Softmax（Flash Attention 核心） ----
                sij_scale = sij * scalar                          # (n1, s2_tile) 乘以 1/sqrt(d_q)
                tilda_mij = sij_scale.amax(dim=-1, keepdims=True) # (n1, 1) 当前 tile 的行最大值
                t_sub = sij_scale - tilda_mij                     # (n1, s2_tile) 减去最大值防溢出
                tilda_pij = torch.exp(t_sub)                      # (n1, s2_tile) exp(S - max)
                tilda_pij_f16 = tilda_pij.to(input_dtype)         # 转 BF16 用于 C2 matmul
                # ---- C2: softmax_exp × V, shape=(N_Q, D_V) ----
                q1 = torch.matmul(tilda_pij_f16.to(torch.float32), vj.to(torch.float32)).to(torch.float32)
                tilda_lij = tilda_pij.sum(dim=-1, keepdims=True)  # (n1, 1) 当前 tile 的 exp 之和

                # ---- Flash Attention 增量更新 ----
                # 首个 s2 tile: 直接初始化 oi/li/mi
                if s2_idx == 0:
                    oi_tmp = q1
                    # 若只有一个 tile，直接归一化
                    if bn_per_batch == 1:
                        oi_update = oi_tmp / tilda_lij
                    else:
                        oi_update = oi_tmp
                    li_update = tilda_lij   # 累积 exp sum
                    mi_update = tilda_mij   # 累积 max
                    tmp_out[b_idx, s1_idx, :] = tilda_lij.reshape(n1)
                    continue

                # 后续 tile: 用 online softmax 公式修正历史累加值
                oi = oi_update     # 历史 O（未归一化）
                li = li_update     # 历史累积 exp sum, shape=(N_Q, 1)
                mi = mi_update     # 历史累积 max, shape=(N_Q, 1)

                # 新 max = max(历史 max, 当前 tile max)
                mi_new = torch.maximum(mi, tilda_mij)
                # 历史值的修正因子: exp(旧max - 新max)
                t1 = mi - mi_new
                t2 = torch.exp(t1)           # 历史指数修正系数
                # 当前 tile 的修正因子: exp(当前max - 新max)
                t3 = tilda_mij - mi_new
                t4 = torch.exp(t3)           # 当前 tile 指数修正系数
                # 更新累积 exp sum: li_new = exp(旧max-新max)*li + exp(当前max-新max)*当前sum
                t5 = t4 * tilda_lij
                t6 = t2 * li
                li_new = t6 + t5
                # 更新 O: oi_new = exp(旧max-新max)*旧oi + exp(当前max-新max)*当前q
                q3 = oi * t2                 # 修正历史 O
                q2 = q1 * t4                 # 修正当前 O_partial
                oi_tmp = q3 + q2
                # 最后一个 tile 时归一化: O_final = oi / li
                if s2_idx == bn_per_batch - 1:
                    oi_update = oi_tmp / li_new
                else:
                    oi_update = oi_tmp
                li_update = li_new
                mi_update = mi_new

            attention_output[b_idx, s1_idx, :, :] = oi_update.to(input_dtype)

    return attention_output, tmp_out


def compute_attention_no_flash(input_data, params, s2_tile):
    """
    标准 Softmax golden 参考实现（PyTorch）
    每个 s2 tile 独立做完整 softmax 归一化后直接得到 O。
    不使用 online softmax 增量更新，适合 s1 较小的 decode 场景。
    对应 PyPTO kernel 中的 sparse_flash_attention_quant_compute（decode 模式）。

    注意: 此实现仅保留最后一个 s2 tile 的计算结果作为输出，
          因为 s2 循环内 atten_out_part 会被覆盖（适合 s1=1/2 的 decode 场景）。

    计算流程（每个 s2 tile）：
        C1: S_ij = Q × K^T           (FP32 matmul)
        V1: 标准 softmax              (exp - max / sum)
        C2: O = softmax × V           (FP32 matmul)
    """
    q, kn, kr, kn_scales, topk_indices, block_table, actual_seq = input_data
    block_size, scalar, topk, d_v, is_kn_quant = params

    # 维度: q=(B,S1,N_Q,D_Q), kn=(total_kv,D_K=512), kr=(total_kv,D_V=64)
    b, s1, n1, dq = q.shape
    _, dk = kn.shape
    _, dv = kr.shape

    # topk_indices: (B*S1, topk)
    if topk_indices.ndim > 2:
        topk_indices = topk_indices.reshape(b * s1, topk)

    atten_out_shape = [b, s1, n1, d_v]
    input_dtype = q.dtype
    kn_dtype = kn.dtype

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)
    tmp_out = torch.zeros([b, s1, n1], dtype=input_dtype)

    # ---- LOOP_L0: batch 维度 ----
    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        # ---- LOOP_L1: s1 (query seq) 维度 ----
        for s1_idx in range(s1):
            # 当前 token 实际可关注的 KV 数量（因果 mask + topk 截断）
            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)
            # s2 tile 数量（向上取整）
            bn_per_batch = math.ceil(cur_seq / s2_tile)

            # qi: 当前 (b, s1) 的 query, shape=(N_Q, D_Q)
            qi = q[b_idx, s1_idx, :, :] # (n1, dk)

            # ---- LOOP_L4: s2 (KV seq tile) 维度 ----
            for s2_idx in range(bn_per_batch):
                # 最后一个 tile 可能不足 s2_tile
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                s2_end = s2_start + s2_tile_cur

                # 取出当前 tile 的 topk_indices
                topk_indices_tmp = topk_indices[b_idx * s1 + s1_idx, s2_start:s2_end]

                # 为当前 tile 的 KV 分配临时空间
                slc_kn = torch.zeros([s2_tile_cur, dk], dtype=kn_dtype)
                slc_kr = torch.zeros([s2_tile_cur, dv], dtype=input_dtype)
                slc_kn_scales = torch.zeros([s2_tile_cur, 4], dtype=torch.float32)

                # ---- Gather 阶段: topk_index → KV Cache 物理偏移 ----
                # PagedAttention 映射: topk_index → block_idx → block_table[b, block_idx] → 物理块号
                # 物理偏移 = 物理块号 * block_size + 块内偏移
                offset = torch.zeros([s2_tile_cur], dtype=torch.int32)
                for cur_s2_idx in range(s2_tile_cur):
                    s2_idx_tmp = s2_start + cur_s2_idx
                    topk_index = topk_indices_tmp[s2_idx_tmp]
                    block_idx_in_batch = topk_index // block_size      # 逻辑块索引
                    slc_block_idx = block_table[b_idx, block_idx_in_batch]  # 物理块号
                    tail = topk_index % block_size                     # 块内偏移
                    offset[cur_s2_idx] = slc_block_idx * block_size + tail  # 全局物理偏移

                # ---- 按 offset 从 2D KV Cache 中 Gather 出对应的 kn/kr/scales ----
                for cur_s2_idx in range(s2_tile_cur):
                    slc_idx = offset[cur_s2_idx]
                    slc_kn[cur_s2_idx, :] = kn[slc_idx, :]
                    slc_kr[cur_s2_idx, :] = kr[slc_idx, :]
                    slc_kn_scales[cur_s2_idx, :] = kn_scales[slc_idx, :]

                # ---- 反量化: INT8 Key → BF16 ----
                # kn 按 128 元素一组量化，每组 1 个 scale，共 512/128=4 组
                # 反量化: kn_bf16 = int8_kn * scale (逐组)
                if is_kn_quant:
                    kn_bs = slc_kn.reshape(-1, 128).to(torch.float)       # (s2*4, 128) INT8→FP32
                    kn_scales_tmp = slc_kn_scales.reshape(-1, 1)           # (s2*4, 1) scale
                    kn_tmp = kn_bs * kn_scales_tmp                         # (s2*4, 128) 反量化
                    kn_tmp = kn_tmp.reshape(-1, 512).to(input_dtype)       # (s2, 512) → BF16
                else:
                    kn_tmp = slc_kn
                kr_tmp = slc_kr
                # V = Key 的 nope 部分 (kv_lora_rank=512)
                vj = kn_tmp

                # 拼接 Key: [nope(512) | rope(64)] = 576 维
                kj_view = torch.cat([kn_tmp, kr_tmp], dim=-1)

                # ---- C1: Q × K^T, shape=(N_Q, s2_tile_cur) ----
                sij = torch.matmul(qi.to(torch.float32), kj_view.transpose(1, 0).to(torch.float32)).to(torch.float32)

                # ---- V1: 标准 Softmax ----
                sij_scale = sij * scalar                          # (n1, s2_tile) 乘以 1/sqrt(d_q)
                tilda_mij = sij_scale.amax(dim=-1, keepdims=True) # (n1, 1) 行最大值（防溢出）
                t_sub = sij_scale - tilda_mij                     # (n1, s2_tile) 减最大值
                tilda_pij = torch.exp(t_sub)                      # (n1, s2_tile) exp(S - max)
                tilda_lij = tilda_pij.sum(dim=-1, keepdims=True)  # (n1, 1) sum of exp
                # softmax = exp / sum, 转回 BF16
                tmp_softmax = (tilda_pij / tilda_lij).to(input_dtype)
                # ---- C2: softmax × V, shape=(N_Q, D_V) ----
                atten_out_part = torch.matmul(tmp_softmax.to(torch.float32), vj.to(torch.float32)).to(torch.float32)

            # 注意: decode 场景 s1=1 或 s1=2, bn_per_batch 通常为 1,
            #       所以最后一个 s2 tile 的结果即为最终输出
            attention_output[b_idx, s1_idx, :, :] = atten_out_part.to(input_dtype)

    return attention_output, tmp_out


def gen_block_table(act_seq, block_size, s1, need_indices=False):
    """
    生成 PagedAttention 的 block_table，模拟 KV Cache 的不连续内存布局。
    block_table[b, j] = 物理块号，表示 batch b 的第 j 个逻辑块映射到哪个物理块。
    物理块号随机排列以模拟真实场景。

    Args:
        act_seq: 各 batch 的实际 KV 序列长度, shape=(B,)
        block_size: 每个逻辑块的大小（默认 128）
        s1: query 序列长度（need_indices=True 时用于计算 cache_index）
        need_indices: 是否同时返回全局 cache_index（用于直接索引 KV Cache）

    Returns:
        block_num: 总物理块数（所有 batch 之和）
        block_table: shape=(B, max_blocknum_perbatch), dtype=INT32
        cache_index: shape=(B, S1) 或 None
    """
    block_num = 0
    block_num_each = []
    b = act_seq.shape[0]
    max_kv = max(act_seq)
    # 统计每个 batch 需要的块数
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    # block_table shape: (B, max_blocknum_perbatch)
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    # 生成随机排列的物理块号，模拟不连续内存布局
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    # 初始化为 -1（无效块），逐 batch 填入物理块号
    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_table_bidx = 0
    block_idx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    # 可选: 生成直接索引 KV Cache 的全局 offset
    if need_indices:
        cache_index = -torch.ones((b, s1), dtype=torch.int64)
        for i in range(b):
            cur_act = act_seq[i]
            for j in range(s1):
                pos = cur_act - s1 + j
                block_idx_in_seq = pos // block_size
                global_block_id = block_table[i, block_idx_in_seq]

                offset_in_block = pos % block_size
                global_index = global_block_id * block_size + offset_in_block
                cache_index[i, j] = global_index
    else:
        cache_index = None

    return block_num, block_table, cache_index


def gen_gather_select_attention_golden(dtype, bn1n2s1, is_kn_quant, actual_seq):
    """
    生成 DeepSeek V3/V2 Sparse Flash Attention 的完整测试数据与 golden 输出。

    MLA 结构参数:
        kv_lora_rank = 512   (Key/Value 低秩压缩维度, 即 nope 部分)
        qk_rope_dim  = 64    (旋转位置编码维度, 即 rope 部分)
        d_q = d_k = 576      (kv_lora_rank + qk_rope_dim)
        d_v = 512            (= kv_lora_rank)
        topk = 2048          (每个 token 选出的 KV 数量)
        block_size = 128     (PagedAttention block 大小)

    数据布局:
        q:   (B, S1, N_Q, 576)   [nope(512) | rope(64)]
        kn:  (total_kv, 512)      Key nope 部分, INT8 或 BF16
        kr:  (total_kv, 64)       Key rope 部分, BF16
        kn_scales: (total_kv, 4)  INT8 反量化 scale, FP32 (每 128 元素一组, 512/128=4)

    Args:
        dtype: 数据精度 (torch.bfloat16)
        bn1n2s1: (B, N_Q, N_KV, S1)
        is_kn_quant: 1=INT8 量化, 0=BF16
        actual_seq: 各 batch 实际 KV 序列长度

    Returns:
        input_params: [B, S1, N_Q, N_KV, max_kv_seq, kv_lora_rank, qk_rope_dim,
                       block_num, block_size, topk, is_kn_quant, scalar]
        input_data_map: [q_nope, q_rope, kn, kr, kn_scales, topk_indices,
                         block_table, actual_seq]
        atten_out: golden 注意力输出, shape=(B, S1, N_Q, kv_lora_rank)
    """
    block_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1  # e.g. (4, 128, 1, 2) decode / (1, 128, 1, 256) prefill
    kv_lora_rank = 512            # Key/Value 低秩压缩维度 (nope 部分)
    qk_rope_dim = 64              # 旋转位置编码维度 (rope 部分)
    topk = 2048                   # 每个 token 的 top-k KV 数量
    np.random.seed(None)
    # q head dim = kv_lora_rank + qk_rope_dim = 576
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim = kv_lora_rank + qk_rope_dim = 576
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim = kv_lora_rank = 512
    d_v = kv_lora_rank
    # softmax scale = 1/sqrt(d_q) = 1/sqrt(576)
    scalar = d_q ** -0.5
    if isinstance(actual_seq, int):
        actual_seq = [actual_seq] * b
    elif isinstance(actual_seq, list):
        if len(actual_seq) == b:
            actual_seq = actual_seq
        else:
            raise RuntimeError("unsupported actual_seq list length")
    else:
        raise RuntimeError("unsupported actual_seq data type")

    # ---- 1. 构造输入 shape ----
    shape_q = [b, s_q, n_q, d_q]  # (B, S1, N_Q, 576)

    # 统计所有 batch 的总 block 数
    block_num_per_batch = []
    block_num_min = 0
    block_num = 0
    for actual_seq_tmp in actual_seq:
        block_num_per_batch.append(math.ceil(actual_seq_tmp / block_size))
        block_num_min += math.ceil(actual_seq_tmp / block_size)
    block_num = block_num_min

    # kn: (block_num, block_size, kv_lora_rank) 即 (block_num, 128, 512)
    shape_kn = [block_num, block_size, kv_lora_rank]
    # kr: (block_num, block_size, qk_rope_dim) 即 (block_num, 128, 64)
    shape_kr = [block_num, block_size, qk_rope_dim]

    max_kv_seq = max(actual_seq)
    # 生成 PagedAttention block_table
    block_num, block_table, _ = gen_block_table(torch.tensor(actual_seq), block_size, s_q, need_indices=False)
    # topk_indices: (B, S1, topk) → 后续 reshape 为 (B*S1, N_KV*topk)
    topk_indices = torch.zeros(b, s_q, topk).to(torch.int32)
    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))

    # 生成 topk_indices: 序列长度 < topk 时顺序取, 否则随机排列取前 topk
    for b_i in range(b):
        for s_q_i in range(s_q):

            if slc_actual_seq[b_i] < topk:
                topk_indices[b_i, s_q_i, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indices[b_i, s_q_i, :] = perm[:topk]

    topk_indices = topk_indices.reshape(b * s_q, n_kv * topk)

    # ---- 2. 生成随机输入数据 ----
    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)            # (B, S1, N_Q, 576) BF16
    kn_bsnd_tmp = gen_uniform_data(shape_kn, -1, 1, dtype)      # (block_num, 128, 512) BF16

    # INT8 量化: 按 128 元素分组求 amax → scale = amax/127 → quant = round(val/scale).clamp(-128,127)
    kn_bsnd_reshape = kn_bsnd_tmp.reshape(block_num * block_size, 4, 128).to(torch.float32)
    kn_scales = kn_bsnd_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    if is_kn_quant == 1:
        kn_quant = kn_bsnd_tmp.reshape(block_num * block_size, 4, 128) / kn_scales
        kn = torch.round(kn_quant).clamp(-128, 127).to(torch.int8)  # (total_kv*4, 128) INT8
    else:
        kn = kn_bsnd_tmp                                                # (block_num, 128, 512) BF16
    kr = gen_uniform_data(shape_kr, -1, 1, dtype)               # (block_num, 128, 64) BF16

    # 展平为 2D: (total_kv, dim)
    kn = kn.reshape(block_num * block_size, kv_lora_rank)       # (total_kv, 512)
    kn_scales = kn_scales.reshape(block_num * block_size, 4)    # (total_kv, 4)
    kr = kr.reshape(block_num * block_size, qk_rope_dim)        # (total_kv, 64)

    # ---- 3. 计算 golden attention（标准 softmax 版本） ----
    params = [block_size, scalar, topk, kv_lora_rank, is_kn_quant]
    input_data = [q_bsnd, kn, kr, kn_scales, topk_indices, block_table, actual_seq]

    s2_tile = 2048
    atten_out, tmp_out = compute_attention_no_flash(input_data, params, s2_tile)

    # ---- 4. 拆分 Query 为 nope + rope, 与 PyPTO kernel 输入格式对齐 ----
    # q_nope: (B*S1*N_Q, 512), q_rope: (B*S1*N_Q, 64)
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope = q_nope.reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_rope.reshape(b * s_q * n_q, qk_rope_dim)
    # input params
    input_params = [b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk,
                    is_kn_quant, scalar]
    input_data_map = [q_nope, q_rope, kn, kr, kn_scales, topk_indices, block_table, actual_seq]

    return input_params, input_data_map, atten_out


def do_test_sparse_attention_func(bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p, is_soc_950):
    """
    执行 NPU 端 sparse flash attention 并与 golden 对比。

    根据 (is_p, is_soc_950) 选择不同的 kernel 入口和 TileShape 配置:
        - is_p=True,  is_soc_950=False → sparse_flash_attention_quant_p   (910B Prefill)
        - is_p=False, is_soc_950=False → sparse_flash_attention_quant_d   (910B Decode)
        - is_p=False, is_soc_950=True  → sparse_flash_attention_quant_d_950 (950 Decode)
    """
    b, n1, n2, s1 = bn1n2s1

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # ---- TileShape 配置: 控制各级算子的 tile 切分 ----
    # g_tile: GQA group 维度 tile, 一次处理多少个 query head (group=N_Q/N_KV=128)
    # s_kv_tile: KV 序列维度 tile, 一次处理多少个 KV token
    # gather_vec_tile_shape: [行, 列] Gather 向量算子 tile (从 KV Cache 搬运数据)
    # c1_tile_shape: [M0,M1, K0,K1, N0,N1] C1(Q×K^T) Cube matmul tile
    # v1_tile_shape: [行, 列] V1(Softmax) 向量算子 tile
    # c2_tile_shape: [M0,M1, K0,K1, N0,N1] C2(Softmax×V) Cube matmul tile
    # v2_tile_shape: [行, 列] V2(Flash 归一化更新) 向量算子 tile, 仅 prefill 使用

    if is_p:
        # 910B Prefill 配置 (s1=256)
        tile_config = SaTileShapeConfig(
            g_tile=128,                              # 一次处理 128 个 group
            s_kv_tile=2048,                          # 一次处理 2048 个 KV token
            gather_vec_tile_shape=[32, 512],         # Gather: 32行 × 512列
            c1_tile_shape=[128, 128, 128, 128, 128, 128],  # C1: M=128, K=128, N=128
            v1_tile_shape=[8, 2048],                  # Softmax: 8行 × 2048列
            c2_tile_shape=[128, 128, 128, 128, 128, 128],  # C2: M=128, K=128, N=128 (C1的N轴与C2的K轴一致)
            v2_tile_shape=[64, 128]                   # Flash 归一化更新: 64行 × 128列
        )
    else:
        # 910B Decode 配置 (s1=1 或 2)
        tile_config = SaTileShapeConfig(
            g_tile=128,                              # 一次处理 128 个 group
            s_kv_tile=2048,                          # 一次处理 2048 个 KV token
            gather_vec_tile_shape=[32, 512],         # Gather: 32行 × 512列
            c1_tile_shape=[128, 128, 128, 128, 128, 128],  # C1: M=128, K=128, N=128
            v1_tile_shape=[8, 2048],                  # Softmax: 8行 × 2048列
            c2_tile_shape=[128, 128, 128, 128, 128, 128],  # C2: M=128, K=128, N=128
            v2_tile_shape=[64, 256]                   # Flash 归一化更新: 64行 × 256列
        )
    
    if is_soc_950:
        # 950 Decode 配置 (适配 950 芯片的 UB/L1 容量)
        tile_config = SaTileShapeConfig(
            g_tile=128,                              # 一次处理 128 个 group
            s_kv_tile=2048,                          # 一次处理 2048 个 KV token
            gather_vec_tile_shape=[64, 512],         # Gather: 64行 × 512列 (950 UB 更大, 行翻倍)
            c1_tile_shape=[128, 128, 128, 128, 64, 64],    # C1: M=128, K=128, N=64 (950 N 轴减半)
            v1_tile_shape=[4, 2048],                  # Softmax: 4行 × 2048列 (行减半)
            c2_tile_shape=[128, 128, 128, 128, 128, 128],  # C2: M=128, K=128, N=128
            v2_tile_shape=[64, 256]                   # Flash 归一化更新: 64行 × 256列
        )

    b, s1, n_q, n_kv, max_kv_seq, kv_lora_rank, qk_rope_dim, block_num, block_size, topk, \
        is_kn_quant, softmax_scale = input_params
    q_nope, q_rope, kn, kr, kn_scales, topk_indices, block_table, kv_actual_seqs = input_data
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    # ---- 将所有输入 tensor 搬运到 NPU ----
    q_nope_npu = q_nope.npu()
    q_rope_npu = q_rope.npu()
    kn_npu = kn.npu()
    kr_npu = kr.npu()
    kn_scales_npu = kn_scales.npu()
    topk_indices_npu = topk_indices.npu()
    block_table_npu = block_table.npu()
    kv_act_seqs_npu = kv_act_seqs.npu()
    pto_inputs = [q_nope_npu, q_rope_npu, kn_npu, kr_npu, kn_scales_npu, topk_indices_npu, block_table_npu,
                  kv_act_seqs_npu]

    # 输出 tensor, shape=(B, S1, N_Q, kv_lora_rank), BF16
    calc_attention_out = torch.zeros([b, s1, n_q, kv_lora_rank], dtype=torch.bfloat16)
    calc_attention_out_npu = calc_attention_out.npu()
    pto_outputs = [calc_attention_out_npu]

    # max_blocknum_perbatch: 用于 block_table 的列数
    max_blocknum_perbatch = math.ceil(max_kv_seq / block_size)

    # ---- 根据 (is_p, is_soc_950) 选择对应的 JIT kernel ----
    if is_p and not is_soc_950:
        # 910B Prefill: Flash Attention (online softmax)
        sparse_flash_attention_quant_p(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size, \
            max_blocknum_perbatch, tile_config)
    elif not is_p and not is_soc_950:
        # 910B Decode: 标准 softmax
        sparse_flash_attention_quant_d(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size, \
            max_blocknum_perbatch, tile_config)
    else:
        # 950 Decode: 标准 softmax (适配 950 芯片)
        sparse_flash_attention_quant_d_950(*pto_inputs, *pto_outputs, n_q, n_kv, softmax_scale, topk, block_size, \
            max_blocknum_perbatch, tile_config)
    # 等待 NPU 计算完成
    torch_npu.npu.synchronize()
    # 精度对比: atol=0.0001, rtol=0.005, 最多报告 100 个误差点
    compare(calc_attention_out_npu.cpu(), atten_out, "atten_out", atol=0.0001, rtol=0.005, max_error_count=100)


def get_case_config(case_name: str):
    """
    测试用例参数配置。
    每个用例: ((B, N_Q, N_KV, S1), is_kn_quant, actual_seq, is_soc_950)

    命名规则: sfa_{dtype}_b{B}_s{S1}_seq{KV_LEN}_{quant}_{mode}
      - mode 后缀: d=decode, p=prefill
      - quant 后缀: int8=INT8 量化 Key, bf16=纯 BF16 Key
      - seq 后缀: total=变长序列, per=等长序列, 64K=65536
    """
    test_case_config = {
        # 变长序列 decode (INT8): B=4, S1=2, 各 batch 序列长度不同
        "sfa_bf16_b4_s2_seq64K_total_int8_d": (
            (4, 128, 1, 2), 1, [65536, 16381, 666, 15], 0
        ),
        # 等长序列 decode (INT8): B=4, S1=2, 所有 batch seq=65536
        "sfa_bf16_b4_s2_seq64K_per_int8_d": (
            (4, 128, 1, 2), 1, [65536] * 4, 0
        ),
        # 等长序列 decode (BF16): B=4, S1=2, Key 无量化
        "sfa_bf16_b4_s2_seq64K_per_bf16_d": (
            (4, 128, 1, 2), 0, [65536] * 4, 0
        ),
        # prefill (INT8): B=1, S1=256, 单 batch 长序列
        "sfa_bf16_b1_s256_seq64K_int8_p": (
            (1, 128, 1, 256), 1, [65536], 0
        ),
        # 950 decode (BF16): B=4, S1=2, 950 芯片适配
        "sfa_bf16_b4_s2_seq64K_per_int8_d_950": (
            (4, 128, 1, 2), 0, [65536] * 4, 1
        ),
    }
    case_config = test_case_config.get(case_name)
    return case_config


def do_test_sfa_entry(case_name: str, is_p: bool, is_soc_950: bool):
    """
    测试入口: 根据用例名获取参数 → 生成 golden → 执行 NPU 计算 → 精度对比。
    """
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    bn1n2s1, is_kn_quant, actual_seq, is_soc_950 = case_config

    # 生成 golden 数据与参考输出
    input_params, input_data, atten_out = gen_gather_select_attention_golden(
        torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq
    )
    # 执行 NPU kernel 并与 golden 对比
    do_test_sparse_attention_func(
        bn1n2s1, actual_seq, input_params, input_data, atten_out, is_p, is_soc_950
    )
    return True


@pytest.mark.soc("950", "910")
def test_sfa_bf16_b4_s2_seq64k_total_int8_d():
    """
    Decode 变长序列测试 (INT8 量化 Key)
    配置: B=4, S1=2, N_Q=128, N_KV=1, topk=2048
    序列长度: [65536, 16381, 666, 15] — 各 batch 差异大，验证变长处理逻辑
    芯片: 910B / 950
    模式: is_p=False (decode), is_soc_950=False
    """
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_total_int8_d", is_p=False, is_soc_950=False)


@pytest.mark.skip(reason="perf")
def test_sfa_bf16_b4_s2_seq64k_per_int8_d():
    """
    Decode 等长序列测试 (INT8 量化 Key)
    配置: B=4, S1=2, N_Q=128, N_KV=1, topk=2048
    序列长度: [65536] × 4 — 所有 batch 等长
    芯片: 910B / 950
    模式: is_p=False (decode), is_soc_950=False
    """
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_per_int8_d", is_p=False, is_soc_950=False)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="perf")
def test_sfa_bf16_b4_s2_seq64k_per_int8_d_950():
    """
    Decode 等长序列测试 — 950 芯片适配 (BF16 Key)
    配置: B=4, S1=2, N_Q=128, N_KV=1, topk=2048
    序列长度: [65536] × 4
    芯片: 950 (专用 TileShape: gather 行翻倍, C1 N 轴减半, V1 行减半)
    模式: is_p=False (decode), is_soc_950=True
    """
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_per_int8_d_950", is_p=False, is_soc_950=True)


@pytest.mark.skip(reason="bf16 perf")
def test_sfa_bf16_b4_s2_seq64k_per_bf16_d():
    """
    Decode 等长序列测试 (纯 BF16 Key, 无量化)
    配置: B=4, S1=2, N_Q=128, N_KV=1, topk=2048
    序列长度: [65536] × 4
    芯片: 910B / 950
    模式: is_p=False (decode), is_soc_950=False
    目的: 验证 Key 未量化时 kernel 的 BF16 路径（Gather → 直接使用，不经反量化）
    """
    do_test_sfa_entry("sfa_bf16_b4_s2_seq64K_per_bf16_d", is_p=False, is_soc_950=False)


@pytest.mark.skip(reason="large test case")
def test_sfa_bf16_b1_s256_seq64k_int8_p():
    """
    Prefill 测试 (INT8 量化 Key, Flash online softmax)
    配置: B=1, S1=256, N_Q=128, N_KV=1, topk=2048
    序列长度: [65536]
    芯片: 910B
    模式: is_p=True (prefill), is_soc_950=False
    特点: S1=256 较大，使用 Flash Attention 增量更新 oi/li/mi 跨 tile 归一化
    """
    do_test_sfa_entry("sfa_bf16_b1_s256_seq64K_int8_p", is_p=True, is_soc_950=False)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    # ---- 直接运行入口: 取消注释即可执行对应用例 ----
    # 当前激活: BF16 等长 decode (无量化, 数据量最小, 适合快速验证)
    # test_sfa_bf16_b4_s2_seq64k_total_int8_d()   # 变长 INT8 decode (默认 pytest 用例)
    # test_sfa_bf16_b4_s2_seq64k_per_int8_d()     # 等长 INT8 decode
    # test_sfa_bf16_b1_s256_seq64k_int8_p()       # prefill INT8 (S1=256, 耗时较长)
    # test_sfa_bf16_b4_s2_seq64k_per_int8_d_950() # 950 decode BF16
    test_sfa_bf16_b4_s2_seq64k_per_bf16_d()         # 等长 BF16 decode (无量化)
