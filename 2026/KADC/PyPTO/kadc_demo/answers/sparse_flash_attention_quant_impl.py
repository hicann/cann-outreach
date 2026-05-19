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
Sparse Flash Attention Quantization Module — PyPTO Kernel 实现

本模块实现了 DeepSeek V3/V2 MLA 稀疏注意力的 PyPTO kernel，支持：
  - KV Cache INT8 量化（Key nope 部分）与纯 BF16 两种路径
  - PagedAttention（block_table 映射不连续 KV Cache）
  - Top-K 稀疏选择 + Gather 索引
  - 两种 softmax 算法：标准 softmax（Decode）和 Flash online softmax（Prefill）

计算流水线（每个 s2 tile）：
  Sa_V0: Gather — 从 KV Cache 按 topk_indices 搬运 Key/Value 数据，INT8 路径含反量化
  Sa_C1: Cube   — Q × K^T (FP32 matmul)
  Sa_V1: Vector — Softmax（标准归一化 或 Flash 增量更新）
  Sa_C2: Cube   — Softmax × V (BF16 matmul)
  Sa_V2: Vector — Flash 归一化更新（仅 Prefill 模式）

入口函数（@pypto.frontend.jit 装饰，编译后运行于 NPU）：
  - sparse_flash_attention_quant_d     : 910B Decode（标准 softmax）
  - sparse_flash_attention_quant_d_950 : 950  Decode（标准 softmax）
  - sparse_flash_attention_quant_p     : 910B Prefill（Flash online softmax）

依赖：
  - pypto.experimental.gather_in_ub : 从 UB 中按索引 Gather 数据
  - pypto.experimental.gather_in_l1 : 从 L1 中按索引 Gather 数据（Cube 可直接消费）
"""
import os
import math
from dataclasses import dataclass
import numpy as np
import pypto
from pypto.experimental import gather_in_l1, gather_in_ub


@dataclass
class SaTileShapeConfig:
    """
    Sparse Attention 各级算子的 Tile 切分配置。

    默认参数值来自 deepseekv32_sparse_flash_attention_quant.py 中的测试用例配置，
    不同芯片/模式使用不同配置。

    MLA 结构参数（固定）:
        kv_lora_rank = 512 (Key nope / Value 维度, 即 dn)
        qk_rope_dim  = 64  (Key rope 维度, 即 dr)
        d_q = d_k = 576    (dn + dr = 512 + 64)
        topk = 2048        (每个 token 选出的 KV 数量)
        block_size = 128   (PagedAttention 块大小)

    Attributes:
        g_tile: GQA group 维度 tile 大小。
                一次处理的 query head 组数, group = N_Q / N_KV = 128。
                当前配置: 128 (即一次处理全部 128 个 group)
        s_kv_tile: KV 序列维度 tile 大小。
                每个 s2 tile 处理的 KV token 数量。
                当前配置: 2048
        gather_vec_tile_shape: [行, 列] Gather 向量算子 tile。
                控制从 KV Cache Gather 数据时的 UB 切分。
                当前配置: [32, 512] (910B) / [64, 512] (950)
        c1_tile_shape: [M0,M1, K0,K1, N0,N1] C1 (Q×K^T) Cube matmul tile。
                M=Q 侧 group tile, K=K 侧 s2 tile, N=Q 侧 group tile (因 b_trans)。
                当前配置: [128, 128, 128, 128, 128, 128] (910B)
                         [128, 128, 128, 128, 64, 64]   (950, N 轴减半)
        v1_tile_shape: [行, 列] V1 (Softmax) 向量算子 tile。
                控制 softmax 计算（exp/sub/div）的 UB 切分。
                当前配置: [8, 2048] (910B) / [4, 2048] (950)
        c2_tile_shape: [M0,M1, K0,K1, N0,N1] C2 (Softmax×V) Cube matmul tile。
                M=Q 侧 group tile, K=softmax 输出, N=Value 维度。
                注意: C2 的 K 轴必须与 C1 的 N 轴一致。
                当前配置: [128, 128, 128, 128, 128, 128]
        v2_tile_shape: [行, 列] V2 (Flash 归一化更新) 向量算子 tile。
                控制 Flash Attention oi/li/mi 增量更新的 UB 切分。
                仅 Prefill 模式使用。
                当前配置: [64, 128] (Prefill 910B) / [64, 256] (Decode)
    """
    g_tile: int                   # GQA group tile, 默认 128
    s_kv_tile: int                # KV 序列 tile, 默认 2048
    gather_vec_tile_shape: list   # Gather 向量算子 tile, 默认 [32, 512]
    c1_tile_shape: list           # C1 Cube tile [M0,M1,K0,K1,N0,N1], 默认 [128,128,128,128,128,128]
    v1_tile_shape: list           # V1 Softmax tile [行,列], 默认 [8, 2048]
    c2_tile_shape: list           # C2 Cube tile [M0,M1,K0,K1,N0,N1], 默认 [128,128,128,128,128,128]
    v2_tile_shape: list           # V2 Flash 归一化 tile [行,列], 默认 [64, 128]


def sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                         k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                         attention_out, nq, n_kv, softmax_scale, topk,
                                         block_size, max_blocknum_perbatch, tile_config):
    """标准 Softmax 模式 — Decode 阶段专用。

    每个 s2 tile 独立做完整 softmax 归一化后直接写入输出，不使用 Flash 增量更新。
    适用于 s1=1 或 s1=2 的 decode 场景（通常 bn_per_batch=1）。
    被 sparse_flash_attention_quant_d / _d_950 JIT 入口调用。

    计算流水线（每个 s2 tile）：
        Sa_V0: Gather Key (nope+rope) + INT8 反量化（如需）
        Sa_C1: S_ij = Q × K^T           (FP32 Cube matmul)
        Sa_V1: softmax(S * scale)        (标准归一化: exp-max / sum)
        Sa_C2: O = softmax × V           (BF16 Cube matmul)

    Args:
        query_nope:  (t*n_q, kv_lora_rank=512)  BF16, Query 低秩压缩部分
        query_rope:  (t*n_q, qk_rope_dim=64)    BF16, Query 旋转位置编码部分
        key_nope_2d: (total_kv, kv_lora_rank=512) INT8 或 BF16, Key nope 部分 (KV Cache)
        key_rope_2d: (total_kv, qk_rope_dim=64)   BF16, Key rope 部分 (KV Cache)
        k_nope_scales: (total_kv, 4)  FP32, INT8 反量化 scale (每 128 元素一组, 512/128=4)
        topk_indices:  (t, n_kv*topk)  INT32, 每个 query token 的 top-k 索引
        block_table:   (B, max_blocknum_perbatch)  INT32, PagedAttention 块映射表
        kv_act_seqs:   (B,)  INT32, 各 batch 实际 KV 序列长度
        attention_out: (B, S1, N_Q, kv_lora_rank=512)  BF16, 输出
        nq, n_kv: query / kv head 数量
        softmax_scale: 1/sqrt(d_q) = 1/sqrt(576)
        topk: 每个 token 选出的 KV 数量 (2048)
        block_size: PagedAttention 块大小 (128)
        max_blocknum_perbatch: block_table 列数
        tile_config: SaTileShapeConfig tiling 配置
    """
    dtype = query_nope.dtype            # BF16
    kn_dtype = key_nope_2d.dtype        # INT8 或 BF16
    dn = query_nope.shape[1]            # kv_lora_rank = 512
    dr = query_rope.shape[1]            # qk_rope_dim = 64
    group = nq // n_kv                  # GQA group 数, 128/1 = 128
    # 从 tile_config 提取各级 tile 参数
    gather_vec_tile = tile_config.gather_vec_tile_shape   # [32, 512]
    group_tile = tile_config.g_tile                       # 128
    s2_tile = tile_config.s_kv_tile                       # 2048
    c1_tile = tile_config.c1_tile_shape                   # [128,128,128,128,128,128]
    v1_tile = tile_config.v1_tile_shape                   # [8, 2048]
    c2_tile = tile_config.c2_tile_shape                   # [128,128,128,128,128,128]
    n_kv_sym = n_kv                                       # 1

    batch_size_sym = kv_act_seqs.shape[0]                 # B

    s1_n2_gsym = query_nope.shape[0] // batch_size_sym    # S1 * N_Q
    s1_sym = s1_n2_gsym // nq                             # S1

    g_loop_sym = group // group_tile                      # 128/128 = 1

    # 输出中间 tensor: 展平为 2D 方便按 offset 写入
    atten_out_2dim = pypto.tensor([batch_size_sym * s1_n2_gsym, dn], dtype, "attenOut2Dim")

    # ---- LOOP_L0: batch 维度 (可并行) ----
    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_L0_idx", idx_name="bIdx", parallel=True):
        cur_act_seq = kv_act_seqs[batch_idx]              # 当前 batch 的实际 KV 序列长度
        # ---- LOOP_L1: s1 (query seq) 维度 ----
        for slc_idx in pypto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
            # 当前 token 可关注的 KV 数量（因果 mask + topk 截断）
            cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
            cur_seq.as_variable()
            # s2 tile 数量（向上取整）
            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile

            # ---- LOOP_L2: n_kv (KV head) 维度 ----
            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                # ---- LOOP_L3: GQA group 维度 ----
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = group_tile            # 当前处理的 group 大小 (128)
                    # 当前 group 在展平 2D query 中的偏移: batch * S1*N_Q + s1*N_Q + n_kv*group + g*tile
                    cur_offset = batch_idx * s1_n2_gsym + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile

                    # ---- LOOP_L4: s2 (KV seq tile) 维度 (loop_unroll, unroll_list={1}) ----
                    for s2_idx, _ in pypto.loop_unroll(0, bn_per_batch, 1,
                        name="LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
                        cur_s2_tile = s2_tile              # 当前 s2 tile 大小 (2048)

                        # ---- Sa_V0: 准备 Gather 参数 ----
                        # 取出当前 (batch, s1) 的 topk_indices 切片: [1, s2_tile]
                        cur_topk_indices = pypto.view(topk_indices, [1, cur_s2_tile],
                                                [batch_idx * s1_sym + slc_idx, s2_idx * cur_s2_tile],
                                                valid_shape=[1, (cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile)])
                        # 取出当前 batch 的 block_table 切片: [1, max_blocknum_perbatch]
                        cur_block_table = pypto.view(block_table, [1, max_blocknum_perbatch], [batch_idx, 0])

                        kn = pypto.tensor([s2_tile, dn], dtype, "kn")

                        # ===== INT8 量化 Key 路径 =====
                        if kn_dtype == pypto.DT_INT8:
                            pypto.set_semantic_label("Sa_V0")
                            pypto.set_vec_tile_shapes(16, 1024)  # Vec: 16行 × 1024列, INT8 Gather+Cast 用较大列宽
                            # view scales: (total_kv, 8) 物理列比实际 4 列大，用于对齐
                            k_nope_scale_view = pypto.view(k_nope_scales, [k_nope_scales.shape[0], 8],
                                [0, 0], valid_shape=[k_nope_scales.shape[0], 4])
                            # Gather INT8 scales: 按 topk_indices + block_table 从 KV Cache 搬入 UB
                            kn_scale = gather_in_ub(k_nope_scale_view, cur_topk_indices, cur_block_table,
                                                    block_size, -2)
                            # view INT8 key_nope: (total_kv, 512)
                            k_nope_2d_view = pypto.view(key_nope_2d, [key_nope_2d.shape[0], dn],
                                [0, 0], valid_shape=[key_nope_2d.shape[0], dn])
                            # Gather INT8 quantized kn: 按 topk_indices + block_table 从 KV Cache 搬入 UB
                            kn_quant = gather_in_ub(k_nope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)

                            # 反量化: INT8 → FP16 → FP32 → × scale → BF16
                            kn_quant_fp16 = pypto.cast(kn_quant, pypto.DT_FP16)           # INT8 → FP16
                            kn_quant_fp32 = pypto.cast(kn_quant_fp16, pypto.DT_FP32)      # FP16 → FP32
                            # concat 扩展列: (s2*4, 128) → (s2*4, 256)，用于后续 reshape 对齐
                            kn_quant_fp32 = pypto.concat([kn_quant_fp32, kn_quant_fp32], -1)
                            kn_quant_fp32_tmp = pypto.reshape(kn_quant_fp32, [s2_tile * 8, 128])
                            kn_scale_tmp = pypto.reshape(kn_scale, [s2_tile * 8, 1])
                            # 逐元素乘: INT8_val * scale → FP32 反量化结果
                            pypto.set_vec_tile_shapes(128, 128)  # Vec: 128行 × 128列, INT8 × FP32 scale 逐元素乘
                            kn_fp32 = pypto.mul(kn_quant_fp32_tmp, kn_scale_tmp)
                            kn_fp32_reshape = pypto.reshape(kn_fp32, [s2_tile, dn * 2])
                            # view 取有效区域: (cur_s2_tile, dn)
                            pypto.set_vec_tile_shapes(16, 512)   # Vec: 16行 × 512列, view 反量化结果切有效区域
                            cur_kn_fp32 = pypto.view(kn_fp32_reshape, [cur_s2_tile, dn], [0, 0],
                                valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                            kn = pypto.cast(cur_kn_fp32, dtype)                           # FP32 → BF16

                            # ---- Sa_C1: Q × K^T ----
                            pypto.set_semantic_label("Sa_C1")
                            pypto.set_vec_tile_shapes(gather_vec_tile[0], gather_vec_tile[1])  # Vec: 32行 × 512列 (910B) / 64行 × 512列 (950)
                            pypto.set_cube_tile_shapes([c1_tile[0],
                                c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                            # Cube C1: M=[128,128], K=[128,128], N=[128,128] (910B) / N=[64,64] (950)
                            kr = gather_in_l1(key_rope_2d, cur_topk_indices, cur_block_table, block_size, dr,
                                            is_b_matrix=True, is_trans=True)
                            
                            # 拼接 Key: [nope(512) | rope(64)] = 576 维
                            kj = pypto.tensor([cur_s2_tile, dn + dr], dtype, "kj")
                            pypto.assemble(kn, [0, 0], kj)        # kj[:, 0:512] = kn
                            pypto.assemble(kr, [0, dn], kj)       # kj[:, 512:576] = kr
                            kj_view = pypto.view(kj, [cur_s2_tile, dn + dr], [0, 0],
                                valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn + dr])

                            # 拼接 Query: [nope(512) | rope(64)] = 576 维
                            qn = pypto.view(query_nope, [cur_group_tile, dn], [cur_offset, 0],
                                            valid_shape=[cur_group_tile, dn])
                            qr = pypto.view(query_rope, [cur_group_tile, dr], [cur_offset, 0],
                                            valid_shape=[cur_group_tile, dr])
                            qi = pypto.tensor([cur_group_tile, dn + dr], dtype, "qi")
                            pypto.assemble(qn, [0, 0], qi)        # qi[:, 0:512] = q_nope
                            pypto.assemble(qr, [0, dn], qi)       # qi[:, 512:576] = q_rope

                            # C1 matmul: S = Q × K^T, shape=(group_tile, s2_tile), FP32
                            sij = pypto.matmul(qi, kj_view, pypto.DT_FP32, a_trans=False, b_trans=True)

                        # ===== BF16 纯精度 Key 路径 =====
                        else:
                            pypto.set_semantic_label("Sa_V0")
                            pypto.set_vec_tile_shapes(gather_vec_tile[0], gather_vec_tile[1])  # Vec: 32行 × 512列 (910B) / 64行 × 512列 (950)
                            # view BF16 key_nope: (total_kv, 512)
                            k_nope_2d_view = pypto.view(key_nope_2d, [key_nope_2d.shape[0], dn],
                                [0, 0], valid_shape=[key_nope_2d.shape[0], dn])
                            # Gather BF16 kn: 按 topk_indices + block_table 从 KV Cache 搬入 UB
                            kn = gather_in_ub(k_nope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)

                            # ---- Sa_C1: Q × K^T ----
                            pypto.set_semantic_label("Sa_C1")
                            pypto.set_vec_tile_shapes(gather_vec_tile[0], gather_vec_tile[1])  # Vec: 32行 × 512列 (910B) / 64行 × 512列 (950)
                            pypto.set_cube_tile_shapes([c1_tile[0],
                                c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                            # Cube C1: M=[128,128], K=[128,128], N=[128,128] (910B) / N=[64,64] (950)
                            key_rope_2d_view = pypto.view(key_rope_2d, [key_rope_2d.shape[0], dr],
                                                            [0, 0], valid_shape=[key_rope_2d.shape[0], dr])
                            kr = gather_in_ub(key_rope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)

                            # 拼接 Key: [nope(512) | rope(64)] = 576 维
                            kj = pypto.tensor([cur_s2_tile, dn + dr], dtype, "kj")
                            pypto.assemble(kn, [0, 0], kj)
                            pypto.assemble(kr, [0, dn], kj)
                            kj_view = pypto.view(kj, [cur_s2_tile, dn + dr], [0, 0],
                                valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn + dr])

                            # 拼接 Query: [nope(512) | rope(64)] = 576 维
                            qn = pypto.view(query_nope, [cur_group_tile, dn], [cur_offset, 0],
                                            valid_shape=[cur_group_tile, dn])
                            qr = pypto.view(query_rope, [cur_group_tile, dr], [cur_offset, 0],
                                            valid_shape=[cur_group_tile, dr])
                            qi = pypto.tensor([cur_group_tile, dn + dr], dtype, "qi")
                            pypto.assemble(qn, [0, 0], qi)
                            pypto.assemble(qr, [0, dn], qi)

                            # C1 matmul: S = Q × K^T, shape=(group_tile, s2_tile), FP32
                            sij = pypto.matmul(qi, kj_view, pypto.DT_FP32, a_trans=False, b_trans=True)

                        # ---- Sa_V1: 标准 Softmax（INT8/BF16 路径汇合） ----
                        pypto.set_semantic_label("Sa_V1")
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])  # Vec: 8行 × 2048列 (910B) / 4行 × 2048列 (950)
                        sij_scale = pypto.mul(sij, softmax_scale)                  # S * 1/sqrt(d_q)
                        tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)  # 行最大值 (防溢出)
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)              # S - max
                        tilda_pij = pypto.exp(t_sub)                                # exp(S - max)
                        tilda_lij_reduce = pypto.sum(tilda_pij, dim=-1, keepdim=True)   # sum of exp
                        # softmax = exp / sum, INTRINSIC 精度除法
                        t_softmax = pypto.div(tilda_pij, tilda_lij_reduce, pypto.PrecisionType.INTRINSIC)
                        tilda_pij_f16 = pypto.cast(t_softmax, dtype)                # FP32 → BF16

                        # ---- Sa_C2: softmax × V, shape=(group_tile, dn) ----
                        pypto.set_semantic_label("Sa_C2")
                        pypto.set_cube_tile_shapes([c2_tile[0],
                            c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]]) # Cube C2: M=[128,128], K=[128,128], N=[128,128]

                        # V = Key nope 部分 (dn=512)
                        vj = pypto.view(kn, [cur_s2_tile, dn], [0, 0],
                                        valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                        q1 = pypto.matmul(tilda_pij_f16, vj, dtype)

                        # ---- 写回输出: assemble 到 2D 中间 tensor → reshape 为 4D 输出 ----
                        pypto.assemble(q1, [cur_offset, 0], atten_out_2dim)

                        attention_out[:] = pypto.reshape(atten_out_2dim,
                                                    [attention_out.shape[0], attention_out.shape[1],
                                                     attention_out.shape[2], attention_out.shape[3]], inplace=True)


def sparse_flash_attention_quant_compute_flash(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                               k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                               attention_out, nq, n_kv, softmax_scale, topk,
                                               block_size, max_blocknum_perbatch, tile_config):
    """Flash Online Softmax 模式 — Prefill 阶段专用。

    使用 Flash Attention 算法，跨 s2 tile 增量更新 oi/li/mi 三个运行状态：
      - oi_update: 累积注意力输出 (未归一化)
      - li_update: 累积 exp sum, shape=(1, group_tile)
      - mi_update: 累积 max, shape=(1, group_tile)
    仅在最后一个 s2 tile 时做最终归一化 O_final = oi / li。
    适用于 s1=256 的 prefill 场景。
    被 sparse_flash_attention_quant_p JIT 入口调用。

    计算流水线（每个 s2 tile）：
        Sa_V0: Gather Key (nope+rope) + INT8 反量化（如需）
        Sa_C1: S_ij = Q × K^T           (FP32 Cube matmul)
        Sa_V1: partial softmax           (exp(S-max), 不做最终除法)
        Sa_C2: O_partial = exp_softmax × V (FP32 Cube matmul)
        Sa_UpdateVec2: 增量更新 oi/li/mi (非首个 tile)
        Sa_V2: 最终归一化 O = oi / li    (最后一个 tile)

    Flash Attention 增量更新公式：
        mi_new = max(mi, tilda_mij)
        li_new = exp(mi - mi_new) * li + exp(tilda_mij - mi_new) * tilda_lij
        oi_new = exp(mi - mi_new) * oi + exp(tilda_mij - mi_new) * q1

    Args: 同 sparse_flash_attention_quant_compute，额外使用 tile_config.v2_tile_shape。
    """
    dtype = query_nope.dtype            # BF16
    kn_dtype = key_nope_2d.dtype        # INT8 或 BF16
    dn = query_nope.shape[1]            # kv_lora_rank = 512
    dr = query_rope.shape[1]            # qk_rope_dim = 64
    group = nq // n_kv                  # 128
    # 从 tile_config 提取各级 tile 参数
    group_tile = tile_config.g_tile                       # 128
    s2_tile = tile_config.s_kv_tile                       # 2048
    c1_tile = tile_config.c1_tile_shape                   # [128,128,128,128,128,128]
    v1_tile = tile_config.v1_tile_shape                   # [8, 2048]
    c2_tile = tile_config.c2_tile_shape                   # [128,128,128,128,128,128]
    v2_tile = tile_config.v2_tile_shape                   # [64, 128]
    n_kv_sym = n_kv                                       # 1

    batch_size_sym = kv_act_seqs.shape[0]                 # B

    s1_n2_gsym = query_nope.shape[0] // batch_size_sym    # S1 * N_Q
    s1_sym = s1_n2_gsym // nq                             # S1

    g_loop_sym = group // group_tile                      # 128/128 = 1

    # ---- FLASH_LOOP_L0: batch 维度 ----
    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="FLASH_LOOP_L0_idx", idx_name="bIdx"):
        cur_act_seq = kv_act_seqs[batch_idx]
        # ---- FLASH_LOOP_L1: s1 (query seq) 维度 ----
        for slc_idx in pypto.loop(0, s1_sym, 1, name="FLASH_LOOP_L1_s1_SA", idx_name="s1Idx"):
            # 当前 token 可关注的 KV 数量（因果 mask + topk 截断）
            cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
            cur_seq.as_variable()
            # s2 tile 数量（向上取整）
            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile

            # ---- FLASH_LOOP_L2: n_kv (KV head) 维度 ----
            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="FLASH_LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                # ---- FLASH_LOOP_L3: GQA group 维度 ----
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="FLASH_LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = group_tile            # 128
                    # Flash Attention 三个运行状态: oi (输出), li (exp sum), mi (max)
                    oi_update = pypto.tensor([cur_group_tile, dn], pypto.DT_FP32, "oi_update")  # (128, 512)
                    li_update = pypto.tensor([1, cur_group_tile], pypto.DT_FP32, "li_update")   # (1, 128)
                    mi_update = pypto.tensor([1, cur_group_tile], pypto.DT_FP32, "mi_update")   # (1, 128)

                    # 当前 group 在 2D query 中的偏移
                    cur_offset = batch_idx * s1_n2_gsym + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile
                    # 输出 4D attention_out 的写入偏移 [b, s1, head_start, 0]
                    oi_offset = [batch_idx, slc_idx, n_kv_idx * group + group_idx * cur_group_tile, 0]

                    # ---- FLASH_LOOP_L4: s2 (KV seq tile) 维度 (loop_unroll) ----
                    for s2_idx, _ in pypto.loop_unroll(0, bn_per_batch, 1,
                        name="FLASH_LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
                        cur_s2_tile = s2_tile              # 2048

                        # ---- Sa_V0: 准备 Gather 参数 ----
                        pypto.set_semantic_label("Sa_V0")
                        # 取出当前 (batch, s1) 的 topk_indices 切片
                        cur_topk_indices = pypto.view(topk_indices, [1, cur_s2_tile],
                                                  [batch_idx * s1_sym + slc_idx, s2_idx * cur_s2_tile],
                                                  valid_shape=[1, (cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile)])
                        # 取出当前 batch 的 block_table
                        cur_block_table = pypto.view(block_table, [1, max_blocknum_perbatch], [batch_idx, 0])
                        # view key_nope: (total_kv, dn=512)
                        k_nope_2d_view = pypto.view(key_nope_2d, [key_nope_2d.shape[0], dn],
                            [0, 0], valid_shape=[key_nope_2d.shape[0], dn])
                        # view scales: (total_kv, 4)
                        k_nope_scale_view = pypto.view(k_nope_scales, [k_nope_scales.shape[0], 4],
                            [0, 0], valid_shape=[k_nope_scales.shape[0], 4])

                        kn = pypto.tensor([s2_tile, dn], dtype, "kn")

                        # ===== INT8 量化 Key 路径 =====
                        if kn_dtype == pypto.DT_INT8:
                            pypto.set_vec_tile_shapes(32, 512)  # Vec: 32行 × 512列, INT8 Gather scales + quantized kn
                            # Gather INT8 scales
                            kn_scale = gather_in_ub(k_nope_scale_view, cur_topk_indices,
                                                    cur_block_table, block_size, -2)
                            # Gather INT8 quantized kn
                            kn_quant = gather_in_ub(k_nope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)
                            # 反量化: INT8 → FP16 → FP32 → × scale → BF16
                            kn_quant_fp16 = pypto.cast(kn_quant, pypto.DT_FP16)
                            kn_quant_fp32 = pypto.cast(kn_quant_fp16, pypto.DT_FP32)
                            # reshape 为 (s2*4, 128): 512维=4组×128元素
                            kn_quant_fp32_tmp = pypto.reshape(kn_quant_fp32, [s2_tile * 4, 128])
                            kn_scale_tmp = pypto.reshape(kn_scale, [s2_tile * 4, 1])
                            # 逐元素乘: INT8_val * scale
                            pypto.set_vec_tile_shapes(128, 128)  # Vec: 128行 × 128列, INT8 × FP32 scale 逐元素乘
                            kn_fp32 = pypto.mul(kn_quant_fp32_tmp, kn_scale_tmp)
                            kn_fp32_reshape = pypto.reshape(kn_fp32, [s2_tile, dn])
                            # view 有效区域
                            pypto.set_vec_tile_shapes(32, 512)   # Vec: 32行 × 512列, view 反量化结果切有效区域
                            cur_kn_fp32 = pypto.view(kn_fp32_reshape, [cur_s2_tile, dn], [0, 0],
                                valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                            kn = pypto.cast(cur_kn_fp32, dtype)   # FP32 → BF16

                        # ===== BF16 纯精度 Key 路径 =====
                        else:
                            # BF16 kn 直接通过 gather_in_l1 送入 L1，Cube 可直接消费
                            pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]],
                                [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                            # Cube C1: M=[128,128], K=[128,128], N=[128,128] (910B Prefill)
                            kn = gather_in_l1(key_nope_2d,
                                cur_topk_indices, cur_block_table, block_size, dn, is_b_matrix=True, is_trans=True)

                        # ---- Sa_C1: Q × K^T ----
                        pypto.set_semantic_label("Sa_C1")
                        pypto.set_cube_tile_shapes([c1_tile[0],
                            c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                        # Cube C1: M=[128,128], K=[128,128], N=[128,128] (910B Prefill)

                        # Gather key_rope: 送入 L1 供 Cube 消费
                        kr = gather_in_l1(key_rope_2d, cur_topk_indices, cur_block_table, block_size, dr,
                                          is_b_matrix=True, is_trans=True)
                        # 拼接 Key: [nope(512) | rope(64)] = 576 维
                        kj = pypto.tensor([cur_s2_tile, dn + dr], dtype, "kj")
                        pypto.assemble(kn, [0, 0], kj)
                        pypto.assemble(kr, [0, dn], kj)
                        kj_view = pypto.view(kj, [cur_s2_tile, dn + dr], [0, 0],
                                             valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn + dr])

                        # 拼接 Query: [nope(512) | rope(64)] = 576 维
                        qn = pypto.view(query_nope, [cur_group_tile, dn], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dn])
                        qr = pypto.view(query_rope, [cur_group_tile, dr], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dr])
                        qi = pypto.tensor([cur_group_tile, dn + dr], dtype, "qi")
                        pypto.assemble(qn, [0, 0], qi)
                        pypto.assemble(qr, [0, dn], qi)

                        # C1 matmul: S = Q × K^T, shape=(group_tile, s2_tile), FP32
                        sij = pypto.matmul(qi, kj_view, pypto.DT_FP32, a_trans=False, b_trans=True)

                        # ---- Sa_V1: Partial Softmax（不做最终除法） ----
                        pypto.set_semantic_label("Sa_V1")
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])  # Vec: 8行 × 2048列 (910B Prefill)
                        sij_scale = pypto.mul(sij, softmax_scale)                  # S * 1/sqrt(d_q)
                        tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)  # (group_tile, 1) 当前 tile max
                        tilda_mij = pypto.reshape(tilda_mij_reduce, [1, cur_group_tile]) # (1, group_tile) 转置用于广播
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)              # S - max
                        tilda_pij = pypto.exp(t_sub)                                # exp(S - max), 不除以 sum
                        tilda_pij_f16 = pypto.cast(tilda_pij, dtype)                # → BF16 用于 C2
                        tilda_lij_reduce = pypto.sum(tilda_pij, dim=-1, keepdim=True)   # (group_tile, 1) exp sum
                        tilda_lij = pypto.reshape(tilda_lij_reduce, [1, cur_group_tile]) # (1, group_tile)

                        # ---- Sa_C2: exp_softmax × V ----
                        pypto.set_semantic_label("Sa_C2")
                        pypto.set_cube_tile_shapes([c2_tile[0],
                            c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                        # Cube C2: M=[128,128], K=[128,128], N=[128,128] (910B Prefill)
                        pypto.set_matrix_size([tilda_pij_f16.shape[0],
                            tilda_pij_f16.shape[1], kn.shape[1]])

                        # V = Key nope 部分
                        q1 = pypto.tensor([cur_group_tile, dn], dtype)
                        if kn_dtype == pypto.DT_INT8:
                            # INT8 路径: V 从已反量化的 kn 中取
                            vj = pypto.view(kn, [cur_s2_tile, dn], [0, 0],
                                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                            q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
                        else:
                            # BF16 路径: V 通过 gather_in_l1 重新 Gather（不转置）
                            vj = gather_in_l1(key_nope_2d, cur_topk_indices, cur_block_table, block_size,
                                dn, is_b_matrix=True, is_trans=False)
                            q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)

                        # ---- Flash Attention 增量更新 ----
                        if pypto.cond(pypto.is_loop_begin(s2_idx)):
                            # ===== 首个 s2 tile: 初始化 oi/li/mi =====
                            oi_tmp = q1
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])  # Vec: 64行 × 128列 (910B Prefill)
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                # 只有一个 tile: 直接归一化 O = q1 / li
                                pypto.set_semantic_label("Sa_V2")
                                oi_update[:] = oi_tmp / tilda_lij_reduce
                                # FP32 → BF16, reshape 为 4D 并写入输出
                                pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])  # Vec: 1×1 + 64×128, cast+assemble 写回输出
                                oi_update_4_dim = pypto.cast(pypto.reshape(oi_update,
                                    [1, 1, cur_group_tile, dn]), dtype)
                                pypto.assemble(oi_update_4_dim, oi_offset, attention_out)
                            else:
                                # 多 tile: 存储未归一化的 q1, 后续 tile 会修正
                                oi_update[:] = oi_tmp
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])  # Vec: 64行 × 128列, 更新 li/mi
                            li_update[:] = tilda_lij      # 初始化累积 exp sum
                            mi_update[:] = tilda_mij      # 初始化累积 max
                        else:
                            # ===== 后续 s2 tile: 用 online softmax 公式修正历史值 =====
                            pypto.set_semantic_label("Sa_UpdateVec2")
                            oi = oi_update     # 历史累积 O (未归一化)
                            li = li_update     # 历史累积 exp sum, (1, group_tile)
                            mi = mi_update     # 历史累积 max, (1, group_tile)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])  # Vec: 64行 × 128列, max/sub/exp 修正因子
                            # 新 max = max(历史 max, 当前 tile max)
                            mi_new = pypto.maximum(mi, tilda_mij)
                            # 历史值修正因子: exp(旧max - 新max)
                            t1 = pypto.sub(mi, mi_new)
                            t2 = pypto.exp(t1)
                            # 当前 tile 修正因子: exp(当前max - 新max)
                            t3 = pypto.sub(tilda_mij, mi_new)
                            t4 = pypto.exp(t3)
                            # 更新累积 exp sum: li_new = exp(旧max-新max)*li + exp(当前max-新max)*当前sum
                            t5 = pypto.mul(t4, tilda_lij)
                            t6 = pypto.mul(t2, li)
                            li_new = pypto.add(t6, t5)
                            # 更新 O: oi_new = exp(旧max-新max)*旧oi + exp(当前max-新max)*q1
                            q3 = pypto.mul(oi, pypto.reshape(t2, [cur_group_tile, 1]))    # 修正历史 O
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])  # Vec: 64行 × 128列, mul 修正当前 O_partial
                            q2 = pypto.mul(q1, pypto.reshape(t4, [cur_group_tile, 1]))    # 修正当前 O_partial
                            oi_tmp = pypto.add(q3, q2)

                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                # 最后一个 tile: 最终归一化 O = oi / li
                                oi_update[:] = pypto.div(oi_tmp,
                                    pypto.reshape(li_new, [cur_group_tile, 1]), pypto.PrecisionType.INTRINSIC)
                                # FP32 → BF16, reshape 为 4D 并写入输出
                                pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])  # Vec: 1×1 + 64×128, cast+assemble 写回输出
                                oi_update_4_dim = pypto.cast(pypto.reshape(oi_update,
                                    [1, 1, cur_group_tile, dn]), dtype)
                                pypto.assemble(oi_update_4_dim, oi_offset, attention_out)
                            else:
                                # 中间 tile: 存储未归一化结果
                                oi_update[:] = oi_tmp
                            li_update[:] = li_new
                            mi_update[:] = mi_new


# ========================================================================================
# JIT 入口函数: @pypto.frontend.jit 装饰，编译后运行于 NPU
# ========================================================================================
#
# pass_options: 编译期 Pass 配置
#   - vec_nbuffer_setting:  向量算子 UB buffer 数量, {-1: 全局默认, 0: 第0个scope, -2: 倒数第2个scope}
#   - cube_l1_reuse_setting: Cube 算子 L1 复用策略, {-1: 全局默认}
#
# runtime_options: 运行时配置
#   - stitch_function_max_num: 最大 stitch 段数 (控制算子分段数上限)
#   - device_sched_mode: 设备调度模式 (3=多核并行调度)
#
# debug_options: 调试配置 (仅开发阶段使用)
#   - runtime_debug_mode: 运行时调试开关
#   - compile_debug_mode: 编译期调试开关
# ========================================================================================


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 4, -2: 1},      # 全局 4 个 UB buffer, 倒数第 2 scope 1 个
        "cube_l1_reuse_setting": {-1: 8},            # 全局 L1 复用系数 8
    },
    runtime_options={
        "stitch_function_max_num": 128,              # stitch 最大段数 128
        "device_sched_mode": 3                       # 多核并行调度
    }
)
def sparse_flash_attention_quant_d_950(
    query_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    key_nope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], ), # int8 or bf16
    key_rope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    k_nope_scales: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),

    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
):
    """950 芯片 Decode 入口 — 标准 softmax。

    调用 sparse_flash_attention_quant_compute（标准 softmax 归一化）。
    适用于 Ascend 950, s1=1/2 的 decode 场景。

    Tensor 类型标注:
        DYNAMIC = 运行时动态维度, STATIC = 编译期固定维度
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                        k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                        attention_out, nq, n_kv, softmax_scale, topk,
                                        block_size, max_blocknum_perbatch, tile_config)


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 8},       # 全局 2 个 UB buffer, 第 0 scope 8 个
        "cube_l1_reuse_setting": {-1: 2},            # 全局 L1 复用系数 2
    },
    runtime_options={
        "stitch_function_max_num": 128,              # stitch 最大段数 128
        "device_sched_mode": 3                       # 多核并行调度
    },
    debug_options={
        "runtime_debug_mode": 1,                     # 运行时调试开启
        "compile_debug_mode": 1                      # 编译期调试开启
    }
)
def sparse_flash_attention_quant_d(
    query_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    key_nope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], ), # int8 or bf16
    key_rope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    k_nope_scales: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),

    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
):
    """910B Decode 入口 — 标准 softmax。

    调用 sparse_flash_attention_quant_compute（标准 softmax 归一化）。
    适用于 Ascend 910B, s1=1/2 的 decode 场景。
    注意: debug_options 仅开发阶段开启，正式发布时应移除。
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                        k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                        attention_out, nq, n_kv, softmax_scale, topk,
                                        block_size, max_blocknum_perbatch, tile_config)


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 4, 0: 16},      # 全局 4 个 UB buffer, 第 0 scope 16 个 (prefill 需更多 buffer)
        "cube_l1_reuse_setting": {-1: 4},            # 全局 L1 复用系数 4
    },
    runtime_options={
        "stitch_function_max_num": 128               # stitch 最大段数 128 (prefill 不设 device_sched_mode)
    }
)
def sparse_flash_attention_quant_p(
    query_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    key_nope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC],), # int8 or bf16
    key_rope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    k_nope_scales: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),

    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
):
    """910B Prefill 入口 — Flash online softmax。

    调用 sparse_flash_attention_quant_compute_flash（Flash Attention 增量更新 oi/li/mi）。
    适用于 Ascend 910B, s1=256 的 prefill 场景。
    注意: Prefill 不设 device_sched_mode (与 Decode 不同)。
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                          k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                          attention_out, nq, n_kv, softmax_scale, topk,
                                          block_size, max_blocknum_perbatch, tile_config)
