import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Softmax Kernel")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--block-m", type=int, default=2, help="Block M size")
args = parser.parse_args()

M = args.m
N = args.n
block_M = args.block_m

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def softmax(M, N, block_M, dtype="float"):
    VEC_NUM = 2
    ROWS = block_M // VEC_NUM
    m_num = M // block_M

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * ROWS

            a_ub = T.alloc_ub((ROWS, N), dtype)
            max_ub = T.alloc_ub((ROWS, 1), dtype)
            max_tile = T.alloc_ub((ROWS, N), dtype)
            exp_ub = T.alloc_ub((ROWS, N), dtype)
            sum_ub = T.alloc_ub((ROWS, 1), dtype)
            sum_tile = T.alloc_ub((ROWS, N), dtype)
            b_ub = T.alloc_ub((ROWS, N), dtype)

            T.copy(A[row_start : row_start + ROWS, :], a_ub)

            T.reduce_max(a_ub, max_ub, dim=-1)
            T.tile.broadcast(max_tile, max_ub)
            T.tile.sub(exp_ub, a_ub, max_tile)
            T.tile.exp(exp_ub, exp_ub)

            T.reduce_sum(exp_ub, sum_ub, dim=-1)
            T.tile.broadcast(sum_tile, sum_ub)
            T.tile.div(b_ub, exp_ub, sum_tile)

            T.copy(b_ub, B[row_start : row_start + ROWS, :])

    return main


func = softmax(M, N, block_M)

torch.manual_seed(0)

a = torch.randn(M, N).npu()
torch.npu.synchronize()
print("init successful!")

b = func(a)

ref_b = torch.nn.functional.softmax(a, dim=-1)

torch.testing.assert_close(b, ref_b, rtol=1e-4, atol=1e-4)
print("Softmax Kernel Output Match!")