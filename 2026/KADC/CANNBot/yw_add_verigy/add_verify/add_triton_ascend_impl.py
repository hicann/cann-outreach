import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, num_cores: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, num_blocks, num_cores):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch_npu
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, x, y):
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()

        output = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 2048

        add_kernel[(self.VEC_CORE_NUM,)](x, y, output, n_elements, num_cores=self.VEC_CORE_NUM, BLOCK_SIZE=BLOCK_SIZE)
        return output