import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


try:
    VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
except Exception:
    VEC_CORE_NUM = 40

print(f"VEC_CORE_NUM: {VEC_CORE_NUM}")
print(f"torch version: {torch.__version__}")
print(f"torch_npu available: {torch_npu.npu.is_available()}")

x = torch.randn(1024, 1024, dtype=torch.float16).npu()
y = torch.randn(1024, 1024, dtype=torch.float16).npu()

output = torch.empty_like(x)
n_elements = x.numel()
BLOCK_SIZE = 2048
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

print(f"grid size: {grid}")
print(f"n_elements: {n_elements}")
import time
start = time.time()
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
print("11111",time.time() - start)
golden = x + y
max_diff = (output.cpu() - golden.cpu()).abs().max().item()
all_close = torch.allclose(output.cpu(), golden.cpu(), atol=1e-3)

print(f"Max diff: {max_diff}")
print(f"All close (atol=1e-3): {all_close}")
print("TEST PASSED" if all_close else "TEST FAILED")
