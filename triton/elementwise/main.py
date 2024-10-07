import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_add_f16_kernel(a_ptr, b_ptr, c_ptr, N, block_size: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * block_size + tl.arange(0, block_size)
    mask = idx < N
    
    a = tl.load(a_ptr + idx, mask=mask, other=0)
    b = tl.load(b_ptr + idx, mask=mask, other=0)

    c = a + b
    tl.store(c_ptr + idx, c, mask=mask)

def sigmoid(x_ptr, N, block_size: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * block_size + tl.arange(0, block_size)
    mask = idx < N

    x = tl.load(x_ptr + idx, mask=mask, other=0)
    y = 1 / (1 + tl.exp(-x))
    tl.store(x_ptr + idx, y, mask=mask)

def test_elementwise_add_f16():
    N = 1024
    dtype = torch.float16
    block_size = 128

    a = torch.randn(N, dtype=dtype, device='cuda')
    b = torch.randn(N, dtype=dtype, device='cuda')
    c = torch.empty(N, dtype=dtype, device='cuda')

    c_triton = torch.empty_like(a)

    grid = (triton.cdiv(N, block_size),)
    elementwise_add_f16_kernel[grid](a, b, c_triton, N, block_size)
    assert torch.allclose(c_triton, a + b)
    print(c_triton)
    print(a + b)

test_elementwise_add_f16()