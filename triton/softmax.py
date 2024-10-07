import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x, output, stride_xm, stride_xn, stride_om, stride_on, BLOCK_SIZE:tl.constexpr):
    row_id = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x + row_id * stride_xm + offset * stride_xn
    output_ptrs = output + row_id * stride_om + offset * stride_on

    x = tl.load(x_ptrs, mask=offset < BLOCK_SIZE, other=-float('inf'))
    max_val = tl.reduce.max(x, axis=0)

    x = tl.exp(x - max_val)
    sum_exp = tl.reduce.sum(x, axis=0)

    softmax_res = x / sum_exp
    tl.store(output_ptrs, softmax_res, mask=offset < BLOCK_SIZE)


def softmax(x):
    x = x.to(device='cuda')
    output = torch.empty_like(x)
    m, n = x.shape
    BLOCK_SIZE = 128
    grid = triton.cdiv(m, BLOCK_SIZE)
    softmax_kernel[grid, BLOCK_SIZE](x, output, x.stride(0), x.stride(1), output.stride(0), output.stride(1), BLOCK_SIZE)
    return output

if __name__ == "__main__":
    # Create input tensor
    X = torch.randn(1024, 1024, device="cuda")
    
    # Call softmax function
    output = softmax(X)
    
    # Check result
    print(output[:5, :5])  # Print first 5 elements of the result
