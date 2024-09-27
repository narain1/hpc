import torch
import os
from torch.utils.cpp_extension import load_inline
from PIL import Image
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ

def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(
        cuda_sources=[cuda_src],
        cpp_sources=[cpp_src],
        functions=funcs,
        verbose=verbose,
        with_cuda=True,
        extra_cuda_cflags=['-O2'] if opt else [],
        extra_cflags=['-std=c++14'],  # Use C++14 standard
        name='cuda_ext'
    )

cuda_begin = r'''
#include 
#include 
#include 

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''


cuda_src = cuda_begin + r'''
__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>>(
        input.data_ptr(), output.data_ptr(), w*h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}'''
     
cpp_src = r'''torch::Tensor rgb_to_grayscale(torch::Tensor input);'''

module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale'], opt=True, verbose=True)

img = Image.open('test.jpg')
img = np.array(img)

img = torch.tensor(img).permute(2, 0, 1).float().cuda()

out = module.rgb_to_grayscale(img)
print(out.shape)

out = out.cpu().numpy()
out = out[0]
out = Image.fromarray(out)
out.save('out.jpg')