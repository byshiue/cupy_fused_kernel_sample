import cupy as cp
import numba as nb
import numpy as np
from datetime import datetime

fused_kernel = cp.RawKernel(r'''
    extern "C"
    __global__ void my_kernel(float *output, float *input_1,
                              float *input_2, float *input_3, 
                              float *input_4, float *input_5, 
                              float *input_6, float *input_7, 
                              unsigned int n) {
        int tid = threadIdx.x;
        int bid = blockIdx.x;

        for(int i = bid * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x)
        {
            output[i] = (input_1[i] * input_1[i] + input_2[i]) * input_3[i] / (input_4[i] + 1e-20) + (input_5[i] * input_6[i]) / (input_7[i] + 1e-20);
        }
    }
    ''', 'my_kernel')

data_size_1 = 64
data_size_2 = 512
data_size_3 = 512
block_size = 128 # number of threads per block
grid_size = 256 # number of blocks per grid

a = cp.random.rand(data_size_1, data_size_2, data_size_3).astype(cp.float32)
b = cp.random.rand(data_size_1, data_size_2, data_size_3).astype(cp.float32)
c = cp.random.rand(data_size_1, data_size_2, data_size_3).astype(cp.float32)
d = cp.random.rand(data_size_1, data_size_2, data_size_3).astype(cp.float32)
e = cp.random.rand(data_size_1, data_size_2, data_size_3).astype(cp.float32)
f = cp.random.rand(data_size_1, data_size_2, data_size_3).astype(cp.float32)
g = cp.random.rand(data_size_1, data_size_2, data_size_3).astype(cp.float32)

d1 = (a**2 + b) * c / d + (e * f) / g

d2 = cp.zeros_like(a)
fused_kernel((grid_size,), (block_size, ), (d2, a, b, c, d, e, f, g, data_size_1*data_size_2*data_size_3))

print("[INFO] max relative difference: {}".format(abs((d1 - d2) / d1).max()))

warmup_iteration = 10
test_iteration = 50

# warmup
for i in range(warmup_iteration):
    d1 = (a**2 + b) * c / d + (e * f) / g
t1 = datetime.now()
for i in range(test_iteration):
    d1 = (a**2 + b) * c / d + (e * f) / g
t2 = datetime.now()

print("[INFO] cupy time:       {} ms".format((t2-t1).total_seconds() * 1000 / test_iteration))

# warmup
for i in range(warmup_iteration):
    fused_kernel((grid_size,), (block_size, ), (d2, a, b, c, d, e, f, g, data_size_1*data_size_2*data_size_3))

t1 = datetime.now()
for i in range(test_iteration):
    fused_kernel((grid_size,), (block_size, ), (d2, a, b, c, d, e, f, g, data_size_1*data_size_2*data_size_3))
t2 = datetime.now()

print("[INFO] raw kernel time: {} ms".format((t2-t1).total_seconds() * 1000 / test_iteration))
