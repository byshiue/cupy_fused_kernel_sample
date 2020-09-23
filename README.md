# cupy_fused_kernel_sample

This is an example to show that how to fuse the cupy kernel by cupy raw_kernel function (in fact, the cuda codes).

The comparison under RTX 2080 ti:
```bash
[INFO] max relative difference: 3.4637577e-07
[INFO] cupy time:       2.50198 ms
[INFO] raw kernel time: 1.66084 ms
```
