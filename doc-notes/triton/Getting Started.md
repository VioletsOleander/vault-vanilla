# Installation
For supported platform/OS and supported hardware, review the [Compatibility](https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility) section on Github.

## Binary Distributions
You can install the latest stable release of Triton from pip:

```
pip install triton
```

Binary wheels are available for CPython 3.8-3.12 and PyPy 3.8-3.9.

And the latest nightly release:

```
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

## From Source
## Python Package
You can install the Python package from source by running the following commands:

```
git clone https://github.com/triton-lang/triton.git;
cd triton/python;
pip install ninja cmake wheel; # build-time dependencies
pip install -e .
```

Note that, if llvm is not present on your system, the `setup.py` script will download the official LLVM static libraries and link against that.

For building with a custom LLVM, review the [Building with a custom LLVM](https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm) section on Github.

You can then test your installation by running the unit tests:

```
pip install -e '.[tests]'
pytest -vs test/unit/
```

and the benchmarks

```
cd bench
python -m run --with-plots --result-dir /tmp/triton-bench
```

# Tutorials
Below is a gallery of tutorials for writing various basic operations with Triton. It is recommended that you read through the tutorials in order, starting with the simplest one.

To install the dependencies for the tutorials:

```
cd triton
pip install -e './python[tutorials]'
```

## Vector Addition
In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

- The basic programming model of Triton.
- The `triton.jit` decorator, which is used to define Triton kernels.
- The best practices for validating and benchmarking your custom ops against native reference implementations.

### Compute Kernel

```python
import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

> 本例中：
> `tl.program_id()` 用于获取 grid 内的 block id；`tl.load()` 接受指针，返回数据对象，用于加载数据到 SRAM；`tl.store()` 用于写回到 DRAM
> `@triton.jit` 修饰的 kernel 一般接受的参数也是 `x_ptr/y_ptr/output_ptr` ，即指针

Let’s also declare a helper function to (1) allocate the z tensor and (2) enqueue the above kernel with appropriate grid/block sizes:

```python
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
```

> 本例中：
> - `add() ` 函数负责分配 ` output ` 张量的空间，并且计算合适的 grid/block 大小参数，调用 ` @triton.jit ` 修饰的 kernel
> -  `add()` 函数接受 `torch.Tensor` ，在将 `torch.Tensor` 传入给 `@triton.jit` 修饰的 kernel 时，`torch.Tensor` 会被隐式转化为指向其第一个元素的指针
> -  `grid` 用于指定 grid 参数，它应该是 `Tuple[int]` 或者 `Callable(metaparamters) -> Tuple[int]` ，即一个整数元组或者一个接受 (kernel 的) 元参数，返回整数元组的可调用对象
> - `@triton.jit` 修饰的 kernel 调用时需要添加 `[grid]` 以传递 grid 参数，`@triton.jit` 修饰的 kernel 的 meta-parameter 通过关键字参数的形式传入

We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

```

```
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')

The maximum difference between torch and triton is 0.0
```

Seems like we’re good to go!

### Benchmark
We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch. To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops. for different problem sizes.

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
```

> 本例中：
> `@triton.testing.perf_report(triton.testing.Benchmark(...))` 用于修饰 `benchmark()` 函数


We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or `save_path=’/path/to/results/’` to save them to disk along with raw CSV data:
> 被修饰的 `benchmark` 调用 `run()`

```python
benchmark.run(print_data=True, show_plots=True)
```

![01 vector add](https://triton-lang.org/main/_images/sphx_glr_01-vector-add_001.png)

```
vector-add-performance:
           size       Triton        Torch
0        4096.0     9.600000     8.000000
1        8192.0    15.999999    15.999999
2       16384.0    31.999999    31.999999
3       32768.0    63.999998    63.999998
4       65536.0   127.999995   127.999995
5      131072.0   219.428568   219.428568
6      262144.0   384.000001   384.000001
7      524288.0   614.400016   614.400016
8     1048576.0   819.200021   819.200021
9     2097152.0  1023.999964  1023.999964
10    4194304.0  1260.307736  1228.800031
11    8388608.0  1424.695621  1424.695621
12   16777216.0  1560.380965  1560.380965
13   33554432.0  1631.601649  1624.859540
14   67108864.0  1669.706983  1662.646960
15  134217728.0  1684.008546  1678.616907
```

## Fused Softmax
In this tutorial, you will write a fused softmax operation that is significantly faster than PyTorch’s native op for a particular class of matrices: those whose rows can fit in the GPU’s SRAM.

In doing so, you will learn about:

- The benefits of kernel fusion for bandwidth-bound operations.
- Reduction operators in Triton.

### Motivation
Custom GPU kernels for elementwise additions are educationally valuable but won’t get you very far in practice. Let us consider instead the case of a simple (numerically stabilized) softmax operation:

```python
import torch

import triton
import triton.language as tl
from triton.runtime import driver

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
```

When implemented naively in PyTorch, computing `y = naive_softmax(x)` for $x\in R^{M\times N}$ requires reading 5MN+2M elements from DRAM and writing back 3MN+2M elements. This is obviously wasteful; we’d prefer to have a custom “fused” kernel that only reads X once and does all the necessary computations on-chip. Doing so would require reading and writing back only MN bytes, so we could expect a theoretical speed-up of ~4x (i.e., (8MN+4M)/2MN). 
> 使用原生 PyTorch 实现的 `y = native_softmax(x)` ，需要从 DRAM 读 5MN+2M 个元素，向 DRAM 写 3MN+2M 个元素
> 考虑将 kernel 融合，仅读取一次，在片上完成所有运算之后再写回，减少中间结果的写回和读取，完全融合后，仅需要读 MN 个元素，写回 MN 个元素
> 理论计算上，读写的元素数量减少了4倍，因此我们期望4倍的理论速度提升

The `torch.jit.script` flags aims to perform this kind of “kernel fusion” automatically but, as we will see later, it is still far from ideal.

### Compute Kernel
Our softmax kernel works as follows: each program loads a set of rows of the input matrix X strided by number of programs, normalizes it and writes back the result to the output Y.
> softmax kernel 实现的思想是：每个 block 处理输入 X 的一部分行，将结果写回输出 Y

Note that one important limitation of Triton is that each block must have a power-of-two number of elements, so we need to internally “pad” each row and guard the memory operations properly if we want to handle any possible input shapes:
> Triton 限制每个 block 的元素数量为2的幂次，因此需要添加 pad，并且注意防止越界访问

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

> 本例中：
> - 每个 block 负责处理一定数量的行，且是按照循环一行一行地处理
> - `BLOCK_SIZE` 被设定为 `n_cols` 之后最近的2次幂，以在满足约束的情况下将一整行元素放入 block 中
> - `mask = col_offsets < n_cols` 用于防止访问大于 `n_cols` 之后的内容
> `tl.load()` 返回的 `row` 被传入 `tl.max()` 得到其中的最大元素

We can create a helper function that enqueues the kernel and its (meta-)arguments for any given input tensor.

```python
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
            # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
            # ISA SECTION (3.6.4 for CDNA3)
            # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
            # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
            # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
            # not required to be equal numbers of both types.
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2

            # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
            # When we divide this number with WARP_SIZE we get maximum number of waves that can
            # execute on a CU (multi-processor)  in parallel.
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y
```


> 本例中：
> - `driver.active.utils.get_device_properties()` 用于获得设备性质
> - `NUM_REGS = properties['max_num_regs]` 用于获得常规目的的寄存器数量，大多数情况下它就等于总寄存器数量，在 CDNA 设备中它是总寄存器数量的一半
> - 块大小通过 `triton.next_power_of_2(n_cols)` 得到
> - `kernel, num_programs = kernel.get(BLOCK_SIZE, (None, 0))` 用于检查是否有为特定的 `BLOCK_SIZE` 预编译 kernel，如果没有，则进行预热以编译内核
> - `@triton.jit` 修饰的 kernel 可以调用 `warmup` 函数，其中 `num_stages` 参数指定流水线阶段，`num_warps` 指定每个 block 的 warp 数量
> - CUDA 架构下的 occupancy 计算通过 `occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)` 得到，其逻辑就是总的寄存器数除去每个 block 使用的寄存器数，得到每个 SM 上允许执行的 block 数量
> - HIP/ROCm 架构下的 occupancy 计算通过 `occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps` 得到，其逻辑和 CUDA 类似，先计算得到一个 CU 上允许执行的最大 warp 数量，然后除以 `num_warps` 得到一个 CU 上允许执行的 block 数量
> - `occupancy = min(occupancy, SIZE_SMEM // size_smem)` 中 `SIZE_SMEM // size_smem` 表示总的共享内存大小除去 block 使用的共享内存大小，结果和之前计算的 `occupancy` ，防止 block 使用的共享内存大小溢出
> - `kernels[BLOCK_SIZE] = (kernel, num_programs)` 将预编译的内核和计算得到的 `num_programs` (内核的总 block 数量) 储存
> - `kernel[(num_programs, 1, 1)]()` 指定了 grid 维度，并执行内核

### Unit Test
We make sure that we test our kernel on a matrix with an irregular number of rows and columns. This will allow us to verify that our padding mechanism works.

```python
torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```

As expected, the results are identical.

### Benchmark
Here we will benchmark our operation as a function of the number of columns in the input matrix – assuming 4096 rows. We will then compare its performance against (1) `torch.softmax` and (2) the `naive_softmax` defined above.

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

benchmark.run(show_plots=True, print_data=True)
```

![02 fused softmax](https://triton-lang.org/main/_images/sphx_glr_02-fused-softmax_001.png)
```
softmax-performance:
          N       Triton        Torch
0     256.0   467.281196   691.275536
1     384.0   600.267002   792.442207
2     512.0   750.566100   911.741697
3     640.0   786.656276   956.004763
4     768.0   884.372361  1032.142255
5     896.0   941.707952  1078.092805
6    1024.0   998.903997  1125.458375
7    1152.0  1104.239994   616.045326
8    1280.0  1146.565722   668.975793
9    1408.0  1154.056042   724.545544
10   1536.0  1181.310455   779.473184
11   1664.0  1210.392170   814.663759
12   1792.0  1233.023973   856.933000
13   1920.0  1258.013461   905.249844
14   2048.0  1275.098280   959.882167
15   2176.0  1260.615343   978.911807
16   2304.0  1270.985516  1010.343346
17   2432.0  1298.706553  1055.540156
18   2560.0  1299.525777  1086.191253
19   2688.0  1307.978918  1105.374782
20   2816.0  1329.616494  1128.466656
21   2944.0  1322.502833  1165.328125
22   3072.0  1354.416086  1186.814340
23   3200.0  1351.484642  1195.221714
24   3328.0  1356.262941  1221.399947
25   3456.0  1378.904464  1248.566579
26   3584.0  1377.324251  1257.543915
27   3712.0  1386.380776  1267.970688
28   3840.0  1391.258774  1302.876883
29   3968.0  1390.334429  1312.960725
30   4096.0  1401.728764  1329.262438
31   4224.0  1337.066363  1158.075238
32   4352.0  1331.429740  1171.547175
33   4480.0  1350.808522  1183.737454
34   4608.0  1360.017013  1193.975480
35   4736.0  1361.146931  1197.801400
36   4864.0  1376.395393  1224.530979
37   4992.0  1370.606604  1233.908818
38   5120.0  1368.293097  1251.846144
39   5248.0  1376.872375  1258.562458
40   5376.0  1374.009037  1287.512105
41   5504.0  1381.671457  1296.067825
42   5632.0  1390.747236  1311.979928
43   5760.0  1392.012632  1323.641333
44   5888.0  1389.371200  1342.100512
45   6016.0  1397.546268  1352.851652
46   6144.0  1407.388078  1373.381405
47   6272.0  1417.184694  1377.612273
48   6400.0  1416.638566  1386.946274
49   6528.0  1413.334722  1392.439469
50   6656.0  1422.477491  1399.509639
51   6784.0  1411.632687  1414.965372
52   6912.0  1426.880070  1422.606521
53   7040.0  1419.748248  1431.026367
54   7168.0  1428.399392  1436.115399
55   7296.0  1430.955907  1440.152890
56   7424.0  1429.077395  1444.309616
57   7552.0  1422.911632  1451.591721
58   7680.0  1436.778312  1458.544767
59   7808.0  1435.296339  1464.168167
60   7936.0  1434.243822  1468.045165
61   8064.0  1442.351566  1476.899580
62   8192.0  1437.638752  1483.758419
63   8320.0  1390.354168  1402.088901
64   8448.0  1380.110385  1404.056311
65   8576.0  1392.483416  1395.325416
66   8704.0  1389.379659  1398.940098
67   8832.0  1383.790777  1407.367235
68   8960.0  1403.171683  1414.067564
69   9088.0  1409.472516  1417.743883
70   9216.0  1402.239377  1424.541599
71   9344.0  1399.805960  1422.833778
72   9472.0  1398.837157  1433.937609
73   9600.0  1396.976628  1436.228621
74   9728.0  1399.829709  1441.932406
75   9856.0  1412.644414  1443.187583
76   9984.0  1401.650221  1453.386876
77  10112.0  1414.289866  1454.271647
78  10240.0  1420.681968  1470.101736
79  10368.0  1414.870553  1466.089317
80  10496.0  1418.043238  1466.991355
81  10624.0  1407.442543  1467.421230
82  10752.0  1400.539418  1473.930018
83  10880.0  1400.318432  1478.920078
84  11008.0  1419.957690  1479.072945
85  11136.0  1420.641223  1482.984149
86  11264.0  1428.326650  1485.436424
87  11392.0  1416.182104  1491.346399
88  11520.0  1421.600843  1494.929764
89  11648.0  1422.317185  1497.560617
90  11776.0  1432.157751  1503.500886
91  11904.0  1444.587297  1509.671873
92  12032.0  1425.898365  1507.281683
93  12160.0  1418.345563  1511.607885
94  12288.0  1435.003789  1393.721063
95  12416.0  1447.971066  1391.516473
96  12544.0  1442.744803  1394.784957
97  12672.0  1446.732135  1393.614559
```

In the above plot, we can see that:

- Triton is 4x faster than the Torch JIT. This confirms our suspicions that the Torch JIT does not do any fusion here.
- Triton is noticeably faster than `torch.softmax` – in addition to being **easier to read, understand and maintain**. Note however that the PyTorch softmax operation is more general and will work on tensors of any shape.

## Matrix Multiplication
In this tutorial, you will write a very short high-performance FP16 matrix multiplication kernel that achieves performance on par with cuBLAS or rocBLAS.

You will specifically learn about:

- Block-level matrix multiplications.
- Multi-dimensional pointer arithmetic.
- Program re-ordering for improved L2 cache hit rate.
- Automatic performance tuning.

### Motivations
Matrix multiplications are a key building block of most modern high-performance computing systems. They are notoriously hard to optimize, hence their implementation is generally done by hardware vendors themselves as part of so-called “kernel libraries” (e.g., cuBLAS). Unfortunately, these libraries are often proprietary and cannot be easily customized to accommodate the needs of modern deep learning workloads (e.g., fused activation functions). In this tutorial, you will learn how to implement efficient matrix multiplications by yourself with Triton, in a way that is easy to customize and extend.

Roughly speaking, the kernel that we will write will implement the following blocked algorithm to multiply a (M, K) by a (K, N) matrix:

```python
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
# Do in parallel
for n in range(0, N, BLOCK_SIZE_N):
 acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
 for k in range(0, K, BLOCK_SIZE_K):
   a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
   b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
   acc += dot(a, b)
 C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
```

where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.
> 该例中，最外两层循环是 block 级别并行
> block 内部则在 K 维度进行了划分

### Compute Kernel
The above algorithm is, actually, fairly straightforward to implement in Triton. The main difficulty comes from the computation of the memory locations at which blocks of `A` and `B` must be read in the inner loop. For that, we need multi-dimensional pointer arithmetic.
> 处理指针算数：找到每个 block 应该读取的存储区域

#### Pointer Arithmetic
For a row-major 2D tensor `X`, the memory location of `X[i, j]` is given by `&X[i, j] = X + i*stride_xi + j*stride_xj`. 
> 例如对于二维张量 `X` ，`X[i, j]` 元素的逻辑位置是在 `X` 的第 `i` 行第 `j` 列，但其指针/物理位置应该是 `X` 的起始指针/物理位置加上 `i*stride_xi + j*stride_xj` ，其中 `stride_xi` 表示 `i` 维度每加1，元素的指针/物理位置需要移动多少，`stride_xj` 表示 `j` 维度每加1，元素的指针/物理位置需要移动多少

> 指针算数就是要正确地根据元素的逻辑位置和张量的起始地址计算出元素的指针/物理位置
> Triton 是以 block 思维编程的，因此我们需要通过指针算数，根据 block 需要处理的所有元素的逻辑位置计算出该 block 需要处理的所有元素的指针位置

Therefore, blocks of pointers for `A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and `B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:

```
&A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);

&B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
```

Which means that pointers for blocks of A and B can be initialized (i.e., `k=0`) in Triton as the following code. Also note that we need an extra modulo to handle the case where `M` is not a multiple of `BLOCK_SIZE_M` or `N` is not a multiple of `BLOCK_SIZE_N`, in which case we can pad the data with some useless values, which will not contribute to the results. For the `K` dimension, we will handle that later using masking load semantics.
> 注意如果 `M` 不是 `BLOCK_SIZE_M` 的倍数，以及 `N` 不是 `BLOCK_SIZE_N` 的倍数，我们可以做 padding，也可以添加边界处理
> 对于 `K` 维，如果 `K` 不是 `BLOCK_SIZE_K` 的倍数，我们将使用 masking load (在 `tl.load` 中指定 `mask` 参数)

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
```

> 该例中，`offs_am` 存储了 block 需要处理的 `A`  中元素各自在 ` A ` 的第几行，` offs_bn ` 存储了 block 需要处理的 `B` 中元素各自在在 ` B ` 的第几列
> `offs_k` 存储了 block 需要处理的 `A` / `B` 中元素各自在 ` A ` 的第几列/在 ` B ` 的第几行

And then updated in the inner loop as follows:
> k-loop 中，每次在 `K` 维度处理 `BLOCK_SIZE_K ` 个元素，也就是 `K` 维度在循环之间会前进 `BLOCK_SIZE_K` 个元素，故各个指针都相应前进 `stride_ak/stride_bk * BLOCK_SIZE_K`

```python
a_ptrs += BLOCK_SIZE_K * stride_ak;
b_ptrs += BLOCK_SIZE_K * stride_bk;
```

#### L2 Cache Optimizations
As mentioned above, each program instance computes a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of `C`. 
> 每个程序实例负责为 `[BLOCK_SIZE_M, BLOCK_SIZE_N]` 的 `C` 块 
> (可以看出 Triton 的编码单元是 block，写代码时需要站在 block 的角度思考，
> CUDA 则是 thread)

It is important to remember that the order in which these blocks are computed does matter, since it affects the L2 cache hit rate of our program, and unfortunately, a simple row-major ordering
> 这些 block 计算的顺序对性能有影响，其顺序会影响程序的 L2 命中率

```python
pid = tl.program_id(axis=0)
grid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m = pid // grid_n
pid_n = pid % grid_n
```

is just not going to cut it.

One possible solution is to launch blocks in an order that promotes data reuse. This can be done by ‘super-grouping’ blocks in groups of `GROUP_M` rows before switching to the next column:
> 提高 L2 命中率就是提高数据重用，一个思路就是对 block 进行分组，将 `M` 方向的 `GROUP_M` 个 block 分为一组，一起处理完 `M` 方向的 `GROUP_M` 个 block 之后，处理 `N` 方向的下一个 block 组
> 这样做会提高矩阵 `B` 的 block 的数据重用率

```python
# Program ID
pid = tl.program_id(axis=0)
# Number of program ids along the M axis
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
# Number of programs ids along the N axis
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
# Number of programs in group
num_pid_in_group = GROUP_SIZE_M * num_pid_n
# Id of the group this program is in
group_id = pid // num_pid_in_group
# Row-id of the first program in the group
first_pid_m = group_id * GROUP_SIZE_M
# If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
# *Within groups*, programs are ordered in a column-major order
# Row-id of the program in the *launch grid*
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
# Col-id of the program in the *launch grid*
pid_n = (pid % num_pid_in_group) // group_size_m
```

For example, in the following matmul where each matrix is 9 blocks by 9 blocks, we can see that if we compute the output in row-major ordering, we need to load 90 blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped ordering, we only need to load 54 blocks.

> ![../../_images/grouped_vs_row_major_ordering.png](https://triton-lang.org/main/_images/grouped_vs_row_major_ordering.png)

In practice, this can improve the performance of our matrix multiplication kernel by more than 10% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
> 示例图中，Row-major ordering 的算术密度显然要低于 Grouped ordering 的算数密度，这在 A100 上会带来 10% 的性能差距

### Final Result

```python
import torch

import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)
```

> 本例中
> - `get_cuda/hip_autotune_config()` 返回 `List[triton.Config]` ，其中 `triton.Config` 封装了 kernel 的配置参数，包括 `BLOCK_SIZE_M/N/K` , `GROUP_SIZE_M` , `num_stages` , `num_warps` 
> - `@triton.autotune` 装饰器用于装饰 `@triton.jit` 装饰的 kernel，`@triton.autotune` 接受 `List[triton.Config]` 形式的预定义配置以及 `List[str]` 形式的一组 keys，key 应该是某个传入给 kernel 的参数名称，当 key 发生变化，`autotune` 就会将 kernel 按所有预定义配置运行一遍。本例中，keys 是 `['M', 'N', 'K']` ，即矩阵维度
> - `stride_am` 表示 `ptr_a` 随着 `M` 维度的坐标变化相应加上 `stride_am` ，其他的与此类似
> - `tl.dot(a, b, accumulator)` 直接对已经 `tl.load` 的两个矩阵 `a,b` 进行乘法，将结果累加到 `accumulator` 返回更新的 `accumulator`
> - `c = accumulator.to(tl.float16)` 转化了 `accumulator` 的精度
> - `matmul_kernel` 可以调用 `@triton.jit` 修饰的 `leaky_relu`

We can now create a convenience wrapper function that only takes two input tensors, and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
> 定义完 `matmul_kernel` ，我们接着定义该 kernel 的包装函数，负责：
> 接受 tensor 输入、检查 tensor 形状限制、分配输出、发起 `matmul_kernel`

```python
def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c
```

### Unit Test
We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

```python
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
# Bigger tolerance for AMD MI200 devices.
# MI200 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 1e-2 if is_hip_mi200() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

```
triton_output_with_fp16_inputs=tensor([[-10.9531,  -4.7109,  15.6953,  ..., -28.4062,   4.3320, -26.4219],
        [ 26.8438,  10.0469,  -5.4297,  ..., -11.2969,  -8.5312,  30.7500],
        [-13.2578,  15.8516,  18.0781,  ..., -21.7656,  -8.6406,  10.2031],
        ...,
        [ 40.2812,  18.6094, -25.6094,  ...,  -2.7598,  -3.2441,  41.0000],
        [ -6.1211, -16.8281,   4.4844,  ..., -21.0312,  24.7031,  15.0234],
        [-17.0938, -19.0000,  -0.3831,  ...,  21.5469, -30.2344, -13.2188]],
       device='cuda:0', dtype=torch.float16)
torch_output_with_fp16_inputs=tensor([[-10.9531,  -4.7109,  15.6953,  ..., -28.4062,   4.3320, -26.4219],
        [ 26.8438,  10.0469,  -5.4297,  ..., -11.2969,  -8.5312,  30.7500],
        [-13.2578,  15.8516,  18.0781,  ..., -21.7656,  -8.6406,  10.2031],
        ...,
        [ 40.2812,  18.6094, -25.6094,  ...,  -2.7598,  -3.2441,  41.0000],
        [ -6.1211, -16.8281,   4.4844,  ..., -21.0312,  24.7031,  15.0234],
        [-17.0938, -19.0000,  -0.3831,  ...,  21.5469, -30.2344, -13.2188]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
triton_output_with_fp8_inputs=tensor([[-21.4375,  13.1719,   6.0352,  ...,  28.7031,   8.6719, -40.7500],
        [ 10.0000,  37.0000,  -5.5664,  ...,  20.9844,  46.8125,  30.8281],
        [ 19.5625,  -3.0078, -20.0469,  ...,  -2.1309,  -8.0625,  12.5625],
        ...,
        [-18.1562, -34.1562, -27.4219,  ..., -27.3906, -24.0938, -12.3516],
        [ -3.3945,  -8.6250, -23.6562,  ...,  -4.1094,  -3.5332, -16.0781],
        [-23.9688,  -3.2637, -33.6875,  ...,  17.3125, -36.6250,  25.8594]],
       device='cuda:0', dtype=torch.float16)
torch_output_with_fp8_inputs=tensor([[-21.4375,  13.1719,   6.0352,  ...,  28.7031,   8.6719, -40.7500],
        [ 10.0000,  37.0000,  -5.5664,  ...,  20.9844,  46.8125,  30.8281],
        [ 19.5625,  -3.0078, -20.0469,  ...,  -2.1309,  -8.0625,  12.5625],
        ...,
        [-18.1562, -34.1562, -27.4219,  ..., -27.3906, -24.0938, -12.3516],
        [ -3.3945,  -8.6250, -23.6562,  ...,  -4.1094,  -3.5332, -16.0781],
        [-23.9688,  -3.2637, -33.6875,  ...,  17.3125, -36.6250,  25.8594]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
```

### Benchmark
#### Square Matrix Performance
We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices, but feel free to arrange this script as you wish to benchmark any other matrix shape.

```python
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)
```

![03 matrix multiplication](https://triton-lang.org/main/_images/sphx_glr_03-matrix-multiplication_001.png)
 ![03 matrix multiplication](https://triton-lang.org/main/_images/sphx_glr_03-matrix-multiplication_002.png)

```
matmul-performance-fp16:
         M       N       K      cuBLAS      Triton
0    256.0   256.0   256.0    4.096000    4.096000
1    384.0   384.0   384.0   12.288000   12.288000
2    512.0   512.0   512.0   26.214401   26.214401
3    640.0   640.0   640.0   42.666665   42.666665
4    768.0   768.0   768.0   63.195428   68.056616
5    896.0   896.0   896.0   78.051553   87.808000
6   1024.0  1024.0  1024.0  110.376426   99.864382
7   1152.0  1152.0  1152.0  135.726544  129.825388
8   1280.0  1280.0  1280.0  157.538463  163.840004
9   1408.0  1408.0  1408.0  155.765024  132.970149
10  1536.0  1536.0  1536.0  176.947204  157.286398
11  1664.0  1664.0  1664.0  183.651271  179.978245
12  1792.0  1792.0  1792.0  172.914215  204.353162
13  1920.0  1920.0  1920.0  200.347822  168.585369
14  2048.0  2048.0  2048.0  223.696203  190.650180
15  2176.0  2176.0  2176.0  211.827867  211.827867
16  2304.0  2304.0  2304.0  228.592087  225.357284
17  2432.0  2432.0  2432.0  203.583068  203.583068
18  2560.0  2560.0  2560.0  224.438347  221.405396
19  2688.0  2688.0  2688.0  200.704002  198.602388
20  2816.0  2816.0  2816.0  212.752230  212.752230
21  2944.0  2944.0  2944.0  222.482283  222.482283
22  3072.0  3072.0  3072.0  209.715208  213.672083
23  3200.0  3200.0  3200.0  216.216207  218.430042
24  3328.0  3328.0  3328.0  207.467716  206.871539
25  3456.0  3456.0  3456.0  217.308808  219.080343
26  3584.0  3584.0  3584.0  219.305830  225.351853
27  3712.0  3712.0  3712.0  206.399476  217.641271
28  3840.0  3840.0  3840.0  210.250955  212.268710
29  3968.0  3968.0  3968.0  208.587935  217.899880
30  4096.0  4096.0  4096.0  219.668951  220.029067
matmul-performance-fp8:
         M       N       K      Triton
0    256.0   256.0   256.0    3.276800
1    384.0   384.0   384.0    9.216000
2    512.0   512.0   512.0   18.724571
3    640.0   640.0   640.0   32.000000
4    768.0   768.0   768.0   42.130286
5    896.0   896.0   896.0   58.538665
6   1024.0  1024.0  1024.0   61.680940
7   1152.0  1152.0  1152.0   80.702267
8   1280.0  1280.0  1280.0   99.902441
9   1408.0  1408.0  1408.0   82.602666
10  1536.0  1536.0  1536.0   98.303997
11  1664.0  1664.0  1664.0  115.370671
12  1792.0  1792.0  1792.0  133.802668
13  1920.0  1920.0  1920.0  100.173911
14  2048.0  2048.0  2048.0  114.130722
15  2176.0  2176.0  2176.0  120.500882
16  2304.0  2304.0  2304.0  134.201527
17  2432.0  2432.0  2432.0  132.521057
18  2560.0  2560.0  2560.0  146.285712
19  2688.0  2688.0  2688.0  117.439807
20  2816.0  2816.0  2816.0  128.277083
21  2944.0  2944.0  2944.0  140.383190
22  3072.0  3072.0  3072.0  143.896072
23  3200.0  3200.0  3200.0  139.433550
24  3328.0  3328.0  3328.0  131.852184
25  3456.0  3456.0  3456.0  139.002705
26  3584.0  3584.0  3584.0  148.866543
27  3712.0  3712.0  3712.0  141.698358
28  3840.0  3840.0  3840.0  137.895263
29  3968.0  3968.0  3968.0  147.016795
30  4096.0  4096.0  4096.0  154.985826
```

# Low-Memory Dropout[¶](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#low-memory-dropout "Link to this heading")

In this tutorial, you will write a memory-efficient implementation of dropout whose state will be composed of a single int32 seed. This differs from more traditional implementations of dropout, whose state is generally composed of a bit mask tensor of the same shape as the input.

In doing so, you will learn about:

- The limitations of naive implementations of Dropout with PyTorch.
    
- Parallel pseudo-random number generation in Triton.
    

## Baseline[¶](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#baseline "Link to this heading")

The _dropout_ operator was first introduced in [[SRIVASTAVA2014]](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#srivastava2014) as a way to improve the performance of deep neural networks in low-data regime (i.e. regularization).

It takes a vector as input and produces a vector of the same shape as output. Each scalar in the output has a probability p of being changed to zero and otherwise it is copied from the input. This forces the network to perform well even when only 1−p scalars from the input are available.

At evaluation time we want to use the full power of the network so we set p=0. Naively this would increase the norm of the output (which can be a bad thing, e.g. it can lead to artificial decrease in the output softmax temperature). To prevent this we multiply the output by 11−p, which keeps the norm consistent regardless of the dropout probability.

Let’s first take a look at the baseline implementation.

import tabulate
import torch

import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

# Input tensor
x = torch.randn(size=(10, )).cuda()
# Dropout mask
p = 0.5
x_keep = (torch.rand(size=(10, )) > p).to(torch.int32).cuda()
#
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))

/home/runner/_work/triton/triton/python/triton/language/semantic.py:1598: UserWarning: tl.where with a non-boolean condition is deprecated and will error out in a future triton release. Got int32
  warnings.warn(
---------  -------  ---------  --------  --------  --------  --------  --------  --------  ---------  ---------
input      1.541    -0.293429  -2.17879  0.568431  -1.08452  -1.3986   0.403347  0.838026  -0.719258  -0.403344
keep mask  1         1          0        1          0         1        1         0          0          0
output     3.08199  -0.586858   0        1.13686    0        -2.79719  0.806694  0          0          0
---------  -------  ---------  --------  --------  --------  --------  --------  --------  ---------  ---------

## Seeded dropout[¶](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#seeded-dropout "Link to this heading")

The above implementation of dropout works fine, but it can be a bit awkward to deal with. Firstly we need to store the dropout mask for backpropagation. Secondly, dropout state management can get very tricky when using recompute/checkpointing (e.g. see all the notes about preserve_rng_state in [https://pytorch.org/docs/stable/checkpoint.html](https://pytorch.org/docs/stable/checkpoint.html)). In this tutorial we’ll describe an alternative implementation that (1) has a smaller memory footprint; (2) requires less data movement; and (3) simplifies the management of persisting randomness across multiple invocations of the kernel.

Pseudo-random number generation in Triton is simple! In this tutorial we will use the `triton.language.rand` function which generates a block of uniformly distributed `float32` values in [0, 1), given a seed and a block of `int32` offsets. But if you need it, Triton also provides other [random number generation strategies](https://triton-lang.org/main/python-api/triton.language.html#random-number-generation).

Note

Triton’s implementation of PRNG is based on the Philox algorithm (described on [[SALMON2011]](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#salmon2011)).

Let’s put it all together.

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

x = torch.randn(size=(10, )).cuda()
# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))

-------------------  ---------  --------  --------  -------  --------  --------  ---------  ---------  ---------  ---------
input                -0.952835  0.371721  0.408716  1.42142  0.149397  -0.67086  -0.214186  -0.431969  -0.707878  -0.106434
output (seed = 123)   0         0.743443  0         0        0         -1.34172   0          0         -1.41576   -0.212868
output (seed = 123)   0         0.743443  0         0        0         -1.34172   0          0         -1.41576   -0.212868
output (seed = 512)   0         0         0.817432  2.84284  0         -1.34172  -0.428372   0          0          0
-------------------  ---------  --------  --------  -------  --------  --------  ---------  ---------  ---------  ---------

Et Voilà! We have a triton kernel that applies the same dropout mask provided the seed is the same! If you’d like explore further applications of pseudorandomness in GPU programming, we encourage you to explore the python/triton/language/random.py!

## Exercises[¶](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#exercises "Link to this heading")

1. Extend the kernel to operate over a matrix and use a vector of seeds - one per row.
    
2. Add support for striding.
    
3. (challenge) Implement a kernel for sparse Johnson-Lindenstrauss transform which generates the projection matrix on the fly each time using a seed.
    

## References[¶](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#references "Link to this heading")

[[SALMON2011](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#id2)]

John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, “Parallel Random Numbers: As Easy as 1, 2, 3”, 2011

[[SRIVASTAVA2014](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#id1)]

Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting”, JMLR 2014