---
completed: true
version: 2.6.0
---
Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.
>  不同的 PyTorch 发布、提交，或者不同的平台不能保证完全可复现的结果
>  CPU 和 GPU 执行的结果也可能是不能互相复现的

However, there are some steps you can take to limit the number of sources of nondeterministic behavior for a specific platform, device, and PyTorch release. First, you can control sources of randomness that can cause multiple executions of your application to behave differently. Second, you can configure PyTorch to avoid using nondeterministic algorithms for some operations, so that multiple calls to those operations, given the same inputs, will produce the same result.
>  用于控制特定平台上的不确定性的可以做的步骤包括：1. 控制随机性 2. 配置 PyTorch 以避免特定运算的不确定性算法执行，确保给定相同输入产生相同输出

> [! Warning]
> Deterministic operations are often slower than nondeterministic operations, so single-run performance may decrease for your model. However, determinism may save time in development by facilitating experimentation, debugging, and regression testing.

## Controlling sources of randomness
### PyTorch random number generator
You can use [`torch.manual_seed()`](https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed "torch.manual_seed") to seed the RNG for all devices (both CPU and CUDA):
>  `torch.maunal_seed()` 用于控制所有设备的随机种子

```python
import torch
torch.manual_seed(0)
```

Some PyTorch operations may use random numbers internally. [`torch.svd_lowrank()`](https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html#torch.svd_lowrank "torch.svd_lowrank") does this, for instance. Consequently, calling it multiple times back-to-back with the same input arguments may give different results. However, as long as [`torch.manual_seed()`](https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed "torch.manual_seed") is set to a constant at the beginning of an application and all other sources of nondeterminism have been eliminated, the same series of random numbers will be generated each time the application is run in the same environment.

It is also possible to obtain identical results from an operation that uses random numbers by setting [`torch.manual_seed()`](https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed "torch.manual_seed") to the same value between subsequent calls.

>  一些 PyTorch 运算本身会使用随机数，例如 `torch.svd_lowrand()` 
>  因此连续地用相同的输入调用这些运算会得到不同的结果，但只要保证在运算的开始之前设定好 `torch.maunal_seed()`，在相同的环境下，应用运行时 PyTorch 生成的随机数序列就是相同的

### Python
For custom operators, you might need to set python seed as well:

```python
import random
random.seed(0)
```

>  对于自定义的算子，可能还需要设定 python 随机种子，如上所示

### Random number generators in other libraries
If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG with:

```python
import numpy as np
np.random.seed(0)
```

However, some applications and libraries may use NumPy Random Generator objects, not the global RNG ([https://numpy.org/doc/stable/reference/random/generator.html](https://numpy.org/doc/stable/reference/random/generator.html)), and those will need to be seeded consistently as well.

>  如果使用了 NumPy 或任何依赖于 NumPy 的库，可以如上例设定 NumPy 的全局随机数生成器
>  一些应用或库使用 NumPy 随机生成器对象，而不是全局随机数生成器，这些生成器对象也需要设定种子

If you are using any other libraries that use random number generators, refer to the documentation for those libraries to see how to set consistent seeds for them.

### CUDA convolution benchmarking
The cuDNN library, used by CUDA convolution operations, can be a source of nondeterminism across multiple executions of an application. When a cuDNN convolution is called with a new set of size parameters, an optional feature can run multiple convolution algorithms, benchmarking them to find the fastest one. Then, the fastest algorithm will be used consistently during the rest of the process for the corresponding set of size parameters. Due to benchmarking noise and different hardware, the benchmark may select different algorithms on subsequent runs, even on the same machine.
>  CUDA 卷积运算使用的是 cuDNN 库，该库也会导致程序执行存在不确定性
>  cuDNN 卷积被用一组新的尺寸参数调用时，会运行多个卷积算法，进行基准测试，找到最快的一个，然后在后续对于相同的尺寸参数都使用该算法。因为基准测试存在噪声，同时不同硬件存在差异，即便在相同机器上，多次的运行下，基准测试都可能选择出不同的算法

Disabling the benchmarking feature with `torch.backends.cudnn.benchmark = False` causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
>  要取消这一特性，设定 `torch.backends.cudnn.benchmark = False` ，令 cuDNN 确定性地选择一个算法，当然可能导致性能下降

However, if you do not need reproducibility across multiple executions of your application, then performance might improve if the benchmarking feature is enabled with `torch.backends.cudnn.benchmark = True`.

Note that this setting is different from the `torch.backends.cudnn.deterministic` setting discussed below.

## Avoiding nondeterministic algorithms
[`torch.use_deterministic_algorithms()`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms "torch.use_deterministic_algorithms") lets you configure PyTorch to use deterministic algorithms instead of nondeterministic ones where available, and to throw an error if an operation is known to be nondeterministic (and without a deterministic alternative).
>  `torch.use_deterministic_algorithms()` 用于配置 PyTorch 在确定性算法可用时都是用确定性算法而不是非确定性算法，且如果在确定某个运算是不确定性时抛出错误

Please check the documentation for [`torch.use_deterministic_algorithms()`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms "torch.use_deterministic_algorithms") for a full list of affected operations. If an operation does not act correctly according to the documentation, or if you need a deterministic implementation of an operation that does not have one, please submit an issue: [https://github.com/pytorch/pytorch/issues?q=label:%22module:%20determinism%22](https://github.com/pytorch/pytorch/issues?q=label:%22module:%20determinism%22)

For example, running the nondeterministic CUDA implementation of [`torch.Tensor.index_add_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_ "torch.Tensor.index_add_") will throw an error:

```
>>> import torch
>>> torch.use_deterministic_algorithms(True)
>>> torch.randn(2, 2).cuda().index_add_(0, torch.tensor([0, 1]), torch.randn(2, 2))
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
RuntimeError: index_add_cuda_ does not have a deterministic implementation, but you set
'torch.use_deterministic_algorithms(True)'. ...
```

When [`torch.bmm()`](https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm "torch.bmm") is called with sparse-dense CUDA tensors it typically uses a nondeterministic algorithm, but when the deterministic flag is turned on, its alternate deterministic implementation will be used:

```
>>> import torch
>>> torch.use_deterministic_algorithms(True)
>>> torch.bmm(torch.randn(2, 2, 2).to_sparse().cuda(), torch.randn(2, 2, 2).cuda())
tensor([[[ 1.1900, -2.3409],
         [ 0.4796,  0.8003]],
        [[ 0.1509,  1.8027],
         [ 0.0333, -1.1444]]], device='cuda:0')
```

Furthermore, if you are using CUDA tensors, and your CUDA version is 10.2 or greater, you should set the environment variable CUBLAS_WORKSPACE_CONFIG according to CUDA documentation: [https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)

### CUDA convolution determinism
While disabling CUDA convolution benchmarking (discussed above) ensures that CUDA selects the same algorithm each time an application is run, that algorithm itself may be nondeterministic, unless either `torch.use_deterministic_algorithms(True)` or `torch.backends.cudnn.deterministic = True` is set. The latter setting controls only this behavior, unlike [`torch.use_deterministic_algorithms()`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms "torch.use_deterministic_algorithms") which will make other PyTorch operations behave deterministically, too.
>  取消了 CUDA 卷积基准测试特性可以确保 CUDA 在每次应用运行时选择相同的算法，但该算法本身可能是非确定性的，除非设定了 `torch.use_deterministic_algorithms(True)` 或 `torch.backends.cudnn.deterministic = True`

### CUDA RNN and LSTM
In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior. See [`torch.nn.RNN()`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN "torch.nn.RNN") and [`torch.nn.LSTM()`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM "torch.nn.LSTM") for details and workarounds.

### Filling uninitialized memory
Operations like [`torch.empty()`](https://pytorch.org/docs/stable/generated/torch.empty.html#torch.empty "torch.empty") and [`torch.Tensor.resize_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html#torch.Tensor.resize_ "torch.Tensor.resize_") can return tensors with uninitialized memory that contain undefined values. Using such a tensor as an input to another operation is invalid if determinism is required, because the output will be nondeterministic. But there is nothing to actually prevent such invalid code from being run. So for safety, [`torch.utils.deterministic.fill_uninitialized_memory`](https://pytorch.org/docs/stable/deterministic.html#torch.utils.deterministic.fill_uninitialized_memory "torch.utils.deterministic.fill_uninitialized_memory") is set to `True` by default, which will fill the uninitialized memory with a known value if `torch.use_deterministic_algorithms(True)` is set. This will prevent the possibility of this kind of nondeterministic behavior.
>  类似 `torch.empty()` 和 `torch.Tensor.resize_()` 会返回带有未定义值张量 (因为占据了未初始化的内存)，为了安全起见，`torch.utils.deterministic.fill_uinitialized_memory` 默认被设定为 `True`，这使得在 `torch.use_determinstic_algorithms(True)` 被设定时，未初始化的内存都会被设定为一个已知的值

However, filling uninitialized memory is detrimental to performance. So if your program is valid and does not use uninitialized memory as the input to an operation, then this setting can be turned off for better performance.
>  注意填充未初始化的内存也会影响性能，因此如果程序没有用到未初始化的内存作为运算的输入，将该设定关闭可以提高性能

## DataLoader
DataLoader will reseed workers following [Randomness in multi-process data loading](https://pytorch.org/docs/stable/data.html#data-loading-randomness) algorithm. Use `worker_init_fn()` and generator to preserve reproducibility:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g,
)
```
