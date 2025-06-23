---
version: "12.9"
---
# Overview
CUDA is a parallel computing platform and programming model developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU. It allows developers to accelerate compute-intensive applications using C, C++, and Fortran, and is widely adopted in fields such as deep learning, scientific computing, and high-performance computing (HPC).

# What Is the CUDA C Programming Guide?
The CUDA C Programming Guide is the official, comprehensive resource that explains how to write programs using the CUDA platform. It provides detailed documentation of the CUDA architecture, programming model, language extensions, and performance guidelines. Whether you’re just getting started or optimizing complex GPU kernels, this guide is an essential reference for effectively leveraging CUDA’s full capabilities.

# Introduction
## The Benefits of Using GPUs
The Graphics Processing Unit (GPU) [1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fn1) provides much higher instruction throughput and memory bandwidth than the CPU within a similar price and power envelope. Many applications leverage these higher capabilities to run faster on the GPU than on the CPU (see [GPU Applications](https://www.nvidia.com/object/gpu-applications.html)). Other computing devices, like FPGAs, are also very energy efficient, but offer much less programming flexibility than GPUs.

This difference in capabilities between the GPU and the CPU exists because they are designed with different goals in mind. While the CPU is designed to excel at executing a sequence of operations, called a _thread_, as fast as possible and can execute a few tens of these threads in parallel, the GPU is designed to excel at executing thousands of them in parallel (amortizing the slower single-thread performance to achieve greater throughput).

The GPU is specialized for highly parallel computations and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control. The schematic [Figure 1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing) shows an example distribution of chip resources for a CPU versus a GPU.
> GPU 聚焦于高度并行的计算，因此相较于 CPU，它的更多晶体管被设计用于数据处理而不是数据缓存和流程控制
> GPU 和 CPU 的芯片资源的分布差异见 Figure 1

[![The GPU Devotes More Transistors to Data Processing](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)

Figure 1 The GPU Devotes More Transistors to Data Processing

Devoting more transistors to data processing, for example, floating-point computations, is beneficial for highly parallel computations; the GPU can hide memory access latencies with computation, instead of relying on large data caches and complex flow control to avoid long memory access latencies, both of which are expensive in terms of transistors.
> GPU 通过计算来隐藏 memory access latency，而不是像 CPU 通过大的数据缓存和复杂的流程控制来隐藏 memory access latency

In general, an application has a mix of parallel parts and sequential parts, so systems are designed with a mix of GPUs and CPUs in order to maximize overall performance. Applications with a high degree of parallelism can exploit this massively parallel nature of the GPU to achieve higher performance than on the CPU.
> 具有高度并行性质的应用在 GPU 上会取得比在 CPU 上显著更好的表现

## CUDA®: A General-Purpose Parallel Computing Platform and Programming Model
In November 2006, NVIDIA® introduced CUDA®, a general purpose parallel computing platform and programming model that leverages the parallel compute engine in NVIDIA GPUs to solve many complex computational problems in a more efficient way than on a CPU.
> CUDA 是一个通用目的的并行计算平台和编程模型
> CUDA 利用 NVIDIA GPUs 的计算引擎来解决计算问题

CUDA comes with a software environment that allows developers to use C++ as a high-level programming language. As illustrated by [Figure 2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-general-purpose-parallel-computing-architecture-cuda-is-designed-to-support-various-languages-and-application-programming-interfaces), other languages, application programming interfaces, or directives-based approaches are supported, such as FORTRAN, DirectCompute, OpenACC.

[![GPU Computing Applications. CUDA is designed to support various languages and application programming interfaces.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-computing-applications.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-computing-applications.png)

Figure 2 GPU Computing Applications. CUDA is designed to support various languages and application programming interfaces.

## A Scalable Programming Model
The advent of multicore CPUs and manycore GPUs means that mainstream processor chips are now parallel systems. The challenge is to develop application software that transparently scales its parallelism to leverage the increasing number of processor cores, much as 3D graphics applications transparently scale their parallelism to manycore GPUs with widely varying numbers of cores.

The CUDA parallel programming model is designed to overcome this challenge while maintaining a low learning curve for programmers familiar with standard programming languages such as C.
> 可拓展的应用程序可以随着处理器核心数量的增长透明地拓展自身的并行性
> CUDA 并行编程模型的设计目的就是编写可拓展的应用程序

At its core are three key abstractions — a hierarchy of thread groups, shared memories, and barrier synchronization — that are simply exposed to the programmer as a minimal set of language extensions.
> CUDA 的三大核心抽象是：
> thread group 层次结构
> shared memory
> barrier 同步
> 这三大核心抽象都通过语言拓展的方式暴露给程序员

These abstractions provide fine-grained data parallelism and thread parallelism, nested within coarse-grained data parallelism and task parallelism. They guide the programmer to partition the problem into coarse sub-problems that can be solved independently in parallel by blocks of threads, and each sub-problem into finer pieces that can be solved cooperatively in parallel by all threads within the block.
> 这些抽象引导程序员将问题先划分为粗粒度的子问题
> 这些子问题可以各自并行地由 thread block 解决，而每个子问题被更细粒度地划分给 block 内的 thread 协作解决

This decomposition preserves language expressivity by allowing threads to cooperate when solving each sub-problem, and at the same time enables automatic scalability. Indeed, each block of threads can be scheduled on any of the available multiprocessors within a GPU, in any order, concurrently or sequentially, so that a compiled CUDA program can execute on any number of multiprocessors as illustrated by [Figure 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#scalable-programming-model-automatic-scalability), and only the runtime system needs to know the physical multiprocessor count.
> 这种分解方式通过允许线程协作解决子问题保留了语言表达性，同时启用了自动的可拓展性
> 每个 thread block 可以被调度到任意可用的 SM，顺序任意，也就是编译好的 CUDA 程序可以在任意的 SM 上执行，仅 runtime 系统需要知道具体是哪个 SM

This scalable programming model allows the GPU architecture to span a wide market range by simply scaling the number of multiprocessors and memory partitions: from the high-performance enthusiast GeForce GPUs and professional Quadro and Tesla computing products to a variety of inexpensive, mainstream GeForce GPUs (see [CUDA-Enabled GPUs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus) for a list of all CUDA-enabled GPUs).
> 这样可拓展的编程模型使得 CUDA 程序可以在高端和低端的 GPU 上都可以运行

![Automatic Scalability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/automatic-scalability.png)

Figure 3 Automatic Scalability

Note
A GPU is built around an array of Streaming Multiprocessors (SMs) (see [Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation) for more details). A multithreaded program is partitioned into blocks of threads that execute independently from each other, so that a GPU with more multiprocessors will automatically execute the program in less time than a GPU with fewer multiprocessors.

# Revision History

Table 1 Revision History

| Version | Changes                                                                                                                                                                                                                              |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 12.9    | Added section [Error Log Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-log-management) and CUDA_LOG_FILE to [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#env-vars) |
| 12.8    | Added section [TMA Swizzle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tma-swizzle)                                                                                                                                      |

[1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id2) The _graphics_ qualifier comes from the fact that when the GPU was originally created, two decades ago, it was designed as a specialized processor to accelerate graphics rendering. Driven by the insatiable market demand for real-time, high-definition, 3D graphics, it has evolved into a general processor used for many more workloads than just graphics rendering.

# Programming Model
This chapter introduces the main concepts behind the CUDA programming model by outlining how they are exposed in C++.
> 本章通过阐述 CUDA 编程模型的主要概念是如何暴露给 C++的来介绍 CUDA 编程模型的主要概念

An extensive description of CUDA C++ is given in [Programming Interface](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface).

Full code for the vector addition example used in this chapter and the next can be found in the [vectorAdd CUDA sample](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition).

## Kernels
CUDA C++ extends C++ by allowing the programmer to define C++ functions, called _kernels_, that, when called, are executed N times in parallel by N different _CUDA threads_, as opposed to only once like regular C++ functions.
> CUDA 允许程序员定义称为 kernel 的 C++ 函数
> 普通 C++ 函数仅被执行一次，kernel 会被 N 个不同的 CUDA 线程并行执行 N 次

A kernel is defined using the `__global__` declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new `<<<...>>>` _execution configuration_ syntax (see [C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)). Each thread that executes the kernel is given a unique _thread ID_ that is accessible within the kernel through built-in variables.
> kernel 通过 `__global__` 声明指示符定义
> kernel 调用时需要通过 `<<<...>>>` 给定执行配置，指定执行 kernel 的 CUDA 线程数量
> 每个执行 kernel 的 thread 都具有唯一的 thread ID，thread ID 可以在 kernel 中通过内建变量 `threadIdx` 访问

As an illustration, the following sample code, using the built-in variable `threadIdx`, adds two vectors _A_ and _B_ of size _N_ and stores the result into vector _C_:

```cpp
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

Here, each of the _N_ threads that execute `VecAdd()` performs one pair-wise addition.

## Thread Hierarchy
For convenience, `threadIdx` is a 3-component vector, so that threads can be identified using a one-dimensional, two-dimensional, or three-dimensional _thread index_, forming a one-dimensional, two-dimensional, or three-dimensional block of threads, called a _thread block_. This provides a natural way to invoke computation across the elements in a domain such as a vector, matrix, or volume.
> `threadIdx` 是一个三成员 vector，指定 thread 的索引，因此 thread 可以被一维/二维/三维索引指定，对应一维/二维/三维的 thread block

The index of a thread and its thread ID relate to each other in a straightforward way: For a one-dimensional block, they are the same; for a two-dimensional block of size _(Dx, Dy)_, the thread ID of a thread of index _(x, y)_ is _(x + y Dx)_; for a three-dimensional block of size _(Dx, Dy, Dz)_, the thread ID of a thread of index _(x, y, z)_ is _(x + y Dx + z Dx Dy)_.
> thread ID 和 thread 索引的关系：
> 对于一维 thread block，二者相等
> 对于二维 thread block of size (Dx, Dy)，索引为 (x, y) 的 thread 的 ID 为 (x + y Dx)，即 x 为列索引 (横向)，y 为行索引 (纵向)
> 对于三维 thread block of size (Dx, Dy, Dz)，索引为 (x, y, z) 的 thread 的 ID 为 (x + y Dx + z Dx Dy)，即 x 为二维块中的列索引，y 为二维块中的行索引，z 为二维块的索引

As an example, the following code adds two matrices _A_ and _B_ of size _NxN_ and stores the result into matrix _C_:

```cpp
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited memory resources of that core. On current GPUs, a thread block may contain up to 1024 threads.
> SM 以 thread block 为单位进行调度，因此一个 thread block 内的所有线程都会驻留在相同的 SM 核心上，共享该核心的内存资源
> 因此每个 thread block 的 thread 数量存在上限，目前被设定为 1024 threads

However, a kernel can be executed by multiple equally-shaped thread blocks, so that the total number of threads is equal to the number of threads per block times the number of blocks.
> kernel 可以被多个相同形状的 thread block 同时执行
> 因此同时执行 kernel 的 thread 数量等于每个 thread block 内 thread 的数量乘以 thread block 的数量

Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional _grid_ of thread blocks as illustrated by [Figure 4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy-grid-of-thread-blocks). The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
> thread block 可以是一维、二维、三维的 thread 组合
> grid 则可以是一维、二维、三维的 thread block 组合
> 一般 grid 的形状、大小具体根据需要处理的数据决定，整个 grid 包含的 thread block 数量一般也会超过 GPU 总的 SM 数量

[![Grid of Thread Blocks](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)
Figure 4 Grid of Thread Blocks

The number of threads per block and the number of blocks per grid specified in the `<<<...>>>` syntax can be of type `int` or `dim3`. Two-dimensional blocks or grids can be specified as in the example above.
> `<<<...>>>` 语法中指定的 grid 形状和 thread block 形状的类型可以是 `int` 或 `dim3`

Each block within the grid can be identified by a one-dimensional, two-dimensional, or three-dimensional unique index accessible within the kernel through the built-in `blockIdx` variable. The dimension of the thread block is accessible within the kernel through the built-in `blockDim` variable.
> 在 kernel 中，grid 内的 block 索引通过内建变量 `blockIdx` 访问，block 自身的维度通过内建变量 `blockDim` 访问

Extending the previous `MatAdd()` example to handle multiple blocks, the code becomes as follows.

```cpp
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

A thread block size of 16x16 (256 threads), although arbitrary in this case, is a common choice. The grid is created with enough blocks to have one thread per matrix element as before. For simplicity, this example assumes that the number of threads per grid in each dimension is evenly divisible by the number of threads per block in that dimension, although that need not be the case.
> 本例中，每个 thread 处理一个矩阵元素，将矩阵划分为 grid of thread blocks

Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. This independence requirement allows thread blocks to be scheduled in any order across any number of cores as illustrated by [Figure 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#scalable-programming-model-automatic-scalability), enabling programmers to write code that scales with the number of cores.
> thread blocks 需要能够独立执行，也就是可以按照任意顺序执行
> 因此 thread blocks 可以在任意数量的核心上被按照任意顺序被调度，故程序可以随着核心的数量而扩展

Threads within a block can cooperate by sharing data through some _shared memory_ and by synchronizing their execution to coordinate memory accesses. More precisely, one can specify synchronization points in the kernel by calling the `__syncthreads()` intrinsic function; `__syncthreads()` acts as a barrier at which all threads in the block must wait before any is allowed to proceed. [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) gives an example of using shared memory. In addition to `__syncthreads()`, the [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) provides a rich set of thread-synchronization primitives.
> thread block 内的 threads 可以通过 shared memory 共享数据，而 thread 之间的内存访问需要进行同步和协调
> kernel 中，内置函数 `__syncthreads()` 的功能就是同步 thread block 内的所有线程

For efficient cooperation, the shared memory is expected to be a low-latency memory near each processor core (much like an L1 cache) and `__syncthreads()` is expected to be lightweight.

### Thread Block Clusters
With the introduction of NVIDIA [Compute Capability 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-9-0), the CUDA programming model introduces an optional level of hierarchy called Thread Block Clusters that are made up of thread blocks. Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.
> Compute Capability 9.0 之后，CUDA 开始支持一层新的结构层次，即 thread block cluster
> thread block 内的 threads 一定会被一起调度到同一个流式多处理器上
> thread block cluster 内的 thread blocks 一定会被一起调度到同一个 GPU Processing Cluster 上

Similar to thread blocks, clusters are also organized into a one-dimension, two-dimension, or three-dimension as illustrated by [Grid of Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters-grid-of-clusters). The number of thread blocks in a cluster can be user-defined, and a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA. Note that on GPU hardware or MIG configurations which are too small to support 8 multiprocessors the maximum cluster size will be reduced accordingly. Identification of these smaller configurations, as well as of larger configurations supporting a thread block cluster size beyond 8, is architecture-specific and can be queried using the `cudaOccupancyMaxPotentialClusterSize` API.
> thread block cluster 的形状也可以是一维、二维、三维的
> cluster 内的 thread block 数量由用户定义，在可移植的 CUDA 程序中最大为 8
> 如果 GPU 对 cluster 支持的最多 SM 数量小于 8，则会相应减少

[![Grid of Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png)

Figure 5 Grid of Thread Block Clusters

Note
In a kernel launched using cluster support, the gridDim variable still denotes the size in terms of number of thread blocks, for compatibility purposes. The rank of a block in a cluster can be found using the [Cluster Group](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group-cg) API.
> 注意
> 使用了 thread block cluster 不会改变变量 gridDim 的语义，也就是仍然表示 grid 中各维度 thread blocks 的数量
> thread block 在 cluster 内的 rank 则通过 Cluster Group API 获取

A thread block cluster can be enabled in a kernel either using a compiler time kernel attribute using `__cluster_dims__(X,Y,Z)` or using the CUDA kernel launch API `cudaLaunchKernelEx`. 
> thread block cluster 可以通过以下两种方式启用：
> 编译时 kernel 属性 `__cluster__dims__(X,Y,Z)`
> CUDA kernel launch API `cudaLaunchKernelEx`

The example below shows how to launch a cluster using compiler time kernel attribute. The cluster size using kernel attribute is fixed at compile time and then the kernel can be launched using the classical `<<< , >>>`. If a kernel uses compile-time cluster size, the cluster size cannot be modified when launching the kernel.
> 使用 kernel attribute `__cluster__dims__(X,Y,Z)` 时，cluster 形状需要在编译时确定，在运行时发起 kernel 时，无法修改编译时确定的 cluster 形状
> 被 `__cluster__dims__(X,Y,Z)` 修饰的 kernel 仍然可以通过经典的 `<<<,>>>` 语法发起，一个额外的要求是 gridDim 需要时 cluster size 的整数倍

```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

A thread block cluster size can also be set at runtime and the kernel can be launched using the CUDA kernel launch API `cudaLaunchKernelEx`. The code example below shows how to launch a cluster kernel using the extensible API.
> 未使用 `__cluster_dims__(X,Y,Z)` 属性修饰的 kernel 可以通过 CUDA kernel launch API 在运行时设定 cluster size

```cpp
// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

In GPUs with compute capability 9.0, all the thread blocks in the cluster are guaranteed to be co-scheduled on a single GPU Processing Cluster (GPC) and allow thread blocks in the cluster to perform hardware-supported synchronization using the [Cluster Group](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group-cg) API `cluster.sync()`. Cluster group also provides member functions to query cluster group size in terms of number of threads or number of blocks using `num_threads()` and `num_blocks()` API respectively. The rank of a thread or block in the cluster group can be queried using `dim_threads()` and `dim_blocks()` API respectively.
> cluster 内的所有 thread blocks 一定会被共同调度到同一个 GPU 处理簇上，并且 cluster 内的所有 thread blocks 可以通过 `cluster.sync()` 同步
> `num_threads()` 和 `num_blocks()` 分别用于查询 cluster 内的 thread 数量和 thread block 数量
> `dim_thread()` 和 `dim_blocks()` 分别用于查询当前 thread 和 thread block 在 cluster 内的 rank

Thread blocks that belong to a cluster have access to the Distributed Shared Memory. Thread blocks in a cluster have the ability to read, write, and perform atomics to any address in the distributed shared memory. [Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory) gives an example of performing histograms in distributed shared memory.
> 属于同一个 cluster 的 thread blocks 可以访问分布式 shared memory，即各个 thread block 的 shared memory

## Memory Hierarchy
CUDA threads may access data from multiple memory spaces during their execution as illustrated by [Figure 6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy-memory-hierarchy-figure). Each thread has private local memory. Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. Thread blocks in a thread block cluster can perform read, write, and atomics operations on each other’s shared memory. All threads have access to the same global memory.
> CUDA thread 在执行时可以访问多层次的存储空间，见 Figure 6
> 每个 thread 可以访问：
> thread 级别的 local memory
> thread block 级别的 shared memory
> thread block cluster 级别的 distributed shared memory
> grid 级别的 global memory

There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces. The global, constant, and texture memory spaces are optimized for different memory usages (see [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)). Texture memory also offers different addressing modes, as well as data filtering, for some specific data formats (see [Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)).
> 所有的 thread 还可以访问只读的 constant memory 和 texture memory 空间
> global、constant、texture memory 空间各自为不同的访存模式做出了优化
> texture memory 还提供了不同的寻址模式，以及针对特定数据格式的数据过滤

The global, constant, and texture memory spaces are persistent across kernel launches by the same application.
> global, constant, texture memory 空间的声明周期和 kernel launch 相同

![Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png)

Figure 6 Memory Hierarchy

## Heterogeneous Programming
As illustrated by [Figure 7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming-heterogeneous-programming), the CUDA programming model assumes that the CUDA threads execute on a physically separate _device_ that operates as a coprocessor to the _host_ running the C++ program. This is the case, for example, when the kernels execute on a GPU and the rest of the C++ program executes on a CPU.
> Figure 7 展示了 CUDA 编程模型假设了 CUDA threads 在执行时所处于的物理设备是和执行 C++ 程序的主机分离的
> 例如，kernel 在 GPU 上执行，剩余的 C++ 程序在 CPU 上执行

The CUDA programming model also assumes that both the host and the device maintain their own separate memory spaces in DRAM, referred to as _host memory_ and _device memory_, respectively. Therefore, a program manages the global, constant, and texture memory spaces visible to kernels through calls to the CUDA runtime (described in [Programming Interface](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)). This includes device memory allocation and deallocation as well as data transfer between host and device memory.
> CUDA 编程模型同时假设主机和设备各自维护分离的 DRAM 存储空间
> 因此，程序需要通过对 CUDA runtime 的调用来管理对于 kernel 可见的 global, constant, texture memory 空间，包括设备内存的分配与释放以及主机和设备内存之间的传输

Unified Memory provides _managed memory_ to bridge the host and device memory spaces. Managed memory is accessible from all CPUs and GPUs in the system as a single, coherent memory image with a common address space. This capability enables oversubscription of device memory and can greatly simplify the task of porting applications by eliminating the need to explicitly mirror data on host and device. See [Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd) for an introduction to Unified Memory.
>统一内存提供了“托管内存”以连接主机和设备内存空间
>托管内存提供了一个具有公共地址空间的单一、一致的内存映像，可以被系统中的所有CPU和GPU访问，该功能允许设备内存的超额订阅，并且通过消除了显式地在主机和设备上镜像数据的需求来显著简化了将应用程序移植的任务

![Heterogeneous Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/heterogeneous-programming.png)

Figure 7 Heterogeneous Programming

Note
Serial code executes on the host while parallel code executes on the device.

## Asynchronous SIMT Programming Model
In the CUDA programming model a thread is the lowest level of abstraction for doing a computation or a memory operation. Starting with devices based on the NVIDIA Ampere GPU architecture, the CUDA programming model provides acceleration to memory operations via the asynchronous programming model. The asynchronous programming model defines the behavior of asynchronous operations with respect to CUDA threads.
> CUDA 编程模型中，thread 是执行计算或内存操作的最低抽象层次
> Ampere 架构开始，CUDA 编程模型通过异步编程模型提供了对内存操作的加速
> 异步编程模型定义了 CUDA thread 的异步操作行为

The asynchronous programming model defines the behavior of [Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier) for synchronization between CUDA threads. The model also explains and defines how [cuda::memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies) can be used to move data asynchronously from global memory while computing in the GPU.
> 异步编程模型还定义了用于同步 CUDA threads 的异步屏障
> 异步编程模型还定义了可以如何使用 `cuda::memcpy_async` 来在 GPU 执行计算的同时异步从 global memory 中移动数据

### Asynchronous Operations
An asynchronous operation is defined as an operation that is initiated by a CUDA thread and is executed asynchronously as-if by another thread. In a well formed program one or more CUDA threads synchronize with the asynchronous operation. The CUDA thread that initiated the asynchronous operation is not required to be among the synchronizing threads.
> 异步操作的定义为由一个 CUDA thread 发起的操作，该操作可以被视为似乎是由另一个 thread 异步执行的
> 在结构良好的程序中，一个或者多个 CUDA threads 需要和异步操作同步，发起异步操作的 CUDA thread 并不要求一定要参与同步

Such an asynchronous thread (an as-if thread) is always associated with the CUDA thread that initiated the asynchronous operation. An asynchronous operation uses a synchronization object to synchronize the completion of the operation. Such a synchronization object can be explicitly managed by a user (e.g., `cuda::memcpy_async`) or implicitly managed within a library (e.g., `cooperative_groups::memcpy_async`).
> 发起异步操作的 CUDA thread 总是和一个异步 thread (似乎在执行操作的 thread) 关联
> 异步操作使用同步对象来同步操作的完成，这类同步对象可以被用户显式管理 (例如 `cuda::memcpy_async` ) 或者在库中被隐式管理 (例如 `cooperative_groups::memcpy_async`)

A synchronization object could be a `cuda::barrier` or a `cuda::pipeline`. These objects are explained in detail in [Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier) and [Asynchronous Data Copies using cuda::pipeline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies). These synchronization objects can be used at different thread scopes. A scope defines the set of threads that may use the synchronization object to synchronize with the asynchronous operation. The following table defines the thread scopes available in CUDA C++ and the threads that can be synchronized with each.
> 同步对象可以是 `cuda::barrier` 或 `cuda::pipeline`
> 这些同步对象可以在不同的 thread 范围内使用，thread 范围定义了可以使用同步对象和异步操作进行同步的一组 threads
> CUDA 目前提供的可以同步的 thread 范围如下表所示

| Thread Scope                              | Description                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------------- |
| `cuda::thread_scope::thread_scope_thread` | Only the CUDA thread which initiated asynchronous operations synchronizes.                  |
| `cuda::thread_scope::thread_scope_block`  | All or any CUDA threads within the same thread block as the initiating thread synchronizes. |
| `cuda::thread_scope::thread_scope_device` | All or any CUDA threads in the same GPU device as the initiating thread synchronizes.       |
| `cuda::thread_scope::thread_scope_system` | All or any CUDA or CPU threads in the same system as the initiating thread synchronizes.    |

These thread scopes are implemented as extensions to standard C++ in the [CUDA Standard C++](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes) library.
> 这些 thread 范围在 CUDA 标准 C++ 库中被实现，作为标准 C++ 的拓展

## Compute Capability
The _compute capability_ of a device is represented by a version number, also sometimes called its “SM version”. This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.
> 设备的 compute capability 由版本号表示，或者称为 SM 版本
> 该版本号标识了 GPU 硬件支持的特性，应用在运行时使用版本号来确定当前的 GPU 硬件特征和支持的指令

The compute capability comprises a major revision number _X_ and a minor revision number _Y_ and is denoted by _X.Y_.
> compute capability 包括主版本号和子版本号，记作 X.Y

The major revision number indicates the core GPU architecture of a device. Devices with the same major revision number share the same fundamental architecture. The table below lists the major revision numbers corresponding to each NVIDIA GPU architecture.
>  主版本号表示设备的核心 GPU 架构，相同主版本号的设备有相同的核心架构

Table 2 GPU Architecture and Major Revision Numbers

| Major Revision Number | NVIDIA GPU Architecture         |
| --------------------- | ------------------------------- |
| 9                     | NVIDIA Hopper GPU Architecture  |
| 8                     | NVIDIA Ampere GPU Architecture  |
| 7                     | NVIDIA Volta GPU Architecture   |
| 6                     | NVIDIA Pascal GPU Architecture  |
| 5                     | NVIDIA Maxwell GPU Architecture |
| 3                     | NVIDIA Kepler GPU Architecture  |

> 各架构的主版本号如下：
> Hopper - 9
> Ampere - 8
> Volta - 7
> Pascal - 6
> Maxwell - 5
> Kepler - 3

The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new features.
> 子版本号对应于对核心架构的增量式提升，可能会带有新特性

Table 3 Incremental Updates in GPU Architectures

|Compute Capability|NVIDIA GPU Architecture|Based On|
|---|---|---|
|7.5|NVIDIA Turing GPU Architecture|NVIDIA Volta GPU Architecture|

> 版本号为 7.5 的架构为 Turing 架构，该架构是 Volta 架构的增量更新

[CUDA-Enabled GPUs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-enabled-gpus) lists of all CUDA-enabled devices along with their compute capability. [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) gives the technical specifications of each compute capability.

Note
The compute capability version of a particular GPU should not be confused with the CUDA version (for example, CUDA 7.5, CUDA 8, CUDA 9), which is the version of the CUDA _software platform_. The CUDA platform is used by application developers to create applications that run on many generations of GPU architectures, including future GPU architectures yet to be invented. While new versions of the CUDA platform often add native support for a new GPU architecture by supporting the compute capability version of that architecture, new versions of the CUDA platform typically also include software features that are independent of hardware generation.
> 注意
> 设备的 compute capability 不应该和 CUDA 版本混淆
> CUDA 版本是指 CUDA 软件平台的版本，开发者使用 CUDA 平台开发可以运行于多个版本的设备上的程序，包括了未来版本的设备
> 新版本的 CUDA 一般会为新的 GPU 架构添加支持，但一般也会添加独立于硬件代数的软件特性

The _Tesla_ and _Fermi_ architectures are no longer supported starting with CUDA 7.0 and CUDA 9.0, respectively.
> CUDA 7.0, CUDA 9.0 开始分别不再支持 Tesla 和 Fermi 架构

# Programming Interface
CUDA C++ provides a simple path for users familiar with the C++ programming language to easily write programs for execution by the device.
>  CUDA C++ 基于 C++ 提供了编程接口

It consists of a minimal set of extensions to the C++ language and a runtime library.
>  CUDA C++  由对 C++ 语言的拓展集和一个运行时库组成

The core language extensions have been introduced in [Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model). They allow programmers to define a kernel as a C++ function and use some new syntax to specify the grid and block dimension each time the function is called. A complete description of all extensions can be found in [C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions). Any source file that contains some of these extensions must be compiled with `nvcc` as outlined in [Compilation with NVCC](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc).
>  核心的语言拓展在上一节 Programming Model 已经介绍，它们用于将 kernel 定义为 C++ 函数，并指定函数被调用时使用的 grid, block dimension
>  所有的语言拓展见 C++ Language Extensions 小节
>  任意包含了 CUDA C++ 语言拓展的源文件都必须用 `nvcc` 编译

The runtime is introduced in [CUDA Runtime](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-runtime). It provides C and C++ functions that execute on the host to allocate and deallocate device memory, transfer data between host memory and device memory, manage systems with multiple devices, etc. A complete description of the runtime can be found in the CUDA reference manual.
>  CUDA Runtime 为 host 端提供了在设备分配和释放显存、在 host 和设备内存之间迁移数据、管理带有多个设备的系统等等功能的 C/C++ 函数

The runtime is built on top of a lower-level C API, the CUDA driver API, which is also accessible by the application. The driver API provides an additional level of control by exposing lower-level concepts such as CUDA contexts - the analogue of host processes for the device - and CUDA modules - the analogue of dynamically loaded libraries for the device. Most applications do not use the driver API as they do not need this additional level of control and when using the runtime, context and module management are implicit, resulting in more concise code. As the runtime is interoperable with the driver API, most applications that need some driver API features can default to use the runtime API and only use the driver API where needed. The driver API is introduced in [Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api) and fully described in the reference manual.
>  CUDA Runtime 基于 CUDA driver API 构建，应用可以访问 CUDA driver API
>  CUDA driver API 将更低层次的概念例如 CUDA contexts (设备端对于主机进程的抽象), CUDA modules (设备端对于主机动态链接库的抽象)
>  使用 CUDA Runtime 时，对 context, module 的管理是隐式的，大多数需要使用 driver API 的应用可以通过 Runtime API 进行控制，仅在必要时候使用 driver API

## Compilation with NVCC
Kernels can be written using the CUDA instruction set architecture, called _PTX_, which is described in the PTX reference manual. It is however usually more effective to use a high-level programming language such as C++. In both cases, kernels must be compiled into binary code by `nvcc` to execute on the device.
>  kernel 可以用 CUDA C++ 编写，也可以用 CUDA 指令集架构 - PTX 编写
>  无论如何，kernel 都被会被 `nvcc` 编译为二级制码以在设备执行

`nvcc` is a compiler driver that simplifies the process of compiling _C++_ or _PTX_ code: It provides simple and familiar command line options and executes them by invoking the collection of tools that implement the different compilation stages. This section gives an overview of `nvcc` workflow and command options. A complete description can be found in the `nvcc` user manual.
>  `nvcc` 是一个编译器驱动器，用于编译 CUDA C++ 或 PTX 代码
>  `nvcc` 会自行调用对应编译各个阶段的一系列工具完成编译

### Compilation Workflow
#### Offline Compilation
Source files compiled with `nvcc` can include a mix of host code (i.e., code that executes on the host) and device code (i.e., code that executes on the device). `nvcc` ’s basic workflow consists in separating device code from host code and then:

- compiling the device code into an assembly form (_PTX_ code) and/or binary form (_cubin_ object),
- and modifying the host code by replacing the `<<<...>>>` syntax introduced in [Kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) (and described in more details in [Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)) by the necessary CUDA runtime function calls to load and launch each compiled kernel from the _PTX_ code and/or _cubin_ object.

>  `nvcc` 会自行分离源码中的 host code 和 device code，然后
>  - 将 device code 编译为 PTX 码或二进制码 (cubin 对象)
>  - 将 host code 中的 `<<<...>>>` 语法替换为必要的 CUDA Runtime 函数调用，来执行将编译好的 kernel 加载并启动的工作

The modified host code is output either as C++ code that is left to be compiled using another tool or as object code directly by letting `nvcc` invoke the host compiler during the last compilation stage.
>  修改后的 host code 可以输出为 C++ 码，被其他工具继续编译，或者 `nvcc` 直接在最后的编译阶段调用 host 端编译器，输出对象码

Applications can then:

- Either link to the compiled host code (this is the most common case),
- Or ignore the modified host code (if any) and use the CUDA driver API (see [Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api)) to load and execute the _PTX_ code or _cubin_ object.

>  应用可以
>  - 链接到编译好的 host code
>  - 忽略编译好的 host code，直接使用 CUDA driver API 加载并执行 PTX 码或 cubin 对象

#### Just-in-Time Compilation
Any _PTX_ code loaded by an application at runtime is compiled further to binary code by the device driver. This is called _just-in-time compilation_. Just-in-time compilation increases application load time, but allows the application to benefit from any new compiler improvements coming with each new device driver. It is also the only way for applications to run on devices that did not exist at the time the application was compiled, as detailed in [Application Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#application-compatibility).

When the device driver just-in-time compiles some _PTX_ code for some application, it automatically caches a copy of the generated binary code in order to avoid repeating the compilation in subsequent invocations of the application. The cache - referred to as _compute cache_ - is automatically invalidated when the device driver is upgraded, so that applications can benefit from the improvements in the new just-in-time compiler built into the device driver.

Environment variables are available to control just-in-time compilation as described in [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)

As an alternative to using `nvcc` to compile CUDA C++ device code, NVRTC can be used to compile CUDA C++ device code to PTX at runtime. NVRTC is a runtime compilation library for CUDA C++; more information can be found in the NVRTC User guide.

### Binary Compatibility

Binary code is architecture-specific. A _cubin_ object is generated using the compiler option `-code` that specifies the targeted architecture: For example, compiling with `-code=sm_80` produces binary code for devices of [compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability) 8.0. Binary compatibility is guaranteed from one minor revision to the next one, but not from one minor revision to the previous one or across major revisions. In other words, a _cubin_ object generated for compute capability _X.y_ will only execute on devices of compute capability _X.z_ where _z≥y_.

Note

Binary compatibility is supported only for the desktop. It is not supported for Tegra. Also, the binary compatibility between desktop and Tegra is not supported.

### PTX Compatibility

Some _PTX_ instructions are only supported on devices of higher compute capabilities. For example, [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions) are only supported on devices of compute capability 5.0 and above. The `-arch` compiler option specifies the compute capability that is assumed when compiling C++ to _PTX_ code. So, code that contains warp shuffle, for example, must be compiled with `-arch=compute_50` (or higher).

_PTX_ code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability. Note that a binary compiled from an earlier PTX version may not make use of some hardware features. For example, a binary targeting devices of compute capability 7.0 (Volta) compiled from PTX generated for compute capability 6.0 (Pascal) will not make use of Tensor Core instructions, since these were not available on Pascal. As a result, the final binary may perform worse than would be possible if the binary were generated using the latest version of PTX.

_PTX_ code compiled to target [architecture conditional features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#feature-availability) only run on the exact same physical architecture and nowhere else. Arch conditional _PTX_ code is not forward and backward compatible. Example code compiled with `sm_90a` or `compute_90a` only runs on devices with compute capability 9.0 and is not backward or forward compatible.

### Application Compatibility

To execute code on devices of specific compute capability, an application must load binary or _PTX_ code that is compatible with this compute capability as described in [Binary Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#binary-compatibility) and [PTX Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility). In particular, to be able to execute code on future architectures with higher compute capability (for which no binary code can be generated yet), an application must load _PTX_ code that will be just-in-time compiled for these devices (see [Just-in-Time Compilation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation)).

Which _PTX_ and binary code gets embedded in a CUDA C++ application is controlled by the `-arch` and `-code` compiler options or the `-gencode` compiler option as detailed in the `nvcc` user manual. For example,

nvcc x.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_70,code=\"compute_70,sm_70\"

embeds binary code compatible with compute capability 5.0 and 6.0 (first and second `-gencode` options) and _PTX_ and binary code compatible with compute capability 7.0 (third `-gencode` option).

Host code is generated to automatically select at runtime the most appropriate code to load and execute, which, in the above example, will be:

- 5.0 binary code for devices with compute capability 5.0 and 5.2,
    
- 6.0 binary code for devices with compute capability 6.0 and 6.1,
    
- 7.0 binary code for devices with compute capability 7.0 and 7.5,
    
- _PTX_ code which is compiled to binary code at runtime for devices with compute capability 8.0 and 8.6.
    

`x.cu` can have an optimized code path that uses warp reduction operations, for example, which are only supported in devices of compute capability 8.0 and higher. The `__CUDA_ARCH__` macro can be used to differentiate various code paths based on compute capability. It is only defined for device code. When compiling with `-arch=compute_80` for example, `__CUDA_ARCH__` is equal to `800`.

If `x.cu` is compiled for [architecture conditional features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#feature-availability) example with `sm_90a` or `compute_90a`, the code can only run on devices with compute capability 9.0.

Applications using the driver API must compile code to separate files and explicitly load and execute the most appropriate file at runtime.

The Volta architecture introduces _Independent Thread Scheduling_ which changes the way threads are scheduled on the GPU. For code relying on specific behavior of [SIMT scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) in previous architectures, Independent Thread Scheduling may alter the set of participating threads, leading to incorrect results. To aid migration while implementing the corrective actions detailed in [Independent Thread Scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#independent-thread-scheduling-7-x), Volta developers can opt-in to Pascal’s thread scheduling with the compiler option combination `-arch=compute_60 -code=sm_70`.

The `nvcc` user manual lists various shorthands for the `-arch`, `-code`, and `-gencode` compiler options. For example, `-arch=sm_70` is a shorthand for `-arch=compute_70 -code=compute_70,sm_70` (which is the same as `-gencode arch=compute_70,code=\"compute_70,sm_70\"`).

### C++ Compatibility

The front end of the compiler processes CUDA source files according to C++ syntax rules. Full C++ is supported for the host code. However, only a subset of C++ is fully supported for the device code as described in [C++ Language Support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support).

### 64-Bit Compatibility

The 64-bit version of `nvcc` compiles device code in 64-bit mode (i.e., pointers are 64-bit). Device code compiled in 64-bit mode is only supported with host code compiled in 64-bit mode.

## CUDA Runtime
The runtime is implemented in the `cudart` library, which is linked to the application, either statically via `cudart.lib` or `libcudart.a`, or dynamically via `cudart.dll` or `libcudart.so`. 
>  CUDA Runtime 被实现为 `cudart` 库，该库需要链接到应用
>  链接可以是静态链接，通过 `cudart.lib` (Windows) 或 `libcudart.a` (Unix-like: Linux, Mac)，也可以是动态链接，通过 `cudart.dll` (Windows) 或 `libcudart.so` (Unix-like: Linux, Mac)

Applications that require `cudart.dll` and/or `cudart.so` for dynamic linking typically include them as part of the application installation package. It is only safe to pass the address of CUDA runtime symbols between components that link to the same instance of the CUDA runtime.
>  使用 `cudart.dll` 或 `cudart.so` 进行动态链接的应用通常会将这两个文件作为应用自身安装包的一部分
>  只有在链接到相同的 CUDA runtime 实例的组件之间传递 CUDA runtime 符号地址才是安全的

All its entry points are prefixed with `cuda`.
>  CUDA runtime 提供的所有函数都以 `cuda` 为前缀

As mentioned in [Heterogeneous Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming), the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. [Device Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory) gives an overview of the runtime functions used to manage device memory.
>  CUDA 编程模型假设了系统是一个异构系统，包含了一个 host 和一个 device，二者有自己独立的内存

[Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) illustrates the use of shared memory, introduced in [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy), to maximize performance.

[Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory) introduces page-locked host memory that is required to overlap kernel execution with data transfers between host and device memory.

[Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) describes the concepts and API used to enable asynchronous concurrent execution at various levels in the system.

[Multi-Device System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system) shows how the programming model extends to a system with multiple devices attached to the same host.

[Error Checking](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking) describes how to properly check the errors generated by the runtime.

[Call Stack](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#call-stack) mentions the runtime functions used to manage the CUDA C++ call stack.

[Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory) presents the texture and surface memory spaces that provide another way to access device memory; they also expose a subset of the GPU texturing hardware.

[Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphics-interoperability) introduces the various functions the runtime provides to interoperate with the two main graphics APIs, OpenGL and Direct3D.

### Initialization
As of CUDA 12.0, the `cudaInitDevice()` and `cudaSetDevice()` calls initialize the runtime and the primary context associated with the specified device. Absent these calls, the runtime will implicitly use device 0 and self-initialize as needed to process other runtime API requests. One needs to keep this in mind when timing runtime function calls and when interpreting the error code from the first call into the runtime. 
>  CUDA 12.0 之后，`cudaInitDevice()` 和 `cudaSetDevice()` 用于初始化特定设备关联的 runtime 和 primary context
>  如果没有显式调用它们，runtime 会隐式地使用 device 0，并且在处理其他 runtime API 请求时，必要地自行初始化，在对 runtime 函数调用计时以及解释第一次 runtime 调用的错误码时，必须注意到这一点

Before 12.0, `cudaSetDevice()` would not initialize the runtime and applications would often use the no-op runtime call `cudaFree(0)` to isolate the runtime initialization from other api activity (both for the sake of timing and error handling).
>  CUDA 12.0 之前，`cudaSetDevice()` 不会初始化 runtime，一般会特意使用 no-op runtime call `cudaFree(0)` 来初始化 runtime

The runtime creates a CUDA context for each device in the system (see [Context](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context) for more details on CUDA contexts). This context is the _primary context_ for this device and is initialized at the first runtime function which requires an active context on this device. It is shared among all the host threads of the application. 
>  runtime 会为系统中每个设备都创建一个 CUDA context，该 context 就是该设备的 primary context，它会在第一个对需求该设备的 active context 的 runtime 函数被调用时被初始化
>  设备的 primary context 由所有 device thread 共享

As part of this context creation, the device code is just-in-time compiled if necessary (see [Just-in-Time Compilation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation)) and loaded into device memory. This all happens transparently. If needed, for example, for driver API interoperability, the primary context of a device can be accessed from the driver API as described in [Interoperability between Runtime and Driver APIs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interoperability-between-runtime-and-driver-apis).
>  设备的 context 被创建时，device code 会在必要情况下被 just-in-time 编译，然后被加载到设备内存 
>  这些事情的发生都是透明的 (程序员不可见)
>  设备的 primary context 可以通过 driver API 访问

When a host thread calls `cudaDeviceReset()`, this destroys the primary context of the device the host thread currently operates on (that is, the current device as defined in [Device Selection](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-selection)). The next runtime function call made by any host thread that has this device as current will create a new primary context for this device.
>  host thread 调用 `cudaDeviceReset()` 会销毁 host 当前正在操作的设备 (current device) 的 primary context
>  之后，如果有 host thread 再选择该设备为 current device，然后调用 runtime 函数时，该调用会被该设备创建新的 primary context

> [!Note]
> The CUDA interfaces use global state that is initialized during host program initiation and destroyed during host program termination. The CUDA runtime and driver cannot detect if this state is invalid, so using any of these interfaces (implicitly or explicitly) during program initiation or termination after main) will result in undefined behavior.
>> CUDA 接口使用的是全局状态，全局状态在 host program 启动时创建，在 host program 终止时销毁
>> 但 CUDA 接口不会检测状态是否有效，故如果在 host program 启动之前或 host program 终止之后调用 runtime 或 driver API，会出现未定义的行为
> 
> As of CUDA 12.0, `cudaSetDevice()` will now explicitly initialize the runtime after changing the current device for the host thread. Previous versions of CUDA delayed runtime initialization on the new device until the first runtime call was made after `cudaSetDevice()`. This change means that it is now very important to check the return value of `cudaSetDevice()` for initialization errors.
>> CUDA 12.0 后，`cudaSetDevice()` 会在为 host thread 切换 current device 之后，显式地初始化 runtime，之前的版本则将 runtime 初始化推迟到 `cudaSetDevice()` 之后的第一次 runtime 调用
>> 因此，CUDA 12.0 后，需要检查 `cudaSetDevice()` 的返回值，确定初始化是否成功
> 
> The runtime functions from the error handling and version management sections of the reference manual do not initialize the runtime.
>> 用于错误处理和版本管理的 runtime 函数不会初始化 runtime

### Device Memory
As mentioned in [Heterogeneous Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming), the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. Kernels operate out of device memory, so the runtime provides functions to allocate, deallocate, and copy device memory, as well as transfer data between host memory and device memory.
>  CUDA kernel 基于 device memory 运行，故 runtime 提供了分配、释放、拷贝 device memory 的函数，以及在 host memory 和 device memory 之间传输数据的函数

Device memory can be allocated either as _linear memory_ or as _CUDA arrays_.
>  device memory 可以以 linear memory 的形式分配，也可以以 CUDA arrays 的形式分配

CUDA arrays are opaque memory layouts optimized for texture fetching. They are described in [Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory).
>  CUDA arrays 主要是针对 texture fetching 优化的不透明内存布局 (程序员不可知, opaque = transparent)

Linear memory is allocated in a single unified address space, which means that separately allocated entities can reference one another via pointers, for example, in a binary tree or linked list. The size of the address space depends on the host system (CPU) and the compute capability of the used GPU:

Table 1 Linear Memory Address Space

|                                          | x86_64 (AMD64) | POWER (ppc64le) | ARM64       |
| ---------------------------------------- | -------------- | --------------- | ----------- |
| up to compute capability 5.3 (Maxwell)   | 40bit          | 40bit           | 40bit       |
| compute capability 6.0 (Pascal) or newer | up to 47bit    | up to 49bit     | up to 48bit |

>  Linear memory 都在单个统一的地址空间分配
>  地址空间的大小取决于 host system 的 CPU 和 GPU 的 compute capability
>  x86_64 (AMD64) 的 CPU + Pascal 以后的 GPU 的地址空间大小为 47 bit

> [!Note]
> On devices of compute capability 5.3 (Maxwell) and earlier, the CUDA driver creates an uncommitted 40bit virtual address reservation to ensure that memory allocations (pointers) fall into the supported range. This reservation appears as reserved virtual memory, but does not occupy any physical memory until the program actually allocates memory.

Linear memory is typically allocated using `cudaMalloc()` and freed using `cudaFree()` and data transfer between host memory and device memory are typically done using `cudaMemcpy()`. In the vector addition code sample of [Kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels), the vectors need to be copied from host memory to device memory:
>  Linear Memoery:
>  - `cudaMalloc()` 分配
>  - `cudaFree()` 释放
>  - `cudaMemcpy()` 拷贝 (与 host memory)

```cpp
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    ...
}
```

Linear memory can also be allocated through `cudaMallocPitch()` and `cudaMalloc3D()`. These functions are recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements described in [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses), therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory (using the `cudaMemcpy2D()` and `cudaMemcpy3D()` functions). The returned pitch (or stride) must be used to access array elements. The following code sample allocates a `width` x `height` 2D array of floating-point values and shows how to loop over the array elements in device code:

```cpp
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```

The following code sample allocates a `width` x `height` x `depth` 3D array of floating-point values and shows how to loop over the array elements in device code:

```cpp
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                    height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
```

Note

To avoid allocating too much memory and thus impacting system-wide performance, request the allocation parameters from the user based on the problem size. If the allocation fails, you can fallback to other slower memory types (`cudaMallocHost()`, `cudaHostRegister()`, etc.), or return an error telling the user how much memory was needed that was denied. If your application cannot request the allocation parameters for some reason, we recommend using `cudaMallocManaged()` for platforms that support it.

The reference manual lists all the various functions used to copy memory between linear memory allocated with `cudaMalloc()`, linear memory allocated with `cudaMallocPitch()` or `cudaMalloc3D()`, CUDA arrays, and memory allocated for variables declared in global or constant memory space.

The following code sample illustrates various ways of accessing global variables via the runtime API:

```cpp
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

`cudaGetSymbolAddress()` is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space. The size of the allocated memory is obtained through `cudaGetSymbolSize()`.

# Hardware Implementation
The NVIDIA GPU architecture is built around a scalable array of multithreaded _Streaming Multiprocessors_ (_SMs_). 
>  NVIDIA GPU 架构的核心概念是流式多处理器阵列

When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. 
>  CUDA 程序调用 kernel grid，grid 的 blocks 被分配给有冗余执行容量的流式多处理器

The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors.
>  block 中的 threads 在流式多处理器上并发执行
>  一个流式多处理器可以并发执行多个 block，上一个 block 结束，下一个 block 启动

A multiprocessor is designed to execute hundreds of threads concurrently. To manage such a large number of threads, it employs a unique architecture called _SIMT_ (_Single-Instruction, Multiple-Thread_) that is described in [SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture). 
>  流式多处理器在设计上可以并发执行上百个 thread，因为采用了 SIMT，故这些 thread 执行的都是同一条指令

The instructions are pipelined, leveraging instruction-level parallelism within a single thread, as well as extensive thread-level parallelism through simultaneous hardware multithreading as detailed in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading). Unlike CPU cores, they are issued in order and there is no branch prediction or speculative execution.
>  单个 thread 对指令的执行采用了指令流水线，以提高指令级并行
>  thread 执行的指令按照顺序发送，且 thread 没有分支预测或投机执行的能力，这点和 CPU core 不一样

>  层次结构:
>  A CUDA Program: Multiple kernel launch
>  A kernel launch: A grid
>  A grid: Multiple blocks, 同一个 grid 内，block 和 block 之间在不同/相同的 SM 上并发
>  A block: Multiple threads, 同一个 block 内，thread 和 thread 之间在相同的 SM 上并发

[SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture) and [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading) describe the architecture features of the streaming multiprocessor that are common to all devices. [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x), and [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x) provide the specifics for devices of compute capabilities 5.x, 6.x, and 7.x respectively.

The NVIDIA GPU architecture uses a little-endian representation.
>  NV GPU 采用小端序

## SIMT Architecture
The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called _warps_. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. 
>  SM 创建、管理、调度、执行的最小单元是 warp (32 个并行线程)
>  warp 内的线程从相同的程序地址开始执行，但每个 thread 有自己的指令地址计数器和寄存器状态，故 thread 与 thread 的执行是完全独立的

The term _warp_ originates from weaving, the first parallel thread technology. A _half-warp_ is either the first or second half of a warp. A _quarter-warp_ is either the first, second, third, or fourth quarter of a warp.

When a multiprocessor is given one or more thread blocks to execute, it partitions them into warps and each warp gets scheduled by a _warp scheduler_ for execution. The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0. [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy) describes how thread IDs relate to thread indices in the block.
>  SM 将 block 按照 warp 为单位划分，SM 中的 warp schedule 调度 warp 的执行
>  block 到 warps 的划分方式就是将 ID 连续的一组 32 个 threads 划分为同一个 warp

A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path. Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjoint code paths.
>  虽然 warp 内 thread 和 thread 的执行是完全独立的，但 warp 的完全并行要求 warp 内所有的 threads 执行相同的指令，因为 warp 一次执行一次**共同**的指令
>  如果 warp 内的部分 threads 出现了 divergence, warp 需要分多次执行不同的分支

The SIMT architecture is akin to SIMD (Single Instruction, Multiple Data) vector organizations in that a single instruction controls multiple processing elements. A key difference is that SIMD vector organizations expose the SIMD width to the software, whereas SIMT instructions specify the execution and branching behavior of a single thread. In contrast with SIMD vector machines, SIMT enables programmers to write thread-level parallel code for independent, scalar threads, as well as data-parallel code for coordinated threads. For the purposes of correctness, the programmer can essentially ignore the SIMT behavior; however, substantial performance improvements can be realized by taking care that the code seldom requires threads in a warp to diverge. In practice, this is analogous to the role of cache lines in traditional code: Cache line size can be safely ignored when designing for correctness but must be considered in the code structure when designing for peak performance. Vector architectures, on the other hand, require the software to coalesce loads into vectors and manage divergence manually.
>  减少 warp divergence 可以显著提高性能

Prior to NVIDIA Volta, warps used a single program counter shared amongst all 32 threads in the warp together with an active mask specifying the active threads of the warp. As a result, threads from the same warp in divergent regions or different states of execution cannot signal each other or exchange data, and algorithms requiring fine-grained sharing of data guarded by locks or mutexes can easily lead to deadlock, depending on which warp the contending threads come from.
>  NVIDIA Volta 之前，warp 内所有 threads 共享一个程序计数器，即所有 threads 必须执行相同的指令
>  如果遇到分支，GPU 按照分支条件，通过 active mask 标记 warp 中应该活跃的 threads，执行一条路径，然后切换 active mask，执行另一条路径
>  在执行某一条路径时，被 active mask 标记为 “不活跃” 的线程将进入显式的等待状态，如果有 “不活跃” 的线程持有了共享数据的锁，而有 “活跃” 的线程需要这个锁，程序就会进入死锁
>  因此，这就限制了同一个 warp 内的线程对数据的交换

Starting with the NVIDIA Volta architecture, _Independent Thread Scheduling_ allows full concurrency between threads, regardless of warp. With Independent Thread Scheduling, the GPU maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity, either to make better use of execution resources or to allow one thread to wait for data to be produced by another. A schedule optimizer determines how to group active threads from the same warp together into SIMT units. This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity.
>  NVIDIA Volta 之后采用了 Independent Thread Scheduling, GPU 为每一个 thread 都维护它的执行状态，包括了程序计数器和调用栈, GPU 可以以线程为单位让出执行 (暂停和恢复执行)
>  schedule optimizer 负责将 active threads 组合为 SMIT 单元
>  (应该是不再要求 divergence 的情况下强制串行执行了)

Independent Thread Scheduling can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity [2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn2) of previous hardware architectures. In particular, any warp-synchronous code (such as synchronization-free, intra-warp reductions) should be revisited to ensure compatibility with NVIDIA Volta and beyond. See [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x) for further details.

Note
The threads of a warp that are participating in the current instruction are called the _active_ threads, whereas threads not on the current instruction are _inactive_ (disabled). Threads can be inactive for a variety of reasons including having exited earlier than other threads of their warp, having taken a different branch path than the branch path currently executed by the warp, or being the last threads of a block whose number of threads is not a multiple of the warp size.

If a non-atomic instruction executed by a warp writes to the same location in global or shared memory for more than one of the threads of the warp, the number of serialized writes that occur to that location varies depending on the compute capability of the device (see [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x), and [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)), and which thread performs the final write is undefined.

If an [atomic](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions) instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined.

[2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id125) The term _warp-synchronous_ refers to code that implicitly assumes threads in the same warp are synchronized at every instruction.

## Hardware Multithreading
The execution context (program counters, registers, and so on) for each warp processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp. Therefore, switching from one execution context to another has no cost, and at every instruction issue time, a warp scheduler selects a warp that has threads ready to execute its next instruction (the [active threads](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes) of the warp) and issues the instruction to those threads.
>  SM 在片上保存每个 warp 的执行上下文，故 warp 的执行上下文切换没有开销
>  在指令发射期间，SM 的 warp scheduler 选择针对当前指令，具有活跃线程的 warp，将指令发送给这些活跃线程

In particular, each multiprocessor has a set of 32-bit registers that are partitioned among the warps, and a _parallel data cache_ or _shared memory_ that is partitioned among the thread blocks.
>  SM 在一组 32 bit 寄存器上存储 warps 的执行上下文 (故寄存器的大小决定了可以驻留的 warps 数量，同一个 warp 内的线程共享该 warp 的寄存器组)
>  SM 在 parallel data cache, shard memory 存储 blocks 的执行上下文 (故 shared memory 的大小决定了可以驻留的 blocks 数量，同一个 block 内的线程共享该 block 的 shared memory, parallel data cache 区域)

The number of blocks and warps that can reside and be processed together on the multiprocessor for a given kernel depends on the amount of registers and shared memory used by the kernel and the amount of registers and shared memory available on the multiprocessor. 
>  当然，warps 和 blocks 的驻留数量还取决于 kernel 中 warp 和 block 的资源使用量

There are also a maximum number of resident blocks and a maximum number of resident warps per multiprocessor. These limits as well the amount of registers and shared memory available on the multiprocessor are a function of the compute capability of the device and are given in [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities). 
>  硬件对驻留的 blocks 和 warps 也设置了最大数量上限

If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch.
>  如果 kernel 中，一个 warp 使用的资源或一个 block 使用的资源超过了硬件配置，kernel 将启动失败

The total number of warps in a block is as follows:

$$\text{ceil}\left( \frac{T}{W_{size}},1 \right)$$

- _T_ is the number of threads per block,
- $W_{size}$ is the warp size, which is equal to 32,
- ceil(x, y) is equal to x rounded up to the nearest multiple of y.

The total number of registers and total amount of shared memory allocated for a block are documented in the CUDA Occupancy Calculator provided in the CUDA Toolkit.

# Performance Guidelines
## Overall Performance Optimization Strategies
Performance optimization revolves around four basic strategies:

- Maximize parallel execution to achieve maximum utilization;
- Optimize memory usage to achieve maximum memory throughput;
- Optimize instruction usage to achieve maximum instruction throughput;
- Minimize memory thrashing.

>  性能优化遵循四个基本的策略:
>  - 最大化并行执行，达到最高的硬件利用率
>  - 优化内存使用，最大化内存吞吐
>  - 优化指令使用，最大化指令吞吐
>  - 最小化内存抖动

>  最大化并行执行:
>  - 任务分解、并行化
>  - 负载均衡，避免单个并行单元成为瓶颈

>  优化内存使用:
>  - 提高缓存命中率: 优化内存排布，让经常访问的彼此靠近 (空间局部性)，并按顺序访问 (时间局部性)
>  - 合并内存访问: 多个并行线程**同时**访问连续的内存位置
>  - 最小化冗余数据复制
>  - 精度允许时，使用更小的数据类型
>  - 利用内存层次结构

>  优化指令使用:
>  - 使用更高效的指令，例如 AVX 等 SIMD 指令，例如用乘法计算替代除法计算
>  - 最小化控制流分歧: 如果同一个 wrap 内的线程采用了不同的路径，则所有路径会被串行执行，导致效率降低
>  - 编译器优化: 循环展开、合并、分解、冗余计算消除

>  最小化内存抖动
>  - 工作集大小管理

Which strategies will yield the best performance gain for a particular portion of an application depends on the performance limiters for that portion; optimizing instruction usage of a kernel that is mostly limited by memory accesses will not yield any significant performance gain, for example. Optimization efforts should therefore be constantly directed by measuring and monitoring the performance limiters, for example using the CUDA profiler.
>  具体哪一种优化策略的效果最好取决于当前应用的效率瓶颈在哪里，即 performance limiter

 Also, comparing the floating-point operation throughput or memory throughput—whichever makes more sense—of a particular kernel to the corresponding peak theoretical throughput of the device indicates how much room for improvement there is for the kernel.
>  以及将当前的性能，例如浮点数计算吞吐和内存吞吐，和硬件的峰值性能比较，可以知道是否存在更多的优化空间

## Maximize Utilization
To maximize utilization the application should be structured in a way that it exposes as much parallelism as possible and efficiently maps this parallelism to the various components of the system to keep them busy most of the time.
>  最大化利用率的思路就是最大化暴露应用的并行性，同时将暴露的并行性映射到系统的各个组件，保持这些组件在大部分时间都忙碌

### Application Level
At a high level, the application should maximize parallel execution between the host, the devices, and the bus connecting the host to the devices, by using asynchronous functions calls and streams as described in [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution). 
>  在应用层，应用应该最大化 host, device, bus 的并行执行
>  方法就是采用异步函数调用和 streams

It should assign to each processor the type of work it does best: serial workloads to the host; parallel workloads to the devices.
>  此外，应用应该将最合适的工作分配给最合适的硬件: host 执行串行负载, device 执行并行负载

For the parallel workloads, at points in the algorithm where parallelism is broken because some threads need to synchronize in order to share data with each other, there are two cases: Either these threads belong to the same block, in which case they should use `__syncthreads()` and share data through shared memory within the same kernel invocation, or they belong to different blocks, in which case they must share data through global memory using two separate kernel invocations, one for writing to and one for reading from global memory. 
>  只要有并行化和数据共享，往往就需要有同步点
>  相互同步的线程要么都属于同一个 block，在同一次 kernel 调用中，通过 shared memory 共享数据，此时使用 `__syncthreds()` 同步；要么属于不同的 blocks，在不同的 kernel 调用中，通过 (也只能通过) global memory 共享数据，一次 kernel 调用写，另一次 kernel 调用读

The second case is much less optimal since it adds the overhead of extra kernel invocations and global memory traffic. Its occurrence should therefore be minimized by mapping the algorithm to the CUDA programming model in such a way that the computations that require inter-thread communication are performed within a single thread block as much as possible.
>  第二种同步模型的出现频率应该尽可能小，故需要尽可能将线程间通信映射到 block 内部

### Device Level
At a lower level, the application should maximize parallel execution between the multiprocessors of a device.

Multiple kernels can execute concurrently on a device, so maximum utilization can also be achieved by using streams to enable enough kernels to execute concurrently as described in [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution).

>  在设备级别，因为设备可以同时执行多个 kernel，故应用需要用 streams 让多个 kernel 并发执行，以达到最大的设备利用率

### Multiprocessor Level
At an even lower level, the application should maximize parallel execution between the various functional units within a multiprocessor.
>  在多处理器级别，因为多处理器有多个功能单元 (例如整数运算单元、浮点运算单元、内存加载和存储单元，这里指的应该是 threads)，故应用需要最大化并行多处理器的多个功能单元之间的并行执行

As described in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading), a GPU multiprocessor primarily relies on thread-level parallelism to maximize utilization of its functional units. Utilization is therefore directly linked to the number of resident warps. 
>  SM 的最大利用率主要和驻留的 wraps 数量相关

At every instruction issue time, a warp scheduler selects an instruction that is ready to execute. This instruction can be another independent instruction of the same warp, exploiting instruction-level parallelism, or more commonly an instruction of another warp, exploiting thread-level parallelism. 
>  在每次的指令发射时间 (指令的执行被流水线化，故每个时钟周期都有机会发射新的指令)，SM 的 warp scheduler 都会选择一条准备被执行的指令: 这条指令可以是相同 warp 要执行的另一条独立指令 -> 指令级并行，也 (更通常) 是另一个 warp 要执行的指令 -> 线程间并行

If a ready to execute instruction is selected it is issued to the [active](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes) threads of the warp. 
>  选择要指令后，warp scheduler 会将指令发送给对应 warp 的活跃线程

The number of clock cycles it takes for a warp to be ready to execute its next instruction is called the _latency_, and full utilization is achieved when all warp schedulers always have some instruction to issue for some warp at every clock cycle during that latency period, or in other words, when latency is completely “hidden”. 
>  对于一个 warp，从它收到上一条指令时开始，到它准备好执行下一条指令时结束，这段时长 (时钟周期数) 被称为 latency
>  如果 SM 的所有 warp schedulers 在每个时钟周期都可以将一条指令发送给某个 warp, SM 就达到了满的利用率，也就是 warps 执行指令的 latency 被完全掩盖

The number of instructions required to hide a latency of L clock cycles depends on the respective throughputs of these instructions (see [Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions) for the throughputs of various arithmetic instructions). If we assume instructions with maximum throughput, it is equal to:
> 隐藏 L 个时钟周期的 latency 所需要的指令数量取决于 SM 的指令发布吞吐 (单位时间内能够发送的指令数量)

- _4L_ for devices of compute capability 5.x, 6.1, 6.2, 7.x and 8.x since for these devices, a multiprocessor issues one instruction per warp over one clock cycle for four warps at a time, as mentioned in [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities).
- _2L_ for devices of compute capability 6.0 since for these devices, the two instructions issued every cycle are one instruction for two different warps.

>  对于 compute capability `5.x, 6.1, 6.2, 7.x, 8.x` 的设备，SM 的指令发布吞吐是 4，即它们的 SM 一个周期内可以同时向四个 warp 发送指令
>  对于 compute capability `6.0`，SM 的指令发布吞吐是 2

The most common reason a warp is not ready to execute its next instruction is that the instruction’s input operands are not available yet.
>  warp 没准备好执行下一条指令的最常见原因就是其输入 operands 尚未准备好

If all input operands are registers, latency is caused by register dependencies, i.e., some of the input operands are written by some previous instruction(s) whose execution has not completed yet. In this case, the latency is equal to the execution time of the previous instruction and the warp schedulers must schedule instructions of other warps during that time. Execution time varies depending on the instruction. On devices of compute capability 7.x, for most arithmetic instructions, it is typically 4 clock cycles. This means that 16 active warps per multiprocessor (4 cycles, 4 warp schedulers) are required to hide arithmetic instruction latencies (assuming that warps execute instructions with maximum throughput, otherwise fewer warps are needed). If the individual warps exhibit instruction-level parallelism, i.e. have multiple independent instructions in their instruction stream, fewer warps are needed because multiple independent instructions from a single warp can be issued back to back.
>  如果所有的输入 operands 都在寄存器，则 warp 的 latency 的来源就是寄存器依赖: 前面的 warp 还没有完成指令执行，将该 warp 需要的 operands 写入寄存器
>  此时，该 warp 的 latency 就是上一个 warp 的指令执行时间

If some input operand resides in off-chip memory, the latency is much higher: typically hundreds of clock cycles. The number of warps required to keep the warp schedulers busy during such high latency periods depends on the kernel code and its degree of instruction-level parallelism. In general, more warps are required if the ratio of the number of instructions with no off-chip memory operands (i.e., arithmetic instructions most of the time) to the number of instructions with off-chip memory operands is low (this ratio is commonly called the arithmetic intensity of the program).
>  如果一些输入 operands 在片外存储，latency 的主要来源将会是访存取数，通常高达几百个时钟周期
>  在这几百个时钟周期内保持 warp scheduler 忙碌所需要的 warps 数量取决于 kernel 代码和 warp 的指令级并行程度，一般来说，如果 $\frac {\text{operand 不需要通过片外访存获取的指令 (通常是算术指令) 数量}}{\text{operand 需要通过片外访存获取的指令数量}}$ 较低，就需要更多的 warp 数量来掩盖 latency
>  这一比率通常称为算术密度

Another reason a warp is not ready to execute its next instruction is that it is waiting at some memory fence ([Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-functions)) or synchronization point ([Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions)). A synchronization point can force the multiprocessor to idle as more and more warps wait for other warps in the same block to complete execution of instructions prior to the synchronization point. Having multiple resident blocks per multiprocessor can help reduce idling in this case, as warps from different blocks do not need to wait for each other at synchronization points.
>  warp 无法立即执行下一条指令的另一个原因是内存屏障或同步点
>  同步点会导致多个 warps 等待同一个 block 内的其他 warps 完成它们的执行，如果没有来自其他 block 的 warp 被调度，执行这些等待的 warps 的执行单元将处于空闲状态
>  提高 SM 的驻留 block 数量可以减少这类空闲

The number of blocks and warps residing on each multiprocessor for a given kernel call depends on the execution configuration of the call ([Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)), the memory resources of the multiprocessor, and the resource requirements of the kernel as described in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading). Register and shared memory usage are reported by the compiler when compiling with the `--ptxas-options=-v` option.
>  SM 的驻留 block, wrap 数量和 kernel 的 launch 参数、SM 的资源有关
>  使用 `--ptxas-options=-v` 编译可以报告寄存器和 shared memory 使用量

The total amount of shared memory required for a block is equal to the sum of the amount of statically allocated shared memory and the amount of dynamically allocated shared memory.

The number of registers used by a kernel can have a significant impact on the number of resident warps. For example, for devices of compute capability 6.x, if a kernel uses 64 registers and each block has 512 threads and requires very little shared memory, then two blocks (i.e., 32 warps) can reside on the multiprocessor since they require 2x512x64 registers, which exactly matches the number of registers available on the multiprocessor. But as soon as the kernel uses one more register, only one block (i.e., 16 warps) can be resident since two blocks would require 2x512x65 registers, which are more registers than are available on the multiprocessor. Therefore, the compiler attempts to minimize register usage while keeping register spilling (see [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)) and the number of instructions to a minimum. Register usage can be controlled using the `maxrregcount` compiler option, the `__launch_bounds__()` qualifier as described in [Launch Bounds](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-bounds), or the `__maxnreg__()` qualifier as described in [Maximum Number of Registers per Thread](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximum-number-of-registers-per-thread).

The register file is organized as 32-bit registers. So, each variable stored in a register needs at least one 32-bit register, for example, a `double` variable uses two 32-bit registers.

The effect of execution configuration on performance for a given kernel call generally depends on the kernel code. Experimentation is therefore recommended. Applications can also parametrize execution configurations based on register file size and shared memory size, which depends on the compute capability of the device, as well as on the number of multiprocessors and memory bandwidth of the device, all of which can be queried using the runtime (see reference manual).

The number of threads per block should be chosen as a multiple of the warp size to avoid wasting computing resources with under-populated warps as much as possible.

#### Occupancy Calculator
Several API functions exist to assist programmers in choosing thread block size and cluster size based on register and shared memory requirements.

- The occupancy calculator API, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, can provide an occupancy prediction based on the block size and shared memory usage of a kernel. This function reports occupancy in terms of the number of concurrent thread blocks per multiprocessor.
    
    - Note that this value can be converted to other metrics. Multiplying by the number of warps per block yields the number of concurrent warps per multiprocessor; further dividing concurrent warps by max warps per multiprocessor gives the occupancy as a percentage.
        
- The occupancy-based launch configurator APIs, `cudaOccupancyMaxPotentialBlockSize` and `cudaOccupancyMaxPotentialBlockSizeVariableSMem`, heuristically calculate an execution configuration that achieves the maximum multiprocessor-level occupancy.
    
- The occupancy calculator API, `cudaOccupancyMaxActiveClusters`, can provided occupancy prediction based on the cluster size, block size and shared memory usage of a kernel. This function reports occupancy in terms of number of max active clusters of a given size on the GPU present in the system.
    

The following code sample calculates the occupancy of MyKernel. It then reports the occupancy level with the ratio between concurrent warps versus maximum warps per multiprocessor.

```
// Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

// Host code
int main()
{
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

    return 0;
}

The following code sample configures an occupancy-based kernel launch of MyKernel according to the user input.

// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// Host code
int launchMyKernel(int *array, int arrayCount)
{
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device
                        // launch
    int gridSize;       // The actual grid size needed, based on input
                        // size

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)MyKernel,
        0,
        arrayCount);

    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
    cudaDeviceSynchronize();

    // If interested, the occupancy can be calculated with
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor

    return 0;
}
```

The following code sample shows how to use the cluster occupancy API to find the max number of active clusters of a given size. Example code below calucaltes occupancy for cluster of size 2 and 128 threads per block.

Cluster size of 8 is forward compatible starting compute capability 9.0, except on GPU hardware or MIG configurations which are too small to support 8 multiprocessors in which case the maximum cluster size will be reduced. But it is recommended that the users query the maximum cluster size before launching a cluster kernel. Max cluster size can be queried using `cudaOccupancyMaxPotentialClusterSize` API.

```
{
  cudaLaunchConfig_t config = {0};
  config.gridDim = number_of_blocks;
  config.blockDim = 128; // threads_per_block = 128
  config.dynamicSmemBytes = dynamic_shared_memory_size;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = 2; // cluster_size = 2
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;

  int max_cluster_size = 0;
  cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)kernel, &config);

  int max_active_clusters = 0;
  cudaOccupancyMaxActiveClusters(&max_active_clusters, (void *)kernel, &config);

  std::cout << "Max Active Clusters of size 2: " << max_active_clusters << std::endl;
}
```

The CUDA Nsight Compute User Interface also provides a standalone occupancy calculator and launch configurator implementation in `<CUDA_Toolkit_Path>/include/cuda_occupancy.h` for any use cases that cannot depend on the CUDA software stack. The Nsight Compute version of the occupancy calculator is particularly useful as a learning tool that visualizes the impact of changes to the parameters that affect occupancy (block size, registers per thread, and shared memory per thread).

## 8.3. Maximize Memory Throughput
The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth.

That means minimizing data transfers between the host and the device, as detailed in [Data Transfer between Host and Device](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-transfer-between-host-and-device), since these have much lower bandwidth than data transfers between global memory and the device.

That also means minimizing data transfers between global memory and the device by maximizing use of on-chip memory: shared memory and caches (i.e., L1 cache and L2 cache available on devices of compute capability 2.x and higher, texture cache and constant cache available on all devices).

Shared memory is equivalent to a user-managed cache: The application explicitly allocates and accesses it. As illustrated in [CUDA Runtime](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime), a typical programming pattern is to stage data coming from device memory into shared memory; in other words, to have each thread of a block:

- Load data from device memory to shared memory,
    
- Synchronize with all the other threads of the block so that each thread can safely read shared memory locations that were populated by different threads,
    
- Process the data in shared memory,
    
- Synchronize again if necessary to make sure that shared memory has been updated with the results,
    
- Write the results back to device memory.
    

For some applications (for example, for which global memory access patterns are data-dependent), a traditional hardware-managed cache is more appropriate to exploit data locality. As mentioned in [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x), [Compute Capability 8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x) and [Compute Capability 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0), for devices of compute capability 7.x, 8.x and 9.0, the same on-chip memory is used for both L1 and shared memory, and how much of it is dedicated to L1 versus shared memory is configurable for each kernel call.

The throughput of memory accesses by a kernel can vary by an order of magnitude depending on access pattern for each type of memory. The next step in maximizing memory throughput is therefore to organize memory accesses as optimally as possible based on the optimal memory access patterns described in [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses). This optimization is especially important for global memory accesses as global memory bandwidth is low compared to available on-chip bandwidths and arithmetic instruction throughput, so non-optimal global memory accesses generally have a high impact on performance.

### 8.3.1. Data Transfer between Host and Device

Applications should strive to minimize data transfer between the host and the device. One way to accomplish this is to move more code from the host to the device, even if that means running kernels that do not expose enough parallelism to execute on the device with full efficiency. Intermediate data structures may be created in device memory, operated on by the device, and destroyed without ever being mapped by the host or copied to host memory.

Also, because of the overhead associated with each transfer, batching many small transfers into a single large transfer always performs better than making each transfer separately.

On systems with a front-side bus, higher performance for data transfers between host and device is achieved by using page-locked host memory as described in [Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory).

In addition, when using mapped page-locked memory ([Mapped Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory)), there is no need to allocate any device memory and explicitly copy data between device and host memory. Data transfers are implicitly performed each time the kernel accesses the mapped memory. For maximum performance, these memory accesses must be coalesced as with accesses to global memory (see [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)). Assuming that they are and that the mapped memory is read or written only once, using mapped page-locked memory instead of explicit copies between device and host memory can be a win for performance.

On integrated systems where device memory and host memory are physically the same, any copy between host and device memory is superfluous and mapped page-locked memory should be used instead. Applications may query a device is `integrated` by checking that the integrated device property (see [Device Enumeration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)) is equal to 1.


