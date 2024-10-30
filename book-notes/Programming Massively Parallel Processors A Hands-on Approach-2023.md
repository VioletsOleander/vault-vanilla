# 1 Introduction
# 2 Heterogeneous data parallel computing
## 2.1 Data parallelism
Independent evaluation of different pieces of data is the basis of data parallelism.
## 2.2 CUDA C program structure
CUDA C extends the popular ANSI C programming language with minimal new syntax and library functions to let programmers target heterogeneous computing systems containing both CPU cores and massively parallel GPUs.

CUDA programmers can assume that these threads take very few clock cycles to generate and schedule, owing to efficient hardware support. This assumption contrasts with traditional CPU threads, which typically take thousands of clock cycles to generate and schedule. 
## 2.3 A vector addition kernel
whenever there is a need to distinguish between host and device data, we will suffix the names of variables that are used by the host with “\_h” and those of variables that are used by a device with “\_d” to remind ourselves of the intended usage of these variables.

In practice, such a “transparent” outsourcing model can be very inefficient because of all the copying of data back and forth. One would often keep large and important data structures on the device and simply invoke device functions on them from the host code.
## 2.4 Device global memory and data transfer
The CUDA runtime system (typically running on the host) provides applications programming interface (API) functions to perform these activities on behalf of the programmer.

![[PMPP-Fig2.6.png]]
The first parameter to the cudaMalloc function is the address of a pointer variable that will be set to point to the allocated object. The address of the pointer variable should be cast to (void ) because the function expects a generic pointer;

The cudaMalloc function writes to the pointer variable whose address is given as the first parameter.
(
`cudaMalloc` 不直接返回指针，而是覆盖写参数中给定的指针变量
)

Since size is in number of bytes, the programmer needs to translate from the number of elements in an array to the number of bytes when determining the value of size.

Dereferencing a device global memory pointer in host code can cause exceptions or other types of runtime errors. (不要在主机代码上解引用设备指针)

The cudaMemcpy function takes four parameters. The first parameter is a pointer to the destination location for the data object to be copied. The second parameter points to the source location. The third parameter specifies the number of bytes to be copied. The fourth parameter indicates the types of memory involved in the copy: from host to host, from host to device, from device to host, and from device to device.
![[PMPP-Fig2.7.png]]

The two symbolic constants, `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`, are recognized, predefined constants of the CUDA programming environment.

The `vecAdd` function allocates space in device global memory, requests data transfers, and calls the kernel that performs the actual vector addition. We refer to this type of host code as a *stub* for calling a kernel. ( `vecAdd` 函数在设备全局内存中分配空间，请求数据传输，并调用执行实际向量加法的核心函数，我们称这种类型的主机代码为调用核心函数的存根 )

CUDA API functions return flags that indicate whether an error has occurred when they served the request.
## 2.5 Kernel functions and threading
In CUDA C, a kernel function specifies the code to be executed by all threads during a parallel phase. Since all these threads execute the same code, CUDA C programming is an instance of the wellknown single-program multiple-data (SPMD) parallel programming style

When a program’s host code calls a kernel, the CUDA runtime system launches a grid of threads that are organized into a two-level hierarchy. Each grid is organized as an array of thread blocks, which we will refer to as blocks for brevity. All blocks of a grid are of the same size; each block can contain up to 1024 threads on current systems.
(
网格 grid = 线程块数组 array of thread blocks
)

> **Built-in Variables**
> Many programming languages have *built-in* variables. These variables have special meaning and purpose. The values of these variables are often *pre-initialized* by the runtime system and are typically read-only in the program. The programmers should refrain from redefining these variables for any other purposes.

The total number of threads in each thread block is specified by the host code when a kernel is called. The same kernel can be called with different numbers of threads at different parts of the host code.

For a given grid of threads, the number of threads in a block is available in a built-in variable named `blockDim`. The `blockDim` variable is a struct with three unsigned integer fields (x, y, and z) that help the programmer to organize the threads into a one-, two-, or threedimensional array.

In general, it is recommended that the number of threads in each dimension of a thread block be a multiple of 32 for hardware efficiency reasons.

CUDA kernels have access to two more built-in variables (`threadIdx` and `blockIdx`) that allow threads to distinguish themselves from each other and to determine the area of data each thread is to work on.
(
三个内建变量：`blockDim` , `threadIdx` , `blockIdx`
)
These built-in variables are the means for threads to access hardware registers that provide the identifying coordinates to threads. Different threads will see different values in their `threadIdx.x`, `blockIdx.x`, and `blockDim.x` variables.

a unique global index i is calculated as i= `blockIdx.x * blockDim. x + threadIdx.x`.

The syntax of a kernel is ANSI C with some notable extensions.

First, there is a CUDA-Cspecific keyword “`__global__`” in front of the declaration of the `vecAddKernel` function. This keyword indicates that the function is a kernel and that it can be called to generate a grid of threads on a device.

In general, CUDA C extends the C language with three qualifier keywords (修饰符关键词) that can be used in function declarations.
![[PMPP-Fig2.11.png]]

The “`__global__`” keyword indicates that the function being declared is a CUDA C kernel function. Such a kernel function is executed on the device and can be called from the host.
(
在主机端/设备端调用，在设备端执行
)

The “`__device__`” keyword indicates that the function being declared is a CUDA device function. A device function executes on a CUDA device and can be called only from a kernel function or another device function. The device function is executed by the device thread that calls it and does not result in any new device threads being launched.
(
在设备端由内核调用，在设备端执行，不会发起新的设备线程
)

The “`__host__`” keyword indicates that the function being declared is a CUDA host function. A host function is simply a traditional C function that executes on the host and can be called only from another host function. By default, all functions in a CUDA program are host functions if they do not have any of the CUDA keywords in their declaration.

Note that one can use both “`__host__`” and “`__device__`” in a function declaration. This combination tells the compilation system to generate two versions of object code for the same function. One is executed on the host and can be called only from a host function. The other is executed on the device and can be called only from a device or kernel function.

Note that there is an if (i , n) statement in addVecKernel in Fig. 2.10. This is because not all vector lengths can be expressed as multiples of the block size. For example, let’s assume that the vector length is 100. The smallest efficient thread block dimension is 32. Assume that we picked 32 as block size. One would need to launch four thread blocks to process all the 100 vector elements. However, the four thread blocks would have 128 threads. We need to disable the last 28 threads in thread block 3 from doing work not expected by the original program. Since all threads are to execute the same code, all will test their i values against n, which is 100. With the if (i , n) statement, the first 100 threads will perform the addition, whereas the last 28 will not. This allows the kernel to be called to process vectors of arbitrary lengths.
(
一句话概括就是注意越界检查
)
## 2.6 Calling kernel functions
When the host code calls a kernel, it sets the grid and thread block dimensions via *execution configuration parameters.*
The first configuration parameter gives the number of blocks in the grid. The second specifies the number of threads in each block.
(
`<<<number of blocks in grid, number of threads per block>>>`
)

Note that all the thread blocks operate on different parts of the vectors. They can be executed in any arbitrary order. The programmer must not make any assumptions regarding execution order.

It is important to point out again that the vector addition example is used for its simplicity. In practice, the overhead of allocating device memory, input data transfer from host to device, output data transfer from device to host, and deallocating device memory will likely make the resulting code slower than the original sequential code. This is because the amount of calculation that is done by the kernel is small relative to the amount of data processed or transferred.

Real applications typically have kernels in which much more work is needed relative to the amount of data processed, which makes the additional overhead worthwhile. Real applications also tend to keep the data in the device memory across multiple kernel invocations so that the overhead can be amortized. (在多个内核调用中保持数据在设备内存以摊销数据传输的开销)
## 2.7 Compilation
![[PMPP-Fig2.14.png]]
The NVCC compiler processes a CUDA C program, using the CUDA keywords to separate the host code and device code. The host code is straight ANSI C code, which is compiled with the host’s standard C/C++ compilers and is run as a traditional CPU process. The device code, which is marked with CUDA keywords that designate CUDA kernels and their associated helper functions and data structures, is compiled by NVCC into virtual binary files called PTX files. These PTX files are further compiled by a runtime component of NVCC into the real object files and executed on a CUDA-capable GPU device.
## 2.8 Summary
### 2.8.1 Function declarations
Using one of “`__global__`,” “`__device__`,” or “`__host__`,” a CUDA C programmer can instruct the compiler to generate a kernel function, a device function, or a host function.
### 2.8.2 Kernel call and grid launch
CUDA C extends the C function call syntax with kernel execution configuration parameters surrounded by <<< and >>> These execution configuration parameters are only used when calling a kernel function to launch a grid. We discussed the execution configuration parameters that define the dimensions of the grid and the dimensions of each block. (定义了 grid 的维度和 block 的维度)
### 2.8.3 Built-in (predefined) variables
We discussed the `threadIdx`, `blockDim`, and `blockIdx`
### 2.8.4 Runtime application programming interface
CUDA supports a set of API functions to provide services to CUDA C programs. The services that we discussed in this chapter are `cudaMalloc`, `cudaFree`, and `cudaMemcpy` functions.
# 3 Multidimensional grids and data
## 3.1 Multidimensional grid orgranization
In CUDA, all threads in a grid execute the same kernel function.

The execution configuration parameters in a kernel call statement specify the dimensions of the grid and the dimensions of each block. These dimensions are available via the `gridDim` and `blockDim` (built-in) variables.

In general, a grid is a three-dimensional (3 D) array of blocks, and each block is a 3 D array of threads. When calling a kernel, the program needs to specify the size of the grid and the blocks in each dimension. These are specified by using the execution configuration parameters (within ,, ,. . . .. .) of the kernel call statement. The first execution configuration parameter specifies the dimensions of the grid in number of blocks. The second specifies the dimensions of each block in number of threads. Each such parameter has the type `dim3`, which is an integer vector type of three elements `x`, `y`, and `z`. These three elements specify the sizes of the three dimensions. The programmer can use fewer than three dimensions by setting the size of the unused dimensions to 1. (没有注明就认为该维度是 1)
```c
dim3 dimGrid(...,...,...);
dim3 dimBlock(...,...,...);

vecAddKernel<<<dimGrid, dimBlock>>>(...);
```
Note that `dimBlock` and `dimGrid` are host code variables that are defined by the programmer. These variables can have any legal C variable name as long as they have the type `dim3`.

Once the grid has been launched, the grid and block dimensions will remain the same until the entire grid has finished execution

For convenience, CUDA provides a special shortcut for calling a kernel with one-dimensional (1 D) grids and blocks. Instead of using dim 3 variables, one can use arithmetic expressions to specify the configuration of 1 D grids and blocks. In this case, the CUDA compiler simply takes the arithmetic expression as the x dimensions and assumes that the y and z dimensions are 1.
Readers who are familiar with C++ would realize that this “shorthand” convention for 1 D configurations takes advantage of how C++ constructors and default parameters work. The default values of the parameters to the dim 3 constructor are 1. When a single value is passed where a dim 3 is expected, that value will be passed to the first parameter of the constructor, while the second and third parameters take the default value of 1. The result is a 1 D grid or block in which the size of the x dimension is the value passed and the sizes of the y and z dimensions are 1.

Within the kernel function, the x field of variables `gridDim` and `blockDim` are preinitialized according to the values of the execution configuration parameters.

In CUDA C the allowed values of `gridDim.x` range from 1 to $2^{31}-1$ (INT 32), and those of `gridDim.y` and `gridDim.z` range from 1 to $2^{16}-1$ (65,535 INT 16 ).

The total size of a block in current CUDA systems is limited to 1024 threads. (目前 CUDA 的块内线程数量最大为 1024)

Each block is labeled with (`blockIdx.y, blockIdx.x`). For example, block (1,0) has `blockIdx.y` = 1 and `blockIdx.x` = 0. Note that the ordering of the block and thread labels is such that highest dimension comes first. This notation uses an ordering that is the reverse of that used in the C statements for setting configuration parameters, in which the lowest dimension comes first. This reversed ordering for labeling blocks works better when we illustrate the mapping of thread coordinates into data indexes in accessing multidimensional data.
(
注意是先 `blockIdx.y` ，然后 `blockIdx.x`
)

For example, thread (1,0,2) has `threadIdx.z = 1`, `threadIdx.y = 0`, and `threadIdx.x = 2`.
(
同理，先 `threadIdx.z` ，以此类推
)

(但是 `gridDim` 和 `blockDim` 的参数接受顺序是 `xyz`)
## 3.2 Mapping threads to multidimensional data
the ANSI C standard on the basis of which CUDA C was developed requires the number of columns in `Pin` to be known at compile time for `Pin` to be accessed as a 2 D array.

programmers need to explicitly linearize, or “flatten,” a dynamically allocated 2 D array into an equivalent 1 D array in the current CUDA C.

The linearized access to a 3 D array `P` will be in the form of `P[plane*m*n +row*m+col]`.
## 3.3 Image blur: a more complex kernel
In real CUDA C programs, threads often perform complex operations on their data and need to cooperate with each other.

Image blurring smoothes out abrupt variation of pixel values while preserving the edges that are essential for recognizing the key features of the image.
## 3.4 Matrix multiplication
> **Linear Algebra Functions**
> In the Basic Linear Algebra Subprograms (BLAS), a de facto 
>standard for publishing libraries that perform basic algebra operations, there are three levels of linear algebra functions. As the level increases, the number of operations performed by the function increases.
>level 1: $\symbfit y = \alpha\symbfit x + \symbfit y$
>level 2: $\symbfit y = \alpha \symbfit A \symbfit x + \beta \symbfit y$
>level 3: $\symbfit C = \alpha \symbfit A \symbfit B + \beta \symbfit C$

To implement matrix multiplication using CUDA, we can map the threads in the grid to the elements of the output matrix `P` with the same approach that we used for `colorToGrayscaleConversion`. That is, each thread is responsible for calculating one `P` element.

This thread-to-data mapping effectively divides `P` into tiles, one of which is shown as a light-colored square in Fig. 3.10. Each block is responsible for calculating one of these tiles.
![[PMPP-Fig3.10.png]]

In the situation in which output matrices larger than this limit are to be computed, one can divide the output matrix into submatrices whose sizes can be covered by a grid and use the host code to launch a different grid for each submatrix. Alternatively, we can change the kernel code so that each thread calculates more P elements. We will explore both options later in this book.
## 3.5 Summary
CUDA grids and blocks are multidimensional with up to three dimensions. The multidimensionality of grids and blocks is useful for organizing threads to be mapped to multidimensional data.
The kernel execution configuration parameters define the dimensions of a grid and its blocks. Unique coordinates in `blockIdx` and `threadIdx` allow threads of a grid to identify themselves and their domains of data. It is the programmer’s responsibility to use these variables in kernel functions so that the threads can properly identify the portion of the data to process.
# 4 Compute architecture and scheduling
## 4.1 Architecture of a modern GPU
![[PMPP-Fig4.1.png]]
Fig. 4.1 shows a high-level, CUDA C programmer’s view of the architecture of a typical CUDA-capable GPU. It is organized into an array of highly threaded streaming multiprocessors (SMs). ( 流多处理器数组 )

Each SM has several processing units called streaming processors or CUDA cores (hereinafter referred to as just cores for brevity), shown as small tiles inside the SMs in Fig. 4.1, that share control logic and memory resources. ( 流多处理器由多个流处理器/CUDA 核心构成，共享控制逻辑和内存资源 )

For example, the Ampere A 100 GPU has 108 SMs with 64 cores each, totaling 6912 cores in the entire GPU.

The SMs also come with different on-chip memory structures collectively labeled as “Memory” in Fig. 4.1. (片上存储)

GPUs also come with gigabytes of off-chip device memory, referred to as “Global Memory” in Fig. 4.1. (片外存储/全局显存)

While older GPUs used graphics double data rate synchronous DRAM, more recent GPUs starting with NVIDIA’s Pascal architecture may use HBM (high-bandwidth memory) or HBM 2, which consist of DRAM (dynamic random access memory) modules tightly integrated with the GPU in the same package. For brevity we will broadly refer to all these types of memory as DRAM for the rest of the book.
## 4.2 Block scheduling
When a kernel is called, the CUDA runtime system launches a grid of threads that execute the kernel code. These threads are assigned to SMs on a block-by-block basis. That is, all threads in a block are simultaneously assigned to the same SM. ( 线程以块为单位分配 )
![[PMPP-Fig4.2.png]]
Fig. 4.2 illustrates the assignment of blocks to SMs. Multiple blocks are likely to be simultaneously assigned to the same SM. For example, in Fig. 4.2, three blocks are assigned to each SM. However, blocks need to reserve hardware resources to execute, so only a limited number of blocks can be simultaneously assigned to a given SM. The limit on the number of blocks depends on a variety of factors that are discussed in Section 4.6. ( 一个流处理器可以分配到多个块的线程，当然块的数量有限制)

With a limited number of SMs and a limited number of blocks that can be simultaneously assigned to each SM, there is a limit on the total number of blocks that can be simultaneously executing in a CUDA device. Most grids contain many more blocks than this number. To ensure that all blocks in a grid get executed, the runtime system maintains a list of blocks that need to execute and assigns new blocks to SMs when previously assigned blocks complete execution. ( CUDA 设备同一时间执行的块的数量有限，grid 中的块是需要由运行时系统调度的)

The assignment of threads to SMs on a block-by-block basis guarantees that threads in the same block are scheduled simultaneously on the same SM. This guarantee makes it possible for threads in the same block to interact with each other in ways that threads across different blocks cannot. 1 This includes barrier
synchronization, which is discussed in Section 4.3. It also includes accessing a low-latency shared memory that resides on the SM, which is discussed in Chapter 5, Memory Architecture and Data Locality. ( 由于线程以块为单位分配，同一块内的线程保证是被同时调度的，故可以相互通讯，而块与块之间则无法保证，因此无法相互通讯)
## 4.3 Synchronization and transparent scalability
CUDA allows threads in the same block to coordinate their activities using the barrier synchronization function `__syncthreads()`. ( 同步同一块内的所有线程 )

When a thread calls `__syncthreads()`, it will be held at the program location of the call until every thread in the same block reaches that location. This ensures that all threads in a block have completed a phase of their execution before any of them can move on to the next phase.

In CUDA, if a `__syncthreads()` statement is present, it must be executed by all threads in a block. When a `__syncthreads()` statement is placed in an if statement, either all threads in a block execute the path that includes the `__syncthreads()` or none of them does. For an if-then-else statement, if each path has a `__syncthreads()` statement, either all threads in a block execute the then-path or all of them execute the else-path. ( 否则会导致死锁 deadlock)

Barrier synchronization imposes execution constraints on threads within a block. These threads should execute in close time proximity with each other to avoid excessively long waiting times. More important, the system needs to make sure that all threads involved in the barrier synchronization have access to the necessary resources to eventually arrive at the barrier.

The CUDA runtime system satisfies this constraint by assigning execution resources to all threads in a block as a unit, as we saw in Section 4.2. Not only do all threads in a block have to be assigned to the same SM, but also they need to be assigned to that SM simultaneously. That is, a block can begin execution only when the runtime system has secured all the resources needed by all threads in the block to complete execution. This ensures the time proximity of all threads in a block and prevents an excessive or even indefinite waiting time during barrier synchronization. ( 运行时系统只在保证了块内所有线程所需的资源的情况下才会将块内所有线程同时分配给流处理器 )

![[PMPP-Fig4.5.png]]
This leads us to an important tradeoff in the design of CUDA barrier synchronization. By not allowing threads in different blocks to perform barrier synchronization with each other, the CUDA runtime system can execute blocks in any order relative to each other, since none of them need to wait for each other. ( 块与块之间的执行顺序可以任意 )

This flexibility enables scalable implementations, as shown in Fig. 4.5. Time in the figure progresses from top to bottom. In a low-cost system with only a few execution resources, one can execute a small number of blocks at the same time, portrayed as executing two blocks a time on the left-hand side of Fig. 4.5. In a higher-end implementation with more execution resources, one can execute many blocks at the same time, portrayed as executing four blocks at a time on the right-hand side of Fig. 4.5. A high-end GPU today can execute hundreds of blocks simultaneously.

The ability to execute the same application code with a wide range of speeds allows the production of a wide range of implementations according to the cost, power, and performance requirements of different market segments. The ability to execute the same application code on different hardware with different amounts of execution resources is referred to as *transparent scalability*, which reduces the burden on application developers and improves the usability of applications.
## 4.4 Warps and SIMD hardware
Conceptually, one should assume that threads in a block can execute in any order with respect to each other. The correctness of executing a kernel should not depend on any assumption that certain threads will execute in synchrony with each other without the use of barrier synchronizations. ( 块内线程的执行顺序不可预测，是任意的 )

Thread scheduling in CUDA GPUs is a hardware implementation concept and therefore must be discussed in the context of specific hardware implementations.

In most implementations to date, once a block has been assigned to an SM, it is further divided into 32-thread units called *warps*.

Each warp consists of 32 threads of consecutive threadIdx values: threads 0 through 31 form the first warp, threads 32 through 63 form the second warp, and so on. ( 线程束中的线程是连续的 )

Blocks are partitioned into warps on the basis of thread indices. If a block is organized into a one-dimensional array, that is, only threadIdx. x is used, the partition is straightforward. The threadIdx. x values within a warp are consecutive and increasing. For a warp size of 32, warp 0 starts with thread 0 and ends with thread 31, warp 1 starts with thread 32 and ends with thread 63, and so on. In general, warp n starts with thread 32 3 n and ends with thread $32 \times (n+1) - 1$.

For a block whose size is not a multiple of 32, the last warp will be padded with inactive threads to fill up the 32 thread positions. For example, if a block has 48 threads, it will be partitioned into two warps, and the second warp will be padded with 16 inactive threads

For blocks that consist of multiple dimensions of threads, the dimensions will be projected into a linearized row-major layout before partitioning into warps. The linear layout is determined by placing the rows with larger y and z coordinates after those with lower ones. That is, if a block consists of two dimensions of threads, one will form the linear layout by placing all threads whose `threadIdx.y` is 1 after those whose `threadIdx.y` is 0. Threads whose `threadIdx.y` is 2 will be placed after those whose `threadIdx.y` is 1, and so on. Threads with the same `threadIdx.y` value are placed in consecutive positions in increasing `threadIdx.x` order.

For a three-dimensional block, we first place all threads whose `threadIdx.z` value is 0 into the linear order. These threads are treated as a two-dimensional block, as shown in Fig. 4.7. All threads whose `threadIdx.z` value is 1 will then be placed into the linear order, and so on. For example, for a three-dimensional $2 \times 8 \times 4$ block (four in the x dimension, eight in the y dimension, and two in the z dimension), the 64 threads will be partitioned into two warps, with $T_{0,0,0}$ through $T_{0,7,3}$ in the first warp and $T_{1,0,0}$ through $T_{1,7,3}$ in the second warp.

An SM is designed to execute all threads in a warp following the single-instruction, multiple-data (SIMD) model. That is, at any instant in time, one instruction is fetched and executed for all threads in the warp (see the “Warps and SIMD Hardware” sidebar) ( 任意时刻，同一线程束内的线程执行同一指令 )

Fig. 4.8 shows how the cores in an SM are grouped into processing blocks in which every 8 cores form a processing block and share an instruction fetch/dispatch unit. As a real example, the Ampere A 100 SM, which has 64 cores, is organized into four processing blocks with 16 cores each. Threads in the same warp are assigned to the same processing block, which fetches the instruction for the warp and executes it for all threads in the warp at the same time.
![[PMPP-Fig4.8.png]]

The advantage of SIMD is that the cost of the control hardware, such as the instruction fetch/dispatch unit, is shared across many execution units. ( 控制硬件的开销由多个执行单元分摊 )

![[PMPP-SIMD figure.png]]
The processor, which corresponds to a processing block in Figure 4.8, has only one control unit that fetches and dispatches instructions. The same control signals (arrows that go from the Control Unit to the Processing Units in Figure 4.8 ) go to multiple processing units that each correspond to a core in the SM, each of which executes one of the threads in a warp. Since all processing units are controlled by the same instruction in the Instruction Register (IR) of the Control Unit, their execution differences are due to the different data operand values in the register files. This is called Single-Instruction-Multiple-Data (SIMD) in processor design. For example, although all processing units (cores) are controlled by an instruction, such as add r 1, r 2, r 3, the contents of r 2 and r 3 are different in different processing units. 
Control units in modern processors are quite complex, including sophisticated logic for fetching instructions and access ports to the instruction cache. Having multiple processing units to share a control unit can result in significant reduction in hardware manufacturing cost and power consumption.
## 4.5 Control divergence
SIMD execution works well when all threads within a warp follow the same execution path, more formally referred to as control flow,

However, when threads within a warp take different control flow paths, the SIMD hardware will take multiple passes through these paths, one pass for each path. For example, for an if-else construct, if some threads in a warp follow the if-path while others follow the else path, the hardware will take two passes. One pass executes the threads that follow the if-path, and the other executes the threads that follow the else-path. During each pass, the threads that follow the other path are not allowed to take effect. ( 控制发散即线程束内线程没有全部遵循同一控制流 )

当同一 warp 中的线程遵循不同的执行路径时，我们说这些线程表现出控制发散 ( control divergence )，即它们在执行中发散 
发散的 warp 执行的多遍方法 ( multipass approach ) 扩展了 SIMD 硬件的能力以实现 CUDA 线程完整语义，虽然硬件为 warp 中的所有线程执行相同的指令，但它选择性地让这些线程只在对应于它们所走路径的遍 (pass) 中生效，使每个线程看起来都走自己的控制流路径，这在利用 SIMD 硬件降低成本的同时，保持了线程的独立性，然而，发散的成本是硬件需要执行额外的遍，以允许 warp 中的不同线程做出自己的决策，并且每遍中不活跃线程都需要消耗执行资源

![[PMPP-Fig4.9.png]]
Fig. 4.9 shows an example of how a warp would execute a divergent if-else statement. In this example, when the warp consisting of threads 0-31 arrives at the if-else statement, threads 0-23 take the then-path, while threads 24-31 take the else-path. In this case, the warp will do a pass through the code in which threads 0-23 execute A while threads 24-31 are inactive. The warp will also do another pass through the code in which threads 24-31 execute B while threads 0-23 are inactive. The threads in the warp then reconverge and execute C. In the Pascal architecture and prior architectures, these passes are executed sequentially, meaning that one pass is executed to completion followed by the other pass. From the Volta architecture onwards, the passes may be executed concurrently, meaning that the execution of one pass may be interleaved with the execution of another pass. This feature is referred to as independent thread scheduling.

One can determine whether a control construct can result in thread divergence by inspecting its decision condition. If the decision condition is based on `threadIdx` values, the control statement can potentially cause thread divergence. Similarly, a loop can cause thread divergence if its loop condition is based on thread index values.
## 4.6 Warp scheduling and latency tolerance
When threads are assigned to SMs, there are usually more threads assigned to an SM than there are cores in the SM. That is, each SM has only enough execution units to execute a subset of all the threads assigned to it at any point in time.

the hardware can execute instructions only for a subset of all warps in the SM

When an instruction to be executed by a warp needs to wait for the result of a previously initiated long-latency operation, the warp is not selected for execution. Instead, another resident warp that is no longer waiting for results of previous instructions will be selected for execution. If more than one warp is ready for execution, a priority mechanism is used to select one for execution. This mechanism of filling the latency time of operations from some threads with work from other threads is often called “latency tolerance” or “latency hiding” ( 通过线程束的调度隐藏延迟 )

With enough warps around, the hardware will likely find a warp to execute at any point in time, thus making full use of the execution hardware while the instructions of some warps wait for the results of these long-latency operations. ( 在任意时刻都可以调度出一个线程束执行，避免硬件闲着 )

The selection of warps that are ready for execution does not introduce any idle or wasted time into the execution timeline, which is referred to as *zero-overhead thread scheduling*

Zero-overhead scheduling refers to the GPU’s ability to put a warp that needs to wait for a long-latency instruction result to sleep and activate a warp that is ready to go without introducing any extra idle cycles in the processing units. Traditional CPUs incur such idle cycles because switching the execution from one thread to another requires saving the execution state (such as register contents of the out-going thread) to memory and loading the execution state of the incoming thread from memory. GPU SMs achieves zero-overhead scheduling by holding all the execution states for the assigned warps in the hardware registers so there is no need to save and restore states when switching from one warp to another. ( GPU 在寄存器中保存了所有线程束执行所需要的状态，因此没有上下文切换的开销 )

For latency tolerance to be effective, it is desirable for an SM to have many more threads assigned to it than can be simultaneously supported with its execution resources to maximize the chance of finding a warp that is ready to execute at any point in time. For example, in an Ampere A 100 GPU, an SM has 64 cores but can have up to 2048 threads assigned to it at the same time. Thus the SM can have up to 32 times more threads assigned to it than its cores can support at any given clock cycle. This oversubscription of threads to SMs is essential for latency tolerance. It increases the chances of finding another warp to execute when a currently executing warp encounters a long-latency operation. ( 保证线程束数量足够多，方便随时都可以调度出来一个 )
## 4.7 Resource partitioning and occupancy
it may not always be possible to assign to the SM the maximum number of warps that the SM supports. The ratio of the number of warps assigned to an SM to the maximum number it supports is referred to as *occupancy.*

The execution resources in an SM include registers, shared memory (discussed in Chapter 5, Memory Architecture and Data Locality), thread block slots, and thread slots. ( 寄存器、共享存储、线程块槽、线程槽 ) These resources are dynamically partitioned across threads to support their execution. ( 占用率 = SM 使用的线程数量/线程槽数量 )

A 100 每个流处理器有 32 个线程块槽、2048 个线程槽，每个线程块分到的线程数量即 2048/线程块数量，注意线程块数量为 $[1,32]$，此即动态划分 ( dynamic partitioning)

Dynamic partitioning of resources can lead to subtle interactions between resource limitations, which can cause underutilization of resources. ( 例如分给每个线程块的线程数量小于 2048/线程块数量，即 blocksize 设置不当 )

Another situation that could negatively affect occupancy occurs when the maximum number of threads per block is not divisible by the block size. In the example of the Ampere A 100, we saw that up to 2048 threads per SM can be supported. However, if a block size of 768 is selected, the SM will be able to accommodate only 2 thread blocks (1536 threads), leaving 512 thread slots unutilized. In this case, neither the maximum threads per SM nor the maximum blocks per SM are reached.

automatic variables declared in a CUDA kernel are placed into registers. Some kernels may use many automatic variables, and others may use few of them. Therefore one should expect that some kernels require many registers per thread and some require few. By dynamically partitioning registers in an SM across threads, the SM can accommodate many blocks if they require few registers per thread and fewer blocks if they require more registers per thread.

A 100 中每个流处理器最多有 $2^{16} = 65536$ 个寄存器

To run at full occupancy, each SM needs enough registers for 2048 threads, which means that each thread should not use more than (65,536 registers)/(2048 threads) = 32 registers per thread. For example, if a kernel uses 64 registers per thread, the maximum number of threads that can be supported with 65,536 registers is 1024 threads. In this case, the kernel cannot run with full occupancy regardless of what the block size is set to be.

In some cases, the compiler may perform register spilling ( 寄存器溢出，即寄存器数量不足时，将部分内容存于内存 ) to reduce the register requirement per thread and thus elevate the level of occupancy. However, this is typically at the cost of increased execution time for the threads to access the spilled register values from memory and may cause the total execution time of the grid to increase.

Assume that a programmer implements a kernel that uses 31 registers per thread and configures it with 512 threads per block. In this case, the SM will have (2048 threads)/(512 threads/block) = 4 blocks running simultaneously. These threads will use a total of (2048 threads) $\times$ (31 registers/thread) = 63,488 registers, which is less than the 65,536 register limit. Now assume that the programmer declares another two automatic variables in the kernel, bumping the number of registers used by each thread to 33. The number of registers required by 2048 threads is now 67,584 registers, which exceeds the register limit. The CUDA runtime system may deal with this situation by assigning only 3 blocks to each SM instead of 4, thus reducing the number of registers required to 50,688 registers. However, this reduces the number of threads running on an SM from 2048 to 1536; that is, by using two extra automatic variables, the program saw a reduction in occupancy from 100% to 75%. This is sometimes referred to as a “performance cliff,” in which a slight increase in resource usage can result in significant reduction in parallelism and performance achieved (Ryoo et al., 2008). ( 额外多使用了自动变量可能导致占用率大幅降低 )
## 4.8 Querying device properties
There is often a need for the application to query the available resources and capabilities of the underlying hardware

The amount of resources in each CUDA device SM is specified as part of the *compute capability* of the device. The compute capability of GPUs tends to increase from generation to generation. The Ampere A 100 GPU has compute capability 8.0.

In CUDA C, there is a built-in mechanism for the host code to query the properties of the devices that are available in the system. The CUDA runtime system (device driver 设备驱动) has an API function `cudaGetDeviceCount` that returns the number of available CUDA devices in the system. The host code can find out the number of available CUDA devices by using the following statements:
```c
int devCount;
cudaGetDeviceCount(&devCount);
```

The CUDA runtime numbers all the available devices in the system from 0 to `devCount-1`. It provides an API function `cudaGetDeviceProperties` that returns the properties of the device whose number is given as an argument.
```c
cudaDeviceProp devProp;
cudaGetDeviceProperties(&devProp, 0)
```
The built-in type `cudaDeviceProp` is a C struct type with fields that represent the properties of a CUDA device.

The number of SMs in the device is given in `devProp.multiProcessorCount`.
the clock frequency of the device is in `devProp.clockRate`.

the maximum number of threads allowed along each dimension of a block in fields `devProp.maxThreadsDim[0]` (for the x dimension), `devProp.maxThreadsDim[1]` (for the y dimension), and `devProp.maxThreadsDim[2]` (for the z dimension).
An example of use of this information is for an automated tuning system to set the range of block dimensions when evaluating the best performing block dimensions for the underlying hardware

Similarly, it can find the maximum number of blocks allowed along each dimension of a grid in `devProp. maxGridSize[0]` (for the x dimension), `devProp.maxGridSize[1]` (for the y dimension), and `devProp.maxGridSize[2]` (for the z dimension). A typical use of this information is to determine whether a grid can have enough threads to handle the entire dataset or some kind of iterative approach is needed

The field `devProp.regsPerBlock` gives the number of registers that are available in each SM. ( 注意该变量给出的是整个 SM 可用的寄存器数量，在一些计算能力等级，SM 内线程块可用的寄存器数量是要比 SM 的小的 )

The size of warps can be obtained from the `devProp.warpSize` field.
## 4.9 Summary
A GPU is organized into SM, which consist of multiple processing blocks of cores that share control logic and memory resources. When a grid is launched, its blocks are assigned to SMs in an arbitrary order, resulting in transparent scalability of CUDA applications. The transparent scalability comes with a limitation: Threads in different blocks cannot synchronize with each other.

Threads are assigned to SMs for execution on a block-by-block basis. Once a block has been assigned to an SM, it is further partitioned into warps. Threads in a warp are executed following the SIMD model. If threads in the same warp diverge by taking different execution paths, the processing block executes these paths in passes in which each thread is active only in the pass corresponding to the path that it takes.

An SM may have many more threads assigned to it than it can execute simultaneously. At any time, the SM executes instructions of only a small subset of its resident warps. This allows the other warps to wait for long-latency operations without slowing down the overall execution throughput of the massive number of processing units. The ratio of the number of threads assigned to the SM to the maximum number of threads it can support is referred to as occupancy. The higher the occupancy of an SM, the better it can hide long-latency operations.

Each CUDA device imposes a potentially different limitation on the amount of resources available in each SM. For example, each CUDA device has a limit on the number of blocks, the number of threads, the number of registers, and the amount of other resources that each of its SMs can accommodate. For each kernel, one or more of these resource limitations can become the limiting factor for occupancy. CUDA C provides programmers with the ability to query the resources available in a GPU at runtime.
# 5 Memory architecture and data locality
The CUDA kernels that we have studied so far will likely achieve only a tiny fraction of the potential speed of the underlying hardware. This poor performance is because global memory, which is typically implemented with off-chip DRAM, tends to have long access latency (hundreds of clock cycles) and finite access bandwidth. ( 访问全局存储需要上百个时钟周期，慢 )

While having many threads available for execution can theoretically tolerate long memory access latencies, one can easily run into a situation in which traffic congestion in the global memory access paths prevents all but a very few threads from making progress, thus rendering some of the cores in the streaming multiprocessors (SMs) idle. ( 全局存储无法一次处理过多的线程访存，导致堵车，处理器等存储器 )

To circumvent such congestion, GPUs provide a number of additional on-chip memory resources for accessing data that can remove the majority of traffic to and from the global memory.
## 5.1 Importance of memory access efficiency
![[PMPP-Fig5.1.png]]
In every iteration of the loop, two global memory accesses are performed for one floating-point multiplication and one floating-point addition. Thus the ratio of floating-point operations (FLOP) to bytes (B) accessed from global memory is 2 FLOP to 8 B, or 0.25 FLOP/B. We will refer to this ratio as the *compute to global memory access ratio*, defined as the number of FLOPs performed for each byte access from the global memory within a region of a program. ( 每字节全局存储访问执行的浮点操作数量 )
This ratio is sometimes also referred to as *arithmetic intensity* or *computational intensity* in the literature.

The compute to global memory access ratio has major implications for the performance of a CUDA kernel. For example, the Ampere A 100 GPU has a peak global memory bandwidth of 1555 GB/second. Since the matrix multiplication kernel performs 0.25 OP/B, the global memory bandwidth limits the throughput of single-precision FLOPs that can be performed by the kernel to 389 giga FLOPs per second (GFLOPS), obtained by multiplying 1555 GB/ second with 0.25 FLOP/B. However, 389 GFLOPS is only 2% of the peak single-precision operation throughput of the A 100 GPU, which is 19,500 GFLOPS.

We refer to programs whose execution speed is limited by memory bandwidth as *memory-bound* programs

To achieve higher performance for this kernel, we need to increase the compute to global memory access ratio of the kernel by reducing the number of global memory accesses it performs. For example, to fully utilize the 19,500 GFLOPS that the A 100 GPU provides, a ratio of at least (19,500 GOP/second)/ (1555 GB/second)=12.5 OP/B is needed. The extent to which such a ratio can be achieved depends on the intrinsic data reuse in the computation at hand.

The execution speed of matrix multiplication functions can vary by orders of magnitude, depending on the level of reduction of global memory accesses.
## 5.2 CUDA memory types
![[PMPP-Fig5.2.png]]
the constant memory supports short-latency, high-bandwidth read-only access by the device.

Another type of memory is the local memory, which can also be read and written. The local memory is actually placed in global memory and has similar access latency, but it is not shared across threads. Each thread has its own section of global memory that it uses as its own private local memory where it places data that is private to the thread but cannot be allocated in registers. This data includes statically allocated arrays, spilled registers, and other elements of the thread’s call stack. ( 每个线程在全局存储中的私有区域即局部存储 )

Registers and shared memory in Fig. 5.2 are on-chip memories. Variables that reside in these types of memory can be accessed at very high speed in a highly parallel manner. ( 寄存器和共享存储是片上存储，高速 ) Registers are allocated to individual threads; each thread can access only its own registers. A kernel function typically uses registers to hold frequently accessed variables that are private to each thread.

Shared memory is allocated to thread blocks; all threads in a block can access shared memory variables declared for the block. Shared memory is an efficient means by which threads can cooperate by sharing their input data and intermediate results. ( 块内线程使用共享存储协作 )

GPUs achieve zero-overhead scheduling by keeping the registers of all the threads that are scheduled on the processing block in the processing block’s register file. This way, switching between warps of threads is instantaneous because the registers of the incoming threads are already in the register file. Consequently, GPU register files need to be substantially larger than CPU register files. (GPU 的寄存器堆远大于 CPU )

We also saw in Chapter 4, Compute Architecture and Scheduling, that GPUs support dynamic resource partitioning where an SM may provision few registers per thread and execute a large number of threads, or it my provision more registers per thread and execute fewer threads. For this reason, GPU register files need to be designed to support such dynamic partitioning of registers. In contrast, the CPU register architecture dedicates a fixed set of registers per thread regardless of the thread’s actual demand for registers. ( GPU 的寄存器堆要支持能动态划分寄存器 )

![[PMPP-Fig5.3.png]]
virtually all modern processors find their root in the model proposed by John von Neumann in 1945, which is shown in Fig. 5.3. CUDA devices are no exception. The global memory in a CUDA device maps to the Memory box in Fig. 5.3.
The processor box corresponds to the processor chip boundary that we typically see today. The global memory is off the processor chip and is implemented with DRAM technology, which implies long access latencies and relatively low access bandwidth. ( 全局存储在处理器芯片外，是 DRAM )

The registers correspond to the “Register File” of the von Neumann model. The Register File is on the processor chip, which implies very short access latency and drastically higher access bandwidth when compared to the global memory. ( 寄存器堆在片上 )

In a typical device, the aggregated access bandwidth of all the register files across all the SMs is at least two orders of magnitude higher than that of the global memory. ( 寄存器堆和全局存储的访问时间一般相差两个数量级 )

A subtler point is that each access to registers involves fewer instructions than an access to the global memory. Arithmetic instructions in most modern processors have “built-in” register operands. if an operand value is in the global memory, the processor needs to perform a memory load operation to make the operand value available to the ALU. ( 操作数都在寄存器内，一条指令 `fadd` 就足够，否则还需要先通过 `load` 指令载入操作数 ) Since the processor can fetch and execute only a limited number of instructions per clock cycle, the version with an additional load will likely take more time to process than the one without. This is another reason why placing the operands in registers can improve execution speed.

 Finally, there is yet another subtle reason why placing an operand value in registers is preferable. In modern computers the energy that is consumed for accessing a value from the register file is at least an order of magnitude lower than for accessing a value from the global memory. ( 访问寄存器组的能耗要比访问全局存储低一个数量级 )

![[PMPP-Fig5.4.png]]
Shared memory is designed as part of the memory space that resides on the processor chip. When the processor accesses data that resides in the shared memory, it needs to perform a memory load operation, just as in accessing data in the global memory. However, because shared memory resides on-chip, it can be accessed with much lower latency and much higher throughput than the global memory. Because of the need to perform a load operation, shared memory has longer latency and lower bandwidth than registers. In computer architecture terminology the shared memory is a form of *scratchpad memory*. ( 共享存储的速度介于寄存器和全局存储之间 )

Threads in a block can be spread across these processing units. Therefore the hardware implementations of the shared memory in these CUDA devices are typically designed to allow multiple processing units to simultaneously access its contents to support efficient data sharing among threads in a block.

![[PMPP-Table5.1.png]]
Table 5.1 presents the CUDA syntax for declaring program variables into the various memory types. Each such declaration also gives its declared CUDA variable a scope and lifetime.
Scope identifies the set of threads that can access the variable: a single thread only, all threads of a block, or all threads of all grids. ( 变量可以被谁访问 ) 
Lifetime tells the portion of the program’s execution duration when the variable is available for use: either within a grid’s execution or throughout the entire application. ( 变量存在多久 ) If a variable’s lifetime is within a grid’s execution, it must be declared within the kernel function body and will be available for use only by the kernel’s code. If the kernel is invoked several times, the value of the variable is not maintained across these invocations. Each invocation must initialize the variable in order to use it. ( 生命周期为 Grid 即生命周期为当前的内核调用，需要声明于内核函数之中 )

On the other hand, if a variable’s lifetime is throughout the entire application, it must be declared outside of any function body. The contents of these variables are maintained throughout the execution of the application and available to all kernels. ( 内核函数之外声明的变量生命周期等同于整个应用程序 )

When a kernel function declares an automatic variable, a private copy of that variable is generated for every thread that executes the kernel function. When a thread terminates, all its automatic variables cease to exist.
accessing these variables is extremely fast and parallel, but one must be careful not to exceed the limited capacity of the register storage in the hardware implementations.

Automatic array variables are not stored in registers. ( There are some exceptions to this rule. The compiler may decide to store an automatic array into registers if all accesses are done with constant index values. ) Instead, they are stored into the thread’s local memory. The scope of these arrays, like that of automatic scalar variables, is limited to individual threads. That is, a private version of each automatic array is created for and used by every thread. Once a thread terminates its execution, the contents of its automatic array variables cease to exist. 

带 `__shared__` 或 `__device__ __shared__` 关键字的变量通常在内核函数或设备函数内声明，Shared variables reside in the shared memory. The scope of a shared variable is within a thread block; that is, all threads in a block see the same version of a shared variable. A private version of the shared variable is created for and used by each block during kernel execution. The lifetime of a shared variable is within the duration of the kernel execution. When a kernel terminates its grid’s execution, the contents of its shared variables cease to exist.

CUDA programmers often use shared variables to hold the portion of global memory data that is frequently used and reused in an execution phase of the kernel. One may need to adjust the algorithms that are used to create execution phases that heavily focus on small portions of the global memory data, as we will demonstrate with matrix multiplication in Section 5.4.

带 `__constant__` 或 `__device__ __constant__` 关键字的变量通常在函数体外声明，Declaration of constant variables must be outside any function body. The scope of a constant variable is all grids, meaning that all threads in all grids see the same version of a constant variable. The lifetime of a constant variable is the entire application execution. Constant variables are often used for variables that provide input values to kernel functions. ( 常变量的作用域和声明周期和全局变量一致，一般用于给内核函数提供输入 ) The values of the constant variables cannot be changed by the kernel function code. Constant variables are stored in the global memory but are cached for efficient access. With appropriate access patterns, accessing constant memory is extremely fast and parallel. Currently, the total size of constant variables in an application is limited to 65,536 bytes (64 KB).

仅带 `__device__` 关键字的变量为全局变量，One important advantage of global variables is that they are visible to all threads of all kernels. Their contents also persist through the entire execution. Thus global variables can be used as a means for threads to collaborate across blocks.
However, one must be aware that there is currently no easy way to synchronize between threads from different thread blocks or to ensure data consistency across threads in accessing global memory other than using atomic operations or terminating the current kernel execution. Therefore global variables are often used to pass information from one kernel invocation to another kernel invocation ( 用于在两次内核调用之间传递信息 )

In CUDA, pointers can be used to point to data objects in the global memory. ( 指针一般仅在访问全局内存时使用 ) There are two typical ways in which pointer use arises in kernel and device functions. First, if an object is allocated by a host function, the pointer to the object is initialized by memory allocation API functions such as `cudaMalloc` and can be passed to the kernel function as a parameter, The second type of use is to assign the address of a variable that is declared in the global memory to a pointer variable. For example, the statement `{float* ptr=&GlobalVar;}` in a kernel function assigns the address of `GlobalVar` into an automatic pointer variable `ptr`.
## 5.3 Tiling for reduced memory traffic
We have an intrinsic tradeoff in the use of device memories in CUDA: The global memory is large but slow, whereas the shared memory is small but fast. A common strategy is to partition the data into subsets called *tiles* so that each tile fits into the shared memory. The term tile draws on the analogy that a large wall (i.e., the global memory data) can be covered by small tiles (i.e., subsets that can each fit into the shared memory). An important criterion is that the kernel computation on these tiles can be done independently of each other. ( 将全局存储中的数据划分，分配至一个个共享存储 )

![[PMPP-Fig5.6.png]]
Fig. 5.6 shows the global memory accesses done by all threads in block $_{0,0}$. The threads are listed in the vertical direction, with time of access increasing from left to right in the horizontal direction. Note that each thread accesses four elements of M and four elements of N during its execution. Among the four threads highlighted, there is a significant overlap in the M and N elements that they access. For example, thread $_{0,0}$ and thread $_{0,1}$ both access M $_{0,0}$ as well as the rest of row 0 of M. Similarly, thread $_{0,1}$ and thread $_{1,1}$ both access N $_{0,1}$ as well as the rest of column 1 of N. ( 注意到不同线程需要访问的元素是存在大量重合的 )

The kernel in Fig. 3.11 is written so that both thread $_{0,0}$ and thread $_{0,1}$ access row 0 elements of M from the global memory. If we can somehow manage to have thread 0,0 and thread 0,1 collaborate so that these M elements are loaded from global memory only once, we can reduce the total number of accesses to the global memory by half. In fact, we can see that every M and N element is accessed exactly twice during the execution of block $_{0,0}$. Therefore if we can have all four threads collaborate in their accesses to global memory, we can reduce the traffic to the global memory by half.

The reader should verify that the potential reduction in global memory traffic in the matrix multiplication example is proportional to the dimension of the blocks that are used. With Width $\times$ Width blocks, the potential reduction of global memory traffic would be Width. That is, if we use 16 $\times$ 16 blocks, we can potentially reduce the global memory traffic to 1/16 of the original level through collaboration between threads.

![[PMPP-Fig5.7.png]]
我们将介绍一种平铺矩阵乘法算法 (tiled matrix multiplication algorithm)，其基本思想是让线程协作地将 M 和 N 元素的子集加载到共享内存中，然后它们分别在点积计算中使用这些元素，要注意的是，共享内存的大小相当小，当将这些 M 和 N 元素加载到共享内存时，必须小心不要超过共享内存的容量，因此需要将 M 和 N 矩阵划分为更小的平铺，这些平铺的大小应该可以适应共享内存的大小，在最简单的情况下，平铺的尺寸等于块的尺寸，如图 5.7 所示

在图 5.7 中，我们将 M 和 N 划分为 2 x 2 的平铺，现在，每个线程执行的点积计算将被划分为多个阶段，在每个阶段，块中的所有线程协作将 M 的平铺和 N 的平铺 (a tile of M and a tile of N) 加载到共享内存中，我们可以让块中的每个线程将一个 M 元素和一个 N 元素加载到共享内存中，如图 5.8 所示
![[PMPP-Fig5.8.png]]
图 5.8 的每一行显示了一个线程的执行活动，请注意，时间从左到右推进，我们只需要了解块 (0,0) 中线程的活动，其他块的行为都是相同的
M 元素的共享内存数组称为 Mds，N 元素的共享内存数组称为 Nds

在第一阶段开始时，块 (0,0) 的四个线程协作将 M 的平铺加载到共享内存中：线程 (0,0) 将 M (0,0) 加载到 Mds (0,0) 中，线程 (0,1) 将 M (0,1) 加载到 Mds (0,1) 中，线程 (1,0) 将 M (1,0) 加载到 Mds (1,0) 中，线程 (1,1) 将 M (1,1) 加载到 Mds (1,1) 中，N 的平铺也以类似的方式加载

将 M 和 N 的两个平铺加载到共享内存后，这些元素将用于计算点积
请注意，共享内存中的每个值都使用了两次，例如，由线程 (1,1) 加载到 Mds (1,1) 中的 M (1,1) 值，被线程 (1,0) 和线程 (1,1) 各使用一次
通过将每个全局内存值加载到共享内存中，使其可以多次使用，我们减少了对全局内存的访问次数，在这种情况下，我们将对全局内存的访问次数减少了 2 倍，如果平铺是 NxN 元素，读者应该验证减少的倍数是 N

Note that the calculation of each dot product is now performed in two phases, shown as phase 1 and phase 2 in Fig. 5.8. In each phase, each thread accumulates products of two pairs of the input matrix elements into the Pvalue variable.

In general, if an input matrix is of dimension Width and the tile size is `TILE_WIDTH`, the dot product would be performed in `Width/TILE_WIDTH` phases. ( `TILE_WIDTH` 显然越大越好)

The creation of these phases is key to the reduction of accesses to the global memory. With each phase focusing on a small subset of the input matrix values, the threads can collaboratively load the subset into the shared memory and use the values in the shared memory to satisfy their overlapping input needs in the phase.

Note also that Mds and Nds are reused across phases. In each phase, the same Mds and Nds are reused to hold the subset of M and N elements used in the phase. This allows a much smaller shared memory to serve most of the accesses to global memory. This is because each phase focuses on a small subset of the input matrix elements. Such focused access behavior is called *locality.* (局部性) When an algorithm exhibits locality, there is an opportunity to use small, high-speed memories to serve most of the accesses and remove these accesses from the global memory.
## 5.4 A tiled matrix multiplication kernel
![[PMPP-Fig5.9.png]]
Lines 11 and 12 determine the row index and column index, respectively, of the P element that the thread is to produce. As shown in line 12, the horizontal (x) position, or the column index of the P element to be produced by a thread, can be calculated as `bx * TILE_WIDTH+tx`. This is because each block covers `TILE_WIDTH` elements of P in the horizontal dimension.

Line 16 of Fig. 5.9 marks the beginning of the loop that iterates through all the phases of calculating the P element. Each iteration of the loop corresponds to one phase of the calculation shown in Fig. 5.8. The `ph` variable indicates the number of phases that have already been done for the dot product. Recall that each phase uses one tile of M and one tile of N elements. Therefore at the beginning of each phase, `ph * TILE_WIDTH` pairs of M and N elements have been processed by previous phases.

The barrier `__syncthreads()` in line 21 ensures that all threads have finished loading the tiles of M and N into Mds and Nds before any of them can move forward.

The barrier `__syncthreads()` in line 26 ensures that all threads have finished using the M and N elements in the shared memory before any of them move on to the next iteration and load the elements from the next tiles. Thus none of the threads would load the elements too early and corrupt the input values of other threads.
![[PMPP-Fig5.10.png]]
The two `__syncthreads()` calls in lines 21 and 26 demonstrate two different types of data dependence that parallel programmers often have to reason about when they are coordinating between threads. 
The first is called a read-after-write dependence because threads must wait for data to be written to the proper place by other threads before they try to read it. ( 在其他所有线程完成写入后，才能开始读取 )
The second is called a write-after-read dependence because a thread must wait for the data to be read by all threads that need it before overwriting it. ( 在其他所有线程完成读取后，才能进行覆盖写 )

Other names for read-after-write and write-after-read dependences are true and false dependences, respectively. 
A read-after-write dependence is a true dependence because the reading thread truly needs the data supplied by the writing thread, so it has no choice but to wait for it. 
A write-after-read dependence is a false dependence because the writing thread does not need any data from the reading thread. The dependence is caused by the fact that they are reusing the same memory location and would not exist if they used different locations.

The loop nest from line 16 to line 28 illustrates a technique called *stripmining*(条带化/条带挖掘), which takes a long-running loop and break it into phases. Each phase involves an inner loop that executes a few consecutive iterations of the original loop. The original loop becomes an outer loop whose role is to iteratively invoke the inner loop so that all the iterations of the original loop are executed in their original order. By adding barrier synchronizations before and after the inner loop, we force all threads in the same block to focus their work on the same section of input data during each phase. Strip-mining is an important means to creating the phases that are needed by tiling in data parallel programs. ( 简而言之，就是拆分循环 )

After all phases of the dot product are complete, the execution exits the outer loop. In Line 29, all threads write to their `P` element using the linearized index calculated from `Row` and `Col`.

The benefit of the tiled algorithm is substantial. For matrix multiplication, the global memory accesses are reduced by a factor of `TILE_WIDTH`. With 16 $\times$ 16 tiles, one can reduce the global memory accesses by a factor of 16. This increases the compute to global memory access ratio from 0.25 OP/B to 4 OP/B.
(
ref to Figure 5.9，每个 thread 进行了两次全局存储访问，取得 8 B 数据，执行了 `TILE_WIDTH` 次乘加运算，共 `TILE_WIDTH * 2` FLOP，故算数密度为 4 OP/B
)

One can further optimize the code to reduce the number of global memory accesses and improve throughput. We will see some of these optimizations later in the book, while other advanced optimizations will not be covered. Because of the importance of matrix multiplication in many domains, there are highly optimized libraries, such as cuBLAS and CUTLASS, that already incorporate many of these advanced optimizations.
## 5.5 Boundary checks
We now extend the tiled matrix multiplication kernel to handle matrices with arbitrary width.

Problematic accesses can arise in all phases. Fig. 5.12 shows the memory access pattern of block 1,1 during phase 0.
![[PMPP-Fig5.12.png]]
Note that these problematic accesses cannot be prevented by simply excluding the threads that do not calculate valid P elements. For example, thread 1,0 in block 1,1 does not calculate any valid P element. However, it needs to load M 2,1 during phase 0 for other threads in block 1,1 to use. Furthermore, note that some threads that calculate valid P elements will attempt to access M or N elements that do not exist. For example, as we saw in Fig. 5.11, thread 0,1 of block 0,0 calculates a valid P element P 0,1. However, it attempts to access a nonexisting M 0,3 during phase 1.

These two facts indicate that we will need to use different boundary condition tests for loading M tiles, loading N tiles, and calculating/storing P elements. A rule of thumb to follow is that every memory access needs to have a corresponding check that ensures that the indices used in the access are within the bounds of the array being accessed. ( 在每一次访问/读取 or 写入数组元素时进行边界检查 )

f the condition is false, the thread should not load the element. The question is what should be placed into the shared memory location. The answer is 0.0, a value that will not cause any harm if it is used in the inner product calculation.

The kernel code with the additional boundary condition checks is shown in Fig. 5.13.
![[PMPP-Fig5.13.png]]
With the boundary condition checks, the tile matrix multiplication kernel is just one step away from being a general matrix multiplication kernel. In general, matrix multiplication is defined for rectangular matrices: a j $\times$ k M matrix multiplied with a k $\times$ l N matrix results in a j $\times$ l P matrix. Our kernel can handle only square matrices so far.

Fortunately, it is quite easy to extend our kernel further into a general matrix multiplication kernel. We need to make a few simple changes. First, the Width argument is replaced by three unsigned integer arguments: j, k, l. Where Width is used to refer to the height of M or height of P, replace it with j. Where Width is used to refer to the width of M or height of N, replace it with k. Where Width is used to refer to the width of N or width of P, replace it with l. The revision of the kernel with these changes is left as an exercise.
## 5.6 Impact of memory usage on occupancy
In general, the more resources each thread requires, the fewer the number of threads that can reside in each SM

We saw in Chapter 4, Compute Architecture and Scheduling, how register usage can be a limiting factor for occupancy. Shared memory usage can also limit the number of threads that can be assigned to each SM. For example, the A 100 GPU can be configured to have up to 164 KB of shared memory per SM and supports a maximum of 2048 threads per SM. Thus for all 2048 thread slots to be used, a thread block should not use more than an average of (164 KB)/(2048 threads)=82 B/thread.

However, consider a kernel that has thread blocks that use 32 KB of shared memory, each of which has 256 threads. In this case, the kernel uses an average of (32 KB)/(256 threads)=132 B/thread of shared memory. With such shared memory usage, the kernel cannot achieve full occupancy. Each SM can host a maximum of only (164 KB)/(132 B/thread)=1272 threads. Therefore the maximum achievable occupancy of this kernel will be (1272 assigned threads)/(2048 maximum threads)=62%.

the field `devProp. sharedMemPerBlock` gives the amount of shared memory that is available in each SM.

We can enable such adjustment with a different style of declaration in CUDA by adding a C `extern` keyword in front of the shared memory declaration and omitting the size of the array in the declaration. Based on this style, the declarations for `Mds` and `Nds` need to be merged into one dynamically allocated array:
```c
extern __shared__ Mds_Nds[];
```
Note that the merged array is one-dimensional.

At runtime, when we call the kernel, we can dynamically configure the amount of shared memory to be used for each block according to the device query result and supply that as a third configuration parameter to the kernel call. The size is expressed in number of bytes.
![[PMPP-Fig5.14.png]]
## 5.7 Summary
the execution speed of a program in modern processors can be severely limited by the speed of the memory. To achieve good utilization of the execution throughput of a CUDA devices, one needs to strive for a high compute to global memory access ratio in the kernel code

We use matrix multiplication as an example to illustrate tiling, a popular strategy to enhance locality of data access and enable effective use of shared memory. In parallel programming, tiling uses barrier synchronization to force multiple threads to jointly focus on a subset of the input data at each phase of the execution so that the subset data can be placed into these special memory types to enable much higher access speed.

it is important for CUDA programmers to be aware of the limited sizes of these special types of memory. Their capacities are implementation dependent. Once their capacities have been exceeded, they limit the number of threads that can be executing simultaneously in each SM and can negatively affect the GPU’s computation throughput as well as its ability to tolerate latency.

Our goal for this chapter was to introduce the concept of locality, tiling, and different CUDA memory types. We introduced a tiled matrix multiplication kernel using shared memory. We further studied the need for boundary test conditions to allow for arbitrary data dimensions in applying tiling techniques. We also briefly discussed the use of dynamically sized shared memory allocation so that the kernel can adjust the size of shared memory that is used by each block according to the hardware capability. We did not discuss the use of registers in tiling. We will explain the use of registers in tiled algorithms when we discuss parallel algorithm patterns in Part II of the book.
# 6 Performance considerations
## 6.1 Memory coalescing
In Chapter 5, Memory Architecture and Data Locality, we studied tiling techniques that leverage the shared memory to reduce the total amount of data that must be accessed from the global memory by a collection of threads in each thread block.

Memory coalescing techniques are often used in conjunction with tiling techniques to allow CUDA devices to reach their performance potential by efficiently utilizing the global memory bandwidth.

Each time a DRAM location is accessed, a range of consecutive locations that include the requested location are accessed. Many sensors are provided in each DRAM chip, and they all work in parallel. Each senses the content of a bit within these consecutive locations. Once detected by the sensors, the data from all these consecutive locations can be transferred at high speed to the processor. These consecutive locations accessed and delivered are referred to as DRAM *bursts*. If an application makes focused use of data from these bursts, the DRAMs can supply the data at a much higher rate than would be the case if a truly random sequence of locations were accessed. ( 简而言之就是尽量访问连续的数据 )

the most favorable access pattern is achieved when all threads in a warp access consecutive global memory locations. In this case, the hardware combines, or *coalesces*, all these accesses into a consolidated access to consecutive DRAM locations. ( 线程束内的数据访问是连续的 )
For example, for a given load instruction of a warp, if thread 0 accesses global memory location X, thread 1 accesses location X + 1, thread 2 accesses location X + 2, and so on, all these accesses will be coalesced, or combined into a single request for consecutive locations when accessing the DRAM. Such coalesced access allows the DRAM to deliver data as a burst

One can tell by inspecting the code that the accesses to M can be coalesced. The index of the array M is `k*Width+col`. The variables `k` and `Width` have the same value across all threads in the warp. The variable `col` is defined as `blockIdx. x*blockDim.x+threadIdx.x`, which means that consecutive threads (with consecutive `threadIdx.x` values) will have consecutive values of `col` and will therefore access consecutive elements of `M`. ( 相邻的线程所访问的数据也是相邻的 )
![[PMPP-Fig6.2.png]]

Fig. 6.3 illustrates how consecutive threads iterate through consecutive columns when the matrix is stored in column-major order. One can tell by inspecting the code that the accesses to M are not favorable for coalescing. The index of the array M is `col*Width+k`. As before, col is defined as `blockIdx.x* blockDim.x+threadIdx.x`, which means that consecutive threads (with consecutive threadIdx. x values) will have consecutive values of `col`. However, in the index to M, col is multiplied by Width, which means that consecutive threads will access elements of M that are Width apart. Therefore the accesses are not favorable for coalescing. ( 相邻的线程访问的数据物理上不相邻 )
![[PMPP-Fig6.3.png]]

There are various strategies for optimizing code to achieve memory coalescing when the computation is not naturally amenable to it. One strategy is to rearrange how threads are mapped to the data; another strategy is to rearrange the layout of the data itself. ( 一种策略是重新改变线程与数据间的映射 )

another strategy is to transfer the data between global memory and shared memory in a coalesced manner and carry out the unfavorable access pattern in shared memory, which provides faster access latency. ( 另一种策略是仅考虑在全局存储和共享存储之间传递数据时保证合并 )

We will also see example optimizations that use this strategy throughout this book, including an optimization that we will apply now to matrix-matrix multiplication in which the second input matrix is in column-major layout. This optimization is called *corner turning*. ( 拐角处理/边界处理 )

![[PMPP-Fig6.4.png]]
In this example, A is an input matrix that is stored in row-major layout in global memory, and B is an input matrix that is stored in column-major layout in global memory. They are multiplied to produce an output matrix C that is stored in rowmajor layout in global memory.

The access to the input tile in matrix A is similar to that in Chapter 5, Memory Architecture and Data Locality. The four threads load the four elements at the top edge of the input tile. Each thread loads an input element whose local row and column indices within the input tile are the same as those of the thread’s output element within the output tile. These accesses are coalesced because consecutive threads access consecutive elements in the same row of A that are adjacent in memory according to the row-major layout. ( 相邻线程对于全局存储中的 A 的访问是相邻的 )

On the other hand, the access to the input tile in matrix B
Even though the four threads are logically loading the four consecutive elements at the top edge of the input tile, the elements that are loaded by consecutive threads are far away from each other in the memory because of the column-major layout of the B elements.

This problem can be solved by assigning the four consecutive threads to load the four consecutive elements at the left edge (the same column) in the input tile, as shown in Fig. 6.4 (B). ( 因为 B 是列主排列，因此让线程循列顺序取元素，使得它们对 B 的访问是相邻的 )
Intuitively, we are exchanging the roles of threadIdx. x and threadIdx. y when each thread calculates the linearized index for loading the B input tile.

The main advantage of memory coalescing is that it reduces global memory traffic by combining multiple memory accesses into a single access. ( 合并的目的就物理上在于减少全局存储访问 )


If multiple threads access data from the same DRAM location, they can potentially form a “carpool” and combine their accesses into one DRAM request. However, this requires the threads to have similar execution schedules so that their data accesses can be combined into one. Threads in the same warp are the perfect candidates because they all execute a load instruction simultaneously by virtue of SIMD execution.
## 6.2 Hiding memory latency
bursting alone is not sufficient to realize the level of DRAM access bandwidth required by modern processors. DRAM systems typically employ two more forms of parallel organization: banks and channels.

At the highest level, a processor contains one or more channels. Each channel is a memory controller with a bus that connects a set of DRAM banks to the processor.
![[PMPP-Fig6.7.png]] Fig. 6.7 illustrates a processor that contains four channels, each with a bus that connects four DRAM banks to the processor. In real systems a processor typically has one to eight channels, and a large number of banks is connected to each channel. ( 处理器的每个 channel 就是一个存储控制器，与一条总线相连，该总线和数个 DRAM 存储体相连 )

The data transfer bandwidth of a bus is defined by its width and clock frequency. Modern *double data rate* (DDR) busses perform two data transfers per clock cycle: one at the rising edge and one at the falling edge of each clock cycle. ( DDR 总线一个时钟周期执行两次数据传输 ) 

For example, a 64-bit DDR bus with a clock frequency of 1 GHz has a bandwidth of `8B*2*1 GHz=16GB/s`.
A modern CPU might require a memory bandwidth of at least 32 GB/s, whereas a modern GPU might require 256 GB/s. For this example the CPU would require 2 channels, and the GPU would require 16 channels. ( 注意一条 bus 对应一个 channel )


For each channel, the number of banks that is connected to it is determined by the number of banks required to fully utilize the data transfer bandwidth of the bus. Each bank contains an array of DRAM cells, the sensing amplifiers for accessing these cells, and the interface for delivering bursts of data to the bus
![[PMPP-Fig6.8.png]]
Fig. 6.8 (A) illustrates the data transfer timing when a single bank is connected to a channel. It shows the timing of two consecutive memory read accesses to the DRAM cells in the bank. Recall from Section 6.1 that each access involves long latency for the decoder to enable the cells and for the cells to share their stored charge with the sensing amplifier. This latency is shown as the gray section at the left end of the time frame. The time for transferring the burst data through the bus is shown as the left dark section of the time frame ( 多个 bank 相对单个 bank 可以提高 channel/bus 数据传输的利用率，因为 bus 对同一个 bank 的连续访问之间是存在访问延迟的 ) In reality, the access latency (the gray sections) is much longer than the data transfer time (the dark section)

For example, if the ratio of DRAM cell array access latency to the data transfer time is 20:1, the maximal utilization of the channel bus would be 1/21=4.8%; that is a 16 GB/s channel would deliver data to the processor at a rate no more than 0.76 GB/s. This problem is solved by connecting multiple banks to a channel bus.

When two banks are connected to a channel bus, an access can be initiated in the second bank while the first bank is serving another access. Therefore one can overlap the latency for accessing the DRAM cell arrays.
n Fig. 6.8. Shortly after the first bank starts accessing its cell array, the second bank also starts to access its cell array. Once bank 0 completes its data transfer, bank 1 can transfer its burst data (the second dark section). This pattern repeats for the next accesses.

From Fig. 6.8 (B), we can see that by having two banks, we can potentially double the utilization of the data transfer bandwidth of the channel bus. In general, if the ratio of the cell array access latency and data transfer time is R, we need to have at least R + 1 banks if we hope to fully utilize the data transfer bandwidth of the channel bus. ( 即每个 bank 灰色部分/访问延迟的长度/时间与黑色部分/数据传输时间的比值是 R: 1 时，我们需要再多 R 个 bank 来填充灰色部分，保证 bus/channel 总是忙碌的 )

n general, the number of banks connected to each channel bus needs to be larger than R for two reasons. 
One is that having more banks reduces the probability of multiple simultaneous accesses targeting the same bank, a phenomenon called *bank conflict*. ( 同一时间内对一个 bank 的多次访问会造成堵车/bank 冲突，此时这些访问只能排队了 ) Since each bank can serve only one access at a time, the cell array access latency can no longer be overlapped for these conflicting accesses. Having a larger number of banks increases the probability that these accesses will be spread out among multiple banks.

The second reason is that the size of each cell array is set to achieve reasonable latency and manufacturability. This limits the number of cells that each bank can provide. One may need many banks just to be able to support the memory size that is required. ( bank 多一点存储空间自然也更多 )


To achieve the memory access bandwidth specified for device, there must be a sufficient number of threads making
simultaneous memory accesses. This observation reflects another benefit of maximizing occupancy. Recall that in Chapter 4, Compute Architecture and Scheduling, we saw that maximizing occupancy ensures that there are enough threads resident on the streaming multiprocessors (SMs) to hide core pipeline latency, thereby utilizing the instruction throughput efficiently. As we see now, maximizing occupancy also has the additional benefit of ensuring that enough memory access requests are made to hide DRAM access latency, thereby utilizing the memory bandwidth efficiently. ( 我们已经知道保持 SM 高的占用率可以保证有足够多的线程在遇到长等待命令时用于调度保持处理器忙碌，提高指令吞吐量，同时我们也知道了线程数量多也利于保持总线忙碌，最大化利用存储带宽)

![[PMPP-Fig6.9.png]]
Fig. 6.9 shows a toy example of distributing the elements of an array M to channels and banks. We assume a small burst size of two elements (8 bytes). The distribution is done by hardware design. The addressing of the channels and backs are such that the first 8 bytes of the array (M[0] and M[1]) are stored in bank 0 of channel 0, the next 8 bytes (M[2] and M[3]) in bank 0 of channel 1, the next 8 bytes (M[4] and M[5]) in bank 0 of channel 2, and the next 8 bytes (M[6] and M[7]) in bank 0 of channel 3. ( 注意数组中连续的元素是存储在不同的 bank 中的 ) The distribution scheme illustrated in Fig. 6.9, often referred to as *interleaved data distribution* ( 交错式数据分布 ), spreads the elements across the banks and channels in the system. This scheme ensures that even relatively small arrays are spread out nicely. In our toy example, as long as we have at least 16 elements, the distribution will involve all the channels and banks for storing the elements


We now illustrate the interaction between parallel thread execution and the parallel memory organization. We will use the example in Fig. 5.5, replicated as Fig. 6.10.
![[PMPP-Fig6.10.png]]

We assume that the multiplication will be performed with 2 $\times$ 2 thread blocks and 2 $\times$ 2 tiles. During phase 0 of the kernel’s execution, all four thread blocks will be loading their first tile. The M elements that are involved in each tile are shown in Fig. 6.11.
![[PMPP-Fig6.11.png]]
Assume that all thread blocks are executed in parallel. We see that each block will make two coalesced accesses. According to the distribution in Fig. 6.9, these coalesced accesses will be made to the two banks in channel 0 as well as the two banks in channel 2. These four accesses will be done in parallel to take advantage of two channels as well as improving the utilization of the data transfer bandwidth of each channel.

We also see that Block 0,0 and Block 0,1 will load the same M elements. Most modern devices are equipped with caches that will combine these accesses into one as long as the execution timing of these blocks are sufficiently close to each other. In fact, the cache memories in GPU devices are mainly designed to combine such accesses and reduce the number of accesses to the DRAM system. ( 执行时间相近的 block 对同块数据的访问可以 cache )

Rows 4 and 5 show the M elements loaded during phase 1 of the kernel execution. We see that the accesses are now done to the banks in channel 1 and channel 3. Once again, these accesses will be done in parallel.

It should be clear to the reader that there is a symbiotic relationship between the parallel execution of the threads and the parallel structure of the DRAM system. ( 并行执行的线程是便于利用 DRAM 的并行结构，即 bank 和 channel 的 )

The reader is invited to verify that multiplying two larger matrices, such as 8 3 8 with the same 2 3 2 thread block configuration, will make use of all the four channels in Fig. 6.9. On the other hand, an increased DRAM burst size would require multiplication of even larger matrices to fully utilize the data transfer bandwidth of all the channels.
## 6.3 Thread coarsening
So far, in all the kernels that we have seen, work has been parallelized across threads at the finest granularity. That is, each thread was assigned the smallest possible unit of work.

The advantage of parallelizing work across threads at the finest granularity is that it enhances transparent scalability. If the hardware has enough resources to perform all the work in parallel, then the application has exposed enough parallelism to fully utilize the hardware. Otherwise, if the hardware does not have enough resources to perform all the work in parallel, the hardware can simply serialize the work by executing the thread blocks one after the other.

The disadvantage of parallelizing work at the finest granularity comes when there is a “price” to be paid for parallelizing that work. This price of parallelism can take many forms, such as redundant loading of data by different thread blocks, redundant work, synchronization overhead, and others. When the threads are executed in parallel by the hardware, this price of parallelism is often worth paying. However, if the hardware ends up serializing the work as a result of insufficient resources, then this price has been paid unnecessarily. In this case, it is better for the programmer to partially serialize the work and reduce the price that is paid for parallelism. This can be done by assigning each thread multiple units of work, which is often referred to as *thread coarsening*. ( 为每个线程分配多个单位的工作 )

Fig. 6.12 depicts the memory access pattern of computing two horizontally adjacent output tiles of the output matrix P. For each of these output tiles, we observe that different input tiles of the matrix N need to be loaded. However, the same input tiles of the matrix M are loaded for both the output tiles.
![[PMPP-Fig6.12.png]]
In the tiled implementation in Chapter 5, Memory Architecture and Data Locality, each output tile is processed by a different thread block. Because the shared memory contents cannot be shared across blocks, each block must load its own copy of the input tiles of matrix M. Although having different thread blocks load the same input tile is redundant, it is a price that we pay to be able to process the two output tiles in parallel using different blocks. If these thread blocks run in parallel, this price may be worth paying. On the other hand, if these thread blocks are serialized by the hardware, the price is paid in vain. In the latter case, it is better for the programmer to have a single thread block process the two output tiles, whereby each thread in the block processes two output elements. This way, the coarsened thread block would load the input tiles of M once and reuse them for multiple output tiles. ( 如果这两个 thread block 是串行执行，那不如仅使用一个 thread block，其中的线程执行两单位的工作，这样就减少了一次 M 的全局存储访问 )

Fig. 6.13 shows how thread coarsening can be applied to the tiled matrix multiplication code from Chapter 5,
![[PMPP-Fig6.13.png]]
On line 02 a constant `COARSE_FACTOR` is added to represent the *coarsening factor*, which is the number of original units of work for which each coarsened thread is going to be responsible. 

On line 13 the initialization of the column index is replaced with an initialization of `colStart`, which is the index of the first column for which the thread is responsible, since the thread is now responsible for multiple elements with different column indices.

In calculating colStart, the block index bx is multiplied by `TILE_WIDTH * COARSE_FACTOR` instead of just `TILE_WIDTH`, since each thread block is now responsible for `TILE_WIDTH * COARSE_FACTOR` columns. ( 注意现在每个 block 负责 `TILE_WIDTH * COARSE_FACTOR` 个列的元素 )

On lines 16-19, multiple instances of `Pvalue` are declared and initialized, one for each element for which the coarsened thread is responsible. 

The loop on line 17 that iterates over the different units of work for which the coarsened thread is responsible is sometimes referred to as a *coarsening loop*. ( 遍历线程负责的工作单元 )

Inside the loop on line 22 that loops over the input tiles, only one tile of M is loaded in each loop iteration, as with the original code. However, for each tile of M that is loaded, multiple tiles of N are loaded and used by the coarsening loop on line 27. ( 对 M 的一次访问可以配合对 N 的多次访问执行多个工作单元的计算 )

At the end, on lines 44-47, another coarsening loop is used for each coarsened thread to update the output elements for which it is responsible.

there are several pitfalls to avoid in applying thread coarsening. First, one must be careful not to apply the optimization when it is unnecessary. Recall that thread coarsening is beneficial when there is a price paid for parallelization that can be reduced with coarsening, such as redundant loading of data, redundant work, synchronization overhead, or others. Not all computations have such a price. For example, in the vector addition kernel in Chapter 2, Heterogeneous Data Parallel Computing, no price is paid for processing different vector elements in parallel. Therefore applying thread coarsening to the vector addition kernel would not be expected to make a substantial performance difference. The same applies to the RGB-to-grayscale conversion kernel in Chapter 3, Multidimensional Grids and Data ( 注意不是所有的内核都适合粗化，粗化的目的是通过减少并行化以减少相应的开销，而不是所有的内核并行化会存在额外开销 )

The second pitfall to avoid is not to apply so much coarsening that the hardware resources become underutilized. Recall that exposing as much parallelism as possible to the hardware enables transparent scalability. It provides the hardware with the flexibility of parallelizing or serializing work, depending on the amount of execution resources it has. When programmers coarsen threads, they reduce the amount of parallelism that is exposed to the hardware. If the coarsening factor is too high, not enough parallelism will be exposed to the hardware, resulting in some parallel execution resources being unutilized. ( 粗化会减少暴露给硬件的并行性 )

The third pitfall of applying thread coarsening is to avoid increasing resource consumption to such an extent that it hurts occupancy. Depending on the kernel, thread coarsening may require using more registers per thread or more shared memory per thread block. If this is the case, programmers must be careful not to use too many registers or too much shared memory such that the occupancy is reduced. The performance penalty from reducing occupancy may be more detrimental than the performance benefit that thread coarsening may offer. ( 粗化使得每个线程需要更多的资源，注意不要超过限制 )
## 6.4 A checklist of optimization
Throughout this first part of the book, we have covered various common optimizations that CUDA programmers apply to improve the performance of their code. We consolidate these optimizations into a single checklist, shown in Table 6.1.
![[PMPP-Table6.1.png]]
The first optimization in Table 6.1 is maximizing the occupancy of threads on SMs. the importance of having many more threads than cores was emphasized as a way to have enough work available to hide long-latency operations in the core pipeline.

The second optimization in Table 6.1 is using coalesced global memory accesses by ensuring that threads in the same warp access adjacent memory locations. it is the hardware’s ability to combine accesses to adjacent memory locations into a single memory request
There are multiple strategies that can be employed for achieving coalescing in applications with irregular access patterns. One strategy is to load data from global memory to shared memory in a coalesced manner and then perform the irregular accesses on shared memory

The third optimization in Table 6.1 is minimizing control divergence. the importance of threads in the same warp taking the same control path was emphasized as a means for ensuring that all cores are productively utilized during SIMD execution. So far, the kernels that we have looked at in this part of the book have not exhibited control divergence, except for the inevitable divergence at boundary conditions.

The fourth optimization in Table 6.1 is tiling data that is reused within a block by placing it in the shared memory or registers and accessing it repetitively from there, such that it needs to be transferred between global memory and the SM only once.
We will also observe that tiles of data can be stored in registers, not just shared memory. We will additionally observe that tiling is applicable to output data that is accessed repeatedly, not just input data.

The fifth optimization in Table 6.1 is privatization. This optimization has not yet been introduced, but we mention it here for completeness. Privatization relates to the situation in which multiple threads or blocks need to update a universal output. To avoid the overhead of updating the same data concurrently, a private copy of the data can be created and partially updated, and then a final update can be made to the universal copy from the private copy when done.

The sixth optimization in Table 6.1 is thread coarsening, in which multiple units of parallelism are assigned to a single thread to reduce the price of parallelism if the hardware was going to serialize the threads anyway.
Thread coarsening was introduced in this chapter in the context of tiled matrix multiplication, in which the price of parallelism was loading of the same input tile redundantly by multiple thread blocks that process adjacent output tiles. In this case, assigning one thread block to process multiple adjacent output tiles enables loading an input tile once for all the output tiles.
## 6.5 Knowing your computation's bottleneck
The resource that limits the performance of a computation is often referred to as a performance *bottleneck*. Optimizations typically use more of one resource to reduce the burden on another resource.

For example, shared memory tiling increases the use of shared memory to reduce the pressure on the global memory bandwidth. This optimization is great when the bottleneck resource is the global memory bandwidth and the data being loaded is reused. However, if, for example, the performance is limited by occupancy and occupancy is constrained by the use of too much shared memory already, then applying shared memory tiling is likely to make things worse.
## 6.6 Summary
In this chapter we covered the off-chip memory (DRAM) architecture of a GPU and discussed related performance considerations, such as global memory access coalescing and hiding memory latency with memory parallelism. We then presented an important optimization: thread granularity coarsening. With the insights that were presented in this chapter and earlier chapters, readers should be able to reason about the performance of any kernel code that they come across. We concluded this part of the book by presenting a checklist of common performance optimizations that are widely used to optimize many computations.
# 10 Reduction 
A reduction derives a single value from an array of values. The single value could be the sum, the maximum value, the minimal value, and so on among all elements.
## 10.1 Background
Mathematically, a reduction can be defined for a set of items based on a binary operator if the operator has a well-defined identity value. For example, a floatingpoint addition operator has an identity value of 0.0; that is, an addition of any floating-point value v and value 0.0 results in the value v itself. Thus a reduction can be defined for a set of floating-point numbers based on the addition operator that produces the sum of all the floating-point numbers in the set.

A reduction can be performed by sequentially going through every element of the array.

Reduction can be defined for many other operators. A product reduction can be defined for a floating-point multiplication operator whose identity value is 1.0. A product reduction of a set of floating-point numbers is the product of all these numbers. A minimum (min) reduction can be defined for a minimum comparison operator that returns the smaller value of the two inputs. For real numbers, the identity value for the minimum operator is $+\infty$. A maximum (max) reduction can be defined for a maximum (max) comparison operator that returns the larger value of the two input. For real numbers, the identity value for the maximum operator is $-\infty$.

Fig. 10.2 shows a general form of reduction for an operator, which is defined as a function that takes two inputs and returns a value.
![[PMPP-Figure10.2.png]]
The sequential algorithm ends when all the elements have been visited by the for-loop. For a set of N elements the for-loop iterates N iterations and produces the reduction result at the exit of the loop.
## 10.2 Reduction trees
The basic concept of parallel reduction is illustrated in Fig. 10.3
![[PMPP-Figure10.3.png]]
Note that the order of performing the operations will be changed from a sequential reduction algorithm to a parallel reduction algorithm  ( 并行化之后，操作的顺序会发生变化 )

As we have seen, parallel reduction assumes that the order of applying the operator to the input values does not matter. This property is guaranteed mathematically if the operator is associative. ( 如果算子可结合，则可以改变运算顺序，将串行归约优化为并行归约树 ) if an operator is associative, one can insert parentheses at arbitrary positions of an expression involving the operator and the results are all the same. With this equivalence relation, one can convert any order of operator application to any other order while preserving the equivalence of results.

We will apply an optimization in Section 10.4 that not only rearranges the order of applying the operator but also rearranges the order of the operands. To rearrange the order of the operands, this optimization further requires the operator to be commutative. ( 如果算子可交换，则可以考虑重新排列操作数顺序进一步优化 ) That is, the position of the operands can be rearranged in an expression and the results are the same. Note that the max operator is also commutative, as are many other operators such as min, sum, and product.

the for-loop in the sequential code iterates eight times, or takes eight time steps, to visit all of the input elements and produce the final result. On the other hand, with the parallel operations in Fig. 10.3 the parallel reduction tree approach takes only three time steps: four max operations during the first time step, two during the second, and one during the third. ( 归约树显著减少了时间步 )

There is, of course, a cost to the parallel approach: One must have enough hardware comparators to perform up to four max operations in the same time step. For N input values, a reduction tree performs 1/2 N operations during the first round, 1/4 N operations in the second round, and so on. Therefore the total number of operations that are performed is defined by the geometric series 1/2 N+1/4 N+1/8 N+. . . 1/N N = N - 1 operations, which is similar to the sequential algorithm.

In terms of time steps, a reduction tree takes $log_2N$ steps to complete the reduction process for N input values.During the first step, we need 1/2 N = 512 execution resources! Note that the number of resources that are needed diminishes quickly as we progress in time steps. During the final time step, we need to have only one execution resource.

The average parallelism is the total number of operations that are performed divided by the number of time steps, which is $(N - 1)/log_2N$. ( 平均并行度即平均每个时间步执行的操作数目 ) For N 5 1024 the average parallelism across the ten time steps is 102.3, whereas the peak parallelism is 512 (during the first time step). Such variation in the level of parallelism and resource consumption across time steps makes reduction trees a challenging parallel pattern for parallel computing systems. ( 归约树随着时间步的进行，其并行度和资源消耗在不断变化 )
## 10.3 A simple reduction kernel
Since the reduction tree requires collaboration across all threads, which is not possible across an entire grid, we will start by implementing a kernel that performs a sum reduction tree within a single block. That is, for an input array of N elements we will call this simple kernel and launch a grid with one block of 1/2N threads. During the first time step, all 1/2N threads will participate, and each thread adds two elements to produce 1/2N partial sums. During the next time step, half of the threads will drop off, and only 1/4N threads will continue to participate to produce 1/4N partial sums. This process will continue until the last time step, in which only one thread will remain and produce the total sum.

Fig. 10.6 shows the code of the simple sum kernel function
![[PMPP-Figure10.6.png]]
We assume that the `input` array is in the global memory and that a pointer to the array is passed as an argument when the kernel function is called.
Fig. 10.7 illustrates the execution of the reduction tree that is implemented by this code.
![[PMPP-Figure10.7.png]]
Each thread is assigned to a data location that is 2threadIdx.x (line 02). That is, the threads are assigned to the even locations in the input array: thread 0 to `input[0]`, thread 1 to `input[2]`, thread 2 to `input[4]`, and so on, as shown in the top row of Fig. 10.7. Each thread will be the “owner” of the location to which it is assigned and will be the only thread that writes into that location. The design of the kernel follows the “owner computes” approach, in which every data location is owned by a unique thread and can be updated only by that owner thread. ( 每个数据区域由一个线程独有，仅有它更新 )

In Fig. 10.6 a stride variable is used for the threads to reach for the appropriate partial sums for accumulation into their owner locations. The stride variable is initialized to 1 (line 03). The value of the stride variable is doubled in each iteration so the stride variable value will be 1, 2, 4, 8, etc., until it becomes greater than blockIdx.x, the total number of threads in the block.

The condition of the ifstatement is set up to select the active threads in each iteration. during iteration n, the threads whose thread index (threadIdx.x) values are multiples of $2^n$ are to perform addition. As the iterations progress, fewer and fewer threads remain active. At the last iteration, only thread 0 remains active and produces the sum reduction result.

The `__syncthreads()` statement (line 07 of Fig. 10.6) in the for-loop ensures that all partial sums that were calculated by the iteration have been written into their destination locations in the input array before any one of threads is allowed to begin the next iteration. ( 在开始下一个迭代之前，要保证本次迭代所有的部分和已经写入 ) The `__syncthreads()` statement ensures that all these partial sums from the first iteration have indeed been written to the even locations of the input array and are ready to be used by the active threads in the second iteration.
## 10.4 Minimizing control divergence
The kernel code in Fig. 10.6 implements the parallel reduction tree in Fig. 10.7 and produces the expected sum reduction result. Unfortunately, its management of active and inactive threads in each iteration results in a high degree of control divergence. ( 归约树会导致很高的线程间控制分歧 ) control divergence can significantly reduce the execution resource utilization efficiency, or the percentage of resources that are used in generating useful results. The waste of execution resources due to divergence increases over time.

If the size of the input array is greater than 32, entire warps will become inactive after the fifth iteration. For example, for an input size of 256, 128 threads or four warps would be launched. During the sixth iteration, warp 1 and warp 3 would become completely inactive and thus exhibit no control divergence. On the other hand, warp 0 and warp 2 would have only one active thread, exhibiting control divergence and wasting 31/32 of the execution resource. During the seventh iteration, only warp 0 would be active, exhibiting control divergence and wasting 31/32 of the execution resource.

In general, the execution resource utilization efficiency for an input array of size N can be calculated as the ratio between the total number of active threads to the total number of execution resources that are consumed. ( 执行资源利用效率即活跃线程的数量与消耗的总执行资源的比值 )

The total number of execution resources that are consumed is proportional to the total number of active warps across all iterations, since every active warp, no matter how few of its threads are active, consumes full execution resources. This number can be calculated as follows: ( 以warp为单位计算执行资源消耗，只要warp内有一个线程活跃，就认为它要消耗整个warp共32个线程的计算资源 )
$$
(N/64 * 5+N/64 * 1/2+N/64 * 1/4+\dots+1)*32
$$
Here, N/64 is the total number of warps that are launched, since N/2 threads will be launched and every 32 threads form a warp. The N/64 term is multiplied by 5 because all launched warps are active for five iterations. After the fifth iteration the number of warps is reduced by half in each successive iteration. The expression in parentheses gives the total number of active warps across all the iterations. The second term reflects that each active warp consumes full execution resources for all 32 threads regardless of the number of active threads in these warps. ( 注意我们计算的是所有时间步的总的执行资源消耗 )

The number of execution results committed by the active threads is the total number of active threads across all iterations: ( 由活跃线程提交的执行结果的数量就是所有时间步的总活跃线程数量 )
$$
N/64*(32 + 16 + 8 + 4 + 2 + 1) + N/64*1/2*1 + N/64*1/4*1+\dots+1
$$
The terms in the parenthesis give the active threads in the first five iterations for all N/64 warps. Starting at the sixth iteration, the number of active warps is reduced by half in each iteration, and there is only one active thread in each active warp. ( 在第六轮迭代之后，每个线程束内的活跃线程有且仅有1个 )

For an input array size of 256, the total number of committed results is `4*(32+16+8+4+2+1)+2+1 = 255`. This result should be intuitive because the total number of operations that are needed to reduce 256 values is 255.

Putting the previous two results together, we find that the execution resource utilization efficiency for an input array size of 256 is 255/736 = 0.35. This ratio states that the parallel execution resources did not achieve their full potential in speeding up this computation. On average, only about 35% of the resources consumed contributed to the sum reduction result. That is, we used only about 35% of the hardware’s potential to speed up the computation. ( 我们仅利用了我们所消耗的计算资源的35% ) 

Based on this analysis, we see that there is widespread control divergence across warps and over time. ( 其原因在于warp内的控制发散程度随着时间步增加而增加 ) there may be a better way to assign threads to the input array locations to reduce control divergence and improve resource utilization efficiency.

The problem with the assignment illustrated in Fig. 10.7 is that the partial sum locations become increasingly distant from each other, and thus the active threads that own these locations are also increasingly distant from each other as time progresses. This increasing distance between active threads contributes to the increasing level of control divergence. ( 之前方法的问题在于活跃线程之间的距离随着时间步的增加而指数增加 )

There is indeed a better assignment strategy that significantly reduces control divergence. The idea is that we should arrange the threads and their owned positions so that they can remain close to each other as time progresses. That is, we would like to have the stride value decrease, rather than increase, over time. ( 我们希望stride随着时间步增加而减小 ) The revised assignment strategy is shown in Fig. 10.8 for an input array of 16 elements.
![[PMPP-Figure10.8.png]]
Here, we assign the threads to the first half of the locations.
During each subsequent iteration, half of the active threads drop off, and all remaining active threads add an input element whose position is the number of active threads away from its owner position.

Note that if we compare the operation and operand orders of Fig. 10.8 to Fig. 10.7, there is effectively a reordering of the operands in the list rather than just inserting parentheses in different ways. For the result to always remain the same with such reordering, the operation must be commutative as well as being associative. ( 注意这种归约要求算子是可交换的 )

![[PMPP-Figure10.9.png]]
Fig. 10.9 shows a kernel with some subtle but critical changes to the simple kernel in Fig. 10.6. The owner position variable i is set to `threadIdx.x` rather than `2*threadIdx.x` (line 02). Thus the owner positions of all threads are now adjacent to each other, as illustrated in Fig. 10.8. The stride value is initialized as `blockDim.x` and is reduced by half until it reaches 1 (line 03). In each iteration, only the threads whose indices are smaller than the stride value remain active (line 04). Thus all active threads are of consecutive thread indices, as shown in Fig. 10.8. ( 每个迭代，都仅有 `stride` 个线程保持活跃，它们的索引是连续的 ) 

The number of threads that execute an addition operation (line 06) in each iteration is the same as in Fig. 10.6. ( 该算法每个迭代执行归约的活跃线程的数量相较于之前的算法是没有变化的 )Then why should there be a difference in control divergence between the two kernels? The answer lies in the positions of threads that perform the addition operation relative to those that do not.

Let's consider the example of an input array of 256 elements. During the first iteration, all threads are active, so there is no control divergence. During the second iteration, threads 0 through 63 execute the add statement (active), while threads 64 through 127 do not (inactive). The pairwise sums are stored in elements 0 through 63 during the second iteration. Since the warps consist of 32 threads with consecutive threadIdx.x values, all threads in warp 0 through warp 1 execute the add statement, whereas all threads in warp 2 through warp 3 become inactive. Since all threads in each warp take the same path of execution, there is no control divergence! ( 我们让相邻的线程执行保持相同的状态，减少了控制分歧 )

However, the kernel in Fig. 10.9 does not completely eliminate the divergence caused by the if-statement. The reader should verify that for the 256-element example, starting with the fourth iteration, the number of threads that execute the addition operation will fall below 32. That is, the final five iterations will have only 16, 8, 4, 2, and 1 thread(s) performing the addition. This means that the kernel execution will still have divergence in these iterations. ( 但是在最后5个迭代中，warp0中仍然会有控制分歧 ) However, the number of iterations of the loop that has divergence is reduced from ten to five. We can calculate the total number of execution resources consumed as follows:
$$
(N/64*1 + N/64*1/2+\dots+1+5*1)*32
$$
The part in parentheses reflects the fact that in each subsequent iteration, half of the warps become entirely inactive and no longer consume execution resources. ( 可以看到，每经过一次迭代，活跃的warp数量就减半，即一半的warp不再消耗计算资源 ) This series continues until there is only one full warp of active threads. The last term `(5*1)` reflects the fact that for the final five iterations, there is only one active warp, and all its 32 threads consume execution resources even though only a fraction of the threads are active ( 但最后的5轮迭代仍要消耗一整个warp的计算资源 ) Thus the sum in the parentheses gives the total number of warp executions through all iterations, which, when multiplied by 32, gives the total amount of execution resources that are consumed. ( 括号内就是所有迭代下所活跃的warp总数量 )

For our 256-element example the execution resources that are consumed are (4+2+1+51)32 = 384, which is almost half of 736, the resources that were consumed by the kernel in Fig. 10.6. Since the number of active threads in each iteration did not change from Fig. 10.7 to Fig. 10.8, the efficiency of the new kernel in Fig. 10.9 is 255/384 5 66%, which is almost double the efficiency of the kernel in Fig. 10.6. Note also that since the warps are scheduled to take turns executing in a streaming multiprocessor of limited execution resources, the total execution time will also improve with the reduced resource consumption.
## 10.5 Minimizing memory divergence
The simple kernel in Fig. 10.6 has another performance issue: memory divergence. it is important to achieve memory coalescing within each warp. That is, adjacent threads in a warp should access adjacent locations when they access global memory. ( 一个warp内的线程都是相邻的，如果它们可以访问global memory的相邻位置，就可以进行memory合并 )

Unfortunately, in Fig. 10.7, adjacent threads do not access adjacent locations. In each iteration, each thread performs two global memory reads and one global memory write. The first read is from its owned location, the second read is from the location that is of stride distance away from its owned location, and the write is to its owned location. ( 在每次迭代中，每个线程都会执行两次global memory读和一次global memory写，两次global memory读的位置相差 `stride` ) Since the locations owned by adjacent thread are not adjacent locations, the accesses that are made by adjacent threads will not be fully coalesced. During each iteration the memory locations that are collectively accessed by a warp are of stride distance away from each other.( 先前的算法中，warp内相邻的线程并没有保持相邻物理位置的访问，相邻的线程要访问的物理位置都相隔 `stride` )

For example, as shown in Fig. 10.7, when all threads in a warp perform their first read during the first iteration, the locations are two elements away from each other. As a result, two global memory requests are triggered, ( 每次global memory请求必须是连续的一块，要cover到整个warp范围，需要两块，因此每个warp每次读/写需要两次请求 ) and half the data returned will not be used by the threads. The same behavior occurs for the second read and the write. During the second iteration, every other thread drops out, and the locations that are collectively accessed by the warp are four elements away from each other. Two global memory requests are again performed, and only onefourth of the data returned will be used by the threads. This will continue until there is only one active thread for each warp that remains active. ( 前5个迭代过后，仅剩1个线程在各个warp中时，不需要cover到整个warp范围，每个warp每次读/写仅需一次请求 ) Only when there is one active thread in the warp will the warp perform one global memory request. Thus the total number of global memory requests is as follows:
$$
(N/64*5*2 + N/64*1 + N/64 *1/2 + N/64 * 1/4 + \dots + 1)*3
$$

The first term `(N/64*2)` corresponds to the first five iterations, in which all N/64 warps have two or more active threads, so each warp performs two global memory requests. The remaining terms account for the final iterations, in which each warp has only one active thread and performs one global memory request and half of the warps drop out in each subsequent iteration. The multiplication by 3 accounts for the two reads and one write by each active thread during each iteration.

For the kernel in Fig. 10.9 the adjacent threads in each warp always access adjacent locations in the global memory, so the accesses are always coalesced. As a result, each warp triggers only one global memory request on any read or write. ( 每个warp每次读/写仅需要一次请求，不需要cover到整个warp范围 ) As the iterations progress, entire warps drop out, so no global memory access will be performed by any thread in these inactive warps. Half of the warps drop out in each iteration until there is only one warp for the final five iterations. ( 最后留下一个warp，用5轮迭代完成归约，每轮迭代每次读/写同样需要一次请求 )Therefore the total number of global memory requests performed by the kernel is as follows:
$$
((N/64 + N/64*1/2 + N/64 * 1/4 + \dots + 1)+5)*3
$$

For the 256-element example the total number of global memory requests performed is ((4+2+1)+5) $\times$ 3 = 36. The improved kernel results in 141/36 = 3.9 $\times$

In conclusion, the convergent kernel offers more efficiency in using both execution resources and DRAM bandwidth. The advantage comes from both reduced control divergence and improved memory coalescing.
## 10.6 Minimizing global memory accesses
The convergent kernel in Fig. 10.9 can be further improved by using shared memory. Note that in each iteration, threads write their partial sum result values out to the global memory, and these values are reread by the same threads and other threads in the next iteration. ( 每次迭代，各线程计算的部分和可以不写入global memory，而是写入shared memory，因为下一个迭代还需要使用 ) we can further improve the execution speed by keeping the partial sum results in the shared memory. This idea is illustrated in Fig. 10.10
![[PMPP-Fig10.10.png]]
![[PMPP-Fig10.11.png]]

Since the first iteration is already done when accessing the global memory locations outside the loop, the for-loop starts with blockDim.x/2 (line 04) instead of blockDim.x.

The `__syncthreads()` is moved to the beginning of the loop to ensure that we synchronize between the shared memory accesses and the first iteration of the loop.

Using the kernel in Fig. 10.11, the number of global memory accesses are reduced to the initial loading of the original contents of the input array and the final write to input[0]. Thus for an N-element reduction the number of global memory accesses is just N+1. Note also that both global memory reads in Fig. 10.11 (line 04) are coalesced. So with coalescing, there will be only (N/32) +1 global memory requests. ( 每个元素一次，由于合并，一次其实可以取到32个连续的元素 )

Another benefit of using shared memory, besides reducing the number of global memory accesses, is that the input array is not modified. This property is useful if the original values of the array are needed for some other computation in another part of the program. ( 使用shared memory还可以保持kernel计算过程中对global memory中的原数据没有inplace 修改 )
## 10.7 Hierarchical reduction for arbitrary input length
For large input arrays that contain millions or even billions of elements, we can benefit from launching more threads to further accelerate the reduction process. Since we do not have a good way to perform barrier synchronization among threads in different blocks, we will need to allow threads in different blocks to execute independently. ( 难以同步不同block，就需要保持block执行相互独立 )
![[PMPP-Fig10.13.png]]
Fig. 10.12 illustrates the concept of hierarchical, segmented multiblock reduction using atomic operations, and Fig. 10.13 shows the corresponding kernel implementation. The idea is to partition the input array into segments so that each segment is of appropriate size for a block. All blocks then independently execute a reduction tree ( 每个block各自执行一个归约树 ) and accumulate their results to the final output using an atomic add operation.

That is, each block processes 2blockDim.x elements. Thus when we multiply the size of each segment by blockIdx.x of a block, we have the starting location of the segment to be processed by the block.
Once we know the starting location for each block, all threads in a block can simply work on the assigned segment as if it is the entire input data.

Once the reduction tree for-loop is complete, the partial sum for the segment is in input_s[0]. The if-statement in line 16 of Fig. 10.13 selects thread 0 to contribute the value in input_s[0] to output, as illustrated in the bottom part of Fig. 10.12. This is done with an atomic add, as shown in line 14 of Fig. 10.13. Once all blocks of the grid have completed execution, the kernel will return, ( kernel在grid内所有block完成执行之后返回 ) and the total sum is in the memory location pointed to by output.
## 10.8 Thread coarsening for reduced overhead
The reduction kernels that we have worked with so far all try to maximize parallelism by using as many threads as possible. That is, for a reduction of N elements, N/2 threads are launched. With a thread block size of 1024 threads, the resulting number of thread blocks is N/2048. However, in processors with limited execution resources the hardware may have only enough resources to execute a portion of the thread blocks in parallel. In this case, the hardware will serialize the surplus thread blocks, executing a new thread block whenever an old one has completed. ( 设备无法并行执行所有的block，就会串行执行分配后盈余的block )

To parallelize reduction, we have actually paid a heavy price to distribute the work across multiple thread blocks. As we saw in earlier sections, hardware underutilization increases with each successive stage of the reduction tree because of more warps becoming idle and the final warp experiencing more control divergence. The phase in which the hardware is underutilized occurs for every thread block that we launch.

It is an inevitable price to pay if the thread blocks are to actually run in parallel. However, if the hardware is to serialize these thread blocks, we are better off serializing them ourselves in a more efficient manner. ( 以相较于硬件串行化更好的方式串行化工作 ) As we discussed in Chapter 6, Performance Considerations, thread granularity coarsening, or thread coarsening for brevity, is a category of optimizations that serialize some of the work into fewer threads to reduce parallelization overhead. We start by showing an implementation of parallel reduction with thread coarsening applied by assigning more elements to each thread block. ( 为每个线程块分配更多的元素 )

Fig. 10.14 illustrates how thread coarsening can be applied to the example in Fig. 10.10. In Fig. 10.10, each thread block received 16 elements, which is two elements per thread. Each thread independently adds the two elements for which it is responsible; then the threads collaborate to execute a reduction tree. In Fig. 10.14 we coarsen the thread block by a factor of 2. Hence each thread block receives twice the number of elements, that is, 32 elements, which is four elements per thread. ( 每个线程在第一轮迭代负责4个元素的归约 )
![[PMPP-Fig10.14.png]]
The three steps to add the four elements are illustrated by the first three rows of arrows in Fig. 10.14

Note that all threads are active during these three steps. Moreover, since the threads independently add the four elements for which they are responsible, they do not need to synchronize, and they do not need to store their partial sums to shared memory until after all four elements have been added. The remaining steps in performing the reduction tree are the same as those in Fig. 10.10. ( 第二轮迭代开始，元素数=线程数，为了尽可能利用多的线程提高块内并行性，我们仍然让每个线程仅负责两个元素 )

Fig. 10.15 shows the kernel code for implementing reduction with thread coarsening for the multiblock segmented kernel. Compared to Fig. 10.13, the kernel has two main differences. The first difference is that when the beginning of the block’s segment is identified, we multiply by COARSE_FACTOR to reflect the fact that the size of the block’s segment is COARSE_FACTOR times larger (line 03). The second difference is that when adding the elements for which the thread is responsible, rather than just adding two elements (line 06 in Fig. 10.13), we use a coarsening loop to iterate over the elements and add them based on COARSE_FACTOR (lines 06-09 in Fig. 10.15). Note that all threads are active throughout this coarsening loop, the partial sum is accumulated to the local variable sum, and no calls to `__syncthreads()` are made in the loop because the threads act independently.
![[PMPP-Fig10.16.png]]

Fig. 10.16 compares the execution of two original thread blocks without coarsening serialized by the hardware, shown in Fig. 10.16A with one coarsened thread block performing the work of two thread blocks, shown in Fig. 10.16B. 

Fig. 10.16A the first thread block performs one step in which each thread adds the two elements for which it is responsible. All threads are active during this step, so the hardware is fully utilized. The remaining three steps execute the reduction tree in which half the threads drop out each step, underutilizing the hardware. Moreover, each step requires a barrier synchronization as well as accesses to shared memory. When the first thread block is done, the hardware then schedules the second thread block, which follows the same steps but on a different segment of the data. Overall, the two blocks collectively take a total of eight steps, of which two steps fully utilize the hardware and six steps underutilize the hardware and require barrier synchronization and shared memory access. ( 完成相同的工作需要未corsen的两个线程块采取一共8个迭代，它们只有在各自的第一次迭代有完全的硬件利用率，因此8个迭代中有两次迭代是完全利用了硬件的，同时剩余的六个迭代都需要block内的线程同步 ) By contrast, in Fig. 10.16B the same amount of data is processed by only a single thread block that is coarsened by a factor of 2. This thread block initially takes three steps in which each thread adds the four elements for which it is responsible. All threads are active during all three steps, so the hardware is fully utilized, and no barrier synchronizations or accesses to shared memory are performed. The remaining three steps execute the reduction tree in which half the threads drop out each step, underutilizing the hardware, and barrier synchronization and accesses to shared memory are needed. Overall, only six steps are needed (instead of eight), of which three steps (instead of two) fully utilize the hardware and three steps (instead of six) underutilize the hardware and require barrier synchronization and shared memory access. ( corsen之后，一个线程块需要6个迭代完成归约，其中3个迭代完整利用了硬件资源，另外3个迭代需要block内线程铜鼓 ) Therefore thread coarsening effectively reduces the overhead from hardware underutilization, synchronization, and access to shared memory. ( 因此，如果方案A中两个线程块是串行执行的，显然方案B要比方案A更高效 )

Theoretically, we can increase the coarsening factor well beyond two. However, one must keep in mind that as we coarsen threads, less work will be done in parallel. Therefore increasing the coarsening factor will reduce the amount of data parallelism that is being exploited by the hardware. If we increase the coarsening factor too much, such that we launch fewer thread blocks than the hardware is capable of executing, we will no longer be able to take full advantage of the parallel hardware execution resources. The best coarsening factor ensures that there are enough thread blocks to fully utilize the hardware, which usually depends on the total size of the input as well as the characteristics of the specific device.
## 10.9 Summary
The parallel reduction pattern is important, as it plays a key row in many dataprocessing applications. Although the sequential code is simple, it should be clear to the reader that several techniques, such as thread index assignment for reduced divergence, using shared memory for reduced global memory accesses, segmented reduction with atomic operations, and thread coarsening, are needed to achieve high performance for large inputs. The reduction computation is also an important foundation for the prefix-sum pattern that is an important algorithm component for parallelizing many applications and will be the topic of Chapter 11, Prefix Sum (Scan).