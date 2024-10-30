# 2 Programming Model
## 5.1 Kernels
CUDA C++ extends C++ by allowing the programmer to define C++ functions, called kernels, that, when called, are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C++ functions.

A kernel is defined using the `__global__` declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new <<<...>>>execution configuration syntax
## 5.2 Thread Hierarchy
threadIdx is a 3-component vector, so that threads can be identified using a onedimensional, two-dimensional, or three-dimensional thread index, forming a one-dimensional, twodimensional, or three-dimensional block of threads, called a thread block.

for a two-dimensional block of size (Dx, Dy), the thread ID of a thread of index (x, y) is (x + y Dx); for a three-dimensional block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy).

On current GPUs, a thread block may contain up to 1024 threads.

a kernel can be executed by multiple equally-shaped thread blocks, so that the total number of threads is equal to the number of threads per block times the number of blocks.

The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.

The number of threads per block and the number of blocks per grid specified in the <<<...>>> syntax can be of type int or dim3

Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. This independence requirement allows thread blocks to be scheduled in any order across any number of cores as illustrated by Figure 3, enabling programmers to write code that scales with the number of cores.

one can specify synchronization points in the kernel by calling the `__syncthreads()` intrinsic function; `__syncthreads()` acts as a barrier at which all threads in the block must wait before any is allowed to proceed.
### 5.2.1 Thread Block Clusters
With the introduction of NVIDIA Compute Capability 9.0, the CUDA programming model introduces an optional level of hierarchy called Thread Block Clusters that are made up of thread blocks. ( 线程块簇 ) Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU. ( 同一个线程块内的线程保证被一个SM同时调度，同一个线程块簇内的线程块保证被一个GPU处理簇同时调度 ) 注意这要求Compute Capability 9.0

Similar to thread blocks, clusters are also organized into a one-dimension, two-dimension, or threedimension as illustrated by Figure 5. ( 线程块簇也可以按照一/二/三维组织 )
![[CUDA C++ Programming Guide-Fig5.png]]
在CUDA中，一个簇中的线程块数量可以由用户定义，CUDA支持的最大的可移植的簇大小是一个簇中8个线程块，如果GPU硬件或MIG配置较弱而无法支持8个多处理器，则相应的最大簇大小将相应减小
可以使用`cudaOccupancyMaxPotentialClusterSize API`查询特定架构所支持的最大线程块簇的大小

> Note: In a kernel launched using cluster support, the gridDim variable still denotes the size in terms of number of thread blocks, for compatibility purposes. The rank of a block in a cluster can be found using the Cluster Group API

A thread block cluster can be enabled in a kernel either using a compiler time kernel attribute using` __cluster_dims__(X,Y,Z)` or using the CUDA kernel launch API `cudaLaunchKernelEx`. ( 使用属性标识内核或使用API发起内核以启用线程块簇 )


The cluster size using kernel attribute is fixed at compile time and then the kernel can be launched using the classical <<< , >>>. If a kernel uses compile-time cluster size, the cluster size cannot be modified when launching the kernel. ( 使用属性标识内核时，要注明簇的大小，该大小不可改变，被标识的内核可以用<<<,>>>正常发起 )
```
∕∕ Kernel definition
∕∕ Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{ 

}

int main()
{
float *input, *output;
∕∕ Kernel invocation with compile time cluster size

dim3 threadsPerBlock(16, 16);
dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);

∕∕ The grid dimension is not affected by cluster launch, and is still enumerated using number of blocks.
∕∕ The grid dimension must be a multiple of cluster size.

cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output); 
}
```

A thread block cluster size can also be set at runtime and the kernel can be launched using the CUDA kernel launch API `cudaLaunchKernelEx`. ( 想要在运行时设定线程块簇的大小，就使用API发起内核 )
```
∕∕ Kernel definition

∕∕ No compile time attribute attached to the kernel

__global__ void cluster_kernel(float *input, float* output)
{ 

}

int main() { 
    float *input, *output; 
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
    ∕∕ Kernel invocation with runtime cluster size
    { 
        cudaLaunchConfig_t config = {0};
        ∕∕ The grid dimension is not affected by cluster launch, and is still enumerated using number of blocks.
        ∕∕ The grid dimension should be a multiple of cluster size.

        config.gridDim = numBlocks; 
        config.blockDim = threadsPerBlock;
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; 
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1; 
        config.attrs = attribute; 
        config.numAttrs = 1; 
        cudaLaunchKernelEx(&config, cluster_kernel, input, output); 
    } 
}
```

In GPUs with compute capability 9.0, all the thread blocks in the cluster are guaranteed to be coscheduled on a single GPU Processing Cluster (GPC) and allow thread blocks in the cluster to perform hardware-supported synchronization using the Cluster Group API `cluster.sync()`. ( 同一个线程块簇内的线程块一定会被一个GPU处理簇同时调度，块簇内的线程块可以通过API同步 )

Cluster group also provides member functions to query cluster group size in terms of number of threads or number of blocks using `num_threads()` and `num_blocks()` API respectively.
The rank of a thread or block in the cluster group can be queried using `dim_threads()` and `dim_blocks()` API respectively.

Thread blocks that belong to a cluster have access to the Distributed Shared Memory. Thread blocks in a cluster have the ability to read, write, and perform atomics to any address in the distributed shared memory.
## 5.3 Memory Hierarchy
Each thread has private local memory. Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. Thread blocks in a thread block cluster can perform read, write, and atomics operations on each other’s shared memory. ( 同一簇内的线程块可以对簇内其他块的共享存储读写 ) All threads have access to the same global memory
![[CUDA C++ Programming Guide-Fig6.png]]

There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces. Texture memory also offers different addressing modes, as well as data filtering, for some specific data formats

The global, constant, and texture memory spaces are persistent across kernel launches by the same application. ( global, constan, texture memory的生命周期和应用程序相同 )
## 5.4 Heterogeneous Programming
As illustrated by Figure 7, the CUDA programming model assumes that the CUDA threads execute on a physically separate device that operates as a coprocessor to the host running the C++ program.
## 5.5 Asynchronous SIMT Programming Model
In the CUDA programming model a thread is the lowest level of abstraction for doing a computation or a memory operation. Starting with devices based on the NVIDIA Ampere GPU architecture, the CUDA programming model provides acceleration to memory operations via the asynchronous programming model. The asynchronous programming model defines the behavior of asynchronous operations with respect to CUDA threads. ( 可以通过异步编程模型进行对存储操作的加速 )

The asynchronous programming model defines the behavior of Asynchronous Barrier for synchronization between CUDA threads. The model also explains and defines how `cuda::memcpy_async` can be used to move data asynchronously from global memory while computing in the GPU.  
### 5.5.1 Asynchronous Operations
An asynchronous operation is defined as an operation that is initiated by a CUDA thread and is executed asynchronously as-if by another thread.

In a well formed program one or more CUDA threads synchronize with the asynchronous operation. The CUDA thread that initiated the asynchronous operation is not required to be among the synchronizing threads. ( 发起异步操作的CUDA线程不需要参与到对该异步操作的同步 )

这样一个异步线程(即“as-if”线程)总是与启动异步操作的CUDA线程相关联，异步操作使用同步对象(synchronization object)来同步操作的完成，这样的同步对象可以由用户显式管理(例如，`cuda::memcpy_async`)，也可以在库中隐式管理(例如，`cooperative_groups::memcpy_async`)

A synchronization object could be a `cuda::barrier` or a `cuda::pipeline`.
These synchronization objects can be used at different thread scopes. A scope defines the set of threads that may use the synchronization object to synchronize with the asynchronous operation. ( scope定义了需要和异步操作同步的线程范围 ) The following table defines the thread scopes available in CUDA C++ and the threads that can be synchronized with each.
![[CUDA C++ Programming Guide-ThreadScopes.png]]
( scope由小到大分别为线程、块、GPU设备、系统)
## 5.5 Copmute Capability
The compute capability of a device is represented by a version number, also sometimes called its “SM version”. This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU. ( 计算能力决定了硬件支持的特性，便于我们相对应地编程 )

The compute capability comprises a major revision number X and a minor revision number Y and is denoted by X.Y

Devices with the same major revision number are of the same core architecture. The major revision number is 9 for devices based on the NVIDIA Hopper GPU architecture, 8 for devices based on the NVIDIA Ampere GPU architecture, 7 for devices based on the Volta architecture, 6 for devices based on the Pascal architecture, 5 for devices based on the Maxwell architecture, and 3 for devices based on the Kepler architecture ( 主版本相同，核心架构相同 )

The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new features.

Turing is the architecture for devices of compute capability 7.5, and is an incremental update based on the Volta architecture.
# 3 Programming Interface
The runtime is introduced in CUDA Runtime. It provides C and C++ functions that execute on the host to allocate and deallocate device memory, transfer data between host memory and device memory, manage systems with multiple devices, etc.

The runtime is built on top of a lower-level C API, the CUDA driver API( CUDA驱动API), which is also accessible by the application. The driver API provides an additional level of control by exposing lower-level concepts such as CUDA contexts - the analogue of host processes for the device - and CUDA modules - the analogue of dynamically loaded libraries for the device. (CUDA context to 设备好比 process to 主机，CUDA module to 设备 好比 动态链接库 to 主机) when using the runtime, context and module management are implicit, resulting in more concise code.

As the runtime is interoperable with the driver API, most applications that need some driver API features can default to use the runtime API and only use the driver API where needed. The driver API is introduced in Driver API and fully described in the reference manual. ( 可以通过runtime API使用驱动API )
## 6.1 Compilation with NVCC
Kernels can be written using the CUDA instruction set architecture, called PTX ( PTX是用于描述内核的CUDA指令集架构 ) It is however usually more effective to use a high-level programming language such as C++. In both cases, kernels must be compiled into binary code by `nvcc` to execute on the device.

nvcc is a compiler driver that simplifies the process of compiling C++ or PTX code: It provides simple and familiar command line options and executes them by invoking the collection of tools that implement the different compilation stages. This section gives an overview of nvcc workflow and command options.
### 6.1.1 Compilation Workflow
#### 6.1.1.1 Offline Compilation
Source files compiled with nvcc can include a mix of host code (i.e., code that executes on the host) and device code (i.e., code that executes on the device). nvcc’s basic workflow consists in separating device code from host code and then:
-  compiling the device code into an assembly form (*PTX* code) and/or binary form (*cubin* object),
-  and modifying the host code by replacing the <<<...>>> syntax by the necessary CUDA runtime function calls to load and launch each compiled kernel from the PTX code and/or cubin object.
( nvcc 先分离设备代码和主机代码，然后将设备代码编译为PTX汇编形式或cubin文件，然后修改主机代码，即将<<<...>>>语法替换为CUDA运行时函数调用，这些函数调用用于装载和发起先前编译好的内核 )

The modified host code is output either as C++ code that is left to be compiled using another tool or as object code directly by letting nvcc invoke the host compiler during the last compilation stage. ( 得到的修改后的主机代码可以是C++源码的形式，也可以是目标码的形式 )

Applications can then:
-  Either link to the compiled host code (this is the most common case),
- Or ignore the modified host code (if any) and use the CUDA driver API (see Driver API) to load and execute the PTX code or cubin object.
#### 6.1.1.2 Just-in-Time Compilation
Any PTX code loaded by an application at runtime is compiled further to binary code by the device driver. This is called just-in-time compilation. ( nvcc仅将设备码编译到PTX码就停止，然后在应用运行并将其加载时再由设备驱动器将其进一步编译为二进制码，就是即时编译 ) Just-in-time compilation increases application load time, but allows the application to benefit from any new compiler improvements coming with each new device driver. ( 即时编译可以方便享受新设备驱动带来的编译优化 ) It is also the only way for applications to run on devices that did not exist at the time the application was compiled, as detailed in Application Compatibility.

When the device driver just-in-time compiles some PTX code for some application, it automatically caches a copy of the generated binary code in order to avoid repeating the compilation in subsequent invocations of the application. The cache - referred to as *compute cache* - is automatically invalidated when the device driver is upgraded, so that applications can benefit from the improvements in the new just-in-time compiler built into the device driver.

As an alternative to using nvcc to compile CUDA C++ device code, NVRTC can be used to compile CUDA C++ device code to PTX at runtime. NVRTC is a runtime compilation library for CUDA C++;
### 6.1.2 Binary Compatibility
Binary code is architecture-specific. A *cubin* object is generated using the compiler option `-code` that specifies the targeted architecture: For example, compiling with `-code=sm_80` produces binary code for devices of compute capability 8.0. (cubin文件是特定于架构的) Binary compatibility is guaranteed from one minor revision to the next one, but not from one minor revision to the previous one or across major revisions. (针对新架构的cubin无法在旧设备上执行，即cubin只能向上兼容小版本)
### 6.1.3 PTX Compatibility
Some PTX instructions are only supported on devices of higher compute capabilities. For example, Warp Shuffle Functions are only supported on devices of compute capability 5.0 and above.

The `-arch` compiler option specifies the compute capability that is assumed when compiling C++ to PTX code. So, code that contains warp shuffle, for example, must be compiled with `-arch=compute_50` (or higher).

PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability. Note that a binary compiled from an earlier PTX version may not make use of some hardware features. As a result, the final binary may perform worse than would be possible if the binary were generated using the latest version of PTX. (PTX可以向上兼容大版本，即针对旧架构PTX可以编译出能在新架构上运行的cubin)

PTX code compiled to target architecture conditional features only run on the exact same physical architecture and nowhere else. Arch conditional PTX code is not forward and backward compatible. Example code compiled with sm_90a or compute_90a only runs on devices with compute capability 9.0 and is not backward or forward compatible.
### 6.1.4 Application Compatibility
Which PTX and binary code gets embedded in a CUDA C++ application is controlled by the `-arch` and `-code` compiler options or the `-gencode` compiler option as detailed in the nvcc user manual. For example,
```
nvcc x.cu
         -gencode arch=compute_50,code=sm_50
         -gencode arch=compute_60,code=sm_60
         -gencode arch=compute_70,code=\"compute_70,sm_70\"
```
embeds binary code compatible with compute capability 5.0 and 6.0 (first and second `-gencode` options) and PTX and binary code compatible with compute capability 7.0 (third `-gencode` option).

Host code is generated to automatically select at runtime the most appropriate code to load and execute, which, in the above example, will be:
- 5.0 binary code for devices with compute capability 5.0 and 5.2,
- 6.0 binary code for devices with compute capability 6.0 and 6.1,
- 7.0 binary code for devices with compute capability 7.0 and 7.5,
- PTX code which is compiled to binary code at runtime for devices with compute capability 8.0 and 8.6.

x.cu can have an optimized code path that uses warp reduction operations, for example, which are only supported in devices of compute capability 8.0 and higher. The `__CUDA_ARCH__` macro can be used to differentiate various code paths based on compute capability. It is only defined for device code. When compiling with `-arch=compute_80 `for example, `__CUDA_ARCH__` is equal to 800.

If x.cu is compiled for architecture conditional features example with `sm_90a` or `compute_90a`, the code can only run on devices with compute capability 9.0

Applications using the driver API must compile code to separate files and explicitly load and execute the most appropriate file at runtime.

The nvcc user manual lists various shorthands for the -arch, -code, and -gencode compiler options. For example, `-arch=sm_70` is a shorthand for `-arch=compute_70 -code=compute_70, sm_70` (which is the same as `-gencode arch=compute_70,code=\"compute_70,sm_70\"`).
### 6.1.5 C++ Compatibility
The front end of the compiler processes CUDA source files according to C++ syntax rules. Full C++ is supported for the host code. However, only a subset of C++ is fully supported for the device code as described in C++ Language Support. (nvcc 的前端对主机代码支持全部C++语法，但对设备代码仅支持部分C++语法 )
### 6.1.6 64-Bit Compatibility
The 64-bit version of nvcc compiles device code in 64-bit mode (i.e., pointers are 64-bit). Device code compiled in 64-bit mode is only supported with host code compiled in 64-bit mode
## 6.2 CUDA Runtime
The runtime is implemented in the `cudart` library, which is linked to the application, either statically via `cudart.lib` or `libcudart.a`, or dynamically via `cudart.dll` or `libcudart.so`. ( CUDA运行时实现于 `cudart` 库，可以动态链接也可以静态链接 )

需要`cudart.dll`和/或`cudart.so`进行动态链接的应用程序通常将它们作为应用程序安装包的一部分

It is only safe to pass the address of CUDA runtime symbols between components that link to the same instance of the CUDA runtime.(只有在链接到CUDA运行时的同一实例的组件之间传递CUDA运行时符号的地址才是安全的)

All its entry point are prefixed with `cuda` ( API/入口点都以 `cuda` 为前缀)

CUDA Runtime提供了大量API用于管理设备存储
### 6.2.1 Initialization
As of CUDA 12.0, the `cudaInitDevice()` and `cudaSetDevice()` calls initialize the runtime and the primary context associated with the specified device. 

Absent these calls, the runtime will implicitly use device 0 and self-initialize as needed to process other runtime API requests. ( 如果没有调用以上两个函数，CUDA运行时会隐式使用device 0，并根据需要自初始化以处理其他运行时API请求 )，在计时运行时函数调用和解释来自运行时的第一个调用的错误代码时，需要记住这一点

在12.0之前，`cudaSetDevice()`不会初始化运行时，应用程序通常会使用无操作运行时调用`cudaFree(0)` (no-op runtime call) 来隔离运行时初始化与其他API活动(既为了计时，也为了错误处理)

The runtime creates a CUDA context for each device in the system (see Context for more details on CUDA contexts). This context is the *primary context* for this device and is initialized at the first runtime function which requires an active context on this device. (CUDA runtime为每个设备创建CUDA context，作为设备的主context，context实际会在第一个需要context的函数调用时被初始化 ) It is shared among all the host threads of the application. As part of this context creation, the device code is just-in-time compiled if necessary (see Just-in-Time Compilation) and loaded into device memory. This all happens transparently. ( 设备码的即时编译以及被加载到设备存储的过程也是runtime为设备创建context的一部分，)

When a host thread calls `cudaDeviceReset()`, this destroys the primary context of the device the host thread currently operates on (i.e., the current device as defined in Device Selection). The next runtime function call made by any host thread that has this device as current will create a new primary context for this device. (reset后再调用runtime函数，runtime会再次为设备初始化context)

As of CUDA 12.0, `cudaSetDevice()` will now explicitly initialize the runtime after changing the current device for the host thread. Previous versions of CUDA delayed runtime initialization on the new device until the first runtime call was made after `cudaSetDevice()`. This change means that it is now very important to check the return value of `cudaSetDevice()` for initialization errors. ( CUDA 12.0以后，`cudaSetDevice()` 会显式初始化runtime，因此需要注意其返回状态 )The runtime functions from the error handling and version management sections of the reference manual do not initialize the runtime.
### 6.2.2 Device Memory
the runtime provides functions to allocate, deallocate, and copy device memory, as well as transfer data between host memory and device memory

Device memory can be allocated either as *linear memory* or as *CUDA arrays*

CUDA arrays are opaque memory layouts optimized for texture fetching. They are described in Texture and Surface Memory.

Linear memory is allocated in a single unified address space, which means that separately allocated entities can reference one another via pointers, for example, in a binary tree or linked list.
![[CUDA C++ Programming Guide-Table1.png]]
Linear memory is typically allocated using `cudaMalloc()` and freed using `cudaFree()` and data transfer between host memory and device memory are typically done using `cudaMemcpy().`

Linear memory can also be allocated through `cudaMallocPitch()` and `cudaMalloc3D()`. These functions are recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements described in Device Memory Accesses, therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory (using the `cudaMemcpy2D()` and `cudaMemcpy3D()` functions). The returned pitch (or stride) ( 间距 ) must be used to access array elements.
```
/∕ Host code

int width = 64, height = 64; 
float* devPtr; 
size_t pitch; 
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height); 
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

∕∕ Device code

__global__ void MyKernel(float* devPtr,size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) { 
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) { 
            float element = row[c]; 
        } 
    }
}
```
( pitch 表示了二维数据每一行所占的数据量/bytes )

```
∕∕ Host code

int width = 64, height = 64, depth = 64; 
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth); 
cudaPitchedPtr devPitchedPtr; 
cudaMalloc3D(&devPitchedPtr, extent); 
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

∕∕ Device code

__global__ void MyKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth) { 
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
cudaGetSymbolAddress() is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space. The size of the allocated memory is obtained through cudaGetSymbolSize().
### 6.2.3 Device Memory L2 Access Management
When a CUDA kernel accesses a data region in the global memory repeatedly, such data accesses can be considered to be persisting.( 反复访问同一个global memory区域 ) On the other hand, if the data is only accessed once, such data accesses can be considered to be streaming ( 仅访问该global memory区域一次 )

Starting with CUDA 11.0, devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache, potentially providing higher bandwidth and lower latency accesses to global memory
#### 6.2.3.1 L2 cache Set-Aside for Persisting Accesses
A portion of the L2 cache can be set aside to be used for persisting data accesses to global memory. Persisting accesses have prioritized use of this set-aside portion ( 留出区域 )of L2 cache, whereas normal or streaming, accesses to global memory can only utilize this portion of L2 when it is unused by persisting accesses. ( 设定一部分L2 cache区域专门用于persisting访问 )

The L2 cache set-aside size for persisting accesses may be adjusted, within limits: ( 在限制下调整留出区域的大小 )
```
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); ∕* set-aside 3∕4 of L2 cache ,for persisting accesses or the max allowed*∕
```
When the GPU is configured in Multi-Instance GPU (MIG) mode, the L2 cache set-aside functionality is disabled.

When using the Multi-Process Service (MPS), the L2 cache set-aside size cannot be changed by cudaDeviceSetLimit. Instead, the set-aside size can only be specified at start up of MPS server through the environment variable `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT.`
#### 6.2.3.2 L2 Policy for Persisting Access
An access policy window specifies a contiguous region of global memory and a persistence property in the L2 cache for accesses within that region. 

The code example below shows how to set an L2 persisting access window using a CUDA Stream
**CUDA Stream Example**
...
When a kernel subsequently executes in CUDA stream, memory accesses within the global memory extent `[ptr..ptr+num_bytes)` are more likely to persist in the L2 cache than accesses to other global memory locations.
### 6.2.4 Shared Memory
shared memory is allocated using the `__shared__` memory space specifier.

Shared memory is expected to be much faster than global memory as mentioned in Thread Hierarchy and detailed in Shared Memory. It can be used as scratchpad memory (or software managed cache) to minimize global memory accesses from a CUDA block as illustrated by the following matrix multiplication example

The following code sample is a straightforward implementation of matrix multiplication that does not take advantage of shared memory. Each thread reads one row of A and one column of B and computes the corresponding element of C as illustrated in Figure 8. A is therefore read B.width times from global memory and B is read A.height times. ( 在naive的实现中，global memory中，A矩阵的每个元素都被读取了`B.width`次，以计算C矩阵中的一行，B矩阵的每个元素都被读取了`A.height`次，以计算C矩阵中的一列)
![[CUDA C++ Programming Guide-Fig8.png]]

The following code sample is an implementation of matrix multiplication that does take advantage of shared memory. In this implementation, each thread block is responsible for computing one square sub-matrix Csub of C and each thread within the block is responsible for computing one element of Csub. By blocking the computation this way, we take advantage of fast shared memory and save a lot of global memory bandwidth since A is only read `(B.width / block_size)` times from global memory and B is read `(A.height / block_size)` times. ( 对于A中的一个元素，对其进行一次读取就可以计算C矩阵中的 `block_size` 个元素，因此计算完C的一行只需要读 `B.width / block_size` 即可 )
![[CUDA C++ Programming Guide-Fig9.png]]
### 6.2.8 Asynchronous Concurrent Execution
#### 6.2.8.8 Events
### 6.2.9 Multi-Device System
#### 6.2.9.1 Device Enumeration
#### 6.2.9.2 Device Selection
## 6.3 Versioning and Compatibility
There are two version numbers that developers should care about when developing a CUDA application: The compute capability that describes the general specifications and features of the compute device (see Compute Capability) and the version of the CUDA driver API that describes the features supported by the driver API and runtime ( Compute Capability版本用于确定计算设备的特性和规格，CUDA驱动API版本用于确定CUDA驱动API和运行时所支持的特性 )

The version of the driver API is defined in the driver header file as `CUDA_VERSION`. the driver API is *backward compatible*, meaning that applications, plug-ins, and libraries (including the CUDA runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases as illustrated in Figure 12. ( 驱动API是向后兼容的，即旧的API可以兼容新的设备 )
![[CUDA C++ Programming Guide-Figure25.png]]
The driver API is not *forward compatible*, which means that applications, plug-ins, and libraries (including the CUDA runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver. ( 驱动API不向前兼容，即新的驱动API无法兼容旧的设备驱动 )

It is important to note that there are limitations on the mixing and matching of versions that is supported:
- Since only one version of the CUDA Driver can be installed at a time on a system, the installed driver must be of the same or higher version than the maximum Driver API version against which any application, plug-ins, or libraries that must run on that system were built. ( 要保证当前设备驱动的版本比驱动API的版本相同或更新 )
- All plug-ins and libraries used by an application must use the same version of the CUDA Runtime unless they statically link to the Runtime, in which case multiple versions of the runtime can coexist in the same process space. ( 程序中使用的所有插件和库都必须和CUDA运行时版本相同，除非它们是静态链接到CUDA运行时的 ) Note that if nvcc is used to link the application, the static version of the CUDA Runtime library will be used by default, and all CUDA Toolkit libraries are statically linked against the CUDA Runtime. ( `nvcc` 默认使用静态的CUDA运行时库，并且静态链接CUDA工具包库 )
- All plug-ins and libraries used by an application must use the same version of any libraries that use the runtime (such as cuFFT, cuBLAS, …) unless statically linking to those libraries ( 和上一点类似，程序使用的所有插件和库必须和程序中任意一个使用运行时的库的版本相同，除非静态链接 )

For Tesla GPU products, CUDA 10 introduced a new forward-compatible upgrade path for the usermode components of the CUDA Driver. This feature is described in CUDA Compatibility. The requirements on the CUDA Driver version described here apply to the version of the user-mode components.
# 4 Hardware Implementation
The NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity.

the threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors. ( 当前线程块停止执行，SM就会调度新的线程块 )

A multiprocessor is designed to execute hundreds of threads concurrently. To manage such a large number of threads, it employs a unique architecture called SIMT (Single-Instruction, Multiple-Thread)

The instructions are pipelined, leveraging instruction-level parallelism within a single thread, as well as extensive thread-level parallelism through simultaneous hardware multithreading ( 每个线程的指令执行都是流水线的，因此单线程内就存在指令级并行 ) Unlike CPU cores, they are issued in order and there is no branch prediction or speculative execution. ( 指令按序发送，不存在分支预测和预测执行 )

The NVIDIA GPU architecture uses a little-endian representation.
## 7.1 SIMT Architecture
The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. ( warp内的线程从同一个程序地址开始执行，每个线程有自己的指令地址计数器和寄存器 ) A half-warp is either the first or second half of a warp. A quarter-warp is either the first, second, third, or fourth quarter of a warp.

When a multiprocessor is given one or more thread blocks to execute, it partitions them into warps and each warp gets scheduled by a *warp scheduler* for execution. ( SM实际以warp为单位调度，调度由warp调度器决定 ) The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0.

A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path. ( warp内出现分歧时，warp会按序执行分歧的各个分支 ) Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjoint code paths.  ( 分支分歧仅在warp内出现，不同warp的执行相互独立 )

For the purposes of correctness, the programmer can essentially ignore the SIMT behavior; however, substantial performance improvements can be realized by taking care that the code seldom requires threads in a warp to diverge. In practice, this is analogous to the role of cache lines in traditional code: Cache line size can be safely ignored when designing for correctness but must be considered in the code structure when designing for peak performance. Vector architectures, on the other hand, require the software to coalesce loads into vectors and manage divergence manually.

Starting with the NVIDIA Volta architecture, *Independent Thread Scheduling* allows full concurrency between threads, regardless of warp. With Independent Thread Scheduling, the GPU maintains execution state per thread, including a program counter and call stack ( 每个线程有自己的程序计数器和调用栈 ) and can yield execution at a per-thread granularity, either to make better use of execution resources or to allow one thread to wait for data to be produced by another. A schedule optimizer determines how to group active threads from the same warp together into SIMT units. This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity.

> **Note**: The threads of a warp that are participating in the current instruction are called the *active* threads, whereas threads not on the current instruction are *inactive* (disabled). Threads can be inactive for a variety of reasons including having exited earlier than other threads of their warp, having taken a different branch path than the branch path currently executed by the warp, or being the last threads of a block whose number of threads is not a multiple of the warp size.
## 7.2 Hardware Multithreading
The execution context (program counters, registers, and so on) for each warp processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp. ( 每个warp的执行上下文保存于片上，持续warp的整个声明周期 ) Therefore, switching from one execution context to another has no cost, and at every instruction issue time, a warp scheduler selects a warp that has threads ready to execute its next instruction (the active threads of the warp) and issues the instruction to those threads. ( 在每个指令发出时间，warp调度器都会选择一个包含了活跃线程的warp，对这些线程发出指令 )

In particular, each multiprocessor has a set of 32-bit registers that are partitioned among the warps, ( 每个warp有自己的一组32位寄存器 ) and a parallel data cache or shared memory that is partitioned among the thread blocks.

The number of blocks and warps that can reside and be processed together on the multiprocessor for a given kernel depends on the amount of registers and shared memory used by the kernel and the amount of registers and shared memory available on the multiprocessor. ( 可以驻留在SM上的warp和block数量受kernel所需的资源量限制 ) There are also a maximum number of resident blocks and a maximum number of resident warps per multiprocessor. These limits as well the amount of registers and shared memory available on the multiprocessor are a function of the compute capability of the device and are given in Compute Capabilities. If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch.

The total number of warps in a block is as follows:
$$
\text{ceil}(\frac T {W_{size}})
$$
where $W_{size} = 32$ is the warp size，$T$ is the number of threads per block
# 5 Performance Guidelines
## 8.1 Overall Performance Optimizatino Strategies
Performance optimization revolves around four basic strategies: 
- Maximize parallel execution to achieve maximum utilization;  ( 最大化利用率 )
- Optimize memory usage to achieve maximum memory throughput;  ( 最大化存储吞吐量 )
- Optimize instruction usage to achieve maximum instruction throughput; ( 最大化指令吞吐量 )
-  Minimize memory thrashing. ( 最小化存储抖动 )

Optimization efforts should therefore be constantly directed by measuring and monitoring the performance limiters, for example using the CUDA profiler. Also, comparing the floating-point operation throughput or memory throughput—whichever makes more sense—of a particular kernel to the corresponding peak theoretical throughput of the device indicates how much room for improvement there is for the kernel.
## 8.2 Maximize Utilization
To maximize utilization the application should be structured in a way that it exposes as much parallelism as possible and efficiently maps this parallelism to the various components of the system to keep them busy most of the time.
### 8.2.1 Application Level
At a high level, the application should maximize parallel execution between the host, the devices, and the bus connecting the host to the devices, by using asynchronous functions calls and streams as described in Asynchronous Concurrent Execution. It should assign to each processor the type of work it does best: serial workloads to the host; parallel workloads to the devices.

the algorithm to the CUDA programming model should try best to be in a way that the computations that require inter-thread communication are performed within a single thread block as much as possible.
### 8.2.2 Device Level
At a lower level, the application should maximize parallel execution between the multiprocessors of a device. ( 尽可能利用多的SM )

Multiple kernels can execute concurrently on a device, so maximum utilization can also be achieved by using streams to enable enough kernels to execute concurrently as described in Asynchronous Concurrent Execution.
### 8.2.3 Multiprocessor Level
At an even lower level, the application should maximize parallel execution between the various functional units within a multiprocessor. ( 最大化SM内的函数单元执行的并行度 ) tilization is therefore directly linked to the number of resident warps.

At every instruction issue time, a warp scheduler selects an instruction that is ready to execute. This instruction can be another independent instruction of the same warp, exploiting instruction-level parallelism ( 调度器把指令发送给同一个warp，即指令流水线，就是指令级并行 ), or more commonly an instruction of another warp, exploiting thread-level parallelism. ( 发送给另一个warp，就是线程级并行 ) The number of clock cycles it takes for a warp to be ready to execute its next instruction is called the *latency*, ( 一个warp到准备执行下一条指令的阶段所花费的时钟周期称为延迟 ) and full utilization is achieved when all warp schedulers always have some instruction to issue for some warp at every clock cycle during that latency period, or in other words, when latency is completely “hidden”. ( 调度器每个时钟周期都可以发指令发送给某个warp，说明延迟被完全隐藏了，即调度器不需要等待warp了 ) The number of instructions required to hide a latency of L clock cycles depends on the respective throughputs of these instructions (see Arithmetic Instructions for the throughputs of various arithmetic instructions). If we assume instructions with maximum throughput, it is equal to:
-  4L for devices of compute capability 5.x, 6.1, 6.2, 7.x and 8.x since for these devices, a multiprocessor issues one instruction per warp over one clock cycle for four warps at a time( 每个时钟周期向四个warp各发送一条指令 ), as mentioned in Compute Capabilities.
 -  2L for devices of compute capability 6.0 since for these devices, the two instructions issued every cycle are one instruction for two different warps.

The most common reason a warp is not ready to execute its next instruction is that the instruction’s input operands are not available yet. ( warp难以执行下一条指令的主要原因一般是下一条指令的输入操作数尚未准备好 ) If all input operands are registers, latency is caused by register dependencies, i.e., some of the input operands are written by some previous instruction(s) whose execution has not completed yet. In this case, the latency is equal to the execution time of the previous instruction and the warp schedulers must schedule instructions of other warps during that time. Execution time varies depending on the instruction. On devices of compute capability 7.x, for most arithmetic instructions, it is typically 4 clock cycles. This means that 16 active warps per multiprocessor (4 cycles, 4 warp schedulers) are required to hide arithmetic instruction latencies (assuming that warps execute instructions with maximum throughput, otherwise fewer warps are needed). ( 有4个时钟周期的等待时间，要让四个warp调度器保持忙碌，这4个周期，它们在每个周期都会向4个warp发送指令，四个周期就一共需要16个warp接受指令 ) If the individual warps exhibit instruction-level parallelism, i.e. have multiple independent instructions in their instruction stream, fewer warps are needed because multiple independent instructions from a single warp can be issued back to back. ( 如果正在执行指令的warp可以再接受新指令，即存在指令级并行，需要的空闲warp数量就可以减少 )

If some input operand resides in off-chip memory, the latency is much higher: typically hundreds of clock cycles. The number of warps required to keep the warp schedulers busy during such high latency periods depends on the kernel code and its degree of instruction-level parallelism. In general, more warps are required if the ratio of the number of instructions with no off-chip memory operands (i.e., arithmetic instructions most of the time) to the number of instructions with off-chip memory operands is low (this ratio is commonly called the arithmetic intensity of the program). ( 算数密度越低，访存越频繁，延迟时间就越长，要隐藏越长的延迟时间，就需要越多的空闲warp )

Another reason a warp is not ready to execute its next instruction is that it is waiting at some memory fence (Memory Fence Functions) or synchronization point (Synchronization Functions). A synchronization point can force the multiprocessor to idle as more and more warps wait for other warps in the same block to complete execution of instructions prior to the synchronization point. Having multiple resident blocks per multiprocessor can help reduce idling in this case, as warps from different blocks do not need to wait for each other at synchronization points. ( 线程同步是block级的，因此block内所有warp都需要等待，为了防止SM空闲，这时就需要调度其他block的warp，因此block数量多有助于隐藏同步带来的延迟 )

The number of blocks and warps residing on each multiprocessor for a given kernel call depends on the execution configuration of the call (Execution Configuration), the memory resources of the multiprocessor, and the resource requirements of the kernel as described in Hardware Multithreading. ( 驻留块/warp数量由SM可用资源数量、kernel请求资源数量、kernel call的执行配置共同决定 ) Register and shared memory usage are reported by the compiler when compiling with the `--ptxas-options=-v` option.

The total amount of shared memory required for a block is equal to the sum of the amount of statically allocated shared memory and the amount of dynamically allocated shared memory

The number of registers used by a kernel can have a significant impact on the number of resident warps. For example, for devices of compute capability 6.x, if a kernel uses 64 registers and each block has 512 threads and requires very little shared memory, then two blocks (i.e., 32 warps) can reside on the multiprocessor since they require 2x512x64 registers, which exactly matches the number of registers available on the multiprocessor. But as soon as the kernel uses one more register, only one block (i.e., 16 warps) can be resident since two blocks would require 2x512x65 registers, which are more registers than are available on the multiprocessor. Therefore, the compiler attempts to minimize register usage while keeping register spilling (see Device Memory Accesses) and the number of instructions to a minimum. Register usage can be controlled using the `maxrregcount` compiler option, the `__launch_bounds__()` qualifier as described in Launch Bounds, or the `__maxnreg__()` qualifier as described in Maximum Number of Registers per Thread.

The register file is organized as 32-bit registers. So, each variable stored in a register needs at least one 32-bit register, for example, a double variable uses two 32-bit registers. ( 寄存器堆由一组32位寄存器组成，存储在寄存器堆中的变量至少需要1个寄存器 )

The effect of execution configuration on performance for a given kernel call generally depends on the kernel code. Experimentation is therefore recommended. Applications can also parametrize execution configurations based on register file size and shared memory size, which depends on the compute capability of the device, as well as on the number of multiprocessors and memory bandwidth of the device, all of which can be queried using the runtime (see reference manual).

The number of threads per block should be chosen as a multiple of the warp size to avoid wasting computing resources with under-populated warps as much as possible. ( block中除以warp size余下的thread是inactive的 )
#### 8.2.3.1 Occupancy Calculator
Several API functions exist to assist programmers in choosing thread block size and cluster size based on register and shared memory requirements.
- The occupancy calculator API, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, can provide an occupancy prediction based on the block size and shared memory usage of a kernel. This function reports occupancy in terms of the number of concurrent thread blocks per multiprocessor. ( 基于内核使用的块大小和共享存储大小，预测其占用率，返回形式是SM活跃的线程块数量 )
     - Note that this value can be converted to other metrics. Multiplying by the number of warps per block yields the number of concurrent warps per multiprocessor; further dividing concurrent warps by max warps per multiprocessor gives the occupancy as a percentage.
 - The occupancy-based launch configurator APIs, `cudaOccupancyMaxPotentialBlockSize` and `cudaOccupancyMaxPotentialBlockSizeVariableSMem`, heuristically calculate an execution configuration that achieves the maximum multiprocessor-level occupancy. ( 为给定内核计算达到最大占用率的执行配置 )
 - The occupancy calculator API, `cudaOccupancyMaxActiveClusters`, can provided occupancy prediction based on the cluster size, block size and shared memory usage of a kernel. This function reports occupancy in terms of number of max active clusters of a given size on the GPU present in the system.

The CUDA Nsight Compute User Interface also provides a standalone occupancy calculator and launch configurator implementation in `<CUDA_Toolkit_Path>∕include∕cuda_occupancy.h` for any use cases that cannot depend on the CUDA software stack. The Nsight Compute version of the occupancy calculator is particularly useful as a learning tool that visualizes the impact of changes to the parameters that affect occupancy (block size, registers per thread, and shared memory per thread).
## 8.3 Maximize Memory Throughput
The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth.

That means minimizing data transfers between the host and the device, as detailed in Data Transfer between Host and Device, since these have much lower bandwidth than data transfers between global memory and the device.

That also means minimizing data transfers between global memory and the device by maximizing use of on-chip memory: shared memory and caches (i.e., L1 cache and L2 cache available on devices of compute capability 2.x and higher, texture cache and constant cache available on all devices).

for devices of compute capability 7.x, 8.x and 9.0, the same on-chip memory is used for both L1 and shared memory, and how much of it is dedicated to L1 versus shared memory is configurable for each kernel call.

The throughput of memory accesses by a kernel can vary by an order of magnitude depending on access pattern for each type of memory. The next step in maximizing memory throughput is therefore to organize memory accesses as optimally as possible based on the optimal memory access patterns described in Device Memory Accesses. This optimization is especially important for global memory accesses as global memory bandwidth is low compared to available on-chip bandwidths and arithmetic instruction throughput, so non-optimal global memory accesses generally have a high impact on performance.
### 8.3.1 Data Transfer between Host and Device
Applications should strive to minimize data transfer between the host and the device. One way to accomplish this is to move more code from the host to the device, even if that means running kernels that do not expose enough parallelism to execute on the device with full efficiency. Intermediate data structures may be created in device memory, operated on by the device, and destroyed without ever being mapped by the host or copied to host memory. ( 让一些中间数据在设备管理，不涉及主机，减少主机设备之间的数据传输 )

Also, because of the overhead associated with each transfer, batching many small transfers into a single large transfer always performs better than making each transfer separately.

(rest content passed)
### 8.3.2 Device Memory Accesses
An instruction that accesses addressable memory (i.e., global, local, shared, constant, or texture memory) might need to be re-issued multiple times depending on the distribution of the memory addresses across the threads within the warp. How the distribution affects the instruction throughput this way is specific to each type of memory and described in the following sections. For example, for global memory, as a general rule, the more scattered the addresses are, the more reduced the throughput is. ( 对全局存储的寻址越离散，吞吐量越低 )

**Global Memory**
Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. ( 对全局存储的访问以内存事务为单位 ) These memory transactions must be naturally aligned: Only the 32-, 64-, or 128- byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions. ( 存储事物访问的首地址一定是对齐的 )

When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions depending on the size of the word accessed by each thread and the distribution of the memory addresses across the threads. ( warp执行访存指令时，它会根据warp内各线程的访问数据大小以及数据的地址分布，对这些线程的访存进行合并，合并为1个或多个存储事务 ) In general, the more transactions are necessary, the more unused words are transferred in addition to the words accessed by the threads, reducing the instruction throughput accordingly. For example, if a 32-byte memory transaction is generated for each thread’s 4-byte access, throughput is divided by 8.

To maximize global memory throughput, it is therefore important to maximize coalescing by: ( 要最大化全局存储吞吐，就要最大化合并 )
-  Following the most optimal access patterns based on Compute Capability 5.x, Compute Capability 6.x, Compute Capability 7.x, Compute Capability 8.x and Compute Capability 9.0
-  Using data types that meet the size and alignment requirement detailed in the section Size and Alignment Requirement below,
-  Padding data in some cases, for example, when accessing a two-dimensional array as described in the section Two-Dimensional Arrays below.

**Size and Alignment Requirement**
Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes. ( 全局内存指令支持对1/2/4/8/16字节数据的读写 ) Any access (via a variable or a pointer) to data residing in global memory compiles to a single global memory instruction if and only if the size of the data type is 1, 2, 4, 8, or 16 bytes and the data is naturally aligned (i.e., its address is a multiple of that size). ( 当且仅当要访问的数据大小恰好为1/2/4/8/16字节，且数据的首地址是对齐的，对该数据的访问可以仅编译成一条指令 )

If this size and alignment requirement is not fulfilled, the access compiles to multiple instructions ( 数据不满足对齐要求，则需要多条指令访问 ) with interleaved access patterns ( 交错访问模式 ) that prevent these instructions from fully coalescing. It is therefore recommended to use types that meet this requirement for data that resides in global memory.

The alignment requirement is automatically fulfilled for the Built-in Vector Types.

For structures, the size and alignment requirements can be enforced by the compiler using the alignment specifiers`__align__(8)` or` __align__(16)` 

Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes. ( 全局存储中任何数据的首地址的索引保证是256的倍数 )

**Two-Dimensional Arrays**
A common global memory access pattern is when each thread of index (tx,ty) uses the following address to access one element of a 2D array of width width, located at address BaseAddress of type `type*` (where `type` meets the requirement described in Maximize Utilization):
```
 BaseAddress + width * ty + tx
```
For these accesses to be fully coalesced, both the width of the thread block and the width of the array must be a multiple of the warp size. ( 宽度大小为warp的倍数方便将warp映射到处理连续的数据 )

In particular, this means that an array whose width is not a multiple of this size will be accessed much more efficiently if it is actually allocated with a width rounded up to the closest multiple of this size and its rows padded accordingly. The `cudaMallocPitch()` and `cuMemAllocPitch()` functions and associated memory copy functions described in the reference manual enable programmers to write non-hardware-dependent code to allocate arrays that conform to these constraints.

**Local Memory**
Local memory accesses only occur for some automatic variables as mentioned in Variable Memory Space Specifiers. ( 局部存储访问仅针对一些情况下的自动变量 ) Automatic variables that the compiler is likely to place in local memory are:
 - Arrays for which it cannot determine that they are indexed with constant quantities,
 - Large structures or arrays that would consume too much register space,
 - Any variable if the kernel uses more registers than available (this is also known as register spilling).

Inspection of the PTX assembly code (obtained by compiling with the `-ptx` or `-keep` option) will tell if a variable has been placed in local memory during the first compilation phases as it will be declared using the `.local` mnemonic and accessed using the `ld.local` and `st.local` mnemonics. ( PTX码中，局部变量会以 `.local` 声明 ) Even if it has not, subsequent compilation phases might still decide otherwise though if they find it consumes too much register space for the targeted architecture: Inspection of the *cubin* object using `cuobjdump` will tell if this is the case。 Also, the compiler reports total local memory usage per kernel (`lmem`) when compiling with the `--ptxas-options=-v` option. Note that some mathematical functions have implementation paths that might access local memory.

local memory accesses have the same high latency and low bandwidth as global memory accesses and are subject to the same requirements for memory coalescing。 Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address (for example, same index in an array variable, same member in a structure variable). On devices of compute capability 5.x onwards, local memory accesses are always cached in L2 in the same way as global memory accesses (see Compute Capability 5.x and Compute Capability 6.x)

**Shared Memory**
To achieve high bandwidth, shared memory is divided into equally-sized memory modules, called banks, ( bank是shared存储中等大小的存储模块 ) which can be accessed simultaneously ( 不同的bank可以被同时访问，因此bank可以用于共享存储访问的并行化 ). Any memory read or write request made of n addresses that fall in n distinct memory banks can therefore be serviced simultaneously, yielding an overall bandwidth that is n times as high as the bandwidth of a single module. 

However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized. ( 对同一bank内的地址的连续访问会被并行化，即存储体冲突 )The hardware splits a memory request with bank conflicts into as many separate conflict-free requests as necessary, decreasing throughput by a factor equal to the number of separate memory requests. If the number of separate memory requests is n, the initial memory request is said to cause n-way bank conflicts.

To get maximum performance, it is therefore important to understand how memory addresses map to memory banks in order to schedule the memory requests so as to minimize bank conflicts. This is described in Compute Capability 5.x, Compute Capability 6.x, Compute Capability 7.x, Compute Capability 8.x, and Compute Capability 9.0 for devices of compute capability 5.x, 6.x, 7.x, 8.x, and 9.0 respectively.

**Constant Memory**
passed

**Texture and Surface Memory**
passed
## 8.4 Maximize Instruction Throughput
To maximize instruction throughput the application should:
 -  Minimize the use of arithmetic instructions with low throughput; this includes trading precision for speed when it does not affect the end result, such as using intrinsic instead of regular functions (intrinsic functions are listed in Intrinsic Functions), single-precision instead of doubleprecision, or flushing denormalized numbers to zero; ( 少使用低吞吐的算数指令，包括以精度换速度、使用内联函数 )
 -  Minimize divergent warps caused by control flow instructions as detailed in Control Flow Instructions ( 最小化由控制指令引起的warp内分歧 )
 -  Reduce the number of instructions, for example, by optimizing out synchronization points whenever possible as described in Synchronization Instruction or by using restricted pointers as described in `__restrict__`.

In this section, throughputs are given in number of operations per clock cycle per multiprocessor. ( 每个SM每个时钟周期执行的操作数 ) For a warp size of 32, one instruction corresponds to 32 operations, so if N is the number of operations per clock cycle, the instruction throughput is N/32 instructions per clock cycle. ( 每个指令对应32个操作 )

All throughputs are for one multiprocessor. They must be multiplied by the number of multiprocessors in the device to get throughput for the whole device.
### 8.4.1 Arithmetic Instructions
The following table gives the throughputs of the arithmetic instructions that are natively supported in hardware for devices of various compute capabilities.
( 总结一下native指令有哪些：
- 16bit浮点加、乘、乘加
- 32位浮点加、乘、乘加
- 64位浮点加、乘、乘加
- 32bit浮点倒数、平方根倒数、二为底对数 `__log2f` 、二为底指数 `exp2f` 、正弦 `__sinf` 、余弦 `__cosine`)
- 32位整型加、拓展精度加、减、拓展精度减
- 32位整型乘、乘加、拓展精度乘加
- 24位整型乘 `__[u]mul24`
- 32位整型移位
- 比较、最小、最大
- 32位整型反转
- 位字段提取/插入
- 32位异或、与、或
- 先导零计数、最高有效非符号位
- 计数位为1的个数(population count)
- 线程束内数据交换(warp shuffle)
- 线程束内归约(warp reduce)
- 线程束内投票(warp vote)
- 绝对差值和(sum of absolute difference)
- SIMD video instructions `vabs-diff2`
- SIMD video instructions `vabs-diff2`
- All other SIMD video instructions
- 从8位、16位整型到32位整型的类型转换
- 到64位整型的类型转换和从64位整型的类型转换(from and to)
- 所有其他的类型转换
- 16位DPX
- 32位DPX
)
In general, code compiled with `-ftz=true` (denormalized numbers are flushed to zero 非正规化数刷新至零) tends to have higher performance than code compiled with `-ftz=false`. Similarly, code compiled with `-prec-div=false` (less precise division 低精度除法) tends to have higher performance code than code compiled with `-prec-div=true`, and code compiled with `-prec-sqrt=false` (less precise square root 低精度平方根) tends to have higher performance than code compiled with `-prec-sqrt=true`. The nvcc user manual describes these compilation flags in more details

partial content is passed

**Half Precision Arithmetic**
In order to achieve good performance for 16-bit precision floating-point add, multiply or multiply-add, it is recommended that the `half2` datatype is used for `half` precision and `__nv_bfloat162` be used for `__nv_bfloat16` precision. Vector intrinsics (for example, `__hadd2`, `__hsub2`, `__hmul2`, `__hfma2`) can then be used to do two operations in a single instruction. Using `half2` or `__nv_bfloat162` in place of two calls using `half` or `__nv_bfloat16` may also help performance of other intrinsics, such as warp shuffles.

The intrinsic `__halves2half2` is provided to convert two half precision values to the `half2` datatype. The intrinsic `__halves2bfloat162` is provided to convert two `__nv_bfloat` precision values to the `__nv_bfloat162` datatype.
### 8.4.2 Control Flow Instructions
To obtain best performance in cases where the control flow depends on the thread ID, the controlling condition should be written so as to minimize the number of divergent warps. A trivial example is when the controlling condition only depends on (`threadIdx ∕ warpSize`) where `warpSize` is the warp size. In this case, no warp diverges since the controlling condition is perfectly aligned with the warps.

Sometimes, the compiler may unroll loops or it may optimize out short if or switch blocks by using branch predication instead, as detailed below. In these cases, no warp can ever diverge. The programmer can also control loop unrolling using the `#pragma unroll` directive

When using branch predication none of the instructions whose execution depends on the controlling condition gets skipped. ( 不同分支内的动作的对应指令所有线程都会执行 ) Instead, each of them is associated with a per-thread condition code or predicate that is set to true or false based on the controlling condition and although each of these instructions gets scheduled for execution, only the instructions with a true predicate are actually executed. Instructions with a false predicate do not write results, and also do not evaluate addresses or read operands.
### 8.4.3 Synchronization Instruction
Throughput for `__syncthreads()` is 32 operations per clock cycle for devices of compute capability 6.0, 16 operations per clock cycle for devices of compute capability 7.x as well as 8.x and 64 operations per clock cycle for devices of compute capability 5.x, 6.1 and 6.2 ( `__syncthreads()` 的吞吐量的意思和其他指令的吞吐量是一样的，即每个时钟周期内可以完成几次 `__syncthreads()` 操作 )
## 8.5 Minimize Memory Thrashing
passed
# 6 CUDA-Enabled GPUs
https://developer.nvidia.com/cuda-gpus lists all CUDA-enabled devices with their compute capability.

The compute capability, number of multiprocessors, clock frequency, total amount of device memory, and other properties can be queried using the runtime (see reference manual).
# 7 C++ Language Extensions
## 10.1 Function Execution Space Specifiers
Function execution space specifiers denote whether a function executes on the host or on the device and whether it is callable from the host or from the device.
### 10.1.1. `__global__`
The `__global__` execution space specifier declares a function as being a kernel. Such a function is: 
-  Executed on the device, ( 执行于设备 )
-  Callable from the host, ( 可从主机调用 )
-  Callable from the device for devices of compute capability 5.0 or higher (see CUDA Dynamic Parallelism for more details). ( 可从设备调用 )

A `__global__` function must have void return type, and cannot be a member of a class. ( 返回类型必须为 `void` ，不可以是类成员函数 )

Any call to a `__global__` function must specify its execution configuration as described in Execution Configuration. ( 调用时必须指定执行配置 )
 
A call to a `__global__` function is asynchronous, meaning it returns before the device has completed its execution. ( 对 `__global__` 函数的调用是异步的，即 `__global__` 函数会在设备完成执行之前就返回 )
### 10.1.2. `__device__`

The `__device__` execution space specifier declares a function that is:
-  Executed on the device,
-  Callable from the device only. ( 仅能从设备调用 )

The `__global__` and `__device__` execution space specifiers cannot be used together.
### 10.1.3. `__host__`
The `__host__` execution space specifier declares a function that is:
-  Executed on the host, ( 执行于主机 )
-  Callable from the host only. ( 仅能从主机调用 )

It is equivalent to declare a function with only the `__host__` execution space specifier or to declare it without any of the `__host__`, `__device__`, or `__global__` execution space specifier; in either case the function is compiled for the host only. ( `__host__` 相当于默认的执行空间指定符 )

The `__global__` and `__host__` execution space specifiers cannot be used together. The `__device__` and `__host__` execution space specifiers can be used together however, in which case the function is compiled for both the host and the device. ( 可以和 `__device__` 一起使用，此时函数既为设备编译，也为主机编译 ) The `__CUDA_ARCH__` macro introduced in Application Compatibility can be used to differentiate code paths between host and device
### 10.1.4. Undefined behavior
passed
### 10.1.5 `__noinline__` and `__forceinline__`
The compiler inlines any `__device__` function when deemed appropriate.

the rest is passed
## 10.2 Variable Memory Space Specifiers
Variable memory space specifiers denote the memory location on the device of a variable. 
An automatic variable declared in device code without any of the `__device__`, `__shared__` and `__constant__` memory space specifiers described in this section generally resides in a register. ( 没有存储空间指示符修饰的自动变量默认至于寄存器 ) However in some cases the compiler might choose to place it in local memory, which can have adverse performance consequences as detailed in Device Memory Accesses.
### 10.2.3 `__shared__`
The __shared__ memory space specifier, optionally used together with __device__, declares a variable that:
 - Resides in the shared memory space of a thread block,
 -  Has the lifetime of the block,( 生命周期等同于线程块 )
 -  Has a distinct object per block, ( 每个线程块都拥有其一个唯一的对象 )
 -  Is only accessible from all the threads within the block,
 -  Does not have a constant address.
### 10.2.6  `__restrict__`
`nvcc` supports restricted pointers via the `__restrict__` keyword.

Restricted pointers were introduced in C99 to alleviate the aliasing problem that exists in C-type languages, and which inhibits all kind of optimization from code re-ordering to common sub-expression elimination.

By making `a`, `b`, and `c` restricted pointers, the programmer asserts to the compiler that the pointers are in fact not aliased which in this case means writes through c would never overwrite elements of a or b. This changes the function prototype as follows:
```
void foo(const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ c);
```
Note that all pointer arguments need to be made restricted for the compiler optimizer to derive any benefit.

The effects here are a reduced number of memory accesses and reduced number of computations. This is balanced by an increase in register pressure due to “cached” loads and common subexpressions. ( 访问和计算次数减少，但是寄存器压力增大 )

Since register pressure is a critical issue in many CUDA codes, use of restricted pointers can have negative performance impact on CUDA code, due to reduced occupancy. ( 寄存器压力增大可能导致占用率减少 )
## 10.4 Built-in Variables
Built-in variables specify the grid and block dimensions and the block and thread indices. They are only valid within functions that are executed on the device. ( 内建变量仅在正在设备上执行的函数内有效 )
## 7.6 Synchronization Functions
```
void __syncthreads();
```
waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to `__syncthreads()` are visible to all threads in the block. `__syncthreads()` is used to coordinate communication between the threads of the same block.

When some threads within a block access the same addresses in shared or global memory, there are potential read-after-write, write-after-read, or write-after-write hazards for some of these memory accesses. These data hazards can be avoided by synchronizing threads in-between these accesses.

`__syncthreads()` is allowed in conditional code but only if the conditional evaluates identically across the entire thread block, otherwise the code execution is likely to hang or produce unintended side effects.

the rest is passed
## 10.10 Read-Only Data Cache Load Function
The read-only data cache load function is only supported by devices of compute capability 5.0 and higher.
```
T __ldg(const T* address);
```
returns the data of type T located at address address

With the `cuda_fp16.h` header included, T can be `__half` or `__half2`.
Similarly, with the `cuda_bf16`.h header included, T can also be `__nv_bfloat16` or `__nv_bfloat162`.
## 7.24 Warp Matrix Functions
C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form `D=A*B+C`. ( C++线程束矩阵操作会利用张量核心加速矩阵乘加计算 ) These operations are supported on mixed-precision floating point data ( 这些操作支持混合精度浮点数据 ) for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a warp. ( 该操作需要线程束内所有线程协同操作 ) In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire warp, otherwise the code execution is likely to hang.
### 7.24.1 Description
All following functions and types are defined in the namespace `nvcuda::wmma`. Sub-byte operations are considered preview, i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. ( 该节中的亚字节操作未来可能会有改动 ) This extra functionality is defined in the `nvcuda::wmma::experimental` namespace.
```cpp
template<typename Use, int m, int n, int k, typename T, typename Layout=void> 
class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm); void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);

void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);

void fill_fragment(fragment<...> &a, const T& v);

void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

**fragment**
 An overloaded class containing a section of a matrix distributed across all threads in the warp. ( fragment是一个重载类，它包含了一个矩阵的一部分，这部分数据分散于warp内的所有线程 ) The mapping of matrix elements into `fragment` internal storage is unspecified and subject to change in future architectures. ( 矩阵元素到fragment内部存储的映射无法指定 )

Only certain combinations of template arguments are allowed. The first template parameter specifies how the fragment will participate in the matrix operation. ( 第一个模板参数指定了fragment如何参与到矩阵操作中 ) Acceptable values for `Use` are:
-  `matrix_a` when the fragment is used as the first multiplicand, A,
-  `matrix_b` when the fragment is used as the second multiplicand, B, or
-  `accumulator` when the fragment is used as the source or destination accumulators (`C` or `D`, respectively).

The `m`, `n` and `k` sizes describe the shape of the warp-wide matrix tiles participating in the multiplyaccumulate operation. The dimension of each tile depends on its role. For `matrix_a` the tile takes dimension `m x k`; for `matrix_b` the dimension is `k x n`, and accumulator tiles are `m x n`.

The data type, `T`, may be `double`, `float`, `__half`, `__nv_bfloat16`, `char`, or `unsigned char` for multiplicands ( 被乘数 ) and `double`, `float`, `int`, or `__half` for accumulators. ( 累加数 ) As documented in Element Types and Matrix Sizes, limited combinations of accumulator and multiplicand types are supported. ( 仅支持受限组合的累加数和被乘数类型组合 ) The `Layout` parameter must be specified for `matrix_a` and `matrix_b` fragments. `row_major` or `col_major` indicate that elements within a matrix row or column are contiguous in memory, respectively. ( 对于被乘数，其 `layout` 参数用于指定它是按行存储还是按列存储 )The `Layout` parameter for an accumulator matrix should retain the default value of `void`.  ( 对于累加数，其 `layout` 参数一般保持默认即可 ) A row or column layout is specified only when the accumulator is loaded or stored as described below.

**load_matrix_sync**
Waits until all warp lanes have arrived at `load_matrix_sync` and then loads the matrix `fragment` a from memory. ( warp内所有线程同步以后，开始装载 `fragment` ) `mptr` must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. `ldm` describes the stride in elements ( stride的单位是element ) between consecutive rows (for row major layout) or columns (for column major layout) and must be a multiple of 8 for `__half` element type or multiple of 4 for `float` element type. (i.e., multiple of 16 bytes in both cases). ( stride的大小必须是16字节的倍数 ) If the fragment is an accumulator, ( 若fragment是累加数，则需要在此处指定其layout ) the layout argument must be specified as either `mem_row_major` or `mem_col_major`. For `matrix_a` and `matrix_b` fragments, the layout is inferred from the fragment’s layout parameter. ( 若fragment是被乘数，可以不指定 ) The values of `mptr`, `ldm`, `layout` and all template parameters for a must be the same for all threads in the warp. This function must be called by all threads in the warp, or the result is undefined. ( 参数对于warp内的所有线程都应一致，warp内的所有线程都应该调用这个函数 )

**store_matrix_sync**
Waits until all warp lanes have arrived at `store_matrix_sync` and then stores the matrix fragment a to memory. ( warp内所有线程同步以后，将 `fragment` 写入存储 ) `mptr` must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. `ldm` describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) and must be a multiple of 8 for `__half` element type or multiple of 4 for float element type. (i.e., multiple of 16 bytes in both cases). The layout of the output matrix must be specified as either `mem_row_major` or `mem_col_major`. The values of `mptr`, `ldm`, `layout` and all template parameters for a must be the same for all threads in the warp. ( 其余规则和 `load_matrix_sync` 一致 )

**fill_fragment**
Fill a matrix fragment with a constant value `v`. Because the mapping of matrix elements to each fragment is unspecified, this function is ordinarily called by all threads in the warp with a common value for `v`. 

**mma_sync**
Waits until all warp lanes have arrived at `mma_sync`, and then performs the warp-synchronous matrix multiply-accumulate operation `D=A*B+C`. ( 等待warp内所有线程同步，然后执行矩阵乘累加 ) The in-place operation, `C=A*B+C`, is also supported. The value of `satf` and template parameters for each matrix fragment must be the same for all threads in the warp. ( 同样，warp内所有线程传入的参数必须一致 ) Also, the template parameters m, n and k must match between fragments A, B, C and D. This function must be called by all threads in the warp, or the result is undefined. ( 同样，warp内所有线程都必须调用这一函数 )

If `satf` (saturate to finite value) mode is `true`, the following additional numerical properties apply for the destination accumulator:
-  If an element result is `+Infinity`, the corresponding accumulator will contain `+MAX_NORM`
-  If an element result is `-Infinity`, the corresponding accumulator will contain `-MAX_NORM`
-  If an element result is `NaN`, the corresponding accumulator will contain `+0`

Because the map of matrix elements into each thread’s `fragment` is unspecified, individual matrix elements must be accessed from memory (shared or global) after calling `store_matrix_sync`. ( 因为 `fragment` 的哪个元素存于warp内哪个线程是未指定的，要访问特定的单个矩阵元素，就需要通过 `store_matrix_sync` 将全部矩阵元素储存后再访问 ) In the special case where all threads in the warp will apply an element-wise operation uniformly to all fragment elements, direct element access can be implemented using the following `fragment` class members.
```cpp
enum fragment<Use, m, n, k, T, Layout>::num_elements; 
T fragment<Use, m, n, k, T, Layout>::x[num_elements];
```
### 10.24.7 Example
ref to book
## 10.26 Asynchronous Barrier
The NVIDIA C++ standard library introduces a GPU implementation of std::barrier. Along with the implementation of std::barrier the library provides extensions that allow users to specify the scope of barrier objects. ( NVIDIA C++标准库还提供了指定 barrier 目标范围的拓展 ) The barrier API scopes are documented under Thread Scopes. Devices of compute capability 8.0 or higher provide hardware acceleration for barrier operations and integration of these barriers with the `memcpy_async` feature. ( compute capability 8.0以上的设备提供了对 barrier 操作的硬件加速，以及和 `memecpy_async` 特性的集成 ) On devices with compute capability below 8.0 but starting 7.0, these barriers are available without hardware acceleration.

`nvcuda::experimental::awbarrier` is deprecated in favor of `cuda::barrier`
### 10.26.1 Simple Synchronization Pattern
Without the arrive/wait barrier, synchronization is achieved using `__syncthreads()` (to synchronize all threads in a block) or group.sync() when using Cooperative Groups.

This pattern has three stages:
- Code before sync performs memory updates that will be read after the sync.
- Synchronization point
- Code after sync point with visibility of memory updates that happened before sync point.
### 10.26.2 Temporal Splitting and Five Stages of Synchronization
The temporally-split synchronization pattern with the `std::barrier` is as follows

code ref to book

In this pattern, the synchronization point (`block.sync()`) is split into an arrive point (`bar. arrive()`) and a wait point (`bar.wait(std::move(token)`)). A thread begins participating in a `cuda::barrier` with its first call to `bar.arrive()`. ( 一个同步点被分成了一个到达点和一个等待点，一个线程在它第一次调用 `bar.arrive()` 时开始参与 `cuda::barrier` ) When a thread calls `bar. wait(std::move(token)`) it will be blocked until participating threads have completed `bar.arrive()` the expected number of times as specified by the expected arrival count argument passed to `init()`. ( 当一个线程调用了 `bar.wait(std::move(token))` 它就会被阻塞，直到完成 `bar.arrive()` 的线程数量达到 `init()` 中设定的预期数量 ) Memory updates that happen before participating threads’ call to `bar.arrive()` are guaranteed to be visible to participating threads after their call to `bar.wait(std::move(token))`. ( 在参与线程调用了 `bar.wait(std::move(token))` 之后，在 `bar.arrive()` 之前的存储更新将保证对所有参与线程可见 ) Note that the call to `bar.arrive()` does not block a thread, it can proceed with other work that does not depend upon memory updates that happen before other participating threads’ call to `bar.arrive()`

The *arrive* and then *wait* pattern has five stages which may be iteratively repeated: 
- Code before arrive performs memory updates that will be read after the wait.
- Arrive point with implicit memory fence (i.e., equivalent to `atomic_thread_fence(memory_order_seq_cst, thread_scope_block)`).
- Code between arrive and wait.
- Wait point.
- Code after the wait, with visibility of updates that were performed before the arrive.
### 10.26.3 Bootstrap Initialzation, Expected Arrival Count, and Participation
Initialization must happen before any thread begins participating in a `cuda::barrier`. ( 在任意线程加入 `cuda::barrier` 之前，必须进行初始化 )

Before any thread can participate in `cuda::barrier`, the barrier must be initialized using `init()` with an expected arrival count, `block.size()` in this example. Initialization must happen before any thread calls `bar.arrive()` ( 在任意线程达到 `bar.arrive()`， 之前，线程0必须要执行 `init()` ) This poses a bootstrapping challenge in that threads must synchronize before participating in the cuda::barrier, but threads are creating a cuda::barrier in order to synchronize. 

The second parameter of `init()` is the expected arrival count, i.e., the number of times bar. arrive() will be called by participating threads before a participating thread is unblocked from its call to `bar.wait(std::move(token))`.

A cuda::barrier is flexible in specifying how threads participate (split arrive/wait) and which threads participate. In contrast this_thread_block.sync() from cooperative groups or `__syncthreads()` is applicable to whole-thread-block and `__syncwarp(mask)` is a specified subset of a warp
If the intention of the user is to synchronize a full thread block or a full warp we recommend using `__syncthreads()` and `__syncwarp(mask)` respectively for performance reasons.
### 10.26.4 A Barrier's Phase: Arrival, Countdown, Completion, and Reset
A cuda::barrier counts down from the expected arrival count to zero as participating threads call bar.arrive(). When the countdown reaches zero, a cuda::barrier is complete for the current phase. ( 计数至0的时候，一个 `cuda::barrier` 对于当前阶段就结束了 ) When the last call to bar.arrive() causes the countdown to reach zero, the countdown is automatically and atomically reset. The reset assigns the countdown to the expected arrival count, and moves the cuda::barrier to the next phase. ( 然后计数被重置，进入下一阶段的同步 )

A token object of class `cuda::barrier::arrival_token`, as returned from `token=bar.arrive()`, is associated with the current phase of the barrier. ( 由 `bar.arrive()` 返回的 `cuda::barrier::arrival_token` 和当前阶段的barrier相关 ) A call to `bar.wait(std::move(token))` blocks the calling thread while the `cuda::barrier` is in the current phase, i.e., while the phase associated with the token matches the phase of the `cuda::barrier`. ( 当一个线程调用了 `bar.wait(std::move(token))` ，若参数 `token` 匹配当前阶段的 `cuda::barrier` 该线程被阻塞 ) If the phase is advanced (because the countdown reaches zero) before the call to `bar.wait(std::move(token))` then the thread does not block; ( 若阶段不匹配，说明已经有足够数量的线程完成了上一阶段的同步，则线程不被阻塞 ) if the phase is advanced while the thread is blocked in bar.wait(std::move(token)), the thread is unblocked ( 若线程正在阻塞中，发现阶段不匹配了，说明足够多的线程已经 `bar.arrive()` ，则线程被解锁 )

It is essential to know when a reset could or could not occur, especially in non-trivial arrive/wait synchronization patterns.
- A thread’s calls to `token=bar.arrive()` and `bar.wait(std::move(token))` must be sequenced such that `token=bar.arrive()` occurs during the cuda::barrier’s current phase, and bar.wait(std::move(token)) occurs during the same or next phase.
- A thread’s call to bar.arrive() must occur when the barrier’s counter is non-zero. After barrier initialization, if a thread’s call to bar.arrive() causes the countdown to reach zero then a call to bar.wait(std::move(token)) must happen before the barrier can be reused for a subsequent call to bar.arrive().
 - `bar.wait()` must only be called using a token object of the current phase or the immediately preceding phase. For any other values of the token object, the behavior is undefined.

For simple arrive/wait synchronization patterns, compliance with these usage rules is straightforward.
### 10.26.5 Spatial Partitioning (also known as Warp Specialization)
A thread block can be spatially partitioned such that warps are specialized to perform independent computations. ( 线程块可以执行空间划分，使得warps执行各自独立的计算 ) Spatial partitioning is used in a producer or consumer pattern, where one subset of threads produces data that is concurrently consumed by the other (disjoint) subset of threads. ( 空间划分可以用于生产者消费者模式，其中一部分线程生产数据，另一部分线程同时消耗数据 )

A producer/consumer spatial partitioning pattern requires two one sided synchronizations to manage a data buffer between the producer and consumer. ( 生产者/消费者空间划分模式需要同步以管理共同的数据缓存 )
![[CUDA C++ Programming Guide-ProducerConsumer.png]]

Producer threads wait for consumer threads to signal that the buffer is ready to be filled; ( 生产者等待消费者发出缓存等待被填充的信号 )however, consumer threads do not wait for this signal. Consumer threads wait for producer threads to signal that the buffer is filled; ( 消费者等待生产者发出缓存已经填充满的信号 ) however, producer threads do not wait for this signal. For full producer/consumer concurrency this pattern has (at least) double buffering where each buffer requires two `cuda::barriers`. ( 这需要双缓冲，以及两个 barrier )

code ref to book

the rest content is temporarily passed
## 10.27 Asynchronous Data Copies
CUDA 11 introduces Asynchronous Data operations with `memcpy_async` API to allow device code to explicitly manage the asynchronous copying of data. ( 使用 `memcpy_async` 以管理设备数据的异步拷贝 ) The `memcpy_async` feature enables CUDA kernels to overlap computation with data movement. ( 重叠数据移动和计算 )
### 10.27.1 `memcpy_async` API
The `memcpy_async` APIs are provided in the `cuda∕barrier`, `cuda∕pipeline`, and `cooperative_groups∕memcpy_async.h` header files.

The `cuda::memcpy_async` APIs work with `cuda::barrier` and `cuda::pipeline` synchronization primitives, while the `cooperative_groups::memcpy_async` synchronizes using `coopertive_groups::wait`. ( `cuda::memcpy_async` 可以与 `cuda::barrier` 和 `cuda::pipline` 原语进行同步，而 `cooperative_groups::memcpu_async` 需要与 `coopertive_groups::wait` 进行同步 )

These APIs have very similar semantics: copy objects from `src` to `dst` as-if performed by another thread which, on completion of the copy, can be synchronized through `cuda::pipeline`, `cuda::barrier`, or `cooperative_groups::wait`.

The complete API documentation of the `cuda::memcpy_async` overloads for `cuda::barrier` and `cuda::pipeline` is provided in the libcudacxx API documentation along with some examples.

The API documentation of `cooperative_groups::memcpy_async` is provided in the cooperative groups Section of the documentation.

The `memcpy_async` APIs that use `cuda::barrier` and `cuda::pipeline` require compute capability 7.0 or higher. On devices with compute capability 8.0 or higher, `memcpy_async` operations from global to shared memory can benefit from hardware acceleration.
### 10.27.2. Copy and Compute Pattern - Staging Data Through Shared Memory
CUDA applications often employ a *copy and compute* pattern that:
-  fetches data from global memory,
-  stores data to shared memory, and
-  performs computations on shared memory data, and potentially writes results back to global memory.

The following sections illustrate how this pattern can be expressed without and with the `memcpy_async` feature:
-  The section Without `memcpy_async` introduces an example that does not overlap computation with data movement and uses an intermediate register to copy data.
-  The section With `memcpy_async` improves the previous example by introducing the `cooperative_groups::memcpy_async` and the `cuda::memcpy_async` APIs to directly copy data from global to shared memory without using intermediate registers.
-  Section Asynchronous Data Copies using `cuda::barrier` shows memcpy with cooperative groups and barrier
- Section Single-Stage Asynchronous Data Copies using cuda::pipeline show memcpy with single stage pipeline
-  Section Multi-Stage Asynchronous Data Copies using cuda::pipeline show memcpy with multi stage pipeline
### 10.27.3. Without memcpy_async
Without memcpy_async, the copy phase of the copy and compute pattern is expressed as `shared[local_idx]` = `global[global_idx]`. This global to shared memory copy is expanded to a read from global memory into a register, followed by a write to shared memory from the register. ( 通常的global memory到shared memory的写入会经过中间的寄存器 )

the rest is passed
### 10.27.4 With `memcpy_async`
With memcpy_async, the assignment of shared memory from global memory
```
shared[local_idx] = global_in[global_idx];
```
is replaced with an asynchronous copy operation from cooperative groups
```
cooperative_groups::memcpy_async(group, shared, global_in + batch_idx, sizeof(int) * ,block.size());
```
The cooperative_groups::memcpy_async API copies sizeof(int) * block.size() bytes from global memory starting at global_in + batch_idx to the shared data. This operation happens as-if performed by another thread, which synchronizes with the current thread’s call to cooperative_groups::wait after the copy has completed.

On devices with compute capability 8.0 or higher, memcpy_async transfers from global to shared memory can benefit from hardware acceleration, which avoids transfering the data through an intermediate register ( 数据传输不需要经过中间寄存器 )

code ref to the book
### 10.27.6 Performance Guidance for `memcpy_async`
For compute capability 8.x, the pipeline mechanism is shared among CUDA threads in the same CUDA warp. This sharing causes batches of memcpy_async to be entangled within a warp, which can impact performance under certain circumstances.

This section highlights the warp-entanglement effect on commit, wait, and arrive operations.
#### 10.27.6.1 Alignment
On devices with compute capability 8.0, the `cp.async` family of instructions allows copying data from global to shared memory asynchronously. These instructions support copying 4, 8, and 16 bytes at a time. If the size provided to memcpy_async is a multiple of 4, 8, or 16, and both pointers passed to memcpy_async are aligned to a 4, 8, or 16 alignment boundary, ( 需要传输的数据大小是4/8/16字节的倍数，且指针于4/8/16字节对齐，若没有对齐，行为未定义 ) then memcpy_async can be implemented using exclusively asynchronous memory operations. 

Additionally for achieving best performance when using memcpy_async API, an alignment of 128 Bytes for both shared memory and global memory is required. ( 与128字节对齐，性能最好 )

For pointers to values of types with an alignment requirement of 1 or 2, it is often not possible to prove that the pointers are always aligned to a higher alignment boundary. Determining whether the `cp. async` instructions can or cannot be used must be delayed until run-time. Performing such a runtime alignment check increases code-size and adds runtime overhead.

The cuda::aligned_size_t<size_t Align>(size_t size)Shape can be used to supply a proof that both pointers passed to memcpy_async are aligned to an Align alignment boundary and that size is a multiple of Align, by passing it as an argument where the memcpy_async APIs expect a Shape:
```
cuda::memcpy_async(group, dst, src, cuda::aligned_size_t<16>(N * block.size()), pipeline);
```
If the proof is incorrect, the behavior is undefined.
#### 7.27.6.2 Trivially copyable
On devices with compute capability 8.0, the cp.async family of instructions allows copying data from global to shared memory asynchronously. If the pointer types passed to memcpy_async do not point to TriviallyCopyable types, the copy constructor of each output element needs to be invoked, and these instructions cannot be used to accelerate memcpy_async. ( 自Compute capability 8.0开始，支持不是可以简单拷贝的拷贝数据类型 )
#### 7.27.6.3 Warp Entanglement - Commit
The sequence of `memcpy_async` batches is shared across the warp. ( warp内共享需要异步拷贝的 batches 序列 ) The commit operation is coalesced such that the sequence is incremented once for all converged threads that invoke the commit operation. ( 提交操作会被合并，也就是说所有warp内收敛的线程调用的提交操作只会让序列中增加一个新的需求 )  If the warp is fully converged, the sequence is incremented by one; if the warp is fully diverged, the sequence is incremented by 32. ( 如果warp完全收敛，则一个warp只会给序列贡献一个需求，如果warp完全发散，则一个warp会给序列贡献32个需求 )

- Let _PB_ be the warp-shared pipeline’s _actual_ sequence of batches.
    `PB = {BP0, BP1, BP2, …, BPL}`
- Let _TB_ be a thread’s _perceived_ sequence of batches, as if the sequence were only incremented by this thread’s invocation of the commit operation. ( TB为线程观察到的batch序列，从线程的角度看，它的commit操作只会让序列中batch数量加一 )
    `TB = {BT0, BT1, BT2, …, BTL}`
    The `pipeline::producer_commit()` return value is from the thread’s _perceived_ batch sequence. ( `pipeline::producer_commit()` 返回的值就是线程观察到的序列 )
- An index in a thread’s perceived sequence always aligns to an equal or larger index in the actual warp-shared sequence. The sequences are equal only when all commit operations are invoked from converged threads. ( 一个线程观察到的序列中的索引总是和一个比它更大的或和它相等的在真实的warp共享序列中的索引对齐，也就是说线程的观察有滞后性，线程观察到的序列和warp共享序列只有在所有的提交操作都由收敛线程完成才相等 )
    `BTn ≡ BPm` where `n <= m`

For example, when a warp is fully diverged:
- The warp-shared pipeline’s actual sequence would be: `PB = {0, 1, 2, 3, ..., 31}` (`PL=31`).
- The perceived sequence for each thread of this warp would be:
    - Thread 0: `TB = {0}` (`TL=0`)
    - Thread 1: `TB = {0}` (`TL=0`)
    - `…`
    - Thread 31: `TB = {0}` (`TL=0`)
( 例如完全发散的warp中，线程序列仅一个batch，warp序列则有32个 )
#### 7.27.6.4 Warp Entanglement - Wait
A CUDA thread invokes either `pipeline_consumer_wait_prior<N>()` or `pipeline::consumer_wait()` to wait for batches in the _perceived_ sequence `TB` to complete. ( 线程调用 `pipeline_consumer_wait_prior<N>()` 或 `pipeline::consumer_wait()` 来等待观察到的序列 `TB` 中的batch完成它们的操作 ) Note that `pipeline::consumer_wait()` is equivalent to `pipeline_consumer_wait_prior<N>()`, where `N = PL`. ( `pipeline_consumer_wait_prior<PL>()` 和 `pipeline::consumer_wait()` 等价，即只等待最旧的batch )

The `pipeline_consumer_wait_prior<N>()` function waits for batches in the _actual_ sequence at least up to and including `PL-N`. ( `pipeline_consumer_wait_prior<N>()` 函数等待实际序列中从最开始到 `PL-N` 的batch完成 ) Since `TL <= PL`, waiting for batch up to and including `PL-N` includes waiting for batch `TL-N`.  ( 因为 `TL <= PL` ，`PL-N >= TL-N` ，因此 batch `TL-N` 及之前的batch一定会被等待完成 ) Thus, when `TL < PL`, the thread will unintentionally wait for additional, more recent batches. ( 当 `TL < PL` ， `PL - N > TL - N` 线程还会比预想中多等一些batch )

In the extreme fully-diverged warp example above, each thread could wait for all 32 batches.
#### 7.27.6.5 Warp Entanglement - Arrive-On
Warp-divergence affects the number of times an `arrive_on(bar)` operation updates the barrier. If the invoking warp is fully converged, then the barrier is updated once. If the invoking warp is fully diverged, then 32 individual updates are applied to the barrier. ( warp离散影响 `arrive_on(bar` 操作更新barrier的次数，如果warp完全收敛，则barrir仅更新一次，如果完全离散，则会更新32次 )
#### 7.27.6.6 Keep Commit and Arrive-On Operations Converged
It is recommended that commit and arrive-on invocations are by converged threads:
- to not over-wait, by keeping threads’ perceived sequence of batches aligned with the actual sequence, and
- to minimize updates to the barrier object.

When code preceding these operations diverges threads, then the warp should be re-converged, via `__syncwarp` before invoking commit or arrive-on operations.
## 7.28 Asynchronous Data Copies using `cuda::pipeline`
CUDA provides the `cuda::pipeline` synchronization object to manage and overlap asynchronous data movement with computation. ( 将异步数据移动与计算重叠 )

The API documentation for `cuda::pipeline` is provided in the libcudacxx API. A pipeline object is a double-ended N stage queue with a head and a tail, and is used to process work in a first-in first-out (FIFO) order. ( 一个流水线对象是带头部和尾部的N阶段两端队列，它处理工作的顺序是先进先出 )The pipeline object has following member functions to manage the stages of the pipeline.

![[CUDA C++ Programming Guide-PipelineMemberFunction.png]]
`producer_acquire` : 在流水线内部队列中获得当前阶段
`producer_commit` : 将在 `producer_acquire` 调用之后发出的异步操作提交到之前获得的流水线阶段中
`consumer_wait` : 等待流水线最旧的阶段内的所有异步操作完成
`consumer_release` : 释放流水线的最旧的阶段，使其可以复用，既被释放的阶段可以继续被 `producer_acquire` 获得
### 10.28.1 Single-Stage Asynchronous Data Copies using  `cuda::pipeline`
In previous examples we showed how to use cooperative_groups and cuda::barrier to do asynchronous data transfers. In this section, we will use the cuda::pipeline API with a single stage to schedule asynchronous copies. And later we will expand this example to show multi staged overlapped compute and copy.

code ref to the book
### 10.28.2 Multi-Stage Asynchronous Data Copies using `cuda::pipeline`
In the previous examples with cooperative_groups::wait and cuda::barrier, the kernel threads immediately wait for the data transfer to shared memory to complete. This avoids data transfers from global memory into registers, but does not hide the latency of the memcpy_async operation by overlapping computation. ( 之间的单阶段 `memcpy_async` 没有对计算和加载存储进行重叠，不过利用到了 `memcpy_async` 本身的从global memory到shared memory的加速 )

For that we use the CUDA pipeline feature in the following example. It provides a mechanism for managing a sequence of memcpy_async batches, enabling CUDA kernels to overlap memory transfers with computation.

The following example implements a two-stage pipeline that overlaps data-transfer with computation. It:
- Initializes the pipeline shared state (more below)
- Kickstarts the pipeline by scheduling a `memcpy_async` for the first batch.
- Loops over all the batches: it schedules `memcpy_async` for the next batch, blocks all threads on the completion of the `memcpy_async` for the previous batch, and then overlaps the computation on the previous batch with the asynchronous copy of the memory for the next batch. ( 调度对下一个batch的 `memcpy_async` ，同时等待对当前batch的 `memcpy_async` 完成，并对其进行计算 )
- Finally, it drains the pipeline by performing the computation on the last batch.

Note that, for interoperability with `cuda::pipeline`, `cuda::memcpy_async` from the `cuda∕ pipeline` header is used here. ( 为了与流水线 `cuda::pipeline` 协调，我们使用来自 `cuda/pipeline` 中的 `cuda::memcpy_async` )

code ref to the book

A pipeline object is a double-ended queue with a head and a tail, and is used to process work in a first-in first-out (FIFO) order. Producer threads commit work to the pipeline’s head, while consumer threads pull work from the pipeline’s tail. ( 生产者线程向流水线头部提交工作，消费者线程从流水线尾部拉取工作 ) In the example above, all threads are both producer and consumer threads. The threads first commit `memcpy_async` operations to fetch the next batch while they wait on the previous batch of `memcpy_async` operations to complete. ( 在上一例中，所有的线程即是生产者也是消费者，线程首先提交 `memcpy_async` 操作以装载下一个batch的数据，同时，它们等待当前batch的数据到达并对其进行计算 )

-  Committing work to a pipeline stage involves:
    - Collectively acquiring the pipeline head from a set of producer threads using `pipeline.producer_acquire()`. ( 使用 `producer_acquire()` 获取流水线头部 )
    - Submitting `memcpy_async` operations to the pipeline head. 
    - Collectively commiting (advancing) the pipeline head using `pipeline.producer_commit()`.  ( 将 `memcpy_async` 提交到流水线头部，同时流水线头部向前进 )
-  Using a previously commited stage involves:
    - Collectively waiting for the stage to complete, e.g., using `pipeline.consumer_wait()` to wait on the tail (oldest) stage. ( 使用 `consumer_wait()` 等待流水线尾部阶段 )
    - Collectively releasing the stage using `pipeline.consumer_release()`. ( 使用 `consumer_release()` 释放流水线尾部的阶段 )
 `cuda::pipeline_shared_state<scope, count>` encapsulates the finite resources that allow a pipeline to process up to `count` concurrent stages. If all resources are in use, `pipeline.producer_acquire()` blocks producer threads until the resources of the next pipeline stage are released by consumer threads.

This example can be written in a more concise manner by merging the prolog and epilog of the loop with the loop itself as follows:

code ref to the book

The `pipeline<thread_scope_block>` primitive used above is very flexible, and supports two features that our examples above are not using: any arbitrary subset of threads in the block can participate in the pipeline, and from the threads that participate, any subsets can be producers, consumers, or both. In the following example, threads with an “even” thread rank are producers, while other threads are consumers: ( `pipline<thread_scope_block>` 还支持块内任意线程子集加入流水线，并且支持部分线程作为生产者，另一部分线程作为消费者 )

code ref to the book

There are some optimizations that `pipeline` performs, for example, when all threads are both producers and consumers, but in general, the cost of supporting all these features cannot be fully eliminated. For example, `pipeline` stores and uses a set of barriers in shared memory for synchronization, which is not really necessary if all threads in the block participate in the pipeline. ( `pipeline` 会在shared memory中储存一系列barrier用于同步，如果block内的所有线程都参与pipline的话，这些实际上是不必要的 )

For the particular case in which all threads in the block participate in the pipeline, we can do better than `pipeline<thread_scope_block>` by using a `pipeline<thread_scope_thread>` combined with `__syncthreads()`: ( 若block内的所有线程都参与pipeline，我们可以通过将 `pipeline<thread_scope_block>` 与 `__syncthreads()` 结合来进行优化 )

code ref to the book

If the `compute` operation only reads shared memory written to by other threads in the same warp as the current thread, `__syncwarp()` suffices. ( 如果 `compute` 仅需要访问同warp内的线程取来的数据，则 `__syncwarp()` 就足够了 )
### 10.28.3 Pipeline Interface
The complete API documentation for `cuda::memcpy_async` is provided in the [libcudacxx API](https://nvidia.github.io/libcudacxx) documentation along with some examples.

The `pipeline` interface requires
- at least CUDA 11.0,
- at least ISO C++ 2011 compatibility, e.g., to be compiled with `-std=c++11`, and
- `#include <cuda/pipeline>`.

For a C-like interface, when compiling without ISO C++ 2011 compatibility, see [Pipeline Primitives Interface](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-primitives-interface).
### 10.28.4 Pipeline Primitives Interface
Pipeline primitives are a C-like interface for `memcpy_async` functionality. ( 流水线原语即C风格的 `memcpy_async` 功能接口 )The pipeline primitives interface is available by including the `<cuda_pipeline.h>` header. When compiling without ISO C++ 2011 compatibility, include the `<cuda_pipeline_primitives.h>` header.
#### 10.28.4.1 `memcpy_asunc` Primitive
#### 10.28.4.2 Commit Primitive
#### 10.28.4.3 Wait Primitive
## 10.29 Asynchronous Data Copies using Tensor Memory Access (TMA)
Need Compute Capability 9.0
## 10.30 Profiler Counter Function
Each multiprocessor has a set of sixteen hardware counters that an application can increment with a single instruction by calling the `__prof_trigger()` function. ( 每个SM都有一系列一组16个的硬件计数器，应用程序可以通过调用 `__prof_trigger()` 函数使用一条指令增加硬件计数器 )
```cpp
void __prof_trigger(int counter);
```
increments by one per warp the per-multiprocessor hardware counter of index counter. Counters 8 to 15 are reserved and should not be used by applications.

The value of counters 0, 1, …, 7 can be obtained via nvprof by nvprof --events prof_trigger_0x where x is 0, 1, …, 7. All counters are reset before each kernel launch (note that when collecting counters, kernel launches are synchronous as mentioned in Concurrent Execution between Host and Device).
## 10.34 Formatted Output
Formatted output is only supported by devices of compute capability 2.x and higher. 
```cpp
int printf(const char *format[, arg, ...]);
```
prints formatted output from a kernel to a host-side output stream.

The in-kernel printf() function behaves in a similar way to the standard C-library printf() function, and the user is referred to the host system’s manual pages for a complete description of printf() behavior. In essence, the string passed in as format is output to a stream on the host, with substitutions made from the argument list wherever a format specifier is encountered. Supported format specifiers are listed below.

The printf() command is executed as any other device-side function: per-thread, and in the context of the calling thread. From a multi-threaded kernel, this means that a straightforward call to printf() will be executed by every thread, using that thread’s data as specified. Multiple versions of the output string will then appear at the host stream, once for each thread which encountered the printf(). ( 注意kernel中的 `printf()` 每个线程都会调用 )

Unlike the C-standard printf(), which returns the number of characters printed, CUDA’s printf() returns the number of arguments parsed. If no arguments follow the format string, 0 is returned. If the format string is NULL, -1 is returned. If an internal error occurs, -2 is returned.

the rest is temporarily passed
## 10.37 Execution Configuration
Any call to a `__global__` function must specify the execution configuration for that call. The execution configuration defines the dimension of the grid and blocks that will be used to execute the function on the device, as well as the associated stream (see CUDA Runtime for a description of streams)

The execution configuration is specified by inserting an expression of the form `<<< Dg, Db, Ns, S >>>` between the function name and the parenthesized argument list, where:
- `Dg` is of type `dim3` (see [dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3)) and specifies the dimension and size of the grid, such that `Dg.x * Dg.y * Dg.z` equals the number of blocks being launched;
- `Db` is of type `dim3` (see [dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3)) and specifies the dimension and size of each block, such that `Db.x * Db.y * Db.z` equals the number of threads per block;
- `Ns` is of type `size_t` and specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory; this dynamically allocated memory is used by any of the variables declared as an external array as mentioned in [__shared__](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared); `Ns` is an optional argument which defaults to 0;
- `S` is of type `cudaStream_t` and specifies the associated stream; `S` is an optional argument which defaults to 0.

The arguments to the execution configuration are evaluated before the actual function arguments.

the rest is passed
## 10.38 Launch Bounds
As discussed in detail in Multiprocessor Level, the fewer registers a kernel uses, the more threads and thread blocks are likely to reside on a multiprocessor, which can improve performance.

Therefore, the compiler uses heuristics to minimize register usage while keeping register spilling (see Device Memory Accesses) and instruction count to a minimum.

the rest is passed
## 10.40 `#pragma unroll`
By default, the compiler unrolls small loops with a known trip count. The `#pragma unroll` directive however can be used to control unrolling of any given loop. It must be placed immediately before the loop and only applies to that loop. ( 该杂注放在循环之间，编译器则展开该循环 )

It is optionally followed by an integral constant expression (ICE). If the ICE is absent, the loop will be completely unrolled if its trip count is constant. If the ICE evaluates to 1, the compiler will not unroll the loop. The pragma will be ignored if the ICE evaluates to a non-positive integer or to an integer greater than the maximum value representable by the `int` data type.
# 8 Cooperative Groups
## 11.1. Introduction
Cooperative Groups is an extension to the CUDA programming model, introduced in CUDA 9, for organizing groups of communicating threads.

Historically, the CUDA programming model has provided a single, simple construct for synchronizing cooperating threads: a barrier across all threads of a thread block, as implemented with the `__syncthreads()` intrinsic function.
## 11.3 Programming Model Concept
The Cooperative Groups programming model describes synchronization patterns both within and across CUDA thread blocks. It provides both the means for applications to define their own groups of threads, and the interfaces to synchronize them.

To write efficient code, its best to use specialized groups (going generic loses a lot of compile time optimizations), and pass these group objects by reference to functions that intend to use these threads in some cooperative fashion.

To use Cooperative Groups, include the header file:
ref to book
and use the Cooperative Groups namespace:
ref to book
## 11.4 Group Types
### 11.4.1 Implicit Groups
Implicit groups represent the launch configuration of the kernel. Regardless of how your kernel is written, it always has a set number of threads, blocks and block dimensions, a single grid and grid dimensions.

Although you can create an implicit group anywhere in the code, it is dangerous to do so. Creating a handle for an implicit group is a collective operation—all threads in the group must participate. ( 为隐式群组创建句柄是一个集体操作，组内所有线程都需要参与 )If the group was created in a conditional branch that not all threads reach, this can lead to deadlocks or data corruption. For this reason, it is recommended that you create a handle for the implicit group upfront (as early as possible, before any branching has occurred) and use that handle throughout the kernel. Group handles must be initialized at declaration time (there is no default constructor) for the same reason and copy-constructing them is discouraged.
#### 11.4.1.1 Thread Block Group
Any CUDA programmer is already familiar with a certain group of threads: the thread block. The Cooperative Groups extension introduces a new datatype, `thread_block`, to explicitly represent this concept within the kernel. 
`class thread_block`; 
Constructed via:
```
thread_block g = this_thread_block();
```

public member functions ref to the book
#### 11.4.1.3 Grid Group
This group object represents all the threads launched in a single grid. APIs other than sync() are available at all times, but to be able to synchronize across the grid, you need to use the cooperative launch API
`class grid_group; `
Constructed via:
```
grid_group g = this_grid();
```
## 11.6 Group Collectives
Cooperative Groups library provides a set of collective operations that can be performed by a group of threads. These operations require participation of all threads in the specified group in order to complete the operation. All threads in the group need to pass the same values for corresponding arguments to each collective call, unless different values are explicitly allowed in the argument description. Otherwise the behavior of the call is undefined.
### 11.6.1 Synchronization
#### 11.6.1.2 `sync`
```cpp
static void T::sync();  // static memeber function of class T

template <typename T> 
void sync(T& group); // a template function which receives paramenters of type T
```
`sync` synchronizes the threads named in the group. ( 同步组内所有线程 ) Group type T can be any of the existing group types, as all of them support synchronization. Its available as a member function in every group type or as a free function taking a group as parameter. If the group is a `grid_group` or a `multi_grid_group` the kernel must have been launched using the appropriate cooperative launch APIs. Equivalent to `T.barrier_wait(T.barrier_arrive())`.
### 11.6.2 Data Transfer
#### 11.6.2.1 `memecpy_async`
`memcpy_async` is a group-wide collective memcpy that utilizes hardware accelerated support for nonblocking memory transactions from global to shared memory. ( 组级别的集体函数，利用了硬件对非阻塞式从全局存储到共享存储的存储事务的支持 ) Given a set of threads named in the group, `memcpy_async` will move specified amount of bytes or elements of the input type through a single pipeline stage. ( `memcpy_async` 会通过一个流水线阶段移动数据 ) Additionally for achieving best performance when using the `memcpy_async` API, an alignment of 16 bytes for both shared memory and global memory is required. It is important to note that while this is a memcpy in the general case, it is only asynchronous if the source is global memory and the destination is shared memory and both can be addressed with 16, 8, or 4 byte alignments. ( 该函数仅在源是全局存储且目标是共享存储，同时双方的地址是4/8/16字节对齐的情况下，才是异步的 ) Asynchronously copied data should only be read following a call to `wait` or `wait_prior` which signals that the corresponding stage has completed moving data to shared memory. ( 异步拷贝的数据只有在 `wait` 或 `wati_prior` 调用之后才可以被读取，`wait` 和 `wait_prior` 调用会在数据移动完成之后发送信号 )

Having to wait on all outstanding requests can lose some flexibility (but gain simplicity). In order to efficiently overlap data transfer and execution, its important to be able to kick off an N+1 `memcpy_async` request while waiting on and operating on request N. ( 要让数据传输和执行进行交叉，我们需要发起第N+1个 `memcpy_async` 请求的同时，处理第N个请求返回的数据 )To do so, use `memcpy_async` and wait on it using the collective stage-based `wait_prior` API. ( 为此，我们需要使用 `memcpy_async` 进行异步拷贝，然后使用 `wait_prior` 等待拷贝完成 ) See wait and wait_prior for more details.

`memcpy_async` usage ref to book

Codegen Requirements: Compute Capability 5.0 minimum, Compute Capability 8.0 for asynchronicity, C++11
`cooperative_groups∕memcpy_async.h` header needs to be included
#### 11.6.2.2 `wait` and `wait_prior`
`wait` and `wait_prior` collectives allow to wait for `memcpy_async` copies to complete. `wait` blocks calling threads until all previous copies are done. `wait_prior` allows that the latest NumStages are still not done and waits for all the previous requests. So with N total copies requested, it waits until the first N-NumStages are done and the last NumStages might still be in progress. Both wait and wait_prior will synchronize the named group.

Codegen Requirements: Compute Capability 5.0 minimum, Compute Capability 8.0 for asynchronicity, C++11
`cooperative_groups∕memcpy_async.h` header needs to be included

`wait` and `wait_prior` usage ref to book
# 9 CUDA Dynamic Parallelism
## 12.1 Introduction
### 12.1.1 Overview
...
# 16 Compute Capabilities
## 19.1 Feature Availability
A compute feature is introduced with a compute architecture with the intention that the feature will be available on all subsequent architectures. ( 计算特性在引入之后一般会一直保持在之后的架构可用 )

Highly specialized compute features ( 高度专业化的计算特征 ) that are introduced with an architecture may not be guaranteed to be available on all subsequent compute capabilities. These features target acceleration of specialized operations which are not intended for all classes of compute capabilities (denoted by the compute capability’s minor number) or are likely to significantly change on future generations (denoted by the compute capability’s major number).

**Compute Capability #.#:** The predominant set of compute features that are introduced with the intent to be available for subsequent compute architectures. These features and their availability are summarized in Table 20.

**Compute Capability #.#a:** A small and highly specialized set of features that are introduced to accelerate specialized operations, which are not guaranteed to be available or might change significantly on subsequent compute architecture. These features are summarized in the respective “Compute Capability #.#”” subsection.

Compilation of device code targets a particular compute capability. A feature which appears in device code must be available for the targeted compute capability. For example: ( 编译时，根据设备代码使用的计算特征选择目标计算能力 )
 -  The `compute_90` compilation target allows use of Compute Capability 9.0 features but does not allow use of Compute Capability 9.0a features.
 -  The `compute_90a` compilation target allows use of the complete set of compute device features, both 9.0a features and 9.0 features.
## 19.4 Compute Capability 5.x
### 19.4.2 Global Memory
Global memory accesses are always cached in L2.

Data that is read-only for the entire lifetime of the kernel ( 在整个内核生命周期都保持只读 ) can also be cached in the unified L1/texture cache described in the previous section by reading it using the `__ldg()` function (see Read-Only Data Cache Load Function).When the compiler detects that the read-only condition is satisfied for some data, it will use `__ldg()` to read it. ( 当编译器认为数据满足只读条件，它就会使用 `__ldg()` 去读取它 ) The compiler might not always be able to detect that the readonly condition is satisfied for some data. Marking pointers used for loading such data with both the `const` and `__restrict__` qualifiers increases the likelihood that the compiler will detect the readonly condition. ( 用 `const` 和 `__restrict__` 修饰只读数据提高编译器优化其读取的概率 )

the rest is passed
### 19.4.3 Shared Memory
Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle. ( 每个bank32位长，其带宽是32位每周期 )

A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word (even though the two addresses fall in the same bank). In that case, for read accesses, the word is broadcast to the requesting threads and for write accesses, each address is written by only one of the threads (which thread performs the write is undefined). 

Figure 22 shows some examples of strided access.
![[CUDA C++ Programming Guide-Figure35.png]]


Figure 23 shows some examples of memory read accesses that involve the broadcast mechanism.
![[CUDA C++ Programming Guide-Figure36.png]]

## 19.7 Compute Capability 8.x
### 19.7.1 Architecture
A Streaming Multiprocessor (SM) consists of:
- 64 FP32 cores for single-precision arithmetic operations in devices of compute capability 8.0 and 128 FP32 cores in devices of compute capability 8.6, 8.7 and 8.9 
- 32 FP64 cores for double-precision arithmetic operations in devices of compute capability 8.0 and 2 FP64 cores in devices of compute capability 8.6, 8.7 and 8.9
- 64 INT32 cores for integer math,
- 4 mixed-precision Third-Generation Tensor Cores supporting half-precision (`fp16`), `__nv_bfloat16`, `tf32`, sub-byte and double precision (`fp64`) matrix arithmetic for compute capabilities 8.0, 8.6 and 8.7 (see Warp matrix functions for details), ( 4个混合精度第三代Tensor core)
- 4 mixed-precision Fourth-Generation Tensor Cores supporting `fp8`, `fp16`, `__nv_bfloat16`, `tf32`, sub-byte and `fp64` for compute capability 8.9 (see Warp matrix functions for details) ( 4个混合精度第四代Tensor core，相较于第三代多了对 `fp8` 的支持)
- 16 special function units for single-precision floating-point transcendental functions ( 超越函数 )
- 4 warp schedulers.

An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any. ( 每次的指令发射时间，一个warp调度器发送一条指令给一个warp )

An SM has:
-  a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
-  a unified data cache and shared memory with a total size of 192 KB for devices of compute capability 8.0 and 8.7 (1.5x Volta’s 128 KB capacity) and 128 KB for devices of compute capabilities 8.6 and 8.9.( A100 shared memory 大小为192KB )

Shared memory is partitioned out of the unified data cache, and can be configured to various sizes (see Shared Memory section). The remaining data cache serves as an L1 cache and is also used by the texture unit that implements the various addressing and data filtering modes mentioned in Texture and Surface Memory ( shared memory和L1 cache/Texture memory共用空间)
### 19.7.2 Global Memory
Global memory behaves the same way as for devices of compute capability 5.x
### 19.7.3 Shared Memory
Similar to the Volta architecture, the amount of the unified data cache reserved for shared memory is configurable on a per kernel basis. For the NVIDIA Ampere GPU architecture, the unified data cache has a size of 192 KB for devices of compute capability 8.0 and 8.7 and 128 KB for devices of compute capabilities 8.6 and 8.9. The shared memory capacity can be set to 0, 8, 16, 32, 64, 100, 132 or 164 KB ( shared memory最大164KB ) for devices of compute capability 8.0 and 8.7, and to 0, 8, 16, 32, 64 or 100 KB for devices of compute capabilities 8.6 and 8.9.

An application can set the carveout, i.e., the preferred shared memory capacity, with the `cudaFuncSetAttribute()`.
```
cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
```
The API can specify the carveout either as an integer percentage of the maximum supported shared memory capacity of 164 KB for devices of compute capability 8.0 and 8.7 and 100 KB for devices of compute capabilities 8.6 and 8.9 respectively, or as one of the following values: {`cudaSharedmemCarveoutDefault`, `cudaSharedmemCarveoutMaxL1`, or `cudaSharedmemCarveoutMaxShared`. When using a percentage, the carveout is rounded up to the nearest supported shared memory capacity. For example, for devices of compute capability 8.0, 50% will map to a 100 KB carveout instead of an 82 KB one. Setting the `cudaFuncAttributePreferredSharedMemoryCarveout` is considered a hint by the driver; the driver may choose a different configuration, if needed.  ( 每个内核调用需要多少shared memory可以由API设定 )

Devices of compute capability 8.0 and 8.7 allow a single thread block to address up to 163 KB of shared memory, ( 单个线程块最多可以寻址163KB的shared memory ) while devices of compute capabilities 8.6 and 8.9 allow up to 99 KB of shared memory. Kernels relying on shared memory allocations over 48 KB per block are architecture-specific, and must use dynamic shared memory rather than statically sized shared memory arrays. These kernels require an explicit opt-in by using `cudaFuncSetAttribute`() to set the `cudaFuncAttributeMaxDynamicSharedMemorySize`; see Shared Memory for the Volta architecture. ( 对于每个线程块需要使用超过48KB的shared memory的内核，需要设定属性 ) Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory partition available per SM. The 1 KB of shared memory not made available to a thread block is reserved for system use. ( 1KB不可寻址的shared memory由系统使用)