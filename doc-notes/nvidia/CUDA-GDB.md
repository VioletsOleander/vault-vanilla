---
version: "12.9"
---
# 1 Introduction
This document introduces CUDA-GDB, the NVIDIA® CUDA® debugger for Linux and QNX targets.

## 1.1 What is CUDA-GDB
CUDA-GDB is the NVIDIA tool for debugging CUDA applications running on Linux and QNX. CUDA-GDB is an extension to GDB, the GNU Project debugger. The tool provides developers with a mechanism for debugging CUDA applications running on actual hardware. This enables developers to debug applications without the potential variations introduced by simulation and emulation environments.

## 2.1 Supported Features
CUDA-GDB is designed to present the user with a seamless debugging environment that allows simultaneous debugging of both GPU and CPU code within the same application. Just as programming in CUDA C is an extension to C programming, debugging with CUDA-GDB is a natural extension to debugging with GDB. The existing GDB debugging features are inherently present for debugging the host code, and additional features have been provided to support debugging CUDA device code.
>  CUDA-GDB 中，GDB 的 debugging feature 完全适用于 host code 的调试，CUDA device code 的调试则具有额外的功能

CUDA-GDB supports debugging C/C++ and Fortran CUDA applications. Fortran debugging support is limited to 64-bit Linux operating system.

CUDA-GDB allows the user to set breakpoints, to single-step CUDA applications, and also to inspect and modify the memory and variables of any given thread running on the hardware.
> CUDA-GDB 允许为 CUDA 应用设置断点、单步调试，以及审查和修改硬件中任意给定线程的存储和变量 

CUDA-GDB supports debugging all CUDA applications, whether they use the CUDA driver API, the CUDA runtime API, or both.
>  无论是使用 CUDA driver API 还是 CUDA runtime API 的 CUDA 应用都可以用 CUDA-GDB 调试

CUDA-GDB supports debugging kernels that have been compiled for specific CUDA architectures, such as `sm_75` or `sm_80`, but also supports debugging kernels compiled at runtime, referred to as just-in-time compilation, or JIT compilation for short.
>  CUDA-GDB 即支持针对特定 CUDA 架构编译的 kernel，也支持在运行时 JIT 编译的 kernel

## 1.3. About This Document
This document is the main documentation for CUDA-GDB and is organized more as a user manual than a reference manual. The rest of the document will describe how to install and use CUDA-GDB to debug CUDA kernels and how to use the new CUDA commands that have been added to GDB. Some walk-through examples are also provided. It is assumed that the user already knows the basic GDB commands used to debug host applications.

# 2. Release Notes

# 3 Getting Started
The CUDA toolkit can be installed by following instructions in the [Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/).

Further steps should be taken to set up the debugger environment, build the application, and run the debugger.

## 3.1 Setting Up the Debugger Environment
### 3.1.1 Temporary Directory
By default, CUDA-GDB uses `/tmp` as the directory to store temporary files. To select a different directory, set the `$TMPDIR` environment variable.

Note
The user must have write and execute permission to the temporary directory used by CUDA-GDB. Otherwise, the debugger will fail with an internal error.

Note
The value of `$TMPDIR` must be the same in the environment of the application and CUDA-GDB. If they do not match, CUDA-GDB will fail to attach onto the application process.

Note
Since `/tmp` folder does not exist on Android device, the `$TMPDIR` environment variable must be set and point to a user-writeable folder before launching cuda-gdb.

## 3.2 Compiling the Application
### 3.2.1 Debug Compilation
Using this line to compile the CUDA application `foo.cu`
- forces `-O0` compilation, with the exception of very limited dead-code eliminations and register-spilling optimizations. ( 强制使用 `-O0` 编译，仅在dead code消除和register spill方面有一点优化  )
- makes the compiler include debug information in the executable ( 在可执行文件中包含debug信息 )

NVCC, the NVIDIA CUDA compiler driver, provides a mechanism for generating the debugging information necessary for CUDA-GDB to work properly. The `-g -G` option pair must be passed to NVCC when an application is compiled for ease of debugging with CUDA-GDB; for example,
>  要进行 debug，必须在编译时向 NVCC 传递 `-g -G`

```
nvcc -g -G foo.cu -o foo
```

Using this line to compile the CUDA application `foo.cu`

- forces `-O0` compilation, with the exception of very limited dead-code eliminations and register-spilling optimizations.
- makes the compiler include debug information in the executable

>  传递 `-g -G` 编译时
>  - NVCC 将使用 `-O0` 编译，仅执行有限的死代码消除和寄存器溢出优化
>  - NVCC 会将 debug 信息包含到可执行文件中

Note
Enabling the `-G` option increases the binary size by including debug information and reduces performance due to the absence of compiler optimizations.

### 3.2.2 Compilation With Linenumber Information
Several enhancements were made to cuda-gdb’s support for debugging programs compiled with `-lineinfo` but not with `-G`. This is intended primarily for debugging programs built with OptiX/RTCore.

Note that `-lineinfo` can be used when trying to debug optimized code.  In this case, debugger stepping and breakpoint behavior may appear somewhat erratic.
>  `-lineinfo` 可以用于debug优化的代码

- The PC may jump forward and backward unexpectedly while stepping. ( 在步入的时候 PC 可能会胡乱跳跃 )
- The user may step into code that has no linenumber information, leading to an inability to determine which source-file/linenumber the code at the PC belongs to.
- Breakpoints may break on a different line than they were originally set on.

When debugging OptiX/RTCore code, the following should be kept in mind:
- NVIDIA internal code cannot be debugged or examined by the user.
- OptiX/RTCode debugging is limited to `-lineinfo`, and building this code with full debug infomation (`-G`) is not supported.
- OptiX/RTCode code is highly optimized, and as such the notes above about debugging optimized code apply.

### 3.2.3 Compiling For Specific GPU architectures
By default, the compiler will only generate code for the compute_52 PTX and sm_52 cubins. ( 编译器的默认目标是 compute_52的PTX和 sm_52的 cubin ) For later GPUs, the kernels are recompiled at runtime from the PTX for the architecture of the target GPU(s). Compiling for a specific virtual architecture guarantees that the application will work for any GPU architecture after that, for a trade-off in performance. This is done for forward-compatibility.

It is highly recommended to compile the application once and for all for the GPU architectures targeted by the application, and to generate the PTX code for the latest virtual architecture for forward compatibility. ( 推荐为应用的各个目标架构都编译一次，并针对最新的虚拟架构编译一次PTX码 )

A GPU architecture is defined by its compute capability. The list of GPUs and their respective compute capability, see [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). The same application can be compiled for multiple GPU architectures. Use the `-gencode` compilation option to dictate which GPU architecture to compile for. The option can be specified multiple times. ( 代码可以为多个GPU架构编译，使用 `-gencode` 选项表明编译的目标架构，`-gencode` 选项可以使用多次 )

For instance, to compile an application for a GPU with compute capability 7.0, add the following flag to the compilation command:
```
-gencode arch=compute_70,code=sm_70
```

To compile PTX code for any future architecture past the compute capability 7.0, add the following flag to the compilation command:
```
-gencode arch=compute_70,code=compute_70
```

For additional information, please consult the compiler documentation at [https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#extended-notation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#extended-notation)
## 3.3 Using the Debugger
CUDA-GDB can be used in the following system configurations:
### 3.3.3 Remote Debugging
There are multiple methods to remote debug an application with CUDA-GDB. In addition to using SSH or VNC from the host system to connect to the target system, it is also possible to use the `target remote` GDB feature. Using this option, the local `cuda-gdb` (client) connects to the `cuda-gdbserver` process (the server) running on the target system. ( 本地的 `cuda-gdb` 连接到运行于目标系统的 `cuda-gdbserver` ) This option is supported with a Linux client and a Linux or QNX server.

Setting remote debugging that way is a 2-step process:

**Launch the cuda-gdbserver on the remote host** ( 在远程主机启动 cuda-gdbserver )

cuda-gdbserver can be launched on the remote host in different operation modes.

Option 1: Launch a new application in debug mode. ( 在debug模式启动一个新的应用 )
To launch a new application in debug mode, invoke cuda-gdb server as follows: 
```
$ cuda-gdbserver :1234 app_invocation
``` 
Where `1234` is the TCP port number that `cuda-gdbserver` will listen to for incoming connections from `cuda-gdb`, and `app-invocation` is the invocation command to launch the application, arguments included.

Option 2: Attach `cuda-gdbserver` to the running process  ( 将 `cuda-gdbserver` 关联到一个运行中的进程 )
To attach cuda-gdbserver to an already running process, the `--attach` option followed by process identification number (PID) must be used:
```
$ cuda-gdbserver :1234 --attach 5678
```
Where `1234` is the TCP port number and `5678` is process identifier of the application cuda-gdbserver must be attached to.

Attaching to an already running process is not supported on QNX platforms.
    

**Launch cuda-gdb on the client** ( 在客户端启动 cuda-gdb )

Configure `cuda-gdb` to connect to the remote target using either:
```
(cuda-gdb) target remote
```

or
```
(cuda-gdb) target extended-remote
```

It is recommended to use `set sysroot` command if libraries installed on the debug target might differ from the ones installed on the debug host. ( 本地和远程下载的库不同 ) For example, cuda-gdb could be configured to connect to remote target as follows:
```
(cuda-gdb) set sysroot remote://
(cuda-gdb) target remote 192.168.0.2:1234
```

Where `192.168.0.2` is the IP address or domain name of the remote target, and `1234` is the TCP port previously opened by `cuda-gdbserver`.
# 4 CUDA-GDB Extensions
## 4.1 Command Naming Convention
The existing GDB commands are unchanged. Every new CUDA command or option is prefixed with the CUDA keyword. ( 现存的GDB命令没有改变，cuda命令都有CUDA关键词前缀 ) As much as possible, CUDA-GDB command names will be similar to the equivalent GDB commands used for debugging host code. For instance, the GDB command to display the host threads and switch to host thread 1 are, respectively:
```
(cuda-gdb) info threads
(cuda-gdb) thread 1
```

To display the CUDA threads and switch to cuda thread 1, the user only has to type:
```
(cuda-gdb) info cuda threads
(cuda-gdb) cuda thread 1
```
## 4.2 Getting Help
As with GDB commands, the built-in help for the CUDA commands is accessible from the `cuda-gdb` command line by using the help command:
```
(cuda-gdb) help cuda name_of_the_cuda_command
(cuda-gdb) help set cuda name_of_the_cuda_option
(cuda-gdb) help info cuda name_of_the_info_cuda_command
```
Moreover, all the CUDA commands can be auto-completed by pressing the TAB key, as with any other GDB command.

CUDA commands can also be queried using the `apropos` command.
## 4.3 Initialization File
The initialization file for CUDA-GDB is named `.cuda-gdbinit` and follows the same rules as the standard `.gdbinit` file used by GDB. The initialization file may contain any CUDA-GDB command. Those commands will be processed in order when CUDA-GDB is launched.
## 4.5 GPU core dump support
There are two ways to configure the core dump ( 核心转储 ) options for CUDA applications. Environment variables set in the application environment or programmatically from the application with the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__COREDUMP.html#group__CUDA__COREDUMP/).

**Compilation for GPU core dump generation**

GPU core dumps will be generated regardless of compilation flags used to generate the GPU application.  ( 不论编译标志符合，GPU核心转储都可以被生成 ) For the best debugging experience, it is recommended to compile the application with the `-g -G` or the `-lineinfo` option with NVCC. See [Compiling the Application](https://docs.nvidia.com/cuda/cuda-gdb/index.html#compiling-the-application) for more information on passing compilation flags for debugging.

**Enabling GPU core dump generation on exception with environment variables**

Set the `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` environment variable to `1` in order to enable generating a GPU core dump when a GPU exception is encountered. ( 令GPU遭遇异常时可以生成核心转储文件 ) This option is disabled by default.

Set the `CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION` environment variable to `0` in order to disable generating a CPU core dump when a GPU exception is encountered. This option is enabled by default when GPU core dump generation is enabled.

Set the `CUDA_ENABLE_LIGHTWEIGHT_COREDUMP` environment variable to `1` in order to enable generating lightweight corefiles ( 轻量核心文件 ) instead of full corefiles. When enabled, GPU core dumps will not contain the memory dumps (local, shared, global) of the application. ( 不会包含应用的存储转储，包括local、shared、global ) This option is disabled by default.

**Controlling behavior of GPU core dump generation**

The `CUDA_COREDUMP_GENERATION_FLAGS` environment variable can be used when generating GPU core dumps to deviate from default generation behavior. Multiple flags can be provided to this environment variable and are delimited by `,`. These flags can be used to accomplish tasks such as reducing the size of the generated GPU core dump or other desired behaviors that deviate from the defaults. The table below lists each flag and the behavior when present.

ref to doc

**Limitations and notes for core dump generation**

The following limitations apply to core dump support:
- For Windows WDDM, GPU core dump is only supported on a GPU with compute capability 6.0 or higher. Windows TCC supports GPU core dump on all supported compute capabilities.
- GPU core dump is unsupported for the Windows Subsystem for Linux on GPUs running in SLI mode. Multi-GPU setups are supported, but SLI mode cannot be enabled in the Driver Control Panel.
- GPU core dump is supported for the Windows Subsystem for Linux only when the [hardware scheduling mode](https://devblogs.microsoft.com/directx/hardware-accelerated-gpu-scheduling/) is enabled.
- Generating a CPU core dump with `CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION` is currently unsupported on the QNX platform.
- GPU core dump is unsupported for the NVIDIA CMP product line.
- Per-context core dump can only be enabled on a GPU with compute capability 6.0 or higher. GPUs with compute capability less than 6.0 will return `CUDA_ERROR_NOT_SUPPORTED` when using the Coredump Attributes Control API.
- If an MPS client triggers a core dump, every other client running on the same MPS server will fault. The indirectly faulting clients will also generate a core dump if they have core dump generation enabled.
- GPU core dump is unsupported when other developer tools, including CUDA-GDB, are interacting with the application. Unless explicitly documented as a supported use case (e.g `generate-cuda-core-file` command). ( 其他开发工具和应用交互时，GPU core dump是不支持的 )
- When generating a coredump on exception, if the kernel exits before the exception has been recognized it may result in failure to generate the corefile. See the note in [GPU Error Reporting](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-error-reporting) for strategies on how to work around this issue.

**Naming of GPU core dump files**
By default, a GPU core dump is created in the current working directory. It is named `core_TIME_HOSTNAME_PID.nvcudmp` where `TIME` is the number of seconds since the Epoch, `HOSTNAME` is the host name of the machine running the CUDA application and `PID` is the process identifier of the CUDA application.

**Displaying core dump generation progress**
By default, when an application crashes and generates a GPU core dump, the application may appear to be unresponsive or frozen until fully generated.

Set the `CUDA_COREDUMP_SHOW_PROGRESS` environment variable to `1` in order to print core dump generation progress messages to `stderr`. This can be used to determine how far along the coredump generation is:
```
coredump: SM 1/14 has finished state collection
coredump: SM 2/14 has finished state collection
coredump: SM 3/14 has finished state collection
coredump: SM 4/14 has finished state collection
coredump: SM 5/14 has finished state collection
coredump: SM 6/14 has finished state collection
coredump: SM 7/14 has finished state collection
coredump: SM 8/14 has finished state collection
coredump: SM 9/14 has finished state collection
coredump: SM 10/14 has finished state collection
coredump: SM 11/14 has finished state collection
coredump: SM 12/14 has finished state collection
coredump: SM 13/14 has finished state collection
coredump: SM 14/14 has finished state collection
coredump: Device 1/1 has finished state collection
coredump: Calculating ELF file layout
coredump: ELF file layout calculated
coredump: Writing ELF file to core_TIME_HOSTNAME_PID.nvcudmp
coredump: Writing out global memory (1073741824 bytes)
coredump: 5%...
coredump: 10%...
coredump: 15%...
coredump: 20%...
coredump: 25%...
coredump: 30%...
coredump: 35%...
coredump: 40%...
coredump: 45%...
coredump: 50%...
coredump: 55%...
coredump: 60%...
coredump: 65%...
coredump: 70%...
coredump: 75%...
coredump: 80%...
coredump: 85%...
coredump: 90%...
coredump: 95%...
coredump: 100%...
coredump: Writing out device table
coredump: Finalizing
coredump: All done
```

**Inspecting GPU and GPU+CPU core dumps in cuda-gdb**
Use the following command to load the GPU core dump into the debugger ( 将GPU core dump装载至debugger )
- `(cuda-gdb) target cudacore core.cuda.localhost.1234`
    This will open the core dump file and print the exception encountered during program execution. ( 打开core dump文件，并打印程序执行时的异常 ) Then, issue standard cuda-gdb commands to further investigate application state on the device at the moment it was aborted.

Use the following command to load CPU and GPU core dumps into the debugger ( 装载CPU和GPU core dump文件 )
- `(cuda-gdb) target core core.cpu core.cuda`
    This will open the core dump file and print the exception encountered during program execution. Then, issue standard cuda-gdb commands to further investigate application state on the host and the device at the moment it was aborted.

Coredump inspection does not require that a GPU be installed on the system
# 5 Kernel Focus
A CUDA application may be running several host threads and many device threads. To simplify the visualization of information about the state of application, commands are applied to the entity in focus. ( 命令会被应用于当前聚焦的实体 )

When the focus is set to a host thread, the commands will apply only to that host thread (unless the application is fully resumed, for instance). On the device side, the focus is always set to the lowest granularity level–the device thread.
## 5.1 Software Coordinates vs. Hardware Coordinates
A device thread belongs to a block, which in turn belongs to a kernel. Thread, block, and kernel are the software coordinates of the focus. ( 线程、块、内核是当前焦点的软件坐标 ) A device thread runs on a lane. A lane belongs to a warp, which belongs to an SM, which in turn belongs to a device. Lane, warp, SM, and device are the hardware coordinates of the focus. ( 通道、线程束、SM、设备是当前焦点的硬件坐标 ) Software and hardware coordinates can be used interchangeably and simultaneously as long as they remain coherent. ( 二者保持一致时就可以同时并且交换使用 )

Another software coordinate is sometimes used: the grid. ( grid也是软件坐标 ) The difference between a grid and a kernel is the scope. The grid ID is unique per GPU whereas the kernel ID is unique across all GPUs. ( grid ID在每个GPU上是独一无二的，kernel ID在所有的GPU上是独一无二的 ) Therefore there is a 1:1 mapping between a kernel and a (grid,device) tuple. ( 一个kernel映射到一个 grid, device 元组，即一个kernel运行于一个设备的一个grid上 )

If software preemption is enabled (`set cuda software_preemption on`), hardware coordinates corresponding to a device thread are likely to change upon resuming execution on the device. However, software coordinates will remain intact and will not change for the lifetime of the device thread.
## 5.2 Current Focus
To inspect the current focus, use the cuda command followed by the coordinates of interest: ( 使用cuda命令查看当前焦点 )
```
(cuda-gdb) cuda device sm warp lane block thread
block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0
(cuda-gdb) cuda kernel block thread
kernel 1, block (0,0,0), thread (0,0,0)
(cuda-gdb) cuda kernel
kernel 1
```
## 5.3 Switching Focus
To switch the current focus, use the cuda command followed by the coordinates to be changed: ( 使用cuda命令切换焦点 )
```
(cuda-gdb) cuda device 0 sm 1 warp 2 lane 3
[Switching focus to CUDA kernel 1, grid 2, block (8,0,0), thread
(67,0,0), device 0, sm 1, warp 2, lane 3]
374 int totalThreads = gridDim.x * blockDim.x;
```
If the specified focus is not fully defined by the command, the debugger will assume that the omitted coordinates are set to the coordinates in the current focus, including the subcoordinates of the block and thread. ( 未指定全时，debugger将假定略去的坐标与当前焦点的一致 )
```
(cuda-gdb) cuda thread (15)
[Switching focus to CUDA kernel 1, grid 2, block (8,0,0), thread
(15,0,0), device 0, sm 1, warp 0, lane 15]
374 int totalThreads = gridDim.x * blockDim.x;
```
The parentheses for the block and thread arguments are optional.
```
(cuda-gdb) cuda block 1 thread 3
[Switching focus to CUDA kernel 1, grid 2, block (1,0,0), thread (3,0,0),
device 0, sm 3, warp 0, lane 3]
374 int totalThreads = gridDim.x * blockDim.
```
# 6 Program Execution
Applications are launched the same way in CUDA-GDB as they are with GDB by using the run command. This chapter describes how to interrupt and single-step CUDA applications
## 6.1 Interrupting the Application
If the CUDA application appears to be hanging or stuck in an infinite loop, it is possible to manually interrupt the application by pressing CTRL+C. When the signal is received, the GPUs are suspended and the `cuda-gdb` prompt will appear.  ( 在debugger中可以ctrl+c中断程序并开始审查啊 )

At that point, the program can be inspected, modified, single-stepped, resumed, or terminated at the user’s discretion.

This feature is limited to applications running within the debugger. It is not possible to break into and debug applications that have been launched outside the debugger.
## 6.2 Single Stepping
Single-stepping device code is supported. However, unlike host code single-stepping, device code single-stepping works at the warp level. ( 设备码的单步是warp级别的 ) This means that single-stepping a device kernel advances all the active threads in the warp currently in focus. ( 即设备内核的单步调试会让同一warp内的所有线程前进 ) The divergent threads in the warp are not single-stepped. ( 当然divergent的线程不会前进 ) When the CUDA thread in focus becomes divergent, behavior depends on the value of `set cuda step_divergent_lanes`. When on (default), the warp in focus will be continuously single-stepped until the CUDA thread in focus becomes active. ( 默认设置下，若focus于divergent的线程，单步调试会让warp一直走到当前线程活跃为止 ) When off, the warp in focus will be stepped and the focused CUDA thread will be changed to the nearest active lane in the warp. ( 另一种情况即warp前进，且focus被切换到最近的活跃线程 )

In order to advance the execution of more than one warp, a breakpoint must be set at the desired location and then the application must be fully resumed. ( 如果我们需要前进超过一个的warp，我们需要在适当的地方设定断点，然后让整个应用程序继续执行至该断点 )

A special case is single-stepping over thread barrier calls like: `__syncthreads()` or cluster-wide barriers. In this case, an implicit temporary breakpoint is set immediately after the barrier and all threads are resumed until the temporary breakpoint is hit. ( 特例是单步步过线程barrier调用，例如 `__syncthreads()` ，在这种情况下，一个隐式的暂时断点会被设定在该barrier之后，然后所有的线程会继续执行直到到达该断点 )

You can step in, over, or out of the device functions as long as they are not inlined. ( 只要设备函数没有内联，我们都可以步入，步过，步出该函数 ) To force a function to not be inlined by the compiler, the `__noinline__` keyword must be added to the function declaration. ( 在函数的声明添加 `__noinline__` 关键词可以让编译器不对其内联 )

Asynchronous SASS instructions executed on the device, such as the warpgroup instructions, at prior PCs are not guaranteed to be complete. ( 先前执行的异步SASS指令不一定会完成 )

With Dynamic Parallelism, several CUDA APIs can be called directly from device code. The following list defines single-step behavior when encountering these APIs:
- When encountering device side kernel launches (denoted by the `<<<>>>` launch syntax), the `step` and `next` commands will have the same behavior, and both will **step over** the launch call.
- On devices prior to Hopper (SM 9.0), stepping into the deprecated `cudaDeviceSynchronize()` results in undefined behavior. Users shall step over this call instead.
- When stepping a device grid launch to completion, focus will automatically switch back to the CPU. The `cuda kernel` focus switching command must be used to switch to another grid of interest (if one is still resident).

It is not possible to **step into** a device launch call (nor the routine launched by the call).
( CUDA-GDB不支持步入内核调用，只会步过，我们应该在主机代码中设置断点，检查设备启动调用之前的变量状态和程序流程，并在CUDA内核代码中也设置断点，以便当内核执行到这些点时暂停执行，检查变量状态、线程行为，如果需要调试CUDA内核代码，应该在内核代码中设置断点，然后使用调试器的“continue”命令来运行程序，直到内核开始执行并遇到这些断点，但不能从主机代码直接步入内核代码的执行，这是由于主机和设备代码在不同的执行环境中运行，它们的调试模型也相应地不同 )
# 7 Breakpoints and Watchpoints
There are multiple ways to set a breakpoint on a CUDA application. These methods are described below. The commands used to set a breakpoint on device code are the same as the commands used to set a breakpoint on host code.


If a breakpoint is set on device code, the breakpoint will be marked pending until the ELF image of the kernel is loaded. At that point, the breakpoint will be resolved and its address will be updated. ( 在设备代码中设置断点时，调试器会将该断点标记为“待定”，这意味着断点已经被创建，但还没有与设备代码的实际内存地址关联起来，ELF(Executable and Linkable Format)是Unix和类Unix系统上常用的一种文件格式，用于可执行文件、可重定位的目标代码和共享库，在这里，"ELF image"指的是内核函数的二进制表示形式，当CUDA内核被加载到GPU内存中时，它的ELF映像会被映射到调试器的地址空间中，这通常发生在内核执行前，调试器需要加载这个映像以解析断点，一旦ELF映像被加载，调试器将能够将待定的断点与设备代码中的实际内存地址关联起来这个过程称为“解析”，它使得断点从一个未确定的状态变为一个可以在执行到该地址时触发的状态，随着断点被解析，它的地址也会被更新为实际的内存地址，这个地址是内核代码中断点所在位置的地址；这整个过程确保了即使设备代码在调试会话开始时尚未被加载，调试器也能够正确地设置和管理断点，当设备代码被加载并准备执行时，调试器会更新断点状态，使其能够在适当的时刻触发，从而允许开发者在设备代码中进行调试。这对于调试GPU上运行的并行程序是非常重要的 ) 

When a breakpoint is set, it forces all resident GPU threads to stop at this location when it reaches the corresponding PC. ( 设定好了断点以后，debugger就会让所有的线程达到对应的PC之后停止 )

When a breakpoint is hit by one thread, there is no guarantee that the other threads will hit the breakpoint at the same time.  ( 并不保证一个线程达到断点之后，另一个线程也会同时达到 ) Therefore the same breakpoint may be hit several times, and the user must be careful with checking which thread(s) actually hit(s) the breakpoint. The `disable` command can be used to prevent hitting the breakpoint by additional threads.  ( 可以通过 `disable` 防止其他线程达到断点 )
## 7.1 Symbolic Breakpoints
To set a breakpoint at the entry of a function, use the `break` command followed by the name of the function or method: ( 在函数/方法的入口处设定断点 ):
```
(cuda-gdb) break my_function
(cuda-gdb) break my_class::my_method
```
For templatized functions and methods, the full signature must be given: ( 模板函数必须给定完整签名 )
```
(cuda-gdb) break int my_templatized_function<int>(int)
```
The mangled name of the function can also be used. To find the mangled name of a function, you can use the following command: ( Mangled name是编译器为了唯一标识每个函数，包括其参数类型和返回类型，而生成的编码过的函数名称 )
```
(cuda-gdb) set demangle-style none
(cuda-gdb) info function my_function_name
(cuda-gdb) set demangle-style auto
```
## 7.2 Line Breakpoints
To set a breakpoint on a specific line number, use the following syntax: ( 在特定的一行设定断点 )
```
(cuda-gdb) break my_file.cu:185
```
If the specified line corresponds to an instruction within templatized code, multiple breakpoints will be created, one for each instance of the templatized code. ( 如果改行涉及模板代码，会创建多个断点，对应模板代码的多个实例 )
## 7.3 Address Breakpoints
To set a breakpoint at a specific address, use the `break` command with the address as argument: ( 在特定地址设定断点 )
```
(cuda-gdb) break *0x1afe34d0
```
The address can be any address on the device or the host. ( 地址可以是设备也可以是主机地址 )
## 7.4 Kernel Entry Breakpoints
To break on the first instruction of every launched kernel, set the `break_on_launch` option to application: ( 设定在每次内核发起时打断 )
```
(cuda-gdb) set cuda break_on_launch application
```
See [set cuda break_on_launch](https://docs.nvidia.com/cuda/cuda-gdb/index.html#set-cuda-break-on-launch) for more information.
## 7.5 Conditional Breakpoints
To make the breakpoint conditional, use the optional if keyword or the cond command. ( 设定条件断点 )
```
(cuda-gdb) break foo.cu:23 if threadIdx.x == 1 && i < 5
(cuda-gdb) cond 3 threadIdx.x == 1 && i < 5
```
Conditional expressions may refer any variable, including built-in variables such as `threadIdx` and `blockIdx`. Function calls are not allowed in conditional expressions. ( 条件表达式可以引用任意变量，但不可以包含函数调用 )

Note that conditional breakpoints are always hit and evaluated, but the debugger reports the breakpoint as being hit only if the conditional statement is evaluated to TRUE. ( 注意条件断点总是会被击中并且被评估，但是debugger凭情况决定是否报告 ) The process of hitting the breakpoint and evaluating the corresponding conditional statement is time-consuming. Therefore, running applications while using conditional breakpoints may slow down the debugging session. Moreover, if the conditional statement is always evaluated to FALSE, the debugger may appear to be hanging or stuck, although it is not the case. You can interrupt the application with CTRL-C to verify that progress is being made.

Conditional breakpoints can be set on code from CUDA modules that are not already loaded. The verification of the condition will then only take place when the ELF image of that module is loaded. Therefore any error in the conditional expression will be deferred until the CUDA module is loaded. To double check the desired conditional expression, first set an unconditional breakpoint at the desired location and continue. When the breakpoint is hit, evaluate the desired conditional statement by using the `cond` command.
## 7.6 Watchpoints
Watchpoints on CUDA code are not supported.

Watchpoints on host code are supported. The user is invited to read the GDB documentation for a tutorial on how to set watchpoints on host code.
# 8. Inspecting Program State
## 8.1. Memory and Variables
The GDB print command has been extended to decipher the location of any program variable and can be used to display the contents of any CUDA program variable including:
- data allocated via `cudaMalloc()`
- data that resides in various GPU memory regions, such as shared, local, and global memory
- special CUDA runtime variables, such as `threadIdx`
## 8.2. Variable Storage and Accessibility
Depending on the variable type and usage, variables can be stored either in registers or in `local`, `shared`, `const` or `global` memory. You can print the address of any variable to find out where it is stored and directly access the associated memory. ( 可以print变量的地址，以查询变量储存在哪里，并直接访问其相关的存储 )

The example below shows how the variable array, which is of type `shared int *`, can be directly accessed in order to see what the stored values are in the array.
```
(cuda-gdb) print &array
$1 = (@shared int (*)[0]) 0x20
(cuda-gdb) print array[0]@4
$2 = {0, 128, 64, 192}
```

You can also access the shared memory indexed into the starting offset to see what the stored values are:
```
(cuda-gdb) print *(@shared int*)0x20
$3 = 0
(cuda-gdb) print *(@shared int*)0x24
$4 = 128
(cuda-gdb) print *(@shared int*)0x28
$5 = 64
```

The example below shows how to access the starting address of the input parameter to the kernel.
```
(cuda-gdb) print &data
$6 = (const @global void * const @parameter *) 0x10
(cuda-gdb) print *(@global void * const @parameter *) 0x10
$7 = (@global void * const @parameter) 0x110000</>
```
## 8.3. Info CUDA Commands
These are commands that display information about the GPU and the application’s CUDA state. The available options are:

`devices`
information about all the devices

`sms`
information about all the active SMs in the current device

`warps`
information about all the active warps in the current SM ( 当前SM的所有活跃warp )

`lanes`
information about all the active lanes in the current warp

`kernels`
information about all the active kernels

`blocks`
information about all the active blocks in the current kernel

`threads`
information about all the active threads in the current kernel ( 当前kernel内所有活跃线程 )

`launch trace`
information about the parent kernels of the kernel in focus

`launch children`
information about the kernels launched by the kernels in focus

`contexts`
information about all the contexts

A filter can be applied to every `info cuda` command. The filter restricts the scope of the command. ( 限制命令的范围 ) A filter is composed of one or more restrictions. A restriction can be any of the following:
- `device n`
- `sm n`
- `warp n`
- `lane n`
- `kernel n`
- `grid n`
- `block x[,y]` or `block (x[,y])`
- `thread x[,y[,z]]` or `thread (x[,y[,z]])`
- `breakpoint all` and `breakpoint n`

where `n`, `x`, `y`, `z` are integers, or one of the following special keywords: `current`, `any`, and `all`. `current` indicates that the corresponding value in the current focus should be used. `any` and `all` indicate that any value is acceptable.

The `breakpoint all` and `breakpoint n` filter are only effective for the `info cuda threads` command.

### 8.3.1. info cuda devices
This command enumerates all the GPUs in the system sorted by device index. A `*` indicates the device currently in focus. This command supports filters. The default is `device all`. This command prints `No CUDA Devices` if no active GPUs are found. A device is not considered active until the first kernel launch has been encountered. ( 内核发起后，设备才认为是活跃 )
```
(cuda-gdb) info cuda devices
  Dev PCI Bus/Dev ID                Name Description SM Type SMs Warps/SM Lanes/Warp Max Regs/Lane Active SMs Mask
    0        06:00.0 GeForce GTX TITAN Z      GK110B   sm_35  15       64         32           256 0x00000000
    1        07:00.0 GeForce GTX TITAN Z      GK110B   sm_35  15       64         32           256 0x00000000
```
### 8.3.2. info cuda sms
This command shows all the SMs for the device and the associated active warps on the SMs. This command supports filters and the default is `device current sm all`. A `*` indicates the SM is focus. The results are grouped per device.
```
(cuda-gdb) info cuda sms
 SM Active Warps Mask
Device 0
* 0 0xffffffffffffffff
  1 0xffffffffffffffff
  2 0xffffffffffffffff
  3 0xffffffffffffffff
  4 0xffffffffffffffff
  5 0xffffffffffffffff
  6 0xffffffffffffffff
  7 0xffffffffffffffff
  8 0xffffffffffffffff
...
```
### 8.3.3. info cuda warps
This command takes you one level deeper and prints all the warps information for the SM in focus. This command supports filters and the default is `device current sm current warp all`. The command can be used to display which warp executes what block. ( 当前SM上所有的活跃warp )
```
(cuda-gdb) info cuda warps
Wp /Active Lanes Mask/ Divergent Lanes Mask/Active Physical PC/Kernel/BlockIdx
Device 0 SM 0
* 0    0xffffffff    0x00000000 0x000000000000001c    0    (0,0,0)
  1    0xffffffff    0x00000000 0x0000000000000000    0    (0,0,0)
  2    0xffffffff    0x00000000 0x0000000000000000    0    (0,0,0)
  3    0xffffffff    0x00000000 0x0000000000000000    0    (0,0,0)
  4    0xffffffff    0x00000000 0x0000000000000000    0    (0,0,0)
  5    0xffffffff    0x00000000 0x0000000000000000    0    (0,0,0)
  6    0xffffffff    0x00000000 0x0000000000000000    0    (0,0,0)
  7    0xffffffff    0x00000000 0x0000000000000000    0    (0,0,0)
 ...
```
### 8.3.4. info cuda lanes
This command displays all the lanes (threads) for the warp in focus. This command supports filters and the default is `device current sm current warp current lane all`. In the example below you can see that all the lanes are at the same physical PC. The command can be used to display which lane executes what thread.
```
(cuda-gdb) info cuda lanes
  Ln    State  Physical PC        ThreadIdx
Device 0 SM 0 Warp 0
*  0    active 0x000000000000008c   (0,0,0)
   1    active 0x000000000000008c   (1,0,0)
   2    active 0x000000000000008c   (2,0,0)
   3    active 0x000000000000008c   (3,0,0)
   4    active 0x000000000000008c   (4,0,0)
   5    active 0x000000000000008c   (5,0,0)
   6    active 0x000000000000008c   (6,0,0)
   7    active 0x000000000000008c   (7,0,0)
   8    active 0x000000000000008c   (8,0,0)
   9    active 0x000000000000008c   (9,0,0)
  10    active 0x000000000000008c  (10,0,0)
  11    active 0x000000000000008c  (11,0,0)
  12    active 0x000000000000008c  (12,0,0)
  13    active 0x000000000000008c  (13,0,0)
  14    active 0x000000000000008c  (14,0,0)
  15    active 0x000000000000008c  (15,0,0)
  16    active 0x000000000000008c  (16,0,0)
 ...
```
### 8.3.5. info cuda kernels
This command displays on all the active kernels on the GPU in focus. It prints the SM mask, kernel ID, and the grid ID for each kernel with the associated dimensions and arguments. The kernel ID is unique across all GPUs whereas the grid ID is unique per GPU. The `Parent` column shows the kernel ID of the parent grid. This command supports filters and the default is `kernel all`.
```
(cuda-gdb) info cuda kernels
  Kernel Parent Dev Grid Status   SMs Mask   GridDim  BlockDim      Name Args
*      1      -   0    2 Active 0x00ffffff (240,1,1) (128,1,1) acos_main parms=...
```

This command will also show grids that have been launched on the GPU with Dynamic Parallelism. Kernels with a negative grid ID have been launched from the GPU, while kernels with a positive grid ID have been launched from the CPU.
### 8.3.6. info cuda blocks
This command displays all the active or running blocks for the kernel in focus. ( 对于当前kernel多有活跃的block ) The results are grouped per kernel. This command supports filters and the default is `kernel current block all`. The outputs are coalesced by default.
```
(cuda-gdb) info cuda blocks
   BlockIdx   To BlockIdx  Count  State
Kernel 1
*  (0,0,0)    (191,0,0)    192    running
```

Coalescing can be turned off as follows in which case more information on the Device and the SM get displayed:
```
(cuda-gdb) set cuda coalescing off
```

The following is the output of the same command when coalescing is turned off.
```
(cuda-gdb) info cuda blocks
  BlockIdx   State    Dev SM
Kernel 1
*   (0,0,0)   running   0   0
    (1,0,0)   running   0   3
    (2,0,0)   running   0   6
    (3,0,0)   running   0   9
    (4,0,0)   running   0  12
    (5,0,0)   running   0  15
    (6,0,0)   running   0  18
    (7,0,0)   running   0  21
    (8,0,0)   running   0   1
 ...
```
### 8.3.7. info cuda threads
This command displays the application’s active CUDA blocks and threads with the total count of threads in those blocks. ( 应用内活跃的线程 ) Also displayed are the virtual PC and the associated source file and the line number information. The results are grouped per kernel. The command supports filters with default being `kernel current block all thread all`. The outputs are coalesced by default as follows:
```
(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx ThreadIdx Count   Virtual PC    Filename   Line
Device 0 SM 0
* (0,0,0  (0,0,0)    (0,0,0)  (31,0,0)    32  0x000000000088f88c   acos.cu   376
  (0,0,0)(32,0,0)  (191,0,0) (127,0,0) 24544  0x000000000088f800   acos.cu   374
 ...
```

Coalescing can be turned off as follows in which case more information is displayed with the output.
```
(cuda-gdb) info cuda threads
   BlockIdx  ThreadIdx  Virtual PC         Dev SM Wp Ln   Filename  Line
Kernel 1
*  (0,0,0)    (0,0,0)  0x000000000088f88c   0  0  0  0    acos.cu    376
   (0,0,0)    (1,0,0)  0x000000000088f88c   0  0  0  1    acos.cu    376
   (0,0,0)    (2,0,0)  0x000000000088f88c   0  0  0  2    acos.cu    376
   (0,0,0)    (3,0,0)  0x000000000088f88c   0  0  0  3    acos.cu    376
   (0,0,0)    (4,0,0)  0x000000000088f88c   0  0  0  4    acos.cu    376
   (0,0,0)    (5,0,0)  0x000000000088f88c   0  0  0  5    acos.cu    376
   (0,0,0)    (6,0,0)  0x000000000088f88c   0  0  0  6    acos.cu    376
   (0,0,0)    (7,0,0)  0x000000000088f88c   0  0  0  7    acos.cu    376
   (0,0,0)    (8,0,0)  0x000000000088f88c   0  0  0  8    acos.cu    376
   (0,0,0)    (9,0,0)  0x000000000088f88c   0  0  0  9    acos.cu    376
 ...
```

In coalesced form, threads must be contiguous in order to be coalesced. If some threads are not currently running on the hardware, they will create _holes_ in the thread ranges. For instance, if a kernel consist of 2 blocks of 16 threads, and only the 8 lowest threads are active, then 2 coalesced ranges will be printed: one range for block 0 thread 0 to 7, and one range for block 1 thread 0 to 7. Because threads 8-15 in block 0 are not running, the 2 ranges cannot be coalesced. ( 不再运行中的thread无法被合并打印出 )

The command also supports `breakpoint all` and `breakpoint breakpoint_number` as filters. The former displays the threads that hit all CUDA breakpoints set by the user. ( 击中用户设定的所有断点的线程 ) The latter displays the threads that hit the CUDA breakpoint _breakpoint_number_.
```
(cuda-gdb) info cuda threads breakpoint all
  BlockIdx ThreadIdx         Virtual PC Dev SM Wp Ln        Filename  Line
Kernel 0
   (1,0,0)   (0,0,0) 0x0000000000948e58   0 11  0  0 infoCommands.cu    12
   (1,0,0)   (1,0,0) 0x0000000000948e58   0 11  0  1 infoCommands.cu    12
   (1,0,0)   (2,0,0) 0x0000000000948e58   0 11  0  2 infoCommands.cu    12
   (1,0,0)   (3,0,0) 0x0000000000948e58   0 11  0  3 infoCommands.cu    12
   (1,0,0)   (4,0,0) 0x0000000000948e58   0 11  0  4 infoCommands.cu    12
   (1,0,0)   (5,0,0) 0x0000000000948e58   0 11  0  5 infoCommands.cu    12

(cuda-gdb) info cuda threads breakpoint 2 lane 1
  BlockIdx ThreadIdx         Virtual PC Dev SM Wp Ln        Filename  Line
Kernel 0
   (1,0,0)   (1,0,0) 0x0000000000948e58   0 11  0  1 infoCommands.cu    12
```
### 8.3.8. info cuda launch trace
This command displays the kernel launch trace for the kernel in focus. The first element in the trace is the kernel in focus. The next element is the kernel that launched this kernel. The trace continues until there is no parent kernel. In that case, the kernel is CPU-launched.

For each kernel in the trace, the command prints the level of the kernel in the trace, the kernel ID, the device ID, the grid Id, the status, the kernel dimensions, the kernel name, and the kernel arguments.

A kernel that has been launched but that is not running on the GPU will have a `Pending` status. ( 被发起但尚未在GPU运行 ) A kernel currently running on the GPU will be marked as `Active`. A kernel waiting to become active again will be displayed as `Sleeping`. When a kernel has terminated, it is marked as `Terminated`. For the few cases, when the debugger cannot determine if a kernel is pending or terminated, the status is set to `Undetermined`.

This command supports filters and the default is `kernel all`.

With `set cuda software_preemption on`, no kernel will be reported as active.
### 8.3.9. info cuda launch children
This command displays the list of non-terminated kernels launched by the kernel in focus. For each kernel, the kernel ID, the device ID, the grid Id, the kernel dimensions, the kernel name, and the kernel parameters are displayed.

This command supports filters and the default is `kernel all`.
### 8.3.10. info cuda contexts
This command enumerates all the CUDA contexts running on all GPUs. A `*` indicates the context currently in focus. This command shows whether a context is currently active on a device or not.
### 8.3.11. info cuda managed
This command shows all the static managed variables  ( 所有静态管理的变量 ) on the device or on the host depending on the focus.
## 8.4. Disassembly
The device SASS code can be disassembled using the standard GDB disassembly instructions such as `x/i` and `display/i`.
```
(cuda-gdb) x/4i $pc-32
   0xa689a8 <acos_main(acosParams)+824>: MOV R0, c[0x0][0x34]
   0xa689b8 <acos_main(acosParams)+840>: MOV R3, c[0x0][0x28]
   0xa689c0 <acos_main(acosParams)+848>: IMUL R2, R0, R3
=> 0xa689c8 <acos_main(acosParams)+856>: MOV R0, c[0x0][0x28]
```

For disassembly instruction to work properly, `cuobjdump` must be installed and present in your `$PATH`.

In the disassembly view, the current pc is prefixed with `=>`. For Maxwell (SM 5.0) and newer architectures, if an instruction triggers an exception it will be prefixed with `*>`.( 触发异常 ) If the pc and errorpc are the same instruction it will be prefixed with `*=>`.

For example, consider the following exception:
```
CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x555555c08620 (memexceptions_kernel.cu:17)

Thread 1 "memexceptions" received signal CUDA_EXCEPTION_14, Warp Illegal Address.
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]
0x0000555555c08fb0 in exception_kernel<<<(1,1,1),(1,1,1)>>> (data=0x7fffccc00000, exception=MMU_FAULT) at memexceptions_kernel.cu:50
50  }
(cuda-gdb)
```

The `disas` command can be used to view both the PC and the error PC that triggered the exception.
## 8.5. Registers
The device registers code can be inspected/modified using the standard GDB commands such as `info registers`.

```
(cuda-gdb) info registers $R0 $R1 $R2 $R3
R0             0xf0 240
R1             0xfffc48 16776264
R2             0x7800   30720
R3             0x80 128
```

The registers are also accessible as `$R<regnum>` built-in variables,  ( 内建变量 ) for example:
```
(cuda-gdb) printf "%d %d\n", $R0*$R3, $R2
30720 30720
```

Values of predicate and CC registers can be inspecting by printing system registers group or by using their respective pseudo-names: `$P0`..`$P6` and `$CC`.
```
(cuda-gdb) info registers system
P0             0x1  1
P1             0x1  1
P2             0x0  0
P3             0x0  0
P4             0x0  0
P5             0x0  0
P6             0x1  1
CC             0x0  0
```
## 8.6. Const banks
Memory allocated in the constant address space of GPU memory resides in two dimensional arrays called constant banks. Constant banks are noted `c[X][Y]` where `X` is the bank number and `Y` the offset. The memory address of a given bank/offset pair is obtained via the convenience function `$_cuda_const_bank(bank, offset)`.
```
(cuda-gdb) disass $pc,+16
Dump of assembler code from 0x7fffd5043d40 to 0x7fffd5043d50:
=> 0x00007fffd5043d40 <_Z9acos_main10acosParams+1856>:  MOV R0, c[0x0][0xc]
End of assembler dump.
(cuda-gdb) p *$_cuda_const_bank(0x0,0xc)
$1 = 8
```
# 9. Event Notifications
As the application is making forward progress, CUDA-GDB notifies the users about kernel events and context events. ( 内核事件和上下文事件 ) Within CUDA-GDB, _kernel_ refers to the device code that executes on the GPU, while _context_ refers to the virtual address space on the GPU for the kernel. ( GPU上对于当前kernel的虚拟地址空间 ) You can enable output of CUDA context and kernel events to review the flow of the active contexts and kernels. By default, only context event messages are displayed.
## 9.1. Context Events
Any time a CUDA context is created, pushed, popped, or destroyed by the application, CUDA-GDB can optionally display a notification message. The message includes the context id and the device id to which the context belongs.

By default, context event notification is disabled. The context event notification policy is controlled with the `context_events` option.
```
(cuda-gdb) set cuda context_events off
```
CUDA-GDB does not display the context event notification messages (default).

```
(cuda-gdb) set cuda context_events on
```    
CUDA-GDB displays the context event notification messages.
## 9.2. Kernel Events
Any time CUDA-GDB is made aware of the launch or the termination of a CUDA kernel, a notification message can be displayed. The message includes the kernel id, the kernel name, and the device to which the kernel belongs.

The kernel event notification policy is controlled with `kernel_events` and `kernel_events_depth` options.
```
(cuda-gdb) set cuda kernel_events none
```

Possible options are:
`none`
no kernel, application or system (default)

`application`
kernel launched by the user application

`system`
any kernel launched by the driver, such as memset

`all`
any kernel, application and system
```
(cuda-gdb) set cuda kernel_events_depth 0
```

Controls the maximum depth of the kernels after which no kernel event notifications will be displayed. A value of zero means that there is no maximum and that all the kernel notifications are displayed. ( 0表示不设限 ) A value of one means that the debugger will display kernel event notifications only for kernels launched from the CPU (default).
# 10. Automatic Error Checking

## 10.1. Checking API Errors
CUDA-GDB can automatically check the return code of any driver API or runtime API call. ( 自动检查API返回码 ) If the return code indicates an error, the debugger will stop or warn the user.

The behavior is controlled with the `set cuda api_failures` option. Three modes are supported:
- `hide` CUDA API call failures are not reported
- `ignore` Warning message is printed for every fatal CUDA API call failure (default)
- `stop` The application is stopped when a CUDA API call returns a fatal error
- `ignore_all` Warning message is printed for every CUDA API call failure
- `stop_all` The application is stopped when a CUDA API call returns any error

The success return code and other non-error return codes are ignored. For the driver API, those are: `CUDA_SUCCESS` and `CUDA_ERROR_NOT_READY`. For the runtime API, they are `cudaSuccess` and `cudaErrorNotReady`.
## 10.2. GPU Error Reporting
With improved GPU error reporting in CUDA-GDB, application bugs are now easier to identify and easy to fix. The following table shows the new errors that are reported on GPUs with compute capability `sm_20` and higher.

**Continuing the execution of your application after these errors are found can lead to application termination or indeterminate results.**

Warp errors may result in instructions to continue executing before the exception is recognized and reported. The reported `$errorpc` shall contain the precise address of the instruction that caused the exception. If the warp exits after the instruction causing exception has executed, but before the exception has been recognized and reported, it may result in the exception not being reported. CUDA-GDB relies on an active warp present on the device in order to report exceptions. To help avoid this scenario of unreported exceptions:

> - For Volta+ architectures, compile the application with `-G`. See [Compiling the Application](https://docs.nvidia.com/cuda/cuda-gdb/index.html#compiling-the-application) for more information.
> - Add `while(1);` before kernel exit. This shall ensure the exception is recognized and reported.
> - Rely on the compute-sanitizer `memcheck` tool to catch accesses that can lead to an exception.
## 10.3. Autostep
Autostep is a command to increase the precision of CUDA exceptions to the exact lane and instruction, when they would not have been otherwise.

Under normal execution, an exception may be reported several instructions after the exception occurred, or the exact thread where an exception occurred may not be known unless the exception is a lane error. However, the precise origin of the exception can be determined if the program is being single-stepped when the exception occurs. Single- stepping manually is a slow and tedious process; stepping takes much longer than normal execution and the user has to single-step each warp individually.

Autostep aides the user by allowing them to specify sections of code where they suspect an exception could occur, and these sections are automatically and transparently single- stepped the program is running. ( 选择可能认为出现异常的区域，并自动步入 ) The rest of the program is executed normally to minimize the slow-down caused by single-stepping. The precise origin of an exception will be reported if the exception occurs within these sections. ( 可以捕获异常精确的来源 ) Thus the exact instruction and thread where an exception occurred can be found quickly and with much less effort by using autostep.

**Autostep Usage**
```
autostep [LOCATION]
autostep [LOCATION] for LENGTH [lines|instructions]
```

- `LOCATION` may be anything that you use to specify the location of a breakpoint, such as a line number, function name, or an instruction address preceded by an asterisk. If no `LOCATION` is specified, then the current instruction address is used. ( 默认使用当前指令地址 )
- `LENGTH` specifies the size of the autostep window in number of lines or instructions (_lines_ and _instructions_ can be shortened, e.g., _l_ or _i_). If the length type is not specified, then _lines_ is the default. If the `for` clause is omitted, then the default is 1 line. ( 默认以行为单位，默认1行 )
- `astep` can be used as an alias for the `autostep` command.
- Calls to functions made during an autostep will be stepped over. ( 函数调用会被步过 )
- In case of divergence, the length of the autostep window is determined by the number of lines or instructions the first active lane in each warp executes.   ( 窗口长度以warp内第一个活跃线程为基准 )
    Divergent lanes are also single stepped, but the instructions they execute do not count towards the length of the autostep window.
- If a breakpoint occurs while inside an autostep window, the warp where the breakpoint was hit will not continue autostepping when the program is resumed. However, other warps may continue autostepping.
- Overlapping autosteps are not supported.

If an autostep is encountered while another autostep is being executed, then the second autostep is ignored.

If an autostep is set before the location of a memory error and no memory error is hit, then it is possible that the chosen window is too small. This may be caused by the presence of function calls between the address of the autostep location and the instruction that triggers the memory error. In that situation, either increase the size of the window to make sure that the faulty instruction is included, or move to the autostep location to an instruction that will be executed closer in time to the faulty instruction.

**Related Commands**
Autosteps and breakpoints share the same numbering so most commands that work with breakpoints will also work with autosteps.

`info autosteps` shows all breakpoints and autosteps. It is similar to `info breakpoints`.

`disable autosteps` disables an autostep. It is equivalent to `disable breakpoints n`.

`delete autosteps n` deletes an autostep. It is quivalent to `delete breakpoints n`.

`ignore n i` tells the debugger to not single-step the next _i_ times the debugger enters the window for autostep _n_. This command already exists for breakpoints.
