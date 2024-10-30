# 1 Release Notes
# 2 Kernel Profiling Guide
## 2.1 Introduction
### 2.1.1 Profiling Applications
During regular execution, a CUDA application process will be launched by the user. It communicates directly with the CUDA user-mode driver, and potentially with the CUDA runtime library. ( CUDA应用直接与CUDA用户模式驱动通信，也会与CUDA运行时库通信 )
![[NVIDIA Nsight Compute-Figure1.png]]

When profiling an application with NVIDIA Nsight Compute, the behavior is different. The user launches the NVIDIA Nsight Compute frontend (either the UI or the CLI) on the host system, which in turn starts the actual application as a new process on the target system. While host and target are often the same machine, the target can also be a remote system with a potentially different operating system. ( 使用Nsight，则用户与Nsight交互，由Nsight启动CUDA应用 )
![[NVIDIA Nsight Compute-Figure2.png]]
## 2.2 Metric Collection
Collection of performance metrics is the key feature of NVIDIA Nsight Compute. Since there is a huge list of metrics available, it is often easier to use some of the tool’s pre-defined [sets or sections](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sets-and-sections) to collect a commonly used subset. Users are free to adjust which metrics are collected for which kernels as needed, but it is important to keep in mind the [Overhead](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead) associated with data collection. ( 用户可以定义需要收集的性能度量指标集合，注意收集性能度量指标存在额外开销 )
### 2.2.1 Sets and Sections
NVIDIA Nsight Compute uses _Section Sets_ (short _sets_) to decide, on a very high level, the number of metrics to be collected. Each set includes one or more _Sections_, ( 一个或多个section构成一个set ) with each section specifying several logically associated metrics. ( section内的度量是逻辑上相关的 ) For example, one section might include only high-level SM and memory utilization metrics, while another could include metrics associated with the memory units, or the HW scheduler. 

The number and type of metrics specified by a section has significant impact on the overhead during profiling.

The `basic` set is collected when no `--set`, `--section` and no `--metrics` options are passed on the command line. The full set of sections can be collected with `--set full`. ( 默认收集 `basic` set，仅包含少量的高层次的度量 )

A file named `.ncu-ignore` may be placed in any directory to have its contents ignored when the tool looks for section (and rule) files. When adding section directories [recursively](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options-profile), even if the file is present, sub-directories are still searched.
### 2.2.2 Sections and Rules
### 2.2.3 Replay
Depending on which metrics are to be collected, kernels might need to be _replayed_ one or more times, since not all metrics can be collected in a single _pass_. For example, the number of metrics originating from hardware (HW) performance counters that the GPU can collect at the same time is limited. ( GPU一次可以从硬件计数器收集的度量的数量是有限的 ) In addition, patch-based software (SW) performance counters can have a high impact on kernel runtime and would skew results for HW counters.

**Kernel Replay**
In _Kernel Replay_, all metrics requested for a specific kernel instance in NVIDIA Nsight Compute are grouped into one or more passes. For the first pass, all GPU memory that can be accessed by the kernel is saved. After the first pass, the subset of memory that is written by the kernel is determined. ( 内核经过第一次执行后，它写入的存储区域会被记录 ) Before each pass (except the first one), this subset is restored in its original location to have the kernel access the same memory contents in each replay pass. ( 之后的replay会保持kernel访问和第一遍相同的区域 )

NVIDIA Nsight Compute attempts to use the fastest available storage location for this save-and-restore strategy. For example, if data is allocated in device memory, and there is still enough device memory available, it is stored there directly. If it runs out of device memory, the data is transferred to the CPU host memory. Likewise, if an allocation originates from CPU host memory, the tool first attempts to save it into the same memory location, if possible.

As explained in [Overhead](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead), the time needed for this increases the more memory is accessed, especially written, by a kernel. If NVIDIA Nsight Compute determines that only a single replay pass is necessary to collect the requested metrics, no save-and-restore is performed at all to reduce overhead. ( 注意这种存储和恢复的方法也会带来开销 )
![[NVIDIA Nsight Compute-Figure3.png]]

**Application Replay**
In _Application Replay_, all metrics requested for a specific kernel launch in NVIDIA Nsight Compute are grouped into one or more passes. In contrast to [Kernel Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-replay), the complete application is run multiple times, so that in each run one of those passes can be collected per kernel. ( kernel replay仅重跑kernel，application replay重跑整个程序，注意一个application可以有多个kernel )

For correctly identifying and combining performance counters collected from multiple application replay passes of a single kernel launch into one result, the application needs to be deterministic with respect to its kernel activities and their assignment to GPUs, contexts, streams, and potentially NVTX ranges. ( 要在程序执行多遍的情况下正确收集和某个kernel相关的度量，应用对于kernel应该是完全确定的 ) Normally, this also implies that the application needs to be deterministic with respect to its overall execution.

Application replay has the benefit that memory accessed by the kernel does not need to be saved and restored via the tool, as each kernel launch executes only once during the lifetime of the application process. ( application replay中，每次应用执行kernel也仅执行一次 ) Besides avoiding memory save-and-restore overhead, application replay also allows to disable [Cache Control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control). This is especially useful if other GPU activities preceding a specific kernel launch are used by the application to set caches to some expected state.

In addition, application replay can support profiling kernels that have interdependencies to the host during execution. With kernel replay, this class of kernels typically hangs when being profiled, because the necessary responses from the host are missing in all but the first pass. In contrast, application replay ensures the correct behavior of the program execution in each pass. ( application replay可以用于剖析依赖于host相应的、和host相互依赖的kernel，而kernel replay不行 )

In contrast to kernel replay, multiple passes collected via application replay imply that all host-side activities of the application are duplicated, too. If the application requires significant time for e.g. setup or file-system access, the overhead will increase accordingly.
![[NVIDIA Nsight Compute-Figure4.png]]

Across application replay passes, NVIDIA Nsight Compute matches metric data for the individual, selected kernel launches. The matching strategy can be selected using the `--app-replay-match` option. For matching, only kernels within the same process and running on the same device are considered. By default, the _grid_ strategy is used, which matches launches according to their kernel name and grid size. ( 默认根据kernel的名称和grid size进行匹配 ) When multiple launches have the same attributes (e.g. name and grid size), they are matched in execution order.
![[NVIDIA Nsight Compute-Figure5.png]]

**Range Replay**
In _Range Replay_, all requested metrics in NVIDIA Nsight Compute are grouped into one or more passes. In contrast to [Kernel Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-replay) and [Application Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-replay), _Range Replay_ captures and replays complete ranges of CUDA API calls and kernel launches within the profiled application. Metrics are then not associated with individual kernels but with the entire range. This allows the tool to execute kernels without serialization and thereby supports profiling kernels that should be run concurrently for correctness or performance reasons. ( 可以使用range replay定义范围，来同时分析需要并行运行的多个kernel )
![[NVIDIA Nsight Compute-Figure6.png]]

*Defining Ranges*
Range replay requires you to specify the range for profiling in the application. A range is defined by a start and an end marker and includes all CUDA API calls and kernels launched between these markers from any CPU thread. The application is responsible for inserting appropriate synchronization between threads to ensure that the anticipated set of API calls is captured. Range markers can be set using one of the following options:

ref to doc

Ranges must fulfill several requirements:
- It must be possible to synchronize all active CUDA contexts at the start of the range.
- Ranges must not include unsupported CUDA API calls. See [Supported APIs](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-replay-supported-apis) for the list of currently supported APIs.

the rest is passed

*Supported APIs*
Range replay supports a subset of the CUDA API for capture and replay.  If an unsupported API call is detected in the captured range, an error is reported and the range cannot be profiled. The groups listed below match the ones found in the [CUDA Driver API documentation](https://docs.nvidia.com/cuda/cuda-driver-api/index.html).

Generally, range replay only captures and replay CUDA _Driver_ API calls. CUDA _Runtime_ APIs calls can be captured when they generate only supported CUDA Driver API calls internally. Deprecated APIs are not supported.

ref to doc

**Application Range Replay**
In _Application Range Replay_, all requested metrics in NVIDIA Nsight Compute are grouped into one or more passes. Similar to [Range Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-replay), metrics are not associated with individual kernels but with the entire selected range. This allows the tool to execute workloads (kernels, CUDA graphs, …) without serialization and thereby supports profiling workloads that must be run concurrently for correctness or performance reasons.

In contrast to Range Replay, the range is not explicitly captured and executed directly for each pass, but instead the entire application is re-run multiple times, with one pass collected for each range in every application execution. This has the benefit that no application state must be observed and captured for each range and API calls within the range do not need to be supported explicitly, as correct execution of the range is handled by the application itself.

Defining ranges to profile is identical to [Range Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-replay-define-range). The CUDA context for which the range should be profiled must be current to the thread defining the start of the range and must be active for the entire range.

![[NVIDIA Nsight Compute-Figure7.png]]

**Graph Profiling**
passed
### 2.2.5 Profile Seriese
The performance of a kernel is highly dependent on the used launch parameters.  Small changes to the launch parameters can have a significant effect on the runtime behavior of the kernel.

Profile Series provide the ability to automatically profile a single kernel multiple times with changing parameters. The parameters to be modified and values to be tested can be independently enabled and configured. For each combination of selected parameter values a unique profile result is collected. And the modified parameter values are tracked in the description of the results of a series. By comparing the results of a profile series, the kernel’s behavior on the changing parameters can be seen and the most optimal parameter set can be identified quickly.
### 2.2.6 Overhead
As with most measurements, collecting performance data using NVIDIA Nsight Compute CLI incurs some runtime overhead on the application. The overhead does depend on a number of different factors:

ref to doc

Furthermore, only a limited number of metrics can be collected in a single _pass_ of the kernel execution. If more metrics are requested, the kernel launch is _replayed_ multiple times, with its accessible memory being saved and restored between subsequent passes to guarantee deterministic execution. Therefore, collecting more metrics can significantly increase overhead by requiring more replay passes and increasing the total amount of memory that needs to be restored during replay.

There is a relatively high one-time overhead for the first profiled kernel in each context to generate the metric configuration. This overhead does not occur for subsequent kernels in the same context, if the list of collected metrics remains unchanged.
## 2.3 Metrics Guide
### 2.3.1 Hardware Model
**Compute Model**
All NVIDIA GPUs are designed to support a general purpose heterogeneous parallel programming model, commonly known as _Compute_. This model decouples the GPU from the traditional graphics pipeline and exposes it as a general purpose parallel multi-processor.

The number of CTAs that fit on each SM depends on the physical resources required by the CTA. These resource limiters include the number of threads and registers, shared memory utilization, and hardware barriers.

Each CTA can be scheduled on any of the available SMs, where there is no guarantee in the order of execution. As such, CTAs must be entirely independent, which means it is not possible for one CTA to wait on the result of another CTA.

CTAs are further divided into groups of 32 threads called _Warps_. If the number of threads in a CTA is not dividable by 32, the last warp will contain the remaining number of threads.

The total number of CTAs that can run concurrently on a given GPU is referred to as _Wave_. Consequently, the size of a Wave scales with the number of available SMs of a GPU, but also with the occupancy of the kernel.

**Streaming Multiprocessor**
The SM is designed to simultaneously execute multiple CTAs. CTAs can be from different grid launches.

The SM maintains execution state per thread, including a program counter (PC) and call stack.

Each SM is partitioned into four processing blocks, called _SM sub partitions_. ( 每个SM分为4个处理块称为SM子分区 ) The SM sub partitions are the primary processing elements on the SM. Each sub partition contains the following units: 
- Warp Scheduler
- Register File
- Execution Units/Pipelines/Cores
    - Integer Execution units
    - Floating Point Execution units
    - Memory Load/Store units
    - Special Function unit
    - Tensor Cores

Shared within an SM across the four SM partitions are:
- Unified L1 Data Cache / Shared Memory
- Texture units
- RT Cores, if available

A warp is allocated to a sub partition and resides on the sub partition from launch to completion. A warp is referred to as _active_ or _resident_ when it is mapped to a sub partition. A sub partition manages a fixed size pool of warps. ( warp被调度到SM子分区上 ) On Volta architectures, the size of the pool is 16 warps. On Turing architectures the size of the pool is 8 warps. Active warps can be in _eligible_ state if the warp is ready to issue an instruction. This requires the warp to have a decoded instruction, all input dependencies resolved, and for the function unit to be available. Statistics on active, eligible and issuing warps can be collected with the [Scheduler Statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) section.

A warp is _stalled_ when the warp is waiting on
- an instruction fetch,
- a memory dependency (result of memory instruction),
- an execution dependency (result of previous instruction), or
- a synchronization barrier.

The most important resource under the compiler’s control is the number of registers used by a kernel. Each sub partition has a set of 32-bit registers, which are allocated by the HW in fixed-size chunks. The [Launch Statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) section shows the kernel’s register usage.

**Memory**
One difference between global and local memory is that local memory is arranged such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address (e.g., same index in an array variable, same member in a structure variable, etc.). ( warp内线程对各自local memory中同一个相对地址的访问总是合并的 )


Shared memory can be shared across a compute CTA. Compute CTAs attempting to share data across threads via shared memory must use synchronization operations (such as `__syncthreads()`) between stores and loads to ensure data written by any one thread is visible to other threads in the CTA. Similarly, threads that need to share data via global memory must use a more heavyweight global memory barrier.

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks that can be accessed simultaneously. ( 32个各为32bit的bank ) Any 32-bit memory read or write request made of 32 addresses that fall in 32 distinct memory banks can therefore be serviced simultaneously, yielding an overall bandwidth that is 32 times as high as the bandwidth of a single request. However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized.

A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word (even though the two addresses fall in the same bank). When multiple threads make the same read access, one thread receives the data and then broadcasts it to the other threads. When multiple threads write to the same location, only one thread succeeds in the write; which thread that succeeds is undefined.

Detailed memory metrics are collected by the [Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) section.

**Caches**
All GPU units communicate to main memory through the Level 2 cache, also known as the L2. The L2 cache sits between on-chip memory clients and the framebuffer. L2 works in physical-address space. In addition to providing caching functionality, L2 also includes hardware to perform compression and global atomics.

The Level 1 Data Cache, or L1, plays a key role in handling global, local, shared, texture, and surface memory reads and writes, as well as reduction and atomic operations. On Volta and Turing architectures there are , there are two L1 caches per TPC, one for each SM. For more information on how L1 fits into the texturing pipeline, see the [TEX unit](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-tex-surf) description. Also note that while this section often uses the name “L1”, it should be understood that the L1 data cache, shared data, and the Texture data cache are one and the same. ( L1 = Texture = shared )

L1 receives requests from two units: the SM and TEX. L1 receives global and local memory requests from the SM and receives texture and surface requests from TEX.
### 2.3.2 Metrics Structure
**Metrics Overvies**
NVIDIA Nsight Compute uses an advanced metrics calculation system, designed to help you determine what happened (counters and metrics), and how close the program reached to peak GPU performance (throughputs as a percentage).  Every counter has associated peak rates in the database, to allow computing its throughput as a percentage.

Throughput metrics return the maximum percentage value of their constituent counters. These constituents have been carefully selected to represent the sections of the GPU pipeline that govern peak performance.

Two types of peak rates are available for every counter: burst and sustained. Burst rate is the maximum rate reportable in a single clock cycle. Sustained rate is the maximum rate achievable over an infinitely long measurement period, for “typical” operations. For many counters, burst equals sustained. Since the burst rate cannot be exceeded, percentages of burst rate will always be less than 100%. Percentages of sustained rate can occasionally exceed 100% in edge cases.

**Metrics Entities**
While in NVIDIA Nsight Compute, all performance counters are named _metrics_, they can be split further into groups with specific properties.

**Counters** may be either a raw counter from the GPU, or a calculated counter value. Every counter has four sub-metrics under it, which are also called _roll-ups_ ( 每个计数器有4个子度量 )

the rest is passed
## 2.8 Roofline Charts
Roofline charts provide a very helpful way to visualize achieved performance on complex processing units, like GPUs. This section introduces the Roofline charts that are presented within a profile report.
### 2.8.1 Overview
A typical roofline chart combines the peak performance and memory bandwidth of the GPU, with a metric called _Arithmetic Intensity_ (a ratio between _Work_ and _Memory Traffic_), into a single chart

This chart actually shows two different rooflines. However, the following components can be identified for each:
- **Vertical Axis** - The vertical axis represents _Floating Point Operations per Second_ (FLOPS). For GPUs this number can get quite large and so the numbers on this axis can be scaled for easier reading (as shown here). In order to better accommodate the range, this axis is rendered using a logarithmic scale. ( 纵轴FLOPS )
- **Horizontal Axis** - The horizontal axis represents _Arithmetic Intensity_, which is the ratio between _Work_ (expressed in floating point operations per second), and _Memory Traffic_ (expressed in bytes per second). The resulting unit is in floating point operations per byte. This axis is also shown using a logarithmic scale. ( 横轴算数密度 FLOP/byte )
- **Memory Bandwidth Boundary** - The memory bandwidth boundary is the _sloped_ part of the roofline. By default, this slope is determined entirely by the memory transfer rate of the GPU but can be customized inside the _SpeedOfLight_RooflineChart.section_ file if desired. ( 斜坡：存储带宽限制 )
- **Peak Performance Boundary** - The peak performance boundary is the _flat_ part of the roofline By default, this value is determined entirely by the peak performance of the GPU but can be customized inside the _SpeedOfLight_RooflineChart.section_ file if desired. ( 平地：计算限制 )
- **Ridge Point** - The ridge point is the point at which the memory bandwidth boundary meets the peak performance boundary. This point is a useful reference when analyzing kernel performance. ( 脊点：内存带宽足够 )
- **Achieved Value** - The achieved value represents the performance of the profiled kernel. If baselines are being used, the roofline chart will also contain an achieved value for each baseline. The outline color of the plotted achieved value point can be used to determine from which baseline the point came.
### 2.8.2 Analysis
As shown here, the _ridge point_ partitions the roofline chart into two regions. The area shaded in blue under the sloped _Memory Bandwidth Boundary_ is the _Memory Bound_ region, while the area shaded in green under the _Peak Performance Boundary_ is the _Compute Bound_ region. The region in which the _achieved value_ falls, determines the current limiting factor of kernel performance.
## 2.9 Memory Chart
The _Memory Chart_ shows a graphical, logical representation of performance data for memory subunits on and off the GPU. Performance data includes transfer sizes, hit rates, number of instructions or requests, etc.
### 2.9.1 Overview
**Logical Units (green)**
Logical units are shown in green color.
- Kernel: The CUDA kernel executing on the GPU’s Streaming Multiprocessors
- Global: CUDA [global memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-memory)
- Local: CUDA [local memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-memory)
- Texture: CUDA [texture memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-tex-surf)
- Surface: CUDA [surface memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-tex-surf)
- Shared: CUDA [shared memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-memory)
- Load Global Store Shared: Instructions loading directly from global into shared memory without intermediate register file access

**Physical Units (blue)**
Physical units are shown in blue color.
- L1/TEX Cache: The [L1/Texture cache](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-caches). The underlying physical memory is split between this cache and the user-managed _Shared Memory_.
- Shared Memory: CUDA’s user-managed [shared memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-memory). The underlying physical memory is split between this and the _L1/TEX Cache_.
- L2 Cache: The [L2 cache](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-caches)
- L2 Compression: The memory compression unit of the _L2 Cache_
- System Memory: Off-chip [system (CPU) memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-memory)
- Device Memory: On-chip [device (GPU) memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-memory) of the CUDA device that executes the kernel
- Peer Memory: On-chip [device (GPU) memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-memory) of other CUDA devices

**Links**
Links between _Kernel_ and other logical units represent the number of executed instructions (_Inst_) targeting the respective unit. ( 执行指令数量 ) For example, the link between _Kernel_ and _Global_ represents the instructions loading from or storing to the global memory space. Instructions using the NVIDIA A100’s _Load Global Store Shared_ paradigm are shown separately, as their register or cache access behavior can be different from regular global loads or shared stores.
## 2.10 Memory Tables
The _Memory Tables_ show detailed metrics for the various memory HW units, such as shared memory, the caches, and device memory. For most table entries, you can hover over it to see the underlying metric name and description. Some entries are generated as derivatives from other cells, and do not show a metric name on their own, but the respective calculation.
# 3 Nsight Compute
## 3.1 Introduction
### 3.1.1 Overview
NVIDIA Nsight Compute is an interactive kernel profiler for CUDA applications.
## 3.2 Quickstart
The UI executable is called ncu-ui. A shortcut with this name is located in the base directory of the NVIDIA Nsight Compute installation.
### 3.2.1 Interactive Profile Activity
1. **Launch the target application from NVIDIA Nsight Compute**
     In the _Activity_ panel, select the Interactive Profile activity to initiate a session that allows controlling the execution of the target application and selecting the kernels of interest interactively. Press _Launch_ to start the session.
2. **Launch NVIDIA Nsight Compute and connect to target application**
    Select the target machine at the top of the dialog to connect and update the list of attachable applications. By default, _localhost_ is pre-selected if the target matches your current local platform. Select the _Attach_ tab and the target application of interest and press _Attach_. Once connected, the layout of NVIDIA Nsight Compute changes into stepping mode that allows you to control the execution of any calls into the instrumented API. When connected, the _API Stream_ window indicates that the target application waits before the very first API call.
3. **Control application execution**
    Use the _API Stream_ window to step the calls into the instrumented API. The dropdown at the top allows switching between different CPU threads of the application. _Step In_ (F11), _Step Over_ (F10), and _Step Out_ (Shift + F11) are available from the _Debug_ menu or the corresponding toolbar buttons. While stepping, function return values and function parameters are captured. 1. Use _Resume_ (F5) and _Pause_ to allow the program to run freely. Freeze control is available to define the behavior of threads currently not in focus, i.e. selected in the thread drop down. By default, the _API Stream_ stops on any API call that returns an error code. This can be toggled in the _Debug_ menu by _Break On API Error_.
4. **Isolate a kernel launch**
    To quickly isolate a kernel launch for profiling, use the _Run to Next Kernel_ button in the toolbar of the _API Stream_ window to jump to the next kernel launch. The execution will stop before the kernel launch is executed.
### 3.2.2 Non-interactive Profile Activity
1. **Launch the target application from NVIDIA Nsight Compute**
     Then, fill in the launch details. In the _Activity_ panel, select the _Profile_ activity to initiate a session that pre-configures the profile session and launches the command line profiler to collect the data. Provide the _Output File_ name to enable starting the session with the _Launch_ button.
2. **Additional Launch Options**
### 3.2.3 System Trace Activity
### 3.2.4 Navigate the Report
## 3.3 Connection Dialog
Use the _Connection Dialog_ to launch and attach to applications on your local and remote platforms.

When using a remote platform, you will be asked to select or create a _Connection_ in the top drop down. To create a new connection, select _+_ and enter your connection details. When using the local platform, _localhost_ will be selected as the default and no further connection settings are required. You can still create or select a remote connection, if profiling will be on a remote system of the same platform.

Fill in the following launch details for the application:
- **Application Executable:** Specifies the root application to launch. Note that this may not be the final application that you wish to profile. It can be a script or launcher that creates other processes.
- **Working Directory:** The directory in which the application will be launched.
- **Command Line Arguments:** Specify the arguments to pass to the application executable.
- **Environment:** The environment variables to set for the launched application.

Select _Attach_ to attach the profiler to an application already running on the target platform. This application must have been started using another NVIDIA Nsight Compute CLI instance. The list will show all application processes running on the target system which can be attached. Select the refresh button to re-create this list.

Finally, select the _Activity_ to be run on the target for the launched or attached application. Note that not all activities are necessarily compatible with all targets and connection options. Currently, the following activities exist:
- [Interactive Profile Activity](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#connection-activity-interactive)
- [Profile Activity](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#connection-activity-non-interactive)
- [System Trace Activity](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#quick-start-system-trace)
- [Occupancy Calculator](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator)
### 3.3.1 Remote Connections
Note that once either activity type has been launched remotely, the tools necessary for further profiling sessions can be found in the _Deployment Directory_ on the remote device.

On Linux and Mac host platforms, NVIDIA Nsight Compute supports SSH remote profiling on target machines which are not directly addressable from the machine the UI is running on through the `ProxyJump` and `ProxyCommand` SSH options.

These options can be used to specify intermediate hosts to connect to or actual commands to run to obtain a socket connected to the SSH server on the target host and can be added to your SSH configuration file.
### 3.3.2. Interactive Profile Activity
The _Interactive Profile_ activity allows you to initiate a session that controls the execution of the target application, similar to a debugger. You can step API calls and workloads (CUDA kernels), pause and resume, and interactively select the kernels of interest and which metrics to collect.
### 3.3.3 Profile Activity
The _Profile_ activity provides a traditional, pre-configurable profiler. After configuring which kernels to profile, which metrics to collect, etc, the application is run under the profiler without interactive control. The activity completes once the application terminates. For applications that normally do not terminate on their own, e.g. interactive user interfaces, you can cancel the activity once all expected kernels are profiled.
## 3.4 Main Menu and Toolbar
### 3.4.1 Main Menu
- **Freeze API**
    When disabled, all CPU threads are enabled and continue to run during stepping or resume, and all threads stop as soon as at least one thread arrives at the next API call or launch. This also means that during stepping or resume the currently selected thread might change as the old selected thread makes no forward progress and the API Stream automatically switches to the thread with a new API call or launch. When enabled, only the currently selected CPU thread is enabled. All other threads are disabled and blocked.
    Stepping now completes if the current thread arrives at the next API call or launch. The selected thread never changes. However, if the selected thread does not call any further API calls or waits at a barrier for another thread to make progress, stepping may not complete and hang indefinitely. In this case, pause, select another thread, and continue stepping until the original thread is unblocked. In this mode, only the selected thread will ever make forward progress.
## 3.6 Profiler Report

