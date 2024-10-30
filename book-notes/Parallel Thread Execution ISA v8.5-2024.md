# 1 Introduction
This document describes PTX, a low-level _parallel thread execution_ virtual machine and instruction set architecture (ISA). PTX exposes the GPU as a data-parallel computing _device_. ( PTX将GPU暴露为数据并行计算设备 )
## 1.1 Scalable Data-Parallel Computing using GPUs
The GPU is especially well-suited to address problems that can be expressed as data-parallel computations - the same program is executed on many data elements in parallel - with high arithmetic intensity - the ratio of arithmetic operations to memory operations. Because the same program is executed for each data element, there is a lower requirement for sophisticated flow control;( 更低的对复杂流控制的要求 ) and because it is executed on many data elements and has high arithmetic intensity, the memory access latency can be hidden with calculations instead of big data caches. 
## 1.2 Goals of PTX
High level language compilers for languages such as CUDA and C/C++ generate PTX instructions, which are optimized for and translated to native target-architecture instructions.
## Programming Model
## 2.1 A Highly Mutithreaded Coprocesor
More precisely, a portion of an application that is executed many times, but independently on different data, can be isolated into a kernel function that is executed on the GPU as many different threads
## 2.2 Thread Hierarchy
### 2.2.1 Cooperative Thread Arrays
The _Parallel Thread Execution (PTX)_ programming model is explicitly parallel: a PTX program specifies the execution of a given thread of a parallel thread array. A _cooperative thread array_, or CTA, is an array of threads that execute a kernel concurrently or in parallel.

The thread identifier is a three-element vector `tid`, (with elements `tid.x`, `tid.y`, and `tid.z`) that specifies the thread’s position within a 1D, 2D, or 3D CTA. Each thread identifier component ranges from zero up to the number of thread ids in that CTA dimension. ( 线程标识符是一个三元素向量 `tid` )

Each CTA has a 1D, 2D, or 3D shape specified by a three-element vector `ntid` (with elements `ntid.x`, `ntid.y`, and `ntid.z`). The vector `ntid` specifies the number of threads in each CTA dimension. ( CTA的形状由三元素向量 `ntid` 标识 )

Some applications may be able to maximize performance with knowledge of the warp size, so PTX includes a run-time immediate constant, `WARP_SZ`, which may be used in any instruction where an immediate operand is allowed. ( PTX包括运行时立即数常量 `WARP_SZ` )
### 2.2.2 Cluster of Cooperative Thread Arrays
temporarily passed
### 2.2.3 Grid of Clusters
 Each grid also has a unique temporal grid identifier (_gridid_). Threads may read and use these values through predefined, read-only special registers `%tid`, `%ntid`, `%clusterid`, `%nclusterid`, and `%gridid`. ( 各个id存储于预定义的只读寄存器 `%..id` 中，可以供线程读取 )

Each CTA has a unique identifier (_ctaid_) within a grid. Each grid of CTAs has 1D, 2D, or 3D shape specified by the parameter _nctaid_. Thread may use and read these values through predefined, read-only special registers `%ctaid` and `%nctaid`.
## 2.3 Memory Hierarchy
 Each thread has a private local memory. Each thread block (CTA) has a shared memory visible to all threads of the block and to all active blocks in the cluster and with the same lifetime as the block. Finally, all threads have access to the same global memory.

There are additional state spaces accessible by all threads: the constant, param, texture, and surface state spaces. Constant and texture memory are read-only; ( Constant和Texture存储为只读 ) surface memory is readable and writable. ( Surface存储可读写 ) The global, constant, param, texture, and surface state spaces are optimized for different memory usages. For example, texture memory offers different addressing modes as well as data filtering for specific data formats. Note that texture and surface memory is cached, and within the same kernel call, the cache is not kept coherent with respect to global memory writes and surface memory writes, ( Texture和Surface Memory存在缓存，可能和Global Memory的内容不一致 ) so any texture fetch or surface read to an address that has been written to via a global or a surface write in the same kernel call returns undefined data. In other words, a thread can safely read some texture or surface memory location only if this memory location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread from the same kernel call.

The global, constant, and texture state spaces are persistent across kernel launches by the same application.
# 3 PTX Machine Model
## 3.1 A Set of SIMT Multiprocessors
The NVIDIA GPU architecture is built around a scalable array of multithreaded _Streaming Multiprocessors (SMs)_. When a host program invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity.

The threads of a thread block execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors. ( 前一个block结束执行，新block在空闲SM上被启动 )

A multiprocessor consists of multiple _Scalar Processor (SP)_ cores, a multithreaded instruction unit, and on-chip shared memory. The multiprocessor creates, manages, and executes concurrent threads in hardware with zero scheduling overhead. It implements a single-instruction barrier synchronization. ( 单指令barrier同步 ) Fast barrier synchronization together with lightweight thread creation and zero-overhead thread scheduling ( 快速的barrier同步、轻量thread创建、零开销thread调度 ) efficiently support very fine-grained parallelism, allowing, for example, a low granularity decomposition of problems by assigning one thread to each data element (such as a pixel in an image, a voxel in a volume, a cell in a grid-based computation).

When a multiprocessor is given one or more thread blocks to execute, it splits them into warps that get scheduled by the SIMT unit. The way a block is split into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0.

At every instruction issue time, the SIMT unit selects a warp that is ready to execute and issues the next instruction to the active threads of the warp. A warp executes one common instruction at a time, ( warp一次只能执行一条共同的指令，因此diverge会导致warp串行执行 ) so full efficiency is realized when all threads of a warp agree on their execution path.  If threads of a warp diverge via a data-dependent conditional branch, the warp serially executes each branch path taken, disabling threads that are not on that path, and when all paths complete, the threads converge back to the same execution path. Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjointed code paths.

How many blocks a multiprocessor can process at once depends on how many registers per thread and how much shared memory per block are required for a given kernel since the multiprocessor’s registers and shared memory are split among all the threads of the batch of blocks. If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch. ( 如果一个kernel所需的资源超过了一个SM的资源，包括shared memory和register，则kernel无法发起 )
![[PTX ISA-Fig4.png]]
## 3.2 Independent Thread Scheduling
Starting with the Volta architecture, _Independent Thread Scheduling_ allows full concurrency between threads, regardless of warp. With _Independent Thread Scheduling_, the GPU maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity, ( GPU维护每个thread的执行状态，包括程序计数器和调用栈 ) either to make better use of execution resources or to allow one thread to wait for data to be produced by another. A schedule optimizer determines how to group active threads from the same warp together into SIMT units. ( 调度优化器决定如何将warp内的活跃thread组合为一个SIMT单元 ) This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity. ( thread现在可以以sub-warp粒度diverge和converge )

_Independent Thread Scheduling_ can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity of previous hardware architectures.
## 3.3 On-chip Shared Memory
As illustrated by [Figure 4](https://docs.nvidia.com/cuda/parallel-thread-execution/#set-of-simt-multiprocessors-hardware-model), each multiprocessor has on-chip memory of the four following types:
- One set of local 32-bit _registers_ per processor,
- A parallel data cache or _shared memory_ that is shared by all scalar processor cores and is where the shared memory space resides,
- A read-only _constant cache_ that is shared by all scalar processor cores and speeds up reads from the constant memory space, which is a read-only region of device memory, 
- A read-only _texture cache_ that is shared by all scalar processor cores and speeds up reads from the texture memory space, which is a read-only region of device memory; each multiprocessor accesses the texture cache via a _texture unit_ that implements the various addressing modes and data filtering.
# 4 Syntax
PTX programs are a collection of text source modules (files). PTX source modules have an assembly-language style syntax with instruction operation codes and operands. Pseudo-operations specify symbol and addressing management. ( 伪操作用于指定符号和寻址管理 ) The ptxas optimizing backend compiler optimizes and assembles PTX source modules to produce corresponding binary object files. ( ptxas为后端汇编器，汇编PTX源模块产生二进制目标文件 )
## 4.1 Source Format
Source modules are ASCII text. Lines are separated by the newline character (`\n`).

All whitespace characters are equivalent; whitespace is ignored except for its use in separating tokens in the language.

The C preprocessor cpp may be used to process PTX source modules. Lines beginning with `#` are preprocessor directives. The following are common preprocessor directives:

`#include`, `#define`, `#if`, `#ifdef`, `#else`, `#endif`, `#line`, `#file`

PTX is case sensitive and uses lowercase for keywords.

Each PTX module must begin with a `.version` directive specifying the PTX language version, followed by a `.target` directive specifying the target architecture assumed. See [PTX Module Directives](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-module-directives) for a more information on these directives. ( 每个PTX模块以 `.version` 指令开始，指定PTX语言版本，之后是 `.target` 指令，指定目标架构 )
## 4.2 Comments
Comments in PTX follow C/C++ syntax, using non-nested `/*` and `*/` for comments that may span multiple lines, and using `//` to begin a comment that extends up to the next newline character, which terminates the current line. Comments cannot occur within character constants, string literals, or within other comments. ( 注意不要注释嵌套注释 )

Comments in PTX are treated as whitespace.
## 4.3 Statements
A PTX statement is either a directive or an instruction. Statements begin with an optional label and end with a semicolon. ( PTX语句要么是编译器指令，用于指定编译器做出决策，要么是操作指令/机器指令，表示实际的操作，语句可以由一个标签为起始，以semicolon结束 )
### 4.3.1 Directive Statements
Directive keywords begin with a dot, so no conflict is possible with user-defined identifiers. ( 编译器指令关键字以 `.` 为起始 ) The directives in PTX are listed in [Table 1](https://docs.nvidia.com/cuda/parallel-thread-execution/#directive-statements-ptx-directives) and described in [State Spaces, Types, and Variables](https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces-types-and-variables) and [Directives](https://docs.nvidia.com/cuda/parallel-thread-execution/#directives).
![[PTX ISA-Table 1.png]]
### 4.3.2 Instruction Statements
Instructions are formed from an instruction opcode followed by a comma-separated list of zero or more operands, ( 操作指令由一个指令操作码起始，之后跟着0个或多个操作数 ) and terminated with a semicolon.  Operands may be register variables, constant expressions, address expressions, or label names. ( 操作数可以是寄存器变量、常量表达式、地址表达式或标签名 )Instructions have an optional guard predicate which controls conditional execution. ( 指令有一个可选的守护谓词，它控制条件执行 ) The guard predicate follows the optional label and precedes the opcode, and is written as `@p`, where `p` is a predicate register. The guard predicate may be optionally negated, written as `@!p`. ( 守护谓词在可选的标签之后，在opcode之前，写为 `@p` ，其中 `p` 是谓词寄存器，谓词可以被否定 )

The destination operand is first, followed by source operands. ( 目标操作数在前，源操作数在后 )

Instruction keywords are listed in [Table 2](https://docs.nvidia.com/cuda/parallel-thread-execution/#instruction-statements-reserved-instruction-keywords). All instruction keywords are reserved tokens in PTX.
## 4.4 Identifiers
User-defined identifiers follow extended C++ rules: they either start with a letter followed by zero or more letters, digits, underscore, or dollar characters; or they start with an underscore, dollar, or percentage character followed by one or more letters, digits, underscore, or dollar characters:

PTX does not specify a maximum length for identifiers and suggests that all implementations support a minimum length of at least 1024 characters.

 PTX allows the percentage sign as the first character of an identifier. The percentage sign can be used to avoid name conflicts, e.g., between user-defined variable names and compiler-generated names.

PTX predefines one constant and a small number of special registers that begin with the percentage sign, listed in [Table 4](https://docs.nvidia.com/cuda/parallel-thread-execution/#identifiers-predefined-identifiers). ( PTX预定义了一个常量和一些特殊寄存器，它们以 `%` 开始 )
![[PTX ISA-Table 4.png]]
## 4.5 Constants
PTX supports integer and floating-point constants and constant expressions. These constants may be used in data initialization and as operands to instructions. Type checking rules remain the same for integer, floating-point, and bit-size types. For predicate-type data and instructions, integer constants are allowed and are interpreted as in C, i.e., zero values are `False` and non-zero values are `True`. ( 整型常量可以用作谓词类型数据 )
### 4.5.1 Integer Constant
Integer constants are 64-bits in size and are either signed or unsigned, i.e., every integer constant has type `.s64` or `.u64`. ( 整型常量一定为64位 ) When used in an instruction or data initialization, each integer constant is converted to the appropriate size based on the data or instruction type at its use. ( 整型常量用于数据初始化或用于指令中的时候，它们会被转化为合适的大小 )

Integer literals may be written in decimal, hexadecimal, octal, or binary notation. The syntax follows that of C. Integer literals may be followed immediately by the letter `U` to indicate that the literal is unsigned.

Integer literals are non-negative and have a type determined by their magnitude and optional type suffix as follows: literals are signed (`.s64`) unless the value cannot be fully represented in `.s64` or the unsigned suffix is specified, in which case the literal is unsigned (`.u64`). ( 整型字面值非负，它们的类型取决于它们的数量级和可选的类型后缀，例如 `.s64` `u64`，默认在可表示范围下，为 `.s64`  )

The predefined integer constant `WARP_SZ` specifies the number of threads per warp for the target platform; to date, all target architectures have a `WARP_SZ` value of 32.
### 4.5.2 Floating-Point Constants
Floating-point constants are represented as 64-bit double-precision values, and all floating-point constant expressions are evaluated using 64-bit double precision arithmetic. ( 浮点常量以及所有的浮点常量表达式都是64位双精度浮点数 )  The only exception is the 32-bit hex notation for expressing an exact single-precision floating-point value; such values retain their exact 32-bit single-precision value and may not be used in constant expressions.  Each 64-bit floating-point constant is converted to the appropriate floating-point size based on the data or instruction type at its use.  ( 64位浮点常量同样在使用时被类型转换 )

Floating-point literals may be written with an optional decimal point and an optional signed exponent. Unlike C and C++, there is no suffix letter to specify size; literals are always represented in 64-bit double-precision format.

PTX includes a second representation of floating-point constants for specifying the exact machine representation using a hexadecimal constant. ( PTX中可以用16进制表示浮点常量 ) To specify IEEE 754 double-precision floating point values, the constant begins with `0d` or `0D` followed by 16 hex digits. To specify IEEE 754 single-precision floating point values, the constant begins with `0f` or `0F` followed by 8 hex digits.
### 4.5.3 Predicate Constants
In PTX, integer constants may be used as predicates. For predicate-type data initializers and instruction operands, integer constants are interpreted as in C, i.e., zero values are `False` and non-zero values are `True`.
### 4.5.4 Constant Expressions
Constant expressions are formed from constant literals, unary plus and minus, basic arithmetic operators (addition, subtraction, multiplication, division), comparison operators, the conditional ternary operator ( `?:` ), and parentheses. ( 常量表达式由常量字面值和运算符构成 ) Integer constant expressions also allow unary logical negation (`!`), bitwise complement (`~`), remainder (`%`), shift operators (`<<` and `>>`), bit-type operators (`&`, `|`, and `^`), and logical operators (`&&`, `||`).

Constant expressions in PTX do not support casts between integer and floating-point. ( PTX不支持整型常量表达式和浮点型常量表达式之间的转化 )
### 4.5.5 Integer Constant Expression Evaluation
Integer constant expressions are evaluated at compile time ( 整型常量表达式在编译时评估 ) according to a set of rules that determine the type (signed `.s64` versus unsigned `.u64`) of each sub-expression. These rules are based on the rules in C, but they’ve been simplified to apply only to 64-bit integers, and behavior is fully defined in all cases (specifically, for remainder and shift operators).

- Literals are signed unless unsigned is needed to prevent overflow, or unless the literal uses a `U` suffix. For example: ( 字面值默认为有符号，为了防止溢出是为无符号 )
    - `42`, `0x1234`, `0123` are signed.
    - `0xfabc123400000000`, `42U`, `0x1234U` are unsigned.
- Unary plus and minus preserve the type of the input operand. For example:
    - `+123`, `-1`, `-(-42)` are signed.
    - `-1U`, `-0xfabc123400000000` are unsigned. ( 一元加和减运算符不会改变操作数的类型，即有符号或无符号 )
- Unary bitwise complement (`~`) interprets the source operand as unsigned and produces an unsigned result. 
- Some binary operators require normalization of source operands. This normalization is known as _the usual arithmetic conversions_ and simply converts both operands to unsigned type if either operand is unsigned.
- Casting of expressions to signed or unsigned is supported using (`.s64`) and (`.u64`) casts.
### 4.5.6 Summary of Constant Expression Evaluation Rules
# 5 State Spaces, Types, and Variables
While the specific resources available in a given target GPU will vary, the kinds of resources will be common across platforms, and these resources are abstracted in PTX through state spaces and data types. ( GPU资源被PTX通过状态空间和数据类型抽象 )
## 5.1 State Spaces
A state space is a storage area with particular characteristics. ( 状态空间即一个存储区域 ) All variables reside in some state space. The characteristics of a state space include its size, addressability, access speed, access rights, and level of sharing between threads. ( 所有的变量都存在于某些状态空间中，状态空间的特性包括其大小，可寻址性，访问速度，访问权限，以及线程间的共享级别 )

The state spaces defined in PTX are a byproduct of parallel programming and graphics programming. The list of state spaces is shown in [Table 7](https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces-state-spaces-tab),and properties of state spaces are shown in [Table 8](https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces-properties-state-spaces).
![[PTX ISA-Table 7.png]]
![[PTX ISA-Table 8.png]]
`.sreg` 的共享级别是per-CTA，`.const` 的共享级别是per-grid
### 5.1.1 Register State Space
Registers (`.reg` state space) are fast storage locations. The number of registers is limited, and will vary from platform to platform. When the limit is exceeded, register variables will be spilled to memory, causing changes in performance. ( register超过用量就会spill ) For each architecture, there is a recommended maximum number of registers to use

Registers may be typed (signed integer, unsigned integer, floating point, predicate) or untyped. ( register可以有类型也可以无类型 ) Register size is restricted; aside from predicate registers which are 1-bit, ( 谓词寄存器仅1bit ) scalar registers have a width of 8-, 16-, 32-, 64-, or 128-bits, and vector registers have a width of 16-, 32-, 64-, or 128-bits. The most common use of 8-bit registers is with `ld`, `st`, and `cvt` instructions, or as elements of vector tuples.

Registers differ from the other state spaces in that they are not fully addressable, ( register不是可完全寻址的 ) i.e., it is not possible to refer to the address of a register. When compiling to use the Application Binary Interface (ABI), register variables are restricted to function scope and may not be declared at module scope. When compiling legacy PTX code (ISA versions prior to 3.0) containing module-scoped `.reg` variables, the compiler silently disables use of the ABI. Registers may have alignment boundaries required by multi-word loads and stores. 
### 5.1.2 Special Register State Space
The special register (`.sreg`) state space holds predefined, platform-specific registers, such as grid, cluster, CTA, and thread parameters, clock counters, and performance monitoring registers. All special registers are predefined. ( 所有的特殊register都是预定义的 )
### 5.1.3 Constant State Space
The constant (`.const`) state space is a read-only memory initialized by the host. Constant memory is accessed with a `ld.const` instruction. Constant memory is restricted in size, currently limited to 64 KB which can be used to hold statically-sized constant variables. There is an additional 640 KB of constant memory, organized as ten independent 64 KB regions. Since the ten regions are not contiguous, the driver must ensure that constant buffers are allocated so that each buffer fits entirely within a 64 KB region and does not span a region boundary.

Statically-sized constant variables have an optional variable initializer; constant variables with no explicit initializer are initialized to zero by default. Constant buffers allocated by the driver are initialized by the host, and pointers to such buffers are passed to the kernel as parameters. ( 指向constant buffer的指针可以用作kernel参数 )
### 5.1.4 Global State Space
The global (`.global`) state space is memory that is accessible by all threads in a context. It is the mechanism by which threads in different CTAs, clusters, and grids can communicate. Use `ld.global`, `st.global`, and `atom.global` to access global variables.

Global variables have an optional variable initializer; global variables with no explicit initializer are initialized to zero by default.
### 5.1.5 Local State Space
The local state space (`.local`) is private memory for each thread to keep its own data. It is typically standard memory with cache. The size is limited, as it must be allocated on a per-thread basis. Use `ld.local` and `st.local` to access local variables.
### 5.1.6 Parameter State Space
The parameter (`.param`) state space is used (1) to pass input arguments from the host to the kernel, ( 其一，用于host向kernel传递参数 )(2a) to declare formal input and return parameters for device functions called from within kernel execution,  ( 其二，用于向kernel执行中调用的其他设备函数声明形式输入和返回参数 ) and (2b) to declare locally-scoped byte array variables that serve as function call arguments, typically for passing large structures by value to a function. ( 其三，用于声明用作函数调用参数的局部域字节数组变量，一般用作向一个函数值传递一个大的结构 ) Kernel function parameters differ from device function parameters in terms of access and sharing (read-only versus read-write, per-kernel versus per-thread) ( 核函数的参数和设备函数的参数在访问权限和共享范围方面有区别，核函数的参数为只读，共享范围为整个kernel，设备函数的参数为可读写，共享范围为thread )  The use of parameter state space for device function parameters was introduced in PTX ISA version 2.0 and requires target architecture `sm_20` or higher. Additional sub-qualifiers `::entry` or `::func` can be specified on instructions with `.param` state space to indicate whether the address refers to kernel function parameter or device function parameter. ( 对于带有 `.param` 状态空间的指令，可以额外注明 `::entry` 或 `::func` 子修饰符，表明地址是指向核函数或设备函数 ) If no sub-qualifier is specified with the `.param` state space, then the default sub-qualifier is specific to and dependent on the exact instruction. ( 若没有注明，则根据指令选择默认的子修饰符 ) For example, `st.param` is equivalent to `st.param::func` whereas `isspacep.param` is equivalent to `isspacep.param::entry`. Refer to the instruction description for more details on default sub-qualifier assumption.

The location of parameter space is implementation specific. For example, in some implementations kernel parameters reside in global memory. No access protection is provided between parameter and global space in this case. ( 参数空间的具体位置视情况而定 ) Though the exact location of the kernel parameter space is implementation specific, the kernel parameter space window is always contained within the global space window. ( 核函数参数总是位于global memory ) Similarly, function parameters are mapped to parameter passing registers and/or stack locations based on the function calling conventions of the _Application Binary Interface (ABI)_. Therefore, PTX code should make no assumptions about the relative locations or ordering of `.param` space variables.
#### 5.1.6.1 Kernel Functino Parameters
Each kernel function definition includes an optional list of parameters. These parameters are addressable, read-only variables declared in the `.param` state space. ( 核函数参数为 `.param` 状态空间的可寻址只读变量 ) Values passed from the host to the kernel are accessed through these parameter variables using `ld.param` instructions. ( 由主机向设备内核传递的参数值通过使用 `ld.param` 指令访问 )  The kernel parameter variables are shared across all CTAs from all clusters within a grid.

The address of a kernel parameter may be moved into a register using the `mov` instruction. The resulting address is in the `.param` state space and is accessed using `ld.param` instructions. ( 可以用 `mov` 将核函数参数移动至寄存器，但其地址仍位于 `.param` 状态空间，并使用 `ld.param` 访问 )

Kernel function parameters may represent normal data values, or they may hold addresses to objects in constant, global, local, or shared state spaces. In the case of pointers, the compiler and runtime system need information about which parameters are pointers, and to which state space they point. Kernel parameter attribute directives are used to provide this information at the PTX level. See [Kernel Function Parameter Attributes](https://docs.nvidia.com/cuda/parallel-thread-execution/#kernel-function-parameter-attributes) for a description of kernel parameter attribute directives.
#### 5.1.6.2 Kernel Function Parameter Attributes
Kernel function parameters may be declared with an optional .ptr attribute to indicate that a parameter is a pointer to memory, ( 带有 `.ptr` 属性声明的核函数参数说明该参数是指针 ) and also indicate the state space and alignment of the memory being pointed to.
#### 5.1.6.3 Kernel Parameter Attribute `.ptr`
**.ptr**
Kernel parameter alignment attribute.

**Description**
Used to specify the state space and, optionally, the alignment of memory pointed to by a pointer type kernel parameter. ( 指定指针指向的内核参数所在的状态空间和对齐状态 ) The alignment value _N_, if present, must be a power of two. If no state space is specified, the pointer is assumed to be a generic address pointing to one of const, global, local, or shared memory. If no alignment is specified, the memory pointed to is assumed to be aligned to a 4 byte boundary.

Spaces between `.ptr`, `.space`, and `.align` may be eliminated to improve readability.
#### 5.1.6.4 Device Function Paramters
PTX ISA version 2.0 extended the use of parameter space to device function parameters. The most common use is for passing objects by value that do not fit within a PTX register, such as C structures larger than 8 bytes In this case, a byte array in parameter space is used. ( 值传递超过32字节的结构时，可以使用参数空间传递设备函数参数 ) Typically, the caller will declare a locally-scoped `.param` byte array variable that represents a flattened C structure or union. ( 调用者声明局部范围的 `.param` 字节数组 ) This will be passed by value to a callee, which declares a `.param` formal parameter having the same size and alignment as the passed argument.  ( 被调用者声明 `.param` 的形式参数，对齐和大小和传入参数一致 )

Function input parameters may be read via `ld.param` and function return parameters may be written using `st.param`; ( 函数输入参数使用 `ld.param` 读取，函数返回参数使用 `st.param` 写入 ) it is illegal to write to an input parameter or read from a return parameter. ( 不可向输入参数写入，或从返回参数读取 )

Aside from passing structures by value, `.param` space is also required whenever a formal parameter has its address taken within the called function. ( 除了用于值传递结构， `.param` 状态空间在形式参数在被调用函数中的地址被改变时也要使用 ) In PTX, the address of a function input parameter may be moved into a register using the `mov` instruction. ( PTX中，函数输入参数的地址可以用 `mov` 移动至寄存器 ) Note that the parameter will be copied to the stack if necessary, and so the address will be in the `.local` state space and is accessed via `ld.local` and `st.local` instructions. It is not possible to use `mov` to get the address of or a locally-scoped `.param` space variable. Starting PTX ISA version 6.0, it is possible to use `mov` instruction to get address of return parameter of device function.
### 5.1.7 Shared State Space
The shared (`.shared`) state space is a memory that is owned by an executing CTA and is accessible to the threads of all the CTAs within a cluster. An address in shared memory can be read and written by any thread in a CTA cluster.

Additional sub-qualifiers `::cta` or `::cluster` can be specified on instructions with `.shared` state space to indicate whether the address belongs to the shared memory window of the executing CTA or of any CTA in the cluster respectively. ( 在带有 `.shared` 状态空间的指令后可以添加 `::cta` , `::cluster` 子修饰符，以表明该地址是属于当前执行的CTA的shared memory窗口还是属于cluster内的任意一个CTA的shared memory地址窗口 ) The addresses in the `.shared::cta` window also fall within the `.shared::cluster` window. ( 当然，`.shared::cta` 窗口内的地址也在 `.shared::cluster` 窗口内 ) If no sub-qualifier is specified with the `.shared` state space, then it defaults to `::cta`. For example, `ld.shared` is equivalent to `ld.shared::cta`. ( 默认的子修饰符是 `::cta` ，例如 `ld.shared` 等价于 `ld.shared::cta` )

Variables declared in `.shared` state space refer to the memory addresses in the current CTA. ( 声明于 `.shared` 状态空间的变量指向的是当前CTA的shared memory ) Instruction `mapa` gives the `.shared::cluster` address of the corresponding variable in another CTA in the cluster.

Shared memory typically has some optimizations to support the sharing. One example is broadcast; where all threads read from the same address. ( 所有的线程读同一地址会广播 ) Another is sequential access from sequential threads. ( 顺序线程顺序访问地址 )
## 5.2 Types
### 5.2.1 Fundamental Types
In PTX, the fundamental types reflect the native data types supported by the target architectures. ( PTX中的基本类型即目标架构支持的原生数据类型 ) A fundamental type specifies both a basic type and a size. ( 基本类型指定了基础类型和其大小 ) Register variables are always of a fundamental type, ( 寄存器变量一定是基本类型 ) and instructions operate on these types.  ( 指令对基本类型进行操作 ) The same type-size specifiers are used for both variable definitions and for typing instructions,  ( 类型大小指示符用于变量定义和指令中 ) so their names are intentionally short.

[Table 9](https://docs.nvidia.com/cuda/parallel-thread-execution/#fundamental-types-fundamental-type-specifiers) lists the fundamental type specifiers for each basic type:
![[PTX ISA-Table 9.png]]

Most instructions have one or more type specifiers, needed to fully specify instruction behavior. Operand types and sizes are checked against instruction types for compatibility. ( 大多数指令有1到2个类型指示符来完全指定指令的行为，操作数类型和大小要求和其兼容 )

Two fundamental types are compatible if they have the same basic type and are the same size. Signed and unsigned integer types are compatible if they have the same size. ( 相同大小的有符号和无符号整数类型兼容 ) The bit-size type is compatible with any fundamental type having the same size. ( bits类型和其余任意有相同大小的基本类型兼容 )

In principle, all variables (aside from predicates) could be declared using only bit-size types,  ( 原则上，除了谓词的所有变量都可以用bits类型声明 ) but typed variables enhance program readability and allow for better operand type checking.
### 5.2.2 Restricted Use of Sub-Word Sizes
The `.u8`, `.s8`, and `.b8` instruction types are restricted to `ld`, `st`, and `cvt` instructions. ( 仅 `ld` , `st` ,`cvt` 指令用到了 `.u8` , `.s8` , `.b8` 类型 ) The `.f16` floating-point type is allowed only in conversions to and from `.f32`, `.f64` types, in half precision floating point instructions and texture fetch instructions. ( `.f16` 类型仅用于向 `.f32/64` 类型的转换，以及在半精度浮点指令和texture fetch指令中 ) The `.f16x2` floating point type is allowed only in half precision floating point arithmetic instructions and texture fetch instructions. ( `.fp16x2` 仅用于半精度浮点算数指令以及texture fetch指令中 )

For convenience, `ld`, `st`, and `cvt` instructions permit source and destination data operands to be wider than the instruction-type size, so that narrow values may be loaded, stored, and converted using regular-width registers. ( `ld`, `st` , `cvt` 指令允许源和目标操作数大小大于指令类型大小，因此窄数据可以放在宽寄存器中 ) For example, 8-bit or 16-bit values may be held directly in 32-bit or 64-bit registers when being loaded, stored, or converted to other types and sizes.
### 5.2.3 Alternate Floating-Point Data Formats
The fundamental floating-point types supported in PTX have implicit bit representations that indicate the number of bits used to store exponent and mantissa. ( PTX支持的基本浮点类型有隐式的位表达，用于表示存储指数和小数的位数 ) For example, the `.f16` type indicates 5 bits reserved for exponent and 10 bits reserved for mantissa. In addition to the floating-point representations assumed by the fundamental types, PTX allows the following alternate floating-point data formats: ( PTX支持以下除基本类型以外的浮点类型 )

`bf16` data format:
This data format is a 16-bit floating point format with 8 bits for exponent and 7 bits for mantissa. A register variable containing `bf16` data must be declared with `.b16` type. ( 存储 `bf16` 的寄存器变量需要以 `.b16` 类型声明 )

`e4m3` data format:
This data format is an 8-bit floating point format with 4 bits for exponent and 3 bits for mantissa. The `e4m3` encoding does not support infinity and `NaN` values are limited to `0x7f` and `0xff`. A register variable containing `e4m3` value must be declared using bit-size type. ( 同样，变量需要以bit-size类型声明 )

`e5m2` data format:
This data format is an 8-bit floating point format with 5 bits for exponent and 2 bits for mantissa. A register variable containing `e5m2` value must be declared using bit-size type.

`tf32` data format:
This data format is a special 32-bit floating point format supported by the matrix multiply-and-accumulate instructions, with the same range as `.f32` and reduced precision (>=10 bits). ( 该类型是矩阵乘累加MMA指令支持的特殊32位浮点格式，其精度比常规 `f32` 更低 ) the internal layout of `tf32` format is implementation defined. PTX facilitates conversion from single precision `.f32` type to `tf32` format. A register variable containing `tf32` data must be declared with `.b32` type. ( 变量需要以bit-size类型声明 )

Alternate data formats cannot be used as fundamental types. They are supported as source or destination formats by certain instructions. ( 以上数据类型被部分指令支持，可以作为源或目标数据格式 )
### 5.2.4 Packed Data Types
Certain PTX instructions operate on two sets of inputs in parallel, and produce two outputs. ( 部分PTX执行平行处理两组输入，产生两组输出 ) Such instructions can use the data stored in a packed format. ( 这类指令可以使用打包格式的数据 ) PTX supports packing two values of the same scalar data type into a single, larger value. ( PTX支持将相同标量数据类型的值打包为单个值，其类型为打包数据类型 ) The packed value is considered as a value of a _packed data type_. In this section we describe the packed data types supported in PTX.
#### 5.2.4.1 Packed Floating Point Data Types
PTX supports the following four variants of packed floating point data types:
1. `.f16x2` packed type containing two `.f16` floating point values.
2. `.bf16x2` packed type containing two `.bf16` alternate floating point values.
3. `.e4m3x2` packed type containing two `.e4m3` alternate floating point values.
4. `.e5m2x2` packed type containing two `.e5m2` alternate floating point values.

`.f16x2` is supported as a fundamental type. ( `.f16x2` 为基本类型，故可以在任意指令中使用 )  `.bf16x2`, `.e4m3x2` and `.e5m2x2` cannot be used as fundamental types - they are supported as instruction types on certain instructions.  ( 仅有特定指令支持 ) A register variable containing `.bf16x2` data must be declared with `.b32` type. ( 存储 `.bf16x2` 的寄存器变量需要以 `.b32` 类型声明 ) A register variable containing `.e4m3x2` or `.e5m2x2` data must be declared with `.b16` type. ( 同样，变量需要以bit-size类型声明 )
#### 5.2.4.2 Packed Integer Data Types
PTX supports two variants of packed integer data types: `.u16x2` and `.s16x2`. The packed data type consists of two `.u16` or `.s16` values. A register variable containing `.u16x2` or `.s16x2` data must be declared with `.b32` type. Packed integer data types cannot be used as fundamental types. They are supported as instruction types on certain instructions.
## 5.4 Variables
In PTX, a variable declaration describes both the variable’s type and its state space. ( PTX中的变量声明包括其状态空间和变量类型 ) In addition to fundamental types, PTX supports types for simple aggregate objects such as vectors and arrays.
### 5.4.1 Variable Declarations
All storage for data is specified with variable declarations. Every variable must reside in one of the state spaces enumerated in the previous section.

A variable declaration names the space in which the variable resides, its type and size, its name, an optional array size, an optional initializer, and an optional fixed address for the variable. ( 变量声明包括变量处于的状态空间，其类型和大小，名称，optional: 初始值、固定地址、数组大小 )

Predicate variables may only be declared in the register state space. ( 谓词变量仅能在寄存器地址空间声明 )
#### 5.4.2 Vectors
Limited-length vector types are supported. Vectors of length 2 and 4 of any non-predicate fundamental type can be declared by prefixing the type with `.v2` or `.v4`.  ( 对于非谓词的基本类型，在声明中的类型之前添加 `.v2` , `.v4` 可以声明长度为2/4的向量 ) Vectors must be based on a fundamental type, and they may reside in the register space. ( 向量必须基于基本类型，且可以处于寄存器空间 ) Vectors cannot exceed 128-bits in length; ( 向量长度不可超过16字节 ) for example, `.v4 .f64` is not allowed. Three-element vectors may be handled by using a `.v4` vector, where the fourth element provides padding. This is a common case for three-dimensional grids, textures, etc.  ( 三元素向量使用 `.v4` 处理 )

By default, vector variables are aligned to a multiple of their overall size (vector length times base-type size), ( 向量变量默认于它们的总大小对齐 ) to enable vector load and store instructions which require addresses aligned to a multiple of the access size. ( 因为向量存取指令一般要求地址与访问大小对齐 )
### 5.4.3 Array Declarations
Array declarations are provided to allow the programmer to reserve space. To declare an array, the variable name is followed with dimensional declarations similar to fixed-size array declarations in C. The size of each dimension is a constant expression. ( 数组声明和C类似，其各维度大小为常量表达式，数组的空间会被预留 )

When declared with an initializer, the first dimension of the array may be omitted. ( 第一维度可以忽略 ) The size of the first array dimension is determined by the number of elements in the array initializer.
### 5.4.4 Initializers
Declared variables may specify an initial value using a syntax similar to C/C++, where the variable name is followed by an equals sign and the initial value or values for the variable. A scalar takes a single value, while vectors and arrays take nested lists of values inside of curly braces (the nesting matches the dimensionality of the declaration).

As in C, array initializers may be incomplete, i.e., the number of initializer elements may be less than the extent of the corresponding array dimension, with remaining array locations initialized to the default value for the specified array type.

Currently, variable initialization is supported only for constant and global state spaces.( 变量初始化目前仅支持constant和global状态空间 ) Variables in constant and global state spaces with no explicit initializer are initialized to zero by default. Initializers are not allowed in external variable declarations.

Variable names appearing in initializers represent the address of the variable; this can be used to statically initialize a pointer to a variable. ( 初始化值中出现的变量名表示变量的地址 ) Initializers may also contain _var+offset_ expressions, where _offset_ is a byte offset ( offset单位是字节 ) added to the address of _var_. Only variables in `.global` or `.const` state spaces may be used in initializers.  ( 仅有global和constant状态空间的变量可以用在初始化值中 ) By default, the resulting address is the offset in the variable’s state space (as is the case when taking the address of a variable with a `mov` instruction). An operator, `generic()`, is provided to create a generic address for variables used in initializers.

Starting PTX ISA version 7.1, an operator `mask()` is provided, where `mask` is an integer immediate. ( `mask` 是整型立即数 ) The only allowed expressions in the `mask()` operator are integer constant expression and symbol expression representing address of variable. ( `mask()` 操作符仅作用域表示变量地址的表达式 ) The `mask()` operator extracts `n` consecutive bits from the expression used in initializers and inserts these bits at the lowest position of the initialized variable. The number `n` and the starting position of the bits to be extracted is specified by the integer immediate `mask`. PTX ISA version 7.1 only supports extracting a single byte starting at byte boundary from the address of the variable. PTX ISA version 7.3 supports Integer constant expression as an operand in the `mask()` operator.

Supported values for `mask` are: 0xFF, 0xFF00, 0XFF0000, 0xFF000000, 0xFF00000000, 0xFF0000000000, 0xFF000000000000, 0xFF00000000000000.

PTX 3.1 redefines the default addressing for global variables in initializers, from generic addresses to offsets in the global state space. Legacy PTX code is treated as having an implicit `generic()` operator for each global variable used in an initializer. PTX 3.1 code should either include explicit `generic()` operators in initializers, use `cvta.global` to form generic addresses at runtime, or load from the non-generic address using `ld.global`.

Device function names appearing in initializers represent the address of the first instruction in the function; ( 初始化值中的设备函数名称表示该函数的第一个指令的地址 ) this can be used to initialize a table of function pointers to be used with indirect calls. Beginning in PTX ISA version 3.1, kernel function names can be used as initializers e.g. to initialize a table of kernel function pointers, to be used with CUDA Dynamic Parallelism to launch kernels from GPU. See the _CUDA Dynamic Parallelism Programming Guide_ for details.

Labels cannot be used in initializers.

Variables that hold addresses of variables or functions should be of type `.u8` or `.u32` or `.u64`. ( 储存地址的变量的类型应该是无符号整型 )

Type `.u8` is allowed only if the `mask()` operator is used.

Initializers are allowed for all types except `.f16`, `.f16x2` and `.pred`.
### 5.4.5 Alignment
Byte alignment of storage for all addressable variables can be specified in the variable declaration. Alignment is specified using an optional `.align` _byte-count_ specifier immediately following the state-space specifier. ( 对齐指示符在状态空间指示符之后，单位为字节 )The variable will be aligned to an address which is an integer multiple of byte-count. ( 变量地址将会是对齐值的整数倍 ) The alignment value byte-count must be a power of two. For arrays, alignment specifies the address alignment for the starting address of the entire array, not for individual elements. ( 数组变量只能保证起始地址的对齐，而非每个元素 )

The default alignment for scalar and array variables is to a multiple of the base-type size. The default alignment for vector variables is to a multiple of the overall vector size. ( 对于标量和数组，默认对齐大小即变量的基础类型大小，但对于向量，其默认对齐则是整个向量大小 )

Note that all PTX instructions that access memory require that the address be aligned to a multiple of the access size. ( 所有访存的PTX指令都要求访问地址是访问大小的倍数，即访问地址应该和访问大小对齐 ) The access size of a memory instruction is the total number of bytes accessed in memory. ( 指令的访问大小即指令访问的总字节数 ) For example, the access size of `ld.v4.b32` is 16 bytes, while the access size of `atom.f16x2` is 4 bytes.
### 5.4.6 Parameterized Variable Names
Since PTX supports virtual registers, it is quite common for a compiler frontend to generate a large number of register names. Rather than require explicit declaration of every name, PTX supports a syntax for creating a set of variables having a common prefix string appended with integer suffixes. ( PTX支持批量创建变量，其名称遵循规律，有共同前缀 )

For example, suppose a program uses a large number, say one hundred, of `.b32` variables, named `%r0`, `%r1`, …, `%r99`. These 100 register variables can be declared as follows:
```ptx
.reg .b32 %r<100>;
```

This shorthand syntax may be used with any of the fundamental types and with any state space, and may be preceded by an alignment specifier. Array variables cannot be declared this way, nor are initializers permitted.
### 5.4.7 Variable Attributes
Variables may be declared with an optional `.attribute` directive which allows specifying special attributes of variables. Keyword `.attribute` is followed by attribute specification inside parenthesis. Multiple attributes are separated by comma.

[Variable and Function Attribute Directive: .attribute](https://docs.nvidia.com/cuda/parallel-thread-execution/#variable-and-function-attribute-directive-attribute) describes the `.attribute` directive.
### 5.4.8 Variable and Function Attribute Directive `.attribute`
**.attribute**
Variable and function attributes

**Description**
Used to specify special attributes of a variable or a function.

The following attributes are supported.
`.managed`
`.managed` attribute specifies that variable will be allocated at a location in unified virtual memory environment where host and other devices in the system can reference the variable directly. ( 变量被分配于联合的虚拟存储环境中 ) This attribute can only be used with variables in .global state space. See the _CUDA UVM-Lite Programming Guide_ for details.

`.unified`
`.unified` attribute specifies that function has the same memory address on the host and on other devices in the system. ( 函数在主机和其他设备中都有相同的地址 ) Integer constants `uuid1` and `uuid2` respectively specify upper and lower 64 bits of the unique identifier associated with the function or the variable. This attribute can only be used on device functions or on variables in the `.global` state space. Variables with `.unified` attribute are read-only and must be loaded by specifying `.unified` qualifier on the address operand of `ld` instruction, otherwise the behavior is undefined.
## 5.5 Tensors
A tensor is a multi-dimensional matrix structure in the memory. Tensor is defined by the following properties:
- Dimensionality
- Dimension sizes across each dimension
- Individual element types
- Tensor stride across each dimension

PTX supports instructions which can operate on the tensor data. PTX Tensor instructions include:
- Copying data between global and shared memories
- Reducing the destination tensor data with the source.

The Tensor data can be operated on by various `wmma.mma`, `mma` and `wgmma.mma_async` instructions.

PTX Tensor instructions treat the tensor data in the global memory as a multi-dimensional structure and treat the data in the shared memory as a linear data. ( tensor指令视global memory中的tensor数据为多维结构，视shared memory中的tensor为线性数据 )
### 5.5.1 Tensor Dimension, size and format
Tensors can have dimensions: 1D, 2D, 3D, 4D or 5D.

Each dimension has a size which represents the number of elements along the dimension. The elements can have one the following types:
- Bit-sized type: `.b32`, `.b64`
- Integer: `.u8`, `.u16`, `.u32`, `.s32`, `.u64`, `.s64`
- Floating point and alternate floating point: `.f16`, `.bf16`, `.tf32`, `.f32`, `.f64` (rounded to nearest even).
    
Tensor can have padding at the end in each of the dimensions to provide alignment for the data in the subsequent dimensions. Tensor stride can be used to specify the amount of padding in each dimension. ( tensor stride可以用于指定tensor每一维度的padding )
### 5.5.2 Tensor Access Modes
Tensor data can be accessed in two modes:
- Tiled mode:
    In tiled mode, the source multi-dimensional tensor layout is preserved at the destination. ( 源tensor的多维布局被保留 )
- Im2col mode:
    In im2col mode, the elements in the Bounding Box of the source tensor are rearranged into columns at the destination. Refer [here](https://in.mathworks.com/help/images/ref/im2col.html) for more details. 
### 5.5.3 Tiled Mode
#### 5.3.3.1 Bounding Box
A tensor can be accessed in chunks known as _Bounding Box_. The Bounding Box has the same dimensionality as the tensor they are accessing into. Size of each bounding Box must be a multiple of 16 bytes. The address of the bounding Box must also be aligned to 16 bytes. ( bounding box的维度和tensor相同，大小为16字节的倍数，地址与16字节对齐 )

Bounding Box has the following access properties:
- Bounding Box dimension sizes
- Out of boundary access mode
- Traversal strides

The tensor-coordinates, specified in the PTX tensor instructions, ( PTX tensor指令中会指定tensor坐标 ) specify the starting offset of the bounding box. ( 该坐标指明了bounding box的起始偏移 ) Starting offset of the bounding box along with the rest of the bounding box information together are used to determine the elements which are to be accessed. ( bounding box的起始偏移和bounding box本身的信息用于定位需要访问的元素 )
#### 5.5.3.2 Traversal-Stride
While the Bounding Box is iterating the tensor across a dimension, the traversal stride specifies the exact number of elements to be skipped. ( traversal stride 指定需要跳过的元素数量 ) If no jump over is required, default value of 1 must be specified.

The traversal stride in dimension 0 can be used for the [Interleave layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-interleaved-layout). For non-interleaved layout, the traversal stride in dimension 0 must always be 1.

[Figure 5](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode-bb-example) illustrates tensor, tensor size, tensor stride, Bounding Box size and traversal stride.
#### 5.5.3.3 Out of Boundary Access
PTX Tensor operation can detect and handle the case when the Bounding Box crosses the tensor boundary in any dimension. ( tensor操作会检查并处理bounding box在任意维度越过tensor界的情况 ) There are 2 modes:
- Zero fill mode:
    Elements in the Bounding Box which fall outside of the tensor boundary are set to 0.
- `OOB-NaN` fill mode:
    Elements in the Bounding Box which fall outside of the tensor boundary are set to a special NaN called `OOB-NaN`.

[Figure 6](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-oob-access) shows an example of the out of boundary access.
### 5.5.6 Swizzling Modes
The layout of the data in the shared memory can be different to that of global memory, for access performance reasons. The following describes various swizzling modes:
- No swizzle mode:
    There is no swizzling in this mode and the destination data layout is exactly similar to the source data layout.
# 6 Instruction Operands
## 6.1 Operand Type Information
All operands in instructions have a known type from their declarations. Each operand type must be compatible with the type determined by the instruction template and instruction type. There is no automatic conversion between types. ( 类型间没有自动转换 )

The bit-size type is compatible with every type having the same size. Integer types of a common size are compatible with each other. Operands having type different from but compatible with the instruction type are silently cast to the instruction type. ( 但兼容的类型可以转换 )
## 6.2 Source Operands
The source operands are denoted in the instruction descriptions by the names `a`, `b`, and `c`. ( 源操作数的名称只会是 `a` , `b`, `c` ) PTX describes a load-store machine, so operands for ALU instructions must all be in variables declared in the `.reg` register state space. For most operations, the sizes of the operands must be consistent.

The `cvt` (convert) instruction takes a variety of operand types and sizes, as its job is to convert from nearly any data type to any other data type (and size).

The `ld`, `st`, `mov`, and `cvt` instructions copy data from one location to another. Instructions `ld` and `st` move data from/to addressable state spaces to/from registers. ( `ld` , `st` 指令用于在可寻址的状态空间和寄存器之间移动数据 ) The `mov` instruction copies data between registers. ( `mov` 指令用于在寄存器之间拷贝数据 )

Most instructions have an optional predicate guard that controls conditional execution, and a few instructions have additional predicate source operands. ( 额外的谓词源操作数 ) Predicate operands are denoted by the names `p`, `q`, `r`, `s`. ( 谓词操作数的名称只能是 `p q r s` )
## 6.3 Destination Operands
PTX instructions that produce a single result store the result in the field denoted by `d` (for destination) in the instruction descriptions. ( 目标操作数的名称为 `d` ) The result operand is a scalar or vector variable in the register state space.
## 6.4 Using Addresses, Arrays, and Vectors
### 6.4.1 Addresses as Operands
All the memory instructions take an address operand that specifies the memory location being accessed. This addressable operand is one of:

`[var]`
the name of an addressable variable `var`.

`[reg]`
an integer or bit-size type register `reg` containing a byte address.

`[reg+immOff]`
a sum of register `reg` containing a byte address plus a constant integer byte offset (signed, 32-bit).

`[var+immOff]`
a sum of address of addressable variable `var` containing a byte address plus a constant integer byte offset (signed, 32-bit).

`[immAddr]`
an immediate absolute byte address (unsigned, 32-bit).

`var[immOff]`
an array element as described in [Arrays as Operands](https://docs.nvidia.com/cuda/parallel-thread-execution/#arrays-as-operands).

The address must be naturally aligned to a multiple of the access size. If an address is not properly aligned, the resulting behavior is undefined. For example, among other things, the access may proceed by silently masking off low-order address bits to achieve proper rounding, or the instruction may fault.

Addresses are zero-extended to the specified width as needed, and truncated if the register width exceeds the state space address width for the target architecture.

The `mov` instruction can be used to move the address of a variable into a pointer. ( 将变量的地址移动至一个指针 )The address is an offset in the state space in which the variable is declared. Load and store operations move data between registers and locations in addressable state spaces. The syntax is similar to that used in many assembly languages, where scalar variables are simply named and addresses are de-referenced by enclosing the address expression in square brackets. Address expressions include variable names, address registers, address register plus byte offset, and immediate address expressions which evaluate at compile-time to a constant address.
#### 6.4.1.1 Generic Addressing
If a memory instruction does not specify a state space, the operation is performed using generic addressing. ( 若存储相关指令没有指定状态空间，操作使用通用寻址 ) The state spaces `.const`, [Kernel Function Parameters](https://docs.nvidia.com/cuda/parallel-thread-execution/#kernel-function-parameters) (`.param`), `.local` and `.shared` are modeled as windows within the generic address space. ( 状态空间 `.const, .param, .local, .shared` 被建模为通用地址空间内的窗口 ) Each window is defined by a window base and a window size that is equal to the size of the corresponding state space. ( 窗口有对应的基地址和大小 ) A generic address maps to `global` memory unless it falls within the window for `const`, `local`, or `shared` memory. The [Kernel Function Parameters](https://docs.nvidia.com/cuda/parallel-thread-execution/#kernel-function-parameters) (`.param`) window is contained within the `.global` window. Within each window, a generic address maps to an address in the underlying state space by subtracting the window base from the generic address.
### 6.4.2 Arrays as Operands
Arrays of all types can be declared, and the identifier becomes an address constant in the space where the array is declared. ( 数组被声明时，其标识符成为了状态空间内的一个地址常量 ) The size of the array is a constant in the program. ( 数组的大小是常量 )

Array elements can be accessed using an explicitly calculated byte address, or by indexing into the array using square-bracket notation. The expression within square brackets is either a constant integer, a register variable, or a simple _register with constant offset_ expression, where the offset is a constant expression that is either added or subtracted from a register variable. ( 括号内的索引可以是常整数、寄存器变量、寄存器加上整数偏移的表达式 ) If more complicated indexing is desired, it must be written as an address calculation prior to use. Examples are:
### 6.4.3 Vectors as Operands
Vector operands are supported by a limited subset of instructions, ( 仅一部分指令支持向量操作数 ) which include `mov`, `ld`, `st`, `atom`, `red` and `tex`. Vectors may also be passed as arguments to called functions.

Vector elements can be extracted from the vector with the suffixes `.x`, `.y`, `.z` and `.w`, as well as the typical color fields `.r`, `.g`, `.b` and `.a`. ( 向量元素可以通过后缀提取 )

A brace-enclosed list is used for pattern matching to pull apart vectors.

Vector loads and stores can be used to implement wide loads and stores, which may improve memory performance. ( 向量存取可以用于实现宽存取，以提高性能 ) The registers in the load/store operations can be a vector, or a brace-enclosed list of similarly typed scalars. Here are examples: ( load store 操作中的寄存器可以是向量，或括号包裹的类似类型的标量 )

Elements in a brace-enclosed vector, say {Ra, Rb, Rc, Rd}, correspond to extracted elements as follows:
```
Ra = V.x = V.r
Rb = V.y = V.g
Rc = V.z = V.b
Rd = V.w = V.a
```
### 6.4.4 Labels and Function Names as Operands
Labels and function names can be used only in `bra`/`brx.idx` and `call` instructions respectively. Function names can be used in `mov` instruction to get the address of the function into a register, for use in an indirect call.

Beginning in PTX ISA version 3.1, the `mov` instruction may be used to take the address of kernel functions, to be passed to a system call that initiates a kernel launch from the GPU. This feature is part of the support for CUDA Dynamic Parallelism. See the _CUDA Dynamic Parallelism Programming Guide_ for details.
## 6.5 Type Conversion
All operands to all arithmetic, logic, and data movement instruction must be of the same type and size, ( 指令的两个操作数需要具有相同类型和大小 ) except for operations where changing the size and/or type is part of the definition of the instruction. Operands of different sizes or types must be converted prior to the operation.
### 6.5.1 Scalar Conversion
[Table 15](https://docs.nvidia.com/cuda/parallel-thread-execution/#scalar-conversions-convert-instruction-precision-and-format) shows what precision and format the cvt instruction uses given operands of differing types. For example, if a `cvt.s32.u16` instruction is given a `u16` source operand and `s32` as a destination operand, the `u16` is zero-extended to `s32`.

Conversions to floating-point that are beyond the range of floating-point numbers are represented with the maximum floating-point value (IEEE 754 Inf for `f32` and `f64`, and ~131,000 for `f16`).
### 6.5.2 Rounding Modifiers
Conversion instructions may specify a rounding modifier. ( 舍入修饰符 ) In PTX, there are four integer rounding modifiers and four floating-point rounding modifiers. [Table 16](https://docs.nvidia.com/cuda/parallel-thread-execution/#rounding-modifiers-floating-point-rounding-modifiers) and [Table 17](https://docs.nvidia.com/cuda/parallel-thread-execution/#rounding-modifiers-integer-rounding-modifiers) summarize the rounding modifiers.
## 6.6 Operand Costs
Operands from different state spaces affect the speed of an operation. Registers are fastest, while global memory is slowest. Much of the delay to memory can be hidden in a number of ways. The first is to have multiple threads of execution so that the hardware can issue a memory operation and then switch to other execution. Another way to hide latency is to issue the load instructions as early as possible, as execution is not blocked until the desired result is used in a subsequent (in time) instruction. The register in a store operation is available much more quickly. [Table 18](https://docs.nvidia.com/cuda/parallel-thread-execution/#operand-costs-cost-estimates-for-sccessing-state-spaces) gives estimates of the costs of using different kinds of memory.
# 7 Abstracting the ABI
Rather than expose details of a particular calling convention, stack layout, and Application Binary Interface (ABI),( ABI指 PTX 语言与底层硬件之间的二进制接口规范 ) PTX provides a slightly higher-level abstraction and supports multiple ABI implementations. In this section, we describe the features of PTX needed to achieve this hiding of the ABI. These include syntax for function definitions, function calls, parameter passing, support for variadic functions (`varargs`), and memory allocated on the stack (`alloca`).
## 7.1 Function Declarations and Definitions
In PTX, functions are declared and defined using the `.func` directive. A function _declaration_ specifies an optional list of return parameters, the function name, and an optional list of input parameters; ( 函数声明指定了可选的返回参数列表、函数名、可选的输入参数列表 ) together these specify the function’s interface, or prototype. A function _definition_ specifies both the interface and the body of the function. A function must be declared or defined prior to being called. ( 函数定义指定其接口和函数体，函数被定义前，需要被声明或被定义 )

The simplest function has no parameters or return values, and is represented in PTX as follows:
```
.func foo
{
    ...
    ret;
}

    ...
    call foo;
    ...
```
Here, execution of the `call` instruction transfers control to `foo`, implicitly saving the return address. ( `call` 指令转移控制权，并隐式保留返回地址 ) Execution of the `ret` instruction within `foo` transfers control to the instruction following the call. ( `ret` 指令将控制权转移给函数调用之后的一个指令 )

Scalar and vector base-type input and return parameters may be represented simply as register variables. At the call, arguments may be register variables or constants, and return values may be placed directly into register variables. ( 传入函数的实参可以是寄存器变量或常量，返回值可以直接放入寄存器变量 ) The arguments and return variables at the call must have type and size that match the callee’s corresponding formal parameters. ( 实参和形参的类型和大小要匹配 )

When using the ABI, `.reg` state space parameters must be at least 32-bits in size. Subword scalar objects in the source language should be promoted to 32-bit registers in PTX, or use `.param` state space byte arrays described next.

Objects such as C structures and unions are flattened into registers or byte arrays in PTX and are represented using `.param` space memory. ( 类似结构体和联合体的对象会被展平为寄存器或字节数组，并用 `.param` 状态空间表示 ) For example, consider the following C structure, passed by value to a function:
```
struct {
    double dbl;
    char   c[4];
};
```
In PTX, this structure will be flattened into a byte array. Since memory accesses are required to be aligned to a multiple of the access size, the structure in this example will be a 12 byte array with 8 byte alignment so that accesses to the `.f64` field are aligned. The `.param` state space is used to pass the structure by value:
```
.func (.reg .s32 out) bar (.reg .s32 x, .param .align 8 .b8 y[12])
{
    .reg .f64 f1;
    .reg .b32 c1, c2, c3, c4;
    ...
    ld.param.f64 f1, [y+0];
    ld.param.b8  c1, [y+8];
    ld.param.b8  c2, [y+9];
    ld.param.b8  c3, [y+10];
    ld.param.b8  c4, [y+11];
    ...
    ... // computation using x,f1,c1,c2,c3,c4;
}

{
     .param .b8 .align 8 py[12];
     ...
     st.param.b64 [py+ 0], %rd;
     st.param.b8  [py+ 8], %rc1;
     st.param.b8  [py+ 9], %rc2;
     st.param.b8  [py+10], %rc1;
     st.param.b8  [py+11], %rc2;
     // scalar args in .reg space, byte array in .param space
     call (%out), bar, (%x, py);
     ...
```
In this example, note that `.param` space variables are used in two ways. First, a `.param` variable `y` is used in function definition bar to represent a formal parameter. Second, a `.param` variable `py` is declared in the body of the calling function and used to set up the structure being passed to bar.

The following is a conceptual way to think about the `.param` state space use in device functions.

For a caller,
- The `.param` state space is used to set values that will be passed to a called function and/or to receive return values from a called function. ( 对于调用者， `.param` 状态空间用于设定要向被调用函数传递的值，以及用于接受被调用函数的返回值 ) Typically, a `.param` byte array is used to collect together fields of a structure being passed by value. ( `.param` 字节数组用于存储值传递的结构体的域 )
For a callee,
- The `.param` state space is used to receive parameter values and/or pass return values back to the caller. ( 接受参数值以及传递返回值 )
    
The following restrictions apply to parameter passing.
For a caller,
- Arguments may be `.param` variables, `.reg` variables, or constants.
- In the case of `.param` space formal parameters that are byte arrays, the argument must also be a `.param` space byte array with matching type, size, and alignment. A `.param` argument must be declared within the local scope of the caller. ( 形参为 `.param` 字节数组 ， 实参必须为 `.param` 字节数组 )
- In the case of `.param` space formal parameters that are base-type scalar or vector variables, the corresponding argument may be either a `.param` or `.reg` space variable with matching type and size, or a constant that can be represented in the type of the formal parameter. ( 形参为 `.param` 基础类型标量或向量变量，实参可以为 `.reg` ，也可以为常量 )
- In the case of `.reg` space formal parameters, the corresponding argument may be either a `.param` or `.reg` space variable of matching type and size, or a constant that can be represented in the type of the formal parameter. ( 形参为 `.reg` 实参可以为 `.param`，也可以为标量 )
- In the case of `.reg` space formal parameters, the register must be at least 32-bits in size. ( `.reg` 为形参，寄存器必须至少32位 )
- All `st.param` instructions used for passing arguments to function call must immediately precede the corresponding `call` instruction and `ld.param` instruction used for collecting return value must immediately follow the `call` instruction without any control flow alteration. ( 传递参数的 `st` 和 收集返回值的 `ld` 必须和 `call` 紧挨 )`st.param` and `ld.param` instructions used for argument passing cannot be predicated. This enables compiler optimization and ensures that the `.param` variable does not consume extra space in the caller’s frame beyond that needed by the ABI. The `.param` variable simply allows a mapping to be made at the call site between data that may be in multiple locations (e.g., structure being manipulated by caller is located in registers and memory) to something that can be passed as a parameter or return value to the callee.

For a callee,
- Input and return parameters may be `.param` variables or `.reg` variables.
- Parameters in `.param` memory must be aligned to a multiple of 1, 2, 4, 8, or 16 bytes.
- Parameters in the `.reg` state space must be at least 32-bits in size. ( `.reg` 参数至少32位 )
- The `.reg` state space can be used to receive and return base-type scalar and vector values, including sub-word size objects when compiling in non-ABI mode. Supporting the `.reg` state space provides legacy support.

Note that the choice of `.reg` or `.param` state space for parameter passing has no impact on whether the parameter is ultimately passed in physical registers or on the stack.  ( 选择 `.reg` 或 `.param` 状态空间对物理上的参数传递位置没有影响 ) The mapping of parameters to physical registers and stack locations depends on the ABI definition and the order, size, and alignment of parameters.
## 7.3 Alloca
PTX provides `alloca` instruction for allocating storage at runtime on the per-thread local memory stack. ( `alloc` 指令用于在运行时在各线程的local memory栈上分配空间 ) The allocated stack memory can be accessed with `ld.local` and `st.local` instructions using the pointer returned by `alloca`. 

In order to facilitate deallocation of memory allocated with `alloca`, PTX provides two additional instructions: `stacksave` which allows reading the value of stack pointer in a local variable, and `stackrestore` which can restore the stack pointer with the saved value.

`alloca`, `stacksave`, and `stackrestore` instructions are described in [Stack Manipulation Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#stack-manipulation-instructions).
# 8 Memory Consisitency Model
The axioms introduced by the memory consistency model specify exactly which contradictions are forbidden between the orders observed by different threads.
## 8.1 Scope and applicability of the model
The constraints specified under this model apply to PTX programs with any PTX ISA version number, running on `sm_70` or later architectures.

The memory consistency model does not apply to texture (including `ld.global.nc`) and surface accesses.
## 8.2 Memory operations
The fundamental storage unit in the PTX memory model is a byte, consisting of 8 bits. Each state space available to a PTX program is a sequence of contiguous bytes in memory. ( 每个状态空间对于PTX程序就是存储中的连续字节序列 ) Every byte in a PTX state space has a unique address relative to all threads that have access to the same state space.

Each PTX memory instruction specifies an address operand and a data type. The address operand contains a virtual address that gets converted to a physical address during memory access. The physical address and the size of the data type together define a physical memory location, which is the range of bytes starting from the physical address and extending up to the size of the data type in bytes.

The memory consistency model specification uses the terms “address” or “memory address” to indicate a virtual address, and the term “memory location” to indicate a physical memory location. ( memory address表示虚拟地址，memory location表示物理地址 )

Each PTX memory instruction also specifies the operation — either a read, a write or an atomic read-modify-write — to be performed on all the bytes in the corresponding memory location.
### 8.2.1 Overlap
Two memory locations are said to overlap when the starting address of one location is within the range of bytes constituting the other location. ( 物理地址有交叉即overlap ) Two memory operations are said to overlap when they specify the same virtual address and the corresponding memory locations overlap. ( 存储操作指定了相同虚拟地址即overlap ) The overlap is said to be complete when both memory locations are identical, and it is said to be partial otherwise.
### 8.2.2 Aliases
Two distinct virtual addresses are said to be aliases if they map to the same memory location. ( 两个虚拟地址指向相同物理位置 )
### 8.2.3 Multimem Addresses
A multimem address is a virtual address which points to multiple distinct memory locations across devices. ( 指向跨设备的多个物理地址的虚拟地址 )
### 8.2.4 Memory Operations on Vector Data Types
The memory consistency model relates operations executed on memory locations with scalar data types, which have a maximum size and alignment of 64 bits. ( 标量数据类型最大大小和对齐为64位 ) Memory operations with a vector data type are modelled as a set of equivalent memory operations with a scalar data type, executed in an unspecified order on the elements in the vector. ( 对向量数据类型的存储操作被建模为一系列对标量数据类型的存储操作，其执行顺序不确定 )
### 8.2.5 Memory Operations on Packed Data Types
A packed data type consists of two values of the same scalar data type, ( 相同类型的两个值 ) as described in [Packed Data Types](https://docs.nvidia.com/cuda/parallel-thread-execution/#packed-data-types). These values are accessed in adjacent memory locations. ( 其物理位置相邻 ) A memory operation on a packed data type is modelled as a pair of equivalent memory operations on the scalar data type, ( 建模一对等价的对标量数据类型的存储操作 ) executed in an unspecified order on each element of the packed data.
## 8.4 Operation types
![[PTX ISA-Table 19.png]]
## 8.5 Scope
Each _strong_ operation must specify a _scope_, which is the set of threads that may interact directly with that operation and establish any of the relations ( 直接与操作交互的线程集合 ) described in the memory consistency model. There are four scopes:
![[PTX ISA-Table 20.png]]

Note that the warp is not a _scope_; the CTA is the smallest collection of threads that qualifies as a _scope_ in the memory consistency model.
## 8.6 Proxies
A _memory proxy_, or a _proxy_ is an abstract label applied to a method of memory access. ( 存储代理是对存储访问方法的一个抽象标签 ) When two memory operations use distinct methods of memory access, they are said to be different _proxies_. ( 当两个存储操作使用不同的存储访问方法，它们就是不同的代理 )

Memory operations as defined in [Operation types](https://docs.nvidia.com/cuda/parallel-thread-execution/#operation-types) use _generic_ method of memory access, i.e. a _generic proxy_. Other operations such as textures and surfaces all use distinct methods of memory access, also distinct from the _generic_ method. ( 通常的存储操作使用的存储访问是通用方法，因此是通用代理 )

A _proxy fence_ is required to synchronize memory operations across different _proxies_.  ( 不同代理之间的存储操作需要用proxy fence同步 ) Although virtual aliases use the _generic_ method of memory access, since using distinct virtual addresses behaves as if using different _proxies_, they require a _proxy fence_ to establish memory ordering. 
## 8.7 Morally strong operations
Two operations are said to be _morally strong_ relative to each other ( morally 强相关 ) if they satisfy all of the following conditions:
1. The operations are related in _program order_ (i.e, they are both executed by the same thread), or each operation is _strong_ and specifies a _scope_ that includes the thread executing the other operation. ( 两个操作由同一线程执行，即程序顺序上相关；或具有域内线程由交叉的两个强操作 )
2. Both operations are performed via the same _proxy_.
3. If both are memory operations, then they overlap completely.

Most (but not all) of the axioms in the memory consistency model depend on relations between _morally strong_ operations.
## 8.8 Release and Acquire Patterns
Some sequences of instructions give rise to patterns that participate in memory synchronization as described later. The _release_ pattern makes prior operations from the current thread visible to some operations from other threads. ( 让当前线程先前的操作对其他线程的一些操作可见 ) The _acquire_ pattern makes some operations from other threads visible to later operations from the current thread. ( 让其他线程的一些操作对当前线程的后续操作可见 )
## 8.9 Ordering of memroy operations
The sequence of operations performed by each thread is captured as _program order_ while _memory synchronization_ across threads is captured as _causality order_. ( 每个线程的操作执行顺序即程序顺序，线程之间的存储同步即因果顺序 ) The visibility of the side-effects of memory operations to other memory operations is captured as _communication order_. ( 存储操作的副作用对其他存储操作的可见性即通讯顺序 ) The memory consistency model defines contradictions that are disallowed between communication order on the one hand, and _causality order_ and _program order_ on the other.
### 8.9.1 Program Order
The _program order_ relates all operations performed by a thread to the order in which a sequential processor will execute instructions in the corresponding PTX source. It is a transitive relation ( 传递关系 ) that forms a total order over the operations performed by the thread, but does not relate operations from different threads. ( 程序顺序不会关联不同线程的操作 )
#### 8.9.1.1 Asynchronous Operations
Some PTX instructions (all variants of `cp.async`, `cp.async.bulk`, `cp.reduce.async.bulk`, `wgmma.mma_async`) perform operations that are asynchronous to the thread that executed the instruction. ( 对于执行线程异步的操作 ) These asynchronous operations are ordered after prior instructions in the same thread (except in the case of `wgmma.mma_async`), but they are not part of the program order for that thread. Instead, they provide weaker ordering guarantees as documented in the instruction description.

For example, the loads and stores performed as part of a `cp.async` are ordered with respect to each other, but not to those of any other `cp.async` instructions initiated by the same thread, nor any other instruction subsequently issued by the thread with the exception of `cp.async.commit_group` or `cp.async.mbarrier.arrive`. The asynchronous mbarrier [arrive-on](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation performed by a `cp.async.mbarrier.arrive` instruction is ordered with respect to the memory operations performed by all prior `cp.async` operations initiated by the same thread, but not to those of any other instruction issued by the thread. The implicit mbarrier [complete-tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation) operation that is part of all variants of `cp.async.bulk` and `cp.reduce.async.bulk` instructions is ordered only with respect to the memory operations performed by the same asynchronous instruction, and in particular it does not transitively establish ordering with respect to prior instructions from the issuing thread.
### 8.9.4 Memory synchronization
Synchronizing operations performed by different threads synchronize with each other at runtime as described here. The effect of such synchronization is to establish _causality order_ across threads. ( 同步操作为线程之间建立因果关系 )
1. A _fence.sc_ operation X _synchronizes_ with a _fence.sc_ operation Y if X precedes Y in the _Fence-SC_ order.
2. A _bar{.cta}.sync_ or _bar{.cta}.red_ or _bar{.cta}.arrive_ operation _synchronizes_ with a _bar{.cta}.sync_ or _bar{.cta}.red_ operation executed on the same barrier.
3. A `barrier.cluster.arrive` operation synchronizes with a `barrier.cluster.wait` operation.
4. A _release_ pattern X _synchronizes_ with an _acquire_ pattern Y, if a _write_ operation in X precedes a _read_ operation in Y in _observation order_, and the first operation in X and the last operation in Y are _morally strong_.
# 9 Instruction Set
## 9.2 PTX Instructions
PTX instructions generally have from zero to four operands, plus an optional guard predicate appearing after an `@` symbol to the left of the `opcode`:  ( 0-4个操作数，以及一个可选的守护谓词 )
- `@p   opcode;`
- `@p   opcode a;`
- `@p   opcode d, a;`
- `@p   opcode d, a, b;`
- `@p   opcode d, a, b, c;`

For instructions that create a result value, the `d` operand is the destination operand, while `a`, `b`, and `c` are source operands. ( 对于创建返回值的指令， `d` 操作数会作为目标操作数，其余为源操作数 )

The `setp` instruction writes two destination registers. We use a `|` symbol to separate multiple destination registers.
```
setp.lt.s32  p|q, a, b;  // p = (a < b); q = !(a < b);
```

For some instructions the destination operand is optional. A _bit bucket_ operand denoted with an underscore (`_`) may be used in place of a destination register.
## 9.3 Predicated Execution
In PTX, predicate registers are virtual and have `.pred` as the type specifier. ( 谓词寄存器为虚拟的， `.pred` 为类型标识符 ) So, predicate registers can be declared as
```
.reg .pred p, q, r;
```

All instructions have an optional _guard predicate_ which controls conditional execution of the instruction. The syntax to specify conditional execution is to prefix an instruction with `@{!}p`, where `p` is a predicate variable, optionally negated. Instructions without a guard predicate are executed unconditionally. ( 没有守护谓词的指令会被无条件执行 )

Predicates are most commonly set as the result of a comparison performed by the `setp` instruction.

As an example, consider the high-level code
```
if (i < n)
    j = j + 1;
```
This can be written in PTX as
```
      setp.lt.s32  p, i, n;    // p = (i < n)
@p    add.s32      j, j, 1;    // if i < n, add 1 to j
```

To get a conditional branch or conditional function call, use a predicate to control the execution of the branch or call instructions.  ( 谓词同样可以控制分支和函数调用指令 ) To implement the above example as a true conditional branch, the following PTX instruction sequence might be used:
```
      setp.lt.s32  p, i, n;    // compare i to n
@!p   bra  L1;                 // if False, branch over
      add.s32      j, j, 1;
L1:     ...
```
### 9.3.1 Comparisons
#### 9.3.1.1 Integer and Bit-Size Comparisions
The signed integer comparisons are the traditional `eq` (equal), `ne` (not-equal), `lt` (less-than), `le` (less-than-or-equal), `gt` (greater-than), and `ge` (greater-than-or-equal). The unsigned comparisons are `eq`, `ne`, `lo` (lower), `ls` (lower-or-same), `hi` (higher), and `hs` (higher-or-same). The bit-size comparisons are `eq` and `ne`; ordering comparisons are not defined for bit-size types.
![[PTX ISA-Table 21.png]]
#### 9.3.1.2 Floating Point Comparisions
The ordered floating-point comparisons are `eq`, `ne`, `lt`, `le`, `gt`, and `ge`. If either operand is `NaN`, the result is `False`. [Table 22](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-comparisons-floating-point-operators) lists the floating-point comparison operators.

To aid comparison operations in the presence of `NaN` values, unordered floating-point comparisons are provided: `equ`, `neu`, `ltu`, `leu`, `gtu`, and `geu`. If both operands are numeric values (not `NaN`), then the comparison has the same result as its ordered counterpart. If either operand is `NaN`, then the result of the comparison is `True`.

To test for `NaN` values, two operators `num` (`numeric`) and `nan` (`isNaN`) are provided. `num` returns `True` if both operands are numeric values (not `NaN`), and `nan` returns `True` if either operand is `NaN`. [Table 24](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-comparisons-floating-point-operators-testing-nan) lists the floating-point comparison operators testing for `NaN` values.
### 9.3.2 Manipulating Predicates
Predicate values may be computed and manipulated using the following instructions: `and`, `or`, `xor`, `not`, and `mov`.

There is no direct conversion between predicates and integer values, and no direct way to load or store predicate register values. However, `setp` can be used to generate a predicate from an integer, ( `setp` 用于生成谓词 ) and the predicate-based select (`selp`) instruction ( 基于谓词选择 ) can be used to generate an integer value based on the value of a predicate; for example:
```
selp.u32 %r1,1,0,%p;    // convert predicate to 32-bit value
```
## 9.4 Type Information for Instructions and Operands
Typed instructions must have a type-size modifier. ( 有类型的指令需要有一个类型大小修饰符 ) For example, the `add` instruction requires type and size information to properly perform the addition operation (signed, unsigned, float, different sizes), and this information must be specified as a suffix to the opcode. ( 这些信息作为opcode的后缀注明 )
```
.reg .u16 d, a, b;

add.u16 d, a, b;    // perform a 16-bit unsigned add
```

Some instructions require multiple type-size modifiers, most notably the data conversion instruction `cvt`. ( 一些指令需要多个类型大小修饰符 ) It requires separate type-size modifiers for the result and source, and these are placed in the same order as the operands. ( 修饰符的顺序和对应的操作数一致 ) For example:
```
.reg .u16 a;
.reg .f32 d;

cvt.f32.u16 d, a;   // convert 16-bit unsigned to 32-bit float
```

n general, an operand’s type must agree with the corresponding instruction-type modifier. The rules for operand and instruction type conformance are as follows:
- Bit-size types agree with any type of the same size.
- Signed and unsigned integer types agree provided they have the same size, and integer operands are silently cast to the instruction type if needed. For example, an unsigned integer operand used in a signed integer instruction will be treated as a signed integer by the instruction.
- Floating-point types agree only if they have the same size; i.e., they must match exactly.

Some operands have their type and size defined independently from the instruction type-size. For example, the shift amount operand ( 位移量操作数 ) for left and right shift instructions always has type `.u32`, while the remaining operands have their type and size determined by the instruction type.
```
// 64-bit arithmetic right shift; shift amount 'b' is .u32
shr.s64 d,a,b;
```
### 9.4.1 Operand Size Exceeding Instruction-Type Size
For convenience, `ld`, `st`, and `cvt` instructions permit source and destination data operands to be wider than the instruction-type size, ( 允许源和目标操作数比指令类型大小更宽 ) so that narrow values may be loaded, stored, and converted using regular-width registers. For example, 8-bit or 16-bit values may be held directly in 32-bit or 64-bit registers when being loaded, stored, or converted to other types and sizes. The operand type checking rules are relaxed for bit-size and integer (signed and unsigned) instruction types; floating-point instruction types still require that the operand type-size matches exactly, unless the operand is of bit-size type. 

When a source operand has a size that exceeds the instruction-type size, the source data is truncated (chopped) to the appropriate number of bits specified by the instruction type-size.

When a destination operand has a size that exceeds the instruction-type size, the destination data is zero- or sign-extended to the size of the destination register. ( 目标寄存器的大小超过了指令类型大小，则目标数据会被拓展 ) If the corresponding instruction type is signed integer, the data is sign-extended; otherwise, the data is zero-extended.
## 9.5 Divergence of Threads in Control Constructs
Threads in a CTA execute together, at least in appearance, until they come to a conditional control construct such as a conditional branch, conditional function call, or conditional return. If threads execute down different control flow paths, the threads are called _divergent_. If all of the threads act in unison and follow a single control flow path, the threads are called _uniform_. Both situations occur often in programs.

A CTA with divergent threads may have lower performance than a CTA with uniformly executing threads, so it is important to have divergent threads re-converge as soon as possible. All control constructs are assumed to be divergent points unless the control-flow instruction is marked as uniform, ( 所有的控制结构都被认为是 divergent 点，除非控制流指令被 `.uni` 后缀标记为 uniform ) using the `.uni` suffix. For divergent control flow, the optimizing code generator automatically determines points of re-convergence. Therefore, a compiler or code author targeting PTX can ignore the issue of divergent threads, but has the opportunity to improve performance by marking branch points as uniform when the compiler or author can guarantee that the branch point is non-divergent. ( 如果可以保证分支点不发生diverge，可以通过标记uniform提高性能 )
## 9.6 Semantics
The goal of the semantic description of an instruction is to describe the results in all cases in as simple language as possible. The semantics are described using C, until C is not expressive enough.
### 9.6.1 Machine-Specific Semantics of 16-bit Code
A PTX program may execute on a GPU with either a 16-bit or a 32-bit data path. When executing on a 32-bit data path, 16-bit registers in PTX are mapped to 32-bit physical registers, ( PTX内16位寄存器被映射到32位物理寄存器 ) and 16-bit computations are _promoted_ to 32-bit computations. ( 16位计算被提升为32位计算 ) This can lead to computational differences between code run on a 16-bit machine versus the same code run on a 32-bit machine, since the promoted computation may have bits in the high-order half-word of registers that are not present in 16-bit physical registers. These extra precision bits can become visible at the application level, for example, by a right-shift instruction.

At the PTX language level, one solution would be to define semantics for 16-bit code that is consistent with execution on a 16-bit data path. This approach introduces a performance penalty for 16-bit code executing on a 32-bit data path, since the translated code would require many additional masking instructions to suppress extra precision bits in the high-order half-word of 32-bit registers.

Rather than introduce a performance penalty for 16-bit code running on 32-bit GPUs, the semantics of 16-bit instructions in PTX is machine-specific. A compiler or programmer may chose to enforce portable, machine-independent 16-bit semantics by adding explicit conversions to 16-bit values at appropriate points in the program to guarantee portability of the code. However, for many performance-critical applications, this is not desirable, and for many applications the difference in execution is preferable to limiting performance.
## 9.7 Instructions
All PTX instructions may be predicated. In the following descriptions, the optional guard predicate is omitted from the syntax.
### 9.7.1 Integer Arithmetic Instructions
Integer arithmetic instructions operate on the integer types in register and constant immediate forms. The integer arithmetic instructions are:
- `add`
- `sub`
- `mul`
- `mad`
- `mul24`
- `mad24`
- `sad`
- `div`
- `rem`
- `abs`
- `neg`
- `min`
- `max`
- `popc`
- `clz`
- `bfind`
- `fns`
- `brev`
- `bfe`
- `bfi`
- `bmsk`
- `szext`
- `dp4a`
- `dp2a`
### 9.7.2 Extended-Precision Integer Arithmetic Instructions
Instructions `add.cc`, `addc`, `sub.cc`, `subc`, `mad.cc` and `madc` reference an implicitly specified condition code register (`CC`) ( 条件码寄存器 ) having a single carry flag bit (`CC.CF`) ( 进位标志位 ) holding carry-in/carry-out or borrow-in/borrow-out. These instructions support extended-precision integer addition, subtraction, and multiplication. No other instructions access the condition code, ( 没有其他的指令会访问该条件码 ) and there is no support for setting, clearing, or testing the condition code. The condition code register is not preserved across calls and is mainly intended for use in straight-line code sequences for computing extended-precision integer addition, subtraction, and multiplication.

The extended-precision arithmetic instructions are:
- `add.cc`, `addc`
- `sub.cc`, `subc`
- `mad.cc`, `madc`
### 9.7.3 Floating-Point Instructions
Floating-point instructions operate on `.f32` and `.f64` register operands and constant immediate values. The floating-point instructions are:
- `testp`
- `copysign`
- `add`
- `sub`
- `mul`
- `fma`
- `mad`
- `div`
- `abs`
- `neg`
- `min`
- `max`
- `rcp`
- `sqrt`
- `rsqrt`
- `sin`
- `cos`
- `lg2`
- `ex2`
- `tanh`

Instructions that support rounding modifiers are IEEE-754 compliant. Double-precision instructions support subnormal inputs and results. Single-precision instructions support subnormal inputs and results by default for `sm_20` and subsequent targets, and flush subnormal inputs and results to sign-preserving zero for `sm_1x` targets. The optional `.ftz` modifier on single-precision instructions provides backward compatibility with `sm_1x` targets by flushing subnormal inputs and results to sign-preserving zero regardless of the target architecture.

Single-precision `add`, `sub`, `mul`, and `mad` support saturation of results to the range [0.0, 1.0], with `NaN`s being flushed to positive zero. `NaN` payloads are supported for double-precision instructions (except for `rcp.approx.ftz.f64` and `rsqrt.approx.ftz.f64`, which maps input `NaN`s to a canonical `NaN`). Single-precision instructions return an unspecified `NaN`. Note that future implementations may support `NaN` payloads for single-precision instructions, so PTX programs should not rely on the specific single-precision `NaN`s being generated.
### 9.7.4 Half Precision Floating-Point Instructions
Half precision floating-point instructions operate on `.f16` and `.f16x2` register operands. The half precision floating-point instructions are:
- `add`
- `sub`
- `mul`
- `fma`
- `neg`
- `abs`
- `min`
- `max`
- `tanh`
- `ex2`

Half-precision `add`, `sub`, `mul`, and `fma` support saturation of results to the range [0.0, 1.0], with `NaN`s being flushed to positive zero. Half-precision instructions return an unspecified `NaN`.
### 9.7.7 Comparison and Selection Instructions
The comparison select instructions are:
- `set`
- `setp`
- `selp`
- `slct`

As with single-precision floating-point instructions, the `set`, `setp`, and `slct` instructions support subnormal numbers for `sm_20` and higher targets and flush single-precision subnormal inputs to sign-preserving zero for `sm_1x` targets. The optional `.ftz` modifier provides backward compatibility with `sm_1x` targets by flushing subnormal inputs and results to sign-preserving zero regardless of the target architecture.
### 9.7.8 Half Precision Comparison Instruction
The comparison instructions are:
- `set`
- `setp`
### 9.7.9 Logic and Shift Instructions
The logic and shift instructions are fundamentally untyped, performing bit-wise operations on operands of any type, provided the operands are of the same size. This permits bit-wise operations on floating point values without having to define a union to access the bits. Instructions `and`, `or`, `xor`, and `not` also operate on predicates.

The logical shift instructions are:
- `and`
- `or`
- `xor`
- `not`
- `cnot`
- `lop3`
- `shf`
- `shl`
- `shr`
### 9.7.10 Data Movement and Conversion Instructions
These instructions copy data from place to place, and from state space to state space, possibly converting it from one format to another. `mov`, `ld`, `ldu`, and `st` operate on both scalar and vector types. The `isspacep` instruction is provided to query whether a generic address falls within a particular state space window. ( 查询是否一个通用地址落在特定的状态空间窗口 ) The `cvta` instruction converts addresses between `generic` and `const`, `global`, `local`, or `shared` state spaces. ( 转换通用地址至状态空间的地址 )

Instructions `ld`, `st`, `suld`, and `sust` support optional cache operations.

The Data Movement and Conversion Instructions are:
- `mov`
- `shfl.sync`
- `prmt`
- `ld`
- `ldu`
- `st`
- `st.async`
- `multimem.ld_reduce`, `multimem.st`, `multimem.red`
- `prefetch`, `prefetchu`
- `isspacep`
- `cvta`
- `cvt`
- `cvt.pack`
- `cp.async`
- `cp.async.commit_group`
- `cp.async.wait_group`, `cp.async.wait_all`
- `cp.async.bulk`
- `cp.reduce.async.bulk`
- `cp.async.bulk.prefetch`
- `cp.async.bulk.tensor`
- `cp.reduce.async.bulk.tensor`
- `cp.async.bulk.prefetch.tensor`
- `cp.async.bulk.commit_group`
- `cp.async.bulk.wait_group`
- `tensormap.replace`
### 9.7.13 Control Flow Instructions
The following PTX instructions and syntax are for controlling execution in a PTX program:
- `{}`
- `@`
- `bra`
- `call`
- `ret`
- `exit`

### 9.7.14 Parallel Synchronization and Communication Instructions
These instructions are:
- `bar{.cta}`, `barrier{.cta}`
- `barrier.cluster`
- `bar.warp.sync`
- `membar`
- `atom`
- `red`
- `red.async`
- `vote`
- `match.sync`
- `activemask`
- `redux.sync`
- `griddepcontrol`
- `elect.sync`
- `mbarrier.init`
- `mbarrier.inval`
- `mbarrier.arrive`
- `mbarrier.arrive_drop`
- `mbarrier.test_wait`
- `mbarrier.try_wait`
- `mbarrier.pending_count`
- `cp.async.mbarrier.arrive`
- `tensormap.cp_fenceproxy`
### 9.7.15 Warp Level Matrix Multiply-Accumulate Instructions
The matrix multiply and accumulate operation has the following form:
```
D = A * B + C
```
where `D` and `C` are called accumulators and may refer to the same matrix.

PTX provides two ways to perform matrix multiply-and-accumulate computation:
- Using `wmma` instructions:
    - This warp-level computation is performed collectively by all threads in the warp as follows:
        - Load matrices A, B and C from memory into registers using the `wmma.load` operation. When the operation completes, the destination registers in each thread hold a fragment of the loaded matrix.
        - Perform the matrix multiply and accumulate operation using the `wmma.mma` operation on the loaded matrices. When the operation completes, the destination registers in each thread hold a fragment of the result matrix returned by the `wmma.mma` operation.
        - Store result Matrix D back to memory using the `wmma.store` operation. Alternately, result matrix D can also be used as argument C for a subsequent `wmma.mma` operation.
        
        The `wmma.load` and `wmma.store` instructions implicitly handle the organization of matrix elements ( 隐式处理矩阵元素的组织 ) when loading the input matrices from memory for the `wmma.mma` operation and when storing the result back to memory.
- Using `mma` instruction:
    - Similar to `wmma`, `mma` also requires computation to be performed collectively by all threads in the warp however distribution of matrix elements across different threads in warp needs to be done explicitly  ( 显式完成矩阵元素的分布 ) before invoking the `mma` operation. The `mma` instruction supports both dense as well as sparse matrix A. The sparse variant can be used when A is a structured sparse matrix as described in [Sparse matrix storage](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-sparse-matrix-storage).
#### 9.7.15.1 Matrix Shape
The matrix multiply and accumulate operations support a limited set of shapes for the operand matrices A, B and C. The shapes of all three matrix operands are collectively described by the tuple `MxNxK`, where A is an `MxK` matrix, B is a `KxN` matrix, while C and D are `MxN` matrices.
#### 9.7.15.2 Matrix Data-types
#### 8.7.15.4 Matrix multiply-accumulate operation using mma instruction
##### 9.7.15.4.1 Matrix Fragments for mma.m8n8k4 with .f16 floating point type
A warp executing `mma.m8n8k4` with .f16 floating point type will compute 4 MMA operations of shape `.m8n8k4`.

Elements of 4 matrices need to be distributed across the threads in a warp. The following table shows distribution of matrices for MMA operations.

| MMA Computation   | Threads participating in MMA computation                        |
| ----------------- | --------------------------------------------------------------- |
| MMA computation 1 | Threads with `%laneid` 0-3 (low group) and 16-19 (high group)   |
| MMA computation 2 | Threads with `%laneid` 4-7 (low group) and 20-23 (high group)   |
| MMA computation 3 | Threads with `%laneid` 8-11 (low group) and 24-27 (high group)  |
| MMA computation 4 | Threads with `%laneid` 12-15 (low group) and 28-31 (high group) |

For each of the individual MMA computation shown above, each of the required thread holds a fragment of the matrix ( 每个线程都保存一个矩阵fragment ) for performing mma operation as follows:

- Multiplicand A:

| .atype | Fragment                                                                                                                                                                                     | Elements (low to high) |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix A. ( 矩阵A的fragment是一个向量表达式，它包含两个 `.f16x2` 寄存器，共存储了矩阵A的4个 `.f16` 元素 ) | a0, a1, a2, a3         |

- Fragment layout for Row Major matrix A is shown in [Figure 22](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-884-a-row-f16). ( 对于row major的A，在MMA computation 1中，fragment为8行4列，T0-T3和T16-T19每个线程占据一行，A中的4个元素a0-a3各占据一列，其余的MMA computation中，各列的元素依然都是a0-a3，但各行的线程不同，共4个MMA computation，用到了warp内全部32个线程 )

The row and column of a matrix fragment can be computed as:
```
row =            %laneid % 4          if %laneid < 16
                (%laneid % 4) + 4     otherwise

col =            i                    for ai where i = {0,..,3}
```

- Fragment layout for Column Major matrix A is shown in [Figure 23](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-884-a-col-f16). ( 对于col major的A，在MMA computation 1中，fragment仍然为8行4列，T0-T3和T16-T19每个线程占据4列，A中的四个元素a0-a3各占据4列中的一行，但各个4列的线程不同 )

The row and column of a matrix fragment can be computed as:
```
row =        i % 4            for ai  where i = {0,..,3}   if %laneid < 16
            (i % 4) + 4       for ai  where i = {0,..,3}   otherwise

col =        %laneid % 4
```

- Multiplicand B:

| .btype | Fragment                                                                                                                                                                   | Elements (low to high) |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix B. ( 同样，矩阵B的fragment是4个 `.f16` 数据，储存在线程中的两个寄存器中 ) | b0, b1, b2, b3         |

- Fragment layout for Row Major matrix B is shown in [Figure 24](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-884-b-row-f16). ( 对于row major的矩阵B，在MMA computation 1中，其fragment排布为4行8列，T0-T3和T16-T19中的每个线程占据一行以及4列，4列分别存储b0-b3 )

```
row =        %laneid % 4

col =         i      for bi   where i = {0,..,3}   if %laneid < 16
             i+4     for bi   where i = {0,..,3}   otherwise
```

- Fragment layout for Column Major matrix B is shown in [Figure 25](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-884-b-col-f16). ( 对于col major的矩阵B，computation 1中，其fragment排布为4行8列，每个线程占据1列，每个元素占据1行 )
```
row =       i                 for bi   where i = {0,..,3}

col =      %laneid % 4        if %laneid < 16
          (%laneid % 4) + 4   otherwise
```

- Accumulators C (or D):

| .ctype / .dtype | Fragment                                                                                                                                                                          | Elements (low to high)         |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `.f16`          | A vector expression containing four `.f16x2` registers, with each register containing two `.f16` elements from the matrix C (or D). ( C 的fragment中包含8个 `.f16` 元素，储存在每个线程的4个寄存器中 ) | c0, c1, c2, c3, c4, c5, c6, c7 |
| `.f32`          | A vector expression of eight `.f32` registers.<br>                                                                                                                                |                                |

- Fragment layout for accumulator matrix when `.ctype` is `.f16` is shown in [Figure 26](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-884-c-f16). ( row major的C的fragment形状为8x8，其中每个线程占据1行，每个元素占据1列 )

##### 9.7.15.4.7. Matrix Fragments for mma.m16n8k8
A warp executing `mma.m16n8k8` will compute an MMA operation of shape `.m16n8k8`.

A:
fragment仍然是4个元素，由两个寄存器保存

The layout of the fragments held by different threads is shown in [Figure 47](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-a-f16). ( fragment为16x8(8x2x8)，每一行4个线程，每一列一个元素)


B:
fragment是2个元素，由1个寄存器保存

The layout of the fragments held by different threads is shown in [Figure 50](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-b-f16).
(fragment为8x8，每一行1个元素，每一列4个线程 )

C:
fragment为4个元素，由两个寄存器保存

The layout of the fragments held by different threads is shown in [Figure 53](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-c-f16-f32).
( fragmnet 为16x8，每一列1个元素，每8行32个线程 )
##### 9.7.15.4.8. Matrix Fragments for mma.m16n8k16 with floating point type
A warp executing `mma.m16n8k16` floating point types will compute an MMA operation of shape `.m16n8k16`.

A:
fragment包含8个元素，占据4个寄存器
The layout of the fragments held by different threads is shown in [Figure 55](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-a-f16).
(fragment为16x16，划分为4个4x4)

B:
fragment包含4个元素，占据2个寄存器
The layout of the fragments held by different threads is shown in [Figure 57](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-b-f16).
(fragment为16x8)

C:
fragment包含4个元素，占据2个寄存器
The layout of the fragments held by different threads is shown in [Figure 59](https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c).
(fragment为16x8)
##### 9.7.15.4.14. Multiply-and-Accumulate Instruction: mma
##### 9.7.15.4.15. Warp-level matrix load instruction: ldmatrix
### 9.7.16 Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions
The warpgroup level matrix multiply and accumulate operation has either of the following forms, where matrix `D` is called accumulator:
- `D = A * B + D`
- `D = A * B`, where the input from accumulator D is disabled.

The `wgmma` instructions perform warpgroup level matrix multiply-and-accumulate operation by having all threads in a warpgroup collectively perform the following actions:
1. Load matrices A, B and D into registers or into shared memory.
2. Perform the following `fence` operations:
    - `wgmma.fence` operations to indicate that the register/shared-memory across the warpgroup have been written into.
    - `fence.proxy.async` operation to make the generic proxy operations visible to the async proxy.
3. Issue the asynchronous matrix multiply and accumulate operations using the `wgmma.mma_async` operation on the input matrices. The `wgmma.mma_async` operation is performed in the async proxy.
4. Create a wgmma-group and commit all the prior outstanding `wgmma.mma_async` operations into the group, by using `wgmma.commit_group` operation.
5. Wait for the completion of the required wgmma-group.
6. Once the wgmma-group completes, all the `wgmma.mma_async` operations have been performed and completed.
### 9.7.17 Stack Manipulation Instructions
The stack manipulation instructions can be used to dynamically allocate and deallocate memory on the stack frame of the current function.

The stack manipulation instrucitons are:
- `stacksave`
- `stackrestore`
- `alloca`
# 10 Special Registers
PTX includes a number of predefined, read-only variables, which are visible as special registers and accessed through `mov` or `cvt` instructions.

The special registers are:
- `%tid`
- `%ntid`
- `%laneid`
- `%warpid`
- `%nwarpid`
- `%ctaid`
- `%nctaid`
- `%smid`
- `%nsmid`
- `%gridid`
- `%is_explicit_cluster`
- `%clusterid`
- `%nclusterid`
- `%cluster_ctaid`
- `%cluster_nctaid`
- `%cluster_ctarank`
- `%cluster_nctarank`
- `%lanemask_eq`, `%lanemask_le`, `%lanemask_lt`, `%lanemask_ge`, `%lanemask_gt`
- `%clock`, `%clock_hi`, `%clock64`
- `%pm0, ..., %pm7`
- `%pm0_64, ..., %pm7_64`
- `%envreg0, ..., %envreg31`
- `%globaltimer`, `%globaltimer_lo`, `%globaltimer_hi`
- `%reserved_smem_offset_begin`, `%reserved_smem_offset_end`, `%reserved_smem_offset_cap`, `%reserved_smem_offset<2>`
- `%total_smem_size`
- `%aggr_smem_size`
- `%dynamic_smem_size`
- `%current_graph_exec`
## 11 Directives
## 11.1 PTX Module Directives
The following directives declare the PTX ISA version of the code in the module, the target architecture for which the code was generated, and the size of addresses within the PTX module.
- `.version`
- `.target`
- `.address_size`
## 11.2 Specifying Kernel Entry Points and Functions
The following directives specify kernel entry points and functions.
- `.entry`
- `.func`
## 11.3 Control Flow Directives
PTX provides directives for specifying potential targets for `brx.idx` and `call` instructions. See the descriptions of `brx.idx` and `call` for more information.
- `.branchtargets`
- `.calltargets`
- `.callprototype`

## 11.4 Performance-Tuning Directives
To provide a mechanism for low-level performance tuning, PTX supports the following directives, which pass information to the backend optimizing compiler.
- `.maxnreg`
- `.maxntid`
- `.reqntid`
- `.minnctapersm`
- `.maxnctapersm` (deprecated)
- `.pragma`

The `.maxnreg` directive specifies the maximum number of registers to be allocated to a single thread; the `.maxntid` directive specifies the maximum number of threads in a thread block (CTA); the `.reqntid` directive specifies the required number of threads in a thread block (CTA); and the `.minnctapersm` directive specifies a minimum number of thread blocks to be scheduled on a single multiprocessor (SM). These can be used, for example, to throttle the resource requirements (e.g., registers) to increase total thread count and provide a greater opportunity to hide memory latency. The `.minnctapersm` directive can be used together with either the `.maxntid` or `.reqntid` directive to trade-off registers-per-thread against multiprocessor utilization without needed to directly specify a maximum number of registers. This may achieve better performance when compiling PTX for multiple devices having different numbers of registers per SM.

Currently, the `.maxnreg`, `.maxntid`, `.reqntid`, and `.minnctapersm` directives may be applied per-entry and must appear between an `.entry` directive and its body. The directives take precedence over any module-level constraints passed to the optimizing backend. A warning message is generated if the directives’ constraints are inconsistent or cannot be met for the specified target device.

A general `.pragma` directive is supported for passing information to the PTX backend. The directive passes a list of strings to the backend, and the strings have no semantics within the PTX virtual machine model. The interpretation of `.pragma` values is determined by the backend implementation and is beyond the scope of the PTX ISA. Note that `.pragma` directives may appear at module (file) scope, at entry-scope, or as statements within a kernel or device function body.
## 11.5 Debugging Directives
DWARF-format debug information is passed through PTX modules using the following directives:
- `@@DWARF`
- `.section`
- `.file`
- `.loc`

The `.section` directive was introduced in PTX ISA version 2.0 and replaces the `@@DWARF` syntax. The `@@DWARF` syntax was deprecated in PTX ISA version 2.0 but is supported for legacy PTX ISA version 1.x code.

Beginning with PTX ISA version 3.0, PTX files containing DWARF debug information should include the `.target debug` platform option. This forward declaration directs PTX compilation to retain mappings for source-level debugging.
## 11.6 Linking Directives
- `.extern`
- `.visible`
- `.weak`
