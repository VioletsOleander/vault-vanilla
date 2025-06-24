# Abstract
Deep neural network models are becoming increasingly popular and have been used in various tasks such as computer vision, speech recognition, and natural language processing. 

Machine learning models are commonly trained in a resource-rich environment and then deployed in a distinct environment such as high availability machines or edge devices. To assist the portability of models, the open-source community has proposed the Open Neural Network Exchange (ONNX) standard.
>  通常 ML 模型的训练和部署环境是不一定一致的
>  ONNX 标准存在的意义就是提高 ML 模型的可移植性

 In this paper, we present a high-level, preliminary report on our onnx-mlir compiler, which generates code for the inference of deep neural network models described in the ONNX format. Onnx-mlir is an open-source compiler implemented using the Multi-Level Intermediate Representation (MLIR) infrastructure recently integrated in the LLVM project. Onnx-mlir relies on the MLIR concept of dialects to implement its functionality. We propose here two new dialects: (1) an ONNX specific dialect that encodes the ONNX standard semantics, and (2) a loop-based dialect to provide for a common lowering point for all ONNX dialect operations. 
 >  onnx-mlir 编译器为 ONNX 格式的模型的推理进行代码生成
 >  onnx-mlir 依赖于 MLIR，提出了两个新的 dialect:
 >  - ONNX dialect，编码 ONNX 标准的语义
 >  - a loop-based dialect，为 ONNX dialect 中的 operation 提供共同的 lowering point (lowering point 就是指这个 dialect 本身，ONNX dialect 会被 lower 到这个 dialect)

> [! info] Lowering
> 在编译器中，降低 (lowering) 是指将高层抽象的表示 (如 ONNX operation) 逐步转换为低层抽象的表示 (如更接近硬件指令的循环、内存访问等) 的过程

Each intermediate representation facilitates its own characteristic set of graph-level and loop-based optimizations respectively. We illustrate our approach by following several models through the proposed representations and we include some early optimization work and performance results.

# 1. Introduction
Deep neural network models have been used widely for various tasks such as computer vision, speech recognition, and natural language processing. The success of such models was mainly originated from the development of accelerators, especially GPU accelerators, back in 2012 [3]. Since then, many deep learning frameworks, such as Torch, Caffe, Theano, and TensorFlow, have been developed to facilitate the training and inferencing of deep neural network models, which significantly speeds up the explosion of deep learning in many areas. However, training and inferencing are often done on different environments due to their different optimization characteristics. For example, a model is trained using a large-scale distributed system since it might need weeks or months to finish, and can then be used on lightweight devices such as Internet of Things or mobile phones for inferencing. Hence, it is desirable to dynamically rewrite a trained model so that it runs efficiently on a target environment.
>  因为训练环境和推理环境的解耦，故能够将训练好的模型重写为可以在目标推理环境高效运行的模型的技术是有必要存在的

Many deep learning frameworks utilize a highly-optimized library written for a target accelerator. Rewriting a model for inferencing consists of replacing the operations in the model with the function calls in the library. While such a library-call approach simplifies the rewritten procedure and would lead to improved performance, it exposes the following drawbacks. Firstly, the number of models that can be rewritten is limited by the provided functions in the library. Secondly, it is often the case that users need to install additional packages to make the library work well. Thirdly, it lacks the ability to tailor code specific to different problems since the same function may be used for them.
>  一种简单的重写方式是将模型中的函数调用替换为对针对目标加速设备的库函数的调用
>  其优势是简单，可以快速提高性能
>  其劣势是:
>  - 受到库提供的库函数数量的限制
>  - 用于需要下载额外的包/库才能跑起来模型
>  - 缺乏灵活性，不便于为不同问题调节代码
>  (如果库足够强大的话，感觉以上这些问题都不是大事)

We tackle these drawbacks by developing a compiler that rewrites a trained model to native code for a target hardware. It uses many mature optimization techniques developed during the long history of compiler, such as the ability to tailor code for a specific problem, memory optimizations, and parallelization. 
>  我们开发了将训练好的模型重写为针对目标硬件的 native code 的编译器
>  该编译器使用了许多成熟的编译优化技术，例如为特定问题定制代码的能力、内存优化、并行化

> [!info] Native code
>  原生代码就是硬件可以**直接**理解和执行的机器指令，它未经任何解释器或虚拟机的中间转换，直接与特定的处理器架构 (例如 x86, ARM, RISC-V) 和操作系统相关联，因此往往是高性能的

Our compiler is completely based on open-source software. In particular, we chose Open Neural Network Exchange (ONNX) [1] as a format to represent the input model of our compiler. ONNX is an open-source machine-independent format and widely used for exchanging neural network models. It has been actively maintained by and contributed from open source communities. Our compiler was written using Multi-level Intermediate Representation (MLIR) [5], a modern open source compiler infrastructure for multi-level intermediate representations and a subproject inside LLVM [4].

Our compiler is completely open-sourced and a subproject inside the ONNX project. Although it is still under development, it can already compile some popular models such MNIST and ResNet50 to native code on x86 machines, IBM Power Systems, and IBM System Z. In this paper, we will introduce our compiler by 
- presenting its overall design and architecture of the compiler,  
- introducing two new dialects: onnx dialect to encode the ONNX standard semantics, and krnl dialect to provide for a common lowering point for all ONNX dialect operations. 
- introducing optimization passes such as graph rewriting, constant propagation, and memory management, and 
- discussing some problems we encountered when emitting native code for different architectures.

The remainder of the paper is organized as follows. In Sec. 2, we briefly discuss ONNX and MLIR on which our compiler is based. In Sec. 3, we introduce our compiler, its design principle, and architecture. We also discuss in this section two new dialects, i.e., `onnx` and `krnl`, and some optimization passes. In Sec. 4, we present some preliminary experimental results for MNIST and ResNet50 models on IBM Power Systems. Finally, we conclude our paper and discuss future work in Sec. 5.

# 2. Background
## 2.1 ONNX
Open Neural Network Exchange (ONNX) [1] is an open source format for artificial intelligence models, including both deep learning and traditional machine learning. It defines an extensible computational graph model, operators, and standard data types, which provides a common IR for different frameworks. 
>  ONNX 标准定义了可拓展的计算图模型、算子、标准数据类型
>  ONNX 标准的目的是作为不同框架之间的中间表示

There are two ONNX variants: the neural-network-only ONNX variant recognizes only tensors as input and output types, while the classic machine learning ONNX-ML also recognizes sequences and maps. ONNX-ML extends the ONNX operator set with machine learning algorithms that are not based on neural networks. 
>  ONNX 的两个变体:
>  - neural-network-only ONNX: 输入和输出类型都仅识别 tensor
>  - classic machine learning ONNX-ML: 输入和输出类型识别 sequences, maps
>  ONNX-ML 拓展了 ONNX 算子集，添加了经典 ML 算法相关的，不基于神经网络的算子

In this paper, we focus on the neural-network-only ONNX variant and refer to it as just ONNX.
>  本文专注 neural-network-only ONNX

In ONNX, the top-level structure is a ‘Model’ to associate metadata with a graph. Operators in ONNX are divided into a set of primitive operators and functions, where a function is an operator whose calculation can be expressed via a subgraph of other operators. 
>  ONNX 的顶级结构是 Model, Model = graph + metadata
>  ONNX 中的 operator 分为: primitive operators, functions，其中 function 本质是一个 subgraph，但将其整体视作一个 operator

A graph is used to describe a function. There are lists of nodes, inputs, outputs, and initializers (constant values or default values for inputs) in a graph. An acyclic dataflow graph is constructed as a topological sort of the list of nodes in the graph. 
>  graph = a list of nodes, inputs, outputs, initializers
>  graph 中的 nodes 按照拓扑排序，构成一个无环的数据流图

Each node in a graph contains the name of the operator it invokes, inputs, outputs, and attributes associated with the operator. Inputs and outputs can be marked as variadic or optional. 
>  node = operator (with attributes), inputs, outputs, 
>  node 的 inputs 可以是 variadic 或 optional

There are three data types used to define inputs and outputs, i.e., ‘Tensor’, ‘Sequence’, and ‘Map’.
>  ONNX 为 inputs, outputs 限定了三种数据类型: Tensor, Sequence, Map

![[pics/Compiling Open Neural Network Models Using MLIR-Listing1.png]]

ONNX uses the Protocol Buffers definition language for its syntax. Listing 1 shows an example of an ONNX model for the LeakyRelu operator. There is one node in the graph (Lines 4–13), which is associated with LeakyRelu, and has one input, one output, and one attribute. The input and output tensors have the shape of $\langle 3 { \times } 4 { \times } 5 \rangle$ and element type of float32 (elem type: 1 at Lines 19 and 38).
>  ONNX 的语法采用 Protocol Buffers (还挺像 json 的)

## 2.2 MLIR
Multi-level Intermediate Representation (MLIR) [5] is a modern compiler infrastructure which is reusable and extensible. It reduces the cost of building domain-specific compilers by facilitating the design and implementation of code generators, translators, and optimizers at different abstraction levels. MLIR is a subproject of the LLVM project [6] and has many similarities to the LLVM compiler infrastructure [4]. In this section, we briefly review some of the features in MLIR that were used to build our compiler. For more information about MLIR, one can refer to a previous study [5]. Readers who are familiar with MLIR can skip this section.
>  MLIR 的设计目的是减少构建领域特定编译器的成本

Similar to LLVM, MLIR is a three-address static single assignment (SSA)-based IR, where values are defined before use and have a scope defined by their dominance relations. 
>  MLIR 是三地址 SSA IR
>  MLIR 中，值在使用前定义 (SSA 的自然结果，必须先给一个 SSA 值赋值，才能在后续操作中使用它)
>  MLIR 中，作用域由支配关系定义 (控制流图中，对于两个 operation，如果从入口点到第二个 operation 的所有路径都需要经过第一个 operation，就称第一个 operation 支配第二个 operation，这里的 SSA 值的任何使用点都必须被 SSA 值的定义点支配)
>  (这两个原则和近似，值在使用前定义或许更多强调的是程序顺序，作用域由支配关系定义或许更多强调在控制流图上的精确形式化)

Operations may produce zero or more results, and each operation is a distinct SSA value with its own type defined by the type system. The type system in MLIR is open, and one can define application-specific types. 
>  MLIR 中，每个 operation 本身就是一个独一无二的 SSA value，且具有类型系统定义的类型
>  MLIR 的类型系统提供了内建类型，也允许自定义类型

There are a number of primitive types, e.g., integers, as well as aggregate types for tensors and memory buffers, e.g., ‘Tensor’ and ‘MemRef’ types. 
>  MLIR 的内建类型包括基本类型，例如 integers，也包括聚合类型，例如 Tensor, MemRef

A Tensor type is abstracted and does not have a pointer to the data while a MemRef type is a lower representation, referring to a region of memory. In MLIR, Tensor and MemRef types are syntactically represented as tensor $\langle \mathsf { D } _ { 1 } \times \mathsf { D } _ { 2 } \times \ldots \times \mathsf { D } _ { \mathsf { N } } \times \mathsf { d } \mathsf { t y p e } \rangle$ and memref $\langle \mathsf { D } _ { 1 } \times \mathsf { D } _ { 2 } \times \ldots \times \mathsf { D } _ { \mathsf { N } } \times \mathsf { d } \mathrm { t y p e } \rangle$ , respectively, where $D _ { 1 } , D _ { 2 } , \ldots , D _ { N }$ are intergers representing the dimensions of a tensor or memref, and dtype is the type of the elements in a tensor or memref, e.g., $\mathsf { f } 3 2$ for float32. $\langle \mathsf { D } _ { 1 } \times \mathsf { D } _ { 2 } \times \ldots \times \mathsf { D } _ { \mathsf { N } } \rangle$ is called the shape of a tensor or memref. Tensor and MemRef types can be unranked when their shapes are unknown. In MLIR, unranked Tensor and MemRef types are syntactically represented as tensor $\langle { \ast } \times \mathsf { d t y p e } \rangle$ and memref $\langle * \times \mathsf { d t y p e } \rangle$ , respectively.
>  Tensor 类型是抽象类型，仅涉及数据的逻辑结构，不包含指向数据的指针 (不涉及数据的物理内存布局)，主要用于高层 IR
>  MemRef 类型则引用了内存区域，故描述了更多数据的存储信息，主要用于低层 IR
>  Tensor 和 MemRef 类型允许其形状未知，称为 unranked (例如在早期分析时，无法确定具体形状)

> [!info] SSA
> 静态单赋值是一种中间表示形式，它要求每个变量都只赋值一次
> 如果程序中，同一个变量被赋值多次，则在它的 IR 中，该变量会被分解为多个不同的、带版本号的变量 (例如 `x -> x1, x2`)
>  
> SSA 引入了一个特殊的函数，称为 $\Phi$ 函数，用于处理控制流汇聚点 (`if-else` 语句的末尾或循环语句的末尾)，$\Phi$ 函数用于在遇到来自不同路径的变量版本需要合并时，选择正确的版本
> 
> SSA 的优势是数据流简单，依赖关系清晰 (例如 `y` 依赖于 `x` 的特定版本)，便于编译器分析和优化、转换
>  
> 一个 $\Phi$ 函数的例子
> ```
> if (condition) {
>     x = 1
> } else {
>     x=2
> }
> y = x+1
> ```
>  其 SSA 形式为
> ```
> if (condition) {
>     x1 = 1
> } else {
>     x2=2
> }
> x3 = Phi(x1, x2)
> y = x3 + 1
> ```
>  即 $\Phi$ 函数负责了根据实际的控制流选择变量版本

![[pics/Compiling Open Neural Network Models Using MLIR-Fig1.png]]

An operation is the unit of code in MLIR. To define an operation, a TableGen-based [7] specification for an operation descriptor is used. Figure 1 shows the structure of an operation. An operation has a list of SSA operands and may have attributes that store static information. An operation can hold a region which is a list of blocks. A block contains a list of operations and ends with a terminator operation that may have successor blocks to which the control flow may be transferred. 
>  MLIR 的代码单元就是 operation (类似汇编语言、机器语言中的指令，它是最小的、可独立执行的单元，所有的控制流和计算都基于这个单元来表示)
>  MLIR 采用 TableGen 定义 operation
>  见 Fig1，operation = Attributes (静态信息) + SSA operands + Regions + Successors
>  其中 Region = a list of blocks, Region 本身就是一个独立的控制流图
>  Block = a list of operations (在这里循环定义了)，Block 的最后一个 operator 是一个特殊的 terminator operator，该 operator 决定控制流应该被转移到哪里

That being said, nested regions becomes a firstclass concept in MLIR, which is efficient to represent control flow graphs. A function is an operation with a single region and attributes. A module is an operation with a single region containing a single block and terminated by a dummy operation.
>  因此，控制流图在 MLIR 中，都以嵌套的 regions 的形式表示 (例如 if-else 嵌套了两个区域，一个用于 if, 一个用于 else)
>  Function 是仅含一个 Region 的 operation，该 Region 就是函数体的完整控制流图
>  Module 是仅含一个 Region 的 operation，且该 Region 仅含一个 Block

>  可以看到，MLIR 的所有一切概念都基于 operation 定义，无论是 Function，Module，本质都视作一个特殊的 operation

To develop a compiler using MLIR, users often need to define dialects and optimization passes. A dialect serves as an abstraction level or intermediate representation, and an optimization pass is to enable optimization at an abstraction level or transformation among abstraction levels.
>  MLIR 的用户通常需要定义: dialects, optimization passes
>  dialect 表示了一个抽象的层次，例如 `tf` 抽象了 tensorflow 的计算操作，`linalg` 抽象了线性代数的计算操作，`arith` 抽象了基本的整数和浮点运算操作，`llvm` 抽象了 LLVM IR
>  optimization pass 用于在抽象层次上的优化和抽象层次上的转换 (即 dialect 上的优化和转换)

There are dialects in MLIR that are ready to use, e.g., llvm, std, scf, and affine. The llvm dialect is a low-level dialect. It wraps the LLVM IR types and instructions into MLIR types and operations. The std dialect includes standard operations such as load, store, addi, addf, absf, and call. The scf dialect defines control flow operations such as for and if. The affine dialect provides an abstraction for affine operations and analyses.

Optimization passes can be roughly classified into three categories: general transformation, conversion, and dialect-specific. General transformation passes includes common passes such as ‘canonicalize’ pass for operation canonicalization, ‘CSE’ pass to eliminate common sub-expressions, and passes to print IR information such as ‘print-op-graph’, ‘print-op-stats’, and ‘print-cfg-graph’. Conversion passes are to convert operations in one dialect to operations in another dialect, e.g., ‘convert-std-to-llvm’ pass to convert standard operations into LLVM instructions. Finally, dialect-specific passes are for transformation in a dialect, e.g., ‘affine-loop-unroll-jam’ pass to unroll and jam affine loops in the affine dialect. MLIR passes can be expressed via Declarative Rewriting Rules (DRRs) using tablegen records or via writing code in $C + +$ .
>  Optimization passes 大致分为三类:
>  - general transformation
>  - conversion
>  - dialect-specific
>  General transformation 即泛用的 optimization pass，包括 canonicalization, common sub-expression elimination，以及用于 print IR 信息的 passes 等
>  Conversion 即用于 dialect 之间转换的 passes
>  Dialect-specific 即用于 dialect 之内转换的 passes
>  MLIR passes 同样由 TableGen 定义

To denote an operation in a dialect, we explicitly use a form of dialect name.operation name. For example, std.load means the operation load of dialect std. Optimization passes are named with prefix ‘--’, for example, --canonicalize is the canonlicalization pass.
>  Dialect 内的 operation 表示格式为 `dialect-name.operation-name`
>  Optimization pass 的表示格式为 `--<pass-name>`

![[pics/Compiling Open Neural Network Models Using MLIR-Listing2.png]]

Listing 2 shows an example for calculating the exponential of a given input tensor, element-wise, using std and affine dialects. The top level is a module containing a function ‘exp’. The function ‘exp’ accepts one input that is of memref type, and produces an output of the same type. The memory for the output is allocated via std.alloc (Line 3). There is a nested loop (Lines 4–10), iterating over dimensions of the inputs using affine.for, loading each element from the input using affine.load (Line 6), computing the exponential using std.exp (Line 7), and storing the result in the output using affine.store (Line 8). The output of the function is finally returned using std.return.

# 3. Compiling ONNX Models
This section introduces our compiler, onnx-mlir. We first discuss its overall architecture. We then introduce two new dialects, `onnx` and `krnl` dialects. Finally, we present MLIR passes for carrying out optimization.

## 3.1 Overview

![[pics/Compiling Open Neural Network Models Using MLIR-Fig2.png]]

Figure 2 shows the overall architecture of onnx-mlir. The input is an ONNX model, and the output is a library containing the compiled code. The output library contains an entry function called ‘ `_dyn_entry_point_main_graph` ’ whose inputs and outputs are similar to the ONNX model’s inputs and outputs, respectively. 
>  onnx-mlir 的架构见 Fig2
>  它接收 ONNX 模型，输出二进制库或可执行文件
>  它输出的库的入口函数称为 `_dyn_entry_point_main_graph`，该入口函数接收和 ONNX 模型的输入类似的输入，输出和 ONNX 模型的输出类似的输出

To carry out inference with the output library, users write their program to call the entry function by passing inputs to the function and obtain results.
>  编译之后，用户调用入口函数，传入输入即可执行推理

There are five main dialects in onnx-mlir, i.e., onnx, krnl, affine, std and llvm, organized into four abstraction levels. Two new dialects, onnx and krnl, are discussed in Sections 3.2 and 3.3, respectively. The first abstraction level is a high-level representation of ONNX operations. It consists of operations in onnx and std dialects, where the onnx dialect is automatically generated via an importer that is a python script. The second abstraction level includes krnl, affine and std dialects. krnl dialect provides a representation that is suitable for loop optimizations, which is able to carry out affine transformations such as tile, skew, and permutation easily. It plays as an intermediate dialect for efficiently lowering the onnx dialect into low-level dialects (e.g., affine, std and llvm). The third abstraction level includes affine and std dialects where existing optimization passes in MLIR can be freely applied. The forth abstraction level includes only llvm dialect that is ready to generate bitcode.
>  onnx-mlir 涉及五个 dialect: `onnx, krnl, affine, std, llvm`，其中 `onnx, krnl` 为新加入的 dialect
>  这五个 dialects 组织为四个抽象层次:
>  - ONNX 层: 包含 `onnx, std` dialect 中的 operation
>  - Krnl 层: 包含 `krnl, affine, std` dialect 中的 operation，这一层开始将高层的 ONNX operation 分解为更细粒度的、基于循环的计算，其中 `krnl` dialect 的设计目的是为循环优化提供合适的表示，它引入了显式的循环结构、内存访问模式，便于编译器执行仿射变换，包括 tile (将大循环分解为更小的嵌套循环，改善 cache locality), skew (改变循环迭代的顺序，以避免数据依赖，暴露并行性), permutation (交换嵌套循环的顺序), `krnl` dialect 的作用是作为 `onnx` dialect lower 到低层 dialect 如 `affine, std, llvm` 的中间层
>   - Affine & Std 层: 包含 `affine, std` dialect
>   - LLVM 层: 包含 `llvm` dialect，用于 bit code generation

There are MLIR passes for converting one dialect to another, and for doing optimizations at a specific dialect. onnx dialect is converted to krnl dialect via pass --convert-onnx-to-krnl. Then krnl dialect (except some of its operations) is converted into affine and std dialects via pass --convert-krnl-to-affine. The remaining operations in krnl dialect and operations in affine and std dialects are directly converted into instructions in llvm via pass --convert-krnl-to-llvm. The right side of Fig. 2 shows optimization passes that can be carried out at each abstraction level.
>  `onnx` dialect 通过 `--convert-onnx-to-krnl` pass 转化为 `krnl` dialect
>  `krnl` dialect 的大部分 operations 通过 `--convert-kerl-to-affine` pass 转化为 `affine` 和 `std` dialect
>  `krnl` dialect 的少部分 operations 和 `affine, std` 中的 operations 通过 `--convert-krnl-to-llvm` 转换为 `llvm` dialect

We only enumerate the important optimizations here, and the list of optimization passes is not exhaustive.

![[pics/Compiling Open Neural Network Models Using MLIR-Fig3.png]]

Before discussing dialects and optimization passes in detail, we give a brief running example and go through dialects in onnx-mlir. This example is a testcase model in ONNX that performs element-wise binary addition. Figure 3 shows this ONNX model of the testcase. Operation add accepts two tensors of type $\langle 3 { \times } 4 { \times } 5 { \times } \mathsf { f } 3 2 \rangle$ (element type is float 32) and returns a result tensor, i.e., sum, of the same type. Listings 3, 4, and 5 show emitted programs in different dialects onnx, krnl, affine, respectively. We omit the program in llvm due to space limitations.

![[pics/Compiling Open Neural Network Models Using MLIR-Listing3.png]]

In onnx dialect, operations are represented similarly to their descriptions in ONNX. The ONNX model is converted into the function main graph. To generate an entry point function into which users feed their inputs, we create a helper operation in the onnx dialect, i.e., onnx.EntryPoint, which keeps meta-data in the operation’s attributes such as function name to call and the number of inputs and outputs.
>  整个 ONNX model 会被转换为 `onnx` dialect 中名为 `main_graph` 的 function
>  一个名为 `onnx.EntryPoint` 的 operation 会被生成，作为 `main_graph` 的入口点，该 operation 会记录输入输出数量等元信息

In krnl dialect, operation onnx.Add is translated into a loop-based computation represented by operations in the krnl dialect, where scalar computation is represented by primitive operations in the affine and std dialects. We can apply loop optimizations, such as tile, skew, or transpose, to loop-based computation. At this level, we allocate memory for output tensors, and memory management can be performed.
>  在 `krnl` dialect 的 level，会为输出 tensor 分配内存，以及执行内存管理

In affine dialect, optimized loop-based computation in krnl dialect is translated into affine.for loops. At this level, we still have an operation in krnl, i.e., krnl.entry point. Such an operation is not related to the main computation and will be directly converted to llvm. Operations in the affine dialect will be converted to operations in the std and scf dialects before being lowered to instructions in the llvm dialect.
>  在 `affine` dialect 的 level，`krnl` 表示的计算会被转换为 `affine.for` loops
>  `krnl` 中的 `krnl.entry_point` 这个 operation 会被保留
>  `affine` dialect 中的 operations 后续都会被转换为 `std, scf` dialect

## 3.2 `onnx` dialect
onnx dialect is the first abstraction level in onnx-mlir and represents an ONNX model in MLIR language. 
>  onnx-mlir 的第一层抽象层即 `onnx` dialect，该 dialect 将 ONNX 模型转化到 MLIR 框架中

We wrote a python script to automatically import ONNX operations into the tablegen-based operation definitions in MLIR. These imported operations are organized into the onnx dialect. Thanks to tablegen, the operation definition in the onnx dialect is quite similar to the operation description in ONNX, where we are able to represent all necessary information, such as inputs, outputs, attributes, and description, into a single tablegen-based definition in human-readable textual form.
>  ONNX operations 到 MLIR 中 tablegen-based operation definition 由一个 python 脚本完成
>  `onnx` dialect 中对各个 operation 的定义和 ONNX 对各个 operation 的描述非常相似，故所有的必要信息，例如输入、输出、属性、描述都可以写在单个 tablegen 文件中

We also created a new operation in the onnx dialect, i.e., onnx.EntryPoint to keep information related to the dynamic list of inputs in an ONNX model. This operation will be lowered to generate the entry function ‘ dyn entry point main graph’ of the generated library.
>  除了转译 ONNX 中的 operation 定义以外，`onnx` dialect 中还引入了一个新的 operation: `onnx.EntryPoint` 
>  该 operation 用于保存对相对于 ONNX 模型的动态输入列表信息，且会被 lower 以生成最终的入口函数 `_dyn_entry_point_main_graph`

![[pics/Compiling Open Neural Network Models Using MLIR-Listing6.png]]

Listing 6 shows a tablegen-based definition for the relu operation imported via the importer in onnx-mlir. The operation description is represented in the ‘description’ field (Line 4). Inputs and attributes are represented in the ‘arguments’ field, while outputs were represented in the ‘results’ field (Lines 5–6). All inputs and outputs will be imported as a tensor in MLIR. The importer automatically infers element types for inputs, attributes, and outputs. However, the shape of a tensor will be inferred via the --shape-inference pass, which is a trait in the LeakyRelu operation (Line 2). MLIR generates a $\mathrm { C } { + } { + }$ class definition for an operation from its tablegen-based definition. If users want to define custom declaration in the class, it can be done via the ‘extraClassDeclaration’ field (Line 7).
>  Listing 6 展示了 ONNX operation `relu` 在 `onnx` dialect 中对应的 tablegen 定义
>  operation 描述在 `description` 中
>  operation 输入和属性在 `arguments` 中
>  operation 输出在 `results` 中
>  所有的输入和输出都会被表示为 MLIR 中的 Tensor
>  导入脚本会在导入时自动推导输入、属性和输出的元素类型
>  但 tensor 的形状需要通过 `--shape-inference` pass 执行，该 pass 定义为该 operation 的 trait
>  该 tablegen 定义会自动生成一个 C++ 类定义
>  `extraClassDeclaration` 留给用户的自定义声明

## 3.3 `krnl` dialect
A computation kernel in a neural network workload has local structural simplicity in which loop nests are often simple, e.g., hyper-rectangle and statements carry quite straightforward arithmetic semantics. 
>  神经网络中的计算核的循环结构通常是较简单的，例如循环通常是超矩形，且循环内语句具有相当直接的计算语义 (超矩形意味着循环边界是常量或简单的线性表达式，且每个维度的迭代范围是独立的；直接的计算语义意味着循环内执行的通常是直接的计算操作，而不是更复杂的控制流，例如嵌套的 `if-else`)

Such a characteristic is quite suitable to be represented in a polyhedral model for optimization [8]. krnl dialect aims to host both loop optimization and scalar semantic optimization in a single representation. 
>  因此，这样的计算核非常适合用多面体模型进行优化
>  `krnl` dialect 的设计目标是同时适用于循环优化 (主要目标) 和标量语义优化 (对循环体内的指令进行优化，例如加法、乘法、变量赋值，常见的优化由公共子表达式消除、常量传播、死代码消除)

It is expected to provide interpretability where not only is polyhedral representation readable but it also makes program semantics (or what to execute) and program schedules (how and when to execute) independent. 
>  `krnl` dialect 的设计目标还包括了可解释性:
>  - 可读的多面体表示，即 `krnl` 的 IR 设计旨在清晰反映多面体模型中的循环和数据访问概念，同时保持可读性
>  - 程序语义和程度调度独立: 程序语义是要执行什么，即算法的逻辑，程序调度是如何和什么时候执行，即循环顺序，并行化策略，内存访问模式等，也就是说 `krnl` dialect 在 IR 中分离了这两部分信息，故我们可以独立地定义程序的计算逻辑和执行调度

In other words, our goal is to optimize not only programs but also the composition of individual schedules, which is a feature that is often lacking in other existing systems.
>  因为语义和调度的分离，故 `krnl` 不仅可以优化单独的调度，还可以对语义和调度组合

> [!info] Polyhedral model
> 多面体模型是一个专门针对优化具有复杂循环嵌套程序的编译优化框架
> 多面体模型将程序的循环迭代空间和数据访问模式表示为数学上的多面体，从而可以应用线性代数和整数规划等数学工具来找到最优的循环变换
> 
> 简而言之，多面体模型将代码的执行视为一个几何问题，然后用数学方法求解它，找到最优的执行模式
> 
> 多面体模型的工作步骤:
> (1) 程序转换为 AST: 这也是传统编译器的第一步
> (2) 提取仿射循环: 仿射循环是该模型的优化对象，如果一个循环的循环边界、步长、循环内部的数组访问索引表达式都是循环变量或外部参数的线性函数，或常数，该循环就是仿射循环
> (3) 构建迭代空间: 每个循环迭代都看作一个空间中的一个点，一个 $N$ 重嵌套的循环可以被表示为一个 $N$ 维的整数点几何
> 例如，二重循环 `for (i = 0; i < N; i ++) {for (j = 0; j < M; j++) {}}` 的迭代空间就是坐标点 `(i, j)` 的集合，其中 `0 <= i < N, 0 <= j < M`，故循环的迭代空间是一个二维空间的矩形
> (4) 构建访问关系: 为循环体内的每次内存访问构建访问关系，访问关系是从迭代空间到地址空间的映射
> 例如 `A[i][j]` 的访问关系就是 `(i, j) -> (i, j)`，`B[i+1][j-1]` 的访问关系就是 `(i, j) -> (i+1, j-1)`
> (5) 分析数据依赖: 通过迭代空间和访问关系分析程序中的数据依赖 (例如读后写、写后读、写后写)，这些依赖被表示为迭代空间中的依赖向量
>  数据依赖是执行循环转换的约束条件，任何转换都不能破坏数据依赖
> (6) 应用循环变换: 将各种循环优化 (tile, skew, permutation 等) 表示为对迭代空间进行线性变换
> 这些变换被表示为一个整数矩阵，迭代空间点乘该矩阵，得到新的迭代空间，目标就是找到最优的变换矩阵，同时不违反依赖
> (7) 生成优化后的代码: 基于新的迭代空间进行

Below is an example that defines a nested loop in krnl: 

```
1 %ii, %jj = krnl.define_loops 2
2 krnl.iterate(%ii , %jj ) with (%ii -> %i = 0 
    to 10 , %jj -> %j = 0 to 10) {
3  %foo = std.addi %i , %j : index
4 }
```

where `krnl.define_loops` defines two loops, called ii and jj. These loop variables will be used to express both program semantics and schedules. Operation krnl.iterate semantically accepts two types of loop variables: variables for original loops and variables for scheduled loops. In syntactic sugar form, we separate the two types of loops by the keyword with, i.e. (scheduled loops) with (original loops). Induction variables, e.g., i and j in the above example, will be defined by using original loops. If there is no schedule (e.g. block, skew, etc.), the scheduled loops are similar to the original loops.

>  示例代码中，`krnl.define_loops` 定义了两个循环 `ii, jj` ，循环变量 (循环指示符) `ii, jj` 即会用于表示程序语义，也会用于表示程序调度
>  `krnl.iterate` 表示一个循环执行体，它接收两种类型的循环变量: 表示原循环的循环变量和表示调度后循环的循环变量，调度后循环即应用了各种优化调度后得到的循环，也就是优化后的循环
>  示例中用关键词 `with` 分离了这两种类型的循环变量 (分离了 scheduled loops 和 original loops，注意 scheduled loops 在 `with` 之前)
>  `i,j` 为归纳变量，即传统 `for` 循环中用于迭代的变量，归纳变量是基于原始循环定义的，也就是即便循环已经被调度/优化，其实际执行顺序发生了变化，但循环体内的语义表达仍然基于原始循环来表示

Now, we insert a schedule for blocking or tiling. Without loss of generality, we define just one loop instead of two.
>  接下来我们利用 `krnl` 对循环进行优化，我们插入一个 blocking/tiling 的调度

> [!info] Loop blocking/tiling
> 循环分块/平铺是用于改善数据局部性的循环优化技术
> 它将大的循环分解为两个或者多个嵌套的循环，外层循环迭代 “块” 的索引，内层循环迭代块内部元素的索引
> 这样可以在处理小数据块时，将数据全部加载到缓存中，减少对主存的访问

```
1 %ii = krnl.define_loops 1
2 %ib , %il = krnl.block %ii 2 :
    (!krnl.loop) ->(!krnl.loop, !krnl.loop)
3 krnl.iterate (%ib , %il) with (%ii -> %i = 0 to 10) {
4     %foo = std.addi %i , %i : index
5 }
```

Operation krnl.block (Line 2) takes a loop and integer as inputs, where the integer is the tile size with which we want to carry out blocking. Results are two loop variables: one for the outer loop and the other for the inner loop. The two loops will be used as the result of scheduling and be passed to krnl.iterate (Line 3). It is worth noting that the original loops and computation in krnl.iterate remained unchanged while inserting a schedule, which is exactly what we want for separating program semantics and schedules in our krnl dialect.

>  示例代码中
>  `$ii = krnl.defin_loops 1` 定义了一个循环变量，循环变量是循环的抽象表示符
>  `krnl.block` 接收一个循环和一个整数，整数表示 blocking 采用的 tile size，返回两个循环变量: 一个是外层循环，一个是内层循环
>  这两个循环变量就是调度的结果，可以被传递给 `krnl.iterate`，注意到 `krnl.iterate` 中对计算的表示依然按照原来的循环表示，这就是 `krnl` 对程序语义和调度的分离

The --convert-krnl-to-affine pass automatically generates optimized affine.for based loops as follows.

```
1 #map0 = affine_map <(d0) -> (d0)>
2 #map1 = affine_map <(d0) -> (d0 + 2) >
3 affine.for %arg0 = 0 to 10 step 2 {
4     affine.for %arg1 = # map0 ( %arg0 ) to 
        #map1 ( %arg0 ) {
5     %0 = addi %arg1 , %arg1 : index
6     }
7 }
```

The outer affine.for iterates with step 2 i.e., the tile size, and the inner affine.for iterates over the elements in a tile.

>  `--convert-krnl-to-affine` pass 将 `krnl` 优化的循环转换为基于 `affine.for` 的循环，如上所示
>  其中外部的 `affine.for` 以 tile size 作为步长迭代，内部的 `affine.for` 迭代 tile 内的元素

Other schedules, such as skew and permutation are used in a similar manner. All schedules are composable and can be nested.
> 其他的循环调度，例如 skew, permutation 的使用也是类似的，所有的调度都可以组合和嵌套

## 3.4 Optimization Passes
In this section, we discuss some of the optimization passes in onnx-mlir. Thanks to the expressive power of MLIR, many optimizations can be expressed easily via Declarative Rewriting Rules (DRRs) using tablegen records or writing code in C++.

### 3.4.1 Operation Decomposition
In ONNX, many operations can be expressed using other basic operations. For example, ReduceL1 over a vector $x$ is mathematically calculated by summing up the absolute values of the elements in $x$ . In other words, we have

$$
{ \mathsf { R e d u c e L 1 } } = { \mathsf { R e d u c e S u m } } \ ( { \mathsf { A b s \ x } } )
$$

>  ONNX 中的许多 operation 都可以用其他的 basic operation 表示
>  例如 ReduceL1 就可以表示为先对目标向量取绝对值，然后执行 ReduceSum

We only need to lower a subset of operations in the onnx dialect to krnl dialect, while the remaining operations in the onnx dialect will be decomposed into operations in the subset.
>  我们只定义了 onnx dialect 中的一部分 operations 到 krnl 的下降规则
>  onnx dialect 中的各种 operation 会被分解为由这部分 operation 组成的形式，然后再被下降到 krnl

Using the DRRs in MLIR, operation decomposition is concisely written as the following pattern:

```
def ReduceL1Pattern: Pat< 
2  (ReduceL1Op $x , $axes , $keepdims ), 
3  (ReduceSumOp (AbsOp $x), $axes , $keepdims ) 
4 >;
```

where ReduceL1Op, ReduceSumOp, and AbsOp are programmable forms of operations onnx.ReduceL1, onnx.ReduceSum, and onnx.Abs respectively. Variables x, axes, and keepdims are for keeping input values of operation ReduceL1Op. The pattern ‘ReduceL1Pattern’ contains a source pattern to match a graph of one operation ReduceL1Op (Line 2) and a destination pattern to generate a graph of two operations ReduceSumOp and AbsOp (Line 3). Whenever an operation ReduceL1Op appears in an ONNX model, it will be replaced with a combination of ReduceSumOp and AbsOp.

>  operation 分解基于 MLIR 的 DRR 写成，一个示例如上
>  上述代码定义了一个名为 `ReduceL1Pattern` 的 DRR，`Pat<...>` 是 DRR 的语法骨架，`Pat` 表示这是一个模式 (Pattern)
>  源模式: `ReduceL1Op $x, $axes, $keepdims` 表示在遇到 `ReduceL1Op` 时，捕获它的 `x, axes, keepdims`
>  目标模式: `ReduceSumOp (AbsOp $x), $axes, $keepdims` 表示匹配后要生成的新模式，它基于之前捕获的变量

### 3.4.2 Shape Inference
The --shape-inference pass attempts to infer shapes for all tensors in a program at onnx. The pass traverses all operations in a program, infers the shapes of tensors with unrank shapes (i.e. tensor $\left. * \times \mathsf { f } 3 2 \right.$ ), propagates the ranked shapes to consuming operations, and terminates once all tensors have ranked shapes. 
>  `--shape-inference` pass 的目标是为所有的 tensor 推理出形状
>  该 pass 会遍历程序中的所有 operations，然后为具有 unrank shape (形状信息不完整) 的 tensor 推理形状信息
>  tensor 的形状信息 (ranked shape) 被推理出来后，就会被传递给接收该 tensor 的后续 operation，帮助确定该 operation 的输出 tensor 的形状
>  这样的过程持续执行直到所有的 tensor 的形状都确定

For one operation, if its inputs have static shapes, it is likely that the --shape-inference pass will be able to infer static shapes for its outputs. If the inputs have dynamic shapes (e.g. tensor $\langle ? { \times } ? { \times } ? { \times } \mathsf { f } 3 2 \rangle$ ), the outputs will also have dynamic shapes also, except for some operations whose output tensors’ shapes are specified in the operation attributes.
>  一般来说，如果一个 operation 的输入是静态形状，则它的输出一般也是静态形状，故 `--shape-inference` 一般可以推导出其输出的静态形状
>  而如果一个 operation 的输入是动态形状，而它的输出则一般也是动态形状，则形状只有在运行时才能确定，编译时的 `--shape-inference` 一般推理不出它的形状 (除了一些特殊的 operation，其输出形状完全不依赖于其输入形状，而是由其属性指定)

### 3.4.3 Graph Rewriting
Graph rewriting is a powerful optimization tool. It is intensively applied to neural networks since calculation in a neural network is expressed via a dataflow graph. In MLIR, graph rewriting rules are conveniently represented using DRRs.
>  图重写 (匹配计算图中的特定模式，将其替换为等效或更优的模式，例如融合、去除冗余、内存布局优化等) 是广泛用于神经网络优化的优化方法，因为神经网络的计算本身可以描述为一个数据流图
>  MLIR 用 DRR 表示图重写规则

For example, the following rule is to fuse onnx.Add and onnx.MatMul into a single operation onnx.Gemm under the condition that the result of MatMulOp is only consumed by $\mathsf { A d d O p }$ :

```
1 def MulAddToGemmPattern : Pat < 
2   (AddOp (MatMulOp: $res $m1, $m2), $m3), 
3   (GemmOp $m1 , $m2 , $m3), 
4   [( HasOneUse $res )] 
5 >;
```
 
 >  例如上述示例将一个矩阵乘 + 加法算子的模式替换为通用的矩阵乘加算子，替换的条件是 MatMulOp 的结果仅由 AddOp 使用 (否则如果 `$res` 被多个算子使用，融合后再计算的 `$res` 就会破坏其他算子的输入正确性)

Another example is to remove an IdentityOp operation by passing its input directly to its consuming operations.

```
1 def IdentityEliminationPattern : Pat <
2   (ONNXIdentityOp $arg ),
3   (replaceWithValue $arg )
4 >;
```

>  上述示例将 IdentityOp 消除，方法是将该 Op 的结果直接替换为该 Op 的输入

Users can write as many rewriting rules as possible in the same manner.

### 3.4.4 Constant propagation
Constant propagation is a well-known optimization in compilers. 
>  常量传播是常用的编译优化技术，其原理是在编译时识别出各个表达式中可以确定为常数的值，然后用这些常数直接替换掉表达式中的变量或子表达式，以减少运行时计算

In onnx-mlir, we created a pass to do this during compilation. There are two key ideas in constant propagation: (1) if all the inputs of an operation are constant, compute its outputs during compilation and remove the operation, (2) if there is a mix of constant and non-constant inputs, normalize the operation. Normalization is to increase the possibility of constant propagation and strongly depends on the mathematical properties of an operation. 
>  我们创建了一个 pass 来执行常量传播
>  常量传播中有两个关键思想
>  1. 如果 operation 的所有输入都是常数，就在编译时计算它的输出，然后直接移除该 operation
>  2. 如果 operation 的输入有常数也有非常数，就规范化该 operation
>  规范化的目的是重写 operation 以增加未来常量传播的机会，规范化强烈依赖于 operation 的数学性质

Below are some normalization rules in onnx-mlir for the onnx.Add operation whose properties are associative and communicative.

$$
\begin{array} { l } { c + x \Rightarrow x + c } \\ { ( x + c _ { 1 } ) + c _ { 2 } \Rightarrow x + ( c _ { 1 } + c _ { 2 } ) } \\ { ( x + c ) + y \Rightarrow ( x + y ) + c } \\ { x + ( y + c ) \Rightarrow ( x + y ) + c } \\ { ( x + c _ { 1 } ) + ( y + c _ { 2 } ) \Rightarrow ( x + y ) + ( c _ { 1 } + c _ { 2 } ) } \end{array}
$$

where $x$ and $y$ are non-constant values, and $c$ , $c _ { 1 }$ , and $c _ { 2 }$ are constant values. Normalization rules are expressed by using the DRRs in MLIR.

>  onnx-mlir 为 `onnx.Add` 定义的规范化规则如上，它利用了该 operation 的结合率和交换律
>  规范化规则同样用 DRR 定义

# 4. Preliminary Experiments
## 4.1 ONNX operation support and testcases
ONNX provides a set of test cases for each operation. When we support any operation in onnx-mlir, we enable its ONNX test cases to check whether the operation behaves correctly and produces correct result. 
>  ONNX 本身为其 operation 提供了一系列 test cases，我们利用这些 test case 检验重写后 operation 的正确性

At the time of writing this paper, onnx-mlir supports 51 operations out of 139 operations in ONNX, including important operations such as convolution, pooling, Gemm, and LSTM. These are enough to compile and execute major networks such as MNIST and ResNet50. 

On the GitHub repository of onnx-mlir, we enable continuous integration on different environments, i.e., Windows, Linux, and Docker environments, and different systems, i.e., x86 machines, IBM Power Systems, and System Z. All supported operations have passed tests on the above environments.
>  我们为不同环境、不同系统下都设定了 CI

## 4.2 MNIST and ResNet50
In this section, we present some of our preliminary results for two neural network models in the ONNX Model Zoo: MNIST and ResNet50 [2]. The MNIST $^5$ and ResNet50 $^6$ models have already been trained in the CNTK and Caffe2 frameworks, respectively. 

We ran inferences on the given test data set in each model. The experiments were conducted on a machine with 2.3-GHz POWER9 processors. For onnx-mlir, graph rewriting and canonicalization passes were enabled. In this paper, we only provide a reference implementation that is not optimized, thus performance measurements are not applicable.

Table 1: Run inferencing with MNIST and ResNet50 on a POWER9 machine. Time in seconds. 

<html><body><center><table><tr><td>Model</td><td>Compilation time</td><td>Inference time</td></tr><tr><td>MNIST</td><td>0.237</td><td>0.001</td></tr><tr><td>ResNet50</td><td>7.661</td><td>7.540</td></tr></table></center></body></html>

Table 1 shows the running times for the MNIST and ResNet50 models when doing inferencing. 

For each model, we measured the compilation time for compiling the model to native code and inference time for running the native code with real inputs. MNIST is a small model with two convolutional operations, one max pooling operation and a matrix multiplication followed by an element-wise addition. Compiling the MNIST model and carrying out inferencing was rather fast, i.e., finished in less than one second. In the MNIST model, the graph rewriting rule MulAddToGemmPattern mentioned in Sec. 3.4.3 was applied to fuse matrix multiplication and element-wise addition into a Gemm operation. 

ResNet50 is a complex deep model consisting of 50 layers of operations such as convolutions and poolings. The model is about 100 megabytes including learned weights. For ResNet50, the current version of onnx-mlir does not have any optimization applied to the model during compilation. However, we believe that the compilation time looks reasonable and the inference time is not so slow. We hope that once we integrate important optimizations, such as polyhedral optimizations, SIMD optimization, and loop fusion in near future, the inference time will be significantly reduced.

## 4.3 Supported Systems
Although onnx-mlir is completely built upon widely-used open source software such as ONNX and MLIR, we found a problem related to supporting different systems. In particular, we could not run ONNX models on Linux on IBM System Z (s390-linux) because the big-endian format was not well-supported in ONNX and MLIR. 

There are two reasons for such a problem. First, a large amount of public input data and models in ONNX are stored in little-endian format. Hence, they must be converted to big-endian format before they are used in a big-endian system. Second, we found that constant values in ONNX models are not correctly loaded in MLIR. LLVM was well-supported in big-endian, but MLIR was not. 

We created two patches to solve this problem: one in ONNX $^7$ and one in MLIR $^8$ , and they are now available at the master branches of ONNX and MLIR. As a result, onnx-mlir now supports Linux on x86 (x86-Linux), Linux on Power Systems (ppc64le-Linux), Linux on IBM Z (s390-Linux), and Windows.

# 5. Conclusion
We are developing an open source compiler called onnxmlir for compiling ONNX models into native code. 

MLIR was used as an infrastructure to build the compiler, and two novel dialects were introduced, i.e., onnx and krnl. We also discussed some optimizations such as graph rewriting and constant propagation. It is worth noting that new optimizations can be easily integrated into onnx-mlir thanks to the MLIR infrastructure. 

In the future, we will add more optimizations, e.g., polyhedral optimization, loop fusion, SIMD optimization, and enable code generation for accelerators.
