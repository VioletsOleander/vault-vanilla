The difficulty of deploying various deep learning (DL) models on diverse DL hardware has boosted the research and development of DL compilers in the community. Several DL compilers have been proposed from both industry and academia such as Tensorflow XLA and TVM. Similarly, the DL compilers take the DL models described in different DL frameworks as input, and then generate optimized codes for diverse DL hardware as output. However, none of the existing survey has analyzed the unique design architecture of the DL compilers comprehensively. In this paper, we perform a comprehensive survey of existing DL compilers by dissecting the commonly adopted design in details, with emphasis on the DL oriented multi-level IRs, and frontend/backend optimizations. We present detailed analysis on the design of multi-level IRs and illustrate the commonly adopted optimization techniques. Finally, several insights are highlighted as the potential research directions of DL compiler. This is the first survey paper focusing on the design architecture of DL compilers, which we hope can pave the road for future research towards DL compiler.
>  DL 编译器接受不同 DL 框架下的 DL 模型作为输入，为不同的 DL 硬件生成优化的代码 (作为输出)
>  本文对目前 DL 编译器的设计细节进行剖析，着重描述面向 DL 的多层 IR 和前后端优化，聚焦 DL 编译器的设计架构

Additional Key Words and Phrases: Neural Networks, Deep Learning, Compiler, Intermediate Representation, Optimization

# 1 Introduction
The development of deep learning (DL) has generated profound impact on various scientific fields. It has not only demonstrated remarkable value in artificial intelligence such as natural language processing (NLP) [64] and computer vision (CV) [26], but also proved great success in broader applications such as e-commerce [36], smart city [68] and drug discovery [15]. With the emergence of versatile deep learning models such as convolutional neural network (CNN) [54], recurrent neural network (RNN) [80], long short-term memory (LSTM) [38] and generative adversarial network (GAN) [29], it is critical to ease the programming of diverse DL models in order to realize their widely adoption.

With the continuous efforts from both industry and academia, several popular DL frameworks have been proposed such as TensorFlow [1], PyTorch [75], MXNet [16] and CNTK [81], in order to simplify the implementation of various DL models. Although there are strengths and weaknesses among the above DL frameworks depending on the tradeoffs in their designs, the interoperability becomes important to reduce the redundant engineering efforts when supporting emerging DL models across the existing DL models. To provide interoperability, ONNX [66] has been proposed, that defines a unified format for representing DL models to facilitate model conversion between different DL frameworks.
>  DL 框架包括 TensorFlow, PyTorch, MXNet, CNTK 等
>  各个框架各有优劣，ONNX 为各框架提供了互操作性
>  ONNX 定义了表示 DL 模型的统一格式，以促进不同 DL 框架之间的模型转换

In the meanwhile, the unique computing characteristics such as matrix multiplication have spurred the passion of chip architects to design customized DL accelerators for higher efficiency. Internet giants (e.g., Google TPU [44], Hisilicon NPU [56], Apple Bonic [49]), processor vendors (e.g., NVIDIA Turing [72], Intel NNP [41]), service providers (e.g., Amazon Inferentia [8], Alibaba Hanguang [7]), and even startups (e.g., Cambricon [57], Graphcore [43]) are investing tremendous workforce and capital in developing DL chips in order to boost the performance for DL models. Generally, the DL hardware can be divided into the following categories: 1) general-purpose hardware with software-hardware co-design, 2) dedicated hardware fully customized for DL models, and 3) neuromorphic hardware inspired by biological brain science. For example, the general purpose hardware (e.g., CPU, GPU) has added special hardware components such as AVX512 vector units and tensor core to accelerate DL models. Whereas for dedicated hardware such as Google TPU, application-specific integrated circuits (e.g., matrix multiplication engine and high-bandwidth memory) have been designed to elevate the performance and energy efficiency to extreme. To the foreseeable future, the design of DL hardware would become even more diverse.  
>  DL 硬件可以分为三类
>  1. 通用目的的硬件，硬件和软件协同设计
>  2. 为 DL 模型定制的专有硬件
>  3. 类脑硬件
>  例如，通用目的的硬件添加了特别的硬件成分例如 AVX512 向量单元和 tensor core 以加速 DL 模型；专有硬件例如 Google TPU、针对应用的集成电路 (矩阵乘引擎和高带宽内存)

To embrace the hardware diversity, it is important to map the computation to DL hardware efficiently. On general-purpose hardware, the highly optimized linear algebra libraries such as Basic Linear Algebra Subprograms (BLAS) libraries (e.g., MKL and cuBLAS) serve as the basics for efficient computation of DL models. Take the convolution operation for example, the DL frameworks convert the convolution to matrix multiplication and then invoke the GEMM function in the BLAS libraries. In addition, the hardware vendors have released specially optimized libraries tailored for DL computations (e.g., MKL-DNN and cuDNN), including forward and backward convolution, pooling, normalization, and activation. More advanced tools have also been developed to further speedup the DL operations. For example, TensorRT [73] supports graph optimization (e.g., layer fusion) and low-bit quantization with large collection of highly optimized GPU kernels. On dedicated DL hardware, similar libraries are also provided [43 , 57]. However, the drawback of relying on the libraries is that they usually fall behind the rapid development of DL models, and thus fail to utilize the DL chips efficiently.
>  硬件存在多样性，因此需要考虑将计算高效映射到硬件
>  通用目的硬件上，基础线性代数子程序库 (MKL, cuBLAS) 为 DL 模型提供高效计算，以卷积运算为例，DL 框架将卷积运算转化为矩阵乘，然后调用 BLAS 库中的 GEMM 函数执行矩阵乘
>  同时，硬件厂商还发布了特别针对 DL 计算 (前向、后向卷积、池化、规范化、激活) 的库 (MKL-DNN, cuDNN) 为 DL 模型提供高效计算
>  同时更高级的工具正在开发中，以进一步加速 DL 运算，例如 TensorRT 支持图优化 (例如层融合) 和低比特量化，并且有大量高度优化的 GPU kernel
>  专用硬件上，同样存在类似的库
>  库的劣势在于库的开发落后于 DL 模型的进展，故难以高效利用 DL 芯片

To address the drawback of DL libraries and tools, as well as alleviate the burden of optimizing the DL models on each DL hardware manually, the DL community has resorted to the domain specific compilers for rescue. Rapidly, several popular DL compilers have been proposed such as TVM [17], Tensor Comprehension [91], Glow [79], nGraph [21] and XLA [53], from both industry and academia. The DL compilers take the model definitions described in the DL frameworks as inputs, and generate efficient code implementations on various DL hardware as outputs. The transformation between model definition and specific code implementation are highly optimized targeting the model specification and hardware architecture. Specifically, they incorporate DL oriented optimizations such as layer and operator fusion, which enables highly efficient code generation. Moreover, existing DL compilers also leverage mature tool-chains from general-purpose compilers (e.g., LLVM [51]), which provides better portability across diverse hardware architectures. Similar to traditional compiler, DL compilers also adopt the layered design including frontend, intermediate representation (IR) and backend. However, the uniqueness of DL compiler lies in the design of multi-level IRs and DL specific optimizations.
>  为了解决 DL 库和工具的缺陷，同时缓解在各个 DL 硬件上手动优化 DL 模型的负担，社区开始设计领域特定编译器
>  DL 编译器包括 TVM, Tensor Comprehension, Glow, nGraph, XLA
>  DL 编译器接受 DL 框架对 DL 模型的描述作为输入，为各个 DL 硬件生成高效的代码实现 (作为输出)
>  从模型定义到特定代码实现的转化是针对模型规格和硬件架构高度优化的，这些优化包括面向 DL 的优化例如层和算子融合
>  现存 DL 编译器也利用通用目的编译器 (例如 LLVM) 的成熟工具链，以提供更高的跨硬件架构可以执行
>  DL 编译器和传统编译器类似，也采取层级设计，包括前端、中间表示、后端，DL 编译器的独特性在于它的多层 IR 和针对 DL 的优化

In this paper, we provide a comprehensive survey of existing DL compilers by dissecting the compiler design into frontend, multi-level IRs and backend, with special emphasis on the IR design and optimization methods. To the best of our knowledge, this is the first paper that provides a comprehensive survey on the design of DL compiler. 
>  本文对现存 DL 编译器设计在前端、多层 IR、后端的设计进行调查，同时重点关注 IR 设计和优化方法

Specifically, this paper makes the following contributions:
>  本文贡献如下

We dissect the commonly adopted design architecture of existing DL compilers, and provide 
- detailed analysis of the key design components such as multi-level IRs, frontend optimizations (including node-level, block-level and dataflow-level optimizations) and backend optimizations (including hardware-specific optimization, auto-tuning and optimized kernel libraries). 
- We provide a comprehensive taxonomy of existing DL compilers from various aspects, which corresponds to the key components described in this survey. The target of this taxonomy is to provide guidelines about the selection of DL compilers for the practitioners considering their requirements, as well as to give a thorough summary of the DL compilers for researchers.  
- We have provided the quantitative performance comparison among DL compilers on CNN models, including full-fledged models and lightweight models. We have compared both end-to-end and per-layer (convolution layers since they dominate the inference time) performance to show the effectiveness of optimizations. The evaluation scripts and results are open sourced for reference. 
- We highlight several insights for the future development of DL compilers, including dynamic shape and pre-/post-processing, advanced auto-tuning, polyhedral model, subgraph partitioning, quantization, unified optimizations, differentiable programming and privacy protection, which we hope to boost the research in the DL compiler community.

>  本文分析了目前 DL 编译器采取的设计架构，并提供
>  - 对关键设计成分，例如多层 IR、前端优化 (包括 node 级别、block 级别、数据流级别优化) 、后端优化 (包括针对硬件的欧化、自动微调和优化的 kernel 库) 的详细分析
>  - 对现存 DL 编译器的详细分类
>  - 现存 DL 编译器在 CNN 模型 (包括完整模型和轻量模型) 上的性能比较，性能比较包括端到端的和层之间的
>  - DL 编译器的未来发展方向，包括动态形状、预处理/后处理，高级自动微调，polyhedral 模型，子图划分，量化，统一优化，可微编程，隐私保护

The rest of this paper is organized as follows. Section 2 presents the background of DL compilers, including the DL frameworks, DL hardware, as well as hardware (FPGA) specific DL code generators. Section 3 describes the common design architecture of DL compilers. Section 4 discusses the key components of DL compilers, including multi-level IRs, frontend optimizations and backend optimizations. Section 5 presents a comprehensive taxonomy. Section 6 provides the quantitative performance comparison. Section 7 highlights the future directions for DL compiler research.

# 2 Background
## 2.1 Deep Learning Frameworks
In this section, we provide an overview of popular DL frameworks. The discussion might not be exhaustive but is meant to provide a guideline for DL practitioners. Figure 1 presents the landscape of DL frameworks including currently popular frameworks, historical frameworks and ONNX supported frameworks.
>  本节对 DL 框架进行概览

**TensorFlow** - Among all the DL frameworks, TensorFlow has the most comprehensive support for language interfaces, including C++ , Python, Java, Go, R, and Haskell. TensorFlow employs a dataflow graph of primitive operators extended with restricted control edges to represent differentiable programs [78]. TensorFlow Lite is designed for mobile and embedded deep learning and provides an Android neural network API. To reduce the complexity of using TensorFlow, Google adopts Keras as a frontend to the TensorFlow core. Furthermore, The eager-mode in TensorFlow applies an approach similar to PyTorch to support dynamic computation graphs better.
>  TensorFlow 支持的语言接口最广泛，TensorFlow 用受限的控制边对原始算子的数据流图进行拓展，以表示可微程序
>  TensorFlow Lite 面向移动和嵌入式 DL，提供了 Android NN API
>  Google 使用 Keras 作为 TensorFlow 内核的前端
>  TensorFlow 的 eager-mode 采用类似于 PyTorch 的方法支持动态计算图

**Keras** - Keras [19] is a high-level neural network library for quickly building DL models, written in pure Python. Though not a DL framework on its own, Keras provides a high-level API that integrates with TensorFlow, MXNet, Theano, and CNTK. With Keras, DL developers can build a neural network with just a few lines of code. Besides, Keras can integrate with other common DL packages, such as scikit-learn. However, Keras is not flexible enough due to over-encapsulation, which makes it too difficult to add operators or obtain low-level data information.
>  Keras 为快速构建 DL 模型的高级 NN 库，以纯 Python 写成
>  Keras 本身不是 DL 框架，但提供了集成了 TensorFlow, MXNet, Theano, CNTK 的高级 API
>  Keras 可以和其他常见 DL 包集成，例如 scikit-learn
>  Keras 封装度较高，不够灵活

**PyTorch** - Facebook has rewritten the Lua-based DL framework Torch in Python and refactored all modules on Tensor level, which leads to the release of PyTorch. As the most popular dynamic framework, PyTorch embeds primitives for constructing dynamic dataflow graphs in Python, where the control flow is executed in the Python interpreter. PyTorch 1.0 integrated the codebases of PyTorch 0.4 and Caffe2 to create a unified framework. This allows PyTorch to absorb the benefits of Caffe2 to support efficient graph execution and mobile deployment. FastAI [39] is an advanced API layer based on PyTorch’s upper-layer encapsulation. It fully borrows Keras to ease the use of PyTorch.
>  Facebook 将基于 Lua 的 DL 框架 Torch 用 Python 重写为 PyTorch，在 Tensor 级别重构了所有模块
>  PyTorch 将构建动态数据流图的原语嵌入 Python，因此控制流由 Python 解释器执行
>  PyTorch 1.0 统一集成了 PyTorch 0.4 和 Caffe 2，进而支持高效的图执行和移动部署
>  FastAI 基于 PyTorch 的高层封装构建更高级的 API 层，类似 Keras

**Caffe/Caffe2** - Caffe [42] was designed for deep learning and image classification by UC Berkeley. Caffe has the command line, Python, and MATLAB APIs. Caffe’s simplicity makes the source codes easy to extend, which is suitable for developers to analyze in-depth. Therefore, Caffe is mainly positioned in research, which has made it popular from the beginning to the present. Caffe2 is built upon the original Caffe project. Caffe2 is similar to TensorFlow in code structure, albeit with a lighter API and easier access to the intermediate results in the computation graph.
>  Caffe 由 UCB 设计用于 DL 和图像分类
>  Caffe 2 基于 Caffe，代码结构类似 TensorFlow，但 API 更轻量，且更易于访问计算图中的中间结果

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/54cee69ca17adb87b03c3efafea0fe195a0decf480870d933efb03600581173f.jpg)  

Fig. 1. DL framework landscape: 1) Currently popular DL frameworks; 2) Historical DL frameworks; 3) ONNX supported frameworks.

**MXNet** - MXNet supports multiple language APIs including Python, C++ , R, Scala, Julia, Matlab, and JavaScript. It was intended to be scalable and was designed from the perspective to reduce data loading and I/O complexity [16]. MXNet offers different paradigms: declarative programming like Caffe and Tensorflow as well as imperative like PyTorch. In December 2017, Amazon and Microsoft jointly released Gluon [69] based on MXNet, which is an advanced interface similar to Keras and FastAI. Gluon supports both flexible, dynamic graphs and efficient, static graphs.
>  MXNet 支持多语言 API，其设计目的是减少数据装载和 IO 复杂度
>  Gluon 是 MXNet 的高级接口，类似 Keras, FastAI

**CNTK** - CNTK can be used through Python, $C++$ and $C\#$ APIs, or its own scripting language (i.e., Brain Script). CNTK is designed to be easy-to-use and production-ready for large-scale data in production [37]. However, CNTK does not yet support the ARM architecture, which limits its usage on mobile devices. It uses the static computation graph similar to TensorFlow and Caffe, in which a DL model is treated as a series of computational steps through a directed graph.
>  CNTK 不支持 ARM 结构，故不能用于移动设备
>  CNTK 使用静态计算图，类似 TensorFlow 和 Caffe
>  静态计算图中，DL 模型被视作有向图中的一系列计算步骤

**Paddle Paddle** - The original design of Paddle Paddle [11] is similar to Caffe, where each model can be represented as a set of layers. However, Paddle Paddle v2 has adopted the concept of operators with reference to TensorFlow, which breaks layers into finer-grained operators, thereby supporting more complex DL models. And Paddle Paddle Fluid is similar to PyTorch because it provides own interpreter so as to avoid the limited performance of Python interpreter.
>  Paddle Paddle 的初始设计中，每个模型由一组层表示，和 Caffe 类似
>  Paddle Paddle v2 参考 TensorFlow，采用算子的概念，将层划分到更细粒度的算子
>  Paddle Paddle Fluid 提供了自己的解释器

**ONNX** - The Open Neural Network Exchange (ONNX) [66] defines a scalable computation graph model, and thus computation graphs built by different DL frameworks can be easily transformed into ONNX. With ONNX, it becomes easier to convert models between DL frameworks. For example, it allows developers to build an MXNet model and then run the model using PyTorch for inference. As shown in Figure 1, ONNX has been integrated into PyTorch, MXNet, Paddle Paddle, and so on. For several DL frameworks (e.g., TensorFlow and Keras) that are not directly supported yet, and ONNX adds converters to them.
>  ONNX 定义了可拓展的计算图模型，不同 DL 框架的计算图可以转化为 ONNX 并由 ONNX 转化而来

**Historical Frameworks** - Due to the rapid evolvement in DL community, many historical DL frameworks are no longer active. For example, PyTorch has replaced Torch [20]. As one of the oldest DL frameworks, Theano [86] is no longer under maintenance. Deeplearning4J [85] a distributed DL framework based on Java and Scala, however becomes inactive due to the lack of large developer community. Chainer [87] was once the preferred framework for dynamic computation graphs, however replaced by MXNet, PyTorch and TensorFlow with similar features.

Previous works [10 , 25 , 35 , 70 , 82 , 100] have compared the performance of DL frameworks on different applications (e.g., computer vision and image classification) and different hardware (e.g., CPU, GPU, and TPU). For detailed information about each DL framework, the readers can refer to [37]. Different from them, this survey focuses on the research efforts on DL compilers which provide more general approach to execute various DL models on diverse hardware efficiently.

## 2.2 Deep Learning Hardware
The DL hardware can be divided into three categories based on the generality: 1) general-purpose hardware that can support DL workloads through hardware and software optimization; 2) dedicated hardware that focus on accelerating DL workloads with fully customized circuit design; 3) neuromorphic hardware that function by mimicking the human brain.
>  DL 硬件分为三类
>  - 通用目的硬件，通过硬件和软件优化支持 DL workload
>  - 针对 DL workload 的专有硬件
>  - 类脑硬件，通过模仿人类大脑工作

**General-purpose Hardware** - The most representative general-purpose hardware for DL models is Graphic Processing Unit (GPU), which achieves high parallelism with many-core architecture. For example, Nvidia GPUs have introduced tensor cores since the Volta architecture. Tensor cores can accelerate mixed-precision matrix multiply-and-accumulate calculations in parallel, which are widely used in DL models during both training and inference. Co-optimized with the hardware, NVIDIA also launches highly optimized DL libraries and tools such as cuDNN [18] and TensorRT [73] to further accelerate the computation of DL models.
>  代表性为 GPU，由多核结构实现高并行度
>  NV GPU 在 Volta 架构中引入了 Tensor core，用于并行加速混合精度矩阵乘累加计算
>  NV 的 DL 库和工具有 cuDNN, TensorRT

**Dedicated Hardware** - Dedicated hardware is fully customized for DL computation to improve performance and energy efficiency to extreme. The rapid expansion of DL applications and algorithms has spurred many startups developing dedicated DL hardware (e.g., Graphcore GC2, Cambricon MLU270). Besides, traditional hardware companies (e.g., Intel NNP, Qualcomm Cloud AI 100) and cloud service providers (e.g., Google TPU, Amazon Inferentia, and Alibaba Hanguang) have also invested in this field. The most well known dedicated DL hardware is Google’s TPU series. A TPU includes Matrix Multiplier Unit (MXU), Unified Buffer (UB), and Activation Unit (AU), which is driven with CISC instructions by the host processor. The MXU is mainly composed of a systolic array, which is optimized for power and area efficiency in performing matrix multiplications. Compared to CPU and GPU, TPU is still programmable but uses a matrix as a primitive instead of a vector or scalar. The Amazon Inferentia has also attracts the attention recently. This chip has four NeuroCores that are designed for tensor-level operations, and it has large on-chip cache to avoid the frequent main memory access.
>  专有硬件例如 Graphcore GC2, Cambricon MLU270, Intel NNP, Qualcomm Cloud AI 100, Google TPU, Amazon Infernia, Alibaba Hanguang
>  TPU 包含矩阵乘法单元 MXU, 联合缓存 UB, 激活单元 AU, 由主机处理器通过 CISC 指令驱动
>  MXU 由一个脉动阵列构成
>  TPU 可编程，使用矩阵作为基本元素，而不是标量或者向量
>  Amazon Inferentia 有 4 个 NeuroCores 用于 tensor 级别运算，以及大面积片上缓存

**Neuromorphic Hardware** - Neuromorphic chips use electronic technology to simulate the biological brain. Representative products of the this kind are IBM’s TrueNorth and Intel’s Loihi. Neuromorphic chips (e.g., TrueNorth) have very high connectivity between their artificial neurons. Neuromorphic chips also replicate a structure similar to the brain tissue: neurons can simultaneously store and process the data. Traditional chips distribute processors and memory in different locations, but Neuromorphic chips usually have many microprocessors, each of which has a small amount of local memory. Compared to TrueNorth, Loihi has a learning ability more similar to the brain. Loihi introduces the pulse-time-dependent synaptic plasticity model (STDP), a mechanism that regulates synaptic strength by the relative time of pre-synaptic and post-synaptic pulses. However, Neuromorphic chips are far away from Large-scale commercial production. Despite that, in computer science domain, Neuromorphic chips can help to capture the process of rapid, life-long learning which is ignored by regular DL models, and in neurology domain, they are helpful to figure out how the various parts of the brain work together to create thoughts, feelings, and even consciousness.
>  类脑芯片用电子技术模拟生物大脑
>  代表有 IBM TrueNorth, Intel Loihi，类脑芯片的人工神经元高度连接，并且复刻了大脑组织类似的结构：神经元同时储存和处理数据
>  传统芯片的处理器和内存在不同位置，类脑芯片有多个微处理器，每个微处理器有少量局部内存
>  Loihi 引入了脉冲时间依赖的突触可塑性模型，它通过突触前脉冲和突触后脉冲的时间差来调节突触强度
>  类脑芯片尚不能大规模商业部署

## 2.3 Hardware-specific DL Code Generator
Field Programmable Gate Arrays (FPGAs) are reprogrammable integrated circuits that contain an array of programmable logic blocks. Programmers can configure them after manufacturing. Besides the reprogrammable nature, the low-power and high-performance nature of the FPGA make it widely used in so many domains, such as communication, medical, image processing, and ASIC prototyping. As for the domain of deep learning, the high-performance CPUs and GPUs are highly-re programmable but power-hungry, while the power-efficient ASICs are specialized for fixed applications. However, the FPGA can bridge the gap between CPUs/GPUs and ASICs, which causes the FPGA to be an attractive platform for deep learning.  
The High-Level Synthesis (HLS) programming model enables the FPGA programmers to generate effective hardware designs conveniently using high-level languages such as C and $\mathrm{C++}$ . It avoids writing lots of Verilog or VHDL descriptions, which lowers the programming threshold and reduces the long design circle. Xilinx Vivado HLS and Intel FPGA SDK for OpenCL are two of the popular HLS tools targeting their own FPGAs. However, mapping DL models to FPGAs remains a complicated work even with HLS, because that 1) DL models are usually described by the languages of DL frameworks rather than bare mental $\mathrm{C}/\mathrm{C++}$ code, and 2) DL-specific information and optimizations are hard to be leveraged.
>  FPGA 为可重编程的集成电路，由可编程的逻辑块阵列组成
>  GPU 和 CPU 的能耗较大但通用，ASIC 则节能但专用，FPGA 介于二者之间
>  HLS 编程模型提供 C/C++ 接口来编程 FPGA，免于写 Verilog 或 VHDL
>  两大流行的 HLS 工具为 Xilinx Vivado HLS 和 Intel FPGA SDK for OpenCL
>  将 DL 模型映射到 FPGA 仍然复杂，因为 DL 模型一般用 DL 框架描述，而不是 C/C++；以及针对 DL 的信息和优化难以利用

The hardware-specific code generator targeting FPGA take the DL models or their domain-specific languages (DSLs) as the input, conduct the domain-specific (about FPGA and DL) optimizations and mappings, then generate the HLS or Verilog/VHDL and finally generate the bitstream. They can be classified into two categories according to the generated architectures of FPGA-based accelerators: the processor architecture and the streaming architecture [93].
>  针对硬件的 FPGA 代码生成器，接受 DL 模型或者其领域特定语言 (DSLs) 作为输入，执行领域特定的 (关于 FPGA 和 DL) 优化和映射，生成 HLS 或 Verilog/VHDL，最后再生成 bitstream
>  根据针对的基于 FPGA 的架构，这些代码生成器可以分为
>  - 针对处理器架构
>  - 针对流架构

**The processor architecture** has similarities with general-purpose processors. An FPGA accelerator of this architecture usually comprises several Processing Units (PUs), which are comprised of on-chip buffers and multiple smaller Processing Engines (PEs). It usually has a virtual instruction set (ISA), and the control of hardware and the scheduling of the execution should be determined by software. What’s more, the static scheduling method avoids the overheads of von Neumann execution (including instruction fetching and decoding). 
>  处理器架构类似通用目的处理器，该架构的 FPGA 加速设备由多个处理单元 PU 构成，每个 PU 包括片上缓存和多个更小的处理引擎 PE
>  这类加速设备一般有一个虚拟指令集架构，硬件的控制和执行调度由软件决定
>  这类加速设备的静态调度方法避免了 von Neumann 架构的执行开销 (取指令和解码)

A hardware template is a generic and fundamental implementation with configurable parameters. The DL code generator targeting this architecture adopt the hardware templates to generate the accelerator designs automatically. With the configurable parameters of templates, the code generator achieve the scalability and flexibility [104]. The scalability means that the code generator can generate designs for FPGAs ranging from high-performance to power-efficient, and the flexibility means that the code generator can generate designs for various DL models with different layer types and parameters. The number of PUs and the number of PEs per PU are template parameters of importance. Besides, the tilling size and batch size are also essential scheduling parameters about mapping the DL models to PUs and PEs. All these parameters are usually determined by the design space exploration using various strategies, such as combining the performance model and auto-tuning. 
>  硬件模板是具有可配置参数的泛型和基础实现，目标是处理器架构的 DL 代码生成器采用硬件模板来自动生成加速器设计
>  硬件模板的可配置参数带来了可拓展性和灵活性，可拓展性意思是代码生成器可以生成高性能的 FPGA 设计，也可以生成低功耗的 FPGA 设计；灵活性意思是代码生成器可以为不同的 DL 模型 (层不同，参数不同) 生成 FPGA 设计
>  PUs 的数量和每个 PU 上 PEs 的数量是重要的模板参数，将 DL 模型映射到 PUs 和 PEs 上时，tiling size 和 batch size 则是必要的调度参数
>  这些参数的决定都通过设计空间探索实现

DNN Weaver [83], AngelEye [33], ALAMO [63], FP-DNN [32], SysArrayAccel [101] are typical FPGA DL code generator targeting the processor architecture. What’s more, the PUs and PEs are usually responsible for coarse-grained basic operations such as matrix-vector multiplication, matrix-matrix multiplication, pooling, and some element-wise operations. The optimizations of these basic operations are mainly guided by the tradeoff between the parallelism and data reuse, which is similar to general optimizations.
>  针对处理器架构的 FPGA DL 代码生成器有 DNN Weaver, AngelEye, ALAMO, FP-DNN, SysArrayAccel
>  另外，PUs 和 PEs 一般还负责粗粒度的基本运算，包括矩阵-向量乘、矩阵-矩阵乘、池化、一些逐元素操作
>  这些基本运算的优化由并行性和数据重用性之间的 tradeoff 主导，和通用优化类似

**The streaming architecture** has similarities with pipelines. An FPGA accelerator of this architecture consists of multiple different hardware blocks, and it nearly has one hardware block for each layer of an input DL model. With the input data of a DL model, this kind of accelerators process the data through the different hardware blocks in the same sequence with layers. Additionally, with the streaming input data, all hardware blocks can be fully utilized in a pipeline manner. However, the streaming architecture usually follows an initial assumption that the on-chip memory and the computation resources on target FPGA are sufficient to accommodate the DL models, which bring barriers to deploy deep models with complicated layers. The DL code generator targeting this architecture can solve this problem by leveraging the re configurability of FPGA or adopting dynamic control flow. And the further optimization of a single block resembles that of basic operations of the processor architecture. fpgaConvNet [92], Deep Burning [98], Haddoc2 [2], and Auto Code Gen [59] are typical corresponding DL code generator.
>  流式架构类似流水线，流式架构的 FPGA 加速器由多个不同的硬件块组成
>  对于输入的 DL 模型，一般其每一层都对应一个硬件块，数据按照层的顺序在一个个的硬件块中处理
>  对于流式输入的数据，可以以流水线的方式利用硬件块
>  流式架构一般遵循一个假设：目标 FPGA 的片上内存和计算资源足以容纳 DL 模型，故部署带有复杂层的 DL 模型较为困难
>  要解决这一问题，代码生成器可以利用 FPGA 的可重配置性，或采用动态控制流
>  对单个块的进一步优化类似于针对处理器架构上的优化
>  针对流式架构的代码生成器有 fpgaConvNet, Deep Burning, Haddoc2, Auto Code Gen

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/00e875fda4506a93cefa6b3c32fbd9955488abd20d3e10ed81aa57fee6a66667.jpg)  
Fig. 2. The overview of commonly adopted design architecture of DL compilers.

For the detailed survey of specific compilation techniques that map DL models to FPGAs, the readers can refer to [34 , 93 , 104]. Different from [34 , 93 , 104], this survey focuses on general DL compilation techniques that can be applied to broader DL hardware other than bounding to FPGA.

# 3 Common Design Architectures of DL Compilers
The common design architecture of a DL compiler primarily contains two parts: the compiler frontend and the compiler backend, as shown in Figure 2. The intermediate representation (IR) is spread across both the frontend and the backend. Generally, IR is an abstraction of the program and is used for program optimizations. Specifically, the DL models are translated into multi-level IRs in DL compilers, where the high-level IR resides in the frontend, and the low-level IR resides in the backend. Based on the high-level IR, the compiler frontend is responsible for hardware-independent transformations and optimizations. Based on the low-level IR, the compiler backend is responsible for hardware-specific optimizations, code generation, and compilation. Note that this survey focuses on the design principles of DL compilers. For functional and experimental comparisons of DL compilers, the readers can refer to [55, 102].
>  DL 编译器常见的设计结构主要包含两部分：编译器前端、编译器后端
>  IR 贯穿前端和后端，一般地说，IR 是程序的抽象，并且会被用于程序优化
>  特别地，DL 模型会被 DL 编译器转化为多层 IR，其中高层 IR 位于前端，低层 IR 位于后端
>  编译器前端基于高层 IR 负责独立于硬件的转换和优化，编译器后端基于低层 IR 负责针对硬件的优化、代码生成、编译

**The high-level IR** , also known as graph IR, represents the computation and the control flow and is hardware-independent. The design challenge of high-level IR is the ability of abstraction of the computation and the control flow, which can capture and express diverse DL models. The goal of the high-level IR is to establish the control flow and the dependency between the operators and the data, as well as provide an interface for graph-level optimizations. It also contains rich semantic information for compilation as well as offers extensibility for customized operators. The detailed discussion of high-level IR is presented in Section 4.1.  
>  高层 IR 即图 IR，它表示独立于硬件的计算和控制流
>  高层 IR 的设计挑战在于它能否对计算和控制流进行抽象，使得它可以捕获和表示不同的 DL 模型
>  高层 IR 的目的是建立算子和数据之间的控制流和依赖，同时为**图级别的优化**提供接口
>  高层 IR 也包含了用于编译的丰富语义信息同时为自定义算子提供了可拓展性

**The low-level IR** is designed for hardware-specific optimization and code generation on diverse hardware targets. Thus, the low-level IR should be fine-grained enough to reflect the hardware characteristics and represent the hardware-specific optimizations. It should also allow the use of mature third-party tool-chains in compiler backends such as Halide [77], polyhedral model [31], and LLVM [51]. The detailed discussion of low-level IR is presented in Section 4.2.
>  低层 IR 被设计用于针对硬件的优化和对于不同硬件目标的代码生成
>  因此，低层 IR 应该足够细粒度，以反映硬件特征并表示针对硬件的优化
>  低层 IR 应该允许使用成熟的第三方编译器后端工具链，例如 Halide, polyhedral model, LLVM

**The frontend** takes a DL model from existing DL frameworks as input, and then transforms the model into the computation graph representation (e.g., graph IR). The frontend needs to implement various format transformations To support the diverse formats in different frameworks. The computation graph optimizations incorporate the optimization techniques from both general-purpose compilers and the DL specific optimizations, which reduce the redundancy and improve the efficiency upon the graph IR. Such optimizations can be classified into node-level (e.g., nop elimination and zero-dim-tensor elimination), block-level (e.g., algebraic simplification, operator fusion, and operator sinking) and dataflow-level (e.g., CSE, DCE, static memory planning, and layout transformation). After the frontend, the optimized computation graph is generated and passed to the backend. The detailed discussion of the frontend is presented in Section 4.3.
>  前端接受现有 DL 框架的 DL 模型作为输入，然后将模型转化为计算图表示 (例如图 IR)，前端需要实现多种格式的转化，以支持不同框架的不同格式
>  计算图优化涉及了通用编译器的优化技巧和针对 DL 的优化，计算图优化的目的是基于图 IR 减少冗余并且提高效率
>  计算图优化可以被分类为：
>  节点级别 (nop 消除, 零维张量消除)
>  块级别 (代数简化、算子融合、算子沉降)
>  数据流级别 (公共子表达式消除 CSE、死代码消除 DCE、静态内存规划、布局转换)
>  前端生成优化的计算图，传递给后端

**The backend** transforms the high-level IR into low-level IR and performs hardware-specific optimizations. On the one hand, it can directly transform the high-level IR to third-party toolchains such as LLVM IR to utilize the existing infrastructures for general-purpose optimizations and code generation. On the other hand, it can take advantage of the prior knowledge of both DL models and hardware characteristics for more efficient code generation, with customized compilation passes. The commonly applied hardware-specific optimizations include hardware intrinsic mapping, memory allocation and fetching, memory latency hiding, parallelization as well as loop oriented optimizations. To determine the optimal parameter setting in the large optimization space, two approaches are widely adopted in existing DL compilers such as auto-scheduling (e.g., polyhedral model) and auto-tuning (e.g., AutoTVM). The optimized low-level IR is compiled using JIT or AOT to generate codes for different hardware targets. The detailed discussion of the backend is presented in Section 4.4.
>  后端将高级 IR 转化为低级 IR，然后执行针对硬件的优化
>  一方面，后端可以直接将高级 IR 转化为第三方工具链例如 LLVM IR，利用现存工具进行通用目的的优化和代码生成
>  另一方面，它可以利用 DL 模型和硬件特性的先验知识通过自定义的编译流程进行更高效的代码生成；常用的针对硬件的优化包括硬件内建函数映射、内存分配和获取、内存延迟隐藏、并行化、面向循环的优化
>  为了在大的优化空间决定最优参数，现存的 DL 编译器广泛采用两个方法：自动调度 (例如多面体模型)、自动调节 (例如 AutoTVM)
>  优化后的低级 IR 可以使用 JIT (Just in Time) 编译或 AOT (Ahead of Time) 编译以为不同的硬件目标生成代码

# 4 Key Components of DL Compilers
## 4.1 High-level IR
To overcome the limitation of IR adopted in traditional compilers that constrains the expression of complex computations used in DL models, existing DL compilers leverage high-level IR (as known as graph IR) with special designs for efficient code optimizations. To better understand the graph IR used in the DL compilers, we describe the representation and implementation of graph IR as follows.
>  传统编译器采用的 IR 约束了对 DL 模型中复杂计算的表达，现存的 DL 编译器使用高级 IR (图 IR)

### 4.1.1 Representation of Graph IR
The representation of graph IR influences the expressiveness of graph IR and also decides the way the DL compilers analyze the graph IR.
>  图 IR 的表示影响图 IR 的表达能力，并且决定了 DL 编译器如何分析图 IR

**DAG-based IR** - DAG-based IR is one of the most traditional ways for the compilers to build a computation graph, with nodes and edges organized as a directed acyclic graph (DAG). In DL compilers [17 , 21 , 53 , 79 , 91], the nodes of a DAG represent the atomic DL operators (convolution, pooling, etc.), and the edges represent the tensors. And the graph is acyclic without loops, which differs from the data dependence graphs [50] (DDG) of generic compilers [51 , 52]. And with the help of the DAG computation graph, DL compilers can analyze the relationship and dependencies between various operators and use them to guide the optimizations. There are already plenty of optimizations on DDG, such as common sub-expression elimination (CSE) and dead code elimination (DCE). By combining the domain knowledge of DL with these algorithms, further optimizations can be applied to the DAG computation graph, which will be elaborated in Section 4.3. DAG-based IR is convenient for programming and compiling due to its simplicity, but it has deficiencies such as semantic ambiguity caused by the missing definition of computation scope.  
>  基于 DAG 的 IR
>  基于 DAG 的 IR 是编译器构建计算图最传统的方式
>  DL 编译器用 DAG 中的一个节点表示原子 DL 算子 (卷积、池化等)，用 DAG 中的边表示 tensor
>  DAG 没有自环，这和通用编译器使用的数据依赖图 DDG 不同
>  DL 编译器通过 DAG 计算图分析不同算子之间的依赖关系，进而指导优化
>  DDG 相关的优化有很多，例如共同子表达式消除 CSE，死代码消除 DCE
>  DL 编译器将这些算法和 DL 领域知识结合，得到可以应用于 DAG 计算图的进一步优化算法
>  基于 DAG 的 IR 因为其简单性而便于编程和编译，但其缺陷在于缺少计算作用域的定义而会导致语义模糊性

**Let-binding-based IR** - Let-binding is one method to solve the semantic ambiguity by offering let expression to certain functions with restricted scope used by many high-level programming languages such as Javascript [30], $\mathrm{F}\#$ [76], and Scheme [3]. When using the let keyword to define an expression, a let node is generated, and then it points to the operator and variable in the expression instead of just building computational relation between variables as a DAG. In DAG-based compiler, when a process needs to get the return value of one expression, it first accesses the corresponding node and searches related nodes, also known as recursive descent technique. In contrast, the let-binding based compiler figures out all results of the variables in let expression and builds a variable map. When a particular result is needed, the compiler looks up this map to decide the result of the expression. Among the DL compilers, the Relay IR [78] of TVM adopts both DAG-based IR and let-binding-based IR to obtain the benefits of both.
>  基于 let 绑定的 IR
>  let 绑定是一种通过为许多高级编程语言 (例如 JS, F#, Scheme) 中具有受限作用域的函数提供 let 表达式来解决语义模糊的办法
>  当使用 `let` 关键字定义表达式时，会生成一个 let 节点，该节点会指向表达式中的运算符和变量而不是像 DAG 中仅构建变量之间的计算关系
>  在基于 DAG 的编译器中，但需要获得某个表达式的返回值时，它会首先访问对应的节点，然后搜索相关的节点，该技术称为递归下降技术
>  基于 let 绑定的编译器会计算出 let 表达式中所有变量的结果，并构建一个变量映射表，当需要特定结果时，编译器查询该映射表以决定表达式的值
>  Relay IR 和 TVM 同时采用了基于 DAG 和基于 let 绑定的 IR

**Representing Tensor Computation** - Different graph IRs have different ways to represent the computation on tensors. The operators of diverse DL frameworks are translated to graph IRs according to such specific representations. And the customized operators also need to be programmed in such representation. The representation of tensor computation can be divided into the following three categories.
>  不同的图 IR 表示 tensor 计算的方式不同，各种 DL 框架的算子会根据这些特定表示方式被转化为图 IR
>  另外，自定义的算子也需要在这种表示下编写
>  tensor 计算的表示分为以下三类

1. Function-based: The function-based representation just provides encapsulated operators, which is adopted by Glow, nGraph and XLA. Take High Level Optimizer (HLO, the IR of XLA) for example, it consists of a set of functions in symbolic programming, and most of them have no side-effect. The instructions are organized into three levels, including HloModule (the whole program), HloComputation (a function), and Hlo Instruction (the operation). XLA uses HLO IR to represent both graph IR and operation IR so that the operation of HLO ranges from the dataflow level to the operator level.
>  基于函数
>  Glow, nGraph, XLA 采用
>  基于函数的表示仅提供封装的算子
>  例如，HLO (XLA 的 IR) 由符号编程的一组函数组成，其中大多数函数都没有 side-effect
>  HLO 的指令分为三个级别：HloModule (整个程序)、HloComputation (一个函数)、HloInstruction (运算)
>  XLA 用 HLO IR 来表示图 IR 和运算 IR，故 HLO 的操作范围包括了数据流级别和算子级别

2. Lambda expression: The lambda expression, an index formula expression, describes calculation by variable binding and substitution. Using lambda expression, programmers can define a computation quickly without implementing a new function. TVM represents the tensor computation using the tensor expression, which is based on the lambda expression. In TVM, computational operators in tensor expression are defined by the shape of output tensor and the lambda expression of computing rules.
>  Lambda 表达式
>  lambda 表达式是一种索引公式表达式，通过变量绑定和替换来描述计算
>  TVM 使用 tensor 表达式来表示 tensor 计算，tensor 计算基于 lambda 表达式
>  TVM 中，tensor 表达式中的运算符由其输出 tensor 的大小和表示计算规则的 lambda 表达式定义

3. Einstein notation: The Einstein notation, also known as the summation convention, is a notation to express summation. Its programming simplicity is superior to lambda expression. Taking TC for example, the indexes for temporary variables do not need to be defined. The IR can figure out the actual expression by the occurrence of undefined variables based on Einstein notation. In Einstein notation, the operators need to be associative and commutative. This restriction guarantees the reduction operator can be executed by any order, making it possible for further parallelization.
>  Einstein 记号
>  Einstein 记号也称为求和约定，该记号用于表示求和，它比 lambda 表达式更简洁，因为不需要定义临时变量的索引
>  IR 通过 Einstein 记号中未定义变量的出现来推断真正的表达式
>  Einstein 记号中，算子应该是可结合且可交换的，这保证归约操作可以按任意顺序执行，为进一步并行化提供可能

### 4.1.2 Implementation of Graph IR
The implementation of graph IR in DL compilers fulfills the management of data and operation.
>  图 IR 的实现完成了数据和运算的管理

**Data representation** - The data in DL compilers (e.g., inputs, weights, and intermediate data) are usually organized in the form of tensors, which are also known as multi-dimensional arrays. The DL compilers can represent tensor data directly by memory pointers, or in a more flexible way by placeholders. A placeholder contains the size for each dimension of a tensor. Alternatively, the dimension sizes of the tensor can be marked as unknown. For optimizations, the DL compilers require the data layout information. In addition, the bound of iterators should be inferred according to the placeholders.
>  数据表示
>  DL 编译器中的数据 (输入、权重、中间数据) 一般以 tensor/多维数组的形式组织，DL 编译器可以直接用内存指针表示 tensor 数据
>  一种更灵活的方式是占位符，占位符包含了 tensor 每个维度的大小 (也可以标记为 unknown)，在优化时，DL 编译器需要数据布局信息，另外，还需要根据占位符推导迭代器的边界

1. **Placeholder**: Placeholder is widely used in symbolic programming (e.g., Lisp [65], Tensorflow [1]). A placeholder is simply a variable with explicit shape information (e.g., size in each dimension), and it will be populated with values at the later stage of the computation. It allows the programmers to describe the operations and build the computation graph without concerning the exact data elements, which helps separate the computation definition from the exact execution in DL compilers. Besides, it is convenient for the programmers to change the shape of input/output and other corresponding intermediate data by using placeholders without changing the computation definition.
>  占位符
>  占位符广泛用于符号编程 (Lisp, TensorFlow)，占位符本质上就是具有显式形状信息 (例如每一维度大小) 的变量，且它会在计算的后期阶段被赋予具体的值
>  占位符允许程序员在不关心实际数据元素的情况下描述运算并构建计算图，有助于在 DL 编译器中将计算定义和实际的计算执行分离
>  使用占位符时，程序员可以方便地更改输入/输出和其他中间数据的形状，而无序修改计算定义

2. **Unknown (Dynamic) shape representation**: The unknown dimension size is usually supported when declaring the placeholders. For instance, TVM uses Any to represent an unknown dimension (e.g., Tensor $\langle(A n y,3),f p32\rangle\rangle$ ); XLA uses None to achieve the same purpose (e.g., $t f$ . placeholder ( “ float ” , [None ,3]) ); nGraph uses its PartialShape class. The unknown shape representation is necessary to support the dynamic model. However, to fully support dynamic model, the bound inference and dimension checking should be relaxed. In addition, extra mechanism should be implemented to guarantee memory validity.
>  未知/动态形状表示
>  声明占位符时也支持未知的维度大小
>  例如 TVM 使用 $Any$ 表示未知维度 (e.g. $Tensor\langle (Any, 3), fp32\rangle$)；XLA 使用 $None$ (e.g. $tf. placeholder(\text {``} float\text", [None, 3])$)；nGraph 使用其 $PartialShape$ 类
>  未知形状的表示对于动态模型的支持是必须的，同时为了完全支持动态模型，需要松弛边界推理和维度检查，另外还需要额外的机制保证内存有效性

3. **Data layout**: The data layout describes how a tensor is organized in memory, and it is usually a mapping from logical indices to memory indices. The data layout usually includes the sequence of dimensions (e.g., NCHW and NHWC), tiling, padding, striding, etc. TVM and Glow represent data layout as operator parameters and require such information for computation and optimization. However, combining data layout information with operators rather than tensors enables intuitive implementation for certain operators and reduces the compilation overhead. XLA represents data layout as constraints related to its backend hardware. Relay and MLIR are going to add data layout information into their type systems for tensors.
>  数据布局
>  数据布局描述了 tensor 如何在内存中组织，数据布局通常是从逻辑索引到内存索引的映射
>  数据布局通常包括维度序列 (NCHW, NHWC), tiling, padding, striding 等
>  TVM 和 Glow 将数据布局表示为算子参数，在计算和优化时需要这些信息
>  将数据布局信息和算子结合而不是和 tensor 结合简化了特定算子的实现，且减少了编译开销
>  XLA 将数据布局表示为和后端硬件相关的约束
>  Relay 和 MLIR 的数据布局信息被添加到 tensor 的类型系统中

4. **Bound inference**: The bound inference is applied to determine the bound of iterators when compiling DL models in DL compilers. Although the tensor representation in DL compilers is convenient to describe the inputs and outputs, it exposes special challenges for inferring the iterator bound. The bound inference is usually performed recursively or iteratively, according to the computation graph and the known placeholders. For example, in TVM the iterators form a directed acyclic hyper-graph, where each node of the graph represents an iterator and each hyper-edge represents the relation (e.g., split , fuse or rebase ) among two or more iterators. Once the bound of the root iterator is determined based on the shapes of placeholders, other iterators can be inferred according to the relations recursively.
>  边界推理
>  边界推理用于在编译 DL 模型时决定迭代器的边界
>  张量表示便于描述输入输出，但不便于推导迭代器边界
>  边界推理通常根据计算图和已知占位符递归或者迭代式执行，例如，TVM 中，迭代器构成一个有向无环的超图，图中每个节点表示一个迭代器，每个边表示关系

**Operators supported** - The operators supported by DL compilers are responsible for representing the DL workloads, and they are nodes of the computation graph. The operators usually include algebraic operators (e.g., $+,\times,$ , exp and topK), neural network operators (e.g., convolution and pooling), tensor operators (e.g., reshape, resize and copy), broadcast and reduction operators (e.g., min and argmin), as well as control flow operators (e.g., conditional and loop). Here, we choose three representative operators that are frequently used across different DL compilers for illustration. In addition, we discuss the case for customized operators.
>  DL 编译器支持的算子需要用于表示 DL workload，它们将成为计算图中的节点
>  这些算子一般包括代数算子 (例如 $+, -, \exp, \text{topK}$)，NN 算子 (例如卷积、池化)，tensor 算子 (例如 reshape, resize, copy)，广播算子，归约算子 (例如 $\min, \arg\min$)，控制流算子 (例如条件和循环)

1. **Broadcast**: The broadcast operators can replicate the data and generate new data with compatible shape. Without broadcast operators, the input tensor shapes are more constrained. For example, for an add operator, the input tensors are expected to be of the same shape. Some compilers such as XLA and Relay relax such restriction by offering the broadcasting operator. For example, XLA allows the element-wise addition on a matrix and a vector by replicating it until its shape matches the matrix.
>  广播算子
>  广播算子可以复制数据并生成具有兼容形状的新数据
>  如果没有广播算子，则输入 tensor 的形状限制更大，例如输入 tensor 应该和加法算子的形状相同，XLA 和 Relay 提供了广播算子，松弛了这一约束
>  例如 XLA 允许对矩阵和向量进行逐元素加法，其操作是对向量进行广播，直到其形状匹配矩阵

2. **Control flow**: Control flow is needed when representing complex and flexible models. Models such as RNN and Reinforcement learning (RL) depend on recurrent relations and data-dependent conditional execution [103], which requires control flow. Without supporting control flow in graph IR of DL compilers, these models must rely on the control flow support of the host languages (e.g., $i f$ and while in Python) or static unrolling, which deteriorates the computation efficiency. Relay notices that arbitrary control flow can be implemented by recursion and pattern, which has been demonstrated by functional programming [78]. Therefore, it provides $i f$ operator and recursive function for implementing control flow. On the contrary, XLA represents control flow by special HLO operators such as while and conditional .
>  控制流
>  在表示复杂和灵活模型时，控制流是必要的
>  例如 RNN 和 RL 模型依赖于 recurrent 关系和数据依赖的条件执行，它们需要控制流来实现
>  如果 DL 的图 IR 不支持控制流，那么就需要依赖于宿主语言的控制流支持 (例如 Python 中的 `if, while` ) 或者静态展开，而这会降低计算效率
>  函数式编程证明了任意的控制流都可以通过递归和模式实现，因此 Relay 提供了 `if` 算子和递归函数来实现控制流
>  XLA 通过特殊的 HLO 算子例如 `while, conditional` 实现控制流

3. **Derivative**: The derivative operator of an operator $O p$ takes the output gradients and the input data of $O p$ as its inputs, and then calculates the gradient of $O p$ . Although some DL compilers (e.g., TVM and TC) support automatic differentiation [88], they require the derivatives of all operators in high-level IR when the chain rule is applied. TVM is working towards providing the derivative operators of both algebraic operators and neural network operators. The programmers can use these derivative operators for building the derivatives of customized operators. On the contrary, PlaidML can generate derivative operators automatically, even for customized operators. Notably, DL compilers unable to support derivative operators fail to provide the capability of model training.
>  导数
>  一个算子 `Op` 的导数算子接受 `Op` 的输入数据和 `Op` 的输出的梯度，计算 ` Op ` 的梯度
>  一些 DL 编译器 (例如 TVM，TC) 支持自动微分，但它们在应用链式法则时，需要高级 IR 中全部算子的导数
>  TVM 正致力于为代数算子和 NN 算子提供导数算子，程序员可以使用这些导数算子为自定义算子构建导数
>  PlaidML 可以自动生成导数算子
>  不能提供导数算子的 DL 编译器不能支持模型训练

4. **Customized operators**: It allows programmers to define their operators for a particular purpose. Providing support for customized operators improves the extensibility of DL compilers. For example, when defining new operators in Glow, the programmers need to realize the logic and node encapsulation. In addition, extra efforts are needed, such as the lowering step, operation IR generation, and instruction generation, if necessary. Whereas, TVM and TC require less programming efforts except describing the computation implementation. Specifically, the users of TVM only need to describe the computation and the schedule and declare the shape of input/output tensors. Moreover, the customized operators integrate Python functions through hooks, which further reduces the programmers’ burden.
>  自定义算子
>  DL 编译器提供自定义算子可以提高其可拓展性
>  Glow 中定义新算子时，程序员需要实现逻辑和节点封装，必要时还需要 lowering step、操作 IR 生成、指令生成
>  TVM 和 TC 除了需要描述计算实现以外，编程工作较少，具体地说，TVM 的用于仅需要描述计算和调度，并声明输入/输出张量的形状，还可以用钩子集成 Python 函数，进一步减少编程负担

### 4.1.3 Discussion
Nearly all DL compilers have their unique high-level IRs. However, they share similar design philosophies, such as using DAG and let-binding to build the computation graph. In addition, they usually provide convenient ways for programmers to represent tensor computation. The data and operators designed in high-level IRs are flexible and extensible enough to support diverse DL models. More importantly, the high-level IRs are hardware-independent and thus can be applied with different hardware backend.
> 几乎所有 DL 编译器都有其唯一的高级 IR，但其设计思想存在类似，例如使用 DAG 和 let-绑定来构建计算图
> 此外，这些 IR 还提供便利的方式便于程序员表示张量计算
> 在高级 IR 中设计的数据和算子足够灵活和可拓展，可以支持各种深度学习模型，注意高级 IR 是独立于硬件的，因此可以应用于不同的硬件后端

## 4.2 Low-level IR
### 4.2.1 Implementation of Low-Level IR
Low-level IR describes the computation of a DL model in a more fine-grained representation than that in high-level IR, which enables the target-dependent optimizations by providing interfaces to tune the computation and memory access. In this section, we classify the common implementations of low-level IRs into three categories: Halide-based IR, polyhedral-based IR, and other unique IR.
>  低级 IR 以更细粒度的表示描述 DL 模型的计算，低级 IR 为我们对真实的计算和内存访问的调节提供了接口，我们进而可以用低级 IR 实现依赖目标的优化
>  本节将低级 IR 的常用实现分类为三类：基于 Halide 的 IR、基于 polyhedral 的 IR、其他独立 IR

**Halide-based IR** - Halide is firstly proposed to parallelize image processing, and it is proven to be extensible and efficient in DL compilers (e.g., TVM).
>  基于 Halide 的 IR
>  Halide 最初被提出用于并行化图像处理，目前已经证明在 DL 编译器 (例如 TVM) 是可拓展且高效的

The fundamental philosophy of Halide is the separation of computation and schedule . Rather than giving a specific scheme directly, the compilers adopting Halide try various possible schedule and choose the best one. The boundaries of memory reference and loop nests in Halide are restricted to bounded boxes aligned to the axes. Thus, Halide cannot express the computation with complicated patterns (e.g., non-rectangular). Fortunately, the computations in DL are quite regular to be expressed perfectly by Halide. Besides, Halide can easily parameterize these boundaries and expose them to the tuning mechanism. The original IR of the Halide needs to be modified when applied to backend of DL compilers. For example, the input shape of Halide is infinite, whereas the DL compilers need to know the exact shape of data in order to map the operator to hardware instructions. Some compilers, such as TC, require the fixed size of data, to ensure better temporal locality for tensor data.  
>  Halide 的基本哲学是分离计算和调度，编译器不会直接给出具体的方案，而是使用 Halide 尝试多个可能的调度，从中选择最优的一个
>  Halide 中的内存引用边界和循环嵌套被限制在和坐标轴对齐的有界框内，因此无法表达具有复杂模式 (如非矩形) 的计算；幸运的是，DL 中的计算非常规整，因此可以用 Halide 完美表达，并且 Halide 可以轻易对这些边界进行参数化，并将其暴露给调优机制
>  当应用于 DL 编译器后端时，Halide 的原始 IR 需要被修改，例如，Halide 的输入形状是无限的，但 DL 编译器需要知道数据的确切形状，以便将运算映射到硬件指令
>  一些编译器 (例如 TC) 要求数据的大小固定，以确保 tensor 数据具有更好的时间局部性

TVM has improved Halide IR into an independent symbolic IR by following efforts. It removes the dependency on LLVM and refactors the structure of both the project module and the IR design of Halide, pursuing better organization as well as accessibility for graph IR and frontend language such as Python. The re-usability is also improved, with a runtime dispatching mechanism implemented to add customized operators conveniently. TVM simplifies the variable definition from string matching to pointer matching, guaranteeing that each variable has a single define location (static single-assignment, SSA) [22]).
>  TVM 将 Halide IR 改进为了独立的符号 IR，它移除了对 LLVM 的依赖，并重构了 Halide 的项目模块结构和 IR 设计
>  TVM 通过运行时分派机制，方便了添加自定义的算子，提高了可重用性
>  TVM 简化了变量定义，从字符串匹配改为指针匹配，保证每个变量只有一个定义位置 (静态单赋值)

**Polyhedral-based IR** - The polyhedral model is an important technique adopted in DL compilers. It uses linear programming, affine transformations, and other mathematical methods to optimize loop-based codes with static control flow of bounds and branches. In contrast to Halide, the boundaries of memory reference and loop nests can be polyhedrons with any shapes in the polyhedral model. Such flexibility makes polyhedral models widely used in generic compilers. However, such flexibility also prevents the integration with the tuning mechanisms. Nevertheless, due to the ability to deal with deeply nested loops, many DL compilers, such as TC and PlaidML (as the backend of nGraph), have adopted the polyhedral model as their low-level IR. The polyhedral-based IR makes it easy to apply various polyhedral transformations (e.g., fusion, tiling, sinking, and mapping), including both device-dependent and device-independent optimizations. There are many toolchains that are borrowed by polyhedral-based compilers, such as isl [96], Omega [48], PIP [23], Polylib [60], and PPL [9].
>  基于多面体的 IR
>  多面体模型是 DL 编译器中采用的重要技术，它使用线性规划、仿射变换，以及其他数学方法来优化带有静态控制流 (分支、边界) 的循环代码
>  和 Halide 不同，多面体模型中内存引用边界和循环嵌套可以是任意形状的多面体，因此通用编译器中，多面体模型也广泛应用
>  但该灵活性阻碍了和调优机制的集成
>  尽管如此，因为可以处理深度嵌套的循环，许多 DL 编译器例如 TC, PlaidML (nGraph 的后端) 都采用多面体模型作为低级 IR，对这类 IR 进行各种多面体变换 (融合、分块、下沉、映射) 都非常容易，这些变换既包括了设备相关的优化，也包括了设备无关的优化
>  基于多面体的编译器采用了许多工具链，例如 isl, Omega, PIP, Polylib, PPL

TC has its unique design in low-level IR, which combines the Halide and polyhedral model. It uses Halide-based IR to represent the computation and adopts the polyhedral-based IR to represent the loop structures. TC presents detailed expressions through abstract instances and introduces specific node types. In brief, TC uses the domain node to specify the ranges of index variables and uses the context node to describe new iterative variables that are related to hardware. And it uses the band node to determine the order of iterations. A filter node represents an iterator combined with a statement instance. Set and sequence are keywords to specify the execution types (parallel and serial execution) for filters . Besides, TC uses extension nodes to describe other necessary instructions for code generation, such as the memory movement.
>  TC 的低级 IR 有独特的设计，它结合了 Halide 和多面体模型
>  它使用基于 Halide 的 IR 表示计算，并采用基于多面体的 IR 表示循环结构
>  TC 通过引入特定的节点类型和抽象实例来表示详细表达式，简单地说，TC 使用 domain 节点来指定索引变量的类型；使用 context 节点来描述和硬件相关的新迭代变量；使用 band 节点决定迭代顺序；使用 filter 节点表示和语句实例结合的迭代器，为 filter 节点指定执行类型 (并行还是串行) 的关键字为 `set/sequence`；使用 extension 节点为代码生成描述必要的指令，例如内存移动

PlaidML uses polyhedral-based IR (called Stripe) to represent tensor operations. It creates a hierarchy of parallelizable code by extending the nesting of parallel polyhedral blocks to multiple levels. Besides, it allows nested polyhedrons to be allocated to nested memory units, providing a way to match the computation with the memory hierarchy. In Stripe, the hardware configuration is independent of the kernel code. The tags in Stripe (known as passes in other compilers) do not change the kernel structure, but provide additional information about the hardware target for the optimization passes. Stripe splits the DL operators into tiles that fit into local hardware resources.
>  PlaidML 使用基于多面体的 IR (称为 Stripe) 表示张量运算
>  PlaidML 通过将并行多面体块的嵌套层次拓展到多个级别来创建层次化的并行代码，同时 PlaidML 允许嵌套的多面体被分配给嵌套的内存单元，从而提供了一种使计算和内存层次匹配的方式
>  Stripe 中，硬件配置独立于内核代码，Stripe 中的 tags (其他编译器称为 passes) 不会改变内核结构，但会提供关于硬件目标的额外信息以用于优化 passes
>  Stripe 将 DL 算子分为适合于本地硬件资源的 tiles

**Other unique IR** - There are DL compilers implementing customized low-level IRs without using Halide and polyhedral model. Upon the customized low-level IRs, they apply hardware-specific optimizations and lowers to LLVM IR.
>  其他独立 IR
>  有其他 DL 编译器不使用 Halide 和多面体模型，实现了自定义的低级 IR，在自定义的 IR 上，它们应用针对硬件的优化，并且 lower 到 LLVM IR

The low-level IR in Glow is an instruction-based expression that operates on tensors referenced by addresses [79]. There are two kinds of instruction-based functions in Glow low-level IR: declare and program . The first one declares the number of constant memory regions that live throughout the lifetime of the program (e.g., input, weight, bias). The second one is a list of locally allocated regions, including functions (e.g., conv and pool) and temporary variables. Instructions can run on the global memory regions or locally allocated regions. Besides, each operand is annotated with one of the qualifiers: @in indicates the operand reads from the buffer; @out indicates that the operand writes to the buffer; @inout indicates that the operand reads and writes to the buffer. These instructions and operand qualifiers help Glow determine when certain memory optimizations can be performed.  
>  Glow 的低级 IR 是基于指令的表达式，表达式对张量进行操作，张量由地址引用
>  Glow 低级 IR 有两类基于指令的函数: declare 和 program
>  declare 用于声明在程序的整个生命周期中保持存在的常量内存区域 (例如输入、权重、偏置) 的数量
>  program 是一系列局部分配的区域，包括函数 (例如 conv, pool) 和临时变量
>  指令可以运行于全局内存区域和局部分配的内存区域
>  另外，每个操作数都由以下的修饰符之一标记：
>  `@in` 表示操作数需要从缓存中读取
>  `@out` 表示操作数需要写入缓存中
>  `@inout` 表示操作数从缓存中读取，且写入缓存
>  Glow 的这些指令和操作数修饰符帮助 Glow 决定什么时候执行特定的内存优化

MLIR is highly influenced by LLVM, and it is a purer compiler infrastructure than LLVM. MLIR reuses many ideas and interfaces in LLVM, and sits between the model representation and code generation. MLIR has a flexible type system and allows multiple abstraction levels, and it introduces dialects to represent these multiple levels of abstraction. Each dialect consists of a set of defined immutable operations. The current dialects of MLIR include TensorFlow IR, XLA HLO IR, experimental polyhedral IR, LLVM IR, and TensorFlow Lite. The flexible transformations between dialects are also supported. Furthermore, MLIR can create new dialects to connect to a new low-level compiler, which paves the way for hardware developers and compiler researchers.
>  MLIR 受 LLVM 的高度影响，同时它是比 LLVM 更纯粹的编译器 infrastructure
>  MLIR 复用了 LLVM 的许多思想和接口，MLIR 处于模型表示和代码生成之间
>  MLIR 具有灵活的类型系统，同时允许多层抽象层次
>  MLIR 引入了方言来表示不同的抽象级别，每个方言由一组定义好的不可变操作组成
>  MLIR 当前的方言包括了 TensorFlow IR, XLA HLO IR，实验性的多面体 IR，LLVM IR 和 TensorFlow Lite，MLIR 支持这些方言这些的灵活转化
>  MLIR 还可以创建新的方言来连接到低级编译器

The HLO IR of XLA can be considered as both high-level IR and low-level IR because HLO is fine-grained enough to represent the hardware-specific information. Besides, HLO supports hardware-specific optimizations and can be used to emit LLVM IR.
>  XLA 的 HLO IR 可以认为即是高级 IR 也是低级 IR，因为 HLO 足够细粒度，可以表示硬件特定的信息，并且 HLO 也支持针对硬件的优化，并且可以用于生成 LLVM IR

### 4.2.2 Code Generation based on Low-Level IR
The low-level IR adopted by most DL compilers can be eventually lowered to LLVM IR, and benefits from LLVM’s mature optimizer and code generator. Furthermore, LLVM can explicitly design custom instruction sets for specialized accelerators from scratch. However, traditional compilers may generate poor code when passed directly to LLVM IR. In order to avoid this situation, two approaches are applied by DL compilers to achieve hardware-dependent optimization: 1) perform target-specific loop transformation in the upper IR of LLVM (e.g., Halide-based IR and polyhedral-based IR), and 2) provide additional information about the hardware target for the optimization passes. Most DL compilers apply both approaches, but the emphasis is different. In general, the DL compilers that prefer frontend users (e.g., TC, TVM, XLA, and nGraph) might focus on 1) , whereas the DL compilers that are more inclined to backend developers (e.g., Glow, PlaidML, and MLIR) might focus on 2) .
>  大多数 DL 编译器的低级 IR 最终会被 lower 到 LLVM IR，以利用 LLVM 成熟的优化器和代码生成器
>  同时，LLVM 可以显式为特定的加速器从零开始设计自定义指令集
>  DL 编译器采用了两种方法实现依赖硬件的优化
>  1. 在 LLVM 的高层 IR (例如基于 Halide 的 IR 和基于多面体的 IR) 执行针对目标的循环转换
>  2. 为优化 passes 提供关于硬件目标的额外信息
>  大多数 DL 编译器同时采用两种方法，但侧重不同，一般偏向前端用户的 DL 编译器 (例如 TC, TVM, XLA, nGraph) 侧重第一点，偏向后端开发者的 DL 编译器 (Glow, PlaidML, MLIR) 侧重第二点

The compilation scheme in DL compilers can be mainly classified into two categories: justin-time (JIT) and ahead-of-time (AOT). For JIT compilers, it can generate executable codes on the fly, and they can optimize codes with better runtime knowledge. AOT compilers generate all executable binaries first and then execute them. Thus they have a larger scope in static analysis than JIT compilation. In addition, AOT approaches can be applied with cross-compilers of embedded platforms (e.g., C-GOOD [46]) as well as enable execution on remote machines (TVM RPC) and customized accelerators.
>  DL 编译器的编译方案可以分为两类
>  1. 及时 (JIT) 
>  2. 提前 (AOT)
>  JIT 编译器实时生成可执行代码，可以利用更多运行时知识优化代码
>  AOT 编译器提前生成所有可执行二进制文件，然后执行，故它们的静态分析范围更大
>  AOT 方法的优势在于可以用于嵌入式平台的交叉编译器 (如 C-GOOD) 、可以用于远程机器执行 (TVM RPC) 和自定义加速器

### 4.2.3 Discussion
In DL compilers, the low-level IR is a fine-grained representation of DL models, and it reflects detailed implantation of DL models on diverse hardware. The low-level IRs include Halide-based IRs, polyhedral-based IRs, and other unique IRs. Although they differ in designs, they leverage the mature compiler tool-chains and infrastructure, to provide tailored interfaces of hardware-specific optimizations and code generation. The design of low-level IRs can also impact the design of new DL accelerators (e.g., TVM HalideIR and Inferentia, as well as XLA HLO and TPU).
>  DL 编译器的低级 IR 是 DL 模型的细粒度表示，反映了 DL 模型在各种硬件上的详细实现
>  低级 IR 包括了 Halide-based IR, polyhedral-based IR 和其他独立 IR
>  这些 IR 都利用了成熟的编译工具链来提供针对硬件的优化和代码生成的接口
>  低级 IR 的设计也会影响 DL 加速器的设计 (例如 TVM HdlideIR 影响了 Inferentia 的设计，XLA HLO 影响了 TPU 的设计)

## 4.3 Frontend optimizations
After constructing the computation graph, the frontend applies graph-level optimizations. Many optimizations are easier to be identified and performed at graph level because the graph provides a global view of the computation. These optimizations are only applied to the computation graph, rather than the implementations on backends. Thus they are hardware-independent and can be applied to various backend targets.
>  计算图构建后，编译器前端会进行图级别的优化
>  图为计算提供了全局视角，因此许多优化在图级别更易于执行
>  前端的优化仅应用于计算图，和后端实现无关，因此它们也和硬件无关，不同的后端可以共享前端优化

The frontend optimizations are usually defined by passes , and can be applied by traversing the nodes of the computation graph and performing the graph transformations. The frontend provides methods to 1) capture the specific features from the computation graph and 2) rewrite the graph for optimization. Besides the pre-defined passes , the developers can also define customized passes in the frontend. Most DL compilers can determine the shape of both input tensors and output tensors of every operation once a DL model is imported and transformed as a computation graph. This feature allows DL compilers to perform optimizations according to the shape information. 
>  前端优化一般由 passes 定义，passes 被用于遍历计算图节点并执行图转化
>  passes 一般提供了执行以下操作的方法：
>  1. 从计算图中捕获特定特征
>  2. 重写计算图以优化
>  除了预定义的 passes，开发者还可以定义自定义的 passes
>  大多数 DL 编译器可以在 DL 模型被代入并且转化为计算图后就确定其中每个运算的输入输出张量的形状，进而依赖形状信息执行优化

Figure 3 shows an example of computation graph optimizations with Tensorflow XLA.

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/6b1225e22de5652bb312c243366061a6d8bc5b55af2f55fc5c5bdc19575d7980.jpg)  
Fig. 3 Example of computation graph optimizations, taken from the HLO graph of Alexnet on VoltaGPU using Tensorflow XLA

In this section, we classify the frontend optimizations into three categories: 1) node-level optimizations, 2) block-level (peephole, local) optimizations, and 3) dataflow-level (global) optimizations.
>  本节将前端优化分为三类：
>  1. 节点级别优化
>  2. 块级别优化
>  3. 数据流级别 (全局) 优化

### 4.3.1 Node-level optimizations. 
The nodes of the computation graph are coarse enough to enable optimizations inside a single node. And the node-level optimizations include node elimination that eliminates unnecessary nodes and node replacement that replaces nodes with other lower-cost nodes.
>  节点级别的优化包括了节点消除 (消除不必要的节点) 和节点替换 (用更低开销的节点替换当前节点)
>  计算图的节点是较为粗糙的，因此单个节点内部也可以进行优化

In general-purpose compilers, Nop Elimination removes the no-op instructions which occupy a small amount of space but specify no operation. In DL compilers, Nop Elimination is responsible for eliminating the operations lacking adequate inputs. For example, the sum node with only one input tensor can be eliminated, the padding node with zero padding width can be eliminated.
>  通用目的的编译器中，Nop 消除会移除 no-op 指令，该指令会占据少量空间，但不执行任何操作
>  DL 编译器中的 Nop 消除则是消除缺乏足够输入的操作，例如只有单个输入 tensor 的 sum 节点就会被直接消除，padding 宽度为 0 的 padding 节点也会被直接消除

Zero-dim-tensor elimination is responsible for removing the unnecessary operations whose inputs are zero-dimension tensors. Assume that $A$ is a zero-dimension tensor, and $B$ is a constant tensor, then the sum operation node of $A$ and $B$ can be replaced with the already existing constant node $B$ without affecting the correctness. Assume that $C$ is a 3-dimension tensor, but the shape of one dimension is zero, such as ${\{0,2,3\}}$ , therefore, $C$ has no element, and the argmin/argmax operation node can be eliminated.
>  零维度 tensor 消除会移除输入是零维 tensor 的不必要操作
>  例如，假设 A 为零维 tensor，B 为常数 tensor，则 A 和 B 的 sum 运算节点会被直接替换为现存的常量节点 B
>  例如，假设 C 为三维 tensor，但其 shape 的一个维度是 0，例如 C 的 shape 为 `{0,2,3}` ，这表示 C 根本没有元素，故对应的节点可以被消除

### 4.3.2 Block-level optimizations. 
**Algebraic simplification** - The algebraic simplification optimizations consist of 1) algebraic identification, 2) strength reduction, with which we can replace more expensive operators by cheaper ones; 3) constant folding, with which we can replace the constant expressions by their values. Such optimizations consider a sequence of nodes, then take advantage of commutativity, associativity, and distributivity of different kinds of nodes to simplify the computation.  
>  代数简化
>  代数简化优化包括了 
>  1. 代数识别 
>  2. 强度缩减，即用成本更低的算子替换成本更高的算子 
>  3. 常量折叠，即将常量表达式替换为它们的值
>  代数简化优化会考虑一个节点序列，然后利用不同类型节点的交换律、结合律、分配律来简化计算

In addition to the typical operators $(+,\times,\mathrm{etc.})$ , the algebraic simplification can also be applied to DL specific operators (e.g., reshape , transpose , and pooling ). The operators can be reordered and sometimes eliminated, which reduces redundancy and improves the efficiency. Here we illustrate the common cases where algebraic simplification can be applied: 1) optimization of computation order, in such case, the optimization finds and removes reshape/transpose operations according to specific characteristics. Taking the matrix multiplication (GEMM) for example, there are two matrices (e.g., $A$ and $B$ ), both matrices are transposed (to produce $A^{T}$ and $B^{T}$ , respectively), then $A^{T}$ and $B^{T}$ are multiplied together. However, a more efficient way to implement GEMM is to switch the order of the arguments $A$ and $B$ , multiply them together, and then transpose the output of the GEMM, which reduces two transpose to just one; 2) optimization of node combination, in such case, the optimization combines multiple consecutive transpose nodes into a single node, eliminates identity transpose nodes, and optimizes transpose nodes into reshape nodes when they actually move no data; 3) optimization of ReduceMean nodes, in such case, the optimization performs substitutions of ReduceMean with AvgPool node (e.g., in Glow), if the input of the reduce operator is 4D with the last two dimensions to be reduced.
>  除了典型的算子 (例如 $+,\times$)，代数简化还可以被应用于特定的 DL 算子 (例如 reshape, transpose, pooling)
>  通过代数简化，算子会被重新排序甚至消除，以减少冗余，提高效率
>  代数简化经常应用的场景有：
>  1. 优化计算顺序，此时代数简化优化会根据特定的特点，找到并移除 reshape/transpose 运算；以 GEMM 为例，对于计算 $A^TB^T$，朴素方法是分别转置 $A, B$，然后进行乘法，但优化的方法是先计算 $BA$，然后对结果转置，这便消去了一次转置
>  2. 优化节点组合，此时代数简化优化会将多个连续的转置节点结合为单个节点，同时消除恒等转置节点，并且如果转置节点没有移动数据，则会将转置节点优化为 reshape 节点
>  3. 优化 ReduceMean 节点，如果 reduce 算子的输入 tensor 为 4D，且要规约的维度是最后两维度，则此时代数简化优化会将 ReduceMean 节点替换为 AvgPool 节点

**Operator fusion** - Operator fusion is indispensable optimization of DL compilers. It enables better sharing of computation, eliminates intermediate allocations, facilitates further optimization by combining loop nests [78], as well as reduces launch and synchronization overhead [91]. In TVM, the operators are classified into four categories: injective, reduction, complex-out-fusible, and opaque. When the operators are defined, their corresponding categories are determined. Targeting the above categories, TVM designs the fusion rules across operators. In TC, fusion is performed differently based on the automatic polyhedron transformations. However, how to identify and fuse more complicated graph patterns, such as blocks with multiple broadcast and reduce nodes, remains to be a problem. Recent works [61 , 62] try to tackle this problem and propose a framework to explore and optimize aggressive fusion plans. It supports not only element-wise and reduction nodes, but also other computation/memory intensive nodes with complex dependencies.
>  算子融合
>  算子融合提高计算共享，消除中间的内存分配，结合循环嵌套，减少发起和同步开销
>  TVM 将算子分为了四类：内射型、归约型、复杂输出可融合型、不透明型，当定义算子时，它们对应的类别也需要确定
>  TVM 针对以上四类算子设计了跨算子的融合规则
>  TC 中则基于自动多面体变换执行融合，但如何识别和融合更复杂的图模式，例如包含多个广播和归约节点的块，仍然是一个问题
>  近期的工作提出一个框架来探索了优化激进的融合规划，该框架不仅支持逐元素计算节点和归约节点，还支持具有复杂依赖关系的计算/内存密集型节点

**Operator sinking** - This optimization sinks the operations such as transposes below operations such as batch normalization, ReLU, sigmoid, and channel shuffle. By this optimization, many similar operations are moved closer to each other, creating more opportunities for algebraic simplification.
>  算子下沉
>  算子下沉优化将类似转置的算子下沉到例如 batch normalization, ReLU, sigmoid, channel shuffle 等算子之下，这使得许多相似的操作更靠近彼此，从而为代数简化创造了更多机会

### 4.3.3 Dataflow-level optimizations
**Common sub-expression elimination (CSE)** - An expression $E$ is a common sub-expression if the value of $E$ is previously computed, and the value of $E$ has not to be changed since previous computation [6]. In this case, the value of $E$ is computed once, and the already computed value of $E$ can be used to avoid re computing in other places. The DL compilers search for common sub-expressions through the whole computation graph and replace the following common sub-expressions with the previously computed results.
>  公共子表达式消除
>  对于一个表达式 $E$，如果它的值已经被计算过，并且自上次计算以来其值没有发生变化，那么 $E$ 就是一个公共子表达式
>  在这种情况下，表达式 $E$ 只需计算一次，然后在其他地方使用已经计算好的值
>  DL 编译器在整个计算图中搜索公共子表达式，然后将后续出现的公共子表达式替换为之前计算的结果

**Dead code elimination (DCE)** - A set of code is dead if its computed results or side-effects are not used. And the DCE optimization removes the dead code. The dead code is usually not caused by programmers but is caused by other graph optimizations. Thus, the DCE, as well as CSE, are applied after other graph optimizations. Other optimizations, such as dead store elimination (DSE), which removes stores into tensors that are never going to be used, also belong to DCE.
>  死代码消除
>  对于一组代码，如果其计算结果或副作用没有被使用，则它就是死代码
>  死代码消除会移除死代码，死代码往往不是由程序员直接导致，而是由其他图优化导致
>  因此，死代码消除和公共子表达式消除一般在图优化之后应用，其他优化，例如死存储消除 (移除从未使用过的对张量进行存储的操作) 也属于死代码消除

**Static memory planning** - Static memory planning optimizations are performed to reuse the memory buffers as much as possible. Usually, there are two approaches: in-place memory sharing and standard memory sharing. The in-place memory sharing uses the same memory for input and output for an operation, and just allocates one copy of memory before computing. Standard memory sharing reuses the memory of previous operations without overlapping. The static memory planning is done offline, which allows more complicated planning algorithms to be applied. A recent work [4] firstly designs and performs memory-aware scheduling to minimize the peak activation memory footprint on edge devices, which presents new research directions of memory planning on memory-constrained devices.  
>  静态内存规划
>  静态内存规划旨在尽可能重用内存缓冲区，一般有两种方法：就地内存共享和标准内存共享
>  就地内存共享为一个算子的输入和输出使用同一块内存，在计算之前就仅分配一块内存
>  标准内存共享在不重叠的情况下复用之前算子的内存
>  静态内存规划是离线完成的，这允许我们应用更复杂的规划算法
>  工作 [4] 设计和实现了内存感知的调度来最小化边缘设备的峰值激活内存占用，为内存受限设备上的内存规划开辟了新的研究方向

**Layout transformation** - Layout transformation tries to find the best data layouts to store tensors in the computation graph and then inserts the layout transformation nodes to the graph. Note that the actual transformation is not performed here, instead, it will be performed when evaluating the computation graph by the compiler backend.
>  布局转换尝试找到存储计算图中的 tensors 的最优数据布局，然后将布局转换节点插入到图中
>  但实际的布局转换不会在该节点执行 (即不会在前端执行)，而是由编译器后端评估计算图时执行

In fact, the performance of the same operation in different data layouts is different, and the best layouts are also different on different hardware. For example, operations in the NCHW format on GPU usually run faster, so it is efficient to transform to NCHW format on GPU (e.g., TensorFlow). Some DL compilers rely on hardware-specific libraries to achieve higher performance, and the libraries may require certain layouts. Besides, some DL accelerators prefer more complicated layouts (e.g., tile). In addition, edge devices usually equip heterogenous computing units, and different units may require different data layouts for better utilization, thus layout transformation needs careful considerations. Therefore, the compilers need to provide a way to perform layout transformations across various hardware.
>  相同的算子在不同的数据布局上的性能是不同的，不同硬件上的最优布局同样不同；例如，按照 NCHW 格式的算子在 GPU 上一般运行更快，因此对于 GPU 运算可以考虑将布局转化为 NCHW 格式
>  一些 DL 编译器依赖针对硬件的库达到更高表现，这些针对硬件的库一般就需要特定的数据布局；另外，一些 DL 加速设备偏好更复杂的布局 (例如 tile)
>  边缘设备一般具备异构的计算单元，而不同的单元可能需要不同的数据布局
>  因此 DL 编译器需要为多类硬件提供布局转换

Not only the data layouts of tensors have a nontrivial influence on the final performance, but also the transformation operations have a significant overhead. Because they also consume the memory and computation resource.
>  注意布局转换操作也存在较大的开销，因为它们也需要消耗内存和计算资源

A recent work [58] based on TVM targeting on CPUs alters the layout of all convolution operations to $\mathrm{NCHW}[x]\mathrm{c}$ first in the computation graph, in which c means the split sub-dimension of channel C and $x$ indicates the split size of the sub-dimension. Then all $x$ parameters are globally explored by auto-tuning when providing hardware details, such as cache line size, vectorization unit size, and memory access pattern, during hardware-specific optimizations.
>  [58] 基于 TVM，为 CPUs 将计算图中的所有卷积算子布局转化为 NCHW\[x\]c，其中 c 表示通道 C 的子维度划分，x 表示子维度的大小
>  在提供硬件详细信息，包括缓存行大小、向量单元大小、内存访问模式，的情况下，所有的 x 参数会在针对硬件的优化中通过自动调优进行全局探索

### 4.3.4 Discussion
The frontend is one of the most important components in DL compilers, which is responsible for transformation from DL models to high-level IR (e.g., computation graph) and hardware-independent optimizations based on high-level IR. Although the implementation of frontend may differ in the data representation and operator definition of high-level IR across DL compilers, the hardware-independent optimizations converge at three levels: node-level, block-level, and dataflow-level. The optimization methods at each level leverage the DL specific as well as general compilation optimization techniques, which reduce the computation redundancy as well as improve the performance of DL models at the computation graph level.
>  DL 编译器的前端负责将 DL 模型转化为高级 IR (例如计算图)，同时基于高级 IR 执行独立于硬件的优化
>  不同 DL 编译器的高级 IR 在数据表示和算子定义存在不同，但独立于硬件的优化都收敛到三个层次：节点级别、块级别、数据流级别
>  其中每个级别的优化方法都同时利用了针对 DL 的和通用的优化技巧，目标是在计算图界别减少计算冗余的同时提高 DL 模型的性能

## 4.4 Backend optimizations
The backends of DL compilers have commonly included various hardware-specific optimizations, auto-tuning techniques, and optimized kernel libraries. Hardware-specific optimizations enable efficient code generation for different hardware targets. Whereas, auto-tuning has been essential in the compiler backend to alleviate the manual efforts to derive the optimal parameter configurations. Besides, highly-optimized kernel libraries are also widely used on general-purpose processors and other customized DL accelerators.
>  DL 编译器的后端一般包含各种针对硬件的优化、自动调优技术、优化的 kernel 库

### 4.4.1 Hardware-specific Optimization
Hardware-specific optimizations, also known as target dependent optimizations, are applied to obtain high-performance codes targeting specific hardware. One way to apply the backend optimizations is to transform the low-level IR into LLVM IR, to utilize the LLVM infrastructure to generate optimized CPU/GPU codes. The other way is to design customized optimizations with DL domain knowledge, leveraging the target hardware more efficiently. Since hardware-specific optimizations are tailored for particular hardware and cannot be included exhaustively in this paper, we present five widely adopted approaches in existing DL compilers. The overview of these hardware-specific optimizations is shown in Figure 4, and the detailed descriptions are provided as follows.
>  针对硬件的优化，也称为依赖于目标的优化
>  一种应用后端优化的方式是将低级 IR 转化为 LLVM IR，利用 LLVM infrastructure 生成优化的 CPU/GPU 代码
>  另一种方式是根据 DL 领域知识设计自定义的优化
>  本节展示现存 DL 编译器广泛采用的 5 种方法

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/cd64707c7eca0d6b6e73b545163ce3e10ac62f041edb7b4bf0f05f2abfca95e1.jpg)  
Fig. 4. Overview of hardware-specific optimizations applied in DL compilers.

**Hardware intrinsic mapping** - Hardware intrinsic mapping can transform a certain set of low-level IR instructions to kernels that have already been highly optimized on the hardware. In TVM, the hardware intrinsic mapping is realized in the method of extensible tensorization , which can declare the behavior of hardware intrinsic and the lowering rule for intrinsic mapping. This method enables the compiler backend to apply hardware implementations as well as highly optimized handcraft micro-kernels to a specific pattern of operations, which results in a significant performance gain. Whereas, Glow supports hardware intrinsic mapping such as quantization . It can estimate the possible numeric range for each stage of the neural network and support profile guided optimization to perform quantization automatically. Besides, Halide/TVM maps specific IR patterns to SIMD opcodes on each architecture to avoid the inefficiency of LLVM IR mapping when encountering vector patterns.
>  硬件 intrinsic 映射
>  硬件 intrinsic 映射将特定的一组低级 IR 指令转化为已经在目标硬件上高度优化过的 kernel
>  TVM 通过可拓展的张量化方法实现硬件 intrinsic 映射，可拓展的张量化方法支持声明硬件 intrinsic 的行为，以及 intrinsic 映射的降级规则，便于编译器后端将高度优化的手写微 kernel 映射到特定的算子模式
>  Glow 也支持硬件 intrinsic 映射，例如量化，它为 NN 的每个阶段估计数值泛化，并支持基于配置文件的优化自动执行量化
>  Halide/TVM 将特定的 IR 模式直接映射到每个架构上的 SIMD opcodes，避免 LLVM IR 映射在处理向量模式的低效性

**Memory allocation and fetching** - Memory allocation is another challenge in code generation, especially for GPUs and customized accelerators. For example, GPU contains primarily shared memory space (lower access latency with limited memory size) and local memory space (higher access latency with large capacity). Such memory hierarchy requires efficient memory allocation and fetching techniques for improving data locality. To realize this optimization, TVM introduces the scheduling concept of memory scope . Memory scope schedule primitives can tag a compute stage as shared or thread-local . For compute stages tagged as shared , TVM generates code with shared memory allocation as well as cooperative data fetching, which inserts memory barrier at the proper code position to guarantee correctness. Besides, TC also provides similar features (known as memory promotion) by extending PPCG [97] compiler. However, TC only supports limited predefined rules. Particularly, TVM enables special buffering in accelerators through memory scope schedule primitives.
>  内存分配和获取
>  内存分配和获取技术可以用于提高数据局部性
>  TVM 引入了内存作用域的调度概念，内存作用域调度原语将计算阶段标记为共享的或线程局部的，TVM 为标记为共享的计算阶段生成带有 shared memory 分配和协作式数据获取的代码，以及在适当代码位置插入 memory barrier 确保正确性
>  TC 通过拓展 PPCG 编译器提供了类似特性 (称为内存提升)，但仅支持有限的预定义的规则
>  TVM 还可以通过内存作用域调度原语启用加速设备的特殊缓存技术

**Memory latency hiding** - Memory latency hiding is also an important technique used in the backend by reordering the execution pipeline. As most DL compilers support parallelization on CPU and GPU, memory latency hiding can be naturally achieved by hardware (e.g., warp context switching on GPU). But for TPU-like accelerators with decoupled access-execute (DAE) architecture, the backend needs to perform scheduling and fine-grained synchronization to obtain correct and efficient codes. To achieve better performance as well as reduce programming burden, TVM introduces virtual threading schedule primitive, which enables users to specify the data parallelism on virtualized multi-thread architecture. Then TVM lowers these virtually parallelized threads by inserting necessary memory barriers and interleaves the operations from these threads into a single instruction stream, which forms a better execution pipeline of each thread to hide the memory access latency.  
>  内存延迟隐藏
>  DL 编译器后端通过重新排序执行流水线来进行内存延迟隐藏
>  因为大多数 DL 编译器都支持在 CPU 和 GPU 上的并行化，故内存延迟隐藏可以由硬件自然实现 (例如 GPU 上的 warp 上下文切换)
>  对于类似 TPU，具有解耦的访问-执行架构的加速设备，DL 编译器后端需要执行调度和细粒度同步以确保生成的代码高效且正确
>  TVM 引入了虚拟线程调度原语，使用于可以在虚拟化的多线程架构上指定数据并行，TVM 通过插入必要的 memory barrier 以及将这些线程的运算交错为单个指令流来将这些虚拟化的并行线程降级，这为每个线程构成了更好的执行管道，以隐藏内存延迟

**Loop oriented optimizations** - Loop oriented optimizations are also applied in the backend to generate efficient codes for target hardware. Since Halide and LLVM [51] (integrated with the polyhedral method) have already incorporated such optimization techniques, some DL compilers leverage Halide and LLVM in their backends. The key techniques applied in loop oriented optimizations include loop fusion, sliding windows, tiling, loop reordering, and loop unrolling.
>  面向循环的优化
>  Halide 和 LLVM (和多面体方法集成) 已经包含了该优化技术，可以直接被后端利用
>  面向循环优化的关键技术包括了循环融合、滑动窗口、tilling、循环重排序、循环展开

1. **Loop fusion**: Loop fusion is a loop optimization technique that can fuse loops with the same boundaries for better data reuse. For compilers such as PlaidML, TVM, TC, and XLA, such optimization is performed by the Halide schedule or polyhedral approach, while Glow applies loop fusion by its operator stacking .
>  循环融合将相同边界的循环融合，提高数据重用率
>  PlaidML, TVM, TC, XLA 通过 Halide 调度或多面体方法执行这类优化
>  Glow 通过其算子堆叠来实现循环融合

2. **Sliding windows**: Sliding windows is a loop optimization technique adopted by Halide. Its central concept is to compute values when needed and store them on the fly for data reuse until they are no longer required. As sliding windows interleaves the computation of two loops and make them serial, it is a tradeoff between parallelism and data reuse.
>  滑动窗口是 Halide 采用的循环优化技术
>  其核心概念是在需要时计算值，并存储这些值用于数据重用，直到不再需要这些值
>  但滑动窗口交错了两个循环的计算，使其顺序执行，因此滑动窗口方法是并行性和数据重用性之间的 tradeoff

3. **Tiling**: Tiling splits loops into several tiles, and thus loops are divided into outer loops iterating through tiles and inner loops iterating inside a tile. This transformation enables better data locality inside a tile by fitting a tile into hardware caches. As the size of a tile is hardware-specific, many DL compilers determine the tiling pattern and size by auto-tuning.
>  Tiling 将循环划分为多个 tiles，此时循环被划分为外层循环和内存循环
>  外层循环在 tiles 之间迭代，内层循环在 tile 内迭代
>  tiling 通过将一个 tile 放入硬件缓存中提高了 tile 内的数据局部性
>  tile 的大小则针对于硬件，许多 DL 编译器通过自动调优决定 tiling 模式和大小

4. **Loop reordering**: Loop reordering (also known as loop permutation) changes the order of iterations in a nested loop, which can optimize the memory access and thus increase the spatial locality. It is specific to data layout and hardware features. However, it is not safe to perform loop reordering when there are dependencies along the iteration order.
>  循环重排序也称为循环置换，它改变嵌套循环中的迭代顺序，以优化内存访问并提高空间局部性
>  循环重排序取决于数据布局和硬件特性
>  注意在迭代之间存在顺序依赖时，循环重排序是不安全的

5. **Loop unrolling**: Loop unrolling can unroll a specific loop to a fixed number of copies of loop bodies, which allows the compilers to apply aggressive instruction-level parallelism. Usually, loop unrolling is applied in combination with loop split, which first splits the loop into two nested loops and then unrolls the inner loop completely.
>  循环展开讲特定的循环展开为固定数量的循环体拷贝，以允许编译器应用激进的指令级别并行
>  循环展开一般和循环划分一起使用，即首先将循环划分为两个嵌套循环，然后再完全展开内层循环

**Parallelization** - As modern processors generally support multi-threading and SIMD parallelism, the compiler backend needs to exploit parallelism to maximize hardware utilization for high performance. Halide uses a schedule primitive called parallel to specify the parallelized dimension of the loop for thread-level parallelization and supports GPU parallelization by mapping loop dimensions tagged as parallel with annotation of block and thread . And it replaces a loop of size $n$ with a n-wide vector statement, which can be mapped to hardware-specific SIMD opcodes through hardware intrinsic mapping.
>  并行化
>  现代处理器一般都支持多线程和 SIMD 并行，编译器后端需要利用该并行化特性来最大化硬件利用率
>  Halide 使用称为 `parallel` 的调度原语为线程级别并行化的循环指定并行维度，`parallel` 还可以额外添加注解 ` block ` 或 ` thread ` 以支持 GPU 并行化
>  Halide 会将大小为 n 的循环替换为宽度为 n 的向量化语句，该语句可以通过硬件 intrinsic 映射被映射到特定硬件的 SIMD opcodes

Stripe develops a variant of the polyhedral model called nested polyhedral model , which introduces parallel polyhedral block as its basic execution element of iteration. After this extension, a nested polyhedral model can detect hierarchy parallelization among levels of tiling and striding. 
>  Stripe 开发了多面体模型的一个变体，称为嵌套多面体模型，该模型引入了并行多面体块作为迭代的基本执行元素；嵌套多面体模型可以在分块 (tiling) 和步长 (striding) 的不同层级中检测层次并行性

In addition, some DL compilers rely on handcraft libraries such as Glow or optimized math libraries provided by hardware vendors (discussed in Section 4.4.3). In the meanwhile, Glow offloads the vectorization to LLVM because the LLVM auto-vectorizer works well when the information of tensor dimension and loop trip count is provided. However, exploiting the parallelism entirely by compiler backend allows to apply more domain-specific knowledge of DL models, and thus leads to higher performance at the expense of more engineering efforts.  
>  另外，一些 DL 编译器依赖于人工编写的库，例如 Glow，或者硬件供应商提供的优化的数学库
>  Glow 还将向量化的工作交由 LLVM，因为 LLVM 的自动向量化器在提供了 tensor 维度和循环次数信息时，效果良好

### 4.4.2 Auto-tuning
Due to the enormous search space for parameter tuning in hardware-specific optimizations, it is necessary to leverage auto-tuning to determine the optimal parameter configurations. Among the studied DL compilers in this survey, TVM, TC, and XLA support the auto-tuning. Generally, the auto-tuning implementation includes four key components, such as parameterization, cost model, searching technique, and acceleration. .
>  针对硬件的优化中，自动调优用于确定最优的参数配置
>  TVM, TC, XLA 都支持自动调优
>  自动调优的实现包括四个关键组件：参数化、成本模型、搜索技术、加速方法

**Parameterization** - 1) Data and target: The data parameter describes the specification of the data, such as input shapes. The target parameter describes hardware-specific characteristics and constraints to be considered during optimization scheduling and code generation. For example, for the GPU target, the hardware parameters such as shared memory and register size need to be specified. 2) Optimization options : The optimization options include the optimization scheduling and corresponding parameters, such as loop oriented optimizations and tile size. In TVM, both pre-defined and user-defined scheduling, as well as parameters, are taken into consideration. Whereas, TC and XLA prefer to parameterize the optimizations, which have a strong correlation with performance and can be changed later at a low cost. For example, the minibatch dimension is one of the parameters that is usually mapped to grid dimensions in CUDA and can be optimized during auto-tuning.
>  参数化
>  1. 数据和目标：数据参数描述了数据的规格，例如 input shape；目标参数描述了硬件特定的特点和约束，在优化调度和代码生成需要考虑到这些，例如，对于 GPU 目标，就需要指定它的 shared memory size 和 register size
>  2. 优化选项：优化选项包括了优化调度和相关参数，例如面向循环的优化和 tilde size；TVM 会同时考虑预定义的和用户定义的调度，以及参数，TC 和 XLA 则倾向于参数化和性能有强相关且可以在之后低成本更改的优化，例如 minibatch 维度是通常被映射到 CUDA 的 grid dimension 的参数之一，该参数可以在自动调优中被参数化

**Cost model** - The comparison of different cost models applied in auto-tuning are as follows. 1) Black-box model : This model only considers the final execution time rather than the characteristics of the compilation task. It is easy to build a black-box model, but easily ends up with higher overhead and less optimal solution without the guidance of task characteristics. TC adopts this model. 2) ML-based cost model : ML-based cost model is a statistical approach to predict performance using a machine learning method. It enables the model to update as the new configuration is explored, which helps achieve higher prediction accuracy. TVM and XLA adopt this kind of model, for example, gradient tree boosting model (GBDT) and feed forward neural network [47] (FNN) respectively. 3) Pre-defined cost model : An approach based on a pre-defined cost model expects a perfect model built on the characteristics of the compilation task and able to evaluate the overall performance of the task. Compared to the ML-based model, the pre-defined model generates less computation overhead when applied, but requires large engineering efforts for re-building the model on each new DL model and hardware.
>  成本模型
>  我们比较不同成本模型在自动调优中应用的情况
>  1. 黑盒模型：该模型仅考虑最终的执行时间，不考虑编译任务的特点；黑盒模型易于构建，但在缺乏任务特性指导的情况下，容易得到较高的开销和次优的解，TC 采用黑盒模型
>  2. 基于 ML 的成本模型：基于 ML 的成本模型是使用机器学习方法预测性能的统计方法，基于机器学习的开销模型可以随着新配置的探索而更新，帮助达到更高的预测准确性，TVM 采用 GBDT，XLA 采用 FFN
>  3. 预定义的成本模型：基于预定义成本模型的方法期望有一个根据编译任务特性构建的完美模型，能够评估任务的整体性能，预定义的模型相较于基于 ML 的模型的计算开销更少，但需要针对每个新的 DL 模型和硬件重新构建模型

**Searching technique** - 1) Initialization and searching space determination : The initial option can either be set randomly or based on the known configurations, such as configurations given by users or historical optimal configurations. In terms of searching space, it should be specified before auto-tuning. TVM allows developers to specify the searching space with their domain-specific knowledge and provides automatic search space extraction for each hardware target based on the computational description. In contrast, TC relies on the compilation cache and the pre-defined rules. 2) Genetic algorithm (GA) [28]: GA considers each tuning parameter as genes and each configuration as a candidate. The new candidate is iteratively generated by crossover, mutation, and selection according to the fitness value, which is a meta heuristic inspired by the process of natural selection. And finally, the optimal candidate is derived. The rate of crossover, mutation, and selection is used for controlling the tradeoff between exploration and exploitation. TC adopts GA in its auto-tuning technique. 3) Simulated annealing algorithm (SA) [12]: SA is also a meta heuristic inspired by annealing. It allows us to accept worse solutions in a decreasing probability, which can find the approximate global optimum and avoid the precise local optimum in a fixed amount of iterations. TVM adopts SA in its auto-tuning technique. 4) Reinforcement learning (RL) : RL performs with learning to maximize reward given an environment by the tradeoff between exploration and exploitation. Chameleon [5] (built upon TVM) adopts RL in its auto-tuning technique.
>  搜索技巧
>  1. 初始化和搜索空间确定：初始化可以随机，也可以基于用户配置或历史最优配置确定；TVM 允许开发者利用其领域知识指定搜索空间，并为每个硬件目标提供了基于计算描述的自动搜索空间提取；TC 则依赖于编译缓存和预定义的规则
>  2. 遗传算法：遗传算法将每个调优参数视作基因，每个配置视作候选方案，新的候选方案通过交叉、变异和选择迭代式生成，交叉、变异和选择操作根据适应度值确定，遗传算法的目标是得到最优候选；交叉、变异、选择率用于控制探索和利用之间的 tradeoff；TC 使用 GA 作为其自动调优技巧
>  3. 模拟退火算法：模拟退火也是元启发式方法，它允许我们以递减的概率接受更差的解决方案，从而在固定的迭代次数内找到近似全局最优并且避免精确的局部最优解；TVM 使用 SA 作为其自动调优技巧
>  4. RL：RL 通过在探索和利用之间的权衡来最大化给定环境下的奖励；Chameleon 采用 RL 作为其自动调优技巧

**Acceleration** - 1) Parallelization : One direction for accelerating auto-tuning is parallelization. TC proposes a multi-thread, multi-GPU strategy considering that the genetic algorithm needs to evaluate all candidates in each generation. First, it enqueues candidate configurations and compiles them on multiple CPU threads. The generated code is evaluated on GPUs in parallel, and each candidate owns its fitness used by the parent choosing step. After finishing the whole evaluation, the new candidate is generated, and the new compilation job is enqueued, waiting for compiling on CPU. Similarly, TVM supports cross-compilation and RPC, allowing users to compile on the local machine and run the programs with different auto-tuning configurations on multiple targets. 2) Configuration reuse : Another direction for accelerating auto-tuning is to reuse the previous auto-tuning configurations. TC stores the fastest known generated code version corresponding to the given configuration by compilation cache. The cache is queried before each kernel optimization during the compilation, and the auto-tuning is triggered if cache miss. Similarly, TVM produces a log file that stores the optimal configurations for all scheduling operators and queries the log file for best configurations during compilation. It is worth mentioning that TVM performs auto-tuning for each operator in Halide IR (e.g., conv2d), and thus the optimal configurations are determined for each operator separately. 
>  加速
>  1. 并行化：加速自动调优的一个方向是并行化
>  TC 考虑到遗传算法在每一代中需要评估所有的候选，提出了多线程、多 GPU 策略，它首先将候选配置入队，然后在多个 CPU 线程上编译它们，编译生成的代码在 GPUs 上并行执行，每个候选都有其适应度，用于父代选择步骤
>  完成所有执行后，新的候选被生成，新的编译任务入队，等待在 CPU 上编译
>  TVM 支持交叉编译和 RPC，允许用户在本地编译，并且使用多个自动调优配置在多个目标上运行程序
>  2. 配置重用：重用之间的自动调优配置
>  TC 将对应给定配置的已知最快的代码版本储存在编译缓存中，在编译时，每次 kernel 优化前该缓存会被查询，如果缓存为命中，自定调优就会被启动
>  TVM 会生成存储了所有调度算子的最优配置的日志文件，并且在编译时会查询该文件获得最优配置，注意 TVM 为 Halide IR 中每个算子 (例如 conv2d) 执行自动调优，因此最优配置是逐算子分别确定的

### 4.4.3 Optimized Kernel Libraries
There are several highly-optimized kernel libraries widely used to accelerate DL training and inference on various hardware. DNNL (previously MKL-DNN) from Intel, cuDNN from NVIDIA, and MIOpen from AMD are widely used libraries. Both computation intensive primitives (e.g., convolution, GEMM, and RNN) and memory bandwidth limited primitives (e.g., batch normalization, pooling, and shuffle) are highly optimized according to the hardware features (e.g., AVX-512 ISA, tensor cores). And customizable data layouts are supported to make it easy to integrate into DL applications and avoid frequent data layout transformations. Besides, low-precision training and inference, including FP32, FP16, INT8, and non-IEEE floating-point format bfloat16 [45] are also supported. Other customized DL accelerators also maintain their specific kernel libraries [43, 57].
>  DNNL (previously MKL-DNN) from Intel, cuDNN from NVIDIA, MIOpen from AMD 是广泛使用的高度优化的 kernel 库
>  这些库根据硬件特性 (例如 AVX-512 ISA, tensor cores) 高度优化了计算密集的原语 (例如卷积, GEMM, RNN) 和内存带宽限制的原语 (例如 batch normalization, pooling, shuffle) 
>  它们还支持可自定义的数据布局，以便集成到 DL 应用中，并且避免频繁的数据布局转换
>  另外，这些库还支持低精度训练和推理，包括 FP32, FP16, INT8 和非 IEEE 浮点格式的 bfloat16

Existing DL compilers, such as TVM, nGraph, and TC, can generate the function calls to these libraries during code generation. However, if DL compilers need to leverage the existing optimized kernel libraries, they should first transform the data layouts and fusion styles into the types that are pre-defined in kernel libraries. Such transformation may break the optimal control flow. Moreover, the DL compilers treat the kernel libraries as a black box. Therefore they are unable to apply optimizations across operators (e.g., operator fusion) when invoking kernel libraries. In sum, using optimized kernel libraries achieves significant performance improvement when the computation can be satisfied by specific highly-optimized primitives, otherwise it may be constrained from further optimization and suffer from less optimal performance.
>  现存的 DL 编译器，例如 TVM, nGraph, TC，在代码生成过程中可以生成调用这些库的函数
>  但如果 DL 编译器需要利用现存的优化 kernel 库，它们需要首先将数据布局和融合方式转化为 kernel 库中预定义的类型，这种转换可能会破坏最优的控制流
>  另外，DL 编译器将 kernel 库视作黑盒，因此 DL 编译器无法在调用这些库时执行跨算子的优化 (例如算子融合)
>  因此，如果计算可以被特定的优化原语满足，使用 kernel 库可以提高性能，否则反而会约束进一步的优化，使得性能次优

### 4.4.4 Discussion
The backend is responsible for bare-metal optimizations and code generation based on low-level IR. Although the design of backends may differ due to various low-level IRs, their optimizations can be classified into hardware-specific optimizations: auto-tuning techniques, and optimized kernel libraries. These optimizations can be performed separately or combined, to achieve better data locality and parallelization by exploiting the hardware/software characteristics. Eventually, the high-level IR of DL models is transformed into efficient code implementation on different hardware.
>  后端负责裸机的优化和基于低级 IR 的代码生成
>  后端的优化可以分类为：针对硬件的优化, 自动调优和优化的 kernel 库
>  这些优化可以分别执行也可以结合，通过利用硬件/软件特性达到更好的数据局部性和并行性
>  经过后端，DL 模型的高级 IR 会被转化为在不同硬件上的高效代码实现

# 5 Taxonomy of DL Compilers
The DL compilers studied in this survey include TVM, nGraph, Tensor Comprehension (TC), Glow, and XLA. We select these compilers since they are well-known, well maintained, and most importantly, widely used. Thus, we can find enough papers, documents, and discussions from both industry and academia in order to study their designs and implementations in-depth. Table 1 illustrates the taxonomy of the selected DL compilers from four perspectives, including frontend, backend, IR, and optimizations, which corresponds with the key components described in this survey.  
>  本文讨论的 DL 编译器包括了 TVM, nGraph, Tensor Comprehension (TC), Glow, XLA
>  Table 1 从四个方面：前端、后端、IR、优化对这些编译器做了分类

Specifically, we provide more information about the compilers to the best of our knowledge. We not only provide whether a compiler supports a specific feature, but also describe how to use this feature through its programming interface. In addition, we also describe the developing status of specific features and the reasons why specific features are not supported in particular compilers. The target of this taxonomy is to provide guidelines about the selection of DL compilers for the practitioners considering their requirements, as well as to give a thorough summary of the DL compilers for researchers.

In Table 1, we present the features of each DL compiler, including developer, programming language, ONNX/framework support, training support, and quantization support in the frontend category, and we present the compilation methods and supported devices in the backend category. These features are summarized because they strongly affect the usage of DL compilers in particular scenarios. Based on these features, practitioners or researchers can easily decide which DL compiler they would like to work upon.

Table 1, together with Figure 2 can serve as a systematic summary of this survey. Through them, readers can identify the features each compiler supports as well as the key components of each compiler. More detailed information is presented in the following sections.


Table 1. The comparison of DL compilers, including TVM, nGraph, TC, Glow, and XLA.
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/5a9433866fef8241ed039efd6aa8e4fe2cc91ec08c1d965c181e78da7e6c199b.jpg)

# 6 Evaluation
## 6.1 Experimental Setup
Our experiments are conducted on two GPU-equipped machines, and the hardware configuration is shown in Table 2. We evaluate the performance of TVM (v0.6.0), nGraph (0.29.0-rc.0), TC (commit fd01443), Glow (commit 7e68188) and XLA (TensorFlow 2.2.0) on CPU and GPU. We select 19 neural network models in ONNX format as our datasets, which are converted from the Torchvision 2 model zoo and the GluonCV 3 model zoo. These models include full-fledged models: ResNet , DenseNet and VGG series, and lightweight models: MobileNet and MNASNet series. To import the ONNX models, as shown in Table 1, we use the built-in `tvm.relay.frontend.from_onnx` interface of TVM, the `ngraph-onnx` Python package of nGraph, the built-in `ONNXModelLoader` of Glow, and the ` tensorflow-onnx` Python package of XLA. Notably, TC lacks the support of ONNX, so we only evaluate it in the following per-layer performance comparison. Each model is executed for 15 times, and we report the average execution time of the last 10 executions for each compiler, because we regard the first 5 executions as the warm-up to eliminate the overhead of JIT compilation.
>  数据集使用的是 ONNX 格式的 19 个 NN 模型，这些模型来自于 Trochvision 2 model zoo 和 GluonCV 3 model zoo
>  这些模型包括完整的模型: ResNet, DenseNet, VGG 系列，以及轻量的模型: MobileNet, MNASNet 系列
>  导入 ONNX 模型时，TVM 使用的是 `tvm.relay.frontend.from_onnx` 接口，nGraph 使用的是其 `ngraph-onnx` Python 包，Glow 使用的是其内建的 `ONNXModelLoader` ，XLA 使用的是 `tensorflow-onnx` Python 包
>  TC 不支持 ONNX，因此仅在后续的逐层性能比较中评估它
>  每个模型运行 15 次，我们报告后 10 次的平均执行时间，前 5 次视作预热，消除即时编译的开销

Table 2. The hardware configuration.
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/8a059b51d917058750dafec44b444c391c446da3a4b5576d92616221bedb1a52.jpg)


## 6.2 End-to-end Performance Comparison
As shown in Figure 5, we compare the performance of end-to-end inference across TVM, nGraph, Glow, and XLA. We evaluate these compilers on both CPUs (Broadwell and Skylake) and GPUs (V100 and 2080Ti). Note that, we omit the comparison of TC here. Because TC is more similar to a kernel library other than fully functional DL compiler, and it requires the users to implement all layers of a model with its Einstein notion manually, which leads to heavy engineering efforts for a fair comparison. Another reason is that TC only supports running on GPU, thus we cannot obtain its performance results on CPU. However, for detailed comparisons (Figure 6 and 8), we still implement several ResNet and Mobile NetV2 models in TC. In sum, we compare and analyze the performance results from the following perspectives.  
>  在端到端的比较中，我们忽略了 TC，因为 TC 更类似于一个 kernel 库而不是功能完全的 DL 编译器，同时也因为 TC 仅支持在 GPU 上运行
>  我们从以下几个方面比较和分析了 DL 编译器的表现结果


![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/081c4f5b8727d7697fc362e8d98bd6ced11426b569a35a91b25fd06eaacc1ae7.jpg)  
Fig. 5. The performance comparison of end-to-end inference across TVM, nGraph, Glow and XLA on CPU and GPU.

**Compatibility** - Although nGraph and XLA claims to support ONNX , there are still compatibility problems. 1) nGraph fails to run the `DenseNet121` , `VGG16/19` and `MNASNet0_5/1_0` models due to tensors with dynamic shapes. Alternatively, we replace the `DenseNet121` , `VGG16/19` models with the corresponding models from the ONNX model zoo , while `MNASNet0_5/1_0` models are not available. Besides, when we set PlaidML as the backend of nGraph on GPU, we fail to run  all `MobileNet` models. Because PlaidML cannot handle the inconsistent definition of operators across different DL frameworks. 2) XLA can run all selected models, however, the performance is quite low. Thus, we replace the selected ONNX models with the saved models from the Tensorflow Hub , while the `MNASNet0_5/1_0` models are not available. With models from Tensorflow Hub, XLA becomes two orders of magnitude faster, and the performance of XLA becomes competitive with other compilers.
>  兼容性
>  nGraph 和 XLA 对 ONNX 仍然存在兼容性问题：
>  1) 由于存在动态形状的张量，nGraph 无法运行 `DenseNet121`、`VGG16/19` 和 `MNASNet0_5/1_0` 
>  我们用 ONNX model zoo 中的对应模型替换了 `DenseNet121` 和 `VGG16/19` 模型，而 `MNASNet0_5/1_0` 模型不可用
>  此外，当设置 PlaidML 作为 nGraph 在 GPU 上的后端时，我们无法运行所有 `MobileNet` 模型，因为 PlaidML 无法处理不同 DL 框架之间不一致的算子定义
>  2) XLA 可以运行所有选定的模型，但是性能非常低
>  因此，我们将选定的 ONNX 模型替换为来自 TensorFlow Hub 的模型，而 `MNASNet0_5/1_0` 模型不可用，使用来自 TensorFlow Hub 的模型后，XLA 的性能提高了两个数量级，并且其性能与其他编译器相当

**Performance** - From Figure 5, we have several observations about the performance illustrated as follows.

1. **On CPU, the performance of Glow is worse than other compilers .** This is because Glow does not support thread parallelism. Thus it cannot fully utilize the multi-core CPU. Whereas TVM, nGraph, and XLA can leverage all CPU cores.
>  CPU 上 Glow 的性能最差
>  这是因为 Glow 不支持线程并行，因此无法完全利用多核 CPU，而 TVM, nGraph, XLA 可以利用全部的 CPU 核心

2. **XLA has the similar end-to-end inference performance for both full-fledged models ( ResNet , DenseNet and VGG series) and lightweight models ( MobileNet and MNASNet series). Besides, its inference performance on CPU and GPU is almost the same.** It is known that XLA is embedded in the Tensorflow framework. Tensorflow contains a complicated runtime compared to TVM, nGraph, and Glow, which introduces non-trivial overhead to XLA. In addition, if we increase the batch size (set to one by default in our evaluation) and focus on the throughput of DL compilers, then the overhead of XLA can be ignored with higher throughput.
>  XLA 在完整模型和轻量模型上的端到端推理速度相似，且 CPU 和 GPU 上的推理性能几乎一样
>  XLA 内嵌于 TensorFlow 框架，相较于 TVM, nGraph, Glow，TensorFlow 包含了复杂的 runtime，这位 XLA 引入了不能忽视的开销
>  但是，如果我们增大 batch size (评估中默认值为 1)，并且主要关注 DL 编译器的吞吐量，则随着吞吐量的提高，XLA 的开销可以忽略不计

3. **In general, on CPU, TVM and nGraph achieve better performance across all models than other DL compilers ,** due to the limitations of Glow and XLA described above. TVM has comparable performance with nGraph on full-fledged models, while it is better than nGraph on lightweight models. nGraph relies on the DNNL (previously MKL-DNN) library for acceleration. Thus, nGraph can offload the optimized subgraphs to DNNL and benefit from DNNL’s fine-grained instruction-level JIT optimizations tailored for Intel CPU.
>  总的来看，CPU 上，TVM 和 nGraph 在所有的模型上都取得了比其他 DL 编译器更好的表现
>  TVM 和 nGraph 在完全模型上性能可比，且 TVM 在轻量模型上的性能更好
>  nGraph 依赖 DNNL 库进行加速，因此，nGraph 可以将优化后的子图直接卸载到 DNNL，以受益于 DNNL 针对 Intel CPU 优化的细粒度指令级即时编译优化


![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/44a0e8622e0fc0675e4c954f525f8e855ca952e2586e784d4fcfc5ffb9484f4e.jpg)  
Fig. 6. The performance comparison of convolution layers in Mobile Ne tV 2 1.0 across TVM, TC, Glow and XLA on $\mathsf{V}100\;\mathsf{G P U}$ .


![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/ba6b5ba17ea3beb924eee421706940fff32552e2baa157da7031dd027fcf1dda.jpg)  
Fig. 7. The performance comparison of convolution layers in MobileNetV2 1.0 across TVM, nGraph and Glow on Broadwell CPU.

4. **The tuned TVM (tuned with 200 trials) almost achieves the best performance on both CPU and GPU across all models, especially on lightweight models ( MobileNet , MNASNet series)** . Based on our investigation, this is because the schedules of classic operators inside these models have already been well designed by TVM developers, with the default parameters provided in TVM tophub . The default schedules and parameters can help TVM to achieve similar performance compared to other DL compilers. In addition, the performance difference between the tuned TVM and untuned TVM is negligible on CPU but quite significant on GPU ( $41.26\times$ speedup on average). This is because the GPU has more complicated thread and memory hierarchy than CPU, thus to exploit the computation power, GPU requires more fine-grained scheduling (e.g., tile , split , and reorder in TVM). Therefore, it is crucial to determine the optimal scheduling parameters on GPU, where the autotuning exhibits its effectiveness.
>  经过调优的 TVM 几乎在 CPU 和 GPU 上的所有模型都取得了最佳，尤其在轻量模型
>  这是因为这些模型中的经典算子的调度在 TVM tophub 中提供的默认参数下已经被 TVM 开发者很好地设计过
>  默认的调度和参数可以帮助 TVM 实现和其他 DL 编译器相似的性能，调节过的 TVM 和未调节的 TVM 的性能差异在 CPU 上可以忽略，但在 GPU 上很差异很显著，这是因为 GPU 可以利用更细粒度的调度 (TVM 中的 tile, split, reorder)，因此决定 GPU 上的最优调度参数非常重要，此时自动调优的效果就显现了

## 6.3 Per-layer Performance Comparison
To further compare the capability of backend optimizations of DL compilers, we evaluate the per-layer (convolution layers since they dominate the inference time) performance of the `ResNet50` and ` MobileNetV2_1.0 ` on V100 GPU and Broadwell CPU (single-threaded since Glow lacks multithreading support).
>  我们评估了 DL 编译器在 `ResNet50` 和 `MobileNetV2_1.0` 的各个卷积层上的性能

**Methodology** - To measure the execution time of individual layers, we adopt different methods considering the DL compilers, the hardware (CPU/GPU), and the CNN models. Specifically, 1) On TVM, we re-use the logs of autotuning to extract the kernel shapes and the optimal schedule. Then we rebuild the individual convolution layers and use the `time_evaluator` for evaluation. 2) We extract the execution time through the tracing files of Glow. 3) And we measure the execution time of hand-written kernels on TC. 4) As for nGraph, we make use of the timeline to measure the execution time on CPU. However, the timeline is not supported by its PlaidML backend (which provides GPU support through OpenCL). Besides, there are no available methods to profile the command queues within OpenCL. Therefore, we leave the profiling of the per-layer performance of nGraph on GPU for future work. 4) As for XLA, we leverage the built-in `tf.profiler.experimental` method for CPU performance and the DLProf [71] toolkit from Nvidia for GPU performance.
>  方法
>  针对不同的 DL 编译器、硬件以及 CNN 模型，我们采用不同的方法
>  1. TVM 上，我们重用自动调优的日志，提取 kernel 形状和最优调度，然后重构单个卷积层，使用 `time_evaluator` 进行评估
>  2. Glow 上，我们通过追踪文件提取执行时间
>  3. TC 上，我们评估手写 kernel 的执行时间
>  4. nGraph 上，我们用时间线度量 CPU 上的执行时间，但其 PlaidML 后端不支持时间线，且其 PlaidML 后端目前通过 OpenCL 提供 GPU 支持，而目前没有可用的方法来分析 OpenCL 中的命令队列，因此 nGraph 在 GPU 上的逐层分析暂未实现
>  5. XLA 上，我们在 CPU 上用内建的 `tf.profiler.experimental` 方法，在 GPU 上用 Nvidia 的 DLProf 工具包

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/fd140b9adc2470d62ee5931e00e4252fd357fe1e074739c0c24e37e3a6ed4b68.jpg)  
Fig. 8. The performance comparison of convolution layers in ResNet50 across TVM, TC and Glow on V100 GPU.

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/58980e350486e6a01a934ff58ef0b74b80f4d05f23accf0c84be348a6775bc11.jpg)  
Fig. 9. The performance comparison of convolution layers in ResNet50 across TVM, nGraph and Glow on Broadwell CPU.


**Performance** - From Figure 6, 7 8, 9, we have several observations about the performance illustrated as follows.

1. **nGraph achieves a better performance of the convolution layers on CPU** , which benefits from the co-design of hardware (Intel CPU) and software (compiler, library, and runtime). Whereas, **TVM performs better on GPU across these compilers .** On `MobileNetV2_1.0` , the performance of TVM is not stable, especially on conv1 layer. This is because the autotuning process is affected by other processes on the same machine, and thus it tends to derive the imprecise, even negative scheduling parameters.
>  nGraph 在 CPU 上的表现更好
>  这得益于硬件 (Intel CPU) 和软件 (编译器，库，运行时) 的协同设计
>  TVM 在 GPU 上表现更好
>  TVM 在 `MobileNetV2_1.0` 上表现不稳定，因为其自动调优过程受到同一台机器上其他进程的影响，因此倾向于得出不精确甚至负的调度参数

2. TC allows users to define a tensor computation kernel (e.g., convolution) by the Einstein notion without specifying the shape of input/output tensors (e.g., kernel size). Then the kernel is autotuned and stored in its compilation cache to accelerate further autotuning and compilation. However, in our evaluation, we find the performance of TC heavily relies on the initially compiled kernels . Take `MobileNetV2_1.0` for example, if we initialize the autotuning with layer c1 , then c1 can perform well. But the following ` c*_b*_*` layers become much slower as the layers go deeper (far away from c1 layer). To derive a consistent performance, we need to tune each kernel separately.
>  TC 允许用户用 Einstein 记号，在不指定输入输出 tensor 形状的情况下 (例如 kernel size)，定义 tensor 计算 kernel (例如卷积等)
>  该 kernel 会被自动调优，并且其编译缓存会被存储，以加速进一步的调优和编译
>  在评估中，我们发现 TC 的表现严重依赖初始的编译 kernel
>  例如，`MobileNetV2_1.0` 中，如果我们用 layer c1 初始化，则 c1 表现良好，但之后的 `c*_b*_*` 层随着层数增多会变得越加缓慢
>  因此，为了得到一致的表现，我们需要分别调优各个 kernel

3. **Glow falls behind other compilers to optimize the $1\!\times\!1$ convolutions (e.g., the `b_*` linear layers) of `MobileNetV2_1.0` as well as the depth-wise separable convolutions (e.g., `c*_b*_2` layers) of ResNet50** . It takes a longer time to compute these convolutions both on GPU and CPU. We notice the convolutions are usually fused with other layers (e.g., ReLU, BatchNorm) on Glow, which could be why the lower performance compared to other compilers. Moreover, on CPU, the convolutions at the end of `MobileNetV2_1.0` take a quite shorter time than convolutions at the beginning. According to the tracing log, we notice these convolutions are accelerated by the CPU Conv DKK C8 optimization [79], which applies tiling, layout transformation, and vectorization to convolutions with specific patterns.
>  Glow 在优化 `MobileNetV2_1.0` 的 1x1 卷积和 `RestNet50` 的深度可分离卷积时落后于其他编译器
>  在 Glow 上，这些卷积一般与其他层 (例如 ReLU, BatchNorm) 融合，这或许是其性能低的原因之一
>  此外，CPU 上，`MobileNetV2_1.0` 在末尾的卷积比开头的卷积所花时间要短很多，因为末尾的卷积得到了 CPU Conv DKK C8 优化，它应用了 tiling, layout transformation, vectorization

4. As for XLA, it can automatically compile ( `_XlaCompile` ) the eligible subgraphs from Tensorflow and replace the subgraphs with the resultant binaries ( `_XlaRun` ). In addition, the convolution layers may be clustered with other kernels, and thus their performance is not easy to measure individually. Therefore, we have counted the clustered and the non-clustered convolutions, and the data is shown in Table 3. Note that the `MobileNetV2_1.0` model in Tensorflow is a little bit different from the ONNX model for the beginning and ending layers, however, the linear bottleneck layers are the same. Moreover, if a convolution is to be clustered, it could be measured at most twice till the finishing of `XlaCompile` . Therefore, there are five extreme value in Figure 6 (corresponding with 5 clustered convolutions in `MobileNetV2_1.0` ). Actually, only the clustered kernels are optimized by XLA, while the non-clustered ones are optimized by Tensorflow. Therefore, it is impossible to measure the execution time of a standalone convolution layer optimized by XLA. Consequently, we decide not to include the performance of XLA in Figure 7 - 9.
>  XLA 可以自动编译 `_XlaCompile` 来自 TensorFlow 的符合条件的子图，然后用生成的二进制文件 `_XlaRun` 替换这些子图
>  此外，卷积层会和其他 kernel 聚类在一起，进而难以独立衡量
>  我们统计了聚类和非聚类的卷积
>  TensorFlow 中的 `MobileNetV2_1.0` 在开始和结束层方面和 ONNX 模型略有不同，但线性瓶颈层相同
>  此外，如果一个卷积层需要被聚类，那么它最多可以测量两次，直到 `XlaCompile` 完成
>  因此，图 6 中有五个极值（对应于 `MobileNetV2_1.0` 中的 5 个聚类卷积）
>  实际上，只有聚类的内核被 XLA 优化，而非聚类的内核由 TensorFlow 优化，因此，无法测量单独由 XLA 优化的卷积层的执行时间，因此，我们决定不在图 7-9 中包括 XLA 的性能


Table 3. The number of the clustered and non-clustered convolutions of XLA on V100 GPU and Broadwell CPU.
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/f117d5c02eb74bf771c46746b48000df5c1672ed39947ccb167f1819f43f118c.jpg)


## 6.4 Discussion
Through the above quantitative performance comparison across DL compilers, we can in-depth analyze the coarse-grained end-to-end performance with both frontend (graph-level) and backend (operator-level) optimizations, as well as the fine-grained per-layer performance about the convolutions with backend optimizations. However, there are still open challenges to accurately measure the effectiveness of the optimizations adopted by different DL compilers. One particular difficulty during our evaluation is that the frontend and backend optimizations are usually tightly coupled in existing DL compilers, because 1) the frontend optimizations usually affect a series of operators. Thus the optimized operators as the inputs to the backend optimizations differ across different compilers; 2) these optimizations tend to be co-designed for further exploit the performance opportunities (e.g., clustering in XLA and more advanced optimizations [58 , 61]). Therefore, it is difficult if not impossible to evaluate and compare specific optimizations across DL compilers individually. 
>  通过对不同深度学习编译器的上述定量性能比较，我们可以深入分析前端（图级别）和后端（算子级别）优化的粗粒度端到端性能，以及后端优化的卷积层的细粒度逐层性能
>  然而，准确衡量不同深度学习编译器所采用优化的有效性仍存在一些挑战
>  在我们的评估过程中，一个特别的难点在于现有深度学习编译器中的前端和后端优化通常是紧密耦合的，因为：
>  1) 前端优化通常会影响一系列算子，因此作为后端优化输入的优化算子在不同编译器中有所不同；
>  2) 这些优化通常被共同设计以进一步挖掘性能机会（例如 XLA 中的聚类和其他高级优化[58, 61]）
>  因此，单独评估和比较不同深度学习编译器中的特定优化可能是困难的，甚至是不可能的

To tackle this problem, we have been working on building a universal benchmarking framework for existing DL compilers to measure the per-layer performance. The fundamental idea is to extract the necessary structures and parameters of the target layers (we name them as model fragments ), and rebuild the layers as acceptable inputs to a particular DL compiler, which allows the compiler to apply corresponding frontend and backend optimizations faithfully. We can then measure the performance of these optimized model fragments to understand the effectiveness of DL compilers at layers of interests. The benchmarking framework using model fragments is scalable to customized layers (e.g., fused layers) of interest. With such benchmarking framework available, we can derive both coarse-grained (e.g., end-to-end) and fine-grained (e.g., per-layer) performance metrics for each DL compiler, and thus compare the effectiveness of optimizations across different DL compilers at the level of interest. Currently, we have successfully experimented by extracting the target layers from the state-of-the-art CNN models, such as the bottleneck of $\mathsf{R e s N e t}50$ and the linear bottleneck of `MobileNetV2_1.0` . Our benchmarking framework is still under rapid development, and we hope to make it available to the community soon.
>  为了解决这个问题，我们一直在努力构建一个通用的基准测试框架，用于现有深度学习编译器的逐层性能测量
>  基本思想是从目标层提取必要的结构和参数（我们称之为模型片段），并重建这些层作为特定深度学习编译器的可接受输入，从而允许编译器忠实地应用相应的前端和后端优化
>  然后，我们可以测量这些优化模型片段的性能，以了解深度学习编译器在感兴趣层面上的有效性，基于模型片段的基准测试框架可以扩展到感兴趣的自定义层（例如融合层）
>  有了这样的基准测试框架，我们可以为每个深度学习编译器推导出粗粒度（例如端到端）和细粒度（例如逐层）的性能指标，从而比较不同深度学习编译器在感兴趣层面上优化的有效性
>  目前，我们已经成功地从最先进的CNN模型中提取了目标层，例如 $\mathsf{ResNet}50$ 的瓶颈层和 `MobileNetV2_1.0` 的线性瓶颈层。我们的基准测试框架仍在快速发展中，我们希望尽快将其提供给社区

# 7 Conclusion and Future Directions
In this survey, we present a thorough analysis of the existing DL compilers targeting the design principles. First, we take a deep dive into the common architecture adopted in the existing DL compilers including the multi-level IR, the frontend and the backend. We present the design philosophies and reference implementations of each component in detail, with the emphasis on the unique IRs and optimizations specific to DL compilers. We summarize the findings in this survey and highlight the future directions in DL compiler as follows:
>  在这篇综述中，我们对现有的针对设计原则的深度学习编译器进行了全面的分析
>  首先，我们深入探讨了现有深度学习编译器中常见的架构，包括多级中间表示（IR）、前端和后端
>  我们详细介绍了每个组件的设计理念和参考实现，重点强调了独特的IR和专用于深度学习编译器的优化
>  我们总结本综述中的发现，并指出以下未来深度学习编译器的发展方向：

**Dynamic shape and pre/post processing** - Dynamic model becomes more and more popular in the field of DL, whose input shape or even model itself may change during execution. Particularly, in the area of NLP, models may accept inputs of various shapes, which is challenging for DL compilers since the shape of data is unknown until runtime. Existing DL compilers require more research efforts to support dynamic shape efficiently for emerging dynamic models.
>  动态形状和前/后处理
>  动态模型在深度学习领域越来越受欢迎，其输入形状或甚至模型本身在执行过程中可能会发生变化
>  特别是在自然语言处理（NLP）领域，模型可能接受各种形状的输入，这对深度学习编译器提出了挑战，因为数据的形状在运行时才知道

In addition, as future DL models become more complex, their entire control flow may inevitably include complicated pre/post-processing procedures. Currently, most DL compilers use Python as their programming language, the pre/post-processing could become a performance bottleneck when it is executed by the Python interpreter. Such potential performance bottleneck has not yet been considered by existing DL compilers. Supporting the entire control flow in DL compiler enables express and optimize the pre/post-processing along with DL models, which opens up new opportunities for performance acceleration in model deployment.
>  此外，随着未来的深度学习模型变得更加复杂，其整个控制流可能不可避免地包含复杂的前处理和后处理步骤
>  目前，大多数深度学习编译器使用Python作为编程语言，当由Python解释器执行时，这些前处理和后处理可能会成为性能瓶颈，现有深度学习编译器尚未考虑这种潜在的性能瓶颈
>  支持深度学习编译器中的整个控制流可以表达和优化与深度学习模型相关的前处理和后处理，从而为模型部署带来新的性能加速机会

**Advanced auto-tuning** - Existing auto-tuning techniques focus on the optimization of individual operators. However, the combination of the local optimal does not lead to global optimal. For example, two adjacent operators that apply on different data layouts can be tuned together without introducing extra memory transformations in between. Besides, with the rise of edge computing, execution time is not only the optimization objective for DL compilers. New optimization targets should also be considered in the auto-tuning such as memory footprint and energy consumption.
>  高级自动调优
>  现有的自动调优技术主要集中在单个算子的优化上
>  然而，局部最优的结合并不一定导致全局最优，例如，两个在不同数据布局上操作的相邻算子可以在不引入额外内存转换的情况下一起调优
>  此外，随着边缘计算的兴起，执行时间不再是深度学习编译器的唯一优化目标，自动调优还应考虑新的优化目标，如内存占用和能耗。

Particularly, for the ML-based auto-tuning techniques, there are several directions worth further exploring. First, the ML techniques can be applied in other stages of auto-tuning, other than the cost model. For example, in the stage of selecting compiler options and optimization schedules, ML techniques can be used to predict the possibility directly and develop algorithms to determine the final configurations. Second, the ML-based auto-tuning techniques can be improved based on the domain knowledge. For example, incorporating the feature engineering (selecting features to represent program) [99] in auto-tuning techniques could be a potential direction for achieving better tuning results.
>  特别是在基于机器学习的自动调优技术方面，有几个值得进一步探索的方向
>  首先，机器学习技术可以应用于自动调优的其他阶段，而不仅仅是成本模型，例如，在选择编译器选项和优化调度阶段，可以使用机器学习技术直接预测可能性并开发算法确定最终配置
>  其次，基于机器学习的自动调优技术可以结合领域知识进行改进，例如，在自动调优技术中融入特征工程（选择表示程序的特征）[99] 可以是一个潜在的方向，以获得更好的调优效果

**Polyhedral model** - It is a promising research direction to combine polyhedral model and auto-tuning techniques in the design of DL compilers for efficiency. On one hand, the auto-tuning can be applied to minimize the overhead of polyhedral JIT compilation by reusing the previous configurations. On the other hand, the polyhedral model can be used to perform auto-scheduling, which can reduce the search space of auto-tuning. 
>  多面体模型
>  将多面体模型与自动调优技术结合在深度学习编译器的设计中是一个有前景的研究方向
>  一方面，自动调优可以通过重用之前的配置来最小化多面体即时编译的开销
>  另一方面，多面体模型可以用于自动调度，从而减少自动调优的搜索空间。  

Another challenge of applying polyhedral model in DL compilers is to support the sparse tensor. In general, the format of a sparse tensor such as CSF [84] expresses the loop indices with index arrays (e.g., $a[b[i]])$ that is no longer linear. Such indirect index addressing leads to non-affine subscript expressions and loop bounds, which prohibits the loop optimization of the polyhedral model [14 , 90]. Fortunately, the polyhedral community has made progress in supporting sparse tensor [94 , 95], and integrating the latest advancement of the polyhedral model can increase the performance opportunities for DL compilers.
>  在深度学习编译器中应用多面体模型的另一个挑战是支持稀疏张量
>  一般来说，例如 CSF [84] 格式的稀疏张量通过索引数组（如 $a[b[i]]$）表示循环索引，这不再是线性的。这种间接索引寻址导致非仿射下标表达式和循环边界，禁止了多面体模型的循环优化 [14, 90]
>  幸运的是，多面体社区已经在支持稀疏张量方面取得了进展 [94, 95]，结合最新的多面体模型进展可以为深度学习编译器增加更多性能机会

**Subgraph partitioning** - DL compilers supporting subgraph partitioning can divide the computation graph into several subgraphs, and the subgraphs can be processed in different manners. The subgraph partitioning presents more research opportunities for DL compilers. First, it opens up the possibility to integrate graph libraries for optimization. Take nGraph and DNNL for example, DNNL is a DL library with graph optimizations leveraging vast collection of highly optimized kernels. The integration of DNNL with nGraph enables DNNL to speedup the execution of the subgraphs generated by nGraph. Secondly, it opens up the possibility of heterogeneous and parallel execution. Once the computation graph is partitioned into subgraphs, the execution of different subgraphs can be assigned to heterogeneous hardware targets at the same time. Take the edge device for example, its computation units may consist of ARM CPU, Mail GPU, DSP, and probably NPU. Generating subgraphs from the DL compilers that utilizes all computation units efficiently can deliver significant speedup of the DL tasks.
>  子图划分
>  支持子图划分的深度学习编译器可以将计算图划分为几个子图，并以不同的方式处理这些子图，子图划分为深度学习编译器提供了更多的研究机会
>  首先，它开启了整合图形库进行优化的可能性，例如，nGraph 和 DNNL，DNNL 是一种利用大量高度优化的内核进行图优化的深度学习库，DNNL 与 nGraph 的集成使得 DNNL 能够加速由 nGraph 生成的子图的执行
>  其次，它开启了异构和并行执行的可能性，一旦计算图被划分为子图，不同子图的执行可以同时分配给异构硬件目标，例如，在边缘设备上，其计算单元可能包括 ARM CPU、Mail GPU、DSP，甚至 NPU，由深度学习编译器生成的所有计算单元高效利用的子图可以显著加快深度学习任务的速度

**Quantization** - Traditional quantization strategies applied in DL frameworks are based on a set of fixed schemes and datatypes with little customization for codes running on different hardware. Whereas, supporting quantization in DL compilers can leverage optimization opportunities during compilation to derive more efficient quantization strategies. For example, Relay [78] provides a quantization rewriting flow that can automatically generate quantized code for various schemes.
>  量化
>  传统的量化策略在深度学习框架中基于一组固定的方案和数据类型，对不同硬件上运行的代码几乎没有定制
>  然而，支持量化可以使编译器在编译过程中利用优化机会，从而得出更有效的量化策略，例如，Relay [78]提供了一个自动化的量化重写流程，可以为各种方案自动生成量化代码

To support quantization, there are several challenges to be solved in DL compilers. The first challenge is how to implement new quantized operators without heavy engineering efforts. The attempt from AWS points out a possible direction that uses the concept of dialect to implement new operators upon basic operators, so that the optimizations at graph level and operator level can be reused. The second challenge is the interaction between quantization and other optimizations during compilation. For example, determining the appropriate stage for quantization and collaborating with optimizations such as operator fusion require future research investigations.
>  为了支持量化，深度学习编译器需要解决几个挑战
>  第一个挑战是如何在不进行大量工程工作的前提下实现新的量化算子，AWS 的一个尝试指出了一个可能的方向，即使用方言的概念在其基础算子之上实现新算子，从而可以重用图形层面和算子层面的优化
>  第二个挑战是在编译过程中量化与其他优化之间的交互，例如，确定量化的适当阶段并与算子融合等优化协作需要进一步的研究调查

**Unified optimizations** - Although existing DL compilers adopt similar designs in both computation graph optimizations and hardware-specific optimizations, each compiler has its own advantages in certain aspects. There is a missing way to share the state-of-the-art optimizations, as well as support of emerging hardware targets across existing compilers. We advocate unifying the optimizations from existing DL compilers so that the best practices adopted in each DL compiler can be reused. In addition, unifying the optimizations across DL compilers can accumulate a strong force to impact the design of general-purpose and dedicated DL accelerators, and provide an environment for efficient co-design of DL compiler and hardware.
>  统一优化
>  虽然现有的深度学习编译器在计算图优化和硬件特定优化方面采用了类似的设计，但每个编译器在某些方面都有自己的优势
>  目前缺少一种方法来共享最先进的优化以及支持现有编译器中的新兴硬件目标
>  我们提倡统一现有深度学习编译器的优化，以便可以重用每个深度学习编译器采用的最佳实践
>  此外，跨深度学习编译器统一优化可以积累强大的力量影响通用和专用深度学习加速器的设计，并为深度学习编译器和硬件的高效协同设计提供环境

Currently, Google MLIR is a promising initiative towards such direction. It provides the infrastructure of multi-level IRs, and contains IR specification and toolkit to perform transformations across IRs at each level. It also provides flexible dialects , so that each DL compiler can construct its customized dialects for both high-level and low-level IRs. Through transformation across dialects , optimizations of one DL compiler can be reused by another compiler. However, the transformation of dialects requires further research efforts to reduce the dependency on delicate design.  
>  目前，Google MLIR 是朝着这一方向迈出的有希望的一步，它提供了多级 IR 的基础设施，并包含了在各级 IR 之间进行转换的 IR 规范和工具包
>  它还提供了灵活的方言，使得每个深度学习编译器可以为其高低级 IR 构建自定义方言，通过方言间的转换，一个编译器的优化可以被另一个编译器重用
>  然而，方言的转换还需要进一步的研究努力来减少对精巧设计的依赖

**Differentiable programming** - Differentiable programming is a programming paradigm, where the programs are differentiable thoroughly. Algorithms written in differentiable programming paradigm can be automatically differentiated, which is attractive for DL community. Many compiler projects have adopted differentiable programming, such as Myia [89], Flux [40] and Julia [13]. Unfortunately, there is little support for differential programming in existing DL compilers.
>  可微分编程
>  可微分编程是一种编程范式，在这里程序是完全可微的
>  用可微分编程范式编写的算法可以自动进行微分，这对于深度学习社区非常有吸引力，许多编译器项目已经采用了可微分编程，如 Myia [89]、Flux [40]和 Julia [13]，不幸的是，现有的深度学习编译器很少支持可微分编程。

To support differential programming is quite challenging for existing DL compilers. The difficulties come from not only data structure, but also language semantic. For example, to realize the transformation from Julia to XLA HLO IR, one of the challenges [24] is that the control flow is different between the imperative language used by Julia and the symbolic language used by XLA. In order to use HLO IR efficiently, the compiler also needs to provide operation abstraction for Julia in order to support the particular semantic of XLA, such as MapReduce and broadcast . Moreover, the semantic difference of differentiation between Julia and XLA, also requires significant changes of compiler designs.
>  支持可微分编程对现有的深度学习编译器来说极具挑战。这些挑战不仅来自数据结构，还来自语言语义
>  例如，要将 Julia 转换为 XLA HLO IR，其中一个挑战是 Julia 使用的指令式语言与 XLA 使用的符号语言之间的控制流不同。为了有效地使用 HLO IR，编译器还需要为 Julia 提供操作抽象，以便支持 XLA 特有的语义，如 MapReduce 和广播
>  此外，Julia 和 XLA 之间微分语义的差异也需要对编译器设计进行重大修改。

**Privacy protection** - In edge-cloud system, the DL models are usually split into two halves with each partial model running on the edge device and cloud service respectively, which can provide better response latency and consume less communication bandwidth. However, one of the drawbacks with the edge-cloud system is that the user privacy becomes vulnerable. The reason is that the attackers can intercept the intermediate results sent from the edge devices to cloud, and then use the intermediate results to train another model that can reveal the privacy information deviated from the original user task.
>  隐私保护
>  在边缘-云系统中，深度学习模型通常会被分割成两半，每部分模型分别在边缘设备和云服务上运行，这可以提供更好的响应延迟并消耗较少的通信带宽。
>  然而，边缘-云系统的一个缺点是用户隐私变得脆弱。原因在于攻击者可以拦截从边缘设备发送到云端的中间结果，然后使用这些中间结果训练另一个可以揭示与原始用户任务无关的隐私信息的模型。

To protect privacy in edge-cloud system, existing approaches [27 , 67 , 74] propose to add noise with special statistic properties to the intermediate results that can reduce the accuracy of the attacker task without severely deteriorating the accuracy of the user task. However, the difficulty is to determine the layer where the noise should be inserted, which is quite labor intensive to identify the optimal layer. The above difficulty presents a great opportunity for DL compilers to support privacy protection, because the compilers maintain rich information of the DL model, which can guide the noise insertion across layers automatically.
>  为了在边缘-云系统中保护隐私，现有方法 [27, 67, 74] 提出向中间结果添加具有特殊统计特性的噪声，以降低攻击者的任务准确性而不严重损害用户的任务准确性
>  然而，确定噪声应该插入的层非常耗时。上述困难为深度学习编译器支持隐私保护提供了巨大的机会，因为编译器维护了大量的深度学习模型信息，可以自动引导跨层的噪声插入。

**Training support** - In general, the model training is far less supported in current DL compilers. As shown in Table 1, nGraph only supports training on the Intel NNP-T accelerator, TC only supports the auto differentiation of a single kernel, Glow has experimental training support for limited models, the training support of TVM is under development, while XLA relies on the training support of TensorFlow. In sum, current DL compilers mainly focus on bridging the gap of deploying DL models onto diverse hardware efficiently, and thus they choose inference as their primary optimization targets. However, expanding the capability of DL compilers to support model training would open up a large body of research opportunities such as optimization of gradient operators and high-order auto differentiation.
>  训练支持
>  总体而言，当前的深度学习编译器对模型训练的支持远远不够。如表 1 所示，nGraph 仅支持在 Intel NNP-T 加速器上的训练，TC 仅支持单个内核的自动微分，Glow 对有限的模型提供实验性的训练支持，TVM 的训练支持正在开发中，而 XLA 则依赖于 TensorFlow 的训练支持
>  总之，当前的深度学习编译器主要关注高效地将深度学习模型部署到多样化的硬件上，因此它们选择推理作为主要的优化目标。然而，扩展深度学习编译器的能力以支持模型训练将开启大量的研究机会，例如梯度算子的优化和高阶自动微分。