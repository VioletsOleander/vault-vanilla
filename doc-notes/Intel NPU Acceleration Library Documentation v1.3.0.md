# Library Overview
## 1 Quickstart
The Intel® NPU Acceleration Library is a Python library designed to boost the efficiency of your applications by leveraging the power of the Intel Neural Processing Unit (NPU) to perform high-speed computations on compatible hardware.
### 1.1 Installation
Check that your system has an available NPU ([how-to](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html)).
You can install the packet in your machine with

```
pip install intel-npu-acceleration-library
```
### 1.2 Run a LLaMA model on the NPU
To run LLM models you need to install the transformers library

```
pip install transformers
```

You are now up and running! You can create a simple script like the following one to run a LLM on the NPU

```python
from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = NPUModelForCausalLM.from_pretrained(model_id, use_cache=True, dtype=torch.int8).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

query = input("Ask something: ")
prefix = tokenizer(query, return_tensors="pt")["input_ids"]

generation_kwargs = dict(
   input_ids=prefix,
   streamer=streamer,
   do_sample=True,
   top_k=50,
   top_p=0.9,
   max_new_tokens=512,
)

print("Run inference")
_ = model.generate(**generation_kwargs)
```

Take note that you only need to use *intel_npu_acceleration_library.compile* to offload the heavy computation to the NPU.

Feel free to check [Usage](https://intel.github.io/intel-npu-acceleration-library/usage.html) and [LLM](https://intel.github.io/intel-npu-acceleration-library/llm.html) and the [examples](https://github.com/intel/intel-npu-acceleration-library/tree/main/examples) folder for additional use-cases and examples.
## 2 Quick overview of Intel's Neural Processing Unit (NPU)
![[Intel NPU Architecture.png]]

The Intel NPU is an AI accelerator integrated into Intel Core Ultra processors, characterized by a unique architecture comprising compute acceleration and data transfer capabilities. Its compute acceleration is facilitated by Neural Compute Engines, which consist of hardware acceleration blocks for AI operations like Matrix Multiplication and Convolution, alongside Streaming Hybrid Architecture Vector Engines for general computing tasks.
> Intel NPU 是集成入 Intel Core Ultra 处理器的 AI 加速设备
> NPU 通过神经计算引擎加速计算，神经计算引擎由针对 AI 运算例如矩阵乘法和卷积的 AI 加速块构成，NPU 还有流混合架构向量引擎，用于通用计算任务

- **Scalable Multi-Tile Design:** The heart of the NPU's compute acceleration capability lies in its scalable tiled based architecture known as Neural Compute Engines.
- **Hardware Acceleration Blocks:** These engines are equipped with specific hardware blocks designed to handle AI operations that demand high levels of computation, such as Matrix Multiplication and Convolution.
- **Streaming Hybrid Architecture:** Alongside the dedicated AI operation units, the Neural Compute Engines are built with Streaming Hybrid Architecture Vector Engines (SHAVE). This enables them to perform high-performance parallel computing for general compute needs.
- **DMA Engines:** Direct Memory Access (DMA) engines are integral to the NPU, responsible for moving data efficiently between the system memory DRAM and the software-managed cache.
- **Memory Management:** The incorporation of a built-in device MMU, alongside an IOMMU, allows support for multiple concurrent hardware contexts. This is crucial for maintaining security isolation between these contexts in line with the Microsoft Compute Driver Model (MCDM) architectural standards.
> 可拓展的多 Tile 设计：NPU 计算加速能力的核心就是 scalable tiled based architecture 的神经计算引擎
> 硬件加速块：神经计算引擎中由针对 AI 运算例如矩阵乘法和卷积的硬件块 
> 流混合架构：除了专用 AI 模块，神经计算引擎还有流混合架构向量引擎 (Streaming Hybrid Architecture Vector Engines SHAVE)，针对通用计算需求执行高性能并行计算
> DMA 引擎：直接存储访问引擎和 NPU 集成，负责在系统 DRAM 和软件管理的缓存之间高效移动数据
> 存储管理：内建的设备 MMU 和 IOMMU 允许支持多个并发硬件上下文，这对于维护遵循 Microsoft Compute Diver Model 架构标准的上下文之间的安全隔离十分重要
### 2.1 The Role of Software
While the hardware is undoubtedly advanced, the true "magic" of the Intel NPU is realized through a sophisticated MLIR based compiler. It is through compiler technology that Intel's NPU reaches its full potential by optimizing and orchestrating AI workloads.
> 通过基于 MLIR 的编译器使用 NPU

- **Parallel Workload Execution:** The compiler ensures that AI tasks are executed in parallel, directing both compute and data flows in a tiling pattern with built-in and programmable control flows.
- **Maximizing Compute Utilization:** By prioritizing execution primarily out of scratchpad SRAM and reducing the data transfers between SRAM and DRAM, the compiler helps in achieving optimum performance-to-power ratios for AI workloads.

> 并行工作负载执行：编译器保证 AI 任务可以并行执行，引导计算和数据流在内建可编程控制流下以 tiling 的模式进行
> 最大化计算利用：通过优先执行暂存区 SRAM 中的数据，减少 SRAM 和 DRAM 之间的数据传输，编译器 AI 工作负载达到最优的性能-能耗比值
## 3 Basic usage
For implemented examples, please check the `examples` folder
### 3.1 Run a single MatMul in the NPU
```python
from intel_npu_acceleration_library.backend import MatMul
import numpy as np

inC, outC, batch = ... # Define your own values

# Create both inputs
X1 = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

mm = MatMul(inC, outC, batch, profile=False)

result = mm.run(X1, X2)

```
### 3.2 Compile a model for the NPU
If you have `pytorch` >=2.0.0 installed you can use torch compile to optimize your model for the NPU
> `pytorch` 版本大于 2.0.0 时，可以使用 `torch.compile` 针对 NPU 优化模型

```python
import intel_npu_acceleration_library
import torch

# Compile model for the NPU
# model a torch.nn.Module class. Model can be quantized JIT
optimized_model = torch.compile(model, backend="npu")

# Use the model as usual

```

In windows torch.compile is not supported yet. So you might want to use the explicit function `intel_npu_acceleration_library.compile`. This is true also if you use a `pytorch` version < 2.0.0
> Windows 上尚且不支持 torch.compile，因此需要使用显式函数 `intel_npu_acceleration_library.compile` ，这在 pytorch 版本小于2.0.0时也可以使用

To do this, you just need to call the `compile` function with your model and the compiler configuration `CompilerConfig` to compile and optimize the model for the NPU.
> `intel_npu_acceleration_library.compile(model, compiler_conf)`

```python
import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig
compiler_conf = CompilerConfig(dtype=torch.int8)
optimized_model = intel_npu_acceleration_library.compile(model, compiler_conf)

# Use the model as usual

```

To compile and optimize a single layer of a model to be pushed to the NPU as one block, you can set `use_to=True` in the the compiler configuration `CompilerConfig`.
> 要编译并且优化模型的一层，并将它作为一个块推送到 NPU，我们可以在编译器配置 `CompilerConfig` 中设定 `use_to=True`

```python
import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig
compiler_conf = CompilerConfig(use_to=True, dtype=torch.int8)
optimized_block = intel_npu_acceleration_library.compile(single_block, compiler_conf)

```
### 3.3 Training (**Experimental!**)
It is possible to use Intel® NPU Acceleration Library to train a model. As before you just need to call the `compile` function, this time with `training=True`. This allows to use the same training script you use in other device with a very minimal modifications.
> 使用 NPU Acceleration Library 来训练一个模型是可能的
> 我们需要调用 `compile` 函数，并且设定  CompilerConfig 的 `training=True`  `

```python
import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig
compiler_conf = CompilerConfig(dtype=torch.float32, training=True)
compiled_model = intel_npu_acceleration_library.compile(model, compiler_conf)
```
## 4 Advanced Setup
You can install the package by typing

```
pip install "intel-npu-acceleration-library @ git+https://github.com/intel/intel-npu-acceleration-library.git"
```

To build the package you need a compiler in your system (Visual Studio 2019 suggested for Windows build). MacOS is not yet supported.

For development packages use (after cloning the repo)

```
pip install .[dev]
```
# Applications
## 5 Large Language models
### 5.1 Run an LLM on the NPU
You can use your existing LLM inference script on the NPU with a simple line of code
> 唯一要对现存的推理脚本修改的就是将 `model` 经过 `intel_npu_acceleration_library.compile` 编译一下

```python
# First import the library
import intel_npu_acceleration_library

# Call the compile function to offload kernels to the NPU.
model = intel_npu_acceleration_library.compile(model)
```

Here a full example:

```python
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
from threading import Thread
import intel_npu_acceleration_library
import torch
import time
import sys

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

print("Compile model for the NPU")
model = intel_npu_acceleration_library.compile(model)

query = "What is the meaning of life?"
prefix = tokenizer(query, return_tensors="pt")["input_ids"]

generation_kwargs = dict(
    input_ids=prefix,
    streamer=streamer,
    do_sample=True,
    top_k=50,
    top_p=0.9,
)

print("Run inference")
_ = model.generate(**generation_kwargs)
```
## 6 Decoding LLM performance
![[LLM inference phases.png]]

Decoding and understanding the performance of large language models (LLMs) is critical for optimizing their efficiency and effectiveness. The inference process of an LLM can be broken down into three distinct phases, each with its unique characteristics and performance considerations, as shown in the following figure.
> LLM 的推理分解为三个阶段
### 6.1 Load phase
The load phase encompasses the initial steps of bringing an LLM into action, starting from loading the model into memory until the `model.generate()` call is initiated.
> 装载阶段从将模型装载到内存中开始，指导 `model.generate()` 调用被发起结束
#### 6.1.2 Phase Steps
- Weight Loads: load phase latency is largely dependent on how quickly the model weights can be loaded from the disk as part of model initialization.
- Quantization: Quantization involves the process of reducing the precision of the weights, which can impact the performance. This step is designed to balance the trade-off between model accuracy and the computational efficiency of the model. Quantizing the weights involves analyzing the entire model to lower the precision of its parameters. Depending on its implementation, it can be an expensive process and might require fine-tuning the model for best performance.
- Compilation: is the process of transforming the original model into a format that can be run on the NPU. It involves some model optimizations as well as lowering the operation into a NPU runtime format.
> 装载阶段步骤：
> Weight Loads：装载阶段的延迟大部分取决于模型权重从磁盘装载出来的速度
> Quantization：量化是减少权重精度的过程，量化是模型准确率和计算效率之间的 trade-off，量化可以是一个昂贵的过程，且可能需要对模型进行微调
> Compilation：将原始模型转化为可以运行在 NPU 上的格式，编译包括了一些模型优化操作以及将运算降低到 NPU 运行时格式的操作
#### 6.1.2 Implications
- CPU/Disk Bound: since this phase relies heavily on I/O operations and CPU activities, the underlying CPU and disk speed is what bounds performance.
- Pre-compilation: quantizing and in a lesser extent compiling a model might result in a significative latency. It is suggested to prepare the model offline and not during the application if it is possible. An example of how this can be done is in the `export.py` script in the `script` folder. That do not removes the needs to load the weights from the disk at initialization stage but remove the compilation and quantization latency.
> CPU/Disk Bound: 装载阶段依赖于大量 IO 操作和 CPU 活动，因此 CPU 和磁盘速度或限制装载阶段的速度
> Pre-compilation: 编译/量化模型可能会导致显著的延迟，建议在 offline 时准备好模型，即预编译，而不是在应用程序运行时再编译，这样可以直接移除量化和编译延迟
### 6.2 Prefill phase
In the prefill phase, the model analyzes the user prompt to produce the initial output token. The primary metric used is `prefill-time` (a.k.a. first inference latency), which gauges the duration from the LLM's initiation to the generation of the first token. This interval is commonly interpreted by users as the "LLM startup time" as it denotes the period from when they commence typing to when the LLM begins its response. A brief `prefill-time` enhances system responsiveness and user satisfaction.
> 在预填充阶段，模型会分析用户 prompt 以生成初始的输出 token
> 其主要使用的度量是 `prefill-time` ，即第一次推理延迟，也就是从 LLM 启动到第一次 token 生成所经过的时间，这一段时间一般会被用户称为 LLM 启动时间
#### 6.2.1 Phase Steps
- Fist inference: model first inference on the user's prompt. This process can be computationally intensive, particularly with long prompts, as it processing requires significant matrix-matrix multiplications.
- Key-Value cache (KV-cache): the prompt key and value output from every attention layer can be cached for the next tokens generation in order to save computation.
> 预填充阶段步骤：
> First inference: 模型对用户的 prompt 第一次推理，该过程可以是计算密集的 (例如对于长的 prompt，需要大量的矩阵乘法)
> Key-Value cache: prompt 在每一个 attention 层的 key 和 value 输出可以被缓存用于下一个 token 生成，以节约计算
#### 6.2.2 Implications
- Compute bounded (NPU): the initial inference process is primarily limited by computational resources (NPU) due to the typically substantial size of the user's prompt.
- Input prompt size: The latency during this phase is contingent upon the length of the user's prompt. A lengthier prompt results in a quadratic increase in runtime due to the LLM's multi-head attention block.
> Compute bounded: 初始推理过程主要由计算资源限制，因为用户 prompt 一般会比较大
> Input prompt size: 预填充阶段的延迟主要依赖于用户的 prompt，prompt 的长度增加会导致运行时间平方地增加
### 6.3 Token Phase
After the prefill, the LLM enters the token phase, where it generates the remaining tokens in the output sequence. The primary metrics used are `token-time` and `tokens/s`.
> 预填充之后，LLM 进入 token 阶段
> LLM 在 token 阶段生成输出序列中剩余的 tokens，该阶段主要使用的度量是 `token-time` 和 `tokens/s` 
#### 6.3.1 Phase Steps
- Inference: The generated token alongside the KV-cache is passed as input to the model. Because of KV-cache optimization, the required compute is fairly limited as effectively the LLM runs with a single new token as input.
- Weight loads: while compute is limited, the model still needs to load the entire weight-set (potentially billions of parameters) to perform the computation. Therefore, execution is mostly limited by DRAM bandwidth rather than compute capability.
> Token 阶段步骤：
> Inference: 生成的 token 和 KV-cache 一起作为输入传递给模型，因为 KV 已经缓存，故此时需要的计算就是 LLM 计算单个 token 输出所需的计算
> Weight loads: 虽然计算量有限制，但是模型仍然需要将整个权重集合装载以执行计算，因此此时的执行主要受到 DRAM 带宽限制而不是计算能力
#### 6.3.2 Implications
- DRAM Bandwidth: This stage of the inference is driven significantly by the bandwidth of the DRAM. The rate at which the LLM parameters are transferred from DRAM to the processing units has a considerable effect on the token time.
- Performance Factors: Although NPU performance still matters, it becomes less of the bottleneck compared to the available DRAM bandwidth.
> DRAM Bandwidth: token 阶段的推理主要由 DRAM 驱动，LLM 参数从 DRAM 传输到处理单元的速度对于 token time 有显著的影响
> Performance Factors: NPU 性能此时相较于 DRAM 带宽不再是性能瓶颈
### 6.4 System/application parameters
Beyond the phases, certain system parameters significantly influence the performance of LLMs.
> 除了这三个阶段，一些系统参数也会影响 LLM 的表现

- Model Architecture and Size: The architecture and the size of the model dictate its performance. Larger models, which have more parameters, may provide more accurate results but are also more challenging to fit within the physical memory limits of a system.
- DRAM size and speed: Once DRAM is filled, the performance can become bottlenecked. If the model and its KV-cache overflow the available DRAM, the system will need to swap memory to disk leading to a much slower inference.
- Prompt Length: Different applications may require support for varying prompt lengths. Longer prompts translate into larger context sizes, increasing the demand on cache and tensor resources.
- LLM Context Size: As the context size grows (large prompt and/or significative number of newly generated tokens) and hits the DRAM limit, performance may again become SWAP/SSD bounded due to insufficient DRAM to contain the larger KV-cache tensors.
> Model Architecture and Size: 模型的架构和大小决定了它的性能，但是越大的模型参数越多，对于系统的物理内存大小的挑战越大
> DRAM size and speed: 只要 DRAM 会被填充满，DRAM 就会成为瓶颈，模型的 KV-cache 导致 DRAM 溢出就需要系统从磁盘中交换内存，导致推理变慢
> Prompt Length: 不同的应用可能需要对于不同的 prompt 长度的支持，而越长的 prompt 就意味着上下文大小越大，就需要更多的缓存和 tensor 资源
> LLM Context Size: 随着上下文大小增长，KV-cache 张量数量增多，达到了 DRAM 限制，性能就会被 SWAP/SSD 限制
### 6.5 Performance improvement
Increasing the DRAM size/speed:

Model Quantization: quantization reduces model footprint and enables faster computations on supported hardware. This is expected to give performance benefits on all inference phases. It is important to notice that quantization by itself might reduce model quality and accuracy and so LLM performance should be the target of extensive investigation.

Static shape inference: many inference AI accelerators (Intel NPU, IPU, TPU, etc...) requires static shapes get maximum performance. Static shapes allows the NN graph compiler to improve memory management, schedule and overall network performance. For a example implementation, you can refer to the `intel_npu_acceleration_library.nn.llm.generate_with_static_shape` or `transformers` library [StaticCache](https://huggingface.co/docs/transformers/v4.38.1/en/internal/generation_utils#transformers.StaticCache)
> 提高性能，考虑：
> 增加 DRAM 大小/速度
> 模型量化: 减少模型足迹，并让计算更快，但会降低模型质量和准确率
> 静态形状推理: 许多推理 AI 加速设备 (Intel NPU, IPU, TPU 等) 需要静态形状以达到最高性能，静态形状允许 NN 图编译器提高存储管理、调度和整体的网络性能
### 6.6 Conclusions
Understanding these phases and system parameters is crucial to diagnose performance bottlenecks, to fairly compare LLM performance over different accelerators and to develop strategies for optimizing the deployment and execution of LLMs on client and edge platform. By paying close attention to these aspects, one can ensure that the model operates efficiently, providing quick and accurate responses to user prompts.
# Developments guide
## 7 Developer Guide
Install developer packages by typing

```
pip install .[dev]
```

It is suggested to install the package locally by using `pip install -e .[dev]`
### 7.1 Git hooks
All developers should install the git hooks that are tracked in the `.githooks` directory. We use the pre-commit framework for hook management. The recommended way of installing it is using pip:

```
pre-commit install
```

If you want to manually run all pre-commit hooks on a repository, run `pre-commit run --all-files`. To run individual hooks use `pre-commit run <hook_id>`.

Uninstalling the hooks can be done using

```
pre-commit uninstall
```
### 7.2 Testing the library

#### 7.2.1 Python test
Python test uses `pytest` library. Type

```
cd test/python && pytest
```

to run the full test suite.
### 7.3 Build the documentation
This project uses `sphinx` to build and deploy the documentation. To serve locally the documentation type

```
mkdocs serve
```

to deploy it into github pages type

```
cd docs
python build_doc.py gh-deploy
```
### 7.4 Generate python packages
On windows:

```
python setup.py sdist
set CIBW_BUILD=cp*
cibuildwheel --platform windows --output-dir dist
```
### 7.5 Publishing packets
Install twine

```
python3 -m pip install --upgrade twine
```

Then check on the built sdist and wheel that are properly formatted (all files should return a green `PASSED`)

```
twine check dist/*
```

Upload the packets to `testpypi`

```
twine upload --repository testpypi dist/*
```

To upload them to the real index (**verify first with testpypi**)

```
twine upload dist/*
```
## 8 Adding New Operations in the Library
This document outlines the process for integrating a new operation into the existing code library. The integration process involves several key steps: defining the operation’s interface, implementing the operation ensuring compatibility with the library’s architecture, and providing testing to validate the operation.
> 向代码库集成的过程包括几个关键步骤：
> 定义操作的接口
> 保证和库架构的兼容性，实现操作
> 提供测试，验证操作

An example of implementing new operations can be found here: [Implementing reduce operations](https://github.com/intel/intel-npu-acceleration-library/commit/4f17015a75c146fe8d569ac71a2e2a0a960fc652)
### Step 1: Defining the OpenVINO interface
The first step is defining the call to the OpenVino method of the new operation through the OpenVINO Runtime C++ API. This is done in the `nn_factory.h` header. In this file, a new operation is created by interfacing with the OpenVINO operation. This includes specifying input and output parameters, and data types of the operation’s interface and then calling and returning the OpenVINO method. The interface should align with the library’s existing design patterns and naming conventions.
> 首先通过 OpenVINO 运行时 C++ API，定义对新操作的 OpenVINO 方法的调用
> 在 `nn_factory.h` 文件中，通过接口 OpenVINO 操作来创建新的操作，这包括指定输入和输出参数、指定操作接口的数据类型，然后调用和返回 OpenVINO 方法
> 接口应该和库现存的设计模式和命名规范匹配

A simple example of defining a new operation:

```python
ov::op::Op* new_operation(ov::op::Op* input) {
    auto new_operation = std::make_shared<ov::opset1::NewOp>(input->output(0));
    operations.push_back(new_operation);
    return new_operation.get();
}
```
### Step 2: Defining the C++ bindings
The next step is defining the C++ binding in the `binding.cpp` source file. This is the method that will be called in Python. This method has the operation’s input node as a parameter and additional arguments of the operation are defined in the method.
> 然后在 `binding.cpp` 中定义 C++绑定，该方法在 Python 中被调用
> 该方法以 operation 的输入节点作为参数，operation 的额外参数也要在该方法中定义

An example of defining the binding:

```cpp
intel_npu_acceleration_library_DLL_API ov::op::Op* new_operation(intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* input) {
    return factory->new_operation(input);
}
```
### Step 3: Adding new operation to list of supported operation
The new operation is added to the list of supported NPU operations in the `ops.py` script. The information of the new operation that must be provided is:

- the operation name
- the number of inputs
- the optional parameters types

> 新的 operation 需要在 `ops.py` 文件中被加入到支持的 NPU operations 列表中
> 需要提供的新的 operation 的信息包括：名称、输入的数量、可选形参类型
### Step 4: Adding extra functionality to the operation’s function
Ctypes is used to interface between C++ and Python. (Documentation is found here: [Python Ctypes](https://docs.python.org/3/library/ctypes.html))

If there is additional logic that you may want to add to the function, this can be done by defining a Python function that calls the C++ method in the `factory.py` file. Otherwise, if you directly call the functions to C++, then you do not need to define a Python function.
> C++ 和 Python 之间使用 Ctypes 作为接口
> 如果有需要额外的添加到函数中的逻辑，可以在 `factory.py` 中定义调用 C++方法的 Python 函数，或者也可以直接调用 C++函数
### Step 5: Adding PyTorch wrapper for the new operation
Additionally, to define a wrapper to use PyTorch native functions, this can be implemented in the `functional.py` file. In this step, a function of the same name as the PyTorch equivalent is created, which is used instead of the PyTorch implementation of the operation. If there is additional logic that you may want to add to the function to interface with the new operation, it can also be added in this function.
> 如果要为 PyTorch 原生函数定义包装器，可以通过实现 `functional.py` 文件完成
> 我们需要创建一个和 PyTorch 对应函数相同名称的函数，这个函数的实现会被实际使用，而不是 PyTorch 的实现，我们还可以在该函数实现中为这个函数接口添加额外的逻辑

It is common for the new operation to have the same name as the PyTorch equivalent operation, however this is not always the case and to show which operation we are referring to, we refer to the newly implemented operation as `new_operation` and the PyTorch operation and `operation`.
> 新的 operation 和 PyTorch 的等价操作有相同的名称很常见，我们称新实现的操作为 `new_operation` ，称 PyTorch 操作为 `operation`

The basic structure of PyTorch wrapper for a PyTorch operation, referred to as `torch.operation`, which returns the output of the implemented `new_operation`:

```python
@implements(torch.operation)
def operation(x: Tensor) -> Tensor:
    """Return the output tensor of the operation.

    Args:
        x (Tensor): The input tensor.
    Returns:
        Tensor: Output tensor.
    """
    return generate_op(x, "new_operation")
```
### Step 6: Building the library
To update the library, run the command:

```
pip install .
```
### Step 7: Adding tests for the new operation
A test for the new operation can be added in the `test_op.py` script. The new operation should be compared with a reference to ensure correct implementation.
> 可以在 `test_op.py` 中添加对于新的 operation 的测试，该 operation 应该和一个 reference 进行比较，保证其实现的正确性 `

The following is a basic structure to use the new operation:

```python
X = torch.rand((16, 128)).to(torch.float16)  # defining the input tensor

model = NNFactory()
input = model.parameter(X.shape)             # creating the input node
_ = model.new_operation(input)               # _ = torch.operation(input) is equivalent if using the PyTorch wrapper
model.compile()
out = model.run(X.numpy())
```

Using pytest to run all of the tests in the file:

```
pytest <name of the file>
```

Using pytest to run a single test in the file:

```
pytest <name of the file>::<name of the test>
```