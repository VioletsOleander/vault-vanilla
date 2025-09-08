# Abstract
This paper introduces two extensions to the popular PyTorch machine learning framework, TorchDynamo and TorchInductor, which implement the torch.compile feature released in PyTorch 2. 
>  本文介绍了两个对 PyTorch ML 框架的拓展: TorchDynamo, TorchInductor
>  这两个拓展实现了 PyTorch 2 发布的 `torch.compile` 特性

TorchDynamo is a Python-level just-in-time (JIT) compiler that enables graph compilation in PyTorch programs without sacrificing the flexibility of Python. It achieves this by dynamically modifying Python bytecode before execution and extracting sequences of PyTorch operations into an FX graph, which is then JIT compiled using one of many extensible backends. TorchInductor is the default compiler backend for TorchDynamo, which translates PyTorch programs into OpenAI's Triton for GPUs and  ${C} + +$  for CPUs. 
>  TorchDynamo 是一个 Python level 的即时编译器，它能够在不牺牲 Python 灵活性的前提下，使 PyTorch 程序实现图编译
>  TorchDynamo 通过在执行前动态修改 Python 字节码，并将 PyTorch operations 序列提取为 FX 图，然后使用多种可拓展的后端对其进行即时编译，实现对 PyTorch 程序的图编译
>  TorchInductor 是 TorchDynamo 的默认编译后端，对于 GPU，它将 PyTorch 程序转换为 OpenAI Triton，对于 CPU，它将 PyTorch 程序转换为 C++

Results show that TorchDynamo is able to capture graphs more robustly than prior approaches while adding minimal overhead, and TorchInductor is able to provide a  $2.27\times$  inference and  $1.41\times$  training geometric mean speedup on an NVIDIA A100 GPU across  $180+$  real-world models, which outperforms six other compilers. 
>  实验结果表明，TorchDynamo 能够相对于之前其他方法，更健壮地捕获图，并且仅添加最小开销，而 TorchInductor 则能够提供 2.27 倍和 1.41 倍的几何平均加速，性能优于其他六种编译器

These extensions provide a new way to apply optimizations through compilers in eager mode frameworks like PyTorch.
>  这两个拓展为类似于 PyTorch 的即时模式框架中通过编译器的优化提供了新的方式

# 1 Introduction
Modern machine learning frameworks can be divided into eager mode frameworks, such as PyTorch [32] and JAX [8], and graph mode frameworks, such as TensorFlow [2], Caffe [25], Theano [5], and CNTK [37]. Eager mode frameworks use an imperative define-by-run [47] approach where a machine learning model is represented as code that is executed each time one wants to run the model. Graph mode frameworks take a more declarative define-and-run [47] approach, where they expose a graph building API that requires users to first construct a graph and then later execute that graph.
>  机器学习框架可以分为即时模式框架，例如 PyTorch, JAX，以及图模式框架，例如 TensorFlow, Caffe, Theano, CNTK
>  即时模式框架使用命令式的 define-by-run 方法，其中机器学习模型被表示为代码，在每次运行模型时执行该代码
>  图模式框架采用声明式的 define-before-run 方法，它们提供构建图的 API，要求用户先构建图，然后执行该图

Users of machine learning frameworks, and especially researchers, have shown an overwhelming preference for the eager programming model [22]. The eager mode model is easier to understand and can be debugged using standard tools such as print and pdb in Python [23]. This user preference towards eager mode has caused traditionally graph mode frameworks to switch to eager mode programming models [4].
>  即时模式的模型更易于理解，并且可以用标准工具进行调试 (如 Python 中的 print 和 pdb)
>  研究人员对即时模式的偏好促使传统上基于图模式的框架开始转向即时模式

The downside of eager mode frameworks is that they make it harder to apply graph-level optimizations through compilers. The framework only has visibility of a single operator at a time, and thus cannot automatically perform optimizations, like fusion or scheduling, that cross operator boundaries. To address this, there have been attempts to allow graph capture in PyTorch through record/replay [17, 34], Python parsing [17], and lazy evaluation [39]. 
>  即时模式框架的劣势是它们使得应用编译器执行图级别的优化更加困难
>  框架一次只能看到一个 operator，因此不能自动执行跨 operator 边界的优化，例如融合或调度 (融合即算子融合，调度即优化内存复用、计算顺序、并行执行等)
>  为了解决这个问题，人们尝试通过记录/重放，Python 解析和懒惰求值来在 PyTorch 中实现图编译

>  记录/重放是记录运行时的操作序列，然后重放生成图
>  Python 解析是静态分析代码，试图从中提取出计算图
>  惰性求值是延迟执行操作，先构建图，再批量执行

Unfortunately, these approaches have sacrificed much of the usability that draws users to PyTorch. Record/replay is unsound and can produce incorrect behavior [17]. Python parsing works for simple programs, but has not been able to replicate the complex semantics of all of Python, so results will show it fails on over half of real-world models. Lazy evaluation incurs high run-time overheads and adds latency to kernel launches. 
>  这些方法牺牲了 PyTorch 的易用性
>  记录/重放方法不安全，可能导致错误行为 (例如两次运行之间修改了变量，record/replay 会错误地重放旧行为，导致结果不对)
>  Python 解析适用于简单的程序，但无法复制 Python 的全部复杂语义，故在超过一半的真实模型中会失败
>  惰性求值会带来较高的运行时开销，并增加内核启动的延迟 (惰性求值需要维护一个延迟执行的 IR，每次操作不是立即执行，而是记录下来，这带来了额外的运行时开销，并且每个 kernel 都有启动延迟)

Additionally, an exclusively graph mode backend for PyTorch is intractable for some models. Due to the flexibility provided by PyTorch, many model authors take advantage of features that do not easily map to graphs, such as: dictionaries, lists, custom classes, third party libraries (numpy, logging, etc), disk/network, multiprocessing, exceptions, and handwritten kernels.
>  此外，如果强制要求所有 PyTorch 模型必须能在 graph 模式下运行，那是不可行的，因为很多模型使用了非图结构化的编程特性
>  许多模型作者都利用了这些难以映射到图的特性，例如字典、列表、自定义类、第三方库、磁盘/网络操作、多进程、异常以及手写的内核

This paper presents two open source extensions to PyTorch: TorchDynamo and TorchInductor. These extensions are behind the torch.compile feature introduced in PyTorch 2 and officially released in March 2023. TorchDynamo is a Python-level JIT compiler designed to allow graph compilation in PyTorch programs while retaining the full flexibility of Python. TorchDynamo hooks into the Python frame evaluation API [9] in CPython to dynamically modify Python bytecode right before it is executed. It rewrites Python bytecode in order to extract sequences of PyTorch operations into an FX graph [34] which is then just-in-time compiled with many extensible backends. It creates this FX graph through bytecode analysis and is designed to generate smaller graph fragments that can be mixed with Python execution to get the best of both worlds: usability and performance.
>  本文介绍两个 PyTorch 的拓展: TorchDynamo, TorchInductor
>  他们是 `torch.compile` 功能的核心，TorchDynamo 是一个 Python 级别的 JIT 编译器，旨在允许在 PyTorch 程序中进行图编译，同时保留 Python 全部的灵活性
>  TorchDynamo 通过 CPython 的帧评估 API 购入到 Python 执行过程中，在代码执行之前动态修改 Python 字节码，将 PyTorch 操作序列提取为 FX 图
>  FX 图会被多种可拓展的后端进行即时编译
>  TorchDynamo 通过字节码分析生成 FX 图，并且在设计上是生成可以和 Python 执行混合的较小的图片段，从而兼顾易用性和性能

TorchInductor is a new compiler backend for TorchDynamo. It translates PyTorch programs into OpenAI's Triton [46] for GPUs and  $\mathrm{C + + / OpenMP}$  [15] for CPUs. TorchInductor is able to support the flexibility and dynamism of PyTorch by using similar abstractions to PyTorch eager mode. It introduces a new define-by-run loop-level intermediate representation (IR) to make it easy to add new operator lowerings. Additionally, it is implemented in Python, so it is easy for PyTorch users to extend and modify to meet their needs.
>  TorchInductor 是 TorchDynamo 的新编译后端，对于 GPU，它将 PyTorch 程序转化为 OpenAI Triton，对于 CPU，转化为 C++/OpenMP
>  TorchInductor 通过使用与 PyTorch 即时模式相似的抽象来支持 PyTorch 的灵活性和动态性
>  它引入了一种新的 define-by-run loop-level IR，使得添加新的 operator lowering 更简单
>  此外，TorchInductor 使用 Pythohn 实现，因此用户可以对其进行拓展和修改

Experimental results show that TorchDynamo is able to capture graphs more robustly than prior approaches while adding minimal overhead. TorchDynamo is able to capture a single whole-program graph for most models and can gracefully fall back to partial graphs when needed. Measurements show TorchInductor produces faster code on average than six other PyTorch compiler backends. Performance comparisons include both training and inference, CPU and GPU, float32 and float16, and three large benchmark suites containing  $180+$  full-sized models taken from real-world applications.
>  实验结果表明，TorchDynamo 相较于之前的方法，在添加最小开销的同时，可以更健壮地捕获图
>  对于大多数模型，TorchDynamo 可以捕获一个完整的程序图，并且可以在需要的时候回退回部分图

# 2 Prior Attempts at PyTorch Graph Capture
Graph capture in PyTorch presents unique challenges when compared to graph mode frameworks [2, 25, 5, 37], where the user is restricted to only using constructs that are representable in the graph. With PyTorch and other eager mode frameworks, the user is free to embed arbitrary code, including non-PyTorch libraries, inside their models. This results in frequent conversion from PyTorch Tensors to Python types (via .item(), .tolist(), etc), usage of external libraries (numpy, logging, etc), and usage of Python constructs (classes, closures, exceptions, control flow, etc) that do not map well to a fixed graph abstraction. 
>  和图模式框架比较时，PyTorch 中的图捕获面临独特的挑战
>  在 PyTorch 这样的即时模式框架中，用于可以自由在模型中嵌入任意代码，包括非 PyTorch 库，这导致了频繁的: PyTorch 张量转化为 Python 类型 (通过 `.item(), .tolist()` 等)、对外部库的使用 (numpy, looging)、对难以映射到固定的图抽象的 Python 构造的使用 (class, closure, execption, control flow)

Due to this mismatch between the flexibility provided by Python/PyTorch, and the inflexibility of graph representations, prior attempts at graph capture in PyTorch have needed to place restrictions on the user experience. While this tension between flexibility and representation is solved by TorchDynamo, we examine prior art in the space to provide context and background.
>  由于这个 Python/PyTorch 提供的灵活性和图表示的僵化性之间的不匹配，之前在 PyTorch 的图捕获尝试不得不对用户体验施加了限制
>  尽管 TorchDynamo 解决了灵活性和表示能力之间的矛盾，但我们仍然会研究该领域的先前工作，提供背景和上下文

## 2.1 `torch.jit.trace`
`torch.jit.trace` uses record/replay with example inputs to produce a TorchScript [17] graph. The recording is done at the PyTorch dispatcher level, which is inside the  $\mathrm{C + + }$  portion of PyTorch and used to dispatch operators to device-specific kernels and for autograd. Because the recording is done in  $\mathrm{C + + }$  , torch.jit.trace does not capture any control flow in Python. Consider this example:
>  `torch.jit.trace` 使用带有示例输入的记录/重放来生成 TorchScript 图
>  记录是在 PyTorch 的分派器级别进行的，该级别位于 PyTorch 的 C++部分，用于将 operators 分发为设备特定的 kernels 以及自动求导
>  因为记录是在 C++ 中进行的，因此 `torch.jit.trace` 无法捕获任何 Python 中的控制流

>  `torch.jit.trace` 是一种将 Python 函数或 `nn.Module` 转换为 TorchScript 的方法，它会提供一个示例输入，运行该函数，并记录在这个输入下实际执行的所有 PyTorch 操作，生成一个静态的计算图
>  这个过程发生在 PyTorch 的 C++ 分派器中，也就是 PyTorch 内部调度算子到具体设备 kernel 的地方

```python
def example1(x):
    if len(torch.nonzero(x)) > 1:
        return x + 1
    return x - 1
```

With example input ` torch.tensor([0, 0])`, torch.jit.trace would capture a graph equivalent to:

```python
def example1_incorrect_capture(x):
    torch.nonzero(x)
    return x - 1
```

>  在示例输入 `torch.tensor([0, 0])` 下：
>  1. `torch.nonzero(torch.tensor([0, 0]))` → `tensor([])`  (空张量)
>  2. `len(...)` → `0`
>  3. 条件 `0 > 1` 为 `False`，所以走 `return x - 1`
>  因此，`torch.jit.trace` 只记录了：
>  - `torch.nonzero(x)`  (因为它被调用了)
>  - `x - 1` (实际返回的操作)
> 但它没有记录 `if` 判断本身，也没有记录 `x + 1`，因为那条分支没执行
> 所以生成的 TorchScript 图等价于：

```python
def example1_incorrect_capture(x):
    torch.nonzero(x) # 只是副作用，实际图中可能被优化掉
    return x - 1 # 固化为这条路径
```

Since the path through the program is specialized on the example input, a different input (such as torch.tensor([1, 1])) will give incorrect results. Additionally, any non-PyTorch operators (such as external libraries, prints, logging, side effects, etc.) will be omitted from the captured graph.
>  由于程序的执行路径是针对示例输入进行专门化的，因此不同的输入将导致错误的结果
>  此外，任何非 PyTorch 操作 (例如外部库、print、logging、副作用) 都将被排除在捕获的图之外

## 2.2 `torch.jit.script`
`torch.jit.script` also constructs a TorchScript [17] graph, but does so by parsing the Python AST and performing static analysis. It is able to capture example1 above correctly and, unlike torch.jit.trace, it is a sound approach that should not produce incorrect results.
>  `torch.jit.script` 也会构建一个 TorchScript 图，但是通过解析 Python AST 并执行静态分析来实现的
>  `torch.jit.script` 可以正确捕获 example1 的图，并且和 `torch.jit.trace` 不同，它是一种可靠的方法，不应该产生错误的结果

The major challenge torch.jit.script faces is that it is trying to reimplement all of Python as a static language. This approach is all or nothing: encountering an unimplemented component of Python makes the entire program unfit for capture. Emulating all of Python statically is a daunting task and, in practice, torch.jit.script only supports a subset of Python. Experimental results show that torch.jit.script works only about half the time on real-world models in the TorchBench benchmark suite, and anecdotally we have heard stories of it taking weeks or months to "torchscript" large models, which leads to a frustrating user experience.
>  `torch.jit.script` 的主要挑战是它试图将 Python 重新实现为一种静态语言
>  这种方法是 "all or nothing": 一旦遇到未实现的 Python 成分，整个程序就无法被捕获
>  静态地模拟整个 Python 是一个困难的任务，在实践中，`torch.jit.script` 仅支持 Python 的一个子集
>  在 TorchBench 上，`torch.jit.script` 只有一半的时间可以正常工作

## 2.3 Lazy Tensors
Lazy Tensors were introduced in the PyTorch/XLA [42, 39] project, which is primarily focused on supporting Google TPUs [26] with PyTorch. Lazy Tensors is a  $\mathrm{C + + }$  level graph capture technology. Every iteration, it defers execution of operations to accumulate a graph and then sends the accumulated graph to the XLA [45] compiler. By hashing this graph, Lazy Tensors can avoid recompilation when the captured graph is identical across iterations. While this approach is effective and sound, it has a few major downsides:
>  Lazy Tensors 在 PyTorch/XLA 项目中引入，该项目主要专注于为 PyTorch 添加 Google TPU 支持
>  Lazy Tensor 是一种 C++ 级别的图捕获技术，在每次迭代，它延迟操作的执行，以累积出一张图，然后将这个图发送给 XLA 编译器
>  通过哈希该图，Layze Tensors 可以避免捕获的图在不同迭代时相同时对图进行重编译
>  这种方法有效且可靠，但也有一些缺点:

- Higher overheads: Lazy Tensors incurs additional work when compared to PyTorch eager. Besides running the same Python code and PyTorch dispatcher stack that eager does, it must maintain additional graph data structures that incur added runtime costs. 
>  更高的开销: 相对于 PyTorch eager 模式，Lazy Tensors 会带来额外的计算负担，因为它除了运行和 eager 模式相同的 Python 代码和 PyTorch dispatcher stack 以外，它必须维护额外的图数据结构，这会增加运行时成本

- Introduced delays: PyTorch eager issues the first kernel on the first operation of the model, after which point host-side code is run in parallel with kernels on the GPU or accelerator thus hiding overheads. In contrast, Lazy Tensors doesn't issue the first kernel until the model's code has finished executing, resulting in added delays before the first kernel is issued and after any operation that requires a round trip to the CPU (which are common in real-world models). Thus, Lazy Tensors often serializes host execution with GPU/accelerator utilization, which amplifies host side overheads. Models, loss logging, and optimizers need to be modified to work around this issue. 
>  引入的延迟: PyTorch eager 模式在模型的第一个操作后立即发出第一个 kernel，之后主机端代码可以和 GPU kernel 并行运行，从而隐藏开销 (例如 Python 解释器、内存拷贝等)
>  相较之下，Lazy Tensors 要等到模型代码执行完毕之后才发出第一个 kernel，
>  这样引入的延迟不仅出现在开始阶段，也出现在任何需要从 GPU 把数据拉回 CPu 的操作中，例如计算 loss 并打印、使用 `item()` 获取标量值、梯度裁剪和学习率调整等 optimizer 操作
>  每当发生这种 GPU -> CPU -> GPU 的往返，Lazy Tensor 必须把当前延迟的所有操作编译成图，在 GPU 上执行图，将结果传回 CPU，等待 CPU 执行完再继续后续操作
>  这使得 GPU 和主机的执行串行化，放大了主机端的开销，为了缓解这个问题，开发者往往需要修改模型代码，例如避免频繁调用 `.item()` 获取 loss 值、将 loss 记录和日志打印等操作延迟到多个 step 后执行等

>  eager 模式即代码逐行执行，每个操作都会立即调用 GPU 上的 kernel 运行
>  Lazy Tensor 则不会立即执行操作，而是先记录计算图，等到必要时候，例如需要结果返回 CPU，才真正编译并执行整个计算图

- Recompilation: Whenever the captured graph has a new hash, Lazy Tensors must recompile. This can lead to some pathological cases where recompilation happens frequently.
>  重编译: 捕获的图具有新的哈希时，Lazy Tensors 就需要重编译，这可能导致极端情况下的频繁重新编译

The PyTorch/XLA project has built [17] an integration with TorchDynamo which uses a hybrid of both Lazy Tensors and TorchDynamo. This integration hides the overheads of Lazy Tensors by only running Lazy Tensors once, rather than every iteration, and using TorchDynamo to figure out when recapture is needed. The PyTorch/XLA results later in the paper use this integration.
>  PyTorch/XLA 项目已经构建了一个与 TorchDynamo 的集成，该集成结合了 Lazy Tensors 和 TorchDynamo
>  该集成通过只运行一次 Lazy Tensors (而不是每次迭代都运行)，并使用 TorchDynamo 来判断何时需要重新捕获来隐藏开销

## 2.4 `torch.fx.symbolic_trace`
`torch.fx.symbolic_trace` [34] is the newest of these systems and introduced the FX graph format that is shared by TorchDynamo. It takes a similar record/replay-based approach to torch.jit.trace, but does its tracing at the Python level as opposed to at the PyTorch  $\mathrm{C + + }$  dispatcher level. 
>  `torch.fx.symbolic_trace` 是这类系统中最新的一个，它引入了 FX 图格式 (TorchDynamo 也使用 FX 图格式)
>  它采用与 `torch.jit.trace` 类似的基于记录/重放的方式，但在 Python 级别进行记录，而不是在 PyTorch 的 C++ dispatcher 级别

>  在 C++ dispatcher 级别记录的是实际的 tensor 操作，`symbolic_trace` 则是在 Python 层面进行记录，通过拦截 Python 函数调用和操作来构建图

It runs the user code using a Proxy Python object to record its behavior and uses the torch_function [3] extension point in PyTorch.
>  它使用一个 Proxy Python 对象来记录其行为，并利用 PyTorch 中的 `torch_function` 拓展点

>  `symbolic_trace` 不使用真实的张量 (如 `torch.Tensor`)，而是使用一个叫 Proxy 的特殊对象来代替输入张量
>  当模型运行时，所有对 Proxy 的操作 (`x + y, x.size(), F.relu(x)`) 都会被记录下来，而不是真正执行数值计算
>  这个机制依赖于 PyTorch 的 `__torch_function__` 协议，这是一个允许用户拦截和重写所有 PyTorch 操作的钩子

By recording higher up at the Python level, symbolic_trace is able to capture many operations that torch.jit.trace cannot. Since it records using Proxy objects instead of real tensors, it is able to detect many cases where torch.jit.trace would be incorrect, e.g., when trying to read sizes or values from Proxy tensors or when using them in control flow, such as example1 above. 
>  由于在 Python 层面记录，`symbolic_trace` 相对于 `torch.jit.trace` 可以捕获更复杂的控制流和动态行为
>  因为 `symbolic_trace` 使用 Proxy 对象而不是真实张量，它不会真正计算数值，所以当代码试图访问 `size(), .item()` 时，`symbolic_trace` 可以知道这个操作，并把他们记录为图的一部分

It also suffers from the all-or-nothing limitation of many solutions described above. For example, in the control flow case above, the user is still forced to rewrite the code they want to trace.
>  尽管 `symbolic_trace` 很强大，但仍然存在 “all-or-nothing” 的问题，用户可能需要修改代码为支持 `symbolic_trace` 导出的形式 (比如用 `torch.cond` 代替 Python 的条件语句)

Unfortunately, torch.fx.symbolic_trace is still unsound and can produce incorrect results. Consider this example, which increments a global variable and calls a function not dependent on the function input:
>  `torch.fx.symbolic_trace` 仍然可能生成不正确的图，图的行为和源代码的行为不同

```python
def example3(x):
    global call_count
    call_count += 1
    return torch.rand(10) + x
```

If one runs torch.fx.symbolic_trace on this example it produces a graph equivalent to:

```python
def example3_incorrect_capture(x):
    return _tensor_constant() + x
```

The call to torch.rand got removed and the result of it got burned into the graph as a fixed constant. Subsequent uses of the graph will not get new randomness, but instead reuse whatever value was generated during capture. This type of incorrect capture can be difficult to debug and may go unnoticed by users.
>  例如上述例子中，`torch.rand` 会被移除，其结果会被记录为图中的一个常量

The call_count operations are completely lost because they did not interact with the Proxy object x. Instead, the call_count got incremented to 1 during tracing and will not be incremented when the graph is called. This is also an example of something that is not supported by any of the graph representations. Nearly all graphs formats for machine learning have no concept of a Python global, so even if this could be captured, it is not supported by downstream backend compilers.

>  例如上述例子中， `call_count` 会被移除，因为没有和 Proxy 对象 `x` 交互
>  这也是任何图表示也不会支持的语义，几乎所有的 ML 图格式都没有 Python 全局变量这个概念，因此即使可以捕获到这个行为，下游的后端编译器也不会支持它

## 2.5 `torch.onnx.export`
ONNX [31] export is not actually a graph capture mechanism, but some people confuse it for one, so we include it here for completeness. Internally, ONNX export uses torch.jit.trace and torch.jit.script (Section 2.1 and 2.2), so it faces all the same limitations imposed by those systems. Additionally, the conversion from TorchScript to the ONNX format can fail due to ONNX not supporting all PyTorch operators. Thus, the set of models supported by ONNX is a subset of those supported by TorchScript.
>  ONNX 导出实际上并不是一种图捕获机制
>  在内部，ONNX 导出使用了 `torch.jit.trace` 和 `torch.jit.script`，因此它也面临这些系统所施加的限制
>  此外，由于 ONNX 不支持所有的 PyTorch 算子，故从 TorchScript 转化到 ONNX 格式可能会失败，因此 ONNX 支持的模型集是 TorchScript 的一个子集

The ONNX team is working on an integration with TorchDynamo that will replace TorchScript with a direct TorchDynamo integration. Once finished, this will increase the number of models ONNX works on.
>  ONNX 团队正在与 TorchDynamo 进行集成，也就是用和 TorchDynamo 的直接集成取代 TorchScript，这可以增加 ONNX 可以处理的模型数量

## 2.6 Comparison To Graph Capture In JAX
JAX [8] largely doesn't face the same challenges being solved by TorchDynamo. The initial design of JAX was heavily coupled to the design of XLA [45], and JAX has been backed by XLA from its inception. This has the effect of forcing JAX programs to conform to the constraints built into the design coming up the stack from XLA. Thus, JAX uses a simpler capture mechanism, and expects users to write their programs to meet the constraints of that capture mechanism. As an example, jax.jit **does not support data-dependent Python control flow and requires user code to be functionally pure.**
>  JAX 在很大程度上并不面临 TorchDynamo 所解决的相同挑战
>  JAX 的最初设计与 XLA 紧密耦合，且 JAX 在一开始就以 XLA 作为后端，这促使 JAX 程序遵循 XLA 中的约束
>  因此，JAX 使用了更简单的图捕获机制，并期望用户编写满足其捕获机制约束的程序
>  例如 `jax.jit` 不支持依赖数据的 Python 控制流，并要求用户代码是纯函数式的

In contrast, PyTorch started as an eager-only framework without any compiler-minded constraints built into its design. A large corpus of models has grown on top of PyTorch, most of which were written without any regard to how hard to capture and compile they would be.
>  相较之下，PyTorch 最初就是一个即时模式的框架，其设计中没有内置任何编译相关的约束
>  大量模型在 PyTorch 上构建起来，其中大多数在编写的时候并未考虑它们在编译和捕获方面的难度

On an implementation level, the capture mechanism in JAX is similar to torch.fx.symbolic_trace (Section 2.4), but somewhat simpler, because JAX programs are purely functional and thus do not need to worry about state. The Torch FX paper [34] contains a more detailed comparison with JAX.
>  在实现层面，JAX 的捕获机制类似于 `torch.fx.symbolic_trace`，但更简单，因为 JAX 程序是纯函数式的，因此无需担心状态问题

# 3 TorchDynamo Design and Implementation
TorchDynamo takes a fundamentally different approach from prior graph capture systems in PyTorch. Rather than trying to remove or replace Python, TorchDynamo tries to work with CPython by just-in-time (JIT) compiling Python bytecode. 
>  TorchDynamo 与 PyTorch 之前的图捕获系统采用了根本不同的方法，它并不是试图移除或代替 Python，而是尝试通过即时编译字节码来和 CPython 协作

TorchDynamo is a Python bytecode to Python bytecode translator, where it extracts PyTorch operations from the original bytecode and replaces them with calls to compiled artifacts that fuse many PyTorch operations together. Figure 1 provides an overview of how TorchDynamo operates and will be explained in the remainder of this section.
>  TorchDynamo 是一个 Python 字节码到 Python 字节码的翻译器，它从原始字节码中提取 PyTorch 操作，将它们替换为对已编译的组件的调用，这些组件会将多个 PyTorch 操作融合在一起

![[pics/PyTorch2-Fig1.png]]

## 3.1 Usage API
The primary API introduced in this paper is `torch.compile`. It can be used either by calling it on a PyTorch Module or as a function decorator. It has the following keyword options:
>  本文介绍的主要 API 是 `torch.compile`，它可以在 PyTorch Module 上调用，或者作为一个函数装饰器，它具有以下关键字参数

- **backend**: allows the user to provide a custom compile function which takes a torch.fx.Graph and a list of example inputs and returns a Python callable. This defaults to TorchInductor, but can also be set to one of many builtin backends or a user-defined backend. 
- **options**: An optional dictionary of backend-specific configuration flags. 
- **mode**: Shorthand strings for a predefined set of options: "default","reduce-overhead", or "max-autotune".

>  - backend: 用户提供一个自定义的编译函数，接收 `torch.fx.Graph` 以及一个示例输入列表，返回一个 Python 可调用对象，默认的编译函数为 TorchInductor，但也可以是其他的内置后端或者用户定义的后端
>  - options: 一个可选的字典，包含特定于后端的配置标志
>  - mode: 预定义选项集合的简写字符串: `default, reduce-overhead, max-autotune`

When you run a module with torch.compile, the module is executed with the modified CPython behavior shown in Figure 1. Specifically, a custom CPython frame evaluation hook will rewrite the bytecode of of each Python function being executed in order to extract and compile sequences of PyTorch operations. This bytecode rewriting process is cached, but the analysis relies on certain dynamic properties of the program that we use guards to check on subsequent calls.
>  使用 `torch.compile` 运行一个 module 时，该 module 会以 Figure 1 所示的修改后的 CPython 行为执行
>  具体地说，一个自定义的 CPython 帧评估钩子将重写正在执行的每个 Python 函数的字节码，以提取并编译 PyTorch 操作序列
>  这个字节码重写过程会被缓存，但分析依赖于程序的特定动态特性，我们在后续调用中使用 guard 来检查这些特性

## 3.2 CPython Frame Evaluation Hook
PEP 523 [9] introduced the frame evaluation API into the CPython interpreter. A frame is the data structure in CPython used to represent a function call. This is the main extension point used by TorchDynamo, and it was designed to facilitate just in time (JIT) compilers and debuggers in Python. PEP 523 added an eval_frame function pointer to PyInterpreterState, which allows overriding the core function used to interpret a single function call in CPython. Whenever CPython calls a function, it first creates a PyFrameObject, then it calls this user defined eval_frame hook. By default, eval_frame points to `_PyEval_EvalFrameDefault`, which contains the main interpreter loop for CPython. TorchDynamo modifies eval_frame to replace this standard CPython interpreter loop with one that performs JIT compilation of Python frames.
>  PEP 523 为 CPython 解释器引入了帧评估 API，**帧是 CPython 中用于表示函数调用的数据结构**
>  帧评估 API 是 TorchDynamo 使用的主要拓展点，这个 API 的设计原意是促进 Python 的 JIT 编译器和 debugger
>  PEP 523 向 `PyInterpreterState` 中添加了一个 ` eval_fram ` 函数指针，允许覆盖 CPython 中用于解释单个函数调用的核心函数
>  当 CPython 调用一个函数时，它首先创建一个 `PyFrameObject`，然后调用 `eval_frame` 指向的函数，默认情况下，它指向 `_PyEval_EvalFrameDefault`，该函数包含了 CPython 的主解释器循环
>  TorchDynamo 修改了 `eval_frame`，用一个执行 Python 帧即时编译的循环替换了原有的标准 CPython 解释器循环

>  也就是 TorchDynamo 通接入帧评估 API，改变了对 Python 函数调用的解释方式 (CPython 默认是调用 `_PyEval_EvalFrameDefault`)

The custom eval frame function installed by TorchDynamo performs the following operations:
>  TorchDynamo 执行的自定义 `eval_frame` 函数执行以下的操作:

- Check if the frame should be skipped due to filename exclusion, previous failures in analysis (which mark the frame to be skipped), or exceeded cache size limits. Filename exclusions are used for common libraries, like Python standard libraries and numpy, which will not contain PyTorch operations. For skipped files, call `_PyEval_EvalFrameDefault` on the original bytecode and return. 
>  检查是否该帧应该被跳过，原因可能包括文件名排除、之前的分析失败 (这些失败会标记该帧应该跳过)、或超过了缓存大小限制
>  文件名排除用于常见的库，例如 Python 标准库和 numpy，这些库中不会包含 PyTorch 操作
>  对于跳过的文件，直接对原始的字节码调用 `_PyEval_EvalFrameDefault` 并返回

- Check if the frame has previously been compiled and is cached; if so, execute the generated guard function (Section 3.3) for each entry in the cache. If a guard function returns True, run the matching cached compiled bytecode with `_PyEval_EvalFrameDefault` and return. 
>  检查该帧是否之前被编译并被缓存
>  如果是，对缓存中的每个条目执行生成的 guard 函数，如果 guard 返回 True，就使用 `_PyEval_EvalFrameDefault` 运行缓存的编译好的字节码并返回

- Perform symbolic analysis (instruction by instruction) of the function bytecode to extract an FX graph [34], guards, and side effects. This analysis can stop partway through the function if it encounters an unsupported operation. 
>  对函数的字节码进行符号分析 (逐条指令) 以提取 FX 图, guards 和 side effects
>  如果分析过程中遇到不支持的操作，分析可以提前终止

- Compile the FX graph with a user-defined compiler function specified by the `backend=` argument provided to `torch.compile`. 
> 使用用户通过 `torch.compile` 的 `backend` 参数指定的编译函数来编译提取出的 FX 图

- Generate and compile a single Python function that checks all of the guards. It returns True if the guards pass and the existing compiled artifact can be reused. 
>  生成并编译一个 Python 函数，用于检查所有的 guards 条件
>  如果 guards 条件通过，并且可以复用已有的编译结果，则该函数返回 True

- If the analysis did not reach the end of the function, generate resume_at_xx continuation functions. Continuation functions run the remainder of the function in a new frame and are described in Section 3.8. 
>  如果分析没有达到函数末尾，将生成 `resume_at_xx` 的继续执行函数，继续执行函数会在一个新的帧中运行函数的剩余部分

- Generate new Python bytecode. This new bytecode will: 1) call the compiled FX graph; 2) store and reconstruct the local/stack state; 3) perform side effects the original function should have had, see Section 3.7; 4) either return or implement a graph break by falling back to the original bytecode and calling the generated continuation function(s). 
>  生成新的 Python 字节码，这个新的字节码将 1. 调用已编译的 FX 图 2. 存储和重构局部/栈状态 3. 执行原始函数应产生的副作用 4. 或者返回结果，或者通过回退到原始字节码并调用生成的继续函数来实现图断点

- Install the generated Python bytecode and guard function in the cache, run the generated bytecode with `_PyEval_EvalFrameDefault`, and return.
>  将生成的字节码和 guard 函数存储缓存，使用 `_PyEval_EvalFrameDefault` 执行生成的字节码，然后返回

## 3.3 Guards
Guards are the mechanism TorchDynamo uses to recheck dynamic properties used by JIT compilation to determine is a cached compilation can be reused. TorchDynamo generates a guard function for each transformed PyCodeObject that returns True if it is safe to reuse a compiled artifact. Both the guards and the transformed code are stored using the `_PyCode_SetExtra` extension point introduced in PEP 523 [9].
>  Guards 是 TorchDynamo 用于重新检查 JIT 编译过程中使用的动态属性的机制，以决定是否可以重用缓存的编译结果
>  TorchDynamo 会为每个经过转换的 `PyCodeObject` 生成一个 guard 函数，当可以安全复用已编译的结果时，该函数返回 True
>  guard 函数和转换后的代码都通过 PEP 523 引入的 `_PyCode_SetExtar` 拓展点存储

Guards are accumulated during analysis and can point to variables originating from globals/locals or nested within python data structures. At the time of writing there were 30 different types of guards. Guards include: checking many torch.Tensor properties, Python types, constant specialization, attributes, dicts/lists/tuples, nn.Module instances, and global PyTorch state. The guard system spans across TorchDynamo, AOTAutograd, and TorchInductor. Any layer can introduce guards to protect specializations. Guards are all independent checks and do not interact with each other beyond deduplication.
>  在分析过程中会累积多个 guards，它们可以指向来自于全局/局部的变量，或者嵌套在 Python 数据结构中的变量
>  guards 包括: 检查许多 `torch.Tensor` 的属性、Python 类型、常量特化、属性、字典/列表/元组、`nn.Module` 实例、全局 Python 状态
>  guard 系统贯穿于 TorchDynamo, ATOAutograd 和 TorchInductor 中，任意一层都可以引入 guards 来保护特定的优化
>  所有的 guards 都是独立的检查项，彼此之间不会相互影响

>  guard 是 PyTorch 自动优化系统中保护代码特殊化 (specialization) 安全性的检查机制
>  当 PyTorch 尝试编译或优化一段代码时，它会基于当前的输入条件做一次假设，然后生成已高效版本的代码
>  但这个高效版本仅对当前情况有效，如果后续运行时环境变了 (例如传入了不同类型的张量)，这个优化版本就可能出错
>  因此 guards 就是监控这些变化的哨兵，它们确保一旦运行时状态发生变化，就触发重新编译，guards 包含了对各种各样东西的检查，具体看上面的列举

## 3.4 Symbolic Evaluation
A fundamental part of TorchDynamo is the symbolic Python bytecode evaluator which is responsible for analyzing Python bytecode and modeling effects of each instruction. Symbolic evaluation contains data structures that keep track of: 1) stack state; 2) local variables; 3) exception contexts; 4) accumulated FX graph [34]; 5) accumulated guards; and 6) side effects. The algorithm operates one Python bytecode at a time and contains a function corresponding to every Python bytecode instruction type.
>  TorchDynamo 的一个基本部分是符号化 Python 字节码求值器，它负责分析 Python 字节码并建模每条指令的效果
>  符号化求值包含用于跟踪以下数据结构的结构: 1. 栈状态 2. 局部变量 3. 异常上下文 4. 累积的 FX 图 5. 累积的 guards 6. 副作用
>  该算法一次处理一条 Python 字节码，并且每个 Python 字节码指令类型都有一个对应的函数

>  符号化即不使用具体的数值，而是使用抽象的符号来表示变量

At the start of symbolic evaluation, function arguments are examined and converted to a symbolic representation, VariableTracker. If bytecodes access data structures such as class attributes or global variables, new symbolic representations for these constructs are added lazily. This representation is discussed more in Section 3.5. The symbolic evaluator starts at the first bytecode instruction of the function, and continues processing the function one bytecode at a time. The soundness of this analysis can be shown via induction: as long each individual bytecode is processed correctly, the overall algorithm will be correct.
>  符号化求值开始时，函数参数会被检查并被转化为符号表示，即 `VariableTracker`
>  如果字节码访问了诸如类属性或全局变量这样的数据结构，则会惰性地为这些结构添加新的符号表示
>  符号求值器从函数的第一个字节码指令开始，然后字节码地处理该函数
>  这种分析的正确性可以通过归纳法证明: 只要每条单独的字节码都被正确处理，整个算法就是正确的

As an example, suppose the first instruction was LOAD_FAST, a Python bytecode that pushes a local variable on to the stack. The handler for LOAD_FAST will take the representation variable from the symbolic local variables and push it on to the symbolic stack data structure. The handler for BINARY_ADD, will pop two symbolic variables off the stack then push their result on to the stack. The result is computed depending on the types of those variables and the dispatch will vary based on those types. If the value represents a PyTorch tensor, then a new add node will be added to the FX graph [34], and a new symbolic tensor pointing to the result node will be created.
>  例如，假设第一条指令是 `LOAD_FAST`，这是将局部变量推送到栈上的 Python 字节码，`LOAD_FAST` 的处理函数会从符号化局部变量中获取对应的表示变量，并将其推送到符号化栈数据结构中
>  对 `BINARY_ADD` 的处理函数会从栈中弹出两个符号变量，然后将它们的运算结果推入栈上
>  结果的计算取决于变量的类型，分发方式也会根据类型而变化，如果该值表示 PyTorch 张量，则会为 FX 图添加新的加法节点，并创建新的符号张量指向该节点

>  PyTorch 中，用户写的模型代码通常是 Python 函数，例如

```python
def forward(x):
    return x @ w + b
```

>  这些函数每次运行都有不同输入，普通的 JIT 编译器难以直接优化这种动态语言
>  TorchDynamo 使用符号化分析，即不真正执行代码，而是模拟执行过程，使用符号代替真实数据
>  TorchDynamo 的符号化求值器维护六个数据结构来追踪程序状态:
>  - stack: 模拟 Python 解释器的运算栈
>  - local variables: 记录 `a = 5, y = x + 1` 等局部变量的符号表示
>  - execption context: 跟踪 `try-catch` 块，确保控制流正确
>  - 累积的 FX 图
>  - 累积的 guards
>  - side effects: 是否修改了全局状态 (如 print, write to file)
>  TorchDynamo 开始分析函数时，会先将函数参数转化为叫 `VariableTracker` 的符号表示
>  逐条处理字节码时，处理对象也是符号变量
>  例如 `a = x + 1` 的 `LOAD_FAST a` 会查找局部变量中名为 `a` 的符号，将它的 `VariableTracker` 压入符号栈
>  例如 `x + 1` 的 `BINARY_ADD` 会判断两个变量的类型，如果是普通数字，可能进行常量折叠，如果是 PyTorch 张量，就在图中添加节点 (`aten::add`)，并创建一个新的 `VariableTracker` 表示该加法的结果，后续基于该结果的操作也能记录下来，形成完整的计算链
>  如果每个字节码指令都能完美处理，那 TorchDynamo 的处理就完美反映了 Python 程序本来的语义，函数分析结果就是正确的

>  一个完整的例子

```python
def f(x):
    y = x + 1
    z = y * 2
    return z
```

>  TorchDynamo 的符号化求值过程如下：

| 字节码               | 处理动作                                                       |
| :---------------- | :--------------------------------------------------------- |
| `LOAD_FAST x`     | 把`x`的符号（`VariableTracker(tensor)`）压入栈                      |
| `LOAD_CONST 1`    | 把常数`1`的符号压入栈                                               |
| `BINARY_ADD`      | 弹出`x`和`1`，发现都是 tensor → 在 FX 图中添加`aten::add`节点，生成新符号`y`并压栈 |
| `LOAD_CONST 2`    | 把`2`压栈                                                     |
| `BINARY_MULTIPLY` | 弹出`y`和`2`→ 添加`aten::mul`节点，生成`z`                           |
| `RETURN_VALUE`    | 返回`z`，FX 图完成                                               |

>  最终得到一个完整的计算图：`x → add(1) → mul(2) → return`

## 3.5 Modeling Python Data Structures
Many semantics of Python are in libraries and data structures, so any Python analysis must model the behavior of these different types. To analyze the behavior of each variable or stack entry, TorchDynamo has a class hierarchy that models common behaviors of different data types. Each of these data structures is a subclass of VariableTracker. Notable types of variable trackers include:
>  Python 的许多语义都包含在库和数据结构中，故任何 Python 分析都必须对这些不同类型的行为进行建模
>  为了分析每个变量或栈项的行为，TorchDynamo 有一个类层次结构，用于对不同数据类型的常见行为进行建模，这些数据结构都被表示为 `VariableTracker` 的子类，一些值得注意的数据类型包括:

- *TensorVariable* represents a torch.Tensor. It does not store an underlying tensor value, but instead stores a fx.Proxy which points into the partially constructed FX graph [34] as well as a "fake" tensor (see Section 5) that represents the metadata of a tensor without its actual data.
>  `TensorVariable` 表示 `torch.Tensor`，它不存储张量值，而是存储一个指向部分构建的 FX 图的 `fx.Proxy`，以及一个 “假” 张量，表示 tensor 的元数据而不包含实际的数据

- *ConstDictVariable* and *DataClassVariable* are used to represent key/value pairs where the keys are constant strings and the values can be anything, including nested dicts/lists.
>  `ConstDictVariable, DataClassVariable` 用于表示键值对，其中 keys 为常量字符串，values 可以是任意内容，包括嵌套的字典/列表

- *ListVariable* and *TupleVariable* represent list/tuple and can contain any other type of symbolic variable.
>  `ListVaraible, TupleVariable` 表示列表和元组，可以包含任何其他类型的符号变量

- *UserFunctionVariable* and *UserMethodVariable* represent user defined functions that can be inlined. They also support functions constructed dynamically containing closures.
>  `UserFunctionVariable, UserMethodVariable` 表示可以内联的用户定义函数，它们还支持包含闭包的动态构造的函数

- *UserDefinedClassVariable* represent user-defined classes and *UserDefinedObjectVariable* represents instances. We lazily specialize on these as their attributes are accessed, and track mutation on them (Section 3.7).
>  `UserDefinedClassVariable` 表示用户定义的类，`UserDefinedObjectVariable` 表示其实例
>  我们在访问其属性时会惰性地进行特化，并跟踪它们的修改

There are many other variable tracker types that represent other situations. In addition to type-specific data, every VariableTracker instance also contains a set of guards, which are initialized when they are created and propagated through operations via union. Additionally, each instance also tracks where it came from so that it can be loaded or mutated in output bytecode.
>  还有表示其他情况的许多其他 `VariableTracker` 类型
>  每个 `VariableTracker` 实例除了包含特定于类型的数据，还包含了一组 guards，这些 guards 在实例被创建时被初始化，并通过操作之间的并集进行传播 (两个 `VariableTracker` 合并时，如 `a+b`，它们的 guards 会做并集，也就是只有所有条件满足才允许使用该图)
>  此外，每个实例还会追踪它来自哪里，以便在输出字节码中加载或修改它 (也就是每个 `VariableTraker` 需要记住它是由哪个表达式产生的，它属于哪个局部变量名，它是否后续被修改过，才能在后续生成的字节码被正确 `LOAD_FAST, STORE_FAST`)

## 3.6 Inlining, Control Flow, and Closures
Function calls can either happen directly from user code, or implicitly through magic methods such as `__getitem__`. To collect bigger graphs, TorchDynamo will attempt to inline function calls and flatten programs. When a function call is encountered, TorchDynamo first creates a checkpoint of the current symbolic state. Next, it recursively tries to symbolically evaluate the called functions, passing in any input symbolic state and recording any changes that are made. If this recursive analysis hits a case that would cause a graph break (Section 3.8) or other errors, TorchDynamo rolls back to the symbolic state before the function call and generates a graph break on that function call. Otherwise, the recursive analysis returns and the analysis of the parent function continues.
>  函数调用可以直接来自于用户代码，也可以隐式通过魔法方法例如 `__getitem__`
>  为了收集更大的图，TorchDynamo 会尝试内联函数调用并展开程序
>  当遇到一个函数调用时，TorchDynamo 首先会创建当前符号状态的检查点，然后，它会递归地尝试对被调用的函数进行符号求值，传入任意的输入符号状态，并记录所作的任何更改
>  如果递归分析遇到了会导致 graph break 或其他错误的情况，TorchDynamo 会回滚到函数调用之前的符号状态，并在该函数调用处生成一个 graph break
>  否则，递归分析完成后，父函数的分析将继续进行

Most cases of control flow in Python bytecode are optimized away and handled through specialization. For example, when iterating over a list of torch.nn.Module, TorchDynamo will guard that the list doesn't change and unroll the loop. For control flow based on the type, size, and shape of tensors, TorchDynamo will guard on those properties and remove the control flow. In less common cases where there is control flow that cannot be removed (for example, branching on the value of a tensor rather than the metadata), TorchDynamo will generate a graph break that will trigger the branch bytecode to run in CPython, and analysis will resume after the jump.
>  Python 字节码中的大多数控制流情况都会被优化掉，并被专门化
>  例如，在遍历 `torch.nn.Module` 列表时，TorchDynamo 会检查该列表是否发生变化，并展开遍历循环
>  对于基于张量类型、大小、形状的控制流，TorchDYnamo 会针对这些属性进行检查，并移除相应的控制流
>  在一些比较少见的情况，如果存在无法移除的控制流 (例如，根据张量的值而不是元数据进行分支)，TorchDynamo 会生成 graph break, graph break 会触发该分支的字节码在 CPython 中执行，分析在跳转之后继续进行

Another challenge is closures. Consider this example:

```python
def closure_example(x):
    y = torch.sigmoid(x)
    return lambda z: y + z
```

Here the variable  $y$  is in a closure which is represented by what CPython calls a cell, which adds a layer of indirection to allow variables in closures to be mutated. 
>  另一个挑战是闭包
>  在上述示例中，变量 `y` 是一个闭包，CPython 使用称为 cell 的结构表示闭包
>  cell 增加了一层间接引用，允许对闭包中的变量进行修改

>  闭包: 函数可以捕获外部作用域的变量，即便外部函数已经返回，本质是一个函数 + 它所依赖的外部变量 (闭包捕获的变量)
>  闭包的三要素是:
>  1. 外部函数定义变量
>  2. 内部函数引用该变量
>  3. 内部函数被返回或传递出去

>  Python 中，内部函数要访问外部变量，不能直接拿到值，而需要通过 cell 数据结构
>  而要访问 cell 数据结构，需要通过 `LOAD_DEREF, STORE_DEFER` 字节码
>  这些都是运行时机制，编译器很难直接优化

>  示例中的 `lambda z: y + z` 就是一个闭包，它记住了 `y` 的值，即使 `closure_example()` 返回了，这个 `y` 仍然可以通过返回的函数访问
>  在 CPython 中，这种捕获的变量不是直接存储的，而是通过叫 cell 的中间结构实现的
>  当一个变量被闭包捕获后，它会被包装为一个 `cell` 对象，cell 中的值通过 `LOAD_DEREF, STORE_DEREF` 读写

>  TorchDynamo 相对函数进行 JIT 编译优化，例如将 `x + 1` 优化为算子融合、提前计算常量表达式、生成高效 CUDA 代码等
>  但由于闭包引入了不可见的间接层: cell，静态分析就十分困难

There are a number of different cases of closures that TorchDynamo must handle:
>  TorchDynamo 需要针对不同类型的闭包情况采取不同的策略

- Cell variables created outside the captured region must be accessed differently than other variables. If they are accessed from the top-level function, they can be accessed by generating the LOAD_DEREF and STORE_DEREF bytecodes. When inlining, this bytecode cannot be used and instead TorchDynamo generates code to read-/write directly from the inlined function cell, for example `fn.__closure__[0].cell_contents`. If the content of a cell is mutated, TorchDynamo tracks the mutation in the same way as other mutations (Section 3.7). 
>  在捕获区域之外创造的 cell 变量需要于不同于其他变量的方式进行访问
>  如果它们是从顶层函数中访问的，可以通过生成 `LOAD_DEREF, STORE_DEFER` 字节码访问
>  但在内联时，这些字节码无法使用，TorchDynamo 会生成代码，直接从内联的函数 cell 中读写，例如 `fn.__closure__[0].cell_contents`
>  如果 cell 内容被修改，TorchDynamo 会以追踪其他变更的方式追踪这个变更

>  第一种情况是变量在外部创建，在内部被使用，例如

```python
def outer():
    x = 5
    def inner():
        return x
    return inner
```

>  `x` 是在 `outer` 中定义的，被 `inner` 捕获
>  如果在 `inner` 内部访问 `x`，CPython 会生成 `LOAD_DEREF` 字节码

```python
# 伪代码（TorchDynamo 生成的优化代码）
def inner():
    cell_x = fn.__closure__[0]      
    value = cell_x.cell_contents    
    return value + 1
```

>  而如果 `inner` 被内联，就不能使用 `LOAD_DEFER` (这是在运行时才能生效的指令)，而是直接访问 cell 的内容，例如 `fn.__closure__[0].cell_contents`，其中 `__closure__` 是函数对象的一个属性，保存所有被捕获的 cell，`cell_contents` 是 cell 中存储的值

>  也就是绕开了运行时机制，在编译时就访问捕获的变量
>  和下面的情况的区别是，这里 `x` 可能在运行时被改变，因此需要访问获取具体的值，下面的情况是直接固定为常数

- Cell variables both created and destroyed within the captured region are the easiest to handle and most common. In this case, TorchDynamo statically optimizes away the closure. 
>  完全在被捕获区域内被创建以及摧毁的 cell 变量最常见，也最容易解决，在这种情况下，TorchDynamo 会静态地优化掉闭包

>  例如

```python
def make_adder(n):
    def adder(x):
        return x + n
    return adder
```

>  TorchDynamo 会直接把 `adder` 函数内的 `n` 提取出来，变成常量或参数
>  优化后相当于

```python
def adder(x, n = 5):
    return x + n
```

>  完全消除了闭包结构，无需 cell

- Cell variables that are created in the captured region, but escape the frame are the most difficult to handle. In this case, TorchDynamo will optimize away all uses of the closure inside the captured region. Then, at the very end in the generated bytecode, it will create any needed cells and Python function objects to return. From the outside, callers will not be able to tell that the returned closure was created differently than the original program.
>  在捕获区域内被创建但逃出了当前帧的 cell 变量最难处理
>  此时，TorchDynamo 会优化掉捕获区域内部对闭包的所有使用，然后，在生成的字节码的最后阶段，它会创建所需的 cells 和 Python 函数对象以返回
>  从外部来看，调用者无法察觉返回的闭包和原始程序的闭包有什么不同

## 3.7 Mutation and Side Effects
Python functions sometimes have side effects. TorchDynamo handles side effects by deferring them until after the FX graph [34] has been called, then generating output bytecode that applies all side effects at the end. To do this, TorchDynamo has a side effects data structure that tracks all side effects that the original code would have. If the code tries to read a value that would have been mutated by a pending side effect, it instead reads that pending value. 
>  有时 Python 函数存在 side effect, TorchDynamo 将推迟到 FX 图被调用之后，再生成在最后应用所有 side effects 的输出字节码
>  为此，TorchDynamo 使用一个 side effect 数据结构来追踪原始代码中所有的 side effects
>  如果代码尝试读取一个会被待处理的 side effect 修改的值，它会改为读取这个待处理的值

After the graph is generated, a garbage collection pass removes side effects that didn't escape the analysis context, and TorchDynamo generates output code to apply the needed side effects. Handling side effects this way results in multiple writes to the same value being collapsed into a single write. 
>  在图生成之后，一个 GC pass 会移除没有逃出分析上下文的 side effects (不会对外部产生影响的 side effects)，然后 TorchDynamo 会生成输出代码来应用剩余的，有必要应用的 side effects
>  用这种方式处理 side effects 可以将对同一值的多次写入合并为一次写入

TorchDynamo supports the following types of side effects:

- Writes to global variables result in a STORE_GLOBAL bytecode if the target global is in the same file. If it is in a different file (because of inlining), code is generated to mutate the global in the other module.
- Writes to attributes (such as on classes) are handled similarly and mapped to STORE_ATTR in output bytecodes. We use the source on the VariableTracker to determine how to load a reference to the object that must be mutated.
- Writes to cells/closures are tracked and handled in a number of ways (see Section 3.6). 
- Class construction is handled by creating a placeholder symbolic object, inlining the `__init__` method, and tracking all the attribute mutation on that placeholder object. If the object is live at the end of the function, the output bytecode will create the object (bypassing the constructor) and set the needed attributes.
- Dictionary and list mutation can also cause side effect if the dict/list was passed in as an input or loaded from a global/attribute. The VariableTracker representations of dict/lists will guard on the initial symbolic state of these objects, then symbolically track all changes through the entire function. The captured FX graph [34] will have all of these operations optimized away. In the output bytecode, a new dict/list will be created to match the final state and the original list object will be mutated to match that object. This recreation is not needed for lists/dicts that do not escape the captured region because their mutations cannot be observed, and therefore they can be completely removed.

>  TorchDynamo 支持的 side effects 类型有:
>  - 对全局变量的写入: 如果目标全局变量在同一个文件中，则生成 `STORE_GLOBAL` 字节码，如果在不同文件中 (由于内联)，则生成代码来修改其他模块中的全局变量
>  - 对属性的写入 (例如类上的属性): 处理方式类似，生成 `STORE_ATTR` 字节码，我们使用 `VariableTracker` 上的源信息来确定如何加载对要修改的对象的引用
>  - 对 cell/closures 的写入: 前文介绍过
>  - 类的构造: 通过创建一个占位符符号对象，内联其 `__init__` 方法 (把 `__init__` 的内容当作函数体处理，用符号追踪所有属性修改)，并跟踪该占位符对象上的所有属性修改实现，如果该对象在函数结束后仍然有效，输出字节码将创建该对象 (绕过构造函数)，并设置所需的属性
>  - 字典和列表变更: 只有当字典/列表是作为输入传入，或者是从全局变量/属性中加载的，其变更才视作 side effect (本地创建，并变更的不视作 side effect)；字典/列表的 `VariableTracker` 表示会针对这些对象的初始符号状态进行 guard，然后在整个函数中符号化追踪对它们的修改；FX 图中会优化掉这些修改操作；在输出的字节码中，会创建一个新的字典/列表，匹配原始的字典/列表对象被修改后的最终状态，且原始的字典/列表也会被修改为与该对象一致；对于不会逃逸出捕获区域的字典/列表，不需要重新创建，因为它们的修改无法被观察到，因此可以被完全移除

## 3.8 Graph Breaks and Continuation Functions
When TorchDynamo encounters a Python bytecode it cannot handle, for example a call to an external library, it generates what we call a graph break to split the bytecode being analyzed into multiple pieces. Essentially, TorchDynamo will mix compiled fragments into the original Python code to get a hybrid execution. Any pending partial FX graph [34] is compiled. In the output code when the partial graph will be called, the unsupported bytecode will be executed, and then we will recursively use TorchDynamo to analyze the remainder of the function. 
>  当 TorchDynamo 遇到无法处理的 Python 字节码时，例如对外部库的调用，它会生成 graph break，将正在分析的字节码分为多个部分
>  本质上，TorchDynamo 会将编译好的代码片段混合到原始的 Python 代码中，以实现混合执行
>  任何待处理的 FX 图都会被编译，在输出代码中，当调用该部分图时，不支持的字节码将被执行，然后我们会递归使用 TorchDynamo 分析函数的剩余部分

>  TorchDynamo 遇到无法分析的字节码，会:
>  1. 停止当前 FX 图构建
>  2. 把前面已生成的部分编译成可执行代码
>  3. 在该处插入一个图断点，让原生 Python 执行无法处理的部分
>  4. 继续分析后面的代码

To trigger this recursive analysis, TorchDynamo generates one or more continuation functions which take the form:

```python
def resume_at_X(... livevars ...):
    ... restore try/except/stack state ...
    JUMP_ABSOLUTE X
    ... original functional bytecode ...
```

This continuation function looks very similar to the original function except for a few changes: 1) the arguments are changed to reflect whatever variables are live across the graph break; 2) a prefix is added to restore the stack/exception state, which may also be passed in as an argument; 3) a JUMP_ABSOLUTE instruction is created so execution resumes in the middle of the function.

>  为了支持断点后分析，TorchDynamo 会生成连续函数，形式如上
>  连续函数非常类似于原始的函数，一些差异在于: 1. 参数被修改，以反映哪些变量在跨越 graph break 之后还活跃 2. 添加一个前缀，用于恢复 stack/exception 状态 3. `JUMP_ABSOLUTE` 指令，使得执行从函数中间继续

TorchDynamo will either generate one of these functions, or two of these functions in the case of control flow (all control flow bytecodes have exactly two branches), to continue execution right after the unsupported bytecode. The advantage of structuring continuations as Python functions is that it will recursively trigger TorchDynamo through the frame evaluation API. When TorchDynamo processes a continuation function, it treats it exactly the same as any other Python function.
>  TorchDynamo 要么会生成一个这样的函数，或者在存在控制流时生成两个这样的函数 (所有的控制流字节码正好有两个分支)，以在不受支持的字节码之后继续执行
>  将延续结构化为 Python 函数的优势在于它可以通过 frame evaluation API 递归地触发 TorchDynamo
>  当 TorchDynamo 处理一个延续函数时，它会将其视为与其他任何 Python 函数完全相同

## 3.9 AOTAutograd
AOTAutograd is a reusable component in PyTorch that is called by many PyTorch compiler backends to add training support and use shared operator decompositions. TorchDynamo captures the forwards of a model, but, to support training, we also need to generate the backwards pass. In Pytorch eager, the backwards graph is generated dynamically using a tape-based autograd [32]. AOTAutograd turns the forwards graph into a forwards and backwards graph in a way that supports partial program graphs. 
>  AOTAutograd 是 PyTorch 的一个可重用组件，它被许多 PyTorch 编译器后端调用，以添加训练支持，并使用共享的算子分解
>  TorchDynamo 会捕获模型的前向计算，但为了支持训练我们还需要生成反向过程
>  PyTorch 即时模式下，反向图是通过基于 tape 的自动求导动态生成的，AOTAutograd 则以一种支持部分程序图的方式，将前向图转换为前向和反向图

AOTAutograd works by running the PyTorch eager mode autograd engine on fake tensor inputs and recording a joint forwards and backwards graph. Data-dependent operations do not work with fake tensors (since there is no backing data), so we graph break on these operations in TorchDynamo and run them outside the graph. 
>  AOTAutograd 的原理是在 fake tensor inputs 上运行 PyTorch eager mode 的自动微分引擎，然后记录一个联合的前向和反向图
>  依赖于数据的操作无法使用 fake tensor (因为没有实际的数据支持)，故我们在 TorchDynamo 中对这些操作 graph break，在图外执行它们

AOTAutograd then uses a min-cut algorithm [55] to split this joint graph into separate forward and backward graphs in a way that optimizes for memory usage. As part of this min-cut algorithm, we apply backend-specific optimizations to rematerialize certain activations that are cheap to recompute in the backwards graph.
>  然后 AOTAutograd 会使用一个最小割算法，以优化内存使用的方式，将联合图划分为独立的前向和反向图
>  作为该最小割算法的一部分，我们会应用针对于后端的优化，来丢弃一些便于重计算的中间激活

As part of AOTAutograd, other dispatcher-level transformations are also applied to the graph. Decompositions are where AOTAutograd maps some PyTorch operators into a smaller set of more primitive operators. AOTAutograd also makes the graph purely functional by removing operations that perform mutation and replacing them with their functional equivalents.
>  AOTAutograd 还会对图应用其他 dispatcher-level 的转换
>  AOTAutograd 会将一些 PyTorch 算子映射到更小的一组 primitive operators，这个过程称为 Decompositions
>  AOTAutograd 还会通过移除执行了变更的操作，将它们替换为其函数式的等价形式，使得整个图是纯函数式的

# 4 TorchInductor Design and Implementation
While TorchDynamo solves the graph capture problem in PyTorch, to be useful it must be paired with a backend compiler that can take the captured FX graph [34] and generate fast code from it. We created TorchInductor as a reference compiler backend. It is designed to be general purpose and can be used both directly by users and as a starting point for other backends.
>  TorchInductor 是 TorchDynamo 的参考编译器后端，接收 FX 图，生成代码
>  TorchInductor 在设计上是通用的，既可以直接使用，也可以作为其他后端的起始点

## 4.1 Design Principles and Key Technologies
Before diving into the design of TorchInductor, let's first discuss some principles and technologies that motivated its design:
>  TorchInductor 的一些设计原则如下:

**PyTorch Native**: PyTorch made many design choices that differ from other frameworks and compilers: Tensors have exposed strides that can be manipulated by users, aliasing views are commonplace, and both data and metadata can be mutated in-place. Any compiler with a dramatically different model will face many challenges in representing PyTorch programs. We wanted TorchInductor to share similar abstractions to PyTorch eager to allow support of all of PyTorch, with a thin translation layer.
>  原生 PyTorch: PyTorch 在许多设计选择上与其他框架和编译器不同，张量具有可以被用户操作的步长，别名视图很常见，数据和元数据都可以原地修改
>  任何采用不同的模型的编译器都难以表示 PyTorch 程序
>  在设计上 TorchInductor 和 PyTorch eager 共享了类似的抽象，以支持对所有 PyTorch 功能的支持

**Python First**: The majority of PyTorch users are most comfortable in Python. The Python parts of PyTorch get far more community contribution than the  $\mathrm{C + + }$  parts of PyTorch. We chose to implement TorchInductor in Python to make it easy to understand and hackable by PyTorch users.
>  Python 优先: PyTorch 的大多数用户更熟悉 Python, PyTorch 的 Python 部分的社区贡献远远多于 C++ 部分
>  我们用 Python 实现 TorchInductor 以便用户修改和自定义

**Breadth First**: Rather than focusing on a narrow set of models (e.g. ResNet/BERT) that are already well studied, we intentionally put an early focus on supporting a wide variety of operators, hardware, and optimization. This helped make TorchInductor a general purpose compiler that can scale to many scenarios. This is also why the early focus was on training, since training is a much harder compiler problem than inference.
>  宽度优先: 不是聚焦于特定的一组模型，而是聚焦于支持更广的算子、硬件和优化
>  即 TorchInductor 应该是通用目的的编译器，因此早期的焦点也在训练，因为训练是比推理更难的编译器问题

**Reuse State-Of-The-Art Languages**: For an output language, we took inspiration from how PyTorch users were writing high performance kernels. We observed rapidly increasing popularity of the OpenAI Triton [46] DSL for writing GPU kernels, and those kernels are often outperforming other compilers and state-of-the-art libraries. High performance CPU kernels are typically written in  $\mathrm{C + + }$  /OpenMP [15]. TorchInductor generates both Triton and  $\mathrm{C + + }$  as output code, which allows us to leverage the technology of those projects as well as generate output code that is understandable by PyTorch users.
>  复用 SOTA 的语言: 将 Triton 和 C++ 作为输出代码

## 4.2 Decompositions
Rather than implementing lowerings for all operators in PyTorch to TorchInductor's IR, many operators in PyTorch are decomposed into a simpler set of operators that are easier to handle. These decompositions happen using AOTAutograd (Section 3.9), which is called by TorchInductor with a dictionary of desired decompositions. 
>  TorchInductor 没有实现所有 PyTorch 算子到 TorchInductor IR 的下降
>  许多 PyTorch 算子会被分解为更简单的一组算子，这个分解通过 AOTAutograd 实现，TorchInductor 会调用 AOTAutograd ，传入一个包含了期望分解的字典

Decompositions are written as a Python implementation of a PyTorch operator in terms of other operators, for example the following decomposes log2 into log and mul:

```python
log2_scale = 1 / math.log(2)

@register_decomposition(torch.ops.aten.log2)
def log2(x):
    return torch.log(x) * log2_scale
```

>  分解的形式是在 Python 层以其他算子实现 PyTorch 算子
>  上例展示了 `log2` 算子的分解形式 (分解为 `log , mul`)

This decomposition will be recursively traced down and normalized, and can possibly trigger additional decompositions in that process until a fixed point is reached. Note that the active decomposition set must not contain cycles.
>  算子的分解会被递归地追踪并被规范化，并可能在过程中触发更多的分解，直到达到一个固定点
>  注意当前激活的分解集合不能包含环 (不能出现相互依赖的情况，否则会无限循环)

At the time of writing, TorchInductor used 191 decompositions (387 including overloads). The majority of these decompositions are not specific to TorchInductor and are available for any other backend to use via the `torch._decomp` module, while some are TorchInductor specific.

## 4.3 Lowerings and Define-By-Run Loop-Level IR
The next phase of compilation is lowering from an FX graph of PyTorch operations into TorchInductor's define-by-run IR. A define-by-run IR means the IR uses executable Python code to define the bodies of loops, giving TorchInductor's IR much of the power of full Python, removing the need for a large amount of boilerplate, and allowing lowerings to be written concisely.
>  编译的下一个阶段是从一个包含 PyTorch 操作的 FX 图下降到 TorchInductor 的 define-by-run IR
>  define-by-run IR 意味着 IR 使用可执行的 Python 代码来定义循环体，这使得 TorchInductor IR 具备了大部分 Python 的功能

Lowering is done by symbolically interpreting the FX graph and applying lowering functions which do the conversion for a single operator. At the time of writing, TorchInductor has lowerings for 433 PyTorch operators (1605 including overloads). If an unknown operator is encountered, it is automatically converted into a fallback kernel node which runs the original PyTorch code.
>  下降是通过符号上解释 FX 图，并应用对单个算子执行转换的下降函数是实现的
>  如果遇到了未知算子，它会被自动转化为 fallback kernel node，以原始 PyTorch 代码运行

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-07/9ed94e03-5712-4e4d-babd-00dd15e3db8a/b6440b8729a1147702f46fec9fabd5585ae5df4d41f419cdd1fa5060b3baf5b7.jpg)  

Figure 2. TorchInductor IR for `torch.log2` on a 2D tensor.

In the example IR shown in Figure 2, inner_fn_buf0 is a Python function that defines how to compute a single element of the tensor buf0 in terms of calls to TorchInductor's primitive operators in the ops.* namespace. The function takes a list of SymPy [28] symbols (i0 and i1) representing the symbolic coordinates of the element to be computed. SymPy symbols s0 and s1 represent the sizes of the tensor to be computed and are used for both sizes and strides. These size symbols are captured in a Python closure and registered on the graph object.
>  Fig2 中，`inner_fn_buf0` 是定义了如何计算 tensor `buf0` 中单个元素的 Python 函数
>  该函数中都是对 `ops.*` 命名空间下的 TorchInductor 的 primitive 算子的调用
>  函数接收一个 SymPy 符号 (i0, i1) 的列表，它们表示需要计算的元素的符号坐标
>  SymPy 符号 s0, s1 表示要计算的张量大小，并用于表示大小和步长
>  这些大小符号会被捕获到一个 Python 闭包，并注册在图对象上

TensorBox and StorageBox are abstractions that match PyTorch torch.Tensor and torch.Storage objects and allow the handling of views, aliasing, and mutation during the lowering process. ComputedBuffer represents a tensor that will be computed using generated code (in contrast to ones created via fallback kernels or inputs). Pointwise represents that the ComputedBuffer is a data parallel pointwise computation. The IR also supports Reduction and Scatter for handling other types of operators.
>  `TensorBox, StorageBox` 是匹配 `torch.Tensor, torch.Storage` 对象的抽象，允许处理视图、别名和下降过程中的变更
>  `ComputedBuffer` 表示一个会使用生成的代码计算的张量 (而不是同故宫 fallback kernels 的张量或者输入张量)
>  `Pointwise` 表示 `ComputedBuffer` 是一个数据并行的点对点计算

The key advantage of this IR is that it is easy to construct because it has the full power of Python. One can compose different IR nodes together and embed logic within them. The example above would not be initially constructed as a single flat function, but rather many smaller function closures defined in the lowering process. The function created for ops.mul will call into another function created for ops.log, which calls into another function created for loading the input argument.
>  这个 IR 的关键优势是容易构造，因为它具有 Python 的全部表示能力，我们可以将不同的 IR 节点组合在一起，并在其中嵌入逻辑
>  上述示例不会一开始就构造为单一的扁平函数，而是由多个较小的函数闭包组成，这些函数是在下降过程中逐步定义和生成的
>  为 `ops.mul` 创建的函数会调用为 `ops.log` 创建的函数，进而调用为加载输入参数调用的函数

The way to compile and analyze this IR rests in the virtualized namespace for `ops.*`, which can be dynamically overridden to perform different functions. To perform analysis on this IR, we make ops point to an analysis pass which can perform actions like record memory accesses or high/low watermarks for strength reduction optimizations. To perform codegen with this IR, we make ops point to something which writes out Triton or C++ code. To transform this IR, we make use of FX tracing, which gives access to graph representation for these Python functions.
>  编译和分析这个 IR 的方式定义在虚拟化命名空间 `ops.*`，这个命名空间可以被动态重载以执行不同的函数
>  要对 IR 执行分析，我们让 `ops` 指向一个分析 pass，该 pass 可以执行例如记录内存访问或高低水位线 (用于优化强度缩减，即用更简单的复杂替代更复杂的运算，例如移位替代乘法)
>  为了对这个 IR 执行代码生成，我们让 `ops` 指向能够输出 Triton 或 C++ 代码的组件
>  为了转化该 IR，我们利用 FX tracing，以访问这些 Python 函数的图表示

At the time of writing, the loop-level TorchInductor IR consisted of 54 primitive operators:

- ops.load and ops.store access. Tensor memory from a provided buffer name and a SymPy index specifying a symbolic memory location.
- ops.reduction operates like ops.store where the reduction happens implicitly inside the write. It combines stored values along the reduction dimension of the current node using a supplied reduction type. Supported reduction types are: argmin, argmax, any, max, min, prod, sum, xor_sum, and wellfold Combine [50].
- ops.index_expr converts from SymPy expressions used for indexing into values used for compute.
- ops.indirect_indexing converts from computed values into SymPy expressions used for indexing by introducing a new SymPy variable bound dynamically.
- ops.masked implements conditional execution. It takes a condition and a Python function (recursively using the same IR) with no args. This gets mapped to masks in Triton and conditionals in C++.
- ops.load_seed, ops.rand, ops.randan, and ops.randaint64 are used for computing random numbers.
- The remaining ops are elementwise math operations.

>  目前这一循环级别的 TorchInductor IR 包含 54 个 primitive operators:
>  - `ops.load, ops.store`: 通过一个提供的 buffer name 访问 Tensor memory，通过一个 SymPy index 指定符号的内存位置
>  - `ops.reduction` : 类似于 `ops.store`，但会在写入时隐式执行规约
>  - `ops.index_expr`: 将用于索引的 SymPy 表达式转化为用于计算的值
>  - `ops.masked`: 实现了条件执行，接收一个条件和无参数的 Python 函数 (递归地使用相同的 IR)，该 op 在 Triton 中映射为掩码，在 C++ 中映射为条件语句
>  - `ops.load_seed, ops.rand, ops.randan, ops.randaint64`: 计算随机数
>  - 剩余的 `ops` 为逐元素的数学运算

## 4.4 Scheduling
The scheduling phase of TorchInductor determines which operators get fused, what order kernels run in, and does memory planning for buffer removal and/or reuse. Scheduling starts by converting every buffer in the IR into a subclass of BaseSchedulerNode. SchedulerNode represents a standard kernel that TorchInductor will codegen the body of. ExternKernelSchedulerNode represents calls to library code or user-defined kernels. Additionally, NepKernelSchedulerNode maps to nothing, but is used to add dependency edges to ensure the ordering of kernels (for example, a concatenate kernel which has been handled by making producers write directly to the combined buffer). Finally, a FusedSchedulerNode represents a set of two or more SchedulerNodes fused into a single kernel.
>  TorchInductor 的调度阶段决定哪些 operators 需要融合、kernels 的运行顺序、并执行内存规划 (缓存收集和重用)
>  调度阶段会先将 IR 中的每个 buffer 转化为 `BaseSchedulerNode` 的一个子类
>  `SchedulerNode` 表示一个标准的 kernel，其函数体就是 TorchInductor codegen 的结果
>  `ExternalKernelSchedulerNode` 表示对库代码的调用或者用户定义的 kernel
>  `NepKernelSchedulerNode` 映射到空，但会被用于添加依赖边以确保 kernel 的执行顺序  (例如在 concatenate kernel 中，生产者直接写入合并后的缓冲区，此时需要通过该节点来保证顺序)
>  `FusedSchedulerNode` 表示融合为单个 kernel 的 `SchedulerNodes` 

Next, the scheduler converts the memory read/write sets of each kernel into dependency edges between nodes. Dependency edges are annotated with the symbolic memory address being read. Symbolic memory addresses are important in determining which fusions are legal. For example, if one kernel writes buf0 in forwards order, but a consumer reads in reverse order (using `ops.load("buf0", s0 -1 -i0)`), then those nodes cannot be fused.
>  然后，调度器将每个 kernel 的内存读写集合转化为节点之间的依赖边
>  依赖边使用被读取的符号内存地址标记
>  符号内存地址用于决定哪些融合是合法的，例如，如果某 kernel 在前向写入了 `buf0`，但消费者

Fusion is controlled by two key functions:
- Scheduler.can_fuse(node1, node2) returns True if two nodes can be fused together. This checks dependency edges, and also checks many other properties to ensure correctness of a fusion. There are some heuristics here as well, for example, if config.aggressive_fusion=False, then can_fuse will prevent fusion of nodes that do not share any common memory accesses. There is also backend specific logic here, for example, TorchInductor supports reduction-broadcast reduction fusions for Triton but not  $C + +$ .
- Scheduler.score_fusion(node1, node2) is used to order different fusion possibilities. Some fusions are mutually exclusive, so TorchInductor chooses the one with the higher score. The fusion score orders fusions by: 1) the category of the fusion (e.g. pointwise/reduction/template); 2) estimated bytes of memory traffic saved by the fusion; and 3) shorter distance between nodes in the original graph

In a loop, until no additional fusions remain (since some fusions can open additional fusion opportunities), TorchInductor will perform the following greedy algorithm: 1) find all fusion opportunities; 2) score each of the fusion opportunities and sort by that score; 3) for each fusion opportunity, check if that fusion remains legal and if so apply it. When two nodes are fused, any pending fusion opportunities pointing to the constituent nodes are updated to point to the new fused node.

## 4.5 Triton Code Generation
Triton codegen is responsible for mapping TorchInductor's IR to output Triton [46] kernels. Figure 3 shows the code generated for the 1og2 example above. This kernel operates on a block of xBLOCK elements at a time. If the number of elements is not a multiple of xBLOCK, some elements may be masked off at the end. During codegen, we simplify indexing. For example, the 2D strided load in the IR is converted to a contiguous load in this case. Codegen is also responsible for common subexpression elimination (CSE), which is done via a cache while printing lines of code and assigning intermediate variable names starting with tmp. The pointwise decorator encodes boilerplate code used to facilitate block size heuristics, auto-tuning, and ahead-of-time kernel compilation. The decorator is the type of kernel being generated (pointwise, reduction, or template), and its arguments are required metadata about the kernel like data alignments.

When generating reduction kernels, TorchInductor has two modes of codegen. For smaller reductions, it will generate a persistent reduction where the entire reduction is loaded in a single block and retained in registers/shared memory; in this case reductions map directly to Triton reduction operators. For larger reductions, TorchInductor generates a loop using an entire block as an accumulator with a call to a Triton reduction at the end of the loop.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-07/9ed94e03-5712-4e4d-babd-00dd15e3db8a/b9885467063a6a070c291a0e5381fe557dae8a7128cf3967180e91d95ac81f7d.jpg)  
Figure 3. Generated Triton code for Figure 2.

For more complex operations (matmuls and convolutions), TorchInductor has its own template system for generating Triton code that mixes handwritten Triton with generated Triton. Templates are written using Jinja [29] with helper methods to interact with TorchInductor's codegen system.

# 4.6  $\mathbf{C} + +$  Code Generation

4.6  $\mathbf{C} + +$  Code GenerationFor the CPU backend, TorchInductor generates  $\mathrm{C + + }$  with OpenMP [15]. Within the  $\mathrm{C + + }$  backend there are two variants, a vectorized variant and a non-vectorized variant. The vectorized variant performs tiling and maps most operations to the at::vec::vectorized class included in the PyTorch source code. This class operates on 16 elements at a time, which is the same way standard PyTorch kernels are vectorized and supports multiple SIMD instruction sets. The non-vectorized variant generates relatively standard  $\mathrm{C + + }$  code using many  $\mathrm{C + + }$  standard template library [24] (STL) functions. Both of these variants are parallelized using #pragma omp for annotations, with some heuristics to decide how many levels of loops to parallelize. Reductions are mapped to the OpenMP reduction annotation if the reduction dimension loop is parallelized, and a  $\mathrm{C + + }$  loop with accumulator otherwise.

# 4.7 Wrapper Codegen

Wrapper codegen is responsible for generating the code that calls the kernels from Triton,  $\mathrm{C + + }$  and external sources. It also does tensor size calculations and handles memory allocation and deallocation. There are two different wrapper codegen implementations, one that generates Python code, and another that generates  $\mathrm{C + + }$  code. The Python backend is more flexible and supports some corner cases that the  $\mathrm{C + + }$  one does not, while the  $\mathrm{C + + }$  one is lower overhead.

When enabled with mode="reduce-overhead", TorchInductor uses CUDA Graphs [20] to completely eliminate the overhead from wrapper code. CUDA Graphs records and replays kernel launches at the CUDA driver level and is lower overhead than even the  $\mathrm{C + + }$  wrapper code. To ensure soundness, CUDA Graphs is only used when safety requirements are

met and is automatically disabled in some cases (for example with dynamic shapes, non-CUDA tensors, etc).

# 4.8 Related Deep Learning Compilers

There is lots of exciting work in the deep learning compiler space. Since most PyTorch users use GPUs, our main reason for selecting Triton [46] as an output target was its proven ability to generate kernels faster than handwritten libraries [30, 13, 43] with simple input code. Very few compilers have been able to do that consistently, and many widely used deep learning compilers simply call those libraries directly without trying to compete in GPU codegen for complex kernels.

Many compilers use designs inspired by Halide [33], including: TVM [11], nvFuser [36], and NNC [60]. These designs have a split semantics language and scheduling language that allow exploring different schedules without changing the semantics of the program. Researchers have explored many different ways of expressing the search space [18, 38, 48, 51, 58, 7, 59, 19] and searching that space automatically [57, 12, 54, 56, 6].

XLA [45] is the compiler behind TensorFlow [1] and JAX [8]. XLA provides multiple levels of IR including a high level IR, HLO, that has become a standard for TPUs [26] and similar accelerators. Many newer compilers are emerging in the MLIR [27] ecosystem, including IREE [44] (now part of OpenXLA [45]). The latest version of Triton [46] also uses MLIR for its internal representation.

# 5 Dynamic Shapes

Deep learning compilers commonly only work for static shapes, that is to say, they produce compiled programs which only work for a single specific configuration of input shapes, and must recompile if any input shape changes. This assumption works well for the majority of commonly run deep learning models today, but there are a few situations where it is insufficient:

Some dimensions, such as batch size or sequence length, may vary. For example, an inference service performing adaptive batching will execute inference requests with varying batch sizes depending on how many requests it received within its batching window. We may also want to consider padding out variable-size sequences only to the maximum sequence length within a batch, which may vary from batch to batch. Some models exhibit data-dependent output shapes, that is to say, the size of their outputs and intermediates may depend on the actual input data which may vary across runs. For example, detection models may first generate a variable number of potential bounding boxes before running a more expensive image recognition model to identify if the subject is in a bounding box. The number of bounding boxes is data-dependent.

One particularly important case of data-dependent shapes occurs when dealing with sparse representations, such as sparse tensors, jagged tensors, and graph neural networks. In all of these cases, the amount of data to be processed depends on the sparse structure of the problem, which will typically vary in a data-dependent way.

In supporting dynamic shapes, we chose not to support dynamic rank programs, e.g., programs whose inputs tensors change in dimensionality, as this pattern rarely occurs in real-world deep learning programs, and it avoids the need to reason inductively over symbolic lists of shapes.

# 5.1 Symbolic Shape Guards

The use of straight line traces in TorchDynamo was motivated by the need to reuse preexisting code written in Python/C++ targeting the PyTorch API. We continue this philosophy with dynamic shapes, unlike a fully symbolic system which might capture both branches of a conditional, we always pick one branch and specialize our trace under the assumption that this trace will only be reused when the assumptions hold. To do this, we maintain a size hint for every symbolic size saying what its concrete value was on the first input that triggered the just-in-time compilation. When we perform a condition on the shape of a tensor, we consult the hint to find out which branch to take and add a guard.

This greatly simplifies the symbolic shape formulas we produce, as we do not need to represent conditionals, but it means we have a much more involved system for managing guards. Consider, for example, the following program:

def f(x,y: z=torch.cat[x,y]) if z.size0>2: return z.mul2 return z.add2

The final IR we will compile with TorchInductor will either be torch.cat[x,y]).add2) or torch.cat[x,y]).mul2) (with the condition flattened away), but to determine which branch we are in, we would need to know the size of z, an intermediate. Because TorchDynamo must know up-front if a compiled trace is valid (we do not support hallouts, like some JTF compilers), we must be able to reduce z.size(o) to an expression in terms of the inputs, x.sizeo-y.sizeo). This is done by writing meta functions for all operators in PyTorch. Meta functions propagate size information to the output of a tensor without actually performing computation on the node. At the time of writing, coverage for meta functions was 2657 out of 3028 PyTorch ops (including overloads), which covers the vast majority of real-world models since there is a long tail of rarely/never used operators. There is also mechanism for defining your own meta functions for custom ops.

# 5.2 Optimizing Dynamic Shapes Reasoning

A major motivation of dynamic shapes is to reduce compile time, as a compiler which supports only static shapes must recompile a kernel for every possible combination of possible input shapes. However, reasoning over symbolic shapes comes with its own costs: in the limit, the shape expressions for output tensors may be quite complicated. We employ a variety of strategies to reduce the performance impact of symbolic shapes reasoning:

Our default API for dynamic shapes does not require any user annotation: we assume that all inputs are potentially dynamic, model weights are static, and we infer the true dynamism by stepping through the model and analyzing the interactions between the two. We also support a mode assume_static_by_default which forces all input dimensions to be assumed static unless a user explicitly marks them as dynamic with mark_dynamic(tensor, dim). Code in PyTorch often performs tests on whether or not a size of a variable is zero or one; for example, when constructing a tensor, PyTorch computes if it is contiguous. A zero element tensor is always contiguous, so we always test if each dimension of a tensor is zero. Instead of forcing our symbolic reasoning system to rediscover this fact every trace, we instead proactively  $0 / 1$  specialize: if an input size is O or 1, instead of assigning it a symbolic variable, we treat it as a constant and add an appropriate guard. Specializing on 1 is important to capture broadcasting semantics in PyTorch and performance optimizations. Importantly, we can make a negative inference when we do allocate a symbolic variable: any symbolic variable must not equal  $0 / 1$  and so if we test if it is equal to  $0 / 1$  ,we can evaluate the expression to false without having to introduce a additional guard. As we process the user program, we incrementally simplify our symbolic expressions as we learn more facts from guards. Our current implementation simplifies unification and divisibility on the fly, and we also use SymPy [28] to help us determine if a requested guard is already statically known, in which case we can eliminate it.

# 5.3 Hint-Free (Unbacked) Symbolic Integers

To resolve control flow, we check the actual value of a symbolic integer to determine which branch to take and guard on. Unbacked symbolic integers arise when a size variable emerges from a data-dependent operation like .nonzero() or .item() and the actual value is unknown. It is illegal to perform control flow on these symbolic integers, so we must graph break on these operations.

Naively implemented, this is too restrictive and results in too many graph breaks. The most important enhancements to work around these are: 1) On tensor creation, PyTorch precomputes data about a tensor; for example, when using empty_strided to create a tensor, it will sort the strides and determine whether the tensor is non-overlapping and dense. Sorts produce a lot of guards. However, it is more common to produce a tensor directly with a higher-level API like empty, which is guaranteed to produce a non-overlapping and dense tensor. We modified PyTorch to avoid needlessly recomputing these properties. 2) Even if nontrivial compute is needed, sometimes a property is never used. Making these precomputed properties lazy allows us to avoid guarding on unused properties. 3) It is generally unknown whether or not data within an integer tensor may be negative. However, we provide an API constrain_range whereby a user can specify that a size is bounded above and below by known limits.

# 6 Experimental Results

We run our evaluation on three different benchmark suites. TorchBench [14] is a benchmark suite containing a diverse set of models taken from open source repositories selected from highly cited projects as ranked by Papers with Code [35]. HuggingFace [53] is a popular library for Transformer [49] models. TIMM [52] is a popular library containing vision models in PyTorch. To turn the last two libraries into a benchmark suite, we selected representative models covering every category of model available.

Our benchmarking infrastructure is open source [40] in the hope that other publications will use it. Additional results can be found in the TorchInductor Performance Dashboard [41], including: per-model performance, different TorchInductor settings, and daily updates for PyTorch nightly builds. Experiments were run on an NVIDIA A100 GPU, CUDA 11.6, and an Intel Xeon 8275CL CPU. Experiments were repeated 100 times to reduce noise, with 3 warm up iterations. We applied a timeout of 30 minutes per model and count timeouts as failures. TorchInductor was run with a PyTorch nightly build from 8/30/2023, with max-autotune, freezing, and cudagraphs enabled. Other versions used are: nvFuser 2.0; NNC 2.0; Hidet 0.2.2; TVM 0.11.1; ONNX Runtime (ONNXRT) 1.14.1; and PyTorch/XLA 2.1. For training experiments, we measure a single step of both the forwards and backwards pass excluding the optimizer.

# 6.1 TorchDynamo's Ability to Capture Graphs

The first section of Table 1 shows experimental results comparing TorchDynamo to TorchScript [17] in terms of their ability to capture different benchmark suites. For HuggingFace, TorchScript fails on every model because HuggingFace models returns a Model0output container class that TorchScript does not support. Most TIMM models work with TorchScript because the maintainers of TIMM use TorchScript in their workflows and have put in effort to adapt their models. On TorchBench, TorchDynamo works on more than twice as

Table 1. TorchDynamo statistics from each benchmark suite, measured using float32 inference on an NVIDIA A100 GPU.  

<table><tr><td></td><td>TorchBench</td><td>HuggingFace</td><td>TIMM</td></tr><tr><td>Model Count</td><td>80</td><td>46</td><td>62</td></tr><tr><td>Works with TorchDynamo</td><td>74 (93%)</td><td>46 (100%)</td><td>62 (100%)</td></tr><tr><td>Compare with TorchScript [1]</td><td>36 (45%)</td><td>0 (0%)</td><td>61 (98%)</td></tr><tr><td>Operators Captured</td><td>91.8%</td><td>99.8%</td><td>100%</td></tr><tr><td>Mean Operators per Graph</td><td>252.8</td><td>612.6</td><td>450.7</td></tr><tr><td>Mean Graphs per Model</td><td>21.1</td><td>7.7</td><td>1</td></tr><tr><td>Models with 0 graph breaks</td><td>52 (70%)</td><td>41 (89%)</td><td>62 (100%)</td></tr><tr><td>Models with 1 to 9 graph breaks</td><td>4 (8%)</td><td>1 (2%)</td><td>0 (0%)</td></tr><tr><td>Models with 10+ graph breaks</td><td>16 (22%)</td><td>4 (9%)</td><td>0 (0%)</td></tr></table>

<table><tr><td></td><td>Inference</td><td>Training</td></tr><tr><td>TorchDynamo</td><td>5%</td><td>1%</td></tr><tr><td>Lazy Tensors</td><td>38%</td><td>90%</td></tr><tr><td>Lazy Tensors + cross-iteration pipelining</td><td>31%</td><td>86%</td></tr></table>

Table 2. Overheads (lower is better) as a percentage of eager PyTorch execution time for graph capture. This experiment uses the same kernels as eager PyTorch, so overheads are graph capture cost only.Measured using float32 TorchBench on an NVIDIA V100 GPU.

many models as TorchScript. TorchBench is the most representative of the three benchmark suites for graph capture comparison because it is made up of models taken from diverse sources.

The second section of Table 1 provides statistics about the quality of graphs captured by TorchDynamo, normalized as a percentage of working models. Unlike prior systems, which were all-or-nothing, TorchDynamo can capture partial programs and multiple graphs. TorchDynamo is able to capture a single whole-program graph most of the time, and even when there are graph breaks, typical graphs are hundreds of operators in size. The most common reason for graph breaks are: usage of non-PyTorch libraries such as numpy [21]; conversion to Python types such as tolist() and data-dependent control flow operations. There is support for compiling numpy operations with torch.compile, which is not enabled for this experiment.

# 6.2 Overheads of Graph Capture

Table 2 measures the runtime overheads introduced by graph capture for both TorchDynamo and Lazy Tensors. The other systems are ahead-of-time and do not introduce runtime overhead. We run each system using the same kernels as PyTorch eager, so slowdowns are from graph capture overhead only. We take the geometric mean slowdown on TorchBench and subtract 1 to get a percentage overhead added. As with all our results, we exclude warm up iterations from timing, so this is measuring steady-state performance.

While the overheads of TorchDynamo are less than  $5\%$  Lazy Tensors adds a large amount of overhead. These Lazy

Tensor overheads are not uniform across models. For training with cross-iteration pipelining: one third of models are better than  $10\%$  overhead, one third of models are between  $10\%$  and  $66\%$  overhead, and one third of models are between  $66\%$  and  $1759\%$  overhead.

One way to mitigate Lazy Tensor overheads in training and offline inference is cross-iteration pipelining. This helps with the fact that for a single iteration of Lazy Tensors, the GPU is idle while the CPU captures, then the CPU is idle while the GPU executes what was captured. By running multiple iterations one can overlap the capture of iteration  $N$  with the execution of iteration  $N -1$ . Lazy Tensors + cross-iteration pipelining in Table 2 measures this amortization effect by measuring 10 iterations rather than 1 iteration. There is a small improvement in Lazy Tensor overheads from this strategy.

For many models, Lazy Tensor capture is too slow to saturate the GPU. This is especially true for smaller models or ones with large numbers of operations. In these cases, pipelining does not help because the limiting factor is Lazy Tensor overheads. For some PyTorch models, there is code like if torch.any(torch.isnan(x)) or print(loss.item()). Both of these operations take values from within PyTorch tensors and convert them into Python bool of float types. This type of code is fast in eager mode PyTorch, but defeats any cross iteration pipelining, because with a (not yet computed) Lazy Tensor, you have no way of knowing what torch.any() should return (which controls the branch the code will take) or the values to print. Since Lazy Tensors has zero visibility into the Python code calling it, this pattern forces a flush of any accumulated pipeline of ops and requires the CPU capture to stall and wait for the GPU to catch up.

# 6.3 TorchInductor Speedups

Table 3 shows the geometric mean speedups of TorchInductor and six other TorchDynamo backends over PyTorch eager across our three benchmark suites and many configurations. In this experiment, we hold the graph capture mechanism (TorchDynamo) constant and only vary the backend compiler, so every backend gets the same input graphs and incurs the same capture overheads. Figure 4 is based on the same data as Table 3, but shows the Cumulative Distribution Function (CDF) of speedups with the three benchmark suites combined. This helps better understand how speedups are distributed.

TorchInductor is faster than other backends in most cases. nvFuser [36] and NNC [60] both have speedups clustered around  $1\times$  because they make use of eager PyTorch kernels and only generate code for a subset of PyTorch. PyTorch/XLA [42] has more varied performance, in many cases generating large speedups and, in other cases, large slowdowns which pull down the average. It performs better for GPU float16 inference compared to other configurations, especially on the vision models in TIMM. ONNX Runtime [16],

<table><tr><td rowspan="3" colspan="2"></td><td colspan="5">TorchBench</td><td colspan="5">HuggingFace</td><td colspan="4">TIMM</td></tr><tr><td colspan="2">Inference</td><td colspan="2">Training</td><td colspan="2">Inference</td><td colspan="2">Training</td><td colspan="2">Inference</td><td colspan="2">Training</td><td colspan="2">Training</td></tr><tr><td>Models Working</td><td>Geomean Speedup</td><td>Models Working</td><td>Geomean Speedup</td><td>Models Working</td><td>Geomean Speedup</td><td>Models Working</td><td>Geomean Speedup</td><td>Models Working</td><td>Geomean Speedup</td><td>Models Working</td><td>Geomean Speedup</td><td>Models Working</td><td>Geomean Speedup</td></tr><tr><td rowspan="3">NVIDIA</td><td>None (TorchDynamonomy)</td><td>74/74</td><td>0.95×</td><td>59/59</td><td>0.99×</td><td>44/46</td><td>1.01×</td><td>46/46</td><td>0.98×</td><td>62/62</td><td>1.00×</td><td>62/62</td><td>1.00×</td><td></td><td></td></tr><tr><td>torchinductor</td><td>74/74</td><td>2.73×</td><td>58/59</td><td>1.38×</td><td>44/46</td><td>1.47×</td><td>46/46</td><td>1.24×</td><td>62/62</td><td>2.48×</td><td>62/62</td><td>1.38×</td><td></td><td></td></tr><tr><td>nvFuser</td><td>53/74</td><td>1.23×</td><td>45/59</td><td>1.04×</td><td>33/46</td><td>1.09×</td><td>36/46</td><td>1.09×</td><td>57/62</td><td>1.16×</td><td>56/62</td><td>1.03×</td><td></td><td></td></tr><tr><td>A100 GPU</td><td>ONNXRT</td><td>[46]</td><td>53/74</td><td>1.12×</td><td>42/59</td><td>1.03×</td><td>28/46</td><td>0.98×</td><td>21/46</td><td>0.94×</td><td>57/62</td><td>1.02×</td><td>58/62</td><td>0.96×</td><td></td></tr><tr><td rowspan="4">float32</td><td>PyTorch/XLA</td><td>[42]</td><td>57/74</td><td>0.80×</td><td>42/59</td><td>0.73×</td><td>33/46</td><td>1.03×</td><td>18/46</td><td>0.98×</td><td>53/62</td><td>1.24×</td><td>52/62</td><td>1.11×</td><td></td></tr><tr><td>ONNXRT</td><td>[16]</td><td>34/74</td><td>0.86×</td><td>N/A</td><td>N/A</td><td>22/46</td><td>0.84×</td><td>N/A</td><td>N/A</td><td>29/62</td><td>0.92×</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>TVM</td><td>[18]</td><td>41/74</td><td>0.16×</td><td>N/A</td><td>N/A</td><td>0.74×</td><td>N/A</td><td>N/A</td><td>N/A</td><td>37/62</td><td>0.92×</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>Hidden</td><td>[15]</td><td>15/74</td><td>0.54×</td><td>N/A</td><td>N/A</td><td>28/46</td><td>0.09×</td><td>N/A</td><td>N/A</td><td>5/62</td><td>0.30×</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td rowspan="4">NVIDIA</td><td>None (TorchDynamonomy)</td><td>74/74</td><td>0.95×</td><td>57/57</td><td>0.99×</td><td>43/45</td><td>1.00×</td><td>45/45</td><td>0.97×</td><td>60/60</td><td>1.00×</td><td>60/60</td><td>1.00×</td><td></td><td></td></tr><tr><td>torchinductor</td><td>74/74</td><td>2.59×</td><td>57/57</td><td>1.50×</td><td>43/45</td><td>1.91×</td><td>45/45</td><td>1.45×</td><td>60/60</td><td>2.77×</td><td>60/60</td><td>1.50×</td><td></td><td></td></tr><tr><td>torchinductor</td><td>74/74</td><td>2.59×</td><td>57/57</td><td>1.50×</td><td>43/45</td><td>1.91×</td><td>45/45</td><td>1.45×</td><td>60/60</td><td>2.87×</td><td>60/60</td><td>1.50×</td><td></td><td></td></tr><tr><td>nvFuser</td><td>[36]</td><td>53/74</td><td>1.27×</td><td>45/57</td><td>1.04×</td><td>37/45</td><td>1.07×</td><td>35/45</td><td>1.04×</td><td>56/60</td><td>1.13×</td><td>54/60</td><td>1.01×</td><td></td></tr><tr><td rowspan="5">A100 GPU</td><td>ONNXRT</td><td>[60]</td><td>53/74</td><td>1.14×</td><td>46/57</td><td>1.03×</td><td>33/45</td><td>0.98×</td><td>37/45</td><td>0.94×</td><td>56/60</td><td>1.00×</td><td>56/60</td><td>0.95×</td><td></td></tr><tr><td>PyTorch/XLA</td><td>[42]</td><td>56/74</td><td>0.82×</td><td>42/57</td><td>0.80×</td><td>33/45</td><td>1.16×</td><td>18/45</td><td>0.24×</td><td>53/60</td><td>1.59×</td><td>50/60</td><td>1.27×</td><td></td></tr><tr><td>ONNXRT</td><td>[16]</td><td>34/74</td><td>0.86×</td><td>N/A</td><td>N/A</td><td>22/45</td><td>0.84×</td><td>N/A</td><td>N/A</td><td>29/60</td><td>0.92×</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>TVM</td><td>[11]</td><td>40/74</td><td>0.17×</td><td>N/A</td><td>N/A</td><td>33/45</td><td>0.18×</td><td>N/A</td><td>N/A</td><td>34/60</td><td>0.10×</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>Hidden</td><td>[18]</td><td>15/74</td><td>0.55×</td><td>N/A</td><td>N/A</td><td>0.45</td><td>N/A</td><td>N/A</td><td>N/A</td><td>5/60</td><td>0.46×</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td rowspan="2">Intel Xeon</td><td>None (TorchDynamonomy)</td><td>74/74</td><td>1.00×</td><td>57/57</td><td>0.99×</td><td>43/46</td><td>1.00×</td><td>46/46</td><td>1.00×</td><td>62/62</td><td>1.06×</td><td>61/61</td><td>1.00×</td><td></td><td></td></tr><tr><td>torchinductor</td><td>74/74</td><td>1.14×</td><td>57/57</td><td>1.35×</td><td>43/46</td><td>2.54×</td><td>40/46</td><td>1.36×</td><td>58/62</td><td>2.55×</td><td>52/61</td><td>1.42×</td><td></td><td></td></tr><tr><td>8275CL CPU</td><td>PyTorch/XLA</td><td>[60]</td><td>51/74</td><td>1.39×</td><td>47/57</td><td>0.99×</td><td>33/46</td><td>1.15×</td><td>38/46</td><td>0.92×</td><td>58/62</td><td>1.04×</td><td>58/61</td><td>0.85×</td><td></td></tr><tr><td rowspan="2">float32</td><td>ONNXRT</td><td>[16]</td><td>46/74</td><td>1.06×</td><td>N/A</td><td>N/A</td><td>33/46</td><td>0.64×</td><td>N/A</td><td>N/A</td><td>54/62</td><td>1.02×</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>TVM</td><td>[11]</td><td>44/74</td><td>0.64×</td><td>N/A</td><td>N/A</td><td>10/46</td><td>0.10×</td><td>N/A</td><td>N/A</td><td>47/62</td><td>1.46×</td><td>N/A</td><td>N/A</td><td></td></tr></table>

Table 3. Geometric mean speedups (higher is better) over PyTorch eager for different TorchDynamo backends and the number of models they work on in each benchmark suite. Only working models are included in speedup calculation. Comparisons use same precision in eager mode. N/A means the backend does not support that configuration. All backends use TorchDynamo as frontend to capture graphs in this experiment and receive the same initial graphs. None is included as a way to estimate overheads (or speedups) of TorchDynamo without any graph optimizations applied.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-07/9ed94e03-5712-4e4d-babd-00dd15e3db8a/2550289d6b7ecd45f882ce6956319b4b32c4693dca49b5a5eb526e01304868e4.jpg)  
Figure 4. Cumulative Distribution Function (CDF) of speedups over PyTorch eager mode. For speedups (x-axis) higher is better and PyTorch eager is  $1\times$ . Same underlying data as Table 3.

Table 4. Ablation study measuring the impact of removing optimizations from TorchInductor. Geometric mean speedups over eager PyTorch on float16 HuggingFace on an NVIDIA A100 GPU. Parenthesis is difference from All TorchInductor optimizations.  

<table><tr><td></td><td>Inference</td><td>Training</td></tr><tr><td>All TorchInductor optimizations</td><td>1.91×</td><td>1.45×</td></tr><tr><td>Without loop/layout reordering</td><td>1.91× (-0.00)</td><td>1.28× (-0.17)</td></tr><tr><td>Without matmul templates</td><td>1.85× (-0.06)</td><td>1.41× (-0.04)</td></tr><tr><td>Without parameter freezing</td><td>1.85× (-0.06)</td><td>1.45× (-0.00)</td></tr><tr><td>Without pattern matching</td><td>1.83× (-0.08)</td><td>1.45× (-0.00)</td></tr><tr><td>Without cudagraphs</td><td>1.81× (-0.10)</td><td>1.37× (-0.08)</td></tr><tr><td>Without fusion</td><td>1.68× (-0.23)</td><td>1.27× (-0.18)</td></tr><tr><td>Without nlining</td><td>1.58× (-0.33)</td><td>1.31× (-0.14)</td></tr><tr><td>Without fusion and nlining</td><td>0.80× (-1.11)</td><td>0.59× (-0.86)</td></tr></table>

TVM [11], and Hidet [18] are inference-only and fail to run many models due to missing operator implementations and other issues. On CPU, the ONNX runtime generates speedups above  $8x$  for 5 models (compared to 1 for TorchInductor), but these results did not generalize -more than half of models show slowdowns. On GPU, TVM and Hidet produce slowdowns for all except 4, and 2, models respectively. On CPU, TVM performs significantly better for some models while generating large slowdowns on others. TVM would have been the second fastest CPU inference backend on TorchBench (behind TorchInductor) if we excluded the models where it generated large slowdowns.

# 6.4Sources of TorchInductor Speedups

Table 4 explores where TorchInductor's speedups are coming from by disabling optimizations one at a time an measuring the impact on geometric mean speedup on HuggingFace models. If removing a specific optimization results in a bigger slowdown, this implies that it is responsible for more of the speedup.

The biggest speedups in TorchInductor come from combining pointwise, reduction, and scatter kernels together into a smaller number of fused kernels, which reduces memory traffic since values can be reused without requiring a round trip to memory. In TorchInductor, these kernel combinations happen in two places: 1) Inlining happens during lowering, and duplicates the body of pointwise kernels into all their consumers when thresholds are met. 2) Fusion happens during scheduling, and combines remaining kernels together, and also does horizontal consumer/consumer fusions. There is a lot of overlap between those passes, so we also include a line without fusion and nlining that disables both. Without both of those passes, TorchInductor generates slowdowns rather than speedups. This is because the decompositions performed by TorchInductor break up larger optimized operators into many smaller primitive operators, and we rely on fusions to recombine them to recover  $1\times$  performance.

The remaining optimizations measured in Table 4 are: 1) Loop/layout reordering uses a voting algorithm to reorder loops in kernels and change data layouts to match usage. 2) Matmul templates use Triton templates with pointwise epilogue fusion for matrix multiply instead of cuBLAS/cuDNN. There is an autotuner (enabled by mode="max-autotune") to select when to use these templates. Without this optimization, TorchInductor does not use templates at all. 3) Parameter freezing is an inference-only optimization that constant-folds away parts of the model that only depend on parameters. 4) Pattern matching uses graph-level peephole optimizations to rewrite the input graph before it is lowered to TorchInductor. 5) Cudagraphs is a way to reduce kernel launch overheads at the CUDA driver level. TorchInductor will automatically use this when static analysis shows it to be safe and it is enabled in the configuration.

# 7 Conclusions

In this paper, we presented two extensions to PyTorch: TorchDynamo and TorchInductor, which deliver speedups through graph compilation in PyTorch programs while retaining the flexibility and usability of the eager programming model PyTorch is known for. By enabling graph compilation in PyTorch programs, we hope to empower researchers and practitioners to tackle larger and more complex machine learning problems with greater efficiency and flexibility.

# Acknowledgements

We gratefully thank the anonymous reviewers and our shepherd, Martin Maas, for their suggestions and feedback that helped improve this paper. Thanks to Brett Simmers for proofreading. Thanks to everyone working on Triton, TorchInductor's GPU backend would not have been possible without it. Thanks to the Intel PyTorch team: Guobing Chen, Leslie Fang, Jiong Gong, Xuan Liao, Yudong Si, Chuanqi Wang, Eikan Wang, Chunyuan Wu, Weiwen Xia, Xiaobing Zhang, Fan Zhao, and Beilei Zheng. Their work greatly improved TorchInductor's CPU backend. Finally, thanks to the thousands of people who have contributed code to PyTorch. This work would not have been possible without the countless contributions that collectively made PyTorch what is is today.

# A Artifact Appendix

# A.1 Abstract

The source code for this work is included in PyTorch which is available at https://github.com/pytorch/pytorch/. TorchDynamo can be found in the torch/_dynamo directory and TorchInductor can be found in the torch/_inductor directory. Benchmarking code to reproduce the results in the

paper can be found at https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo.

Since this paper includes a large number of experiments that in aggregate will take weeks to run, the instructions here will focus on reproducing the TorchInductor GPU HuggingFace results. The workflow to reproduce other results is very similar to this and is described at the end. Additional instructions are included in the README .md included in the benchmarks/dynamo directory in PyTorch.

# A.2 Artifact check-list (meta-information)

A.2 Artifact check-list (meta-information)-Binary: distributions available at https://pytorch.org/-Hardware: NVIDIA A100 GPU, Intel Xeon 8275CL CPU-Metrics: Geomean speedup over PyTorch eager mode-How much disk space required (approximately)? 50 GB-How much time is needed to prepare workflow (approximately)? 1 hour-How much time is needed to complete experiments (approximately)?  $< 1$  day per-backend, per-configuration for most experiments-Publicly available?: Yes-Code licenses (if publicly available?): BSD-3

# A.3 Description

# A.3.1 How to access.

A.3.1 How to access.-Source code and benchmark code: https://github.com/pytorch/pytorch/-PyTorch binaries: https://pytorch.org/-TorchBench: https://github.com/pytorch/benchmark/

# A.3.2 Hardware dependencies.

A.3.2 Hardware dependencies.-To match configurations in this paper: NVIDIA A100 GPU and Intel Xeon 8275CL CPU-Benchmarks can run with an NVIDIA GPU with SM80+ and 40GB+ of memory, most benchmarks can run with less-CPU results can be run without a GPU

# A.3.3 Software dependencies.

A.3.3 Software dependencies.-A recent Linux distribution-NVIDIA kernel drivers-CUDA version compatible with the chosen version of PyTorch-gcc/g++ compatible with the chosen CUDA-Miniconda installed (https://docs.conda.io/projects/miniconda/en/latest/)-PyTorch (and dependencies)-Additional python packages: pandas, scipy, psutil, and tqdm

# A.4 Installation

There are a number of options to install PyTorch which are described on https://pytorch.org/. A minimal installation including dependencies can be achieved using the following commands:

create a new conda environment conda create --name=pt2 python=3.10 conda activate pt2

install dependencies for benchmark code conda install pandas scipy psutil tqdm

install PyTorch using release build conda install pytorch torchvision torchauto pytorch-cuda=12.1 -c pytorch -c nvidia

Next, download the PyTorch source code in order to access benchmarking scripts:

clone the PyTorch repository to get benchmark code git clone --recursive --branch=release/v2.1 https://github.com/pytorch/pytorch

benchmark code should be run from the root PyTorch directory cd pytorch

# A.5 Experiment workflow

To reproduce TorchInductor speedups over eager PyTorch on HuggingFace, float16, GPU, inference run:

TORCHINDUCTOR_MAX_AUTOTUNE \(= 1\) ./benchmarks/dynamo/huggingface.py --performance --no-skip --duda --float16 --inference --inductor ----------------------------------------------------------------------------------------------------\) --output= pwd /results.csv

This downloads HuggingFace models and runs them both with and without TorchDynamo to compute speedups compared to PyTorch eager mode. Results are written to results.csv in the current working directory. If one runs additional experiments, --output should be set to a unique absolute filename for each one.

# A.6 Evaluation and expected results

A.6 Evaluation and expected resultsThe chosen output file (results.csv) should contain 46 entries showing speedup numbers (and other metrics) for each model. All models should be working (failures are represented as a zero speedup) and the geomean of all the speedups should be similar to the speedups reported in the paper.

# A.7 Experiment customization

The above command can be customized in many ways:

./benchmarks/dynamo/huggingface.py can be substituted with the scripts ./benchmarks/dynamo/timm_models.py or ./benchmarks/dynamo/torchbench.py for the three benchmark suites. Note that TorchBench requires additional installation steps, while the other two auto-download dependencies. -dcuda can be replaced with -dcpu for CPU -float16 can be replaced with -float32 or --amp --inference can be replaced with --training --inductor can be replaced with --backend=eager (for "None"), --backend=nvfuser, --backend=nnc, --xla, --backend=onnxrt, --backend=svm, or --backend=hidet. Note that each backend has different dependencies and setup instructions.

--freezing and/or TORCHINDUCTOR_MAX_AUTOTUNE=1 can be removed to disable those optimizations in TorchInductor. Many more optimization flags can be found in torch/_inductor/config.py.-Many other options and backends are available via 
--help

The results in this paper include the combinatorial product of most of these flags.

# A.8 Notes

-Speedups and model coverage results have improved in recent versions of PyTorch compared to results shown in this paper. We recommend running the latest PyTorch version for future comparisons.-Performance results can be sensitive to environment setup, such as hardware and CUDA versions, so some small differences are expected.-Additional installation steps are required for Torch-Bench and non-TorchInductor backends.-A performance dashboard based on these scripts is available at https://hud.pytorch.org/benchmark/compilers.

# References

[1] [SW] Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vamhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng, TensorFlow. Large-scale machine learning on heterogeneous systems Nov. 2015. doi: 10.5281/zenodo.4724125. [2] Martin Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2016. Tensorflow: a system for large-scale machine learning. In Proceedings of the 12th USENIX Conference on Operating Systems Design and Implementation (OSDI'16). USENIX Association, Savannah, GA, USA, 265-283. isbn: 9781931971331. [3] Hameer Abbasi, Edward Z Yang, and Ralf Gommers. 2020. Improving subclassing Tensor by propagating subclass instances. https://github.com/pytorch/rfcs/blob/master/RFC-0001-torch-function-for-me thods.md. (Aug. 2020). [4] Akshay Agrawal, Akshay Naresh Modi, Alexandre Passos, Allen Lavoie, Ashish Agarwal, Asim Shankar, Igor Ganichev, Josh Levenberg, Mingsheng Hong, Rajat Monga, and Shanqing Cai. 2019. TensorFlow Eager: A Multi-Stage, Python-Embedded DSL for Machine Learning. CoRR, abs/1903.01855. http://arxiv.org/abs/1903.01855 [5] Rami Al-Rfou et al. 2016. Theano: A Python framework for fast computation of mathematical expressions. CoRR, abs/1605.02688. http://arxiv.org/abs/1605.02688 arXiv: 1605.02688. [6] Jason Ansel, Shoaib Kamil, Kalyan Veeramachaneni, Jonathan RaganKelley, Jeffrey Bosboom, Una-May O'Reilly, and Saman Amarasinghe.

2014. Opentuner: an extensible framework for program autotuning. In Proceedings of the 23rd International Conference on Parallel Architectures and Compilation (PACT '14). Association for Computing Machinery, Edmonton, AB, Canada, 303-316. ISBN: 9781450328098. DOI: 10.1145/2628071.2628092. [7] Riyadh Baghdadi, Jessica Ray, Malek Ben Romdhane, Emanuele Del Sozzo, Abdurrahman Akkas, Yunming Zhang, Patricia Suriana, Shoaib Kamil, and Saman Amarasinghe. 2019. Tiramisu: a polyhedral compiler for expressing fast and portable code. In Proceedings of the 2019 IEEE/ACM International Symposium on Code Generation and Optimization (CGO 2019). IEEE Press, Washington, DC, USA, 193-205. ISBN: 9781728114361. [8] [SW] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Madlaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang, JAX: composable transformations of Python+NumPy programs version 0.3.13, 2018. URL: http://github.com/google/jax. [9] Dino Viehland Brett Cannon. 2016. PEP 523: adding a frame evaluation API to CPython. https://peps.python.org/pep-0523/. (2016). [10] Jack Cao. 2022. PyTorch/XLA 2022 Q4 dev update. https://dev-discuss.pytorch.org/t/pytorch-xla-2022-Q4-dev-update/961. (2022). [11] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. 2018. TVM: an automated End-to-End optimizing compiler for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18). USENIX Association, Carlsbad, CA, (Oct. 2018), 578-594. ISBN: 978-1-939133-08-3. https://www.usinix.org/conference/osdi18/presentation/chen. [12] Tianqi Chen, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. 2018. Learning to optimize tensor programs. In Proceedings of the 32nd International Conference on Neural Information Processing Systems (NIPS'18). Curran Associates Inc., Montréal, Canada, 3393-3404. [13] Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. 2014. cuDNN: efficient primitives for deep learning. (2014). arXiv: 1410.0759 [cs:NE]. [14] Will Constable, Xu Zhao, Victor Bittorf, Eric Christoffersen, Taylor Robie, Eric Han, Peng Wu, Nick Korovaiko, Jason Ansel, Orion Reblitz-Richardson, and Soumith Chintala. 2020. TorchBench: a collection of open source benchmarks for PyTorch performance and usability evaluation. https://github.com/pytorch/benchmark. (Sept. 2020). [15] Leonardo Dagum and Ramesh Menon. 1998. OpenMP: an industry standard API for shared-memory programming. Computational Science & Engineering, IEEE, 5, 1, 46-55. [16] ONNX Runtime developers. 2021. ONNX runtime. https://www.onnxruntime.ai. (2021). [17] Zachary DeVito et al. 2018. TorchScript. https://pytorch.org/docs/1.9.0/jit.html. (Sept. 2018). [18] Yaoyao Ding, Cody Hao Yu, Bojian Zheng, Yizhi Liu, Yida Wang, and Gennady Pekhimenko. 2023. Hidet: task-mapping programming paradigm for deep learning tensor programs. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS 2023). Association for Computing Machinery, Vancouver, BC, Canada, 370-384. ISBN: 9781450399166. DOI: 10.1145/3575693.3575702. [19] Siyuan Feng, Bohan Hou, Hongyi Jin, Wuwei Lin, Junru Shao, Ruihang Lai, Zihao Ye, Lianmin Zheng, Cody Hao Yu, Yong Yu, and Tianqi Chen. 2022. TensorIR: an abstraction for automatic tensorized program optimization. (2022). arXiv: 2207.04296 [cs.LG].

[20] Alan Gray. 2019. Getting started with CUDA graphs. https://developer.nvidia.com/blog/cuda-graphs/. (2019).  [21] Charles R. Harris, K. Jarrod Millman, Stefan J van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernandez del Rio, Mark Wiebe, Pearu Peterson, Pierre Gerard-Marchant, Kevin Shoppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. 2020. Array programming with NumPy. Nature, 585, 357-362. doi: 10.1038/s41586-020-2649-2.  [22] Horace He. 2019. The state of machine learning frameworks in 2019. https://thegradient.pub.state-of-ml-frameworks-2019-pytorch-dominates-research-tensorflow-dominates-industry/. (2019).  [23] Mike Innes et al. 2017. On machine learning and programming languages. https://julialang.org/blog/2017/12/ml-pl/. (Dec. 2017).  [24] ISO. 1998. ISO/IEC 14882:1998: Programming languages -C++. (Sept. 1998), 732. http://webstore.ansi.org/ansidocstore/product.asp?sku=ISO%2FIEC+14882%2D1998.  [25] Yangqing Jia, Evan Shellhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross B. Girshick, Sergio Guadarrama, and Trevor Darrell. 2014. Caffe: Convolutional Architecture for Fast Feature Embedding. CoRR, abs/1408.5093. http://arxiv.org/abs/1408.5093 arXiv: 1408.5093.  [26] Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, Gau-. rav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, Rick Boyle, Pierre-luc Cantin, Clifford Chao, Chris Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb, Tara Vazir Ghaemmaghane, Rajendra Gottipati, William Gulland, Robert Hagmann, C. Richard Ho, Doug Hogberg, John Hu, Robert Hundt, Dan Hunt, Julian Ibarz, Aaron Jaffey, Alek Jaworski, Alexander Kaplan, Hurshit Khaitan, Daniel Killebrew, Andy Koch, Naveen Kumar, Steve Lacy, James Laudon, James Law, Diemthu Le, Chris Leary, Zhuyuan Liu, Kyle Lucke, Alan Lundin, Gordon MacKean, Adriana Maggiore, Maure Mahony, Kieran Miller, Rahul Nagarajan, Ravi Narayanaswami, Ray Ni, Kathy Nix, Thomas Norrie, Mark Omernick, Narayana Panukonda, Andy Phelps, Jonathan Ross, Matt Ross, Amir Salek, Ema I Samadiani, Chris Severn, Gregory Sizikov, Matthew Snelham, Jed Souter, Dan Steinberg, Andy Swing, Mercedes Tan, Gregory Thorson, Bo Tian, Horia Toma, Erick Tuttle, Vijay Vasudevan, Richard Walter, Walter Wang, Eric Wilcox, and Doe Hyun Yoon. 2017. In-datacenter performance analysis of a tensor processing unit. In Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA '17). Association for Computing Machinery, Toronto, ON, Canada, 1-12. ISBN: 9781450348928. doi: 10.1145/3079856.3080246.  [27] Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Tinenko. 2021. MLIR: scaling compiler infrastructure for domain specific computation. In 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), 2-14. DOI: 10.1109/CGO51591.2021.9370308.  [28] Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondrej Certik, Sergey B. Kirpichev, Matthew Rocklin, Amit Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger, Richard P. Muller, Fransesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Stépán, Roučka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrmam, and Anthony Scopatz. 2017. SymPy: symbolic computing in Python. PeerJ Computer Science, 3, (Jan. 2017). doi: 10.7717/peerj-cs.103.  [29] Adrian Mönrich, Armin Ronacher, David Lord, Grey Li, Joshua Bronson, Markus Unterwaditzer, and Philip Jones. 2023. Jinja project. https://github.com/pallets/jinja. (2023).

[30] NVIDIA, Péter Vingelmann, and Frank H.P. Fitzek. 2023. CUDA. https://developer.nvidia.com/cuda-toolkit. (2023).  [31] 2023. ONNX. https://onnx.ai/. (2023).  [32] 2019. Pytorch: an imperative style, high-performance deep learning library. Proceedings of the 33rd International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA, 12 pages.  [33] Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, and Saman Amarasinghe. 2013. Halide: a language and compiler for optimizing parallelism, locality, and recompuation in image processing pipelines. In Proceedings of the 34th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI '13). Association for Computing Machinery, Seattle, Washington, USA, 519-530. ISBN: 9781450320146. DOI: 10.1145/2491956.2462176.  [34] James Reed, Zachary DeVito, Horace He, Ansley Ussery, and Jason Ansel. 2022. Torch.fx: practical program capture and transformation for deep learning in python. In Proceedings of Machine Learning and Systems. D. Marculescu, Y. Chi, and C. Wu, (Eds.) Vol. 4, 638-651. https://proceedings.mlsys.org/paper/2022/file/ca46c1b9512a7a8315f3a5c5a946e8265-Paper.pdf.  [35] Elvis Saravia. 2021. Papers with Code 2021: a year in review. https://medium.com/paperswithcode/papers-with-code-2021-a-year-in-review-de75d5a77b8b. (2021).  [36] Christian Sarofeen, Piotr Bialecki, Jie Jiang, Kevin Stephano, Masaki Kozuki, Neal Vaidya, and Stas Bekman. 2022. Introducing nvFuser, a deep learning compiler for PyTorch. https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/. (2022).  [37] Frank Seide and Amit Agarwal. 2016. CNTK: microsoft' open-source deep-learning toolkit. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). Association for Computing Machinery, San Francisco, California, USA, 2135. ISBN: 9781450342322. doi: 10.1145/2939672.294597.  [38] Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, Wuwei Lin, Masahiro Masuda, Cody Hao Yu, and Tianqi Chen. 2022. Tensor program optimization with probabilistic programs. (2022). arXiv: 2205.13603 [cs. LG].  [39] Alex Suhan, Davide Libenzi, Ailing Zhang, Parker Schuh, Brennan Saeta, Jie Young Sohn, and Denys Shabalin. 2021. LazyTensor: combining eager execution with domain-specific compilers. arXiv preprint arXiv:2102.13267.  [40] PyTorch Team. 2023. TorchDynamo Benchmarking Code. https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo. (2023).  [41] PyTorch Team. 2023. TorchInductor Performance Dashboard. https://hud.pytorch.org/benchmark/complilers. (2023).  [42] PyTorch XLA Team. 2023. PyTorch/XLA. https://github.com/pytorch/xla. (2023).  [43] Vijay Thakkar, Pradeep Ramani, Cris Ceyka, Aniket Shivam, Honghao Lu, Ethan Yan, Jack Kosaian, Mark Hoemmen, Haicheng Wu, Andrew Kerr, Matt Nicely, Duane Merrill, Dustyn Blasig, Fengqi Qiao, Piotr Majcher, Paul Springer, Markus Hohnerbach, Jin Wang, and Manish Gupta. 2023. CUTLASS. https://github.com/NVIDIA/cutlass. Version 3.0.0. (Jan. 2023).  [44] [SW] The IREE Authors, IREE Sept. 2019. URL: https://github.com/openxla/tree.  [45] The XLA Team. 2017. XLA -Tensorflow, compiled. https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html. (Mar. 2017).  [46] Philippe Tillet, H. T. Kung, and David Cox. 2019. Triton: an intermediate language and compiler for tiled neural network computations. In (MAPL 2019). Association for Computing Machinery, Phoenix, AZ, USA, 10-19. ISBN: 9781450367196. DOI: 10.1145/3315508.3329973.  [47] Seiya Tokui, Ryosuke Okuta, Takuya Akiba, Yusuke Niitani, Toru Ogawa, Shunta Saito, Shuji Suzuki, Kota Uenishi, Brian Vogel, and

Hiroyuki Yamazaki Vincent. 2019. Chainer: A Deep Learning Framework for Accelerating the Research Cycle. CoRR, abs/1908.00213. http://arxiv.org/abs/1908.00213 arXiv: 1908.00213. [48] Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal, Zachary DeVito, William S. Moses, Sven Verdoolaege, Andrew Adams, and Albert Cohen. 2018. Tensor comprehensions: framework-agnostic high-performance machine learning abstractions. (2018). arXiv: 1802.04730 [cs.PL. [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Long Beach, California, USA, 6000-6010. isBN: 9781510860964. [50] B. P. Welford. 1962. Note on a method for calculating corrected sums of squares and products. Technometrics, 4, 3, 419-420. doi: 10.1080/00401706.1962.10490022. [51] Jian Weng, Animesh Jain, Jie Wang, Leyuan Wang, Yida Wang, and Tony Nowatzki. 2021. Unit: unifying tensorized instruction compilation. In 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), 77-89. doi: 10.1109/CGO51591.2021.93703 30. [52] Ross Wightman. 2019. PyTorch image models. https://github.com/r wightman/pytorch-image-models. (2019). doi: 10.5281/zenodo.4414 861. [53] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Perric Cistac, Clara Ma, Yacine Jennite, Julien Plu, Canwen Xu, Steven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In Association for Computational Linguistics, (Oct. 2020), 38-45. https://www.aclweb .org/anthology/2020. ernlp-demos.6. [54] Jiarong Xing, Leyuan Wang, Shang Zhang, Jack Chen, Ang Chen, and Yibo Zhu. 2022. Boot: bridging the gap between auto-tuners and hardware-native performance. In Proceedings of Machine Learning and Systems. D. Marculescu, Y. Chi, and C. Wu, (Eds.) Vol. 4, 204-216. https://proceedings.mloys.org/paper_files/paper/2022/file/38b3eff 8baf56627478ec76a704e9b52-Paper.pdf. [55] Shangdi Yu and Horace He. 2023. Transcending runtime-memory tradeoffs in checkpointing by being fusion aware. In Proceedings of Machine Learning and Systems. [56] Bojian Zheng, Ziheng Jiang, Cody Hao Yu, Haichen Shen, Joshua Fromm, Yizhi Liu, Yida Wang, Luis Ceze, Tianqi Chen, and Gennady Pekhimenko. 2022. Dialecte: automatic optimization for dynamic tensor programs. In Proceedings of Machine Learning and Systems. D. Marculescu, Y. Chi, and C. Wu, (Eds.) Vol. 4, 848-863. https://pro ceedings.mlsys.org/paper_files/paper/2022/file/fa7cdfad1a5aaf837 0ebeda47a1ff1c3-Paper.pdf. [57] Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, and Ion Stoica. 2020. Ansor: generating high-performance tensor programs for deep learning. In Proceedings of the 14th USENIX Conference on Operating Systems Design and Implementation (OSDI'20) Article 49. USENIX Association, USA, 17 pages. isBN: 978-1-939133-19-9. [58] Size Zheng, Renze Chen, Anjiang Wei, Yicheng Jin, Qin Han, Liqiang Lu, Bingyang Wu, Xiuhong Li, Shengen Yan, and Yun Liang. 2022. AMOS: enabling automatic mapping for tensor computations on spatial accelerators with hardware abstraction. In Proceedings of the 49th Annual International Symposium on Computer Architecture (ISCA '22). Association for Computing Machinery, New York, New York, 874-887. isBN: 9781450386104. doi: 10.1145/3470496.3527440. [59] Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, Haoyu Li, Chen Zhang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Wei Cui, Fan Yang,

Mao Yang, Lidong Zhou, Asaf Cidon, and Gennady Pekhimenko. 2022. ROLLER: fast and efficient tensor compilation for deep learning. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). USENIX Association, Carlsbad, CA, (July 2022), 233-248. isBN: 978-1-939133-28-1. https://www.usenix.org/conference/osdi22/presentation/zhu. [60] Mikhail Zolotukhin. 2021. NNC walkthrough: how PyTorch ops get fused. https://dev-discuss.pytorch.org/tunc-walkthrough-how-pytorch-ops-get-fused/125. (2021).