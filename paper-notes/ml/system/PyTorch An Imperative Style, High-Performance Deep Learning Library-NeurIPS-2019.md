# Abstract
Deep learning frameworks have often focused on either usability or speed, but not both. PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs.
>  DL 框架要么聚焦可用性，要么聚焦速度，而非两者同时
>  PyTorch 同时达成了两者: 提供了命令式和 Pythonic 的编程风格来构造模型、和流行的科学计算库兼容便于 debug，同时保持高效，支持例如 GPUs 的硬件加速器

In this paper, we detail the principles that drove the implementation of PyTorch and how they are reflected in its architecture. We emphasize that every aspect of PyTorch is a regular Python program under the full control of its user. We also explain how the careful and pragmatic implementation of the key components of its runtime enables them to work together to achieve compelling performance.
>  本文描述驱动 PyTorch 实现的原则以及它们在结构中的体现
>  PyTorch 的每一个部分都是由用户完全控制的 Python 程序
>  本文还描述了 PyTorch 运行时关键组件的实现方式

We demonstrate the efficiency of individual subsystems, as well as the overall speed of PyTorch on several common benchmarks.

# 1 Introduction
With the increased interest in deep learning in recent years, there has been an explosion of machine learning tools. Many popular frameworks such as Caffe [1], CNTK [2], TensorFlow [3], and Theano [4], construct a static dataflow graph that represents the computation and which can then be applied repeatedly to batches of data. This approach provides visibility into the whole computation ahead of time, and can theoretically be leveraged to improve performance and scalability. However, it comes at the cost of ease of use, ease of debugging, and flexibility of the types of computation that can be represented.
>  大多数框架构造静态数据流图来表示计算，然后将图反复应用于批量数据
>  该方法可以提前知道整个计算的流程，进而可以进行理论分析和优化
>  但其代价是不便于使用、debug 以及能够表示的计算类型不够灵活

Prior work has recognized the value of dynamic eager execution for deep learning, and some recent frameworks implement this define-by-run approach, but do so either at the cost of performance (Chainer [5]) or using a less expressive, faster language (Torch [6], DyNet [7]), which limits their applicability.
>  之前的工作意识到了动态即时运行的价值，一些最近的框架也实现了这种 “定义即运行” 的方式
>  但它们的性能要么不行，要么使用表达性更弱但是更快的语言

>  define-by-run 和 eager execution 是同义词，对应的词是 define-before-run
>  define-by-run 也就是我们写下的代码就是模型的定义，模型的执行和 Python 语句的解释挂钩，因此我们实际上是在一边定义模型，一边运行模型
>  define-before-run 则是构建了完整的计算图，再运行

However, with careful implementation and design choices, dynamic eager execution can be achieved largely without sacrificing performance. This paper introduces PyTorch, a Python library that performs immediate execution of dynamic tensor computations with automatic differentiation and GPU acceleration, and does so while maintaining performance comparable to the fastest current libraries for deep learning. This combination has turned out to be very popular in the research community with, for instance, 296 ICLR 2019 submissions mentioning PyTorch.
>  在精心设计下，动态即时执行可以在不牺牲性能的前提下实现
>  PyTorch 使用自动微分和 GPU 加速执行动态张量的即时执行的 Python 库，同时保持了和当前最快的 DL 库可比的性能

# 2 Background
Four major trends in scientific computing have become increasingly important for deep learning.
>  科学计算领域的四大趋势在 DL 中越来越重要

First, starting in the 1960s, the development of domain specific languages such as APL [8], MATLAB [9], R [10] and Julia [11], turned multidimensional arrays (often referred to as tensors) into first-class objects supported by a comprehensive set of mathematical primitives (or operators) to manipulate them. Separately, libraries such as NumPy[12], Torch[6], Eigen[13] and Lush[14] made array-based programming productive in general purpose languages such as Python, Lisp,  $\mathrm{C + + }$  and Lua.
>  首先，早期的 DSL 的发展将多维数组 (tensors) 变为一系列数学原语支持的第一类对象
>  此外，许多库使得基于 array 的编程在通用目的的语言中投入生产

Second, the development of automatic differentiation [15] made it possible to fully automate the daunting labor of computing derivatives. This made it significantly easier to experiment with different machine learning approaches while still allowing for efficient gradient based optimization. The autograd [16] package popularized the use of this technique for NumPy arrays, and similar approaches are used in frameworks such as Chainer [5], DyNet [7], Lush [14], Torch [6], Jax [17] and Flux.jl [18].
>  其次，自动微分的发展使得计算微分完全自动化，大大简化了尝试不同 ML 方法的实验过程
>  autograd 包推广了这一技术在 NumPy 数组中的使用，类似的方法也被用于各种框架中

Third, with the advent of the free software movement, the scientific community moved away from closed proprietary software such as Matlab[9], and towards the open-source Python ecosystem with packages like NumPy [12], SciPy [19], and Pandas [20]. This fulfilled most of the numerical analysis needs of researchers while allowing them to take advantage of a vast repository of libraries to handle dataset preprocessing, statistical analysis, plotting, and more. Moreover, the openness, interoperability, and flexibility of free software fostered the development of vibrant communities that could quickly address new or changing needs by extending the existing functionality of a library or if needed by developing and releasing brand new ones. While there is a rich offering of open-source software for neural networks in languages other than Python, starting with Lush [14] in Lisp, Torch [6] in  $\mathrm{C + + }$ , Objective-C and Lua, EBLearn [21] in  $\mathrm{C + + }$ , Caffe [1] in  $\mathrm{C + + }$ , the network effects of a large ecosystem such as Python made it an essential skill to jumpstart one's research. Hence, since 2014, most deep learning frameworks converged on a Python interface as an essential feature.
>  第三，随着自由软件的兴起，科学社区逐渐从 Matlab 这样的封闭软件转向开源 Python 生态系统，例如 Numpy, SciPy, Pandas
>  自 2014 年，大多数 DL 框架都将 Python 接口作为核心功能

Finally, the availability and commoditization of general-purpose massively parallel hardware such as GPUs provided the computing power required by deep learning methods. Specialized libraries such as cuDNN [22], along with a body of academic work (such as [23] and [24]), produced a set of high-performance reusable deep learning kernels that enabled frameworks such as Caffe [1], Torch7 [25], or TensorFlow [3] to take advantage of these hardware accelerators.
>  最后，大规模并行硬件例如 GPU 提供了 DL 方法所需的计算能力，专用库例如 cuDNN 提供了一组高性能和可复用的深度学习 kernel，使得各个框架可以利用 GPU 加速

PyTorch builds on these trends by providing an array-based programming model accelerated by GPUs and differentiable via automatic differentiation integrated in the Python ecosystem.
>  PyTorch 基于这些趋势，提供基于数组的编程模型，由 GPU 加速，具备自动微分，并且融合入 Python 生态

# 3 Design principles
PyTorch's success stems from weaving previous ideas into a design that balances speed and ease of use. There are four main principles behind our choices:
>  PyTorch 的成功来源于其设计平衡了速度和易用性，其四大原则如下:

**Be Pythonic** Data scientists are familiar with the Python language, its programming model, and its tools. PyTorch should be a first-class member of that ecosystem. It follows the commonly established design goals of keeping interfaces simple and consistent, ideally with one idiomatic way of doing things. It also integrates naturally with standard plotting, debugging, and data processing tools.
>  Pythonic: 接口简单，一致，理想情况下，每项任务只有一种惯用的实现方式，同时可以自然和标准绘图工具、debug 工具、数据处理工具集成

**Put researchers first** PyTorch strives to make writing models, data loaders, and optimizers as easy and productive as possible. The complexity inherent to machine learning should be handled internally by the PyTorch library and hidden behind intuitive APIs free of side-effects and unexpected performance cliffs.
>  研究者优先: PyTorch 使得编写模型、数据加载器、优化器尽可能简单和高效，PyTorch 应该处理 ML 内生的复杂性，提供没有副作用和非预期性能下降的 API

**Provide pragmatic performance** To be useful, PyTorch needs to deliver compelling performance, although not at the expense of simplicity and ease of use. Trading  $10\%$  of speed for a significantly simpler to use model is acceptable;  $100\%$  is not. Therefore, its implementation accepts added complexity in order to deliver that performance. Additionally, providing tools that allow researchers to manually control the execution of their code will empower them to find their own performance improvements independent of those that the library provides automatically.
>  性能: PyTorch 的实现会为了增加性能而提高复杂度，同时为研究者提供控制代码执行的工具

**Worse is better** [26] Given a fixed amount of engineering resources, and all else being equal, the time saved by keeping the internal implementation of PyTorch simple can be used to implement additional features, adapt to new situations, and keep up with the fast pace of progress in the field of AI. Therefore it is better to have a simple but slightly incomplete solution than a comprehensive but complex and hard to maintain design.
>  更简单更好: 保持 PyTorch 内部实现简洁，节约时间实现额外功能，适应新情况，跟上 AI 发展的步伐
>  因此与其使用一个全面但难以维护的设计，不如选择一个简单但略有不足的解决方案

# 4 Usability centric design
## 4.1 Deep learning models are just Python programs
In a surprisingly short amount of time, machine learning grew from recognizing individual digits [27] into autonomously playing StarCraft [28]. Consequently, the neural networks themselves evolved rapidly from simple sequences of feed forward layers into incredibly varied numerical programs often composed of many loops and recursive functions. To support this growing complexity, PyTorch foregoes the potential benefits of a graph-metaprogramming based approach to preserve the imperative programming model of Python. This design was pioneered for model authoring by Chainer[5] and Dynet[7]. PyTorch extends this to all aspects of deep learning workflows. Defining layers, composing models, loading data, running optimizers, and parallelizing the training process are all expressed using the familiar concepts developed for general purpose programming.
>  神经网络从 FFN 序列进化为包含循环和递归函数的复杂数值程序，为了支持这样的复杂性，PyTorch 放弃了基于图元编程方法的优势，保留 Python 的命令式编程模型
>  定义层、组合模型、加载数据、运行优化去、并行化训练过程都通过通用编程中已有的熟悉概念来表达

This solution ensures that any new potential neural network architecture can be easily implemented with PyTorch. For instance, layers (which in modern machine learning should really be understood as stateful functions with implicit parameters) are typically expressed as Python classes whose constructors create and initialize their parameters, and whose forward methods process an input activation. Similarly, models are usually represented as classes that compose individual layers, but let us state again that nothing forces the user to structure their code in that way. Listing 1 demonstrates how an entire model can be created by composing functionality provided by PyTorch such as 2d convolution, matrix multiplication, dropout, and softmax to classify gray-scale images. Note that linear layers are of course part of the library, but we show an example implementation to highlight how simple it is.
>  这确保任意可能的 NN 架构都能用 PyTorch 表示
>  例如 layers 使用 Python 类表示，构造函数创建并初始化其参数，`forward` 方法处理输入激活
>  类似地，模型用类表示，模型类包含了层类

```python
class LinearLayer(Module):
    def __init__(self, in_sz, out_sz):
        super().__init__()
        t1 = torch.randn(in_sz, out_sz)
        self.w = nn.Parameter(t1)
        t2 = torch.randn(out_sz)
        self.b = nn.Parameter(t2)
    def forward(self, activation):
        t = torch.mm(activations, self.w)
        return t + self.b
```

```python
class FullBasicModel(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.conv = nn.Conv2d(1, 128, 3)
        self.fc = LinearLayer(128, 10) 
    def forward(self, x):
        t1 = self.conv(x) 
        t2 = nn.functional.relu(t1)
        t3 = self.fc(t1)
        return nn.functional.softmax(t3)
```

Listing 1: A custom layer used as a building block for a simple but complete neural network.

This "everything is a just a program" philosophy is not limited to just the models, and applies to optimizers and data loaders as well. This facilitates the experimentation of new training techniques.
>  这种 “任何事都仅仅是一个程序” 的哲学不限于模型，对于优化器和数据加载器也是同理
>  这有助于新训练技术的实验

For example, to implement the very popular generative adversarial networks, one needs to specify two separate models (the generator and the discriminator), and two loss functions that depend on both models at the same time. Rigid APIs would struggle with this setup, but the simple design employed in PyTorch easily adapts to this setting as shown in Listing 2.
> 例如，为了实现 GAN，需要指定两个模型 generator, discrimnator 以及统一依赖于这两个模型的损失函数
> 僵化的 API 会在这种设置下遇到困难，但 PyTorch 采用的简单设计能够轻松适应这种场景

```python
discriminator = create_discriminator()
generator = create_generator()
optimD = optim.Adam(discriminator.parameters())
optimG = optim.Adam(generator.parameters())

def step(real_sample):
    #(1) Update Discriminator
    errD_real = loss(discriminator(real_sample), real_label)
    errD_real.backward()
    fake = generator(get_noise())
    errD_fake = loss(discriminator(fake.detach()), fake_label)
    errD_fake.backward()
    optimD.step()
    
    #(2) Update Generator
    errG = loss(discriminator(fake), real_label)
    errG.backward()
    optimG.step()
```

Listing 2: Simplified training of a generative adversarial networks.

Since PyTorch programs execute eagerly, all the features of Python are available throughout the whole design process. Print statements, standard debuggers, and common visualization tools like matplotlib all work as expected. Users do not have to wait for lengthy compilation before they can start running their programs, and more importantly intermediate computations can be observed to understand how a model works and whether its results are correct.
>  因为 PyTorch 程序是即时执行，所有的 Python 特性在整个设计过程中都可以使用: print 语句, 标准 debuggers, 通用的可视化工具等
>  用户不需要等待编译，可以直接运行其程序，并且**即时计算结果可以被观察到**，以判断模型是否正常工作

## 4.2 Interoperability and extensibility
Easy and efficient interoperability is one of the top priorities for PyTorch because it opens the possibility to leverage the rich ecosystem of Python libraries as part of user programs. Hence, PyTorch allows for bidirectional exchange of data with external libraries. For example, it provides a mechanism to convert between NumPy arrays and PyTorch tensors using the torch.from_numpy() function and .numpy() tensor method. Similar functionality is also available to exchange data stored using the DLPack [29] format. Note that this exchange happens in both cases without any data copying – objects on both sides only describe how to interpret a memory region which is shared among them. Hence, those operations are actually extremely cheap, and take constant time no matter how large the converted arrays are.
>  PyTorch 的一大优先级就是互操作性，便于利用 Python 的生态
>  PyTorch 提供了和外部库的双向数据交换，例如将 NumPy array 转化为 PyTorch tensors (`torch.from_numpy(), tensor.numpy()`)，也支持和使用 DLPack 格式存储的数据进行交换
>  数据交换不会涉及任何数据拷贝 - 双方的对象仅描述如何解释一个内存区域，因此这些操作是非常轻量的，无论转换的数组多大，时间都是常量级别

Moreover, many of the critical systems are designed specifically to be extensible. For instance, the automatic differentiation system allows users to add support for custom differentiable functions. To do that users can define a new subclass of torch.autograd.Function that implements forward() and backward() methods, which specify the function and its derivative (or more formally the vector-Jacobian product). Similarly new datasets can be added by subclassing torch.utils.data.Dataset and implementing two methods: `__getitem__` (the indexing operator) and `__len__` (the length operator), making datasets behave like (possibly lazy) lists. How these work is completely up to the implementer, and many users leverage other Python packages for data loading. The DataLoader class consumes objects conforming to this interface and provides an iterator over the data which takes care of shuffling, batching, parallelization, and management of pinned CUDA memory to improve throughput.
>  此外，PyTorch 的许多关键系统都设计为可拓展的
>  例如，自动微分系统允许用户为自定义微分方程添加支持，用户定义 `torch.autograd.Function` 的子类，实现 `forward(), backward()` 方法即可
>  类似地，新的数据集可以通过继承 `torch.utils.data.Dataset` 类，实现 `__getitem__` 和 `__len__` 方法即可，就能让数据集的行为和 list 类似
>  `DataLoader` 类会消费符合该接口的对象 (`Dataset` 类)，并提供一个数据迭代器，处理 shuffling, batching, 并行化的逻辑，以及管理固定的 CUDA 显存以提高吞吐

Most importantly, users are free to replace any component of PyTorch that does not meet the needs or performance requirements of their project. They are all designed to be completely interchangeable, and PyTorch takes great care not to impose any particular solution.

## 4.3 Automatic differentiation
Since gradient based optimization is vital to deep learning, PyTorch must be able to automatically compute gradients of models specified by our users, and those can be arbitrary Python programs. However, Python is a dynamic programming language that allows changing most behaviors at runtime, making ahead of time source-to-source differentiation cumbersome. Instead, PyTorch uses the operator overloading approach, which builds up a representation of the computed function every time it is executed. In its current implementation [30], PyTorch performs reverse-mode automatic differentiation, which computes the gradient of a scalar output with respect to a multivariate input. Differentiating functions with more outputs than inputs is more efficiently executed using forward-mode automatic differentiation, but this use case is less common for machine learning applications. PyTorch can be easily extended to perform forward-mode differentiation using array-level dual numbers [31, 32].
>  Python 是动态编程语言，允许在运行时更改大多数行为 (例如变量类型，函数行为等)，因此事先的 source-to-source 微分难以实现
>  PyTorch 使用运算符重载的方法，在每次计算函数被执行时构造它的表示
>  PyTorch 执行反向模式自动微分，根据计算多变量输入相对于一个标量输出的梯度
>  对于输出比输入多的函数，使用前向模式自动微分会更加高效，但 DL 通常都采用反向传播

>  source-to-source 是指在程序运行前，将整个代码翻译为另一端能求导的代码 (比如生成 C++ 或中间表示)，这是 `TensorFlow 1.x` 的做法 (构建计算图)
>  PyTorch 的运算符重载即 tensor 之间的运算例如 `+, - , *, /` 都重载为一个类，这个类不会直接计算，会先构建计算图 (完善已有的计算图，也就是 define-by-run)，便于后续的反向传播，然后再计算

Another interesting and uncommon feature of our system is that it can differentiate through code employing mutation on tensors, which is one of the basic building blocks of imperative programs. To ensure safety, we have implemented a versioning system for tensors, which lets us track their modifications and ensure that we always use the data we expect. One interesting tradeoff is that while we could utilize techniques like copy-on-write to support arbitrary programs, we chose to not go down this path, as performance-wise it is usually beneficial for the users to rewrite their code to ensure that no copies have to be performed. Hence, while most mutations are benign and can be handled automatically, the really complicated cases result in a user error, which lets them know that they likely want to restructure the program. This allows us to avoid introducing subtle and hard-to-find performance cliffs.
>  PyTorch 的另一个不常见的特性是能对直接修改张量的代码进行求导，而直接修改张量的操作正是命令式程序的基本部分
>  为了确保安全，我们为张量实现了版本控制系统，可以跟踪它们的修改，并确保我们始终使用预期的数据
>  虽然我们可以用 copy-on-write 来支持任意程序，但从性能角度来看，让用户重写代码以确保不进行任何复制则更有利
>  因此，虽然大多数 mutation 是无害的，可以自动处理，但在复杂的情况下会导致错误，要求用户重新设计程序，以避免引入难以发现的性能瓶颈

>  直接修改张量的代码例如

```python
x = torch.tensor([1.0, 2.0])
x[0] = 3.0 # mutation
```

>  这样的编程就是命令式风格的核心优势
>  为了确保一个张量被多次修改后，能够知道哪个版本被用于前向传播，需要为张量添加版本号信息，在反向传播时，检查版本号是否和前向计算时的一致，如果不一致，需要报错，例如下面的情况

```python
x = torch.tensor ([1.0], requires_grad=True)
y = x * 2
x[0] = 5.0  # 修改 x
z = y + 1   # 这里 y 是旧的 x 计算出来的！
```

>  copy-on-write 初始时不复制数据，在 tensor 要被修改时才复制一份，有点事节约内存和时间，缺点是隐藏了性能开销，用户不知道什么时候 copy 发生，因此不知道为什么执行变慢

# 5 Performance focused implementation
Running deep learning algorithms efficiently from a Python interpreter is notoriously challenging: for instance, the global interpreter lock [33] effectively ensures that only one of any number of concurrent threads is running at any given time. Deep learning frameworks based on the construction of a static data-flow graph sidestep this problem by deferring the evaluation of the computation to a custom interpreter.
>  从 Python 解释器高效运行 DL 算法一直是一个挑战: 例如，全局解释器锁有效地确保在任何给定时间内只能有一个并发线程在运行
>  基于静态数据流图的 DL 框架将计算的求值推迟到自定义解释器中来规避这个问题

PyTorch solved the problem differently, by carefully optimizing every aspect of its execution while simultaneously empowering its users to easily leverage additional optimization strategies.

## 5.1 An efficient C++ core
Despite being closely integrated in the Python ecosystem, most of PyTorch is written in  $\mathbf{C} + +$  to achieve high performance. This core libtorch library implements the tensor data structure, the GPU and CPU operators, and basic parallel primitives. It also provides the automatic differentiation system, including the gradient formulas for most built-in functions. This ensures that the computation of the derivatives of functions composed of core PyTorch operators is executed entirely in a multithreaded evaluator which does not require holding the Python global interpreter lock [33]. Python bindings are generated using YAML meta-data files. 
>  PyTorch 的大部分使用 C++ 编写，其核心的 libtorch 库实现了张量数据结构、GPU 和 CPU 算子、以及基本的并行原语，libtorch 还提供了自动微分系统，包括了大多数内建函数的梯度公式
>  这确保由核心 PyTorch 算子组成的函数的导数的计算完全在多线程求值器中执行，而无需持有 Python 全局解释器锁
>  Python 绑定通过 YAML 元数据文件生成

An interesting side-effect of this approach is that it allowed our community to quickly create bindings to multiple other languages resulting in projects like NimTorch [34], hasktorch [35] and others. 
>  这种方法的一个副作用是可以让社区快速为其他语言创建绑定

This design also allowed us to create first-class  $\mathrm{C + + }$  bindings and modeling libraries that can be used in places where Python is inconvenient, such as the game engine for Starcraft [36] or on mobile platforms. It is even possible to take the Python code describing a PyTorch model and run it without Python using the TorchScript engine [37].
>  这种设计还允许我们创建一流的 C++ 绑定和建模库
>  这些库可以在 Python 不便于使用的地方使用，例如移动平台
>  甚至可以使用 TorchScript 引擎，在不依赖 Python 的情况下运行描述 PyTorch 模型的 Python 代码

## 5.2 Separate control and data flow
PyTorch maintains a strict separation between its control (i.e. program branches, loops) and data flow (i.e. tensors and the operations performed on them). The resolution of the control flow is handled by Python and optimized  $\mathrm{C + + }$  code executed on the host CPU, and result in a linear sequence of operator invocations on the device. Operators can be run either on CPU or on GPU.
>  PyTorch 严格分离控制流 (程序分支、循环) 和数据流 (张量以及在张量上执行的计算)
>  控制流的解析由 Python 处理，优化的 C++ 代码会在主机 CPU 上执行，得到一系列在设备上的算子调用
>  算子可以在 CPU 上运行，也可以在 GPU 上运行

PyTorch is designed to execute operators asynchronously on GPU by leveraging the CUDA stream mechanism [38] to queue CUDA kernel invocations to the GPUs hardware FIFO. This allows the system to overlap the execution of Python code on CPU with tensor operators on GPU. Because the tensor operations usually take a significant amount of time, this lets us saturate the GPU and reach peak performance even in an interpreted language with fairly high overhead like Python. Note that this mechanism is nearly invisible to the user. Unless they implement their own multi-stream primitives all of the CPU-GPU synchronization is handled by the library.
>  PyTorch 利用 CUDA stream 机制，将 CUDA kernel 调用排队到 GPU 硬件 FIFO 中，以在 GPU 上异步执行算子
>  这使得系统可以重叠 CPU 上 Python 代码的执行以及 GPU 上张量计算的执行
>  因为张量计算通常需要更长时间，这使得即便是在 Python 具有较高开销的解释性语言中，我们也可以充分利用 GPU 并达到峰值性能
>  这种机制对于用户是不可见的，所有的 CPU-GPU 同步操作都由库处理

PyTorch could leverage a similar mechanism to also execute operators asynchronously on the CPU. However the costs of cross-thread communication and synchronization would negate the performance benefit of such an optimization.

## 5.3 Custom caching tensor allocator
Almost every operator must dynamically allocate an output tensor to hold the result of its execution. It is therefore critical to optimize the speed of the dynamic memory allocators. PyTorch can rely on optimized libraries [39-41] to handle this task on CPU. However, on GPU the cudaFree routine may block its caller until all previously queued work on all GPUs completes. To avoid this bottleneck, PyTorch implements a custom allocator which incrementally builds up a cache of CUDA memory and reassigns it to later allocations without further use of CUDA APIs. The incremental allocation is also crucial for better interoperability, because taking up all GPU memory ahead of time would prevent the user from utilizing other GPU-enabled Python packages.
>  几乎所有算子都需要动态分配输出张量来保存执行结果，因此优化动态内存分配器的速度至关重要
>  PyTorch 可以依赖优化的库来处理 CPU 上的这个任务
>  但在 GPU 上，`cudaFree` 会阻塞调用者直到 GPU 上所有排队的任务完成，为了避免这个瓶颈，PyTorch 执行了一个自定义的分配器，它逐步构建 CUDA 显存的一个缓存，并在后续分配时重新使用这些内存，而无需再调用 CUDA API
>  这种渐进式分配对于更好的互操作性也非常重要，因为提前占用全部 GPU 内存会组织用户使用其他 GPU 支持的 Python 包

To further improve its effectiveness, this allocator was tuned for the specific memory usage patterns of deep learning. For example, it rounds up allocations to multiples of 512 bytes to avoid fragmentation issues. Moreover, it maintains a distinct pool of memory for every CUDA stream (work queue).
>  为了提高效率，该分配器针对 DL 的内存使用模式进行了优化
>  例如，它会将分配向上取整为 512 的倍数以避免碎片化，此外，它为每个 CUDA stream (工作队列) 维护了一个独立的显存池

The one-pool-per-stream design assumption simplifies the implementation and improves the performance of the allocator: because the CPU runs ahead of the GPU, memory is freed on the CPU before its last use on the GPU finishes. 
>  每个流对应一个显存池的假设简化了实现 (不同 stream 之间的显存管理互不干扰，避免了锁竞争问题) 并提高了分配器的性能: 由于 CPU 的执行速度快于 GPU (在异步执行下，CPU 提交了 kernel 之后就会执行下一条语句)，因此可能 GPU 还正在用某一块显存，CPU 已经发出了释放显存的指令 

>  例如

```python
x = torch.randn(1000, 1000).cuda()        # 分配显存
y = x * 2                                 # 提交到 stream，GPU 开始计算
del x                                     # CPU 立刻释放 x 的显存
z = torch.zeros(1000, 1000).cuda()        # 分配新显存 —— 可以复用 x 的？
```

>  其中 GPU 可能还没算完 `y = x * 2`，CPU 就发出了 `del x`
>  当然这个释放不是立即释放，实际的物理释放需要等到 `del x` 之前的发出的 GPU 指令都执行完才可以执行

Since streams serialize execution, if the free precedes the reallocation on the CPU, the same order will occur on the GPU.
>  因为 stream 是串行执行指令，因此如果 CPU 发出的显存释放是在显存重分配指令之前，GPU 也会以相同的顺序先释放显存再重分配显存

So the allocator can reallocate memory freed on the CPU immediately as long as the new allocation is used on the same stream as the freed region. However, if an allocation was last used on one stream and then allocated on another, additional synchronization is needed.
 >  这样，PyTorch 的 allocator 可以立刻将之前释放的显存重分配给相同的流上新分配的显存使用 (实际没有调用 `cudaMalloc, cudaFree`，就是逻辑上的重复呢配)
>   但如果分配的显存最后在 stream A 使用，释放后被分配给 stream B，就需要额外的同步，因为 stream A 和 stream B 可能并行执行，不能让 stream B 在 stream A 完成执行之前就占用这块显存，否则就出现了数据竞争

The one-pool-per-stream design seems limiting since the allocations end up fragmented per stream, but in practice PyTorch almost never uses multiple streams. It is notoriously hard to write CUDA kernels in a way that would let them cooperatively share the GPU because exact scheduling is hardware controlled. In practice, kernel writers usually resort to monolithic kernels that combine multiple tasks. 
>  每个 stream 使用一个显存池的设计似乎存在局限，因为每个为每个 stream 分配显存池会导致碎片化
>  但实际使用中 PyTorch 几乎不会使用多个 stream，因为非常难以编写协同共享 GPU 的 CUDA kernels，因为实际的调度都是硬件控制的，我们无法控制哪个 kernel 先执行，它们是否并行，以及资源的分配
>  实际上，kernel 开发者会将多个任务合并为一个 kernel

Data loading and distributed computing utilities are exceptions to the one stream design, and they carefully insert additional synchronization to avoid bad interactions with the allocator.
>  虽然大多数计算都是在一个 stream 上，但有两个例外: 
>  - Data Loading: 数据从 CPU 到 GPU 的过程是异步的，通常会使用单独的 stream 来重叠数据传输和计算
>  - Distributed Computing: 例如 NCCL 进行 all-reduce，NCCL 通常在自己的 stream 中运行，避免阻塞主计算 stream
>  故在这些 stream 中需要插入额外的同步操作，避免和 allocator 发生数据竞争

While this design is susceptible to certain corner cases, it almost never exhibits unwanted behaviors in practical code. Most of our users are not aware of its existence.

## 5.4 Multiprocessing
Due to the global interpreter lock (GIL) Python's default implementation does not allow concurrent threads to execute in parallel. To alleviate this problem, the Python community has established a standard multiprocessing module, containing a number of utilities that allow users to easily spawn child processes and implement basic inter-process communication primitives.
>  由于全局解释器锁，Python 的默认实现不允许并发线程并行执行
>  为了解决这个问题，Python 社区建立了标准的 multiprocessing 模块，其中包含一系列工具，使用户可以轻松生成子进程，并实现基本的进程间通信原语

However, the implementation of the primitives uses the same form of serialization used for on-disk persistence, which is inefficient when dealing with large arrays. 
>  但这些原语的实现使用了和磁盘持久化相同的序列化方法，处理大型数组时效率较低 (数据从一个进程传递到另一个进程时，使用的是 pickle 序列化机制)

>  这个机制的过程为:
>  原始张量在进程 A 的内存中, `pickle` 会先将整个张量序列化为字节流 (拷贝一份)，通过 IPC (进程间通信通道) 发送字节流，接收端再反序列化该字节流 (再拷贝一份)，这样流程 (序列化 + 序列化) 内存占比高，且耗时

Hence, PyTorch extends the Python multiprocessing module into torch. multiprocessing, which is a drop-in replacement for the built in package and automatically moves the data of tensors sent to other processes to shared memory instead of sending it over the communication channel.
>  PyTorch 拓展了 Python 的 multiprocessing 模块 (`torch.multiprocessing`)，它是 `multiprocessing` 的即插即用的替代品，自动将发送给其他进程的张量数据移动到共享内存中，而不是通信通道传输

>  也就是进程 A 将数据直接放在共享内存中，进程 B 通过指针访问，没有复制和序列化反序列化以及通信的开销，但需要注意进程间同步

This design greatly improves performance and makes the process isolation weaker, resulting in a programming model which more closely resembles regular threaded programs. Users can easily implement heavily parallel programs that operate on independent GPUs but later synchronize gradients using all-reduce style primitives.
>  这个设计显著提高了性能，并弱化了进程隔离，使得编程模型更接近常规的多线程程序
>  用户可以轻松实现在 GPU 上运行的高并行程序，之后再通过 all-reduce 类型的原语同步梯度

Another unique feature of this system is that it transparently handles sharing of CUDA tensors, making it easy to implement techniques like Hogwild [42].

## 5.5 Reference counting
Users often design their models to utilize all memory available during training, and increasing batch sizes is a common technique of speeding up the process. Therefore, to deliver great performance, PyTorch has to treat memory as a scarce resource that it needs to manage carefully.

Libraries with eager semantics have to manage tensor memory without knowing how it will be used in the future. Garbage collection is the typical way to handle this automatically because it has good amortized performance. In this approach, the runtime periodically investigates the state of the system, enumerates used objects and frees everything else. However, by deferring the deallocation, it causes the program to use more memory overall [43]. Given the scarcity of GPU memory, these overheads are unacceptable. In fact, Torch7 utilized the garbage collector built into Lua, and a common antipattern among the users was to sprinkle the program with explicit triggers to the garbage collector, hoping that the memory errors go away.
>  具有立即执行语义的库需要在无法预知张量未来如何被使用的情况下管理张量内存
>  垃圾收集是自动处理这类问题通常的方式，因为它具有良好的摊销性能
>  在 GC 中，运行时会定期审查系统状态，遍历所有对象，找出哪些对象还被引用，并释放所有没有被引用的对象
>  使用 GC 会使得内存不是即时释放而是被推迟释放，会导致程序整体上占用更多的内存

PyTorch takes a different approach: it relies on a reference counting scheme to track the number of uses of each tensor, and frees the underlying memory immediately once this count reaches zero. Note that PyTorch tracks both references internal to the libtorch library and external references made by users in their Python code by integrating with Python's own reference counting mechanism. This ensures that memory is released exactly when tensors become unneeded.
>  PyTorch 采用了不同的方式: 它依赖于引用计数机制来跟踪每个张量的使用次数，并在该计数变为零时立即释放其内存
>  PyTorch 既追踪 libtorch 库内部的引用，也通过和 Python 自身的引用计数机制继承，追踪用户在 Python 代码中创建的外部引用，这确保了张量的内存能在不再需要时就被立即释放 (libtorch 内追踪了 tensor 在计算图内的引用，Python 层则追踪 tensor 对象的引用，PyTorch 感知到 Python 上对 tensor 对象的引用增加和减少，例如赋值和删除，并同步更新 libtorch 层的计数)

One notable caveat is that we can only guarantee the desired performance characteristics in implementations of languages that either already utilize reference counting (CPython, Swift, but not PyPy or many scripting languages such as Lua), and those that allow for user-defined behavior for assignment, copies, and moves (e.g.  C++  , Rust). Bindings to implementations that do not satisfy those criteria will have to implement their own specialized memory management on top of PyTorch.
>  一个值得注意的例外是: 我们只能在那些已经使用引用计数的语言实现中保证预期的性能，以及那些允许用户自定义赋值、复制和移动行为的语言中保证预期性能

# 6 Evaluation
In this section we compare the performance of PyTorch with several other commonly-used deep learning libraries, and find that it achieves competitive performance across a range of tasks. All experiments were performed on a workstation with two Intel Xeon E5-2698 v4 CPUs and one NVIDIA Quadro GP100 GPU.

## 6.1 Asynchronous dataflow
We start by quantifying the ability of PyTorch to asynchronously execute dataflow on GPU. We use the built-in profiler [44] to instrument various benchmarks and record a timeline of the execution of a single training step.

Figure 1 shows a representative timeline of execution for the first few operations of a ResNet-50 model. The host CPU which queues the work quickly outpaces the execution of the operators on the GPU. This allows PyTorch to achieve almost perfect device utilization. In this example, GPU execution takes around three times longer than CPU scheduling. The exact ratio depends on the relative performance of the host CPU and the GPU, as well as the number of elements in each tensor and the average arithmetic complexity of the floating point computations to be performed on the GPU.
>  主机 CPU 队列的速度远远快于 GPU 上 operators 的执行速度，这使得 PyTorch 能够实现几乎完美的设备利用率 (始终保持 GPU 忙碌)

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/af6f2a7d-a6c9-4cbe-b989-9c6b18fdac65/6fba9a2ab4dadf70ca661be48bcd687dd4d243f39c485619e3b9c2b4d67fd8d0.jpg)  

Figure 1: A trace of the first few operators of Resnet-50. The top row depicts the execution of the control flow running on the host CPU. The gray areas are Python code executed by its interpreter. The colored areas correspond to the work done on the host CPU to queue various operators (convolution, batch normalization, and so on). The bottom row shows the corresponding execution of those operators on the GPU. The arrows pair the two events in time.

## 6.2 Memory management
We used the NVIDIA profiler to trace the execution of the CUDA runtime as well as the execution of the CUDA kernels launched during one training iteration of the ResNet-50 model. As shown in Figure 2, the behavior of the first iteration differs significantly from that of subsequent ones. At first, calls to the CUDA memory management functions (cudaMalloc and cudaFree) slow down the execution quite dramatically by blocking the CPU thread for long periods of time, hence lowering the utilization of the GPU. This effect disappears in subsequent iterations as the PyTorch caching memory allocator starts reusing previously allocated regions.
>  第一次迭代和后续迭代的内存行为有显著不同
>  最初，对 CUDA 内存管理函数的调用会显著减慢执行速度因为这些调用会阻塞 CPU 线程较长时间，降低 GPU 利用率
>  在后续迭代中，这种影响会消失，因为 PyTorch 的缓存内存分配器开始重用之前分配的内存区域

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/af6f2a7d-a6c9-4cbe-b989-9c6b18fdac65/75ff9e53e9eecf00a0718caf3b73d751406b22970a670e9cb95efad2a2c4f279.jpg)  

Figure 2: Annotated traces of the execution of ResNet-50 on GPU.

## 6.3 Benchmarks
Finally, we can get an overall sense of single-machine eager mode performance of PyTorch by comparing it to three popular graph-based deep learning frameworks (CNTK, MXNet and TensorFlow), a define-by-run framework (Chainer), and production oriented platform (PaddlePaddle). The Appendix details all the steps needed to reproduce our setup.

Our results are summarized in Table 1. On all the benchmarks, the performance of PyTorch is within  $17\%$  of that of the fastest framework. We attribute this result to the fact that these tools offload most of the computation to the same version of the cuDNN and cuBLAS libraries.

Table 1: Training speed for 6 models using 32bit floats. Throughput is measured in images per second for the AlexNet, VGG-19, ResNet-50, and MobileNet models, in tokens per second for the GNMTv2 model, and in samples per second for the NCF model. The fastest speed for each model is shown in bold.  

<table><tr><td rowspan="2">Framework</td><td colspan="6">Throwinput (higher is better)</td></tr><tr><td>AlexNet</td><td>VGG-19</td><td>ResNet-50</td><td>MobileNet</td><td>GNMTv2</td><td>NCF</td></tr><tr><td>Chainer</td><td>778 ± 15</td><td>N/A</td><td>219 ± 1</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>CNTK</td><td>845 ± 8</td><td>84 ± 3</td><td>210 ± 1</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MXNet</td><td>1554 ± 22</td><td>113 ± 1</td><td>218 ± 2</td><td>444 ± 2</td><td>N/A</td><td>N/A</td></tr><tr><td>PaddlePaddle</td><td>933 ± 123</td><td>112 ± 2</td><td>192 ± 4</td><td>557 ± 24</td><td>N/A</td><td>N/A</td></tr><tr><td>TensorFlow</td><td>1422 ± 27</td><td>66 ± 2</td><td>200 ± 1</td><td>216 ± 15</td><td>9631 ± 1.3%</td><td>4.866 ± 2.9%</td></tr><tr><td>PyTorch</td><td>1547 ± 316</td><td>119 ± 1</td><td>212 ± 2</td><td>463 ± 17</td><td>15512 ± 4.8%</td><td>5.466 ± 3.4%</td></tr></table>

## 6.4 Adoption
The validity of design decisions and their impact on ease-of-use is hard to measure. As a proxy, we tried to quantify how well the machine learning community received PyTorch by counting how often various machine learning tools (including Caffe, Chainer, CNTK, Keras, MXNet, PyTorch, TensorFlow, and Theano) are mentioned on arXiv e-Prints since the initial release of PyTorch in January 2017. In Figure 3 we report the monthly number of mentions of the word "PyTorch" as a percentage of all mentions among these deep learning frameworks. We counted tools mentioned multiple times in a given paper only once, and made the search case insensitive to account for various spellings.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/af6f2a7d-a6c9-4cbe-b989-9c6b18fdac65/8322c519f4d0f0f90ce0cccdbff66fc3c8ded3f8dd3d49d4ce4bdc57ac5e220b.jpg)  

Figure 3: Among arXiv papers each month that mention common deep learning frameworks, percentage of them that mention PyTorch.

# 7 Conclusion and future work
PyTorch has become a popular tool in the deep learning research community by combining a focus on usability with careful performance considerations. In addition to continuing to support the latest trends and advances in deep learning, in the future we plan to continue to improve the speed and scalability of PyTorch. Most notably, we are working on the PyTorch JIT: a suite of tools that allow PyTorch programs to be executed outside of the Python interpreter where they can be further optimized. We also intend to improve support for distributed computation by providing efficient primitives for data parallelism as well as a Pythonic library for model parallelism based around remote procedure calls.
>  PyTOrch 是将易用性和性能优化结合的 DL 工具
>  我们未来将持续提升 PyTorch 的速度和可拓展性，其中最显著的进展是我们在开发 PyTorch JIT: 一套允许 PyTorch 程序在 Python 解释器之外执行的工具，从而可以进一步进行优化
>  我们还计划通过提供高效的数据并行原语以及基于远程过程调用的模型并行 Python 库，来增强对分布式计算的支持


