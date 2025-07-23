---
completed: true
version: "3.10"
---
# Basic Concepts
## Introduction
In this page, an outline of the oneDNN programming model is presented, and the key concepts are discussed, including Primitives, Engines, Streams, and Memory Objects. 

In essence, the oneDNN programming model consists in executing one or several primitives to process data in one or several memory objects. The execution is performed on an engine in the context of a stream. The relationship between these entities is briefly presented in Figure 1, which also includes additional concepts relevant to the oneDNN programming model, such as primitive attributes and descriptors. These concepts are described below in much more details.
>  本质上，oneDNN 的编程模型是通过一个或多个原语来执行一个或多个内存对象中的数据
>  执行是在某个引擎的上下文中通过流进行的
>  这些实体的关系如 Figure 1 所示，该图还包含与 oneDNN 编程模型相关的其他概念，例如原语属性和描述符

![Figure 1: Overview of oneDNN programming model. Blue rectangles denote oneDNN objects, and red lines denote dependencies between objects.](https://uxlfoundation.github.io/oneDNN/_images/img_programming_model.png)

### Primitives
oneDNN is built around the notion of a primitive ([dnnl::primitive](https://uxlfoundation.github.io/oneDNN/struct_dnnl_primitive-2.html#doxid-structdnnl-1-1primitive)). A primitive is an object that encapsulates a particular computation such as forward convolution, backward LSTM computations, or a data transformation operation. Additionally, using primitive attributes ([dnnl::primitive_attr](https://uxlfoundation.github.io/oneDNN/struct_dnnl_primitive_attr-2.html#doxid-structdnnl-1-1primitive-attr)) certain primitives can represent more complex fused computations such as a forward convolution followed by a ReLU.
>  oneDNN 是围绕原语 (`dnnl::primitive`) 的概念构建的
>  原语是一个封装了特定计算的对象，例如前向卷积，反向 LSTM 计算或数据转换操作
>  此外，通过使用原语属性 (`dnnl::primitive_attr`)，某些原语可以表示更复杂的融合计算，例如先进行前向卷积再接 ReLU 操作

The most important difference between a primitive and a pure function is that a primitive can store state.
>  原语和纯函数之间最重要的区别在于原语可以存储状态

One part of the primitive’s state is immutable. 
>  oneDNN 中，一个原语一旦被创建，它的某些配置参数就不可变的，称为不可变状态，这些不可变状态一般涉及张量形状，卷积核大小，数据类型等

For example, convolution primitives store parameters like tensor shapes and can pre-compute other dependent parameters like cache blocking. This approach allows oneDNN primitives to pre-generate code specifically tailored for the operation to be performed. 
>  例如，卷积原语会存储张量形状等参数，并可以预先计算其他依赖参数，如缓存块大小
>  这种方法使得 oneDNN 原语可以针对要执行的特定操作预先生成代码
>  也就是说，oneDNN 利用了这些不可变参数，在原语创建时执行了一些预计算，以提升后续执行效率，例如我们要运行一个卷积运算，oneDNN 如果知道输入格式为 `NCHW`，输入尺寸为 `(1, 3, 224, 224)`，它就可以提前确定
>  - 内存如何布局最高效
>  - 数据如何重排成内部优化格式，例如 `nChw16c`
>  - 进行缓存分块——将大矩阵拆成适合 CPU 缓存的小块，避免频繁访问主存

The oneDNN programming model assumes that the time it takes to perform the pre-computations is amortized by reusing the same primitive to perform computations multiple times.
>  oneDNN 编程模型假设执行预先计算所花费的时间可以通过多次重复使用同一原语进行计算来分摊
>  类比于创建 primitive 是花 5 分钟磨刀，执行计算多次是用快刀砍 100 根木头，这样总体效率就远高于每次都用慢刀

>  因此，oneDNN 的哲学就是一次配置，多次运行，以初始化代价换运行速度

The mutable part of the primitive’s state is referred to as a scratchpad. It is a memory buffer that a primitive may use for temporary storage only during computations. The scratchpad can either be owned by a primitive object (which makes that object non-thread safe) or be an execution-time parameter.
>  原语的可变状态称为 “临时缓冲区”
>  它是一个原语在计算过程中可能用于临时存储的内存缓冲区
>  临时缓冲区可以由一个原语对象自身拥有，也可以作为执行时的参数传递

>  如果临时缓冲区由原语对象拥有，那么这个对象就不是线程安全的，因为如果多个线程同时访问这个对象并修改它的临时缓冲区，就会导致数据不一致或错误
>  如果临时缓冲区作为运行时传入的参数，那么实际上就是每个调用者提供自己申请的临时缓冲区，就避免了多线程下的竞争问题

### Engines
Engines ([dnnl::engine](https://uxlfoundation.github.io/oneDNN/struct_dnnl_engine-2.html#doxid-structdnnl-1-1engine)) is an abstraction of a computational device: a CPU, a specific GPU card in the system, etc. Most primitives are created to execute computations on one specific engine. 
>  引擎 (`dnnl::engine`) 是对一个计算设备的抽象: 例如 CPU、系统中的特定一张 GPU 等
>  大多数原语都是为了在某个特定的引擎上执行而创建的

The only exceptions are reorder primitives that transfer data between two different engines.
>  唯一的例外是 reorder 原语，这个原语用于在两个不同的引擎间传递数据

### Streams
Streams ([dnnl::stream](https://uxlfoundation.github.io/oneDNN/struct_dnnl_stream-2.html#doxid-structdnnl-1-1stream)) encapsulate execution context tied to a particular engine. For example, they can correspond to OpenCL command queues.
>  流 (`dnnl::stream`) 封装了与特定引擎执行相关的上下文，用于控制操作的执行顺序、同步、资源管理等
>  例如，它们可以对应于 OpenCL 命令队列

>  之所以叫 “流”，是为了接近 “数据流” 和 “执行流” 的概念，这里的 `dnnl::stream` 的概念更接近 “执行流”

### Memory Objects
Memory objects ([dnnl::memory](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory)) encapsulate handles to memory allocated on a specific engine, tensor dimensions, data type, and memory format – the way tensor indices map to offsets in linear memory space. Memory objects are passed to primitives during execution.
>  内存对象 (`dnnl::memory`) 封装了在特定引擎上分配的内存句柄、张量维度、数据类型以及内存格式 (内存格式反映了即张量索引如何映射到线性内存空间中的偏移量)
>  在执行过程中，内存对象会被传递给原语

## Levels of Abstraction
Conceptually, oneDNN has multiple levels of abstractions for primitives and memory objects in order to expose maximum flexibility to its users.

- Memory descriptors ([dnnl_memory_desc_t](https://uxlfoundation.github.io/oneDNN/group_dnnl_api_memory.html#doxid-group-dnnl-api-memory-1gad281fd59c474d46a60f9b3a165e9374f), [dnnl::memory::desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory_desc-2.html#doxid-structdnnl-1-1memory-1-1desc)) define a tensor’s logical dimensions, data type, and the format in which the data is laid out in memory. The special format any ([dnnl::memory::format_tag::any](https://uxlfoundation.github.io/oneDNN/enum_dnnl_memory_format_tag.html#doxid-structdnnl-1-1memory-1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec)) indicates that the actual format will be defined later (see [Memory Format Propagation](https://uxlfoundation.github.io/oneDNN/page_memory_format_propagation_cpp.html#doxid-memory-format-propagation-cpp)).
- Primitives descriptors fully define an operations’s computation using the memory descriptors ([dnnl_memory_desc_t](https://uxlfoundation.github.io/oneDNN/group_dnnl_api_memory.html#doxid-group-dnnl-api-memory-1gad281fd59c474d46a60f9b3a165e9374f), [dnnl::memory::desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory_desc-2.html#doxid-structdnnl-1-1memory-1-1desc)) passed at construction, as well as the attributes. They also dispatch specific implementation based on the engine. Primitive descriptors can be used to query various primitive implementation details and, for example, to implement [Memory Format Propagation](https://uxlfoundation.github.io/oneDNN/page_memory_format_propagation_cpp.html#doxid-memory-format-propagation-cpp) by inspecting expected memory formats via queries without having to fully instantiate a primitive. oneDNN may contain multiple implementations for the same primitive that can be used to perform the same particular computation. Primitive descriptors allow one-way iteration which allows inspecting multiple implementations. The library is expected to order the implementations from the most to least preferred, so it should always be safe to use the one that is chosen by default.
- Primitives, which are the most concrete, and embody the actual executable code that will be run to perform the primitive computation.

>  oneDNN 为原语和内存对象提供了多层抽象

>  (1) 内存描述符 (`dnnl_memory_desc_t`, `dnnl::memory::desc`) 定义了张量的逻辑维度、数据类型和数据在内存中的布局格式
>  特殊格式 `any` (`dnnl::memory::format_tag::any`) 表示实际的格式将在稍后定义 (通过内存格式传播之后定义)

>  内存描述符侧重 “描述”，它表示一个张量的逻辑结构，包括数据类型、维度、内存布局等信息

>  (2) 原语描述符使用在构造时传入的内存描述符 (以及属性) 定义了一个操作的计算
>  原语描述符还会根据引擎选择特定的实现
>  原语描述符可以用于查询各种原语实现的细节，例如通过检查期望的内存格式来实现内存格式传播，而无需完全实例化一个原语
>  oneDNN 可能包含对相同原语的多个不同实现，这些实现执行的是相同的特定计算
>  原语描述符允许对这些实现进行遍历，oneDNN 库将会对这些实现进行排序，因此默认选择的实现通常都是最好的

>  原语描述符同样侧重 “描述”，它是对某个计算操作 (例如卷积) 的描述性定义，包含了操作类型、算法、输入输出的内存描述符、属性、引擎等信息
>  原语描述符不执行任何操作，而是用来: 查询可能的实现、获取内存格式信息 (用于格式传播)、获取 scratchpad 内存需求、为后续创建 primitive 做准备

>  (3) 原语是最具体的抽象层次，它们包含了实际可执行的代码，用于执行原语计算

## Creating Memory Objects and Primitives
### Memory Objects
Memory objects are created from the memory descriptors. 
>  内存对象是根据内存描述符创建的

It is not possible to create a memory object from a memory descriptor that has memory format set to [dnnl::memory::format_tag::any](https://uxlfoundation.github.io/oneDNN/enum_dnnl_memory_format_tag.html#doxid-structdnnl-1-1memory-1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec).
>  无法从一个内存格式为 `dnnl::memory::format_tag::any` 的内存描述符创建内存对象

There are two common ways for initializing memory descriptors:

- By using [dnnl::memory::desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory_desc-2.html#doxid-structdnnl-1-1memory-1-1desc) constructors or by extracting a descriptor for a part of a tensor via [dnnl::memory::desc::submemory_desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory_desc-2.html#doxid-structdnnl-1-1memory-1-1desc-1a7de2abef3b34e94c5dfa16e1fc3f3aab)
- By querying an existing primitive descriptor for a memory descriptor corresponding to one of the primitive’s parameters (for example, [dnnl::convolution_forward::primitive_desc::src_desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_convolution_forward_primitive_desc.html#doxid-structdnnl-1-1convolution-forward-1-1primitive-desc-1a585a3809a4f28938e53f901ed103da24)).

>  用两种方式可以用于初始化内存描述符:
>  - 使用 `dnnl::memory::desc` 构造函数，或者通过 `dnnl::memory::desc::submemory_desc` 为张量的某一部分提取描述符
>  - 通过查询现有的描述符，获取与原语某个参数对应的内存描述符 (例如 `dnnl::convolution_forward::primitive_desc::src_desc`)

Memory objects can be created with a user-provided handle (a `void *` on CPU), or without one, in which case the library will allocate storage space on its own.
>  内存对象可以通过一个用户提供的句柄 (CPU 上的 `void *`) 创建，也可以不提供，那么 oneDNN 就会自己分配空间

### Primitives
The sequence of actions to create a primitive is:

1. Create a primitive descriptor via, for example, [dnnl::convolution_forward::primitive_desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_convolution_forward_primitive_desc.html#doxid-structdnnl-1-1convolution-forward-1-1primitive-desc). The primitive descriptor can contain memory descriptors with placeholder [format_tag::any](https://uxlfoundation.github.io/oneDNN/enum_dnnl_memory_format_tag.html#doxid-structdnnl-1-1memory-1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec) memory formats if the primitive supports it.
2. Create a primitive based on the primitive descriptor obtained in step 1.

>  创建一个原语的步骤为:
>  - 创建原语描述符，例如通过 `dnnl::convolution_forward::primitive_desc`，如果原语支持，原语描述符可以包含一个具有 `fortmat_tag::any` 的内存描述符
>  - 基于原语描述符创建原语

## Graph Extension
Graph extension is a high level abstraction in oneDNN that allows you to work with a computation graph instead of individual primitives. This approach allows you to make an operation fusion:

- Transparent: the integration efforts are reduced by abstracting backend-aware fusion logic.
- Scalable: no integration code change is necessary to benefit from new fusion patterns enabled in oneDNN.

>  图拓展是 oneDNN 的高级抽象，它允许我们使用计算图来代替单独的原语进行操作，这种方法可以用于实现操作融合: 
>  - 透明化: 通过抽象后端的融合逻辑，减少了集成工作量
>  - 可拓展性: 无需更改集成代码，也可以利用 oneDNN 的融合模式

The programming model for the graph extension is detailed in the [graph basic concepts section](https://uxlfoundation.github.io/oneDNN/dev_guide_graph_basic_concepts.html#doxid-dev-guide-graph-basic-concepts).

## Micro-kernel Extension
The Micro-kernel API extension (ukernel API) is a low-level abstraction in oneDNN that implements sequential, block-level operations. This abstraction typically allows users to implement custom operations by composing those block-level computations. Users of the ukernel API has full control of the threading and blocking logic, so they can be tailored to their application.
>  Micro-kernel API 拓展是 oneDNN 中的一个低级抽象，用于实现顺序的、块级别的操作
>  Micro-kernel API 还允许用户通过组合这些块级别的计算来实现自定义操作
>  使用 ukernel API 的用户可以完全控制线程和分块逻辑，因此可以根据应用需求进行定制