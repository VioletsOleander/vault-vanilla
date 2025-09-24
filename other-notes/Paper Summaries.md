# 2025
## September
### Week 1
**ORCA**: [[paper-notes/gen-ai/language/infra/system/Orca A Distributed Serving System for Transformer-Based Generative Models-2022-OSDI|Orca A Distributed Serving System for Transformer-Based Generative Models-2022-OSDI]]

两个关键技术: iteration-level scheduling, selective batching
iteration-level scheduling 即每次推理运行只执行整个 batch 的一个 iteration，而不是将 batch 中所有的 requests 都推理完，这提高了 batching requests 的灵活性，解决了 early finishing, late arriving 的问题，可以显著降低推理 latency

selective batching 则是为了在 iteration-level scheduling 的基础上能够灵活 batching 而提出的，因为使用 iteration-level scheduling 之后，同一个 batch 内的 requests 就不能保证处于相同阶段 (prefill or decode)，或者 decode 相同位置的 token，这使得不同 requests 的 Attention 计算要处理的 tensor shape 不同，进而不能像传统的批处理一样，通过堆叠相同 shape 的 tensor 为单个大 tensor，传递给 batch operator 来实现批处理

为此，selective batching 选择不对 Attention 计算执行 batching，而其他的计算例如 Linear, GeLU 则仍然可以通过堆叠 flattened tensor 来执行批处理 (因为 Attention 由 request 的 concept 需要读取同一个 request 的 KVCache 对该 request 的 token 进行 decode，其他计算没有)
分别处理 Attention 并不会太大地影响效率，因为 Attention 计算不需要模型参数，因此批处理带来地减少参数 load 的作用对 Attention 计算没有意义

ORCA system 的 scheduler 执行 iteration-level scheduling, engine 对收到的 batch 执行 selective batching 计算

ORCA 也采用了层内 + 层间并行 (张量并行 + 流水线并行)，并且分离了控制信息和数据信息的传输 (控制信息在 CPU 之间通过 gRPC，数据信息在 GPU 之间通过 NCCL)

**Transformers Libraray**: [[paper-notes/gen-ai/language/infra/library/Transformers State-of-the-Art Natural Language Processing-2020-EMNLP|Transformers State-of-the-Art Natural Language Processing-2020-EMNLP]]

这是 HuggingFace 开发的 Python 库，它解决的问题就是为市面上多种多样的预训练模型提供一个统一的调用接口

在该库中，每个市面上的预训练模型都封装为一个类，这些类可以通过会尽量具备相同的接口

这个库将一个 Transformer 模型的实例化分为三个部分: tokenizer, model, head
tokenizer 将原始文本转化为索引向量
model 将索引向量输入，得到语义嵌入
head 将语义嵌入转化为有意义的输出

tokenizer 实现为单独的库
model 实现为 Python 类
head 实现为对 model 类的封装类

### Week 2
**PyTorch**: [[paper-notes/ml/system/PyTorch An Imperative Style, High-Performance Deep Learning Library-NeurIPS-2019|PyTorch An Imperative Style, High-Performance Deep Learning Library-NeurIPS-2019]]

在用户体验方面:
PyTorch 的设计原则是用户体验优先，为了提高 API 的易用性，并和 Python 生态融合，PyTorch 采用 define-by-run/eager 的执行模式，也就类似于 Python 的解释型执行

之前框架的静态数据流图方法就类似于编译型语言的编译型执行，要求使用提供的图 API 构造完整的计算图，编译后再执行模型

这种动态的执行模式很大程度提高了 PyTorch 的易用性和灵活性，PyTorch 分离了控制流和数据流，Python 的控制流语句由 CPython 处理，实际的 kernel 执行则调用底层的 C++/CUDA 库，将计算发布到设备，也就是数据流和控制流的分离

PyTorch 将深度学习中的组件例如模型、层、优化器、参数、数据集、数据加载器等都封装为 Python 类，实现自定义的这些东西就是继承这些类，实现对应的方法

在性能方面:
PyTorch 依赖于 libtorch 中编写的高性能算子执行计算，内存分配使用自己的分配器来维护内存/显存池，避免频繁调用底层 API

PyTorch 也在开发自己的 JIT 编译器以允许 PyTorch 程序在 Python 解释器之外运行，以引入一定的图编译提高性能，也就是 `torch.compile`

### Week 3
**PyTorch 2:** [[paper-notes/compilation/PyTorch 2 Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation-2024-ASPLOS|PyTorch 2 Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation-2024-ASPLOS]]

PyTorch 引入的新特性为 `torch.compile`，该特性由 TorchDynamo 和 TorchInductor 支持

TorchDynamo 是 PyTorch 引入的抓图工具，其工作机制是自定义 `eval_frame` 函数，重新解释 Python 帧中的字节码，将帧中的 PyTorch 操作提取为 FX 图
TorchDynamo 本质还是采用 record/replay 的方式，但是它的层级比 Lazy Tensor 要高，TorchDynamo 是在 Python 字节码的层级执行 record，而 Lazy Tensor 只能在 C++ dispatcher 的层级执行 record

TorchInductor 为 PyTorch 引入的图编译器，它接收 FX 图，将图下降到 TorchIndoctur IR，然后生成 Triton 或 C++ 代码，TorchInductor 执行的优化主要是: 分解 (将 PyTorch 操作分解为 ATen primitives)，融合 (融合分解后的 ATen primitives)

**FSDP:** [[paper-notes/gen-ai/language/infra/system/PyTorch FSDP Expreiences on Scaling Fully Sharded Data Parallel-2023-VLDB|PyTorch FSDP Expreiences on Scaling Fully Sharded Data Parallel-2023-VLDB]]

FSDP 基本思想和 ZeRO 一致，即本质是对基础 DP 的增强，shard 参数、优化器状态、梯度

FSDP 和 ZeRO 思想上有差异的一点在于 FSDP 还会额外将模型划分为多个 units，每次 unshard 单个 unit 的参数，保持其他 units 的参数 sharded

(前向传播和反向传播时) 连续 unit 的参数 unshard 通信和计算可以进行重叠
(反向传播时) 连续 unit 的梯度 shard 规约和计算可以进行重叠

示例:

```
前向传播:
| unit1 计算
| unit2 参数 unshard
|                      unit2 计算
|                      unit3 参数 unshard
```

```
反向传播:
| unit3 计算
| unit2 参数 unshard
|                   unit2 计算
|                   unit3 梯度规约 + unit1 参数 unshard
|                                                     unit1 计算
|                                                     unit2 梯度规约
```

混合 shard 结合了 sharding 和 replication，每个 sharding group 内进行 sharding，各个 sharding groups 之间则进行 replication，本质是两层 DP

FSDP + Pipeline Parallelism: 用 FSDP 封装每个流水线阶段
FSDP + Tensor Parallelism: 2D mesh，一个维度为 PyTorch 分布式张量 `DTensor`，另一个维度为 FSDP

### Week 4
Ray: [[paper-notes/ml/system/Ray A Distributed Framework for Emerging AI Applications-2018-OSDI|Ray A Distributed Framework for Emerging AI Applications-2018-OSDI]]

Ray 应该是第一个完全针对 RL workload 的分布式系统

RL workload 主要包含三个: training, serving, simulation，在纯推理场景下，仅涉及 serving，生成 rollout，在纯训练场景下，serving 和 simulation 负责 rollout 及其奖励的生成，training 接受这些信息执行梯度下降

Ray 把这些 workload 所涉及的计算分为两类: 有状态计算和无状态计算，并用 Task, Actor 来抽象地表示这两个概念
顾名思义，二者的差异就在于是否需要跨计算维护共享状态
Task 本质是一个纯函数式的远程调用函数，Actor 则是一个类，其方法是远程调用函数，方法之间共享类维护的状态

Ray 的 API 层
Task 和 Actor 的方法调用都是异步执行，返回 future, future 可以作为参数传递给其他 Task, Actor，这种数据的依赖构造出了 Ray 的动态任务计算图，用户使用 Ray API 编写的程序，都会 just-in-time 构造出计算图，然后交由 scheduler 来调度实际计算

因为实际计算都是纯函数式，因此计算满足幂等，Ray 在执行计算图的时候会存储图中每个数据的计算血缘 (在另外的对象存储中)，借由计算血缘实现计算的容错性

Ray 的系统层
全局控制状态存储 (由 chain replication + redis 实现) 存储各个任务的元数据，元数据的容错机制由 chain replication 实现；调度器负责调度任务到实际节点，调度时优先由本地调度器处理，其次是全局调度器；存内对象存储负责 Actor 状态的存储

SGLang: [[paper-notes/gen-ai/language/infra/system/SGLang Efficient Execution of Structured Language Modeling Programs-2024-NeurIPS|SGLang Efficient Execution of Structured Language Modeling Programs-2024-NeurIPS]]

SGLang 包含 1. 为用户提供的 API 层，用于构造 LM program 2. 支持 API 运行的 runtime 层

Runtime 层的三个优化: 1. 针对 KV cache 前缀复用的 RadixAttention 2. 针对结构化输出的压缩 FSM 3. 针对 API 调度的推测解码

RadixAttention 为所有请求持续维护 radix tree，实现跨所有请求最大限度的 KV cache prefix 复用
为了提高缓存命中率，SGLang 没有采用 FCFS 调度，而是根据 requests 的前缀和当前 radix tree 的匹配程度进行排序和有效调度，即 cache-aware 调度
