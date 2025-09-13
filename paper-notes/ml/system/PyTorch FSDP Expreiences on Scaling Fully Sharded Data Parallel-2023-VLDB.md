# Abstract
It is widely acknowledged that large models have the potential to deliver superior performance across a broad range of domains. Despite the remarkable progress made in the field of machine learning systems research, which has enabled the development and exploration of large models, such abilities remain confined to a small group of advanced users and industry leaders, resulting in an implicit technical barrier for the wider community to access and leverage these technologies. 

In this paper, we introduce PyTorch Fully Sharded Data Parallel (FSDP) as an industry-grade solution for large model training. FSDP has been closely co-designed with several key PyTorch core components including Tensor implementation, dispatcher system, and CUDA memory caching allocator, to provide non-intrusive user experiences and high training efficiency. 
>  本文提出 PyTorch 全分片数据并行，这是一个工业级的大模型训练方案
>  FSDP 和几个关键 PyTorch 核心组件进行了协同设计，包括了张量实现、分发系统和 CUDA 内存缓冲分配器，以提供无侵入性的用户体验和高训练效率

Additionally, FSDP natively incorporates a range of techniques and settings to optimize resource utilization across a variety of hardware configurations. 
>  此外，FSDP 原生集成了多种技术和设置，以优化在各种硬件配置下的资源利用率

The experimental results demonstrate that FSDP is capable of achieving comparable performance to Distributed Data Parallel while providing support for significantly larger models with near-linear scalability in terms of TFLOPS.
>  实验表明，FSDP 在保持和分布式数据并行相同的性能下能够显著支持更大的模型，在 TFLOPS 上表现出近线性的可拓展性

# 1 Introduction
The magnitude of neural network models is growing at an unprecedented rate, facilitating breakthroughs across a wide spectrum of domains. Upon inception, the 175-billion-parameter GPT-3 [3] model set a new record for almost all Natural Language Processing tasks. The product applications constructed on top of GPT models [23] have quickly demonstrated their potential to revolutionize the entire industry. Modern large scale recommendation models [19, 33] can reach beyond 1 trillion parameters, replete with rapidly growing dense layer components. These models power applications that serve multi-billions of users every day. As large neural networks continue to push the limits of science and technology, an industry-grade tool to simplify the training of such models with high efficiency would help expedite the progress.

In recent years, the community has introduced and investigated numerous advanced methodologies to enlarge neural network models. Pipeline parallelism [6, 8, 11, 15, 20] partitions a model instance into stages and distributes stages across multiple devices, where activations and gradients are communicated across stage boundaries. Tensor parallelism [9, 21, 31, 32] shards model parameters, conducts partial computation on individual devices and communicates activations at required layer boundaries. Zero-Redundancy parallelism [27, 28, 30] shards parameters as well but communicates parameters on-demand to recover their unsharded form and executes the model as if it were replicated on every device. 
>  流水线并行将模型实例化为 stages 将 stages 分布到多个设备上，激活和梯度在 stage 边界通信
>  张量并行划分模型参数，在单个设备上执行部分计算，在层边界对激活进行通信
>  零冗余并行也划分参数，但会按需通信参数以恢复参数的完整形式

The aforementioned techniques have served as the fundamental building blocks to enable the training of large neural networks across various applications. Nevertheless, two challenges still persist. Firstly, some of these methods are tightly integrated with specific model architectures, which hinder them from being utilized as a generic solution for training large models. Secondly, some of these techniques are built on top of rapidly-evolving internal interfaces of underlying machine learning frameworks, which become vulnerable to changes in framework implementations. Therefore, it is more robust and efficient to have a native solution co-designed with the core functionalities of machine learning frameworks. Additionally, constructing such a solution in a composable and customizable manner could potentially facilitate the community's future innovations as well.
>  仍然存在两个挑战，首先，一些方法和特定模型结构紧密耦合，其次一些方法基于低层的 ML 框架的快速变化的内部架构上构建，容易受到低层框架实现变化的影响
>  更健壮和高效的方式是与 ML 框架的核心功能协同设计一个原生的解决方案，此外，以可组合和可定制的方式构建这样的解决方案也可以促进社区的创新

This paper presents PyTorch [24] Fully Sharded Data Parallel (FSDP), which enables the training of large-scale models by sharding model parameters. The FSDP algorithm is motivated by the ZeroRedundancyOptimizer [27, 28] technique from DeepSpeed but with a revised design and implementation that is aligned with the other components of PyTorch. 
>  本文提出 PyTorch FSDP，通过分片模型参数实现大规模训练
>  FSDP 算法受到 DeepSpeed 的零冗余优化器的启发，但其设计和实现进行了改进以和 PyTorch 其他组件对齐

FSDP breaks down a model instance into smaller units and then flattens and shards all of the parameters within each unit. The sharded parameters are communicated and recovered on-demand before computations, and then they are immediately discarded afterwards. This approach ensures that FSDP only needs to materialize parameters from one unit at a time, which significantly reduces peak memory consumption. 
>  FSDP 将模型实例拆分为更小的单元，并对每个单元内的参数进行展平和分片
>  分片的参数会在计算之前通信以按需恢复，然后立即被丢弃
>  该方法确保 FSDP 只需要实例化/加载一个单元的参数 (materialize 指将参数从存储加载到内存中)，大大减少了峰值内存占用 (不需要加载整个模型的参数)

The design and implementation of FSDP faces the following challenges.

- **User Experience** is critical for achieving broad adoption. When working on prior PyTorch distributed training features such as DistributeDataParallel (DDP) [14], we observed that aligning the user experience of distributed training with that of local training can significantly lower the learning barrier. Techniques like DDP require the model to be replicated on every device, which implies that the entire model can be constructed on the target device. However, although FSDP can easily adopt DDP's API design, large models might not fit into one GPU device and therefore cannot even be initialized efficiently.
>  用户体验: 在之前开发 PyTorch 的分布式数据并行 (DDP) 时，我们发现将分布式训练的用户体验和本地训练的用户体验对齐可以显著降低学习门槛
>  DDP 要求模型复制到每个设备，意味着整个模型要能在目标设备上构建
>  虽然 FSDP 可以使用 DDP 的 API 设计，但由于更大模型无法放入单个 GPU，因此甚至无法进行高效地初始化

- **Hardware Heterogeneity** often exists in modern GPU clusters, whereby interconnects are partitioned into high-bandwidth islands within each machine and low-bandwidth mesh across machines. Additionally, there may be further hierarchical structures at the rack or pod levels. Consequently, the design of FSDP must accommodate such heterogeneity and optimize accordingly.
>  硬件异构型: 机器内高速互联，机器间低速互联，且在机架和 pod (pod 是比机架更高的单位，通常包含多个机架，pod 内的设备通常有高速互联网络，pod 之间则一般通过低带宽，高延迟的网络连接) 层级还有更复杂的层次结构
>  FSDP 的设计需要适应这种异构型

- **Resource Utilization** is usually tightly linked with capital and operational expenditures, especially for companies that depend on large GPU clusters to power their mission-critical systems. To ensure that GPU devices remain fully utilized during distributed training, it is essential to minimize downtime caused by non-computational operations.
>  资源利用: 确保 GPU 设备在分布式训练保持满载，减少非计算操作引起的停机时间

- **Memory Planning** plays a crucial role in large model training. PyTorch makes GPU memory block allocation efficient and transparent through caching. Frequent memory defragmentations can significantly slow down training, which becomes particularly acute when working with large models. In such scenarios, practitioners typically seek to saturate GPU memory as much as possible to accommodate the largest batches or models. However, operating near GPU memory capacity significantly increases the chance to trigger defragmentations.
>  内存规划: PyTorch 通过缓存机制使得 GPU 内存块分配高效且透明
>  频繁的内存碎片会减慢训练，为了尽可能饱和 GPU 内存，以接近 GPU 内存容量运行会显著提高触发内存碎片化的可能性

FSDP tackles the aforementioned challenges through a variety of techniques. Firstly, to improve user experience, FSDP introduces deferred initialization that allows users to create a model instance on a dummy device and record operations invoked during initialization. Then, the model can be initialized and sharded unit by unit by replaying the recorded operations on a real GPU device. With this technique, FSDP can provide similar user experiences as local training, while effectively scaling large models. 
>  为了提高用户体验，FSDP 引入了延迟初始化，允许用户在虚拟设备上创建模型实例并记录在初始化时调用的操作，随后可以通过在真实 GPU 设备上重放记录的操作来初始化模型并逐单元分片
>  通过延迟初始化，FSDP 可以提供和本地训练类似的用户体验

Secondly, FSDP offers configurable sharding strategies that can be customized to match the physical interconnect topology of the cluster to handle hardware heterogeneity. 
>  针对异构集群，FSDP 提供了可配置的分片策略，便于自定义以匹配集群的物理互联拓扑

Thirdly, although parameter sharding design inevitably inserts communications, which might block computations and introduces bubbles during execution, FSDP can squeeze out bubbles using an abundant set of tools to aggressively overlap communication with computation through operation reordering and parameter prefetching. 
>  针对资源利用，虽然参数分片需要插入通信，进而阻塞计算，为执行引入气泡，但 FSDP 可以利用大量工具通过重排操作和参数预取重叠通信和计算

Lastly, FSDP optimizes memory usage by prudently restricting the amount of blocks allocated for inflight unsharded parameters and suspending CPU execution if necessary.
>  针对内存规划，FSDP 通过谨慎限制用于分配给为分片参数块数量并在必要时暂停 CPU 执行来优化内存使用

We evaluated the performance of FSDP on various models including popular language models and recommendation system models, utilizing up to 512 80GB A100 GPUs. The experiments showed that FSDP can achieve similar performance to that of DDP on small models. Beyond that FDSP can facilitate significantly larger models with near-linear scalability in terms of TFLOPS. FSDP is currently a beta feature as of PyTorch 2.0 release, and has been battle-tested by both industrial and research applications.
>  在小模型上，FSDP 可以达到和 DDP 类似的性能
>  对于更大的模型，FSDP 可以实现 TFLOPS 的线性可拓展性

To simplify presentation, the rest of this paper uses FSDP to refer to the techniques in general and FullyShardedDataParallel to denote the Python implementation. The remainder of the paper is organized as follows. Section 2 introduces background on some popular distributed training techniques. Section 3 and Section 4 elaborate system design and implementation details. Evaluations are presented in Section 5. Section 6 surveys related work, and Section 7 discusses topics related to FSDP but falls outside of FSDP core. Finally, Section 8 concludes the paper.

# 2 Background
PyTorch [24] has emerged as a fundamental cornerstone for a plethora of machine learning endeavors. PyTorch stores values in Tensor objects, which are versatile n-dimensional arrays featuring a rich set of data manipulation operations. Every Tensor object has an associated storage that is allocated on a specific device. When Tensors only represent simple transformations such as reshape and split, they can share the same underlying storage. 
>  PyTorch 以 Tensor 对象存储值，每个 Tensor 对象都有关联的存储，存储在特定设备上分配
>  如果 Tensors 仅表示简单的转换例如 reshape, split，它们可以共享底层存储

Each Module describes a transformation from input to output values, and its behavior during the forward pass is specified by its forward member function. Such a module may feature Tensor objects as parameters, with the Linear module being an example that contains both weight and bias parameters. During the forward pass, the Linear module applies these parameters to the input to produce the output by means of multiplication and addition operations, respectively.
>  PyTorch 的每个 Module 描述了从输入到输出值的转换，其前向传播的行为由 `forward` 成员函数指定
>  Module 可以以 Tensor 对象作为参数，在前向过程中，对输入应用参数进行计算

As both the data size and model complexity continue to escalate at a staggering pace, the need for an industry-grade distributed training framework becomes increasingly imperative for applications built on top of PyTorch. This section elucidates the trajectory of PyTorch's distributed training capabilities.

## 2.1 Model Replication
Model replication approaches are designed to tackle high-volume datasets by scaling out and distributing computations across multiple devices. DistributedDataParallel (DDP) [14] is the first end-to-end distributed training feature in PyTorch that falls into this category. DDP's adoption has been extensive, spanning both the academic and industrial domains.
>  Model replication 方法旨在通过 scale out 将计算分布到多个设备上来解决大型数据集的计算
>  DDP 是 PyTorch 的第一个端到端分布式训练特性，属于 Model replication

DDP maintains a model replica on each device and synchronizes gradients through collective AllReduce operations in the backward pass, thereby ensuring model consistency across replicas during training. To expedite training, DDP overlaps gradient communication with backward computation, facilitating concurrent workload executions on diverse resources. However, one conspicuous limitation is that DDP requires all model parameters, gradients, and optimizer states to fit in the memory of one GPU device. Consequently, DDP is inadequate for supporting large models, which are critical for cutting-edge machine learning breakthroughs. For example, when training models with more than one billion parameters using a 40GB GPU device, DDP will likely encounter out-of-memory errors on each device.
>  DDP 在每个设备上都维护一个 model replica，在反向过程中通过 AllReduce 集合通信来同步梯度，确保训练时 model replicas 之间的一致性
>  为了加速训练，DDP 将梯度通信和反向计算重叠
>  DDP 的一个显著局限是要求所有模型参数、梯度、优化器状态装入单个 GPU 设备，故 DDP 不足以支持大模型

## 2.2 Model Partitioning
As the size of models grow, they may no longer fit in a single GPU device. In such cases, a viable solution is to partition the model into smaller components and distribute them across multiple devices. Both pipeline parallelism [8] and Tensor RPC [25] are along this direction. Pipeline parallelism involves breaking a sequence of layers into stages and feeding inputs to different stages in a pipelined fashion to optimize resource utilization. On the other hand, Tensor RPC provides a lower-level toolkit that enables arbitrary computations to be executed on remote devices. While both techniques are capable of scaling large models across multiple devices, they either limit the model to a sequence of stages or require modifications to the model authoring code to insert remote computations, which can pose a significant obstacle to users' adoption. Moreover, many industrial training infrastructures only support the single-program multi-data paradigm, which necessitates a simpler entry point to handle large models.
>  Model Partitioning 划分模型，流水线并行和 Tensor RPC 都属于这类方法
>  流水线并行将 layers 序列划分阶段，以流水线方式向不同阶段提供输入
>  Tensor RPC 提供了一个底层工具包，支持在远程设备上执行任意计算
>  它们要求要么模型由一系列阶段构成，要么需要修改模型代码，插入远程计算操作

## 2.3 Model Sharding
In addition to partitioning, sharding the parameters of a model can also help reduce its memory footprint and support models with sizes beyond the memory capacity of a single GPU device. After sharding models, each rank only holds a shard of the model parameters, which prevents it from performing the same computations as local training. To guarantee correctness, the training process needs to employ one or both of the following techniques:

- Perform computations with parameter shards and communicate activations accordingly. With this approach, ranks never need to fully materialize any parameter. However, each communication will appear in the critical path as it is inserted between two consecutive and dependent computation operations. As a result, this communication cannot easily overlap with computations, unless non-dependent computations or computations from other iterations can be re-ordered to overlap with communication.
- Perform the same computation as local training by communicating parameter on-demand before computations. Since parameter communications do not have any data dependency on preceding computations, they can overlap with the preceding computations performed in the same forward or backward pass. However, this approach requires that the on-demand communicated parameters could be fully materialized and could fit in the memory of a single GPU device.

>  Model Sharding 下，训练过程需要使用以下的方式确保正确性:
>  - 使用参数 shards 执行计算并相应地通信激活值，该方法下，设备不需要获取全部参数，但这类通信难以和计算重叠
>  - 在计算之前通信参数，设备本地执行完整计算，因为参数通信对先前的计算没有数据依赖，因此可以参数通信可以和先前的计算重叠，但该方法要求单个设备能容纳全部参数

FSDP falls into the second category of communicating parameters. Based on our observations and experiments, this approach is sufficient to support the vast majority of large model applications today and in the near future. It is worth noting that if the requirement of fully materializing each parameter unit on GPU becomes a blocker, we can further combine both techniques to support such use cases.
>  FSDP 属于第二类，实际上两种方法可以结合 (对参数 shards 再划分，计算时恢复参数 shard)

# 3 System Design
Fully Sharded Data Parallel (FSDP) is capable of scaling to accommodate large models that may not fit in a single GPU device by sharding the dense parameters. More specifically, FSDP decomposes the model instance into smaller units and handles each unit independently. During forward and backward computation, FSDP only materializes unsharded parameters and gradients of one unit at a time, and otherwise, it keeps parameters and gradients sharded. Throughout the training loop, the optimizer states are kept sharded. The memory requirements for FSDP are proportional to the size of the sharded model plus the size of the largest fully-materialized FSDP unit.
>  FSDP 将模型实例划分为 units，独立处理每个 unit
>  在前向和反向计算中，FSDP 一次仅恢复单个 unit 的参数和梯度，其他 units 的参数和梯度保持 sharded
>  整个训练过程中，优化器状态保持 sharded
>  FSDP 的内存需求和 sharded model 的大小 + 最大的 fully-materialized FSDP unit 的大小成比例

![[pics/FSDP-Fig1.png]]

Figure 1 demonstrates the overall workflow using a simple six layer model. Suppose FSDP decomposes the model into three parts, namely, `[layer0, layer3]`, `[layer1, layer2]`, and `[layer4, layer5]`. The decomposition behavior can be controlled by user-defined functions. FSDP then wraps each of these three parts into one FSDP unit and shards parameters accordingly. 
>  Fig1 中，模型分为了三个 units，然后对每个 unit 的参数都做了 sharding

To ensure correctness, FSDP needs to recover the unsharded parameters before corresponding computations. Let us consider FSDP unit1 that contains `[layer1, layer2]` to explain this process. Before forward computation enters layer1, FSDP collects the unsharded parameters for layer1 and layer2 by gathering shards from other peer ranks. With the unsharded parameters, FSDP runs the local computation of those layers and then frees the peer shards it just collected to reduce memory footprint. 
>  FSDP 需要在计算之前恢复 unsharded parameters
>  例如在前向计算进入 layer1 (unit1) 之前，FSDP 从其他设备收集 layer1, layer2 的参数 shards，然后进行计算，再丢弃收集到的参数

Therefore, during the entire forward pass, FSDP only needs to fully materialize one unit at a time, while all other units can stay sharded. 
>  这样在整个前向过程中，FSDP 一次只需要完整化单个 unit，其他 units 保持 sharded

Similarly, during the backward computation, FSDP unit1 recovers the unsharded parameters for layer1 and layer2 before backward reaches layer2. When the autograd engine finishes the backward computation of these two layers, FSDP frees the peer shards and launches ReduceScatter to reduce and shard gradients. Hence, after backward computation, each rank only keeps a shard of both parameters and gradients.
>  类似地，反向过程中，在达到 layer2 之前，FSDP 恢复 unit1 的 unsharded parameters
>  在自动微分引擎完成反向计算后，FSDP 释放 shards，并发起 ReduceScatter 来规约并 shard 梯度 (和其他数据并行设备规约这个 unit 全部参数的梯度，然后保留自己的参数 shard 对应的梯度)
>  这样，反向过程后，每个设备仅保留参数和梯度的一个 shard

FSDP offers a wide spectrum of optimizations and knobs to account for diverse model structures and hardware capabilities. The remainder of this section delves further into the intricacies of model initialization, sharding strategies, communication optimizations, and memory management, which are all critical components of FSDP's underlying design.

## 3.1 Model Initialization
Before the advent of FSDP, PyTorch mandated the full materialization of the entire model instance on one device. Although users can allocate different sub-modules to different devices, this would require modifying the model source code, which may not be feasible, particularly if model authors and application developers belong to different parties. 
>  FSDP 之前，PyTorch 要求单个设备 full materialize 整个模型实例
>  虽然用户可以为不同设备分配不同 sub-modules，但这要求修改模型源码

To facilitate a smooth transition from local to distributed training, FSDP must effectively aid in the materialization and initialization of a massive model, which poses two challenges:

- How to create a model instance without materializing any tensor storage, postponing initialization until a storage on a concrete device is attached to the tensor. 
- How to ensure accurate initialization of model parameters in line with the user's implementation, even when the model is too large to fit on a single GPU.

>  FSDP 需要有效支持大模型的实例化和初始化，这带来两个挑战:
>  - 如何在不实例化任何张量存储的情况下创建模型实例，将初始化推迟到张量绑定到具体设备的存储之后
>  - 模型大到无法放入单个 GPU 时，如何确保模型参数的初始化和用户的实现逻辑保持一致

To overcome the first challenge, we have introduced a mechanism called deferred initialization, which involves the allocation of model parameter tensors on a simulated or "fake" device. During this process, all initialization operations performed on the tensor are recorded. Subsequently, when the tensor is moved from the "fake" device to a GPU device, all recorded operations are automatically replayed. By adopting this technique, users can generate a model instance from any third-party library without allocating any GPU memory blocks, while still accurately capturing their parameter initialization implementations.
>  对于第一个挑战，我们引入延迟初始化机制，它将模型参数张量分配到一个模拟的或虚假的设备，并记录该分配过程中对张量执行的所有初始化操作
>  之后，当张量从虚假设备移动到真实设备时，自动重放所有被记录的操作
>  在该机制下，用户可以在不分配任意 GPU memory blocks 的情况下从任意第三方库生成模型实例，同时正确捕获其参数初始化实现

As illustrated in Figure 1, once the FSDP has wrapped the model, it is evenly distributed across all GPUs, with each device holding only one shard in its memory. Therefore, in order to address the second challenge, each rank should ideally only materialize and initialize the shard that it owns. However, this is not always practical, since we cannot predict what initialization logic the user will implement in the model init method. The initialization logic may rely on having a unsharded parameter on the device, which makes it impossible to shard the initialization. Consequently, FSDP must prepare the unsharded parameters before executing Tensor initialization operations and simultaneously reduce the memory footprint. Given that sharding initialization is unsafe, FSDP applies the same approach as how it handles model forward and backward passes, i.e., initialize one FSDP unit at a time and shard the unit before moving on to the next one. 
>  对于第二个挑战，因为 FSDP 会均匀将参数 sharding 到多个设备，故每个设备理想下应该仅实例化并初始化它拥有的参数
>  但我们无法预测用户在模型 `init` 方法中实现的初始化逻辑，可能初始化逻辑会依赖于 “设备上具有 unsharded 参数” 的假设，这样就无法对初始化进行 shard
>  因此，FSDP 必须在执行张量初始化操作之前准备好 unsharded 参数，同时要减少内存使用
>  FSDP 采用和它处理模型前向和反向传播相同的方式，即一次实例化一个 FSDP unit，并且在处理下一个 unit 之前，对当前 unit 执行 shard

When combined with deferred initialization, FSDP traverses the fake device model instance to decompose it into FSDP units, moves one unit to a GPU device at a time, and replays the recorded initialization operations for tensors in that FSDP unit.
>   和延迟初始化结合时，FSDP 会遍历虚假设备上的模型实例，将其分解为 FSDP units，一次将一个 unit 移动到 GPU，然后对该 unit 中的张量重放记录的初始化操作

## 3.2 Sharding Strategies
The sharding strategy is an important element in FSDP that plays a significant role in determining the memory footprint and communication overhead. FSDP offers a variety of sharding strategies, ranging from fully replicated to fully sharded. To generalize these sharding strategies, we introduce the sharding factor  $F$  as the number of ranks over which parameters are sharded. By setting the sharding factor to 1, FSDP fully replicates the model and simplifies to vanilla data parallelism that uses AllReduce for gradient reduction. By setting the sharding factor equal to the number of devices (i.e., global world size  $W$ ), FSDP fully shards the model, with each device only holding  $\frac{1}{W}$  of the model. Hybrid sharding occurs when the sharding factor ranges between 1 and  $W$ . The remainder of this section focuses on full sharding and hybrid sharding since the full replication strategy is similar to the existing DDP [14].
>  FSCP 引入了 sharding factor $F$，表示参数 shard 到的设备的数量
>  $F=1$ 表示复制完整参数，等价于朴素数据并行
>  $F=W$ (等于设备数量) 表示完全 shard 模型，每个模型持有 $\frac 1 W$ 的参数

### 3.2.1 Full Sharding
The full sharding strategy leads to the lowest memory footprint but incurs the most communication overhead, for example, full sharding has 1.5x communication overhead and volume over DDP if using bandwidth optimal ring algorithm. Therefore, FSDP must carefully organize communications to maximize its efficiency under this strategy.
>  full sharding 策略的内存占用最小，但通讯开销最大，其通讯开销为标准数据并行的 1.5x

![[pics/FSDP-Fig2.png]]

We conducted two sets of experiments to understand the impact of input size on collective communication efficiency. Results are shown in Figure 2, which helped identify two ingredients for efficiencies:

(1) Even Input Size: The Nvidia NCCL [22] library offers efficient collective implementations for all-gather and reduce-scatter that require even input tensor sizes across ranks.

(2) Larger Input Size: For fixed communication volume, batching data and issuing fewer collectives improves performance by avoiding the collectives' launch overhead and increasing network bandwidth utilization.

>  我们执行两组实验来理解输入大小对集合通信效率的影响，Fig2 中的结果帮助我们识别出影响效率的两个关键因素:
>  1. 均匀的输入大小: NVIDIA NCCL 库提供了高效的 all-gather, reduce-scatter 集合通信实现，但要求各个设备上的张量尺寸均匀
>  2. 更大的输入大小: 对于固定的通信量，通过批处理数据以发起更少的集合通信可以提高性能，因为这减少了集合通信的发起开销，并提高了网络带宽利用率

For (1), NCCL's AllGather API requires even input tensor size and writes outputs into one single tensor. PyTorch's ProcessGroup wraps the NCCL API and enhances it by supporting uneven input tensor sizes across ranks and allowing users to provide a list of output tensors. 
>  对于 Fig2 (1)，NCCL AllGather API 要求均匀的输入张量大小，并且将输出写入单个张量，PyTorch `ProcessGroup` 封装了 NCCL API，支持不同设备上不均匀的输入，并允许用户提供一系列输出张量

The flexibility comes with an efficiency trade-off, as shown in Figure 2 (a). We use All-Gather Base to denote NCCL's AllGather behavior, and All-Gather to denote the one that takes a list of tensors as outputs. The latter incurs additional copies between the individual output tensors and the consolidated single large output tensor before and after the communication. Moreover, for uneven inputs, ProcessGroup mimics AllGather's behavior using group Broadcast, which is slower than All-Gather Base. In the experiments, we created artificial unevenness by moving 1 element and 1e6 elements from rank 1 to rank 0 respectively. The results show that the All-Gather Base with even input size achieved highest efficiency.
>  虽然提高了灵活性，但实际牺牲了效率

For (2), Figure 2 (b) fixes the total communication to be  $2^{30} \approx 1B$  FP32 elements and varies the size per All-Gather, i.e., smaller AllGather size means more AllGather invocations. Once the AllGather size decreases below 33M elements, the total communication time begins increasing rapidly.

Thus, to deliver highly efficient communications, FSDP organizes all parameters within one FSDP unit into a large FlatParameter, where the FlatParameter coalesces the communications of its individual parameters and also evenly shards them across ranks. More specifically, the FlatParameter is a 1D tensor constructed by concatenating  $p$  flattened original parameters and padding on the right to achieve a size divisible by the sharding factor. To shard the FlatParameter, FSDP divides it into equal-sized chunks, where the number of chunks equals the sharding factor, and assigns one chunk per rank. 
>  为了提供最高的效率，FSDP 将单个 FSDP unit 内的所有参数组织为一个大的 `FlatParameter`，`FlatParameter` 会合并它之中各个参数的通信并且均匀将参数 shard 到各个设备
>  具体地说，`FlatParameter` 是通过拼接 `p` 个展平的原始参数，并做 padding 构造的 1D 张量，使得其大小可以被 sharding factor 整除
>  对 `FlatParameter` shard 时，FSDP 将它划分为相同大小的 chunks, chunks 数量等于 sharding factor，每个设备获得一个 chunk

![[pics/FSDP-Fig3.png]]

The FlatParameter's gradient inherits the same unsharded and sharded shapes from the FlatParameter, and the FlatParameter and its gradient own the underlying storage of the original parameters and their gradients, respectively. Figure 3 depicts one example, where we use one FSDP unit to shard a  $4 \times 3$  nn.Linear layer across 16 GPUs. In this case, every GPU only holds one element from the FlatParameter with the last rank holding the padded value.
>  `FlatParameter` 的梯度的 unsharded, sharded 形状和 `FlatParameter` 相同
>  `FlatParameter` 和其梯度持有原始参数和其梯度的底层存储
>  Fig3 中，我们使用一个 FSDP unit，将 4x3 的 `nn.Linear` 层划分到 16 个 GPUs，在这个情况下，每个 GPU 从 `FlatParameter` 获得一个权重元素，最后一个 GPU 获得 padded value

This flatten-concat-chunk algorithm permits each original parameter to have arbitrary shape while minimizing the required padding (to be at most  $F -1$ ), reflecting its generality. Moreover, under this algorithm, the sharded and unsharded FlatParameter and its gradient have the exact data layout expected by AllGather and ReduceScatter, respectively. This enables calling the collectives without any additional copies for either the input or output tensors.
>  上述的 flatten-concat-chunk 方法允许每个原始参数具有任意的大小，同时最小化的要求的 padding (最多 $F-1$)
>  此外，该算法下，sharded, unsharded `FlatParameter` 和其梯度具有 AllGather, ReduceScatter 期望的相同 data layout

More formally, suppose for a model with  $\Psi$  number of elements, FSDP constructs  $N$  FlatParameterS with tunnels  $\psi_{1}, \ldots , \psi_{N}$ , where  $\sum_{i = 1}^{N} \psi = \Psi$ . For sharding factor  $F$ , the peak parameter memory contribution is in  $O(\sum_{i = 1}^{N} \frac{\psi_{i}}{F} + \max_{i = 1}^{N} \psi_{i})$  because FSDP always keeps each local sharded FlatParameter with size  $\frac{\psi_{i}}{F}$  in GPU memory and must materialize each unsharded FlatParameter with size  $\psi_{i}$  one by one during forward and backward. Since the first  $\sum_{i = 1}^{N} \psi_{i} = \Psi$  is fixed, the peak parameter memory contribution is determined by  $\max_{i = 1}^{N} \psi_{i}$ . 
>  形式化地说，假设一个有 $\Psi$ 参数的模型
>  FSDP 构造 $N$ 个 `FlatParameter` $\psi_1, \dots, \psi_N$，满足 $\sum_{i=1}^N \psi = \Psi$ ($N$ 个 units)
>  sharding factor 为 $F$ 时，FSDP 的峰值内存消耗为 $O (\sum_{i=1}^N \frac {\psi_i}{F} + \max_{i=1}^N \psi_i)$
>  这是因为 FSDP 总是保留大小为 $\frac {\psi_i} F$ 的 `FlatParameter`，并每次需要实例化完整的一个 `FlatParameter`

At the same time, the number of collectives per iteration is in  $O(N)$ . This evidences FSDP's memory-throughput trade-off: Finer-grained FlatParameter construction decreases peak memory but may decrease throughput by requiring more collectives. Users can control this trade-off by specifying how to wrap sub-modules into FSDP units.
>  同时，每次迭代的集合通信次数为 $O (N)$ ($N$ 个 units)
>  这也说明了 FSDP 的内存-吞吐权衡: 更细粒度的 `FlatParameter` 构造 (更多 units) 减少峰值内存，但需要更多集合通信，因此可能降低吞吐

### 3.2.2 Hybrid Sharding.
We refer to the strategy when the sharding factor is greater than 1 but less than  $W$  as hybrid sharding, as it combines both sharding and replication. For global world size  $W$  and sharding factor  $F$ , the parameters are sharded within each group  $S_{1}, \ldots , S_{W / F}$  and are replicated within each complementary group  $R_{1}, \ldots , R_{F}$ , where each  $S_{i}, R_{j} \subseteq \{1, \ldots , W\}$  gives the ranks in the sharded or replicated group, respectively.
>  如果 sharding factor 大于 1 但小于设备数量 $W$，我们称为混合 sharding，因为它结合了 sharding, replication
>  混合 sharding 下，参数在每个 group 内 sharding: $S_1, \dots, S_{W/F}$，但 group 之间 replication: $R_1, \dots, R_F$ 

>  每个 sharding group 内按照 sharding factor $F$ 将完整的参数 shard 到 $F$ 个设备上
>  总设备数量为 $W$，那么 sharding groups 的数量就是 $W/F$
>  从完整参数的角度出发，各个 sharding group 都持有完整的参数，因此所有的 sharding groups 构成一个 replication group, replication group 的数量就是 1，同时 replication degree 等于 sharding groups 的数量 $W/F$
>  从 sharded 参数的角度出发，每个 sharding group 内的一个设备持有 $1/F$ 的参数，它和其他的 $W/F - 1$ 个 sharding groups 中持有这个 $1/F$ 参数的设备构成了一个 replication group，因此 replication group 的数量就是参数 shards 的数量，即 $F$，同时 replication degree 等于 sharding groups 的数量 $W/F$

For gradient reduction, the single reduce-scatter over all ranks becomes a reduce-scatter within each of the sharded groups followed by an all-reduce within each of the replicated groups to reduce the sharded gradients. The equivalence follows from the decomposition

$$
\sum_{r = 1}^{W}g_{r} = \sum_{i = 1}^{W / F}\sum_{r\in S_{i}}g_{r}, \tag{1}
$$

where  $g_{r}$  represents the gradient on rank  $r$ .

>  梯度规约时，原来对所有设备的 reduce-scatter 变为在每个 sharded group 内的 reduce-scatter + replication groups 之间的 all-reduce

![[pics/FSDP-Fig4.png]]
Hybrid sharding can take advantage of datacenter locality for accelerated training and can reduce cross host traffic to avoid as much contention in the oversubscribed environment as possible. At the same time, it provides a graduating trade-off between memory saving and throughput degradation, which is particularly helpful for models whose required memory footprint when trained with full replication is just slightly above the device capacity and do not want full sharding. Figure 4 shows one example.
>  混合 sharding 可以利用数据中心的局部性优势，减少跨主机通信流量，从而避免在资源过载环境下的竞争冲突
>  它也为内存节省和吞吐下降之间提供了一个可调节的权衡，适用于在 full replication 时略微高出设备容量，又不希望 full sharding 的模型十分有益

Specifically, datacenters typically adopt a fat-tree network topology [16] with over-subscription, leading to abundant locality to exploit and a well-motivated reason to reduce cross-host traffic [17]. Hybrid sharding can provide a natural mechanism to map the device mesh into the datacenter layout to exploit such locality. For example, consider a cluster as a group of  $W$  accelerators grouped into hosts of of  $G$  accelerators each (where the communication among accelerators on the same host is much faster than the communication across hosts), we can set  $F = \frac{W}{G}$  to limit the AllGather (and ReduceScatter) operations within the same host, while creating a replication group for accelerators with the same local rank across hosts. 
>  具体地说，数据中心通常采用具有过载特定的胖树网络拓扑，这带来了丰富的局部性优势以利用，也使得减少跨主机通信成为合理且必要的目标
>  混合 sharding 可以自然地将设备网格映射到数据中心布局中，以利用其局部性
>  例如，考虑一个集群，由 $W$ 个加速器组成，这些加速器被划分为 $\frac W G$ 台主机，每台主机 $G$ 个加速器，我们可以设定 $F = \frac W G$ 来限制 AllGather, ReduceScatter 发生在相同主机内，同时为不同主机上相同 rank 的设备组创建 replication group
>  (感觉应该是 $F = G$)

For an  $M$ -sized model, we can then compute the total cross-host traffic per GPU in the hybrid setup to be  $2M\frac{W -1}{GW}$ , a drastic reduction compared to full replication's  $2M\frac{W -1}{W}$  and full sharding's  $3M\frac{W -1}{W}$ . Additionally, since the AllReduce collectives used in hybrid sharding operates at a smaller world size, they empirically achieve a better performance than invoking collectives at the global scale (in the case of full replication and full sharding), due to straggler effects and larger network interference.
>  对于大小为 $M$ 的模型，在混合 sharding 下的每个 GPU 的总跨主机流量为 $2M\frac {W-1}{GW}$，其中 $2M$ 表示 AllReduce 梯度一次的通信量 (每次 AllReduce，每个设备都需要发送自己的梯度，并接收来自其他设备的梯度) (不是很理解这里的计算)
>  此外，因为此时 AllReduce 集合通信在更小的 world size 下进行，其性能通常比在 global scale 上使用 AllReduce 要好

Another important design motivation for hybrid sharding is the needs from medium-sized models. These models are large enough to cause out of memory issues when trained with full replication but are not large enough to fully utilize accelerator memory when used with full sharding, leading to both runtime overhead and memory waste. The hybrid sharding strategy creates a much richer memory-throughput trade-off space by simply adjusting  $F$ .

### 3.2.3 Autograd.
FSDP's FlatParameter must inter-operate with PyTorch's autograd engine to ensure (1) correct gradient propagation and (2) timely gradient reduction. For (1), recall that the FlatParameter and its gradient own the underlying storage of the original parameters and their gradients, respectively. To achieve this, before forward computation, FSDP sets the original parameters to be views into their unsharded FlatParameter using autograd-visible torch.split() and torch.view() calls. Then, the autograd engine naturally allocates the unsharded FlatParameter gradient and writes each original parameter's gradient to the appropriate offset as defined by torch.split()'s backward function. For (2), FSDP registers a gradient hook that only runs once the FlatParameter's gradient is finalized. The hook represents the post-backward logic and includes the gradient reduction. 
>  FSDP 的 `FlatParameter` 必须和 PyTorch 的自动微分引擎协同工作，以确保:
>  1. 梯度传播正确 2. 梯度及时规约
>  对于 1，`FlatParameter` 和其梯度拥有原始参数和梯度的底层存储，在前向计算之前，FSDP 使用 autograd 可见的 `torch.split(), torch.view()` 将原始参数设定为 unsharded ` FlatParameter ` 的视图，这样，autograd 引擎自然地分配 unsharded `FlatParameter` 梯度，并将原始参数的梯度写到由 `torch.split()` 的反向函数定义的 offset 上
>  对于 2，FSDP 注册 gradient hook，它仅在 `FlatParameter` 的梯度最终确定后执行，该 hook 表示 post-backward 逻辑，包含了梯度规约

Notably, FSDP's approach builds on top of PyTorch's autograd engine instead of hacking around it. As a result, FSDP automatically handles unconventional cases such as when not all parameters are used in the forward or when there are multiple forwards before a backward.
>  FSDP 方法构建于 PyTorch 自动微分引擎之上，而不是修改它
>  因此，FSDP 可以自动处理各种非典型情况，例如并非所有参数都在前向传播中使用，或在一次反向传播之前存在多次前向传播

## 3.3 Communication Optimizations
The FSDP framework incorporates a range of native communication optimization techniques. This section unveils four major ones: overlapping, backward pre-fetching, forward pre-fetching, and accumulation.
>  本节介绍 FSDP 的四种通信优化技术: overlapping, backward pre-fetching, forward pre-fetching, accumulation

### 3.3.1 Overlapping Communication and Computation.
The PyTorch c10d library has a ProcessGroup abstraction that represents a group of processes that can run collectives together. For the NCCL backend, the ProcessGroupNCCL implementation has an internal NCCL stream per device, where the separate internal stream is for asynchronous execution with the current stream, which is typically the default stream running computation. 
>  PyTorch `c10d` 提供了 `ProcessGroup` 抽象，表示一组一起运行集合通信的进程
>  对于 NCCL 后端，`ProcessGroupNCCL` 会为每个设备维护一个内部 NCCL stream，这些 stream 会与当前默认 stream (运行计算的 stream) 异步执行

Those asynchronous collectives return Work objects, where calling Work.wait() blocks the CPU thread until the collective finishes. 
>  这些异步的集合通信操作会 (立即) 返回 `Work` 对象，调用 `Work.wait()` 会阻塞 CPU 线程直到集合通信结束

For general correctness, ProcessGroupNCCL synchronizes the internal stream with the current stream before running the collective. DistributedDataParallel leverages the async-collective-and-wait() approach to overlap the gradient All-Reduces with backward computation. 
>  为了保证一般的正确性，`ProcessGroupNCCL` 在执行内部通信流之前，会先和当前计算流同步
>  DDP 利用了异步通信 + `wait()` 的方法来重叠梯度的 All-Reduce 和后向计算
>  DDP 的反向过程可以在计算之前发起 AllReduce 来重叠计算和通信

>  假设一个三层的网络如下

```
Input → Layer3 → Layer2 → Layer1 → Output
```

> DDP：在 backward 中对**梯度**做 AllReduce (所有 GPU 同步梯度)
> FSDP：在 forward 中对**参数**做 AllGather (每个 FSDP 单元在计算前聚合自己的分片参数)
> 我们假设：每个层的计算耗时 `10ms` ，每次通信 (AllReduce / AllGather) 耗时 `20ms`
> 目标是让通信和计算重叠，总时间尽可能短

>  DDP: async collective and wait (通信前置)

```python
# DDP backward 流程（伪代码）

loss.backward()  # 触发反向传播，从 Layer1 开始算梯度

# Step 1: 计算 Layer1 的梯度 (10ms)
compute_grad_layer1()

# Step 2: 启动异步 AllReduce（立即返回 Work 对象，不阻塞）
allreduce_work_layer1 = dist.all_reduce(grad_layer1, async_op=True)

# Step 3: 计算 Layer2 的梯度（此时 AllReduce 已在后台运行）
compute_grad_layer2()  # 10ms，与上一步的 AllReduce 并行！

# Step 4: 启动异步 AllReduce for Layer2
allreduce_work_layer2 = dist.all_reduce(grad_layer2, async_op=True)

# Step 5: 计算 Layer3 的梯度
compute_grad_layer3()  # 10ms，与 Layer2 的 AllReduce 并行！

# Step 6: 启动异步 AllReduce for Layer3
allreduce_work_layer3 = dist.all_reduce(grad_layer3, async_op=True)

# Step 7: 等待所有 AllReduce 完成（此时计算已全部完成，通信也快收尾了）
allreduce_work_layer1.wait()
allreduce_work_layer2.wait()
allreduce_work_layer3.wait()

# Step 8: 使用同步后的梯度更新参数
optimizer.step()
```

> 时间线图示 (单位：ms)

| 时间轴 | 0–10ms      | 10–20ms      | 20–30ms                    | 30–50ms                    |
| --- | ----------- | ------------ | -------------------------- | -------------------------- |
| 计算  | Layer1 grad | Layer2 grad  | Layer3 grad                | -                          |
| 通信  | -           | AllReduce L1 | AllReduce L1, AllReduce L2 | AllReduce L2, AllReduce L3 |

> 总时间 ≈ 计算总时间 (30ms) + 最后一个通信的等待 (注意通信是并发的)，所以 ≈ 50ms (而不是 30+20×3=90ms)

However, in contrast to DDP's backward where the AllReduce proceeds the computation with which to overlap, FSDP's forward issues the AllGather following the computation with which to overlap since in eager execution, FSDP cannot know which FlatParameter to AllGather next to reorder it before the computation. This difference in kernel-issue order makes following the async-collective-and-wait() approach infeasible for FSDP. Namely, since ProcessGroupNCCL synchronizes with the current (default) stream, the All-Gather will not run until the computation with which to overlap finishes.
>  DDP 不需要考虑前向过程中的通信，因为每个设备都有完整参数
>  DDP 在要重叠的计算启动之前发起通信
>  FSDP 还需要考虑前向过程中的通信，需要在要重叠的计算发起之后启动通信，这是因为在即时执行模式下，下一次执行那一层需要在知道这一次执行那一层才能确定 (例如直接跳转到了 layer2，在确定了这次执行的是 layer2 才能确定下次执行 layer3，这是我能想到的最合理的解释，或许是和 unit 有关)，也就是无法知道要提前对 `FlatParameter` 的哪一部分进行 AllGather，因此需要等到计算发起之后，确定了要收集哪些参数，再发起通信
>  因为 `ProcessGroupNCCL` 会和当前默认流同步，故 AllGather 会等待它要重叠的计算结束之后再运行
 
To address this, FSDP uses a separate CUDA stream to issue the AllGatherS, bypassing the false dependency on preceding computation in the default stream and allowing each AllGather to overlap. 
>  为了解决这一点，FSDP 使用单独的 CUDA stream 来发起 AllGather，绕过默认流中对于前置计算的虚假依赖关系，使得 AllGather 可以和计算重叠

![[pics/FSDP-Fig5.png]]

As a result, FSDP's collective synchronization operates on streams, not simply Work objects. Figure 5 illustrates one example. Note that the backward pass excludes the AG0 All-Gather because FSDP intentionally keeps the outermost FSDP unit's parameters in memory to avoid redundantly freeing at the end of forward and then re-All-Gathering to begin backward.
>  因此，FSDP 的集合同步基于流，而不仅仅是 `Work` 对象

### 3.3.2 Backward Prefetching.
FSDP enforces a single CUDA device per rank and uses a single process group for both AllGather and ReduceScatter, which means that its collectives run sequentially in the process group's internal NCCL stream. In the backward pass, FSDP issues the ReduceScatter for the current FlatParameter and then the AllGather for the next FlatParameter. Hence, the single NCCL stream forces the ReduceScatter to block the next AllGather, which in turn blocks the next gradient computation and may become exposed on the critical path.
>  FSDP 要求每个 rank (进程) 仅使用一个 CUDA 设备，并且对 AllGather, ReduceScatter 共用一个进程组
>  这意味着 FSDP 的所有集合通信操作都会在该进程组内部的 NCCL stream 上顺序运行
>  在反向过程中，FSDP 先为当前 `FlatParameter` 发起 ReduceScatter，然后为下一个 `FlatParameter` 发起 AllGather
>  由于只有一个 NCCL 流，AllGather 将等待 ReduceScatter 完成之后再进行 (通信没有并发)，故下一次的梯度计算也会阻塞 (等待 AllGather)

To avoid two consecutive exposed communication calls in the backward pass, FSDP backward prefetching issues the next AllGather before the current ReduceScatter. However, as mentioned before, a challenge for eager execution is knowing which FlatParameter to AllGather next. FSDP resolved this challenge by recording the reverse forward execution order of modules as the proxy of their backward execution order. Moreover, the forward order is freshly recorded each iteration, meaning that the backward prefetching is compatible with dynamism across iterations.
>  为了避免反向过程中出现两个连续的暴露的通信调用，FSDP 反向预取会在当前 ReduceScatter 之前发起下一次 AllGather (这样暴露的就只有 AllGather，ReduceScatter 可以和下一次的梯度计算重叠)
>  但即时执行的挑战是知道下一次要 AllGather 那个 `FlatParameter`，为此，FSDP 通过模块在正向传播过程中的逆向顺序，作为模块的反向执行过程的代理
>  此外，每次迭代都会重新记录前向顺序，因此反向预取机制能够兼容跨迭代的动态性

### 3.3.3 Forward Prefetching
For some workloads with relatively slow CPU execution, the CPU thread may not be able to issue the next forward AllGather early enough to efficiently fill the NCCL stream. If the model follows a static computational graph across iterations, then FSDP can assume the forward execution order of modules from the previous iteration and prefetch the next AllGather explicitly in the forward pass. This forward prefetching issues the next AllGather before forward computation of current FSDP unit.
>  对于一些 CPU 执行非常慢的 workload，CPU 线程可能无法即时发起下一次 forward AllGather 以填满 NCCL 流
>  如果模型在迭代之间遵循静态的计算图，FSDP 就可以根据之前迭代的前向执行顺序在前向传播中显式预取下一次 AllGather
>  这种正向预取机制会在当前 FSDP unit 的前向计算之前发起下一次 AllGather

### 3.3.4 Gradient Accumulation.
FSDP offers two variations of gradient accumulation: with and without communication. With communication, FSDP still reduces gradients across ranks, and each rank saves the sharded gradients. Simply running multiple iterations without clearing gradients achieves this. Without communication, FSDP does not reduce gradients across ranks, and each rank saves the unsharded gradients. This latter variation trades off increased memory usage with decreased communication, which can increase end-to-end throughput.
>  FSDP 提供了梯度累积的两类变体: 有通信和没有通信
>  有通信时，FSDP 仍然在各个进程之间规约梯度，且每个进程保存 sharded 梯度，直接运行多次迭代，不清理梯度就达成了梯度累积
>  没有通信时，FSDP 不会在各个进程之间规约梯度，每个进程保存 unsharded 梯度，这个变体用更高的内存使用来换减少的通信量，以提高吞吐

## 3.4 Memory Management
PyTorch uses a CUDA caching allocator as a middle layer to serve GPU allocation and free requests for PyTorch programs. In order to effectively manage memory, FSDP uses a rate limiter to take into account the memory impact of the caching allocator on programs that use several CUDA streams and run fast CPU threads.
>  PyTorch 使用 CUDA caching allocator 作为 GPU 显存管理和 PyTorch 程序之间的中间层
>  FSDP 引入了速率限制器来考虑 caching allocator 对于使用多个 CUDA streams 并且运行快速 CPU 显存的程序的影响

>  回忆一下 PyTorch 的文献中提到 caching allocator 会为每个 CUDA stream 维护单独的显存池

### 3.4.1 How Does PyTorch Caching Allocator Affect Memory.
The caching allocator avoids frequent calls to cudaMalloc and cudaFree, where the latter incurs a costly device synchronization. Specifically, the caching allocator requests CUDA memory blocks and internally determines how to split and reuse the blocks without returning them to CUDA with the goal being to reach a steady state without further calls to cudaMalloc and cudaFree.

The caching allocator runs from the CPU thread, meaning that it must decide which caching allocator block to use for an allocation when the CPU thread processes the allocation request. It cannot wait until the GPU kernel needing the allocation actually runs, which may be much later.
>  caching allocator 由 CPU 线程运行，意味着当 CPU 线程处理分配请求时，caching allocator 必须决定用哪个 caching allocator block 来执行分配
>  它无法等到实际需要该显存的 GPU kernel 运行，这个时间可能要晚得多

For a single stream, the caching allocator can directly reuse memory blocks by the stream's sequential ordering semantics. However, for separate producer and consumer streams, there are no interstream ordering guarantees, and the caching allocator cannot be certain that a block is safe to reuse until the last GPU kernel depending on that memory finishes running. Hence, if the CPU thread runs far ahead of the GPU execution, then the caching allocator cannot reuse blocks for the producer stream with pending GPU kernels from the consumer stream.
>  对于单个 stream, caching allocator 可以直接利用 stream 的顺序语义直接复用内存块
>  但对于不同的生产者和消费者 streams，没有 interstream 顺序保证，caching allocator 需要等待最后一个依赖于该内存的 GPU kernel 完成执行才能确定这个内存块可以被复用
>  因此，如果 CPU 线程的执行远远超前于 GPU 的实际执行进度，caching allocator 就无法为生产者流复用内存块，因为消费者流中仍然还有使用该内存块的 GPU kernels 等待执行

>  生产者流的过度分配就是指 CPU 线程跑得太快，提前分配了很多现在用不上的显存块

Furthermore, caching allocator blocks are allocated per stream and cannot be reused for a different stream, this over-allocates blocks to the producer stream that could otherwise be used for the consumer stream (e.g. for activations). The GPU itself may have enough memory to serve a new allocation in the consumer stream, but the overallocation to the producer stream may lead to the caching allocator failing to serve it. This forces a blocking sequence of cudaFrees to reset the caching allocator memory state called a cudaMalloc retry that greatly degrades training throughput.
>  此外，caching allocator blocks 是按 stream 分配的，无法为不同的 stream 复用，因此为生产者流过度分配的 blocks 无法被消费者流复用 (例如无法复用激活的内存块)
>  或许 GPU 可以容纳消费者流的内存分配，但生产者流的过度分配可能会导致 caching allocator 无法满足这一需求
>  这迫使需要使用一系列阻塞式的 `cudaFree` 来重置 caching allocator 内存状态，这被成为 `cudaMalloc` rety，会显著降低训练吞吐

### 3.4.2 Rate Limiter.
FSDP allocates the AllGather destination tensor representing the unsharded FlatParameter in a producer stream, and the forward and backward computations using the AllGathered parameters run in a consumer stream (typically the default stream). For a fast CPU thread, there may be pending GPU computation kernels when the caching allocator must serve the next AllGather, leading to no block reuse. 
>  FSDP 在生产者流 (通信流) 中分配表示 unsharded `FlatParameter` 的 AllGather 目标张量，而前向和反向计算则使用消费者流 (通常是默认流) AllGathered 参数
>  对于快速的 CPU 线程，当 caching allocator 必须为下一个 AllGather 分配显存时，会存在 pending GPU kernels (也就是上一层的计算还没搞定，已经发出对下一层的参数 AllGather 的显存分配要求了)，导致没有出现显存块的重用 (重用上一层的参数 AllGather 的显存块来存下一层的参数 AllGather)

Even after the blocks are not active in the AllGather producer stream, these reserved blocks can not serve default computation stream's allocation requests, and thus may force blocking cudaFrees and cudaMalloc.
>  即便 AllGather 生产者流中的块不再活跃，这些保留的块也不能作为消费者流的分配请求的复用 (不同流之间的显存池独立)
>  进而可能导致阻塞式的 `cudaFree, cudaMalloc`

FSDP offers a rate limiter that intentionally blocks the CPU thread to ensure proper caching allocator block reuse. It allows at most two inflight AllGathers, which is the minimum amount to still achieve communication and computation overlap.
>  FSDP 提供了一个速率限制器，有意地阻塞 CPU 线程，以确保合适的 caching allocator block 重用
>  速率限制器最多允许两个 AllGather 同时进行，这是实现计算和通信重叠的最少数量 (计算当前层的同时发起对下一层参数的 AllGather 通信)

# 4 Implementation
This section delves into the intricacies of FSDP implementation, which although do not alter the FSDP core algorithm, are crucial to understand before adopting FSDP.

Users can access FSDP through two APIs, FullyShardedDataParallel model wrapper and fully_shard module annotator. The former wraps the entire model and replaces sub-modules with corresponding FSDP units. In contrast, the latter installs FSDP logic as nn.Module forward and backward hooks, preserving both model structures and parameter fully-qualified names.
>  用户通过两个 API 访问 FSDP: `FullyShardedDataParallel` 模型封装器和 `fully_shard` 模块标记器
>  前者将整个模型封装，使用对应的 FSDP units 替换该模型的子模块
>  后者在 `nn.Module` 上安装前向和反向钩子来注入 FSDP 逻辑，保留原有模型的结构和参数的名称

## 4.1 Initialization
Section 3.2.1 described FSDP's solution to efficiently initialize large models, which works well when sub-module initializations are self-contained. 
>  之前描述的 FSDP 的模型初始化方式可以处理子模块之间的初始化的独立的情况

In a rare situation where one sub-module's initialization depends on a parameter from the different sub-module, the on-demand materialization and record-replay approach might break if the parameter belongs to a different FSDP unit, because the unsharded version of that parameter could have been discarded to reduce memory footprint. 
>  如果子模块之间的初始化依赖于不同的子模块的参数，而该子模块的参数属于不同的 FSDP unit，之前描述的按需实例化和 record-replay 方法就失效了
>  因为在实例化当前子模块时，其他子模块的 unsharded 参数可能已经被丢弃以减少内存消耗

Therefore, besides the advanced deferred initialization, FSDP offers two more options:
>  因此，除了高级的延迟初始化以外，FSDP 还提供了两个选项:

- **Initialize unsharded model on GPU.** The memory requirement for model initialization may be smaller than that for training since training also involves gradients, activations, and optimizer states. Consequently, if the training step cannot be performed on a single GPU device, users might still be able to initialize the entire model on a GPU and pass it to FSDP. Then, optimizers should be instantiated after FSDP shards the model, to reduce the memory footprint and align with the sharded gradients produced by FSDP.
>  在 GPU 初始化 unsharded model: 
>  模型初始化的内存需求一般小于模型训练的内存需求，因为训练还涉及了梯度、激活和优化器状态
>  因此即便单个 GPU 无法执行全模型训练，用户也有可能可以在单个 GPU 上初始化整个模型，然后再交给 FSDP 进行训练
>  优化器应该再 FSDP 对模型 shards 之后再初始化，以减少内存消耗，并且和 FSDP 产生的 sharded 梯度对齐

- **Initialize unsharded model on CPU.** If the size of the unsharded model surpasses the capacity of GPU memory and can only be accommodated in CPU memory, it becomes impracticable to move the unsharded model entirely to the GPU before handing it over to FSDP for parameter sharding. To overcome this challenge, FSDP adopts a streaming approach, where the model is migrated to the GPU unit by unit. Upon arrival to the GPU, the parameters of each unit are immediately sharded, which in turn reduces the memory overhead before processing the next unit. This approach remains viable even when there are cross-submodule dependencies during initialization, given that all parameters of the entire unsharded model are present in the CPU memory.
>  在 CPU 上初始化模型: 如果 unsharded model 的大小超过了单 GPU 显存，FSDP 使用流式方法，将模型逐 unit 地迁移到 GPU 上
>  每个 unit 的参数到达 GPU 之后，会被立刻 sharded，在处理下一个 unit 之前减少内存开销

>  以上两种选项都可以用于处理初始化中存在跨 unit 依赖的情况

Note that both approaches above are subject to their own limitations. The first method entails the entire model fitting within a single GPU device and thus becomes infeasible for larger models. The second method, on the other hand, can handle larger models since the CPU has considerably larger memory. However, this approach may experience substantial slowdowns in comparison to deferred initialization due to the limited memory bandwidth and parallelization capabilities of the CPU. 
>  以上两种方法都存在其限制，第一个方法要求整个模型可以放入单个 GPU
>  第二个方法的会由于 CPU 的有限内存带宽和并行化能力，初始化速度会比 deferred initialization 慢

In light of these observations, users may still prefer deferred initialization, even when dealing with models of the size range encompassed by the previous two methods.

To delimit the scope of each FSDP unit, users may choose to employ the FullyShardedDataParallel wrapper by intrusively applying it to sub-modules in model source code, or alternatively, provide a custom function to the auto_wrap_policy argument upon instantiation. Selecting the optimal wrapping approach typically requires some experiments and measurements.
>  为了限定每个 FSDP unit 的范围，用户可以选择用侵入性方式对模型的源代码中的子模块应用 `FullyShardedDataParallel` 包装器
>  或者在初始化时为 `auto_wrap_policy` 参数提供一个自定义的函数

## 4.2 Flat Parameters
The FlatParameter class inherits from nn.Parameter and behaves like an nn.Parameter. FSDP implements an accompanying FlatParamHandle class that is responsible for managing individual FlatParameter instances. The frontend, either FullyShardedDataParallel or fully_shard, interfaces with the FlatParameters only through FlatParamHandle.
>  `FlatParameter` 类继承自 `nn.Parameter`，行为也和 `nn.Parameter` 类似
>  FSDP 实现了 `FlatParamHandle` 类，负责管理单独的 `FlatParameter` 实例
>  FSDP 的前端，无论是 `FullyShardedDataParallel` 还是 `fully_shard` 接口都通过 `FlatParamHandle` 来和 `FlatParameters` 交互

One FlatParameter accommodates storage for all parameter tensors within one FSDP unit. The boundary of the FSDP unit controls the timing for AllGather and ReduceScatter, which has a direct impact on overall FSDP performance. In the ideal case, FSDP unit boundaries should align with model execution order.
>  一个 `FlatParameter` 实例容纳单个 FSDP unit 中的全部参数张量的存储
>  FSDP unit 的边界控制 AllGather 和 ReduceScatter 的时机，这个时机对 FSDP 的性能有直接影响
>  理想情况下，FSDP unit 边界应该和模型执行顺序对齐 (也就是最优的重叠效率)

>  FSDP unit 的边界即哪些层/模块被封装在一个 FSDP unit 中
>  如果整个模型封装为一个 unit，那么前向开始需要 AllGather 所有参数，通讯次数少，存储需求大
>  如果按层封装，则通讯次数多，存储需求少

FSDP has access to the model's static nn.Module structure at construction time. Fortunately, although this structure does not guarantee to faithfully represent model execution order, model authors conventionally translate layers and broader blocks to nested nn.Module definitions that may naturally have the desired parameter locality. FSDP can leverage that structure to choose the FlatParameter construction. 
>  FSDP 在构造时会访问模型的静态 `nn.Module` 结构
>  虽然该结构不会保证忠实地反映模型的执行顺序，但通常情况下，模型作者会将 layers 和更宽的模型 blocks 使用嵌套的 `nn.Module` 定义，这些结构往往天然具备期望的参数局部性
>  FSDP 可以利用这个结构来选择 `FlatParameter` 构造

Indeed, FSDP supports annotating nnModules and follows a simple rule: All parameters in the annotated nn.Module are assigned to one FlatParameter, excluding those parameters already assigned. This rule lends itself naturally to nested annotation, where blocks are annotated, forming well-sized FlatParameters, and any residual parameters are assigned to their parent.
>  FSDP 支持标记 `nn.Module`，被标记的 `nn.Module` 中的所有参数 (除了哪些已经分配的参数) 都会被分配给单个 `FlatParameter`
>  这个规则天然支持嵌套注解，可以注解高层模块，也可以注解底层模块，如果子模块被注解，它的参数就被打包为单独的 `FlatParameter`，如果子模块没有被注解，它的参数就打包到它的父模块的 `FlatParameter`

Another approach we explored is using the execution order and reconstructing FlatParameters dynamically. This approach starts with an initial small FlatParameter construction, runs a possibly inefficient first iteration while observing the execution order, and reconstructs the FlatParameters by coalescing the existing small FlatParameters according to the observed order.
>  另一种方式是动态重构 `FlatParameter`，从初始的小 `FlatParameter` s 构造开始，执行一次迭代，观察执行顺序，根据观察到的顺序合并现存的小 `FlatParameter` s

## 4.3 Runtime
FSDP augments a local model instance by incorporating communication operations to reduce gradients and gather parameters. Timely initiation of these operations is of paramount importance for ensuring both correctness and efficiency. Starting communication too soon would cause the parameters or gradients with pending updates to be consumed, while initiating communication too late would result in wasting network bandwidth and delay in subsequent computations.
>  FSDP 通过引入通信操作来规约梯度并聚合参数，即时启动这些通信操作对于保证正确性和效率至关重要
>  如果过早启动通信操作，会导致尚未完成更新的梯度或参数被使用，如果过晚启动通信操作会浪费网络带宽，延迟后续计算

To insert communication-related code to the model forward pass, the FullyShardedDataParallel nn.Module wrapper overrides nn.Module's forward() method to install pre-forward and post-forward logic, whereas the functional fully_shard implements them by registering nn.Module hooks through methods such as `register_forward_pre_hook()` and `register_forward_hook()`. 
>  为了给模型 forward pass 插入通信相关的代码，`FullyShardedDataParallel nn.Module` 封装器覆盖了 `nn.Module` 的 `forward()` 方法，以添加 pre-forward, post-forward 逻辑
>  而函数式接口 `fully_shard` 则通过调用如 `register_forward_pre_hook(), register_forward_hook()` 等方法，注册 `nn.Module` 的钩子来实现相同的功能

It is more challenging to capture appropriate signals from the backward pass, as PyTorch automatically and transparently handles the backward pass. Fortunately, the auto-grad engine exposes a variety of hooks that enable the installation of custom logic with precise granularity.
>  从 backward pass 捕获适当的信号更具挑战，因为 PyTorch 自动且透明地处理 backward pass
>  不过自动微分引擎暴提供了许多钩子机制，使得开发者能够以极细粒度的方式插入自定义逻辑

- Hooks on Tensor through `register_hook()` allows to run custom function when the gradient of a Tensor is generated. This can help anchor FSDP logic to an activation's gradient computation in the backward pass. FSDP registers this type of hook to the forward output tensor of every FSDP unit to insert communications before backward pass enters that FSDP unit.
>  通过 `register_hook()` 在张量上注册的钩子可以在生成该张量的梯度时运行自定义函数，这有助于将 FSDP 的逻辑与反向传播中某个激活值的梯度计算锚定在一起
>  FSDP 为每个 FSDP unit 的前向输出张量注册这类钩子，从而在进入该 FSDP unit 的反向传播过程之前插入通信操作

- Hooks on `backward()` through `queue_callback()` run right before exiting the current autograd GraphTask, which is usually the end of overall backward pass. FSDP relies on this hook to wait for pending communication so that the subsequent optimizer step will not consume gradients too early.
>  通过 `queue_callback()` 注册的对 `backward()` 的钩子在退出当前自动求导 `GraphTask` 之前运行，而退出 `GraphTask` 通常意味着整个反向过程的结束
>  FSDP 依赖于这个钩子来等待尚未完成的通信操作，确保后续的优化器不会过早地消耗梯度 `

- Hooks on AccumulateGrad autograd function fires when the gradient of a parameter has finished accumulation in the current backward pass. FSDP attaches this type of hook to each FlatParameter's AccumulateGrad function to immediately launch ReduceScatter when gradients are ready. Note that the Tensor hook mentioned above can potentially achieve the same behavior, but might incur unnecessary delay as it needs to wait for gradient computations for input activations as well.
>  对 `AccumulateGrad` 自动求导函数的钩子会在当前反向传播中某个参数的梯度累积完成后触发
>  FSDP 将这类钩子附加到每个 `FlatParameter` 的 `AccumulateGrad` 函数上，以在梯度准备就绪之后立刻发起 ReduceScatter 通信
>  虽然前述的张量钩子理论上也能实现类似的行为，但可能引起不必要的延迟，因为它需要等待输入激活的梯度计算也完成之后才能触发 (而这个钩子直接钩在参数上)

The aforementioned methodologies collectively integrate the FSDP algorithm with the PyTorch nn.Module and autograd engine in a non-intrusive and efficient manner.
>  上述方法将 FSDP 算法以非侵入的形式结合到了 `nn.Module` 和自动微分引擎

## 4.4 Native Mixed Precision
FSDP offers a versatile native mixed precision mechanism. In terms of parameter management, it adheres to the standard mixed precision technique, which maintains both low and full precision copies of parameters [18]. Forward and backward computation use the low precision, and the optimizer step uses full precision. FSDP permits user-specified precisions for parameters, gradient reduction, and non-trainable buffers, each independently if desired.
>  FSDP 提供了原生混合精度机制
>  在参数管理上，它遵循标准混合精度技术，维护参数的低精度和全精度拷贝
>  前向和反向计算使用低精度，优化器使用全精度
>  FSDP 允许用户指定参数、梯度规约、不可训练缓存的精度

For  $\Psi$  number of parameter elements (torch.numel),  $K_{\mathrm{low}}$  bytes per low precision element, and  $K_{full}$  bytes per full precision element, this approach to mixed precision normally increases the memory overhead from  $K_{\mathrm{full}}\Psi$  to  $(K_{\mathrm{low}} + K_{\mathrm{full}})\Psi$  due to maintaining both precision copies. However, FSDP can sidestep the problem given our design to always keep each local sharded FlatParameter in GPU memory and only dynamically allocate the unsharded FlatParameter. For  $N$  FlatParameters with numels given by  $\psi_{1},\ldots ,\psi_{N}$  , the parameter peak memory contribution for FSDP actually decreases from  $\begin{array}{r}\frac{K_{\mathrm{full}}}{F}\sum_{i = 1}^{N}\psi_{i} + K_{\mathrm{full}}\max_{i = 1}^{N}\psi_{i} \end{array}$  to  $\begin{array}{r}\frac{K_{\mathrm{full}}}{F}\sum_{i = 1}^{N}\psi_{i} + K_{\mathrm{low}}\max_{i = 1}^{N}\psi_{i} \end{array}$  bytes. In other words, FSDP directly reduces the second  $K_{\mathrm{full}}\max_{i = 1}^{N}\psi_{i}$  term to  $K_{\mathrm{low}}\max_{i = 1}^{N}\psi_{i}$
>  对于 $\Psi$ 的参数 (由 `torch.numel` 统计)，我们假设每个低精度元素占 $K_{low}$ bytes, 每个全精度元素占 $K_{full}$ bytes
>  按照常规方法，需要同时保存两个精度的副本，内存开销将从 $K_{full}\Psi$ 提高到 $(K_{low} + K_{full})\Psi$
>  因为 FSDP 总是保存 sharded `FlatParameter`，故在混合精度下，对于 $N$ 个 `FlatParameter` s，FSDP 的峰值内存实际上从 $\frac {K_{full}}{F}\sum_{i=1}^N \psi_i + K_{full}\max_{i=1}^N \psi_i$ 降低到了 $\frac {K_{full}}F\sum_{i=1}^N \psi_i + K_{low}\max_{i=1}^N \psi_i$ 字节
>  也就是将第二个全精度项降低到了低精度项

In contrast to `torch.amp.autocast` that performs just-in-time casts at the operator level, FSDP's native mixed precision only incurs a full-to-low-precision cast per FlatParameter in its pre-forward and, if resharding after forward, its pre-backward. Moreover, FSDP's mixed precision permits running all collectives in the low precision, which saves communication volume.
>  `torch.amp.autocast` 是在算子的级别执行即时的转换，相较之下，FSDP 的原生混合精度只会对于每个 `FlatParameter` 在它的 pre-forward 阶段执行 full-to-low-precision cast，如果前向传播之后会 resharding，则还会在它的 pre-backward 阶段执行 full-to-low-precision cast
>  此外，FSDP 允许低精度下执行所有的集合通信，减少了通信量

Users most commonly choose FP16 or BF16 as the low precision and FP32  as the full precision.  FP16  's smaller dynamic range compared that of FP32 exposes FP16 to greater risk of numeric underflow and overflow. The standard solution includes a gradient scaler [1] that scales gradients to a safe magnitude. However, since FSDP shards gradients across ranks, a normal local gradient scaler implementation breaks mathematical equivalence, and instead, FSDP provides its own sharded gradient scaler.
>  用户通常使用 FP16, BF16 作为低精度，使用 FP32 作为全精度
>  FP16 的动态范围更低 (指数位少)，故更容易上溢或下溢，标准的方法是使用梯度缩放因子将梯度缩放到安全的范围
>  然而，由于 FSDP 将梯度 shard，普通的本地梯度缩放器实现会破坏数学等价性，故 FSDP 提供了自己的 sharded gradient scaler

# 5 Evaluation
We conducted an empirical evaluation of FSDP on large language models and recommendation system models and compared the results with those of DDP. Experiment specifications are described in Section 5.1. Then, we organize experiments into three categories. Section 5.2 focuses on how well FSDP handles different sizes of models. Then, Section 5.3 discusses the impact of throttling communications. Finally, Section 5.4 demonstrate FSDP's ability to scale to gigantic models.

## 5.1 Experiment Setup
In these experiments, we conducted evaluations on the Hugging-Face T5-11B transformer [26], minGPT-175B transformer [3], and DHEN recommendation model [33]. The recommendation models consist of 768B sparse parameters and 550M dense parameters, the sparse parameter tensors were sharded using the first approach mentioned in Section 2.3, which communicates activations instead of parameters, while the dense parameters were trained using FSDP on 8 to 512 A100 80GB GPUs interconnected by a 2Tb/s RoCE network. The objective was to assess the capability and scalability of FSDP in training large-scale models. Additionally, we employed T5-611M, T5-2B and T5-11B transformers to evaluate the performance of various sharding strategies, communication efficiency of prefetching, and communication throttling using rate limiter. 

Metrics employed in these experiments included TFLOPS per GPU, latency per batch, peak memory allocated, peak memory active, and peak memory reserved.
>  试验中的指标包括: 每 GPU 的 TFLOPS，每个 batch 的延迟，峰值分配内存，峰值活跃内存，峰值保留内存

## 5.2 Model Scale
In this section, we investigate the performance of FSDP when dealing with models of different sizes, spanning from 611M to 175B, and make a comparison with DDP [14].

![[pics/FSDP-Fig6.png]]

The experimental results on T5 models are displayed in Figure 6 (a). The performance of FSDP and DDP is similar when evaluating 611M and 2.28B models. However, DDP encounters an out-of-memory error when attempting to wrap models larger than 2.28B. In contrast, FSDP can effortlessly accommodate the 11B model and achieve significantly higher TFLOPS by turning on BF16. These experiments illustrate that practitioners can utilize FSDP for both small and large models, and seamlessly transition across different model configurations.

Then, we conduct additional experiments to measure the acceleration attained through backward pre-fetching. This time we use a larger GPT-175B model, where communication overhead is more prominent. Results are presented in Figure 6 (b), where pre-fetching leads to approximately  $18\%$  speedup, and this TFLOPS gain persists across different GPU cluster sizes. Therefore, for subsequent experiments, we always turn-on backward pre-fetching.
>  backward prefetching 可以提供 18% 的加速 (重叠了更多通信和计算)

## 5.3 Throttle Communications
In the subsequent analysis, we investigate the implications of throttling FSDP communications. As expounded in Section 3.4, launching AllGather too aggressively can lead to unnecessarily high memory footprint, as the CPU thread needs to allocate CUDA memory blocks when the communication kernel is added into the CUDA stream. This predicament may sometimes result in significant performance problems when the CPU thread runs too fast in comparison to CUDA streams. 
>  我们探究对 FSDP 通信进行节流的影响
>  过度发起 AllGather 会导致不必要的高内存占用，因为在通信 kernel 被添加到 CUDA stream 中时，CPU 线程就需要分配 CUDA 内存块

To gauge its efficacy in varying scenarios, we apply rate limiting to three different types of models and applied the maximum feasible batch size in each experiment.

- RegNet [29]: model size 9B, and batch size 48 for 2 nodes and 72 for 4 nodes. 
- T5 [26]: model size 11B, and batch size 2. 
- DeepViT [36]: model size 8B, and batch size 105 for 2 nodes and 120 for 4 nodes.

Experiment results are plotted in Figure 6 (c). One notable observation is that the rate limiter's effectiveness is not consistent, as it does not attain any speedups in the RegNet experiments, and even impedes the DeepViT ones. This behavior is expected since throttling the communications can only boost training if the fast CPU thread aggressively allocates GPU memory blocks and causes defragmentations. 
>  rate limiter 对不同模型的效果并不一致，这是可预期的，因为只有 CPU 线程过度分配 GPU 内存块时，节流才会有效果

If it is difficult to identify with certainty from latency measurements or profiled traces, CUDA malloc retry can serve as a helpful indicator, which can be obtained from the num_alloc_retries key in the torch.cuda.memory_stats() dictionary.
>  如果仅从延迟度量或性能剖析轨迹中难以确定是否存在有这类问题，可以以 CUDA malloc rety 作为指标，这可以从 `torch.cuda.memory_stats()` 字典中通过 ` num_alloc_retires ` key 获取

The experiments conducted with T5 models have demonstrated that the rate limiter technique can greatly benefit training efficiency, yielding up to 5X speedups. 

However, for DeepViT models, introducing communication throttling can result in an additional  $5\%$  overhead. This is due to the fact that delaying the Allgather communication can potentially block subsequent model computations that rely on the Allgathered parameters, especially in cases where communication is the dominant factor. Therefore, before enabling rate limiting, practioners should verify whether defragmentation has taken place during training.

## 5.4 Efficient Training for Large Models
To evaluate capability of using FSDP for large models, we ran three types of models using Full Sharding with prefetching and rate limiter turned on. Activation checkpointing and BF16 mixed precision are also applied in these experiments. Adam optimizer is used to reflect a production workload setup and to incur the costly two optimizer states per parameter.

- DHEN large recommendation model [33]: model size 768B sparse parameters and 550M dense parameters, and batch size 1024. 
- minGPT transformer [10]: model size 175B, vocab size 50000, block size 2048, batch size 1 and 2 for 128, 192, 256, 384 and 512 GPUs. 
- HuggingFace T5 transformer [26]: model size 11B, sequence length 512, batch size 8 and 16 for 8, 16, 32, 64, 128, 256, 512 GPUs.

In the DHEN experiments, we further combine sharding strategies with two different configurations:
>  在 DHEN 试验中，我们进一步将 sharding 策略和两个不同配置结合

- RAF: reshard-after-forward frees Allgathered shards from other GPUs after forward pass and unshards them again before backward computation. This reduces peak memory consumption at the cost of higher communication overhead. 
>  reshard-after-forward 在前向传播后释放所有从其他 GPU 获取的 Allgathered shards，在反向传播之前再 unshards
>  这减少了峰值内存消耗，但增加了通信开销

- NRAF: no-reshard-after-forward is the opposite where the unsharded model parameters stay in GPU memory after forward pass until backward computations finish, which trades higher memory footprint for lower communication overhead.
>  no-reshard-after-forward 在前向过程之后保留 unsharded 模型参数直到反向计算完成，用更高的内存开销换更低的通信开销

![[pics/FSDP-Fig7.png]]

The experimental results in Figure 7 (a) and Figure 8 (a) indicate that FSDP is capable of accommodating DHEN models on a large GPU cluster. It was observed that Full Sharding with RAF yields the smallest memory footprint but with a corresponding trade-off of reduced QPS. Conversely, Hybrid Sharding with NRAF demonstrated the opposite behavior, as it has employs both a smaller sharding group and skips one reshard. When adding more GPUs to in the cluster, the peak memory usage consistently decreases as a result of a decrease in the size of each rank's model shard.

With the 175B model, the experiments achieved more than 173 and 186 TFLOPS per GPU with batch size equal to 1 and 2 respectively as shown in Figure 7 (b). This is equivalent to approximately  $55\%$  and  $60\%$  of GPU hardware utilization, given that the A100's peak is 312 TFLOPS using the BF16 tensor core. 

Furthermore, the model demonstrated linear scalability from 128 GPUs to 512 GPUs, in terms of TFLOPS, which affirms the efficacy of FSDP in handling large models with expensive computations or high-speed network interconnections. 

Notably, with 128 GPUs, setting the batch size to 2 resulted in a considerably lower per-GPU TFLOPs in comparison to other scenarios. This was due to CUDA memory defragmentation during the backward pass. The backward pass contributed  $85.56\%$  of the iteration latency for the 128 GPUs batch size equals 2 case, while a normal backward pass only accounted for about  $67\%$  in these experiments. Using 128 GPUs is more likely to trigger defragmentation, as each GPU needs to accommodate a larger model shard. Figure 8 confirms this explanation, where the PyTorch CUDA caching allocator depletes all 80GB of the CUDA memory as shown on the top left corner.
>  这里的 defragmentation 应该指的就是显存不够导致 GPU 和 CPU 频繁进行数据交换，即震荡

Finally, for T5-11B models as shown in Figure 8 (c), all experiments are executed comfortably below GPU memory capacity, where defragmentations are unlikely to happen. Nevertheless, as the number of GPUs increases from 8 to 512, a  $7\%$  regression in per-GPU TFLOPS is still evident as illustrated in Figure 7 (c). This suggests that communications begin to outweigh computations on large clusters, and a near-perfect overlap between communication and computation is no longer attainable.

# 6 Related Work
The DDP [14] model wrapper, which is based on the model replication design, was an initial distributed training feature introduced in PyTorch [24]. Although it can handle large datasets, it cannot accommodate the ever-increasing model sizes that are now prevalent in the field.

ZeRO [27, 28] and cross-replica sharding [30] inspired the FSDP design, but FSDP is intrinsically different. Prior work employs model partitioning or per-parameter sharding to distribute parameter tensors, and rely on broadcast and Gather collective communication primitives to synchronize values. Although this design can achieve the same functionality, it could lead to uneven workload distribution across GPU devices, which hampers the efficiency of synchronized distributed training. Additionally, since this approach modifies the internals of the machine learning framework, such as tensor storage and memory management, it might no longer work when the internal implementation is updated or new features are introduced. Therefore, a native solution that is co-designed with the core components of the framework would provide a more robust and consistent user experience.
>  ZeRO 和 cross-replica sharding 启发了 FSDP，但 FSDP 在本质上与之不同
>  之前的工作采用 model partitioning 或 per-parameter sharding 的方式来分布参数张量 (FSDP 按照 unit 的 `FlatParameter` 来 sharding)，且依赖广播或 Gather 通信原语来同步 values
>  这些方法修改了 ML 框架的内部机制，例如张量存储和内存管理，可能会在框架引入了新功能时不兼容，因此和框架核心组件协同设计的方案可以提供更一致的体验

MiCS [34] and FSDP differ in gradient communication strategies. MiCS uses a global AllReduce followed by sharding within each partition group, whereas FSDP employs AllGather and ReduceScatter. As a result, each rank in MiCS must hold the entire model gradients, leading to higher memory usage than FSDP's approach of sharding a single layer. While both MiCS and FSDP use a hybrid communication strategy to improve efficiency at scale, FSDP's approach schedules AllGather within a flexibly-sized sharded group, potentially resulting in lower runtime latency than the two-hop AllGather utilized by MiCS. This reduced latency is crucial as the AllGather operation is critical to execution, and limiting the world size and participants of AllGather to accelerators within a group with good locality can result in lower latency and higher throughput.
>  MiCS 和 FSDP 的差异在梯度通信策略
>  MiCS 使用全局 AllReduce + 在每个 partition group 内 sharding
>  FSDP 使用 AllGather + ReduceScatter
>  FSDP 在一个灵活大小的 sharded group 内调度 AllGather

Pipeline parallelism [5, 8] involves partitioning model parameters and their activations across multiple devices through the division of models into pipeline stages. However, pipeline parallelism requires model changes and meticulous tuning for microbatch sizes, number of stages and partitions, as well as intricate scheduling procedures to optimize performance by squeezing out bubbles.
>  流水线并行要求修改模型结构，并调节 microbatch 大小、阶段数量和调度策略以减少气泡

Additionally, specific attention is given to high profile architectures such as transformers. For example, sequence parallelism [13] reduces activation memory in conjunction with tensor parallelism; Pipetransformer [6] designed a dynamic 2D parallelism that allows changing the dimensions of pipeline and data parallelism on the fly, depending on learning signals. These methods are highly effective but can be difficult to generalize as they either rely on the specific implementation or the model's layered structure.

Many existing solutions combine data parallelism with other parallelisms to achieve speedup. For example, Megatron [21] demonstrated highly efficient deep transformer training on large clusters using 3D (data, tensor and pipeline) parallelism. Further, compiler-based techniques such as Alpa [35], GSPMD [31], and FlexFlow [9] leverage profiling, performance modeling, user annotations and search to find the best configuration across the parallelism space of data, tensor and pipeline for a given cluster. In all cases, FSDP provides the benefit of being a drop-in replacement for data parallelism that reduces data redundancy along the data parallel axis.
>  基于编译器的技术例如 Alpa, GSPMD, FlexFlow 使用性能分析、性能建模、用户标记来搜索以找到给定集群，3D 并行的最佳配置
>  FSDP 的优势在于即插即用，有效减少数据并行维度的数据冗余

Orthogonal memory-saving techniques include gradient compression [2], mixed-precision training [7], tensor rematerialization [12] and CPU-offloading [4], but they could have implications on model accuracy and incur overhead in (un)compression, quantization, recomputation, and host-to-device copies, respectively.
>  正交的内存节约技术包括梯度压缩、混合精度训练、张量重实例化和 CPU-offloading

# 7 Discussion
This section discusses how FSDP can be combined with other parallelism paradigms and known limitations when adopting FSDP.

## 7.1 FSDP Interoperability
Further increasing scalability and efficiency of distributed training requires combining FSDP with other paradigms. This section briefly highlights how the FSDP design enables mixing and matching with other types of parallelisms.

### 7.1.1 Pipeline Parallelism.
Pipeline parallel can be functionally integrated with FSDP by employing FSDP to wrap each individual pipeline stage. However, as pipeline parallel divides input mini-batches into smaller micro-batches, the default full sharding strategy in FSDP would have to unshard model parameters for every micro-batch. Consequently, combining these approaches with default FSDP configurations may lead to significant communication overhead. 
>  可以通过使用 FSDP 封装每个单独的流水线阶段来集成流水线并行
>  由于流水线并行会将输入 mini-batches 划分为更小的 micro-batches，故 FSDP 的默认 full sharding 策略需要为每个 micro-batch unshard 模型参数
>  因此直接和默认 FSDP 配置结合会导致显著的通信开销

Fortunately, FSDP offers alternative sharding strategies that can keep parameters unsharded after the forward pass, avoiding unnecessary Allgather communications per micro-batch. Admittedly, this requires storing parameters of an entire pipeline stage on the GPU device, but FSDP can still reduce memory usage as it still shards gradients and optimizer states.
>  FSDP 提供了其他可选的 sharding 策略，可以在前向传播之后保持参数 unsharded，避免在每个 micro-batch 上重复执行 Allgather 通信
>  这要求但 GPU 设备上存储整个流水线阶段的参数，不过 FSDP 仍然可以减少内存使用，因为它会 shard 梯度和优化器状态

### 7.1.2 Tensor Parallelism.
In contrast to FSDP, tensor parallel keeps parameters sharded during computation, which is necessary if any sub-module is too large to fit in GPU memory. Presently, PyTorch provides a prototype feature called parallelize module that can be combined with FSDP to construct 2D parallelism. It works by organizing devices into a 2D mesh where PyTorch's distributed tensor DTensor manages tensor parallelism on one dimension and FSDP applies sharded data parallelism on the other dimension. These two dimensions communicate activations and parameters, respectively. 
>  TP 在计算时也保持参数 sharded，这在任意 submodule 都无法放入单个设备时是必要的
>  目前，PyTorch 提供了一个原型功能 `parallelize_module`，可以和 FSDP 结合成为 2D 并行
>  它将设备组织为一个 2D 网格，PyTorch 的分布式张量 `DTensor` 在一个维度管理张量并行，FSDP 在另一个维度施加 sharded 数据并行
>  这两个维度分别会通信激活值和参数

We usually keep the tensor-parallel communications, which block subsequent computation, intra-node to leverage the higher network bandwidth, and allow the FSDP communications operate on the other mesh dimension inter-node.
>  我们通常将 (会阻塞后续计算的) 张量并行通信保持在节点内部，允许 FSDP 通信在节点间进行

## 7.2 Limitations
During our work with production and research applications, we have encountered certain limitations associated with FSDP. This section aims to discuss two tricky caveats that are not readily apparent and pose significant challenges when it comes to troubleshooting.

### 7.2.1 Mathematical Equivalence.
FSDP cannot ensure that it always achieves the same mathematical equivalence as local training, especially with respect to the optimizer computation. This stems from the fact that the optimizer step operates on the sharded parameters, whose data layout is a function of FSDP's FlatParameter sharding algorithm that does not respect individual parameter boundaries. As a result, any optimizer computation that depends on an original parameter's unsharded value (e.g. vector norm), its tensor structure (e.g. approximate second-order optimizers), or require global states over all parameters will become invalid. 
>  FSDP 不能确保和本地训练效果相同，尤其是在优化器相关的计算上
>  因为优化器是对 unsharded 参数进行计算，而 unsharded 参数的数据布局时 FSDP `FlatParameter` sharding 算法的函数
>  因此任意依赖于原始参数的 unsharded value 的优化器计算，例如 vector norm, 依赖于原始参数 unsharded value 张量结构的优化器计算，例如 approximate second-order，或者需求参数全局状态的计算都会无效

Addressing this requires uneven sharding, padding, or extra communication, all of which hurt performance. Co-designing such optimizer computations with sharding is an open research question.

### 7.2.2 Shared Parameters.
For shared parameters, FSDP must ensure to not flatten them into multiple FlatParameters and to ensure that they are unsharded properly when needed for all usages. If handled incorrectly, PyTorch may raise an error regarding missing tensor storage or size mismatch, which can happen when an FSDP unit attempts to use a shared parameter that has already been resharded by a preceding FSDP unit. The current recommendation is to construct FSDP units such that the shared parameter belongs to the lowest-common-ancestor unit to ensure that the shared parameter is unsharded throughout all usages. This may require some inspection of the model structure to do correctly and may undesirably keep the FlatParameters unsharded for a large interval, so we are investigating approaches to improve shared parameter handling.
>  对于共享参数，FSDP 必须确保不将它 flatten 为多个 `FlatParameter`，并确保在需要的时候可以正确地对它 unshard
>  如果处理错误，PyTorch 会抛出关于确实张量存储或大小不匹配的错误，这可能在一个 FSDP unit 尝试已经被之前的 FSDP shard 的共享参数时出现
>  目前的推荐是构造 FSDP units 使得共享存储属于所有相关使用路径的最低公共祖先 unit，以确保该共享参数在整个使用过程中保持 unsharded
>  这通常要求对模型结构进行仔细分析才能实现，且可能会导致 `FlatParameter` 在较长时间下保持 unsharded，影响内存效率

# 8 Conclusion
This manuscript elucidates the underlying rationale, design philosophy, and implementation of FullyShardedDataParallel as of PyTorch 2.0 release. 

FSDP attains usability and efficiency through a set of advanced techniques, including deferred initialization, flexible sharding strategies, communication overlapping and prefetching, and rate limiting communication collectives. All of these techniques are closely co-designed with other key PyTorch components to ensure the solution is sound and robust. Evaluations show that FSDP can facilitate large language and recommendation models with near linear scalability.
>  FSDP 涉及了一系列技术，包括延迟初始化, 灵活 sharding 策略, 通信重叠和预取, 对集合通信的速率限制
