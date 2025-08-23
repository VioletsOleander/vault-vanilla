```
Samyam Rajbhandari*, Jeff Rasley, Olatunji Ruwase, Yuxiong He
```

# Abstract
Large deep learning models offer significant accuracy gains, but training billions to trillions of parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to fit these models into limited device memory, while obtaining computation, communication and development efficiency. We develop a novel solution, Zero Redundancy Optimizer (ZeRO), to optimize memory, vastly improving training speed while increasing the model size that can be efficiently trained. ZeRO eliminates memory redundancies in data- and model- parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency. Our analysis on memory requirements and communication volume demonstrates: ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware.
>  我们提出零冗余优化器，以优化模型训练时的内存使用，以提高训练速度和能够训练的模型大小
>  ZeRO 消除了数据或模型并行训练下的内存冗余，同时保留了低通信量和高计算粒度，使得模型规模可以随着设备数量增长而增长
>  我们的内存需求分析和通信量分析表明: ZeRO 有潜力在当前硬件条件下拓展至超过 1 万亿参数 (1000 B)

We implement and evaluate ZeRO: it trains large models of over 100B parameter with super-linear speedup on 400 GPUs, achieving throughput of 15 Petaflops. This represents an 8x increase in model size and 10x increase in achievable performance over state-of-the-art. In terms of usability, ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for scientists to apply. Last but not the least, researchers have used the system breakthroughs of ZeRO to create the world's largest language model (17B parameters) with record breaking accuracy.
>  ZeRO 在 400 块 GPU 上训练超过 100B 的模型可以达到超线性加速，吞吐量达到 15 Pflops
>  这相较于当前最先进的系统，模型规模拓展了 8 倍，性能提升了 10 倍
>  ZeRO 可以在不使用模型并行的情况下训练 13B 的模型

# 1 Extended Introduction
Deep Learning (DL) models are becoming larger, and the increase in model size offers significant accuracy gain. In the area of Natural Language Processing (NLP), the transformers have paved way for large models like Bert-large (0.3B) [1], GPT-2 (1.5B) [2], Megatron-LM (8.3B) [3], T5 (11B) [4]. To enable the continuation of model size growth from 10s of billions to trillions of parameters, we experience the challenges of training them - they clearly do not fit within the memory of a single device, e.g., GPU or TPU, and simply adding more devices will not help scale the training.

Basic data parallelism (DP) does not reduce memory per device, and runs out of memory for models with more than 1.4B parameters on current generation of GPUs with 32 GB memory. Other existing solutions such as Pipeline Parallelism (PP), Model Parallelism (MP), CPU-Offloading, etc, make trade-offs between functionality, usability, as well as memory and compute/communication efficiency, but all of which are crucial to training with speed and scale.
>  基础的数据并行不会减少每个设备的内存需求，因此在模型超过 1.4B 参数时，32GB 的 GPU 就会 OOM

Among different existing solution for training large models, MP is perhaps the most promising. The largest models in the current literature, the 11B T5 model [4], and Megatron-LM 8.3B [3], were both powered by model parallelism, implemented in Mesh-Tensorflow [5] and Megatron-LM[3], respectively. However, MP cannot scale much further beyond these models sizes. MP splits the model vertically, partitioning the computation and parameters in each layer across multiple devices, requiring significant communication between each layer. As a result, they work well within a single node where the inter-GPU communication bandwidth is high, but the efficiency degrades quickly beyond a single node [3]. We tested a 40B parameter model using Megatron-LM across two DGX-2 nodes and observe about 5Tflops per V100 GPU (less than  $5\%$  of hardware peak).
>  目前的最大模型都是通过模型并行实现的
>  但模型并行也难以进一步拓展 (超过 11B 参数后)，模型并行是将模型垂直分割，将每一层的计算和参数分配到多个设备上，这需要层与层之间的大量通信
>  因此，它们在单个节点内表现良好，因为节点内的 GPU 间带宽高，但超出单个节点，效率就会迅速下降
>  我们在两个 DGX-2 节点上用 Megatron-LM 测试了一个 40B 模型，发现每个 V100 的 GPU 性能为 5Tflops，不到硬件峰值的 5%

So, how can we overcome the limitations of existing solutions and train large models more efficiently? To answer this question, we first analyze the full spectrum of memory consumption of the existing systems on model training and classify it into two parts: 1) For large models, the majority of the memory is occupied by model states which include the optimizer states (such as momentum and variances in Adam [6]), gradients, and parameters. 2) The remaining memory is consumed by activation, temporary buffers and unusable fragmented memory, which we refer to collectively as residual states. We develop ZeRO — Zero Redundancy Optimizer — to optimize memory efficiency on both while obtaining high compute and communication efficiency. As these two parts face different challenges, we develop and discuss their solutions correspondingly.
>  我们分析了现存系统在模型训练时的内存消耗全貌，将其分为两个部分
>  1. 对于大模型，内存主要由模型状态组成，模型状态包括了优化器状态 (例如 Adam 中的动量和方差)、梯度、参数
>  2. 剩余的内存被激活值、临时缓存、不可用的碎片化内存消耗，我们将它们称为残留状态
>  我们开发 ZeRO 来优化这两个部分的内存效率，由于这两个部分面临不同的挑战，我们分别开发并讨论了相应的解决方案

**Optimizing Model State Memory** Model states often consume the largest amount of memory during training, but existing approaches such as DP and MP do not offer satisfying solution. DP has good compute/communication efficiency but poor memory efficiency while MP can have poor compute/communication efficiency. More specifically, DP replicates the entire model states across all data parallel process resulting in redundant memory consumption; while MP partition these states to obtain high memory efficiency, but often result in too fine-grained computation and expensive communication that is less scaling efficient. Furthermore, all of these approaches maintain all the model states required over the entire training process statically, even though not all model states are required all the time during the training. 
>  优化模型状态内存
>  DP 具有良好的计算/通信效率 (不用通信)，但内存效率低 (复制模型)
>  MP 的计算/通信效率低 (要通信)，但内存效率高 (划分模型)
>  DP 将整个模型状态复制为多个 replica，故存在冗余内存消耗
>  MP 将模型状态划分，内存效率高，但通信和计算控制使得它拓展效率低
>  DP 和 MP 都要求在整个训练期间维持所有模型状态，即便事实是训练期间并不一直需要完整的模型状态

Based on these observations, we develop ZeRO-DP, ZeRO-powered data parallelism, that achieves the computation/communication efficiency of DP while achieving memory efficiency of MP. ZeRO-DP removes the memory state redundancies across data-parallel processes by partitioning the model states instead of replicating them, and it retains the compute/communication efficiency by retaining the computational granularity and communication volume of DP using a dynamic communication schedule during training.
>  我们提出 ZeRO-DP，它是基于 ZeRO 的数据并行方法，在保留了 DP 的计算/通信效率的同时也达到了 MP 的内存效率
>  ZeRO-DP 移除了数据并行下的冗余内存状态，方法是划分模型状态，而不是复制模型状态，ZeRo-DP 保留了数据并行的计算和内存效率，方法是通过训练时的动态通信调度来保留 DP 的计算粒度和通信量

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/25877288-f38c-4883-a49c-40ce8af650c2/a27e86a05349f9c6fe1887214c36cddf69ca506ca2b090cdf0080e408b3d5485.jpg)  

Figure 1: Comparing the per-device memory consumption of model states, with three stages of ZeRO-DP optimizations.  $\Psi$  denotes model size (number of parameters),  $K$  denotes the memory multiplier of optimizer states, and    denotes DP degree. In the example, we assume a model size of  $\Psi = 7.5B$  and DP of  $N_{d} = 64$  with  $K = 12$  based on mixed-precision training with Adam optimizer.

ZeRO-DP has three main optimization stages (as depicted in Figure 1), which correspond to the partitioning of optimizer states, gradients, and parameters. When enabled cumulatively: 
1) Optimizer State Partitioning  $(P_{os})$  : 4x memory reduction, same communication volume as DP; 
2) Add Gradient Partitioning  $(P_{os + g})$  : 8x memory reduction, same communication volume as DP; 
3) Add Parameter Partitioning  $(P_{os + g + p})$  : Memory reduction is linear with DP degree  $N_{d}$ . For example, splitting across 64 GPUs  $(N_{d} = 64)$  will yield a 64x memory reduction. There is a modest  $50\%$  increase in communication volume.

>  ZeRO-DP 有三个主要优化阶段，对应于划分优化器状态、划分梯度、划分参数
>  1. 优化器状态划分可以带来 4x 的内存减少，且通信量和 DP 一致
>  2. 加上梯度划分可以进一步带来 8x 的内存减少，且通信量仍然和 DP 一致
>  3. 加上参数划分后，内存减少和 DP 程度 $N_d$ 呈线性关系，例如 64 路 DP (划分到 64 个设备上) 就是 64x 内存减少，通信量比 DP 高了 50%

ZeRO-DP eliminates memory redundancies and makes the full aggregate memory capacity of a cluster available. With all three stages enabled, ZeRO can train a trillion-parameter model on just 1024 NVIDIA GPUs. A trillion-parameter model with an optimizer like Adam [6] in 16-bit precision requires approximately 16 terabytes (TB) of memory to hold the optimizer states, gradients, and parameters. 16TB divided by 1024 is 16GB, which is well within a reasonable bound for a GPU (e.g., with 32GB of on-device memory).
>  ZeRO-DP 消除了内存冗余，使得整个集群的全部内存容量可以被充分利用
>  启用全部三个阶段的情况下，ZeRO 可以在 1024 个 GPU 上训练 1T 的模型
>  一个用 Adam 的 1T 模型，在 16bit 精度下需要大约 16TB 保存优化器状态、梯度和参数，16TB/1024 = 16GB，这可以在单个 GPU 下容纳

**Optimizing Residual State Memory** After ZeRO-DP boosts memory efficiency for model states, the rest of the memory consumed by activations, temporary buffers, and unusable memory fragments could become a secondary memory bottleneck. We develop ZeRO-R to optimize the residual memory consumed by these three factors respectively.
1) For activations (stored from forward pass in order to perform backward pass), we noticed checkpointing [7] helps but not sufficient for large models. Thus ZeRO-R optimizes activation memory by identifying and removing activation replication in existing MP approaches through activation partitioning. It also offloads activations to CPU when appropriate.
2) ZeRO-R defines appropriate size for temporary buffers to strike for a balance of memory and computation efficiency.
3) We observe fragmented memory during training due to variations in the lifetime of different tensors. Lack of contiguous memory due to fragmentation can cause memory allocation failure, even when enough free memory is available. ZeRO-R proactively manages memory based on the different lifetime of tensors, preventing memory fragmentation.

>  优化残留状态内存
>  在 ZeRO-DP 提升了模型状态的内存效率后，由激活值、临时缓存和不可用内存碎片消耗的剩余内存会成为另一个内存瓶颈
>  我们开发 ZeRO-R 来优化这三方面所消耗的残留内存
>  1. 对于激活值，我们注意到梯度检查点技术有所帮助，但对于大模型仍然不够，因此，ZeRO-R 通过激活划分，来识别和移除现有模型并行方法中的激活冗余从而优化激活内存，它还会在适当的时候将激活值卸载到 CPU
>  2. 对于临时缓冲区，ZeRO-R 会定义合适的大小，在内存和计算效率之间取得平衡
>  3. 我们观察到训练过程中不同张量生命周期的差异会导致内存碎片化，ZeRO-R 会根据张量的不同生命周期主动管理内存，防止内存碎片化

ZeRO-DP and ZeRO-R combined together forms a powerful system of memory optimizations for DP training that we collectively refer to as ZeRO.
>  ZeRO-DP + ZeRO-R 结合在一起，就得到了 DP 训练的内存优化系统，称为 ZeRO

**ZeRO and MP**: Since ZeRO eliminates the memory inefficiency in DP, it is natural to ask: Do we still need MP, and when? How does ZeRO work with MP? With ZeRO, MP becomes a less attractive option for the purpose of fitting large models alone. ZeRO-DP is at least as effective on reducing per-device memory footprint as MP, or more effective sometimes when MP cannot divide the model evenly. It also has comparable or better scaling efficiency. Furthermore, data parallelism is so easy to use that it is widely applicable across different workloads, while MP approaches today often need some work from model developers to revise their model, system developers to work out distributed operators, and existing work like Megatron-LM only supports a limited set of operators and models.
>  ZeRO-DP 在减少每个设备的内存占用方面至少和 MP 一样有效，如果 MP 无法均匀划分模型，ZeRO-DP 甚至更有效
>  ZeRO-DP 的拓展效率和 MP 可比甚至更好
>  此外，DP 易于使用，适用于各种 workload，MP 则需要模型开发者修改模型，系统开发者实现分布式算子
>  现有的工作例如 Megatron-LM 仅支持有限的算子和模型

That being said, there are still cases where we want to leverage MP: i) When used with  $ZeRO$ -R, MP can reduce activation memory footprint for very large models. ii) For smaller models where activation memory is not an issue, MP can also have benefits when aggregated batch size using DP alone is too big to have good convergence. In those case, one can combine  $ZeRO$  with MP to fit the model with an acceptable aggregated batch size.
>  在某些情况下，我们仍然希望利用 MP
>  1. 和 ZeRO-R 结合使用时，MP 可以减少非常大模型的激活内存占用
>  2. 对于激活内存不是问题的小模型，当仅使用 DP 时整体 batch size 太大而优化效果不佳时，可以结合 MP 使得整体 batch size 大小更适配模型

We show that  $ZeRO$  can be combined with MP, resulting in a max theoretical memory reduction of  $N_d \times N_m$  times on each device with a DP degree of  $N_d$  and MP degree of  $N_m$ . This could allow us to fit a trillion parameter model on 1024 GPUs with 16-way model parallelism (within each DGX2 node) and 64-way data parallelism across nodes, and run it efficiently using a modest batch size!
>  ZeRO 和 MP 结合，可以在每个设备上实现理论上最大 $N_d \times N_m$ 的内存减少，其中数据并行度为 $N_d$，模型并行度为 $N_m$
>  这使得我们可以在 1024 GPU 上训练 1T 模型，使用 16 路模型并行 (每个 DGX2 节点内)，和 64 路数据并行 (DGX2 节点之间)，使用适中的 batch 大小就能高效运行

**Implementation & Evaluation** The complete set of optimizations in  $ZeRO$  could allow us to run models with trillion parameters on the high-end hardware cluster today (e.g., with 1K V100 GPUs), however, the hardware compute capacity is still too limited and training time can be impractically long ( $>1$  year). Therefore, our focus for this implementation is to efficiently support models with 10x parameters ( $\sim 100$ B parameters) than state-of-the-art (SOTA) while still being within reach of the compute capabilities of current hardware. We implement and evaluate a subset of optimizations in  $ZeRO$  called  $ZeRO -100$ B —  $P_{os + g}$  of  $ZeRO$ -DP plus  $ZeRO$ -R — that allow us to achieve this goal. The results show:
>  ZeRO 的完整优化方案可以使得我们在如今的高端硬件模型上训练具有万亿参数的模型，然而，硬件计算能力仍然有限，训练时间可能不切实际 (超过 1 年)
>  因此我们实现的重点是高效支持 10x SOTA 模型参数量的模型 (~100B)
>  我们实现并评估了 ZeRO 的优化的一个子集: ZeRO-DP 的 $P_{os+g}$ +  ZeRO-R，称为 ZeRO-100B

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/25877288-f38c-4883-a49c-40ce8af650c2/4e58cbda7f127584a66860b0318b89ce54aef1421900333cbc9ea1964b5f7282.jpg)  

Figure 2:  $ZeRO$  training throughput and speedup w.r.t SOTA baseline for varying model sizes. For  $ZeRO$ , the MP always fit in a node, while for baseline, models larger than 40B require MP across nodes.

*Model Size* Combined with MP,  $ZeRO -100$ B runs 170B parameter models efficiently, while the existing system like using Megatron alone cannot scale efficiently beyond 40B parameters, as shown in Figure 2. This is an over 8x increase in model size compared to SOTA.
>  模型规模方面，ZeRO-100B 可以运行 170B 的模型，现存系统无法拓展超过 40B 参数

*Speed* Improved memory efficiency powers higher throughput and faster training. As shown in Figure 2,  $ZeRO$  runs 100B parameter models on a 400 Nvidia V100 GPU cluster with over 38 TFlops per GPU, and aggregate performance over 15 Petaflops. This is more than 10x improvement in training speed compared to SOTA for the same model size.
>  速度方面，内存效率使得吞吐更高，训练更快
>  ZeRo 在 400 个 GPU 上运行 100B 模型达到总吞吐高于 15Pflops，这是 SOTA 系统运行相同大小模型的 10x

*Scalability* We observe super linear speedup in the regime of 64-400 GPUs, where the performance more than doubles when we double the number of GPUs. This is a property of ZeRO-DP which reduces the memory footprint of the model states as we increase the DP degree allowing us to fit larger batch sizes per GPU resulting in better performance. We expect this behaviour to continue further as we increase the number of GPUs beyond 400.
>  可拓展性方面，我们发现 64-400 GPU 的加速比是超线性，也就是双倍 GPU 数量，性能超过了双倍
>  这是 ZeRO-DP 的特定，它随着 DP degree 的增加而减少模型状态的内存占用，从而允许每块 GPU 装载更大的 batch 大小

*Democratization of Large Model Training* ZeRO-100B powers data scientist to train models with up to 13B parameters without any MP or PP that requires model refactoring, where 13B is more parameters than the largest model in literature (T5 with 11B parameters). Data scientists can thus experiment freely with large models without worrying about parallelism. In comparison, exist systems (e.g., PyTorch Distributed Data Parallel) runs out of memory with 1.4B parameter models.
>  大模型训练的民主化
>  ZeRO-100B 能够在不进行需要模型重构的 MP, PP 的前提下训练 13B 的模型

*New SOTA Model* ZeRO powers the largest language model with 17B parameters and record-breaking accuracy, Turing-NLG [9].

We share ZeRO as a part of our open source DL training optimization library called DeepSpeed. We plan to release all implementations described in this paper by end of May 2020 and extend it further to support 1 trillion parameters by enabling ZeRO-DP stage 3 partitioning parameters  $(P_{os + g + p})$ . We plan to make ZeRO fully accessible to the DL community to catalyze the evolution and democratization of large model training at scale.
>  我们将 ZeRO 作为开源 DL 训练优化库 DeepSpeed 的一部分

# 2 Related Work
## 2.1 Data, Model and Pipeline Parallelism
Parallelization is a key strategy on training large models at scale. For a model that fits in the device memory for training, data parallelism (DP) is used to scale training to multiple devices. In DP, model parameters are replicated on each device. At each step, a mini-batch is divided evenly across all the data parallel processes, such that each process executes the forward and backward propagation on a different subset of data samples, and uses averaged gradients across processes to update the model locally.
>  模型可以放入单个设备内存时，可以用 DP 将模型训练拓展到多个设备
>  DP 中，模型参数被复制到多个设备, mini-batch 被均匀划分给所有的 DP 进程，每个 DP 进程在数据子集上执行前向和反向传播，并使用全局的平均梯度来更新本地模型

When a model does not fit in the device memory, model parallelism (MP) [5, 3] and pipeline parallelism (PP) [10, 11] split the model among processes, in vertical and horizontal way respectively. Sec. 1 discussed how  $ZeRO$  relates to DP and MP. We now discuss PP and how it relates to reducing memory consumption.
>  模型放不下单个设备内存时，可以采用模型并行和流水线并行，分别在垂直方向和水平方向划分模型

PP splits a model horizontally across layers running each partition on a different device and use micro-batching to hide the pipeline bubble [10, 11]. Model functionalities such as tied-weights and batch-normalization are difficult to implement due to horizontal splitting and micro-batching, respectively. 
>  流水线并行在水平方向划分模型，每个 partition 在不同的设备上运行，并使用 micro-batching 来掩盖流水线气泡
>  由于水平划分，流水线并行难以实现权重共享，由于 micro-batching，流水线并行难以实现 batch norm

Popular PP implementation such as G-pipe [10] partitions both model parameters and total activations but requires a batch size proportional to number of pipeline partitions to hide the pipeline bubble. The large batch size can affect the convergence rate, while also requiring significant memory to store activations. A different implementation of PP in PipeDream [12] keeps multiple copies of stale parameters to hide the pipeline bubble without increasing the batch size significantly, making it less memory efficient. Additionally, the implementation is not equivalent to the standard DL training and has implications on training convergence. In contrast,  $ZeRO$  obtains the same or better memory efficiency than PP without incurring functionality, performance and convergence related restrictions of PP.
>  G-pipe 划分模型参数和激活值，但要求 batch size 和 micro batch size 成比例，以隐藏流水线气泡，太大的 batch size 会影响收敛，同时也要求更多内存存储激活值
>  PipeDream 保留多个过时参数的副本来隐藏流水线气泡，而不会显著增加 batch size，但这使得其内存效率过低
>  相比直线，ZeRO 在不带来关于 PP 的性能和收敛限制的情况下获得了比 PP 更好的内存效率

## 2.2 Non-parallelism based approach to reduce memory
In addition to MP and PP, there are multiple lines of work that target reducing memory overheads of DL training.
>  也存在基于非并行的方法来减少 DL 训练的内存的工作

### 2.2.1 Reducing Activation Memory
Multiple efforts have focused on reducing the memory footprint of activations through compression [13], activation checkpointing [7, 14], or live analysis [15]. These efforts are complimentary and can work together with  $ZeRO$ . In fact, activation memory reduction in  $ZeRO$ -R works in parallel with activation checkpointing.
>  一些工作聚焦于较少激活值的内存占用，包括压缩、检查点、活跃性分析
>  这些方法可以和 ZeRO 一同使用，实际上，ZeRO-R 中的激活内存减少可以和激活检查点一同使用

### 2.2.2 CPU Offload
[16, 17] exploit heterogeneous nature of today's compute nodes, offloading model states to CPU memory through algorithmic design of virtualized memory, respectively. Up to  $50\%$  of training time can be spent on GPU-CPU-GPU transfers [16].  $ZeRO$  differs in that it reduces the memory consumption significantly without storing the model states to CPU memory whose bandwidth is severely constrained due to PCI-E. On rare cases,  $ZeRO$ -R may offload just the activation checkpoints for very large models to improve performance (see Sec. 6.1 for details).
>  一些工作设计了虚拟化内存，将模型状态卸载到 CPU 内存，但会导致超过 50% 的训练时间花费在 GPU-CPU-GPU 的数据传输 (速度受 PCI-E 带宽限制)
>  ZeRO 不需要将模型状态卸载到 GPU，也可以显著减少内存消耗
>  在极少的情况下，ZeRO-R 会将激活检查点卸载，仅针对非常大的模型

### 2.2.3 Memory Efficient Optimizer
[18, 19] focus on reducing memory consumption of adaptive optimization methods by maintaining coarser-grained statistics of model parameters and gradients, with potential impact on model convergence guarantees.  $ZeRO$  is orthogonal to these efforts, and its optimizations do not change the model optimization method or affect model convergence, but effectively reduce memory footprint of optimizer states and gradients per device.
>  一些工作通过设计自适应的优化方法，维护模型参数和梯度的粗粒度统计量，来减少训练内存消耗，这可能会影响模型收敛保证
>  ZeRO 和这些方法正交，ZeRO 不会影响模型的优化方法或收敛，但会减少每个设备上优化器状态和梯度的内存

## 2.3 Training Optimizers
Adaptive optimization methods [20, 6, 21, 22] are crucial to achieving SOTA performance and accuracy for effective model training of large models. Compared to SGD, by maintaining fine-grained first-order and second-order statistics for each model parameter and gradient at the cost of significant memory footprint.  $ZeRO$  can reduce the memory footprint of these optimizers by orders of magnitude, making these sophisticated optimization methods practical for training large models on hardware with modest device memory. It also makes it possible to develop and use even more complex and memory hungry optimizers that may have better convergence.
>  自适应的优化方法对模型的训练影响很大，但相较于 SGD，为每个模型参数和梯度维护细粒度的一阶和二阶统计量会显著提高内存使用
>  ZeRO 可以将这些优化器的内存占用降低几个数量级

# 3 Where Did All the Memory Go?
Let's take a step back to examine the memory consumption of the current training system. For example, a 1.5B parameter GPT-2 model requires 3GB of memory for its weights (or parameters) in 16-bit precision, yet, it cannot be trained on a single GPU with 32GB memory using Tensorflow or PyTorch. One may wonder where all the memory goes. During model training, most of the memory is consumed by model states, i.e., tensors comprising of optimizer states, gradients, and parameters. Besides these model states, the rest of the memory is consumed by activations, temporary buffers and fragmented memory which we call residual states. We look at the memory consumption from both in details.
>  我们探究当前训练系统的内存消耗
>  例如，1.5B 的 GPT-2 模型在 16 位精度下需要存储 3GB 的权重/参数，然而，它却无法在单个 32GB 机器上使用 TensorFlow 或 PyTorch 训练
>  在模型训练中，大多数内存都被模型状态消耗，即由优化器状态、梯度、参数构成的张量
>  剩余的内存则被激活、临时缓存和碎片化内存消耗，我们称为残留状态

## 3.1 Model States: Optimizer States, Gradients and Parameters
Majority of the device memory is consumed by model states during training. Consider for instance, Adam [6], one of the most popular optimizers for DL training. Adam requires storing two optimizer states, i) the time averaged momentum and ii) variance of the gradients to compute the updates. Therefore, to train a model with ADAM, there has to be enough memory to hold a copy of both the momentum and variance of the gradients. In addition, there needs to be enough memory to store the gradients and the weights themselves. Of these three types of the parameter-related tensors, the optimizer states usually consume the most memory, specially when mixed-precision training is applied.
>  大多数设备内存被训练时的模型状态消耗
>  例如 Adam 需要存储两个优化器状态: 1. 时间平均的动量 2. 梯度的方差，来计算更新
>  因此，使用 Adam 训练模型就要求有足够的内存存储梯度的动量和方差，同时需要存储梯度和权重本身
>  在这些和参数相关的张量中，优化器状态通常消耗最多的内存，尤其是在应用了混合精度训练时

**Mixed-Precision Training** The state-of-the-art approach to train large models on the current generation of NVIDIA GPUs is via mixed precision (fp16/32) training [23], where parameters and activations are stored as fp16, enabling the use of the high throughput tensor core units [24] on these GPUs. During mixed-precision training, both the forward and backward propagation are performed using fp16 weights and activations. However, to effectively compute and apply the updates at the end of the backward propagation, the mixed-precision optimizer keeps an fp32 copy of the parameters as well as an fp32 copy of all the other optimizer states.
>  在 NVIDIA GPU 训练大模型的最新方法是使用混合精度 (fp16/32) 训练，其中参数和激活存储为 fp16, 以利用 tensor core 的高吞吐
>  混合精度训练中，前向和反向计算都使用 fp16 的权重和激活进行，但为了在反向计算完成后应用梯度，混合精度优化器需要保存参数的 fp32 拷贝，以及其他优化器状态的 fp32 拷贝 (计算用参数的 fp16 版本以利用 tensor core 追求速度，但是为了数值稳定性，还需要维护 fp32 版本的参数用于更新)

Let's take Adam as a concrete example. Mixed precision training of a model with  $\Psi$  parameters using Adam requires enough memory to hold an fp16 copy of the parameters and the gradients, with memory requirements of  $2\Psi$  and  $2\Psi$  bytes respectively. In addition, it needs to hold the optimizer states: an  $fp32$  copy of the parameters, momentum and variance, with memory requirements of  $4\Psi$ ,  $4\Psi$ , and  $4\Psi$  bytes, respectively. Let's use  $K$  to denote the memory multiplier of the optimizer states, i.e., the additional memory required to store them is  $K\Psi$  bytes. Mixed-precision Adam has  $K = 12$ . In total, this results in  $2\Psi + 2\Psi + K\Psi = 16\Psi$  bytes of memory requirement. For a model such as GPT-2 with 1.5 Billion parameters, this leads to a memory requirement of at least  $24GB$ , which is significantly higher than the meager  $3GB$  of memory required to hold the  $fp16$  parameters alone.
>  以 Adam 为例，使用 Adam 混合精度训练一个 $\Psi$ 参数量的模型要求有足够内存保存参数和梯度的 fp16 拷贝，故分别需要 $2\Psi$ 字节和 $2\Psi$ 的内存
>  此外，需要保存优化器状态: 参数、动量、方差的 fp32 拷贝，各自需要 $4\Psi$ 的内存
>  我们用 $K$ 表示优化器状态的内存乘数，即存储优化器状态所需要的额外内存为 $K\Psi$ 字节，那么混合精度 Adam 训练的 $K=12$
>  那么总共需要的内存就是 $16\Psi$，对于 1.5B 参数的模型，内存需求就是 24GB，这显著高于单单存储一份 fp16 参数所需要的 3GB 内存

## 3.2 Residual Memory Consumption
**Activations** can take up a significant amount of memory [7] during training. As a concrete example, the 1.5B parameter GPT-2 model trained with sequence length of 1K and batch size of 32 requires about  $60\mathrm{GB}$  of memory<sup>3</sup>. Activation checkpointing (or activation recomputation) is a common approach to reduce the activation memory by approximately the square root of the total activations at the expense of  $33\%$  re-computation overhead [7]. This would reduce the activation memory consumption of this model to about 8GB.
>  激活会在训练中占据很多内存，例如 1.5B 的 GPT-2，使用 1K 的序列和 32
>  batch size 训练时，将需要 60GB 的内存
>  激活检查点是常用的减少激活占用的方法，可以以 33% 的额外计算开销近似将内存消耗减少到次线性，也就是近似将激活内存占用减少到 8GB

Despite the significant reduction, the activation memory can grow quite large for bigger models even with activation checkpointing. For example, a GPT-like model with 100 billion parameters requires around  $60\mathrm{GB}$  of memory for batch size 32, even when using activation checkpointing.
>  即便如此，对于大的模型，存储激活仍然需要很多内存，例如 100B 的模型使用激活检查点也需要 60GB 的内存来存储

**Temporary buffers** used for storing intermediate results consumes non-trivial amount of memory for large models. Operations such as gradient all-reduce, or gradient norm computation tend to fuse all the gradients into a single flattened buffer before applying the operation in an effort to improve throughput. For example, the bandwidth of all-reduce across devices improves with large message sizes. While the gradient themselves are usually stored as fp16 tensors, the fused buffer can be an fp32 tensor depending on the operation. When the size of the model is large, these temporary buffer sizes are non-trivial. For example, for a model with 1.5B parameters, a flattened fp32 buffer would required  $6GB$  of memory.
>  对于大型模型来说，用于存储中间结果的临时缓冲区也会消耗许多内存
>  例如，梯度 all-reduce 操作和梯度范数计算 (用于梯度裁剪以防止训练不稳定) 通常会将所有梯度融合到一个扁平化的缓冲区中，然后再进行计算以提高吞吐量
>  例如，跨设备的 all-reduce 带宽随着消息大小的增加而提升，虽然梯度本身通常以 fp16 张量的形式存储，但融合后的缓冲区可能根据操作的不同而成为 fp32 张量
>  如果模型比较大，例如 1.5B 的模型，一个扁平化的 fp32 缓冲区将需要 6GB 的内存

>  GPU 之间的通信有固定的建立连接和同步的开销，如果发送很多小消息，例如每层发一次梯度，相较于将所有层的梯度合并为一个大消息，通信效率就会更低

**Memory Fragmentation**: So far we have discussed the actual memory consumption during training. Additionally, it is possible to run out of usable memory even when there is plenty of available memory. This can happen with memory fragmentation. A request for a memory will fail if there isn't enough contiguous memory to satisfy it, even if the total available memory is larger than requested. We observe significant memory fragmentation when training very large models, resulting in out of memory issue with over  $30\%$  of memory still available in some extreme cases.
>  我们发现训练非常大的模型时存在显著的内存碎片问题，导致了在一些极端情况下，即便还有超过 30% 的内存可用，也会出现内存不足的问题

# 4 ZeRO: Insights and Overview
ZeRO has two sets of optimizations: i) ZeRO-DP aimed at reducing the memory footprint of the model states, and ii) ZeRO-R targeted towards reducing the residual memory consumption. We present an overview of the optimizations and the insights behind, which allows ZeRO to reduce memory footprint while remaining efficient. Please note efficiency is a key here: without this constraint, trivial solutions like moving all the parameter states to the CPU memory, or increasing the MP degree arbitrarily can reduce memory footprint.
>  ZeRO 有两类优化:
>  1. ZeRO-DP 的目标是减少模型状态的内存消耗
>  2. ZeRO-R 的目标是减少残留内存的消耗
>  在优化时，我们关注的是效率，如果不考虑效率，简单的方案例如将参数状态卸载到 CPU 内存，或者任意增加模型并行度也可以减少内存占用

## 4.1 Insights and Overview: ZeRO-DP
ZeRO powered DP is based on three key insights:

a) DP has better scaling efficiency than MP because MP reduces the granularity of the computation while also increasing the communication overhead. Beyond a certain point, lower computational granularity reduces the efficiency per GPU, while the increased communication overhead, hiders the scalability across GPUs, especially when crossing node boundaries. On the contrary, DP has both higher computational granularity and lower communication volume, allowing for much higher efficiency.
b) DP is memory inefficient as model states are stored redundantly across all data-parallel processes. On the contrary, MP partitions the model states to obtain memory efficiency.
c) Both DP and MP keep all the model states needed over the entire training process, but not everything is required all the time. For example, parameters corresponding to each layer is only needed during the forward propagation and backward propagation of the layer.

>  ZeRO-DP 基于三个关键思想:
>  1. 与 MP 相比，DP 的拓展效率更好，因为 MP 会**降低计算的粒度**，同时增加通信开销，MP 超过一定规模时，较低的计算粒度会减少每个 GPU 的效率，增加的通信开销会阻碍跨 GPU 的可拓展性，尤其是需要跨节点通讯时；相反，DP 的计算粒度更高，通信量也更低，进而有更好的效率
>  2. DP 的内存效率很低，因为模型状态会在所有数据并行进程中冗余存储，相反，MP 通过划分模型来提高内存效率
>  3. DP 和 MP 都会在整个训练过程中保留所有所需的模型状态，但并非所有内容在任何时候都是必须的，例如每一层的参数仅在该层的前向传播和反向传播期间需要

Based on these insights, ZeRO-DP retains the training efficiency of DP while achieving the memory efficiency of MP. ZeRO-DP partitions the model states instead of replicating them (Section 5) and uses a dynamic communication schedule that exploits the intrinsically temporal nature of the model states while minimizing the communication volume (Section 7). By doing so, ZeRO-DP reduces per-device memory footprint of a model linearly with the increased DP degree while maintaining the communication volume close to that of the default DP, retaining the efficiency.
>  基于这些思想，ZeRO-DP 在保留了 DP 的训练效率的同时达到了 MP 的内存效率
>  ZeRO-DP 对模型状态进行划分而不是复制，并使用一种动态通信调度机制，利用了模型状态固有的时间性质，最小化通信量
>  由此，ZeRO-DP 随着 DP 度的增加线性地降低了每个设备的内存占用，同时将通信量保持在接近默认 DP 的通信量的水平，维持了整体效率

## 4.2 Insights and Overview: ZeRO-R
### 4.2.1 Reducing Activation Memory
Two key insights are:

a) MP partitions the model states but often requires replication of the activation memory. For example, if we split the parameters of a linear layer vertically and compute them in parallel across two GPUs, each GPU requires the entire activation to compute its partition
b) For models such as GPT-2 or larger, the arithmetic intensity (ratio of the amount of computation per iteration to amount of activation checkpoints per iteration) is very large  $(\geq 10K)$  and increases linearly with hidden dimension making it possible to hide the data-movement cost for the activation checkpoints, even when the bandwidth is low.

>  ZeRO-R 的两个关键思想是:
>  1. MP 会划分模型状态，但通常需要复制激活内存，例如，我们将一个线性层的参数垂直分割，在两个 GPU 上并行计算，每个 GPU 都需要完整的激活值 (输入) 来计算自己对应的部分
>  2. 对于像 GPT-2 或更大的模型来说，其算术密度 (每次迭代的计算量与每次迭代的激活检查点数据量之比) 非常大，并且随着隐藏层维度增大而线性增大，这使得即时在带宽较低的情况下，也可以隐藏激活检查点的数据移动成本

ZeRO removes the memory redundancies in MP by partitioning the activations checkpoints across GPUs, and uses allgather to reconstruct them on demand. The activation memory footprint is reduced proportional to the MP degree. For very large models, ZeRO can even choose to move the activation partitions to the CPU memory, while still achieving good efficiency due to large arithmetic intensity in these models.
>  ZeRO 通过在 GPU 之间划分激活检查点来消除 MP 的内存冗余，并使用 all-gather 按需重构这些检查点
>  激活内存的使用量会随着 MP degree 成比例减少
>  对于非常大的模型，ZeRO 甚至可以选择将激活检查点移动到 CPU 内存中，由于这些模型具有较大的算术密度，因此仍然可以保持良好的效率

### 4.2.2 Managing Temporary buffers
ZeRO-R uses constant size buffers to avoid temporary buffers from blowing up as the model size increases, while making them large enough to remain efficient.
>  ZeRO-R 使用固定大小的缓存来避免随着模型增大缓冲区爆炸，同时确保缓冲区足够大以保持效率

### 4.2.3 Managing fragmented Memory
Memory fragmentation is a result of interleaving between short lived and long lived memory objects. During the forward propagation activation checkpoints are long lived but the activations that recomputed are short lived. Similarly, the backward computation, the activation gradients are short lived while the parameter gradients are long lived. Based on this insight, ZeRO performs on-the-fly memory defragmentation by moving activation checkpoints and gradients to pre-allocated contiguous memory buffers. This not only increases memory availability but also improves efficiency by reducing the time it takes for the memory allocator to find free contiguous memory.

>  内存碎片是由短期和长期内存对象的交错导致的
>  在前向传播过程中，激活检查点是长期存在的，而(需要在反向传播时被) 重计算的激活值则是短期存在的 (会被丢弃)
>  类似地，在反向计算中，激活梯度是短期存在的 (中间输出相对于损失的梯度，用于计算中间层的参数梯度)，参数梯度则是长期存在的
>  基于这个观察，ZeRO 通过将激活检查点和梯度 (长期存在的内存对象) 移动到连续的内存缓冲区，实现了实时的内存去碎片化
>  这不仅增加了可用内存，还通过减少内存分配器找到连续空闲内存的时间提高了效率

# 5 Deep Dive into ZeRO-DP
While the existing DP approach replicates the model states at each device and introduces significant memory overhead, ZeRO-DP eliminates this memory redundancy by partitioning them — optimizer states, gradients and parameters — across data parallel processes. Figure 1 quantifies and visualizes the memory requirement with and without ZeRO-DP. The figure shows the memory footprint after partitioning (1) optimizer state, (2) gradient and (3) parameter redundancies accumulatively. We refer to them as the three optimization phases of ZeRO-DP:  $P_{os}$ ,  $P_{g}$ , and  $P_{p}$ , which we elaborate below.
>  现存的 DP 方法将模型状态复制到各个设备，引入了显著的内存开销
>  ZeRO-DP 通过将模型状态 (包括优化器状态、梯度、参数) 划分给 DP 进程，消除了这些冗余

## 5.1 $P_{os}$ : Optimizer State Partitioning
For a DP degree of  $N_{d}$ , we group the optimizer states into  $N_{d}$  equal partitions, such that the  $i^{th}$  data parallel process only updates the optimizer states corresponding to the  $i^{th}$  partition. Thus, each data parallel process only needs to store and update  $\frac{1}{N_{d}}$  of the total optimizer states and then only update  $\frac{1}{N_{d}}$  of the parameters. We perform an all-gather across the data parallel process at the end of each training step to get the fully updated parameters across all data parallel process.
>  DP degree 为 $N_d$ 时，我们将优化器状态划分为 $N_d$ 个同等大小的 partition，使得第 $i$ 个数据并行进程仅更新第 $i$ 个 partition 的优化器状态
>  这样，每个数据并行进程仅需要存储 $\frac 1 {N_d}$ 的总优化器状态，并且只更新 $\frac 1 {N_d}$ 的参数
>  我们在每个训练步骤结束时，在所有数据并行进程中执行 all-gather 操作，以获取完整的更新后的参数

>  每个设备仍然保存了完整的 fp16 模型参数/权重，进而可以进行计算
>  优化器状态包括了 fp32 的参数、动量、方差，实际上每个设备都可以自行计算完整的优化器状态，但仅仅计算并保存其中自己负责的 $\frac 1 {N_d}$，进而仅更新 $\frac 1 N_d$ 的参数，以减小内存占用
>  在所有设备完成参数更新后，再执行 all-gather 获得完整参数

**Memory Savings:** As shown in Figure 1, the memory consumption after optimizing state partition reduces from  $4\Psi + K\Psi$  to  $4\Psi + \frac{K\Psi}{N_{d}}$ . As the concrete example depicted in Figure 1, a 7.5 B parameter model requires 31.4GB of memory using  $P_{os}$  with 64-way DP ( $N_{d} = 64$ ), while requiring 120 GB with standard DP. Furthermore, when  $N_{d}$  is large, the memory requirement on model states reduces from  $4\Psi + 12\Psi = 16\Psi$  bytes to  $4\Psi + \frac{12\Psi}{N_{d}} \approx 4\Psi$  bytes, leading to a 4x reduction.
>  执行了优化器状态划分之后，内存占用从 $4\Psi + K\Psi$ 减少到 $4\Psi + \frac {K\Psi}{N_d}$
>  我们知道混精的 Adam 训练的系数 $K = 12$，当 DP degree 较大时，就可以将 $4\Psi + 12 \Psi = 16\Psi$ 减少到 $4\Psi + \frac {12\Psi}{N_d} \approx 4\Psi$ 字节，即 4x 倍的占用量减少

## 5.2 $P_{g}$ : Gradient Partitioning
As each data parallel process only updates its corresponding parameter partition, it only needs the reduced gradients for the corresponding parameters. Therefore, as each gradient of each layer becomes available during the backward propagation, we only reduce them on the data parallel process responsible for updating the corresponding parameters. After the reduction we no longer need the gradients and their memory can be released. This reduces the memory footprint required to hold the gradients from  $2\Psi$  bytes to  $\frac{2\Psi}{N_{d}}$ .
>  因为每个数据并行进程仅更新自己对应的参数 partition，因此仅需要对应的梯度 partition 即可
>  因此，当反向传播过程中每一层的梯度可用时，每个设备都会把梯度拆开来，然后根据梯度的 partition，仅和负责这个 partition 的设备进行 reduce 计算，使得这个设备获得这个 partition 的规约后梯度
>  在和这个设备规约后，这个 partition 的梯度就不再需要了 (因为我负责的是更新其他 partition 的参数)，存储这个 partition 梯度的内存就可以倍释放
>  这将存储梯度所要求的内存量从 $2\Psi$ 减少到 $\frac {2\Psi} {N_d}$

>  标准数据并行中，每个设备有自己的梯度，所有设备在迭代后对梯度进行 all-reduce，使得每个设备都保存了完整的梯度
>  而当每个设备仅负责更新一部分参数时，它只需要知道这部分参数的平均梯度，不需要知道全部梯度

Effectively this is a Reduce-Scatter operation, where gradients corresponding to different parameters are reduced to different process. To make this more efficient in practice, we use a bucketization strategy, where we bucketize all the gradients corresponding to a particular partition, and perform reduction on the entire bucket at once. This is similar in spirit to how NVIDIA's AMP [25] optimizer bucketizes the all-reduce gradient computation to overlap communication and computation. In our case we perform a reduction instead of an all-reduce at the partition boundaries to reduce memory footprint and overlap computation and communication.
>  这实际上是一个 Reduce-Scatter 操作，也就是将数据按块划分，每块数据仅在一个 GPU 上完成规约，规约结果进而仅存在那个 GPU 上
>  为了提高这个操作的效率，我们使用分桶策略，每算完一层梯度，就把它加入对应 partition 的梯度桶，当桶满了，就触发一次 Reduce-Scatter，一次性规约桶里的梯度
>  这类似于 NVIDIA 的自动混合精度优化器训练中对 all-reduce 梯度计算的分桶策略

>  如果把梯度全部算完再通信，则显存可能会爆
>  如果每算完一次梯度就通信，则通信太频繁，效率低
>  这种方案是分桶，例如桶的大小为 2 层的梯度，通满了之后就通信一次，这样通信和计算也可以重叠

**Memory Savings:** By removing both gradient and optimizer state redundancy, we reduce the memory footprint further down to  $2\Psi + \frac{12\Psi}{N_{d}} \approx 2\Psi$ . As the example in Figure 1, a 7.5 B parameter model requires only 16.6 GB of memory using  $P_{os + g}$  with 64-way DP ( $N_{d} = 64$ ), while requiring 120 GB with standard DP. When $N_d$ is large, the memory requirement of model states reduces from  $2\Psi +14\Psi = 16\Psi$  bytes to  $2\Psi +\frac{14\Psi}{N_d}\approx 2\Psi$  bytes, leading to a 8x reduction.
>  移除了梯度和优化器状态的冗余之后，我们将内存使用量进一步减少到了 $2\Psi + \frac {12 \Psi} {N_d} \approx 2\Psi$，相较于 $16\Psi$ 就是 8x 的占用量减小

## 5.3  $P_{p}$ : Parameter Partitioning
Just as with the optimizer states, and the gradients, each process only stores the parameters corresponding to its partition. When the parameters outside of its partition are required for forward and backward propagation, they are received from the appropriate data parallel process through broadcast. While this may seem to incur significant communication overhead at first glance, we show that this approach only increases the total communication volume of a baseline DP system to 1.5x, while enabling memory reduction proportional to  $N_{d}$ .
>  类似于优化器状态、梯度，每个数据并行进程可以存储对应于其 partition 的对应参数，当前向计算和反向计算过程中需要 partition 之外的参数时，则通过广播从对应的数据并行进程获取
>  这实际上仅会相较于基础的 DP 提高 1.5x 的通讯量，但可以使得内存以 $N_d$ 的倍数减小

>  虽然划分了参数，但并不是模型并行，模型并行是各个参数 shard 计算各自的结果，然后 reduce 得到总结果，这里则是在需要参数的时候获取其他参数 shard，在本地用完整的参数计算结果

**Memory Savings:** With parameter partitioning, we reduce the memory consumption of an  $\Psi$  parameter model from  $16\Psi$  to  $\frac{16\Psi}{N_d}$ . As the example in Figure 1, a 7.5 B parameter model requires 1.9 GB of model-state memory using  $P_{os + p + g}$  with 64-way DP ( $N_{d} = 64$ ), while requiring 120 GB with standard DP. This has a profound implication:  $ZeRO$  powers  $DP$  to fit models with arbitrary size as long as there are sufficient number of devices to share the model states.
>  使用参数划分时，我们将参数量为 $\Psi$ 的模型的内存消耗从 $16\Psi$ 减少到了 $\frac {16\Psi}{N_d}$，这一改进具有很大的意义: ZeRO 可以使得 DP 能够支持任意大小的模型，只要存在足够多的设备来分担模型状态的存储

## 5.4 Implication on Model Size

![[pics/ZeRO-Table1.png]]

The three phases of partitioning  $P_{os}$ ,  $P_{os + g}$ , and  $P_{os + g + p}$  reduces the memory consumption of each data parallel process on model states by up to 4x, 8x, and  $N_{d}$  respectively. Table 1 analyzes model-state memory consumption of a few example models under the 3 stages of  $ZeRO$ -DP optimizations for varying DP degree. Without  $ZeRO$ , the memory consumption is equal to the first row in the table, regardless of the DP degree. Note that, with  $N_{d} = 64$ ,  $ZeRO$  can train models with up to 7.5B, 14B, and 128B parameters using  $P_{os}$ ,  $P_{os + g}$ , and  $P_{os + p + p}$ , respectively. When  $N_{d} = 1024$ ,  $ZeRO$  with all of its optimizations enabled ( $P_{os + g + p}$ ) could train models with 1 TRILLION parameters! Or potentially, models with ARBITRARY size! Without  $ZeRO$ , the largest model DP alone can run has less than 1.5 Billion parameters.
>  $P_{os}, P_{os+g}, P_{os+g+p}$ 分别将每个数据并行进程在模型状态上的内存消耗减少了 4x, 8x 和 $N_d$
>  当 $N_d=64$ 时，启用 ZeRO-DP 的三个阶段可以训练达到 7.5B, 14B, 128B 的模型
>  当 $N_d = 1024$，启用 ZeRO 可以训练 1T 的模型

# 6 Deep Dive into ZeRO-R
## 6.1 $P_a$ : Partitioned Activation Checkpointing
As discussed in 4.2, MP by design requires a replication of the activations, resulting in redundant copies of the activations across model parallel GPUs. ZeRO eliminates this redundancy by partitioning the activations, and only materializes them in a replicated form one activation layer at a time, right before the activation is used in computation. 
>  正如 4.2 节讨论的，MP 由于设计原因需要复制激活值 (虽然划分了参数，但是每个设备仍然需要完整的输入)，导致在模型并行的 GPU 之间产生冗余的激活副本
>  ZeRO 通过划分激活值来消除这种冗余，并且仅在激活值即将用于计算时，按层依次将其复原

>  流程示例：正向传播后

```
Layer: Linear(W)
Input: x (1024 dim)
Output: y = Wx

GPU0: W0 (前半权重), 接收完整 x → 计算 y0
GPU1: W1 (后半权重), 接收完整 x → 计算 y1
```

> 传统做法：两个 GPU 都存完整 `x` → 冗余
> ZeRO 的 $P_a$ ​ ：
> 1. 正向传播完成后，`x` 不再需要完整副本
> 2. ZeRO 把 `x` **拆成两半**：
    - GPU0 存 `x[0:512]`
    - GPU1 存 `x[512:1024]`
> 3. 显存节省一半

More specifically, once the forward propagation for a layer of a model is computed, the input activations are partitioned across all the model parallel process, until it is needed again during the backpropagation. At this point, ZeRO uses an all-gather operation to re-materialize a replicated copy of the activations. We refer to this optimization as  $P_{a}$  . It works in conjunction with activation checkpointing [7], storing partitioned activation checkpoints only instead of replicated copies. 
>  更具体地说，一旦某一层的前向传播计算完成，输入激活就会被划分到所有模型并行进程中，直到在反向传播过程中再次需要，ZeRO 使用 all-gather 重新生成一份激活值的复制版本
>  我们将这一优化称为 $P_a$，它与激活检查点配合使用，存储划分后的激活检查点

Furthermore, in the case of very large models and very limited device memory, these partitioned activation checkpoints can also be offloaded to the CPU reducing the activation memory overhead to nearly zero at an additional communication cost, which we will discuss in 7. We refer to this as $P_{a+cpu}$.
>  此外，在模型非常大且设备内存非常有限的情况下，这些划分后的激活检查点也可以卸载到 CPU 上，从而将激活内存开销降低到零，但这会增加额外的通信成本，这一方法称为 $P_{a+cpu}$

**Memory Saving** With partitioned activation checkpointing, ZeRO reduces the activation footprint by a factor proportional to the MP degree. Consider training a 100B model shown in Table 4 with a batch size of 32, sequence length of 1024 and a MP degree of 16. If we checkpoint a single activation for each transformer layer, it would require about 33 GB of memory per GPU just to store the activation checkpoints. But with  $P_{a}$  in ZeRO, it can be reduced to about 2 GB per GPU. Furthermore, this 2GB can be offloaded to the CPU reducing the memory footprint for activations to nearly zero.
>  对激活检查点进行划分后，ZeRO 将激活内存占用按照 MP degree 成比例降低
>  假设训练一个 100B 的模型，batch size 为 32，序列长度为 1024，MP degree 为 16，如果我们为每个 transformer 层存储激活检查点，则每个 GPU 需要保存 33GB 的激活检查点
>  而使用 $P_a$ 后，这个数值就可以降到 2GB (MP degree = 16)，此外，这 2GB 还可以卸载到 CPU 上，使得激活占用为零

## 6.2 $C_B$ : Constant Size Buffers
ZeRO carefully selects the sizes of the temporal-data buffers to balance memory and compute efficiency. During training, the computational efficiency of some operations can be highly dependent on the input size, with larger inputs achieving higher efficiency. For example, a large all-reduce operation achieves much higher bandwidth than a smaller one. 
>  ZeRO 精心选择临时数据缓冲区的大小，以在内存效率和计算效率之间取得平衡
>  在训练过程中，某些运算的计算效率可能高度依赖于输入大小，更大的输入可以实现更大的效率
>  例如，更大的 all-reduce 操作的带宽比更小的 all-reduce 操作更好

Hence, to get better efficiency, high performance libraries such as NVIDIA Apex or Megatron fuses all the parameters into a single buffer before applying these operations. However, the memory overhead of the fused buffers is proportional to the model size, and can become inhibiting. For example, for a 3B parameter model, a 32-bit fused buffer will require 12 GB of memory. To address this issue, we simply use a performance-efficient constant-size fused buffer when the model becomes too large. By doing so, the buffer size does not depend on the model size, and by keeping the buffer size large enough, we can still achieve good efficiency.
>  因此，为了获得更好的效率，像 NVIDIA Apex 或 Megatron 这样的高性能库会在应用这些操作之前将所有的参数融合到单个缓冲区
>  然而，融合缓冲区的内存开销和模型大小成正比，例如对于一个 3B 的模型，32bit 的融合缓冲区将需要 12GB 的内存
>  为了解决这个问题，当模型变得太大时，我们简单地使用一个性能高效的固定大小缓冲区，此时缓冲区大小不再依赖于模型大小，并且通过保持缓冲区足够大，我们仍然可以实现良好的效率

## 6.3  $M_D$  : Memory Defragmentation
Memory fragmentation in model training occurs as a result of activation checkpointing and gradient computation. During the forward propagation with activation checkpointing, only selected activations are stored for back propagation while most activations are discarded as they can be recomputed again during the back propagation. This creates an interleaving of short lived memory (discarded activations) and long lived memory (checkpointed activation), leading to memory fragmentation. Similarly, during the backward propagation, the parameter gradients are long lived, while activation gradients and any other buffers required to compute the parameter gradients are short lived. Once again, this interleaving of short term and long term memory causes memory fragmentation.
>  模型训练中的内存碎片化是激活检查点和梯度计算引起的
>  在使用激活检查点进行前向传播时，只有部分激活值会被保存，大多数激活值会被丢弃，这会导致短期内存 (被丢弃的激活) 和长期内存 (保存的激活检查点) 的叫交错，从而产生内存碎片
>  类似地，在反向传播时，参数梯度会长期保存，而激活梯度以及其他用于计算参数梯度的缓冲区则是短暂的
>  这种短期和长期内存的交错再次导致了内存碎片化

Limited memory fragmentation is generally not an issue, when there is plenty of memory to spare, but for large model training running with limited memory, memory fragmentation leads to two issues, i) OOM due to lack of contiguous memory even when there is enough available memory, ii) poor efficiency as a result of the memory allocator spending significant time to search for a contiguous piece of memory to satisfy a memory request.
>  对于在有限内存下的大模型训练，内存碎片化会导致:
>  1. 即时可用空间足够，由于缺乏连续内存也会出现 OOM
>  2. 由于内存分配器需要花费大量时间搜索满足内存请求的连续内存块，效率会很低下

ZeRO does memory defragmentation on-the-fly by pre-allocating contiguous memory chunks for activation checkpoints and gradients, and opening them over to the pre-allocated memory as they are produced.  $\mathrm{M}_D$  not only enables  $ZeRO$  to train larger models with larger batch sizes, but also improves efficiency when training with limited memory.
>  ZeRO 通过预先分配连续的内存块来存储激活检查点和梯度，在激活检查点和梯度生成时就将它们放到预先分配的内存中，来进行实时的去碎片化
>  $M_D$ 提高了内存使用效率，使得 ZeRO 可以训练更大的模型

# 7 Communication Analysis of ZeRO-DP
As  $ZeRO$  boosts model size by removing memory redundancy, it is only natural to ask if we are trading communication volume for memory efficiency. In other words, what is the communication volume of  $ZeRO$ - powered DP approach compared to a baseline DP approach? The answer is in two parts: i)  $ZeRO$ - DP incurs no additional communication using  $P_{os}$  and  $P_g$ , while enabling up to 8x memory reduction, ii)  $ZeRO$ - DP incurs a maximum of 1.5x communication when using  $P_p$  in addition to  $P_{os}$  and  $P_g$ , while further reducing the memory footprint by  $N_d$  times. We present the analysis in this section. We begin by first presenting a brief overview of the communication volume for standard DP.
>  ZeRO 移除了内存冗余，但我们也要考虑是否增加了通信量
>  换句话说，我们要考虑使用了 ZeRO 优化的 DP 和普通 DP 的通信量比较，答案分为两部分:
>  1. ZeRO-DP 在使用 $P_{os}, P_g$ 时没有额外通信量，且带来了 8x 的内存减少
>  2. ZeRO-DP 在进一步使用 $P_p$ 时将通信量变为了原来的 1.5x，但带来了 $N_d$ 的内存减少

## 7.1 Data Parallel Communication Volume
During data parallel training, gradients across all data parallel processes are averaged at the end of the backward propagation before computing the updates for the next step. The averaging is performed using an all-reduce communication collective. For a large model size, the all-reduce communication is entirely communication bandwidth bound, and therefore, we limit our analysis to the total communication volume send to and from each data parallel process.
>  在数据并行训练中，所有数据并行进程的梯度都会在反向传播的结束阶段，参数更新的开始阶段之前进行平均
>  对于大模型，要传输的梯度数据量很大，all-reduce 通信的时间完全由网络带宽决定 (而不是计算延迟或启动开销)，因此，在考虑通讯效率时，我们仅分析各个数据并行进程发送和接收的数据量 (即通信量)，因为这是决定通信时间的主要因素

State-of-art implementation of all-reduce uses a two-step approach, where the first step is a reduce-scatter operation, which reduces different part of the data on different process. The next step is an all-gather operation where each process gathers the reduced data on all the process. The result of these two steps is an all-reduce. Both reduce-scatter and all-gather are implemented using a pipelined approach, that results in a total data movement of  $\Psi$  elements (for a data with  $\Psi$  elements) for each. Therefore, the standard DP incurs  $2\Psi$  data movement during each training step.
>  all-reduce 的 SOTA 实现使用两步方法:
>  第一步为 reduce-scatter 操作，对不同进程上的不同部分的数据进行规约，第二步为 all-gather 操作，每个进程从其他进程收集规约后的数据，两步的结果就是 all-reduce
>  reduce-scatter, all-gather 都使用流水线方法实现，在流水线优化后，对于每个操作，总的数据移动量就是 $\Psi$ 个元素，因此标准 DP 在每一步训练步 (的 all-reduce) 需要 $2\Psi$ 的数据移动

> 第一步：Reduce-Scatter
> 假设有 P 个进程 (GPU)，每个进程持有一个完整的梯度向量 (大小为 $\Psi$), reduce-scatter 将这个向量均匀分割成 P 个块，然后每个进程负责对某一个块执行跨所有进程的归约
> 例如，进程 0 负责第 0 块，它从所有进程中收集第 0 块的数据，求和，得到该块的规约值
> 最终，每个进程只拥有归约后的一个块 (即某一段梯度的总和)，而不是完整的归约结果
> 
> 第二步：All-Gather
> 每个进程已经拥有一个归约后的数据块之后，执行 all-gather，每个进程将自己拥有的那个块发送给所有其他进程，同时接收来自其他进程的块。
> 最终，每个进程都能收集到所有 P 个块，从而拼接出完整的归约后向量

> 为了提高效率，reduce-scatter 和 all-gather 都采用流水线实现，尤其是在使用环形拓扑时，在流水线设计中，数据被进一步细分成小片段，逐步在进程间传递和处理，从而重叠通信与计算，减少总延迟
>  我们仔细考虑一下通信量:
>  第一步 reduce-scatter 中，数据划分为了 P 块，每一块为 $\frac \Psi P$，宏观上看，每一块 GPU 都需要接收 $\frac {P-1}P \Psi \approx \Psi$ 的数据，并发送 $\frac {P-1}P \Psi \approx \Psi$ 的数据，在 ring 实现和流水线重叠下，各个 GPU 的发送和接收会重叠，因此可以认为所有 GPU 的总通信量就是 $\Psi$
>  第二步也是类似的

## 7.2 ZeRO-DP Communication Volume
### 7.2.1 Communication Volume with  $P_{os + g}$
With gradient partitioning, each process only stores the portion of the gradients, that is required to update its corresponding parameter partition. As such, instead of an all-reduce, ZeRO only requires a scatter-reduce operation on the gradients, incurring communication volume of  $\Psi$ . After each process updates the partition of the parameters that it is responsible for, an all-gather is performed to collect all the updated parameters from all the data parallel process. This also incurs a communication volume of  $\Psi$ . So the total communication volume per training step is  $\Psi + \Psi = 2\Psi$ , exactly the same as the baseline DP.
>  采用优化器状态划分 + 梯度划分后，每个进程仅存储一个 partition 的梯度
>  故 ZeRO 需要一次 scatter-reduce 对各个梯度 partition 进行规约，通信量为 $\Psi$
>  在各个进程更新了自己的参数 partition 之后，ZeRO 需要一次 all-gather 来收集更新后的参数 partitions，通信量为 $\Psi$
>  因此总的通信量为 $2\Psi$，和基础 DP 一样

### 7.2.2 Communication Volume with  $P_{os + g + p}$
After parameter partitioning, each data parallel process only stores the parameters that it updates. Therefore, during the forward propagation it needs to receives the parameters for all the other partitions. 
>  采用参数划分后，每个数据并行进程仅存储它需要更新的参数，这要求它在前向传播过程中获取其他 partitions 的参数

However, this can be pipelined to avoid the memory overhead. Before computing the forward propagation on the part of the model corresponding to a particular partition, the data parallel process responsible for that partition can broadcast the weights to all the data parallel processes. Once the forward propagation for that partition is done, the parameters can be discarded. The total communication volume is thus  $\begin{array}{r}\frac{\Psi\times N_d}{N_d} = \Psi \end{array}$  .
>  这个收集过程可以进行流水线化，以避免内存开销 (将原本一次性 all-gather 所有参数的操作，拆成多个小广播，穿插在前向传播中执行)
>  在计算某个特定 partition 参数的前向传播过程之前，负责该参数 partition 的设备/并行进程可以将其参数广播给所有数据并行进程
>  一旦该 partition 的前向传播计算完毕，其他进程就可以丢弃这个 partition 的参数，这样总的通信量就是 $\frac {\Psi \times N_d} {N_d} = \Psi$ (每个 GPU 都会广播一次它的参数 partiton，因此总的参数发送量就是 $\Psi$)

In other words, we reschedule the parameter all-gather by spreading it across the entire forward propagation, and discarding the parameters once they have been used. Note however that this all-gather needs to happen once again for the backward propagation in the reverse order.
>  换句话说，我们重新调度了参数 all-gather，将整个 all-gather 过程拆成了在整个前向传播过程中的多次对参数 partition 的广播过程，并且在参数 partition 被使用后就丢弃它
>  注意在反向传播过程中这个过程也要执行一次

The total communication volume is therefore the sum of the communication volumes incurred by these all-gathers in addition to the communication volume incurred by the reduce-scatter of the gradients. The total volume is therefore  $3\Psi$  which is  $1.5x$  compared to the baseline.
>  总的通信量就是由 all-gather 引起的通信量 (前向和反向的按需广播都等同于一次 all-gather) 再加上梯度的 reduce-scatter (反向计算结束后，所有的数据并行进程对各自计算的梯度进行规约，然后保留自己更新的那部分参数的梯度)
>  这样总的通信量就是 $3\Psi$，是普通 DP 的 1.5x

Both gradient and parameter partitioning leverage the insight that - not all states of gradients and parameters are needed all the time - to optimize memory by communicating the states judiciously.
 >  梯度和参数划分都利用了一个思想: 不是所有的梯度和参数状态在所有时刻都需要，以此通过通讯传播状态来优化内存

# 8 Communication Analysis of ZeRO-R
We compare the communication volume of partitioned activation checkpointing (  $P_{a}$  ) in ZeRO-R with baseline MP, and show that  $P_{a}$  incurs a communication volume increase that is in general less than one tenth of the baseline MP. Furthermore, we analyze the communication overhead of  $P_{a}$  in relation to DP communication volume to identify scenarios when  $P_{a}$  improves efficiency by allowing for a larger batch size and reducing DP communication. We leverage such analysis to decide if and when to apply  $P_{a}$  as well as  $P_{a + cpu}$
>  我们将 ZeRO-R 中的检查点划分 ($P_a$) 和 baseline MP 比较，以展示 $P_a$ 导致的通信量增长通常小于 baseline MP 的 1/10
>  此外，我们分析 $P_a$ 的通信开销和 DP 通信量的关系，以确定哪些情况下 $P_a$ 允许通过更大的 batch size 来减少 DP 通讯

Communication volume trade-off of partitioning activation checkpoints depends on the model size, checkpointing strategy and the MP strategy. To share concrete insights, we perform the analysis in the context of transformer based models implemented using SOTA MP approach, Megatron-LM.
>  激活检查点划分的通信量权衡取决于模型大小、检查点策略和 MP 策略
>  我们以 Transformer 模型为例，在 SOTA 的 MP 方法 Megatron-LM 下进行分析

In Megatron-LM with activation checkpointing, each transformer block performs two allreduce operations of size batch  $\times$  seq_length  $\times$  hidden_dim in the forward propagation, two all-reduce for forward re-computation and two more in the backward propagation. The total communication per block is  $12\times$  seq_length  $\times$  hidden_dim since communication volume of an all-reduce is  $2\times$  message_size.
>  在使用激活检查点的 Megatron-LM 中，每个 transformer block 在前向传播中执行两次大小为 `batch x seq_length x hidden_dim` 的 all-reduce 操作，在反向传播过程中，会有两次 all-reduce 用于前向重新计算，两次 all-reduce 用于反向传播
>  由于一次 all-reduce 的通信量为 `2 x message_size` (对于大小为 `message_size` 的数据，all-reduce 的通信量为 `2 x message_size`，all-reduce 中，数据会先被规约到各个节点，再广播回去)，故 transformer block 的总通信量为 ` 12 x batch_size x seq_length x hidden_dim ` (6 次 all-reduce 的通信量)

When ZeRO-R partitions activation checkpoints, it requires an additional all-gather operation before the forward recomputation of the back-propagation on each activation checkpoint. In general, we checkpoint the input activation for each transformer block, requiring one all-gather per transformer block. The communication overhead  $P_{a}$  is therefore seq_length x hidden_dim, since the communication volume of an all-gather is message size. Therefore, the total communication overhead of  $P_{a}$  is less than  $10\%$  of the original communication volume for model parallelism.
>  ZeRO-R 划分了激活检查点之后，在反向传播过程的前向重计算过程之前，它需要一次额外的 all-gather 操作
>  我们通常为每个 transformer 块保存它的输入激活，因此每个 transformer 块需要一次 all-gather 操作
>  因此 $P_a$ 的通信开销就是 `batch_size x seq_length x hidden_dim` ，因为 all-gather 的通信量就等于消息大小，因此 $P_a$ 的总通信开销少于原来的 MP 通信开销的 1/10

When MP is used in conjunction with DP,  $P_{a}$  can be used to reduce the data-parallel communication volume by an order of magnitude at the expense of a  $10\%$  increase in model-parallel communication volume, and significantly boost efficiency when data-parallel communication is a performance bottleneck. Notice that  $P_{a}$  reduces the activation memory consumption by the MP degree allowing for a proportional increase in batch size. 
>  MP 和 DP 一同使用的情况下，$P_a$ 可以在 MP 通信量增加 10% 的代价下，将 DP 通信量减少一个数量级，并在数据并行通信称为瓶颈时显著提升效率
>  这么说的理由是，$P_a$ 将激活内存消耗降低为原来的 $\frac 1 {\text{MP degree}}$，从而允许 batch size 按比例增加 (变为原来的 MP degree 倍)

For large models, MP can be as large as 16 ( `#GPUs` on a DGX-2 node), allowing for up to 16x increase in the batch size. The communication volume of a data-parallel training is inversely proportional to the batch size. Therefore, an order of magnitude increase in batch size due to  $P_{a}$  could result in an order-of-magnitude decrease in data-parallel communication volume.
>  对于较大的模型，MP degree 可以是 16 (DGX-2 node 的 GPU 数量)，故使用 $P_a$ 可以带来 batch size 16x 的提升
>  而 DP 的通信量和 batch size 成反比，因此 $P_a$ 带来的 batch size 的一个数量级的增加可以带来 DP 通信量一个数量级的减少

>  DP 中每次梯度 all-reduc 的通信量都等于模型参数量，与 batch size 无关，但是在数据集大小固定的情况下，batch size 的增大意味着每个 epoch 迭代数量的减少，因为 epoch 的总通信量 = 总迭代数 x 模型参数量，因此迭代数量减少意味着通信量减少

Finally if  $P_{a + cpu}$  is applied, partitioned activation checkpoints are offloaded to CPU, reducing the activation memory requirement to nearly zero at the expense of 2x added data movement to and from CPU memory compared to  $P_{a}$ . In extreme cases where DP communication volume is the major bottleneck due to a small batch size even with  $P_{a}$ ,  $P_{a + cpu}$  can improve efficiency by increasing the batch size as long as the CPU data transfer overhead is less than the DP communication volume overhead, which is generally true for small batch sizes.
>  如果使用 $P_{a + cpu}$，划分的激活检查点会被卸载到 CPU，将激活内存需求减少到几乎为零，但代价是需要两次额外的 CPU 和 GPU 之间的数据移动
>  在极端情况下，即时使用了 $P_a$ 也无法大幅提高 batch size 使得 DP 通讯量称为瓶颈，可以通过 $P_{a+cpu}$ 来提高 batch size，只要 CPU 数据传输开销小于 DP 通信量开销，这在小 batch size 通常是成立的

Given model and hardware characteristics, we leverage the above analysis to decide if and when to apply  $P_{a}$  and  $P_{a + cpu}$ .

# 9 Step Towards 1 Trillion Parameters
The largest published models today are in the range of 10 billion parameters, which are already challenging to train. Getting to a trillion parameters, 3-orders of magnitude larger, will inevitably happen, but the road will be full of hurdles, surprises and innovations. While we do not claim knowing or addressing all of them,  $ZeRO$  addresses one of the most fundamental challenges from a system perspective: the ability to fit a model of this scale on current hardware while allowing it to train with good system scalability.
>  目前发表的最大模型为 10B 参数
>  ZeRO 可以将 1T 参数的模型在当前的硬件上训练

**A Leap from State-of-Art** The largest model that the state-of-art framework, Megatron, can train with acceptable throughput is a 16 - 20B parameter model in a DGX-2 system. Scaling further by having model parallelism across multiple DGX nodes results in significant efficiency drop due to limited internode bandwidth.
>  Megatron 可以在一个 DGX-2 系统训练 16 - 20B 的参数，进一步 scale 到多个 DGX node 将由于节点间带宽导致严重的效率下降

<table><tr><td rowspan="2">MP</td><td rowspan="2">GPUs</td><td colspan="4">Max Theoretical Model Size</td><td colspan="2">Measured Model Size</td></tr><tr><td>Baseline</td><td>Pos</td><td>Pos+g</td><td>Pos+g+p</td><td>Baseline</td><td>ZeRO-DP (Pos)</td></tr><tr><td>1</td><td>64</td><td>2B</td><td>7.6B</td><td>14.4B</td><td>128B</td><td>1.3B</td><td>6.2B</td></tr><tr><td>2</td><td>128</td><td>4B</td><td>15.2B</td><td>28.8B</td><td>256B</td><td>2.5B</td><td>12.5B</td></tr><tr><td>4</td><td>256</td><td>8B</td><td>30.4B</td><td>57.6B</td><td>0.5T</td><td>5B</td><td>25B</td></tr><tr><td>8</td><td>512</td><td>16B</td><td>60.8B</td><td>115.2B</td><td>1T</td><td>10B</td><td>50B</td></tr><tr><td>16</td><td>1024</td><td>32B</td><td>121.6B</td><td>230.4B</td><td>2T</td><td>20B</td><td>100B</td></tr></table>

Table 2: Maximum model size through memory analysis (left) and the measured model size when running with  $ZeRO -OS$  (right). The measured model size with  $P_{os}$  matches the theoretical maximum, demonstrating that our memory analysis provides realistic upper bounds on model sizes.

$ZeRO$  vastly increase the efficiently-runnable model size. It enables the current generation of hardware to run significantly larger models without requiring fine-grained model parallelism to go across the node boundaries. As demonstrated in Table 1,  $ZeRO$ , with all optimizations turned on  $(P_{os + g + p})$ , could fit more than 1 Trillion parameters on 1024 GPUs using DP only. Alternatively, when combined with model parallelism (as shown in Table 2),  $ZeRO$  could fit more than 1 Trillion parameters on 1024 GPUs with 16-way model parallelism (within each DGX2 node) and 64-way data parallelism across nodes. Running a model with a trillion parameters efficiently is no longer impossible!
>  ZeRO 显著提高了可以高效运行的模型规模
>  如 Table 1 中，将所有优化启动的 ZeRO 可以仅使用 DP 就将 1T 模型放到 1024 GPU，此外，和 MP 结合后，ZeRO 可以将超过 1T 的模型放入 1024 GPU (在每个 DGX-2 节点中使用 16-way MP，在节点之间使用 64-way DP)

**Compute Power Gap** Training a trillion parameter model end-to-end within an acceptable time range, however, could still require significant amount of compute power, which is lacking in today's AI clusters.

To understand the resource requirement, we present a brief comparison with Bert-Large. Bert-Large can be trained in 67 minutes on a 1024 GPU DGX-2H cluster [26]. A 1 Trillion Parameter model can easily contain 3000x (1 trillion / 330 million) more computation than a Bert-Large model for a data sample. Even if we assume the same sequence length and the total number of samples required to train the model, training a 1T model would take 140 days, assuming the same hardware and similar computational efficiency. In practice, both data samples and sequence length are likely to increase with the increased model size requiring over a year to train. It would require an exa-flop system to train a 1T parameter model in a reasonable time. But when such compute capacity becomes available, we hope ZeRO will provide the system technology to run the 1T models efficiently.


![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/25877288-f38c-4883-a49c-40ce8af650c2/70553319a65ba0bfa1fb9efaf7c1b625c4d511d1bde6a5d54fd043884b44fae8.jpg)  

Figure 5: SOTA Turing-NLG enabled by ZeRO.


# 10 Implementation and Evaluation
We focus our implementation on supporting efficient training of models with  $\sim 100$ B parameters, which are an order-of-magnitude larger than the largest published models today (e.g., T5-11B [4]) while trainable within a reasonable time frame on current hardware (e.g., with 1K V100 GPUs). 

We implement and evaluate a subset of optimizations in  $ZeRO -P_{os + g}$  in ZeRO-DP plus ZeRO-R — that allows us to achieve this goal. We will refer to this implementation as ZeRO-100B. Our results show that ZeRO-100B can efficiently train models with up to 170B parameters, 8x bigger than SOTA, up to 10x faster and with improved usability. ZeRO-100B powers Turing-NLG, the largest published model in the world with new SOTA accuracy.

## 10.1 Implementation and Methodology
**Implementation** We implemented ZeRO-100B in PyTorch including the full set of optimizations in  $P_{os + g}$  and ZeRO-R. Its interface is compatible with any model implemented as an torch.nn.module. Users can simply wrap their models using this interface and leverage ZeRO-powered DP as they use classic DP. Users do not need to modify their model. ZeRO-powered DP can be combined with any form of MP including Megatron-LM.
>  我们在 PyTorch 中实现了 ZeRO-100B，其接口和任意用 `torch.nn.module` 实现的模型都兼容
>  ZeRO-powered DP 可以和任意形式的 MP 结合

**Hardware** We conducted our experiments on a cluster of 400 V100 GPUs (25 DGX-2 nodes) with 800 Gbps internode communication bandwidth.
>  实验在 400 个 V100 (25 DGX-2 节点) 上实现

**Baseline** For experiments without MP, we use torch's distributed data parallel (DDP) as baseline. For experiments with MP, we use Megatron-LM because it is, to our knowledge, the state-of-art. We use the open-source version of Megatron-LM from NVIDIA with a date of September 2019. The most recent Megatron-LM results report the ability to scale up to 16B parameter models using 32 DGX-2 nodes (total of 512 32GB V100 GPUs) [3].
>  没有 MP 的 baseline 使用 torch distributed data parallel (DDP)
>  有 MP 的 baseline 使用 Megatron-LM，最近的 Megatron-LM 报告的结果是在 32 DGX-2 上训练了 16B 模型

**ZeRO** Experiments without MP, use the ZeRO-powered DP implementation in ZeRO-100B. Experiments with MP, combine ZeRO-powered DP with MP of Megatron-LM.

**Model Configurations** The models presented in this section are GPT-2 [2] like transformer based models. We vary the hidden dimension and the number of layers to obtain models with different number of parameters. Table 4 shows the configuration parameters used in our experiments with additional details in AE Appendix.

## 10.2 Speed and Model Size
ZeRO-100B efficiently run models with up to 170B parameters on 400 GPUs, more than 8x bigger than Megatron-LM. Figure 2 shows throughput per GPU for varying model sizes using ZeRO-100B with MP versus using Megatron MP alone. ZeRO-100B achieves a sustained throughput of 15 PetaFlops (over  $30\%$  of the peak) on average for models with 8B to 100B parameters. 
>  ZeRO-100B 在 400 GPU 上运行了 170B 的模型，比 Megatron-LM 大 8x

In comparison, the baseline MP performance degrades quickly with the increase in model size: MP incurs high communication volume between GPUs, and going beyond a single node to fit larger models causes a communication bandwidth drop from 300GB/sec per link (NVSwitch) to 12.5 GB/sec per link (Infiniband EDR), resulting in a significant performance drop. ZeRO-100B achieves up to 10x speedup over baseline, significantly outperforming on large models.
>  MP 则随着模型变大，效果变差: GPU 之间的通信成为瓶颈

For ZeRO-100B, the slight reduction in performance beyond 100B is due to lack of enough memory to run larger batch sizes. We expect the performance to improve as we increase the number of GPUs due to super-linear speedup of ZeRO-100B as we discuss next.

## 10.3 Super-Linear Scalability

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/25877288-f38c-4883-a49c-40ce8af650c2/2b96d718ca969d9f5b198c26adeb783df576dffa793b06b13686d89d44e21be1.jpg)  

Figure 3: Superlinear scalability and per GPU training throughput of a 60B parameter model using ZeRO-100B.

ZeRO-100B demonstrates super-linear scalability for very large model sizes. Figure 3 shows scalability results for a 60B parameter model going from 64 to 400 GPUs and we expect this trend to continue further for more GPUs. 

$P_{os + g}$  reduces per GPU memory consumption of ZeRO-100B with increase in DP degree, allowing ZeRO-100B to fit larger batch sizes per  $\mathrm{GPU^5}$ , which in turn improves throughput as a result of increasing arithmetic intensity.
>  $P_{os + g}$ 通过提高 DP degree 减少了每个 GPU 上的内存消耗，使得 ZeRO-100B 可以在每个 GPU 放入更大的 batch，进而提高了算术密度，因此提高了吞吐

## 10.4 Democratizing Large Model Training

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/25877288-f38c-4883-a49c-40ce8af650c2/c02c1c58cd2aba9b9eccf02eb6b18cedf801c540f41e80ccd8125c5e5c1c2fa3.jpg)  

Figure 4: Max model throughput with ZeRO-DP.

Using MP and PP is challenging for many data scientists, which is a well-known hurdle to train large models. ZeRO does not require any changes to the model itself and it can be used as simple as baseline DP while delivering significantly boosted model size and speed. 
>  ZeRO 不需要对模型对任何修改，只需要和传统 DP 一样使用，但是却能显著提高效率

Fig. 4 shows that ZeRO-100B can train models with up to 13B parameters without MP on 128 GPUs, achieving throughput over 40 TFlops per GPU on average. In comparison, without ZeRO, the largest trainable model with DP alone has 1.4B parameters with throughput less than 20 TFlops per GPU. Furthermore, in the absence of the communication overhead from MP, these models can be trained with lower-end compute nodes without very fast intra-node interconnect such as NVLINK or NVSwitch, which is required to achieve good efficiency with MP.
>  ZeRO-100B 不需要 MP 就可以在 128 GPU 训练 13B 模型
>  没有 ZeRO，使用 DP 能够训练的最大模型仅 1.4B
>  此外，因为没有 MP 的通信开销，模型可以在更低级的计算节点训练，不需要快速的节点互联

## 10.5 Memory and Performance Analysis

Table 3: ZeRO configurations  

<center><table><tr><td></td><td>ZeRO-DP</td><td>ZeRO-R</td></tr><tr><td>1</td><td>P_os</td><td>C_B+M_D</td></tr><tr><td>2</td><td>P_os</td><td>C_B+M_D+P_a</td></tr><tr><td>3</td><td>P_os+g</td><td>C_B+M_D</td></tr><tr><td>4</td><td>P_os+g</td><td>C_B+M_D+P_a</td></tr><tr><td>5</td><td>P_os+g</td><td>C_B+M_D+P_a+cpu</td></tr></table></center>

We look into the benefits and impact of different optimizations on maximum model size, memory consumption and performance. These optimizations are referred to as Config 1 to 5 (C1-C5) in Table. 3.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/25877288-f38c-4883-a49c-40ce8af650c2/277bec9be6e4fb18afbbb2f0f8d92cd3bd75d673f932868abeb268fc9eceee47.jpg)  

Figure 6: Max model size

**Maximum Model Size** Figure 6 shows the largest trainable model by enabling different ZeRO optimizations for a fixed batch size and MP of 16. The model size increase from 40B to 60B when trained with C1 vs C2 due to a 16x (MP degree) reduction in activation memory from using  $P_{a}$ , while the jump to 140B using C4 is from enabling  $P_{os + g}$  which halves the memory requirement by the model states compared to  $P_{os}$  in C2. The increase to 150B using C5 is solely due to further reduction in activation memory from offloading the partitioned activation checkpoints to the CPU memory.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/25877288-f38c-4883-a49c-40ce8af650c2/718dac3b23e57975fdb61290fe77e9c31dd9e18f5e312278c3ec2f279c744114.jpg)  

Figure 7: Max cache allocated.

**Max Cached Memory** Figure 7 shows the maximum memory cached by PyTorch during each training iteration for a 40B and a 100B parameter model. The decrease of the cached memory size is as expected from C1 to C2. The difference in memory consumption between C2 and C3 depends on the size of the model states in comparison to the activation memory, and can increase when activation memory is larger, or decrease when the model states are larger. It is noteworthy that the cached memory does not decrease from C4 to C5 for 40B but it does for 100B. This is simply because the activation memory for 100B is much larger for the decrease to be noticeable. This makes  $P_{a + cpu}$  a valuable tool to fit a larger batch size when we get to
very large models. In Figure 8,  $P_{a + cpu}$  is needed for 170B model to execute without running out of memory.

Table 4: Configurations for different model sizes, number of layers, and hidden dimensions (HD) across Figures 2,3,4.  

<table><tr><td colspan="3">Figure 2</td><td colspan="3">Figures 3, 4</td></tr><tr><td></td><td>Layers</td><td>HD</td><td></td><td>Layers</td><td>HD</td></tr><tr><td>1.5B</td><td>48</td><td>1600</td><td>1.16B-2.5B</td><td>24,34,54</td><td>1920</td></tr><tr><td>8B</td><td>72</td><td>3072</td><td>4B</td><td>64</td><td>2304</td></tr><tr><td>0B-60B</td><td>88,132</td><td>4096</td><td>0B-8B</td><td>52,72</td><td>3072</td></tr><tr><td>80B-170B</td><td>100,125,150</td><td>8192</td><td>10B-13B</td><td>50,54,58,62</td><td>4096</td></tr><tr><td>140B-170B</td><td>175,212</td><td>8192</td><td>60B</td><td>75</td><td>8192</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-19/25877288-f38c-4883-a49c-40ce8af650c2/b9572685109e888fa63afb13f67a8ee19c335c74bd92a7cebc59d1984adb7953.jpg)  

Figure 8: Throughput per GPU.

**Max Achievable Performance** Figure 8 shows the best achievable performance for different set of optimizations. Notice that performance improvement corresponds to decrease in memory consumption between the optimizations. As mentioned earlier, lower memory consumption allows for larger batch size which improves performance. The only caveat is the performance drop between C4 and C5 for 60B parameter model. Despite lower memory consumption, C5 incurs activation movement to and from the CPU, this will result in worse performance in most cases, except for a few where the model is so large that the model simply cannot run without C5 or the batch size that can run without C5 is very small (such as model with 170B parameters in Figure 8). During training,  $P_{a + cpu}$  is turned on only when it is beneficial.

## 10.6 Turing-NLG, the SOTA language model with 17B parameters
As of May 12th, 2020, Turing-NLG is the largest model in the world with over 17B parameters. It achieved the new SOTA for language models with Webtext-103 perplexity of 10.21. Turing-NLG was trained end-to-end using ZeRO-100B and Fig. 5 shows the validation perplexity over 300K iterations compared to previous SOTA, Megatron-LM 8.3B parameter model. ZeRO-100B achieves a sustained throughput of 41.4 TFlops/GPU for this model.

# 11 Concluding Remarks
From a HPC and system perspective, we believe that  $ZeRO$  represents a revolutionary transformation in the large model training landscape. While our implementation, ZeRO-100B, enables 8x increase in model sizes, over 10x in throughput improvement, achieves super-linear speedups on modern GPU clusters, and trains the largest model in the world, it is still just a tip of the iceberg. ZeRO in its entirety has the potential to increase the model size by yet another order of magnitude, enabling the training of trillion parameter models of the future.
>  从高性能计算和系统角度来看，我们相信 ZeRO 代表了大模型训练领域的一次革命性变革
>  我们的实现: ZeRO-100B 可以使模型规模增大 8x，吞吐量提升超过 10x，在现代 GPU 集群上实现了超线性加速，并训练出了目前世界上最大的模型
>  ZeRO 还可以将模型规模再提升一个数量级，从而支持未来万亿参数的训练

Perhaps, what we feel most optimistic about  $ZeRO$  is that it imposes no hurdles on the data scientists. Unlike existing approaches such as MP and PP, no model refactoring is necessary, and it is as easy to use as standard DP, making  $ZeRO$  a prime candidate for future investigations on large model training. Through open sourcing and community feedback, we plan to make  $ZeRO$  fully accessible to the DL community to catalyze the evolution and democratization of large model training at scale.
>  ZeRO 无需对模型进行重构，使用起来和标准的 DP 一样简单
