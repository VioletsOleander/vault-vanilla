# Abstract
Reinforcement Learning from Human Feedback (RLHF) is widely used in Large Language Model (LLM) alignment. Traditional RL can be modeled as a dataflow, where each node represents computation of a neural network (NN) and each edge denotes data dependencies between the NNs. RLHF complicates the dataflow by expanding each node into a distributed LLM training or generation program, and each edge into a many-to-many multicast. 
>  传统的 RL 可以建模为一个数据流，其中每个节点表示 NN 的计算，每个边表示 NN 与 NN 之间的数据依赖
>  RLHF 将每个节点拓展为一个分布式 LLM 训练或生成程序，并将每条边拓展到多对多的组播，复杂化了数据流

>  传统 RL 中，策略网络用于决定动作，价值网络用于评估状态价值，策略网络的输出需要交给价值网络以评估价值
>  RLHF 中，策略本身是 LLM，不是传统 RL 中常用的小型网络，LLM 策略的决策是分布式的推理，LLM 策略的更新是分布式的训练，故此时数据流图中的简单 NN 节点变成了内部也包含了复杂数据流和并行计算的 “子图”

Traditional RL frameworks execute the dataflow using a single controller to instruct both intra-node computation and inter-node communication, which can be inefficient in RLHF due to large control dispatch overhead for distributed intra-node computation. Existing RLHF systems adopt a multi-controller paradigm, which can be inflexible due to nesting distributed computation and data communication. 
>  传统 RL 框架使用单一控制器来指导节点内部和节点间的计算和通信，在 RLHF 场景下，这就变得低效，因为 RLHF 中存在节点内分布式计算的大量控制分发开销 (也就是单一的控制器成为瓶颈)
>  现有的 RLHF 系统采用多控制器范式，但由于嵌套分布式计算和数据通信，可能会不够灵活

We propose HybridFlow, which combines single-controller and multi-controller paradigms in a hybrid manner to enable flexible representation and efficient execution of the RLHF dataflow. We carefully design a set of hierarchical APIs that decouple and encapsulate computation and data dependencies in the complex RLHF dataflow, allowing efficient operation orchestration to implement RLHF algorithms and flexible mapping of the computation onto various devices. 
>  我们提出 HybridFlow，它结合了单控制器和多控制器范式，以灵活表示和执行 RLHF 数据流
>  我们设计了一组分层 API，将 RLHF 数据流中的计算和数据依赖关系解耦并封装起来，以高效编排 RLHF 算法操作，灵活地将计算映射到各个设备上

We further design a 3D-HybridEngine for efficient actor model resharding between training and generation phases, with zero memory redundancy and significantly reduced communication overhead. 
>  我们进一步设计了 3D-HybridEngine，用于在训练和生成阶段之间高效地 resharding actor model，实现零内存冗余和显著降低的通信开销

Our experimental results demonstrate  $1.53 \times \sim 20.57 \times$  throughput improvement when running various RLHF algorithms using HybridFlow, as compared with state-of-the-art baselines. HybridFlow source code will be available at https://github.com/volcengine/verl
>  实验结果表明，相较于 SOTA 的基线，HybridFlow 运行各种 RLHF 算法的吞吐提高了 1.53-20.57 倍

CCS Concepts:  Computing methodologies  $\rightarrow$  Distributed computing methodologies; Machine learning.

Keywords: Distributed systems, Reinforcement Learning from Human Feedback

# 1 Introduction
Large language models (LLMs) such as GPT [11], Llama [73] and Claude [7] have revolutionized various artificial intelligence (AI) applications, ranging from writing [2], searching [52] to coding [63]. LLMs are first pre-trained on trillions of tokens from books, websites, etc., via next-word prediction to accumulate broad knowledge [11]. Next, LLMs are trained on domain-specific datasets via supervised fine-tuning (SFT), to be able to follow human instructions [11]. Despite the outstanding capabilities of LLMs on natural language tasks after pre-training and SFT, the detrimental and biased contents in the training datasets may still mislead an LLM to generate toxic and undesirable content. Reinforcement Learning from Human Feedback (RLHF) is introduced to further align an LLM to human values, for building helpful and harmless AI applications [7, 55].

RLHF is built upon traditional RL algorithms [4, 68, 78], e.g., Proximal Policy Optimization (PPO) [68] and REINFORCE [78]. The widely adopted PPO-based RLHF system typically consists of four LLMs [7, 55]: an actor, a critic, a reference policy network and a reward model. PPO-based RLHF proceeds in iterations, each with three stages: (1) response generation using the actor model with a batch of prompts; (2) preparation of training data by scoring the generated responses through a single forward pass of the critic, reference policy, and reward models;  (3) learning from human preference by updating actor and critic through forward and backward computation. Other RLHF variants [19, 43] follow similar stages but involves different numbers of models and data dependencies among the models.
>  广泛采用的 PPO-based  RLHF 系统通常由四个 LLM 组成: actor, critic, reference policy network, reward model
>  PPO-based RLHF 迭代执行三个阶段:
>  1. 使用 actor model，为 batch of prompts 生成 response (前向传播)
>  2. 使用 critic, reference policy, reward models，为生成的 response model 打分，以准备训练数据 (前向传播)
>  3. 更新 actor, critic 以学习人类偏好 (反向传播)

Traditional RL can be modeled as a dataflow [46], which is a directed acyclic graph (DAG): each node in the RL dataflow represents computation of a neural network (e.g., actor or critic network which can be CNN or MLP); each edge denotes data dependency between NN computations (e.g., output of the critic is used as input to actor training [68].) RLHF dataflow is more complex, with more complicated models involved (e.g., LLMs for the actor/critic/reference/reward models), each running distinct computation, and more diverse data dependencies among them (i.e., multicast between distributed model partitions). 
>  传统 RL 可以建模为数据流，是一个有向无环图
>  其中每个节点表示 NN 的计算，每条边表示 NN 计算之间的依赖 (例如 critic 的输出作为 actor 训练的输入)
>  RLHF 中使用 LLM 作为 NN，故数据流更加复杂，模式的分布式 partitions 之间存在多对多组播

Training and generation of an LLM in the RLHF dataflow requires distributed computation (e.g., using tensor/pipeline/data parallelism) [40, 71]. Therefore, each node in the RLHF dataflow is a complex distributed program, corresponding to distributed computation of the respective LLM. Models in different nodes typically use different parallelism strategies as their workloads vary. The edge represents data resharding, which is often a many-to-many multicast. Consequently, Flexible representation and efficient execution of the complex and resource intensive RLHF is imperative.
>  在 RLHF 数据流中，LLM 的训练和生成需要分布式计算 (利用 tensor/pipeline/data parallelism)，因此数据流图中的每个节点都是一个分布式程序，不同的节点根据不同的 workload 采用不同的并行策略
>  边表示数据重分片，这通常是多对多的组播

Traditional RL frameworks such as RLLib [45] and RLLib Flow [46] utilize a hierarchical single-controller paradigm to run RL dataflows. A centralized controller assigns nodes in the dataflow to different processes and coordinates their execution order. Each node process can further spawn more workers to perform computation, again following the single-controller paradigm. However, they only provide primitives for data-parallel training and are constrained to neural networks that are at most hundreds of MB in size [45, 46]. In the RLHF dataflow, each node corresponds to an LLM with up to billions of operators, computed using some complex parallelism. A single-controller paradigm is inefficient due to the substantial overhead of dispatching operators to distributed accelerators [1, 9].
>  传统 RL 框架，例如 RLLib, RLLib Flow 使用层次化的单控制器范式来运行 RL 数据流
>  其中，中心的控制器将数据流中的节点分配给不同的进程，并协调其执行顺序，每个节点进程可以进一步生成更多的工作进程来执行计算 (仍然遵循单控制器范式)，由此构成了一个层次化的形式
>  RLHF 数据流中，每个节点对应于一个 LLM，其中包含了数十亿个 operators，这些 operators 使用特定的并行策略执行计算
>  在这个场景下，单控制器需要分发的 operators 数量太多，调度开销较大，故效率较低

Existing RLHF systems adopt a multi-controller paradigm to manage intra-node computation and inter-node data-resharding [17, 30, 80]. Each controller independently manages the computation of one device and uses multiple point-to-point operations to coordinate data dependencies between different nodes. This multi-controller paradigm introduces negligible dispatch overhead when performing LLM computation (detailed in §2.2).
>  现存的 RLHF 系统采用多控制器范式来管理节点内部的计算和节点之间的数据重分片，其中每个控制器独立管理一个设备上的计算，控制器之间使用多个点对点操作来协调不同节点之间的数据依赖关系
>  这种多控制器范式在执行计算时引入的调度开销非常少

However, without central control, it is inflexible to implement various RLHF dataflow, as modifying a single node to adapt to different data dependencies requires changing all dependent nodes' implementation, hindering code reuse.
>  但没有中心控制的情况下，要实现各种 RLHF 数据流会不够灵活，因为修改一个节点以适应不同的数据依赖关系需要更改所有相关节点的实现，这会阻碍代码的复用

To address these limitations, we propose HybridFlow, a flexible and efficient RLHF framework to easily represent and execute diverse RLHF dataflows, attaining high throughput. Our key observation is that utilizing the single-controller paradigm on the inter-node level enables flexible expression of various data dependencies and easy coordination of inter-node data resharding with minimal overhead, while integrating the multi-controller paradigm within intra-node computation enhances computation efficiency substantially. We advocate a hierarchical hybrid programming model to generate RLHF dataflows. At the node level, multiple model classes are provided that encapsulate distributed computation (training, inference and generation) of different LLMs in the dataflow into primitive APIs. These APIs can seamlessly support various parallelism strategies from the existing LLM frameworks, including 3D parallelism [71], ZeRO [59], and PyTorch FSDP [57]), and perform distributed computation under the multi-controller paradigm. Among the nodes, a set of transfer protocols are designed to hide the complexity of data resharding from users, as coordinated by a single controller. This programming model abstracts away the complexity of distributed computing, allowing users to implement an RLHF dataflow in a few lines of code and run RLHF through a single process of the single controller. It also effectively decouples intra-node computation and inter-node data transfer, allowing independent optimization of each model without changing the code of other models in the dataflow.
>  HybridFlow 目的的解决这个问题
>  我们的关键思路是: 在节点间的层级使用单控制器范式可以灵活表达各种数据依赖关系，并以最小的开销协调节点间的数据重分片，而在节点内使用多控制器范式则能显著提高计算效率
>  我们提出一个分层的混合编程模型来生成 RLHF 数据流:
>  - 在节点层级，提供了多个模型类，将数据流中的 LLM 的分布式计算 (训练、推理和生成) 封装为 primitive API，这些 API 可以无缝支持现有 LLM 框架中的多种并行策略，包括 3D 并行, ZeRO 和 PyTorch FSDP
>  - 在节点间，设计了一组传输协议，向用户隐藏了数据重分片的复杂性
>  这一编程模型抽象了分布式计算的复杂性，允许用户使用几行代码就能实现 RLHF 数据流，并通过单一控制器的单一进程运行 RLHF
>  它还有效解耦了节点内计算和节点间数据传输，允许对每个模型进行独立优化而无需修改数据流中其他模型的代码

Training and generation of the actor model represent major computation in the RLHF dataflow. We further design a 3D-HybridEngine to enable efficient execution of training and generation of the actor model, introducing zero memory redundancy and significantly reduced communication overhead during model parameter resharding between the training and generation stages. Our hybrid programming model also facilitates flexible placement of models onto the same or different sets of GPU devices. This allows us to provide an effective algorithm to optimize GPU allocation and placement of the models, with various model sizes and distinct workloads, for any RLHF dataflow. 
>  actor 模型的训练和生成是 RLHF 数据流中的主要部分
>  我们进一步设计了 3D-HybridEngine，以实现 actor 模型训练和生成的高效执行，为训练和生成阶段之间的模型参数重分片引入了零内存冗余和显著降低的通信开销
>  我们的混合编程模型也支持将模型灵活地放在相同或不同的 GPU 设备上，使得我们能够为任何 RLHF 数据流提供一种有效的算法，用于优化 GPU 的分配和模型的部署，适用于各种模型大小和 workload

Our contributions in designing HybridFlow are summarized as follows:

- We propose a hierarchical hybrid programming model for conveniently building the RLHF dataflow. This programming model enables efficient distributed execution of intra-node computation and flexible inter-node data resharding and transfer, for various RLHF algorithms (§4).
- We design a 3D-HybridEngine that executes training and generation of the actor model with high computation efficiency and zero-redundancy transition between the training stage and the generation stage (§5).
- We devise an effective mapping algorithm to automatically identify optimized GPU allocation and placement of each node (model) in the RLHF dataflow (§6).
- We conduct extensive experiments comparing HybridFlow with state-of-the-art RLHF systems [17, 30, 82] under various RLHF algorithms, model sizes and cluster scales. Our evaluation demonstrates  $1.53 \times 20.57 \times$  throughput improvements.

>  贡献如下:
>  - 提出了层次化混合编程模型，用于构建 RLHF 数据流，该模型中，节点内分布式计算高效，节点间数据重分片和传输灵活
>  - 设计了 3D-HybridEngine，它高效执行 actor 的训练和生成，为训练和生成阶段引入零冗余传输
>  - 设计了高效的映射算法，自动识别 RLHF 数据流中，每个节点的优化的 GPU 分配和放置策略
>  - 执行了实验，发现吞吐显著提高

We have open-sourced HybridFlow and believe that HybridFlow can boost future RLHF research and development.

# 2 Background and Motivation
## 2.1 Reinforcement Learning from Human Feedback
**RLHF Workflow.** RLHF aligns the linguistic space of LLMs with human values, using a set of human-ranked candidates of given prompts [7, 19, 41, 43, 55, 70, 91]. An RLHF system typically consists of multiple models, e.g., an actor, a critic, a reference policy, and one or multiple reward models. The actor and the reference are each pre-trained/fined-tuned LLM (i.e., the LLM that is undergoing RLHF). The critic and reward models can be different LLMs fine-tuned on the human preference dataset, with the language modeling head replaced by a scalar output head [7, 55]. 
>  一个 RLHF 系统通常包含多个模型，例如 actor, critic, reference policy, reward models
>  actor 和 reference policy 通常是 pre-trained/fined-tuned LLM (要进行 RLHF 的 LLM)，critic 和 reward models 可以是在偏好数据集上微调的其他 LLM (把 language modeling head 替换为 scalar output head)

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/89b7418ed69f5eebeac04bcf874620d31351c455508a34ef4c2686a1b696d828.jpg)  

Figure 1. Dataflow graph of 3 RLHF algorithms [19, 43, 55]. Stage $(1)$ $(2)$ $(3)$  represent Generation, Preparation, and Training, respectively.

The RLHF workflow can be decomposed into 3 stages (Figure 1) and we take PPO as an example:

- Stage 1 (Generation): The actor produces responses from a batch of prompts using auto-regressive generation. 
- Stage 2 (Preparation): Using prompts and generated responses, the critic computes their values [66, 68], the reference policy computes their reference log probabilities, and the reward model computes their rewards [7, 55], all via a single pass of forward computation of the respective model. 
- Stage 3 (Learning/Training): The actor and the critic are updated via Adam [38], using the batch of data produced by previous stages and the loss function [55].

>  RLHF workflow 分为三个阶段:
>  - Generation: actor 接收一个 batch 的 prompts，生成 responses (接收状态，输出动作)
>  - Preparation: critic 接收 prompts 和 responses (状态和动作)，计算动作价值，reference policy 计算参考对数概率，reward model 计算奖励，这些都是前向计算
>  - Learning/Training: actor 和 critic 基于上一个阶段产生的批量数据，计算损失 (损失和奖励以及对数概率有关，也就是 entropy-augmented policy gradient)，使用 Adam 更新

>  可以看到整个过程中，奖励模型和参考策略都是固定的，故流程实际上仍然是经典的 actor-critic，只不过进行了批量化: actor 接收一批量状态，生成一批量动作，critic 为一批量的状态-动作对给出价值，然后更新

Other RLHF algorithms largely follow the 3-stage workflow as well (Figure 1 (b,c)). Safe-RLHF [19] introduces an auxiliary pretrain loss following PPO-ptx [55] and includes an additional cost model to fit human preferences and safety labels simultaneously. ReMax [43] requires an additional generation pass for variance reduction and eliminates the critic model in the dataflow. Researchers are actively exploring novel RLHF algorithms [41, 70, 91] and integrating traditional RL methods into RLHF domains [37]. These variances necessitate a flexible representation of the RLHF dataflow graph to accommodate diverse algorithmic requirements.
>  其他的 RLHF 算法基本遵循这个 3 阶段 workflow
>  RLHF 算法有许多变体，故能够灵活表示 RLHF 数据流图十分重要

**Parallelism Strategies.** LLMs are trained and served with data, pipeline, and tensor parallelism [36, 40, 54]. With data parallelism (DP), the input data is split into multiple subsets; each subset is processed by a separate device (e.g., a GPU) [69]. ZeRO [59] is a memory-optimized solution for DP training, progressively sharding optimizer states, gradients, and model parameters across GPUs. Pipeline parallelism (PP) [32, 53] and tensor parallelism (TP) [71] distribute model parameters, gradients and optimizer states across multiple GPUs. 
>  LLM 的训练和推理的并行化策略有三种: 数据并行、流水线并行、张量并行
>  数据并行 (DP): 输入数据被划分为多个子集，每个子集由单独的设备处理，ZeRO 是一个针对 DP 的内存优化方案，它通过初步分片优化器状态、梯度和模型参数到不同 GPU 上，来减少不同 GPU 的显存占用
>  流水并行 (PP): 将模型参数划分为多个阶段，每个阶段分配给一组设备，数据像流水线一样经过这些阶段
>  张量并行 (TP): 将单个模型层中的大型张量划分到多个设备上

Modern distributed training frameworks like Megatron-LM [71] and MegaScale [36] utilize 3D parallelism or PTD parallelism [54], where P, T, D stand for PP, TP, DP, respectively. In 3D parallelism, PP size represents the number of pipeline stages in model training, TP size refers to the number of shards that a tensor is partitioned into, and DP size is the number of model replicas. LLM serving systems employ 3D parallelism similar to training while only model parameters and KVCache are sharded [16, 29, 40].
>  现代分布式训练框架例如 Megatron-LM, MegaScale 使用 3D 并行，或称 PTD 并行 (流水线并行 + 张量并行 + 数据并行)
>  3D 并行中，PP size 表示流水线阶段数量，TP size 表示 tensor 划分为的 shards 数量，DP size 表示 model replicas 的数量
>  LLM 服务系统和训练系统类似，也采用 3D 并行，差异在于 TP 并行中仅 shard 模型参数和 KVCache (不涉及梯度、优化器状态)

LLM models in the RLHF dataflow may perform distinct computations, including training (one forward pass, one backward pass and model update), inference (one forward pass) and generation (auto-regressive generation with multiple forward passes). In particular, training and generation are performed on the actor model, training and inference on the critic, and inference on reference policy and reward models. Distinct parallel strategies can be applied to different models for varied computations to achieve optimal throughput.
>  RLHF 数据流中，LLM 会执行三类计算:
>  1. 训练: 一次 foward pass, 一次 backward pass，一次模型参数更新
>  2. 推理: 一次 fowrad pass
>  3. 生成: 多次 foward pass
>  具体地说，RLHF 中:
>  - actor 执行训练和生成
>  - critic 执行训练和推理
>  - reference policy, reward model 执行推理
>  因此，可以对不同的模型应用不同的并行策略，以达到最大吞吐

## 2.2 Programming Model for Distributed ML

![[pics/HybridFlow-Fig2.png]]

**Single-Controller.** It employs a centralized controller to manage the overall execution flow of the distributed program. With centralized control logic, users can build core functionalities of the dataflow as a single process (Figure 2(b)), while the controller automatically generates distributed workers to carry out the computation. With a global view of the hardware and dataflow graph, the single-controller paradigm allows flexible and optimized resource mapping and execution order coordination among dataflow tasks. However, coordination messages are passed from the controller to all workers, incurring significant dispatch overhead when executing expansive dataflow graphs on large clusters [1, 9]. 
>  Single-Controller 模型使用中心化的 controller 管理分布式程序的整体执行流
>  中心化的 controller 具有硬件和数据流图的全局知识，依据这些知识进行数据流任务之间的资源映射和执行顺序协调，并自动将计算分配给 workers
>  但中心化的 controller 需要自己向所有 workers 传递协调消息，因此在大型集群上执行大规模数据流图是，会存在显著的分发开销

**Multi-Controller.** Each device (aka worker) has its own controller. State-of-the-art distributed LLM training and serving systems adopt the multi-controller paradigm, due to its scalability and low dispatch overhead (control messaging largely passed from CPU to GPU over fast PCIe links) [36, 40, 60, 71]. 
>  Multi-Controller 模型中，每个设备 (worker) 都有自己的 controller
>  SOTA 的分布式 LLM 训练和服务系统采用 multi-controller 模型，因为该模型的分发开销小 (控制消息直接从 CPU 通过本地 PCIe 发送到 GPU，不需要网络通讯)，并且拓展性强

As shown in the example that employs multi-controller RLHF implementation in Figure 2(a), a separate program is run for each model, and all workers of one model execute the same program. Each worker only possesses a local view of the system state and requires point-to-point communication between two models (blue code and arrows) to coordinate model execution order. 
>  Figure 2a 中，每个模型 (actor model, critic model, reward model) 运行自己独立的程序，该模型对应的所有 workers 都运行这一相同的程序
>  该范式下，每个 worker 处理系统状态的局部视图，并且 model 与 model 之间在协调执行顺序时是点对点通信

To implement an RLHF workflow in the multi-controller architecture, a user must intricately integrate the code for collective communication, computation, and point-to-point data transfer in the program run at each device. This leads to deeply nested code of computation and data transfer, challenging to develop, maintain, and optimize. In Figure 2(a), each model performs local computation and all gather operations (black code), while the actor model must explicitly manage send operations to the critic and reward models, and the latter must correspondingly implement receive operations at precise points in their program.
>  为了在 Multi-Controller 架构下实现 RLHF 工作流，用户需要在 (会在每个设备上都运行的) 程序中集成用于集合通讯、计算、点对点数据传输的代码
>  这导致了计算和数据传输的代码嵌套，难以开发和维护
>  例如 Figure 2a 中，每个 model 都需要执行本地计算和 all gather 操作，但同时 actor model 必须显式管理和 critic, reward model 的通信，critic, reward model 也必须在其程序中实现接收语义

## 2.3 RLHF Characteristics
**Heterogeneous model workloads.** The actor, critic, reference and reward models in RLHF may execute training, inference or generation at different stages, with different memory footprint and computation demand. For reference policy and reward models, only their model parameters need to be stored in GPU memory, as they perform only the forward pass computation. For the actor and the critic, their model parameters, gradients, and optimizer states must be stored as they undergo model training. Moreover, a small actor model (e.g., a 7B pre-trained/fine-tuned LLM) can be paired with larger critic and reward models (e.g., 70B LLMs) in RLHF for better alignment [7]. Given such heterogeneity, different parallelism strategies and tailored optimizations are needed for running each model during RLHF.
>  异构的模型工作负载
>  RLHF 中，actor, critic, reference, reward model 会在不同阶段执行训练、推理或生成，故具有不同的内存消耗和计算需求
>  reference model 和 reward model 仅执行前向计算，故只需要在显存中存储模型参数
>  actor, critic 还需要进行反向计算，故需要存储模型参数、梯度、优化器状态
>  此外，为了更好的对齐效果，小 actor 模型会搭配更大的 critic, reward 模型
>  考虑到这样的异构性，需要为 RLHF 中不同的 model 设计不同的并行策略和优化方法

**Unbalanced computation between actor training and generation.** In the RLHF dataflow, training and generation of the actor model are represented by two nodes (Figure 1), which often render majority of the workload in each RLHF iteration (e.g.,  $58.9\%$  of total RLHF time with HybridFlow). Actor training is computation bound [24], often requiring a larger model-parallel (MP) size (i.e., the number of partitions the model is partitioned into) and distributing the workload to more GPUs, e.g., 8 partitions of a 7B model on 8 GPUs. Using the same parallelism strategy (e.g., the same MP size) for generation can lead to underutilization of GPU computation resources due to its memory-bound nature [40]. 
>  actor 训练和生成的计算不平衡
>  RLHF 数据流图中，actor model 的训练和生成使用两个不同的节点表示，这两个节点通常是每次 RLHF 迭代的主要 workload
>   actor training 为 computation bound，通常要求更大的模型并行 size (即模型的 partitions 数量) 来将计算划分到更多的 GPUs
>   actor generation 则是 memory bound，如果对 actor generation 采用和 actor training 相同的并行策略 (例如相同的 partitions 数量)，反而不会充分利用 GPU 计算资源

Previous studies show that combining a larger DP size with a smaller MP size (hybrid data and model parallelism), e.g., partition a 7B model into two and replicate it four times on 8 GPUs, can improve the generation throughput [44, 92]. 
>  之前的研究发现使用更大的 DP size，结合更小的 MP size，例如 MP=2, DP=4，在 8 个 GPU 上并行，可以提高生成吞吐 (MP size 小一点，减少通讯开销，往计算密集的方向靠)

Although using different parallelism strategies for actor training and generation may optimize throughput in both stages, resharding the actor model weights at runtime between the two stages can incur significant communication and memory overhead. For example, aligning a 70B actor model requires transferring 140GB of model weights from training to generation per RLHF iteration, taking up to  $36.4\%$  of an iteration time when the two stages are on different devices [30].
>  虽然为 actor training, actor generation 使用不同的并行策略可以同时优化两个阶段的吞吐，但在 actor training, actor generation 的 (模型) 并行策略不同的情况下，需要在运行时的 training, generation 阶段之间对 actor model 权重进行 resharding，这会导致严重的通讯和访存开销
>  例如，如果 actor model 的参数量为 70B，我们就需要在**每一次**RLHF 迭代迁移 140GB 的模型权重，如果 training, generation 在不同设备上执行，这将占据 36.4% 的迭代时间


![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/042181790649a55a9042731b9d324ca507a8874d5855db74f397bc84e55bdad5.jpg)  

Figure 3. Dataflow execution given a model placement plan. Blocks with numbers represent GPUs. In dashed boxes, the models are placed on different sets of devices and can be concurrently computed. Reference model (blue) and reward model (green) are colocated on the same set of GPUs and executed sequentially.

**Diverse model placement requirements.** Strategic device placement of models in the RLHF dataflow is necessary, according to computation workloads and data dependencies of the models. Figure 3 gives an example model placement plan and the corresponding RLHF execution flow. Models placed on different sets of devices can be executed in parallel if no data dependencies exist. Models placed on the same set of GPUs, referred to as colocated models, share the GPU memory and are executed sequentially in a time-sharing manner, as out-of-memory (OOM) error may easily happen if colocated LLMs execute concurrently.
>  多样的 model placement 需求
>  Figure3 给出了一个 model placement 方法和对应的 RLHF 执行流
>  如果没有数据依赖存在，放置在不同组设备上的 model 可以并行执行
>  放置在相同的一组 GPUs 上的 model 共享 GPU 显存，以时分方式顺序执行 (因为并发执行容易导致显存溢出)

We observe a compromise: placing models on different devices permits parallel processing but may inevitably lead to some GPU idle time, given staged model execution in RLHF. In Figure 3, actor and critic are placed separately, performing training in parallel, but incurring 1/3 of their GPU time being idle, during other RLHF stages. Supporting various placement strategies and maximizing device utilization are crucial for optimizing RLHF performance at any model size and cluster scale.
>  将模型放在不同的设备上允许了并行执行，但会不可避免地导致一些 GPU 存在空闲时间，因为 RLHF 数据流的执行是分阶段的
>  例如 Figure3 中，actor, critic 是放置在不同组设备上的，进而二者的训练计算可以并行执行，但在整个迭代中，其 $1/3$ 的 GPU 时间是空闲的

Table 1. Comparison of RLHF frameworks. Figures ilustrate execution of one PPO iteration. Numbers 1-6 represent response generation, reward model inference, reference model inference, critic inference, actor training, and critic training, respectively.   placement strategies and maximizing device utilization are crucial for optimizing RLHF performance at any model size and cluster scale.

## 2.4 Limitations of existing RLHF systems
**Inflexible support for various RLHF dataflow graphs.** Existing RLHF systems adopt the multi-controller paradigm for dataflow implementation [17, 30, 80, 82]. To implement various RLHF algorithms, a user must navigate and manage code that mixes collective communication, model computation (potentially using various distributed training/serving frameworks), and point-to-point data transfer. This code structure lacks modularity/function encapsulation, making the RLHF systems tightly coupled with specific LLM training and serving frameworks. 
>  对各种 RLHF 数据流图的支持不灵活
>  现存的 RLHF 系统都采用 multi-controller 范式，如果用户想要在这样的系统上执行 RLHF 算法，就需要自行定位并管理集合通讯、模型计算 (可能使用不同的分布式训练/服务框架)、点对点数据传输
>  这使得代码结构缺乏模块性和函数封装，进而导致 RLHF 系统和特定的 LLM 训练和服务框架紧密耦合

Consequently, a user needs to implement and optimize different RLHF dataflows case-by-case [46], hindering code reuse and increasing the risk of making mistakes. Existing RLHF frameworks only support the PPO algorithm. In addition, limited parallel strategies are supported due to implementation complexity. For example, to incorporate 3D parallelism for LLM training and generation in DeepSpeed-Chat [82], one may have to re-implement the whole system due to the mixed code structure.
>  结果就是，用户需要逐一实现并优化不同的 RLHF 数据流，代码重用性低
>  现存的 RLHF 框架仅支持 PPO 算法，并且由于实现的复杂性，仅支持有限的并行策略

**Inefficient RLHF execution.** Table 1 summarizes parallelism strategies, model placement, and execution patterns adopted by the existing RLHF systems. DeepSpeed-Chat [82] and OpenRLHF [30] adopt ZeRO-3 for actor training and TP for actor generation. OpenRLHF uses different copies of the actor model on different devices for training and generation, incurring redundant memory usage and frequent weight synchronization among devices. DeepSpeed-Chat maintains the same copy of actor model on the same set of devices for training and generation, and reshards model weights between training and generation (due to different parallelisms used in the two stages), which may still incur substantial memory and communication overhead for large models (detailed in §5.4). NeMo-Aligner [17] uses the same 3D parallelism configurations in actor training and generation, experiencing low generation throughput (§8.4).
>  低效的 RLHF 执行
>  Table 总结了现存 RLHF 系统采用的并行策略、模型放置、执行模式
>  DeepSpeed-Chat, OpenRLHF 采用 ZeRO-3 执行 actor-training，采用 TP 执行 actor generation
>  OpenRLHF 将 actor model 进行了拷贝，training 和 generation 使用的是独立的存储空间，导致了冗余的内存使用和设备间频繁的权重同步
>  DeepSpeed-Chat 在 training 和 generation 时在同一组设备上维护 actor model 的单一拷贝，在 training, generation 阶段之间对模型权重进行 reshard (因为两个阶段使用不同的并行策略)，这对于大型模型仍然存在显著的内存和通信开销
>  NeMo-Alinger 在 training, generation 阶段使用相同的并行策略，导致 generaton 吞吐较低

![[pics/HybridFlow-Table1.png]]

Existing RLHF frameworks are limited to one model placement plan and hence one RLHF execution pattern, as shown in Table 1. Implementing a different placement is difficult, requiring changing the inner logic of model initialization and inter-node data transfer as highlighted in blue in Figure 2. 
>  如上所述，现存的 RLHF 框架都限制在一种 model placement 方法，因此限制在一种 RLHF 执行模式
>  要实现一个不同的 placement 则需要修改模型初始化和节点间数据传输的内在逻辑

OpenRLHF and NeMo-Aligner allow concurrent model computation in the preparation and learning stages; in the generation stage, models except the actor are idle, wasting the GPUs they occupy. DeepSpeed-Chat colocates all models on the same set of devices, and each device runs each model sequentially according to the RLHF dataflow. With unbalanced workloads among the models, such a placement can be inefficient in resource utilization (evaluated in §8.3).
>  OpenRLHF 和 NeMo-Aligner 允许在 perparation, learning stage 中模型计算的并发执行，但在 generation 阶段，除了 actor 以外，其他模型都空闲，浪费了它们占据的 GPU
>  DeepSpeed-Chat 将所有 models 都放置在同一组 GPU 上，进而根据 RLHF 数据流顺序执行各个 model，如果 models 的 workload 是不平衡的，这样的 placement 的资源利用非常低效

## 2.5 Design Considerations
To tackle limitations of existing systems, the key question is 
- **How to design a flexible and efficient programming model to implement RLHF dataflow?** 
    A single-controller design is particularly advantageous at the inter-node level due to its flexibility in coordinating data transfer, execution order, and resource virtualization among distributed computation of different models [9, 50]. The RLHF dataflow graph typically consists of only a few nodes. Dispatching control messages to different nodes from the single-controller incurs negligible overhead as compared to distributed computation required for nodes (models) in the dataflow. The multi-controller paradigm, known for its low latency in dispatching operators to accelerators [20], can be leveraged in distributed computation of each model. With these insights, we propose a hierarchical hybrid programming model for RLHF dataflow implementation. Our key design principle is to combine single-controller and multi-controller paradigms in a hybrid manner. This design ensures flexible expression and efficient execution of RLHF dataflow, maintaining low control overhead at both inter-node and intra-node levels. As shown in Figure 2(b), this paradigm decouples intra-node distributed computation and inter-node data transfer, allowing each model to focus solely on local computation without managing inter-node communication.

>  要解决现存系统的限制，关键的问题就是: 如何为 RLHF 数据流的实现设计一个灵活且高效的编程模型
>  single-controller 设计在节点间级别十分有用，因为它在协调分布式计算的数据传输、执行顺序、资源虚拟化上非常灵活
>  RLHF 数据流图通常仅包括几个节点，从单个 controller 向不同节点发送控制消息的信息传递开销非常小 (相较于数据流中节点的计算开销)
>  multi-controller 范式向加速器发送 operators 的延迟低，进而可以在每个 model 自己的分布式计算中利用
>  根据上述的思路，我们为 RLHF 数据流实现提出一个层次化的混合编程模型，其关键的设计原则是结合 single-controller 和 multi-controller 范式，同时在节点间和节点内级别维护低的控制开销
>  这一范式解耦了节点内分布式计算和节点间数据传输，允许各个模型仅仅管理本地计算，不需要管理节点间通讯

# 3 HybridFlow Overview

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/721c20cbd3caf71dd58b20cca68a588c2d84a64d0c29074201443b712f680af5.jpg)  

Figure 4. Architecture of HybridFlow.

Figure 4 depicts the architecture of HybridFlow, which consists of three major components: Hybrid Programming Model, 3D-HybridEngine and Auto-Mapping algorithm. The hybrid programming model includes a set of hierarchical APIs to enable flexible expression of the RLHF dataflow and efficient computation of models in the dataflow ($\S$ 4 ). The 3D-HybridEngine is particularly designed for efficient training and generation of the actor model, allowing different 3D parallel configurations in the two stages and enabling zero memory redundancy and minimized communication overhead during the transition between two stages ($\S$ 5). The auto-mapping algorithm determines optimized device placement of each model to maximize the throughput of RLHF ($\S$ 6).
>  HybridFlow 的结构如 Figure 4 所示，它包含三个主要成分: Hybrid 编程模型、3D-HybridEngine, Auto-Mapping 算法
>  Hybrid 编程模型包含了一组层次化 API，允许灵活表示 RLHF 数据流和数据流中模型的高效计算 (Section 4)
>  3D-HybridEngine 针对 actor model 的高效训练和生成设计，允许对训练和生成阶段配置不同的 3D 并行配置，并且提供了零内存冗余以及两阶段转移之间最小化的通讯开销 (Section 5)
>  Auto-Mapping 算法为每个模型决定最优的 device placement，以最大化 RLHF 的吞吐

The workflow of our RLHF system goes as follows. A user provides the following inputs to start the RLHF system: (i) model specifications, including the architecture and size of the actor/critic/reference policy/reward models in the RLHF dataflow; (ii) device placement of the models in the dataflow, as obtained by running the auto-mapping algorithm under given GPU cluster configurations; (iii) parallelism strategy for running each model in each stage, e.g., a tuple of (p, t, d) for 3D parallelism, where p, t, d represent PP size, TP size and DP size, respectively. The single controller program takes these inputs to initialize models in the RLHF dataflow and virtualized resource pool, dispatches operations/models to devices according to the placement plan, and invokes functions run by the multiple controllers on devices to carry out distributed computation of each model.
>  HybridFlow 系统的 workflow 为:
>  1. 用户为系统提供以下输入: 
>  i) 模型规格，包括了 actor/critic/reference policy/reward model 的架构和大小 
>  ii) RLHF 数据流中的 device placement (实际上是在给定 GPU cluter 配置后，运行 auto-mapping 算法得到的) 
>  iii) 在每个计算，运行每个模型的并行策略，例如 3D 并行使用 tuple (p, t, d) 指定并行配置，表示 PP size, TP size, DP size
>  2. single-controller 程序接收这些输入，初始化 RLHF 数据流中的模型并虚拟化资源池，根据 placement plane 向设备发送 operations/models，并为每个 model 调用 multi-controller 的函数来执行 model 的分布式计算

The multi-controller program implements the ParallelWorker class: it constructs parallel groups of each model among allocated devices according to its parallelism strategies, invokes the 3D-HybridEngine for actor training and generation, and can be integrated seamlessly with existing LLM engines [40, 57, 60, 71] for training, inference and generation of other models. The transfer protocols are coordinated by the single controller program to support resharding of data (including prompts, responses, and other model outputs in RLHF) between models with distinct parallelism strategies. The data resharding of the actor between training and generation is handled by 3D-HybridEngine.
>  3. multi-controller 程序实现了 ` ParallelWorker` 类，它在根据并行策略，在分配的设备中为每个模型构造并行组，并调用 3D-HybridEngine 执行 actor training 和 generation
>  single-controller 程序协调传输协议，以支持数据 (包括 prompts, responses, other model outputs) 在具有不同并行策略的模型之间的 resharding 
>  3D-HybridEngine 负责 actor training, generation 阶段之间的 data resharding

# 4 Hybrid Programming Model
## 4.1 Hierarchical APIs
**Intra-node: encapsulating distributed program.** For distributed computation of each model in different RLHF stages, we provide a base class, 3DParallelWorker. Given allocated devices, it facilitates distributed model weight initialization and establishes 3D parallel groups for each model. A parallel group includes a set of GPUs to host a specific parallel dimension of the model, e.g., different tensor shards in TP and different model replicas in DP. Figure 5(a) illustrates initialization of the actor model with our APIs, while initialization of other models is similar.
>  节点内: 封装分布式程序
>  我们为每个 model 在不同 RLHF 阶段的分布式计算提供基类 `3DParallelWorker`
>  给定分配的设备，该类执行分布式的模型权重初始化，并为每个模型构建 3D 并行组
>  并行组包含了一组 GPUs，各自存储 model 的特定并行维度，例如 TP 中不同的 tensor shards，DP 中不同的 model replicas

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/1436b45487f67375b1be02d32cdedb30686cfdc4a14b812f010c0cb74899c709.jpg)  

Figure 5. An illustration of hierarchical APIs. (a) Model with 3D parallel configuration, resource allocation, and 3DParallelWorker initialization. (b) Asynchronous data resharding between two models with collect and distribute functions in 3D_PROTO.

Inheriting from the 3DParallelWorker class, several model classes, for actor, critic, reference, and reward model, respectively, are provided. Each of these model classes encapsulates APIs to implement the model's distributed forward and backward computation, auto-regressive generation, and optimizer updates, decoupling the distributed computation code with data dependencies with other models. These APIs can be easily implemented by reusing the computation scripts from existing LLM systems. For example, the computation involved in update_actor function of ActorWorker (the class for the actor model) is similar to the pre-training scripts in Megatron-LM [71]. A model class encapsulates fundamental operations for implementing various RLHF algorithms, e.g., generate_sequences in the actor model class for generating responses based on the prompts and compute_reward in the reward model class for evaluating responses through a forward pass. (More APIs are detailed in Appendix A).
>  我们为各个 model 提供了继承了 `3DParallelWorker` 的类
>  这些 model 类封装了执行 model 的分布式前向和反向计算、自回归生成、优化器更新的 API，将分布式计算代码和与其他模型的数据依赖解耦
>  这些 API 的实现通过复用现存 LLM 系统的计算脚本达成，例如 `ActorWorker` 类的 `update_actor` 函数类似于 Megatron-LM 中的预训练脚本
>  model 类封装了实现各种 RLHF 算法的基本操作，例如 `ActorWorker` 中的 `generate_sequence` 方法，`RewardWorker` 中的 ` compute_reward ` 方法

Besides base class 3DParallelWorker that implements 3D parallelism, we further provide base classes for PyTorch FSDP (FSDPWorker) and ZeRO (ZeROWorker), and the corresponding model classes inheriting each base class, to support different parallelism strategies in model computation. ParallelWorker in Figure 4 denotes one of these base classes.
>  基类 `3DParallelWorker` 实现了 3D 并行，此外我们还为 FSDP, ZeRO 提供了 `FSDPWorker, ZeROWorker`，以及继承了这些类的模型类，以支持模型计算中的不同并行策略
>  Figure4 中的 ParallelWorker 就表示 `3DParallelWorker, FSDPWorker, ZeROWorker` 其中之一

**Inter-node: unifying data resharding implementation between models.** Many-to-many multicast is involved for data transfer between models employing different parallelism strategies on different devices. We unify this data transfer implementation by associating each operation in each model class with a transfer protocol, using @register. Each transfer protocol consists of a collect function and a distribute function, to aggregate output data and distribute input data according to the parallelism strategy of each model. 
>  节点间: 统一模型之间的 data resharding 实现
>  在采用不同的并行策略、位于不同设备上的 models 之间存在多对多的组播，我们通过将每个模型类中的每个操作都关联到一个传输协议来统一这个数据传输的实现
>  操作和传输协议的关联使用 `@register` 实现
>  每个传输协议都包含一组收集函数和分布函数，用以根据模型的并行策略聚合输出数据和分布输入数据

In the example in Figure 5(a), update_actor operation is registered to transfer protocol 3D_PROTO, as 3D parallelism is used for actor training. In 3D_PROTO, the collect function gathers all the output data of corresponding model function (e.g., the loss scalar return from the update_actor) in each DP group to the single controller, and the distribute function distributes the input data to the registered function (e.g., advantages for the update_actor) to each DP group. 
>  Figure 5a 的示例中，`update_actor` 操作被注册到 `3D_PROTO` 传输协议，因为 actor training 训练使用了 3D 并行
>  `3D_PROTO` 协议中，收集函数在每个 DP 组收集对应的模型函数的所有输出数据 (例如 `update_actor` 返回的标量损失值)，返回给 single controller，分布函数将输入数据分布给每个 DP 组中注册的函数 (例如将 advantages 分布给 `update_actor`)

Data resharding is enabled using the source model's output collect function and the destination model's input distribute function. Figure 5(b) illustrates data resharding between the actor (generation) and the critic (inference), where computation of the models adopts different 3D parallelism strategies. The single controller gathers data futures using the collect function in 3D_PROTO of actor (steps  $(1) -(3)$ ) and sends it to critic (step  $(4)$ ); critic distributes the received data futures to each DP group using the distribute function in its 3D_PROTO (step  $(5)$ ). Then remote data is retrieved from actor to critic, with each of critic's GPUs only fetching the required local batch of the actor's output data according to its DP rank (step  $(6)$ ). The actual data transfer only occurs between GPUs, avoiding any central bottleneck.
>  我们使用 source model 的输出收集函数和 destination model 的输入分布函数实现 data resharding
>  Figure 5b 展示了 actor (generation) 和 critic (inference) 之间的 data resharding，其中 actor, critic 采用的是不同的 3D 并行策略
>  Figure 5b 中，single controller 使用 actor 的 `3D_PROTO` 中的收集函数收集 data futures，然后发送给 critic, critic 使用 `3D_PROTO` 中的分布函数将 data futures 分布给各个 DP 组
>  然后 actor 开始向 critic 发送数据，过程中每个 critic GPUs 仅根据其 DP rank 按需获取 actor 输出中的局部 batch，实际的数据传输仅在 GPUs 之间发生，避免了 central bottleneck (central 模式仅用于传输元数据，即 data futures)

We provide 8 transfer protocols, including 3D_PROTO, DP PROTO, ONE_TO_ALL, etc., that cover most data resharding scenarios (detailed in Appendix B). A user can further extend the transfer protocols through implementing customized collect and distribute functions.
>  我们提供了 8 个传输协议，覆盖了大多数 data resharding 场景
>  用户可以通过实现自定义的收集和分布函数自定义传输协议

**Facilitating flexible model placement.** We provide a ResourcePool class that virtualizes a set of GPU devices. When applying a ResourcePool instance to a model class (Figure 5(a)), distributed computation of the model will be mapped to the devices. Models utilizing the same ResourcePool instance are colocated on the same set of GPUs; models are placed on different sets of GPUs when different Resource Pool instances are applied in their model classes. We assume no overlap between different ResourcePool instances.
>  灵活的 model placement
>  我们提供 `ResourcePool` 类来虚拟化一组 GPU 设备
>  当对 model class 应用 `ResourcePool` 实例时，模型的分布式计算就会被映射到设备上
>  使用了相同 `ResourcePool` instance 的模型即 colocated 在相同一组 GPUs 上的模型，放置在不同组 GPUs 上的模型使用不同的 `ResourcePool` 实例
>  我们假设了 `ResourcePool` 实例没有交叉

**Asynchronous dataflow execution.** When models are placed on separate sets of devices, their execution is triggered automatically as soon as their inputs become available [50]. In Figure 5(b), the data future from actor is immediately returned after the controller's call (steps  $(1) -(3)$ ); the controller then initiates a new call to critic and distributes the futures following the transfer protocol (steps  $(4) -(5)$ ). When some models are placed on the same set of devices, they are executed sequentially based on the calling order. With our programming model, HybridFlow is flexible in supporting diverse distributed execution patterns without any code change of the RLHF algorithm (Figure 6).
>  异步数据流执行
>  模型被放置在不同组设备上时，其执行完全异步，一旦其输入可用，执行就开始，模型放置在同组设备上时，则基于调用顺序顺序执行
>  HybridFlow 的编程模型掩盖了自行安排的复杂性，不需要修改 RLHF 算法，也可以支持多种分布式执行模式

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/2eb0087695d7dcb49ade86896b02b173eab2f3ffd61fefd3b48da5b31d71f9de.jpg)  

Figure 6. Implementation of PPO [55], ReMax [43], and Safe-RLHF [19]. Users can adapt to different RLHF algorithms by simply adding or deleting a few lines of code.

## 4.2 Implementation of different RLHF algorithms
Our APIs enable streamlined development of various RLHF algorithms (dataflows). Users can implement an RLHF algorithm in a few lines of code as a single process program to run on the single controller, that involves a sequence of primitive API calls to invoke distributed computation of models. 
>  HybridFlow API 下，用户仅需要几行代码就可以实现 RLHF 算法
>  用户编程的出发点是 single controller 运行的单个程序，因此只需要调用一系列 primitive API 来调用 models 的分布式计算

Examples of PPO, ReMax, and Safe-RLHF are given in Figure 6. PPO can be implemented in just 8 lines by invoking model operations including compute_values and generate_sequences, which are executed under the multi-controller paradigm on multiple GPUs. To adapt to Safe-RLHF which integrates an additional cost model to evaluate safety preferences and the pre-tainting loss for actor, only 5 more lines of code are added on top of PPO implementation. To adapt to ReMax, one additional call to actor generation is needed, and the critic-related code can be removed.
>  例如 Figure6 中，`compute_values, generate_sequences` 都是在 multi-controller 范式下，在多个 GPUs 执行的

**Achieving flexible.** This flexibility of extension is crucial for researchers to explore different RLHF algorithms: they can reuse distributed computation encapsulated in each model class and simply adjust the code for numerical computations according to specific algorithms, such as GAE [67] and KL divergence in compute_advance and loss functions of actor and critic. 
>  灵活性
>  对于用户来说，他们可以简单修改代码中的数值计算部分来变更 RLHF 算法，可以直接复用 model class 中封装的分布式计算代码

The streamlined development can be attributed to the hybrid programming model. Our modular API design simplifies development, facilitates extensive code reuse, and enables directly incorporating the codebase of existing LLM training/serving frameworks. It also decouples model computation and data transfer among models. Any change in the distributed frameworks does not affect the code of the RLHF algorithm (Figure 6), enabling individualized optimization for each model's execution (§5). Flexible placement of models with diverse workloads is supported, enabling optimized mapping of RLHF dataflow onto various devices (§6).
>  这种开发上的简化归功于混合编程模型的模块化 API
>  模块化 API 促进了代码复用，可以直接使用现存 LLM 训练/服务框架的代码，并且解耦了模型计算和模型间数据传输 (如果不解耦，有没有 overlap 优化的空间？)

# 5 3D-HybridEngine
We design the 3D-HybridEngine to support efficient training and generation of the actor model, targeting significant RLHF throughput improvement.
>  3D-HybridEngine 的设计目标是支持 actor model 的高效训练和生成，最大化 RLHF 吞吐

## 5.1 Parallel Groups
To eliminate redundant actor model copies, we advocate deploying actor training and generation stages on the same set of devices, $N_{a}$ GPUs allocated to the actor, and execute them sequentially on the same copy of actor model weights. Nonetheless, actor training and generation may well adopt different 3D parallelism strategies, i.e., the generation stage typically requires smaller TP and PP sizes but a larger DP size, than the training stage ($\S$ (2.3). 3D-HybridEngine enables efficient model parameter resharding between actor training and generation across the same set of devices in this context.
>  为了消除冗余的 actor model 拷贝，我们将 actor training, generation 阶段部署在相同的一组设备上，这两个阶段在相同的 actor model weights 上顺序执行
>  虽然部署在相同的设备上，training 和 generation 可以采用不同的 3D 并行策略，即 generation 相较于 training 通常 TP size, PP size 更小, DP size 更大
>  3D-HybridEngine 负责在相同的一组设备上执行高效的 model parameter resharding

Let  $p -t -d$  denote 3D parallel groups constructed for actor training, corresponding to the set of GPUs to host  $p$  pipeline stages,  $t$  tensor shards, and  $d$  model replicas [54]. 3D-HybridEngine builds different parallel groups for actor training and generation, according to their different 3D parallelism strategies, respectively. We use  $p_{g}, t_{g},$  and  $d_{g}$  to denote the size of generation pipeline parallel group, generation tensor parallel group, and micro data parallel group, respectively, in the generation stage.  $d_{g}$  indicates the ratio of model replica number in generation over that in training, i.e., each DP replica in training becomes  $d_{g}$  micro DP replicas, to process  $d_{g}$  microbatches of prompts and responses. We have  $N_{a} = p \times t \times d = p_{g} \times t_{g} \times d_{g} \times d$  such that  $d_{g} = \frac{pt}{p_{g}t_{g}}$ . The micro DP groups are employed exclusively in actor generation stage to render a larger DP size for full device utilization. The generation parallel groups are denoted by  $p_{g} -t_{g} -d_{g} -d$ .
>  记 actor training 的 3D parallel group 为 p-t-d，对应于 p 个流水线阶段, t 个 tensor shards, d 个 model replicas
>  记 actor generation 的 3D parallel group 为 $p_g, t_g, d_g,d$，其中 $d_g$ 表示 micro data parallel group 的数量
>  actor training 中的每个 DP model replica (共 $d$ 个) 都会成为 actor generation 阶段中 $d_g$ 个 micro DP replicas，处理 $d_g$ 个 microbatches of prompts and responses
>  总的 GPU 数量为 $N_a = p \times t \times d = p_g \times t_g \times d_g \times d$，因此 $d_g = \frac {pt}{p_gt_g}$
>  micro DP groups 仅在 generation 阶段使用，目的是提高 DP size，进而增大设备利用率

## 5.2 3D-HybridEngine Workflow

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/d992875e5408601e47e98f3065e6872eea0eb9bc080d2a16b96f4df8050ec20e.jpg)  

Figure 7. 3D-HybridEngine workflow in one RLHF iteration. 4 GPUs are used for actor training and generation. 1-2-2  $(p -t -d)$  parallel groups are used in training and 1-1-2-2  $(p_{g} -t_{g} -d_{g} -d)$  parallel groups are used in generation.

Between actor training in iteration  $i$  of RLHF and actor generation in iteration  $i + 1$ , the actor model parameters need to be resharded and prompts data to be distributed, following the parallel group configurations in the two stages. 
>  在第 i 次迭代的 actor generation 和第 i+1 次迭代的 actor training 之间，需要遵循两个阶段的 parallel group 设置，reshard actor 参数并分布 prompts

In iteration  $i + 1$  of RLHF, 3D-HybridEngine gathers the actor model parameters updated in iteration  $i$  (step  $①$  in Figure 7), for generation within each micro DP group. Then, the batch of prompts are loaded to each model replica (step  $②$ ), which generates responses (Generation stage of RLHF). Following this, 3D-HybridEngine performs an all-gather operation on the generation results within each micro DP group (step  $③$ ), and re-partitions model parameters according to the 3D parallelism for actor training (step  $④$ ). With model weights, prompts and responses correctly re-distributed, the loss of the actor model is computed and actor model weights are updated following the RLHF algorithm (step  $⑤$ ) - actor training stage of iteration  $i + 1$ .
>  如 Figure7 所示，在 RLHF 的第 i+1 次迭代，3D-HybridEngine 收集第 i 次迭代更新的 actor model 参数 (step 1)
>  之后，batch of prompts 会被加载到每个 model replica, model replica 开始生成 responses，即 RLHF 的 generation 阶段 (step 2)
>  然后，3D-HybridEngine 在每个 mirco DP group 的生成结果上执行一次 all-gather 操作，使得每个 DP group 中的每个设备都拥有该 DP group 完整的输入和输出 (step 3)
>  然后，3D-HybridEngine 将模型参数再划分，以计算梯度，执行 actor 模型训练，即 RLHF 的 training 阶段 (step 4)
>  这样就完成了 RLHF 的一次迭代，过程中模型参数、prompts, responses 都会被依照并行配置进行重分配和分发

## 5.3 Zero redundancy model resharding

Parallel grouping methods in 3D parallelism are typically as follows: PP and TP groups are formed by assigning consecutive ranks to pipeline stages and tensor shards, respectively; DP groups are constructed by selecting ranks at regular intervals, determined by the product of PP size and TP size. 
>  3D 并行中的并行分组方法通常是:
>  将编号连续的 GPU 分配给多个流水线阶段来构造流水线并行组 (例如 PP = 2，就将前一半 GPU 分配给第一个流水线阶段，将后一半 GPU 分配个第二个流水线阶段)
>  将编号连续的 GPU 分配给多个 tensor shards 来构造张量并行组 (例如 TP = 2，则连续的两个 GPU 负责计算一个 tensor 的两个 shards)
>  数据并行组的构成按照间隔采样，间隔大小就是 PP size x TP size (也就是用于计算单个 model replica 的 GPU 数量)

>  计算流程:
>  前向传播时，DP 组内没有通信，TP 组内互相通信协调 (TP all-reduce) 完成计算
>  反向传播时，DP 组内没有通信，TP 组内互相通信协调 (TP all-reduce) 完成计算
>  在反向传播最后，TP 组内，每个 GPU 都计算出了自己所负责的权重的梯度之后，再在各自的 DP 组内，在数据并行维度上对相同分片的梯度做平均 (DP all-reduce)
>  DP 同步是梯度平均，而不是模型同步

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/5a78fe283f77562d66e7215a72368321a9b0006d25889bd6245c28c0afae4ca1.jpg)  

Figure 8. Model weights resharding. 2 machines each with 4 GPUs are used for actor training and generation.

In Figure 8(a), actor training uses 3D parallel groups, 1-4-2: there is one PP group for all GPUs (for illustration clarify); the TP groups are `[G1, G2, G3, G4], [G5, G6, G7, G8]`, and the DP groups are `[G1, G5], [G2, G6], [G3, G7], [G4, G8]`. Suppose the same parallel grouping methods are used but with different parallel sizes, e.g., 1-2-2-2 for generation in Figure 8(a). During the transition from training to generation, 3D-HybridEngine applies all-gather operations among the model parallel groups to aggregate all parameters, and then retain only a subset of model weights on each device for its generation, according to the parallel groups the device belongs to. On some GPUs (e.g., G2, G3, G6, G7), there is no overlap between training and generation model weights, and separate memory is needed to maintain weights for subsequent training as well (grey boxes in Figure 8(a)). We call the system HybridFlow-V, when 3D-HybridEngine uses the above vanilla parallel grouping methods in the two stages.
>  Figure 8a 中，actor training 使用了 1-4-2 的 3D 并行: 
>  - 所有 GPU 构成一个 PP 组
>  - 所有 GPU 构成 2 个 TP 组，每个 TP 组有 4 个 GPU
>  - 所有 GPU 构成 4 个 DP 组，每个 DP 组有 2 个 GPU
>  假设 actor generation 使用了 1-2-2-2 并行
>  在从 training 到 generation 的过程中，3D-HybridEngine 会应用 all-gather 来聚合参数 (在每个模型并行组内，即 PP + TP 组内执行 all-gather，让每个 GPU 获取模型 replica 的完整权重)，然后 GPU 会根据并行配置，保留自己用于生成的那部分参数
>  在一些 GPU 上，训练和生成的模型参数不存在交叉，因此为了下一个迭代的训练，GPU 需要预留位置为下一个迭代训练所需要用到的参数 (下一次迭代训练需要用到的参数不能丢弃)

We further design a new parallel grouping method for 3D HybridEngine to use in the generation stage, that eliminates the redundancy in weights storage and leads to minimal memory footprint and communication due to actor model resharding between training and generation. 
>  为了消除这种权重存储的冗余性，我们为 3D HybridEngine 在 generation 阶段设计了一种新的并行分组方法 (不按照常用的 3D 并行中的分组方法)，这可以让 actor model 在 training 和 generation 阶段之间的内存消耗和通信减到最小

Specifically, we form generation TP and PP groups by selecting ranks at regular intervals, determined by  $\frac{t}{t_g}$  and  $\frac{p}{p_g}$ , and construct micro DP groups by sequentially assigning ranks along the generation TP or PP dimensions. 
>  具体地说，我们通过按照一定编号间隔选择 GPUs 来构造 generation TP 和 PP 组，间隔为 $t/t_g, p/p_g$，并且沿着 TP 或 PP 维度顺序选择 GPUs 来构造 micro DP 组

In Figure 8(b), 1-2-2-2 parallel groups are used in generation: the generation TP groups are `[G1, G3], [G2, G4], [G5, G7], [G6, G8]`; and the micro DP groups are `[G1, G2], [G3, G4], [G5, G6], [G7, G8]`. This strategic rearrangement of generation parallel groups leads to overlap between training and generation model weights on each device, enabling reuse of training weights during generation and zero redundancy in device memory usage due to model resharding. In addition, 3D-HybridEngine conducts several all-gather operations concurrently, one within each micro DP group, leading to significantly reduced communication overhead.
>  这种策略使得 generation 并行组中 TP 权重 shard 和 training 并行组中的 TP 权重 shard 可以完全交叉，进而可以在 genration 过程中复用 training 过程中保留的权重，并且不需要额外存储冗余的 training 权重
>  此外，这种策略下 3D-HybridEngine 可以并发在各个 micro DP 组内执行多个 all-gather，显著降低了通讯开销

## 5.4 Transition overhead

![[pics/HybridFlow-Table2.png]]
In Table 2, we compare communication overhead and memory footprint during the transition between training and generation stages, among different actor engine designs. We assume model size of the actor is  $M$  and  $N_{a}$  GPUs are used for its training and generation. The actor engine in DeepSpeed-Chat conducts an all-gather operation across all GPUs during transition; HybridFlow-V performs this all-gather within training TP and PP groups. The communication volumes for these operations are  $\frac{N_{a} -1}{N_{a}} M = \frac{tpd -1}{tpd} M$  for DeepSpeedChat and  $\frac{tp -1}{tp} M$  for HybridFlow-V, calculated following [13]. Both engines aggregate all model parameters in each GPU's memory before subsequently partitioning model states according to the generation parallel groups, resulting in a peak memory usage of model parameters  $M$ . As they cannot reuse training weights during generation on some GPUs, training weights need to be maintained on them, amounting to  $\frac{1}{tpd}$  and  $\frac{1}{tp}$  redundant memory consumption, respectively.
>  Table 2 比较了不同系统中 training, generation 阶段之间转移的通信和内存开销
>  DeepSpeed-Chat 在所有 GPU 上执行 all-gather
>  HybridFlow-V 在 training TP + PP 组内执行 all-gather
>  这两个系统都先在每个 GPU 上收集全部的模型权重，然后再根据 generation 阶段的配置再划分，因此峰值的内存用量都是 $M$
>  并且他们无法在 generation 阶段复用 training 阶段的权重分片，导致了冗余的内存消耗

>  all-gather 的通信量计算:
>  每个 GPU 持有 $\frac M {N_a}$ 的权重 (假设均匀分片)，要 gather 其他 $N_a - 1$ 个分片，总接收量就是 $\frac {N_a - 1}{N_a}M$ 

With our parallel grouping method for the generation stage, HybridFlow confines the all-gather operation within each micro DP group. The communication overhead is reduced to  $\frac{d_g -1}{tp} M = \frac{tp -tg_g}{t_gp_gt_p} M$ . Each GPU only needs to collect remote parameters within its micro DP group and can reuse the training weights in generation. Therefore, the peak memory usage of model parameters in HybridFlow precisely matches the model partition size on each GPU in generation, eliminating any redundancy in GPU memory usage.
>  HybridFlow 则只需要在 micro DP group 内执行 all-gather，并且可以在 generation 阶段复用 training 阶段的权重，因此峰值内存用量为 generation 阶段每个 GPU 的 model partition 大小，同时没有冗余 GPU 内存消耗

# 6 Auto Device Mapping
Our hybrid programming model requires users to input the following configurations, which are referred to as a mapping of the RLHF dataflow to the given devices: (a) device placement of the models in the dataflow; (b) the corresponding parallelism strategy for running each model in each stage.
>  Hybrid 编程模型要求用户输入以下配置:
>  1. RLHF 数据流中，models 的设备放置
>  2. 在每个阶段中，运行每个模型的对应并行策略

![[pics/HybridFlow-Algorithm1.png]]

We provide an efficient algorithm (Algorithm 1) for users to identify the optimized mapping of executing the RLHF dataflow on a given cluster of devices, that minimizes the end-to-end latency of each RLHF iteration. 
>  我们为用户提供了一个获取在给定设备集群时，确定执行 RLHF 数据流的优化的映射的算法，该算法最小化每次 RLHF 迭代的端到端延迟

Given a dataflow  $D$ , we first explore all possible placement plans  $\mathcal{P}$  for the models in the given cluster (Line 3). For example, the PPO algorithm involves four models, resulting in 15 possible placements (from the Bell partition problem [10, 62]), ranging from a completely standalone placement where all models are placed on different devices (e.g., OpenRLHF's placement) to colocating all models on the same set of devices (e.g., DeepSpeed-Chat's placement). 
>  给定数据流 $D$，我们探索给定集群中所有可能的放置方案 $\mathcal P$ (例如 PPO 算法涉及 4 个模型，将产生 15 种可能的放置)
>  可能的部署包括了完全独立的部署，即所有模型都部署在不同设备上，例如OpenRLHF 的部署，以及共存部署，即所有模型都部署在相同的一组设备上，例如 DeepSpeed-Chat 的部署

We refer to colocated models on the same set of GPUs as a colocated set. Models in a colocated set can employ different parallelism strategies across the same set of GPUs. 
>  我们将共同部署在相同一组 GPU 上的模型称为一个共存集
>  共存集种的 GPU 可以在同一组 GPU 上采取不同的并行策略

We identify the smallest number of GPUs to be allocated to each of the colocated model sets,  $A_{min}$ , based on memory consumption of colocated models, ensuring no out-of-memory errors (Line 9).
>  算法会根据共存的模型内存消耗分配给每个共存集所需要的最小 GPUs 数量，确保不出现 OOM 错误

Next, starting from the minimal GPU allocation in  $A_{min}$ , we enumerate all feasible device allocations to each colocated model set (Lines 10-12). Given device allocation  $A$  to the colocated set and computation workload  $W$  of models in the set, we explore optimized parallelism strategies for each model in the auto_parallel module, that minimizes model execution latency. 
>  我们从最小的 GPU 分配开始，枚举每个共存集中所有可能的模型分配 $A$ 和该共存集中模型的计算量 $W$，我们在 `auto_parallel` 模块中探索每个模型的优化并行策略，以最小化模型执行延迟

The workload  $W$  includes input and output shapes and computation (training, inference or generation) of each model. In auto_parallel, we utilize a simulator module simu to estimate the latency of different parallel strategies, following previous research [42, 84, 90, 92] (outline in Appendix. C).
>  计算量 $W$ 包括了每个模型的输入输出形状以及计算 (training, inference, generation)
>  `auto_parallel` 中使用了 `simu` 来评估不同并行策略的延迟

The d_cost module estimates the end-to-end latency of the RLHF dataflow under given model placement and parallelism strategies, by iterating through all stages in the dataflow graph and summing up latencies of all stages (Lines 17, 25). 
>  `d_cost` 评估了给定模型放置和并行策略下，RLHF 数据流的端到端延迟，方法是遍历数据流图中的所有阶段，并累加所有阶段的延迟

For models in the same colocated set and involving computation in the same stage (such as actor and critic both performing model update in RLHF training stage), their execution latencies are summed up (Line 32). For models in different colocated sets, their execution within the same stage can be parallelized, and the latency of the stage is determined by the maximum execution time among different sets (Line 33). 
>  在同一共存集中的模型，并且涉及了同一阶段的计算 (例如 actor, critic 都在 training 阶段执行计算)，其执行延迟会被累加
>  在不同共存集中的模型，他们在相同阶段的计算可以被并行化，延迟就是最大的一方

We identify the best device placement of the models with their corresponding parallelism strategies, achieving minimal execution time per RLHF iteration (Lines 18-23).
>  这样遍历下来，我们确定了能够在每次 RLHF 迭代达到最小执行时间的模型的最优设备放置以及其并行策略

The complexity of Algorithm 1 is  ${O}(\frac{(N -1)!}{(k -1)!(N -k)!})$ , where  $k$  is the number of models in the dataflow and  $N$  is the total number of devices to run the dataflow. This is the worst-case complexity for enumerating all possible device allocations for a placement strategy (i.e., the standalone placement), calculated by assigning  $N$  devices to  $k$  models (known as the integer partition problem [6]). For better efficiency, we cache parallelism strategies identified for each model on a number of devices  $A$ , to eliminate redundant searches for the same parallelism strategies when the model is placed on different sets of  $A$  GPUs in different placement strategies.

Though we assume  $N$  homogeneous GPUs when running the auto mapping algorithm, Algorithm 1 can be readily extended for optimizing model mapping over heterogeneous devices, by considering heterogeneous devices in simu and auto_parallel modules [88].

# 7 Implementation
HybridFlow is implemented in around 12k lines of Python code (LoC).

**Hybrid programming model.** The hierarchical APIs are implemented with 1.8k LoC. The centralized single controller is built on top of Ray [50] and uses Remote Process Calls (RPC) to coordinate the execution order of different models and transfer data between models following the dataflow. These intermediate data are stored in TensorDict [57]. In our multi-controller paradigm for distributed computation, each model function runs on a separate process across various devices, with control messages relayed from each controller's CPU process to the corresponding GPU. Our implementation supports Megatron-LM, PyTorch FSDP, and DeepSpeed as the LLM training and inference engines, and vLLM for autoregressive generation. In vLLM, we replace the centralized KVCache manager with a distributed manager to align with the multi-controller paradigm.
>  Hybrid 编程模型
>  中心化 single controller 基于 Ray 构建，使用 RPC 来协调不同模型的执行顺序以及在顺着数据流，在模型之间传递数据
>  中间数据存储在 TensorDict 中
>  multi-controller 中，每个模型函数在多个设备上运行，控制消息从每个 controller 的 CPU 进程传递到对应的 GPU
>  我们支持 Magatron-LM, PyTorch FSDP 和 DeepSpeed 作为 LLM 训练和推理迎请，使用 vLLM 进行自回归生成

**3D-HybridEngine.** Its main logic is implemented with 2.4k LoC on top of Megatron-LM and vLLM. We store actor model weights for training and generation stages on separate memory buffers, offload generation weights to the CPU memory during training, reload generation weights back to GPU memory during the transition, and use both buffers in generation. We use NCCL communication primitives [35] to collect and concatenate model parameters in each micro DP group during the transition between training and generation. We offload KVCache to CPU memory after generation and reload it back to GPU in the next iteration.
>  3D-HybridEngine 的逻辑基于 Megatron-LM 和 vLLM 实现
>  我们将 actor model 用于训练和生成阶段的权重存储在不同的内存缓冲区中，在训练时将生成权重卸载到 GPU 内存，在转换期间再重新加载，然后同时使用这两个缓冲区
>  我们用 NCCL 通信原语来收集和拼接每个 micro DP group 的模型参数 (也就是实现转移阶段的 all-gather)
>  我们在生成之后将 KVCache 卸载到 CPU 内存，然后在下一次迭代后再重新加载到 GPU

**Auto-Mapping Algorithm** is implemented with 1.9k LoC, together with three simulators for training, inference, and generation workloads. The algorithm is run before starting the RLHF dataflow on CPU, to generate device mapping and parallelism strategies for dataflow initialization.
>  该算法在启动 RLHF 数据流之前运行，以生成设备映射和并行策略

# 8 Evaluation
## 8.1 Experimental Setup
**Testbed.** We deploy HybridFlow on a cluster of 16 machines (128 GPUs). Each machine is equipped with 8 NVIDIA A100-80GB GPUs inter-connected with 600GB/s NVLink. The inter-machine bandwidth is 200Gbps. Our experiments use the following software versions: CUDA12.1, PyTorch 2.1.2, Megatron-core 0.6.0, NCCL 2.18.1, and vLLM 0.3.1.
>  我们在 16 个机器的集群上 (128GPUs) 部署 HybridFlow，每台机器有 8 张 A100-80GB，使用 600GB NVLink 互联，机器间带宽为 200Gbps

**Models and RLHF algorithms.** We run the RLHF dataflow (Figure 1) of PPO [68], ReMax [43] and Safe-RLHF [19] algorithms. PPO is one of the most popular algorithms for RLHF [7, 55], consisting of actor, critic, reference policy, and reward models. Each model is a Llama [73] model with sizes ranging from 7B to 70B. Safe-RLHF has an additional cost model whose architecture and size are the same as the reward model and ReMax eliminates the critic model. We use mixed precision for actor and critic training, i.e., BF16 for model parameters and FP32 for gradient and optimizer states, with Adam [38] optimizer in all experiments. BF16 is used in model inference and auto-regressive generation. If not specified, the experiment results are obtained from PPO.
>  我们运行 PPO, ReMax, Safe-RLHF 算法
>  actor, critic, reference policy, reward models 都是 Llama 模型，大小从 7B 到 70B
>  我们采用混精训练，模型参数为 BF16，梯度和优化器状态和 FP32，模型推理和生成都使用 BF16

**Baselines.** We compare HybridFlow with state-of-the-art RLHF systems including DeepSpeed-Chat [82] v0.14.0, OpenRLHF [30] v0.2.5, and NeMo-Aligner [17] v0.2.0 (detailed in Table 1). NeMo-Aligner doesn't support ReMax algorithm. We do not compare HybridFlow to other frameworks such as Trlx [27], HuggingFaceDDP [79], and Collosal-Chat [15] as they are less representative and slower than the above baselines (as reported in [82]).
>  我们将 HybridFlow 和 DeepSpeed-Chat, OpenRLHF, NeMo-Aligner 比较

We use RLHF throughput (tokens/sec) as the performance metric, computed by dividing the total number of tokens in prompts and responses in a global batch by one RLHF iteration time. All reported performance numbers are averaged over 5 training iterations after a warm-up of 10 iterations.
>  我们使用 RLHF 吞吐 (tokens/s) 作为性能度量
>  吞吐的计算方式是将一次 RLHF 迭代时间的 global batch 中 prompts, responses 的总 tokens 数量除以时间

**Datasets and hyperparameters.** We perform RLHF on "Dahoas/ful-hh-rlhf" dataset [7] of HuggingFace, which is widely used for LLM alignment [64, 85]. As the baseline systems may not incorporate continuous-batching optimization [83] during generation, for a fair comparison, we enforce the same length on all responses to be generated. In each experiment, the input prompt length and the output response length are both 1024 and the global batch size of input prompts to the actor model is 1024. The number of PPO epochs is 1 and the number of PPO update iterations per epoch is 8, aligning with previous RLHF research [31, 55, 81].
>  试验中，input prompt 长度和 output response 长度都是 1024
>  PPO epochs 数量为 1，PPO 每个 epoch 的更新迭代数量为 8

## 8.2 End-to-End performance

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/467f6266b3376dd057e2b520fe3a33cb67feb7a45b5099e852fbcd98bcc3bff9.jpg)  

Figure 9. PPO throughput. Numbers in parentheses are HybridFlow speedups compared with baselines.

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/ab74013b40d2b821e3f32609e83e131e691830e11320f2c0eeb675dbd6d35f72.jpg)  

Figure 10. ReMax throughput. Numbers in parentheses are HybridFlow speedups compared with baselines

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/5edff52b8b545b91656d0c2f0317c566340a7215ae4b2be66b7e72290cac8ba2.jpg)  

Figure 11. Safe-RLHF throughput. Numbers in the parentheses are HybridFlow speedups compared with the baselines

Figures 9, 10, and 11 show RLHF throughput when running PPO, ReMax, and Safe-RLHF respectively. The actor, critic, reference, and reward models in this set of experiments are of the same size, following previous practice [7, 55, 82]. The number of GPUs used in experiments of different model sizes ranges from the smallest number of GPUs to run RLHF without OOM to 128 GPUs. We do not enable offloading optimizer states [61] in the experiments for fair comparison. 

**Overall performance.** We observe that HybridFlow consistently outperforms the baselines across all model scales. In Figure 9 for PPO, HybridFlow outperforms DeepSpeed-Chat, OpenRLHF and NeMo-Aligner by  $3.67\times$  up to  $7.84\times$ ,  $3.25\times$  up to  $5.93\times$  and  $12.52\times$  up to  $20.57\times$ , respectively. This is mainly because HybridFlow effectively executes generation, inference, and training in all RLHF stages by sharding the models with different parallelism strategies to fit various computation workloads. HybridFlow achieves the highest average speedup of  $9.64\times$  when training 70B models, as HybridFlow reduces the transition overhead by up to  $71.2\%$  and  $89.1\%$  compared to DeepSpeed-Chat and OpenRLHF, which also incurs large inter-machine communication when training with ZeRO-3. Due to the lack of KVCache in generation engine, NeMo-Aligner's main performance bottleneck lies in the generation stage, which accounts for up to  $81.2\%$  of its RLHF iteration time. Similar results can be observed in Figures 10, 11 validating the efficiency of HybridFlow on running various RLHF algorithms.
>  HybridFlow 的吞吐优于其他系统，这是因为 HybridFlow 根据不同的阶段的 workload 特点使用不同的并行策略，进而高效执行 generation, inference, training
>  模型越大，HybridFlow 优势越明显，因为它有效降低了 transition overhead，其他的系统则存在很大的机器间通信开销

**Scalability.** HybridFlow achieves at least  $2.09\times$  speedup on 8 GPUs. With increasing GPUs, the strong scaling efficiency of HybridFlow on various model scales is  $66.8\%$ , computed by dividing $\frac {\text{throughput in largest scale}}{\text{throughput in smallest scale}}$ by $\frac{\text{max\# of GPUs}}{\text{min\# of GPUs}}$ of GPUs [5], averaging over three algorithms and all model scales. Scaling to a large number of GPUs with a fixed global batch size results in smaller local batch sizes for each worker, potentially causing GPU underutilization. Running 7B models on 128 GPUs, HybridFlow still outperforms the best baseline OpenRLHF for  $1.68\times$ ,  $1.53\times$ , and  $1.71\times$  on PPO, ReMax, and Safe-RLHF respectively. This can be attributed to HybridFlow's ability to adapt the best placement strategies for different models and cluster sizes to minimize RLHF time. OpenRLHF performs better in a larger GPU cluster but less efficiently on smaller ones.

## 8.3 Model Placement
In this experiment, we implement various model placements of the PPO algorithm in HybridFlow, under the same model and cluster settings as in Sec. 8.2: (i) colocate, the placement strategy in DeepSpeed-Chat; (ii) standalone, that in OpenRLHF and; (iii) split, NeMo-Aligner's colocation placement (actor and reference policy on the same set of devices and critic and reward model on another); (iv) hybridflow, the optimized placement obtained by Algorithm 1.
>  我们评估各种放置策略:
>  1. colocate, DeepSpeed-Chat 的放置策略
>  2. standalone, OpenRLHF 的放置策略
>  3. split, NeMo-Aligner 的放置策略 (actor, reference policy 在相同一组设备, critic, reward model 在另一组设备)
>  4. hybridflow, 算法 1 得到的放置策略

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/ea076df956308caa3bbb6833a7cf2ac8fbd6b0d923ef3d272eb47489dfd6460b.jpg)  

Figure 12. Throughput of HybridFlow under different placements

**Comparison of different model placements.** Figure 12 reveals that optimized placement of HybridFlow under different numbers of GPUs varies. From 16 to 64 GPUs, colocating all models on the same set of devices yields the best performance. For 96 to 128 GPUs with 34B models and 96 GPUs with 13B models, the split strategy becomes optimal. The split strategy divides GPUs evenly between the two sets of models, as their sizes are equal. For 13B models on 128 GPUs, the standalone strategy achieves the highest throughput. In this case, HybridFlow allocates 64 GPUs for the actor, 32 for the critic, and 16 each for the reference and reward model. 

In smaller clusters, computation of all models can fully utilize GPU resources; the colocate strategy ensures maximum GPU usage in different RLHF stages. In larger clusters, RLHF throughput under colocate placement fails to scale up linearly as the batch size is fixed and the computation-to-communication ratio decreases with a larger DP size on more GPUs. Standalone and split strategies place models on different devices with a smaller DP size for each model in larger clusters, facilitating parallel execution of different models in the same stages. In all cases, our Algorithm 1 produces the best placement with the highest training throughput.
>  在小集群中，所有模型的计算可以完全利用 GPU 资源，故 colocate 策略确保了 RLHF 不同阶段的最大 GPU 使用率
>  在大集群中，colocate 下的吞吐无法线性 scale，因为更多的 GPUs 下也就是提高 DP size，但 batch size 不能太大 (否则影响性能)，只能固定，这使得计算资源无法充分利用，并且通信开销随着 GPU 数量增大而增大
>  在大集群中，standalone, split 策略将模型放在不同的设备上，相较于 colocate 的 DP size 更小，相对更优

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/547d370abce242f7a8f8ca6353d26747f3f9b3090de7c8b59af10b769752acdc.jpg)  

Figure 13. Placement comparison under 13B actor and reference policy & 70B critic and reward model.

**Larger critic and reward model.** We further evaluate model placements when running PPO with a 13B actor and reference policy and 70B critic and reward models (larger critic and reward models are expected to produce better alignment [7]). 
>  通常更大的 critic, reward 模型来 RLHF 的效果会更优

Figure 13 shows that the colocate strategy still outperforms others by  $44.8\%$  on average with up to 64 GPUs. The split strategy achieves higher throughput with 96 GPUs. When scaling to 128 GPUs, the best placement obtained by Algorithm 1 colocates actor, reference, and reward models on 64 GPUs while allocating the remaining 64 GPUs to critic. 

On the same number of GPUs, actor and reference policy's computation time is much smaller than critic and reward model, and colocating the reward model with actor and reference policy reduces the GPU idle time in the experience preparation stage. 

In general, distributing actor and critic on different devices for parallel execution in the training stage leads to higher throughput in large clusters.
>  总的来说，将 actor 和 critic 分布在不同的设备上以在 training 阶段并行执行可以在大集群中带来更高的吞吐

## 8.4 3D-HybridEngine

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/366f6ad9919d2dbed7bb161e6bb4b04325af98f7e0e0a9ee7cb2e5512f5db4cf.jpg)  

Figure 14. Transition time between actor training and generation.

**Transition time comparison.** Figure 14 shows the transition time between actor training and generation stages on various model scales, which is the time to reshard model weights from training to generation, under the same settings in  $\S 8.2$ . OpenRLHF's transition time includes weight synchronization time between two copies of the actor model on different devices. HybridFlow reduces the transition time by  $55.2\%$  (11.7s) on average and the transition overhead by up to  $89.1\%$  (78.2s) with 70B models, while maintaining consistent overhead across different cluster scales. This is attributed to our new parallel grouping method for the generation stage ($\S$ 5.4). In baseline methods, all model parameters must be collected during transition, necessitating layer-by-layer collections multiple times to prevent OOM. HybridFlow enables zero memory redundancy during transition and requires only one all-gather operation per micro DP group.
>  actor training, generation 阶段之间的 transition 时间如 Figure 14 所示
>  OpenRLHF 需要将不同设备上的 actor model 进行权重同步
>  HybridFlow 大幅降低 transition 时间，并且随着 cluster 增大，transition 时间不会显著增大，这归功于针对 generation 阶段的新并行分组方法
>  baseline 方法需要在 transition 收集全部的模型权重，因此为了避免 OOM，需要一层一层地收集，HybridFlow 没有内存冗余，并且仅需要 micro DP group 内的一次 all-gather

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/3edeb0be9e270269c675a508d7c1975cd18365cf7326c9be6d2939524b1eae7e.jpg)  

Figure 15. Time breakdown on different generation parallel sizes of the actor model on 16 GPUs.

**Transition and generation time** We further validate the need to use different parallel sizes in actor training and generation in HybridFlow. In this experiment, all models are colocated on the same set of GPUs, and the KVCache for generation is allocated using the remaining GPU memory (i.e., best-effort allocation). 
>  我们进一步验证 actor training, generation 使用不同并行配置的必要性

Figure 15 gives the transition and generation time when running RLHF on 16 GPUs with 7B and 13B models, respectively, with training parallel groups 1-8-2 (following p-t-d convention) and varying generation TP group size  $t_g$  from 1 to 8. The generation PP group size remains constant at  $p_g = 1$  and the micro DP group size  $d_g$  is computed as  $\frac{8}{t_g}$ . We observe that applying a smaller generation TP group size,  $t_g = 2$ , for 7B models and  $t_g = 4$  for 13B models reduces the generation latency by  $60.3\%$  and  $36.4\%$  respectively. Conversely, using the same TP size as training  $(t_g = 8)$ , following the NefMo-Aligner approach, results in the largest generation latency due to GPU underutilization. Further reducing  $t_g$  fails to achieve higher speedup, as a smaller  $t_g$  necessitates maintaining a larger KVCache per GPU.
>  我们发现，在 generation 阶段使用相较于 training 阶段更小的 TP 组大小可以显著降低生成延迟
>  同时，进一步降低 TP 组大小无法获得更高的加速效果，因为这要求每个 GPU 维护更大的 KVCache

>  KVCache 是按层分布在 TP 组内的
>  假设模型有 32 层，使用 TP=4，那么每个 GPU 负责 32 层中的部分参数，例如每层的 1/4 Q, K, V 投影，因此每个 GPU 需要为它参与计算的每一层保存对应的 K, V 向量
>  训练用大 TP + 小 DP 就是为了降低显存压力，推理用小 TP + 大 DP，小 TP 增加一点显存压力，减少通讯开销，大 DP 则直接增加吞吐

## 8.5 Algorithm Runtime

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/f427a711-1a01-4272-9036-791b5f217452/57751a4210aaff4477a64a4920a3d090d876722443003f3a2a7f06b7ab8e44f4.jpg)  

Figure 16. Runtime of device mapping algorithm. The model size and # of GPUs are simultaneously scaled.

Figure 16 shows the running time of Algorithm 1, which is significantly shorter than days of actual RLHF training. A linear growth of running time is exhibited, revealing good scalability of the device mapping algorithm with model size and cluster size. Most of the running time is spent on estimating the execution latency of each model's parallel strategies. More parallelism strategies are available for a larger model, requiring more simulations to identify the optimal one for each placement plan. Our caching of optimal parallelism strategies of the models to be reapplied across different placements reduces the search time for the best placement to at most half an hour.

# 9 Discussions
**Fault Tolerance.** HybridFlow is orthogonal to existing fault-tolerance approaches [22, 34, 49, 76, 93] and already incorporates checkpointing. Failures can be detected by NCCL errors and silent-data-corruption by checksums. Our programming model enables the single controller to coordinate checkpoint operations via RPC, allowing the saving of model states within each ParallWorker Group. This includes saving parameters of actor/critic models, dataloader IDs, and Random Number Generator (RNG) states to ensure systemwide consistency. Moreover, HybridFlow can also employ redundancy-based fault-tolerance methods, such as broadcast parameters and CPU checkpoint, for fast recovery if enough healthy model replicas are available [76, 93].
>  HybridFlow 可以使用现存的容错方法
>  single controller 通过 RPC 协调 checkpoint 操作，在每个 ParallelWorker 组内存储模型状态，包括了存储 actor/critic 模型参数, datalodaer ID, 随机数生成器

**Placement Insights.** We conclude three main insights for model placement and GPU allocation in RLHF training. 1) Allocating more GPUs to the actor model can reduce the time-consuming generation latency, which cannot be parallelized with other models. 2) When each model computation can fully utilize GPU resources, colocating all the models is most effective when training on relatively small-scale clusters. 3) When scaling up to large-scale clusters (i.e., strong scaling), distributing the actor and critic models on different devices for parallel execution in the training and preparation stages would help achieve higher throughput.
>  RLHF 训练的 model placement 和 GPU 分配的思考:
>  1. 给 actor 分配更多 GPUs 可以减少 generation latency (generation 阶段无法和其他模型的阶段并行，因此有必要减少其 latency)
>  2. 当每个模型的计算都可以完全利用 GPU 资源，将所有模型 colocate 是在相对小规模集群上训练的最有效方法
>  3. 拓展到大集群上时 (strong scaling)，将 actor, critic 模型分配到不同设备，以在 perparation, training 阶段并行执行可以达到更高吞吐

**Resource multiplexing.** HybridFlow enables colocation of models on shared devices by utilizing time-sharing for GPU computation. Recent research in DNN task scheduling has developed fine-grained resource multiplexing techniques, primarily aimed at achieving the service-level objectives of individual tasks [8, 18, 26, 26, 47, 56, 77]. Although the ResourcePool implementation supports parallel execution of collocated models, HybridFlow generally adheres to sequential execution to prevent GPU resource contention or OOM issues as discussed in Section 2.3. 
>  HybridFlow 允许对 GPU 计算使用时分复用，来将模型共置在共享设备上
>  虽然 ResourcePool 实现支持共置模型的并行执行，但 HybridFlow 通常采用顺序执行的方式以避免 GPU 资源竞争和 OOM 的问题

Applying GPU sharing and heterogeneous resources in RLHF training poses distinct challenges, as it seeks to balance the computation workload and manage complex data dependencies among various tasks. Investigating fine-grained auto-mapping algorithms for GPU sharing in RLHF training, coupled with model offload optimization and integration of heterogeneous devices, would be a promising direction for future research. 
>  在 RLHF 训练中应用 GPU 共享和异构资源存在独特的挑战，因为需要在计算负载之间取得平衡，并管理不同任务之间的复杂数据依赖关系
>  因此，研究 RLHF 训练时细粒度的 GPU 共享自动映射算法，结合 model offload 优化和异构设备的集成，将是一个具有前景的方向

**From alignment to reasoning.** In RLHF for LLM alignment, the reward signal is generated by the reward model. Besides alignment tasks, similar algorithms (e.g., PPO and GRPO [70]) can be applied to other domains, such as code generation and mathematical reasoning. For these tasks, a ground truth may exist for each prompt, which can be determined by assessing the correctness of the output value for each code test case and verifying the accuracy of mathematical results. Therefore, the reward model can be replaced by non-neural-network reward modules, such as a sandbox environment [87] for evaluating generated code or a reward function [14, 65] to validate mathematical results. HybridFlow can seamlessly integrate these reward modules by wrapping them as remote functions and orchestrating their execution within the single-process script, providing a flexible and efficient framework for diverse reinforcement learning applications.
>  RLHF 中，奖励信号由 reward model 生成
>  除了对齐任务以外，类似的算法可以用于其他领域，例如代码生成和数学推理，对于这些热为奴，每个 prompt 存在 groud truth 的答案，此时奖励模型可以被一个非神经网络的模块替换，例如一个单纯的函数或一个沙盒环境

# 10 Related Work
**RL frameworks.** There have been plenty of frameworks for RL, ranging from general-purpose RL systems design for small-scale DNNs [12, 25, 28, 39, 45, 46] to RLHF systems specifically optimized for LLMs [15, 17, 30, 80, 82]. We have thoroughly examined closely related work in §2 and we discuss more RL frameworks in this section. These RL frameworks [12, 25, 28, 39, 74], similar to recent RLHF systems, use a hodgepodge of multi-controller frameworks to implement their algorithms. They establish multiple long-running distributed programs with each component coordinating the execution order with hard-coded data synchronization. Gear [74] further optimized the experience replay segment of the RL pipeline. However, all these frameworks fail to support LLM training, inference, and generation in RLHF.
>  已经有许多 RL 框架，从通用目的的，针对小规模 DNN 的 RL 系统设计到专门为 LLM 优化的 RLHF 系统
>  这些 RL 框架，类似于最近的 RLHF 系统，使用多种多样的 multi-controller 框架来实现其算法
>  他们通过多个长期运行的分布式程序来执行任务，每个组件通过硬编码的数据同步来协调执行顺序
>  Gear 进一步优化了 RL 流水线中的经验回放部分，然而所有这些库昂加都无法支持在 RLHF 中进行 LLM 的训练、推理和生成

**LLM training and serving systems.** TorchDDP [57] and Horovod [69] support data parallel training. ByteScheduler [58] and DeepSpeed [60] extend data parallelism with communication and memory optimizations. Numerous systems [23, 36, 48, 54, 71, 75, 89] optimized large model training through model parallelisms such as tensor parallelism and pipeline parallelism to partition models across devices. LLM serving systems [3, 16, 40, 72, 83, 92] also adopts data and model parallelism to accelerate auto-regressive generation with specialized optimizations like continuous-batching [83] and chunked-prefill [3]. Note that all the above frameworks adopt multi-controller paradigm for efficient computation.
>  TorchDDP, Horovod 支持数据并行训练
>  ByteScheduler, DeepSpeed 对数据并行进行了通信和内存优化
>  许多系统通过模型并行优化大规模模型训练，例如张量并行和流水线并行，将模型划分到多个设备
>  LLM 服务系统也采用数据和模型并行来加速自回归生成，并且使用了特殊的优化例如 continuous-batching, chucked-prefill
>  所有的以上框架都采用 mult-controller 范式

**Dataflow systems.** Dataflow systems like MapReduce [21], Spark [86], Dryad [33], and Naiad [51] are popular for analytics and ML workloads but they lack support for dynamic task graphs. Ray [50] unifies task-parallel and actor programming models in a single dynamic task graph and implements a scalable distributed scheduler and a global control store, which is adopted by many RL frameworks [45, 46]. Pathways [9], a closed-source project for TPUs, are designed to easily express complex parallelism patterns and fine-grain control flow within a single DNN model, such as pipeline parallelism and Mixture-of-Experts with sparse computation. It employs an asynchronous distributed dataflow design that enables parallel control plane execution despite data dependencies, reducing the dispatch overhead from single-controller paradigm. Its main focus lies on single-model training, requiring complex compilations of each sub-network of a DNN model. HybridFlow can integrate Pathways as a submodule to implement the computation of models in the RLHF dataflow.
>  数据流系统缺乏对动态任务图的支持
>  Ray 将任务并行和 actor programming 模型统一到单个动态任务图，并且实现了一个可拓展的分布式调度器和全局控制存储，被许多 RL 框架采用
>  Pathways 是一个针对 TPU 的闭源项目，设计目标是可以轻松表示单个 DNN 模型的复杂的并行模式和细粒度控制流，例如流水线并行和带有稀疏计算的 MoE
>  Pathways 采用异步的分布式数据流设计，在存在数据依赖的情况下仍然可以实现并行控制平面执行，从而减少了 single-controller 范式下的犯法开销
>  Pathways 的主要关注点是但模型训练，需要对 DNN 模型的每个子网络进行复杂的编译
>  HybridFlow 可以将 Pathways 作为子模块集成，用于实现 RLHF 数据流中的模型计算

# 11 Conclusion
HybridFlow is an RLHF framework that enables flexible representation and efficient execution of diverse RLHF algorithms. We propose a hybrid programming model that allows users to easily build RLHF dataflow in a few lines of code by encapsulating distributed computation of different LLMs into primitive APIs and hiding the complexity of data resharding among nodes. Our 3D-HybridEngine ensures efficient execution of training and generation of the actor model, with zero memory redundancy and significantly reduced communication overhead for model parameter resharding. Furthermore, our effective mapping algorithm optimizes GPU allocation and placement of models in the RLHF dataflow. Extensive experiments demonstrate that HybridFlow achieves  $1.53 \times$  to  $20.57 \times$  speedup compared to state-of-the-art RLHF systems under various model sizes and cluster scales.
>  HybridFlow 是一个 RLHF 框架，可以灵活表示和高效执行各种 RLHF 算法
>  我们提出了一个混合编程模型，允许用户使用几行代码构造 RLHF 数据流，将分布式计算封装到了 primitive APIs 中，且隐藏了节点之间 data resharding 的复杂性
>  我们提出的 3D-HybridEngine 确保了 actor model 的训练和生成的高效执行，具有零内存冗余且显著降低了模型参数 resharding 的通讯开销
>  我们的高效映射算法优化 RLHF 数据流中的模型放置和 GPU 分配

# A Primitive APIs in HybridFlow
In HybridFlow, we implemented the primitive of each model in RLHF training by inheriting the 3DParallelWorker, FSDP Worker and ZeROWorker. The functions of these model classes are designed to decouple the distributed computation code and provide fundamental operations in RLHF for the users. This primitive design is compatible with the auto-regressive generation, forward pass, backward pass, and model update operations in the existing distributed inference and training frameworks. Users can easily customize the RLHF training dataflow (by adapting the numerical computation in the provided functions) according to the algorithm's design and benefit from reusing the underlying distributed computation implementation. We illustrate the meaning and the actual computations of these APIs in Table 4.
>  我们通过继承 `3DParallelWorker, FSDPWorker, ZeROWorker` 来实现各个模型类
>  这些模型类的函数被设计为解耦分布式计算代码，并为用户提供 RLHF 所需的基础操作
>  这些模型类的设计和现有的分布式推理和训练框架中的自回归生成、前向传播、反向传播和模型更新操作兼容
>  用户可以自定义 RLHF 训练数据流 (通过修改所提供的函数中的数值计算)，并受益于底层分布式计算实现的复用

Table 4. Key functions provided in each model class. The users can use these provided functions to construct various RLHF algorithms in a few lines of code.  

<table><tr><td>Model</td><td>APIs</td><td>Computation</td><td>Interpretation</td></tr><tr><td rowspan="4">Actor</td><td>generate_sequence</td><td>auto-regressive generation</td><td>Based on a batch of prompts, the actor model generates a batch of responses and returns the log probability of each token in the responses.</td></tr><tr><td>compute_log_prob</td><td>a forward pass</td><td>The actor model computes the log probability of each token in the prompts and responses. This log probability is the same as the return log probability when performing generation using the same model precision. (Optional in PPO)</td></tr><tr><td>compute_loss</td><td>a forward pass</td><td>The actor model computes the pretrain loss based on the pertaining dataset [7, 19, 55].</td></tr><tr><td>update_actor</td><td>a forward, backward pass and model update</td><td>Based on the advantages, returns (calculated from computeadvantage) and pertaining loss, the actor model calculate the training loss and update its weights. We implement various loss for diverse RLHF algorithms including PPO [55], Safe-RLHF [19], ReMax [43], GRPO [70] and others.</td></tr><tr><td rowspan="2">Critic</td><td>compute_values</td><td>a forward pass</td><td>The critic model computes the values for each prompt and response.</td></tr><tr><td>update_critic</td><td>a forward, backward pass and model update</td><td>Based on the values and returns, the critic computes a squared-error loss to update its weights. We also implement critic loss for diverse RLHF algorithms including PPO [55], Safe-RLHF [19], ReMax [43], GRPO [70] and others.</td></tr><tr><td>Reference Policy</td><td>compute_ref_log_prob</td><td>a forward pass</td><td>The reference model computes the reference log probability of each token in the prompts and responses. This log probability is utilized as a benchmark to evaluate the divergence of the actor model and constrain its learning process.</td></tr><tr><td>Reward</td><td>compute_reward</td><td>a forward pass</td><td>The reward model conducts forward computation to calculate scores for a given set of prompts and responses. The rewards could be token level or sample-level.</td></tr><tr><td>-</td><td>computeadvantage</td><td>numerical computation</td><td>Based on the values rewards from the value model and reward model respectively, the function estimates the advantages on the given prompts and the current policy model& #x27 ;s responses. This computation involves no model forward passes.</td></tr></table>
# B Transfer Protocols

![[pics/HybridFlow-Table3.png]]

We implemented transfer protocols that cover all common use cases of data resharding between models in RLHF dataflow. Users can utilize these pre-defined protocols to generate any RLHF dataflow. Moreover, Users can easily define their own transfer protocols by implementing a collect function and a distribute function. Transfer protocols decoupled the complicated data resharding and distributed training. We denote  $p$ ,  $t$ ,  $d$  as the rank of the worker in pipeline-, tensor- and data- parallel group respectively. We illustrate these predefined protocols in Table 3.
>  我们实现了覆盖了 RLHF 数据流中 data resharding 中所有常用用例的传输协议，供用户使用
>  用户可以自定义一个 collect 函数和一个 distribute 函数来实现自己的传输协议

# C Auto-Parallelism Algorithm
Algorithm 2 outlines the search process of the optimal parallelism strategy of each model. Starting from the minimal model parallelism size of each model (to prevent OOM when colocating with multiple workers), we enumerate all feasible parallel configurations based on the number of GPUs and the number of GPUs per machine  $U$  . The default number of  $U$  is set to 8. We use simu module to estimate the latency of each model based on their workload. This module includes three simulators for training, inference, and generation workload, all are analytical models following previous research [42, 84, 92]. The training and inference workload is compute-bound while the generation workload is memory-bound. For the actor model, we first find the parallelism strategy for training and record the memory usage in the training stage. During actor generation, KVCache requirements are calculated using the batch size and max sequence length. If the model-parallel size for the generation stage cannot accommodate both parameters and KVCache, we increase it. Then, we seek the optimal strategy with corresponding KVCache allocation by comparing the latency estimation. Developing a comprehensive autoregressive generation simulator that accounts for variable KVCache sizes could further enhance the auto-mapping process in RLHF research.
>  算法 2 从最小的模型并行 size 开始搜索
>  `simu` 被用于估计每个模型的延迟
>  **training, inference workload 为 compute-bound，而 generation workload 则是 memory-bound**
>  我们先为 actor model 找到训练的并行策略，并记录训练阶段的内存使用，在 actor 生成阶段，KVCache 需求则使用 batch size 和 max seq length 计算
>  如果 model-parallel size 无法容纳生成阶段的模型参数和 KVCache，我们就增大它

