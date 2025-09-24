```
Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan, and Ion Stoica, UC Berkeley
```

# Abstract
The next generation of AI applications will continuously interact with the environment and learn from these interactions. These applications impose new and demanding systems requirements, both in terms of performance and flexibility. In this paper, we consider these requirements and present Ray-a distributed system to address them. Ray implements a unified interface that can express both task-parallel and actor-based computations, supported by a single dynamic execution engine. To meet the performance requirements, Ray employs a distributed scheduler and a distributed and fault-tolerant store to manage the system's control state. In our experiments, we demonstrate scaling beyond 1.8 million tasks per second and better performance than existing specialized systems for several challenging reinforcement learning applications.
>  Ray 实现了一个统一的接口，可以表示任务并行和基于 actor 的计算，该接口由一个动态执行引擎支撑
>  为了满足性能需求，Ray 使用一个分布式调度器和一个分布式容错存储来管理系统的控制状态

# 1 Introduction
Over the past two decades, many organizations have been collecting-and aiming to exploit-ever-growing quantities of data. This has led to the development of a plethora of frameworks for distributed data analysis, including batch [20, 64, 28], streaming [15, 39, 31], and graph [34, 35, 24] processing systems. The success of these frameworks has made it possible for organizations to analyze large data sets as a core part of their business or scientific strategy, and has ushered in the age of "Big Data."

More recently, the scope of data-focused applications has expanded to encompass more complex artificial intelligence (AI) or machine learning (ML) techniques [30]. The paradigm case is that of supervised learning, where data points are accompanied by labels, and where the workhorse technology for mapping data points to labels is provided by deep neural networks. The complexity of these deep networks has led to another flurry of frameworks that focus on the training of deep neural networks and their use in prediction. These frameworks often leverage specialized hardware (e.g., GPUs and TPUs), with the goal of reducing training time in a back setting. Examples include TensorFlow [7], MXNet [18], and PyTorch [46].

The promise of AI is, however, far broader than classical supervised learning. Emerging AI applications must increasingly operate in dynamic environments, react to changes in the environment, and take sequences of actions to accomplish long-term goals [8, 43]. They must aim not only to exploit the data gathered, but also to explore the space of possible actions. These broader requirements are naturally framed within the paradigm of reinforcement learning (RL). RL deals with learning to operate continuously within an uncertain environment based on delayed and limited feedback [56]. RL-based systems have already yielded remarkable results, such as Google's AlphaGo beating a human world champion [54], and are beginning to find their way into dialogue systems, UAVs [42], and robotic manipulation [25, 60].
>  RL 不仅需要利用收集到的数据，还需要探索可能的动作空间
>  RL 致力于在不确定的环境中，基于延迟的或有限的反馈，持续学习如何操作

The central goal of an RL application is to learn a policy - a mapping from the state of the environment to a choice of action - that yields effective performance over time, e.g., winning a game or piloting a drone. Finding effective policies in large-scale applications requires three main capabilities. First, RL methods often rely on simulation to evaluate policies. Simulations make it possible to explore many different choices of action sequences and to learn about the long-term consequences of those choices. Second, like their supervised learning counterparts, RL algorithms need to perform distributed training to improve the policy based on data generated through simulations or interactions with the physical environment. Third, policies are intended to provide solutions to control problems, and thus it is necessary to serve the policy in interactive closed-loop and open-loop control scenarios.
>  RL 的核心目标是学习策略 - 从环境中的状态到动作选择的映射 - 该策略能够在长时间下提供高效的表现
>  在大规模应用中寻找有效的策略，需要具备三项关键能力:
>  1. RL 方法依赖于模拟来评估策略，模拟使得我们可以探索许多不同的动作序列，并了解这些选择的长期后果
>  2. 和监督学习方法类似，RL 算法需要支持分布式训练，以基于模拟生成或与物理环境交互所产生的数据来优化策略
>  3. 策略旨在解决控制问题，因此需要能在开环和闭环的控制场景中进行部署和推理

>  open-loop 指开环控制/预编程控制，其定义是策略在一开始就规划好整个动作序列，在执行时不依赖或很少依赖状态反馈
>  例如根据初始状态直接预测好完整的动作轨迹，机器人就按照预定轨迹移动，即时遇到障碍也不调整 (直到重新规划)
>  close-loop 指闭环控制/状态反馈控制，其定义是策略在每一步都基于当前观测到的状态来选择动作，这种策略形式在 RL 中最常见

These characteristics drive new systems requirements: a system for RL must support fine-grained computations (e.g., rendering actions in milliseconds when interacting with the real world, and performing vast numbers of sim ulations), must support heterogeneity both in time (e.g., a simulation may take milliseconds or hours) and in resource usage (e.g., GPUs for training and CPUs for simulations), and must support dynamic execution, as results of simulations or interactions with the environment can change future computations. Thus, we need a dynamic computation framework that handles millions of heterogeneous tasks per second at millisecond-level latencies.
>  面向 RL 的系统必须支持细粒度计算 (例如在和现实时间交互时，在毫秒内渲染动作，并执行大量的模拟)，必须支持时间和资源使用上的异构性 (例如模拟可以需要数毫秒也可以需要数小时；例如 GPU 做训练，CPU 做模拟)，必须支持动态执行，因为模拟或和环境交互的结构会改变未来的运算
>  因此，我们需要一个能够每秒动态处理百万个任务，提供毫秒级延迟的动态计算框架

>  模拟就是环境动态，即接受状态和动作，给出状态转移

Existing frameworks that have been developed for Big Data workloads or for supervised learning workloads fall short of satisfying these new requirements for RL. Bulk-synchronous parallel systems such as MapReduce [20], Apache Spark [64], and Dryad [28] do not support fine-grained simulation or policy serving. Task-parallel systems such as CIEL [40] and Dask [48] provide little support for distributed training and serving. The same is true for streaming systems such as Naiad [39] and Storm [31]. Distributed deep-learning frameworks such as TensorFlow [7] and MXNet [18] do not naturally support simulation and serving. Finally, model-serving systems such as TensorFlow Serving [6] and Clipper [19] support neither training nor simulation.
>  批量同步并行系统不支持细粒度模拟或策略服务
>  任务并行系统在分布式训练或服务方面支持有限
>  分布式 DL 框架不天然支持模拟和策略服务
>  模型服务系统则既不支持训练也不支持模拟

While in principle one could develop an end-to-end solution by stitching together several existing systems (e.g., Horovod [53] for distributed training, Clipper [19] for serving, and CIEL [40] for simulation), in practice this approach is untenable due to the tight coupling of these components within applications. As a result, researchers and practitioners today build one-off systems for specialized RL applications [58, 41, 54, 44, 49, 5]. This approach imposes a massive systems engineering burden on the development of distributed applications by essentially pushing standard systems challenges like scheduling, fault tolerance, and data movement onto each application.
>  原则上，可以通过组合多个现有系统来构建一个端到端的解决方案，但在实际应用中则不可行，因为 RL 应用的各个组件高度耦合

In this paper, we propose Ray, a general-purpose cluster-computing framework that enables simulation, training, and serving for RL applications. The requirements of these workloads range from lightweight and stateless computations, such as for simulation, to long-running and stateful computations, such as for training. To satisfy these requirements, Ray implements a unified interface that can express both task-parallel and actor-based computations. Tasks enable Ray to efficiently and dynamically load balance simulations, process large inputs and state spaces (e.g., images, video), and recover from failures. In contrast, actors enable Ray to efficiently support stateful computations, such as model training, and expose shared mutable state to clients, (e.g., a parameter server). Ray implements the actor and the task abstractions on top of a single dynamic execution engine that is highly scalable and fault tolerant.
>  Ray 是一个通用目的的集群计算框架，为 RL 应用提供模拟、训练、服务支持
>  这些 workloads 的要求差异很大: 从轻量、无状态计算，例如模拟，到长时间运行、有状态的计算，例如训练
>  为了满足这些要求，Ray 实现了一个能够同时表示 task-parallel 和 actor-based 计算的统一接口
>  Task: 使 Ray 可以高效且动态地对模拟进行负载均衡，处理大规模输入和状态空间，并具备故障恢复能力
>  Actor: 使 Ray 可以支持有状态计算，例如模型训练，并向客户端暴露可变的共享状态 (例如参数服务器)
>  Ray 在一个可拓展且容错的动态执行引擎上实现了 actor 和 task 抽象

>  实际上模拟也是训练的一部分，如果细粒度地分，这里的模拟是指训练过程中的环境动态，环境动态给出的下一个状态会交给价值函数，用于计算策略梯度，这里的训练是指训练过程中的策略梯度下降
>  模拟是无状态计算，给定输入动作和输入状态，给出输出状态即可，不需要维护什么状态，Ray 的 task-parallel 计算就是针对模拟
>  训练是有状态计算，需要维护 policy/actor 的梯度状态，Ray 的 actor-based 计算就是针对训练

To meet the performance requirements, Ray distributes two components that are typically centralized in existing frameworks [64, 28, 40]: (1) the task scheduler and (2) a metadata store which maintains the computation lineage and a directory for data objects. This allows Ray to schedule millions of tasks per second with millisecond-level latencies. Furthermore, Ray provides lineage-based fault tolerance for tasks and actors, and replication-based fault tolerance for the metadata store.
>  为了满足性能需求，Ray 将两个通常在现有框架中中心化的组件进行了分布式:
>  1. task scheduler 2. metadata store (维护计算血缘关系并管理数据对象目录)
>  这使得 Ray 可以每秒在毫秒级延迟下调度百万个任务
>  此外，Ray 为 tasks, actors 提供了基于血缘的容错，为元数据存储提供了基于复制的容错

>  computation lineage: 指一个计算结果的生成过程所依赖的所有上游输入、操作和中间步骤的历史记录

While Ray supports serving, training, and simulation in the context of RL applications, this does not mean that it should be viewed as a replacement for systems that provide solutions for these workloads in other contexts. In particular, Ray does not aim to substitute for serving systems like Clipper [19] and TensorFlow Serving [6], as these systems address a broader set of challenges in deploying models, including model management, testing, and model composition. Similarly, despite its flexibility, Ray is not a substitute for generic data-parallel frameworks, such as Spark [64], as it currently lacks the rich functionality and APIs (e.g., straggler mitigation, query optimization) that these frameworks provide.
>  虽然 Ray 支持 RL 应用背景下的服务、训练和模拟，但不应该被视为在其他应用背景下专门解决特定 workload 的替代

We make the following **contributions**:

- We design and build the first distributed framework that unifies training, simulation, and serving -- necessary components of emerging RL applications. 
- To support these workloads, we unify the actor and task-parallel abstractions on top of a dynamic task execution engine. 
- To achieve scalability and fault tolerance, we propose a system design principle in which control state is stored in a sharded metadata store and all other system components are stateless. 
- To achieve scalability, we propose a bottom-up distributed scheduling strategy.

>  本文的贡献如下:
>  - 设计并构建了第一个统一了训练、模拟、服务的分布式框架
>  - 在动态任务执行引擎上统一了 actor, task-parallel 抽象
>  - 为了解决可拓展性和容错，我们提出了一个系统设计原则，其中控制状态存储在 sharded metadata store，同时所有其他系统组件无状态
>  - 为了达成可拓展性，我们提出了自底向上分布式调度策略

# 2 Motivation and Requirements

![[pics/Ray-Fig1.png]]

We begin by considering the basic components of an RL system and fleshing out the key requirements for Ray. As shown in Figure 1, in an RL setting, an agent interacts repeatedly with the environment. The goal of the agent is to learn a policy that maximizes a reward. A policy is a mapping from the state of the environment to a choice of action. The precise definitions of environment, agent, state, action, and reward are application-specific.

To learn a policy, an agent typically employs a two-step process: (1) policy evaluation and (2) policy improvement. To evaluate the policy, the agent interacts with the environment (e.g., with a simulation of the environment) to generate trajectories, where a trajectory consists of a sequence of (state, reward) tuples produced by the current policy. Then, the agent uses these trajectories to improve the policy; i.e., to update the policy in the direction of the gradient that maximizes the reward. 
>  要学习策略，agent 采用两步过程: 1. policy evaluation 2. policy improvement
>  为了评估策略，agent 和环境交互，生成轨迹，轨迹包含了当前策略生成的 (state, reward) 序列，然后 agent 使用该序列来提升策略，也就是朝着最大化奖励的梯度方向更新策略

>  RL 的两大基础方法 1. Value Iteration 2. Policy Iteration
>  这里考虑的就是 Policy Iteration，所有策略梯度方法都归于这个框架，总体来说它就是两步迭代 1. policy evaluation 2. policy improvement

![[pics/Ray-Fig2.png]]

Figure 2 shows an example of the pseudocode used by an agent to learn a policy. This pseudocode evaluates the policy by invoking rollout(environment, policy) to generate trajectories. train_policy() then uses these trajectories to improve the current policy via policy.update(trajectories). This process repeats until the policy converges.
>  agent 学习策略的伪代码如 Fig2 所示
>  agent 通过调用 `rollout(environment, policy)` 来评估策略，生成轨迹；然后 `train_policy()` 使用这些轨迹，通过 `policy.update(trajectories)` 提升当前策略

Thus, a framework for RL applications must provide efficient support for training, serving, and simulation (Figure 1). Next, we briefly describe these workloads.
>  针对 RL 应用的框架必须提供对 training, serving, simulation 的支持
>  我们简要描述这些 workloads

*Training* typically involves running stochastic gradient descent (SGD), often in a distributed setting, to update the policy. Distributed SGD typically relies on an allreduce aggregation step or a parameter server [32].
>  Training 涉及运行随机梯度下降，且通常是分布式的，来更新策略
>  分布式 SGD 通常依赖于一个 allreduce 聚合步或者一个参数服务器

*Serving* uses the trained policy to render an action based on the current state of the environment. A serving system aims to minimize latency, and maximize the number of decisions per second. To scale, load is typically balanced across multiple nodes serving the policy.
>  Serving 使用训练好的策略，基于当前状态给出动作
>  服务系统的目标是最小化延迟，并最大化每秒做出的决策数量
>  进行拓展时，workload 会在执行服务的多个节点上均衡

Finally, most existing RL applications use simulations to evaluate the policy - current RL algorithms are not sample-efficient enough to rely solely on data obtained from interactions with the physical world. These simulations vary widely in complexity. They might take a few ms (e.g., simulate a move in a chess game) to minutes (e.g., simulate a realistic environment for a self-driving car).
>  Simulation: 大多数现存 RL 应用使用模拟来评估策略 - 目前的 RL 算法没有样本高效到可以仅依赖于从物理世界交互得到的样本进行训练
>  不同的模拟的复杂性差异很大，可能有的消耗几毫秒 (例如模拟棋盘上的移动)，有的消耗几分钟 (模拟汽车在真实环境的行驶)

In contrast with supervised learning, in which training and serving can be handled separately by different systems, in RL all three of these workloads are tightly coupled in a single application, with stringent latency requirements between them. Currently, no framework supports this coupling of workloads. In theory, multiple specialized frameworks could be stitched together to provide the overall capabilities, but in practice, the resulting data movement and latency between systems is prohibitive in the context of RL. As a result, researchers and practitioners have been building their own one-off systems.
>  有监督学习下，训练和服务可以在不同的系统上处理，但 RL 中，这三类 workloads 都在单个应用中紧密绑定，彼此之间具有严格的延迟要求
>  目前没有框架支持这种耦合的 workload
>  理论上，可以组合多个专用框架实现，但实际中，这种系统间的数据移动和延迟难以满足 RL 场景下的需求

This state of affairs calls for the development of new distributed frameworks for RL that can efficiently support training, serving, and simulation. In particular, such a framework should satisfy the following requirements:
>  因此我们需要一个新的分布式 RL 框架，高效支持训练、服务和模拟
>  具体地说，这样的框架应该满足以下的要求:

*Fine-grained, heterogeneous computations*. The duration of a computation can range from milliseconds (e.g., taking an action) to hours (e.g., training a complex policy). Additionally, training often requires heterogeneous hardware (e.g., CPUs, GPUs, or TPUs).
>  细粒度、异构计算: 计算的时间可以从数毫秒 (执行一个动作) 到数小时 (训练一个复杂的策略)，此外，训练通常也需要异构硬件

*Flexible computation model*. RL applications require both stateless and stateful computations. Stateless computations can be executed on any node in the system, which makes it easy to achieve load balancing and movement of computation to data, if needed. Thus stateless computations are a good fit for fine-grained simulation and data processing, such as extracting features from images or videos. In contrast stateful computations are a good fit for implementing parameter servers, performing repeated computation on GPU-backed data, or running third-party simulators that do not expose their state.
>  灵活的计算模型: RL 应用同时需要无状态计算和有状态计算
>  无状态计算可以在系统中任意节点上执行，这使得实现负载均衡和在必要时将计算迁移到数据所在处十分容易，因此无状态计算非常是和细粒度模拟和数据处理任务，例如从图像或视频中提取特征
>  相较之下，有状态计算则更适合实现参数服务器，在基于 GPU 的数据上执行反复的计算，或者运行那些不暴露它们状态的第三方模拟器

*Dynamic execution*. Several components of RL applications require dynamic execution, as the order in which computations finish is not always known in advance (e.g., the order in which simulations finish), and the results of a computation can determine future computations (e.g., the results of a simulation will determine whether we need to perform more simulations).
>  动态执行: RL 应用的多个组件都要求动态执行，因为计算完成的顺序往往无法预先知道 (例如不同模拟任务的完成顺序不确定)，以及计算的结果会决定未来计算 (例如模拟的结果会决定我们是否需要执行更多模拟，即是否达到终止状态)

We make two final comments. First, to achieve high utilization in large clusters, such a framework must handle millions of tasks per second. (Assume 5ms single-core tasks and a cluster of 200 32-core nodes. This cluster can run (1s/5ms)×32×200 = 1.28M tasks/sec) Second, such a framework is not intended for implementing deep neural networks or complex simulators from scratch. Instead, it should enable seamless integration with existing simulators [13, 11, 59] and deep learning frameworks [7, 18, 46, 29].
>  我们再做两点补充说明
>  首先，为了在大型集群达成更高的利用率，这样的框架必须能够每秒处理数百万任务
>  其次，这样的框架并非用于从零构建深度网络或复杂的模拟器，相反，它应该支持和现有的模拟器和深度学习框架无缝集成

# 3 Programming and Computation Model
Ray implements a dynamic task graph computation model, i.e., it models an application as a graph of dependent tasks that evolves during execution. On top of this model, Ray provides both an actor and a task-parallel programming abstraction. This unification differentiates Ray from related systems like CIEL, which only provides a task-parallel abstraction, and from Orleans [14] or Akka [1], which primarily provide an actor abstraction.
>  Ray 实现了一个动态任务图计算模型，它将应用建模为一个由互相依赖的任务构成的图，这些任务会在执行过程中不断演化 (任务图即任务构成的图)
>  在这个模型之上，Ray 提供了 actor 和 task-parallel 编程抽象
>  这个抽象的统一使得 Ray 有别于其他系统，例如 CIEL 仅提供 task-parallel 抽象，而 Orleans, Akka 主要提供 actor 抽象

## 3.1 Programming Model

![[pics/Ray-Table1.png]]

**Tasks.** A task represents the execution of a remote function on a stateless worker. When a remote function is invoked, a future representing the result of the task is returned immediately. Futures can be retrieved using ray.get() and passed as arguments into other remote functions without waiting for their result. This allows the user to express parallelism while capturing data dependencies. Table 1 shows Ray's API.
>  Task : 一个 task 表示一个 remote 函数在无状态 worker 上的执行，当一个 remote 函数被调用时，表示该 task 的结果的 future 变量会立刻被返回
>  future 变量的值可以使用 `ray.get()` 获取
>  用户也可以将 future 传递给其他远程函数，无需等待结果就绪，这允许用户在表达并行性的同时也确立数据依赖关系

Remote functions operate on immutable objects and are expected to be stateless and side-effect free: their outputs are determined solely by their inputs. This implies idempotence, which simplifies fault tolerance through function re-execution on failure.
>  远程函数作用于不可变对象，应保持无状态且无副作用: 其输出完全由输入决定
>  这意味着函数具有幂等性，从而在发生错误时直接重新执行函数来简化容错处理

**Actors.** An actor represents a stateful computation. Each actor exposes methods that can be invoked remotely and are executed serially. A method execution is similar to a task, in that it executes remotely and returns a future, but differs in that it executes on a stateful worker. A handle to an actor can be passed to other actors or tasks, making it possible for them to invoke methods on that actor.
>  Actor: 一个 actor 表示一个有状态的计算
>  每个 actor 都暴露了一系列可以远程调用的方法，这些方法会被顺序执行
>  一个方法的执行类似于 task，即也是远程执行，并返回 future，差异在于它是在一个有状态的 worker 上执行
>  actor 的句柄可以传递给其他 actors 或 tasks，从而使它们可以调用该 actor 上的方法

Table 2: Tasks vs. actors tradeoffs.  

| Tasks (stateless)               | Actors (stateful)              |
| ------------------------------- | ------------------------------ |
| Fine-grained load balancing     | Coarse-grained load balancing  |
| Support for object locality     | Poor locality support          |
| High overhead for small updates | Low overhead for small updates |
| Efficient failure handling      | Overhead from checkpointing    |

Table 2 summarizes the properties of tasks and actors. Tasks enable fine-grained load balancing through leveraging load-aware scheduling at task granularity, input data locality, as each task can be scheduled on the node storing its inputs, and low recovery overhead, as there is no need to checkpoint and recover intermediate state. In contrast, actors provide much more efficient fine-grained updates, as these updates are performed on internal rather than external state, which typically requires serialization and deserialization. For example, actors can be used to implement parameter servers [32] and GPU-based iterative computations (e.g., training). In addition, actors can be used to wrap third-party simulators and other opaque handles that are hard to serialize.
>  Tasks 通过在 task 粒度上采用负载感知的调度、输入数据局部性 (task 可以被调度到存储其输入数据的节点上)、低恢复开销 (无需检查点和恢复中间状态，出错了直接重新调度)，实现了细粒度的负载均衡
>  相较之下，actors 提供了更高效的细粒度更新，因为这些更新直接作用于内部状态，而非外部状态 - 后者通常需要额外的序列化和反序列化操作
>  例如，actor 可以被用于实现参数服务器和基于 GPU 的迭代计算 (例如训练)
>  此外，actors 可以被用于封装第三方模拟器和其他难以序列化的不透明句柄

>  Task 和 Actor 抽象还是比较 low-level 的，它们是在 RL 的 context 之下，就是纯粹地表示无状态计算和有状态计算的抽象，使用者需要将 Task 和 Actor 映射到 RL 的 context

To satisfy the requirements for heterogeneity and flexibility (Section 2), we augment the API in three ways. First, to handle concurrent tasks with heterogeneous durations, we introduce ray.wait(), which waits for the first  $k$  available results, instead of waiting for all results like ray.get(). Second, to handle resource-heterogeneous tasks, we enable developers to specify resource requirements so that the Ray scheduler can efficiently manage resources. Third, to improve flexibility, we enable nested remote functions, meaning that remote functions can invoke other remote functions. This is also critical for achieving high scalability (Section 4), as it enables multiple processes to invoke remote functions in a distributed fashion.
>  为了满足异构型和灵活性的要求，我们以三种方式增强了 API
>  首先，为了解决并发执行且处理时间各异的任务，我们引入了 `ray.wait()`，它会等待前 $k$ 个可用的结果，而不是像 `ray.get()` 等待全部的结果
>  第二，为了应对资源异构的任务，我们允许开发者指定资源需求，便于 Ray scheduler 高效管理资源
>  第三，为了提高灵活性，我们支持嵌套远程函数，即远程函数可以调用其他远程函数，这对于实现高可拓展性也至关重要，因为它允许多个进程以分布式方式调用远程函数

## 3.2 Computation Model

```python
@ray.remote 
def create_policy():
    # Initialize the policy randomly.
    return policy

@ray.remote(num_gpus=1)
class Simulator(object):
    def __init__(self):
        # Initialize the environment.
        self.env = Environment()
    def rollout(self, policy, num_steps):
        observations = []
        observation = self.env.current_state()
        for _ in range(num_steps):
            action = policy(observation)
            observation = self.env.step(action) 
            observations.append(observation)
        return observations

@ray.remote(num_gpus=2)
def update_policy(policy, *rollouts):
    # Update the policy.
    return policy

@ray.remote
def train_policy():
    # Create a policy.
    policy_id = create_policy.remote()
    # Create 10 actors.
    simulators = [Simulator.remote() for _ in range(10)]
    # Do 100 steps of training.
    for _ in range(100):
        # Perform one rollout on each actor.
        rollout_ids = [s.rollout.remote(policy_id) for s in simulators]
        # Update the policy with the rollouts.
        policy_id = update_policy.remote(policy_id, *rollout_ids) 
    return ray.get(policy_id)
```

Figure 3: Python code implementing the example in Figure 2 in Ray. Note that  `@ray.remote` indicates remote functions and actors. Invocations of remote functions and actor methods return futures, which can be passed to subsequent remote functions or actor methods to encode task dependencies. Each actor has an environment object self.env shared between all of its methods.
>  `@ray.remote` 表示远程函数和 actors
>  对远程函数和 actor 方法的调用返回 futures, futures 可以被传递给后续的远程函数或 actor 方法，表达任务间的依赖关系
>  每个 actor 都有一个环境对象 `self.env`，由它的所有方法共享

![[pics/Ray-Fig4.png]]

Ray employs a dynamic task graph computation model [21], in which the execution of both remote functions and actor methods is automatically triggered by the system when their inputs become available. In this section, we describe how the computation graph (Figure 4) is constructed from a user program (Figure 3). This program uses the API in Table 1 to implement the pseudocode from Figure 2.
>  Ray 采用了动态任务图计算模型，其中系统会在远程函数和 actor 方法的输入可用的时候自动调用它们
>  本节描述 Ray 如何根据用户程序构造该计算图 (Fig4)，Fig3 中的程序使用了 Table1 中的 API，实现了 Fig2 中的伪代码

>  所谓计算图，实际上就是由 task, actor methods 之间的 future 依赖而构造出的数据流图
>  静态的依赖关系应该通过静态程序分析就可以得到了，但是每个任务的实际执行时间需要实际执行才能确定，因此图的实际数据流动是在运行时决定的

>  所谓动态，也就是 PyTorch 的 define-by-run 概念，Python 的字节码执行时，再发送出执行调用，而不是用一堆图 API 构造好图再把整张图交给运行时
>  这样好处一是 API 风格简单，Python 化，另一个是可以任意嵌入 Python 控制流和对各种 Python 库的调用，还有一个好处是实现起来也比较方便
>  坏处就是缺失了对计算图的编译优化机会

>  分布式背景下，对计算图的优化应该是可以获得不少性能便利的，主要应该是在通讯和计算的重叠方面 (但是这方面的优化也需要对图中的计算和通讯操作所耗时间的启发式知识)

Ignoring actors first, there are two types of nodes in a computation graph: data objects and remote function invocations, or tasks. There are also two types of edges: data edges and control edges. Data edges capture the de pendencies between data objects and tasks. More precisely, if data object  $D$  is an output of task  $T$  we add a data edge from  $T$  to  $D$  .Similarly, if  $D$  is an input to  $T$  we add a data edge from  $D$  to  $T$  . Control edges capture the computation dependencies that result from nested remote functions (Section 3.1): if task  $T_{1}$  invokes task  $T_{2}$  then we add a control edge from  $T_{1}$  to  $T_{2}$
>  忽略 actors，我们可以看到计算图中有两类节点: 数据对象和远程函数调用 (tasks)，也有两类边: 数据边和控制边
>  数据边捕获数据对象和 task 之间的依赖，具体地说，如果数据对象 $D$ 是任务 $T$ 的输出，我们就添加一个 $T$ 到 $D$ 的数据边，类似地，如果 $D$ 是任务 $T$ 的输入边，我们就添加一个 $D$ 到 $T$ 的数据边
>  控制边则捕获来自嵌套远程函数 (嵌套任务) 中的结果的计算依赖: 如果任务 $T_1$ 调用了任务 $T_2$，则我们添加一个 $T_1$ 到 $T_2$ 的控制边

Actor method invocations are also represented as nodes in the computation graph. They are identical to tasks with one key difference. To capture the state dependency across subsequent method invocations on the same actor, we add a third type of edge: a stateful edge. If method  $M_{j}$  is called right after method  $M_{i}$  on the same actor, then we add a stateful edge from  $M_{i}$  to  $M_{j}$  .Thus, all methods invoked on the same actor object form a chain that is connected by stateful edges (Figure 4). This chain captures the order in which these methods were invoked.
>  actor 方法调用也表示为计算图中的节点，它们基本和任务相同，只有一个关键的差异: 为了捕获相同 actor 上后续方法调用之间的状态依赖，我们添加了第三类边 - 状态边
>  如果相同 actor 上，方法 $M_j$ 是在方法 $M_i$ 之后调用的，我们就添加一条 $M_i$ 到 $M_j$ 的状态边

Stateful edges help us embed actors in an otherwise stateless task graph, as they capture the implicit data dependency between successive method invocations sharing the internal state of an actor. Stateful edges also enable us to maintain lineage. As in other dataflow systems [64], we track data lineage to enable reconstruction. By explicitly including stateful edges in the lineage graph, we can easily reconstruct lost data, whether produced by remote functions or actor methods (Section 4.2.3).
>  状态边帮助我们将 actors 嵌入到无状态的任务图中，因为状态边捕获了 actor 内共享内部状态的连续方法调用的隐式数据依赖
>  状态边也使得我们可以维护血缘性，和其他数据流系统类似，我们追踪数据血缘来支持数据重建
>  通过在血缘图中显式包含状态边，我们可以轻松重构丢失的数据，无论该数据是由远程函数构建还是 actor 方法构建

# 4 Architecture
Ray's architecture comprises (1) an application layer implementing the API, and (2) a system layer providing high scalability and fault tolerance.
>  Ray 的架构包含: 1. 实现了 API 的应用层 2. 提供高可拓展性和容错的系统层

## 4.1 Application Layer
The application layer consists of three types of processes:

- *Driver*: A process executing the user program.
- *Worker*: A stateless process that executes tasks (remote functions) invoked by a driver or another worker. Workers are started automatically and assigned tasks by the system layer. When a remote function is declared, the function is automatically published to all workers. A worker executes tasks serially, with no local state maintained across tasks.
- *Actor*: A stateful process that executes, when invoked, only the methods it exposes. Unlike a worker, an actor is explicitly instantiated by a worker or a driver. Like workers, actors execute methods serially, except that each method depends on the state resulting from the previous method execution.

>  应用层由三类进程组成:
>  - 驱动程序: 执行用户程序的进程
>  - 工作进程: 一个执行任务 (远程函数) 的**无状态的进程**，它由 driver 或者其他 worker 调用；工作进程会由系统层自动启动并分配任务，当一个远程函数被声明，该函数会自动被推送到所有工作进程；worker 顺序执行任务，任务之间不会维护局部状态
>  - 参与者: 一个**有状态的进程**，当被调用时，仅执行它暴露的方法；和 worker 不同，actor 需要被 worker 或者 driver 显式地实例化；和 worker 类似的是，actor 顺序执行方法，只不过每个方法所依赖的状态都和之前的方法执行有关

## 4.2 System Layer
The system layer consists of three major components: a global control store, a distributed scheduler, and a distributed object store. All components are horizontally scalable and fault-tolerant.
>  系统层包含三个主要组件: 全局控制存储、分布式调度器、分布式对象存储
>  这些组件均具有水平拓展能力和容错性

### 4.2.1 Global Control Store (GCS)
The global control store (GCS) maintains the entire control state of the system, and it is a unique feature of our design. At its core, GCS is a key-value store with pub-sub functionality. We use sharding to achieve scale, and per-shard chain replication [61] to provide fault tolerance. 
>  全局控制存储维护系统的整个控制状态，这也是我们设计的独特特性
>  本质上，GCS 是一个具备发布-订阅功能的键值存储
>  我们使用 sharding 来达成可拓展性，并使用 per-shard chain replication 来提供容错

The primary reason for the GCS and its design is to maintain fault tolerance and low latency for a system that can dynamically spawn millions of tasks per second.
>  GCS 及其设计的主要目的就是在确保系统能够每秒创建数百万个任务的情况下，保持容错和低延迟

Fault tolerance in case of node failure requires a solution to maintain lineage information. Existing lineage-based solutions [64, 63, 40, 28] focus on coarse-grained parallelism and can therefore use a single node (e.g., master, driver) to store the lineage without impacting performance. However, this design is not scalable for a fine-grained and dynamic workload like simulation. Therefore, we decouple the durable lineage storage from the other system components, allowing each to scale independently.
>  对于节点故障时的容错，需要一种维护血缘信息的解决方案
>  现存的基于血缘的方法聚焦于粗粒度的并行，因此可以使用单个节点 (例如 master, driver) 来存储血缘关系，也不会影响性能
>  但这种设计无法适应 simulation 这种细粒度且动态的 workload，因此，我们将持久化的血缘存储和其他系统组件解耦，使它们能够独立拓展

>  粗粒度并行即并行的任务规模较大，数量较少
>  例如有 10 个大任务，每个任务耗时几个小时，分布到 10 台机器上并行，其特点是任务少，粒度大，通信频率低
>  此时，血缘信息量不大 (10 个任务对应 10 条血缘信息)，且更新频率低，完全可以由一个中心节点维护

>  将血缘存储和系统组件解耦就是使用了单独的系统来长期保持血缘信息，它能够承受大量写入，具有高可用性

Maintaining low latency requires minimizing overheads in task scheduling, which involves choosing where to execute, and subsequently task dispatch, which involves retrieving remote inputs from other nodes. Many existing dataflow systems [64, 40, 48] couple these by storing object locations and sizes in a centralized scheduler, a natural design when the scheduler is not a bottleneck. However, the scale and granularity that Ray targets requires keeping the centralized scheduler off the critical path. Involving the scheduler in each object transfer is prohibitively expensive for primitives important to distributed training like allreduce, which is both communication-intensive and latency-sensitive. Therefore, we store the object metadata in the GCS rather than in the scheduler, fully decoupling task dispatch from task scheduling.
>  维护低延迟要求最小化任务调度时的开销，任务调度涉及了选择在哪里执行任务以及随后的任务分发，任务分发涉及了从其他节点获取远程输入
>  许多现存数据流系统通过将对象位置和大小存储在中心的调度器来耦合这两个过程，这在调度器不是性能瓶颈是是自然的设计方式
>  Ray 针对的规模和粒度则要求中心化调度器不能在关键路径上，让每次对象传输都涉及中心化调度器，对于分布式训练中至关重要的原语，例如 allreduce，是不可接受的，因为这些原语即通信密集又对延迟敏感
>  因此，我们将对象元数据存储在 GCS 而不是调度器，完全解耦任务分发和任务调度

>  对于异步执行调用，其延迟就是任务调度所花的时间，任务调度之后，函数就返回 future 了
>  Ray 考虑了 tasks 极多的场景，因此中心化调度器的 load 非常大，很自然的想法就是去中心化

In summary, the GCS significantly simplifies Ray's overall design, as it enables every component in the system to be stateless. This not only simplifies support for fault tolerance (i.e., on failure, components simply restart and read the lineage from the GCS), but also makes it easy to scale the distributed object store and scheduler independently, as all components share the needed state via the GCS. An added benefit is the easy development of debugging, profiling, and visualization tools.
>  总的来说，GCS 显著简化了 Ray 的总体设计，因为它使得系统中的每个组件都是无状态的
>  这不仅简化了对容错的支持 (即故障时，组件简单地重启，并从 GCS 读取血缘即可)，也使得分布式对象存储和调度器可以独立拓展，因为所有的组件都通过 GCS 共享所需的状态
>  还有的好处就是易于 debugging, profiling, visualization

### 4.2.2 Bottom-Up Distributed Scheduler
As discussed in Section 2, Ray needs to dynamically schedule millions of tasks per second, tasks which may take as little as a few milliseconds. None of the cluster schedulers we are aware of meet these requirements. Most cluster computing frameworks, such as Spark [64], CIEL [40], and Dryad [28] implement a centralized scheduler, which can provide locality but at latencies in the tens of ms. Distributed schedulers such as work stealing [12], Sparrow [45] and Canary [47] can achieve high scale, but they either don't consider data locality [12], or assume tasks belong to independent jobs [45], or assume the computation graph is known [47].
>  Ray 需要每秒动态调度数百万任务，任务的执行时间可能仅有几毫秒
>  目前没有集群调度器满足这个要求，大多数集群计算框架都实现中心化的调度器，这提供了局部性，但也存在数十毫秒的延迟
>  分布式调度器例如 work stealing, Sparrow, Canary 可以实现高可拓展，但它们要么不考虑数据局部性，要么假设任务属于独立作业，要么假设计算图是已知的

To satisfy the above requirements, we design a two-level hierarchical scheduler consisting of a global scheduler and per-node local schedulers. To avoid overloading the global scheduler, the tasks created at a node are submitted first to the node's local scheduler. A local scheduler schedules tasks locally unless the node is overloaded (i.e., its local task queue exceeds a predefined threshold), or it cannot satisfy a task's requirements (e.g., lacks a GPU). If a local scheduler decides not to schedule a task locally, it forwards it to the global scheduler. Since this scheduler attempts to schedule tasks locally first (i.e., at the leaves of the scheduling hierarchy), we call it a bottom-up scheduler.
>  为了满足上述需求，我们设计了一个两层的调度器，包含了一个全局调度器和每个节点的本地调度器
>  为了避免全局调度器过载，由节点创建的任务首先会提交到节点的本地调度器
>  本地调度器在本地调度任务，除非节点过载 (即本地任务队列超过了预定义的阈值)，或者它无法满足任务需求 (例如没有 GPU)，如果本地调度器决定不在本地调度任务，它将任务发送给全局调度器
>  因为这个两层的调度器首先尝试在本地调度 (即在调度结构的底层)，我们称其为自底向上调度器

>  先在本地调度，本地没办法满足了再把调度的责任交给全局调度器 (具有全局知识)

The global scheduler considers each node's load and task's constraints to make scheduling decisions. More precisely, the global scheduler identifies the set of nodes that have enough resources of the type requested by the task, and of these nodes selects the node which provides the lowest estimated waiting time. At a given node, this time is the sum of (i) the estimated time the task will be queued at that node (i.e., task queue size times average task execution), and (ii) the estimated transfer time of task's remote inputs (i.e., total size of remote inputs divided by average bandwidth). 
>  全局调度器会考虑每个节点的负载和任务约束来调度任务
>  具体地说，全局调度器识别具有能够满足所请求的任务需求的节点集合，然后从这些节点中选择具有估计最短等待时间的节点
>  在一个给定的节点上，这个等待时间是以下的和:
>  1. 该任务在该节点上的估计排队时间 (任务队列大小 x 平均任务执行时间)
>  2. 任务的远程输入的估计传输时间 (远程输入的总大小 / 平均带宽)

>  全局调度器的调度决策依据依旧是调度时间开销，即延迟

The global scheduler gets the queue size at each node and the node resource availability via heartbeats, and the location of the task's inputs and their sizes from GCS. Furthermore, the global scheduler computes the average task execution and the average transfer bandwidth using simple exponential averaging. If the global scheduler becomes a bottleneck, we can instantiate more replicas all sharing the same information via GCS. This makes our scheduler architecture highly scalable.
>  全局调度器通过 heartbeats 获取每个节点的队列大小和节点资源可用性，并且从 GCS 获取任务的输入和其大小
>  此外，全局调度器会使用指数平均来计算平均任务执行时间和平均传输带宽
>  如果全局调度器成为瓶颈，我们可以通过 GCS 实例化更多共享相同信息的副本，这使得我们的调度器结构高度可拓展

>  指数平均是一种统计方法，新数据的影响随着时间衰减，其公式为

$$
\text{new\_avg} = \alpha \times \text{new\_value} + (1-\alpha) \times \text{old\_avg}
$$

>  其中 $\alpha$ 为权重，表示新数据重要性
>  指数平均实际上等价于一个无穷级数

$$
\begin{align}
y_t &= \alpha x_t + (1-\alpha)y_{t-1}\\
&= \alpha x_t + (1-\alpha)[\alpha x_{t-1} + (1-\alpha) y_{t-2}]\\
&= \alpha x_t + (1-\alpha)\alpha x_{t-1} + (1-\alpha)^2\alpha x_{t-2}+\dots\\
&=\sum_{k=0}^\infty\alpha(1-\alpha)^k x_{t-k}
\end{align}
$$

>  显然占据最后结果的一般大部分是最近的几次数据

### 4.2.3 In-Memory Distributed Object Store
To minimize task latency, we implement an in-memory distributed storage system to store the inputs and outputs of every task, or stateless computation. On each node, we implement the object store via shared memory. This allows zero-copy data sharing between tasks running on the same node. As a data format, we use Apache Arrow [2].
>  为了最小化任务延迟，我们实现了存内的分布式存储系统，存储每个任务 (无状态计算) 的输入和输出
>  在每个节点上，我们通过共享内存实现对象存储，这使得在同一个节点上运行的任务可以进行零拷贝数据共享
>  在数据格式上，我们使用 Apache Arrow

If a task's inputs are not local, the inputs are replicated to the local object store before execution. Also, a task writes its outputs to the local object store. Replication eliminates the potential bottleneck due to hot data objects and minimizes task execution time as a task only reads/writes data from/to the local memory. This increases throughput for computation-bound workloads, a profile shared by many AI applications. 
>  如果任务的输入不是本地的，输入在执行之前被复制到本地的对象存储
>  另外，任务会将其输入也写入本地对象存储
>  通过数据复制，可以消除因热点数据对象带来的潜在瓶颈，并最小化任务执行时间，因为任务只需要从本地内存读写数据，这为 computation-bound workloads 增加了吞吐

For low latency, we keep objects entirely in memory and evict them as needed to disk using an LRU policy.
>  为了实现低延迟，我们将数据对象完全保持在内存中，并使用 LRU (最近最少使用) 策略将它们写入到磁盘

As with existing cluster computing frameworks, such as Spark [64], and Dryad [28], the object store is limited to immutable data. This obviates the need for complex consistency protocols (as objects are not updated), and simplifies support for fault tolerance. In the case of node failure, Ray recovers any needed objects through lineage re-execution. The lineage stored in the GCS tracks both stateless tasks and stateful actors during initial execution; we use the former to reconstruct objects in the store.
>  在现存的集群计算框架中，对象存储限制在不可变数据，这避免了对复杂一致性协议的支持 (因为对象不会被更新)，同时也简化了容错支持
>  当节点故障时，Ray 通过血缘重执行恢复任意所需的对象
>  存储在 GCS 中的血缘信息在初始执行时同时追踪了无状态任务和有状态 actors，我们利用前者来重构存储中的对象

For simplicity, our object store does not support distributed objects, i.e., each object fits on a single node. Distributed objects like large matrices or trees can be implemented at the application level as collections of futures.
>  我们的对象存储不支持分布式对象，即每个对象都能放入单个节点
>  分布式的对象例如大型矩阵和树可以以 collections of futures 实现在应用层

### 4.2.4 Implementation
Ray is an active open source project developed at the University of California, Berkeley. Ray fully integrates with the Python environment and is easy to install by simply running `pip install ray`. 

The implementation comprises  $\approx 40\mathrm{K}$  lines of code (LoC),  $72\%$  in  $\mathrm{C + + }$  for the system layer,  $28\%$  in Python for the application layer. The GCS uses one Redis [50] key-value store per shard, with entirely single-key operations. GCS tables are shared by object and task IDs to scale, and every shard is chain-replicated [61] for fault tolerance. 
>  实现包含了 4 万行代码，72% 是系统层的 C++ 代码，28% 是应用层的 Python 代码
>  GCS 每个 shard 都使用一个 Redis key-value store，所有操作均为单键操作
>  GCS tables 由对象和 task IDs 共享来实现可拓展性，每个 shard 均采用 chain-replication 实现容错 (Redis + chain replication 不如直接上 etcd)

We implement both the local and global schedulers as event-driven, single-threaded processes. Internally, local schedulers maintain cached state for local object metadata, tasks waiting for inputs, and tasks ready for dispatch to a worker. To transfer large objects between different object stores, we stripe the object across multiple TCP connections.
>  本地和全局调度器均实现为事件驱动，单线程的进程
>  在内部，本地调度器会缓存本地对象元数据、等待输入的任务、准备派发给 worker 的任务状态
>  在不同对象存储之间传输大型对象时，系统会将对象切分为多个部分，并通过多个 TCP 连接并行传输，提升效率

## 4.3 Putting Everything Together

![[pics/Ray-Fig7.png]]

Figure 7 illustrates how Ray works end-to-end with a simple example that adds two objects  $a$  and  $b$  which could be scalars or matrices, and returns result  $c$ . The remote function add() is automatically registered with the GCS upon initialization and distributed to every worker in the system (step 0 in Figure 7a).
>  Fig7 描述了一个将两个对象 a, b 相加，返回结果 c 的例子
>  远程函数 `add()` 在系统初始化时会自动注册到 GCS 中，并被分发到系统的每个工作节点

Figure 7a shows the step-by-step operations triggered by a driver invoking `add.remote(a,b)`, where  $a$  and  $b$  are stored on nodes N1 and N2 ,respectively. The driver submits `add(a,b)`  to the local scheduler (step 1), which forwards it to a global scheduler (step 2). Next, the global scheduler looks up the locations of `add(a,b)`  's arguments in the GCS (step 3) and decides to schedule the task on node  $N2$  which stores argument  $b$  (step 4). The local scheduler at node N2 checks whether the local object store contains `add(a,b)`  's arguments (step 5). Since the local store doesn't have object  $a$  it looks up  $a$  's location in the GCS (step 6). Learning that  $a$  is stored at  $N1$, $N2$  s object store replicates it locally (step 7). As all arguments of add() are now stored locally, the local scheduler invokes add() at a local worker (step 8), which accesses the arguments via shared memory (step 9).
> Fig7a 展示了 driver 调用 `add.remote(a,b)` 之后发生的步骤，其中 `a, b` 分别存储在节点 N1, N2，流程为
> 1. driver 将 `add(a, b)` 提交给本地调度器
> 2. 本地调度器将它转发给全局调度器 (N1 也可以决定在本地调度)
> 3. 全局调度器在 GCS 中查找 `add(a,b)` 的参数的位置
> 4. 全局调度器决定将任务调度到存储了参数 `b` 的节点 N2
> 5. 节点 N2 的本地调度器检查是否本地对象存储包含 `add(a,b)` 的参数
> 6. 节点 N2 的本地存储没有 `a`，故它在 GCS 中查找 `a` 的位置
> 7. 节点 N2 发现 `a` 存储在 N1, N2 的对象存储将它拷贝过来
> 8. 节点 N2 的本地调度器在本地 worker 调度 `add()`
> 9. worker 通过共享内存访问参数

Figure 7b shows the step-by-step operations triggered by the execution of ray.get () at  $N1$  and of add () at  $N2$  respectively. Upon ray.get  $(id_{c})$  s invocation, the driver checks the local object store for the value  $c$  using the future  $id_{c}$  returned by add () (step 1). Since the local object store doesn't store  $c$  it looks up its location in the GCS. At this time, there is no entry for  $c$  as  $c$  has not been created yet. As a result,  $N1$  s object store registers a callback with the Object Table to be triggered when  $c$  s entry has been created (step 2). Meanwhile, at  $N2$  add () completes its execution, stores the result  $c$  in the local object store (step 3), which in turn adds  $c$  s entry to the GCS (step 4). As a result, the GCS triggers a callback to  $N1$  s object store with  $c$  s entry (step 5). Next,  $N1$  replicates  $c$  from  $N2$  (step 6), and returns  $c$  to ray.get () (step 7), which finally completes the task.
>  Fig7b 展示了 N1 调用 `ray.get()` 和 N2 调用 `add()` 之后的执行步骤
>  1. 调用 `ray.get(id_c)` 时，driver 使用 `add(a, b)` 返回的 future `id_c` 检查本地对象存储是否有 `c`
>  2. 发现没有，在 GCS 查询它的位置，发现没有 `c` 的 entry，故 N1 会在 Object Table 上注册一个回调函数，它会在 `c` 的 entry 被创建后被调用
>  3. 同时，N2 的 `add()` 完成了执行，将结果 `c` 存储在本地对象存储
>  4. N2 的本地对象存储也将 `c` 的 entry 加入了 GCS
>  5. GCS 触发了对 N1 的对象存储的回调函数
>  6. N1 从 N2 复制 `c`
>  7. N1 返回 `ray.get()`，完成了它的任务

While this example involves a large number of RPCs, in many cases this number is much smaller, as most tasks are scheduled locally, and the GCS replies are cached by the global and local schedulers.
>  虽然这个例子涉及了许多 RPCs，但在许多情况下 RPCs 的数量是很小的，因为大多数任务都在本地调度，并且 GCS 的回复会被全局和本地调度器缓存

# 5 Evaluation
In our evaluation, we study the following questions:

1. How well does Ray meet the latency, scalability, and fault tolerance requirements listed in Section 2? (Section 5.1) 
2. What overheads are imposed on distributed primitives (e.g., allreduce) written using Ray's API? (Section 5.1) 
3. In the context of RL workloads, how does Ray compare against specialized systems for training, serving, and simulation? (Section 5.2) 
4. What advantages does Ray provide for RL applications, compared to custom systems? (Section 5.3)

>  我们研究以下问题:
>  1. Ray 对延迟、可拓展性、容错性要求满足得怎么样
>  2. 使用 Ray 的 API 编写的分布式原语有什么样的开销
>  3. 在 RL workload 的背景下，Ray 和专门用于训练、服务、模拟的系统比较起来表现如何
>  4. 相较于自定义系统，Ray 为 RL 应用提供了哪些优势

All experiments were run on Amazon Web Services. Unless otherwise stated, we use m4.16xlarge CPU instances and p3.16xlarge GPU instances.

## 5.1 Microbenchmarks

![[pics/Ray-Fig8.png]]

**Locality-aware task placement.** Fine-grain load balancing and locality-aware placement are primary benefits of tasks in Ray. Actors, once placed, are unable to move their computation to large remote objects, while tasks can. In Figure 8a, tasks placed without data locality awareness (as is the case for actor methods), suffer 1-2 orders of magnitude latency increase at 10-100MB input data sizes. Ray unifies tasks and actors through the shared object store, allowing developers to use tasks for e.g., expensive postprocessing on output produced by simulation actors.
>  Locality-aware task placement
>  Ray 的 tasks 的主要优势就在于细粒度的负载均衡和基于局部性的任务放置
>  Actors 一旦被放置，则无法将它们的计算移动到大型远程对象上，而任务则可以
>  Fig8a 中，没有局部性感知的任务放置 (即 actor methods 的情况) 在输入数据规模为 10-100MB 时，延迟高 1-2 个数量级
>  Ray 通过共享对象存储统一了 tasks, actors，使得开发者能够利用 tasks 对模拟 actors 的输出进行后处理等操作

**End-to-end scalability.** One of the key benefits of the Global Control Store (GCS) and the bottom-up distributed scheduler is the ability to horizontally scale the system to support a high throughput of fine-grained tasks, while maintaining fault tolerance and low-latency task scheduling. In Figure 8b, we evaluate this ability on an embarrassingly parallel workload of empty tasks, increasing the cluster size on the x-axis. We observe near-perfect linearity in progressively increasing task throughput. 
>  End-to-end scalability
>  GCS 和自底向上的调度器的一个关键优势就是横向拓展系统以支持细粒度任务的高吞吐，同时把保持容错和低延迟任务调度
>  Fig8b 中，我们用极其并行的空任务 workload 来评估这个能力，我们发现任务吞吐量随着集群大小增长呈现出近乎完美的线性关系

>  老实说，空任务都可以本地调度当然线性拓展
>  不过，RL context 下，如果每个节点都有环境，那环境交互的任务确实可以线性拓展

Ray exceeds 1 million tasks per second throughput at 60 nodes and continues to scale linearly beyond 1.8 million tasks per second at 100 nodes. The rightmost datapoint shows that Ray can process 100 million tasks in less than a minute (54s), with minimum variability. As expected, increasing task duration reduces throughput proportionally to mean task duration, but the overall scalability remains linear. While many realistic workloads may exhibit more limited scalability due to object dependencies and inherent limits to application parallelism, this demonstrates the scalability of our overall architecture under high load.

![[pics/Ray-Fig9.png]]

**Object store performance.** To evaluate the performance of the object store (Section 4.2.3), we track two metrics: IOPS (for small objects) and write throughput (for large objects). In Figure 9, the write throughput from a single client exceeds 15GB/s as object size increases. For larger objects, memcpy dominates object creation time. For smaller objects, the main overheads are in serialization and IPC between the client and object store.
>  Object store performance
>  为了评估对象存储的性能，我们追踪两个指标: IOPS (针对小对象)、写吞吐 (针对大对象)
>  Fig9 中，单个 client 的写吞吐随着对象大小增长可以超过 15GB/s，对于更大的对象，主要的创建开销都是 `memcpy`，对于更小的对象，主要的开销是串行化以及客户端和对象存储之间的 IPC

![[pics/Ray-Fig10.png]]

**GCS fault tolerance.** To maintain low latency while providing strong consistency and fault tolerance, we build a lightweight chain replication [61] layer on top of Redis. Figure 10a simulates recording Ray tasks to and reading tasks from the GCS, where keys are 25 bytes and values are 512 bytes. The client sends requests as fast as it can, having at most one in-flight request at a time. Failures are reported to the chain master either from the client (having received explicit errors, or timeouts despite retries) or  from any server in the chain (having received explicit errors). Overall, reconfigurations caused a maximum client-observed delay of under  $30\mathrm{ms}$  (this includes both failure detection and recovery delays).
>  GCS fault tolerance
>  为了在维持低延迟的同时提供强一致性和容错，我们在 Redis 上构建了一层轻量的 chain replication
>  Fig10a 模拟了在 GCS 记录 tasks 和读取 tasks，客户端以尽可能快的速度发送请求，且仅允许一个未完成的请求在传输中
>  故障会被报告给 chain master 节点，故障来源可能是客户端 (收到显式的错误，或重试后仍超时) 也可能是 chain 中任意服务器 (收到显式错误)
>  总体而言，重新配置导致客户端观测延迟最大不超过 30ms (包含了故障检测和恢复延迟)

>  它这个链太短了，要是长一点就不是这样的延迟了

**GCS flushing.** Ray is equipped to periodically flush the contents of GCS to disk. In Figure 10b we submit 50 million empty tasks sequentially and monitor GCS memory consumption. As expected, it grows linearly with the number of tasks tracked and eventually reaches the memory capacity of the system. At that point, the system becomes stalled and the workload fails to finish within a reasonable amount of time. With periodic GCS flushing, we achieve two goals. First, the memory footprint is capped at a user-configurable level (in the microbenchmark we employ an aggressive strategy where consumed memory is kept as low as possible). Second, the flushing mechanism provides a natural way to snapshot lineage to disk for long-running Ray applications.
>  GCS flushing
>  Ray 会定期将 GCS 刷写到磁盘
>  Fig10b 中，我们顺序提交了 5 千万个空 tasks，并观察 GCS 内存消耗
>  正如预期，如果没有 flush, GCS 的内存占用随着追踪的 tasks 数量线性增长，最终达到系统的内存容量上限

![[pics/Ray-Fig11.png]]

**Recovering from task failures.** In Figure 11a, we demonstrate Ray's ability to transparently recover from worker node failures and elastically scale, using the durable GCS lineage storage. The workload, run on m4. xlarge instances, consists of linear chains of  $100\mathrm{ms}$  tasks submitted by the driver. As nodes are removed (at 25s, 50s, 100s), the local schedulers reconstruct previous results in the chain in order to continue execution. Overall per-node throughput remains stable throughout.
>  Recovering fom task failures
>  Fig11a 展示了 Ray 可以使用持久化的 GCS 血缘存储，透明地从 worker node 故障中恢复，并弹性拓展

**Recovering from actor failures.** By encoding actor method calls as stateful edges directly in the dependency graph, we can reuse the same object reconstruction mechanism as in Figure 11a to provide transparent fault tolerance for stateful computation. Ray additionally leverages user-defined checkpoint functions to bound the reconstruction time for actors (Figure 11b). With minimal overhead, checkpointing enables only 500 methods to be re-executed, versus 10k re-executions without checkpointing. In the future, we hope to further reduce actor reconstruction time, e.g., by allowing users to annotate methods that do not mutate state.
>  Recovering from actor failures
>  通过将 actor 方法调用表示为依赖图中的状态边，我们可以复用如 Fig11a 的相同的对象重建机制，为有状态计算提供透明的容错能力
>  Ray 还利用用户定义的检查点函数来减少 actors 的重建时间

![[pics/Ray-Fig12.png]]

**Allreduce.** Allreduce is a distributed communication primitive important to many machine learning workloads. Here, we evaluate whether Ray can natively support a ring allreduce [57] implementation with low enough overhead to match existing implementations [53]. We find that Ray completes allreduce across 16 nodes on 100MB in  $\sim 200\mathrm{ms}$  and 1GB in  $\sim 1200\mathrm{ms}$ , surprisingly outperforming OpenMPI (v1.10), a popular MPI implementation, by  $1.5\times$  and  $2\times$  respectively (Figure 12a). We attribute Ray's performance to its use of multiple threads for network transfers, taking full advantage of the 25Gbps connection between nodes on AWS, whereas OpenMPI sequentially sends and receives data on a single thread [22]. For smaller objects, OpenMPI outperforms Ray by switching to a lower overhead algorithm, an optimization we plan to implement in the future.
>  Allreduce
>  Allreduce 是一个分布式通讯原语，我们评估 Ray 是否可以以足够低的开销支持 ring allreduce 实现
>  Ray 在 16 个节点上可以以 200ms 完成 100MB 的 allreduce，比 OpenMPI 的实现要好
>  我们将这个性能归功于 Ray 使用了多线程网络传输，完全利用了 AWS 节点上的 25Gbps 连接，而 OpenMPI 则顺序地在单线程收发数据

Ray's scheduler performance is critical to implementing primitives such as allreduce. In Figure 12b, we inject artificial task execution delays and show that performance drops nearly  $2\times$  with just a few ms of extra latency. Systems with centralized schedulers like Spark and CIEL typically have scheduler overheads in the tens of milliseconds [62, 38], making such workloads impractical. Scheduler throughput also becomes a bottleneck since the number of tasks required by ring reduce scales quadratically with the number of participants.

## 5.2 Building blocks
End-to-end applications (e.g., AlphaGo [54]) require a tight coupling of training, serving, and simulation. In this section, we isolate each of these workloads to a setting that illustrates a typical RL application's requirements. Due to a flexible programming model targeted to RL, and a system designed to support this programming model, Ray matches and sometimes exceeds the performance of dedicated systems for these individual workloads.
>  端到端应用要求 training, serving, simulation 的紧密绑定
>  在本节，我们将这些 worloads 隔离到一个能体现典型 RL 应用需求的场景中
>  Ray 凭借针对 RL 的灵活编程模型以及为支持该编程模型而设计的系统，Ray 可以匹配甚至超过针对单独 workloads 的专用系统

### 5.2.1 Distributed Training
We implement data-parallel synchronous SGD leveraging the Ray actor abstraction to represent model replicas. Model weights are synchronized via allreduce (5.1) or parameter server, both implemented on top of the Ray API.
>  我们利用 Ray actor 抽象来表示模型副本，实现了数据并行同步 SGD
>  模型权重通过 allreduce 或参数服务器同步，都基于 Ray API 实现

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-13/d4503e99-660f-447e-8cb3-088acaf9d033/a4479a516b86e804c870419408a18cc9edc74e4f9289c9dcfa9b30a9bdfabb7e.jpg)  

Figure 13: Images per second reached when distributing the training of a ResNet-101 TensorFlow model (from the official TF benchmark). All experiments were run on p3.16xl instances connected by 25Gbps Ethernet, and workers allocated 4 GPUs per node as done in Horovod [53]. We note some measurement deviations from previously reported, likely due to hardware differences and recent TensorFlow performance improvements. We used OpenMPI 3.0, TF 1.8, and NCCL2 for all runs.

In Figure 13, we evaluate the performance of the Ray (synchronous) parameter-server SGD implementation against state-of-the-art implementations [53], using the same TensorFlow model and synthetic data generator for each experiment. We compare only against TensorFlow-based systems to accurately measure the overhead imposed by Ray, rather than differences between the deep learning frameworks themselves. In each iteration, model replica actors compute gradients in parallel, send the gradients to a shared parameter server, then read the summed gradients from the parameter server for the next iteration.
>  Fig13 比较了基于 Ray API 的参数服务器 SGD 实现和当前 SOTA 的实现，比较使用了相同的 TensorFlow 模型和合成数据
>  在每次迭代，model replica actors 并行计算梯度，将梯度发送给共享的参数服务器，然后从参数服务器读取规约的梯度

Figure 13 shows that Ray matches the performance of Horovod and is within  $10\%$  of distributed TensorFlow (in `distributed_replicated` mode). This is due to the ability to express the same application-level optimizations found in these specialized systems in Ray's general-purpose API. 

A key optimization is the pipelining of gradient computation, transfer, and summation within a single iteration. To overlap GPU computation with network transfer, we use a custom TensorFlow operator to write tensors directly to Ray's object store.
>  一个关键优化是将单次迭代内的梯度计算、传输和求和过程进行流水线处理，以重叠 GPU 计算和网络传输

### 5.2.2 Serving
Model serving is an important component of end-to-end applications. Ray focuses primarily on the embedded serving of models to simulators running within the same dynamic task graph (e.g., within an RL application on Ray). In contrast, systems like Clipper [19] focus on serving predictions to external clients.
>  Ray 聚焦于在动态任务图内，为 simulators 运行模型服务
>  其他的系统聚焦于对外部客户端进行服务

Table 3: Throughput comparisons for Clipper [19], a dedicated serving system, and Ray for two embedded serving workloads. We use a residual network and a small fully connected network, taking  $10\mathrm{ms}$  and  $5\mathrm{ms}$  to evaluate, respectively. The server is queried by clients that each send states of size 4KB and 100KB respectively in batches of 64.

<table><tr><td>System</td><td>Small Input</td><td>Larger Input</td></tr><tr><td>Clipper</td><td>4400 ± 15 states/sec</td><td>290 ± 1.3 states/sec</td></tr><tr><td>Ray</td><td>6200 ± 21 states/sec</td><td>6900 ± 150 states/sec</td></tr></table>

In this setting, low latency is critical for achieving high utilization. To show this, in Table 3 we compare the server throughput achieved using a Ray actor to serve a policy versus using the open source Clipper system over REST. Here, both client and server processes are colocated on the same machine (a p3.8xlarge instance). This is often the case for RL applications but not for the general web serving workloads addressed by systems like Clipper. 
>  在服务场景下，低延迟对于实现高利用率至关重要
>  Table3 比较使用 Ray actor 来服务策略和使用 Clipper 通过 REST 服务的吞吐，client 和 server 进程位于同一台机器上，这在 RL 应用中很常见，但不适用于一般的 web servering worloads

Due to its low-overhead serialization and shared memory abstractions, Ray achieves an order of magnitude higher throughput for a small fully connected policy model that takes in a large input and is also faster on a more expensive residual network policy model, similar to one used in AlphaGo Zero, that takes smaller input.
>  因为 Ray 具有低开销序列化和共享内存抽象机制，对于一个接收大型输入的小型的全连接策略网络，其吞吐高了一个数量级，在更大的残差策略网络 (类似于 AlphaGo Zero 使用的网络) 上也更快

### 5.2.3 Simulation
Simulators used in RL produce results with variable lengths ("timesteps") that, due to the tight loop with training, must be used as soon as they are available. 
>  RL 中的 simulators 生成变长的结果 (长度即时间步)，由于它和训练过程紧密耦合，其结果必须在生成后立刻被使用

The task heterogeneity and timeliness requirements make simulations hard to support efficiently in BSP-style systems. To demonstrate, we compare (1) an MPI implementation that submits  $3n$  parallel simulation runs on  $n$  cores in 3 rounds, with a global barrier between rounds, to (2) a Ray program that issues the same  $3n$  tasks while concurrently gathering simulation results back to the driver. 
>  任务异构型和时效性的需求使得 simulations 难以被 Bulk Synchronous Parallel 风格的系统高效支持
>  为了说明这一点，我们比较了两个实现:
>  1. 一个 MPI 实现，在 n 个核心上分三轮提交 3n 个并行模拟任务，每轮之间添加全局 barrier
>  2. 一个 Ray 程序，发出相同的 3n 个任务，同时并发地将模拟结果收集回 driver

Table 4 shows that both systems scale well, yet Ray achieves up to  $1.8\times$  throughput. This motivates a programming model that can dynamically spawn and collect the results of fine-grained simulation tasks.
>  两个系统都可以良好拓展，但 Ray 的吞吐更高，说明了能够动态启动模拟任务并细粒度收集模拟任务结果的编程模型的重要性

<table><tr><td>System, programming model</td><td>1 CPU</td><td>16 CPUs</td><td>256 CPUs</td></tr><tr><td>MPI, bulk synchronous</td><td>22.6K</td><td>208K</td><td>2.16M</td></tr><tr><td>Ray, asynchronous tasks</td><td>22.3K</td><td>290K</td><td>4.03M</td></tr></table>

Table 4: Timesteps per second for the Pendulum-v0 simulator in OpenAI Gym [13]. Ray allows for better utilization when running heterogeneous simulations at scale.

## 5.3 RL Applications
Without a system that can tightly couple the training, simulation, and serving steps, reinforcement learning algorithms today are implemented as one-off solutions that make it difficult to incorporate optimizations that, for example, require a different computation structure or that utilize different architectures. 

Consequently, with implementations of two representative reinforcement learning applications in Ray, we are able to match and even outperform custom systems built specifically for these algorithms. The primary reason is the flexibility of Ray's programming model, which can express application-level optimizations that would require substantial engineering effort to port to custom-built systems, but are transparently supported by Ray's dynamic task graph execution engine.

### 5.3.1 Evolution Strategies
To evaluate Ray on large-scale RL workloads, we implement the evolution strategies (ES) algorithm and compare to the reference implementation [49]-a system specially built for this algorithm that relies on Redis for messaging and low-level multiprocessing libraries for datasharing. The algorithm periodically broadcasts a new policy to a pool of workers and aggregates the results of roughly 10000 tasks (each performing 10 to 1000 simulation steps).

As shown in Figure 14a, an implementation on Ray scales to 8192 cores. Doubling the cores available yields an average completion time speedup of  $1.6\times$  .Conversely, the special-purpose system fails to complete at 2048 cores, where the work in the system exceeds the processing capacity of the application driver. To avoid this issue, the Ray implementation uses an aggregation tree of actors, reaching a median time of 3.7 minutes, more than twice as fast as the best published result (10 minutes).

Initial parallelization of a serial implementation using Ray required modifying only 7 lines of code. Performance improvement through hierarchical aggregation was easy to realize with Ray's support for nested tasks and actors. In contrast, the reference implementation had several hundred lines of code dedicated to a protocol for communicating tasks and data between workers, and would require further engineering to support optimizations like hierarchical aggregation.

### 5.3.2 Proximal Policy Optimization
We implement Proximal Policy Optimization (PPO) [51] in Ray and compare to a highly-optimized reference implementation [5] that uses OpenMPI communication primitives. The algorithm is an asynchronous scatter-gather, where new tasks are assigned to simulation actors as they return rollouts to the driver. Tasks are submitted until 320000 simulation steps are collected (each task produces between 10 and 1000 steps). The policy update performs 20 steps of SGD with a batch size of 32768. The model parameters in this example are roughly 350KB. These experiments were run using p2.16xlarge (GPU) and m4.16xlarge (high CPU) instances.
>  我们在 Ray 实现 PPO，并和使用 OpenMPI 通信原语的高度优化的参考实现比较
>  该算法是一个异步的 scatter-gather，当 simulation actors 为 driver 返回了 rollouts，就会有新任务赋予给它
>  任务持续提交，直到收集到 320000 个模拟步 (每个任务产生 10 到 1000 步)
>  策略更新执行 20 步 SGD, batch size 为 32768

As shown in Figure 14b, the Ray implementation outperforms the optimized MPI implementation in all experiments, while using a fraction of the GPUs. The reason is that Ray is heterogeneity-aware and allows the user to utilize asymmetric architectures by expressing resource requirements at the granularity of a task or actor. The Ray implementation can then leverage TensorFlow's single-process multi-GPU support and can pin objects in GPU memory when possible. This optimization cannot be easily ported to MPI due to the need to asynchronously gather rollouts to a single GPU process. Indeed, [5] includes two custom implementations of PPO, one using MPI for large clusters and one that is optimized for GPUs but that is restricted to a single node. Ray allows for an implementation suitable for both scenarios.
>  虽然 Ray 的实现使用的 GPU 更少，但更优，因为 Ray 具备对异构性的感知能力，允许用户在 task 或 actor 的粒度表达资源要求，从而利用不对称架构

Ray's ability to handle resource heterogeneity also decreased PPO's cost by a factor of 4.5 [4], since CPU-only tasks can be scheduled on cheaper high-CPU instances. In contrast, MPI applications often exhibit symmetric architectures, in which all processes run the same code and require identical resources, in this case preventing the use of CPU-only machines for scale-out. Furthermore, the MPI implementation requires on-demand instances since it does not transparently handle failure. Assuming  $4\times$  cheaper spot instances, Ray's fault tolerance and resource-aware scheduling together cut costs by  $18\times$

# 6 Related Work
**Dynamic task graphs.** Ray is closely related to CIEL [40] and Dask [48]. All three support dynamic task graphs with nested tasks and implement the futures abstraction. CIEL also provides lineage-based fault tolerance, while Dask, like Ray, fully integrates with Python. However, Ray differs in two aspects that have important performance consequences. First, Ray extends the task model with an actor abstraction. This is necessary for efficient stateful computation in distributed training and serving, to keep the model data collocated with the computation. Second, Ray employs a fully distributed and decoupled control plane and scheduler, instead of relying on a single master storing all metadata. This is critical for efficiently supporting primitives like allreduce without system modification. 
>  Ray 支持带有嵌套任务的动态任务图，并实现了 future 抽象，同时提供了基于血缘的容错
>  Ray 使用 actor 抽象拓展了 task 模型，这对于分布式训练和服务中的高效有状态计算是必要的
>  Ray 采用了完全分布式的和解耦的控制平台和调度器，而不是依赖于单个 master 存储所有元数据，则对于在不修改系统的情况下高效支持例如 allreduce 的通信原语是必要的

At peak performance for 100MB on 16 nodes, allreduce on Ray (Section 5.1) submits 32 rounds of 16 tasks in  $200\mathrm{ms}$  . Meanwhile, Dask reports a maximum scheduler throughput of 3k tasks/s on 512 cores [3]. With a centralized scheduler, each round of allreduce would then incur a minimum of  $\sim 5\mathrm{ms}$  of scheduling delay, translating to up to  $2\times$  worse completion time (Figure 12b). Even with a decentralized scheduler, coupling the control plane information with the scheduler leaves the latter on the critical path for data transfer, adding an extra roundtrip to every round of allreduce.

**Dataflow systems.** Popular dataflow systems, such as MapReduce [20], Spark [65], and Dryad [28] have widespread adoption for analytics and ML workloads, but their computation model is too restrictive for a finegrained and dynamic simulation workload. Spark and MapReduce implement the BSP execution model, which assumes that tasks within the same stage perform the same computation and take roughly the same amount of time. Dryad relaxes this restriction but lacks support for dynamic task graphs. Furthermore, none of these systems provide an actor abstraction, nor implement a distributed scalable control plane and scheduler. Finally, Naiad [39] is a dataflow system that provides improved scalability for some workloads, but only supports static task graphs.
>  流行的数据流系统的计算模型对于细粒度和动态的 simulation workload 限制性过强
>  Spark 和 MapReduce 实现的是批量同步并行模型，假设了相同阶段的任务执行相同的计算，并消耗大约相同的时间
>  Dryad 放宽了这个限制，但不支持动态任务图
>  此外，这些系统没有提供 actor 抽象，也没有实现分布式的可拓展控制平面和调度器

**Machine learning frameworks.** TensorFlow [7] and MXNet [18] target deep learning workloads and efficiently leverage both CPUs and GPUs. While they achieve great performance for training workloads consisting of static DAGs of linear algebra operations, they have limited support for the more general computation required to tightly couple training with simulation and embedded serving. TensorFlow Fold [33] provides some support for dynamic task graphs, as well as MXNet through its internal  $\mathrm{C + + }$  APIs, but neither fully supports the ability to modify the DAG during execution in response to task progress, task completion times, or faults. TensorFlow and MXNet in principle achieve generality by allowing the programmer to simulate low-level message-passing and synchronization primitives, but the pitfalls and user experience in this case are similar to those of MPI. OpenMPI [22] can achieve high performance, but it is relatively hard to program as it requires explicit coordination to handle heterogeneous and dynamic task graphs. Furthermore, it forces the programmer to explicitly handle fault tolerance.

**Actor systems.** Orleans [14] and Akka [1] are two actor frameworks well suited to developing highly available and concurrent distributed systems. However, compared to Ray, they provide less support for recovery from data loss. To recover stateful actors, the Orleans developer must explicitly checkpoint actor state and intermediate responses. Stateless actors in Orleans can be replicated for scale-out, and could therefore act as tasks, but unlike in Ray, they have no lineage. Similarly, while Akka explicitly supports persisting actor state across failures, it does not provide efficient fault tolerance for stateless computation (i.e., tasks). For message delivery, Orleans provides at-least-once and Akka provides at-most-once semantics. In contrast, Ray provides transparent fault tolerance and exactly-once semantics, as each method call is logged in the GCS and both arguments and results are immutable. We find that in practice these limitations do not affect the performance of our applications. Erlang [10] and  $\mathrm{C + + }$  Actor Framework [17], two other actor-based systems, have similarly limited support for fault tolerance.
>  Orleans, Akka 是两个非常适合构建高可用和高并发的分布式系统的 actor 框架
>  但相较于 Ray，它们在数据丢失的恢复方面的支持较弱
>  为了恢复有状态的 actors, Orleans 的开发者必须显式对 actor 的状态和中间响应进行检查点，Orleans 的无状态 actor 可以作为 task，但没有血缘
>  Akka 显式支持持久化 actor 状态，但没有为无状态计算 (tasks) 提供高效的容错
>  在消息传递方面，Orleans 提供至少一次语义，Akka 提供最多一次语义
>  相比之下，Ray 提供了透明的容错能力和精确一次语义，因为每次调用都会被记录在 GCS 中，且所有参数和结果都为不可变对象，我们发现这些限制在实践中不会影响应用的性能

**Global control store and scheduling.** The concept of logically centralizing the control plane has been previously proposed in software defined networks (SDNs) [16], distributed file systems (e.g., GFS [23]), resource management (e.g.,Omega [52]), and distributed frameworks e.g.,MapReduce [20],BOOM [9]),to name a few. Ray draws inspiration from these pioneering efforts, but provides significant improvements. In contrast with SDNs, BOOM, and GFS, Ray decouples the storage of the control plane information (e.g., GCS) from the logic implementation (e.g., schedulers). This allows both storage and computation layers to scale independently, which is key to achieving our scalability targets. Omega uses a distributed architecture in which schedulers coordinate via globally shared state. To this architecture, Ray adds global schedulers to balance load across local schedulers, and targets ms-level, not second-level, task scheduling.
>  将控制平面逻辑上中心化的概念此前已经在软件定义网络，分布式文件系统，资源管理和分布式框架中提出过
>  Ray 借鉴了这些工作，并做出了改进，Ray 解耦了控制平面信息 (GCS) 的存储和逻辑实现 (调度器)，使得存储层和计算层可以独立拓展
>  Omega 使用分布式架构，其中调度器通过全局共享状态进行协调，Ray 在此基础上引入了全局调度器，来平衡本地调度器的负载，并将任务调度的粒度粒度改为毫秒级，而不是秒级

Ray implements a unique distributed bottom-up scheduler that is horizontally scalable, and can handle dynamically constructed task graphs. Unlike Ray, most existing cluster computing systems [20, 64, 40] use a centralized scheduler architecture. While Sparrow [45] is decentralized, its schedulers make independent decisions, limiting the possible scheduling policies, and all tasks of a job are handled by the same global scheduler. Mesos [26] implements a two-level hierarchical scheduler, but its top-level scheduler manages frameworks, not individual tasks.
>  Ray 实现了一种独特的分布式自底向上调度器，具备水平可拓展性，并且可以处理动态构造的任务图
>  和 Ray 不同，大多数现存集群计算系统使用中心调度器结构，尽管 Sparrow 是去中心化的，其调度器会做独立决策，限制了可能的调度策略，并且一个作业的所有 tatks 会由相同的全局调度器处理
>  Mesos 实现了两次分级调度器，但其顶层调度器管理的是框架而不是单独的任务

Canary [47] achieves impressive performance by having each scheduler instance handle a portion of the task graph, but does not handle dynamic computation graphs.

Cilk [12] is a parallel programming language whose work-stealing scheduler achieves provably efficient load-balancing for dynamic task graphs. However, with no central coordinator like Ray's global scheduler, this fully parallel design is also difficult to extend to support data locality and resource heterogeneity in a distributed setting.

# 7 Discussion and Experiences
Building Ray has been a long journey. It started two years ago with a Spark library to perform distributed training and simulations. However, the relative inflexibility of the BSP model, the high per-task overhead, and the lack of an actor abstraction led us to develop a new system. Since we released Ray roughly one year ago, several hundreds of people have used it and several companies are running it in production. Here we discuss our experience developing and using Ray, and some early user feedback.
>  Ray 项目始于两年前，最初是一个用于分布式训练和模拟的 Spark 库
>  然而，BSP 模型的不灵活性、高的 per-task 开销和对 actor 抽象的缺乏促使我们构建一个新系统

**API.** In designing the API, we have emphasized minimalism. Initially we started with a basic task abstraction. Later, we added the wait() primitive to accommodate rollouts with heterogeneous durations and the actor abstraction to accommodate third-party simulators and amortize the overhead of expensive initializations. While the resulting API is relatively low-level, it has proven both powerful and simple to use. We have already used this API to implement many state-of-the-art RL algorithms on top of Ray, including A3C [36], PPO [51], DQN [37], ES [49], DDPG [55], and Ape-X [27]. In most cases it took us just a few tens of lines of code to port these algorithms to Ray. Based on early user feedback, we are considering enhancing the API to include higher level primitives and libraries, which could also inform scheduling decisions.
>  在设计 API 时，我们始终强调极简主义，最初我们仅提供基础的 task 抽象，之后，我们添加了 `wait()` 原语来支持时间各异的 rollouts，以及添加了 actor 抽象来支持第三方 simulators 并摊销昂贵的初始化开销
>  尽管最终的 API 相对底层，但其功能强大且易于使用，我们已经在 Ray 上实现了许多 RL 算法，大多数情况下，仅几十行代码就能完成这些算法到 Ray 的迁移

**Limitations.** Given the workload generality, specialized optimizations are hard. For example, we must make scheduling decisions without full knowledge of the computation graph. Scheduling optimizations in Ray might require more complex runtime profiling. In addition, storing lineage for each task requires the implementation of garbage collection policies to bound storage costs in the GCS, a feature we are actively developing.
>  考虑到工作负载的通用性，实现专门的优化比较困难
>  例如，我们在缺乏对计算图的完整知识的情况下就必须做出调度决策，Ray 中的调度优化可能需要更复杂的运行时间分析
>  此外，为每个 task 存储血缘需要在 GCS 中实现垃圾回收策略以控制存储成本

>  如果在一个输入都是静态图的部署场景，是不是可以搞一个专门的框架，实现更好的优化？

**Fault tolerance.** We are often asked if fault tolerance is really needed for AI applications. After all, due to the statistical nature of many AI algorithms, one could simply ignore failed rollouts. Based on our experience, our answer is "yes". First, the ability to ignore failures makes applications much easier to write and reason about. Second, our particular implementation of fault tolerance via deterministic replay dramatically simplifies debugging as it allows us to easily reproduce most errors. This is particularly important since, due to their stochasticity, AI algorithms are notoriously hard to debug. Third, fault tolerance helps save money since it allows us to run on cheap resources like spot instances on AWS. Of course, this comes at the price of some overhead. However, we found this overhead to be minimal for our target workloads.
>  虽然 AI 算法的统计特性使得我们可以直接忽略失败的 rollouts，但我们认为仍然需要为 AI 应用实现容错机制
>  首先，能够忽略失败使得应用程序的编写更加简单且容易分析
>  其次，我们通过确定性重放实现的容错机制极大地简化了 debugging，因为它可以轻松复现大多数错误，这对于 AI 算法尤为重要，因为 AI 算法存在随机性，故它们历来难以 debug
>  第三，容错有助于节省成本，因为它允许我们在廉价的资源，例如 AWS 的抢占实例上运行任务
>  虽然容错带来了额外开销，但是对于目标 workloads 而言，这种开销非常少

**GCS and Horizontal Scalability.** The GCS dramatically simplified Ray development and debugging. It enabled us to query the entire system state while debugging Ray itself, instead of having to manually expose internal component state. In addition, the GCS is also the backend for our timeline visualization tool, used for application-level debugging.
>  GCS 极大地简化了 Ray 的开发和 debugging，它使得我们可以在 debugging Ray 本身时查询整个系统状态，而不需要手动暴露各个内部组件的状态
>  此外，GCS 也是我们的时间线可视化工具后端，该工具用于应用层级 debugging

The GCS was also instrumental to Ray's horizontal scalability. In Section 5, we were able to scale by adding more shards whenever the GCS became a bottleneck. The GCS also enabled the global scheduler to scale by simply adding more replicas. Due to these advantages, we believe that centralizing control state will be a key design component of future distributed systems.
>  GCS 在实现 Ray 的水平可拓展性方面也起到了关键作用
>  当 GCS 成为性能瓶颈，我们只需要通过添加更多 shards 来 scale
>  GCS 还使得可以通过添加更多 replicas 来使得全局调度器可以拓展
>  基于这些优势，我们认为集中化控制状态会成为未来分布式系统的关键设计成分

# 8 Conclusion
No general-purpose system today can efficiently support the tight loop of training, serving, and simulation. To express these core building blocks and meet the demands of emerging AI applications, Ray unifies task-parallel and actor programming models in a single dynamic task graph and employs a scalable architecture enabled by the global control store and a bottom-up distributed scheduler. 
>  目前尚没有任何通用目的的系统可以高效支持 training, serving, simulation 之间的紧密循环
>  为了表达这些核心构建模块，并满足新型 AI 应用的要求，Ray 在一个**单一动态任务图**中统一了 task-parallel 和 actor programming model，并借助全局控制存储和自底向上的分布式调度器实现了可拓展的架构

The programming flexibility, high throughput, and low latencies simultaneously achieved by this architecture is particularly important for emerging artificial intelligence workloads, which produce tasks diverse in their resource requirements, duration, and functionality. 
>  这个架构同时达成的编程灵活性、高吞吐、低延迟对于新型的 AI workloads 非常重要，这些 workloads 在资源需求、持续时间和功能上都具有高度的多样性

Our evaluation demonstrates linear scalability up to 1.8 million tasks per second, transparent fault tolerance, and substantial performance improvements on several contemporary RL workloads. Thus, Ray provides a powerful combination of flexibility, performance, and ease of use for the development of future AI applications.
>  我们的评估表明，Ray 可以线性地拓展到每秒处理 180 万任务，同时提供透明的容错，以及在多个主流的 RL workloads 上实现显著的性能提升
>  因此，Ray 为未来的 AI 应用开发提供了兼具灵活性、性能和易用性的平台
