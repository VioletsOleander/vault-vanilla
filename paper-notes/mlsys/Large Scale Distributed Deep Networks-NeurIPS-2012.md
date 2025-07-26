# Abstract
Recent work in unsupervised feature learning and deep learning has shown that being able to train large models can dramatically improve performance. In this paper, we consider the problem of training a deep network with billions of parameters using tens of thousands of CPU cores. We have developed a software framework called DistBelief that can utilize computing clusters with thousands of machines to train large models. 
>  本文考虑用上千个 CPU cores 训练十亿参数级别的模型，相关框架称为 DistBelief

Within this framework, we have developed two algorithms for large-scale distributed training: (i) Downpour SGD, an asynchronous stochastic gradient descent procedure supporting a large number of model replicas, and (ii) Sandblaster, a framework that supports a variety of distributed batch optimization procedures, including a distributed implementation of L-BFGS. 
>  DistBelief 框架包含了两个用于大规模分布式训练的算法:
>  1. Downpour SGD: 支持大量模型副本的异步 SGD 过程
>  2. Sandblaster: 支持多种分布式批量优化过程 (包括 L-BFGS 的分布式实现) 的框架

Downpour SGD and Sandblaster L-BFGS both increase the scale and speed of deep network training. We have successfully used our system to train a deep network 30x larger than previously reported in the literature, and achieves state-of-the-art performance on ImageNet, a visual object recognition task with 16 million images and 21k categories. We show that these same techniques dramatically accelerate the training of a more modestly-sized deep network for a commercial speech recognition service. Although we focus on and report performance of these methods as applied to training large neural networks, the underlying algorithms are applicable to any gradient-based machine learning algorithm.
>  基于该系统，我们训练了比之前大 30x 的网络，在 ImageNet 上达到 SOTA
>  该系统的底层算法适用于任意基于梯度的 ML 算法

# 1 Introduction
Deep learning and unsupervised feature learning have shown great promise in many practical applications. State-of-the-art performance has been reported in several domains, ranging from speech recognition [1, 2], visual object recognition [3, 4], to text processing [5, 6].

It has also been observed that increasing the scale of deep learning, with respect to the number of training examples, the number of model parameters, or both, can drastically improve ultimate classification accuracy [3, 4, 7]. These results have led to a surge of interest in scaling up the training and inference algorithms used for these models [8] and in improving applicable optimization procedures [7, 9]. The use of GPUs [1, 2, 3, 8] is a significant advance in recent years that makes the training of modestly sized deep networks practical. A known limitation of the GPU approach is that the training speed-up is small when the model does not fit in GPU memory (typically less than 6 gigabytes). To use a GPU effectively, researchers often reduce the size of the data or parameters so that CPU-to-GPU transfers are not a significant bottleneck. While data and parameter reduction work well for small problems (e.g. acoustic modeling for speech recognition), they are less attractive for problems with a large number of examples and dimensions (e.g., high-resolution images).
>  使用 GPU 的一个限制就是如果模型太大，GPU 显存装不下时，GPU 的优势就很有限
>  为此，研究者只能减少模型参数大小，避免 CPU-GPU 数据传输成为瓶颈

In this paper, we describe an alternative approach: using large-scale clusters of machines to distribute training and inference in deep networks. We have developed a software framework called DistBelief that enables model parallelism within a machine (via multithreading) and across machines (via message passing), with the details of parallelism, synchronization and communication managed by the framework. In addition to supporting model parallelism, the DistBelief framework also supports data parallelism, where multiple replicas of a model are used to optimize a single objective. 
>  我们提出用大规模机器集群来分布式训练和推理深度网络
>  相关软件框架称为 DistBelief，它允许单机器上的模型并行 (多线程)，和跨机器的模型并行 (消息传递)，这个框架管理了通信、并行、同步的细节
>  DistBelief 也支持数据并行，数据并行即使用多个模型副本优化单个目标

Within this framework, we have designed and implemented two novel methods for large-scale distributed training: (i) Downpour SGD, an asynchronous stochastic gradient descent procedure which leverages adaptive learning rates and supports a large number of model replicas, and (ii) Sandblaster L-BFGS, a distributed implementation of L-BFGS that uses both data and model parallelism. Both Downpour SGD and Sandblaster L-BFGS enjoy significant speed gains compared to more conventional implementations of SGD and L-BFGS.
>  DistBelief 涉及了两个算法:
>  1. Downpour SGD: SGD 的分布式实现，一个异步梯度下降过程，支持大规模的模型副本
>  2. Sandblaster L-BFGS: L-BFGS 的分布式实现，同时使用数据和模型并行

Our experiments reveal several surprising results about large-scale nonconvex optimization. Firstly, asynchronous SGD, rarely applied to nonconvex problems, works very well for training deep networks, particularly when combined with Adagrad [10] adaptive learning rates. Secondly, we show that given sufficient resources, L-BFGS is competitive with or faster than many variants of SGD.

With regard to specific applications in deep learning, we report two main findings: that our distributed optimization approach can both greatly accelerate the training of modestly sized models, and that it can also train models that are larger than could be contemplated otherwise. To illustrate the first point, we show that we can use a cluster of machines to train a modestly sized speech model to the same classification accuracy in less than 1/10th the time required on a GPU. To illustrate the second point, we trained a large neural network of more than 1 billion parameters and used this network to drastically improve on state-of-the-art performance on the ImageNet dataset, one of the largest datasets in computer vision.

# 2 Previous work
In recent years commercial and academic machine learning data sets have grown at an unprecedented pace. In response, a great many authors have explored scaling up machine learning algorithms to cope with this deluge of data [11, 12, 13, 14, 15, 16, 17]. Much of this research has focused on linear, convex models [11, 12, 17]. In the convex case, distributing gradient computation [18] is the naturally first step, but sometimes suffers slowdowns due to synchronization issues. There have been some promising efforts to address this problem, such as lock-less parameter updates in asynchronous stochastic gradient descent, e.g Hogwild [19]. Unfortunately, extending any of these methods to dense nonconvex problems, such as those encountered when training deep architectures, is largely uncharted territory. In particular, it is not known whether it is possible to average parameters or perform dense asynchronous parameter updates in the presence of multiple local minima.

In the context of deep learning, most work has focused on training relatively small models on a single machine (e.g., Theano [20]). An interesting suggestion for scaling up deep learning is the use of a farm of GPUs to train a collection of many small models and subsequently averaging their predictions [21], or modifying standard deep networks to make them inherently more parallelizable [22]. In contrast with previous work, our focus is scaling deep learning techniques in the direction of training very large models, those with a few billion parameters, and without introducing restrictions on the form of the model. In this context, model parallelism, in a spirit similar to [23], is an essential ingredient, but one which must be combined with clever distributed optimization techniques that leverage data parallelism.
>  DL 背景下，大多数工作聚焦在单机上训练小模型
>  一种拓展 DL 的方法是使用 GPU 集群训练多个小模型，然后对预测结果取平均
>  或者是修改标准的 DL 网络，使其本身更具并行性
>  我们聚焦拓展 DL 模型到非常大的级别，具有数十亿参数，并且不对模型的形式加以限制
>  在这个情况下，模型并行性是一个关键要素，且必须结合数据并行进行分布式优化

We considered a number of existing large-scale computational tools for application to our problem, MapReduce [24] and GraphLab [25] being notable examples. We concluded that MapReduce, designed for parallel data processing, was ill-suited for the iterative computations inherent in deep network training; whereas GraphLab, designed for general (unstructured) graph computations, would not exploit computing efficiencies available in the structured graphs typically found in deep networks.
>  现有的大规模计算工具，例如 MapReduce 和 GraphLab 也不适合 DL 训练
>  MapReduce 不适合 DL 训练中的迭代计算 (DL 计算会反复更新参数，MapReduce 针对的是超大规模数据处理，数据是读盘写盘的，对于这种全局可变状态的管理则没有支持)
>  GraphLab 针对的是非结构化的图计算，而 DL 的图是结构化的，GraphLab 无法利用其结构最大化效率

# 3 Model parallelism
To facilitate the training of very large deep networks, we have developed a software framework, DistBelief, that supports distributed computation in neural networks and layered graphical models. The user defines the computation that takes place at each node in each layer of the model, and the messages that should be passed during the upward and downward phases of computation. For large models, the user may partition the model across several machines (Figure 1), so that responsibility for the computation for different nodes is assigned to different machines. The framework automatically parallelizes computation in each machine using all available cores, and manages communication, synchronization and data transfer between machines during both training and inference.
>  DistBelief 专门支持神经网络和层级图模型的分布式计算
>  DistBelief 中，用户定义模型中每一层每一个节点中发生的计算，以及计算的前向和反向计算需要传递的消息
>  如果模型太大，可以将它划分到多台机器上，这样不同节点的计算责任就划分到不同的机器上
>  DistBelief 会自动利用每台机器上的所有核心进行计算，并在训练和推理时管理机器之间的通信、同步和数据传输

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/92b2d390-b4dc-49e1-88c1-d75c37ee78a3/b02b9f0d345a8773cf863da72bdf91a12f7016d38a7d59e6c864d2cbcfbe74b9.jpg)  

Figure 1: An example of model parallelism in DistBelief. A five layer deep neural network with local connectivity is shown here, partitioned across four machines (blue rectangles). Only those nodes with edges that cross partition boundaries (thick lines) will need to have their state transmitted between machines. Even in cases where a node has multiple edges crossing a partition boundary, its state is only sent to the machine on the other side of that boundary once. Within each partition, computation for individual nodes will the parallelized across all available CPU cores.

>  DistBelief 的模型并行如上所示
>  Figure 1 展示了一个 5 层网络，划分到四台机器，其中跨 partition 边界的节点需要在机器之间传输状态
>  partition 内部的计算会划分到多个 CPU cores 上并行计算

The performance benefits of distributing a deep network across multiple machines depends on the connectivity structure and computational needs of the model. Models with a large number of parameters or high computational demands typically benefit from access to more CPUs and memory, up to the point where communication costs dominate. 
>  具有高计算需求的模型通常会从 scale up 中受益，直到通信成为瓶颈

We have successfully run large models with up to 144 partitions in the DistBelief framework with significant speedups, while more modestly sized models show decent speedups for up to 8 or 16 partitions. (See Section 5, under the heading Model Parallelism Benchmarks, for experimental results.) Obviously, models with local connectivity structures tend to be more amenable to extensive distribution than fully-connected structures, given their lower communication requirements. 
>  我们在 DistBelief 框架中运行了最多划分为 144 个 partitions 的大规模模型，中等规模的模型也可以划分为 8 到 16 个 partitions
>  具有局部连接结构的模型通常相较于全连接结构更适合分布式计算

The typical case of less-than-ideal speedups is variance in processing times across the different machines, leading to many machines waiting for the single slowest machine to finish a given phase of computation. Nonetheless, for our largest models, we can efficiently use 32 machines where each machine achieves an average CPU utilization of 16 cores, for a total of 512 CPU cores training a single large neural network. When combined with the distributed optimization algorithms described in the next section, which utilize multiple replicas of the entire neural network, it is possible to use tens of thousands of CPU cores for training a single model, leading to significant reductions in overall training times.
>  通常速度提升不理想的情况是由于不同机器上的处理时间存在差异，导致多台机器需要等待最慢的机器

# 4 Distributed optimization algorithms
Parallelizing computation within the DistBelief framework allows us to instantiate and run neural networks considerably larger than have been previously reported. But in order to train such large models in a reasonable amount of time, we need to parallelize computation not only within a single instance of the model, but to distribute training across multiple model instances. In this section we describe this second level of parallelism, where we employ a set of DistBelief model instances, or replicas, to simultaneously solve a single optimization problem.
>  上节讨论了 DistBelief 中，单个模型 instance 的模型并行
>  本节进一步讨论多个模型 instances 的并行化性训练，也就是第二层的并行化
>  在第二层的并行化中，我们使用一组 DistBelief model instances/replicas，同时求解单个优化问题

We present a comparison of two large-scale distributed optimization procedures: Downpour SGD, an online method, and Sandblaster L-BFGS, a batch method. Both methods leverage the concept of a centralized sharded parameter server, which model replicas use to share their parameters. Both methods take advantage of the distributed computation DistBelief allows within each individual replica. But most importantly, both methods are designed to tolerate variance in the processing speed of different model replicas, and even the wholesale failure of model replicas which may be taken offline or restarted at random.
>  我们比较两种大规模分布式优化方法: Downpour SGD 和 Sandblaster L-BFGS
>  这两种方法都利用了中心化 sharded parameter server 的概念，模型副本通过参数服务器共享它们的参数
>  这两个方法都利用了 DistBelief 在每个单独的模型副本中进行分布式计算，更重要的是，这两个方法都被设计为可以容忍不同模型副本之间处理速度的差异，甚至容忍处理失败，处理失败的部分会被随机下线或重新启动

In a sense, these two optimization algorithms implement an intelligent version of data parallelism. Both approaches allow us to simultaneously process distinct training examples in each of the many model replicas, and periodically combine their results to optimize our objective function.
>  这个角度看，这两个优化方法实现了一个数据并行的智能版本
>  两个方法都允许我们在多个模型部分中同时处理不同的训练样本，并定期合并它们的结果以优化目标函数

## 4.1 Downpour SGD
Stochastic gradient descent (SGD) is perhaps the most commonly used optimization procedure for training deep neural networks [26, 27, 3]. Unfortunately, the traditional formulation of SGD is inherently sequential, making it impractical to apply to very large data sets where the time required to move through the data in an entirely serial fashion is prohibitive.
>  传统的 SGD 公式本质是顺序的，故不适合处理非常大的数据集，因为需要完全串行地遍历所有数据

To apply SGD to large data sets, we introduce Downpour SGD, a variant of asynchronous stochastic gradient descent that uses multiple replicas of a single DistBelief model. The basic approach is as follows: We divide the training data into a number of subsets and run a copy of the model on each of these subsets. The models communicate updates through a centralized parameter server, which keeps the current state of all parameters for the model, sharded across many machines (e.g., if we have 10 parameter server shards, each shard is responsible for storing and applying updates to 1/10th of the model parameters) (Figure 2). This approach is asynchronous in two distinct aspects: the model replicas run independently of each other, and the parameter server shards also run independently of one another.
>  Downpour SGD 是一个异步 SGD 算法，它针对数据并行中，多个模型副本的情况
>  其基本方法是: 将训练数据划分为多个子集，在每个子集上运行一个模型副本，模型副本通过中心的参数服务器通信和更新
>  参数服务器将模型所有参数的当前状态 shard 到多台机器上 (例如，10 台 parameter server，每台负责存储和应用 1/10 的模型参数)
>  Downpour SGD 在两个方面都是异步的: 模型副本彼此独立运行、参数服务器的各个 shard 也彼此独立运行

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/92b2d390-b4dc-49e1-88c1-d75c37ee78a3/89872700fbe860ee7a581e70a4eac06e4f5727f0d620cf04c6d8de09506a5de5.jpg)  

Figure 2: Left: Downpour SGD. Model replicas asynchronously fetch parameters  $w$  and push gradients  $\Delta w$  to the parameter server. Right: Sandblaster L-BFGS. A single 'coordinator' sends small messages to replicas and the parameter server to orchestrate batch optimization.

In the simplest implementation, before processing each mini-batch, a model replica asks the parameter server service for an updated copy of its model parameters. Because DistBelief models are themselves partitioned across multiple machines, each machine needs to communicate with just the subset of parameter server shards that hold the model parameters relevant to its partition. After receiving an updated copy of its parameters, the DistBelief model replica processes a mini-batch of data to compute a parameter gradient, and sends the gradient to the parameter server, which then applies the gradient to the current value of the model parameters.
>  在最简单的实现中，在处理每个 mini-batch 之前，模型副本会向参数服务器请求更新的模型参数
>  因为 DistBelief 模型本身被划分到了多台机器上，故每台机器只需要向部分的参数服务器 shards 通信即可
>  更新完参数后，DistBelief 模型再处理 batch，计算梯度，然后将梯度发送给参数服务器
>  参数服务器再将梯度应用到当前的模型参数上

It is possible to reduce the communication overhead of Downpour SGD by limiting each model replica to request updated parameters only every  $n_{fetch}$  steps and send updated gradient values only every  $n_{push}$  steps (where  $n_{fetch}$  might not be equal to  $n_{push}$ ). In fact, the process of fetching parameters, pushing gradients, and processing training data can be carried out in three only weakly synchronized threads (see the Appendix for pseudocode). In the experiments reported below we fixed  $n_{fetch} = n_{push} = 1$  for simplicity and ease of comparison to traditional SGD.
>  如果要限制通信开销，可以让每个模型部分每隔 $n_{fetch}$ 步才请求更新的参数，并且仅每个 $n_{push}$ 步才发送更新的梯度值
>  实际上，获取参数、推送梯度、处理训练数据仅由三个弱同步的线程来执行

Downpour SGD is more robust to machines failures than standard (synchronous) SGD. For synchronous SGD, if one machine fails, the entire training process is delayed; whereas for asynchronous SGD, if one machine in a model replica fails, the other model replicas continue processing their training data and updating the model parameters via the parameter servers. On the other hand, the multiple forms of asynchronous processing in Downpour SGD introduce a great deal of additional stochasticity in the optimization procedure. Most obviously, a model replica is almost certainly computing its gradients based on a set of parameters that are slightly out of date, in that some other model replica will likely have updated the parameters on the parameter server in the meantime. But there are several other sources of stochasticity beyond this: Because the parameter server shards act independently, there is no guarantee that at any given moment the parameters on each shard of the parameter server have undergone the same number of updates, or that the updates were applied in the same order. Moreover, because the model replicas are permitted to fetch parameters and push gradients in separate threads, there may be additional subtle inconsistencies in the timestamps of parameters. There is little theoretical grounding for the safety of these operations for nonconvex problems, but in practice we found relaxing consistency requirements to be remarkably effective.
>  相较于标准异步 SGD, Downpour SGD 对机器故障更加健壮
>  如果一个模型副本的机器故障，其他模型副本仍然可以处理其数据，并且通过参数服务器更新参数
>  另一方面，Downpour SGD 的多种异步处理方式在优化过程中引入了大量额外随机性，最明显的是: 一个模型副本几乎总是基于一组略微过时的参数计算梯度，因为在同时期，其他副本可能已经更新了参数服务器上的参数
>  此外，还有其他的一些随机性来源: 由于参数服务器分片是独立运行的，无法保证在任何时刻参数服务器的每个分片上都经历了相同次数的更新，或者更新是以相同的顺序进行的
>  此外，由于模型副本被允许在不同的线程中获取参数和推送梯度，参数的时间戳可能出现微妙的不一致
>  对于非凸问题，这些操作的安全性缺乏依据，但在实践中，我们发现放松一致性非常有效

One technique that we have found to greatly increase the robustness of Downpour SGD is the use of the Adagrad [10] adaptive learning rate procedure. Rather than using a single fixed learning rate on the parameter sever ( $\eta$  in Figure 2), Adagrad uses a separate adaptive learning rate for each parameter. Let  $\eta_{i,K}$  be the learning rate of the  $i$ -th parameter at iteration  $K$  and  $\Delta w_{i,K}$  its gradient, then we set:  $\eta_{i,K} = \gamma /\sqrt{\sum_{j = 1}^{K}\Delta w_{i,j}^2}$ . Because these learning rates are computed only from the summed squared gradients of each parameter, Adagrad is easily implemented locally within each parameter server shard. The value of  $\gamma$ , the constant scaling factor for all learning rates, is generally larger (perhaps by an order of magnitude) than the best fixed learning rate used without Adagrad. The use of Adagrad extends the maximum number of model replicas that can productively work simultaneously, and combined with a practice of "warmstarting" model training with only a single model replica before unleashing the other replicas, it has virtually eliminated stability concerns in training deep networks using Downpour SGD (see results in Section 5).
>  我们发现的一个能显著提高 Downpour SGD 健壮性的方法是使用 Adagrad 自适应学习率方法
>  Adagrad 为不同的参数使用不同的自适应学习率，设 $\eta_{i, K}$ 为第 $i$ 个参数在迭代 $K$ 的学习率，其梯度为 $\Delta w_{i, K}$，我们将学习率设为 $\eta_{i, K} = \gamma / \sqrt{\sum_{j=1}^K \Delta w_{i, j}^2}$
>  因为学习率仅基于每个参数的平方梯度和得出，故每个参数服务器可以在本地实现 Adagrad
>  Adagrad 再结合 warmstarting (在启用其他副本之前，先仅训练一个模型副本)，几乎消除了使用 Downpour SGD 训练深度网络的稳定性问题

## 4.2 Sandblaster L-BFGS
Batch methods have been shown to work well in training small deep networks [7]. To apply these methods to large models and large datasets, we introduce the Sandblaster batch optimization framework and discuss an implementation of L-BFGS using this framework.

A key idea in Sandblaster is distributed parameter storage and manipulation. The core of the optimization algorithm (e.g L-BFGS) resides in a coordinator process (Figure 2), which does not have direct access to the model parameters. Instead, the coordinator issues commands drawn from a small set of operations (e.g., dot product, scaling, coefficient-wise addition, multiplication) that can be performed by each parameter server shard independently, with the results being stored locally on the same shard. Additional information, e.g the history cache for L-BFGS, is also stored on the parameter server shard on which it was computed. This allows running large models (billions of parameters) without incurring the overhead of sending all the parameters and gradients to a single central server. (See the Appendix for pseudocode.)
>  Sandblaster 的关键思想是分布式参数存储和操作
>  优化算法的核心 (例如 L-BFGS)，位于一个协调者进程中，该进程不直接访问模型参数，协调者发出一些有限的操作命令 (例如点积、缩放、逐元素加法、乘法)，这些操作命令可以被每个参数服务器分片独立执行，结果也存储在分片本地
>  这使得我们可以运行大规模模型，而无需承担将所有的参数或梯度发送到单一中央服务器的开销

In typical parallelized implementations of L-BFGS, data is distributed to many machines and each machine is responsible for computing the gradient on a specific subset of data examples. The gradients are sent back to a central server (or aggregated via a tree [16]). Many such methods wait for the slowest machine, and therefore do not scale well to large shared clusters. 
>  在 L-BFGS 的典型并行实现中，数据被分发到多台机器上，每台机器负责在一个特定的数据子集上计算梯度，梯度会被发送回中心服务器 (或者通过树状结构聚合)
>  许多这样的方法会等待最慢的机器完成任务，因此在大规模集群中，这类方法拓展性较差

To account for this problem, we employ the following load balancing scheme: The coordinator assigns each of the N model replicas a small portion of work, much smaller than 1/Nth of the total size of a batch, and assigns replicas new portions whenever they are free. With this approach, faster model replicas do more work than slower replicas. To further manage slow model replicas at the end of a batch, the coordinator schedules multiple copies of the outstanding portions and uses the result from whichever model replica finishes first. This scheme is similar to the use of "backup tasks" in the MapReduce framework [24]. 
>  为了解决这个问题，我们采用了一种负载均衡方法:
>  coordinator 为 N 个模型副本的每一个分配一部分工作，远小于一个 batch 大小的 1/N，并且在副本空闲时，就为它分配新工作
>  在这个调度下，更快的模型副本就会比更慢的模型副本处理更多的任务
>  为了进一步在 batch 处理接近结束时管理慢的模型副本，coordinator 会调度尚未完成的部分给多个模型副本，使用最先完成的模型副本的工作
>  这个方案类似于 MapReduce 中 backup task 的使用

Prefetching of data, along with supporting data affinity by assigning sequential portions of data to the same worker makes data access a non-issue. In contrast with Downpour SGD, which requires relatively high frequency, high bandwidth parameter synchronization with the parameter server, Sandblaster workers only fetch parameters at the beginning of each batch (when they have been updated by the coordinator), and only send the gradients every few completed portions (to protect against replica failures and restarts).
>  通过预取数据，并将连续的数据分配给同一个 worker 以提高 data affinity，可以使得数据访问不再是问题
>  Downpour SGD 需要频繁地，高带宽地和参数服务器进行参数同步，Sandblaster workers 则仅在每个 batch 的开始时获取参数 (此时参数以由 coordinator 更新)，并且在完成几个部分后才发送梯度

>  首先，Downpour SGD 和 Sandblaster 方法下，计算节点上都拥有完整的模型副本，但注意这个节点实际上是 DistBelief 的抽象节点，实际的模型会被 DistBelief 进行划分
>  其次，Downpour SGD 和 Sandblaster 方法下，计算节点都会向参数服务器获取完整的模型参数，但注意实际上，DistBelief 下的分布式节点会向各自的参数服务器 shard 获取各自的参数 (当然整个 DistBelief 的前向传播和反向更新还是需要自己内部同步的)
>  因此，我们可以先明确，这两个方法的核心都是数据并行，每个计算节点都负责一个数据集子集，基于完整的模型计算局部梯度 (模型并行由 DistBelief 实现)

>  二者的差异在于:
>  Downpour SGD 优先考虑最大化训练吞吐量，故允许计算节点几乎完全异步地独立更新参数
>  Downpour SGD 中，每个计算节点在 mini-batch 开始时就拉取最新参数、计算梯度、推送梯度，以此循环，并且参数服务器收到新的 (局部) 梯度，就直接更新参数，不进行梯度的等待和聚合
>  可以看到，Downpour SGD 中，计算节点之间不存在相互等待，参数服务器也不会等待所有计算节点完成任务，故 Downpour SGD 的异步性很强，不过通信频率也很高 (每个 mini-batch 都通信)，这使得各个计算节点的参数都会略微过时 (但不至于太过时)，其优势就是高并行，高吞吐，劣势就是不稳定，随机大

>  Sandblaster 则更多考虑了稳定性
>  Sandblaster 中，每个大迭代 (batch) 开始时，所有计算节点会同步从参数服务器获取**一致**的参数
>  随后，每个计算节点在本地处理数据集子集 (一般是**多个** mini-batch)，完成前向和反向计算之后，将梯度推送给参数服务器
>  参数服务器会等待所有计算节点完成，然后执行梯度聚合，并更新参数，然后再次发放参数，以此迭代

>  因此，二者的主要差异就在于细粒度和粗粒度，以及异步和同步
>  Downpour SGD 是细粒度的全异步计算，Sandblaster 是粗粒度的同步计算
>  Downpour 的细粒度就是为了缓解异步计算带来的随机性 (增大通信频率)
>  Sandblaster 的粗粒度就是为了同步带来的等待开销 (减小通信频率)

# 5 Experiments
We evaluated our optimization algorithms by applying them to training models for two different deep learning problems: object recognition in still images and acoustic processing for speech recognition.

The speech recognition task was to classify the central region (or frame) in a short snippet of audio as one of several thousand acoustic states. We used a deep network with five layers: four hidden layer with sigmoidal activations and 2560 nodes each, and a softmax output layer with 8192 nodes. The input representation was 11 consecutive overlapping  $25~\mathrm{ms}$  frames of speech, each represented by 40 log-energy values. The network was fully-connected layer-to-layer, for a total of approximately 42 million model parameters. We trained on a data set of 1.1 billion weakly labeled examples, and evaluated on a hold out test set. See [28] for similar deep network configurations and training procedures.

For visual object recognition we trained a larger neural network with locally-connected receptive fields on the ImageNet data set of 16 million images, each of which we scaled to  $100\mathrm{x}100$  pixels [29]. The network had three stages, each composed of filtering, pooling and local contrast normalization, where each node in the filtering layer was connected to a  $10\mathrm{x}10$  patch in the layer below. Our infrastructure allows many nodes to connect to the same input patch, and we ran experiments varying the number of identically connected nodes from 8 to 36. The output layer consisted of 21 thousand one-vs-all logistic classifier nodes, one for each of the ImageNet object categories. See [30] for similar deep network configurations and training procedures.

**Model parallelism benchmarks:** To explore the scaling behavior of DistBelief model parallelism (Section 3), we measured the mean time to process a single mini-batch for simple SGD training as a function of the number of partitions (machines) used in a single model instance. In Figure 3 we quantify the impact of parallelizing across N machines by reporting the average training speed-up: the ratio of the time taken using only a single machine to the time taken using N. Speedups for inference steps in these models are similar and are not shown here.
>  为了探索 DistBelief 模型并行的拓展行为，我们衡量处理的单个 mini-batch 的 SGD 训练时间和单个 model insatnce 划分的 partitions (machines) 数量的关系
>  我们报告平均加速比来衡量并行到 N 台机器的效果

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/92b2d390-b4dc-49e1-88c1-d75c37ee78a3/ff0ecb4d2e800758aab4682423ac1820f8c605dceaa51534d4ff6dcb2811bc61.jpg)  

Figure 3: Training speed-up for four different deep networks as a function of machines allocated to a single DistBelief model instance. Models with more parameters benefit more from the use of additional machines than do models with fewer parameters.

The moderately sized speech model runs fastest on 8 machines, computing  $2.2\times$ faster than using a single machine. (Models were configured to use no more than 20 cores per machine.) Partitioning the model on more than 8 machines actually slows training, as network overhead starts to dominate in the fully-connected network structure and there is less work for each machine to perform with more partitions.
>  中等大小的模型在并行到 8 台机器上加速比最大，比单台机器超过 2.2x
>  超过 8 台机器反而减缓训练，因为网络开销开始占主导，同时每个 partition 的工作也开始变少

In contrast, the much larger, locally-connected image models can benefit from using many more machines per model replica. The largest model, with 1.7 billion parameters benefits the most, giving a speedup of more than  $12\times$  using 81 machines. For these large models using more machines continues to increase speed, but with diminishing returns.
>  更大规模的模型随着模型副本数量增大而加速比增大

**Optimization method comparisons:** To evaluate the proposed distributed optimization procedures, we ran the speech model described above in a variety of configurations. We consider two baseline optimization procedures: training a DistBelief model (on 8 partitions) using conventional (single replica) SGD, and training the identical model on a GPU using CUDA [28]. The three distributed optimization methods we compare to these baseline methods are: Downpour SGD with a fixed learning rate, Downpour SGD with Adagrad learning rates, and Sandblaster L-BFGS.

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/92b2d390-b4dc-49e1-88c1-d75c37ee78a3/4f2ff5640902d118fb95866bf3f981fe1ee9aa50c579e3943b08bda1d49ae2f2.jpg)  

Figure 4: Left: Training accuracy (on a portion of the training set) for different optimization methods. Right: Classification accuracy on the hold out test set as a function of training time. Downpour and Sandblaster experiments initialized using the same  $\sim 10$  hour warmstart of simple SGD.

Figure 4 shows classification performance as a function of training time for each of these methods on both the training and test sets. Our goal is to obtain the maximum test set accuracy in the minimum amount of training time, regardless of resource requirements. Conventional single replica SGD (black curves) is the slowest to train. Downpour SGD with 20 model replicas (blue curves) shows a significant improvement. Downpour SGD with 20 replicas plus Adagrad (orange curve) is modestly faster. Sandblaster L-BFGS using 2000 model replicas (green curves) is considerably faster yet again. The fastest, however, is Downpour SGD plus Adagrad with 200 model replicas (red curves). Given access to sufficient CPU resources, both Sandblaster L-BFGS and Downpour SGD with Adagrad can train models substantially faster than a high performance GPU.

Though we did not confine the above experiments to a fixed resource budget, it is interesting to consider how the various methods trade off resource consumption for performance. We analyze this by arbitrarily choosing a fixed test set accuracy  $(16\%)$ , and measuring the time each method took to reach that accuracy as a function of machines and utilized CPU cores, Figure 5. One of the four points on each traces corresponds to a training configuration shown in Figure 4, the other three points are alternative configurations.

![](https://cdn-mineru.openxlab.org.cn/result/2025-07-26/92b2d390-b4dc-49e1-88c1-d75c37ee78a3/9d9a0848b3d5ead221d0ae9ec831e32792029a04ac16ac8a6b934bf8679548b0.jpg)  

Figure 5: Time to reach a fixed accuracy (16%) for different optimization strategies as a function of number of the machines (left) and cores (right).

In this plot, points closer to the origin are preferable in that they take less time while using fewer resources. In this regard Downpour SGD using Adagrad appears to be the best trade-off: For any fixed budget of machines or cores, Downpour SGD with Adagrad takes less time to reach the accuracy target than either Downpour SGD with a fixed learning rate or Sandblaster L-BFGS. For any allotted training time to reach the accuracy target, Downpour SGD with Adagrad used few resources than Sandblaster L-BFGS, and in many cases Downpour SGD with a fixed learning rate could not even reach the target within the deadline. The Sandblaster L-BFGS system does show promise in terms of its scaling with additional cores, suggesting that it may ultimately produce the fastest training times if used with an extremely large resource budget (e.g., 30k cores).

**Application to ImageNet:** The previous experiments demonstrate that our techniques can accelerate the training of neural networks with tens of millions of parameters. However, the more significant advantage of our cluster-based approach to distributed optimization is its ability to scale to models that are much larger than can be comfortably fit on single machine, let alone a single GPU. As a first step toward exploring the capabilities of very large neural networks, we used Downpour SGD to train the 1.7 billion parameter image model described above on the ImageNet object classification task. As detailed in [30], this network achieved a cross-validated classification accuracy of over 15%, a relative improvement over 60% from the best performance we are aware of on the 21k category ImageNet classification task.
>  之前的实验表明，我们的技术可以加速具有数千万参数的 DNN 的训练
>  实际上，基于集群的分布式优化方法的更大优势在于它能拓展到比单台机器 (更不用说单台 GPU) 所能容纳的模型大得多的模型
>  我们使用 Downpour SGD 在 ImageNet 上训练的 1.7 B 的模型，该网络达到了 SOTA

# 6 Conclusions
In this paper we introduced DistBelief, a framework for parallel distributed training of deep networks. Within this framework, we discovered several effective distributed optimization strategies. We found that Downpour SGD, a highly asynchronous variant of SGD works surprisingly well for training nonconvex deep learning models. Sandblaster L-BFGS, a distributed implementation of L-BFGS, can be competitive with SGD, and its more efficient use of network bandwidth enables it to scale to a larger number of concurrent cores for training a single model. That said, the combination of Downpour SGD with the Adagrad adaptive learning rate procedure emerges as the clearly dominant method when working with a computational budget of 2000 CPU cores or less.
>  本文介绍了 DistBelief，一个用于 DNN 分布式训练的框架
>  在该框架内，我们探索了一些有效的分布式优化策略
>  我们发现 Downpour SGD —— 一个高度异步的 SGD 变体，在训练非凸 DL 模型时的表现非常好

Adagrad was not originally designed to be used with asynchronous SGD, and neither method is typically applied to nonconvex problems. It is surprising, therefore, that they work so well together, and on highly nonlinear deep networks. We conjecture that Adagrad automatically stabilizes volatile parameters in the face of the flurry of asynchronous updates, and naturally adjusts learning rates to the demands of different layers in the deep network.
>  Adagrad 最初并不设计用于异步 SGD，这两个方法通常也不用于非凸问题
>  但它们协同对于高度非线性的 DNN 异步优化效果很好
>  我们推断 Adgrad 在面对大量异步更新时会自动稳定波动的参数，并根据 DNN 中不同层的需求自然调整学习率

Our experiments show that our new large-scale training methods can use a cluster of machines to train even modestly sized deep networks significantly faster than a GPU, and without the GPU's limitation on the maximum size of the model. To demonstrate the value of being able to train larger models, we have trained a model with over 1 billion parameters to achieve better than state-of-the-art performance on the ImageNet object recognition challenge.

# 7 Appendix
For completeness, here we provide pseudocode for the model replica (client) side of Downpour SGD (Algorithm 0.1), and Sandblaster L-BFGS (Algorithm 0.2).

Sandblaster is a framework for distributed batch optimization procedures. An essential concept in Sandblaster is decomposing operations into local computation on the DistBelief parameter server. By way of example, suppose we have 1 billion parameters and 10 parameter server shards, so that each shard has 1/10 of the parameters. It is possible to decompose L-BFGS into a sequence of scalar-vector products  $(\alpha \times \mathbf{x})$  and vector-vector inner products  $(\mathbf{x}^T \mathbf{y})$ , where each vector is 1 billion dimensional. If one shard is always responsible for the first 1/10 of every vector used internally in L-BFGS, and a second shard is always responsible for the second 1/10 of every vector, and so on up to the final shard always being responsible for the final 1/10 of every vector, it is possible to show that these scalar-vector and vector-vector operations can all be done in a distributed fashion with very little communication, so that any intermediate vector-valued results are automatically stored in the same distributed fashion, and any intermediate scalar-valued result is communicated to all the shards.
