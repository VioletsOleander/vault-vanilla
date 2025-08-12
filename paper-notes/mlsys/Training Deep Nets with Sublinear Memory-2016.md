```
Tianqi Chen , Bing Xu , Chiyuan Zhang , and Carlos Guestrin
```

# Abstract
We propose a systematic approach to reduce the memory consumption of deep neural network training. Specifically, we design an algorithm that costs  $O(\sqrt{n})$  memory to train a  $n$  layer network, with only the computational cost of an extra forward pass per mini-batch. As many of the state-of-the-art models hit the upper bound of the GPU memory, our algorithm allows deeper and more complex models to be explored, and helps advance the innovations in deep learning research. 
>  我们提出减少 DNN 训练的 memory 消耗的方法
>  我们设计了一个算法，训练 $n$ 层网络需要的 memory 为 $O(\sqrt n)$，代价是每个 mini-batch 需要有一次额外的前向计算开销 (相当于重复算两遍)
>  由于许多 SOTA 模型达到了单个 GPU memory 的上限，我们的算法可以帮助探索更深更复杂的模型

We focus on reducing the memory cost to store the intermediate feature maps and gradients during training. Computation graph analysis is used for automatic in-place operation and memory sharing optimizations. We show that it is possible to trade computation for memory giving a more memory efficient training algorithm with a little extra computation cost. 
>  我们聚焦于训练时减少存储中间特征图和梯度的 memory cost
>  我们通过计算图分析实现自动的原地操作和内存共享优化
>  我们证明了可以通过增加少量计算成本来换取更高的内存效率

In the extreme case, our analysis also shows that the memory consumption can be reduced to  $O(\log n)$  with as little as  $O(n\log n)$  extra cost for forward computation. Our experiments show that we can reduce the memory cost of a 1,000-layer deep residual network from 48G to 7G on ImageNet problems. Similarly, significant memory cost reduction is observed in training complex recurrent neural networks on very long sequences.
>  在极限情况下，我们可以将内存消耗减少到 $O(\log n)$，只需要 $O(n\log n)$ 的前向计算开销
>  我们可以将 1000 层的网络的内存开销从 48G 减少到 7G

# 1 Introduction
In this paper, we propose a systematic approach to reduce the memory consumption of deep neural network training. We mainly focus on reducing the memory cost to store intermediate results (feature maps) and gradients, as the size of the parameters are relatively small comparing to the size of the intermediate feature maps in many common deep architectures. We use a computation graph analysis to do automatic in-place operation and memory sharing optimizations. More importantly, we propose a novel method to trade computation for memory. As a result, we give a practical algorithm that cost  $O(\sqrt{n})$  memory for feature maps to train a  $n$  layer network with only double the forward pass computational cost. Interestingly, we also show that in the extreme case, it is possible to use as little as  $O(\log n)$  memory for the features maps to train a  $n$  layer network.
>  我们聚焦于减少存储中间结果和梯度的内存开销，不考虑参数大小
>  我们使用计算图分析执行自动原地运算和内存共享优化 (即将能够原地运算的运算优化为原地运算，将能够共享内存的运算优化为共享内存)
>  我们提出了用计算换内存的方法，得到了 $O(\sqrt n)$ 内存开销的算法，代价是翻倍了前向传播计算开销
>  我们还展示了在极端情况下，内存开销可以进一步压缩 da 到 $O(\log n)$

We have recently witnessed the success of deep neural networks in many domains [8], such as computer vision, speech recognition, natural language processing and reinforcement learning. Many of the success are brought by innovations in new architectures of deep neural networks. Convolutional neural networks [15, 14, 13, 10] model the spatial patterns and give the state of art results in computer vision tasks. Recurrent neural networks, such as long short-term memory [12], show inspiring results in sequence modeling and structure prediction. One common trend in those new models is to use deeper architectures [18, 14, 13, 10] to capture the complex patterns in a large amount of training data. Since the cost of storing feature maps and their gradients scales linearly with the depth of network, our capability of exploring deeper models is limited by the device (usually a GPU) memory. For example, we already run out of memories in one of the current state-of-art models as described in [11]. In the long run, an ideal machine learning system should be able to continuously learn from an increasing amount of training data. Since the optimal model size and complexity often grows with more training data, it is very important to have memory-efficient training algorithms.

Reducing memory consumption not only allows us to train bigger models. It also enables larger batch size for better device utilization and stability of batchwise operators such as batch normalization [13]. For memory limited devices, it helps improve memory locality and potentially leads to better memory access patterns. It also enables us to switch from model parallelism to data parallelism for training deep convolutional neural networks, which can be beneficial in certain circumstances. Our solution enables us to train deeper convolutional neural networks, as well as recurrent neural networks with longer unrolling steps. We provide guidelines for deep learning frameworks to incorporate the memory optimization techniques proposed in this paper. We will also make our implementation of memory optimization algorithm publicly available.

# 2 Related Works
We can trace the idea of computational graph and liveness analysis back to the literatures of compiler optimizations [3]. Analogy between optimizing a computer program and optimizing a deep neural network computational graph can be found. For example, memory allocation in deep networks is similar to register allocation in a compiler. 
>  计算图和活跃性分析的思想可以追溯到编译优化技术
>  我们可以找到优化计算机程序和优化 DNN 计算图之间的类比，例如 DNN 的内存分配类似于编译器中的寄存器分配

>  活跃性分析即判断某个变量在程序的某个点是否还会被使用，如果不再被使用，它的内存就可以被回收

The formal analysis of computational graph allows us save memory in a principled way. Theano [5, 4] is a pioneering framework to bring the computation graph to deep learning, which is joined by recently introduced frameworks such as CNTK [2], Tensorflow [1] and MXNet [6]. Theano and Tensorflow use reference count based recycling and runtime garbage collection to manage memory during training, while MXNet uses a static memory allocation strategy prior to the actual computation. However, most of the existing framework focus on graph analysis to optimize computation after the gradient graph is constructed, but do not discuss the computation and memory trade-off.
>  对计算图的形式化分析提供了一种节约内存的原则方式
>  Theano 框架将计算图引入了 DL，之后的框架也加入这一行列
>  Theano 使用基于引用计数的回收和运行时垃圾回收来管理训练过程中的内存，MXNet 采用在实际计算之前的静态内存分配策略
>  大多数框架聚焦于在构建了梯度图之后，通过图分级优化计算图，但没有讨论计算和存储之间的权衡

>  形式化分析指使用数学方法，例如数据流/依赖分析精确定义每个张量的定义-使用-死亡

The trade-off between memory and computation has been a long standing topic in systems research. Although not widely known, the idea of dropping intermediate results is also known as gradient checkpointing technique in automatic differentiation literature [9]. We bring this idea to neural network gradient graph construction for general deep neural networks. Through the discussion with our colleagues [19], we know that the idea of dropping computation has been applied in some limited specific use-cases. In this paper, we propose a general methodology that works for general deep neural networks, including both convolutional and recurrent neural networks. Our results show that it is possible to train a general deep neural network with sublinear memory cost.
>  system research 中，存算之间的 tradeoff 是长期讨论的话题
>  “丢弃中间结果“ 的想法在自动微分文献中也被称为梯度检查点，我们将这个想法报道针对通用 DNN 的梯度图构建中
>  我们提出一种适用于通用 DNN (包括 CNN, RNN) 的通用方法，结果表明可以以次线性的内存成本训练通用 DNN

 More importantly, we propose an automatic planning algorithm to provide a good memory plan for real use-cases. The proposed gradient graph optimization algorithm can be readily combined with all the existing memory optimizations in the computational graph to further reduce the memory consumption of deep learning frameworks.

There are other ways to train big models, such as swapping of CPU/GPU memory and use of model parallel training [7, 16]. These are orthogonal approaches and can be used together with our algorithm to train even bigger models with fewer resources. Moreover, our algorithm does not need additional communication over PCI-E and can save the bandwidth for model/data parallel training.
>  我们的方法不涉及通讯

# 3 Memory Optimization with Computation Graph
We start by reviewing the concept of computation graph and the memory optimization techniques. Some of these techniques are already used by existing frameworks such as Theano [5, 4], Tensorflow [1] and MXNet [6]. A computation graph consists of operational nodes and edges that represent the dependencies between the operations. Fig. 1 gives an example of the computation graph of a two-layer fully connected neural network. Here we use coarse grained forward and backward operations to make the graph simpler. We further simplify the graph by hiding the weight nodes and gradients of the weights. A computation graph used in practice can be more complicated and contains mixture of fine/coarse grained operations. The analysis presented in this paper can be directly used in those more general cases.
>  计算图包含运算节点和边，边表示运算之间的依赖
>  Fig1 给出了计算图的示例
>  我们通过隐藏权重节点和权重的梯度来简化计算图

Once the network configuration (forward graph) is given, we can construct the corresponding backward pathway for gradient calculation. A backward pathway can be constructed by traversing the configuration in reverse topological order, and apply the backward operators as in normal backpropagation algorithm. 
>  给定网络配置 (前向图)，我们就可以构造对应的反向路径来进行梯度计算
>  反向路径可以通过按逆拓扑排序遍历前向图来构建，然后按照正常反向传播算法应用 backward operator 即可

The backward pathway in Fig. 1 represents the gradient calculation steps explicitly, so that the gradient calculation step in training is simplified to just a forward pass on the entire computation graph (including the gradient calculation pathway). 
>  Fig1 的反向路径明确地表示梯度计算步骤，故训练中的梯度计算步骤就简化为对整个计算图 (包括了梯度计算路径) 执行一次前向计算

Explicit gradient path also offers some other benefits (e.g. being able to calculate higher order gradients), which is beyond our scope and will not be covered in this paper.
>  显式的梯度路径还有一些其他好处，例如能够计算高阶梯度

>  前向图的拓扑排序决定了前向传播的计算顺序，例如先计算 `a`，才能计算 `b`
>  逆拓扑排序就是将拓扑排序反过来，这也正是反向传播的计算顺序，例如我们先计算 `b` 的梯度，才能计算 `a` 的梯度

>  例如，Fig1 的前向顺序/拓扑排序为 `input -> fullc-forward -> sigmoid-forward -> fullc-forward -> softmax-forward`
>  对应的逆拓扑排序为 `softmax-forward -> fullc-forward -> sigmoid-forward -> fullc-forward -> input`，这个顺序决定了反向传播的计算顺序
>  训练的第一步是计算损失，也就是沿着前向路径一直计算到图中额外添加的 ` log-loss ` 节点，这个节点依赖于 ` softmax-forward `, ` label `
>  之后，需要根据损失计算梯度，这里因为 softmax 激活函数和交叉熵损失的性质，输出值的梯度计算不需要涉及损失函数的具体值，只需要 `label` 就可以，因此我们根据 `softmax-backward` 算子计算 `softmax` 层的梯度，这个节点依赖于 `label, softmax-forward`
>  接着我们一路沿着反向数据流即可，各个 `xxx-backward` 算子的具体依赖取决于它自己的性质，最后我们可以得到 `input-grad`

> [!info] Topological Order
> 拓扑排序是一种对有向无环图的顶点进行线性排序的算法，这个排序满足一个条件: 如果图中存在从点 A 到点 B 的路径，那么在排序结果中，A 一定出现在 B 的前面
> 拓扑排序的核心是处理任务之间的依赖关系，有些任务必须先完成，才能完成后面的任务

![[pics/Training Deep Nets with Sublinear Memory Cost-Fig1.png]]

When training a deep convolutional/recurrent network, a great proportion of the memory is usually used to store the intermediate outputs and gradients. Each of these intermediate results corresponds to a node in the graph. A smart allocation algorithm is able to assign the least amount of memory to these nodes by sharing memory when possible. 
>  当训练深度 CNN, RNN，大量的内存会被用于存储中间输出和梯度，每个中间结果都对应了图中的一个节点 (该节点的计算结果)
>  一个只能的内存分配算法可以在可能的情况下通过共享内存来为这些节点分配最少量的内存

Fig. 1 shows a possible allocation plan of the example two-layer neural network. Two types of memory optimizations can be used 

- Inplace operation: Directly store the output values to memory of a input value.
- Memory sharing: Memory used by intermediate results that are no longer needed can be recycled and used in another node.

>  Fig 1 展示了对示例的计算图的一个内存分配方案，它使用了两类内存优化方法:
>  - 原地运算: 直接将输出值存储到输出值的内存中
>  - 内存共享: 由不再需要的中间结果使用的内存可以被其他的结点直接使用

Allocation plan in Fig. 1 contains examples of both cases. The first sigmoid transformation is carried out using inplace operation to save memory, which is then reused by its backward operation. The storage of the softmax gradient is shared with the gradient by the first fully connected layer. Ad hoc application of these optimizations can leads to errors. For example, if the input of an operation is still needed by another operation, applying inplace operation on the input will lead to a wrong result.
>  Fig1 中，第一个 sigmoid 转换就使用原地运算来节省内存，并且后续的 sigmoid-backward 运算也使用原地运算，使用同一块内存
>  softmax-backward 的运算结果则和第一个 fullc-backward 的运算结果共享存储
>  如果随意应用这些优化，可能会导致错误，例如，如果某个操作的输入仍被另一个操作所需要，对该输入进行原地计算就会导致错误的结果

We can only share memory between the nodes whose lifetime do not overlap. There are multiple ways to solve this problem. One option is to construct the conflicting graph of with each variable as node and edges between variables with overlapping lifespan and then run a graph-coloring algorithm. This will cost  $O(n^2)$  computation time. We adopt a simpler heuristic with only  $O(n)$  time. 
>  我们只能在生命周期不重叠的节点之间共享内存
>  解决这个问题有许多方法，一个选择是构建一个冲突图，其中节点表示变量，如果变量之间的生命周期重叠，就在它们之间建立边，然后运行一个图着色算法 (使得相邻的节点有不同的颜色，那么生命周期重叠的节点就有不同的内存块)
>  这将花费 $O(n^2)$ 的计算时间，其中 $n$ 是变量的数量

The algorithm is demonstrated in Fig. 2. It traverses the graph in topological order, and uses a counter to indicate the liveness of each record. An inplace operation can happen when there is no other pending operations that depend on its input. Memory sharing happens when a recycled tag is used by another node. This can also serve as a dynamic runtime algorithm that traverses the graph, and use a garbage collector to recycle the outdated memory. 
>  算法如 Fig2 所示
>  该算法按拓扑排序遍历图，并使用一个计数器来表示每个记录 (通常是张量) 的活跃状态 (有多少个待定操作依赖于这个数据)
>  当没有其他待定操作依赖于这个数据时，这个张量的内存就可以复用，即可以进行原地运算 (也就是计数器 = 1 时)
>  当计数器降为 0 时，说明这个张量已经 “死亡”，不再被任何特定操作依赖，此时它的内存就可以被回收或者被重用
>  这也可以作为一个动态的运行时算法，遍历图，并使用垃圾收集器来回收过时的内存

We use this as a static memory allocation algorithm, to allocate the memory to each node before the execution starts, in order to avoid the overhead of garbage collection during runtime.
>  我们将其作为静态内存分配算法，在执行开始前为每个节点分配内存，以避免运行时垃圾回收的开销
![[pics/Training Deep Nets with Sublinear Memory Cost-Fig2.png]]

**Guidelines for Deep Learning Frameworks** As we can see from the algorithm demonstration graph in Fig. 2. The data dependency causes longer lifespan of each output and increases the memory consumption of big network. It is important for deep learning frameworks to

- Declare the dependency requirements of gradient operators in minimum manner.
- Apply liveness analysis on the dependency information and enable memory sharing.

>  从 Fig2 中的算法演示图中，我们可以知道，数据依赖性会导致每个输出的生命周期编程，进而增大大型网络的内存消耗
>  因此，对于 DL 框架来说，以下几点非常重要:
>  - 以最小的方式声明梯度算子的依赖关系
>  - 对依赖信息进行活跃性分析，并启用内存共享

>  以最小的方式声明依赖也就是只声明必要的，最少的依赖，例如 `sigmoid-backward` 只需要 `sigmoid` 的输出值来计算梯度，而不需要 `sigmoid` 的输入值
>  这是为了最小化中间结果的生命周期

It is important to declare minimum dependencies. For example, the allocation plan in Fig. 1 won't be possible if sigmoid-backward also depend on the output of the first full-forward. The dependency analysis can usually reduce the memory footprint of deep network prediction of a  $n$  layer network from  $O(n)$  to nearly  $O(1)$  because sharing can be done between each intermediate results. The technique also helps to reduce the memory footprint of training, although only up to a constant factor.
>  声明最小化依赖非常重要
>  在声明最小化依赖后的依赖分析通常可以将推理的内存占用从 $O(n)$ 降低到接近 $O(1)$，因为可以在每个中间结果之间进行内存共享
>  该技术也可以帮助减少训练时的内存占用，尽管只能减少一个常数因子

# 4 Trade Computation for Memory
## 4.1 General Methodology
The techniques introduced in Sec. 3 can reduce the memory footprint for both training and prediction of deep neural networks. However, due to the fact that most gradient operators will depend on the intermediate results of the forward pass, we still need  $O(n)$  memory for intermediate results to train a  $n$  layer convolutional network or a recurrent neural networks with a sequence of length  $n$ . In order to further reduce the memory, we propose to drop some of the intermediate results, and recover them from an extra forward computation when needed.
>  上一节介绍的技术可以用于减少推理和训练的内存占用
>  但是由于大多数梯度算子依赖于前向过程的中间结果，在训练时，我们仍然需要 $O(n)$ 的内存
>  为了进一步减少训练时的内存占用，我们提出丢弃部分中间结果，在需要的使用通过额外的前向计算恢复它们

![[pics/Training Deep Nets with Sublinear Memory Cost-Algorithm1.png]]

More specifically, during the backpropagation phase, we can re-compute the dropped intermediate results by running forward from the closest recorded results. To present the idea more clearly, we show a simplified algorithm for a linear chain feed-forward neural network in Alg. 1. Specifically, the neural network is divided into several segments. The algorithm only remembers the output of each segment and drops all the intermediate results within each segment. The dropped results are recomputed at the segment level during back-propagation. As a result, we only need to pay the memory cost to store the outputs of each segment plus the maximum memory cost to do backpropagation on each segment.
>  更具体的说，在反向传播过程中，我们从最近记录的中间结果重新运行前向计算来重新计算被丢弃的中间结果
>  在 Algorithm 1 中，网络被分成几段，该算法仅记住每个段的输出，并丢弃每个段内的所有中间结果
>  在反向传播过程中，被丢弃的结果会在段级别上被重新计算
>  这样，我们只需要存储各个段的结果需要的内存，以及在各个单独的段上进行反向传播时，重计算需要的内存

Alg. 1 can also be generalized to common computation graphs as long as we can divide the graph into segments. However, there are two drawbacks on directly applying Alg. 1: 1) users have to manually divide the graph and write customized training loop; 2) we cannot benefit from other memory optimizations presented in Sec 3. 
>  直接运用 Algorithm 1 有两个劣势:
>  1. 用户需要手动划分图，并编写自定义的训练循环
>  2. 无法利用 Sec3 提到的内存优化技术 (算法 1 没有进行内存复用方面的优化)

![[pics/Training Deep Nets with Sublinear Memory Cost-Algorithm2.png]]

We solve this problem by introducing a general gradient graph construction algorithm that uses essentially the same idea. The algorithm is given in Alg. 2. In this algorithm, the user specify a function  $m_i: \mathcal{V} \to \mathbb{N}$  on the nodes of a computation graph to indicate how many times a result can be recomputed. We call  $m$  the mirror count function as the re-computation is essentially duplicating (mirroring) the nodes. When all the mirror counts are set to 0, the algorithm degenerates to normal gradient graph. 
>  为了解决这个问题，我们引入了一个通用的梯度图构造算法，利用的是相同的思想
>  该算法中，用户指定一个定义在计算图节点上的函数 $m_i: \mathcal V \to \mathbb N$  ，该函数表明了该节点的结果的重计算次数
>  我们称 $m$ 为镜像计数函数，因为重计算本质上是复制 (镜像) 节点
>  当所有镜像计数设置为 0 时，算法退化到普通的梯度图 (没有重计算，存储所有中间结果)

To specify re-computation pattern in Alg. 2, the user only needs to set the  $m(v) = 1$  for nodes within each segment and  $m(v) = 0$  for the output node of each segment. The mirror count can also be larger than 1, which leads to a recursive generalization to be discussed in Sec 4.4. 
>  要指定算法 2 中的重计算模式，用户需要将每个段内的节点的 $m(v)$ 设定为 1，将每个段的输出节点的 $m(v)$ 设置为 0
>  镜像计数也可以大于 1，这将导致递归的泛化

![[pics/Training Deep Nets with Sublinear Memory Cost-Fig3.png]]

>  Fig3 中，虚线表示控制依赖，即一个操作必须要在另一个操作完成之后才能进行，但二者之间没有数据依赖

>  Fig3 中，例如我们要计算 `bn-forward` 的梯度，但是 `bn-forward` 的前向计算结果已经被丢弃，我们需要寻找离 `bn-forward` 最近的，已经被存储的激活，在图中就是 `conv-forward`，我们利用该结果，重新计算 `bn-forward` 的前向结果

>  图中引入的控制依赖是为了避免内存复用影响结果，例如 `relu-forward` 指向 `bn-forward` 重计算的虚线表示 `bn-forward` 的重计算必须等待 `relu-forward` 完成才可以，这是因为 `relu-forward` 的重计算会复用 `relu-forward` 的内存，故需要等待 `relu-forward` 完成计算 (且其 users 都读取了 `relu-forwad` 的结果) 之后，才能覆盖其内存

Fig. 3 shows an example of memory optimized gradient graph. Importantly, Alg. 2 also outputs a traversal order for the computation, so the memory usage can be optimized. Moreover, this traversal order can help introduce control flow dependencies for frameworks that depend on runtime allocation.
>  Fig3 展示了一个内存优化的梯度图示例
>  重要的是，算法 2 还会输出计算的遍历顺序，从而优化内存使用
>  此外，这个遍历顺序可以帮助那些依赖于运行时内存分配的框架引入控制流依赖关系

## 4.2 Drop the Results of Low Cost Operations
One quick application of the general methodology is to drop the results of low cost operations and keep the results that are time consuming to compute. This is usually useful in a Conv-BatchNorm-Activation pipeline in convolutional neural networks. We can always keep the result of convolution, but drop the result of the batch normalization, activation function and pooling. In practice this will translate to a memory saving with little computation overhead, as the computation for both batch normalization and activation functions are cheap.
>  上述通用方法的一个快速应用就是丢弃低成本 operation 的结果，保留那些计算耗时的结果
>  这在 CNN 中的 Conv-BatchNorm-Activation 流水线中通常很有用，我们会保留卷积的结果，丢弃 batch normalization, activation function, pooling 的结果
>  实践中，这会显著节约内存，并且额外的计算开销很小，因为 batch normalization 和 activation function 的计算成本都很低

## 4.3 An  $O(\sqrt{n})$  Memory Cost Algorithm
Alg. 2 provides a general way to trade computation for memory. It remains to ask which intermediate result we should keep and which ones to re-compute. Assume we divide the  $n$  network into  $k$  segments the memory cost to train this network is given as follows.
>  算法 2 提供了用计算换内存的通用方式，但我们需要考虑哪些中间结果需要保留，哪些需要重计算
>  假设我们将 $n$ 个网络划分为 $k$ 个片段，训练该网络的内存开销如下所示:

$$
{\text{cost-total}}=\max _{i=1,\ldots,k}{\text{cost-of-segment}}(i)+O(k)=O\left(\frac{n}{k}\right)+O(k) \tag{1}
$$

The first part of the equation is the memory cost to run back-propagation on each of the segment. Given that the segment is equally divided, this translates into  $O(n / k)$  cost. The second part of equation is the cost to store the intermediate outputs between segments. Setting  $k = \sqrt{n}$ , we get the cost of  $O(2\sqrt{n})$ . 
>  公式的第一部分是在每个 segment 上运行反向传播的内存开销，因为 segments 是等长划分的，故第一部分的开销的数量级就是 $O(\frac n k)$
>  公式的第二部分是存储 segments 之间的中间结果的开销
>  如果 segments 数量为 $k = \sqrt n$，那么开销就是 $O(2\sqrt n)$

This algorithm only requires an additional forward pass during training, but reduces the memory cost to be sub-linear. Since the backward operation is nearly twice as time consuming as the forward one, it only slows down the computation by a small amount.
>  该算法仅需要训练时额外的前向传播，但能够将内存开销降低到 sub-linear
>  因为反向计算的开销一般是前向计算的两倍，故多一次前向计算不会增加总体计算成本太多

>  sub-linear 就是增长率小于线性 $O(n)$ 的情况，比线性函数增长的慢的情况包括:
>  - $O(\log n)$
>  - $O(\log^c n)$
>  - $O(\sqrt n)$
>  - $O(n^c), 0<c<1$

In the most general case, the memory cost of each layer is not the same, so we cannot simply set  $k = \sqrt{n}$ . However, the trade-off between the intermediate outputs and the cost of each stage still holds. In this case, we use Alg. 3 to do a greedy allocation with a given budget for the memory cost within each segment as a single parameter  $B$ . 
>  在大多数情况下，每一层的内存开销一般不一样，故我们不能直接设定 $k = \sqrt n$
>  但中间输出和每个阶段的开销之间的权衡仍然成立
>  在这种情况下，我们使用算法 3 进行贪心分配，将每个 segment 内的内存成本预算记作一个参数 $B$

Varying  $B$  gives us various allocation plans that either assign more memory to the intermediate outputs, or to computation within each stage. When we do static memory allocation, we can get the exact memory cost given each allocation plan. We can use this information to do a heuristic search over  $B$  to find optimal memory plan that balances the cost of the two. 
>  通过调整 $B$，我们可以得到不同的内存分配方案，这些方案要么为中间输出分配更多内存，要么为每个阶段的计算分配更多的内存
>  如果我们进行静态内存而分配，可以根据每种分配方案精确计算内存成本
>  我们可以利用这些信息，对 $B$ 进行启发式搜索，以找到二者之间取得平衡的最优的内存方案

The details of the searching step is presented in the supplementary material. We find this approach works well in practice. We can also generalize this algorithm by considering the cost to run each operation to try to keep time consuming operations when possible.
>  在实践中，这个方法效果很好
>  我们可以进一步泛化该算法，考虑每个计算的运行成本，以尽可能保留耗时的计算

![[pics/Training Deep Nets with Sublinear Memory Cost-Algorithm3.png]]

## 4.4 More General View: Recursion and Subroutine

![[pics/Training Deep Nets with Sublinear Memory Cost-Fig4.png]]

In this section, we provide an alternative view of the memory optimization scheme described above. Specifically, we can view each segment as a bulk operator that combines all the operations inside the segment together. The idea is illustrated in Fig. 4. The combined operator calculates the gradient by executing over the sub-graph that describes its internal computation. This view allows us to treat a series of operations as subroutines. The optimization within the sub-graph does not affect the external world. As a result, we can recursively apply our memory optimization scheme to each sub-graph.
>  本节为上一节的内存优化方案提供另一种视角
>  具体地说，我们将每个 segment 视为一个 bulk operator，它将 segment 内的所有 operations 组合在一起
>  如 Fig4 所示，combined operator 通过执行描述了它内部计算的子图来计算梯度
>  这种视角使得我们将 operations 视作一系列子例程，子图内部的优化不会影响外部世界，因此，我们可以递归地将内存优化方案应用到每个子图中

**Pay Even Less Memory with Recursion** Let  $g(n)$  to be the memory cost to do forward and backward pass on a  $n$  layer neural network. Assume that we store  $k$  intermediate results in the graph and apply the same strategy recursively when doing forward and backward pass on the sub-path. We have the following recursion formula.
>  令 $g(n)$ 为在 $n$ 层 NN 上进行前向和反向传播的内存成本
>  假设我们在图中存储 $k$ 个中间结果，并对子路径进行前向和反向传播递归地应用相同的策略，我们得到以下的递归公式:

$$
g(n) = k + g\left(n / (k + 1)\right) \tag{2}
$$

>  $n$ 层网络的内存开销等于存储 $k$ 个中间结果的内存开销和一个 $n/(k+1)$ 层网络的内存开销
>  (存储 $k$ 个中间结果等于将网络划分为 $n/(k+1)$ 个 segments)

Solving this recursion formula gives us

$$
g(n) = k\log_{k + 1}(n) \tag{3}
$$

As a special case, if we set  $k = 1$ , we get  $g(n) = \log_2 n$ . This is interesting conclusion as all the existing implementations takes  $O(n)$  memory in feature map to train a  $n$  layer neural network. This will require  $O(\log_2 n)$  cost forward pass cost, so may not be used commonly. But it demonstrates how we can trade memory even further by using recursion.
>  解这个递归公式，我们得到 Eq3
>  作为特殊情况，如果我们设置 $k=1$，则得到 $g(n) = \log_2n$，这是一个有趣的结论，因为现有的所有实现训练一个 $n$ 层的网络都需要 $O(n)$ 的内存
>  这样的递归实现会导致前向传播的成本变为原来的 $O(\log_2n)$ 倍，因此可能不常使用，但它展示了我们如何通过递归进一步权衡内存使用

## 4.5 Guideline for Deep Learning Frameworks
In this section, we have shown that it is possible to trade computation for memory and combine it with the system optimizations proposed in Sec 3. It is helpful for deep learning frameworks to

- Enable option to drop result of low cost operations.
- Provide planning algorithms to give efficient memory plan.
- Enable user to set the mirror attribute in the computation graph for memory optimization.

While the last option is not strictly necessary, providing such interface enables user to hack their own memory optimizers and encourages future researches on the related directions. Under this spirit, we support the customization of graph mirror plan and will make the source code publicly available.

>  本节中，我们讨论了如何用计算换取内存，并将其与 Section3 提出的系统优化相结合
>  因此，DL 框架可以:
>  - 提供丢弃低成本运算结果的选项
>  - 提供生成高效内存方案的规划算法
>  - 允许用户在计算图中设置镜像属性，以进行内存优化

# 5 Experiments
## 5.1 Experiment Setup
We evaluate the memory cost of storing intermediate feature maps using the methods described in this paper. We our method on top of MXNet [6], which statically allocate all the intermediate feature maps before computation. This enables us to report the exact memory cost spend on feature maps. Note that the memory cost of parameters and temporal memory (e.g. required by convolution) are not part of the memory cost report. 
>  我们在 MXNet 上评估本文所述方法存储中间特征图的内存成本
>  MXNet 会在计算之前静态分配所有的内存特征图，这使得我们可以准确报告用于内存特征图的内存成本
>  我们不考虑参数的内存成本和临时内存 (例如卷积所需的内存) 的成本

We also record the runtime total memory cost by running training steps on a Titan X GPU. Note that all the memory optimizations proposed in this paper gives equivalent weight gradient for training and can always be safely applied. We compare the following memory allocation algorithms

- no optimization, directly allocate memory to each node in the graph without any optimization.
- inplace, enable inplace optimization when possible.
- sharing, enable inplace optimization as well as sharing. This represents all the system optimizations presented at Sec. 3.
- drop bn-relu, apply all system optimizations, drop result of batch norm and relu, this is only shown in convolutional net benchmark.
- sublinear plan, apply all system optimizations, use plan search with Alg 3 to trade computation with memory.

>  我们比较了以下几种内存分配算法:
>  - 无优化: 直接为图中每个节点分配内存，没有任何优化
>  - inplace: 可能的情况下启用 inplace 优化
>  - sharing: 同时启用 inplace 优化和共享机制，即 Section 3 介绍的所有系统优化
>  - drop bn-relu: 应用所有系统优化，并丢弃 batch norm 和 relu 的结果
>  - sublinear plan: 应用所有系统优化，并使用算法 3 进行规划搜索来用计算换内存

## 5.2 Deep Convolutional Network
We first evaluate the proposed method on convolutional neural network for image classification. We use deep residual network architecture [11] (ResNet), which gives the state of art result on this task. Specifically, we use 32 batch size and set input image shape as (3, 224, 224). We generate different depth configuration of ResNet by increasing the depth of each residual stage.

We show the results in Fig. 5. We can find that the system optimizations introduced in Sec. 3 can help to reduce the memory cost by factor of two to three. However, the memory cost after optimization still exhibits a linear trend with respect to number of layers. Even with all the system optimizations, it is only possible to train a 200 layer ResNet with the best GPU we can get. On the other hand, the proposed algorithm gives a sub-linear trend in terms of number of layers. By trade computation with memory, we can train a 1000 layer ResNet using less than 7GB of GPU memory.
>  Section3 的内存优化方法已经可以减少 2 倍到 3 倍的内存开销，但优化后，内存开销仍然和层数呈线性关系
>  而我们的算法则给出了次线性的关系，这使得我们可以在 7GB 显存上训练 1000 层的网络

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-06/1c21a2b7-93d9-4761-b7a0-5f9333e16446/03af37e538d5f25de2d6062a7c0bb146dcb69a7b2eafde488c09ea6055c13617.jpg)  


Figure 5: The memory cost of different allocation strategies on deep residual net configurations. The feature map memory cost is generated from static memory allocation plan. We also use nvidia-smi to measure the total memory cost during runtime (the missing points are due to out of memory). The figures are in log-scale, so  $y = \alpha x^{\beta}$  will translate to  $\log (y) = \beta \log (x) + \log \alpha$ . We can find that the graph based allocation strategy indeed helps to reduce the memory cost by a factor of two to three. More importantly, the sub-linear planning algorithm indeed gives sub-linear memory trend with respect to the workload. The real runtime result also confirms that we can use our method to greatly reduce memory cost deep net training.

## 5.3 LSTM for Long Sequences
We also evaluate the algorithms on a LSTM under a long sequence unrolling setting. We unrolled a four layer LSTM with 1024 hidden states equals 64 over time. The batch size is set to 64. The input of each timestamp is a continuous 50 dimension vector and the output is softmax over 5000 class. This is a typical setting for speech recognition[17], but our result can also be generalized to other recurrent networks. Using a long unrolling step can potentially help recurrent model to learn long term dependencies over time. We show the results in Fig. 6. We can find that inplace helps a lot here. This is because inplace optimization in our experiment enables direct addition of weight gradient to a single memory cell, preventing allocate space for gradient at each timestamp. The sub-linear plan gives more than  $4x$  reduction over the optimized memory plan.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-06/1c21a2b7-93d9-4761-b7a0-5f9333e16446/d8a077d2ddaf7684ab333ee41e3ddeeda37e1e046b3ae97410e1049f7a9c181a.jpg)  

Figure 6: The memory cost of different memory allocation strategies on LSTM configurations. System optimization gives a lot of memory saving on the LSTM graph, which contains a lot of fine grained operations. The sub-linear plan can give more than 4x reduction over the optimized plan that do not trade computation with memory.

## 5.4 Impact on Training Speed
We also measure the runtime cost of each strategy. The speed is benchmarked on a single Titan X GPU. The results are shown in Fig. 7. Because of the double forward cost in gradient calculation, the sublinear allocation strategy costs  $30\%$  additional runtime compared to the normal strategy. By paying the small price, we are now able to train a much wider range of deep learning models.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-06/1c21a2b7-93d9-4761-b7a0-5f9333e16446/d7439115deb8693321db7fbc1faf99e8cfb466d58959d258a674c608ded10960.jpg)  

Figure 7: The runtime speed of different allocation strategy on the two settings. The speed is measured by a running 20 batches on a Titan X GPU. We can see that using sub-linear memory plan incurs roughly  $30\%$  of additional runtime cost compared to linear memory allocation. The general trend of speed vs workload remains linear for both strategies.

# 6 Conclusion
In this paper, we proposed a systematic approach to reduce the memory consumption of the intermediate feature maps when training deep neural networks. Computation graph liveness analysis is used to enable memory sharing between feature maps. We also showed that we can trade the computation with the memory. By combining the techniques, we can train a  $n$  layer deep neural network with only  $O(\sqrt{n})$  memory cost, by paying nothing more than one extra forward computation per mini-batch.
>  本文提出了一种系统的方法，减少训练深度网络时中间特征图的内存消耗
>  我们使用计算图的存活分析来实现特征图之间的内存共享，我们还展示了如何用计算来换取内存
>  通过结合这些计数，我们在以每个 mini-batch 额外一次前向计算的情况下，用 $O(\sqrt n)$ 的内存成本训练一个 $n$ 层的 DNN

# A Search over Budget $B$
Alg. 3 allows us to generate an optimized memory plan given a single parameter  $B$  . This algorithm relies on approximate memory estimation for faster speed. After we get the plan, we can use the static allocation algorithm to calculate the exact memory cost. We can then do a grid search over  $B$  to find a good memory plan.

To get the setting of the grid, we first run the allocation algorithm with  $B = 0$  , then run the allocation algorithm again with  $B = \sqrt{xy}$  . Here  ${x}$  and  $y$  are the outputs from Alg. 3 in the first run. Here  $x$  is the approximate cost to store inter-stage feature maps and  $y$  is the approximate cost to run each stage.  $B = \sqrt{xy}$  an estimation of each stage's memory cost. This can already give a good memory plan. We then set grid around  $B = \sqrt{xy}$  to further refine the solution.

In practice, we find that using a size 6 grid on  $[B / \sqrt{2},\sqrt{2} B]$  can already give good memory plans in the experiments. We implemented the allocation algorithm in python without any attempt to optimize for speed. Our code costs a few seconds to get the plans needed in the experiments.