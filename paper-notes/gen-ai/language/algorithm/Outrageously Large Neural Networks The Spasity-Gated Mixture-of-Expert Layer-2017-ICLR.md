# Abstract
The capacity of a neural network to absorb information is limited by its number of parameters. 
>  模型能力依赖于其参数数量

Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation. 
>  条件计算: 每个输入样本都是被网络的部分激活参数处理
>  条件计算可以在不成比例增加计算量的情况下增加模型能力

In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, achieving greater than  $1000x$  improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. 

We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. 
>  本文提出稀疏门控的 MoE 层，它包含上千个 FFN 子网络
>  一个可训练的门控网络为每个样本决定一种 experts 的稀疏组合并交由处理

We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.
>  MoE 模型的计算成本低于 SOTA 模型，但效果却更好

# 1 Introduction And Related Work
## 1.1 Conditional Computation
Exploiting scale in both training data and model size has been central to the success of deep learning. When datasets are sufficiently large, increasing the capacity (number of parameters) of neural networks can give much better prediction accuracy. This has been shown in domains such as text (Sutskever et al., 2014; Bahdanau et al., 2014; Jozefowicz et al., 2016; Wu et al., 2016), images (Krizhevsky et al., 2012; Le et al., 2012), and audio (Hinton et al., 2012; Amodei et al., 2015). For typical deep learning models, where the entire model is activated for every example, this leads to a roughly quadratic blow-up in training costs, as both the model size and the number of training examples increase. Unfortunately, the advances in computing power and distributed computation fall short of meeting such demand.

Various forms of conditional computation have been proposed as a way to increase model capacity without a proportional increase in computational costs (Davis & Arel, 2013; Bengio et al., 2013; Eigen et al., 2013; Ludovic Denoyer, 2014; Cho & Bengio, 2014; Bengio et al., 2015; Almahairi et al., 2015). In these schemes, large parts of a network are active or inactive on a per-example basis. The gating decisions may be binary or sparse and continuous, stochastic or deterministic. Various forms of reinforcement learning and back-propagation are proposed for training the gating decisions.
>  门控决策可以是二元的、稀疏的、连续的、随机的、确定性的
>  也有个各种 RL 方法和反向传播方法被用于训练门控决策

While these ideas are promising in theory, no work to date has yet demonstrated massive improvements in model capacity, training time, or model quality. We blame this on a combination of the following challenges:

- Modern computing devices, especially GPUs, are much faster at arithmetic than at branching. Most of the works above recognize this and propose turning on/off large chunks of the network with each gating decision. 
- Large batch sizes are critical for performance, as they amortize the costs of parameter transfers and updates. Conditional computation reduces the batch sizes for the conditionally active chunks of the network. 
- Network bandwidth can be a bottleneck. A cluster of GPUs may have computational power thousands of times greater than the aggregate inter-device network bandwidth. To be computationally efficient, the relative computational versus network demands of an algorithm must exceed this ratio. Embedding layers, which can be seen as a form of conditional computation, are handicapped by this very problem. Since the embeddings generally need to be sent across the network, the number of (example, parameter) interactions is limited by network bandwidth instead of computational capacity.
- Depending on the scheme, loss terms may be necessary to achieve the desired level of sparsity per-chunk and/or per example. Bengio et al. (2015) use three such terms. These issues can affect both model quality and load-balancing.
- Model capacity is most critical for very large data sets. The existing literature on conditional computation deals with relatively small image recognition data sets consisting of up to 600,000 images. It is hard to imagine that the labels of these images provide a sufficient signal to adequately train a model with millions, let alone billions of parameters.

>  通过 MoE 显著提高模型能力和质量的挑战包括:
>  - 现代计算设备的分支能力弱于算术能力，意识到这点的工作提出在每次门控决策关闭或开启网络的大块区域
>  - 大批量数据有利于性能，因为摊销了参数传输和更新的成本，而条件计算会降低网络中条件激活块的处理的 batch size
>  - 集群的网络带宽可能成为瓶颈
>  - 可能需要损失项来达到 per-example/per-chunk 所需的稀疏度，损失项也会影响模型质量和负载均衡
>  - 需要有能和超大模型能力匹配的超大数据集

In this work, we for the first time address all of the above challenges and finally realize the promise of conditional computation. We obtain greater than  $1000x$  improvements in model capacity with only minor losses in computational efficiency and significantly advance the state-of-the-art results on public language modeling and translation data sets.
>  我们解决上述挑战，以计算效率的略微损失换来了 1000x 的模型能力/容量

## 1.2 Our Approach: The Sparsely-Gated Mixture-of-Experts Layer

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-24/99f62b99-1c34-4bc5-b208-a1ecc85307d3/09d261900ea80e10a035bf286208f4edd6c1149fea851f934d1c6c9e28ed802a.jpg)  

Figure 1: A Mixture of Experts (MoE) layer embedded within a recurrent language model. In this case, the sparse gating function selects two experts to perform computations. Their outputs are modulated by the outputs of the gating network.

Our approach to conditional computation is to introduce a new type of general purpose neural network component: a Sparsely-Gated Mixture-of-Experts Layer (MoE). The MoE consists of a number of experts, each a simple feed-forward neural network, and a trainable gating network which selects a sparse combination of the experts to process each input (see Figure 1). All parts of the network are trained jointly by back-propagation.
>  我们提出稀疏门控 MoE 层，它包含一组 experts (每个 expert 实际上就是一个 FFN)、一个可训练的门控网络 (负责为每个输入选择一个 experts 的稀疏组合)
>  该网络所有组件一起通过 BP 训练

While the introduced technique is generic, in this paper we focus on language modeling and machine translation tasks, which are known to benefit from very large models. In particular, we apply a MoE convolutionally between stacked LSTM layers (Hochreiter & Schmidhuber, 1997), as in Figure 1. The MoE is called once for each position in the text, selecting a potentially different combination of experts at each position. The different experts tend to become highly specialized based on syntax and semantics (see Appendix E Table 9). 
>  Fig1 将 MoE 嵌入到 LSTM，为文本的每个位置都可能选择一个不同的 expert 组合

On both language modeling and machine translation benchmarks, we improve on best published results at a fraction of the computational cost.

## 1.3 Related Work on Mixtures of Experts
Since its introduction more than two decades ago (Jacobs et al., 1991; Jordan & Jacobs, 1994), the mixture-of-experts approach has been the subject of much research. Different types of expert architectures have been proposed such as SVMs (Collobert et al., 2002), Gaussian Processes (Tresp, 2001; Theirs & Bethge, 2015; Deisenroth & Ng, 2015), Dirichlet Processes (Shahbaba & Neal, 2009), and deep networks. Other work has focused on different expert configurations such as a hierarchical structure (Yao et al., 2009), infinite numbers of experts (Rasmussen & Ghahramani, 2002), and adding experts sequentially (Aljundi et al., 2016). Garmash & Monz (2016) suggest an ensemble model in the format of mixture of experts for machine translation. The gating network is trained on a pre-trained ensemble NMT model.
>  这么一看，MoE 实际上和 ML 之前流行的 ensemble model 很类似

The works above concern top-level mixtures of experts. The mixture of experts is the whole model. Eigen et al. (2013) introduce the idea of using multiple MoEs with their own gating networks as parts of a deep model. It is intuitive that the latter approach is more powerful, since complex problems may contain many sub-problems each requiring different experts. They also allude in their conclusion to the potential to introduce sparsity, turning MoEs into a vehicle for computational computation.
>  上述工作考虑的是 top-level MoE，每个 expert 是整个模型
>  将 MoE with 门控函数作为网络的一部分直观上更好，因为复杂的问题可能包含多个子问题，每个子问题需要不同的 experts 解决
>  为 experts 组合引入稀疏性可以实现高的计算效率

Our work builds on this use of MoEs as a general purpose neural network component. While Eigen et al. (2013) uses two stacked MoEs allowing for two sets of gating decisions, our convolutional application of the MoE allows for different gating decisions at each position in the text. We also realize sparse gating and demonstrate its use as a practical way to massively increase model capacity.
>  我们通过卷积的方式应用 MoE，在文本的不同位置做出不同门控决策，并且我们实现了稀疏门控，以大幅提升模型能力 (并且不大幅提升模型计算量)

# 2 The Structure of The Mixture-of-Experts Layer
The Mixture-of-Experts (MoE) layer consists of a set of  $n$  "expert networks"  $E_{1},\dots ,E_{n}$  , and a "gating network"  $G$  whose output is a sparse  $n$  -dimensional vector. Figure 1 shows an overview of the MoE module. The experts are themselves neural networks, each with their own parameters. Although in principle we only require that the experts accept the same sized inputs and produce the same-sized outputs, in our initial investigations in this paper, we restrict ourselves to the case where the models are feed-forward networks with identical architectures, but with separate parameters.
>  MoE 层包含 $n$ 个专家和一个门控网络 $G$
>  $G$ 的输出是一个稀疏的 $n$ 维向量
>  专家本身也是网络，具有独立的参数
>  对专家的约束就是需要接收相同大小的输入，给出相同大小的输出
>  我们先考虑专家都是相同结构的 FFN 的情况

Let us denote by  $G(x)$  and  $E_{i}(x)$  the output of the gating network and the output of the  $i\cdot$  -th expert network for a given input  $x$  . The output  $y$  of the MoE module can be written as follows:

$$
y = \sum_{i = 1}^{n}G(x)_{i}E_{i}(x) \tag{1}
$$

>  MoE 网络的输出就是各个专家网络的输出以门控网络的输出为权重，加权求和

We save computation based on the sparsity of the output of  $G(x)$  . Wherever  $G(x)_{i} = 0$  , we need not compute  $E_{i}(x)$  . 
>  在稀疏场景下，如果门控网路的某一项为 0，我们就不需要执行对应的专家计算

In our experiments, we have up to thousands of experts, but only need to evaluate a handful of them for every example. If the number of experts is very large, we can reduce the branching factor by using a two-level hierarchical MoE. In a hierarchical MoE, a primary gating network chooses a sparse weighted combination of "experts", each of which is itself a secondary mixture-of-experts with its own gating network. In the following we focus on ordinary MoEs. We provide more details on hierarchical MoEs in Appendix B.
>  如果专家非常多，可以使用两级 MoE (两级门控) 来减少分支因子的大小
>  两级 MoE 中，主门控选择一个 “experts” 的稀疏组合，而 “experts” 本身是一个带有自己门控的 MoE

Our implementation is related to other models of conditional computation. A MoE whose experts are simple weight matrices is similar to the parameterized weight matrix proposed in (Cho & Bengio, 2014). A MoE whose experts have one hidden layer is similar to the block-wise dropout described in (Bengio et al., 2015), where the dropped-out layer is sandwiched between fully-activated layers.
>  如果 MoE 中的专家是简单的权重矩阵，MoE 就类似于参数化权重矩阵 (权重矩阵不固定，根据输入生成)
>  如果专家是有一层隐藏层的小型网络，MoE 就类似于 block-wise dropout (不是 dropout 单个神经元，而是随机 dropout 某一层)

## 2.1 Gating Network
**Softmax Gating:** A simple choice of non-sparse gating function (Jordan & Jacobs, 1994) is to multiply the input by a trainable weight matrix  $W_{g}$  and then apply the Softmax function.

$$
G_{\sigma}(x) = Softmax(x\cdot W_{g}) \tag{2}
$$

>  Softmax gating 为非稀疏门控网络，门控单元维护一个可训练的参数矩阵，该矩阵和输入相乘并做 softmax 得到各专家权重

**Noisy Top-K Gating:** We add two components to the Softmax gating network: sparsity and noise. Before taking the softmax function, we add tunable Gaussian noise, then keep only the top k values, setting the rest to  $-\infty$  (which causes the corresponding gate values to equal 0). 

$$
G(x) = Softmax(KeepTopK(H(x),k)) \tag{3}
$$

$$
H(x)_i = (x\cdot W_g)_i + StandardNormal(\cdot) \cdot Softplus((x\cdot W_{noise})_i) \tag{4}
$$

$$
K e e p T o p K(v,k)_{i} = \left\{ \begin{array}{l l}{v_{i}} & {\mathrm{if} v_{i}\mathrm{is~in~the~top} k\mathrm{~elements~of} v.}\\ {-\infty} & \mathrm{otherwise.} \end{array} \right. \tag{5}
$$

>  我们为 Softmax 门控添加稀疏性和噪声
>  在计算 Softmax 之前，我们为 $x\cdot W_g$ 添加可调节的高斯噪声，并且仅保留 top k 的值，剩余值设定为 $-\infty$ (进而使得对应的 softmax 之后的 gate value 为 0，实现了稀疏性)

>  其中的 softplus 函数是 ReLU 函数的可微分近似版本，定义为
>  $\text{Softplus}(x) = \log (1 + e^x)$
>  它在负值区域趋近于零，在正值区域趋近于 $x$，中间平滑过渡

The sparsity serves to save computation, as described above. While this form of sparsity creates some theoretically scary discontinuities in the output of gating function, we have not yet observed this to be a problem in practice. 
>  这种稀疏性在理论上创造了门控函数输出的非连续性，但在实践中暂时没有发现问题

>  非连续性即输入发生微小变化时，输出可能发生跳变，而数学优化中，梯度无法在不连续点定义，因此 BP 过程可能出现梯度震荡，导致训练不稳定
>  在实践中暂时没有问题的原因可能有:
>  1. 噪声的存在平滑了决策边界
>  2. 实际数据分布相对稳定，输入不会剧烈波动
>  3. 优化算法足够 robust，可以处理轻微的不连续

The noise term helps with load balancing, as will be discussed in Appendix A. The amount of noise per component is controlled by a second trainable weight matrix  $W_{noise}$
>  噪声项的作用是负载均衡，每个成分 (每个专家) 的噪声强度由可学习的参数矩阵 $W_{noise}$ 控制

>  如果一些专家被频繁调用，其他专家几乎不使用，就会出现热点问题
>  而加入噪声后，即使两个专家的得分相近，也可能因为噪声而出现不同的选择结果，因此噪声一定程度上避免了固定偏好

>  可学习的噪声让噪声可以适应于不同的任务、数据集和位置

**Training the Gating Network** We train the gating network by simple back-propagation, along with the rest of the model. If we choose  $k > 1$  , the gate values for the top  $k$  experts have nonzero derivatives with respect to the weights of the gating network. This type of occasionally-sensitive behavior is described in (Bengio et al., 2013) with respect to noisy rectifiers. Gradients also backpropagate through the gating network to its inputs. Our method differs here from (Bengio et al., 2015) who use boolean gates and a REINFORCE-style approach to train the gating network.
>  门控网络的训练就是简单 BP, top-k 门控值会相对于激活的专家网络具有非零的导数，梯度也会随着门控网络传播

# 3 Addressing Performance Challenges
## 3.1 The Shrinking Batch Problem
On modern CPUs and GPUs, large batch sizes are necessary for computational efficiency, so as to amortize the overhead of parameter loads and updates. If the gating network chooses  $k$  out of  $n$  experts for each example, then for a batch of  $b$  examples, each expert receives a much smaller batch of approximately  $\frac{kb}{n} \ll b$  examples. This causes a naive MoE implementation to become very inefficient as the number of experts increases. The solution to this shrinking batch problem is to make the original batch size as large as possible. However, batch size tends to be limited by the memory necessary to store activations between the forwards and backwards passes. 
>  MoE 网络面对 batch 时，每个专家收到的样本数量 $\frac {kb}n$ 通常小于 batch size $b$
>  因此专家数量增大，数据效率会下降
>  简单的解决方法是增大原始 batch size，但受到了内存大小的限制 (内存要存储一个批量的数据、激活、梯度)

>  对于每个样本，每个专家平均有 $\frac k n$ 的概率收到这个样本，可以看作每个专家平均处理 $\frac k n$ 个样本
>  因此对于 $b$ 个样本，每个专家平均处理 $\frac {kb}n$ 个样本

We propose the following techniques for increasing the batch size:
>  提高 batch size 的方法

**Mixing Data Parallelism and Model Parallelism:** In a conventional distributed training setting, multiple copies of the model on different devices asynchronously process distinct batches of data, and parameters are synchronized through a set of parameter servers. In our technique, these different batches run synchronously so that they can be combined for the MoE layer. We distribute the standard layers of the model and the gating network according to conventional data-parallel schemes, but keep only one shared copy of each expert. Each expert in the MoE layer receives a combined batch consisting of the relevant examples from all of the data-parallel input batches. 
>  混合 DP, MP
>  传统数据并行训练中，各个 replica 和 batches 异步运行，参数通过参数服务器同步
>  我们让这些 batches 同步运行: 我们仅为模型的标准层和门控网络进行分布，但仅保留一个专家的共享副本
>  这样，MoE 层的专家就能收到来自所有数据并行 batch 的相关样本

The same set of devices function as data-parallel replicas (for the standard layers and the gating networks) and as model-parallel shards (each hosting a subset of the experts). 
>  相同的一组设备既充当数据并行副本 (针对标准层和门控层)，也充当模型并行分片 (每个设备持有一组专家的子集)

If the model is distributed over  $d$  devices, and each device processes a batch of size  $b$ , each expert receives a batch of approximately  $\frac{kbd}{n}$  examples. Thus, we achieve a factor of  $d$  improvement in expert batch size.
>  这样，每个专家平均收到的 batch 大小就增大了 DP degree 倍

>  它引入更大 batch size 的目的主要是为了摊销参数加载和更新的开销，来提升计算效率，实际上它这种 DP+MP 的方式和完全 DP (把专家也完全拷贝) 的训练效果上应该是完全一致的，因为最后都会对梯度进行 AllReduce
>  因此就需要考虑这样的混合执行带来的开销和更大 batch 带来的对效率增长的权衡，毕竟 MP 一定会引入额外的通信，不过因为是稀疏结合，通讯的设备量会大大减少

In the case of a hierarchical MoE (Section B), the primary gating network employs data parallelism, and the secondary MoEs employ model parallelism. Each secondary MoE resides on one device.
>  两级 MoE 中，则主门控网络使用 DP，子 MoE 使用模型并行，每个设备上一个子 MoE

This technique allows us to increase the number of experts (and hence the number of parameters) by proportionally increasing the number of devices in the training cluster. The total batch size increases, keeping the batch size per expert constant. The memory and bandwidth requirements per device also remain constant, as do the step times, as does the amount of time necessary to process a number of training examples equal to the number of parameters in the model. It is our goal to train a trillion-parameter model on a trillion-word corpus. We have not scaled our systems this far as of the writing of this paper, but it should be possible by adding more hardware.
>  这种形式使得我们可以在增加专家数量时，通过成比例增长设备数量，使得: 总 batch size 增加、每个专家的 batch size 恒定、每个设备的带宽和内存需求恒定、模型每一步的计算时间不变、处理相当于模型参数数量的训练样本的时间也不变

**Taking Advantage of Convolutionality:** In our language models, we apply the same MoE to each time step of the previous layer. If we wait for the previous layer to finish, we can apply the MoE to all the time steps together as one big batch. Doing so increases the size of the input batch to the MoE layer by a factor of the number of unrolled time steps.
>  利用卷积性质
>  这里的卷积是指结构上的重复性和局部一致性，也就是网络对于每个时间步的计算逻辑都是相同的
>  语言模型中，输入文本会被拆分为多个时间步 (tokens)
>  如果 MoE 不存在时间步依赖，MoE 就可以等待多个时间步，然后批处理多个时间步 (然而 MoE 通常也是语言模型的一部分吧)

**Increasing Batch Size for a Recurrent MoE:** We suspect that even more powerful models may involve applying a MoE recurrently. For example, the weight matrices of a LSTM or other RNN could be replaced by a MoE. Sadly, such models break the convolutional trick from the last paragraph, since the input to the MoE at one timestep depends on the output of the MoE at the previous timestep. Gruslys et al. (2016) describe a technique for drastically reducing the number of stored activations in an unrolled RNN, at the cost of recomputing forward activations. This would allow for a large increase in batch size.
>  对于存在时间步依赖的 MoE，可以结合用于减少存储的激活数量的重计算技术，来提高 MoE 的 batch size

## 3.2 Network Bandwidth
Another major performance concern in distributed computing is network bandwidth. Since the experts are stationary (see above) and the number of gating parameters is small, most of the communication involves sending the inputs and outputs of the experts across the network. To maintain computational efficiency, the ratio of an expert's computation to the size of its input and output must exceed the ratio of computational to network capacity of the computing device. 
>  大多数通信涉及的是向专家发送输入，以及专家将输入发送出来
>  为了维护计算效率，每个专家的计算密度 (专家的计算量相对于它的输入输出大小的比值) 必须超过设备的计算能力和网络带宽的比值 (使得系统的效率由计算主导，通信开销可以忽略)

For GPUs, this may be thousands to one. In our experiments, we use experts with one hidden layer containing thousands of RELU-activated units. Since the weight matrices in the expert have sizes input_size  $\times$  hidden_size and hidden_size  $\times$  output_size, the ratio of computation to input and output is equal to the size of the hidden layer. Conveniently, we can increase computational efficiency simply by using a larger hidden layer, or more hidden layers.
>  GPU 的计算能力和通讯效率的比值可能上千，我们的试验中，专家的计算密度等于隐藏层大小，使用更大和更多的隐藏层可以提高计算密度

# 4 Balancing Expert Utilization
We have observed that the gating network tends to converge to a state where it always produces large weights for the same few experts. This imbalance is self-reinforcing, as the favored experts are trained more rapidly and thus are selected even more by the gating network. 
>  我们发现门控网络容易收敛到总是为相同的少部分专家分配大权重
>  这种不平衡是自我强化的，因为被偏好的专家会被训练得更快，进而更容易被选择

>  有点类似于 RL 中的探索缺乏，在确定某个局部空间可以达到较优的时候就陷入了这个这个局部空间
>  那么最大熵的思想或许可以迁移过来

Eigen et al. (2013) describe the same phenomenon, and use a hard constraint at the beginning of training to avoid this local minimum. Bengio et al. (2015) include a soft constraint on the batch-wise average of each gate.
>  Eigen 等人在训练初期使用硬性约束来避免这一局部最优
>  Bengio 等人为每个门控单元的 batch-wise 平均值施加了软性约束

We take a soft constraint approach. We define the importance of an expert relative to a batch of training examples to be the batchwise sum of the gate values for that expert. We define an additional loss  $L_{\text{importance}}$ , which is added to the overall loss function for the model. This loss is equal to the square of the coefficient of variation of the set of importance values, multiplied by a hand-tuned scaling factor  $w_{\text{importance}}$ . This additional loss encourages all experts to have equal importance.

$$
I m p o r t a n c e(X) = \sum_{x\in X}G(x) \tag{6}
$$

$$
L_{importance}(X) = w_{importance}\cdot CV(Importance(X))^2 \tag{7}
$$

>  我们采取软性约束方法
>  我们将一个专家相对于一个批量的训练样本的重要性定义为这批样本对于该专家的门控值的总和
>  我们定义一个额外损失 $L_{importance}$，这个损失等于重要性值集合的变异系数的平方，再乘上一个手动调优的缩放因子 $w_{importance}$
>  这个额外损失鼓励所有的专家都具有相同的重要性 (让变异系数尽量小)

>  其实这里也是一个权衡，即专家要通用还是专用，这种方式会使得一个批量的数据均匀地分散到专家中，如果批量足够 diverse，就比较适合，但如果批量偏向于同质，会导致训练出来的专家也趋向于同质

>  变异系数的定义是标准差除以平均值，它衡量了数据的相对离散程度: $CV = \frac {\sigma}{\mu}$
>  不使用方差/标准差是因为方差/标准差是绝对尺度，我们要关心的是相对不均衡程度
>  例如 `[1, 2, 3]` 和 `[100, 101, 102]` 的方差都约为 0.82，但 `[1,2,3]` 的相对不均衡尺度明显要更大，故只看方差而判断二者一样离散，是错误的
>  使用变异系数的平方应该是为了放大相对差异，并且平方项会更加平滑，凸性更好

While this loss function can ensure equal importance, experts may still receive very different numbers of examples. For example, one expert may receive a few examples with large weights, and another may receive many examples with small weights. This can cause memory and performance problems on distributed hardware. To solve this problem, we introduce a second loss function,  $L_{load}$ , which ensures balanced loads. 
>  虽然这一损失函数可以确保同等重要，但专家收到的样本数量仍然会有很大差异
>  例如一个专家可以收到少量的大权重样本，另一个专家会收到大量的小权重样本，这样重要性的变异系数也不会很大
>  这种情况会导致分布式场景下的内存和性能问题，我们引入另一个损失 $L_{load}$ 来确保负载均衡

Appendix A contains the definition of this function, along with experimental results.

# 5 Experiments
## 5.1 1 Billion Word Langauge Modeling Benchmark
**Dataset:** This dataset, introduced by (Chelba et al., 2013) consists of shuffled unique sentences from news articles, totaling approximately 829 million words, with a vocabulary of 793,471 words.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-24/99f62b99-1c34-4bc5-b208-a1ecc85307d3/560dbf6dcd2c5e1eeb41cee873c0db80a62945f55434e3c0263fb546934cf38e.jpg)  

Figure 2: Model comparison on 1-Billion-Word Language-Modeling Benchmark. On the left, we plot test perplexity as a function of model capacity for models with similar computational budgets of approximately 8-million-ops-per-timestep. On the right, we plot test perplexity as a function of computational budget. The top line represents the LSTM models from (Jozefowicz et al., 2016). The bottom line represents 4-billion parameter MoE models with different computational budgets.

**Previous State-of-the-Art:** The best previously published results (Jozefowicz et al., 2016) use models consisting of one or more stacked Long Short-Term Memory (LSTM) layers (Hochreiter & Schmidhuber, 1997; Gers et al., 2000). The number of parameters in the LSTM layers of these models vary from 2 million to 151 million. Quality increases greatly with parameter count, as do computational costs. Results for these models form the top line of Figure 2-right.

**MoE Models:** Our models consist of two stacked LSTM layers with a MoE layer between them (see Figure 1). We vary the sizes of the layers and the number of experts. For full details on model architecture, training regimen, additional baselines and results, see Appendix C.

**Low Computation, Varied Capacity:** To investigate the effects of adding capacity, we trained a series of MoE models all with roughly equal computational costs: about 8 million multiply-and-adds per training example per timestep in the forward pass, excluding the softmax layer. We call this metric (ops/timestep). We trained models with flat MoEs containing 4, 32, and 256 experts, and models with hierarchical MoEs containing 256, 1024, and 4096 experts. Each expert had about 1 million parameters. For all the MoE layers, 4 experts were active per input.
>  低计算量，多样化容量
>  我们在保持和 baseline 模型大约相同的计算开销的情况下训练一系列 MoE 模型
>  计算开销的指标定义为每个样本每次前向传播执行的乘加操作次数，称为 ops/timestep
>  所有的 MoE 层一次激活 4 个专家

The results of these models are shown in Figure 2-left. The model with 4 always-active experts performed (unsurprisingly) similarly to the computationally-matched baseline models, while the largest of the models (4096 experts) achieved an impressive  $24\%$  lower perplexity on the test set.
>  从结果可以看到，即便保持激活的专家数量不变，网络内的专家越多，就越能在保持计算开销不变的情况下，提高模型性能

**Varied Computation, High Capacity:** In addition to the largest model from the previous section, we trained two more MoE models with similarly high capacity (4 billion parameters), but higher computation budgets. These models had larger LSTMs, and fewer but larger experts. Details can be found in Appendix C.2. Results of these three models form the bottom line of Figure 2-right. 
>  高容量，多样化计算量
>  我们还训练了两个容量相似但是计算量更高的 MoE 模型，它们的 LSTM 结构更大，专家更大，但是专家数量更少

Table 1 compares the results of these models to the best previously-published result on this dataset. Even the fastest of these models beats the best published result (when controlling for the number of training epochs), despite requiring only  $6\%$  of the computation.
>  结果显示即便是计算量更低的 MoE 模型 (之前 SOTA 的 6%)，也优于之前 SOTA 的模型

Table 1: Summary of high-capacity MoE-augmented models with varying computational budgets, vs. best previously published results (Jozefowicz et al., 2016). Details in Appendix C.  

<table><tr><td></td><td>Test Perplexity 10 epochs</td><td>Test Perplexity 100 epochs</td><td>#Parameters excluding embedding and softmax layers</td><td>ops/timestep</td><td>Training Time 10 epochs</td><td>TFLOPS /GPU</td></tr><tr><td>Best Published Results</td><td>34.7</td><td>30.6</td><td>151 million</td><td>151 million</td><td>59 hours, 32 k40s</td><td>1.09</td></tr><tr><td>Low-Budget MoE Model</td><td>34.1</td><td></td><td>430 million</td><td>8.9 million</td><td>15 hours, 16 k40s</td><td>0.74</td></tr><tr><td>Medium-Budget MoE Model</td><td>31.3</td><td></td><td>4313 million</td><td>33.8 million</td><td>17 hours, 32 k40s</td><td>1.22</td></tr><tr><td>High-Budget MoE Model</td><td>28.0</td><td></td><td>4371 million</td><td>142.7 million</td><td>47 hours, 32 k40s</td><td>1.56</td></tr></table>

**Computational Efficiency:** We trained our models using TensorFlow (Abadi et al., 2016) on clusters containing 16-32 Tesla K40 GPUs. For each of our models, we determine computational efficiency in TFLOPS/GPU by dividing the number of floating point operations required to process one training batch by the observed step time and the number of GPUs in the cluster. The operation counts used here are higher than the ones we report in our ops/timestep numbers in that we include the backwards pass, we include the importance-sampling-based training of the softmax layer, and we count a multiply-and-add as two separate operations. 
> TensorFlow 训练，16-32 GPU  
>  我们将处理一个训练 batch 所需的 FLOPs 除以观察到的 step 时间 (一次训练步/step 包括了: 前向传播、计算损失、反向传播、参数更新)，再除以集群 GPU 数量，得到单位为 TFLOPS/GPU 的计算效率指标
>  这里的计算次数计数和我们上面的 ops/timestep 中对计算次数的计数不同，在这里我们还包含了反向过程的计算、softmax 层，并且还将乘和加算作两个操作

For all of our MoE models, the floating point operations involved in the experts represent between  $37\%$  and  $46\%$  of the total.
>  所有的 MoE 模型中，专家的计算量占据了 37% 到 46%

For our baseline models wtih no MoE, observed computational efficiency ranged from 1.07-1.29 TFLOPS/GPU. For our low-computation MoE models, computation efficiency ranged from 0.74-0.90 TFLOPS/GPU, except for the 4-expert model which did not make full use of the available parallelism. Our highest-computation MoE model was more efficient at 1.56 TFLOPS/GPU, likely due to the larger matrices. These numbers represent a significant fraction of the theoretical maximum of 4.29 TFLOPS/GPU claimed by NVIDIA. Detailed results are in Appendix C, Table 7.
>  低计算的 MoE 模型的 TFLOPS/GPU 略低于 baseline，高计算的 MoE 模型的 TFLOPS/GPU 略高于 baseline，都小于硬件的峰值 TFLOPS/GPU

## 5.2 100 Billion Word Google News Corpus
On the 1-billion-word corpus, adding additional capacity seems to produce diminishing returns as the number of parameters in the MoE layer exceeds 1 billion, as can be seen in Figure 2-left. We hypothesized that for a larger training set, even higher capacities would produce significant quality improvements.
>  在 1B 数据集上，MoE 层参数数量超过 1B 之后，收益减少
>  但我们认为数据集更大时，更多参数应该还会提升质量

We constructed a similar training set consisting of shuffled unique sentences from Google's internal news corpus, totalling roughly 100 billion words. Similarly to the previous section, we tested a series of models with similar computational costs of about 8 million ops/timestep. In addition to a baseline LSTM model, we trained models augmented with MoE layers containing 32, 256, 1024, 4096, 16384, 65536, and 131072 experts. This corresponds to up to 137 billion parameters in the MoE layer. Details on architecture, training, and results are given in Appendix D.
>  我们使用 100B 数据集，训练更大的 MoE 模型

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-24/99f62b99-1c34-4bc5-b208-a1ecc85307d3/04dcb564c8bcdf7b69da0d98bf13033ceb1b22a9e2418a910f3aca4e7f366f79.jpg)  

Figure 3: Language modeling on a 100 billion word corpus. Models have similar computational budgets (8 million ops/timestep).

**Results:** Figure 3 shows test perplexity as a function of capacity after training on 10 billion words (top line) and 100 billion words (bottom line). When training over the full 100 billion words, test perplexity improves significantly up to 65536 experts (68 billion parameters), dropping  $39\%$  lower than the computationally matched baseline, but degrades at 131072 experts, possibly a result of too much sparsity. The widening gap between the two lines demonstrates (unsurprisingly) that increased model capacity helps more on larger training sets.
>  在 100B 数据集上，MoE 模型达到 68B 参数时 ($2^{16} = 65536$ experts)，效果显著，但更大的模型的效果反而下降，可能的原因是稀疏性太强
>  结果显示了在更大数据集上就需要训练更大的模型，实现更强的性能

Even at 65536 experts (99.994% layer sparsity), computational efficiency for the model stays at a respectable 0.72 TFLOPS/GPU.

## 5.3 Machine Translation (Single Language Pair)
**Model Architecture:** Our model was a modified version of the GNMT model described in (Wu et al., 2016). To reduce computation, we decreased the number of LSTM layers in the encoder and decoder from 9 and 8 to 3 and 2 respectively. We inserted MoE layers in both the encoder (between layers 2 and 3) and the decoder (between layers 1 and 2). Each MoE layer contained up to 2048 experts each with about two million parameters, adding a total of about 8 billion parameters to the models. Further details on model architecture, testing procedure and results can be found in Appendix E.

**Datasets:** We benchmarked our method on the WMT'14 En  $\rightarrow$  Fr and  $\mathrm{En}\rightarrow \mathrm{De}$  corpora, whose training sets have 36M sentence pairs and 5M sentence pairs, respectively. The experimental protocols were also similar to those in (Wu et al., 2016): newstest2014 was used as the test set to compare against previous work (Luong et al., 2015a; Zhou et al., 2016; Wu et al., 2016), while the combination of newstest2012 and newstest2013 was used as the development set. We also tested the same model on Google's Production English to French data.

Table 2: Results on WMT'14 En  $\longrightarrow$  Fr newstest2014 (bold values represent best results).  

<table><tr><td>Model</td><td>Test Perplexity</td><td>Test BLEU</td><td>ops/timestep</td><td>Total #Parameters</td><td>Training Time</td></tr><tr><td>MoE with 2048 Experts</td><td>4.64</td><td>26.03</td><td>85M</td><td>8.7B</td><td>1 day/64 k40s</td></tr><tr><td>MoE with 2048 Experts (longer training)</td><td>5.25</td><td>24.91</td><td>214M</td><td>278M</td><td>1 day/96 k80s</td></tr><tr><td>GNMT (Wu et al., 2016)</td><td>8.08</td><td>24.66</td><td>214M</td><td>278M</td><td>1 day/96 k80s</td></tr><tr><td>GNMT+RL (Wu et al., 2016)</td><td>2.79</td><td>39.22</td><td>214M</td><td>278M</td><td>6 days/96 k80s</td></tr><tr><td>PBMT (Durrani et al., 2014)</td><td>2.96</td><td>39.92</td><td>214M</td><td>278M</td><td>6 days/96 k80s</td></tr><tr><td>LSTM (6-layer) (Luong et al., 2015b)</td><td></td><td>31.5</td><td></td><td></td><td></td></tr><tr><td>LSTM (6-layer+PosUnk) (Luong et al., 2015b)</td><td></td><td>33.1</td><td></td><td></td><td></td></tr><tr><td>DeepAtt (Zhou et al., 2016)</td><td></td><td>37.7</td><td></td><td></td><td></td></tr><tr><td>DeepAtt+PosUnk (Zhou et al., 2016)</td><td></td><td>39.2</td><td></td><td></td><td></td></tr></table>

Table 3: Results on WMT'14 En  $\longrightarrow$  De newstest2014 (bold values represent best results).  

<table><tr><td>Model</td><td>Test Perplexity</td><td>Test BLEU</td><td>ops/timestep</td><td>Total #Parameters</td><td>Training Time</td></tr><tr><td>MoE with 2048 Experts</td><td>4.64</td><td>26.03</td><td>85M</td><td>8.7B</td><td>1 day/64 k40s</td></tr><tr><td>GNMT (Wu et al., 2016)</td><td>5.25</td><td>24.91</td><td>214M</td><td>278M</td><td>1 day/96 k80s</td></tr><tr><td>GNMT+RL (Wu et al., 2016)</td><td>8.08</td><td>24.66</td><td>214M</td><td>278M</td><td>1 day/96 k80s</td></tr><tr><td>PBMT (Durrani et al., 2014)</td><td></td><td>20.7</td><td></td><td></td><td></td></tr><tr><td>DeepAtt (Zhou et al., 2016)</td><td></td><td>20.6</td><td></td><td></td><td></td></tr></table>

Table 4: Results on the Google Production  $\mathrm{En}\rightarrow \mathrm{Fr}$  dataset (bold values represent best results).  

<table><tr><td>Model</td><td>Eval Perplexity</td><td>Eval BLEU</td><td>Test Perplexity</td><td>Test BLEU</td><td>ops/timestep</td><td>Total #Parameters</td><td>Training Time</td></tr><tr><td>MoE with 2048 Experts</td><td>2.60</td><td>37.27</td><td>2.69</td><td>36.57</td><td>85M</td><td>8.7B</td><td>1 day/64 k40s</td></tr><tr><td>GNMT (Wu et al., 2016)</td><td>2.78</td><td>35.80</td><td>2.87</td><td>35.56</td><td>214M</td><td>278M</td><td>6 day/96 k80s</td></tr></table>

**Results:** Tables 2, 3, and 4 show the results of our largest models, compared with published results. Our approach achieved BLEU scores of 40.56 and 26.03 on the WMT'14 En→Fr and En→De benchmarks. As our models did not use RL refinement, these results constitute significant gains of 1.34 and 1.12 BLEU score on top of the strong baselines in (Wu et al., 2016). The perplexity scores are also better. On the Google Production dataset, our model achieved 1.01 higher test BLEU score even after training for only one sixth of the time.

## 5.4 MultiLingual Machine Translation
**Dataset:** (Johnson et al., 2016) train a single GNMT (Wu et al., 2016) model on a very large combined dataset of twelve language pairs. Results are somewhat worse than those for 12 separately trained single-pair GNMT models. This is not surprising, given that the twelve models have 12 times the capacity and twelve times the aggregate training of the one model. We repeat this experiment with a single MoE-augmented model. See Appendix E for details on model architecture. We train our model on the same dataset as (Johnson et al., 2016) and process the same number of training examples (about 3 billion sentence pairs). Our training time was shorter due to the lower computational budget of our model.

**Results:** Results for the single-pair GNMT models, the multilingual GNMT model and the multilingual MoE model are given in Table 5. The MoE model achieves  $19\%$  lower perplexity on the dev set than the multilingual GNMT model. On BLEU score, the MoE model significantly beats the multilingual GNMT model on 11 of the 12 language pairs (by as much as 5.84 points), and even beats the monolingual GNMT models on 8 of 12 language pairs. The poor performance on English  $\rightarrow$  Korean seems to be a result of severe overtraining, as for the rarer language pairs a small number of real examples were highly oversampled in the training corpus.

Table 5: Multilingual Machine Translation (bold values represent best results).  

<table><tr><td></td><td>GNMT-Mono</td><td>GNMT-Multi</td><td>MoE-Multi</td><td>MoE-Multi vs. GNMT-Multi</td></tr><tr><td rowspan="3">Parameters ops/timestep training time, hardware Perplexity (dev)</td><td>278M / model</td><td>278M</td><td>8.7B</td><td rowspan="3">-</td></tr><tr><td>212M</td><td>212M</td><td>102M</td></tr><tr><td>various</td><td>21 days, 96 k20s</td><td>12 days, 64 k40s</td></tr><tr><td>French → English Test BLEU</td><td>36.47</td><td>34.40</td><td>3.35</td><td>-19%</td></tr><tr><td>German → English Test BLEU</td><td>31.77</td><td>31.17</td><td>34.86</td><td>+3.06</td></tr><tr><td>Japanese → English Test BLEU</td><td>23.41</td><td>21.62</td><td>25.91</td><td>+4.63</td></tr><tr><td>Korean → English Test BLEU</td><td>25.42</td><td>22.87</td><td>28.71</td><td>+5.84</td></tr><tr><td>Portuguese → English Test BLEU</td><td>44.40</td><td>42.53</td><td>46.13</td><td>+3.60</td></tr><tr><td>Spanish → English Test BLEU</td><td>38.00</td><td>36.04</td><td>39.39</td><td>+3.35</td></tr><tr><td>English → French Test BLEU</td><td>35.37</td><td>34.00</td><td>36.59</td><td>+2.59</td></tr><tr><td>English → German Test BLEU</td><td>26.43</td><td>23.15</td><td>24.53</td><td>+1.38</td></tr><tr><td>English → Japanese Test BLEU</td><td>23.66</td><td>21.10</td><td>22.78</td><td>+1.68</td></tr><tr><td>English → Korean Test BLEU</td><td>19.75</td><td>18.41</td><td>16.62</td><td>-1.79</td></tr><tr><td>English → Portuguese Test BLEU</td><td>38.40</td><td>37.35</td><td>37.90</td><td>+0.55</td></tr><tr><td>English → Spanish Test BLEU</td><td>34.50</td><td>34.25</td><td>36.21</td><td>+1.96</td></tr></table>

# 6 Conclusion
This work is the first to demonstrate major wins from conditional computation in deep networks. We carefully identified the design considerations and challenges of conditional computing and addressed them with a combination of algorithmic and engineering solutions. While we focused on text, conditional computation may help in other domains as well, provided sufficiently large training sets. We look forward to seeing many novel implementations and applications of conditional computation in the years to come.
>  本文首次展现了条件计算在 DNN 中的优势

# Appendices
# A Load-Balancing Loss
As discussed in section 4, for load-balancing purposes, we want to define an additional loss function to encourage experts to receive roughly equal numbers of training examples. Unfortunately, the number of examples received by an expert is a discrete quantity, so it can not be used in backpropagation. 
>  为了负载平衡，我们定义了额外的损失来鼓励专家收到大致相同数量的训练样本
>  但专家收到的样本数量为离散值，不能用于反向传播

Instead, we define a smooth estimator  $Load(X)$  of the number of examples assigned to each expert for a batch  $X$  of inputs. The smoothness allows us to back-propagate gradients through the estimator. This is the purpose of the noise term in the gating function. 
>  我们定义平滑估计器 $Load (X)$，对于 $X$ 个输入的 batch，估计分配给每个专家的样本数量，平滑性质使得可以进行反向传播
>  门控函数中的噪声项的目的就是平滑性质

We define  $P(x,i)$  as the probability that  $G(x)_i$  is nonzero, given a new random choice of noise on element  $i$  but keeping the already-sampled choices of noise on the other elements. 
>  我们定义 $P (x, i)$ 为: 保持对其他样本已经采样的噪声不变，为元素 $i$ 重新随机选择噪声后，$G (x)_i$ 非零的概率

To compute  $P(x,i)$  , we note that the  $G(x)_i$  is nonzero if and only if  $H(x)_i$  is greater than the  $k^{th}$  - greatest element of  $H(x)$  excluding itself. The probability works out to be:

$$
\begin{array}{r}P(x,i) = Pr\Big((x\cdot W_g)_i + StandardNormal()\cdot Softplus(x\cdot W_{noise})_i)\\ >kth\_ excluding(H(x),k,i)\Big) \end{array} \tag{8}
$$

Where $kth\_excluding (v,k,i)$  means the kth highest component of  $v$  , excluding component  $i$  . 
>  为了计算 $P (x, i)$，注意到 $G (x)_i$ 非零当且仅当 $H (x)_i$ 大于 $H (x)$ 中除了自己后的第 $k$ 大的元素，因此 $P (x, i)$ 可以按照 Eq 8 定义，即定义为该事件发生的概率

Simplifying, we get:

$$
P(x,i) = \Phi \Big(\frac{(x\cdot W_g)_i -kth\_excluding(H(x),k,i)}{Softplus((x\cdot W_{noise})_i)}\Big) \tag{9}
$$

Where  $\Phi$  is the CDF of the standard normal distribution.

>  对 Eq 8 进行简化，得到 Eq 9，其中 $\Phi$ 表示标准正态分布的累积分布函数


$$
L o a d(X)_{i} = \sum_{x\in X}P(x,i) \tag{10}
$$

>  我们进而通过 $P (x, i)$ 定义平滑估计器 $Load (X)$，给定数据 $X$，对第 $i$ 个专家的平滑估计项就是对 $X$ 中所有 $P (x, i)$ 求和，它大致表示了专家 $i$ 对于在给定数据 $X$ 时能够收到的样本的平均数量

We can now define the load loss to be the square of the coefficient of variation of the load vector, multiplied by a hand-tuned scaling factor  $w_{load}$

$$
L_{load}(X) = w_{load}\cdot CV(L_{load}(X))^{2} \tag{11}
$$

>  我们将负载损失定义为 load vector 的变异系数平方，以惩罚过度不均匀的情况，并乘上一个系数 $w_{load}$

**Initial Load Imbalance:** To avoid out-of-memory errors, we need to initialize the network in a state of approximately equal expert load (since the soft constraints need some time to work). To accomplish this, we initialize the matrices  $W_{g}$  and  $W_{noise}$  to all zeros, which yields no signal and some noise.
>  网络初始化为均匀的专家负载，故 $W_g, W_{noise}$ 都初始化为零，即一开始仅根据标准高斯噪声进行选择

**Experiments:** We trained a set of models with identical architecture (the MoE-256 model described in Appendix C), using different values of  $w_{importance}$  and  $w_{load}$ . We trained each model for 10 epochs, then measured perplexity on the test set. We also measured the coefficients of variation in Importance and Load, as well as ratio of the load on the most overloaded expert to the average load. This last value is significant for load balancing purposes on distributed hardware. All of these metrics were averaged over several training batches.

Table 6: Experiments with different combinations of losses.  

<table><tr><td>wimportance</td><td>wload</td><td>Test Perplexity</td><td>CV (Importance(X))</td><td>CV (Load(X))</td><td>max(Load(X)) / mean(Load(X))</td></tr><tr><td>0.0</td><td>0.0</td><td>39.8</td><td>3.04</td><td>3.01</td><td>17.80</td></tr><tr><td>0.2</td><td>0.0</td><td>35.6</td><td>0.06</td><td>0.17</td><td>1.47</td></tr><tr><td>0.0</td><td>0.2</td><td>35.7</td><td>0.22</td><td>0.04</td><td>1.15</td></tr><tr><td>0.1</td><td>0.1</td><td>35.6</td><td>0.06</td><td>0.05</td><td>1.14</td></tr><tr><td>0.01</td><td>0.01</td><td>35.7</td><td>0.48</td><td>0.11</td><td>1.37</td></tr><tr><td>1.0</td><td>1.0</td><td>35.7</td><td>0.03</td><td>0.02</td><td>1.07</td></tr></table>

**Results:** Results are reported in Table 6. All the combinations containing at least one the two losses led to very similar model quality, where having no loss was much worse. Models with higher values of  $w_{load}$  had lower loads on the most overloaded expert.
>  结果显示，单独使用软性约束 (Importance) 或负载均衡 (Load) 或者一起使用，效果都比两个都不使用好得多
>  $w_{load}$ 越高，负载最高的 expert 上的 load 越低

# B Hierachical Mixture of Experts
If the number of experts is very large, we can reduce the branching factor by using a two-level hierarchical MoE. In a hierarchical MoE, a primary gating network chooses a sparse weighted combination of "experts", each of which is itself a secondary mixture-of-experts with its own gating network. 

If the hierarchical MoE consists of  $a$  groups of  $b$  experts each, we denote the primary gating network by  $G_{primary}$ , the secondary gating networks by  $(G_{1},G_{2}.G_{a})$ , and the expert networks by  $(E_{0,0},E_{0,1}..E_{a,b})$ . The output of the MoE is given by:

$$
y_{H} = \sum_{i = 1}^{a}\sum_{j = 1}^{b}G_{primary}(x)_{i}\cdot G_{i}(x)_{j}\cdot E_{i,j}(x) \tag{12}
$$

Our metrics of expert utilization change to the following:

$$
I m p o r t a n c e_{H}(X)_{i,j} = \sum_{x\in X}G_{p r i m a r y}(x)_{i}\cdot G_{i}(x)_{j} \tag{13}
$$

$$
Load_{H}(X)_{i,j} = \frac{Load_{primary}(X)_{i}\cdot Load_{i}(X^{(i)})_{j}}{|X^{(i)}|} \tag{14}
$$

$Load_{primary}$  and  $Load_{i}$  denote the Load functions for the primary gating network and  $i^{th}$  secondary gating network respectively.  $X^{(i)}$  denotes the subset of  $X$  for which  $G_{primary}(x)_{i} > 0$

>  对于两级 MoE，可以为主门控网络和次级门控网络分别定义 Load function

It would seem simpler to let  $Load_{H}(X)_{i,j} = Load_{i}(X_{i})_{j}$ , but this would not have a gradient with respect to the primary gating network, so we use the formulation above.

## C.1 8-Million-Operations-Per-Timestep Models
**Model Architecture:** Our model consists of five layers: a word embedding layer, a recurrent Long Short-Term Memory (LSTM) layer (Hochreiter & Schmidhuber, 1997; Gers et al., 2000), a MoE layer, a second LSTM layer, and a softmax layer. The dimensionality of the embedding layer, the number of units in each LSTM layer, and the input and output dimensionality of the MoE layer are all equal to 512. For every layer other than the softmax, we apply dropout (Zaremba et al., 2014) to the layer output, dropping each activation with probability  $DropProb$ , otherwise dividing by  $(1 -DropProb)$ . After dropout, the output of the previous layer is added to the layer output. This residual connection encourages gradient flow (He et al., 2015).

**MoE Layer Architecture:** Each expert in the MoE layer is a feed forward network with one ReLU-activated hidden layer of size 1024 and an output layer of size 512. Thus, each expert contains  $[512*1024] + [1024*512] = 1M$  parameters. The output of the MoE layer is passed through a sigmoid function before dropout. We varied the number of experts between models, using ordinary MoE layers with 4, 32 and 256 experts and hierarchical MoE layers with 256, 1024 and 4096 experts. We call the resulting models MoE-4, MoE-32, MoE-256, MoE-256-h, MoE-1024-h and MoE-4096-h. For the hierarchical MoE layers, the first level branching factor was 16, corresponding to the number of GPUs in our cluster. We use Noisy-Top-K Gating (see Section 2.1) with  $k = 4$  for the ordinary MoE layers and  $k = 2$  at each level of the hierarchical MoE layers. Thus, each example is processed by exactly 4 experts for a total of 4M ops/timestep. The two LSTM layers contribute 2M ops/timestep each for the desired total of 8M.

**Computationally-Matched Baselines:** The MoE-4 model does not employ sparsity, since all 4 experts are always used. In addition, we trained four more computationally-matched baseline models with no sparsity:

- MoE-1-Wide: The MoE layer consists of a single "expert" containing one ReLU-activated hidden layer of size 4096.
- MoE-1-Deep: The MoE layer consists of a single "expert" containing four ReLU-activated hidden layers, each with size 1024. .
- 4xLSTM-512: We replace the MoE layer with two additional 512-unit LSTM layers. . 
- LSTM-2048-512: The model contains one 2048-unit LSTM layer (and no MoE). The output of the LSTM is projected down to 512 dimensions (Sak et al., 2014). The next timestep of the LSTM receives the projected output. This is identical to one of the models published in (Jozefowicz et al., 2016). We re-ran it to account for differences in training regimen, and obtained results very similar to the published ones.

**Training:** The models were trained on a cluster of 16 K40 GPUs using the synchronous method described in Section 3. Each batch consisted of a set of sentences totaling roughly 300,000 words. In the interest of time, we limited training to 10 epochs, (27,000 steps). Training took 12-16 hours for all models, except for MoE-4, which took 18 hours (since all the expert computation was performed on only 4 of 16 GPUs). We used the Adam optimizer (Kingma & Ba, 2015). The base learning rate was increased linearly for the first 1000 training steps, and decreased after that so as to be proportional to the inverse square root of the step number. 
>  16 GPU，12-16 小时训练
>  Adam，学习率线性 warmup，平方下降

The Softmax output layer was trained efficiently using importance sampling similarly to the models in (Jozefowicz et al., 2016). For each model, we performed a hyper-parameter search to find the best dropout probability, in increments of 0.1.
>  Softmax 输出层使用重要性采样高效训练
>  重要性采样是在不直接计算全部可能性的情况下近似估计某个期望值或损失
>  如果词表很大，Softmax 需要对很多词求指数和，计算成本高
>  重要性采样随机采样一部分负样本，即更容易被误判的样本，用这些采样项来近似真实损失，加速训练

To ensure balanced expert utilization we set  $w_{importance} = 0.1$  and  $w_{load} = 0.1$  , as described in Section 4 and Appendix A.

**Results:** We evaluate our model using perplexity on the holdout dataset, used by (Chelba et al., 2013; Jozefowicz et al., 2016). We follow the standard procedure and sum over all the words including the end of sentence symbol. Results are reported in Table 7. For each model, we report the test perplexity, the computational budget, the parameter counts, the value of DropProb, and the computational efficiency.

Table 7: Model comparison on 1 Billion Word Language Modeling Benchmark. Models marked with \* are from (Jozefowicz et al., 2016).  

<table><tr><td>Model</td><td>Test Perplexity 10 epochs</td><td>Test Perplexity (final)</td><td>ops/timestep (millions)</td><td>#Params excluding embed. &amp;amp; softmax (millions)</td><td>Total #Params (billions)</td><td>Drop-Prob</td><td>TFLOPS per GPU (observed)</td></tr><tr><td>Kneser-Ney 5-gram*</td><td></td><td>57.6</td><td>0.00001</td><td></td><td>1.8</td><td></td><td></td></tr><tr><td>LSTM-512-512*</td><td></td><td>64.1</td><td>2.4</td><td></td><td>0.8</td><td>0.1</td><td></td></tr><tr><td>LSTM-1024-512*</td><td></td><td>48.2</td><td>4.7</td><td>4.7</td><td>0.8</td><td>0.1</td><td></td></tr><tr><td>LSTM-2048-512*</td><td>45.0</td><td>43.7</td><td>9.4</td><td>9.4</td><td>0.8</td><td>0.1</td><td>0.61</td></tr><tr><td>LSTM-2048-512</td><td>44.7</td><td></td><td>9.4</td><td>9.4</td><td>0.8</td><td>0.1</td><td>1.21</td></tr><tr><td>4xLSTM-512</td><td>46.0</td><td></td><td>8.4</td><td>8.4</td><td>0.8</td><td>0.1</td><td>1.07</td></tr><tr><td>MoE-1-Wide</td><td>46.1</td><td></td><td>8.4</td><td>8.4</td><td>0.8</td><td>0.1</td><td>1.29</td></tr><tr><td>MoE-1-Deep</td><td>45.7</td><td></td><td>8.4</td><td>8.4</td><td>0.8</td><td>0.1</td><td>1.29</td></tr><tr><td>MoE-4</td><td>45.0</td><td></td><td>8.4</td><td>8.4</td><td>0.8</td><td>0.1</td><td>0.52</td></tr><tr><td>MoE-32</td><td>39.7</td><td></td><td>8.4</td><td>37.8</td><td>0.9</td><td>0.1</td><td>0.87</td></tr><tr><td>MoE-256</td><td>35.7</td><td></td><td>8.6</td><td>272.9</td><td>1.1</td><td>0.1</td><td>0.81</td></tr><tr><td>MoE-256-h</td><td>36.0</td><td></td><td>8.4</td><td>272.9</td><td>1.1</td><td>0.1</td><td>0.89</td></tr><tr><td>MoE-1024-h</td><td>34.6</td><td></td><td>8.5</td><td>1079.0</td><td>1.9</td><td>0.2</td><td>0.90</td></tr><tr><td>MoE-4096-h</td><td>34.1</td><td></td><td>8.9</td><td>4303.4</td><td>5.1</td><td>0.2</td><td>0.74</td></tr><tr><td>2xLSTM-8192-1024*</td><td>34.7</td><td>30.6</td><td>151.0</td><td>151.0</td><td>1.8</td><td>0.25</td><td>1.09</td></tr><tr><td>MoE-34M</td><td>31.3</td><td></td><td>33.8</td><td>4313.9</td><td>6.0</td><td>0.3</td><td>1.22</td></tr><tr><td>MoE-143M</td><td>28.0</td><td></td><td>142.7</td><td>4371.1</td><td>6.0</td><td>0.4</td><td>1.56</td></tr></table>

## C.2 More Expensive Models
We ran two additional models (MoE-34M and MoE-143M) to investigate the effects of adding more computation in the presence of a large MoE layer. These models have computation budgets of 34M and 143M ops/timestep. Similar to the models above, these models use a MoE layer between two LSTM layers. The dimensionality of the embedding layer, and the input and output dimensionality of the MoE layer are set to 1024 instead of 512. For MoE-34M, the LSTM layers have 1024 units. For MoE-143M, the LSTM layers have 4096 units and an output projection of size 1024 (Sak et al., 2014). MoE-34M uses a hierarchical MoE layer with 1024 experts, each with a hidden layer of size 2048. MoE-143M uses a hierarchical MoE layer with 256 experts, each with a hidden layer of size 8192. Both models have 4B parameters in the MoE layers. We searched for the best DvspzPro for each model, and trained each model for 10 epochs.

The two models achieved test perplexity of 31.3 and 28.0 respectively, showing that even in the presence of a large MoE, more computation is still useful. Results are reported at the bottom of Table 7. The larger of the two models has a similar computational budget to the best published model from the literature, and training times are similar. Comparing after 10 epochs, our model has a lower test perplexity by  $18\%$
>  在专家数量很多时，增加专家的参数量也是有益的

# D 100 Billion Word Google News Corpus - Epxerimental Details
**Model Architecture:** The models are similar in structure to the 8-million-operations-per-timestep models described in the previous section. We vary the number of experts between models, using an ordinary MoE layer with 32 experts and hierarchical MoE layers with 256, 1024, 4096, 16384, 65536 and 131072 experts. For the hierarchical MoE layers, the first level branching factors are 32, 32, 64, 128, 256 and 256, respectively.

**Training:** Models are trained on a cluster of 32 Tesla K40 GPUs, except for the last two models, which are trained on clusters of 64 and 128 GPUs so as to have enough memory for all the parameters. For all models, training batch sizes are approximately 2.5 million words. Models are trained once-through over about 100 billion words.

We implement several memory optimizations in order to fit up to 1 billion parameters per GPU. First, we do not store the activations of the hidden layers of the experts, but instead recompute them on the backwards pass. 
>  使用重计算减少显存占用

Secondly, we modify the optimizer on the expert parameters to require less auxiliary storage:

The Adam optimizer (Kingma & Ba, 2015) keeps first and second moment estimates of the per-parameter gradients. This triples the required memory. To avoid keeping a first-moment estimator, we set  $\beta_{1} = 0$  . To reduce the size of the second moment estimator, we replace it with a factored approximation. For a matrix of parameters, instead of maintaining a full matrix of second-moment estimators, we maintain vectors of row-wise and column-wise averages of that matrix. At each step, the matrix of estimators is taken to be the outer product of those two vectors divided by the mean of either one. This technique could similarly be applied to Adagrad (Duchi et al., 2010).
>  修改对专家参数的优化器以减少显存占用
>  Adam 会保存参数梯度的一阶和二阶矩估计，导致需要三倍显存
>  为了避免保存一阶矩估计，我们设置 $\beta_t$，使得 $m_t$ 就等于当前梯度，即使用当前梯度表示历史平均梯度
>  为了减少二阶矩估计的大小，我们使用**分解近似**替代它，对于参数矩阵，我们维护该矩阵的平均行向量和平均列向量，整个矩阵近似为这两个向量的外积除以二者之一的均值

>  一阶矩估计是梯度的指数移动平均，记作 $m_t$
>  二阶矩估计是梯度平方的指数移动平均，记作 $v_t$

Table 8: Model comparison on 100 Billion Word Google News Dataset  

<table><tr><td>Model</td><td>Test 
Perplexity 
.1 epochs</td><td>Test 
Perplexity 
1 epoch</td><td>ops/timestep (millions)</td><td>#Params excluding closed &amp;amp; softmax (millions)</td><td>Total #Params (billions)</td><td>TFLOPS per GPU (observed)</td></tr><tr><td>Kneser-Ney 5-gram</td><td>67.1</td><td>45.3</td><td>0.00001</td><td></td><td>76.0</td><td></td></tr><tr><td>4x LSTM-512</td><td>54.5</td><td>47.0</td><td>8.4</td><td>8.4</td><td>0.1</td><td>1.23</td></tr><tr><td>MoE-32</td><td>48.5</td><td>40.4</td><td>8.4</td><td>37.8</td><td>0.1</td><td>0.83</td></tr><tr><td>MoE-256-h</td><td>42.8</td><td>35.3</td><td>8.4</td><td>272.9</td><td>0.4</td><td>1.11</td></tr><tr><td>MoE-1024-h</td><td>40.3</td><td>32.7</td><td>8.5</td><td>1079.0</td><td>1.2</td><td>1.14</td></tr><tr><td>MoE-4096-h</td><td>38.9</td><td>30.9</td><td>8.6</td><td>4303.4</td><td>4.4</td><td>1.07</td></tr><tr><td>MoE-16384-h</td><td>38.2</td><td>29.7</td><td>8.8</td><td>17201.0</td><td>17.3</td><td>0.96</td></tr><tr><td>MoE-65536-h</td><td>38.2</td><td>28.9</td><td>9.2</td><td>68791.0</td><td>68.9</td><td>0.72</td></tr><tr><td>MoE-131072-h</td><td>39.8</td><td>29.2</td><td>9.7</td><td>137577.6</td><td>137.7</td><td>0.30</td></tr></table>

**Results:** We evaluate our model using perplexity on a holdout dataset. Results are reported in Table 8. Perplexity after 100 billion training words is  $39\%$  lower for the 68-billion-parameter MoE model than for the baseline model. It is notable that the measured computational efficiency of the largest model (0.30 TFLOPS/GPU) is very low compared to the other models. This is likely a result of the fact that, for purposes of comparison to the other models, we did not increase the training batch size proportionally to the number of GPUs. For comparison, we include results for a computationally matched baseline model consisting of 4 LSTMs, and for an unpruned 5-gram model with Kneser-Ney smoothing (Kneser & Ney, 1995).

# E Machine Translation - Experimental Details
**Model Architecture for Single Language Pair MoE Models:** Our model is a modified version of the GNMT model described in (Wu et al., 2016). To reduce computation, we decrease the number of LSTM layers in the encoder and decoder from 9 and 8 to 3 and 2 respectively. We insert MoE layers in both the encoder (between layers 2 and 3) and the decoder (between layers 1 and 2). We use an attention mechanism between the encoder and decoder, with the first decoder LSTM receiving output from and providing input for the attention. All of the layers in our model have input and output dimensionality of 512. Our LSTM layers have 2048 hidden units, with a 512-dimensional output projection. We add residual connections around all LSTM and MoE layers to encourage gradient flow (He et al., 2015). Similar to GNMT, to effectively deal with rare words, we used subword units (also known as "wordpieces") (Schuster & Nakajima, 2012) for inputs and outputs in our system.

We use a shared source and target vocabulary of 32K wordpieces. We also used the same beam search technique as proposed in (Wu et al., 2016).

We train models with different numbers of experts in the MoE layers. In addition to a baseline model with no MoE layers, we train models with flat MoE layers containing 32 experts, and models with hierarchical MoE layers containing 512 and 2048 experts. The flat MoE layers use  $k = 4$  and the hierarchical MoE models use  $k = 2$  at each level of the gating network. Thus, each input is processed by exactly 4 experts in each MoE layer. Each expert in the MoE layer is a feed forward network with one hidden layer of size 2048 and ReLU activation. Thus, each expert contains  $[512*2048] + [2048*512] = 2M$  parameters. The output of the MoE layer is passed through a sigmoid function. We use the strictly-balanced gating function described in Appendix F.

**Model Architecture for Multilingual MoE Model:** We used the same model architecture as for the single-language-pair models, with the following exceptions: We used noisy-top-k gating as described in Section 2.1, not the scheme from Appendix F. The MoE layers in the encoder and decoder are non-hierarchical MoEs with  $n = 512$  experts, and  $k = 2$ . Each expert has a larger hidden layer of size 8192. This doubles the amount of computation in the MoE layers, raising the computational budget of the entire model from 85M to 102M ops/timestep.

**Training:** We trained our networks using the Adam optimizer (Kingma & Ba, 2015). The base learning rate was increased linearly for the first 2000 training steps, held constant for an additional 8000 steps, and decreased after that so as to be proportional to the inverse square root of the step number. For the single-language-pair models, similarly to (Wu et al., 2016), we applied dropout (Zaremba et al., 2014) to the output of all embedding, LSTM and MoE layers, using  $DropProb = 0.4$ . Training was done synchronously on a cluster of up to 64 GPUs as described in section 3. Each training batch consisted of a set of sentence pairs containing roughly 16000 words per GPU.

To ensure balanced expert utilization we set  $w_{importance} = 0.01$  and  $w_{load} = 0.01$ , as described in Section 4 and Appendix A.

**Metrics:** We evaluated our models using the perplexity and the standard BLEU score metric. We reported tokenized BLEU score as computed by the multi-bleu.pl script, downloaded from the public implementation of Moses (on Github), which was also used in (Luong et al., 2015a).

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-24/99f62b99-1c34-4bc5-b208-a1ecc85307d3/c7d601a4905f86e2cf442dd8bde061e29e9fa7e1f0902d599fe53f2d38c9f133.jpg)  

Figure 4: Perplexity on WMT'14 En  $\longrightarrow$  Fr (left) and Google Production  $\mathrm{En}\rightarrow \mathrm{Fr}$  (right) datasets as a function of number of words processed. The large differences between models at the beginning of training are due to different batch sizes. All models incur the same computational budget (85M ops/timestep) except the one with no experts.

**Results:** Tables 2, 3 and 4 in Section 5.3 show comparisons of our results to other published methods. Figure 4 shows test perplexity as a function of number of words in the (training data's) source sentences processed for models with different numbers of experts. As can be seen from the Figure, as we increased the number of experts to approach 2048, the test perplexity of our model continued to improve.
>  我们发现专家数量增长，模型能力增长

We found that the experts indeed become highly specialized by syntax and/or semantics, as can be seen in Table 9. For example, one expert is used when the indefinite article "a" introduces the direct object in a verb phrase indicating importance or leadership.
>  我们发现专家会在语法和语义上高度专业化，例如当不定冠词 `a` 出现在表示重要性或领导力的动词短语中引入宾语时，会使用到某个特定的专家 `

Table 9. Contexts corresponding to a few of the 2048 experts in the MoE layer in the encoder portion of the WMT'14  $\mathrm{En}\rightarrow \mathrm{Fr}$  translation model. For each expert  $i$  we sort the inputs in a training batch in decreasing order of  $G(x)_{i}$  , and show the words surrounding the corresponding positions in the input sentences.  

<table><tr><td>Expert 381</td><td>Expert 752</td><td>Expert 2004</td></tr><tr><td>... with researchers, ... 
... to innovation 
... tics researchers 
... the generation of ... 
... technology innovations is ... 
... technological innovations, ... 
... support innovation throughout ... 
... hole innovation will ... 
... research scientist ... 
... promoting innovation where ... 
...</td><td>... plays a core ... 
... plays a critical ... 
... provides a legislative ... 
... play a leading ... 
... assume a leadership ... 
... plays a central ... 
... take a leading ... 
... established a reconciliation ... 
... played a vital ... 
... have a central ...</td><td>... with rapidly growing ... 
... under static conditions ... 
... to swift ly ... 
... to drastically ... 
... the rapid and ... 
... the fast est ... 
... the Quick Method ... 
... recurrent) ... 
... provides quick access ... 
... of volatile organic ... 
...</td></tr></table>

# F Strictly Balanced Gating
Due to some peculiarities in our infrastructure which have since been fixed, at the time we ran some of the machine translation experiments, our models ran faster if every expert received exactly the same batch size. To accommodate this, we used a different gating function which we describe below.

Recall that we define the softmax gating function to be:

$$
G_{\sigma}(x) = S o f t m a x(x\cdot W_{g}) \tag{15}
$$

**Sparse Gating (alternate formulation):** To obtain a sparse gating vector, we multiply  $G_{\sigma}(x)$  component-wise with a sparse mask  $M(G_{\sigma}(x))$  and normalize the output. The mask itself is a function of  $G_{\sigma}(x)$  and specifies which experts are assigned to each input example:

$$
G(x)_i = \frac{G_\sigma(x)_iM(G_\sigma(x))_i}{\sum_{j = 1}^nG_\sigma(x)_jM(G_\sigma(x))_j} \tag{16}
$$

>  稀疏门控: 我们为 softmax 门控函数的输出按元素乘上一个稀疏掩码 $M (G_\sigma (x))$，然后规范化输出
>  稀疏掩码 $M (G_\sigma (x))$ 本身是一个关于 $G_\sigma (x)$ 的函数，对于每个输入样本要分配给哪个专家

**Top-K Mask:** To implement top-k gating in this formulation, we would let  $M(v) = TopK(v,k)$ , where:

$$
TopK(v,k)_i = \left\{ \begin{array}{ll}1 & \mathrm{if}v_i\mathrm{is~in~the~top}k\mathrm{~elements~of}v.\\ 0 & \mathrm{otherwise}. \end{array} \right. \tag{17}
$$

>  如果要实现 top-k 掩码，将掩码函数定义为 top-k 掩码即可

**Batchwise Mask:** To force each expert to receive the exact same number of examples, we introduce an alternative mask function,  $M_{batchwise}(X,m)$ , which operates over batches of input vectors. Instead of keeping the top  $k$  values per example, we keep the top  $m$  values per expert across the training batch, where  $m = \frac{k|X|}{n}$ , so that each example is sent to an average of  $k$  experts.

$$
M_{batchwise}(X,m)_{j,i} = \left\{ \begin{array}{ll}1 & \mathrm{if}X_{j,i}\mathrm{is~in~the~top}m\mathrm{values~for~to~expert}i\\ 0 & \mathrm{otherwise} \end{array} \right. \tag{18}
$$

>  要让每个专家收到的样本数量都完全相同，我们引入另一个掩码函数 $M_{batchwise}(X, m)$，该函数接收一个 batch 的输入向量
>  该函数不是为每个样本 (的门控向量) 保留 top-k 个值，而是为每个专家保留 batch 中的 top-m 个值

>  例如给定一个矩阵，每一行对应 batch 中一个样本的门控向量 $G_\sigma (x)$，行数等于 batch size
>  这个矩阵的列数是专家数量
>  如果使用 top-k，那就是为每一行选出 top-k 个值
>  如果使用上面介绍的 batchwise mask，那就是为每个列选出 top-m 个值，$m = \frac {k\times \text{batch size}}{\text{expert num}}$，也就是和 top-k 情况下，batch 内总的激活专家数量保持一致，那么每个样本实际上还是平均发送给 k 个专家

As our experiments suggest and also observed in (Ioffe & Szegedy, 2015), using a batchwise function during training (such as  $M_{batchwise}$ ) requires modifications to the inference when we may not have a large batch of examples. 
>  但是在训练时使用 batchwise mask 利用了训练时使用大 batch 样本的性质
>  在推理时，我们则通常不会有大 batch 样本，因此还需要对 mask 函数做一定修改

Our solution to this is to train a vector  $T$  of per-expert threshold values to approximate the effects of the batchwise mask. We use the following mask at inference time:

$$
M_{threshold}(x,T)_i = \left\{ \begin{array}{ll}1 & \mathrm{if}x_i > T_i\\ 0 & \mathrm{otherwise} \end{array} \right. \tag{19}
$$

>  我们的方法是训练一个阈值向量 $T$，在推理时，如果样本对于该专家的门控值超过了该专家的阈值 $T_i$，就激活该专家

To learn the threshold values, we apply an additional loss at training time which is minimized when the batchwise mask and the threshold mask are identical.

$$
L_{batchwise}(X,T,m) = \sum_{j = 1}^{|X|}\sum_{i = 1}^{n}(M_{threshold}(x,T)_i -M_{batchwise}(X,m)_{j,i})(X_{j,i} -T_i) \tag{20}
$$

>  为了学习阈值向量，我们使用一个额外损失，它让阈值向量给出的掩码尽量接近 batchwise mask 给出的掩码
>  其中 $X_{j, i} - T_i$ 类似于给 error 加权，大致意思是样本值高出阈值越多，就越强化它被激活的可能性

# G Attention Function
The attention mechanism described in GNMT (Wu et al., 2016) involves a learned "Attention Function"  $A(x_{i},y_{j})$  which takes a "source vector"  $x_{i}$  and a "target vector"  $y_{j}$ , and must be computed for every source time step  $i$  and target time step  $j$ . In GNMT, the attention function is implemented as a feed forward neural network with a hidden layer of size  $n$ . It can be expressed as:

$$
A_{GNMT}(x_i,y_j) = \sum_{d = 1}^n V_d\tanh ((x_iU)_d + (y_jW)_d) \tag{21}
$$

Where  $U$  and  $W$  are trainable weight matrices and  $V$  is a trainable weight vector.

For performance reasons, in our models, we used a slightly different attention function:

$$
A(x_{i},y_{j}) = \sum_{d = 1}^{n}V_{d}\tanh ((x_{i}U)_{d})\tanh ((y_{j}W)_{d}) \tag{22}
$$

With our attention function, we can simultaneously compute the attention function on multiple source time steps and multiple target time steps using optimized matrix multiplications. We found little difference in quality between the two functions.
