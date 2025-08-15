# Abstract
Scaling up deep neural network capacity has been known as an effective approach to improving model quality for several different machine learning tasks. In many cases, increasing model capacity beyond the memory limit of a single accelerator has required developing special algorithms or infrastructure. These solutions are often architecture-specific and do not transfer to other tasks. 

To address the need for efficient and task-independent model parallelism, we introduce GPipe, a pipeline parallelism library that allows scaling any network that can be expressed as a sequence of layers. By pipelining different sub-sequences of layers on separate accelerators, GPipe provides the flexibility of scaling a variety of different networks to gigantic sizes efficiently. 
>  为了解决对高效且任务无关的模型并行性的需求，我们提出 GPipe，这是一个流水线并行库，允许将任何可以表示为一系列层的网络进行拓展

Moreover, GPipe utilizes a novel batch splitting pipelining algorithm, resulting in almost linear speedup when a model is partitioned across multiple accelerators. 
>  此外，GPipe 使用了一种 batch splitting 流水线算法，在将模型划分到多个加速器上时，可以实现接近线性的加速效果

We demonstrate the advantages of GPipe by training large-scale neural networks on two different tasks with distinct network architectures: (i) Image Classification: We train a 557-million-parameter AmoebaNet model and attain a top-1 accuracy of  $84.4\%$  on ImageNet-2012, (ii) Multilingual Neural Machine Translation: We train a single 6-billion-parameter, 128-layer Transformer model on a corpus spanning over 100 languages and achieve better quality than all bilingual models.

# 1 Introduction
Deep learning has seen great progress over the last decade, partially thanks to the development of methods that have facilitated scaling the effective capacity of neural networks. This trend has been most visible for image classification, as demonstrated by the accuracy improvements on ImageNet with the increase in model capacity (Figure 1a). A similar phenomenon can also be observed in the context of natural language processing (Figure 1b) where simple shallow models of sentence representations [1, 2] are outperformed by their deeper and larger counterparts [3, 4].

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-12/827e9d60-47db-4482-b695-9005bb3c6a97/2c19911e1e3c36c3260b6bd869c5f30002572fd95dd62dc24ea3eef589c63d70.jpg)  

Figure 1: (a) Strong correlation between top-1 accuracy on ImageNet 2012 validation dataset [5] and model size for representative state-of-the-art image classification models in recent years [6, 7, 8, 9, 10, 11, 12]. There has been a  $36\times$  increase in the model capacity. Red dot depicts  $84.4\%$  top-1 accuracy for the 550M parameter AmoebaNet model. (b) Average improvement in translation quality (BLEU) compared against bilingual baselines on our massively multilingual in-house corpus, with increasing model size. Each point,  $T(L,H,A)$ , depicts the performance of a Transformer with  $L$  encoder and  $L$  decoder layers, a feed-forward hidden dimension of  $H$  and  $A$  attention heads. Red dot depicts the performance of a 128-layer 6B parameter Transformer.

While larger models have brought remarkable quality improvements to several fields, scaling neural networks introduces significant practical challenges. Hardware constraints, including memory limitations and communication bandwidths on accelerators (GPU or TPU), force users to divide larger models into partitions and to assign different partitions to different accelerators. However, efficient model parallelism algorithms are extremely hard to design and implement, which often requires the practitioner to make difficult choices among scaling capacity, flexibility (or specificity to particular tasks and architectures) and training efficiency. As a result, most efficient model-parallel algorithms are architecture and task-specific. With the growing number of applications of deep learning, there is an ever-increasing demand for reliable and flexible infrastructure that allows researchers to easily scale neural networks for a large variety of machine learning tasks.
>  模型更大要求用户将模型划分为多个部分，并将不同的部分分配到不同的加速器上
>  然而，设计和实现高效的模型并行算法非常困难
>  因此，大多数模型并行算法都是针对特定任务和架构的

To address these challenges, we introduce GPipe, a flexible library that enables efficient training of large neural networks. GPipe allows scaling arbitrary deep neural network architectures beyond the memory limitations of a single accelerator by partitioning the model across different accelerators and supporting re-materialization on every accelerator [13, 14]. With GPipe, each model can be specified as a sequence of layers, and consecutive groups of layers can be partitioned into cells. Each cell is then placed on a separate accelerator. 
>  我们提出 GPipe，它能将模型划分到不同的加速器上，并支持在每个加速器上实现重计算
>  GPipe 中，每个模型被定义为一系列层，连续的层组可以被划分为不同的单元 (cells)，每个单元被放置在独立的加速器上

Based on this partitioned setup, we propose a novel pipeline parallelism algorithm with batch splitting. We first split a mini-batch of training examples into smaller micro-batches, then pipeline the execution of each set of micro-batches over cells. We apply synchronous mini-batch gradient descent for training, where gradients are accumulated across all micro-batches in a mini-batch and applied at the end of a mini-batch. 
>  我们基于这种划分方式提出带有 batch splitting 的流水线并行算法
>  我们首先将 mini-batch 内的训练样本划分到多个 micro-batch，然后各个 cells 对这些 micro-batch 进行流水线执行
>  训练采用同步的 mini-batch 梯度下降方法，也就是梯度在 mini-batch 内的所有 micro-batch 上累积梯度，并在 mini-batch 完成计算后统一应用

Consequently, gradient updates using GPipe are consistent regardless of the number of partitions, allowing researchers to easily train increasingly large models by deploying more accelerators. GPipe can also be complemented with data parallelism to further scale training.
>  因此，使用 GPipe 的梯度更新结果与划分数量无关，此外，GPipe 还可以与数据并行结合，进一步提升训练规模

We demonstrate the flexibility and efficiency of GPipe on image classification and machine translation. For image classification, we train the AmoebaNet model on  $480\times 480$  input from the ImageNet 2012 dataset. By increasing the model width, we scale up the number of parameters to 557 million and achieve a top-1 validation accuracy of  $84.4\%$ . On machine translation, we train a single 128-layer 6-billion-parameter multilingual Transformer model on 103 languages (102 languages to English). We show that this model is capable of outperforming the individually trained 350-million-parameter bilingual Transformer Big [15] models on all 102 language pairs.

# 2 The GPipe Library
We now describe the interface and the main design features of GPipe. This open-source library is implemented under the Lingvo [16] framework. The core design features of GPipe are generally applicable and can be implemented for other frameworks [17, 18, 19].
>  我们描述 GPipe 的接口和主要设计特性
>  GPipe 在 Lingvo 框架下实现
>  GPipe 的核心设计特性具有普遍适用性，可以应用于其他框架

## 2.1 Interface
Any deep neural network can be defined as a sequence of  $L$  layers. Each layer  $L_{i}$  is composed of a forward computation function  $f_{i}$ , and a corresponding set of parameters  $w_{i}$ . GPipe additionally allows the user to specify an optional computation cost estimation function,  $c_{i}$ . With a given number of partitions  $K$ , the sequence of  $L$  layers can be partitioned into  $K$  composite layers, or cells. 
>  任意深度网络都可以定义为 $L$ 层的序列
>  每层 $L_i$ 由一个前向计算函数 $f_i$ 和对应的参数集 $w_i$ 组成，GPipe 还允许用户指定一个可选的计算成本估计函数 $c_i$
>  给定划分数量 $K$，$L$ 层可以划分为 $K$ 个复合层，或称 cells

Let  $p_{k}$  consist of consecutive layers between layers  $i$  and  $j$ . The set of parameters corresponding to  $p_{k}$  is equivalent to the union of  $w_{i}$ ,  $w_{i + 1}, \ldots , w_{j}$ , and its forward function would be  $F_{k} = f_{j} \circ \ldots \circ f_{i + 1} \circ f_{i}$ . The corresponding back-propagation function  $B_{k}$  can be computed from  $F_{k}$  using automatic symbolic differentiation. The cost estimator,  $C_{k}$ , is set to  $\sum_{l = i}^{j} c_{l}$ .
>  记 $p_k$ 为第 $k$ 个 cell，它包含了第 $i$ 层到第 $j$ 层之间的连续多层，它对应的参数集为 $w_i, w_{i+1}, \dots, w_{j}$ 的并集，它对应的前向函数为 $F_k = f_j \circ \dots \circ f_{i+1} \circ f_i$
>  它对应的反向函数 $B_k$ 可以通过 $F_k$ 使用自动符号微分得到
>  它对应的成本估计器记作 $C_k$，定义为 $C_k = \sum_{l=i}^j c_l$

The GPipe interface is extremely simple and intuitive, requiring the user to specify: (i) the number of model partitions  $K$ , (ii) the number of micro-batches  $M$ , and (iii) the sequence and definitions of  $L$  layers that define the model. Please refer to supplementary material for examples.
>  GPipe 的接口非常简单，它要求用户指定:
>  - 模型划分的数量 $K$
>  - micro-batch 的数量 $M$
>  - 定义了模型的 $L$ 个层的序列和各自的定义

## 2.2 Algorithm
Once the user defines the sequence of layers in their network in terms of model parameters  $w_{i}$ , forward computation function  $f_{i}$ , and the cost estimation function  $c_{i}$ , GPipe partitions the network into  $K$  cells and places the  $k$ -th cell on the  $k$ -th accelerator. Communication primitives are automatically inserted at partition boundaries to allow data transfer between neighboring partitions. 
>  用户定义好网络的层序列之后 (本质上是定义好 $w_i, f_i, c_i$ 序列)，GPipe 就会将网络划分到 $K$ 个 cells，然后将第 $k$ 个 cell 放置在第 $k$ 个加速器
>  GPipe 会在 partition 边界处自动插入通信原语，允许相邻单元之间的数据传输

The partitioning algorithm minimizes the variance in the estimated costs of all cells in order to maximize the efficiency of the pipeline by syncing the computation time across all partitions.
>  GPipe 的划分算法旨在最小化所有 cells 的估计成本的方差，从而通过同步所有 cells 的计算时间来最大化流水线效率

>  也就是让所有 cells 的计算时间尽可能接近

During the forward pass, GPipe first divides every mini-batch of size  $N$  into  $M$  equal micro-batches, which are pipelined through the  $K$  accelerators. During the backward pass, gradients for each micro-batch are computed based on the same model parameters used for the forward pass. At the end of each mini-batch, gradients from all  $M$  micro-batches are accumulated and applied to update the model parameters across all accelerators. 
>  在前向过程中，GPipe 首先将每个大小为 $N$ 的 mini-batch 划分为 $M$ 个相同大小的 micro-batch (micro-batch 大小就为 $N/M$)
>  然后在 $K$ 个加速器上进行流水线处理这 $M$ 个 micro-batch
>  在反向传播过程中，每个 micro-batch 的梯度是基于前向传播中使用的相同模型参数计算的
>  在 mini-batch 计算完成后，所有 $M$ 个 micro-batch 的梯度会被累积，并应用于所有加速器上的模型参数更新

This sequence of operations is illustrated in Figure 2c.


![](https://cdn-mineru.openxlab.org.cn/result/2025-08-12/827e9d60-47db-4482-b695-9005bb3c6a97/684994e85827386473f7aa0c4d75a4c2464fa872cf64255fafa09cfcf5f1aab5.jpg)


Figure 2: (a) An example neural network with sequential layers is partitioned across four accelerators.  $F_{k}$  is the composite forward computation function of the  $k$ -th cell.  $B_{k}$  is the back-propagation function, which depends on both  $B_{k + 1}$  from the upper layer and  $F_{k}$ . (b) The naive model parallelism strategy leads to severe under-utilization due to the sequential dependency of the network. (c) Pipeline parallelism divides the input mini-batch into smaller micro-batches, enabling different accelerators to work on different micro-batches simultaneously. Gradients are applied synchronously at the end.

If batch normalization [20] is used in the network, the sufficient statistics of inputs during training are computed over each micro-batch and over replicas if necessary [21]. We also track the moving average of the sufficient statistics over the entire mini-batch to be used during evaluation.
>  如果网络中使用了 batch norm，那么训练过程中会计算每个 micro-batch 的统计量
>  如果使用了多卡数据并行，还会在不同卡的 replicas 上进行跨设备平均
>  同时，GPipe 会在整个 mini-batch 尺度上维护均值和方差的移动平均值，这个平均值会在推理时使用

## 2.3 Performance Optimization
In order to reduce activation memory requirements, GPipe supports re-materialization [14]. During forward computation, each accelerator only stores output activations at the partition boundaries. During the backward pass, the  $k$ -th accelerator recomputes the composite forward function  $F_{k}$ . 
>  为了减少存储中间特征的内存需求，GPipe 支持重计算
>  在前向计算中，每个加速器仅存储 partition 边界的输出激活
>  在反向计算中，第 $k$ 个加速器会重新计算其前向过程

As a consequence, peak activation memory requirement is reduced to  $O(N + \frac{L}{K} \times \frac{N}{M})$ , where  $\frac{N}{M}$  is the micro-batch size and  $\frac{L}{K}$  is the number of layers per partition. In comparison, memory requirement without re-materialization and partitioning would be  $O(N \times L)$ , since computing the gradients  $b_{i}$  requires both the upper layer gradients  $b_{i + 1}$  and the cached activations  $f_{i}(x)$ .
>  这样，峰值激活内存需求被减少到了 $O(N + \frac L K \times \frac N M)$ ，$\frac N M$ 为 micro-batch 大小，$\frac L K$ 为每个 partition 的层数
>  如果没有重计算，峰值激活内存需求将是 $O(N\times L)$ ($N$ 为 batch size, $L$ 为网络层数)，也就是存储每个层的激活

As illustrated in Figure 2c, partitioning introduces some idle time per accelerator, which we refer to as the bubble overhead. This bubble time is  $\textstyle O(\frac{K-1}{M + K-1})$  amortized over the number of micro-steps  $M$  .
>  从 Fig2c 也可以看到，partitioning 会在每个加速器上引入一些空闲时间，我们称为气泡开销
>  气泡时间在 $M$ 个 micro-steps 上的平均开销为 $O(\frac {K-1}{M + K-1})$ , $K$ 为流水线阶段数，$M$ 为 micro-batch 数量

>  气泡就是指流水线并行起来之后，出现了设备空闲的情况

>  关于气泡开销，我们假设一个 $M = 8$, $K = 4$ 的情况
>  $K$ 为加速设备数量，可以类比为工人
>  $M$ 为 micro-batch 数量，可以类比为需要加工的产品数

```
时间 →
GPU0 : [1F][2F][3F][4F][5F][6F][7F][8F]
GPU1 :     [1F][2F][3F][4F][5F][6F][7F][8F]
GPU2 :         [1F][2F][3F][4F][5F][6F][7F][8F]
GPU3 :             [1F][2F][3F][4F][5F][6F][7F][8F]

              ↑
              这里开始出现“气泡” (bubble)——前面的 GPU 在等
```

>  每个产品需要经过 $K = 4$ 个工人顺序处理，就像流水线
>  在开始和结束阶段，存在 GPU 空间，这就是气泡
>  整个流水线处理完全部产品的总时间步数为 $M + K -1$，这是因为做出第一个产品需要 $K$ 个时间步，之后的 $M-1$ 个产品都只需要 1 个时间步就能做出来
>  整个流水线中，每个 GPU 都需要处理 $M$ 个产品，即工作 $M$ 个时间步，因此在整个 $M + K - 1$ 的流程中，每个 GPU 都会空闲 $M + K -1 - M = K - 1$ 个时间步
>  因此，(平均) 每个 GPU 的空闲时间占整个流程的时间比例就是

$$
\text{Bubble Overhead} = \frac {\text{Idle Time}}{\text{Total Time}} = \frac {K - 1}{M + K - 1}
$$

>  实际上，$K-1$ 的空闲时间不可避免，因为流水线一定存在 $K-1$ 的启动延迟，但是随着流水线变长 (要处理的产品数 $M$ 增大)，这个开销就会被分摊

In our experiments, we found the bubble overhead to be negligible when  $M\geq 4\times K$  . This is also partly because re-computation during the backward pass can be scheduled earlier, without waiting for the gradients from earlier layers.
>  提高 micro batch 数量可以减小 bubble overhead
>  在反向过程中，重计算可以提前调度，而不需要等待上面层的梯度，进而可以减小 bubble overhead

GPipe also introduces low communication overhead, given that we only need to pass activation tensors at the partition boundaries between accelerators. Therefore, we can achieve efficient scaling performance even on accelerators without high-speed interconnects.
>  GPipe 也引入了少量通信开销，因为需要在 partition 边界传递激活值
>  因为通信开销少，故在没有高速互联的加速器上也可以实现高效拓展

Figure 2c assumes partitions are evenly balanced. However, memory requirements and computation flops at different layers are often quite imbalanced. In such scenarios, imperfect partitioning algorithms might lead to load imbalance. Better partitioning algorithms can potentially improve the performance over our heuristic approach.
>  不平衡的 partition 会导致 workload 不平衡，更好的划分算法可以提升性能

# 3 Performance Analyses
We evaluate GPipe performance with two very different types of model architectures: an AmoebaNet [12] convolutional model and a Transformer [15] sequence-to-sequence model. We ran experiments to study their scalability, efficiency and communication cost.
>  我们在卷积模型和 Transformer 模型上评估 GPipe 性能

Table 1: Maximum model size of AmoebaNet supported by GPipe under different scenarios. Naive-1 refers to the sequential version without GPipe.  Pipeline- $k$  means  $k$  partitions with GPipe on  $k$  accelerators. AmoebaNet-D (L, D): AmoebaNet model with  $L$  normal cell layers and filter size  $D$  Transformer-L: Transformer model with  $L$  layers, 2048 model and 8192 hidden dimensions. Each model parameter needs 12 bytes since we applied RMSProp during training.  

<table><tr><td>NVIDIA GPUs (8GB each)</td><td>Naive-1</td><td>Pipeline-1</td><td>Pipeline-2</td><td>Pipeline-4</td><td>Pipeline-8</td></tr><tr><td>AmoebaNet-D (L, D)</td><td>(18, 208)</td><td>(18, 836)</td><td>(18, 544)</td><td>(36, 544)</td><td>(72, 312)</td></tr><tr><td># of Model Parameters</td><td>82M</td><td>318M</td><td>542M</td><td>1.05B</td><td>1.8B</td></tr><tr><td>Total Model Parameter Memory</td><td>1.05GB</td><td>3.8GB</td><td>6.45GB</td><td>12.53GB</td><td>24.62GB</td></tr><tr><td>Peak Activation Memory</td><td>6.26GB</td><td>3.46GB</td><td>8.11GB</td><td>15.21GB</td><td>26.24GB</td></tr><tr><td>Cloud TPUv3 (16GB each)</td><td>Naive-1</td><td>Pipeline-1</td><td>Pipeline-8</td><td>Pipeline-32</td><td>Pipeline-128</td></tr><tr><td>Transformer-L</td><td>3</td><td>13</td><td>103</td><td>415</td><td>1663</td></tr><tr><td># of Model Parameters</td><td>282.2M</td><td>78.8M</td><td>5.3B</td><td>21.0B</td><td>83.9B</td></tr><tr><td>Total Model Parameter Memory</td><td>11.7G</td><td>8.8G</td><td>59.5G</td><td>235.1G</td><td>937.9G</td></tr><tr><td>Peak Activation Memory</td><td>3.15G</td><td>6.4G</td><td>50.9G</td><td>199.9G</td><td>796.1G</td></tr></table>

We expect both re-materialization and pipeline parallelism to benefit memory utilization and thus make fitting giant models feasible. We report the biggest model size GPipe can support under reasonably large input size in Table 1. For AmoebaNet, we ran the experiments on Cloud TPUv2s with 8GB memory per accelerator. We used a fixed input image size of  $224\times 224$  and mini-batch size of 128. Without GPipe, a single accelerator can train up to an 82M-parameter AmoebaNet, constrained by device memory limits.
>  我们预计重计算和流水线并行都能提高内存利用率，使得训练大型模型成为可能

 Owing to re-materialization in back-propagation and batch splitting, GPipe reduces the intermediate activation memory requirements from 6.26GB to 3.46GB, enabling a 318M-parameter model on a single accelerator. With model parallelism, we were able to scale AmoebaNet to 1.8 billion parameters on 8 accelerators,  $25x$  more than what is possible without GPipe. In this case, the maximum model size did not scale perfectly linearly due to the imbalanced distribution of model parameters over different layers in AmoebaNet.

We next trained Transformer models using Cloud TPUv3s with 16GB memory per accelerator core. We used a fixed vocabulary size of  $32\mathrm{k}$  sequence length 1024 and batch size 32. Each Transformer layer has 2048 for model dimension, 8192 for feed-forward hidden dimension and 32 attention heads. We scaled the model by varying the number of layers. Re-materialization allows training a  $2.7\times$  larger model on a single accelerator. With 128 partitions, GPipe allows scaling Transformer up to 83.9B parameters, a  $298\times$  increase than what is possible on a single accelerator. Different from AmoebaNet, the maximum model size scales linearly with the number of accelerators for Transformer, since each layer has the same number of parameters and input sizes.
>  Transformer 模型的最大模型规模随着加速器的增长而线性增长，因为每一层的参数数量和输入大小都相同

Table 2: Normalized training throughput using GPipe with different # of partitions  $K$  and different # of micro-batches  $M$  on TPUs. Performance increases with more micro-batches. There is an almost linear speedup with the number of accelerators for Transformer model when  $M\gg K$  .Batch size was adjusted to fit memory if necessary.  

<center><table><tr><td>TPU</td><td colspan="3">AmoebaNet</td><td colspan="3">Transformer</td></tr><tr><td>K =</td><td>2</td><td>4</td><td>8</td><td>2</td><td>4</td><td>8</td></tr><tr><td>M = 1</td><td>1</td><td>1.13</td><td>1.38</td><td>1</td><td>1.07</td><td>1.3</td></tr><tr><td>M = 4</td><td>1.07</td><td>1.26</td><td>1.72</td><td>1.7</td><td>3.2</td><td>4.8</td></tr><tr><td>M = 32</td><td>1.21</td><td>1.84</td><td>3.48</td><td>1.8</td><td>3.4</td><td>6.3</td></tr></table></center>

To evaluate efficiency, we report the normalized training throughput of AmoebaNet-D (18, 256) and Transformer-48 using GPipe with different numbers of partitions and different numbers of micro-batches in Table 2. Each partition is assigned to a separate accelerator. We observe that when the number of micro-batches  $M$  is at least  $4\times$  the number of partitions, the bubble overhead is almost negligible. For Transformer model, there is a  $3.5\times$  speedup when it is partitioned across four times more accelerators. Furthermore, training throughput scales almost linearly with the number of devices, thanks to the computation being evenly distributed across Transformer layers. In contrast, the AmoebaNet model achieves sub-linear speedup due to its imbalanced computation distribution. When  $M$  is relatively small, the bubble overhead can no longer be negligible. When  $M$  is 1, there is effectively no pipeline parallelism. We observe relatively constant throughput regardless of the number of accelerators used, indicating only one device is actively computing at any given time.
>  在 micro-batch 数量至少是 partition 数量的四倍时，气泡开销可以忽略不计
>  由于 Transformer 模型的计算量在设备之间均匀分布，训练吞吐量可以随着设备增长而线性增长
>  当 micro-batch 数量较少的时候，气泡开销就不能忽略不记，当 micro-batch 数量为 1，实际上就没有流水线并行，任何时候都只有一个设备在进行计算

Table 3: Normalized training throughput using GPipe on GPUs without high-speed interconnect.  

<center><table><tr><td>GPU</td><td colspan="3">AmoebaNet</td><td colspan="3">Transformer</td></tr><tr><td>K =</td><td>2</td><td>4</td><td>8</td><td>2</td><td>4</td><td>8</td></tr><tr><td>M = 32</td><td>1</td><td>1.7</td><td>2.7</td><td>1</td><td>1.8</td><td>3.3</td></tr></table></center>

To measure the effect of communication overhead with GPipe, we ran our experiments on a single host with multiple NVIDIA P100 GPUs but without NVLinks. Data transfer across GPUs then has to involve the relatively slow device-to-host and host-to-device transfers through PCI-E. The number of micro-batches was fixed at 32. As shown in Table 3, we observe  $2.7\times$  speedup for AmoebaNet-D (18, 128) when we increase the number of partitions from 2 to 8. For the 24-layer Transformer, the speedup is  $3.3\times$  .There is similar linear speedup to what we observe on TPUs where high-speed interconnects are equipped. The communication bandwidth between devices is no longer a bottleneck for model parallelism since GPipe only transfers activation tensors at the boundaries of partitions.
>  为了测试 GPipe 的通信开销，我们不使用 NVLink，那么 GPU 之间的数据传输就需要通过 device-to-host, host-to-device，经过 PCI-E 传输
>  我们发现吞吐量的提升倍数和使用高速互联的情况类似，因为 GPipe 仅在 partition 边界传输激活张量，故设备之间的通信带宽不是模型并行的瓶颈

## 3.1 Performance Overhead Breakdown
To study opportunities for future performance improvements, we identified the key factors that affect the performance of GPipe on Cloud TPUs. We measured the time spent on different activities listed in Table 4. 

We found that re-computation time was the main contributor to GPipe overhead, taking up to  $23\%$  of the total step time. Another source of overhead was load imbalance. With two partitions, overhead caused by load imbalance was only  $3.2\%$ . 
>  我们发现重计算是 GPipe 的主要开销，另一个开销是负载不平衡

The theoretical bubble overhead is  $O\left(\frac{K -1}{M + K -1}\right)$  where  $K$  is the number of partitions and  $M$  is the number of micro-batches in each minibatch. The observed bubble overhead was slightly lower than the theoretical value partly because re-computation was scheduled early to overlap with the bubble. 
>  观察到的气泡开销略微低于理论值，部分原因是重计算被提前调度，和气泡时间重叠

Weight update time for gradient aggregation at the end of pipeline was also small, thanks to high-speed interconnections between the accelerators.
>  流水线尾部的梯度聚合时间也很小，因为加速器之间存在高速互联

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-12/827e9d60-47db-4482-b695-9005bb3c6a97/e05be344037e00c2e2d0b13955161e6e2f01c5f5a5c14059f72ce8bca2a2498e.jpg)  


Table 4: Time step breakdown

# 4 Image Classification
As a proof of concept, we first used GPipe to scale AmoebaNet. We increased the number of channels in an AmoebaNet and scaled the input image size to  $480\times 480$ . We trained this 557-million-parameter AmoebaNet-B(18, 512) on the ImageNet 2012 dataset, using the same hyper-parameters as described in [12]. The network was divided into 4 partitions. This single model achieves  $84.4\%$  top-1 and  $97\%$  top-5 validation accuracy with single-crop.

Table 5: Image classification accuracy using AmoebaNet-B (18, 512) first trained on ImageNet 2012 then fine-tuned on others. Please refer to the supplementary material for a detailed description of our training setup. Our fine-tuned results were averaged across 5 fine-tuning runs. Baseline results from Real et al. [12] and Cubuk et al. [26] were directly trained from scratch. \*Mahajan et al.'s model [27] achieved  $85.4\%$  top-1 accuracy but it was pretrained on non-public Instagram data. Ngiam et al. [28] achieved better results by pre-training with data from a private dataset (JFT-300M).  

<table><tr><td>Dataset</td><td># Train</td><td># Test</td><td># Clusters</td><td>Accuracy (%)</td><td>Previous Best (%)</td></tr><tr><td>ImageNet-2012</td><td>1,281,167</td><td>50,000</td><td>1000</td><td>84.4</td><td>83.9 [12] (85.4* [27])</td></tr><tr><td>CIFAR-10</td><td>50,000</td><td>10,000</td><td>10</td><td>99.0</td><td>98.5 [26]</td></tr><tr><td>CIFAR-100</td><td>50,000</td><td>10,000</td><td>100</td><td>91.3</td><td>89.3 [26]</td></tr><tr><td>Stanford Cars</td><td>8,144</td><td>8,041</td><td>196</td><td>94.0</td><td>94.8* [26]</td></tr><tr><td>Oxford Pets</td><td>3,680</td><td>3,369</td><td>37</td><td>95.9</td><td>93.8* [29]</td></tr><tr><td>Food-101</td><td>75,750</td><td>25,250</td><td>101</td><td>93.0</td><td>90.4* [30]</td></tr><tr><td>FGVC Aircraft</td><td>6,667</td><td>3,333</td><td>100</td><td>92.7</td><td>92.9* [31]</td></tr><tr><td>Biosnap</td><td>47,386</td><td>2,443</td><td>500</td><td>83.6</td><td>80.2* [32]</td></tr></table>


We further demonstrate the effectiveness of giant convolution networks on other image datasets through transfer learning [22, 23]. Specifically, we used the pre-trained ImageNet model to fine-tune on a variety of target datasets ranging from general to fine-grained classification. We changed the number of output units in the last softmax classification layer to the number of classes in the target dataset and initialized the new softmax layer randomly. All the other layers were initialized from ImageNet pre-training. Input images to the network during training were resized to  $480 \times 480$ , horizontally flipped randomly and augmented using cutout [24]. Training hyper-parameters were the same as those used for ImageNet (a detailed description of our training setup is provided in supplementary material). In Table 5, we report the average single-crop test accuracy over 5 fine-tuning runs for each dataset. Our giant models obtain competitive results on all target datasets. For example, CIFAR-10 error rate is reduced to  $1\%$  and CIFAR-100 error rate to  $8.7\%$ . These results corroborate the findings by Kornblith et al. [25], i.e., better ImageNet models transfer better.

# 5 Massive Massively Multilingual Machine Translation
Next, we demonstrate the flexibility of GPipe by scaling up models used for Natural Language Processing (NLP). Due to an abundance of available parallel corpora, neural machine translation (NMT) has become a benchmark task for any architecture used for NLP [33, 15, 34, 35, 36]. For this reason, we continue our GPipe experiments on a large-scale multilingual NMT task. We use a corpus of parallel documents over 102 languages and English, containing a total of 25 billion training examples, ranging from  $10^{4}$  to  $10^{9}$  per language [37]. This dataset creates a realistic test bed for experiments on scalability by spanning a diverse set of languages from data-scarce (low-resource) to data-rich (high resource). For the first time in machine translation, we show that a large enough NMT model can learn the mapping between more than 100 language pairs simultaneously, while achieving better than bilingual model performance for all languages. This further brings out the importance of having efficient and flexible model-parallelism tools.

Our comparison is based on the performance of a single Transformer [15] trained on all language pairs in this corpus. We scale the architecture along two dimensions to stress the flexibility of GPipe: (i) along the depth by increasing the number of layers in the model and (ii) along the width by increasing the hidden dimension in the feed-forward layers and the number of attention heads (as well as # attention channels) in multi-head attention layers similar to Shazeer et al. [34]. Please refer to the supplementary material for a detailed description of our dataset, baselines, training configuration and optimization hyper-parameters.

We start with a standard 400M-parameter Transformer Big model,  $T(6,8192,16)^1$  , as described in Chen et al. [35], with a vocabulary size of 64k. In Figure 3, we compare its performance against a 1.3B-parameter deep model,  $T(24,8192,16)$ , a 1.3B-parameter wide model,  $T(12,16384,32)$ , a 3B-parameter model,  $T(32,16384,32)$  and a 6B-parameter model,  $T(64,16384,32)$ . All of the models are trained on all language pairs simultaneously, using temperature-based sampling as employed for multilingual BERT $^2$  [3].  $T(12,16384,32)$ ,  $T(24,8192,32)$ ,  $T(32,16384,32)$  and  $T(64,16384,32)$  are partitioned over 2, 4, 8 and 16 accelerators respectively.

From Figure 3, we can observe that increasing the model capacity from 400M to 1.3B parameters significantly improves performance across all languages. Scaling up the model from 1.3B parameters to 6B parameters shows further improvement, especially for high-resource languages. Below we discuss some of our empirical findings based on these large-scale experiments.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-12/827e9d60-47db-4482-b695-9005bb3c6a97/407e62ba4505d6f1d4c5405effe3e09156d6658641e61c5739d8c3a3499ae7fb.jpg)  

Figure 3: Translation quality across all languages with increasing multilingual model capacity. Languages are arranged in the order of decreasing training dataset size from left to right.  $T(L,H,A)$  depicts the performance of a Transformer with  $L$  encoder and  $L$  decoder layers, a feed-forward hidden dimension of  $H$  and  $A$  attention heads. We notice that increasing the model capacity, from 400M params  $(T(64,16384,32))$  leads to significant quality improvements across all languages. We also notice huge quality improvements for low-resource languages (right side of the plot), when compared against bilingual baselines, highlighting the significant transfer gains resulting from training a multilingual model.

**Depth-Width Trade-off:** We study the trade-off between depth and width in our multilingual setup and compare the performance of 1.3B wide model  $T(12,16384,32)$  and 1.3B deep model  $T(24,8192,16)$ . While the quality of these two models on high-resource languages (left of Figure 3) is very similar, the deeper model outperforms by huge margins on low-resource languages, suggesting that increasing model depth might be better for generalization. Further, the quality improvements for low-resource languages (right side of Figure 3), when comparing the 1.3B deep model against the 400M model, are almost as large as the improvements for high-resource languages, indicating that increasing depth might potentially increase the extent of transfer to low-resource tasks.
>  增大 Depth 经验上由于增大 Width

**Trainability Challenges with Deep Models:** Although depth increases the representational capacity of neural networks, it also complicates the optimization problem. In our large-scale experiments, we encountered severe trainability issues arising from a combination of sharp activations (positive kurtosis) and dataset noise. We observed that after training for a few thousand steps, the model predictions would become extremely peaky and vulnerable to noise, which frequently resulted in non-finite or large gradients that eventually destroyed the learning progress. 
>  增大 depth 会让模型变得难训，我们发现在训练了几千步之后，模型预测会变得非常尖锐，并且容易收到数据集噪声影响，这经常导致非常大的梯度，破坏学习过程

To counter these problems, we apply two methods: (i) Following Zhang et al. [38], we scale down the initialization of all transformer feed-forward layers by the number of layers. (ii) We clip the logit predictions (softmax pre-activations) whenever their magnitude exceeds a certain value. A combination of these two approaches allows us to mitigate the training instability posed by scaling model depth.
>  我们采用两种方法，来缓解深度增加带来的不稳定性:
>  - 将所有 Transformer FFN 层的初始化值按照层数进行缩放
>  - 当 logits 预测超过一定值时，我们对其进行截断

# 6 Design Features and Trade-Offs
Several approaches have been proposed to enable efficient large-scale model parallelism. However, each approach chooses its own set of trade-offs, making it suitable for scaling specific architectures under particular hardware constraints. The core idea of model parallelism involves partitioning a network into different computational units, which are then placed on different devices [39, 40, 41, 42]. Conceptually this supports scaling a large spectrum of models to huge capacities. However these approaches typically suffer from low hardware utilization and communication bottlenecks. Single Program Multiple Data (SPMD) and pipeline parallelism have been proposed as solutions to counter these challenges.
>  模型并行的核心思想是将网络划分为不同的计算单元，并将其放在不同的设备上
>  模型并行通常会面临硬件利用率低和通信瓶颈的问题
>  单程序多数据 (SPMD) 和流水线并行已经被提出作为这些挑战的解决方案

Mesh-Tensorflow [34] follows the SPMD paradigm, which extends the Single Instruction-Multiple Data (SIMD) approach used for data parallelism to other tensor dimensions. SPMD allows splitting every computation across multiple devices, allowing the user to scale the size of individual matrix multiplications (and thus, the model parameters of individual layers) linearly with the number of accelerators. 
>  Mesh-TensorFlow 遵循 SPMD 范式
>  SPMD 拓展了传统的 SIMD，不仅仅对 batch 维度并行，还可以对模型内部的张量维度 (hidden size, embedding dim, channel 等) 进行切分
>  SPMD 使得用户可以线性地随着设备数量拓展矩阵乘法的规模 (从而扩展了单个层的模型参数量)

However, this also introduces high communication overhead between the accelerators due to an abundance of AllReduce-like operations used to combine the outputs of each parallelized matrix multiplication. This limits the applicability of the approach to scenarios where accelerators are connected with high speed interconnects. 
>  但 SPMD 也为加速器之间引入了大量的通信开销，例如 AllReduce 操作被用于结合每个并行化的矩阵乘法之间的输出，故 SPMD 被限制在只能用于具有高速互联的加速器的场景

Further, SPMD limits the type of operations that can be efficiently scaled, restricting its use to a specific set of network architectures and machine learning tasks. For example, splitting along the channel dimension of convolution layers under this paradigm is not efficient given that channels are effectively fully connected, whereas splitting along the spatial dimension requires sophisticated techniques for the halo regions. While SPMD allows scaling the model depth by making each operation smaller, it requires splitting each layer over a larger number of accelerators, which in turn further increases the communication overhead across devices.
>  此外，SPMD 也限制了能够有效拓展的计算类型，故仅适用于特定的网络结构

Other approaches have attempted to utilize pipeline-parallelism-based approaches to scale neural networks [43, 44]. The most recent iteration of pipeline parallelism applied to neural network training is PipeDream [45], which targets reducing the communication overhead for parameter servers [46]. PipeDream pipelines the execution of forward passes and intersperses them with backward passes in an attempt to maximize hardware utilization. This design suffers from weight staleness introduced by asynchronous backward updates. To avoid optimization issues stemming from the weight staleness, PipeDream requires maintaining multiple versioned copies of the model parameters on each accelerator in order to compute the gradient updates accurately, preventing users from scaling to bigger models.
>  其他方法尝试用基于流水线并行的方法来拓展网络
>  PipeDream 旨在减少参数服务器的通信开销，它将前向传播的执行流水线化，并在其中穿插反向传播以最大化硬件利用率
>  这个设计由于反向传播是异步的，故引入了权重过时的问题
>  为了避免权重过时导致的优化问题，PipeDream 要求在每个加速器上维护多版本的模型参数，以准确计算梯度更新，这限制了用户对更大模型的拓展

GPipe introduces a new brand of pipeline parallelism that pipelines the execution of micro-batches before applying a single synchronous gradient update for the entire mini-batch. Our novel batch-splitting pipeline parallelism algorithm, when combined with re-materialization, allows scaling to a large number of micro-batches. This minimizes the bubble overhead without the need for asynchronous gradient updates. 
>  GPipe 引入了一种新的流水线并行方法，它在对整个 mini-batch 应用梯度更新之前，对 micro-batches 的执行进行流水线
>  这种 batch-splitting 的流水线并行算法，结合重计算，可以拓展到大量的 micro-batches，进而最小化了气泡开销，而不需要异步梯度更新

GPipe enables the user to scale model size linearly with the number of accelerators used. Unlike SPMD, pipeline parallelism introduces little additional communication overhead when scaling the model. Inter-device communication only takes place at partition boundaries for every micro-batch and the introduced communication overhead is marginal, extending the utility of GPipe to situations where high-speed device interconnects are not available. 
>  GPipe 让用户可以根据所使用的加速器数量线性拓展模型大小
>  和 SPMD 不同，GPipe 引入了很少的通信开销，设备之间的通信仅在 partition 边界处发生

However, GPipe currently assumes that a single layer fits within the memory requirements of a single accelerator. Additionally, micro-batch splitting requires complicated strategies to support layers that require computations across the batch (for example, BatchNorm uses statistics over the micro-batch during training, but accumulates mini-batch statistics for evaluation).
>  但目前 GPipe 假设了单个层可以放入单个加速器的内存，此外 micro-batch 划分需要复杂的策略来支持需要跨 micro-batch 进行计算的层 (例如 batchnorm)

# 7 Conclusion
In this work, we introduce GPipe, a scalable model-parallelism library for training giant networks. We propose a novel batch-splitting pipeline-parallelism algorithm that uses synchronous gradient updates, allowing model parallelism with high hardware utilization and training stability. 
>  本文提出了 GPipe，一个用于训练大规模网络的可拓展模型并行库
>  我们提出了 batch-splitting 流水线并行算法，它使用同步梯度更新，实现了高硬件利用率和训练稳定性

We leverage GPipe to train large-scale convolutional and transformer-based models and demonstrate strong empirical results on both image classification and multilingual machine translation. 

We highlight three key attributes of GPipe: 1) Efficiency: Using a novel batch-splitting pipelining algorithm, GPipe achieves almost linear speedup with the number of devices. 2) Flexibility: GPipe supports any sequential neural networks. 3) Reliability: GPipe utilizes synchronous gradient descent and guarantees consistent training regardless of the number of partitions.
>  GPipe 的三个关键性质:
>  - 效率: batch-splitting 流水线可以达到随着设备数量增大而线性增加的吞吐
>  - 灵活性: 支持任何 sequential NN
>  - 可靠性: 使用同步梯度下降，保证无论划分多少个 partition，结果都是一致的


