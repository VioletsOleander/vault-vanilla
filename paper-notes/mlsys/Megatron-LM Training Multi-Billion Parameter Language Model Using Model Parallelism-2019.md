```
Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro
```

# Abstract
Recent work in language modeling demonstrates that training large transformer models advances the state of the art in Natural Language Processing applications. However, very large models can be quite difficult to train due to memory constraints. 

In this work, we present our techniques for training very large transformer models and implement a simple, efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our approach does not require a new compiler or library changes, is orthogonal and complementary to pipeline model parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. 
>  本文介绍训练非常大 Transformer 模型 (十亿级参数) 的技术，并实现了一个简单高效的层内模型并行方法
>  我们的方法与 pipeline model parallelism 是正交互补的，可以通过在原生 PyTorch 中插入少量通信操作来实现

We illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the entire application with  $76\%$  scaling efficiency when compared to a strong single GPU baseline that sustains 39 TeraFLOPs, which is  $30\%$  of peak FLOPs. 
>  我们使用该方法，在 512 个 GPU 上训练了 8.3B 的 Transformer 模型
>  与一个能维持 39TFLOPs (峰值 FLOPs 的 30%) 的单 GPU baseline 相比，我们的方法可以在整个应用中维持 15.1 PFLOPs，拓展效率为 76%

To demonstrate that large language models can further advance the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of  $63.2\%$ ) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy of  $89.4\%$ ).
>  我们发现对 layer norm 的位置的仔细关注对于随着模型增长而提高性能至关重要

# 1. Introduction
Natural Language Processing (NLP) is advancing quickly in part due to an increase in available compute and dataset size. The abundance of compute and data enables training increasingly larger language models via unsupervised pretraining (Devlin et al., 2018; Radford et al., 2019). Empirical evidence indicates that larger language models are dramatically more useful for NLP tasks such as article completion, question answering, and natural language inference (Lan et al., 2019; Raffel et al., 2019). By finetuning these pretrained language models on downstream natural language tasks, one can achieve state of the art results as shown in recent work Devlin et al., 2018; Peters et al., 2018; Howard & Ruder, 2018; Radford et al., 2018; 2017; Ramachandran et al., 2016; Liu et al., 2019b; Dai et al., 2019; Yang et al., 2019; Liu et al., 2019a; Lan et al., 2019).

As these models become larger, they exceed the memory limit of modern processors, and require additional memory management techniques such as activation checkpointing (Chen et al., 2016). Widely used optimization algorithms such as ADAM require additional memory per parameter to store momentum and other optimizer state, which reduces the size of models that can be effectively trained. Several approaches to model parallelism overcome this limit by partitioning the model such that the weights and their associated optimizer state do not need to reside concurrently on the processor. For example, GPipe (Huang et al., 2018) and Mesh-Tensorflow (Shazeer et al., 2018) provide frameworks for model parallelism of different kinds. However, they require rewriting the model, and rely on custom compilers and frameworks that are still under development.
>  随着模型变得更大，模型会超过处理器的内存限制，故需要额外的内存管理技术，例如激活检查点
>  广泛使用的优化算法例如 ADAM 需要为每个参数存储动量和其他优化器状态，这进一步减小了可以有效训练的模型大小
>  一些模型并行方法将模型划分，使得权重和其优化器状态不需要同时存在于处理器上，来克服内存限制，这些方法需要重写模型，并依赖于自定义的编译器和框架

In this work, we implement a simple and efficient model parallel approach using intra-layer model-parallelism. We exploit the inherent structure in transformer based language models to make a simple model-parallel implementation that trains efficiently in PyTorch, with no custom  $\mathrm{C + + }$  code or compiler required. This approach is orthogonal to pipeline-based model parallelism as advocated by approaches such as GPipe (Huang et al., 2018).
>  本文使用层内模型并行的思路，实现一个简单高效的模型并行方法
>  我们利用了基于 Transformer 的语言模型固有的结构来实现简单的模型并行，可以在 PyTorch 中高效训练，不需要自定义 C++ 代码或编译器
>  该方法和 pipeline-based model parallelism 是正交的

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-01/5c8e1c44-7a4c-472d-821c-b125518222ab/fed1abf5f58dc47e7a8e1ba863e3b29dd6c612a087333348b1454fea47a548a9.jpg)  

Figure 1. Model (blue) and model+data (green) parallel FLOPS as a function of number of GPUs. Model parallel (blue): up to 8-way model parallel weak scaling with approximately 1 billion parameters per GPU (e.g. 2 billion for 2 GPUs and 4 billion for 4 GPUs). Model+data parallel (green): similar configuration as model parallel combined with 64-way data parallel.

To demonstrate the scalability of our approach, we establish a baseline by training a model of 1.2 billion parameters on a single NVIDIA V100 32GB GPU, that sustains 39 TeraFLOPs. This is  $30\%$  of the theoretical peak FLOPS for a single GPU as configured in a DGX-2H server, and is thus a strong baseline. Scaling the model to 8.3 billion parameters on 512 GPUs with 8-way model parallelism, we achieve up to 15.1 PetaFLOPs per second sustained over the entire application. This is  $76\%$  scaling efficiency compared to the single GPU case. Figure 1 shows more detailed scaling results.
>  在 512 个 GPU 上使用 8 路模型并行将模型拓展到 8.3B 参数时，我们在整个应用程序中实现了 15.1 PFLOPs 的持续性能，相对于单 GPU 情况达到了 76% 的拓展效率

To analyze the effect of model size scaling on accuracy, we train both left-to-right GPT-2 (Radford et al., 2019) language models as well as BERT (Devlin et al., 2018) bidirectional transformers and evaluate them on several downstream tasks. We show that the existing BERT architecture results in model degradation as the size increases. We overcome this challenge by rearranging the layer normalization and residual connection in the transformer layers and show that with this change, results for the downstream tasks on development sets improve monotonically as the model size increases. In addition, we show that our models achieve test set state of the art (SOTA) results on WikiText103, cloze-style prediction accuracy on LAMBADA, and reading comprehension RACE datasets.
>  我们发现 BERT 随着模型规模增大，性能会下降，我们通过重新排列 Transformer 层中的 layer norm 层和残差连接来克服这一挑战，并证明经过这种修改后，随着模型规模的增大，development set 上的下游任务会随着模型增大而单调提升

In summary, our contributions are as follows:

- We implement a simple and efficient model parallel approach by making only a few targeted modifications to an existing PyTorch transformer implementation.
- We perform an in-depth empirical analysis of our model and data parallel technique and demonstrate up to  $76\%$  scaling efficiency using 512 GPUs.
- We show that careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased accuracies as the model grows.
- We demonstrate that scaling the model size results in improved accuracies for both GPT-2 (studied up to 8.3 billion parameters) and BERT (studied up to 3.9B parameters) models.
- We showcase that our models achieve state of the art results on test sets: perplexity on WikiText103 (10.8 ppi), accuracy on LAMBADA (66.5%), and accuracy on RACE (90.9%).
- We open source our code along with the training and evaluation pipelines at https://github.com/NVIDIA/Megatron-LM

>  我们的贡献如下:
>  - 我们仅对现有的 PyTorch Transformer 实现进行了少量修改，实现了简单高效的模型并行方法
>  - 我们对模型和数据并行技术进行了深入分析，并展示了 512 个 GPUs 的 76% 拓展效率
>  - 我们发现在类 BERT 模型中，layer norm 层的位置和随着模型规模增大准确率的提高至关重要
>  - 我们展示了扩大模型规模可以增强模型能力

# 2. Background and Challenges
## 2.1. Neural Language Model Pretraining
Pretrained language models have become an indispensable part of NLP researchers' toolkits. Leveraging large corpus pretraining to learn robust neural representations of language is an active area of research that has spanned the past decade. 

Early examples of pretraining and transferring neural representations of language demonstrated that pretrained word embedding tables improve downstream task results compared to word embedding tables learned from scratch (Mikolov et al., 2013; Pennington et al., 2014; Turian et al., 2010). Later work advanced research in this area by learning and transferring neural models that capture contextual representations of words (Melamud et al., 2016; McCann et al., 2017; Peters et al., 2018; Radford et al., 2017; 2019). 
>  早期的预训练和语言的迁移神经表示说明了，与从头开始学习的词嵌入表相比，预训练的词嵌入表可以提高下游任务的结果
>  后续的工作通过学习和迁移能够捕获单词的上下文表示的神经模型，进一步推动了这一领域的研究

Recent parallel work (Ramachandran et al., 2016; Howard & Ruder, 2018; Radford et al., 2018; Devlin et al., 2018; Liu et al., 2019b; Dai et al., 2019; Yang et al., 2019; Liu et al., 2019a; Lan et al., 2019) further builds upon these ideas by not just transferring the language model to extract contextual word representations, but by also finetuning the language model in an end to end fashion on downstream tasks. Through these works, the state of the art has advanced from transferring just word embedding tables to transferring entire multi-billion parameter language models. This progression of methods has necessitated the need for hardware, systems techniques, and frameworks that are able to operate efficiently at scale and satisfy increasing computational needs. Our work aims to provide the tools necessary to take another step forward in this trend.

## 2.2. Transformer Language Models and Multi-Head Attention
Current work in NLP trends towards using transformer models (Vaswani et al., 2017) due to their superior accuracy and compute efficiency. The original transformer formulation was designed as a machine translation architecture that transforms an input sequence into another output sequence using two parts, an Encoder and Decoder. However, recent work leveraging transformers for language modeling such as BERT (Devlin et al., 2018) and GPT-2 (Radford et al., 2019) use only the Encoder or Decoder depending on their needs. This work explores both a decoder architecture, GPT-2, and an encoder architecture, BERT.
>  BERT 为 encoder-only, GPT 为 decoder-only

![[pics/Megatron-Fig2.png]]

Figure 2 shows a schematic diagram of the model we used. We refer the reader to prior work for a detailed description of the model architecture (Vaswani et al., 2017; Devlin et al., 2018; Radford et al., 2019). It is worthwhile to mention that both GPT-2 and BERT use GeLU (Hendrycks & Gimpel, 2016) nonlinearities and layer normalization (Ba et al., 2016) to the input of the multi-head attention and feed forward layers, whereas the original transformer (Vaswani et al., 2017) uses ReLU nonlinearities and applies layer normalization to outputs.
>  GPT-2 和 BERT 对 MHA 层和 FF 层的输入使用了 GeLU 和 layer norm
>  原始的 Transformer 则对输入层使用 ReLU，对输出层使用 layer norm

## 2.3. Data and Model Parallelism in Deep Learning
There are two central paradigms for scaling out deep neural network training to numerous hardware accelerators: data parallelism (Valiant, 1990) where a training minibatch is split across multiple workers, and model parallelism in which the memory usage and computation of a model is distributed across multiple workers. 
>  将 DNN 拓展到多个硬件加速设备的范式是数据并行和模型并行
>  数据并行中，一个训练 minibatch 被划分给多个 workers
>  模型并行中，一个模型的内存使用和计算被分配到多个 workers

By increasing the minibatch size proportionally to the number of available workers (i.e. weak scaling), one observes near linear scaling in training data throughput.
>  通过随着可用工作节点的数量成比例地增加 minibatch 大小 (weak scaling)，可以观察到训练数据吞吐量接近线性增长

> [!info] weak scaling vs strong scaling
> weak scaling 中，问题规模随着处理器增加而成比例增加，故每个处理器的工作量实际上保持不变
> weak scaling 衡量的是在处理更大规模的问题时，总计算时间是否保持不变，其理想表现就是总计算时间保持恒定
> strong scaling 中，问题规模随着处理器增加而固定不变，故每个处理器的工作量会减少
>  strong scaling 衡量的是在处理固定规模的问题时，总计算时间减少多少，其理想表现就是总计算时间随着处理器增加而线性减小

 However, large batch training introduces complications into the optimization process that can result in reduced accuracy or longer time to convergence, offsetting the benefit of increased training throughput (Keskar et al., 2017). Further research (Goyal et al., 2017; You et al., 2017; 2019) has developed techniques to mitigate these effects and drive down the training time of large neural networks. To scale out training even further, parallel work (Chen et al., 2016) has combined data parallelism with activation checkpointing: recomputing activations in the backward pass without storing them in the forward pass to reduce memory requirements.
>  然而，large batch 训练会为优化过程引入复杂性，进而导致准确率下降或收敛时间延长
>  后续研究开发了技术来缓解这些影响并减少了大型 NN 的训练时间
>  为了进一步 scale out，相关工作将数据并行和激活检查点结合使用: 在反向传播中重新计算激活值，而不是在正向传播中存储它们，以降低内存需求

> [!info] large batch vs small batch
> 使用 small batch 训练时，每次更新的梯度是一个有噪声的估计，这个梯度是基于一小部分数据计算出来的，而不是整个数据集，这些噪声被认为具有正则化的效果，有助于优化器跳出狭窄、尖锐的局部最小值
>  使用 large batch 训练时，梯度估计会变得非常平滑和准确，更接近整个数据集的真实梯度，这种平滑的梯度会减少随机性，导致优化器更容易沿着陡峭的路径，冲向最近的、但可能泛化能力较差的尖锐最小值

However, these techniques have one fundamental limitation in the problem size they can tackle: the model must fit entirely on one worker. With language models of increasing size and complexity like BERT and GPT-2, neural networks have approached the memory capacity of modern hardware accelerators. One solution to this problem is to employ parameter sharing to reduce the memory footprint of the model (Lan et al., 2019), but this limits the overall capacity of the model. Our approach is to utilize model parallelism to split the model across multiple accelerators. This not only alleviates the memory pressure, but also increases the amount of parallelism independently of the microbatch size.
>  但这些方法都要求模型必须能够完整放在一个工作节点上
>  解决这个问题的一个方法是采用参数共享来减少模型从内存占用，但这会限制模型的整体能力
>  我们的方法是利用模型并行性，将模型拆分到多个加速设备上，这不仅缓解了内存压力，还独立于 microbatch 大小增加了并行性

Within model parallelism, there are two further paradigms: layer-wise pipeline parallelism, and more general distributed tensor computation. In pipeline model parallelism, groups of operations are performed on one device before the outputs are passed to the next device in the pipeline where a different group of operations are performed. Some approaches (Harlap et al., 2018; Chen et al., 2018) use a parameter server (Li et al., 2014) in conjunction with pipeline parallelism. However these suffer from inconsistency issues. The GPipe framework for TensorFlow (Huang et al., 2018) overcomes this inconsistency issue by using synchronous gradient decent. This approach requires additional logic to handle the efficient pipelining of these communication and computation operations, and suffers from pipeline bubbles that reduce efficiency, or changes to the optimizer itself which impact accuracy.
>  模型并行下有两种范式: 按层的流水线并行，更通用的分布式张量计算
>  在流水线模型并行中，一组 operations 在一个设备上执行，然后将输出传递给流水线中的下一个设备，该设备执行另一组 operations
>  一些方法结合了流水线并行和参数服务器，但这些方法存在不一致的问题
>  TensorFlow 的 GPipe 框架通过使用同步梯度下降克服不一致性，但需要额外的逻辑来处理通信和计算操作的高效流水线化，并会收到流水线气泡的影响

Distributed tensor computation is an orthogonal and more general approach that partitions a tensor operation across multiple devices to accelerate computation or increase model size. 
>  分布式张量计算是一种正交且更通用的方法，它将张量计算划分到多个设备上以加速计算或增加模型规模

FlexFlow (Jia et al., 2018), a deep learning framework orchestrating such parallel computation, provides a method to pick the best parallelization strategy. Recently, Mesh-TensorFlow (Shazeer et al., 2018) introduced a language for specifying a general class of distributed tensor computations in TensorFlow (Abadi et al., 2015). The parallel dimensions are specified in the language by the end user and the resulting graph is compiled with proper collective primitives. 
>  FlexFlow 是一个协调此类并行计算的深度学习框架，提供了一种选择最佳并行化策略的方法
>  Mesh-TensorFlow 在 TensorFlow 中引入了一种语言，用于指定一类通用的分布式张量计算，用户通过该语言指定并行维度，生成的计算图会通过适当的集合原语进行编译

We utilize similar insights to those leveraged in Mesh-TensorFlow and exploit parallelism in computing the transformer's attention heads to parallelize our transformer model. However, rather than implementing a framework and compiler for model parallelism, we make only a few targeted modifications to existing PyTorch transformer implementations. Our approach is simple, does not require any new compiler or code re-writing, and can be fully implemented by inserting a few simple primitives, as described in the next section.
>  我们借鉴了 Mesh-TensorFlow 中的类似见解，利用计算 transformer attention heads 的并行性来并行化我们的 transformer 模型
>  我们并没有实现自己的框架和编译器来实现模型并行，而是对现存的 PyTorch transformer 实现进行了少量修改
>  我们的方法完全可以通过插入少量的简单原语实现

# 3. Model Parallel Transformers
We take advantage of the structure of transformer networks to create a simple model parallel implementation by adding a few synchronization primitives. A transformer layer consists of a self attention block followed by a two-layer, multi-layer perceptron (MLP) as shown in Figure 2. We introduce model parallelism in both of these blocks separately.
>  我们通过添加少量的同步原语实现了简单的模型并行
>  一个 transformer 层包含一个 self attention block + 一个两层的 MLP
>  我们将分别介绍这两个 block 的模型并行

We start by detailing the MLP block. The first part of the block is a GEMM followed by a GeLU nonlinearity:

$$
Y = \mathrm{GeLU}(XA) \tag{1}
$$

One option to parallelize the GEMM is to split the weight matrix  $A$  along its rows and input  $X$  along its columns as:

$$
X = [X_{1},X_{2}],A = \begin{bmatrix} A_{1}\\ A_{2} \end{bmatrix} . \tag{2}
$$

This partitioning will result in  $Y = \mathrm{GeLU}(X_1A_1 + X_2A_2)$ . Since GeLU is a nonlinear function,  $\mathrm{GeLU}(X_1A_1 + X_2A_2) \neq \mathrm{GeLU}(X_1A_1) + \mathrm{GeLU}(X_2A_2)$  and this approach will require a synchronization point before the GeLU function.

>  我们首先介绍 MLP block 的并行
>  MLP block 的第一部分是 GEMM + GeLU，一种并行化 GEMM 的方式是沿着行划分权重矩阵 $A$，沿着列划分输入 $X$，进而计算变为 $Y = \mathrm{GeLU}(X_1A_1 + X_2A_2)$
>  因为 GeLU 是非线性函数，故 $\mathrm{GeLU}(X_1A_1 + X_2A_2) \neq \mathrm{GeLU}(X_1A_1) + \mathrm{GeLU}(X_2A_2)$，因此这样的划分要求在计算完 GEMM，计算 GeLU 之前进行一次同步

Another option is to split  $A$  along its columns  $A = [A_{1},A_{2}]$ . This partitioning allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM:

$$
[Y_1,Y_2] = [\mathrm{GeLU}(XA_1),\mathrm{GeLU}(XA_2)] \tag{3}
$$

>  另一种并行化方式是沿着列划分权重矩阵 $A$，这使得 GeLU 可以独立地应用于输出的不同部分 $Y_1, Y_2$

![[pics/Megatron-Fig3.png]]

This is advantageous as it removes a synchronization point. Hence, we partition the first GEMM in this column parallel fashion and split the second GEMM along its rows so it takes the output of the GeLU layer directly without requiring any communication as shown in Figure 3a. The output of the second GEMM is then reduced across the GPUs before passing the output to the dropout layer. 
>  第二种方式的优势在于移除了同步点
>  因此，我们将 MLP block 的第一个 GEMM 按照列并行的方式划分，将 MLP block 的第二个 GEMM 按照行并行的方式划分 (第一个 GEMM 后面有 GeLU，第二个 GEMM 后面没有 GeLU)
>  如 Fig3a 所示，这使得第二个 GEMM 可以直接接收第一个 GEMM 的输出而不需要通讯，第二个 GEMM 的输出则会在传递给 dropout layer 之前，在 GPUs 之间进行 reduce

This approach splits both GEMMs in the MLP block across GPUs and requires only a single all-reduce operation in the forward pass ( $g$  operator) and a single all-reduce in the backward pass ( $f$  operator). 
>  这种并行方式将 MLP block 中的两个 GEMM 划分到多个 GPUs，并且前向传播仅需要一次 all-reduce，反向传播也仅需要一次 all-reduce

These two operators are conjugates of each other and can be implemented in PyTorch with only a few lines of code. As an example, the implementation of the  $f$  operator is provided below:
>  $f$ operator 和 $g$ operator 共轭 (也就是在数学上这两个 operator 互为各自的逆，二者之间存在对称关系，实现逻辑是相互对应的)
>  二者都可以在 PyTorch 中通过几行代码实现，例如 $f$ operator 的实现如下所示:

```python
class f(torch.autograd.Function): 
    def forward(ctx, x): 
        return x 
    def backward(ctx, gradient): 
        all_reduce(gradient) 
        return gradient
```

Code 1. Implementation of  $f$  operator.  $g$  is similar to  $f$  with identity in the backward and all-reduce in the forward functions.

As shown in Figure 3b, for the self attention block we exploit inherent parallelism in the multihead attention operation, partitioning the GEMMs associated with key  $(A)$ , query  $(Q)$ , and value  $(V)$  in a column parallel fashion such that the matrix multiply corresponding to each attention head is done locally on one GPU. This allows us to split per attention head parameters and workload across the GPUs, and doesn't require any immediate communication to complete the self-attention. 
>  如 Fig3b 所示，对于 self attention block，我们采用 MHA 运算的内在并行性，将和 key $A$, query $Q$, value $V$ 有关的 GEMMs 以列并行的形式划分，使得每个 attention head 的矩阵乘在单个 GPU 上实现
>  这样的划分可以将每个 attention head 的参数和计算任务分配到不同的 GPU 上，且不需要立即通信来完成 self-attention

The subsequent GEMM from the output linear layer (after self attention) is parallelized along its rows and takes the output of the parallel attention layer directly, without requiring communication between the GPUs. 
>  self attention 之后的 output linear layer 的 GEMM 计算则沿着参数的行并行，进而可以直接接收上一层并行 attention 层的输出，不需要 GPUs 之间的通信

![[pics/Megatron-Fig4.png]]

This approach for both the MLP and self attention layer fuses groups of two GEMMs, eliminates a synchronization point in between, and results in better scaling. This enables us to perform all GEMMs in a simple transformer layer using only two all-reduces in the forward path and two in the backward path (see Figure 4).
>  这种方法在 MLP 和 self attention 层都合并了两个 GEMM 操作，消除了中间的一个同步点，进而实现了更好的拓展性，使得我们可以在前向传播和反向传播中仅使用两次 all-reduce 操作就完成所有的 GEMM 计算

>  $Y = XA$ 的并行方式: 完全并行

$$
\begin{align}
Y &= XA\\
Y &= X[A_1, A_2]\\
[Y_1, Y_2] &= [XA_1, XA_2]
\end{align}
$$

>  $Z = YB$ 的并行方式: 需要一次加法 reduce

$$
\begin{align}
Z &= YB\\
Z &= [Y_1, Y_2] B\\
Z &= [Y_1, Y_2] \begin{bmatrix}
B_1\\
B_2
\end{bmatrix}\\
Z &= Y_1B_1 + Y_2 B_2
\end{align}
$$

>  $Z = YB = XAB$ 的并行方式: 需要一次加法 reduce

$$
\begin{align}
Z &= Y_1B_1 + Y_2B_2\\
Z &= XA_1B_1 + XA_2B_2
\end{align}
$$

The transformer language model has an output embedding with the dimension of hidden-size  $(H)$  times vocabulary-size  $(v)$ . Since the vocabulary size is on the order of tens of thousands of tokens for modern language models (for example, GPT-2 used a vocabulary size of 50,257), it is beneficial to parallelize the output embedding GEMM. However, in transformer language models, the output embedding layer shares weights with the input embedding, requiring modifications to both. 
>  transformer 语言模型的输出嵌入维度为隐藏层大小 $H$ 乘上词袋大小 $v$
>  由于现代语言模型的词袋大小通常在数万个 token 级别，因此，对输出嵌入的 GEMM 进行并行化是有必要的
>  在 transformer 语言模型中，输出嵌入层和输入嵌入层共享权重，因此需要对两者进行修改

We parallelize the input embedding weight matrix  $E_{H \times v}$  along the vocabulary dimension  $E = [E_{1},E_{2}]$  (column-wise). Since each partition now only contains a portion of the embedding table, an all-reduce (  $g$  operator) is required after the input embedding. 
>  我们沿着词袋维度 (列) 并行化输入嵌入权重矩阵 $E_{H\times v}$，此时每一个部分仅包含一部分 embedding table，故需要在并行计算之后执行一次 all-reduce ($g$ operator)

>  输入嵌入层实际上就是查找各个 tokens 的 embedding，因为 embedding table 太大，故需要拆分到多个 GPUs 上，因此需要执行 all-reduce 才能让每个 GPU 都具有完整的输入 sequence 中各个 tokens 的 embedding

For the output embedding, one approach is to perform the parallel GEMM  $[Y_{1},Y_{2}] = [XE_{1},XE_{2}]$  to obtain the logits, add an all-gather  $Y = \text{all-gather}([Y_1,Y_2])$  , and send the results to the cross-entropy loss function. However, for this case, the all-gather will communicate  $b\times s\times v$  elements  $b$  is the batch-size and  $s$  is the sequence length) which is huge due to vocabulary size being large. To reduce the communication size, we fuse the output of the parallel GEMM  $[Y_{1},Y_{2}]$  with the cross entropy loss which reduces the dimension to  $b\times s$  Communicating scalar losses instead of logits is a huge reduction in communication that improves the efficiency of our model parallel approach.
>  对于输出嵌入层，一种方法是执行并行 GEMM $[Y_1, Y_2] = [XE_1, XE_2]$ 来获取 logits (logits 表示原始，未归一化的分数)，然后使用 all-gather 来让每个 GPU 具有完整的 $Y$，然后将 $Y$ 发送给交叉熵损失函数
>  在该情况下，all-gather 会通信 $b \times s \times v$ 个元素，因为词袋大小 $v$ 很大，故通信量很大
>  为了减少通信量，我们将并行 GEMM 的输出 $[Y_1, Y_2]$ 和交叉熵损失融合，进而将通信量减少到 $b \times s$
>  对标量损失进行通信而不是 logits 极大减少了通信量，提高了模型并行方法的效率

>  输出层中，模型会将自己的隐藏状态 $X$ 和嵌入矩阵 $E$ 进行矩阵乘，得到每个 logits 矩阵 $Y = XE$ ($Y$ 的行数为序列长度 $s$，每一行都是一个词袋长度 $v$ 的向量，表示了序列中这个位置上的 tokens logits 分布)
>  如果不通信，每个 GPU 具有 seq 中一部分位置的 logits 分布，然后在本地计算这部分位置的 logits 分布对应的损失值，再通信聚合这些本地计算的标量损失值

Much of our model parallel approach can be characterized as techniques aimed at reducing communication and keeping the GPUs compute bound. 
>  我们的模型并行方法的大部分可以描述为旨在减少通信并保持 GPU compute bound 的技术

Rather than having one GPU compute part of the dropout, layer normalization, or residual connections and broadcast the results to other GPUs, we choose to duplicate the computation across GPUs. Specifically, we maintain duplicate copies of layer normalization parameters on each GPU, and take the output of the model parallel region and run dropout and residual connection on these tensors before feeding them as input to the next model parallel regions. To optimize the model we allow each model parallel worker to optimize its own set of parameters. Since all values are either local to or duplicated on a GPU, there is no need for communicating updated parameter values in this formulation.
>  与其让一个 GPU 计算 dropout, layer norm or residual connections 然后将结果广播到其他 GPU，我们选择在多个 GPU 上复制计算
>  具体地说，我们在每个 GPU 上维护 layer norm 参数 (scale, bias) 的副本，并对模型并行的输出区域执行 dropout 和残差连接计算，然后将这些张量作为输入传递给下一个模型并行区域

>  也就是 layer norm, drop, residual connections 也是各个 GPU 独立做自己的

We present further details about the hybrid model and data parallelism and handling random number generation in Appendix B for reference. 

In summary, our approach as described above is simple to implement, requiring only a few extra all-reduce operations added to the forward and backward pass. It does not require a compiler, and is orthogonal and complementary to the pipeline model parallelism advocated by approaches such as (Huang et al., 2018).
>  总之，如上所述，我们的方法实现起来很简单，只需要在前向和反向传播中添加少量的 all-reduce 操作
>  它不需要编译器，并且与流水线并行是正交的

# 4. Setup
Pretrained language understanding models are central tasks in natural language processing and language understanding. There are several formulations of language modeling. In this work we focus on GPT-2 (Radford et al., 2019), a left-to-right generative transformer based language model, and BERT (Devlin et al., 2018), a bi-directional transformer model based on language model masking. We explain our configurations for these models in the following section and refer to the original papers for more details.
>  我们关注 GPT-2: 一个从左到右的生成式 transformer-based 模型，以及 BERT: 一个基于 masking 的双向 transformer 模型

## 4.1. Training Dataset
To collect a large diverse training set with long-term dependencies we aggregate several of the largest language modeling datasets. We create an aggregate dataset consisting of Wikipedia (Devlin et al., 2018), CC-Stories (Trinh & Le, 2018), RealNews (Zellers et al., 2019), and OpenWebtext (Radford et al., 2019). To avoid training set leakage into our downstream tasks we remove the Wikipedia articles present in the WikiText103 test set (Merity et al., 2016). We also remove unnecessary newlines from the CC-Stories corpus introduced by preprocessing artifacts. For BERT models we include BooksCorpus (Zhu et al., 2015) in the training dataset, however, this dataset is excluded for GPT-2 trainings as it overlaps with LAMBADA task.
>  为了收集一个包含长期依赖关系的大规模多样化训练集，我们整合了几个最大的语言模型数据集
>  为了避免训练集泄漏到下游任务，我们移除了 WikiText103 测试集中存在的 Wikipedia 文章

We combined all the datasets and then filtered out all the documents with content length less than 128 tokens from the aggregated dataset. Since similar content might appear multiple times in the aggregated datasets, we used locality-sensitive hashing (LSH) to deduplicate content with a jaccard similarity greater than 0.7. The resulting aggregate corpus contains 174 GB of deduplicated text.
>  我们在合并了所有数据集之后，从合并数据集中过滤掉了长度小于 128 tokens 的文档
>  由于聚合数据集中会出现重复内容，我们使用了 LSH 来去重，去重标准是 Jaccard 相似度大于 0.7，最终的数据集为 174GB 大小

## 4.2. Training Optimization and Hyperparameters
To train our models efficiently we utilize mixed precision training with dynamic loss scaling to take advantage of the V100's Tensor Cores (Micikevicius et al., 2017; NVIDIA, 2018). We start by initializing our weights  $W$  with a simple normal distribution  $W \sim \mathcal{N}(0, 0.02)$ . We then scale weights immediately before residual layers by  $\frac{1}{\sqrt{2N}}$  where  $N$  is the number of transformer layers comprised of self attention and MLP blocks. For our optimizer we utilize Adam (Kingma & Ba, 2014) with weight decay (Loshchilov & Hutter, 2019)  $\lambda = 0.01$ . Additionally, we use global gradient norm clipping of 1.0 to improve the stability of training large models. In all cases, a dropout of 0.1 is used. Lastly, to better manage our memory footprint we utilize activation checkpointing (Chen et al., 2016) after every transformer layer.
>  为了高效训练，我们使用了混合精度训练，并结合了动态损失缩放，以利用 V100 的 Tensor cores
>  我们使用正态分布初始化权重，然后在残差层之前对权重进行缩放，缩放因子为 $\frac 1 {\sqrt {2N}}$，其中 $N$ 是 transformer layer 的数量
>  我们使用全局梯度范数裁剪，以提高训练稳定性
>  为了更好管理内存占用，我们在每个 transformer 层之后使用激活检查点

For GPT-2 models, all training is performed with sequences of 1024 subword units at a batch size of 512 for 300k iterations. Our learning rate of 1.5e-4 utilizes a warmup period of  $3\mathrm{k}$  iterations before following a single cycle cosine decay over the remaining 297k iterations. We stop the decay at a minimum learning rate of 1e-5.
>  GPT-2 的所有训练都是 batch size = 512，序列长度为 1024，进行 300k 次迭代

For BERT models, we largely follow the training process described in (Lan et al., 2019). We use the original BERT dictionary with vocab size of 30,522. In addition, we replace the next sentence prediction head with sentence order prediction as suggested by (Lan et al., 2019) and use whole word n-gram masking of (Joshi et al., 2019). For all cases, we set the batch size to 1024 and use a learning rate of 1.0e-4 warmed up over 10,000 iterations and decayed linearly over 2 million iterations. Other training parameters are kept the same as (Devlin et al., 2018).
>  BERT 使用原始的 BERT 词袋，大小为 30522

# 5. Experiments
All of our experiments use up to 32 DGX-2H servers (a total of 512 Tesla V100 SXM3 32GB GPUs). Our infrastructure is optimized for multi-node deep learning applications, with 300 GB/sec bandwidth between GPUs inside a server via NVSwitch and 100 GB/sec of interconnect bandwidth between servers using 8 InfiniBand adapters per server.
>  所有的实验最多使用 32 台 DGX-2H 服务器 (一共 512 块 V100)
>  每台服务器内部通过 NVSwitch 实现每台 GPU 之间 300GB/s 的带宽
>  服务器之间通过每台服务器配备的 8 个 InifiniBand 网卡实现 100GB/s 的带宽

## 5.1. Scaling Analysis

Table 1. Parameters used for scaling studies. Hidden size per attention head is kept constant at 96.  

<table><tr><td>Hidden Size</td><td>Attention heads</td><td>Number of layers</td><td>Number of parameters (billions)</td><td>Model parallel GPUs</td><td>Model +data parallel GPUs</td></tr><tr><td>1536</td><td>16</td><td>40</td><td>1.2</td><td>1</td><td>64</td></tr><tr><td>1920</td><td>20</td><td>54</td><td>2.5</td><td>2</td><td>128</td></tr><tr><td>2304</td><td>24</td><td>64</td><td>4.2</td><td>4</td><td>256</td></tr><tr><td>3072</td><td>32</td><td>72</td><td>8.3</td><td>8</td><td>512</td></tr></table>

To test the scalability of our implementation, we consider GPT-2 models with four sets of parameters detailed in Table 1. To have consistent GEMM sizes in the self attention layer, the hidden size per attention head is kept constant at 96 while the number of heads and layers are varied to obtain configurations ranging from 1 billion to 8 billion parameters. 
>  我们考虑了 GPT-2 模型的四种参数配置，如 Table 1
>  为了在 self-attention 层有一致的 GEMM 大小，每个 attention head 的 hidden size 固定为 96，attention head 的数量和 transformer 层数则可以调整，进而获得不同参数量的模型

The configuration with 1.2 billion parameters fits on a single GPU whereas the 8 billion parameter model requires 8-way model parallelism (8 GPUs). 
>  1.2B 的模型可以在单个 GPU 上运行，8B 的模型则需要 8 路模型并行

The original vocabulary size was 50,257, however, to have efficient GEMMs for the logit layer, it is beneficial for the per-GPU vocabulary size to be a multiple of 128. Since we study up to 8-way model parallelism, we pad the vocabulary such that it is divisible by  $128 \times 8 = 1024$ , resulting in a padded vocabulary size of 51,200. 
>  原始词袋大小为 50257
>  为了让 logit layer 的 GEMM 更高效，每个 GPU 的词袋大小最好是 128 的倍数，故我们对词袋进行了 pad，使得它可以被 128x8=1024 整除，填充后词袋大小为 51200

We study both model and model+data parallel scaling. For the model parallel scaling, a fixed batch size of 8 is used across all configurations. Data parallel scaling is necessary for training many state of the art models which typically use a much larger global batch size. To this end, for the model+data parallel cases we fix the global batch size to 512 for all experiments which corresponds to 64-way data parallelism.
>  我们研究两种模型并行和模型+数据并行的拓展方式
>  对于模型并行拓展，所有配置都使用 batch size = 8
>  数据并行对于训练 SOTA 模型是必要的，因为这些模型通常使用更大的 global batch size，因此，在模型+数据并行的情况下，所有实验的全局 batch size 固定为 512，相当于 64 路数据并行 (64 个模型 replica，各自处理各自大小为 8 的 minibatch，每个模型 replica 在多个 GPU 上并行计算)

>  模型+数据并行
>  1. 多个模型 replica 各自处理各自的 minibatch
>      每个模型 replica 的参数划分为多个 partition 分布到多个 GPU 上执行计算
>      计算过程中，各个 GPU 上的 partition 可能在计算中进行通信以同步数据
>      计算完成后，各个 GPU 上的 partition 进行通信获取全局结果
>  2. 各个模型 replica 的计算结果通过 all-reduce 聚合
>  3. 各个模型 replica 根据计算结果进行本地参数更新

### 5.1.1. Model and Data Parallelism
Throughout this section, we will showcase weak scaling with respect to the model parameters for both model parallel and model+data parallel cases. Weak scaling is typically done by scaling the batch-size, however, this approach does not address training large models that do not fit on a single GPU and it leads to training convergence degradation for large batch sizes. 
>  本节展示模型并行+数据并行随着模型参数的弱拓展性
>  传统的弱拓展是通过拓展 batch-size 实现的，但这种方法无法解决哪些无法放入单个 GPU 的大模型训练问题，且大 batch-size 会导致训练的收敛问题

In contrast, here we use weak scaling to train larger models that were not possible otherwise. The baseline for all the scaling numbers is the first configuration (1.2 billion parameters) in Table 1 running on a single GPU. This is a strong baseline as it achieves 39 TeraFLOPS during the overall training process, which is  $30\%$  of the theoretical peak FLOPS for a single GPU in a DGX-2H server.
>  相较之下，我们的弱拓展中，拓展的 "问题规模" 是模型参数而不是 batch size (也就是通过增大模型参数来拓展计算量，而不是增大 batch size 来拓展计算量)
>  单个 GPU 上的 baseline 可以达到峰值 FLOPS 的 30%



![](https://cdn-mineru.openxlab.org.cn/result/2025-08-01/5c8e1c44-7a4c-472d-821c-b125518222ab/8d9963b3f7ea47219e9a6aad0e0c99ef07a543d31d18829c4a2b2f839ff9a874.jpg)  

Figure 5. Model and model + data parallel weak scaling efficiency as a function of the number of GPUs.

Figure 5 shows scaling values for both model and model+data parallelism. We observe excellent scaling numbers in both settings. For example, the 8.3 billion parameters case with 8-way (8 GPU) model parallelism achieves  $77\%$  of linear scaling. Model+data parallelism requires further communication of gradients and as a result the scaling numbers drop slightly. However, even for the largest configuration (8.3 billion parameters) running on 512 GPUs, we achieve  $74\%$  scaling relative to linear scaling of the strong single GPU baseline configuration (1.2 billion parameters). Further scaling analysis is provided in Appendix D
>  Figure5 展示了模型并行以及模型+数据并行的拓展效率
>  模型并行的拓展效率达到了 77%，近似线性
>  模型+数据并行需要进一步的通信以收集梯度，故拓展效率略微下降
>  但即便是对于在 512 个 GPUs 上运行的模型+数据并行配置，也可以达到近似象形的拓展效率 74%

## 5.2. Language Modeling Results Using GPT-2
To demonstrate that large language models can further advance the state of the art, we consider training GPT-2 models of the sizes and configurations listed in Table 2. The 355M model is equivalent in size and configuration of BERT-Large model (Devlin et al., 2018). The 2.5B model is bigger than the previous largest GPT-2 model, and the 8.3B model is larger than any left-to-right transformer language model ever trained, to the best of our knowledge. To train and evaluate our language models we use the procedure described in section 4. 
>  我们考虑了 Table2 中列出的大小和配置的 GPT-2 模型

Table 2 also lists the time it takes to advance one epoch which is equivalent to 68,507 iterations. For example, for the 8.3B model on 512 GPUs, each epoch takes around two days. Compared to the configurations used for our scaling studies in Table 1, the 2.5B model is the same, the 8.3B model has 24 attention heads instead of 32, and the 355M is much smaller than any seen previously while still using 64 GPUs to train, leading to the much lower time per epoch.
>  Table2 还列出了完成一个 epoch 所需要的时间，例如 512 块 GPU 上运行 8.3B 模型大约需要两天时间

Table 2. Model configurations used for GPT-2.  

<center><table><tr><td>Parameter Count</td><td>Layers</td><td>Hidden Size</td><td>Attn Heads</td><td>Hidden Size per Head</td><td>Total GPUs</td><td>Time per Epoch (days)</td></tr><tr><td>355M</td><td>24</td><td>1024</td><td>16</td><td>64</td><td>64</td><td>0.86</td></tr><tr><td>2.5B</td><td>54</td><td>1920</td><td>20</td><td>96</td><td>128</td><td>2.27</td></tr><tr><td>8.3B</td><td>72</td><td>3072</td><td>24</td><td>128</td><td>512</td><td>2.10</td></tr></table></center>

Table 3. Zero-shot results. SOTA are from (Khandelwal et al., 2019) for Wikitext103 and (Radford et al., 2019) for LAMBADA.  

<center><table><tr><td>Model</td><td>Wikitext103 Perplexity ↓</td><td>LAMBADA Accuracy ↑</td></tr><tr><td>355M</td><td>19.31</td><td>45.18%</td></tr><tr><td>2.5B</td><td>12.76</td><td>61.73%</td></tr><tr><td>8.3B</td><td>10.81</td><td>66.51%</td></tr><tr><td>Previous SOTA</td><td>15.79</td><td>63.24%</td></tr></table></center>

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-01/5c8e1c44-7a4c-472d-821c-b125518222ab/5f8ce6ca6db981dd12a5b3ac08f383bcc994016b7a088a1fdf48be004f9ebf97.jpg)  

Figure 6. Validation set perplexity. All language models are trained for  $300\mathrm{k}$  iterations. Larger language models converge noticeably faster and converge to lower validation perplexities than their smaller counterparts.

Figure 6 shows validation perplexity as a function of number of iterations. As the model size increases, the validation perpelixity decreases and reaches a validation perplexity of 9.27 for the 8.3B model. We report the zero-shot evaluation of the trained models on the LAMBADA and WikiText103 datasets in Table 3. 
>  Fig6 展示了验证集 perplexity 和迭代次数的关系，随着模型大小增大，Perplexity 越来越低
>  Table3 报告了零样本下的 accuracy 和 perplexity

For more details on evaluation methodology, see Appendix E. We observe the trend that increasing model size also leads to lower perplexity on WikiText103 and higher cloze accuracy on LAMBADA. Our 8.3B model achieves state of the art perplexity on the WikiText103 test set at a properly adjusted perplexity of 10.81. At  $66.51\%$  accuracy, the 8.3B model similarly surpasses prior cloze accuracy results on the LAMBADA task. We have included samples generated from the 8.3 billion parameters model in the Appendix C. Recently researchers from Microsoft in collaboration with NVIDIA trained a 17 billion parameter GPT-2 model called Turing-NLG (Microsoft, 2020) using Megatron and showed that the accuracies further improve as they scale the model, highlighting the value of larger models.

>  Perplexity 即整个句子的平均负对数概率的指数

$$
\text{Perplexity} = \exp(-\frac 1 N \sum_{i=1}^N \log P(w_i))
$$

>  使用 $\log$ 的原因是直接连乘会出现数值下溢
>  使用平均的原因是为了能够比较不同长度的序列
>  实际上基于信息论，模型预测下每个词的信息量就是它的负对数概率，因此 Perplexity 实际上是计算了整个句子的平均信息量，然后取指数
>  根据信息论，这个平均信息量的指数就表示了模型在预测下一个词时，平均下来，大致的 “候选词数量”
>  例如，从 $K$ 个候选词的均匀分布上选取词的 Perplexity 就等于

$$
\exp(-\log\frac 1 K) = \exp(\log K) = K
$$

>  因此我们可以把最后计算得到的 Perplexity $K$ 看作是从一个 $K$ 个候选词的均匀分布上选取词得到的 Perplexity

To ensure we do not train on any data found in our test sets, we calculate the percentage of test set 8-grams that also appear in our training set as done in previous work (Radford et al., 2019). The WikiText103 test set has at most $10.8\%$  overlap and the LAMBADA test set (Paperno et al., 2016) has at most  $1.4\%$  overlap. We should note that the WikiText103 test set has already  $9.09\%$  overlap with the WikiText103 training set (Radford et al., 2019). As these are consistent with previous work, we are confident that no documents from our test data are inadvertently included in our training data.

Table 4. Model configurations used for BERT.  

<center><table><tr><td>Parameter Count</td><td>Layers</td><td>Hidden Size</td><td>Attention Heads</td><td>Total GPUs</td></tr><tr><td>336M</td><td>24</td><td>1024</td><td>16</td><td>128</td></tr><tr><td>1.3B</td><td>24</td><td>2048</td><td>32</td><td>256</td></tr><tr><td>3.9B</td><td>48</td><td>2560</td><td>40</td><td>512</td></tr></table></center>

## 5.3. Bi-directional Transformer Results Using BERT
In this section, we apply our methodology to BERT-style transformer models and study the effect of model scaling on several downstream tasks. Prior work (Lan et al., 2019) found that increasing model size beyond BERT-large with 355M parameters results in unexpected model degradation. To address this degradation, the authors of that work (Lan et al., 2019) introduced parameter sharing and showed that that their models scale much better compared to the original BERT model.
>  之前的工作发现增大 BERT 大小会导致性能下降，通过参数共享可以解决这个问题

We further investigated this behaviour and empirically demonstrated that rearranging the order of the layer normalization and the residual connections as shown in Figure 7 is critical to enable the scaling of the BERT-style models beyond BERT-Large. The architecture (b) in Figure 7 eliminates instabilities observed using the original BERT architecture in (a) and also has a lower training loss. To the best of our knowledge, we are the first to report such a change enables training larger BERT models.
>  我们发现重新排布 layer norm 和 residual connection 的顺序可以解决这个问题，进而可以训练更大的 BERT

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-01/5c8e1c44-7a4c-472d-821c-b125518222ab/f77a9c8adc79bf6ad7e784084749ddc69272810ed7afa28d30b6ee6dda0d4f8d.jpg)  


Figure 7. Training loss for BERT model using the original architecture (a) and the rearranged architecture (b). Left figure shows the training loss for 336M and 752M BERT model. While the original architecture performs well on the 336M model, the modifications in (b) enable stable training with lower training loss.

Using the architecture change in Figure 7(b), we consider three different cases as detailed in Table 4. The 336M model has the same size as BERT-large. The 1.3B is the same as the BERT-xlarge configuration that was previously shown to get worse results than the 336M BERT-large model (Lan et al., 2019). We further scale the BERT model using both larger hidden size as well as more layers to arrive at the 3.9B parameter case. In all cases, the hidden size per attention head is kept constant at 64.336M and 1.3B models are trained for 2 million iterations while the 3.9B model is trained for 1.5 million iterations and is still training.
>  通过调整 residual connection 和 layer norm 的顺序，我们发现 BERT 可以随着参数量增大而性能提升

On a  $3\%$  held-out set, 336M, 1.3B, and 3.9B models achieve validation set perplexity of 1.58, 1.30, and 1.16, respectively, a monotonic decrease with the model size. 

We finetune the trained models on several downstream tasks including MNLI and QQP from the GLUE benchmark (Wang et al., 2019), SQuAD 1.1 and SQuAD 2.0 from the Stanford Question answering dataset (Rajpurkar et al., 2016; 2018), and the reading comprehension RACE dataset (Lai et al., 2017). For finetuning, we follow the same procedure as (Liu et al., 2019b). We first perform hyperparameter tuning on batch size and learning rate. Once we obtain the best values, we report the median development set results over 5 different random seeds for initialization. The hyperparameters used for each model and task are provided in the Appendix A. Table 5 shows the development set results for MNLI, QQP, SQuAD 1.1, and SQuAD 2.0 and test set results for RACE. For the test set results of RACE, we first use the development set to find the checkpoint that gives us the median score on the 5 random seeds and we report the results from that checkpoint on the test set. We also report 5-way ensemble results for the development set of SQuAD and test set of RACE. 

Table 5. Development set results for MNLI, QQP, SQuAD 1.1 and SQuAD 2.0 and test set results for RACE. The trained tokens represents consumed tokens during model pretraining (proportional to batch size times number of iterations) normalized by consumed tokens during model pretraining for our 336M model.  

<table><tr><td>Model</td><td>trained tokens ratio</td><td>MNLI m/mm accuracy (dev set)</td><td>QQP accuracy (dev set)</td><td>SQuAD 1.1 F1/EM (dev set)</td><td>SQuAD 2.0 F1/EM (dev set)</td><td>RACE m/h accuracy (test set)</td></tr><tr><td>RoBERTa (Liu et al., 2019b)</td><td>2</td><td>90.2 / 90.2</td><td>92.2</td><td>94.6 / 88.9</td><td>89.4 / 86.5</td><td>83.2 (86.5 / 81.8)</td></tr><tr><td>ALBERT (Lan et al., 2019)</td><td>3</td><td>90.8</td><td>92.2</td><td>94.8 / 89.3</td><td>90.2 / 87.4</td><td>86.5 (89.0 / 85.0)</td></tr><tr><td>XLNet (Yang et al., 2019)</td><td>2</td><td>90.8 / 90.8</td><td>92.3</td><td>95.1 / 89.7</td><td>90.6 / 87.9</td><td>85.4 (88.6 / 84.0)</td></tr><tr><td>Megatron-336M</td><td>1</td><td>89.7 / 90.0</td><td>92.3</td><td>94.2 / 88.0</td><td>88.1 / 84.8</td><td>83.0 (86.9 / 81.5)</td></tr><tr><td>Megatron-1.3B</td><td>1</td><td>90.9 / 91.0</td><td>92.6</td><td>94.9 / 89.1</td><td>90.2 / 87.1</td><td>87.3 (90.4 / 86.1)</td></tr><tr><td>Megatron-3.9B</td><td>1</td><td>91.4 / 91.4</td><td>92.7</td><td>95.5 / 90.0</td><td>91.2 / 88.5</td><td>89.5 (91.8 / 88.6)</td></tr><tr><td>ALBERT ensemble (Lan et al., 2019)</td><td></td><td></td><td></td><td>95.5 / 90.1</td><td>91.4 / 88.9</td><td>89.4 (91.2 / 88.6)</td></tr><tr><td>Megatron-3.9B ensemble</td><td></td><td></td><td></td><td>95.8 / 90.5</td><td>91.7 / 89.0</td><td>90.9 (93.1 / 90.0)</td></tr></table>

From Table 5 we observe that (a) as the model size increases, the downstream task performance improves in all cases, (b) our 3.9B model establishes state of the art results on the development set compared to other BERT based models, and (c) our 3.9B model achieves both single model as well as ensembled SOTA results on RACE test set.


# 6. Conclusion and Future Work
In this work, we successfully surpassed the limitations posed by traditional single-GPU-per-model training by implementing model parallelism with only a few modifications to the existing PyTorch transformer implementations. We efficiently trained transformer based models up to 8.3 billion parameter on 512 NVIDIA V100 GPUs with 8-way model parallelism and achieved up to 15.1 PetaFLOPs sustained over the entire application. 
>  本工作中，我们通过仅对现有的 PyTorch transformer 实现进行了少量修改实现模型并行，成功突破了传统单 GPU 模型训练的限制
>  我们在 512 个 V100 GPUs 上使用 8 路模型并行，训练了 8.3B 的模型，在整个过程中实现了高达 15.1 PFLOPs 的持续算力

We also showed that for BERT models, careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased accuracies as the model size increases. We study the effect of model size on down-stream task accuracy and achieve far superior results on downstream tasks and establish new SOTA for WikiText103, LAMBADA, and RACE datasets. Finally, we open sourced our code to enable future work leveraging model parallel transformers.
>  我们还发现，对于 BERT 模型而言，对 BERT 类模型中 layer norm 位置的细致处理对于随着模型规模增大而提高准确率至关重要

There are several directions for future work. Continuing to increase the scale of pretraining is a promising line of investigation that will further test existing deep learning hardware and software. To realize this, improvements in the efficiency and memory footprint of optimizers will be needed. In addition, training a model with more than 16 billion parameters will demand more memory than is available within 16 GPUs of a DGX-2H box. For such models, a hybrid intra-layer and inter-layer model parallelism along with inter-node model parallelism would be more suitable. Three other directions of investigation include (a) pretraining different model families (XLNet, T5), (b) evaluating performance of large models across more difficult and diverse downstream tasks (e.g. Generative Question Answering, Summarization, and Conversation), and (c) using knowledge distillation to train small student models from these large pretrained teacher models.
>  进一步扩大预训练规模需要提高优化器的效率并减少内存占用

# A. BERT Finetuning Hyperparameters
Table 6 presents the hyperparameters used for each model and task during finetuning.

Table 6. Hyperparameters for finetuning BERT model on downstream tasks.  

<table><tr><td>Task</td><td>Model</td><td>Batch size</td><td>Learning rate</td><td>Training epochs</td></tr><tr><td rowspan="2">MNLI</td><td>336M</td><td>128</td><td rowspan="2">1e-5</td><td rowspan="2">10</td></tr><tr><td>1.3B</td><td>3.8B</td></tr><tr><td rowspan="3">QQP</td><td>336M</td><td>128</td><td>5e-5</td><td>12</td></tr><tr><td>1.3B</td><td>3.8B</td><td>128</td><td rowspan="2">3e-5</td></tr><tr><td>336M</td><td>128</td><td>4e-5</td></tr><tr><td rowspan="3">SQUAD 1.1</td><td>336M</td><td>1.3B</td><td>64</td><td>3e-5</td></tr><tr><td>3.8B</td><td>3.8B</td><td>48</td><td>3e-5</td></tr><tr><td>336M</td><td>1.3B</td><td>48</td><td>1e-5</td></tr><tr><td rowspan="2">SQUAD 2.0</td><td>336M</td><td>1.3B</td><td>64</td><td>3e-5</td></tr><tr><td>3.8B</td><td>3.8B</td><td>48</td><td>1e-5</td></tr><tr><td rowspan="3">RACE</td><td>336M</td><td>1.3B</td><td>32</td><td>2e-5</td></tr><tr><td>1.3B</td><td>3.8B</td><td>16</td><td>1e-5</td></tr><tr><td>3.8B</td><td>3.8B</td><td>32</td><td>2e-5</td></tr></table>

# B. Model Parallel Supplementary Material
In this section, we present further details about the hybrid model and data parallelism and handling random number generation.

## B.1. Hybrid Model and Data Parallelism

![[pics/Megatron-Fig8.png]]

Model parallelism is orthogonal to data parallelism, and so we can use both simultaneously to train large models in a reasonable amount of time. 
>  模型并行和数据并行是正交的，我们可以同时使用二者

Figure 8 shows a grouping of GPUs for hybrid model and data parallelism. Two or more GPUs within the same server form model parallel groups (for example GPUs 1 to 8 in Figure 8), and contain one instance of the model distributed across these GPUs. The remaining GPUs, which could be within the same server but more typically are located in other servers, run additional model parallel groups. 
>  Fig8 中，同一个服务器内的多个 GPU 形成模型并行组，这些 GPU 上分布着同一个模型的一个实例，其余的 GPU 通常位于其他服务器，运行其他模型并行组

GPUs with the same position in each of the model parallel groups (for example GPUs 1, 9, ..., 505 in Figure 8) form data parallel groups so that all GPUs within a data parallel group hold the same model parameters. 
>  每个模型并行组中处于相同位置的 GPU 组成数据并行组，这些数据并行组中的 GPU 都保存相同的模型参数

During back propagation we run multiple gradient all-reduce operations in parallel to reduce weight gradients within each distinct data parallel group. 
>  反向传播时，我们并行运行多个梯度 all-reduce 来规约各个数据并行组内的权重梯度

The total number of required GPUs is the product of the number of model and data parallel groups. For example, for the 8.3 billion parameter model we use 8 GPUs per model parallel group and 64-way data parallelism, for a total of 512 GPUs. 
>  GPUs 的所需总数是模型并行组的数量乘以数据并行组的数量

All communication is implemented in PyTorch by Python calls to NCCL. GPUs within each model parallel group perform all-reduces amongst all GPUs within the group. For data parallelism, each of the all-reduce operations takes place with one of the GPUs from each model parallel group.
>  所有的通讯都通过 PyTorch 调用 NCCL 实现
>  每个模型并行组内的 GPUs 会在组内执行 all-reduce 操作
>  每个数据并行组内会在不同服务器上的各个 GPU 之间执行 all-reduce 操作

## B.2. Model Parallel Random Number Generation
Techniques that utilize random number generation, such as dropout, are a staple of modern deep learning training. Transformers have dropout layers outside the model parallel regions before residual connections and within model parallel regions in the self attention block. Because some dropout layers are in a model parallel region, while others are not, we need to treat random number generation carefully to ensure dropout works correctly. 
>  transformers 在残差连接之前有 dropout 层，位于模型并行区域之外，在 self attention block 中也有 dropout 层，位于模型并行区域之内
>  故我们需要仔细处理随机数生成，确保 dropout 正常工作

To synchronize residual connection dropout across model parallel workers we seed the random number generators at the beginning of training with the same seed. This results in identical dropout patterns across all model parallel workers. 
>  为了在模型并行的各个 workers 之间同步残差连接 dropout，我们在训练开始时使用相同的种子，确保模型并行 workers 具有相同的 dropout 模式

However, dropout within a model parallel region should result in different random patterns for each worker to achieve randomness across the entire operation. To achieve this we maintain a separate random number generator for dropout within model parallel regions. This random number generator is uniquely seeded for each model parallel worker.
>  同时，在模型并行区域内的 dropout 应该为每个 worker 生成不同的随机模型，以保持随机性
>  为此我们在模型并行区域内使用单独的随机数生成器，每个模型并行 worker 都有唯一的种子

# C. Text Samples
Below are some text samples generated by Megatron-LM using a context prompt. Some of the texts are cut short.

# D. Further Scaling Analysis
In this section we study the effect of number of attention heads on the scaling results. We also present strong scaling results for our 1.2 billion parameter model.

## D.1. Attention Heads and Scaling
This section studies the effect of attention heads on model parallel scaling. To this end, we consider the 8.3 billion parameter configuration with 8-way model parallelism and vary the number of heads from 16 to 32. The results are presented in Table 7. 

As the number of attention heads increases, some of the GEMMS inside the self-attention layer become smaller and also the number of elements in the self attention softmax increases. This results in a slight decrease in scaling efficiency. Future research should be wary of this hyperparameter to design large transformer models that balance model speed and model accuracy.
>  随着 attention head 数量增加，self attention 层内的一些 GEMMS 变小，同时 softmax 元素也增加，这导致了拓展效率的轻微下降

Table 7. Effect of number of attention heads on scaling on 8.3 billion of parameters with 8-way model parallelism.  

<center><table><tr><td>Attention heads</td><td>Hidden size per head</td><td>Scaling Efficiency</td></tr><tr><td>16</td><td>192</td><td>82%</td></tr><tr><td>24</td><td>128</td><td>80%</td></tr><tr><td>32</td><td>96</td><td>77%</td></tr></table></center>

## D.2. Strong Scaling
Our model parallelism is primarily designed to enable training models larger than what can fit in the memory of a single GPU, but it can also accelerate the training of smaller models without increasing the batch size. To measure this acceleration we train a model with a fixed 1.2 billion parameters. We use a fixed batch size of 8 samples per iteration and increase the number of GPUs using model parallelism. The results are listed in Table 8. Using two GPUs makes training  $64\%$  faster. Above that we see diminishing returns as the per-GPU computation decreases and the memory bandwidth and communication overheads begin to dominate.
>  我们的模型并行主要针对于训练一个 GPU 放不下的模型，但也可以在不增大 batch size 的情况下 (保持计算量不变，增加计算核心数) 增大小模型的训练

Table 8. Speedup obtained for the 1.2 billion parameters model using model parallelism while keeping the batch size constant.  

<center><table><tr><td># of GPUs</td><td>1</td><td>2</td><td>4</td><td>8</td></tr><tr><td>Speedup</td><td>1.0</td><td>1.64</td><td>2.34</td><td>2.98</td></tr></table></center>


# E. Evaluating Language Models Using WikiText103 and LAMBADA
In this section we detail our evaluation methodology for the WikiText103 dataset (Merity et al., 2016) and cloze-style prediction accuracy on the LAMBADA dataset(Paperno et al., 2016).

## E.1. Wikitext103 Perplexity
WikiText103 perplexity is an evaluation criterion that has been well studied over the past few years since the creation of the benchmark dataset. Perplexity is the exponentiation of the average cross entropy of a corpus (Mikolov et al., 2011). This makes it a natural evaluation metric for language models which represent a probability distribution over entire sentences or texts.

$$
PPL = \exp (-\frac{1}{T_o}\sum_t^T\log P(t|0:t -1)) \tag{4}
$$

To calculate perplexity in (4) we tokenize the WikiText103 test corpus according to our subword vocabulary and sum the cross entropy loss from each token  $[0,T]$  .We then normalize the cross entropy loss by the number of tokens in the original tokenization scheme  $T_{o}$  . The WikiText103 test corpus already comes pre-tokenized with word level tokens that prior works have used to compute perplexity. To evaluate our models' perplexities on a level playing field with prior works we must normalize by the original number of tokens,  $T_{o}$  rather than the number of tokens,  $T$  actually in the tokenized data fed as input to our model. This pre-tokenization also introduces artifacts in the text that are not present in our training data. To alleviate this distributional mismatch, we first preprocess the WikiText103 test dataset with invertible detokenizers to remove various artifacts related to punctuation and whitespace. The value of  $T_{o}$  is calculated before this preprocessing. For WikiText103's test set  $T_{o} = 245566$  and  $T = 270329$

We must also make one further transformer-specific modification to the perplexity calculation. Unlike RNN-based language models, transformers operate on a fixed window input size. Therefore they cannot fully calculate  $P(t|0:t -1)$  and can only calculate  $P(t|t -w:t -1)$  where  $w$  is the size of our context: 1024 tokens. However, calculating this value for every token in our dataset is prohibitively expensive since we must compute approximately  $T$  evaluations of a  $w$  sized context. To evaluate our models efficiently we take a middle ground approach termed overlapping evaluation where we advance the sliding window by some overlap  $o$  each time and only compute the cross entropy losses corresponding to the last  $o$  tokens of the window. In our experiments we utilize an overlap  $o$  of 32, and compute losses over all sliding windows in such a fashion.

## E.2. LAMBADA Cloze Accuracy
The capability to handle long term contexts is crucial for state of the art language models and is a necessary prerequisite for problems like long-form generation and document-based question answering. Cloze-style datasets like LAMBADA are designed to measure a model's ability to operate in and reason about these types of long term contexts. Cloze-style reading comprehension uses a context of word tokens  $x = x_{1:t}$  with one token  $x_{j}$  masked; the models objective is to correctly predict the value of the missing  $j^{\mathrm{th}}$  token. To accurately predict the missing token, the model requires an in-depth understanding of the surrounding context and how language should be used in such a context. LAMBADA uses cloze-style reading comprehension to test generative left-to-right language models by constructing examples of 4-5 sentences where the last word in the context  $x_{t}$  is masked. Our models utilize subword units, so for LAMBADA evaluation we utilize the raw, unprocessed LAMBADA dataset and require that our model predict the multiple subword tokens that make up the word token. We use teacher forcing, and consider an answer correct only when all output predictions are correct. This formulation is equivalent to the original task of word token prediction.