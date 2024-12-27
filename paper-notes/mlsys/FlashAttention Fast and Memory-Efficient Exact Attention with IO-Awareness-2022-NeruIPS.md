# Abstract 
Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. 
>Transformers在长序列上运行缓慢且占用大量内存，因为自注意力的时间和内存复杂度与序列长度呈二次方关系

Approximate attention methods have attempted to address this problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms $I O-$ aware —accounting for reads and writes between levels of GPU memory. 
> 近似注意力方法试图通过牺牲模型质量来降低计算复杂度，但通常无法实现实际速度提升 (wall-clock speedup)，我们认为一个缺失的原则是使注意力算法具有 IO 意识——考虑 GPU 内存层次之间的读写操作

We propose FlashAttention , an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention , showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. 
>我们提出 FlashAttention，这是一种 IO 意识的精确注意力算法，它使用 tiling 减少 GPU HBM 和 GPU 片上 SRAM 之间的内存读写次数
>我们分析了 FlashAttention 的 IO 复杂度，表明它比标准注意力需要更少的 HBM 访问，并且其 IO 复杂度对于一系列 SRAM 大小都是最优的

We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method.
>我们还将 FlashAttention 拓展到块稀疏注意力，得到了一种比任何现有近似注意力方法都快的近似注意力算法

 FlashAttention trains Transformers faster than existing baselines: $15\%$ end-to-end wall-clock speedup on BERT-large (seq. length compared to the MLPerf 1.1 training speed record, $3\times$ speedup on GPT-2 (seq. length 1K), and 2.4 × speedup on long-range arena (seq. length 1K-4K). 
>FlashAttention 训练 Transformer 的速度比现有基线更快：
>与 MLPerf 1.1 训练速度记录相比，在 BERT-large (序列长度 512) 上实现了 15%的端到端实际速度提升 (end-to-end wall-clock speedup)；
>GPT-2 (序列长度 1K) 的速度提升了 3 倍，long-range arena (序列长度 1K-4K) 的速度提升了 2.4 倍

FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, $61.4\%$ accuracy) and Path-256 (seq. length 64K, $63.1\%$ accuracy). 
>FlashAttention 和块稀疏 FlashAttention 使 Transformer 能够处理更长的上下文，从而产生更高质量的模型 (GPT-2 上的困惑度提高了 0.7，长文档分类提高了 6.4 个百分点) 和全新的能力：首次实现在 Path-X 挑战 (序列长度 16K，准确率 61.4%) 和 Path-256 (序列长度 64K，准确率 63.1%) 上实现比随机猜测更好的性能的 Transformer

# 1 Introduction 
![[FlashAttention-Fig1.png]]

Transformer models [ 82 ] have emerged as the most widely used architecture in applications such as natural language processing and image classification. Transformers have grown larger [ 5 ] and deeper [ 83 ], but equipping them with longer context remains difficult [ 80 ], since the self-attention module at their heart has time and memory complexity quadratic in sequence length. 
> Transformer 模型[82]已成为自然语言处理和图像分类等应用中最广泛使用的架构，Transformer 已经变得更大[5]和更深[83]，但是让它们具有更长的上下文 (longer context) 仍然很困难[80]，因为它们核心的自注意力模块在序列长度上具有二次时间和内存复杂度

An important question is whether making attention faster and more memory-efficient can help Transformer models address their runtime and memory challenges for long sequences. Many approximate attention methods have aimed to reduce the compute and memory requirements of attention. These methods range from sparse-approximation [ 51 , 74 ] to low-rank approximation [ 12 , 50 , 84 ], and their combinations [ 3 , 9 , 92 ]. Although these methods reduce the compute requirements to linear or near-linear in sequence length, many of them do not display wall-clock speedup against standard attention and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not correlate with wall-clock speed) and tend to ignore overheads from memory access (IO). 
>一个重要的问题是，使注意力更快、更节省内存是否可以帮助Transformer模型解决长序列的运行时间和内存挑战
>许多近似注意力方法旨在减少注意力的计算和内存需求，这些方法包括稀疏近似[51,74]到低秩近似[12,50,84]，以及它们的组合[3,9,92]，尽管这些方法将计算需求减少到线性或接近线性的序列长度，但其中许多并没有显示出与标准注意力相比的实时速度提升，并且没有得到广泛采用
>一个主要原因是它们专注于FLOP减少(这可能与实时速度无关)，并且倾向于忽略来自内存访问(IO)的开销

In this paper, we argue that a missing principle is making attention algorithms $I O$ -aware [ 1 ]—that is, carefully accounting for reads and writes to diﬀerent levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM [45], Figure 1 left). 
>在本文中，我们认为一个被忽视原则是使注意力算法IO感知[1]——即仔细考虑对不同级别快速和慢速内存的读写(例如，在快速的GPU片上SRAM和相对较慢的GPU高带宽内存[45]之间，如Figure 1 left所示)

On modern GPUs, compute speed has out-paced memory speed [ 61 , 62 , 63 ], and most operations in Transformers are bottlenecked by memory accesses [ 43 ]. IO-aware algorithms have been critical for similar memory-bound operations, when reading and writing data can account for a large portion of the runtime—such as database joins [ 71 ], image processing [ 70 ], numerical linear algebra [ 4 ], and more [ 40 , 85 ]. However, common Python interfaces to deep learning such as PyTorch and Tensorﬂow do not allow fine-grained control of memory access. 
> 在现代 GPU 上，计算速度已经超过了内存速度，Transformer 中的大多数运算都受到内存访问的限制
> 当读写数据可以占据大部分的运行时间时——例如数据库连接 (database joins)[71]、图像处理 (image processing)[70]、数值线性代数[4]等[40,85]，IO 感知算法对于类似的受内存限制的运算就是至关重要的，然而，像 PyTorch 和 Tensorflow 这样的常见 Python 深度学习接口不允许对内存访问进行细粒度控制

We propose FlashAttention , a new attention algorithm that computes exact attention with far fewer memory accesses. Our main goal is to avoid reading and writing the attention matrix to and from HBM. This requires (i) computing the softmax reduction without access to the whole input (ii) not storing the large intermediate attention matrix for the backward pass.  
>我们提出了 FlashAttention，这是一种新的注意力算法，它使用更少的内存访问进行精确的注意力计算，我们的主要目标是避免读写注意力矩阵到 HBM 中，这需要：
>1. 在不访问整个输入的情况下计算softmax归约
>2. 不为反向传播存储大型中间注意力矩阵

We apply two well-established techniques to address these challenges. (i) We restructure the attention computation to split the input into blocks and make several passes over input blocks, thus incrementally performing the softmax reduction (also known as tiling ). (ii) We store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the backward pass, which is faster than the standard approach of reading the intermediate attention matrix from HBM.
>我们应用两种成熟的技术来解决这些挑战：
>1. 我们重新组织注意力计算，将输入分成块，并在输入块上进行多次传递 (make several passes over input blocks)，从而逐步执行 softmax 归约 (该技术也称为平铺 tiling)
>2. 我们存储前向传播中的 softmax 归一化因子，以便在反向传播中快速在片上重新计算注意力，这比从 HBM 读取中间注意力矩阵的标准方法更快

We implement FlashAttention in CUDA to achieve fine-grained control over memory access and fuse all the attention operations into one GPU kernel. Even with the increased FLOPs due to recomputation, our algorithm both runs faster (up to 7.6x on GPT-2 [ 67 ], Figure 1 right) and uses less memory —linear in sequence length—than standard attention, thanks to the massively reduced amount of HBM access.
>我们在 CUDA 中实现了 FlashAttention，以实现对内存访问的细粒度控制，并将所有注意力操作融合到一个 GPU 内核中，即使由于重复计算 (recomputation) 而增加了 FLOPs，我们的算法相较于标准注意力在运行速度上更快 (在 GPT-2[67]上高达 7.6 倍，如 Figure 1 right 所示) 并且使用更少的内存——线性于序列长度 (linear in sqeuence length)，这要归功于大大减少的 HBM 访问量

We analyze the IO complexity [ 1 ] of FlashAttention , proving that it requires $O(N^{2}d^{2}M^{-1})$ HBM accesses where 𝑑 is the head dimension and 𝑀 is the size of SRAM, as compared to $\Omega(N d+N^{2})$ of standard attention. For typical values of $d$ and $M$ , FlashAttention requires many times fewer HBM accesses compared to standard attention (up to $9\times$ fewer, as shown in Fig. 2). Moreover, we provide a lower bound, showing that no exact attention algorithm can asymptotically improve on the number of HBM accesses over all SRAM sizes. 
>我们分析了 FlashAttention 的 IO 复杂度[1]，证明它需要 $O(N^2d^2M^{-1})$ HBM 访问，其中 $d$ 是头维度，$M$ 是 SRAM 的大小，而标准注意力则需要 $\Omega(Nd+N^2)$
>对于典型的 $d$ 和 $M$ 值，FlashAttention 需要比标准注意力少得多的 HBM 访问 (高达 9 倍，如 Figure 2 所示)
>此外，我们提供了一个下界，表明没有精确的注意力算法可以在所有 SRAM 大小上渐近地改进 HBM 访问次数

We also show that FlashAttention can serve as a useful primitive for realizing the potential of approximate attention algorithms by overcoming their issues with memory access overhead. As a proof of concept, we implement block-sparse FlashAttention , a sparse attention algorithm that is 2-4 × faster than even FlashAttention , scaling up to sequence length of 64k. We prove that block-sparse FlashAttention has better IO complexity than FlashAttention by a factor proportional to the sparsity ratio. 
>我们还展示了 FlashAttention 通过克服潜在的近似注意力算法的内存访问开销的问题，可以作为实现它们的有用原语 (primitive)
>作为一个概念验证，我们实现了块稀疏 FlashAttention，这是一种稀疏注意力算法，比 FlashAttention 快 2-4 倍，可扩展到 64k 的序列长度，我们证明了块稀疏 FlashAttention 的 IO 复杂度比 FlashAttention 好一个与稀疏比成比例的因素

We discuss further extensions to other operations (attention on multi-GPU, kernel regression, block-sparse matrix multiply) in Section 5. We open-source FlashAttention to make it easier to build on this primitive. 
>我们在第 5 节中讨论了对其他运算 (多 GPU 上的注意力、核回归、块稀疏矩阵乘法) 的进一步扩展

We empirically validate that FlashAttention speeds up model training and improves model quality by modeling longer context. We also benchmark the runtime and memory footprint of FlashAttention and block-sparse FlashAttention compared to prior attention implementations. 
>我们经验上地验证了 FlashAttention 加速了模型训练并通过建模更长的上下文提高了模型质量，我们还对FlashAttention和块稀疏FlashAttention的运行时间和内存占用(memory footprint)与以前的注意力实现进行了基准测试和比较

- Faster Model Training. FlashAttention trains Transformer models faster in wall-clock time. We train BERT-large (seq. length 512) $15\%$ faster than the training speed record in MLPerf 1.1 [ 58 ], GPT2 (seq. length 1K) $3\times$ faster than baseline implementations from HuggingFace [ 87 ] and Megatron-LM [ 77 ], and long-range arena (seq. length 1K-4K) 2.4 × faster than baselines. 
>更快的模型训练
>FlashAttention 在实时时间中可以更快训练 Transformer 模型
>我们训练 BERT-large (序列长度 512) 比在 MLPerf 1.1[58]中的训练速度记录快 15%，训练 GPT2 (序列长度 1K) 比在 HuggingFace[87]和 Megatron-LM[77]中的基线实现快 3 倍，长距离竞技场 (序列长度 1K-4K) 比基线快 2.4 倍

- Higher Quality Models. FlashAttention scales Transformers to longer sequences, which improves their quality and enables new capabilities. We observe a 0.7 improvement in perplexity on GPT-2 and 6.4 points of lift from modeling longer sequences on long-document classification [13]. FlashAttention enables the first Transformer that can achieve better-than-chance performance on the Path-X [ 80 ] challenge, solely from using a longer sequence length (16K). Block-sparse FlashAttention enables a Transformer to scale to even longer sequences (64K), resulting in the first model that can achieve better-than-chance performance on Path-256. 
>更高质量的模型
>FlashAttention 将 Transformer 扩展到更长的序列，这提高了它们的质量并启用了新功能 (enables new capabilities)
>我们在 GPT-2上观察到困惑度提高了0.7，在长文档分类[13]上通过建模更长的序列提高了6.4个百分点的准确率，FlashAttention 实现了第一个仅通过使用更长的序列长度 (16K)，就能在 Path-X[80]挑战中达到比偶然更好的性能 (better-than-chance) 的 Transformer，块稀疏 FlashAttention 使 Transformer 能够扩展到更长的序列 (64K)，从而实现了第一个可以在 Path-256上达到比偶然更好的性能的模型

- Benchmarking Attention. FlashAttention is up to $3\times$ faster than the standard attention implemen- tation across common sequence lengths from 128 to 2K and scales up to 64K. Up to sequence length of 512, FlashAttention is both faster and more memory-efficient than any existing attention method, whereas for sequence length beyond 1K, some approximate attention methods (e.g., Linformer) start to become faster. On the other hand, block-sparse FlashAttention is faster than all existing approximate attention methods that we know of. 
>基准测试注意力计算
>FlashAttention 在从 128 到 2K 的常见序列长度上比标准注意力实现快 3 倍，并且可以扩展到 64K
>在序列长度小于 512 时，FlashAttention 在速度和内存效率方面都比任何现有的注意力方法更快，而对于超过 1K 的序列长度，一些近似注意力方法 (例如，Linformer) 开始变得更快，另一方面，块稀疏 FlashAttention 比我们所知道的所有现有近似注意力方法都快

# 2 Background 
We provide some background on the performance characteristics of common deep learning operations on modern hardware (GPUs). We also describe the standard implementation of attention. 
## 2.1 Hardware Performance 
We focus here on GPUs. Performance on other hardware accelerators are similar [46, 48]. 
>我们在本小节关注 GPU，其他硬件加速器 (hardware accelerators) 的性能也类似[46, 48]

**GPU Memory Hierarchy.** The GPU memory hierarchy (Fig. 1 left) comprises multiple forms of memory of diﬀerent sizes and speeds, with smaller memory being faster. As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [ 44 , 45 ]. The on-chip SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size. As compute has gotten faster relative to memory speed [ 61 , 62 , 63 ], operations are increasingly bottlenecked by memory (HBM) accesses. Thus exploiting fast SRAM becomes more important. 
>**GPU Memory Hierarchy**
>GPU 内存层次结构 (Figure 1 left) 包括不同大小和速度的多种内存，较小的内存速度更快，例如，A100 GPU 拥有40-80GB 的高带宽内存 (HBM)，带宽为1.5-2.0TB/s，每个108个流式多处理器各有192KB 的片上 SRAM，带宽估计约为19TB/s[44, 45]
>片上 SRAM 比 HBM 快一个数量级 (an order of magnitude)，但大小小很多个数量级，随着计算相对于内存速度变得更快[61, 62, 63]，运算 (operations) 越来越受到内存 (HBM) 访问的限制，因此，利用快速 SRAM 变得更加重要

**Execution Model.** GPUs have a massive number of threads to execute an operation (called a kernel). Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM. 
>**Execution Model**
>GPU有大量的线程来执行一个运算(称为内核)，每个内核从HBM加载输入到寄存器和SRAM，计算，然后将输出写回HBM

**Performance characteristics.** Depending on the balance of computation and memory accesses, op- erations can be classified as either compute-bound or memory-bound. This is commonly measured by the arithmetic intensity [85], which is the number of arithmetic operations per byte of memory access.
>**Performance characteristics**
>根据计算和内存访问的平衡，运算可以被分类为计算受限(compute-bound)或内存受限(memory-bound)，这通常通过算术密度来衡量，即每个字节的内存访问的算术操作数

 1. Compute-bound: the time taken by the operation is determined by how many arithmetic operations there are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner dimension, and convolution with large number of channels.
 >计算受限 (compute-buond)：操作所需时间由算术操作的数量决定，而访问 HBM 的时间要小得多，典型的例子包括具有大的内维度的矩阵乘法和具有大量通道的卷积

 2. Memory-bound: the time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. Examples include most other operations: elementwise (e.g., activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm). Kernel fusion. The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise operations [ 53 , 65 , 75 ]. 
 >内存受限 (memory-bound)：操作所需时间由内存访问次数决定，而计算所花费的时间要小得多，例子包括大多数其他操作：逐元素操作 (例如，激活，dropout) 和归约操作 (例如，求和，softmax，批量归一化，层归一化)

**Kernel fusion.** The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise operations [53, 65, 75].
>**Kernel funsion**
>加速内存受限操作的最常见方法是内核融合：如果有多个运算应用于同一输入，则输入可以从 HBM 加载一次，而不是为每个运算都加载一次
>编译器可以自动融合许多逐元素操作[53, 65, 75]，然而，在模型训练的背景下，中间值仍需要写回 HBM 以保存用于反向传递 (backward pass)，降低了简单内核融合的有效性 (effectiveness of naive kernel fuse)

## 2.2 Standard Attention Implementation 
Given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ , where $N$ is the sequence length and $d$ is the head dimension, we want to compute the attention output $\mathbf{O}\in\mathbb{R}^{N\times d}$ : 

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$ 
where softmax is applied row-wise. 
>给定输入序列 $Q,K,V \in \mathbb R^{N\times d}$，其中 $N$ 是序列长度，$d$ 是头维度，我们想要计算注意力输出 $O\in \mathbb R^{N\times d}$，
>其中softmax是逐行应用的(applied row-wise)

Sta attention impleme materi he matrices $\mathbf{S}$ and $\mathbf{P}$ to HBM, which takes $O(N^{2})$ memory. Often $N\gg d$ (e.g., for GPT2, $N=1024$ = and $d=64$ ). We describe the standard attention implementation in Algorithm 0. As some or most of the operations are memory-bound (e.g., softmax), the large number of memory accesses translates to slow wall-clock time. 
> 标准注意力实现将矩阵 $S$ 和 $P$ 存储 (materialize) 到 HBM 中，这需要 $O(N^2)$ 的内存，通常来说 $N\gg d$ (例如，对于 GPT2，$N=1024$，$d=64$)
> 我们在 Algorithm 0 中描述了标准注意力实现，对于标准的注意力实现，由于一些或全部操作是内存受限的(例如，softmax)，大量的内存访问会导致慢的实际运行时间

This problem is exacerbated by other elementwise operations applied to the attention matrix, such as masking applied to S or dropout applied to $\mathbf{P}$ . As a result, there have been many attempts to fuse several elementwise operations, such as fusing masking with softmax [77]. 
>而这个问题还会被应用于注意力矩阵的其他逐元素操作而加剧，例如应用于 $S$ 的掩码(masking)或应用于 $P$ 的dropout，因此，已经有许多工作尝试融合几个逐元素操作，例如将掩码与softmax融合[77]

In Section 3.2, we will show that the standard attention implementation performs HBM accesses quadratic in the sequence length $N$ . We also compare the number of FLOPs and number of HBM accesses of standard attention and of our method ( FlashAttention ). 
>在第3.2节中，我们将展示标准注意力实现执行的 HBM 访问是序列长度 $N$ 的二次方，我们还比较了标准注意力和我们的方法 (FlashAttention) 的 FLOPs 数量和 HBM 访问数量

**Algorithm 0** Standard Attention Implementation
**Require:** Matrices $Q,K,V \in \mathbb R^{N\times d}$ in HBM
  1: Load $Q, K$ by blocks from HBM, computes $S = QK^T$, writes $S$ to HBM.
  2: Read $S$ from HBM, compute $P = \text{softmax}(S)$, write $P$ to HBM.
  3: Load $P$ and $V$ by blocks from HBM, compute $O = PV$, write $O$ to HBM.
  4: Return $O$.

# 3 FlashAttention: Algorithm, Analysis, and Extensions 
We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass. This yields an attention algorithm that is both memory efficient and faster in wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM accesses compared to standard attention. We further show that FlashAttention can serve as a useful primitive by extending it to handle block-sparse attention. 
>我们将展示如何在使用更少的高带宽存储器(HBM)读写次数和不存储用于反向传播的大型中间矩阵的情况下进行精确的注意力计算，这是一个既节省内存又在实际时间上更快的注意力算法
>我们分析了其I/O复杂性，表明我们的方法与标准注意力相比需要更少的HBM访问，我们通过将其扩展到处理块稀疏注意力，进一步展示了FlashAttention可以作为一个有用的原语

We focus here on the forward pass for ease of exposition; Appendix B contains details for the backward. 
>本节内容专注于前向传播，以便于解释；Appendix B包含了反向传播的详细信息

## 3.1 An Efficient Attention Algorithm With Tiling and Recomputation 
Given the inputs $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ in HBM, we aim to compute the attention output $\mathbf{O}\in\mathbb{R}^{N\times d}$ and write it to HBM. Our goal is to reduce the amount of HBM accesses (to sub-quadratic in $N$ ). 
>给定在HBM中的输入 $Q,K,V \in R^{N×d}$，我们需要计算注意力输出 $O \in \mathbb R^{N\times d}$ 并将其写入HBM，我们的目标是减少 HBM 访问次数(使其在 $N$ 的次方下为次线性 sub-quadratic in $N$)

We apply two established techniques (tiling, recomputation) to overcome the technical challenge of computing exact attention in sub-quadratic HBM accesses. We describe this in Algorithm 1. The main idea is that we split the inputs $\mathbf{Q},\mathbf{K},\mathbf{V}$ into blocks, load them from slow HBM to fast SRAM, then compute the attention output with respect to those blocks. By scaling the output of each block by the right normalization factor before adding them up, we get the correct result at the end. 
>我们应用两种成熟的技术(平铺 tiling、重计算 recomputation)来克服在次线性HBM访问中计算精确注意力的技术挑战，见 Algorithm 1
>其主要思想是我们将输入 $Q, K, V$ 分成块(split into blocks)，将它们从慢速HBM加载到快速SRAM中，然后计算与这些块相关的注意力输出，再在将它们相加之前将每个块的输出乘以正确的归一化因子，我们最终得到了正确的结果

**Tiling.** We compute attention by blocks. Softmax couples columns of $\mathbf{K}$ , so we decompose the large softmax with scaling [51, 60, 66]. 
>我们按块计算注意力
>Softmax将 $K$ 的列耦合在一起，因此我们使用缩放分解了大的Softmax

For numerical stability, the softmax of vector $x\in\mathbb{R}^{B}$ is computed as: 
>为了数值稳定性(numerical stability)，向量 $x \in \mathbb R^{B}$ 的Softmax计算为：

$$
m(x):=\max_i x_i,f(x):=[e^{x_1-m(x)}\dots e^{x_B - m(x)}], \mathscr l(x):=\sum_i f(x)_i, \text{softmax}(x):=\frac {f(x)}{\mathscr l(x)}
$$

( 计算准确性：
$$\text{softmax}(x) = \frac {f(x)}{\mathscr l(x)} = \frac {\frac 1 {e^{m(x)}}[e^{x_1}\dots e^{x_B}]}{\frac 1 {e^{m(x)}}\sum_i e^{x_i}} = \frac {[e^{x_1}\dots e^{x_B}]}{\sum_i e^{x_i}}$$
数值稳定性：
$f(x)_i = e^{x_i- m(x)}$，因为$x_i - m(x) \le 0$，故满足$0\le f(x)_i \le e^{0} = 1$
$\mathscr l(x) = \sum_i f(x)_i$，因为 $0\le f(x)_i \le 1$，故满足 $0\le \mathscr l(x) \le B$ )

For vectors $x^{(1)},x^{(2)}\in\mathbb{R}^{B}$ , we can decompose the softmax of the concatenated $x=\left[x^{(1)}\;x^{(2)}\right]\in\mathbb{R}^{2B}$ as: 
>对于向量 $x^{(1)}, x^{(2)} \in \mathbb R^B$，我们可以将拼接的 $x = [x^{(1)}, x^{(2)}]\in \mathbb R^{2B}$ 的softmax分解为：

$$
\begin{align}
m(x) &= m([x^{(1)}\ x^{(2)}]) = \max(m(x^{(1)}), m(x^{(2)})),\\
f(x) &= [e^{m(x^{(1)})- m(x)}f(x^{(1)})\quad e^{m(x^{(2)})- m(x)}f(x^{(2)})],\\
\\
\mathscr l(x)&= \mathscr l([x^{(1)}\ x^{(2)}]) = e^{m(x^{(1)})-m(x)}\mathscr l(x^{(1)}) + e^{m(x^{(2)})-m(x)}\mathscr l(x^{(2)}),\\
\text{softmax}(x)&=\frac {f(x)}{\mathscr l(x)}
\end{align}
$$

( 计算准确性：
$$
\begin{align}
f(x) &= [e^{m(x^{(1)})- m(x)}f(x^{(1)}), e^{m(x^{(2)})- m(x)}f(x^{(2)})]\\
&=e^{-m(x)}[e^{x^{(1)}_1},\dots ,e^{x^{(1)}_B}, e^{x^{(2)}_1},\dots  ,e^{x^{(2)}_B}]\\
&=e^{-m(x)}[e^{x_1},\dots, e^{x_{2B}}]
\\
\\
\mathscr l(x) &=e^{m(x^{(1)})-m(x)}\mathscr l(x^{(1)}) + e^{m(x^{(2)})-m(x)}\mathscr l(x^{(2)})\\
&=e^{-m{(x)}}(\sum_i e^{x^{(1)}_i} + \sum_i e^{x^{(2)}_i})\\
&=e^{-m(x)}\sum_i e^{x_i}
\\
\\
\text{softmax}(x)&=\frac {f(x)}{\mathscr l(x)}=\frac {[e^{x_1},\dots,e^{x_{2B}}]}{\sum_i e^{x_i}}
\end{align}
$$
数值稳定性：
$f(x) = [e^{m(x^{(1)})- m(x)}f(x^{(1)})\ e^{m(x^{(2)})- m(x)}f(x^{(2)})]$，因为$m(x^{(j)})- m(x)\le 0(j = 1, 2)$，故$0 \le e^{m(x^{(j)})-m(x)}\le 1(j=1,2)$，又$0 \le f(x^{(j)})_i \le 1(j=1,2)$，故满足$0 \le f(x)_i \le 1$
$\mathscr l(x) =  \mathscr l([x^{(1)}\ x^{(2)}]) = e^{-m(x)}\sum_i e^{x_i} = \sum_i e^{x_i-m(x)}$，因为$x_i - m(x)\le 0(i=1,\dots,2B)$，故$0\le e^{x_i-m(x)} \le 1$，故满足$0 \le \mathscr l(x) \le 2B$ )

Therefore if we keep track of some extra statistics $(m(x),\ell(x))$ , we can compute softmax one block at a time. We thus split the inputs $\mathbf{Q},\mathbf{K},\mathbf{V}$ into blocks (Algorithm 1 line 3), compute the softmax values along with extra statistics (Algorithm 1 line 10), and combine the results (Algorithm 1 line 12). 
>因此，如果我们跟踪一些额外的统计数据 $(m(x), \mathscr l(x))$，我们可以一次计算一个块的 softmax
>因此，我们将输入 $Q,K,V$ 分成块(Algorithm 1 line 3)，计算softmax值以及额外的统计数据(Algorithm 1 line 10)，并组合结果(Algorithm 1 line 12)

![[FlashAttention-Fig2.png]]

**Recomputation.** One of our goals is to not store $O(N^{2})$ intermediate values for the backward pass. The backward pass typically requires the matrices ${\bf S},{\bf P}\in\mathbb{R}^{N\times N}$ to compute the gradients with respect to $\mathbf{Q},\mathbf{K},\mathbf{V}$ . However, by storing the output O and the softmax normalization statistics $(m,\ell)$ , we can recompute the attention matrix S and $\mathbf{P}$ easily in the backward pass from blocks of $\mathbf{Q},\mathbf{K},\mathbf{V}$ in SRAM. This can be seen as a form of selective gradient checkpointing [ 10 , 34 ]. While gradient checkpointing has been suggested to reduce the maximum amount of memory required [ 66 ], all implementations (that we know oﬀ) have to trade speed for memory. In contrast, even with more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses (Fig. 2). The full backward pass description is in Appendix B. 
>我们的目标之一是不需要存储 $O(N^2)$ 的中间值来用于反向传播
>反向传播通常需要矩阵 $S,P \in \mathbb R^{N\times N}$ 来计算相对于 $Q, K, V$ 的梯度，然而，通过存储输出 $O$ 和 softmax 归一化统计数据 $(m,\mathscr l)$，我们可以在反向传播中从 SRAM 中的 $Q,K,V$ 块轻松重新计算注意力矩阵 $S$ 和 $P$
>这可以看作是一种选择性的梯度检查点，虽然梯度检查点已被建议用于减少所需的最大内存量，但其所有实现 (我们知道的) 都必须在速度和内存之间进行权衡
>相比之下，即使有了更多的 FLOPs，我们的重新计算由于减少了 HBM 访问 (Figure 2) 而加速了反向传播，完整的反向传播描述在 Appendix B 中

**Implementation details: Kernel fusion.** Tiling enables us to implement our algorithm in one CUDA kernel, loading input from HBM, performing all the computation steps (matrix multiply, softmax, optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout in Appendix B). This avoids repeatedly reading and writing of inputs and outputs from and to HBM. 
>平铺使我们能够在一个 CUDA 内核中实现我们的算法，包括了从 HBM 加载输入，执行所有计算步骤 (矩阵乘法、softmax、可选的掩蔽和 dropout、矩阵乘法)，然后将结果写回 HBM (掩蔽和 dropout 见 Appendix B)，这避免了反复从 HBM 读取和写入输入和输出

We show FlashAttention ’s correctness, runtime, and memory requirement (proof in Appendix C). 
>我们将展示FlashAttention的正确性、运行时间以及内存需求(证明见 Appendix C)

**Theorem 1.** Algorithm 1 returns $\mathbf{O}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$ with $O(N^{2}d)$ FLOPs and requires $O(N)$ additional memory beyond inputs and output. 
> 定理1
> 算法1返回 $\mathbf O = \text{softmax}(\mathbf Q \mathbf K^T)\mathbf V$，FLOPs 为 $O (N^2 d)$，输入输出以外的内存需求为 $O (N)$

![[FlashAttention-Algorithm1.png]]

**Algorithm 1** FlashAttention
**Require:** Matrices $Q,K,V\in \mathbb R^{N\times d}$ in HBM, on-chip SRAM of size $M$.
  1: Set block sizes $B_c = \lceil {\frac M {4d}} \rceil$，$B_r = \min(\lceil {\frac M {4d}} \rceil,d)$.
  2: Initialize $O = (0)_{N\times d}\in \mathbb R^{N\times d}, \mathscr l = (0)_N \in \mathbb R^N, m = (-\infty)_N \in \mathbb R^N$ in HBM.
  3: Divide $Q$ into $T_r = \lceil \frac N {B_r} \rceil$ blocks $Q_1, \dots ,Q_{T_r}$ of size $B_r \times d$ each, and divide $K, V$ into $T_c = \lceil \frac N{B_c} \rceil$ blocks $K_1, \dots, K_{T_c}$ and $V_1, \dots, V_{T_c}$ of size $B_c\times d$ each.
> 划分 $Q, K, V$，划分时保持嵌入维度 $d$ 不变，从序列长度的维度划分
> $Q$ 划分单位为 $B_r \times d$，$K, V$ 划分单位为 $B_c\times d$
> 得到 $T_r$ 个 $Q$ 块，得到 $T_c$ 个 $K, V$ 块

  4: Divide $O$ into $T_r$ blocks $O_i, \dots, O_{T_r}$ of size $B_r \times d$ each, divide $\mathscr l$ into $T_r$ blocks $\mathscr l_i,\dots, \mathscr l_{T_r}$ of size $B_r$ each, divide $m$ into $T_r$ blocks $m_1, \dots, m_{T_r}$ of size $B_r$ each.
> 划分 $O$，划分时保持嵌入维度 $d$ 不变，从序列长度的维度划分
> $O$ 划分单位为 $B_r \times d$
> 得到 $T_r$ 个 $O$ 块，初始值为全零
> 划分 $\mathscr l$，从序列长度的维度划分
> $\mathscr l$ 的划分单位为 $B_r$
> 得到 $T_r$ 个 $\mathscr l$ 块，初始值为全零
> 划分 $m$，从序列长度的维度划分
> $m$ 的划分单位为 $B_r$
> 得到 $T_r$ 个 $m$ 块，初始值为全负无穷

  5: **for** $1\le j \le T_c$ **do**
  6:     Load $K_j, V_j$ from HBM to on-chip SRAM.
> 外层循环：装载 $K, V$ 块到 SRAM
> $K, V$ 块占据空间 $2dB_c= 2d\lceil \frac M {4d} \rceil$
> 因为 $\lceil \frac M {4d} \rceil \ge \frac M {4d}$，故 $2d\lceil \frac M {4d} \rceil \ge \frac M 2$

  7:     **for** $1\le i \le T_r$ **do**
  8:         Load $Q_i, O_i, \mathscr l_i, m_i$ from HBM to on-chip SRAM.
  9:         On chip, computes $S_{ij} = Q_iK^T_j \in \mathbb R^{B_r\times B_c}$.
 10:        On chip, compute $\tilde m_{ij} = \text{rowmax}(S_{ij}) \in \mathbb R^{B_r}$，$\tilde P_{ij} = \exp(S_{ij}-\tilde m_{ij})\in \mathbb R^{B_r\times B_c}$(pointwise)，$\tilde {\mathscr l}_{ij} = \text{rowsum}(\tilde P_{ij}) \in \mathbb R^{B_r}$.
 11:         On chip, compute $m_i^{new} = \max(m_i, \tilde m_{ij})\in \mathbb R^{B_r}, \mathscr l_{i}^{new} = e^{m_i - m_i^{new}}\mathscr l_i + e^{\tilde m_{ij} - m_i^{new}}\tilde {\mathscr l}_{ij} \in \mathbb R^{B_r}$.
 12:        Write $O_i \leftarrow \text{diag}(\mathscr l_i^{new})^{-1}(\text{diag}(\mathscr l_i)e^{m_i - m_i^{new}}O_i+e^{\tilde m_{ij}- m_i^{new}}\tilde P_{ij} V_j)$ to HBM.
 13:       Write $\mathscr l_i \leftarrow \mathscr l_i^{new}, m_i \leftarrow m_i^{new}$ to HBM.
> 内层循环：装载 $Q, O,\mathscr l, m$ 块到 SRAM
> $Q, O$ 块占据空间 $2dB_r = 2d\min (\lceil \frac M {4d} \rceil, d)$，$\mathscr l, m$ 块占据空间 $2B_r = 2\min (\lceil \frac {M}{4d} \rceil, d)$
> 
> 在片上计算 $S$ 块：$S = QK^T \in \mathbb R^{B_r\times B_c}$ (score 是 final 的)
> 按行取最大值: $\tilde m = \text{rowmax}(S) \in \mathbb R^{B_r}$，
> 按行规范化 $S$: $S = S - \tilde m \in \mathbb R^{B_r\times B_c}$，
> 取指数: $\tilde P = \exp (S-\tilde m) \in \mathbb R^{B_r \times B_c}$ ，($\exp (S-\tilde m) = \frac {\exp (S)}{\exp (\tilde m)}$，$\exp (S)$ 是 final 的)
> 按行求和: $\mathscr {\tilde l} = \text{rowsum}(\tilde P) \in \mathbb R^{B_r}$
>  
> 计算 $m^{new} = \max (m, \tilde m) \in \mathbb R^{B_r}$，即更新记录的每行最大值；
> 计算 $e^{m - m^{new}}\mathscr l\in \mathbb R^{B_r}$ ，即用更新的最大值重放缩目前为止累加的各行指数和，
> 计算 $e^{\tilde m - m^{new}}\mathscr {\tilde l}\in \mathbb R^{B_r}$ ，即用更新的最大值重放缩当前 $S$ 块的各行指数和，
> 计算 $\mathscr l^{new} = e^{m-m^{new}}\mathscr l + e^{\tilde m - m^{new}}\mathscr {\tilde l} \in \mathbb R^{B_r}$，即累加/更新目前为止的各行指数和；
> 
> 计算 $\text{diag}(\mathscr l) e^{m - m^{new}}O$，可以视为：对于每一行，先乘上目前为止的各行指数和，恢复目前为止注意到的样本的指数分数，然后用更新的最大值重放缩目前为止注意到的样本的指数分数，注意对于每一行，目前为止注意到的样本数量随着外层循环增长；
> 计算 $e^{\tilde m_{ij} - m_i^{new}}\tilde P_{ij}V_j$，可以视为：对于每一行，用更新的最大值重放缩当前块注意到的样本的指数分数，然后按照指数分数对注意到的样本加权求和；
> 计算 $\text{diag}(\mathscr l) e^{m - m^{new}}O + e^{\tilde m_{ij} - m_i^{new}}\tilde P_{ij}V_j$，可以视为：对于每一行，补充注意到的（当前块）样本的加权和；
> 计算 $\text{diag}(\mathscr l_i^{new})^{-1}(\text{diag}(\mathscr l) e^{m - m^{new}}O + e^{\tilde m_{ij} - m_i^{new}}\tilde P_{ij}V_j)$，可以视为：对于每一行，规范化注意力权重（即除以各行的放缩指数分数和）；
>
> 将 $\mathscr l^{new}, m^{new}$ 写回 HBM，即更新 $\mathscr l, m$

 14:     **end for**
 15: **end for**
 16: Return $O$.

> 前向传播的图示见[[#Figure Illustration for FlashAttention Forward Algorithm|附录]]

## 3.2 Analysis: IO Complexity of FlashAttention
We analyze the IO complexity of FlashAttention , showing significant reduction in HBM accesses compared to standard attention. We also provide a lower bound, proving that no exact attention algorithm can asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in Appendix C. 
>我们分析了FlashAttention的I/O复杂性，与标准注意力相比，其HBM访问显著减少，我们还提供了一个下界，证明没有任何精确的注意力算法可以在所有SRAM大小上渐近地改善HBM访问，证明在 Appendix C中

**Theorem 2** Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d\le M \le Nd$. Standard attention (Algorithm 0) requires $\Theta(Nd + N^2)$ HBM accesses, while FlashAttention (Algorithm 1) requires $\Theta(N^2d^2M^{-1})$ HBM accesses.
> 定理2
> 序列长度 $N$，头维度 $d$，SRAM 大小 $M$，满足 $d\le M \le Nd$
> 标准的 attention 算法需要 $\Theta (Nd + N^2)$ HBM 访问
> FlashAttention 算法需要 $\Theta (N^2d^2M^{-1})$ HBM 访问

For typical values of $d$ (64-128) and $M$ (round 100KB), $d^{2}$ is many times smaller than $M$ , and thus FlashAttention requires many times fewer HBM accesses than standard implementation. This leads to both faster execution and lower memory footprint, which we validate in Section 4.3. 
>对于典型的𝑑值(64-128)和𝑀值(大约100KB)，$d^2$ 比𝑀小很多倍，因此FlashAttention需要的HBM访问次数比标准实现少很多倍，这带来了更快的执行和更低的内存占用，我们将在第4.3节中验证这一点

The main idea of the proof is that given the SRAM size of $M$ , we can load blocks of $\mathbf{K},\mathbf{V}$ of size $\Theta(M)$ each (Algorithm 1 line 6). For each block of $\mathbf{K}$ and $\mathbf{V}$ , we iterate over all blocks of $\mathbf{Q}$ (Algorithm 1 line 8) to compute the intermediate values, resulting in $\Theta(N d M^{-1})$ passes over $\mathbf{Q}$ . Each pass loads Θ ( 𝑁𝑑 ) elements, which amounts to $\Theta(N^{2}d^{2}M^{-1})$ HBM accesses.
>证明的主要思想是，给定SRAM大小𝑀，我们可以加载大小各为 $\Theta(M)$ 的 $K,V$ 块(Algorithm 1 line6)，对于每个 $K$ 和 $V$ 块，我们遍历所有 $Q$ 块(Algorithm 1 line 8)以计算中间值，这将会总共对 $Q$ 进行 $\Theta(NdM^{-1}$) 次遍历，每次遍历加载 $\Theta(Nd)$ 个元素，这相当于 $\Theta(N^2d^2M^{-1})$ 次HBM访问
>(标准注意力算法：没有考虑 SRAM，直接读写 HBM，$K,Q,V$ 的读取次数为 $\Theta (Nd)$，$S$ 的读取次数为 $\Theta (N^2)$，故总读取次数为 $\Theta(Nd + N^2)$)

We similarly prove that the backward pass of standard attention requires $\Theta(N d+N^{2})$ HBM accesses while the backward pass of FlashAttention requires $\Theta(N^{2}d^{2}M^{-1})$ HBM accesses (Appendix B). 
>类似地，我们可以证明标准注意力的反向传播需要 $\Theta(Nd + N^2)$ 次HBM访问，而FlashAttention的反向传播需要 $\Theta(N^2d^2M^{-1})$ 次HBM访问(Appendix B)

We prove a lower-bound: one cannot asymptotically improve on the number of HBM accesses for all values of $M$ (the SRAM size) when computing exact attention. 
>我们证明了一个下界：在计算精确注意力时，对于所有 $M$ 值(SRAM大小)，不能渐近地改善HBM访问次数

**Proposition 3.** 
Let $N$ be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with $d\leq M\leq N d$ . There does not exist an algorithm to compute exact attention with $o(N^{2}d^{2}M^{-1})$ HBM accesses for all 𝑀 in the range $[d,N d]$ . 
> 命题3：
> 序列长度 $N$，头维度 $d$，SRAM 大小 $M$，满足 $d\le M \le Nd$，对于在 $[d, Nd]$ 范围内的 $M$ ，不存在可以以 $o (N^2 d^2 M^{-1})$ HBM 访问计算精确注意力的算法

The proof relies on the fact that for $M=\Theta(N d)$ , any algorithm must perform $\Omega(N^{2}d^{2}M^{-1})\,=\Omega(N d)$ HBM accesses. This type of lower bound over a subrange of $M$ is common in the streaming algorithms literature [88]. We leave proving parameterized complexity [27] lower bounds in terms of $M$ as exciting future work. 
> 证明基于一个事实：对于 $M = \Theta (Nd)$，任意算法必须执行 $\Omega (N^2 d^2 M^{-1}) = \Omega (Nd)$ 次 HBM 访问 
>参数化的复杂性下界分析留待之后的工作

![[FlashAttention-Fig2.png]]

We validate that the number of HBM accesses is the main determining factor of attention run-time. In Fig. 2 (left), we see that even though FlashAttention has higher FLOP count compared to standard attention (due to recomputation in the backward pass), it has much fewer HBM accesses, resulting in much faster runtime. In Fig. 2 (middle), we vary the block size $B_{c}$ of FlashAttention , which results in diﬀerent amounts of HBM accesses, and measure the runtime of the forward pass. As block size increases, the number of HBM accesses decreases (as we make fewer passes over the input), and runtime decreases. For large enough block size (beyond 256), the runtime is then bottlenecked by other factors (e.g., arithmetic operations). Moreover, larger block size will not fit into the small SRAM size. 
> 我们将验证 HBM 的访问次数将是 attention 运行时间的主要决定因素
> Fig2 left中，可以看到，FlashAttention 对比于标准 attention 计算有更多的 FLOP 数量 (反向传播中的重计算)，但因为其少得多的 HBM 访问次数，其运行时间大大减少
> Fig2 middle 展示了 HBM 访问次数和 $K, V$ 块大小 $B_c$ 的关系，可以看到块越大，HBM 访问次数越少，前向传播时间越短，块足够大时 (超过256)，运行时间的瓶颈由其他因素制约 (如算数运算)，不再随着块大小增大而减少，当然，块过大 SRAM 也放不下

## 3.3 Extension: Block-Sparse FlashAttention 
We extend FlashAttention to approximate attention: we propose block-sparse FlashAttention , whose IO complexity is smaller than FlashAttention by a factor proportional to the sparsity. 
> 我们将 FlashAttention 拓展为近似 attention 计算，即 block-sparse FlashAttention，其 IO 复杂度比 FlashAttention 小一个正比于稀疏度的因子

Given inputs $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ and a mask matrix $\tilde{\mathbf{M}}\in\{0,1\}^{N\times N}$ , we want to compute: 

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S}\odot\mathbb{1}_{\tilde{\mathbf{M}}})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$ 
where $(\mathbf{S}\odot\mathbb{1}_{\tilde{\mathbf{M}}})_{k l}=\mathbf{S}_{k l}$ if $\tilde{\mathbf{M}}_{k l}=1$ and $-\infty$ if $\mathbf{M}_{k l}=0$ . 
> 稀疏 Attention 计算比标准 Attention 计算多了掩码矩阵 $\tilde {\mathbf M}\in \{0, 1\}^{N\times N}$，掩码矩阵和分数矩阵 $\mathbf S$ 逐元素运算，掩码为0就将分数设为负无穷，否则不变

We require $\tilde{\textbf{M}}$ to have block form: for some block sizes $B_{r},B_{c}$ , for all $k,l$ , $\tilde{\mathbf{M}}_{kl}=\mathbf{M}_{i j}$ with $i=\lfloor k/B_{r}\rfloor,j=\lfloor l/B_{c}\rfloor$ for some $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$ . 
> 我们要求 $\tilde {\mathbf M}$ 具有块形式，块大小为 $B_r \times B_c$
> $\tilde {\mathbf M}$ 压缩之后的矩阵为 $\mathbf M \in \{0, 1\}^{N/ B_r\times N/ B_c}$
> $\tilde {\mathbf M}$ 的一个 $B_r\times B_c$ 块内的元素都是相同的，映射到压缩矩阵 $\mathbf M$ 中的一个元素
> 具体地说，就是满足对于所有的 $k, l$，$\tilde {\mathbf M}_{kl} = \mathbf M_{ij}$，其中 $i = \lfloor k / B_r \rfloor$ ( $k$ 在第几个块行)，$j = \lfloor l / B_c \rfloor$ ( $j$ 在第几个块列)

Given a predefined block sparsity mask $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$ we can easily adapt Algorithm 1 to only compute the nonzero blocks of the attention matrix. The algorithm is identical to Algorithm 1, except we skip zero blocks. We reproduce the algorithm description in Algorithm 5 in Appendix B. 
> 称 $\tilde {\mathbf M}$ 压缩得到的 $\mathbf M$ 为块稀疏掩码，给定块稀疏掩码，我们可以简单调整算法1，使其仅计算非零块的 attention 矩阵，实际算法和算法1一致，差别仅在跳过零块

We also analyze the IO complexity of block-sparse FlashAttention . 

**Proposition 4.** 
Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with $d\,\leq\,M\,\leq\,N d$ . Block-sparse FlashAttention (Algorithm 5) requires $\Theta(N d+N^{2}d^{2}M^{-1}s)$ HBM accesses where 𝑠 is the fraction of nonzero blocks in the block-sparsity mask. 
> 命题4：
> 序列长度 $N$，$d$ 为头维度，$M$ 为 SRAM 大小，满足 $d\le M \le Nd$
> 块稀疏 FlashAttention (算法5) 需要 $\Theta (Nd + N^2d^2 M s)$ 次 HBM 访问，其中 $s$ 为块稀疏掩码中的非零块比例

We see that applying block-sparsity yields a direct improvement by the sparsity to the larger term in the IO complexity. For large sequence lengths $N$ , is often set to $N^{-1/2}$ [11] or $N^{-1}\log N$ [3 ,17 ,92], resulting in $\Theta(N{\sqrt{N}})$  or $\Theta(N\log N)$ IO complexity. For downstream experiments, we use the fixed butterfly sparsity pattern [17], which has been shown to be able to approximate arbitrary sparsity [16]. 
> 可以看到 IO 复杂度随着块稀疏程度而下降，对于大的 $N$，稀疏度一般设为 $N^{-1/2}$ 或 $N^{-1}\log N$，对应的 IO 复杂度降为 $\Theta (N\sqrt N)$ 或 $\Theta (N\log N)$

In Fig. 2 (right), we validate that as the sparsity increases, the runtime of block-sparse FlashAttention improves proportionally. On the LRA benchmark, block-sparse FlashAttention achieves $2.8\times$ speedup, while performing on par with standard attention (Section 4). 
> Fig2 (right) 中可以看到，块稀疏 FlashAttention 的运行时间随着稀疏度提高而成比例下降

# 4 Experiments 
We evaluate the impact of using FlashAttention to train Transformer models. We validate two claims about training time and model accuracy, and report attention runtime and memory benchmarks.
> 我们验证 FlashAttention 的训练时间和模型准确度

- **Training Speed.** FlashAttention outperforms the MLPerf 1.1 [58] speed record for BERT by $15\%$ , and speeds up GPT-2 up to 3× over HuggingFace [87] and $1.8\times$ over Megatron over standard Transformers. FlashAttention speeds up the long-range arena (LRA) benchmark 2.4× . 
> 训练速度：
> FlashAttention 训 BERT 上比 MLPerf 1.1 快15%，训 GPT-2 比 HuggingFace 快3倍，训标准 Transformer 比 Megatron 快 1.8 倍
> FlashAttention 比 LRA benchmark 快2.4倍

- **Quality.** FlashAttention scales Transformers to longer sequences, yielding higher quality. FlashAttention trains GPT-2 with context length 4K faster than Megatron trains GPT-2 with context length 1K, while achieving 0.7 better perplexity. Modeling longer sequences yields 6.4 points of lift on two long-document classification tasks. Finally, FlashAttention yields the first Transformer that can achieve better-than-random performance on the challenging Path-X task (sequence length 16K), and block-sparse FlashAttention yields the first sequence model that we know of that can achieve better-than-random performance on Path-256 (sequence length 64K).
> 质量：
> FlashAttention 将 Transformer 扩展到更长序列，故质量更高
> FlashAttention 训 GPT-2 的窗口为 4K 长度，比 Megatron 训 1K 长度的速度还快，困惑度也更高，长文档分类任务准确率也更高
> FlashAttention 训练出第一个在 Path-X 任务表现比随机好的 Transformer (序列长度16K)，块稀疏 FlashAttention 训练出第一个在 Path-256 任务表现比随机好的 Transformer (序列长度64K)

- **Benchmarking Attention.** We measure the runtime and memory performance of FlashAttention and block-sparse FlashAttention based on sequence length. We confirm that the memory footprint of FlashAttention scales linearly with seq. length and is up to 3x faster than standard attention for common seq. lengths (up to 2K). We confirm that runtime of block-sparse FlashAttention scales linearly in seq. length and is faster than all existing approximate attention baselines. 
> FlashAttention 的内存占用和序列长度成线性关系，在常规序列长度上三倍快于标准 attention
> 块稀疏 FlashAttention 的运行时间和序列长度成线性关系，快于所有的现存近似 attention 算法

Additional experiment details are in Appendix E. 

## 4.1 Faster Models with FlashAttention 
**BERT.** FlashAttention yields the fastest single-node BERT training speed that we know of. We train a BERT-large [22] model with FlashAttention on Wikipedia. Table 1 compares our training time to the implementation from Nvidia that set the training speed record for MLPerf 1.1 [58]. Our implementation is $15\%$ faster. 
> BERT 训练时间比 MLPerf 快15%

![[FlashAttention-Table1.png]]


**GPT-2.** FlashAttention yields faster training times for GPT-2 [67] on the large OpenWebtext dataset [32] than the widely used HuggingFace [87] and Megatron-LM [77] implementations. Table 2 shows up to 3× end-to-end speedup compared to Huggingface and 1.7× speedup compared to Megatron-LM. FlashAttention achieves the same perplexity as the other two implementations, as we do not change the model definition. Appendix E includes plots of the validation perplexity throughout training, confirming that FlashAttention is as numerically stable as the baselines and produces the same training/validation curves. 
> GPT-2在 OpenWebtext 数据集上训练时间比 HuggingFace 和 Megatron-LM 快，且困惑度一样
> Appendix E 提供了训练时的验证 perplexity 曲线，FlashAttention 的数值稳定性和 baseline 一致，训练和验证曲线也一致

![[FlashAttention-Table2.png]]

**Long-range Arena.** We compare vanilla Transformer (with either standard implementation or FlashAt- tention ) on the long-range arena (LRA [ 80 ]) benchmark. We measure accuracy, throughput, and training time of all models. Each task has a diﬀerent sequence length varying between 1024 and 4096. We follow the implementation and experimental setting in Tay et al. [80] and Xiong et al. [90] . Table 3 shows that FlashAt- tention achieves up $2.4\times$ speed-up compared to standard attention. Block-sparse FlashAttention is faster than all of the approximate attention methods that we have tested. 
> LRA benchmark 上，FlashAttention 比标准 Attention 快2.4倍，块稀疏 FlashAttention 比所有近似 attention 方法都快

![[FlashAttention-Table3.png]]

## 4.2 Better Models with Longer Sequences 
**Language Modeling with Long Context.** The runtime and memory-efficiency of FlashAttention allow us to increase the context length of GPT-2 by 4x while still running faster than the optimized implementation from Megatron-LM. Table 4 shows that that GPT-2 with FlashAttention and context length 4K is still 30% faster than GPT-2 from Megatron with context length 1K, while achieving 0.7 better perplexity. 
> 训练 GPT-2 时，FlashAttention 将上下文长度扩展到原来4倍，仍然比 Megatron-LM 快30%，且 perplexity 更高

![[FlashAttention-Table4.png]]

**Long Document Classification.** Training Transformers with longer sequences with FlashAttention improves performance on the MIMIC-III [47] and ECtHR [6 , 7] datasets. MIMIC-III contains intensive care unit patient discharge summaries, each annotated with multiple labels. ECtHR contains legal cases from the  European Court of Human Rights, each of which is mapped to articles of the Convention of Human Rights that were allegedly violaged. Both of these datasets contain very long text documents; the average number of tokens in MIMIC is 2,395 tokens, and the longest document contains 14,562 tokens, while the average and longest numbers in ECtHR are 2,197 and 49,392, respectively. We evaluate lift from increasing the sequence length of a pretrained RoBERTa model [56] (we repeat the positional embeddings, as in Beltagy et al. [3]). 
> 使用更长的序列训练 Transformer 提高了 MIMIC-III 和 EctHR 上的分类表现，这两个数据集都包含非常长的文本，平均 token 数量分别是 2,395 和 2,197，最长 token 数量分别是 14,562 和 49,392

Table 5 shows that sequence length 16K outperforms length 512 by 4.3 points on MIMIC, and that length 8K outperforms length 512 by 8.5 points on ECtHR. The discrepancies may be due to subtle distribution shifts: MIMIC-III contains specialized medical text and thus may be more susceptible to a distribution shift in the document length, whereas ECtHR contains general language. 

![[FlashAttention-Table5.png]]

**Path-X and Path-256.** The Path-X and Path-256 benchmarks are challenging tasks from the long-range arena benchmark designed to test long context. The task is to classify whether two points in a black and white 128 $\times$ 128 (or 256 $\times$ 256) image have a path connecting them, and the images are fed to the transformer one pixel at a time. In prior work, all transformer models have either run out of memory, or only achieved random performance [80]. There has been a search for alternative architectures that can model such long context [37]. We present here the first result of Transformer models being able to solve Path-X and Path-256 (Table 6). We pretrain a transformer on Path-64, and then transfer to Path-X by spatially interpolating the positional embeddings. FlashAttention achieves 61.4 accuracy on Path-X. Additionally, block-sparse FlashAttention enables the Transformers to scale to sequence length 64K, achieving 63.1 accuracy on Path-256. 
> 预训练于 Path-64，然后通过空间插值位置嵌入迁移到 Path-X

## 4.3 Benchmarking Attention 
We vary sequence length and measure runtime and memory usage of FlashAttention and block-sparse FlashAttention against various attention baselines on one A100 GPU with 40 GB HBM, with dropout and a padding mask. We compare against reference implementations for exact attention, approximate attention, and sparse attention. We report a subset of baselines in the main body; Appendix E contains more baselines and full details. 

![[FlashAttention-Fig3.png]]

**Runtime.** Figure 3 (left) reports the runtime in milliseconds of the forward + backward pass of FlashAttention and block-sparse FlashAttention compared to the baselines in exact, approximate, and sparse attention (exact numbers in Appendix E). Runtime grows quadratically with sequence length, but FlashAttention runs significantly faster than exact attention baselines, up to $3\times$ faster than the PyTorch implementation. The runtimes of many approximate/sparse attention mechanisms grow linearly with sequence length, but FlashAttention still runs faster than approximate and sparse attention for short sequences due to fewer memory accesses. The approximate attention runtimes begin to cross over with FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations of exact, sparse, and approximate attention that we know of, across all sequence lengths. 
> 各个 attention 算法的运行时间和序列长度的关系图见 Fig3 (left)，可以看到，运行时间随着序列长度增大而二次增加
> 在所有的准确 attention 算法中，FlashAttention 最快，并且在序列长度短时比一些近似 attention 算法还快
> block-sparse FlashAttention 比所有的 attention 算法都快

**Memory Footprint.** Figure 3 (right) shows the memory footprint of FlashAttention and block-sparse FlashAttention compared to various exact, approximate, and sparse attention baselines. FlashAttention and block-sparse FlashAttention have the same memory footprint, which grows linearly with sequence length. FlashAttention is up to $20\times$ more memory efficient than exact attention baselines, and is more memory-efficient than the approximate attention baselines. All other algorithms except for Linformer run out of memory on an A100 GPU before 64K, and FlashAttention is still  $2\times$ more efficient than Linformer. 
> 各个 attention 算法的内存使用和序列长度的关系见 Fig3 (right)，可以看到，FlashAttention 和 block-sparse FlashAttention 有相同的内存占用，占用大小随着序列长度线性增加，其内存效率比近似 attention 算法和准确 attention 算法都高
> 除了 Linformer，所有其他算法都在序列长度超过 64K 后内存溢出

# 5 Limitations and Future Directions 
We discuss limitations of our approach and future directions. Related work is given in Appendix A. 

**Compiling to CUDA.** Our current approach to building IO-aware implementations of attention requires writing a new CUDA kernel for each new attention implementation. This requires writing the attention algorithm in a considerably lower-level language than PyTorch, and requires significant engineering eﬀort. Implementations may also not be transferrable across GPU architectures. These limitations suggest the need for a method that supports writing attention algorithms in a high-level language (e.g., PyTorch), and compiling to IO-aware implementations in CUDA—similar to eﬀorts such as Halide in image processing [70]. 

**IO-Aware Deep Learning.** We believe that the IO-aware approach can extend beyond attention. Attention is the most memory-intensive computation in Transformers, but every layer in a deep network touches GPU HBM. We hope our work inspires IO-aware implementations of additional modules. We discuss these potential extensions in Appendix D. 
> attention 是 Transformer 中最为内存密集的计算

**Multi-GPU IO-Aware Methods.** Our IO-aware implementation of attention is optimal within constants for computing attention on a single GPU. However, the attention computation may be parallelizable across multiple GPUs [72]. Using multiple GPUs adds an additional layer to IO analysis—accounting for data transfer between GPUs. We hope our work inspires future work in this direction. 
> 使用多 GPU 实现还需要额外考虑 GPU 之间的数据传输，故添加了额外的一层 IO 分析

# A Related Work 
**IO-Aware Runtime Optimization.** The broad concept of optimizing for reading and writing to fast/slow memory has a long history in computer science and has been known by many names. We draw the most direct connection to the literature of analyzing I/O complexity in this work [1], but concepts of memory hierarchies are fundamental and has appeared in many forms, from the working set model [21], to data locality [86], to the Rooﬂine model of arithmetic intensity [85], to analyses of scalability [59], to standard textbook treatments of computer architecture [40]. We hope that this work encourages the community to adopt these ideas in more parts of the deep learning stack. 

**Efficient ML Models with Structured Matrices.** Matrix multiply is the core computational bottleneck of most machine learning models. To reduce the computational complexity, there have been numerous approaches to learn over a more efficient set of matrices. These matrices are called structured matrices , which have subquadratic ( $o(n^{2})$ for dimension $n\times n$ ) number of parameters and runtime. Most common examples of structured matrices are sparse and low-rank matrices, along with fast transforms commonly encountered in signal processing (Fourier, Chebyshev, sine/cosine, orthogonal polynomials). There have been several more general classes of structured matrices proposed in machine learning: Toeplitz-like [78], low-displacement rank [49], quasi-separable [25]). The butterﬂy pattern we use for our block-sparse attention is motivated by the fact that butterﬂy matrices [15 , 64] and their products have been shown to be able to express any structured matrices with almost optimal runtime and number of parameters [16 , 20]. However, even though structured matrices are efficient in theory, they have not seen wide adoption since it is hard to translate their efficiency to wall-clock speedup since dense unconstrained matrix multiply has very optimize implementation, a phenomenon known as the hardware lottery [41]. Extensions of butterﬂy matrices [17 , 18] aimed to make butterﬂy matrices more hardware-friendly. 
> 矩阵乘是大多数机器学习模型的核心计算瓶颈，为了减少计算复杂性，许多方法研究了更高效的一系列矩阵，这些矩阵被称为结构化矩阵，它们具有次二次的参数和运行时间
> 最常见的结构化矩阵就是低秩和稀疏矩阵，以及在信号处理中常见的快速变换矩阵
> 我们在块稀疏 attention 中使用的蝴蝶模式来源于蝴蝶矩阵，其乘积可以以几乎最优的运行时间和参数数量表示任意结构化的矩阵
> 结构化矩阵在理论中是高效的，但并未广泛在实际中使用，因为不容易将它们的理论效率转化为实际的速度提升，稠密的矩阵乘实际上已经有高度优化的实现

**Sparse Training.** Our block-sparse FlashAttention can be seen as a step towards making sparse model training more efficient. Sparse models have seen success in compressing models for inference (pruning) by sparsifying the weight matrices [23 , 38 , 39 , 55 , 76]. For model training, the lottery tickets hypothesis [28 , 29 , 30] suggests that there are a set of small sub-networks derived from a larger dense network that performs as well as the original dense network. Out block-sparse FlashAttention can also be seen as a fixed lottery ticket in the context of attention: we fix the sparsity pattern to be the butterﬂy pattern through training, and observe that it performs almost as well as the (dense) FlashAttention on the Long-range Arena tasks. 
> 块稀疏 FlashAttention 可以视作让稀疏模型训练更高效的方法
> 稀疏方法通过稀疏化权重矩阵来压缩模型，提高推理效率
> 对于模型训练，lottery tickets 假设表明：从更大的密度网络中衍生的一组小的自网络可以和原网络的表现相近
> 块稀疏 FlashAttention 可以视为固定的 lottery ticket：将稀疏模式在训练时固定为蝴蝶模式，发现模型的表现和密度 FlashAttention 在 LRA 任务上相近

**Efficient Transformer.** Transformer-based models have become the most widely-used architecture in natural language processing [22] and computer vision [24 , 91]. However, one of their computational bottlenecks is that their time and memory scales quadratic in the sequence length. There are numerous approaches to overcome this bottleneck, including approximation with hashing (i.e., sparse) such as Reformer [51] and Smyrf [19] and with low-rank approximation such as Performer [12 , 54]. One can even combine sparse and low-rank approximation for better accuracy (e.g., Longformer [3 ], BigBird [92], Scatterbrain [9], Long-short transformer [94], Combiner [73 ]). Other approaches include compressing along the sequence dimension to attend to multiple tokens at once [52 , 57 , 79 , 89]. One can also attend over the states from previous sequences to help lengthen the context (e.g., Transformer-XL [14] and Compressive Transformer [69]). We recommend the survey [81] for more details. 
> Transformer 的时间和内存都随着序列长度而二次增长
> 对应的解决方法有：使用哈希近似（即稀疏）的 Reformer, Smyrf ；使用低秩近似的 Performer；Longformer, BIgBird 等将稀疏和低秩近似结合
> 还有的方法压缩序列维度，使得一次 attend 多个 token
> 也可以 attend 之前序列的状态来帮助增长上下文，例如 Transformer-XL

There are several lines of work on developing other modules instead of attention to model longer context. HiPPO [35] and its extensions, most notably S4 [31 , 36 , 37] projects the history on a polynomial basis, allowing accurate reconstruction of the history through state-space models. They combine the strengths of CNNs (efficient training), RNNs (efficient inference), and continuous models (robust to change in sampling rates). LambdaNetworks [2], AFT [93] and FLASH [42] are other attempts at replacing attention in the context of image classification and language modeling. 

# B Algorithm Details 
We first derive the forward and backward passes of attention and show that they can be computed in a memory-efficient manner (requiring extra memory linear instead of quadratic in the sequence length). Though they reduce the amount of extra memory required, naively they still incur quadratic HBM accesses, resulting in slower execution speed. We describe the FlashAttention algorithm to implement both the forward and the backward passes on GPUs that reduces HBM accesses, leading to both faster runtime and smaller memory footprint. 
> 我们首先推导 attention 计算的前向和反向过程，然后表明它们可以以内存高效的形式计算（对额外内存的需求线性于序列长度，而不是二次），虽然减少了对额外内存的需求，但朴素算法仍然需要二次的 HBM 访问
> 我们接着介绍 FlashAttention 算法在 GPUs 上的正向和反向传播的实现，以减少 HBM 访问，这样我们就同时具有了更少的内存需求和更少的运行时间

## B.1 Memory-efficient forward pass 
The main challenge in making attention memory-efficient is the softmax that couples the columns of $\mathbf{K}$ (and columns of $\mathbf{V}$ ). Our approach is to compute the softmax normalization constant separately to decouple the columns. This technique [60] has been used in the literature [51 , 66] to show that attention computation does not need quadratic extra memory (though the number of HBM accesses is still quadratic, resulting in slow run-time). 
> 要让 attention 内存高效，主要的挑战就是 softmax 计算，它绑定了 $\mathbf K, \mathbf V$ 的列
> 我们的方法为分别计算 softmax 规范化常数，以解耦这些列，该技术被用于 [51, 66]，展示了 attention 并不需要二次的额外内存（当然 HBM 访问的次数仍然是二次的，故计算时间尚未优化）

For simplicity, we omit here the max-shifting step during softmax. The full algorithm in Appendix B.3 contains all the steps. 

Recall that given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ , we want to compute the attention output $\mathbf{O}\in\mathbb{R}^{N\times d}$ : 

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d}.
$$ 
We have that $S_{i j}=q_{i}^{T}{k}_{j}$ where $q_i$ and $k_{j}$ are the $i$ -th and $j$ -th columns of $\mathbf{Q}$ and $\mathbf{K}$ respectively. 
> $q_i$ 是 $\mathbf Q$ 的第 $i$ 行向量（写为列向量），$k_j$ 是 $\mathbf K$ 的第 $j$ 行向量（写为列向量）

Define the normalization constants of softmax: 
> softmax 每一行的规范化因子定义为 $L_i = \sum_j e^{S_{ij}} = \sum_j  e^{q_i^Tk_j}$

$$
L_{i}=\sum_{j}e^{q_{i}^{T}k_{j}}.\tag{1}
$$ 
Let $v_j$ be the $j$ -th columns of $\mathbf{V}$ , then the $i$ -th columns of the output is
> 输出 $\mathbf O$ 的第 $i$ 行 $o_i = P_{i:}\mathbf V$，令 $v_j$ 为 $\mathbf V$ 的第 $j$ 行向量
> 故 $o_i = P_{i:}\mathbf V = \sum_{j} P_{ij}v_j$
> 将 $P_{ij}$ 显式写为 $\frac {e^{S_{ij}}}{L_i} = \frac {e^{q_i^Tk_j}}{L_i}$ 就得到了 (2)

$$
o_{i}=P_{i:}{\bf V}=\sum_{j}P_{i j}v_{j}=\sum_{j}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}v_{j}.\tag{2}
$$ 
We see that once $L_{i}$ is computed, we can compute $o_{i}$ without extra memory by repeatedly summing $\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}v_{j}$ . Therefore the forward pass can be computed with $O(n)$ extra memory: 

1. Compute $L_{i}$ for all $i$ according to Eq. (1), which takes $O(n)$ extra memory.
2. Compute $o_{i}$ for all $i$ according to Eq. (2), which takes $O(d)$ extra memory. 

> 当每一行的规范化常数 $L_i$ 得到后，输出 $\mathbf O$ 的第 $i$ 行 $o_i$ 的计算就是反复求和 $\frac {e^{q_i^Tk_j}}{L_i}v_j$
> 为此，可以以 $O (n)$ 的额外内存需求计算前向传播：
> 1. 对于所有的 $i$，根据 (1) 计算 $L_i$，需要 $O (n)$ 额外内存
> 2. 对于所有的 $o_i$，根据 (2) 计算结果，需要 $O (d)$ 额外内存

> 在常规的 attention 计算中，中间结果分数矩阵 $\mathbf S\in \mathbb R^{N\times N}$ 是直接通过矩阵乘法计算得到的，存储这个中间结果需要 $O (n^2)$ 的额外内存
> 这里介绍的方法不直接计算出完整的分数矩阵 $\mathbf S$，而是用循环来一行一行地计算考虑算法的第一步：
> 1. for all $i$，compute $L_i$ 等价于 for all $i$，compute $\mathbf S_{i:}$，然后计算 $L_i = \sum_je ^{S_{ij}}$
> 算法的第一步计算每一行的规范化常数，因为每次计算仅关心 $\mathbf S$ 的一行，计算时需求的内存是 $O (d)$，最后存储全部的 $L_i$ 需求的内存是 $O (n)$，注意计算得到的 $\mathbf S$ 的结果没有存储下来，它需要在第二步再重新计算
> 考虑算法的第二步：
> 2. for all $o_i$, copmute $o_i$
> 这里同样是逐行地计算 $o_i$，此时还需要重新计算 $\mathbf S_{i:}$ ，取指数后，除以第一步计算得到的 $L_i$ 规范化得到权重，然后对 $\mathbf V$ 的所有行加权求和，计算权重需要的额外内存是 $O (d)$

## B.2 Memory-efficient backward pass 
We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe and Staats [66] suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly and show how it can be computed in a memory-efficient manner. 
> 我们推导 attention 计算的反向过程，表面它同样可以以线性内存的方式计算
> [66]中的内存高效的反向传播是通过为内存高效的前向传播应用梯度检查点
> 我们直接显式为反向传播进行推导

Suppose that there is a scalar loss function $\phi$ , and let the output gradient be $\mathbf{d}\mathbf{O}\in\mathbb{R}^{n\times d}$ (where $\mathbf{dO}$ denotes $\frac{\partial\phi}{\partial{\bf O}}$ ). We want to compute the input gradients $\mathbf{dQ},\mathbf{K},\mathbf{dV}\,\in\,\mathbb{R}^{n\times d}$ (where $\bf {dQ} , \bf {dK} , \bf {dV}$ denote $\frac{\partial\phi}{\partial\mathbf{Q}},\frac{\partial\phi}{\partial\mathbf{K}},\frac{\partial\phi}{\partial\mathbf{V}}$ respectively). 
> 记标量损失函数为 $\phi$，记输出梯度为 $\bf {dO} = \frac {\partial \phi}{\partial \bf O}$
> 我们需要计算梯度 $\bf {dQ} = \frac {\partial \phi}{\partial \bf Q},\bf {dK} = \frac {\partial \phi}{\partial \bf K},\bf {dV} = \frac {\partial \phi}{\partial \bf V}$

The gradient $\mathbf{dV}$ is easy to see. Applying reverse-mode autodiff by hand (aka the chain rule), we obtain (in matrix notation) $\bf {dV} = \bf P^T \bf {dO}$, and so:
> 关于 $\mathbf {dV} = \mathbf P^T \mathbf {dO}$ 的推导见[[#Deduction for chain rule of matrix multiplication|附录]]

$$
dv_{j}=\sum_{i}P_{i j}d o_{i}=\sum_{i}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}d o_{i}.\tag{3}
$$ 
> $dv_j = \sum_{i}P_{ij}do_i$ 来源于：

$$
\begin{align}
dV_{j:} &= (\mathbf P^T \mathbf {dO})_{j:}\\
&= [(P^T)_{j:} dO_{: 1}, \cdots, (P^T)_{j:} dO_{: d}]\\
&= (P^T)_{j:}[dO_{: 1}, \cdots, dO_{:d}]\\
&=(P^T)_{j:}\mathbf {dO}\\
&=(P_{:j})^T\mathbf {dO}\\
&=\sum_{i}P_{ij}dO_{i:}\\
\end{align}
$$

Since we already computed $L_{i}$ , $dv_{j}$ can be computed without extra memory by repeated summing. 
> 由 (3) 可知，$\mathbf {dV}$ 中的行 $dv_j$ 和 $P$ 的第 $j$ 列 $P_{:j}$ 和 $\mathbf {dO}$ 的所有行 $do_i$ 有关
> 计算时，我们逐行计算 $\mathbf {dV}$，也就是每次计算一个 $dv_j$，
> 这需要我们实时计算 $P$ 的第 $j$ 列 $P_{:j}$，其中 $P_{ij} = \frac {e^{q_i^T k_j}}{L_i}$，
> 然后根据 $\sum_i P_{ij}do_i$ 对 $\mathbf {dO}$ 的所有行进行加权求和得到 $dv_j$

The gradients $\mathbf {dQ}$ and $\mathbf{dK}$ are a little more complicated. We go through the gradients $\mathbf{dP}$ and $\mathbf {dS}$ first. From Eq. (2), we have that $\mathbf{d}\mathbf{P}=\mathbf{d}\mathbf{O}\mathbf{V}^{T}$ , and so: 
> 关于 $\mathbf {dP} = \mathbf {dOV}^T$ 的推导见[[#Deduction for chain rule of matrix multiplication|附录]]

$$
\begin{array}{r}{d P_{i j}=d o_{i}^{T}v_{j}.}\end{array}
$$ 
> $dP_{ij} = do_i^T v_j$ 来源于：

$$
\begin{align}
dP_{ij} &= dO_{i:}V^T_{:j}\\
&=dO_{i:}V_{j:}
\end{align}
$$

Recall that $P_{i:}=\mathrm{softmax}(S_{i:})$ . Using the fact that the Jacobian of $y=\mathrm{softmax}(x)$ is $\mathrm{diag}(y)-y y^{T}$ , we have that 
> 关于 $y = \text{softmax}(x)$ 的 Jacobian 的推导见[[#Deduction for Jacobian of softmax|附录]]

$$
\begin{array}{r}{d S_{i\colon}=(\mathrm{diag}(P_{i\colon})-P_{i\colon}P_{i\colon}^{T})d P_{i\colon}=P_{i\colon}\circ d P_{i\colon}-(P_{i\colon}^{T}d P_{i\colon})P_{i\colon},}\end{array}
$$ 
where ${\circ}$ denotes pointwise multiplication. 
> 关于 $dS_{i:}$ 的推导见[[#Deduction for $dS_{i }$|附录]]

Define 

$$
D_{i}=P_{i:}^{T}d P_{i:}=\sum_{j}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}d o_{i}^{T}v_{j}=d o_{i}^{T}\sum_{j}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}v_{j}=d o_{i}^{T}o_{i},\tag{4}
$$

then

$$
d S_{i:}=P_{i:}\circ d P_{i:}-D_{i}P_{i:}.
$$ 
Hence

$$
d S_{i j}=P_{i j}d P_{i j}-D_{i}P_{i j}=P_{i j}(d P_{i j}-D_{i}).
$$

Now we can get the gradients $\bf {dQ}$ and $\mathbf{dK}$ . Recall that $S_{i j}=q_{i}^{T}k_{j}$ , so 

$$
d q_{i}=\sum_{j}d S_{i j}k_{j}=\sum_{j}P_{i j}(d P_{i j}-D_{i})k_{j}=\sum_{j}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}(d o_{i}^{T}v_{j}-D_{i})k_{j}.\tag{5}
$$ 

> $\frac {\partial \phi}{\partial q_i} = \sum_k\sum_l \frac {\partial \phi}{S_{kl}}\frac {\partial S_{kl}}{\partial q_i} = \sum_l \frac {\partial \phi} {\partial S_{il}} \frac {\partial S_{il}}{\partial q_i} = \sum_{j} dS_{ij}\frac {\partial S_{ij}}{\partial q_i} = \sum_j dS_{ij}k_j$

Similarly, 

$$
d k_{j}=\sum_{i}d S_{i j}q_{i}=\sum_{i}P_{i j}(d P_{i j}-D_{i})q_{i}=\sum_{i}\frac{e^{q_{i}^{T}k_{j}}}{L_{i}}(d o_{i}^{T}v_{j}-D_{i})q_{i}.\tag{6}
$$ 

> $\frac {\partial \phi}{\partial k_j} = \sum_{k}\sum_{l}\frac {\partial \phi}{\partial S_{kl}}\frac {\partial S_{kl}}{\partial k_j} = \sum_{k}\frac {\partial \phi}{\partial S_{kj}}\frac {\partial S_{kj}}{\partial k_j} = \sum_{i}dS_{ij}\frac {\partial S_{ij}}{\partial k_j} = \sum_i dS_{ij}q_i$

Therefore the backward pass can also be computed with $O(n)$ extra memory: 

1. Compute $dv_{j}$ for all $j$ according to Eq. (3), which takes $O(d)$ extra memory. 
2. Compute $D_{i}$ for all $i$ according to Eq. (4), which takes $O(n)$ extra memory. 
3. Compute $d q_{i}$ for all $i$ according to Eq. (5), which takes $O(d)$ extra memory. 
4. Compute $d k_{j}$ for all $j$ according to Eq. (6), which takes $O(d)$ extra memory. 

> 使用 $O (n)$ 的额外内存的方向传播：
> 1. 计算 $\mathbf {dV}$：根据 eq (3) 计算 $dv_j$ (也就是逐行计算 $\mathbf {dV}$ )，其中需要的额外内存来自于 $do_j$ ，即 $O (d)$
> 2. 根据 eq (4) 计算全部的 $D_i$，一共 $n$ 个，故需要 $O (n)$ 的额外内存
> 3. 计算 $\mathbf {dQ}$ 和 $\mathbf {dK}$：根据 (5), (6) 计算 $dq_i, dk_j$ ( 同样是逐行计算 $\mathbf {dQ}, \mathbf {dK}$ )，其中需要的额外内存都来自于 $do_i$，即 $O (d)$

## B.3 FlashAttention : Forward Pass 
We describe the full details of FlashAttention forward pass. Given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ , we want to compute the attention output $\mathbf{O}\in\mathbb{R}^{N\times d}$ :

$$
\begin{align}
\mathbf S &= \tau \mathbf {QK}^T \in \mathbb R^{N\times N},\\
\mathbf S^{\text{masked}} &= \text{MASK}(\mathbf S)\in \mathbb R^{N\times N},\\
\mathbf P &= \text{softmax}(\mathbf S^{\text{masked}})\in \mathbb R^{N\times N},\\
\mathbf P^{\text{dropped}} &= \text{dropout}(\mathbf P, p_{\text{drop}}),\\
\mathbf O &= \mathbf P^{\text{dropped}}\mathbf V \in\mathbb R^{N\times d},
\end{align}
$$

where $\tau\in\mathbb{R}$ is some softmax scaling (typically $\textstyle{\frac{1}{\sqrt{d}}}$ ), mask is some masking function that sets some entries of the input to $-\infty$ and keep other entries the same (e.g., key padding mask when sequences in the batch don’t have the same lengths and are padded), and dropout $(x,p)$ applies dropout to $x$ elementwise (i.e., output $\scriptstyle{\frac{x}{1-p}}$ − with probability $1-p$ and output $0$ with probability $p$ for each lement $x$ ). 

The full algorithm is in Algorithm 2. We save the output $\mathbf O$ , the softmax statistics $\ell$ and $m$ , and the pseudo-random number generator state $\mathcal{R}$ for the backward pass. 
> 前向传播中，需要保存 $\mathbf O, \ell, m$ 以及随机数生成状态 $\mathcal R$ 用于反向传播

![[FlashAttention-Algorithm2.png]]

**Algorithm 2** FlashAttention Forward Pass
**Require**: Matrices $\mathbf {Q, K, V} \in \mathbb R^{N\times d}$ in HBM, on-chip SRAM of size $\bf M$, softmax scaling constant $\tau \in \mathbb R$, masking function $\text{MASK}$, dropout probability $p_{\text{drop}}$.
 1: Initialize the pseudo-random number generator state $\mathcal R$ and save to HBM.
 2: Set block sizes $B_c = \lceil \frac {\bf M}{4d} \rceil, B_r = (\lceil \frac {\bf M}{4d}\rceil, d)$.
 3: Initialize $\mathbf O = (0)_{N\times d} \in \mathbb R^{N\times d}$, $\ell = (0)_N \in \mathbb R^N$, $m = (-\infty)_N \in \mathbb R^N$ in HBM.
 4: Divide $\mathbf Q$ into $T_r = \lceil \frac {N}{B_r} \rceil$ blocks $\mathbf Q_1, \dots, \mathbf Q_{T_r}$ of size $B_r\times d$ each, and divide $\mathbf K, \mathbf V$ into $T_c = \lceil \frac {N}{B_c} \rceil$ blocks $\mathbf K_q, \dots, \mathbf K_{T_c}$ and $\mathbf V_1, \dots, \mathbf V_{T_c}$, of size $B_c \times d$ each.
 5: Divide $\mathbf O$ into $T_r$ blocks $\mathbf O_i, \dots, \mathbf O_{T_r}$ of size $B_r \times d$ each, divide $\ell$ into $T_r$ blocks $\ell_i, \dots, \ell_{T_r}$ of size $B_r$ each, divide $m$ into $T_r$ blocks $m_1, \dots, m_{T_r}$ of size $B_r$ each.
 6: **for** $1\le j \le T_c$ **do**
 7:    Load $\mathbf K_j, \mathbf V_j$ from HBM to on-chip SRAM.
 8:    **for** $1\le i \le T_r$ **do**
 9:    Load $\mathbf Q_i, \mathbf O_i , \ell_i, m_i$ from HBM to on-chip SRAM.
10:    On chip, computes $\mathbf S_{ij} = \tau \mathbf Q_i \mathbf K_{j}^T\in \mathbb R^{B_r \times B_c}$.
11:    On chip, computes $\mathbf S_{ij}^{\text{masked}}  = \text{MASK}(\mathbf S_{ij})$.
> 相较于 Algorithm 1，多出来一步 scale 和 mask

12:    On chip, computes $\tilde m_{ij} = \text{rowmax}(\mathbf S_{ij}^{\text{masked}}) \in \mathbb R^{B_r}$, $\tilde {\mathbf P}_{ij}=\exp (\mathbf S_{ij}^{\text{masked}} - \tilde m_{ij})\in \mathbb R^{B_r \times B_c}$ (pointwise), $\tilde \ell_{ij} = \text{rowsum}(\tilde {\mathbf P}_{ij} )\in \mathbb R^{B_r}$.
13:    On chip, compute $m_i^{\text{new}} = \max (m_i, \tilde m_{ij}) \in \mathbb R^{B_r}$, $\ell_i^{\text{new}} = e^{m_i - m_i^{new}} \ell_i  + e^{\tilde m_{ij} - m_i^{\text{new}}} \tilde \ell_{ij} \in \mathbb R^{B_r}$.

14:    On chip, compute $\tilde {\mathbf P}_{ij}^{\text{dropped}} = \text{dropout}(\tilde {\mathbf P}_{ij}, p_{\text{drop}})$.
> 相较于 Algorithm 1，多出来一步 dropout

15:    Write $\mathbf O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1}(\text{diag}(\ell_i)e^{m_i-m_i^{\text{new}}}\mathbf O_i + e^{\tilde m_{ij} - m_i^{\text{new}}}\tilde {\mathbf P}_{ij}^{\text{dropped}}\mathbf V_j)$ to HBM.
16:    Write $\ell_i \leftarrow \ell_i^{\text{new}}, m_i \leftarrow m_i^{\text{new}}$ to HBM.
17:    **end for**
18:  **end for**
19: Return $\mathbf O,\ell, m, \mathcal R$.

## B.4 FlashAttention : Backward Pass 
We des ull details of FlashAttetion backward pass. Given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ output $\mathbf{O}\in\mathbb{R}^{N\times d}$ , and the output gradient $\bf dO$ , we want to compute the input gradients $\mathbf {dQ}$ , $\mathbf {dK}$ , $\mathbf{d}\mathbf{V}\in\mathbb{R}^{N\times d}$ . 
> 本节讨论 FlashAttention 的反向传播算法，
> 算法的输入包括输入序列的 $\mathbf {Q, K, V}\in \mathbb R^{N\times d}$、输出序列 $\bf O \in \mathbb R^{N\times d}$、输出序列的梯度 $\bf dO$
> 算法需要计算输入序列的梯度 $\mathbf {dQ, dK, dV}\in \mathbb R^{N\times d}$

We first describe the standard attention backward pass in Algorithm 3 for completeness. 

![[FlashAttention-Algorithm3.png]]


**Algorithm 3** Standard Attention Backward Pass
**Requrie**: Matrices $\mathbf {Q, K, V, dO}\in \mathbb R^{N\times d}, \mathbf P \in \mathbb R^{N\times N}$ in HBM.
> 标准的 attention 反向传播在 HBM 中存储了完整的“权重矩阵” $\mathbf P \in \mathbb R^{N\times N}$

1: Load $\bf P, dO$ by blocks from HBM, compute $\mathbf {dV = P^T dO} \in \mathbb R^{N\times d}$, write $\bf dV$ to HBM.
> 根据 $\bf P$ 和 $\bf dO$ 计算 $\bf dV$

2: Load $\bf O, V$ by blocks from HBM, compute $\mathbf {dP = dO V^T} \in \mathbb R^{N\times N}$, write $\bf dP$ to HBM.
3: Read $\bf P, dP$ from HBM, compute $\mathbf {dS}\in \mathbb R^{N\times N}$ where $dS_{ij} = P_{ij}(dP_{ij} - \sum_l P_{il}dP_{il})$, write $\bf dS$ to HBM.
> 根据 $\mathbf {OV}$ 计算 $\bf dP$ 
> 根据 $\bf dP$ 计算 $\bf dS$

4: Load $\bf dS$ and $\bf K$ by blocks from HBM, compute $\bf dQ = \bf dSK$, write $\bf dQ$ to HBM.
5: Load $\bf dS$ and $\bf Q$ by blocks from HBM, compute $\bf dK = \bf dS^TQ$, write $\bf dK$ to HBM.
> 根据 $\bf dS$ 和 $\bf K$ 计算 $\bf dQ$
> 根据 $\bf dS$ 和 $\bf Q$ 计算 $\bf dK$

We now make two observations about FlashAttention backward pass: 

1. We do not need to store the dropout mask of size $O(N^{2})$ from the forward pass. Instead, we can save the pseudo-random number generator states from the forward pass and re-generate the dropout mask in the backward pass. This allows us to only use $O(N)$ extra memory. 

2. When computing the softmax gradient, we use Eq. (4) to compute $D_{i}=P_{i:}^{\top}d P_{i}$ : without reducing over $P_{i}$ and $d P_{i}$ of size $N$ (they might not fit into SRAM). Instead we can rewrite $D_{i}=d o_{i}^{\top}o_{i}$ and compute the dot product between vectors of size $d$ . 

> FlashAttention 的方向传播过程：
> 1. 不会显式存储大小为 $O (N^2)$ 的前向传播中计算得到的 dropout mask，而是保存生成 dropout mask 的随机数种子，用它在反向传播过程中重新生成dropout mask，这允许我们仅使用 $O (N)$ 的额外内存
> 2. 计算 softmax 梯度时，我们需要根据 eq (4) 计算 $D_i$，我们不使用 $D_i = P_{i:} ^\top dP_i$ 来计算(这是两个大小为 $N$ 的向量，可能放不进 SRAM)，而是使用 $D_i = do_i^\top o_i$ (这是两个大小为 $d$ 的向量)

The full FlashAttention backward pass algorithm is in Algorithm 4. Conceptually it is just a block version of the derivation in Appendix B.2. 
> 完整的 FlashAttention 反向传播算法如下
> 其本质上也就是根据 Appendix B.2 中的推导的 tiled block version

![[FlashAttention-Algorithm4.png]]

**Algorithm 4** FlashAttention Backward Pass
**Require**: Matrices $\mathbf {Q, K, V, O, dO} \in \mathbb R^{N\times d}$ in HBM, vectors $\ell , m \in \mathbb R^N$ in HBM, on-chip SRAM of size $M$, softmax scaling constant $\tau \in \mathbb R$, masking function $\text{MASK}$, dropout probability $p_{\text{drop}}$, pseudo-random number generator state $\mathcal R$ from the forward pass.
 1: Set the pseudo-random number generator state to $\mathcal R$.
 2: Set block sizes $B_c = \lceil \frac {M}{4d} \rceil, B_r = \min(\lceil \frac M {4d}\rceil, d)$.
 3: Divide $\bf Q$ into $T_r = \lceil \frac N {B_r} \rceil$ blocks $\bf Q_1, \dots, Q_{T_r}$ of size $B_r \times d$ each, and divide $\bf K, V$ into $T_c = \lceil \frac N {B_c} \rceil$ blocks $\bf K_1, \dots, K_{T_c}$ and $\bf V_1, \dots, V_{T_c}$ of size $B_c \times d$ each.
 4: Divide $\bf O$ into $T_r$ blocks $\bf O_1, \dots, O_{T_r}$ of size $B_r \times d$ each, divide $\bf dO$ into $T_r$ blocks $\bf dO_i, \dots, dO_{T_r}$ of size $B_r \times d$ each, divide $\ell$ into $T_r$ blocks $\ell_1, \dots, \ell_{T_r}$ of size $B_r$ each, divide $m$ into $T_r$ blocks $m_1, \dots, m_{T_r}$ of size $B_r$ each.
 5: Initialize $\mathbf {dQ} = (0)_{N\times d}$ in HBM and divide it into $T_r$ blocks $\bf dQ_1, \dots, dQ_{T_r}$ of size $B_r \times d$ each. Initialize $\mathbf {dK} = (0)_{N\times d}, \mathbf {dV} = (0)_{N\times d}$ in HBM and divide it into $T_c$ blocks of size $B_c\times d$ each.
> 设定随机数生成器、分块 $\mathbf {K, Q, V, O,dO}, \ell, m$、初始化 $\mathbf {dK, dQ,dV}$ 并分块

 6: **for** $1\le j \le T_c$ **do**
 7:   Load $\mathbf {K_j, V_j}$ from HBM to on-chip SRAM.
 8:   Initialize $\mathbf {\tilde {dK}}_j = (0)_{B_c\times d}, \mathbf {\tilde {dV}}_j = (0)_{B_c\times d}$ on SRAM.
> 外层循环迭代 $\mathbf {K, V, dK, dV}$ 块

 9:   **for*** $1 \le i \le T_r$ **do**
10:     Load $\mathbf {Q_i, O_i, dO_i, dQ_i}, \ell_i, m_i$ from HBM to on-chip SRAM.
> 内层循环迭代 $\mathbf {Q, O, dO, dQ}, \ell, m$ 块

11:     On chip, compute $\mathbf {S}_{ij} = \tau \mathbf Q_i \mathbf K_j^{\top} \in \mathbb R^{B_r \times B_c}$.
12:     On chip, compute $\mathbf {S}_{ij}^{\text{masked}} = \text{MASK}({\mathbf S_{ij}})$.
13:     On chip, compute $\mathbf {P}_{ij} = \text{diag}(l_i)^{-1}\exp (\mathbf S_{ij}^{\text{masked}} - m_i)\in\mathbb R^{B_r \times B_c}$.
>  在片上根据 $\mathbf {K_j, V_j, Q_i, O_i}$ 重新计算得到权重矩阵 $\mathbf P$ 的块 $\mathbf P_{ij}$

14:     On chip, compute dropout mask $\mathbf Z_{ij} \in \mathbb R^{B_r \times B_c}$ where each entry has value $\frac {1}{1 - p_{\text{drop}}}$ with probability $1 - p_{\text{drop}}$ and value $0$ with probability $p_{\text{drop}}$.
15:     On chip, compute $\mathbf P_{ij}^{\text{dropped}} = \mathbf P_{ij} \circ \mathbf Z_{ij}$ (pointwise multiply).
> 在片上根据随机数种子计算 dropout mask $\mathbf Z_{ij}\in \mathbb R^{B_r\times B_c}$ 并进行 dropout
 
16:     On chip, compute $\mathbf {\tilde {dV}}_j \leftarrow \mathbf {\tilde {dV}}_j + (\mathbf P_{ij}^{\text{dropped}})^\top \mathbf {dO}_i  \in\mathbb R^{B_c \times d}$.
> 更新 $\mathbf {dV}_j$：本质是 $\mathbf {dV} = \mathbf P^\top \mathbf {dO}$ 分块形式，图解见[[#$ mathbf {dV}$|附录]]

17:     On chip, compute $\mathbf {dP}_{ij}^{\text{dropped}} = \mathbf {dO}_{i}\mathbf V_j^{\top}\in \mathbb R^{B_r \times B_c}$.
18:     On chip, compute $\mathbf {dP}_{ij}= \mathbf {dP}_{ij}^{\text{dropped}} \circ \mathbf Z_{ij}$ (pointwise multiplly).
> 计算 $\mathbf {dP}_{ij}^{\text{dropped}}$ ：本质是 $\mathbf {dP} = \mathbf {dOV}^\top$ 的分块形式，图解见[[#$ mathbf {dP}$|附录]]
> 计算 $\mathbf {dP}_{ij}$： 因为 $\frac {\partial \phi} {\partial P_{ij}} = \frac {\partial \phi}{\partial P_{ij}^{\text{dropped}}}\frac {\partial P_{ij}^{\text{dropped}}}{\partial P_{ij}} = dP_{ij}^{\text{dropped}}\cdot \frac {\partial P_{ij}\cdot Z_{ij}}{\partial P_{ij}} = dP_{ij}^{\text{dropped}}\cdot Z_{ij}$，
> 故 $\mathbf {dP} = \mathbf {dP}^{\text{dropped}} \circ \mathbf {Z}$

19:     On chip, compute $D_i = \text{rowsum}(\mathbf {dO}_i \circ \mathbf O_i)\in \mathbb R^{B_r}$.
20:    On chip, compute $\mathbf {dS}_{ij} = \mathbf P_{ij} \circ (\mathbf {dP}_{ij}-D_i) \in \mathbb R^{B_r \times B_c}$.
> 根据 $\mathbf {dO}$ 和 $\mathbf O$ 计算当前 $B_r$ 行的 $D_i$
> 根据 $\mathbf {dP}_{ij}$ 和 $D_i$ 计算 $\mathbf {dS}_{ij}$  ($d S_{i j}=P_{i j}(d P_{i j}-D_{i})$)

21:     Write $\mathbf {dQ}_i \leftarrow \mathbf {dQ}_i + \tau\mathbf {dS}_{ij}\mathbf K_j\in \mathbb R^{B_r \times d}$ to HBM.
> 更新 $\mathbf {dQ}_i$：$\mathbf {dQ}_i$ 每次外层循环更新一次，因此 $\mathbf {dQ}_i$ 块在内层循环一直保存在片上不现实，故需要写回 HBM，图解见[[#$ mathbf {dQ}$|附录]]

22:     On chip, compute $\tilde {\mathbf {dK}}_j  \leftarrow \tilde {\mathbf {dK}_j} + \tau \mathbf {dS}_{ij}^\top\mathbf Q_i\in\mathbb R^{B_c \times d}$.
> 更新 $\mathbf {dK}_j$：本质是 $\mathbf {dK} = \mathbf {dS}^\top \mathbf {Q}$ 的分块形式，图解见[[#$ mathbf {dK}$|附录]]

23:   **end for**
24:   Write $\mathbf {dK}_j \leftarrow \mathbf {\tilde {dK}}, \mathbf {dV}_j \leftarrow  \mathbf {\tilde {dV}}$ to HBM.
> 内层循环结束后，可以得到计算好的 $\mathbf {dK, dV}$ 块

25: **end for**
26: Return $\mathbf {dQ, dK, dV}$.
> 外层循环结束后，$\mathbf {dQ}$ 才能完整算完

We see that similar to the forward pass, the backward pass performs $O(N^{2})$ FLOPs and only requires $O(N)$ extra memory beyond inputs, output, output gradient, and input gradients. We analyze the IO-complexity of the backward pass, similar to the forward pass (Theorem 2). 
> 反向传播需要 $O (N^2)$ 的 FLOPs，但仅需要 $O (N)$ 的额外内存 (除去输入、输出、输出梯度、输入梯度所占用的内存)

**Theorem 5.** Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with $d\leq M\leq N d$ . Standard attention (Algorithm 0) backward pass requires $\Theta(N d+N^{2})$ HBM accesses, while FlashAttention backward pass (Algorithm 4) requires $\Theta(N^{2}d^{2}M^{-1})$ HBM accesses. 
> 定理：
> $N$ 为序列长度，$d$ 为头维度，$M$ 为 SRAM 大小，满足 $d\le M \le Nd$
> 标准 attention 反向传播需要 $\Theta (Nd + N^2)$ HBM 访问
> FlashAttention 反向传播需要 $\Theta (N^2d^2 M^{-1})$ HBM 访问

The proof is in Appendix C. 

## B.5 Comparison with Rabe and Staats
We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and Staats [66]. 

Conceptually, both FlashAttention and Rabe and Staats [66] operate on blocks of the attention matrix using the well-established technique of tiling (or softmax scaling) [ 51 , 60 ]. To reduce the memory footprint, both methods avoid storing the large attention matrix in the forward pass and recompute it in the backward pass. 
>在概念上，FlashAttention 和 Rabe 与 Staats[66]都使用 tiling (或 softmax scaling) 技术对注意力矩阵进行分块的运算；为了减少内存占用，两种方法都在前向传播中避免存储大的注意力矩阵，并在反向传播中重新计算它

The first major diﬀerence is that Rabe and Staats [66] focuses on the reducing the total memory footprint (maximum amount of GPU memory required) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the primary determining factor of runtime. Reducing memory accesses also necessarily reduces the total amount of memory required (e.g., if an operation incurs $A$ memory accesses, then its total memory requirement is at most $A$ ). As a result, FlashAttention is faster than standard attention (2-4x) while Rabe and Staats [66] is around the same speed or slightly slower than standard attention. In terms of total memory required, both methods offer substantial memory saving. 
> 第一个主要差异：
> [66]关注减少总的显存使用量 (GPU 显存的最大需求量)，而 FlashAttention 关注减少显存访问数 (显存读写的数量)
> 内存访问次数是运行时间的主要决定因素，故减少访问次数可以有效减少运行时间，而减少访问次数的同时也必要地减少了所需要的总内存 (例如一个运算访问内存 $A$ 次，则其最多的内存需求量就是 $A$)

The second difference between the two methods is the way information is summarized from each block to pass to the next block. Rabe and Staats [66] summarizes each block with its temporary output along with the softmax normalization statistics. At the end of the forward pass, the temporary outputs of all the blocks are combined using the statistics to produce the final output. FlashAttention instead incrementally updates the output (Algorithm 1 line 12) after processing each block, so only one copy of the output is needed (instead of $K$ copies for $K$ blocks). This means that FlashAttention has smaller total memory requirement compared to Rabe and Staats [66]. 
> 第二个差异：
> 信息从每个 block 总结并传递给下一个块的方式不同
> [66]中每个块都会给出自己暂时的输出以及 softmax 规范化统计量，在前向传播最后结合所有块的暂时输出，然后用统计量计算出最后结果
> FlashAttention 在处理每个块之后递增地更新输出 (Algorithm1 line 12)，因此仅需要一个输出拷贝 (而不是 $K$ 个块有 $K$ 个拷贝)， 因此 FlashAttention 有更少的总内存需求

The final major diﬀerence is the way the backward pass is computed. Rabe and Staats [66] uses gradient checkpointing to recompute the attention matrix and the temporary output of each block. FlashAttention instead simplifies the backward pass analytically (Appendices B.2 and B.4). It only recomputes the attention matrix and does not recompute the temporary output of each block. This reduces the memory requirement for the backward pass and yields speedup. 
> 第三个主要差异：反向传播
> [66]使用梯度检查点重新计算 attention 和每个块的暂时输出
> FlashAttention 解析上地简化了反向传播，仅重新计算 attention 矩阵，不需要重新计算每个块的暂时输出，这减少了反向传播的内存需求

# C Proofs 
*Proof of Theorem 1.* 
We first count the number of FLOPs and extra memory required. 

The dominating FLOPs are from matrix multiplication. In the inner loop, (Algorithm 1 line 9), we compute $\mathbf{Q}_{i}\mathbf{K}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$ for $\mathbf{Q}_{i}\in\mathbb{R}^{B_{r}\times d}$ and $\mathbf{K}_{j}\in\mathbb{R}^{B_{c}\times d}$ , which takes $O(B_{r}B_{c}d)$ FLOPs. We also compute (Algorithm 1 line 12) $\tilde{\mathbf{P}}_{i j}\mathbf{V}_{j}\in\mathbb{R}^{B_{r}\times d}$ for $\tilde{\mathbf{P}}_{i j}\in\mathbb{R}^{B_{r}\times B_{c}}$ and $\mathbf{V}_{j}\in\mathbb{R}^{B_{c}\times d}$ , which takes $O(B_{r}B_{c}d)$ FLOPs. We execute the inner loops $\begin{array}{r}{T_{c}T_{r}=\left\lceil\frac{N}{B_{c}}\right\rceil\left\lceil\frac{N}{B_{r}}\right\rceil}\end{array}$ times. 
Therefore the total number of FLOPs is 

$$
O\left(\frac{N^{2}}{B_{c}B_{r}}B_{r}B_{c}d\right)=O(N^{2}d).
$$ 
> FLOPs:
> Algorithm 1 的内层循环中，line 9 计算了 $\mathbf Q_i \mathbf K_j^\top \in \mathbb R^{B_r \times B_c}$，其中 $\mathbf Q_i \in \mathbb R^{B_r\times d}, \mathbf K_j \in \mathbb R^{B_c \times d}$，该矩阵乘法的 FLOPs 是 $O (B_r B_c d)$
> Algorithm 1 的内层循环中，line 12 计算了 $\tilde {\mathbf P}_{ij}\mathbf V_j \in \mathbb R^{B_r \times d}$，其中 $\tilde {\mathbf P}_{ij} \in \mathbb R^{B_r \times B_c}, \mathbf V_j \in \mathbb R^{B_c \times d}$，该矩阵乘法的 FLOPs 是 $O (B_rB_cd)$
>
> 内层循环一共被执行了 $T_c T_r = \lceil \frac N {B_c} \rceil \lceil \frac N {B_r} \rceil$ 次
> 故总 FLOPs 即它们相乘，如上所示，得到 $O (N^2 d)$

In terms of extra memory required, we see that we need $O(N)$ memory to store the statistics $(\ell,m)$ . 
> 统计量 $\ell, m$ 都为 $N$ 维向量，前向传播中，需要额外的 $O (N)$ 内存来存储这两个统计量

We now prove the algorithm’s correctness by induction on $j$ for $0\,\leq\,j\,\leq\,T_{c}$ . Let $\mathbf{K}_{:j}\,\in\,\mathbb{R}^{j B_{c}\times d}$ be the first $j B_{c}$ rows of $\mathbf K$ , and similarly $\mathbf{V}_{:j}\in\mathbb{R}^{j B_{c}\times d}$ the the first $j B_{c}$ rows of $\mathbf{V}$ . Let $\mathbf{S}_{:,:j}=\mathbf{Q}\mathbf{K}_{:j}^{\top}\in\mathbb{R}^{N\times j B_{c}}$ , and $\mathbf{P}_{:,:j}=\mathrm{softmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N\times j B_{c}}$ (softmax applied row-wise). Let $m^{(j)},\ell^{(j)},\mathbf{O}^{(j)}$ be the values of $m,\ell,\mathbf{O}$ in HBM after the $j$ -th iteration of the outer loop (Algorithm 1 line 5). (Note that these values of $m,\ell,\mathbf{O}$ are updated after each iteration of the outer loop.) 
> 接下来通过对 $0\le j \le T_c$  ($j$ 是外层循环次数)归纳来证明算法的正确性
> $\mathbf K, \mathbf V$ 的前 $jB_c$ 行记为 $\mathbf K_{: j}, \mathbf V_{:j}  \in \mathbb R^{jB_c \times d}$，$\mathbf S_{:, : j} = \mathbf Q\mathbf K_{: j}^\top \in \mathbb R^{N\times jB_c}$, $\mathbf P_{:,: j} = \text{softmax}(\mathbf S_{:,: j}) \in \mathbb R^{N\times jB_c}$，记 $m^{(j)}, \ell^{(j)}, \mathbf O^{(j)}$ 为第 $j$ 次外层循环之后 $m, \ell, \mathbf O$ 在 HBM 中的值 (它们的值在每一次外层循环更新一次)

We want to show that after the $j$ -th iteration of the outer loop, we have computed in HBM: 
> 要证明的是在第 $j$ 轮外层循环之后，HBM 中的计算结果为：

$$
\begin{align}
m^{(j)} &= \text{rowmax}(\mathbf S_{:,:j})\in \mathbb R^N\\
\ell^{(j)} &= \text{rowsum}(\exp(\mathbf S_{:,:j}-m^{(j)})\in \mathbb R^N\\
\mathbf O^{(j)}&= \mathbf P_{:,:j}\mathbf V_{:,j} \in \mathbb R^{N\times d}
\end{align}
$$

Based on our initialization (Algorithm 1 line 2), this claim is true for $j=0$ (i.e., before the any iteration of the outer loop is executed). Suppose that the claim holds for some $j=0,\dots,T_{c}-1$ . We want to show that the claim also holds for $j+1$ . 

Indeed, when we update the statistics in the inner loop (Algorithm 1 line 10) on the $(j+1)$ -th iteration of the outer loop, we update $m^{(j+1)}=\operatorname*{max}(m^{(j)},\tilde{m})$  where $\tilde{m}\in\mathbb{R}^{N}$ is the row-max of $\mathbf{S}_{:,j:j+1}$ , the slice of $\mathbf S$ from column $j B_{c}$ to column $(j+1)B_{c}-1$ . This implies that 
> 在第 $j+1$ 次外层循环中，我们在 Algorithm 1 line 11 按照 $m^{(j+1)} = \max (m^{(j)}, \tilde m)$ 更新统计量 $m$，
> 其中 $\tilde m \in \mathbb R^N$ 就是切片 $\mathbf S_{:, j:j+1}$ (从第 $jB_c$ 列到第 $(j+1) B_c - 1$ 列) 的 rowmax，而 $m^{(j)}$ 则是 $\mathbf S_{:, :j}$ (从第 $1$ 列到第 $jB_c - 1$ 列)的 rowmax，显然，$m^{(j+1)}$ 就是 $\mathbf S_{:,:j+1}$ (从第 $1$ 列到第 $(j+1) B_c - 1$ 列) 的 rowmax

$$
\begin{array}{r}{m^{(j+1)}=\operatorname{rowmax}(\mathbf{S}_{:,:j+1})\in\mathbb{R}^{N}.}\end{array}
$$ 
Similarly, we update 

$$
\ell^{(j+1)}=e^{m^{(j)}-m^{(j+1)}}\ell^{(j)}+e^{\tilde{m}-m^{(j+1)}}\tilde{\ell},
$$ 
where $\begin{array}{r}{\tilde{\ell}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,j:j+1}-\tilde{m}))\in\mathbb{R}^{N}}\end{array}$ . By the same algebraic manipulation in Section 3.1, we obtain: 

$$
\ell^{(j+1)}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,:j+1}-m^{(j+1)}))\in\mathbb{R}^{N}.
$$
> 类似地，$\mathcal \ell$ 的更新公式 $\ell^{(j+1)}=e^{m^{(j)}-m^{(j+1)}}\ell^{(j)}+e^{\tilde{m}-m^{(j+1)}}\tilde{\ell}$ 中，$\ell^{(j)}$ 为 $\mathbf S_{:, :j}$ 的放缩后指数 rowsum，$\tilde \ell$ 为 $\mathbf S_{:, j:j+1}$ 的放缩后指数 rowsum，更新公式使用最新的 $m$ 重放缩 $\ell^{(j)}$ 和 $\tilde \ell$，然后相加，得到 $\mathbf S_{:, :j+1}$ 的放缩后指数 rowsum

Let $\mathbf{V}_{j:j+1}$ be the slice of $\mathbf{V}$ from column $j B_{c}$ to column $(j+1)B_{c}-1$ , we also update: 

$$\begin{align*} 
\mathbf{O}^{(j+1)} &= \mathrm{diag}(\ell^{(j+1)})^{-1} \left( \mathrm{diag}(\ell^{(j)}) e^{m^{(j)} - m^{(j+1)}} \mathbf{O}^{(j)} + e^{\tilde{m} - m^{(j+1)}} \exp(\mathbf{S}_{j:j+1} - \tilde{m}) \mathbf{V}_{j:j+1} \right) \\ 
&= \mathrm{diag}(\ell^{(j+1)})^{-1} \left( \mathrm{diag}(\ell^{(j)}) e^{m^{(j)} - m^{(j+1)}} \mathbf{P}_{:,:j} \mathbf{V}_{:j} + e^{-m^{(j+1)}} \exp(\mathbf{S}_{j:j+1}) \mathbf{V}_{j:j+1} \right) \\
&= \mathrm{diag}(\ell^{(j+1)})^{-1} \left( \mathrm{diag}(\ell^{(j)}) e^{m^{(j)} - m^{(j+1)}} \mathrm{diag}(\ell^{(j)})^{-1}\exp(\mathbf{S}_{:,:j} - m^{(j)}) \mathbf{V}_{:j} + e^{-m^{(j+1)}} \exp(\mathbf{S}_{j:j+1}) \right) \\ 
&= \mathrm{diag}(\ell^{(j+1)})^{-1} \left( e^{-m^{(j+1)}} \exp(\mathbf{S}_{:,:j}) \mathbf{V}_{:j} + e^{-m^{(j+1)}} \exp(\mathbf{S}_{j:j+1}) \mathbf{V}_{j:j+1} \right) \\ 
&= \mathrm{diag}(\ell^{(j+1)})^{-1} \left( \exp(\mathbf{S}_{:,:j} - m^{(j+1)}) \mathbf{V}_{:j} + \exp(\mathbf{S}_{j:j+1} - m^{(j+1)}) \mathbf{V}_{j:j+1} \right) \\ 
&= \mathrm{diag}(\ell^{(j+1)})^{-1} \left( \exp \left( \left[ \mathbf{S}_{:,:j} \quad \mathbf{S}_{j:j+1} \right] - m^{(j+1)} \right) \right) \begin{bmatrix}\mathbf V_{:j}\\\mathbf{V}_{j:j+1}\end{bmatrix}\\ 
&= \mathrm{softmax}(\mathbf{S}_{:,:j+1}) \mathbf{V}_{:j+1}. \end{align*}$$

>  $\mathbf O$ 在外层循环之间的更新公式如上，它做的事情包括：
>  调节 $\mathbf V_{:, j}$ 加权求和的权重 (权重的相对大小没有改变，只是随着 $\ell$ 的更新而更新了归一化常数)
>  调节 $\mathbf V_{:, j: j+1}$ 加权求和的权重 (同样，相对大小不变，只是更新了归一化常数)
>  对 $\mathbf V_{:, j}$ 和 $\mathbf V_{:, j:j+1}$ 加权求和，然后相加，得到更新的 $\mathbf O$

We then see that the claim is also true for $j+1$ . By duction, the claim is true for all $j=0,\dots,T_{c}$ . When $j=T_{c}$ , we conclude that the final value of $\mathbf O$ in HBM is $\text{softmax}(\mathbf S)\mathbf V = \text{softmax }(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$ . 

*Proof of Theorem 2.* 
We first analyze the IO complexity of standard attention implementation. The inputs $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ reside in HBM, and the at the end of the algorithm the output $\mathbf{O}\in\mathbb{R}^{N\times d}$ is written to HBM. 
> 首先分析标准 attention 实现的 IO 复杂度
> 输入 $\mathbf {Q, K, V}\in \mathbb R^{N\times d}$ 存储于 HBM 中，输出 $\mathbf O \in \mathbb R^{N\times d}$ 需要写回 HBM

In the first step of computing the matrix multiply $\begin{array}{r}{\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}}\end{array}$ , the inputs $\mathbf{Q},\mathbf{K}$ are read from HBM and the output $\mathbf{S}\in\mathbb{R}^{N\times N}$ is written to HBM (Algorithm 0 line 1). This incurs $\Theta(N d+N^{2})$ HBM accesses. 
> 第一步 (Algorithm 0 line 1) 计算 $\mathbf S = \mathbf Q \mathbf K^\top$，输入 $\mathbf {Q, K}\in \mathbb R^{N\times d}$ 需要从 HBM 读取，输出 $\mathbf S\in \mathbb R^{N\times N}$ 需要写回 HBM，一共需要 $\Theta (Nd + N^2)$ HBM 访问

In the second step of computing $\mathbf{P}=\mathrm{softmax}(\mathbf{S})$ , the input $\mathbf S$ is read from HBM and the output $\mathbf{P}$ is written to HBM (Algorithm 0 lin 2). This incurs $\Theta(N^{2})$ HBM accesses. 
> 第二步 (Algorithm 0 line 2) 计算 $\mathbf P = \text{softmax}(\mathbf S)$，输入 $\mathbf S \in \mathbb R^{N\times N}$ 从 HBM 读取，输出 $\mathbf P\in \mathbb R^{N\times N}$ 需要写回 HBM，一共需要 $\Theta (N^2)$ HBM 访问

In the last step of computing $\mathbf{O}=\mathbf{P}\mathbf{V}$ , the inputs $\mathbf{P},\mathbf{V}$ are read from global memory and the output $\mathbf{O}$ is written to HBM (Algorithm 0 line 3). This incurs $\Theta(N d+N^{2})$ HBM accesses. 
> 第三步 (Algorithm 0 line 3) 计算 $\mathbf O = \mathbf {PV}$，输入 $\mathbf P \in \mathbb R^{N\times N}, \mathbf V \in \mathbb R^{N\times d}$ 从 HBM 读取，输出 $\mathbf O\in \mathbb R^{N\times N}$ 需要写回 HBM，一共需要 $\Theta (Nd + N^2)$ HBM 访问

Overall, standard attention implementation requires $\Theta(N d+N^{2})$ global memory accesses. 
> 故标准算法需要 $\Theta (Nd + N^2)$ HBM 访问

We now analyze the IO complexity of streaming attention. 

Following Algorithm 1, we see that each element of $\mathbf{K}$ and $\mathbf{V}$ is loaded from HBM once (Algorithm 1 line 6). We make $T_{c}$ passes over $\mathbf{Q}$ and $\mathbf{O}$ , each pass loading all of $\mathbf{Q}$ and all of $\mathbf{O}$ to HBM (Algorithm 1 line 8). Therefore the number of HBM accesses is $\Theta\left(N d+N d T_{c}\right)=\Theta(N d T_{c})$ . 
> Algorithm 1中，$\mathbf {K, V}$ 中的每个元素仅从 HBM 中装载到 SRAM 中一次
> Algorithm 1中，我们对 $\mathbf Q, \mathbf O$ 进行了 $T_c$ 次遍历 ($T_c$ 次外层循环)，每次遍历都会陆续将 $\mathbf {Q, O}$ 的全部元素从 HBM 中装载到 SRAM 中一次
> 因此 HBM 访问次数是 $\Theta (Nd + NdT_c) = \Theta (Nd T_c)$

We derive the conditions on the block sizes $B_{c}$ and $B_{r}$ . We need the blocks $\mathbf{K}_{j}$ and $\mathbf{V}_{j}$ of size $B_{c}\times d$ to fit into on-chip memory, which translates to: 

$$
B_{c}d=O(M)\Leftrightarrow B_{c}=O\left({\frac{M}{d}}\right).
$$ 
Similarly, we need the blocks $\mathbf{Q}_{i},\mathbf{0}_{i}$ of size $B_{r}\times d$ to fit into on-chip memory, which translates to: 

$$
B_{r}d=O(M)\Leftrightarrow B_{r}=O\left({\frac{M}{d}}\right).
$$ 
Finally, we need the block $\mathbf{S}_{i j}$ of size $B_{r}\times B_{c}$ to fit into on-chip memory, which translates to: 

$$
B_{r}B_{c}={{O}}(M).
$$

> $B_c, B_r$ 需要满足 $B_c = O (\frac M d), B_r = O (\frac M d), B_rB_c = O (M)$

We therefore set: 

$$
B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\operatorname*{min}\left(\frac{M}{d},\frac{M}{B_{c}}\right)\right)=\Theta\left(\operatorname*{min}\left(\frac{M}{d},d\right)\right).
$$ 
We then have: 

$$
T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{N d}{M}\right).
$$ 
As a result, the number of HBM accesses is: 

$$
\Theta\left(N d T_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).
$$ 
*Proof of Proposition 3.* For contradiction, suppose that there exists an algorithm that computes exact attention where the number for HBM access for all $M\in[d,N d]$ is 
> 反证法，假定存在复杂度如下的精确 attention 算法

$$
o\left(\frac{N^{2}d^{2}}{M}\right).
$$ 
In the regime of $M=\Theta(N d)$ , this results in the number of HBM accesses: 

$$
o\left(\frac{N^{2}d^{2}}{N d}\right)=o(N d).
$$ 
However, the input to attention (matrices $\mathbf{Q},\mathbf{K},\mathbf{V}$ ) and the output $\mathbf{O}$ have size $N d$ and they start out being in HBM, so if the algorithm computes exact attention it must incur at least $\Omega(N d)$ HBM accesses. This is a contradiction.
> attention 的输入和输出的大小都为 $Nd$，从 HBM 读取输入的访问次数就至少为 $\Omega (Nd)$，故矛盾

*Proof of Theorem 5.* The IO complexity of the attention backward is very similar to the IO complexity of the attention forward (Theorem 2). Here we provide a sketch of the proof. 

We first analyze the IO complexity of standard attention backward pass. The inputs $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{d}\mathbf{O}\in\mathbb{R}^{N\times d}$ reside in HBM, and the at the end of the algorithm the outputs $\mathbf{dQ},\mathbf{dK},\mathbf{dV}\in\mathbb{R}^{N\times d}$ are written to HBM. 
> 首先分析标准 attention 实现的 IO 复杂度
> 输入 $\mathbf {Q, K, V, dO}\in \mathbb R^{N\times d}$ 存储于 HBM 中，输出 $\mathbf {dQ, dK, dV} \in \mathbb R^{N\times d}$ 需要写回 HBM

At each step of the standard attention backward pass, one needs to load inputs of size 𝑁𝑑 or $N^{2}$ from HBM, and needs to write the outputs of size $N^{2}$ or $N d$ to HBM. This incurs $\Theta(N d+N^{2})$ HBM accesses.
> 标准算法中的每一步都需要从 HBM 中装载大小为 $Nd$ 或 $N^2$ 的输入，并且需要将大小为 $Nd$ 或 $N^2$ 的输出写回 HBM，因此访问次数为 $\Theta (Nd + N^2)$

We now analyze the IO complexity of FlashAttention backward pass. 

Similar to Theorem 2, we see that each element of $\mathbf{K}$ and $\mathbf{V}$ is loaded from HBM once. Each element of $\mathbf {dK}$ and $\mathbf {dV}$ is only written to HBM once. We make $T_{c}$ passes over $\mathbf{Q},\mathbf{O},\mathbf{dO}$ , each pass loading all of $\mathbf {Q , O , dO}$ to HBM. We also make $T_{c}$ passes over $\mathbf{d}\mathbf{Q}$ , each pass reading/writing all of $\mathbf {dQ}$ from/to HBM. Therefore the number of HBM accesses is $\Theta\left(N d+N d T_{c}\right)=\Theta(N d T_{c})$ . 
> $\mathbf {K, V}$ 的每个元素仅装载一次，$\mathbf {dK, dV}$ 的每个元素仅写回一次
> 算法对 $\mathbf {Q, O, dO}$ 遍历了 $T_c$ 次，每次遍历装载全部的 $\mathbf {Q, O ,dO}$
> 算法对 $\mathbf {dQ}$ 遍历了 $T_c$ 次，每次遍历读写全部的 $\mathbf {dQ}$
> 故 HBM 访问次数为 $\Theta (Nd + NdT_c) = \Theta (NdT_c)$

As in the proof of Theorem 2, the constraints on the block sizes are that: 

$$
B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\operatorname*{min}\left(\frac{M}{d},d\right)\right).
$$ 
We then have: 

$$
T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{N d}{M}\right).
$$ 
As a result, the number of HBM accesses is: 

$$
\Theta\left(N d T_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).
$$ 
# D Extension Details 
## D.1 Block-sparse FlashAttention 
We describe the full block-sparse FlashAttention algorithm in Algorithm 5. The algorithm is identical to Algorithm 2, except that we skip zero blocks. 
> 块稀疏 FlashAttention 的算法和普通前向算法唯一的区别就是跳过了零块

![[FlashAttention-Algorithm 5.png]]

**Algorithm 5** Block-Sparse FlashAttention Forward Pass
**Require**: Matrices $\mathbf {Q, K, V} \in \mathbb R^{N\times d}$ in HBM, on-chip SRAM of size $\bf M$, softmax scaling constant $\tau \in \mathbb R$, masking function $\text{MASK}$, dropout probability $p_{\text{drop}}$, block sizes $B_c = \lceil \frac {M}{4d} \rceil, B_r = \min (\lceil \frac {M}{4d}\rceil, d)$, block sparsity mask $M \in \{0, 1\}^{N/B_r \times N/B_c}$
 1: Initialize the pseudo-random number generator state $\mathcal R$ and save to HBM.
 2: Initialize $\mathbf O = (0)_{N\times d} \in \mathbb R^{N\times d}$, $\ell = (0)_N \in \mathbb R^N$, $m = (-\infty)_N \in \mathbb R^N$ in HBM.
 3: Divide $\mathbf Q$ into $T_r = \lceil \frac {N}{B_r} \rceil$ blocks $\mathbf Q_1, \dots, \mathbf Q_{T_r}$ of size $B_r\times d$ each, and divide $\mathbf K, \mathbf V$ into $T_c = \lceil \frac {N}{B_c} \rceil$ blocks $\mathbf K_q, \dots, \mathbf K_{T_c}$ and $\mathbf V_1, \dots, \mathbf V_{T_c}$, of size $B_c \times d$ each.
 4: Divide $\mathbf O$ into $T_r$ blocks $\mathbf O_i, \dots, \mathbf O_{T_r}$ of size $B_r \times d$ each, divide $\ell$ into $T_r$ blocks $\ell_i, \dots, \ell_{T_r}$ of size $B_r$ each, divide $m$ into $T_r$ blocks $m_1, \dots, m_{T_r}$ of size $B_r$ each.
 5: **for** $1\le j \le T_c$ **do**
 6:    Load $\mathbf K_j, \mathbf V_j$ from HBM to on-chip SRAM.
 7:    **for** $1\le i \le T_r$ **do**
 8:      **if** $M_{ij} \ne 0$ **then**
 9:        Load $\mathbf Q_i, \mathbf O_i , \ell_i, m_i$ from HBM to on-chip SRAM.
10:       On chip, computes $\mathbf S_{ij} = \tau \mathbf Q_i \mathbf K_{j}^T\in \mathbb R^{B_r \times B_c}$.
11:        On chip, computes $\mathbf S_{ij}^{\text{masked}}  = \text{MASK}(\mathbf S_{ij})$.
12:        On chip, computes $\tilde m_{ij} = \text{rowmax}(\mathbf S_{ij}^{\text{masked}}) \in \mathbb R^{B_r}$, $\tilde {\mathbf P}_{ij}=\exp (\mathbf S_{ij}^{\text{masked}} - \tilde m_{ij})\in \mathbb R^{B_r \times B_c}$ (pointwise), $\tilde \ell_{ij} = \text{rowsum}(\tilde {\mathbf P}_{ij} )\in \mathbb R^{B_r}$.
13:        On chip, compute $m_i^{\text{new}} = \max (m_i, \tilde m_{ij}) \in \mathbb R^{B_r}$, $\ell_i^{\text{new}} = e^{m_i - m_i^{new}} \ell_i  + e^{\tilde m_{ij} - m_i^{\text{new}}} \tilde \ell_{ij} \in \mathbb R^{B_r}$.
14:        On chip, compute $\tilde {\mathbf P}_{ij}^{\text{dropped}} = \text{dropout}(\tilde {\mathbf P}_{ij}, p_{\text{drop}})$.
15:        Write $\mathbf O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1}(\text{diag}(\ell_i)e^{m_i-m_i^{\text{new}}}\mathbf O_i + e^{\tilde m_{ij} - m_i^{\text{new}}}\tilde {\mathbf P}_{ij}^{\text{dropped}}\mathbf V_j)$ to HBM.
16:        Write $\ell_i \leftarrow \ell_i^{\text{new}}, m_i \leftarrow m_i^{\text{new}}$ to HBM.
17:      **end if**
18:    **end for**
19:  **end for**
20: Return $\mathbf O,\ell, m, \mathcal R$.

We prove the IO-complexity of block-sparse FlashAttention . 

*Proof of Proposition 4.* The proof is very similar to the proof of Theorem 2. For the block-sparse case, notice that we only need to load blocks corresponding to nonzero blocks. As a result, the number of HBM accesses are scaled by $s$ , the fraction of nonzero blocks in the block-sparsity mask. However, for small values of $s$ , we would still need to write the result $\mathbf{O}\in\mathbb{R}^{N\times d}$ . Therefore the number of HBM accesses is 
> 类似定理二的证明，实际的 HBM 访问应该为 $\Theta (Nd + N^2d^2 T_cs)$，然后将 $T_c$ 代入即可得到 $\Theta (Nd + \frac {N^2d^2}{M}s)$
> 证明定理二时，将 $\Theta (Nd + N^2 d^2 T_c)$ 化简为了 $\Theta (N^2d^2T_c)$，因为 $T_c$ 显然大于 $1$，故 $N^2d^2 T_c$ 的数量级显然高于 $Nd$，而 block-sparse 下该项会受 $s$ 放缩，如果 $s$ 很小，则 $Nd$ 不可忽视，故此时前面的 $Nd$ 项选择不化简而保留

$$
\Theta\left(N d+\frac{N^{2}d^{2}}{M}s\right)\,.
$$ 
## D.2 Potential Extensions 
We discuss here a few potential extensions of the IO-aware approach to speed up deep learning training. 

**Multi-GPU Attention.** Large language models are trained on hundreds or thousands of GPUs, and one typically splits the attention computation between 4-8 GPUs on the same node [77]. This introduces another level of memory hierarchy: beside GPU SRAM and GPU HBM, we also have the HBM of other GPUs. For very long sequences, the diﬀerent GPUs on the same node can cooperate to compute attention by taking into account the asymmetry of diﬀerent levels of memory hierarchy. 
> 多 GPU 上的 attention 计算引入了新的内存层次：其他 GPU 的 HBM 和 SRAM

**Sparse MLP layers.** Typical dense MLP layers are compute-bound and not memory-bound. To improve their efficiency, MLP layers with sparse weight matrices can be used [17]. However, many sparse MLP layers are instead memory-bound, and their speedup is often not proportional to the sparsity. We believe that an IO-aware implementation can alleviate this issue and realize the benefits of sparsity. We are excited about future work in this direction, to reduce the computational requirement of large models and improve their wall-block runtime. 
> dense MLP 层一般是 compute-bound 而不是 memory-bound
> sparse MLP 层则存在 memory-bound，存在 IO-aware 优化的可能性

**Kernel machine learning.** Our approach in FlashAttention relies on the fact that the $N\times N$ attention matrix is a function of a low-rank matrix $\mathbf {QK}^{\top}$ (of rank $d\ll N$  ). As a result, we can repeatedly load the inputs $\mathbf{Q},\mathbf{K}$ and recompute the block of the attention matrix that we need, significantly reducing HBM access. As similar scenario happens in kernel machine learning: each element $K_{i j}$ of the $N\times N$ kernel matrix $\mathbf{K}$ is a function of two vectors of size $d\ll N$ , as it measures the similarity between two datapoints $x_{i}$ and $x_{j}$ . The KeOps library [8 , 26] is a successful example of how reducing memory reads/writes can speed up kernel operations. We hope that this will motivate kernel methods that focus more on reducing IOs instead of just FLOPs. 
> $N\times N$ 的 attention 矩阵实际是由低秩矩阵 $\mathbf {QK}^{\top}$ 计算得到 ($d\ll N$)，FlashAttention 利用这一点，在 $N$ 维 tile，反复装载 $\mathbf {Q, K}$ 块，重计算所需的 attention 矩阵块
> kernel ML 中，$N\times N$ 的 $\mathbf K$ 矩阵中 $K_{ij}$ 是关于两个大小为 $d\ll N$ 的向量的函数，是相似的

# E Full Experimental Results 
## E.1 BERT 
We train BERT-large following the training procedure and hyperparameters of the reference MLPerf 1.1 implementation. In particular, we use the LAMB optimizer with learning rate 3.75e-3, with batch size 448, trained for at most 7100 steps. The training is stopped once the validation accuracy (for masked language modeling) reaches the target $72.0\%$ , and the wall-clock run-time is measured. We train with FP16 precision using Apex AMP (with O2 optimization level). 

We compare our results with the reported training speed from Nvidia that was submitted to MLPerf 1.1 (Table 1). We use the same train / validation data split provided by MLPerf 1.1 reference implementation. In particular, we evaluate on the same 10000 validation examples as the baseline from Nvidia. We train the model on 8 $\times$ A100-80GB GPUs. Each training run takes between 16 and 19 minutes, and we average the results of 10 runs. 
> 模型：BERT-large
> 超参数：和 MLPerf 1.1 一致
> 精度：FP16 Apex AMP

## E.2 GPT-2 
We use the standard implementations of GPT-2 [67] from Huggingface transformers library and from Nvidia’s Megatron-LM repo. We follow the training recipe of the Megatron-LM repo. 

We use an eﬀective batch size of 512, and use gradient accumulation to fit into available GPU memory. We use the AdamW optimizer, with learning rate 6e-4 for GPT-2 small and 1.5e-4 for GPT-2 medium, and weight decay of 0.1. All models are trained with the same hyperparameters for 400K steps. We run all implementations with mixed-precision training (PyTorch AMP). 

We use the Openwebtext dataset, with the GPT-2 BPE tokenizer. We randomly select $0.5\%$ of the dataset as the validation set, with the rest being used as training set. This random selection of validation set is done once, and all models are evaluated on the same validation set. 

We train the model on 8 $\times$ A100-40GB GPUs, and we measure the wall-clock training time. Training GPT-2 small takes between 2.7-9.5 days, and training GPT-2 medium takes between 6.9-21.0 days (Table 2). 
> 模型：GPT-2 small/medium
> 精度：PyTorch AMP
> 数据集：Openwebtext

In Fig. 4, we plot of the validation perplexity throughout training of GPT-2 small/medium, using either HuggingFace implementation or our FlashAttention implementation. We see that FlashAttention behaves the same as the baseline implementation and the validation perplexity curves of the two implementations almost lie on top of each other. 
> FlashAttention 的验证集 perplexity 曲线和 HuggingFace 的实现完全重合

![[FlashAttention-Fig4.png]]

**Long Document Classification.** For MIMIC-III and ECtHR, we follow the hyperparameters of Dai et al. [13]. 
## E.3 LRA details 
We follow the hyperparameters from the Long-range arena paper [ 80 ], the Long-range arena repo ( https: //github.com/google-research/long-range-arena ), and the Nyströmformer reproduction [ 90 ]. To be generous to the baseline methods, if we are unable to reproduce the performance of any baseline for any of the five tasks, we report the better performance from Tay et al. [80] or Xiong et al. [90] for that baseline on that task. 

After hyperparameter tuning, almost all of the attention methods achieve similar accuracy on all of the five LRA tasks. We run all methods with mixed-precision training, except for Performer (not stable with mixed precision) and Local Attention (implementation does not support FP16). To calculate the overall wallclock-time speedup, we take the geometric mean of the wallclock-time speedup of each of the five tasks. 

**Path-X** For Path-X and Path-256, we follow the hyperparameters from the PathFinder-32 experiments from the long-range arena paper [80]. For both, we first pretrain a model on Path-64. We take the checkpoint after 200 epochs, upsample its positional embedding (we duplicate the positional embeddings gridwise in space), and fine-tune it on the downstream task for 200 epochs with one epoch of linear warmup, and cosine decay of the learning rate. For Path-X, we take the best performing checkpoint (according to val accuracy), and additionally fine-tune it for 200 epochs with the same warmup and learning rate (this adds roughly 4 points of accuracy to FlashAttention for Path-X, but the model starts overfitting afterwards). 

## E.4 Comparison with Apex FMHA 
We compare our method/implementation with Apex FMHA ( https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha ). 

![[FlashAttention-Table7.png]]

When we started this project, Apex FMHA was the fastest implementation of attention (that we knew of), tailored for short sequences of length at most 512. In fact, almost all MLPerf submissions for BERT training benchmark running on Nvidia GPUs use FMHA for their model code, as of MLPerf 1.1 [58]. Since FMHA targets BERT models, it only supports head dimension 64, and only runs on A100 GPUs. FMHA fuses the attention computation $\text{dropout}(\text{softmax}(\text{MASK}(\mathbf {QK}^{\top})))\mathbf V$ into one CUDA kernel. In the forward pass, it stores the attention matrix $\text{softmax} (\text{MASK} (\mathbf {QK}))$ to HBM to be used in gradient computation. As a result, it does not oﬀer substantial memory saving (though for shorter sequences memory footprint is often not a primary concern). 
> Apex FMHA 仅支持 64 维 head dimension，仅运行于 A100 GPU
> FMHA 将计算 $\text{dropout}(\text{softmax}(\text{MASK}(\mathbf {QK}^{\top})))\mathbf V$ 融合到一个 CUDA kernel，但在前向传播时，它将 attention 矩阵 $\text{softmax} (\text{MASK} (\mathbf {QK}))$ 存储到 HBM 以在反向传播中使用，故并没有节省内存

We use FMHA code as a starting point, and apply two well-established techniques (tiling and recomputation) to deal with long sequences and to save memory as mentioned in Section 3. As a result, we can support much longer sequences (e.g., up to length 64K). We also support more head dimensions (16, 32, 64, 128) and broader GPU types (all Turing and Ampere GPUs at the time of writing). 
> 我们用 tiling 和 recomputation 拓展了 FMHA，使其可以处理更长序列、支持更多 GPU 类型

In Table 7, we compare the performance of FlashAttention and Apex FMHA for short sequences (as FMHA only supports sequence length at most 512). Generally FlashAttention is slightly faster than FMHA in the forward pass and slightly slower than FMHA in the backward pass. This is because we do not store the attention matrix in the forward pass and recompute it in the backward pass. Compared to FMHA, the overall runtime of FlashAttention is about 4% slower for sequence length 128, 8% faster for sequence length 256, and 5% faster for sequence length 512. 
> FlashAttention 在短序列长度下反向传播比 FMHA 略慢，原因是 FlashAttention没有存储前向传播中的 attention 矩阵而是重计算

## E.5 Speedup On Different Hardware and Configurations 
Speedup varies between diﬀerent types of GPU types and generations depending on HBM bandwidth and SRAM size. In this section, we profile FlashAttention speedup on diﬀerent GPUs and configurations. 

**A100** Figure 5 shows speedup on an A100 GPU with batch size 8, head dimension 64, and 12 attention heads, across diﬀerent sequence lengths. We generally see 2-4 × speedup, and we see more speedup when using dropout and masking due to kernel fusion. 

**A100, Head Dimension 128** Speedup also changes when we increase the head dimension. Each block requires more memory, so we need to use smaller block sizes to fit into SRAM. Figure 6 shows speedup with head dimension 128 on an A100 (batch size 16, 12 heads). We see less speedup overall—but we can still see significant speedup (up to $3\times$ ) with a causal mask, where half the blocks are masked out. 
> head dimension 增大后，每个块的大小需要相应变小
> causal mask 下，speedup 幅度更大

**RTX 3090** Figure 7 shows speedup on an RTX 3090 GPU. Here, we use batch size 12 with 12 attention heads. We observe slightly higher speedups on the RTX 3090 (between 2.5-4.5 $\times$ ), since the memory bandwidth on an RTX 3090 is lower than on an A100 (roughly 900 GB/s vs. 1.5 TB/s). 

**T4** Figure 8 shows speedup on a T4 GPU. T4 SRAM is smaller than A100, so we need to make the block sizes smaller in FlashAttention . As a result, we observe less speedup on T4, which matches the IO complexity analysis in Section 3.2. T4 GPUs are commonly used for inference, so we also report speedup on the forward pass only. 

## E.6 Full Benchmarking Results 
We report the full benchmarking results and experimental details on A100. 

**Baselines** We compare against reference implementations for exact attention from PyTorch/HuggingFace and Megatron, approximate attention, and sparse attention. For approximate attention, we compare against reference implementations of Reformer [ 51 ], Local Attention [ 68 ], Linformer Attention [ 84 ], Smyrf [ 19 ], and LongShortFormer (LSFormer) [ 94 ]. For sparse attention, we compare against reference implementations of Block-Sparse Attention form OpenAI [ 11 ], Longformer[ 3 ], and BigBird Attention [ 92 ]. For the approximate and sparse attention, we use a compression ratio of $1/8$ , or a compressed sequence length of 256, whichever is smaller. 

**Setup** We measure runtime and memory usage of the attention computation with 8 heads of dimension 64, and batch size 16 on a machine with one A100 GPU with 40 GB of GPU HBM. We vary sequence length in our experiments. We compute attention on random vectors for $\mathbf{Q}$ , $\mathbf{K}$ , and $\mathbf{V}$ (we do not measure the projection from the hidden layer). For dropout, we use dropout 0.1; for masking, we use a padding mask with uniformly-random mask lengths between the total sequence length and the total sequence length minus 20. To measure runtime, we take the average of 100 measurements of the attention call. We only measure memory footprint once, since it does not vary between runs. 
> mask 用 padding mask，其长度通过在 \[序列长度-20, 序列长度\] 中均匀随机采样得到

We report timing results on the forward pass, backward pass, and combined forward $^+$ backward pass. We measure each method with and without dropout, masking, or both—except for Block Sparse, Longformer, and BigBird. These methods did not successfully run the backward pass with masking due to a bug in external libraries, so we measured them without masking to be generous. We use FP16 for all measurements, except for Local Attention, whose implementation only supports FP32. 

For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the following exceptions: The Megatron implementation does not support sequence lengths longer than 2048. Block-Sparse (OpenAI) does not support sequence lengths longer than 4096. Longformer and BigBird do not support sequence lengths longer than 8092. 

We measure memory usage on the combined forward $^+$ backward pass, without dropout or masking. 

**Results** Table 8 summarizes all the experimental configurations and contains pointers to the results tables. 

# Appendix
## Figure Illustration for FlashAttention Forward Algorithm
### $\mathbf S$
内层循环 + 外层循环：

![[FlashAttention-App-Fig8.png]]

内层循环：

![[FlashAttention-App-Fig9.png]]

### $\mathbf O$
内层循环 + 外层循环：

![[FlashAttention-App-Fig10.png]]

内层循环：

![[FlashAttention-App-Fig11.png]]

外层循环：

![[FlashAttention-App-Fig12.png]]

### Generalization for forward pass
内层循环 + 外层循环：

![[FlashAttention-App-Fig13.png]]

内层循环：

![[FlashAttention-App-Fig14.png]]

外层循环：

![[FlashAttention-App-Fig15.png]]

## Deductions
### Deduction for chain rule of matrix multiplication
考虑矩阵乘法 $\mathbf {LR} = \mathbf Y$，其中 $\mathbf L \in \mathbb R^{m\times k}, \mathbf R \in \mathbb R^{k\times n}, \mathbf Y \in \mathbb R^{m\times n}$

有 $\mathbf Y$ 相对于某个标量函数 $\phi$ 的导数，记作 $\frac {\partial \phi}{\partial \mathbf Y} = \mathbf {dY}$

#### $\mathbf {dL}$
考虑 ${\mathbf {dL}}$ 中的第 $ij$ 个元素：

$$
\begin{align}
\frac {\partial \phi}{\partial L_{ij}} &= \sum_{k=1}^m\sum_{l=1}^n\frac {\partial \phi}{\partial Y_{kl}}\frac {\partial Y_{kl}}{\partial L_{ij}}\\
&=Tr\left[\left(\frac {\partial \phi}{\partial \mathbf Y}\right)^T\frac {\partial \mathbf Y}{\partial L_{ij}}\right]\\
&=Tr\left[(\mathbf {dY})^T\frac {\partial \mathbf Y}{\partial L_{ij}}\right]
\end{align}
$$

其中：

$$
\begin{align}
\frac {\partial \mathbf Y}{\partial L_{ij}}&= \frac {\partial\begin{bmatrix}
Y_{11} & \cdots & Y_{1n}\\
\vdots & \ddots & \vdots \\
Y_{m1} & \cdots & Y_{mn}
\end{bmatrix}}{\partial L_{ij}}\\
&=\frac {\partial\begin{bmatrix}
\sum_{t=1}^k L_{1t}R_{t1} & \cdots & \sum_{t=1}^k L_{1t}R_{tn}\\
\vdots & \ddots & \vdots \\
\sum_{t=1}^k L_{mt}R_{t1} & \cdots & \sum_{t=1}^k L_{mt}R_{tn}\\
\end{bmatrix}}{\partial L_{ij}}\\
&=\frac {\partial\begin{bmatrix}
0& \cdots & 0\\
\vdots &  & \vdots \\
\sum_{t=1}^k L_{it}R_{t1} &\cdots & \sum_{t=1}^kL_{it}R_{tn}\\
\vdots & & \vdots\\
0 & \cdots & 0\\
\end{bmatrix}}{\partial L_{ij}}\\
&=\frac {\partial\begin{bmatrix}
0& \cdots & 0\\
\vdots &  & \vdots \\
 L_{ij}R_{j1} &\cdots & L_{ij}R_{jn}\\
\vdots & & \vdots\\
0 & \cdots & 0\\
\end{bmatrix}}{\partial L_{ij}}\\
&= {\begin{bmatrix}
0& \cdots & 0\\
\vdots &  & \vdots \\
 R_{j1} &\cdots & R_{jn}\\
\vdots & & \vdots\\
0 & \cdots & 0\\
\end{bmatrix}}\\
&=\begin{bmatrix}
\mathbf 0_n^T\\
\vdots\\
R_{j:}\\
\vdots\\
\mathbf 0_n^T
\end{bmatrix}
\end{align}
$$

注意 $\frac  {\partial \mathbf Y}{\partial L_{ij}} \in \mathbb R^{m\times n}$ 的第 $i$ 行是 $R_{j:}$，也就是 $\mathbf R$ 的第 $j$ 行，其他所有的行都是 $\mathbf 0_n^T$

因此：

$$
\begin{align}
\frac {\partial \phi}{\partial L_{ij}} 
&=Tr\left[(\mathbf {dY})^T\frac {\partial \mathbf Y}{\partial L_{ij}}\right]\\
&=Tr\left[\mathbf {d}\mathbf {Y}^T\frac {\partial \mathbf Y}{\partial L_{ij}}\right]\\
&=Tr\left[\mathbf {dY}^T\begin{bmatrix}
\mathbf 0_n^T\\
\vdots \\
R_{j:}\\
\vdots\\
\mathbf 0_n^T
\end{bmatrix}\right]\\
&=Tr\left[ [(dY^T)_{:1}, \cdots, (dY^T)_{:m}]\begin{bmatrix}
\mathbf 0_n^T\\
\vdots \\
R_{j:}\\
\vdots\\
\mathbf 0_n^T
\end{bmatrix}\right]\\
&=Tr\left[ [(dY_{1:})^T, \cdots, (dY_{m:})^T]\begin{bmatrix}
\mathbf 0_n^T\\
\vdots \\
R_{j:}\\
\vdots\\
\mathbf 0_n^T
\end{bmatrix}\right]\\
&=Tr[R_{j1}(dY_{i:})^T, R_{j2}(dY_{i:})^T,\cdots,R_{jn}(dY_{i:}^T)]\\
&=R_{j1}dY_{i1} + R_{j2}dY_{i2} + \cdots + R_{jn}dY_{in}\\
&=\sum_{t=1}^n dY_{it}R_{jt}\\
&=\sum_{t=1}^n dY_{it}(R^T)_{tj}\\
&=\langle dY_{i:}, R^T_{:j}\rangle
\end{align}
$$

因此 $dL_{ij} = \langle dY_{i:}, R_{: j}^T \rangle$，故显然 

$$\mathbf {dL} = \mathbf {dYR}^T$$

#### $\mathbf {dR}$
将 $\mathbf {LR} = \mathbf Y$ 左右同时转置，得到 $\mathbf {R}^T \mathbf {L}^T = \mathbf {Y}^T$

因此容易知道：

$$
\begin{align}
\mathbf {dR}^T &= \mathbf {dY}^T (\mathbf L^T)^T\\
&=\mathbf {dY}^T \mathbf L\\
\mathbf {dR}&=\mathbf L^T \mathbf {dY}
\end{align}
$$

### Deduction for Jacobian of softmax
考虑 $\pmb y = \text{softmax}(\pmb x)$，其中 $\pmb y, \pmb x \in \mathbb R^{n\times 1}$ ，满足

$$
y_i = \frac {\exp(x_i)} {\sum_{i=1}^n\exp{(x_i)}},i=1,\cdots, n
$$

为了书写方便，记 $\sum_{i=1}^n \exp (x_i) = L$

$\pmb x$ 相对于 $\pmb y$ 的 Jacobian 写作：

$$
\begin{align}
J = \begin{bmatrix}
\frac {\partial y_1}{\partial x_1}& \cdots & \frac {\partial y_n}{\partial x_1}\\
\vdots & \ddots & \vdots \\
\frac {\partial y_1}{\partial x_1}& \cdots & \frac {\partial y_n}{\partial x_1}
\end{bmatrix}
\end{align}
$$

其中，对角线元素满足：

$$
\begin{align}
\frac {\partial y_i}{\partial x_i}&=\frac {\partial y_i}{\partial \exp(x_i)}\frac {\partial \exp(x_i)}{\partial x_i}\\
&=\frac {\partial \frac {t}{t+c}}{\partial t}\cdot \exp(x_i)\\
&=\frac {c}{(t+c)^2}\cdot \exp(x_i)\\
&=\frac {L-\exp(x_i)}{L^2}\cdot \exp(x_i)\\
&=\frac {(L-\exp(x_i))\exp(x_i)}{L^2}\cdot \\
&=\frac {L-\exp(x_i)}{L}\cdot \frac {\exp(x_i)}{L}\\
&=(1-y_i)y_i\\
&=y_i - y_i^2
\end{align}$$

非对角线元素满足：

$$
\begin{align}
\frac {\partial y_j}{\partial x_i}&=\frac {\partial y_j}{\partial \exp(x_i)}\frac {\partial \exp(x_i)}{\partial x_i}\\
&=\frac {\partial \frac {\exp(x_j)}{L}}{\partial \exp(x_i)}\cdot \exp(x_i)\\
&=\frac {\partial \frac {1}{L}}{\partial \exp(x_i)}\cdot \exp(x_j)\cdot\exp(x_i)\\
&=\frac {\partial \frac {1}{t+c}}{\partial t}\cdot \exp(x_j)\cdot\exp(x_i)\\
&=\frac {-1}{(t+c)^2}\cdot \exp(x_j)\cdot\exp(x_i)\\
&=\frac {-\exp(x_i)\exp(x_j)}{L^2}\\
&=-\frac {\exp(x_j)}{L}\frac {\exp(x_i)}{L}\\
&=-y_iy_j
\end{align}
$$

考虑 $\text{diag}(\pmb y) - \pmb y\pmb y^T$ ：

$$
\begin{align}
\text{diag}(\pmb y) - \pmb y\pmb y^T&=\begin{bmatrix}
y_1 - y_1^2 & \cdots & -y_1y_n\\
\vdots & \ddots & \vdots \\
-y_ny_1 & \cdots & y_n - y_n^2
\end{bmatrix}
\end{align}
$$

故显然我们有：

$$
J = \text{diag}(\pmb y) - \pmb y \pmb y^T
$$

### Deduction for $\mathbf {dS}_{i:}$
考虑 $\mathbf {dS}$ 中的第 $ij$ 个元素：

$$
\begin{align}
\frac {\partial \phi}{\partial S_{ij}} &= \sum_{k=1}^n\sum_{l=1}^n \frac {\partial \phi}{\partial P_{kl}} \frac {\partial P_{kl}}{\partial S_{ij}}\\
&=\sum_{l=1}^n\frac {\partial \phi}{\partial P_{il}}\frac {\partial P_{il}}{\partial S_{ij}}\\
&=\sum_{l=1}^ndP_{il}\frac {\partial P_{il}}{\partial S_{ij}}\\
&=\langle dP_{i:}, \frac {\partial P_{i:}}{\partial S_{ij}}\rangle
\end{align}
$$

考虑 $\mathbf {dS}$ 的第 $i$ 行：

$$
\begin{align}
\frac {\partial \phi}{\partial S_{i:}} &=\left[\frac {\partial \phi}{\partial S_{i1}}, \dots, \frac {\partial \phi}{\partial S_{in}}\right]^T\\
&=\left[\langle dP_{i:}, \frac {\partial P_{i:}}{\partial S_{i1}}\rangle, \dots, \langle dP_{i:}, \frac {\partial P_{i:}}{\partial S_{in}}\rangle\right]^T\\
&=\left[\frac {\partial P_{i:}}{\partial S_{i1}}, \dots, \frac {\partial P_{i:}}{\partial S_{in}}\right]^TdP_{i:}\\
&=\begin{bmatrix}
\frac {\partial P_{i1}}{\partial S_{i1}} & \cdots & \frac {\partial P_{in}}{\partial S_{i1}}\\
\vdots & \ddots & \vdots\\
\frac {\partial P_{i1}}{\partial S_{in}} & \cdots &\frac {\partial P_{in}}{\partial S_{in}}
\end{bmatrix}dP_{i:}\\
&=J\cdot dP_{i:}\\
&=(\text{diag}(P_{i:}) - P_{i:}P_{i:}^T)dP_{i:}\\
&=\text{diag}(P_{i:})dP_{i:} - P_{i:}P_{i:}^TdP_{i:}\\
&=P_{i:}\circ dP_{i:} - (P_{i:}^TdP_{i:})P_{i:}
\end{align}
$$

## Figure Illustration for FlashAttention Backward Algorithm
### $\mathbf {dV}$
外层 + 内层循环：

![[FlashAttention-App-Fig1.png]]

内层循环：

![[FlashAttention-App-Fig2.png]]

### $\mathbf {dP}$
外层 + 内层循环：

![[FlashAttention-App-Fig3.png]]
### $\mathbf {dQ}$
外层 + 内层循环：

![[FlashAttention-App-Fig4.png]]

外层循环：

![[FlashAttention-App-Fig5.png]]

### $\mathbf {dK}$
外层 + 内层循环：

![[FlashAttention-App-Fig6.png]]
内层循环：

![[FlashAttention-App-Fig7.png]]

### Generalization for backward pass
#### $\mathbf {dV}$
外层 + 内层：

![[FlashAttention-App-Fig16.png]]

内层：

![[FlashAttention-App-Fig17.png]]

外层：

![[FlashAttention-App-Fig18.png]]

#### $\mathbf {dQ}$
内层 + 外层：

![[FlashAttention-App-Fig19.png]]

内层：

![[FlashAttention-App-Fig20.png]]

外层：

![[FlashAttention-App-Fig21.png]]

#### $\mathbf {dK}$
外层 + 内层：

![[FlashAttention-App-Fig22.png]]

内层：

![[FlashAttention-App-Fig23.png]]

外层：

![[FlashAttention-App-Fig24.png]]

