# Abstract
Scaling Transformers to longer sequence lengths has been a major problem in the last several years, promising to improve performance in language modeling and high-resolution image understanding, as well as to unlock new applications in code, audio, and video generation. The attention layer is the main bottleneck in scaling to longer sequences, as its runtime and memory increase quadratically in the sequence length. 
> 拓展 Transformer 序列长度的难点在于 attention 层的运行时间和内存都随序列长度二次增长

FlashAttention (Dao et al., 2022) exploits the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic) and runtime speedup ( $2–4\times$ compared to optimized baselines), with no approximation. However, FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only $25.40\%$ of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. 
> FlashAttention 相对于 GEMM 算子仅能达到 25.40% 的理论最大 FLOPs/s
> 原因在于 GPU 上 thread block 和 warp 之间的工作划分是次优的，导致低 occupancy 或不必要的 shared memory 读写

We propose FlashAttention-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. 
> FlashAttention-2 有更好的工作划分机制，具体为：
> 1. 调整了算法，减少了 non-mamul FLOPs
> 2. 在多个 thread block 中并行化 attention 计算以提高 occupancy
> 3. 在每个 thread block 内将工作分布给 warps 以减少通过 shared memory 的通讯

These yield around $2\times$ speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations. We empirically validate that when used end-to-end to train GPT-style models, FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU ( $72\%$ model FLOPs utilization). 
> FlashAttention 达到 50-73% A100的理论最大 FLOPs/s，效率接近 GEMM 算子 (225 TFLOPs/s)

# 1 Introduction
Scaling up the context length of Transformers (Vaswani et al., 2017) is a challenge, since the attention layer at their heart has runtime and memory requirements quadratic in the input sequence length. Ideally, we would like to go beyond the standard 2k sequence length limit to train models to understand books, high resolution images, and long-form videos. Just within the last year, there have been several language models with much longer context than before: GPT-4 (OpenAI, 2023) with context length 32k, MosaicML’s MPT with context length 65k , and Anthropic’s Claude with context length 100k. Emerging use cases such as long document querying and story writing have demonstrated a need for models with such long context. 
> 上下文长度：标准 Transformer (2k), GPT-4 (32k), MPT (65k), Claude (100k)

To reduce the computational requirement of attention on such long context, there have been numerous methods proposed to approximate attention (Kitaev et al., 2020; Roy et al., 2021; Wang et al., 2020; Katharopoulos et al., 2020; Choromanski et al., 2020; Beltagy et al., 2020; Zaheer et al., 2020; Chen et al., 2021). Though these methods have seen some use cases, as far as we know, most large-scale training runs still use standard attention. Motivated by this, Dao et al. (2022) proposed to reorder the attention computation and leverages classical techniques (tiling, recomputation) to significantly speed it up and reduce memory usage from quadratic to linear in sequence length. This yields $2–4\times$ wall-clock time speedup over optimized baselines, up to 10-20 × memory saving, with no approximation, and as a result FlashAttention has seen wide adoption in large-scale training and inference of Transformers. 
> FlashAttention 将内存需求降为线性

![[FlashAttention2-Fig6.png]]

![[FlashAttention2-Fig7.png]]

However, context length increases even more, FlashAttention is still not nearly as efficient as other primitives such as matrix-multiply (GEMM). In particular, while FlashAttention is already $2–4\times$ faster than a standard attention implementation, the forward pass only reaches $30.50\%$ of the theoretical maximum $\mathrm{FLOPs}/\mathrm{s}$ of the device (Fig. 6), while the backward pass is even more challenging, reaching only $25.35\%$ of maximum throughput on A100 GPU (Fig. 7). In contrast, optimized GEMM can reach up to $80–90\%$ of the theoretical maximum device throughput. Through careful profiling, we observe that FlashAttention still has suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. 
> 但 FlashAttention 尚未像 GEMM 这类原语算子一样高效，
> 例如 FlashAttention 前向仅达到 A100 理论最大 FLOPs/s 的 30.50%，反向仅达到 23.35%，而 GEMM 可以达到 80-90%
> 我们通过 profiling 发现 FlashAttention 在 GPU 上不同的 thread block 和 warps 之间的工作划分是次优的，导致 low-occupancy 或不必要的 shared memory 读写

Building on FlashAttention, we propose FlashAttention-2 with better parallelism and work partitioning to address these challenges. 

1. In Section 3.1, we tweak the algorithms to reduce the number of non-matmul FLOPs while not changing the output. While the non-matmul FLOPs only account for a small fraction of the total FLOPs, they take longer to perform as GPUs have specialized units for matrix multiply, and as a result the matmul throughput can be up to $16\times$ higher than non-matmul throughput. It is thus important to reduce non-matmul FLOPs and spend as much time as possible doing matmul FLOPs.

 2. We propose to parallelize both the forward pass and backward pass along the sequence length dimension, in addition to the batch and number of heads dimension. This increases occupancy (utilization of GPU resources) in the case where the sequences are long (and hence batch size is often small).

 3. Even within one block of attention computation, we partition the work between different warps of a thread block to reduce communication and shared memory reads/writes. 

> FlashAttention-2 基于 FlashAttention，具有更好的工作划分和并行性
> 具体为：
> 1. FlashAttention 的算法被微调，减少了 non-matmul FLOPs (理由：non-matmul FLOPs 仅占总 FLOPs 的一小部分，但 GPUs 有高度优化的 GEMM 算子，故 non-matmul 的吞吐量仅为 matmul 的 16分之一)
> 2. 在序列长度维度、batch 维度、头数量维度并行化前向和反向传播，这在长序列长度 (batch size 相应地小) 的情况下提高了 occupancy (GPU 资源的利用率)
> 3. 在单个 attention block 计算中经工作划分给 thread block 内不同的 warp，减少了 shared memory 的读写

In Section 4, we empirically validate that FlashAttention-2 yields significant speedup compared to even FlashAttention . Benchmarks on different settings (with or without causal mask, different head dimensions) show that FlashAttention-2 achieves around $2\times$ speedup over FlashAttention , reaching up to 73% of the theoretical max throughput in the forward pass, and up to 63% of the theoretical max throughput in the backward pass. During LLM inference, FlashAttention-2’s kernel is up to $7\times$ faster than the attention kernel from Faster Transformer. When used end-to-end to train GPT-style models, we reach training speed of up to 225 TFLOPs/s per A100 GPU. 
> 不同设定 (有无 causal mask、不同头维度) 下的 benchmark 表明 FlashAttention-2 速度为 FlashAttention 的两倍，前向传播下达到 73% 的理论最大吞吐，反向传播下达到 63%的理论最大吞吐
> 用于 LLM 推理时，FlashAttention 7 倍快于 Faster Transformer
> 端到端训练 GPT-style 模型时，A100下达到 225 TFLOPs/s

# 2 Background
We provide some background on the performance characteristics and execution model of GPUs. We also describe the standard implementation of attention, as well as FlashAttention. 

## 2.1 Hardware Characteristics
**GPU performance characteristics.** The GPU consists of compute elements (e.g., ﬂoating point arithmetic units) and a memory hierarchy. Most modern GPUs contain specialized units to accelerate matrix multiply in low-precision (e.g., Tensor Cores on Nvidia GPUs for FP16/BF16 matrix multiply). The memory hierarchy comprise of high bandwidth memory (HBM), and on-chip SRAM (aka shared memory). As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s (Jia et al., 2018; Jia and Van Sandt, 2021). As the L2 cache is not directly controllable by the programmer, we focus on the HBM and SRAM for the purpose of this discussion. 
> GPU 的 memory 层次包括 HBM 和 SRAM
> A100 HBM 大小为 40-80GB，带宽为 1.5-2.0TB/s，SRAM 大小为 192KB，108个 SM 上各自有一个，带宽为 19TB/s

**Execution Model.** GPUs have a massive number of threads to execute an operation (called a kernel). Threads are organized into thread blocks, which are scheduled to run on streaming multiprocessors (SMs). Within each thread blocks, threads are grouped into warps (a group of 32 threads). Threads within a warp can communicate by fast shuffle instructions or cooperate to perform matrix multiply. Warps within a thread block can communicate by reading from / writing to shared memory. Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM. 
> thread block 被调度于 SM 上运行，thread block 内的调度单位是 warp
> 同 warp 内的 thread 可以通过快速 shuffle 指令通讯，也可以通过向 shared memory 读写通讯；同 warp 内的 thread 可以协同执行矩阵乘法

## 2.2 Standard Attention Implementation
Given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ where $N$ is the sequence length and $d$ is the head dimension, we want to compute the attention output $\mathbf{O}\!\in\!\mathbb{R}^{N\times d}$ : 

$$
\mathbf{S}\!=\!\mathbf{Q}\mathbf{K}^{\top}\!\in\!\mathbb{R}^{N\times N},\quad\mathbf{P}\!=\!\operatorname{softmax}(\mathbf{S})\!\in\!\mathbb{R}^{N\times N},\quad\mathbf{O}\!=\!\mathbf{P}\mathbf{V}\!\in\!\mathbb{R}^{N\times d},
$$ 
where $\text{softmax}$ is applied row-wise. 

For multi-head attention (MHA), this same computation is performed in parallel across many heads, and parallel over the batch dimension (number of input sequences in a batch). 
> MHA 下，attention 计算在多个头中并行执行，而 MHA 在 batch 维度下并行执行 (为 batch 中的多个输入序列计算 MHA)

The backward pass of attention proceeds as follows. Let $\mathbf{dO}\in\mathbb{R}^{N\times d}$ be the gradient of $\mathbf O$ with respect to some loss function. Then by the chain rule (aka backpropagation):  

$$
\begin{array}{r l r l}&{\mathbf{d}\mathbf{V}\!=\!\mathbf{P}^{\top}\mathbf{d}\mathbf{O}\!\in\!\mathbb{R}^{N\times d}}&&{\mathbf{d}\mathbf{P}\!=\!\mathbf{d}\mathbf{O}\mathbf{V}^{\top}\!\in\!\mathbb{R}^{N\times N}}\\ &{\mathbf{d}\mathbf{S}\!=\!\mathrm{d}\mathrm{softmax}(\mathbf{d}\mathbf{P})\!\in\!\mathbb{R}^{N\times N}}&&{\mathbf{d}\mathbf{Q}\!=\!\mathbf{d}\mathbf{S}\mathbf{K}\!\in\!\mathbb{R}^{N\times d}}&&{\mathbf{d}\mathbf{K}\!=\!\mathbf{d}\mathbf{S}^{\top}\mathbf{Q}\!\in\!\mathbb{R}^{N\times d},}\end{array}
$$ 
where $\text{dsoftmax}$ is the gradient (backward pass) of softmax applied row-wise. One can work out that if $p=\operatorname{softmax}(s)$ for some vector $s$ and $p$ , then with output gradient $d p$ , the input gradient $\begin{array}{r}{d s\!=\!(\mathrm{diag}(p)\!-\!p p^{\top})d p}\end{array}$ . 

Standard attention implementation materialize the matrices $\mathbf S$ and $\mathbf{P}$ to HBM, which takes $O(N^{2})$ memory. Often $N\!\gg\!d$ (typically 𝑁 is on the order of 1k–8k and 𝑑 is around 64–128). The standard attention implementation (1) calls the matrix multiply (GEMM) subroutine to multiply $\begin{array}{r}{\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}}\end{array}$ , writes the result to HBM, then (2) loads $\mathbf S$ from HBM to compute softmax and write the result $\mathbf{P}$ to HBM, and finally (3) calls GEMM to get $\mathbf{O}\!=\!\mathbf{P}\mathbf{V}$ . As most of the operations are bounded by memory bandwidth, the large number of memory accesses translates to slow wall-clock time. Moreover, the required memory is $O(N^{2})$ due to having to materialize $\bf S$ and $\bf P$ . Moreover, one has to save ${\bf P}\!\in\!\mathbb{R}^{{N}\times N}$ for the backward pass to compute the gradients. 
> 标准 attention 实现需要将 $\mathbf {S, P} \in \mathbb R^{N\times N}$ 写入 HBM，占用 $O (N^2)$ 内存
> ( $N$ 的数量级一般在 1k-8k，$d$ 的数量级一般在 64-128，故 $N\gg d$)
> 标准 attention 的计算流程为：
> 1. GEMM 计算 $\mathbf {S = QK}^{\top}$，$\bf S$ 写回 HBM
> 2. load $\mathbf S$，计算 $\bf P$，$\bf P$ 写回 HBM
> 3. GEMM 计算 $\bf O = PV$
> 标准 attention 计算的劣势：
> 1. memory bound
> 2. 需要 $O (N^2)$ memory
> 3. 需要存储 $\mathbf P \in \mathbb R^{N\times N}$ 用于反向

## 2.3 FlashAttention 
To speed up attention on hardware accelerators such as GPU, (Dao et al., 2022) proposes an algorithm to reduce the memory reads/writes while maintaining the same output (without approximation). 

### 2.3.1 Forward pass
FlashAttention applies the classical technique of tiling to reduce memory IOs, by (1) loading blocks of inputs from HBM to SRAM, (2) computing attention with respect to that block, and then (3) updating the output without writing the large intermediate matrices $\bf S$ and $\mathbf{P}$ to HBM. As the softmax couples entire rows or blocks of row, online softmax (Milakov and Gimelshein, 2018; Rabe and Staats, 2021) can split the attention computation into blocks, and rescale the output of each block to finally get the right result (with no approximation). By significantly reducing the amount of memory read/writes, FlashAttention yields $2–4\times$ wall-clock speedup over optimized baseline attention implementations. 
> FlashAttention 减少了 memory IO，其流程为：
> 1. 将输入的 block 从 HBM load 到 SRAM
> 2. 计算 block attention
> 3. 不写回中间结果 $\bf {S, P}$，直接在片上更新 $\bf O$ 
> FlashAttention 利用了 online softmax 将 attention 计算划分为块，通过 rescale 保持 block attention 的计算结果是正确的
> FlashAttention 将计算加速了 2-4 倍

We describe the online softmax technique (Milakov and Gimelshein, 2018) and how it is used in attention (Rabe and Staats, 2021). For simplicity, consider just one row block of the attention matrix $\bf S$ , of the form $\left[\mathbf{S}^{(1)}\quad\mathbf{S}^{(2)}\right]$ for some matrices ${\mathbf{S}}^{(1)},\mathbf{S}^{(2)}\in{\mathbb{R}}^{B_{r}\times B_{c}}$ , where $B_{r}$ and $B_{c}$ are the row and column block sizes. We want to compute softmax of this row block and multiply with the value, of $\bf V$ the form $\begin{bmatrix}\mathbf V^{(1)} \\ \mathbf V^{(2)}\end{bmatrix}$ for some matrices $\mathbf{V}^{(1)},\mathbf{V}^{(2)}\in\mathbb{R}^{B_{c}\times d}$ . Standard softmax would compute: 

$$\begin{align*} m &= \operatorname*{max}(\mathrm{rowmax}(\mathbf{S}^{(1)}), \mathrm{rowmax}(\mathbf{S}^{(2)})) \in \mathbb{R}^{B_{r}} \\ 
\ell &= \mathrm{rowsum}(e^{\mathbf S^{(1)} - m}) + \mathrm{rowsum}(e^{\mathbf S^{(2)} - m}) \in \mathbb{R}^{B_{r}} \\ 
\mathbf{P} &= \left[\mathbf{P}^{(1)} \quad \mathbf{P}^{(2)}\right] = \mathrm{diag}(\ell)^{-1} \left[e^{\mathbf S^{(1)} - m} \quad e^{\mathbf S^{(2)} - m}\right] \in \mathbb{R}^{B_{r} \times 2B_{c}} \\ 
\mathbf{O} &= \left[\mathbf{P}^{(1)} \quad \mathbf{P}^{(2)}\right] \begin{bmatrix}\mathbf{V}^{(1)} \\ \mathbf V^{(2)}\end{bmatrix} = \mathrm{diag}(\ell)^{-1} \left(e^{\mathbf S^{(1)} - m} \mathbf{V}^{(1)} + e^{\mathbf S^{(2)} - m} \mathbf{V}^{(2)}\right) \in \mathbb{R}^{B_{r} \times d}. \end{align*}$$

Online softmax instead computes "local" softmax with respect to each block and rescale to get the right out at the end:

$$
\begin{align}
m^{(1)} & = \text{rowmax}(\mathbf S^{(1)})\in \mathbb R^{B_r}\\
\ell^{(1)} &= \text{rowsum}(e^{\mathbf S^{(1)}-m^{(1)}}) \in \mathbb R^{B_r}\\
\tilde {\mathbf P}^{(1)} &= \text{diag}(\ell^{(1)})^{-1} e^{\mathbf S^{(1)}- m^{(1)}} \in \mathbb R^{B_r \times B_c}\\
\mathbf O^{(1)}&=\tilde {\mathbf P}^{(1)}\mathbf V^{(1)} =\text{diag}(\ell^{(1)})^{-1}e^{\mathbf S^{(1)}- m^{(1)}}\mathbf V^{(1)} \in \mathbb R^{B_r \times d}\\\\
m^{(2)} & = \max(m^{(1)},\text{rowmax}(\mathbf S^{(2)})) = m\\
\ell^{(2)} &= e^{m^{(1)} - m^{(2)}}\ell^{(1)} + \text{rowsum}(e^{\mathbf S^{(2)}-m^{(2)}}) = \text{rowsum}(e^{\mathbf S^{(1)}-m}) +\text{rowsum}(e^{\mathbf S^{(2)}-m}) = \ell\\
\tilde {\mathbf P}^{(2)} &= \text{diag}(\ell^{(2)})^{-1} e^{\mathbf S^{(2)}- m^{(2)}} \in \mathbb R^{B_r \times B_c}\\
\mathbf O^{(2)}&=\text{diag}(\ell^{(1)}/\ell^{(2)})e^{m^{(1)}-m}\mathbf O^{(1)} + \tilde {\mathbf P}^{(2)}\mathbf V^{(2)} =
\text{diag}(\ell^{(2)})^{-1}e^{\mathbf S^{(1)}- m}\mathbf V^{(1)}+
\text{diag}(\ell^{(2)})^{-1}e^{\mathbf S^{(2)}- m}\mathbf V^{(1)}=\mathbf O
\end{align}
$$

> 常规的 softmax 耦合了输入矩阵的所有的列，online softmax 对其进行解耦合
> 考虑每一行，在列维度进行分块时，online softmax 计算每个块的局部 softmax，并随着列维度上块的遍历不断更新 softmax 统计量 $\ell, m$ (规范化指数和、最大值)，用更新的最大值重缩放之前块的局部 softmax 计算结果

We show how FlashAttention uses online softmax to enable tiling (Fig. 1) to reduce memory reads/writes. 
> FlashAttention 使用 online softmax 将 attention 计算 tile 为多个 block attention
> 其中，每个 block attention 更新 softmax 统计量，重缩放当前的累积 values 加权和，计算该 block 相关的 values 加权和并将其累积
> block attention 仅计算一块 $\mathbf S, \mathbf P$，并且用后即弃，不写入 HBM，故节约了读写中间结果 $\mathbf S, \mathbf P$ 需要的大量 HBM 访问，同时节约了 HBM 空间
> FlashAttention 减少了 $\mathbf {S, P}$ 的 HBM 读写次数，但实际上相应增加了 $\mathbf {Q, O}$ 的读写次数，但由于 $N\gg d$，故总体的 HBM 读写的次数是大幅减少的，因此 FlashAttention 本质上利用了 $\mathbf {QK}^\top$ 的低秩性质

![[FlashAttention2-Fig1.png]]

### 2.3.2 Backward pass
In the backward pass, by re-computing the values of the attention matrices $\bf S$ and $\mathbf{P}$ once blocks of inputs $\mathbf{Q},\mathbf{K},\mathbf{V}$ are already loaded to SRAM, FlashAttention avoids having to store large intermediate values. By not having to save the large matrices $\bf S$ and $\mathbf{P}$ of size $N{\times}N$ , FlashAttention yields $10–20\times$ memory saving depending on sequence length (memory required in linear in sequence length 𝑁 instead of quadratic). The backward pass also achieves 2-4 × wall-clock speedup due to reduce memory reads/writes. 
> FlashAttention 在前向中没有保存 $\bf S, P$，在反向过程会根据 SRAM 上的 $\bf Q, K, V$ 重新计算 $\bf S, P$，这将 memory 需求降为了 $O (N)$，同时也减少了 HBM 读写，原理和前向完全类似

The backward pass applies tiling to the equations in Section 2.2. Though the backward pass is simpler than the forward pass conceptually (there is no softmax rescaling), the implementation is significantly more involved. This is because there are more values to be kept in SRAM to perform 5 matrix multiples in the backward pass, compared to just 2 matrix multiples in the forward pass. 
> 反向没有重复的 rescale，故在概念上简单于前向
> 反向的实现则比前向显著复杂，前向仅需要完成两个矩阵乘 ($\mathbf {QK}^\top = \mathbf S$, $\mathbf {PV} = \mathbf O$)，反向需要完成五个矩阵乘 ($\mathbf {QK}^\top = \mathbf S$, $\mathbf {dV} = \mathbf P^\top \mathbf {dO}$, $\mathbf {dP} = \mathbf {dO}\mathbf V^\top$, $\mathbf {dQ} = \mathbf {dS}\mathbf K$, $\mathbf {dK} = \mathbf {dS}^{\top}\mathbf {dQ}$)，因而在 SRAM 中需要保存更多的矩阵

# 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning
We describe the FlashAttention-2 algorithm, which includes several tweaks to FlashAttention to reduce the number of non-matmul FLOPs. We then describe how to parallelize the computation on different thread blocks to make full use the GPU resources. Finally we describe we partition the work between different warps within one thread block to reduce the amount of shared memory access. These improvements lead to $2–3\times$ speedup as validated in Section 4. 

## 3.1 Algorithm
We tweak the algorithm from FlashAttention to reduce the number of non-matmul FLOPs. This is because modern GPUs have specialized compute units (e.g., Tensor Cores on Nvidia GPUs) that makes matmul much faster. As an example, the A100 GPU has a max theoretical throughput of 312 TFLOPs/s of FP16/BF16 matmul, but only 19.5 TFLOPs/s of non-matmul FP32. Another way to think about this is that each non-matmul FLOP is $16\times$ more expensive than a matmul FLOP. To maintain high throughput (e.g., more than 50% of the maximum theoretical TFLOPs/s), we want to spend as much time on matmul FLOPs as possible. 
> A100的 Tensor core 处理 FP16/BF16 的 matmul 运算的理论峰值吞吐量为 312 TFLOPs/s，而 Cuda core 处理 FP32的 non-matmul 运算的理论峰值吞吐量仅为 19.5 TFLOPs/s
> 可以理解为每个非 matmul 的浮点运算都16倍昂贵于 matmul 浮点运算，故需要提高 matmul 运算的比例

### 3.1.1 Forward pass
We revisit the online softmax trick as shown in Section 2.3 and make two minor tweaks to reduce non-matmul FLOPs: 

(1) We do not have to rescale both terms of the output update by $\mathrm{diag}(\ell^{(2)})^{-1}$ : 

$$
{\bf O}^{(2)}\!=\!\mathrm{diag}(\ell^{(1)}/\ell^{(2)})e^{m^{(1)}-m^{(2)}}{\bf O}^{(1)}\!+\!\mathrm{diag}(\ell^{(2)})^{-1}e^{{\bf S}^{(2)}-m^{(2)}}{\bf V}^{(2)}.
$$ 
We can instead maintain an “un-scaled” version of $\mathbf{O}^{(2)}$ and keep around the statistics $\ell^{(2)}$ : 

$$
\tilde {\mathbf O}^{(2)} = \text{diag}(\ell^{(1)})^{-1}e^{m^{(1)} - m^{(2)}}\mathbf O^{(1)} + e^{\mathbf S^{(2)}-m^{(2)}}\mathbf V^{(2)}
$$

Only at the every end of the loop do we scale the final $\tilde{\mathbf{O}}^{(\mathrm{last})}$ by $\mathrm{diag}(\ell^{(\mathrm{last})})^{-1}$ to get the right output. 

(2) We do not have to save both the max $m^{(j)}$ and the sum of exponentials $\ell^{(j)}$ for the backward pass. We only need to store the logsumexp $L^{(j)}{=}m^{(j)}{+}{\log}(\ell^{(j)})$ . 

> FlashAttention-2 对 FlashAttention 做的两项微调：
> 1. 不再用 $\text{diag}(\ell^{(i)})$ 在每一步缩放 $\mathbf O^{(i)}$，仅在累积到最后时用最终的 $\text{diag}(\ell )$ 进行缩放，此时 $\ell^{(i)}$ 在算法中仅需要保持更新，不参与计算。换句话说，算法中除了最后一步，对 values 的加权求和都不会对权重进行归一化 (但权重本身一定在 $[0,1]$ 之间，因此对数值稳定性不会有太大影响)
> 2. 将分别储存 $m^{(j)}, \ell^{(j)}$ (用于反向传播) 改为仅储存 $L^{(j)} = m^{(j)} + \log (\ell^{(j)})$

In the simple case of 2 blocks in Section 2.3, the online softmax trick now becomes:

$$\begin{align} 
m^{(1)} &= \mathrm{rowmax}(\mathbf{S}^{(1)}) \in \mathbb{R}^{B_{r}}\\
\ell^{(1)} &= \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m^{(1)}}) \in \mathbb{R}^{B_{r}} \\ 
\tilde {\mathbf{O}}^{(1)} &= e^{\mathbf{S}^{(1)} - m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R}^{B_{r} \times d}\\\\
m^{(2)} &= \operatorname*{max}(m^{(1)}, \mathrm{rowmax}(\mathbf{S}^{(2)})) = m \\
\ell^{(2)} &= e^{m^{(1)} - m^{(2)}} \ell^{(1)} + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m^{(2)}}) = \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m}) = \ell \\
\tilde{\mathbf{P}}^{(2)} &= \mathrm{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \\ 
\tilde{\mathbf{O}}^{(2)} &= \mathrm{diag}(e^{m^{(1)} - m^{(2)}}) \tilde{\mathbf{O}}^{(1)} + e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)} = e^{s^{(1)} - m} \mathbf{V}^{(1)} + e^{s^{(2)} - m} \mathbf{V}^{(2)} \\ 
\mathbf{O}^{(2)} &= \mathrm{diag}(\ell^{(2)})^{-1} \tilde{\mathbf{O}}^{(2)} = \mathbf{O}. 
\end{align}$$

We describe the full FlashAttention-2 forward pass in Algorithm 1. 

![[FlashAttention2-Algorithm 1.png]]

**Algorithm 1** FlashAttention-2 forward pass
**Require:** Matrices $\mathbf {Q,K,V}\in \mathbb R^{N\times d}$ in HBM, block sizes $B_c, B_r$.
  1: Divide $\mathbf Q$ into $T_r = \lceil \frac N {B_r} \rceil$ blocks $\mathbf Q_1, \dots ,\mathbf Q_{T_r}$ of size $B_r \times d$ each, and divide $\mathbf K, \mathbf V$ into $T_c = \lceil \frac N{B_c} \rceil$ blocks $\mathbf K_1, \dots, \mathbf K_{T_c}$ and $\mathbf V_1, \dots, \mathbf V_{T_c}$ of size $B_c\times d$ each.
> 划分 $\mathbf Q, \mathbf K, \mathbf V$，划分时保持嵌入维度 $d$ 不变，从序列长度的维度划分
> $\mathbf Q$ 划分单位为 $B_r \times d$，$\mathbf K, \mathbf V$ 划分单位为 $B_c\times d$
> 得到 $T_r$ 个 $\mathbf Q$ 块，得到 $T_c$ 个 $\mathbf K, \mathbf V$ 块

  2: Divide $\mathbf O$ into $T_r$ blocks $\mathbf O_i, \dots, \mathbf O_{T_r}$ of size $B_r \times d$ each, divide the logsumexp $L$ into $T_r$ blocks $L_i,\dots, L_{T_r}$ of size $B_r$ each.
> 划分 $\mathbf O$，划分时保持嵌入维度 $d$ 不变，从序列长度的维度划分
> $\mathbf O$ 划分单位为 $B_r \times d$
> 得到 $T_r$ 个 $\mathbf O$ 块
> 划分 $L$，从序列长度的维度划分
> $L$ 的划分单位为 $B_r$
> 得到 $T_r$ 个 $L$ 块

  3: **for** $1\le i \le T_r$ **do**
  4:     Load $\mathbf Q_i$ from HBM to on-chip SRAM.
  5:     On chip, initialize $\mathbf O_{i}^{(0)} = (0)_{B_r \times d}\in \mathbb R^{B_r \times d}$, $\ell^{(0)}_{i} = (0)_{B_r}\in \mathbb R^{B_r}$, $m_i^{(0)} = (\infty)_{B_r}\in \mathbb R^{B_r}$.
> 外层循环：
> 装载 $\mathbf Q$ 块到 SRAM
> 在片上初始化 $\mathbf O, \ell, m$ 块

  7:     **for** $1\le j \le T_c$ **do**
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

**Causal masking.** 
One common use case of attention is in auto-regressive language modeling, where we need to apply a causal mask to the attention matrix $\bf S$ (i.e., any entry $\mathbf{S}_{i j}$ with $j\!>\! i$ is set to $-\infty.$ ). 

1. As FlashAttention and FlashAttention -2 already operate by blocks, for any blocks where all the column indices are more than the row indices (approximately half of the blocks for large sequence length), we can skip the computation of that block. This leads to around $1.7{\cdot}1.8\times$ speedup compared to attention without the causal mask. 

2. We do not need to apply the causal mask for blocks whose row indices are guaranteed to be strictly less than the column indices. This means that for each row, we only need apply causal mask to 1 block (assuming square block). 

Correctness, runtime, and memory requirement. As with FlashAttention , Algorithm 1 returns the correct output $\mathbf{O}\!=\!\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\intercal})\mathbf{V}$ (with no approximation), using $O (N^{2}d)$ FLOPs and requires $O (N)$ additional memory beyond inputs and output (to store the logsumexp $L$ ). The proof is almost the same as the proof of Dao et al. (2022, Theorem 1), so we omit it here. 

### 3.1.2 Backward pass
The backward pass of FlashAttention -2 is almost the same as that of FlashAttention . We make a minor tweak to only use the row-wise logsumexp $L$ instead of both the row-wise max and row-wise sum of exponentials in the softmax. We include the backward pass description in Algorithm 2 for completeness. 

Multi-query attention and grouped-query attention. Multi-query attention (MQA) (Shazeer, 2019) and grouped-query attention (GQA) (Ainslie et al., 2023) are variants of attention where multiple heads of query attend to the same head of key and value, in order to reduce the size of KV cache during inference. Instead of having to duplicate the key and value heads for the computation, we implicitly manipulate the indices into the head to perform the same computation. In the backward pass, we need to sum the gradients dK and dV across different heads that were implicitly duplicated. 

# 3.2 P ARALLELISM 

The first version of FlashAttention parallelizes over batch size and number of heads. We use 1 thread block to process one attention head, and there are overall batch size $\cdot^{\bullet}$ number of heads thread blocks. Each thread block is scheduled to run on a streaming multiprocessor (SM), and there are 108 of these SMs on an A100 GPU for example. This scheduling is efficient when this number is large $(\mathrm{day}\geq80)$ ), since we can effectively use almost all of the compute resources on the GPU. 

In the case of long sequences (which usually means small batch sizes or small number of heads), to make better use of the multiprocessors on the GPU, we now additionally parallelize over the sequence length dimension. This results in significant speedup for this regime. 

Forward pass. We see that the outer loop (over sequence length) is embarrassingly parallel, and we schedule them on different thread blocks that do not need to communicate with each other. We also parallelize over the batch dimension and number of heads dimension, as done in FlashAttention . The increased parallelism over sequence length helps improve occupancy (fraction of GPU resources being used) when the batch size and number of heads are small, leading to speedup in this case. 

Backward pass. Notice that the only shared computation between different column blocks is in update dQ in Algorithm 2, where we need to load $\mathbf{d}\mathbf{Q}_{i}$ from HBM to SRAM, then on chip, update $\mathbf{d}\mathbf{Q}_{i}\longleftarrow\mathbf{d}\mathbf{Q}_{i}\!+\!\mathbf{d}\mathbf{S}_{i}^{(j)}\mathbf{K}_{j}$ , and write back to HBM. We thus parallelize over the sequence length dimension as well, and schedule 1 thread block for each column block of the backward pass. We use atomic adds to communicate between different thread blocks to update dQ . 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/1a2571b795cfc527f7e7c1d158ef7e22dfc7ff76c5b15f08269b7010f1f37bd1.jpg) 
We describe the parallelization scheme in Fig. 2. 

Figure 2: In the forward pass (left), we parallelize the workers (thread blocks) where each worker takes care of a block of rows of the attention matrix. In the backward pass (right), each worker takes care of a block of columns of the attention matrix. 

Decoding. During LLM inference, most of the time is spent on iterative decoding, where one token is predicted at a time. The bottleneck for the attention operation during decoding is different from that during training or prefill (prompt processing), because the query length is very short (often query length is 1 since only the new extra token is attending to all the previous tokens, stored in the KV cache). As a result, the bottleneck is no longer the read/write of intermediate matrices (the scores $\mathbf{Q}\mathbf{K}^{\top}$ and attention probabilities softmax ( QK ⊤ ) ). Instead, the bottleneck is to load the KV cache as quickly as possible. 
To accommodate this setting, we split the KV cache loading among different thread blocks, to increase occupancy and saturate the HBM bandwidth. However, since the thread blocks cannot easily communicate with each other, we write intermediate results to HBM, then call a separate kernel to reduce the results and produce final output. 

# 3.3 W ORK P ARTITIONING B ETWEEN W ARPS 

As Section 3.2 describe how we schedule thread blocks, even within each thread block, we also have to decide how to partition the work between different warps. We typically use 4 or 8 warps per thread block, and the partitioning is described in Fig. 3. 

Forward pass. For each block, FlashAttention splits $\mathbf{K}$ and $\mathbf{V}$ across 4 warps while keeping $\mathbf{Q}$ accessible by all warps. Each warp multiplies to get a slice of $\mathbf{Q}\mathbf{K}^{\top}$ , then they need to multiply with a slice of $\mathbf{V}$ and communicate to add up the result. This is referred to as the “split-K” scheme. However, this is inefficient since all warps need to write their intermediate results out to shared memory, synchronize, then add up the intermediate results. These shared memory reads/writes slow down the forward pass in FlashAttention . 

In FlashAttention -2, we instead split $\mathbf{Q}$ across 4 warps while keeping $\mathbf{K}$ and $\mathbf{V}$ accessible by all warps. After each warp performs matrix multiply to get a slice of $\mathbf{Q}\mathbf{K}^{\top}$ , they just need to multiply with their shared slice of V to get their corresponding slice of the output. There is no need for communication between warps. The reduction in shared memory reads/writes yields speedup (Section 4). 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/0d812ae6906a74cbf83469a8f2de51c17a2da9bff18f01fda4733814b1600665.jpg) 
Figure 3: Work partitioning between different warps in the forward pass 

Backward pass. Similarly for the backward pass, we choose to partition the warps to avoid the “split-K” scheme. However, it still requires some synchronization due to the more complicated dependency between all the different inputs and gradients Q , K , V , O , dO , dQ , dK , dV . Nevertheless, avoiding “split-K” reduces shared memory reads/writes and again yields speedup (Section 4). 

Tuning block sizes Increasing block sizes generally reduces shared memory loads/stores, but increases the number of registers required and the total amount of shared memory. Past a certain block size, register spilling causes significant slowdown, or the amount of shared memory required is larger than what the GPU has available, and the kernel cannot run at all. Typically we choose blocks of size $\{64{,}128\}{\times}\{64{,}128\}$ , depending on the head dimension $d$ and the device shared memory size. 

We manually tune for each head dimensions since there are essentially only 4 choices for block sizes, but this could benefit from auto-tuning to avoid this manual labor. We leave this to future work. 

# 4 E MPIRICAL V ALIDATION 

We evaluate the impact of using FlashAttention -2 to train Transformer models. 

• Benchmarking attention. We measure the runtime of FlashAttention -2 across different sequence lengths and compare it to a standard implementation in PyTorch, FlashAttention , and F LASH A TTENTI ton. We confirm that FlashAttention -2 is $1.7{-}3.0\times$ faster than FlashAttention , 1.3-2.5 × faster than FlashAttention in Triton, and 3-10 × faste n a standard attention implementation. FlashAttention -2 reaches up to 230 TFLOPs/s, 73% of the theoretical maximum TFLOPs/s on A100 GPUs. 

• End-to-end training speed When used end-to-end to train GPT-style models of size 1.3B and 2.7B on sequence lengths either 2k or 8k, FlashAttention -2 yields up to $1.3\times$ speedup compared to FlashAttention and $2.8\times$ speedup compar a baseline without FlashAttention . FlashAttention -2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization) per A100 GPU. 
# 4.1 B ENCHMARKING A TTENTION F OR T RAINING 

We measure the runtime of different attention methods on an A100 80GB SXM4 GPU for different settings (without / with causal mask, head dimension 64 or 128). We report the results in Fig. 4, Fig. 6 and Fig. 7, showing that FlashAttention -2 is around $2\times$ faster than FlashAttention and FlashAttention in xformers (the “cutlass” implementation). FlashAttention -2 is around $1.3{\cdot}1.5\times$ faster than FlashAttention in Triton in the forward pass and around $2\times$ faster in the backward pass. Compared to a standard attention implementation in PyTorch, FlashAttention -2 can be up to $10\times$ faster. 

Benchmark setting: we vary the sequence length from 512, 1k, ..., 16k, and set batch size so that the total number of tokens is 16k. We set hidden dimension to 2048, and head dimension to be either 64 or 128 (i.e., 32 heads or 16 heads). To calculate the FLOPs of the forward pass, we use: 

4 · seqlen 2 · head dimension $\cdot^{\bullet}$ number of heads . 

With causal mask, we divide this number by 2 to account for the fact that approximately only half of the entries are calculated. To get the FLOPs of the backward pass, we multiply the forward pass FLOPs by 2.5 (since there are 2 matmuls in the forward pass and 5 matmuls in the backward pass, due to recomputation). 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/df731d02fc830251617ceeb6cdb9495b88010377510f12f3054d3b439aaada9d.jpg) 

Just running the same implementation on H100 GPUs (using no special instructions to make use of new features such as TMA and 4th-gen Tensor Cores), we obtain up to 335 TFLOPs/s (Fig. 8). We expect that by using new instructions, we can obtain another $1.5\mathbf{x}–2\mathbf{x}$ speedup on H100 GPUs. We leave that to future work. 

# 4.2 B ENCHMARKING A TTENTION F OR I NFERENCE 

We benchmark the attention kernel during decoding for the case of multi-query attention, where the bot- tleneck is loading the KV cache. In Fig. 5, we see that the attention kernel from FlashAttention -2 is up to $28\times$ faster than a naive implementation in PyTorch, and up to $7\times$ faster than an implementation from Faster Transformer. This is thanks to better work partitioning where multiple thread blocks are loading the KV cache at the same time to saturate HBM bandwidth. 
![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/764256b172e075c48d8fff865760692c7818486d9a9570ff5f185b83c3416e23.jpg) 

Figure 5: Attention decoding time on A100 80GB, with hidden dimension 2048 and multi-query atten- tion e attention kernel from FlashAttention -2 is up to $7\times$ faster than that of Faster Transformer and 28 $28\times$ faster than a naive implementation in PyTorch. 

# 4.3 E ND - TO - END P ERFORMANCE 

We measure the training throughput of GPT-style models with either 1.3B or 2.7B parameters, on $8{\times}\mathrm{A}100$ 80GB SXM4. As shown in Table 1 SH A TTENTION -2 yields $2.8\times$ speedup compared to a baseline without FlashAttention and 1.3 $1.3\times$ × speedup compared to FlashAttention , reaching up to 225 TFLOPs/s per A100 GPU. 

Note that we calculate the FLOPs by the formula, following Megatron-LM (Shoeybi et al., 2019) (and many other papers and libraries): 

6 · seqlen · number of params $+12$ · number of layers · hidden dim · seqlen 2 . 

The first term accounts for the FLOPs due to weight–input multiplication, and the second term accounts for the FLOPs due to attention. However, one can argue that the second term should be halved, as with causal mask we only need to compute approximately half the number of elements in attention. We choose to follow the formula from the literature (without dividing the attention FLOPs by 2) for consistency. 

Table 1: Training speed (TFLO GPU) of GPT-style models on $8{\times}\mathrm{A}100$ GPUs. FlashAttention - 2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization). We compare against a baseline running without FlashAttention . 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/e27f364eb48d9a351d38ecdfcfca0cf8e337dc6af52a7284ad404ffae7987f3f.jpg) 

# 5 D ISCUSSION AND F UTURE D IRECTIONS 

FlashAttention -2 is $2\times$ faster than FlashAttention , which ans that we can train models with 16k longer context for the same price as previously training a 8k context model, for the same number of tokens. We are excited about how this can be used to understand long books and reports, high resolution images, audio and video. FlashAttention -2 will also speed up training, finetuning, and inference of existing models. 

In the near future, we plan to collaborate with researchers and engineers to make FlashAttention widely applicable in different kinds of devices (e.g., H100 GPUs, AMD GPUs), as well as new data types such as FP8. As an immediate next step, we plan to optimize FlashAttention-2 for H100 GPUs to use new hardware features (TMA, 4th-gen Tensor Cores, fp8). Combining the low-level optimizations in FlashAttention-2 with high-level algorithmic changes (e.g., local, dilated, block-sparse attention) could allow us to train AI models with much longer context. We are also excited to work with compiler researchers to make these optimization techniques easily programmable. 
A CKNOWLEDGMENTS 

We thank Phil Tillet and Daniel Haziza, who have implemented versions of FlashAttention in Triton (Tillet et al., 2019) and the xformers library (Lefaudeux et al., 2022). FlashAttention -2 was motivated by exchange of ideas between different ways that attention could be implemented. We are grateful to the Nvidia CUTLASS team (especially Vijay Thakkar, Cris Cecka, Haicheng Wu, and Andrew Kerr) for their CUTLASS library, in particular the CUTLASS 3. x release, which provides clean abstractions and powerful building blocks for the implementation of FlashAttention -2. We thank Driss Guessous for integrating FlashAttention to PyTorch. FlashAttention -2 has benefited from helpful discussions with Phil Wang, Markus Rabe, James Bradbury, Young-Jun Ko, Julien Launay, Daniel Hesslow, Michaël Benesty, Horace He, Ashish Vaswani, and Erich Elsen. Thanks to Stanford CRFM and Stanford NLP for the compute support. We thank Dan Fu and Christopher Ré for their collaboration, constructive feedback, and constant encouragement on this line of work of designing hardware-efficient algorithms. We thank Albert Gu and Beidi Chen for their helpful suggestions on early drafts of this paper. 

# R EFERENCES 

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv: 2305.13245 , 2023. Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv: 2004.05150 , 2020. Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré. Scatterbrain: Unifying sparse and low-rank attention. In Advances in Neural Information Processing Systems (NeurIPS) , 2021. Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In International Conference on Learning Representations (ICLR) , 2020. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems , 2022. Zhe Jia and Peter Van Sandt. Dissecting the Ampere GPU architecture via micro benchmarking. GPU Technology Conference, 2021. Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza. Dissecting the nvidia Volta GPU architecture via micro benchmarking. arXiv preprint arXiv: 1804.06826 , 2018. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning , pages 5156–5165. PMLR, 2020. Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In The International Conference on Machine Learning (ICML) , 2020. Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. https://github. com/facebook research/xformers , 2022. Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. arXiv preprint arXiv: 1805.02867 , 2018. OpenAI. Gpt-4 technical report. ArXiv , abs/2303.08774, 2023. 

Markus N Rabe and Charles Staats. Self-attention does not need memory. arXiv preprint arXiv: 2112.05682 , 2021. 
Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics , 9:53–68, 2021. Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv: 1911.02150 , 2019. Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv: 1909.08053 , 2019. Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages , pages 10–19, 2019. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017. Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv: 2006.04768 , 2020. Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems , 33, 2020. 
![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/c88e132808d7d8735d3f71e85f641f961b594fabb6e5b0e01a13c0c7bc498e5b.jpg) 

B B ENCHMARKING A TTENTION ON A100 AND H100 
![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/da07f1de71a48f73dce13fa52763cf975a6080249f97593e274d969230e6ba52.jpg) 
(a) Without causal mask, head dimension 64 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/9a1b0f81f4a9dfe81cedbd6d2774d8d731a3c2dfa335efd5258281b75a2e266f.jpg) 
(c) With causal mask, head dimension 64 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/1fe84e601aa41556ce0c2a9afdaf2d35d7f2935fe6db80f34d7a795d40fdced6.jpg) 
(a) Without causal mask, head dimension 64 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/b9aedb923fc371e587d9bf04e704d3cf55742ff1354ab58695b6b0a133c58d9b.jpg) 
(c) With causal mask, head dimension 64 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/43017a99d3bd4be0920abb6434e53cefb728e6cecf477288342a95bd0c8c7d5a.jpg) 
(b) Without causal mask, head dimension 128 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/0c0a3b071be6aef435aa6398a893e549de0f9fc447d4bb7ea4a28b290b48a7b1.jpg) 
(d) With causal mask, head dimension 128 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/6d4d59ff96591187a2b2d11392cd0be238db3a91649b6e979e8a43737940f21d.jpg) 
(b) Without causal mask, head dimension 128 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/d7defaeb59214e18501604a94cb3479ce74f78c01ac8282ac5c3ac4578210c3e.jpg) 
(d) With causal mask, head dimension 128 
![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/07c34ed1c43de023eb676c885d13a940d51c18cbe401e21e521cb0455711bc66.jpg) 
(a) Without causal mask, head dimension 64 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/becf30417d036fbd22d7b8ea62ff4aef1dbc1dbdc2c9e2607a851ebb51ddb5a3.jpg) 
(c) With causal mask, head dimension 64 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/2d91ab1dfd89e21f84e8b1a53af696feed5341addf288d6aa70f50ec773ea4f9.jpg) 
(b) Without causal mask, head dimension 128 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/02d75799-1854-43ad-ab99-f2dda6088b50.pdf/fcdb03e4f306f8b69366b5ace9403fb825565be0678cdbcbc34f7c00ed0924dd.jpg) 
(d) With causal mask, head dimension 128 
