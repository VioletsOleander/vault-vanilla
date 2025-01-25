# 2024
## July
### Week 4
Date: 2024.7.29-2024.8.5

\[Book\]
- [[Programming Massively Parallel Processors A Hands-on Approach-2023|Programming Massively Parallel Processor A Hands-on-Approach]]: CH2-CH6.3, CH10
    Derived Ideas:
        1. Tiling: 搬运数据 from Global Memory to Shared Memory
        2. Coalescing: 利用 DRAM burst 优化 Tiling 过程中对 Global Memory 的访问次数 
        3. Coarsening (Optional)
- [[CUDA C++ Programming Guide v12.5-2024|CUDA C++ Programming Guide v12.5]]: CH5-CH11, CH19
    Derived Ideas:
        1. Avoid Bank Conflict: 数据访问对齐32bit 的 Bank Size
        2. Occupancy Calculator: 利用工具计算一下合适的 Blocksize 和 Gridsize
        3. Coalescing: Warp 对 Shared Memory 的访问同样可以进行合并优化
        4. `memcpy_async()` (Questioned): 异步访问，访存与计算流水线
- [[Managing Projects with GNU Make-2011|Managing Projects with GNU Make]]: CH1-CH2.7

\[Doc\]
- [[NVIDIA Nsight Compute]]: CH2

\[Blog\]
- [CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275): 0%-50%
    Derived Ideas:
        1. Thread Tile: 改变 Thread Tile 内矩阵的运算顺序，利用 Register 减少对 global memory 的访问次数；其中 Thread tile 的长宽 $M_{frag},N_{frag}$ 的选取与线程内 FFMA 指令对非 FFMA 指令如 LDS 指令的延迟覆盖是相关的

## August
### Week 1
Date: 2024.8.5-2024.8.12

\[Book\]
- [[Parallel Thread Execution ISA v8.5-2024|PTX ISA v8.5]]: All

\[Doc\]
- [[CUDA-GDB v12.6]]: CH1-CH8

\[Blog\] 
- [CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275): 0%-50%
    Derived Ideas:
        1. Arithmetic Intensity: 通过衡量计算方式的算数密度，将其乘上相应带宽，可以得到理论的 FLOPS 上限
        2. Thread Block Tile: 减少 Global Memory 读取
        3. Thread Tile & Warp Tile: 改变矩阵乘法顺序，调整 Tile 形状，提高 Arithmetic Intensity，使 FMA 可以掩盖 LDS 的延迟
        4. Pipeline: 由于改变矩阵乘法顺序增大了单线程的寄存器使用量，导致 Warp 数量降低，进一步导致 Occupancy 降低，因此考虑流水并行 Global Memory to Shared Memory、Shared Memory to Register、Computation in Register 这三个操作，提高 Warp 的指令并行度，以提高硬件占用率
- [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370): All
    Derived Ideas:
        1. Coarsening: 一个线程计算 $4\times 4$ 的结果，提高线程的算数密度
        2. `LDS.128`: 读取 `float4` 向量类型，减少 Shared Memory 访问
- [cuda 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788)
    Derived Ideas:
        1. Align: 令 Shared Memory 内数据地址对齐
- [CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://zhuanlan.zhihu.com/p/518857175)

\[Code\]
- CUDA GEMM Optimization Project
    `matmul_v0.cu` : naive implementation

### Week 2
Date: 2024.8.12-2024.8.19

\[Code\]
- CUDA GEMM Optimization Project
    `matmul_v1.cu` - `matmul_v7.cu`
        `matmul_v1.cu` : block tiled implementation
        `matmul_v5.cu` : block tiled and thread tiled implementation
        `matmul_v6/v7.cu` : block/thread tiled and pipelined implementation
    `matmul_t_v0.cu` - `matmul_t_v2.cu` 
        `matmul_t_v0/v1.cu` : block tiled and warp tiled implementation
        `matmul_t_v2.cu` : bank conflict partially solved implementation

### Week 3
Date: 2024.8.19-2024.8.26

\[Code\]
- CUDA GEMM Optimization Project
    `matmul_t_v3.cu` - `matmul_t_v4.cu`
        `matmul_t_v3.cu` : swizzled implementation
        `matmul_t_v4.cu` : adjusted the tile size
### Week 4
Date: 2024.8.26-2024.9.9

\[Paper\]
- [[paper-notes/llm/A Survey of Large Language Models v13-2023|2023-A Survey of Large Language Models v13]]: Sec1-Sec5

\[Book\]
- [[Pro Git]]: CH 7.1
- [[Mastering CMake]]: CH1-CH7

## September
### Week 1
Date: 2024.9.2-2024.9.9

\[Book\] 
- [[Mastering CMake]]: CH8-CH13、CH14 (CMake Tutorial)

### Week 2
Date: 2024.9.9-2024.9.16

\[Paper\]
- [[paper-notes/llm/A Survey of Large Language Models v13-2023|2023-A Survey of Large Language Models v13]]: Sec6
    Sec6-Utilization
        Prompt tricks: (input-output) pair, (input-reasoning step-output) triplet, plan
- [[Are Emergent Abilities of Large Language Models a Mirage-2023-NeurIPS|2023-NeurIPS-Are Emergent Abilities of Large Language Models a Mirage?]]: All

\[Book\]
- [[book-notes/Introductory Combinatorics-2009|Introductory Combinatorics]]: CH1
    CH1-What is Combinatorics
        Combinatorics: existence, enumeration, analysis, optimization of discrete/finite structures
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH2
    CH2-Foundations
        Basic knowledges: Conditional Independence, MAP query, Conditional density function, graphs

\[Doc\]
- [[Intel NPU Acceleration Library Documentation v1.3.0]]
- [[The Python Tutorial]]: CH1-CH16

### Week 3
Date: 2024.9.16-2024.9.23

\[Book\]
- [[book-notes/Introductory Combinatorics-2009|Introductory Combinatorics]]: CH2
    CH2-Permutations and Combinations
        Permutation/Combination of Sets (combination = permutation + division), Permutation/Combination of Multisets (permutation of sets + division/solutions of linear equation) , classical probability
- [[book-notes/Convex Optimization|Convex Optimization]]: CH2-CH2.5
    CH2-Convex Sets
        Lots of definitions: convex combination, affine combination, some typical convex sets, operations that preserve convexity, supporting/separating hyperplane
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH3-CH3.3
    CH3-The Bayesian Network Representation
        Bayesian Network: Express conditional independencies in joint probability in a graph semantics, factorizing the joint probability into a product of CPDs according to the graph structure
- [[A Tour of C++]]: CH1-CH1.7

### Week 4
Date: 2024.9.23-2024.9.30

\[Paper\]
- [[paper-notes/llm/A Survey of Large Language Models v13-2023|2023-A Survey of Large Language Models v13]]: Sec7
    Sec7-Capacity and Evaluation
        LLM abilities: 1. basic ability: language generation (including code), knowledge utilization (e.g. knowledge-intensive QA) , complex reasoning (e.g. math) ; 2. advanced ability: human alignment, interaction with external environment (e.g. generate proper action plan for embodied AI), tool manipulate (e.g. call proper API according to tasks); introduction to some benchmarks

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH5
    CH5-Local Probabilistic Models
        Compact CPD representation: Utilize context-specific independence to compactly represent CPD; Independent causal influence model: noisy-or model, BN2O model, generalized linear model (scores are linear to all parent variables), conditional linear gaussian model ( induces a joint distribution that has the form of a mixture of Gaussians)
- [[A Tour of C++]]: CH1.7-CH3.5

## October
### Week 1
Date: 2024.9.30-2024.10.7

\[Paper\]
- [[paper-notes/llm/A Survey of Large Language Models v13-2023|2023-A Survey of Large Language Models v13]]: CH8-CH9
    CH8-Applicatoin
        LLM application in various tasks
    CH9-Conclusion and future directions
- [[Importance Sampling A Review-2010|2010-Importance Sampling: A Review]]: All
    IS is all about variance reduction for Monte Carlo approximation;
    Adaptive parametric Importance Sampling: $q (x)$ be defined as a multivariate normal or student distribution, then optimizing a variation correlated metric to derive an optimal parameter setting for that distribution;
    Sequential Importance Sampling: Chain decompose $p (x)$, and chain construct $q (x)$;
    Anneal Importance Sampling: Sequentially approximate $p (x)$, much like diffusion;

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH6-CH6.2
    CH6-Template-based Representations
        temporal models; Markov assumption + 2-TBN = DBN; DBN usually be modeled as state-observation model (the state and observation are considered separately; observation doesn't affect the state), two examples: HMM, linear dynamic system (all the dependencies are linear Gaussian)
- [[book-notes/面向计算机科学的组合数学|面向计算机科学的组合数学]]: CH1.7
    CH1-排列组合
        CH1.7-生成全排列
            中介数和排列之间的一一对应关系
- [[A Tour of C++]]: CH3.5-CH5

### Week 2
Date: 2024.10.7-2024.10.14

\[Paper\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: Sec0-Sec3.1
    Sec0-Abstract
    Sec1-Introduction
    Sec2-Background
    Sec3-FlashAttention: Algorithm, Analysis, and Extensions
        Sec3.1-An Efficient Attention Algorithm with Tiling and Recomputation
            Algorithm 1 is basically a tiled implementation of attention calculation. What makes algorithm 1 looks not so intuitive is the repetitive rescaling of softmax factor, whose aim is to stabilize the computation. In algorithm 1, each query's attention result is accumulated gradually by the outer loop, and the already accumulated partial attention result's weights for the corresponding value is dynamically updated/changed by the outer loop.

\[Book\]
- [[A Tour of C++]]: CH5
- [[book-notes/面向计算机科学的组合数学|面向计算机科学的组合数学]]: CH2.1-CH2.3
    CH2-鸽巢原理
        鸽巢原理仅解决存在性问题
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH4-CH4.3.1
    CH4-Undirected Graphical Models
        Markov Network's parameterization: the idea was derived from statistical physics, which is pretty intuitive by using factor to represent two variables' interaction/affinity, and using a normalized product of factors to represent a joint probability (Gibbs distribution) to describe the probability of particular configuration; separation criterion in Markov network is sound and weakly complete (sound: independence holds in network --> independence holds in all distribution factorizing over network; weakly complete: independence does not hold in network --> independence does not hold in some distribution factorizing over network)

\[Doc\]
- [[doc-notes/python/packages/ultralytics]] : Quickstart, Usage(Python usage, Callbacks, Configuration, Simple Utilities, Advanced Customization)
    Brief Introduction to YOLO model's python API, which is pretty simple

### Week 3
Date: 2024.10.14-2024.10.21

\[Paper\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: Sec3.1-Sec5
    Sec3-FlashAttention: Algorithm, Analysis and Extensions
        Sec3.1-IO Analysis
            The IO complexity of FlashAttention is $\Theta(N^2d^2M^{-1})$ while the IO complexity of the standard attention computation is $\Theta (Nd + N^2)$. The main difference is in $M$ and $N^2$. Standard attention computation does not use SRAM at all, all memory accesses are global memory access, in which the process of fetching "weight matrix" $P\in \mathbb R^{N\times N}$ contributes most of the IO complexity. FlashAttention utilized SRAM, and do not store "weight matrix" into DRAM, but keep a block of it on chip the entire time, thus effectively reduced the IO complexity.
        Sec3.2-Block sparse Flash-Attention
            The main difference between FlashAttention is that the range of "attention" is restricted, thereby the computation and memory accesses is reduced by skipping the masked entries.
    Sec4-Experiments
        FlashAttention trains faster; FlashAttention trains more memory-efficient (linear), thus allowing longer context window in training. The reason for that is FlashAttention do not compute the entire "weight matrix" $P\in \mathbb R^{N\times N}$ one time, but do a two level loop, compute one row of $P$ each time. The FLOP is actually increased, but the memory usage is restricted to $O (N)$ instead of $O (N^2)$ and the additional computation time brought by the increased FLOP is eliminated by the time reduced by less DRAM accesses.
- [[Spatial Interaction and the Statistical Analysis of Lattice Systems-1974|1974-Spatial Interaction and the Statistical Analysis of Lattice Systems]]: Sec0-Sec2
    Sec0-Summary
        This paper proposed an alternative proof of HC theorem, thereby reinforcing the importance of conditional probability models over joint probability models for modeling spatial interaction.
    Sec1-Introduction
    Sec2-Markov Fields and The Hammersley-Clifford Theorem
        For positive distribution, conditional probability can be used to deduce the overall joint probability. This is made possible by HC theorem.

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH4.3.1-CH4.4.2
    CH4-Undirected Graphical Models
        CH4.3-Markov Network Independencies: 
            Markov network encodes three types of independence: pairwise independence, local independence (Markov blanket), global independence (d-separation). For positive distribution, they are equivalent. For non-positive distribution (those with deterministic relationships), they are not equivalent. This is because the semantics of Markov network is not enough to convey deterministic relationships. By HC theorem, $P$ factorizes over Markov network $\mathcal H$ is equivalent to $P$ satisfies the three types of independence encoded by $\mathcal H$.
- [[book-notes/面向计算机科学的组合数学|面向计算机科学的组合数学]]: CH3-CH3.3
    CH3-母函数
        使用幂级数表示数列（数列由幂级数的系数构造）
- [[A Tour of C++]]: CH6

\[Doc\]
- [[Pytorch 2.x]]: CH0
    CH0-General Introduction
        `torch.compile` : TorchDynamo --> FX Graph in Torch IR --> AOTAutograd --> FX graph in Aten/Prims IR --> TorchInductor --> Triton code/OpenMP code...
- [[doc-notes/triton/Getting Started|Triton: Tutorials]]: Vector Addition, Fused Softmax
    Triton is basically simplified CUDA in python, the general idea about parallel computing is similar. The most advantageous perspective about Triton is that it encapsulates all the complicated memory address mapping work into a single api `tl.load` . Memory address mapping work is the most difficult part of writing CUDA code.

### Week 4
Date: 2024.10.21-2024.10.28

\[Paper\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: SecA-SecE
    SecA-Related Work
    SecB-Algorithm Details
        Memory-efficient forward/backward pass: using for-loop to avoid storing $O(N^2)$ intermediate matrix; FlashAttention backward pass: In implementation, the backward algorithm of FlashAttention is actually simpler than the forward algorithm, because it's just about tiled matrix multiplication without bothering softmax rescaling
    SecC-Proofs
        just counting, nothing special
    SecE-Extension Details
        block-sparse implementation is just skipping masked block, nothing special
    SecF-Full Experimental Results
- [[Spatial Interaction and the Statistical Analysis of Lattice Systems-1974|1974-Spatial Interaction and the Statistical Analysis of Lattice Systems]]: Sec3
    Sec3-Markov Fields and the Hammersley-Clifford Theorem
        define ground state -> define Q function -> expand Q function -> proof the terms in Q function (G function) are only not null when their relating variables form a clique

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH4.5
    CH4.5-Bayesian Networks and Markov Networks
        chordal graph can be represented by either structure without loss of information 

## November
### Week 1
Date: 2024.10.28-2024.11.4

\[Paper\]
- [[FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning-2024-ICLR|2024-ICLR-FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning]]
    FlashAttention-2: 
    (1) tweak the algorithm, reducing the non-matmul op: remove the rescale of softmax weights in each inner loop, only do it in the end of inner loop
    (2) parallize in thread blocks to improve occupancy: exchange the inner loop and outer loop,  which makes each iteration in outer loop independent of each other, therefore parallelize them by assigning $\mathbf {O}$ blocks to thread blocks
    (3) distribute the work between warps to reduce shared memory communication: divide $\mathbf Q$ block to warps and keep $\mathbf {K, V}$ blocks intact, the idea is similar to exchanging outer loop and inner loop, which makes the $\mathbf O$ blocks the warp responsible for be independent of each other, thus primarily reducing the shared memory reads/writes for the final accumulation
    FlashAttention-2 also uses thread blocks to load KV cache in parallel for iterative decoding

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH4.6.1
    CH4-Undirected Graphical Models
        CH4.6-Partially Directed Models
            CH4.6.1-Conditional Random Fields
                CRF models conditional distribution by partially directed graph, whose advantage lies in its more flexibility. CRF allows us to use Markov network's factor decomposition semantics to represent conditional distribution. The specification of factors has lots of flexibility compared to explicitly specifying CPD in conditional Bayesian networks. But this flexibility in turn restrict expandability, because the parameters learned has less semantics on their own.
- [[book-notes/面向计算机科学的组合数学|面向计算机科学的组合数学]]: CH4-CH4.4.1
    CH4-线性常系数递推关系
        Make general term the coefficient in generating function to relating generating function with recurrence relation, and then turn recurrence formula into a equation about generating function, thus solve the generating function, then derive the general term of the recurrence.

\[Doc\]
- [[Learn the Basics|pytorch/tutorial/beginner/Learn the Basics]] 
- [[doc-notes/python/packages/pillow]] : Overview, Tutorial, Concepts
- [[Repositories|huggingface/hub/Repositories]]: Sec1-Sec4
- [[doc-notes/triton/Getting Started|triton/Getting Started]]:  Tutorials/Matrix Multiply
- [[Argparse Tutorial|python/how/general/Argparse Tutorial]] 
- [[nvidia/CUDA C++ Programming Guide v12.6]]: CH1

### Week 2
Date: 2024.11.4-2024.11.11

\[Paper\]
- [[Efficient Memory Management for Large Language Model Serving with PagedAttention-2023-SOSP|2023-SOSP-Efficient Memory Management for Large Language Model Serving with PagedAttention]]: Sec0-Sec4.5
    Sec0-Abstract
    Sec1-Introduction: 
        Existing systems preallocated a continous chunk of memory space whose size is the max length of the request. This method causes internal and external fragmentation in GPU DRAM, and do not support KV cache sharing between requests. 
        PagedAttention Manage KV cache blockwisely, allocating physical memory in block granularity, thus reducing internal fragmentation and eliminate external fragmentation, and also enable KV cache sharing between requests in block granularity.
        By increasing GPU DRAM utilization rate significantly, PagedAttention significantly improve the batch size of requests, thus in turn improve the throughtput.
    Sec2-Background:
        Prompt phase: Process user input, generating KV cache. Use Matrix-Matrix multiplication(use masked self-attention to implement causal attention).
        Auto regressive generation phase: Iteratively generate new tokens. Use Vector-Matrix multiplication.
    Sec3-Memory Challenges in LLM Serving:
        LLM service is bounded by memory, especially by the space reserved for KV cache.
        The memory utilization for KV cache storage space is inefficient, due to the inefficient continous memory allocation startegy.

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH7, CH9.2-CH9.3
    CH7-Gaussian Network Models
        CH7.1-Multivariate Gaussians: 
            Two parameterization: Standard, Information Matrix
            Marginal and Conditional density of Joint Gaussian is Gaussian, and the Conditional density is also linear Gaussian model
            Zero in $\Sigma$ implies linear independence in Joint Gaussian, which in turn implies statistical independence
            Zero in $J$ implies conditional independence
        CH7.2-Gaussian Bayesian Networks:
            All CPDs being linear Gaussian model implies the joint density is Gaussian
        CH7.3-Gaussian Markov Random Fields:
            Any Gaussian's information representetaion can be related to a pairwise Markov network with quadratic node and edge potentials
            The converse is not necessarily true, the positive definiteness of $J$ should be guaranteed.
            There exists two sufficient condition to make the converse true: diagonally dominate, pairwise normalizable
        CH7.4-Summary:
            A multivariate Gaussian can be represented both by Gaussian Bayesian network and Gaussian Markov network
            Gaussians are representationally compact and computationally tractable. Even in complicated problem, we can assume the prior to be Gaussian or approximate the inference to make the intermediate be Gaussian to ensure the computational tractability
    CH9-Exact Inference: Variable Elimination
        CH9.2-'Variable Elimination: The Basic Ideas':
            The basic idea of variable elimination is dynamic programming. To calculate the marginal $P(X)$ from CPD $P (X \mid Y)$, we first calculate the marginal of $Y$ and store it, thus calculate the marginal $P (X)$ by $P (X) = \sum_y P (y) P (X\mid y)$, avoiding recalculate $P (Y)$ for every $x \in Val (X)$
        CH9.3-Variable Elimination:
            We viewing the joint density as a product of factors. To calculate the marginalization over a subset of variables, we sum out of the other variables. This process is generalized as the sum-product variable elimication algorithm. The calculation of this algorithm can be simplified by using the property of factor's limited scope to isolated only the related factors to summation (distributive law of multiplication)
            To deal with evidence, we first use the evidence to reduce the factors (leaving the factors compatible with the evidence), and do the same algorithm to the reduced set of factors.
- [[A Tour of C++]] : CH7-CH8

\[Doc\]
- [[python/howto/general/Annotations Best Practices]]
    Best Practice after Python 3.10: use `inspect.get_annotations()` to get any object's annotation
- [[huggingface/hub/Repositories]]: Sec4-Sec10

### Week 3
Date: 2024.11.11-2024.11.18

\[Paper\]
- [[Efficient Memory Management for Large Language Model Serving with PagedAttention-2023-SOSP|2023-SOSP-Efficient Memory Management for Large Language Model Serving with PagedAttention]]: Sec4.6-Sec10
    Sec4-Method: 
        Sec4.1-PagedAttention:
            PagedAttention partitions each sequences KV cache into blocks, each block contains a certain number of tokens' KV cache.
            The blocks are not necessarily contiguous in physical memory. In decoding computation, PagedAttention fetches all the relevant block (not necessarily contiguous in physical memory) to do the computation.
            The blockwise management of KV cache allows us to use flexible paged memory management in vLLM.
        Sec4.2-KV Cache Manager:
            In vLLM, a request's KV cache will be partitioned into contiguous logical blocks, which are mapped to non-contiguous physical blocks. Blocks are filled from left to right.
            On GPU worker, the block engine allocates contiguous DRAM and divide it into blocks.
            The mapping from logical blocks physical blocks for each request are maintained in the block table by block manager.
            vLLM thus can dynamically grow the KV cache memory in block granularity instead of reserving all positions.
        Sec4.3-Decoding with PagedAttention and vLLM
            In prefill, vLLM first allocates physical blocks for prompt, and use conventional self-attention algorithm to compute the KV cache for prompt, and generate the first token. In autoregressive decoding, vLLM uses PagedAttention to compute query token's KV cache and generate new tokens one by one. The new computed KV cache will be filled into empty slots from left to right. If the block is filled, vLLM creates new logical block and allocates physical block.
            Globally, in each decoding iteration, vLLM selects a set of candidate sequences for baching.
            Larger block size allows vLLM to process more position's KV cache in parallel, but will in turn increase internal fragmentation.
        Sec4.4-Application to Other Decoding Scenarios
            In addition to the basic greedy decoding algorithm, vLLM can handle more complex decoding algorithm.
            Parallel sampling: 
            LLM generates multiple outputs sequences for one request, thus the prompt's KV cache can be shared. Sharing means the logical block for each sequence's prompt are mapped into the same physical block. The number of sharers is rercorded by the physical block's reference count. When new token is to be written into a logical block whose corresponding physical block's reference count is larger then 1, vLLM adopts write-on-copy mechanism to copy the physical block' content to a new physical block and write to it. The original physical block's reference count is decresed by 1. Thus most KV cache of the prompt will be shared between a request's multiple outputs in parallel sampling.
            Beam search:
            In each iteration, beam search retains the top-k sequence in $k\cdot |V|$ candidates.
            The beam candidates share initial prompt KV blocks and possibly more KV blocks if they come from the same prefix. The sharing pattern will change dynamically, possibly diverge at some point and converge later.
            In previous LLM service system, the convergence of diverged beam candidate will require copying a large amount of KV cache. In vLLM, they are simply shaing the same physical blocks. The copy will only happen in the copy-on-write of a diverging block, and the overhead is limited to the size of one block.
            Shared prefix:
            The system prompt's KV cache blocks are shared. Their physical blocks can also be cached in advance.
            Mixed decoding methods:
            vLLM supports processing request with different decoding methods simultaneously, because vLLM conceals the memory sharing pattern with the mapping layer which translates logical blocks to physical blocks, thus the execution kernel only need to handle the list of physical block IDs for a sequence instead of managing the sharing pattern for the sequence explicitly.
        Sec4.5-Scheduling and preemption:
            vLLM adopts FCFS schduling policy for requests. vLLM implements all-or-nothing eviction policy for preempted sequences. The recovery methods include swapping and recomputation. Note that the recomputaton latency will significantly lower then the original latency, because all the needed tokens are konwn.
        Sec4.6-Distributed Execution:
            vLLM uses Megatron-LM tensor model parallelism strategy. The attention operator is split on the attention head dimension.
            Each model shard still processes the same set of tokens for a request, thus requiring KV cache for the same potisions. Although the physical block IDs are shared between GPU workers, a worker only store a portion of KV cache for its responsible attention head.
            In each step of execution, the scheduler brodcast the token IDs and block tables for each reqeust to GPU workers. The GPU workers do not need to synchronize on memory management, because they receive all the memory management information at the beginning of each decoding iteration.
    Sec5-Implementation:
        PagedAttention uses three kernels to optimize memory access pattern, including
        fused reshape and block write kernel to split the new KV cache in every transformer layers into blocks and reshape and store to the specified position according to the block table; 
        fused block read and attention kernel to read KV cache according to block table and compute attention. Each block's read is assigned to a warp to ensure coalesced memory access.
        fused  block copy kernel to batchwise process multiple blocks' copy-on-write operation.
        PagedAttention use `fork/append/free` to implement multiple decoding algorithms. `fork` creates new sequence from the existing one. `append` appends a new token to the existing sequence. `free` deletes a finished sequence.
    Sec6-Evaluation:
        The request arrival times are generated by Possion distribution with different request rates.
        The key metrics is throughput. The normalized latency of the system is measured under the workloads with different request rate.
        The normalized latency is equal to the mean of every requests' end-to-end latency divided by its token number.
        With the request rate increasing, the normalized latency increase gradually until the request rate surpassinig the capacity. vLLM can significant improve the capacity by saving lots of KV cache memory, thus can sustain higher request rate, thus enabe batching more requests.
        In compute-bound scenario (shorter sequences, larger memory spaces), vLLM's advantage is less profound.
        The block sharing in advanced decoding scenarios is also efficient, and bring more improvements as the number of parallel samples or the beam width or the length of prefix increases.
    Sec7-Abalation Study:
        PagedAttention kernel involves more overheads of accessing block table, executing extra branches and handling variable sequence lengths, leading to more kernel latency. But the end-to-end performance is bettern due to memory saving.
        Small block size may lead to inefficiency in reading and processing KV cache in parallel. Large block size may lead to more internal fragmentation and lower probibility of sharing. Block size = 16 is a nice tradeoff.
        As for the recovery strategy, the overhead of recomputation is invariant to block size, because it does not involve data transmisson. Swapping is more efficient when block size is large.
    Sec8-Discussion:
        Blockwise memory management mechanism is effective for managing KV cache because the workload for LLM serving can not know the output length in priori, thus requiring dynamic memory allocation. The blockwise management can minimize the framentation of memory allocation in such scenario.
    Sec9-Related Work:
        The iteration-level scheduling in Orca and PagedAttention in vLLM are complementary techniques. Orca schedules and interleaves requests to process more requests in parallel. vLLM increases the memory utilization to fit a larger working set of requests into memory.
        By reducing memory fragmentation and enabling sharing, vLLM can run more requests in parallel.
    Sec10-Conclusion:
        PagedAttention manage KV cache in block granularity, reducing memory fragmentation and enabling sharing. vLLM is built upon PagedAttention, it shows how established techniques like virtual memory and copy-on-write can be used to efficiently manage KV cache and handle various decoding algorithm in LLM serving.

\[Book\]
- [[A Tour of C++]] : CH9.1-CH9.2
    CH9-Strings and Regular Expressions
        CH9.1-Introduction
        CH9.2-Strings
            `string` is a `Regular` type.
            `string` support cacatenation, comparison, subscripting, substring operations, lexicographical ordering etc.
            `s` suffix's corresponding operator is defined in `std::literals::string_literals`.
            `string` 's implementation is shor-string optimized.
            `string` is actually an alias of `basic_string<char>`.
- [[book-notes/面向计算机科学的组合数学|面向计算机科学的组合数学]]: CH4.4.1-CH4.5.2
    CH4-线性常系数递推关系
        Write characteristic polynomial directly from the recurrence relation, and solve the characteristic equation to get $\alpha_i$ s. Then write the general term in terms of $\alpha_i$ s and undermined coefficients. Finally use the initial values to solve the coefficients, and derive the general term formula.

\[Doc\]
- [[nvidia/CUDA C++ Programming Guide v12.6]]: CH2
    CH2-Programming Model:
        Kernel is executed by each CUDA thread
        Thread hierarchy: thread -> thread block -> thread block cluster -> grid
        Memory hierarchy: local memory -> shared memory -> distributed shared memory -> global/constant/texture memory
        Host program manage global/constant/texture memory space for kernels via calls to CUDA runtime
        Unified memory provides coherent memory space for all devices in system
        In CUDA, thread is the lowest level of abstraction doing memory and computation operation
        Asynchronous programming model (started from Ampere) provides `cuda::memcpy_async/cooperative_groups::memcpy_async` operation, and the asynchronous operation use synchronization objects (`cuda::barrier` , `cuda::pipeline`) to synchronize threads in thread scope
        The thread scope in CUDA includes `cuda::thread_scope::thread_scope_thread/block/device/system`
        Compute Capability is the version of SM architecture, denoted by a major version number and a minor version number
        CUDA version is the version of CUDA software platform
- [[docker/get-started/What is Docker]] 
    Containers include everything needed for running an application
    Use containers to be the unit of distributing and deploying applications
    Docker client ( `docker` ) use Docker API to communicate with Docker daemon ( `dockerd` ), which is responsible for managing containers
    Docker registry stores images. `docker pull` pulls image from registry, and `docker push` pushes image to registry
    Image is an read-only template of instructions for creating container. Image is defined by Dockerfile, and is consists of layers. Each instruction in Dockerfile defines a layer in image. Once created, the image can not be modified. Container is a runnable instance of an image.
- [[docker/get-started/Docker Concepts]]
    The Basics: 
    Container is essentially an isolated process. Multiple containers share the same kernel.
    Container image packages all the needed binaries, files, configurations, libraries to run a container. Image is read-only, and consists of layers, each of which represents a set of filesystem changes.
    Repository is a collection of related images in the registry. Keep each container doing only one thing.
    Docker Compose uses yaml file to define the configurations and interactions for all related containers. Docker Compose is an declarative tool.
    Building Images:
    The image is consists of multiple layers. We reuse other images' base layers to define custom layer.
    After each layer is downloaded, it is extracted into the own directory in the host filesystem.
    When running a container from an image, a union filesystem is created, where layers are stacked on top of each other. The container's root directory will be changed to the location of the unified directory by `chroot`.
    In the union filesystem, in addition to the image layers, Docker will create a new directory for the container, which allows the container make filesystems changes while keeping original layers untouched.
    Dockerfile provides instructions to the image builder. Common instructions include `FROM/WORKDIR/COPY/RUN/ENV/EXPOSE/USER/CMD` etc.
    A Dockerfile typically 1. determine base image 2. install dependencies 3. copy in sources/binaries 4. configure the final image.
    Use `docker build` to build image from Dockerfile. 
    Image's name pattern is `[HOST[:PORT_NUMBER]/]PATH[:TAG]` 
    Use `docker build -t` to specify a tag for the image when building. Use `docker image tag` to specify another tag for an image.
    Use `docker push` to push built image.
    Modify `RUN` 's command will invalidate the build cache for this layer.
    Modify files be `COPY` ed or `ADD` ed will invalidate the build cache for this layer.
    All the following layer of an invalidated layer will be invalidated.
    When writing Dockerfile, considering the invalidation rule to ensure the Dockerfile can build as efficient as possible.
    Multi-stage build introduces multiple stages in Dockerfile. It is recommended to use one stage to build and minify code for interpreted languages or use one stage to compile code for compiled languages. Then use another stage, copying in the artifacts in the previous stage, only bundle the runtime environment, thus reducing the image size.
    Use `FROM <image-name> AS <stage-name>` to define stage. Use `--from=<stage-name>` in `COPY` to copy previous stages artifacts.

### Week 4
Date: 2024.11.18-2024.11.25

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH10.1-CH10.3, CH11.1-CH11.3.4
    CH10-Exect Inference: Clique Trees
        CH10.1-Variable Elimination and Clique Trees
            We consider a factor $\psi_i$ to be a computational data structure, which takes a message $\tau_j$ generated by factor $\psi_j$ and send message $\tau_i$ to another factor.
            In cluster graph, each node/cluster is associated with a set of variables. Two cluster is connected by an edge if and only their associated variable set has overlap.
            The variable elimination algorithm's execution process defines a cluster graph. Each factor $\psi_i$ used in the computation corresponds to a cluster $\pmb C_i$, whose associating variable set is the scope of the factor.
            The computation in variable elimination will eliminate a variable in the associating factor. If the message $\tau_i$ generated by the variable elimination in $\psi_i$ will be used in generating $\tau_j$ in $\psi_j$, then we draw an edge between $\pmb C_i, \pmb C_j$.
            The cluster graph generated by variable elimination process is a tree (each factor in the algorithm will only be used once, thus each node in the graph will only has one parent node). What's more, this cluster tree satisfies running intersection property. Such a cluster tree can also be called as clique tree.
            The running intersection property implies independencies (Theorem 10.2)
        CH10.2-Message Passing: Sum Product
            The same clique tree can be used for many different executions of variable elimination. The clique tree can cache computation, allowing multiple variable elimination execution to be more efficient.
            The clique tree can be used to guide the operations of variable elimination. The clique dictates the operations to perform on the factors in it, and defines the partial order of these operations.
            In the clique tree message passing algorithm, each clique $\pmb C_j$ 's initial potential is the product of all of its associated factors. For each clique, to compute the message to pass, it first multiply all incoming messages, and then sum out all variables in its scope except those in the sepset between itself and the message receiver. The process proceeds up to the root clique, the root finally get the unnormalized marginal measure over its scope (equals to the joint measure summed out of all other variables), which is called as belief. (Corollary 10.1)
            Intuitively, the message between $\pmb C_i , \pmb C_j$ is the products of all factors in $\mathcal F_{\prec i(\rightarrow j)}$ , marginalized over the variables over in the sepset between $\pmb C_i$ and $\pmb C_j$. (Theorem 10.3)
            In all executions of the clique tree algorithm, whenever the message is sent between two cliques in the same direction, it is necessarily the same. Thus each edge in the clique tree essentially associates two messages, one for each direction.
            The sum-product belief propagation utilizes this property. It consists of an upward pass and an downward pass. After the two passes, each edge will get its associated two messages, therefore each clique's belief/marginal measure can be immediately computed. This is an efficient algorithm to use when computing all the cliques beliefs. (Corollary 10.2)
            This algorithm will also calibrate all the neighboring cliques in the Tree. Thus the clique tree is calibrated.
            The belief over the sepset associated with the edge is precisely the product of its two associated messages. Further, the unnormalized Gibbs measure over the clique tree equals to the product of all cliques' beliefs divided by the product of all sepset's beliefs. (Proposition 10.3)
            Thus the clique and sepset beliefs provides a reparameterization of the unnormalized measure. This property is called the clique tree invariant.
        CH10.3-Message Passing: Belief Update
            In sum-product-division algorithm, the entire message passing process is executed in terms of the belief of cliques and sepsets instead of initial potential and messages. The message maintained by the edge is used to avoid double-counting: Whenever a new message is passed along the edge, it is divided by the old message, therefore eliminating the previous/old message from updating the clique (who sent the previous/old message).
            At convergence, we will have a calibrated tree, because in order to make the message update have no effect, we need to have $\sigma_{i\rightarrow j} = \mu_{i, j} = \sigma_{j\rightarrow i}$ for all $i, j$, which means the neighboring cliques agree on the variables in the sepset.
            sum-product message propagation is equivalent to belief-update.
            In the execution of belief-update message passing, the clique tree invariant equation (10.10) holds initially and after every message passing step (Corollary 10.3)
            Incremental update: multiply the distribution with a new factor. In clique tree, we multiply the new factor into a relevant clique, and do another pass to update other relevant cliques in the tree.
            Queries outside a clique: construct the marginal over the query in containing subtree, instead of the entire tree.
            Multiple queries: compute the marginal over each clique pair using DP.
    CH11-Inference as Optimization
        CH11.1-Introduction
            The approximation method's approximation arises from constructing an approximation to the target distribution $P_\Phi$.
            The approximation distribution takes a simpler form, generally exploiting the local factorization structure.
            The inference task is thus reformulated as optimizatin an objective function over the approximation distribution class $\mathcal Q$, which falls into the category of constrained optimization problem.
            There are three categories of methods:
            1. Use clique-tree message passing scheme in structures other than clique-tree, such as loopy belief propagation. It can be understood as optimizating the approximate form of energy functional.
            2. Use message propagation in clique trees with approximation messages, konwn as the expection propagation algorithm. It maximize the exact energy functional with relaxed constraint.
            3. Generalize the mean field method, using a class of $\mathcal Q$ which has simple factorization, using the exact energy functional.
            Exact Inference can also be reformulated as searching for a calibrated distribution which is close to $P_\Phi$. The commonly-used measurement of "closeness" between two distributions is KL divergence. The calibrated distribution is represented by a set of clique beliefs and sepset beliefs.
            $D(Q|| P_\Phi)$ can be rewrite in terms of the energy functional, thus minimizing the KL divergence is equivalent to maximizing the energy functional $F[\tilde P_\Phi, Q]$ (Theorem 11.2).
            The energy functional is the lower bound of the logarithm of the partition function $Z$ for any choice of $Q$, that is, $\ln Z \ge F[\tilde P_\Phi, Q]$. The equality holds if and only if $D(Q||P_\Phi) = 0$.
            The chapter explores inference methods which can be viewed as strategies for optimizaing the energy functional. They are often referred as variational methods.
        CH11.2-Exact Inference as Optimization
            Given an cluster tree associated with a set of beliefs $Q$, we can factorize the energy functional $F[\tilde P_\Phi, Q]$ in terms of clique and sepset beliefs.
            We then define a constrained optimization problem 'CTree-Optimize' over the space of belief sets. The objective is to maximize the factorized energy functional, and the constraint is to ensure the marginal consistency (ensure the beliefs are calibrated).
            Utilizing Lagrangian multiplier method, we can derive the forms/equations that the beliefs in the optimal solution $Q$ should conform to. (Theorem 11.3)
            Most importantly, equation (11.10) defines each message in terms of other messages, which allows us to use iterative approach to solve the fixed-point equations.
            To solve the fixed-point equation, we apply each equation as assignments, iteratively use the RHS as the new value for the LHS until convergence. In certain conditions (such as the cluster tree is a clique tree), the convergence is guaranteed. Therefore, in clique tree, this iterative method is equivalent to the belief-update/sum-product message passing algorithm we discussed previously. 
            The running intersection property for cluster graph is relaxed, but can still prevent direct 'cyclic argument'. However, the loop in cluster graph can still lead to circular reasoning. The definition for general cluster graph is also weaker, which only requires the agreement on sepset variables instead of all common variables (sepset is not necessarily equal to the intersection set in general cluster graph).
            Though the beliefs in cluster graph are not guaranteed to be $P_\Phi$ 's marginals, the cluster graph still maintains the invariant property, which means the cluster graph can be viewed as a reparameterization of the unnormalized original distribution $P_\Phi$. 
            What's more, the beliefs in cluster graph is essentially $P_{\mathcal T}$ 's marginal, where $\mathcal T$ is a subtree in the graph. This property is called tree consistency.
            For pairwise Markov networks, we can introduce a cluster for each potential, and put edges between clusters which have overlapping scope. Loopy belief propagation was originally based on this construction.
            For more complex network, we can use Bethe cluster graph, which uses bipartite graph.
- [[book-notes/面向计算机科学的组合数学|面向计算机科学的组合数学]]: CH5.1-CH5.2

\[Doc\]
- [[Models|huggingface/hub/Models]]: Sec0-Sec1
- [[doc-notes/python/pep/PEP 8-Style Guide for Python Code]]

## December
### Week 1
Date: 2024.11.25-2024.12.2

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH11.5.1, CH12.1-CH12.3
    CH11-Inference as Optimization
        CH11.5-Structured Variational Inference
            CH11.5.1-The Mean Field Approximation
                The mean field approximation assumes all the variables are independent from each other. Thus $Q$ is fully factorized.
                The optimization for the energy functional takes the form of iterative optimization (coordinate ascent). In each iteration, we only optimize $Q(X_i)$, other variables' marginal is fixed. The iterative coordinate ascent algorithm is guaranteed to converge, because the energy functional is bounded, and guaranteed to be nondecreasing under the coordinate ascent process.
                The computation for optimal $Q(X_i)$ only involves the potentials that contains variable $X_i$ .
    CH12-Partical-Based Approximation Inference
        This chapter talks about Monte Carlo methods, i.e. , how to generate samples from the target distribution or an approximate distribution and how to construct estimator for the desired expectation from these samples.
        CH12.1-Forward Sampling
            Forward sampling utilizes Bayesian network's factorization theorem. It samples variables according to the partial order of BN. Thus with parent variables' sample value determined, each variable's sampling process only involves its CPD.
            To sample from the posterior, a simple method is rejection sampling, which rejects the samples inconsistent with the evidence.
            The problem of rejection sampling is that too many samples will be rejected in case the evidence has low probability, thus leading to low efficiency.
        CH12.2-Likelihood Weighting and Importance Sampling
            To address rejection sampling's inefficiency problem, likelihood weighting algorithm directly sets evidence variables' value to the observed value. Thus no rejection is needed anymore.
            However, in this way, the relevance between the evidence value and other variables' sampled value is missed. To compensate for it, the algorithm give a weight to each sample. A sample's weight is the probability of evidence occurring together with other variables' sampled value in this sample.
            Importance sampling samples from another distribution called proposal distribution, and weight the sample accordingly to ensure the expectation still remains unchanged.
            Importance sampling achieves the lowest variance when $Q(\pmb X) \propto |f(\pmb X)|P(\pmb X)$.
            Normalized importance sampling assumes we can only access the unnormalized version of $P$ (i.e. $\tilde P$). In this case, the estimator for an expectation with respect to $P$ can still be constructed, by estimating the normalization constant $Z$ simultaneously. 
            The normalized importance sampling estimator is not unbiased. Its bias and variance goes down as the reciprocal of sample number $M$ (i.e. $\frac 1 M$).
            In practice, the normalized importance sampling estimator is typically lower then the unnormalized one (no theory guarantee). This reduction in variation often outweighs the bias term, so that the normalized one is used often even if $P$ is known. 
            The effective sample size for a particular set of samples is defined in terms of the variance. That is, the variance of $M_{\text{eff}}$ samples from $P$ is equivalent to the variance of $M$ samples from $Q$.
            In BN, using importance sampling with $Q$ defined by the mutilated network is equivalent to using likelihood weighting algorithm.
        CH12.3-Markov Chain Monte Carlo Methods
            In likelihood weighting, the evidence will only affect the sampling process for the decedents, so the non-decedents are essentially sampled from the prior instead of the posterior. If the divergence between the prior and the posterior is too large, the weight is not enough to compensate for it.
            The MCMC method adopts an entirely different pattern from the weighting methods. The MCMC methods is inspired by a physical observation, which says an particle's state evolves in a Markov chain, and its distribution will converge to a stationary distribution as the Markov chain proceeds.
            The MCMC method defines a Markov chain whose stationary distribution is the desire sampling distribution $P$, and make the sample generated from the initial distribution (usually the prior) keep evolve its assignment(state) in the Markov chain. Because the Markov chain's state distribution will eventually converge to its stationary distribution, the distribution that the sample conforms to will eventually converge to the desired sampling distribution. Thus we can eventually treat the sample as sampled from the desired sampling distribution. In the process, the sample's distribution will gradually get closer to the desired sampling distribution.
            To ensure the Markov chain has an unique stationary distribution, the state space should be ergodic (i.e. the transition matrix should have all its entries positive).
            Gibbs sampling algorithm is an implementation of the MCMC method. It constructs a separate transition model for each variable (as the posterior in $P$ as this variable given all other variables' current sampled value ), and combines them as a whole transition model for the Markov chain. This construction is proved to make the Markov chain converge to the desired distribution $P$.
            If a Markov chain $\mathcal T$ satisfies the detailed balance equation with respect to some distribution $\pi$, then $\mathcal T$ is reversible, and $\pi$ is a stationary distribution. If $\mathcal T$ is regular, then $\pi$ is the unique one.
            MH algorithm is a general method for constructing a Markov chain for a desired stationary distribution $P$. It uses any proposal distribution $Q$ as part of the transition model, and define the acceptance probability in terms of the $Q$ and $P$. The proposal distribution and acceptance probability together define the transition model for the Markov chain, whose stationary distribution is proved to be $P$.
- [[book-notes/一份（不太）简短的 LaTeX2e 介绍|一份（不太）简短的 LaTeX2e 介绍]]: CH1-CH2
    CH1-LaTeX 的基本概念
        LaTeX 命令分为两种：`\` + 一串字母；`\` + 单个非字母符号
        字母形式的命令忽略其后的空格字符
        LaTeX 的环境由 `\begin,\end` 命令包围
        LaTeX 用 `{}` 划分分组，限制命令的作用范围
        `\documentclass` 指定文档类，`\begin{document}` 开启文档环境，二者之间为导言区，用于用 `\usepackage` 使用宏包
        `\include, \input` 用于插入文件
    CH2-用 LaTeX 排版文字
        UTF-8 是对 Unicode 字符集的一种编码方式
        XeTeX 和 LuaTeX 完全支持 UTF-8，`fontspec` 宏包用于调节字体
        `ctex` 宏包和文档类 (`ctexart, ctexbook, ctexrep` ) 用于支持中文排版
        LaTeX 将空格和 Tab 视作空白字符，连续的空白视作一个空白，行首的空白会被忽略
        连续两个换行符生成一个空行，将文字分段 (等价于 `\par` )，连续空行视作一个空行
        LaTeX 会自动在合适位置断行断页，也可以手动用命令控制

\[Doc\]
- [[doc-notes/python/pep/PEP 257-Docstring Conventions|python/pep/PEP 257–Docstring Conventions]]

### Week 2
Date: 2024.12.2-2024.12.9

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH17.1-CH17.3
    CH17-Parameter Estimation
        Two main approaches for parameter estimation is MLE and Bayesian approach.
        CH17.1-Maximum Likelihood Estimation
            The hypothesis space contains all the possibilities we are considering. The objective function measures how good different hypotheses in the space are relative to the data set.
            The 'goodness' of one hypothesis is how well it can predict the observed data set. If the data is likely given the hypothesis/parameter, then the hypothesis/parameter is a good predictor.
            Likelihood function is a function respect to parameter $\theta$ and the data set $\mathcal D$. It measures the posterior probability of the data set given the parameter $\theta$.
            The parameter values with higher likelihood are more likely to generate the observed data. The parameter value which maximize the likelihood is called the maximum likelihood estimator (MLE)
            The log likelihood is monotonically related to the likelihood. Thus maximizing one is equivalent to maximizing the other.
            Confidence intervals measure our confidence about our estimation.
            A sufficient statistic is a function of the data, which summarizes the relevant information for computing the likelihood. A static becomes sufficient statistic if given the same parameter, the equality of any two data sets' statistic implies the equality of their likelihood.
            If the two parameter's likelihood for all possible choices of $\mathcal D$ is equivalent, then the two parameter is indistinguishable.
        CH17.2-MLE for Bayesian Networks
            In Bayesian network, the likelihood function can be factorized according to the network's structure. Each CPD's local likelihood function can be further decomposed according to the parent variables' assignment. (with an implicit assumption that each CPD's parameter is independent with each other)
            This property is called the decomposability of the likelihood function. This property holds if each CPD is parameterized by a separate set of parameters that do not overlap.
            If the decomposability of the likelihood function holds, then the global MLE is the combination of each CPD's local MLE. The global problem is decomposed into independent subproblems.
            The problem of MLE is related to the data fragmentation phenomenon. With the dimensionality of the parents set growing large, the dataset will be partitioned into a large number of small subsets. The number of parameters will explode exponentially, and the number of samples to estimate a parameter will become too small. This is the key factor limiting our learning BN from data.
        CH17.3-Bayesian Parameter Estimation
            Compared with MLE, Bayesian statistics takes prior knowledge into account.
            We encode our prior knowledge about the parameter $\theta$ as a probability distribution: the prior distribution. We create a joint distribution over the observed data and the parameter, which can be represented as a meta-network.
            We can decompose this joint distribution into the product of the likelihood and the prior. The posterior of the parameter is proportional to the product of the likelihood and the prior.
            The prediction of next sample if rewritten as the integral of the product of the conditional probability and the posterior over all possible values of $\theta$ .
            In the toss experiment setting, for uniform priors, the form of the prediction (Bayesian estimator) is much like the MLE, except that is adds one "imaginary" sample to each count. The Bayesian estimator and the MLE estimator converges to the same value as the number of samples grows.
            The Bayesian estimator under uniform prior is called Laplace's correction.
            Another natural choice for prior is Beta distribution, which is parameterized by two hyperparameters. which corresponds the imaginary heads and tails respectively.
            If the prior is the Beta distribution and the likelihood function is the Bernoulli likelihood function, the posterior is also the Beta distribution with its hyperparameters corrected by the observation. Therefore, we say that the Beta distribution is conjugate to the Bernoulli likelihood function.
            As we obtain more data, the effect of the prior diminishes.
            The Bayesian framework allows us to capture prior knowledge as the prior probability, also allows us to capture the distinction a few samples and many samples by the peakedness of the posterior. 
            In Bayesian approach, we view the parameter as a random variable, and use probabilities to describe the uncertainty about the parameter, and then use Bayes rule to take into account the observations to see how they affect our belief about the parameter.
            Whether we can compactly describe the posterior depends on the form of the prior.
            The Dirichlet priors are conjugate to the multinomial model, which means if the likelihood takes the form of multinomial distribution and the prior is an Dirichlet distribution, then the posterior is also an Dirichlet distribution.
            The estimation for this situation takes a similar form to MLE, the difference is that we added the hyperparameters to our counts. Thus the Dirichlet hyperparameters are also called pseudo-counts.
            The summation of pseudo-counts reflects our confidence of the prior, which is called the equivalent sample size.
            The estimation in this situation can be also viewed as the weighted average of the prior mean and the MLE estimation, therefore, it's easy to see that the Bayesian prediction converges to the MLE estimation when $M\rightarrow \infty$. Intuitively, large data set will make the prior's contribution negligible, while small data set's estimation will be biased toward to the prior probability.
            The existence of prior makes Bayesian estimation more stable then the MLE estimation. This smoothing effect results in more robust estimates when the data is not enough. If we do not have enough prior knowledge, we can use uniform prior, to prevent our estimates from taking extreme values

\[Doc\]
- [[docker/get-started/Docker Concepts]]
    Running Containers:
    `-p HOST_PORT:CONTAINER_PORT` is used to publish the container's port `CONTAINER_PORT` and forward the traffic of `HOST_PORT` to the container's port.
    `-P` is used to publish all the ports specified in `EXPOSE`
    `-e ...=...` is used to set environment variables
    environment variables can be specified in `.env` , and use `--env-file` to pass this file.
    `--memory/--cpus` is used to set container's available resource limit.
    `-v` is used to mount volume. A volume can be mounted to multiple containers simultaneously
    `--mount` can be used to specify bind mount. Permissions is specified by `:ro/:rw` .
    Keep each container do one thing. Docker Compose use `compose.yml` to define multiple containers' configuration and connection.    

### Week 3
Date: 2024.12.9-2024.12.16

\[Paper\]
- [[paper-notes/mlsys/The Deep Learning Compiler A Comprehensive Survey-2020-TDPS|2020-TDPS-The Deep Learning Compiler A Comprehensive Survey]]: Sec1-Sec3
    Sec1-Introduction
        ONNX defines a unified format to represent DL models
        DL hardware can be divided into three categories: 1. general-purpose 2. dedicated 3. neuromorphic
        DL compiler aims to alleviate the burden of optimizing DL models on each DL hardware manually. DL compiler includes TVM, Tensor Comprehension, nGraph, Glow, XLA.
        DL compiler takes model description in DL frameworks as input and output the optimized code implementation of this model for DL hardware. The optimization is specific to model specification and hardware architecture.
        DL compiler also adopts layered design, including frontend, IR, backend, but the IR has multiple levels.
    Sec2-Background
        Deep Learning Frameworks
        TensorFlow employs a dataflow graph of primitive operators extended with restricted control edges.
        Keras is TensorFlow's frontend, written in pure Python.
        PyTorch embeds primitives for constructing dynamic dataflow graph in Python, where the control flow is executed by the Python interpreter.
        FastAI is PyTorch's frontend.
        ONNX defines a scalable computation graph model.
        Deep Learning Hardware
        TPU includes Matrix Multiplier Unit, Unified Buffer, and Activation Unit. MMU mainly consists of a systolic array. TPU is programmable but use matrix as primitive instead of vector or scalar.
        Hardware-specific DL Code Generator
        FPGA lies between CPUs/GPUs and ASIC. HLS programming model provides C/C++ programming interface to program FPGA. However, the DL model is usually described by the languages of DL framework instead of bare C/C++. Thus mapping DL models to FPGA remains a complicated work.
        The hardware-specific code generator targeting FPGA take DL model as input, output HLS or Verilog/VHDL. Based on the generated architecture of FPGA-based accelerators, the code generator can be classified as the processor architecture specific or the streaming architecture specific.
        The processor architecture FPGA accelerator comprises of several Processing Units, which are comprised of on-chip buffer and multiple smaller Processing Engines. The DL code generator targeting this architecture adopt hardware templates to generate the accelerator design automatically. The number of PUs and the number of PEs per PU are important template parameter. Tiling size and batch size are also essential scheduling parameters about mapping DL models to PUs and PEs. All these parameters are usually determined by design space exploration.
        The streaming architecture FPGA accelerator comprises of multiple different hardware blocks, and usually have one block for each layer of the input Dl model. All hardware block can be utilized in a pipeline manner with streaming input data.
    Sec3-Common Design Architectures of DL Compilers
        DL model will be translated into multi-level IRs by the DL compiler, where the high level IR resides on the frontend and the low-level IR resides on the backend. The high-level IR is associated with hardware-independent optimizations and transformations, and the low-level IR is associated with the hardware-specific optimizations and transformations.
        The high-level IR is also known as the graph IR, which represents hardware-independent computation and control flow. It establishes the control flow and the dependency between the operators and the data. It provides an interface for graph-level optimization.
        Low-level IR is fine-grained enough to reflect the hardware characteristics. Low-level IR should allow the usage of third-party tool-chains in compiler backends.
        The frontend takes DL model as input and output graph IR. The optimization on graph IR can be classified into: node-level, block-level, dataflow-level. The optimized computation graph will be passed to the backend.
        The backend takes graph IR as input and output low-level IR. The backend can directedly convert graph IR into third party toolchains' IR like LLVM IR for general purpose code generation and optimization. The backend can also use customized compilation pass to do better. The commonly-applied hardware-specific optimizations include hardware intrinsic mapping, memory allocation and fetching, memory latency hiding, parallelization, loop oriented optimization.
        Existing backend uses auto-scheduling and auto-tuning to determine the optimal parameter setting.
        Low-level IR can be compiler JIT or AOT.

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH17.4, CH19.1-CH19.2, CH20.1-CH20.3
    CH17-Parameter Estimation
        CH17.4-Bayesian Parameter Estimation in Bayesian Networks
            Global parameter independence: each CPDs' parameter's prior is independent from each other. Thus the prior of the whole parameter has a fully decomposed form.
            If global parameter independence holds, then complete data d-separates each CPDs' parameter, which in turn indicates that the posterior of the whole parameter had a fully decomposed form.
            According to Bayes rule, the parameter's posterior can be rewritten as the product of the likelihood function and the prior divided by the marginal probability of the data set. The likelihood function can be decomposed into the product of local likelihoods and the prior can be decomposed into the product of local priors (with global parameter independence holds). Therefore the posterior also has a fully decomposed form. The fully decomposed form can also be directly derived from the meta-Bayesian network.
            To do prediction, we need to integrate over all legal parameter values to calculate the posterior probability. If the data is IID, and the global parameter independence holds, then the calculation can be factorized into a product of local integration associated with each CPD.
            Therefore, what we need to do is solve the local Bayesian estimation problem independently and combine them into the global one.
            Local parameter independence indicates that the local prior can be further factorized according to the parent variables' assignment.
            If the CPDs are not multinominal CPDs, we may not have a conjugate prior or a closed-form integral for the Bayesian integral. When a full Bayesian solution is impractical, we may resort to maximum a posterior estimation. If we have a large amount of data, the posterior is often sharply peaked around its maximum, therefore in this case the Bayesian integral is roughly equivalent to the MAP estimation.
            MAP estimation can also be viewed as provide regularization over the likelihood function. The regularization term's effect will diminish with the increase of the number of samples.
            MAP estimation can be used in practice, because we will usually choose a well formed prior.
    CH19-Partially Observed Data
        CH19.1-Foundations
            To analyze the probabilistic model of the observed training set, we must consider not only the data-generation mechanism, but also the mechanism by which data are hidden. Every observation is derived by the combination of the two mechanisms.
            If the outcome variables and the observation variables are marginally independent, then we say the data missing model is missing completely at random. In this situation, the whole likelihood can be decomposed as the product of the likelihood of the outcome variables and the likelihood of the observations variables. We can maximize the likelihood of interest independently.
            Given the observed outcome variables, if the hidden outcome variables and the observation variables are conditionally independent, then we say the data missing model is missing at random. In this situation, we can also decompose the likelihood, and use only the observed variables to optimize the parameters of the outcome distribution.
            However, in general, the likelihood function of the observation is a sum of likelihood function of the observation with all possible hidden assignments, each of which defines a unimodal function. Thus the likelihood function with incomplete data is a multimodal function and takes the form of "a mixture of peaks".
            Thus the likelihood function is not decomposable again, and will be hard to optimize.
        CH19.2-Parameter Estimation
            Because the likelihood function is multimodal thus hard to optimize, when doing MLE, we have to maximize a highly nonlinear in a high dimensional space. There are two main classes of methods for performing this optimization: generic nonconvex optimization algorithm (gradient ascent), more specialized method for optimizing likelihood functions (EM).
            The gradient of the log-likelihood function with respect to a single CPD entry $P(x\mid \pmb u)$ is stated in Theorem 19.2. This theorem provides the form of the gradient of table-CPDs. For other CPDs, we can use the chain rule the derivatives to compute the gradient.
            To compute the gradient, we need to compute the joint distribution $P(X[m], \pmb U[m], \mid \pmb o[m], \pmb \theta)$ for each $m$, therefore we need to do inference for each data case using one clique tree calibration.
            After computing the gradient, there is a issue that all components of the gradient vector is nonnegative (since increasing each of the parameters will lead to higher likelihood). Thus, a step in the gradient direction will increase all parameters, leading to an illegal probability distribution.
            There are two common approaches to solve this issue. 
            The first one is to modify the gradient ascent procedure to respect these constraints. We project the gradient vector to the hyperplane that satisfies the linear constraints of the parameters, and ensure the gradient stepping do not step out of bounds so that the parameters will be nonnegative. 
            The second one is to reparametrize the problem, introduce new parameters $\lambda_{x\mid \pmb u}$ to define $P(x\mid \pmb u)$. Now the value of $\lambda$ is not under constraint. We can use standard gradient ascent procedure to update $\lambda$ s.
            Another way is to use Lagrange multipliers.
            The gradient ascent will only guarantee we converge to a local maximum. Therefore some methods like choosing random starting points, applying random perturbations to converge points can be used for help.
            EM algorithm is an alternative way to optimize a likelihood function. The intuition is to fill in the missing value, and then  use standard, complete data learning procedure. Such approaches are called data imputation methods in statistics.
            When learning with missing data, we are actually trying to solve two problems at once: learning the parameters, and hypothesizing values for unobserved variables in each data cases. Each of these task is fairly easy when we have the solution to the other.
            EM algorithm iteratively solve one of the two problems. We start from random initial point, and iteratively do the following two steps: 1. infer the expected sufficient statistics for the unobserved variables (E-step) 2. infer the parameter based on the complete data (M-step)
            This sequence of steps provably improve our parameters, which means the likelihood will be non-decreasing. Therefore, this algorithm is guaranteed to converge to a local maximum.
            Note that we use posterior probabilities to compute the expected sufficient, therefore we consider both the observed data (evidence) and the current parameter.
            In practice, EM generally converges to a local maximum of the likelihood function.
            Apply EM in clustering, we are viewing the data as coming from a mixture distribution and attempts to use the hidden variable to separate out the mixture into its components. If we use hard-assignment EM, we get k-means.
            Hard-assignment version tends to increase the contrast between different classes, since assignments have to choose between them. Soft-assignment can learn classes that are overlapping, since many instances contribute to two or more classes.
            Hard-assignment traverses the combinatorial space of assignments to the hidden variables $\mathcal H$. Soft-assignment traverses the continuous space of parameter assignments. The former makes discrete steps, and will converge faster. The latter can take paths that are infeasible to the former, and can shift two clusters' mean in a coordinated way, while the former can only "jump", since it cannot simultaneously reassign multiple instances and change the class means.
    CH20-Learning Undirected Models
        CH20.1-Overview
            The biggest difference between MN and BN is the partition function. This global factor couples all of the parameters  across the network, preventing us from decomposing the problem and estimating local groups of parameters separately.
            Therefore, even MLE estimation in the complete data case cannot be solved in a closed form (except for chordal MN, which is equivalent to BN).
            To learn MN, we generally resort to iterative methods, and each iteration step of it requires us to run inference in the network.
            Bayesian estimation for MN also has no closed-form solution. Thus the integration associated it must be performed using approximate inference (variational methods or MCMC).
            In this area, part of the work focuses on the formulation of alternative, more tractable objectives of this estimation problem. Another part of the work focuses on the approximate inference algorithms.
            Structure learning for MN also need approximation methods for similar reason. The advantage of MN's structure learning over BN's structure learning is the lack of acyclicity constraint. The acyclicity constraint will couple decisions regarding the family of different variables. 
        CH20.2-The Likelihood Function
            For MN which has equivalent structure to BN, we can use BN's CPDs to represent MN's potentials. The BN's MLE solution is exactly the MN's MLE solution.
            We use log-linear format to represent the Gibbs distribution. Thus the parameters to learn corresponds to the weight we put on each feature. In this setting, the sufficient statistics of the likelihood function are the sums of the feature values in the instances in $\mathcal D$.
            In this setting, the likelihood function can be described as a sum of two functions, the first one is linear in the parameters. 
            We can prove that $\ln Z(\pmb \theta)$ is convex with respect the $\pmb \theta$ (It has semi-positive Hessian). The first derivative of $\ln Z(\pmb \theta)$ with respect to $\theta_i$ is $E_{\pmb \theta}[f_i]$. The second derivative of $\ln Z(\pmb \theta)$ with respect to $\theta_i, \theta_j$ is $Cov_{\pmb \theta}[f_i; f_j]$.
            Therefore, $-\ln Z(\pmb \theta)$ is concave in $\pmb \theta$, the sum of a linear function and a concave function is concave. Thus the log-likelihood function is concave. Therefore the log-likelihood function has no local maximum. (only has multiple equivalent global maximum)
        CH20.3-Maximum (Conditional) Likelihood Parameter Estimation
            For a concave function, its maxima are precisely the points at which the gradient is zero. Thus we can precisely characterize the maximum likelihood parameters $\hat {\pmb \theta}$.
            At the maximal likelihood parameter $\hat {\pmb \theta}$, the expected value of each feature relative to $P_{\hat {\pmb \theta}}$ matches its empirical expectation in $\mathcal D$. In other words, we want the expected sufficient statistics in the learned distribution to match the empirical expectations. This type of equality constraint is called moment matching. Therefore, the MLE estimation is consistent: if the model if sufficiently expressive to capture the data-generating distribution, then, at the large sample limit, the optimum of the likelihood objective is the true model.
            Although the likelihood function is concave, there is no analytical form of its maximum. Thus we have to use iterative methods to search for the global optimum.
            The gradient can be computed according to (20.4), it is the difference between the feature's empirical count in the data and the expected count relative to the current parameterization $\pmb \theta$.
            To compute the expected count, we have to compute the different probabilities of the form $P_{\pmb \theta}(a, b)$, which needs us to do inference at each iteration. To reduce the computational cost, we may use approximate methods.
            In practice, standard gradient ascent converges slowly and is sensitive to the step size. Mush faster convergence can be obtained with second-order methods, which utilize the Hessian to provide the quadratic approximation to the function. The computation of Hessian is illustrated in (20.5), which may also need approximation.
            If we only need the model to do inference, we can train a discriminative model. In other words, we train a CRF that encodes a conditional distribution $P(\pmb Y\mid \pmb X)$.
            Now the objective is the conditional likelihood and its log. The objective can be proved to be concave. Each data instance $\pmb y[m]$ 's log-likelihood is the log-likelihood of the data case in the MN reduced to the context $\pmb x[m]$.
            In unconditional case, each gradient step requires only a single execution of inference. When training a CRF, we must execute inference for each data case (in the reduced, simpler MN). If the domain of $\pmb X$ is very large, the reduction will be more beneficial. Thus in this case training a discriminative network will be more economical.
            In the missing data case, the likelihood function will be multiple modal, thus losing its concavity.
            In this case, according to (20.9), the gradient of feature $f_i$ is the difference between two expectations - the feature expectation over the data and the hidden variables minus the feature expectation over all the variables.
            Applying EM in MN is similar to BN. The difference is in M-step, where MN needs run inference to get gradient.
            The trade-off between gradient method and EM method is more subtle in MN.

### Week 4
Date: 2024.12.16-2024.12.23-2024.12.30

\[Paper\]
- [[paper-notes/mlsys/The Deep Learning Compiler A Comprehensive Survey-2020-TDPS|2020-TDPS-The Deep Learning Compiler A Comprehensive Survey]]: Sec4-Sec7
    Sec4-Key Components of DL Compilers
        Sec4.1-High-level IR
            High-level IR is also known as graph IR.
            4.1.1 Representation of Graph IR
            The representation of graph IR can be categorized into two classes: DAG-based IR, Let-binding-based IR.
            In DAG-based IR, the node represents atomic DL operator, the edge represents tensor.
            In DAG-based IR, the graph is acyclic without loops, therefore differs from the data dependency graph of generic compilers.
            There are already plenty of optimizations on DDG, like Common Subexpression Elimination, Dead Code Elimination. These algorithm can be combined with DL domain knowledge to optimize the DAG computation graph.
            DAG-based IR is simple, but may cause semantic ambiguity because of missing the definition of the computation scope.
            Let-binding offers let-expression to certain functions with restricted scope in high-level language to solve semantic ambiguity. 
            When using `let` keyword to define expression, a let node will be generated which points to the operator and the variable in the expression, instead of just building the computational relation between variables in DAG.
            In DAG-based compiler, to get the return value of certain expression, the corresponding node will the accessed and the related nodes will be searched. It is known as recursive descent technique.
            The let-binding based compiler will compute the results of all variables in a let expression, and builds a variable map. The compiler looks up this map to decide the value of the expression.
            TVM and Relay IR adopts both.
            The ways graph IR to represent tensor computation can be categorized into three classes: Function-based, Lambda expression, Einstein notation
            Glow, nGraph, XLA's IR (XLA's IR is HLO) use function-based representation to represent tensor computation. The function-based representation only provides encapsulated operators.
            Lambda expression uses variable binding and substitution to represent calculation. TVM uses tensor expression to represent tensor computation, in which the computational operator are defined by the output tensor's shape and the lambda expression of computing rules.
            Einstein notation is used to expression summation, in which the indexes for temporary variables do not need to be defined. The actual expression can be deduced from the Einstein notation. In Einstein notation, the operators should be associative and commutive, and thus the reduction operators can be executed by any order.
            4.1.2 Implementation of Graph IR
            Data representation
            Tensor can be represented by placeholder which only carries the shape information of the tensor. It helps separate the computation definition and actual computing. To support dynamic shape/dynamic model, placeholder should support unknown dimension size. Also, the bound inference and dimension checking should be relaxed, and extra mechanism is needed to guarantee memory validity.
            Data layout describes tensor's organization in memory, which is usually a mapping from logical indices to memory indices. The data layout includes the sequence of dimensions, tilling, padding, striding, etc.
            Bound inference is used to determine the bound of iterators when compiling DL models. The bound inference is often performed iteratively of recursively according to the computation graph and placeholders.
            Operators supported
            Operators supported by DL compilers will be the node in the computation graph.
            4.1.3 Discussion
            The data and operators designed in high-level IR are flexible and extensible enough to support diverse DL models. The high-level IRs are hardware-independent.
        Sec4.2-Low-level IR
            Low-level IR provides interface to tune the computation and memory access. The common implementation of low-level IR can be classified into 3 categories: Halide-based IR, polyhedral-based IR, other unique IR.
            Halide-based IR
            Halide's philosophy is to separate computation and schedule. Compilers adopting Halide try various possible schedules and choose the best one. TVM improved Halide-IR to independent symbolic IR.
            Polyhedral-based IR
            Different from Halide, the boundaries of memory bounds and loop nests can be polyhedrons with any shapes in the polyhedral model. The polyhedral-based IR makes it easy to apply polyhedral transformations (fusion, tiling, sinking, mapping).
            Other unique IR
            MLIR has a flexible type system and allows multiple abstraction levels. It induces dialects to represent multiple levels of abstraction. Each dialect consists of a set of defined immutable operations. Current dialects includes TensorFlow IR, XLA HLO IR, experimental polyhedral IR, LLVM IR, TensorFlow Lite.
            Most DL compiler's low-level IR will eventually lowered to LLVM IR to use LLVM's optimizer and code generator. LLVM also supports custom instruction set for specialized accelerator.
            DL compiler adopts two approaches to achieve hardware-dependent optimization: 1. perform target-specific loop transformation in the upper IR of LLVM 2. provide additional information about the hardware target for optimization passes.
        Sec4.3-Frontend optimizations
            Frontend optimizations are shared by different backends.
            The frontend optimizations are defined by passes. The passes traverse the graph's nodes and perform graph transformation. (rewrite the graph for optimization)
            Passes can be pre-defined or customized by developers. Most DL compliers can capture shape information in computation graph to do optimization.
            The frontend optimization can be classified into three categories: 1. node-level 2. block-level (local) 3. dataflow-level (global)
            Node-level optimizations
            The nodes of computation are coarse enough to enable optimizations inside a node.
            Node elimination: eliminate unnecessary nodes (e.g. operations lacking adequate inputs, zero-dim-tensor elimination)
            Node replacement: use lower-cost nodes
            Block-level optimizations
            Algebraic simplification: optimization in computation order, optimization in node combination, optimization of ReduceMean nodes
            Operator fusion: 
            Operator sinking: make similar operations closer in order to create opportunities for algebraic simplification 
            Dataflow-level optimizations
            CSE: use previously computed sub-expression's value to substitute other occurrences of that sub-expression in the graph
            DCE: a set of code is dead if it's computation result or side-effect are not used; DCE includes dead store elimination (remove never used tensor's storage operation)
            Static memory planning: done offline, aims to reuse memory as much as possible; in-place memory sharing: allocate only one copy for operation, for sharing between the input and output; standard memory sharing: reuse previous operations' memory without overlapping.
            Layout transformation: aims to find the best data layout for tensors in the computation and insert layout transformation node into the computation graph. The actual layout transformation is preformed by the backend. To find the best data layout, the hardware details are required, like cache line size, vectorization unit size, memory access pattern etc.
        Sec4.4-Backend optimizations
            Hardware-specific optimization
            includes: 
            1. hardware intrinsic mapping: transform a certain set of low-level IR instructions to highly optimized kernels in target hardware.
            2. memory allocation and fetching: 
            3. memory latency hiding: reorder the execution pipeline
            4. loop oriented optimizations: includes 1) loop fusion to fuse loops with the same boundaries 2) sliding windows 3) tiling: the tiling patter and size can be determined by auto-tuning 4) loop reordering (loop permutation): changes the order of iterations in a nested loop to increase spatial locality. It requires the loop is free of data-flow dependency between iterations 5) loop unrolling: usually applied in combination with loop split: first split the loop into two nested loops and unroll the inner loop
            5. Parallelization: utilizes accelerator's multi-thread and SIMD parallelism.
            Auto-tuning
            four key components:
            1. parameterization: the data parameter describes the data's specification; the target parameter describes hardware-specific characteristics (e.g.  shared memory and register size) and constraints
            2. cost model: 1) black-box model: only considers the final execution time 2) ML-based cost model: e.g. GBDT 3) pre-defined cost model
            3. searching technique: 
            4. acceleration: 1) parallelization 2) configuration reuse
            Optimized kernel libraries 
    Sec5-Taxonomy of DL Compilers
    Sec6-Evaluation
    Sec7-Conclusion and Future Directions

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH18.1-CH18.3
    CH18-Structure Learning in Bayesian Networks
        CH18.1-Introduction
            In structure learning, we aims to recover $\mathcal G^*$ or its I-equivalence based on data. $\mathcal G^*$ is $P^*$ 's perfect map.
            The more edges our structure have, the more parameters we need to learn. Because of data fragmentation, the quality of estimated parameter will decrease if the number of samples is fixed. (Note that the standard deviation of MLE estimate if $1/\sqrt M$)
            Thus when doing density estimation from limited data, we prefer sparse structure even if the true structure $\mathcal G^*$ is more dense. Because we need to avoid overfitting.
            There are three methods for structure learning. 
            The first one is constraint-based structure learning, which tests the independence in data and find a network that best explains these independencies. This type of method is sensitive to failures in individual independencies test. If one of these tests return a wrong answer, the network construction will be misled.
            The second one is score-based structure learning. This method defines a hypothesis space of potential models and a score function that measures how well the model fits the observation. The task is to search the model that maximize the score in the hypothesis space. Score-based method consider the whole structure at once, thus is less sensitive to individual failures.
            The third method does not learn a single model but an ensemble of multiple possible structures.
        CH18.2-Constraint-Based Approaches
            Determining whether two variables are independent is often referred to as hypothesis testing.
        CH18.3-Structure Scores
            Score-based methods approach the problem of structure learning as an optimization problem.
            Intuitively, we need to find a model that would make the data as probable as possible. In this case, our model is pair $\langle \mathcal G,\pmb \theta_{\mathcal G}\rangle$. The likelihood score directly defines $\pmb \theta_{\mathcal G}$ to be its the MLE estimation $\hat {\pmb \theta}_{\mathcal G}$, and tries to find structure $\mathcal G$ that maximize $score_{L}(\mathcal G:\mathcal D) = \ell(\hat {\pmb \theta}_{\mathcal G}:\mathcal G)$.
            The likelihood score can decompose according to (18.4). We can observe that the likelihood measures the strength of the dependencies between variables and their parents. 
            For BN, the process of choosing a network structure is often subject to constraints. Some constraints are a consequence of the acyclicity requirement, others may be due to a preference for simpler structures.
            Because the property of mutual information, adding edge to a network will never decrease its likelihood score. Thus likelihood score will result in fully connected network in most cases. Therefore, the likelihood score can not avoid overfitting.
            The Bayesian method put a distribution on possible structures $\mathcal G$ and is proportional to the posterior $P(\mathcal G\mid \mathcal D)$. The Bayesian score is defined as $score_B(\mathcal G: \mathcal D) = \log P(\mathcal D\mid \mathcal G) + \log P(\mathcal G)$.
            The calculation of marginal likelihood $P(\mathcal D\mid \mathcal G)$ need us to integrate the whole parameter space $\Theta_{\mathcal G}$. Therefore, we are measuring the expected likelihood, averaged over different possible choices of $\pmb \theta_{\mathcal G}$ decreasing the sensitivity of the likelihood to the particular choice of parameters. 
            Another perspective to explain the Bayesian score is derived from the holdout testing methods. The Bayesian score can be viewed as a form of prequential analysis, where each instance is evaluated in incremental order, and contributes both to our evaluation of the model and to our final model score. The sequence order can be arbitrary. According to (18.8), the marginal likelihood can be approximately viewed as the estimation of the model' expected likelihood in the underlying distribution.
            If the parameter's priors are in conjugate case, the marginal likelihood of a single variable's form can be easily written. As a consequence, the marginal likelihood of the dataset can be further written simpler according to (18.9).
            The Bayesian score for BN cane be decomposed under the assumption of parameter independence. The the local independence is also satisfied, (18.9) can be applied to substitute the local terms of the factorized Bayesian score.
            If $M\to \infty$, the $\log P(\mathcal D\mid \mathcal G)$ can be represented as Theorem 18.1. We can observe that the Bayesian score tends to trade off the likelihood (fit the data) and the model complexity. Omitting the constant term, we get the BIC score.
            The log-likelihood term increase linear to $M$, and the model complexity term increase log to $M$, therefore the emphasis on data fitting will increase as $M$.
            BIC score and the Bayesian score are consistent, which means with adequate data, the score will select $\mathcal G^*$. or its I-equivalence.
            Consistency is an asymptotic property, and thus it does not imply much about the properties of networks learned with limited amounts of data.
            We call the prior satisfies parameter modularity if two structure's local structure are the same, their prior will be the same
            Under parameter modularity, Bayesian score will be decomposable, and thus the searching can be done locally and separately.
            The likelihood score is naturally decomposable.
- [[book-notes/面向计算机科学的组合数学|面向计算机科学的组合数学]]: CH1.1-CH1.6, CH2.1-CH2.2, CH3.1-CH3.2, CH7
    CH1-排列组合
    CH2-鸽巢原理
    CH3-母函数
    CH7-Polya 计数理论

\[Doc\]
- [[doc-notes/mlir/Toy Tutorial|mlir/Toy Tutorial]]: CH1-CH2
    CH1-Toy Language and AST
        unranked tensor parameter: the dimension is unknown, and will be specialized at call sites
    CH2-Emitting Basic MLIR
        There is no closed set of attributes, operations, types in MLIR.
        MLIR's extensibility is supported by dialects, which groups operations, attributes, types under the the same abstraction level.
        MLIR's core computation and abstraction unit are operations, which can be used to represent all core IR structures in LLVM like instructions, globals, modules.
        Operation's results and arguments are SSA values.
        Operation's name is prefixed with dialect's name.
        Operation can have zero or multiple attributes.
        Concepts to model an operation includes: name, SSA arguments, attributes, result values' types, source location, successor blocks, regions.
        Note that every operation has an associated mandatory source location. The debug info in MLIR is core requirement.
        All IR elements (refer to the concepts that model an operation) can be customized in MLIR.
        In C++, a dialect is implemented as a derived class of `mlir::Dialect`. It's attributes, operations, types are registered by an initializer method called by the constructor.
        Using tablegen to declaratively define a dialect is more simple.
        `MLIRContext` only loads builtin dialects by default. Customized dialect should be passed to template method `loadDialect` to be explicitly loaded.
        In C++, an operation is defined as a derived class of `mlir::Op` using CRTP. CRTP means that `mlir::Op` is a template class, whose template argument is the operation class (its derived class). By CRTP, `mlir::Op` can know its derived class in compilation, and thus can safely use `static_cast` to invoke derived class's method to achieve polymorphism in compilation.
        `mlir::Op` can take optional traits as template arguments to represent operation's properties and invariants.
        Operation class can also define its method to provide additional verification beyond the attached traits.
        Operation class should define static `build` method in order to be invoked by the `bulider` class to generate this operation's instance from a set of input values.
        Operation's `build` method should populate a `mlir::OperationState` with its possible discrete elements.
        After defining the operation class, we can invoke `addOperation` with its template argument being the operation class to register this operation into the dialect.
        `Operation` class is used to generally model all operations, and it does not describe the properties and types of a particular operation. Thus it is used as a generic API.
        Each specific operation is a derived class of `Op` .
        `Op` act as a smart pointer wrapper of `Operation*` . All the data of an operation is stored in the referenced `Operation` class. The `Op` class is an interface/wrapper to interact with `Op`, and thus is usually passed by-value.
        A `Operation*` can be `dyn_cast` to the corresponding `Op` . 
        Operations can also be defined by tablegen. 

# 2025
## January
### Week 1
Date: 2024.12.30-2025.1.6

\[Paper\]
- [[paper-notes/Latent Dirichlet Allocation-2003-JMLR|2003-JMLR-Latent Dirichlet Allocation]]: All
    Sec1-Introduction
        This paper focuses on modeling collections of discrete data, and aims to find efficient and statistically meaningful representation of the member in the collections.
        A basic method of processing documents in a corpora is tf-idf scheme, whose basic idea is using a word's tf-idf value to represent a word's importance to a document, and using a tf-idf vector to represent a document.
        Tf-idf scheme's description length is not small, and thus reveal little inter and intra document statistical structure.
        LSI do SVD to tf-idf matrix to capture a linear subspace in the tf-idf feature space, thus achieving dimension reduction.
        Another method of modeling data is simply using maximum likelihood of Bayesian method to fit a generative model for the data.
        pLSI model view each word in a document as sampled from a mixture model, where the mixture component is the a conditional multinominal distribution over vocabulary given a topic. Each document is represented by a probability distribution over the topics. The words in the document can be essentially viewed as first sample a topic from the topic distribution and sample the word from the multinominal distribution given the topic. By this way, the document's representation is significantly reduced.
        The problem of pLSI is the lack of modeling generative probabilistic model for the topic's distribution. Thus the parameter increases linearly with the size of the corpus and it is not clear to assign topic distribution for new document.
        pLSI assumes exchangeability of words in a document and documents in a corpus. According to Finetti's representation theorem, any collection of exchangeable random variables can have a joint mixture distribution. This lead us to not only consider a mixture distribution over words, but also consider a mixture distribution over documents. 
    Sec2-Notation and terminology
        We aims to find a probabilistic model that not only assign high probability to members of the corpus but also assign high probability to other similar documents.
    Sec3-Latent Dirichlet Allocation
        LDA's basic idea is to represent a document as a random mixture of latent topics, where each topic is characterized by a multinominal distribution over words. Note the difference between LDA's representation and pLSI's representation is that LDA is represented by a **random** mixture of topics, which indicates there is a distribution of the topic mixture. While in pLSI, the documents' topic mixture is not modeled as random, thus being deterministic.
        In LDA, the document is generated by: first sample $N$ (word counts) from a Poisson distribution, next sample $\theta$ (topic mixture proportion/document representation) from a Dirichlet distribution with parameter $\alpha$, next for each word, sample a topic from the multinominal defined by $\theta$ and sample a word from the multinominal defined by the sampled topic.
        LDA assumes the topic number is known and fixed, and the word's multinominal is fixed and is to be estimated from the corpus.
        In LDA, given $\alpha$, topic mixture $\theta$ is a $k$ -dimensional Dirichlet random variable and take values in a $(k-1)$ -simplex, with $\theta_i \ge 0, \sum_{i=1}^k \theta_i = 1$. Topic mixture $\theta$ defines a multinominal over $k$ possible topics.
        There are three levels to the LDA representation. $\alpha, \beta$ are corpus-level parameters, sampled once when generating the corpus. $\theta$ are document-level parameters, sampled once for each document. $z, w$  are word-level variables, sampled once for each word. In LDA, a document can associate multiple topics.
        In LDA, to obtain a document's marginal probability, we need to marginalize out the topic variables $z$ for each word and marginalize out the topic mixtures $\theta$ for the document.
    Sec4-Relationship with other latent variables
        In unigram model, each word are drawn from a single multinomial.
        In mixture of unigram model, each document associates a topic, and each topic defines a multinomial over the vocabulary. The multinominal can be viewed as a representation of the topic. The difference between mixture of unigram model and LDA is that LDA allow a document exhibit multiple topics, and defines a distribution over the topics.
        pLSI allow a document associate multiple topics, but each document's topic mixture is deterministic. The number of parameters grow linear with the corpus size.
        LDA treat the topic mixture as a random variable, making generalizing to new document easy and removing the linear dependency of the parameter number.
    Sec5-Inference and Parameter Estimation
        In inference, we need to compute the posterior $p(\theta, \mathbf z\mid \mathbf w, \alpha, \beta)$. We construct a variational model representing the approximate posterior and optimize the KL divergence between the variational approximate posterior and the true posterior to find the optimal approximate posterior.
        Note that the optimization for variational parameter is conducted for fixed $\mathbf w$. Therefore the variational parameter is conditioned on the document. The variational Dirichlet parameter $\gamma$ can be viewed as the representation of the document.
        In parameter estimation, we use variational EM procedure to maximize the log-likelihood. The E-step finds the optimal variational parameters, which is the same as the inference. The M-step finds the optimal model parameters $\alpha, \beta$ by MLE or by Bayesian estimation (smoothing). 
    Sec6-Example
        The prior Dirichlet parameters subtracted from the posterior Dirichlet parameters indicate the expected number of words which were allocated to each topic for a particular document.
    Sec7-Applications and Empirical Results
        Perplexity monotonically decrease with the likelihood of test data and is algebraically equivalent to the inverse of the geometric mean per-word likelihood. Lower perplexity indicates higher test set likelihood and thud indicates better generalization performance.
    Sec8-Discussion

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH19.2.2.5, CH19.2.3, CH19.2.4
    CH19-Partially Observed Data
        CH19.2-Parameter Estimation
            CH19.2.2-Expectation Maximization
                CH19.2.2.5-Theoretical Foundations
                    Each iteration of the EM process can be viewed as maximizing an auxiliary function. Maximizing the auxiliary function will yield better log-likelihood.
                    For exponential family models, the expected log-likelihood is a linear function of the expected sufficient statistics. Thus to maximize the expected log-likelihood, we first derive the expected sufficient statistics, and then compute the parameters that maximize the expected log-likelihood. That's precisely the EM process.
                    In each EM iteration, we are actually optimizing a function of the parameter $\pmb \theta$ and the posterior choice $Q$. We define the energy functional associated with $P$ and $Q$ as $F[P, Q] = E_Q[\log \tilde P] + H_Q$. Then we can prove $\log Z = F[P, Q] + D(Q||P)$.
                    Let $P = P(\mathcal H\mid \mathcal D,\pmb \theta) = P(\mathcal H, \mathcal D\mid \pmb \theta)/P(\mathcal D\mid \pmb \theta)$. According to the previous conclusion, we can get $\ell(\pmb \theta : \mathcal D) = F[P, Q] + D(Q|| P)$. Therefore, $\ell(\pmb \theta: \mathcal D) = E_Q[\log P(\mathcal H, \mathcal D\mid \pmb \theta)] +H_Q+ D(Q|| P(\mathcal H\mid \mathcal D, \pmb \theta))$. That is, we get an equivalent form of the log-likelihood, which is written as a function of $Q$ and $\pmb \theta$.
                    Take a step further, we can get $\ell(\pmb \theta : \mathcal D) = E_Q[\ell(\pmb \theta :\langle \mathcal D, \mathcal H \rangle)] + H_Q +  D(Q||P(\mathcal H\mid \mathcal D, \pmb \theta))$.
                    Because entropy and KL divergence are non-negative, the expected log-likelihood is a lower bound of the actual log-likelihood. Also, the energy functional (expected log-likelihood + entropy term) is a tight lower bound of the actual log-likelihood.
                    EM procedure is actually optimize the energy functional in a coordinate ascent way. Given fixed parameter $\pmb \theta$, it first search the optimal $Q$ the minimize the KL divergence, and then, given fixed $Q$, it tries to find the optimal $\pmb \theta$ that maximize the expected log-likelihood.
                    Because the energy functional is a tight lower bound of the actual log-likelihood, improving the energy functional will guarantee to improve the log-likelihood. And the improvement is guaranteed to be as large as the energy functional's improvement.
                    For most learning problem, the log-likelihood is upper bounded, and because EM can monotonically improve the log-likelihood, EM is guaranteed to converge to a stationary point of the log-likelihood function.
            CH19.2.3-Comparisoin: Gradient Ascent versus EM
                EM and gradient ascent are both local, greedy in nature, and both guarantee to converge to a stationary point of the log-likelihood function.
            CH19.2.4-Approximate Inference
                When computing gradient approximately, the approximation error will dominate if the stationary point is close.
                There is no guarantee that approximate inference will find a local maxima, but in practice, approximation is unavoidable.
                In variational EM, the E-step is to find the optimal variational $Q$ for each instance. Each instance's optimal variational posterior is different. The algorithm is essentially performing coordinate-wise ascent alternating between optimization of $Q$ and $\pmb \theta$. Therefore it is not necessarily to take too many steps to find best variational $Q$ in one iteration.
                In variational EM, the energy functional is not necessarily a tight lower bound of the actual log-likelihood. It depends on the choice of the variational distribution family. Therefore, variational EM has no convergence guarantee and it is easy to get oscillations both within a E-step and over several steps.

### Week2
Date: 2025.1.6-2025.1.13

\[Paper\]
- [[paper-notes/precipitation-nowcasting/Skilful Nowcasting of Extreme Precipitation with NowcastNet-2023-Nature|2023-Nature-Skilful Nowcasting of Extreme Precipitation with NowcastNet]]: All
    Sec0-Abstract
        Pure physics-based method can not capture convective dynamics. Data-driven learning can not obey physical laws like advective conservation. NowcastNet unify physical-evolution scheme and conditional learning methods, and optimize forecast error end-to-end.
    Sec1-Introduction
        Weather radar echoes provide cloud observations at sub-2km spatial resolution and up to 5-min temporal resolution.
        DARTS and pySTEPS are based on advection scheme. They predict the future motion fields and intensity residuals from radar observations and iteratively advect the past radar field according to the predicted motion field, with the addition of intensity residuals, to obtain the future fields. The advection scheme respects the physical conservation laws, but does not account for the convective influence. Also, existing advection method does not incorporate nonlinear evolution simulation and end-to-end forecast error optimization.
        Deep learning based methods do not account for the physical laws explicitly, and thus may produce unnatural motion and intensity, high location error and large cloud dissipation at increasing lead times.
        NowcastNet combines deep learning methods and physical principles. It integrates advective conservation into a learning based model, and thus can predict long-lived mesoscale advective pattern and short-lived convective detail.
    Sec2-NowcastNet
        Given past radar fields $\mathbf x_{-T_0:0}$, the generative model generate future fields $\widehat {\mathbf x}_{1:T}$ from random Gaussian vector $\mathbf z$ conditioned on the evolution network's prediction.
        The random Gaussian vector $\mathbf z$ introduce randomness to the generation to capture chaotic dynamics. Integration over the latent vector enables ensemble forecast.
        The evolution network's prediction aims to comply with the physical advection process and produce physically plausible prediction for advective features at 20km scale. The generative network's aims to generate fine-grained prediction that captures convective features at 1-2km scale. This scale disentanglement mitigates error propagation between scales in multiscale prediction network.
        The physical-conditioning mechanism is implemented by spatially adaptive normalization technique. In forward pass, in each layer of nowcast decoder, the mean and variance of the activations are replaced by spatially corresponding statistics computed from the evolution network predictions. Therefore the nowcast decoder combines mesoscale advective pattern governed by the physical laws and convective-scale details revealed by radar observations.
        NowcastNet is trained in an adversarial way. The temporal discriminator on the nowcast decoder takes the pyramid features in several time windows as input and output the possibility that the input is fake or real radar field. By deceiving the discriminator, NowcastNet can learn to generate convective details present in the radar observation but left out by the advection-based evolution network.
        To make the generated field spatially consistent with the read observation, the loss also includes pool regularization term, which enforce pooling-level consistency between ensemble prediction and real observation.
    Sec3-Evolution network
        Previous implementation of advection schemes' disadvantages include: 1. advection operation is not differentiable 2.  can not provide nonlinear modelling ability 3. auto regressive generation prevents direct optimization of forecast error and accumulates estimation error of initial states, motion field, intensity residual.
        The evolution network's evolution operator is differentiable and directly optimize the forecast error throughout the time horizon by back propagation.
        The evolution operator takes the motion field, intensity residual and the current field as input and output the field in the next time step by one step of advection. The evolution operator can finally produce predictions $\mathbf x_{1:T}$ for several time steps. The gradient can be passed through the evolution operator, to directly optimize the motion decoder, intensity decoder, and the evolution encoder.
        To avoid numerical instability from discontinuous interpolation, in the evolution operator, the gradient between each time step is detached.
        The motion decoder and intensity decoder simultaneously predict motion fields and intensity residuals at all future time steps. 
        The objective of evolution network is minimizing the forecast error throughout the time horizon. The accumulated error term in the loss involves the distance between the predict field $\mathbf x_{t}''$ and the real field $\mathbf x_t$. It also involves the distance between the advected field $\mathbf x_t'$ and the real field $\mathbf x_t$ to short cut the gradient. It can be viewed as a residual shortcut. Therefore the intensity $\mathbf s_t$ in learning will tend to fit the residual between the real field and the advected field.
        To ensure continuity and to ensure the large precipitation pattern's motion field being more smooth than the small ones. The loss also involves a motion regularization term, which penalize the motion field's gradient norm in a weighted way.
    Sec4-Evaluation settings
        An importance sampling strategy is applied to create datasets more representative of extreme-precipitation events.
    Sec5-Precipitation events
    Sec6-Meterologist evaluation
    Sec7-Quantitive evaluation
    Sec8-Methods
        According to the continuity equation, the temporal evolution of precipitation can be modelled as a composition of advection by motion fields and addition by intensity residuals. The evolution operator is constructed in this principle, and the motion field and intensity residuals are predicted by neural networks based on past radar observations.
        The evolution network is responsible for predicts future radar fields at 20-km scale. The backbone of evolution network is a two-way U-Net, where each convolution layer is spectral normalized and the skip connections concatenate the temporal dimension/the channel dimension.
        The evolution operator views motion field $\mathbf v_{1:T}$ as departure offset and views intensity residual $\mathbf s_{1:T}$ precipitation intensity growth or decay.
        As applying bilinear interpolation continuously will blur the field. The advected field $\mathbf x_t'$ is computed by nearest interpolation, but the gradient and loss is computed from the bilinear interpolated field $(\mathbf x_t')_{\mathrm {bili}}$.
        Gradient between two consecutive time steps in the operator is detached because successive bilinear interpolation will make end-to-end optimization unstable.
        To balance different rainfall levels, the distance calculation in the accumulation loss is weighted proportional to the rain rate.
        The generative network is responsible for generating the final predicted precipitation field at a 1-2km scale. The backbone of generative network is also a U-Net encoder decoder structure. The encoder is identical to the evolution network's encoder, and it takes the concatenation of $\mathbf x_{-T_0:0}$ and $\mathbf x_{1: T}''$ as input. The decoder has different structure, and it takes encoder's encoded representation added by the transformed latent random vector as input.
        The conditioning mechanism is implemented by applying the spatially adaptive normalization to each convolutional layer in the decoder. 
        The pooling-regularization term is calculated from the ensembled prediction.
    Sec9-Datasets
    Sec10-Evaluation
\[Book\]
- [[book-notes/深度强化学习|深度强化学习]]: CH1

### Week 3
Date: 2025.1.13-2025.1.20

\[Paper\]
- [[paper-notes/normalization/Semantic Image Synthesis with Spatially-Adaptive Normalization-2019-CVPR|2019-CVPR-Semantic Image Synthesis with Spatially-Adaptive Normalization]]: All
    Sec0-Abstract  
        Directly use semantic layout as input to the network is suboptimal, because the normalization layer tend to wash away semantic information.
        Spatially-Adaptive Normalization use semantic layout to modulate the activations in the normalization layer spatially-adaptively.
    Sec1-Introduction
        Semantic image synthesis focuses on converting semantic segmentation mask to a photorealistic image.
        In traditional architecture, the normalization layer will wash away the information in the semantic mask, preventing semantic information propagating through the network.
    Sec2-Related Work
        Conditional normalization layers requires external data and generally first normalize activations to zero mean and unit deviation, then modulate the normalized activations using the external semantic input.
        Generating image from semantic mask requires applying a spatially-varying transformation/modulation to the normalized activations.
    Sec3-Semantic Image Synthesis
        SPatially Adaptive (DE) normalization first apply batch normalization, and then modulate the normalized activations with learned scale and bias spatially-adaptively. That is, the modulation parameters depend on the input mask and vary with respect to the location $(x, y)$.
        With SPADE, there is no need to feed the mask to the first layer of the generator. The generator can take a random vector as input to support multi-modal synthesis. That is, for the same mask, different random vector can leads to different but semantically consistent images.
        In SPADE, the semantic mask is fed through spatially adaptive modulation, without normalization. Therefore the semantic information is better preserved.
        By replacing the input noise with the embedding vector of the style image computed by the image encoder, we can further control the style of the synthesized image.
    Sec4-Experiments
    Sec5-Conclusion
- [[paper-notes/precipitation-nowcasting/Skilful Precipitation Nowcasting Using Deep Generative Models of Radar-2021-Nature|2021-Nature-Skilful Precipitation Nowcasting Using Deep Generative Models of Radar]]: All
    Sec0-Abstract
        Operational nowcasting methods typically advect precipitation fields with radar-based wind estimates, and struggle to capture non-linear events like convective initiations.
        Deep learning methods directly predict future rain rates, free of physical constraints. Deep learning methods can predict low-intensity rainfall, while the lack of constraints lead to blurry prediction at longer lead time and heavier rain events.
    Sec1-Introduction
        Radar data is available every five minutes at 1km x 1km grid resolution.
        Advective methods rely on the advection equation, using optical flow and smoothness penalty to estimate motion field.
        Deep learning based models are directly trained on large corpora of radar observations and do not rely on in-built physical assumptions. Deep learning based model conduct optimization end-to-end and has fewer inductive biases, therefore greatly improve forecast quality at low precipitation levels.

### Week4
Date: 2025.1.20-2025.1.27

\[Doc\]
- [[doc-notes/onnx/Introduction to ONNX|onnx/Introduction to ONNX]]: All
- [[doc-notes/pytorch/tutorials/beginner/ONNX|pytorch/tutorials/beginner/ONNX]]: All