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
        1. Avoid Bank Conflict: 数据访问对齐 32bit 的 Bank Size
        2. Occupancy Calculator: 利用工具计算一下合适的 Blocksize 和 Gridsize
        3. Coalescing: Warp 对 Shared Memory 的访问同样可以进行合并优化
        4. `memcpy_async()` (Questioned): 异步访问，访存与计算流水线
- [[Managing Projects with GNU Make-2011|Managing Projects with GNU Make]]: CH1-CH2.7

\[Doc\]
- [[doc-notes/nvidia/NVIDIA Nsight Compute|nvidia/NVIDIA Nsight Compute]]: CH2

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
- [[doc-notes/nvidia/CUDA-GDB v12.6|nvidia/CUDA-GDB v12.6]]: CH1-CH8

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
    CH2-Getting Started on Your Computer：CMake 构建流程纵览
    CH3-Writing CMakeLists Files：CMake 语言纵览
    CH4-CMake Cache：CMake 缓存机制介绍：`CMakeCache.txt`
    CH5-Key Concepts：CMake 概念介绍：源文件、目标文件、属性
    CH6-Policies：CMake 策略机制：为了兼容性
    CH7-Modules：CMake 模块：CMake 提供的 utility

## September
### Week 1
Date: 2024.9.2-2024.9.9

\[Book\] 
- [[Mastering CMake]]: CH8-CH13、CH14 (CMake Tutorial)
    CH8-Installing Files：`Install()` 命令
    CH9-System Inspections：借助宏编写跨平台软件；`try_run/compile()` 命令
    CH10-Finding Packages：借助 CMake 分发的软件依赖包
    CH11-Custom Commands：为自定义目标添加自定义构建规则
    CH12-Packing with CPack：借助 CPack 调用本地打包工具
    CH13-Testing with CMake and CTest：`add_test()`
    CH14-CMake Tutorial：纵览：简单构建、添加库、添加属性、通过系统审查添加宏、添加测试、简单安装、添加共享库、生成器表达式（实际构建时才确定值的变量）、导出 CMake 软件包

### Week 2
Date: 2024.9.9-2024.9.16

\[Paper\]
- [[paper-notes/llm/A Survey of Large Language Models v13-2023|2023-A Survey of Large Language Models v13]]: Sec6
    Sec6-Utilization
        Prompt tricks: (input-output) pair, (input-reasoning step-output) triplet, plan
- [[Are Emergent Abilities of Large Language Models a Mirage-2023-NeurIPS|2023-NeurIPS-Are Emergent Abilities of Large Language Models a Mirage]]: All
    研究者对于度量的选取造就了 LLM 具有涌现能力的“海市蜃楼” 

\[Book\]
- [[book-notes/Introductory Combinatorics-2009|Introductory Combinatorics]]: CH1
    CH1-What is Combinatorics
        Combinatorics: existence, enumeration, analysis, optimization of discrete/finite structures
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH2
    CH2-Foundations
        Basic knowledges: Conditional Independence, MAP query, Conditional density function, graphs

\[Doc\]
- [[doc-notes/Intel NPU Acceleration Library Documentation v1.3.0|Intel NPU Acceleration Library Documentation v1.3.0]]
- [[doc-notes/python/The Python Tutorial|python/The Python Tutorial]]: CH1-CH16

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
- [[Importance Sampling A Review-2010|2010-Importance Sampling A Review]]: All
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
- [[paper-notes/mlsys/FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeurIPS|2022-NeurIPS-FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness]]: Sec0-Sec3.1
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
- [[doc-notes/python/packages/ultralytics|python/packages/ultralytics]] : Quickstart, Usage (Python usage, Callbacks, Configuration, Simple Utilities, Advanced Customization)
    Brief Introduction to YOLO model's python API, which is pretty simple

### Week 3
Date: 2024.10.14-2024.10.21

\[Paper\]
- [[paper-notes/mlsys/FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeurIPS|2022-NeurIPS-FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness]]: Sec3.1-Sec5
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
- [[doc-notes/pytorch/Pytorch 2.x|pytorch/Pytorch 2.x]]: CH0
    CH0-General Introduction
        `torch.compile` : TorchDynamo --> FX Graph in Torch IR --> AOTAutograd --> FX graph in Aten/Prims IR --> TorchInductor --> Triton code/OpenMP code...
- [[doc-notes/triton/Getting Started|triton/Getting Started]]: Vector Addition, Fused Softmax
    Triton is basically simplified CUDA in python, the general idea about parallel computing is similar. The most advantageous perspective about Triton is that it encapsulates all the complicated memory address mapping work into a single api `tl.load` . Memory address mapping work is the most difficult part of writing CUDA code.

### Week 4
Date: 2024.10.21-2024.10.28

\[Paper\]
- [[paper-notes/mlsys/FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeurIPS|2022-NeurIPS-FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness]]: SecA-SecE
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
- [[doc-notes/pytorch/tutorials/beginner/Learn the Basics|pytorch/tutorials/beginner/Learn the Basics]] 
- [[doc-notes/python/packages/pillow|python/packages/pillow]] : Overview, Tutorial, Concepts
- [[doc-notes/huggingface/hub/Repositories|huggingface/hub/Repositories]]: Sec1-Sec4
- [[doc-notes/triton/Getting Started|triton/Getting Started]]:  Tutorials/Matrix Multiply
- [[doc-notes/python/howto/general/Argparse Tutorial|python/howto/general/Argparse Tutorial]] 
- [[doc-notes/nvidia/CUDA C++ Programming Guide v12.6|nvidia/CUDA C++ Programming Guide v12.6]]: CH1

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
        Prompt phase: Process user input, generating KV cache. Use Matrix-Matrix multiplication (use masked self-attention to implement causal attention).
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
- [[doc-notes/python/howto/general/Annotations Best Practices|python/howto/general/Annotations Best Practices]]
    Best Practice after Python 3.10: use `inspect.get_annotations()` to get any object's annotation
- [[doc-notes/huggingface/hub/Repositories|huggingface/hub/Repositories]]: Sec4-Sec10

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
- [[doc-notes/nvidia/CUDA C++ Programming Guide v12.6|nvidia/CUDA C++ Programming Guide v12.6]]: CH2
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
- [[doc-notes/docker/get-started/What is Docker|docker/get-started/What is Docker]] 
    Containers include everything needed for running an application
    Use containers to be the unit of distributing and deploying applications
    Docker client ( `docker` ) use Docker API to communicate with Docker daemon ( `dockerd` ), which is responsible for managing containers
    Docker registry stores images. `docker pull` pulls image from registry, and `docker push` pushes image to registry
    Image is an read-only template of instructions for creating container. Image is defined by Dockerfile, and is consists of layers. Each instruction in Dockerfile defines a layer in image. Once created, the image can not be modified. Container is a runnable instance of an image.
- [[doc-notes/docker/get-started/Docker Concepts|docker/get-started/Docker Concepts]]
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
- [[doc-notes/huggingface/hub/Models|huggingface/hub/Models]]: Sec0-Sec1
- [[doc-notes/python/pep/PEP 8-Style Guide for Python Code|python/pep/PEP 8-Style Guide for Python Code]]

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
            The MCMC method defines a Markov chain whose stationary distribution is the desire sampling distribution $P$, and make the sample generated from the initial distribution (usually the prior) keep evolve its assignment (state) in the Markov chain. Because the Markov chain's state distribution will eventually converge to its stationary distribution, the distribution that the sample conforms to will eventually converge to the desired sampling distribution. Thus we can eventually treat the sample as sampled from the desired sampling distribution. In the process, the sample's distribution will gradually get closer to the desired sampling distribution.
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
- [[doc-notes/python/pep/PEP 257-Docstring Conventions|python/pep/PEP 257-Docstring Conventions]]

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
- [[doc-notes/docker/get-started/Docker Concepts|docker/get-started/Docker Concepts]]
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
Date: 2024.12.16-2024.12.23
Date: 2024.12.23-2024.12.30

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
- [[doc-notes/mlir/code-documentation/tutorials/Toy Tutorial|mlir/code-documentation/tutorials/Toy Tutorial]]: CH1-CH2
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

### Week 2
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
    CH1-机器学习基础
        CH1.1-线性模型
            偏置与属性值无关，可以认为是本身存在的知识/事实
            正则化项等于权重的 L2 范数的平方称为岭回归，等于权重的 L1 范数称为 LASSO，正则化项系数可以在验证集上通过交叉验证选取
            逻辑斯蒂回归模型 = 线性层 + Sigmoid 激活层，激活层用于输出概率值
            分布的熵等于分布和分布本身的交叉熵，并且当 $q = p$ 时，交叉熵取到最小值等于分布 $p$ 的熵
            KL 散度即 $q$ 相对于 $p$ 的交叉熵减去 $p$ 本身的熵，KL 散度一定非负，分布 $p$ 固定时，最小化 KL 散度等价于最小化交叉熵
            Softmax 分类 = 线性层 + Softmax 激活层，Softmax 激活将 Sigmoid 拓展到向量输入
        CH1.2-神经网络
            全连接层 = 线性层 + 激活函数
            卷积层接受三阶张量作为输入，输出三阶张量
            目标函数为实数时，它相对于张量的梯度的形状和张量本身相同
            随机梯度下降 SGD 不计算整个数据集上的梯度，而是计算随机抽取的一个样本的梯度，相对于普通梯度下降，SGD 更容易跳出鞍点
            链式法则对于向量值输入输出的函数也成立，只不过需要使用矩阵乘法，同时乘法顺序颠倒了

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
        This paper develops an observation-driven approach for probabilistic nowcasting using deep generative models (DGMs). DGMs are able to simulate many samples from the conditional distribution of future radar given historical radar, generating a collection of forecasts similar to ensemble methods.
        DGMs can predict small-scale weather phenomena that are inherently difficult to predict because of the underlying stochasticity.
    Sec2-Generative models of radar
        DGMR is a conditional generative model, generating future $N$ radar observations fields given the past $M$ ones.
        In training, 4 consecutive radar observation frames are feed into the generator. The generator samples multiple realizations of future precipitation, each of which consists of 18 consecutive future radar frames.
        The objective consists of two losses and one regularization term. The first loss is defined by the spatial discriminator, whose aim is to distinguish individual generated fields with observed fields, ensuring spatial consistency and discouraging blurry predictions. The second loss is defined by the temporal discriminator, whose aim is to distinguish generated sequence and observed sequence, imposing temporal consistency and penalize jumpy predictions.
        The spatial discriminator is a 2D convolutional neural network, and the temporal is a 3D convolutional neural network.
        When used alone, these losses yield accuracy on par with the Eulerian persistence model. The regularization term, which penalizes deviations at grid cell resolution between real radar sequence and the model prediction mean (computed with multiple samples), is introduced to improve accuracy.
        Ablation study shows the importance for generating location-accurate predictions.
        DGMR is trained on a large corpus of precipitation events, which are 256 x 256 crops extracted from the radar stream, of length 22 frames. An importance sampling scheme is used to create a dataset more representative of heavy precipitations.
    Sec3-Intercomparison case study
    Sec4-Forecast skill evaluation
    Sec5-Forecast value evaluation
    Sec6-Conclusion
        The prediction of heavy precipitation at long lead times remains difficult for all approaches.
    Sec7-Methods
        Sec7.1-Datasets
            Most radar composites little to no rain. Medium to heavy precipitation comprises fewer than 0.4% of grid cells in the dataset. Therefore the dataset is rebalanced to include more data with heavier precipitation radar observations.
            Each example in the dataset is a sequence of 24 radar observations of size 1536 x 1280. 256 x 256 crops are extracted and an importance sampling scheme is imposed to reduce the number of examples containing little precipitation.
        Sec7.2-Model details and baselines
            The expectation over latent variables are estimated by Monte Carlo estimation. Per input samples 6 latent variables.
            During evaluation, full radar observation of size 1535 x 1280 and latent variables with height and width 1/32 of radar observation is used as inputs. 
        Sec7.3-Performance evaluation

### Week 4
Date: 2025.1.20-2025.1.27

\[Doc\]
- [[doc-notes/onnx/Introduction to ONNX|onnx/Introduction to ONNX]]: All
    ONNX Concepts
        A ML model implemented with ONNX is referred as an ONNX graph.
        `onnx` implements a python runtime to evaluate ONNX models and ONNX ops.
        In ONNX graph, each node has its type, which is one of the ONNX operators.
        Inputs that never change can be stored in the graph as initializers.
        Operators' attributes refer to its fixed parameters. They can not be changed once the ONNX graph is loaded.
        ONNX uses protobuf to serialize the graph into one single block.
        The main list of ONNX operators is within the `ai.onnx` domain. Another domain `ai.onnx.ml` includes more machine learning related operators. ONNX only officially define these two domains.
        An ONNX tensor is defined by its element type, shape and contiguous array. The array should be a full dense array with no stride.
        ONNX is strongly typed and dost not support implicit cast. Therefore, to add two tensors with different types, an explicit cast must be inserted in a graph.
        The version of the opset is incremented with the minor version of `onnx` package. An opset is attached to every ONNX graphs. It defines the versions of all operators in the graph.
        If the graph contains operators from several domains, the graph should define a global opset for each domain.
        Tests and loops are implemented by operator `If`, `Scan`, and `Loop`. They all take another ONNX graph as an attribute. 
        `If` executes one of the two graphs depending on the condition evaluation. These two graphs should product the same number of outputs.
        `Scan` implements a loop with a fixed number of iterations. It loops over an axis of the input and concatenates the outputs along this axis.
        `Loop` can do a fixed number of iterations or depending on a condition. Outputs can be concatenates along an axis as `Scan` does or be concatenates into a sequence of tensors.
        Function is defined with existing operators. Once defined, functions can behave like operators.
    ONNX with Python
        Every object in onnx can be serialized to byte stream with method `SerializeToSting` 
        ONNX essentially treats initializer as default values for the corresponding inputs.
        Operator `Constant` is the only operator changing an attribute into an input.
    Converters
- [[doc-notes/pytorch/tutorials/beginner/ONNX|pytorch/tutorials/beginner/ONNX]]: Introduction to ONNX, Export a PyTorch model to ONNX
    Introduction to ONNX
        `torch.onnx.dynamo_export` use TorchDynamo to hook into Python's frame evaluation API and dynamically rewrites its bytecode into an FX graph. The FX graph is polished and converted into an ONNX graph.  
    Export a PyTorch model to ONNX

\[Code\]
- NowcastNet rewritten project

## February
### Week 1
Date: 2025.1.27-2025.2.3

\[Doc\]
- [[doc-notes/pytorch/tutorials/beginner/Saving and Loading Models|pytorch/tutorials/beginner/Saving and Loading Models]]: All
    `torch.save` first utilize `pickle` for serialization, and then saves the serialized object to disk.
    `torch.load` utilize `pickle` for deserialization pickled object to memory.
    `torch.nn.Module.load_state_dict` accept a deserialized `state_dict` (essentially a Python dictionary object) and load a model's parameters. Setting its keyword parameter `strict=False` can ignore non-matching keys.

\[Code\]
- NowcastNet rewritten project

### Week 2
Date: 2025.2.3-2025.2.10

\[Book\]
- [[book-notes/深度强化学习|深度强化学习]]: CH2-CH3
    CH2-蒙特卡洛
    CH3-强化学习基本概念
        CH3.1-马尔可夫决策过程
            强化学习的主体是智能体，智能体做出决策，执行动作
            环境是与智能体交互的对象，环境在每个时刻具有一个状态，环境的状态是智能体做出决策的依据
            状态空间指环境所有可能状态的集合，记作 $\mathcal S$
            动作空间是智能体所有可能动作的集合，记作 $\mathcal A$
            奖励是智能体执行动作后，环境给智能体的反馈，一般建模为关于当前状态、动作、下一时刻状态的函数，平稳的奖励函数不随时间变化
            状态转移指智能体执行动作后，环境状态的变化，一般由状态转移概率函数定义 (和当前状态、动作相关)，一般也假设为平稳的
        CH3.2-策略
            策略指根据智能体根据状态做出决策，RL 的目标即得到策略函数，Markov 性质的策略函数仅依赖于当前状态，与历史状态无关
            智能体和环境交互的过程为：智能体根据当前状态 $s$，依据策略函数 $\pi(a\mid s)$ 决策，环境根据动作 $a$ 和当前状态 $s$，依据状态转移函数 $p(s'\mid s, a)$ 更新状态，并给出奖励 $r(a, s, s')$
            回合指智能体从游戏开始到游戏结束的整个过程
        CH3.3-随机性的来源
            强化学习的随即性来源于策略函数 $\pi(a\mid s)$ 和状态转移函数 $p(s'\mid s, a)$
            动作的随机性来源于策略函数，状态的随机性来源于状态转移函数
            奖励的随机性来源于未观测到的动作和状态
            轨迹指智能体一回合中观测到的所有状态、动作、奖励
        CH3.4-回报与折扣回报
            回报指当前时刻到回合结束的奖励总和
            折扣回报为未来的奖励乘上折扣因子，无限期 MDP 中，折扣因子和奖励函数的有界性一起保证了折扣回报的收敛性
            回报中的随机性来自于未观测到的奖励
        CH3.5-价值函数
            观测到当前状态和动作的条件下，回报的期望称为动作价值函数，该函数依赖于当前状态和动作，以及策略
            最优动作价值函数是所有策略中最优的策略定义的动作价值函数，该函数依赖于当前状态和动作
            观测到当前状态的条件下，回报的期望称为状态价值函数，该函数依赖于当前状态，以及策略

\[Doc\]
- [[doc-notes/onnx/API Reference|onnx/API Reference]]: Index, Protos, Serialization
    Index
        Each ONNX object is defined based on a protobuf message, and has name ended with suffix `Proto`
    Protos
        It is recommended to use functions in module `onnx.helper` to create Protos instead of explicitly instantiate them. All Protos can be printed by `print()` and will be rendered as a json string.
        ModelProto is a top level file/container format for bundling a ML model and associating its computation graph with its meta data.
        NodeProto defines an operators.
    Serialization
        Every Proto class implements method `SerializeToSting` . 
        Protobuf does not store any information about the class of the saved data. The target class must be known before restoring from an object.
- [[doc-notes/pytorch/tutorials/advanced/Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime|pytorch/tutorials/advanced/Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime]]: All
    It is important to call `model.eval()` or `model.train(False)` to turn the model to inference mode before exporting the model, because some operators' behaviour is different in training and inference modes.

### Week 3
Date: 2025.2.10-2025.2.17

\[Book\]
- [[book-notes/深度强化学习|深度强化学习]]: CH4, CH5
    CH4-DQN 与 Q 学习
        最优动作价值函数 $Q_*(s, a)$ 可以用深度神经网络 $Q(s, a;\pmb w)$ 近似，该网络称为 DQN，DQN 的训练目标就是让近似函数 $Q(s, a;\pmb w)$ 尽可能接近最优动作价值函数 $Q_*(s, a)$
        实践中，DQN 接受当前状态 $s$，直接输出 $|\mathcal A|$ 维的向量，给出了在动作空间所有可能动作下 $Q(s, a;\pmb w)$ 的预测
        DQN 最常用的训练算法是时间差分 TD 算法，本质上 DQN 的训练目标就是让近似函数 $Q(s, a;\pmb w)$ 尽可能满足近似形式的最优 Bellman 方程，该近似方程的 RHS 就是 TD 目标，我们希望近似方程的 LHS 即 DQN 的预测尽可能接近 TD 目标，故损失函数可以定义为 DQN 的预测和 TD 目标的平方差，即 TD 误差的平方
        DQN 依照 TD 算法训练时，需要的数据是四元组 $(s_t, a_t, r_t, s_{t+1})$，DQN 的训练和具体策略 $\pi$ 无关 (因为都是取最大)，唯一需要收集的事实反馈就是奖励，故收集数据时可以使用任意行为策略，例如 $\epsilon$ -greedy 策略
        存储四元组 $(s_t, a_t, r_t, s_{t+1})$ 的数组称为经验回放数组
        数据收集和 DQN 参数更新不一定要分离，可以同时进行，即收集一部分数据就更新
        Q Learning 算法的目标是学习最优动作价值函数 $Q_*$，SARSA 算法的目标是学习特定策略的动作价值函数 $Q_\pi$
        对于有限的状态空间和动作空间，最优动作价值函数 $Q_*(s, a)$ 可以表示为一个表格，每个状态-动作对都对应到表格中的一个确切的值，给定状态和动作，对应的价值可以直接在表格中搜索得到。在表格表示法下，Q Learning 的学习目标就是表格中的每一项数值，即直接根据 TD 误差对表格中的各项数值进行更新
        同策略指用于收集经验/数据的行为策略和目标策略相同，异策略就是不同。使用 $\epsilon$ -greedy 作为行为策略就是异策略，像 $\epsilon$ -greedy 这样的带有随机性的行为策略的好处是可以探索更多的状态，同时可以通过让 $\epsilon$ 衰减来让随机性衰减
    CH5-SARSA 算法
        策略 $\pi$ 定义的动作价值函数 $Q_\pi(s, a)$ 可以用于判断策略的好坏
        SARAS 算法的思想依旧是让近似函数 $q(s,a;\pmb w)$ 尽可能满足近似形式的 Bellman 方程，和 Q Learning 的差异在于，此时的近似 Bellman 方程涉及到了动作的采样值 $\tilde a_{t+1}$，该动作依赖于策略 $\pi$ 而采样
        SARSA 算法需要的数据是五元组 $(s_t, a_t, r_t, s_{t+1}, \tilde a_{t+1})$，因此 SRSA 算法的行为策略只能是目标策略 $\pi$ 本身，不能任意
        单步 TD 的 SARSA 算法可以进一步推广到多步 TD，其本质同样依赖于近似的 Bellman 方程
        多步 TD 介于蒙特卡洛和自举之间，蒙特卡洛即采样到结束，自举即单步 TD，蒙特卡洛的好处是 $U_t$ 的估计是无偏的，坏处是方差大，自举的好处是采样少，方差少，坏处是有偏差
- [[book-notes/Reinforcement Learning An Introduction|Reinforcement Learning An Introduction]]: CH3.6-CH3.9
    CH3-Finite Markov Decision Process
        The Bellman equation average over all possibilities, weighting each by its probability of occurring. It states that the value of the start state must equal to the discounted value of the expected next state plus the expected reward along the way.
        The value function $v_\pi$ is the only solution to its Bellman equation.
        Bellman optimality equation states that the value of a state under an optimal policy equals to the expected return (value) for the best action from that state.
        For finite MDP, Bellman optimality equation has a unique solution independent of the policy.
        Bellman equation is actually a system of equations, one for each state. If the environment dynamics is known, the system of equation can be solved in principle.
        After solving out the optimal value function, the optimal policy can be defined accordingly. Any policy that is greedy with respect to the optimal value function is optimal, because the optimal value function has considered all the possibilities in the future.
        Explicitly solving the Bellman equation system requires three assumptions to hold: 1. environment dynamics known 2. sufficient computational resource 3. Markov property holds, which are generally violated in practice. Also, the cardinality of the state set is exponentially large. Therefore, explicitly solving the Bellman equation system is not realistic. We have to resort to approximate approaches.

\[Doc\]
- [[doc-notes/pytorch/docs/developer-notes/Reproducibility|pytorch/docs/developer-notes/Reproducibility]]: All
    `torch.backends.cudnn.benchmark=False` will disable the benchmarking feature of cuDNN.
    `torch.use_deterministic_algorithms()` will cause use the deterministic alternative of non-deterministic operations.

### Week 4
Date: 2025.2.17-2025.2.24

\[Paper\]
- [[paper-notes/distributed-system/MapReduce Simplified Data Processing on Large Clusters-2004-OSDI|2004-OSDI-MapReduce Simplified Data Processing on Large Clusters]]: All
    Abstract
        In MapReduce, the user specified `map` function should process a key-value pair to generate a set of intermediate key-value pairs, the user specified `reduce` function should merge all intermediate values with the same associated key.
        The run-time system will take care of all the other distributed computing details, including data partition, program scheduling, failure handling and inter-machine communication, etc.
    Introduction
        MapReduce's primary mechanism for fault tolerance is re-execution.
        MapReduce defines an interface for automatic parallelization and distribution of large-scale computations, and provides an implementation of this interface.
    Programming Model
        The MapReduce computation takes a set of input key/value pairs and output an other set of key/value pairs. The users use `map` and `reduce` function to express the computation.
        MapReduce will group all intermediate key/value pairs with the same key and pass them to `reduce` . The intermediate values are actually supplied to `reduce` via an iterator, therefore allows handling large lists that can not fit into memory.
    Implementation
        Execution Overview
            The input data will be partition into M pieces, with each piece 16MB to 64MB. The intermediate key/value pairs will be partitioned into R pieces using a partitioning function. The M pieces correspond to M map task, and the R pieces correspond to R reduce tasks.
            MapReduce will start many copies of the program, with one of them being the master, and the others are all workers. The master is responsible to pick idle worker and assign an map or reduce a task to it.
            The worker assigned a map task will read the contents of the corresponding input data piece, parsing the data, and invoke `map` to produce intermediate key/value pairs into memory buffer. 
            The buffer will periodically write its content to local disk, which is partitioned into R regions by the partitioning function. The local location of these buffered pairs will be forwarded to master by the worker.
            The worker assigned a reduce task will use remote procedure calls to read the buffered data from the map workers' local disk. After finishing all the intermediate data, the worker sorts all the intermediate key/value pairs by keys so that the pairs with same keys will be grouped together.  Then, the worker iterates over sorted intermediate key/values. For each unique intermediate key, it passes the key and corresponding values to `reduce` , and write the output of `reduce` to a final output file for the reduce partition.
            After all map and reduce tasks are finished, master wakes up the user program, and MapReduce returns.
            The output of MapReduce can be found in the R output files, with one per reduce task.
        Master Data Structures
            The master will store each map and reduce task's status (idle, in-progress, completed) and the executer worker's ID.
            The master also stores the locations and sizes of the R intermediate file regions produced by each map task.
        Fault Tolerance
            The master pings each worker periodically. If no response is received from a worker for a certain amount of time, the worker will be marked as failed. What' more, the map tasks' status complete (or in-progress) by this worker will be reset to idle, and will be scheduled to other workers, because the failed worker's disk can not be accessed any more.
            The master will write periodic checkpoints in case of failure. The checkpoints can serve as starting points for restoring from failure.
            For deterministic `map` and `reduce` , MapReduce will produce the same output as produced by a non-faulting sequential execution of the entire program. This property relies on the atomic commit of map and reduce tasks' output. Each in-progress task write its output to its private files. The map worker produce R such files, and their names will be recorded by the master. The reduce worker produce 1 such file, and will rename the file to the correct name by completion.
        Locality
            The master will consider the location information of the input files, and schedule a map task on a machine that contains a replica of the corresponding input data. On failing, it attempts to schedule to another machine near that one.
        Task Granularity
            Ideally, M and R should be much larger than the number of worker machines. Having each worker performing different tasks improves dynamic load balancing.
            However, the master must make $O(M + R)$ scheduling decisions and keeps $O(M*R)$ state in memory. Therefore, there are practical bounds on how large M and R can be.
            In practice, M is chosen to make each map task with 16MB - 64MB of input data, and R is chosen to be a small multiple of the number of worker machines.
        Backup Tasks
            One common causes that lengthens the total time of MapReduce is straggler, which means a machine taking unusual long time to complete the last few map or reduce tasks.
            A general mechanism to alleviate the problem of stragglers is to schedule backup executions of remaining in-progress tasks when the MapReduce operation is close to completion. The task is marked completed if either the primary or the backup execution is completed. This will significantly reduce the times to complete large MapReduce operations.
    Refinements
        Default partition function uses hashing, which tends to result in fairly well-balanced partitions. Partition function can be customized by users.
        It is guaranteed that within a given partition, the intermediate key/value pairs will be ordered in an increasing key order.
        For some tasks, there is significant repetition of intermediate keys produced by each map task, and the `reduce` is commutative and associative. In this case, the user can define an optional `combiner` function. The `combiner` function is invoked by each map workers. Usually the code of `combiner` is the same as `reduce`, the output of `combiner` will be buffered can finally sent to a reduce task. 
        Each worker process installs a signal handler that catches segmentation violations and bus errors. Before invoking `map` or `reduce` , MapReduce will store the sequence number of the corresponding argument/record in a global variable. If the user code generates a signal, the signal handler will sends a UDP packet containing the sequence number to master.
        If the master has seen more than one failures in a particular record, it will indicate that the record should be skipped on the next re-execution scheduling.
        MapReduce library provides a counter facility to count occurrences of various events. The counter values from an individual worker machine will be periodically propagated to the master (attached on the ping response). The master aggregates the counter values from successful map and reduce tasks.
    Performance
    Experience
    Related Work
    Conclusion
        Restricting programming model makes it easy to parallelize and distribute computations and to make such computations fault-tolerant.
        Locality optimization can save a lot amount of network bandwidth.
        Redundant execution can be used to reduce the impact of slow machines, as well as handling machine failure and data loss.

\[Book\]
- [[book-notes/深度强化学习|深度强化学习]]: CH6
    CH6-价值学习高级技巧
        CH6.1-经验回放
            使用行为策略收集数据时，连续的四元组 $(s_t, a_t, r_t, s_{t+1})$, $(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})$ 之间有很强的相关性，经验回放从整个数组中每次随机抽取一个用于优化，可以尽可能让相邻两次优化使用的四元组是相互独立的，有助于缓解相关性的影响
            经验回放可以重复利用数据，增大收敛速率
            普通经验回放中，各个四元组权重相同，执行的是均匀抽样，优先经验回访中，各个四元组权重不同，执行非均匀抽样
            四元组的权重可以由 TD 误差 (的绝对值) 计算，TD 误差越大，权重越大，该四元组就更容易被抽中 (抽样概率由权重定义)
            非均匀抽样会带来偏差，为了缓解，需要进一步基于抽样概率调节学习率，容易被抽中的样本对应的学习率小
            虽然容易被抽中的样本的学习率小，但在小学习率下被多次抽中对于权重的影响实际上会大于在大学习率下被少量抽中，因此重要的样本仍然会被更多地利用
        CH6.2-高估问题及解决方法
            朴素的 Q Learning 算法训练的 DQN 倾向于高估真实的价值，原因有：1. 自举产生的偏差传播 2. 最大化导致 TD 目标高估真实价值
            如果所有的动作价值都被均匀地高估，则高估也不会对决策产生实际影响，但实践中，四元组的权重不同，被优化的频率也不同，更频繁被选中的四元组中的动作的价值就更倾向于被高估，因此高估是不均匀的
            缓解自举的偏差传播的方法是使用目标网络，目标网络结构和 DQN 相同，参数不同，目标网络会用于计算 TD 目标，目标网络的参数使用加权平均，根据原网路的参数更新
            基于目标网络，双 Q 学习可以缓解最大化造成的高估，双 Q 学习将 TD 目标的计算分为两步：1. 选择 2. 求值。双 Q 学习中，选择时使用 DQN，求值时使用目标网络，这使得求值不一定得到最大值，故得到的 TD 目标更小，进而缓解了最大化导致的高估问题
        CH6.3-对决网络
            对决网络也是对  $Q_*$ 的近似，可以使用和 DQN 完全相同的算法训练对决网络
            对决网络用近似最优优势函数和近似最优状态价值函数计算近似最优动作价值函数，注意公式 6.1 中的最右项不能忽略，否则网络参数会存在不唯一性的问题，且实际实现时，用 $\text{mean}$ 替代 $\max$ 有更优表现
        CH6.4-噪声网络
            噪声网络将普通神经网络中的参数 $\pmb w$ 替换为 $\pmb \mu + \pmb \sigma \circ \pmb \xi$，其中 $\pmb \xi$ 中的元素从标准正态分布中抽取 $\mathcal N(0, 1)$，也就是显式假设了网络参数 $w_i$ 服从均值为 $\mu_i$，标准差为 $\sigma_i$ 的正态分布
            噪声网络的可训练参数数量是原网络的两倍
            噪声 DQN 本身带有随机性，因此直接作为行为策略用于收集数据也可以让智能体尝试更多的动作
            更新参数时，预测和计算 TD 目标时也会采样不同的噪声
            训练完毕后，推理时不再需要噪声，可以将 $\pmb \sigma$ 设置为全零，仅保留 $\pmb \mu$
            噪声训练使得模型具有更强的健壮性，在参数出现轻微扰动的情况下也不会有较大偏差的预测值

## March
### Week 1
Date: 2025.2.24-2025.3.3

\[Paper\]
- [[paper-notes/distributed-system/Scalability But at what COST-2015-HotOS|2015-HotOS-Scalability But at what COST]]: All
    Abstract
        COST (Configuration that Outperforms a Single Thread) of a given platform for a given problem is the hardware overhead requires that the platform outperforms a single-thread implementation.
        COST weighs a system's scalability against the overhead brought by the system. Systems with high scalability at the expense of introducing substantial overhead will have high COST.
        Before requiring more resources, please first consider whether you have fully utilized the current resources at hand.
    Introduction
        We should consider to what extent the distributed systems truly improves the performance as opposed to just parallelizing the overheads that they introduce.
        A system with unbounded COST means that there is no configuration for that system to outperforms the best single-thread implementation for the target problem.
    Basic Graph Computations
    Better Baselines
        In some cases, the appealing scaling property of an algorithm essentially originate from the algorithm's inherent sub-optimality.
    Applying COST to prior work
    Lessons Learned
        The implementation of a distributed system may introduce overheads that single-thread implementation does not have. To properly assess a system's capability, the overheads introduced by it should be also understood.
    Future directions

\[Book\]
- [[book-notes/Reinforcement Learning An Introduction|Reinforcement Learning An Introduction]]: CH4.1-CH4.4, CH4.6
    CH4-Dynamic Programming
        Classical dynamic programming algorithm needs a perfect model and have great computational expense.
        We assume the environment is finite MDP, that is, the state, action, and reward set are all finite set. Roughly speaking, DP is only applicable in finite MDP.
        The key idea of DP is to use the value function to organize policy, that is, we use DP to compute the optimal value function, and use the optimal value function to define the optimal policy.
        DP algorithm turns Bellman equation to assignment, or to say, the update rule for improving the approximating value function.
        CH4.1-Policy Evaluation
            In DP, policy evaluation refers to compute the state-value function $v_\pi$ for a policy $\pi$. According to the Bellman equation, if the environment dynamics are known, solving $v_\pi$ can be formulated to solving a linear system of $|\mathcal S|$ equations. As long as $\gamma<1$ or the eventual termination is guaranteed from all states under $\pi$. The existence and uniqueness of $v_\pi$ is guaranteed.
            Iterative methods are suitable solution for this problem. This method turns the Bellman equation into an update rule for all $s\in \mathcal S$ to iteratively improve the approximating value function. Obviously, $v_k = v_\pi$ is a fixed point for this updating rule. It can be shown that under the same condition that guarantee the existence of $v_\pi$, sequence $\{v_k\}$ will converge to $v_\pi$ as $k\to \infty$.
            This algorithm is called iterative policy evaluation.
            To produce $v_{k+1}$ from $v_k$, iterative policy evaluation applies the same operation to each state $s$: replace the old approximating value with the new value, which is computed according to all possible one-step environment dynamic and rewards. This kind of operation is called full backup.
            Every iteration of iterative policy evaluation backs up the value of each state, and get the new approximating value function.
            All backups in DP is full backup, which means they are based on all possible next states rather a sample next state.
            The order by which the state space is traversed in each iteration has a significant impact on the convergence rate of the in-place iterative policy iteration.
        CH4.2-Policy Improvement
            After policy evaluation, we have known $v_\pi$ for $\pi$. Therefore, we have known that from $s$, following $\pi$, we are expected to get return $v_\pi(s)$. Next, we need to improve $\pi$ , to get more expected return for the same starting state.
            The policy improvement theorem states that, for any pair of deterministic policies $\pi, \pi'$ such that for all $s\in \mathcal S$: $q_\pi(s, \pi'(s))\ge v_\pi(s)$. Then $\pi'$ is at least a better policy than $\pi'$, which means for all $s\in \mathcal S$: $v_{\pi'}(s)\ge v_\pi(s)$
            Therefore, we can define $\pi'$ as follows: known $v_\pi$ and environment dynamics, we can compute $q_\pi(s, a)$, and known $q_\pi(s, a)$, we just modify $\pi$ to deterministically choose $\max_a q_\pi(s, a)$ at $s$ for all $s\in \mathcal S$. In this way, by the policy improvement theorem, $\pi'$ is at least a better policy than $\pi$.
            In other words, the new policy is greedy to $v_\pi$. This policy satisfies the conditions of the policy improvement theorem, and is at least as good as the original policy. This process is called policy improvement.
            When the new policy is as good as the old policy, or to say $v_\pi = v_{\pi'}$, according to the Bellman optimal equation, the value function will be the optimal value function. Therefore, the policy improvement process will give a strictly better policy unless the original policy is optimal.
        CH4.3-Policy Iteration
            Once a policy have been improved, we can recalculate its value function by value iteration, and apply policy improvement. Repeating this process yields a sequence of monotonically improving policies and value functions. Each policy in the sequence is strictly better than the previous one. Because finite MDP has only a finite number of policies, this process must converge to an optimal policy and optimal value in a finite number of iterations.
            This algorithm is called policy iteration. Notice that in policy iteration, the new policy's evaluation can start from the previous policy's value function to speed up convergence. Policy iteration often converges in a few iterations.
        Ch4.4-Value Iteration
            Value iteration can be understood as a special version of policy iteration wherein the policy evaluation is stopped after just one sweep (one backup of each state). The convergence of value iteration is also guaranteed.
            An other way to understand value iteration is to view Bellman optimal equation as the updating rule.
        CH4.6-Generalized Policy Iteration
            In policy iteration, the policy evaluation and policy improvement process can interact in more different ways. The ultimate results is the same: convergence to the optimal value function and an optimal policy.

\[Doc\]
- [[doc-notes/go/getting-started/Tutorial|go/getting-started/Tutorial]]: Get started with Go, Create a Go module, Getting started with multi-module workspaces
- [[doc-notes/go/getting-started/A Tour of Go|go/getting-started/A Tour of Go]]: All
    Basics
        In Go, a name is exported if it begins with a capital letter. When a package is imported, only its exported names can be used.
        A function can return any number of results. The return values can be named, and a naked return statement will return the named values.
        Inside a function, `xxx := value` is equivalent to `var xxx = value`. Outside the function, every statement should begin with a keyword, therefore `:=` is not available.
        Expression `T(v)` converts value `v` to type `T`. In Go, assignments between items of different types requires explicit type conversion.
        `for` without any semicolons is equivalent to `while`
        `defer` defers execution of a function until the surrounding function returns. The argument of deferred function will be evaluated immediately. The deferred functions are stored in a stack, and will be executed in a last-in-first-out way.
        Type `[]T` represents a slice with elements of type `T`. Slices have dynamic length, and slices does not store data, just referencing.
        Slice literal means first creating an array, and then reference it.
        The length of a slice refers to the number of elements it contains, and the capacity of a slice refers to the number of elements in the underlying array, counting from the first element in the slice.
        By re-slicing itself, slice can extend its length under the range of capacity.
        Slices can be created by `make()`
        Slices can be iterated by `range`
        Maps can be created by `make()`
        Functions are values too, and can be passed around and returned.
        A closure is a function value that references variables outside of the function body. The function can access and assign those variables, that is, the function is bound to the variables.
    Methods and interface
        In Go, we can define methods for types. Methods are functions with a special receiver argument. We can only declare a method for types defined in the same package.
        Receivers can be pointer, because Go only pass values, to modify the receiver's values in the method, we must use the pointer receiver.
        An interface type is defined as a set of method signatures. A value of interface type can hold any value whose type implements those methods. A type implements an interface by implementing its methods. Calling a method on an interface value executes the method of the same name on its underlying type.
        The interface type that specifies zero methods is the empty interface. An empty interface can hold values of any type, because every type implements at least zero methods.
        Type assertion can be used to judge an interface value's concrete type.
    Generics
        In Go, we can define generic functions by using type parameters. Go also supports generic types, that is, a type can be parameterized with a type parameter.
    Concurrency
        A go routine is a lightweight thread managed by Go runtime. State `go ...` can start a new go routine. The evaluation of function arguments happened in current routine. The execution happened in the new routine.
        Go routines run in the same address space, so accesses to the shared memory must be synchronized.
        Channels are typed conduit. We use channels to send and receive values between go routines. Channels can be created by `make()` .
        By default, send and receive block until the other side is ready.
        Channels can be buffered. Sends to a buffered channel block only when the channel is full. Receives from a buffered channel block only when the channel is empty.
        A sender can `close` a channel to indicate that there is no more values to sent.
        `select` statement lets a go routine wait on multiple communication operations. It will blocks until one of its case can run, then executes it. The `default` case can run if no other case is ready.
        To guarantee a variable is accessed by only one routine at each time. We should use `sync.Mutex`.
- [[doc-notes/python/packaging/Overview of Python Packaging|python/packaging/Overview of Python Packaging]]

\[Code\]
- NowcastNet rewritten project
    evaluation framework

### Week 2
Date: 2025.3.3-2025.3.10

\[Book\]
- [[book-notes/Reinforcement Learning An Introduction|Reinforcement Learning An Introduction]]: CH5.1-CH5.3, CH6.1-CH6.5, CH7.1-CH7.3
    CH5-Monte Carlo Methods
        MC methods require only experience, i.e. sample sequence of states, actions, and rewards from actual of simulated interaction with an environment.
        In many cases, it is easy to generate experience sampled according to the desired probability distribution, but infeasible to obtain the distributions in explicit form.
        We define MC methods only for episodic tasks. The value estimates and policies only changed on the completion of one episode. Thus, MC methods is incremental in an episode-by-episode sense instead of in a step-by-step sense.
        CH5.1-Monte Carlo Prediction
            For first-visit Monte Carlo, each return is an i.i.d estimate of $V_\pi(s)$ with finite variances. By the law of large numbers, the sequence of averages of these estimates converges to their expected value. Each average is itself an unbiased estimation, the standard deviation of its error falls as $1/\sqrt n$, where $n$ is the number of return averaged.
            In MC, the estimates for each state is independent, and does not built upon the estimates for another state. Therefore, in MC, the computational expense of estimating the value for a single state is independent of the number of states.
        CH5.2-Monte Carlo Estimation of Action Values
            When dynamics are unavailable, estimating action value function is more important than estimating state value function, because state value alone is not sufficient to determine a policy. Therefore, the primary goal of MC is to estimate $q_*$.
            When using MC estimating the action value function, the samples are (state, action) pairs. As the visit number to a (state, action) pair goes to infinite, the estimation goes to the expectation. The problem now is to maintain exploration to ensure every (state, action) will be visited sufficiently.
            One way is to specify that the episode begin with particular (state, action) pair, and every pair has non-zero probability of being selected as the beginning. This is called the assumption of exploring starts.
            In actual environment, this assumption is not so useful, because the start condition is hard to specify. The more common alternative is to use a stochastic policy to ensure each pair has non-zero probability to be visited.
        CH5.3-Monte Carlo Control
            The idea of MC Control is still general policy iteration, in which the policy evaluation in executed in the MC way, and directly evaluating the action value function. The policy improvement theorem still holds in this way. Therefore, the convergence is guaranteed.
            If we alternate between policy evaluation and improvement in an episode-by-episode basis. After each episode, the observed returns are used for policy evaluation, and then the policy is improved at all states visited in the episode. We get MC with Exploring Starts algorithm.
        CH5.4-Monte Carlo in Control without Exploring Starts
            To avoid exploring starts assumption, we need to consider off-policy methods. An usual method is using $\epsilon$ -greedy policy. That is, in policy improvement, we only improve the policy to an $\epsilon$ -greedy policy instead of a complete greedy one. 
            For any $\epsilon$ -soft policy $\pi$, any $\epsilon$ -greedy policy with respect to $q_\pi$ is guaranteed to be better than or equal to $\pi$. Therefore, policy iteration works for $\epsilon$ -soft policy.
    CH6-Temporal Difference Learning
        TD combines the idea of MC and DP. Like MC, TD methods learn from experience. Like DP, TD methods update estimates based on another estimate.
        For the control problem (searching for the optimal policy), TD, MC, DP all use the idea of GPI, the difference lies on the approach to solve the prediction problem.
        CH6.1-TD Prediction
            Unlike MC, TD(0) will not wait till the end of the episode, but only wait for one time step, and immediately update $V(s_t)$. The target for MC update is $G_t$, and for TD(0) update is $R_t + \gamma V(s_{t+1})$.
            The 'estimation' of MC comes from sampling, and the 'estimation' of DP comes from bootstrapping. The 'estimation' of TD comes from both.
        CH6.2-Advantages of TD Prediction Methods
            TD's advantage over DP is that TD does not need dynamics. TD's advantage over MC is that TD does not need to wait the episode to end, and can be implemented in a naturally on-line, incremental fashion.
            Though without theoretical proof, in practice, TD methods usually converge faster than constant- $\alpha$ MC methods.
        CH6.3-Optimality of TD(0)
            Suppose there is only a finite amount of experience available, a common approach is to present these experience repeatedly until the method converges upon an answer.
            In batch updating, the value function is changed by the sum of all increments computed from a batch of experience. That is, updates are made only after processing each complete batch of training data.
            Under batch updating, TD(0) converges deterministically to a single answer independent of the step-size parameter $\alpha$, as long as $\alpha$ is chosen sufficiently small.
        CH6.4-Sarsa: On-Policy TD Control
            When using TD methods for policy evaluation (learning action value function for $\pi$), we consider transition from (state, action) pair to (state, action) pair.  The TD(0) updating rule in this case uses the quintuple of events $(S_t, A_t, R_t, S_{t+1}, A_{t+1})$, therefore, the algorithm is called Sarsa.
        CH6.5-Q Learning: Off-Policy TD Control
            Q Learning uses TD methods to directly approximate $q_*$. The updating rule does not require sampling action, thus off-policy is allowed. All that requires for convergence is that all pairs continue to be updated.
    CH7-Eligibility Traces
        Almost every TD method can be combined with eligibility traces to obtain more general methods that may learn more efficiently.
        In a more theoretical view, TD methods augmented with eligibility traces are a bridge from TD(0) to Monte Carlo methods.
        In a more mechanistic view, an eligibility trace is a temporary record of the occurrence of an event. The event are marked with an memory parameter to represent eligibility.  Only eligible events will be considered as the source of TD error.
        CH7.1-n-Step TD Prediction
            For any value function $v$, the expected value of n-step return using $v$ is guaranteed to be a better estimate of $v_\pi$ than $v$ is. The worst error under the new estimate is guaranteed to be less than or equal to $\gamma^n$ times the worst error under $v$. This is called the error reduction property of n-step returns.
            However, n-step TD methods are rarely used in practice because of the inconvenience to implement.
        CH7.2-The Forward View of TD($\lambda$)
            Further, the target can be a weight sum of multiple n-step returns. Such a composite return possesses an error reduction property similar to that of the individual n-step return.
            TD($\lambda$) can be understood as one particular way of averaging n-step backups. This average contains all the $n$ -step returns, each weighted proportional to $\lambda^{n-1}$, and normalized by a factor of $1-\lambda$ to ensure that the weights sum to 1. The resulting target is called $\lambda$ -return.
            In $\lambda$ -return, the one-step return has largest weight, and the weights fade with $\lambda$ by each successive step.
            The $\lambda$ -return algorithm is the method that performs update using the $\lambda$ -return as target on each time step $t$. The overall performance of $\lambda$ -return algorithm is comparable to that of $n$ -step algorithms. Both get the best performance in the intermediate value of the truncation parameter $n$ or $\lambda$.
        CH7.3-The Backward View of TD($\lambda$)
            The forward view is not directly implementable because it is acausal, which means it uses knowledge of the future in each time step update.
            The backward view provides a causal, incremental mechanism for approximating the forward view, and in off-line case, they are equivalent.
            In the backward view of TD ($\lambda$), there is a memory parameter associated with each state, called its eligibility trace. The eligibility trace for state $s$ at time $t$ is a random variable denoted $E_t(s)$. On each time step, the eligibility trace of all non-visited states decay by $\gamma\lambda$. Henceforth, $\lambda$ is also referred as the trace-decay parameter.
            For the visited state, the classical eligibility trace decays and then increments by 1. This kind of eligibility trace is called accumulating trace.
            Eligibility traces keep a record of which states have recently been visited. The traces indicates the degree to which each state is eligible for undergoing learning changes. The changes for each state is the current TD error multiplied by its eligibility trace.
            The backward view of TD($\lambda$) is oriented backward in time. At each time step, we get current TD error, and assign it to each prior state according to the state's eligibility trace now.
            When $\lambda = 0$, TD($\lambda$) is equivalent to TD(0). When $\lambda = 1$, the eligibility trace only decays by $\gamma$, TD(1) turns out to achieve the Monte Carlo behaviour.
- [[book-notes/深度强化学习|深度强化学习]]: CH7
    CH7-策略梯度方法
        策略网络直接近似策略 $\pi(a\mid s;\pmb \theta)$，接受状态作为输入，输出 $|\mathcal A|$ 个概率值
        策略所定义的价值函数衡量了策略的优劣程度，故策略网络学习的目标函数 $J(\pmb \theta)$ 定义为策略 $\pi(a\mid ;\pmb \theta)$ 的价值函数的期望，即 $J(\pmb \theta) = \mathbb E_S[V_\pi(S)]$，$J(\pmb \theta)$ 相对于 $\pmb \theta$ 的梯度称为策略梯度
        策略梯度定理给出了策略梯度的详细形式，可以看出策略梯度的计算和策略从动作价值函数相关。策略梯度定理基于 Markov chain 已经达到稳态的假设，此时状态从属于 Markov chain 的稳态分布
        策略梯度的形式是一个期望，需要 Monte Carlo 近似，这涉及到从稳态分布中对状态采样以及根据策略网络对动作采样。并且还需要对策略的动作价值进行近似。
        REINFORCE 直接使用 Monte Carlo 方法近似动作价值，即直接将 $Q_\pi(s, a)$ 替换为实际回报 $u$
        Actor-Critic 使用另外的价值网络近似动作价值函数，价值网络使用 SARSA 和策略网络一起训练
        价值网络的训练涉及到了自举，因此也可以采用目标网络方法缓解自举造成的误差传播问题

\[Doc\]
- [[doc-notes/go/getting-started/How to Write Go Code|go/getting-started/How to Write Go Code]]: All
    Code Organization 
        Go programs are organized into packages. A package is a collection of source files in the same directory that are compiled together. Functions, variables, constants are all visible to other source files within the same package.
        A repository contains one or more modules. A module is a collection of packages to be released together. Typically, a Go repository contains only one module, located in the root directory of the repo.
        `go.mod` in the repo root declares the module path, which is the path prefix for all packages within the module. The module path also directs `go` command where to install it. For example, `go` will consult `https://golang.org/x/tools` to install the module whose path is `golang.org/x/tools`.
        The import path for a package is its module path + its subdirectory within the module
    Your first program
        The first statement in a Go source file must be `package name`. Executable commands must always use `package main`
        To build and install the program, use `go install`
        The install directory is controlled by the `GOPATH, GOBIN` environment variables. 
        To test the packages compiles, use `go build` . This won't produce an output file but saves the compiled package in the local build cache. When built, the packages can be imported.
        `go mod tidy` will automatically manage dependencies on external modules.
    Testing
- [[doc-notes/mlir/code-documentation/tutorials/Toy Tutorial|mlir/code-documentation/tutorials/Toy Tutorial]]: CH3-CH7
- [[doc-notes/python/packages/gymnasium/Introduction|python/packages/gymnasium/Introduction]]: All
    Basic Usage
        The four key function of `gymnasium` is `make(), Env.reset(), Env.step(), Env.render()`
        The core of `gymnasium` is `Env` , which is a python class representing a MDP.
        `Wrapper` are provided to augment/modify the environment.
        `Env.action_space, Env.observation_space` are instances of `Space`
    Training an Agent
    Create a Custom Environment
        Custom environment should inherit from `Env`
    Recording Agents
    Speeding Up Training

### Week 3
Date: 2025.3.10-2025.3.17

\[Paper\]
- [[paper-notes/distributed-system/Paxos Made Simple-2001|2001-Paxos Made Simple]]: All
    Abstract
    Introduction
    The Consensus Algorithm
        The Problem
            For a collection of processes that can propose values. A consensus algorithm ensures among all proposed values, only a single one is chosen. If a value has been chosen, the processes should be able to learn these values.
            To achieve consensus, the saft requirements are: 1. Only one proposed value will be chosen. 2. processes can not know in advance that a certain value will be finally chosen.
            The goal is to finally choose a value and let processes learn this choice.
            There are three roles: proposer, acceptor, learner in the consensus algorithm.
        Choosing a Value
            To choose a value, the simplest way is have only one acceptor, who chooses the first proposed value it received. However, the failure of the acceptor will make the system not progress.
            Considering multiple acceptors, a natural way to define 'chosen' is that a majority of acceptors have accepted the value.
            In the absence of message missing and failure, we want even only one proposer proposed only one value, a value will be chosen. This suggests requirement P1: an acceptor must accept the first proposal it receives.
            However, if we only allow an acceptor only accept one proposal, it will be possible that there will be a situation that no proposal reach majority acceptance.
            Therefore, we must allow an acceptor to accept multiple proposals. We assign a number to each proposal to distinguish different proposals.
            We can allow multiple proposals to be chosen, but we must ensure that all the chosen proposals have the same value.
            By induction on the proposals, to satisfy this requirement, we can in turn satisfy requirement P2: if a proposal with value $v$ is chosen, then every higher-numbered proposal that is chosen has value $v$.
            Condition P2 guarantees that only one value will be chosen.
            To satisfy P2, we can in turn satisfy $\mathrm{P2}^a$: If a proposal with value $v$ is chosen, then every higher-numbered proposal accepted by any acceptor has value $v$.
            However, to maintain P1, an acceptor must accept any proposal it first receives even the proposal has unsatisfactory value. This conflicts with $\mathrm{P2}^a$. To maintain P1 and $\mathrm{P2}^a$ simultaneously, we need to strength $\mathrm{P2}^a$ to $\mathrm{P2}^b$: If a proposal with value $v$ is chosen, then every higher-numbered proposal issued by any proposer has value $v$. 
            That is, we transfer the responsibility from acceptors to proposers.
            To satisfy $\mathrm{P2}^b$, we can in turn satisfy $\mathrm{P2}^c$: For any $v$ and $n$, if a proposal with value $v$ and number $n$ is issued, then there is a set $S$ consisting of a majority of acceptors such that either (a) no acceptor in $S$ has accepted any proposal numbered less than $n$, or (b) $v$ is the value of the highest-numbered proposal among all proposals numbered less than $n$ by the acceptors in $S$. Otherwise, the proposer can not issue the proposal.
            To maintain $\mathrm{P2}^c$, proposer must learn the highest-numbered proposal with number less than $n$, that has been accepted by each acceptor in some majority of acceptors. Also, after the proposer leant it, the acceptors should not accept any proposal with larger number and different value than $v$, otherwise the proposer will break $\mathrm{P2}^c$.
            Therefore, the proposer will request the acceptors not accept any proposal with number less than $n$. 
            For a proposer, it can send two kinds of request. One is prepare request, which is used to learn the highest-numbered proposal and let acceptors make their promise. Another one is accept request, which is used to request the acceptors to accept the proposal.
            An acceptor can receive those two kinds of request from the proposer. An acceptor can always respond to a prepare request and can accept an accept request iff it has not make promise to refuse it. This leads to $\mathrm{P1}^a$: An acceptor can accept a proposal numbered $n$ iff it has not responded to a prepare request having number greater than $n$.  $\mathrm{P1}^a$ subsumes P1.
            An acceptor only needs to remember the highest-numbered proposal that it has ever accepted and the number of the highest-numbered prepare request it has responded, even when failure occurs, because $\mathrm{P2}^c$ should be kept disregard of failure.
            Combining proposers' and acceptors' actions, the algorithm has the following two phase: 1. (a) a proposer selects a number and sends prepare request to acceptors (b) If an acceptors receives it, and $n$ is larger than any prepare request's number is has responded, then the acceptor will respond 2. (a) If the proposer receives the response. it will send accept requests (b) If an acceptor receives an accept requests, it will accept it unless it has already responded a prepare request with larger number.
        Learning a Chosen Value
            We let acceptors send their acceptance to a single learner, and the learner will notify other learners the choice.
        Progress
            In this algorithm, it is easy to construct that when multiple proposers co-exist, the system may not progress, though safety is maintained.
            To ensure progress, a distinguished proposer should be selected, which is the only one has the authority to issue proposals.
        The Implementation
            To ensure different proposers never issue proposals with the same number, different proposers should choose their number from disjoint sets of numbers.
    Implementing a State Machine
        The central server can be described as a deterministic state machine that performs client commands with a certain sequence.
        The state machine takes current command as input and produce an output and next state.
        When there are multiple central servers, to ensure they all have same state sequence and output sequence, we must ensure they all receive the same input command sequence.
        We use consensus algorithm to determine the command sequence. The consensus algorithm ensures each command in the command sequences is a consensus without ambiguity, therefore the command sequence is a consensus without ambiguity.
        We implement a sequence of separate instances of Paxos algorithm, instance i is responsible for choosing the i-th command in the sequence.
        In normal operation, a server will be selected to be the distinguished learner and proposer. Clients send commands to the server, the server will determine which command should be the i-th command. Then, the server will try to make this decision be a consensus among all servers. It might fail because server failure or because another server believes itself to be the leader. However, the consensus algorithm ensures only one command will be chosen.
        Considering the previous leader just failed and another leader has been selected. The new leader is also a learner, therefore it may know most of the commands that has been chosen. The new leader will execute the algorithm's phase 1 for undetermined commands. When receiving response, it can execute phase 2 for corresponding commands.
        Note that the execution of phase 1 for infinitely large algorithm instances is possible, the new leader can use the same proposal numbers. A single short message can achieve this. Therefore the effective cost is just the cost of executing phase 2.
        Note that multiple leaders may appear, but the safety is guaranteed.
- [[paper-notes/distributed-system/The Google File System-2003-SOSP|2003-SOSP-The Google File System]]: All
    Abstract
    Introduction
        GFS shares many the same goals as previous distributed file systems such as performance, scalability, reliability, availability. Driven by the typical workload of Google, GFS explores different points in the design space.
        The system design assumptions of GFS includes:
        1. Component failure is common rather than exception. Virtually at any given time, some components are not functional and some will not recover from their current failures. Therefore, system must have the ability of constant monitoring, error detection, fault tolerance, automatic recovery.
        2. Files are huge, multi-GB files are common. Therefore, IO operation and block sizes should be considered carefully.
        3. Most file mutations are appending new data instead of rewriting existing data. Given this accessing pattern, appending is the focus of performance optimization and atomicity guarantees.
        4. GFS is codesigned with the application to increase flexibility. The consistency model of GFS is relaxed to simplify the system, therefore the application takes some responsibility to guarantee consistency. Atomic append is introduced to let multiple clients concurrently append to the same file without extra synchronization.
    Design Overview
        Assumptions
        Interface
            GFS provides create, delete, open, close, read, write, snapshot, and record append operations.
        Architecture
            A single GFS cluster consists of a single master and multiple chunkservers and is accessed by multiple clients. Each of these is typically a commodity Linux machine running a user-level server process.
            Files are divided into fixed-size chunks, each identified by an immutable and globally unique 64 bit chunk handle assigned by master at chunk creation. Chunkservers store chunks as Linux files on local disks and read or write chunk data specified by chunk handle and byte range. Each chunk is replicated on multiple chunkservers for reliability.
            Master maintains all filesystem metadata including the namespace, access control information, mapping from files to chunks, current location of chunks. Master controls system-level activities like chunk lease management, GC of orphaned chunks, chunk migration between chunkservers.
            Master and chunkservers will periodically communicate with each other in Heartbeat messages.
            Client code implements the file system API, and communicates with master and chunkservers on behalf of its upper level application. Clients communicate with master for metadata operations, and communicate with chunkservers for other data-bearing communications.
        Single Master
            The master's involvement in reads and writes should be minimized in case of becoming the bottleneck.
            For a simple reading operation, the client code first translates the file name and byte offsets specified by the application into a chunk index within the file. Then, the client sends a request containing the file name and chunk index to the master. The master replies the corresponding chunk handle and chunk location. The client will cache these information, then request the chunk data from nearby chunkservers.
        Chunk Size
            Chunk size is 64 MB, and lazy space allocation is used to alleviate internal fragmentation.
            Large chunk size reduces the clients' need to interact with master, as well as reducing the size of metadata stored in the master.
        Metadata
            Three major types of metadata are stored in the master's memory: namespace of files and chunks, locations of chunk replicas, mapping from files to chunks.
            The locations of chunk replicas will not be persistently stored. The master will polls chunkservers for those information at startup. 
            Namespace and mapping information will be persistently stored, and the mutations will be logged into operation logs. The master recovers its file system state by replaying the operation log. Whenever the log goes beyond certain size, the master will checkpoint it, so that it can recover by loading the latest checkpoint. Older checkpoints and log files can be freely deleted.
        Consistency Model
            File namespace mutations are atomic. They are handled by master exclusively. Namespace locking guarantees atomicity and correctness and the operation log defines the global total order of those mutation operations.
    System Interactions
        Lease and Mutation Order
            Leases are used to maintain a consistent mutation order across replicas. The mutation order will be decided by the replica holding the lease, called the primary.
            Lease mechanism can be considerer as master delegating its power to the primary, letting the primary deciding the operation order, thus minimizing the master's involvement while maintaining operation consistency.
            If a write by the application is large and across multiple chunks, the clients will breaks it into multiple mutations, creating the possibility that the final region being consistent but undefined.
        Data Flow
            The data flow and control flow are decoupled for higher network efficiency. Control flows from client to primary and then to all secondaries. Data flows from client to all chunkservers in a linear, pipelined fashion, to utilize each machine's full outbound bandwidth. Pipeline means once a chunkserver receives some data, it starts forwarding immediately.
        Atomic Record Appends
            GFS guarantees the data to be appended will be appended to the file atomically at least once. The specific will be chosen by GFS and returned to the client.
            If record append operation failed in any replica, GFS will retry. Therefore, replicas of the same chunk main contain different data, including possible duplicates of a record. The replicas are not guaranteed to be bytewise identical.
        Snapshot
            Snapshot operation is implemented as a copy-on-write way. When receiving an snapshot operation, master will first revoke the related leases to make sure when clients want to write, they will first request master to find the lease holder.
            At this time, master will notice that the chunk to be written has reference count larger than 1, then the response will be delayed. The master will request the chunkservers to create actual local copies.
    Master Operation
        Namespace Management and Locking
            Master store the namespace as a hash table, which maps full pathnames to metadata.
            Locks of each node in the namespace tree are used to ensure proper serialization of concurrent namespace mutations.
        Replica Placement
            Replicas are placed in different machines, and different racks.
        Creation, Re-replication, Rebalancing
        Garbage Collection
        Stale Replica Detection
            Chunk replicas may become stale if chunkserver misses some mutation operation. Master maintains a version number for each chunk to distinguish stale replicas.
    Fault Tolerance and Diagnosis
        High Availability
            GFS use fast recovery and replication to keep high availability.
            Replication includes chunk replication and shadow masters.
        Data Integrity
            Verified by checksumming.
        Diagnostic Tools
    Measurements
    Experiences
    Related Works
        GFS assumes a large amount of unreliable components, the core design is fault tolerance.
    Conclusions
        GFS provides fault tolerance by constant monitoring, replicating crucial data, and fast atomic recovery.
        By decoupling control flow and data flow (large chunk size, lease mechanism), the single master in GFS does not become bottleneck.

\[Book\]
- [[book-notes/深度强化学习|深度强化学习]]: CH8
    CH8-带基线的策略梯度方法
        CH8.1-策略梯度中的基线
            只要 $b$ 不是关于 $A$ 的函数，将 $Q_\pi(S, A)$ 替换为 $Q_\pi(S, A)- b$ 不会影响策略梯度定理的正确性
            $b = V_\pi(S)$ 是较常用的基线
        CH8.2-带基线的 REINFORCE 算法
            带有基线时，我们用另一个神经网络近似 $V_\pi(S)$
            实践中，策略网络和价值网络可以共享用于提取特征的卷积层参数
            价值网络使用 Monte Carlo 更新
            $Q_\pi(S, A) - V_\pi(S)$ 中，$Q_\pi(S, A)$ 使用 Monte Carlo 近似
        CH8.3-Advantage Actor-Critic (A2C)
            常规的 Actor-Critic 方法中，价值网络用于近似 $Q_\pi(S, A)$，而 A2C 中，价值网络用于近似 $V_\pi(S)$
            价值网络使用 TD 更新
            $Q_\pi(S, A) - V_\pi(S)$ 中，$Q_\pi(S_t, A_t)$ 使用 $R_t + \gamma V_\pi(S_{t+1})$ 近似，因此实际上使用 TD 误差 $R_t + \gamma V_\pi(S_{t+1}) - V_\pi(S_t)$ 近似了优势值
        CH8.4-证明带基线的策略梯度定理

\[Doc\]
- [[doc-notes/mlir/code-documentation/tutorials/Understanding the IR Structure|mlir/code-documentation/tutorials/Understanding the IR Structure]]: All
    A pass is always rooted with an operation
    The IR is recursively nested, an `Operation` can have multiple `Region` s, an `Region` can have multiple `Block` s, an `Block` can have multiple `Operation` s
    Besides nesting relationship, an other relationship in the IR is the link relationship between a `Value` and its users. Each `Value` is either a `BlockArgument` or the result of one `Operation`
- [[doc-notes/mlir/code-documentation/Pass Infrastructure|mlir/code-documentation/Pass Infrastructure]]
- [[doc-notes/python/packages/hydra/Tutorials|python/packages/hydra/Tutorials]]
- [[doc-notes/python/library/file-formats/tomllib - Parse TOML files|python/library/file-formats/tomllib - Parse TOML files]]: All
- [[doc-notes/TOML v1.0.0|TOML v1.0.0]]: All

### Week 4
Date: 2025.3.17-2025.3.24

\[Paper\]
- [[paper-notes/distributed-system/In Search of an Understandable Consensus Algorithm (Extended Version)-2014-ATC|2014-ATC-In Search of an Understandable Consensus Algorithm (Extended Version)]]: All
    0-Abstract
        Raft is a consensus algorithm for managing replicated logs. It produces a result equivalent to multi-Paxos and is as efficient as Paxos.
        Raft separates the key elements of consensus: leader election, log replication, safety, and includes a new mechanism for changing cluster membership, which uses overlapping majorities to ensure safety.
    1-Introduction
        Compared to existing consensus algorithms, Raft has several new features: 1. strong leader: log entries only flow from leader to followers. This simplifies the management of replicated logs 2. leader election: Raft use randomized timers to solve leader election conflicts efficiently 3. membership change: Raft uses a joint consensus approach to overlap the majorities of two different configurations (overlapping $C_{old}$ with $C_{old, new}$ and $C_{old, new}$ with $C_{new}$ ), which ensures safety and allows the cluster to continue operating during transition.
    2-Replicated state machines
        Replicated state machines are typically implemented using a replicated log, which stores a series of commands. Each replicated log contains the same commands in the same order.
        Keeping the replicated log consistent is the job of the consensus algorithm, even if some servers may fail.
        Once the commands are properly replicated, each server's state machine processes them in log order, and the outputs return to the clients will be consistent even if the leader failed and another leader was elected. As a result, the servers appear to form a single, highly reliable state machine.
        Consensus algorithms for practical system typically has following properties: 1. ensure safety (never return incorrect result to clients) under all non-Byzantine conditions, including all network issues. 2. fully functional (available) as long as any majority of servers are operational and can communicate with each other and with clients. 3. servers do not depend on timing to ensure safety. Timing issues like delays and faulty clocks will only cause availability problems 4. a command execution can complete (that is, the command is committed) as soon as a majority responded a single round of RPC, there is no straggler issues.
    3-What' wrong with Paxos
        Multi-Paxos is extended from single-decree Paxos, which mainly focus on consensus on a single entry. We believe that the overall problem of reaching consensus on multiple decisions (i.e. a log instead of an entry) can be decomposed more directly.
        There is little benefit to choosing a collection of log entries independently and then combine them into a sequential log. It's simpler and more efficient to design a system around the log.
        Paxos uses a symmetric peer-to-peer approach, which only make sense when only making one single decision. If a series of decision must be made, it is simpler and faster to elect a leader, and have the leader coordinate the decisions.
    4-Designing for understandability
    5-The Raft consensus algorithm
        Raft implements consensus by first electing a leader, who has the complete responsibility for managing the replicated log. The leader accepts entries from clients and replicate them to other servers' logs, and tell them when it is safe to apply their entries to their state machines.
        A leader may fail or disconnect, in such case, a new leader should be elected.
        Given the leader approach, Raft decomposes the consensus problem into three relatively independent sub-problems: leader election, log replication, safety
        5.1-Raft basics
            In Raft, a servers may be as a leader, follower, or candidate at any given time. 
            In normal operation, there is only one leader, others are followers. Followers are passive. They issue no RPCs on their own, but only respond to requests from leader and candidates. The leader handle all client requests. If the client request a follower, the follower will redirect it to the leader.
            Time is divided into terms of arbitrary length, which is numbered by consecutive integers. Each term starts with an election, the winner candidate will become the leader. When split votes happen, the term will terminate with no leader and next term will begin soon.
            Terms act as a logical clock in Raft. Servers use term to detect stale leaders. Each server stores a current term number, which are exchanged when two servers communicate. If one server find its term obsolete, then it will update it. If a candidate or leader finds its term obsolete, it will return to the follower state. Requests with obsolete term number will be rejected for any server.
            The basic consensus is reached with only two RPCs: `AppendEntries` and `RequestVote`. `AppendEntries` is issued when leader replicate entries or sending heartbeats. `RequestVote` is issued when candidate start election.
            Servers will retry RPCs if they do not receive a timely response, and multiple RPCs will be send in parallel to improve performance.
        5.2-Leader election
            Leader election is triggered by the heartbeat mechanism. All servers start as follower. When they do not receive periodical heartbeat from the leader, after the election timeout, they will assume there is no leader, and become candidate to start an election.
            When starting an election, a follower increase its term number, become candidate and votes for itself, then issues `RequestVote` RPCs in parallel to all other servers. Each server will votes for sender of the first-come `RequestVote` .
            The majority rule ensures at most one candidate will win an election at a given term. Once a candidate becomes leader, it sends heartbeats to establish its authority, preventing new elections.
            If the candidate receives other leader's heartbeat with larger or equal term, the candidate recognize the leader and go back to follower. If the term is smaller, the candidate rejects the RPC and stays for voting.
            If the candidate neither receive majority votes nor receive new leader's heartbeat. It will timeout and starts a new election with increased term.
            Raft uses randomized election timeouts to ensure split votes are rare and can be solved quickly. Each server's election timeout is sampled from a fixed interval. In most case, there will be the first time-out server to start the next election and quickly win the election before other candidate's timeout.
            If the candidate timeout (split vote happened), it will sample another timeout, and waits for that timeout to elapse before starting the next election. This reduces the likelihood of split vote in the next election.
        5.3-Log replication
            Once a leader is elected, it begins handling client requests. When receiving a command, the leader appends it to its log, then issues `AppendEntries` to replicate that entry. 
            When the leader is sure that the entry is safely replicated, it will apply that command to its state machine, and return the execution result to the clients.
            If a follower does not reply `AppendEntries` timely, the leader will indefinitely retry until it is sure that all followers eventually stored all log entries.
            Each entry stores a command, a term number, and is associated with an index.
            Note that when to apply the entry is decided by the leader. An entry which is considered safe to apply by the leader is called committed. The leader will commit the entry once it has replicated that entry to a majority. When an entry is committed, all preceding entries in the log will be committed.
            Leader keeps track of the highest index of the committed entry, that index will be included in future `AppendEntries` to let followers know it and apply it as well.
            Raft maintains two following properties which together constitutes the Log Matching Property: 1. two entries with same term and index will store the same command 2. if two entries have same term and index, all their preceding entries will be the same
            The first property holds because a leader will only create at most one entry at any given index in any term. Thus, in the whole system duration, an index and a term number will unique identify an entry.
            The second property is guaranteed by the consistency check of `AppendEntries` . `AppendEntries` will fail if the preceding entry in the log to be appended is not the desired. The consistency check acts like an induction step: the initial state satisfy the Log Matching Property, and the consistency check makes sure that successful `AppendEntries` will keep the Log Matching Property. As long as `AppendEntries` return true, the leader knows the follower's log is consistent with its.
            A follower may missing entries that are present on the leader, may have extra entries that are not present on the leader, or both.
            In Raft, the leader force the followers to be consistent with its to solve inconsistency. This means conflicting entries in the followers' logs will be overwritten by the leader. The leader will find the latest consistent entry in the follower's log and overwritten all entries following. All these actions happen in response to the consistency check by `AppendEntries` RPCs.
            Under this mechanism, without tacking extra actions, a leader will converge the followers' log in normal operation.
        5.4-Safety
            To prevent the situation that a follower become unavailable which the leader commits several entries, and then become the new leader to overwrite the committed entries, we need another mechanism to ensure safety.
            Raft adds a restriction on which servers may be elected leader, to make sure only the candidate storing all committed entries can become the leader.
            To win an election, the candidate must communicate with a majority, which means each committed entries must be stored at least in one server of the majority. If the candidate's log is up-to-date compared to all serves in the majority, then we can make sure it has all committed entries.
            The `RequestVote` RPC implements this restriction: the RPC contains information about the candidate's log, if the candidates' log is not up-to-date, the voter will not vote for it.
            Raft determines which of two logs are up-to-date by comparing their last entries, if the term are same, the longer log is more up-to-date, if the term are different, the log with larger term is more up-to-date.
        5.5-Follower and candidate crashes
            Raft RPCs are idempotent.
        5.6-Timeing and availability
            broadcast time << election timeout << mean time between failures
    6-Cluster membership changes
        To make configuration transition safe, there must be no point during the transition where is it possible for two leaders elected by two disjoint majority in the same term.
        Therefore, the core is to prevent two disjoint majority emerge during the configuration transition.
        Raft use the joint consensus configuration as the transition configuration, in that configuration, 1. log entries should be replicated to all servers in both configurations 2. any server from either configuration can be elected leader 3. agreement requires separate majorities from both configuration.
    7-Log compaction
        Snapshotting with `InstallSnapshot` RPC
    8-Client interaction
    9-Implementation and evaluation
    10-Related work
    11-Conclusion
- [[paper-notes/rl/Proximal Policy Optimization Algorithms-2017|2017-Proximal Policy Optimization Algorithms]]: All
    Abstract
        Standard policy gradient method performs one gradient update per data sample. Whereas the new proposed objective function supports perform multiple epochs of minibatch updates based on multiple data samples collected through previous policy.
        Compared with TRPO, PPO is simpler to implement and empirically have better sample complexity.
    Introduction
        Deep Q-Learning failed on many simple continuous control problem. Standard policy gradient methods have poor data efficiency and robustness (because they are purely on-policy). TRPO is relatively complicated and not compatible with architectures that include noise or parameter sharing.
        PPO uses a new objective function with clipped probability ratio, which is a lower bound of the traditional objective for policy gradient methods. In optimization, it iterates between sampling data from policy and performing several epochs of optimization based on the sampled data. 
        In experiment, the objective with clipped probability ratio has best performance over other variant objectives.
    Background: Policy Optimization
        Empirically, performing multiple steps of optimization of the traditional policy gradient objective $L^{PG}$ using the same trajectory will lead to destructively large policy updates. 
        TRPO imposed a constraint on the size of policy update, the objective is also modified with importance weights to adapt to the stale samples.
        The theory justifying TRPO actually suggests using penalty term instead of constraint. TRPO use hard constraint instead of penalty term because it is hard to choose a single trade-off coefficient $\beta$ working for different problems.
    Clipped Surrogate Objective
        $L^{CLIP}$ add a clipped term into TRPO's objective $L^{CPI}$ and throws the constraint away. The $\min$ guarantees the final objective is a lower bound of the unclipped objective.
        Clipping of the probability ratio removes the incentive of optimizing $\theta$ too far away (moving $r_t(\theta)$ outside of interval $[1-\epsilon, 1 + \epsilon]$), therefore prevent radical policy update to a certain degree.
        More intuitively, clipping the probability ratio makes the actor do not totally trust the evaluation given by the critic, because the actor's belief on the critic is some kind of limited by the imposed probability ratio interval.
        Intuitively, this is reasonable, because 1. the critic typically learns slower than the actor even in the original policy gradient methods 2. when the actor learns multiple epochs with the stale samples, the critic's evaluation will be even less referential.
    Adaptive KL Penalty Coefficient
        An alternative to the clipped surrogate objective is using the penalty on KL divergence as regularization. In experiments, this method performs worse than the clipped surrogate objective.
    Algorithm
        The implementation has just minor difference between the typical policy gradient implementation. All we need to do is to substitute the traditional loss $L^{PG}$ with $L^{CLIP}$, and perform multiple steps of stochastic gradient ascent on this objective.
        We use a truncated version of generalized advantage estimation.
    Experiments
        PPO outperforms the previous methods on almost all the continuous environments.
    Conclusion

\[Book\]
- [[book-notes/深度强化学习|深度强化学习]]: CH9, CH10, CH13
    CH9-策略学习高级技巧
        CH9.1-Trust Region Policy Optimization (TRPO)
            相较于朴素的策略梯度方法，TRPO 表现更稳定，收敛曲线不会剧烈波动，对学习率不敏感，并且用更少的经验就能达到和策略梯度相同的表现
            置信域方法要求在给定邻域内，近似函数和目标函数的差异较小，由此可以用较简单的近似函数替代目标函数执行参数优化，以间接地找到能够近似最大化目标函数的参数
            置信域方法主要分为两步：1. 做近似 2. 最大化
            利用重要性采样，策略学习的目标函数可以写为对另一个策略求期望的方式，并且，如果行为策略和目标策略在置信域内足够接近，可以用行为策略的价值函数 (critic) 替代公式中的目标策略的价值函数
            由此，我们可以完全基于行为策略采集样本，计算目标，然后近似优化目标策略的参数
            虽然通过重要性采样，我们引入了一定的 off-policy，但为了避免方差过大，采样分布/行为策略和目标分布/目标策略不能相距过远，因此优化需要限制在置信域内执行
            置信域可以使用参数的直接距离定义，也可以基于采样分布和目标分布的 KL 散度定义
        CH9.2-熵正则 (Entropy Regularization)
            朴素的策略网络学习是 on-policy，故策略网络的输出决定了 agent 的探索程度，为了确保 agent 保持一定的探索，我们希望策略网络的输出保持一定的不确定性
            我们使用策略网络的输出的熵的期望度量其不确定性，将其作为正则化项，使得策略保持一定随机性
    CH10-连续控制
        CH10.1-离散控制和连续控制的区别
            连续控制中，动作空间是无限集，并且通常是多维的
            直接将离散控制的方法应用到连续控制的一种思路是将连续动作空间离散化，但存在维度灾难问题
        CH10.2-确定策略梯度 (DPG)
            最常用的连续控制方法是确定策略梯度，DPG 方法和朴素策略梯度方法的差异主要在于策略网络 (actor) 的输出不是概率质量函数/分布，而是确定的动作向量
            可以以该动作向量为均值定义多维高斯分布，引入一定的动作选择随机性
            DPG 可以使用异策略方法训练，即样本形式为 $(s_t, \pmb a_t, r_t, s_{t+1})$，原因在于 actor 是确定性时，$a_{t+1}$ 可以直接由 $s_{t+1}$ 计算得到，不需要执行 on-policy 采样 (这就类似于 DQN 中，直接取 $\arg\max_a$ )
        CH10.3-深度分析 DPG
            DPG 也存在 (近似) 最大化带来的高估问题和自举带来的误差传播问题
        CH10.4-双延时确定策略梯度 (TD3) 
            缓解高估问题的一种方法是为策略网络训练目标网络，我们认为该目标网络不会过好地近似 $\pi(s;\pmb\theta) = \arg\max_{a\in \mathcal A}Q_\pi(s, a;\pmb w)$
            缓解误差传播的方法仍是为价值网络训练目标网络
            可以进一步添加一个价值网络，同时训练两个价值网络，两个价值网络各自有其目标网络
            进一步的改进方法包括：为策略网络的输出添加噪声 (从截断正态分布中抽取)，让其进一步偏移最优动作；减缓策略网络和目标网络的更新频率，目的是让价值网络追上策略网络的更新频率 (一般策略的收敛比价值快)，以更好地拟合策略的价值函数
            在 DPG 中应用这些额外技巧的算法即 TD3，它仍是 off-policy 算法
        CH10.5-随机高斯策略
            为了避免 $\sigma$ 的非负性为优化模型带来约束，实践中会近似 $\ln \sigma^2$，而不是直接近似 $\sigma$
    CH13-并行计算
        CH13.1-并行计算基础
            数据并行指 workers 拥有全部模型参数，处理部分数据，模型并行指 workers 拥有全部数据，处理部分模型参数
            加速比通过钟表时间计算，并行情况下，处理器时间不变，钟表时间减少
            延迟通常与通信次数呈正比，即每次通信存在固定延迟，但无关通信量
        CH13.2-同步与异步
            异步情况下，worker 在完成计算后直接传递随机梯度给 server, server 执行更新，返回新参数
            该情况下，server 可能会收到过时的梯度，即基于旧参数计算的梯度，因此理论上，异步梯度下降具有很强随机性
        CH13.3-并行强化学习
            异步并行双 Q 学习使用异步梯度下降训练 DQN，每个 worker 维护自己的目标网络，在自己的环境中收集经验
            异步并行 A2C 类似

\[Doc\]
- [[doc-notes/numpy/user-guide/fundamentals-and-usage/NumPy Fundamentals|numpy/user-guide/fundamentals-and-usage/NumPy Fundamentals]]
- [[doc-notes/python/packages/gymnasium/environments/Classic Control|python/packages/gymnasium/environments/Classic Control]]

Date: 2025.3.24-2025.3.31

\[Paper\]
- [[paper-notes/distributed-system/ZooKeeper Wait-free coordination for Internet-scale systems-2010-ATC|2010-ATC-ZooKeeper Wait-free coordination for Internet-scale systems]]: All
    0-Abstract
        ZooKeeper aims to provide simple and high performance kernel for building complex coordination primitives at client.
        ZooKeeper service has wait-free property. ZooKeeper service provides a per client guarantee of FIFO execution of requests and provides linearizability for all requests that change ZooKeeper states.
        ZooKeeper service handle read requests in local servers.
    1-Introduction
        Large-scale distributed applications require different forms of coordination, like configuration, group membership, leader election, and locks.
        Services that implement stronger coordination primitives can be used to implemented weaker ones. For example, locking service with strong synchronization guarantees can be used to implement leader election, group membership, etc.
        ZooKeeper implements an coordination kernel, and provides API for distributed applications to implement their own coordination primitives.
        ZooKeeper implements an API that manipulates simple wait-free data objects: znodes. znodes are organized hierarchically as in file systems. Therefore, ZooKeeper API signatures is similar to file system API
        The wait-free property is important for performance and fault-tolerance, but is not sufficient for coordination. Thus, ZooKeeper also provides order guarantees for operations. Specifically, ZooKeeper guarantees FIFO client ordering of all operations and linearizability for write operations.
        ZooKeeper service is comprised of an ensemble of servers using replication, and is implemented using a pipelined architecture. The pipelined architecture allows requests to be asynchronously committed and naturally enables the FIFO execution of the requests from a client.
        The FIFO guarantee is provided by the pipelined architecture, and the linearizability guarantee is provided by a leader-based atomic broadcast protocol, called Zab.
        However, the typical workload of ZooKeeper is read. Thus, to improve read throughput, ZooKeeper servers process read locally, and will not be totally ordered by Zab.
        ZooKeeper clients can cache the identifier of the current leader. By watch mechanism, clients can be notified when the watching znode is updated.
    2-The ZooKeeper Service
        ZooKeeper client established a session when connecting to a ZooKeeper server, and uses the session handle to send requests.
        2.1-Service overview
            ZooKeeper provides an abstraction of hierarchically organized znodes for the clients. All znodes store data, and all znodes, except for ephemeral znodes, have children.
            A client can create two types of znodes: regular and ephemeral. Regular znodes should be explicitly deleted, and Ephemeral znodes will be automatically deleted when the session ends.
            When creating a new znode, the client can set a sequential flag, to append a monotonically increasing counter to its name.
            When send reading request, the client can set a watch flag, to make the server promise to notify the client when the information returned is changed in the future.
            Watches are one-time trigger in a session, and they will be unregistered once triggered or the session closes. Watches indicate the a change has happened, and does not provide the change.
            The data model of ZooKeeper is essentially a file system with simplified API and only full data write and reads, or a key/value table with hierarchical keys. The hierarchical namespace is useful for allocating subtrees of the namespace for different applications.
            Znodes are for metadata storage, and are essentially map to abstractions of the client applications, typically corresponding to metadata used for coordination purpose.
            ZooKeeper allows clients to used znode store some general meta data or configuration data. For example, the leader can store its information in a specific znode for other servers to read.
            Znodes are associated with timestamps and version numbers, which can be used to track changes to znodes and for conditional updates.
            Sessions enable a client to move transparently from one server to another, and hence persists across ZooKeeper servers.
        2.2-Client API
            All methods both have synchronous version and asynchronous version. Asynchronously committed requests are guaranteed to be executed in the commit order.
            Client does not use handle to access znodes. Each requests include the full path of the znode.
            All update methods receive an expected version number, which is used for conditional updates. If the actual version number does not match the desired one, update will fail.
        2.3-ZooKeeper guarantees
            ZooKeeper provides two basic guarantees: Linearizable writes and FIFO client order.
            Linearizable writes indicate that all update requests will be serialized and respect precedence..
        2.4-Examples of primitives
    3-ZooKeeper Applications
    4-ZooKeeper Implementation
        4.1-Request Processor
            When the leader receives a write request, it calculates what the state of system will be after applying this write operation. Then, the leader generate a transaction that captures the state. The transactions are idempotent.
        4.2-Atomic Broadcast
            Zab is just an implementation of multi-Paxos.
        4.3-Replicated Database
            ZooKeeper snapshots are called fuzzy snapshots, because the state will not be locked when snapshotting. Therefore, the snapshot result may not correspond to the state of ZooKeeper at any point in time. Because the transactions are idempotent, therefore it does not matter to take fuzzy snapshots.
        4.4-Client-Server Interactions
            When a server processes a write request, it will also send notifications to the watchers. Note that server handle notifications locally, only the server that a client is connected to will trigger the notification. Therefore, a write operation is committed does not mean that all watchers will receive notification immediately.
            Each read request is processed and tagged with a zxid, which corresponds to the last transaction processed by the server.
            `sync` primitive is used to synchronize the server with the leader, to ensure the server has the latest data.
            Consistency in server switch is handled by also zxid.
    5-Evaluation
        The atomic broadcast protocol limits the performance of ZooKeeper more than any other components.
    6-Related work
        Unlike Chubby, ZooKeeper allows clients connect to every ZooKeeper server instead of only the leader. ZooKeeper provides a more relaxed consistency model that allows server to handle read request locally.
    7-Conclusions
        ZooKeeper exposes wait-free data objects to clients to provide an approach to handle coordination problem.
        Although the consistency guarantees for read and watches is weak, but the combination of the linearizable writes and FIFO client requests are sufficient to implement sophisticated coordination protocols.
- [[paper-notes/rl/Continuous Control with Deep Reinforcement Learning-2016-ICLR|2016-ICLR-Continuous Control with Deep Reinforcement Learning]]: All
    Abstract
        This paper adapt the idea of Deep Q-Learning to the continuous action domain, and leads to a model-free, actor-critic algorithm based on deterministic policy gradient.
    Introduction
        DQN solves problem with high-dimensional observation space, but can only handle discrete or low-dimensional action spaces.
        However, many physical control has continuous and high-dimensional action space. DQN can not be directly applied to continuous problem because the optimal action selection process will need iterative optimization.
        This paper combine deterministic policy gradient with deep neural network. By the insights of DQN, we also use replay buffer to minimize the correlations between samples and use target networks to stabilize training.
    Background
    Algorithm
        DDPG create target network both for the policy network and value network. They are used to compute the target value for the TD update of the value network.
        The weights of target networks are soft updated.
        Target networks constrain the target value to change slowly, greatly improving the stability of learning.
        To unify different physical units in the observation dimensions and thus improve the generalizability of the algorithm. Batch norm is applied.
        To improve exploration, the behaviour policy is constructed by adding noise to the actor policy.
    Results
        The trained value network still tend to overestimate actual values.
    Related Work
    Conclusion
        The convergence steps of DDPG is significantly lower than those of DQN, about a factor of 20.

\[Doc\]
- [[doc-notes/python/packaging/discussions/install_requires vs requirements files|python/packaging/discussions/install_requires vs requirements files]]: All
- [[doc-notes/python/packages/gymnasium/environments/Box2D|python/packages/gymnasium/environments/Box2D]]: All
- [[doc-notes/go/package-documentation/plugin|go/package-documentation/plugin]]: All
- [[doc-notes/go/references/go.mod file reference|go/references/go.mod file reference]]
- [[doc-notes/go/command-documentation/go|go/command-documentation/go]]
- [[doc-notes/go/using-and-understanding-go/Effective Go|go/using-and-understanding-go/Effective Go]]
    Introduction
    Formatting
        Use tab for indentation. No parentheses for control structures.
    Commentary
    Names
        Packages name should be single-word, lowercase, and should be the base name of its source code directory.
    Semicolons

\[Blog\]
- [[casual-notes/setup.py vs requirements.txt|setup.py vs requirements.txt]]

## April
### Week 1
Date: 2025.3.31-2025.4.7

\[Paper\]
- [[paper-notes/rl/Approximately Optimal Approximate Reinforcement Learning-2002-ICML|2002-ICML-Approximately Optimal Approximate Reinforcement Learning]]: All (except Appendix)
    0-Abstract
        The conservative policy iteration algorithm uses a restart distribution, and an approximate greedy policy chooser to find an approximately optimal policy.
    1-Introduction
        In CPI, the policy is improved in a more uniform manner over the state-space and a more conservative policy update is performed where the new policy is a mixture of the current policy and the greedy policy.
        Improve policy in a more uniform manner incorporates exploration, and perform mixture update avoid the pitfalls of greedy dynamic programming methods.
        Such an algorithm can converge in a small number of steps and return an approximately optimal policy. The quantified claim of 'small' and 'approximately' is not related to the state space size.
    2-Preliminaries
        If the restart distribution $\mu$ is chosen to be a relatively uniform distribution, then $\mu$ can eliminate the explicit need for exploration.
        The goal of agent is the maximize the $\gamma$ -discounted average reward from the starting state distribution $D$.
        The value function can be normalized to $[0, R]$ when multiplied by $(1-\gamma)$.
        We can define a $\gamma$ -discounted future state distribution for a start distribution $\mu$, and rewrite the value function in terms of the discounted future state distribution.
        As $\gamma \to 1$, this distribution tends to be the stationary distribution for all $s$.
        The agent's goal can also be rewritten as an expectation with respace to the discounted future distribution.
    3-The Problems with Current Methods
        3.1-Approximate Value Function Methods
            Approximate value function methods refers to methods that use approximate value function to do policy iteration.
            Define $\epsilon$ as the $l_\infty$ loss between some approximator $\tilde V(s)$ and the truth value function $V(s)$. The performance of the greedy policy $\pi'$ based on this approximator is not guaranteed to improve monotonically. The possible decrease bound is related to $\epsilon$.
            The time required to make the policy attain certain performance level is also not guaranteed.
        3.2-Policy Gradient Methods
            Direct policy gradient method attempts to find a good policy among some restricted class of policies.
            In policy gradient method, the interested performance measure is guaranteed to improve.
            However, the gradient should also be estimated in practice, and the estimation will also rely on the estimation of the value function.
            The update of the estimated value function rely on the actually received reward during exploration. Therefore, exploration eventually will affect the estimation of the gradient.
    4-Approximately Optimal RL
        The problem of policy gradient method is that the optimization target $\eta_D$ is insensitive to policy improvement at unlikely states. And the policy improvement at unlikely states may be necessary for the agent to achieve the optimal.
        An alternative is to use another start distribution $\mu$ (instead of the original $D$) to weight the improvement from all states more uniformly.
        4.1-Policy Improvement
            The policy advantage measures the degree to which $\pi'$ is choosing actions with large advantages, with respect to the sate of states visited under $\pi$ starting from a state $s\sim \mu$.
            The greedy policy with respect to the value function will maximize the policy advantage.
            Using the policy advantage, the increment of $\eta_\mu$ ($\Delta \eta_\mu$) when updating the policy with the conservative update formula can be approximated by its first-order Taylor series with respect to $\alpha$.
            Therefore, for sufficiently small $\alpha$, if the policy advantage is positive, then the policy will improve. The larger the policy advantage, the more the policy will improve.
        4.2-Answering Question 3
            The Conservative policy iteration algorithm will finds a policy $\pi$ near the optimal in polynomial time with respect to $\epsilon$.
            The algorithm will stop when an $\epsilon$ small policy advantage is obtained, which means the optimal policy advantage of returned policy is less than $2\epsilon$.
            The convergence time bound of CPI do not rely on $\mu$, but the performance of the policy found do, since $\text{OPT}\mathbb (\mathbb A_{\pi, \mu}(\pi^*)) < 2\epsilon$.
            The performance difference of the policy found with respect to the real optimal policy (in terms of $\eta_D$) is bounded by a factor which represent the mismatch between the state distribution of current policy and the optimal policy. And a more uniform starting distribution will ensure the bound is not too large.
    5-Conservative Policy Iteration
    6-How Good is the Policy Found?
        Lemma 6.1 shows that the exact performance difference between the updated policy and the original policy is measured by an expectation with respect to the state distribution of the updated policy.
        Therefore, the policy advantage, which is calculated with respect to the state distribution of the original policy may not affect the true improvement.
        This motivates us to use a more uniform start distribution to drag the two state distribution more close to a certain amount of extent.
        Theorem 6.2 further supports this motivation.
    7-Discussion
        The greedy policy chooser aims to find a policy with greater policy advantage. Note that its convergence time is related to the size of the state space.
        The bounds presented in this paper show the importance of ensuring the agent starts in the state where the optimal policy tends to visit it.
        We can use the prior knowledge of which state an optimal policy tends to visit to choose $\mu$.
- [[paper-notes/rl/Trust Region Policy Optimization-2015-ICML|2015-ICML-Trust Region Policy Optimization]]: All (except Appendix)
    0-Abstract
        TRPO is effective for optimizing large nonlinear policies such as neural networks
        Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters.
    1-Introduction
        Approximate dynamic programming methods and gradient-based methods can not beat gradient-free random search. This is unsatisfying.
        We first prove that minimizing a certain surrogate objective function guarantees policy improvement with non-trivial step sizes. Then after applying a series of approximation to the theoretically-justified algorithm, we can get a practical algorithm.
    2-Preliminaries
        The equation 2 rewrite the objective $\eta(\tilde \pi)$ and implies that any policy update $\pi \to \pi'$ that has a nonnegative expected advantage at every state $s$, i.e. $\sum_a \tilde \pi(a\mid s) A_\pi(s, a)\ge0$, is guaranteed to increase the policy performance $\eta$.
        This also implies the classic policy iteration update, which use the deterministic policy $\tilde \pi(s) = \arg\max_a A_\pi(s, a)$, will improve the policy if there is at least one state-action pair with a positive advantage value and nonzero state visitation probability.
        Eq2 depends on $\rho_{\tilde \pi}(s)$. To make it easy to optimize directly, we introduce approximation, which substitute $\rho_{\tilde \pi}(s)$ with $\rho_\pi(s)$. Therefore we get an $L_\pi(\tilde \pi)$ as a surrogate objective for $\eta(\tilde \pi)$.
        Let us parameterize the policy with parameter vector $\theta$. The CPI paper indicates that the surrogate objective function $L_\pi$ matches the true objective function $\eta$ in the first order. As Equation4 indicates. 
        Therefore, a sufficiently small step that improves the surrogate objective will improve the true objective. But the actual step size is not determined.
        The CPI paper uses the mixture update, and provides explicit bound on the improvement of the true objective $\eta$. The bound is stated in equation 6, however, the bound only applies to mixture policies.
        It is desirable for a practical policy update scheme to be applicable to all general stochastic policy classes.
    3-Monotonic Improvements Guarantee for General Stochastic Policies
        The principle theoretical result of this paper is that the policy improvement bound in equation 6 can be extended to general stochastic polices, rather than just mixture policies.
        To generalize the policy update, the mixture coefficient is replaced with a distance measure between $\pi$ and $\tilde \pi$. The constant $\epsilon$ is changed appropriately.
        The particular distance measure we use is the total variance divergence.
        The extended bound is presented in equation 8.
        Following the relationship between the total variation divergence and the KL divergence, equation 8 can be extended to equation 9.
        Based on equation 9, algorithm 1 improves the lower bound on each step, and therefore is guaranteed to improve the policy.
        TRPO is an approximation to algorithm 1 by using a constraint on the KL divergence rather than a penalty to robustly allow large updates.
    4-Optimization of Parameterized Policies
        By approximation, we use the average KL divergence to substitute the largest KL divergence, and the penalty is also transformed into a constraint.
    5-Sample-Based Estimation of the Objective and Constraint
        By approximation, we substitute the expectations with their Monte Carlo estimation, and importance sampling is also used.
    6-Practical Algorithm
    7-Connections with Prior Work
- [[paper-notes/rl/Soft Actor-Critic Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor-2018-ICML|2018-ICML-Soft Actor-Critic Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor]]: All (except Appendix)
    0-Abstract
        Model-free RL algorithm suffer from two major challenges: very high sample complexity and brittle convergence properties.
        Soft actor-critic is an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework. The actor aims to maximize the expected return as well as the entropy.
   1-Introduction
       Model-free RL algorithm are expensive in terms of their sample complexity, and often sensitive with respect to their hyperparameters like learning rates and exploration constants.
       One cause of the poor sample efficiency is on-policy learning, which requires new samples to be collected at each gradient step.
       DDPG uses the actor to perform the maximization in Q-Learning, and thus provides sample-efficient learning, but DDPG is very sensitive to hypermeters.
       Therefore, we search for a model-free, sample-efficient, stable, off-policy algorithm for problems with continuous action spaces.
       Maximum entropy reinforcement learning alters the RL objective by adding an entropy term of the policy. The introduction of this term brings substantial improvement in exploration and robustness. Maximum entropy policies are robust in the face of model and estimation errors.
       SAC is better than previous methods in sample efficiency and performance.
    2-Related Work
        Actor-critic algorithms are typically derived from policy iteration. In large-scale RL problems, two stages of policy iteration: policy evaluation and policy improvement are impractical to run until convergence. Therefore the policy and value function are optimized jointly. In this case, the policy is referred as actor and the value function is referred as critic.
        DDPG use an deterministic actor to achieve off-policy actor-critic training. SAC use an stochastic actor to achieve off-policy actor-critic training. (The stochastic refers to the policy also aims to maximize its entropy, therefore leads to some kind of stochasticity. However, the policy itself in definition is still an deterministic transformation, its stochastic comes from the noise in the reparameterization trick, but it does not mean that the policy will always give a normal distribution, otherwise the maximum entropy term is meaningless. The policy function is actually map a Gaussian to another distribution, and the state directs the rough range of mapped values of a noise value. This range indicates the high possible area of actions to chose based on the state. The entropy term will make the policy inclined to expand the range)
    3-Preliminaries
        The new objective with entropy term motivates the policy to explore more widely, and to capture multiple modes of near optimal behaviour. We observe that the entropy term can also considerably improve the learning speed.
    4-From Soft Policy Iteration to Soft Actor-Critic
        Soft policy iteration alternates between policy evaluation and policy improvement in the maximum entropy framework.
        The soft policy evaluation compute the value function based on the maximum entropy objective. The lower bound of the soft value function is the original value function.
        The soft policy improvement finds the projection with the lowest KL divergence.
        Soft policy iteration is proved to converge to the optimal entropy policy among the specified policy set.
        To apply soft policy iteration to large continuous domain, we apply approximations. The policy and value functions are approximated by NN, and instead of running evaluation and improvements to convergence, we alternate between optimizing both networks with SGD.
        In training, two Q-functions are utilized to mitigate positive bias in the policy improvement step. We found that two Q-functions can significantly speed up training on hard tasks.
    5-Experiments
        In policy, the introduction of entropy prevents the premature convergence of the policy variance. In the value function, the introduction of entropy encourages exploration by increasing the value of regions of state space that leads to high entropy behaviour.
        The deterministic variant of SAC exhibits very high variance between seeds. This indicates that learning a stochastic policy with entropy term can stabilize training.
        It is beneficial to make the final policy deterministic.
        SAC is sensitive to the scaling of reward signal.
    6-Conclusion

\[Book\]
- [[book-notes/深度强化学习|深度强化学习]]: CH12
    CH12-模仿学习
        模仿学习不属于强化学习，但是目的与强化学习相同: 学习控制 agent 的最优策略
        模仿学习向人类专家学习，学习人类专家的策略，强化学习完全依赖于环境反馈学习
        CH12.1-行为克隆
            行为克隆本质是监督学习，其数据集由 (状态，动作) 二元组构成，其中动作都是人类专家策略基于状态做出的动作
            对于连续动作问题，策略网络输出动作向量，即执行回归
            对于离散动作问题，策略网络输出选择各个动作的概率向量，即执行多分类
            如果行为克隆的数据集不包含较为多样的状态和动作，训练出的策略在真实环境面临未见过的动作时，容易做出不好的决策，进而容易导致 “错误累加” 问题
            强化学习的探索性优于行为克隆，故强化学习训练的智能体可以高于人类专家水平，而行为克隆做不到
            强化学习的缺点在于需要和环境交互，需要探索时间，并且会改变环境，在真实物理世界中应用强化学习，需要考虑到初始化和探索带来的成本
            行为克隆的优势在于离线训练，避免和真实环境交互，成本低
        CH12.2-逆向强化学习
            逆向强化学习中，智能体可以和环境交互，但环境只给出下一个状态，不给出奖励
            逆向强化学习假设人类专家策略就是环境中的最优策略，利用人类专家策略反推奖励函数，然后用学习到的奖励函数执行强化学习
            根据最优策略的轨迹可以大致推断出奖励函数，但不能推断出奖励的具体大小，且最优策略不一定唯一地对应一个奖励函数
        CH12.3-生成判别模仿学习 (GAIL)
            GAIL 引入 GAN 的思想，Generator 拟合策略函数，接收状态输出各个动作的概率，Discriminator 同样接收状态输出各个动作的概率，对于人类专家可能执行的动作赋予高概率，对于策略网络可能执行的动作赋予低概率
            Generator 的训练基于强化学习方法，其中状态-动作对的奖励由判别器定义，本质目的仍然是让自己在 Discriminator 眼中更接近真实 (人类专家策略)
            Discriminator 的训练目是区分真实轨迹和 Generator 生成的轨迹

\[Doc\]
- [[doc-notes/gerrit/Quickstart for Installing Gerrit on Linux|gerrit/Quickstart for Installing Gerrit on Linux]]: All
- [[doc-notes/gerrit/about-gerrit/Why Code Review|gerrit/about-gerrit/Why Code Review]]: All
    gerrit implements a web interface for the workflow of code review
    gerrit uses change-id to identify a conceptual change. The change-id is the footer of every commit message.
    The web interface will group commits based on their change-id, so the reviewers can see how a conceptual change evolves.
    The change-id is randomly generated at first, and should be retained by utilizing the fact that `git commit --amend` retain the commit message by default.
- [[doc-notes/gerrit/about-gerrit/Product Overview|gerrit/about-gerrit/Product Overview]]: All
- [[doc-notes/gerrit/about-gerrit/How Gerrit Works|gerrit/about-gerrit/How Gerrit Works]]: All
    A typical project contains a central source repository (authoritative repository).
    Gerrit will add an additional pending changes repository, and all code changes are sent to the pending changes repository for review. A commit in the pending changes can be submit to the authoritative repository only when enough reviewers approve it.

### Week 2
Date: 2025.4.7-2025.4.14

\[Paper\]
- [[paper-notes/distributed-system/Chain Replication for Supporting High Throughput and Availability-2004-OSDI|2004-OSDI-Chain Replication for Supporting High Throughput and Availability]]: All
    0-Abstract
        Chain replication is an approach for coordinating clusters of fail-stop storage servers. This approach is intended for supporting storage service with high throughput, high availability, and strong consistency.
    1-Introduction
        A storage system typically implements operations like store, retrieve, change data for the clients. File systems and database systems are two common storage systems. File systems implement idempotent read and write operations for a single object, and database systems implement transaction for accessing multiple objects, and database systems implement serializable transaction for accessing multiple objects.
        This paper focus on storage systems that sit between file system and database system, called storage service. The storage service store objects, and support query and update operations. The update operations automatically change the state of a single object.
        A storage service with strong consistency guarantee satisfies (1) operations to query and update individual objects are executed in some sequential order. (2) the effects of update operations are reflected in results returned by subsequent query operations.
    2-A Storage Service Interface
        Chain replication provides two client API: `update, query`.
        The query operation is idempotent, and the update operation is not. Therefore, a client that re-issue a nonidempotent update request must take precautions to ensure the update has not already been performed. For example, issue a query first to check whether the result is already updated.
        The functionality of the storage service provided by chain replication can be specified by the object states and state transitions from the client view.
        The state are described by two variables: $Hist$ and $Pending$, which represents the list of update operations executed and the list of operations to be executed respectively.
        There are three possible state transitions, transition T1 add the arriving request to $Pending$, transition T2 ignore some specified pending request, transition T3 remove an request from $Pending$, and process it.
    3-Chain Replication Protocol
        In chain replication, an object replicated to $N$ servers can tolerate $N-1$ server failure while providing availability.
        In chain replication, every update request will be redirected to the head, and be automatically processed by the head. Every query update request will be redirected to the tail, and be processed by the tail. 
        The tail is responsible for generating response for all requests. The update request will be passed from the head to the tail. Only when the tail finish the updating, the whole process is considered finished.
        Because update and query operations are all sequentially processed in a single server, the strong consistency is guaranteed.
        3.1-Protocol Details
            In chain replication's implementation of the storage service functionality specification, $Hist$ is only stored in the tail, $Pending$ is the set of client requests received by any server in the chain but not yes processed by the tail.
            In the chain replication protocol, every transition is equivalent to either a no-op or to one of the three specified transitions. Therefore, chain replication protocol satisfies the specification.
        3.2-Coping with Server Failures
            We employ a master service to detect failure, change the configuration correspondingly, and notify the clients the new head and tail.
            When the head fail, the master remove it, and make the successor the new head. The master should also remove the requests that received by the original head but not forwarded to it successor from $Pending$. This operation is consistent with transition T2. For the client, if they find that their request does not be replied on time, they will just resend the request.
            When the tail fail, the master remove it, and make the predecessor the new tail. The master should also decrease $Pending$ and increase $Hist$, because the predecessor may have finished more requests than the original tail.
            When other servers fail, the master remove it, and notify its predecessor and successor the new configuration. The master should also guarantee that requests received by the failed server before its failure can still be forwarded (instead of be missed).
            Therefore, to guarantee it, the predecessor should forward the update requests that possibly have not be received by the successor to the successor. Only when those requests are sent, the predecessor can begin forward the newly received requests in the new configuration.
            When extending a chain, the new server is added as the tail. The new tail can begin operation after the $Hist$ is transmitted completely, so the clients will not see inconsistency in the query reply.
    4-Primary/Backup Protocols
        Compared to primary/backup protocols, the chain replication protocol spilt the task of update and query, ensuring the low latency and low cost of query processing.
        The disadvantage is that the update request is forwarded sequentially, therefore the reply generation for the an update request requires more time.
        Both protocols ensure strong consistency.
        The failure processing in chain replication protocol is faster than primary/backup protocol.
    5-Simluation Experiments
- [[paper-notes/Make LLM a Testing Expert Bringing Human-like Interaction to Mobile GUI Testing via Functionality-aware Decisions-2024-ICSE|2024-ICSE-Make LLM a Testing Expert Bringing Human-like Interaction to Mobile GUI Testing via Functionality-aware Decisions]]: All

\[Doc\]
- [[doc-notes/pytorch/docs/python-api/torch.onnx/torch.onnx|pytorch/docs/python-api/torch.onnx/torch.onnx]]: All
    `torch.onnx` module captures the computation graph from native PyTorch `torch.nn.Module` and convert it to an ONNX graph.
    There are two types of exporter to be called. Both can be called through function `torch.onnx.export()`
    TorchDynamo based exported utilize Dynamo to rewrite Python bytecode into an FX Graph. The FX Graph will be polished and then converted to an ONNX graph. The advantage is that the dynamic nature of model is preserved by the bytecode analysis.
    TorchScript based exported utilize TorchScript to trace the model and capture a static computation graph. The resulting computation graph does not record any control-flow, does not differentiate `train` and `eval` mode, and does not truly handle dynamic inputs.
- [[doc-notes/pytorch/docs/python-api/torch.onnx/TorchDynamo-based ONNX Exporter|pytorch/docs/python-api/torch.onnx/TorchDynamo-based ONNX Exporter]]: All
    The TorchDynamo-based ONNX exporter uses TorchDynamo to first convert `torch.nn.Module` into an FX graph, and then the export optimize the FX graph, and translate it to ONNX graph.
    When exporting, all we need is to provide the model and input to `torch.onnx.export()`, and the function will return an instance of `torch.onnx.ONNXProgram` which contains the exported graph and other information.
    `ONNXProgram.optimize()` method can optimize the graph with constant folding and redundant operator elimination. The optimization is done in-place.
- [[doc-notes/onnx-mlir/how-tos/Inference Using C, C++|onnx-mlir/how-tos/Inference Using C, C++]]: All
    ONNX C Runtime API provides four data structures: `OMTensor`, `OMTensorList`, `OMEntryPoint`, `OMSignature`
    All compiled model will have the same exact C function signature, which takes a list of tensor as input and output a list of tensor.
    When `omTensorDestory` is called, if the tensor has `onwership` flag to `true`, then the associated memory buffer will be freed. If `false` , then the user should free the buffer manually.
    `omTensorListDestroy` will invoke `omTensorDestroy` for all tensors it contains.
- [[doc-notes/onnx-mlir/how-tos/Inference Using Python|onnx-mlir/how-tos/Inference Using Python]]: All
    onnx-mlir have 5 binary libraries importable by Python using pybind. `PyOMCompileSession.hpp` to compile model, `PyExecutionSession.hpp` and `PyRuntime.py` to run models, `PyOMCompileExecutionSession.hpp` and `PyCompileAndRuntime.py` to compile and run models.
- [[doc-notes/onnx-mlir/development/Add an Operation|onnx-mlir/development/Add an Operation]]
- [[doc-notes/onnx-mlir/development/Testing Guidelines|onnx-mlir/development/Testing Guidelines]]
- [[doc-notes/llvm/getting-involved/LLVM Coding Standards|llvm/getting-involved/LLVM Coding Standards]]
    Introduction
    Languages, Libraries, and Standards
        LLVM are written using standard C++17.
        LLVM support libraries implement specialized data structures or functionalities missing in the standard library. (in namespace `llvm`)
        Prefer LLVM support library when the support library provides similar functionality as the standard library.
- [[doc-notes/cmake/reference-manuals/cmake-commands/cmake_parse_arguments|cmake/reference-manuals/cmake-commands/cmake_parse_arguments]]
- [[doc-notes/cmake/command-line-tools/cmake|cmake/command-line-tools/cmake]]
- [[doc-notes/github/ci,cd-and-devops/github-actions/About GitHub Actions|github/ci,cd-and-devops/github-actions/About GitHub Actions]]: All
- [[doc-notes/YAML v1.2|YAML v1.2]]
- [[doc-notes/go/using-and-understanding-go/Effective Go|go/using-and-understanding-go/Effective Go]]: Control Structures, Functions
    Control Structures
        In Go, the only loop structure is `for`. 
        The loop structure and condition structure can accept optional initialization statements.
        `break` and `continue` can take optional label to identify where to break or continue.
        There are no parentheses for the control structure, and the code block must be brace-limited. 
    Functions
        The return or result parameters can have names.
        `defer` defer the function to be run immediately before the function executing the `defer` returns. `defer` is effective for handling resource release regardless which path the function takes. The canonical examples are unlocking a mutex or closing a file.


