# 2024
## July
### Week 4
\[Book\]
-  [[Programming Massively Parallel Processors A Hands-on Approach-2023|Programming Massively Parallel Processor A Hands-on-Approach]]: CH2-CH6.3, CH10
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
-  [[NVIDIA Nsight Compute]]: CH2

\[Blog\]
-  [CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275): 0%-50%
    Derived Ideas:
        1. Thread Tile: 改变 Thread Tile 内矩阵的运算顺序，利用 Register 减少对 global memory 的访问次数；其中 Thread tile 的长宽 $M_{frag},N_{frag}$ 的选取与线程内 FFMA 指令对非 FFMA 指令如 LDS 指令的延迟覆盖是相关的
## August
### Week 1
\[Book\]
-  [[Parallel Thread Execution ISA v8.5-2024|PTX ISA v8.5]]

\[Doc\]
-  [[CUDA-GDB v12.6]]: CH1-CH8

\[Blog\] 
-  [CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275): 0%-50%
    Derived Ideas:
        1. Arithmetic Intensity: 通过衡量计算方式的算数密度，将其乘上相应带宽，可以得到理论的 FLOPS 上限
        2. Thread Block Tile: 减少 Global Memory 读取
        3. Thread Tile & Warp Tile: 改变矩阵乘法顺序，调整 Tile 形状，提高 Arithmetic Intensity，使 FMA 可以掩盖 LDS 的延迟
        4. Pipeline: 由于改变矩阵乘法顺序增大了单线程的寄存器使用量，导致 Warp 数量降低，进一步导致 Occupancy 降低，因此考虑流水并行 Global Memory to Shared Memory、Shared Memory to Register、Computation in Register 这三个操作，提高 Warp 的指令并行度，以提高硬件占用率
-  [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370)
    Derived Ideas:
        1. Coarsening: 一个线程计算 $4\times 4$ 的结果，提高线程的算数密度
        2. `LDS.128`: 读取 `float4` 向量类型，减少 Shared Memory 访问
-  [cuda 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788)
    Derived Ideas:
        1. Align: 令 Shared Memory 内数据地址对齐
- [CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://zhuanlan.zhihu.com/p/518857175)

\[Code\]
- CUDA GEMM Optimization Project
    `matmul_v0.cu` : naive implementation

### Week 2
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
\[Code\]
- CUDA GEMM Optimization Project
    `matmul_t_v3.cu` - `matmul_t_v4.cu`
        `matmul_t_v3.cu` : swizzled implementation
        `matmul_t_v4.cu` : adjusted the tile size
### Week 4
\[Paper\]
- [[A Survey of Large Language Models v13-2023|A Survry of Large Language Models]]: Sec1-Sec5

\[Book\]
- [[Pro Git]]: CH 7.1
- [[Mastering CMake]]: CH1-CH7

## September
### Week 1
\[Book\] 
-  [[Mastering CMake]]: CH8-CH13、CH14 (Cmake Tutorial)

### Week 2
\[Paper\]
-  [[A Survey of Large Language Models v13-2023|A Survey of Large Language Models]]: CH6-CH7
    CH6: Prompt tricks: (input-output) pair, (input-reasoning step-output) triplet, plan
-  [[Are Emergent Abilities of Large Language Models a Mirage-2023-NeurIPS|Are Emergent Abilities of Large Language Models a Mirage?]]

\[Book\]
- [[Introductory Combinactorics-2009|Introductory Combinactorics]]: CH1
    CH1: Combinactorics: existence, enumeration, analysis, optmization of discrete/finite structures
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH2
    CH2: Basic knowledges: Conditional Independence, MAP query, Condisional density function, graphs

\[Doc\]
- [[Intel NPU Acceleration Library Documentation v1.3.0]]
- [[The Python Tutorial]]: CH1-CH16

### Week 3
\[Book\]
- [[Introductory Combinactorics-2009|Introductory Combinactorics]]: CH2
    CH2-Permutations and Combinations: Permutation/Combination of Sets (combination = permutation + division), Permutation/Combination of Multisets (permutation of sets + division/solutions of linear equation) , classical probability
- [[book-notes/Convex Optimization|Convex Optimization]]: CH2-CH2.5
    CH2-CH2.5-Convex Sets: Lots of definitions: convex combination, affine combination, some typical convex sets, operations that preserve convexity, supporting/seperating hyperplane
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH3-CH3.3
    CH3-CH3.3-The Baysian Network Representation: Baysian Network: Express conditional indepdencies in joint probability in a graph semantics, factorizing the joint probability into a product of CPDs according to the graph structure
- [[A Tour of C++]]: CH1-CH1.7

### Week 4
\[Paper\]
- [[A Survey of Large Language Models v13-2023|A Survry of Large Language Models]]: CH7
    CH7-Capacity and Evaluation: LLM abilities: 1. basic ability: language generation (including code), knowledge utilization (e.g. knowledge-intensive QA) , complex reasoning (e.g. math) ; 2. advanced ability: human alignment, interaction with external environment (e.g. generate proper action plan for embodied AI), tool manipulate (e.g. call proper API according to tasks); introduction to some benchmarks

\[Book\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH5
    CH5-Local Probabilistic Models: Compact CPD representation: Utilize context-specific independence to compactly represent CPD; Independent causal influence model: noisy-or model, BN2O model, generalized linear model (scores are linear to all parent variables), conditional linear gaussian model ( induces a joint distribution that has the form of a mixture of Gaussians)
- [[A Tour of C++]]: CH1.7-CH3.5

## October
### Week 1
\[Paper\]
- [[A Survey of Large Language Models v13-2023|A Survry of Large Language Models]]: CH8-CH9
    CH8-Applicatoin: LLM application in various tasks
    CH9-Conclusion and future directions
- [[Importance Sampling A Review-2010|Importance Sampling: A Review]]
    IS is all about variance reduction for Monte Carlo approximation;
    Adaptive parametric Importance Sampling: $q (x)$ be defined as a multivariate normal or student distribution, then optimizing a variation correlated metric to derive an optimal parameter setting for that distribution;
    Sequential Importance Sampling: Chain decompose $p (x)$, and chain construct $q (x)$;
    Anneal Importance Sampling: Sequentially approximate $p (x)$, much like diffusion;

\[Book\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH6-CH6.2
    CH6-Template-based Representations: temporal models; Markov assumption + 2-TBN = DBN; DBN usually be modeled as state-observation model (the state and observation are considered seperately; observation doesn't affect the state), two examples: HMM, linear dynamic system (all the dependencies are linear Gaussian)
- [[面向计算机科学的组合数学]]: CH1.7
    CH1.7-生成全排列: 中介数和排列之间的一一对应关系
- [[A Tour of C++]]: CH3.5-CH5

### Week 2
\[Paper\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: CH0-CH3.1
    CH0-CH3.1: Abstract, Background, Algorithm 1; Algorithm 1 is basically a tiled implementation of attention calculation. What makes algorithm 1 looks not so intuitive is the repetitive rescaling of softmax factor, whose aim is to stabilize the computation. In algorithm 1, each query's attention result is accumlated gradually by the outer loop, and the already accumlated partial attention result's weights for the corresponing value is dynamically updated/changed by the outer loop.

\[Book\]
- [[A Tour of C++]]: CH5
- [[面向计算机科学的组合数学]]: CH2.1-CH2.3
    CH2-鸽巢原理: 鸽巢原理仅解决存在性问题
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH4-CH4.3.1
    CH4-CH4.3.1: Markov Network's parameterization: the idea was derived from statictical physics, which is pretty intuitive by using factor to represent two variables' interaction/affinity, and using a normalized product of factors to represent a joint probability (Gibbs distribution) to describe the probability of paticular configuration; seperation criterion in Markov network is sound and weakly complete (sound: independence holds in network --> independence holds in all distribution factorizing over network; weakly complete: independence does not hold in network --> independence does not hold in some distribution factorizing over network)

\[Doc\]
- [[ultralytics v8.3.6]] : Quickstart, Usage(Python usage, Callbacks, Configuration, Simple Utilities, Advanced Customization)
    Brief Introduction to YOLO model's python API, which is pretty simple

### Week 3
\[Paper\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: Sec3.1-Sec5
    Sec3.1-IO Analysis: The IO complexity of FlashAttention is $\Theta(N^2d^2M^{-1})$ while the IO compexity of the standard attention computation is $\Theta (Nd + N^2)$. The main difference is in $M$ and $N^2$. Standard attention computation does not use SRAM at all, all memory accesses are global memory access, in which the process of fetching "weight matrix" $P\in \mathbb R^{N\times N}$ contributes most of the IO complesity. FlashAttention utilized SRAM, and do not store "weight matrix" into DRAM, but keep a block of it on chip the entire time, thus effectively reduced the IO complexity.
    Sec3.2-Block sparse Flash-Attention: The main difference between FlashAttention is that the range of "attention" is restricted, thereby the computation and memory accesses is reduced by skipping the masked entries.
    Sec4-Experiments: FlashAttention trains faster; FlashAttention trains more memory-efficient (linear), thus allowing longer context window in training. The reason for that is FlashAttention do not compute the entire "weight matrix" $P\in \mathbb R^{N\times N}$ one time, but do a two level loop, compute one row of $P$ each time. The FLOP is actually increased, but the memory usage is restricted to $O (N)$ instead of $O (N^2)$ and the additional computation time brought by the increased FLOP is eliminated by the time reduced by less DRAM accesses.
- [[Spatial Interaction and the Statistical Analysis of Lattice Systems-1974|1974-Spatial Interaction and the Statistical Analysis of Lattice Systems]]: Sec0-Sec2
    Sec0-Summary: This paper proposed an alternative proof of HC theorem, thereby reinforcing the importance of conditional probability models over joint probability models for modeling spatial interaction.
    Sec1-Sec2: For positive distribution, conditional probability can be used to deduce the overall joint probability. This is made possible by HC theorem.

\[Book\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH4.3.1-CH4.4.2
    CH4.3.1-CH4.4.2: Markov network encodes three types of independence: pairwise independence, local independence (Markov blanket), global independence (d-seperation). For positive distribution, they are equivalent. For non-positive distribution (those with deterministic relationships), they are not equivalent. This is because the semantics of Markov network is not enough to convey deterministic relationships. By HC theorem, $P$ factorizes over Markov network $\mathcal H$ is equivalent to $P$ satisfies the three types of independence encoded by $\mathcal H$.
- [[面向计算机科学的组合数学]]: CH3-CH3.3
    母函数：使用幂级数表示数列（数列由幂级数的系数构造）
- [[A Tour of C++]]: CH6

\[Doc\]
- [[Pytorch 2.x]]: CH0
    CH0-General Introduction: `torch.compile` : TorchDynamo --> FX Graph in Torch IR --> AOTAutograd --> FX graph in Aten/Prims IR --> TorchInductor --> Triton code/OpenMP code...
- [[doc-notes/triton/Getting Started|Triton: Tutorials]]: Vector Addition, Fused Softmax
    Triton is basically simplified CUDA in python, the general idea about parallel computing is similar. The most advantageous perspective about Triton is that it encapasulates all the compilcated memory address mapping work into a single api `tl.load` . Memory address mapping work is the most difficult part of writing CUDA code.

### Week 4
\[Paper\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: SecA-SecE
    SecA-Related Work
    SecB-Algorithm Details: Memory-efficient forward/backward pass: using for-loop to avoid storing $O(N^2)$ intermediate matrix; FlashAttention backward pass: In implementation, the backward algorithm of FlashAttention is actually simpler than the forward algorithm, because it's just about tiled matrix multiplication without bothering softmax rescaling
    SecC-Proofs: just counting, nothing special
    SecE-Extension Details: block-sparse implementation is just skipping masked block, nothing special
    SecF-Full Experimental Results
- [[Spatial Interaction and the Statistical Analysis of Lattice Systems-1974|1974-Spatial Interaction and the Statistical Analysis of Lattice Systems]]: Sec3
    Sec3-Markov Fields and the Harmmersly-Clifford Theorem: define ground state -> define Q function -> expand Q function -> proof the terms in Q function (G function) are only not null when their relating variables form a clique

\[Book\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH4.5
    CH4.5-Bayesian Networks and Markov Networks: chordal graph can be represented by either sturcture without loss of information 

## November
### Week 1
\[Paper\]
- [[FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning-2024-ICLR|2024-ICLR-FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning]]
    FlashAttention-2: 
    (1) tweak the algorithm, reducing the non-matmul op: remove the rescale of softmax weights in each inner loop, only do it in the end of inner loop
    (2) parallize in thread blocks to improve occupancy: exchange the inner loop and outer loop,  which makes each iteration in outer loop independent of each other, therefore parallelize them by assigning $\mathbf {O}$ blocks to thread blocks
    (3) distribute the work between warps to reduce shared memory communication: divide $\mathbf Q$ block to warps and keep $\mathbf {K, V}$ blocks intact, the idea is similar to exchanging outer loop and inner loop, which makes the $\mathbf O$ blocks the warp responsible for be independent of each other, thus primarily reducing the shared memory reads/writes for the final accumulation
    FlashAttention-2 also uses thread blocks to load KV cache in parallel for iterative decoding

\[Book\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH4.6.1
    CH4.6.1-Conditional Random Fields: CRF models conditional distribution by partially directed graph, whose advantage lies in its more flexibility. CRF allows us to use Markov network's factor decomposition semantics to represent conditional distribution. The specification of factors has lots of flexibility compared to explicitly specifying CPD in conditional Bayesian networks. But this flexibility in turn restrict expandability, because the parameters learned has less semantics on their own.
- [[面向计算机科学的组合数学]]: CH4-CH4.4.1
    Make general term the coefficient in generating function to relating generating function with recurrence relation, and then turn recurrence formula into a equation about generating function, thus solve the generating function, then derive the general term of the recurrence.

\[Doc\]
- [[Learn the Basics|pytorch/tutorial/beginner/Learn the Basics]] 
- [[python/packages/pillow v11.0.0]] : Overview, Tutorial, Concepts
- [[Repositories|huggingface/hub/Repositories]]: Sec1-Sec4
- [[doc-notes/triton/Getting Started|triton/Getting Started]]:  Tutorials/Matrix Multiply
- [[Argparse Tutorial|python/how/general/Argparse Tutorial]] 
- [[nvidia/CUDA C++ Programming Guide v12.6]]: CH1

### Week 2
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
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH7, CH9.2-CH9.3
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
    CH9.2-'Variable Elimination: The Basic Ideas':
        The basic idea of variable elimination is dynamic programming. To calculate the margianal $P(X)$ from CPD $P (X \mid Y)$, we first calculate the marginal of $Y$ and store it, thus calculate the marginal $P (X)$ by $P (X) = \sum_y P (y) P (X\mid y)$, avoiding recalculate $P (Y)$ for every $x \in Val (X)$
    CH9.3-Variable Elimination:
        We viewing the joint density as a product of factors. To calculate the marginalization over a subset of variables, we sum out of the other variables. This process is generalized as the sum-product variable elimication algorithm. The calculation of this algorithm can be simplified by using the property of factor's limited scope to isolated only the related factors to summation (distributive law of multiplication)
        To deal with evidence, we first use the evidence to reduce the factors (leaving the factors compatible with the evidence), and do the same algorithm to the reduced set of factors.
- [[A Tour of C++]] : CH7-CH8

\[Doc\]
- [[python/howto/general/Annotations Best Practices]]
    Best Practice after Python 3.10: use `inspect.get_annotations()` to get any object's annotation
- [[huggingface/hub/Repositories]]: Sec4-Sec10

### Week 3
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
- [[面向计算机科学的组合数学]]: CH4.4.1-CH4.5.2
    Write characteristic polynominal directly from the recurrence relation, and slove the characteristic equation to get $\alpha_i$ s. Then write the general term in terms of $\alpha_i$ s and undermined coefficients. Finally use the initial values to solve the coefficients, and derive the general term formula.


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
        Repository is a collection of related images in the registry.
        Keep each container doing only one thing.
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
\[Book\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH10.1-CH10.3, CH11.1-CH11.3.4
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
- [[面向计算机科学的组合数学]]: CH5.1-CH5.2

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
- [[paper-notes/mlsys/The Deep Learning Compiler A Comprehensive Survey-2020-TDPS|2020-TDPS-The Deep Learning Compiler A Comprehensive Survey]]

\[Book\]
- [[book-notes/Probabilistic Graphical Models-Principles and Techniques|Probabilistic Graphical Models-Principles and Techniques]]: CH17.4, CH19, CH20
- [[book-notes/一份（不太）简短的 LaTeX2e 介绍|一份（不太）简短的 LaTeX2e 介绍]]: CH3

\[Doc\]
- [[doc-notes/python/howto/general/Regular Expression HOWTO|python/howto/general/Regular Expression HOWTO]]
- [[doc-notes/matplotlib/user-guide/Quick start guide|matplotlib/user-guide/Quick start guide]]
