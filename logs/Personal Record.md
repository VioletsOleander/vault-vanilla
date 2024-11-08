# 2024
## July
### Week 4 
**1. Project: CUDA GEMM Optimization [Working]** 
**Book**: [[Programming Massively Parallel Processors A Hands-on Approach-2023|Programming Massively Parallel Processor A Hands-on-Approach]]
  Range: CH2-CH6.3、CH10
  State: Quick Reading and Noting Finished
  Derived Ideas: 
  1. Tiling: 搬运数据from Global Memory to Shared Memory
  2. Coaleasing: 利用DRAM burst优化Tiling过程中对Global Memory的访问次数 
  3. Corsening(Optional)

**Blog**: [CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275) 
  Range: 0%-50%
  State: Quick Reading Finished
  Derived Ideas:
1. Thread Tile: 改变Thread Tile内矩阵的运算顺序，利用Register减少对global memory的访问次数；其中Thread tile的长宽$M_{frag},N_{frag}$的选取与线程内FFMA指令对非FFMA指令如LDS指令的延迟覆盖是相关的

**Book**: [[CUDA C++ Programming Guide v12.5-2024|CUDA C++ Programming Guide v12.5]] 
  Range: CH5-CH11、CH19
  State: Quick Reading and Noting Finished
  Derived Ideas:
  1. Avoid Bank Confilct: 数据访问对齐32bit的Bank Size
  2. Occupancy Calculator: 利用工具计算一下合适的Blocksize和Gridsize
  3. Coaleasing: Warp对Shared Memory的访问同样可以进行合并优化
  4. `memcpy_async()` (Questioned): 异步访问，访存与计算流水线

**Book**: [[Managing Projects with GNU Make-2011|Managing Projects with GNU Make]]
  Range: CH1-CH2.7
  State: Quick Reading and Noting Finished

**Doc**: [[NVIDIA Nsight Compute]]
  Range: CH2
  State: Quick Rough Reading and Noting Finished
## Augest
### Week 1
**1. Project: CUDA GEMM Optimization [Working]**
**Blog**: [CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275) 
  Range: 0%-50%(Rough)
  State: Quick Reading Finished
  Derived Ideas:
  1. Arithmetic Intensity: 通过衡量计算方式的算数密度，将其乘上相应带宽，可以得到理论的FLOPS上限
  2. Thread Block Tile: 减少Global Memory读取
  3. Thread Tile & Warp Tile: 改变矩阵乘法顺序，调整Tile形状，提高Arithmetic Intensity，使FMA可以掩盖LDS的延迟
  4. Pipeline: 由于改变矩阵乘法顺序增大了单线程的寄存器使用量，导致Warp数量降低，进一步导致Occupancy降低，因此考虑流水并行Global Memory to Shared Memory、Shared Memory to Register、Computation in Register这三个操作，提高Warp的指令并行度，以提高硬件占用率

**Blog**: [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370)
  Range: 0%-100%(Rough)
  State: Quick Reading Finished
  Derived Ideas:
  1. Corsening: 一个线程计算$4\times 4$的结果，提高线程的算数密度
  2. `LDS.128`: 读取 `float4` 向量类型，减少Shared Memory访问

**Blog**: [cuda 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788)
  Range: 0%-100%(Rough)
  State: Quick Reading Finished
  Derived Ideas:
  1. Align: 令Shared Memory内数据地址对齐

**Blog**: [CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://zhuanlan.zhihu.com/p/518857175)
  Range: 0%-100%(Rough)
  State: Quick Reading Finished

**Book**: [[Parallel Thread Execution ISA v8.5-2024|PTX ISA v8.5]]
  Range: All Chapters(Rough)
  State: Quick Reading and Noting Finished

**Doc**: [[CUDA-GDB v12.6]]
  Range: CH1-CH8(Rough)
  State: Quick Reading and Noting Finished

**Code**: `matmul_v0.cu`
  `matmul_v0.cu` : naive implementation
### Week 2
**1. Project: CUDA GEMM Optimization [Working]**
**Code**: `matmul_v1.cu` - `matmul_v7.cu`
  `matmul_v1.cu` : block tiled implementation
  `matmul_v5.cu` : block tiled and thread tiled implementation
  `matmul_v6/v7.cu` : block/thread tiled and pipelined implementation

**Code**: `matmul_t_v0.cu` - `matmul_t_v2.cu` 
  `matmul_t_v0/v1.cu` : block tiled and warp tiled implementation
  `matmul_t_v2.cu` : bank conflict partially solved implementation
### Week 3
**1. Project: CUDA GEMM Optimization [Working]**
**Code**: `matmul_t_v3.cu` - `matmul_t_v4.cu`
`matmul_t_v3.cu` : swizzled implementation
`matmul_t_v4.cu` : adjusted the tile size
### Week 4
**Paper**: [[A Survey of Large Language Models v13-2023|A Survry of Large Language Models]] 
  Range: Section1-Section5
  State: Read and Noted

**Book**: [[Pro Git]]
  Range: CH7.1
  State: Read and Noted

**Book:** [[Mastering CMake]]
  Range: CH1-CH7
  State: Read and Noted
## September
### Week 1
\[**Book**\] 
-  [[Mastering CMake]]: CH8-CH13、CH14 (Cmake Tutorial)
    Read and Practiced
### Week 2
\[**Paper**\]
-  [[A Survey of Large Language Models v13-2023|A Survey of Large Language Models]]: CH6-CH7
    CH6: Prompt tricks: (input-output) pair, (input-reasoning step-output) triplet, plan
-  [[Are Emergent Abilities of Large Language Models a Mirage-2023-NeurIPS|Are Emergent Abilities of Large Language Models a Mirage?]]: All

\[**Doc**\]
- [[Intel NPU Acceleration Library Documentation v1.3.0]]
- [[The Python Tutorial]]: CH1-CH16

\[**Book**\]
- [[Introductory Combinactorics-2009|Introductory Combinactorics]]: CH1
    CH1: Combinactorics: existence, enumeration, analysis, optmization of discrete/finite structures
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH2
    CH2: Basic knowledges: Conditional Independence, MAP query, Condisional density function, graphs
### Week 3
\[**Book**\]
- [[Introductory Combinactorics-2009|Introductory Combinactorics]]: CH2
    CH2-Permutations and Combinations: Permutation/Combination of Sets (combination = permutation + division), Permutation/Combination of Multisets (permutation of sets + division/solutions of linear equation) , classical probability
- [[book-notes/Convex Optimization|Convex Optimization]]: CH2-CH2.5
    CH2-CH2.5-Convex Sets: Lots of definitions: convex combination, affine combination, some typical convex sets, operations that preserve convexity, supporting/seperating hyperplane
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH3-CH3.3
    CH3-CH3.3-The Baysian Network Representation: Baysian Network: Express conditional indepdencies in joint probability in a graph semantics, factorizing the joint probability into a product of CPDs according to the graph structure
- [[A Tour of C++]]: CH1-CH1.7

### Week4
\[**Paper**\]
- [[A Survey of Large Language Models v13-2023|A Survry of Large Language Models]]: CH7
    CH7-Capacity and Evaluation: LLM abilities: 1. basic ability: language generation (including code), knowledge utilization (e.g. knowledge-intensive QA) , complex reasoning (e.g. math) ; 2. advanced ability: human alignment, interaction with external environment (e.g. generate proper action plan for embodied AI), tool manipulate (e.g. call proper API according to tasks); introduction to some benchmarks

\[**Book**\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH5
    CH5-Local Probabilistic Models: Compact CPD representation: Utilize context-specific independence to compactly represent CPD; Independent causal influence model: noisy-or model, BN2O model, generalized linear model (scores are linear to all parent variables), conditional linear gaussian model ( induces a joint distribution that has the form of a mixture of Gaussians)
- [[A Tour of C++]]: CH1.7-CH3.5

## October
### Week 1
\[**Paper**\]
- [[A Survey of Large Language Models v13-2023|A Survry of Large Language Models]]: CH8-CH9
    CH8-Applicatoin: LLM application in various tasks
    CH9-Conclusion and future directions
- [[Importance Sampling A Review-2010|Importance Sampling: A Review]]
    IS is all about variance reduction for Monte Carlo approximation;
    Adaptive parametric Importance Sampling: $q (x)$ be defined as a multivariate normal or student distribution, then optimizing a variation correlated metric to derive an optimal parameter setting for that distribution;
    Sequential Importance Sampling: Chain decompose $p (x)$, and chain construct $q (x)$;
    Anneal Importance Sampling: Sequentially approximate $p (x)$, much like diffusion;

\[**Book**\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH6-CH6.2
    CH6-Template-based Representations: temporal models; Markov assumption + 2-TBN = DBN; DBN usually be modeled as state-observation model (the state and observation are considered seperately; observation doesn't affect the state), two examples: HMM, linear dynamic system (all the dependencies are linear Gaussian)
- [[面向计算机科学的组合数学]]: CH1.7
    CH1.7-生成全排列: 中介数和排列之间的一一对应关系
- [[A Tour of C++]]: CH3.5-CH5

### Week 2
\[**Paper**\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: CH0-CH3.1
    CH0-CH3.1: Abstract, Background, Algorithm 1; Algorithm 1 is basically a tiled implementation of attention calculation. What makes algorithm 1 looks not so intuitive is the repetitive rescaling of softmax factor, whose aim is to stabilize the computation. In algorithm 1, each query's attention result is accumlated gradually by the outer loop, and the already accumlated partial attention result's weights for the corresponing value is dynamically updated/changed by the outer loop.

\[**Book**\]
- [[A Tour of C++]]: CH5
- [[面向计算机科学的组合数学]]: CH2.1-CH2.3
    CH2-鸽巢原理: 鸽巢原理仅解决存在性问题
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH4-CH4.3.1
    CH4-CH4.3.1: Markov Network's parameterization: the idea was derived from statictical physics, which is pretty intuitive by using factor to represent two variables' interaction/affinity, and using a normalized product of factors to represent a joint probability (Gibbs distribution) to describe the probability of paticular configuration; seperation criterion in Markov network is sound and weakly complete (sound: independence holds in network --> independence holds in all distribution factorizing over network; weakly complete: independence does not hold in network --> independence does not hold in some distribution factorizing over network)

\[**Doc**\]
- [[ultralytics v8.3.6]] : Quickstart, Usage(Python usage, Callbacks, Configuration, Simple Utilities, Advanced Customization)
    Brief Introduction to YOLO model's python API, which is pretty simple

### Week 3
\[**Paper**\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: Sec3.1-Sec5
    Sec3.1-IO Analysis: The IO complexity of FlashAttention is $\Theta(N^2d^2M^{-1})$ while the IO compexity of the standard attention computation is $\Theta (Nd + N^2)$. The main difference is in $M$ and $N^2$. Standard attention computation does not use SRAM at all, all memory accesses are global memory access, in which the process of fetching "weight matrix" $P\in \mathbb R^{N\times N}$ contributes most of the IO complesity. FlashAttention utilized SRAM, and do not store "weight matrix" into DRAM, but keep a block of it on chip the entire time, thus effectively reduced the IO complexity.
    Sec3.2-Block sparse Flash-Attention: The main difference between FlashAttention is that the range of "attention" is restricted, thereby the computation and memory accesses is reduced by skipping the masked entries.
    Sec4-Experiments: FlashAttention trains faster; FlashAttention trains more memory-efficient (linear), thus allowing longer context window in training. The reason for that is FlashAttention do not compute the entire "weight matrix" $P\in \mathbb R^{N\times N}$ one time, but do a two level loop, compute one row of $P$ each time. The FLOP is actually increased, but the memory usage is restricted to $O (N)$ instead of $O (N^2)$ and the additional computation time brought by the increased FLOP is eliminated by the time reduced by less DRAM accesses.
- [[Spatial Interaction and the Statistical Analysis of Lattice Systems-1974|1974-Spatial Interaction and the Statistical Analysis of Lattice Systems]]: Sec0-Sec2
    Sec0-Summary: This paper proposed an alternative proof of HC theorem, thereby reinforcing the importance of conditional probability models over joint probability models for modeling spatial interaction.
    Sec1-Sec2: For positive distribution, conditional probability can be used to deduce the overall joint probability. This is made possible by HC theorem.
\[**Book**\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH4.3.1-CH4.4.2
    CH4.3.1-CH4.4.2: Markov network encodes three types of independence: pairwise independence, local independence (Markov blanket), global independence (d-seperation). For positive distribution, they are equivalent. For non-positive distribution (those with deterministic relationships), they are not equivalent. This is because the semantics of Markov network is not enough to convey deterministic relationships. By HC theorem, $P$ factorizes over Markov network $\mathcal H$ is equivalent to $P$ satisfies the three types of independence encoded by $\mathcal H$.
- [[面向计算机科学的组合数学]]: CH3-CH3.3
    母函数：使用幂级数表示数列（数列由幂级数的系数构造）
- [[A Tour of C++]]: CH6
\[**Doc**\]
- [[Pytorch 2.x]]: CH0
    CH0-General Introduction: `torch.compile` : TorchDynamo --> FX Graph in Torch IR --> AOTAutograd --> FX graph in Aten/Prims IR --> TorchInductor --> Triton code/OpenMP code...
- [[Getting Started|Triton: Tutorials]]: Vector Addition, Fused Softmax
    Triton is basically simplified CUDA in python, the general idea about parallel computing is similar. The most advantageous perspective about Triton is that it encapasulates all the compilcated memory address mapping work into a single api `tl.load` . Memory address mapping work is the most difficult part of writing CUDA code.

### Week4
\[**Paper**\]
- [[FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness-2022-NeruIPS|2022-NeurIPS-FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]]: SecA-SecE
    SecA-Related Work
    SecB-Algorithm Details: Memory-efficient forward/backward pass: using for-loop to avoid stroing $O(N^2)$ intermediate matrix; FlashAttention backward pass: In implementation, the backward algorithm of FlashAttention is actually simpler than the forward algorithm, because it's just about tiled matrix multiplication without bothering softmax rescaling
    SecC-Proofs: just counting, nothing special
    SecE-Extension Details: block-sparse implementation is justing skipping masked block, nothing special
    SecF-Full Experimental Results
- [[Spatial Interaction and the Statistical Analysis of Lattice Systems-1974|1974-Spatial Interaction and the Statistical Analysis of Lattice Systems]]: Sec3
    Sec3-Markov Fields and the Harmmersly-Clifford Theorem: define ground state -> define Q function -> expand Q function -> proof the terms in Q function (G function) are only not null when their relating variables form a clique
\[**Book**\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH4.5
    CH4.5-Bayesian Networks and Markov Networks: chordal graph can be represented by either sturcture without loss of information 

## November
### Week 1
\[Paper\]
- [[FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning-2024-ICLR|2024-ICLR-FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning]]
    FlashAttention-2: 
    (1) tweak the algorithm, reducing the non-mamul op: remove the rescale of softmax weights in each inner loop, only do it in the end of inner loop
    (2) parallize in thread blocks to improve occupancy: exchange the inner loop and outerloop,  which makes each iteration in outerloop independent of each other, therefore parallelize them by assigning $\mathbf {O}$ blocks to thread blocks
    (3) distribute the work between warps to reduce shared memory communication: divide $\mathbf Q$ block to warps and keep $\mathbf {K, V}$ blocks intact, the idea is similar to exchanging outer loop and inner loop, whichi makes the $\mathbf O$ blocks the warp responsible for be independent of each other, thus primarily reducing the shared memory reads/writes for the final accumlation
    FlashAttention-2 also uses thread blocks to load KV cache in parallel for iterative decoding

\[Book\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH4.6.1
    CH4.6.1-Conditional Random Fields: CRF models conditional distribution by partially directely graph, whose advantage lies in its more flexibility. CRF allows us to use Markov network's factor decomposition semantics to represent conditoinal distribution. The specification of factors has lots of flexibility compared to explicitly specifying CPD in conditional Bayesian networks. But this flexibility in turn restrict explanability, because the parameters learned has less semantics on their own.
- [[面向计算机科学的组合数学]]: CH4-CH4.4.1
    Make general term the coefficient in generating function to relating generating function with recurrence relation, and then turn recurrence formula into a equation about generating function, thus solve the generating function, then derive the general term of the recurrence.

\[Doc\]
- [[Learn the Basics|pytorch-tutorials-beginner: Learn the Basics]]
- [[pillow v11.0.0]]: Overview, Tutorial, Concepts
- [[Repositories|huggingface-hub:Repositories]]: Sec1-Sec4
- [[Getting Started|Triton: Tutorials]]: Matrix Multiply
- [[Argparse Tutorial|argparse tutorial]]
- [[CUDA C++ Programming Guide v12.6]]: CH1

## Week2
\[Paper\]
- [[Efficient Memory Management for Large Language Model Serving with PagedAttention-2023-SOSP|2023-SOSP-Efficient Memory Management for Large Language Model Serving with PagedAttention]]

\[Book\]
- [[Probabilistic Graphical Models-Principles and Techniques]]: CH7.1-CH7.2
    CH7.1-Multivariate Gaussians: 
        Two parameterization: Standard, Information Matrix
        Marginal and Conditional density of Joint Gaussian is Gaussian, and the Conditional density is also linear Gaussian model
        Zero in $\Sigma$ implies linear independence in Joint Gaussian, which in turn implies statistical independence
        Zero in $J$ implies conditional independence
    CH7.2-Gaussian Bayesian Networks:
        All CPDs being linear Gaussian model implies the joint density is Gaussian
- [[A Tour of C++]] : CH7-CH8


\[Doc\]
- [[Annotations Best Practices]]
    Best Practice after Python 3.10: use `inspect.get_annotations()` to get any object's annotation
- [[Repositories|huggingface-hub:Repositories]]: Sec4-Sec10
- [[CUDA C++ Programming Guide v12.6]]: CH1