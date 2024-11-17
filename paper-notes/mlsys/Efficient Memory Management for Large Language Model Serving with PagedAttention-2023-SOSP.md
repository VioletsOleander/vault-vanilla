# Abstract 
High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. 
> LLM 的高吞吐服务要求同时批量处理多个请求
> 现有系统的问题在于每个请求的 KV cache 占用很大的内存，并且会动态地增长和缩小，如果管理不当，KV cache 相关的内存会因碎片化和冗余复制而被大量浪费，故限制了能批量处理的请求数量

To address this problem, we propose Paged Attention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by $2–4\times$ with the same level of latency compared to the state-of-the-art systems, such as Faster Transformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms. vLLM’s source code is publicly available at https://github.com/vllm-project/vllm . 
> 我们提出 Paged Attention，该算法灵感来源于 OS 中经典的虚拟内存和分页技术
> 基于 Paged Attention，我们构建 LLM 服务系统 vLLM，它实现了
> (1) KV cache 内存几乎零浪费
> (2) 在请求内部和请求之间灵活共享 KV cache，进一步减少内存使用
> 评估表明和 SOTA 的系统相比，在相同的延迟水平下，vLLM 将流行 LLM 的吞吐量提高了 2-4 倍，且模型越大、序列越长、解码算法越复杂，提升越明显

# 1 Introduction 
The emergence of large language models ( LLMs ) like GPT [5 , 37] and PaLM [9] have enabled new applications such as programming assistants [6 , 18] and universal chatbots [19 , 35] that are starting to profoundly impact our work and daily routines. Many cloud companies [34 , 44] are racing to provide these applications as hosted services. However, running these applications is very expensive, requiring a large number of hardware accelerators such as GPUs. According to recent estimates, processing an LLM request can be $10\times$ more expensive than a traditional keyword query [43]. Given these high costs, increasing the throughput—and hence reducing the cost per request—of *LLM serving systems* is becoming more important.  

At the core of LLMs lies an auto regressive Transformer model [53]. This model generates words (tokens), one at a time , based on the input (prompt) and the previous sequence of the output’s tokens it has generated so far. For each request, this expensive process is repeated until the model outputs a termination token. This sequential generation process makes the workload memory-bound , under utilizing the computation power of GPUs and limiting the serving throughput. 
> LLM 在输出时需要基于输入 (prompt) 和之前的输出 token 序列自回归生成 tokens 直到 termination token
> 该序列生成过程是 memory-bound，未能完全利用计算资源，因此降低了服务吞吐

Improving the throughput is possible by batching multiple requests together. However, to process many requests in a batch, the memory space for each request should be efficiently managed. For example, Fig. 1 (left) illustrates the memory distribution for a 13B-parameter LLM on an NVIDIA A100 GPU with 40GB RAM. Approximately $65\%$ of the memory is allocated for the model weights, which remain static during serving. Close to $30\%$ of the memory is used to store the dynamic states of the requests. For Transformers, these states consist of the key and value tensors associated with the attention mechanism, commonly referred to as KV cache [41], which represent the context from earlier tokens to generate new output tokens in sequence. The remaining small percentage of memory is used for other data, including activations – the ephemeral tensors created when evaluating the LLM. Since the model weights are constant and the activations only occupy a small fraction of the GPU memory, the way the KV cache is managed is critical in determining the maximum batch size. When managed inefficiently, the KV cache memory can significantly limit the batch size and consequently the throughput of the LLM, as illustrated in Fig. 1 (right). 
> 提高吞吐的一个方式是批量处理多个请求，这要求我们高效管理批量内各个请求的内存空间
> Figure 1 left 展示了 A100 上 13B 参数 LLM 的内存分配，约65%的内存用于模型权重，这部分内存在 LLM 服务时保持固定，越30%的内存用于存储请求的动态状态，也就是 KV cache，它们表示之前 tokens 用于生成新输出 token 的上下文，最后一小部分内存用于其他数据，包括激活 (评估 LLM 时创建的暂时的张量)
> 参数的内存固定，激活占用的内存小，故 KV cache 管理的方式将决定我们可以获得的最大批量大小
> KV cache 管理不当，batch size 将被明显限制，故而限制 LLM 的吞吐量

![[vLLM-Fig1.png]]

In this paper, we observe that existing LLM serving systems [31 , 60] fall short of managing the KV cache memory efficiently. This is mainly because they store the KV cache of a request in contiguous memory space, as most deep learning frameworks [33 , 39] require tensors to be stored in contiguous memory. However, unlike the tensors in the traditional deep learning workloads, the KV cache has unique characteristics: it dynamically grows and shrinks over time as the model generates new tokens, and its lifetime and length are not known a priori. These characteristics make the existing systems’ approach significantly inefficient in two ways: 
> 现存的 LLM 服务系统将一个请求的 KV cache 存储在连续的内存空间，因为大多数 DL 框架要求 tensor 的内存连续
> 但 KV cache 和传统 DL 工作负载中的 tensor 不同的是：随着模型生成新的 tokens，KV cache 会随着时间动态地增长和缩小，并且其声明周期和长度事先是未知的
> 该特性使得现存系统在以下两方面显著低效：

First, the existing systems [31 , 60] suffer from internal and external memory fragmentation. To store the KV cache of a request in contiguous space, they pre-allocate a contiguous chunk of memory with the request’s maximum length (e.g., 2048 tokens). This can result in severe internal fragmentation, since the request’s actual length can be much shorter than its maximum length (e.g., Fig. 11). Moreover, even if the actual length is known a priori, the pre-allocation is still inefficient: As the entire chunk is reserved during the request’s lifetime, other shorter requests cannot utilize any part of the chunk that is currently unused. Besides, external memory fragmentation can also be significant, since the preallocated size can be different for each request. Indeed, our profiling results in Fig. 2 show that only $20.4\%-38.2\%$ of the KV cache memory is used to store the actual token states in the existing systems. 
> 首先：现存的系统存在内部和外部内存碎片的问题
> 为了将一个请求的 KV cache 存储在连续的空间，现存系统会预分配一个连续的内存块，大小为请求的最大长度 (例如 2048 tokens)，这会导致内部碎片，因为请求的实际长度可能比最大长度短很多
> 并且即便实际长度预先知道，预分配也是低效的，因为整个内存块在请求的声明周期一直被预留，其他更短的请求无法利用该块中当前没有使用的部分
> 外部碎片的问题同样存在，因为每个请求预分配的大小可能各不相同
> Figure 2 展示了现存系统 KV cache 使用的内存中仅有 20.4%-38.2% 实际用于存储 token 状态

![[vLLM-Fig2.png]]

Second, the existing systems cannot exploit the opportunities for memory sharing. LLM services often use advanced decoding algorithms, such as parallel sampling and beam search, that generate multiple outputs per request. In these scenarios, the request consists of multiple sequences that can partially share their KV cache. However, memory sharing is not possible in the existing systems because the KV cache of the sequences is stored in separate contiguous spaces. 
> 其次：现存系统无法利用内存共享的机会
> LLM 服务经常使用高级的解码算法，例如并行采样和束搜索，这些算法对于每次请求会生成多个输出，这种情况下，由多个序列构成的请求可以部分地共享它们的 KV cache
> 现存系统无法实现内存共享，因为各个序列的 KV cache 都存储在分离的连续空间

To address the above limitations, we propose PagedAttention , an attention algorithm inspired by the operating system’s (OS) solution to memory fragmentation and sharing: virtual memory with paging . Paged Attention divides the request’s KV cache into blocks, each of which can contain the attention keys and values of a fixed number of tokens. In Paged Attention, the blocks for the KV cache are not necessarily stored in contiguous space. Therefore, we can manage the KV cache in a more flexible way as in OS’s virtual memory: one can think of blocks as pages, tokens as bytes, and requests as processes. This design alleviates internal fragmentation by using relatively small blocks and allocating them on demand. Moreover, it eliminates external fragmentation as all blocks have the same size. Finally, it enables memory sharing at the granularity of a block, across the different sequences associated with the same request or even across the different requests. 
> PagedAttention 启发自 OS 对于内存碎片和共享的解决方案：分页式虚拟内存
> PagedAttention 将请求的 KV cahe 分块，每个块包含固定数量 token 对应的keys 和 values，这些块不必要存储在连续的空间，因此用类似 OS 管理虚拟内存的方式管理 KV cache：将 KV cache 块视作 page，将 token 视作 byte，将请求视作 process
> 通过使用较小的块，并且按需分配它们，就可以减少内部碎片，并且因为所有的块都有相同大小，它也消除了外部碎片
> 该方法还使得我们可以在块的粒度上，在相同请求的不同序列之间甚至不同请求的不同序列之间进行内存共享

In this work, we build vLLM , a high-throughput distributed LLM serving engine on top of Paged Attention that achieves near-zero waste in KV cache memory. vLLM uses block-level memory management and preemptive request scheduling that are co-designed with Paged Attention. vLLM supports popular LLMs such as GPT [5], OPT [62], and LLaMA [52] with varying sizes, including the ones exceeding the memory capacity of a single GPU. Our evaluations on various models and workloads show that vLLM improves the LLM serving throughput by $2–4\times$ compared to the state-of-the-art systems [31 , 60], without affecting the model accuracy at all. The improvements are more pronounced with longer sequences, larger models, and more complex decoding algorithms (§4.3). In summary, we make the following contributions: 
> 我们基于 PagedAttention 机制构建分布式高吞吐 LLM 服务引擎 vLLM
> vLLM 对于 KV cache 内存几乎零浪费
> vLLM 还使用了与 PagedAttention 共同设计的块级内存管理和抢占式请求调度
> vLLM 对于超过单个 GPU 内存容量的模型也提供支持
> vLLM 在多个模型和工作负载下相较于 SOTA 系统提高了 2-4x 的吞吐量，且不影响模型精度，改进在更长的序列、更大的模型和更复杂的解码算法中更为明显
> 贡献总结为以下几点：

- We identify the challenges in memory allocation in serving LLMs and quantify their impact on serving performance.
- We propose Paged Attention, an attention algorithm that operates on KV cache stored in non-contiguous paged memory, which is inspired by the virtual memory and paging in OS.
- We design and implement vLLM, a distributed LLM serving engine built on top of Paged Attention.
- We evaluate vLLM on various scenarios and demonstrate that it substantially outperforms the previous state-of-theart solutions such as Faster Transformer [31] and Orca [60]. 

>- 我们识别了 LLM 服务中的内存分配挑战，并量化了它们对服务性能的影响
>- 我们提出了 Paged Attention，这是一种注意力算法，它在存储在非连续分页内存的 KV cache 上进行运算，其灵感来源于操作系统中的虚拟内存和分页技术
>- 我们设计并实现了 vLLM，这是一种基于 Paged Attention 构建的分布式 LLM 服务引擎
>- 我们在各种场景中评估了 vLLM，证明其显著优于先前的最先进解决方案，如 Faster Transformer [31] 和 Orca [60]

# 2 Background 
In this section, we describe the generation and serving procedures of typical LLMs and the iteration-level scheduling used in LLM serving. 

## 2.1 Transformer-Based Large Language Models 
The task of language modeling is to model the probability of a list of tokens $\left(x_{1},\ldots,x_{n}\right)$ . Since language has a natural sequential ordering, it is common to factorize the joint probability over the whole sequence as the product of conditional probabilities (a.k.a. auto regressive decomposition [3]): 
> 语言是自然有序的，因此将整个序列上的联合分布分解为条件分布的乘积是合理的，即自回归分解：

$$
P(x)=P(x_{1})\cdot P(x_{2}\mid x_{1})\cdot\cdot\cdot P(x_{n}\mid x_{1},.\,.\,,x_{n-1}).\tag{1}
$$ 
Transformers [53] have become the de facto standard architecture for modeling the probability above at a large scale. The most important component of a Transformer-based language model is its self-attention layers. For an input hidden state sequence $\left(x_{1},\dots\right.,x_{n})\,\in\,\mathbb{R}^{n\times d}$ , a self-attention layer first applies linear transformations on each position 𝑖 to get the query, key, and value vectors: 

$$
q_{i}=W_{q}x_{i},\;k_{i}=W_{k}x_{i},\;v_{i}=W_{v}x_{i}.\tag{2}
$$ 
Then, the self-attention layer computes the attention score $a_{i j}$ by multiplying the query vector at one position with all the key vectors before it and compute the output $o_{i}$ as the weighted average over the value vectors: 

$$
a_{i j}=\frac{\exp({q_{i}^{\top}k_{j}}/{\sqrt{d}})}{\sum_{t=1}^{i}\exp({q_{i}^{\top}k_{t}}/{\sqrt{d}})},\ o_{i}=\sum_{j=1}^{i}a_{i j}v_{j}.\tag{3}
$$ 
Besides the computation in Eq. 3, all other components in the Transformer model, including the embedding layer, feed-forward layer, layer normalization [2], residual connection [22], output logit computation, and the query, key, and value transformation in Eq. 2, are all applied independently position-wise in a form of $y_{i}=f(x_{i})$ . 
> 除了 eq 3 的 casual attention 计算，Transformer 中所有其他的计算，包括 embedding layer、FFP、layer normalization、残差连接、output logit computation 都是 position-wise 独立计算，也就是 per token 计算

## 2.2 LLM Service & Auto regressive Generation 
Once trained, LLMs are often deployed as a conditional generation service (e.g., completion API [34] or chatbot [19 , 35]). A request to an LLM service provides a list of input prompt tokens $\left(x_{1},\dots,x_{n}\right)$ , and the LLM service generates a list of output tokens $\left(x_{n+1},.\,.\,.\,,x_{n+T}\right)$ according to Eq. 1. We refer to the concatenation of the prompt and output lists as sequence . 
> LLM 在部署后处理的任务是条件生成任务
> 对于 LLM 服务的请求会提供输入 prompt token 序列 $(x_1, \dots, x_n)$ ，LLM 根据 eq 1 生成输出 token 序列 $(x_{n+1}, \dots, x_{n+T})$

Due to the decomposition in Eq. 1, the LLM can only sample and generate new tokens one by one, and the generation process of each new token depends on all the previous tokens in that sequence, specifically their key and value vectors. In this sequential generation process, the key and value vectors of existing tokens are often cached for generating future tokens, known as KV cache . Note that the KV cache of one token depends on all its previous tokens. This means that the KV cache of the same token appearing at different positions in a sequence will be different. 
> 根据 eq 1 的分解，容易知道 LLM 一次仅能采样并且生成一个 token，新 token 的生成依赖于序列中前面全部的 tokens，具体地说就是它们的 keys 和 values
> 故在序列生成过程中，前面 tokens 的 keys 和 values 可以缓存，即 KV cache
> 注意一个 tokens 的 keys 和 values (KV cache) 依赖于它之前的所有 tokens，因此出现在同一序列的不同位置的相同 token 的 KV cache 将不同

Given a request prompt, the generation computation in the LLM service can be decomposed into two phases: 
> 给定 request prompt，LLM 服务中的生成式计算可以被分解为以下两个阶段：

**The prompt phase** takes the whole user prompt $\left(x_{1},\ldots,x_{n}\right)$ as input and computes the probability of the first new token $P(x_{n+1}\mid x_{1},.\,.\,,x_{n})$ . During this process, also generates the key vectors $k_{1},\ldots,k_{n}$ and value vectors $v_{1},\dots,v_{n}$ . Since prompt tokens $x_{1},\ldots,x_{n}$ are all known, the computation of the prompt phase can be parallelized using matrix-matrix multiplication operations. Therefore, this phase can efficiently use the parallelism inherent in GPUs. 
> prompt 阶段
> 将整个用户 prompt 序列 $(x_1, \dots, x_n)$ 作为输入，计算第一个新 token 的概率 $P (x_{n+1}\mid x_1, \dots, x_n)$
> 该过程中会为 prompt tokens $(x_1, \dots, x_n)$ 生成 key 向量 $k_1, \dots, k_n$ 和 value 向量 $v_1, \dots, v_n$，因为 prompt tokens 全部已知，该阶段的计算可以使用矩阵-矩阵乘法算子并行 (所有的输入 tokens 都要作为 query 被编码，进行 masked self attention 计算)，故可以利用 GPU 中内在的并行特性

**The auto regressive generation phase** generates the remaining new tokens sequentially. At iteration $t$ , the model takes one token $x_{n+t}$ as input and computes the probability $P(x_{n+t+1}\mid x_{1},.\,.\,.\,,x_{n+t})$ with the key vectors $k_{1},.\,.\,.\,,k_{n+t}$  and value vectors $v_{1},.\,.\,.\,,v_{n+t}$ . Note that the key and value vectors at positions $1$ to $n+t-1$ are cached revious iterations, only the new key and value vector $k_{n+t}$ and $v_{n+t}$ are computed at this iteration. This phase completes either when the sequence reaches a maximum length (specified by users or limited by LLMs) or when an end-of-sequence $(<\!e o s\!>)$ token is emitted. The computation at different iterations cannot be parallelized due to the data dependency and often uses matrix-vector multiplication, which is less efficient. As a result, this phase severely under utilizes GPU computation and becomes memory-bound, being responsible for most portion of the latency of a single request. 
> 自回归生成阶段
> 该阶段顺序生成新 tokens
> 在迭代 $t$ 时，模型接受一个 token $x_{n+t}$ 作为输入，利用 key 向量 $k_1, \dots, k_{n+t}$ 和 value 向量 $v_1, \dots, v_{n+t}$ 计算概率 $P (x_{n+t+1}\mid x_1, \dots, x_{n+t})$
> 注意位置 $1$ 到 $n+t-1$ 的 keys 和 values 在之前的迭代已经被缓存，因此本次迭代仅需要计算新的 key 和 value 向量 $k_{n+t}, v_{n+t}$
> 该阶段在序列达到指定的最大长度或者生成了 *\<eos\>* token 后结束
> 该阶段中，不同迭代之间不能并行，因为存在顺序的数据依赖，并且该阶段一般使用相对低效的矩阵-向量乘法 (仅有新 token 作为 query 需要被编码)，因此该阶段没有充分使用 GPU 的计算资源，为 memory-bound，故该阶段的计算会占据单个 request 的大多数延迟时间

## 2.3 Batching Techniques for LLMs 
The compute utilization in serving LLMs can be improved by batching multiple requests. Because the requests share the same model weights, the overhead of moving weights is amortized across the requests in a batch, and can be overwhelmed by the computational overhead when the batch size is sufficiently large. However, batching the requests to an LLM service is non-trivial for two reasons. First, the requests may arrive at different times. A naive batching strategy would either make earlier requests wait for later ones or delay the incoming requests until earlier ones finish, leading to significant queueing delays. Second, the requests may have vastly different input and output lengths (Fig. 11). A straightforward batching technique would pad the inputs and outputs of the requests to equalize their lengths, wasting GPU computation and memory. 
> 可以通过批量处理多个 request 提高 LLM 服务中的计算资源利用，因为多个 request 共享模型权重，故移动权重的开销会在 batch 中的 requests 之间摊销，并且如果 batch size 足够大，移动权重的开销在足够的的计算开销下就不显得重要
> 在 LLM 服务中批处理多个 request 存在两点困难：
>  1. requests 可能在不同时刻到达，朴素的批处理策略要么让较早的 requests 等待较晚的 requests，要么延迟正在传入的 requests 直到较早的 requests 处理完成，因此存在显著的排队延迟
>  2. requests 的输入和输出长度可能显著不同，直接的批处理策略将填充 requests 的输入和输出使其具有相同长度，导致浪费 GPU 计算和内存

To address this problem, fine-grained batching mechanisms, such as cellular batching [16] and iteration-level scheduling [60], have been proposed. Unlike traditional methods that work at the request level, these techniques operate at the iteration level. After each iteration, completed requests are removed from the batch, and new ones are added. Therefore, a new request can be processed after waiting for a single iteration, not waiting for the entire batch to complete. Moreover, with special GPU kernels, these techniques eliminate the need to pad the inputs and outputs. By reducing the queueing delay and the inefficiencies from padding, the fine-grained batching mechanisms significantly increase the throughput of LLM serving. 
>为了解决这些问题，细粒度的批处理机制，例如细胞批处理和迭代级调度被提出
>传统方法工作在 request 级别，而这类方法工作在 iteration 级别，在每个 iteration，完成的 request 将从批量中被移除，新的 request 会被加入，因此新的 request 可以在等待单个 iteration 之后就被处理，而不是等待整个批量完成
>此外，通过使用特殊的GPU kernel，这些技术消除了填充输入和输出的需求
>细粒度的批处理机制通过减少排队延迟和填充所带来的低效率，显著提高了LLM服务的吞吐量

# 3 Memory Challenges in LLM Serving 
Although fine-grained batching reduces the waste of computing and enables requests to be batched in a more flexible way, the number of requests that can be batched together is still constrained by GPU memory capacity, particularly the space allocated to store the KV cache. In other words, the serving system’s throughput is memory-bound . Overcoming this memory-bound requires addressing the following challenges in the memory management: 
> 细粒度的批处理减少了计算浪费 (消除了填充)，并且使得 requests 可以更灵活地批处理，但可以批处理的 requests 数量仍然受限于 GPU 显存容量，尤其是分配于存储 KV cache 的那部分显存空间大小
> 换句话说，LLM 服务系统的吞吐量是 memory-bound，要克服它，需要我们解决以下显存管理的挑战：

**Large KV cache.** The KV Cache size grows quickly with the number of requests. As an example, for the 13B parameter OPT model [62], the KV cache of a single token demands 800 KB of space, calculated as 2 (key and value vectors) $\times5120$ (hidden state size) $\times\ 40$ (number of layers) $\times\;2$ (bytes per FP16). Since OPT can generate sequences up to 2048 tokens, the memory required to store the KV cache of one request can be as much as 1.6 GB. Concurrent GPUs have memory capacities in the tens of GBs. Even if all available memory was allocated to KV cache, only a few tens of requests could be accommodated. Moreover, inefficient memory management can further decrease the batch size, as shown in Fig. 2. Additionally, given the current trends, the GPU’s computation speed grows faster than the memory capacity [17]. For example, from NVIDIA A100 to H100, The FLOPS increases by more than $2\mathrm{x}$ , but the GPU memory stays at 80GB maximum. Therefore, we believe the memory will become an increasingly significant bottleneck. 
> Large KV cache
> KV cache 的大小会随着 requests 数量快速增长
> 例如，13B OPT 模型中单个 token 的 FP16 KV cache 需要 800KB 的空间 
> ($2\times 5120\times 40 \times 2\  \text{bytes} = 800\ \text{KB}$)，OPT 生成序列长度上限是 2048 tokens，故单个 request 所需的 KV cache 空间可以达到 1.6 GB
> 当前的 GPU 设备显存容量在几十 GB 的规模，因此即便全部显存分配给 KV cache，也仅能存下十几个 requests 的 KV cache
> 并且，低效的内存管理会进一步减少 batch size
> 当前的发展趋势是 GPU 的计算速度增长快于显存容量，例如 A100 到 H100 FLOPs 增长1倍，而显存保持 80GB 最大不变
> 因此，显存将逐渐成为越加显著的瓶颈

**Complex decoding algorithms.** LLM services offer a range of decoding algorithms for users to select from, each with varying implications for memory management complexity. For example, when users request multiple random samples from a single input prompt, a typical use case in program suggestion [18], the KV cache of the prompt part, which accounts for $12\%$ of the total KV cache memory in our experiment (§6.3), can be shared to minimize memory usage. On the other hand, the KV cache during the auto regressive generation phase should remain unshared due to the different sample results and their dependence on context and position. The extent of KV cache sharing depends on the specific decoding algorithm employed. In more sophisticated algorithms like beam search [49], different request beams can share larger portions (up to $55\%$ memory saving, see $\S6.3)$ of their KV cache, and the sharing pattern evolves as the decoding process advances. 
> 复杂解码算法
> LLM 服务提供了一系列解码算法供用户选择，这些算法各自都对内存管理的复杂有不同的影响
> 例如，当用户从单个输入 prompt 请求多个随机样本时，prompt 部分的 KV cache (在我们的实验中占总 KV cache 内存的 $12\%$ )，可以被共享以最小化内存使用，当然自回归生成阶段的 KV cache 仍然是不共享的，因为每个采样的生成结果不同
> KV cache 的共享程度取决于采用的特定解码算法，在更为复杂的算法例如 beam search 中，不同的 request beam 可以共享更大部分的 KV cache (因此节约最多 55% 的内存)，并且随着解码过程的推进，它们的共享模式也会发生变化

**Scheduling for unknown input & output lengths.** The requests to an LLM service exhibit variability in their input and output lengths. This requires the memory management system to accommodate a wide range of prompt lengths. In addition, as the output length of a request grows at decoding, the memory required for its KV cache also expands and may exhaust available memory for incoming requests or ongoing generation for existing prompts. The system needs to make scheduling decisions, such as deleting or swapping out the KV cache of some requests from GPU memory. 
> 对未知输入、输出长度的调度
> 对 LLM 服务的 requests 的输入和输出长度一般都是不同的，这要求内存管理系统能够适应各种长度的 prompt
> 另外，随着 request 的输出长度在解码中增长，其 KV cache 所需的内存也将增长，进而消耗掉为新的 request 或者为现存 prompt 的生成过程所准备的内存
> 因此，系统需要进行调度决策，例如从 GPU 显存中删去或者换出一些 requests 的 KV cache

## 3.1 Memory Management in Existing Systems
Since most operators in current deep learning frameworks [33 , 39] require tensors to be stored in contiguous memory, previous LLM serving systems [31 , 60] also store the KV cache of one request as a contiguous tensor across the different positions. Due to the unpredictable output lengths from the LLM, they statically allocate a chunk of memory for a request based on the request’s maximum possible sequence length, irrespective of the actual input or eventual output length of the request. 
> 当前的 DL 框架的大多数算子要求 tensor 存储在连续内存中，故之前的 LLM 服务系统将一个 request 的 KV cache 也作为连续的 tensor 存储
> 因为 request 的输出长度不同，故这些系统基于 request 的最大可能序列长度为 request 静态地分配一个内存块，不关心 request 的实际输入和最终输出长度

Fig. 3 illustrates two requests: request A with 2048 maximum possible sequence length and request B with a maximum of 512. The chunk pre-allocation scheme in existing systems has three primary sources of memory wastes: reserved slots for future tokens, internal fragmentation due to over-provisioning for potential maximum sequence lengths, and external fragmentation from the memory allocator like the buddy allocator. The external fragmentation will never be used for generated tokens, which is known before serving a request. Internal fragmentation also remains unused, but this is only realized after a request has finished sampling. They are both pure memory waste. Although the reserved memory is eventually used, reserving this space for the entire request’s duration, especially when the reserved space is large, occupies the space that could otherwise be used to process other requests. We visualize the average percentage of memory wastes in our experiments in Fig. 2, revealing that the actual effective memory in previous systems can be as low as $20.4\%$ . 
> 如 Figure 3 所示，request A 的最大可能序列长度为 2048，request B 的最大可能序列长度是 512，现存系统的内存块预分配策略存在三种主要的内存浪费：为未来的 token 预留的 slot、为最大序列长度过度分配的内存导致的内部碎片、来自内存分配器 (例如 buddy 分配器) 的外部碎片
> 其中，外部碎片在服务 request 之前就已知不会被使用，内部碎片仅在 request 完成采样之后才能确定不会被使用，二者都是完全的内存浪费
> 而为未来 token 预留的内存虽然最终会被使用，但该空间也会在 request 的整个持续周期被预留，当预留的空间较大时，这也会占用本可以用于处理其他 request 的空间
> Figure 2 展示了之前系统的实际有效内存使用率可能低至 20.4%

![[vLLM-Fig3.png]]

Although compaction [54] has been proposed as a potential solution to fragmentation, performing compaction in a performance-sensitive LLM serving system is impractical due to the massive KV cache. Even with compaction, the pre-allocated chunk space for each request prevents memory sharing specific to decoding algorithms in existing memory management systems. 
> 一个解决碎片的方法是 compaction，但在性能敏感的 LLM 服务系统中执行 compaction 是不现实的，因为其 KV cache 十分庞大
> 且即便使用 compaction，为每个 request 预留块空间的方法也不能实现 request 在特定的解码算法下共享内存

# 4 Method 
In this work, we develop a new attention algorithm, PagedAttention , and build an LLM serving engine, vLLM , to tackle the challenges outlined in $\S3$ . The architecture of vLLM is shown in Fig. 4. vLLM adopts a centralized scheduler to coordinate the execution of distributed GPU workers. The KV cache manager effectively manages the KV cache in a paged fashion, enabled by Paged Attention. Specifically, the KV cache manager manages the physical KV cache memory on the GPU workers through the instructions sent by the centralized scheduler. 
> 我们提出新的 attention 算法 PagedAttention，并基于此构建 LLM 服务引擎 vLLM，vLLM 框架如 Figure 4 所示
> Figure 4 中，中心化的调度器来协调分布式 GPU workers 的执行，KV cache 管理器通过中心化的调度器发送指令来管理 GPU workers 中的物理 KV cache 内存

![[vLLM-Fig4.png]]

Next, We describe the Paged Attention algorithm in $\S4.1$ . With that, we show the design of the KV cache manager in $\S4.2$ and how it facilitates Paged Attention in $\S4.3$ , respectively. Then, we show how this design facilitates effective memory management for various decoding methods (§4.4) and handles the variable length input and output sequences (§4.5). Finally, we show how the system design of vLLM works in a distributed setting (§4.6). 

## 4.1 Paged Attention 
To address the memory challenges in $\S3$ , we introduce PagedAttention , an attention algorithm inspired by the classic idea of paging [25] in operating systems. Unlike the traditional attention algorithms, Paged Attention allows storing continuous keys and values in non-contiguous memory space. Specifically, Paged Attention partitions the KV cache of each sequence into KV blocks . Each block contains the key and value vectors for a fixed number of tokens, which we denote as $K V$ block size ( $B$ ). Denote the key block $K_{j}=(k_{(j-1)B+1},\dots,k_{j B})$ and value block $V_{j}=(v_{(j-1)B+1},\dots,v_{j B})$ . The attention computation in Eq. 4 can be transformed into the following blockwise computation: 

$$
A_{i j}=\frac{\exp(q_{i}^{\top}K_{j}/\sqrt{d})}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_{i}^{\top}K_{t}\mathbf 1/\sqrt{d})},\;o_{i}=\sum_{j=1}^{\lceil i/B\rceil}V_{j}A_{i j}^{\top},\tag{4}
$$

where $A_{i j}=\left(a_{i,(j-1)B+1},\dots,a_{i,j B}\right)$ is the row vector of attention score on $j$ -th KV block. 

> PagedAttention 允许将连续的 keys 和 values 存储在非连续的内存空间
> 具体地说，PagedAttention 将每个序列的 KV cache 划分为 KV  blocks，包括 key blocks 和 value blocks，每个 key/value block 包含序列中 block size ( $B$ ) 个 tokens 对应的 keys 和 values，分别记作 $K_j = (k_{(j-1) B + 1}, \dots, k_{jB})$ 和 $V_j = (v_{(j-1)B + 1}, \dots, v_{jB})$
> PagedAttention 进而将 eq 3 的 attention 计算转化为如上的逐块的运算，其中 $A_{ij} = (a_{i, (j-1) B+1}, \dots, a_{i, jB})$ 为 $q_i$ 相对于第 $j$ 个 K block 的 attention score 向量
> (注：公式 (4) 显然存在错误，正确的公式应该将 $\mathbf 1$ 放在 $\exp$ 外，并且 $\mathbf 1$ 应该同时作为 indicator function，满足第 $\lceil i / B \rceil$ 的块中 $j > i$ 的 $k_j$ 对应的 $\exp (q_i^\top k_j/\sqrt d)$ 乘上零) 

During the attention computation, the Paged Attention kernel identifies and fetches different KV blocks separately. We show an example of Paged Attention in Fig. 5: The key and value vectors are spread across three blocks, and the three blocks are not contiguous on the physical memory. At each time, the kernel multiplies the query vector $q_{i}$ of the query token (*"forth"*) and the key vectors $K_{j}$ in a block (e.g., key vectors of *“Four score and seven”* for block 0) to compute the attention score $A_{i j}$ , and later multiplies $A_{i j}$ with the value vectors $V_{j}$ in a block to derive the final attention output $o_{i}$ . 
> 在 attention 计算过程中，PagedAttention kernel 会分别识别并获取不同的 KV blocks
> PagenAttention 的示例见 Fig5，可以看到序列 "Four score and seven years ago our fathers brought forth" 的全部 keys 和 values 向量分为三个块存储，块之间在物理内存中是不必连续的
> 在每一次计算中，query token ("forth") 的查询向量 $q_i$ 仅和一个块的 key vectors $K_j$ 相乘 (例如 block 0 中 "Four score and seven" 的 key vectors)，计算出对应的 attention score $A_{ij}$，之后在计算最终 attention 输出 $o_i$ 时，还会将 $A_{ij}$ 和块中的 value vectors $V_j$ 相乘

![[vLLM-Fig5.png]]

In summary, the Paged Attention algorithm allows the KV blocks to be stored in non-contiguous physical memory, which enables more flexible paged memory management in vLLM. 
> 总之，PagedAttention 算法允许 KV blocks 存储在非连续的物理内存中，这使得我们可以在 vLLM 对内存进行灵活的分页管理

## 4.2 KV Cache Manager 
The key idea behind vLLM’s memory manager is analogous to the virtual memory [25] in operating systems. OS partitions memory into fixed-sized pages and maps user programs’ logical pages to physical pages. Contiguous logical pages can correspond to non-contiguous physical memory pages, allowing user programs to access memory as though it were contiguous. Moreover, physical memory space needs not to be fully reserved in advance, enabling the OS to dynamically allocate physical pages as needed. vLLM uses the ideas behind virtual memory to manage the KV cache in an LLM service. Enabled by Paged Attention, we organize the KV cache as fixed-size KV blocks, like pages in virtual memory. 
> vLLM 进行内存管理的核心思想类似于 OS 的虚拟内存
> OS 将内存划分为固定大小的页，然后将用户程序的逻辑页映射到物理页，连续的逻辑页可以对应于不连续的物理页，而用户程序可以将内存当作连续的来访问
> 另外，物理内存空间并不需要完全预先预留，故 SO 可以按照需要动态地分配物理页
> vLLM 利用了虚拟内存的这种思想来管理 LLM 服务中的 KV cache，通过 PagedAttention 算法，我们将 KV cache 划分为固定大小的 KV blocks 来管理，类似于虚拟内存中的页

A request’s KV cache is represented as a series of logical KV blocks , filled from left to right as new tokens and their KV cache are generated. The last KV block’s unfilled positions are reserved for future generations. On GPU workers, a block engine allocates a contiguous chunk of GPU DRAM and divides it into physical KV blocks (this is also done on CPU RAM for swapping; see $\S4.5)$ . The KV block manager also maintains block tables —the mapping between logical and physical KV blocks of each request. Each block table entry records the corresponding physical blocks of a logical block and the number of filled positions. Separating logical and physical KV blocks allows vLLM to dynamically grow the KV cache memory without reserving it for all positions in advance, which eliminates most memory waste in existing systems, as in Fig. 2. 
> 一个请求的 KV cache 被表示为一系列逻辑 KV blocks，随着新的 tokens 和它们的 KV cache 被生成，KV blocks 也会从左到右被填充，最后一个 KV block 中未填充的位置为未来的生成预留
> 在 GPU worker 上，block engine 负责分配连续的 GPU DRAM 块，然后将该 DRAM 块划分为多个物理 KV 块 (在 swapping 时，CPU RAM 也会进行这样的划分)
> KV block manager 同时维护 block 表，block 表的每个表项记录了每个 request 的逻辑 KV 块到物理 KV 块之间的映射，以及块中已经被填充的位置的数量
> 通过分离逻辑 KV 块和物理 KV 块，vLLM 得以在不提前为所有的位置预留内存的情况下动态增长 KV cache 占用的内存，这消除了现存系统中大多数的内存浪费

## 4.3 Decoding with Paged Attention and vLLM 
Next, we walk through an example, as in Fig. 6, to demonstrate how vLLM executes Paged Attention and manages the memory during the decoding process of a single input sequence: 
1. As in OS’s virtual memory, vLLM does not require reserving the memory for the maximum possible generated sequence length initially. Instead, it reserves only the necessary KV blocks to accommodate the KV cache generated during prompt computation. In this case, The prompt has 7 tokens, so vLLM maps the first 2 logical KV blocks (0 and 1) to 2 physical KV blocks (7 and 1, respectively). In the prefill step, vLLM generates the KV cache of the prompts and the first output token with a conventional self-attention algorithm (e.g., [13]). vLLM then stores the KV cache of the first 4 tokens in logical block 0 and the following 3 tokens in logical block 1. The remaining slot is reserved for the subsequent auto regressive generation phase. 
2. In the first auto regressive decoding step, vLLM generates the new token with the Paged Attention algorithm on physical blocks 7 and 1. Since one slot remains available in the last logical block, the newly generated KV cache is stored there, and the block table’s \#filled record is updated. 
3. At the second decoding step, as the last logical block is full, vLLM stores the newly generated KV cache in a new logical block; vLLM allocates a new physical block (physical block 3) for it and stores this mapping in the block table.

> 本节展示一个例子，描述 vLLM 如何执行 PagedAttention 并在为单个输入序列解码时管理内存，图见 Fig6
> 1. vLLM 并不会在最初请求预留出容纳最大可能生成的序列长度的 KV cache 对应的内存空间，而是仅预留必要的 KV blocks 以容纳 prompt 计算时生成的 KV cache，例如在本例中，prompt 有 7 个 tokens，vLLM 故仅将前两个逻辑 KV 块 (block 0, 1) 映射到物理 KV 块 (block 7, 1)
>    在 prefill 步骤中，vLLM 使用常规的 self-attention 算法为 prompt 和第一个 output token 生成 KV cache，然后 vLLM 将前四个 tokens 的 keys 和 values 存储在逻辑块0，将之后三个 tokens 的 keys 和 values 存储在逻辑块1，剩下的 slot 为后面自动回归生成阶段的 token 的 keys 和 values 预留
> 2. 在第一个自回归解码步骤中，vLLM 使用 PagedAttention 算法根据物理块 7 和 1 中的 KV cache 生成新 token，生成新 token 时，因为上一个逻辑块中还有空的 slot，故新生成的 KV cache 会先填充到该 slot 中，并且更新 block table 中该逻辑块的 `#filled` 数量条目
> 3. 在第二个自回归解码步骤中，因为上一个逻辑块已经装满，vLLM 将新生成的 KV cache 存储在新的逻辑块中，并且为该逻辑块分配新的物理块，在 block table 中存储该映射关系
>    (自回归解码使用 FlashAttention 没有意义，因为需要编码的序列长度永远满足 $N=1$，当然 FlashAttention2 提出在自回归解码时多线程并行 load KV cache，该思路可以和 PagedAttention 结合，也就是多线程并行 load 物理 KV cache 块，同时 tiling 的思想仍然可以在 PagedAttention 的实现中应用)

![[vLLM-Fig6.png]]

Globally, for each decoding iteration, vLLM first selects a set of candidate sequences for batching (more in $\S4.5$ ), and allocates the physical blocks for the newly required logical blocks. Then, vLLM concatenates all the input tokens of the current iteration (i.e., all tokens for prompt phase requests and the latest tokens for generation phase requests) as one sequence and feeds it into the LLM. During LLM’s computation, vLLM uses the Paged Attention kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks. Storing multiple tokens within a KV block (block size $>1$ ) enables the Paged Attention kernel to process the KV cache across more positions in parallel, thus increasing the hardware utilization and reducing latency. However, a larger block size also increases memory fragmentation. We study the effect of block size in $\S7.2$ . 
> 全局上，在每一次解码迭代中，vLLM 首先选择一组候选序列进行批处理，并且为新需要的逻辑块分配物理块；然后，vLLM 将当前迭代的所有输入 tokens (对于 prompt 阶段的 requests 就是所有的 tokens，对于 generation 阶段的 requests 就是最后的一个 token) 串联成一个序列，并将其输入到 LLM
> 在 LLM 进行计算时，vLLM 使用 PagedAttention kernel 访问之前的 KV cache (以逻辑 KV 块的形式存储)，并且将新生成的 KV cache 储存到物理 KV 块
> 一个 KV 块存储多个 tokens 的 KV cache (即 block size > 1) 使得 PagedAttention 可以并行处理多个位置的 KV cache，进而提高了硬件利用率并降低了延迟，但更大的 block size 也会提高内存碎片

Again, vLLM dynamically assigns new physical blocks to logical blocks as more tokens and their KV cache are generated. As all the blocks are filled from left to right and a new physical block is only allocated when all previous blocks are full, vLLM limits all the memory wastes for a request within one block, so it can effectively utilize all the memory, as shown in Fig. 2. This allows more requests to fit into memory for batching—hence improving the throughput. Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests. 
> vLLM 随着更多的 tokens 和它们的 KV cache 被生成的时候，动态地将新的物理块分配给逻辑块
> 因为所有的块都会从左到右进行填充，并且 vLLM 仅在所有之前的逻辑块都填满时才分配新的物理块，故 vLLM 将一个 request 的内存浪费大小限制在了块大小以内，故可以高效利用内存，这也允许更多的 requests 可以放入内存进行批处理，故进而提高了吞吐量
> 一旦一个 request 完成了其生成，它的 KV 块就可以被释放，以存储其他 requests 的 KV cache

In Fig. 7, we show an example of vLLM managing the memory for two sequences. The logical blocks of the two sequences are mapped to different physical blocks within the space reserved by the block engine in GPU workers. The neighboring logical blocks of both sequences do not need to be contiguous in physical GPU memory and the space of physical blocks can be effectively utilized by both sequences. 
> Figure 7 展示了 vLLM 管理两个序列的内存的示例
> 两个序列的逻辑块各自被映射到不同的物理块 (物理块的空间由 GPU worker 上的 block engine 预留)，其中相邻的逻辑块并不要求其物理块连续
> 可以看到物理块空间被两个序列同时高效利用

![[vLLM-Fig7.png]]

## 4.4 Application to Other Decoding Scenarios 
$\S4.3$ shows how Paged Attention and vLLM handle basic decoding algorithms, such as greedy decoding and sampling, that take one user prompt as input and generate a single output sequence. In many successful LLM applications [18 , 34], an LLM service must offer more complex decoding scenarios that exhibit complex accessing patterns and more opportunities for memory sharing. We show the general applicability of vLLM on them in this section. 
> 上一节介绍了 vLLM 是如何处理基本的解码算法的 (例如贪心解码和采样，即接受用户 prompt 作为输入，然后生成单个输出序列)
> 本节介绍 vLLM 对于更复杂解码算法的处理，更复杂的解码算法会有更复杂的内存访问模式，同时也有更多内存共享的机会

**Parallel sampling.** In LLM-based program assistants [6 , 18], an LLM generates multiple sampled outputs for a single input prompt; users can choose a favorite output from various candidates. So far we have implicitly assumed that a request generates a single sequence. In the remainder of this paper, we assume the more general case in which a request generates multiple sequences. In parallel sampling, one request includes multiple samples sharing the same input prompt, allowing the KV cache of the prompt to be shared as well. Via its Paged Attention and paged memory management, vLLM can realize this sharing easily and save memory. 
> 并行采样
> 在基于 LLM 的程序助手中 (例如 copilot)，LLM 会对单个输入 prompt 生成多个采样的输出，用户从多个输出候选中进行选择
> 目前为止，我们都假设一个 request 生成单个序列，之后，我们都认为一个 request 生成多个序列
> 在并行采样中，单个 request 会对应多个输出序列 (共享相同的 prompt)，因此 prompt 的 KV cache 就可以被共享，vLLM 同样实现了这一点

Fig. 8 shows an example of parallel decoding for two outputs. Since both outputs share the same prompt, we only reserve space for one copy of the prompt’s state at the prompt phase; the logical blocks for the prompts of both sequences are mapped to the same physical blocks: the logical block 0 and 1 of both sequences are mapped to physical blocks 7 and 1, respectively. Since a single physical block can be mapped to multiple logical blocks, we introduce a reference count for each physical block. In this case, the reference counts for physical blocks 7 and 1 are both 2. At the generation phase, the two outputs sample different output tokens and need separate storage for KV cache. vLLM implements a copy-on-write mechanism at the block granularity for the physical blocks that need modification by multiple sequences, similar to the copy-on-write technique in OS virtual memory (e.g., when forking a process). Specifically, in Fig. 8, when sample A1 needs to write to its last logical block (logical block 1), vLLM recognizes that the reference count of the corresponding physical block (physical block 1) is greater than 1; it allocates a new physical block (physical block 3), instructs the block engine to copy the information from physical block 1, and decreases the reference count to 1. Next, when sample A2 writes to physical block 1, the reference count is already reduced to 1; thus A2 directly writes its newly generated KV cache to physical block 1. 
> 单个输入多个输出的并行解码过程示例见 Figure 8
> 可以看到，因为两个输出共享相同的 prompt，我们在 prompt 阶段仅保留一份 prompt 状态，两个序列的逻辑块被映射到相同的物理块
> 因为单个物理块可以被映射到多个逻辑块，我们为每个物理块添加引用计数，此例中，物理块 1, 7 的引用计数都是 2
> 在输出阶段，不同的输出会采样不同的 token，故这些新生成的 tokens 的 KV cache 需要分别存储
> 为此，vLLM 在块级别的粒度实现写时拷贝机制，对需要被多个序列修改的物理块进行写时拷贝，这也类似于 OS 虚拟内存管理中的写时拷贝技术 (例如 fork 进程时)
> 在 Figure8 中，当样本 A1 需要向它最新的逻辑块中**写入**时，vLLM 识别到该逻辑块对应的物理块的引用计数大于 1，因此分配一个新的物理块，并让 block engine 将原来块的信息拷贝到新块，并且将原来块的引用计数减一
> 之后，当样本 A2 需要向它的逻辑块写入时，vLLM 识别到该逻辑块对应的物理块引用计数仅为 1，即没有被复用，独属于 A2，因此 A2 直接将其新 token 的 KV cache 写入到原来的物理块中

![[vLLM-Fig8.png]]

In summary, vLLM enables the sharing of most of the space used to store the prompts’ KV cache across multiple output samples, with the exception of the final logical block, which is managed by a copy-on-write mechanism. By sharing physical blocks across multiple samples, memory usage can be greatly reduced, especially for long input prompts . 
> 总之，vLLM 使得多个输出样本之间可以共享存储 prompt 的 KV cache 的大多数空间 (只有最后一个逻辑块中的部分 prompt tokens 的 KV cache 不能共享)，这进而减少了多个输出样本情况下的内存使用量，尤其是对于长的输入 prompt

**Beam search.** In LLM tasks like machine translation [59], the users expect the top-k most appropriate translations output by the LLM. Beam search [49] is widely used to decode the most probable output sequence from an LLM, as it mitigates the computational complexity of fully traversing the sample space. The algorithm relies on the beam width parameter $k$ , which determines the number of top candidates retained at every step. During decoding, beam search expands each candidate sequence in the beam by considering all possible tokens, computes their respective probabilities using the LLM, and retains the top-k most probable sequences out of $k\cdot|V|$ candidates, where $|V|$ is the vocabulary size. 
> 束搜索
> 在像机器翻译这样的 LLM 任务中，用户期待 LLM 输出 top-k 个最合适的翻译结果
> beam search 被广泛用于从 LLM 中解码最可能的输出序列，该方法缓解了完全遍历样本空间的计算复杂性 (对于长度为 $n$ 的序列，词袋大小为 $|V|$，则完整样本空间的大小为 $|V|^n$)
> beam search 算法依赖于 beam width 参数 $k$，该参数决定了每一步需要保留的前 $k$ 个候选 token，在解码时，beam search 通过考虑所有可能的 tokens 展开 beam 中的每个候选序列，使用 LLM 计算它们各自的概率，然后从 $k\cdot |V|$ 个候选序列中保留 top-k 个最可能的序列

Unlike parallel decoding, beam search facilities sharing not only the initial prompt blocks but also other blocks across different candidates, and the sharing patterns dynamically change as the decoding process advances, similar to the process tree in the OS created by compound forks. Fig. 9 shows how vLLM manages the KV blocks for a beam search example with $k\,=\,4$ . Prior to the iteration illustrated as the dotted line, each candidate sequence has used 4 full logical blocks. All beam candidates share the first block 0 (i.e., prompt). Candidate 3 digresses from others from the second block. Candidates 0-2 share the first 3 blocks and diverge at the fourth block. At subsequent iterations, the top-4 probable candidates all originate from candidates 1 and 2. As the original candidates 0 and 3 are no longer among the top candidates, their logical blocks are freed, and the reference counts of corresponding physical blocks are reduced. vLLM frees all physical blocks whose reference counts reach 0 (blocks 2, 4, 5, 8). Then, vLLM allocates new physical blocks (blocks 9-12) to store the new KV cache from the new candidates. Now, all candidates share blocks 0, 1, 3; candidates 0 and 1 share block 6, and candidates 2 and 3 further share block 7. 
> 和并行解码不同，beam search 不仅共享初始的 prompt 块，还共享不同候选之间的 KV 块，并且共享模式随着解码过程动态改变，类似于 OS 通过复合 fork 创建的进程树
> Figure 9 展示了 $k=4$ 时的一个示例，在虚线之前的迭代中，所有的候选序列都各自使用四个完全的逻辑块，所有的候选序列在物理上共享 block 0 (prompt)，候选3从第二个块开始分离，候选0-2共享前3个块，在第四个块分离
> 在后续的迭代中，前 $k=4$ 个最可能的候选都来自于候选 1 和 2，则原来的候选 0 和 3 不再需要，则它们的逻辑块被释放，对应的物理块的引用计数减少，vLLM 会释放引用计数减少为 0 的物理块
> 然后 vLLM 分配新的物理块来存储新的候选的 KV cache，此时候选 0, 1 共享 block 6，候选 2, 3 共享 block 7，所有候选共享 block 0, 1, 3


![[vLLM-Fig9.png]]

Previous LLM serving systems require frequent memory copies of the KV cache across the beam candidates. For example, in the case shown in Fig. 9, after the dotted line, candidate 3 would need to copy a large portion of candidate 2’s KV cache to continue generation. This frequent memory copy overhead is significantly reduced by vLLM’s physical block sharing. In vLLM, most blocks of different beam candidates can be shared. The copy-on-write mechanism is applied only when the newly generated tokens are within an old shared block, as in parallel decoding. This involves only copying one block of data. 
> 之前的 LLM 服务系统需要在 beam 候选中频繁地拷贝 KV cache 的内存，例如在 Figure 9 中，候选 3 需要拷贝候选 2 的大部分 KV cache 以继续生成
> vLLM 的物理块共享显著降低了这类频繁内存拷贝的开销，在 vLLM 中，多数 beam 候选的块可以被共享，写时拷贝机制仅在新生成的 token 位于旧的共享块中才执行 (和并行解码中的情况一样)，这仅涉及拷贝一块数据

**Shared prefix.** Commonly, the LLM user provides a (long) description of the task including instructions and example inputs and outputs, also known as system prompt [36]. The description is concatenated with the actual task input to form the prompt of the request. The LLM generates outputs based  on the full prompt. Fig. 10 shows an example. Moreover, the shared prefix can be further tuned, via prompt engineering, to improve the accuracy of the downstream tasks [26, 27]. 
> 共享前缀
> 一般情况下，LLM 用户会提供对任务的描述 (包括指令、示例输入输出)，这类 prompt 也称为系统 prompt
> 该描述会和实际的任务输入进行拼接，得到 request 的完整 prompt，LLM 基于完整 prompt 生成输出
> Figure 10 展示了共享前缀的一个示例

![[vLLM-Fig10.png]]

For this type of application, many user prompts share a prefix, thus the LLM service provider can store the KV cache of the prefix in advance to reduce the redundant computation spent on the prefix. In vLLM, this can be conveniently achieved by reserving a set of physical blocks for a set of predefined shared prefixes by the LLM service provider, as how OS handles shared library across processes. A user input prompt with the shared prefix can simply map its logical blocks to the cached physical blocks (with the last block marked copy-on-write). The prompt phase computation only needs to execute on the user’s task input. 
> 对于这类应用，许多用户 prompt 会共享同一个前缀，因此 LLM 服务提供者可以将该前缀的 KV cache 提前存储，以减少重复计算
> vLLM 中，可以为预定义的前缀的 KV cache 预留一组物理块，类似于 OS 在多个进程之间处理共享库，使用共享前缀的用户输入 prompt 可以直接将其逻辑块映射到这些缓存的物理块 (最后一个块进行写时拷贝)，prompt 阶段的计算就只需要对用户的任务输入进行

**Mixed decoding methods.** The decoding methods discussed earlier exhibit diverse memory sharing and accessing patterns. Nonetheless, vLLM facilitates the simultaneous processing of requests with different decoding preferences, which existing systems cannot efficiently do. This is because vLLM conceals the complex memory sharing between different sequences via a common mapping layer that translates logical blocks to physical blocks. The LLM and its execution kernel only see a list of physical block IDs for each sequence and do not need to handle sharing patterns across sequences. Compared to existing systems, this approach broadens the batching opportunities for requests with different sampling requirements, ultimately increasing the system’s overall throughput. 
> 混合解码方法
> 虽然之前讨论的解码方法具有不同的内存共享和访问模式，但 vLLM 也可以同时处理具有不同解码偏好的 requests，这是现存系统无法高效做到的
> 这是因为 vLLM 通过一个将逻辑块转化为物理块的通用的映射层隐藏了不同序列之间复杂的内存共享模式，LLM 和其执行 kernel 仅看到每个序列的物理块 ID 列表，而不需要处理序列之间的共享模式
> 相较于现有系统，该方法扩大了具有不同采样/解码请求的 requests 的批处理机会，最终提高了系统的整体吞吐量

## 4.5 Scheduling and Preemption 
When the request traffic surpasses the system’s capacity, vLLM must prioritize a subset of requests. In vLLM, we adopt the first-come-first-serve (FCFS) scheduling policy for all requests, ensuring fairness and preventing starvation. When vLLM needs to preempt requests, it ensures that the earliest arrived requests are served first and the latest requests are preempted first. 
> 当 request 流量超过了系统处理能力，vLLM 必须有限处理一部分 requests
> vLLM 对于所有的 requests 采用先到先服务调度策略，以确保公平性并防止饥饿现象的发生
> 当 vLLM 需要抢占 requests 时，它确保最早到达的 requests 优先被服务，而最晚/最近的 requests 则优先被抢占

LLM services face a unique challenge: the input prompts for an LLM can vary significantly in length, and the resulting output lengths are not known a priori, contingent on both the input prompt and the model. As the number of requests and their outputs grow, vLLM can run out of the GPU’s physical blocks to store the newly generated KV cache. There are two classic questions that vLLM needs to answer in this context: (1) Which blocks should it evict? (2) How to recover evicted blocks if needed again? Typically, eviction policies use heuristics to predict which block will be accessed furthest in the future and evict that block. Since in our case we know that all blocks of a sequence are accessed together, we implement an all-or-nothing eviction policy, i.e., either evict all or none of the blocks of a sequence. Furthermore, multiple sequences within one request (e.g., beam candidates in one beam search request) are gang-scheduled as a sequence group . The sequences within one sequence group are always preempted or rescheduled together due to potential memory sharing across those sequences. To answer the second question of how to recover an evicted block, we consider two techniques: 
> LLM 服务的一个独特挑战是：LLM 的输入 prompt 的长度之间的差异会很大，以及输出的长度也不是预先可知的，而是取决于输入 prompt 和模型
> 随着 requests 的数量和它们的输出长度增长，vLLM 可能会耗尽 GPU 的物理块，以至于无法存储新生成的 KV cache，在该背景下，vLLM 需要回答两个经典问题：
> 1. 应该淘汰哪些块？
> 2. 如果需要再次使用这些块，如何恢复它们？
> 一般地，淘汰策略使用启发式算法预测哪个块将在未来最不容易被访问
> 对于第一个问题的回答：
> 在处理一个序列时，我们需要访问序列中的所有 KV 块，因此我们实现 all-or-nothing 淘汰策略，也就是要么不淘汰，要么淘汰一个序列所有的 KV 块
> 另外，单个 request 对应多个序列 (例如 beam search request 的多个 beam candidates) 则作为序列组共同被调度
> 因为一个序列内的序列可能存在内存共享，因此一个组内的序列总是被一起抢占或重新调度
> 对于第二个问题的回答，我们考虑以下两个技术：

**Swapping.** This is the classic technique used by most virtual memory implementations which copy the evicted pages to a swap space on the disk. In our case, we copy evicted blocks to the CPU memory. As shown in Fig. 4, besides the GPU block allocator, vLLM includes a CPU block allocator to manage the physical blocks swapped to CPU RAM. When vLLM exhausts free physical blocks for new tokens, it selects a set of sequences to evict and transfer their KV cache to the CPU. Once it preempts a sequence and evicts its blocks, vLLM stops accepting new requests until all preempted sequences are completed. Once a request completes, its blocks are freed from memory, and the blocks of a preempted sequence are brought back in to continue the processing of that sequence. Note that with this design, the number of blocks swapped to the CPU RAM never exceeds the number of total physical blocks in the GPU RAM, so the swap space on the CPU RAM is bounded by the GPU memory allocated for the KV cache. 
> 交换
> 交换是虚拟内存实现中的经典技术，它将被淘汰的页交换到磁盘上的交换区
> vLLM 将淘汰的块写到 CPU 内存，如 Figure 4 所示，vLLM 有 GPU block allocator 和 CPU block allocator，CPU block allocator 负责管理交换到 CPU RAM 的物理块，当 vLLM 耗尽可以用于新 token 的物理块后，它选择一组序列，将其 KV cache 块交换到 CPU RAM
> 当 vLLM 抢占了一个序列，并将其 KV cache 块全部淘汰后，vLLM 将停止接受新的请求，直到所有被抢占的序列完成 (防止饥饿)
> 抢占请求完成后，它的 KV 块就从内存中被释放，被它抢占的序列的块会被交换回来，vLLM 继续处理该序列
> 在该设计下，交换到 CPU RAM 的 KV 块的数量将永远不会超过 GPU RAM 中的总物理块数量，因此 CPU RAM 的交换空间的使用上限就是 GPU 内存为 KV cache 分配大小的上限 (最坏情况下，被交换出的序列占据了 GPU RAM 的全部物理块，故此时 CPU RAM 中交换区的大小就等于 GPU 内存中为 KV cache 块分配的总空间大小)

**Recomputation.** In this case, we simply recompute the KV cache when the preempted sequences are rescheduled. Note that re-computation latency can be significantly lower than the original latency, as the tokens generated at decoding can be concatenated with the original user prompt as a new prompt—their KV cache at all positions can be generated in one prompt phase iteration. 
> 重计算
> 另一种选择是当被抢占的序列被重新调度时，直接重新计算它的 KV cache
> 注意重新计算 KV cache 的延迟会显著低于原来的延迟，因为在解码阶段生成的 tokens 可以和用户 prompt 拼接起来，作为新的 prompt，因此在单个 prompt 阶段就生成了之前所有 tokens 的 KV cache

The performances of swapping and re-computation depend on the bandwidth between CPU RAM and GPU memory and the computation power of the GPU. We examine the speeds of swapping and re-computation in $\S7.3$ . 
> 交换策略和重计算策略的性能取决于 CPU RAM 和 GPU DRAM 之间的带宽和 GPU 的计算能力

## 4.6 Distributed Execution 
Many LLMs have parameter sizes exceeding the capacity of a single GPU [5 , 9]. Therefore, it is necessary to partition them across distributed GPUs and execute them in a model parallel fashion [28 , 63]. This calls for a memory manager capable of handling distributed memory. vLLM is effective in distributed settings by supporting the widely used Megatron-LM style tensor model parallelism strategy on Transformers [47]. This strategy adheres to an SPMD (Single Program Multiple Data) execution schedule, wherein the linear layers are partitioned to perform block-wise matrix multiplication, and the the GPUs constantly synchronize intermediate results via an allreduce operation. Specifically, the attention operator is split on the attention head dimension, each SPMD process takes care of a subset of attention heads in multi-head attention. 
> 许多 LLM 的参数大小超过了单个 GPU 的 DRAM 大小，因此需要将这些参数划分到多个 GPU 上，并采用模型并行的方式进行训练和推理
> vLLM 支持 Megatron-LM 风格的 tensor 模型并行策略，因此在分布式执行的情况下同样高效
> 该策略遵循单程序多数据的执行调度，其中线性层被划分以执行分块的矩阵乘法，并且 GPUs 持续地通过 allreduce 操作进行同步
> 特别地，attention 算子在 attention head 维度上被划分，每个 SPMD 进程处理多头注意力计算中的一部分头

We observe that even with model parallel execution, each model shard still processes the same set of input tokens, thus requiring the KV Cache for the same positions. Therefore, vLLM features a single KV cache manager within the centralized scheduler, as in Fig. 4. Different GPU workers share the manager, as well as the mapping from logical blocks to physical blocks. This common mapping allows GPU workers to execute the model with the physical blocks provided by the scheduler for each input request. Although each GPU worker has the same physical block IDs, a worker only stores a portion of the KV cache for its corresponding attention heads. 
> 在模型并行执行下，每个模型碎片仍然要处理相同的一组输入 tokens，因此需要相同位置的 KV cache
> 因此，vLLM 在中心化的调度器中仅需要一个 KV cache 管理器，如 Figure4 所示，不同的 GPU worker 共享该管理器，以及共享逻辑块到物理块的映射
> 处理一个输入 request 的 KV cache 时，虽然每个 GPU worker 都具有相同的物理块 IDs，但一个 GPU worker (的物理块) 仅存储对应的 attention heads 的一部分 KV cache

In each step, the scheduler first prepares the message with input token IDs for each request in the batch, as well as the block table for each request. Next, the scheduler broadcasts this control message to the GPU workers. Then, the GPU workers start to execute the model with the input token IDs. In the attention layers, the GPU workers read the KV cache according to the block table in the control message. During execution, the GPU workers synchronize the intermediate results with the all-reduce communication primitive without the coordination of the scheduler, as in [47]. In the end, the GPU workers send the sampled tokens of this iteration back to the scheduler. In summary, GPU workers do not need to synchronize on memory management as they only need to receive all the memory management information at the beginning of each decoding iteration along with the step inputs. 
> 在执行的每一步，调度器首先根据 batch 中每个 request 的输入 tokens 确定 token IDs，并且为每个 request 准备 block table；然后将带有 token IDs 和 table 信息的控制消息广播到 GPU workers
> 之后，GPU workers 根据收到的 token IDs 开始执行模型
> 在 attention 层，GPU worker 根据控制消息中的 block table 读取 KV cache
> 在执行时，GPU workers 之间通过 all-reduce 通讯原语同步中间结果，不需要调度器的协助
> 最后，GPU workers 将该次迭代采样得到的 tokens 返回给调度器
> 总之，GPU workers 不需要在 memory 管理上同步，因为它们仅需要在每次解码迭代的开始接受所有的 memory 管理信息 (以及该步的输入)

# 5 Implementation
vLLM is an end-to-end serving system with a FastAPI [15] frontend and a GPU-based inference engine. The frontend extends the OpenAI API [34] interface, allowing users to customize sampling parameters for each request, such as the maximum sequence length and the beam width $k$ . The vLLM engine is written in 8.5K lines of Python and 2K lines of C++/CUDA code. We develop control-related components including the scheduler and the block manager in Python while developing custom CUDA kernels for key operations such as Paged Attention. For the model executor, we implement popular LLMs such as GPT [5], OPT [62], and LLaMA [52] using PyTorch [39] and Transformers [58]. We use NCCL [32] for tensor communication across the distributed GPU workers. 
> vLLM 是端到端的服务系统，具有 FastAPI 前端和基于 GPU 的推理引擎
> 前端拓展了 OpenAI API 接口，允许用户为每个 request 自定义采样参数，例如最大序列长度和束宽度 $k$
> vLLM 引擎为 Python + C++/CUDA，和控制相关的组件，包括调度器和块管理器都用 Python 开发，对于关键运算例如 PagedAttention 则实现为 CUDA
>  kernel
>  常见的 LLM 例如 GPT、OPT、LLaMA 等使用 PyTorch 和 Transformers 实现
>  GPU workers 之间的 tensor 通讯使用 NCCL 实现

## 5.1 Kernel-level Optimization 
Since Paged Attention introduces memory access patterns that are not efficiently supported by existing systems, we develop several GPU kernels for optimizing it. (1) *Fused reshape and block write.* In every Transformer layer, the new KV cache are split into blocks, reshaped to a memory layout optimized for block read, then saved at positions specified by the block table. To minimize kernel launch overheads, we fuse them into a single kernel. (2) *Fusing block read and attention.* We adapt the attention kernel in Faster Transformer [31] to read KV cache according to the block table and perform attention operations on the fly. To ensure coalesced memory access, we assign a GPU warp to read each block. Moreover, we add support for variable sequence lengths within a request batch. (3) *Fused block copy.* Block copy operations, issued by the copy-on-write mechanism, may operate on discontinuous blocks. This can lead to numerous invocations of small data movements if we use the `cudaMemcpyAsync` API. To mitigate the overhead, we implement a kernel that batches the copy operations for different blocks into a single kernel launch. 
> PagedAttention 的内存访问模式由多个 GPU kernel 优化，包括
> (1) 融合的 reshape 和 block write kernel
> 在每个 Transformer 层中，新的 KV cache 会被划分为块，并 reshape 到适合 blockwise 读取的内存布局，然后存储/写到 block table 指定的位置
> 这两个操作被融合为一个 kernel 以减少 kernel launch 开销
> (2) 融合的 block read 和 attention kernel
> 我们采用 Faster Transformer 中的 attention kernel 来根据 block table 读取 KV cache，并同时执行 attention 操作
> 为了保证合并的内存访问，我们为每个 block 的读取分配单个 GPU warp，另外，我们还为序列长度不一的 request batch 添加了支持
> (3) 融合的 block copy kernel
> block copy 操作由写时拷贝机制发起，该操作可能对非连续的 block 执行，如果我们使用 `cudaMemcpyAsync` API，这会导致调用许多次小数据移动
> 为了缓解该开销，我们实现将不同的 block 的拷贝操作进行批处理的 kernel

## 5.2 Supporting Various Decoding Algorithms 
vLLM implements various decoding algorithms using three key methods: `fork` , `append` , and `free` . The `fork` method creates a new sequence from an existing one. The `append` method appends a new token to the sequence. Finally, the `free` method deletes the sequence. For instance, in parallel sampling, vLLM creates multiple output sequences from the single input sequence using the `fork` method. It then adds new tokens to these sequences in every iteration with `append` , and deletes sequences that meet a stopping condition using `free` . The same strategy is also applied in beam search and prefix sharing by vLLM. We believe future decoding algorithms can also be supported by combining these methods. 
> vLLM 使用三个关键方法：`fork/append/free` 实现多种解码算法
> `fork` 方法从现有序列创建新序列
> `append` 方法将新 token 添加到序列上
> `free` 方法删除序列
> 例如，在并行采样时，vLLM 首先用 `fork` 从单个输入序列创建多个输出序列，然后在每个迭代使用 `append` 将新的 tokens 各自添加到这些输出序列上，最后使用 `free` 删除满足停止条件的序列
> beam search 和 prefix sharing 也采用同样策略

# 6 Evaluation 
In this section, we evaluate the performance of vLLM under a variety of workloads. 

## 6.1 Experimental Setup 
**Model and server configurations.** We use OPT [62] models with 13B, 66B, and 175B parameters and LLaMA [52] with 13B parameters for our evaluation. 13B and 66B are popular sizes for LLMs as shown in an LLM leader board [38], while 175B is the size of the famous GPT-3 [5] model. For all of our experiments, we use A2 instances with NVIDIA A100 GPUs on Google Cloud Platform. The detailed model sizes and server configurations are shown in Table 1. 
> Model and server configurations
> 模型使用 OPT 13/66/175B 和 LLaMA 13B

![[vLLM-Table 1.png]]

**Workloads.** We synthesize workloads based on ShareGPT [51] and Alpaca [50] datasets, which contain input and output texts of real LLM services. The ShareGPT dataset is a collection of user-shared conversations with ChatGPT [35]. The Alpaca dataset is an instruction dataset generated by GPT3.5 with self-instruct [57]. We tokenize the datasets and use their input and output lengths to synthesize client requests. As shown in Fig. 11, the ShareGPT dataset has $8.4\times$ longer input prompts and $5.8\times$ longer outputs on average than the Alpaca dataset, with higher variance. Since these datasets do not include timestamps, we generate request arrival times using Poisson distribution with different request rates. 
> Workloads
> workload 基于 ShareGPT 和 Alpaca 数据集进行合成，这些数据集包含了真实 LLM 服务的输入和输出文本，其中ShareGPT 数据集是一组用户和 ChatGPT 的对话，Alpaca 数据集是由 GPT3.5 在 self-instruct 下生成的指令数据集
> 我们将这些数据集 tokenize，然后使用它们的输入和输出长度来合成 requests
> 如 Figure 11，可以看到 ShareGPT 的输入和输出长度都长于 Alpaca，同时方差更大
> 因为数据集不包含时间戳，我们使用不同请求率的 Possion 分布生成 request 到达时间

![[vLLM-Figure11.png]]

**Baseline 1: Faster Transformer.** Faster Transformer [31] is a distributed inference engine highly optimized for latency. 

As Faster Transformer does not have its own scheduler, we implement a custom scheduler with a dynamic batching mechanism similar to the existing serving systems such as Triton [30]. Specifically, we set a maximum batch size $B$ as large as possible for each experiment, according to the GPU memory capacity. The scheduler takes up to $B$ number of earliest arrived requests and sends the batch to Faster Transformer for processing. 

> Baseline 1: Faster Transformer
> Faster Transformer 是分布式的推理引擎
> Faster Transformer 没有自己的调度器，我们为其实现了带有动态 batching 机制的自定义调度器，类似于现存的服务系统，例如 Triton
> 每次试验的最大 batch size $B$ 都设定为越大越好，调度器最多接受 $B$ 个最早到达的 requests，然后将其作为 batch 发送给 Faster Transformer

**Baseline 2: Orca.** Orca [60] is a state-of-the-art LLM serving system optimized for throughput. Since Orca is not publicly available for use, we implement our own version of Orca. We assume Orca uses the buddy allocation algorithm to determine the memory address to store KV cache. We implement three versions of Orca based on how much it over-reserves the space for request outputs: 

- **Orca (Oracle).** We assume the system has the knowledge of the lengths of the outputs that will be actually generated for the requests. This shows the upper-bound performance of Orca, which is infeasible to achieve in practice.
- **Orca (Pow2).** We assume the system over-reserves the space for outputs by at most $2\times$  . For example, if the true output length is 25, it reserves 32 positions for outputs.
- **Orca (Max).** We assume the system always reserves the space up to the maximum sequence length of the model, i.e., 2048 tokens. 

> Baseline 2: Orca
> Orca 为 SOTA 的 LLM 服务系统
> 我们实现了自己的 Orca，其中假定了 Orca 使用 buddy 分配算法来决定存储 KV cache 的内存地址
> 基于 Orca 是如何为 request 的输出预留内存空间的，我们实现了三个版本的 Orca，包括：
> - Orca (Orcale)，假设系统知道输出序列的长度，该版本是 Orca 的性能上限，在实际中不会达到
> - Orca (Pow2)，假设系统为输出序列预留的空间为大于输出序列长度的最小的2的幂次，例如为长度为 25 的输出序列预留 32 个位置
> - Orca (Max)，假设系统预留的空间总是保持模型的最大序列长度，例如 2048

**Key metrics.** We focus on serving throughput. Specifically, using the workloads with different request rates, we measure normalized latency of the systems, the mean of every request’s end-to-end latency divided by its output length, as in Orca [60]. A high-throughput serving system should retain low normalized latency against high request rates. For most experiments, we evaluate the systems with 1-hour traces. As an exception, we use 15-minute traces for the OPT-175B model due to the cost limit. 
> Key metrics
> 我们聚焦于吞吐量
> 具体地说，我们使用不同请求率下的工作负载度量系统的规范化延迟，即每个请求的端到端延迟除以它的输出长度的平均值
> 高吞吐量的服务系统应该在高的请求率下保持低的规范化延迟
> 大多数试验使用一小时的跟踪数据评估，OPT-175B 使用15分钟的跟踪数据

## 6.2 Basic Sampling
We evaluate the performance of vLLM with basic sampling (one sample per request) on three models and two datasets. The first row of Fig. 12 shows the results on the ShareGPT dataset. The curves illustrate that as the request rate increases, the latency initially increases at a gradual pace but then suddenly explodes. This can be attributed to the fact that when the request rate surpasses the capacity of the serving system, the queue length continues to grow infinitely and so does the latency of the requests. 
> 我们在三个模型和两个数据集上评估了 vLLM 的基础采样 (每个 request 仅采样一个样本)，如 Figure 12 所示
> 随着请求率逐渐增大，延迟一开始逐渐提升，然后突然猛增，其原因在于当请求率超过了服务系统的能力，排队等待处理的请求数量将无限制增大，故延迟也将无限制增大

![[vLLM-Figure12.png]]

On the ShareGPT dataset, vLLM can sustain $1.7\times-2.7\times$ higher request rates compared to Orca (Oracle) and $2.7\times-8\times$ compared to Orca (Max), while maintaining similar latencies. This is because vLLM’s Paged Attention can efficiently manage the memory usage and thus enable batching more requests than Orca. For example, as shown in Fig. 13a, for OPT-13B vLLM processes $2.2\times$ more requests at the same time than Orca (Oracle) and $4.3\times$ more requests than Orca (Max). Compared to Faster Transformer, vLLM can sustain upto $22\times$ higher request rates, as Faster Transformer does not utilize a fine-grained scheduling mechanism and inefficiently manages the memory like Orca (Max). 
> 在 ShareGPT 数据集上，vLLM 相较于 Orca (Orcale) 可以维持 1.7x-2.7x 倍更高的请求率，相较于 Orca (Max) 可以维持 2.7x-8x 倍更高的请求率，同时延迟接近
> 原因是 vLLM 的 PagedAttention 可以高效管理内存使用，故可以批处理更多的请求

The second row of Fig. 12 and Fig. 13b shows the results on the Alpaca dataset, which follows a similar trend to the ShareGPT dataset. One exception is Fig. 12 (f), where vLLM’s advantage over Orca (Oracle) and Orca (Pow2) is less pronounced. This is because the model and server configuration for OPT-175B (Table 1) allows for large GPU memory space available to store KV cache, while the Alpaca dataset has short sequences. In this setup, Orca (Oracle) and Orca (Pow2) can also batch a large number of requests despite the inefficiencies in their memory management. As a result, the performance of the systems becomes compute-bound rather than memory-bound. 
> Alpaca 上的结果和 ShareGPT 上的结果类似
> 在 OPT-175B 时 vLLM 的优势相对不高，原因在于 OPT-175B 的模型和服务配置 (Table 1) 允许使用更大的 GPU 显存空间存储 KV cache，同时 Alpaca 的序列主要是短序列，因此 Orca (Oracle), Orca (Pow2) 即便内存管理低效，也可以批处理大量请求，因此系统的性能更倾向于 compute-bound 而不是 memory-bound

![[vLLM-Figure13.png]]

## 6.3 Parallel Sampling and Beam Search 
We evaluate the effectiveness of memory sharing in PagedAttention with two popular sampling methods: parallel sampling and beam search. In parallel sampling, all parallel sequences in a request can share the KV cache for the prompt. As shown in the first row of Fig. 14, with a larger number of sequences to sample, vLLM brings more improvement over the Orca baselines. Similarly, the second row of Fig. 14 shows the results for beam search with different beam widths. Since beam search allows for more sharing, vLLM demonstrates even greater performance benefits. The improvement of vLLM over Orca (Oracle) on OPT-13B and the Alpaca dataset goes from $1.3\times$ in basic sampling to $2.3\times$ in beam search with a width of 6. 
> 我们使用并行采样和束搜索评估 PagedAttention 的内存共享的有效性
> 并行采样中，request 的所有并行序列共享 prompt 的 KV cache，如 Figure 14所示，当并行采样的数量越多，vLLM 相较于 Orca 的优势就越大
> 束搜索的结果也类似，当 beam 宽度越大，vLLM 优势越大，并且由于 beam search 可以有更多的共享机会，vLLM 的优势也更加显著

![[vLLM-Figure14.png]]

Fig. 15 plots the amount of memory saving, computed by the number of blocks we saved by sharing divided by the number of total blocks without sharing. We show $6.1\%-9.8\%$ memory saving on parallel sampling and $37.6\%\textrm{-}55.2\%$ on beam search. In the same experiments with the ShareGPT dataset, we saw $16.2\%\textrm{-}30.5\%$ memory saving on parallel sampling and $44.3\%\textrm{-}66.3\%$ on beam search. 
> vLLM 对内存节约的比例如 Figure 15 所示，比例通过将共享的块的数量除以没有共享的块的数量得到
> Alpaca 数据集上，在并行采样时，内存的节约程度达到 6.1%-9.8%，beam search 时，内存的节约程度达到 37.6%-55.2%

![[vLLM-Figure15.png]]

## 6.4 Shared prefix 
We explore the effectiveness of vLLM for the case a prefix is shared among different input prompts, as illustrated in Fig. 10. For the model, we use LLaMA-13B [52], which is multilingual. For the workload, we use the WMT16 [4] Englishto-German translation dataset and synthesize two prefixes that include an instruction and a few translation examples. The first prefix includes a single example (i.e., one-shot) while the other prefix includes 5 examples (i.e., few-shot). As shown in Fig. 16 (a), vLLM achieves $1.67\times$ higher throughput than Orca (Oracle) when the one-shot prefix is shared. Furthermore, when more examples are shared (Fig. 16 (b)), vLLM achieves $3.58\times$ higher throughput than Orca (Oracle). 
> 我们使用 LLaMA-13B 探究 vLLM 对于不同输入 prompt 共享前缀的效率，我们使用 WMT16 English-German 翻译数据集，为数据合成了两个前缀，每个前缀包括一个指令和一部分翻译示例样本，作为 workload
> 第一个前缀是 one-shot，仅包含单个示例样本，第二个前缀为 few-shot，包含5个示例样本
> 结果见 Figure 16，可以看到前缀越长，效果越明显

![[vLLM-Figure16.png]]

## 6.5 Chatbot 
A chatbot [8 , 19 , 35] is one of the most important applications of LLMs. To implement a chatbot, we let the model generate a response by concatenating the chatting history and the last user query into a prompt. We synthesize the chatting history and user query using the ShareGPT dataset. Due to the limited context length of the OPT-13B model, we cut the prompt to the last 1024 tokens and let the model generate at most 1024 tokens. We do not store the KV cache between different conversation rounds as doing this would occupy the space for other requests between the conversation rounds. 
> 要实现 chatbot，我们要让模型将聊天历史和用户查询拼接为 prompt
> 我们使用 ShareGPT 数据集合成聊天历史和用户查询
> OPT-13B 的上下文窗口长度有限，故我们将 prompt 长度设定为最后的 1024 tokens，并且让模型最多生成 1024 tokens
> 我们不保存对话轮次之间的 KV cache，防止在对话轮次之间占用其他请求的空间

Fig. 17 shows that vLLM can sustain $2\times$ higher request rates compared to the three Orca baselines. Since the ShareGPT dataset contains many long conversations, the input prompts for most requests have 1024 tokens. Due to the buddy allocation algorithm, the Orca baselines reserve the space for 1024 tokens for the request outputs, regardless of how they predict the output lengths. For this reason, the three Orca baselines behave similarly. In contrast, vLLM can effectively handle the long prompts, as Paged Attention resolves the problem of memory fragmentation and reservation. 
> vLLM 相较于 Orca 可以维持 2x 以上的请求率
> ShareGPT 包含许多长对话，因此许多请求 prompt 都达到 1024 tokens
> Orca 的 buddy 分配算法总是为请求输出预留 1024 tokens 的空间，无论实际输出多长，因此三种 Orca 实现的表现都类似
> 而 vLLM 解决了内存碎片和预留的问题，故可以高效处理长的 prompt

![[vLLM-Figure17.png]]

# 7 Ablation Studies 
In this section, we study various aspects of vLLM and evaluate the design choices we make with ablation experiments. 

## 7.1 Kernel Microbenchmark 
The dynamic block mapping in Paged Attention affects the performance of the GPU operations involving the stored KV cache, i.e., block read/writes and attention. Compared to the existing systems, our GPU kernels (§5) involve extra overheads of accessing the block table, executing extra branches, and handling variable sequence lengths. As shown in Fig. 18a, this leads to $20{-}26\%$ higher attention kernel latency, compared to the highly-optimized Faster Transformer implementation. We believe the overhead is small as it only affects the attention operator but not the other operators in the model, such as Linear. Despite the overhead, Paged Attention makes vLLM significantly outperform Faster Transformer in end-to-end performance (§6). 
> PagedAttention 的动态 block 映射会影响涉及到 KV cache 的 GPU 操作的表现，即 block 读写和 attention 计算
> 相较于现有系统，我们的 GPU kernel 包含了访问 block table、执行额外分支、处理可变序列长度的额外开销，如 Figure 18a 所示，这将导致 attention kernel 比 Faster Transformer 实现多出 20-26% 的延迟
> 但 vLLM 的端到端表现仍然显著高于 Faster Transformer

![[vLLM-Figure18.png]]

## 7.2 Impact of Block Size 
The choice of block size can have a substantial impact on the performance of vLLM. If the block size is too small, vLLM may not fully utilize the GPU’s parallelism for reading and processing KV cache. If the block size is too large, internal fragmentation increases and the probability of sharing decreases. 
> KV block 太小时，vLLM 可能无法完全利用 GPU 的并行性质优化 KV cache 的读取和处理，KV block 太大时，则会导致更大的内部碎片，且能共享的概率降低

In Fig. 18b, we evaluate the performance of vLLM with different block sizes, using the ShareGPT and Alpaca traces with basic sampling under fixed request rates. In the ShareGPT trace, block sizes from 16 to 128 lead to the best performance. In the Alpaca trace, while the block size 16 and 32 work well, larger block sizes significantly degrade the performance since the sequences become shorter than the block sizes. In practice, we find that the block size 16 is large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads. Accordingly, vLLM sets its default block size as 16. 
> Figure 18b 评估了 vLLM 在不同 block size 下的表现 (basic sampling, fixed request rate)
> ShareGPT 的 block size 最好在 16-128，Alpaca 的 block size 过大时表现显著降低，因为 block size 超过了序列长度
> 实践中，block size = 16 的效果最优，可以在利用 GPU 并行性的同时避免大多数 workload 中的过大内部碎片

## 7.3 Comparing Recomputation and Swapping 
vLLM supports both recomputation and swapping as its recovery mechanisms. To understand the tradeoffs between the two methods, we evaluate their end-to-end performance and micro benchmark their overheads, as presented in Fig. 19. Our results reveal that swapping incurs excessive overhead with small block sizes. This is because small block sizes often result in numerous small data transfers between CPU and GPU, which limits the effective PCIe bandwidth. In contrast, the overhead of re computation remains constant across different block sizes, as re computation does not utilize the KV blocks. Thus, recomputation is more efficient when the block size is small, while swapping is more efficient when the block size is large, though recomputation overhead is never higher than $20\%$ of swapping’s latency. For medium block sizes from 16 to 64, the two methods exhibit comparable end-to-end performance. 
> vLLM 支持的抢占恢复机制有重计算和交换
> 我们评估了这两种方法的端到端表现，并且测试了它们的开销，如 Figure 19 所示
> 交换方法在 block size 较小时会显著开销，因为小的 block size 容易导致 CPU 和 GPU 之间有过多的小数据传输，限制了有效 PCIe 带宽
> 重计算的开销随 block size 变化基本不变，因为重计算不涉及数据传输，不会使用 KV blocks
> 因此 block size 较小时重计算较高效，block size 较大时交换方法较高效，block size 为中等大小时，二者的端到端表现可比

![[vLLM-Figure19.png]]

# 8 Discussion 
**Applying the virtual memory and paging technique to other GPU workloads.** The idea of virtual memory and paging is effective for managing the KV cache in LLM serving because the workload requires dynamic memory allocation (since the output length is not known a priori) and its performance is bound by the GPU memory capacity. However, this does not generally hold for every GPU workload. For example, in DNN training, the tensor shapes are typically static, and thus memory allocation can be optimized ahead of time. For another example, in serving DNNs that are not LLMs, an increase in memory efficiency may not result in any performance improvement since the performance is primarily compute-bound. In such scenarios, introducing the vLLM’s techniques may rather degrade the performance due to the extra overhead of memory indirection and non-contiguous block memory. However, we would be excited to see vLLM’s techniques being applied to other workloads with similar properties to LLM serving. 
> Applying the virtual memory and paging techinque to other GPU workloads
> 虚拟内存和分页机制在管理 KV cache 时高效的原因在于 LLM 的 workload 需要动态内存分配 (因为输出长度不能提前预知)，故性能受 GPU 显存容量限制
> 对于其他的 GPU workload 这一点不一定成立
> 例如训练 DNN 时，张量的形状一般是静态的，因此可以提前优化内存分配；同时对于不是 LLM 的 DNN 来说，内存效率的提升不一定会让性能提升，因为性能也可能是 compute-bound
> 对于这样的场景，vLLM 技术可能反而会降低性能，因为间接内存和不连续的块式内存会引入额外开销

**LLM-specific optimizations in applying virtual memory and paging.** vLLM re-interprets and augments the idea of virtual memory and paging by leveraging the application specific semantics. One example is vLLM’s all-or-nothing swap-out policy, which exploits the fact that processing a request requires all of its corresponding token states to be stored in GPU memory. Another example is the recomputation method to recover the evicted blocks, which is not feasible in OS. Besides, vLLM mitigates the overhead of memory in direction in paging by fusing the GPU kernels for memory access operations with those for other operations such as attention.
> LLM-specific optimizations in applying virtual memory and paging
> vLLM 在针对应用程序的语义上重新解释并强化了虚拟内存和分页的思想
> 一个例子就是 vLLM 的全有或全无的换出策略，这基于的事实是处理一个请求需要它所有对应的 tokens 的状态都被存储在 GPU 显存中
> 另一个例子是可以用重计算恢复被驱逐的数据块，这在 OS 中是不可行的
> 此外，vLLM 通过融合了执行内存访问操作的 kernel 和执行其他操作例如 attention 的 kernel 来缓解了内存间接访问的开销

# 9 Related Work 
**General model serving systems.** Model serving has been an active area of research in recent years, with numerous systems proposed to tackle diverse aspects of deep learning model deployment. Clipper [11], TensorFlow Serving [33], Nexus [45], InferLine [10], and Clockwork [20] are some earlier general model serving systems. They study batching, caching, placement, and scheduling for serving single or multiple models. More recently, DVABatch [12] introduces multi-entry multi-exit batching. REEF [21] and Shepherd [61] propose preemption for serving. AlpaServe [28] utilizes model parallelism for statistical multiplexing. However, these general systems fail to take into account the autoregressive property and token state of LLM inference, resulting in missed opportunities for optimization. 
> 通用模型服务系统
> 模型服务近年来一直是研究的热点领域，许多系统被提出以解决深度学习模型部署的各种方面问题
> Clipper [11]、TensorFlow Serving [33]、Nexus [45]、InferLine [10] 和Clockwork [20] 是一些较早的通用模型服务系统。它们研究了批量处理、缓存、部署位置和调度等问题，用于服务单个或多个模型
> 最近，DVABatch [12] 引入了多入口多出口批量处理。REEF [21] 和Shepherd [61] 提出了预调度服务的方法。AlpaServe [28] 利用了模型并行性来进行统计复用
> 然而，这些通用系统未能考虑到LLM推理中的自回归特性和token状态，导致错失了优化的机会。

**Specialized serving systems for transformers.** Due to the significance of the transformer architecture, numerous specialized serving systems for it have been developed. These systems utilize GPU kernel optimization s [1, 29, 31, 56], advanced batching mechanisms [14 , 60], model parallelism [1 , 41 , 60], and parameter sharing [64] for efficient serving. Among them, Orca [60] is most relevant to our approach. 
>  针对 transformers 的服务系统
>  有许多专门的针对 transformer 架构的服务系统，这些系统利用了 GPU 内核优化[1, 29, 31, 56]、高级批量处理机制[14, 60]、模型并行性[1, 41, 60]以及参数共享[64]，以实现高效的服务。其中，Orca [60] 最接近我们的方法。

**Comparison to Orca.** The iteration-level scheduling in Orca [60] and Paged Attention in vLLM are complementary techniques: While both systems aim to increase the GPU utilization and hence the throughput of LLM serving, Orca achieves it by scheduling and interleaving the requests so that more requests can be processed in parallel, while vLLM is doing so by increasing memory utilization so that the working sets of more requests fit into memory. By reducing memory fragmentation and enabling sharing, vLLM runs more requests in a batch in parallel and achieves a $2–4\times$ speedup compared to Orca. Indeed, the fine-grained scheduling and interleaving of the requests like in Orca makes memory management more challenging, making the techniques proposed in vLLM even more crucial. 
> 与 Orca 的对比
> Orca [60]中的迭代级别调度和 vLLM 中的分页注意力机制是互补的技术：虽然两个系统都旨在提高 GPU 利用率和 LLM 服务的吞吐量，但 Orca 通过调度和交错请求使得更多请求可以并行处理，而 vLLM 则是通过增加内存利用率使更多请求的工作集可以容纳在显存中
> 通过减少内存碎片并启用共享，vLLM 能够并行运行更多的请求，并相较于 Orca 实现了2-4倍的速度提升
> 实际上，像 Orca 那样对请求进行细粒度调度和交错处理会使内存管理更加复杂，这也使得 vLLM 中提出的技术更为关键

**Memory optimizations.** The widening gap between the compute capability and memory capacity of accelerators has caused memory to become a bottleneck for both training and inference. Swapping [23 , 42 , 55], re computation [7 , 24] and their combination [40] have been utilized to reduce the peak memory of training. Notably, FlexGen [46] studies how to swap weights and token states for LLM inference with limited GPU memory, but it does not target the online serving settings. OLLA [48] optimizes the lifetime and location of tensors to reduce fragmentation, but it does not do finegrained block-level management or online serving. FlashAttention [13] applies tiling and kernel optimization s to reduce the peak memory of attention computation and reduce I/O costs. This paper introduces a new idea of block-level memory management in the context of online serving. 
> 内存优化
> 加速设备的计算能力和内存容量之间的差距越来越大，导致内存成为了训练和推理的瓶颈，交换[23, 42, 55]、重计算[7, 24]及二者的组合[40]已被用来减少训练的峰值内存需求。
> 值得注意的是，FlexGen [46]研究了如何在有限的 GPU 内存下通过交换权重和 token 状态来进行 LLM 推理，但它并不针对在线服务场景。OLLA [48]优化了张量的生命周期和位置，以减少碎片化，但并没有进行细粒度的块级管理和在线服务。FlashAttention [13]通过分块和内核优化来减少注意力计算的峰值内存和 I/O 成本
> 本文介绍了一种新的基于块级别的内存管理思想，适用于在线服务场景。

# 10 Conclusion 
This paper proposes Paged Attention, a new attention algorithm that allows attention keys and values to be stored in non-contiguous paged memory, and presents vLLM, a high-throughput LLM serving system with efficient memory management enabled by Paged Attention. Inspired by operating systems, we demonstrate how established techniques, such as virtual memory and copy-on-write, can be adapted to efficiently manage KV cache and handle various decoding algorithms in LLM serving. Our experiments show that vLLM achieves $2–4\times$ throughput improvements over the state-of-the-art systems. 
> 本文提出了 PagedAttention，该算法允许 attention keys 和 values 被存储在不连续的分页内存
> 本文展示了 vLLM，一个高吞吐的 LLM 服务系统，使用 PagedAttention 进行高效内存管理
> 我们展示了如何应用成熟的技术，例如虚拟内存和写时拷贝，来高效管理 KV cache 并处理 LLM 服务中的多种解码算法
> 试验标识了 vLLM 相较于 SOTA 系统实现了 2-4x 的吞吐提升