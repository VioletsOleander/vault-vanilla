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

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/f379f4f8ff7936981b822cbad79be890db2bfab117b4c0dbac83d617b029faa5.jpg) 
Figure 1. Left: Memory layout when serving an LLM with 13B parameters on NVIDIA A100. The parameters (gray) persist in GPU memory throughout serving. The memory for the KV cache (red) is (de) allocated per serving request. A small amount of memory (yellow) is used ephemeral ly for activation. Right: vLLM smooths out the rapid growth curve of KV cache memory seen in existing systems [31 , 60], leading to a notable boost in serving throughput. 

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

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/b918c57e870b0e0ef80a9b8c2e6e2a41f77338643f7d981db61ef664830c81a3.jpg) 
Figure 2. Average percentage of memory wastes in different LLM serving systems during the experiment in $\S6.2$ . 

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


![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/0eae5fffcfcd85d0cc86275c3efdcda8af4e0bab90529cf121b6be6eb02bfc9d.jpg) 
Figure 3. KV cache memory management in existing systems. Three types of memory wastes – reserved, internal fragmentation, and external fragmentation – exist that prevent other requests from fitting into the memory. The token in each memory slot represents its KV cache. Note the same tokens can have different KV cache when at different positions. 

Although compaction [54] has been proposed as a potential solution to fragmentation, performing compaction in a performance-sensitive LLM serving system is impractical due to the massive KV cache. Even with compaction, the pre-allocated chunk space for each request prevents memory sharing specific to decoding algorithms in existing memory management systems. 
> 一个解决碎片的方法是 compaction，但在性能敏感的 LLM 服务系统中执行 compaction 是不现实的，因为其 KV cache 十分庞大
> 且即便使用 compaction，为每个 request 预留块空间的方法也不能实现 request 在特定的解码算法下共享内存

# 4 Method 
In this work, we develop a new attention algorithm, PagedAttention , and build an LLM serving engine, vLLM , to tackle the challenges outlined in $\S3$ . The architecture of vLLM is shown in Fig. 4. vLLM adopts a centralized scheduler to coordinate the execution of distributed GPU workers. The KV cache manager effectively manages the KV cache in a paged fashion, enabled by Paged Attention. Specifically, the KV cache manager manages the physical KV cache memory on the GPU workers through the instructions sent by the centralized scheduler. 
> 我们提出新的 attention 算法 PagedAttention，并基于此构建 LLM 服务引擎 vLLM，vLLM 框架如 Figure 4 所示
> Figure 4 中，中心化的调度器来协调分布式 GPU workers 的执行，KV cache 管理器通过中心化的调度器发送指令来管理 GPU workers 中的物理 KV cache 内存


![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/f21af158804c76b4975845649f35c1d88155ee443f9934ddeb5b6bda7f1fa0da.jpg) 

Figure 4. vLLM system overview. 

Next, We describe the Paged Attention algorithm in $\S4.1$ . With that, we show the design of the KV cache manager in $\S4.2$ and how it facilitates Paged Attention in $\S4.3$ , respectively. Then, we show how this design facilitates effective memory management for various decoding methods (§4.4) and handles the variable length input and output sequences (§4.5). Finally, we show how the system design of vLLM works in a distributed setting (§4.6). 

## 4.1 Paged Attention 
To address the memory challenges in $\S3$ , we introduce PagedAttention , an attention algorithm inspired by the classic idea of paging [25] in operating systems. Unlike the traditional attention algorithms, Paged Attention allows storing continuous keys and values in non-contiguous memory space. Specifically, Paged Attention partitions the KV cache of each sequence into KV blocks . Each block contains the key and value vectors for a fixed number of tokens, which we denote as $K V$ block size ( $B$ ). Denote the key block $K_{j}=(k_{(j-1)B+1},\dots,k_{j B})$ and value block $V_{j}=(v_{(j-1)B+1},\dots,v_{j B})$ . The attention computation in Eq. 4 can be transformed into the following blockwise computation: 

$$
A_{i j}=\frac{\exp(q_{i}^{\top}K_{j}/\sqrt{d})}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_{i}^{\top}K_{t}\mathbf 1/\sqrt{d})},\;o_{i}=\sum_{j=1}^{\lceil i/B\rceil}V_{j}A_{i j}^{\top},\tag{4}
$$

where $A_{i j}=\left(a_{i,(j-1)B+1},\dots,a_{i,j B}\right)$ is the row vector of attention score on $j$ -th KV block. 

> PagedAttention 允许将连续的 keys 和 values 存储在非连续的内存空间
> 具体地说，PagedAttention 将每个序列的 KV cache 划分为 KV  blocks，包括 key blocks 和 value blocks，每个 key/value block 包含序列中 block size ( $B$ ) 个 token 对应的 keys 和 values，分别记作 $K_j = (k_{(j-1) B + 1}, \dots, k_{jB})$ 和 $V_j = (v_{(j-1)B + 1}, \dots, v_{jB})$
> PagedAttention 进而将 eq 3 的 attention 计算转化为如上的逐块的运算，其中 $A_{ij} = (a_{i, (j-1) B+1}, \dots, a_{i, jB})$ 为 $q_i$ 相对于第 $j$ 个 K block 的 attention score 向量
> (注：公式 (4) 显然存在错误，正确的公式应该将 $\mathbf 1$ 放在 $\exp$ 外，并且 $\mathbf 1$ 应该同时作为 indicator function，满足第 $\lceil i / B \rceil$ 的块中 $j > i$ 的 $k_j$ 对应的 $\exp (q_i^\top k_j/\sqrt d)$ 乘上零) 

During the attention computation, the Paged Attention kernel identifies and fetches different KV blocks separately. We show an example of Paged Attention in Fig. 5: The key and value vectors are spread across three blocks, and the three blocks are not contiguous on the physical memory. At each time, the kernel multiplies the query vector $q_{i}$ of the query token (*"forth"*) and the key vectors $K_{j}$ in a block (e.g., key vectors of *“Four score and seven”* for block 0) to compute the attention score $A_{i j}$ , and later multiplies $A_{i j}$ with the value vectors $V_{j}$ in a block to derive the final attention output $o_{i}$ . 
> 在 attention 计算过程中，PagedAttention kernel 会识别并分别取不同的 KV blocks


![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/045cb8319a5cda9cc8c962eb581c9386cc9a1faf746e8d1e0f845d6759f1402f.jpg) 
Figure 5. Illustration of the Paged Attention algorithm, where the attention key and values vectors are stored as non-contiguous blocks in the memory.

In summary, the Paged Attention algorithm allows the KV blocks to be stored in non-contiguous physical memory, which enables more flexible paged memory management in vLLM. 

## 4.2 KV Cache Manager 
The key idea behind vLLM’s memory manager is analogous to the virtual memory [25] in operating systems. OS partitions memory into fixed-sized pages and maps user programs’ logical pages to physical pages. Contiguous logical pages can correspond to non-contiguous physical memory pages, allowing user programs to access memory as though it were contiguous. Moreover, physical memory space needs not to be fully reserved in advance, enabling the OS to dynamically allocate physical pages as needed. vLLM uses the ideas behind virtual memory to manage the KV cache in an LLM service. Enabled by Paged Attention, we organize the KV cache as fixed-size KV blocks, like pages in virtual memory. A request’s KV cache is represented as a series of logical 

KV blocks , filled from left to right as new tokens and their KV cache are generated. The last KV block’s unfilled positions are reserved for future generations. On GPU workers, a block engine allocates a contiguous chunk of GPU DRAM and 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/5dfa6b7482d88a38fb7e1f6d7931d79921965766d28de08990b6db8bc50e55db.jpg) 
Figure 6. Block table translation in vLLM. 

divides it into physical KV blocks (this is also done on CPU RAM for swapping; see $\S4.5)$ . The KV block manager also maintains block tables —the mapping between logical and physical KV blocks of each request. Each block table entry records the corresponding physical blocks of a logical block and the number of filled positions. Separating logical and physical KV blocks allows vLLM to dynamically grow the KV cache memory without reserving it for all positions in advance, which eliminates most memory waste in existing systems, as in Fig. 2. 

# 4.3 Decoding with Paged Attention and vLLM 

Next, we walk through an example, as in Fig. 6, to demonstrate how vLLM executes Paged Attention and manages the memory during the decoding process of a single input sequence: $\textcircled{1}$ As in OS’s virtual memory, vLLM does not requirereserving the memory for the maximum possible generated sequence length initially. Instead, it reserves only the necessary KV blocks to accommodate the KV cache generated during prompt computation. In this case, The prompt has 7 tokens, so vLLM maps the first 2 logical KV blocks (0 and 1) to 2 physical KV blocks (7 and 1, respectively). In the prefill step, vLLM generates the KV cache of the prompts and the first output token with a conventional self-attention algorithm (e.g., [13]). vLLM then stores the KV cache of the first 4 tokens in logical block 0 and the following 3 tokens in logical block 1. The remaining slot is reserved for the subsequent auto regressive generation phase. $\textcircled{2}$ In the first auto regressive decoding step, vLLM generates the new token with the Paged Attention algorithm on physical blocks 7 and 1. Since one slot remains available in the last logical block, the newly generated KV cache is stored there, and the block table’s #filled record is updated. $\textcircled{3}$ At the second decoding step, as the last logical block is full, vLLM stores the newly generated KV cache in a new logical block; vLLM allocates a new physical block (physical block 3) for it and stores this mapping in the block table. 

Globally, for each decoding iteration, vLLM first selects a set of candidate sequences for batching (more in $\S4.5_{c}$ ), and allocates the physical blocks for the newly required logical blocks. Then, vLLM concatenates all the input tokensof the current iteration (i.e., all tokens for prompt phase 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/59345147feb60e4f8286e1f50a1d901d9e0ec5b34c96249cf5ad8d7b988b3bed.jpg) 
Figure 7. Storing the KV cache of two requests at the same time in vLLM. 

requests and the latest tokens for generation phase requests) as one sequence and feeds it into the LLM. During LLM’s computation, vLLM uses the Paged Attention kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks. Storing multiple tokens within a KV block (block size $>1$ ) enables the Paged Attention kernel to process the KV cache across more positions in parallel, thus increasing the hardware utilization and reducing latency. However, a larger block size also increases memory fragmentation. We study the effect of block size in $\S7.2$ . 

Again, vLLM dynamically assigns new physical blocks to logical blocks as more tokens and their KV cache are generated. As all the blocks are filled from left to right and a new physical block is only allocated when all previous blocks are full, vLLM limits all the memory wastes for a request within one block, so it can effectively utilize all the memory, as shown in Fig. 2. This allows more requests to fit into memory for batching—hence improving the throughput. Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests. In Fig. 7, we show an example of vLLM managing the memory for two sequences. The logical blocks of the two sequences are mapped to different physical blocks within the space reserved by the block engine in GPU workers. The neighboring logical blocks of both sequences do not need to be contiguous in physical GPU memory and the space of physical blocks can be effectively utilized by both sequences. 

# 4.4 Application to Other Decoding Scenarios 

$\S4.3$ shows how Paged Attention and vLLM handle basic de-coding algorithms, such as greedy decoding and sampling, that take one user prompt as input and generate a single output sequence. In many successful LLM applications [18 , 34], an LLM service must offer more complex decoding scenarios that exhibit complex accessing patterns and more opportunities for memory sharing. We show the general applicability of vLLM on them in this section. 

Parallel sampling. In LLM-based program assistants [6 , 18], an LLM generates multiple sampled outputs for a single input prompt; users can choose a favorite output from various candidates. So far we have implicitly assumed that a request 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/a3f1a34985ef7a6e20dd0ef8bd8cb4283b9dd5ecd855a6254c1c2256958d641b.jpg) 
Figure 8. Parallel sampling example. 

generates a single sequence. In the remainder of this paper, we assume the more general case in which a request generates multiple sequences. In parallel sampling, one request includes multiple samples sharing the same input prompt, allowing the KV cache of the prompt to be shared as well. Via its Paged Attention and paged memory management, vLLM can realize this sharing easily and save memory. 

Fig. 8 shows an example of parallel decoding for two outputs. Since both outputs share the same prompt, we only reserve space for one copy of the prompt’s state at the prompt phase; the logical blocks for the prompts of both sequences are mapped to the same physical blocks: the logical block 0 and 1 of both sequences are mapped to physical blocks 7 and 1, respectively. Since a single physical block can be mapped to multiple logical blocks, we introduce a reference count for each physical block. In this case, the reference counts for physical blocks 7 and 1 are both 2. At the generation phase, the two outputs sample different output tokens and need separate storage for KV cache. vLLM implements a copy-on- write mechanism at the block granularity for the physical blocks that need modification by multiple sequences, similar to the copy-on-write technique in OS virtual memory (e.g., when forking a process). Specifically, in Fig. 8, when sample A1 needs to write to its last logical block (logical block 1), vLLM recognizes that the reference count of the corresponding physical block (physical block 1) is greater than 1; it allocates a new physical block (physical block 3), instructs the block engine to copy the information from physical block 1, and decreases the reference count to 1. Next, when sample A2 writes to physical block 1, the reference count is already reduced to 1; thus A2 directly writes its newly generated KV cache to physical block 1. 

In summary, vLLM enables the sharing of most of thespace used to store the prompts’ KV cache across multiple output samples, with the exception of the final logical block, which is managed by a copy-on-write mechanism. By sharing physical blocks across multiple samples, memory usage can be greatly reduced, especially for long input prompts . 

Beam search. In LLM tasks like machine translation [59], the users expect the top $\cdot k$ most appropriate translations output by the LLM. Beam search [49] is widely used to decode the most probable output sequence from an LLM, as it mitigates the computational complexity of fully traversing the 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/8ad17cb16de4134df2474fe54124dbc0cdc5063f0159eabfaf748ad6a0ec6430.jpg) 
Figure 9. Beam search example. 

sample space. The algorithm relies on the beam width parameter $k$ , which determines the number of top candidates retained at every step. During decoding, beam search expands each candidate sequence in the beam by considering all possible tokens, computes their respective probabilities using the LLM, and retains the top $\cdot k$ most probable sequences out of $k\cdot|V|$ candidates, where $|V|$ is the vocabulary size. 

Unlike parallel decoding, beam search facilities sharing not only the initial prompt blocks but also other blocks across different candidates, and the sharing patterns dynamically change as the decoding process advances, similar to the process tree in the OS created by compound forks. Fig. 9 shows how vLLM manages the KV blocks for a beam search ex-ample with $k\,=\,4$ . Prior to the iteration illustrated as the dotted line, each candidate sequence has used 4 full logical blocks. All beam candidates share the first block 0 (i.e., prompt). Candidate 3 digresses from others from the second block. Candidates 0-2 share the first 3 blocks and diverge at the fourth block. At subsequent iterations, the top-4 probable candidates all originate from candidates 1 and 2. As the original candidates 0 and 3 are no longer among the top candidates, their logical blocks are freed, and the reference counts of corresponding physical blocks are reduced. vLLM frees all physical blocks whose reference counts reach 0 (blocks 2, 4, 5, 8). Then, vLLM allocates new physical blocks (blocks 9-12) to store the new KV cache from the new candidates. Now, all candidates share blocks 0, 1, 3; candidates 0 and 1 share block 6, and candidates 2 and 3 further share block 7. 

Previous LLM serving systems require frequent memory copies of the KV cache across the beam candidates. For example, in the case shown in Fig. 9, after the dotted line, candidate 3 would need to copy a large portion of candidate 2’s KV cache to continue generation. This frequent memory copy overhead is significantly reduced by vLLM’s physical block sharing. In vLLM, most blocks of different beam candidates can be shared. The copy-on-write mechanism is applied only when the newly generated tokens are within an old shared block, as in parallel decoding. This involves only copying one block of data. 

Shared prefix. Commonly, the LLM user provides a (long) description of the task including instructions and example inputs and outputs, also known as system prompt [36]. The description is concatenated with the actual task input to form the prompt of the request. The LLM generates outputs based 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/cf094f79f6740f639d96c78fb0fc6ab04c2eba196f592e711d06af94bc4c3b06.jpg) 
Figure 10. Shared prompt example for machine translation. The examples are adopted from [5]. 

on the full prompt. Fig. 10 shows an example. Moreover, the shared prefix can be further tuned, via prompt engineering, to improve the accuracy of the downstream tasks [26, 27]. 

For this type of application, many user prompts share a prefix, thus the LLM service provider can store the KV cache of the prefix in advance to reduce the redundant computation spent on the prefix. In vLLM, this can be conveniently achieved by reserving a set of physical blocks for a set of predefined shared prefixes by the LLM service provider, as how OS handles shared library across processes. A user input prompt with the shared prefix can simply map its logical blocks to the cached physical blocks (with the last block marked copy-on-write). The prompt phase computation only needs to execute on the user’s task input. 

Mixed decoding methods. The decoding methods discussed earlier exhibit diverse memory sharing and accessing patterns. Nonetheless, vLLM facilitates the simultaneous processing of requests with different decoding preferences, which existing systems cannot efficiently do. This is because vLLM conceals the complex memory sharing between different sequences via a common mapping layer that translates logical blocks to physical blocks. The LLM and its execution kernel only see a list of physical block IDs for each sequence and do not need to handle sharing patterns across sequences. Compared to existing systems, this approach broadens the batching opportunities for requests with different sampling requirements, ultimately increasing the system’s overall throughput. 

# 4.5 Scheduling and Preemption 

When the request traffic surpasses the system’s capacity, vLLM must prioritize a subset of requests. In vLLM, we adoptthe first-come-first-serve (FCFS) scheduling policy for all requests, ensuring fairness and preventing starvation. When vLLM needs to preempt requests, it ensures that the earliest arrived requests are served first and the latest requests are preempted first. 

LLM services face a unique challenge: the input prompts for an LLM can vary significantly in length, and the resulting output lengths are not known a priori, contingent on both the input prompt and the model. As the number of requests and their outputs grow, vLLM can run out of the GPU’s physical blocks to store the newly generated KV cache. There are two classic questions that vLLM needs to answer in this context: (1) Which blocks should it evict? (2) How to recover evicted blocks if needed again? Typically, eviction policies use heuristics to predict which block will be accessed furthest in the future and evict that block. Since in our case we know that all blocks of a sequence are accessed together, we implement an all-or-nothing eviction policy, i.e., either evict all or none of the blocks of a sequence. Furthermore, multiple sequences within one request (e.g., beam candidates in one beam search request) are gang-scheduled as a sequence group . The sequences within one sequence group are always preempted or rescheduled together due to potential memory sharing across those sequences. To answer the second question of how to recover an evicted block, we consider two techniques: 

Swapping. This is the classic technique used by most virtual memory implementations which copy the evicted pages to a swap space on the disk. In our case, we copy evicted blocks to the CPU memory. As shown in Fig. 4, besides the GPU block allocator, vLLM includes a CPU block allocator to manage the physical blocks swapped to CPU RAM. When vLLMexhausts free physical blocks for new tokens, it selects a set of sequences to evict and transfer their KV cache to the CPU. Once it preempts a sequence and evicts its blocks, vLLM stops accepting new requests until all preempted sequences are completed. Once a request completes, its blocks are freed from memory, and the blocks of a preempted sequence are brought back in to continue the processing of that sequence. Note that with this design, the number of blocks swapped to the CPU RAM never exceeds the number of total physical blocks in the GPU RAM, so the swap space on the CPU RAM is bounded by the GPU memory allocated for the KV cache. 

Re computation. In this case, we simply recompute the KV cache when the preempted sequences are rescheduled. Note that re computation latency can be significantly lower than the original latency, as the tokens generated at decoding can be concatenated with the original user prompt as a new prompt—their KV cache at all positions can be generated in one prompt phase iteration. 

The performances of swapping and re computation depend on the bandwidth between CPU RAM and GPU memory and the computation power of the GPU. We examine the speeds of swapping and re computation in $\S7.3$ . 

# 4.6 Distributed Execution 

Many LLMs have parameter sizes exceeding the capacity of a single GPU [5 , 9]. Therefore, it is necessary to partition them across distributed GPUs and execute them in a model parallel fashion [28 , 63]. This calls for a memory manager capable of handling distributed memory. vLLM is effective in distributed settings by supporting the widely used Megatron-LM style tensor model parallelism strategy on Transformers [47]. This strategy adheres to an SPMD (Single Program Multiple Data) execution schedule, wherein the linear layers are partitioned 
Table 1. Model sizes and server configurations. 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/f8023cf1d1db4dc398dcabf465e05515db7e888520173d79e46b4d2213b6fe87.jpg) 

to perform block-wise matrix multiplication, and the the GPUs constantly synchronize intermediate results via an allreduce operation. Specifically, the attention operator is split on the attention head dimension, each SPMD process takes care of a subset of attention heads in multi-head attention. 

We observe that even with model parallel execution, each model shard still processes the same set of input tokens, thus requiring the KV Cache for the same positions. Therefore, vLLM features a single KV cache manager within the centralized scheduler, as in Fig. 4. Different GPU workers share the manager, as well as the mapping from logical blocks to physical blocks. This common mapping allows GPU workers to execute the model with the physical blocks provided by the scheduler for each input request. Although each GPU worker has the same physical block IDs, a worker only stores a portion of the KV cache for its corresponding attention heads. 

In each step, the scheduler first prepares the message with input token IDs for each request in the batch, as well as the block table for each request. Next, the scheduler broadcasts this control message to the GPU workers. Then, the GPU workers start to execute the model with the input token IDs. In the attention layers, the GPU workers read the KV cache according to the block table in the control message. During execution, the GPU workers synchronize the intermediate results with the all-reduce communication primitive without the coordination of the scheduler, as in [47]. In the end, the GPU workers send the sampled tokens of this iteration back to the scheduler. In summary, GPU workers do not need to synchronize on memory management as they only need to receive all the memory management information at the beginning of each decoding iteration along with the step inputs. 

# 5 Implementation 

vLLM is an end-to-end serving system with a FastAPI [15] frontend and a GPU-based inference engine. The frontend extends the OpenAI API [34] interface, allowing users to customize sampling parameters for each request, such as the maximum sequence length and the beam width $k$ . The vLLM engine is written in 8.5K lines of Python and 2K lines of $\mathsf{C++/C U D A}$ code. We develop control-related components including the scheduler and the block manager in Python while developing custom CUDA kernels for key operations such as Paged Attention. For the model executor, we implement popular LLMs such as GPT [5], OPT [62], and LLaMA [52] using 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/1cc5db82835fb611fd1fad8f060a5ed147c8938b33aa68e9963184553ccb8e71.jpg) 
Figure 11. Input and output length distributions of the (a) ShareGPT and (b) Alpaca datasets. 

PyTorch [39] and Transformers [58]. We use NCCL [32] for tensor communication across the distributed GPU workers. 

# 5.1 Kernel-level Optimization 

Since Paged Attention introduces memory access patterns that are not efficiently supported by existing systems, we develop several GPU kernels for optimizing it. (1) Fused reshape and block write. In every Transformer layer, the new KV cache are split into blocks, reshaped to a memory layout optimized for block read, then saved at positions specified by the block table. To minimize kernel launch overheads, we fuse them into a single kernel. (2) Fusing block read and attention. We adapt the attention kernel in Faster Transformer [31] to read KV cache according to the block table and perform attention operations on the fly. To ensure coalesced memory access, we assign a GPU warp to read each block. Moreover, we add support for variable sequence lengths within a request batch. (3) Fused block copy. Block copy operations, issued by the copy-on-write mechanism, may operate on discontinuous blocks. This can lead to numerous invocations of small data movements if we use the cuda Mem cp yA sync API. To mitigate the overhead, we implement a kernel that batches the copy operations for different blocks into a single kernel launch. 

# 5.2 Supporting Various Decoding Algorithms 

vLLM implements various decoding algorithms using three key methods: fork , append , and free . The fork method creates a new sequence from an existing one. The append method appends a new token to the sequence. Finally, the free method deletes the sequence. For instance, in parallel sampling, vLLM creates multiple output sequences from the single input sequence using the fork method. It then adds new tokens to these sequences in every iteration with append , and deletes sequences that meet a stopping condition using free . The same strategy is also applied in beam search and prefix sharing by vLLM. We believe future decoding algorithms can also be supported by combining these methods. 

# 6 Evaluation 

In this section, we evaluate the performance of vLLM under a variety of workloads. 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/dcd575301f6153ab61f2786ec751d6a832d264a53bdba63ae0175764ca5c3504.jpg) 
Figure 12. Single sequence generation with OPT models on the ShareGPT and Alpaca dataset 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/0a7a18292c1d7f0fb5305cd330e33015c181e740980fd79909efec794605e261.jpg) 
Figure 13. Average number of batched requests when serving OPT-13B for the ShareGPT (2 reqs/s) and Alpaca (30 reqs/s) traces. 

# 6.1 Experimental Setup 

Model and server configurations. We use OPT [62] models with 13B, 66B, and 175B parameters and LLaMA [52] with 13B parameters for our evaluation. 13B and 66B are popular sizes for LLMs as shown in an LLM leader board [38], while 175B is the size of the famous GPT-3 [5] model. For all of our experiments, we use A2 instances with NVIDIA A100 GPUs on Google Cloud Platform. The detailed model sizes and server configurations are shown in Table 1. 

Workloads. We synthesize workloads based on ShareGPT [51] and Alpaca [50] datasets, which contain input and output texts of real LLM services. The ShareGPT dataset is a collection of user-shared conversations with ChatGPT [35]. The Alpaca dataset is an instruction dataset generated by GPT3.5 with self-instruct [57]. We tokenize the datasets and use their input and output lengths to synthesize client requests. As shown in Fig. 11, the ShareGPT dataset has $8.4\times$ longer input prompts and $5.8\times$ longer outputs on average than the Alpaca dataset, with higher variance. Since these datasets do not include timestamps, we generate request arrival times using Poisson distribution with different request rates. 

Baseline 1: Faster Transformer. Faster Transformer [31] is a distributed inference engine highly optimized for latency. 

As Faster Transformer does not have its own scheduler, we implement a custom scheduler with a dynamic batching mechanism similar to the existing serving systems such as Triton [30]. Specifically, we set a maximum batch size $B$ as large as possible for each experiment, according to the GPU memory capacity. The scheduler takes up to $B$ number of earliest arrived requests and sends the batch to Faster Transformer for processing. 

Baseline 2: Orca. Orca [60] is a state-of-the-art LLM serving system optimized for throughput. Since Orca is not publicly available for use, we implement our own version of Orca. We assume Orca uses the buddy allocation algorithm to determine the memory address to store KV cache. We implement three versions of Orca based on how much it over-reserves the space for request outputs: 

• Orca (Oracle). We assume the system has the knowledge of the lengths of the outputs that will be actually generated for the requests. This shows the upper-bound performance of Orca, which is infeasible to achieve in practice.

 • Orca (Pow2). We assume th ystem over-reserves the space for outputs by at most 2 $2\times$ × . For example, if the true output length is 25, it reserves 32 positions for outputs.

 • Orca (Max). We assume the system always reserves the space up to the maximum sequence length of the model, i.e., 2048 tokens. 

Key metrics. We focus on serving throughput. Specifically, using the workloads with different request rates, we measure normalized latency of the systems, the mean of every request’s end-to-end latency divided by its output length, as in Orca [60]. A high-throughput serving system should retain low normalized latency against high request rates. For most experiments, we evaluate the systems with 1-hour traces. As an exception, we use 15-minute traces for the OPT-175B model due to the cost limit. 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/268cd8463c639841dccf21d35b13e9598485c1306cd0602ab1c30ba007eb079d.jpg) 
Figure 14. Parallel generation and beam search with OPT-13B on the Alpaca dataset. 

# 6.2 Basic Sampling 

We evaluate the performance of vLLM with basic sampling (one sample per request) on three models and two datasets. The first row of Fig. 12 shows the results on the ShareGPT dataset. The curves illustrate that as the request rate increases, the latency initially increases at a gradual pace but then suddenly explodes. This can be attributed to the fact that when the request rate surpasses the capacity of the serving system, the queue length continues to grow infinitely and so does the latency of the requests. 

On the ShareGPT dataset, vLLM can sustain $1.7\times-2.7\times$ higher request rates compared to Orca (Oracle) and 2 $2.7\times-8\times$ × × compared to Orca (Max), while maintaining similar latencies. This is because vLLM’s Paged Attention can efficiently manage the memory usage and thus enable batching more requests than Orca. For example, as shown in Fig. 13a, for OPT-13B vLLM processes $2.2\times$ more requests at the same time than Orca (Oracle) and $4.3\times$ more requests than Orca (Max). Compared to Faster Transformer, vLLM can sustain upto $22\times$ higher request rates, as Faster Transformer does not utilize a fine-grained scheduling mechanism and inefficiently manages the memory like Orca (Max). 

The second row of Fig. 12 and Fig. 13b shows the results on the Alpaca dataset, which follows a similar trend to the ShareGPT dataset. One exception is Fig. 12 (f), where vLLM’s advantage over Orca (Oracle) and Orca (Pow2) is less pronounced. This is because the model and server configuration for OPT-175B (Table 1) allows for large GPU memory space available to store KV cache, while the Alpaca dataset has short sequences. In this setup, Orca (Oracle) and Orca (Pow2) can also batch a large number of requests despite the inefficiencies in their memory management. As a result, the performance of the systems becomes compute-bound rather than memory-bound. 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/fc05050fdc3ed50ec30e43a0048218135b6d71c35710dbf6d01451bcb23c291d.jpg) 
Figure 15. Average amount of memory saving from sharing KV blocks, when serving OPT-13B for the Alpaca trace. 

# 6.3 Parallel Sampling and Beam Search 

We evaluate the effectiveness of memory sharing in PagedAttention with two popular sampling methods: parallel sampling and beam search. In parallel sampling, all parallel sequences in a request can share the KV cache for the prompt. As shown in the first row of Fig. 14, with a larger number of sequences to sample, vLLM brings more improve-ment over the Orca baselines. Similarly, the second row of Fig. 14 shows the results for beam search with different beam widths. Since beam search allows for more sharing, vLLM demonstrates even greater performance benefits. The improvement of vLLM over Orca (Oracle) on OPT-13B and the Alpaca dataset goes from $1.3\times$ in basic sampling to $2.3\times$ in beam search with a width of 6. 

Fig. 15 plots the amount of memory saving, computed by the number of blocks we saved by sharing divided by the number of total blocks without sharing. We show $6.1\%-9.8\%$ memory saving on parallel sampling and $37.6\%\textrm{-}55.2\%$ on beam search. In the same experiments with the ShareGPT dataset, we saw $16.2\%\textrm{-}30.5\%$ memory saving on parallel sampling and $44.3\%\textrm{-}66.3\%$ on beam search. 

# 6.4 Shared prefix 

We explore the effectiveness of vLLM for the case a prefix is shared among different input prompts, as illustrated in 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/3bd361f3426674089b4e07911ac87d9efcd834cdc3892d06e59246b4a662b301.jpg) 
(a) 1-shot prefix prompt (b) 5-shot prefix prompt 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/d330cc6d64ac87fed639c9a9352e7b2890d9ade9d2c05887c3042f5a91524e3e.jpg) 
Figure 17. Performance on chatbot workload. 

Fig. 10. For the model, we use LLaMA-13B [52], which is multilingual. For the workload, we use the WMT16 [4] Englishto-German translation dataset and synthesize two prefixes that include an instruction and a few translation examples. The first prefix includes a single example (i.e., one-shot) while the other prefix includes 5 examples (i.e., few-shot). As shown in Fig. 16 (a), vLLM achieves $1.67\times$ higher throughput than Orca (Oracle) when the one-shot prefix is shared. Furthermore, when more examples are shared (Fig. 16 (b)), vLLM achieves $3.58\times$ higher throughput than Orca (Oracle). 

# 6.5 Chatbot 

A chatbot [8 , 19 , 35] is one of the most important applications of LLMs. To implement a chatbot, we let the model generate a response by concatenating the chatting history and the last user query into a prompt. We synthesize the chatting history and user query using the ShareGPT dataset. Due to the limited context length of the OPT-13B model, we cut the prompt to the last 1024 tokens and let the model generate at most 1024 tokens. We do not store the KV cache between different conversation rounds as doing this would occupy the space for other requests between the conversation rounds. 

Fig. 17 shows that vLLM can sustain $2\times$ higher request rates compared to the three Orca baselines. Since the ShareGPT dataset contains many long conversations, the input prompts for most requests have 1024 tokens. Due to the buddy allocation algorithm, the Orca baselines reserve the space for 1024 tokens for the request outputs, regardless of how they predict the output lengths. For this reason, the three Orca baselines behave similarly. In contrast, vLLM can effectively 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/b27b296e41d5900dd9eba28f8b2dd0065f6d4c90b5820d60ffdbe9522a21c8b9.jpg) 
(a) Latency of attention kernels. (b) End-to-end latency with different block sizes. Figure 18. Ablation experiments. 

handle the long prompts, as Paged Attention resolves the problem of memory fragmentation and reservation. 

# 7 Ablation Studies 

In this section, we study various aspects of vLLM and evaluate the design choices we make with ablation experiments. 

# 7.1 Kernel Micro benchmark 

The dynamic block mapping in Paged Attention affects the performance of the GPU operations involving the stored KV cache, i.e., block read/writes and attention. Compared to the existing systems, our GPU kernels (§5) involve extra overheads of accessing the block table, executing extra branches, and handling variable sequence lengths. As shown in Fig. 18a, this leads to $20{-}26\%$ higher attention kernel latency, compared to the highly-optimized Faster Transformer implementation. We believe the overhead is small as it only affects the attention operator but not the other operators in the model, such as Linear. Despite the overhead, Paged Attention makes vLLM significantly outperform Faster Transformer in end-to-end performance (§6). 

# 7.2 Impact of Block Size 

The choice of block size can have a substantial impact on the performance of vLLM. If the block size is too small, vLLMmay not fully utilize the GPU’s parallelism for reading and processing KV cache. If the block size is too large, internal fragmentation increases and the probability of sharing decreases. 

In Fig. 18b, we evaluate the performance of vLLM with dif-ferent block sizes, using the ShareGPT and Alpaca traces with basic sampling under fixed request rates. In the ShareGPT trace, block sizes from 16 to 128 lead to the best performance. In the Alpaca trace, while the block size 16 and 32 work well, larger block sizes significantly degrade the performance since the sequences become shorter than the block sizes. In practice, we find that the block size 16 is large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads. Accordingly, vLLM sets its default block size as 16. 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/faab2f327a9bb710f4e814057401abe32472d4633d1d426af644c3148b818cf5.jpg) 
Figure 19. (a) Overhead of re computation and swapping for different block sizes. (b) Performance when serving OPT-13B with the ShareGPT traces at the same request rate. 

# 7.3 Comparing Re computation and Swapping 

vLLM supports both re computation and swapping as its recovery mechanisms. To understand the tradeoffs between the two methods, we evaluate their end-to-end performance and micro benchmark their overheads, as presented in Fig. 19. Our results reveal that swapping incurs excessive overhead with small block sizes. This is because small block sizes often result in numerous small data transfers between CPU and GPU, which limits the effective PCIe bandwidth. In contrast, the overhead of re computation remains constant across different block sizes, as re computation does not utilize the KV blocks. Thus, re computation is more efficient when the block size is small, while swapping is more efficient when the block size is large, though re computation overhead is never higher than $20\%$ of swapping’s latency. For medium block sizes from 16 to 64, the two methods exhibit comparable end-to-end performance. 

# 8 Discussion 

Applying the virtual memory and paging technique to other GPU workloads. The idea of virtual memory and paging is effective for managing the KV cache in LLM serving because the workload requires dynamic memory allocation (since the output length is not known a priori) and its performance is bound by the GPU memory capacity. However, this does not generally hold for every GPU workload. For example, in DNN training, the tensor shapes are typically static, and thus memory allocation can be optimized ahead of time. For another example, in serving DNNs that are not LLMs, an increase in memory efficiency may not result in any performance improvement since the performance is primarily compute-bound. In such scenarios, introducing the vLLM’s techniques may rather degrade the performance due to the extra overhead of memory in direction and non-contiguous block memory. However, we would be excited to see vLLM’s techniques being applied to other workloads with similar properties to LLM serving. 

LLM-specific optimization s in applying virtual memory and paging. vLLM re-interprets and augments the idea of virtual memory and paging by leveraging the applicationspecific semantics. One example is vLLM’s all-or-nothing swap-out policy, which exploits the fact that processing a request requires all of its corresponding token states to be stored in GPU memory. Another example is the recomputation method to recover the evicted blocks, which is not feasible in OS. Besides, vLLM mitigates the overhead of memory in direction in paging by fusing the GPU kernels for memory access operations with those for other operations such as attention. 

# 9 Related Work 

General model serving systems. Model serving has been an active area of research in recent years, with numerous systems proposed to tackle diverse aspects of deep learning model deployment. Clipper [11], TensorFlow Serving [33], Nexus [45], InferLine [10], and Clockwork [20] are some earlier general model serving systems. They study batching, caching, placement, and scheduling for serving single or multiple models. More recently, DVABatch [12] introduces multi-entry multi-exit batching. REEF [21] and Shepherd [61] propose preemption for serving. AlpaServe [28] utilizes model parallelism for statistical multiplexing. However, these general systems fail to take into account the autoregressive property and token state of LLM inference, resulting in missed opportunities for optimization. 

Specialized serving systems for transformers. Due to the significance of the transformer architecture, numerous specialized serving systems for it have been developed. These systems utilize GPU kernel optimization s [1, 29, 31, 56], advanced batching mechanisms [14 , 60], model parallelism [1 , 41 , 60], and parameter sharing [64] for efficient serving. Among them, Orca [60] is most relevant to our approach. 

Comparison to Orca. The iteration-level scheduling in Orca [60] and Paged Attention in vLLM are complementary techniques: While both systems aim to increase the GPU utilization and hence the throughput of LLM serving, Orca achieves it by scheduling and interleaving the requests so that more requests can be processed in parallel, while vLLM is doing so by increasing memory utilization so that the working sets of more requests fit into memory. By reducing memory fragmentation and enabling sharing, vLLM runs more requests in a batch in parallel and achieves a $2–4\times$ speedup compared to Orca. Indeed, the fine-grained scheduling and interleaving of the requests like in Orca makes memory management more challenging, making the techniques proposed in vLLM even more crucial. 

Memory optimization s. The widening gap between the compute capability and memory capacity of accelerators has caused memory to become a bottleneck for both training and inference. Swapping [23 , 42 , 55], re computation [7 , 24] and their combination [40] have been utilized to reduce the peak memory of training. Notably, FlexGen [46] studies how to swap weights and token states for LLM inference with limited GPU memory, but it does not target the online serving settings. OLLA [48] optimizes the lifetime and location of tensors to reduce fragmentation, but it does not do finegrained block-level management or online serving. FlashAttention [13] applies tiling and kernel optimization s to reduce the peak memory of attention computation and reduce I/O costs. This paper introduces a new idea of block-level memory management in the context of online serving. 
# 10 Conclusion 

This paper proposes Paged Attention, a new attention algorithm that allows attention keys and values to be stored in non-contiguous paged memory, and presents vLLM, a high-throughput LLM serving system with efficient memory management enabled by Paged Attention. Inspired by operating systems, we demonstrate how established techniques, such as virtual memory and copy-on-write, can be adapted to efficiently manage KV cache and handle various decoding algorithms in LLM serving. Our experiments show that vLLM achieves $2–4\times$ throughput improvements over the state-of-the-art systems. 

# Acknowledgement 

We would like to thank Xiaoxuan Liu, Zhifeng Chen, Yanping Huang, anonymous SOSP reviewers, and our shepherd, Lidong Zhou, for their insightful feedback. This research is partly supported by gifts from Andreessen Horowitz, Anyscale, Astronomer, Google, IBM, Intel, Lacework, Microsoft, Mo- hamed Bin Zayed University of Artificial Intelligence, Samsung SDS, Uber, and VMware. 

# References 

[1] Reza Yazdani Aminabadi, Samyam Raj bh and ari, Minjia Zhang, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Jeff Rasley, Shaden Smith, Olatunji Ruwase, et al . 2022. DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale. arXiv preprint arXiv:2207.00032 (2022).

 [2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer normalization. arXiv preprint arXiv:1607.06450 (2016).

 [3] Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. 2000. A neural probabilistic language model. Advances in neural information processing systems 13 (2000).

 [4] Ond rej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Varvara Logacheva, Christof Monz, Matteo Negri, Aurelie Neveol, Mariana Neves, Martin Popel, Matt Post, Raphael Rubino, Car- olina Scarton, Lucia Specia, Marco Turchi, Karin Verspoor, and Marcos Zampieri. 2016. Findings of the 2016 Conference on Machine Translation. In Proceedings of the First Conference on Machine Translation . Association for Computational Linguistics, Berlin, Germany, 131–198. http://www.aclweb.org/anthology/W/W16/W16-2301

 [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neel a kant an, Pranav Shyam, Girish Sastry, Amanda Askell, et al . 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.

 [6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas 

Joseph, Greg Brockman, et al . 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 (2021).

 [7] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174 (2016).

 [8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An Open-Source Chatbot Impressing GPT-4 with $90\%^{*}$ ChatGPT Quality. https://lmsys. org/blog/2023-03-30-vicuna/

 [9] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al . 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 (2022).

 [10] Daniel Crankshaw, Gur-Eyal Sela, Xiangxi Mo, Corey Zumar, Ion Stoica, Joseph Gonzalez, and Alexey Tumanov. 2020. InferLine: latencyaware provisioning and scaling for prediction serving pipelines. In Proceedings of the 11th ACM Symposium on Cloud Computing . 477–491.

 [11] Daniel Crankshaw, Xin Wang, Guilio Zhou, Michael J Franklin, Joseph E Gonzalez, and Ion Stoica. 2017. Clipper: A Low-Latency Online Prediction Serving System. In 14th USENIX Symposium on Networked Systems Design and Implementation (NSDI 17) . 613–627.

 [12] Weihao Cui, Han Zhao, Quan Chen, Hao Wei, Zirui Li, Deze Zeng, Chao Li, and Minyi Guo. 2022. DVABatch: Diversity-aware MultiEntry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs. In 2022 USENIX Annual Technical Conference (USENIX ATC 22) . 183–198.

 [13] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flash attention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems 35 (2022), 16344–16359.

 [14] Jiarui Fang, Yang Yu, Chengduo Zhao, and Jie Zhou. 2021. TurboTransformers: an efficient GPU serving system for transformer models. In Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming . 389–402.

 [15] FastAPI. 2023. FastAPI. https://github.com/tiangolo/fastapi .

 [16] Pin Gao, Lingfan Yu, Yongwei Wu, and Jinyang Li. 2018. Low latency rnn inference with cellular batching. In Proceedings of the Thirteenth EuroSys Conference . 1–15.

 [17] Amir Gholami, Zhewei Yao, Sehoon Kim, Michael W Mahoney, and Kurt Keutzer. 2021. Ai and memory wall. RiseLab Medium Post 1 (2021), 6.

 [18] Github. 2022. https://github.com/features/copilot

 [19] Google. 2023. https://bard.google.com/

 [20] Arpan Gujarati, Reza Karimi, Safya Alzayat, Wei Hao, Antoine Kaufmann, Ymir Vigfusson, and Jonathan Mace. 2020. Serving $\{\mathrm{DNS}\}$ like Clockwork: Performance Predictability from the Bottom Up. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20) . 443–462.

 [21] Mingcong Han, Hanze Zhang, Rong Chen, and Haibo Chen. 2022. Microsecond-scale Preemption for Concurrent GPU { accelerated }{ DNN } Inferences. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22) . 539–558.

 [22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition . 770–778.

 [23] Chien-Chin Huang, Gu Jin, and Jinyang Li. 2020. Swap advisor: Pushing deep learning beyond the gpu memory limit via smart swapping. In Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems . 1341–1355.

 [24] Paras Jain, Ajay Jain, Aniruddha Nrusimha, Amir Gholami, Pieter Abbeel, Joseph Gonzalez, Kurt Keutzer, and Ion Stoica. 2020. Checkmate: Breaking the memory wall with optimal tensor re materialization. Proceedings of Machine Learning and Systems 2 (2020), 497–511. 
[25] Tom Kilburn, David BG Edwards, Michael J Lanigan, and Frank H Sumner. 1962. One-level storage system. IRE Transactions on Electronic Computers 2 (1962), 223–235.

 [26] Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691 (2021).

 [27] Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190 (2021).

 [28] Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E Gonzalez, et al . 2023. AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving. arXiv preprint arXiv:2302.11665 (2023).

 [29] Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, and Lidong Zhou. 2020. Rammer: Enabling holistic deep learning compiler optimization s with rtasks. In Proceedings of the 14th USENIX Conference on Operating Systems Design and Implementation . 881–897.

 [30] NVIDIA. [n. d.]. Triton Inference Server. https://developer.nvidia.com/ nvidia-triton-inference-server .

 [31] NVIDIA. 2023. Faster Transformer. https://github.com/NVIDIA/ Faster Transformer .

 [32] NVIDIA. 2023. NCCL: The NVIDIA Collective Communication Library. https://developer.nvidia.com/nccl .

 [33] Christopher Olston, Noah Fiedel, Kiril Gorovoy, Jeremiah Harmsen, Li Lao, Fangwei Li, Vinu Raja she khar, Sukriti Ramesh, and Jordan Soyke. 2017. Tensorflow-serving: Flexible, high-performance ml serving. arXiv preprint arXiv:1712.06139 (2017).

 [34] OpenAI. 2020. https://openai.com/blog/openai-api

 [35] OpenAI. 2022. https://openai.com/blog/chatgpt

 [36] OpenAI. 2023. https://openai.com/blog/custom-instructions-forchatgpt

 [37] OpenAI. 2023. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL]

 [38] LMSYS ORG. 2023. Chatbot Arena Leader board Week 8: Introducing MT-Bench and Vicuna-33B. https://lmsys.org/blog/2023-06-22- leader board/.

 [39] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al . 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32 (2019).

 [40] Shishir G Patil, Paras Jain, Prabal Dutta, Ion Stoica, and Joseph Gonzalez. 2022. POET: Training Neural Networks on Tiny Devices with Integrated Re materialization and Paging. In International Conference on Machine Learning . PMLR, 17573–17583.

 [41] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2022. Efficiently Scaling Transformer Inference. arXiv preprint arXiv:2211.05102 (2022).

 [42] Jie Ren, Samyam Raj bh and ari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. 2021. ZeRO-Offload: Democratizing Billion-Scale Model Training.. In USENIX Annual Technical Conference . 551–564.

 [43] Reuters. 2023. https://www.reuters.com/technology/tech-giants-ailike-bing-bard-poses-billion-dollar-search-problem-2023-02-22/

 [44] Amazon Web Services. 2023. https://aws.amazon.com/bedrock/

 [45] Haichen Shen, Lequn Chen, Yuchen Jin, Liangyu Zhao, Bingyu Kong, Matthai Philipose, Arvind Krishna mur thy, and Ravi Sundaram. 2019. Nexus: A GPU cluster engine for accelerating DNN-based video analysis. In Proceedings of the 27th ACM Symposium on Operating Systems Principles . 322–337.

 [46] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E Gonzalez, et al . 2023. High-throughput Generative Inference of Large Language Models with a Single GPU. arXiv preprint arXiv:2303.06865 (2023). 

[47] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multibillion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053 (2019).

 [48] Benoit Steiner, Mostafa Elhoushi, Jacob Kahn, and James Hegarty. 2022. OLLA: Optimizing the Lifetime and Location of Arrays to Reduce the Memory Usage of Neural Networks. (2022). https://doi.org/10.48550/ arXiv.2210.12924

 [49] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. 2014. Sequence to sequence learning with neural networks. Advances in neural information processing systems 27 (2014).

 [50] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An Instruction-following LLaMA model. https:// github.com/tatsu-lab/stanford alpaca .

 [51] ShareGPT Team. 2023. https://sharegpt.com/

 [52] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie- Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al . 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).

 [53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

 [54] Jing Wang, Youyou Lu, Qing Wang, Minhui Xie, Keji Huang, and Jiwu Shu. 2022. Pacman: An Efficient Compaction Approach for Log { Structured }{ Key-Value } Store on Persistent Memory. In 2022 USENIX Annual Technical Conference (USENIX ATC 22) . 773–788.

 [55] Linnan Wang, Jinmian Ye, Yiyang Zhao, Wei Wu, Ang Li, Shuai- wen Leon Song, Zenglin Xu, and Tim Kraska. 2018. Super neurons: Dynamic GPU memory management for training deep neural networks. In Proceedings of the 23rd ACM SIGPLAN symposium on principles and practice of parallel programming . 41–53.

 [56] Xiaohui Wang, Ying Xiong, Yang Wei, Mingxuan Wang, and Lei Li. 2021. LightSeq: A High Performance Inference Library for Transformers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Papers . 113–120.

 [57] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-Instruct: Aligning Language Model with Self Generated Instructions. arXiv preprint arXiv:2212.10560 (2022).

 [58] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al . 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations . 38–45.

 [59] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al . 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144 (2016).

 [60] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A Distributed Serving System for { Transformer-Based } Generative Models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22) . 521–538.

 [61] Hong Zhang, Yupeng Tang, Anurag Khandelwal, and Ion Stoica. 2023. SHEPHERD: Serving DNNs in the Wild. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23) . USENIX Association, Boston, MA, 787–808. https://www.usenix.org/conference/ nsdi23/presentation/zhang-hong 
[62] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al . 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022).

 [63] Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P Xing, et al . 2022. Alpa: Automating Inter-and Intra-Operator Parallelism for Distributed Deep Learning. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22) . 559–578. 

[64] Zhe Zhou, Xuechao Wei, Jiejing Zhang, and Guangyu Sun. 2022. PetS: A Unified Framework for Parameter-Efficient Transformers Serving. In 2022 USENIX Annual Technical Conference (USENIX ATC 22) . 489–504. 
