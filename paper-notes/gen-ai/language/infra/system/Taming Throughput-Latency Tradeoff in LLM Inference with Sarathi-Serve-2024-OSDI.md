# Abstract
Each LLM serving request goes through two phases. The first is prefill which processes the entire input prompt and produces the first output token and the second is decode which generates the rest of output tokens, one-at-a-time. 

Prefill iterations have high latency but saturate GPU compute due to parallel processing of the input prompt. In contrast, decode iterations have low latency but also low compute utilization because a decode iteration processes only a single token per request. This makes batching highly effective for decodes and consequently for overall throughput. However, batching multiple requests leads to an interleaving of prefill and decode iterations which makes it challenging to achieve both high throughput and low latency.
>  prefill 迭代的延迟高，但能饱和 GPU 计算
>  decode 迭代的延迟低，但计算利用率低 (每个请求仅处理单 token)
>  decode 计算利用率低，因此批处理可以提高计算效率，进而提高吞吐
>  但请求不仅仅只有 decode 阶段，如果将 prefill, decode 组为 batch，会导致 decode 延迟增加，因此这是吞吐和延迟之间的 trade-off

We introduce an efficient LLM inference scheduler, Sarathi-Serve, to address this throughput-latency tradeoff. 
>  Sarathi-Serve 意在解决这一 throughput-latency tradeoff:

Sarathi-Serve introduces chunked-prefills which splits a prefill request into near equal sized chunks and creates stall-free schedules that adds new requests in a batch without pausing ongoing decodes. Stall-free scheduling unlocks the opportunity to improve throughput with large batch sizes while minimizing the effect of batching on latency. 
>  1. chuned-prefills 将一个 prefill 请求分离为多个相同大小的 chunks，并且在不暂停正在执行的 decodes 的情况下为 batch 添加新请求，实现无停顿调度，无停顿调度意在最小化批处理导致的延迟增加

Furthermore, uniform batches in Sarathi-Serve ameliorate the imbalance between iterations, resulting in minimal pipeline bubbles.
>  2. uniform batches 缓解迭代之间的不平衡，最小化流水线气泡

Our techniques yield significant improvements in inference performance across models and hardware under tail latency constraints. For Mistral-7B on single A100 GPUs, we achieve  $2.6\times$  higher serving capacity and up to  $3.7\times$  higher serving capacity for the Yi-34B model on two A100 GPUs as compared to vLLM. When used with pipeline parallelism on Falcon-180B, Sarathi-Serve provides up to  $5.6\times$  gain in the end-to-end serving capacity. The source code for Sarathi-Serve is available at https://github.com/microsoft/sarathi-serve.
>  这些技术显著提高了推理性能，单个 A100 上，Mistral-7B 的服务容量为 2.6x, 两个 A100 上，Yi-34B 的服务容量为 3.7x
>  对 Falcon-180B 使用流水线并行，服务容量为 5.6x

>  服务容量通常和吞吐同义，但是一般也隐含了延迟要求，因此理解为 goodput 更好

>  尾部延迟指系统中最慢的那部分请求的响应时间，通常使用百分数，例如 P90 表示 90% 的请求的延迟都小于这个值
>  它关注的是最差体验，即少数极端慢的请求 (例如 P99.9 = 1000ms)

# 1 Introduction
Large language models (LLMs) [34,35,52,57,71] have shown impressive abilities in a wide variety of tasks spanning natural language processing, question answering, code generation, etc. This has led to tremendous increase in their usage across many applications such as chatbots [2, 5, 6, 57], search [4, 9, 11, 19, 25], code assistants [1, 8, 20], etc. The significant GPU compute required for running inference on large models, coupled with significant increase in their usage, has made LLM inference a dominant GPU workload today. Thus, optimizing LLM inference has been a key focus for many recent systems [29, 53, 58, 59, 63, 75, 77].

![[pics/Sarathi-Serve-Fig1.png]]

Optimizing throughput and latency are both important objectives in LLM inference since the former helps keep serving costs tractable while the latter is necessary to meet application requirements. In this paper, we show that current LLM serving systems have to face a tradeoff between throughput and latency. In particular, LLM inference throughput can be increased significantly with batching. However, the way existing systems batch multiple requests leads to a compromise on either throughput or latency. For example, Figure 1b shows that increasing load can significantly increase tail latency in a state-of-the-art LLM serving system vLLM [53].
>  我们先说明目前的 LLM 推理系统需要面临 throughput 和 latency 的 tradeoff
>  具体地说，通过 batching 可以提高推理吞吐，但是现存系统的 batching 方式导致了这一 tradeoff
>  Fig1b 说明了在 vLLMs 上提高 load 会显著提高尾部延迟

Each LLM inference request goes through two phases -a prefill phase followed by a decode phase. The prefill phase corresponds to the processing of the input prompt and the decode phase corresponds to the autoregressive token generation. The prefill phase is compute-bound because it processes all tokens of an input prompt in parallel whereas the decode phase is memory-bound because it processes only one token per-request at a time. Therefore, decodes benefit significantly from batching because larger batches can use GPUs more efficiently whereas prefills do not benefit from batching.
>  prefill 为 compute-bound，因为并行处理输入 prompt 中的所有 tokens
>  decode 为 memory-bound 因为 per-request 一次仅处理一个 token
>  因此批处理有益于 decode，对 prefill 则无太大意义

![[pics/Sarathi-Serve-Fig2.png]]

Current LLM inference schedulers can be broadly classified into two categories, namely, prefill-prioritizing and decode-prioritizing depending on how they schedule the prefill and decode phases while batching requests. In this paper, we argue that both strategies have fundamental pitfalls that make them unsuitable for serving online inference (see Figure 2).
>  目前的调度器可以分为两类: prefill 优先、decode 优先
>  二者都存在缺点，不适合 online inference 的场景

Traditional request-level batching systems such as FasterTransformer [7] employ decode-prioritizing scheduling. These systems submit a batch of requests to the execution engine that first computes the prefill phase of all requests and then schedules their decode phase. The batch completes only after all requests in it have finished their decode phase i.e., new prefills are not scheduled as long as one or more requests are doing decodes. This strategy optimizes inference for latency metric time-between-tokens or TBT - an important performance metric for LLMs. This is because new requests do not affect the execution of ongoing requests in their decode phase. However, decode-prioritizing schedulers severely compromise on throughput because even if some requests in a batch finish early, the execution continues with reduced batch size until the completion of the last request.
>  FasterTransformer 采用 request-level batching，一次提交的 requests batch 会在 perfill, decode 都全部完成之后才返回，故只要还存在 decode，新的 perfill 不会提交
>  该策略对于 time-between-tokens 友好，因为新请求不会影响正在 decoding 请求的执行
>  该策略对于 throughput 不友好，因为实际的 batch size 会随着部分 requests 的执行完成而减小，直到最后一个 request 执行完成

Orca [75] introduced iteration-level batching wherein requests can dynamically enter or exit a batch at the granularity of individual iterations. Iteration-level batching improves throughput by avoiding inefficiencies of request-level batching systems. Orca and several other recent systems like vLLM [23] combine iteration-level batching with prefill-prioritizing scheduling wherein they eagerly schedule the prefill phase of one or more requests first i.e., whenever GPU memory becomes available. This way, prefill-prioritizing schedulers have better throughput because computing prefills first allows subsequent decodes to operate at high batch sizes. 
>  ORCA 采用 iteration-level batching，请求可以动态进出 batch
>  vLLM 和 ORCA 将 iteration-level batching 和 prefill 优先的调度结合: 一旦 GPU 显存可用，就立即调度一个或多个请求的 prefill
>  prefill 有效调度利于 throughput，因为优先计算 prefill 允许后续 decode 以更大 batch size 运行

However, prioritizing prefills leads to high latency because it interferes with ongoing decodes. Since prefills can take arbitrarily long time depending on the lengths of the given prompts, prefill-prioritizing schedulers lead to an undesirable phenomenon that we refer to as generation stalls in this paper. For example, Figure 1a shows that a generation stall in vLLM can last over several seconds.
>  但优先 prefill 会导致高 latency，因为和进行中的 decode 互相干扰
>  prefill 花费的时间取决于给定 prompt 长度
>  我们称 prefill 有效调度导致的现象为 prefill stall

Another challenge introduced by traditional iteration-level scheduling systems like Orca [75] is pipeline stalls or bubbles [49]. These appear in pipeline-parallelism (PP) deployments that are needed to scale LLM inference across several nodes. In servers with high bandwidth connectivity such as NVIDIA DGX A100 [16], tensor-parallelism (TP) [64] can enable deployment of an LLM on up to 8 GPUs, supporting large batch sizes with low latencies. However, TP can have prohibitively high latencies when hyper-clusters are unavailable [33]. Thus, as an alternative to TP, pipeline-parallelism (PP) [33, 55] is typically used across commodity networks. Existing systems rely on micro-batches to mitigate pipeline stalls or bubbles [49]. However, the standard micro-batch based scheduling can still lead to pipeline bubbles due to the unique characteristics of LLM inference. Specifically, LLM inference consists of a mixture of varying length prefills and decodes. The resulting schedule can thus have wildly varying runtimes across different micro-batches that waste GPU cycles and degrade the overall system throughput.
>  iteration-level 调度的另一个问题是 pipeline stall/bubble
>  TP 的效率依赖于高性能互联，PP 通常作为 TP 在基于通用网络的部署环境的替代
>  PP 依赖于 microbatch 调度，而 microbatch 中 PD 的混合比例不同，就会导致 microbatch 各自的执行时间不统一，进而导致 bubble

To address these challenges, we propose Sarathi-Serve, a scheduler to balance the throughput-latency tradeoff for scalable online LLM inference serving. Sarathi-Serve is based on two key ideas: chunked-prefills and stall-free scheduling. Chunked-prefills splits a prefill request into equal compute-sized chunks and computes a prompt's prefill phase over multiple iterations (each with a subset of the prompt tokens). Stall-free scheduling allows new requests to join a running batch without pausing ongoing decodes. This involves constructing a batch by coalescing all the on-going decodes with one (or more) prefill chunks from new requests such that each batch reaches the pre-configured chunk size. 
>  Sarathi-Serve 是一个平衡 throughput-latency tradeoff 的调度器
>  他包括两个核心思想: chunked-prefills, stall-free scheduling
>  chunked-prefill 将 prefill 划分为多个计算量相等的 chunks，并在多个迭代计算 prefill (每次迭代计算一部分 prompt tokens)
>  stall-free scheduling 允许不延缓正在进行的 decode 同时将新的 request 加入 batch，实质上是通过合并所有正在进行的 decode + 一个或多个新请求的 prefill chunks 来构造 batch，让 batch 达到预配置的 chunk size

Sarathi-Serve builds upon iteration-level batching but with an important distinction: it throttles the number of prefill tokens in each iteration while admitting new requests in a running batch. This not only bounds the latency of each iteration, but also makes it nearly independent of the total length of input prompts. This way, Sarathi-Serve minimizes the effect of computing new prefills on the TBT of ongoing decodes enabling both high throughput and low TBT latency.
>  Sarathi 基于 iteration-level 构建，但限制了每个 iteration 的 prefill tokens 数量
>  这不仅可以限制每个 iteration 的延迟，也使得 iteration 的延迟几乎独立于 input prompts 的总长度
>  通过这种方法，Sarathi 最小化计算新 prefill 对进行中 decode 的 time-between-tokens 的影响，同时实现高吞吐和低 TBT 延迟

In addition, hybrid batches (consisting of prefill and decode tokens) constructed by Sarathi-Serve have a near-uniform compute requirement. With pipeline-parallelism, this allows us to create balanced micro-batching based schedules that significantly reduce pipeline bubbles and improve GPU utilization, thus allowing efficient and scalable deployments.
>  此外，Sarathi 构造的 PD 混合的 batches 会有接近均匀的计算要求，因此在 PP 下减少了 bubble

We evaluate Sarathi-Serve across different models and hardware Mistral-7B on a single A100, Yi-34B on 2 A100 GPUs with 2-way tensor parallelism, LLaMA2-70B on 8 A40 GPUs, and Falcon-180B with 2-way pipeline and 4-way tensor parallelism across 8 A100 GPUs connected over commodity ethernet. 
>  评估: 
>  Mistra-7B on 1 A100; 
>  Yi-34B on 2 A100 (2-way TP); 
>  LLaMA2-70B on 8 A40;
>  Falcon-180B on 8 A100 (2-way PP, 4-way TP，通过通用以太网互联)

For Yi-34B, Sarathi-Serve improves system serving capacity by up to  $3.7\times$  under different SLO targets. Similarly for Mistral-7B, we achieve up to  $2.6\times$  higher serving capacity. Sarathi-Serve also reduces pipeline bubbles, resulting in up to  $5.6\times$  gains in end-to-end serving capacity for Falcon-180B deployed with pipeline parallelism.

The main contributions of our paper include:

1. We identify a number of pitfalls in the current LLM serving systems, particularly in the context of navigating the throughput-latency tradeoff. 
2. We introduce two simple-yet-effective techniques, chunked-prefills and stall-free batching, to improve the performance of an LLM serving system. 
3. We show generality through extensive evaluation over multiple models, hardware, and parallelism strategies demonstrating that Sarathi-Serve improves model serving capacity by up to an order of magnitude.

>  贡献为:
>  1. 发现了目前 LLM 服务系统在 throughput-latency tradeoff 方面的缺陷
>  2. 引入了两个技术 chunked-prefill, stall-free batcing 以提高服务系统表现
>  3. 通过在多个模型、硬件、并行策略上的实验展示了 Sarathi 可以将模型服务容量提升一个数量级

# 2 Background
In this section, we describe the typical LLM model architecture along with their auto-regressive inference process. We also provide an overview of the scheduling policies and important performance metrics.

## 2.1 The Transformer Architecture
Popular large language models, like, GPT-3 [18], LLaMA [66], Yi [24] etc. are decoder-only transformer models trained on next token prediction tasks. These models consist of a stack of layers identical in structure. Each layer contains two modules - self-attention and feed-forward network (FFN).

**Self-attention module:** The self-attention module is central to the transformer architecture [67], enabling each part of a sequence to consider all previous parts for generating a contextual representation. During the computation of self-attention, first the Query  $(Q)$  ,Key  $(K)$  and Value  $(V)$  vectors corresponding to each input token are obtained via a linear transformation. Next, the attention operator computes a semantic relationship among all tokens of a sequence. This involves computing a dot-product of each  $Q$  vector with  $K$  vectors of all preceding tokens of the sequence, followed by a softmax operation to obtain a weight vector, which is then used to compute a weighted average of the  $V$  vectors. This attention computation can be performed across multiple heads, whose outputs are combined using a linear transformation.

**Feed-forward network (FFN):** FFN typically consists of two linear transformations with a non-linear activation in between. The first linear layer transforms an input token embedding of dimension  $h$  to a higher dimension  $h2$  . This is followed by an activation function, typically ReLU or GELU [27,46]. Finally, the second linear layer, transforms the token embedding back to the original dimension  $h$

## 2.2 LLM Inference Process
**Autoregressive decoding:** LLM inference consists of two distinct phases - a prefill phase followed by a decode phase. The prefill phase processes the user's input prompt and produces the first output token. Subsequently, the decode phase generates output tokens one at a time wherein the token generated in the previous step is passed through the model to generate the next token until a special end-of-sequence token is generated. Note that the decode phase requires access to all the keys and values associated with all the previously processed tokens to perform the attention operation. To avoid repeated recomputation, contemporary LLM inference systems store activations in KV-cache [7, 64, 75].

![[pics/Sarathi-Serve-Table1.png]]

A typical LLM prompt contains 100s-1000s of input tokens Table 2, [76]. During the prefill phase all these prompt tokens are processed in parallel in a single iteration. The parallel processing allows efficient utilization of GPU compute. On the contrary, the decode phase involves a full forward pass of the model over a single token generated in the previous iteration. This leads to low compute utilization making decodes memory-bound.
>  典型的 LLM prompt 包含数百到数千个输入 tokens
>  prefill 阶段，所有输入 tokens 会被并行处理，故允许高效利用 GPU 计算能力
>  decode 阶段，仅涉及对上一次生成的单个 token 的 forward pass，对 GPU 计算能力利用率低，故 memory-bound

**Batched LLM inference in multi-tenant environment:** A production serving system must deal with concurrent requests from multiple users. Naively processing requests in a sequential manner leads to a severe under-utilization of GPU compute. In order to achieve higher GPU utilization, LLM serving systems leverage batching to process multiple requests concurrently. This is particularly effective for the decode phase processing which has lower computational intensity at low batch sizes. Higher batch sizes allows the cost of fetching model parameters to be amortized across multiple requests.
>  生产级别的服务系统需要处理多用户的并发请求
>  为了提高 GPU 利用率，服务系统利用 batching 并发处理多请求
>  batching 对于 decode 高效，因为 decode 在 low batch size 下计算密度低，大 batch size 摊销了获取模型参数的开销

>  但是为每个 request 获取 kvcache 的开销无法摊销
>  一定的前缀复用可以缓解
>  不过根据 vLLM 文中的图，大头还是模型权重，kvcache 大约占模型权重的一半

Recently, several complementary techniques have been proposed to optimize throughput by enabling support for larger batch sizes. Kwon et al. propose PagedAttention [53], which allows more requests to concurrently execute, eliminating fragmentation in KV-cache. The use of Multi Query Attention (MQA) [61], Group Query Attention (GQA) [30] in leading edge LLM models like LLaMA2 [66], Falcon [31] and Yi [24] has also significantly helped in alleviating memory bottleneck in LLM inference. For instance, LLaMA2-70B model has a  $8\times$  smaller KV-cache footprint compared to LLaMA-65B.
>  paged attention 消除了 kvcache 碎片，允许更多请求并发执行
>  MQA, GQA 显著缓解了 LLM 推理的内存瓶颈，例如 LLaMA2-70B 的 kvcache 占用为 LLaMA65B 的 1/8

## 2.3 Multi-GPU LLM Inference
With ever-increasing growth in model sizes, it becomes necessary to scale LLMs to multi-GPU or even multi-node deployments [22, 59]. Furthermore, LLM inference throughput, specifically that of the decode phase is limited by the maximum batch size we can fit on a GPU. Inference efficiency can therefore benefit from model-parallelism which allows larger batch sizes by sharding model weights across multiple GPUs. Prior work has employed both tensor-parallelism (TP) [64] and pipeline-parallelism (PP) [7, 72, 75] for this purpose.
>  由于推理吞吐，尤其是 decode 阶段，受 GPU 能够容纳的最大 batch size 限制
>  因此模型增大后，MP 就是必要的，为 batch of requests 腾出内存

TP shards each layer across the participating GPUs by splitting the model weights and KV-cache equally across GPU workers. This way, TP can linearly scale per-GPU batch size. However, TP involves a high communication cost due to two all-reduce operations per layer - one in attention computation and the other in FFN [64]. Moreover, since these communication operations are in the critical path, TP is preferred only within a single node where GPUs are connected via high bandwidth interconnects like NVLink.
>  TP shard 模型权重和 kv cache (TP 会划分 attention heads，即每个设备一个 head)
>  TP 在每个 Transformer 涉及两次 all-reduce: 一次在 FFN，一次在 Attention，且 all-reduce 在关键路径上
>  因此 TP 仅适用于节点内高带宽场景

Compared to TP, PP splits a model layer-wise, where each GPU is responsible for a subset of layers. To keep all GPUs in the 'pipeline' busy, micro-batching is employed. These micro-batches move along the pipeline from one stage to the next at each itersation. PP has much better compute-communication ratio compared to TP, as it only needs to send activations once for multiple layers of compute. Furthermore, PP requires communication only via point-to-point communication operations, compared to the more expensive all-reduces in TP. Thu, PP is more efficient than TP when high-bandwidth interconnects are unavailable e.g., in cross-node deployments.
>  PP 的计算-通讯比率远高于 TP，因为 PP 中多层计算 (一个 stage) 仅对应一次激活通讯，并且仅需要点对点通讯
>  因此非高带宽互联场景下，例如跨节点，PP 更高效

## 2.4 Performance Metrics
There are two primary latency metrics of interest for LLM serving: TTFT (time-to-first-token) and TBT (time-between-tokens). For a given request, TTFT measures the latency of generating the first output token from the moment a request arrives in the system. This metric reflects the initial responsiveness of the model. TBT on the other hand measures the interval between the generation of consecutive output tokens of a request, and affects the overall perceived fluidity of the response. 
>  TTFT 衡量请求到达到第一个 token 生成的时间
>  TBT 衡量连续输出 tokens 之间的间隔时间

When system is under load, low throughput can lead to large scheduling delays and consequently higher TTFT.

In addition, we use a throughput metric, Capacity, defined as the maximum request load (queries-per-second) a system can sustain while meeting certain latency targets. Higher capacity is desirable because it reduces the cost of serving.
>  容量定义为系统满足特定延迟目标条件下承受的最大请求负载 (rps)

## 2.5 Scheduling Policies for LLM Inference
The scheduler is responsible for admission control and batching policy. For the ease of exposition, we investigate existing LLM inference schedulers by broadly classifying them under two categories - prefill-prioritizing and decode-prioritizing.
>  现存 LLM 推理调度器可以分为: prefill 优先和 decode 优先

![[pics/Sarathi-Serve-Alg1.png]]

>  和下面的调度算法不同，这里的 decode 会完整处理所有请求的 decoding，周才允许继续接新的 prefill，因此不存在被新请求的 prefill-stall 的问题
>  但 decode batch 中各个请求的 iteration 数不同，就会导致计算资源利用率低的问题，故对吞吐不友好，对 TBT 友好

Conventional inference engines like FasterTransformer [7], Triton Inference Server [17] use decode-prioritizing schedules with request-level batching i.e., they pick a batch of requests and execute it until all requests in the batch complete (Algorithm 1). This approach reduces the operational complexity of the scheduling framework but at the expense of inefficient resource utilization. Different requests in a batch typically have a large variation in the number of input and output tokens. Request-level schedulers pad shorter requests with zeros to match their length with the longest request in the batch which results in wasteful compute and longer wait times for pending requests [75].
>  FasterTransformer, Triton Inference 不完成 batch 内的所有请求不返回的方式属于 decode 优先调度
>  该方法资源利用率不高，对短 request 的 pad 浪费了计算资源，并且不完成所有请求不返回会导致新请求等待时间过长

![[pics/Sarathi-Serve-Alg2.png]]

>  这个调度算法中，每次都先看能不能收集到 prefill 请求，如果有，就调度一个只有 prefill 的 batch，然后把结束 prefill 的请求加入另一个待调度 batch
>  如果没有，才调度都是 decode 请求的待调度 batch

>  其实也是合理的，因为少量 prefill 请求就可以饱和 GPU 资源，而 decode 请求则需要大 batch 才不容易浪费计算资源
>  此外，prefill 的完成只需要一个 iteration, decode 的完成则无法确定
>  因此在这个调度算法中，batching 的 decode 请求每完成一个 iteration，就有可能需要等待新来的 request 的 prefill，这就是作者所说的 prefill stall
>  因此这个调度对吞吐友好，因为最大化了计算资源利用，但是对 TBT 极度不友好

To avoid wasted compute of request-level batching, Orca [75] introduced a fine-grained iteration-level batching mechanism where requests can dynamically enter and exit a batch after each model iteration.(Algorithm 2). This approach can significantly increase system throughput and is being used in many LLM inference serving systems today e.g., vLLM [23], TensorRT-LLM [21], and LightLLM [12].
>  iteration-level 调度中，请求可以在每次迭代结束后动态进出 batch，避免了 request-level 调度的计算资源浪费

Current iteration-level batching systems such as vLLM [23] and Orca [75] use prefill-prioritizing schedules that eagerly admit new requests in a running batch at the first available opportunity, e.g., whenever GPU memory becomes available. Prioritizing prefills can improve throughput because it increases the batch size of subsequent decode iterations.
>  目前的 iteration-level batching 系统使用 prefill 优先调度，即在运行的 batch 一旦出现可用资源 (例如 GPU 显存空余)，就容纳新请求
>  prefill 优先调度有助于提高吞吐，因为可以提高后续 decode 迭代的 batch size (意思是 prefill 完成的多了，decode 需求就多了，一个 batch 里就可以塞更多的 decode)

# 3 Motivation
In this section, we first analyse the cost of prefill and decode operations. We then highlight the throughput-latency trade-off and pipeline bubbles that appear in serving LLMs.

## 3.1 Cost Analysis of Prefill and Decode

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/065d3d748d6139b2db0e5636d0d8b4761c0417d52ecbdc5a4ba0aabe07a80ae3.jpg)  

Figure 3: Throughput of the prefill and decode phases with different batch sizes for Mistral-7B running on a single A100 GPU. We use prompt length of 1024 for both prefill and decode experiments. Note that different y-axis, showing prefills are much more efficient than decode. Further, note that batching boosts decode throughput almost linearly but has a marginal effect on prefill throughput.

As discussed in  $\S 2.2$  , while the prefill phase processes all input tokens in parallel and effectively saturates GPU compute, the decode phase processes only a single token at a time and is very inefficient. Figure 3 illustrates throughput as a function of batch size, and we can observe that while for decode iterations throughput increases roughly linearly with batch size, prefill throughput almost saturates even with a single request.
>  Fig3 展示了 prefill, decode 阶段的吞吐和 batch size 的大小
>  decode 的吞吐随着 batch size 几乎线性增长
>  prefill 则在单个 request 下就几乎饱和了吞吐，增加 batch size 带来的吞吐增长有限，甚至会降低

**Takeaway-1:** The two phases of LLM inference  - prefill and decode -demonstrate contrasting behaviors wherein batching boosts decode phase throughput immensely but has little effect on prefill throughput.
>  batching 可以显著提高 decode 阶段吞吐，但对 prefill 阶段没有太大效果

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/8af91d0e3952e44be57fdf3a21c33c70061e5381985b2be536dd4e7159ad344d.jpg)  

Figure 4: Prefill and decode time with different input sizes for Mistral-7B running on single A100 GPU. Linear layers contribute to the majority of runtime in both prefill and decode phases. Due to the low arithmetic intensity in decode batches, the cost of linear operation for 1 decode token is nearly same as 128 prefill tokens.

>  decode batch 的低计算密度使得对 1 个 token decode 的 linear 计算开销和对 128 个 tokens 的 prefill 开销接近

Figure 4 breaks down the prefill and decode compute times into linear, attention and others, and shows their individual contributions. From the figure, we see that linear operators contribute to the majority of the runtime cost. While attention cost grows quadratically with sequence length, linear operators still contribute more than  $80\%$  to the total time even at high sequence lengths. Therefore, optimizing linear operators is important for improving LLM inference.
>  Fig4 分析了 prefill, decode 中 linear, attention 和其他计算的时间
>  linear 计算占据了大部分开销，attention 则随着序列长度二次增长，但即便在长序列 (2K)，linear 仍然贡献了 80% 的时间，因此 linear 是主要的优化目标

**Low Compute Utilization during Decodes:** Low compute utilization during the decode phase is a waste of GPU's processing capacity. To understand this further, we analyze the arithmetic intensity of prefill and decode iterations. Since the majority of the time in LLM inference is spent in linear operators, we focus our analysis on them.
>  我们分析 prefill, decode 的算术密度来了解 decode 阶段的低计算资源利用率
>  因为大多数计算时间在于 linear 算子，故主要分析 linear

Matrix multiplication kernels overlap memory accesses along with computation of math operations. The total execution time of an operation can be approximated to  $T = \max (T_{\mathrm{math}},T_{\mathrm{mem}})$  , where  $T_{\mathrm{math}}$  and  $T_{\mathrm{mem}}$  represent the time spent on math and memory fetch operations respectively. 
>  GEMM kernel 通常会重叠访存和计算，一个算子的总执行时间可以近似为 $T = \max (T_{math}, T_{mem})$，即算子用于访存和计算的时间的最大值

>  通常也不会完美重叠，这个估计实际上是往小了估计

An operation is considered memory-bound if  $T_{\mathrm{math}}< T_{\mathrm{mem}}$  Memory-bound operations have low Model FLOPs Utilization (MFU) [35]. On the other hand, compute-bound operations have low Model Bandwidth Utilization (MBU). When  $T_{\mathrm{math}} = T_{\mathrm{mem}}$  , both compute and memory bandwidth utilization are maximized. 
>  $T_{math} < T_{mem}$ 即为 memory-bound，这类计算的模型浮点运算利用率 (MFU) 较低 (运算单元等待访存)
>  compute-bound 计算的模型带宽利用率 (MBU) 则较低 (访存单元等待计算)
>  $T_{math} = T_{mem}$ 时，计算和内存带宽都被最大化

Arithmetic intensity quantifies the number of math operations performed per byte of data fetched from the memory. At the optimal point, the arithmetic intensity of operation matches the FLOPS-to-Bandwidth ratio of the device. 
>  算术密度量化了从内存每读取一个字节所执行的计算操作数
>  最优情况下，算子的计算密度匹配设备的 FLOPS-to-Bandwidth 比率

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/11317c09fe18115fb11a9e79094bf5cebd26176e79df91d8d4bc5b5311b8475a.jpg)  

Figure 5: Arithmetic intensity trend for LLaMA2-70B linear operations with different number of token running on four A100s. Decode batches have low arithmetic intensity i.e., they are bottlenecked by memory fetch time, leading to low compute utilization. Prefill batches are compute bound with sub-optimal bandwidth utilization. Sarathi-Serve forms balanced batches by combining decodes and prefill chunks to maximize both compute and bandwidth utilization.

>  perfill batch 能够在没有冗余计算/重计算的情况下实现 compute-bonud，如果只有 prefill 需求，这已经是最优的了，即便存在了访存带宽的浪费
>  但我们还有 decode 需求，且它是轻计算重访存的
>  因此 PD 按照一定比例混合，才可以同时最大化利用硬件的计算能力和访存能力，这样的整体吞吐应该可以达到最优

>  当然 PD 混合避免不了 batch 中的 decode 延迟增大，因为需要等待 prefill
>  在给定强延迟约束下，完全的 PD 分离，虽然浪费了一定硬件资源 (并且还要存两份权重)，应该可以实现更高的 goodput

>  硬件的峰值计算能力和峰值访存能力的设计，应该是让算子开发者能够在饱和内存带宽的情况下，达到一定的计算密度，就能饱和计算单元
>  要让算子尽可能高效，一点是最大化计算密度，另一点是饱和内存带宽，在饱和内存带宽的情况下，实现该算子能达到的最大计算密度，就能实现该算子在硬件上的最优计算效率 ($\text{计算效率} = \max(\text{访存效率} \times \text{计算密度}，\text{计算单元峰值效率})$)
>  不过如果算子的计算密度非常大，在没有饱和内存带宽的情况下也可以饱和计算单元，这同样也是实现了该算子在硬件上的最优计算效率

>  最大化计算密度和饱和内存带宽实际上是存在一定互斥关系的，因为我们可以以算换存或以存换算
>  以算换存降低访存频率，提高计算频率，如果算子过饱和了内存带宽，导致算等存，以算换存既可以缓解内存带宽的过饱和，也可以提高计算密度，一举两得
>  以存换算降低了计算密度，提高了访存频率，如果算子存在大量无效的计算 (重计算)，没有利用硬件的访存效能，以存换算既可以降低无效的计算，也可以利用硬件的访存效能，一举两得

>  因此最后看的是算子能不能在最高的计算效率 (最少的冗余计算/重计算) 的情况下完全利用硬件的计算能力
>  故最后的评价指标还是每秒能够完成多少有效次计算，既 operations per second

Figure 5 shows arithmetic intensity as a function of the number of tokens in the batch for linear layers in LLaMA2-70B running on four A100 GPUs. Prefill batches amortize the cost of fetching weights of the linear operators from HBM memory to GPU cache over a large number of tokens, allowing it to have high arithmetic intensity. In contrast, decode batches have very low computation intensity. 
>  LLaMA2-70B 中 linear 层中，算术密度和 batch 中 tokens 数量的关系如 Fig5
>  prefill batch token 数量多，摊销了 linear 计算中模型权重从 HBM 到 GPU cache 的时间
>  decode batch 计算密度非常低

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/b2c2ab0d08c882dcc0949cb89ef7269a44b7dbda6e51e0e49c720846146865a2.jpg)  

Figure 6: Linear layer execution time as function of number of tokens in a batch for LLaMA2-70B on A100(s) with different tensor parallel degrees. When the number of tokens is small, execution time is dictated by the cost of fetching weights from HBM memory. Hence, execution time is largely stagnant in the 128-512 tokens range, especially for higher tensor parallel degrees. Once the number of tokens in the batch cross a critical threshold, the operation become compute bound and the runtime increases linearly with number of tokens.

Figure 6 shows the total execution time of linear operators in an iteration for LLaMA2-70B as a function of the number of tokens. Note that execution time increases only marginally in the beginning i.e., as long as the batch is in a memory-bound regime, but linearly afterwards i.e., when the batch becomes compute-bound.
>  Fig6 显示，只要还在 memory-bound 区域，增大 tokens 数量，执行时间不会受到影响，但在 batch 达到 compute-buond，执行时间就随着 tokens 数量线性增长

**Takeaway-2:** Decode batches operate in memory-bound regime leaving compute underutilized. This implies that more tokens can be processed along with a decode batch without significantly increasing its latency.
>  decode batches 处于 memory-bound 区域，计算利用率低
>  这也意味着我们可以在不显著增加延迟的情况下，让 decode batch 处理更多 tokens (直到跨越 memory-bound 的点)

>  这里考虑的都是 linear，请求之间是共享 linear 的权重的，因此 tokens/请求数量增加可以显著提高计算密度
>  如果考虑的是单纯的 attention，请求之间不共享 kv cache，此时 tokens/请求数增加对计算密度影响不大，除非使用了一定的 kv cache 前缀复用

## 3.2 Throughput-Latency Trade-off
Iteration-level batching improves system throughput but we show that it comes at the cost of high TBT latency due to a phenomenon we call generation stalls.
>  iteration-level batching 提高了系统吞吐，但会降低 TBT 延迟，原因是 generation stall 现象

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/128ccdcf05e36f619673404aa91d52cdc3d90c4fbd8a9a071ada27bb89be40b3.jpg)  

Figure 7: A generation stall occurs when one or more prefills are scheduled in between consecutive decode iterations of a request. A, B, C and D represent different requests. Subscript  $d$  represents a decode iteration,  $p$  represents a full prefill and  $p0$ ,  $p1$  represent two chunked prefills of a given prompt. vLLM induces generation stalls by scheduling as many prefills as possible before resuming ongoing decodes. Despite supporting hybrid batches, Orca cannot mitigate generation stalls because the execution time of batches containing long prompts remains high. Faster Transformer is free of generation stalls as it finishes all ongoing decodes before scheduling a new prefill but compromises on throughput due to low decode batch size. In contrast, Sarathi-Serve generates a schedule that eliminates generation stalls yet delivers high throughput.

Figure 7 compares different scheduling policies. The example shows a timeline (left to right) of requests A, B, C and D. Requests A and B are in decode phase at the start of the interval and after one iteration, requests C and D enter the system. Orca and vLLM both use FCFS iteration-level batching with eager admission of prefill requests but differ in their batch composition policy. Orca supports hybrid batches composed of both prefill and decode requests whereas vLLM only supports batches that contain either all prefill or all decode requests. Irrespective of this difference, both Orca and vLLM can improve throughput by maximizing the batch size in subsequent decode iterations.
>  Fig7 中，Orca, vLLM 都使用 FCFS 的 iteration-level 调度，对 perfill 请求采取主动接纳策略，但二者的 batch 组成策略有所不同
>  Orca 支持混合 batch, vLLM 则支持只由 prefill 或 decode 组成的 batch
>  虽然存在这一区别，Orca 和 vLLM 都通过最大化后续 decode 迭代的 batch size 来提高吞吐

However, eagerly scheduling prefills of requests C and D delays the decodes of already running requests A and B because an iteration that computes one or more prefills can take several seconds depending on the lengths of input prompts. Therefore, prefill-prioritizing schedulers can introduce generation stalls for ongoing decodes resulting in latency spikes caused by high TBT.
> 但是，对新请求的 prefill 的即时调度延缓了正在执行请求的 decode，因为计算一个 prefill batch 的迭代一般需要花费数秒  
>  因此，prefill 优先的调度器会为正在进行的 decode 引入 generation stall，导致 TBT 显著升高，出现 latency spike

In contrast to iteration-level batching, request-level batching systems such as FasterTransformer [7] do not schedule new requests until all the already running requests complete their decode phase (line 3 in Algorithm 1). In Figure 7, the prefills for requests C and D get stalled until requests A and B both exit the system. Therefore, decode-prioritizing systems provide low TBT latency albeit at the cost of low system throughput. For example, Kwon et al. [53] show that iteration-level batching with PagedAttention can achieve an order of magnitude higher throughput compared to FasterTransformer.
>  request-level batching 等到所有正在执行的请求完成 decode 阶段才调度新 request，这类 decode 优先调度系统的 TBT 延迟低，但是系统吞吐低 (因为资源利用率低)
>  vLLM 的吞吐可以比 FasterTransformer 高一个数量级 (一般 2-5 倍吧)

One way to reduce latency spikes in iteration-level batching systems is to use smaller batch sizes as recommended in Orca [75]. However, lowering batch size adversely impacts throughput as shown in §2.2. Therefore, existing systems are forced to trade-off between throughput and latency depending on the desired SLOs.
>  iteration-level batching 系统中减少 latency spike 的一个方式是使用更小的 batch size，但这也会影响吞吐
>  因此现存的系统都存在 throughput 和 latency 之间的权衡

**Takeaway-3:** The interleaving of prefills and decodes involves a trade-off between throughput and latency for current LLM inference schedulers. State-of-the-art systems today use prefill-prioritizing schedules that trade TBT latency for high throughput.
>  目前的 LLM 推理调度器都由于 prefill, decode 的互相干扰，存在 throughput, latency 之间的权衡
>  目前的 SOTA 系统使用 prefill 优先调度，用 TBT 换 throughput

## 3.3 Pipeline Bubbles waste GPU Cycles
Pipeline-parallelism (PP) is a popular strategy for cross-node deployment of large models, owing to its lower communication overheads compared to Tensor Parallelism (TP). A challenge with PP, however, is that it introduces pipeline bubbles or periods of GPU inactivity as subsequent pipeline stages have to wait for the completion of the corresponding micro-batch in the prior stages. Pipeline bubbles is a known problem in training jobs, where they arise between the forward and backward passes due to prior stages needing to wait for the backward pass to arrive. Micro-batching is a common technique used in PP training jobs to mitigate pipeline bubbles [33,49,55].

Inference jobs only require forward computation and therefore one might expect that micro-batching can eliminate pipeline bubbles during inference. In fact, prior work on transformer inference, such as, FasterTransformer [7] and FastServe [72] use micro-batches but do not mention pipeline-bubbles. Recently proposed Orca [75] also suggests that iteration-level scheduling eliminates bubbles in pipeline scheduling (see Figure 8 in [75]). However, our experiments show that even with iteration-level scheduling, pipeline bubbles can waste significant GPU cycles with PP (§5.3).

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/e13c5c3838c24ec645b6837f6945d8130b67406258b787fffda295a2f9eb46aa.jpg)  

Figure 8: A 2-way pipeline parallel iteration-level schedule in Orca across 4 requests (A, B, C, D) shows the existence of pipeline bubbles due to non-uniform batch execution times. Sarathi-Serve is able to minimize these stalls by creating uniform-compute batches.

Each micro-batch (or iteration) in LLM inference can require a different amount of compute (and consequently has varying execution time), depending on the composition of prefill and decode tokens in the micro-batch (see Figure 8). 
>  LLM 推理时，每个 microbatch (iteration) 都可能具有不同的计算量 (进而有不同的执行时间)，具体取决于 microbatch 中 prefill 和 decode tokens 的组成

We identify three types of bubbles during inference: (1) bubbles like  $PB_{1}$  that occur due to the varying number of prefill tokens in two consecutive micro-batches (2) bubbles like  $PB_{2}$  that occur due to different compute times of prefill and decode stages when one is followed by the other, and (3) bubbles like  $PB_{3}$  that occur due to difference in decode compute times between micro-batches since the attention cost depends on the accumulated context length (size of the KV-cache) and varies across requests. 
>  推理时存在三种类型的气泡:
>  1. 由两个连续的 microbatch 含有不同数量的 prefill tokens 导致的气泡
>  2. 由 decode 阶段和 prefill 阶段不同的计算时间导致的气泡
>  3. 由不同 microbatch 的 decode 计算时间不同导致的气泡

For Falcon-180B, a single prompt of 4k tokens takes  $\approx 1150$  ms to execute compared to a decode only iteration with batch size 32 which would take about  $\approx 200$  ms to execute. Interleaving of these iteration could result in a bubble of  $\approx 950$  ms. 
>  对于 Falcon-180B，一个 4k 个 tokens 的 prompt 大约需要 1150ms 执行，而一个 batch size = 32 的 decode only iteration 只需要 200ms 执行
>  对 decode iteration 和 prefill iteration 交错执行，就会导致 950ms 的流水线气泡

These pipeline bubbles are wasted GPU cycles and directly correspond to a loss in serving throughput and increased latency. This problem is aggravated with increase in prompt lengths and batch size, due to longer and more frequent prefill iterations respectively. If we can ensure that each micro-batch performs uniform computation, we can mitigate these pipeline bubbles.
>  气泡意味着 GPU 计算周期被浪费，进而导致吞吐损失和延迟增加
>  随着 prompt 程度增加和 batch size 增大，这个问题会进一步加剧
>  确保 micro-batch 执行均匀计算可以缓解气泡

**Takeaway-4:** There can be a large variance in compute time of LLM iterations depending on composition of prefill-and decode-tokens in the batch. This can lead to significant bubbles when using pipeline-parallelism.
>  PD 混合的 batch 的执行时间存在较大方差，会导致 PP 中出现显著的气泡

# 4 Sarathi-Serve: Design and Implementation
We now discuss the design and implementation of Sarathi-Serve — a system that provides high throughput with predictable tail latency via two key techniques — chunked-prefills and stall-free batching.

## 4.1 Chunked-prefills
As we show in §3.1, decode batches are heavily memory bound with low arithmetic intensity. This slack in arithmetic intensity presents an opportunity to piggyback additional computation in decode batches. 
>  decode batch 的算术密度低，memory-bound
>  而这个性质为在 decode 计算中捎带额外计算提供了机会

Naively, this can be done by creating hybrid batches which combine the memory bound decodes along with compute bound prefills. However, in many practical scenarios, input prompts contain several thousand tokens on average e.g., Table 2 shows that the median prompt size in openchat_sharegpt4 and arxiv_summarization datasets is 1730 and 7059 respectively. Combining these long prefills with decode iterations would lead to high TBT latency.
>  简单地看，可以通过创建 PD 混合 batch 实现捎带计算
>  但输入 prompt 的长度平均达到几千个 tokens，直接将长 prefill 加入 decode iteration 会导致高 TBT 延迟

To tackle this challenge, we present a technique called chunked-prefills which allows computing large prefills in small chunks across several iterations. Chunked-prefills is a prefill splitting mechanism hinged on two key insights. First, as discussed in  $\S 3.1$ , a prefill request with modest sequence length can effectively saturate GPU compute. For example, in Figure 4, prefill throughput starts saturating around sequence length of 512 tokens. Second, in many practical scenarios, input prompts contain several thousand tokens on average (Table 2). This provides an opportunity to break large prefill requests into smaller units of compute which are still large enough to saturate GPU compute. 
>  chunked-prefill 允许通过多次迭代，分小 chunks 计算完成 prefill
>  chunked-prefill 基于两个关键观察:
>  1. 对于长度适中 (512 tokens) 的序列，一个 prefill 请求就可以饱和 GPU 计算
>  2. 许多场景下，输入 prompt 平均包含几千个 tokens
>  故可以将 prompt 分为多个 chunks，各个 chunks 的大小刚好可以饱和 GPU 计算

>  这也要求了要为 chunks 保留 kv cache

In Sarathi-Serve, we leverage this mechanism to form batches with appropriate number of tokens such that we can utilize the compute potential in decode batches without violating the TBT SLO.
>  Sarathi 在不违反 TBT SLO 的前提下，使用 chunked-prefill 来动态构建包含适量 tokens 的 batch，以提高计算资源利用率

## 4.2 Stall-free batching
The Sarathi-Serve scheduler is an iteration-level scheduler that leverages chunked-prefills and coalescing of prefills and decodes to improve throughput while minimizing latency.
>  Sarathi 的调度器仍然为 iteration-level，但使用了 chunked-prefill 来合并 prefills, decodes

![[pics/Sarathi-Serve-Alg3.png]]

Unlike Orca and vLLM which stall existing decodes to execute prefills, Sarathi-Serve leverages the arithmetic intensity slack in decode iterations to execute prefills without delaying the execution of decode requests in the system. We call this approach stall-free batching (Algorithm 3). 
>  Sarathi 利用 decode iterations 的算术密度空隙，在不增加 decode 请求 TBT 的前提下添加 prefill
>  该方法称为 stall-free batching

Sarathi-Serve first calculates the budget of maximum number of tokens that can be executed in a batch based on user specified SLO. We describe the considerations involved in determining this token budget in depth in  $\S 4.3$ . In every scheduling iteration, we first pack all the running decodes in the next batch (lines 6-8 in Algorithm 3). After that, we include any partially completed prefill (lines 9-12). Only after all the running requests have been accommodated, we admit new requests (lines 13-20). 
>  Sarathi 先基于用户指定的 SLO，计算一个 batch 中可以包含的最大 tokens 数量 (token 预算)
>  在每次调度迭代，我们先将所有执行中的 decodes 加入下一个 batch，然后再加入任意尚未完成的 prefill
>  只有在所有执行中的请求都被容纳后，并且还有 token 预算，我们才接收新请求

When adding prefill requests to the batch, we compute the maximum chunk size that can be accommodated within the leftover token budget for that batch (lines 11, 15). 
>  当为 batch 添加 prefill 请求时，我们会先计算该 batch 还剩下多少 token 预算，使用这个值作为最大 chunk size (如果 prefill 低于这个最大 chunk size，就直接加入，否则就从这个 prefill 中划出这样大的一个 chunk)

By restricting the computational load in every iteration, stall-free batching ensures that decodes never experience a generation stall due to a co-running prefill chunk. 
>  stall-free batching 通过限制了每个迭代中的计算量，确保 decodes 不会经历由于和 prefill chunk 运行而导致的 generation stall

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/314565a48f13fef32de13b915fcd68d93e03a7ca4af2fe6c5568bb69a24a6ac6.jpg)  

Figure 9: The incremental cost of coalescing prefills with decode batches. We consider two batching schemes -(i) Decode + Full Prefill represents the hybrid batching of Orca wherein the entire prefill is executed in a single iteration along with ongoing decodes. (ii) Decode + Chunked Prefill represents Sarathi-Serve wherein prefills are chunked before being coalesced with ongoing decodes with a fixed token budget. Sarathi-Serve processes prefill tokens with much lower impact on the latency of decodes. Further, the relative impact of Sarathi-Serve on latency reduces with higher decode batch size and context lengths.

We compare the latency for hybrid batches with and without chunked prefills in Figure 9. Naive hybrid batching leads to dramatic increase of up to  $28.3 \times$  in the TBT latency compared to a decode-only batch. In contrast, Sarathi-Serve provides a much tighter bound on latency with chunking.
>  Fig9 比较了使用 chunked prefill 和不使用 chunked prefill 的混合 batching 导致的延迟
>  朴素的混合 batching 会导致 TBT 延迟提高 28.3x, Sarathi 则最多 3x

Figure 7 shows the scheduling policy of Sarathi-Serve in action, for the same example used in  $\S 3.2$ . The first iteration is decode-only as there are no prefills to be computed. However, after a new request C enters the system, Sarathi-Serve first splits the prefill of C into two chunks and schedules them in subsequent iterations. At the same time, with stall-free batching, it coalesces the chunked prefills with ongoing decodes of A and B. This way, Sarathi-Serve stalls neither decodes nor prefills unlike existing systems, allowing Sarathi-Serve to be largely free of latency spikes in TBT without compromising throughput. 
>  Sarathi 的调度策略见 Fig7，随着新请求 C 加入，Sarathi 将它的 prefill 分为两个 chunks，将 chunked prefill 和 ongoing decodes 组合为一个 batch
>  这样，Sarathi 既没有导致 prefill 等待，也没有导致 decode 延缓，因此在不牺牲 throughput 的前提下避免了 TBG latency spike

Furthermore, stall-free batching combined with chunked-prefills also ensures uniform compute hybrid batches in most cases, which helps reduce bubbles when using pipeline parallelism, thereby enabling efficient and scalable deployments.
>  此外，stall-free batching + chunked-prefill 也确保了均匀的 batch 计算量，减少流水线气泡

## 4.3 Determining Token Budget
The token budget is determined based on two competing factors — TBT SLO requirement and chunked-prefills overhead. From a TBT minimization point of view, a smaller token budget is preferable because iterations with fewer prefill tokens have lower latency. However, smaller token budget can result in excessive chunking of prefills resulting in overheads due to 1) lower GPU utilization and 2) repeated KV-cache access in the attention operation which we discuss below.
>  token 预算由两个相互冲突的因素 - TBT SLO 要求和 chunked-prefill 开销 - 决定
>  更小的 token 预算有利于更低的 TBT，因为捎带了更少 prefill tokens 的 iteration 的计算时间更短
>  但过小的 token 预算会导致 prefill 的 chunks 过多，会出现由于 
>  1. 低 GPU 利用率 
>  2. 反复的 KVCache 访问
>  而导致的开销

During the computation of chunked-prefills, the attention operation for every chunk of a prompt needs to access the KV-cache of all prior chunks of the same prompt. This results in increased memory reads from the GPU HBM even though the computational cost is unchanged. For example, if a prefill sequence is split into  $N$  chunks, then the first chunk's KV-cache is loaded  $N -1$  times, the second chunk's KV-cache is loaded  $N -2$  times, and so on. However, we find that even at small chunk sizes attention prefill operation is compute bound operation. In practice, there can be small overhead associated with chunking due to fixed overheads of kernel launch, etc. We present a detailed study of the overheads of chunked-prefills in §5.4.
> chunked prefill 的 attention 计算需要访问之前所有 chunks 的 KVCache，因此 chunked prefill 相较于 prefill 在计算开销不变的情况下多了 GPU HBM 访存开销
>  例如，如果 prefill 被分为 $N$ 个 chunks，则第一个 chunk 的 KVCache 会被 load $N-1$ 次，第二个 chunk 的 KVCache 会被 load $N-2$ 次
>  但我们发现，即使使用小的 chunk size, prefill 的 attention 仍然是 compute-bound
>  实践中，由于 kernel launch 存在固定开销，chunking 也会带来这一部分的额外开销

Thus, one needs to take into account the trade-offs between prefill overhead and decode latency while determining the token budget. This can be handled with a one-time profiling of batches with different number of tokens and setting the token budget to maximum number of tokens that can be packed in a batch without violating TBT SLO.
> 因此，在决定 token 预算时，需要考虑 prefill 开销和 decode latency 之间的 trade-off
> 可以对带有不同数量 tokens 的 batch 进行 profiling，将 token 预算设定为不超过 TBT SLO 的最大允许 tokens 数量

Another factor that influences the choice of token budget is the tile-quantization effect [13]. GPUs compute matmuls by partitioning the given matrices into tiles and assigning them to different thread blocks for parallel computation. Here, each thread block refers to a group of GPU threads and computes the same number of arithmetic operations. Therefore, matmuls achieve maximum GPU utilization when the matrix dimensions are divisible by the tile size. Otherwise, due to tile-quantization, some thread blocks perform extraneous computation [13]. 
>  另一个影响 token 预算的因素是 tile 量化效应
>  GPU 计算 GEMM 时，会将数据划分为 tile，将 tile 分配给不同 thread block 以并行执行
>  因此 GEMM 在矩阵维度可以被 tile size 整除时才可以达到最大 GPU 利用率，否则部分 thread block 会执行额外计算

We observe that tile-quantization can dramatically increase prefill computation time e.g., in some cases, using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256.
>  我们发现 tile 量化会显著提高 prefill 计算
>  例如 chunk size=257 会比 chunk size=256 的 prefill 时间多 32%

Finally, when using pipeline parallelism the effect of token budget on pipeline bubbles should also be taken into account. Larger chunks lead to higher inter-batch runtime variations that result in pipeline bubbles which results in lower overall system throughput. On the other hand, picking a very small token budget can lead to higher overhead due to lower arithmetic intensity and other fixed overheads.
>  token 预算和流水线气泡也有关
>  大 chunk 容易使得 batch 执行时间方差更大，导致气泡
>  但选择非常小的 token 预算也会导致气泡，因为放不下 chunk

Therefore, selecting a suitable token budget is a complex decision which depends on the desired TBT SLO, parallelism configuration, and specific hardware properties. We leverage Vidur [28], a LLM inference profiler and simulator to determine the token budget that maximizes system capacity under specific deployment scenario.
>  我们使用 Vidur 来决定特定部署场景下，最大化系统容量的 token 预算

## 4.4 Implementation
We implement Sarathi-Serve on top of the open-source implementation of vLLM [23, 53]. We added support for paged chunk prefill using FlashAttention v2 [38] and FlashInfer [74] kernels. We use FlashAttention backend for all the evaluations in this paper due to its support for wider set of models. We also extend the base vLLM codebase to support various scheduling policies, chunked prefills, pipeline parallelism and an extensive telemetry system. We use NCCL [15] for both pipeline and tensor parallel communication. Source code for the project is available at https://github.com/microsoft/sarathi-serve.
>  paged chunk prefill 基于 FlashAttention v2, FlashInfer 实现
>  调度策略，chunked prefill，流水线并行都基于 vLLM 实现

# 5 Evaluation

![[pics/Sarathi-Serve-Table1.png]]

We evaluate Sarathi-Serve on a variety of popular models and GPU configurations (see Table 1) and two datasets (see Table 2). We consider vLLM and Orca as baseline because they represent the state-of-the-art for LLM inference. 
>  四个模型，从 7B 到 180B，从 1 A100 到 8 A100 (TP4, PP2)
>  两个数据集
>  baseline: ORCA, vLLM

Our evaluation seeks to answer the following questions:

1. What is the maximum load a model replica can serve under specific Service Level Objective (SLO) constraints with different inference serving systems (5.1) and how does this load vary with varying SLO constraints (5.2)? 
2. How does Sarathi-Serve perform under various deployments such as TP and PP? (5.3) 
3. What is the overhead of chunked-prefills? (5.4.1) 
4. What is the effect of each of chunked-prefills and stall-free batching in isolation as opposed to using them in tandem? (5.4.2)

**Models and Environment:** We evaluate Sarathi-Serve across four different models Mistral-7B [51], Yi-34B [24], LLaMA2-70B [66] and Falcon-180B [31] -these models are among the best in their model size categories. 

We use two different server configurations. For all models except LLaMA2-70B we use Azure NC96ads v4 VMs, each equipped with 4 NVIDIA 80GB A100 GPUs, connected with pairwise NVLINK. The machines are connected with a 100 Gbps ethernet connection. 

For LLaMA2-70B, we use a server with eight pairwise connected NVIDIA 48GB A40 GPUs. We run Yi-34B in a 2-way tensor parallel configuration (TP-2), and LLaMA2-70B and Falcon-180B in a hybrid parallel configuration with four tensor parallel workers and two pipeline stages for (TP4-PP2). 

**Workloads:** In order to emulate the real-world serving scenarios, we generate traces by using the request length characteristics from the openchat_sharegpt4 [68] and arxiv_summarization [36] datasets (Table 2). The openchat_sharegpt4 trace contains user-shared conversations with ChatGPT-4 [6]. A conversation may contain multiple rounds of interactions between the user and chatbot. Each such interaction round is performed as a separate request to the system. This multi-round nature leads to high relative variance in the prompt lengths. In contrast, arxiv_summarization is a collection of scientific publications and their summaries (abstracts) on arXiv.org [3]. This dataset contains longer prompts and lower variance in the number of output tokens, and is representative of LLM workloads such as Microsoft M365 Copilot [14] and Google Duet AI [10] etc. 

The request arrival times are generated using Poisson distribution. We filter outliers of these datasets by removing requests with total length more than 8192 and 16384 tokens, respectively.
>  outlier filter: 分别移除了两个数据集中总长度超过 8192, 16384 tokens 的请求

**Metrics:** We focus on the median value for the TTFT since this metric is obtained only once per user request and on the 99th percentile (P99) for TBT values since every decode token results in a TBT latency value.
>  TTFT 的中位数
>  P99 TBT

## 5.1 Capacity Evaluation
We evaluate Sarathi-Serve, Orca and vLLM on all four models and both datasets under two different latency configurations: relaxed and strict. 

<table><tr><td rowspan="2">Model</td><td>relaxed SLO</td><td>strict SLO</td></tr><tr><td>P99 TBT (s)</td><td>P99 TBT (s)</td></tr><tr><td>Mistral-7B</td><td>0.5</td><td>0.1</td></tr><tr><td>Yi-34B</td><td>1</td><td>0.2</td></tr><tr><td>LLaMA2-70B</td><td>5</td><td>1</td></tr><tr><td>Falcon-180B</td><td>5</td><td>1</td></tr></table>

Table 3: SLOs for different model configurations.

Similar to Patel et al. [58], to account for the intrinsic performance limitations of a model and hardware pair, we define the SLO on P99 TBT to be equal to  $5\times$  and  $25\times$  the execution time of a decode iteration for a request (with prefill length of 4k and 32 batch size) running without any prefill interference for the strict and relaxed settings, respectively. Table 3 shows a summary of the absolute SLO thresholds. Note that the strict SLO represents the latency target desired for interactive applications like chatbots. On the other hand, the relaxed configuration is exemplary of systems where the complete sequence of output tokens should be generated within a predictable time limit but the TBT constraints on individual tokens is not very strict. 
>  我们将对 P99 TBT 的 strict, relaxed SLO 分别定义为每个请求的 decode 迭代的 5x 和 25x

For all load experiments, we ensure that the maximum load is sustainable, i.e., the queuing delay does not blow up (we use a limit of 2 seconds on median scheduling delay).

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/2a428ab56cdedbffb15c6f4d19c8887460741b1f1851b8d40af3eb49de52a59b.jpg)  

Figure 10: Capacity (in queries per second) of Mistral-7B and Yi-34B with different schedulers under strict (SLO-S) and relaxed (SLO-R) latency SLOs.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/c0d5447d0485dbbf017f1502d8f4c11127b965b7212a65611c3a39e536870baf.jpg)  

Figure 11: Capacity of LLaMA2-70B and Falcon-180B (models with pipeline parallelism) with different schedulers under strict (SLO-S) and relaxed (SLO-R) latency SLOs.

Figure 10 and Figure 11 show the results of our capacity experiments. Sarathi-Serve consistently outperforms Orca and vLLM in all cases across models and workloads. Under strict SLO, Sarathi-Serve can sustain up to  $4.0\times$  higher load compared to Orca and  $3.7\times$  higher load than vLLM under strict SLO (Yi-34B, openchat_sharegpt4). For larger models using pipeline parallelism, Sarathi-Serve achieves gains of up to  $6.3\times$  and  $4.3\times$  compared to Orca and vLLM respectively (LLaMA2-70B, openchat_sharegpt4) due to few pipeline bubbles.
>  相同 SLO 下，Sarathi 能够承受更大的 load，使用 PP 时，优势更明显 (气泡少了)

We observe that in most scenarios, Orca and vLLM violate the P99 TBT latency SLO before they can reach their maximum serviceable throughput. Thus, we observe relaxing the latency target leads to considerable increase in their model serving capacity. 

In Sarathi-Serve, one can adjust the chunk size based on the desired SLO. Therefore, we use a strict token budget and split prompts into smaller chunks when operating under strict latency SLO. This reduces system efficiency marginally but allows us to achieve lower tail latency. 

On the other hand, when the latency constraint is relaxed, we increase the token budget to allow more efficient prefills. We use token budget of 2048 and 512 for all models under the relaxed and strict settings, respectively, except for the LLaMA2-70B relaxed configuration where we use token budget of 1536 to reduce the impact of pipeline bubbles. The system performance can be further enhanced by dynamically varying the token budget based on workload characteristics. We leave this exploration for future work.

>  可以随着 SLO 程度调节 token budget
>  SLO 严格，就小一点，否则可以大一点

We further notice that vLLM significantly outperforms Orca under relaxed setting. The reason for this is two-fold. First, Orca batches prompts for multiple requests together (max sequence length * batch size compared to max sequence length in vLLM), which can lead to even higher tail latency in some cases. Second, vLLM supports a much larger batch size compared to Orca. The lower batch size in Orca is due to the lack of PagedAttention and the large activation memory footprint associated with processing batches with excessively large number of tokens.

Finally, note that the capacity of each system is higher for openchat_sharegpt4 dataset compared to the arxiv_summarization dataset. This is expected because prompts in the arxiv_summarization datasets are much longer - 7059 vs 1730 median tokens as shown in Table 2. The larger prompts makes Orca and vLLM more susceptible to latency violations due to higher processing time of these longer pre-fills.

## 5.2 Throughput-Latency Tradeoff

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/1dde4b6b698def62ead1542f0a33f861653c31be2bae4c1b78775ae1eeee84c9.jpg)  

Figure 12: Latency -Throughput tradeoff in vLLM and Sarathi-Serve for Mistral-7B and Yi-34B models on open-chat_sharegpt4 dataset. We evaluate vLLM with three different max batch sizes of 32, 64 and 128. For Sarathi-Serve, we consider token budget of 512 and 2048 with max batch size of 128. Sarathi-Serve delivers  $3.5\times$  higher capacity under stringent SLOs for Yi-34B using Stall-free batching.

To fully understand the throughput-latency tradeoff in LLM serving systems, we vary the P99 TBT latency SLO and observe the impact on system capacity for vLLM and Sarathi-Serve. Figure 12 shows the results for Mistral-7B and Yi-34B models with five different SLO values for the openchat_sharegpt4 dataset.

We evaluate vLLM with three different batch sizes in an attempt to navigate the latency-throughput trade-off as prescribed by Yu et al. [75]. The maximum capacity of vLLM gets capped due to generation stalls under stringent TBT SLOs. Notably, the capacity of vLLM remains largely identical for all the three batch size settings. This implies that even though PagedAttention enables large batch sizes with efficient memory management - in practical situations with latency constraints, vLLM cannot leverage the large batch size due to the steep latency-throughput tradeoff made by it's prefill-prioritizing scheduler.
>  在 latency 约束下，vLLM 无法达到大 batch

On the other hand, the latency-throughput tradeoff in Sarathi-Serve can be precisely controlled by varying the token budget. Sarathi-Serve achieves  $3.5\times$  higher capacity compared to vLLM under strict SLO (100ms, Mistral-7B) using a small token budget of 512. For scenarios with more relaxed SLO constraints, picking a larger token budget of 2048 allows Sarathi-Serve to operate more efficiently resulting in  $1.65\times$  higher capacity compared to vLLM (1s, Yi-34B).
>  Sarathi 可以通过控制 token 预算调节 latency-throughput trade-off

## 5.3 Making Pipeline Parallel Viable
We now show that Sarathi-Serve makes it feasible to efficiently serve LLM inference across commodity networks with efficient pipeline parallelism. For these experiments, we run Falcon-180B over two nodes, each with four A100 GPUs, connected over 100 Gbps Ethernet. We evaluate model capacity under three configurations: vLLM with 8-way TP, vLLM with our pipeline-parallel implementation and Sarathi-Serve with pipeline-parallel. For PP configurations, we do 4-way TP within node and 2-way PP across nodes.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/e1d20222b6a1dd4e313f3bd65dc6ccf6b563e01eb532641400cd65f317b171dd.jpg)  

Figure 13: TP scales poorly across nodes. (a) Median TBT for decode-only batches: cross node TP increases median TBT by more than  $2\times$  compared to a 4-way TP within node and PP across nodes. (b) Capacity under strict (SLO-S) and relaxed (SLO-R) latency SLOs: Sarathi-Serve increases Falcon-180B's serving capacity by  $4.3\times$  and  $3.6\times$  over vLLM's TP-only and hybrid-parallel configurations under strict SLOs.

Figure 13a shows the latency for decode-only batches for Falcon-180B with purely tensor parallel TP-8 deployment compared to a TP-4 PP-2 hybrid parallel configuration. We observe that the median latency for tensor parallelism is  $\sim 2\times$  higher than pipeline parallelism. This is because TP incurs high communication overhead due to cross-node all-reduces.

Figure 13b shows the capacity for tensor and hybrid parallel configurations for Falcon-180B on openchat_sharegpt4 dataset. Note that unlike the hybrid parallel configuration, TP achieves low capacity even under the relaxed SLO due to high latency. Even though vLLM can support a fairly high load with hybrid parallelism under relaxed SLO, it's performance drops sharply under the strict regime due to pipeline bubbles. Sarathi-Serve on the other hand, leverages chunked-prefills to reduce the variation in the execution time between microbatches to avoid pipeline bubbles, resulting in a  $1.48\times$  increase in capacity under relaxed SLOs and  $3.6\times$  increase in capacity under strict SLOs.

## 5.4 Ablation Study
In this subsection, we conduct an ablation study on different aspects on Sarathi-Serve. In particular, we are interested in answering the following two questions: 1) what is the effect of chunking on prefill throughput, and 2) analyzing the impact of hybrid-batching and chunking on latency. While we provide results only for a few experiments in this section, all the trends discussed below are consistent across various model-hardware combinations.

### 5.4.1 Overhead of chunked-prefills

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-21/55148cf9-ec95-487a-b5c7-645b6b62546b/c0357e721cd444eec6dac2a12e9ee1087ac5b3fab0c4431701b0cf25d63f0af5.jpg)  

Figure 14: Overhead of chunked-prefills in prefill computation for Yi-34B (TP-2) normalized to the cost of no-chunking, shown for various prompt lengths using chunk lengths of 512, 1024 and 2048.

Figure 14 shows how much overhead chunking adds in Yi-34B - on overall prefill runtime. As expected, smaller chunks introduce higher overhead as shown by the gradually decreasing bar heights in Figure 14. However, even with the smallest chunk size of 512, we observe a moderate overhead of at most  $\sim 25\%$ . Whereas with the larger token budget of 2048, chunked prefills have almost negligible overhead.
>  对于小 token budget/chunk size 512，额外开销大约为 25%，对于大 token budget/chunk size 2048，额外开销可以忽略

### 5.4.2 Impact of individual techniques

<table><tr><td rowspan="2">Scheduler</td><td colspan="2">openchat_sharegt4</td><td colspan="2">arxiv_summarization</td></tr><tr><td>P50 TIFT</td><td>P99 TBT</td><td>P50 TIFT</td><td>P99 TBT</td></tr><tr><td>hybrid-batching-only</td><td>0.53</td><td>0.68</td><td>3.78</td><td>1.38</td></tr><tr><td>chunked-prefills-only</td><td>1.04</td><td>0.17</td><td>5.38</td><td>0.20</td></tr><tr><td>Sarathi-Serve (combined)</td><td>0.76</td><td>0.14</td><td>3.90</td><td>0.17</td></tr></table>

Table 4: TTFT and TBT latency measured in seconds for hybrid-batching and chunked-prefills used in isolation as well as when they are used in tandem, evaluated over 128 requests for Yi-34B running on two A100s with a token budget of 1024. By using both hybrid-batching and chunked-prefills, Sarathi-Serve is able to lower both TTFT and TBT.

Finally, Table 4 shows the TTFT and TBT latency with each component of Sarathi-Serve evaluated in isolation i.e., chunked-prefills-only, hybrid-batching-only (mixed batches with both prefill and decode requests) and when they are used in tandem. These results show that the two techniques work best together: chunked-prefills-only increases TTFT as prefill chunks are slightly inefficient whereas hybrid-batching-only increases TBT because long prefills can still create generation stalls. When used together, Sarathi-Serve improves performance along both dimensions.

# 6 Related Work
**Model serving systems:** Systems such as Clipper [37], TensorFlow-Serving [56], Clockwork [45] and BatchMaker [44] study various placement, caching and batching strategies for model serving. However, these systems fail to address the challenges of auto-regressive transformer inference. More recently, systems such as Orca [75], vLLM [53], FlexGen [63], FasterTransformers [7], LightSeq [70], and TurboTransformers [42] propose domain-specific optimizations for transformer inference. FlexGen [63] optimizes LLM inference for throughput in resource-constrained offline scenarios i.e., it is not suitable for online serving. FastServe [72] proposed a preemptive scheduling framework for LLM inference to minimize the job completion times. We present a detailed comparison with Orca and vLLM as they represent the state-of-the-art in LLM inference.

Another approach that has emerged recently is to disaggregate the prefill and decode phases on different replicas as proposed in SplitWise, DistServe and TetrInfer [47, 58, 77]. These solutions can entirely eliminate the interference between prefills and decodes. However, disaggregation requires migrating the KV cache of each request upon the completion of its prefill phase which could be challenging in the absence of high-bandwidth interconnects between different replicas. In addition, this approach also under-utilizes the GPU memory capacity of the prefill replicas i.e., only the decode replicas are responsible for storing the KV cache. On the positive side, disaggregated approaches can execute prefills with maximum efficiency (and therefore yield better TTFT) unlike chunked prefills that are somewhat slower than full prefills. We leave a quantitative comparison between Sarathi-Serve and disaggregation-based solutions for future work.

Recently, Sheng et al. [62] proposed modification to iteration-level batching algorithm to ensure fairness among clients in a multi-tenant environment. FastServe [72] uses a preemption based scheduling mechanism to mitigate head-of-the-line blocking. Such algorithmic optimizations are complementary to our approach and can benefit from lower prefill-decode interference enabled by Sarathi-Serve. Another recent system, APIServe [26] adopted chunked prefills from Sarathi to utilize wasted compute in decode batches for ahead-of-time prefill recomputation for multi-turn API serving.

**Improving GPU utilization for transformers:** Recent works have proposed various optimizations to improve the hardware utilization for transformers. FasterTransformer uses model-specific GPU kernel implementations. CocoNet [50] and [69] aim to overlap compute with communication to improve GPU utilization: these techniques are specially useful while using a high degree of tensor-parallel for distributed models where communication time can dominate compute. Further, the cost of computing self-attention grows quadratically with sequence length and hence can become significant for long contexts. [38,39,60] have proposed various techniques to minimize the memory bottlenecks of self-attention with careful tiling and work partitioning. In addition, various parallelization strategies have been explored to optimize model placement. These techniques are orthogonal to Sarathi-Serve.

**Model optimizations:** A significant body of work around model innovations has attempted to address the shortcomings of transformer-based language models or to take the next leap forward in model architectures, beyond transformers. For example, multi-query attention [61] shares the same keys and values across all the attention heads to reduce the size of the KV-cache, allowing to fit a larger batch size on the GPUs. Several recent works have also shown that the model sizes can be compressed significantly using quantization [40, 41,43,73]. Mixture-of-expert models are aimed primarily at reducing the number of model parameters that get activated in an iteration [32,48,54]. More recently, retentive networks have been proposed as a successor to transformers [65]. In contrast, we focus on addressing the performance issues of popular transformer models from a GPU's perspective.

# 7 Conclusion
Optimizing LLM inference for high throughput and low latency is desirable but challenging. We presented a broad characterization of existing LLM inference schedulers by dividing them into two categories - prefill-prioritizing and decode-prioritizing. In general, we argue that the former category is better at optimizing throughput whereas the latter is better at optimizing TBT latency. However, none of them is ideal when optimizing throughput and latency are both important.
>  prefill 优先调度适合 TTFT 和吞吐，decode 优先调度适合 TBT

To address this tradeoff, we introduce Sarathi-Servea system that instantiates a novel approach comprised of chunked-prefills and stall-free batching.
>  chunked-prefill + stall-free batching 适合在 TBT 约束下实现最大吞吐，在 TBT 约束下提高服务的容量/承受能力

 Sarathi-Serve chunks input prompts into smaller units of work to create stall-free schedules. This way, Sarathi-Serve can add new requests in a running batch without pausing ongoing decodes. Our evaluation shows that Sarathi-Serve improves the serving capacity of Mistral-7B by up to  $2.6x$  on a single A100 GPU and up to  $5.6x$  for Falcon-180B on 8 A100 GPUs.
 >  通过 chunked-prefill，可以在不延缓 decodes 的情况下将新的 prefill 调度

# A Artifact Appendix
## Abstract
Our open source artifact is available on GitHub. This repository contains our implementation of Sarathi-Serve as well as the harnesses and scripts for running and plotting the experiments described in this paper.

This repository originally started as a fork of the vLLM project. Sarathi-Serve is a lightweight high-performance research prototype and doesn't have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.

## Scope
This artifact allows the readers to validate the claims made in the Sarathi-Serve paper (the figures) and provides a means to replicate the experiments described. The artifact can be used to set up the necessary environment, execute the main results, and perform microbenchmarks, thus providing a comprehensive understanding of the key claims in Sarathi-Serve.

## Contents
The repository is structured as follows, the primary source code for the system is contained in directory /sarathi. The implementations for custom CUDA kernels are within the /csrc directory. All the scripts to reproduce the experiments are in /osdi-experiments and finally, the trace files used for the experiments are stored in /data.

## Hosting
You can obtain our artifacts from GitHub: GitHub. The main branch of the Github repository is actively updated, but we will maintain clear and accessible instructions about our artifacts in an easily identifiable README file. All the detailed instructions and README files to reproduce the experiments in the OSDI paper are available in the branch osdi-sarathi-serve.

## Requirements
Sarathi-Serve has been tested with CUDA 12.1 on A100 and A40 GPUs. The specific GPU SKUs on which the experiments were performed and the paralelism strategies used are clearly explained in the README corresponding to the figures in the artifact, for ease of reproducibility.
