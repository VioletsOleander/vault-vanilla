# Abstract
DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding computation. Existing LLM serving systems colocate the two phases and batch the computation of prefill and decoding across all users and requests. We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases. 
>  DistServe 通过分离 prefill 和 decoding 计算来提高 LLM 服务性能
>  现存的 LLM 服务系统将这两个阶段共置，并批处理跨所有用户和请求的计算
>  我们发现这个策略会导致 prefill-decoding 之间的互相干扰，并且会耦合两个阶段的资源分配和并行规划

LLM applications often emphasize individual latency for each phase: time to first token (TTFT) for the prefill phase and time per output token (TPOT) of each request for the decoding phase. In the presence of stringent latency requirements, existing systems have to prioritize one latency over the other, or over-provision compute resources to meet both.
>  LLM 应用通常强调每个阶段单独的延迟: prefill 阶段关注首个 token 的延迟 (time-to-first-token, TTFT)，decoding 阶段关注每输出一个 token 的延迟 (time-per-output-token, TPOT)
>  在严格的延迟约束下，现有系统需要在二者之间权衡，或者过度分配计算资源来同时满足二者

DistServe assigns prefill and decoding computation to different GPUs, hence eliminating prefill-decoding interferences. Given the application's TTFT and TPOT requirements, DistServe co-optimizes the resource allocation and parallelism strategy tailored for each phase. 
>  DistServer 将 prefill 和 decoding 计算分配给不同的 GPUs，消除了 prefill-decoding 干扰
>  给定应用的 TTFT, TPOT 要求，DistServer 为每个阶段协同优化资源分配和并行策略

DistServe also places the two phases according to the serving cluster's bandwidth to minimize the communication caused by disaggregation.
>  DistServer 还会根据服务集群的带宽来放置这两个阶段，最小化由分离导致的通信

As a result, DistServe significantly improves LLM serving performance in terms of the maximum rate that can be served within both TTFT and TPOT constraints on each GPU. Our evaluations show that on various popular LLMs, applications, and latency requirements, DistServe can serve  $7.4\times$  more requests or  $12.6\times$  tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for  $>90\%$  of requests.
>  DistServer 在每个 GPU 都满足 TTFT, TPOT 的约束下显著提高了 LLM 服务的最大吞吐率
>  在流行的 LLM, 应用和延迟需求上的评估表示 DistServer 可以服务 7.4x 的请求，或者服务 12.6x 严格的服务等级目标，同时满足 >90% 请求的延迟要求

# 1 Introduction
Large language models (LLMs), such as GPT-4 [37], Bard [2], and LLaMA [51], represent a groundbreaking shift in generative AI. They start to reshape existing Internet services, ranging from search engines to personal assistants [4], and enable fundamentally new applications, like universal chatbots [1, 16] and programming assistants [15, 42]. Yet, these advances come with a significant challenge: processing an end-to-end LLM query can be substantially slower than a standard search query [41]. 

In order to meet the stringent latency requirements of various applications, service providers need to over-provision compute resources, particularly many GPUs, leading to a shortfall in cost efficiency. Therefore, optimizing the cost per LLM query while adhering to high  $SLO$  attainment (the proportion of requests that meet the SLOs) is becoming increasingly essential for all LLM services.
>  服务提供者为了满足各种应用的严格延迟要求，不得不为 LLM 过度分配计算资源，通常是很多 GPUs，导致成本效率大幅下降
>  因此，我们需要在保持高 SLO 达成率 (满足 SLO 的请求比率) 的前提下，优化 per LLM query 的成本

An LLM service responds to a user query in two phases. The prefill phase processes a user's prompt, composed of a sequence of tokens, to generate the first token of the response in one step. Following it, the decoding phase sequentially generates subsequent tokens in multiple steps; each decoding step generates a new token based on tokens generated in previous steps, until reaching a termination token. 
>  LLM 服务按照两阶段回应用户请求
>  prefill 阶段处理用户 prompt，生成第一个回应 token
>  decoding 阶段顺序生成之后的 tokens，每个 decoding 步基于之前的 tokens 生成新 token，直到终止 token

This dual-phase process distinguishes LLM services from traditional services – an LLM service's latency is uniquely measured by two key metrics: the time to first token (TTFT), which is the duration of the prefill phase, and the time per output token (TPOT), which represents the average time taken to generate a token for each request (except for the first token). 
>  两阶段性质使得 LLM 服务和传统服务不同 - LLM 服务的延迟由两个关键指标衡量: 
>  - 首个 token 生成时间 (TTFT)，即 prefill 阶段的持续时间
>  - 每个输出 token 生成时间 (TPOT)，即为请求生成回应中每个 token 的平均时间 (除去回应中的第一个 token，这个 token 算作由 prefill 生成的，计算这个 token 的 workload 是 prefill 形式而不是 decoding 形式)

>  request 处理的完整延迟时间 = TTFT + TPOT + decoding 阶段生成的 tokens 数量

Different applications place varying demands on each metric. For example, real-time chatbots [1] prioritize low TTFT for response promptness, while TPOT only remains important until it is faster than human reading speed (i.e., 250 words/min). Conversely, document summarization emphasizes low TPOT for faster generation of the summary.
>  不同的应用对不同的度量施加不同的要求
>  例如实时聊天机器人优先考虑 TTFT，确保响应的及时性，TPOT 只需要快过人类的阅读速度 (250 words/min)
>  相比之下，文档总结强调低 TPOT，以更快生成摘要

Hence, given the application's TTFT and TPOT requirements, an effective LLM serving system should balance these needs and maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say,  $90\%$  ) for each GPU provisioned - higher-per-GPU goodput directly translates into lower cost per query.
>  给定应用的 TTFT, TPOT 需求，服务系统应该平衡二者，最大化每 GPU 的有效吞吐量
>  每 GPU 有效吞吐量定义为: 在满足每个已分配 GPU 的 SLO 达成率目标 (例如 90%) 的前提下，所能支持的最大请求服务速率
>  更高的每 GPU 有效吞吐直接意味着更低的 cost per query (因为资源量/GPU 数量保持，一段时间内处理的 query 数量增大，每个 query 的平均成本就低)

As the prefill and decoding phases share the LLM weights and working memory, existing LLM serving systems typically colocate both phases on GPUs and maximize the overall system throughput - tokens generated per second across all users and requests - by batching the prefill and decoding steps across requests [31, 54]. 
>  因为 prefill 和 decoding 阶段共享 LLM 权重和工作内存，现存服务系统通常将两个阶段共置在相同 GPUs 上，并对所有请求的 prefill 和 decoding 阶段进行批处理，以最大化总体吞吐 - 即跨所有用户和请求每秒生成的 tokens 数量

>  回想一下在 ORCA 的 selective batching 中，一个 batch 内的 requests 的 workloads 是可以不同的，可以同时有 prefill 和 decoding 的请求，反正 Attention 计算会分别处理

![[pics/DistServe-Fig1.png]]

However, to meet latency requirements, we find these systems must over-provision compute resources. To see this, Figure 1 illustrates how the P90 TTFT and TPOT shift with increasing request rates when serving a 13B LLM using existing systems [32], with workload pattern and two latency constraints set to emulate using LLM to generate a short summary for an article. 
>  我们发现现存的系统为了满足延迟需求，必须过度分配计算资源
>  Fig1 展示了使用现有系统服务一个 13B 参数的 LLM 时，P90 TTFT (第九十个百分位首 token 延迟) 和 P90 TPOT (第九十个百分位每 token 延迟) 随着请求速率增长的变化情况
>  workload 模式和两个延迟约束被设定为模仿使用 LLM 为文章生成总结的情况

>  第九十个百分位就表示所有的 requests 中，有 90% 的 request 的 TTFT 和 TPOT 低于这个 P90 TTFT, P90 TPOT，因此这个指标衡量了对于大多数 requests/用户来说的体验
>  SLA 表示 Service Level Agreement 服务等级协议，代表系统需要满足的性能目标，即 P90 TTFT, P90 TPOT 都不能高于 0.04

Under the SLO attainment of  $90\%$  , the maximum achievable goodput on a single A100 GPU, which is constrained by the more stringent one of TTFT and TPOT requirements, is about 1.6 requests per second (rps). The performance contrasts sharply when each phase is served independently on a separate GPU, shown by the orange and green curves, which achieve per-GPU goodput of 5.6 rps for the prefill phase and 10 rps for decoding. 
>  在 90% 的 SLO 达成率下，单个 GPU 上最大可达的有效吞吐 (这个指标受限于 TTFT 和 TPOT 中更严格的那个要求) 大约是 1.6 requests per second
>  而当每个阶段由独立的 GPU 服务时，性能表现截然不同，如橙色和绿色的线所示，prefill 阶段每 GPU 有效吞吐可达 5.6 rps, decoding 阶段每 GPU 有效吞吐可达 10rps

Ideally, by allocating 2 GPUs for prefill and 1 GPU for decoding, we can effectively serve the model with an overall goodput of  $10\mathrm{rps}$  , or equally 3.3 rps per GPU, which is 2.1x higher than existing systems. The gap in goodput primarily stems from the colocation of the prefill and decoding - two phases with very distinct computational characteristics and latency requirements  $(\S 2.1)$
>  理想情况下，为 prefill 分配两个 GPU，为 decoding 分配一个 GPU，我们可以以总有效吞吐 10rps 服务模型，等价于 3.3 rps per GPU，是现存系统的 2.1x

>  goodput 衡量的是单位时间内完成且被用户接受的请求数量，即真正有用的输出，排除了失败、重试、延迟超限/未满足 SLO 的要求
>  现存系统的问题主要来自于共置 prefill, decoding，这两个阶段的计算特性和延迟要求是非常不同的

![[pics/DistServe-Fig2.png]]

>  batch size 表示了 batch 中 decoding 任务的数量，蓝色曲线的 batch 中始终有一个 prefill 任务
>  (但是橙色曲线按照道理 batch size = 0 时应该 latency = 0 的)

First, colocation leads to strong prefill-decoding interference. A prefill step often takes much longer than a decoding step. When batched together, decoding steps in the batch are delayed by the prefill steps, significantly elongating their TPOT; similarly, the inclusion of decoding steps contributes to a non-trivial increase in TTFT, as evidenced in Figure 2. 
>  共置 prefill, decoding 会导致互相干扰
>  prefill step 的时间通常大于 decoding step，批量处理时，batch 中的 decoding step 会被 prefill step 延迟，导致它们的 TPOT 提高
>  类似地，decoding step 的存在也会导致 prefill step 的 TTFT 增加

>  虽然 ORCA 是批量处理一个迭代，但是仍然要求批量内的请求需要都完成一个迭代才能返回，因此会存在批量内请求相互等待的问题，归根结底是引擎一次处理的最小单元就是一个批量

Even if we schedule them separately, issues persist as they begin to compete for resources. Decoding tasks awaiting GPU execution are subject to increased queuing delays due to ongoing prefill tasks, and vice versa. Prioritized scheduling of one phase risks failing the latency requirements of the other.
>  即便我们单独调度 prefill, decoding，问题仍然存在，因为它们会争夺计算资源，正在等待 GPU 执行的 decoding 任务会因为持续进行的 prefill 任务面临更长的排队延迟，反之也是
>  若对某阶段进行优先调度，也无法满足另一个阶段对延迟的需求

Second, the prefill and decoding computation differ in latency requirements and preference for different forms of parallelism  $(\S 3)$  . Colocating prefill and decoding, however, couples their resource allocation, and prevents implementing different parallelism strategies more suited to meeting the specific latency requirements of each phase.
>  prefill 和 decoding 计算在延迟要求和对并行形式的偏好也不同
>  共置 prefill, decoding 会耦合它们的资源分配，使得无法针对各个阶段实现更适合满足其延迟需求的并行策略

To overcome these challenges, we propose to disaggregate the prefill and decoding phases of LLM inference, assigning them to separate GPUs. Our approach has two benefits. First, operating each phase independently on different GPUs eliminates prefill-decoding interference. Second, it allows to scale each phase independently with tailored resource allocation and model parallelism strategies to meet their specific latency requirements. Although disaggregation causes communication of intermediate states between GPUs, we show that the communication overhead is insubstantial  $(\S 3.3)$  in modern GPU clusters, and when managed appropriately, disaggregation significantly improves per-GPU goodput.
>  我们提出分离 LLM 推理的 prefill 和 decoding 阶段，将它们分配给不同 GPUs
>  我们的方法有两个好处:
>  1. 在不同 GPU 上独立处理 prefill, decoding 会消除它们的干扰
>  2. 可以独立拓展各个阶段，使用独立的资源分配和模型并行策略来满足它们的延迟需求
>  虽然分离会导致 GPU 之间对中间状态的通信，但通信开销在现代 GPU 集群是不显著的，如果恰当处理，分离可以显著提高 per-GPU 有效吞吐

Based on the above insights, in this work, we build DistServe, a goodput-optimized LLM serving system by disaggregating the prefill and decoding phases. Given TTFT and TPOT requirements, DistServe first scales each phase independently by co-optimizing the GPU allocation and parallelism strategies of the prefill and decoding phase assuming serving a single model replica. The optimization ensures maximizing the per-GPU goodput and may assign different numbers of GPUs and parallelism strategies to each phase depending on their respective latency requirements. 
>  我们提出 DistServe，通过分离 prefill, decoding 来优化有效吞吐的 LLM 服务系统
>  给定 TTFT 和 TPOT 要求，DistServer 首先通过独立拓展每个阶段来实现性能优化，即在假设服务单个模型副本的前提下，协同优化 prefill 和 decode 阶段的 GPU 分配和并行策略
>  优化目标是确保最大化 per GPU goodput 并且可以根据不同阶段的延迟需求分配不同数量的 GPU，并采用不同的并行策略

DistServe then scales this allocation to multiple instances via replication until meeting the user-required traffic rate  $(\S 4)$  . 
>  之后，DistServe 通过复制实例的方式将该分配拓展到多个实例，直到满足用户指定的流量速率

DistServe also features an algorithm to place the prefill and decoding computation according to their allocation schemes and the cluster's bandwidth to minimize the overhead of communicating intermediate states between phases.
>  DistServe 也包含了一个算法，会根据 prefill, decoding 计算的分配方案和集群带宽来放置 prefill, decoding 计算，以最小化阶段间的中间状态通信开销

We implement DistServe as an orchestration layer on top of the LLM inference engine. We evaluate DistServe on various LLMs, varying the workloads based on three important real-world LLM applications: chatbots, programming assistant, and document summary. Compared to state-of-the-art solutions, DistServe can serve up to  $7.4\times$  more requests or  $12.6\times$  tighter SLO under various latency constraints. 
>  我们将 DistServe 实现为 LLM 推理引擎上的编排层
>  我们在多个 LLM 和多个 workloads 上评估了 DistServe: 对话机器人、编程助手、文档总结
>  相较于 SOTA 的方案，DsitServe 可以在不同延迟约束下服务 7.4x 的请求或实现 12.6x 严格的 SLO

Our contributions are:
- Identify the problems of prefill-decoding interference and resource coupling in existing LLM serving systems and propose to disaggregate the two phases. 
- Design a novel placement algorithm to choose the goodput-optimal schema for prefill and decoding instances automatically.
- Conduct a comprehensive evaluation of DistServe with realistic workloads.

>  我们的贡献为:
>  - 发现了 LLM 推理系统中 PD 干扰和资源耦合的情况，提出分离两个阶段
>  - 设计了放置算法，自动为 PD 实例选择有效吞吐最优的方案
>  - 在真实 workloads 上对 DistServe 执行了详细的评估

# 2 Background and Motivation
An LLM service follows a client-server architecture: the client submits a sequence of text as a request to the server; the server hosts the LLM on GPUs, runs inference over the request, and responds (or streams) the generation back to the client. 
>  LLM 服务遵循 client-server 架构: client 将文本序列作为请求提交个 server, server 在 GPU 上部署 LLM，对请求执行推理，并将结果返回 (或流式传输) 给 client

As explained in  $\S 1$ , due to the unique prefill-decoding process, LLM service may impose aggressive service-level objectives (SLOs) on both TTFT and TPOT, varying with the application's needs. The serving system must meet both SLOs while minimizing the cost associated with expensive GPUs. In other words, we want the serving system to maximize the requests served per second adhering to the SLO attainment goal for each GPU provisioned -maximizing per-GPU goodput. 
>  LLM 服务会对 TTFT 和 TPOT 都提出严格的 SLO 指标，服务系统必须在满足二者的 SLO 的同时最小化成本
>  换句话说，我们希望服务系统在每台已分配的 GPU 上，在达成 SLO 目标的情况下，最大化 rps - 即最大化 per GPU goodput

Next, we detail the LLM inference computation ( $\S 2.1$ ) and discuss existing optimizations for LLM serving ( $\S 2.2$ ).

## 2.1 LLM Inference
Modern LLMs [37, 51] predict the next token given an input sequence. This prediction involves computing a hidden representation for each token within the sequence. An LLM can take a variable number of input tokens and compute their hidden representations in parallel, and its computation workload increases superlinearly with the number of tokens processed in parallel. Regardless of the input token count, the computation demands substantial I/O to move LLM weights and intermediate states from the GPU's HBM to SRAM. This process is consistent across varying input sizes.
>  LLM 可以接受任意数量 tokens，并行计算它们的隐藏表示
>  LLM 的计算 workload 随着并行处理的 tokens 数量超线性增长 (平方关系)
>  并且无论输入 token 数量多少，计算都要求大量的 IO 操作，将 LLM 权重和中间状态从 GPU 的 HBM 移动到 SRAM，这个数据搬运过程在不同的输入规模下都是一致的

The prefill step deals with a new sequence, often comprising many tokens, and processes these tokens concurrently. Unlike prefill, each decoding step only processes one new token generated by the previous step. This leads to significant computational differences between the two phases. When dealing with user prompts that are not brief, the prefill step tends to be compute-bound. For instance, for a 13B LLM, computing the prefill of a 512-token sequence makes an A100 near compute-bound (see  $\S 3.1$ ). In contrast, despite processing only one new token per step, the decoding phase incurs a similar level of I/O to the prefill phase, making it constrained by the GPU's memory bandwidth.
>  prefill 并发处理新输入序列，decoding 仅处理上一步生成的新 token
>  当用户 prompt 不短时，prefill 倾向于计算密集型，例如 13B 的 LLM 在处理 512 tokens 的 prefill 会让 A100 处于 compute-bound 状态
>  decoding 阶段的 IO 开销和 prefill 阶段类似，计算量却很少，因此是访存密集型，受 GPU 内存带宽约束

During both phases, intermediate states, known as KV caches [32], are generated at each token position, which are needed again in later decoding steps. To avoid recomputing them, they are saved in GPU memory. Because of the shared use of LLM weights and KV caches in memory, most LLM inference engines opt to colocate the prefill and decoding phases on GPUs, despite their distinct computational characteristics.
>  这两个阶段都会为每个 token 位置生成 KVCache
>  KVCache 会被存储在显存中，因为 prefill 和 decoding 共享显存中的 LLM 权重和 KVCache，故大多数 LLM 推理引擎都倾向于在 GPUs 上共置 prefill 和 decoding 阶段，忽略了二者的计算特性差异

## 2.2 LLM Serving Optimization
In real-time online serving, multiple requests come and must be served within SLOs. Batching and parallelizing their computation is key for achieving low latency, high throughput, and high utilization of GPUs.
>  实时的在线服务场景中，多个请求到来，都必须在 SLO 内服务
>  批处理和并行化的实现低延迟，高吞吐和高 GPU 利用率的关键

**Batching.** Current serving systems [9, 32, 54] utilize a batching technique known as continuous batching. This method batches the prefill of new requests with the decoding of ongoing ones. It boosts the GPU utilization and maximizes the overall system throughput - tokens generated per second across all users and requests. However, as mentioned in  $\S 1$  and elaborated later in  $\S 2.3$ , this approach leads to trade-offs between TTFT and TPOT. An advanced variant of continuous batching [9] attempts to balance TTFT and TPOT by segmenting long prefill into chunks and attaching decoding jobs with a chunked prefill - but essentially, it trades TTFT for TPOT and cannot eliminate the interference ( $\S 2.3$ ). In summary, batching prefill and decoding invariably leads to compromises in either TTFT or TPOT.
>  批处理
>  目前的服务系统利用连续批处理技术，该方法将正在处理的 decoding 请求和新的 prefill 请求组合为一个批量
>  该方法提高了 GPU 利用率，并最大化了总系统吞吐 (跨所有用户和请求，每秒生成的 token 数量)
>  但如上讨论，这个方法导致了 TTFT 和 TPOT 的权衡
>  一个连续批处理的变体尝试将长序列的 prefill 分为 chunks，并将 decoding 任务和这些 chunks 关联起来，来平衡 TTFT 和 TPOT，但本质上它是用 TTFT 换 TPOT，无法消除二者的干扰
>  综上所述，将 prefill 和 decoding 批处理必然会导致 TTFT 或 TPOT 中至少一项的妥协

**Model parallelism.** In LLM serving, model parallelism is generally divided as intra-and inter-operator parallelisms [33, 46, 59]. Both can be used to support larger models but may impact serving performance differently. Intra-operator parallelism partitions computationally intensive operators, such as matrix multiplications, across multiple GPUs, accelerating computation but causing substantial communication. It reduces the execution time, hence latency, particularly for TTFT of the prefill phase, but requires high bandwidth connectivity between GPUs (e.g., NVLINK). Inter-operator parallelism organizes LLM layers into stages, each running on a GPU to form pipelines. It moderately increases execution time due to inter-stage communication, but linearly scales the system's rate capacity with each added GPU. 
>  模型并行
>  LLM 服务中，模型并行通常划分为算子内和算子间并行，二者对于服务性能的影响有所不同
>  算子内并行将计算密集的算子划分到多个 GPU，例如矩阵乘，它减少了执行时间，进而减少了延迟，尤其是 prefill phase 的 TTFT，但要求 GPU 之间有高带宽
>  算子间并行将 LLM 层组织为阶段，每个阶段在不同的 GPU 上运行，构成流水线，这种方法由于阶段间通信，会适度增加执行时间，但会随着 GPU 添加而线性增长系统吞吐率

In this paper, we reveal an additional benefit of model parallelism: reduced queuing delay of both prefill and decoding phases, steaming from shorter execution time. We delve into this further in  $\S 3$ . Besides model parallelism, replicating a model instance, irrespective of its model parallelism configurations, linearly scales the system's rate capacity.
>  本文展示模型并行的额外优势: 通过缩短执行时间，减少 prefill, decoding 阶段的排队延迟
>  此外，无论是否采用模型并行，复制模型实例可以线性增加系统吞吐能力

These parallelism strategies create a complex space of optimization that requires careful trade-offs based on the application's latency requirements.

## 2.3 Problems and Opportunities
Colocating and batching the prefill and decoding computation to maximize the overall system throughput, as in existing systems, is cost-effective for service providers. However, in the presence of SLOs, present approaches struggle to maintain both high service quality and low cost due to the issues discussed below.
>  现存系统共置并批处理 prefill, decoding 计算来最大化吞吐的方式对于服务提供商是成本高效的
>  但是在 SLO 要求下，目前方式无法同时满足高的服务质量和低成本，原因如下

![[pics/DistServe-Fig2.png]]

**Prefill-decoding interference.** As Figure 2 shows, adding a single prefill job to a batch of decoding requests significantly slows down both processes, leading to a marked increase in TTFT and TPOT. Specifically, the decoding tasks in the batch must wait for lengthier prefill jobs to complete, thus extending TPOT; the slowdown intensifies with a longer prefill, shown in Figure 2(b). Adding decoding jobs to prefill also increases the time to complete the prefill task, particularly when the GPU is already at capacity (Figure 2 blue curves).
>  prefill-decoding 干扰
>  如 Fig2 所示，为一批 decoding 请求添加一个 prefill 任务会显著减缓 prefill, decoding 的速度，导致 TTFT 和 TPOT 的增加
>  具体地说，批量中的 decoding 任务必须等待更久的 prefill 任务完成，因此 TPOT 增加，如果 prefill 更长，减慢会更明显
>  为 prefill 添加 decoding 任务也会增加完成 prefill 任务的时间，尤其是在 GPU 已经处于满负载状态时

One attempt to mitigate this interference is called chunked-prefill with piggyback [3,9]. It proposes to split the long prefill into chunks and batch a prefill chunk with a few decoding jobs (a.k.a. piggybacking). This technique alleviates the slowdown of the decoding job caused by the long prefill job, but it does not eliminate it. 
>  缓解这个干扰的一个尝试是 chunked-prefill with piggyback
>  它提出将长的 prefill 任务分为 chunks，然后将每个 prefill chunk 和一小部分 decoding 合为一个 batch (搭便车)
>  这个技术缓解了 decoding 任务由长 prefill 任务导致的减慢，但没有消除它

Additionally, it results in an extra overhead for the prefill job which cannot be easily mitigated by adjusting the chunk size. First, if the chunk size is set much lower than the inflection point that can saturate the GPU, then the prefill job will have a longer execution time since it competes with the decoding job in the same batch and cannot solely utilize the GPU resources. Second, if we increase the chunk size to nearly saturate the GPU, the chance of piggybacking will diminish since the remaining slots for decode tokens are limited. Also, chunked-prefill causes significantly more memory access for the prefill jobs. This is because the KV cache of all previous chunks have to be loaded from HBM to SRAM repeatedly to compute each subsequent chunk. Concretely, if a prefill job is split into  $N$  equal chunks, we need to load  $N + (N -1) + \ldots +1 = \mathsf{O}(N^2)$  chunks of KV Cache in total, compared to  $\mathrm{O(N)}$  in the non-chunked case. This overhead will increase as the context length becomes longer.
>  此外，这个技术还为 prefill 任务带来了额外开销，这个开销难以通过调节 chunk size 轻易缓解
>  首先，如果 chunk size 显著低于能充分饱和 GPU 的拐点时，prefill 任务的执行时间将变长，因为它和同一 batch 中的 decoding 任务竞争 GPU 资源，无法独立利用 GPU 资源 (chunk size 小，搭便车的 decoding job 就多)
>  其次，如果我们增加 chunk size 到近乎饱和 GPU，可用于搭便车的机会将大大减少，因为剩余的为 decode tokens 准备的 slots 是有限的，这是因为所有之前的 chunks 的 KVCache 都必须反复从 HBM load 到 SRAM，以计算后续的 chunk，具体地说，如果 prefill 任务被划分为 $N$ 个等大 chunks，我们需要一共加载 $N + (N-1) + \dots + 1 = O (N^2)$ chunks，而相比之下，在非 chunk 情况下只需要 $O (N)$ 次加载，随着上下文长度上升，这种开销将急剧上升 (chunk size 大，搭便车的 decoding jobs 就少)

**Ineffective scheduling.** Unbatching prefill and decoding jobs and scheduling them sequentially does not mitigate the interference. Decoding jobs may experience longer queuing delays due to waiting for ongoing prefill jobs on GPUs. Moreover, batches dedicated to decoding often lead to GPU underutilization. Prioritizing tasks in either phase adversely affects the latency of the other, rendering priority scheduling ineffective.
>  低效调度
>  将 prefill, decoding 任务分开处理，并顺序调度它们不会缓解干扰
>  decoding 任务会因为等待 GPUs 上正在进行的 prefill 任务而经历更长的排队延迟
>  此外，专门用于 decoding 的 batches 会导致 GPU 没有被充分利用
>  无论有效调度哪个阶段的任务，都会影响另一个阶段的任务的延迟，使得优先级调度策略低效

**Resource and parallelism coupling.** Colocating prefill and decoding phases on the same GPUs unavoidably share their resource and parallelism settings. However, each phase has its unique computational characteristic and latency requirement that calls for more heterogeneous resource allocation. For example, the prefill phase tends to be compute-bound and benefits from more intra-op parallelism to reduce execution time to meet the tight SLO on TTFT. By contrast, the optimal parallelism configuration of the decoding phase depends on the running batch size. In existing systems, due to coupling, resource allocation and parallelism plans are tailored to satisfy the more demanding of TTFT and TPOT, which may not be ideal for the other. This often leads to resource over-provisioning to meet both SLOs.
>  资源和并行解耦
>  将 prefill 和 decoding 阶段共置在相同 GPU 不可避免地需要共享资源和并行设定
>  但每个阶段具有独特的计算特定和延迟要求，因此异构的资源分配更好
>  例如，prefill 阶段倾向于 compute-bound，更适合算子内并行以减少执行时间，满足 SLO 的 TTFT 要求，而 decoding 阶段的最优并行配置依赖于运行的批量大小
>  现存系统中，两个阶段的耦合导致需要配置资源偏好 TTFT, TPOT 的其中更严格一方，对于另一方就不是最优的，这就常常导致为了满足两方的 SLO 而过度分配资源

**Opportunities.** To address these issues, we propose to disaggregate the prefill and decoding phases. We use the term instance to denote a unit of resources that manages exactly one complete copy of model weights. One instance can correspond to many GPUs when model parallelism is applied. Note that when we disaggregate the two phases to different GPUs, each phase manages its copy of the model weights, resulting in prefill instances and decoding instances.
>  为了解决这些问题，我们提出分离 prefill 和 decoding 阶段
>  我们使用 “实例” 表示一个资源单元，管理模型权重的一个完整拷贝，使用 MP 时，一个实例可以对应于多个 GPUs
>  当我们将两个阶段分离在不同 GPUs，不同的阶段管理自己的模型权重拷贝，因此存在 prefill 实例和 decoding 实例

A prefill instance, upon receiving a request, performs only the prefill computation for this request to generate the first output token. It then sends the intermediate results (mainly KV caches) to a decoding instance, which is responsible for subsequent decoding steps. Because decoding computation often has low GPU utilization, we may allocate multiple prefill instances per decoding instance. This allows batching more decoding jobs to achieve higher GPU utilization.
>  prefill 实例接收到请求后，仅执行 prefill 计算，生成第一个输出 token
>  然后它将中间结果 (主要是 KVCache) 发送给 decoding 实例
>  decoding 实例负责后续的解码步
>  因为 decoding 计算的 GPU 利用率低，我们可以为每个 decoding 实例分配多个 prefill 实例 (decoding 实例为同时多个 prefill 实例执行后续 decoding)，这允许我们批处理更多 decoding 任务，达到更高的 GPU 利用率 

>  多个 prefill 实例配一个 decoding 实例带来的坏处就是 decoding 的显存要求更高，因此批处理更多 decoding 任务能不能带来更多 GPU 利用率还是两说

>  根据 vLLM 的图片，GPU 显存中大头是模型权重，剩余就是 KVCache
>  所以有个思路是更细粒度地划分实例，把 decoding 实例的 Attention 计算和其他计算分开
>  Attention 计算不需要模型权重，Attention 实例的显存可以完全存 KVCache，显著缓解显存压力，实现批处理更多 decoding 任务，达成更高的 GPU 利用率
>  但是通信会是问题，需要频繁的 Attention 实例和非 Attention 实例之间的通信
>  或许可以和流水线并行结合，但是就需要把每个 Transformer block 至少划两个阶段: Attention 和 MLP

>  细致的分析还是要看一看显存中各个数据的计算密度是怎么样的
>  GPU 间也是如此，GPU 内的计算密度就是计算量/访存量，GPU 间的计算密度就是计算量/通讯量

Disaggregating prefill and decoding naturally resolves the interference between the two phases and enables each to focus on its optimization target - TTFT or TPOT. Each type of instance can employ different resources and parallelism strategies to meet a variety of latency requirements. By adjusting the number of GPUs and parallelisms provided to the two types of instances, we can maximize the per-device goodput of the overall system, avoiding over-provisioning, eventually translating to reduced cost-per-query adhering to service quality. Next, we develop ways to find out the best resource allocation and parallelism plan for each phase.
>  分离 prefill 和 decoding 自然地解决了两个阶段之间的干扰，使得两个阶段可以独立专注于自己的优化目标: TTFT, TPOT
>  每种类型的实例可以采用不同的资源和并行策略来满足一系列延迟要求，通过调节两个类型的实例的 GPU 数量和并行配置，我们可以最大化系统的 pre GPU 有效吞吐，避免过度分配资源，进而在保持服务质量的情况下减小 cost-per-query

>  有效吞吐: 满足延迟 SLO 要求下的吞吐
>  吞吐: 不考虑延迟 SLO 要求下的吞吐

# 3 Tradeoff Analysis
Disaggregation uncouples the two phases and allows a distinct analysis of the characteristics of each phase, providing valuable insights into the algorithm design. It also expands the design space: now each phase needs to be scaled and scheduled independently based on their latency requirements.

In this section, we analyze the computational pattern of prefill (§3.1) and decoding instances (§3.2) post disaggregation. We aim to identify key parameters and derive guidelines for batching and parallelism in each phase. We then highlight several practical deployment considerations (§3.3). This section lays the foundation for per-gpu goodput optimization.
>  我们分析解耦后，prefill 和 decoding 实例的计算特性

## 3.1 Analysis for Prefill Instance
After disaggregation, the prefill phase generates the first token by processing all tokens of the user prompt in parallel. Assuming a given arrival rate, we aim to fulfill the service's latency requirement on TTFT using the least resources.
>  在解耦后，prefill 阶段并行处理 user prompt 的所有 tokens，生成第一个 token
>  假设一个给定的到达率，我们目标是使用最少的字面满足 TTFT 要求

![[pics/DistServe-Fig3.png]]

**Batching strategy.** The prefill step is typically compute-intensive. Figure 3(a) shows how the throughput of the prefill phase changes with the input length and the batch size. For a 13B parameter LLM, processing a single sequence of 512 tokens can fully engage an A100 GPU. Once the GPU becomes compute-bound, adding more requests to the batch no longer improves GPU efficiency. Instead, it proportionally extends the total processing time for the batch, inadvertently delaying all included requests. 
>  批处理策略
>  prefill 是计算密集的，Fig3a 展示了 prefill 的吞吐随着输入长度和 batch size 的变化
>  对于 13B 的 LLM，处理一个 512 tokens 的序列就可以完全占用一块 A100 的计算能力
>  当 GPU 是 compute-bound，为 batch 中添加更多请求不会再提高 GPU 效率，而是会成比例地增加 batch 的总处理时间，反而增大了 batch 内所有请求的延迟

>  compute-bound 之后就已经过了 throughput-latency 的 trade-off，此时再增加 batch size ，用 latency 换 throughput 的效率会更低

Hence, for prefill instances, it is necessary to profile the specific LLM and GPUs in advance to identify a critical input length threshold, denoted as  $L_{m}$ , beyond which the prefill phase becomes compute-bound. Batching more requests should only be considered when the input length of the scheduled request is below  $L_{m}$ . In practice, user prompts typically average over hundreds of tokens [8]. Batch sizes for the prefill instance are generally kept small.
>  因此，对于 prefill 实例，有必要提前对特定 LLM 和 GPUs 进行性能测试，以确定一个关键的输入长度阈值，记作 $L_m$
>  超过该阈值后，prefill 阶段为 compute-bound，批处理更多请求应该只在调度的请求的长度低于 $L_m$ 才考虑
>  实践中，用户的 prompts 通常平均为几百个 tokens，因此 prefill 实例的 batch size 通常保持得比较小

![[pics/DistServe-Fig3.png]]

**Parallelism plan.** To study the parallelism preferences for prefill-only instances, we serve a 66B LLM on two A100 GPUs with inter-op or intra-op parallelism strategy. To simplify the problem, we assume uniform requests input lengths of 512 tokens and a Poisson arrival process.
>  并行计划
>  为了研究 prefill-only 实例的并行偏好，我们在两个 A100 上服务 66B 模型，使用算子内和算子间并行策略
>  为了简化问题，我们假设请求都为均匀的长度 512 tokens 输入，以及 Possion 到达过程

We compare the resulting average TTFT at various arrival rates in Figure 4(a): intra-op parallelism is more efficient at lower arrival rates, while inter-op parallelism gains superiority as the rate increases. 
>  我们比较了各种到达率的平均 TTFT: 算子内并行在低到达率 (低负载) 下更高效，算子间并行随着到达率增大 (高负载) 变得更高效

Disaggregation enables the prefill phase to function analogously to an M/D/1 queue, so we can use queuing theory to verify the observation.
>  解耦可以让 prefill 阶段类似于 M/D/1 队列模型，我们可以用排队论来验证我们的观察

>  M/D/1 队列是排队论中的经典模型，适用于服务时间恒定的场景
>  M: 表示到达时间服从 Possion 过程，即到达时间间隔服从指数分布，M 表示 Markovian 或 Memoryless
>  D: 表示服务时间是确定性的，D 表示 Deterministic
>  1: 表示系统中只有一个服务台
>  系统假设:
>  顾客按 Possion 到达，平均到达率为 $\lambda$ (单位时间到达的顾客数)
>  每个顾客的服务时间恒定为 $D = 1/\mu$ ($\mu$ 为服务率，即单位时间可服务的顾客数)

>  在系统稳定 (系统利用率 $\rho = \lambda/\mu < 1$) 的前提下，M/D/1 队列有一些经典结论:
>  平均等待时间 (顾客在队列中平均等待多久才开始被服务):

$$
W_q = \frac {\rho}{2\mu(1-\rho)}
$$

>  平均逗留时间 (顾客在系统中从进入到离开的平均时间，等于顾客的等待时间 + 被服务时间):

$$
W = W_q + \frac 1 \mu = \frac 1 \mu + \frac {\rho}{2\mu(1-\rho)}
$$

We start by developing notations using the single-device case without parallelism: each request's execution time, denoted as  $D$ , remains constant due to uniform prefill length. Since one request saturates the GPU, we schedule requests via First-Come-First-Served (FCFS) without batching. 
>  我们先考虑单设备，无并行的情况:
>  每个请求的执行时间记作 $D$，因为 prefill 长度固定，故 $D$ 为常量
>  因为一个请求就会饱和 GPU，我们不进行批处理，使用 FCFS 调度请求

Suppose the Poisson arrival rate is  $R$  and the utilization condition of  $RD < 1$ , the average TTFT  $(Avg\_ TTFT)$  can be modeled by the M/D/1 queue [47] in close form:

$$
Avg\_ TTFT = D + \frac{RD^2}{2(1 -RD)}. \tag{1}
$$

where the first term represents the execution time and the second corresponds to the queuing delay. 
>  假设 Possion 到达率为 $R$ (单位时间请求数)，并且利用率 $RD < 1$ ($RD$ 表示了系统总负载，表示了 GPU 被占用的时间比例，如果 $RD < 1$，系统可以稳定运行，如果 $RD \ge 1$，系统过载，队列会无限增长)
>  平均 TTFT 可以使用 M/D/1 队列来建模，如上所示
>  其中第一项表示执行时间，第二项表示排队延迟

Based on Eq. 1, we incorporate parallelism below.

With 2-way inter-op parallelism, we assume the request-level latency becomes  $D_{s}$ , and the slowest stage takes  $D_{m}$  to finish. We have  $D \approx D_{s} \approx 2 \times D_{m}$ , due to negligible interlayer activation communication [33,59]. The average TTFT with 2-way inter-op parallelism is derived as:

$$
Avg\_ TTFT_{inter} = D_s + \frac{RD_m^2}{2(1 -RD_m)} = D + \frac{RD^2}{4(2 -RD)}. \tag{2}
$$

>  对于 2-way 算子间并行 (两阶段流水线并行)，我们假设请求级延迟为 $D_s$，并且最慢的阶段需要 $D_m$ 完成
>  请求级延迟 $D_s$ 即整个请求从开始到结束所需要的时间 (端到端延迟)
>  $D_m$ 即所有阶段中耗时最长的阶段的时间
>  我们有 $D \approx D_s \approx 2 \times D_m$，其中我们忽略了层间激活通讯开销
>  2-way 算子间并行的平均 TTFT 推导为:
>  在流水线中，每个阶段的服务时间是 $D_m$，排队发生在每个阶段入口处，因此排队延迟基于 $D_m$ 计算，因此用 $D_m$ 替换原公式排队延迟项中的 $D$，就得到了公式 2

For intra-op parallelism, we introduce a speedup coefficient  $K$ , where  $1 < K < 2$ , reflecting the imperfect speedup caused by high communication overheads of intra-op parallelism. With the execution time  $D_{s} = \frac{D}{K}$ , the average TTFT for 2-degree intra-op parallelism is:

$$
Avg\_ TTFT_{intra} = \frac{D}{K} +\frac{RD^2}{2K(K -RD)}. \tag{3}
$$

>  对于算子内并行，我们引入加速系数 $K$，因为算子内并行存在高通信开销，因此 $1 < K < 2$
>  并行下，执行时间为 $D_s = \frac {D}{K}$，故对 2-degree 的算子内并行，我们用 $D_s$ 替换原公式中的执行时间项和排队时间项中的 $D$，得到 Eq3

Comparing Eq. 2 and Eq. 3: at lower rates, where execution time (first term) is the primary factor, intra-op parallelism's reduction in execution time makes it more efficient. As the rate increases and the queuing delay (second term) becomes more significant, inter-op parallelism becomes advantageous, concurred with Figure 4(a).
>  比较 Eq2 和 Eq3 可以直到: 在较低的请求率下，执行时间 (第一项) 是主要的延迟因素，此时算子内对于执行时间的减少使得它更加高效；随着请求率增加，排队延迟 (第二项) 成为主导，此时算子间并行会更有优势 (因为算子内并行的 $K<2$，无法达到 $2$)

The prefill phase's preference for parallelism is also influenced by TTFT SLO and the speedup coefficient  $K$ . Seen from Figure 4(a): A more stringent SLO will make intra-op parallelism more advantageous, due to its ability to reduce execution time. The value of  $K$  depends on factors such as the input length, model architecture, communication bandwidth, and placement [46,59]. As shown in Figure 4(b), a decrease in  $K$  notably reduces the efficacy of intra-op parallelism. §4 develops algorithms that optimize the resource and parallelism configurations taking into consideration these knobs.
>  prefill 阶段对并行性的偏好也受 TTFT SLO 和加速系数 $K$ 的影响: 更严格的 SLO 会使得算子内并行更有优势，因为它可以减少执行时间
>  加速系数 $K$ 的值受多种因素影响，例如输入长度、模型架构、通信带宽和放置，$K$ 的减少会显著影响算子内并行的效率

## 3.2 Analysis for Decoding Instance
Unlike the prefill instance, a decoding instance follows a distinct computational pattern: it receives the KV caches and the first output token from the prefill instance and generates subsequent tokens one at a time. For decoding instances, our optimization goal is to satisfy the application's TPOT requirement using minimal computing resources.
>  decoding 实例的计算模式和 prefill 实例的计算模式完全不同: 它从 prefill 实例接受第一个输出 token 和 KVCaches，然后一次生成一个后续 token
>  对于 decoding 实例，我们的优化目标是使用最小的计算资源满足应用的 TPOT 需求

![[pics/DistServe-Fig3.png]]

**Batching strategy.** Since a single decoding job is heavily bandwidth-bound, batching is key to avoiding low GPU utilization (hence high per-gpu goodput), as shown in Figure 3(b). In existing systems where the prefill and decoding phases are colocated, increasing the decoding batch size is difficult because it conflicts with meeting latency goals, particularly in scenarios with high request rates. This is because sharing GPUs cause competition between prefill and decoding jobs, leading to a trade-off between TTFT and TPOT. For example, a higher arrival rate generates more prefill jobs, demanding greater GPU time to meet TTFT requirements if prioritizing prefill jobs, which in turn adversely affects TPOT.
>  批处理策略
>  因为单个 decoding job 是高度 bandwidth-bound，故批处理是避免低 GPU 利用 (进而高 per GPU goodput) 的关键
>  现存系统中，prefill decoding 共置，提高 decoding batch size 面临困难，因为和满足延迟目标冲突，尤其是在请求率较高的场景下
>  这是因为 prefill, decoding jobs 之间会竞争 GPU 资源，导致 TTFT 和 TPOT 之间的权衡，例如高的请求率会生成更多 prefill jobs，若优先处理 prefill 任务以保障 TTFT 要求，需要占用更多 GPU 时间，进而对 TPOT 产生不利影响

On the contrary, disaggregation offers a solution by enabling the allocation of multiple prefill instances to a single decoding instance. This approach allows for accumulating a larger batch size on dedicated GPUs for the decoding phase without sacrificing TPOT.
>  相较之下，解耦允许为单个 decoding 实例分配多个 prefill 实例，该方法允许在专用于 decoding 的 GPU 上积累更大的 batch size，而无需牺牲 TPOT

**Parallelism plan.** Post-disaggregation, the batch size for decoding may be constrained by GPU memory capacity, as it is necessary to maintain the KV caches for all active requests. Scaling the decoding instance with model parallelism or leveraging advanced memory management techniques for LLM KV caches, such as Paged-Attention [32] and QGA [10], enable further scaling of the decoding batch size to nearly compute-bound. 
>  并行规划
>  在解耦后，decoding 的 batch size 受限于 GPU 显存容量，因为它需要为所有活跃请求维护 KVCache
>  使用模型并行拓展 decoding 实例，或者使用高级的 KVCache 管理技术，例如 Paged-Attention 和 GQA，都可以进一步拓展 decoding batch size，使其接近 compute-bound 状态

As the decoding batch size continue to increase to approach the compute-bound, the decoding computation begins to resemble the prefill phase. 
>  随着 decoding batch size 持续增大，到接近 compute-bound, decoding 计算会开始类似于 prefill 阶段

![[pics/DistServe-Fig5.png]]

With this observation, we investigate how the latency and throughput change under different parallelism degrees under large batch conditions in Figure 5: intra-op parallelism reduces latency with diminishing returns, caused by communication and reduced utilization after partitioning. Inter-op parallelism can almost linearly scale the throughput. Hence, when the TPOT SLO is stringent, intra-op parallelism is essential to reduce TPOT to meet latency goals. Beyond this, inter-op parallelism is preferable to enhance throughput linearly.
>  在这个观察下，我们探究了延迟和吞吐在大 batch 时，在不同的并行度下是如何变化的，如 Fig5 所示:
>  算子内并行能够减少延迟，但是随着并行度增加，其吞吐和延迟收益递减，这是因为分片后带来的通信开销以及资源利用率下降，算子间并行则无法减少延迟，但几乎可以线性拓展吞吐
>  因此，当 TPOT SLO 较严格，算子内并行对于减少 TPOT 以满足延迟要求是必须的
>  而在满足 TPOT SLO 的基础上，则更偏好算子间并行，以实现线性拓展吞吐

It is worth noting that when the model can fit into the memory of a single GPU, replication is a competitive option in addition to model parallelism for both prefill and decoding instances, to linearly scale the system's rate capacity. It may also reduce the queuing delay - as indicated by Eq. 1 -by substituting  $R$  with  $R / N$  assuming requests are equally dispatched to  $N$  replicas, at the cost of maintaining additional replicas of the model weights in GPU memory.
>  值得一提的是，当模型可以放入单个 GPU 内存时，除了采用模型并行外，对 decoding 和 prefill 实例采用复制也是一种具有竞争力的方案，这可以线性提升系统的吞吐能力
>  此外，它可以通过将请求均匀地分发到 $N$ 个副本上，减少排队延迟 (将 Eq 1 中的 $R$ 替换为 $R/N$)，代价是在 GPU 显存中维护 $N$ 个模型权重的副本

## 3.3 Practical Problems
We have developed foundational principles for selecting batching and parallelisms for each phase. In this section, we discuss and address several challenges encountered during the practical deployment of disaggregated prefill and decoding phases.
>  我们为每个阶段的批处理和并行方式构建了基本原则
>  本节讨论实际 PD 分离部署时的挑战

**Variable prefill length.**  $\S 3$  has assumed uniform prompt length across requests. In real deployments, depending on the LLM application, the lengths of requests are non-uniform. The non-uniformity can cause pipeline bubbles [28, 36] for prefill instances applying inter-op parallelism because the execution time of pipeline stages across requests of different lengths will vary. This results in slight deviations from the conclusions indicated by using the M/D/1 queue model. To address the problem,  $\S 4$  develops algorithms that search for parallelisms based on workloads, and resort to scheduling to minimize the bubbles (§4.5).
>  可变 prefill 长度
>  实际部署中，请求的 prefill 长度不是均匀的，这个非均匀性会导致 prefill 实例在引用算子间并行时出现气泡，因为流水线阶段跨不同长度的请求的执行时间会不一样 (例如 stage2 处理短请求 microbatch，stage1 处理长请求 microbatch，短请求 microbatch 的 stage2 处理完之后，stage2 的设备还需要等待 stage1 完成处理，因此出现了气泡)
>  这会导致实际结果和使用 M/D/1 队列模型得出的结论不同
>  为了解决该问题，Section 4 提出了基于 workloads 来搜索并行性的算法，并通过调度优化来最小化气泡

**Communication overhead.** Transferring KV caches from prefill to decoding instances incurs notable overheads. For example, the KV cache size of a single 512-token request on OPT-66B is approximately 1.13GB. Assuming an average arrival rate of  $10~\mathrm{rps}$ , we need to transfer 11.3GB data per second—or equivalently 90Gbps bandwidth to render the overhead invisible. 
>  通信开销
>  将 KVCache 从 prefill 实例迁移到 decoding 实例会有显著开销
>  例如，单个 512 token 的请求在 OPT-66B 的 KVCache 大小近似是 1.13GB
>  假设平均到达率为 10rps，我们就需要每秒传输 11.3GB 数据，等价于需要 90Gbps 的带宽才能使得这个开销可忽略

While many modern GPU clusters for LLMs are equipped with InfiniBand (e.g., 800 Gbps), in cases where cross-node bandwidth is limited, DistServe relies on the commonly available intra-node NVLINK, where the peak bandwidth between A100 GPUs is  $600\mathrm{GB / s}$ , again rendering the transmission overhead negligible (see §6.3). However, this requirement imposes additional constraints on the placement of prefill and decoding instances that we take into consideration in the next section.
>  许多 GPU 集群都有 InfiniBand (800Gbps) 用于跨节点通讯，在跨节点带宽受限的情况下，DistServe 依赖于常见的节点内 NVLINK 互联，其峰值带宽可达 600GB/s，使得传输开销可以忽略不计
>  但这也对 prefill 和 decoding 实例的部署位置施加了约束，这是我们需要考虑的

Through the analysis in this section, we identify the workload pattern, placement constraints, SLO requirements, parallelism strategies, and resource allocation as key parameters that create a web of considerations in designing the disaggregated serving system. How to automatically navigate the search space to find the configuration that achieves optimal per-gpu goodput is challenging, and addressed next.
>  我们识别出设计解耦服务系统时，需要考虑的因素包括 workload 模式、放置约束、SLO 要求、并行策略和资源分配
>  如何自动探索这个复杂的配置空间，以找到实现最优 per-gpu goodput 的方案将在下一节解决

# 4 Method
We built DistServe to solve the above challenges. Given the model, workload characteristic, latency requirements, and SLO attainment target, DistServe will determine (a) the parallelism strategies for prefill and decoding instances, (b) the number of each instance type to deploy, as well as (c) how to place them onto the physical cluster. We call the solution a placement. Our goal is to find a placement that maximizes the per-gpu goodput.
>  DistServe 给定模型、workload 特征、延迟需求和 SLO 达成目标，会决定
>  1. prefill 和 decoding 实例的并行策略
>  2. 每个实例类型需要部署的数量
>  3. 如何将实例部署在物理设备
>  我们称这个解决方案为部署方案，我们的目标是找到一个部署方案来最大化 per-gpu goodput

As explained in §3.3, a key design consideration is to manage communications between disaggregated prefill and decoding phases, given varying cluster setups. In this section, we first present two placement algorithms: one for clusters with high-speed cross-node networks (§4.1) and the other for environments lacking such infrastructure (§4.2); the latter introduces additional constraints. 
>  一个关键的设计思考是在给定不同集群设定时，管理 prefill, decoding 阶段之间的通讯
>  我们展示两个放置算法: 一个用于具有高速跨节点网络的集群，另一个用于没有高速跨节点网络的集群，后者引入了额外的约束

We then develop online scheduling optimizations that adapt to the nuances of real-world workloads (§4.3).
>  随后，我们开发了在线调度优化来适应真实工作负载的细微差异

## 4.1 Placement for High Node-Affinity Cluster
On high node-affinity clusters equipped with Infiniband, KV caches transmission overhead across nodes is negligible, DistServe can deploy prefill and decoding instances across any two nodes without constraints. 
>  在节点间具备 Infiniband 的集群中，跨节点的 KVCache 传输开销是可以忽略的
>  DsitServe 可以在没有约束的情况下，跨任意两个节点部署 prefill, decoding 实例

We propose a two-level placement algorithm for such scenarios: we first optimize the parallelism configurations for prefill and decoding instances separately to attain phase-level optimal per-gpu goodput; then, we use replication to match the overall traffic rate.
>  我们为这类情景提出一个两级部署算法:
>  我们首先为 prefill 和 decoding 实例分别优化并行配置，分别得到各个阶段最优的 per-gpu goodput
>  然后我们使用复制来匹配总的流量速率

However, finding the optimal parallel configuration for a single instance type, such as for the prefill instance, is still challenging, due to the lack of a simple analytical formula to calculate the SLO attainment (a.k.a., percentage of requests that meet TTFT requirement), given that the workload has diverse input, output lengths, and irregular arrival patterns. 
>  由于工作负载具有多样的输入、输出长度和不规则的到达模式，缺乏一个简单的解析形式公式来计算 SLO 达成率 (即满足 TTFT 要求的请求比率)，因此为单个实例类型找到最优的并行配置是有挑战的

Gauging the SLO via real-testbed profiling is time-prohibitive. We thus resort to building a simulator to estimate the SLO attainment, assuming prior knowledge of the workload's arrival process and input and output length distributions. Although short-term interval is impossible to predict, the workload pattern over longer timescales (e.g., hours or days) is often predictable [33, 55]. 
>  通过真实实验平台进行 SLO 估计过于耗时，因此我们构建一个模拟器，在已知工作负载的到达过程和输入输出长度分布的情况下，来估计 SLO 达成率
>  尽管短期内的请求模式难以预测，但长期的工作负载模式是可预测的

DistServe fits a distribution from the history request traces and resamples new traces from the distribution as the input workload to the simulator to compute the SLO attainment.
>  DistServe 从历史请求轨迹中拟合出相应的分布，并从中重采样生成新的轨迹作为模拟器的输入 workload，来计算 SLO 达成率

Next, DistServe simply enumerates the placements and finds the maximum rate that meets the SLO attainment target with binary search and simulation trials.
>  然后，DistServe 枚举所有可能的放置方案，使用二分查找和模拟试验，找到能够满足 SLO 达成率目标的最大吞吐率

![[pics/DistServe-Alg1.png]]

>  算法中, `intra_op` 表示算子内并行即张量并行的并行度，它的取值从 `1` 到每个节点的 GPU 数量 `M` ，也就是最高也仅考虑节点内张量并行
>  `inter_op` 表示算子间并行即流水并行的并行度，它的取值从 `1` 到 $\frac {N\times M}{\text{intra\_op}}$，也就是从不进行流水并行到把实例中除张量并行的 GPU 都进行流水并行
>  之后算法对每种配置进行模拟，分别针对 prefill, decode 得到该并行配置下的模拟总 goodput，然后计算 per GPU goodput，和之前的结果进行比较
>  这样遍历完成之后，就得到了 prefill 和 decode 实例能够获得 (模拟) 最高 per GPU goodput 的配置了，他就是结果的最优放置

>  最后的 `n, m` 是 PD 实例的复制数量，它由通讯流量除以每个实例的估计 goodput 得到，实际上就是为了满足请求量而需要复制多少个实例

Algorithm 1 outlines the process. We enumerate all feasible parallel configurations, subject to cluster capacity limit, for both prefill and decoding instances. Then, for a specific prefill phase configuration, we use simu_prefill to simulate and find its maximum goodput via binary search (similarly for using simu_decode for decoding). 
>  算法 1 为 prefill 和 decoding 实例遍历所有可能的并行配置，使用 `simu_prefill, simu_decode` 通过二分查找，模拟并找到最优的 goodput 配置

After determining the optimal parallel configurations for both prefill and decoding instances, we replicate them to achieve the user-required overall traffic rate according to their goodput.
>  为单个实例确定了最优配置之后，其他的实例直接复制该配置即可

The complexity of Algorithm 1 is  $O(NM^2)$  ,with  $N$  as the node limit per instance and  $M$  representing the typical number of GPUs per node in modern clusters (e.g., 8). The search space is manageable and the solving time is under 1.3 minutes in our largest setting, as demonstrated in  $\S 6.5$

**Simulator building.** Algorithm 1 relies on a simulator to estimate the goodput under various SLOs and SLO attainment goals given the workload and the parallelism plan. To build an accurate simulator, we analyze the FLOPs and the number of memory accesses for prefill and decoding phases respectively, and use a latency model to approximate the inference execution time. See details in Appendix A. The simulator aligns well with real profiling results, thanks to the high predictability of DNN workloads [23, 33], verified in  $\S 6.4$
>  算法 1 中的模拟器需要能够在不同 SLO 达成要求下，给定 workload 和并行方案，估计出 goodput
>  我们分析 prefill, decoding 阶段的 FLOPs 和访存数量，使用延迟模型近似推理执行时间来构建模拟器
>  因为 DNN workload 具有高度可预测性，模拟器可以和真实性能剖析结果吻合

By far, we have developed Algorithm 1 assuming we can place the prefill and decoding instance between any two nodes (or on the same node) of the cluster, and the KV cache transmission utilizes high bandwidth network. 
>  目前为止，我们已经提供了为集群中不同 node 针对 prefill, decoding workload 的并行配置寻找算法
>  node 之间的 KVCache 传输依赖于高带宽网络

In many real clusters, GPUs inside a node access to high-bandwidth NVLINK while GPUs distributed across nodes have limited bandwidth. We next develop an algorithm to address this constraint.
>  下一节我们考虑在节点间没有高带宽网络情况下的放置算法

## 4.2 Placement for Low Node-Affinity Cluster
A straightforward solution is to always colocate prefill and decoding instances on the same node, utilizing the NVLINK, which is commonly available inside a GPU node. For large models, e.g. with 175B parameters (350GB), we may be unable to even host a single pair of prefill and decoding instances in an 8-GPU node  $(80G\times 8 = 640G< 350\times 2GB)$  .We incorporate this as additional placement constraints and cooptimize it with model parallelism, presented in Algorithm 2.
>  节点间缺乏高速互联的情况下，简单的想法是在相同节点内共置 prefill 和 decoding 实例，使用节点内网络在实例间传输 KVCache
>  但这要求节点内需要能放下模型参数 x 2 (两个实例各一个模型参数)，我们将这一点作为算法的约束

![[pics/DistServe-Alg2.png]]

>  算法 2 对流水线并行度进行遍历，针对每个流水线并行度，获取每个流水线阶段的节点内 prefill, decode 实例的 TP 并行度的所有可能，然后使用模拟器获取遍历各个配置下的模拟 goodput，和之前的结果比较
>  遍历完成后，就获取了最优并行配置

>  Alg1, Alg2 的本质都是遍历所有可能，找到最优的那个

The key insight is that KV cache transfer occurs exclusively between corresponding layers of prefill and decoding instances. Leveraging inter-op parallelism, we group layers into stages and divide each instance into segments, termed as instance segments, with each segment maintaining one specific inter-op stage. By colocating prefill and decoding segments of the same stage within a single node, we force the transfer of intermediate states to occur only via NVLINK. 
>  prefill 和 decoding 实例之间的 KVCache 传输实际上是在不同实例间对应的层上执行的
>  流水线并行将每个实例划分为多个阶段，或称为 segments
>  因此我们可以仍然在节点间执行流水线并行，节点内共置 prefill 和 decoding 实例的 segment，这样 KVCache 的传输就都通过 NVLink 实现

Inside a node, we set the same parallelism and resource allocation for segments of the same instance. Given the typical limitation of GPUs per node (usually 8), we can enumerate possible configurations inside one node and use the simulator to identify the configurations that yield the best goodput.

As outlined in Algorithm 2, we begin by enumerating interop parallelism degrees to get all the possible instance segments. For each segment, we get all possible intra-node parallelism configurations by calling `get_intra_node_configs`. Then we use simulation to find the optimal one and replicate it to satisfy the target traffic rate.

## 4.3 Online scheduling

![[pics/DistServe-Fig6.png]]

The runtime architecture of DistServe is shown in Figure 6. DistServe operates with a simple FCFS scheduling policy. All incoming requests arrive at a centralized controller, then dispatched to the prefill instance with the shortest queue for prefill processing, followed by dispatch to the least loaded decoding instance for decoding steps. This setup, while simple, is optimized with several key enhancements tailored to the nuances of real-world workloads.
>  DistServe Runtime: FCFS 调度，所有请求先到达中心化控制器，控制器分发队列中的 prefill, decoding 请求

**Reducing pipeline bubbles.** To mitigate the pipeline bubbles caused by non-uniform prompt lengths (§3.3), we schedule the requests in a way that balances the execution time across all batches in the pipeline. 
>  流水线并行中，microbatch/batch 之间的 prompts 长度差异过大会导致阶段执行时间不均匀，出现 bubble
>  为了缓解这一点，需要平衡 microbatch/batch 之间的 prompts 长度

This is achieved by noting that, for both prefill and decoding instances, the number of new tokens in the batch is a reliable indicator of the batch's real execution time. For prefill instances, we profile the target model and GPU to figure out the shortest prompt length  $L_{m}$  needed to saturate the GPU. We schedule prefill batches with a total sequence length close to  $L_{m}$ , by either batching multiple requests shorter than  $L_{m}$  or individually scheduling requests longer than  $L_{m}$ . For decoding instances, we set  $L_{m}$  as the largest batch size.
>  注意到 batch 中的新 tokens 数量和 batch 的实际执行时间强相关
>  对于 prefill，我们先剖析模型能够饱和 GPU 的最短 prompt 长度 $L_m$，然后组建 prefill batch 的方式就是让整个 batch 的序列长度接近 $L_m$ (可以 batching 多个短的 prefill request 或者调度少量的长的 prefill request)
>  对于 decoding，就让 $L_m$ 为最大 batch size (每个 decoding 请求仅计算一个新 token)

**Combat busrtiness.** Burstiness in workloads can cause a deluge of KV caches to transfer from prefill to decoding instances, risking memory overload on decoding instances. To circumvent this, DistServe employs a "pull" method for KV cache transmission rather than a "push" approach -decoding instances fetch KV cache from prefill instances as needed, using the GPU memory of prefill instances as a queuing buffer. This way, the prefill instance can continue handling other prefill jobs by simply retaining the KV Cache in the GPU memory after processing the prompt. Hence, each type of instance operates at its own pace without complex coordination.
>  突然爆发的 workload 可能会导致 prefill 实例一下子给 decoding 实例传输太多 KV cache
>  为了避免这一点，prefill 实例不会自行将 KVCache 推送到 decoding 实例，decoding 实例自己按需拉取 KVCache，这样可以把 prefill 实例的显存作为排队缓存使用

**Replaning.** The resource and parallelism plan in DistServe is optimized for a specific workload pattern, which may become suboptimal if the workload pattern changes over time. DistServe implement periodic replanning. A workload profiler monitors key parameters such as the average input and output length of the requests, the average arrival rate, etc. If a significant pattern shift is detected, DistServe will trigger a rerun of the placement algorithm based on recent historical data. This process is expedient -the proposed algorithm runs in seconds (§6.5) and reloading LLM weights can be completed within minutes - far shorter than the hourly scale at which real-world workload variations tend to occur.
>  资源和并行计划是针对特定 workload pattern 优化的
>  为了应对 workload pattern 的缓慢变化，需要定时重规划
>  workload profiler 会监控关于 workload pattern 的关键参数，例如平局到达率，请求的平均输入输出长度，发现显著变化，就重新计算配置
>  计算配置可以在几秒内完成，参数的 resharding 也可以在几分钟内完成

**Preemption and fault tolerance.** DistServe does not implement advanced runtime policies like preemption [26] and fault tolerance [58], which are complementary to disaggregation. Nevertheless, we discuss how they fit into DistServe. In DistServe, the FCFS policy can lead to a "convoy effect", where longer requests block shorter ones in the prefill stage. Incorporating preemptive strategies, as suggested in existing literature [53], could enhance efficiency and is feasible within our system's architecture. While not a primary focus in the current DistServe, fault tolerance is a critical aspect for consideration. In traditional colocation-and replication-based systems, a fault in one instance typically does not disrupt other replica instances. However, in DistServe, the dependency between prefill and decoding instances introduces the risk of fault propagation. For example, a fault in a single decoding instance mapped to multiple prefill instances could potentially cripple the entire service and cluster. We leave both as future work.
>  目前没有实现抢占和容错，这里进行讨论
>  FCFS 会导致 “护送效应”: 长请求在 prefill 阶段阻塞短请求，因此抢占可以提高效率
>  其他系统中，实例上的错误通常不会干扰其他实例，但 DistServe 的 prefill 和 decode 实例存在依赖，例如多个 prefill 实例所期待的 decoding 实例出错，就导致了多个 perfill 实例的请求都无法完成

>  最简单的容错就是通过 heatbeat，如果 decoding 实例宕机，就重新跑decoding
>  随之而来的问题是 KVCache 丢失怎么办，那就需要 perfill 实例重传输，这进而要求 perfill 实例在整个请求完成之前需要备份 KVCache

# 5 Implementation
DistServe is an end-to-end distributed serving system for LLMs with a placement algorithm module, a RESTful API frontend, an orchestration layer, and a parallel execution engine. The algorithm module, frontend, and orchestration layer are implemented with 6.5K lines of Python code. The parallel execution engine is implemented with 8.1K lines of $\mathrm{C + + / CUDA}$  code.

The placement algorithm module implements the algorithm and the simulator mentioned in  $\S 4$  which gives the placement decision for a specific model and cluster setting. The frontend supports an OpenAI API-compatible interface where clients can specify the sampling parameters like maximum output length and temperature. The orchestration layer manages the prefill and decoding instances, responsible for request dispatching, KV cache transmission, and results delivery. 
>  placement algorithm module 实现并行配置搜索算法和模拟器
>  frontend 支持 OpenAI API-compatible 接口
>  编排层管理 PD 实例、请求调度、KVCache 传输、返回结果

It utilizes NCCL [6] for cross-node GPU communication and asynchronous CudaMemory for intra-node communication, which avoids blocking the GPU computation during transmission. 

Each instance is powered by a parallel execution engine, which uses Ray [35] actor to implement GPU workers that execute the LLM inference and manage the KV Cache in a distributed manner. It integrates many recent LLM optimizations like continuous batching [54], FlashAttention [20], PagedAttention [32] and supports popular open-source LLMs such as OPT [56] and LLaMA [51].
>  每个实例的执行依赖并行执行引擎，执行引擎使用 Ray actor 实现 GPU workers

# 6 Evaluation
In this section, we evaluate DistServe under different sizes of LLMs ranging from 13B to 175B and various application datasets including chatbot, code-completion, and summarization. 
>  模型从 13B 到 175B

The evaluation shows that DistServe consistently outperforms the current state-of-the-art system across all the settings (6.2). Specifically, DistServe can handle up to 7.4 $\times$ higher rates and 12.6 $\times$ more stringent SLO while meeting the latency requirements for over 90% requests. Additionally, we analyze the latency breakdown in DistServe to show the communication overhead is insubstantial thanks to our bandwidth-aware placement algorithm (6.3) and do ablation studies of our techniques (6.4). Finally, we profile the execution time of our placement algorithm (6.5).

## 6.1 Experiments Setup
**Cluster testbed.** We deploy DistServe on a cluster with 4 nodes and 32 GPUs. Each node has 8 NVIDIA SXM A100-80GB GPUs connected with NVLINK. The cross-node bandwidth is 25Gbps. Due to the limited cross-node bandwidth, we use the low node-affinity placement algorithm (2) for DistServe in most of the experiments except for the ablation study (6.4) which uses simulation.
>  4 节点，32 GPU，跨节点带宽低

Table 1: Workloads in evaluation and latency requirements.  

<table><tr><td>Application</td><td>Model Size</td><td>TTFT</td><td>TPOT</td><td>Dataset</td></tr><tr><td>Chatbot OPT-13B</td><td>126GB</td><td>0.25s</td><td>0.1s</td><td>ShareGPT [8]</td></tr><tr><td>Chatbot OPT-66B</td><td>132GB</td><td>2.5s</td><td>0.15s</td><td>ShareGPT [8]</td></tr><tr><td>Chatbot OPT-175B</td><td>350GB</td><td>4.0s</td><td>0.2s</td><td>ShareGPT [8]</td></tr><tr><td>Code Completion OPT-66B</td><td>132GB</td><td>0.125s</td><td>0.2s</td><td>HumanEval [14]</td></tr><tr><td>Summarization OPT-66B</td><td>132GB</td><td>15s</td><td>0.15s</td><td>LongBench [13]</td></tr></table>

**Model and workloads setup.** Similar to prior work on LLM serving [32], we choose the OPT [56] model series, which is a representative LLM family widely used in academia and industry. Newer GPT model families are adopting memory-efficient attention mechanisms like GQA [10] and MQA [44]. DistServe will show better performance on these models because the transmission overhead is lower due to the decrease in KV cache size. We choose OPT which uses the classic MHA [52] to put enough pressure on the transmission overhead. We use FP16 precision in all experiments.
>  OPT 系列，FP16

For workloads, as shown in Table 1, We choose three typical LLM applications and set the SLOs empirically based on their service target because there exists no available SLO settings for these applications as far as we know. For each application, we select a suitable dataset and sample requests from it for evaluation. Since all the datasets do not include timestamps, we generate request arrival times using Poisson distribution with different request rates. Due to the space limit, we test the chatbot workload on all three OPT models and the other two workloads on OPT-66B, which matches the largest size in the recent open-source LLM series [51].
>  依据经验为不同数据集设定 SLO 目标
>  使用不同请求率的 Possion 生成请求到达时间分布

**Chatbot** [1]: We use the ShareGPT dataset [8] for the chatbot application, which is a collection of user-shared conversations with ChatGPT. For OPT-13B, the TTFT SLO is set to 0.25s for responsiveness and the TPOT SLO is set to 0.1s which is higher than the normal human read speed. For OPT-66B and OPT-175B, we slightly relax the two SLOs due to the increase in model execution latency. 
>  ShareGPT 数据集，TTFT SLO = 0.25s, TPOT SLOT=0.1s

**Code completion** [14]: We use the HumanEval [14] dataset for the code completion task. It includes 164 programming problems with a function signature or docstring which is used to evaluate the performance of code completion models. Since the code completion model is used as a personal real-time coding assistant, we set both SLOs to be stringent.
>  HumanEval 数据集

![[pics/DistServe-Fig7.png]]

**Summarization** [5]: It is a popular LLM task to generate a concise summary for a long article, essay, or even an academic paper. We use LongBench [13] dataset which contains the summarization task. As shown in Figure 7, LongBench has much longer input lengths than the other two datasets. So we set a loose TTFT SLO but require a stringent TPOT.
 >  LongBench 数据集

**Metrics.** We use SLO attainment as the major evaluation metric. Under a specific SLO attainment goal (say,  $90\%$ ), we are concerned with two things: the maximum per-GPU goodput and the minimal SLO the system can handle. We are particularly interested in an SLO attainment of  $90\%$  (indicated by the vertical lines in all curve plots), but will also vary the rate and latency requirements to observe how the SLO attainment changes. We also include the results in the Appendix for an SLO attainment of  $99\%$  to show the system performance under a more stringent SLO attainment target.
>  SLO 达成率完成的前提下，我们关心最大 per GPU goodput
>  我们还关心系统可以支持的最小 SLO (例如 95%，要求再高系统就崩溃)

**Baselines.** We compare DistServe to two baseline systems:
- **vLLM** [32]: vLLM is a representative LLM serving system widely used in both academia and industry. It supports continuous batching [54] to increase throughput and paged-attention [32] to reduce memory fragmentation during KV cache allocation. However, it colocates the prefill and decoding computation to maximize the overall system throughput and struggles to meet the latency requirements cost-efficiently. Since vLLM only supports intra-op parallelism, we follow previous work [32] to set intra-op equals 1, 4, and 8 for the three OPT models, respectively.
>  vLLM 吞吐高，但是 SLO 达成率约束下的 goodput 则不然，故无法 cost-efficiently 满足延迟要求

- **DeepSpeed-MII** [3]: DeepSpeed Model Implementations for Inference (MII) supports chunked-prefill by decomposing long prompts into smaller chunks and composing with short prompts to exactly fill a target token budget. It mitigates but cannot eliminate the prefill-decoding interference caused by the long prefill job. We set its intra-op the same as vLLM for OPT-13B and OPT-66B for a fair comparison. However, DeepSpeed-MII cannot serve OPT-175B whose vocab_size = 50272 because its underlying kernel implementation requires vocab_size/intra_op is a multiple of 8 where intra-op equals 8 does not satisfy. Setting intra-op equals 4 can satisfy this requirement but will cause the out-of-memory issue.

## 6.2 End-to-end Experiments
In this Section, we compare the end-to-end performance of DistServe against the baselines on real application datasets.

![[pics/DistServe-Fig8.png]]

**Chatbot**. We evaluate the performance of DistServe on the chatbot application for all three OPT models. The first row of Figure 8 illustrates that when we gradually increase the rate, more requests will violate the latency requirements and the SLO attainment decreases. The vertical line shows the maximum per-GPU rate the system can handle to meet latency requirements for over  $90\%$  of the requests.
>  请求率增大，SLO 达成率就会下降

On the ShareGPT dataset, DistServe can sustain  $2.0\times -4.6\times$  higher request rate compared to vLLM. This is because DistLLM eliminates the prefill-decoding interference through disaggregation. 

Two phases can optimize their own objectives by allocating different resources and employing tailored parallelism strategies. Specifically, by analyzing the chosen placement strategy for 175B, we find the prefill instance has inter-op = 3, intra-op = 3; and the decoding instance has inter-op = 3, intra-op = 4. 

Under this placement, DistServe can effectively balance the load between the two instances on ShareGPT, meeting latency requirements at the lowest cost. This non-trivial placement strategy is challenging to manually find, proving the effectiveness of the algorithm. 

In the case of vLLM, collocating prefill and decoding greatly slows down the decoding phase, thereby significantly increasing TPOT. Due to the stringent TPOT requirements of chatbot applications, although vLLM meets the TTFT SLO for most requests, the overall SLO attainment is dragged down by a large number of requests that violate the TPOT SLO. 
>  vLLM 的 TPOT 被 prefill 拖累

Compared to DeepSpeed-MII, DistServe can sustain  $1.6\times- 7.4\times$  higher request rate. DeepSpeed-MII shows better performance on larger models because the prefill job is larger and chunked-prefill mitigates the interference to some extent. However, due to the reasons discussed in §2.3, chunked prefill is slower than full prefill, so it struggles to meet the TTFT SLO as a sacrifice for better TPOT.

The second row of Figure 8 indicates the robustness to the changing latency requirements of the two systems. We fix the rate and then linearly scale the two latency requirements in Table 1 simultaneously using a parameter called SLO Scale. As SLO Scale decreases, the latency requirement is more stringent. We aim to observe the most stringent SLO Scale that the system can withstand while still achieving the attainment target. Figure 8 shows that DistServe can achieve  $1.8 \times -3.2 \times$  more stringent SLO than vLLM and  $1.7 \times -1.8 \times$  more stringent SLO than DeepSpeed-MII, thus providing more engaging service quality to the users.
>  SLO Scale 越低，延迟要求越严格

![[pics/DistServe-Fig9.png]]

**Code completion.** Figure 9(a) shows the performance of DistServe on the code completion task when serving OPT-66B. DistServe can sustain  $5.7 \times$  higher request rate and  $1.4 \times$  more stringent SLO than vLLM. Compared to DeepSpeed-MII, DistServe can sustain  $1.6 \times$  higher request rate and  $1.4 \times$  more stringent SLO. As a real-time coding assistant, the code completion task demands lower TTFT than chatbot, this leads to both systems ultimately being constrained by the TTFT requirement. However, in comparison, by eliminating the interference of the decoding jobs and automatically increasing intra-operation parallelism in prefill instances through the searching algorithm, DistServe reduces the average latency of the prefill jobs, thereby meeting the TTFT requirements of more requests.
>  分离的 prefill 实例可以增大 TP 并行度，因此 PD 分离也有利于 TTFT 减小

**Summarization.** Figure 9(b) shows the performance of DistServe on the summarization task when serving OPT-66B. DistServe achieves  $4.3 \times$  higher request rate and  $12.6 \times$  more stringent SLO than vLLM. Compared to DeepSpeed-MII, DistServe achieves  $1.8 \times$  higher request rate and  $2.6 \times$  more stringent SLO. The requests sampled from LongBench dataset have long input lengths, which brings significant pressure to the prefill computation. However, due to the loose requirement of TTFT for the summarization task, the TPOT service quality becomes particularly important. Since vLLM collocates prefill and decoding phases, it experiences a greater slowdown in the decoding phase with long prefill jobs and fails to meet the TPOT requirement.

The results above are all under the  $90\%$  SLO attainment target. We observe that DistServe can have better performance under a more stringent attainment target (say,  $99\%$  ) and present the results in Appendix C.

## 6.3 Latency Breakdown
To understand DistServe's performance in detail, we make a latency breakdown of the requests in DistServe. We divide the processing lifecycle of a request in DistServe into five stages: prefill queuing, prefill execution, transmission, decoding queuing, and decoding execution. The total time consumed by all requests in each stage is then summed up to determine their respective proportions in the system's total execution time.
>  DistServe 的 request 处理分为五阶段:
>  排队等 prefill
>  prefill 执行
>  KVCache 传输
>  排队等 decode
>  decode 执行
>  我们分析这五阶段占总和的比例

![[pics/DistServe-Fig10.png]]

Figure 10(a) shows the latency breakdown for the OPT-175B models on the ShareGPT dataset. We chose OPT-175B because the KV Cache transmission is more demanding for larger models. In fact, even for OPT-175B, the KV Cache transmission only accounts for less than  $0.1\%$  of the total latency. Even by examining the CDF of the absolute transmission time shown in Figure 10(b), we observe that over  $95\%$  of requests experience a delay of less than  $30 \text{ms}$ , despite our testbed having only limited cross-node bandwidth. This is due to the algorithm described in §4.2, where we require the prefill and decoding instance to maintain the same stage on one machine, enabling the use of intra-node NVLINK bandwidth for transmission, thus significantly reducing transmission delay.
>  传输仅占 0.1%，并且 95% 请求的传输延迟都低于 30ms
>  这归功于 prefill 和 decode segment 的共置，利用了 NVLINK

## 6.4 Ablation Studies
We study the effectiveness of the two key innovations in DistServe: disaggregation and the placement searching algorithm. In  $\S 6.2$  , we choose the default parallelism setting for vLLM following its original paper [32]. So we implement vLLM++ which enumerates different parallelism strategies and chooses the best. For DistServe, We also compare the placement found by Alg. 2 (DistServe-Low) with the one found by Alg. 1 (DistServe-High) which has fewer searching constraints and assumes high cross-node bandwidth. 
>  对 DistServe 的两个关键创新: PD 分离和 placement 搜索算法进行消融研究
>  vLLM 表示选择原文的默认并行方式
>  vLLM++ 表示枚举选出最优的并行方式

Since vLLM does not support inter-op parallelism and our physical testbed does not have high cross-node bandwidth, we use simulation for this experiment.

<table><tr><td rowspan="2">Rate (req/s)</td><td colspan="2">vLLM</td><td colspan="2">DistServe-Low</td></tr><tr><td>Real System</td><td>Simulator</td><td>Real System</td><td>Simulator</td></tr><tr><td>1.0</td><td>97.0%</td><td>96.8%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>1.5</td><td>65.5%</td><td>65.1%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>2.0</td><td>52.8%</td><td>51.0%</td><td>99.3%</td><td>99.3%</td></tr><tr><td>2.5</td><td>44.9%</td><td>46.1%</td><td>87.3%</td><td>88.3%</td></tr><tr><td>3.0</td><td>36.7%</td><td>38.3%</td><td>83.0%</td><td>84.1%</td></tr><tr><td>3.5</td><td>27.8%</td><td>29.0%</td><td>77.3%</td><td>77.0%</td></tr><tr><td>4.0</td><td>23.6%</td><td>24.1%</td><td>70.0%</td><td>68.9%</td></tr></table>

Table 2: Comparison of the SLO attainment reported by the simulator and the real system under different rates.

**Simulator accuracy.** Noticing that DNN model execution [24] has high predictability, even under parallel settings [33, 59]. We study the accuracy of the simulator in Tab.2. For "vLLM" and "DistServe-Low", we compare the SLO attainment reported by the simulator and by real runs on our testbed under different rates. The error is less than  $2\%$  in all cases, verifying the accuracy of our simulator.
>  根据 Table 2，模拟器报告的 SLO 达成率的误差在所有情况下都小于 2%

![[pics/DistServe-Fig11.png]]

**Results.** Figure 11 shows the performance of the four systems when serving OPT-66B on the ShareGPT dataset. "vLLM++" has the same performance as "vLLM" because we find the default parallelism setting (intra-op=4) has the best per-GPU goodput. This further demonstrates the importance of disaggregation. The interference between the prefill and decoding phases significantly reduces the potential performance improvement through adjusting parallelism. 

In contrast, "DistLLM-High" can achieve further improvements over "DistLLM-Low" because it is not constrained by the deployment constraint that the prefill and decoding instance on one node should share the same model stage. Through disaggregation, we can use tailored parallelism strategies for prefill and decoding instances and optimize their targets without the coupling effects.

>  消融研究结果显示 PD 分离有助于为不同实例制定不同并行方法，如果不分离，调节并行方法得到的潜在性能提升是有限的

## 6.5 Algorithm Running Time

![[pics/DistServe-Fig12.png]]

Figure 12 shows the running time for Alg. 1 (DistServe-Low) and Alg. 2 (DistServe-High) on an AWS m5d.metal instance with 96 cores as the number of GPUs  $(N\times M)$  provided to a single instance increases. 

According to the results, DistServe scales well with the number of GPUs and is independent of the model size. This is because the simulator only simulates discrete events and the running time is the same no matter how big the model is. On the other hand, both algorithms are highly parallelizable, as the searches for different parallelism strategies are independent of each other, allowing the execution time of the algorithms to accelerate almost linearly with more CPU cores. 
>  模拟器仅模拟离散事件，无论模型多大，模拟器的模拟时间都一致

As the number of GPUs increases, the execution time of "Dist-Low" becomes higher than that of "Dist-High". This is because the search for parallelism strategies for prefill and decoding instances in "Dist-High" is independent and can be parallelized. But for "Dist-Low", due to additional restrictions on deployment, we need to enumerate all the possible intra-node parallelism combinations for prefill and decoding instances. Even so, the execution time of the algorithm is in minutes, and since it only needs to be executed once before each redeployment, this overhead is acceptable.
>  并行方案选择算法的运行开销在秒和分钟级别，开销不大

# 7 Discussion
In this paper, we focus on the goodput-optimized setting and propose DistServe under the large-scale LLM serving scenario. As LLMs are widely used and deployed across various service scenarios with different optimization targets and resource limits, it becomes almost impossible to find a one-size-fits-all solution that effectively addresses all aspects of LLM serving. In this section, we discuss the pros and cons of DistServe and potentially better solutions in other scenarios.
>  本文关注的是 goodput

**Throughput-optimized scenarios.** In offline applications that are not latency-sensitive, users typically have lower requirements for response time [45]. This allows serving systems to shift focus towards maximizing overall throughput instead of goodput and the effectiveness of DistServe may be compromised. In this case, techniques such as chunked-prefill with piggyback [3, 9] may be preferred since it can fill each batch to the compute-bound threshold, thereby maintaining higher GPU utilization in every iteration.
>  对于延迟不敏感的离线应用，关注的点就在 throughput 而不是 goodput，此时偏好 chunked-prefill with piggyback

**Resource-constrained scenarios.** Small-scale enterprises and individual researchers often lack the resources to deploy LLMs on large-scale clusters [45,48]. In resource-constrained scenarios, such as environments with only a few or even a single GPU, the design space for DistServe is significantly limited. It struggles or even fails to adjust the parallel strategies and resource allocation to effectively enhance serving performance. In this case, simpler architectural choices like non-disaggregated systems [3, 32] may reduce deployment complexity and optimize operational efficiency.
>  资源受限的情况下，PD 分离的可用性有限

**Long-context scenarios.** Nowadays, more and more GPT models support extremely long contexts, such as Claude3 [11], Gemini-1.5 [22], and Large World Model (LWM) [34], which all have a 1M context window. In such scenarios, the transmission overhead will increase as the size of the KV cache grows linearly with the prompt length. However, the prefill computation grows quadratically, so the relative duration of transmission and prefill job decreases. Meanwhile, a longer context further exacerbates the disparity in computational demands between prefill and decoding jobs, leading to increased interference between them. Therefore, the disaggregation approach proposed in DistServe remains promising in long-context serving.
>  超长上下文场景
>  KVCache 的传输开销随着 prompt 长度线性增长，同时 prefill 计算开销随着 prompt 长度二次增长，因此传输开销的相对占比实际上是下降
>  超长上下文场景下，prefill 和 decode 的计算特性差距更大，PD 分离更显必要

# 8 Related Work
**Inference serving.** There has been plenty of work on inference serving recently. They range from general-purpose production-grade systems like TorchServe [7] and NVIDIA Triton [19] to systems optimized specifically for Transformer-based LLMs [9, 18, 21, 33, 50, 53, 54, 60]. Among them, Orca [54] introduces continuous batching to increase throughput. vLLM [32] proposes paged-attention for fine-grained KV cache management. SARATHI [9] suggests a chunked-prefill approach, splitting a prefill request into chunks and piggybacking decoding requests to improve hardware utilization. FastServe [53] implements iteration-level preemptive scheduling to mitigate the queuing delay caused by long jobs. However, they all employ a colocation approach for prefill and decoding processing, thus leading to severe interference. There are also concurrent works such as Splitwise [38], Tetri-Infer [27] and DejaVu [49] which adopt similar disaggregation idea to optimize LLM inference, further confirming the effectiveness of this method. Differently, DistServe emphasizes the goodput optimization scenario more and takes a closer look at the aspect of network bandwidth.
>  Orca - continuous batching，提高吞吐
>  vLLM - paged attention，细粒度管理 KVCache
>  SARATHI - chunked-prefill，提高硬件利用率
>  FastServe - 迭代级抢占式调度，避免 convey effect

**Goodput-optimized systems.** Optimizing goodput is a hot topic in DL applications. Pollux [39] improves scheduling performance in DL clusters by dynamically adjusting resources for jobs to increase cluster-wide goodput. Sia [29] introduces a heterogeneous-aware scheduling approach that can efficiently match cluster resources to elastic resource-adaptive jobs. Clockwork [23] and Shepherd [55] provide latency-aware scheduling and preemption to improve the serving goodput, but they only target traditional small models. AlpaServe [33] focuses on LLMs, employing model parallelism to statistically multiplex the GPU execution thus improving the resource utilization. However, it only targets the non-autoregressive generation. DistServe is the first work to optimize the goodput for autoregressive LLM inference.

**Resource disaggregation.** Resource disaggregated systems [17, 25, 43] decouple the hardware resources from the traditional monolithic server infrastructure and separate them into resource pools to manage independently. It allows for more flexible, efficient, and scalable deployment and increases resource utilization. Many applications benefit from a truly disaggregated data center with high-speed network bandwidth and heterogenous hardware support [12, 30, 57]. DistServe shares the concept by disaggregating its system components, allowing for independent resource scaling and management.

**Model parallelism for training.** DistServe is orthogonal to the large body of work on model parallelism in training [28, 36, 40, 46, 59]. As described in §3.3, inference-serving workloads have unique characteristics not found in training settings. Where these systems do intersect with DistServe, is in their methods for implementing model parallelism along various dimensions. DistServe can integrate new parallelism optimizations into its placement searching algorithm.

# 9 Conclusion
We present DistServe, a new LLM serving architecture that disaggregates the prefill and decoding computation. DistServe maximizes the per-gpu goodput – the maximum request rate that can be served adhering to the SLO attainment goal for each GPU provisioned, hence resulting in up to  $7.4 \times$  lower cost per LLM query with guaranteed satisfaction of SLOs. Our findings affirm that as latency becomes an increasingly important metric for LLM services, prefill and decoding disaggregation is a vital strategy in promising improved performance and service quality guarantees.
>  DistServe 的优化目标是 goodput - 满足 SLO 延迟需求下的吞吐

# A Latency Model for LLM Inference
To accurately simulate the goodput of different placement strategies, we use an analytical model to predict the execution time of the prefill and decoding phases in LLM inference.
>  模拟器使用了一个解析式模型来预测 prefill, decoding 阶段的执行时间

In modern LLM serving systems [18, 32, 53], memory-bound operations like Softmax and LayerNorm are usually fused with matrix multiplication kernels for efficiency. Thus the GEMMs dominate the overall latency and our analysis primarily focuses on them.
>  memory-bound 算子例如 Softmax 和 LayerNorm 通常和 GEMM 混合，故 GEMM 通常主导延迟，我们聚焦于 GEMM 分析

## A.1 Symbol Definition
Here are symbols related to the architecture of the model:

- h: hidden size 
- n: number of heads :
- s: head size  $(h = n\cdot s)$  
- m: FFN intermediate size

Note: If tensor parallelism is used,  $h,n$  ,and  $m$  should be divided by the tensor parallelism size.

Below are symbols that characterize the batch to be executed:

- $B$  : batch size  
- $l_0,l_1,\ldots ,l_{B -1}$  : input length of each request within the batch 
- $t$ number of tokens in the batch $t = \sum_{i=0}^{B-1}l_i$
- $t_2$: squared sum of the input lengths $t_2 = \sum_{i=0}^{B-1}l_i^2$
- $b$: block size in the attention kernel. This parameter is used in FlashAttention [20], a common kernel optimization technique adopted by current LLM serving systems.

## A.2 Prefill Phase Latency Modeling
Since the attention operation uses specially optimized kernels, we first discuss the other four matrix multiplications in the prefill phase:

<table><tr><td>GEMM Name</td><td>Shape of M</td><td>Shape of N</td></tr><tr><td>QKV Linear</td><td>(t,h)</td><td>(h,3h)</td></tr><tr><td>Attn Output</td><td>(t,h)</td><td>(h,h)</td></tr><tr><td>FFN Input</td><td>(t,h)</td><td>(h,m)</td></tr><tr><td>FFN Output</td><td>(t,m)</td><td>(m,h)</td></tr></table>

The arithmetic intensity (AI) of these operations is  $O(t)$  On NVIDIA A100-80GE GPU, it is compute-bound when AI is over 156. Since  $t$  usually can reach several hundred in real cases, all of these operations are compute-bound. 
>  prefill 阶段涉及的 GEMM 的算术密度为 $O (t)$ ，即和 batch 内 tokens 数量成正比
>  $t$ 通常为几百，因此 A100 上，这些 GEMM 都是 compute-bound

Therefore, we can model the latency of these operations according to the total FLOPs:

$$
T_{1} = C_{1}\cdot (4th^{2} + 2thm)
$$

>  因此直接根据 FLOPs 建模延迟

>  对于 `M x K @ K x N` 的 GEMM，每个元素需要 `K` 次乘法和 `K-1` 次加法，即 `2K-1` FLOPs，因此 GEMM 的 FLOPs 为 ` (2K-1)MN `，约等于 ` 2KMN `

Next, we discuss the prefill attention operation with FlashAttention [20] optimization. Since the attention only operates among the tokens in the same request, current implementations launch attention kernels for each request in the same batch. 
>  上面考虑的 GEMM 的延迟
>  接着考虑 FlashAttention 的延迟，Attention 的延迟需要逐 request 计算

For one attention head and a request with $l$  tokens, the attention kernel needs to perform a total of  $2sl + 3sl\cdot (l / b)\approx 3sl\cdot (l / b)$  memory reads and writes, alongside  $2sl^2 +sl(l / b)\approx 2sl^2$  FLOPs. So the AI is  $2b / 3 = 10.677$  (when  $b = 16$  or 21.333 (when  $b = 32$  ), indicating that it is a memory-bound operation on A100 GPU. Therefore, the whole attention layer latency (including all requests and all heads) can be modeled as:

$$
T_{2} = C_{2}\cdot n\cdot \sum_{i = 0}^{B -1}\frac{3sl_{i}^{2}}{b} = C_{2}\cdot \frac{3nst_{2}}{b} = C_{2}\cdot \frac{3ht_{2}}{b}
$$

>  FlashAttention 为 memory-bound，因此以内存读写次数建模延迟

Overall, the latency of the prefill phase can be modeled as:

$$
T_{Prefill} = C_{1}\cdot (4th^{2} + 2thm) + C_{2}\cdot \frac{3ht_{2}}{b} +C_{3}
$$

We use  $C_3$  to quantify other overheads like Python Runtime, system noise, and so on. Then we use profiling and interpolation to figure out the values of  $C_1,C_2$  ,and  $C_3$

>  式子中的常数使用性能剖析来确定

## A.3 Decoding Phase Latency Modeling
Similarly, we first focus on the following GEMMs in the decoding phase:

<table><tr><td>GEMM Name</td><td>Shape of M</td><td>Shape of N</td></tr><tr><td>QKV Linear</td><td>(B,h)</td><td>(h,3h)</td></tr><tr><td>Attn Output</td><td>(B,h)</td><td>(h,h)</td></tr><tr><td>FFN Input</td><td>(B,h)</td><td>(h,m)</td></tr><tr><td>FFN Output</td><td>(B,m)</td><td>(m,h)</td></tr></table>

The AI of these operations is  $O(B)$ . $B$  is limited by the GPU memory size and stringent latency requirements, so in existing serving scenarios, these operations are memory-bound. The total memory reads and writes is  $8Bh + 4h^{2} + 2hm + 2Bm$  and since  $h$  and  $m$  are usually significantly larger than  $B$  ,we can model the latency as:

$$
T_{3} = C_{4}\cdot (4h^{2} + 2hm)
$$

As for the decoding attention operation, for one attention head and a request with  $l$  generated tokens, it needs to perform  $3sl$  memory reads and writes, alongside  $2sl$  FLOPs. It is memory-bound, so we can model the latency of decoding attention as:

$$
T_{4} = C_{5}\cdot n\cdot 3s\sum_{i = 0}^{B -1}l_{i} = C_{5}\cdot 3ht
$$

Summing up, the latency of the decoding phase is:

$$
T_{Decoding} = C_{4}\cdot (4h^{2} + 2hm) + C_{5}\cdot 3ht
$$

>  decoding 阶段，GEMM 和 Attention 都是 memory-bound，因此使用内存读写来建模延迟

Here we do not introduce the overhead term (like  $C_3$  in the profiling stage) because  $4h^{2} + 2hm$  is already a constant, and the overhead can be put into  $C_4$  . Similarly, we use profiling and interpolation to figure out the values of  $C_4$  and  $C_5$

# B DistServe Placements in End-to-end Experiments
Table 3 shows the tensor parallelism (TP) and pipeline parallelism (PP) configurations for prefill and decoding instances chosen by DistServe in the end-to-end experiments  $\S 6.2$

Table 3: The parallelism strategies chosen by DistServe in the end-to-end experiments.  

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="2">Prefill</td><td colspan="2">Decoding</td></tr><tr><td>TP</td><td>PP</td><td>TP</td><td>PP</td></tr><tr><td>OPT-13B</td><td>ShareGPT</td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>OPT-66B</td><td>ShareGPT</td><td>4</td><td>1</td><td>2</td><td>2</td></tr><tr><td>OPT-66B</td><td>LongBench</td><td>4</td><td>1</td><td>2</td><td>2</td></tr><tr><td>OPT-66B</td><td>HumanEval</td><td>4</td><td>1</td><td>2</td><td>2</td></tr><tr><td>OPT-175B</td><td>ShareGPT</td><td>3</td><td>3</td><td>4</td><td>3</td></tr></table>

# C End-to-end Results under  $99\%$  SLO attainment
Figure 13 and Figure 14 show the end-to-end performance between DistServe and baselines with the same setup in  $\S 6.2$  except that the SLO attainment goal is changed to  $99\%$ . We can see that under a more stringent SLO attainment goal, compared to vLLM, DistServe can still sustain  $3\times -8\times$  higher rate and  $1.24\times -6.67\times$  more stringent SLO. When compared to DeepSpeed-MII, DistServe can achieve  $1.32\times -8\times$  higher rate and  $1.20\times -1.58\times$  more stringent SLO.

