# Abstract
Large-scale Transformer-based models trained for generation tasks (e.g., GPT-3) have recently attracted huge interest, emphasizing the need for system support for serving models in this family. Since these models generate a next token in an autoregressive manner, one has to run the model multiple times to process an inference request where each iteration of the model generates a single output token for the request. However, existing systems for inference serving do not perform well on this type of workload that has a multi-iteration characteristic, due to their inflexible scheduling mechanism that cannot change the current batch of requests being processed, requests that have finished earlier than other requests in a batch cannot return to the client, while newly arrived requests have to wait until the current batch completely finishes.
>  语言模型以自回归方式生成下一个 token，因此在处理一个推理请求的时候，需要运行模型多次，每次迭代生成单个输出 token
>  现存的推理服务系统对于这类带有多轮迭代性质的 workload 表现不好，因为它们的调度机制不灵活，不能更改正在处理的请求的批次
>  在一批请求中，那些较早完成的请求无法立即返回给客户端，而新到的请求必须等到当前批次完全完成

In this paper, we propose iteration-level scheduling, a new scheduling mechanism that schedules execution at the granularity of iteration (instead of request) where the scheduler invokes the execution engine to run only a single iteration of the model on the batch. 
>  本文提出迭代级别的调度，在迭代的粒度调度执行，而不是在请求的粒度调度执行，该调度下，调度器仅调用执行引擎对一批请求运行模型的一次迭代

In addition, to apply batching and iteration-level scheduling to a Transformer model at the same time, we suggest selective batching, which applies batching only to a selected set of operations. 
>  此外，为了同时对 Transformer 模型同时应用 batching 和 iteration-level scheduling，我们提出了选择性批处理，它仅对一组选定的操作进行批处理

Based on these two techniques, we have implemented a distributed serving system called ORCA, with additional designs for scalability to models with hundreds of billions of parameters. Our evaluation on a GPT-3 175B model shows that ORCA can significantly outperform NVIDIA FasterTransformer in terms of both latency and throughput:  $36.9\times$  throughput improvement at the same level of latency.
>  基于这两个技术，我们实现了一个分布式服务系统 ORCA，能够拓展到 100B+ 的模型
>  ORCA 在延迟和吞吐方面都优于 NVIDIA FasterTransformer: 在相同级别的延迟下，吞吐提高了 36.9x

# 1 Introduction
Language generation tasks are becoming increasingly paramount to many types of applications, such as chatbot [9, 52], summarization [41,45,54], code generation [13], and caption generation [65, 66]. Moreover, recent works published by AI21 Labs [37], DeepMind [26,48], Google [15,21,63], Meta Platforms [10,67], Microsoft [50], Microsoft & NVIDIA [59], and OpenAI [12] have reported that every language processing task, including translation [11, 17], classification [20, 53], question-answering [32, 33, 40] and more, can be cast as a language generation problem and have shown great improvements along this direction. The rise of generative models is not limited to the language domain; the AI community has also given growing interest to generation problems in other domains such as image, video, speech, or a mixture of multiple domains [19,38,51,62]. At the heart of generative models lies the Transformer architecture [60] and its variants [15,47-49]. By relying on the attention mechanism [60], Transformer models can learn better representations where each element of the sequence may have a direct connection with every other element, which was not possible in recurrent models [25]. 
>  生成模型的核心架构是 Transformer，通过注意力机制，Transformer 模型可以学习到更优的表示，其中序列中的每个元素都可以直接与序列中的其他元素建立连接，这是传统 RNN 无法实现的

To use generative models in real-world applications, we often delegate the inference procedure to a separate service responsible for ML inference serving. The growing demands for this service, which should provide inference results for client requests at low latency and high throughput, have facilitated the development of inference serving systems such as Triton Inference Server [7] and TensorFlow Serving [42]. These systems can use a separately-developed DNN execution engine to perform the actual tensor operations. For example, we can deploy a service for language generation tasks by using a combination of Triton and FasterTransformer [4], an execution engine optimized for the inference of Transformer-based models. In this case, Triton is mainly responsible for grouping multiple client requests into a batch, while FasterTransformer receives the batch from Triton and conducts the inference procedure in the batched manner.
>  推理服务应该为客户端请求提供低延迟和高吞吐
>  推理服务系统例如 Triton Inference Server, TensorFlow Serving 可以使用单独开发的 DNN 执行引擎来执行实际的张量操作，例如我们可以结合 DNN 和 FasterTransformer 来部署语言生成任务
>  在这种情况下，Triton 负责将多个客户端请求组合为 batch, FasterTransformer 从 Triton 接收 batch，并以批处理的方式进行推理

Unfortunately, we notice that the existing inference systems, including both the serving system layer and the execution engine layer, have limitations in handling requests for Transformer-based generative models. Since these models are trained to generate a next token in an autoregressive manner, one should run the model as many times as the number of tokens to generate, while for other models like ResNet [24] and BERT [18] a request can be processed by running the model once. 
>  现存的推理系统，包括了服务系统层和执行引擎层在对 Transformer-based 生成模型的计算中都存在限制
>  因为这类模型是以自回归方式生成下一个 token，故我们需要运行模型多次，一个一个生成 token
>  但对于其他模型，例如 ResNet, BERT，一个请求只需运行一次模型即可处理

That is, in order to process a request to the generative model, we have to run multiple iterations of the model; each iteration generates a single output token, which is used as an input in the following iteration. Such multi-iteration characteristic calls into question the current design of inference systems, where the serving system schedules the execution of the engine at the granularity of request. Under this design, when the serving system dispatches a batch of requests to the engine, the engine returns inference results for the entire batch at once after processing all requests within the batch. As different client requests may require different numbers of iterations for processing, requests that have finished earlier than others in the batch cannot return to the client, resulting in an increased latency. Requests arrived after dispatching the batch also should wait for processing the batch, which can significantly increase the requests' queueing time.
>  也就是说，为了处理对生成模型的请求，我们需要将模型运行多次迭代，每次迭代生成一个输出 token，这个输出 token 会作为下一次迭代的输入
>  当前系统都是在 request 的粒度上调度引擎的执行，在该设计下，当服务系统层将 batch of requests 发送给引擎时，引擎需要在处理完 batch 内的所有 requests 之后才返回处理结果
>  由于不同的 request 可能需要不同数量的迭代才能完成处理，较早完成的 request 不能先被返回给客户端，导致延迟增加
>  以及在 batch 发送之后到达的 request 需要等待上一个 batch 处理完成，显著提高了 requests 的等待时间

In this paper, we propose to schedule the execution of the engine at the granularity of iteration instead of request. In particular, the serving system invokes the engine to run only a single iteration of the model on the batch. As a result, a newly arrived request can be considered for processing after waiting for only a single iteration of the model. The serving system checks whether a request has finished processing after every return from the engine - hence the finished requests can also be returned to the clients immediately.
>  我们提出在 iteration 的粒度而不是 request 的粒度调度引擎的执行
>  服务系统一次调度引擎运行 batch 的一次迭代 (而不是完成 batch 的所有迭代)
>  这样，新到达的 request 可以在等待一次迭代之后就有可能被处理
>  服务系统层在引擎每次返回后检查是否有 request 完成处理，进而可以将完成处理的 request 直接返回给 client

Nevertheless, a noticeable challenge arises when we attempt to apply batching and the iteration-level scheduling at the same time. Unlike the canonical request-level scheduling, the proposed scheduling can issue a batch of requests where each request has so far processed a different number of tokens. In such a case, the requests to the Transformer model cannot be processed in the batched manner because the attention mechanism calls for non-batchable tensor operations whose input tensors have variable shapes depending on the number of processed tokens.
>  实现这个思想的一个挑战是我们试图同时应用 batching 和 iteration-level scheduling
>  我们提出的调度和传统的 request-level 调度不同，在该调度中，我们可以发出 a batch of requests，且其中每个 request 到目前为止处理的 token 数量各不相同
>  在这种情况下，对 Transformer 模型的请求难以按照批处理的方式处理，因为 attention 机制需要非批处理的张量操作，其输入张量的形状会根据已处理的 token 数量而变化

>  Transformer 在生成每个 token 时，需要访问之前所有 token 的 KVCache
>  如果每个请求的历史长度不同，则它们的 KVCache 长度也不同 (`seq_leng`)
>  这将导致 KVCache 张量的内部形状不一致，即 `[batch_size, num_heads, seq_len, head_dim]` 中的 `seq_len` 不同，无法堆叠为一个统一的 batch 张量，这样原本可以批处理的矩阵运算就必须逐个处理

To address this challenge, we suggest to apply batching only to a selected set of operations, which we call selective batching. By taking different characteristics of operations into account, selective batching shifts the batch and processes each request individually for the Attention' operation while applying batching to other operations of the Transformer model. We observe that the decision not to batch the executions of the Attention operation has only a small impact on efficiency. Since the Attention operation is not associated with any model parameters, applying batching to Attention has no benefit of reducing the amount of GPU memory reads by reusing the loaded parameters across multiple requests.
>  为了解决这个挑战，我们建议仅对一组选定的 operations 应用 batching，我们称之为 selective batching
>  通过考虑不同 operations 的特性, selective batching 将 batch 分开，单独处理每个 request 的 Attention 计算，同时对 Transformer 模型的其他计算应用批处理
>  我们观察到，不将 Attention 进行批处理对性能的影响很小，因为 Attention 不涉及任何模型参数，因此对 Attention 进行批处理并不能通过在多个 requests 之间复用已加载的参数来减少 GPU 内存读取量

Based on these techniques, we design and implement ORCA, a distributed serving system for Transformer-based generative models. In order to handle large-scale models, ORCA adopts parallelization strategies including intra-layer and inter-layer model parallelism, which were originally developed by training systems [55, 58] for Transformer models. We also devise a new scheduling algorithm for the proposed iteration-level scheduling, with additional considerations for memory management and pipelined execution across workers. 
>  基于这些技术，我们设计了 ORCA，一个面向基于 Transformer 的生成式模型的分布式服务系统
>  为了处理大规模模型，ORCA 采用了并行化策略，包括了层内和层间的模型并行
>  我们也为我们提出的 iteration-level 的调度提出了一个新的调度算法，该算法考虑了内存管理和 workers 之间的流水线执行

We evaluate ORCA using OpenAI GPT-3 [12] models with various configurations, scaling up to 341B of parameters. The results show that ORCA significantly outperforms FasterTransformer [4], showing  $36.9\times$  throughput improvement at the same level of latency. While we use a language model as a driving example throughout the paper and conduct experiments only on language models, generative models in other domains can benefit from our approach as long as the models are based on the Transformer architecture and use the autoregressive generation procedure [19, 38, 51, 62].

# 2 Background
We provide background on the inference procedure of GPT [12, 47], a representative example of Transformer-based generative models that we use throughout this paper, and ML inference serving systems.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/93446162c9d8a243db893f25e482a15911a21ce01acf5fa60f1c67a30bb13075.jpg)  

Figure 1: Illustrations for GPT's inference procedure, Transformer layer, and internal state usage.

**Inference procedure of GPT.** GPT is an autoregressive language model based on one of architectural variants of Transformer [60]. It takes text as input and produces new text as output. In particular, the model receives a sequence of input tokens and then completes the sequence by generating subsequent output tokens. Figure 1a illustrates a simplified computation graph that represents this procedure with a three-layer GPT model, where nodes and edges indicate Transformer layers and dependencies between the layers, respectively. The Transformer layers are executed in the order denoted by the numbers on the nodes, and the nodes that use the same set of model parameters (i.e., nodes representing the same layer) are filled with the same color.
>  GPT 接收输入 tokens 序列，生成输出 tokens 完成该序列
>  Figure1a 展示了生成过程的计算图，节点表示 Transformer layer，边表示 layers 之间的依赖

The generated output token is fed back into the model to generate the next output token, imposing a sequential, one-by-one inference procedure. This autoregressive procedure of generating a single token is done by running all the layers of the model with the input, which is either a sequence of input tokens that came from the client or a previously generated output token. 

We define the run of all layers as an iteration of the model. In the example shown in Figure 1a, the inference procedure comprises three iterations. The first iteration ("iter 1") takes all the input tokens ("I think this") at once and generates the next token ("is"). This iteration composes an initiation phase, a procedure responsible for processing the input tokens and generating the first output token. The next two iterations ("iter 2" and "iter 3"), which compose an increment phase, take the output token of the preceding iteration and generate the next token. In this case, "iter 3" is the last iteration because it produces "\<EOS\>", a special end-of-sequence token that terminates output generation. Note that while the increment phase comprises multiple iterations because each iteration is only able to process a single token, the initiation phase is typically implemented as a single iteration by processing all the input tokens in parallel.
>  我们将运行模型的所有层的过程称为一次迭代
>  Figure1a 的过程展示了三次迭代，第一次迭代为 prefill，接收输入，生成第一个输出 token，之后的迭代为增量式的 decoding，接收一个 token，生成下一个 token
>  生成了 `<EOS>` 的迭代为最后一次迭代

The original Transformer [60] employs two stacks of Transformer layers, while GPT's architecture consists of a single layer stack, namely decoder. Figure 1b shows a Transformer layer used in GPT. Among the operations that compose the Transformer layer, Attention is the essence that distinguishes Transformer from other architectures. At a high level, the Attention operation computes a weighted average of the tokens of interest so that each token in the sequence is aware of the other. It takes three inputs, query, key, and value, computes dot products of the query (for the current token) with all keys (for the tokens of interest), applies Softmax on the dot products to get weights, and conducts weighted average of values associated with the weights.
>  Figure 1b 展示了一个 Transformer layer
>  layer 中，Attention 是关键的计算，Attention 为每个 token 计算其他 tokens 的一个加权平均，使得序列中的 tokens 可以相互认知
>  Attention operation 输入为 query, key, value，计算 query 和所有 keys 的点积，对点积结果应用 softmax 获取权重，然后对 value 进行加权平均

Since the Attention requires keys and values of all preceding tokens, we consider the keys and values as internal states that should be maintained across multiple iterations. A naive, state-less inference procedure would take all tokens in the sequence (including both the client-provided input tokens and the output tokens generated so far) to recompute all the keys and values at every iteration. To avoid such recomputation, fairseq [43] suggests incremental decoding, which saves the keys and values for reuse in successive iterations. Other systems for Transformer such as FasterTransformer [4] and Megatron-LM [3] also do the same.
>  Attention 计算需要所有前置 tokens 的 keys, values，故我们将 KVCache 视作需要在多个迭代中维护的内部状态

Figure 1c illustrates the state usage pattern of Transformer, along with LSTM [25] that also maintains internal states. The main difference is that the size of the states  $k$  for Attention key and  $\nu$  for value) in Transformer increases with iteration, whereas the size of the states  $c$  for LSTM internal memory and  $h$  for LSTM layer's input/output) in LSTM remains constant. When processing the token at index  $t$  the Attention operation takes all previous Attention keys  $k_{l,1:t -1}$  and values  $\nu_{l,1:t -1}$  along with the current key  $k_{l,t}$  and value  $\nu_{l,t}$ . Therefore, the Attention operation should perform computation on tensors of different shapes depending on the number of tokens already processed.
>  如 Figure1c 所示，Transformer 的状态大小随着每次迭代增长，而 LSTM 的内部状态则保持恒定
>  因此，Attention 计算在不同的迭代之间是在不同形状的 tensor 上执行计算，tensor 的形状取决于当前已经处理的 tokens 数量

Prior to the Attention operation, there are the layer normalization operation (LayerNorm) and the QKV Linear (linear and split operations to get the query, key and value). Operations performed after Attention are, in order, a linear operation (Attn Out Linear), an add operation for residual connection (Add), layer normalization operation (LayerNorm), the multilayer perceptron (MLP) operations, and the other residual connection operation (Add).
>  Attention 计算之前的计算是 LayerNorm 和 QKV Linear
>  Attention 计算之后的计算是 Attn Out Linear 和 Add, LayerNorm, MLP, Add

**ML inference serving systems.** Growing demands for ML-driven applications have made ML inference serving service a critical workload in modern data centers. Users (either the end-user or internal microservices of the application) submit requests to an inference service, and the service gives replies on the requests based on a pre-defined ML model using its provisioned resource, typically equipped with specialized accelerators such as GPUs and TPUs. In particular, the service runs a DNN model with input data to generate output for the request. Just like other services operating on datacenters, a well-managed inference service should provide low latency and high throughput within a reasonable amount of cost.
>  用户向 inference service 提交 requests, service 为用户提供 replies
>  inference service 应该提供 low latency 以及 high throughput

To meet such constraints, service operators often use ML inference serving systems such as Triton Inference Server [7] and TensorFlow Serving [42]. These systems can be seen as an abstraction sitting atop underlying model execution engines such as TensorFlow [6], TVM [14], TensorFlow [8], and many others [44,46], being agnostic to various kinds of ML models, execution engines, and computing hardware. While delegating the role of driving the main mathematical operations to the engines, serving systems are in charge of exposing endpoints that receive inference requests, scheduling executions of the engine, and sending responses to the requests. Accordingly, these systems focus on aspects such as batching the executions [7, 16, 35, 42, 56], selecting an appropriate model from multiple model variants [16,27,30,57], deploying multiple models (each for different inference services) on the same device [7, 29, 35, 56], and so on.
>  ML inference serving system 会将主要的计算交给 engines，自己负责暴露接收 requests 的端点、调度 engine 的执行、以及返回回复
>   ML inference serving system 聚焦于对 requests 做 batching，选择合适的模型、部署各种模型等等

Among the features and optimizations provided by serving systems, batching is a key to achieve high accelerator utilization when using accelerators like GPUs. When we run the execution engine with batching enabled, the input tensors from multiple requests coalesce into a single, large input tensor before being fed to the first operation of the model. Since the accelerators prefer large input tensors over small ones to better exploit the vast amount of parallel computation units, the engine's throughput is highly dependent on the batch size, i.e., the number of inference requests the engine processes together. Reusing the model parameters loaded from off-chip memory is another merit in batched execution, especially when the model involves memory-intensive operations.
>  batching 是提高加速器利用率的关键优化
>  启用批处理时，多个 requests 的输入 tensors 被合并为单个大 tensor，然后交给模型计算，这可以更好利用加速器的并行计算单元
>  engine 的吞吐高度依赖于 batch size，也就是同时处理的 requests 数量
>  批处理的另一个优势是复用从片外内存取得的模型参数，这在模型涉及了 memory-intensive 计算时非常有用

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/989cbc958c96c8315a790c41c8b10dd555f96dd4cc23e58a73382c348a032a56.jpg)  


Figure 2: Overall workflow of serving a generative language model with existing systems.

Figure 2 shows an overall workflow of serving a generative language model with existing serving systems and execution engines. The main component of the serving system (e.g., Triton [7]) is the scheduler, which is responsible for  $①$  creating a batch of requests by retrieving requests from a queue and  $②$  scheduling the execution engine (e.g., FasterTransformer [4]) to process the batch. The execution engine  $③$  processes the received batch by running multiple iterations of the model being served and  $④$  returns the generated text back to the serving system. In Figure 2, the serving system schedules the engine to process two requests  $(x_{1}$  : "I think",  $x_{2}$  : "I love") in a batch and the engine generates "this is great" and "you" for requests  $x_{1}$  and  $x_{2}$ , respectively.
>  Figure2 展示了一个系统的服务流
>  serving system 的主要组件是它的 scheduler，负责 1. 从队列中创建 a batch of requests 2. 调度执行引擎来处理 batch
>  执行引擎将 3. 对模型运行多轮迭代，处理收到的 batch 4. 将生成的文本返回给 serving system

# 3 Challenges and Proposed Solutions
In this section, we describe challenges in serving Transformer-based generative models and propose two techniques: iteration-level scheduling and selective batching.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/e2e1858d8828792b65aab26036633ab7541c42d8a5930b38730a4e02feef21e1.jpg)  

Figure 3: An illustration for a case where the requests have the same input length but some requests finish earlier than others. Shaded tokens represent input tokens. "" denotes inputs and outputs of extra computation imposed by the scheduling.

**CI: Early-finished and late-joining requests.** One major limitation of existing systems is that the serving system and the execution engine interact with each other only when (1) the serving system schedules the next batch on an idle engine; or (2) the engine finishes processing the current batch. In other words, these systems are designed to schedule executions at request granularity; the engine maintains a batch of requests fixed until all requests in the batch finish. This can be problematic in the serving of generative models, since each request in a batch may require different number of iterations, resulting in certain requests finishing earlier than the others. In the example shown in Figure 3, although request  $x_{2}$  finishes earlier than request  $x_{1}$ , the engine performs computation for both "active" and "inactive" requests throughout all iterations. Such extra computation for inactive requests ( $x_{2}$  at iter 3 and 4) limits the efficiency of batched execution.
>  现存系统的一个主要限制是 serving system 和 engine 相互的交互只有 1. serving system 将下一个 batch 调度给空闲的 engine 2. engine 完成当前 batch 处理
>  换句话说，这些系统在 request 的粒度调度执行，engine 在完成一个 batch 之前不会返回
>  而生成式模型的每个 request 需要的迭代数量不一定相同，故存在提早完成的 request，对这些不活跃 request 的额外计算限制了批量执行的效率

What makes it even worse is that this behavior prevents an early return of the finished request to the client, imposing a substantial amount of extra latency. This is because the engine only returns the execution results to the serving system when it finishes processing all requests in the batch. Similarly, when a new request arrives in the middle of the current batch's execution, the aforementioned scheduling mechanism makes the newly arrived request wait until all requests in the current batch have finished. 
>  此外，这防止了完成的 request 的提前返回，提高了 latency
>  类似地，当新的 request 到达，需要先等待当前 batch 所有 requests 完成执行

We argue that the current request-level scheduling mechanism cannot efficiently handle workloads with multi-iteration characteristic. Note that this problem of early-finished and late-joining requests does not occur in the training of language models; the training procedure finishes processing the whole batch in a single iteration by using the teacher forcing technique [64].
>  故当前的 request-level 调度无法高效处理具有多轮迭代性质的 workloads
>  注意这个 early-finished, late-joining requests 的问题不会在语言模型的训练中出现，语言模型的训练过程通过 teacher forcing 技术，在单次迭代中完成整个 batch 的处理 (完成整个 batch 的损失计算，而不是一个一个等输出 token 再计算损失)

**S1: Iteration-level scheduling.** To address the above limitations, we propose to schedule executions at the granularity of iteration. At high level, the scheduler repeats the following procedure: (1) selects requests to run next; (2) invokes the engine to execute one iteration for the selected requests; and (3) receives execution results for the scheduled iteration. Since the scheduler receives a return on every iteration, it can detect the completion of a request and immediately return its generated tokens to the client. For a newly arrived request, the request gets a chance to start processing (i.e., the scheduler may select the new request to run next) after execution of the currently scheduled iteration, significantly reducing the queueing delay. With iteration-level scheduling, the scheduler has a full control on how many and which requests are processed in each iteration.
>  为了解决这些限制，我们提出在 iteration 粒度调度执行
>  scheduler 重复以下过程: 1. 选择下次要运行的 requests 2. 调用 engine，对这些 requests 处理一次迭代 3. 收到这次迭代的执行结果
>  这样 scheduler 可以立即检测到 request 的完成，进而直接返回给 client，新到达的 request 也可以被立刻调度执行，显著降低了排队延迟

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/6c9f96fa66b21d1a0372f70948f0ac8f3bb89f8af94601284833eab855c06eb9.jpg)  

Figure 4: System overview of ORCA. Interactions between components represented as dotted lines indicate that the interaction takes place at every iteration of the execution engine.  $x_{ij}$  is the j-th token of the i-th request. Shaded tokens represent input tokens received from the clients, while unshaded tokens are generated by ORCA. For example, request  $x_{1}$  initially arrived with two input tokens  $(x_{11},x_{12})$  and have run two iterations so far, where the first and second iterations generated  $x_{13}$  and  $x_{14}$ , respectively. On the other hand, request  $x_{3}$  only contains input tokens  $(x_{31},x_{32})$  because it has not run any iterations yet.

Figure 4 depicts the system architecture and the overall workflow of ORCA using the iteration-level scheduling. ORCA exposes an endpoint (e.g., HTTPS or gRPC) where inference requests arrive at the system and responses to the requests are sent out. The endpoint puts newly arrived requests in the request pool, a component that manages all requests in the system during their lifetime. The pool is monitored by the scheduler, which is responsible for: selecting a set of requests from the pool, scheduling the execution engine to run an iteration of the model on the set, receiving execution results (i.e., output tokens) from the engine, and updating the pool by appending each output token to the corresponding request. 
>  ORCA 的系统结构如 Figure 4 所示，ORCA 暴露一个端点，接收 requests 和发送回复
>  端点会将 requests 放在 request pool, scheduler 监视 pool
>  scheduler 负责:
>  1. 从 pool 中选择一组 requests,
>  2. 发送给 engine 执行一个迭代
>  3. 回收执行结果
>  4. 将输出 tokens 附加到各个 requests，更新 pool

The engine is an abstraction for executing the actual tensor operations, which can be parallelized across multiple GPUs spread across multiple machines. 
>  eigine 是对实际张量运算的抽象

In the example shown in Figure 4, the scheduler (1) interacts with the request pool to decide which requests to run next and (2) invokes the engine to run four selected requests:  $(x_{1},x_{2},x_{3},x_{4})$ . The scheduler provides the engine with input tokens of the requests scheduled for the first time. In this case,  $x_{3}$  and  $x_{4}$  have not run any iterations yet, so the scheduler hands over  $(x_{31},x_{32})$  for  $x_{3}$  and  $(x_{41},x_{42},x_{43})$  for  $x_{4}$ . The engine (3) runs an iteration of the model on the four requests and (4) returns generated output tokens  $(x_{15},x_{23},x_{33},x_{44})$ , one for each scheduled request. 

Once a request has finished processing, the request pool removes the finished request and notifies the endpoint to send a response. Unlike the method shown in Figure 2 that should run multiple iterations on a scheduled batch until finish of all requests within the batch, ORCA's scheduler can change which requests are going to be processed at every iteration. We describe the detailed algorithm about how to select the requests at every iteration in Section 4.2.
>  当 request 完成了处理, request pool 就移除完成的 request，并告知 endpoint 发送回复
>  ORCA 的调度器可以在每次迭代改变需要处理的 requests

**C2: Batching an arbitrary set of requests.** When we try to use the iteration-level scheduling in practice, one major challenge that we are going to face is batching. To achieve high efficiency, the execution engine should be able to process any selected set of requests in the batched manner. Without batching, one would have to process each selected request one by one, losing out on the massively parallel computation capabilities of GPUs.
>  在实践中使用 iteration-level scheduling 时，一个主要挑战是 engine 需要能够以 batch 的形式处理任意选定的一组 requests
>  如果没有 batching 处理，就只能一个一个地处理选中的 request，显然不是理想的

Unfortunately, there is no guarantee that even for a pair of requests  $(x_{i},x_{j})$ , for the next iteration, their executions can be merged and replaced with a batched version. There are three cases for a pair of requests where the next iteration cannot be batched together: (1) both requests are in the initiation phase and each has different number of input tokens (e.g.,  $x_{3}$  and  $x_{4}$  in Figure 4); (2) both are in the increment phase and each is processing a token at different index from each other  $(x_{1}$  and  $x_{2})$ ; or (3) each request is in the different phase: initiation or increment  $(x_{1}$  and  $x_{3})$ . 
>  但我们无法保证 requests 的执行可以被合并被替换为批处理版本
>  对于一对 request，有三种情况下它们的下一次迭代无法进行批处理:
>  1. 两个 request 都处于 prefill 阶段，且每个 request 的输入 tokens 数量不同
>  2. 两个 request 都处于 decode 阶段，且每个 request 处理的 token 索引不同
>  3. 两个 request 处于不同的阶段

Recall that in order to batch the execution of multiple requests, the execution of each request must consist of identical operations, each consuming identically-shaped input tensors. 
>  要批量化处理多个 requests，每个 request 的执行必须包含相同的计算，并且每个计算都使用形状相同的输入张量 (深度学习框架批处理的前提是多个相同形状的张量可以堆叠为更高维的单个张量，例如 `batch_size, seq_len, hidden_dim`，然后通过相同的计算图一次性处理)

In the first case, the two requests cannot be processed in a batch because the "length" dimension of their input tensors, which is the number of input tokens, are not equal. The requests in the second case have difference in the tensor shape of Attention keys and values because each processes token at different index, as shown in Figure 1c. For the third case, we cannot batch the iterations of different phases because they take different number of tokens as input; an iteration of the initiation phase processes all input tokens in parallel for efficiency, while in the increment phase each iteration takes a single token as its input (we assume the use of fairseq-style incremental decoding [43]).
>  在第一种情况下，两个 request 的输入张量的 `seq_len` 维度不同
>  在第二种情况下，两个 request 的 Attention Key Value 矩阵的形状不同
>  在第三种情况下，prefill, decode 阶段的计算接收的输入 tokens 数量不同 (prefill 并行处理所有输入 tokens, decode 一定只处理一个 token)

>  本质上都是通过相同的计算图一次性处理

Batching is only applicable when the two selected requests are in the same phase, with the same number of input tokens (in case of the initiation phase) or with the same token index (in case of the increment phase). This restriction significantly reduces the likelihood of batching in real-world workloads, because the scheduler should make a wish for the presence of two requests eligible for batching at the same time. The likelihood further decreases exponentially as the batch size increases, making it impractical to use a large batch size that can pull out better throughput without compromising latency.
>  只有在两个选定的 requests 位于相同阶段，带有相同数量的输入 tokens 或者正在 decode 相同索引的 token 才可以进行批处理

**S2: Selective batching.** We propose selective batching, a technique for batched execution that allows high flexibility in composing requests as a batch. Instead of processing a batch of requests by "batchifying" all tensor operations composing the model, this technique selectively apply batching only to a handful of operations.
>  我们提出选择性批处理，这是一种批量执行技术，允许在将 requests 组合为一个 batch 时保持高度的灵活性
>  对于 a batch of requests，选择性批处理不会批量化处理模型中的所有张量计算，而是选择性的批处理一部分张量计算

The main problem regarding batching described above is that the three aforementioned cases correspond to irregularly shaped input (or state) tensors, which cannot be coalesced into a single large tensor and fed into a batch operation. In the canonical batching mechanism, at each iteration, a Transformer layer takes a 3-dimensional input tensor of shape  $[B,L,H]$  generated by concatenating multiple  $[L,H]$  input tensors of requests in a batch, where  $B$  is the batch size,  $L$  is the number of tokens processed together, and  $H$  is the hidden size of the model. 
>  我们之前讨论到的关于批处理的主要问题是在三种情况下，形状不规则的输入/张量无法合并为一个大的张量作为以 batch operation 的输入
>  在标准的批处理机制中，在每个迭代，一个 Transformer layer 层接收一个形状为 `[B, L, H]` 的三维输入张量，该张量是通过拼接一个 batch 中多个形状为 `[L, H]` 的输入张量形成的，其中 `B` 为 batch size, `L` 为同时处理的 token 数量，`H` 是模型的 hidden size

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/e2e1858d8828792b65aab26036633ab7541c42d8a5930b38730a4e02feef21e1.jpg)  

Figure 3: An illustration for a case where the requests have the same input length but some requests finish earlier than others. Shaded tokens represent input tokens. "" denotes inputs and outputs of extra computation imposed by the scheduling.

For example, in Figure 3, "iter 1" (initiation phase) takes an input tensor of shape  $[2,2,H]$  and "iter 2" (increment phase) takes a tensor of shape  $[2,1,H]$ . 
>  例如，Figure3 中，iter 1 接收形状为 `[2, 2, H]` 的张量，iter 2 接收形状为 `[2, 1, H]` 的张量

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/6c9f96fa66b21d1a0372f70948f0ac8f3bb89f8af94601284833eab855c06eb9.jpg)  

Figure 4: System overview of ORCA. Interactions between components represented as dotted lines indicate that the interaction takes place at every iteration of the execution engine.  $x_{ij}$  is the j-th token of the i-th request. Shaded tokens represent input tokens received from the clients, while unshaded tokens are generated by ORCA. For example, request  $x_{1}$  initially arrived with two input tokens  $(x_{11},x_{12})$  and have run two iterations so far, where the first and second iterations generated  $x_{13}$  and  $x_{14}$ , respectively. On the other hand, request  $x_{3}$  only contains input tokens  $(x_{31},x_{32})$  because it has not run any iterations yet.

However, when the scheduler decides to run an iteration on batch  $(x_{1},x_{2},x_{3},x_{4})$  in Figure 4, the inputs for requests in the initiation phase  $(x_{3}:[2,H]$  and  $x_{4}:[3,H])$  cannot coalesce into a single tensor of shape  $[B,L,H]$  because  $x_{3}$  and  $x_{4}$  have different number of input tokens, 2 and 3.
>  但是，在 Figure 4 中，`x1, x2, x3, x4` 组成的 batch 就无法拼接为一个形状为 `[B, L, H]` 的张量

Interestingly, not all operations are incompatible with such irregularly shaped tensors. Operations such as non-Attention matrix multiplication and layer normalization can be made to work with irregularly shaped tensors by flattening the tensors.
>  实际上，不是所有的计算都无法与这种形状不规则的张量兼容
>  例如 non-Attention 矩阵乘法和 layer norm 运算可以通过展平张量来适应形状不规则的张量

For instance, the aforementioned input tensors for  $x_{3}$  and  $x_{4}$  can compose a 2-dimensional tensor of shape  $[\Sigma L,H] = [5,H]$  without an explicit batch dimension. This tensor can be fed into all non-Attention operations including Linear, Layer-Norm, Add, and GeLU operations because they do not need to distinguish tensor elements of different requests. On the other hand, the Attention operation requires a notion of requests (i.e., requires the batch dimension) to compute attention only between the tokens of the same request, typically done by applying cuBLAS routines for batch matrix multiplication.
>  例如，`x3, x4` 组合成的输入张量可以直接在 `seq_len` 维度拼接，得到形状为 `[5, H]` 的张量，没有显式的 batch 维度
>  这个张量可以输入到所有的 non-Attention 操作中，包括 Linear, Layer-Norm, Add, GeLU，因为这些操作不要求区分不同 requests 的张量元素
>  另一方面，Attention 计算需要 request 的概念 (即需要 batch 维度)，以仅在同一个 request 的 tokens 之间计算注意力，这通常是通过 cuBLAS 中的批量矩阵乘实现的

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/8755abb95e2b940dcb78ba979b9da81418c5e0929f3482a23b4f0098fff9be0d.jpg)  

Figure 5: An illustration of ORCA execution engine running a Transformer layer on a batch of requests with selective batching. We only depict the QKV Linear, Attention, and Attention Out Linear operations for simplicity.

Selective batching is aware of the different characteristics of each operation; it splits the batch and processes each request individually for the Attention operation while applying token-wise (instead of request-wise) batching to other operations without the notion of requests. 
>  选择性批处理了解每种计算的不同特性，它会拆分出注意力计算，单独处理每个 request，而对于其他不需要 “请求” 概念的计算，采用 token-wise (而不是 request-wise) 的 batching 方式

Figure 5 presents the selective batching mechanism processing a batch of requests  $(x_{1},x_{2},x_{3},x_{4})$  described in Figure 4. This batch has 7 input tokens to process, so we make the input tensor have a shape of  $[7,H]$  and apply the non-Attention operations. Before the Attention operation, we insert a Split operation and run the Attention operation separately on the split tensor for each request. The outputs of Attention operations are merged back into a tensor of shape  $[7,H]$  by a Merge operation, bringing back the batching functionality to the rest of operations.
>  Figure5 展示了选择性批处理机制处理 Figure4 中 `x1, x2, x3, x4` batch 的方式
>  这个 batch 一共有 7 个需要处理的 tokens，因此我们构造一个输入形状为 `[7, H]` 的张量，对它应用 non-Attention 计算
>  在 Attention 计算之前，我们插入一个 Split 操作，然后为每个拆分下来的每个 request 的张量单独应用 Attention 计算
>  Attention 计算的输入重新组合为形状为 `[7, H]` 的张量，以便后续计算进行批处理

To make the requests in the increment phase can use the Attention keys and values for the tokens processed in previous iterations, ORCA maintains the generated keys and values in the Attention  $K / V$  manager. The manager maintains these keys and values separately for each request until the scheduler explicitly asks to remove certain request's keys and values, i.e., when the request has finished processing. The Attention operation for request in the increment phase  $(x_{1}$  and  $x_{2})$  takes keys and values of previous tokens  $(x_{11},x_{12},x_{13}$  for  $x_{1};x_{21}$  for  $x_{2})$  from the manager, along with the current token's query, key, and value from the Split operation to compute attention between the current token and the previous ones.
>  ORCA 会为每个 request 维护它的 KVCache，manager 会维护 KVCache 直到 scheduler 显式请求移除该 request 的 KVCache，也就是 request 的处理结束后

# 4 ORCA Design
Based on the above techniques, we design and implement ORCA: a distributed serving system for Transformer-based generative models. We have already discussed the system components and the overall execution model of ORCA while describing Figure 4. In this section, we answer the remaining issues about how to build an efficient system that can scale to large-scale models with hundreds of billions of parameters. We also describe the scheduling algorithm for iteration-level scheduling, i.e., how to select a batch of requests from the request pool at every iteration.
>  基于上述技术，我们设计并实现了 ORCA: 一个针对 Transformer-based 生成式模型的分布式服务系统
>  在描述 Figure4 时，我们已经讨论了系统组件和整体的执行模型，在本节，我们回答关于如何构建一个高效的系统以拓展到百 B 级别的参数，我们也描述 iteration-level 的调度算法，即如何在每次 iteration 从 request pool 中选择一批 requests

## 4.1 Distributed Architecture
Recent works [12, 31] have shown that scaling language models can dramatically improve the quality of models. Hence, system support for serving such large language models is getting more importance, especially when the model does not fit in a single GPU. In such a case, one should split the model parameters along with the corresponding computation and distribute them across multiple GPUs and machines.

ORCA composes known parallelization techniques for Transformer models: intra-layer parallelism and inter-layer parallelism. These two model parallelism strategies, which are also used by FasterTransformer [4], have been originally developed for distributed training. Intra-layer parallelism [55, 58] splits matrix multiplications (i.e., Linear and Attention operations) and their associated parameters over multiple GPUs. We omit the detail about how this strategy partitions each matrix multiplication. On the other hand, inter-layer parallelism splits Transformer layers over multiple GPUs. 
>  ORCA 结合了已知的 Transformer 模型的并行化技术: 层内并行和层间并行
>  这两个并行化技术最初是为分布式训练开发的
>  层内并行将矩阵乘法 (即 Linear 和 Attention 计算) 以及它们相关的参数划分到多个 GPU 上
>  层间并行将 Transformer 层划分到多个 GPUs 上

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/f326d4ce6c8d715277a9484dd95c88c621735f3239ae7c57ec5624c67e70a183.jpg)  

Figure 6: An example of intra-and inter-layer parallelism. A vertical dotted line indicates partitioning between layers and a horizontal line indicates partitioning within a layer.

ORCA assigns the same number of Transformer layers to each GPU. Figure 6 illustrates an example application of intra-and inter-layer parallelism to a 4-layer GPT model. The 4 layers are split into 2 inter-layer partitions, and the layers in the partition are subdivided into 3 intra-layer partitions. We assign each partition to a GPU, using a total of 6 GPUs.
>  ORCA 为每个 GPU 分配相同数量的 Transformer layer
>  层内并行和层间并行的一个结合实例如 Figure6 所示，4 个层被划分为两个层间 partition，每个 partitoin 内的层被划分为三个层内 partition

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/4aa6ca5b4b9a09b2f4f012a3fbacc6492c26ac86cdaac3967aa9797efcd0c036.jpg)  

Figure 7: An illustration of the distributed architecture of ORCA's execution engine using the parallelization configuration shown in Figure 6. For example, the first inter-layer partition (Layer1 and Layer2) in Figure 6 is assigned to Worker1, while the second partition is assigned to Worker2.

The ORCA execution engine supports distributed execution using the techniques described above. Figure 7 depicts the architecture of an ORCA engine. Each worker process is responsible for an inter-layer partition of the model and can be placed on a different machine from each other. In particular, each worker manages one or more CPU threads each dedicated for controlling a GPU, the number of which depends on the degree of intra-layer parallelism.
>  ORCA 执行引擎使用上述技术来支持分布式执行
>  ORCA 执行引擎的结构如 Figure7 所示，每个 worker 进程都负责一个层间 partition，该 worker 进程可以和其他 woker 进程在不同的机器上
>  特别地，每个 worker 进程管理一个或多个 CPU 线程，每个 CPU 线程专门控制一个 GPU，CPU 线程的数量取决于层内并行的程度

The execution procedure of the ORCA execution engine is as follows. Once the engine is scheduled to run an iteration of the model for a batch of requests, the engine master forwards the received information about the scheduled batch to the first worker process (Worker1). The information includes tokens for the current iteration and a control message, which is composed of ids of requests within the batch, current token index (for requests in the increment phase), and number of input tokens (for requests in the initiation phase). The controller of Worker1 hands over the information received from the engine master to the GPU-controlling threads, where each thread parses the information and issues proper GPU kernels to its associated GPU.
>  ORCA 执行引擎的执行过程如下所示:
>  当引擎被调度到 a batch of requests 以运行一个迭代时，engine master 会将接收到的关于 scheduled batch 的信息发送给第一个 worker process (Worker1)
>  这些信息包括了当前迭代的 tokens 以及一个控制消息，控制消息包含了本次 batch 中的 requests IDs, 各个 request 的 current token index
>  Worker1 的控制器将收到的信息传递给控制 GPU 的 threads, 每个 thread 解析信息，并为它关联的 GPU 发起适当的 GPU kernel

 For example, the kernel for the Attention operation uses the request id and the current token index to get the GPU memory address of previous keys and values kept by the Attention K/V manager. 
 >  例如 Attention 计算的 kernel 需要使用 request id 和 current index 来获取由 Attention K/V manager 管理的 KVCache 的 GPU 内存地址

In the meantime, the controller also forwards the control message to the controller of the next worker (Worker2), without waiting for the completion of the kernels issued on the GPUs of Worker1. Unlike Worker1, the controller of the last worker (Worker2) waits for (i.e., synchronize with) the completion of the issued GPU kernels, in order to fetch the output token for each request and send the tokens back to the engine master.
>  同时，Worker1 的控制器会将控制信息发送给下一个 worker process (Worker2) 的控制器，注意控制消息的传递不会等待 Worker1 发起的所有 GPU kernels 完成
>  最后一个 Worker 会等待它发起的 GPU kernels 完成后，获取每个 request 的输出 token，然后将 tokens 发送回 engine master

To keep GPUs busy as much as possible, we design the ORCA engine to minimize synchronization between the CPU and GPUs. We observe that current systems for distributed inference (e.g., FasterTransformer [4] and Megatron-LM [3]) have CPU-GPU synchronization whenever each process receives control messages because they exchange the messages through a GPU-to-GPU communication channel - NCCL [5]. The exchange of these control messages occurs at every iteration, imposing a non-negligible performance overhead. On the other hand, ORCA separates the communication channels for control messages (plus tokens) and tensor data transfer, avoiding the use of NCCL for data used by CPUs. 
>  为了尽可能保持 GPU 忙碌，我们将 ORCA engine 设计为最小化 CPU 和 GPUs 之间的同步
>  我们发现现存的分布式推理系统在每个进程收到控制消息都会进行 CPU-GPU 同步，因为它们通过 GPU-to-GPU 的通信通道 (即 NCCL) 交换信息
>  这些控制消息的交换发生在每次迭代中，带来了不可忽略的性能开销
>  ORCA 将控制消息 (以及 token) 的通信通道和张量数据传输的通信通道分离，避免了使用 NCCL 来处理 CPU 使用的数据 (思想: 元数据和真实数据的传输分离)

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/4aa6ca5b4b9a09b2f4f012a3fbacc6492c26ac86cdaac3967aa9797efcd0c036.jpg)  

Figure 7: An illustration of the distributed architecture of ORCA's execution engine using the parallelization configuration shown in Figure 6. For example, the first inter-layer partition (Layer1 and Layer2) in Figure 6 is assigned to Worker1, while the second partition is assigned to Worker2.

Figure 7 shows that the ORCA engine uses NCCL exclusively for exchanging intermediate tensor data (represented by dashed arrows) as this data is produced and consumed by GPUs. Control messages, which is used by the CPU threads for issuing GPU kernels, sent between the engine master and worker controllers by a separate communication channel that does not involve GPU such as gRPC [2].
>  Figure7 展示了 ORCA engine 使用 NCCL 仅用于交换中间张量数据，因为这些数据都是由 GPU 生成和消费的
>  CPU 线程用于发起 GPU kernels 的控制信息则在 engine master 和 worker controllers 之间通过不涉及 GPU 的单独通信通道发送，例如 gRPC

## 4.2 Scheduling Algorithm
The ORCA scheduler makes decisions on which requests should be selected and processed at every iteration. The scheduler has high flexibility in selecting a set of requests to compose a batch, because of the selective batching technique that allows the engine to run any set of requests in the batched manner. Now the main question left is how to select the requests at every iteration.
>  ORCA scheduler 决定在每次迭代时，哪些 requests 应该被选择并交给 engine 处理
>  因为选择性批处理允许以批处理的形式执行任意的一组 requests，因此 scheduler 可以灵活选择组成 batch 的 requests
>  现在留下的问题是怎样在每次迭代中选择 requests

We design the ORCA scheduler to use a simple algorithm that does not change the processing order of client requests; early-arrived requests are processed earlier. That is, we ensure iteration-level first-come-first-served (FCFS) property. 
>  我们将 ORCA scheduler 设计为使用一个不改变 client requests 的处理顺序的简单算法，也就是先来的 requests 会被先处理，也就是我们确保 iteration-level 的 FCFS 性质

We define the iteration-level FCFS property for workloads with multi-iteration characteristics as follows: for any pair of requests  $(x_{i},x_{j})$  in the request pool, if  $x_{i}$  has arrived earlier than  $x_{j}$ $x_{i}$  should have run the same or more iterations than  $x_{j}$  . Note that some late-arrived requests may return earlier to clients if the late request requires a smaller number of iterations to finish.
>  我们为带有多轮 iteratoin 性质的 workloads 的 iteration-level FCFS 性质如下:
>  对于 request pool 中的任意一组 requests $(x_i, x_j)$，如果 $x_i$ 比 $x_j$ 早到，那么 $x_i$ 运行的 iteration 数量应该不小于 $x_j$ 运行的 iteration 数量

Still, the scheduler needs to take into account additional factors: diminishing returns to increasing the batch size and GPU memory constraint. Increasing the batch size trades off increased throughput for increased latency, but as the batch size becomes larger, the amount of return (i.e., increase in throughput) diminishes. Therefore, just like other serving systems [7, 16], ORCA also has a notion of a max batch size: the largest possible number of requests within a batch. The ORCA system operator can tune this knob to maximize throughput while satisfying one's latency budget. We will discuss this in more details with experiment results in Section 6.2.
>  scheduler 需要考虑额外的因素: 增加 batch size 的边际收益递减以及 GPU 内存限制
>  增大 batch size 是以增大延迟为代价换更高的吞吐，而随着 batch size 增大，收益 (吞吐的增加) 会减小
>  因此，和其他的服务系统一样，ORCA 引入了 max batch size 的概念: 一个 batch 内可能包含的最大 requests 数量，ORCA 系统管理员可以调节这个参数，以在满足延迟要求的情况下最大化吞吐

Another factor is the GPU memory constraint. Optimizing memory usage by reusing buffers for intermediate results across multiple operations is a well-known technique used by various systems [4, 6], and ORCA also adopts this technique. However, unlike the buffers for intermediate results that can be reused immediately, buffers used by the Attention K/V manager for storing the keys and values cannot be reclaimed until the ORCA scheduler notifies that the corresponding request has finished processing. A naive implementation can make the scheduler fall into a deadlock when the scheduler cannot issue an iteration for any requests in the pool because there is no space left for storing a new Attention key and value for the next token. This requires the ORCA scheduler to be aware of the remaining size of pre-allocated memory regions for the manager.
>  GPU 内存约束也是 scheduler 需要考虑的因素
>  通过在多个计算中复用存储中间结果的缓冲区是常用的内存优化方式，ORCA 也采用了这一技术
>  但是 Attention K/V manager 用于存储 KVCache 的缓冲区不能被复用，直到 ORCA scheduler 告知了 manager 该 request 已经处理完毕
>  一个朴素的实现会让 scheduler 进入死锁，其中 scheduler 无法为 pool 中的任意 requests 发起 iteration，因为没有空间用于存储下一个 token 的新 KVCache
>  故这要求 ORCA scheduler 能够感知为 manager 预分配的内存区域的剩余大小

![[pics/ORCA-Algorithm1.png]]

The ORCA scheduler takes all these factors into account: it selects at most "max batch size" requests based on the arrival time, while reserving enough space for storing keys and values to a request when the request is scheduled for the first time. We describe the scheduling process in Algorithm 1. The algorithm selects a batch of requests from the request pool (line 4) and schedules the batch (line 5). The Select function (line 17) selects at most  $max\_ bs$  requests from the pool based on the arrival time of the request (lines 20-22). Algorithm 1 does not depict the procedure of request arrival and return; one may think of it as there exist concurrent threads inserting newly arrived requests into `request_pool` and removing finished requests from `request_pool`.
>  ORCA scheduler 会考虑上述讨论的因素: 它根据 request 的到达时间，最多选择 'max batch size'个请求，同时在 request 被首次调度时，为存储 KVCache 保留足够的空间
>  调度过程见 Algorithm1，算法首先从 request pool 选择一个 batch 的 requests，然后调度该 batch
>  line 17 的 `Select` 函数会基于 requests 的到达时间，最多选择 `max_bs` 个 requests

When the scheduler considers a request in the initiation phase, meaning that the request has never been scheduled yet, the scheduler uses the request's max_tokens attribute to reserve max_tokens slots of GPU memory for storing the keys and values in advance (lines 23-26). The scheduler determines whether the reservation is possible (line 25) based on  $n\_ rsr\nu$  , the number of currently reserved slots, where a slot is defined by the amount of memory required for storing an Attention key and value for a single token. 
>  当 scheduler 考虑一个 request 的 prefill 阶段时，说明这个 request 还没有被调度，scheduler 会使用该 request 的 `max_tokens` 的属性来为这个 request 的 GPU 显存中预留 `max_tokens` 个存储 keys, values 的 slots
>  scheduler 会根据当前已经预留的 slots 数量 `n_rsrv` 来决定是否能够进行预留
>  slot 定义为存储单个 token 的 KVCache 需要的内存大小

Here,  $n\_ s l o t s$  is a parameter tuned by the ORCA system operator indicating the size of memory region (in terms of slots) allocated to the Attention K/V manager. Since the number of tokens in a request cannot exceed max_tokens, if the reservation is possible, it is guaranteed that the manager can allocate buffers for the newly generated keys and values until the request finishes.
>  `n_slots` 是由 ORCA 系统管理员调节的参数，表示分配给 Attention K/V manager 的内存区域大小 (slots 的数量)
>  由于 request 中的 tokens 数量不能超过 `max_tokens`，因此如果预留是可能的 (可以预留 `max_tokens` slots)，就保证 manager 可以为新生成的 key 和 value 分配缓冲区，直到 request 完成

Unlike the tuning of max_bs that requires quantifying the trade-off between latency and throughput, the ORCA system operator can easily configure  $n\_ slots$  without any experiments. Given a model specification (e.g., hidden size, number of layers, etc.) and degrees of intra-and inter-layer parallelism, ORCA's GPU memory usage mostly depends on  $n\_ slots$ . That is, the operator can simply use the largest possible  $n\_ slots$  under the memory constraint.
>  `max_bs` 的调节需要在满足延迟和吞吐之间权衡，而 `n_slots` 的调节则相对容易，不需要进行任意实验
>  给定模型规范和层内以及层间的并行度，ORCA 的 GPU 显存用量主要取决于 `n_slots`，也就是说，管理员可以再内存约束下直接使用最大的可能 `n_slots` 值

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/5c7b346124225ac041238092c2de9cbcd6381282733d38e51f28eb2718192610.jpg)  

Figure 8: Comparison of the use of pipeline parallelism in ORCA and FasterTransformer where  $X_{i}$  is the i-th iteration of request  $X$ .

**Pipeline parallelism.** ORCA's scheduler makes the execution of workers in the engine to be pipelined across multiple batches. The scheduler does not wait for the return of a scheduled batch until  $n\_ scheduled$ , the number of currently scheduled batches, reaches  $n\_ workers$  (line 9-10 of Algorithm 1). By doing so, the scheduler keeps the number of concurrently running batches in the engine to be  $n\_ workers$ , which means that every worker in the engine is processing one of the batches without being idle.
>  scheduler 会令 engine 中的 workers 执行在多个 batches 上进行流水线处理
>  scheduler 不会等待已调度 batch 的返回，直到当前已调度的 batch 数量 `n_scheduled` 达到 `n_workers`
>  通过这种方式，scheduler 保持 engine 中并发运行的 batch 数量为 `n_workers`，这意味着 engine 中的每个 worker 都在处理一个 batch，而不会处于空闲状态

Figure 8a depicts the execution pipeline of 3 ORCA workers, using a max batch size of 2. We assume that the request A arrives before B, which arrives before C, and so on. At first, the scheduler selects requests A and B based on the arrival time and schedules the engine to process a batch of requests A and B (we call this batch AB), where Worker1, Worker2, and Worker3 process the batch in turn. The scheduler waits for the return of the batch AB only after the scheduler injects two more batches: CD and EF. Once the batch AB returns, requests A and B get selected and scheduled once again, because they are the earliest arrived requests among the requests in the pool.
>  Figure 8a 展示了 3 个 ORCA workers 的执行流水线, 其中 max batch size = 2
>  batch AB 完成一轮迭代执行后，request A, B 被再次组合为一个 batch，然后继续被调度

In contrast, the interface between current serving systems and execution engines (e.g., a combination of Triton [7] and FasterTransformer [4]) does not allow injecting another batch before the finish of the current running batch, due to the request-level scheduling. 
>  相较之下，当前的服务系统和执行引擎之间的接口由于仅支持 request-level 调度，不允许在当前运行的 batch 完成之前注入下一个 batch (要等上一个 batch 的 iteration 全部跑完)

That is, Triton cannot inject the next request C to FasterTransformer until the current batch AB finishes. To enable pipelined execution of multiple inter-layer partitions under such constraint, FasterTransformer splits a batch of requests into multiple microbatches [28] and pipelines the executions of partitions across the microbatches. 
>  也就是说，Triton 在当前 batch AB 完成前不能将下一个请求 C 注入 FasterTransformer
>  为了在这种限制下实现多个层间 partition 的流水线执行，FasterTransformer 将一个 batch 的 requests 划分为多个 microbatches，然后流水线执行这些 microbatches

In Figure 8b, FasterTransformer splits the batch AB into two microbatches, A and B. Since each partition processes a microbatch (which is smaller than the original batch) in the batched manner, the performance gain from batching can become smaller. Moreover, this method may insert bubbles into the pipeline when the microbatch size is too large, making the number of microbatches smaller than the number of partitions. While FasterTransformer needs to trade batching efficiency (larger microbatch size) for pipelining efficiency (fewer pipeline bubbles), ORCA is free of such a tradeoff --thanks to iteration-level scheduling -and can easily pipeline requests without dividing a batch into microbatches.
>  Figure 8b 中，FasterTransformer 将 batch AB 划分为两个 microbatches A, B
>  由于 microbatch 比完整的 batch 小，故批处理的性能提升优势会变小
>  此外，当 microbatch 大小太大时，使得 microbatch 的数量小于 partitions 的数量时，这种方法会在流水线中注入气泡
>  FasterTransformer 需要在 batching 效率 (更大的 microbatch size) 和流水线效率 (更少的流水线气泡) 之间进行权衡，ORCA 则避免了这种权衡，无需将一个 batch 拆分为 microbatches

# 5 Implementation
We have implemented ORCA with 13K lines of C++, based on the CUDA ecosystem. We use gRPC [2] for the communication in the control plane of the ORCA engine, while NCCL [5] is used in the data plane, for both inter-layer and intra-layer communication. Since we design ORCA to focus on Transformer-based generative models, ORCA provides popular Transformer layers as a building block of models including the original encoder-decoder Transformer [60], GPT [47], and other variants discussed in Raffel et al. [49].
>  ORCA 基于 CUDA 生态系统实现，gRPC 用于 engine 在控制层面的通信，NCCL 用于 engine 在数据层面的通信

We have also implemented fused kernels for LayerNorm, Attention, and GeLU operators, just like other systems for training or inference of Transformer models [1, 4, 58]. For example, the procedure of computing dot products between Attention query and keys, Softmax on the dot products, and weighted average of Attention values are fused into a single CUDA kernel for the Attention operator. In addition, we go one step further and fuse the kernels of the split Attention operators by simply concatenating all thread blocks of the kernels for different requests. Although this fusion makes the thread blocks within a kernel have different characteristics and lifetimes (which is often discouraged by CUDA programming practice) because they process tensors of different shapes, we find this fusion to be beneficial by improving GPU utilization and reducing the kernel launch overhead [34, 39].
>  我们也为 LayerNorm, Attention, GeLU 算子设计了融合算子
>  例如计算 Attention Q, K 之间的点积、在点积结果上的 softmax 和对 Attention V 的加权平均被融合为单个 CUDA kernel，称为 Attention operator
>  此外，我们进一步将不同 requests 的所有线程块连接起来，实现了 split Attention operators 的融合，这种融合使得 kernel 内的线程块具有不同的特性和生命周期，因为它们这些线程块处理的是不同形状的 tensors，但我们发现这种融合有助于提高 GPU 利用率并减少 kernel 启动开销

# 6 Evaluation
In this section, we present evaluation results to show the efficiency of ORCA.

**Environment.** We run our evaluation on Azure ND96asr A100 v4 VMs, each equipped with 8 NVIDIA 40-GB A100 GPUs connected over NVLink. We use at most four VMs depending on the size of the model being tested. Each VM has 8 Mellanox 200Gbps HDR Infiniband adapters, providing an  $1.6\mathrm{Tb / s}$  of interconnect bandwidth between VMs.

Table 1: Configurations of models used in the experiments.  

<center><table><tr><td># Params</td><td># Layers</td><td>Hidden size</td><td># Inter-partitions</td><td># Intra-partitions</td></tr><tr><td>13B</td><td>40</td><td>5120</td><td>1</td><td>1</td></tr><tr><td>101B</td><td>80</td><td>10240</td><td>1</td><td>8</td></tr><tr><td>175B</td><td>96</td><td>12288</td><td>2</td><td>8</td></tr><tr><td>341B</td><td>120</td><td>15360</td><td>4</td><td>8</td></tr></table></center>

**Models.** Throughout the experiments, we use GPT [12] as a representative example of Transformer-based generative models. We use GPT models with various configurations, which is listed in Table 1. The configurations for 13B and 175B models come from the GPT-3 paper [12]. Based on these two models, we change the number of layers and hidden size to make configurations for 101B and 341B models. All models have a maximum sequence length of 2048, following the setting of the original literature [12]. We use fp16-formatted model parameters and intermediate activations for the experiments. We also apply inter-and intra-layer parallelism strategies described in Section 4.1, except for the 13B model that can fit in a GPU. For example, the 175B model is partitioned over a total of 16 GPUs by using 2 inter-layer partitions subdivided into 8 intra-layer partitions, where the 8 GPUs in the same VM belongs to the same inter-layer partition.
>  模型参数和中间激活都使用 fp16

**Baseline system.** We compare with FasterTransformer [4], an inference engine that supports large scale Transformer models via distributed execution. While there exist other systems with the support for distributed execution such as Megatron-LM [3] and DeepSpeed [1], these systems are primarily designed and optimized for training workloads, which makes them show relatively lower performance compared to the inference-optimized systems.

**Scenarios.** We use two different scenarios to drive our evaluation. First, we design a microbenchmark to solely assess the performance of the ORCA engine without being affected by the iteration-level scheduling. In particular, we do not run the ORCA scheduler in this scenario. Instead, given a batch of requests, the testing script repeats injecting the same batch into the ORCA engine until all requests in the batch finishes, mimicking the behavior of the canonical request-level scheduling. We also assume that all requests in the batch have the same number of input tokens and generate the same number of output tokens. We report the time taken for processing the batch (not individual requests) and compare the result with FasterTransformer [4].
>  我们设计两个不同的场景进行评估
>  首先，我们设计了一个微基准测试，仅用于评估 ORCA engine 的性能，不受 iteration-level 调度的影响
>  在这个场景下，我们不运行 ORCA scheduler，给定 a batch of requests，测试脚本重复将相同的 batch 注入到 engine 中，直到 batch 中的所有 requests 都完成模仿传统的 request-level 调度

The second scenario tests the end-to-end performance of ORCA by emulating a workload. We synthesize a trace of client requests because there is no publicly-available request trace for generative language models. Each request in the synthesized trace is randomly generated by sampling the number of input tokens and a max_gen_tokens attribute, where the number of input tokens plus max_gen_tokens equals to the max_tokens attribute described in Section 4.2. We assume that all requests continue generation until the number of generated tokens reaches max_gen_tokens. In other words, we make the model never emit the "\<EOS\>" token. This is because we have neither the actual model checkpoint nor the actual input text so we do not have any information to guess the right timing of the "\<EOS\>" token generation. Once the requests are generated, we synthesize the trace by setting the request arrival time based on the Poisson process. To assess ORCA's behavior under varying load, we change the Poisson parameter (i.e., arrival rate) and adjust the request arrival time accordingly. We report latency and throughput using multiple traces generated from different distributions for better comparison and understanding of the behavior of ORCA and FasterTransformer.
>  第二个场景是端到端测试，我们合成了一组客户端 requests 轨迹

## 6.1 Engine Microbenchmark
We first compare the performance of FasterTransformer and the ORCA engine using the first scenario. We set all requests in the batch to have the same number of input tokens (32 or 128) and generate 32 tokens. That is, in this set of experiments, all requests within the batch start and finish processing at the same time. We conduct experiments using three different models: 13B, 101B, and 175B. For each model, we use the corresponding parallelization strategy shown in Table 1.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/00ab9a729bf3fe597285392ac8afa0eeeee1ab77fb761d554931009102ae4e7b.jpg)  

Figure 9: Execution time of a batch of requests using FasterTransformer and the ORCA engine without the scheduling component. Label "ft (n)" represents results from FasterTransformer processing requests with  $n$  input tokens. Configurations that incurs out of memory error are represented as missing entries (e.g., ft (32) for the 101B model with a batch size of 16).

Figure 9 shows the performance of FasterTransformer and the ORCA engine for processing a batch composed of the same requests. In Figure 9a, the ORCA engine shows a similar (or slightly worse) performance compared to FasterTransformer across all configurations. This is because ORCA does not apply batching to the Attention operations, while FasterTransformer apply batching to all operations. Still, the performance difference is relatively small. Despite not batching the Attention operation, the absence of model parameters in Attention makes this decision has little impact on efficiency as there is no benefit of reusing model parameters across multiple requests.
>  ORCA engine 没有批处理 Attention 计算，但和 FasterTransformer 的性能差距相对较小
>  虽然没有批处理 Attention 计算，但由于 Attention 计算不涉及模型参数，故实际的影响不大，因为批处理之后也不会带来跨多个 requests 的模型参数复用优势

Figure 9b presents similar results for the 101B model that uses all of the 8 GPUs in a single VM. From these results, we can say that the ORCA engine and FasterTransformer have comparable efficiencies in the implementations of CUDA kernels and the communication between intra-layer partitions. Note that FasterTransformer cannot use a batch size of 8 or larger with the 13B model (16 or larger with the 101B model) because of the fixed amount of memory pre-allocation for each request's Attention keys and values, which grows in proportion to the max sequence length of the model (2048 for this case). In contrast, ORCA avoids redundant memory allocation by setting the size of buffers for the keys and values separately for each request based on the max_tokens attribute.
>  ORCA engine 和 FasterTransformer 在 CUDA kernel 的实现以及层内 partition 的通信实现上具有可比的性能
>  ORCA 根据每个请求的 `max_tokens` 属性来预留 KVCache，FasterTransformer 则根据模型的最大序列长度来预留 KVCache，因此 ORCA 的内存效率略微更高

Next, we go one step further and experiment with the 175B model, which splits the layers into two inter-layer partitions. In this case, for better comparison, we disable pipelined execution of the inter-layer partitions for both systems. For FasterTransformer, we set the size of a microbatch to be equal to the batch size to disable pipelining. As shown in Figure 9c, the ORCA engine outperforms FasterTransformer by up to  $47\%$ . We attribute this performance improvement to the control-data plane separation described in Section 4.1. We omit the 341B model as it has similar results compared to the 175B model.
>  面对更大的模型时，由于 ORCA 的控制数据通信和实际数据通信分离机制，ORCA engine 比 FasterTransformer 更优

## 6.2 End-to-end Performance
Now we assess the end-to-end performance of ORCA by measuring the latency and throughput with the synthesized request trace under varying load. When synthesizing the trace, we sample each request's number of input tokens from  $U(32,512)$ , a uniform distribution ranging from 32 to 512 (inclusive). The max_get_tokens attributed is sampled from  $U(1,128)$ , which means that the least and the most time-consuming requests require 1 and 128 iterations of the model for processing, respectively.

Unlike the microbenchmark shown in Section 6.1, to measure the end-to-end performance, we test the entire ORCA software stack including the ORCA scheduler. Client requests arrive to the ORCA scheduler following the synthesized trace described above. We report results from various max batch size configurations. 

For FasterTransformer that does not have its own scheduler, we implement a custom scheduler that receives client requests, creates batches, and injects the batches to an instance of FasterTransformer. We make the custom scheduler create batches dynamically by taking at most max batch size requests from the request queue, which is the most common scheduling algorithm used by existing serving systems like Triton [7] and TensorFlow Serving [42]. Again, we report results from various max batch size configurations, along with varying microbatch sizes, an additional knob in FasterTransformer that governs the pipelining behavior (see Section 4.2).

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/d950224f4e05a0eb2b6524058a50115d849232029545afa8d339c0b2366d8c07.jpg)  

Figure 10: Median end-to-end latency normalized by the number of generated tokens and throughput. Label "orca(max_bs)" represents results from ORCA with a max batch size of max_bs. Label "ft(max_bs, mbs)" represents results from FasterTransformer with a max batch size of max_bs and a microbatch size of mbs.

Figure 10 shows median end-to-end latency and throughput. Since each request in the trace requires different processing time, which is (roughly) in proportion to the number of generated tokens, we report median latency normalized by the number of generated tokens of each request. 
>  因为轨迹中的每个 request 需要的处理时间大致和生成的 tokens 数量成比例，因此我们汇报每个 request 的规范化 latency

From the figure, we can see that ORCA provides significantly higher throughput and lower latency than FasterTransformer. The only exception is the 101B model under low load (Figure 10a). In this case, both ORCA and FasterTransformer do not have enough number of requests to process in a batch. That is, the latency will mostly depend on the engine's performance, which is shown in Figure 9b. 

As the load becomes heavier, ORCA provides higher throughput with a relatively small increase in latency, because the ORCA scheduler makes late-arrived requests hitch a ride with the current ongoing batch. In contrast, FasterTransformer fails to efficiently handle multiple requests that (1) arrive at different times; (2) require different number of iterations to finish; or (3) start with different number of input tokens, resulting in a peak throughput of 0.49 req/s and much higher latency. If we use the 175B or 341B model (Figures 10b and 10c) that employs more than one inter-layer partitions, ORCA outperforms FasterTransformer under every level of load in terms of both latency and throughput, resulting in an order of magnitude higher throughput when we compare results at a similar level of latency. For example, to match a median normalized latency of 190ms for the 175B model, which is a double of the normalized execution time (by the number of generated tokens) of "orca(128)" shown in Figure 9c, FasterTransformer provides a throughput of 0.185 req/s whereas ORCA provides a throughput of 6.81 req/s, which is a  $36.9\times$  speedup.
>  ORCA 的 scheduler 非常优秀，可以随时调度在不同时间到达、需要不同数量迭代、不同输入 tokens 数量的 requests

**Varying batch size configurations.** Figure 10 shows that the increase of the max batch size of ORCA results in a higher throughput without affecting the latency. This is because the iteration-level scheduling of ORCA resolves the problem of early-finished and late-joining requests. Nevertheless, there is no guarantee that increasing the batch size will not negatively affect the latency, for arbitrary hardware settings, models, and workloads. As mentioned in Section 4.2, the max batch size must be set carefully by considering both the required latency and throughput requirements.
>  iteration-level 调度解决了 requests 的到达时间交错问题

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-22/34e13cd9-31e2-42b6-bb35-76c0ba84e79d/f6466fc400ba300bd09f0e7b6443e832de7b76f9fa8cb840138ea7eabf53ba0c.jpg)  

Figure 11: Median end-to-end latency and throughput, using the 175B model with traces composed of homogeneous requests. We do not normalize the latency since all requests have the same characteristic.

Interestingly, larger max batch size in FasterTransformer does not necessarily help improving throughput. By testing all possible combinations of max batch size  $(max\_ bs)$  and microbatch size  $(mbs)$  on all models under varying load, we find that  $(max\_ bs, mbs) = (1, 1)$  or  $(8, 8)$  are the best options. Per our discussion in Section 4.1, FasterTransformer's microbatch-based pipelining can be less efficient because the engine is going to process at most  $mbs$  number of requests in the batched manner, which explains why the configurations with the maximum possible  $mbs$  (which is the same as  $max\_ bs$ ) have better performance than others. In addition, while increasing  $max\_ bs$  can improve performance due to the increased batch size, at the same time, this also increases the likelihood of batching requests with large difference in the number of input tokens or the number of generated tokens. In such cases, FasterTransformer cannot efficiently handle the batch because (1) for the first iteration of the batch, FasterTransformer processes requests as if they all had the same input length as the shortest one; and (2) early-finished requests cannot immediately return to the clients.
>  更大的 batch size 并不会提高 FasterTransformer 的吞吐
>  就如之前的讨论，microbatch 机制会减少一次性批处理的 requests 数量，进而减少了批处理的效率

**Trace of homogeneous requests.** We test the behavior of ORCA and FasterTransformer when using a trace of homogeneous requests, i.e., all requests in a trace have the same number of input tokens and the same max_gen_tokens attribute. Since all requests require the same number of iterations to finish processing, the problem of early-leaving requests does not occur for this trace. As a result, now the increase of the  $max\_ bs$  has a noticeable positive impact on the performance of FasterTransformer, as shown in Figure 11. Still, ORCA outperforms FasterTransformer  $(max\_ bs = 8)$  except for the case using a max batch size of 1, where ORCA degenerates into a simple pipeline of the ORCA workers that does not perform batching.

# 7 Related Work and Discussion
**Fine-grained batching for recurrent models.** We would like to highlight BatchMaker [23] as one of the most relevant previous works. BatchMaker is a serving system for RNNs that performs scheduling and batching at the granularity of RNN cells, motivated by the unique RNN characteristic of repeating the same computation. Once a request arrives, BatchMaker breaks the dataflow graph for processing the request into RNN cells, schedules execution at the granularity of cells (instead of the entire graph), and batches the execution of identical cells (if any). Since each RNN cell always performs the exact same computation, BatchMaker can execute multiple RNN cells in a batched manner regardless of the position (i.e., token index) of the cell. By doing so, BatchMaker allows a newly arrived request for RNN to join (or a finished request to leave) the current executing batch without waiting for the batch to completely finish.

However, BatchMaker cannot make batches of cells for Transformer models because there are too many distinct cells (a subgraph that encapsulates the computation for processing a token; Figure 1c) in the graph. Each cell at a different token index  $t$  must use a different set of Attention Keys/Values. As the cell for each  $t$  is different, the graph comprises  $L$  different cells  $L$  denotes the number of input and generated tokens), significantly lowering the likelihood of cells of the same computation being present at a given moment (e.g., in Figure 10,  $L$  ranges from  $33 = 32 + 1$  to  $640 = 512 + 128$ ). Thus execution of the cells will be mostly serialized, making BatchMaker fall back to non-batched execution. BatchMaker also lacks support for large models that require model and pipeline parallelism.

While BatchMaker is geared towards detecting and aligning batch-able RNN cells, our key principle in designing ORCA is to perform as much computation as possible per each round of model parameter read. This is based on the insight that reading parameters from GPU global memory is a major bottleneck in terms of end-to-end execution time, for large-scale models. Adhering to this principle, we apply iteration-level scheduling and selective batching to process all "ready" tokens in a single round of parameter read, regardless of whether the processing of tokens can be batched (non-Attention ops) or not (Attention ops).

**Specialized execution engines for Transformer models.** The outstanding performance of Transformer-based models encourages the development of inference systems specialized for them. FasterTransformer [4], LightSeq [61], TurboTransformers [22] and EET [36] are such examples. Each of these systems behave as an backend execution engine of existing serving systems like Triton Inference Server [7] and TensorFlow Serving [42]. That is, these systems delegate the role of scheduling to the serving system layer, adhering to the canonical request-level scheduling.

Instead, ORCA suggests to schedule executions at a finer granularity, which is not possible in current systems without changing the mechanism for coordination between the scheduler and the execution engine. Note that among these systems, FasterTransformer is the only one with the support for distributed execution. While systems like Megatron-LM [3] and DeepSpeed [1] can also be used for distributed execution, these systems are primarily optimized for large-scale training rather than inference serving.

**Interface between serving systems and execution engines.** Current general-purpose serving systems such as Triton Inference Server [7] and Clipper [16] serve as an abstraction for handling client requests and scheduling executions of the underlying execution engines. This approach is found to be beneficial by separating the design and implementation of the serving layer and the execution layer. However, we find that the prevalent interface between the two layers is too restricted for handling models like GPT [12], which has the multi-iteration characteristic. Instead, we design ORCA to tightly integrate the scheduler and the engine, simplifying the application of the two proposed techniques: iteration-level scheduling and selective batching. While in this paper we do not study a general interface design that supports the two techniques without losing the separation of abstractions, it can be an interesting topic to explore such possibility; we leave this issue to future work.

# 8 Conclusion
We present iteration-level scheduling with selective batching, a novel approach that achieves low latency and high throughput for serving Transformer-based generative models. Iteration-level scheduling makes the scheduler interact with the execution engine at the granularity of iteration instead of request, while selective batching enables batching arbitrary requests processing tokens at different positions, which is crucial for applying batching with iteration-level scheduling. Based on these techniques, we have designed and implemented a distributed serving system named ORCA. Experiments show the effectiveness of our approach: ORCA provides an order of magnitude higher throughput than current state-of-the-art systems at the same level of latency.
>  我们提出了 iteration-level scheduling 和 selective batching
>  iteration-level scheduling 使得 scheduler 以 iteration-level 的粒度和 engine 交互
>  selective batching 使得可以将处理不同位置 token 的 requests 进行批处理，这是对 iteration-level scheduling 进行批处理的关键
>  基于这两个技术，我们设计并实现了一个分布式服务系统 ORCA


