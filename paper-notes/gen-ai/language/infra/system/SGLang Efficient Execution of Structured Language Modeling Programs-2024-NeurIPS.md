# Abstract
Large language models (LLMs) are increasingly used for complex tasks that require multiple generation calls, advanced prompting techniques, control flow, and structured inputs/outputs. However, efficient systems are lacking for programming and executing these applications. 
>  LLM 正在越来越多地被使用于需要多次 generation 调用、高级提示技术、控制流和结构化输入输出的复杂任务
>  现存的系统缺少编程这些任务以及执行这些任务的能力

We introduce SGLang, a system for efficient execution of complex language model programs. SGLang consists of a frontend language and a runtime. The frontend simplifies programming with primitives for generation and parallelism control. The runtime accelerates execution with novel optimizations like RadixAttention for KV cache reuse and compressed finite state machines for faster structured output decoding. 
>  SGLang 是一个高效执行复杂语言模型程序的系统
>  SGLang 由一个前端语言和一个运行时组成，前端语言提供了生成和并行控制的原语来简化编程，运行时则通过优化的技术来加速执行，例如基于 RadixAttention 进行 KV cache 复用以及基于压缩的有限状态机进行更快的结构化输出解码

Experiments show that SGLang achieves up to  $6.4\times$  higher throughput compared to state-of-the-art inference systems on various large language and multi-modal models on tasks including agent control, logical reasoning, few-shot learning benchmarks, JSON decoding, retrieval-augmented generation pipelines, and multi-turn chat. The code is publicly available at https://github.com/sgl-project/sglang.
>  SGLang 在各种任务基准测试包括智能体控制、逻辑推理、小样本学习、JSON 解码、检索增强的生成流水线、多轮对话中，相较于 SOTA 的推理系统系统，为各种大语言模型和多模态模型实现了 6.4x 的吞吐

# 1 Introduction
Recent increases in the capabilities of LLMs have broadened their utility, enabling them to tackle a wider range of general tasks and act as autonomous agents [35, 6, 36, 52, 46]. In such applications, LLMs engage in multi-round planning, reasoning, and interaction with external environments. This is accomplished through tool usage [41, 38], multiple input modalities [47, 2], and a wide range of prompting techniques [30], like few-shot learning [5], self-consistency [53], skeleton-of-thought [33], and tree-of-thought [56]. All of these new use cases require multiple, often dependent, LLM generation calls, showing a trend of using multi-call structures to complete complex tasks [57, 21].
>  LLM 已经作为自动智能体，用于解决一系列任务，包括了多轮规划、推理、和外部环境的交互
>  在这些任务中，LLM 会使用工具、接收多模态输入、接收一系列提示词技巧 (例如小样本学习、自一致性、思维框架、思维树等)
>  这些任务都需要多次的 (且一般是独立的) LLM 生成调用
>  这说明了解决复杂任务的趋势是多轮调用结构

The emergence of these patterns signifies a shift in our interaction with LLMs, moving from simple chatting to a more sophisticated form of programmatic usage of LLMs, which means using a program to schedule and control the generation processes of LLMs. We refer to these programs as "Language Model Programs" (LM Programs) [4, 20]. The advanced prompting techniques and agentic workflow mentioned above fall within the scope of LM programs. There are two common properties of LM programs: (1) LM programs typically contain multiple LLM calls interspersed with control flow. This is needed to complete complex tasks and improve overall quality. (2) LM programs receive structured inputs and produce structured outputs. This is needed to enable the composition of LM programs and to integrate LM programs into existing software systems.
>  这种模式的出现表示了我们和 LLM 交互形式的改变，从简单的对话到更复杂的，以编程形式的使用方式，即使用一个程序来调度和控制 LLM 的生成过程
>  我们称这类程序为语言模型程序，上述提到的提示技巧和智能体工作流均属于语言模型程序
>  语言模型程序通常有两个特征:
>  1. 通常包含多次 LLM 调用，并穿插着控制流，以解决复杂任务
>  2. 接收结构化的输入，生成结构化的输出，以和现有的软件系统集成

Despite the widespread use of LM programs, current systems for expressing and executing them remain inefficient. We identify two primary challenges associated with the efficient use of LM programs: First, programming LM programs is tedious and difficult due to the non-deterministic nature of LLMs. Developing an LM program often requires extensive string manipulation, experimenting tuning of prompts, brittle output parsing, handling multiple input modalities, and implementing parallelism mechanisms. This complexity significantly reduces the readability of even simple programs (Sec. 2).
>  虽然语言模型程序被广泛使用，但当前用于表示和执行它们的系统并不高效
>  高效使用语言模型程序的挑战主要有两个:
>  1. 由于 LLM 的非确定性特性，编写 LM 程序枯燥且困难
>  开发一个 LM 程序通常涉及大量字符串操作、反复调试提示词、脆弱的输出解析、处理多种模态输入、实现并行机制，这种复杂性显著降低了 LM 程序的可读性

Secondly and importantly, executing LM programs is inefficient due to redundant computation and memory usage. State-of-the-art inference engines (e.g., vLLM [23], TGI [16], and TensorRT-LLM [34]), have been optimized to reduce latency and improve throughput without direct knowledge of the workload. This makes these systems general and robust but also results in significant inefficiencies for any given workload. 
>  2. 由于冗余的计算和内存使用，LM 程序的执行效率低下
>  当前 SOTA 的推理引擎 (例如 vLLM, TGI, TensorRT-LLM) 虽然在降低延迟和提升吞吐量方面进行了优化，但这些优化并未基于对 workload 的直接知识
>  这使得这些系统虽然具有通用性和健壮性，但是在特定的 workload 下存在显著的效率损失

A prominent example is the reuse of the Key-Value (KV) cache (Sec. 3). The KV cache consists of reusable intermediate tensors that are essential for generative inference. During typical batch executions of LM programs, numerous opportunities exist to reuse the KV cache across multiple different LLM calls that share a common prefix. However, current systems lack effective mechanisms to facilitate this reuse, resulting in unnecessary computations and wasted memory. 
>  一个典型的例子是对 KVCache 的复用，KVCache 由可以复用的中间张量构成，是生成式推理中的关键组件
>  在典型的 LM 程序批量执行场景下，存在大量机会可以在多个共享前缀的 LLM 调用中复用 KVCache
>  当前的系统缺乏有效的机制来实现这种复用，导致了不必要的计算和内存浪费

Another example is constrained decoding for structured outputs (e.g., JSON mode), where the output of LLMs is restricted to follow specific grammatical rules defined by a regular expression (Sec. 4). Under these constraints, multiple tokens can often be decoded once. However, existing systems only decode one token at a time, leading to suboptimal decoding speeds.
>  另一个例子是约束解码以生成结构化输出 (例如 JSON 模式)，该场景下，LLM 的输出需要遵守由正则表达式定义的特定语法规则
>  在这些约束下，通常可以一次解码多个 tokens，而现有的系统一次仅解码一个 token

![[pics/SGLang-Fig1.png]]

To address these challenges, we present SGLang, a Structured Generation Language for LLMs. The core idea is to systematically exploit the multi-call structure in LM programs for efficient execution. As shown in Fig. 1, it has two parts: a front-end language and a back-end runtime. The front-end simplifies the programming of LM programs, and the runtime accelerates their execution. The two parts can work together for better performance but can also function independently.
>  SGLange 是一个针对 LLM 的结构化生成语言
>  其核心思想是系统地利用 LM 程序中的多次调用结构来提高执行效率
>  如 Fig1 所示，它有两个部分: 一个前端语言和一个后端运行时
>  前端语言简化了 LM 程序的编写，后端运行时加速了 LM 程序的执行

We introduce SGLang as a domain-specific language embedded in Python. It provides primitives for generation (e.g., extend, gen, select) and parallelism control (e.g., fork, join). SGLang is compatible with Python's control flow and libraries, so users can develop advanced prompting workflows easily with native Python syntax. We provide an interpreter and a compiler for SGLang. The interpreter manages the prompt state as a stream and submits primitive operations to the stream for asynchronous execution, ensuring proper control over synchronization and intra-program parallelism. Additionally, SGLang program can be traced and compiled for more optimizations.
>  SGLang 是一个嵌入在 Python 中的 DSL，它为生成提供了原语 (例如 `extend, gen, select`)，也为并行控制提供了原语 (例如 `fork, join`)
>  SGLang 和 Python 的控制流和 Python 库兼容
>  我们为 SGLang 提供了一个解释器和编译器，解释器以流的形式管理提示词状态，同时将原语操作提交到流中，以进行异步执行，并确保对同步和 program 内并行的正确控制
>  此外，SGLang 程序还可以被追踪和编译，实现更多优化

On the runtime side, we propose several novel optimizations to accelerate the execution of SGLang programs. 
>  在运行时上，我们提出了几个优化来加速 SGLang 程序的执行

The first technique, RadixAttention, enables the automatic reuse of the KV cache across multiple generation calls. In existing inference engines, the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls and significantly slowing down the execution. Instead, our system maintains an LRU cache of the KV cache for all requests within a radix tree. This approach manages the KV cache as a traditional cache and uses a radix tree for efficient matching, insertion, and eviction. It allows the runtime to handle various reuse patterns with a cache-aware scheduling policy efficiently. 
>  第一个技术是 RadixAttention，它实现了在多个生成调用中自动复用 KVCache
>  在现存的推理引擎中，一个请求的 KVCache 会在它的处理完成之后就被丢弃，使得 KVCache 不能跨调用复用
>  我们的系统在一个基数树上为所有请求的 KVCache 维护了 LRU 缓存，这个方法将 KVCache 视为传统的缓存，并使用基数树来实现高效的匹配、插入和淘汰操作
>  这种设计使得运行时可以通过缓存感知的调度策略来处理各种复用模式

The second technique is a compressed finite state machine, which enables faster constrained decoding for structured outputs. Existing systems follow the constraints only for the next token by masking probabilities of disallowed tokens, making them able to decode only one token at a time. Instead, our system analyzes the constraints and builds a compressed finite-state machine to represent the constraint. This approach compresses a multi-token path into a single-step path whenever possible, allowing the decoding of multiple tokens at once to achieve faster decoding speed. 
>  第二个技术是压缩的有限状态机，实现了针对结构化输出的更快的约束解码
>  现存的系统通过屏蔽不允许的 token 的概率，仅对下一个 token 的生成进行约束，故一次只能解码一个 token
>  我们的系统会分析约束，并构建一个压缩的有限状态机来表示约束，该方法尽可能将多 token 路径压缩为单步路径，进而支持一次解码多个 tokens

Lastly, SGLang also supports API-only models like OpenAI's GPT-4, and we introduce the third technique, API speculative execution, to optimize multi-call programs for API-only models.
>  最后，SGLang 也支持仅有 API 的模型，我们通过 API 投机执行来优化 API-only 模型的多调用程序

Using SGLang, we implemented various LLM applications, including agent control, logical reasoning, few-shot learning benchmarks, JSON decoding, retrieval-augmented generation pipelines, multi-turn chat, and multi-modality processing. We tested the performance on models including Llama-7B/70B [49], Mistral-8x7B [17], LLaVA-v1.5-7B (image) [28], and LLaVA-NeXT-34B (video) [62] on NVIDIA A10G and A100 GPUs. Experimental results show that SGLang achieves up to  $6.4 \times$  higher throughput across a wide range of workloads, models, and hardware setups, compared to existing programming and inference systems, including Guidance [13], vLLM [23], and LMQL [4].

# 2 Programming Model
This section introduces the SGLang programming model with a running example, describes its language primitives and execution modes, and outlines runtime optimization opportunities. This programming model can simplify tedious operations in multi-call workflows (e.g., string manipulation, API calling, constraint specification, parallelism) by providing flexible and composable primitives.
>  本节通过实际实例介绍 SGLang 编程模型，描述其语言原语和执行模式，并概述了运行时优化的机会
>  SGLang 编程模型提供了灵活和可组合的原语来简化多调用工作流 (例如字符串处理，API 调用、约束定义和并行) 中的繁琐操作

![[pics/SGLang-Fig2.png]]

**A running example.** The language is a domain-specific language embedded in Python. Fig. 2 shows a program that evaluates an essay about an image using the branch-solve-merge prompting method [40]. 
>  SGLang 是一个嵌入在 Python 中的 DSL
>  Fig2 是一个使用了分支-求解-合并的提示词方法来评估一篇关于图像的论文的 LM 程序

The function `multi_dimensional_judge` takes three arguments: s, path, and essay. s manages the prompt state, path is the image file path, and essay is the essay text. New strings and SGLang primitives can be appended to the state `s`  for execution using the += operator. First, the function adds the image and essay to the prompt. It then checks if the essay is related to the image using select, storing the result in `s["related"]`. If related, the prompt is forked into three copies for parallel evaluation from different dimensions, using `gen` to store results in `f["judgment"]`. Next, it merges the judgments, generates a summary, and assigns a letter grade. Finally, it returns the results in JSON format, following a schema defined by a regular expression constraint regex. 
>  函数 `multi_dimensional_judge` 接收三个参数 `s, path, essay` ，`s` 管理 prompt 状态，`path` 为图像文件路径，`essay` 为论文文本
>   我们可以通过使用 `+=` 运算符来将新的字符串和 SGLang 原语追加到状态 `s` 中以供执行
>  该函数首先将图像和论文都加入到 prompt 中，然后使用 `select` 检查论文是否与图像相关，并将结果存储在 `s["related"]` 中
>  如果相关，通过 `fork` 将 prompt 复制为三份 (三个维度)，使用 `gen` 从不同的维度进行并行评估，并将结果存储到 `f["judgment"]`
>  然后，它将判断结果合并，提供合并的判断结果让 LLM 生成总结，并分配一个字母等级
>  最后，它返回 JSON 格式的结果，格式遵循由正则表达式约束 `regex` 定义的模式

SGLang greatly simplifies this program, as an equivalent program using an OpenAI API-like interface would take  $2.1\times$  as many lines of code due to manual string manipulation and parallelism control.
>  使用 SGLang 编写这个程序简化了实现，若使用 OpenAI API 类似的接口实现这个功能，代码量要增加 2.1x，因为需要手动处理字符串拼接和并行控制

**Language primitives.** SGLang provides primitives for controlling prompt state, generation, and parallelism. They can be used together with Python syntax and libraries. Here are the primitives: "gen" calls a model to generate and stores the results in a variable with the name specified in its first argument. It supports a "regex" argument to constrain the output to follow a grammar defined by a regular expression (e.g., a JSON schema). "select" calls a model to choose the highest probability option from a list. The operator  += or "extend" appends a string to the prompt. The operator "`[variale_name]`" fetches the results of a generation. "fork" creates parallel forks of the prompt state. "join" rejoins the prompt state. "image" and "video" take in image and video inputs.
>  SGLang 为控制 prompt 状态、生成和并行提供了原语，它们可以和 Python 语法和库共同使用
>  `gen` 调用一个模型来进行生成，并将生成结果存储在作为它第一个参数的变量中，它支持 `regex` 参数来约束其输出遵循由正则表达式指定的语法 (例如 JSON 模式)
>  `select` 调用一个模型来从一个列表中选择最高概率的选项
>  运算符 `+=` ，或 `extend` 将字符串追加到 prompt 后
>  索引运算符 `[variable_name]` 获取生成的结果 (存储在变量 `variable_name` 中的结果)
>  `fork` 为 prompt 状态创建并行的分支
>  `join` 合并多个 prompt 状态的分支
>  `image, video` 接收图像和视频输入

**Execution modes.** The simplest way to execute an SGLang program is through an interpreter, where a prompt is treated as an asynchronous stream. Primitives like extend, gen, and select are submitted to the stream for asynchronous execution. These non-blocking calls allow Python code to continue running without waiting for the generation to finish. This is similar to launching CUDA kernels asynchronously. Each prompt is managed by a stream executor in a background thread, enabling intra-program parallelism. 
>  执行 SGLang 程序的最简单形式是通过解释器，解释器将 prompt 视作一个异步流，像 `extend, gen, select` 这样的原语会被提交到流中进行异步执行
>  这些非阻塞的调用可以让 Python 代码继续执行，无需等待生成完成，类似于异步地发起 CUDA kernels
>  每个 prompt 都由一个后台线程的一个流执行器管理，实现了程序内的并行

Fetching generation results will block until they are ready, ensuring correct synchronization. Alternatively, SGLang programs can be compiled as computational graphs and executed with a graph executor, allowing for more optimizations. This paper uses interpreter mode by default and discusses compiler mode results in Appendix D. SGLang supports open-weight models with its own SGLang Runtime (SRT), as well as API models such as OpenAI and Anthropic models.
>  获取生成结果的调用会阻塞，直到结果可用
>  此外，SGLang 程序可以被编译为计算图，被图执行器执行，实现更多优化
>  本文默认使用解释器模式
>  SGLang 使用自己的 SGLang Runtime 执行开源的模型，也支持 API-only 的模型

>  SGLang 中，prompt 不是指文本，而是表示一个动态、可变的状态，它会随着程序运行不断被添加内容，可以将它理解为一个正在处理的任务流程
>  流执行器就是执行这个流的调度者，它知道什么时候调用模型进行生成，什么时候等待，什么时候继续，它在后台线程中执行，后台线程就是独立于 Python 主线程的线程
>  SGLang 中，每个 prompt 都对应一个流和一个流执行器，因此多个 prompt 就会对应多个流，这些流有各自的流执行器，故会并发执行

Table 1: Comparison among LMQL, Guidance, and SGLang.  

<table><tr><td>System</td><td>Syntax</td><td>Language Primitives</td><td>Runtime Backends</td></tr><tr><td>LMQL Guidance</td><td>Custom Python</td><td>extend, gen, select extend, gen, select, image</td><td>HF Transformers, llama.cpp, OpenAI HF Transformers, llama.cpp, OpenAI</td></tr><tr><td>SGLang</td><td>Python</td><td>extend, gen, select, image, video, fork, join</td><td>SGLang Runtime (SRT), OpenAI</td></tr></table>

**Comparison.** Programming systems for LLMs can be classified as high-level (e.g., LangChain, DSPy) and low-level (e.g., LMQL, Guidance, SGLang). High-level systems provide predefined or auto-generated prompts, such as DSPy's prompt optimizer. Low-level systems typically do not alter prompts but allow direct manipulation of prompts and primitives. SGLang is a low-level system similar to LMQL and Guidance. 
>  LLM 的编程系统可以被分类为高级和低级
>  高级的系统提供预定义的或自动生成的 prompts，例如 DSPy 的 prompt 优化器
>  低级的系统通常不会修改 prompt，但也允许对 propmt 进行直接操作，并提供原语
>  SGLang 是类似于 LMQL, Guidance 的低级系统

Table 1 compares their features. SGLang focuses more on runtime efficiency and comes with its own co-designed runtime, allowing for novel optimizations introduced later. High-level languages (e.g., DSPy) can be compiled to low-level languages (e.g., SGLang). We demonstrate the integration of SGLang as a backend in DSPy for better runtime efficiency in Sec. 6.
>  SGLang 的特点在于运行时效率
>  高级的语言 (DSPy) 可以被编译为低级语言 (SGLang)

**Runtime optimizations.** Fig. 2 shows three runtime optimization opportunities: KV cache reuse, fast constrained decoding, API speculative execution. We will discuss them in the following sections.

# 3 Efficient KV Cache Reuse with RadixAttention
SGLang programs can chain multiple generation calls and create parallel copies with the "fork" primitive. Additionally, different program instances often share some common parts (e.g., system prompts). These scenarios create many shared prompt prefixes during execution, leading to numerous opportunities for reusing the KV cache. 
>  SGLang 程序可以将多个生成调用串联，也可以使用 `fork` 原语创建并行拷贝
>  此外，不同的程序实例通常共享部分公共内容 (例如 system prompts)，这些场景在执行过程中会产生大量公共前缀，提供了许多复用 KVCache 的机会

During LLM inference, the KV cache stores intermediate tensors from the forward pass, reused for decoding future tokens. They are named after key-value pairs in the self-attention mechanism [51]. KV cache computation depends only on prefix tokens. Therefore, requests with the same prompt prefix can reuse the KV cache, reducing redundant computation and memory usage. More background and some examples are provided in Appendix A. 
>  在 LLM 推理时，前向过程中的中间张量会以 KVCache 的形式存储，在为未来的 tokens 解码时复用
>  KVCache 的计算只依赖于前缀 token，因此具有相同 prompt 前缀的请求可以复用 KVCache，以减少冗余计算，同时减少内存使用

Given the KV cache reuse opportunity, a key challenge in optimizing SGLang programs is reusing the KV cache across multiple calls and instances. While some systems explore certain KV cache reuse cases [23, 58, 18, 12], they often need manual configurations and cannot handle all reuse patterns (e.g., dynamic tree structures). Consequently, most state-of-the-art inference systems recompute the KV cache for each request. We will discuss their limitations and our differences in Sec. 7.
>  优化 SGLang 程序的一个关键挑战就是在多个调用和实例之间复用 KVCache
>  虽然一些系统探索了特定的 KVCache 复用实例，但它们通常需要人为的配置，并且不能处理所有的复用模式 (例如动态树结构)
>  因此，大多数 SOTA 推理系统为每个 request 重新计算 KVCache

>  调用一次模型的生成操作，实例指一次独立的 prompt state 处理流程，跨调用和实例复用就是在不同程序中以及相同程序中的并行任务都进行复用
>  之前的系统需要手动配置，例如告诉系统 “这两个 request 可以复用 KVCache”，又或者无法处理复杂结构，例如动态树形结构，即只能在运行时才知道哪些分支会共享前缀，不能静态预测，例如

```python
if condition:
    s += "这是关于猫的"
else:
    s += "这是关于狗的"
```

>  此时只有在运行时才知道走哪条路
>  也就是说，动态树形结构将一个程序建模为一棵树:

```
根节点： "请分析这张图"
├── 如果是猫 → 生成“视觉特征”
├── 如果是狗 → 生成“行为特征”
└── 如果是风景 → 生成“构图分析”
```

>  这棵树的分支是运行时决定的 (动态)，传统系统无法提前知道哪些分支会共享前缀，SGLang 的设计允许它在运行时自动追踪这些共享路径，并复用 KVCache 

This section introduces RadixAttention, a novel technique for automatic and systematic KV cache reuse during runtime. Unlike existing systems that discard the KV cache after a generation request finishes, our system retains the cache for prompts and generation results in a radix tree, enabling efficient prefix search, reuse, insertion, and eviction. 
>  RadixAttention 是一个在运行时进行自动和系统化 KVCache 复用的技术
>  现存的系统在 request 的生成完成后丢弃 KVCache，我们的系统会在一个基树上保留 KVCache，实现高效的前缀搜索、复用、插入和淘汰

>  Radix tree 是一个高效的字符串前缀匹配数据结构，它将多个字符串按照公共前缀组织，快速查找是否有某个前缀存在，该数据结构相较于普通哈希表更便于处理前缀相似的场景
>  例如我们要找所有以“苹果”开头的名字：
>  - 苹果手机
>  - 苹果电脑
>  - 苹果公司
>  - 苹果园
>  如果用一个树来组织这些名字，根节点是“苹果”，然后分叉出“手机”、“电脑”等，那查找起来非常快，这就是基数树的工作方式

>  在 SGLang 中，假设系统中有以下请求:

|请求 ID|提示前缀|
|---|---|
|Req1|`请分析这张图`|
|Req2|`请分析这张图` + `这是一张猫的照片`|
|Req3|`请分析这张图` + `这是一张狗的照片`|
|Req4|`请分析这张图` + `这是风景`|
|Req5|`请描述这张图`|

>  系统将这些前缀存入基数树: 

```
[根]
 |
[请分析这张图] ----------------------------
 |                    |                  |
[这是一张猫的照片]   [这是一张狗的照片]   [这是风景]
```

>  当新请求 `Req6`  到来时，系统先看它的前缀是否在树中
>  如果新请求为 `Req6: 请分析这张图 + 这是一只猫` ，就可以直接复用 `请分析这张图` 对应的 KVCache
>  这就是 Radix tree 高效前缀搜索的便利

We implement an LRU eviction policy and a cache-aware scheduling policy to enhance the cache hit rate. RadixAttention is compatible with techniques like continuous batching [60], paged attention [23], and tensor parallelism [44]. In addition, it introduces only negligible memory and time overhead when there is no cache hit.
>  我们实现了 LRU 淘汰策略和缓存感知的调度策略来提高缓存命中率
>  RadisAttention 和 continuous batching, paged attention, tensor parallelism 等技术兼容，并且它在没有缓存命中时引入的内存和时间开销很小

**RadixAttention.** A radix tree is a data structure that serves as a space-efficient alternative to a classical trie (prefix tree). Unlike typical trees, the edges of a radix tree can be labeled not just with single elements but also with sequences of elements of varying lengths, significantly enhancing efficiency. In our system, we utilize a radix tree to manage a mapping between sequences of tokens, and their corresponding KV cache tensors. These KV cache tensors are stored in a non-contiguous, paged layout, where the size of each page is equivalent to one token. 
>  基数树是一种数据结果，作为传统前缀树的空间高效的替代方案
>  和普通的树不同，基数树的边不仅可以标记单个元素，还可以标记变长的元素序列
>  我们使用基数树来管理 token 序列和其对应的 KVCache 张量的映射关系，这些 KVCache 张量以非连续的分页布局存储，每页的大小等于单个 token 的数据量

>  one page one token 的分页方法在 vLLM 中提到这样会导致内存开销过大，同时会因为元数据管理过多和内存碎片化导致性能下降
>  不过 SGLang 为了精确淘汰叶节点和自动重用共享前缀，采用了细粒度的控制

Because GPU memory is quickly filled by the KV cahce, we introduce a simple LRU eviction policy that evicts the least recently used leaf first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves and are also evicted.
>  由于 GPU 显存会迅速被 KVCache 填满，我们引入了一个简单的 LRU 淘汰策略，首先淘汰最近最少使用的叶
>  优先淘汰叶使得我们可以复用它们的公共祖先，直到其祖先变为叶并被淘汰

In the continuous batching setting, we cannot evict nodes used by the currently running batch. Therefore, each node maintains a reference counter indicating how many running requests are using it. A node is evictable if its reference counter is zero. 
>  在 continuous batching 的设定下，我们不能淘汰当前运行的 batch 使用的节点，因此，每个节点都维护一个引用计数器，用于指示当前有多少个请求正在使用它 (应该是指还没有完成的 request 的 KVCache 不能淘汰)
>  只有当节点的引用计数器变为零时，它才可以被淘汰

Note that we do not preallocate a fixed-size memory pool as a cache. Instead, we let the cached tokens and the currently running requests share the same memory pool. Therefore, the system dynamically allocates memory for cache and running requests. When enough waiting requests run, the system will evict all cached tokens in favor of a larger batch size. 
>  我们并未预先分配一个固定大小的内存池作为缓存 (存储 KVCache)，我们让 KVCache 和当前运行的 reqeust 共享相同的内存池
>  这样，系统会动态为 KVCache 和运行中的请求分配内存，当等待的请求足够多时，系统会淘汰所有的 KVCache，以腾出空间支持更大的 batch size

![[pics/SGLang-Fig3.png]]

Fig. 3 shows how the radix tree is maintained for several incoming requests. The frontend interpreter sends full prompts to the runtime, and the runtime performs prefix matching and reuse. The tree structure is stored on the CPU with negligible maintenance overhead. During the execution of the fork primitive, the frontend sends the prefix first as a hint, ensuring the prefix is correctly inserted into the tree. It then sends the remaining prompts. This "Frontend Hint" simplifies runtime scheduling and matching, exemplifying the benefits of frontend-runtime co-design.
>  Fig3 展示了为多个传入的请求维护基数树的情况
>  前端解释器将完整的 prompts 发送给运行时，运行时执行前缀匹配和复用，基数树结构存储在 CPU 中，维护开销可忽略不计
>  在执行 `fork` 原语时，前端首先发送前缀作为提示，确保前缀的被正确插入树种，再发送剩余的 prompts
>  这种前端提示机制简化了运行时的调度和匹配，体现了前端和运行时协同设计的优点 (也就是前缀提取由前端来搞定，减轻运行时负担)

**Cache-aware scheduling.** We define the cache hit rate as  $\frac{\text{number of cached prompt tokens}}{\text{number of prompt tokens}}$ . When there are many requests in the waiting queue, the order in which they are executed can significantly impact the cache hit rate. For example, if the request scheduler frequently switches between different, unrelated requests, it can lead to cache thrashing and a low hit rate. 
>  我们将缓存命中率定义为缓存的 prompts tokens 数量和 prompt tokens 的总数量的比值
>  当等待队列中有多个请求时，它们的执行顺序会显著影响缓存命中率
>  例如，如果 request 调度器经常在不同的、无关的请求之间切换，就会导致缓存抖动和低命中率

We design a cache-aware scheduling algorithm to increase the cache hit rate. In the batch-processing setting we sort the requests by matched prefix length and prioritize requests with longer matched prefixes instead of using a first-come, first-served schedule. Alg. 1 (Appendix) shows the pseudo-code for cache-aware scheduling with contiguous batching. The algorithm uses longest-shared-prefix-first order. In more latency-sensitive settings we may still be able to tolerate limited batch re-ordering to improve cache reuse. 
>  我们设计了缓存感知的调度算法来提高缓存命中率，在批处理设定下，我们通过匹配的前缀长度来对请求进行排序，优先处理具有更长匹配前缀的请求，而不是采用 FCFS 调度
>  Alg1 为连续批处理下缓存感知调度的伪代码，该算法使用最长共享前缀优先的顺序
>  在对延迟更敏感的场景下，我们仍然可以进行有限的 batch 内重排来提高缓存复用

Additionally, we prove the following theorem for optimal scheduling in the offline case.
>  此外，我们证明了离线场景下的最优调度定理

>  离线场景即所有请求在开始前就全都知道，因此系统可以提前看到所有请求的完整内容，以此决定如何调度它们来最大化缓存命中率

**Theorem 3.1.** For a batch of requests, we can achieve an optimal cache hit rate by visiting the radix tree of the requests in the depth-first search order, with a cache size $\ge$ the maximum request length. The longest-shared-prefix-first order is equivalent to a depth-first search order.
>  定理 3.1
>  对于一批请求，按照深度优先搜索对这些请求的基数树进行访问，可以达到最优的缓存命中率，前提是缓存大小大于最大的请求长度
>  最长共享前缀优先顺序等价于深度优先搜索顺序

The proof is in Sec. A.3 (Appendix). In the online case, the DFS order will be disrupted, but our schedule still approximates the DFS behavior on the augmented part of the full radix tree, as described in Sec. A.3. While greedy cache-aware scheduling can achieve high throughput, it can lead to starvation. We leave its integration with other fair scheduling methods [42] as future work.
>  在线场景下，DFS 顺序会被打破，尽管我们的调度仍会近似保持对完整的基数树的拓展部分的 DFS 行为
>  虽然贪心的缓存感知调度可以实现高吞吐，但也会导致饥饿，因此需要和其他的公平调度方法集成

> 为什么 DFS 顺序被“打乱”:
> 在离线时，我们可以预先计算出整个基数树，并按 DFS 顺序安排访问，但在在线时，新请求随时到来，我们无法预先知道所有请求，因此，无法保证始终遵循“先深入再回溯”的 DFS 顺序
> 例如有三个请求: `a/b/c, a/b/d, x/y/z`:
> - 离线: 先处理 `a/b/c` 和 `a/b/d`，再处理 `x/y/z` (DFS)
> - 在线：假设到达顺序为 `a/b/c, x/y/z, a/b/d`， `a/b/c` 先来 → 处理，然后 `x/y/z` 来 → 处理，此时 `a/b/d` 还没有来，这就破坏了 DFS 的深度优先特性

>  augmented part of rull radix tree 指在原始请求构成的基数树基础上，人为添加了一些虚拟节点或路径，以支持更高效的调度决策
>  例如为了更好地表示“潜在的共享前缀”，系统可能会在树中插入一些“占位符”节点，使结构更完整
> 
>  假设原始请求只有:

```
/api/v1/users
/api/v1/posts
```

>  构建的基数树是:

```
api
|
vi -------------
|               |
users          posts
```

>  但如果我们想让调度器“预判”未来可能出现的 `/api/v1/comments`，就可以在树中人工添加一个分支，比如:

```
api
|
vi -------------------------
|               |           |
users          posts       comments
```

> 所以，“augmented part” = 原始树 + 为调度优化而添加的虚拟路径
>  SGLang 的调度算法虽然不能完全按照 DFS 顺序执行，但会尽量优先访问那些在增强树中处于深层、共享前缀长的路径，以尽可能复用缓存中的公共前缀
> 
>  例如当前请求是 `/api/v1/posts`，缓存中已有 `/api/v1/` ，下一个请求可能是 `/api/v1/comments` (尚未出现，但增强树中有)
>  在调度器会“预测”了这个请求的存在的情况下，它就会优先加载 `/api/v1/` 下的 `comments` 路径，从而提升命中率
>  说白了就是对未到达的 requests 进行一定程度的预测来增强 radix tree

**Distributed Cases.** RadixAttention can be extended to multiple GPUs. For tensor parallelism, each GPU maintains a sharded KV cache. There is no need for additional synchronization because the tree operations are the same. Data parallelism with multiple workers is discussed in Sec. A.4 (Appendix).
>  RadixAttention 可以被拓展到多个 GPUs
>  对于张量并行，每个 GPU 都会维护 sharded KVCache, GPU 之间不需要额外同步，因为树操作都是相同的 (树操作应该是由 CPU 统一下发)

# 4 Efficient Constrained Decoding with Compressed Finite State Machine
In LM programs, users often want to constrain the model's output to follow specific formats, such as JSON schemas. This can improve controllability and robustness, and make the output easier to parse. SGLang offers a regex argument to enforce such constraints using regular expressions, which are expressive enough for many practical scenarios. 
>  在 LM 程序中，用户通常希望约束模型的输出遵循特定格式，例如 JSON 格式
>  SGLang 提供了 `regex` 参数，通过正则表达式施加这种约束

Existing systems support this by converting a regular expression into a finite state machine (FSM) [54]. During decoding, they maintain the current FSM state, retrieve allowed tokens from the next states, and set the probability of invalid tokens to zero, decoding token by token. 
>  现存的系统通过正则表达式转化为一个有限状态机来支持这个功能
>  在解码时，它们维护当前 FSM 状态，从下一状态中获取允许的 tokens，将模型输出中无效 tokens 的概率设为零 (再进行采样)，按照这样逐 token 解码

>  有限状态机是一种数学模型，用来表示系统在不同“状态”之间的转换，它由以下几部分组成：
> - 状态集合 (如：开始、已读 `{`、已读 `"summary"`、已读 `:` 等)
> - 转移规则 (从一个状态到另一个状态的条件)
> - 初始状态和接受状态
> 
>  假设我们要匹配字符串 `{"summary": "hello"}` ，我们可以构建如下 FSM:

```
[Start] → { → "summary" → : → " → hello → "
```

>  每一步都对应一个状态，只有合法的 token 才能触发状态转移
>  FSM 可以高效判断某个 token 是否合法，例如当前在 `{"summary"` 状态，下一个只能是 `:`，其他都是非法字符，因此通过 FSM 就可以在解码过程中过滤非法字符

![[pics/SGLang-Fig4.png]]

This token-by-token approach, however, is inefficient when there are opportunities to decode multiple tokens at once. For example, the constant sequence  $\{$  summary": " in Fig. 2 spans multiple tokens in the normal decoding process as shown in Fig. 4 (c), requiring multiple decoding stages, even though there is only one valid next token when decoding it. Therefore, the whole sequence can be decoded in a single step (i.e., forward pass). However, existing systems can only decode one token at a time because the lack of integration between the FSM and the model runner in existing systems prevents multi-token processing, resulting in slow decoding.
>  当存在一次解码多个 tokens，这种逐 token 的方法较低效
>  例如 Fig2 中的常量序列 `{"summary":` 包含了多个 tokens，如 Fig4c 中，它需要多个解码阶段，即便在解码时只有一个有效的 next token
>  因此，整个序列可以在单步被解码
>  然而现存的系统一次仅解码一个 token，因为 FSM 和推理引擎没有集成，故无法实现多 token 解码

SGLang overcomes this limitation by creating a fast constrained decoding runtime with a compressed FSM. This runtime analyzes the FSM and compresses adjacent singular-transition edges in the FSM into single edges as demonstrated in Fig. 4 (b), allowing it to recognize when multiple tokens can be decoded together. In Fig. 4 (d), multiple tokens on the compressed transition edge can be decoded in one forward pass, which greatly accelerates the decoding process. It is also general and applicable to all regular expressions. More details on the background and implementation are in Appendix B.
>  SGLang 构建了基于压缩 FSM 的快速约束解码运行时
>  运行时会分析 FSM，并将其中的单次转移边压缩为单个边，如 Fig4b 所示
>  在压缩的转移边上的多个 tokens 可以通过一次前向传播解码

>  singular-transition edge 指 FSM 从一个状态到另一个状态仅经过一个 token 的边

# 5 Efficient Endpoint Calling with API Speculative Execution
The previous sections introduced optimizations for open-weight models, which require modifications to the model inference process. Additionally, SGLang works with API-access-only models, such as OpenAI's GPT-4. However, for these models, we can only call a black-box API endpoint.
>  之前的部分介绍的都是对开放权重模型的优化，这需要修改模型的推理过程
>  SGLang 也适用于 API-only 模型

This section introduces a new optimization for black-box API models that accelerates execution and reduces the API cost of multi-call SGLang programs using speculative execution. For example, a program may ask the model to generate a description of a character with a multi-call pattern: s += context + "name: " + gen("name", stop="\n") + "job: " + gen("job", stop="\n"). Naively, the two gen primitives correspond to two API calls, meaning that the user needs to pay for the input token fee on the context twice. 
>  SGlang 使用投机执行来加速多次调用 SGLang 程序在 API-only 上的执行和 API 调用开销
>  例如，一个程序可能使用多调用模式请求模型生成对角色的描述: `s += context + "name: " + gen("name", stop="\n") + "job: " + gen("job", stop="\n")`
>  如果按朴素方式处理，这两个 `gen` 原语会对应两次 API 调用，意味着用户需要为 `context` 付两次输入 token 费用

In SGLang, we can enable speculative execution on the first call and let it continue the generation of a few more tokens by ignoring the stop condition. The interpreter keeps the additional generation outputs and matches and reuses them with later primitives. 
>  SGLang 中，我们可以在第一次调用时启用投机执行，允许模型忽略停止条件，继续生成一些 tokens
>  解释器会保留这些额外的生成内容，并将其和后续的原语调用进行匹配和复用

In certain cases, with careful prompt engineering, the model can correctly match the template with high accuracy, saving us the latency and input costs of one API call.
>  在某些情况下，通过 prompt engineering，模型可以以高准确率匹配模板，节省一次 API 调用的延迟和输入开销

>  总感觉有点抽象

# 6 Evaluation
We evaluate the performance of SGLang across diverse LLM workloads. Subsequently, we conduct ablation studies and case studies to demonstrate the effectiveness of specific components. SGLang is implemented in PyTorch [37] with custom CUDA kernels from FlashInfer [59] and Triton [48].
>  SGLang 基于 PyTorch 实现，并使用了基于 FlashInfer 和 Trition 的自定义 CUDA kernels

## 6.1 Setup
**Models.** We test dense Llama-2 models [49], sparse mixture of experts Mixtral models [17], multimodal LLaVA image [27] and video models [62], and API model OpenAI's GPT-3.5. For open-weight models, the number of parameters ranges from 7 billion to 70 billion, and we use float16 precision.

**Hardware.** We run most experiments on AWS EC2 G5 instances, which are equipped with NVIDIA A10G GPUs (24GB). We run 7B models on a single A10G GPU and larger models on multiple A10G GPUs with tensor parallelism [44]. We run some additional experiments on A100G (80GB) GPUs.

**Baselines.** We compare SGLang against both high-level programming systems with their respective languages and default runtimes, as well as low-level inference engines with standard OpenAI-like Completion APIs. Unless otherwise stated, we do not turn on optimizations that will change the computation results so that all systems compute the same results. The baselines include:
>  我们将 SGLang 和各类高级编程系统 (及其各自的语言和默认运行时) 进行比较，也和低级推理引擎 (具有标准 OpenAI 风格的 Completion API) 进行比较

- Guidance[13], a language for controlling LLMs. We use Guidance v0.1.8 with llama.cpp backend.
- vLLM [23], a high-throughput inference engine. We use vLLM v0.2.5 and its default API server.
- LMQL [4], a query language. We use LMQL v0.7.3 with Hugging Face Transformers backend.

**Workloads.** We test the following: 5-shot MMLU [14] and 20-shot HellaSwag [61] benchmarks. We decode one token for MMLU and use primitive select to select the answer with the highest probability for HellaSwag. For the ReAct agent [57] and generative agents [36], we extract the traces from the original papers and replay them. We use the Tree-of-thought [56] for the GSM-8K problems and Skeleton-of-thought [33] for tip generation. We use LLM judges with the branch-solve-merge [40] technique; JSON decoding with a scheme specified by a regular expression. Multi-turn chat with 4 turns, where the input of each turn is randomly sampled between 256-512 tokens. Multi-turn chat (short) means short output (4-8 tokens) and multi-turn chat (long) means long output (256-512 tokens); DSPy retrieval-augmented generation (RAG) pipeline [20] in its official example.
>  我们测试 5-shot MMLU, 20-shot HellaSwag
>  我们为 MMLU 解码一个 token，对于 HellaSwag，我们使用 `select` 选择具有最高概率的答案
>  对于 ReAct agent 和生成式 agents，我们从原始论文提取轨迹并重放它们
>  对于 GSM-8K，我们使用 Tree-of-thought 方法，对于提示生成，我们使用 Skeleton-of-thought
>  ...

**Metrics.** We report two performance metrics: throughput and latency. For throughput, we run a sufficiently large batch of program instances to compute the maximum throughput, comparing the number of program instances executed per second (programs per second, p/s). For latency, we execute a single program at a time without batching and report the average latency for multiple instances.
>  我们报告两个性能指标: 吞吐和延迟
>  吞吐为最大的每秒执行的程序实例数量
>  延迟为多个实例执行的平均延迟

## 6.2 End-to-End Performance

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/79dce60a-bf72-47ad-80c9-b5bbd0455d68/5922545d7108a824e68ca6ad38d01d96e950a5299610769e780a7bbe5c22475c.jpg)  

Figure 5: Normalized throughput of Llama-7B models. Higher is better.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/79dce60a-bf72-47ad-80c9-b5bbd0455d68/c9f7436447ce6c7c83139ec63f585e1d323c86542394fb50847d3773c4c8c627.jpg)  

Figure 6: Normalized latency on Llama-7B models. Lower is better.

**Results on open-weight models.** The latency and throughput results are shown in Fig. 5 and Fig. 6. SGLang improves throughput by up to  $6.4\times$  and reduces latency by up to  $3.7\times$ . These improvements result from KV cache reuse, the exploitation of parallelism within a single program, and faster constrained decoding. Next, we explain the reasons for the speedup in each benchmark.
>  SGLang 的吞吐提升最高大 6.4x，延迟降低最高达 3.7x
>  这主要得益于 KVCache 复用以及单个程序内部并行性的充分利用，以及更快的约束解码机制

On MMLU, SGLang can reuse the KV cache of the 5-shot examples with RadixAttention. RadixAttention benefits both throughput and latency. RadixAttention reduces total memory usage by sharing the KV cache, allowing for a larger batch size to improve maximum throughput. RadixAttention also reduces the computation of prefill, thus decreasing the first token latency. On HellaSwag, SGLang reuses the KV cache of both few-shot examples and the common question prefix for multiple choices, resulting in two-level sharing. For the ReAct and generative agents, SGLang reuses the KV cache of the agent template and previous calls. On Tree-of-thought and Skeleton-of-thought, SGLang parallelizes the generation calls within a single program and reuses the KV cache as much as possible. On JSON decoding, SGLang accelerates decoding by decoding multiple tokens at once with a compressed finite state machine. In multi-turn chat, SGLang reuses the KV cache of the chat history. The speedup is more noticeable for short outputs because KV cache reuse mostly helps reduce the prefix time. For long outputs, because there is not much sharing between different chat sessions and the decoding time dominates, there is almost no speedup. In the DSPy RAG pipeline, SGLang reuses the KV cache of the common context example. On these benchmarks, the cache hit rate ranges from  $50\%$  to  $99\%$ . 
>  MMLU 中，SGLang 可以通过 RadixAttention 复用 5-shot 示例的 KVCache，这同时提高了吞吐并降低了延迟: RadixAttention 通过复用 KVCache 减少了总体内存使用，从而支持更大的批处理规模，提高了吞吐，RadixAttention 也降低了 prefill 阶段的计算量，进而降低了第一个 token 的延迟
>  HellaSwag 中，SGLang 复用了 few-shot examples 和多项选择共有的问题前缀，实现了两级缓存复用
>  ReAct 和生成式代理中，SGLang 复用了 agent template 和之前调用的 KVCache
>  Tree-of-thought, Skeleton-of-thought 中，SGLang 并行化了单个程序中的 generation 调用，并尽可能复用了 KVCache
>  多轮对话中，SGLang 复用了对话历史的 KVCache，对于短输出任务，加速更加明显，因为 KVCache 复用主要减少的是前缀计算时间，对于长输出，因为不同对话会话之间的共享较少，且解码时间占据主导地位，加速可以忽略不计
>  在 DSPy RAG 流水线中，SGLang 复用了公共上下文示例的 KVCache
>  在上述各种基准测试中，缓存命中率介于 50%, 99% 之间

Fig. 13 (Appendix) lists the achieved and optimal cache hit rates for all of them, showing that our cache-aware scheduling approaches  $96\%$  of the optimal hit rate on average.

We exclude LMQL and Guidance from some of the last five benchmarks due to slow performance and missing functionalities. LMQL's issues stem from slow token-level processing and an unoptimized backend, while Guidance lacks batching and parallelism support.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/79dce60a-bf72-47ad-80c9-b5bbd0455d68/b4d75e022eeb16cbc4f25aed2d35ad17402a36c0647546f09a9096ce88a270be.jpg)  

Figure 7: Normalized throughput on Mixtral-8x7B models with tensor parallelism. Higher is better.

**Results on larger models with tensor parallelism.** We run larger models, Mixtral-8x7B and Llama-70B, with tensor parallelism on the same set of benchmarks and report the results in Fig. 7 and Fig. 12 (Appendix). The speedup on larger models shows a trend similar to that observed on smaller models, indicating that our optimization generalizes well to larger models. We omit Guidance and LMQL here because they lack efficient implementations of tensor parallelism.
>  对于更大的模型，加速趋势和小模型上的趋势是类似的

**Results on multi-modal models.** SGLang has native support for multi-modal models with the image and video primitives. The optimizations in this paper are compatible with multi-modal models. For RadixAttention, we compute the hash of the input images and use it as the key in the radix tree, allowing us to reuse the KV cache of the image tokens from the same image. We run LLaVA-v1.5-7B (image) on llava-bench-in-the-wild and LLaVA-NeXT-34B (video) on ActivityNet. Because these models are not well supported by other baseline systems, we use the model authors' original implementation in Hugging Face Transformers as the baseline. As shown in Table 2, SGLang provides throughput up to  $6\times$  higher on these benchmarks. In llava-bench-in-the-wild, there are multiple questions about the same image, and SGLang runtime reuses the KV cache in this case.
>  SGLang 提供了 `image, video` 原语，原生支持多模态模型，文中的优化也适用于多模态模型
>  对于 RadixAttention，我们为输入图像计算哈希，将哈希值作为 radix tree 中的 key，实现对相同图像的 token 缓存的复用
>  对于针对一张图像的多个问题，SGLang 运行时可以复用 KVCache 实现加速

**Production deployment.** SGLang has been deployed in Chatbot Arena [8] to serve open-weight models. Due to low traffic for some models, only one SGLang worker serves each. After one month, we observed a  $52.4\%$  RadixAttention cache hit rate for LLaVA-Next-34B [28] and  $74.1\%$  for Vicuna-33B [7]. Cache hits come from common system messages, frequently reused example images, and multi-turn chat histories. This reduces first-token latency by an average of  $1.7\times$  for Vicuna-33B.
>  在生产部署上，我们发现 LLaVA-Next-34B 的缓存命中率为 52.4%，Vicuna-33B 的缓存命中率为 74.1%
>  缓存命中主要来自于常见的系统消息、频繁复用的示例图像和多轮对话历史

**Results on API models.** We test a prompt that extracts three fields from a Wikipedia page using OpenAI's GPT-3.5 model. By using few-shot prompting, the accuracy of API speculative execution is high, and it reduces the cost of input tokens by about threefold due to the extraction of three fields.

## 6.3 Ablation Study

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/79dce60a-bf72-47ad-80c9-b5bbd0455d68/dba065a052706b78024787b08fbe298c0b336e0c2dfcc81297ac00005b95c30e.jpg)  

Figure 8: (a)(b) Cache hit rate ablation study. (c) RadixAttention ablation study.

**Cache hit rate vs. latency/throughput.** Fig. 8(a)(b) shows the relationship between cache hit rate and performance metrics (first token latency, total latency, batch size, and throughput) on the tree-of-thought benchmark. The figure is obtained by partially disabling matched tokens at runtime. It shows that a higher cache hit rate leads to a larger batch size, higher throughput, and lower latency.
>  Fig8 a, b 展示了缓存命中率和性能指标 (首个 token 延迟、总延迟、批量大小、吞吐) 的关系
>  可以看到更高的缓存命中率能够支持更大的批处理规模、带来更高的吞吐量和更低的延迟

**Effectiveness of RadixAttention.** We test the effectiveness of RadixAttention and its components on several representative benchmarks. As shown in Fig. 8(c), "No Cache" means not using any cache, "No Tree-Structure" means using a simple table-based cache instead of a tree-structured cache, "FCFS Schedule" means using a first-come-first-serve policy instead of our cache-aware scheduling, "Random Schedule" means using a random order to schedule requests, "No Frontend Parallelism" means disabling parallelism in the interpreter, "No Frontend Hint" means disabling sending the fork hints from the interpreters, and "Full optimizations" means we turn on all optimizations. 
>  我们测试 RadixAttention 和它的各个组件的有效性
>  No Cache 表示不使用任何缓存
>  No Tree-Structure 表示使用简单的基于表格的缓存而不是树状的缓存
>  FCFS Schedule 表示使用先来先服务策略而不是缓存感知的调度
>  Random Schedule 表示使用随机顺序调度请求
>  No Frontend Parallelism 表示禁用解释器的并行机制
>  No Frontend Hint 表示禁止解释器向运行时发送 `fork` 提示
>  Full Optimizations 表示开启所有优化

The experimental results show that each of these components is required to achieve the best performance. Disabling parallelism and hints from the frontend interpreter also results in suboptimal runtime performance, highlighting the importance of co-designing the frontend language and runtime.

**Overhead of RadixAttention.** We test the overhead of RadixAttention on a benchmark without any KV cache reuse opportunities. The benchmark measures throughput on the ShareGPT dataset. It takes 74.3 seconds to run 100 requests; however, the time used for managing the RadixAttention data structures is only 0.2 seconds, which is a negligible overhead of less than  $0.3\%$ . This is because the complexity of tree operations is linear and small. Thus, we can turn on RadixAttention by default.
>  我们在没有 KVCache 复用机会的 benchmark 中测试 RadixAttention 的开销，其中管理 RadixAttention 数据结构的时间仅需要 0.2 秒，这个开销几乎可以忽略不计
>  这是因为树操作的复杂性是线性的，因此我们可以默认开启 RadixAttention

**Effectiveness of the compressed finite state machine.** We test the effectiveness of the compressed finite state machine and its components on the JSON decoding benchmark. Experimental results show that the compressed finite state machine increases the throughput by  $1.6\times$  because it can decode multiple tokens at once. In addition, we need to preprocess the state machine and reuse it for a batch of requests. Otherwise, redoing the preprocessing for each request makes the throughput  $2.4\times$  lower.
>  我们在 JSON 解码 benchmark 上测试压缩 FSM 和其组件的有效性
>  这提高了 1.6x 的吞吐，因为它可以一次解码多个 tokens，此外，我们需要为状态机进行预处理并复用于一批请求，否则对每个请求预处理状态机会导致吞吐下降

# 7 Related Work
Various works have explored the reuse of the KV cache, and many of them are concurrent with our work. Uniquely, our RadixAttention first proposes treating the KV cache as a tree-based LRU cache. It is the first solution that supports multi-level sharing, cache-aware scheduling, frontend-runtime co-scheduling, and distributed cases. vLLM [23] and ChunkedAttention [58] explore some simple reuse cases (e.g., system prompt sharing) but do not cover multi-level tree-structured sharing or LRU caching. 
>  许多工作都探究了对 KVCache 的复用，我们的 RadixAttention 首次提出将 KVCache 视为基于树结构的 LRU 缓存
>  它是第一个支持多级共享、缓存感知调度、前端-运行时协同设计以及分布式场景的解决方案
>  vLLM, ChunkAttention 探索了简单的复用场景，例如 system prompt 共享，但没有覆盖到多级树状结构或 LRU 缓存机制

PromptCache [12] proposes the modular reuse of the KV cache beyond the prefix but can impact accuracy by up to a  $43\%$  drop. HydraGen [18], FlashInfer [59], and ChunkedAttention [58] focus on CUDA kernel optimizations and do not include the concept of an LRU cache. API Serve [1] and LLM-SQL [29] study KV cache reuse for specific applications such as interleaving with external API calls and relational databases, but they do not have our radix tree or cache-aware scheduling.

Several LLM programming and agent frameworks exist, such as Guidance [13], LMQL [4], DSPy [20], LangChain [24], AutoGen [55], and LLM Compiler [21]. Guidance and LMQL are most similar to SGLang, and we compare them in Sec. 2. Our innovation lies in novel runtime optimizations for accelerating the proposed programming model. SGLang is compatible with other frameworks and can accelerate them (e.g., the DSPy example in our evaluation). Additionally, SGLang is compatible with many other common inference optimizations: [60, 39, 3, 23, 59, 10, 26, 15, 19, 32, 31, 11].

# 8 Future Directions and Conclusion
**Future directions.** Despite the progress made with SGLang, several limitations remain that reveal promising directions for future research. These include extending SGLang to support additional output modalities, adapting RadixAttention to operate across multiple levels of the memory hierarchy (e.g., DRAM, Disk) [43], enabling fuzzy semantic matching within RadixAttention, providing higher-level primitives atop SGLang, fixing starvation in cache-aware scheduling [42], and enhancing the SGLang compiler to perform advanced static optimizations such as scheduling and memory planning.
>  拓展 SGLang 以支持额外的输出模态
>  将 RadixAttention 适配至内存层次的多个层级 (DRAM, Disk)
>  在 RadixAttention 中引入模糊语义匹配的能力
>  在 SGLang 上提供更高级的原语
>  解决缓存感知调度中调度饥饿问题
>  增强 SGLang 编译器，实现高级静态优化例如调度和内存规划

**Conclusion.** We introduce SGLang, a framework for efficient programming and executing structured language model programs. SGLang significantly improves the throughput and latency of complex LM programs through novel optimizations like RadixAttention, compressed finite state machines, and a language interpreter. It is a valuable tool for developing advanced prompting techniques and agent workflows. The source code is publicly available.
>  SGlang 是一个高效编程和执行结构化语言模型程序的框架
>  SGLang 通过 RadixAttention, 压缩 FSM, 语言解释器等优化提高了复杂 LM 程序的吞吐，降低了延迟

# A Additional Details on RadixAttention
## A.1 Background on the KV Cache
Most LLMs in use today, such as GPT-3 [5], PaLM [9], and LLaMA [49], are based on the autoregressive Transformer architecture [51]. These models predict the probability of the next token in a sequence based on the preceding tokens. During inference, the model first processes a sequence of input tokens through a forward pass (this process is called "prefill"). It then sequentially decodes output tokens, with each token depending on prior tokens (this process is called "decoding"). 
>  自回归模型基于前面的 tokens 预测下一个 token
>  在推理时，模型首先通过一次前向传播处理输入 tokens 序列 (prefill)，然后顺序地解码输出 tokens，解码过程中每个输出也依赖于之前的输出 (decoding)

We refer to the process of taking a sequence of input tokens and generating a sequence of output tokens as a single-generation call. Throughout this process, each token generates some intermediate tensors, which are used for decoding further tokens. These intermediate tensors, known as the "KV Cache," are named for the key-value pairs in the self-attention mechanism. 
>  我们将接收输入 tokens 序列，生产输出 tokens 序列的过程称为一次 generation 调用
>  这个过程中，每个 token 的中间张量会被复用，即 KVCache

An important observation when discussing optimizations in this paper is that the computation of the KV cache only depends on all previous tokens, so different sequences with the same prefix can reuse the KV cache of the prefix tokens and avoid redundant computation.
>  KVCache 的计算只依赖于所有之前的 tokens，因此具有相同前缀的不同序列可以复用相同前缀的 KVCache，减少冗余计算

![[pics/SGLang-Fig9.png]]

Often in LLM programs, multiple text segments and generation calls are appended to a single prompt. Caching the computed KV cache for previous tokens across multiple chained calls can reduce redundant computation. This optimization, however, is neither free nor trivial, as it requires additional storage and more complex memory management. In addition, it is common in LLM programs to generate multiple outputs from a single prompt or to fork a new prompt from the current state [25]. 
>  在 LLM 程序中，常常会将多个文本片段和生成调用附加到同一个 prompt 之后，缓存多个链式调用之前计算的 KVCache 可以减少冗余计算
>  这一优化要求额外的存储和复杂的内存管理
>  此外，LLM 程序中从单个 prompt 生成多个输出，或从当前状态 fork 出新 prompt 的状况非常常见 (就网页对话来看，感觉这个场景基本是没有的，可能还是在 Dify 这种 LM program 中出现得比较多)

Basic prefix sharing has been investigated in vLLM [23]. More advanced sharing patterns like irregular tree-structured sharing can also be employed. Fig. 9 shows four typical patterns of KV cache sharing across multiple calls; none of the existing systems can automatically handle all of them. On the contrary, our RadixAttention in Sec. 3 can handle all of them automatically at runtime.
>  vLLM 已经探索了基本的前缀共享，Fig9 展示了 4 个跨多个调用的典型 KVCache 共享模式，RadixAttention 可以在运行时自动处理它们所有的情况

## A.2 Pseudocode for Cache-Aware Scheduling

![[pics/SGLang-Alg1.png]]

Alg. 1 shows the pseudocode of cache-aware scheduling for RadixAttention with continuous batching.
>  RadixAttention + continuous batching 的缓存感知调度算法如 Alg1

>  核心思想是，在内存有限的情况下，优先调度等待队列中和当前 radix tree 的匹配前缀最长的请求，这样可以实现最大的缓存命中率
>  可以考虑如果不优先调度匹配前缀最长的请求，而是随意调度了一个没有匹配前缀的请求，在有限的内存下，这个请求带来的新前缀可能会导致目前 radix tree 中存储的一些前缀被驱逐，使得等待队列中具有这部分被驱逐前缀的请求再后续被调度的时候缓存未命中，又需要重新计算这些前缀的 KVCache

## A.3 Proof of the Theorem 3.1
**Theorem 3.1.** For a batch of requests, we can achieve an optimal cache hit rate by visiting the radix tree of the requests in the depth-first search order with a cache size  $\geq$  the maximum request length. The longest-shared-prefix-first order is equivalent to a depth-first search order.
>  定理
>  对于一批请求，我们可以通过使用 DFS 顺序访问 radix tree 来达到最优的缓存命中率，条件是缓存大小不小于最大请求长度
>  最长共享前缀优先顺序等价于深度优先顺序

**Proof.** First, we show that the depth-first search (DFS) order achieves an optimal cache hit rate. Let  $R$  denote the set of requests in the batch, and  $T$  denote the radix tree built from  $R$ . For each edge  $e$  of  $T$ , the KV cache associated with  $e$  needs to be computed at least once. Let  $|e|$  denote the size of the KV cache associated with  $e$ . Let  $C$  denote the computational complexity of the KV cache for  $R$ .
>  我们首先证明 DFS 顺序可以达到最优的缓存命中率
>  令 $R$ 表示批量里的一组请求, $T$ 表示从 $R$ 构建的 radix tree
>  **对于 $T$ 中的每个边 $e$，和 $e$ 相关的 KVCache 至少需要被计算一次**
>  令 $|e|$ 表示和 $e$ 相关的 KVCache 的大小，令 $C$ 表示为 radix tree $R$ 计算 KVCache 的计算复杂度

We obtain the lower bound

$$
C\geq \sum_{e\in \mathrm{edges}(T)}|e|.
$$

>  $C$ 的下界如上所示，也就是无论如何至少要为每条边计算它相关的 KVCache

Consider we visit the radix tree  $T$  in DFS order. For each edge  $e$  of  $T$ , the first time we compute the KV cache associated with  $e$ , we will then compute the whole subtree of  $e$ . During the computation of the subtree of  $e$ , the edge  $e$  will be continuously hit, so no additional computation will happen. 
>  考虑我们以深度优先顺序访问 radix tree $T$: 
>  对于 $T$ 中的每条边 $e$，我们第一次计算和 $e$ 相关的 KVCache ，之后会计算 $e$ 的整个子树的 KVCache，在计算它的子树的 KVCache 时，$e$ 的缓存会被反复命中，因此 $e$ 上没有额外的计算

After finishing the computation for the subtree rooted at  $e$ , the edge  $e$  will not be visited again. Notice that, with a cache size  $\geq$  the maximum request length, which equals the longest path in the radix tree  $T$ , edge  $e$  will not be evicted during the computation of its subtree, since the common prefix including  $e$  of the subtree will be continuously hit. 
>  在完成了以 $e$ 为根的子树的计算后，$e$ 就不会再被访问
>  注意，在缓存大小大于最大请求长度的情况下 (最大请求长度等于 radix tree $T$ 中最长的路径)，边 $e$ 不会在它的子树计算时被驱逐，因为子树中的计算会反复命中 $e$ (而如果缓存太小，连一条完整的子树路径都无法存下，那么它就有可能被驱逐，然后被计算存下，然后又被驱逐，出现抖动)

Therefore, the KV cache associated with each edge  $e$  will be computed only once. Thus, we achieve the lower bound
>  因此，和边 $e$ 相关的 KVCache 将仅被计算一次，我们的计算复杂度达到了下界

$$
C = \sum_{e\in \mathrm{edges}(T)}|e|.
$$

The cache hit rate, defined as

$$
\frac{\sum_{r\in R}\mathrm{number~of~cached~prefill~tokens~in~}r}{\sum_{r\in R}\mathrm{number~of~prefill~tokens~in~}r},
$$

equals  $1 -\frac{C}{\sum_{r\in R}\text{number of prefll tokens}}$  reaches its upper bound, delivering optimality.

>  缓存命中率定义为所有请求缓存的 prefill tokens 总数量和所有请求的 prefill tokens 总数量，可以进一步写为

$$
\begin{align}
&\frac{\sum_{r\in R}\mathrm{number~of~cached~prefill~tokens~in~}r}{\sum_{r\in R}\mathrm{number~of~prefill~tokens~in~}r}\\
=& \frac {\sum_{r\in R}(\text{number of prefill tokens in }r - \text{number of non-cached prefill tokens in }r)}{\sum_{r\in R}\text{number of prefill tokens in }r}\\
=& 1 - \frac {\sum_{r\in R}\text{number of non-cached prefill tokens in }r}{\sum_{r\in R}\text{number of prefill tokens in }r}
\end{align}
$$

>  要让缓存命中率最大，就要让减数最小，也就是让每个请求 $r$ 的未命中 tokens 数量最小，对于给定的 radix tree，每个请求 $r$ 的未命中 tokens 数量最大为它的所有 tokens 树，最小为它对应的边上的 tokens 数 (能复用的前缀缓存都已经全部复用，剩下的就是这个请求独有的 tokens)
>  因此在 DFS 下，分子将等于 $C$，缓存命中率就达到了上界最优

Next, we show that the longest-shared-prefix-first order is equivalent to a DFS order by induction.

- (Base) In the beginning, since there is nothing cached, a random request that corresponds to a node  $x$  in  $T$  will be processed. All requests that correspond to the nodes  $\{v_{1},\ldots ,v_{n}\}$  on the path from the root to  $x$  do not need a recomputation. The computation complexity for requests corresponding to the nodes  $\{v_{1},\ldots ,v_{n},x\}$  is aligned with a valid DFS. The path from the root to  $x$  is cached.
- (Induction) Assume we just visited a node  $y$  in  $T$ , and the visited nodes align with a DFS order. Let  $P$  denote the path from the root to  $y$ . Then each node that has not been visited has the lowest common ancestor with the visited nodes on  $P$ . Since nodes on  $P$  are cached, a node  $z$  that has not been visited with the lowest common ancestor on  $P$  will have the longest shared prefix. The longest-shared-prefix-first order will select  $z$ , which is a valid DFS order. The path from the root to  $z$  will be cached because it is the most recent.

>  我们通过归纳法证明最长共享前缀优先顺序等价于 DFS
>  归纳基础: 在开始，没有缓存，会处理一个对应于 $T$ 中节点 $x$ 的随机请求，那么对应于根节点到节点 $x$ 路径上的节点 $\{v_1, \dots, v_n\}$ 的缓存都会被计算，之后对应于节点 $\{v_1, \dots, v_n\}$ 的请求都不需要重计算，在计算该请求时，从根开始访问节点 $\{v_1, \dots, v_n, x\}$ 的顺序符合 DFS 的行为
>  归纳步骤: 假设我们在 $T$ 访问了一些节点，并且访问的节点顺序符合 DFS 的规则，并且最后访问的节点是 $y$；令 $P$ 表示从根到 $y$ 的路径，可以确定剩余的未访问节点的最低公共祖先也在 $P$ 上 (至少是根)；因为 $P$ 上的节点都已经缓存，那么如果一个未访问节点 $z$ 在 $P$ 上具有最低公共祖先，它就具有最长共享前缀，那么最长共享前缀顺序就会选择 $z$，因此下一个要访问节点的选择依据就有具有最低公共祖先的节点，这符合 DFS 的顺序

>  也就是在最长共享前缀优先顺序下，下一个访问节点的选择依据就是具有最低公共祖先的节点，这个选择和深度优先顺序的选择是一样的

In the online case, the DFS order will be disrupted, but the longest shared prefix schedule still approximates the DFS behavior on the augmented part of the implied radix tree. We show this by considering the step of adding a new batch of requests.
>  在在线场景下，DFS 顺序会被打乱，但最长共享前缀调度仍然会近似在隐含的 radix tree 的拓展部分的 DFS 行为
>  我们通过分析添加新的一批请求下的行为来证明这一点

Let  $T$  denote the part of the radix tree that has been visited so far, and  $T^{\prime}$  denote the whole new radix tree after adding the new batch of requests. Let  $C$  denote the set of cached nodes in  $T$  .Let longest  $(C)$  denote the node in  $C$  that has the longest path from the root and has a subtree in  $T^{\prime}$  that has not been fully visited.
>  令 $T$ 表示 radix tree 中已经被访问过的部分，$T'$ 表示在添加了新一批请求后的 radix tree
>  令 $C$ 表示 $T$ 中缓存的节点，令 $\text{longest}(C)$ 表示 $C$ 中从根出发路径最长，且其在 $T'$ 中还有子树尚未被完全访问的节点

>  路径最长即深度最深，子树尚未被访问说明它还有未处理的后代节点，比如它的某个子节点还没被请求过
>  那么从根到 $\text{longest}(C)$ 这段前缀就是当前 $T$ 面对新一批请求中，能够找到的最长前缀

The longest shared prefix schedule will then process the subtree in  $T^{\prime}$  rooted at longest  $(C)$  in a DFS order. During this process, eviction could occur, and the remaining cached nodes from  $C$  become  $C^{(1)}\subseteq C$  . A DFS will then occur for the subtree in  $T^{\prime}$  rooted at longest  $(C^{(1)})$
>  最长共享前缀调度会以 DFS 顺序处理根为 $\text{longest}(C)$ 的子树 $T'$
>  在这个过程中，可能会发生淘汰，我们记原本在 $C$ 中剩余的节点为 $C^{(1)}\subseteq C$
>  处理完了这个子树，下一次的处理对象就是对根为 $\text{longest}(C^{(1)})$ 的在 $T'$ 的子树

Similarly, we will have  $C^{(2)},\ldots ,C^{(k)}$  , until  $C^{(k)}$  contains only one leaf node in  $C^{(k)}$  , that has its subtree in  $T^{\prime}$  not fully visited. At this point, we have reached a valid DFS state. The remaining part of  $T^{\prime}$  will be visited in DFS order as described in the proof of Theorem 3.1.
>  类似地，我们将具有 $C^{(2)}, \dots, C^{(k)}$，直到 $C^{(k)}$ 仅包含一个叶子节点，该节点在 $T'$ 中的子树尚未被完全访问
>  之后 $C$ 中的节点就被完全淘汰，剩余的访问部分就是 (完全按照 $T'$ 中的缓存状态进行的) DFS

## A.4 Data-Parallel Distributed RadixAttention
To adapt the RadixAttention for distributed settings with multiple replica workers (i.e., data parallelism), we developed a mechanism wherein each worker maintains its own sub-tree, while the router oversees a meta-tree. This meta-tree acts as a trie that tracks all sub-trees and their associated devices. Upon the arrival of a new batch of requests at the router, prefix matching is executed on the meta-tree. 
>  为了将 RadixAttention 适配到具有多个部分工作节点 (即数据并行) 的分布式环境中，我们设计了一种机制，其中每个 worker 维护自己的 sub-tree，而 router 则管理一个 meta-tree
>  meta-tree 作为一个前缀树，追踪所有 sub-trees 和它们相关的设备，当新的一批请求到达 router，SGLang 会在 meta-tree 上执行前缀匹配

>  worker 的 sub-tree 即自己所处理的请求所构成的 radix tree，存储了该 worker 对自己负责的请求管理的 KVCache
>  meta-tree 不存储 KVCache，而是存储 sub-trees 的信息
>  meta-tree 本质也是一个前缀树，它记录某个前缀存储在哪个 worker 的 sub-tree 中
>  新的请求到来后，这些请求的 prompt 会在 meta-tree 上进行前缀匹配，以定位存储了最长前缀的 worker

We implement various policies based on each request's affinity—measured by the length of the shared prefix with specific workers and other requests in the same group—to make efficient dispatch decisions that minimize redundant computations. 
>  我们基于每个请求的亲和性实现多种策略，从而实现高效的分发策略，以最小化冗余计算
>  请求的亲和性通过与特定 workers 和其他同组请求的共享前缀长度来衡量

>  亲和性指一个请求与某个 worker 或其他请求之间的相似度，度量方式就是共享前缀长度
>  系统可以将高亲和性的请求发给同一个 worker，以及将请求优先发送给亲和性高的 worker，实现最优调度

Each time new requests are processed, both the router and workers independently update their respective trees. Should an eviction occur at a worker node, it commits this eviction to a queue, which the router then processes to update the meta-tree during periods of low activity. 
>  每当处理新的请求时，router 和 workers 都会独立更新它们各自的树
>  若某个 worker 节点上发生了淘汰，它将这个淘汰操作提交到一个队列中，router 会在低负载时处理该队列，来更新 meta-tree (更新该 worker 的 sub-tree 的信息，也就是该 worker 上的 KVCache 情况)

We benchmarked this distributed configuration using four workers and the MMLU dataset, observing that it achieves linear scaling and an optimal cache hit rate with minimal overhead from this weakly consistent distributed cache design. 
>  我们使用四个 workers 和 MMLU 数据集对这个分布式配置进行了基准测试，结果表明该方案实现了线性拓展，并在最小开销的情况下达到了最优缓存命中率

There exists a trade-off between maximizing data locality and parallel processing efficiency. Exploring advanced scheduling policies to optimize this trade-off is designated as an area for future research. In addition, concurrent work from Preble [45] studies data-parallel scheduling based on an early version of SGLang.
>  这背后存在一个最大化数据局部性和并行处理效率的权衡，探索优化这个权衡的更优调度策略是未来的研究方向
>  此外，Preble 对数据并行调度的相关工作也基于 SGLang 的早期版本

# B Additional Details on Compressed Finite State Machine
This section discusses the background and implementation details of the compressed finite state machine for faster constrained decoding. We aim for LLMs to follow a regular expression (regex), which offers greater expressiveness and can be used to represent common formats such as JSON schemas. To achieve this, we convert the regex into a Finite State Machine (FSM) to guide the generation process during decoding [54]. 
>  在约束解码中，我们将 regex 转化为 FSM，以引导解码时的生成过程

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/79dce60a-bf72-47ad-80c9-b5bbd0455d68/6308e936c48835c02756c5bd96991196e9ec0f06822f449965673b2440b37162.jpg)  

Figure 10: Example of how regex is converted into FSM and how FSM guides the decoding process.

An FSM is essentially a graph with nodes (states) and edges (transitions with strings/characters). Starting from an initial state, each transition appends the string on the edge to move to the next state, using a set of final states to conclude the process. This mechanism guides an LLM's decoding, filtering out invalid tokens based on the FSM's current state transitions, as illustrated in Fig. 10. The decoding might involve multiple transitions in the FSM until reaching a final state.
>  FSM 本质上是一个图结构，节点表示状态，边表示状态之间的转移 (带有字符或字符串的转移)
>  从初始状态开始，每个转移都将字符/字符串追加到当前输出，然后移动到下一个状态，图中的终止状态终止整个过程
>  这个机制引导了 LLM 的解码，基于 FSM 的当前状态转移过滤无效的 tokens
>  解码过程中，模型可能会在 FSM 中进行多次状态转移，直到到达最终状态

The challenge of constrained decoding arises from the fact that constraints are often expressed in natural language formats, i.e., regex is depicted by characters/strings, while LLMs are designed to interpret and process these as tokens. This creates a complex scenario since the mapping between strings and tokens is intricate and lacks a one-to-one correspondence [22].
>  约束解码的挑战在于: 约束通常以自然语言形式表示，例如 regex 通常由字符/字符串表示
>  而 LLM 在设计上是将自然语言内容解析为 tokens，并处理 tokens
>  这导致了一个复杂的情况，因为字符串和 tokens 之间的映射非常复杂，且不存在一一对应的关系 (例如多个 tokens 组合为一个字符)

This section is derived from an earlier blog post. Readers are also encouraged to read the blog post for additional background and easier understanding.

## B.1 Implementation Details of Compressed Finite State Machine
To simplify the construction of Compressed FSM, we build the original FSM on characters/strings instead of on tokens. 
>  为了简化压缩 FSM 的构造，我们在字符/字符串上构造原始 FSM，而不是在 tokens 上

We formally define the concepts of singular transition edge and compressed edge as follows:

- Singular transition edge: A edge is a singular transition edge if 1) its source node has only one successor node, and 2) there is only one acceptable character/string on it.
- Compressed edge: An edge compressed by several consecutively adjacent edges  $(e_0, e_1, \ldots , e_k)$  if and only if  $e_1, \ldots , e_k$  are singular transition edges. The text of the compressed edge is the concatenation of the texts of  $e_0, e_1, \ldots , e_k$ .

>  我们正式定义以下两个概念:
>  - 单一转移边: 如果一条转移边满足 1. 其源节点只有一个后继结点 2. 该边上仅存在一个可接收的字符/字符串，则这条边称为单一转移边
>  - 压缩边: 当一组连续的相邻边 $(e_0, e_1, \dots, e_k)$ 都是单一转移边，这些边可以被压缩为一条边，该压缩边上的文本就是 $e_0, e_1, \dots, e_k$ 的拼接

Starting from an FSM based on characters, we recursively merge singular transition edges into their preceding ones until further compression is unfeasible, resulting in a Compressed FSM. 
>  从基于字符的 FSM 出发，我们递归地将所有的单一转移边和它们的前驱边合并，直到无法进一步压缩，我们就得到了压缩的 FSM

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/79dce60a-bf72-47ad-80c9-b5bbd0455d68/9ae8e259cb34ae7be50c423ae1181fd0e5fea9dd92467c26fa6e257da0789944.jpg)  

Figure 11: Comparison of decoding using Compressed FSM versus normal FSM: The left subfigure depicts the decoding process per forward pass, while the right subfigure explains the origins of various result components.

This approach speeds up the decoding process, as demonstrated by SGLang's runtime efficiency with the Compressed FSM, shown in Fig. 11.

## B.2 Handling Tokenization Artifacts with Retokenization
When a new token is generated, we get the token's string and search all the outgoing edges of the current state find the one that starts with the just decoded string, and then move forward. However, when the transition edge is a well-compressed one and contains a very long string, we may anticipate the next rounds' decoded strings as well. 
>  当新 token 生成后，我们获取该 token 的字符串，并在当前状态的所有出边中搜索以该已解码字符串为开头的边，然后推进状态
>  然而，如果某条转移边是高度压缩的，包含了非常长的字符串，我们可以提前预判后续解码可能产生的字符串

This is where the acceleration happens and we call this process Jump Forward. However, we still need to convert the string into tokens for the next decoding phases and it is not straightforward due to the LLM's specific pretraining and tokenization method; direct partitioning might alter the intended meaning [50]. 
>  这就是加速发生的地方，我们称这个过程为 Jump Forward 
>  然而，我们仍然需要将这些预判出的字符串转为 tokens，以便进入下一轮解码
>  但由于 LLM 具有特定的预训练和 tokenization 方法，直接进行任意切分可能改变原始语义

For example, the compressed text in Fig. 2's regex is  $\{$  summary": ", which can only be tokenized as  $\{$  ", summary, ": and  \_" according to the tokenizer instead of partition them randomly such as  $\{$  ", summa, ry, and ": \_". To address this issue, we use the original tokenizer to retokenize all the previous text as well as the text of the compressed edge, ensuring alignment with the original input format of LLMs. And it only brings minor retokenization overhead.
>  例如，Fig2 的 regex 中，压缩后的文本为 `{ "summray": "`，根据 tokenizer 的规则，它只能被 tokenize 为 `{", summary, ": , _"`，而不能随意划分为例如 `{", summa, ry, ":, _"`
>  为了解决这个问题，我们采用原始 tokenizer 对之前所有文本以及压缩边的文本重新 tokenize，确保和 LLM 源树输入格式对齐

## B.3 Future Extension: Addressing Distorted Probability
The challenge caused by the gap between strings and tokens also brings the problem of skewed probability distribution [50]. For example, in Fig. 2, the regex "`"[ABCD][+-]?"` suggests grades from A+ to D-, but if replaced with broader terms like `Excellent|Above Average|Fair|Below Average`, the runtime may inaccurately map an A to Above Average due to the term Above Average is on a compressed transition, misrepresenting the grade hierarchy. 
>  strings 和 tokens 之间的不匹配还带来了概率分布失真的问题
>  例如 Fig2 中的 regex `[ABCD][+-]` 表示从 `A+` 到 `D-` 的得分，但如果将它替换为更宽泛的术语例如 `Excellent|Above Average|Fair|Below Average`，运行时可能将 `A` 映射到 `Above Average`

This occurs because the LLM doesn't recognize the specific range of choices, leading to inappropriate token sequences. Computing accurate probabilities for each choice requires summing the probabilities of all token sequences that result in each choice, which complicates decoding and adds overhead. 
>  这是因为 LLM 无法识别特定的范围选择，导致选择了不合适的 token 序列
>  为每个选择计算正确的概率需要将该选择中的所有 token 的概率相加

One workaround is to include the choices or the regex directly in the prefill prompt, guiding the LLM to be aware of its choices and output the decision in proper token sequences. However, this approach doesn't solve the underlying issue of distorted probabilities, highlighting the need for further research to improve the compressed FSM's accuracy.

>  没看懂

# C Additional Experimental Setups and Results
**Additional experimental setups.** Fig. 5 and Fig. 6 are obtained by running Llama-7B on a single A10G (24GB) GPU. Fig. 7 are obtained by running Mixtral-8x7B on 8 A10G (24GB) GPUs with tensor parallelism. Fig. 8(c) is obtained by running Llama-7B on a single A10G (24GB) GPU. Fig. 12 are obtained by running Llama-70B on 4 A100G (80GB) GPUs with tensor parallelism. Table 2 are obtained by running LLaVA-v1.5-7B on a single A10G (24GB) GPU and running LLaVA-Next-34B on a single A100G (80GB) GPU. Each bar in the benchmark figures takes several minutes to an hour to run.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/79dce60a-bf72-47ad-80c9-b5bbd0455d68/54844b108b1311291fa031146e4c397234a3eec64004bc4f42ad7da52ff8849f.jpg)  

Figure 12: Normalized throughput on Llama-2-70B models with tensor parallelism. Higher is better.

![](https://cdn-mineru.openxlab.org.cn/result/2025-09-02/79dce60a-bf72-47ad-80c9-b5bbd0455d68/601ac58eac0c2cf455bc36fcbba7462918b6751fbf49161606bcb285af30a9b1.jpg)  

Figure 13: Achieved cache hit rate and optimal cache hit rate on various benchmarks.

**Additional experimental results.** Fig. 13 shows the achieved and optimal cache hit rates on the benchmarks listed in Fig. 5. Fig. 12 shows the throughput on Llama-2-70B with tensor parallelism.

# D Compiler Mode
Besides the interpreter mode used in the main body of the paper, another way to run SGLang programs is to compile them as computational graphs and execute them with a graph executor. This opens up opportunities for more compilation optimizations, as we can rewrite the graph and perform more static planning.
>  正文中介绍的主要是解释器模式，运行 SGLang 程序的另一种方式是将它们编译为计算图，然后使用图执行器执行它

## D.1 Design and Implementation

![[pics/SGLang-Fig14.png]]

We designed an intermediate representation (IR) for SGLang, which represents SGLang program structures and operations as a computational graph. This graph includes nodes for primitive operators and edges for dependencies. See Fig. 14b for the graph corresponding to the program in Fig. 14a. In the program, each call to a decorated function or fork creates a new prompt state or a stream.
>  我们为 SGLang 设计了 IR，它使用计算图表示 SGLang 程序结构和操作
>  图中包含了表示 primitive operators 的节点以及表示依赖的边
>  对装饰的函数的调用或 fork 都会创建一个新的 prompt state/stream

There are several types of nodes. Each operand of the operators  $+ =$  and  $+$  in a SGLang program is represented by an IR node. These include ConstantText, Argument, Gen, Select, Variable, Fork, GetForkItem, and Join. There are two types of dependencies: intra-stream dependency, where operations submitted into a stream using  $+ =$  must be executed after all preceding operations in that stream, and inter-stream dependency, which occurs when one stream needs to fetch variable values from another stream, necessitating synchronization. Operations like fork manipulate multiple streams and thus introduce inter-stream dependencies.
>  SGLang 中的 operators `+=`， `+` 的 operand 都表示为 IR node (怎么又和上面说的不一样的)
>  图中有两类依赖: stream 内依赖和 stream 间依赖
>  stream 内使用 `+=` 提交的 operations 必须顺序执行，stream 之间的依赖来自于一个 stream 获取另一个 stream 的结果，需要同步

To generate the graph, we use tracing to run the program with abstract arguments and construct the graph dynamically. This method is limited to programs without data-dependent control flow, a limitation we plan to address in future work. 
>  我们使用抽象参数追踪程序运行，来记录图，这个方法仅适用于没有依赖于数据的控制流的程序 

Once constructed, the graph can be executed through a graph executor, eliminating the need to reinterpret the original Python program. This results in benefits like graph rewriting optimizations, reduced runtime overhead, and program serialization. For execution, stream executors are launched for each data stream, dispatching IR nodes to the streams in topological order.
>  stream executors 会为每个数据流发起执行，将 IR 节点按照拓扑顺序分发到 stream 中

## D.2 A Case Study of Compiler Optimization: Code Movement for Improving Prefix Sharing
We explore a compilation optimization for SGLang IR: code movement for improving prefix sharing. We anticipate that more classical compiler techniques can also be applied, such as auto-tuning and instruction selection.
>  我们探索一种为 SGLang IR 的编译优化技术: 通过代码移动来增加前缀共享

This optimization aims to improve prefix sharing by reordering nodes in the graph to increase the length of the constant prefix. It does not strictly preserve the original computation, classifying it as an aggressive optimization. For instance, changing the prompt from "Here is a question  $+$  {question}. Please act as a math expert and solve the given question." to "Please act as a math expert and solve the given question. Here is a question  $+$  {question}." results in a longer shareable prefix. 
>  该优化旨在通过重新排序图中节点，来增加常量前缀长度，他不会严格保持原来的计算顺序
>  例如，将 prompt 从 `"Here is a question + {question}. Please act as a math expert and solve the given question."` 改为 `Please act as a math expert and solve the given question. Here is a question +{question}.` 就会得到一个更长的可共享前缀 (这样子优化会不会有点违背原本语义)

This optimization is interesting because traditional program analysis cannot achieve it due to the presence of natural language instructions in SGLang. Instead, we prompt GPT-4 to reorder graph nodes. We write a prompt with several examples to teach GPT-4 the concepts of SGLang IR, and we find that GPT-4 can successfully apply this optimization for some simple SGLang programs.
>  这种优化较为有趣，因为传统程序分析无法实现它，因为 SGLang 中包含自然语言指令，这些语义内容难以通过静态分析识别和重组
>  我们让 GPT-4 辅助重排计算图中的节点，发现它可以为简单的 SGLang 程序实现这类优化

We evaluate the effectiveness of this optimization. We collect 20 prompt templates from the internet and implement them in SGLang. We utilize 5 of these templates as few-shot training examples and the remaining 15 as test cases. Our evaluation shows that, for 12 out of the 13 templates, GPT-4 successfully reorders the graph nodes without altering the semantics, as confirmed by manual inspection of the modified prompts. On average, this optimization results in a 60-token increase in the shareable prefix length, showcasing GPT-4's effectiveness. Failures in creating optimized prompt order come from an incorrect understanding of the semantic meaning behind the graph nodes. It is too aggressive and puts all constants upfront even when such ordering changes the original semantics. This case study aims to explore the use of GPT-4 for compiler optimizations. More work is needed to make these kinds of optimizations reliable in the future.
