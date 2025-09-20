Site: https://lmsys.org/blog/2024-02-05-compressed-fsm/
Date: 2024-02-05

Constraining an LLM to consistently generate valid JSON or YAML that adheres to a specific schema is a critical feature for many applications. In this blog post, we introduce an optimization that significantly accelerates this type of constrained decoding. Our approach utilizes a compressed finite state machine and is compatible with any regular expression, thereby accommodating any JSON or YAML schema. Distinct from existing systems that decode one token at one step, our method analyzes the finite state machine of a regular expression, compresses singular transition paths, and decodes multiple tokens in a single step whenever feasible. 
>  让 LLM 遵循特定的模式，生成有效的 JSON, YAML 是许多应用中的关键功能
>  本文介绍一个加速这类约束解码的优化，我们使用压缩 FSM，和现有系统一次解码一个 token 不同，我们的方法会分析 regex 的 FSM，压缩单一转移路径，在可能的时候一次解码多个 tokens

In comparison to state-of-the-art systems (guidance + llama.cpp, outlines + vLLM), our method can reduce the latency by up to 2x and boost throughput by up to 2.5x. This optimization also makes constrained decoding even faster than normal decoding. You can try it now on [SGLang](https://github.com/sgl-project/sglang/tree/main?tab=readme-ov-file#json-decoding).
>  这个优化降低了 2x 延迟，提高了 2.5x 吞吐，并使得约束解码甚至比正常解码更快

![](https://lmsys.org/images/blog/compressed_fsm/demo.gif)

Figure 1: Comparison of SGLang and Outlines + vLLM in JSON Decoding

## Background
[JSON](https://en.wikipedia.org/wiki/JSON) is one of the most important formats for data interchange. Requiring LLMs to always generate valid JSON can render the output of the LLM easily parsable in a structured manner. Recognizing its significance, OpenAI introduced the JSON mode, which constrains the model to always return a valid JSON object. However, more fine-grained control is often needed to ensure that the generated JSON object adheres to a specific schema, such as
>  OpenAI 的 JSON mode 约束模型总是返回有效的 JSON 对象
>  要让模型返回的 JSON 对象遵循特定的模式，还需要更细粒度的控制

![](https://lmsys.org/images/blog/compressed_fsm/json_schema.png)

Figure 2: Example of Constrained Generation Following a JSON Schema

For local LLMs, there are two major methods to guide the model to generate JSON objects that follow a specific schema.
>  对于本地 LLM，引导模型生成遵循特定模式的 JSON 对象有两类方式

### Method 1: Finite State Machine Based
This method involves transforming the JSON schema into a regular expression. We can then construct a [Finite State Machine(FSM)](https://en.wikipedia.org/wiki/Finite-state_machine) based on the regular expression. The FSM is used to guide the LLM generation. For every state within the FSM, we can calculate the permissible transitions and identify the acceptable next tokens. This allows us to track the current state during decoding and filter out invalid tokens by applying logit bias to the output. You can learn more about this method in the [outlines](https://arxiv.org/abs/2307.09702) paper.
>  基于 FSM 的方法将 JSON 模式转化为正则表达式，然后基于正则表达式构建 FSM
>  FSM 用于引导 LLM 生成，对于 FSM 中的每个状态，我们可以计算出它允许的转移路径，以确定下一个可接收的 tokens
>  FSM 使得我们在解码时可以追踪当前所处的状态，并对输出引用 logit bias，过滤无效 tokens

![](https://lmsys.org/images/blog/compressed_fsm/method1.png)

Figure 3: Constrained Decoding based on FSM and Logits Masking. In the first constrained decoding pass, only `age` is allowed. In the second pass, as the regex requires digits, both `0` and `1` are allowed, but the LLM would sample `1` with a higher probability.

The FSM-based method utilizes generalized regular expressions to define the low-level rules, which can be applied to a wide range of grammars, such as JSON schema, IP addresses, and emails.
>  基于 FSM 的方法利用了 regex，故非常通用

**Limitations:**  
Since the FSM is constructed at the token level, it can transition the state by only one token at each step. Consequently, it can decode only one token at a time, which results in slow decoding.
>  FSM 在 token-level 构建，它每一步只能转移一个 token (状态之间相差一个 token)，因此一次只能 decode 一个 token

### Method 2: Interleaved-Based
Aside from converting the entire JSON schema into a regular expression, another approach is to employ interleaved-based decoding. In this method, a given JSON schema can be broken down into several parts, each containing either a chunked prefill part or a constrained decoding part. These different parts are executed interleavedly by the inference system. Because the chunked prefill can process multiple tokens in a single forward pass, it is faster than token-by-token decoding.
>  FSM 方法将整个 JSON 模式转化为一个 regex
>  交错式解码将给定 JSON 模式分解为多个部分，每个部分要么包含一个 chunked prefill part 要么包含一个 constrainted decoding part
>  这些不同的部分由推理系统交错执行
>  由于 chunked prefill 可以在一次前向传播处理多个 tokens，它比 token-by-token 解码更快

>  大致意思是，反正无论如何都要求 LLM 的输出满足 JSON schema，可以将 schema 中的固定字符串模式，例如下图中的绿色部分，直接作为 prefill 输入给 LLM，让 LLM 计算其 KVCache，然后对于实际要生成的内容，即下图中的蓝色部分，再真正地一步一步 decoding
>  这样，在解码过程中实际上是 prefill 和 decoding 交错进行，自然比完全 decoding 快

[Guidance](https://github.com/guidance-ai/guidance?tab=readme-ov-file#guidance-acceleration) provides a set of syntax rules for interleaved-based decoding, using llama.cpp as a backend.

![](https://lmsys.org/images/blog/compressed_fsm/method2.png)

Figure 4: Interleaved JSON Decoding in Guidance

**Limitations:**

- The interleaved-based method requires custom syntax, making it less versatile and expressive than individual regular expressions.
- It struggles with correctly handling tokenization boundaries due to potential conflicts between the decode and chunked prefill segments.
- Frequent communication between the interpreter and the backend brings additional overhead.

>  基于交错的方法要求自定义语法，比 regex 的形式更不通用
>  不容易处理 tokenization 边界，decode 和 chunked prefill 片段可能存在冲突 (这个问题感觉可以解决，用模型的 tokenizer 处理一下给定的 schema 即可)
>  解释器和后端的频繁通信带来可开销

## Our Method: Jump-Forward Decoding With a Compressed Finite State Machine
We can combine the advantages of FSM-based and interleaved-based methods by introducing a new decoding algorithm, **jump-forward** decoding, based on the compressed finite state machine.
>  我们结合了基于 FSM 方法的优势和交错式方法的优势，提出了基于压缩 FSM 的 jump-forward 解码

During the decoding process guided by the regex converted from the JSON schema, we can predict forthcoming strings when we reach specific junctures:

- In [figure3](https://lmsys.org/blog/2024-02-05-compressed-fsm/#figure3), at the beginning of decoding, according to the regex, we can anticipate the incoming string to be:
    
    ```json
    {
      "name":
    ```
    
    Then comes the actual decoding part.
- Similarly, when the LLM outputs a `G` while filling in the house attribute of a character, we can confidently predict that the next string will be `ryffindor`, thereby completing the full string as `Gryffindor`.

>  在由 regex 引导的解码过程中，我们其实可以在达到某些节点的时候，预测未来的字符串
>  - Fig3 中可以看到，在解码开始时，我们已经可以预判接下来的字符串必须是 `{\n "name":`，然后才是真正的解码部分
>  - 类似地，当 LLM 在解码 `"house"` 的时候输出了 `G`，我们可以根据 schema 预判接下来必须是 `ryffindor`

That is precisely how the jump-forward decoding algorithm makes decoding faster. In the jump-forward algorithm, we examine the finite state machine of the given regular expression, identify all the singular transition edges, and compress consecutive ones together into **singular paths**. Instead of decoding the singular paths token by token, we can directly prefill (extend) them, jumping forward until the next branching point.
>  这就是 jump-forward 解码算法是如何加速解码的
>  在该算法中，我们先分析 FSM，识别单一转移边，将连续的单一转移边合并为一条单一转移边
>  我们不对单一转移边逐 token 解码，而是直接进行 prefill (extend)，直接跳跃到下一条分支点

![](https://lmsys.org/images/blog/compressed_fsm/compare.png)

Figure 5: Comparison of Jump-Forward Decoding with Compressed FSM and Normal Decoding

The RadixAttention mechanism of SGLang greatly simplifies the implementation of the jump-forward decoding algorithm. When executing a jump-forward, we can simply terminate the current request and enqueue a new one. The RadixAttention and efficient **extend** primitive in the SGLang runtime will automatically reuse the KV cache of the previous tokens, thereby avoiding redundant computation.
>  RadixAttention 机制简化了 jump-forward 解码算法的实现
>  当执行 jump-forward 时，我们直接终止当前请求 (终止 `gen`)，并将新请求入队，(原请求排去做 `extend`) RadixAttention 和 SGLang 运行时的高效 ` extend `  原语会自动复用之前 tokens 的 KVCache，避免重复计算

### Tokenization Boundary Handling
When implementing constrained decoding, it is always tricky to deal with the tokenization boundary, due to the complicated possible mapping between characters and tokens.
>  因为字符和 tokens 之间的复杂的可能映射关系，在约束解码时对 tokenization 边界的处理通常不容易

During LLM decoding, it might prefer (means with higher probability) to combine multiple characters into a single token. For instance, when decoding `"Hello"` in the context of JSON decoding, LLMs may output tokens like this:

`"` `He` `llo` `",`

>  在 LLM 解码时，模型可能更倾向于 (即更高概率) 将多个字符合并为单个 token
>  例如，在 JSON 解码的背景下解码 `"Hello"` 时，LLM 可能会输出: `" , He, llo, ",`

Instead of decoding the last `"` , it always prefers to combine it with a following `,` to form a more frequent token `",` . This effect may cause some strange behaviors. For example, in the above case, if the regex is set to `"[\w\d\s]*"` (without the last `,` ), it can lead to endless decoding because an LLM wants to stop with `",` but this token is not allowed.
>  模型没有解码出后面的单独 `"`，而是将它和 `,` 结合为 `",` 表示一个更常见的 token
>  这个现象可能导致一些异常的行为，例如 regex 要求一定以 `"` 为结尾时，LLM 总是无法解码出 `"`，而是只能解码出 `",`，导致解码无法停止

Moreover, during jump-forward decoding, we've found that different tokenization strategies to the jump-forwarded part may lead to different logit distributions for the subsequent tokens. Simply appending the tokenized jump-forwarded section to the current token sequence might yield unexpected outcomes.
>  此外，在 jump-forward 解码中，我们发现对 jump forwarded 的部分的不同 tokenization 策略可能导致后续 tokens 不同的 logit 分布
>  简单地将 tokenized 的 jump-forwarded 部分追加到当前 token 序列可能导致意料之外的结果

To manage these issues, we propose the following solutions:

- We have implemented a re-tokenization mechanism during the jump-forward phase. This involves appending the string instead of the tokens, followed by a re-tokenization of the entire text. This method effectively resolves most tokenization issues and results in only a minor increase in computational overhead, approximately 4%.
- Prefer the use of a comprehensive regular expression to guide the entire decoding process, rather than employing multiple concatenated regular expressions. This approach ensures that both FSM and LLM are cognizant of the entire decoding process, thereby minimizing boundary-related issues as much as possible.

>  为了解决这些问题，我们提出以下的方法:
>  - 我们为 jump-forward 阶段实现 re-tokenization 机制，该机制会将 string 而不是 tokens 追加到当前结果，然后对整个文本 re-tokenization，(然后把 re-tokenzation 的 jump-forward 部分送去 prefill) 这个方法有效地解决了大多数 tokenization 问题，但会带来少量计算开销
>  - 偏好使用一个完整的正则表达式来引导全局解码过程，而不是使用多个拼接的正则表达式，这个方法确保 FSM 和 LLM 都能感知整个解码流程，最大程度地减少边界相关的问题 (主要是避免 FSM 的边界断点，可能两个 FSM 的边界断点结合起来就是 LLM 想输出的，但分开来就导致单独的 FSM 不认可 LLM 的输出)

You can also read some additional discussion in this [blog post](http://blog.dottxt.co/coalescence.html).

## Benchmark Results
We benchmarked our jump-forward decoding on two tasks:

- Crafting a character's data in JSON format, guided by a brief prompt.
- Extracting a city's information from a long document and outputing it in JSON format.

We tested llama-7B on an NVIDIA A10 GPU (24GB), and used vllm v0.2.7, guidance v0.1.0, outlines v0.2.5 and llama.cpp v0.2.38(Python binding) . The figure below shows the throughput (using the maximum batch size supported by each system) and latency (with a batch size of 1) of these methods:

![](https://lmsys.org/images/blog/compressed_fsm/result.png)

Figure 6: Benchmark Results

The results show that SGLang with our decoding algorithm significantly outperforms all other systems. It can reduce the latency by up to 2x and boost throughput by up to 2.5x. In the character generation task, even SGLang without Jump-Forward achieves higher throughput than Outlines+vLLM; we suspect this is due to some overhead in Outlines.
>  SGLang + jump-forward decoding 可以 2x 减低延迟并 2.5x 提高吞吐

## Use Cases
We have been testing this feature with [Boson.ai](https://boson.ai/) for two weeks, who are bringing this feature into their production use cases because it guarantees robust response with higher decoding throughput.

Additionally, another user used this feature to extract structured information from images by utilizing the vision language model, LLaVA.

![](https://lmsys.org/images/blog/compressed_fsm/llava_demo.gif)

Figure 7: Extracting structured information from an image using SGLang and LLaVA

## Link

- You can try this feature now in [SGLang](https://github.com/sgl-project/sglang/tree/main?tab=readme-ov-file#json-decoding).
- Benchmark code is available [here](https://github.com/sgl-project/sglang/tree/main/benchmark/json_jump_forward).
- We thank [outlines](https://github.com/outlines-dev/outlines) for open-sourcing its FSM implementation. We built our compressed FSM based on it.
