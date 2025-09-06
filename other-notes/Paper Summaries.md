# 2025
## September
### Week 1
**ORCA**: [[paper-notes/gen-ai/language/infra/system/Orca A Distributed Serving System for Transformer-Based Generative Models-2022-OSDI|Orca A Distributed Serving System for Transformer-Based Generative Models-2022-OSDI]]

两个关键技术: iteration-level scheduling, selective batching
iteration-level scheduling 即每次推理运行只执行整个 batch 的一个 iteration，而不是将 batch 中所有的 requests 都推理完，这提高了 batching requests 的灵活性，解决了 early finishing, late arriving 的问题，可以显著降低推理 latency

selective batching 则是为了在 iteration-level scheduling 的基础上能够灵活 batching 而提出的，因为使用 iteration-level scheduling 之后，同一个 batch 内的 requests 就不能保证处于相同阶段 (prefill or decode)，或者 decode 相同位置的 token，这使得不同 requests 的 Attention 计算要处理的 tensor shape 不同，进而不能像传统的批处理一样，通过堆叠相同 shape 的 tensor 为单个大 tensor，传递给 batch operator 来实现批处理

为此，selective batching 选择不对 Attention 计算执行 batching，而其他的计算例如 Linear, GeLU 则仍然可以通过堆叠 flattened tensor 来执行批处理 (因为 Attention 由 request 的 concept 需要读取同一个 request 的 KVCache 对该 request 的 token 进行 decode，其他计算没有)
分别处理 Attention 并不会太大地影响效率，因为 Attention 计算不需要模型参数，因此批处理带来地减少参数 load 的作用对 Attention 计算没有意义

ORCA system 的 scheduler 执行 iteration-level scheduling, engine 对收到的 batch 执行 selective batching 计算

ORCA 也采用了层内 + 层间并行 (张量并行 + 流水线并行)，并且分离了控制信息和数据信息的传输 (控制信息在 CPU 之间通过 gRPC，数据信息在 GPU 之间通过 NCCL)

**Transformers Libraray**: [[paper-notes/gen-ai/language/infra/system/Transformers State-of-the-Art Natural Language Processing-2020-EMNLP|Transformers State-of-the-Art Natural Language Processing-2020-EMNLP]]

这是 HuggingFace 开发的 Python 库，它解决的问题就是为市面上多种多样的预训练模型提供一个统一的调用接口

在该库中，每个市面上的预训练模型都封装为一个类，这些类可以通过会尽量具备相同的接口

这个库将一个 Transformer 模型的实例化分为三个部分: tokenizer, model, head
tokenizer 将原始文本转化为索引向量
model 将索引向量输入，得到语义嵌入
head 将语义嵌入转化为有意义的输出

tokenizer 实现为单独的库
model 实现为 Python 类
head 实现为对 model 类的封装类


