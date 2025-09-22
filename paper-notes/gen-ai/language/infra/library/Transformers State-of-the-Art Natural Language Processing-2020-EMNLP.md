```
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, Alexander M. Rush

Hugging Face, Brooklyn, USA / {first-name}@huggingface.co
```

# Abstract
Recent progress in natural language processing has been driven by advances in both model architecture and model pretraining. Transformer architectures have facilitated building higher-capacity models and pretraining has made it possible to effectively utilize this capacity for a wide variety of tasks. 

Transformers is an open-source library with the goal of opening up these advances to the wider machine learning community. The library consists of carefully engineered state-of-the-art Transformer architectures under a unified API. Backing this library is a curated collection of pretrained models made by and available for the community. Transformers is designed to be extensible by researchers, simple for practitioners, and fast and robust in industrial deployments. The library is available at https://github.com/huggingface/transformers.
>  `Transformers` 库为设计好的 Transformer 架构的模型提供了统一的 API
>  支撑该库的是一个由社区成员创建并可供社区使用的预训练好的模型的集合

# 1 Introduction
The Transformer (Vaswani et al., 2017) has rapidly become the dominant architecture for natural language processing, surpassing alternative neural models such as convolutional and recurrent neural networks in performance for tasks in both natural language understanding and natural language generation. The architecture scales with training data and model size, facilitates efficient parallel training, and captures long-range sequence features.

Model pretraining (McCann et al., 2017; Howard and Ruder, 2018; Peters et al., 2018; Devlin et al., 2018) allows models to be trained on generic corpora and subsequently be easily adapted to specific tasks with strong performance. The Transformer architecture is particularly conducive to pretraining on large text corpora, leading to major gains in accuracy on downstream tasks including text classification (Yang et al., 2019), language understanding (Liu et al., 2019b; Wang et al., 2018, 2019), machine translation (Lample and Conneau, 2019a), coreference resolution (Joshi et al., 2019), commonsense inference (Bosselut et al., 2019), and summarization (Lewis et al., 2019) among others.

This advance leads to a wide range of practical challenges that must be addressed in order for these models to be widely utilized. The ubiquitous use of the Transformer calls for systems to train, analyze, scale, and augment the model on a variety of platforms. The architecture is used as a building block to design increasingly sophisticated extensions and precise experiments. The pervasive adoption of pretraining methods has led to the need to distribute, fine-tune, deploy, and compress the core pretrained models used by the community.

Transformers is a library dedicated to supporting Transformer-based architectures and facilitating the distribution of pretrained models. At the core of the library is an implementation of the Transformer which is designed for both research and production. The philosophy is to support industrial-strength implementations of popular model variants that are easy to read, extend, and deploy. On this foundation, the library supports the distribution and usage of a wide-variety of pretrained models in a centralized model hub. This hub supports users to compare different models with the same minimal API and to experiment with shared models on a variety of different tasks.
>  `Transformers` 是一个专注于支持基于 Transformer 架构并促进预训练模型的分发的库
>  该库的核心是对 Transformer 的实现，旨在用于研究和生产环境
>  该库的理念是提供易于阅读、拓展和部署的流行模型变体的工业级实现，在此基础上，该库支持分发和使用中央模型仓库中的一系列预训练模型
>  该库使用户能够以相同的最小 API 比较不同的模型并在各种任务上使用共享模型进行实验

Transformers is an ongoing effort maintained by the team of engineers and researchers at Hugging Face with support from a vibrant community of over 400 external contributors. The library is released under the Apache 2.0 license and is available on GitHub<sup>1</sup>. Detailed documentation and tutorials are available on Hugging Face's website<sup>2</sup>.

# 2 Related Work
The NLP and ML communities have a strong culture of building open-source research tools. The structure of Transformers is inspired by the pioneering tensor2tensor library (Vaswani et al., 2018) and the original source code for BERT (Devlin et al., 2018), both from Google Research. The concept of providing easy caching for pretrained models stemmed from AllenNLP (Gardner et al., 2018). The library is also closely related to neural translation and language modeling systems, such as Fairseq (Ott et al., 2019), OpenNMT (Klein et al., 2017), Texar (Hu et al., 2018), Megatron-LM (Shoeybi et al., 2019), and Marian NMT (Junczys-Dowmunt et al., 2018). Building on these elements, Transformers adds extra user-facing features to allow for easy downloading, caching, and fine-tuning of the models as well as seamless transition to production. Transformers maintains some compatibility with these libraries, most directly including a tool for performing inference using models from Marian NMT and Google's BERT.
>  `Transformers` 的结构受到了其他相关库的影响，此外，`Transformers` 还增加了额外的用户友好功能，使得模型的下载、缓存、微调更加容易，并支持无缝过渡到生产环境

There is a long history of easy-to-use, user-facing libraries for general-purpose NLP. Two core libraries are NLTK (Loper and Bird, 2002) and Stanford CoreNLP (Manning et al., 2014), which collect a variety of different approaches to NLP in a single package. More recently, general-purpose, open-source libraries have focused primarily on machine learning for a variety of NLP tasks, these include Spacy (Honnibal and Montani, 2017), AllenNLP (Gardner et al., 2018), flair (Akbik et al., 2019), and Stanza (Qi et al., 2020). Transformers provides similar functionality as these libraries. Additionally, each of these libraries now uses the Transformers library and model hub as a low-level framework.

Since Transformers provides a hub for NLP models, it is also related to popular model hubs including Torch Hub and TensorFlow Hub which collect framework-specific model parameters for easy use. Unlike these hubs, Transformers is domain-specific which allows the system to provide automatic support for model analysis, usage, deployment, benchmarking, and easy replicability.
>  由于 `Transformers` 为 NLP 模型提供了一个仓库，故它也与一些流行的模型仓库相关，包括了 Torch Hub 和 TensorFlow Hub，这些框架收集了特定架构的模型参数以便于使用

# 3 Library Design
Transformers is designed to mirror the standard NLP machine learning model pipeline: process data, apply a model, and make predictions. Although the library includes tools facilitating training and development, in this technical report we focus on the core modeling specifications. For complete details about the features of the library refer to the documentation available on https://huggingface.co/transformers/.
>  `Transformers` 被设计为模仿标准的 NLP 机器学习模型流水线: 处理数据、应用模型、预测

![[pics/huggingface-transformers-Fig2.png]]

Every model in the library is fully defined by three building blocks shown in the diagram in Figure 2: (a) a tokenizer, which converts raw text to sparse index encodings, (b) a transformer, which transforms sparse indices to contextual embeddings, and (c) a head, which uses contextual embeddings to make a task-specific prediction. Most user needs can be addressed with these three components.
>  库中的每个模型都完全由三个 building blocks 定义:
>  1. tokenizer，将原始数据转化为稀疏的索引编码
>  2. transformer，将稀疏的索引转化为上下文嵌入
>  3. head，使用上下文嵌入执行针对任务的预测

**Transformers** Central to the library are carefully tested implementations of Transformer architecture variants which are widely used in NLP. The full list of currently implemented architectures is shown in Figure 2 (Left). While each of these architectures shares the same multi-headed attention core, there are significant differences between them including positional representations, masking, padding, and the use of sequence-to-sequence design. Additionally, various models are built to target different applications of NLP such as understanding, generation, and conditional generation, plus specialized use cases such as fast inference or multi-lingual applications.
>  该库的核心是完整实现的并详细测试的 Transformer 架构变体
>  完整的列表见 Fig2 (left)，这些架构在位置编码、掩码、填充和序列到序列设计的使用都存在不同
>  此外，不同的模型是针对不同的 NLP 应用设计，例如理解、生成、条件生成等

Practically, all models follow the same hierarchy of abstraction: a base class implements the model's computation graph from an encoding (projection on the embedding matrix) through the series of self-attention layers to the final encoder hidden states. The base class is specific to each model and closely follows the model's original implementation which gives users the flexibility to easily dissect the inner workings of each individual architecture. In most cases, each model is implemented in a single file to enable ease of extensibility.
>  所有的模型遵循相同的抽象层次: base class 实现了模型的计算图，从一个编码开始 (在嵌入矩阵上的投影)，经过一系列 self-attention 层，到最后的 encoder hidden states
>  每个模型都对应一个 base class，在遵循模型的原始实现的同时，也给为用户提供了灵活剖析每个架构内部工作机制的能力
>  在大多数情况下，每个模型用单个文件实现，方便拓展

Wherever possible, different architectures follow the same API allowing users to switch easily between different models. A set of Auto classes provides a unified API that enables very fast switching between models and even between frameworks. These classes automatically instantiate with the configuration specified by the user-specified pretrained model.
>  只要可能，不同的架构会遵循相同的 API，便于用户切换模型
>  一组 `Auto` classes 提供了统一的 API，便于用户切换模型，这些类会根据用户指定的预训练模型的配置自动实例化

**Tokenizers** A critical NLP-specific aspect of the library is the implementations of the tokenizers necessary to use each model. Tokenizer classes (each inheriting from a common base class) can either be instantiated from a corresponding pretrained model or can be configured manually. These classes store the vocabulary token-to-index map for their corresponding model and handle the encoding and decoding of input sequences according to a model's specific tokenization process. The tokenizers implemented are shown in Figure 2 (Right). Users can easily modify tokenizer with interfaces to add additional token mappings, special tokens (such as classification or separation tokens), or otherwise resize the vocabulary.
>  tokenizer classes 都继承自相同的 base class，这些类可以从对应的预训练模型实例化，也可以手动配置
>  tokenizer classes 存储了其对应模型的词表中 token-to-index 的映射，并根据模型特定的分词过程处理模型输入和输出序列的编码和解码
>  用户可以为 tokenizer 添加额外的 token mapping、特殊 tokens，或者改变词表大小

Tokenizers can also implement additional useful features for the users. These range from token type indices in the case of sequence classification to maximum length sequence truncating taking into account the added model-specific special tokens (most pretrained Transformer models have a maximum sequence length).
>  tokenizers 也为用户实现了额外的功能，包括在序列分类情况下使用的 token type 索引，以及在考虑模型特定的特殊 tokens 对序列进行阶段处理 (大多数预训练 Transformer 模型有最大序列长度)

For training on very large datasets, Python-based tokenization is often undesirably slow. In the most recent release, Transformers switched its implementation to use a highly-optimized tokenization library by default. This low-level library, available at https://github.com/huggingface/tokenizers, is written in Rust to speed up the tokenization procedure both during training and deployment.
>  `Transformers` 使用的是自定义的高性能 ` tokenizers ` 库而不是基于 Python 的 tokenization，` tokenizers ` 以 Rust 编写

**Heads** Each Transformer can be paired with one out of several ready-implemented heads with outputs amenable to common types of tasks. These heads are implemented as additional wrapper classes on top of the base class, adding a specific output layer, and optional loss function, on top of the Transformer's contextual embeddings. The full set of implemented heads are shown in Figure 2 (Top). These classes follow a similar naming pattern: XXXForSequenceClassification where XXX is the name of the model and can be used for adaptation (fine-tuning) or pretraining. Some heads, such as conditional generation, support extra functionality like sampling and beam search.
>  每个 Transformer 模型可以与多个已经实现的 heads 搭配使用
>  这些 heads 被实现为对基础类的额外包装类，在 Transfomer 的上下文嵌入之上添加了一个特定的输出层，以及可选的损失函数
>  这些类遵循类似的命名模式 `XXXForSequenceClassification`，其中 `XXX` 是用于微调和预训练的模型名称
>  一些 heads，例如条件生成，还支持额外的功能，例如采样和束搜索

For pretrained models, we release the heads used to pretrain the model itself. For instance, for BERT we release the language modeling and next sentence prediction heads which allows easy for adaptation using the pretraining objectives. We also make it easy for users to utilize the same core Transformer parameters with a variety of other heads for finetuning. While each head can be used generally, the library also includes a collection of examples that show each head on real problems. These examples demonstrate how a pretrained model can be adapted with a given head to achieve state-of-the-art results on a large variety of NLP tasks.
>  对于预训练模型，我们发布了用于预训练模型本身的 head
>  例如，对于 BERT 我们发布了 language modeling 和 next sentence prediction heads
>  库还包含了一系列示例，展示了在真实问题上使用 head 取得 SOTA 的结果

# 4 Community Model Hub

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-28/c68197af-3098-4201-ba14-4940b48c9c8c/f35249ad6e8b263c9fa483b0d38152f6db00b3b43b1e05630657c913b89e208e.jpg)  

Figure 1: Average daily unique downloads of the most downloaded pretrained models, Oct. 2019 to May 2020.

Transformers aims to facilitate easy use and distribution of pretrained models. Inherently this is a community process; a single pretraining run facilitates fine-tuning on many specific tasks. The Model Hub makes it simple for any end-user to access a model for use with their own data. This hub now contains 2,097 user models, both pretrained and fine-tuned, from across the community. Figure 1 shows the increase and distribution of popular transformers over time. While core models like BERT and GPT-2 continue to be popular, other specialized models including DistilBERT (Sanh et al., 2019), which was developed for the library, are now widely downloaded by the community.

The user interface of the Model Hub is designed to be simple and open to the community. To upload a model, any user can sign up for an account and use a command-line interface to produce an archive consisting a tokenizer, transformer, and head. This bundle may be a model trained through the library or converted from a checkpoint of other popular training tools. These models are then stored and given a canonical name which a user can use to download, cache, and run the model either for finetuning or inference in two lines of code. 
>  任意用户都可以注册账户，并使用命令行界面生成一个包含分词器、transformer 和 head 的存档文件
>  这个文件可以是通过该库训练的模型，也可以是由其他模型的 checkpoint 转换而，这些被上传的模型会被存储并拥有一个规范的名称
>  其他用户可以通过两行代码来下载、缓存、运行该模型

To load FlauBERT (Le et al., 2020), a BERT model pretrained on a French training corpus, the command is:

```python
tknzr = AutoTokenizer.from_pretrained(
"flaubert/flaubert_base_uncased") 
model = AutoModel.from_pretrained(
"flaubert/flaubert_base_uncased") 
```

When a model is uploaded to the Model Hub, it is automatically given a landing page describing its core properties, architecture, and use cases. Additional model-specific metadata can be provided via a model card (Mitchell et al., 2018) that describes properties of its training, a citation to the work, datasets used during pretraining, and any caveats about known biases in the model and its predictions. An example model card is shown in Figure 3 (Left).
>  当模型被上传到 Model Hub 后，它会自动获得一个介绍其核心属性、架构和使用场景的首页
>  用户还可以通过 model card 提供额外的模型特定的元数据

Since the Model Hub is specific to transformerbased models, we can target use cases that would be difficult for more general model collections. For example, because each uploaded model includes metadata concerning its structure, the model page can include live inference that allows users to experiment with output of models on a real data. Figure 3 (Right) shows an example of the model page with live inference. Additionally, model pages include links to other model-specific tools like benchmarking and visualizations. For example, model pages can link to exBERT (Hoover et al., 2019), a Transformer visualization library.

**Community Case Studies** The Model Hub highlights how Transformers is used by a variety of different community stakeholders. We summarize three specific observed use-cases in practice. We highlight specific systems developed by users with different goals following the architect, trainer, and end-user distinction of Strobelt et al. (2017):

*Case 1: Model Architects AllenAI*, a major NLP research lab, developed a new pretrained model for improved extraction from biomedical texts called SciBERT (Beltagy et al., 2019). They were able to train the model utilizing data from PubMed to produce a masked language model with state-of-the-art results on targeted text. They then used the Model Hub to distribute the model and promote it as part of their CORD -COVID-19 challenge, making it trivial for the community to use.

*Case 2: Task Trainers Researchers* at NYU were interested in developing a test bed for the performance of Transformers on a variety of different semantic recognition tasks. Their framework Jiant (Pruksachatkun et al., 2020) allows them to experiment with different ways of pretraining models and comparing their outputs. They used the Transformers API as a generic front-end and performed fine-tuning on a variety of different models, leading to research on the structure of BERT (Tenney et al., 2019). 

*Case 3: Application Users Plot.ly*, a company focused on user dashboards and analytics, was interested in deploying a model for automatic document summarization. They wanted an approach that scaled well and was simple to deploy, but had no need to train or fine-tune the model. They were able to search the Model Hub and find DistilBART, a pretrained and fine-tuned summarization model designed for accurate, fast inference. They were able to run and deploy the model directly from the hub with no required research or ML expertise.

# 5 Deployment
An increasingly important goal of Transformers is to make it easy to efficiently deploy model to production. Different users have different production needs, and deployment often requires solving significantly different challenges than training. The library therefore allows for several different strategies for production deployment.

One core property of the library is that models are available both in PyTorch and TensorFlow, and there is interoperability between both frameworks. A model trained in one of frameworks can be saved through standard serialization and be reloaded from the saved files in the other framework seamlessly. This makes it particularly easy to switch from one framework to the other one along the model lifetime (training, serving, etc.).
>  该库的一个核心性质是模型可以在 PyTorch 和 TensorFlow 之间转换

Each framework has deployment recommendations. For example, in PyTorch, models are compatible with TorchScript, an intermediate representation of a PyTorch model that can then be run either in Python in a more efficient way, or in a high-performance environment such as C++. Fine-tuned models can thus be exported to production-friendly environment, and run through TorchServing. TensorFlow includes several serving options within its ecosystem, and these can be used directly.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-28/c68197af-3098-4201-ba14-4940b48c9c8c/433cc0b0672ec0051acca374ee71d3d226d332c4383078dffda194e695c57f40.jpg)  

Figure 4: Experiments with Transformers inference in collaboration with ONNX.

Transformers can also export models to intermediate neural network formats for further compilation. It supports converting models to the Open Neural Network Exchange format (ONNX) for deployment. Not only does this allow the model to be run in a standardized interoperable format, but also leads to significant speed-ups. Figure 4 shows experiments run in collaboration with the ONNX team to optimize BERT, RoBERTa, and GPT-2 from the Transformers library. Using this intermediate format, ONNX was able to achieve nearly a 4x speedup on this model. The team is also experimenting with other promising intermediate formats such as JAX/XLA (Bradbury et al., 2018) and TVM (Chen et al., 2018).

Finally, as Transformers become more widely used in all NLP applications, it is increasingly important to deploy to edge devices such as phones or home electronics. Models can use adapters to convert models to CoreML weights that are suitable to be embedded inside a iOS application, to enable on-the-edge machine learning. Code is also made available. Similar methods can be used for Android devices.

# 6 Conclusion
As Transformer and pretraining play larger roles in NLP, it is important for these models to be accessible to researchers and end-users. Transformers is an open-source library and community designed to facilitate users to access large-scale pretrained models, to build and experiment on top of them, and to deploy them in downstream tasks with state-of-the-art performance. Transformers has gained significant organic traction since its release and is set up to continue to provide core infrastructure while helping to facilitate access to new models.

