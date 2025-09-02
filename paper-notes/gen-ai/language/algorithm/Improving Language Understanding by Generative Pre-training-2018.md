# Abstract
自然语言理解 (Natural language understanding) 包含了广泛的不同的任务 (diverse tasks)，例如文本蕴涵 (textual entailment)，问答 (question answering)，语义相似度评估 (semantic similarity assessment)，以及文档分类 (document classification)，即便大规模的无标签的文本语料库 (unlabeled text corpora) 是充足的，用于学习这些特定任务的有标签数据是稀缺的，这使得针对这些任务训练的模型难以表现良好

我们将表明，可以通过在多样的 (diverse) 无标签文本语料库上生成式预训练 (generative pre-training) 一个语言模型，然后为特定的任务进行判别式微调 (discriminative fine-tuning)，在这些任务上取得很大的性能提升

和之前的方法不同，我们在微调时利用了任务感知的 (task-aware) 输入变换 (input transformations)，以在对模型架构需要最少改变的 (minimal changes) 情况下，达到高效的迁移

我们将展示我们方法在广泛的自然语言理解基准测试中的有效性，我们的通用的任务不可知的模型 (general task-agnostic model) 超过了使用针对任务的架构的判别式训练的模型，在所研究的12个任务中，显著提高了9个任务的 SOTA，例如，我们在常识推理 (commensense reasoning)(故事完型填空测试 Stories Cloze Test) 上提高了8.9%，在问答 (RACE) 上提高了5.7%，在文本蕴含 (MultiNLI) 上提高了1.5%

# 1 Introduction
可以从原始文本 (raw text) 中高效学习的能力对于缓解 NLP 对监督学习的依赖是至关重要的，多数 NLP 方法需要大量人工标注的数据，这限制了这些方法在许多缺乏有注释资源的领域的可用性，在这种情况下，可以利用无标签数据中的语言学信息 (linguistic information) 的模型就提供了对人工收集注释信息的替代方案
进一步说，即便大量的有监督信息是可用的，通过无监督方式学习好的表示是可以显著提升性能的，证据便是对预训练词向量 (pre-trained word embedding) 的广泛使用已经提高了一系列 NLP 任务的性能

但要从未标记的数据中利用超过单词级别 (word-level) 的信息是有挑战性的，首先，尚不明确哪一类优化目标对于学习对迁移有用的文本表示 (text representations that are useful for transfer) 是最高效的，近期的研究关注了许多目标，例如语言建模[44]，机器翻译[38]，语篇连贯 (discourse coherence)[22]，每个方法在各自在一些不同的任务上优于其他方法；其次，对于如何将这些学习到的表示迁移到目标任务上的最有效方式尚无共识，现存的方法包括了对模型架构进行针对任务的改变[43, 44]，使用复杂的学习方案 (intricate learning schemes)[21]，添加辅助学习目标[50]等
上述两点不确定性使得开发高效的用于语言处理的半监督学习方法具有挑战性

本文中，我们探索一种结合了无监督预训练和有监督微调的用于语言理解任务的半监督方法，我们的目标是学习到通用表示 (universal represetations)，可以用最少的适应就迁移到一系列任务
我们假设有一个大规模的无标签文本语料库以及数个有标签的数据集 (对应目标任务)，我们并不要求目标任务和无标签语料库是同一域的 (domain)
我们采用两阶段训练过程，首先，使用语言建模目标 (language modeling objective) 在无标签数据集上学习模型的初始参数，只有，在目标任务的数据集用对应的有监督目标 (supervised objective) 对模型参数微调

我们使用 Transformer 作为模型架构，Transformer 已经在机器翻译、文档生成、句法解析 (syntatic parsing) 展现优秀的性能，相较于循环网络，该架构提供了更结构化的记忆 (structured memory) 以处理长期依赖 (long-term dependencies)，因此具有迁移到多样任务的更健壮的迁移表现
迁移时，我们使用了针对任务的输入适应 (task-specific input adaptations)，它来源于遍历式的方法 (traversal-style approaches)[52]，它会将结构化的文本输入处理为单个连续的 token 序列 (single contiguous sequence of tokens)，就如我们的实验中所示，这些适应方法可以让我们对预训练模型架构进行最小的改变就能高效微调

我们在四类语言理解任务中评估我们的模型——自然语言推理 (natural language inference)，问答，语义相似度，文本分类，
我们的通用的任务不可知的模型超过了使用针对任务的架构的判别式训练的模型，在所研究的12个任务中，显著提高了9个任务的 SOTA，例如，我们在常识推理 (commensense reasoning)(故事完型填空测试 Stories Cloze Test) 上提高了8.9%，在问答 (RACE) 上提高了5.7%，在文本蕴含 (MultiNLI) 上提高了1.5%，在 GLUE 多任务基准测试中提高了5.5%，
我们还在四个不同设定下分析了预训练模型的零样本行为，并展示预训练模型已经获取了对于下流任务有用的语言学知识 (linguistic knowledge)

# 2 Related Work
**Semi-supervised learning for NLP**
我们的工作大致属于自然语言的半监督学习范畴，这种范式吸引了很多研究者的兴趣，它被应用于诸如序列标注 (sequence labeling)[24, 33, 57]或文本分类 (text classification)[41, 70]等任务，
最早的这类方法使用未标记数据来计算单词级或短语级统计信息 (word-level or phrase-level statistics)，然后将其作为特征 (features) 用于监督模型[33]，在过去的几年中，研究人员已经证明了使用在未标记语料库上训练的 (trained on unlabeled copora) 词嵌入 (word embeddings)[11, 39, 42]的好处，以改善各种任务的性能[8, 11, 26, 45]，
然而，这些方法主要迁移的是单词级信息 (transfer word-level information)，而我们的目标是捕获更高层次的语义 (semantics)

最近的方法已经研究了从未标记数据中学习和利用超过单词级语义的可能性，例如短语级 (phrase-level) 或句子级 (sentence-level) 嵌入也可以通过使用未标记语料库进行训练，且它们已被用于将文本编码 (encode text into) 为适合于各种目标任务的向量表示[28, 32, 1, 36, 22, 12, 56, 31]

**Unsupervised pre-training**
无监督预训练是半监督学习的一个特例，其目标是为监督学习找到一个良好的初始化点 (initialization point)，而不是修改监督学习的目标 (objective)，
早期的工作探索了该技术在图像分类[20, 49, 63]和回归任务[3]中的应用，
随后的研究[15]证明，将预训练作为一种正则化方案 (regularization scheme)，可以使得深度神经网络在泛化方面表现更好，在最近的研究中，该方法已被用于帮助训练深度神经网络完成各种任务，如图像分类[69]、语音识别[68]、实体消歧 (entity disambiguation)[17]和机器翻译[48]

与我们的工作最接近的研究线涉及使用语言建模目标 (using language modeling objective) 预训练神经网络，然后在有监督的目标任务上进行微调，
Dai 等人[13]和 Howard 和 Ruder[21]遵循这种方法来改进文本分类，然而，尽管预训练阶段有助于捕获一些语言信息，但他们使用的 LSTM 模型将他们的预测能力在限制在了短范围内 (restrict their prediction ability to a short range)，相比之下，我们选择的 transformer 允许我们捕获更长远的语言结构 (longer-range linguistic structure)，正如我们在实验中所展示的，
此外，我们还证明了我们的模型在更广泛的任务范围内的有效性，包括自然语言推理、释义检测 (parapharse detection) 和故事完成 (story completion)，
其他方法[43, 44, 38]使用来自于预训练的语言或机器翻译模型的隐藏表示仅作为辅助特征 (auxiliary features)，同时在目标任务上训练新的监督模型，这涉及到为每个单独的目标任务引入大量的新参数，而我们在迁移期间只需要对我们的模型架构进行最小的更改

**Auxiliary training objectives**
添加辅助的无监督训练目标 (auxiliary unsupervised training objective) 是半监督学习的另一种形式，Collobert 和 Weston[10]的早期工作使用了多种辅助的 NLP 任务，如词性标注 (POS tagging)、分块 (chunking)、命名实体识别 (named entity recognition) 和语言建模 (language modeling)，以改善语义角色标注 (semantic role labeling)，最近，Rei[50]在他们的目标任务目标中添加了一个辅助语言建模目标，并展示了在序列标注 (sequence labeling) 任务上的性能提升
我们的实验也使用了一个辅助目标，但正如我们所示，无监督预训练已经学习了语义上与目标任务相关的几个方面的知识 (several linguistic aspects relevent to target tasks)

# 3 Framework
训练过程由两阶段，第一阶段是在大文本语料库上学习高容量 (high-capacity) 语言模型，第二阶段是微调，用有标签数据将模型适应至不同的任务

## 3.1 Unsupervised pre-training
给定包含了大量 token 的无监督语料库 $\mathcal U = \{u_1,\dots,u_n\}$，我们使用标准的语言建模目标 (language modeling objective)，最大化以下似然：

$$L_1(\mathcal U) = \sum_i \log P(u_i | u_{i-k},\dots,u_{i-1};\Theta)\tag{1}$$

其中 $k$ 是上下文窗口的大小 (context window)，条件概率 $P$ 使用参数为 $\Theta$ 的 NN 建模，参数使用 SGD 训练

我们使用多层 Transformer decoder 用于语言建模，这是 Transformer 的变体，该模型对输入上下文 tokens 进行多头自注意力运算，然后进行逐位的前向传播，产生在目标 tokens 上的输出分布 (output distribution over targe tokens)：

$$\begin{align}
h_0 &= UW_e + W_p\\
h_l &= \text{transformer\_block}(h_{l-1})\quad\forall i\in[1,n]\\
P(u) &= \text{softmax}(h_nW_e^T)
\end{align}\tag{2}$$

其中 $U = (u_{-k}, \dots, u_{-1})$ 是上下文的 tokens 的向量 (the context vector of tokens)，$n$ 是层数，$W_e$ 是 token 的嵌入矩阵，$W_p$ 是位置嵌入矩阵 (position embedding matrix)

## 3.2 Supervised fine-tuning
使用 eq. 1的目标预训练模型后，我们将参数适应至目标任务，
假设有有标签数据集 $\mathcal C$，数据集中每个实例是一个输入 tokens 序列 $x^1,\dots,x^m$ 和一个标签 $y$，该输入被传递给预训练模型，得到最后一个 transformer 块的激活 $h_l^m$，然后将其输入给一个新加的线性输出层 (参数是 $W_y$) 以预测 $y$：

$$P(y|x^1,\dots,x^m) = \text{softmax}(h_l^mW_y)\tag{3}$$

因此我们需要最大化以下目标：

$$L_2(\mathcal C) = \sum_{(x,y)} \log P(y|x^1,\dots,x^m)\tag{4}$$

我们额外发现对微调添加语言建模作为辅助目标 (auxiliary objective) 可以通过 a) 提高有监督模型的泛化性 b) 加速收敛来帮助学习，这和前人工作是一致的[50, 43]，这些工作同时发现使用辅助目标可以提高性能，具体地说，我们优化以下目标 (其中 $\lambda$ 是权重)：

$$L_3(\mathcal C) = L_2(\mathcal C) + \lambda * L_1(\mathcal C)\tag{5}$$

总的来说，我们在微调时需要的唯一的额外参数就是 $W_y$，以及分隔符 tokens (delimiter tokens) 的嵌入

## 3.3 Task-specific input transformations
![[GPT-Fig1.png]]
对于一些任务，例如文本分类，我们可以直接按照如上所述微调我们的模型，对于其他的一些任务，例如 QA 和文本蕴含，它们具有结构化的输入，例如有序的句子对 (ordered sentence pairs)，或文档、问题和答案的三元组 (triplets of documents, question and answer)，而由于我们的预训练模型是在连续的文本序列 (contiguous sequences of text) 上训练的，我们需要一些修改才能将模型应用于这些任务

先前的工作提出在迁移的表示上学习针对任务的架构[44]，这种方法重新引入了大量的针对任务的自定义内容，而对于这些额外的架构成分是没有应用迁移学习的，
因此，我们使用的是遍历风格方法[52]，我们将结构化的输入转化为我们的预训练模型可以处理的有序序列，该输入变换 (transformation) 允许我们避免对不同任务使用的架构进行大量变换，
所有的变换都包括了添加随机初始化的开始和结束 tokens ($\langle s \rangle, \langle e \rangle$)

**Textual entailment**
对于文本蕴含任务，我们将前件 (premise) $p$ 和后件 (hypothesis) $h$ 各自的 token 序列拼接在一起，中间由一个定界符 (delimiter) token ($) 隔开

**Similarity**
对于相似度任务，两个相互比较的句子之间是没有内在的顺序的 (inherent ordering)，为了反应这一点，我们同样将两个句子之间通过定界符拼接，但提供了两种可能的顺序，模型独立处理两个不同顺序的输入，生成两个序列表示 (sequence representations) $h_l^m$，这两个序列表示会按元素相加，然后输入到线性输出层 (linear output layer)

**Question Answering and Commonsense Reasoning**
这类任务中，我们被给定一个文档 $z$，一个问题 $q$，以及一个可能答案的集合 $\{a_k\}$，我们将文档和问题拼接，然后将其与各个可能答案分别拼接，之间会添加一个定界符 token，即 $[z;q;\$;a_k]$，模型独立处理每一个这样的序列，最后通过一个 softmax 层进行规范化，输出在可能答案上的分布 (produce an output distribution over possible answers)

# 4 Experiments
## 4.1 Setup
**Unsupervised pre-training**
我们使用 BooksCorpus 数据集[71]来预训练语言模型，该数据集包含了超过7000本不同流派的独一无二的未出版 (unpublished) 书籍，包括冒险、奇幻和浪漫等，至关重要的是，它包含了长段连续的文本 (long stretches of contiguous text)，这使得生成模型能够学习条件于长距离信息 (learn to condition on long-range information)，
另一个可替代的数据集是1B Word Benchmark，它由和本文类似的方法 ELMo[44]所使用，它的大小和 BooksCorpus 大致相同，但其内容在句子级别上被打乱 (shuffled at a sentence level)，这破坏了长距离结构 (long-range structure)，我们的语言模型在这个语料库上达到了非常低的词级困惑度 (token level perplexity) 18.4

**Model specifications**
我们的模型在很大程度上遵循了原始的 transformer 工作[62]，我们训练了一个12层的 decoder-only transformer，带有掩蔽自注意力头 (masked self-attention heads)(768维状态和12个注意力头)，对于逐位的前向网络，我们使用了3072维的内部状态 (inner states)

我们使用了 Adam 优化方案[27]，最大学习率为2.5e-4，学习率在前2000次更新中从零线性增加，并使用余弦调度退火至0 (annealed to 0 using a cosine schedule)；
我们使用大小为64的 minibatch (minibatch 由64个随机采样的、连续的512 tokens 序列组成 contiguous sequence of 512 tokens)，训练100个 epoch；
由于在整个模型中广泛使用了 layernorm[2]，使用 $N (0,0.02)$ 进行简单的权重初始化就足够了；
我们使用了一个进行了40000次合并 (with 40,000 merges) 的字节对编码 (BPE bytepair encoding) 词汇表 (vocabulary)[53]；
我们为残差、嵌入和注意力计算都添加了 dropout，dropout 率为0.1，用于正则化，我们还采用了在[37]中提出的修改的 (modified) L2正则化，对所有非偏置或增益权重 (non bias or gain weights) 使用 $w=0.01$；
对于激活函数，我们使用了高斯误差线性单元 (GELU Gaussian Error Linear Unit)[18]，我们使用学习到的位置嵌入 (learned position embeddings)，而不是原始 transformer 工作中提出的正弦版本；
我们使用*ftfy*库来清理 (clean) BooksCorpus 中的原始文本 (raw text)，标准化一些标点符号和空白 (standardize some punctuation and whitespace)，并使用*spaCy*分词器 (tokenizer)

**Fine-tuning details**
除特殊说明外，我们都重用了无监督预训练的超参数设置，我们向分类器添加了 dropout，dropout 率为0.1，
对于大多数任务，我们令学习率为6.25e-5且批量大小为32，
我们的模型微调得很快，对于大多数情况，3个周期 (epoch) 的训练就足够了，我们使用线性学习率衰减调度 (linear learning raet decay schedule)，同时会预热0.2%(with warmup over 0.2%)，$\lambda$ 设置为0.5

## 4.2 Supervised fine-tuning
我们在各种监督任务上进行实验，包括自然语言推理、问答、语义相似性和文本分类，其中一些任务属于 GLUE 多任务基准 (multi-task benchmark)[64]的一部分

**Natural Language Inference**
自然语言推理 (NLI) 任务，也称为识别文本蕴含 (recognizing textual entailment)，它涉及阅读一对句子 (a pair of sentences)，并判断它们之间的关系 (relationship) 是 `entailment` (蕴含)、`constradiction` (矛盾)，还是 `neutral` (中性)，由于存在诸如词汇蕴含 (lexical entailment)、指代 (coreference) 和词汇及句法歧义 (lexical and syntatic ambiguity) 等各种现象，这项任务仍然具有挑战性
我们在五个数据集上进行评估，数据集的来源多样 (diverse source)，包括图片标题 image captions (SNLI)、转录演讲 transcribed speech、流行小说 popular fiction 和政府报告 government reports (MNLI)、维基百科文章 wikipedia articles (QNLI)、科学考试 science exams (SciTail) 或新闻文章 news articles (RTE)

Table2详细列出了我们模型和先前 SOTA 方法在不同 NLI 任务上的各种结果，我们的方法在五个数据集中的四个上显著优于基线，在 MNLI 上提高了1.5%，在 SciTail 上提高了5%，在 QNLI 上提高了5.8%，在 SNLI 上提高了0.6%，超过了之前的 SOTA，这表明了我们的模型能够更好地推理多个句子 (reason over multiple sentences)，并处理语言歧义 (handel aspects of linguistic ambiguity)，
在 RTE 上，我们评估的较小数据集之一 (2490个实例)，我们达到了56%的准确率，低于多任务 (multi-task) biLSTM 模型报告的61.7%，鉴于我们的方法在较大的 NLI 数据集上的强性能，我们的模型可能会从多任务训练中受益 (benefit from multi-task traning)，但目前我们还没有探索这个问题

**Question answering and commensense reasoning**
另一个需要单句和多句推理方面 (aspects and multi-sentence reasoning) 的任务是问答，
我们使用 RACE 数据集[30]，其包含有来自中学和高中考试的英语文章和相关的问题 (English passages with associated questions)，这个语料库被证明包含比其他数据集如 CNN[19]或 SQuaD[47]更多的推理类型问题 (more reasoning type questions)，很适合用于评估我们的被训练于处理长距离上下文 (long-range contexts) 的模型，
此外，我们还在 Story Cloze Test[40]上进行评估，该测试涉及到从两个选项中选择一个作为多句故事 (multi-sentence stories) 的正确结尾 (correct ending)

在这些任务上，我们的模型再次以显著的优势 (significant margins) 超越了之前的最佳结果，在 Story Cloze Test 上提高了8.9%，在 RACE 上整体提高了5.7%，这证明了我们的模型能够有效地处理长距离上下文 (handle long-range context)

**Semantic Similarity**
语义相似性 (或释义检测 paraphrase detection) 任务涉及预测两个句子在语义上是否等价 (semantically equivalent)，该任务的挑战在于识别概念的重述 (recognizing rephrasing of concepts)、理解否定 (understanding negation)，并处理句法歧义 (handling syntatic ambiguity)，
我们使用三个数据集来完成这项任务——微软释义语料库 (MRPC Microsoft Paraphrase corpus)(从新闻来源收集)[14]、Quora 问题对 (QQP Quora Question Pairs) 数据集[9]和语义文本相似性基准 (STS-B Semantic Textual Similarity Benchmark)[6]

我们在三个语义相似性任务中的两个上取得了 SOTA (Table 4)，在 STS-B 上提高了1个百分点，QQP 上比单任务 BiLSTM+ELMo+Attn 相比，提高了4.2%

**Classification**
最后，我们还评估了两个不同的文本分类任务，
其中语言可接受性语料库 (CoLA Corpus of Linguistic Acceptability)[65]包含专家对句子是否符合语法的判断 (whether a sentenc is grammatical or not)，并测试了训练模型的内在语言偏置 (innate linguistic bias)，而斯坦福情感树库 (SST-2 Standford Sentiment Treebank)[54]是一个标准的二元分类任务

我们的模型在 CoLA 上获得了45.4分，超过了之前最好结果35.0，这展示了我们的模型学习到了内在的语言偏置，我们的模型还在 SST-2上实现了91.3%的准确率，与 SOTA 相媲美
我们还在整个 GLUE 基准上获得了72.8的总分，明显优于之前最好的68.9

总体而言，我们的方法在我们评估的12个数据集中的9个上取得了新的 SOTA，在许多情况下超越了集成模型，我们的结果还表明，我们的方法在不同大小的数据集上 (across datasets of different sizes) 都表现良好，从较小的数据集如 STS-B (约5.7k 训练示例)——到最大的 SNLI (约550k 训练示例)

# 5 Analysis
![[GPT-Fig2.png]]
**Impact of number of layers transferred**
我们观察了从无监督预训练到有监督目标任务迁移不同数量的层的影响，Figure2 (left) 展示了我们方法在 MultiNLI 和 RACE 上的性能和迁移层数的关系，我们观察到标准结果，即增加迁移层数可以提高性能，而且每一层 Transformer 层都能提供额外的好处，在 MultiNLI 完全迁移12层 Transformer 层可以比原来的性能可以提高9%，这表明预训练模型中的每一层 (each layer) 都包含了用于解决目标任务的有用功能 (functionality for solving target task)

**Zero-shot Behaviors**
我们希望更好理解 Transformer 语言模型的预训练为什么是有效的，一个假设是生成式模型会在与训练中，为了提高它的语言建模能力 (in order to improve its language modeling capability)，会学习执行我们将评估的许多自然语言任务，并且 Transformer 的结构化的注意力记忆 (attentional memory) 相较于 LSTM 会更利于迁移

我们使用生成式模型 (Transformer 或 LSTM) 在没有监督微调的情况下执行一系列任务 (即零样本执行)，我们在 Figure2 (right) 中可视化了生成式模型在预训练过程中在这些任务上的表现，我们观察到模型的性能在训练过程中是稳步提高的 (steadily increases over training)，这表明生成式预训练 (generative pretraining) 支持了和广泛任务相关的功能的学习 (support the learning of a wide range of task relevant functionality)，我们还观察到 LSTM 的零样本表现有更高的方差，这表明 Transformer 架构的归纳偏差 (inductive bias) 更有助于迁移 (assists in transfer)

在 Figure2 (right) 中，对于 CoLA (linguistic acceptability 语言可接受性)，我们通过生成模型对实例赋予的平均 token 对数概率对实例进行评分 (examples are scored as the average token log-probability the generative model assigns)，并通过阈值化进行预测；对于 SST-2 (sentiment analysis 情感分析)，我们在每个实例后添加一个 `very` token，并将语言模型的输出分布限制为仅有 `positive` 和 `negative` 这两个词，并用它赋予更高概率的 token 作为预测；对于 RACE (question answering 问答)，我们选择生成模型在条件于文档和问题 (conditioned on the document and question) 下赋予最高平均 token 对数概率的答案；对于 DPRD[46] (winograd 模式解析 schema resolution)，我们用两个可能的指代表词 (referrents) 替换了定代词 (definite pronoun)，并选择在替换后生成模型赋予剩余序列更高平均 token 对数概率的解析作为预测 (resolution)

**Ablation studies**
![[GPT-Table5.png]]
我们进行了三种不同的消融研究 (Table 5)，
首先，我们实验了在微调期间没有辅助语言建模目标 (auxiliary LM objective) 的方法的表现，我们观察到辅助目标有助于 NLI 任务和 QQP，总体趋势表明，较大的数据集会从辅助目标中受益，但较小的数据集则不然，
其次，我们通过与使用相同框架的单层2048单元的 LSTM 进行比较，分析了 Transformer 效果，我们观察到使用 LSTM 代替 Transformer 时平均分数下降了5.6，LSTM 只在数据集 MRPC 上胜过 Transformer，
最后，我们还将我们的模型与没有进行预训练，直接在监督目标任务上训练的 Transformer 架构进行了比较，我们观察到预训练的缺少 (the lack of pre-training) 损害了所有任务的性能，与我们完整的模型相比下降了14.8%

# 6 Conclusion
我们介绍了一个框架，通过生成式预训练和判别式微调，使得单一的任务不可知模型 (single task-agnostic model) 达到强大的自然语言理解能力

通过在包含了大量的长连续文本 (long stretches of ontiguous text) 的多样化 (diverse) 语料库上预训练，我们的模型获得了大量的世界知识 (acquire significant world knowledge) 以及处理长距离依赖的能力 (ability to process long-range dependencies)，这些知识和能力可以成功地被迁移，用于处理判别式 (discriminative) 任务，例如问答、语义相似度评估、蕴含确定、文本分类，在我们研究的12个数据集中的9个上提高了 SOTA

使用无监督预训练来提高判别式任务的性能一直是机器学习研究的重要目标，我们的工作表明，实现显著的性能提升确实是可能的，并提供了哪些模型 (Transformers) 和数据集 (具有长距离依赖的文本 text with long range dependencies) 最适合这种方法的线索