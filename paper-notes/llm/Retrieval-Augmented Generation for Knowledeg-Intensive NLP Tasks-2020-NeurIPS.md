# Abstract
我们已经知道大预训练语言模型可以在它们的参数中存储事实性知识(factual knowledge)，然后再为下流NLP任务微调之后达到SOTA的效果，但这些模型访问并精确地运用知识(access and precisely manipulate knowledge)的能力仍然受限，因此在知识密集(knowledge-intensive)的任务中，它们的性能仍落后于针对特定任务的架构

另外，为它们的决策提供出处(provenance)以及更新它们的世界知识(world knowledge)仍是开放的研究问题

具有对显式的非参数记忆的可微访问机制(differentiable access mechanism to explict non-parametric memory)的预训练模型目前为止仅针对提取式(extractive)下游任务进行了研究

我们探索了一种通用的微调方法(general-purpose fine-tuning recipe)，用于检索增强的生成(retrieval-augmented generation RAG)——模型将结合预训练的参数化记忆和非参数化记忆进行语言生成，
我们将介绍RAG模型，RAG模型中，参数化记忆是一个预训练的seq2seq模型，非参数化记忆是维基百科的密集向量索引(dense vector index)，它通过一个预训练的神经检索器访问(neural retriever)

我们比较了两种RAG构成(formulations)，一种在生成整个序列时都条件于相同的检索到的段落(retrieved passages)，另一种则每个token都使用不同的段落

我们在广泛的知识密集的NLP任务中微调并评估了我们的模型，在三个开放领域(open domain)QA任务上达到了SOTA，超越了参数化的seq2seq模型和针对特定任务的检索和提取架构(retrieve-and-extract architectures)，
对于语言生成任务，我们发现RAG模型会生成相对于SOTA的seq2seq参数化模型，会生成更具体、多样和更具事实性的语言(more specific, diverse and factual language)
# 1 Introduction
我们已经知道预训练神经语言模型可以从数据中学习到大量深入的知识，它们不需要访问任意外部记忆，自己具有参数化的隐式知识库(parametrized implicit knowledge base)，这类模型的缺点在于难以轻松拓展或修改它们的记忆(expand or revise their memory)，不能直接为它们的预测提供见解(insight into their predictions)，以及可能产生“幻觉 hallucinations”

混合模型结合了参数记忆和非参数(即基于检索的 retrieval-based)记忆，它可以解决上述的一些问题，因为知识可以直接被修改或拓展，并且访问的知识(accessed knowledge)也可以被审查和解释(be inspected and interpreted)，
REALM和ORQA是两个最近推出的模型，它们将掩蔽(masked)语言模型和一个可微的检索器结合，但它们仅探索了开放域提取式问答(open-domain extractive question answering)，
本文中，我们将混合参数记忆和非参数记忆的方法用于seq2seq模型

我们通过一个通用目的的微调方法(我们称其为检索增强的生成 RAG)，为预训练的、参数记忆的生成模型赋予非参数记忆，
在RAG模型中，参数记忆是预训练的seq2seq transformer，非参数记忆是维基百科的一个密集向量索引，通过预训练的神经检索器访问
我们将这些成分结合入一个概率模型，然后端到端训练(trained end-to-end)，如Figure 1所示
![[RAG-Fig1.png]]
其中的检索器(密集段落检索器 Dense Passage Retriever/DPR)，提供条件于输入的隐文档(latent document conditioned the input)，其中的seq2seq模型(BART)条件于这些隐文档以及输入，生成输出
我们使用top-K近似对隐文档进行边际化(marginalize)，隐文档的使用有两种情况，一种是所有的输出tokens都条件于同一文档，另一种是不同的输出tokens条件于不同的文档

和T5或BART类似，RAG可以在任意seq2seq任务中微调，进行RAG微调时，生成器(generator)和检索器(retriever)是共同学习的

实际上已经有大量的工作提出用非参数记忆来丰富(enrich)系统，但这些系统是针对特定任务从零开始训练的，例如记忆网络(memory networks)，栈增强网络(stack-augmented networks)，记忆层(memory layers)，
与之相反，我们探索的设定中，参数记忆和非参数记忆成分都是预训练和预加载的(pre-trained and pre-loaded)，重要的是，在预训练的知识访问机制下，模型可以在不需要额外训练的情况下，展现出访问知识的能力

我们的结果强调了结合参数记忆和非参数记忆与生成(generation)对知识密集任务的优势，知识密集任务即人类在不访问外部知识源的情况下也难以良好处理的任务
RAG模型在open Natural Questions，WebQuestions和CuratedTrec中达到了SOTA，同时在TriviaQA中显著优于使用了特定的预训练目标的方法，上述任务都是提取式任务(extractive tasks)，除这类任务以外，我们发现RAG不受限的生成(unconstrained generation)也可以优于之前的提取式方法，
对于知识密集的生成，我们在MS-MARCO和Jeopardy question generation中进行了实验，我们发现我们的模型相较于BART可以生成更具事实性、更具体且更多样的的回答(response)，
在FEVER事实验证任务(fact verification)上，我们达到了与使用强检索监督的SOTA流水线模型相差4.3%的结果，
最后，我们证明了随着世界的变化，非参数记忆可以被替换以更新模型的知识
# 2 Methods
RAG模型使用输入序列$x$检索文本文档(text document)$z$，然后在生成目标序列$y$时，使用这些文档作为额外的上下文(context)，如Figure 1所示，模型包括两个成分：
(i) 一个检索器$p_{\eta}(z|x)$，其参数为$\eta$，检索器返回给定查询$x$时，关于文本段落(text passage)的(top-K个)分布 
(ii) 一个生成器$p_{\theta}(y_i | x, z, y_{1:i-1})$，其参数为$\theta$，生成器基于由前$i-1$个tokens $y_{1:i-1}$和原始输入$x$以及检索到的段落$z$组成的上下文，生成当前的token

为了端到端训练检索器和生成器，我们将检索到的文档视为隐变量，我们提出两种模型，用不同的方式在隐文档上边际化(marginalize over the latent documents)以产生生成文本的分布(distribution over generated text)，
其中一种是RAG-Sequence，该模型预测每个目标token都使用相同的文档，
另一种是RAG-Token，该模型可以基于不同的文档预测不同的目标token
## 2.1 Models
**RAG-Sequence Model**
RAG-Sequence模型使用相同的文档生成完整的序列，
技术上说，它将检索到的文档视为单一的隐变量，将其边际化以通过top-K近似得到seq2seq概率$p(y|x)$
具体地说，RAG-Sequence模型通过检索器检索到top K个文档，然后生成器为每个文档产生输出概率(为不同的$z$计算$p_{\theta}(y|x,z)$)，然后对其边际化(乘上$p_{\eta}(z|x)$然后求和)：
$$p_{\text {RAG-Sequence}}(y|x)\approx \sum_{z\in\text{top-}k(p(\cdot|x))}p_{\eta}(z|x)p_{\theta}(y|x,z) = \sum_{z\in\text{top-}k(p(\cdot|x))}p_{\eta}(z|x) \prod_{i}^N p_{\theta}(y_i|x,z,y_{1:i-1})$$

**RAG-Token Model**
RAG-Token模型中，我们可以为不同的目标token选择不同的隐文档，然后对应地边际化，这允许生成器生成答案时可以从多个文档中选择内容，
具体地说，RAG-Token模型使用检索器检索到top K个文档，然后生成器为每个文档都产生下个token的输出概率分布，然后为这个token边际化，然后为下个输出token重复该过程：
$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i}^N \sum_{z\in\text{top-}k(p(\cdot|x))}p_{\eta}(z|x)p_{\theta}(y_i|x,z,y_{1:i-1})$$
通过将目标类别看作长度为一的目标序列，RAG模型就可以用于序列分类任务，此时RAG-Sequence模型和RAG-Token模型等价
## 2.2 Retriever: DPR
检索概率$p_{\eta}(z|x)$基于DPR，DPR遵循一个双编码器架构：
$$p_{\eta}(z|x) \propto \exp(\mathbf d(z)^T \mathbf q(x))\quad \mathbf d(z) = \text {BERT}_d(z),\mathbf q(x) = \text{BERT}_q(x)$$
其中$\mathbf d(z)$表示一个文档的密集表示(dense representation)，由一个$\text{BERT}_{\text{BASE}}$文档编码器(document encoder)生成，而$\mathbf q(x)$是一个查询表示(query representation)，由一个查询编码器(query encoder)生成，同样基于$\text{BERT}_{\text{BASE}}$

计算top-k$(p_{\eta}(\cdot|x)$)，即$k$个具有最高先验概率$p_{\eta}(z|x)$的文档$z$，是一个最大内积搜索问题(MIPS maximum inner product search)，这个问题可以在次线性时间(sub-linear time)解决

我们使用DPR中预训练好的bi-encoder来初始化我们的检索器以及建立文档索引(document index)，该检索器会被训练于检索包含了TriviaQA问题以及Natural Questions问题的答案的文档

我们称文档索引为非参数记忆
## 2.3 Generator: BART
生成概率$p_{\theta}(y_i|x,z,y_{1:i-1})$可以使用任意编码器-解码器结构建模，我们使用BART-large，一个预训练的，参数量为400M的seq2seq transformer，我们将$x$和$z$简单拼接，然后输入给BART

BART是使用去噪(denoising)目标以及一系列不同的去噪函数进行预训练的，它在一系列多样的生成任务上达到SOTA，超过和它大小相近的T5模型

我们称BART生成器的参数$\theta$为参数记忆
## 2.4 Training
我们共同训练检索器和生成器，且没有任何关于应该检索哪个文档的直接监督知识，给定用于微调训练的语料库(一系列输入输出对)$(x_j,y_j)$，我们使用Adam，通过随机梯度下降，最小化每个目标的负边际对数似然$\sum_{j}-\log p(y_j|x_j)$

在训练时更新文档编码器$\text{BERT}_d$是开销很大的，因为这需要周期性地更新文档索引，就如REALM在与训练时做的一样，而我们发现这一步对于优秀的模型性能并非必要，因此在训练时我们保持文档编码器(以及索引)固定，只微调查询编码器$\text {BERT}_q$以及$\text {BART}$生成器
## 2.5 Decoding
在测试时，RAG-Sequence和RAG-Token需要不同的方式以近似$\arg\max_{y}p(y|x)$

**RAG-Token**
RAG-Token模型可以视作一个标准的自回归seq2seq生成器，它的转移概率(trainsition probablity)是：$p'_{\theta}(y_i|x,y_{1:i-1}) = \sum_{z\in \text{top}-k(p(\cdot | x))}p_{\eta}(z_i|x)p_{\theta}(y_i|x,z_i,y_{1:i-1})$，在解码时，我们将$p'_{\theta}(y_i | x, y_{1:i-1})$放入一个标准的beam解码器

**RAG-Sequence**
对于RAG-Sequence模型，似然$p(y|x)$不能分解为由常规的per-token似然相乘的形式，因此我们不能用单个beam搜索来解决，
我们为每个文档$z$都进行一次beam搜索，使用$p_{\theta}(y_i|x,z,y_{1:i-1})$来为每个假设(hypothesis)进行打分，这会得到一个假设集合$Y$，假设集合中的一些假设可能不会在所有文档的beam中都出现，
为了估计一个假设$y$的概率，我们对$y$没有出现在它的beam中的文档$z$都进行一次额外的前向传播(forward pass)，得到概率$p_{\theta}(y|x,z)$，然后我们乘上相应的先验概率$p_{\eta}(z|x)$，再进行求和得到边际概率($\sum_{z\in\text{top-}k(p(\cdot|x)} p_{\eta}(z|x)p_{\theta}(y|x,z) = p_{\theta}(y|x)$)我们称该解码过程为“完全解码 Thorough Decoding”，
对于更长的输出序列，$|Y|$可以变得很大，因此需要更多额外的前向传播，为了更高效的解码，我们可以作一个进一步的近似，当$y$在$x,z_i$的beam搜索中没有生成时，就令$p_{\theta}(y|x,z_i)\approx 0$，此时就不需要再进行额外的前向传播了，只需要利用候选集(candidate set)$|Y|$中现有的概率边际化即可，我们称该解码过程为“快速解码 Fast Decoding”
# 3 Experiments
我们在一系列知识密集任务中实验RAG，对于所有实验，我们使用同一个Wikipedia转储(dump)作为非参数知识源(non-parametric knowledge source)，我们使用的是2018年12月的转储，每个Wikipedia文章被分割为不重叠的100词块(100-word chunks)，最后得到21M个文档(documents)

我们使用文档编码器为每个文档计算嵌入，然后使用FAISS建立一个MIPS索引，该索引使用了层次化可导航小世界(Hierarchical Navigable Small World)近似以实现快速检索

在训练时，我们为每个查询检索出top $k$个文档，训练时$k\in \{5, 10\}$，测试时使用开发数据集(dev data)来设定$k$的值
## 3.1 Open-domain Question Answering
开放域问答 Open-domain question ansewering(QA)是重要的现实世界应用，也是常见的知识密集任务测试平台(testbed)，我们将问题和答案视为输入-输出文本对$(x,y)$，然后通过最小化答案的负对数似然直接训练RAG

我们将RAG和流行的提取式QA范式(extractive QA paradigm)进行比较，其答案主要从文档中提取，即主要依赖于非参数知识
我们也比较了“闭卷QA Closed-Book QA”方法，该方法和RAG类似，是生成答案，但并没有利用检索，也就是完全依赖参数知识

我们考虑四个常用的开放域QA数据集：Natural Questions(NQ)，TriviaQA(TQA)，WebQuestions(WQ)，CuratedTrec(CT)，因为CT和WQ比较小，我们遵循DPR[26]，用我们的RAG NQ模型初始化CT和WQ模型

我们使用和先前工作相同的训练/开发/测试划分，报告精确匹配(EM Exact Match)分数，对于TQA数据集，为了和T5[52]比较，我们还在TQA Wiki测试集上进行了评估
## 3.2 Abstractive Question Answering
RAG模型可以超越简单的提取式问答，通过自由形式的、抽象的文本生成来回答提问，
为了测试RAG在知识密集环境中的自然语言生成(natural language generatoin NLG)能力，我们使用了MSMARCO NLG task v2.1，这项任务包括问题、针对每个问题从搜索引擎检索到的十个黄金段落(ten gold passages)，以及从检索到的段落中标注出的完整句子答案(full sentence answer)，
我们不使用该数据集中提供的段落，只使用问题和答案，将MSMARCO视为一个开放域的抽象问答任务(open-domain abstractive QA task)，
MSMARCO中有一些不访问黄金段落就无法以匹配参考答案的方式回答(be answered in a way that matches the reference answer)的问题，例如“What is the weather in Volcano, CA?”(加利福尼亚州火山地区的天气如何?)，因此如果没有使用黄金段落，性能将会降低，
我们还注意到，有些MSMARCO问题仅使用维基百科是无法回答的，此时，RAG可以依赖参数知识(parametric knowledge)来生成合理的回答
## 3.3 Jeopardy Question Generation
为了在非问答设定(non-QA setting)中评估RAG的生成能力，我们研究了开放域问题生成(open-domain question generation)，
我们没有使用标准开放域QA任务中的问题(这些问题通常较简短和简单)，而是提出了更具挑战性(more demanding)的任务：生成Jeopardy问题

Jeopardy是一种不寻常的格式，它要求根据有关该实体的事实(a fact about the entity)来猜测一个实体，
例如，“The World Cup”是问题“In 1986 Mexico scored as the first country to host this international sports competition twice”的答案，
由于Jeopardy问题是精确的事实陈述(precise factual statements)，因此条件于它们的答案实体(conditoned on answer entitise)生成Jeopardy问题就构成了一个有挑战性的知识密集型生成任务

我们使用了SearchQA[10]的划分，其中包含100K个训练样本、14K个开发样本和27K个测试样本，由于这是一个新任务，我们从头训练了一个BART模型用于和RAG比较，我们使用SQuAD调整的Q-BLEU-1度量(SQuAD-tuned Q-BLEU-1 metric)进行评估，Q-BLEU是BLEU的一个变体，它对匹配实体的权重更高(higher weight for matching entities)，并且相较于标准度量，它与人类对问题生成的判断(human judgment for question generation)的相关性(correlations)更高

我们还进行了两次人类评估，一次是评估生成的事实性(factuality)，另一次是评估明确性(specificity)，我们将事实性定义为一个陈述(statement)是否可以被可信的外部来源证实(corroborated)，将明确性定义为输入和输出之间的相互依赖(mutual dependence between input and output)程度，
我们遵循最佳实践，使用成对比较评估(pairwise comparative evaluation)[34]，评估者被展示一个答案和两个生成的问题，一个来自BART，一个来自RAG，然后，他们被要求在四个选项中选择一个——问题A更好、问题B更好、两个都很好、两个都不好
## 3.4 Fact Verification
FEVER[56]要求分类一个自然语言声明(natural language claim)是否由维基百科支持(supported)或反驳(refuted)，或者是没有足够的信息来决定，
这项任务需要从维基百科检索与声明相关的证据(evidence)，然后对这些证据进行推理(reasoning over)，以将声明分类为是真实的、虚假的，还是仅凭维基百科无法验证的

FEVER是一个检索问题(retrieval problem)，结合了一个具有挑战性的蕴含推理任务(entailment reasoning task)，它还为探索RAG模型处理分类而非生成的能力提供了合适的测试平台，
我们将FEVER的类别标签(支持 supports、反驳 refutes 或信息不足 not enough info)映射到单个输出token，然后直接用claim-class pairs进行训练，
至关重要的是，与大多数其他FEVER方法不同，我们没有使用检索到的证据(retrieved evidence)作为监督信息，在许多现实世界的应用中，检索监督信号(retrieval supervision signals)是不可用的(not available)，因此不依赖此类监督信息的模型将适用于更广泛的任务范围

我们探索了两个变体：标准的三元分类任务(支持/反驳/信息不足)和Thorne和Vlachos[57]研究的二元(支持/反驳)任务，在这两种情况下，我们都报告了标签准确率(label accuracy)
# 4 Results
## 4.1 Open-domain Question Answering
![[RAG-Table 1.png]]
Table 1显示了RAG以及SOTA模型的结果，在所有四个开放域QA任务中，RAG都达到了新的SOTA，
RAG结合了“闭卷 closed-book”(仅参数 parametric only)方法的生成灵活性和“开卷 open-book”基于检索方法的性能，与REALM和T5+SSM不同，RAG在没有昂贵的、专门的“显著跨度掩蔽 salient span masking”预训练[20]的情况下取得了强劲的结果

值得注意的是，RAG的检索器是使用DPR的检索器初始化的，后者在Natural Questions和TriviaQA上使用了检索监督(retrieval supervision)信息来训练，
RAG相较于DPR QA系统更好，而后者使用了基于BERT的“交叉编码器 cross-encoder”重新排名(re-rank)文档，同时使用了一个提取式阅读器(extractive reader)，因此RAG证明了重新排名器(re-ranker)或提取式阅读器对于SOTA的性能并非必要

即使有可能提取(extract)答案，生成(generate)答案也有优势，其一是一些并未包含确切的答案(verbatim answer)但包含有关答案的线索(clues)的文档仍然可以为生成正确答案做出贡献，这在标准提取式方法中是不可能的，因此在多个文档上进行边际化(marginalization over documents)是有效的，此外，即使在任何检索到的文档中都没有正确答案的情况下，RAG仍能生成正确答案，在NQ中这种情况下的准确率达到了11.8%，而提取式模型的得分将是0%
## 4.2 Abstrastive Question Answering
![[RAG-Table 2.png]]
如Table 2所示，RAG-Sequence在Open MS-MARCO NLG任务上以2.6个Bleu点和2.6个Rouge-L点的优势超过了BART，
考虑到 i) 其他模型可以访问包含了生成参考答案所需的特定信息的黄金段落ii) 许多问题是没有黄金段落就无法回答的(unanswerable) iii) 并非所有的问题都可以仅通过维基百科回答，RAG接近SOTA模型的性能是令人印象深刻的

Table 3展示了我们的模型生成的一些答案，从质量上看，我们发现RAG模型产生的幻觉较少，生成事实正确文本(factually correct text)的频率比BART更高
![[RAG-Table 3.png]]
## 4.3 Jeopardy Question Generation
![[RAG-Table 4.png]]
Table 2显示，RAG-Token在Jeopardy问题生成任务上的表现优于RAG-Sequence，同时两个模型在Q-BLEU-1上都超过了BART
Table 4显示了人类对来自BART和RAG-Token的452对生成样本的评估结果，评估者指出，在只有7.1%的情况下，BART比RAG更事实性，而RAG在42.7%的情况下比BART更事实性，还有17%的情况下RAG和BART都是事实性的，这清楚地证明了RAG在任务上比SOTA的生成模型更有效，评估者还发现RAG生成的内容在明确性上大幅度(by large margin)优于BART

Table 3展示了每个模型的典型生成结果

Jeopardy问题通常包含两个独立的信息片段(two separate pieces of information)，RAG-Token表现最好的原因可能是它可以生成结合了多个文档内容的响应(responses that combine content from several documents)

![[RAG-Fig2.png]]
Fig 2展示了一个例子，在生成“Sun”时，提到“The Sun Also Rises”的文档2的后验概率(posterior)很高，类似地，当生成“A Farewell to Arms”时，文档1主导了后验概率
有趣的是，在每本书的第一个token生成后，文档后验概率趋于平缓(the document posterior flattens)，这一观察表明，生成器(generator)可以在不依赖特定文档的情况下完成标题(complete the titles without depending on specific documents)，换句话说，模型的参数知识足以(sufficient)完成标题
我们通过为BART-only模型喂入(feed)部分解码(partial decoding)`“The Sun”`，找到了支持这一假设的证据，我们发现，BART完成了`“The Sun Also Rises” is a novel by this author of "The Sun Also Rises"` 的生成，这表明了标题“The Sun Also Rises”存储在BART的参数中，类似地，在喂给BART `"The Sun Also Rises" is a novel by this author of "A` 之后，BART将完成该部分解码(partial decoding)任务，输出`“The Sun Also Rises” is a novel by this author of “A Farewell to Arms”` 
这个例子展示了参数和非参数记忆如何协同工作(work together)——非参数组件有助于引导生成(guide the generation)，提取出存储在参数记忆中的特定知识(drawing out specific knowledge store in the parametric knowleddge)
## 4.4 Fact Verification
Table 2展示了我们在FEVER上的结果，
对于三元分类，RAG的得分与SOTA模型相差4.3%，而这些SOTA模型是具有特定领域架构(domain-specific architectures)和大量工程构建的复杂流水线系统(pipeline systems)，它们使用中间检索监督(intermediate retrieval supervision)进行训练，而RAG是不需要检索监督的，
对于二元分类，我们与Thorne和Vlachos[57]进行比较，他们训练了RoBERTa[35]来根据给定的黄金证据句子(given gold evidence sentence)将声明分类为真或假，RAG执行推理时只有声明，依靠自己检索证据(retrieving its own evidence)，它达到了与该模型相差2.7%的准确率

我们还分析了RAG检索到的文档是否与FEVER中被注释为黄金证据的文档相对应，我们计算了RAG检索到的top $k$个文档的标题与FEVER中的黄金证据文档的标题之间的重叠(overlap in artical titles)，我们发现，在71%的情况下，检索到的top $1$文档属于黄金证据文档，而在90%的情况下，前10篇(top $10$)检索到的文档中会出现黄金证据文档
## 4.5 Additional Results
**Generation Diversity**
![[RAG-Table 5.png]]
4.3节展示了RAG模型相较于BART模型在Jeopardy问题生成任务中是更具事实性和更具体的(more factual and specific)，遵循最近关于促进多样性解码(diversity-promoting decoding)的工作[33, 59, 39]，我们还通过计算不同模型生成的不同n-gram与总n-gram的比例(the ratio of dictinct ngrams to total ngrams)来研究生成多样性(generation diversity)，
Table 5显示，RAG-Sequence的生成结果比RAG-Token的更多样化，而且两者在不需要任何促进多样性解码(without needing any diversity-promoting decoding)的情况下，都比BART更多样化

**Retrieval Ablations**
RAG的一个关键特性是学习为任务检索相关的信息，为了评估检索机制的有效性，我们进行消融实验，在训练期间固定(freeze)检索器

![[RAG-Table 6.png]]
如Table 6所示，相较于固定检索器，学习到的检索器提高了所有任务的结果

我们将RAG的密集检索器(dense retriever)与基于词重叠的BM25检索器(word overlap-based BM25 retriever)[53]进行比较，我们将RAG的检索器替换为一个固定的(fixed)BM25系统，并在计算$p(z|x)$时将BM25检索得分用作为对率值(logits)，
Table 6显示了结果，对于FEVER，BM25的表现最佳，可能是因为FEVER的声明主要以实体为中心(heavily entity-centric)，因此非常适合基于词重叠的检索(word overlap-based retrieval)，而可微检索(differentiable retrieval)在所有其他任务上都提高了结果，特别是在开放域问答任务(Open-Domain QA)中，这一点至关重要

**Index hot-swapping**
像RAG这样的非参数记忆模型的一个优势是可以在测试时轻松更新知识(knowledge can be easily updated)，像T5或BART这样的仅参数模型(parametric-only)需要进一步训练才能随着世界的变化更新它们的行为

为了展示这一点，我们使用2016年12月的DrQA[5]维基百科转储构建了一个索引(build an index)，并比较了使用此索引的RAG输出与我们主要结果中使用较新索引(2018年12月)的输出，我们准备了一份在这些日期之间发生变化的82位世界领导人的名单，并使用模板“Who is {position}?”(例如“Who is the President of Peru”)来查询使用不同索引的NQ RAG模型，
使用2016年索引的RAG回答2016年世界领导人的正确率为70%，使用2018年索引回答2018年世界领导人的正确率为68%，索引时间和问题中的时间不匹配时，模型的准确率很低(使用2018年索引的RAG回答2016年领导人为12%，使用2016年索引的RAG回答2018年领导人为4%)，这表明我们可以通过简单地替换其非参数记忆来更新RAG的世界知识

**Effect of Retrieving more documents**
RAG模型使用5个或10个检索到的隐文档进行训练，我们没有观察到使用5个文档训练的模型和使用10个文档训练的模型之间的性能有显著差异

![[RAG-Fig3.png]]
在测试时，我们可以灵活调整需要检索的文档数量，文档数量的不同可能会影响性能和运行时间，Figure 3(左)显示在测试时检索更多文档可以单调地提高RAG-Sequence在开放域问答上的表现，但RAG-Token在检索到10个文档时性能达到峰值，Figure 3(右)显示检索更多文档会导致RAG-Token的Rouge-L值提高，但以牺牲Bleu-1值为代价，但对RAG-Sequence的影响不那么明显
# 5 Related Work
**Single-Task Retrieval**
先前的工作已经表明，当将检索任务单独考虑时(considered in isolation)，检索可以提高各种NLP任务的性能，这些任务包括开放域问答 open-domain QA[5, 29]、事实核查 fact checking[56]、事实补全 fact completion[48]、长篇问答 long-form QA[12]、维基百科文章生成 Wikipedia artical generation[36]、对话 dialogue[41, 65, 9, 13]、翻译 translation[17]和语言建模 language modeling[19, 27]
我们的工作统一了之前在将检索纳入单独任务中(incorporating retrieval into individual tasks)的成功经验，展示了一个单独的(single)基于检索的架构(retrieval-based architecture)能够在多个任务上实现强大的性能

**General-Purpose Architectures for NLP**
关于NLP任务的通用架构(general-purpose architecture)的先前工作在没有使用检索的情况下已经显示出巨大的成功，一个单一的、预训练的语言模型在微调后[49, 8]已被证明在GLUE benchmarks[60, 61]的各种分类任务上能够达到强大的性能，GPT-2[50]后来表明，一个单一的、从左到右的预训练语言模型可以在区分式(discriminative)和生成式(generative)任务上都达到强大的性能
为了进一步提高性能，BART[32]和T5[51, 52]提出了一个单一的、预训练的编码器-解码器模型，该模型利用双向注意力(bi-directional attention)来在区分式和生成式任务上实现更强的性能

我们的工作旨在通过学习一个检索模块(retrieval module)来增强预训练的生成式语言模型，从而扩展单一统一架构(single, unified architecture)的可能的任务空间(space of possible tasks)

**Learned Retrieval**
在信息检索领域，有大量关于学习检索文档(documents)的工作，最近更是有使用预训练神经语言模型进行检索文档的工作[44, 26]，一些工作优化(optimize)检索模块以帮助特定的下游任务，如使用搜索[46]、强化学习[6, 63, 62]，或者像我们工作一样的隐变量方法[31, 20]来执行QA任务，
这些工作成功运用了不同的基于检索的架构和优化技术，在单一任务上达到了强大的性能，而我们展示了单一的基于检索的架构(single retrieval-based architecture)可以被微调，以在多种任务上达到强大的性能

**Memory-based Architectures**
我们的文档索引(document index)可以被视作一个可以被神经网络注意到的(attend to)大型的外部记忆(large external memory)，类似于记忆网络(memory networks)[64, 65]
Concurrent work[14]学习为输入的中的每个实体检索一个训练好的嵌入(retrieve a trained embedding for each entity)，而不是像我们的工作中的检索原始文本(raw text)；其他的工作通过关注事实嵌入(attend over fact embedding)，提高了对话模型生成事实性文本的能力

我们的外部记忆的一个关键特性就是它是由原始文本构成的，而不是由分布式的表示(distributed representations)构成的，这使得我们的外部记忆是：i) 人类可读的，为我们的模型提供了一种可解释性形式 ii) 人类可写的，使我们能够通过编辑文档索引(deiting the document index)来动态更新模型的记忆

这种方法也已在知识密集型对话中使用，其中生成器直接条件于检索到的文本，尽管这些文本是通过TF-IDF获得的，而不是通过端到端学习到的检索模获得的[9]

**Retrieve-and-Edit approaches**
我们的方法和检索并编辑(retrieve-and-edit)风格的方法有类似之处，在retrieve-and-edit方法中，给定输入，需要为它检索一个相似的input-output pair，然后对其进行编辑以得到最后输出
该方法已经在包括了机器翻译[18, 22]以及语义解析 semantic parsing[21]等领域取得成功，我们的方法和其有一些不同，包括了我们的方法更少强调了对检索到的项进行轻度编辑，而是更多强调了从检索到的内容中的多个片段中聚合内容，同时我们的方法强调了对隐文档检索的学习，以及检索的是佐证文档而非相关的训练对(training pairs)
# 6 Discussion
在这项工作中，我们提出了具有参数和非参数记忆访问能力的混合生成模型，我们的RAG模型在开放域QA上取得了SOTA，
我们发现，人们更倾向于RAG的生成，而不是纯参数化的BART，人们发现RAG更具有事实性和具体性，我们对学习到的检索组件进行了深入的调查，验证了其有效性，并说明了如何将检索索引(retrieval index)热交换(hot-swapped)以更新模型，而无需任何重新训练
在未来的工作中，可以探索两个组件是否可以从零开始联合预训练，无论是使用与BART类似的去噪目标(denoising objective)还是其他目标
我们的工作开辟了关于参数和非参数记忆如何相互作用(interact)以及如何最有效地结合它们(combine them)的新研究方向，我们的工作具有应用于各种NLP任务的良好前景
# A Implementation Details
# B Human Evaluation
# C Training setup Details
我们使用Fairseq[45]训练所有的RAG模型和BART基线模型，我们使用混合精度算数(mixed precision floating point arithmetic)进行训练[40]，训练分布于8个32GB NVIDIA V100 GPUs
我们发现在CPU上用FAISS进行MIPS已经足够快，因此我们将文档索引向量(document index vector)存储于CPU，这需要大约100GB的CPU内存以存储全部维基百科的文档索引向量
# D Further Details on Open-Domain QA
# E Further Details on FEVER
# F Null Document Probabilities
# G Parameters
我们的RAG模型包含了DPR中BERT-base查询编码器和文档编码器的可训练参数，每个编码器有1.1亿参数(尽管我们自己没有训练文档编码器)，以及来自BART-large的4.06亿可训练参数，总计有6.26亿可训练参数
表现最好的“闭卷”(仅含参数的)开放域问答模型是T5-11B，它拥有110亿可训练参数，与我们的模型参数数量最接近的T5模型是T5-large(7.7亿参数)，在Natural Questions上达到了28.9的EM分数[52]，远低于RAG-Sequence达到的44.5，这表明混合参数/非参数模型(hybrid parametric/non-parametric models)需要少得多的可训练参数便可在开放域QA任务上达到优秀的性能
非参数记忆索引(non-parametric memory index)不包含可训练参数，但包含2100万个728维的向量，总共包含153亿个值，这些可以轻松地以8位浮点精度存储于磁盘中
# H Retrieval Collapse
在初步实验中，我们观察到对于一些任务，如故事生成[11]，检索组件会“崩溃 collapse”并学会不管输入是什么都检索相同的文档，在这些情况下，一旦检索崩溃，生成器就会学会忽略这些文档，RAG模型的表现就会等同于BART
这种崩溃可能是由于某些任务中对事实知识的要求不那么明确(less-explicit requirement)，或者是因为更长的目标序列，这可能导致检索器的梯度信息较少
Perez等人[46]在优化检索组件以提高下游任务性能时，也发现了异常的检索结果
# I Number of instances per dataset
