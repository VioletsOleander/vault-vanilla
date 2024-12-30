# Abstract 
We describe latent Dirichlet allocation (LDA), a generative probabilistic model for collections of discrete data such as text corpora. LDA is a three-level hierarchical Bayesian model, in which each item of a collection is modeled as a finite mixture over an underlying set of topics. Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities. In the context of text modeling, the topic probabilities provide an explicit representation of a document. We present efficient approximate inference techniques based on variational methods and an EM algorithm for empirical Bayes parameter estimation. We report results in document modeling, text classification, and collaborative filtering, comparing to a mixture of unigrams model and the probabilistic LSI model. 
>  本文描述 LDA，一个生成式概率模型，用于建模离散数据集，例如文本语料库
>  LDA 是三层的层次贝叶斯模型
>  该模型中，集合中的每一项都被建模为潜在的一组主题的有限混合，每个主题建模为潜在主题概率的无限混合，在文本建模中，主题概率提供了文档的显式表征
>  我们介绍基于变分方法和 EM 算法的经验贝叶斯参数估计的高效近似推理技术
>  我们报告模型在文档建模、文本分类、协同过滤方面的结果，并与一元模型和概率 LSI 模型的混合模型进行了比较

# 1. Introduction 
In this paper we consider the problem of modeling text corpora and other collections of discrete data. The goal is to find short descriptions of the members of a collection that enable efficient processing of large collections while preserving the essential statistical relationships that are useful for basic tasks such as classification, novelty detection, summarization, and similarity and relevance judgments. 
>  本文考虑建模文本语料库和其他离散数据集合的问题
>  目标是找到对集合成员的简短描述，从而高效处理大规模数据集，同时保留基本任务 (如分类、新颖性检测、摘要生成、相似性和相关性判断) 

Significant progress has been made on this problem by researchers in the field of information retrieval (IR) (Baeza-Yates and Ribeiro-Neto, 1999). The basic methodology proposed by IR researchers for text corpora—a methodology successfully deployed in modern Internet search engines—reduces each document in the corpus to a vector of real numbers, each of which represents ratios of counts. In the popular tf-idf scheme (Salton and McGill, 1983), a basic vocabulary of “words” or “terms” is chosen, and, for each document in the corpus, a count is formed of the number of occurrences of each word. After suitable normalization, this term frequency count is compared to an inverse document frequency count, which measures the number of occurrences of a word in the entire corpus (generally on a log scale, and again suitably normalized). The end result is a term-by-document matrix $X$ whose columns contain the tf-idf values for each of the documents in the corpus. Thus the tf-idf scheme reduces documents of arbitrary length to fixed-length lists of numbers. 
>  信息检索研究者为文本语料库提出的基础方法——已经应用在现代搜索引擎中的方法——将语料库中的每个文档简化为一个实数向量，向量中每个实数成分表示一个计数比率
>  流行的 tf-idf 方案 (Salton and McGill 1983) 选择一个基础词袋，然后为语料库中的每个文档计数每个词出现的次数，经过适当的归一化后 (除以文档总次数)，将词频与逆文档频率计数进行比较
>  逆文档频率计数度量了整个语料库中某个词出现的次数 (通常按照对数尺度，并再次进行适当归一化) ($\log \frac {\text{语料库中的文档总数}} {\text{包含词}t\text{的文档数}}$，衡量一个词在语料库中的普遍程度)
>  (tf-idf 的核心思想是：如果一个词在某篇文档中出现的频率越高，并且在整个语料库中出现该词的文档数量越少，则该词对于这篇文档的重要性越高)
>  最后的结果是一个词-文档矩阵 $X$，其列包含了语料库中每个文档的所有 tf-idf 值
>  因此，tf-idf 方法将任意长度的文档简化为固定长度的数值列表 (一个向量表示)

While the tf-idf reduction has some appealing features—notably in its basic identification of sets of words that are discriminative for documents in the collection—the approach also provides a relatively small amount of reduction in description length and reveals little in the way of inter- or intra-document statistical structure. To address these shortcomings, IR researchers have proposed several other dimensionality reduction techniques, most notably latent semantic indexing (LSI) (Deerwester et al., 1990). LSI uses a singular value decomposition of the $X$ matrix to identify a linear subspace in the space of tf-idf features that captures most of the variance in the collection. This approach can achieve significant compression in large collections. Furthermore, Deerwester et al. argue that the derived features of LSI, which are linear combinations of the original tf-idf features, can capture some aspects of basic linguistic notions such as synonymy and polysemy. 
>  tf-idf 可以识别出文档在数据集中具有区分性的一组单词，其劣势是它对于描述长度的减少程度有限，同时对文档内或文档间的统计结构揭示较少
>  为此，信息检索的研究员提出了其他几种降维技术，最著名的是潜在语义索引 (LSI, Deerwester et al., 1990)
>  LSI 使用矩阵 $X$ 的奇异值分解来识别 tf-idf 特征空间中的一个线性子空间，该子空间捕获了数据集中大部分的方差，该方法可以在大型集合上实现显著的压缩
>  Deerwester et al. 认为 LSI 导出的特征 (原始 tf-idf 特征的线性组合) 可以捕获一些基本的语言概念，例如同义词和多义词 

To substantiate the claims regarding LSI, and to study its relative strengths and weaknesses, it is useful to develop a generative probabilistic model of text corpora and to study the ability of LSI to recover aspects of the generative model from data (Papadimitriou et al., 1998). Given a generative model of text, however, it is not clear why one should adopt the LSI methodology—one can attempt to proceed more directly, fitting the model to data using maximum likelihood or Bayesian methods. 
>  为了验证关于 LSI 的主张，并研究其优劣，(Papadimitriou el al., 1988) 开发了一个生成式的概率文本语料模型，来研究 LSI 从数据中恢复生成模型的各方面的能力
>  但文本生成模型给定后，尚不明确是否需要使用 LSI 方法，我们可以直接用更直接的方法，例如使用极大似然或贝叶斯方法将模型拟合到数据上 (学习模型参数)

A significant step forward in this regard was made by Hofmann (1999), who presented the probabilistic $L S I\left(p L S I\right)$ model, also known as the aspect model , as an alternative to LSI. The pLSI approach, which we describe in detail in Section 4.3, models each word in a document as a sample from a mixture model, where the mixture components are multinomial random variables that can be viewed as representations of “topics.” Thus each word is generated from a single topic, and different words in a document may be generated from different topics. Each document is represented as a list of mixing proportions for these mixture components and thereby reduced to a probability distribution on a fixed set of topics. This distribution is the “reduced description” associated with the document. 
>  Hofmann (1999) 在这里向前迈进了一大步，它提出了概率 LSI 模型 (pLSI)，也称为方面模型，来替代 LSI
>  pLSI 方法将文档中的每个词视作来自于一个混合模型中的一个样本，而混合模型的混合成分是多项式随机变量 (服从多项式分布的随机变量)，这些变量可以视作“主题”的表示
>  因此，每个单词都生成自一个主题，而文档中的不同词可能来自于不同的主题 (也就是将多个主题建模为了隐变量)
>  每个文档被表示为这些混合成分的混合比例列表，因此文档就被简化为了固定主题集合上的概率分布 (隐变量集合上的分布)，该分布就是文档相关的 “简化描述”

While Hofmann’s work is a useful step toward probabilistic modeling of text, it is incomplete in that it provides no probabilistic model at the level of documents. In pLSI, each document is represented as a list of numbers (the mixing proportions for topics), and there is no generative probabilistic model for these numbers. This leads to several problems: (1) the number of parameters in the model grows linearly with the size of the corpus, which leads to serious problems with overfitting, and (2) it is not clear how to assign probability to a document outside of the training set. 
>  Hofmann 的工作是对文本概率建模的一步前进，但其问题是没有在文档层面提供概率模型 (仅建模了词层面的概率模型)
>  pLSI 中，每个文档都表示为一组数字 (主题的混合比例)，但没有建模对这组数字的生成模型，这导致了几个问题：
>  1. 模型的参数随着语料库的大小线性增长，容易导致严重的过拟合问题
>  2. 不清楚如何为训练集外的文档分配概率

To see how to proceed beyond pLSI, let us consider the fundamental probabilistic assumptions underlying the class of dimensionality reduction methods that includes LSI and pLSI. All of these methods are based on the “bag-of-words” assumption—that the order of words in a document can be neglected. In the language of probability theory, this is an assumption of exchange ability for the words in a document (Aldous, 1985). Moreover, although less often stated formally, these methods also assume that documents are exchangeable; the specific ordering of the documents in a corpus can also be neglected. 
>  要考虑如何超越 pLSI，我们考虑包含 LSI 和 pLSI 方法在内的这类降维方法的基本概率假设
>  所有的这些方法都基于 “词袋” 假设——文档中词的顺序可以被忽略
>  用概率论的语言来说，这是假设了文档中词的可交换性
>  另外，这些方法还假设了文档是可交换的，即语料库中文档的具体排序也可以忽略不计

A classic representation theorem due to de Finetti (1990) establishes that any collection of exchangeable random variables has a representation as a mixture distribution—in general an infinite mixture. Thus, if we wish to consider exchangeable representations for documents and words, we need to consider mixture models that capture the exchange ability of both words and documents. 
>  de Finetti 的经典表示理论指出，任意的一组可交换随机变量都可以表示为一个混合分布——通常是无限的混合分布
>  因此，如果我们需要考虑文档和词的可交换表示，我们就需要考虑能够同时捕获单词和文档的混合模型 
>  (词和文档都视作随机变量，且它们可交换，这样的一组可交换随机变量应该可以被一个混合分布表示)

This line of thinking leads to the latent Dirichlet allocation (LDA) model that we present in the current paper. 
>  这一思路引出了本文介绍的 LDA 模型

It is important to emphasize that an assumption of exchange ability is not equivalent to an assumption that the random variables are independent and identically distributed. Rather, exchangeability essentially can be interpreted as meaning “ conditionally independent and identically distributed,” where the conditioning is with respect to an underlying latent parameter of a probability distribution. Conditionally, the joint distribution of the random variables is simple and factored while marginally over the latent parameter, the joint distribution can be quite complex. Thus, while an assumption of exchange ability is clearly a major simplifying assumption in the domain of text modeling, and its principal justification is that it leads to methods that are computationally efficient, the exchange ability assumptions do not necessarily lead to methods that are restricted to simple frequency counts or linear operations. We aim to demonstrate in the current paper that, by taking the de Finetti theorem seriously, we can capture significant intra-document statistical structure via the mixing distribution. 
>  注意，可交换性假设不等价于假设随机变量是独立同分布的
>  可交换性本质上可以被解释为 “条件独立且同分布”，其中的条件指概率分布的潜在参数
>  在给定条件下，随机变量的联合分布可以分解，而在潜在参数的边际分布下，联合分布则十分复杂
>  因此，虽然可交换性假设在文本建模领域显然是一个简化的假设，但该假设可以导出计算效率高的方法
>  同时，可交换性交涉并不一定限制方法为简单的频率计数说线性操作
>  本文中，我们旨在证明，通过认真考虑 de Finetti 定理，我们可以通过混合分布捕获显著的文档内统计结构

It is also worth noting that there are a large number of generalizations of the basic notion of exchange ability, including various forms of partial exchange ability, and that representation theorems are available for these cases as well (Diaconis, 1988). Thus, while the work that we discuss in the current paper focuses on simple “bag-of-words” models, which lead to mixture distributions for single words (unigrams), our methods are also applicable to richer models that involve mixtures for larger structural units such as $n$ -grams or paragraphs. 
> 可交换性的概念也有许多推广形式，包括各种形式的部分可交换性，这些情况也有对应的表示定理
> 本文主要讨论简单的 “词袋” 模型，为单个词 (unigram) 构建了混合分布，但本文的方法也可以应用于涉及更大的结构单元的模型，例如 n-gram 或段落，的混合

The paper is organized as follows. In Section 2 we introduce basic notation and terminology. The LDA model is presented in Section 3 and is compared to related latent variable models in Section 4. We discuss inference and parameter estimation for LDA in Section 5. An illustrative example of fitting LDA to data is provided in Section 6. Empirical results in text modeling, text class i cation and collaborative filtering are presented in Section 7. Finally, Section 8 presents our conclusions. 

# 2. Notation and terminology 
We use the language of text collections throughout the paper, referring to entities such as “words,” “documents,” and “corpora.” This is useful in that it helps to guide intuition, particularly when we introduce latent variables which aim to capture abstract notions such as topics. It is important to note, however, that the LDA model is not necessarily tied to text, and has applications to other problems involving collections of data, including data from domains such as collaborative filtering, content-based image retrieval and bioinformatics. Indeed, in Section 7.3, we present experimental results in the collaborative filtering domain. 
>  LDA 模型并不一定局限于文本，可以应用于其他涉及数据集合的问题，包括来自于协同过滤、基于内容的图像检索和生物信息学等领域的问题

Formally, we define the following terms:
- A *word* is the basic unit of discrete data, defined to be an item from a vocabulary indexed by $\{1,\ldots,V\}$ . We represent words using unit-basis vectors that have a single component equal to one and all other components equal to zero. Thus, using superscripts to denote components, the v-th word in the vocabulary is represented by a $V$ -vector $w$ such that $w^{\nu}=1$ and ${w}^{u}=0$ for $u\ne\nu$ . 
- A *document* is a sequence of $N$ words denoted by $\mathbf{w}=\left(w_{1},w_{2},\ldots,w_{N}\right)$ , where $w_{n}$ is the n-th word in the sequence. 
- A corpus is a collection of $M$ documents denoted by $\pmb D=\left\{\mathbf{w}_{1},\mathbf{w}_{2},\ldots,\mathbf{w}_{M}\right\}$ .

>  定义以下术语
>  - 词 (word) 是离散数据的基本单位
>  它定义为词表中的一个项，词表由 $\{1,\dots, V\}$ 索引，我们用单位基向量表示单词，单位基向量仅有一个成分为 1，其他成分为 0 (one-hot 编码)
>  用上标表示成分，词表中的第 $v$ 个词用一个 $V$ 元向量 $w$ 表示，满足 $w^v = 1$，以及 $w^u = 0\ \text{for}\ u\ne v$
>  - 文档 (document) 为 $N$ 个词构成的序列，记作 $\mathbf w = (w_1, w_2,\dots, w_n)$，其中 $w_n$ 表示序列中的第 $n$ 个词
>  - 语料库是一组 $M$ 个文档的集合，记作 $\pmb D = \{\mathbf w_1, \mathbf w_2,\dots, \mathbf w_M\}$

We wish to find a probabilistic model of a corpus that not only assigns high probability to members of the corpus, but also assigns high probability to other “similar” documents. 
>  我们的目标是找到一个语料库的概率模型，该模型不仅能为语料库中的成员赋予较高的概率，也可以为其他相似的文档赋予较高概率

# 3. Latent Dirichlet allocation 
Latent Dirichlet allocation (LDA) is a generative probabilistic model of a corpus. The basic idea is that documents are represented as random mixtures over latent topics, where each topic is characterized by a distribution over words. 
>  LDA 是一个语料库的生成式概率模型，其基本思想是将文档表示为潜在主题的随机混合，其中每个主题用一个词分布刻画

LDA assumes the following generative process for each document $\mathbf w$ in a corpus $\pmb D$ : 
1. Choose $N\sim{Poisson}(\xi)$ . 
2. Choose $\theta \sim Dir(\alpha)$ . 
3. For each of the $N$ words $w_{n}$ : 
    (a) Choose a topic $z_{n}\sim Mult nomial (\theta)$ .
    (b) Choose a word $w_{n}$ from $p(w_{n}\,|\,z_{n},\beta)$ , a multinomial probability conditioned on the topic $z_{n}$ . 

>  LDA 为语料库 $\pmb D$ 中的每个文档 $\mathbf w$ 假设了以下的生成过程
>  1. 从参数为 $\xi$ 的 Poisson 分布中选择 $N$ (文档的词数)
>  2. 从参数为 $\alpha$ 的 Dirichlet 分布中选择 $\theta$
>  3. 对于每个词 $w_n$ ($1\le n \le N$): 
>      从参数为 $\theta$ 的多项式分布中选择主题 $z_n$
>      从条件于主题 $z_n$ 的多项式分布 $p(w_n\mid z_n, \beta)$ 中选择单词 $w_n$

Several simplifying assumptions are made in this basic model, some of which we remove in subsequent sections. First, the dimensionality $k$ of the Dirichlet distribution (and thus the dimensionality of the topic variable $z$ ) is assumed known and fixed. Second, the word probabilities are parameterized by a $k\times V$ matrix $\beta$ where $\beta_{i j}=p(w^{j}=1\,|\,z^{i}=1)$ , which for now we treat as a fixed quantity that is to be estimated. Finally, the Poisson assumption is not critical to anything that follows and more realistic document length distributions can be used as needed. Furthermore, note that $N$ is independent of all the other data generating variables ( $\theta$ and $z$ ). It is thus an ancillary variable and we will generally ignore its randomness in the subsequent development. 
>  该基本模型做了几个简化假设
>  其一，Dirichlet 分布的维度 $k$ (也是主题变量 $z$ 的维度) 假设是已知且固定的
>  其二，单词概率由一个 $k\times V$ 的矩阵 $\beta$ 参数化，满足 $\beta_{ij} = p(w^j = 1 \mid z^i=1)$ 
>  (主题 $z_i$ 下单词 $w_j$ 出现的概率)，我们将 $\beta$ 视作一个需要估计的固定量
>  其三，Poisson 假设对于后续内容并不是关键的，我们可以按需选择更真实的文档长度分布
>  另外，注意 $N$ 独立于所有其他数据生成变量 $\theta$ 和 $z$，因此它是一个辅助变量，在后续推导中我们一般会忽略其随机性

A $k$ -dimensional Dirichlet random variable $\theta$ can take values in the $(k-1)$ -simplex (a $k$ -vector $\theta$ lies in the $(k-1)$ -simplex if $\theta_{i}\geq0$ , $\textstyle\sum_{i=1}^{k}\theta_{i}=1)$ , and has the following probability density on this simplex: 

$$
p(\theta\!\mid\!\alpha)=\frac{\Gamma\left(\sum_{i=1}^{k}\alpha_{i}\right)}{\prod_{i=1}^{k}\Gamma(\alpha_{i})}\theta_{1}^{\alpha_{1}-1}\cdot\cdot\cdot\theta_{k}^{\alpha_{k}-1},\tag{1}
$$ 
where the parameter $\alpha$ is a $k$ -vector with components $\alpha_{i}>0$ , and where $\Gamma(x)$ is the Gamma function.

>  $k$ 维向量 $\theta$ 满足 $\theta_i \ge 0, \sum_{i=1}^k \theta_i = 1$，因此它实际上在 $(k-1)$ 维单纯形上取值
>  同时 $\theta$ 满足 Dirichlet 分布，故单纯形上的概率密度形式如上
>  其中参数 $\alpha$ 是一个 $k$ 维向量，满足 $\alpha_i > 0$
>  其中 $\Gamma(x)$ 为 Gamma 函数

The Dirichlet is a convenient distribution on the simplex — it is in the exponential family, has finite dimensional sufficient statistics, and is conjugate to the multinomial distribution. In Section 5, these properties will facilitate the development of inference and parameter estimation algorithms for LDA. 
>  Dirichlet 是一个在单纯形上非常方便的分布——它属于指数族，有有限维度的充分统计量，并且和多项式分布共轭

Given the parameters $\alpha$ and $\beta$ , the joint distribution of a topic mixture $\theta$ , a set of $N$ topics $\mathbf{z}$ , and a set of $N$ words $\mathbf{w}$ is given by: 

$$
p({\theta},\mathbf{z},\mathbf{w}\,|\,{\alpha},\beta)=p({\theta}\,|\,{\alpha})\prod_{n=1}^{N}p(z_{n}\,|\,{\theta})p(w_{n}\,|\,z_{n},\beta),\tag{2}
$$ 
where $p(z_{n}\,|\,\theta)$ is simply $\theta_{i}$ for the unique $i$ such that $z_{n}^{i}=1$ . 

>  给定参数 $\alpha, \beta$，主题混合参数 $\theta$，一组 $N$ 个主题 $\mathbf z$ 和一组 $N$ 个单词 $\mathbf w$ 的联合分布如上所示
>  其中概率 $p(z_n \mid \theta)$ 的值在 $z_n^i = 1$ 时就为 $\theta_i$ 
>  (也就是主题选择为第 $i$ 个主题的概率是 $\theta_i$，因为我们已经假设了主题服从参数为 $\theta$ 的多项式分布)

>  Equation (2) 的推导 (根据图模型分解即可)

$$
\begin{align}
&p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)\\
=&p(\theta\mid \alpha, \beta)p(\mathbf w, \mathbf z\mid \alpha, \beta, \theta)\\
=&p(\theta\mid\alpha)\prod_{n=1}^Np(w_n, z_n\mid \beta, \theta)\\
=&p(\theta\mid \alpha)\prod_{n=1}^Np(z_n\mid \theta)p(w_n\mid z_n,\beta)
\end{align}
$$

>  推导完毕

Integrating over $\theta$ and summing over $\mathbf z$ , we obtain the marginal distribution of a document: 

$$
p(\mathbf{w}\,|\,\alpha,\beta)=\int p(\theta\,|\,\alpha)\left(\prod_{n=1}^{N}\sum_{z_{n}}p(z_{n}\,|\,\theta)p(w_{n}\,|\,z_{n},\beta)\right)d\theta.\tag{3}
$$ 
>  在 $\theta$ 上积分 ($\theta$ 为连续随机变量) 并且在 $\mathbf z$  ($\mathbf z$ 为离散随机变量，只有有限的可能值) 上求和，我们得到了一个文档 (大小为 $N$ 的单词集合) 的边际分布如上

>  Equation (3) 推导如下

$$
\begin{align}
p(\mathbf w\mid \alpha, \beta) &= \sum_{\mathbf z}p(\mathbf z, \mathbf w\mid \alpha, \beta)\\
&=\int \left(\sum_{\mathbf z}p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)\right)d\theta\\
&=\int p(\theta\mid\alpha)\left(\sum_{\mathbf z}p(\mathbf z, \mathbf w\mid \alpha, \beta,\theta)\right)d\theta\\
&=\int p(\theta\mid\alpha)\left(\sum_{z_1}\cdots\sum_{z_N}p(\mathbf z, \mathbf w\mid \alpha, \beta,\theta)\right)d\theta\\
&=\int p(\theta\mid\alpha)\left(\sum_{z_1}\cdots\sum_{z_N}\prod_{n=1}^Np(z_n\mid \theta)p(w_n\mid z_n,\theta)\right)d\theta\\
&=\int p(\theta\mid\alpha)\left(\prod_{n=1}^N\sum_{z_n}p(z_n\mid \theta)p(w_n\mid z_n,\theta)\right)d\theta\\
\end{align}
$$

>  推导完毕

Finally, taking the product of the marginal probabilities of single documents, we obtain the probability of a corpus: 

$$
p(\pmb D\mid\alpha,\beta)=\prod_{d=1}^{M}\int p(\theta_{d}\,|\,\alpha)\left(\prod_{n=1}^{N_{d}}\sum_{z_{d n}}p(z_{d n}\,|\,\theta_{d})p(w_{d n}\,|\,z_{d n},\beta)\right)d\theta_{d}.
$$ 
>  将语料库中所有 $M$ 个文档的边际分布相乘，我们得到语料库的分布

The LDA model is represented as a probabilistic graphical model in Figure 1. As the figure makes clear, there are three levels to the LDA representation. The parameters $\alpha$ and $\beta$ are corpus-level parameters, assumed to be sampled once in the process of generating a corpus. The variables $\theta_{d}$ are document-level variables, sampled once per document. Finally, the variables $z_{d n}$ and $w_{d n}$ are word-level variables and are sampled once for each word in each document. 
>  LDA 模型可以表示为 Figure 1 中的概率图模型
>  可以看到 LDA 表示有三个层次
>  参数 $\alpha, \beta$ 为语料库级别参数 (超参数)，在生成语料库的过程中，假设仅采样一次
>  变量 $\theta_d$ 为文档级别变量，每个文档仅采样一次 (文档的主题混合)
>  变量 $z_{dn}$ 和 $w_{dn}$ 为词级别变量，为每个文档内的每个词采样一次 (词的来源主题以及具体哪个词)

![[pics/LDA-Figure1.png]]

It is important to distinguish LDA from a simple Dirichlet-multinomial clustering model. A classical clustering model would involve a two-level model in which a Dirichlet is sampled once for a corpus, a multinomial clustering variable is selected once for each document in the corpus, and a set of words are selected for the document conditional on the cluster variable. As with many clustering models, such a model restricts a document to being associated with a single topic. LDA, on the other hand, involves three levels, and notably the topic node is sampled repeatedly within the document. Under this model, documents can be associated with multiple topics. 
>  LDA 和 Dirichlet-多项式聚类模型不同，典型的聚类模型是一个二层模型，在语料库级别进行一次 Dirichlet 采样，在文档级别进行一次多项式采样 (采样簇变量)，文档内的词条件于该簇变量选择
>  这类模型限制文档只能关联到一个主题，而 LDA 是一个三层模型，其中主题节点在文档内部反复采样，因此文档可以关联到多个主题

Structures similar to that shown in Figure 1 are often studied in Bayesian statistical modeling, where they are referred to as hierarchical models (Gelman et al., 1995), or more precisely as conditionally independent hierarchical models (Kass and Steffey, 1989). Such models are also often referred to as parametric empirical Bayes models , a term that refers not only to a particular model structure, but also to the methods used for estimating parameters in the model (Morris, 1983). Indeed, as we discuss in Section 5, we adopt the empirical Bayes approach to estimating parameters such as $\alpha$ and $\beta$ in simple implementations of LDA, but we also consider fuller Bayesian approaches as well. 
>  类似于 Figure 1 的结构在贝叶斯统计建模中常被研究，这类模型称为层次模型，更精确地说是条件独立层次模型
>  这类模型通常也称为参数经验贝叶斯模型，这个术语不仅指出了模型结构，还指出了用于估计模型参数的方法
>  我们也使用经验贝叶斯方法来估计 LDA 的参数，同时也考虑了更全面的贝叶斯方法

## 3.1 LDA and exchangeability 
A finite set of random variables $\left\{z_{1},\dots,z_{N}\right\}$ is said to be exchangeable if the joint distribution is invariant to permutation. If $\pi$ is a permutation of the integers from $1$ to $N$ : 

$$
\begin{array}{r}{p\big(z_{1},\dots,z_{N}\big)=p\big(z_{\pi(1)},\dots,z_{\pi(N)}\big).}\end{array}
$$ 
An infinite sequence of random variables is infinitely exchangeable if every finite sub sequence is exchangeable. 

>  一组随机变量 $\{z_1, \dots, z_N\}$ 如果满足其联合分布对于置换是不变的，则称它们是可交换的，也就是说，如果 $\pi$ 是从 $1$ 到 $N$ 的整数上的一个置换，则上式要满足
>  如果一个无限的随机变量序列的任意子序列都是可交换的，则该序列是无穷可交换的

De Finetti’s representation theorem states that the joint distribution of an infinitely exchangeable sequence of random variables is as if a random parameter were drawn from some distribution and then the random variables in question were independent and identically distributed , conditioned on that parameter. 
>  De Finetti 的表示定理指出，一个无穷可交换的随机变量序列的联合分布可以看作先从某个分布抽取一个随机参数，然后再给定该参数的条件下，序列中的随机变量都是独立同分布的

In LDA, we assume that words are generated by topics (by fixed conditional distributions) and that those topics are infinitely exchangeable within a document. By de Finetti’s theorem, the probability of a sequence of words and topics must therefore have the form: 

$$
p(\mathbf{w},\mathbf{z})=\int p({\theta})\left(\prod_{n=1}^{N}p(z_{n}\,|\,{\theta})p(w_{n}\,|\,z_{n})\right)d{\theta},
$$

where $\theta$ is the random parameter of a multinomial over topics. We obtain the LDA distribution on documents in Eq. (3) by marginalizing out the topic variables and endowing $\theta$ with a Dirichlet distribution. 

>  LDA 中，我们假设单词由主题生成 (通过固定的条件分布)，并且同一文档内的主题是无穷可交换的
>  那么，根据 de Finetti 的表示定理，一个单词序列和其主题的概率的形式如上，其中 $\theta$ 是主题服从的多项式分布的随机参数
>  LDA 中，我们令 $\theta$ 服从 Dirichlet 分布，并通过 Eq (3) 边际化掉主题，得到一个文档的边际分布

## 3.2 A continuous mixture of unigrams 
The LDA model shown in Figure 1 is somewhat more elaborate than the two-level models often studied in the classical hierarchical Bayesian literature. By marginalizing over the hidden topic variable $z$ , however, we can understand LDA as a two-level model. 
>  Figure 1 中的 LDA 模型比经典的二级层次贝叶斯模型更为复杂
>  但通过边际化消除隐主题变量 $z$，我们可以将 LDA 理解为一个二层模型

In particular, let us form the word distribution $p(w\,|\,\theta,\beta)$ : 

$$
p(w\,|\,\theta,\beta)=\sum_{z}p(w\,|\,z,\beta)p(z\,|\,\theta).
$$

>  我们将单词的分布 $p(w\mid \theta, \beta)$ 在 $w$ 上边际化，重写为以上形式

>  推导

$$
\begin{align}
p(w\mid \theta, \beta) &= \sum_z p(w, z\mid \theta, \beta)\\
&=\sum_z p(z\mid \theta) p(w\mid z,\beta)
\end{align}
$$

>  推导完毕

Note that this is a random quantity since it depends on $\theta$ . 
>  注意该边际分布实际上是一个随机变量，因为它依赖于 $\theta$

We now define the following generative process for a document $\mathbf w$: 
1. Choose $\theta\sim\operatorname{Dir}(\alpha)$ . 
2. For each of the $N$ words $w_{n}$ :
    (a) Choose a word $w_{n}$ from $p(w_{n}\,|\,\theta,\beta)$ . 

>  我们为文档 $\mathbf w$ 定义如下的生成过程
>  1. 选择参数 $\theta \sim Dirchilet(\alpha)$
>  2. 对于文档中的每个单词 $w_n$:
>      从边际分布 $p(w_n\mid \theta, \beta)$ 中选择单词

This process defines the marginal distribution of a document as a continuous mixture distribution: 

$$
p(\mathbf{w}\,|\,{\alpha},\beta)=\int p({\theta}\,|\,{\alpha})\left(\prod_{n=1}^{N}p(w_{n}\,|\,{\theta},\beta)\right)d{\theta},
$$ 
where $p(w_{n}\,|\,\theta,\beta)$ are the mixture components and $p(\theta\,|\,\alpha)$ are the mixture weights.

>  该过程将文档的边际分布定义为一个连续的混合分布如上 (在 $\theta$ 上积分)
>  其中，$p(w_n\mid \theta, \beta)$ 为混合成分，$p(\theta\mid \alpha)$ 为混合权重 
>  (每个可能的 $\theta$ 取值提供一个混合成分)

>  推导

$$
\begin{align}
p(\mathbf w\mid \alpha, \beta) &= \int p(\mathbf w, \theta\mid \alpha, \beta)d\theta\\
&=\int p(\theta\mid \alpha,\beta) p(\mathbf w\mid \theta, \alpha, \beta)d\theta\\
&=\int p(\theta\mid\alpha)p(\mathbf w\mid \theta, \beta)d\theta\\
&=\int p(\theta\mid \alpha) \left(\prod_{n=1}^N p(w_n\mid \theta,\beta)\right)d\theta
\end{align}
$$

>  推导完毕

Figure 2 illustrates this interpretation of LDA. It depicts the distribution on $p(w\mid \theta, \beta)$ which is induced from a particular instance of an LDA model. Note that this distribution on the $(V-1)$ - simplex is attained with only $k + kV$ parameters yet exhibits very interesting multimodal structure.
>  Figure 2 展示了特定参数下的 LDA 模型中，单词分布 $p(w\mid \theta, \beta)$ 上的分布
>  这一 $(V-1)$ 维单纯形上的分布只需要 $k+kV$ 个参数表示，但展示了非常有趣的多模态结构
>  ($V$ 为词袋大小，$k$ 为主题数量，每个主题需要 $V$ 个参数描述给定该主题下的词分布，同时每个主题需要一个权重参数，因此参数数量为 $k+kV$)

![[pics/LDA-Figure2.png]]

>  该图展示了 4 个主题和 3 个词的情况
>  其中 $x-y$ 平面上的每一个点都是一个 $(V-1)$ 维单纯形，描述了一个词分布
>  四个红色的叉对应的点是在给定主题下，词的多项式分布
>  LDA 模型中，词的边际分布 $p(w\mid \theta, \beta)$ 是一个随机变量，它是多个给定主题的词分布的加权混合，权重为依赖于 $\theta$ 的分布 $p(z\mid \theta)$，该分布也是一个随机变量
>  显然，四个红色的叉也对应了分布 $p(z\mid \theta)$ 中特定主题的概率密度为 1 的情况

# 4. Relationship with other latent variable models 
In this section we compare LDA to simpler latent variable models for text—the unigram model, a mixture of unigrams, and the pLSI model. Furthermore, we present a unified geometric interpretation of these models which highlights their key differences and similarities. 
>  本节中，我们将 LDA 与更简单的文本隐变量模型——一元模型、混合一元模型、pLSI 模型——进行比较
>  我们为这些模型提出同一的几何解释，突出它们的关键差异和相似之处

## 4.1 Unigram model 
Under the unigram model, the words of every document are drawn independently from a single multinomial distribution: 

$$
p(\mathbf{w})=\prod_{n=1}^{N}p(w_{n}).
$$ 
>  一元模型中，每个文档的词都从一个多项式分布中独立抽取
>  因此，文档的概率就等于所有词的概率的乘积，如上所示

This is illustrated in the graphical model in Figure 3a. 
>  如 Figure 3a 中，一个文档有 $N$ 个相互独立的词

![[pics/LDA-Figure3.png]]

## 4.2 Mixture of unigrams 
If we augment the unigram model with a discrete random topic variable $z$ (Figure 3b), we obtain a mixture of unigrams model (Nigam et al., 2000). Under this mixture model, each document is generated by first choosing a topic $z$ and then generating $N$ words independently from the conditional multinomial $p(w\,|\,z)$ . The probability of a document is: 

$$
p(\mathbf{w})=\sum_{z}p(z)\prod_{n=1}^{N}p(w_{n}\,|\,z).
$$ 
>  我们为一元模型引入离散的随机主题变量 $z$，就得到了混合一元模型
>  (Figure 3b 中，每个文档关联了一个主题 $z$)
>  混合模型中，每个文档的生成过程包括两步：首先选择一个主题 $z$，然后从条件多项式分布 $p(w\mid z)$ 中独立生成 $N$ 个词
>  此时文档的边际概率写为以上形式，它需要在所有可能的主题上求和

When estimated from a corpus, the word distributions can be viewed as representations of topics under the assumption that each document exhibits exactly one topic. As the empirical results in Section 7 illustrate, this assumption is often too limiting to effectively model a large collection of documents. 
>  这些条件词分布可以从语料库估计，在每个文档仅涉及一个主题的假设下，该条件词分布就可以视作主题的表示
>  Section 7 的经验结果表示，该假设的限制性太高，无法有效建模大量文档

In contrast, the LDA model allows documents to exhibit multiple topics to different degrees. This is achieved at a cost of just one additional parameter: there are $k-1$ parameters associated with $p(z)$ in the mixture of unigrams, versus the $k$ parameters associated with $p(\theta\,|\,\alpha)$ in LDA. 
>  而 LDA 模型允许文档在不同程度上展示多个主题
>  为此，我们需要添加一个额外参数：混合一元模型中，$p(z)$ 相关的参数有 $k-1$ 个($z$ 的多项式分布需要 $k-1$ 个参数)，而 LDA 模型中，$p(\theta\mid \alpha)$ 相关的参数有 $k$ 个 ($\theta$ 的 Dirichlet 分布的 $k$ 个超参数，即 $\alpha$)
>  (可以看到，LDA 和混合一元模型的差异就在于 LDA 不限制文档的主题为 1 个，而是认为文档的主题是随机变量，服从 Dirichlet 分布，且文档中每个词都关联一个主题)

## 4.3 Probabilistic latent semantic indexing 
Probabilistic latent semantic indexing (pLSI) is another widely used document model (Hofmann, 1999). The pLSI model, illustrated in Figure 3c, posits that a document label $d$ and a word $w_{n}$ are conditionally independent given an unobserved topic $z$ : 

$$
p(d,w_{n})=p(d)\sum_{z}p(w_{n}\,|\,z)p(z\,|\,d).
$$ 
>  概率潜在语义索引是另一个广泛使用的文档模型
>  如 Figure 3c 所示，该模型认为，在给定未观察到的主题 $z$ 的条件下，文档标签 $d$ 和词 $w_n$ 是条件独立的 
>  (Figure 3c 中，每个词关联一个主题，每个文档关联一个文档标签，文档标签通过影响每个词的主题来影响每个词)
>  此时，文档标签 $d$ 和词 $w_n$ 的联合分布可以写为以上形式

>  推导

$$
\begin{align}
p(d, w_n) &= \sum_z p(d, w_n, z)\\
&=\sum_z p(d)p(z\mid d)p(w_n\mid z)\\
&=p(d)\sum_zp(w_n\mid z)p(z\mid d)
\end{align}
$$

>  推导结束

The pLSI model attempts to relax the simplifying assumption made in the mixture of unigrams model that each document is generated from only one topic. In a sense, it does capture the possibility that a document may contain multiple topics since $p(z\,|\,d)$ serves as the mixture weights of the topics for a particular document $d$ . However, it is important to note that $d$ is a dummy index into the list of documents in the training set . Thus, $d$ is a multinomial random variable with as many possible values as there are training documents and the model learns the topic mixtures $p(z\,|\,d)$ only for those documents on which it is trained. For this reason, pLSI is not a well-defined generative model of documents; there is no natural way to use it to assign probability to a previously unseen document. 
>  pLSI 模型尝试松弛混合一元模型中的假设：每个文档仅由一个主题生成
>  pLSI 模型确实捕获了文档可能包含多个主题的可能性，其中 $p(z\mid d)$ 作为特定文档 $d$ 的主题的混合权重
>  但 $d$ 是训练集的文档列表中文档的占位符索引，因此，$d$ 是一个多项式随机变量，其可能的值和训练集文档数量相同，模型仅为训练集中的文档学习主题混合 $p(z\mid d)$
>  因此，pLSI 并不是一个良定义的文档生成模型，pLSI 模型没有自然的方法为没有见过的模型赋予概率

A further difficulty with pLSI, which also stems from the use of a distribution indexed by training documents, is that the number of parameters which must be estimated grows linearly with the number of training documents. The parameters for a $k$ -topic pLSI model are $k$ multinomial distributions of size $V$ and $M$ mixtures over the $k$ hidden topics. This gives $k V+k M$ parameters and therefore linear growth in $M$ . The linear growth in parameters suggests that the model is prone to overfitting and, empirically, overfitting is indeed a serious problem (see Section 7.1). In practice, a tempering heuristic is used to smooth the parameters of the model for acceptable predictive performance. It has been shown, however, that overfitting can occur even when tempering is used (Popescul et al., 2001). 
>  pLSI 的另一个劣势也源自使用训练文档索引的分布——pLSI 模型需要估计的参数数量随着训练文档的数量线性增长
>  一个 $k$ 主题 pLSI 模型的参数包括 $k$ 个大小为 $V$ 的多项式分布的参数 (条件于每个主题下 $V$ 个词的多项式分布) 和 $M$ 个关于 $k$ 个隐藏主题的混合 (每个文档都定义了一个关于 $k$ 个主题的多项式分布)，因此需要 $kV + kM$ 个参数，故参数和 $M$ 成线性关系
>  (LDA 则通过一个 Dirichlet 分布生成每个文档的主题混合，因此仅需要 $k$ 个参数，与 $M$ 无关)
>  参数的线性增长说明该模型容易过拟合，并且，在经验上，pLSI 确实遭受严重的过拟合问题，实践中，会使用退火启发式方法来平滑模型的参数以获得可接受的预测性能，但即便使用了退火方法，过拟合仍可能发生

LDA overcomes both of these problems by treating the topic mixture weights as a $k$ -parameter hidden random variable rather than a large set of individual parameters which are explicitly linked to the training set. As described in Section 3, LDA is a well-defined generative model and generalizes easily to new documents. Furthermore, the $k+k V$ parameters in a $k$ -topic LDA model do not grow with the size of the training corpus. We will see in Section 7.1 that LDA does not suffer from the same overfitting issues as pLSI. 
>  LDA 克服了 pLSI 的这两个问题，其方法是将主题混合权重视作一个 $k$ 参数隐随机变量，而不是一个和训练集显式关联的大的独立参数的集合
>  LDA 是一个定义良好的生成模型，并且可以轻松泛化到新的文档，且 LDA 模型的参数不随训练语料库大小增长
>  Section 7.1 的试验证明了 LDA 不会像 pLSI 那样遭受过拟合的问题

## 4.4 A geometric interpretation 
A good way of illustrating the differences between LDA and the other latent topic models is by considering the geometry of the latent space, and seeing how a document is represented in that geometry under each model. 
>  通过考虑潜在空间的几何形状，并查看文档在各个模型下在几何中的表示，可以很好地解释 LDA 和其他潜在主题模型的差异

All four of the models described above—unigram, mixture of unigrams, pLSI, and LDA— operate in the space of distributions over words. Each such distribution can be viewed as a point on the $(V-1)$ -simplex, which we call the word simplex. 
>  上述四种模型——一元模型、混合一元模型、pLSI、LDA——都在词分布的空间运行
>  每个词分布 (多项式分布) 都可以视作 $(V-1)$ 维单纯形上的一个点
>  我们称该 $(V-1)$ 维单纯形为词单纯形，词单纯形就是包含了所有可能词分布的词分布空间

The unigram model finds a single point on the word simplex and posits that all words in the corpus come from the corresponding distribution. The latent variable models consider $k$ points on the word simplex and form a sub-simplex based on those points, which we call the topic simplex. Note that any point on the topic simplex is also a point on the word simplex. The different latent variable models use the topic simplex in different ways to generate a document. 
>  一元模型在词单纯形上找到一个点，然后认为语料库中的所有词都生成自这一分布
>  隐变量模型考虑词单纯形上的 $k$ 个点，基于这些点形成一个子单纯形，我们称其为主题单纯形，注意主题单纯形上的任意一点也在词单纯形上
>  不同的隐变量模型通过不同的方式使用主题单纯形来生成文档

- The mixture of unigrams model posits that for each document, one of the $k$ points on the word simplex (that is, one of the corners of the topic simplex) is chosen randomly and all the words of the document are drawn from the distribution corresponding to that point. 
>  混合一元模型的假设是：要生成每个文档，首先会随机选择词单纯形上的 $k$ 个点之一 (也就是主题单纯形的 $k$ 个角之一)，然后该文档的所有词都从该点对应的分布中抽取

- The pLSI model posits that each word of a training document comes from a randomly chosen topic. The topics are themselves drawn from a document-specific distribution over topics, i.e., a point on the topic simplex. There is one such distribution for each document; the set of training documents thus defines an empirical distribution on the topic simplex. 
>  pLSI 模型的假设是：一个训练文档中的每个词都来自于一个随机选择的主题，主题本身则从一个文档特定的主题分布中抽取，该主题分布就是主题单纯形上的一个点
>  每个文档都对应一个主题分布，因此训练文档集定义了在主题单纯形上的一个经验分布 (主题分布的分布)
>  (主题单纯形就是主题的多项式分布的多项式参数的所有可能取值)

- LDA posits that each word of both the observed and unseen documents is generated by a randomly chosen topic which is drawn from a distribution with a randomly chosen parameter. This parameter is sampled once per document from a smooth distribution on the topic simplex. 
>  LDA 的假设是：观测到的和没观测到的文档都是由一个随机选择的主题生成的，该主题也从一个分布中抽取，该分布的参数是随机选择的
>  每个文档都会从主题单纯形上的一个平滑分布中采样一次参数
>  (主题单纯形上的分布就是主题的多项式分布的多项式参数所服从的分布，在 LDA 中，该参数服从 Dirichlet 分布，LDA 为每个主题从 Dirichlet 分布中采样一个多项式参数)

These differences are highlighted in Figure 4. 

![[pics/LDA-Figure4.png]]

>  Figure 4 展示了三个词和三个主题的单纯形
>  三个顶点为三个词各自概率为 1 的词分布
>  主题单纯形的三个顶点是给定该主题后，词的条件分布
>  混合一元模型中，每个文档的词分布只有主题单纯形的三个顶点可选
>  pLSI 模型中为主题单纯形引入了经验分布 (每个训练集文档都对应主题单纯形中的一点，也就是特定的主题混合)
>  LDA 为主题单纯形引入了平滑的分布 (为主题混合定义了 Dirichlet 分布)

# 5. Inference and Parameter Estimation 
We have described the motivation behind LDA and illustrated its conceptual advantages over other latent topic models. 
>  我们在之前讨论了 LDA 的动机和它在概念上相较于其他模型的优势

In this section, we turn our attention to procedures for inference and parameter estimation under LDA. 
>  本节讨论 LDA 的推理和参数估计

## 5.1 Inference 
The key inferential problem that we need to solve in order to use LDA is that of computing the posterior distribution of the hidden variables given a document: 

$$
p({\theta},\mathbf{z}\,|\,\mathbf{w},{\alpha},\beta)=\frac{p({\theta},\mathbf{z},\mathbf{w}\,|\,{\alpha},\beta)}{p(\mathbf{w}\,|\,{\alpha},\beta)}.
$$

>  使用 LDA 推理时，需要解决的关键问题是计算给定一个文档 (观测) 时，隐变量的后验分布 (主题和主题分布的参数)，如上所示

Unfortunately, this distribution is intractable to compute in general. Indeed, to normalize the distribution we marginalize over the hidden variables and write Eq. (3) in terms of the model parameters: 

$$
p(\left.\mathbf{w}\,|\,\alpha,\beta\right)=\frac{\Gamma(\sum_{i}\alpha_{i})}{\prod_{i}\Gamma(\alpha_{i})}\int\left(\prod_{i=1}^{k}\theta_{i}^{\alpha_{i}-1}\right)\left(\prod_{n=1}^{N}\sum_{i=1}^{k}\prod_{j=1}^{V}(\theta_{i}\beta_{i j})^{w_{n}^{j}}\right)d\theta,
$$

a function which is intractable due to the coupling between $\theta$ and $\beta$ in the summation over latent topics (Dickey, 1983). 

>  该后验分布一般是不可计算的
>  为了规范化该分布 (计算分母)，我们需要在全部的隐变量上边际化
>  我们根据 Eq (3)，代入各个参数的表达式，得到文档的边际分布表示如上
>  该函数是不可计算的，因为在对潜在主题求和中，$\theta$ 和 $\beta$ 绑定了
>  (这是隐变量模型的共性，即对隐变量的边际化求和会绑定参数，使其难以独立优化)

>  推导

$$
\begin{align}
p(\mathbf{w}\,|\,\alpha,\beta)&=\int p(\theta\,|\,\alpha)\left(\prod_{n=1}^{N}\sum_{z_{n}}p(z_{n}\,|\,\theta)p(w_{n}\,|\,z_{n},\beta)\right)d\theta\\
&=\frac {\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)}\int\left(\prod_{i=1}^k \theta_i^{\alpha_i - 1}\right)\left(\prod_{n=1}^N\sum_{z_n}p(z_n\mid \theta)p(w_n\mid z_n,\beta)\right)d\theta\\
&=\frac {\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)}\int\left(\prod_{i=1}^k \theta_i^{\alpha_i - 1}\right)\left(\prod_{n=1}^N\sum_{i=1}^kp(z_i\mid \theta)p(w_n\mid z_i,\beta)\right)d\theta\\
&=\frac {\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)}\int\left(\prod_{i=1}^k \theta_i^{\alpha_i - 1}\right)\left(\prod_{n=1}^N\sum_{i=1}^k\theta_ip(w_n\mid z_i,\beta)\right)d\theta\\
&=\frac {\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)}\int\left(\prod_{i=1}^k \theta_i^{\alpha_i - 1}\right)\left(\prod_{n=1}^N\sum_{i=1}^k\prod_{j=1}^V(\theta_i\beta_{i,j})^{w_n^j}\right)d\theta\\
\end{align}
$$

>  其中：
>  第二个等号将 $\theta$ 服从的 Dirichlet 分布的表达式代入
>  第四个等号将 $p(z_i\mid \theta)$ 替换为 $\theta_i$，因为 $z$ 服从多项式分布
>  第五个等号将 $p(w_n\mid z_i, \beta)$ 替换为 $\prod_{j=1}^V \beta_{i, j}^{w_n^j}$，也就是如果 $w_n$ 的第 $j$ 项为 1，$p(w_n \mid z_i, \beta)$ 的结果就是 $\beta_{i, j}$
>  推导完毕

Dickey shows that this function is an expectation under a particular extension to the Dirichlet distribution which can be represented with special hyper geometric functions. It has been used in a Bayesian context for censored discrete data to represent the posterior on $\theta$ which, in that setting, is a random parameter (Dickey et al., 1987). 
>  Dickey 的研究表明该函数是 Dirichlet 分布的一种特定拓展下的期望，可以用超几何函数表示
>  在贝叶斯环境下，它用于处理删失的离散数据，用于表示 $\theta$ 的后验分布，其中 $\theta$ 是一个随机参数

Although the posterior distribution is intractable for exact inference, a wide variety of approximate inference algorithms can be considered for LDA, including Laplace approximation, variational approximation, and Markov chain Monte Carlo (Jordan, 1999). In this section we describe a simple convexity-based variational algorithm for inference in LDA, and discuss some of the alternatives in Section 8. 
>  该后验分布对于精确推理是不可计算的，但可以考虑为 LDA 应用多种近似推断方法，包括拉普拉斯近似、变分近似、MCMC
>  我们在本节中讨论一种基于凸性的简单变分算法来进行推断

## 5.2 Variational inference 
The basic idea of convexity-based variational inference is to make use of Jensen’s inequality to obtain an adjustable lower bound on the log likelihood (Jordan et al., 1999). Essentially, one considers a family of lower bounds, indexed by a set of variational parameters . The variational parameters are chosen by an optimization procedure that attempts to find the tightest possible lower bound. 
>  基于凸性的变分推理的基本思想是利用 Jensen's inequality 对数似然函数的得到可调整的下界
>  本质上，该方法考虑一组由变分参数索引的下界，变分参数根据优化过程进行选择，优化过程试图找到尽可能紧的下界

![[pics/LDA-Figure5.png]]

A simple way to obtain a tractable family of lower bounds is to consider simple modifications of the original graphical model in which some of the edges and nodes are removed. Consider in particular the LDA model shown in Figure 5 (left). The problematic coupling between $\theta$ and $\beta$ arises due to the edges between $\mathbf{\theta},\mathbf{z}$ and $\mathbf w$ . By dropping these edges and the $\mathbf w$ nodes, and endowing the resulting simplified graphical model with free variational parameters, we obtain a family of distributions on the latent variables. This family is characterized by the following variational distribution: 

$$
q({\theta},\mathbf{z}\,|\,{\gamma},{\phi})=q({\theta}\,|\,\boldsymbol{\gamma})\prod_{n=1}^{N}q(z_{n}\,|\,\phi_{n}),\tag{4}
$$ 
where the Dirichlet parameter $\gamma$ and the multinomial parameters $\left(\phi_{1},\ldots,\phi_{N}\right)$ are the free variational parameters. 

>  获得可计算的一族下界的简单方法是考虑对原始图模型进行简单修改，即移除部分的边和节点
>  LDA 模型中，$\theta$ 和 $\beta$ 出现问题性耦合的原因在于 $\theta, \mathbf z, \mathbf w$ 之间的边
>  移除这些边和 $\mathbf w$ 节点，我们得到简化的图模型，为简化的图模型赋予自由变分参数后，就得到了隐变量上的一族分布
>  该分布用以上变分分布表征
>  其中 Dirichlet 参数 $\gamma$ 和多项式参数 $(\phi_1, \dots, \phi_N)$ 即自由变分参数

Having specified a simplified family of probability distributions, the next step is to set up an optimization problem that determines the values of the variational parameters $\gamma$ and $\phi$ . As we show in Appendix A, the desideratum of finding a tight lower bound on the log likelihood translates directly into the following optimization problem: 

$$
({\gamma}^{*},{\phi}^{*})=\arg\operatorname*{min}_{({\gamma},{\phi})}{D}(q({\theta},\mathbf{z}\,|\,{\gamma},{\phi})\parallel p({\theta},\mathbf{z}\,|\,\mathbf{w},{\alpha},{\beta})).\tag{5}
$$

Thus the optimizing values of the variational parameters are found by minimizing the Kullback-Leibler (KL) divergence between the variational distribution and the true posterior $p(\theta,\mathbf{z}\,|\,\mathbf{w},\alpha,\beta)$ . 

>  之后，我们设定优化问题以决定变分参数 $\gamma, \phi$ 的最优值
>  根据 Appendix A，我们的目标是找到对数似然的最紧下界，该目标最后转化为以上的优化问题
>  也就是变分参数的最优值就是能够最小化变分分布和真实后验之间的 KL 散度的参数值

This minimization can be achieved via an iterative fixed-point method. 
>  该最小化问题可以通过迭代式的固定点方法求解

In particular, we show in Appendix A.3 that by computing the derivatives of the KL divergence and setting them equal to zero, we obtain the following pair of update equations: 

$$
\begin{align}
\phi_{ni}&\propto \beta_{iw_n}\exp\{\mathrm E_q[\log (\theta_i)\mid \gamma]\}\tag{6}\\
\gamma_i &=\alpha_i  +\sum_{n=1}^N \phi_{ni}\tag{7}
\end{align}
$$

>  根据 Appendix A.3，令 KL 散度相对于参数的导数为零，我们得到以上的公式

As we show in Appendix A.1, the expectation in the multinomial update can be computed as follows: 

$$
\begin{array}{r}{\mathrm{E}_{q}[\log(\theta_{i})\,|\,{\gamma}]=\Psi(\gamma_{i})-\Psi\left(\sum_{j=1}^{k}\gamma_{j}\right),}\end{array}\tag{8}
$$

where $\Psi$ is the first derivative of the $\log\Gamma$ function which is computable via Taylor approximations (Abramowitz and Stegun, 1970). 

>  根据 Appendix A.1，其中的期望可以通过以上式子计算
>  其中 $\Psi$ 是 $\log \Gamma$ 函数的一阶导数 (也称为 digamma 函数)，可以通过 Taylor 多项式近似

Eqs. (6) and (7) have an appealing intuitive interpretation. The Dirichlet update is a posterior Dirichlet given expected observations taken under the variational distribution, $\mathrm{E}\!\left[z_{n}\,\vert\,\phi_{n}\right]$ . The multinomial update is akin to using Bayes’ theorem, $p\big(z_{n}\,|\,w_{n}\big)\propto p\big(w_{n}\,|\,z_{n}\big)p\big(z_{n}\big)$ , where $p\!\left(z_{n}\right)$ is approximated by the exponential of the expected value of its logarithm under the variational distribution. 
>  Eq 6, 7 存在直观上的解释
>  更新的 Dirichlet 是给定变分分布下期望观测 ($\mathrm E[z_n\mid \phi_n]$) 的后验 Dirichlet
>  更新的多项式分布类似于使用贝叶斯定理 $p(z_n\mid w_n) \propto p(w_n\mid z_n) p(z_n)$，其中 $p(z_n)$ 通过它在变分分布下的对数期望值的指数近似

It is important to note that the variational distribution is actually a conditional distribution, varying as a function of $\mathbf w$ . This occurs because the optimization problem in Eq. (5) is conducted for fixed $\mathbf w$ , and thus yields optimizing parameters $(\gamma^{*},\phi^{*})$ that are a function of $\mathbf w$ . We can write the resulting variational distribution as $q({\theta},\mathbf{z}\,|\,\gamma^{*}(\mathbf{w}),{\phi}^{*}(\mathbf{w}))$ , where we have made the dependence on $\mathbf w$ explicit. Thus the variational distribution can be viewed as an approximation to the posterior distribution $p(\theta,\mathbf{z}\,|\,\mathbf{w},\alpha,\beta)$ . 
>  注意该变分分布实际上是条件分布，它是关于 $\mathbf w$ 的函数
>  因为 Eq 5 的优化问题是基于固定的 $\mathbf w$ 写出的，因此得到的最优参数 $(\gamma^*, \phi^*)$ 就是关于 $\mathbf w$ 的函数
>  故变分分布 $q(\theta, \mathbf z\mid \gamma^*(\mathbf w), \phi^*(\mathbf w))$ 可以视作对后验分布 $p(\theta, \mathbf z \mid \mathbf w, \alpha, \beta)$ 的近似

In the language of text, the optimizing parameters $({\gamma}^{*}(\mathbf{w}),{\phi}^{*}(\mathbf{w})\big)$ are document-specific. In particular, we view the Dirichlet parameters $\gamma^{\ast}(\mathbf{w})$ as providing a representation of a document in the topic simplex. 
>  在文本处理中，最优参数  $(\gamma^*(\mathbf w), \phi^*(\mathbf w))$ 是特定于文档的
>  我们视 Dirichlet 参数 $\gamma^*(\mathbf w)$ 提供了文档在主题单纯形上的一个表示
>  (参数 $\gamma^*(\mathbf w)$ 定义了文档的主题的 Dirichlet 分布)

We summarize the variational inference procedure in Figure 6, with appropriate starting points for $\gamma$ and $\phi_{n}$ . From the pseudocode it is clear that each iteration of variational inference for LDA requires $\mathrm{O}((N+1)k)$ operations. Empirically, we find that the number of iterations required for a single document is on the order of the number of words in the document. This yields a total number of operations roughly on the order of $N^{2}k$ . 
>  变分推断过程总结与 Figure 6，可以看到 LDA 的变分推断中，每次迭代需要 $O((N+1) k)$ 次操作
>  经验上，我们发现单个文档需要的迭代次数大约为文档中的单词数量，因此一个文档需要的总操作次数数量级为 $N^2k$

![[pics/LDA-Figure6.png]]

## 5.3 Parameter estimation 
In this section we present an empirical Bayes method for parameter estimation in the LDA model (see Section 5.4 for a fuller Bayesian approach). In particular, given a corpus of documents $\pmb D=$ $\left\{\mathbf{w}_{1},\mathbf{w}_{2},\ldots,\mathbf{w}_{M}\right\}$ , we wish to find parameters $\alpha$ and $\beta$ that maximize the (marginal) log likelihood of the data: 
 
$$
\ell(\alpha,\beta)=\sum_{d=1}^{M}\log p(\mathbf{w}_{d}\,|\,\alpha,\beta).
$$ 
>  本节讨论对 LDA 模型进行参数估计的方法
>  具体地说，给定文档语料库 $\pmb D = \{\mathbf w_1, \dots, \mathbf w_M\}$，我们希望找到最大化数据边际对数似然的参数 $\alpha, \beta$，边际对数似然如上所示

As we have described above, the quantity $p(\mathbf{w}\,|\,\alpha,\beta)$ cannot be computed tractably. However, variational inference provides us with a tractable lower bound on the log likelihood, a bound which we can maximize with respect to $\alpha$ and $\beta$ . We can thus find approximate empirical Bayes estimates for the LDA model via an alternating variational EM procedure that maximizes a lower bound with respect to the variational parameters $\gamma$ and $\phi$ , and then, for fixed values of the variational parameters, maximizes the lower bound with respect to the model parameters $\alpha$ and $\beta$ . 
>  因为边际概率 $p(\mathbf w\mid \alpha, \beta)$ 是不可计算的，故数据边际似然也是不可计算的
>  故我们需要借助变分推理，得到对数似然的可计算的下界，我们相对于该下界优化 $\alpha, \beta$
>  因此，我们需要通过交替的变分 EM 过程寻找 LDA 模型的近似经验贝叶斯估计
>  EM 过程中，我们先通过变分方法，找到最大化下界的变分参数 $\gamma, \phi$，然后固定变分参数，找到最大化下界的模型参数 $\alpha, \beta$
>  (根据 Eq (13) ，给定 $\alpha, \beta$，最大化对数似然下界的变分参数 $\gamma, \phi$ 定义的就是最小化和真实后验之间的 KL 散度的近似变分后验
>  )

We provide a detailed derivation of the variational EM algorithm for LDA in Appendix A.4. The derivation yields the following iterative algorithm: 

1. (E-step) For each document, find the optimizing values of the variational parameters $\{\gamma_{d}^{*},\phi_{d}^{*}:$ : $d\in D\}$ . This is done as described in the previous section. 
2. (M-step) Maximize the resulting lower bound on the log likelihood with respect to the model parameters $\alpha$ and $\beta$ . This corresponds to finding maximum likelihood estimates with expected sufficient statistics for each document under the approximate posterior which is computed in the E-step. 

In Appendix A.4, we show that the M-step update for the conditional multinomial parameter $\beta$ can be written out analytically: 

$$
\beta_{i j}\propto\sum_{d=1}^{M}\sum_{n=1}^{N_{d}}\phi_{d n i}^{*}w_{d n}^{j}.
$$ 
We further show that the M-step update for Dirichlet parameter $\alpha$ can be implemented using an efficient Newton-Raphson method in which the Hessian is inverted in linear time. 

## 5.4 Smoothing 
The large vocabulary size that is characteristic of many document corpora creates serious problems of sparsity. A new document is very likely to contain words that did not appear in any of the documents in a training corpus. Maximum likelihood estimates of the multinomial parameters assign zero probability to such words, and thus zero probability to new documents. The standard approach to coping with this problem is to “smooth” the multinomial parameters, assigning positive probability to all vocabulary items whether or not they are observed in the training set (Jelinek, 1997). Laplace smoothing is commonly used; this essentially yields the mean of the posterior distribution under a uniform Dirichlet prior on the multinomial parameters. 
Unfortunately, in the mixture model setting, simple Laplace smoothing is no longer justified as a maximum a posteriori method (although it is often implemented in practice; cf. Nigam et al., 1999). In fact, by placing a Dirichlet prior on the multinomial parameter we obtain an intractable posterior in the mixture model setting, for much the same reason that one obtains an intractable posterior in the basic LDA model. Our proposed solution to this problem is to simply apply variational inference methods to the extended model that includes Dirichlet smoothing on the multinomial parameter. 
In the LDA setting, we obtain the extended graphical model shown in Figure 7. We treat $\beta$ as a $k\times V$ random matrix (one row for each mixture component), where we assume that each row is independently drawn from an exchangeable Dirichlet distribution. We now extend our inference procedures to treat the $\beta_{i}$ as random variables that are endowed with a posterior distribution, conditioned on the data. Thus we move beyond the empirical Bayes procedure of Section 5.3 and consider a fuller Bayesian approach to LDA. 
We consider a variational approach to Bayesian inference that places a separable distribution on the random variables ${\beta},\,\theta.$ , and $\mathbf{z}$ (Attias, 2000): 
$$
q(\beta_{1:k},\mathbf{z}_{1:M},\theta_{1:M}\,|\,\uplambda,\Upphi,\gamma)=\prod_{i=1}^{k}\mathrm{Dir}(\beta_{i}\,|\,\uplambda_{i})\prod_{d=1}^{M}q_{d}(\theta_{d},\mathbf{z}_{d}\,|\,\upphi_{d},\gamma_{d}),
$$ 
where $q_{d}(\boldsymbol{\Theta},\mathbf{z}\,|\,\boldsymbol{\Phi},\Upsilon)$ is the variational distribution defined for LDA in Eq. (4). As is easily verified, the resulting variational inference procedure again yields Eqs. (6) and (7) as the update equations for the variational parameters $\Phi$ and $\gamma,$ respectively, as well as an additional update for the new variational parameter $\uplambda$ : 
$$
\lambda_{i j}=\boldsymbol{\upeta}+\sum_{d=1}^{M}\sum_{n=1}^{N_{d}}\boldsymbol{\upphi}_{d n i}^{*}\boldsymbol{w}_{d n}^{j}.
$$ 
Iterating these equations to convergence yields an approximate posterior distribution on $\beta,\theta$ , and $\mathbf{z}$ . 
We are now left with the hyper parameter $\boldsymbol\upeta$ on the exchangeable Dirichlet, as well as the hyper parameter $\alpha$ from before. Our approach to setting these hyper parameters is again (approximate) empirical Bayes—we use variational EM to find maximum likelihood estimates of these parameters based on the marginal likelihood. These procedures are described in Appendix A.4. 
# 6. Example 
In this section, we provide an illustrative example of the use of an LDA model on real data. Our data are 16,000 documents from a subset of the TREC AP corpus (Harman, 1992). After removing a standard list of stop words, we used the EM algorithm described in Section 5.3 to find the Dirichlet and conditional multinomial parameters for a 100-topic LDA model. The top words from some of the resulting multinomial distributions $p(w\,|\,z)$ are illustrated in Figure 8 (top). As we have hoped, these distributions seem to capture some of the underlying topics in the corpus (and we have named them according to these topics). 
As we emphasized in Section 4, one of the advantages of LDA over related latent variable models is that it provides well-defined inference procedures for previously unseen documents. Indeed, we can illustrate how LDA works by performing inference on a held-out document and examining the resulting variational posterior parameters. 
Figure 8 (bottom) is a document from the TREC AP corpus which was not used for parameter estimation. Using the algorithm in Section 5.1, we computed the variational posterior Dirichlet parameters $\gamma$ for the article and variational posterior multinomial parameters $\phi_{n}$ for each word in the article. 
Recall that the i th posterior Dirichlet parameter $\gamma_{i}$ is approximately the i th prior Dirichlet parameter $\mathfrak{X}_{i}$ plus the expected number of words which were generated by the i th topic (see Eq. 7). Therefore, the prior Dirichlet parameters subtracted from the posterior Dirichlet parameters indicate the expected number of words which were allocated to each topic for a particular document. For the example article in Figure 8 (bottom), most of the $\gamma_{i}$ are close to $\mathfrak{X}_{i}$ . Four topics, however, are sign i cant ly larger (by this, we mean $\gamma_{i}-\alpha_{i}\geq1\mathrm{~}$ ). Looking at the corresponding distributions over words identifies the topics which mixed to form this document (Figure 8, top). 
Further insight comes from examining the $\phi_{n}$ parameters. These distributions approximate $p(z_{n}\,|\,\mathbf{w})$ and tend to peak towards one of the $k$ possible topic values. In the artic , the words are color coded according to these values (i.e., the i th color is used if $q_{n}(z_{n}^{i}=1)>0.9)$ 9). With this illustration, one can identify how the different topics mixed in the document text. 
While demonstrating the power of LDA, the posterior analysis also highlights some of its limitations. In particular, the bag-of-words assumption allows words that should be generated by the same topic (e.g., “William Randolph Hearst Foundation”) to be allocated to several different topics. Overcoming this limitation would require some form of extension of the basic LDA model; in particular, we might relax the bag-of-words assumption by assuming partial exchange ability or Markov ian it y of word sequences. 
# 7. Applications and Empirical Results 
In this section, we discuss our empirical evaluation of LDA in several problem domains—document modeling, document class i cation, and collaborative filtering. 
In all of the mixture models, the expected complete log likelihood of the data has local maxima at the points where all or some of the mixture components are equal to each other. To avoid these local maxima, it is important to initialize the EM algorithm appropriately. In our experiments, we initialize EM by seeding each conditional multinomial distribution with five documents, reducing their effective total length to two words, and smoothing across the whole vocabulary. This is essentially an approximation to the scheme described in Heckerman and Meila (2001). 
# 7.1 Document modeling 
We trained a number of latent variable models, including LDA, on two text corpora to compare the generalization performance of these models. The documents in the corpora are treated as unlabeled; thus, our goal is density estimation—we wish to achieve high likelihood on a held-out test set. In particular, we computed the perplexity of a held-out test set to evaluate the models. The perplexity, used by convention in language modeling, is monotonically decreasing in the likelihood of the test data, and is algebraic ly equivalent to the inverse of the geometric mean per-word likelihood. A lower perplexity score indicates better generalization performance. More formally, for a test set of $M$ documents, the perplexity is: 
$$
p e r p l e x i t y(D_{\mathrm{test}})=\exp\left\{-\frac{\sum_{d=1}^{M}\log p(\mathbf{w}_{d})}{\sum_{d=1}^{M}N_{d}}\right\}.
$$ 
In our experiments, we used a corpus of scientific abstracts from the C. Elegans community (Avery, 2002) containing 5,225 abstracts with 28,414 unique terms, and a subset of the TREC AP corpus containing 16,333 newswire articles with 23,075 unique terms. In both cases, we held out $10\%$ of the data for test purposes and trained the models on the remaining $90\%$ . In preprocessing the data, 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/044de3a4f33b0626810d01546f5295c65a73cfb227f5b914b9c7684f618bb780.jpg) 
The William Randolph Hearst Foundation will give $\S1.25$ million to Lincoln Center, Metropoli-tan Opera Co., New York Philharmonic and Juilliard School. “Our board felt that we had a real opportunity to make a mark on the future of the performing arts with these grants an act every bit as important as our traditional areas of support in health, medical research, education and the social services,” Hearst Foundation President Randolph A. Hearst said Monday in announcing the grants. Lincoln Center’s share will be $\mathbb{S}200{,}000$ for its new building, which will house young artists and provide new public facilities. The Metropolitan Opera Co. and New York Philharmonic will receive $\mathbb{S400,000}$ each. The Juilliard School, where music and the performing arts are taught, will get $\mathbb{S}250{,}000$ . The Hearst Foundation, a leading supporter of the Lincoln Center Consolidated Corporate Fund, will make its usual annual $\mathbb{S}100{,}000$ donation, too. 
Figure 8: An example article from the AP corpus. Each color codes a different factor from which the word is putatively generated. 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/a192ff1a8b109c3e66421ffd55b97a3e95f81d88d2de6f533d559e3c95380227.jpg) 
Figure 9: Perplexity results on the nematode (Top) and AP (Bottom) corpora for LDA, the unigram model, mixture of unigrams, and pLSI. 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/c8ac47f18f60ecf1a6137de1e2b50f143e0363c9f715a8a9d3076677cebc8d0d.jpg) 
Table 1: Overfitting in the mixture of unigrams and pLSI models for the AP corpus. Similar behavior is observed in the nematode corpus (not reported). 
we removed a standard list of 50 stop words from each corpus. From the AP data, we further removed words that occurred only once. 
We compared LDA with the unigram, mixture of unigrams, and pLSI models described in Section 4. We trained all the hidden variable models using EM with exactly the same stopping criteria, that the average change in expected log likelihood is less than $0.001\%$ . 
Both the pLSI model and the mixture of unigrams suffer from serious overfitting issues, though for different reasons. This phenomenon is illustrated in Table 1. In the mixture of unigrams model, overfitting is a result of peaked posteriors in the training set; a phenomenon familiar in the supervised setting, where this model is known as the naive Bayes model (Rennie, 2001). This leads to a nearly deterministic clustering of the training documents (in the E-step) which is used to determine the word probabilities in each mixture component (in the M-step). A previously unseen document may best fit one of the resulting mixture components, but will probably contain at least one word which did not occur in the training documents that were assigned to that component. Such words will have a very small probability, which causes the perplexity of the new document to explode. As $k$ increases, the documents of the training corpus are partitioned into finer collections and thus induce more words with small probabilities. 
In the mixture of unigrams, we can alleviate overfitting through the variational Bayesian smoothing scheme presented in Section 5.4. This ensures that all words will have some probability under every mixture component. 
In the pLSI case, the hard clustering problem is alleviated by the fact that each document is allowed to exhibit a different proportion of topics. However, pLSI only refers to the training documents and a different overfitting problem arises that is due to the dimensionality of the $p(z|d)$ parameter. One reasonable approach to assigning probability to a previously unseen document is by marginal i zing over $d$ : 
$$
p(\mathbf{w})=\sum_{d}\prod_{n=1}^{N}\sum_{z}p(w_{n}\,|\,z)p(z\,|\,d)p(d).
$$ 
Essentially, we are integrating over the empirical distribution on the topic simplex (see Figure 4). 
This method of inference, though theoretically sound, causes the model to overfit. The documentspecific topic distribution has some components which are close to zero for those topics that do not appear in the document. Thus, certain words will have very small probability in the estimates of 
each mixture component. When determining the probability of a new document through marginalization, only those training documents which exhibit a similar proportion of topics will contribute to the likelihood. For a given training document’s topic proportions, any word which has small probability in all the constituent topics will cause the perplexity to explode. As $k$ gets larger, the chance that a training document will exhibit topics that cover all the words in the new document decreases and thus the perplexity grows. Note that pLSI does not overfit as quickly (with respect to $k$ ) as the mixture of unigrams. 
This overfitting problem essentially stems from the restriction that each future document exhibit the same topic proportions as were seen in one or more of the training documents. Given this constraint, we are not free to choose the most likely proportions of topics for the new document. An alternative approach is the “folding-in” heuristic suggested by Hofmann (1999), where one ignores the $p(z|d)$ parameters refits $p(z|d_{\mathrm{new}})$ . Note that this gives the pLSI model an unfair advantage by allowing it to refit $k-1$ − 1 parameters to the test data. 
LDA suffers from neither of these problems. As in pLSI, each document can exhibit a different proportion of underlying topics. However, LDA can easily assign probability to a new document; no heuristics are needed for a new document to be endowed with a different set of topic proportions than were associated with documents in the training corpus. 
Figure 9 presents the perplexity for each model on both corpora for different values of $k$ . The pLSI model and mixture of unigrams are suitably corrected for overfitting. The latent variable models perform better than the simple unigram model. LDA consistently performs better than the other models. 
# 7.2 Document class i cation 
In the text class i cation problem, we wish to classify a document into two or more mutually exclusive classes. As in any class i cation problem, we may wish to consider generative approaches or disc rim i native approaches. In particular, by using one LDA module for each class, we obtain a generative model for class i cation. It is also of interest to use LDA in the disc rim i native framework, and this is our focus in this section. 
A challenging aspect of the document class i cation problem is the choice of features. Treating individual words as features yields a rich but very large feature set (Joachims, 1999). One way to reduce this feature set is to use an LDA model for dimensionality reduction. In particular, LDA reduces any document to a fixed set of real-valued features—the posterior Dirichlet parameters $\gamma^{\ast}(\mathbf{w})$ associated with the document. It is of interest to see how much discriminatory information we lose in reducing the document description to these parameters. 
We conducted two binary class i cation experiments using the Reuters-21578 dataset. The dataset contains 8000 documents and 15,818 words. 
In these experiments, we estimated the parameters of an LDA model on all the documents, without reference to their true class label. We then trained a support vector machine (SVM) on the low-dimensional representations provided by LDA and compared this SVM to an SVM trained on all the word features. 
Using the SVMLight software package (Joachims, 1999), we compared an SVM trained on all the word features with those trained on features induced by a 50-topic LDA model. Note that we reduce the feature space by 99.6 percent in this case. 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/482e3480304e298eec578c02f6053e0999f2549ad6d79730748628f5fc115722.jpg) 
Figure 10: Class i cation results on two binary class i cation problems from the Reuters-21578 dataset for different proportions of training data. Graph (a) is EARN vs. NOT EARN . Graph (b) is GRAIN vs. NOT GRAIN . 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/9606c15444f5bcfe3c4f780529676092ba4431ba89cdfa5408237a29876dd53f.jpg) 
Figure 11: Results for collaborative filtering on the EachMovie data. 
Figure 10 shows our results. We see that there is little reduction in class i cation performance in using the LDA-based features; indeed, in almost all cases the performance is improved with the LDA features. Although these results need further substantiation, they suggest that the topic-based representation provided by LDA may be useful as a fast filtering algorithm for feature selection in text class i cation. 
# 7.3 Collaborative filtering 
Our final experiment uses the EachMovie collaborative filtering data. In this data set, a collection of users indicates their preferred movie choices. A user and the movies chosen are analogous to a document and the words in the document (respectively). 
The collaborative filtering task is as follows. We train a model on a fully observed set of users. Then, for each unobserved user, we are shown all but one of the movies preferred by that user and are asked to predict what the held-out movie is. The different algorithms are evaluated according to the likelihood they assign to the held-out movie. More precisely, define the predictive perplexity on $M$ test users to be: 
$$
\begin{array}{r}{p r e d i c t i v e\mathrm{-}p e r p l e x i t y(D_{\mathrm{test}})=\exp\left\{-\frac{\sum_{d=1}^{M}\log p\left(w_{d,N_{d}}\,\vert\,\mathbf{w}_{d,1:N_{d}-1}\right)}{M})\right\}.}\end{array}
$$ 
We restricted the EachMovie dataset to users that positively rated at least 100 movies (a positive rating is at least four out of five stars). We divided this set of users into 3300 training users and 390 testing users. 
Under the mixture of unigrams model, the probability of a movie given a set of observed movies is obtained from the posterior distribution over topics: 
$$
p(w|\mathbf{w}_{\mathrm{obs}})=\sum_{z}p(w|z)p(z|\mathbf{w}_{\mathrm{obs}}).
$$ 
In the pLSI model, the probability of a held-out movie is given by the same equation except that $p(z|\mathbf{w}_{\mathrm{obs}})$ is computed by folding in the previously seen movies. Finally, in the LDA model, the probability of a held-out movie is given by integrating over the posterior Dirichlet: 
$$
p(w|\mathbf{w}_{\mathrm{obs}})=\int\sum_{z}p(w|z)p(z|\Theta)p(\Theta|\mathbf{w}_{\mathrm{obs}})d\Theta,
$$ 
where $p\big(\boldsymbol{\Theta}|\mathbf{w}_{\mathrm{obs}}\big)$ is given by the variational inference method described in Section 5.2. Note that this quantity is efficient to compute. We can interchange the sum and integral sign, and compute a linear combination of $k$ Dirichlet expectations. 
With a vocabulary of 1600 movies, we find the predictive perplexities illustrated in Figure 11. Again, the mixture of unigrams model and pLSI are corrected for overfitting, but the best predictive perplexities are obtained by the LDA model. 
# 8. Discussion 
We have described latent Dirichlet allocation, a ﬂexible generative probabilistic model for collections of discrete data. LDA is based on a simple exchange ability assumption for the words and topics in a document; it is therefore realized by a straightforward application of de Finetti’s representation theorem. We can view LDA as a dimensionality reduction technique, in the spirit of LSI, but with proper underlying generative probabilistic semantics that make sense for the type of data that it models. 
Exact inference is intractable for LDA, but any of a large suite of approximate inference algorithms can be used for inference and parameter estimation within the LDA framework. We have presented a simple convexity-based variational approach for inference, showing that it yields a fast algorithm resulting in reasonable comparative performance in terms of test set likelihood. Other approaches that might be considered include Laplace approximation, higher-order variational techniques, and Monte Carlo methods. In particular, Leisink and Kappen (2002) have presented a general methodology for converting low-order variational lower bounds into higher-order variational bounds. It is also possible to achieve higher accuracy by dispensing with the requirement of maintaining a bound, and indeed Minka and Lafferty (2002) have shown that improved inferential accuracy can be obtained for the LDA model via a higher-order variational technique known as expectation propagation. Finally, Griffiths and Steyvers (2002) have presented a Markov chain Monte Carlo algorithm for LDA. 
LDA is a simple model, and although we view it as a competitor to methods such as LSI and pLSI in the setting of dimensionality reduction for document collections and other discrete corpora, it is also intended to be illustrative of the way in which probabilistic models can be scaled up to provide useful inferential machinery in domains involving multiple levels of structure. Indeed, the principal advantages of generative models such as LDA include their modularity and their extensibility. As a probabilistic module, LDA can be readily embedded in a more complex model— a property that is not possessed by LSI. In recent work we have used pairs of LDA modules to model relationships between images and their corresponding descriptive captions (Blei and Jordan, 2002). Moreover, there are numerous possible extensions of LDA. For example, LDA is readily extended to continuous data or other non-multinomial data. As is the case for other mixture models, including finite mixture models and hidden Markov models, the “emission” probability $p\big(w_{n}\,\vert\,z_{n}\big)$ contributes only a likelihood value to the inference procedures for LDA, and other likelihoods are readily substituted in its place. In particular, it is straightforward to develop a continuous variant of LDA in which Gaussian observable s are used in place of multinomials. Another simple extension of LDA comes from allowing mixtures of Dirichlet distributions in the place of the single Dirichlet of LDA. This allows a richer structure in the latent topic space and in particular allows a form of document clustering that is different from the clustering that is achieved via shared topics. Finally, a variety of extensions of LDA can be considered in which the distributions on the topic variables are elaborated. For example, we could arrange the topics in a time series, essentially relaxing the full exchange ability assumption to one of partial exchange ability. We could also consider partially exchangeable models in which we condition on exogenous variables; thus, for example, the topic distribution could be conditioned on features such as “paragraph” or “sentence,” providing a more powerful text model that makes use of information obtained from a parser. 
# Acknowledgements 
This work was supported by the National Science Foundation (NSF grant IIS-9988642) and the Multidisciplinary Research Program of the Department of Defense (MURI N00014-00-1-0637). Andrew Y. $\mathrm{Mg}$ and David M. Blei were additionally supported by fellowships from the Microsoft Corporation. 
# References 
M. Abramowitz and I. Stegun, editors. Handbook of Mathematical Functions . Dover, New York, 1970. 
D. Aldous. Exchange ability and related topics. In Ecole d’´et´e de probabilit´es de Saint-Flour, XIII— 1983 , pages 1–198. Springer, Berlin, 1985. H. Attias. A variational Bayesian framework for graphical models. In Advances in Neural Information Processing Systems 12 , 2000. L. Avery. Caen or rh abd it is genetic center bibliography. 2002. URL http://elegans.swmed.edu/wli/cgcbib . R. Baeza-Yates and B. Ribeiro-Neto. Modern Information Retrieval . ACM Press, New York, 1999. D. Blei and M. Jordan. Modeling annotated data. Technical Report UCB//CSD-02-1202, U.C. Berkeley Computer Science Division, 2002. B. de Finetti. Theory of probability. Vol. 1-2 . John Wiley & Sons Ltd., Chichester, 1990. Reprint of the 1975 translation. S. Deerwester, S. Dumais, T. Landauer, G. Furnas, and R. Harshman. Indexing by latent semantic analysis. Journal of the American Society of Information Science , 41(6):391–407, 1990. P. Diaconis. Recent progress on de Finetti’s notions of exchange ability. In Bayesian statistics, 3 (Valencia, 1987) , pages 111–125. Oxford Univ. Press, New York, 1988. J. Dickey. Multiple hyper geometric functions: Probabilistic interpretations and statistical uses. Journal of the American Statistical Association , 78:628–637, 1983. J. Dickey, J. Jiang, and J. Kadane. Bayesian methods for censored categorical data. Journal of the American Statistical Association , 82:773–781, 1987. A. Gelman, J. Carlin, H. Stern, and D. Rubin. Bayesian data analysis . Chapman & Hall, London, 1995. T. Griffiths and M. Steyvers. A probabilistic approach to semantic representation. In Proceedings of the 24th Annual Conference of the Cognitive Science Society , 2002. D. Harman. Overview of the first text retrieval conference (TREC-1). In Proceedings of the First Text Retrieval Conference (TREC-1) , pages 1–20, 1992. D. Heckerman and M. Meila. An experimental comparison of several clustering and initialization methods. Machine Learning , 42:9–29, 2001. T. Hofmann. Probabilistic latent semantic indexing. Proceedings of the Twenty-Second Annual International SIGIR Conference , 1999. F. Jelinek. Statistical Methods for Speech Recognition . MIT Press, Cambridge, MA, 1997. T. Joachims. Making large-scale SVM learning practical. In Advances in Kernel Methods - Support Vector Learning . M.I.T. Press, 1999. M. Jordan, editor. Learning in Graphical Models . MIT Press, Cambridge, MA, 1999. 
M. Jordan, Z. Ghahramani, T. Jaakkola, and L. Saul. Introduction to variational methods for graphical models. Machine Learning , 37:183–233, 1999. R. Kass and D. Steffey. Approximate Bayesian inference in conditionally independent hierarchical models (parametric empirical Bayes models). Journal of the American Statistical Association , 84 (407):717–726, 1989.M. Leisink and H. Kappen. General lower bounds based on computer generated higher order expansions. In Uncertainty in Artificial Intelligence, Proceedings of the Eighteenth Conference , 2002. T. Minka. Estimating a Dirichlet distribution. Technical report, M.I.T., 2000. T. P. Minka and J. Lafferty. Expectation-propagation for the generative aspect model. In Uncertainty in Artificial Intelligence (UAI) , 2002. C. Morris. Parametric empirical Bayes inference: Theory and applications. Journal of the American Statistical Association , 78(381):47–65, 1983. With discussion. K. Nigam, J. Lafferty, and A. McCallum. Using maximum entropy for text class i cation. IJCAI-99 Workshop on Machine Learning for Information Filtering , pages 61–67, 1999. K. Nigam, A. McCallum, S. Thrun, and T. Mitchell. Text class i cation from labeled and unlabeled documents using EM. Machine Learning , 39(2/3):103–134, 2000. C. Papa dimitri ou, H. Tamaki, P. Raghavan, and S. Vempala. Latent semantic indexing: A probabilistic analysis. pages 159–168, 1998. A. Popescul, L. Ungar, D. Pennock, and S. Lawrence. Probabilistic models for unified collaborative and content-based recommendation in sparse-data environments. In Uncertainty in Artificial Intelligence, Proceedings of the Seventeenth Conference , 2001. J. Rennie. Improving multi-class text class i cation with naive Bayes. Technical Report AITR-2001- 004, M.I.T., 2001. G. Ronning. Maximum likelihood estimation of Dirichlet distributions. Journal of Statistcal Computation and Simulation , 34(4):215–221, 1989. G. Salton and M. McGill, editors. Introduction to Modern Information Retrieval . McGraw-Hill, 1983. 
# Appendix A. Inference and parameter estimation 
In this appendix, we derive the variational inference procedure (Eqs. 6 and 7) and the parameter maximization procedure for the conditional multinomial (Eq. 9) and for the Dirichlet. We begin by deriving a useful property of the Dirichlet distribution. 

## A.1 Computing $\mathbf{E}[\log(\theta_{i}\,|\,{\alpha})]$ 
The need to compute the expected value of the log of a single probability component under the Dirichlet arises repeatedly in deriving the inference and parameter estimation procedures for LDA. This value can be easily computed from the natural parameterization of the exponential family representation of the Dirichlet distribution. 
Recall that a distribution is in the exponential family if it can be written in the form: 
$$
p(x\,|\,\boldsymbol{\mathfrak{n}})=h(x)\exp\left\{\boldsymbol{\mathfrak{n}}^{T}T(x)-A(\boldsymbol{\mathfrak{n}})\right\},
$$
  
where $\boldsymbol\upeta$ is the natural parameter, $T(x)$ is the sufficient statistic, and $A(\boldsymbol{\mathfrak{n}})$ is the log of the normalization factor. 
We can write the Dirichlet in this form by exponent i a ting the log of Eq. (1): 
$$
\begin{array}{r}{p(\Theta\,|\,\alpha)=\exp\left\{\left(\sum_{i=1}^{k}(\alpha_{i}-1)\log\theta_{i}\right)+\log\Gamma\left(\sum_{i=1}^{k}\alpha_{i}\right)-\sum_{i=1}^{k}\log\Gamma(\alpha_{i})\right\}.}\end{array}
$$
       
From this form, we imm at the natural parameter of the Dirichlet is $\mathfrak{n}_{i}=\mathfrak{a}_{i}-1$ and the sufficient statistic is $T(\theta_{i})=\log\theta_{i}$ . Furthermore, using the general fact that the derivative of the log normalization factor with respect to the natural parameter is equal to the expectation of the sufficient statistic, we obtain: 
$$
\begin{array}{r}{\mathrm{E}[\log\Theta_{i}\,|\,\mathfrak{a}]=\Psi(\mathfrak{a}_{i})-\Psi\left(\sum_{j=1}^{k}\mathfrak{a}_{j}\right)}\end{array}
$$
    
where $\Psi$ is the digamma function, the first derivative of the log Gamma function. 
# A.2 Newton-Raphson methods for a Hessian with special structure 
In this section we describe a linear algorithm for the usually cubic Newton-Raphson optimization method. This method is used for maximum likelihood estimation of the Dirichlet distribution (Ronning, 1989, Minka, 2000). 
The Newton-Raphson optimization technique finds a stationary point of a function by iterating: 
$$
\alpha_{\mathrm{new}}=\alpha_{\mathrm{old}}-H(\alpha_{\mathrm{old}})^{-1}g(\alpha_{\mathrm{old}})
$$ 
where $H(\alpha)$ and $g(\alpha)$ are the Hessian matrix and gradient respectively at the point $\alpha$ . In general, this algorithm scales as $\mathrm{O}(N^{3})$ due to the matrix inversion. 
If the Hessian matrix is of the form: 
$$
H=\mathrm{diag}(h)+\mathbf{1}z\mathbf{1}^{\mathrm{T}},
$$ 
where $\mathrm{diag}(h)$ is defined to be a diagonal matrix with the elements of the vector $h$ along the diagonal, then we can apply the matrix inversion lemma and obtain: 
$$
H^{-1}=\mathrm{diag}(h)^{-1}-\frac{\mathrm{diag}(h)^{-1}{\bf11^{\mathrm{T}}}\mathrm{diag}(h)^{-1}}{z^{-1}+\sum_{j=1}^{k}h_{j}^{-1}}
$$ 
Multiplying by the gradient, we obtain the i th component: 
$$
(H^{-1}g)_{i}=\frac{g_{i}-c}{h_{i}}
$$ 
where 
$$
c=\frac{\sum_{j=1}^{k}g_{j}/h_{j}}{z^{-1}\!+\!\sum_{j=1}^{k}h_{j}^{-1}}.
$$ 
Observe that this expression depends only on the $2k$ values $h_{i}$ and $g_{i}$ and thus yields a NewtonRaphson algorithm that has linear time complexity. 
## A.3 Variational Inference 
In this section we derive the variational inference algorithm described in Section 5.1. Recall that this involves using the following variational distribution : 

$$
q(\theta,\mathbf{z}\,|\,\gamma,\phi)=q(\theta\,|\,\gamma)\prod_{n=1}^{N}q(z_{n}\,|\,\phi_{n})\tag{11}
$$ 
as a surrogate for the posterior distribution $p({\theta},\mathbf{z},\mathbf{w}\,|\,\alpha,\beta)$ , where the variational parameters $\gamma$ and $\phi$ are set via an optimization procedure that we now describe. 

>  变分推断中，我们用变分分布 (11) 替代后验分布 $p(\theta, \mathbf z\mid \mathbf w, \alpha, \beta)$ 
>  我们优化的目标是变分参数 $\gamma$ 和 $\phi$

Following Jordan et al. (1999), we begin by bounding the log likelihood of a document using Jensen’s inequality. Omitting the parameters $\gamma$ and $\phi$ for simplicity, we have: 

$$
\begin{align}
\log p(\mathbf w\mid \alpha, \beta) &= \log \int \sum_{\mathbf z}p(\theta, \mathbf z, \mathbf w \mid \alpha, \beta)d\theta\\
&=\log \int \sum_{\mathbf z}\frac {p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)q(\theta, \mathbf z)}{q(\theta, \mathbf z)}d\theta\\
&=\log\mathrm E_q\left[\frac {p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)}{q(\theta, \mathbf z)}\right]\\
&\ge\int \sum_{\mathbf z} q(\theta, \mathbf z)\log p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)d\theta - \int \sum_{\mathbf z}q(\theta,\mathbf z)\log q(\theta, \mathbf z)d\theta\\
&=\mathrm E_q[\log p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)] - \mathrm E_q[\log q(\theta, \mathbf z)]\tag{12}
\end{align}
$$

Thus we see that Jensen’s inequality provides us with a lower bound on the log likelihood for an arbitrary variational distribution $q({\theta},\mathbf{z}\,\vert\,{\gamma},{\phi})$ . 

>  我们考虑一个文档的对数似然，经过推导，可以看到，Jensen 不等式为任意一个变分分布 $q(\theta, \mathbf z\mid \gamma, \phi)$ 都为对数似然提供了一个下界

It can be easily verified that the difference between the left-hand side and the right-hand side of the Eq. (12) is the KL divergence between the variational posterior probability and the true posterior probability. That is, letting $L\left(\gamma,\phi;\alpha,\beta\right)$ denote the right-hand side of Eq. (12) (where we have restored the dependence on the variational parameters $\gamma$ and $\phi$ in our notation), we have: 

$$
\log p(\mathbf{w}\,|\,\alpha,\beta)=L(\gamma,\phi;\alpha,\beta)+{D}(q(\theta,\mathbf{z}\,|\,\gamma,\phi)\ ||\ p(\theta,\mathbf{z}\,|\,\mathbf{w},\alpha,\beta)).\tag{13}
$$ 
>  容易知道，Eq 12 的 LHS 减去 RHS 就是变分后验分布和真实后验分布之间的 KL 散度
>  我们将 Eq 12 的 RHS 写为 $L(\gamma, \phi; \alpha, \beta)$，表示它是关于 $\gamma, \phi$ 的函数，可以得到 Eq 13 成立

>  证明

$$
\begin{align}
&L(\gamma,\phi;\alpha, \beta) + D(q(\theta, \mathbf z\mid \gamma,\phi)\parallel p(\theta, \mathbf z\mid \mathbf w, \alpha, \beta))\\
=&\mathrm E_q\left[\log p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)\right] - \mathrm E_q[\log q(\theta, \mathbf z\mid \gamma,\phi)] + D(q(\theta, \mathbf z\mid \gamma, \phi)\parallel p(\theta, \mathbf z\mid \mathbf w, \alpha, \beta))\\
=&\mathrm E_q\left[\log\frac {p(\theta, \mathbf z,\mathbf w\mid \alpha, \beta)}{q(\theta, \mathbf z\mid\gamma,\phi)}\right] + \mathrm E_q\left[\log\frac {q(\theta, \mathbf z\mid \gamma,\phi)}{p(\theta, \mathbf z\mid \mathbf w,\alpha, \beta)}\right]\\
=&\mathrm E_q\left[\log\frac {p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)}{p(\theta, \mathbf z\mid \mathbf w,\alpha, \beta)}\right]\\
=&\mathrm E_q\left[\log\frac {p(\theta, \mathbf z\mid \mathbf w,\alpha, \beta)p(\mathbf w\mid \alpha, \beta)}{p(\theta, \mathbf z\mid \mathbf w,\alpha, \beta)}\right]\\
=&\mathrm E_q\left[\log p(\mathbf w\mid \alpha, \beta)\right]\\
=&\log p(\mathbf w\mid \alpha, \beta)
\end{align}
$$

>  证毕

This shows that maximizing the lower bound $L(\gamma,\phi;\alpha,\beta)$ with respect to $\gamma$ and $\phi$ is equivalent to minimizing the KL divergence between the variational posterior probability and the true posterior probability, the optimization problem presented earlier in Eq. (5). 
>  因此，相对于 $\gamma, \phi$ 最大化对数似然下界 $L(\gamma, \phi; \alpha, \beta)$ 等价于最小化变分后验和真实后验之间的 KL 散度
>  因此我们得到了 Eq 5 的优化问题

We now expand the lower bound by using the factorizations of $p$ and $q$ : 

$$
\begin{align}
L(\gamma, \phi;\alpha, \beta) 
&=\mathrm E_q[\log p(\theta\mid \alpha)]+ \mathrm E_q[\log p( \mathbf z\mid \theta)]+\mathrm E_q[\log p(\mathbf w\mid \mathbf z,\beta)]\\
&\ - \mathrm E_q[\log q(\theta)]-\mathrm E_q[\log q(\mathbf z)]\tag{14}\\
\end{align}
$$

>  我们根据 $p$ 和 $q$ 的分解将该下界展开

>  推导

$$
\begin{align}
L(\gamma, \phi;\alpha, \beta) 
&=\mathrm E_q[\log p(\theta, \mathbf z, \mathbf w\mid \alpha, \beta)] - \mathrm E_q[\log q(\theta, \mathbf z)]\\
&=\mathrm E_q[\log p(\theta\mid \alpha)p( \mathbf z\mid \theta)p(\mathbf w\mid \mathbf z,\beta)] - \mathrm E_q[\log q(\theta)q(\mathbf z)]\\
&=\mathrm E_q[\log p(\theta\mid \alpha)+\log p( \mathbf z\mid \theta)+\log p(\mathbf w\mid \mathbf z,\beta)] - \mathrm E_q[\log q(\theta)+\log q(\mathbf z)]\\
&=\mathrm E_q[\log p(\theta\mid \alpha)]+ \mathrm E_q[\log p( \mathbf z\mid \theta)]+\mathrm E_q[\log p(\mathbf w\mid \mathbf z,\beta)]\\
&\ - \mathrm E_q[\log q(\theta)]-\mathrm E_q[\log q(\mathbf z)]\\\tag{14}
\end{align}
$$

>  推导完毕

Finally, we expand Eq. (14) in terms of the model parameters $(\alpha,\beta)$ and the variational parameters $(\gamma,\phi)$ . Each of the five lines below expands one of the five terms in the bound: 

$$
\begin{align}
L(\gamma, \phi;\alpha, \beta) &= \log \Gamma(\sum_{j=1}^k\alpha_j) - \sum_{i=1}^k\log\Gamma(\alpha_i)+\sum_{i=1}^k(\alpha_i-1)\left(\Psi(\gamma_i)-\Psi\left(\sum_{j=1}^k\gamma_j\right)\right)\\
&+\sum_{n=1}^N\sum_{i=1}^k \phi_{ni} \left(\Psi(\gamma_i)-\Psi\left(\sum_{j=1}^k\gamma_j\right)\right)\\
&+\sum_{n=1}^N\sum_{i=1}^k\sum_{j=1}^V \phi_{ni}w_{n}^j\log \beta_{ij}\\
&-\log \Gamma(\sum_{j=1}^k\gamma_j) + \sum_{i=1}^k\log\Gamma(\gamma_i)-\sum_{i=1}^k(\gamma_i-1)\left(\Psi(\gamma_i)-\Psi\left(\sum_{j=1}^k\gamma_j\right)\right)\\
&-\sum_{n=1}^N\sum_{i=1}^k \phi_{ni} \log\phi_{ni}
\end{align}
$$

where we have made use of Eq. (8). 

>  我们进一步将 (14) 中每个期望展开为关于模型参数 $(\alpha, \beta)$ 和变分参数 $(\gamma, \phi)$ 的形式，得到 Eq (15)，如上所示
>  其中关于 $\theta$ 的期望计算利用了 Eq (8)


>  推导
>  第一个期望为

$$
\begin{align}
\mathrm E_q[\log p(\theta\mid \alpha)]&= \mathrm E_q\left[\log\frac {\Gamma(\sum_{i=1}^k\alpha_i)}{\prod_{i=1}^k\Gamma(\alpha_i)}\prod_{i=1}^k \theta_i^{\alpha_i-1}\right]\\
&=\log \Gamma(\sum_{j=1}^k\alpha_j) - \sum_{i=1}^k\log\Gamma(\alpha_i)+\mathrm E_q\left[\sum_{i=1}^k(\alpha_i-1)\log\theta_i\right]\\
&=\log \Gamma(\sum_{j=1}^k\alpha_j) - \sum_{i=1}^k\log\Gamma(\alpha_i)+\sum_{i=1}^k(\alpha_i-1)\mathrm E_q\left[\log\theta_i\right]\\
&=\log \Gamma(\sum_{j=1}^k\alpha_j) - \sum_{i=1}^k\log\Gamma(\alpha_i)+\sum_{i=1}^k(\alpha_i-1)\left(\Psi(\gamma_i)-\Psi\left(\sum_{j=1}^k\gamma_j\right)\right)\\
\end{align}
$$

>  其中最后一个等号是根据 (8) 展开了期望 $\mathrm E_q[\log \theta_i]$
>  第二个期望为

$$
\begin{align}
\mathrm E_q[\log p(\mathbf z\mid \theta)]
&=\mathrm E_q[\log\prod_{n=1}^N p(z_n\mid \theta)]\\
&=\mathrm E_q[\sum_{n=1}^N\log p(z_n\mid \theta)]\\
&=\sum_{n=1}^N\mathrm E_q[\log p(z_n\mid \theta)]\\
&=\sum_{n=1}^N\mathrm E_q[\log\theta_1^{z_n^1}\cdots\theta_k^{z_n^k}]\\
&=\sum_{n=1}^N\mathrm E_q[\sum_{i=1}^k z_n^{i}\log\theta_i]\\
&=\sum_{n=1}^N\sum_{i=1}^k\mathrm E_q[ z_n^{i}\log\theta_i]\\
&=\sum_{n=1}^N\sum_{i=1}^k\mathrm E_q[ z_n^{i}]\mathrm E_q[\log\theta_i]\\
&=\sum_{n=1}^N\sum_{i=1}^k \phi_{ni} \left(\Psi(\gamma_i)-\Psi\left(\sum_{j=1}^k\gamma_j\right)\right)\\
\end{align}
$$

>  其中 $z_n^i$ 表示的是 $z_n$ 的第 $i$ 个成分
>  第三个期望为

$$
\begin{align}
\mathrm E_q[\log p(\mathbf w\mid \mathbf z, \beta)]&=\mathrm E_q\left[\log \prod_{n=1}^N p(w_n\mid\mathbf z, \beta)\right]\\
&=\mathrm E_q\left[\sum_{n=1}^N\log p(w_n\mid\mathbf z, \beta)\right]\\
&=\sum_{n=1}^N\mathrm E_q\left[\log p(w_n\mid\mathbf z, \beta)\right]\\
&=\sum_{n=1}^N\mathrm E_q\left[\log \beta_{ij}\right]\\
&=\sum_{n=1}^N\sum_{i=1}^k\sum_{j=1}^V \phi_{ni}w_{n}^j\log \beta_{ij}\\
\end{align}
$$

>  第四个期望为

$$
\begin{align}
\mathrm E_q[\log q(\theta)]&= \mathrm E_q\left[\log\frac {\Gamma(\sum_{i=1}^k\gamma_i)}{\prod_{i=1}^k\Gamma(\gamma_i)}\prod_{i=1}^k \theta_i^{\gamma_i-1}\right]\\
&=\log \Gamma(\sum_{j=1}^k\gamma_j) - \sum_{i=1}^k\log\Gamma(\gamma_i)+\mathrm E_q\left[\sum_{i=1}^k(\gamma_i-1)\log\theta_i\right]\\
&=\log \Gamma(\sum_{j=1}^k\gamma_j) - \sum_{i=1}^k\log\Gamma(\gamma_i)+\sum_{i=1}^k(\gamma_i-1)\mathrm E_q\left[\log\theta_i\right]\\
&=\log \Gamma(\sum_{j=1}^k\gamma_j) - \sum_{i=1}^k\log\Gamma(\gamma_i)+\sum_{i=1}^k(\gamma_i-1)\left(\Psi(\gamma_i)-\Psi\left(\sum_{j=1}^k\gamma_j\right)\right)\\
\end{align}
$$

>  第五个期望为

$$
\begin{align}
\mathrm E_q[\log q(\mathbf z)]
&=\mathrm E_q[\log\prod_{n=1}^N q(z_n)]\\
&=\mathrm E_q[\sum_{n=1}^N\log q(z_n)]\\
&=\sum_{n=1}^N\mathrm E_q[\log q(z_n)]\\
&=\sum_{n=1}^N\mathrm E_q[\log\phi_{n1}^{z_n^1}\cdots\phi_{nk}^{z_n^k}]\\
&=\sum_{n=1}^N\mathrm E_q[\sum_{i=1}^k z_n^{i}\log\phi_{ni}]\\
&=\sum_{n=1}^N\sum_{i=1}^k\mathrm E_q[ z_n^{i}\log\phi_{ni}]\\
&=\sum_{n=1}^N\sum_{i=1}^k\mathrm E_q[ z_n^{i}]\mathrm E_q[\log\phi_{ni}]\\
&=\sum_{n=1}^N\sum_{i=1}^k \phi_{ni} \log\phi_{ni}\\
\end{align}
$$

>  其中 $\mathrm E_q[\log \phi_{ni}] = \log \phi_{ni}$ 是因为参数 $\phi_{ni}$ 在变分分布中是给定的，不属于随机变量

In the following two sections, we show how to maximize this lower bound with respect to the variational parameters $\phi$ and $\gamma.$ 
>  之后，我们将讨论如何相对于变分参数 $\phi, \gamma$ 最大化该下界

### A.3.1 Variational Multinomial 
We first maximize Eq. (15) with respect to $\phi_{n i}$ , the probability that the n-th word is generated by latent topic $i$ . Observe that this is a constrained maximization since $\begin{array}{r}{\sum_{i=1}^{k}\phi_{n i}=1}\end{array}$ 
>  首先考虑相对于 $\phi_{ni}$ 最大化 Eq (15)
>  $\phi_{ni}$ 表示了第 $n$ 个单词是由第 $i$ 个隐主题生成的
>  注意此时的最大化问题是一个约束最大化问题，约束为 $\sum_{i=1}^k \phi_{ni} = 1$

We form the Lagrangian by isolating the terms which contain $\phi_{n i}$ and adding the appropriate Lagrange multipliers. 

Let $\beta_{iv}$ be $p(w_{n}^{\nu}=1\,|\,z^{i}=1)$ for the appropriate $v$ . (Recall that each $w_{n}$ is a vector of size $V$ with exactly one component equal to one; we can select the unique $v$ such that $w_{n}^{}=1$ ): 

$$
\begin{array}{r}{L_{[\phi_{n i}]}=\phi_{n i}\left(\Psi(\gamma_{i})-\Psi\left(\sum_{j=1}^{k}\gamma_{j}\right)\right)+\phi_{n i}\log\beta_{iv}-\phi_{n i}\log\phi_{n i}+\lambda_{n}\left(\sum_{j=1}^{k}\phi_{n j}-1\right),}\end{array}
$$
 
where we have dropped the arguments of $L$ for simplicity, and where the subscript $\phi_{n i}$ denotes that we have retained only those terms in $L$ that are a function of $\phi_{n i}$ . 

>  我们从 Eq (15) 中选择出和 $\phi_{ni}$ 相关的项，并添加拉格朗日乘子，以得到拉格朗日函数，如上所示

Taking derivatives with respect to $\phi_{n i}$ , we obtain: 

$$
\frac{\partial L}{\partial\phi_{n i}}=\Psi(\gamma_{i})-\Psi\left(\sum_{j=1}^{k}\gamma_{j}\right)+\log\beta_{iv}-\log\phi_{n i}-1+\lambda.
$$ 
>  将拉格朗日函数对 $\phi_{ni}$ 求导，得到上式

Setting this derivative to zero yields the maximizing value of the variational parameter $\phi_{n i}$ (cf. Eq. 6): 

$$
\begin{array}{r}{\phi_{n i}\propto\beta_{iv}\exp\left(\Psi(\gamma_{i})-\Psi\left(\sum_{j=1}^{k}\gamma_{j}\right)\right).}\end{array}\tag{16}
$$

>  令导数等于零，得到变分参数的最优值如上

>  推导
>  令导数等于零

$$
\begin{align}
0 &= \Psi(\gamma_i) - \Psi\left(\sum_{j=1}^k \gamma_j\right) + \log \beta_{iv} - \log \phi_{ni} - 1 + \lambda\\
\log \phi_{ni} &=\Psi(\gamma_i) - \Psi\left(\sum_{j=1}^k\gamma_j\right) + \log \beta_{iv} - 1 + \lambda\\
\phi_{ni}&=\exp\left\{\Psi(\gamma_i) - \Psi\left(\sum_{j=1}^k\gamma_j\right) + \log \beta_{iv} - 1 + \lambda\right\}\\
\phi_{ni}&=\beta_{iv}\exp\left\{\Psi(\gamma_i) - \Psi\left(\sum_{j=1}^k\gamma_j\right)\right\}\cdot\exp\{ - 1 + \lambda\}\\
\phi_{ni}&\propto\beta_{iv}\exp\left\{\Psi(\gamma_i) - \Psi\left(\sum_{j=1}^k\gamma_j\right)\right\}\\
\end{align}
$$

>  推导完毕

### A.3.2 Variational Dirichlet
Next, we maximize Eq. (15) with respect to $\gamma_{i}$ , the i-th component of the posterior Dirichlet parameter. 
>  我们考虑相对于 $\gamma_i$ 最大化 Eq (15)
>  $\gamma_i$ 表示 Dirichlet 分布的第 $i$ 个参数

The terms containing $\gamma_{i}$ are: 

$$
\begin{array}{r}{L_{[\gamma_i]}=\displaystyle\sum_{i=1}^{k}(\alpha_{i}-1)\left(\Psi(\gamma_{i})-\Psi\left(\sum_{j=1}^{k}\gamma_{j}\right)\right)+\sum_{n=1}^{N}\phi_{n i}\left(\Psi(\gamma_{i})-\Psi\left(\sum_{j=1}^{k}\gamma_{j}\right)\right)}\\ {-\log\Gamma\left(\sum_{j=1}^{k}\gamma_{j}\right)+\log\Gamma(\gamma_{i})-\displaystyle\sum_{i=1}^{k}(\gamma_{i}-1)\left(\Psi(\gamma_{i})-\Psi\left(\sum_{j=1}^{k}\gamma_{j}\right)\right).}\end{array}
$$

>  Eq (15) 中涉及 $\gamma_i$ 的项如上所示

This simplifies to: 

$$
L_{[\gamma_i]}=\sum_{i=1}^{k}\left(\Psi(\gamma_{i})-\Psi\left(\sum_{j=1}^{k}\gamma_{j}\right)\right)\left(\alpha_{i}+\sum_{n=1}^{N}\phi_{n i}-\gamma_{i}\right)-\log\Gamma\left(\sum_{j=1}^{k}\gamma_{j}\right)+\log\Gamma(\gamma_{i}).
$$ 
>  化简，得到上式

We take the derivative with respect to $\gamma_{i}$ : 

$$
\frac{\partial L}{\partial\gamma_{i}}=\Psi^{\prime}(\gamma_{i})\left(\alpha_{i}+\sum_{n=1}^{N}\phi_{n i}-\gamma_{i}\right)-\Psi^{\prime}\left(\sum_{j=1}^{k}\gamma_{j}\right)\sum_{j=1}^{k}\left(\alpha_{j}+\sum_{n=1}^{N}\phi_{n j}-\gamma_{j}\right).
$$ 

>  计算得到 $L$ 相对于 $\gamma_i$ 的偏导数如上
>  $L$ 相对于 $\gamma$ 的梯度就是成分为 $\frac {\partial L}{\partial \gamma_i}$ 的向量

Setting this equation to zero yields a maximum at: 

$$
\begin{array}{r}{\gamma_{i}=\alpha_{i}+\sum_{n=1}^{N}\phi_{n i}.}\end{array}\tag{17}
$$ 
Since Eq. (17) depends on the variational multinomial $\phi$ , full variational inference requires alternating between Eqs. (16) and (17) until the bound converges. 

>  可以知道 $\gamma_i$ 满足 Eq (17) 时，梯度为零
>  因为 Eq (17) 依赖于变分多项式参数 $\phi$，故完整的变分推断需要迭代根据 Eq (16) 和 Eq (17) 更新，直到下界收敛

## A.4 Parameter estimation 
In this final section, we consider the problem of obtaining empirical Bayes estimates of the model parameters $\alpha$ and $\beta$ . We solve this problem by using the variational lower bound as a surrogate for the (intractable) marginal log likelihood, with the variational parameters $\phi$ and $\gamma$ fixed to the values found by variational inference. We then obtain (approximate) empirical Bayes estimates by maximizing this lower bound with respect to the model parameters. 

We have thus far considered the log likelihood for a single document. Given our assumption of exchange ability for the documents, the overall log likelihood of a corpus $D=\left\{\mathbf{w}_{1},\mathbf{w}_{2},\ldots,\mathbf{w}_{M}\right\}$ is the sum of the log likelihoods for individual documents; moreover, the overall variational lower bound is the sum of the individual variational bounds. In the remainder of this section, we abuse notation by using $L$ for the total variational bound, indexing the document-specific terms in the individual bounds by $d$ , and summing over all the documents. 

Recall from Section 5.3 that our overall approach to finding empirical Bayes estimates is based on a variational EM procedure. In the variational E-step, discussed in Appendix A.3, we maximize the bound $L(\gamma,\upphi;\alpha,\beta)$ with respect to the variational parameters $\gamma$ and $\Phi$ . In the M-step, which we describe in this section, we maximize the bound with respect to the model parameters $\alpha$ and $\beta$ . The overall procedure can thus be viewed as coordinate ascent in $L$ . 

A.4.1 C ONDITIONAL multinomial
To maximize with respect to $\beta$ , we isolate terms and add Lagrange multipliers: 

$$
L_{[\beta]}=\sum_{d=1}^{M}\sum_{n=1}^{N_{d}}\sum_{i=1}^{k}\sum_{j=1}^{V}\upphi_{d n i}w_{d n}^{j}\log\beta_{i j}+\sum_{i=1}^{k}\uplambda_{i}\left(\sum_{j=1}^{V}\beta_{i j}-1\right).
$$ 
We take the derivative with respect to ${\beta}_{i j}$ , set it to zero, and find: 

$$
\beta_{i j}\propto\sum_{d=1}^{M}\sum_{n=1}^{N_{d}}\upphi_{d n i}w_{d n}^{j}.
$$ 
A.4.2 D IRICHLET 
The terms which contain $\alpha$ are: 

$$
L_{[\alpha]}=\sum_{d=1}^{M}\left(\log\Gamma\left(\sum_{j=1}^{k}\alpha_{j}\right)-\sum_{i=1}^{k}\log\Gamma(\alpha_{i})+\sum_{i=1}^{k}\left((\alpha_{i}-1)\left(\Psi(\gamma_{d i})-\Psi\left(\sum_{j=1}^{k}\Upsilon_{d j}\right)\right)\right)\right)
$$ 
Taking the derivative with respect to $\alpha_{i}$ gives: 

$$
\frac{\partial L}{\partial\alpha_{i}}=M\left(\Psi\left(\sum_{j=1}^{k}\alpha_{j}\right)-\Psi(\alpha_{i})\right)+\sum_{d=1}^{M}\left(\Psi(\gamma_{d i})-\Psi\left(\sum_{j=1}^{k}\gamma_{d j}\right)\right)
$$ 
This derivative depends on $\alpha_{j}$ , where $j\neq i$ , and we therefore must use an iterative method to find the maximal α . In particular, the Hessian is in the form found in Eq. (10): 
$$
\frac{\partial L}{\partial\alpha_{i}\alpha_{j}}=\updelta(i,j)M\Psi^{\prime}(\alpha_{i})-\Psi^{\prime}\left(\sum_{j=1}^{k}\alpha_{j}\right),
$$ 
and thus we can invoke the linear-time Newton-Raphson algorithm described in Appendix A.2. 
Finally, note that we can use the same algorithm to find an empirical Bayes point estimate of $\boldsymbol\upeta$ , the scalar parameter for the exchangeable Dirichlet in the smoothed LDA model in Section 5.4. 
