# Abstract

目前主导的序列转换模型(sequence transduction model)都基于复杂的循环和卷积神经网络，包含了一个编码器和一个解码器，其中表现最好的模型通过注意力机制连接了编码器和解码器

我们提出 Transformer 架构，仅仅基于注意力机制，不在需要循环机制(recurrence)和卷积计算(convolutions)
在两个机器翻译任务上的实验表明该模型具有更高的并行性以及需要更少的训练时间

我们的模型在 WMT 2014 英语到德语翻译任务中取得了 28.4BLEU，在 WMT 2014 英语到法语翻译任务中，取得了 41.0BLEU，均为 SOTA

# 1 Introduction

RNN，LSTM 以及 GRU 网络在序列建模问题上以及转换问题上(sequence modeling and transduction)，例如语言建模和机器翻译，取得了很好的结果
许多工作都在推进循环语言模型(recurrent language model)和编码器-解码器架构

循环模型通常沿着输入和输出序列的符号位置(symbol positions)进行计算，模型将位置与计算时间步对齐(aligning the positions to steps in computation time)，生成一系列的隐藏状态(hidden states)$h_t$，$h_t$是关于前一个隐藏状态$h_{t-1}$和位置$t$上的输入的函数
这种内在的顺序本质妨碍了训练样本内的并行(parallelization within traning examples)，在序列较长时，内存容量会限制了样本的批量处理
问题的本质来源于顺序计算(sequential computation)

注意力机制允许在无视输入或输出序列之间的内部距离的情况下对依赖建模(dependencies)，一些工作将注意力机制与循环网络一起使用

在该工作中，我们提出 Transformer，该架构避免了循环结构，完全依赖于注意力机制以提取输入和输出之间的全局依赖(global dependencies)

# 2 Background

Extended Neural GPU，ByteNet，ConvS2S 的目标也是减少顺序计算(sequential computation)，它们使用 CNN 作为基础的构成要素(building block)，为所有的输入和输出位置计算隐藏表征(hidden representations)
在这些模型中，要将两个任意的输入或输出位置的信号(signals)联系的操作数随着这两个位置的距离增长，ConvS2S 为线性，ByteNet 为对数
这也使得要学习相距较远的位置之间的依赖更加困难

在 Transformer 中，只需要常数级别的操作数，尽管代价是减少了有效分辨率(reduced effective resolution)(因为要对注意力加权的位置进行平均/due to averaging attention-weighted positions)，但我们用多头注意力对该效果进行抵消(an effect we counteract with Multi-Head Attention)

自注意力，有时称为内部注意力(intra-attention)是关联单个序列内部的不同位置以计算该序列的表征(representations)的机制，
自注意力在阅读理解(reading comprehension)、抽象概括(abstractive summarization)、文字蕴涵(textual entailment)和学习任务独立(task-independent)的句子表征成功应用

端到端的内存网络(End-to-end memory networks)是基于循环注意力机制(recurrent attention mechanism)，而非序列对齐的循环(sequence aligned recurrence)，在简单语言问答任务(simple-language question answering)和语言建模任务中表现出色

Transformer 是第一个完全依赖于自注意力来为输入和输出计算表征的转换模型(transduction model)，没有使用序列对齐的 RNNs 或卷积

# 3 Model Architecture

大多数序列转换模型都具有编码器-解码器结构，
编码器将输入的符号表征序列(input sequence of symbol representations)$(x_1,\dots, x_n)$映射到连续表征序列(continuous representation sequence)$\mathbf z = (z_1, \dots, z_n)$

给定$\mathbf z$，解码器生成一个输出符号序列$(y_1,\dots, y_m)$，每次生成其中一个元素，在每一步，模型都是自回归的，即在生成文本时，利用的之前生成的符号作为额外的输入

Transformer 遵循该整体架构，编码器和解码器都使用了堆叠的自注意力和逐点的全连接层(stack self-attention and point-wise, fully connected layers)

## 3.1 Encoder and Decoder Stacks

**Encoder:** 编码器由$N=6$个完全相同的层堆叠而成，每一层都有两个子层(sub-layers)，第一个子层是一个多头自注意力机制，第二个子层是一个简单的，逐位置的全连接前向网络(position-wise fully connected feed-forward network)，两个子层都采用残差连接，残差连接后是一个层规范化(layer normalization)
即每个子层的输出都可以写为$\text {LayerNorm}(x + \text{Sublayer}(x))$，其中$\text{Sublayer}(x)$就是子层本身执行的函数
为了方便(facilitate)这些残差连接，模型中所有的子层，包括了嵌入层(embedding layers)的输出维度都是$d_{model} = 512$

**Decoder:** 解码器也由$N=6$个完全相同的层堆叠而成，除了有编码器层中的两个子层外，解码器还插入了第三个子层，在编码器栈的输出上执行多头注意力，和编码器类似，解码器的每个子层都采用残差连接，后接一个层规范化
我们也修改了解码器栈中的自注意力子层，防止注意到当前位置之后的位置，这种掩码机制(masking)，结合输入嵌入是偏移了一个位置的事实，保证了对位置$i$的预测只能依赖于位置小于$i$的已知的输出

## 3.2 Attention

一个注意力函数可以被描述为将一个查询和一组键值对映射到一个输出(mapping a query and a set of key-value pairs to an ouput)，其中查询、键、值和输出都是向量
输出通过值的加权求和计算，其中每个值的权重都根据查询和对应的键的相容性函数计算(compability function)

### 3.2.1 Scaled Dot-Product Attention

我们称使用的注意力为缩放点积注意力，
输入由维度是$d_k$的查询和键，以及维度是$d_v$的值组成，我们计算查询和所有键的点积，将结果除以$\sqrt d_k$，然后应用 softmax 函数以得到值的权重

在实际中，我们对一组查询同时计算注意力函数，打包为一个矩阵$Q$，值和键也打包为矩阵$K$和$V$，函数的矩阵形式写为：
$$\text{Attention}(Q,K,V) = \text{softmax}(\frac {QK^T}{\sqrt d_k})V\tag{1}$$
最常用的两个注意力函数是加性注意力(additive)和点积(乘性/multiplicative)注意力，点积注意力和我们的算法相同，除了缩放因子$\frac 1 {\sqrt d_k}$

加性注意力用带一个隐藏层的前向网络计算相容性函数

加性和乘性注意力在理论复杂度上是相似的，但点积注意力在实际中会更快和更节省空间(spece-efficient)，因为它通过高度优化的矩阵乘法代码实现

当$d_k$较小时，两种机制表现相似，在$d_k$较大时，加性注意力的表现由优于点积注意力，如果点积注意力没有进行缩放
我们认为当$d_k$较大时，点积在数量级上变大，将 softmax 函数推向了梯度及其小的区域，为了防止这种效果，我们将点积缩放为原来的$\frac 1 {\sqrt d_k}$

要阐述为什么点积会变大，假设$q$和$k$的成分都是均值为$0$，方差为$1$的独立随机变量，则二者的点积$q\cdot k = \sum_{i=1}^{d_k} q_ik_i$的均值是$0$，方差是$d_k$

### 3.2.2 Multi-Head Attention

我们没有选择用$d_{model}$维的键、值和查询仅执行单个注意力函数，
我们发现用学习到的不同的线性映射，将查询、键和值线性映射到$d_k,d_k$和$d_v$维$h$次是有益的
对于每一个版本的线性映射后的查询、键和值，我们并行执行注意力计算，得到多个$d_v$维的输出值，这些输出值被拼接在一起，然后再进行一次线性映射，得到最终的值

多头注意力允许模型共同地注意到在不同位置来自不同表征子空间的信息(attend to information from different representation subspaces at differnet positions)，如果只用单头注意力，平均化就掩盖了这些信息(averaging inhibits this)

$$
\begin{align}
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O\\
\textbf{where}\ \text{head}_i=\text{Attention}(QW_i^Q,KW_k^K,VW_i^V)
\end{align}
$$

其中映射就是参数矩阵$W_i^Q \in \mathbb R^{d_{model}\times d_k},W_i^K \in \mathbb R^{d_{model}\times d_k},W_i^V \in \mathbb R^{d_{model}\times d_v}$以及$W^O\in \mathbb R^{hd_v \times d_{model}}$

本工作中，我们采用$h=8$的并行注意力层(或头)，对于每一个注意力头，我们采用$d_k = d_v = d_{model}/h = 64$
由于每个头内的维度减少了，整体的运算开销和单个具有完全维度的注意力头相似

### 3.2.3 Applications of Attention in our Model

Transformer 在三种不同的地方使用多头注意力：

- 在”编码器-解码器“注意力层中，查询来自于前一个解码器层，而存下的键和值(memory keys and values)来自于编码器的输出，这允许解码器中的每个位置都可以注意到输入序列的全部位置
- 编码器包含了自注意力层，自注意力层中，所有的键、值和查询都来自同一地方，即编码器前一层的输出，编码器中的每个位置都可注意到编码器前一层的全部位置
- 类似地，解码器中的自注意力层允许解码器中的每个位置都可以注意到该位置之前(包括该位置)的全部位置，在缩放点积注意力中，我们通过将向 softmax 函数的输入中的违法位置的值掩蔽(设定为$-\infty$)对此进行实现

## 3.3 Potision-wise Feed-Forward Networks

除了注意力子层外，编码器和解码器中的每一层还包含了一层全连接前向网络，该子层对每个位置分别且同等地应用(applied to each position seperately and identically)，该子层内部包含了两次线性变换和其中的一次 ReLU 激活：
$$\text{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2\tag{2}$$
对于不同的位置，线性变换是相同的，对于不同的层，线性变换的参数是不同的
全连接前向网络子层的输入和输出的维度都是$d_{model} = 512$，中间隐藏层的维度是$d_{ff} = 2048$

## 3.4 Embeddings and Softmax

和其他的序列转换模型类似，我们使用学习到的嵌入来将输入词元(tokens)和输出词元转换为维度为$d_{model}$的向量，我们也用学习到的线性变换和 softmax 函数将解码器输出转换为预测的下一个词元概率(next-token probabilities)

在我们的模型中，两个嵌入层(输入/输出嵌入层)以及 softmax 的线性变换层共享权重，在嵌入层，权重会被乘上$\sqrt d_{model}$
(倒不如说在做 softmax 前的线性变换时会将点积结果都除以$\sqrt d_{model}$，降低数量级，防止梯度太小；也可以说是让嵌入的数量级相对大一点，避免被位置编码掩盖)

解释：softmax 前的线性变换层和嵌入层使用相同的权重，就是说输入词向量在嵌入层寻找和它内积最大的词的词嵌入作为输入词向量对应的词

## 3.5 Positional Encoding

我们模型不包含循环和卷积，为了让我们的模型利用序列的顺序(order)信息，我们必须注入一些关于序列中符号的相对或绝对位置的信息

我们为在编码器和解码器栈的底端的输入嵌入加上”位置编码“，位置编码的维度和嵌入的维度一样，都是$d_{model}$，因此可以相加
位置编码由许多选择，可以是学习到的，也可以是固定的

本项工作中，使用不同频率的 sine 和 cosine 函数：

$$
\begin{align}
PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})\\
PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})\\
\end{align}
$$

其中$pos$是位置，$i$是维度，因此，位置编码的每个维度都对应一个正弦曲线(sinusoid)，正弦曲线的波长从$2\pi$(维度$1$)逐渐增长到$2000\cdot 2\pi$(维度$d_{model}$)，因此位置编码的维度越高，$pos$变化产生的差异越小

我们选择该函数的原因在于我们假设这会让模型更容易学习到注意相对位置(learn to attend by relative positions)，因为对于一个固定的$k$，$PE_{pos+k}$可以被表示为$PE_{pos}$的一个线性函数

解释：假设$d_{model} = 2$

$$
PE_{pos} = \begin{bmatrix}
sin(\frac {pos} {c})\\
cos(\frac {pos} {c})
\end{bmatrix}
\quad
PE_{pos+k} =
\begin{bmatrix}
sin(\frac {pos+k}{c})\\
cos(\frac {pos+k}{c})
\end{bmatrix}
$$

其中$c = 10000^{0/d_{model}} = 1$对于$pos$和$k$是常数
显然有$$PE_{pos+k} = 
\begin{bmatrix}
sin(\frac {pos}{c} + \frac k {c})\\
cos(\frac {pos}{c} + \frac k {c})
\end{bmatrix}=
\begin{bmatrix}
sin(\frac {pos}{c})cos(\frac k {c}) + cos(\frac {pos}{c})sin(\frac k {c}))\\
cos(\frac {pos}{c})cos(\frac k {c}) - sin(\frac {pos}{c})sin(\frac k {c}))\\
\end{bmatrix}$$
其中第二个等号是因为两角和/差的正/余弦公式
令$\frac k {c} = t$，因为$k$固定，$t$是常数
因此有$$PE_{pos+k} = \begin{bmatrix}
cos(t)&sin(t)\\
-sin(t)&cos(t)
\end{bmatrix}PE_{pos}$$即$PE_{pos+k}$是$PE_{pos}$的线性函数

我们也实验了使用学习到的位置嵌入，发现两种位置编码产生的结果几乎一致，我们选择三角函数的版本，因为它可以允许模型对长于它在训练时遇到的序列长度的序列的位置编码进行推断(extrapolate to)

# 4 Why Self-Attention

本节中，我们将从多个方面将自注意力层与循环层和卷积层进行比较，
之前的工作中，循环层和卷积层都常作为一个典型的序列转换编码器或解码器中的一个隐藏层，用于将一个变长的符号表征序列$(x_1,\dots,x_n)$映射到另一个等长的序列$(z_1,\dots,z_n)$，其中$x_i,z_i\in \mathbb R^d$

我们考虑三个方面：
第一个方面就是每层总的计算复杂度，另一个是可以被并行化的计算量，我们用所需要的顺序运算的最少数量来衡量(the minumum number of sequential operations required)
第三个网络中长范围(long-range)依赖的路径长度(path length)，
学习长范围的依赖是许多序列转换任务中的关键挑战，影响对这类依赖的学习能力的关键因素就是信号在网络中要向前或向后穿过的路径的长度
输入/输出序列中，任意位置的组合之间的路径长度越短，就越容易学到长范围依赖，因此我们会比较不同类型的层中输入/输出中任意两个位置之间的最大路径长度(maximum path length)

如 Table1 所示，自注意力层连接了所有的位置，使用了常数次的顺序执行的运算，而循环层需要$O(n)$的顺序运算；在计算复杂度方面，自注意力层在序列长度$n$小于表征维度$d$时，会比循环层更快，而这种情况是很常见的，在 SOTA 的机器翻译模型中，句子表征(sentence representations)例如 word-piece 表征和 byte-pair 表征常常被使用；要提高自注意力层对非常长的序列的计算性能，可以限制自注意力只考虑以输出位置为中心，左右邻近的$r$个输入位置，但这会将最大路径长度增长至$O(n/r)$

一个卷积核宽度$k<n$的单卷积层不会将输入和输出所有的位置对(pairs of)连接起来，如果要这么做，需要用连续的$O(n/k)$个卷积核，使用膨胀卷积时，则是$O(log_k(n))$个
卷积层往往比循环层更昂贵，要乘上因子$k$，可分离卷积(Separable)将复杂度降为$O(k\cdot n \cdot d + n\cdot d^2)$，在$k=n$时，可分离卷积层的计算复杂度等价于我们模型中的自注意力层加上前向网络层

自注意力具有更好的可解释性，我们在附录中审查了模型中的注意力分布(attention distribution)，发现不仅每个注意力头清楚地学习到了执行不同的任务，而且很多注意力头都展示了和句子中语法和语义结构相关的行为

# 5 Training

## 5.1 Traning Data and Batching

我们在标准 WMT 2014 英语到德语数据集上训练，该数据集由 450 万个句子对(sentence pairs)组成，句子用 byte-pair 编码，得到一个包含了 37000 个词元(tokens)的共享的源-目标词袋(source-target vocabulary)
对于英语-法语任务，我们使用更大的 WMT 2014 英语到法语数据集，该数据集由 3600 万个句子组成，被划分为包含了 32000 个词元的 word-piece 词袋

相似长度的句子对会被放在同一个批量中，每个训练批量包含了一组句子对，大约有 25000 个源词元和 25000 个目标词元

## 5.2 Hardware and Schedule

在 8 张 NVIDIA P100 GPUs 上训练，对于基模型，每个训练步大约需要 0.4 秒，我们总共训练 10 万个训练步共 12 个小时，对于大模型，每个训练步需要 1 秒，我们总共训练 30 万个训练步共 3.5 天

## 5.3 Optimizer

我们使用 Adam 优化器，$\beta_1 = 0.9,\beta_2 = 0.98,\epsilon=10^{-9}$，在训练过程中，我们根据以下公式变化学习率：
$$lrate = d_{model}^{-0.5}\cdot\min(step\_num^{-0.5},step\_num*warmup\_steps^{-1.5})\tag{3}$$
这对应的是在最开始的$warmup\_steps$线性增加学习率，然后按照步数的反平方根的比例减少
我们使用$warmup\_steps = 4000$

解释：
当$step\_num \le warmup\_steps$，有
$step\_num^{-0.5}  = step\_num * step\_num^{-1.5} \ge step\_num * warmup\_steps^{-1.5}$
此时学习率$lrate$为$d_{model}^{-0.5}*step\_num*warmup\_steps^{-1.5}$
显然$lrate$随着$step\_num$线性增加

当$step\_num > warmup\_steps$，有
$step\_num^{-0.5}  = step\_num * step\_num^{-1.5} < step\_num * warmup\_steps^{-1.5}$
此时学习率$lrate$为$d_{model}^{-0.5}*step\_num^{-0.5}$
显然$lrate$随着$step\_num$增大而减小，减小的速度即$step\_num$的开根号的倒数($1/{\sqrt {step\_num}})$减小的速度

## 5.4 Regularization

我们在训练中采用三种类型的正则
**Residual Dropout** 我们对每个子层的输出进行丢弃(dropout)，丢弃是在残差相加和规范化之前，另外我们也对编码器栈和解码器栈的输入，即嵌入和位置编码的和进行丢弃
对于基模型，我们使用丢弃率$P_{drop} = 0.1$

**Label Smoothing** 在训练时，我们采用标签平滑，$\epsilon_{ls} = 0.1$，这会损害困惑度(perplexity)，因为模型会学着变得更加不确定(learns to be more unsure)，但会提高准确率和 BLEU 分数

# 6 Results

## 6.1 Machine Translation

在 WMT 2014 英语到德语翻译任务上，大模型比之前最好的模型的 BLEU 高了 2.0，达到了 28.4，即便是基模型也好于之前所有的模型

在 WMT 2014 英语到法语翻译任务上，大模型的 BLEU 为 41.0，超过了之前所有的单模型，在该任务上训练的大模型使用的丢弃率$P_{drop} = 0.1$而不是$0.3$

对于基模型，我们使用的模型来自于对最后 5 个检查点(checkpoints)进行平均，检查点每 10 分钟写一次，对于大模型，我们对最后 20 个检查点取平均

我们使用束搜索(beam search)，束长度(beam size)是 4，长度惩罚因子(length penalty)$\alpha=0.6$

以上超参数都是根据开发集(development set)上的实验结果选择的

我们设置推理时最大的输出长度是输入长度$+50$，但在可能时尽早结束(terminate early when possible)

我们用训练时间乘以使用的 GPU 数量乘以每个 GPU 的持续单精度浮点能力(sustained single-precision floating-point capacity)来估计训练一个模型需要的浮点运算数量

## 6.2 Model Variations

为了评估 Transformer 模型不同组件的重要性，我们用不同的方式改变我们的基模型，观察在英语到德语翻译数据集的开发集 newstest2013 上任务表现的改变
我们使用之前提到的束搜索，但没有进行检查点平均

Table3 的 A 行中，我们改变注意力头的数量和注意力键和值的维度，保持总的计算量不变，发现只有单个注意力头的 BLEU 比最优设定少 0.9，而注意力头数量过多也会导致表现下降

Table3 的 B 行中，我们发现减少注意力键大小$d_k$会损害模型质量，这说明决定相容性不是一个容易的任务，因此比点积更复杂的相容性函数可能是有益的

Table3 的 C 和 D 行中，我们发现越大的模型越好，以及丢弃对避免过拟合非常有帮助

Table3 的 E 行中，我们将正弦位置编码替换为学习到的位置编码，发现在基模型上的结果几乎一样

# 7 Conclusion

我们提出 Transformer，首个完全基于注意力的序列转换模型，将常用于编码器-解码器架构的循环层替换为多头注意力

对于翻译任务，Transformer 在翻译任务上的训练速度显著快于基于循环或卷积层的架构

我们计划将 Transformer 拓展至输入和输出模态非文本的问题，并且研究局部、受限制的注意力机制，以高效处理大的输入和输出，例如图像、声音和视频，我们的另一个目标是让生成任务更不顺序化(making generation less sequential)
