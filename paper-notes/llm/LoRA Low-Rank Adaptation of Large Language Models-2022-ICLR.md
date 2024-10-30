# Abstract
NLP的一个重要范式是在大规模的通用领域数据上训练，然后对特定领域或任务微调，随着预训练模型的增大，完全的微调(full fine-tuning)，即重训练所有的模型参数显得较不可行，例如GPT-3 175B，对于每一个用于微调模型的样本，重训练所有175B的参数是不现实的

我们提出低秩适应，即LoRA，它固定住预训练好的模型参数，对Transformer结构中的每一层注入可训练的秩分解矩阵(trainable rank decompoisition matrices)，以显著减少对下流任务微调时可训练参数的数量

相较于用Adam微调GPT-3 175B，LoRA可以减少可训练参数的数量10000倍，减少GPU显存需求3倍，且在RoBERTa、DeBERTa、GPT-2、GPT-3上使用LoRA的效果要优于或等价于微调模型的效果，使用LoRA也不会引入额外的推理延迟(inference latency)

我们也对语言模型适应中的秩缺失(rank deficiency)进行了经验性研究，以说明LoRA的有效性
# 1 Introduction
NLP许多应用依赖于使一个大规模预训练的语言模型适应多个下流任务，常用的适应方式是通过微调，微调会更新预训练模型中的所有参数
微调的主要缺陷在于新模型包括了和旧模型同数量的参数

许多工作通过仅调节部分参数或学习新的模型外部的模块(external modules)对此进行缓解，在该方式下，对于每个任务，我们只需要存储和装载数量较小的针对特定任务(task-specific)的参数

但现存的方法会引入推理延迟(因为拓展了模型的深度和减短了模型的可用序列长度 usable sequence length)，更重要的是这些方法的性能难以匹配微调的效果，使得我们不得不在效率和模型质量上权衡

我们认为学习到的过参数化(over-parametrized)的模型实际上存在于低的内在维度上(low intrinsic dimension)，我们假设模型自适应(model adaptation)时权重的改变也有低的内在维度，即低秩适应

LoRA允许我们通过优化神经网络中一些全连接层在自适应时的改变(change)的秩分解矩阵(rank decomposition matrices)，以间接地重训练这些全连接层，同时保持预训练的权重不变，如图1所示
![[LoRA-Fig1.png]]
以GPT3-175B为例，我们将展示LoRA中的秩分解矩阵的秩可以非常低(即图1中的$r$可以是1或2)，即便满秩(即图1中的$d$)会高到12288，这使得LoRA同时是计算高效和存储高效的

LoRA有几大关键优势：
- 一个预训练好的模型可以通过对不同的任务构建不同的小的LoRA模块以适应不同的下游任务，切换任务时，不需要改变模型，只需要替换矩阵$A$和$B$，这显著减少了存储需求和任务切换开销
- LoRA使得训练更加高效，我们不再需要为大多数参数计算梯度或维护优化器状态(optimizer states)，我们只需要优化执行注入的低秩矩阵
- LoRA简单的线性设计允许我们在部署时将可训练矩阵和固定的权重融合，因此不会引入推理延迟
- LoRA和许多之前的方法正交，因此可以和这些方法结合，例如前缀微调(prefix-tuning)
# 2 Problem Statement
LoRA对于训练目标是不可知的，而本项工作中我们聚焦于语言建模问题

假设我们给定一个预训练好的自回归语言模型$P_{\Phi}(y|x)$，模型参数为$\Phi$，$P_{\Phi}(y|x)$可以是一个泛用的多任务模型例如GPT，
考虑使该预训练模型适应下游的条件文本生成任务(conditional text generation)，例如文本总结、机器阅读理解(MRC machine reading comprehension)、自然语言到SQL转换(NL2SQL)，每个下游任务都由一个包含了上下文-目标对(context-target pairs)的训练数据集$\mathcal Z = \{(x_i,y_i)\}_{i=1,\dots,N}$表示，其中$x_i,y_i$都是词元序列

例如，NL2SQL任务中，$x_i$是自然语言查询(query)，$y_i$是对应的SQL指令；文本总结任务中，$x_i$是文章内容，$y_i$是它的总结

在完全微调中，模型按照预训练权重$\Phi_0$初始化，通过梯度上升不断更新$\Phi_0 + \Delta \Phi$，以最大化条件语言建模目标(conditional language modeling objective)：
$$\max_{\Phi}\sum_{(x,y)\in \mathcal Z}\sum_{t=1}^{|y|}\log (P_{\Phi}(y_t|x,y_{<t}))\tag{1}$$
完全微调的主要缺陷就是对于每个下流任务，都需要学习不同的$\Delta \Phi$，其维度$|\Delta \Phi|$等于$|\Phi|$，对于GPT-3，$|\Phi|=$ 175B，完全微调不可行

我们提出的方法中，针对特定任务的参数增量$\Delta \Phi = \Delta \Phi(\Theta)$，即我们用更小规模的参数集$\Theta$($|\Theta| \ll |\Phi_0|$)表示$\Delta \Phi$，找到$\Delta \Phi$的任务转变为对$\Theta$优化：
$$\max_{\Theta}\sum_{(x,y)\in \mathcal Z}\sum_{t=1}^{|y|}\log (p_{\Phi_0 + \Delta \Phi(\Theta)}(y_t|x,y_{<t}))\tag{2}$$
之后的部分中，我们提出用低秩表示(low-rank representation)来编码(encode)$\Delta \Phi$，在GPT-3 175B上，参数规模$|\Theta|$可以是$|\Phi_0|$的$0.01\%$
# 3 Aren't Existing Solutions Good Enough ?
随着迁移学习概念的出现，已经有许多工作目标于使得模型自适应更参数高效和计算高效(parameter- and compute-efficient)
对于语言模型，主要有两种高效自适应的方法：添加适配器层(adapter layer)以及优化某种形式的输入层激活(input layer activations)，但这两种方法在大规模(large scale)以及延迟敏感(latency-sensitive)的生产环境下都存在缺陷

**Adapter Layers Introduct Inference Latency**
Houlsby等的最初的适应器设计中，每个Transformer块都有两个适应器层，Lin的设计中，每个块则只有一个适应器层，以及一个额外的LayerNorm层

显然，额外的适应器层引入了额外的计算开销，虽然适应器层的瓶颈维度(bottleneck dimension)一般都比较小，因此其参数数量也远远小于原模型，但大规模的神经网络依赖于硬件并行以保持低的延迟，但适应器层只能顺序计算，这使得在线推理时(此时batch大小就是1)的延迟会有差异
在没有模型并行(model parallelism)的通用场景下，例如用单个GPU在GPT-2 medium上推理时，我们观察到使用适配器层显著增长了延迟，即便适配器层的瓶颈维度很小，如表1所示

且该问题会在我们分片模型(shard model)时变得更加严重，因为额外的深度需要更多的同步GPU操作(synchronous GPU operations)，例如 `AllReduce` 和 `Broadcast` ，除非我们冗余存储适配器参数多次

**Directly Optimization the Prompt is Hard**
Li&Liang采用的前缀微调(prefix tuning)同样面临挑战，我们发现前缀微调难以优化，且它的表现不随着可训练参数单调改变，
更根本的是，将序列长度的一部分保留给适应必然会减少可用于处理下游任务的序列长度，我们怀疑这使得微调提示(tuning the prompt)的性能差于其他方法
# 4 Our Method
我们描述LoRA的设计以及它的优势，LoRA方法可以应用于深度神经网络中的任何线性层，在这里我们聚焦于Transformer语言模型中权重
## 4.1 Low-Rank-Parameterized Update Matrices
一个神经网络中包含了许多的线性层执行矩阵乘法，这些层的参数矩阵一般都是满秩的
在适应到特定的任务时，Aghajanyan展示了预训练的语言模型有低的“内在维度(intrisic dimension)"，且即使被随机映射到一个更小的子空间，也仍可以高效地学习，我们受此启发，假设模型在适应时的权重的更新也有低的“内在秩(intrisic rank)”

对于一个预训练好的权重矩阵$W_0\in \mathbb R^{d\times k}$，我们对它的更新进行约束，将它的更新表示为一个低秩分解的形式(low-rank decomposition)：$W_0 + \Delta W = W_0 + BA$，其中$B\in \mathbb R^{d\times r}, A\in \mathbb R^{r\times k}$，其中秩$r\ll \min(d,k)$

在训练时，$W_0$固定，不进行梯度更新，$A,B$包含了可训练的参数，注意到$W_0$和$\Delta W = BA$都和相同的输入相乘，它们输出向量最后按元素相加，因此对于模型$h = W_0x$，权重更新后的前向传播写为：
$$h = W_0x + \Delta W x = W_0x + BAx\tag{3}$$
我们使用随机Guassian初始化$A$，$B$则初始化为0，因此$\Delta W = BA$在训练开始时是0
实际实现中，我们会按$\frac {\alpha} r$放缩$\Delta Wx$，其中$\alpha$相对于$r$是常数
使用Adam优化时，如果我们适当地放缩了初始化(scale the initialization appropriately)，调节$\alpha$和调节学习率是大致相同的
我们简单地将$\alpha$的值设置为我们第一次尝试的$r$，之后就不再调节它
该放缩方案帮助减少了当我们改变$r$时对重调超参数的需要(need to retune hyperparameters)

**A Generalization of Full Fine-tuning**
一种更广义形式的微调允许仅训练预训练参数的一个子集，LoRA在其基础上更进一步，不再要求在适应时对权重矩阵的累计梯度更新(accumulated gradient update)是满秩的
这意味着当我们将LoRA应用于所有权重矩阵并训练所有偏置(biases)，通过将LoRA秩$r$设置为预训练权重矩阵的秩，就可以大致恢复完全微调的表现力(expressiveness)

换句话说，随着我们增加可训练参数的数量时，训练LoRA大致收敛到训练原始模型，而基于适配器的方法则收敛到MLP，基于前缀的方法则收敛到一个不能接受长输入序列的模型

**No Additional Inference Latency**
在部署到生产环境中，我们可以显式计算并存储$W = W_0 + BA$，推理方式无需变化
注意$W_0$和$BA$都在$\mathbb R^{d\times k}$中，因此在切换下流任务时，只需要减去$BA$恢复$W$，再加上$B'A'$即可，该操作很迅速，且内存开销小
这保证了我们不引入任何额外的推理延迟
## 4.2 Applying LoRA to Transformer
原则上，我们可以对神经网络中权重矩阵的任意子集应用LoRA以减少可训练参数的数量

Transformer架构中，在自注意力模块(self-attention module)内存在四个权重矩阵$(W_q,W_k,W_v,W_o)$，在MLP模块内存在两个权重矩阵
我们将$W_q,W_k,W_v$都视作一个维度为$d_{model}\times d_{model}$的矩阵，在前向传播时注意它们的输出会被切分为多个注意力头，但与该实现并不冲突

我们聚焦于针对下游任务调节注意力权重，固定MLP模块，以追求简单和参数效率，对于调节MLP层、LayerNorm层，以及偏置的研究留待日后

**Practical Benefits and Limitations**
LoRA最大的优势来自于减少了内存和存储的使用，对于以Adam训练的大型Transformer来说，我们在$r\ll d_{model}$的情况下可以减少$2/3$的VRAM使用，因为我们不需要为固定的参数存储优化器状态(optimizer states)

在GPT-3 175B上，我们令$r=4$，且仅调节$W_q$和$W_v$，将训练时的VRAM开销从1.2TB减少到了350GB，检查点(checkpoint)的大小减少了大约10000倍(从350GB到35MB)，这允许我们用更少的GPUs训练，同时避免I/O瓶颈

LoRA的另一个优势在于我们可以在部署时以极低的开销快速切换任务，因为只需要切换LoRA权重而不是所有的模型参数

在训练GPT-3 175B时，我们观察到相较于完全微调，LoRA的速度提升了25%(完全微调的训练吞吐量是32.5 tokens/s per V100 GPU，在模型并行的权重分片相同weight shards for model parallelsim的情况下，LoRA的吞吐量是43.1 tokens/s per V100 GPU)，因为我们并不需要为大量的参数计算梯度

LoRA也存在限制，例如LoRA不便于处理面向多个任务的批输入(batch inputs to different tasks)，即无法在一次前向传播中使用不同的$A$和$B$(如果我们将$A$和$B$融入权重矩阵$W$以消除额外的推理延迟)，
但在延迟不是关键的情况下，可以不将LoRA权重和$W$融合，而是动态地根据输入选择LoRA模块
# 5 Empiricial Experiments
我们的实验涵盖了大量的任务，从自然语言理解(NLU natural language understanding)到自然语言生成(NLG natural language generation)
具体地说，我们对RoBERTa和DeBERTa评估了GLUE基准，我们遵循Li&Liange在GPT-2上的设定，进行直接的比较，同时为GPT-3上的大规模实验加上了WikiSQL(NL to SQL查询)和SAMSum(对话总结)任务
所有的实验均使用NVIDIA Tesla V100
## 5.1 Baselines
为了广泛比较其他基准方法，我们尽可能重用前人工作所报告的数字，同时复制它们的设定(setups)，这也意味着一些基准方法可能仅出现在特定实验中

**Fine-Tuning(FT)**
微调是常用的适应技术，在微调时，模型被初始化为预训练好的权重和偏置，然后所有的模型参数都进行梯度更新，一个简单的变体是仅更新部分层，固定其他层，例如Li&Liang在GPT-2上所做的工作，即仅调节最后两层($\text{FT}^{\text{Top2}}$)

**Bias-only or BitFit**
该基准方法中，我们仅训练偏置向量(bias vectors)，固定其他所有参数，该基准方法目前也被Zaken的BitFit所研究

**Prefix-embedding tuning(PreEmbed)**
该基准方法在输入tokens中插入了特殊的tokens，这些特殊的tokens的词嵌入是可训练的，且通常不在模型的词表中，这些tokens放置的位置会影响表现，我们聚焦于前缀放置(prefixing)，即将这些tokens插在prompt之前，以及中缀放置(infixing)，即将这些tokens插在prompt之后
我们用$l_p$/$l_i$表示前缀和中缀的tokens数量，则可训练的参数数量就是$|\Theta| = d_{model} \times (l_p + l_i)$

**Prefix-layer tuning(PreLayer)**
该方法是对前缀嵌入调节的延伸，除了为特定的tokens学习词嵌入(或者可以说学习嵌入层之后的激活 activations after the embedding layer)以外，还学习每个Transformer层之后的激活，由之前的层所计算出的激活会简单地被可训练的激活替换
最后的可训练参数数量是$|\Theta| = L\times d_{model}\times (l_p + l_i)$，其中$L$是Transformer层的数量

**Adapter tuning**
由Houlsby提出，在自注意力模块和其之后的残差连接之间插入适配器层(adapter layers)(MLP模块同理)，适配器层包含了两个带有偏置的全连接层，中间带有非线性激活，我们称该原始设计(original design)$\text {Adapter}^{\text H}$
Lin提出了更高效的设计，其中仅在MLP模块后以及LayerNorm层之后插入适配器层，我们称其为$\text{Adapter}^{\text L}$，这和Pfeiffer提出的另一设计也很相似，我们称其为$\text {Adapter}^{\text P}$
我们还考虑另一个名为AdapterDrop的基准方法，该方法为了更高的效率会丢弃一些适配器层，我们称其为$\text {Adapter}^{\text D}$
对于所有的情况，我们有$|\Theta | = \hat {L}_{Adpt} \times (2\times d_{model}\times r + r + d_{model}) + 2\times \hat {L}_{LN}\times d_{model}$其中$\hat L_{Adpt}$是适配器层的数量，$\hat L_{LN}$是可训练LayerNorms的数量(例如在$\text {Adapter}^{\text L}$中)

**LoRA**
LoRA为现存的权重矩阵并行地添加可训练的成对的秩分解矩阵(trainable pairs of rnak decomposition matrices)，为了简单，在大多数实验中我们仅对$W_q$和$W_v$应用LoRA，
可训练参数的数量由秩$r$和原矩阵的形状决定，即$|\Theta | = 2\times \hat L_{LoRA}\times d_{model}\times r$，其中$\hat L_{LoRA}$是我们应用LoRA的权重矩阵的数量
## 5.2 RoBERTa Base/Large
RoBERTa优化了BERT的预训练方案，在不引入更多训练参数的情况下增强了后者的任务表现，我们从HuggingFace Transformer库从取用预训练好的RoBERTa base(125M)和RoBERTa large(335M)，然后在GLUE基准测试中的任务中评估不同适应方法的表现
为了保证公平的比较，我们在将LoRA与适配器方法进行比较时做了两个改变：首先，我们对所有的任务都使用相同的batch size，同时序列长度使用128以匹配适配器基准方法，其次，我们将模型初始化为MRPC、RTE、STS-B预训练好的模型，而非已经适应至MNLI的模型
结果见表格2
## 5.3 DeBERTa XXL
DeBERTa是最近的BERT的变体，训练于更大的规模，且在GLUE、SuperGLUE等基准测试上表现较有竞争力
我们评估是否LoRA可以在GLUE上匹配完全微调的DeBERTa XXL(1.5B)的表现
结果见表格2
## 5.4 GPT-2 Medium/Large
之前的结果已经展示了LoRA在NLU任务上是完全微调的可竞争的替代方法，我们希望测试LoRA在NLG模型上的表现，例如GPT-2 medium和large，我们展示了在E2E NLG Challenge、WebNLG、DART上的结果
结果见表格三
## 5.5 Scaling Up to GPT-3 175B
在GPT-3 175B上的结果见表格四，LoRA在三个数据集上匹配或超过了其他的微调方法，注意不是所有的方法的效果都随着可训练参数的数量单调增长，我们观察到在prefix-embedding调节中，超过256个special tokens会导致效果显著下降，以及在prefix-layer调节中，超过32个special tokens会导致效果显著下降，我们怀疑有过多的special tokens会导致输入分布相对于预训练数据分布偏移过远
# 6 Related Works
**Transformer Language Models**
BERT和GPT-2之后，出现了新的学习范式，即用通用领域的数据大规模预训练后，在针对任务的数据上微调，效果会显著优于直接在针对任务的数据上训练

**Prompt Engineering and Fine-Tuning**

**Parameter-Efficient Adaptation**
在网络中插入适配器层是较为参数高效的调节方式，LoRA相较适配器方法的优势在于不引入推理延迟
目前对适配器的拓展工作是COMPACTER，它使用Kronecker积和一些预定义好的参数共享方案以参数化适配器层，类似地，将LoRA与其他基于张量积的方法也可能提高LoRA的参数效率
优化输入词嵌入也是一种参数高效的调节方式，可以将其视为提示词工程的连续和可微的泛化，但这类工作的放缩只能由在提示词中使用更多的special tokens实现，这会占据task tokens的有效序列长度

**Low-Rank Structures in Deep Learning**
低秩结构是在ML中常见的结构，许多ML问题有特定的内在低秩，同样，对于许多DL任务，尤其是严重过参数化的网络中，也会存在低秩的特性
一些工作甚至在训练网络时显式地施加了低秩的约束，但没有工作考虑到了模型在对下流任务适应时的低秩更新
# 7 Understanding the Low-Rank Updates
我们尝试回答以下问题：
1. 给定参数预算约束，我们应该调节预训练Transformer模型中的权重矩阵的哪个子集以最大化下流任务性能
2. 最优的调节矩阵$\Delta W$真的是秩缺的吗，如果是的话，实践中应该使用的秩是多少
3. $\Delta W$和$W$的关联是什么，$\Delta W$是否高度与$W$相关(correlate)，$\Delta W$相对于$W$有多大(how large)
## 7.1 Which Weight Matrices in Transformer Should We Apply To
在受限的参数预算下，我们应该用LoRA适应哪些类型的权重以在下流任务中达到最优表现？
在4.2部分我们提到我们仅考虑了自注意力模块内的权重矩阵，我们把在GPT-3 175B的参数预算设定为18M(如果存为FP16的话大约是35M)，这对应于我们在所有共96层上仅调节一种类型的注意力权重，且设定$r=8$，或者对应于我们调节两种类型的注意力权重，且设定$r=4$
控制参数预算相同，对GPT-3中不同类型的注意力权重进行LoRA的结果如Table-5所示
![[LoRA-Table 5.png]]

注意到将所有的参数预算单独花在$\Delta W_q$或$\Delta W_k$会导致显著较低的表现，而对$W_q$和$W_v$同时进行适应则产生了最优的表现，这说明了秩$r=4$已经可以捕获$\Delta W$中的足够的信息，因此适应更多的矩阵会比用更大的秩适应一种类型的矩阵更可取
## 7.2 What is the Optimal Rank $r$ for LoRA
我们考虑秩$r$对模型性能的影响，我们分别调节了$\{W_q\},\{W_q,W_v\},\{W_q,W_k,W_v,W_c\}$，并进行了对比，结果见Table-6
![[LoRA-Table6.png]]

可以看到，在秩非常小(等于1)时，LoRA的表现也非常具有竞争力，这表明了权重更新矩阵$\Delta W$可以有非常小的“内在秩”，
为了进一步支持该发现，我们检查了在不同的$r$下和不同的随机种子下学习到的$\Delta W$的子空间的重叠性，我们认为提高$r$不会覆盖更多有意义的子空间，因此低秩适应矩阵是足够的
注意我们不认为小的秩$r$对所有的数据集或下游工作都有效，例如：下游任务的语言和模型预训练所用的语言是不同的，重新训练整个模型(类似于LoRA的秩$r=d_{model}$的效果会好于用小秩的LoRA

**Subspace similarity between different $r$**
给定$A_{r=8},A_{r=64}$是在相同预训练模型下在$r=8,r=64$时学习到的适应矩阵，我们进行奇异值分解获得右奇异值单元矩阵(right-singular unitary matrices)$U_{A_r=8},U_{A_r=64}$
(注意相似性分析也可以用矩阵$B$和左奇异值单元矩阵来分析)

我们希望回答：由$U_{A_r=8}$的前$i$个($1\le i\le8$)奇异值向量张成的子空间有多少被$U_{A_r=64}$的前$j$个($1\le j\le 64$)奇异值向量张成的子空间包含
我们用一个基于Grassmann距离的规范化的子空间相似度度量衡量这个量：
$$\phi(A_{r=8},A_{r=64},i,j) = \frac {\|{U_{A_r=8}^i}^TU_{A_r=64}^j\|_F^2}{\min(i,j)}\in[0,1]\tag{4}$$
其中$U_{A_r=8}^i$表示$U_{A_r=8}$中前$i$个奇异值向量组成的矩阵
$\phi(\cdot)$的范围是$[0,1]$，$1$表示子空间有完全的重叠，$0$表示完全的分离，
随着我们变化$i,j$，$\phi$的变化如Figure3所示，此处只展示了96层中的第48层上的结果，但对于其他的层，结果是一致的
![[LoRA-Fig3.png]]
在Figure3中，我们观察到一个重要的点：
$A_{r=8}$和$A_{r=64}$各自的第一个奇异值向量的方向有很大的重叠，而其他的奇异值向量的方向则没有，具体地说，$A_{r=8}$的$\Delta W_v$(或$\Delta W_q$)和$A_{r=64}$的$\Delta W_v$(或$\Delta W_q$)以大于0.5的规范化相似度(normalized similarity)共享一个一维的子空间，这也为$r=1$时在GPT-3的表现也十分好提供了一个解释

因为$A_{r=8}$和$A_{r=64}$都是用同一个预训练模型学习到的，Figure3表明了$A_{r=8}$的$A_{r=64}$的前几个奇异值向量方向是最有用的，而其他的方向则可能包含大部分的在训练时累积的随机噪声，因此，适应矩阵实际上的秩可以很低

**Subspace similarity between different random seeds**
我们在$r=64$的情况下画出两个随机种子之间的规范化的子空间相似度图，见Figure4，可以看到$\Delta W_q$似乎比$\Delta W_v$有更高的”内在秩“，因为两个随机种子得到的$\Delta W_q$之间有更多共同的奇异值方向，这和我们在Table6中的观察一致
作为比较，我们也画出了两个随机的Gaussian矩阵之间的规范化子空间相似度图，这两个Gussian矩阵并不共享任何的奇异值方向
![[LoRA-Fig4.png]]
## 7.3 How does the Adaptation Matrix $\Delta W$ Compare to $W$
我们进一步研究$\Delta W$和$W$之间的关系，特别地，是否$\Delta W$和$W$高度相关(或数学上，$\Delta W$是否大部分被$W$的前几个奇异值方向包含)，另外，$\Delta W$相较于$W$中其对应的方向到底有多”大“("large")

我们通过计算$U^TWV$，将$W$映射到$\Delta W$的$r$维的子空间，其中$U/V$分别是$\Delta W$的左/右奇异值向量矩阵，然后，我们计算Frobenius范数$\|U^TWV\|_F$和$\|W\|_F$，作为比较，我们还计算了$\|U^TWV\|_F$，其中$U,V$被替换为了$W$的前$r$个奇异值向量或随机的矩阵，比较结果见Table 7
![[LoRA-Table7.png]]

(
$\Delta W$的SVD写为：$\Delta W = U\Sigma V^T = \sigma_1 u_1v_1^T + \cdots + \sigma_ru_rv_r^T=\sum_{i=1}^r u_rv_r^T$，
其中$U = [u_1,u_2,\cdots, u_r],V = [v_1,v_2,\cdots, v_r]$，分别是$\Delta W$的列/行空间的一组正交基

$W$的SVD写为：$W = U'\Sigma'V'^T = \sigma_1'u_1'v_1'^T + \cdots + \sigma_r'u_{r'}'v_{r'}'^T=\sum_{i=1}^{r'}u_i'v_i'^T$，
其中$U' = [u'_1,u'_2,\cdots, u'_{r'}],V = [v'_1,v'_2,\cdots, v'_{r'}]$，分别是$W$的列/行空间的一组正交基
这里$r'$是指$W$的秩，$r'>r$

SVD告诉我们，一个秩为$r$的矩阵可以视为$r$个单秩矩阵的和，可以称它们为矩阵的单秩成分，一个单秩成分$u_iv_i^T$实际就是一个行正交基和列正交基的乘积，它包含了这两个正交基内的方向信息(完整的单秩成分要乘上奇异值$\sigma_i$，$\sigma_i$包含了大小信息)
注意两个相乘的行正交基和列正交基之间可以相互转化，它们对应同一个奇异值，分析详见[[SVD and PCA#2 SVD(Singular Value Decomposition)]]

考虑$U^T W V$：
$$\begin{align}
U^TWV&=\sum_{i=1}^{r'}\sigma'_i U^Tu_i'v_i'^TV\\
&=\sum_{i=1}^{r'}\sigma'_i 
\begin{bmatrix}
u_1^T\\
\vdots\\
u_{r}^T
\end{bmatrix}u_i'v_i'^T[v_1,\cdots,v_{r}]\\
&=\sum_{i=1}^{r'}\sigma_i'\begin{bmatrix}
\langle u_1,u_i'\rangle\\
\vdots\\
\langle u_{r},u_i'\rangle
\end{bmatrix}[\langle v_1,v_i'\rangle,\cdots,\langle v_{r},v_i'\rangle]\\
&=\sum_{i=1}^{r'}\sigma'_i
\begin{bmatrix}
\langle u_1,u_i'\rangle\cdot\langle v_1,v_i'\rangle &\cdots&\langle u_1,u_i'\rangle\cdot\langle v_r,v_i'\rangle \\
\vdots&\ddots&\vdots\\
\langle u_r,u_i'\rangle\cdot\langle v_1,v_i'\rangle &\cdots&\langle u_r,u_i'\rangle\cdot\langle v_r,v_i'\rangle \\
\end{bmatrix}
\end{align}$$
显然$\langle u_1,u_i' \rangle$就是$u_i'$在$u_1$上的投影，$\langle v_1,v_i'\rangle$就是$v_i'$在$v_1$上的投影，投影值在$[0,1]$之间

因此$U^TWV$中的第$i,j$元就是$\sum_{i=1}^{r'}\sigma_i' \langle u_i,u_i'\rangle\cdot\langle v_j,v_i'\rangle$，
这是一个和$W$的所有奇异值向量/行列正交基在$\Delta W$的第$i$个行正交基和第$j$个列正交基上的投影，以及对应的奇异值相关的和式
因为$\sigma_i' \langle u_i,u_i'\rangle\cdot\langle v_j,v_i'\rangle$的值域是$[0,1]$，故可以视$\sigma_i' \langle u_i,u_i'\rangle\cdot\langle v_j,v_i'\rangle$为对$\sigma_i'$的放缩

$\|U^TWV\|_F$实际上考察了$W$的列正交基和$U$/行正交基和$V$的重合程度，
如果有完全重合的$r$对基，即行正交基和列正交基都重合，也就是$W$的某个单秩成分和$UV^T$的某个单秩成分完全重合，则$\|U^TWV\|_F$可以达到上界$\sum \sigma_i'$，
该和式由$r$个$\sigma_i'$构成，对应$W$与$UV^T$对应的$r$个单秩成分各自的奇异值

实际上，在这种情况下，$W$就完全包含了$UV^T$，$UV^T$是$W$的$r'$个($r'>r$)单秩成分其中$r$个构成的子集，也就是$W$完全包含了$\Delta W = U\Sigma V^T$中的方向信息，或者说$W$张成的空间包含了$\Delta W$张成的空间

如果不是考虑$F$范数，而是考虑1范数$\|U^TWV\|_1$，可以知道的是，
如果有$r$个$u_{i}'$落在了空间$U$内，即$u_{i}'$可以表示为$\sum_{j=1}^{r} c_j u_j\quad(s.t.\sum_{j=1}^r c_j = 1)$，
同时，对应的$r$个$v_i'$落在了空间$V$内，即$v_{i}'$可以表示为$\sum_{j=1}^{r} e_j v_j\quad (s.t.\sum_{j=1}^r e_j = 1)$
则$\|U^TWV\|_1$可以达到上界$\sum \sigma_i'$
这种情况下也可以认为$W$完全包含了$UV^T$，注意$r$个$u_i'/v_i'$互相也是正交的，因此各自也可以构成一个$r$维子空间，如果$u_i',v_i'$的$r$维子空间可以由$U/V$表示，说明它们其实就是旋转关系
)

我们从Table 7中可以得出几个结论：首先，$\Delta W$相较于一个随机的矩阵，显然和$W$有更强的相关性，这表明了$\Delta W$放大了一些已经在$W$内的特征；其次，$\Delta W$并不是重复$W$的前几个奇异方向，而是仅放大一些在$W$内不被强调的方向；最后，放大的因子是很大的，$21.5\approx 6.91 / 0.32\text{ for } r=4$
这表明了低秩适应矩阵潜在地放大了为特定下游任务重要的特征，这些特征在通用预训练中被学习到，但不被强调
# 8 Conculsion and Future Work
我们提出LoRA，一个高效的适应策略，它既不引入推理延迟，也不减少输入序列长度，同时保持了高的模型质量，并且，它允许在部署时快速的任务切换，因为大量的模型参数是共享的，LoRA原则上可以用于任意带有线性层的神经网络

未来的工作有几大方向：1) LoRA可以和其他高效的调节方法结合，有可能可以提供正交的提升 2) 微调或LoRA背后的机制尚不明了——在预训练时学习到的特征是如何转换的以在下游任务表现良好？相较于完全微调，LoRA给出了一定的解释 3) 我们大部分情况下采用启发式以选择要应用LoRA的权重矩阵，是否存在更有原则的方式进行选择？4) $\Delta W$的秩缺性质说明了$W$也可能有秩缺的性质，在此基础上可以启发后续工作
# A Large Language Models Still Need Parameter Updates
Table 8显示了在大和小数据集上，微调相较于小样本学习/提示词工程都可以显著提高模型性能
![[LoRA-Table8.png]]
GPT-3在RTE上的小样本学习的结果来自于原论文，MNLI-matched上的结果，我们采用每个类别两个示例(two demostrations per class)，总共六个上下文样本(in-context examples)
# B Inference Latency Introduced by Adapter Layers
适应器层是以顺序串行(sequential)的形式加入到预训练模型内的外部模块，而LoRA则可以视为是以并行的形式加入的外部模块，因此，适应器层会引入额外的推理延迟
有人指出由适应器层可以随着模型的batch size或序列长度的增大，以此完全利用硬件并行，而减小它引入的推理延迟，我们在GPT-2 medium上类似的实验确定了它们的观察，并且我们指出，在在线推理的情况下，batch size较小，适配器层引入的推理延迟会很大

我们在NVIDIA Quadro RTX8000上进行100次实验，取平均值，衡量单次前向传播的延迟随输入batch size、序列长度、适应器瓶颈维度$r$的变化
我们测试了两个适应器设计：$\text {Adapter}^{\text H}$和$\text {Adapter}^{\text L}$来自于Houlsbey et al.和Lini et al.，在Figure 5中，我们画出加入适应器层后相对于无适应器层baseline的推理速度下降比率
![[LoRA-Figure5.png]]
# C Dataset Details
**GLUE Benchmark** 是一个广泛的一系列自然语言理解任务的集合，它包括了MNLI(推理 inference)、SST2(情感分析 sentiment analysis)、MRPC(转述检测 paraphrase detection)、CoLA(语言上可接受性 linguistic acceptability)、QNLI(推理 inference)、QQF(问题回答 qusetion-answering)、RTE(推理 Inference)和STS-B(文本相似度 textual similarity)
GLUE benchmark有广泛的覆盖范围，这使得它成为评估NLP模型的标准度量(metric)，其中每个独立的数据集是在不同的许可证(permissive license)下发布的

**WikiSQL** 包含了56355/8421个训练/验证样本，任务是从自然语言问题和表格示意图(table schemata)中生成SQL查询，我们将上下文表示为$x = \{\text {table schema}, \text {query}\}$，目标表示为$y = \{\text {SQL}\}$

**SAMSum** 包含了14732/819个训练/测试样本，数据由两个人之间的对话和语言学家所写的对应的总结文本构成，我们将上下文表示为用"\\n"连接的文本，最后接上"\\n\\n"，以及目标$y = \{\text {summary}\}$

**E2E NLG Challenge** 该数据集用于训练端到端、数据驱动的自然语言生成系统，且常用于数据到文本(data-to-text)评估，数据集大约由42000个训练样例、4600个验证样例、4600个测试样例构成
每个用于输入的源表格可以有多个参考文本，每个输入样本$(x,y)$都由一个slot-value对序列和对应的自然语言参考文本构成

**DART** 是一个开放域的数据到文本数据集，数据集的输入是结构化的ENTITY——RELATION——ENTITY三元组序列，数据集总共有82K个样本，相较于E2E，DART要显著更大且包含更多复杂的数据到文本任务

**WebNLG** 另一个常用于数据到文本评估的数据集，共14个类，22K个样本，其中9个类在训练时是可见的，5个类是训练时不可见的，因此评估可以划分为在可见类上的评估(S)、在不可见类上的评估(U)、以及一起的评估(A)，数据集的每个输入样本由SUBJECT-PROPERTY-OBJECT三元组序列表示
# D Hyperparameter Used in Experiments
## D.1 RoBERTa
在RoBERTa上训练LoRA时，我们使用AdamW以及线性学习率调度；
我们为LoRA调节了学习率、epoch数量和batch size；
在对MRPC、RTE和STS-B适应时，LoRA模块的初始化使用在MNLI上最优的检查点；
每次训练的结果取自于最优的epoch；
为了公平的比较，我们将模型序列长度限制为128，且对所有任务都固定batch size；
我们在对MRPC、RTE和STS-B适应时，预训练模型使用的是原始的RoBERTa large，而不是已经适应到了MNLI的模型；
使用的超参数详见Table 9
## D.2 DeBERTa
在DeBERTa上训练LoRA时，我们使用AdamW以及线性学习率调度；
我们为LoRA调节了学习率、dropout概率、warm-up步数和batch size；
在对MRPC、RTE和STS-B适应时，LoRA模块的初始化使用在MNLI上最优的检查点；
每次训练的结果取自于最优的epoch；
使用的超参数详见Table 10
## D.3 GPT-2
在GPT-2上训练LoRA时，我们使用AdamW以及线性学习率调度，epoch固定为5；
batch size、学习率和束搜索的beam size的使用同Li&Liang，对LoRA，我们也对这些参数进行了调节；
每次训练的结果取自于最优的epoch；
使用的超参数详见Table 11
## D.4 GPT-3
在GPT-3上训练LoRA时，我们使用AdamW、权重衰退因子0.1、batch size 128，epoch固定为2；
WiKiSQL的序列长度是384，MNLI是768，SAMSum是2048；
我们为LoRA调节了学习率；
使用的超参数详见Table 12
# F Additional Empirical Experiments
## F.1 Additional Experiments on GPT-2
## F.2 Additional Experiments on GPT-3
## F.3 Low-Data Regime
# G Measuring Similarity between Subspeces
本文中，我们使用
$$\phi(A,B,i,j)  = \psi(U_A^i, U_B^j) = \frac {\|{U_A^i}^TU_B^j\|_F^2}{\min\{i,j\}}$$
来衡量两个列正交矩阵$U_A^i \in \mathbb R^{d\times i}, U_B^j \in \mathbb R^{d\times j}$之间的子空间相似性，其中两个列正交矩阵来自于$A,B$的左奇异矩阵的列
我们指出该相似度度量实际上是衡量子空间之间的距离的标准投影度量的逆(a reverse of the standard Projection Metric that measures distance between subspaces)

我们令${U_A^i}^TU_B^j$的奇异值为$\sigma_1,\sigma_2,\cdots, \sigma_p$，其中$p = \min\{i,j\}$，在Ham&Lee中，投影度量(the Projection Metric)定义为：
$$d(U_A^i,U_B^j) = \sqrt {p - \sum_{i=1}^p\sigma_i^2}\in[0,\sqrt p]$$
而我们的相似度度量定义为：
$$\phi(A,B,i,j) = \psi(U_A^i,U_B^j) = \frac {\sum_{i=1}^p\sigma_i^2}{p}=\frac 1 p\left(1-d(U_A^i,U_B^j)^2\right)$$
该相似度度量满足如果$U_A^i$和$U_B^j$共享同个列空间，则$\phi(A,B,i,j) = 1$，若二者的列空间完全正交，则$\phi(A,B,i,j) = 0$，其他情况下，$\phi(A,B,i,j) \in (0,1)$

(
对于一个列正交矩阵$U=[u_1,u_2,\cdots,u_n]\in \mathbb R^{m\times n}$，
考虑$m\ge n$的情况，有$r(U) = n$
考虑$U$的SVD，将$U$写为$n$个单秩矩阵的和：$U = u_1[1,0,\cdots,0] + u_2[0,1,\cdots,0] + \cdots + u_n[0,0,\cdots,1]$
因此$U$的$n$个奇异值都是$1$，其左奇异矩阵就是$U$本身，右奇异矩阵就是$I_{n\times n}$

考虑两个不同的列正交矩阵$U_A^i\in \mathbb R^{d\times i}, U_B^j\in \mathbb R^{d\times j}$
$$\begin{align}
{U_A^i}^TU_B^j &= \begin{bmatrix}u_{a_1}^T\\
\vdots\\
u_{a_i}^T
\end{bmatrix}[u_{b_1},\cdots,u_{b_j}]\\
&=\begin{bmatrix}\langle u_{a_1},u_{b_1}\rangle&\cdots&\langle u_{a_1},u_{b_j}\rangle\\
\vdots & \ddots & \vdots\\
\langle u_{a_i},u_{b_1}\rangle&\cdots & \langle u_{a_i},u_{b_j}\rangle
\end{bmatrix}
\end{align}$$
如果$U_A^i,U_B^j$完全正交，则${U_A^i}^TU_B^j$中的每一项都取到最小值$0$，则$\|{U_A^i}^TU_B^j\|_F^2$取到最小值$0$
如果${U_A^i}^TU_B^j$完全相交，例如$j>i$时，对于$1\le l \le i$的任意$u_{a_l}$都可以由$u_{b_1},u_{b_2},\cdots,u_{b_j}$的线性组合表示，即满足：
$$\begin{align}
&u_{a_l} = c_1u_{b_1} + c_2u_{b_2} + \cdots + c_{j}u_{b_j}\quad \forall l\in\{1,2,\cdots,i\}\\
&s.t.\ c_1^2 + c_2^2 + \cdots+ c_j^2 = 1\text{ or }\sum_{k=1}^j c_k^2 = 1
\end{align}$$
此时，我们将$\|{U_A^i}^TU_B^j\|_F^2$展开，重新排布一下：
$$\begin{align}
\|{U_A^i}^TU_B^j\|_F^2&=\begin{matrix}
\langle u_{a_1},u_{b_1}\rangle^2&+&\cdots&+&\langle u_{a_1},u_{b_j}\rangle^2\\
+& &&  & +\\
\vdots & &\ddots & &\vdots\\
+&&&&+\\
\langle u_{a_i},u_{b_1}\rangle^2&+&\cdots & +&\langle u_{a_i},u_{b_j}\rangle^2
\end{matrix}\\
&=\sum_{l=1}^i\sum_{k=1}^j\langle u_{a_l},u_{b_k}\rangle^2\\
&=\sum_{l=1}^i\sum_{k=1}^j c_k^2\\
&=i
\end{align}$$
因为此时$\sum_{k=1}^j \langle u_{a_l},u_{b_k}\rangle^2$可以取到最大值$1$，故$\|{U_A^i}^TU_B^j\|_F^2$取到最大值$i$
)
# H Additional Experiments on Low-Rank Matrices
## H.1 Correlation between LoRA Modules
## H.2 Effect of $r$ on GPT-2
我们也在GPT-2上进行了7.2节关于$r$的效果的实验，我们使用E2E NLG Challenge数据集，报告了在不同的$r$下，训练了26000步之后的验证损失和测试度量，结果在Table18
GPT-2 Medium的最优的秩在$4$到$16$之间，取决于使用的度量，这和GPT-3 175B时相似的
模型大小和最优的适应秩之间的关系仍然是开放的问题
## H.3 Correlation between $W$ and $\Delta W$
Figure8展示了$W$和$\Delta W$在不同的$r$下的规范化的子空间相似度
![[LoRA-Fig8.png]]
注意$\Delta W$并不包含$W$的前几个奇异方向(top singular direction)，因为$\Delta W$的前4个左奇异方向和$W$的前10%左奇异方向之间的相似度仅仅超过0.2，这证实了$\Delta W$包含了那些“任务特定 tack-specific”的方向，这些方向在$W$中并没有被强调

这引出一个问题：我们要将这些任务特定的方向放大多大以使得模型可以在下游任务表现良好
## H.4 Amplification Factor
考虑一个称为特征放大因子(feature amplification factor)的量：
$$\frac {\|\Delta W\|_F}{\|U^TWV\|_F}$$
其中$U,V$是$\Delta W$的奇异矩阵($U^TWV$给出$W$到$\Delta W$张成的子空间的“投影”)

直觉上，当$\Delta W$大部分包含任务特定的方向，这个量就衡量了这些任务特定的方向在$\Delta W$中被放大了多少倍，如7.3节所示，对于$r=4$，这个放大因子可以大到$\approx 20$
换句话说，(一般地说)在预训练模型$W$的整个特征空间中，每一层都存在四个特征方向需要用一个非常大的放大因子$20$放大，以达到我们所报告的对于特定下游任务的准确率，并且，对于每个不同的下游任务，我们应该看到其对应要放大的特征方向集合是十分不同的

注意到当$r=64$时，放大因子仅在$2$左右，这意味着$r=64$时，$\Delta W$中学习到的大多数方向并没有被放大很多，这实际上证实了需要用于表示“任务特定方向 task-specific directions"的内在秩是低的，作为比较，在$r=4$的$\Delta W$中的方向都以大到了$20$的因子被放大
