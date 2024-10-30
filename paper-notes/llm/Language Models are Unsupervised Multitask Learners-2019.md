# Abstract
自然语言处理任务，例如问答、机器翻译、阅读理解、总结等，一般都使用在针对任务的数据集上的有监督学习方法进行处理，我们将展示当语言模型在一个包含了百万个网页(webpages)的数据集WebText上预训练时，可以在没有任何显式监督(explicit supervision)的情况下学习处理这些任务
当预训练的语言模型条件于文档和问题时，它生成的答案在CoQA数据集达到了55的F1分数，在不需要使用127000多个训练样本的情况下，其表现匹配或超过了四个基线方法中的三个

语言模型的容量(capacity)对于零样本任务迁移(zero-shot task transfer)的成功至关重要，且语言模型在多个任务上的表现提高随它的容量增长呈对数线性关系(log-linear)
我们最大的模型GPT-2是一个含有1.5B参数的Transformer模型，在零样本设定下，它在8个测试的语言建模数据集中的7个达到了SOTA，且该模型仍然没有完全拟合(underfits)WebText

模型所生成的样本也反应了模型处理自然语言任务能力的进步(improvements)，且模型生成的样本中已经包含了连贯的文本段落(coherent paragraphs of text)
这些发现暗示了一条构建语言处理系统的一条具有前景的道路，即构建可以从语言的自然发生的演示中学习执行任务的语言处理系统(learn to perform tasks from their naturally occurring demonstrations)
# 1 Introduction
通过结合使用大数据集、高容量模型、有监督学习方法，机器学习系统已经可以在它们被训练的任务上有出色的表现，但这些系统同时也是脆弱(brittle)的，对数据分布和任务规范(task specification)中的微小变化敏感的(senstive)
我们希望朝着更通用(general)的系统迈进，这些系统能够执行许多任务——最终无需为每一个任务手动创建和标记训练数据集

创建机器学习系统的主导方法是收集一个展示了所需任务的正确行为(correct behaviour for a desired task)的训练示例(training examples)的数据集，然后训练一个系统来模仿(imitate)这些行为，然后在独立同分布(IID)的留出示例(held-out examples)上测试其性能，
但是，字幕模型(captioning models)、阅读理解系统和图像分类器在面对具有多样性和变化性(diversity and variety)的输入时的经常不稳定的行为(erratic behaviour)凸显了这种方法的一些缺点

我们怀疑，在单一领域数据集上进行单一任务训练(single task training on single domain datasets)是系统缺乏泛化能力(lack of generalization)的一个重要因素，
在当前架构下，要朝着更健壮的系统发展，需要我们在广泛的领域和任务上进行训练和性能衡量(training and measuring performance on a wide range of domains and tasks)，
最近提出的几个基准，如GLUE和decaNLP，已经开始关注于对模型在广泛任务上的性能进行评估

多任务学习(multitask learning)是提高通用性能(general performance)的一个有希望的框架，然而，NLP领域的多任务训练仍处于起步阶段，最近的工作仅报告了适度的(modest)性能提升，迄今为止最好的两项工作分别在总共10和17个的 `(dataset, objective)` pair上进行训练，
从元学习的角度来看，每个 `(dataset, objective)` pair都是从数据集和目标的分布中(distribution of datasets and objectives)抽取的单一训练示例，而当前的机器学习系统需要数百到数千个这样的示例来归纳出能泛化良好的函数(induce functions which generalize well)，这表明多任务训练也需要许多的有效训练对(effective training pairs)才能达到良好的通用性能，
但继续将数据集的创建和目标设计(scale the creation of datasets and the design of objectives)扩大到能让依赖于当前技术的模型学习到通用性能所需要的规模是非常困难的，这就促使我们探索执行多任务学习的额外设定(additional setups)

当前在语言任务上表现最佳的系统利用了预训练和有监督微调的结合，这种迁移方法有很长的历史，且趋向于更灵活的迁移形式(more flexible forms of transfer)发展，
在早期阶段，词向量被学习并用作特定任务架构的输入(inputs to task-specific architectures)，之后，迁移的对象变为递归网络的上下文表示(contextual representations of recurrent networks)，最近的工作表明，在迁移时不再需要构建针对任务的架构，对许多自注意力块进行迁移就足够了

但现存的这些方法仍然需要有监督的训练才能执行任务，当只有最小或没有监督数据可用(minimal or no supervised data is available)时，另一系列工作已经展示了语言模型执行特定任务的潜力，如常识推理和情感分析，
在本文中，我们将这两条工作线联系起来，尝试构建更通用的迁移方法，我们将展示语言模型可以在零样本设置下就能执行下游任务(perform down-stream tasks in a zero-shot setting)——无需任何参数或架构修改(without any parameter or architecture modifications)，我们将通过强调语言模型在零样本设置中执行广泛任务的能力来展示这种方法的潜力，我们的模型在不同的任务上都取得了优秀的结果
# 2 Approach
我们方法的核心是语言建模(language modeling)，语言建模常常被构建为一个在一组示例$(x_1,x_2,\dots,x_n)$中的无监督分布估计问题(unsupervised distributino estimation from a set of examples)，每个示例都是一个符号序列(sequence of symbols)$(s_1,s_2,\dots,s_n)$，长度可以各不相同

因为语言是自然上顺序的(natural sequential ordering)，我们通常可以将符号上的联合概率分解为条件概率的乘积：
$$p(x) = \prod_{i=1}^n p (s_n|s_1,\dots,s_{n-1})\tag{1}$$
这种方法允许我们从$p(x)$或任何形式为$p(s_{n-k},\dots,s_n|s_1,\dots,s_{n-k-1})$的条件分布中进行可行的采样和估计(tractable sampling and estimation)，近几年来，能够计算这些条件概率的模型的表达能力(expressiveness)有了显著提升，例如Transformer的自注意力架构

学习执行单个任务可以在一个概率框架下被表示为估计条件分布(estimating a conditional distribution)$p(output|intput)$，而因为通用的系统应该要可以执行多种不同的任务，故即便对于相同的输入，它也应该不仅仅条件于输入，也应该条件于它所要执行的任务(condition not only on the input but also on the task to be performed)，也就是说，它应该建模$p(output|intput,task)$

学习执行单一任务可以用概率框架表示为估计一个条件分布p(output|input)。由于一个通用系统应该能够执行许多不同的任务，即使对于相同的输入，它不仅应该基于输入进行条件判断，还应该基于要执行的任务进行条件判断。也就是说，它应该建模p(output|input; task)。这在多任务学习和元学习环境中有各种不同的形式化方法。任务条件通常在架构层面实现，例如Kaiser等人(2017)中的任务特定编码器和解码器，或者在算法层面实现，如MAML的内外循环优化框架(Finn等人，2017)。但正如McCann等人(2018)所示例，语言提供了一种灵活的方式来指定任务、输入和输出，全部作为符号序列。例如，一个翻译训练示例可以写成序列(翻译成法语，英文文本，法文文本)。同样，一个阅读理解训练示例可以写成(回答问题，文档，问题，答案)。McCann等人(2018)证明了可以训练一个单一模型，MQAN，来推断并执行许多不同的任务，这些任务的例子具有这种类型的格式。
