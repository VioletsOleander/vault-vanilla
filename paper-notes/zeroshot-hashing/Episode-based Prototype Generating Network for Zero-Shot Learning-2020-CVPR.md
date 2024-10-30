# Abstract
# 1 Introduction
深度学习技术通常依赖于人工标注的平衡的训练数据的可用性(availability)，零样本学习旨在解决数据稀缺的问题
多数现存方法致力于设计视觉-语义交互模型(visual-semantic interaction modele)，问题在于训练于可见类的模型对不可见类泛化能力不高，且训练于可见类的模型容易将不可见类实例分类到可见类，导致较大的不平衡分类偏移问题(imbalanced classification shift issue)
现存的生成式方法通过为不可见类合成一些视觉特征，将零样本分类问题转化为传统的分类问题，一定程度上缓解了上述问题，但在广义的零样本学习设置下，由于训练的不稳定性和模式崩溃(mode collapse)问题而表现不佳

受少样本学习中的元学习(meta-learning)的启发，我们提出基于片段(episode-based)的训练范式以解决上述问题
具体地，训练阶段由一系列片段构成，每个片段随机将训练数据集划分为两个类别不相交的子集：支持集(support set)和精炼集(refining set)，因此每个片段模仿了一个假的零样本分类任务
支持集用于训练基模型(base model)，建模视觉模态和类语义(class semantics)模态的语义交互，精炼集通过最小化真实标签和基模型预测的标签的差异以精炼模型
每个片段训练的模型参数用于下一个片段的初始化
模型在逐个片段中积累预测假的不可见类的集成经验(ensamble experience)，提高在真实不可见类上的泛化能力
![[EPGN-Fig1.png]]

我们设计了原型生成网络(prototype generating network/PGN)作为基模型，以类语义原型(semantic prototype)为条件(conditioned on)，生成类级别的视觉原型(visual prototype)
模型有两个生成器，将视觉特征和类语义原型映射到它们的对应物，以及一个区分器，用于区分真实视觉特征和真实类语义原型的拼接(concatenation)和其假的对应物的拼接
我们设计了多模态交叉熵损失，将视觉特征、类别语义原型、类别标签聚合到一个分类网络中，相较于现存方法，省去了额外的辅助分类网络的参数
# 2 Related Work
## 2.1 Generative ZSL
我们的模型也是生成式方法，我们在没有额外噪声输入的情况下，以类语义原型为条件，合成类级别的视觉原型，而不是合成实例级别的视觉特征
我们将视觉原型生成(visual prototype generation)和类语义推断(class semantic inference)通过两个可分离的(seperable)双向映射网络融合入一个统一的框架
## 2.2 Episode-based approach
