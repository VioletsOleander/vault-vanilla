[SimCLR](<file:///D:\Learning\paper\2020-ICML-SimCLR-A Simple Framework for Constrastive Learning of Visual Representations.pdf>)
# Abstract
本文展示SimCLR：一个简单的用于对比学习视觉表示的框架，我们简化了最近提出的子监督对比学习算法，且不需要特定的架构(architecture)或存储体(memory bank)
为了理解是什么使得对比预测任务(constrastive prediction tasks)可以学习到有用的表示，我们系统地研究了我们框架中的主要成分，发现：
1. 数据增强在定义有效的预测任务中起到了关键作用
2. 为表示(repersentation)和对比损失(constrastive loss)之间引入可学习的非线性变换可以极大地提升学习到的表示的质量
3. 相较于有监督学习，对比学习可以从更大的batch size和更多的训练步数中获益

结合这些发现，我们的方法在ImageNet上极大地优于之前的自监督和半监督学习方法，训练于通过SimCLR学习到的子监督表示上的线性分类器可以达到76.5%的top-1准确率，比SOTA提升了7%，匹配了有监督的ResNet-50的准确率
在对仅1%的标签进行微调后，我们达到了85.5%的top-5准确率，优于AlexNet，且微调用的标签数量少了100倍
# 1 Introduction
无监督下学习有效的视觉表示的主流方法有两类：生成式或区分式(generative or discriminative)，
生成式方法学习生成或建模输入空间的像素，但像素级别的学习在计算上较昂贵，且可能对于表示学习是不必要的，
区分式学习使用类似于有监督学习的目标函数，训练网络执行pretext任务，其中输入和标签都从无标签的数据集中得到，许多方法都依赖于启发式以设计pretext任务，而这会限制学习到的表示的泛化性/通用性(generality)
基于在隐空间的对比学习的区分式方法展示了良好前景，取得了SOTA

我们的SimCLR优于之前的工作，且不需要特殊的架构或存储体

为了了解什么造就了好的对比表示学习，我们系统学习了我们框架的主要成分，并说明了：
- 多个数据增强操作的结合在定义可以产生高效表示的对比预测任务时起到了关键的作用，另外，相较于有监督学习，无监督学习可以从更强的数据增强获益更多
- 在表示和对比损失之间引入可学习的非线性变换可以显著提高学习到的表示的质量
- 使用对比交叉熵损失进行表示学习可以从规范化的嵌入(normalized embeddings)和恰当调节的温度参数(temperature paramter)中获益
- 相较于有监督学习，对比学习可以从更大的batch size和更长的训练中获益，和有监督学习相同的是，对比学习也可以从更深和更宽的网络中获益

结合这些发现，我们在ImageNet ILSVRC-2012上取得了自监督和半监督学习的SOTA，在线性评估方案(linear evaluation protocal)下，SimCLR达到了76.5%的top-1准确率，比SOTA提升了7%，在对仅1%的ImageNet标签进行微调后，SimCLR达到了85.5%的top-5准确率，在其他自然图像分类数据集上微调时，SimCLR在10 out of 12个数据集上表现优于或匹配有监督的baseline
# 2 Method
## 2.1 The Constrastive Learning Framework
受最近的对比学习算法启发，SimCLR通过一个在隐空间的对比损失(constrastive loss in the latent space)，最大化相同数据样本的不同增强视图(augmented views)之间的一致性来学习表示

![[SimCLR-Fig2.png]]
SimCLR由四个主成分构成：
- 一个随机的数据增强模块，将任意给定数据样本随机变为两个相关的视图(correlated views)，记为$\symbfit {\tilde x_i}$和$\symbfit {\tilde x_j}$，我们将其视为正例对(postive pair)；本项工作中，我们顺序地应用三个简单的增强：随机裁切(random cropping)、调整大小(resize)回原来的大小、随机颜色失真(random color distortions)和随机高斯模糊(random Gaussian blur)；其中随机裁切和颜色失真的结合对于好的性能十分重要
- 一个NN基编码器(base encoder)$f(\cdot)$，用于从增强的数据样本中提取表示向量，SimCLR框架允许任意选择神经网络类型，我们为了简洁选择了常用的ResNet，因此得到$\symbfit h_i = f(\symbfit {\tilde x_i}) = \text {ResNet}(\symbfit {\tilde x_i})$，其中$\symbfit h_i \in \mathbb R^d$是平均池化层之后的输出
- 一个小的NN映射头(projection head)$g(\cdot)$，用于将表示映射到应用对比损失的空间，我们采用带一层隐藏层的MLP，因此得到$\symbfit z_i = g(\symbfit z_i) = W^{(2)}\sigma(W^{(1)}\symbfit z_i)$，其中$\sigma(\cdot)$是ReLU非线性函数；我们发现在$\symbfit z_i$上定义对比损失比在$\symbfit h_i$上定义对比损失更优
- 一个对比损失函数(constrastive loss function)，对比损失函数为对比预测任务(constrastive prediction task)而定义；给定一个包含了正例对样本$\symbfit {\tilde x_i},\symbfit {\tilde x_j}$的数据集$\{\symbfit {\tilde x_k}\}$，对比预测任务意图对于给定的$\symbfit {\tilde x_i}$，从数据集$\{\symbfit {\tilde x_k}\}_{k\ne i}$中识别出$\symbfit {\tilde x_j}$

我们随机采样包含$N$个样本的minibatch，在该batch上得到的增强样本对上定义对比损失，得到$2N$个数据点，我们不显式地采样负样本，而是将batch内的正例样本对以外的$2(N-1)$个增强样本都视作负样本

令$\text {sim}(\symbfit u, \symbfit v) = \symbfit u^T \symbfit v / \|\symbfit u\| \|\symbfit v\|$表示$\symbfit u,\symbfit v$之间的相似度，实际上就是$\mathscr l_2$规范化的$\symbfit u, \symbfit v$的点积，也就是余弦相似度，则对于一个正例样本对$(i,j)$，损失定义为：
$$\mathscr l_{i,j} = -\log \frac {\exp(\text {sim}(\symbfit z_i,\symbfit z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb 1_{[k\ne i]}\exp(\text {sim}(\symbfit z_i,\symbfit z_k)/\tau)}\tag{1}$$
其中$\mathbb 1_{[k\ne i]}\in \{0,1\}$，是在$k\ne i$时才为1的指示函数，$\tau$是温度参数
最后的损失会在batch内的所有正例样本对上计算，包括了$(i,j)$和$(j,i)$，我们将该损失称为NT-Xent(the normalized tempearture-scaled cross entropy loss)

我们所提出的方法总结于Algorithm1
![[SimCLR-Algorithm1.png]]
## 2.2 Training with Large Batch Size
为了保持简单，我们没有用存储体训练模型，
我们对训练batch size $N$从256到8192进行变化，一个包含了8192个样本的batch对于每个正样本对会16382个负样本，
在大的batch size下使用标准的SGD/Momentum以及线性学习率调度训练可能会不稳定，为了使训练稳定，我们对于所有batch size都使用LARS优化器

**Global BN** 
标准的ResNets使用batch规范化，而在数据并行的分布式训练中，BN均值和方差仅可以在每个设备上局部计算(aggregated locally per device)，
在我们的对比学习框架中，因为正样本对是在同一个设备上计算得到的，模型就可以利用这个局部信息泄露(local information leakage)(也就是同一个设备上计算得到的正样本对特征都经由使用该设备的BN的ResNet得到)，在不提高表示质量的情况下提高预测准确率，
为了解决这个问题，我们在训练时聚合所有设备计算BN均值和方差，其他的方法包括在设备之间shuffle数据样本，或将BN替代为layer规范化
## 2.3 Evaluation Protocal
**Dataset and Metrics**
我们的大多数无监督预训练的研究使用的是ImageNet ILSVRC-2012数据集，也有一些研究使用了CIFAR-10，具体见附录，
我们在广泛的用于迁移学习的数据集上测试了预训练的结果，
为了评估学习到的表示，我们使用广泛应用的线性评估方案(liner evaluation protocal)，即在一个固定的基网络上训练线性分类器，使用测试准确率代表表示质量，
除了线性评估外，我们还比较了SOTA的半监督学习方法和迁移学习方法

**Default setting**
没有特殊说明的情况下，数据增强方法我们使用随机裁切和resize(以及随机翻转)，颜色失真以及高斯模糊，
我们使用ResNet-50作为基编码器网络，以及一个两层的MLP映射头将表示映射到一个128维的隐空间，
损失使用NT-Xent，优化器使用LARS，学习率为4.8(=$0.3\times \text {BatchSize/256}$)，以及衰退因子为$10^{-6}$的权重衰退，
我们将batch size设为4096，训练100个epoch，前10个epoch我们使用线性warmup，
我们用cosine衰退调度对学习率进行衰退，没有restart
# 3 Data Augmentation for Constrastive Representation Learning
**Data augmentation defines predictive tasks**
数据增强以被广泛使用，但并未被考虑作为一个定义对比预测任务的系统的方式，多数现存的方法通过改变架构定义对比预测任务，
例如Hjelm et al. (2018); Bachman et al. (2019)通过限制网络架构中的感受野进行全局到局部的视图预测(global-to-local view prediction)；Oord et al. (2018); Hénaff et al. (2019)通过固定的图像划分过程和一个上下文聚合网络进行邻近视图预测，
我们可以对目标图像进行简单的随机裁切(以及resizing)，就可以创造包括了上述两个任务的一族预测任务，从而避免了上述的复杂性，如Fig3所示
![[SimCLR-Fig3.png]]

这种简单的设计将预测任务从网络架构等其他成分中解耦，通过延伸数据增强方法，然后随机组合它们，就可以定义更广泛的对比预测任务
## 3.1 Composition of data augmentation operations is crucial for learning good representations
本节讨论几个常用的数据增强方法，以系统研究数据增强的影响
一类增强包含了对数据的空间/几何变换，例如裁切和resizing(以及水平翻转)、旋转和镂空(cutout)；另一类增强包含了外观变换，例如颜色失真(包括颜色dropping、明亮brightness、对比contrast、饱和saturation、色度hue)、高斯模糊、Sobel过滤(filtering)
本文研究的数据增强方法如Fig4所示
![[SimCLR-Fig4.png]]

为了理解单独增强的效果和增强结合的重要性，我们研究了在我们的框架中单独应用增强和成对应用增强的效果，
因为ImageNet图片是不同大小的，我们需要都应用随机裁切和resizing，这使得我们难以在不进行裁切的情况下研究其他的增强效果，
为此，我们考虑一个非对称的数据变换设定，具体地说，我们总是随机裁切图像然后resize为相同分辨率(resolution)，然后将想要研究的变换(targeted transformation(s))仅应用于Fig2中的框架中的一个分支，而让另一个分支不做变换，即$t(\symbfit x_i) = \symbfit x_i$，
注意这种非对称的数据增强方式会损害模型性能，但是该设置应该不会过度改变单独的数据增强和它们的结合的影响(impact)(因此可以利用该方式对其进行研究)

![[SimCLR-Fig5.png]]
Fig5展示了单独的增强和增强的组合下的线性评估结果，我们观察到没有一个变换可以在单独使用的情况下就足以学习到好的表示，即使在训练时的对比任务上模型已经可以完美识别正例对(identify the positive pairs)，
当数据增强进行结合，对比预测任务会变得更难，但表示的质量会大大提升

从图中可以看到，随机裁切和随机颜色失真的组合的效果非常好，
我们猜测仅使用随机裁切的一个重要问题是图像中的大多数patch实际上共享一个类似的颜色分布(share a similar color distribution)，
如Fig6所示，通过颜色分布，就足以辨识不同的图片，因此网络可能利用这一捷径来解决预测任务，因此，将裁切和颜色失真结合对于学习到更泛用的特征是很重要的(颜色失真使得模型更难以利用颜色分布信息辨识图片)
![[SimCLR-Fig6.png]]
## 3.2 Constrastive learning needs stronger data augmentation than supervised learning
本节进一步展示颜色增强的重要性，我们调节颜色失真强度(color distortion strength)进行实验，结果如Table1
![[SimCLR-Table1.png]]
可以看到，更强的颜色失真可以显著提升无监督模型的线性评估准确率，
同时，AutoAugment(Cubuk et al., 2019)作为一个复杂的增强策略，其效果并不优于简单的随机裁切+(更强的)颜色失真，
当训练有监督模型时，我们发现在同样的增强策略下，更强的颜色失真非但不会提升性能，反而会损害性能，
因此，我们的实验展示了无监督学习相较于有监督学习可以获益于更强的(颜色)增强，之前的工作报告了有监督学习使用的数据增强对无监督学习同样有益，但我们的工作展示了在有监督学习上不会提高准确率的数据增强方案也可以极大地帮助对比学习
# 4 Architectures for Encoder and Head
## 4.1 Unsupervised constrastive learning benefits(more) from bigger models
![[SimCLR-Fig7.png]]
Fig7说明了增大深度和宽度都可以提高性能，对于有监督模型也有相似的结论

另外，我们发现有监督模型和训练于无监督模型上的线性分类器的性能的差距随着模型大小的增大而减小，这说明了无监督学习下的模型相较于有监督学习下的模型可以从大的模型获益更多(unsuperviesd learning benefits more from bigger models than its supervised counterpart)
## 4.2 A nonlinear projection head improves the representation quality of the layer before it
我们研究SimCLR框架中映射头$g(\symbfit h)$的重要性，
Fig8展示了使用三种不同的映射头结构的线性评估结果，分别是 1) 恒等映射，2) 线性映射，3)SimCLR默认使用的非线性映射，即有一层ReLU隐藏层的MLP
![[SimCLR-Fig8.png]]
我们观察到非线性映射要优于线性映射($+3\%$)，且远远优于无映射($>10\%$)，此外，在使用非线性映射头的时候，映射头前一层的表示($\symbfit h$)，是要远远优于($>10\%$)映射头后一层的表示($\symbfit z = g(\symbfit h))$的，这说明在映射头前一个隐藏层的表示相较于映射头后一层的表示是更好的(the hidden layer before the projection head is a better representation than the layer after)

我们推断非线性映射之前的表示会更优的原因是对比损失会引入信息损失(loss of informatino induced by the constrastive  loss)，
$\symbfit z = g(\symbfit h)$的训练目标是对数据变换保持不变性(trained to be invariant to data transformation)，因此，$g(\cdot)$会移除一些可能对下游任务有帮助的信息，例如物体的颜色或方向，
故通过使用非线性映射$g(\cdot)$，更多的信息可以在$\symbfit h$中形成和保持

为了验证这个猜想，我们用$\symbfit h$或$g(\symbfit h)$来学习预测预训练时对数据应用的变换，其中我们令$g(h) = W^{(2)}\sigma(W^{(1)}h)$，保持输出维度和输入维度一致(2048维)
![[SimCLR-Table3.png]]
结果如Table3所示，可以发现$\symbfit h$显然包含了更多关于对数据应用的变换的信息，而$g(\symbfit h)$则丢失了这些信息
# 5 Loss Functions and Batch Size
## 5.1 Normalized cross entropy loss with adjustable temperature works better than alternatives
我们将NT-Xent损失和其他常用的对比损失函数，例如逻辑斯蒂损失(logistic loss)、边际损失(margin loss)进行比较
Table2展示了这些损失函数以及它们相对于输入的梯度
![[SimCLR-Table2.png]]
(
NT-Xent：
$$
\begin{align}
-\mathscr l(\symbfit u) &= \log \frac {\exp(\symbfit u^T\symbfit v^+/\tau)}{\sum_{\symbfit v\in\{\symbfit v^+,\symbfit v^-\}}\exp(\symbfit u^T\symbfit v/\tau)}\\
&=\symbfit u^T\symbfit v^+/\tau-\log \sum_{\symbfit v\in \{\symbfit v^+,\symbfit v^-\}}\exp(\symbfit u^T\symbfit v/\tau)
\end{align}
$$
$$\begin{align}
-\nabla_{\symbfit u} \mathscr l(\symbfit u) &= \nabla_{\symbfit u}\left(\symbfit u^T\symbfit v^+/\tau\right)-\nabla_{\symbfit u}\left(\log \sum_{\symbfit v\in \{\symbfit v^+,\symbfit v^-\}}\exp(\symbfit u^T\symbfit v/\tau)\right)\\
&=\symbfit v^+/\tau - \frac 1{\sum_{\symbfit v\in \{\symbfit v^+,\symbfit v^-\}}\exp(\symbfit u^T\symbfit v/\tau)}\nabla_{\symbfit u}\left(\sum_{\symbfit v\in \{\symbfit v^+,\symbfit v^-\}}\exp(\symbfit u^T\symbfit v/\tau)\right)\\
&=\symbfit v^+/\tau-\frac 1 {Z(\symbfit u)}\nabla_{\symbfit u}\exp(\symbfit u^T\symbfit v^+/\tau)-\frac 1 {Z(\symbfit u)}\nabla_{\symbfit u}\left(\sum_{\symbfit v\in \symbfit v^-}\exp(\symbfit u^T\symbfit v^-/\tau)\right)\\
&=\symbfit v^+/\tau - \frac {1}{Z(\symbfit u)} \exp(\symbfit u^T\symbfit v^+/\tau)(\symbfit v^+/\tau)  -\frac 1{Z(\symbfit u)}\sum_{\symbfit v^-}\exp(\symbfit u^T\symbfit v^-/\tau)\symbfit (\symbfit v^-/\tau)\\
&=(1-\frac {\exp(\symbfit u^T\symbfit v^+/\tau)} {Z(\symbfit u)})/\tau\symbfit v^+-\sum_{\symbfit v^-}\frac {\exp(\symbfit u^T\symbfit v^-/\tau)}{Z(\symbfit u)}/\tau\symbfit v^-
\end{align}$$
NT-Logistic：
$$\begin{align}
-\mathscr l(\symbfit u) &=-(-\log {\sigma(\symbfit u^T\symbfit v^+/\tau)} -\log{\sigma(-\symbfit u^T\symbfit v^-/\tau)})\\
&=\log {\sigma(\symbfit u^T\symbfit v^+/\tau)} +\log{\sigma(-\symbfit u^T\symbfit v^-/\tau)}
\end{align}$$
$$\begin{align}
-\nabla_{\symbfit u} \mathscr l(\symbfit u) &= \nabla_{\symbfit u}\log \sigma(\symbfit u^T\symbfit v^+/\tau)+\nabla_{\symbfit u}\log \sigma(-\symbfit u^T\symbfit v^-/\tau)\\
&=\sigma(-\symbfit u^T\symbfit v^+/\tau)/\tau\symbfit v^+ -\sigma(\symbfit u^T\symbfit v^-/\tau)\tau\symbfit v^-
\end{align}$$
Margin Triplet：
$$\begin{align}
-\mathscr l(\symbfit u) &=-\max(\symbfit u^T\symbfit v^- - \symbfit u^T\symbfit v^+ + m,0)\\
\end{align}$$
$$\begin{align}
-\nabla_{\symbfit u} \mathscr l(\symbfit u) &= \symbfit v^- -\symbfit v^+\text{ if }\symbfit u^T\symbfit v^+-\symbfit u^T\symbfit v^-<m\text{ else }0
\end{align}$$
)
从这些梯度中，我们观察到 
1) $\mathscr l_2$规范化(即余弦相似度)和温度参数高效地位不同的样本进行了加权，恰当的温度参数可以帮助模型从hard负样本(negatives)学习
2) 和交叉熵不同，其他的目标函数没有用其相对的hardness对负样本进行加权
	因此，使用这些损失的同时必须应用semi-hard负样本挖掘(semi-hard negative minning)：即通过使用semi-hard负项(negative terms)计算梯度，而不是为所有的损失项计算梯度(semi-hard负项就是在损失边际以内，且距离边际最近，比正样本远的负例样本)

为了公平比较，我们对所有的损失函数都进行了相同的$\mathscr l_2$规范化，并调节超参数，报告它们最好的结果，Table4说明了semi-hard负样本挖掘可以帮助提高其他损失的效果，但其最优的结果仍然差于我们的NT-Xent损失
![[SimCLR-Table4.png]]

我们接着测试$\mathscr l_2$规范化(即余弦相似度vs点积)以及温度$\tau$在NT-Xent损失中的重要性，结果如Table5所示

可以看到没有规范化且没有恰当的温度缩放时，性能会显著劣化，没有$\mathscr l_2$规范化，对比任务准确率更高了，但得到的表示更差了
![[SimCLR-Table5.png]]
## 5.2 Constrastive learning benefits (more) from larger batch sizes and longer training
Fig9展示了不同batch大小和不同训练epoch数量的影响，
我们发现在训练epoch数量较少时，更大的batch大小会有显著优势，随着训练epoch数量增加，不同batch大小带来的差异会逐渐降低至消失，
![[SimCLR-Fig9.png]]
和有监督学习不同的是，在对比学习中，更大的batch大小会提供更多的负样本，促进收敛(即对于达到给定的准确率需要更少的epoch数和step数)，更长的训练也会提供更多的负样本，提高效果
# 6 Comparison with State-of-the-art
本节中，和Kolesnikov et al. (2019); He et al. (2019)类似，我们使用3个不同的层宽度(width multiplier $1\times$, $2\times$, $4\times$)的ResNet-50，为了更好的收敛，我们的模型训练了1000个epoch

**Linear evaluation**
见Table6

**Semi-supervised learning**
我们跟随Zhai et al.(2019)的设定，并以类平衡的方式从有标签的ILSVRC-12训练数据集中采样1%或10%的样本(每个类别大约采样出12.8或128张图片)，我们在这些有标签的数据上简单微调整个基网络，同时没有采用正则化

比较结果见Table7

我们的方法在1%和10%的情况下都显著优于SOTA，有趣的是，在完整的ImageNet上微调我们预训练的ResNet-50($2\times, 4\times$)同样要优于从零开始训练的有监督模型

**Transfer learning**
我们在线性评估设定(特征提取器固定)和微调设定下，在12个图像数据集上评估SimCLR的迁移学习表现

Table8展示了ResNet-50($4\times$)的结果

在微调的设定下，我们的自监督模型可以在5个数据集上优于有监督模型，有监督模型则在2个数据集上优于我们的模型，其余的5个数据集上则在统计上是持平的(statistically tied)
# 7 Related Work
让一张图片的在微小的变化下的不同表示互相间一致的思想要追溯到Becker&Hinton(1992)，我们将其进行了延伸，利用了最新的数据增强方法、网络架构和对比损失
在类别标签预测任务中，也有类似的一致性思想(consistency idea)，在半监督学习的背景下由(Xie et al., 2019; Berthelot et al., 2019)探索

**Handcrafted pretext tasks**
最近子监督学习的热潮开始于人工设计的前缀文本任务(pretext tasks)，例如相对patch预测(Doersch et al., 2015)，解拼图游戏(Noroozi & Favaro, 2016)，上色(Zhang et al., 2016)以及旋转预测(Gidaris et al., 2018; Chen et al., 2019)
这些前缀文本任务依赖于特殊的启发式，这限制了学习到的表示的泛用性

**Constrastive visual representation learning**
对比视觉表示学习要追溯回Hadsell et al. (2006)，该方法及之后的方法通过将正例对和负例对之间进行对比来学习表示，
这些方法中，Dosovitskiy et al. (2014)提出了将每个实例视为由一个特征向量表示的类；Wu et al. (2018)提出使用存储体以村塾实例类表示向量，该方法被许多最近的工作采用并延伸
其他的工作探索了用batch内样本进行负样本采样，而不是使用存储体(Doersch & Zisserman, 2017; Ye et al., 2019; Ji et al., 2019)

最近有工作尝试将他们方法的成功归功于最大化隐藏表示之间的互信息(Oord et al., 2018; Hénaff et al., 2019; Hjelm et al., 2018; Bachman et al., 2019)，但是，现在尚不明确对比方法的成功是取决于互信息还是其他特定形式的对比损失(Tschannen et al., 2019)

注意，几乎我们框架中的所有独立成分都在之前的工作中出现过，我们框架相对于之前工作的优越性不能以任意的单个设计选择解释，而是由它们的组合带来
# 8 Conclusion
本工作中，我们提出了一个简单的框架，并将其用于对比视觉表示学习，我们仔细研究了该框架的成分，并展示了不同设计选择的影响

我们的方法和标准的在ImageNet上的有监督学习的差异仅在数据增强的选择、网络的末端的非线性头，以及损失函数，我们框架的成功说明了自监督学习是仍被低估的(undervalued)

# A Data Augmentation Details
# B Additional Experimental Results
## B.1 Batch Size and Training Steps
## B.2 Broader composition of data augmentation further improves performance
## B.3 Effects of Longer Training for Supervised Models
## B.4 Understanding The Non-Linear Projection Head
FigB.3展示了用于计算$\symbfit z = W\symbfit h$的线性映射矩阵$W\in R^{2048\times 2048}$的特征值分布，该矩阵只有相对较少的大特征值，表明了它是近似低秩的(approximately low-rank)
![[SimCLR-FigB.3.png]]

FigB.4是在SimCLR中所得到的最好的ResNet-50(top-1线性评估69.3%)中随机选取的10个类别的$\symbfit h$和$\symbfit z = g(\symbfit h)$的t-SNE可视化，可以看出在$\symbfit h$中类别的表示比在$\symbfit z$中更分离
![[SimCLR-FigB.4.png]]
## B.5 Semi-supervised Learning via Fine-Tuning
## B.6 Linear Evaluation
## B.7 Correlation Between Linear Evaluation and Fine-Tuning
## B.8 Transfer Learning
### B.8.1 Methods
### B.8.2 Results with Standard ResNet
## B.9 CIFAR-10
## B.10 Tuning For Other Loss Functions
# C Further Comparison to Related Methods