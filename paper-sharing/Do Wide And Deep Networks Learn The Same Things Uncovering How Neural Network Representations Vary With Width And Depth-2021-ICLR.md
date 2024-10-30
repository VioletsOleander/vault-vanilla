# 引入
- 标题：Do Wide And Deep Networks Learn The Same Things？ Uncovering How Neural Network Representations Vary With Width and Depth
	- Wide And Deep Networks
		Depth: 网络的层数
		Width: 隐藏层的维度也就是说
		  ax+b 
		(Ref to Fig2 of **_Deep Residual Learning for Image Recognition_**)
		
	```python
		import torch
		import torch.nn as nn

		class Net(nn.Module):
		    def __init__(self):
		        super().__init__()
		        self.fc1 = nn.Linear(128, 1024) # 128 -> 1024
		        self.fc2 = nn.Linear(1024, 128) # 1024 -> 128
		        self.relu = nn.ReLU()
		
		    def forward(self, x):
		        hid = self.relu(self.fc1(x))
		        y = self.fc2(hid) + x # y = f(x) + x
		        return self.relu(y)

		class WiderNet(nn.Module):
		    def __init__(self):
		        super().__init__()
		        self.fc1 = nn.Linear(128, 2048) # 128 -> 2048
		        self.fc2 = nn.Linear(2048, 128) # 2048 -> 128
		        self.relu = nn.ReLU()
		
		    def forward(self, x):
		        hid = self.relu(self.fc1(x))
		        y = self.fc2(hid) + x # y = f(x) + x
		        return self.relu(y)

		class DeeperNet(nn.Module):
			def __init__(self):
		        super().__init__()
		        self.fc1 = nn.Linear(128, 1024) # 128 -> 1024
		        self.fc2 = nn.Linear(1024, 1024) # 1024 -> 1024
		        self.fc3 = nn.Linear(1024,128) # 1024 -> 128
		        self.relu = nn.ReLU()
		
		    def forward(self, x):
		        hid1 = self.relu(self.fc1(x))
		        hid2 = self.relu(self.fc2(hid1))
		        y = self.fc3(hid2) + x # y = f(x) + x
		        return self.relu(y)
	```
	
	- Learn The Same Things/How Neural Network Representations Vary With Width and Depth
		学习到的特征(representations)是如何被网络的宽度和深度影响的,不同深度和宽度的网络它们学习到的特征是一样的吗，是相似的吗
		depth/width： 网络的结构(architecture)
- 关于神经网络结构的相关工作
	- 理论推导上
		万用近似定理(universal approximation theorem)：
		说明在没有对网络的深度和宽度进行限制的情况下，神经网络**有能力**拟合任意精度，任意形状的函数
		
		only shows that such networks **can** be constructed, and provides neither a guarantee of learnability nor a characterization of their performance when trained on finite datasets
		在理论上**可以**构造，但在现实中只能在有限的数据集上训练有限结构的模型
	- 经验理解上(设计实验验证)
		1. optimal accuracy is typically achieved by balancing width and depth
			通过平衡模型的宽度和深度，可以在测试集上取得最优的测试准确率
		2. narrower or shallower neural networks to attain similar accuracy to larger networks when the smaller networks are trained to mimic the larger networks’ predictions
			让小模型的训练目标是大模型的预测结果(大模型去'教'小模型)，可以让小模型在更精简的结构下取得和大模型相似的性能表现
- 本文的聚焦
	develop empirical understanding of the behavior of practical, finite-width neural network architectures after training on real-world data.
	设计实验，通过实验结果获得对网络结构的经验理解
	
	seek to study the impact of width and depth on network internal representations and (per-example) outputs
	探究网络结构的变化对于内部特征/中间特征/隐藏层特征(internal/hidden representations)的影响，以及对于网络输出的影响

# 正文

## 实验基础设置(Experimental setup)
### 概览
- 模型网络
	ResNet famliy
	(Ref to Fig 2,Fig 3 of **_Deep Residual Learning for Image Recognition_**)
- 任务：image classification
- 数据集：
	- CIFAR-10(Canadian Institute for Advanced Research,10 classes)
		60000 32x32 color images 
		共6万张彩图，像素32×32，其中5万张用于训练，1万张用于测试
		
		10 mutually exclusive classes 10个类别
		飞机(airplane)，汽车(automobile)，鸟(bird)，猫(cat)，鹿(deer)，狗(dog), 青蛙(frog)，马(horse)，船(ship)，卡车(truck)
		
		There are 6000 images per class with 5000 training and 1000 testing images per class
		每个类6000张，5000张训练，1000张测试
	- CIFAR-100(Canadian Institute for Advanced Research,100 classes)
		共6万张彩图，像素32×32，其中5万张用于训练，1万张用于测试
		
		100个类(class)，20个超类(superclass)
		
		每个类600张图，500张训练，100张测试
	- ImageNet
		共1400万张图(14,197,122)
		共2.1万类(21,841)
	- ImageNet Large-Scale Visual Recognition Challenge(ILSVRC)
		共1000类
		训练集128万张图
		验证集5万张图
		测试集10万张图
		图片是224×224×3

### CIFAR
- For standard CIFAR ResNet architectures
	the network’s layers are evenly divided between three stages (feature map sizes), with numbers of channels increasing by a factor of two from one stage to the next. 
	根据特征图(feature map)的大小不同，ResNet可以划分为3个阶段，注意特征图的大小减半，通道数(channels)增大一倍
	(Ref to Table 6 of **_Deep Residual Learning for Image Recognition_**)
	
	we adjust the network’s width and depth by increasing the number of channels and layers respectively in each stage
	调整宽度：调整通道(channels)个数
	调整深度：调整堆叠的块(block)个数

- Depth
	Depths for CIFAR-10: {32, 44, 56, 110, 164}
	Depths for CIFAR-100: {32, 44, 56, 110, 164, 218, 224}
- Width 
	Width multipliers: {1, 2, 4, 8, 10} for depths of {14, 20, 26, 38}
(Ref to Table B.1)(小结论：We observe that increasing depth and/or width indeed yields better-performing models)


### ImageNet
- For ImageNet ResNets, ResNet-50 and ResNet-101 architectures differ only by the number of layers in the third (14 × 14) stage. Thus, for experiments on ImageNet, we scale only the width or depth of layers in this stage.
	在原论文中，ResNet-50和ResNet-101仅在第三阶段(特征图的大小是14×14)的结构有差异，故本次实验也仅通过对第三阶段的修改来实现对网络Width和Depth的修改
	(Ref to Table 1 of **_Deep Residual Learning for Image Recognition_**)

start with the ResNet-50 architecture and increase depth or width in the third stage only

## 实验研究方法：特征相似度衡量(Representational similarity measurement)
- 相似度衡量的对象：各隐藏层的特征矩阵之间的相似度
	对于一个样本，从它的原始特征向量开始，每经过网络的一层计算，就得到一层新的特征向量，也就是样本的一个新的表示，我们可以研究这两个特征向量之间的相似度，以发现网络的这一层计算对原特征向量进行了多少的改变，探究网络的行为
	
	对于一个样本集，从它的原始特征矩阵开始，每经过网络的一层计算，就得到一层新的特征矩阵，也就是样本集的一个新的表示，同样可以研究两个特征矩阵之间的相似度，探究网络结构对样本集进行了多少的改变，从而探究网络的行为
	(样本集一般指整个数据集，但受计算资源限制，研究时指的是一个minibatch)
- 衡量两个矩阵之间相似度的方法：Linear CKA
	Linear CKA：
	- $X \in \mathbb R^{m\times p_{1}}\quad Y \in \mathbb R^{m\times p_2}$
		两个特征矩阵(来自于两个隐藏层)，$m\times p$：$m$个样本，每个样本的特征向量是$p$维
	- $K = XX^{T}\quad L = YY^T$
		计算相似度矩阵$K,L$
		$K_{ij} = X_{i.}\cdot X^{T}_{.j}= X_{i.}\cdot X_{j.}$
		$K$的第$ij$元就是$X$的第$i$行(第$i$个样本的特征向量)和$X$的第$j$行(第$j$个样本的特征向量)的内积，$L$同理
		fairness
		reflects the similarities between a pair of examples according to the representations contained in $X$ or $Y$
	- $H = I_{n}-\frac{1}{n}11^{T}, K' = HKH\quad  L' = HLH$
		$\frac{1}{n}11^{T}= \frac{1}{n}\begin{bmatrix}1 & \cdots & 1 \\ \vdots & \ddots & \vdots \\ 1 & \cdots & 1 \end{bmatrix}$
		$HK = K - \frac{1}{n}11^{T}K$
		$\frac{1}{n}11^{T}K$中，每一行都相同，每一行都等于$K$的所有行的均值
		$KH = K - K\frac{1}{n}11^{T}$
		$K\frac{1}{n}11^{T}$中，每一列都相同，每一列都等于$K$的所有列的均值
		
		the similarity matrices with their column and row means subtracted
	- $HSIC(K, L) = vec(K )·vec(L )/(m−1)^2$
		HSIC measures the similarity of these centered similarity matrices by reshaping them to vectors and taking the dot product between these vectors
		
		$HSIC(K,L) = \frac{1}{(m-1)^{2}}\sum_{i=1}^{m}K_{i.}\cdot L_{i.}$
		HSIC is invariant to orthogonal transformations of the representations and, by extension, to permutation of neurons
		
		not invariant to scaling of the original representations
		$HSIC(aK,L) \ne HSIC(K,L)$
		$HSIC(aK,L) \ne HSIC(aK,aL)$
	- $CKA(K,L) = \frac{HSIC(K,L)}{\sqrt{HSIC(K,K)HSIC(L,L)}}$
		CKA further normalizes HSIC to produce a similarity index between 0 and 1 that is invariant to isotropic scaling
		
		$CKA(aK,L) = CKA(K,L)$
		$CKA(aK,aL) = CKA(aK,aL)$
	实际实验中计算的是$CKA_{minibatch}$
	Ref to formula (2)
	在实验中，batch_size = 256,k = 10,无放回抽样
	也就是每次从测试集中抽256个样本，每次抽完不放回，抽十次，计算十次$HSIC(K,L)$，取平均，再计算$CKA$，得到两个隐藏层之间的相似度指标
- CKA representation similarity measure回答的问题
	We begin our study by investigating how the depth and width of a model architecture affects its internal representation structure
	不同结构对内部特征是否有不同的影响
	
	How do representations evolve through the hidden layers in different architectures?
	在不同结构的网络中，内部特征的变化(特征随着网络前向传播的变化)
	
	How similar are different hidden layer representations to each other?
	变化：不同层特征之间的相似度是多少？
### HSIC(Hilbert-Schimit Indenpendence Criterion）希尔伯特-施密特独立性指标

### CKA(Centered kernel alignment)

## 实验发现及其解释
We find that as networks become wider and/or deeper, their representations show a characteristic block structure: many (almost) consecutive hidden layers that have highly similar representations. By training with reduced dataset size, we pinpoint a connection between block structure and model overparameterization — block structure emerges in models that have large capacity relative to the training dataset.

发现1：随着网络的深度和宽度增加，网络中的隐藏层特征逐渐展现出了一种块结构(block sturcture)，即很多连续相连的隐藏层的特征都是极度相似的

发现2：块结构的出现和模型的过参数化(overparameterization)有紧密的联系，即当模型的规模相对于其训练数据集是过大的(冗余的)，块结构就会出现

### 块结构(The Block Structure)
Ref to Figure 1
Emergence of the block structure with increasing width or depth
随着模型的宽度或深度增加，块结构(a large, contiguous set of layers with very similar representations)逐渐出现，即一个范围内的连续的隐藏层的特征都及其相似(a considerable range of hidden layers that have very high representation similarity)
Note:
1. 层数问题
	the total number of layers is much greater than the stated depth of the ResNet, as the latter only accounts for the convolutional layers in the network but we include all intermediate representations
	```python 
	class BasicBlock(nn.Module):
		def forward(self, x):
			residual = x
	
			out = self.conv1(x)
			out = self.bn1(out)
			out = self.relu(out)
	
			out = self.conv2(out)
			out = self.bn2(out)
	
			out += residual
			out = self.relu(out)
	
			return out
	```

	```python
	class BottleNeck(nn.Module):
	    def forward(self, x):
	        residual = x
	
	        out = self.conv1(x)
	        out = self.bn1(out)
	        out = self.relu(out)
	
	        out = self.conv2(out)
	        out = self.bn2(out)
	        out = self.relu(out)
	
	        out = self.conv3(out)
	        out = self.bn3(out)
	        
	        out += residual
	        out = self.relu(out)
	
	        return out
	```
2. block structure在改变了random seed后，同样会出现，即random seed不影响block structure的出现(排除偶然因素)(Ref to Figure D.1)
3. block structure在没有残差连接的网络结构中同样会出现(Ref to Figure C.1)
4. checkerboard-like representation similarity structure
	because representations after residual connections **are more similar to other post-residual representations** than representations inside ResNet block
	残差的成分一直都在网络中通过shortcut传播

### 块结构与模型过参数化(The Block Structure And Model Overparameterization)
Having observed that the block structure emerges as models get deeper and/or wider, we next study whether block structure is a result of this increase in model capacity — namely, is block structure connected to the absolute model size, or to the size of the model relative to the size of the training data?

已经发现了模型的绝对规模增大(深度或宽度增大)将导致块结构的出现，现在要探究块结构与模型能力/容量(model capacity)增长的联系，即块结构是否和模型的相对规模(relative model size: the size of the model **relative to the size of the training data**)有关

relative larget model size $\rightarrow$ model overparameterization
和模型的相对规模有关，即和模型的过参数化有关

Commonly used neural networks have many more parameters than there are examples in their training sets. However, even within this overparameterized regime, larger networks frequently achieve higher performance on held out data
现象：过参数化的模型在测试集上往往表现更佳

we fix a model architecture, but decrease the training dataset size, which serves to inflate the relative model capacity
方法：固定模型规模，减小训练数据集的规模，以控制模型的相对规模

Ref to Figure 2
Block structure emerges in narrower networks when trained on less data
减小数据集的规模，绝对规模小的模型中的特征也开始出现了块结构(因为模型的相对规模增大了)
(Also Ref to Fiture D.2 D.3 D.4)

Together, these observations indicate that block structure in the internal representations arises **in models that are heavily overparameterized relative to the training dataset**
这些发现说明了块结构的出现事实上是和模型的相对规模相关的，随着模型的相对规模增大，块结构会逐渐出现。块结构的出现表明了模型的相对规模过大/模型的容量冗余/模型过参数化

### 对块结构的进一步探索(Probing The Block Structure)
what is happening to the neural network representations as they propagate through the block structure?
特征在经过块结构(表现出块结构的连续的隐藏层)前向传播的时候发生了什么变化？

至少我们现在知道：特征在经过块结构的时候，从CKA衡量的相似度指标来看，它们是保持几乎不变的(层与层之间的相似度都极高)
很显然，如果特征保持数值上的完全不变，即$f(x) = x$，或者是仅仅乘以一个常数，即$f(x) = kx$，用CKA衡量二者之间的相似度，一定得到1

块结构 -> 模型中连续的一段隐藏层之间的**CKA的值都很大/接近1** -> CKA的值接近1说明了什么？ 相似？ 哪些地方相似？ 哪种程度上的相似？ 网络对特征做了什么变化，使得它们经过传播后仍相似？

#### 块结构与第一主成分(The Block Sturcture And The First Principle Component)
- Linear CKA的另一种计算方式/解释方式
	Ref to formula 4
	对formula 4的解释：
	formula 4说明了CKA(X,Y)的值是由一个和式决定的，这个和式由主成分向量的内积乘上样本点映射到这个主成分方向上的方差构成($\lambda_{X}^{i}\lambda_{Y}^{j} {\langle {u_{X}^{i},u_{Y}^{j}}\rangle}^2$)
	
	第一主成分解释的方差($\lambda_{X}^1,\lambda_{Y}^1$)最多(即将样本都投影到第一主成分方向上后，计算得到的方差是最大的)，即这个和式中，最大的一个项是$\lambda_{X}^{1}\lambda_{Y}^{1} {\langle {u_{X}^{1},u_{Y}^{1}}\rangle}^2$
	
	即CKA中，最大的一个项是$\frac{\lambda_{X}^{1}}{\sqrt{\sum{\lambda_i}^2}}\frac{\lambda_{Y}^{1}}{\sqrt{\sum{\lambda_j}^2}}{\langle u_{X}^{1},u_{Y}^{1}\rangle}^2$
	
	As the fraction of the variance explained by the first principal components approaches 1, CKA reflects the squared alignment between these components ${\langle u_{X}^{1},u_{Y}^{1}\rangle}^2$
	随着第一主成分解释的方差比例的增大($\lambda_{1}$在$\sum\lambda_i$中占的比例增大，$\frac{\lambda_1}{\sum\lambda_i}$逐渐接近1)，CKA越能反映X和Y的两个第一主成分向量($u_{X}^{1},u_{Y}^{1}$)的方向的关系(CKA的值主要由${\langle u_{X}^{1},u_{Y}^{1}\rangle}^2$决定)
	
	**在第一主成分解释的方差比例很大的情况下**，X和Y的CKA的值接近1可以说明X和Y的第一主成分方向非常相近

Ref to Figure D.5
We find that, in networks with a visible block structure, the first principal component explains a large fraction of the variance, whereas in networks with no visible block structure, it does not, suggesting that the block structure reflects the behavior of the first principal component of the representations
通过计算可以发现：块结构出现的层数中，第一主成分解释的方差比例比较大，而在没有块结构的层数中，比例则比较小 -> 二者有关联

Ref to Figure 3
- Ref to top right,bottom left
	layers in the block structure have a highly dominant first principal component
	块结构出现的层数中，第一主成分解释了大部分方差，占主导地位
- Ref to top left
	This principal component is also preserved throughout the block structure, seen by comparing the squared cosine similarity of the first principal component across pairs of layers
	在块结构中，第一主成分的方向在前向传播时一直保持
	
	这个结论其实不需要实验验证也可以知道，因为块结构就是由CKA来定义的，而当第一主成分所占比例较大且CKA的值接近1时，就可以说明第一主成分的方向是十分相近的
- Ref to bottom right
	after removing the first principal component from the representations, the block structure is highly reduced — the block structure arises from propagating the first principal component
	将第一主成分减去，块结构大部分消失 -> 块结构的出现来源于网络在前向传播过程中一直保持着第一主成分

因：网络在前向传播中保持第一主成分
果/现象：用CKA来衡量层间相似度，会发现网络存在块结构

Ref to Figure D.8
Although layers inside the block structure have representations with high CKA and similar first principal components, each layer nonetheless computes a nonlinear transformation of its input. Appendix Figure D.8 shows that the sparsity of ReLU activations inside and outside of the block structure is similar. In particular, ReLU activations in the block structure are sometimes in the linear regime and sometimes in the saturating regime, just like activations elsewhere in the network.
ReLU激活层的稀疏性在跳动说明块结构中虽然保持了样本特征空间的第一主成分，但网络所做的仍是非线性的变换

extra note:
(第一主成分解释的方差比例大：数据具有强烈的集中趋势，数据具有明显的相关性，数据可能存在主导因素
第一主成分解释的方差比例小：数据的分散性较大，数据相关性较低，数据可能包含多个相对独立的主要因素)

#### 线性探查与消除块结构(Linear Probes And Collapsing The Block Structure)
how these preserved representations impact task performance throughout the network, and whether the block structure can be collapsed in a way that minimally affects performance
保持主成分并前向传播的这一行为是如何影响模型效果的，如何在对模型性能产生最小的影响的情况下消除块结构

- 实验方法：linear probe
	对每一层的中间隐藏层的特征都作为最后激活层的特征，接上一层全连接层，观察测试表现
- 实验发现 Ref to Figure 4
	In models without the block structure (first 2 panes), we see a monotonic increase in accuracy throughout the network
	对于块结构之外的隐藏层，随着深度增加，准确率也在增加，意味着模型在逐渐提取对任务更加有用的特征
	
	but in models with the block structure (last 2 panes), linear probe accuracy shows little improvement inside the block structure
	对于块结构内部的隐藏层，模型经过这个块的时候，准确率提升的幅度较小，意味着对于任务有用的特征的提取的速度在减缓，更多的是在保持
	
	Comparing the accuracies of probes for layers pre- and post-residual connections, we find that these connections play an important role in preserving representations in the block structure
	而若将残差连接之前的特征去除，发现准确率显著降低，说明残差块内的函数对于有用特征的提取没有起到太大作用，模型只是利用了残差连接对于特征进行了保持
	$y = f(x) + x$中，$f(x)$没有提取到对任务较为有用的特征，$y$主要还是保持了$x$的有效性

- 实验方法：删除特定的残差块，保持残差连接，观察整个模型的效果表现，即确认$y = f(x) + x$中，$f(x)$是否提取到了对于任务有用的特征
- 实验发现：
	the magnitude of the drop in accuracy appears to be connected to the size and the clarity of the block structure present
	块结构越大，去除$f(x)$后，对模型准确率的影响越小
	
	This result suggests that block structure could be an indication of redundant modules in model design
	过大的块结构说明了模型冗余
	
	the similarity of its constituent layer representations could be leveraged for model compression
	模型压缩
	
#### Depth And Width Effects On Representation Across Models
Ref to Figure 6
对于模型间(同结构，不同初始化)不同层的特征的相似度的探索

对于小模型，初始化不同，学到的特征没有什么差异
对于大模型，初始化不同，特征的分布差异很大

while layers not in the block structure exhibit some similarity, the block structure representations are highly dissimilar across models
尤其是块结构，是独一无二的

Ref to Figure E.1
对于模型间(不同结构)不同层的特征的相似度的探索

the block structure representations remain unique to each model.
同样的结论，块结构对于一个模型来说是独一无二的

#### Depth, Width And Effects On Model Predictions

there is considerable diversity in output predictions at the individual example level
对于一个单一示例的预测，不同模型的准确率差异很大

broadly, architectures that are more similar in structure have more similar output predictions
整体上，相似结构的模型有相似的预测结果

On ImageNet we also find that there are statistically significant differences in class-level error rates between wide and deep models, with the former exhibiting a small advantage in identifying classes corresponding to scenes over objects
在ImageNet上，更宽的模型对于景色的识别能力更强，对于物体的识别能力更弱

As the architecture becomes wider or deeper, accuracy on many examples increases, and the effect is most pronounced for examples where smaller networks were often but not always correct. At the same time, there are examples that larger networks are less likely to get right than smaller network
模型规模增大，表现变强，主要是由于更大的模型可以将小模型难以区分的样本加以区分，对于大模型，也有难以分辨的样本，准确率甚至弱于小模型

### 结论
这篇文章是一篇用实验研究网络结构的文章
研究的模型是ResNet famliy
作者通过实验发现，当模型的相对规模过大时，模型会在其中的连续一段隐藏层中保持并传播特征的第一主成分，显现出block structure
第一主成分的保持和传播主要由Residual shortcut完成
这些连续的隐藏层由于主要是在保持和传播第一主成分，其学习到的特征对于模型表现的贡献相对来说不是很大
可以考虑移除block structure中的一部分残差块，在不对模型表现进行大的影响的情况下进行模型压缩

不同的模型中，其block内的特征之间是不相似的，而block外的特征是相似的

深度和宽度不同的模型，对于相同的样本、相同的类也会有不同的预测
# 附录

## 一个矩阵的四大基础子空间(Four Fundamental Subspaces)
对于任意一个$m\times n$的实矩阵$A$，有四个与其相关的子空间
1. Column space C(A)
	矩阵的列向量张成的空间，是$\mathbb R^m$的子空间
2. Null space N(A)
	矩阵的零空间，即$Ax = 0$的所有解向量张成的空间，是$\mathbb R^n$的子空间，且容易知道，N(A)中所有的向量都与A的行空间中的所有向量正交，即N(A)与C($A^T$)正交，
	对比左零空间的定义，可知$Ax=0$这个式子是矩阵$A$右乘一个向量得零，说明这个向量和矩阵的行空间正交，所以也可以叫它矩阵的右零空间
3. Row space C($A^T$)
	矩阵的行空间，也矩阵的转置的列空间(感觉写成R(A)也挺直观的)，即矩阵的行向量张成的空间，是$\mathbb R^n$的子空间
4. Left nullspace N($A^T$)
	矩阵的左零空间，即$A^{T}x = 0$的所有解向量张成的空间，是$\mathbb R^m$的子空间，容易知道，N($A^T$)中的所有向量都与A的列空间的所有向量正交，即N($A^T$)与C(A)正交，之所以叫左零空间，是因为$A^{T}x = 0$可以写成$x^{T}A = 0$，即一个矩阵$A$左乘一个向量得零，说明这个向量和矩阵的列空间正交
关于四个子空间的维度(正交基的个数)：
1. dim(C(A)) = r(A)
2. dim(C($A^T$)) = r($A^T$)= r(A)
	参考三秩相等定理，容易得到r($A^T$) = r(A)
	证明思路：矩阵的秩是由最大非奇异子阵(非奇异：行列式不为0)的大小决定的，而矩阵转置，行列式不变，则容易得到r($A^T$) = r(A)
	要完整证明三秩相等，之后要再证明矩阵的秩等于其列秩(利用行列式和极大无关组的性质来证)，即可得到三秩相等定理：A的列秩 = r(A) = r($A^T$) = $A^T$的列秩 = A的行秩
	即：A的列秩 = A的行秩 = r(A)
3. dim(N(A)) = n - r(A) = n - dim(C($A^T$))
	A的零空间和A的行空间正交，容易知道二者的交集是空集，并集是全集，
	即N(A)$\cup$C($A^T$) = $\mathbb R^n$，两个空间的正交基的并集就是$\mathbb R^n$的一组正交基
4. dim(N($A^T$)) = m - r(A) = m - dim(C(A))
	A的左零空间和A的列空间正交，容易知道二者的交集是空集，并集是全集，
	即N($A^T$)$\cup$C(A) = $\mathbb R^m$，两个空间的正交基的并集就是$\mathbb R^m$的一组正交基

## SVD(Singular Value Decomposition)
对于任意一个大小为$m\times n$，秩为$r$的实矩阵$A$

### $A^TA$
容易知道$A^TA$是一个大小为$n\times n$，秩为$r$的矩阵
(可以证明对于实矩阵$A$，$r(A) = r(A^{T}) = r(AA^{T})= r(A^{T}A)$，
证明思路是证明$Ax = 0$和$A^{T}Ax = 0$同解，即$A$和$A^TA$零空间相同，
则$n-r(A^{T}A) = n-r(A)\rightarrow r(A) = r(A^{T}A)$)

(利用$Ax = 0$和$A^{T}Ax = 0$同解可以进一步推出的结论：
因为$Ax = 0$和$A^{T}Ax = 0$同解，即N(A) = N($A^TA$)
而行空间和零空间正交，那么R(A) = R($A^TA$)
把$A$换成$A^T$，可以得到$A^Tx = 0$和$AA^{T}x = 0$同解
而由于$A^Tx = 0$和$AA^{T}x = 0$同解，那么N($A^T$) = N($AA^{T}$)
而左零空间和列空间正交，那么R($A^T$) = C($AA^T$)
可以知道，事实上$A^TA$和$A$的行空间和零空间都是相等的
进而也可以知道，$AA^T$和$A^T$的行空间和零空间也都是相等的)

$A^{T}A$的性质：
- $A^{T}A = (AA^{T})^T$
	$A^{T}A$是实对称矩阵
- 对于任意非零$n$元实向量$x$，有$x^{T}A^{T}Ax = (Ax)^{T}Ax = \|Ax\|^2\geqslant 0$
	$A^{T}A$是半正定矩阵，则$A^{T}A$的特征值都大于等于0
	(因为对于$A^{T}A$的特征向量$p$来说，$p^{T}A^{T}Ap = \lambda p^{T}p = \lambda\|p\|^2 \geqslant 0$，则$\lambda \geqslant 0$)

因此，作为实对称矩阵$A^{T}A$一定可以相似对角化，即$A^{T}A$存在$n$个相互正交的特征向量，并且由于$A^{T}A$是半正定矩阵，其相应的特征值都大于等于0
因此对$A^TA$进行特征值分解得到：$$A^{T}A = V\Lambda V^T$$
而因为$r(A^{T}A) = r(V\Lambda V^{T}) = r(\Lambda) = r$，可知$A^{T}A$的特征值中，$r$个大于0，$n-r$个为0
($r(V\Lambda V^{T}) = r(\Lambda)$是因为$V$是正交阵，满秩，可逆)

对于$A^TA$的特征值大于0的特征向量：
令它们为：$v_1,v_2,\cdots,v_r$，且令$V_{1}=[v_1,v_2,\cdots,v_r]$
$V_1$的形状为$n\times r$，$V_1$是正交阵
对于$A^TA$的特征值等于0的特征向量：
令它们为：$v_{r+1},v_{r+2},\cdots,v_n$，且令$V_{2}=[v_{r+1},v_{r+2},\cdots,v_n]$
$V_2$的形状为$n\times (n-r)$，$V_2$是正交阵
$V = [V_1,V_2]$

可知$V_2$是$A^{T}A$的零空间的一组标准正交基，也同时是$A$的零空间的一组标准正交基
而$V_1$与$V_2$正交，
可知$V_1$是$A^{T}A$的行空间的一组标准正交基，也同时是$A$的行空间的一组标准正交基

### $AA^T$
同样的，对于矩阵$AA^T$，容易知道它是一个$m\times m$，秩为$r$的半正定矩阵
对$AA^T$也可以相似对角化：$$AA^T=U\Lambda' U^T$$
而由于$r(AA^T)=r(U\Lambda'U^T)=r(\Lambda')=r$，可以知道$\Lambda'$中有$r$个特征值大于0，$n-r$个特征值为0

对于特征值大于0的特征向量：
令它们为：$[u_1,u_2,u_3,\cdots,u_r]$，且令$U_1=[u_1,u_2,u_3,\cdots,u_r]$
$U_1$的形状为$m\times r$，$U_1$是正交阵
对于特征值等于0的特征向量：
令它们为：$[u_{r+1},u_{r+2},\cdots,u_m]$，且令$U_2=[u_{r+1},u_{r+2},\cdots,u_m]$
$U_2$的形状为$m\times(m-r)$，$U_2$是正交阵
$U=[U_1,U_2]$

可知$U_2$是$AA^{T}$的零空间的一组标准正交基，也同时是$A^T$的零空间的一组标准正交基，也就是$A$的左零空间的一组标准正交基
而$U_1$与$U_2$正交，
可知$U_1$是$AA^{T}$的行空间的一组标准正交基，也同时是$A^T$的行空间的一组标准正交基，也就是$A$的列空间的一组标准正交基

### 奇异值
而$V_1,U_1$之间，也就是$A$的行空间的标准正交基和$A$的列空间的标准正交基有什么关系？
$V_2,U_2$之间，也就是$A$的零空间的标准正交基和$A$的左零空间的标准正交基有什么关系？
$V,U$之间，也就是$A^TA$的特征向量和$AA^T$的特征向量之间有什么关系？

我们已经知道$A^TA$的秩是$r$，
对于其大于零的特征值，有对应的特征向量$v_1,v_2,\cdots,v_r$
并且有：$$A^TAv_i=\lambda_iv_i\quad\lambda_i\ne0$$
将等式两边都左乘$A$：
$$\begin{aligned}AA^TAv_i&=\lambda_iAv_i\quad(\lambda_i\ne0)\\(AA^T)(Av_i)&=\lambda_i(Av_i)\quad(\lambda_i\ne0)\end{aligned}$$
发现：
**对于$A^TA$的特征向量$v_i$($\lambda_i\ne0)$，$Av_i$是$AA^T$的特征向量**
且对于$v_i$，$A^TA$的特征值是$\lambda_i$
对于$Av_i$，$A^TA$的特征值也是$\lambda_i$

对于其等于零的特征值，有对应的特征向量$v_{r+1},v_{r+2},\cdots,v_n$
并且有：$$A^TAv_i=0$$
将等式两边都左乘$A$：
$$\begin{aligned}AA^TAv_i&=0\\(AA^T)(Av_i)&=0\end{aligned}$$
和前面的形式很像，但没有同样的结论，因为$A^TA$和$A$的零空间是一样的，容易知道：
$$A^{T}Av_{i}=0\ {\rightarrow}\ Av_i=0$$
因此$\lambda_i=0$时，$Av_i=0$显然是$AA^T$的零空间中的向量之一，因为$(AA^T)(Av_i)=0$是成立的

不论特征值是不是为0，等式$(AA^T)(Av_i)=\lambda_i(Av_i)$都是成立的
综合两种情况，就是对于$A^TA$：
$$A^{T}A=V\Lambda V^{T}$$
$$A^{T}AV=\Lambda V$$
$$AA^{T}AV=\Lambda AV$$
$$(AA^{T})(AV)=\Lambda (AV)$$
其中$V=[V_1,V_2]=[v_1,\cdots,v_r,v_{r+1},\cdots,v_n]$


同理，对于$AA^T$，我们已经知道$AA^T$的秩也是r
对于其大于零的特征值，有对应的特征向量$u_1,u_2,\cdots,u_r$
并且有：$$AA^Tu_i=\lambda'_iu_i\quad\lambda'_i\ne0$$
将等式两边都左乘$A^T$：
$$\begin{aligned}A^TAA^Tu_i&=\lambda'_iA^Tu_i\quad(\lambda'_i\ne0)\\(A^TA)(A^Tu_i)&=\lambda'_i(A^Tu_i)\quad(\lambda'_i\ne0)\end{aligned}$$
发现：
**对于$AA^T$的特征向量$u_i$($\lambda'_i\ne0)$，$A^Tu_i$是$A^TA$的特征向量**
且对于$u_i$，$AA^T$的特征值是$\lambda'_i$
对于$A^Tu_i$，$AA^T$的特征值也是$\lambda'_i$

对于其等于零的特征值，有对应的特征向量$u_{r+1},u_{r+2},\cdots,v_m$
并且有：$$AA^Tu_i=0$$
将等式两边都左乘$A^T$：
$$\begin{aligned}A^TAA^Tu_i&=0\\(A^TA)(A^Tu_i)&=0\end{aligned}$$
和前面的形式很像，但没有同样的结论，因为$AA^T$和$A^T$的零空间是一样的，容易知道：
$$AA^{T}u_{i}=0\ {\rightarrow}\ A^Tu_i=0$$
因此$\lambda'_i=0$时，$A^Tu_i=0$显然是$AA^T$的零空间中的向量之一，因为$(A^TA)(A^Tu_i)=0$是成立的

不论特征值是不是为0，等式$(A^TA)(A^Tu_i)=\lambda'_i(A^Tu_i)$都是成立的
综合两种情况，就是对于$AA^T$：
$$AA^{T}=U\Lambda' U^T$$
$$AA^{T}U = \Lambda' U$$
$$A^{T}AA^{T}U = \Lambda' A^{T}U$$
$$(A^{T}A)(A^{T}U)=\Lambda'(A^{T}U)$$

其中$U=[U_1,U_2]=[u_1,\cdots,u_r,u_{r+1},\cdots,u_m]$




从上述式子我们可以看出：
**当特征值$\lambda_i\ne0$，对于一个$A^TA$的特征向量$v_i$，一定有一个$AA^T$的特征向量$Av_i$与其对应，相应的特征值相等：$\lambda'_i=\lambda_i$**
将其归一化，写为$\sigma_iu_i(Av_i=\sigma_iu_i)$，其中$\sigma_i$是$Av_i$的长度，$\sigma_i\geqslant0$，$u_i$是单位向量
即：
$$v_{i}\ \rightarrow u_{i}\quad \lambda_i=\lambda'_i\ne0$$


左乘一个$A$是确定的运算，因此**一个$v_i$只能对应一个$u_i$**

同理，对于$AA^T$的特征向量，可以令$A^Tu_i=\sigma'_iv_i$，同样有：
$$u_{i}\ \rightarrow \ v_{i}\quad\lambda'_i=\lambda_i\ne0$$
左乘一个$A^T$是确定的运算，因此**一个$u_i$只能对应一个$v_i$**

并且：
先将$v_i$映射成$u_i$
$$Av_i=\sigma_iu_i\rightarrow u_i=\frac{Av_i}{\sigma_i}$$
再从$u_i$映射回来
$$A^T\sigma_iu_i=A^TAv_i=\lambda_iv_i\rightarrow v_i=\frac{A^Tu_i}{\lambda_i/\sigma_i}$$
依旧得到$v_i$
即：
$$v_i\leftrightarrow u_{i}\quad\lambda_i\ne0$$
这说明：
$A^TA$和$AA^T$的那$r$个大于0的**特征值$\lambda_1,\cdots,\lambda_r$是一样的**，相对应的**特征向量$v_i$和$u_i$也是一一对应的**，$v_i$左乘$A$后就和$u_i$平行，$u_i$左乘$A^T$后就和$v_i$平行
(**这其实也说明了$A$的行空间的一组正交基和$A$的列空间的一组正交基是对应的，可以相互转化**)
而$A^TA$和$AA^T$剩余的特征值也都是0，其中$A^TA$有$n-r$个0特征值，$AA^T$有$m-r$个0特征值

如果从$u_i$开始映射到$v_i$，可以得到：
$$A^Tu_i=\sigma'_iv_i\rightarrow v_i=\frac{A^Tu_i}{\sigma'_i}$$
再映射回来：
$$A\sigma'_iv_i=AA^Tu_i=\lambda_iu_{i}\rightarrow u_i=\frac{Av_i}{\lambda_i/\sigma'_i}$$
结合$$v_i=\frac{A^Tu_i}{\sigma'_i}$$和
$$v_i=\frac{A^Tu_i}{\lambda_i/\sigma_i}$$
容易得到：$$\lambda_i=\sigma_i\sigma'_i$$
其中$$\sigma_i=\|Av_i\|\quad\sigma'_i=\|A^Tu_i\|$$
进一步推导：
$$\begin{aligned}\sigma_i&=\|Av_i\|=\sqrt{(Av_i)^T(Av_i)}\\\sigma_i^2&=v_i^TA^TAv_i=\lambda_iv_i^Tv_i=\lambda_i\end{aligned}$$
$$\begin{aligned}\sigma'_i&=\|A^Tu_i\|=\sqrt{(A^Tu_i)^T(A^Tu_i)}\\\sigma_{i}'^{2}&=u_i^TAA^Tu_i=\lambda_iu_i^Tu_i=\lambda_i\end{aligned}$$
显然：$$\sigma_i=\sigma'_i=\sqrt{\lambda_i}=\|Av_i\|=\|A^Tu_i\|$$
对于$r$个$A^TA$和$AA^T$**共同的**非零的特征值$\lambda_1,\cdots,\lambda_r$，对应了$r$个同样非零的$\sigma_1=\sqrt{\lambda_1},\cdots,\sigma_r=\sqrt{\lambda_r}$，我们把它们称为$A$和$A^T$的奇异值(singular value)
一般我们把它们进行从大到小排列，即：
$$\lambda_1\geqslant\cdots\geqslant\lambda_r\gt0$$
$$\sigma_1\geqslant\cdots\geqslant\sigma_r\gt0$$

### 奇异值分解
我们已经知道了$A$的行空间的标准正交基$V_1$和$A$的列空间的标准正交基$U_1$是可以相互转化的：
$$AV_1=A[v_1,\cdots,v_r]=[\sigma_1u_1,\cdots,\sigma_ru_r]$$
$V_1$中每一列是$A^TA$的特征向量中对应$r$个非零特征值的$r$个单位特征向量
$V$中还有剩余的$n-r$个对应的特征值为0的特征向量
$U_1$中每一列是$AA^T$的特征向量中对应$r$个非零特征值的$r$个单位特征向量
$U$中还有剩余的$m-r$个对应的特征值为0的特征向量

如果把等式左边的$V_1$扩充为$V$：
$$AV=A[v_1,\cdots,v_r,v_{r+1},\cdots,v_n]=[\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]$$
($A$和$A^TA$的零空间相同，因此有$A^TAV_2=0\rightarrow AV_2=0$)

等式的右边是一个$(m\times n)\times(n\times n)\rightarrow m\times n$的矩阵

我们构造一个$m\times n$的矩阵$\Sigma$：
$$\Sigma=
\begin{matrix} 
\sigma_1 & \cdots & \cdots & \cdots & \cdots &0\\ 
\vdots&\ddots& &&&\vdots\\ 
\vdots&&\sigma_r&&&\vdots\\
\vdots& & &0&&\vdots\\
\vdots& & & & \ddots&\vdots\\
0&\cdots & \cdots&\cdots &\cdots &0
\end{matrix}$$
$\Sigma$中从$(1,1)$元到$(r,r)$元排列着$r$个奇异值$\sigma$，其余元素都是0

可以知道：
$$U\Sigma =[u_1,\cdots,u_r,u_{r+1},\cdots,u_m]\Sigma=[\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]$$
等式右边是一个$(m\times m)\times (m\times n)\rightarrow m\times n$的矩阵

显然：$$AV=U\Sigma$$
$V$是正交阵，因此：$$A=U\Sigma V^T$$
**此即$A$的奇异值分解**

式子中
$U:m\times m$，为标准正交阵，是$\mathbb R^m$的一组标准正交基
其中的前$r$列是$A$的列空间的标准正交基，后$m-r$列是$A$的左零空间的标准正交基
$V:n\times n$，为标准正交阵，是$\mathbb R^n$的一组标准正交基
其中的前$r$列是$A$的行空间的标准正交基，后$n-r$列是$A$的零空间的标准正交基
$\Sigma:m\times n$
其中有$r\times r$子阵，子阵的对角线上是$A$的奇异值$\sigma$
一般$\Sigma$中的奇异值都是从大到小排列：$$\sigma_1\geqslant\cdots\geqslant\sigma_r\gt0$$
奇异值分解说明了：
任意一个形状的矩阵都可以被分解为左右两个正交阵，中间一个$\Sigma$的形式
也就是说，对于一个$n$维向量，任意一个线性变换(左乘$A$)都可以看成三个过程：
- 旋转(左乘$V^T$)，或者说变基(变为$V^T$中的基)
- 放缩，前$r$行的放缩系数分别是$\sigma_i$，后$m-r$行变为0(如果$m<n$，会使得向量维数降低，如果$m>n$，会使得向量维数升高)
- 旋转(左乘$U$)，或者说变基(变为$U$中的基)
最后得到经过线性变换后的$m$维向量
(如果把放缩和第二次旋转结合，也可以说先旋转一次，变为$V^T$中的基，然后对$U$中的前$r$个基放缩，放缩系数分别是$\sigma_i$，并将$U$的后$m-r$个基消除，然后再旋转一次，变为$U\Sigma$中的基，显然这次旋转的过程中，会导致向量的后$m-r$元变为0，同样，如果$m<n$，会使得向量维数降低，如果$m>n$，会使得向量维数升高)

如果把$$A = U\Sigma V^T$$展开
$$A = [u_1,\cdots,u_r,u_{r+1},\cdots,u_m]\Sigma \begin{bmatrix}
v_1^T \\
\vdots \\
v_r^T \\
v_{r+1}^T \\
\vdots \\
v_n^T
\end{bmatrix} = [\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]\begin{bmatrix}
v_1^T \\
\vdots \\
v_r^T \\
v_{r+1}^T \\
\vdots \\
v_n^T
\end{bmatrix}$$
即
$$A = \sigma_1u_1v_1^T+\cdots+\sigma_ru_rv_r^T$$
$A$被分解为$r$个秩为1的矩阵的和
我们知道
$$\sigma_iu_i = Av_i\quad u_i=\frac{Av_i}{\sigma_i}$$
$$\sigma_iv_i = A^Tu_i\quad v_i=\frac{A^Tu_i}{\sigma_i}$$
那么将$$\sigma_iu_i = Av_i$$代入容易得到
$$\begin{aligned}
A &= Av_iv_i^T+\cdots+Av_rv_r^T \\
&= A(v_iv_i^T+\cdots+v_rv_r^T) \\
& = A[v_i,\cdots,v_r]\begin{bmatrix}
v_i \\
\vdots \\
v_r
\end{bmatrix} \\
& = AV_1V_1^T
\end{aligned}$$
如果$r(A) = n$我们可以得到
$$V_1V_1^T = E$$
同理，如果$r(A) = m$我们可以得到
$$U_1^TU = E$$
当然这个结论也很显然
因为$r(A) = n$时，$A^TA$满秩，可逆，无零特征值
$r(A) = m$时，$AA^T$满秩，可逆，无零特征值
但要注意在$r(A)< m\ \ \&\&\ \  r(A) < n$时没有这个结论
但无论如何这个等式可以看作先将$A$的行向量投影到$V_1$中的每一列代表的基上，然后再用投影值乘以$V_1^T$中的每一行代表的基上，线性组合复原$A$的每一行
另一种对称的有关$U$的情况同理

## PCA(Principle Component Analysis)
### Prerequisite
我们常常用一个矩阵表示数据集
例如，一个$m\times n$的矩阵$A$，表示有$m$个样本，每个样本用一个$n$维的特征向量表示，也就是有$n$个特征
如果需要做数据压缩，就希望能用更少的特征来表示样本，比如$n$维的特征向量能不能压缩到1维，2维，同时还保持了原来的大部分信息？

原来的一个样本用的是$\mathbb R^n$空间的一个$n$维向量表示，空间中有$n$个相互正交的单位基向量，样本在每一个维度上的取值就是其特征向量在这个维度(这个特征)对应的单位基向量上的投影
比如：
当$n=2$
空间中的两个单位基向量是$e_1=[1,0]\quad e_2=[0,1]$
有样本$x_1 = e_1 + 2e_2$，我们表示为$x_1=[1,2]$
有样本$x_2 = 2e_1 + e_2$，我们表示为$x_2 = [2,1]$

如果想要降到1维，也就是只在$\mathbb R^n$空间中选取了一个方向作为新的基向量，把每个样本
的特征向量投影到这个方向，作为样本在这个方向上(这个新特征上)的取值

比如我们选取$e = [\frac{\sqrt2}{2},\frac{\sqrt2}{2}]$作为新的基(新的特征)，这个新的基(新的特征)是原来的基(原来的特征)的线性组合
则将原来每个样本的特征向量投影到这个方向，得到样本在这个新的特征上的取值
$x_1 = \frac{3\sqrt2}{2}$
$x_2= \frac{3\sqrt2}{2}$
我们将特征空间从2维压缩到了1维，只用一个数来表示样本
但这显然不是一个好的投影方向，$x_1$和$x_2$在投影后失去了区分度了，也就是丢失了大部分信息(可以区分不同样本的信息)

那么一个好的投影方向，就是在投影后仍然可以保持大部分信息，也就是保持住原来数据集中每个样本之间的区分度，在投影之后，原来离得近的样本可能会因为部分信息丢失失去区分度，但我们希望能尽量保持住样本之间的距离关系，原来离得远的样本还是离得比较远

我们用方差来描述一个总体离散程度，方差计算了样本离中心点距离的平方的平均值
总体$X$的方差是：
$$Var(X) = E[(X-E[X])^2]$$
当把样本都投影到一个方向上，每个样本$x_i$都用一个常数$a_i$表示，我们得到一个总体$(a_1,\cdots,a_m)$，这个方向上的样本方差就是：
$$Var = \frac{1}{m}\sum_{i = 1}^m (a_i - u)^2$$
其中$u = \frac{1}{m}\sum_{i = 1}^{m}a_i$，即平均值
当均值为$0$时
$$Var = \frac{1}{m}\sum_{i = 1}^m a_i$$
方差衡量了这一批样本的离散程度
我们需要的投影方向就是可以使得投影后，方差最大的方向

如果仅仅只投影到一个方向上，方差就足够了
如果要投影到两个及以上的方向，我们需要保证第二个方向和第一个方向是正交的，第三个方向和前两个方向都是正交的，以此类推
换句话说，我们要找到的是一组正交基，如果第二个方向和第一个方向不正交，说明第二个方向包含的部分信息可以由第一个方向表示，而我们希望的是第二个方向表示的信息是第一个方向不能表示的，同样，第三个方向的信息是前两个方向不能表示的

如果要衡量两个总体的线性相关程度，统计上用协方差：
$$Cov(x,y) = E[(x-E[x])(y-E[y])]$$
并且有：$$Cov(x,x) = E[(x-E[x])^2]=Var(x)$$
如果两个总体的协方差=$0$，可以认为二者没有线性关联，详见[[#协方差与线性关联]]

当把样本都投影到第一个方向上，每个样本$x_i$都用一个常数$a_i$表示
然后投影到第二个方向上，每个样本$x_i$都用一个常数$b_i$表示
协方差表示为：$$Cov = \frac{1}{m}\sum_{i=1}^m (a_i-u_a)(b_i-u_b)$$
如果$u_a = u_b = 0$
$$Cov = \frac{1}{m}\sum_{i=1}^m a_ib_i$$

### PCA
我们对问题的设置稍作总结
我们目前有：
一个数据集，$m\times n$的矩阵$A$表示
矩阵的每一行表示一个样本的$n$维特征向量
$$A = \begin{bmatrix}
v_1^T \\
\vdots \\
v_m^T
\end{bmatrix}$$
我们令$X = A^T$
$$X = [v_1,\cdots,v_m]$$
矩阵中的每一列表示一个样本的$n$维特征向量
我们通过PCA需要满足：
1. 找到一组正交基$e_1,\cdots,e_n$($n$个相互正交的$n$维单位向量)，将样本的特征向量都分别投影到这组正交基上
	即我们需要找到一个矩阵$P(n\times n)$
	$$P = \begin{matrix}
	e_1^T \\
	\vdots\\
	e_n^T
	\end{matrix}$$
	$$PA^T =PX= \begin{matrix}
	e_1^T \\
	\vdots\\
	e_n^T
	\end{matrix}
	v_1,\cdots,v_m = \begin{matrix}
	e_1^Tv_1 & \cdots &e_1^Tv_m \\
	\vdots & \ddots &\vdots \\
	e_n^Tv_1 & \cdots & e_n^Tv_m \\
	\end{matrix} = Y(n \times m)$$
	$Y$中的第$i$列就是样本$i$投影过后的新的特征向量，其中第一行是样本集的第一主成分，第二行是样本集的第二主成分
2. 满足任意两个不同方向$e_i,e_j$之间，样本的投影值$e_i^Tv_1,\cdots,e_i^Tv_m$和$e_j^Tv_1,\cdots,e_j^Tv_m$的协方差是0
	对$Y$做归一化
	$$Y' = Y - YH$$
	其中其中$H(m\times m)$是每个元素都是$\frac 1 m$的矩阵
	$YH$即将$Y$的每一列都变为$Y$中所有列的和的$\frac 1 m$(所有列的平均)
	即$YH$中的每一列都是$$\begin{matrix}
	\frac {e_1^T} m\sum_{i=1}^m v_i \\
	\vdots \\
	\frac {e_n^T} m\sum_{i=1}^m v_i 
	\end{matrix}$$
	因此$Y - YH$即$Y$的每一列都减去所有列的平均
	即$Y-YH$是
	$$\begin{matrix}
	e_1^Tv_1-\frac {e_1^T} m\sum_{i=1}^m v_i & \cdots &e_1^Tv_m-\frac {e_1^T} m\sum_{i=1}^m v_i \\
	\vdots & \ddots &\vdots \\
	e_n^Tv_1-\frac {e_n^T} m\sum_{i=1}^m v_i & \cdots & e_n^Tv_m-\frac {e_n^T} m\sum_{i=1}^m v_i \\
	\end{matrix}$$
	即$$Y' = \begin{matrix}
	e_1^T(v_1-\frac {1} m\sum_{i=1}^m v_i) & \cdots &e_1^T(v_m-\frac {1} m\sum_{i=1}^m v_i) \\
	\vdots & \ddots &\vdots \\
	e_n^T(v_1-\frac {1} m\sum_{i=1}^m v_i) & \cdots & e_n^T(v_m-\frac {1} m\sum_{i=1}^m v_i) \\
	\end{matrix}$$
	容易知道$Y' = Y-YH$中的第$i$列就是样本$i$新的特征向量再进行归一化处理之后的特征向量
	事实上$Y' = Y-YH = PX - PXH = P(X-XH) = PX'$
	实际计算中往往把先把$X$归一化得到$X'$再直接计算$Y'$
	归一化也可以称作把特征进行中心化，令每个特征的均值变为0
	可以把$Y'$写得简洁一点
	$$Y'(n\times m) = \begin{matrix}
	a_1-u_a & \cdots & a_m-u_a \\
	\vdots & \ddots & \vdots \\
	n_1-u_n & \cdots & n_m-u_n
	\end{matrix} = 
	\begin{matrix}
	(\vec a - \vec u_a)^T \\
	\vdots \\
	(\vec n - \vec u_n)^T
	\end{matrix}$$
	
	归一化处理是为了方便我们计算协方差矩阵$\frac 1 m C_{Y'}$
	$$\begin{aligned}\frac 1 m C_{Y'}(n\times n) = \frac 1 m Y'Y'^T = \frac 1 m \begin{bmatrix}
	(\vec a - \vec u_a)^T \\
	\vdots \\
	(\vec n - \vec u_n)^T
	\end{bmatrix}
	[\vec a-\vec u_a,\cdots,\vec n-\vec u_n]
	\\ \\ =\begin{matrix}
	Var_1&Cov_{12}&\cdots&Cov_{1n} \\
	Cov_{21}&Var_2&&\vdots \\
	\vdots& & \ddots&\vdots \\
	Cov_{1n} & \cdots& \cdots& Var_n
	\end{matrix}
	\end{aligned}$$
	显然，我们希望协方差矩阵是对角阵，即任意两个主成分方向之间的协方差都是0
	$$\frac 1 m C_{Y'} =\begin{matrix}
	Var_1&&& \\
	&Var_2&& \\
	& & \ddots& \\
	& & & Var_n
	\end{matrix}$$
3. 满足$e_1$是所有方向中，使得样本的投影值(样本集的第一主成分)$e_1^Tv_1,\cdots,e_1^Tv_m$的方差是最大的那个方向，$e_2$是所有与$e_1$正交的方向中，使得样本的投影值(样本的第二主成分)$e_2^Tv_1,\cdots,e_2^Tv_m$的方差是最大的那个方向，依此类推

要满足前两个要求，即：
找到正交阵$P$，使得$\frac 1 m C_{Y'}$为对角阵
即$$C_{Y'} = Y'Y'^T = PX'X'^{T}P^T$$为对角阵，不妨设为$\Lambda$
$$PX'X'^{T}P^T = \Lambda$$
$$X'X'^{T} = P^T\Lambda P$$
而容易知道$X'X'^{T}$是对称矩阵，并且是半正定矩阵，因此$X'X'^{T}$存在一组相互正交的$n$维特征向量$Q = [q_1,\cdots,q_n]$，可以对角化$X'X'^{T}$，$X'X'^{T}$的特征值都大于等于0
即
$$X'X'^{T} = Q\Lambda Q^T$$
$$Q = P^T$$

一个很自然的想法是选择这些特征向量作为主成分方向
如果选择$X'X'^{T}$的特征向量作为主成分方向
可以满足第一个要求：主成分方向相互正交
也可以满足第二个要求：不同主成分方向之间的协方差为0

并且容易知道，$\Lambda$对角线上的特征值的$\frac 1 m$就是对应的主成分方向上的方差
一般把特征值从大到小排列$$\lambda_1\geqslant\cdots\lambda_n\geqslant 0 $$
特征值最大的特征向量即对应第一主成分方向，依此类推

但把$X'X'^{T}$的特征向量作为主成分方向，是否满足第三个要求？
比如把$q_1$作为第一主成分方向的话，一定保证$q_1$是所有方向中使得方差最大的那个方向吗？
容易知道
$$Var_1=\frac 1 m \sum_{i=1}^m (q_1^Tv_i-\frac 1 m\sum_{i = 1}^mq_1^Tv_i)^2=\frac 1 m \sum_{i=1}^m [q_1^T(v_i-\frac 1 m\sum_{i = 1}^mv_i)]^2$$
平方和的形式实际上可以认为在求一个向量的长度
$$Var_1 =\frac 1 m \|(q_1^TX')^T\|^2 = \frac 1 m \|X'^Tq_1\|^2 = \frac 1 m q_1^TX'X'^Tq_1 = \frac {\lambda_1} m $$
如果将$q_1$换成$\mathbb R^n$中的任意一个其他单位向量比如$t$
因为$q_1,\cdots,q_n$构成了$\mathbb R^n$的一组正交基
可以把$t$写为
$$t = c_1q_1+\cdots+c_nq_n$$
其中由$\|t\| = 1$可得
$$c_1^2+\cdots+c_n^2 = 1$$
因此$$\begin{aligned}
Var_t &= \frac 1 mt^TX'X'^Tt \\
& = \frac 1 m \sum_{i = 1}^nc_iq_i^T
X'X'^T\sum_{i = 1}^nc_iq_i \\
& = \frac {\sum_{i=1}^n c_i\lambda_i} m \leqslant \frac {\lambda_1} m = Var_1
\end{aligned}$$
因此可以说明$q_1$就是所有方向中可以使得方差最大的方向
同理可以证明，$q_2$就是所有和$q_1$正交的方向中可以使得方差最大的方向
依此类推

因此，我们总结PCA的算法步骤
有$m\times n$的矩阵$A$
1. $X = A^T$
2. 归一化$X' = X-XH$
3. 计算协方差矩阵$C = \frac 1 m X'X'^{T}$
4. 对角化协方差矩阵$\frac 1 m X'X'^{T} = \frac 1 m Q\Lambda Q^T$，$\Lambda$中特征值由大到小排列
5. $P = Q^T$，$Y = PX = Q^TX$

因此PCA的核心思想就是寻找一组正交基以对角化协方差矩阵，而这组正交基就是协方差矩阵的特征向量
要补充的两点
- 关于无偏估计
	这里推导中为了简化，计算样本方差和协方差中都是乘上$\frac 1 m$，在实际中，为了得到总体方差的无偏估计，一般是乘上$\frac 1 {m-1}$(因为总体的均值是用样本的均值来估计的，因此自由度应该减一)
- 关于解释的方差
	容易知道，没有进行PCA之前，在原来的每个特征方向上也可以计算方差，所有的特征方向上的方差的和即$\frac 1 {m-1}tr(X'X'^T)$(tr表示矩阵的迹)，显然$$\frac 1 {m-1}tr(X'X'^T) = \frac 1 {m-1} tr(\Lambda)$$
	我们知道对角矩阵$\Lambda$中排列着每个主成分方向上的方差
	因此，进行PCA后，所有特征方向上的方差的和是不变的
	因此，每一个主成分$q_i$解释的方差的比例就是它解释的方差的大小占所有的方向解释的方差的和的比例，即该特征值除以所有特征值的和
	$$explained\ variance\ of\ q_i = \frac {\lambda_i} {tr(\Lambda)}$$

### 与SVD的关系
$A$是原来的矩阵，$A' = A - HA$是进行**归一化之后的矩阵**，即每一行减去了所有行的均值
设$r(A) = r$
$A = X^T, A' = X'^T$
容易知道$A$和$A'$的行空间相等，$X$和$X'$的列空间相等，这四个子空间都相等
$R(A) = R(A') = C(X) = C(X')$

做SVD，即对$X'X'^T$进行对角化，即对$A'^TA'$进行对角化
$$X'X'^T = Q\Lambda Q^T = A'^TA'$$
$Q$是我们需要的正交基，其中的每一列是一个基
其中的$q_1,\cdots,q_r$是我们的主成分方向

而对$A'$进行SVD
$$A' = U\Sigma V^T$$
$\Sigma$中的奇异值$\sigma_i$即$\sqrt \lambda_i$
那么$\sigma_i^2 = \lambda_i$即主成分方向$q_i$上的样本方差的$m-1$倍
因为 $$样本方差 = \frac {\lambda_i} {m-1}$$
而$\sigma_i$即主成分方向$q_i$上的样本标准差的$\sqrt{m-1}$倍
因为 $$样本标准差 = \sqrt {\frac {\lambda_i} {m-1} } = \frac {\sigma_i} {\sqrt {m-1}}$$
我们容易知道$$Q=V$$
$$
A' = U\Sigma Q^T
$$
$$A' = [u_1,\cdots,u_m]\  \Sigma\  \begin{bmatrix}
q_1^T \\
\vdots \\
q_n^T
\end{bmatrix} = [\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]\ \begin{bmatrix}
q_1^T \\
\vdots \\
q_n^T
\end{bmatrix}$$
$$\begin{aligned}
A' &= \sigma_1u_1q_1^T+\cdots+\sigma_ru_rq_r^T \\
& = A'q_1q_1^T+\cdots+A'q_rq_r^T \\
\end{aligned}$$
注意到$Aq_i$为一个列向量，其中的第$j$行即第$j$个样本投影到第$i$个主成分方向上的投影值
这个等式依旧可以视为是利用投影后的投影值乘以对应的基，线性相加，复原原来的特征向量
即$A'$中的第$j$行等于$a_jq_1^T+\cdots+r_jq_r^T$
其中$a_j$到$r_j$分别是第$j$个样本从在第一个主成分方向上的取值到在第$r$个主成分方向上的取值
$A'$写为$r$个秩为1的矩阵的和的形式，可以认为与$r$个主成分关联

而$A'$是归一化之后的形式
$$A' = A-HA = (E-H)A$$
$$(E-H)A = (E-H)A(q_1q_1^T+\cdots+q_rq_r^T)$$
计算行列式可以知道$E-H$是不可逆的
因此对于未做归一化的矩阵$A$，没有相似的结论
因此$A$的PCA是与$A'$的SVD相关的，可以认为对$A$做PCA就是对$A'$做SVD

## PCA与Linear CKA
有矩阵$X(m\times p_1),Y(m\times p_2)$
$X$矩阵中包含了$m$个样本的$p_1$维特征向量。
$Y$矩阵中包含了$m$个样本的$p_2$维特征向量。

矩阵$H = I_m - \frac 1 n 11^T$，作为归一化矩阵
容易知道
$X' = HX,Y' = HY$
$H = H^T$

论文中给出的Linear CKA(之后简称为CKA)的计算方式是
1. 计算$K,L$
	$K = XX^T, L = YY^T$
2. 计算$K',L'$
	$K' = HKH, L' = HLH$
3. 计算$CKA$
	$$CKA(XX^T,YY^T)= CKA(K,L) = \frac {HSIC_0(K,L)}{\sqrt {HSIC_0(K,K)}\sqrt {HSIC_0(L,L)}}$$

	其中$$HSIC_0(K,L) = vec(K')\cdot vec(L')/(m-1)^2$$

我们对其进行解析
首先
$$\begin{aligned}
K'& = HKH \\
&= HXX^TH\\
& = (HX)(HX)^T \\
& = X'X'^T
\end{aligned}$$
同理
$$L' = HLH = Y'Y'^T$$
我们知道
$$HSIC_0(K,L) = vec(K')\cdot vec(L')/(m-1)^2$$
因此$$HSIC_0(K,L) = tr(K'L')/(m-1)^2$$即
$$HSIC_0(XX^T,YY^T) = tr(X'X'^TY'Y'^T)/(m-1)^2$$
而我们可以证明(详细推导过程参考: [[#关于$tr(XX TYY T)$]])
$$tr(X'X'^TY'Y'^T) = \|Y'^TX'\|_F^2$$
而通过矩阵F范数的性质(参考: [[#关于矩阵的F范数]])
我们可以推导出(推导过程参考: [[#关于$ Y TX _F 2$]])
$$\| Y'^TX'\|_F^2 = \sum_{i=1}^{p_1} \sum_{j=1}^{p_2} 
\lambda_X^i\lambda_Y^j\langle u_X^i,u_Y^j\rangle^2$$
其中
$u_X^i$是$X'X'^T$的第$i$个单位特征向量，$\lambda_X^i$是对应的特征值，即奇异值的平方，$u_X^i$之间相互正交
$u_Y^j$是$Y'Y'^T$的第$j$个单位特征向量，$\lambda_Y^j$是对应的特征值，即奇异值的平方，$u_Y^j$之间相互正交

因此
$$\begin{aligned}
HSIC_0(XX^T,YY^T)& = tr(X'X'^TY'Y'^T)/(m-1)^2 \\
& = \| Y'^TX'\|_F^2 /(m-1)^2 \\
& = \sum_{i=1}^{p_1} \sum_{j=1}^{p_2} 
\lambda_X^i\lambda_Y^j\langle u_X^i,u_Y^j\rangle^2 /(m-1)^2
\end{aligned}$$
而
$$
\begin{aligned}
CKA(XX^T,YY^T) &= \frac {HSIC_0(K,L)}{\sqrt {HSIC_0(K,K)}\sqrt {HSIC_0(L,L)}} \\
\\
& = \frac {\sum_{i=1}^{p_1} \sum_{j=1}^{p_2} 
\lambda_X^i\lambda_Y^j\langle u_X^i,u_Y^j\rangle^2 /(m-1)^2}{\sqrt{\sum_{i=1}^{p_1} \sum_{j=1}^{p_1} 
\lambda_X^i\lambda_X^j\langle u_X^i,u_X^j\rangle^2 /(m-1)^2}\sqrt{\sum_{i=1}^{p_2} \sum_{j=1}^{p_2} 
\lambda_Y^i\lambda_Y^j\langle u_Y^i,u_Y^j\rangle^2 /(m-1)^2}}\\
\\
& = \frac {\sum_{i=1}^{p_1} \sum_{j=1}^{p_2} 
\lambda_X^i\lambda_Y^j\langle u_X^i,u_Y^j\rangle^2 }{\sqrt{\sum_{j=1}^{p_1} 
\lambda_X^i\lambda_X^j}
\sqrt{\sum_{j=1}^{p_2} 
\lambda_Y^i\lambda_Y^j}} \\
\\
& = \frac {\sum_{i=1}^{p_1} \sum_{j=1}^{p_2} 
\lambda_X^i\lambda_Y^j\langle u_X^i,u_Y^j\rangle^2 }{\sqrt{\sum_{j=1}^{p_1} 
{\lambda_X^i}^2}
\sqrt{\sum_{j=1}^{p_2} 
{\lambda_Y^i}^2}}
\end{aligned}$$

即我们证明了
$$CKA(XX^T,YY^T) = \frac {\sum_{i=1}^{p_1} \sum_{j=1}^{p_2} 
\lambda_X^i\lambda_Y^j\langle u_X^i,u_Y^j\rangle^2 }{\sqrt{\sum_{j=1}^{p_1} 
{\lambda_X^i}^2}
\sqrt{\sum_{j=1}^{p_2} 
{\lambda_Y^i}^2}}$$
$u_X^i$是$X'X'^T$的第$i$个单位特征向量，$\lambda_X^i$是对应的特征值，即奇异值的平方，$u_X^i$之间相互正交
$u_Y^j$是$Y'Y'^T$的第$j$个单位特征向量，$\lambda_Y^j$是对应的特征值，即奇异值的平方，$u_Y^j$之间相互正交

我们来解释这个式子的含义
我们知道
$u_X^i, \sigma_X^i$都来自于$X'$的奇异值分解
$$X' = U_X\Sigma_XV_X^T$$
我们知道
$$U_X\Sigma_X = X'V_X$$
即
$$\sigma_X^iu_X^i = X'v_X^i$$
$v_X^i$代表了$X$的第$i$主成分方向，那么$\sigma_X^iu_X^i$就表示了这$m$个样本的样本集$X$在进行归一化之后($X'$)的第$i$主成分
同理
$$\sigma_Y^ju_Y^j = Y'v_Y^j$$
$v_Y^j$代表了$Y$的第$j$主成分方向，那么$\sigma_Y^ju_Y^j$就表示了这$m$个样本的样本集$Y$在进行归一化之后($Y'$)的第$j$主成分

而
$$\lambda_X^i\lambda_Y^j\langle u_X^i,u_Y^j\rangle^2 = \langle \sigma_X^iu_X^i,\sigma_Y^ju_Y^j\rangle^2$$
就用二者点积的平方表示了两个主成分之间的关系
联系本文的语境，
这个指标比较的其实是对于同一个数据集，两个不同的隐藏层对于这个数据集的表示方式的不同之处，这两个隐藏层对于这个数据集的表示是相近的意味着数据集的样本之间的相对差异在这两个不同隐藏层的表示下是不大的，反之，说明数据集的样本之间的相对差异在这两个不同隐藏层的表示下是比较大的
而上述的推导告诉我们，在前几个主成分解释的方差(回忆一下主成分解释的方差的计算就是$\lambda_i/{\sum\lambda}$)较大时，这个指标的值实际主要由前几个主成分占主导

### 关于$tr(XX^TYY^T)$
$X$的形状是 $m \times p_1$，$Y$的形状是 $m \times p_2$
$X,Y$都是经过行归一化的矩阵
易知
$$XX^T = \begin{bmatrix} x_{11} & x_{12} & \ldots & x_{1p_1} \\ x_{21} & x_{22} & \ldots & x_{2p_1} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \ldots & x_{mp_1} \end{bmatrix} \begin{bmatrix} x_{11} & x_{21} & \ldots & x_{m1} \\ x_{12} & x_{22} & \ldots & x_{m2} \\ \vdots & \vdots & \ddots & \vdots \\ x_{1p_1} & x_{2p_1} & \ldots & x_{mp_1} \end{bmatrix}$$
对于$XX^T$有
$$(XX^T)_{ij} = \sum_{k=1}^{p_1} x_{ik}x_{jk}$$
同理，对于 $YY^T$有
$$(YY^T)_{ij} = \sum_{k=1}^{p_2} y_{ik}y_{jk}$$
而
$$tr(XX^TYY^T) = \sum_{i=1}^{m} \sum_{j=1}^{m} \left((XX^T)_{ij} \cdot (YY^T)_{ij}\right)$$
将 $(XX^T)_{ij}$ 和 $(YY^T)_{ij}$ 的表达式代入
$$tr(XX^TYY^T) = \sum_{i=1}^{m} \sum_{j=1}^{m} \left(\sum_{k=1}^{p_1} (x_{ik}x_{jk}) \cdot \sum_{l=1}^{p_2} (y_{il}y_{jl})\right)$$
我们研究一下
$$\left(\sum_{k=1}^{p_1} (x_{ik}x_{jk}) \cdot \sum_{l=1}^{p_2} (y_{il}y_{jl})\right)$$
将它写得简单一点
$$\begin{aligned}
\sum_{k=1}^{p_1} (x_{ik}x_{jk}) \cdot \sum_{l=1}^{p_2} (y_{il}y_{jl}) &= \sum_{k=1}^{p_1}a_k\sum_{l=1}^{p_2}b_k \\
& = (a_1+\cdots+a_{p_1})(b_1+\cdots+b_{p_2}) \\
& = \begin{matrix}
a_1b_1+&\cdots&+a_1b_{p_2}+\\
a_2b_1+&\cdots&+a_2b_{p_2}+\\
\vdots & \ddots & \vdots \\
a_{p_1}b_1+&\cdots&+a_{p_1}b_{p_2}
\end{matrix}
\end{aligned}$$
其中$$a_kb_l = x_{ik}x_{jk}y_{il}y_{jl}$$
因此容易知道
$$tr(XX^TYY^T) = \sum_{i=1}^{m} \sum_{j=1}^{m} \left(\sum_{k=1}^{p_1}\sum_{l=1}^{p_2}  x_{ik}x_{jk}y_{il}y_{jl}\right)$$
交换求和顺序得到
$$\begin{aligned}
tr(XX^TYY^T) &= \sum_{k=1}^{p_1} \sum_{l=1}^{p_2} \left(\sum_{i=1}^{m} \sum_{j=1}^{m} x_{ik}x_{jk}y_{il}y_{jl}\right) \\
 &=  \sum_{k=1}^{p_1} \sum_{l=1}^{p_2} \left(\sum_{i=1}^{m} \sum_{j=1}^{m} x_{ik}y_{il}x_{jk}y_{jl}\right) 
\end{aligned}$$
观察括号内和的表达式，我们可以将其分解为两个乘积形式
$$tr(XX^TYY^T) = \sum_{k=1}^{p_1} \sum_{l=1}^{p_2} \left((\sum_{i=1}^{m} x_{ik}y_{il}) (\sum_{j=1}^{m} x_{jk}y_{jl})\right)$$

接着我们研究一下$Y^TX$
形状：$Y^T(p_2\times m)X(m\times p_1) = Y^TX(p_2\times p_1)$
元素：$(Y^TX)_{lk} =\sum_{i=1}^{m} y_{il}x_{ki}$
注意到上式中的两个求和项可以表示为 $Y^TX$ 的元素，即 $(Y^TX)_{lk}$：
$$tr(XX^TYY^T) = \sum_{k=1}^{p_1} \sum_{l=1}^{p_2} (Y^TX)_{lk} \cdot (Y^TX)_{lk} = \sum_{l=1}^{p_2} \sum_{k=1}^{p_1} (Y^TX)_{lk}^2$$
而
$$\| Y^TX\|_F^2 = \sum_{l=1}^{p_2} \sum_{k=1}^{p_1} (Y^TX)_{lk}^2$$
因此
$$tr(XX^TYY^T) = \| Y^TX\|_F^2$$
其中
$$\| Y^TX\|_F^2$$
是$Y^TX$的F范数

### 关于矩阵的F范数
对于任意一个$m\times n$的矩阵$M$
$M$的F范数(Frobenius范数)是指矩阵中所有元素的平方和的平方根
$$\left| M \right|_F = \sqrt{\sum_{i=1}^m\sum_{j=1}^n |M_{i,j}|^2}$$

其中$|M_{i,j}|$表示矩阵$M$在第$i$行第$j$列的元素的绝对值

对于矩阵的F范数，有两个结论：
1. 有
	$$\begin{aligned}
\|M\|_F^2 & = tr(MM^T) 
= tr(M^TM)
= \sum_{i = 1}^r\sigma_i^2
\end{aligned}$$
	其中$r$是$M$的秩，$\sigma_i$是$M$的奇异值
2. F范数对正交变换具有不变性
	即对于一个$m \times m$的正交阵$V$
	有
	$$\|VM\|_F^2 = \|M\|_F^2$$
	同样，对于一个$n \times n$的正交阵$U$
	有
	$$\|MU\|_F^2 = \|M\|_F^2$$
	证明：
	$$\begin{aligned}
	\|VM\|_F^2 &= tr((VM)^T(VM)) \\
	& = tr(M^TV^TVM) \\
	& = tr(M^TM) \\
	& = \|M\|_F^2 \\
	\\
	\|MU\|_F^2 &= tr((MU)(MU)^T) \\
	& = tr(MUU^TM^T) \\
	& = tr(MM^T) \\
	& = \|M\|_F^2 \\
	\end{aligned}$$
### 关于$\| Y^TX\|_F^2$
$X$的形状是 $m \times p_1$，$Y$的形状是 $m \times p_2$
$X,Y$都是经过行归一化的矩阵
从对$X,Y$的SVD我们知道 
$$X = U_X\Sigma_XV_X^T$$
$$Y = U_Y\Sigma_YV_Y^T$$
因此
$$Y^TX = V_Y\Sigma_Y^TU_Y^TU_X\Sigma_XV_X^T$$
故
$$\| Y^TX\|_F^2 = \|V_Y\Sigma_Y^TU_Y^TU_X\Sigma_XV_X^T\|_F^2$$
因为F范数对于正交变换具有不变性(参考: [[#关于矩阵的F范数]])
故
$$\begin{aligned}
\| Y^TX\|_F^2 & = \|V_Y\Sigma_Y^TU_Y^TU_X\Sigma_XV_X^T\|_F^2 \\
& =  \|\Sigma_Y^TU_Y^TU_X\Sigma_X\|_F^2
\end{aligned}$$
其中
$$U_Y\Sigma_Y = 
[u_Y^1,\cdots,u_Y^m]\Sigma_Y = [\sigma_Y^1u_Y^1,\cdots,\sigma_Y^{p_2}u_Y^{p_2}]$$
假设$Y$的秩是$r$，$r\leqslant min\{m,p_2\}$
如果$r<p_2$，我们知道$\sigma_Y^{r+1}$到$\sigma_Y^{p_2}$都是0
所以虽然奇异值一般那只指$r$个大于0的$\sigma_{1}$到$\sigma_r$
我们将其余等于0的用符号$\sigma$表示也是没有影响的
主要是为了公式的简洁
同理
$$U_X\Sigma_X = 
[u_X^1,\cdots,u_X^m]\Sigma_X = [\sigma_X^1u_X^1,\cdots,\sigma_X^{p_1}u_X^{p_1}]$$
其中$r'$是$X$的秩，我们知道$r'\leqslant min\{m,p_1\}$
假设$X$的秩是$r'$，$r'\leqslant min\{m,p_1\}$
如果$r'<p_1$，我们知道$\sigma_X^{r+1}$到$\sigma_X^{p_1}$都是0

因此
$$\Sigma_Y^TU_Y^TU_X\Sigma_X
= \begin{bmatrix}
\sigma_Y^1{u_Y^1}^T \\
\vdots \\
\sigma_Y^{p_2}{u_Y^{p_2}}^T \\
\end{bmatrix}
[\sigma_X^1u_X^1,\cdots,\sigma_X^{p_1}u_X^{p_1}]
= 
\begin{matrix}
\sigma_Y^1\sigma_X^1\langle{u_Y^1}u_X^1\rangle & \cdots & \sigma_Y^1\sigma_X^{p_1}\langle{u_Y^1}u_X^{p_1}\rangle \\
\vdots & \ddots & \vdots \\
\sigma_Y^{p_2}\sigma_X^1\langle{u_Y^{p_2}}u_X^1\rangle & \cdots & \sigma_Y^{p_2}\sigma_X^{p_1}\langle{u_Y^{p_2}}u_X^{p_1}\rangle
\end{matrix}
$$
那么显然
$$\|Y^TX\|_F^2 = \sum_{i=1}^{p_1} \sum_{j=1}^{p_2} (Y^TX)_{ji}^2 = \sum_{i=1}^{p_1} \sum_{j=1}^{p_2} 
\lambda_X^i\lambda_Y^j\langle u_X^i,u_Y^j\rangle^2$$
其中$u_X^i$是$XX^T$的第$i$个单位特征向量，$\lambda_X^i$是对应的特征值，$u_X^i$之间相互正交
$u_Y^j$是$YY^T$的第$j$个单位特征向量，$\lambda_Y^j$是对应的特征值，$u_Y^j$之间相互正交

## 协方差与线性关联

### 相关系数
先回忆一下相关系数的定义：
有两个随机变量$X,Y$，$X,Y$的方差$Var(X)>0,Var(Y)>0$
称$$\rho_{XY}=\frac{Cov[X,Y]}{\sqrt{Var(X)}\sqrt{Var(Y)}}$$
为随机变量$X$和$Y$的相关系数

相关系数有两个性质：
1. $|\rho_{XY}|\leqslant 1$
	证明：
	对任意实数$\lambda$，$$Var[\lambda X+Y]=Cov[\lambda X+Y,\lambda X+Y]=\lambda^2Var[X]+2\lambda Cov[X,Y]+Var[Y]\geqslant0$$
	关于$\lambda$的一元二次函数恒大于等于0说明其判别式恒小于等于0
	故$$4Cov^2[X,Y] - 4Var[X]Var[Y]\leqslant 0$$$$|Cov[X,Y]|\leqslant \sqrt{Var[X]Var[Y]}$$
	$$|\rho_{XY}|\leqslant 1$$
2. 当$\rho_{XY}=1$，存在正常数$a$和实数$b$，使得$Y=aX+b$
	证明：
	当$\rho_{XY}=1$，$Cov[X,Y]=\sqrt{Var[X]Var[Y]}$
	构造随机变量$\sqrt{Var[X]}Y-\sqrt{Var[Y]}X$
	$Var[\sqrt{Var[X]}Y-\sqrt{Var[Y]}X]=2Var[X]Var[Y]-\sqrt{Var[X]Var[Y]}Cov[X,Y]=0$
	随机变量的方差为0，说明它是常数，常数的数学期望为常数本身
	$$\begin{aligned}\sqrt{Var[X]}Y-\sqrt{Var[Y]}X&=E[\sqrt{Var[X]}Y-\sqrt{Var[Y]}X]\\&=\sqrt{Var[X]}E[Y]-\sqrt{Var[Y]}E[X]\\\\
	\frac{Y-E[Y]}{\sqrt{Var[Y]}}&=\frac{X-E[X]}{\sqrt{Var[X]}}
	\\\\Y&=\frac{Var[X]}{Var[Y]}X+E[Y]-\frac{Var[Y]}{Var[X]}E[X]\end{aligned}$$
	
	很显然$Y,X$呈正线性相关
	当相关系数是-1，有类似的结论，$Y,X$呈负线性相关

上述两个性质说明了，如果随机变量$X,Y$之间的协方差取到了最大值1或最小值-1，都可以直接说明两个随机变量是线性相关的，$Y$可以直接用含$X$的一个线性方程表示

### 线性回归
有两个随机变量$X$和$Y$
我们想要用$X$去拟合$Y$，也就是用一个$X$的函数$m(X)$来表示$Y$
希望$E[(Y-m(X))^2]$最小

可以证明
对于每一个可能的$x_i$值，要让上述式子最小，$m(x_i)=E[Y|X=x_i]$，即$m(x)$应该是随机变量$Y$在条件$X=x_i$时的期望，以使得均方误差最小

也就是我们所期待的最优的拟合/回归函数$m(x)=E[Y|X=x]$(注意等式右边也是一个以$x$为自变量的函数)

但是这个函数没有具体的形式，我们一般不可能求出它在每一点$x_i$的取值
为了简化问题，我们会对函数的形式进行假设
线性假设即假设函数具有线性形式:$m(x) = b_0+b_1x$

在线性假设的情况下，我们不需要求出函数在每一点$x_i$处的取值才知道完整的函数，我们只需要知道两个参数$b_0,b_1$的取值应该是多少，就可以知道函数的完整形式，得到函数在每一点处的取值，问题转化为了参数估计的问题

同理，我们希望$E[(Y-b_0-b_1X)^2]$即最小
令两个偏导数都为0$$\frac{\partial E[(Y-b_0-b_1X)^2]}{\partial b_0}=-2E[Y]+2b_0+2b_1E[X]=0$$
$$\frac{\partial E[(Y-b_0-b_1X)^2]}{\partial b_1}=-2Cov[X,Y]-2E[X]E[Y]+2b_0E[X]+2b_1Var[X]+2b_1(E[X])^2=0$$
我们称$b_0,b_1$在满足上述两个偏导数为0时候的取值为$\beta_0,\beta_1$，即$\beta_0,\beta_1$是$b_0,b_1$满足$E[(Y-b_0-b_1X)^2]$最小的情况下的取值，我们可以称其为最优取值(optimal value)

可以知道：$$\beta_1=\frac{Cov[X,Y]}{Var[X]}$$
$$\beta_0=E[Y]-\beta_1E[X]$$
在一般的线性回归问题中，我们会把$\beta_1,\beta_0$加上一个帽子：$\hat \beta_1,\hat \beta_1$，因为我们往往不知道$X,Y$两个总体的统计量具体是多少，一般是用我们观察到的样本对它们进行估计，因此我们求得的$\hat \beta_1,\hat \beta_1$也是对$\beta_1,\beta_0$的估计值

显然可以注意到协方差出现在了式子中：$$\beta_1=\frac{Cov[X,Y]}{Var[X]}$$
注意到
如果协方差取到了最大值或最小值，最优线性估计的斜率$\beta_1$和真实的斜率是一样的
这时二者的相关系数是1或-1，我们说随机变量$X,Y$线性相关
如果两个随机变量的协方差是$0$，可以知道，**两者之间的最优线性估计的斜率**$\beta_1=0$
当两个随机变量的协方差是0，二者的相关系数也是0，我们称随机变量$X,Y$不相关

对于两个向量，或是一个向量组，我们定义线性无关为它们相互正交，表现为任意两个向量的内积都是0
事实上，两个随机变量不相关时，也有类似的性质
当两个随机变量的协方差$Cov[X,Y]=0$
即$$Cov[X,Y]=E[(X-E[X])(Y-E[Y])]=0$$
我们可以不严谨地认为随机变量是一个无限长的向量，第$i$次观测取到的值$x_i$是向量的第$i$维，观测的次数无限，$i$的范围是从1到无穷大
$X=[x_1,x_2,\cdots,x_\infty]$
$Y=[y_1,y_2,\cdots,y_\infty]$
$$E[(X-E[X])(Y-E[Y])]=lim_{n\rightarrow \infty}\frac{1}{n}\sum_{i=1}^{n}(x_i-E[X])(y_i-E[Y])=0$$
我们可以认为这是两个向量分别进行归一化之后求内积的形式，而内积=0
这可以认为是从另一个角度去理解两个随机变量“不相关”的含义

## 希尔伯特空间(Hilbert Space）
We will focus on real Banach and Hilbert spaces, which are, first of all, vector spaces1 over the field $\mathbb R$ of real numbers.
1. Linear Space
	满足线性运算的非空集合，没有规定集合中元素的具体形式，也没有规定加法和数乘如何运算
	对于线性空间$V$中的$n$个元素：$\alpha_1,\alpha_2,...,\alpha_n$，满足：
	- $\alpha_1,\alpha_2,...,\alpha_n$线性无关
	- $V$中任意元素可以由$\alpha_1,\alpha_2,...,\alpha_n$线性表示(因此每一个元素也可以用一个坐标向量表示)
	称$\alpha_1,\alpha_2,...,\alpha_n$是$V$的一个基，而$V$的维数是$n$
	
	线性变换：
	设两个线性空间$U,V$，对于$U$中任一元素$\alpha$，按照某规则$f$，$V$中总有一个确定元素$\beta$与之对应
	$f: U \rightarrow V, \beta = f(\alpha)$
	若$f$可以保持线性关系不变，则称$f$是从$U$到$V$的线性变换
1. Inner Product Space
	定义了内积的线性空间，实数域上(即数乘运算中的数是实数，没有规定集合中元素为实数)的内积空间称为欧几里得空间，即欧氏空间
	两个元素正交即内积为0
3. Cauchy Sequence
	A sequence $\{f_n\}_{n=1}^\infty$ of elements of a normed vector space $(\mathcal F,\|.\|_{\mathcal F})$ is said to be a Cauchy (fundamental) sequence if for every $\epsilon > 0$ there exists $N = N(\epsilon) \in \mathbb N$, such that for all $n, m \geqslant N$, $\| f_{n}-f\|_{\mathcal F}<\epsilon$
	
	A sequence whose elements become arbitrarily close to one another as the sequence progresses.
4. Complete Space
	A space $\mathcal F$ is complete if every Cauchy sequence converges to a member of this space $f \in \mathcal F$
	
	it has a limit, and this limit is in $\mathcal F$
5. Banach Space
	Banach space is a complete normed space, i.e., it contains the limits of all its Cauchy sequences. 
	
	Banach space = complete + normed
6. Hilbert Space
	A Hilbert space is a complete inner product space, i.e., it is a Banach space with an inner product.
	
	A Hilbert space H is an inner product space that is a complete metric space with respect to the norm or distance function **induced by the inner product.**
	
	A Hilbert space = A Banach Space + Inner product
	- 定义了内积(inner product)的空间
	- 利用内积定义了度量(metric), 度量是用内积求得的范数(norm), 或者说，范数由可以由内积推出，而用范数定义了度量
	- 度量是完备(complete)的



The Hilbert space generalizes the Euclidean space to a finite or infinite dimensional space.
在维度有限的情况下，Hilbert space和Euclidean space等价,欧氏空间(Euclidean space)即定义了欧几里得度量(Euclidean metric)的空间

## 再生核希尔伯特空间(Reproducing Kernel Hilbert Space)(RKHS)
- Reproducing Kernal Hilber Space(RKHS)
	a Hilbert space $H$ of functions $f:X\rightarrow R$ with a reproducing kernel $k:X^{2}\rightarrow R$ where $k(x,.) \in H$ and $f(x)=\langle k(x,.),f \rangle$
	- 是函数(functions)的空间
	- 再生核函数(reproducing kernal)，即$k:X^{2}\rightarrow \mathbb R$ 不属于该空间，但$k(x,.) \in H$ 
	- 再生(reproducing)是指$f(x)=\langle k(x,.),f \rangle$，即作内积后不发生变化

对于一个核函数$k(x,y)$,现在有n个点(points):${x_1,x_2,x_3,...,x_n}$，可以由此构造出n个单变量函数$k(x_1,.),k(x_2,.),k(x_3,.)...,k(x_n,.)$，这些单变量函数的所有线性组合可以构成一个再生核希尔伯特空间(RKHS)
RKHS is a function space which is the set of all possible linear combinations of these functions

$$ H = \{ f(.) = \sum_{i=1}^{n}\alpha_{i}k(x_{i},.) \} =\{ f(.) = \sum_{i=1}^{n}\alpha_{i}k_{x_{i}}(.) \}$$

RKHS Being Unique for a Kernel
each kernel generates a new RKHS
RKHS和Kernel一一对应

the basis vectors of RKHS are basis functions named eigenfunctions
RKHS以特征方程为基(RKHS是方程空间，方程空间以特征方程为基，类比向量空间以特征向量为基)

we usually do not know the exact location of pulled points to the RKHS but we know the relation of them as a function
函数(function)表征了关联(relation)

- Reproducing property
	Ref to formula (9)
	the function f is reproduced from the inner product of that function with one of the kernels in the space
	$f(x)$和再生核：$k_x(.)$作内积，结果仍然是$f(x)$

理解：
RKHS中的每一个函数$f(x)$都表征了一种关联，这个关联是自变量$x$和整个数据集的交互，更确切地说是和整个数据集中的每一个样本点：$x_1,x_2,x_3,...,x_n$的交互的总和
因为$f(x)$是$k_{x_{i}}(.)$的线性组合，故$f(x)$是一种关联，这种关联是与每一个$x_i$的关联的线性组合

## $L_p$空间($L_p$Space)
1. $L_p$ norm:
	Consider a function $f$ with domain $[a, b]$
	For $p > 0$, let the $L_p$ norm be defined as:$$\|f\|_{p}= (\int |f(x)|^{p}dx)^{\frac 1p}$$
2. $L_p$ space
	The $L_p$ space is defined as the set of functions with bounded $L_p$ norm:$$L_{p}(a,b)=\{f:[a,b]\rightarrow \mathbb R \; |\; \|f\|_{p}< \infty\} $$

对于定义域是$[a,b]$的单变量函数$f$，若该函数的$L_p$范数是有界的，则该函数属于$L_{p}(a,b)$空间

## Mercer定理(Mercer's Theorem)
Ref to equation(17)~equation(21)
equation(21):$$k(x,y) = \sum_{i=1}^{\infty} \lambda_i\psi_{i}(x)\psi_{i}(y)$$
equation(34):$$k(x,y) = \langle\phi(x),\phi(y)\rangle_{k}= \phi(x)^T\phi(y)$$
the kernel between two points is the inner product of pulled data points to the feature space
$x,y$是两个样本点(points)存在于原空间(Original space/Input space),原空间一般是$d$维欧式空间$\mathbb R^d$
$k(x,y)$是$x,y$特征的内积,$x,y$的特征存在于特征空间(Feature space),特征空间一般是再生核希尔伯特空间(RKHS)

Ref to Remark(4)
- Kernel is a Measure of Similarity
	Inner product is a measure of similarity in terms of angles of vectors or in terms of location of points with respect to origin. According to Eq. (34), kernel can be seen as inner product between feature maps of points; hence, kernel is a measure of similarity between points and this similarity is computed in the feature space rather than input space.

## 参考文献




