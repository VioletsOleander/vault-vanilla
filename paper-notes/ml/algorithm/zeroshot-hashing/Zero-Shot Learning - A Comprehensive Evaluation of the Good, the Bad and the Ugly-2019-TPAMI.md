[Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](<flie:///D:\Learning\paper\2019-TPAMI-Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly.pdf>)
# Abstract
# 1 Introduction
经典的零样本设定(setting)中，测试类别和训练类别是完全不相交的，即仅在不可见类中评估模型
更广泛(genearlized)的零样本设定中，测试类别包括了训练类别，即搜索空间即包括了训练类别，也包括了测试类别

我们从三个方面：方法(methods)、数据集(datasets)、评估协议(evaluation protocal)系统评估零样本学习

我们发布了AWA2(the Animals with Attributes 2)数据集，该数据集和AWA1有大约相同数量的图片，类别数和属性数也完全相同，我们同时提供了图片的ResNet特征

我们强调在验证集上调超参数的必要性，而不是在测试集上调
我们认为每类的平均最高正确率是在数据集中每类的图片数量不平衡时的一个重要的评估标准
我们提出利用预训练过的模型提取图像特征时，要注意该模型预训练的数据集中不包含零样本测试类别，因为图像特征提取也是训练过程的一部分
另外，我们认为小规模且粗粒度数据集，如aPY上的零样本表现是不具备结论性的
我们强调零样本方法也要在罕见的类别(rare classes)中评估
我们推荐更广泛的零样本设定

# 2 Related Work
早期的零样本学习工作利用属性(attributes)，采用两阶段方法以推断不可见类图像的类别，简单来说，在第一阶段预测输入图像的属性，然后在第二阶段通过搜索包含了最相似的一系列属性的类别确认图像的类别
两阶段模型的缺陷在于中间任务和目标任务的域偏移问题(domain shift)

最近的零样本学习直接学习从图像特征空间到语义空间的映射
# 3 Evaulation Methods
给定一个训练集$\mathcal S = \{(x_n,y_n),n=1,\cdots,N\}$，其中$y_n \in \mathcal Y^{tr}$，即都是训练类别
任务是通过最小化带正则的经验损失以学习函数$f:\mathcal X \rightarrow \mathcal Y$
$$\frac 1N \sum_{i=1}^NL(y_n,f(x_n;W))+\Omega(W)\tag{1}$$
其中$L(\cdot)$是损失函数，$\Omega(\cdot)$是正则项
这里的映射是从输入空间到输出空间的，具体定义为
$$f(x;W) = \arg\max_{y\in \mathcal Y}F(x,y;W)\tag{2}$$
## 3.1 Learning Linear Compatibility
Attribute Label Embedding(ALE), Deep Visual Semantic Embedding(DEVSE), Structrued Joint Embedding(SJE)使用双线性的兼容度函数(bi-linear compatibility function)以关联视觉(visual)和辅助(auxiliary)信息
$$F(x,y;W) = \theta(x)^TW\phi(y)\tag{3}$$
其中$\theta(x),\phi(y)$分别是图像嵌入和类别嵌入，二者是给定的，$W$是需要学习的参数
给定一张图片，兼容度学习框架将和该图片计算出最高兼容度分数的类别作为该图片的预测类别

ALE, DEVSE, SJE做了早停处理以隐式地规范化(regularize)随机梯度下降(SGD)
ESZSL, SAE则显式地正则了嵌入模型

**DEVSE**启发于未正则的排序SVM(unregularized ranking SVM)，使用了成对的排序目标(pairwise ranking objective)
$$\sum_{y\in\mathcal Y^{tr}}[\Delta (y_n,y)+F(x_n,y;W)-F(x_n,y_n;W)]\tag{4}$$
其中$\Delta(y_n,y)$当$y_n = y$是为1，否则为0
该目标函数是凸的，并且使用SGD优化

**ALE**排序目标的加权近似
$$\sum_{y\in\mathcal Y^{tr}}\frac {l_{r_{\Delta(x_n,y_n)}}}{r_{\Delta(x_n,y_n)}}[\Delta (y_n,y)+F(x_n,y;W)-F(x_n,y_n;W)]\tag{5}$$

**ESZSL**应用平方误差，并且加入了隐式的正则项
$$\gamma \|W \phi(y)\|^2 + \lambda \|\theta(x)^TW\|^2 + \beta \|W\|^2\tag{7}$$
其中$\gamma,\lambda,\beta$为正则化参数，第一项用于限制投影后属性在特征空间的欧几里得范数，第二项用于限制投影后特征在属性空间的欧几里得范数
该目标函数是凸的，因此有闭式解

**SAE**同样学习从图像嵌入空间到类别嵌入空间的线性投影，但添加了重构损失
$$\min_W \|\theta(x) - W^T\phi(y)\|^2 + \lambda \|W\theta(x)-\phi(y)\|^2\tag{8}$$
Bartels-Stewart算法可以解决该问题

## 3.2 Learning Nonlinear Compatibility
Latent Embeddings(LATEM), Cross Modal Transfer(CMT)为线性兼容性学习框架添加了额外的非线性成分
**LATEM**
$$F(x,y;W_i)=\max_{1\le i\le K}\theta(x)^TW_i\phi(y)\tag{10}$$
其中每一个$W_i$都建模了一个数据的独特的视觉特征，而要选择哪一个$W_i$做映射则是依靠超参数$K$

**CMT**
通过两层的神经网络将图像映射到词语(即类别名称)的语义空间
$$\sum_{y \in \mathcal Y^{tr}}\sum_{x\in \mathcal X_y}\|\phi(y)-W_1tanh(W_1\theta(x))\|^2\tag{11}$$
其中$(W_1,W_2)$是两层神经网络的权重

## 3.3 Learning Intermediate Attribute Classifiers
## 3.4 Hybrid Models
## 3.5 Transductive Zero-Shot Learning Setting
# 4 Datasets
粗粒度、小型的数据集aPY，粗粒度、中型的数据集AWA1
细粒度、中型的数据集SUN和CUB
大型的数据集ImageNet

我们认为类别数量在100到1000的，且图片数量在10K到1M的是中型数据集
## 4.1 Attribute Datasets
aPY：20类训练，12类测试
AWA1：共50类，40类训练，10类测试，共3万多张图片
CUB：共200类，150类训练，50类测试，共1万多张图片
SUN：共717类，645训练，72类测试，共1万多张图片

**Animals with Attributes（AWA2）Dataset**
共50类，和AWA1相同
共3万多张图片

## 4.2 Large-Scale ImageNet
# 5 Evaluation Protocal
我们提出的评估协议包括：图像和类编码(image and class encodings)，数据集划分(dataset splits)，评估标准(evaluation criteria)
## 5.1 Image and Class Embedding
用ResNet-101提取图像特征，ResNet-101模型预训练与ImageNet的1K类的子集上

## 5.2 Dataset Splits
用于训练特征提取网络的数据集(一般是ImageNet 1K)不应该包含任意测试类别的图片，而AWA1、aPY、SUN、CUB的标准分割都存在问题，即部分测试类别被包括在了ImageNet 1K中

所有方法中，这些被包括的测试类别的准确率都高于其他类别，因此我们提出新的分割方式，新的分割方式中，确保没有零样本测试类别和训练类别(包括预训练类别)重叠，但总的测试类别是包括训练类别的