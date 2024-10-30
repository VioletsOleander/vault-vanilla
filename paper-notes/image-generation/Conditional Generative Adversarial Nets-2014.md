# Abstract
生成式对抗网络是训练生成式模型的新方式，在本项工作中我们介绍条件生成式对抗网络，通过对生成器和区分器都输入我们需要以之为条件的数据$y$，就可以构造条件生成式对抗网络

我们展示了该模型可以以类别标签为条件，生成MNIST数字

我们阐述了该模型如何被用于学习一个多模态模型，并提供一个初步的例子，即在一个图像标记应用中，该模型可以为图像生成描述性标记，且该标记不是训练标记的一部分
# 1 Introduction
生成式对抗网络是一个训练生成式模型的新框架，可以避免近似(approximating)许多不可解的概率计算的困难

对抗式网络的优势在于不需要马尔可夫链，只需要反向传播就可获得梯度，学习过程中不需要推理(inference)，并且模型内部可以简单包含广泛的因子和交互(factors and interactions)

在实际的(realistic)样本上，对抗式网络也可以生成SOTA的对数似然估计

非条件的生成式模型中，不能控制生成数据的模式(modes)，如果将模型条件于额外的信息，则有可能引导数据生成过程
模型可以条件于类别标签，或数据的一部分(对于图像修补 inpainting 任务)，甚至是来自不同模态的数据

本项工作中，我们展示如何构建条件生成式对抗网络，我们展示在两个数据集上的实验结果，一项是在MNIST数据集上，以类别标签为条件，另一项是在MIR Flickr 25000上，进行多模态学习
# 2 Related Work
## 2.1 Multi-modal Learning For Image Labelling
有监督神经网络在最近取得了巨大成功(尤其是CNN)，但也存在问题
第一个问题就是这类模型难以拓展到容纳极大数量的预测输出类别(extremely large number of predicted output categoires)，
第二个问题是如今大多数问题聚焦于学习从输入到输出的一对一映射，但许多问题可以更自然地建模为概率上的一对多映射(probabilistic one-to-many mapping)，例如在图像标记任务中，一个图像可以有多个不同的标记(tags)(同义词或近义词)

解决第一个问题一种方法是利用来自其他模态的信息，例如，利用NLP语料库学习标签的向量表示(向量之间的几何关联是语义上有意义的 semantically meaningful)，在该向量空间进行预测时，我们可以保证即使预测错误，也可以尽量接近真实(例如预测成“table”而不是“chair”)，并且我们可以自然泛化到在训练时未见过的标签
[3]已经展示了一个从图像特征空间到词表示空间的简单线性映射也可以提高分类表现

解决第二个问题的一种方法是使用条件概率生成式模型，输入被视作条件变量(conditioning variable)，而一对多映射则建模为一个条件概率预测分布
# 3 Conditional Adversarial Nets
## 3.1 Generative Adversarial Nets
生成式对抗网络包含两个相互对抗的模型：一个生成式模型$G$用于捕获数据分布，一个区分式模型$D$用于估计样本是来自训练数据而非$G$的概率

为了在数据$\symbfit x$上学习生成分布$p_g$，生成器将其建模为从一个先验噪声分布$p_z(z)$到数据空间的映射函数$G(z;\theta_g)$，而区分器$D(x;\theta_d)$输出一个标量，表示$x$是来自训练数据而非$p_g$的概率

$G$和$D$同时进行训练，我们训练$G$以最小化$\log(1-D(G(\symbfit z))$，训练$D$以最大化$\log D(\symbfit x) + \log(1-D(G(\symbfit z))$：
$$\min_G\max_DV(D,G) = \mathbb E_{\symbfit x \sim p_{data}(\symbfit x)}[\log D(\symbfit x)] + \mathbb E_{\symbfit z\sim p_{\symbfit z}(\symbfit z)}[\log (1- D(G(\symbfit z))]\tag{1}$$
## 3.2 Conditional Adversarial Nets
若生成器和区分器都条件于某个额外信息$\symbfit y$，则GAN可以拓展为条件模型，$\symbfit y$可以是任意类型的辅助信息，例如类别标签或来自其他模态的数据
我们通过一层额外的输入层将$\symbfit y$输入给$G$和$D$，使其条件于$\symbfit y$

在生成器中，先验噪声输入$p_{\symbfit z}(\symbfit z)$和条件$\symbfit y$被结合为联合的隐藏表示(joint hidden representation)，注意对抗式训练框架在隐藏表示的构成上有极大的灵活性

在区分器中，$\symbfit x$和$\symbfit y$都作为区分式函数(discriminative function)的输入

目标函数因此写为：
$$\min_G\max_DV(D,G) = \mathbb E_{\symbfit x \sim p_{data}(\symbfit x)}[\log D(\symbfit x| \symbfit y)] + \mathbb E_{\symbfit z\sim p_{\symbfit z}(\symbfit z)}[\log (1- D(G(\symbfit z|\symbfit y))]\tag{2}$$
# 4 Experimental Results
## 4.1 Unimodal
我们在MNIST上训练条件GAN模型，条件于类别标签，类别标签是one-hot向量

生成式网络中，我们从一个单元超立方体内的均匀分布中采样100维的噪声先验$\symbfit z$，$\symbfit z$和$\symbfit y$都首先被映射到各自的第一个隐藏层，激活函数为ReLU，隐藏层维度分别是200和1000，然后二者都被映射到共同的第二个隐藏层，激活函数为ReLU，隐藏层维度是1200，最后再映射到输出层，激活函数为Sigmoid，输出层维度是784维

区分式网络将$\symbfit x$映射到一个有240个单元，分为5段(with 240 units and 5 pieces)的maxout层，将$\symbfit y$映射到有50个单元，分为5段的maxout层，然后一起映射到一个共同的有240个单元，分为4段的maxout层，最后映射到输出层，经过Sigmoid激活得到输出

模型用SGD训练，batchsize为100；初始学习率为0.1，在训练过程中指数下降至0.000001，衰退因子(decay factor)是1.00004；momentum初始值为0.5，在训练过程中逐渐增长至0.7；Dropout在生成器和区分器中都有使用，概率为0.5；在验证集上具有最好对数似然估计的点就是停止点(stopping point)

Table 1展示了MNIST测试集上的Guassian Parzen窗口对数似然估计的结果；我们从10个类别中每个类采样1000个样本，用于Guassian Parzen窗口拟合，然后用Parzen窗口分布估计测试集数据的对数似然
## 4.2 Multimodal
照片网站例如Flickr有大量的有标记图像数据，以及相关的用户生成的元数据(user-generated metadata UGM)，尤其是用户标签(user-tags)

用户生成的元数据和传统的图像标签不同，往往更具描述性，且语义上和人类用自然语言对图片的描述更接近(相较于仅仅只是辨识出图像中的对象)，
UGM的另一个特点是有大量的同义词，不同的用户会用不同的词描述相同的概念，
因此要高效地规范化(normalize)这些标签十分重要，一种方法就是概念词嵌入(conceptual word embedding)，其中相关联的概念会用相似的向量表示

本节我们描述对图像的自动化标记(tagging)，以及多标签预测(multi-label prediction)，我们使用条件GAN生成标签向量(tag-vectors)条件于图像特征的(多模态)条件概率分布

我们使用AlexNet结构，在完整的ImageNet(共有21000个标签)上预训练，作为图像特征提取模型，将最后一个全连接层的4096维输出作为图像表示

我们从YFCC100M数据集的元数据中收集用户标签(user-tags)、标题(titles)和描述(descriptions)构成语料库，在预处理和清理后，训练skip-gram模型提取词向量，词向量维度是200，我们忽略了在词表(vocabulary)中出现次数小于200次的词，最后的词表大小是247465

训练条件GAN时，CNN模型和语言模型都保持固定

我们使用MIR Flickr 25000数据集，使用上述模型提取图像特征和标签特征，没有任何标签的图像会被忽略，图像标注(annotations)会被视作额外的标签，我们使用前15000个样本作为训练集，每个具有多个标签的图像会在数据集中重复出现，每次出现带有其中一个关联的标签

评估时，我们为每张图片生成100个样本，对于每个样本，通过cosine相似度，找到它在词表中最接近的20个词，最后在所有的100×20个词中，选出出现次数最多的十个词

表现最好模型的生成器接受维度为100个Guassian噪声作为噪声先验，然后将它映射到500维的ReLU隐藏层，同时将4096维的图像特征输入映射到2000维的ReLU隐藏层，这两个隐藏层再映射到共同的200维线性层，作为生成的词向量
区分器则将词向量映射到500维的ReLU隐藏层，同时将图像特征映射到1200维的ReLU隐藏层，这两个隐藏层再映射到共同的具有1000个单元，分为3段的maxout层，最后映射到输出层，经过Sigmoid激活得到输出

模型用SGD训练，batchsize为100；初始学习率为0.1，在训练过程中指数下降至0.000001，衰退因子(decay factor)是1.00004；momentum初始值为0.5，在训练过程中逐渐增长至0.7；Dropout在生成器和区分器中都有使用，概率为0.5

超参数和模型结构的选择是通过交叉验证、人工选择以及随机格搜索的结合实现的
# 5 Future Work
本文展现的结果十分初级，但展示了条件GAN的潜力

在当前的实验中，我们是单独地使用每个标签，可以考虑一次使用多个标签(生成问题变为了“集合生成 set generation”)

可以考虑构建联合学习框架，以学习语言模型，[12]展示了我们可以针对特定任务学习一个语言模型