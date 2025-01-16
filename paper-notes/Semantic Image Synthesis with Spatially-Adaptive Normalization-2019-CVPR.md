# Abstract 
We propose spatially-adaptive normalization, a simple but effective layer for synthesizing photorealistic images given an input semantic layout. Previous methods directly feed the semantic layout as input to the deep network, which is then processed through stacks of convolution, normalization, and nonlinearity layers. We show that this is suboptimal as the normalization layers tend to “wash away” semantic information. To address the issue, we propose using the input layout for modulating the activations in normalization layers through a spatially-adaptive, learned transformation. Experiments on several challenging datasets demonstrate the advantage of the proposed method over existing approaches, regarding both visual fidelity and alignment with input layouts. Finally, our model allows user control over both semantic and style. Code is available at https://github.com/NVlabs/SPADE. 
>  我们提出空间自适应规范化，用于在给定输入语义布局时合成逼真图像
>  之前的方法直接将语义布局作为深度网络的输入，语义布局会经过堆叠的卷积层、非线性层、规范化层的处理，我们证明这是次优的，因为规范化层会“冲淡”语义信息
>  为了解决该问题，我们提出通过一个空间自适应的、学习到的变换，用输入语义布局调节规范化层的激活
>  试验证明该方法的视觉保真度和与输入布局的对齐程度具有优势
>  我们的模型还支持用户控制语义和风格

# 1. Introduction 
Conditional image synthesis refers to the task of generating photorealistic images conditioning on certain input data. Seminal work computes the output image by stitching pieces from a single image (e.g., Image Analogies [16]) or using an image collection [7, 14, 23, 30, 35]. Recent methods directly learn the mapping using neural networks [3, 6, 22, 47, 48, 54, 55, 56]. The latter methods are faster and require no external database of images. 
>  条件图像合成指条件于特定输入数据生成逼真图像
>  早期工作通过拼接单个图像的各部分或使用图像集合完成该任务
>  近期方法使用神经网络直接学习映射，更加快速，且不需要外部图像数据库

We are interested in a specific form of conditional image synthesis, which is converting a semantic segmentation mask to a photorealistic image. This form has a wide range of applications such as content generation and image editing [6, 22, 48]. We refer to this form as semantic image synthesis. In this paper, we show that the conventional network architecture [22, 48], which is built by stacking convolutional, normalization, and nonlinearity layers, is at best suboptimal because their normalization layers tend to “wash away” information contained in the input semantic masks. To address the issue, we propose spatially-adaptive normalization, a conditional normalization layer that modulates the activations using input semantic layouts through a spatially-adaptive, learned transformation and can effectively propagate the semantic information throughout the network. 
>  我们关注将语义分割掩码转化为逼真图像，我们称该任务为语义图像合成
>  我们展示传统网络结构 (堆叠卷积、规范化、非线性层) 在该任务是次优的，因为规范化层会“冲淡”输入语义掩码包含的信息
>  我们提出空间自适应规范化，它是一个条件规范化层，使用经过空间自适应、可学习的转换变化过的输入语义布局来调节激活，使得语义信息可以有效在网络中传播

![[pics/Spatially-Adaptive Normalization-Fig 1.png]]

We conduct experiments on several challenging datasets including the COCO-Stuff [4, 32], the ADE20K [58], and the Cityscapes [9]. We show that with the help of our spatially-adaptive normalization layer, a compact network can synthesize significantly better results compared to several state-of-the-art methods. Additionally, an extensive ablation study demonstrates the effectiveness of the proposed normalization layer against several variants for the semantic image synthesis task. Finally, our method supports multimodal and style-guided image synthesis, enabling controllable, diverse outputs, as shown in Figure 1. Also, please see our SIGGRAPH 2019 Real-Time Live demo and try our online demo by yourself. 
>  试验证明空间自适应规范化层会帮助网络合成更好的结果
>  消融试验证实了空间自适应规范化层的有效性
>  方法还支持用户控制图像生成的语义和风格，见 Fig1

# 2. Related Work 
**Deep generative models** can learn to synthesize images. Recent methods include generative adversarial networks (GANs) [13] and variational autoencoder (VAE) [28]. Our work is built on GANs but aims for the conditional image synthesis task. The GANs consist of a generator and a discriminator where the goal of the generator is to produce realistic images so that the discriminator cannot tell the synthesized images apart from the real ones. 
>  **深度生成模型**可以学习合成图像，本工作基于 GAN 实现条件图像合成，GAN 的生成器负责生成逼真图像，判别器负责判断生成图像和真实图像的差异

**Conditional image synthesis** exists in many forms that differ in the type of input data. For example, class-conditional models [3, 36, 37, 39, 41] learn to synthesize images given category labels. Researchers have explored various models for generating images based on text [18,44,52,55]. Another widely-used form is image-to-image translation based on a type of conditional GANs [20, 22, 24, 25, 33, 57, 59, 60], where both input and output are images. Compared to earlier non-parametric methods [7, 16, 23], learning-based methods typically run faster during test time and produce more realistic results. In this work, we focus on converting segmentation masks to photorealistic images. We assume the training dataset contains registered segmentation masks and images. With the proposed spatially-adaptive normalization, our compact network achieves better results compared to leading methods. 
>  **条件图像合成**存在多种形式，差异在于输入数据，例如类条件模型基于类别标签合成图像，又例如图像到图像的转化模型 (基于条件 GAN 实现)
>  本文关注将分割掩码转化为图像，假设训练数据集包含了注册的分割掩码和图像

**Unconditional normalization layers** have been an important component in modern deep networks and can be found in various classifiers, including the Local Response Normalization in the AlexNet [29] and the Batch Normalization (BatchNorm) in the Inception-v2 network [21]. Other popular normalization layers include the Instance Normalization (InstanceNorm) [46], the Layer Normalization [2], the Group Normalization [50], and the Weight Normalization [45]. We label these normalization layers as unconditional as they do not depend on external data in contrast to the conditional normalization layers discussed below. 
>  **无条件规范化层**包括 AlexNet 中的局部相应规范化、Inception-v2 中的批量规范化，以及实例规范化、层规范化、组规范化、权重规范化
>  无条件规范化层不依赖外部数据，和有条件规范化层相反

**Conditional normalization layers** include the Conditional Batch Normalization (Conditional BatchNorm) [11] and Adaptive Instance Normalization (AdaIN) [19]. Both were first used in the style transfer task and later adopted in various vision tasks [3, 8, 10, 20, 26, 36, 39, 42, 49, 54]. Different from the earlier normalization techniques, conditional normalization layers require external data and generally operate as follows. First, layer activations are normalized to zero mean and unit deviation. Then the normalized activations are denormalized by modulating the activation using a learned affine transformation whose parameters are inferred from external data. 
>  **条件规范化层**包括条件批量规范化、自适应实例规范化，二者最初都用于风格迁移任务，之后用于各种视觉任务
>  条件规范化层需要外部数据，一般先将层激活规范化到零均值和单位方差，然后使用仿射变换 (其参数从外部数据推断得到) 调节规范化后的激活，进行去规范化

For style transfer tasks [11, 19], the affine parameters are used to control the global style of the output, and hence are uniform across spatial coordinates. In contrast, our proposed normalization layer applies a spatially-varying affine transformation, making it suitable for image synthesis from semantic masks. 
>  对于风格迁移任务，仿射参数用于控制输出的全局风格，因此仿射参数在空间坐标上是一致的
>  相较之下，本文提出的规范化层应用的是随空间变化的仿射变换，适用于从语义掩码中生成图像

Wang et al. proposed a closely related method for image super-resolution [49]. Both methods are built on spatially-adaptive modulation layers that condition on semantic inputs. While they aim to incorporate semantic information into super-resolution, our goal is to design a generator for style and semantics disentanglement. We focus on providing the semantic information in the context of modulating normalized activations. We use semantic maps in different scales, which enables coarse-to-fine generation. The reader is encouraged to review their work for more details. 
>  Wang et al. 提出过用于图像超分的类似方法，和本文方法一样都使用条件于语义输入的空间自适应调节层
>  他们的目的是将语义信息融入超分辨率，我们的目的是设计解耦风格和语义的生成器
>  我们专注在调节规范化激活中提供语义信息，并使用不同尺度的语义图，使模型可以进行从粗到细粒度的生成

# 3. Semantic Image Synthesis 
Let $\textbf{m}\in\ \mathbb{L}^{H\times W}$ be a semantic segmentation mask where $\mathbb{L}$ is a set of integers denoting the semantic labels, and $H$ and $W$ are the image height and width. Each entry in $\mathbf m$ denotes the semantic label of a pixel. We aim to learn a mapping function that can convert an input segmentation mask $\mathbf m$ to a photorealistic image. 
> 令 $\mathbf m \in \mathbb L^{H\times W}$ 为语义分割掩码，其中 $\mathbb L$ 为语义标签集合 (整数集合)，$H, W$ 为图像高宽
> 掩码 $\mathbf m$ 中每个条目表示对应像素的语义标签 
> 我们的目的是学习映射函数，将输入语义分割掩码 $\mathbf m$ 转化为逼真的图像

**Spatially-adaptive denormalization.** Let $\mathbf{h}^{i}$ denote the activations of the $i$ -th layer of a deep convolutional network for a batch of $N$ samples. Let $C^{i}$ be the number of channels in the layer. Let $H^{i}$ and $W^{i}$ be the height and width of the activation map in the layer. 
>  **空间自适应去规范化**
>  令 $\mathbf h^i$ 表示深度卷积网络第 $i$ 层 (一个批量 $N$ 个样本) 的激活，$C^i$ 表示该层通道数量，$H^i, W^i$ 表示该层激活图的高宽

![[pics/Spatially-Adaptive Normalization-Fig 2.png]]

We propose a new conditional normalization method called the SPatially-Adaptive (DE) normalization (SPADE). Similar to the Batch Normalization [21], the activation is normalized in the channel-wise manner and then modulated with learned scale and bias. Figure 2 illustrates the SPADE design. The activation value at site $(n\in N,c\in C^{i},y\in H^{i},x\in W^{i})$ is 

$$
\gamma_{c,y,x}^{i}(\mathbf{m})\frac{h_{n,c,y,x}^{i}-\mu_{c}^{i}}{\sigma_{c}^{i}}+\beta_{c,y,x}^{i}(\mathbf{m})\tag{1}
$$ 
where $h_{n,c,y,x}^{i}$ is the activation at the site before normalization and $\mu_{c}^{i}$ and $\sigma_{c}^{i}$ are the mean and standard deviation of the activations in channel $c$ : 

$$
\begin{align}
\mu_c^i &= \frac {1}{NH^iW^i}\sum_{n,y,x}h^i_{n,c,y,x}\tag{2}\\
\sigma_c^i &= \sqrt {\frac {1}{NH^iW^i}\sum_{n,y,x}(h^i_{n,c,y,x}-\mu_c^i)^2}\tag{3}
\end{align}
$$

>  我们提出的条件规范化方法称为空间自适应去规范化
>  和批量规范化类似，激活在通道层面规范化，然后使用学习到的尺度和偏置进行调节
>  在第 $i$ 层，我们首先按照 (2), (3) 计算批量中每个通道 $c$ 的均值和标准差，其中 $h_{n, c, y, x}^i$ 表示第 $i$ 层中第 $n$ 个样本在通道 $c$，高宽 $y, x$ 上的原始激活值
>  然后，我们根据 (1) 计算 $(n, c, y, x)$ 上的新激活值，它等于 $h^i_{n, c, y, x}$ 依照之前计算的均值和标准差规范化后，乘上 $\gamma_{c, y, x}^i(\mathbf m)$，再加上 $\beta_{c, y, x}^i(\mathbf m)$

>  因此，整个流程就是先进行批量规范化，然后再根据输入语义分割掩码对规范化后的值进行一定调节，需要注意的是调节参数 $\gamma_{c, y, x}^i, \beta_{c, y, x}^i$ 是依赖于空间坐标，不同空间坐标、不同通道的调节值不同，故规范化后的激活值也是逐元素调节的

The variables $\gamma_{c,y,x}^{i}(\mathbf{m})$ and $\beta_{c,y,x}^{i}(\mathbf{m})$ in (1) are the learned modulation parameters of the normalization layer. In contrast to the BatchNorm [21], they depend on the input segmentation mask and vary with respect to the location $(y,x)$ . We use the symbol $\gamma_{c,y,x}^{i}$ and $\beta_{c,y,x}^{i}$ to denote the functions that convert $\mathbf{m}$ to the scaling and bias values at the site $(c,y,x)$ in the $i$ -th activation map. We implement the functions $\gamma_{c,y,x}^{i}$ and $\beta_{c, y, x}^i$ using a simple two-layer convolutional network, whose design is in the appendix. 
>  $\gamma_{c, y, x}^i(\mathbf m), \beta_{c, y, x}^i(\mathbf m)$ 是规范化层可学习的调节参数，调节参数依赖于输入分割掩码，并且随位置 $(y, x)$ 变化
>  我们用 $\gamma_{c, y, x}^i, \beta_{c, y, x}$ 表示将 $\mathbf m$ 转化为第 $i$ 层的激活图中 $(c, y, x)$ 处的缩放和偏置的函数，该函数使用一个两层卷积网络实现

In fact, SPADE is related to, and is a generalization of several existing normalization layers. First, replacing the segmentation mask $\mathbf{m}$ with the image class label and making the modulation parameters spatially-invariant (i.e., $\gamma_{c,y_{1},x_{1}}^{i}\equiv\gamma_{c,y_{2},x_{2}}^{i}$ and $\beta_{c,y_{1},x_{1}}^{i}\equiv\beta_{c,y_{2},x_{2}}^{i}$ for any $y_{1},y_{2}\in$ $\{1,2,...,H^{i}\}$ and $x_{1},x_{2}\in\{1,2,...,W^{i}\})$ , we arrive at the form of the Conditional BatchNorm [11]. Indeed, for any spatially-invariant conditional data, our method reduces to the Conditional BatchNorm. Similarly, we can arrive at the AdaIN [19] by replacing $\mathbf{m}$ with a real image, making the modulation parameters spatially-invariant, and setting $N=1$ . As the modulation parameters are adaptive to the input segmentation mask, the proposed SPADE is better suited for semantic image synthesis. 
>  SPADE 实际上是现有的几个规范化的泛化
>  将分割掩码 $\mathbf m$ 替换为图像类别标签，并使得调节参数在空间上不变 (仅保持在通道维度变化)，得到的就是条件批量规范化。对于任意满足空间不变性的条件数据，我们的方法等价于条件批量规范化
>  将分割掩码 $\mathbf m$ 替换为真实图像，并使得调节参数在空间上不变，同时设定批量大小 $N=1$，得到的就是自适应实例规范化
>  SPADE 层的输入是分割掩码，调节参数依赖于输入的分割掩码，故 SPADE 更适合语义图像合成

**SPADE generator.** With the SPADE, there is no need to feed the segmentation map to the first layer of the generator, since the learned modulation parameters have encoded enough information about the label layout. Therefore, we discard encoder part of the generator, which is commonly used in recent architectures [22,48]. This simplification results in a more lightweight network. Furthermore, similarly to existing class-conditional generators [36,39,54], the new generator can take a random vector as input, enabling a simple and natural way for multi-modal synthesis [20,60]. 
>  **SPADE 生成器**
>  使用 SPADE 层后，分割掩码就不再需要输入到生成器的第一层，因此 SPADE 层中学习到的调节参数已经编码了足够的标签布局信息
>  因此，我们去掉图像生成器中的编码器部分，和现存的类条件生成器类似，新的生成器接受随机向量作为输入，进而更自然实现了多模态合成 (也就是说，对于相同的条件输入，例如相同的分割图，不同的随机噪声向量可以让生成器生成不同的单合理的输出图像，从而自然地支持了从单一条件生成多种可能的结果)

![[pics/Spatially-Adaptive Normalization-Fig 4.png]]

Figure 4 illustrates our generator architecture, which employs several ResNet blocks [15] with upsampling layers. The modulation parameters of all the normalization layers are learned using the SPADE. Since each residual block operates at a different scale, we downsample the semantic mask to match the spatial resolution. 

>  生成器结构如 Figure 4 所示，包括了数个 ResNet 块和上采样层，ResNet 块中的规范化层都使用 SPADE
>  因为每个残差块都处于不同的尺度，故语义分割掩码在输入 SPADE 前需要先下采样以匹配空间分配率

We train the generator with the same multi-scale discriminator and loss function used in pix2pixHD [48] except that we replace the least squared loss term [34] with the hinge loss term [31,38,54]. We test several ResNet-based discriminators used in recent unconditional GANs [1, 36, 39] but observe similar results at the cost of a higher GPU memory requirement. Adding the SPADE to the discriminator also yields a similar performance. For the loss function, we observe that removing any loss term in the pix2pixHD loss function lead to degraded generation results.  
>  生成器的训练使用了和 pix2pixHD 相同的多尺度判别器和损失函数，差异仅在于将最小二乘损失项替换为了铰链损失项
>  我们测试了几个基于 ResNet 的判别器，得到的结果是相似的
>  我们发现移除 pix2pixHD 中损失函数中任意损失项都会损害生成结果

> [!info] 最小二乘损失和铰链损失
> **最小二乘损失**
> 最小二乘损失常用于回归问题
> 对于某个样本 $x_i$，记真实值为 $y_i$，预测值为 $\hat y_i$，最小二乘损失为 $L_i = (y_i - \hat y_i)^2$ 
> 整个数据集的损失为 $L = \frac 1 N\sum_{i=1}^N L_i = \frac 1 N\sum_{i=1}^N (y_i -\hat y_i)^2$
>  
> 特点:
> 1. 对异常值非常敏感，因为误差被平方了，较大的误差会被放大。
> 2. 在 GANs 中，有时也会用到最小二乘损失来替代传统的交叉熵损失，以减少训练过程中的模式崩溃问题（mode collapse）。
> 
> **铰链损失**
> 铰链损失主要用于 SVMs 以及某些分类器的训练中，尤其是在二分类问题中
> 对于某个样本 $x_i$，记真实值为 $y_i$，预测值为 $\hat y_i$ ，铰链损失表示为 $L_i = \max(0,1-y_i\cdot f(x_i))$ ，其中 $f(x_i)$ 是模型输出的分数或置信度得分
> 整个数据集的损失为 $L = \frac 1 N\sum_{i=1}^N L_i$
> 
> 当样本为正类时，$y_i = +1$ ，若 $f(x_i)\ge 1$，损失为 $0$；否则，损失为 $1-f(x_i)$ 
> 当样本为负类时，$y_i = -1$，若 $f(x_i) \le -1$，损失为 $0$；否则，损失为 $1 + f (x_i)$
> 
> 特点：
> 1. 铰链损失鼓励模型不仅将样本正确分类，而且还要有足够的置信度。它试图最大化决策边界周围的间隔。
> 2. 如果样本已经被正确分类并且有足够的置信度，那么该样本就不会对损失做出贡献。
> 3. 在多类别分类中，可以通过比较目标类别的得分与其他所有非目标类别得分的最大值，拓展成多类别铰链损失（multi-class hinge loss）。

Why does the SPADE work better? A short answer is that it can better preserve semantic information against common normalization layers. Specifically, while normalization layers such as the InstanceNorm [46] are essential pieces in almost all the state-of-the-art conditional image synthesis models [48], they tend to wash away semantic information when applied to uniform or flat segmentation masks. 

Let us consider a simple module that first applies convolution to a segmentation mask and then normalization. Furthermore, let us assume that a segmentation mask with a single label is given as input to the module (e.g., all the pixels have the same label such as sky or grass). Under this setting, the convolution outputs are again uniform, with different labels having different uniform values. Now, after we apply InstanceNorm to the output, the normalized activation will become all zeros no matter what the input semantic label is given. Therefore, semantic information is totally lost. This limitation applies to a wide range of generator architectures, including pix2pixHD and its variant that concatenates the semantic mask at all intermediate layers, as long as a network applies convolution and then normalization to the semantic mask. In Figure 3, we empirically show this is precisely the case for pix2pixHD. Because a segmentation mask consists of a few uniform regions in general, the issue of information loss emerges when applying normalization. 


![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/dec17f5a5b5b8e9055300ef13f9db96ad199078275686a88d271a95a1e307147.jpg) 
Figure 4: In the SPADE generator, each normalization layer uses the segmentation mask to modulate the layer activations. (left) Structure of one residual block with the SPADE. (right) The generator contains a series of the SPADE residual blocks with upsampling layers. Our architecture achieves better performance with a smaller number of parameters by removing the downsampling layers of leading image-to-image translation networks such as the pix2pixHD model [48]. 
In contrast, the segmentation mask in the SPADE Generator is fed through spatially adaptive modulation without normalization. Only activations from the previous layer are normalized. Hence, the SPADE generator can better preserve semantic information. It enjoys the benefit of normalization without losing the semantic input information. 
Multi-modal synthesis. By using a random vector as the input of the generator, our architecture provides a simple way for multi-modal synthesis [20, 60]. Namely, one can attach an encoder that processes a real image into a random vector, which will be then fed to the generator. The encoder and generator form a VAE [28], in which the encoder tries to capture the style of the image, while the generator combines the encoded style and the segmentation mask information via the SPADEs to reconstruct the original image. The encoder also serves as a style guidance network at test time to capture the style of target images, as used in Figure 1. For training, we add a KL-Divergence loss term [28]. 
# 4. Experiments 
Implementation details. We apply the Spectral Norm [38] to all the layers in both generator and discriminator. The learning rates for the generator and discriminator are 0.0001 and 0.0004, respectively [17]. We use the ADAM solver [27] with $\beta_{1}\,=\,0$ and $\beta_{2}\,=\,0.999$ . All the experiments are conducted on an NVIDIA DGX1 with 8 32GB V100 GPUs. We use synchronized BatchNorm, i.e., these statistics are collected from all the GPUs. 
Datasets. We conduct experiments on several datasets. 
• COCO-Stuff [4] is derived from the COCO dataset [32]. It has 118, 000 training images and 5, 000 validation images captured from diverse scenes. It has 182 semantic classes. Due to its vast diversity, existing image synthesis models perform poorly on this dataset. • ADE20K [58] consists of 20, 210 training and 2, 000 validation images. Similarly to the COCO, the dataset contains challenging scenes with 150 semantic classes. • ADE20K-outdoor is a subset of the ADE20K dataset that only contains outdoor scenes, used in Qi et al. [43]. Cityscapes dataset [9] contains street scene images in German cities. The training and validation set sizes are 3, 000 and 500, respectively. Recent work has achieved photorealistic semantic image synthesis results [43, 47] on the Cityscapes dataset. • Flickr Landscapes. We collect 41, 000 photos from Flickr and use 1, 000 samples for the validation set. To avoid expensive manual annotation, we use a well-trained DeepLabV2 [5] to compute input segmentation masks. 
We train the competing semantic image synthesis methods on the same training set and report their results on the same validation set for each dataset. 
Performance metrics. We adopt the evaluation protocol from previous work [6, 48]. Specifically, we run a semantic segmentation model on the synthesized images and compare how well the predicted segmentation mask matches the ground truth input. Intuitively, if the output images are realistic, a well-trained semantic segmentation model should be able to predict the ground truth label. For measuring the segmentation accuracy, we use both the mean Intersectionover-Union (mIoU) and the pixel accuracy (accu). We use the state-of-the-art segmentation networks for each dataset: DeepLabV2 [5, 40] for COCO-Stuff, UperNet101 [51] for ADE20K, and DRN-D-105 [53] for Cityscapes. In addition to the mIoU and the accu segmentation performance metrics, we use the Fre´chet Inception Distance (FID) [17] to measure the distance between the distribution of synthesized results and the distribution of real images. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/f816858faa279df1fa05faf5ec47a290dbf31346ab48b21545d7c20945557f1c.jpg) 
Figure 5: Visual comparison of semantic image synthesis results on the COCO-Stuff dataset. Our method successfully synthesizes realistic details from semantic labels. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/2e31652c3769427ddc9ad43246583701f3ab5640c1c34a6d2b0eb0e3a760a031.jpg) 
Figure 6: Visual comparison of semantic image synthesis results on the ADE20K outdoor and Cityscapes datasets. Our method produces realistic images while respecting the spatial semantic layout at the same time. 
<html><body><table><tr><td rowspan="2">Method</td><td colspan="3">COCO-Stuff</td><td colspan="3">ADE20K</td><td colspan="3">ADE20K-outdoor</td><td colspan="3">Cityscapes</td></tr><tr><td>mIoU</td><td>accu</td><td>FID</td><td>mIoU</td><td>accu</td><td>FID</td><td>mIoU</td><td>accu</td><td>FID</td><td>mIoU</td><td>accu</td><td>FID</td></tr><tr><td>CRN [6]</td><td>23.7</td><td>40.4</td><td>70.4</td><td>22.4</td><td>68.8</td><td>73.3</td><td>16.5</td><td>68.6</td><td>99.0</td><td>52.4</td><td>77.1</td><td>104.7</td></tr><tr><td>SIMS [43]</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>13.1</td><td>74.7</td><td>67.7</td><td>47.2</td><td>75.5</td><td>49.7</td></tr><tr><td>pix2pixHD [48]</td><td>14.6</td><td>45.8</td><td>111.5</td><td>20.3</td><td>69.2</td><td>81.8</td><td>17.4</td><td>71.6</td><td>97.8</td><td>58.3</td><td>81.4</td><td>95.0</td></tr><tr><td>Ours</td><td>37.4</td><td>67.9</td><td>22.6</td><td>38.5</td><td>79.9</td><td>33.9</td><td>30.8</td><td>82.9</td><td>63.3</td><td>62.3</td><td>81.9</td><td>71.8</td></tr></table></body></html>
Table 1: Our method outperforms the current leading methods in semantic segmentation (mIoU and accu) and FID [17] scores on all the benchmark datasets. For the mIoU and accu, higher is better. For the FID, lower is better. 
Baselines. We compare our method with 3 leading semantic image synthesis models: the pix2pixHD model [48], the cascaded refinement network (CRN) [6], and the semiparametric image synthesis method (SIMS) [43]. The pix2pixHD is the current state-of-the-art GAN-based conditional image synthesis framework. The CRN uses a deep network that repeatedly refines the output from low to high resolution, while the SIMS takes a semi-parametric approach that composites real segments from a training set and refines the boundaries. Both the CRN and SIMS are mainly trained using image reconstruction loss. For a fair comparison, we train the CRN and pix2pixHD models using the implementations provided by the authors. As image synthesis using the SIMS requires many queries to the training dataset, it is computationally prohibitive for a large dataset such as the COCO-stuff and the full ADE20K. Therefore, we use the results provided by the authors when available. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/8eacf3934117a5693cc7a9a3ca52a8324033a56e6a5274e4bc939d2717a41606.jpg) 
Figure 7: Semantic image synthesis results on the Flickr Landscapes dataset. The images were generated from semantic layout of photographs on the Flickr website. 
Quantitative comparisons. As shown in Table 1, our method outperforms the current state-of-the-art methods by a large margin in all the datasets. For the COCO-Stuff, our method achieves an mIoU score of 35.2, which is about 1.5 times better than the previous leading method. Our FID is also 2.2 times better than the previous leading method. We note that the SIMS model produces a lower FID score but has poor segmentation performances on the Cityscapes dataset. This is because the SIMS synthesizes an image by first stitching image patches from the training dataset. As using the real image patches, the resulting image distribution can better match the distribution of real images. However, because there is no guarantee that a perfect query (e.g., a person in a particular pose) exists in the dataset, it tends to copy objects that do not match the input segments. 
Qualitative results. In Figures 5 and 6, we provide qualitative comparisons of the competing methods. We find that our method produces results with much better visual quality and fewer visible artifacts, especially for diverse scenes in the COCO-Stuff and ADE20K dataset. When the training dataset size is small, the SIMS model also renders images with good visual quality. However, the depicted content often deviates from the input segmentation mask (e.g., the shape of the swimming pool in the second row of Figure 6). 
<html><body><table><tr><td>Dataset</td><td>Oursvs. CRN</td><td>Oursvs. pix2pixHD</td><td>Oursvs. SIMS</td></tr><tr><td>COCO-Stuff</td><td>79.76</td><td>86.64</td><td>N/A</td></tr><tr><td>ADE20K</td><td>76.66</td><td>83.74</td><td>N/A</td></tr><tr><td>ADE20K-outdoor</td><td>66.04</td><td>79.34</td><td>85.70</td></tr><tr><td>Cityscapes</td><td>63.60</td><td>53.64</td><td>51.52</td></tr></table></body></html>
Table 2: User preference study. The numbers indicate the percentage of users who favor the results of the proposed method over those of the competing method. 
In Figures 7 and 8, we show more example results from the Flickr Landscape and COCO-Stuff datasets. The proposed method can generate diverse scenes with high image fidelity. More results are included in the appendix. 
Human evaluation. We use the Amazon Mechanical Turk (AMT) to compare the perceived visual fidelity of our method against existing approaches. Specifically, we give the AMT workers an input segmentation mask and two synthesis outputs from different methods and ask them to choose the output image that looks more like a corresponding image of the segmentation mask. The workers are given unlimited time to make the selection. For each comparison, we randomly generate 500 questions for each dataset, and each question is answered by 5 different workers. For quality control, only workers with a lifetime task approval rate greater than $98\%$ can participate in our study. 
Table 2 shows the evaluation results. We find that users 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/0628224caa6caa2eed81c36d628fa9c5128fca16114dd398070b6be85d322610.jpg) 
Figure 8: Semantic image synthesis results on COCO-Stuff. Our method successfully generates realistic images in diverse scenes ranging from animals to sports activities. 
<html><body><table><tr><td>Method</td><td>#param</td><td>COCO.</td><td>ADE</td><td>City.</td></tr><tr><td>decoder w/SPADE (Ours) compact decoder w/ SPADE</td><td>96M 61M</td><td>35.2 35.2</td><td>38.5 38.0</td><td>62.3 62.5</td></tr><tr><td>decoderw/Concat pix2pixHD++w/SPADE</td><td>79M 237M</td><td>31.9</td><td>33.6</td><td>61.1</td></tr><tr><td>pix2pixHD++ w/ Concat</td><td></td><td>34.4</td><td>39.0</td><td>62.2</td></tr><tr><td></td><td>195M</td><td>32.9</td><td>38.9</td><td>57.1</td></tr><tr><td>pix2pixHD++</td><td>183M</td><td>32.7</td><td>38.3</td><td>58.8</td></tr><tr><td>compact pix2pixHD++</td><td>103M</td><td>31.6</td><td>37.3</td><td>57.6</td></tr><tr><td>pix2pixHD [48]</td><td>183M</td><td>14.6</td><td>20.3</td><td>58.3</td></tr></table></body></html> 
<html><body><table><tr><td>Method</td><td>COCO</td><td>ADE20K</td><td>Cityscapes</td></tr><tr><td>segmap input random input</td><td>35.2 35.3</td><td>38.5 38.3</td><td>62.3 61.6</td></tr><tr><td>kernelsize5x5 kernelsize3x3 kernelsize 1x1</td><td>35.0 35.2 32.7</td><td>39.3 38.5 35.9</td><td>61.8 62.3 59.9</td></tr><tr><td>#params 141M #params 96M #params 61M</td><td>35.3 35.2 35.2</td><td>38.3 38.5 38.0</td><td>62.5 62.3 62.5</td></tr><tr><td>SyncBatchNorm BatchNorm InstanceNorm</td><td>35.0 33.7 33.9</td><td>39.3 37.9 37.4</td><td>61.8 61.8 58.7</td></tr></table></body></html> 
Table 4: The SPADE generator works with different configurations. We change the input of the generator, the convolutional kernel size acting on the segmentation map, the capacity of the network, and the parameter-free normalization method. The settings used in the paper are boldfaced. 
strongly favor our results on all the datasets, especially on the challenging COCO-Stuff and ADE20K datasets. For the Cityscapes, even when all the competing methods achieve high image fidelity, users still prefer our results. 
Effectiveness of the SPADE. For quantifying importance of the SPADE, we introduce a strong baseline called $\mathrm{pix}2\mathrm{pix}\mathrm{HD}++$ , which combines all the techniques we find useful for enhancing the performance of pix2pixHD except the SPADE. We also train models that receive the segmentation mask input at all the intermediate layers via feature concatenation in the channel direction, which is termed as pix2pixHD $^{++}$ w/ Concat. Finally, the model that combines the strong baseline with the SPADE is denoted as pix2pixHD $^{++}$ w/ SPADE. 
As shown in Table 3, the architectures with the proposed SPADE consistently outperforms its counterparts, in both the decoder-style architecture described in Figure 4 and more traditional encoder-decoder architecture used in the pix2pixHD. We also find that concatenating segmentation masks at all intermediate layers, a reasonable alternative to the SPADE, does not achieve the same performance as SPADE. Furthermore, the decoder-style SPADE generator works better than the strong baselines even with a smaller number of parameters. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/d59ad5c4133a57dbf6d6e8ea8f3772d95bd3a52642cbbbaa8186c0a14bd95d7f.jpg) 
Figure 9: Our model attains multimodal synthesis capability when trained with the image encoder. During deployment, by using different random noise, our model synthesizes outputs with diverse appearances but all having the same semantic layouts depicted in the input mask. For reference, the ground truth image is shown inside the input segmentation mask. 
Variations of SPADE generator. Table 4 reports the performance of several variations of our generator. First, we compare two types of input to the generator where one is the random noise while the other is the downsampled segmentation map. We find that both of the variants render similar performance and conclude that the modulation by SPADE alone provides sufficient signal about the input mask. Second, we vary the type of parameter-free normalization layers before applying the modulation parameters. We observe that the SPADE works reliably across different normalization methods. Next, we vary the convolutional kernel size acting on the label map, and find that kernel size of 1x1 hurts performance, likely because it prohibits utilizing the context of the label. Lastly, we modify the capacity of the generator by changing the number of convolutional filters. We present more variations and ablations in the appendix. 
Multi-modal synthesis. In Figure 9, we show the multimodal image synthesis results on the Flickr Landscape dataset. For the same input segmentation mask, we sample different noise inputs to achieve different outputs. More results are included in the appendix. 
Semantic manipulation and guided image synthesis. In Figure 1, we show an application where a user draws different segmentation masks, and our model renders the corresponding landscape images. Moreover, our model allows users to choose an external style image to control the global appearances of the output image. We achieve it by replacing the input noise with the embedding vector of the style image computed by the image encoder. 
# 5. Conclusion 
We have proposed the spatially-adaptive normalization, which utilizes the input semantic layout while performing the affine transformation in the normalization layers. The proposed normalization leads to the first semantic image synthesis model that can produce photorealistic outputs for diverse scenes including indoor, outdoor, landscape, and street scenes. We further demonstrate its application for multi-modal synthesis and guided image synthesis. 
Acknowledgments. We thank Alexei A. Efros, Bryan Catanzaro, Andrew Tao, and Jan Kautz for insightful advice. We thank Chris Hebert, Gavriil Klimov, and Brad Nemire for their help in constructing the demo apps. Taesung Park contributed to the work during his internship at NVIDIA. His Ph.D. is supported by a Samsung Scholarship. 
# References 
[1] M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein generative adversarial networks. In International Conference on Machine Learning (ICML), 2017. 3 
[2] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 2 
[3] A. Brock, J. Donahue, and K. Simonyan. Large scale gan training for high fidelity natural image synthesis. In International Conference on Learning Representations (ICLR), 2019. 1, 2 
[4] H. Caesar, J. Uijlings, and V. Ferrari. Coco-stuff: Thing and stuff classes in context. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2, 4 
[5] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 40(4):834–848, 2018. 4, 5 
[6] Q. Chen and V. Koltun. Photographic image synthesis with cascaded refinement networks. In IEEE International Conference on Computer Vision (ICCV), 2017. 1, 4, 5, 13, 14, 15, 16, 17, 18 
[7] T. Chen, M.-M. Cheng, P. Tan, A. Shamir, and S.-M. Hu. Sketch2photo: internet image montage. ACM Transactions on Graphics (TOG), 28(5):124, 2009. 1, 2 
[8] T. Chen, M. Lucic, N. Houlsby, and S. Gelly. On self modulation for generative adversarial networks. In International Conference on Learning Representations, 2019. 2 
[9] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The cityscapes dataset for semantic urban scene understanding. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2, 4 
[10] H. De Vries, F. Strub, J. Mary, H. Larochelle, O. Pietquin, and A. C. Courville. Modulating early visual processing by language. In Advances in Neural Information Processing Systems, 2017. 2 
[11] V. Dumoulin, J. Shlens, and M. Kudlur. A learned representation for artistic style. In International Conference on Learning Representations (ICLR), 2016. 2, 3 
[12] X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 249–256, 2010. 12, 13 
[13] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, 2014. 2 
[14] J. Hays and A. A. Efros. Scene completion using millions of photographs. In ACM SIGGRAPH, 2007. 1 
[15] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 3 
[16] A. Hertzmann, C. E. Jacobs, N. Oliver, B. Curless, and D. H. Salesin. Image analogies. 2001. 1, 2 
[17] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, 2017. 4, 5, 13 
[18] S. Hong, D. Yang, J. Choi, and H. Lee. Inferring semantic layout for hierarchical text-to-image synthesis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2 
[19] X. Huang and S. Belongie. Arbitrary style transfer in realtime with adaptive instance normalization. In IEEE International Conference on Computer Vision (ICCV), 2017. 2, 3 
[20] X. Huang, M.-Y. Liu, S. Belongie, and J. Kautz. Multimodal unsupervised image-to-image translation. European Conference on Computer Vision (ECCV), 2018. 2, 3, 4 
[21] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning (ICML), 2015. 2, 3 
[22] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-toimage translation with conditional adversarial networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 1, 2, 3, 11, 12 
[23] M. Johnson, G. J. Brostow, J. Shotton, O. Arandjelovic, V. Kwatra, and R. Cipolla. Semantic photo synthesis. In Computer Graphics Forum, volume 25, pages 407–413, 2006. 1, 2 
[24] L. Karacan, Z. Akata, A. Erdem, and E. Erdem. Learning to generate images of outdoor scenes from attributes and semantic layouts. arXiv preprint arXiv:1612.00215, 2016. 2 
[25] L. Karacan, Z. Akata, A. Erdem, and E. Erdem. Manipulating attributes of natural scenes via hallucination. arXiv preprint arXiv:1808.07413, 2018. 2 
[26] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2 
[27] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015. 4 
[28] D. P. Kingma and M. Welling. Auto-encoding variational bayes. In International Conference on Learning Representations (ICLR), 2014. 2, 4, 11, 12 
[29] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, 2012. 2 
[30] J.-F. Lalonde, D. Hoiem, A. A. Efros, C. Rother, J. Winn, and A. Criminisi. Photo clip art. In ACM transactions on graphics (TOG), volume 26, page 3. ACM, 2007. 1 
[31] J. H. Lim and J. C. Ye. Geometric gan. arXiv preprint arXiv:1705.02894, 2017. 3, 11 
[32] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dolla´r, and C. L. Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision (ECCV), 2014. 2, 4 
[33] M.-Y. Liu, T. Breuel, and J. Kautz. Unsupervised image-toimage translation networks. In Advances in Neural Information Processing Systems, 2017. 2 
[34] X. Mao, Q. Li, H. Xie, Y. R. Lau, Z. Wang, and S. P. Smolley. Least squares generative adversarial networks. In IEEE International Conference on Computer Vision (ICCV), 2017. 3, 11 
[35] T. B. Mathias Eitz, Kristian Hildebrand and M. Alexa. Photosketch: A sketch based image query and compositing system. In ACM SIGGRAPH 2009 Talk Program, 2009. 1 
[36] L. Mescheder, A. Geiger, and S. Nowozin. Which training methods for gans do actually converge? In International Conference on Machine Learning (ICML), 2018. 2, 3, 11 
[37] M. Mirza and S. Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014. 2 
[38] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations (ICLR), 2018. 3, 4, 11 
[39] T. Miyato and M. Koyama. cGANs with projection discriminator. In International Conference on Learning Representations (ICLR), 2018. 2, 3, 11 
[40] K. Nakashima. Deeplab-pytorch. https://github. com/kazuto1011/deeplab-pytorch, 2018. 5 
[41] A. Odena, C. Olah, and J. Shlens. Conditional image synthesis with auxiliary classifier GANs. In International Conference on Machine Learning (ICML), 2017. 2 
[42] E. Perez, H. De Vries, F. Strub, V. Dumoulin, and A. Courville. Learning visual reasoning without strong priors. In International Conference on Machine Learning (ICML), 2017. 2 
[43] X. Qi, Q. Chen, J. Jia, and V. Koltun. Semi-parametric image synthesis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 4, 5, 13, 17, 18 
[44] S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee. Generative adversarial text to image synthesis. In International Conference on Machine Learning (ICML), 2016. 2 
[45] T. Salimans and D. P. Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, 2016. 2 
[46] D. Ulyanov, A. Vedaldi, and V. Lempitsky. Instance normalization: The missing ingredient for fast stylization. arxiv 2016. arXiv preprint arXiv:1607.08022, 2016. 2, 3 
[47] T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, G. Liu, A. Tao, J. Kautz, and B. Catanzaro. Video-to-video synthesis. In Advances in Neural Information Processing Systems, 2018. 1, 4 
[48] T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, A. Tao, J. Kautz, and B. Catanzaro. High-resolution image synthesis and semantic manipulation with conditional gans. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 1, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18 
[49] X. Wang, K. Yu, C. Dong, and C. Change Loy. Recovering realistic texture in image super-resolution by deep spatial feature transform. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 606–615, 2018. 2 
[50] Y. Wu and K. He. Group normalization. In European Conference on Computer Vision (ECCV), 2018. 2 
[51] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun. Unified perceptual parsing for scene understanding. In European Conference on Computer Vision (ECCV), 2018. 5 
[52] T. Xu, P. Zhang, Q. Huang, H. Zhang, Z. Gan, X. Huang, and X. He. Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2 
[53] F. Yu, V. Koltun, and T. Funkhouser. Dilated residual networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 5 
[54] H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena. Selfattention generative adversarial networks. In International Conference on Machine Learning (ICML), 2019. 1, 2, 3, 11 
[55] H. Zhang, T. Xu, H. Li, S. Zhang, X. Huang, X. Wang, and D. Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In IEEE International Conference on Computer Vision (ICCV), 2017. 1, 2 
[56] H. Zhang, T. Xu, H. Li, S. Zhang, X. Wang, X. Huang, and D. Metaxas. Stackgan++: Realistic image synthesis with stacked generative adversarial networks. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2018. 1 
[57] B. Zhao, L. Meng, W. Yin, and L. Sigal. Image generation from layout. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2 
[58] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba. Scene parsing through ade20k dataset. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 2, 4 
[59] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired imageto-image translation using cycle-consistent adversarial networks. In IEEE International Conference on Computer Vision (ICCV), 2017. 2 
[60] J.-Y. Zhu, R. Zhang, D. Pathak, T. Darrell, A. A. Efros, O. Wang, and E. Shechtman. Toward multimodal image-toimage translation. In Advances in Neural Information Processing Systems, 2017. 2, 3, 4 
# A. Additional Implementation Details 
Generator. The architecture of the generator consists of a series of the proposed SPADE ResBlks with nearest neighbor upsampling. We train our network using 8 GPUs simultaneously and use the synchronized version of the BatchNorm. We apply the Spectral Norm [38] to all the convolutional layers in the generator. The architectures of the proposed SPADE and SPADE ResBlk are given in Figure 10 and Figure 11, respectively. The architecture of the generator is shown in Figure 12. 
Discriminator. The architecture of the discriminator follows the one used in the pix2pixHD method [48], which uses a multi-scale design with the InstanceNorm (IN). The only difference is that we apply the Spectral Norm to all the 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/029a1e258bd921b008fb12a2d63784e0f8f860cfabd9a66210802e6d2bb66fcb.jpg) 
Figure 12: SPADE Generator. Different from prior image generators [22,48], the semantic segmentation mask is passed to the generator through the proposed SPADE ResBlks in Figure 11. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/359b6b1b23d8a2d43262bd799884855c6bdc27d67fba46a36b1befeeae618f7e.jpg) 
Figure 10: SPADE Design. The term $3\mathrm{x}3$ -Conv-k denotes a 3-by-3 convolutional layer with $k$ convolutional filters. The segmentation map is resized to match the resolution of the corresponding feature map using nearest-neighbor downsampling. 
Image Encoder. The image encoder consists of 6 stride-2 convolutional layers followed by two linear layers to produce the mean and variance of the output distribution as shown in Figure 14. 
Learning objective. We use the learning objective function in the pix2pixHD work [48] except that we replace its LSGAN loss [34] term with the Hinge loss term [31, 38, 54]. We use the same weighting among the loss terms in the objective function as that in the pix2pixHD work. 
When training the proposed framework with the image encoder for multi-modal synthesis and style-guided image synthesis, we include a KL Divergence loss: 
$$
\mathcal{L}_{\mathrm{KLD}}=\mathcal{D}_{\mathrm{KL}}(q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}))
$$ 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/b4e37d2c9f938e07d26ae1cdf833b3bf8340272789d0579bc18fcf34ca50e6a9.jpg) 
Figure 11: SPADE ResBlk. The residual block design largely follows that in Mescheder et al. [36] and Miyato et al. [39]. We note that for the case that the number of channels before and after the residual block is different, the skip connection is also learned (dashed box in the figure). convolutional layers of the discriminator. The details of the discriminator architecture is shown in Figure 13. 
where the prior distribution $p(\mathbf{z})$ is a standard Gaussian distribution and the variational distribution $q$ is fully determined by a mean vector and a variance vector [28]. We use the reparamterization trick [28] for back-propagating the gradient from the generator to the image encoder. The weight for the KL Divergence loss is 0.05. 
In Figure 15, we overview the training data flow. The image encoder encodes a real image to a mean vector and a variance vector. They are used to compute the noise input to the generator via the reparameterization trick [28]. The generator also takes the segmentation mask of the input image as input with the proposed SPADE ResBlks. The and the output image from the generator as input and aims to classify that as fake. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/914410ffa73ba84a6e2f62a36585822965d12b1b47e64dcfed9af53f09c82f55.jpg) 
Figure 13: Our discriminator design largely follows that in the pix2pixHD [48]. It takes the concatenation the segmentation map and the image as input. It is based on the PatchGAN [22]. Hence, the last layer of the discriminator is a convolutional layer. 
Training details. We perform 200 epochs of training on the Cityscapes and ADE20K datasets, 100 epochs of training on the COCO-Stuff dataset, and 50 epochs of training on the Flickr Landscapes dataset. The image sizes are $256\times256$ , except the Cityscapes at $512\times256$ . We linearly decay the learning rate to 0 from epoch 100 to 200 for the Cityscapes and ADE20K datasets. The batch size is 32. We initialize the network weights using thes Glorot initialization [12]. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/7f132702bc54314cd9c45fe37dcece0e8d337f25f9f8df00de1fbf61d816f6fd.jpg) 
Figure 14: The image encoder consists a series of convolutional layers with stride 2 followed by two linear layers that output a mean vector $\mu$ and a variance vector $\sigma$ . 
discriminator takes concatenation of the segmentation mask 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/25c9eef0c910e118ae6c5c02ad753ffd4355e665a7791a437e9e162501525e62.jpg) 
Figure 15: The image encoder encodes a real image to a latent representation for generating a mean vector and a variance vector. They are used to compute the noise input to the generator via the reparameterization trick [28]. The generator also takes the segmentation mask of the input image as input via the proposed SPADE ResBlks. The discriminator takes concatenation of the segmentation mask and the output image from the generator as input and aims to classify that as fake. 
B. Additional Ablation Study 
<html><body><table><tr><td>Method</td><td>COCO.</td><td>ADE.</td><td>City.</td></tr><tr><td>Ours Oursw/oPerceptualloss Oursw/oGANfeaturematchingloss Oursw/adeeperdiscriminator</td><td>35.2 24.7 33.2 34.9</td><td>38.5 30.1 38.0 38.3</td><td>62.3 57.4 62.2 60.9</td></tr><tr><td>pix2pixHD++ w/ SPADE pix2pixHD++ pix2pixHD++ w/o Sync BatchNorm pix2pixHD++ w/o Sync BatchNorm,</td><td>34.4 32.7 27.4 26.0</td><td>39.0 38.3 31.8 31.9</td><td>62.2 58.8 51.1 52.3</td></tr><tr><td>andw/oSpectralNorm pix2pixHD [48]</td><td>14.6</td><td>20.3</td><td>58.3</td></tr></table></body></html> 
Table 5: Additional ablation study results using the mIoU metric: the table shows that both the perceptual loss and GAN feature matching loss terms are important. Making the discriminator deeper does not lead to a performance boost. The table also shows that all the components (Synchronized BatchNorm, Spectral Norm, TTUR, the Hinge loss objective, and the SPADE) used in the proposed method helps our strong baseline, pix2pixHD $^{++}$ . 
Table 5 provides additional ablation study results analyzing the contribution of individual components in the proposed method. We first find that both of the perceptual loss and GAN feature matching loss inherited from the learning objective function of the pix2pixHD [48] are important. Removing any of them leads to a performance drop. We also find that increasing the depth of the discriminator by inserting one more convolutional layer to the top of the pix2pixHD discriminator does not improve the results. 
In Table 5, we also analyze the effectiveness of each component used in our strong baseline, the pix2pixHD++ method, derived from the pix2pixHD method. We found that the Spectral Norm, synchronized BatchNorm, TTUR [17], and the hinge loss objective all contribute to the performance boost. Adding the SPADE to the strong baseline further improves the performance. Note that the $\mathrm{pix}2\mathrm{pixHD}++\mathrm{w}/\mathrm{c}$ Sync BatchNorm and w/o Spectral Norm still differs from the pix2pixHD in that it uses the hinge loss objective, TTUR, a large batch size, and the Glorot initialization [12]. 
# C. Additional Results 
In Figure 16, 17, and 18, we show additional synthesis results from the proposed method on the COCO-Stuff and ADE20K datasets with comparisons to those from the CRN [6] and pix2pixHD [48] methods. 
In Figure 19 and 20, we show additional synthesis results from the proposed method on the ADE20K-outdoor and Cityscapes datasets with comparison to those from the CRN [6], SIMS [43], and pix2pixHD [48] methods. 
In Figure 21, we show additional multi-modal synthesis results from the proposed method. As sampling different z from a standard multivariate Gaussian distribution, we synthesize images of diverse appearances. 
In the accompanying video, we demonstrate our semantic image synthesis interface. We show how a user can create photorealistic landscape images by painting semantic labels on a canvas. We also show how a user can synthesize images of diverse appearances for the same semantic segmentation mask as well as transfer the appearance of a provided style image to the synthesized one. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/4f77c34d7bdb56515540ba62ce55673b877e2d185ef79a62f56bbac22c850515.jpg) 
Figure 16: Additional results with comparison to those from the CRN [6] and pix2pixHD [48] methods on the COCO-Stuff dataset. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/f60d91f1c70ce94e07cf1cb6a1518aebeef1d8f831a684dced438c8a060ecd80.jpg) 
Figure 17: Additional results with comparison to those from the CRN [6] and pix2pixHD [48] methods on the COCO-Stuff dataset. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/f3604318779a05e15446f21f77e5437bb7040884af9c81ee68af5c6ce0ded89d.jpg) 
Figure 18: Additional results with comparison to those from the CRN [6] and pix2pixHD [48] methods on the ADE20K dataset. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/091c66259aadfa499b52c77838659b25d7040dfcb300753a762216e821186831.jpg) 
Figure 19: Additional results with comparison to those from the CRN [6], SIMS [43], and pix2pixHD [48] methods on the ADE20K-outdoor dataset. 17 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/21599fa8abfe38d4e5954fc55a5b28e95a4fd3b89b2bae348a86ea97c8684684.jpg) 
Figure 20: Additional results with comparison to those from the CRN [6], SIMS [43], and pix2pixHD [48] methods on the Cityscapes dataset. 
![](https://cdn-mineru.openxlab.org.cn/extract/1296bae5-f687-4353-bab4-46bfd5d1a549/3cf7fd1c6a68dfd6fa7279e2a951657973b5138b926eab0c15ae3d99a8d4f4ac.jpg) 
Figure 21: Additional multi-modal synthesis results on the Flickr Landscapes Dataset. By sampling latent vectors from a standard Gaussian distribution, we synthesize images of diverse appearances. 