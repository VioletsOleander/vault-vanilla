# Abstract
我们提出一个新的框架，通过对抗式过程估计生成式模型，
在该框架中，我们同时训练两个模型：一个生成式模型 $G$，用于捕捉数据分布，一个区分式模型 $D$，用于估计一个样本是来自于训练数据而不是生成式模型 $G$ 的概率
$G$ 的训练过程即最大化 $D$ 犯错的概率

该框架对应一个有两个玩家的最大最小游戏，在任意函数 $G$ 和 $D$ 的空间中，存在一个唯一解，使得 $G$ 捕获了训练数据的分布，$D$ 预测的概率都是 $\frac 1 2$

本文中，$G$ 和 $D$ 都定义为多层感知机，整个系统用反向传播训练，在训练或样本生成的过程中，不需要任何马尔可夫链或展开近似推理网络 (unrolled approximate inference network)

实验通过定性和定量的对生成的样本的评估，展示了该框架的潜力
# 1 Introduction
目前为止，深度学习最成功的模型主要是区分式模型，通常是将高维的，丰富的感官输入 (high-dimensional, rich sensory input) 映射到一个类别标签
它们的成功主要基于反向传播 (backpropagation) 和丢弃 (dropout) 算法，使用了梯度尤其稳定 (well-behaved gradient) 的分段 (piecewise) 线性单元

深度的生成式模型，因为近似许多不可解的 (intractable) 概率计算问题 (这些问题往往来自于极大似然估计以及相关的策略) 的难度较大，以及因为难以在生成内容的过程中利用分段线性单元的优势，影响相应较小

我们提出一个新的生成式模型估计过程 (generative model estimation procedure) 以回避这些困难

在我们提出的对抗式网络框架中，生成式模型要和一个区分式模型对抗，区分式模型学习决定一个样本是来自模型分布或数据分布，
生成式模型可以认为是造假者，尝试生产假钞，逃过检测，区分式模型则类比警察，尝试识别出假钞

在该对抗游戏中，双方逐渐改善各自的方法，直到假物和真物难以区分

本文中，生成式模型和区分式模型都是多层感知机，我们向生成式模型传递随机噪声以生成样本，称该网络为对抗式网络 (adversarial nets)
我们仅用高度成功的反向传播和丢弃算法训练这两个模型，且仅用前向传播从生成式模型采样，不需要近似推理和马尔科夫链
# 2 Related work
带有隐变量的有向图模型 (directed graphical model with latent variables) 的一个替代就是带有隐变量的无向图模型，例如限制玻尔兹曼机 (RBM)、深度玻尔兹曼机 (DBM) 以及它们的变体
在这类模型中，交互 (interactions) 用未规范化的势函数的积表示 (the product of unnormalized potential functions)，最后由对随机变量的所有状态的全局和/积分规范化 (normalized by a global summation/integration over all states of the random variables)，而这个量 (quantity)(划分函数 the partition function) 和它的梯度，对于几乎所有的实例 (除了最平凡/trivial 的实例) 外都是不可解的，即使它们可以用马尔可夫链蒙特卡罗方法 (MCMC) 估计

深度信念网络 (DBN) 是包含了单个无向层和多个有向层的混合模型，虽然存在一个快速近似的逐层训练准则 (criterion)，但是 DBN 既会导致与无向模型相关的计算困难，也会导致有向模型相关的计算困难

不对对数似然进行近似或定界 (approximate or bound) 的替代准则也有被提出，例如分数匹配 (score matching) 或噪声对比估计 (noise-contrastive estimation/NCE)，这两个准则都需要将学习到的概率密度解析式地指定到一个规范化常数 (analytically specified up to a normalization constant)

注意在许多包含了多个带隐藏变量的层的生成式模型 (例如 DBN，DBM) 中，要推导出一个可解的未规范化的概率密度是不可能的
一些模型，例如去噪 (denoising) 自动编码机以及收缩 (contractive) 自动编码机学习的规则和将分数匹配应用到 RBMs 非常相似
在 NCE 中，也就如在本项工作中，一个区分式的训练准则 (discriminative training criterion) 被应用于拟合一个生成式模型，但并不是拟合一个分离的区分式模型，而是使用生成式模型本身区分从一个固定噪声分布中生成的样本，因为 NCE 用的是固定的噪声分布，学习会在模型学习到了观察到的变量的一个小子集上的一个近似正确的分布之后变得非常缓慢

一些技术并没有显式地定义一个概率分布，而是训练一个可以从所要求的分布中采样的生成式模型，这类方法的优势在于模型可以用反向传播训练
该领域中较为重要的工作包括生成式随机网络 (GSN Generative Stochastic Networks) 框架，它对广义的去噪自动编码器进行了延伸，而实际上这两项工作都可以视作定义了一个参数化的马尔可夫链，即模型随着在生成式马尔科夫链上前进而学习到参数

相较于 GSNs，对抗式网络框架不需要马尔科夫链进行采样
因为对抗式网络不需要在生成时不需要反馈循环 (feedback loops)，因此可以更好地利用分段线性单元 (分段线性单元可以提高反向传播的效果，但在反馈循环内使用时，会遭遇无界激活 unbounded activation 的问题)

将反向传播嵌入到训练生成式模型的相关工作近来还包括变分贝叶斯自动编码机以及随机反向传播
# 3 Adversarial nets
对抗式建模框架在模型都是多层感知机时最能直观应用
为了学习生成器在数据 $\symbfit x$ 上的分布 $p_g$ (the generator's distribution over data)，我们在输入噪声变量 (input noise variables) 上定义一个先验分布 $p_{\symbfit z}(\symbfit z)$，然后将一个到数据空间的映射表示为 $G(\symbfit z; \theta_g)$，其中 $G$ 是一个可微函数，形式是参数为 $\theta_g$ 的多层感知机
我们还定义第二个多层感知机 $D(\symbfit x;\theta_d)$，它输出一个标量，$D(\symbfit x)$ 表示 $\symbfit x$ 来自于数据而不是 $p_g$ 的概率

我们训练 $D$ 以最大化将正确的标签分配给训练样本以及来自 $G$ 的样本的概率，我们同时训练 $G$ 以最小化 $\log(1-D(G(\symbfit z)))$

换句话说，$D$ 和 $G$ 在进行如下值函数是 $V(G,D)$ 的二玩家最大最小游戏：

$$\min_G\max_D V(G,D) = \mathbb E_{\symbfit x\sim p_{data}(\symbfit x)}[\log D(\symbfit x)]+\mathbb E_{\symbfit z\sim p_{\symbfit z}(\symbfit z)}[\log(1-D(G(\symbfit z))]\tag{1}$$

下一部分，我们会对对抗式网络进行理论分析，以说明该训练准则在 $G$ 和 $D$ 有足够的模型容量 (capacity) 的情况下，即无参数限制 (non-parametric limit) 的情况下，$G$ 可以复原数据生成分布 (data generating distribution)

在实际中，我们必须用一个迭代式的数值方法 (an iterative, numerical approach) 执行该游戏，在内部循环完全优化 $D$ (optimizing $D$ to completion) 在计算上是不可行的，且在有限数据集上会导致过拟合，
因此，我们在 $k$ 步优化 $D$ 和 1 步优化 $G$ 之间交替，只要 $G$ 改变得足够缓慢，这就会将 $D$ 维持在接近它的最优解，
该策略类似 SML/PCD[31, 29]在训练中的一个学习步到下一个学习步的过程中，maintains 来自于马尔可夫链的样本，以避免在学习的内循环中 burning in a Markov chain

具体过程见 Algorithm 1
![[Generative Adversarial Nets-Algorithm 1.png]]

在实际中，等式 1 并不能为 $G$ 提供足够的梯度以使 $G$ 学习得很好，在训练的早期阶段，$G$ 效果较差时，$D$ 可以以高的置信度拒绝 $G$ 生成的样本，因为它们与训练数据有显著的不同，在这种情况下，$\log(1-D(G(\symbfit z)))$ 饱和
(其实就是在 $t$ 接近 $0$ 的时候，$\nabla_t\log(1-t)$ 在数量级上远小于 $\nabla_t\log t$)

因此我们不选择训练 $G$ 以最小化 $\log(1-D(G(\symbfit z)))$，而是训练 $G$ 以最大化 $\log D(G(\symbfit z))$，该函数会在学习的早期提供更多的梯度，且在 $G$ 和 $D$ 的动态变化过程中会和原目标函数落在同一定点
# 4 Theoretical Results
生成器 $G$ 隐式地定义了概率分布 $p_g$，作为当 $\symbfit z \sim p_{\symbfit z}$ 时从 $G(\symbfit z)$ 中得到的样本的分布，因此我们希望 Algorithm 1 在给定足够训练时间和模型容量的情况下，能收敛到对 $p_{data}$ 的一个良好的估计
本节的结果是在非参数设定的情况下推导的，即模型的容量是无限大的，模型在概率密度函数的空间中学习收敛

## 4.1 Global Optimality of $p_g = p_{data}$
我们首先考虑对于任意给定生成器 $G$ 最优的区分器 $D$

**Proposition 1** 对于固定的 $G$，最优的区分器 $D$ 是：

$$D^*_{G}(\symbfit x)=\frac {p_{data}(\symbfit x)}{p_{data}(\symbfit x)+p_g(\symbfit x)}\tag{2}$$

证明：
给定 $G$，对于 $D$ 的训练准则是最大化 $V(G,D)$

$$\begin{align}
V(G,D) &= \int_{\symbfit x}p_{data}(\symbfit x)\log(D(\symbfit x))dx + \int_{\symbfit z}p_{\symbfit z}(\symbfit z)\log(1-D(g(\symbfit z))dz\\
&=\int_{\symbfit x}p_{data}(\symbfit x)\log(D(\symbfit x)) + p_g(\symbfit x)\log(1-D(\symbfit x))dx
\end{align}\tag{3}$$

对于任意 $(a,b)\in \mathbb R^2 \backslash \{0,0\}$，函数 $y\rightarrow a\log(y)+b\log(1-y)$ 在定义域 $[0,1]$ 上在点 $\frac a {a+b}$ 取得最大值
而区分器并不需要定义在 $Supp(p_{data})\cup Supp(p_g)$ 之外，因此证毕

注意 $D$ 的训练目标可以解释为最大化对数似然以估计条件概率分布 $P(Y=y|\symbfit x)$，其中 $Y$ 表明 $\symbfit x$ 是否来自 $p_{data}$ ($y=1$) 或来自 $p_g$ ($y=0$)

然后我们考虑 $G$，$G$ 的训练准则是最小化 $\max_{D}V(G,D)$，即

$$\begin{align}
C(G) &= \max_{D}V(G,D)\\
&=\mathbb E_{\symbfit x\sim p_{data}}[\log D_G^*(\symbfit x)] + \mathbb E_{\symbfit z\sim p_{\symbfit z}}[\log(1-D_G^*(G(\symbfit z)))]\\
&=\mathbb E_{\symbfit x\sim p_{data}}[\log D_G^*(\symbfit x)] + \mathbb E_{\symbfit x\sim p_{g}}[\log(1-D_G^*(\symbfit x))]\\
&=\mathbb E_{\symbfit x\sim p_{data}}\left[\log \frac {p_{data}(\symbfit x)}{p_{data}(\symbfit x) + p_g(\symbfit x)}\right] + \mathbb E_{\symbfit x\sim p_{g}}\left[\log \frac {p_{g}(\symbfit x)}{p_{data}(\symbfit x) + p_g(\symbfit x)}\right] 
\end{align}\tag{4}$$

**Theorem 1** 虚拟训练准则 $C(G)$ 的全局最小值当且仅当 $p_g = p_{data}$ 时可以达到，此时，$C(G)$ 取得最小值 $-\log 4$

证明：
当 $p_g = p_{data}$ 时，$D_G^*(\symbfit x) = \frac 1 2$，将其代入公式 4，容易发现 $C(G) = \log \frac 1 2 + \log \frac 1 2 = -\log 4$

要证明这是 $C(G)$ 最优的可能值，我们首先观察发现：

$$\mathbb E_{\symbfit x\sim p_{data}}[-\log 2]+\mathbb E_{\symbfit x\sim p_g}[-\log 2]=-\log 4$$

我们用 $C(G) = V(D_G^*,G)$ 减去该表达式，得到：

$$C(G) + \log 4 = KL\left(p_{data} ||\frac {p_{data}+p_{g}}{2}\right) +KL\left(p_{g} ||\frac {p_{data}+p_{g}}{2}\right)$$

即

$$C(G) = -\log(4) +  KL\left(p_{data} ||\frac {p_{data}+p_{g}}{2}\right) +KL\left(p_{g} ||\frac {p_{data}+p_{g}}{2}\right)\tag{5}$$

而上式中两个 KL 散度的和可以写为模型的分布和数据生成过程之间的 Jensen-Shannon 散度：

$$C(G) = -\log(4) + 2\cdot JSD(p_{data}||p_g)\tag{6}$$

而两个分布之间的 Jensen-Shannon 散度总是非负的，且仅在两个分布相等时为零，因此 $C^* = -log(4)$ 是 $C(G)$ 的全局最小值，且唯一解是 $p_g = p_{data}$，即生成式模型完美复制了数据生成过程

## 4.2 Convergence of Algorithm 1
**Proposition 2** 若 $G$ 和 $D$ 容量足够，且在算法 1 的每一步，区分器 $D$ 都允许在给定 $G$ 的情况下达到最优解，且 $p_g$ 是根据提升以下指标来更新：
$$\mathbb E_{\symbfit x\sim p_{data}}[\log D_G^*(\symbfit x)]+\mathbb E_{\symbfit x\sim p_g}[\log(1-D_G^*(\symbfit x))]$$
则 $p_g$ 会收敛到 $p_{data}$

证明：
令 $V(G,D) = U(p_g,D)$，作为一个关于 $p_g$ 的函数，注意 $U(p_g,D)$ 是在 $p_g$ 上是凸的
一个凸函数的上确界 (supremum) 的子导数 (subderivatives) 包括了能使得凸函数达到最大值的点的导数，换句话说，如果 $f(x) = \sup_{\alpha \in \mathcal A}f_{\alpha}(x)$，且 $f_{\alpha}(x)$ 满足对于任意 $\alpha$，在 $x$ 上都是凸的，则若 $\beta = \arg\sup_{\alpha \in \mathcal A}f_{\alpha}(x)$，就有 $\partial f_{\beta}(x)\in \partial f$
这等价于在给定 $G$ 相对应的最优的 $D$ 时，对 $p_g$ 进行梯度下降更新，而 $\sup_D U(p_g,D)$ 在 $p_g$ 上是凸的，如 Theorem 1 中所证明，存在唯一的全局最小值，因此在对 $p_g$ 进行足够小的更新的情况下，$p_g$ 可以收敛到 $p_x$，证毕

在实践中，对抗式网络通过函数 $G(\symbfit z;\theta_g)$ 表示有限的 $p_g$ 分布族，我们优化参数 $\theta_g$ 而不是优化 $p_g$ 本身，使用多层感知机定义 $G$ 会在参数空间引入多个临界点 (critical points)，但虽然缺乏理论保证，多层感知机的优秀实际表现说明了使用多层感知机是合理的
# 5 Experiments
我们在 MNIST，TFD (Toronto Face Database)，CIFAR-10 上训练，生成器网络使用 ReLU 和 Sigmoid 激活函数，区分器网络使用 Maxout 激活函数，Dropout 应用于训练区分器网络

我们的理论框架允许我们在生成器的所有中间层都可以使用 dropout 或其他噪声，但我们仅在生成器网络的最底层使用噪声作为输入

我们通过为 $G$ 生成的样本拟合一个 Guassian Parzen 窗口以估计测试集数据在 $p_g$ 下的概率，并报告了该分布下的对数似然，Gaussians 的 $\sigma$ 参数通过在验证集上交叉验证得到
该过程由 Breuleux[8]提出，常用于测试准确的似然是不可解的 (the exact likelihood is not tractable) 生成式模型

结果报告于 Table 1，这种估计似然的方法有较高的方差，且在高维空间的效果不好，但是目前可用的最好方法
# 6 Advantages and disadvantages
该框架的缺点在于没有对 $p_g(\symbfit x)$ 的显式表示，且 $D$ 在训练时必须和 $G$ 同步好
(特别地，$D$ 不能在没有更新 $G$ 的情况下训练过度，以避免“the Helvetica scenario"，其中 $G$ 将太多 $\mathbf z$ 值 collapse 到相同的 $\mathbf x$ 值，以至于不能拥有足够的多样性来建模 $p_{data}$)
该框架的优点在于不需要马尔可夫链，只需要反向传播用于获取梯度，在学习中不需要推理 (inference)，以及模型可以包含大量的函数

以上提到的优势主要是计算上的，对抗式模型在统计上的优势在于，生成式网络不会直接被数据样本更新 (not being updated directly with data examples)，而是被流经区分器的梯度更新，这意味着输入的样例不会被直接拷贝到生成器的参数中，
对抗式网络的另一个优势在于它可以表示非常 sharp 甚至 degenerate 的分布，而基于马尔可夫链的模型则要求分布需要有点 blurry
# 7 Conclusions and future work
1. 一个条件下的生成式模型 $p(\symbfit x | \symbfit c)$ 可以通过对 $G$ 和 $D$ 的输入都加入 $\symbfit c$ 得到
2. 学习到的近似推理 (Learned approximate inference) 可以通过训练一个辅助网络在给定 $\symbfit x$ 时预测 $\symbfit z$ 得到，这和 wake-sleep[15]训练的推理网络类似，但其优势在于推理网络可以在生成器训练完成后，为一个固定的生成器训练
3. 可以通过训练一族共享参数的条件模型，近似建模所有的条件概率 $p(\symbfit x_S | \symbfit x_{\cancel S})$，其中 $S$ 是 $\symbfit x$ 的索引的子集
4. 半监督学习：来自区分器或推理网络的特征可以在标签数据有限时提升分类器的表现
5. 效率提升：设计更好的协调 $G$ 和 $D$ 的方法以加速训练，或决定更好的分布来在训练时采样 $\symbfit z$

