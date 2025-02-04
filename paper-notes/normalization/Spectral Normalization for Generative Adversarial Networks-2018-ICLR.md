# Abstract
One of the challenges in the study of generative adversarial networks is the instability of its training. In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator. Our new normalization technique is computationally light and easy to incorporate into existing implementations. We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques. The code with Chainer (Tokui et al., 2015), generated images and pretrained models are available at https://github.com/pfnet-research/sngan_ projection. 
>  GAN 训练的挑战之一是其不稳定性
>  本文提出新的权重规范化技术，称为谱规范化，以稳定判别器的训练
>  谱规范化计算成本低，易于整合到现有的实现中
>  试验证明了谱规范化的 GAN 相对于使用之前的训练稳定技术，能够生成质量更高的图像

# 1 Introduction
Generative adversarial networks (GANs) (Goodfellow et al., 2014) have been enjoying considerable success as a framework of generative models in recent years, and it has been applied to numerous types of tasks and datasets (Radford et al., 2016; Salimans et al., 2016; Ho & Ermon, 2016; Li et al., 2017). 

In a nutshell, GANs are a framework to produce a model distribution that mimics a given target distribution, and it consists of a generator that produces the model distribution and a discriminator that distinguishes the model distribution from the target. The concept is to consecutively train the model distribution and the discriminator in turn, with the goal of reducing the difference between the model distribution and the target distribution measured by the best discriminator possible at each step of the training. 
>  简而言之，GAN 是一类框架，它生成模仿目标分布的模型分布
>  GAN 由一个生成模型分布的生成器和一个区分模型分布和目标分布的判别器组成，GAN 交替训练判别器和生成器，目标是在每一步训练减少当前最优的判别器所度量了模型分布和目标分布的差异

GANs have been drawing attention in the machine learning community not only for its ability to learn highly structured probability distribution but also for its theoretically interesting aspects. For example, (Nowozin et al., 2016; Uehara et al., 2016; Mohamed & Lakshminarayanan, 2017) revealed that the training of the discriminator amounts to the training of a good estimator for the density ratio between the model distribution and the target. This is a perspective that opens the door to the methods of implicit models (Mohamed & Lakshminarayanan, 2017; Tran et al., 2017) that can be used to carry out variational optimization without the direct knowledge of the density function. 
>  (Nowozin et al., 2016; Uehara et al., 2016; Mohamed & Lakshminarayanan, 2017) 揭示了判别器的训练等价于训练一个密度比 (模型分布比去目标分布) 估计器，该视角引出了隐式模型方法，这些方法可以用于在没有密度函数的直接知识的情况下执行变分优化

A persisting challenge in the training of GANs is the performance control of the discriminator. In high dimensional spaces, the density ratio estimation by the discriminator is often inaccurate and unstable during the training, and generator networks fail to learn the multimodal structure of the target distribution. Even worse, when the support of the model distribution and the support of the target distribution are disjoint, there exists a discriminator that can perfectly distinguish the model distribution from the target (Arjovsky & Bottou, 2017). Once such discriminator is produced in this situation, the training of the generator comes to complete stop, because the derivative of the so-produced discriminator with respect to the input turns out to be 0. This motivates us to introduce some form of restriction to the choice of the discriminator. 
>  GAN 的训练中，判别器的性能控制一直是挑战，判别器在训练中对密度比的估计往往是不准确且不稳定的，导致生成网络无法学习目标分布的多模态结构
>  更糟糕的是，当模型分布的支持集和目标分布的支持集不相交时，存在一个判别器可以完美区分模型分布和目标分布，当在这种情况下产生了这样的判别器，生成器的训练就会完全停止，因为这样的判别器对输入的导数是零
>  这促使我们引入某种限制来约束判别器的选择

> [! info] 支撑集 (Support)
> 实值函数 $f: X\to \mathbb R$ 的支撑集是该函数定义域 $X$ 的一个子集，满足在 $f$ 下，该子集中的元素都被映射到非零值，记作 $\mathrm{supp}(f)$ ，即
> 
> $$\mathrm{supp}(f) = \{x\in X: f(x)\ne 0\} $$ 
> 
> 对于 $X$ 的任意子集 $C$，满足性质：使得所有 $X\backslash C$ 中的元素都被 $X$ 映射为零，的所有子集中，$f$ 的支撑集是其中最小的那个 (再小就会出现非零)
> 
> 如果实值函数 $f = 0$ 仅对于 $X$ 中的有限个点不成立 ($f(x)\ne 0$)，称 $f$ 具有有限的支撑集，也就是支撑集是有限集

In this paper, we propose a novel weight normalization method called spectral normalization that can stabilize the training of discriminator networks. Our normalization enjoys following favorable properties. 

- Lipschitz constant is the only hyper-parameter to be tuned, and the algorithm does not require intensive tuning of the only hyper-parameter for satisfactory performance. 
- Implementation is simple and the additional computational cost is small. 

>  本文提出称为谱规范化的权重规范化方法，用于稳定判别器网络的训练
>  谱规范化的优势有：
>  - 唯一要调节的超参数是 Lipschitz 常数，且算法不需要频繁调节该超参数以获得满意的性能
>  - 实现简单，额外开销小

In fact, our normalization method also functioned well even without tuning Lipschitz constant, which is the only hyper parameter. In this study, we provide explanations of the effectiveness of spectral normalization for GANs against other regularization techniques, such as weight normalization (Salimans & Kingma, 2016), weight clipping (Arjovsky et al., 2017), and gradient penalty (Gulrajani et al., 2017). We also show that, in the absence of complimentary regularization techniques (e.g., batch normalization, weight decay and feature matching on the discriminator), spectral normalization can improve the sheer quality of the generated images better than weight normalization and gradient penalty. 
>  实际上，谱规范化方法在不调节 Lipschitz 常数时也有良好表现
>  本文为谱规范化相对于其他正则化技术的优势进行解释，本文还表明，在缺乏互补的正则化技术 (例如批量规范化、权重衰减和判别器上的特征匹配) 的情况下，谱规范化可以比权重规范化和梯度惩罚更好地提高生成图像的质量

# 2 Method
In this section, we will lay the theoretical groundwork for our proposed method. Let us consider a simple discriminator made of a neural network of the following form, with the input $\pmb x$ : 

$$
f({\pmb x},{\theta})=W^{L+1}a_{L}(W^{L}(a_{L-1}(W^{L-1}(\dots a_{1}(W^{1}{\pmb x})\dots)))),\tag{1}
$$ 
where $\theta\;:=\;\{W^{1},\ldots,W^{L},W^{L+1}\}$ is the learning parameters set, $W^{l}\;\in\;\mathbb{R}^{d_{l}\times d_{l-1}}$ , $W^{L+1}\;\in$ $\mathbb{R}^{1\times d_{L}}$ , and $a_{l}$ is an element-wise non-linear activation function. We omit the bias term of each layer for simplicity.

>  考虑一个由神经网络实现的判别器网络 $f(\pmb x, \theta)$，输入为 $\pmb x$，需要学习的参数集合
>   $\theta := \{W^1, \dots, W^L, W^{L+1}\}$ 
>   $W^l\in \mathbb R^{d_l\times d_{l-1}}$，$W^L \in \mathbb R^{1\times d_L}$，$a_l$ 为逐元素的非线性激活函数
>  为了简化，忽略了每一层的偏置项

The final output of the discriminator is given by 

$$
D(\pmb{x},\theta)=\mathcal A(f(\pmb{x},\theta)),\tag{2}
$$ 
where ${\mathcal{A}}$ is an activation function corresponding to the divergence of distance measure of the user’s choice. 

>  判别器的最终输出 $D(\pmb x, \theta) = \mathcal A(f(\pmb x, \theta))$，其中 $\mathcal A$ 为对应于用户选择的距离度量的激活函数

The standard formulation of GANs is given by 

$$
\operatorname*{min}_{G}\operatorname*{max}_{D}V(G,D)
$$ 
where min and max of $G$ and $D$ are taken over the set of generator and discriminator functions, respectively. 

>  GAN 的标准形式如上，$G$ 的 $\min$ 和 $D$ 的 $\min$ 分别从生成器和判别器的函数集合中取到

The conventional form of $V(G,D)$ (Goodfellow et al., 2014) is given by $\mathrm{E}_{\pmb{x}\sim q_{\mathrm{data}}}[\log D(\pmb{x})]+\mathrm{E}_{\pmb{x}^{\prime}\sim p_{G}}[\log(1-D(\pmb{x}^{\prime}))]$ , where $q_{\mathrm{data}}$ is the data distribution and $p_{G}$ is the (model) generator distribution to be learned through the adversarial min-max optimization. The activation function ${\mathcal{A}}$ that is used in the $D$ of this expression is some continuous function with range $[0,1]$ (e.g, sigmoid function). 

>  $V(G, D)$ 的常规形式为 $\mathrm E_{\pmb x\sim q_{\mathrm{data}}}[\log D(\pmb x)] + \mathrm E_{\pmb x' \sim p_G}[\log (1-D(\pmb x'))]$，其中 $q_{\mathrm{data}}$ 为数据分布，$p_G$ 为通过对抗极大极小优化学习的模型分布
>  该表达式中，判别器 $D$ 中使用的激活函数 $\mathcal A$ 应该是值域为 $[0,1]$ 的连续函数 (例如 sigmoid 函数)

It is known that, for a fixed generator $G$ , the optimal discriminator for this form of ${V}(G,D)$ is given by $D_{G}^{*}(\pmb{x}):=q_{\mathrm{data}}(\pmb{x})/(q_{\mathrm{data}}(\pmb{x})+p_{G}(\pmb{x}))$ . 
>  对于固定的生成器 $G$，对于该形式的 $V(G, D)$，最优的判别器为 $D_G^*(\pmb x) := q_{\mathrm{data}}(\pmb x)/(q_{\mathrm{data}}(\pmb x) + p_G(\pmb x))$

The machine learning community has been pointing out recently that the function space from which the discriminators are selected crucially affects the performance of GANs. A number of works (Uehara et al., 2016; Qi, 2017; Gulrajani et al., 2017) advocate the importance of Lipschitz continuity in assuring the boundedness of statistics. For example, the optimal discriminator of GANs on the above standard formulation takes the form 

$$
\begin{align}
D_{G}^{*}(\pmb x)=\frac{q_{\mathrm{data}}(\pmb x)}{q_{\mathrm{data}}(\pmb x)+p_{G}(\pmb x)}&=\mathrm{sigmoid}(f^{*}(\pmb x)),\\
&\mathrm{where~}f^{*}(\pmb x)=\log q_{\mathrm{data}}(\pmb x)-\log p_{G}(\pmb x),
\end{align}\tag{3}
$$

and its derivative 

$$
\nabla_{\pmb{x}}f^{*}({\pmb x})=\frac{1}{q_{\mathrm{data}}(\pmb{x})}\nabla_{\pmb{x}}q_{\mathrm{data}}(\pmb{x})-\frac{1}{p_{G}(\pmb{x})}\nabla_{\pmb{x}}p_{G}(\pmb{x})\tag{4}
$$ 
can be unbounded or even incomputable. This prompts us to introduce some regularity condition to the derivative of $f({\pmb x})$ . 

>  判别器的函数空间对于 GAN 的性能至关重要，多项研究强调了 Lipschitz 连续性对于确保统计量有界性的重要性
>  例如，我们知道标准形式的 GAN 的最优判别器的形式如上，其中 $f^*(\pmb x)$ 的导数可能无界甚至无法计算，这引导我们引入为 $f(\pmb x)$ 的导数引入一些正则化条件

> [!info] 利普希茨连续 (Lipschitz continuity)
> 利普希茨连续是一个比[一致连续](https://zh.wikipedia.org/wiki/%E4%B8%80%E8%87%B4%E9%80%A3%E7%BA%8C "一致连续")更强的光滑性条件。直觉上，利普希茨连续函数限制了函数改变的速度：符合利普希茨条件的函数的斜率的绝对值，必小于一个称为利普希茨常数的实数（该常数依函数而定）。
> 
> 定义：
> 对于定义在在实数集的某个子集的函数 $f: D\subseteq \mathbb R \to \mathbb R$，若存在常数 $K$，使得
> 
>  $$|f(a) - f(b)|\le K|a-b|\quad\quad \forall a,b\in D$$
> 
> 成立，则称 $f$ 符合利普希茨条件，使 $f$ 能够满足利普希茨条件的最小的常数 $K$ 称为 $f$ 的利普希茨常数
> 
> 若 $K<1$，$f$ 称为收缩映射
> 
> 利普希茨条件也可以对任意度量空间的函数定义：
> 给定两个度量空间 $(M, d_M),(N, d_N)$。若对于函数 $f: U\subseteq M \to N$，存在常数 $K$ 使得
> 
> $$d_N(f(a),f(b))\le Kd_M(a,b)\quad\quad \forall a,b\in U$$
> 
> 则称 $f$ 符合利普希茨条件
> 
> 满足 Lipschitz 条件的函数连续且一致连续 (Lipschitz 条件将函数值之间的距离 $|f(a) - f(b)|$ 用变量之间的距离 $|a-b|$ 界定，要让函数值之间的距离小于特定正数 $\epsilon$，变量之间的距离上界 $\delta$ 选定为 $\frac {\epsilon}{K}$ 即可，故连续性得证，显然，$\delta$ 的选择只与 $\epsilon$ 有关，和变量本身在什么位置无关，故一致连续性得证)

A particularly successful works in this array are (Qi, 2017; Arjovsky et al., 2017; Gulrajani et al., 2017), which proposed methods to control the Lipschitz constant of the discriminator by adding regularization terms defined on input examples $\pmb x$ . We would follow their footsteps and search for the discriminator $D$ from the set of $K$ -Lipschitz continuous functions, that is, 

$$
{\underset{\|f\|_{\mathrm{Lip}}\leq K}{\operatorname{arg\,max}}}\,V(G,D),\tag{5}
$$ 
>  在这方面特别成功的工作包括 (Qi, 2017; Arjovsky el al, 2017; Gulrajani el al, 2017)，这些工作提出通过添加定义域输入样本 $\pmb x$ 上的正则化项来控制判别器的 Lipschitz 常数
>  我们跟随它们的工作，从 K-Lipschitz 连续函数集合中搜索判别器 $D$，也就是对于给定的生成器 $G$，我们在保证判别器的 Lipschitz 常数需要不大于 $K$ 的情况下找到最大化 $V(G, D)$ 的判别器

where we mean by $\|f\|_{\mathrm{Lip}}$ the smallest value $M$ such that $\|f(\pmb{x})-f(\pmb{x}^{\prime})\|/\|\pmb{x}-\pmb{x}^{\prime}\|\le M$ for any $\pmb {x},\pmb {x}^{\prime}$ , with the norm being the $\ell_{2}$ norm. 
>  (5) 中 $\|f\|_{\mathrm{Lip}}$ 即 $f$ 的 Lipschitz 常数，定义为使得 $\|f(\pmb x)- f(\pmb x')\|/\|\pmb x - \pmb x'\| \le M$ 对任意 $\pmb x, \pmb x'$ 满足的最小常数 $M$
>  其中范数是 $\ell_2$ 范数

While input based regularizations allow for relatively easy formulations based on samples, they also suffer from the fact that, they cannot impose regularization on the space outside of the supports of the generator and data distributions without introducing somewhat heuristic means. A method we would introduce in this paper, called spectral normalization, is a method that aims to skirt this issue by normalizing the weight matrices using the technique devised by Yoshida & Miyato (2017). 
>  基于输入的正则化方法可以简单地根据样本进行构造，但无法对生成器和数据分布的支撑集之外的空间施加正则化，除非引入某种启发式方法
>  本文提出谱规范化方法，旨在通过 Yoshida & Miyato (2017) 涉及的权重矩阵规范化方法来规避这一问题

## 2.1 Spectral Normalization
Our spectral normalization controls the Lipschitz constant of the discriminator function $f$ by literally constraining the spectral norm of each layer $g:\pmb h_{i n}\mapsto \pmb h_{o u t}$ . 
>  谱规范化通过约束每层 $g:\pmb h_{in} \mapsto \pmb h_{out}$ 的谱范数来控制判别函数 $f$ 的 Lipschitz 常数

By definition, Lipschitz norm $\|g\|_{\mathrm{Lip}}$ is equal to ${\mathrm{sup}}_{\pmb h}\,{\sigma(\nabla g(\pmb h))}$ , where $\sigma(A)$ is the spectral norm of the matrix $A$ ( $L_{2}$ matrix norm of $A$ ) 

$$
\sigma(A):=\operatorname*{max}_{\pmb h:\pmb h\neq0}\frac{\|A \pmb h\|_{2}}{\|\pmb h\|_{2}}=\operatorname*{max}_{\|\pmb h\|_{2}\leq1}\|A \pmb h\|_{2},\tag{6}
$$ 
>  线性层 $g(\pmb h)=W\pmb h$ 的 Lipschitz 范数/常数 $\|g\|_{\mathrm{Lip}}$ 定义为 $\sup_{\pmb h}\sigma(\nabla g(\pmb h))$，其中 $\sigma(A)$ 表示矩阵 $A$ 的谱范数/ $L_2$ 范数，定义如 (6) 所示

>  推导
>  假设 $g$ 是可微的，根据中值定理的推广形式，对于任意两点 $\pmb x, \pmb y\in \mathbb R^n$，存在某个点 $\pmb z$ 位于 $\pmb x, \pmb y$ 之间的线段上，使得 $g(\pmb x) - g(\pmb y) = \nabla g(\pmb z)(\pmb x- \pmb y)$，进而有 $\|g (\pmb x) - g (\pmb y) \|=\| \nabla g (\pmb z)(\pmb x- \pmb y)\|$ (范数是 $\ell_2$ 范数)
>  对于任意向量 $\pmb v$，有 $\|\nabla g(\pmb z) \pmb v\|\le \sigma(\nabla g(\pmb z))\|\pmb v\|$ (证明见 [[#Proof of Inequality]])
>  因此，我们有 $\|g (\pmb x) - g (\pmb y) \|\le \sigma(\nabla g (\pmb z))\|\pmb x- \pmb y\|$ ，即 $\frac {\|g(\pmb x)- g(\pmb y)\|}{\|\pmb x- \pmb y\|}\le \sigma(\nabla g(\pmb z))$ 对于任意 $\pmb x, \pmb y\in \mathbb R^n$ 成立，同时 $\pmb z$ 依赖于 $\pmb x, \pmb y$ 的选择
>  要找到 $g$ 的 Lipschitz 常数，我们需要考虑所有可能的 $\pmb z$，故需要取上确界，即 $\|g\|_{\mathrm{Lip}}  = \sup_{\pmb z}\sigma(\nabla g(\pmb z))$
>  
>  直观上，可以理解为 $\sigma(\nabla g(\pmb z))$ 给出了点 $\pmb z$ 处 $g$ 局部拉伸的最大比例，而 $g$ 整体的 Lipschitz 常数就需要考虑所有这些局部最大拉伸率中的最大值

which is equivalent to the largest singular value of $A$ . Therefore, for a linear layer $g(\pmb h)=W \pmb h$ , the norm is given by  $\begin{array}{r}{\|g\|_{\mathrm{Lip}}=\operatorname*{sup}_{\pmb h}\sigma(\operatorname{\nabla}\!g(\pmb h))=\operatorname*{sup}_{\pmb h}\sigma(W)=\sigma(W)}\end{array}$ . 
>  矩阵 $A$ 的谱范数实际上等于 $A$ 的最大奇异值，因此，$g$ 的 Lipschitz 范数写为
>  $\|g\|_{\mathrm{Lip}} = \sup_{\pmb h}\sigma(\nabla g(\pmb h))=\sup_{\pmb h}\sigma(W) = \sigma(W)$，也就是 $g$ 相对于输入的 Jacobian 矩阵 (即权重矩阵) 的谱范数，也就是权重矩阵的最大奇异值

>  推导
 > $\begin{array}{r}{\|g\|_{\mathrm{Lip}}=\operatorname*{sup}_{\pmb h}\sigma(\operatorname{\nabla}\!g(\pmb h))=\operatorname*{sup}_{\pmb h}\sigma(W)=\sigma(W)}\end{array}$ 中：
 > 第一个等式源自 Lipschitz 范数的定义
 > 第二个等式求出了 Jacobian 矩阵 $\nabla g(\pmb h) = W$
 > 第三个等式是因为 $\sigma(W)$ 和输入 $\pmb h$ 无关，因此 $\sigma(W)$ 相对于 $\pmb h$ 的上确界就是 $\sigma(W)$

If the Lipschitz norm of the activation function $\|a_{l}\|_{\mathrm{Lip}}$ is equal to $1$ , we can use the inequality $\|g_{1}\circ g_{2}\|_{\mathrm{Lip}}\leq\|g_{1}\|_{\mathrm{Lip}}\cdot\|g_{2}\|_{\mathrm{Lip}}$ to observe the following bound on $\|f\|_{\mathrm{Lip}}$ : 

$$
\begin{array}{r l}&{\|f\|_{\mathrm{Lip}}\leq\|(\pmb h_{L}\mapsto W^{L+1}\pmb h_{L})\|_{\mathrm{Lip}}\cdot\|a_{L}\|_{\mathrm{Lip}}\cdot\|(\pmb h_{L-1}\mapsto W^{L}\pmb h_{L-1})\|_{\mathrm{Lip}}}\\ &{\qquad\qquad\cdot\cdot\|a_{1}\|_{\mathrm{Lip}}\cdot\|(\pmb h_{0}\mapsto W^{1}\pmb h_{0})\|_{\mathrm{Lip}}=\displaystyle\prod_{l=1}^{L+1}\|(\pmb h_{l-1}\mapsto W^{l}\pmb h_{l-1})\|_{\mathrm{Lip}}=\displaystyle\prod_{l=1}^{L+1}\sigma(W^{l}).}\end{array}
$$ 
>  如果所有层的激活函数 $a_l$ 的 Lipschitz 范数都等于 1 (例如 ReLU 和 leaky ReLU)，我们可以利用不等式 $\|g_1\circ g_2\|_{\mathrm{Lip}}\le \|g_1\|_{\mathrm{Lip}}\cdot \|g_2\|_{\mathrm{Lip}}$ 得到 $\|f\|_{\mathrm{Lip}}$ 的一个上界，如上所示
>  可以看到，该上界即所有层的 Lipschitz 范数的积，也就是所有权重矩阵的最大奇异值的乘积

>  推导
>  假设 $g_1: \mathbb R^m \mapsto \mathbb R^p$ 和 $g_2: \mathbb R^n \mapsto \mathbb R^m$ 是两个 Lipschitz 连续函数
>  考虑任意两点 $\pmb x, \pmb y \in \mathbb R^n$，根据 Lipschitz 条件，有
>  $\|g_2(\pmb x) - g_2(\pmb y)\| \le \|g_2\|_{\mathrm{Lip}}\|\pmb x- \pmb y\|$
>  同时对于 $g_2(\pmb x)$ 和 $g_2(\pmb y)$，根据 Lipschitz 条件，有
>  $\|g_1(g_2(\pmb x)) - g_1(g_2(\pmb y)) \|\le \|g_1\|_{\mathrm{Lip}} \|g_2(\pmb x) - g_2(\pmb y)\|$
>  结合两个不等式，得到
>  $\|g_1 (g_2 (\pmb x)) - g_1 (g_2 (\pmb y)) \|\le \|g_1\|_{\mathrm{Lip}} \|g_2 (\pmb x) - g_2 (\pmb y)\|\le\|g_1\|_{\mathrm{Lip}}\cdot \|g_2\|_{\mathrm{Lip}}\|\pmb x-\pmb y\|$
>  也就是说，对于任意的 $\pmb x, \pmb y\in \mathbb R^n$，有
>  $\|g_1 (g_2 (\pmb x)) - g_1 (g_2 (\pmb y)) \|\ \le\|g_1\|_{\mathrm{Lip}}\cdot \|g_2\|_{\mathrm{Lip}}\|\pmb x-\pmb y\|$
>  即 $\|g_1 \circ g_2 (\pmb x) - g_1 \circ g_2 (\pmb y) \|\ \le\|g_1\|_{\mathrm{Lip}}\cdot \|g_2\|_{\mathrm{Lip}}\|\pmb x-\pmb y\|$
>  因此容易得出结论 $\|g_1\circ g_2\|_{\mathrm{Lip}} \le \|g_1\|_{\mathrm {Lip}}\cdot \|g_2\|_{\mathrm{Lip}}$
>  
>  该结论告诉我们，两个 Lipschitz 连续函数的复合仍然是 Lipschitz 连续的，且其 Lipschitz 常数不超过原来两个 Lipschitz 常数的乘积

Our spectral normalization normalizes the spectral norm of the weight matrix $W$ so that it satisfies the Lipschitz constraint $\sigma(W)=1$ : 

$$
\bar{W}_{\mathrm{SN}}(W):=W/\sigma(W).\tag{8}
$$

>  谱规范化将权重矩阵 $W$ 的谱范数 (线性层的 Lipschitz 范数) 进行规范化，令其满足 Lipschitz 约束 $\sigma(W) = 1$，也就是规范化后的 $W$ 的最大奇异值等于 $1$
>  实现时，按照 (8) 将 $W$ 除去 $W$ 的最大奇异值 $\sigma(W)$ 即可

If we normalize each $W^{l}$ using (8), we can appeal to the inequality (7) and the fact that $\sigma\left(\bar{W}_{\mathrm{SN}}(W)\right)=1$ to see that $\|f\|_{\mathrm{Lip}}$ is bounded from above by 1. 
>  当我们根据 (8) 将所有的 $W^l$ 规范化之后 (此时所有的 $g$ 的 Lipschitz 常数的上界是 $1$ )，我们就可以根据不等式 (7) 得到 $f$ 的 Lipschitz 常数 $\|f\|_{\mathrm{Lip}}$ 的上界就是 $1$

Here, we would like to emphasize the difference between our spectral normalization and spectral norm ”regularization” introduced by Yoshida & Miyato (2017). Unlike our method, spectral norm ”regularization” penalizes the spectral norm by adding explicit regularization term to the objective function. Their method is fundamentally different from our method in that they do not make an attempt to ‘set’ the spectral norm to a designated value.
>  谱规范化和 Yoshida & Miyato (2017) 提出的谱范数正则化方法存在不同，谱范数正则化方法通过为目标函数添加显示的正则化项来惩罚，并不试图将谱范数设定为指定的值

Moreover, when we reorganize the derivative of our normalized cost function and rewrite our objective function (12), we see that our method is augmenting the cost function with a sample data dependent regularization function. Spectral norm regularization, on the other hand, imposes sample data independent regularization on the cost function, just like L2 regularization and Lasso. 
>  另外，重新组织规范化成本函数并且重写目标函数 (12) 后，我们可以看到我们的方法是通过一个依赖于样本的正则化函数来增强成本函数。而谱范数正则化则对成本函数施加的是样本无关的正则化项，就如 L2 和 Lasso 正则化

## 2.2 Fast Approximation of The Spectral Norm $\sigma(W)$ 
As we mentioned above, the spectral norm $\sigma(W)$ that we use to regularize each layer of the discriminator is the largest singular value of $W$ . If we naively apply singular value decomposition to compute the $\sigma(W)$ at each round of the algorithm, the algorithm can become computationally heavy. Instead, we can use the power iteration method to estimate $\sigma(W)$ (Golub & Van der Vorst, 2000; Yoshida & Miyato, 2017). With power iteration method, we can estimate the spectral norm with very small additional computational time relative to the full computational cost of the vanilla GANs. Please see Appendix A for the detail method and Algorithm 1 for the summary of the actual spectral normalization algorithm. 
>  我们使用幂迭代方法来估计谱范数 $\sigma(W)$，这样引入的额外的计算成本相对于标准的 GANs 的完全计算成本是非常小的

## 2.3 Gradient Analysis of The Spectrally Normalized Weights 
The gradient of $\bar{W}_{\mathrm{SN}}(W)$ with respect to $W_{i j}$ is: 

$$
\begin{align}
\frac {\partial \bar W_{\mathrm{SN}}(W)}{\partial W_{ij}}=\frac {1}{\sigma(W)}E_{ij} - \frac {1}{\sigma(W)^2}\frac{\partial \sigma(W)}{\partial W_{ij}}W &=\frac{1}{\sigma(W)}E_{ij} - \frac {[\pmb u_1\pmb v_1^T]_{ij}}{\sigma(W)^2}W\tag{9}\\
&=\frac{1}{\sigma(W)}(E_{ij}-[\pmb u_1\pmb v_1^T]_{ij}\bar W_{\mathrm{SN}})\tag{10}
\end{align}
$$

where $E_{i j}$ is the matrix whose $(i,j)$ -th entry is 1 and zero everywhere else, and $\pmb{u}_{1}$ and $\pmb{v}_{1}$ are respectively the first left and right singular vectors of $W$ . 

>  推导

$$
\begin{align}
\frac{\partial \bar W_{\mathrm {SN}}(W)}{\partial W_{ij}} 
&=\frac{\partial }{\partial W_{ij}}\left(\frac{W}{\sigma(W)}\right)\\
&=\frac {\sigma(W)\frac {\partial W}{\partial W_{ij}}-\frac {\partial\sigma(W)}{\partial W_{ij}}W}{\sigma(W)^2}\\
&=\frac {1}{\sigma(W)}\frac {\partial W}{\partial W_{ij}} - \frac {1}{\sigma(W)^2}\frac {\partial \sigma(W)}{\partial W_{ij}}W\\
\end{align}
$$

>  其中 $\frac {\partial W}{\partial W_{ij}}$ 等于矩阵 $E_{ij}$，在位置 $(i, j)$ 处为 $1$，其余都为 $0$
>  进一步有

$$
\begin{align}
\frac{\partial \bar W_{\mathrm {SN}}(W)}{\partial W_{ij}} 
&=\frac {1}{\sigma(W)}\frac {\partial W}{\partial W_{ij}} - \frac {1}{\sigma(W)^2}\frac {\partial \sigma(W)}{\partial W_{ij}}W\\
&=\frac {1}{\sigma(W)}E_{ij} - \frac {1}{\sigma(W)^2}\frac {\partial \sigma(W)}{\partial W_{ij}}W\\
&=\frac {1}{\sigma(W)}E_{ij} - \frac {[\pmb u_1\pmb v_1^T]_{ij}}{\sigma(W)^2}W\\
\end{align}
$$

>  这里应用了结论 $\frac {\partial \sigma(W)}{\partial W_{ij}}=[\pmb u_1\pmb v_1^T]_{ij}$
>  其中 $\pmb u_1,\pmb v_1$ 分别是 $W$ 的第一左奇异向量和第一右奇异向量
>  进一步有

$$
\begin{align}
\frac{\partial \bar W_{\mathrm {SN}}(W)}{\partial W_{ij}} 
&=\frac {1}{\sigma(W)}E_{ij} - \frac {[\pmb u_1\pmb v_1^T]_{ij}}{\sigma(W)^2}W\\
&=\frac 1 {\sigma(W)}\left(E_{ij}-[\pmb u_1\pmb v_1^T]_{ij}\frac {W}{\sigma(W)}\right)\\
&=\frac 1 {\sigma(W)}\left(E_{ij}-[\pmb u_1\pmb v_1^T]_{ij}\bar W_{\mathrm {SN}}\right)\\
\end{align}
$$

>  推导完毕

If $^h$ is the hidden layer in the network to be transformed by $\bar{W}_{S N}$ , the derivative of the $V(G,D)$ calculated over the mini-batch with respect to $W$ of the discriminator $D$ is given by: 

$$
\begin{array}{c}{\displaystyle\frac{\partial V(G,D)}{\partial W}=\frac{1}{\sigma(W)}\left(\hat{\mathrm{E}}\left[\delta\boldsymbol{h}^{\mathrm{T}}\right]-\left(\hat{\mathrm{E}}\left[\delta^{\mathrm{T}}\bar{W}_{\mathrm{SN}}\boldsymbol{h}\right]\right)\boldsymbol{u}_{1}\boldsymbol{v}_{1}^{\mathrm{T}}\right)}\\ {\displaystyle=\frac{1}{\sigma(W)}\left(\hat{\mathrm{E}}\left[\delta\boldsymbol{h}^{\mathrm{T}}\right]-\lambda\boldsymbol{u}_{1}\boldsymbol{v}_{1}^{\mathrm{T}}\right)}\end{array}
$$ 
where $\delta:=\left(\partial V(G,D)/\partial\left(\bar{W}_{\mathrm{SN}}h\right)\right)^{\mathrm{T}},\lambda:=\hat{\mathrm{E}}\left[\delta^{\mathrm{T}}\left(\bar{W}_{\mathrm{SN}}h\right)\right]$ , and $\hat{\mathrm{E}}\!\left[\cdot\right]$ represents empirical expectation over the mini-batch. $\begin{array}{r}{\frac{\partial V}{\partial W}=0}\end{array}$ when $\hat{\mathrm{E}}[\delta h^{\mathrm{T}}]=k\pmb{u}_{1}\pmb{v}_{1}^{T}$ for some $k\in\mathbb{R}$ . 

We would like to comment on the implication of (12). The first term $\hat{\mathrm{~E~}}\big[\delta h^{\mathrm{T}}\big]$ is the same as the derivative of the weights without normalization. In this light, the second term in the expression can be seen as the regularization term penalizing the first singular components with the adaptive regularization coefficient $\lambda$ . $\lambda$ is positive when $\delta$ and $\bar{W}_{\mathrm{SN}}h$ are pointing in similar direction, and this prevents the column space of $W$ from concentrating into one particular direction in the course of the training. In other words, spectral normalization prevents the transformation of each layer from becoming to sensitive in one direction. We can also use spectral normalization to devise a new parametrization for the model. Namely, we can split the layer map into two separate trainable components: spectrally normalized map and the spectral norm constant. As it turns out, this parametrization has its merit on its own and promotes the performance of GANs (See Appendix E). 

# 3 Spectral Normalization vs Other Regularization Techniques
The weight normalization introduced by Salimans & Kingma (2016) is a method that normalizes the $\ell_{2}$ norm of each row vector in the weight matrix. Mathematically, this is equivalent to requiring the weight by the weight normalization $\bar{W}_{\mathrm{WN}}$ : 

$$
\sigma_{1}(\bar{W}_{\mathrm{WN}})^{2}+\sigma_{2}(\bar{W}_{\mathrm{WN}})^{2}+\cdots+\sigma_{T}(\bar{W}_{\mathrm{WN}})^{2}=d_{o},\;\mathrm{where}\;T=\operatorname*{min}(d_{i},d_{o}),
$$ 
where $\sigma_{t}(A)$ is a $t$ -th singular value of matrix $A$ . Therefore, up to a scaler, this is same as the Frobenius normalization, which requires the sum of the squared singular values to be 1. These normalizations, however, inadvertently impose much stronger constraint on the matrix than intended. If $\bar{W}_{\mathrm{WN}}$ is the weight normalized matrix of dimension $d_{i}\times d_{o}$ , the norm $\lVert\bar{W}_{\mathrm{WN}}h\rVert_{2}$ for a fixed unit vector $^h$ is maximized at $\|\bar{W}_{\mathrm{WN}}\underline{{{h}}}\|_{2}\;=\;\sqrt{d_{o}}$ when $\sigma_{1}(\bar{W}_{\mathrm{WN}})\;=\;\sqrt{d_{o}}$ and $\sigma_{t}(\bar{W}_{\mathrm{WN}})\;=\;0$ for $t\,=\,2,\ldots,T$ , which means that $\bar{W}_{\mathrm{WN}}$ is of rank one. Similar thing can be said to the Frobenius normalization (See the appendix for more details). Using such $\bar{W}_{\mathrm{WN}}$ corresponds to using only one feature to discriminate the model probability distribution from the target. In order to retain as much norm of the input as possible and hence to make the discriminator more sensitive, one would hope to make the norm of $\bar{W}_{\mathrm{WN}}h$ large. For weight normalization, however, this comes at the cost of reducing the rank and hence the number of features to be used for the discriminator. Thus, there is a conflict of interests between weight normalization and our desire to use as many features as possible to distinguish the generator distribution from the target distribution. The former interest often reigns over the other in many cases, inadvertently diminishing the number of features to be used by the discriminators. Consequently, the algorithm would produce a rather arbitrary model distribution that matches the target distribution only at select few features. Weight clipping (Arjovsky et al., 2017) also suffers from same pitfall. 

Our spectral normalization, on the other hand, do not suffer from such a conflict in interest. Note that the Lipschitz constant of a linear operator is determined only by the maximum singular value. In other words, the spectral norm is independent of rank. Thus, unlike the weight normalization, our spectral normalization allows the parameter matrix to use as many features as possible while satisfying local 1-Lipschitz constraint. Our spectral normalization leaves more freedom in choosing the number of singular components (features) to feed to the next layer of the discriminator. 
Brock et al. (2016) introduced orthonormal regularization on each weight to stabilize the training of GANs. In their work, Brock et al. (2016) augmented the adversarial objective function by adding the following term: 

$$
\lVert W^{\mathrm{T}}W-I\rVert_{F}^{2}.
$$ 
While this seems to serve the same purpose as spectral normalization, orthonormal regularization are mathematically quite different from our spectral normalization because the orthonormal regularization destroys the information about the spectrum by setting all the singular values to one. On the other hand, spectral normalization only scales the spectrum so that the its maximum will be one. 
Gulrajani et al. (2017) used Gradient penalty method in combination with WGAN. In their work, they placed $K$ -Lipschitz constant on the discriminator by augmenting the objective function with the regularizer that rewards the function for having local 1-Lipschitz constant (i.e. $\|\nabla_{\hat{\mathbf{x}}}f\|_{2}=1\rangle$ ) at discrete sets of points of the form $\hat{\pmb{x}}:=\epsilon\tilde{\pmb{x}}+\bar{(1-\epsilon)}\pmb{x}$ generated by interpolating a sample $\tilde{\pmb{x}}$ from generative distribution and a sample $\textbf{\em x}$ from the data distribution. While this rather straightforward approach does not suffer from the problems we mentioned above regarding the effective dimension of the feature space, the approach has an obvious weakness of being heavily dependent on the support of the current generative distribution. As a matter of course, the generative distribution and its support gradually changes in the course of the training, and this can destabilize the effect of such regularization. In fact, we empirically observed that a high learning rate can destabilize the performance of WGAN-GP. On the contrary, our spectral normalization regularizes the function the operator space, and the effect of the regularization is more stable with respect to the choice of the batch. Training with our spectral normalization does not easily destabilize with aggressive learning rate. Moreover, WGAN-GP requires more computational cost than our spectral normalization with single-step power iteration, because the computation of $\|\nabla_{\hat{\boldsymbol{x}}}f\|_{2}$ requires one whole round of forward and backward propagation. In the appendix section, we compare the computational cost of the two methods for the same number of updates. 

# 4 Experiments
In order to evaluate the efficacy of our approach and investigate the reason behind its efficacy, we conducted a set of extensive experiments of unsupervised image generation on CIFAR-10 (Torralba et al., 2008) and STL-10 (Coates et al., 2011), and compared our method against other normalization techniques. To see how our method fares against large dataset, we also applied our method on ILSVRC2012 dataset (ImageNet) (Russakovsky et al., 2015) as well. This section is structured as follows. First, we will discuss the objective functions we used to train the architecture, and then we will describe the optimization settings we used in the experiments. We will then explain two performance measures on the images to evaluate the images produced by the trained generators. Finally, we will summarize our results on CIFAR-10, STL-10, and ImageNet. 
As for the architecture of the discriminator and generator, we used convolutional neural networks. Also, for the evaluation of the spectral norm for the convolutional weight $W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}\times h\times w}$ , we treated the operator as a 2-D matrix of dimension $d_{\mathrm{out}}\times(d_{\mathrm{in}}h w)^{3}$ . We trained the parameters of the generator with batch normalization (Ioffe & Szegedy, 2015). We refer the readers to Table 3 in the appendix section for more details of the architectures. 

For all methods other than WGAN-GP, we used the following standard objective function for the adversarial loss: 

$$
V(G,D):=\underset{x\sim q_{\mathrm{data}}(x)}{\mathrm{E}}[\log D(\pmb{x})]+\underset{z\sim p(z)}{\mathrm{E}}[\log(1-D(G(z)))],
$$ 
where $z\in\mathbb{R}^{d_{z}}$ is a latent variable, $p(z)$ is the standard normal distribution ${\mathcal{N}}(0,I)$ , and $G:\mathbb{R}^{d_{z}}\rightarrow$ $\mathbb{R}^{d_{0}}$ is a deterministic generator function. We set $d_{z}$ to 128 for all of our experiments. For the updates of $G$ , we used the alternate cost proposed by Goodfellow et al. $(2014)-\mathrm{E}_{z\sim p(z)}[\log(D(G(z)))]$ as used in Goodfellow et al. (2014) and Warde-Farley & Bengio (2017). For the updates of $D$ , we used the original cost defined in (15). We also tested the performance of the algorithm with the so-called hinge loss, which is given by 
$$
\begin{array}{r l}&{V_{D}(\hat{G},D)=\underset{\mathbf{x}\sim q_{\mathrm{data}}(\mathbf{x})}{\mathrm{E}}\left[\operatorname*{min}\left(0,-1+D(\mathbf{x})\right)\right]+\underset{\mathbf{z}\sim p(\mathbf{z})}{\mathrm{E}}\left[\operatorname*{min}\left(0,-1-D\left(\hat{G}(z)\right)\right)\right]}\\ &{V_{G}(G,\hat{D})=-\underset{\mathbf{z}\sim p(z)}{\mathrm{E}}\left[\hat{D}\left(G(z)\right)\right],}\end{array}
$$ 
respectively for the discriminator and the generator. Optimizing these objectives is equivalent to minimizing the so-called reverse KL divergence : $\mathrm{KL}[p_{g}]|q_{\mathrm{data}}\]$ . This type of loss has been already proposed and used in Lim & Ye (2017); Tran et al. (2017). The algorithm based on the hinge loss also showed good performance when evaluated with inception score and FID. For Wasserstein GANs with gradient penalty (WGAN-GP) (Gulrajani et al., 2017), we used the following objective function: $V(G,D):=\operatorname{E}_{x\sim q_{\mathrm{data}}}[D(x)]-\operatorname{E}_{z\sim p(z)}[D(G(z))]-\lambda\operatorname{E}_{\hat{x}\sim p_{\hat{\alpha}}}[(\|\nabla_{\hat{x}}D(\hat{\pmb{x}})\|_{2}-1)^{2}]$ , where the regularization term is the one we introduced in the appendix section D.4. 
For quantitative assessment of generated examples, we used inception score (Salimans et al., 2016) and Fr´echet inception distance (FID) (Heusel et al., 2017). Please see Appendix B.1 for the details of each score. 

## 4.1 Results on CIFAR10 and STL-10 
In this section, we report the accuracy of the spectral normalization (we use the abbreviation: SNGAN for the spectrally normalized GANs) during the training, and the dependence of the algorithm’s performance on the hyperparmeters of the optimizer. We also compare the performance quality of the algorithm against those of other regularization/normalization techniques for the discriminator networks, including: Weight clipping (Arjovsky et al., 2017), WGAN-GP (Gulrajani et al., 2017), batch-normalization (BN) (Ioffe & Szegedy, 2015), layer normalization (LN) (Ba et al., 2016), weight normalization (WN) (Salimans & Kingma, 2016) and orthonormal regularization (orthonormal) (Brock et al., 2016). In order to evaluate the stand-alone efficacy of the gradient penalty, we also applied the gradient penalty term to the standard adversarial loss of GANs (15). We would refer to this method as ‘GAN-GP’. For weight clipping, we followed the original work Arjovsky et al. (2017) and set the clipping constant $c$ at 0.01 for the convolutional weight of each layer. For gradient penalty, we set $\lambda$ to 10, as suggested in Gulrajani et al. (2017). For orthonormal, we initialized the each weight of $D$ with a randomly selected orthonormal operator and trained GANs with the objective function augmented with the regularization term used in Brock et al. (2016). For all comparative studies throughout, we excluded the multiplier parameter $\gamma$ in the weight normalization method, as well as in batch normalization and layer normalization method. This was done in order to prevent the methods from overtly violating the Lipschitz condition. When we experimented with different multiplier parameter, we were in fact not able to achieve any improvement. 
For optimization, we used the Adam optimizer Kingma & Ba (2015) in all of our experiments. We tested with 6 settings for (1) $n_{\mathrm{dis}}$ , the number of updates of the discriminator per one update of the generator and (2) learning rate $\alpha$ and the first and second order momentum parameters $(\beta_{1},\beta_{2})$ of Adam. We list the details of these settings in Table 1 in the appendix section. Out of these 6 settings, A, B, and C are the settings used in previous representative works. The purpose of the settings D, E, and F is to the evaluate the performance of the algorithms implemented with more aggressive learning rates. For the details of the architectures of convolutional networks deployed in the generator and the discriminator, we refer the readers to Table 3 in the appendix section. The number of updates for GAN generator were 100K for all experiments, unless otherwise noted. 
Firstly, we inspected the spectral norm of each layer during the training to make sure that our spectral normalization procedure is indeed serving its purpose. As we can see in the Figure 9 in the C.1, the spectral norms of these layers floats around 1–1.05 region throughout the training. Please see Appendix C.1 for more details. 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/a9747fa4edbc0e961799b3c3e51e6cdadcc2a658097de48f33b9f778de2e3c19.jpg) 
Figure 1: Inception scores on CIFAR-10 and STL-10 with different methods and hyperparameters (higher is better). 
In Figures 1 and 2 we show the inception scores of each method with the settings A–F. We can see that spectral normalization is relatively robust with aggressive learning rates and momentum parameters. WGAN-GP fails to train good GANs at high learning rates and high momentum parameters on both CIFAR-10 and STL-10. Orthonormal regularization performed poorly for the setting E on the STL-10, but performed slightly better than our method with the optimal setting. These results suggests that our method is more robust than other methods with respect to the change in the setting of the training. Also, the optimal performance of weight normalization was inferior to both WGAN-GP and spectral normalization on STL-10, which consists of more diverse examples than CIFAR-10. Best scores of spectral normalization are better than almost all other methods on both CIFAR-10 and STL-10. 
In Tables 2, we show the inception scores of the different methods with optimal settings on CIFAR10 and STL-10 dataset. We see that SN-GANs performed better than almost all contemporaries on the optimal settings. SN-GANs performed even better with hinge loss (17).4. For the training with same number of iterations, SN-GANs fell behind orthonormal regularization for STL-10. For more detailed comparison between orthonormal regularization and spectral normalization, please see section 4.1.2. 
In Figure 6 we show the images produced by the generators trained with WGAN-GP, weight normalization, and spectral normalization. SN-GANs were consistently better than GANs with weight normalization in terms of the quality of generated images. To be more precise, as we mentioned in Section 3, the set of images generated by spectral normalization was clearer and more diverse than the images produced by the weight normalization. We can also see that WGAN-GP failed to train good GANs with high learning rates and high momentums (D,E and F). The generated images with GAN-GP, batch normalization, and layer normalization is shown in Figure 12 in the appendix section. 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/43da615126ee7e4844f6fd8f9365866089e05bd988a8bfcfda1aca39a9a8f981.jpg) 
Figure 2: FIDs on CIFAR-10 and STL-10 with different methods and hyperparameters (lower is better). 
Table 2: Inception scores and FIDs with unsupervised image generation on CIFAR-10. $\dagger$ (Radford et al., 2016) (experimented by Yang et al. (2017)), $^{\ddag}$ (Yang et al., 2017), $^*$ (Warde-Farley & Bengio, 2017), †† (Gulrajani et al., 2017) 
<html><body><table><tr><td colspan="3">Inception score</td><td colspan="2">FID</td></tr><tr><td>Method</td><td>CIFAR-10</td><td>STL-10</td><td>CIFAR-10</td><td>STL-10</td></tr><tr><td>Real data</td><td>11.24±.12</td><td>26.08±.26</td><td>7.8</td><td>7.9</td></tr><tr><td>-Standard CNN-</td><td></td><td></td><td></td><td></td></tr><tr><td>Weight clipping</td><td>6.41±.11</td><td>7.57±.10</td><td>42.6</td><td>64.2</td></tr><tr><td>GAN-GP</td><td>6.93±.08</td><td></td><td>37.7</td><td></td></tr><tr><td>WGAN-GP</td><td>6.68±.06</td><td>8.42±.13</td><td>40.2</td><td>55.1</td></tr><tr><td>Batch Norm.</td><td>6.27±.10</td><td></td><td>56.3</td><td></td></tr><tr><td>Layer Norm.</td><td>7.19±.12</td><td>7.61±.12</td><td>33.9</td><td>75.6</td></tr><tr><td>Weight Norm.</td><td>6.84±.07</td><td>7.16±.10</td><td>34.7</td><td>73.4</td></tr><tr><td>Orthonormal</td><td>7.40±.12</td><td>8.56±.07</td><td>29.0</td><td>46.7</td></tr><tr><td>(ours) SN-GANs</td><td>7.42±.08</td><td>8.28±.09</td><td>29.3</td><td>53.1</td></tr><tr><td>Orthonormal (2x updates)</td><td></td><td>8.67±.08</td><td></td><td>44.2</td></tr><tr><td>(ours) SN-GANs (2x updates)</td><td></td><td>8.69±.09</td><td></td><td>47.5</td></tr><tr><td>(ours) SN-GANs, Eq.(17)</td><td>7.58±.12</td><td></td><td>25.5</td><td></td></tr><tr><td>(ours) SN-GANs, Eq.(17) (2x updates)</td><td></td><td>8.79±.14</td><td></td><td>43.2</td></tr><tr><td>-ResNet-5</td><td></td><td></td><td></td><td></td></tr><tr><td>Orthonormal, Eq.(17)</td><td>7.92±.04</td><td>8.72±.06</td><td>23.8±.58</td><td>42.4±.99</td></tr><tr><td>(ours) SN-GANs, Eq.(17)</td><td>8.22±.05</td><td>9.10±.04</td><td>21.7±.21</td><td>40.1±.50</td></tr><tr><td>DCGANt</td><td>6.64±.14</td><td>7.84±.07</td><td></td><td></td></tr><tr><td>LR-GANst</td><td>7.17±.07</td><td></td><td></td><td></td></tr><tr><td>Warde-Farley et al.*</td><td>7.72±.13</td><td>8.51±.13</td><td></td><td></td></tr><tr><td>WGAN-GP (ResNet)tt</td><td>7.86±.08</td><td></td><td></td><td></td></tr></table></body></html> 
We also compared our algorithm against multiple benchmark methods ans summarized the results on the bottom half of the Table 2. We also tested the performance of our method on ResNet based GANs used in Gulrajani et al. (2017). Please note that all methods listed thereof are all different in both optimization methods and the architecture of the model. Please see Table 4 and 5 in the appendix section for the detail network architectures. Our implementation of our algorithm was able to perform better than almost all the predecessors in the performance. 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/01e71aec2a9b87c5fc89cbe8941152faafe540a0d3cd8b1c1397388c06c310fb.jpg) 
Figure 3: Squared singular values of weight matrices trained with different methods: Weight clipping (WC), Weight Normalization (WN) and Spectral Normalization (SN). We scaled the singular values so that the largest singular values is equal to 1. For WN and SN, we calculated singular values of the normalized weight matrices. 
### 4.1.1 Analysis of SN-GANS 
Singular values analysis on the weights of the discriminator $D$ In Figure 3, we show the squared singular values of the weight matrices in the final discriminator $D$ produced by each method using the parameter that yielded the best inception score. As we predicted in Section 3, the singular values of the first to fifth layers trained with weight clipping and weight normalization concentrate on a few components. That is, the weight matrices of these layers tend to be rank deficit. On the other hand, the singular values of the weight matrices in those layers trained with spectral normalization is more broadly distributed. When the goal is to distinguish a pair of probability distributions on the low-dimensional nonlinear data manifold embedded in a high dimensional space, rank deficiencies in lower layers can be especially fatal. Outputs of lower layers have gone through only a few sets of rectified linear transformations, which means that they tend to lie on the space that is linear in most parts. Marginalizing out many features of the input distribution in such space can result in oversimplified discriminator. We can actually confirm the effect of this phenomenon on the generated images especially in Figure 6b. The images generated with spectral normalization is more diverse and complex than those generated with weight normalization. 
Training time On CIFAR-10, SN-GANs is slightly slower than weight normalization (about 110 $\sim120\%$ computational time), but significantly faster than WGAN-GP. As we mentioned in Section 3, WGAN-GP is slower than other methods because WGAN-GP needs to calculate the gradient of gradient norm $\|\nabla_{\pmb{x}}D\|_{2}$ . For STL-10, the computational time of SN-GANs is almost the same as vanilla GANs, because the relative computational cost of the power iteration (18) is negligible when compared to the cost of forward and backward propagation on CIFAR-10 (images size of STL-10 is larger $(48\times48)_{.}$ ). Please see Figure 10 in the appendix section for the actual computational time. 

### 4.1.2 Comparison Between SN-GANS and Orthonornal Regularization 
In order to highlight the difference between our spectral normalization and orthonormal regularization, we conducted an additional set of experiments. As we explained in Section 3, orthonormal regularization is different from our method in that it destroys the spectral information and puts equal emphasis on all feature dimensions, including the ones that ’shall’ be weeded out in the training process. To see the extent of its possibly detrimental effect, we experimented by increasing the dimension of the feature space 6, especially at the final layer (7th conv) for which the training with our spectral normalization prefers relatively small feature space (dimension $<100$ ; see Figure 3b). As for the setting of the training, we selected the parameters for which the orthonormal regularization performed optimally. The figure 4 shows the result of our experiments. As we predicted, the performance of the orthonormal regularization deteriorates as we increase the dimension of the feature maps at the final layer. Our SN-GANs, on the other hand, does not falter with this modification of the architecture. Thus, at least in this perspective, we may such that our method is more robust with respect to the change of the network architecture. 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/b8f13c36b606553a44fd478d66d5db0cd948495d197ea6aad109db03a779f283.jpg) 
Figure 4: The effect on the performance on STL-10 induced by the change of the feature map dimension of the final layer. The width of the highlighted region represents standard deviation of the results over multiple seeds of weight initialization. The orthonormal regularization does not perform well with large feature map dimension, possibly because of its design that forces the discriminator to use all dimensions including the ones that are unnecessary. For the setting of the optimizers’ hyper-parameters, We used the setting C, which was optimal for “orthonormal regularization” 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/69e0a93191ea8a9a6e38221ed158813790182d57107fbcd7971e1a4514f7f46f.jpg) 
Figure 5: Learning curves for conditional image generation in terms of Inception score for SNGANs and GANs with orthonormal regularization on ImageNet. 

## 4.2 Image Generation on IMAGENET 
To show that our method remains effective on a large high dimensional dataset, we also applied our method to the training of conditional GANs on ILRSVRC2012 dataset with 1000 classes, each consisting of approximately 1300 images, which we compressed to $128\times128$ pixels. Regarding the adversarial loss for conditional GANs, we used practically the same formulation used in Mirza & Osindero (2014), except that we replaced the standard GANs loss with hinge loss (17). Please see Appendix B.3 for the details of experimental settings. 
GANs without normalization and GANs with layer normalization collapsed in the beginning of training and failed to produce any meaningful images. GANs with orthonormal normalization Brock et al. (2016) and our spectral normalization, on the other hand, was able to produce images. The inception score of the orthonormal normalization however plateaued around 20Kth iterations, while SN kept improving even afterward (Figure 5.) To our knowledge, our research is the first of its kind in succeeding to produce decent images from ImageNet dataset with a single pair of a discriminator and a generator (Figure 7). To measure the degree of mode-collapse, we followed the footstep of Odena et al. (2017) and computed the intra MS-SSIM Odena et al. (2017) for pairs of independently generated GANs images of each class. We see that our SN-GANs ((intra MS-SSIM) $_{|=0.101}$ ) is suffering less from the mode-collapse than AC-GANs ((intra MS-SSIM) ${\sim}0.25\$ ). 
To ensure that the superiority of our method is not limited within our specific setting, we also compared the performance of SN-GANs against orthonormal regularization on conditional GANs with projection discriminator (Miyato & Koyama, 2018) as well as the standard (unconditional) GANs. In our experiments, SN-GANs achieved better performance than orthonormal regularization for the both settings (See Figure 13 in the appendix section). 
# 5 Conclusion
This paper proposes spectral normalization as a stabilizer of training of GANs. When we apply spectral normalization to the GANs on image generation tasks, the generated examples are more diverse than the conventional weight normalization and achieve better or comparative inception scores relative to previous studies. The method imposes global regularization on the discriminator as opposed to local regularization introduced by WGAN-GP, and can possibly used in combinations. In the future work, we would like to further investigate where our methods stand amongst other methods on more theoretical basis, and experiment our algorithm on larger and more complex datasets. 

# A The Algorithm of Spectral Normalization
Let us describe the shortcut in Section 2.1 in more detail. We begin with vectors $\tilde{\pmb{u}}$ that is randomly initialized for each weight. If there is no multiplicity in the dominant singular values and if $\tilde{\pmb u}$ is not orthogonal to the first left singular vectors7, we can appeal to the principle of the power method and produce the first left and right singular vectors through the following update rule: 
$$
\begin{array}{r}{\tilde{{\boldsymbol{v}}}\leftarrow{\boldsymbol{W}}^{\mathrm{T}}\tilde{{\boldsymbol{u}}}/\|{\boldsymbol{W}}^{\mathrm{T}}\tilde{{\boldsymbol{u}}}\|_{2},\;\tilde{{\boldsymbol{u}}}\leftarrow{\boldsymbol{W}}\tilde{{\boldsymbol{v}}}/\|{\boldsymbol{W}}\tilde{{\boldsymbol{v}}}\|_{2}.}\end{array}
$$ 
oximate the spectral norm of $W$ with the pair of so-approximated singul 
$$
\boldsymbol{\sigma}(\boldsymbol{W})\approx\tilde{\boldsymbol{u}}^{\mathrm{T}}\boldsymbol{W}\tilde{\boldsymbol{v}}.
$$ 
If we use SGD for updating $W$ , the change in $W$ at each update would be small, and hence the change in its largest singular value. In our implementation, we took advantage of this fact and reused the $\tilde{\pmb u}$ computed at each step of the algorithm as the initial vector in the subsequent step. In fact, with this ‘recycle’ procedure, one round of power iteration was sufficient in the actual experiment to achieve satisfactory performance. Algorithm 1 in the appendix summarizes the computation of the spectrally normalized weight matrix $\bar{W}$ with this approximation. Note that this procedure is very computationally cheap even in comparison to the calculation of the forward and backward propagations on neural networks. Please see Figure 10 for actual computational time with and without spectral normalization. 

Algorithm 1 SGD with spectral normalization 
• Initialize $\tilde{\pmb{u}}_{l}\in\mathcal{R}^{d_{l}}$ for $l=1,\dots,L$ with a random vector (sampled from isotropic distribution). 
• For each update and each layer $l$ : 1. Apply power iteration method to a unnormalized weight $W^{l}$ : 
$$
\begin{array}{l}{\tilde{\pmb{v}}_{l}\leftarrow(W^{l})^{\mathrm{T}}\tilde{\pmb{u}}_{l}/\|(W^{l})^{\mathrm{T}}\tilde{\pmb{u}}_{l}\|_{2}}\\ {\tilde{\pmb{u}}_{l}\leftarrow W^{l}\tilde{\pmb{v}}_{l}/\|W^{l}\tilde{\pmb{v}}_{l}\|_{2}}\end{array}
$$ 
2. Calculate $\bar{W}_{\mathrm{SN}}$ with the spectral norm: 
$$
\bar{W}_{\mathrm{SN}}^{l}(W^{l})=W^{l}/\sigma(W^{l}),\;\mathrm{where}\;\sigma(W^{l})=\tilde{\pmb u}_{l}^{\mathrm{T}}W^{l}\tilde{\pmb v}_{l}
$$ 
3. Update $W^{l}$ with SGD on mini-batch dataset $\mathcal{D}_{M}$ with a learning rate $\alpha$ : 
$$
W^{l}\gets W^{l}-\alpha\nabla_{W^{l}}\ell(\bar{W}_{\mathrm{SN}}^{l}(W^{l}),\mathcal{D}_{M})
$$ 
# B Experimental Settings
## B.1 Performance Measures
Inception score is introduced originally by Salimans et al. (2016): $I(\{x_{n}\}_{n=1}^{N})\quad:=$ $\exp(\mathrm{E}[D_{\mathrm{KL}}[p(y|x)||p(y)]])$ , where $p(y)$ is approximated by nN=1 p(y|xn) and p(y|x) is the trained Inception convolutional neural network (Szegedy et al., 2015), which we would refer to Inception model for short. In their work, Salimans et al. (2016) reported that this score is strongly correlated with subjective human judgment of image quality. Following the procedure in Salimans et al. (2016); Warde-Farley & Bengio (2017), we calculated the score for randomly generated 5000 examples from each trained generator to evaluate its ability to generate natural images. We repeated each experiment 10 times and reported the average and the standard deviation of the inception scores. 
Fre´chet inception distance (Heusel et al., 2017) is another measure for the quality of the generated examples that uses 2nd order information of the final layer of the inception model applied to the examples. On its own, the Fre´chet distance Dowson $\&$ Landau (1982) is 2-Wasserstein distance between two distribution $p_{1}$ and $p_{2}$ assuming they are both multivariate Gaussian distributions: 
$$
F(p_{1},p_{2})=\|\pmb{\mu}_{p_{1}}-\pmb{\mu}_{p_{2}}\|_{2}^{2}+\operatorname{trace}\left(C_{p_{1}}+C_{p_{2}}-2(C_{p_{1}}C_{p_{2}})^{1/2}\right),
$$ 
where $\{\mu_{p_{1}},C_{p_{1}}\}$ , $\{\mu_{p_{2}},C_{p_{2}}\}$ are the mean and covariance of samples from $q$ and $p$ , respectively. If $f_{\ominus}$ is the output of the final layer of the inception model before the softmax, the Fre´chet inception distance (FID) between two distributions $p_{1}$ and $p_{2}$ on the images is the distance between $f_{\odot}\,{\circ}\,p_{1}$ and $f_{\ominus}\circ p_{2}$ . We computed the Fre´chet inception distance between the true distribution and the generated distribution empirically over 10000 and 5000 samples. Multiple repetition of the experiments did not exhibit any notable variations on this score. 
## B.2 Image Generation on CIFAR-10 and STL-10 
For the comparative study, we experimented with the recent ResNet architecture of Gulrajani et al. (2017) as well as the standard CNN. For this additional set of experiments, we used Adam again for the optimization and used the very hyper parameter used in Gulrajani et al. (2017) $(\alpha\,=\,0.0002,\beta_{1}\,=\,0,\beta_{2}\,=\,0.9,n_{d i s}\,=\,5)$ . For our SN-GANs, we doubled the feature map in the generator from the original, because this modification achieved better results. Note that when we doubled the dimension of the feature map for the WGAN-GP experiment, however, the performance deteriorated. 
## B.3 Image Generation on ImageNet 
The images used in this set of experiments were resized to $128\times128$ pixels. The details of the architecture are given in Table 6. For the generator network of conditional GANs, we used conditional batch normalization (CBN) (Dumoulin et al., 2017; de Vries et al., 2017). Namely we replaced the standard batch normalization layer with the CBN conditional to the label information $y\in\{1,\ldots,1000\}$ . For the optimization, we used Adam with the same hyperparameters we used for ResNet on CIFAR-10 and STL-10 dataset. We trained the networks with 450K generator updates, and applied linear decay for the learning rate after 400K iterations so that the rate would be 0 at the end. 
## B.4 Network Architectures 
Table 3: Standard CNN models for CIFAR-10 and STL-10 used in our experiments on image Generation. The slopes of all lReLU functions in the networks are set to 0.1. 
<html><body><table><tr><td>2 E R128 ~ N(0, I)</td></tr><tr><td>dense →> Mg × Mg × 512</td></tr><tr><td>4x4,stride=2deconv.BN256ReLU</td></tr><tr><td>4x4,stride=2dec0nv.BN128ReLU</td></tr><tr><td>4x4,stride=2deconv.BN64ReLU</td></tr><tr><td>3 x3, stride=1 conv. 3 Tanh</td></tr></table></body></html>
(a) Generator, $M_{g}=4$ for SVHN and CIFAR10, and $M_{g}=6$ for STL-10 
<html><body><table><tr><td></td></tr><tr><td>3x3,stride=1c0nv641ReLU 4x4,stride=2conv641ReLU</td></tr><tr><td>3×3,stride=1c0nv1281ReLU 4x4, stride=2 conv 128 1ReLU</td></tr><tr><td>3x3,stride=1c0nv2561ReLU 4x4,stride=2c0nv2561ReLU</td></tr><tr><td>3x3, stride=1 conv. 512 1ReLU</td></tr><tr><td>dense →→ 1</td></tr></table></body></html>
(b) Discriminator, $M=32$ for SVHN and CIFAR10, and $M=48$ for STL-10 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/79d0705c9bbd3a50acfc99273c3525333f6dd5807d8e7ca9d9602989df784846.jpg) 
Figure 8: ResBlock architecture. For the discriminator we removed BN layers in ResBlock. 
Table 4: ResNet architectures for CIFAR10 dataset. We use similar architectures to the ones used in Gulrajani et al. (2017). 
<html><body><table><tr><td>z E R128 ~ N(0, I)</td></tr><tr><td>dense,4 × 4 × 256</td></tr><tr><td>ResBlockup 256</td></tr><tr><td>ResBlock up 256</td></tr><tr><td>ResBlockup 256</td></tr><tr><td>BN,ReLU,3×3conv,3Tanh</td></tr></table></body></html> 
<html><body><table><tr><td>RGB image c E R 32×32x3</td></tr><tr><td>ResBlockdown128</td></tr><tr><td>ResBlockdown128</td></tr><tr><td>ResBlock128</td></tr><tr><td>ResBlock 128</td></tr><tr><td>ReLU</td></tr><tr><td>Global sum pooling</td></tr><tr><td>dense → 1 (b)Dis</td></tr></table></body></html> 
Table 5: ResNet architectures for STL-10 dataset. 
<html><body><table><tr><td>2 E R128 ~ N(0, I)</td></tr><tr><td>dense,6 x 6 × 512</td></tr><tr><td>ResBlockup 256</td></tr><tr><td>ResBlock up128</td></tr><tr><td>ResBlock up 64</td></tr><tr><td>BN,ReLU,3×3conv,3Tanh</td></tr></table></body></html> 
<html><body><table><tr><td></td></tr><tr><td>ResBlockdown64</td></tr><tr><td>ResBlockdown128</td></tr><tr><td>ResBlockdown256</td></tr><tr><td>ResBlockdown512</td></tr><tr><td>ResBlock1024</td></tr><tr><td>ReLU</td></tr><tr><td>Global sum pooling</td></tr><tr><td>dense →→ 1</td></tr></table></body></html> 
Table 6: ResNet architectures for image generation on ImageNet dataset. For the generator of conditional GANs, we replaced the usual batch normalization layer in the ResBlock with the conditional batch normalization layer. As for the model of the projection discriminator, we used the same architecture used in Miyato & Koyama (2018). Please see the paper for the details. 
<html><body><table><tr><td>z E R128 ~ N(0, I)</td></tr><tr><td>dense,4×4×1024</td></tr><tr><td>ResBlockup1024</td></tr><tr><td>ResBlock up 512</td></tr><tr><td>ResBlockup256</td></tr><tr><td>ResBlock up 128</td></tr><tr><td>ResBlock up 64</td></tr><tr><td>BN,ReLU,3x3conv3</td></tr><tr><td>Tanh</td></tr><tr><td></td></tr></table></body></html> 
<html><body><table><tr><td>RGB image c E R128×128×3</td></tr><tr><td>ResBlock down 64</td></tr><tr><td>ResBlockdown128</td></tr><tr><td>ResBlock down 256</td></tr><tr><td>ResBlock down 512</td></tr><tr><td>ResBlock down1024</td></tr><tr><td>ResBlock1024</td></tr><tr><td>ReLU</td></tr><tr><td>Global sum pooling</td></tr><tr><td>dense → 1</td></tr></table></body></html>
(b) Discriminator for unconditional GANs. 
<html><body><table><tr><td></td></tr><tr><td>ResBlockdown64</td></tr><tr><td>ResBlockdown128</td></tr><tr><td>ResBlock down 256</td></tr><tr><td>Concat(Embed(y), h)</td></tr><tr><td>ResBlock down 512</td></tr><tr><td>ResBlock down 1024</td></tr><tr><td>ResBlock 1024</td></tr><tr><td>ReLU</td></tr><tr><td>Global sum pooling</td></tr><tr><td>dense→→ 1</td></tr></table></body></html>
(c) Discriminator for conditional GANs. For computational ease, we embedded the integer label $\begin{array}{r l r}{y}&{{}\in}&{\{0,\dots,1000\}}\end{array}$ into 128 dimension before concatenating the vector to the output of the intermediate layer. 
# C APPENDIX RESULTS 
# C.1 ACCURACY OF SPECTRAL NORMALIZATION 
Figure 9 shows the spectral norm of each layer in the discriminator over the course of the training. The setting of the optimizer is C in Table 1 throughout the training. In fact, they do not deviate by more than 0.05 for the most part. As an exception, 6 and 7-th convolutional layers with largest rank deviate by more than 0.1 in the beginning of the training, but the norm of this layer too stabilizes around 1 after some iterations. 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/f7a372073005901456aee9111bcd1ab33fa24c739643040c20fa10967c5d865d.jpg) 
Figure 9: Spectral norms of all seven convolutional layers in the standard CNN during course of the training on CIFAR 10. 
# C.2 TRAINING TIME 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/7babda767ad69f784803f1778e39fcea4dfb11e4b7ec32ae2a7872f62bb3eedb.jpg) 
Figure 10: Computational time for 100 updates. We set $n_{\mathrm{dis}}=5$ 
# C.3 THE EFFECT OF $n_{d i s}$ ON SPECTRAL NORMALIZATION AND WEIGHT NORMALIZATION 
Figure 11 shows the effect of $n_{d i s}$ on the performance of weight normalization and spectral normalization. All results shown in Figure 11 follows setting D, except for the value of $n_{d i s}$ . For WN, the performance deteriorates with larger $n_{d i s}$ , which amounts to computing minimax with better accuracy. Our SN does not suffer from this unintended effect. 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/429cea8e2a715d42a41b3e2d7eb18f3c1c83c7d47331c3b13af7306bd92a9c33.jpg) 
Figure 11: The effect of $n_{d i s}$ on spectral normalization and weight normalization. The shaded region represents the variance of the result over different seeds. 
C.4 GENERATED IMAGES ON CIFAR10 WITH GAN-GP, LAYER NORMALIZATION AND BATCH NORMALIZATION 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/c81ee068cbff5ad6baf39da24c377a90f861be9293f8191872054d660802c335.jpg) 
Figure 12: Generated images with GAN-GP, Layer Norm and Batch Norm on CIFAR-10 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/22fc5c014f89b6046707d6e34131cce50dd574fa3627eaf254ea5a2369616fd0.jpg) 
C.5 IMAGE GENERATION ON IMAGENET 
Figure 13: Learning curves in terms of Inception score for SN-GANs and GANs with orthonormal regularization on ImageNet. The figure (a) shows the results for the standard (unconditional) GANs, and the figure (b) shows the results for the conditional GANs trained with projection discriminator (Miyato & Koyama, 2018) 
# D SPECTRAL NORMALIZATION VS OTHER REGULARIZATION TECHNIQUES 
This section is dedicated to the comparative study of spectral normalization and other regularization methods for discriminators. In particular, we will show that contemporary regularizations including weight normalization and weight clipping implicitly impose constraints on weight matrices that places unnecessary restriction on the search space of the discriminator. More specifically, we will show that weight normalization and weight clipping unwittingly favor low-rank weight matrices. This can force the trained discriminator to be largely dependent on select few features, rendering the algorithm to be able to match the model distribution with the target distribution only on very low dimensional feature space. 
# D.1 WEIGHT NORMALIZATION AND FROBENIUS NORMALIZATION 
The weight normalization introduced by Salimans & Kingma (2016) is a method that normalizes the $\ell_{2}$ norm of each row vector in the weight matrix8: 
$$
\begin{array}{r}{\bar{W}_{\mathrm{WN}}:=\left[\bar{w}_{1}^{\mathrm{T}},\bar{w}_{2}^{\mathrm{T}},...,\bar{w}_{d_{o}}^{\mathrm{T}}\right]^{\mathrm{T}},\ \mathrm{where}\ \bar{w}_{i}({\boldsymbol w}_{i}):={\pmb w}_{i}/\|{\pmb w}_{i}\|_{2},}\end{array}
$$ 
where $\bar{\pmb{w}}_{i}$ and $\pmb{w}_{i}$ are the ith row vector of $\bar{W}_{\mathrm{WN}}$ and $W$ , respectively. 
Still another technique to regularize the weight matrix is to use the Frobenius norm: 
$$
\bar{W}_{\mathrm{FN}}:=W/\lVert W\rVert_{F},
$$ 
Originally, these regularization techniques were invented with the goal of improving the generalization performance of supervised training (Salimans & Kingma, 2016; Arpit et al., 2016). However, recent works in the field of GANs (Salimans et al., 2016; Xiang & Li, 2017) found their another raison d’etat as a regularizer of discriminators, and succeeded in improving the performance of the original. 
8In the original literature, the weight normalization was introduced as a method for reparametrization of the form W¯WN := γ1 w¯1T , γ2 w¯2T , ..., γdo w¯dTo where $\gamma_{i}\,\in\,\mathbb{R}$ is to be learned in the course of the training. In this work, we deal with the case $\gamma_{i}=1$ so that we can assess the methods under the Lipschitz constraint. 
These methods in fact can render the trained discriminator $D$ to be $K$ -Lipschitz for a some prescribed $K$ and achieve the desired effect to a certain extent. However, weight normalization (25) imposes the following implicit restriction on the choice of $\bar{W}_{\mathrm{WN}}$ : 
$$
\sigma_{1}(\bar{W}_{\mathrm{WN}})^{2}+\sigma_{2}(\bar{W}_{\mathrm{WN}})^{2}+\cdots+\sigma_{T}(\bar{W}_{\mathrm{WN}})^{2}=d_{o},\;\mathrm{where}\;T=\operatorname*{min}(d_{i},d_{o}),
$$ 
where $\sigma_{t}(A)$ is a $t$ -th singular value of matrix $A$ . The above equation holds because $\begin{array}{r}{\sum_{t=1}^{\operatorname*{min}(d_{i},d_{o})}\sigma_{t}(\bar{W}_{\mathrm{WN}})^{2}=\mathrm{tr}(\bar{W}_{\mathrm{WN}}\bar{W}_{\mathrm{WN}}^{\mathrm{T}})=\sum_{i=1}^{d_{o}}\frac{w_{i}}{\|w_{i}\|_{2}}\frac{w_{i}^{\mathrm{T}}}{\|w_{i}\|_{2}}=d_{o}}\end{array}$ . Under this restriction, the n√orm $\lVert\bar{W}_{\mathrm{WN}}h\rVert_{2}$ for a fixed unit vector $^h$ is maximized at $\|\bar{W}_{\mathrm{WN}}h\|_{2}=\sqrt{d_{o}}$ when $\sigma_{1}(\bar{W}_{\mathrm{WN}})=$ $\sqrt{d_{o}}$ and $\sigma_{t}([\bar{W}_{\mathrm{WN}})\,=\,0$ for $t\,=\,2,\ldots,T$ , which means that $\bar{W}_{\mathrm{WN}}$ is of rank one. Using such $W$ corresponds to using only one feature to discriminate the model probability distribution from the target. Similarly, Frobenius normalization requires $\sigma_{1}(\bar{W}_{\mathrm{FN}})^{2}\!+\!\sigma_{2}(\bar{W}_{\mathrm{FN}})^{2}\!+\!\cdot\!\cdot\!\cdot\!+\!\sigma_{T}(\bar{W}_{\mathrm{FN}})^{2}=1.$ , and the same argument as above follows. 
Here, we see a critical problem in these two regularization methods. In order to retain as much norm of the input as possible and hence to make the discriminator more sensitive, one would hope to make the norm of $\bar{W}_{\mathrm{WN}}h$ large. For weight normalization, however, this comes at the cost of reducing the rank and hence the number of features to be used for the discriminator. Thus, there is a conflict of interests between weight normalization and our desire to use as many features as possible to distinguish the generator distribution from the target distribution. The former interest often reigns over the other in many cases, inadvertently diminishing the number of features to be used by the discriminators. Consequently, the algorithm would produce a rather arbitrary model distribution that matches the target distribution only at select few features. 
Our spectral normalization, on the other hand, do not suffer from such a conflict in interest. Note that the Lipschitz constant of a linear operator is determined only by the maximum singular value. In other words, the spectral norm is independent of rank. Thus, unlike the weight normalization, our spectral normalization allows the parameter matrix to use as many features as possible while satisfying local 1-Lipschitz constraint. Our spectral normalization leaves more freedom in choosing the number of singular components (features) to feed to the next layer of the discriminator. 
To see this more visually, we refer the reader to Figure (14). Note that spectral normalization allows for a wider range of choices than weight normalization. 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/31c405b7487f147ca4720fae3ba96cd09491fbe01cad74faf1b3a06d56111cac.jpg) 
Figure 14: Visualization of the difference between spectral normalization (Red) and weight normalization (Blue) on possible sets of singular values. The possible sets of singular values plotted in increasing order for weight normalization (Blue) and for spectral normalization (Red). For the√ set of singular values permitted under the spectral normalization condition, we scaled $\bar{W}_{\mathrm{WN}}$ by $1/\sqrt{d_{o}}$ so that its spectral norm is exactly 1. By the definition of the weight normalization, the area under the blue curves are all bound to be 1. Note that the range of choice for the weight normalization is small. 
In summary, weight normalization and Frobenius normalization favor skewed distributions of singular values, making the column spaces of the weight matrices lie in (approximately) low dimensional vector spaces. On the other hand, our spectral normalization does not compromise the number of feature dimensions used by the discriminator. In fact, we will experimentally show that GANs trained with our spectral normalization can generate a synthetic dataset with wider variety and higher inception score than the GANs trained with other two regularization methods. 
# D.2 WEIGHT CLIPPING 
Still another regularization technique is weight clipping introduced by Arjovsky et al. (2017) in their training of Wasserstein GANs. Weight clipping simply truncates each element of weight matrices so that its absolute value is bounded above by a prescribed constant $c\in\mathbb{R}_{+}$ . Unfortunately, weight clipping suffers from the same problem as weight normalization and Frobenius normalization. With weight clipping with the truncation value $c$ , the value $\|W\mathbf{x}\|_{2}$ for a fixed unit vector $\textbf{\em x}$ is maximized when the rank of $W$ is again one, and the training will again favor the discriminators that use only select few features. Gulrajani et al. (2017) refers to this problem as capacity underuse problem. They also reported that the training of WGAN with weight clipping is slower than that of the original DCGAN (Radford et al., 2016). 
D.3 SINGULAR VALUE CLIPPING AND SINGULAR VALUE CONSTRA 
One direct and straightforward way of controlling the spectral norm is to clip the singular values (Saito et al., 2017), (Jia et al., 2017). This approach, however, is computationally heavy because one needs to implement singular value decomposition in order to compute all the singular values. 
A similar but less obvious approach is to parametrize $W\in\mathbb{R}^{d_{o}\times d_{i}}$ as follows from the get-go and train the discriminators with this constrained parametrization: 
$$
W:=U S V^{\mathrm{{T}}},\ \ \mathrm{subject\to}\ U^{\mathrm{{T}}}U=I,V^{\mathrm{{T}}}V=I,\ \operatorname*{max}_{i}S_{i i}=K,
$$ 
where $U\in\mathbb{R}^{d_{o}\times P}$ , $V\in\mathbb{R}^{d_{i}\times P}$ , and $S\in\mathbb{R}^{P\times P}$ is a diagonal matrix. However, it is not a simple task to train this model while remaining absolutely faithful to this parametrization constraint. Our spectral normalization, on the other hand, can carry out the updates with relatively low computational cost without compromising the normalization constraint. 
# D.4 WGAN WITH GRADIENT PENALTY (WGAN-GP) 
Recently, Gulrajani et al. (2017) introduced a technique to enhance the stability of the training of Wasserstein GANs (Arjovsky et al., 2017). In their work, they endeavored to place $K$ -Lipschitz constraint (5) on the discriminator by augmenting the adversarial loss function with the following regularizer function: 
$$
\lambda\operatorname{\underset{\hat{\pmb{x}}\sim p_{\hat{\pmb{x}}}}{\mathrm{E}}}[(\|\nabla_{\hat{\pmb{x}}}D(\hat{\pmb{x}})\|_{2}-1)^{2}],
$$ 
where $\lambda>0$ is a balancing coefficient and $\hat{\pmb x}$ is: 
$$
\begin{array}{r}{\hat{\pmb{x}}:=\epsilon\pmb{x}+(1-\epsilon)\tilde{\pmb{x}}}\\ {\mathrm{where}\;\epsilon\sim U[0,1],\;\pmb{x}\sim p_{\mathrm{data}},\;\tilde{\pmb{x}}=G(z),\;z\sim p_{z}.}\end{array}
$$ 
Using this augmented objective function, Gulrajani et al. (2017) succeeded in training a GAN based on ResNet (He et al., 2016) with an impressive performance. The advantage of their method in comparison to spectral normalization is that they can impose local 1-Lipschitz constraint directly on the discriminator function without a rather round-about layer-wise normalization. This suggest that their method is less likely to underuse the capacity of the network structure. 
At the same time, this type of method that penalizes the gradients at sample points $\hat{\pmb{x}}$ suffers from the obvious problem of not being able to regularize the function at the points outside of the support of the current generative distribution. In fact, the generative distribution and its support gradually changes in the course of the training, and this can destabilize the effect of the regularization itself. 
On the contrary, our spectral normalization regularizes the function itself, and the effect of the regularization is more stable with respect to the choice of the batch. In fact, we observed in the experiment that a high learning rate can destabilize the performance of WGAN-GP. Training with our spectral normalization does not falter with aggressive learning rate. 
Moreover, WGAN-GP requires more computational cost than our spectral normalization with single-step power iteration, because the computation of $\|\nabla_{\pmb{x}}D\|_{2}$ requires one whole round of forward and backward propagation. In Figure 10, we compare the computational cost of the two methods for the same number of updates. 
Having said that, one shall not rule out the possibility that the gradient penalty can compliment spectral normalization and vice versa. Because these two methods regularizes discriminators by completely different means, and in the experiment section, we actually confirmed that combination of WGAN-GP and reparametrization with spectral normalization improves the quality of the generated examples over the baseline (WGAN-GP only). 
# E REPARAMETRIZATION MOTIVATED BY THE SPECTRAL NORMALIZATION 
We can take advantage of the regularization effect of the spectral normalization we saw above to develop another algorithm. Let us consider another parametrization of the weight matrix of the discriminator given by: 
$$
\tilde{W}:=\gamma\bar{W}_{\mathrm{SN}}
$$ 
where $\gamma$ is a scalar variable to be learned. This parametrization compromises the 1-Lipschitz constraint at the layer of interest, but gives more freedom to the model while keeping the model from becoming degenerate. For this reparametrization, we need to control the Lipschitz condition by other means, such as the gradient penalty (Gulrajani et al., 2017). Indeed, we can think of analogous versions of reparametrization by replacing $\bar{W}_{\mathrm{SN}}$ in (32) with $W$ normalized by other criterions. The extension of this form is not new. In Salimans & Kingma (2016), they originally introduced weight normalization in order to derive the reparametrization of the form (32) with $\bar{W_{\mathrm{SN}}}$ replaced (32) by $W_{\mathrm{WN}}$ and vectorized $\gamma$ . 
E.1 EXPERIMENTS: COMPARISON OF REPARAMETRIZATION WITH DIFFERENTNORMALIZATION METHODS 
In this part of the addendum, we experimentally compare the reparametrizations derived from two different normalization methods (weight normalization and spectral normalization). We tested the reprametrization methods for the training of the discriminator of WGAN-GP. For the architecture of the network in WGAN-GP, we used the same CNN we used in the previous section. For the ResNet-based CNN, we used the same architecture provided by (Gulrajani et al., 2017) 9. 
Tables 7, 8 summarize the result. We see that our method significantly improves the inception score from the baseline on the regular CNN, and slightly improves the score on the ResNet based CNN. 
Figure 15 shows the learning curves of (a) critic losses, on train and validation sets and (b) the inception scores with different reparametrization methods. We can see the beneficial effect of spectral normalization in the learning curve of the discriminator as well. We can verify in the figure 15a that the discriminator with spectral normalization overfits less to the training dataset than the discriminator without reparametrization and with weight normalization, The effect of overfitting can be observed on inception score as well, and the final score with spectral normalization is better than the others. As for the best inception score achieved in the course of the training, spectral normalization achieved 7.28, whereas the spectral normalization and vanilla normalization achieved 7.04 and 6.69, respectively. 
# F THE GRADIENT OF GENERAL NORMALIZATION METHOD 
Let us denote $\bar{W}:=W/N(W)$ to be the normalized weight where $N(W)$ to be a scalar normalized coefficient (e.g. Spectral norm or Frobenius norm). In general, we can write the derivative of loss 
<html><body><table><tr><td>Method</td><td>Inception score</td><td>FID</td></tr><tr><td>WGAN-GP (Standard CNN, Baseline) w/FrobeniusNorm.</td><td>6.68±.06 N/A*</td><td>40.1 N/A*</td></tr><tr><td>w/Weight Norm.</td><td>6.36±.04</td><td>42.4</td></tr><tr><td>w/ Spectral Norm.</td><td>7.20±.08</td><td>32.0</td></tr><tr><td>(WGAN-GP, ResNet, Gulrajani et al. (2017))</td><td>7.86±.08</td><td></td></tr><tr><td>WGAN-GP (ResNet,Baseline)</td><td>7.80±.11</td><td>24.5</td></tr><tr><td>w/ Spectral norm.</td><td>7.85±.06</td><td>23.6</td></tr><tr><td>w/ Spectral norm. (1.5x feature maps in D)</td><td>7.96±.06</td><td>22.5</td></tr></table></body></html> 
Table 7: Inception scores with different reparametrization mehtods on CIFAR10 without label supervisions. $(\mathrm{^{*})W e}$ reported N/A for the inception score and FID of Frobenius normalization because the training collapsed at the early stage. 
<html><body><table><tr><td>Method (ResNet)</td><td>Inceptionscore</td><td>FID</td></tr><tr><td>(AC-WGAN-GP, Gulrajani et al. (2017))</td><td>8.42±.10</td><td></td></tr><tr><td>AC-WGAN-GP (Baseline)</td><td>8.29±.12</td><td>19.5</td></tr><tr><td>w/ Spectral norm.</td><td>8.59±.12</td><td>18.6</td></tr><tr><td>w/ Spectral norm. (1.5x feature maps in D)</td><td>8.60±.08</td><td>17.5</td></tr></table></body></html> 
Table 8: Inception scores and FIDs with different reparametrization methods on CIFAR10 with the label supervision, by auxiliary classifier (Odena et al., 2017). 
with respect to unnormalized weight $W$ as follows: 
$$
\begin{array}{r l}&{\displaystyle\frac{\partial V(G,D(W))}{\partial W}=\frac{1}{N(W)}\left(\frac{\partial V}{\partial\bar{W}}-\mathrm{trace}\left(\left(\frac{\partial V}{\partial\bar{W}}\right)^{\mathrm{T}}\bar{W}\right)\frac{\partial(N(W))}{\partial W}\right)}\\ &{\displaystyle~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}\\ &{\displaystyle~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}\\ &{=\frac{1}{N(W)}\left(\nabla_{\bar{W}}V-\mathrm{trace}\left((\nabla_{\bar{W}}V)^{\mathrm{T}}\bar{W}\right)\nabla_{W}N\right)}\\ &{=\alpha\left(\nabla_{\bar{W}}V-\lambda\nabla_{W}N\right),}\end{array}
$$ 
where $\alpha:=1/N(W)$ and $\lambda:=\mathrm{trace}\left((\nabla_{\bar{W}}V)^{\mathrm{T}}\bar{W}\right)$ . The gradient $\nabla_{\boldsymbol{\Bar{W}}}V$ is calculated by $\hat{\mathrm{~E~}}\big[\delta h^{\mathrm{T}}\big]$ where $\pmb{\delta}:=\left(\partial V(\pmb{G},D)/\partial\left(\bar{W}\pmb{h}\right)\right)^{\mathrm{T}}$ , $^h$ is the hidden node in the network to be transformed by $\Bar{W}$ and $\hat{\mathrm{E}}$ represents empirical expectation over the mini-batch. When $N(W):=\|W\|_{F}$ , the derivative is: 
$$
\frac{\partial V(G,D(W))}{\partial W}=\frac{1}{\|W\|_{F}}\left(\hat{\mathrm{E}}\left[\delta h^{\mathrm{T}}\right]-\mathrm{trace}\left(\hat{\mathrm{E}}\left[\delta h^{\mathrm{T}}\right]^{\mathrm{T}}\Bar{W}\right)\Bar{W}\right),
$$ 
and when $N(W):=\|W\|_{2}=\sigma(W)$ , 
$$
\frac{\partial V(G,D(W))}{\partial W}=\frac{1}{\sigma(W)}\left(\hat{\mathrm{E}}\left[\delta h^{\mathrm{T}}\right]-\mathrm{trace}\left(\hat{\mathrm{E}}\left[\delta h^{\mathrm{T}}\right]^{\mathrm{T}}\Bar{W}\right)u_{1}v_{1}^{\mathrm{T}}\right).
$$ 
Notice that, at least for the case $N(W):=\|W\|_{F}$ or $N(W):=\|W\|_{2}$ , the point of this gradient is given by : 
$$
\nabla_{\boldsymbol{\bar{W}}}V=k\nabla_{\boldsymbol{W}}N.
$$ 
where $\exists\,k\in\mathbb{R}$ 
![](https://cdn-mineru.openxlab.org.cn/extract/23d6439c-9d99-452c-9a35-268c9379ad69/8854080c0b24cd5f4b36516ccbbee8739ae8c25017154c8f682681d39f526ef6.jpg) 
Figure 15: Learning curves of (a) critic loss and (b) inception score on different reparametrization method on CIFAR-10 ; weight normalization (WGAN-GP w/ WN), spectral normalization (WGANGP w/ SN), and parametrization free (WGAN-GP). 

# Supplementary Knowledge
## Matrix norm
In the field of [mathematics](https://en.wikipedia.org/wiki/Mathematics "Mathematics"), [norms](https://en.wikipedia.org/wiki/Vector_norm "Vector norm") are defined for elements within a [vector space](https://en.wikipedia.org/wiki/Vector_space "Vector space"). Specifically, when the vector space comprises matrices, such norms are referred to as **matrix norms**. Matrix norms differ from vector norms in that they must also interact with matrix multiplication.
>  数学中，范数为向量空间中的元素而定义，当这些元素是矩阵时，范数就是矩阵范数
>  矩阵范数和向量范数的差异在于矩阵范数和矩阵乘法相关

### Preliminaries
Given a [field](https://en.wikipedia.org/wiki/Field_(mathematics) "Field (mathematics)") $K$ of either [real](https://en.wikipedia.org/wiki/Real_number "Real number") or [complex numbers](https://en.wikipedia.org/wiki/Complex_number "Complex number") (or any complete subset thereof), let $K^{m\times n}$ be the K-[vector space](https://en.wikipedia.org/wiki/Vector_space "Vector space") of matrices with $m$ rows and $n$ columns and entries in the field $K$ . A matrix norm is a [norm](https://en.wikipedia.org/wiki/Norm_(mathematics) "Norm (mathematics)") on $K^{m\times n}$.


Norms are often expressed with [double vertical bars](https://en.wikipedia.org/wiki/Double_vertical_bar "Double vertical bar") (like so: $\|A\|$ ). Thus, the matrix norm is a [function](https://en.wikipedia.org/wiki/Function_(mathematics) "Function (mathematics)") $\|\cdot\|:K^{m\times n}\to \mathbb R^{0+}$ that must satisfy the following properties:

>  给定一个由实数或复数 (或其任何完备子集) 构成的域 $K$，令 $K^{m\times n}$ 表示 $m$ 行 $n$ 列，元素都来自于 $K$ 的矩阵组成的 $K$ - 向量空间。矩阵范数是定义在 $K^{m\times n}$ 上的范数，其本质是一个函数 $\|\cdot\|: K^{m\times n} \to \mathbb R^{0+}$，满足以下的性质：

For all scalars $\alpha\in K$ and matrices $A, B\in K^{m\times n}$,
- $\|A\| \ge 0$ (positive-valued)
- $\|A\| = 0 \iff  A=0_{m, n}$ (definite)
- $\|\alpha A\| = |\alpha|\cdot \|A\|$ (absolutely homogeneous)
- $\|A+B\| \le \|A\| + \|B\|$ (sub-additive or satisfying the triangle inequality)

> 对于标量 $\alpha \in K$ 和矩阵 $A, B\in K^{m\times n}$，矩阵范数满足：
> - $\|A\|\ge 0$ (值为正)
> - $\|A\| = 0 \iff  A=0_{m, n}$ (确定性，即当且仅当 $A$ 为全零矩阵时，范数才取到一)
> - $\|\alpha A\| = |\alpha| \cdot \|A\|$ (绝对齐次性)
> - $\|A+B\| \le \|A\| + \|B\|$ (次可加性/三角不等式)

The only feature distinguishing matrices from rearranged vectors is [multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication "Matrix multiplication"). 

Matrix norms are particularly useful if they are also **sub-multiplicative**:

$$
\|AB\|\le \|A\|\cdot\|B\|
$$

Every norm on $K^{n\times n}$ can be rescaled to be sub-multiplicative; in some books, the terminology _matrix norm_ is reserved for sub-multiplicative norms.

>  若矩阵范数还满足以上式子，称范数满足子乘性
>  $K^{n\times n}$ 中的任意范数都可以被缩放至满足子乘性

### Matrix norms induced by vector norms  
Suppose a vector norm $||\cdot||_{\alpha}$ on $K^{n}$ and a vector norm $\|\cdot\|_{\beta}$ on $K^{m}$ are given. Any $m\times n$ matrix $A$ induces a linear operator from $K^{n}$ to $K^{m}$ with respect to the standard basis, and one defines the corresponding induced norm or operator norm or subordinate norm on the space $K^{m\times n}$ of all $m\times n$ matrices as follows:  
>  给定 $K^n$ 上的向量范数 $\|\cdot \|_\alpha$ 和 $K^m$ 上的向量范数 $\|\cdot \|_\beta$
>  任意 $m\times n$ 的矩阵 $A$ 都导出了一个相对于标准基的从 $K^n$ 到 $K^m$ 的线性算子
>  我们为空间 $K^{m\times n}$ 中的 $m\times n$ 矩阵定义诱导范数/算子范数/从属范数如下：

$$
\begin{align}
\|A\|_{\alpha, \beta} &=\sup\{\|Ax\|_\beta:x\in K^n\ \mathrm{with}\ \|x\|_\alpha=1\}\\
&=\sup\{\frac {\|Ax\|_\beta}{\|x\|_\alpha}:x\in K^n\ \mathrm{with}\ x\ne 0_n\}
\end{align}
$$

where $\sup$ denotes the supremum. 
>  其中 $\sup$ 表示上确界 (在有限维空间中，$\sup$ 等价于 $\max$)

This norm measures how much the mapping induced by $A$ can stretch vectors. Depending on the vector norms $\|\cdot\|_{\alpha},\|\cdot\|_{\beta}$ used, notation other than $\|\cdot\|_{\alpha,\beta}$ can be used for the operator norm.  
>  该范数度量了由 $A$ 诱导的映射对向量的拉伸程度
>  该范数依赖于所使用的向量范数 $\|\cdot \|_\alpha, \|\cdot\|_\beta$

### Matrix norms induced by vector $p$ -norms  
If the $p$ -norm for vectors $(1\leq p\leq\infty)$ is used for both spaces $K^{n}$ and $K^{m}$ then the corresponding operator norm is:  
>  如果空间 $K^n$ 和 $K^m$ 都使用了向量的 $p$ 范数 ($1\le p \le \infty$)，则对应的算子范数定义如下

$$
\|A\|_{p}=\operatorname*{sup}_{x\neq0}{\frac{\|A x\|_{p}}{\|x\|_{p}}}.
$$  
These induced norms are different from the "entry-wise" $p$ -norms and the Schatten $p$ -norms for matrices treated below, which are also usually denoted by $\|A\|_{p}$  
>  注意这里和之前介绍的这类诱导范数和之后定义的 “按元素” 的 $p$ 范数和 Schatten $p$ 范数是不同的，虽然它们也常常记作 $\|A\|_p$

Geometrically speaking, one can imagine a $p$ -norm unit ball $V_{p,n}=\{x\in K^{n}:\|x\|_{p}\leq1\}$ in $K^{n}$ , then apply the linear map $A$ to the ball. It would end up becoming a distorted convex shape $A V_{p,n}\subset K^{m}$ , and $\|A\|_{p}$ measures the longest "radius" of the distorted convex shape. In other words, we must take a $p$ -norm unit ball $V_{p,m}$ in $K^{m}$ , then multiply it by at least $\|A\|_{p}$ , in order for it to be large enough to contain $A V_{p,n}$  
>  几何上看，可以想象一个 $K^n$ 中 $p$ 范数定义的单位球 $V_{p, n} = \{x\in K^n: \|x\|_p \le 1\}$，然后为该球应用线性映射 $A$，该球会被映射为扭曲的凸集 $AV_{p, n}\subset K^m$。范数 $\|A\|_p$ 度量了该扭曲凸集的最长 “半径”
>  换句话说，需要将 $K^m$ 中的 $p$ 范数球 $V_{p, m}$ 至少放大 $\|A\|_p$ 被，才能包含住 $AV_{p, n}$

**$p=1$ or $\infty$** 
When $p=1$ or $p=\infty$ we have simple formulas.  

$$
\|A\|_{1}=\operatorname*{max}_{1\leq j\leq n}\sum_{i=1}^{m}\left|a_{i j}\right|,
$$

which is simply the maximum absolute column sum of the matrix.  

>  当 $p=1$ 时，$\|A\|_1$ 的形式如上，它等于矩阵的最大绝对列和

$$
\|A\|_{\infty}=\operatorname*{max}_{1\leq i\leq m}\sum_{j=1}^{n}\left|a_{i j}\right|,
$$

which is simply the maximum absolute row sum of the matrix.  

>  当 $p = \infty$ 时，$\|A\|_\infty$ 的形式如上，它等于矩阵的最大绝对行和


**Spectral norm $(p=2)$**  
When $p=2$ (the Euclidean norm or $\ell_{2}$ -norm for vectors), the induced matrix norm is the spectral norm. 
>  当 $p=2$ (向量的欧几里得范数/ $\ell_2$ 范数)，诱导的矩阵范数称为谱范数

The two values do not coincide in infinite dimensions — see Spectral radius for further discussion. The spectral radius should not be confused with the spectral norm. 
>  无限维空间中，谱范数和谱半径不相等

The spectral norm of a matrix $A$ is the largest singular value of $A$ , i.e., the square root of the largest eigenvalue of the matrix $A^{*}A$ where $A^{*}$ denotes the conjugate transpose of $A$ : 

$$
\|A\|_{2}=\sqrt{\lambda_{\operatorname*{max}}\left(A^{*}A\right)}=\sigma_{\operatorname*{max}}(A).
$$  
where $\sigma_{\mathrm{max}}(A)$ represents the largest singular value of matrix $A$  

>  矩阵 $A$ 的谱范数 $\|A\|_2$ 是 $A$ 的最大奇异值，即矩阵 $A^*A$ 的最大特征值的平方根 (其中 $A^*$ 是 $A$ 的共轭转置)，定义如上，可以记作 $\sigma_{\max}(A)$

There are further properties:  

-  $\|A\|_{2}=\operatorname*{sup}\{x^{*}A y:x\in K^{m},y\in K^{n}$ with $\|x\|_{2}=\|y\|_{2}=1\}$ Proved by the Cauchy– Schwarz inequality.
-  $\|A^{*}A\|_{2}=\|A A^{*}\|_{2}=\|A\|_{2}^{2}$ . Proven by singular value decomposition (SVD) on .
- $\begin{array}{r}{\|A\|_{2}=\sigma_{\mathrm{max}}(A)\leq\|A\|_{\mathrm{F}}=\sqrt{\sum_{i}\sigma_{i}(A)^{2}}}\end{array}$ , where $\|A\|_{\mathrm{F}}$ is the Frobenius norm. Equality holds if and only if the matrix $A$ is a rank-one matrix or a zero matrix. 
- Conversely, $\|A\|_{\mathrm{F}}\leq\operatorname*{min}(m,n)^{1/2}\|A\|_{2}$ . 
- $\begin{array}{r}{\|A\|_{2}=\sqrt{\rho(A^{*}A)}\leq\sqrt{\|A^{*}A\|_{\infty}}\leq\sqrt{\|A\|_{1}\|A\|_{\infty}}.}\end{array}$  

### "Entry-wise" matrix norms  
These norms treat an $m\times n$ matrix as a vector of size $m\cdot n$ , and use one of the familiar vector norms. For example, using the $p$ -norm for vectors, $p\geq1$ , we get:  

$$
\|A\|_{p}=\|\mathrm{vec}(A)\|_{p}=\left(\sum_{i=1}^{m}\sum_{j=1}^{n}\left|a_{i j}\right|^{p}\right)^{1/p}
$$

This is a different norm from the induced $p$ -norm (see above) and the Schatten $p$ -norm (see below), but the notation is the same.  

>  “按元素”的矩阵范数将矩阵拉平为 $m\cdot n$ 的向量，然后应用向量范数，此时，矩阵的 $p$ 范数定义如上
>  该范数和诱导的 $p$ 范数以及 Schatten $p$ 范数不同，但符号相同

The special case $p=2$ is the Frobenius norm, and $p=\infty$ yields the maximum norm.  

$$
\|A\|_{\mathrm{F}}={\sqrt{\sum_{i}^{m}\sum_{j}^{n}|a_{i j}|^{2}}}={\sqrt{\operatorname{trace}(A^{*}A)}}={\sqrt{\sum_{i=1}^{\operatorname*{min}\{m,n\}}\sigma_{i}^{2}(A)}},
$$

$$
\|A\|_{\operatorname*{max}}=\operatorname*{max}_{i,j}|a_{i j}|.
$$  
>  $p=2$ 以及 $p = \infty$ 时分别得到 F 范数和最大范数

### Equivalence of norms  
For any two matrix norms $||\cdot||_{\alpha}$ and $\|\cdot\|_{\beta}$ , we have that:  

$$
r\|A\|_{\alpha}\leq\|A\|_{\beta}\leq s\|A\|_{\alpha}
$$  
for some positive numbers $r$ and $s$ , for all matrices $A\in K^{m\times n}$ . 

>  对于任意两个矩阵范数 $\|\cdot \|_\alpha$ 和 $\|\cdot \|_\beta$，我们都有如上式子对于所有矩阵 $A\in K^{m\times n}$ 成立，其中 $r, s$ 为某些正数

In other words, all norms on $K^{m\times n}$ are equivalent; they induce the same topology on $K^{m\times n}$ . This is true because the vector space $K^{m\times n}$ has the finite dimension $m\times n$ .  
>  换句话说，所有 $K^{m\times n}$ 上的范数都等价，它们在 $K^{m\times n}$ 诱导出相同的拓扑结构

**Examples of norm equivalence**  
Let $\|A\|_{p}$ once again refer to the norm induced by the vector $p$ -norm (as above in the Induced norm section).  

For matrix $A\in\mathbb{R}^{m\times n}$ of rank $r$ , the following inequalities hold:

- ${\|A\|_{2}\leq\|A\|_{F}\leq\sqrt{r}\|A\|_{2}}$
- ${\|A\|_{F}\leq\|A\|_{*}\leq\sqrt{r}\|A\|_{F}}$
- ${\|A\|_{\operatorname*{max}}\leq\|A\|_{2}\leq\sqrt{m n}\|A\|_{\operatorname*{max}}}$
- ${\frac{1}{\sqrt{n}}\|A\|_{\infty}\leq\|A\|_{2}\leq\sqrt{m}\|A\|_{\infty}}$
- ${\frac{1}{\sqrt{m}}\|A\|_{1}\leq\|A\|_{2}\leq\sqrt{n}\|A\|_{1}.}$

## Proof of Inequality
不等式：对于任意向量 $\pmb h$，有 $\|A\pmb h\|_2 \le \sigma(A)\|\pmb h\|_2$

证明：
根据矩阵的谱范数的定义，我们知道

$$
\sigma(A) = \max_{\pmb x\ne 0}\frac {\|A\pmb x\|_2}{\|\pmb x\|_2}
$$

因此，对于任意非零向量 $\pmb x$，都有不等式

$$
\begin{align}
\sigma(A)&\ge \frac {\|A\pmb x\|_2}{\|\pmb x\|_2}\\
\sigma(A)\|\pmb x\|_2 &\ge \|A\pmb x\|_2
\end{align}
$$

成立

再考虑全零向量 $\pmb 0$，容易知道此时不等式仍然成立 (取等号)

证毕
