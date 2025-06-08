Site: https://yang-song.net/blog/2021/score/
Date: 5 May 2021

This blog post focuses on a promising new direction for generative modeling. We can learn score functions (gradients of log probability density functions) on a large number of noise-perturbed data distributions, then generate samples with Langevin-type sampling. The resulting generative models, often called _score-based generative models_, has several important advantages over existing model families: GAN-level sample quality without adversarial training, flexible model architectures, exact log-likelihood computation, and inverse problem solving without re-training models. In this blog post, we will show you in more detail the intuition, basic concepts, and potential applications of score-based generative models.
> 本文为生成式建模提出新的思路
> 我们可以在大量的添加噪声的数据分布上学习得分函数 (对数概率密度函数的梯度)，然后使用 Langevin 型采样来生成样本
> 基于该范式得到的生成时模型称为 score-based 生成式模型，其优势包括:
> - 无需对抗训练就获得 GAN 相当的样本质量
> - 灵活的模型架构
> - 精确的对数似然计算
> - 无需重新训练模型即可解决逆问题

## Introduction
Existing generative modeling techniques can largely be grouped into two categories based on how they represent probability distributions.
> 基于如何表示概率分布，现存的生成式建模可以分为两类

1. **likelihood-based models**, which directly learn the distribution’s probability density (or mass) function via (approximate) maximum likelihood. Typical likelihood-based models include autoregressive models [1, 2, 3] , normalizing flow models [4, 5] , energy-based models (EBMs) [6, 7] , and variational auto-encoders (VAEs) [8, 9] .
2. **implicit generative models** [10] , where the probability distribution is implicitly represented by a model of its sampling process. The most prominent example is generative adversarial networks (GANs) [11] , where new samples from the data distribution are synthesized by transforming a random Gaussian vector with a neural network.

> **基于似然的模型**
> 通过 (近似) 极大似然，**直接**学习分布的概率密度 (或质量) 函数
> 典型的基于似然的模型包括自回归模型、规范化流模型、基于能量的模型、变分自动编码机
> **隐式生成式模型**
> 概率分布被隐式表征为对该分布进行采样的采样过程
> 典型的例子是 GAN，它通过神经网络将随机高斯向量转化为服从数据分布的样本

![](https://yang-song.net/assets/img/score/likelihood_based_models.png)

Bayesian networks, Markov random fields (MRF), autoregressive models, and normalizing flow models are all examples of likelihood-based models. All these models represent the probability density or mass function of a distribution.
> 贝叶斯网络、Markov 随机场、自回归模型、规范化流模型都是基于似然的模型
> 这些模型直接建模和表征一个分布的概率质量或密度函数

![](https://yang-song.net/assets/img/score/implicit_models.png)

GAN is an example of implicit models. It implicitly represents a distribution over all objects that can be produced by the generator network.
> GAN 属于隐式模型，它所隐式表征的分布即它的生成器网络生成的所有样本所服从的分布

Likelihood-based models and implicit generative models, however, both have significant limitations. Likelihood-based models either require strong restrictions on the model architecture to ensure a tractable normalizing constant for likelihood computation, or must rely on surrogate objectives to approximate maximum likelihood training. Implicit generative models, on the other hand, often require adversarial training, which is notoriously unstable [12] and can lead to mode collapse [13] .
>   两类模型都存在明显缺陷
>   基于似然的模型要么对模型结构有严格限制以确保似然计算中的规范化常数是可解的 (否则对所有情况积分来求 partition function 是不可解的)，要么必须依赖替代目标来近似极大似然训练 (例如变分推断)
>   隐式生成模型通常需要对抗性训练，不稳定，且容易出现模式崩溃

>   基于似然的模型的缺陷是自然的，因为目标分布本就是复杂的，故将它作为学习目标，为了能够学习，总会进行 compromising
>   隐式生成模型则将学习目标设为高斯分布到目标分布的映射，这减少了复杂性，但这样间接的学习显然也存在问题

In this blog post, I will introduce another way to represent probability distributions that may circumvent several of these limitations. The key idea is to model _the gradient of the log probability density function_, a quantity often known as the (Stein) **score function** [14, 15] . Such **score-based models** are not required to have a tractable normalizing constant, and can be directly learned by **score matching** [16, 17] .
>   本文介绍另一种表示概率分布的方法，以避免上述限制
>   其核心思想是建模对数概率密度函数的梯度 (也是一种间接学习)，通常该量被称为 (Stein) 得分函数
>   基于得分的模型不需要具有可解的归一化常数，可以直接通过得分匹配学习

![](https://yang-song.net/assets/img/score/score_contour.jpg)

Score function (the vector field) and density function (contours) of a mixture of two Gaussians.
>  图中是混合高斯分布的得分函数 (向量场) 和密度函数 (登高线) 示意
>  等高线密集的地方，向量场更大 (变化率大 -> 梯度大)

Score-based models have achieved state-of-the-art performance on many downstream tasks and applications. These tasks include, among others, image generation [18, 19, 20, 21, 22, 23] (Yes, better than GANs!), audio synthesis [24, 25, 26] , shape generation [27] , and music generation [28] . Moreover, score-based models have connections to [normalizing flow models](https://blog.evjang.com/2018/01/nf1.html), therefore allowing exact likelihood computation and representation learning. Additionally, modeling and estimating scores facilitates [inverse problem](https://en.wikipedia.org/wiki/Inverse_problem#:~:text=An%20inverse%20problem%20in%20science,measurements%20of%20its%20gravity%20field) solving, with applications such as image inpainting [18, 21] , image colorization [21] , [compressive sensing](https://en.wikipedia.org/wiki/Compressed_sensing), and medical image reconstruction (e.g., CT, MRI) [29] .
>  基于得分的模型在多项下流工作和任务上已经取得了 SOTA
>  这些任务包括: 图像生成、声音合成、形状生成、音乐生成 (不得不感叹概率建模的泛用性太广了)
>  此外，基于得分的模型还和规范化流模型相关，因此可以精确计算似然和学习表征
>  此外，建模和估计得分有助于解决逆问题，其应用包括图像修复、图像着色、压缩感知、医学影像重建 (不得不再次感叹概率建模的泛用性太广了)

![](https://yang-song.net/assets/img/score/ffhq_samples.jpg)

1024 x 1024 samples generated from score-based models [21]

This post aims to show you the motivation and intuition of score-based generative modeling, as well as its basic concepts, properties and applications.

## The score function, score-based models, and score matching
Suppose we are given a dataset $\{\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_N\}$, where each point is drawn independently from an underlying data distribution $p(\mathbf x)$. Given this dataset, the goal of generative modeling is to fit a model to the data distribution such that we can synthesize new data points at will by sampling from the distribution.
>  假设给定数据集 $\{\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_n\}$，其中每个数据点都独立从数据分布 $p(\mathbf x)$ 采样得到
>  生成式模型的目的是基于数据拟合分布 $p(\mathbf x)$，以从分布中采样新的数据点

In order to build such a generative model, we first need a way to represent a probability distribution. One such way, as in likelihood-based models, is to directly model the [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) (p.d.f.) or [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) (p.m.f.). Let $f_\theta(\mathbf x)\in \mathbb R$ be a real-valued function parameterized by a learnable parameter $\theta$. We can define a p.d.f. $^1$ via 

$$
p_\theta(\mathbf x) = \frac {e^{-f_\theta(\mathbf x)}}{Z_\theta}\tag{1}
$$

where $Z_\theta > 0$ is a normalizing constant dependent on $\theta$, such that $\int p_\theta(\mathbf x)d\mathbf x = 1$. Here the function $f_\theta(\mathbf x)$ is often called an unnormalized probabilistic model, or energy-based model [7] .

>  为了构建生成式模型，我们首先需要一种表征概率分布的方式
>  基于似然的模型直接建模概率密度 (质量) 函数:
>  令 $f_\theta(\mathbf x) \in \mathbb R$ 表示由 $\theta$ 参数化的实值函数，则样本 $\mathbf x$ 的概率可以按照 Eq 1 表示，其中 $Z_\theta > 0$ 为依赖于 $\theta$ 的规范化常数，它确保 $\int p_\theta (\mathbf x) d\mathbf x = 1$
>  函数 $f_\theta(\mathbf x)$ 通常称为未规范化的概率模型，或者基于能量的模型

We can train $p_\theta(\mathbf x)$ by maximizing the log-likelihood of the data 

$$
\max_\theta \sum_{i=1}^N \log p_\theta(\mathbf x_i) \tag{2}
$$

However, equation (2) requires $p_\theta(\mathbf x)$ to be a normalized probability density function. This is undesirable because in order to compute $p_\theta(\mathbf x)$, we must evaluate the normalizing constant $Z_\theta$ —a typically intractable quantity for any general $f_\theta(\mathbf x)$. Thus to make maximum likelihood training feasible, likelihood-based models must either restrict their model architectures (e.g., causal convolutions in autoregressive models, invertible networks in normalizing flow models) to make $Z_\theta$ tractable, or approximate the normalizing constant (e.g., variational inference in VAEs, or MCMC sampling used in contrastive divergence [30] ) which may be computationally expensive.

>  要训练模型 (拟合参数 $\theta$)，我们可以极大化给定数据的对数似然，如 Eq 2
>  但直接求解 Eq 2 要求我们能够评估规范化参数 $Z_\theta$，而对于任意通用的概率模型 $f_\theta(\mathbf x)$， $Z_\theta$ 通常不可解，故直接的极大极大似然是不可行的
>  为了使极大似然训练可行，基于似然的模型要么限制其模型架构 (例如自回归模型中的因果卷积、归一化流模型中的可逆网络) 使得 $Z_\theta$ 可计算，要么对 $Z_\theta$ 进行计算上较为昂贵的近似 (例如 VAE 中的变分推断，或者对比散度中的 MCMC 采样)

>  注意 Transformer 的自回归由于目标是离散词袋上的概率质量函数，故归一化因子通常是直接计算的 (softmax) ，不涉及积分，故不会像连续情况下不可解

By modeling the score function instead of the density function, we can sidestep the difficulty of intractable normalizing constants. The **score function** of a distribution $p(\mathbf x)$ is defined as 

$$
\nabla_{\mathbf x}\log p(\mathbf x)
$$

and a model for the score function is called a **score-based model** [18] , which we denote as $\mathbf s_{\theta}(\mathbf x)$. The score-based model is learned such that $\mathbf s_{\theta}(\mathbf x) \approx \nabla_{\mathbf x} \log p(\mathbf x)$, and can be parameterized without worrying about the normalizing constant. For example, we can easily parameterize a score-based model with the energy-based model defined in equation (1) , via

$$
\mathbf s_{\theta}(\mathbf x) = \nabla_{\mathbf x}\log p_\theta(\mathbf x) = -\nabla_{\mathbf x}f_\theta(\mathbf x) - \underbrace{\nabla_{\mathbf x}Z_{\theta}}_{=0} = -\nabla_{\mathbf x}f_\theta(\mathbf x)\tag{3}
$$

>  让模型建模得分函数而不是直接建模密度函数，我们可以避开求解规范化常数的麻烦
>  分布 $p(\mathbf x)$ 的得分函数被定义为其对数函数的梯度: $\nabla_{\mathbf x}\log p(\mathbf x)$
>  建模该得分函数的模型就称为基于得分的模型，我们记作 $\mathbf s_{\theta}(\mathbf x)$
>  基于得分的模型的学习目标就是拟合得分函数，即 $s_\theta(\mathbf x) \approx \nabla_{\mathbf x}\log p(\mathbf x)$
>  因为得分函数满足

$$
\begin{align}
&\nabla_{\mathbf x} \log p_\theta(\mathbf x)\\
= &\nabla_{\mathbf x} \log \frac {\exp \{-{f_\theta(\mathbf x)}\}}{Z_\theta}\\
= &-\nabla_{\mathbf x} f_{\theta}(\mathbf x) - \nabla_x\log Z_\theta\\
= &-\nabla_{\mathbf x} f_\theta(\mathbf x)
\end{align}
$$

>  故对得分函数的表征和拟合不需要担心规范化常数的问题

Note that the score-based model $\mathbf s_\theta(\mathbf x)$ is independent of the normalizing constant $Z_\theta$ ! This significantly expands the family of models that we can tractably use, since we don’t need any special architectures to make the normalizing constant tractable.
>  因此，基于得分的模型独立于规范化常数，不需要使用特殊的架构使得规范化常数可解
>  对 $\mathbf s_\theta(\mathbf x)$ 的唯一约束就是它需要是一个向量值函数，输入和输出的维度相同，对于其输出的取值则没有任何约束


![](https://yang-song.net/assets/img/score/ebm.gif)

Parameterizing probability density functions. No matter how you change the model family and parameters, it has to be normalized (area under the curve must integrate to one).
>  直接对密度函数参数化 (建模)，无论模型/参数族如何变化，它们都必须满足一个约束: 归一化约束，即曲线下的面积必须积分得 1

![](https://yang-song.net/assets/img/score/score.gif)

Parameterizing score functions. No need to worry about normalization.
>  对得分函数的参数化/建模则不需要考虑归一化约束

Similar to likelihood-based models, we can train score-based models by minimizing the **Fisher divergence** $^2$ between the model and the data distributions, defined as 

$$
\mathbb E_{p(\mathbf x)}[ \|\nabla_{\mathbf x}\log p(\mathbf x) - \mathbf s_\theta(\mathbf x)\|^2_2]\tag{4}
$$

Intuitively, the Fisher divergence compares the squared $\ell_2$ distance between the ground-truth data score and the score-based model. Directly computing this divergence, however, is infeasible because it requires access to the unknown data score $\nabla_{\mathbf x}\log p(\mathbf x)$. 

>  基于分数的模型可以通过最小化模型和数据分布之间的 Fisher 散度来训练
>  直观上看，Fisher 散度比较了真实数据的分数函数和模型拟合的分数函数之间的平方 $\ell_2$ 距离
>  直接计算 Fisher 散度是不可行的，因为我们无法不知道 $p(\mathbf x)$，故无法计算数据分数 $\nabla_{\mathbf x}\log p(\mathbf x)$ (或许可以用经验分布?)

> [!Note] Side Note
> 极大似然估计等价于最小化模型分布 $p_\theta(\mathbf x)$ 和经验分布 $p_{ep}(\mathbf x)$ 之间的 KL 散度，KL 散度的核心是最小化负对数似然 $-\sum_i \log p_\theta(\mathbf x_i)$ ($\mathbf x_i \sim p_{ep}(\mathbf x)$)，故等价于极大似然
> 如果将 Fisher 散度中的得分函数替换为概率密度本身，得到 $\mathbb E_{p_{ep}(\mathbf x)}[\|p_{ep}(\mathbf x) - p_\theta(\mathbf x)\|_2^2]$ (不妨称为 L2 散度)，最小化 L2 散度不等价于极大似然
> 差异在于:
> - KL 散度使用的是对数函数，低概率区域的变化对于损失的影响更大，L2 散度使用平方差，对于任意的区域的变化都一视同仁
> - KL 散度倾向于模式寻找，即模型分布覆盖到所有观测到的数据模式，如果没有覆盖到，则损失会变得无穷 ($p(\mathbf x_i) \to 0$)，故 KL 散度会强烈地避免 0 概率，L2 散度则没有这样的讲究，它更可能 “模糊” 或 “平均化” 模式
> 
 
Fortunately, there exists a family of methods called **score matching** $^3$ [16, 17, 31] that minimize the Fisher divergence without knowledge of the ground-truth data score. Score matching objectives can directly be estimated on a dataset and optimized with stochastic gradient descent, analogous to the log-likelihood objective for training likelihood-based models (with known normalizing constants). We can train the score-based model by minimizing a score matching objective, **without requiring adversarial optimization**.
>  score matching 方法可以在不依赖真实数据分数 ($\nabla_{\mathbf x}\log p(\mathbf x)$) 的情况下最小化 Fisher 散度
>  score matching 的目标函数可以直接在数据集上估计，然后使用随机梯度下降进行优化
>  我们可以最小化 score matching 目标函数来训练 score-based 模型，而不需要对抗优化

Additionally, using the score matching objective gives us a considerable amount of modeling flexibility. The Fisher divergence itself does not require $\mathbf s_\theta(\mathbf x)$ to be an actual score function of any normalized distribution—it simply compares the $\ell_2$ distance between the ground-truth data score and the score-based model, with no additional assumptions on the form of $\mathbf s_\theta(\mathbf x)$. In fact, the only requirement on the score-based model is that it should be a vector-valued function with the same input and output dimensionality, which is easy to satisfy in practice.

As a brief summary, we can represent a distribution by modeling its score function, which can be estimated by training a score-based model of free-form architectures with score matching.
>  小结: 我们通过模型建模/表征一个分布的得分函数来 (间接) 表示该分布，这样模型的架构不会存在限制，我们可以通过 score matching 方法来训练该模型

## Langevin dynamics
Once we have trained a score-based model $\mathbf s_\theta(\mathbf x) \approx \nabla_{\mathbf x}\log p(\mathbf x)$, we can use an iterative procedure called [**Langevin dynamics**](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) [32, 33] to draw samples from it.
>  score-based 模型训练好之后 (满足 $\mathbf s_\theta(\mathbf x) \approx \nabla_{\mathbf x}\log p(\mathbf x)$)，我们可以使用 Langevin dynamics 来从中采样

Langevin dynamics provides an MCMC procedure to sample from a distribution $p(\mathbf x)$ using only its score function $\nabla_{\mathbf x}\log p(\mathbf x)$. Specifically, it initializes the chain from an arbitrary prior distribution $\mathbf x_0\sim \pi(\mathbf x)$, and then iterates the following 

$$
\mathbf x_{i+1} \leftarrow \mathbf x_i + \epsilon \nabla_{\mathbf x} \log p(\mathbf x) + \sqrt {2\epsilon} \mathbf z_i, \quad i = 0, 1, \cdots, K\tag{6}
$$

where $\mathbf z_i \sim \mathcal N(0, I)$. When $\epsilon \to 0$ and $K\to \infty$, $\mathbf x_K$ obtained from the procedure in (6) converges to a sample from $p(\mathbf x)$ under some regularity conditions. In practice, the error is negligible when $\epsilon$ is sufficiently small and $K$ is sufficiently large.

>  Langevin dynamics 是一个基于分布的得分函数 $\nabla_{\mathbf x}\log p(\mathbf x)$ 来从分布 $p(\mathbf x)$ 采样的 MCMC 过程
>  Langevin dynamics 从任意的初始分布 $\pi(\mathbf x)$ 中采集一个样本 $\mathbf x_0$，然后按照 Eq 6 对其进行迭代更新 (其中，$\mathbf z_i \sim \mathcal N(0, I)$)
>  当 $\epsilon \to 0, K\to \infty$，$\mathbf x_K$ 将收敛到 (在一定的规范化条件下) 从 $p(\mathbf x)$ 采样得到的样本

>  Eq 6 还是挺直观的，使用梯度来更新样本

![](https://yang-song.net/assets/img/score/langevin.gif)

Using Langevin dynamics to sample from a mixture of two Gaussians.
>  图示展示了使用 Langevin dynamics 从两个高斯分布的混合采样的过程

Note that Langevin dynamics accesses $p(\mathbf x)$ only through $\nabla_{\mathbf x}\log p(\mathbf x)$. Since $\mathbf s_{\theta}(\mathbf x) \approx \nabla_{\mathbf x} \log p(\mathbf x)$, we can produce samples from our score-based model $\mathbf s_\theta(\mathbf x)$ by plugging it into equation (6).
>  Langevin dynamics 只需要 $\nabla_{\mathbf x}\log p(\mathbf x)$ 而不需要 $p(\mathbf x)$，故我们在实践中，直接用 $\mathbf s_\theta(\mathbf x)$ 替代 $\nabla_{\mathbf x}\log p(\mathbf x)$ 即可

## Naive score-based generative modeling and its pitfalls
So far, we’ve discussed how to train a score-based model with score matching, and then produce samples via Langevin dynamics. However, this naive approach has had limited success in practice—we’ll talk about some pitfalls of score matching that received little attention in prior works [18] .
>  目前为止，我们讨论了用 score matching 训练 score-based 模型，然后通过 Langevin dynamics 生成样本
>  这一方法在实践中的成功实际上有限，我们将讨论一些 score matching 的缺陷

![](https://yang-song.net/assets/img/score/smld.jpg)

Score-based generative modeling with score matching + Langevin dynamics.
>  该图展示了训练 score-based model 和用它生成样本的过程
>  该图假设了原始样本属于一个类似混合高斯的分布，我们通过数据集，学习到的 score-based $s_\theta(\mathbf x)$ 很好地拟合了各个数据点 $\mathbf x$ 的真实 score $\nabla_{\mathbf x}\log p(\mathbf x)$，这样，我们就可以从任意的随机数据点 $\mathbf x$ 开始，通过 Langevin dynamics，逐渐将 $\mathbf x$ 从属的分布收敛到 $p(\mathbf x)$

The key challenge is the fact that the estimated score functions are inaccurate in low density regions, where few data points are available for computing the score matching objective. 
>  使用 score matching 的关键挑战在于所估计的 score function 在低密度区域是不准确的 (从这句话来看，score matching 应该是用数据集的经验分布替代真实分布，因此数据集没有过多覆盖到的样本区域的估计就是不准确的)
>  低密度区域就是数据点很少的区域

This is expected as score matching minimizes the Fisher divergence 

$$
\mathbb E_{p(\mathbf x)} = \left[\|\nabla_{\mathbf x}\log p(\mathbf x)- \mathbf s_{\theta}(\mathbf x)\|_2^2\right] = \int p(\mathbf x)\|\nabla_{\mathbf x}\log p(\mathbf x)- \mathbf s_{\theta}(\mathbf x)\|_2^2d\mathbf x
$$

Since the $\ell_2$ differences between the true data score function and score-based model are weighted by $p(\mathbf x)$, they are largely ignored in low density regions where $p(\mathbf x)$ is small. 
>  这一点在优化公式中也可以看出来
>  score matching 的优化目标是最小化 $\mathbf s_\theta(\mathbf x)$ 和 $p(\mathbf x)$ 之间的 Fisher 散度，公式中，真实分数函数 $\nabla_{\mathbf x}p(\mathbf x)$ 和 score-based model $\mathbf s_{\theta}(\mathbf x)$ 的 $\ell_2$ 距离是由 $p(\mathbf x)$ 加权的，故对于低密度区域的样本点 $\mathbf x$，它们的 $\ell_2$ 距离在目标函数中占的权重本身就小 (因此可以看出这一缺点是使用 Fisher 散度的固有缺点，即便我们得到了真实的 $p(\mathbf x)$，优化方向还是主要倾向于高密度区域的数据点)

This behavior can lead to subpar results, as illustrated by the figure below:
>  这样的行为会导致次优的结果，如下图所示:

![](https://yang-song.net/assets/img/score/pitfalls.jpg)

Estimated scores are only accurate in high density regions.
>  图中展示了原始分布中的低密度区域的 data scores 和 estimated scores 都是不准确的 (即根据数据集评估的 scores 和模型学习到的 scores 都是不准确的)

When sampling with Langevin dynamics, our initial sample is highly likely in low density regions when data reside in a high dimensional space. Therefore, having an inaccurate score-based model will derail Langevin dynamics from the very beginning of the procedure, preventing it from generating high quality samples that are representative of the data.
>  如果数据是高维数据，使用 Langevin dynamics 时，初始的随机数据大概率会出现在低密度区域，故 score-based model 的不准确性将在一开始就误导 Langevin dynamics，从而无法生成能够代表数据的高质量样本

## Score-based generative modeling with multiple noise perturbations
How can we bypass the difficulty of accurate score estimation in regions of low data density? Our solution is to **perturb** data points with noise and train score-based models on the noisy data points instead. When the noise magnitude is sufficiently large, it can populate low data density regions to improve the accuracy of estimated scores. 
>  为了解决低密度区域的 score 估计问题，一个方法是使用噪声对数据点进行扰动，然后基于加噪的数据点训练 score-based model
>  当噪声强度足够大的时候，它可以填充低密度数据区，以提高 estimated score 的正确性

For example, here is what happens when we perturb a mixture of two Gaussians perturbed by additional Gaussian noise.
>  例如，我们可以用额外的高斯噪声扰动之前的高斯混合分布

![](https://yang-song.net/assets/img/score/single_noise.jpg)

Estimated scores are accurate everywhere for the noise-perturbed data distribution due to reduced low data density regions.
>  图中展示了使用高斯噪声扰动后得到的结果
>  添加了额外噪声后，原来的低密度数据区域的数量减少了，进而对于扰动后的分布的 score 估计会更加准确

>  低密度数据区域的密度增加了，高密度的数据区域的密度显然会相应减少，在扰动程度足够大的情况下，或许整个分布会成为均匀分布
>  这也是一个 trade-off:
>  - 在扰动足够大的情况下，低密度的区域会足够少，score 的估计会更加准确，但扰动后分布离原分布的距离更远，故 estimated score 离 accurate score 也一定仍存在距离
>  - 在扰动足够小的情况下，低密度的区域依然存在，score 的估计不准确，虽然扰动后的分布离原分布足够近，但 estimated score 离 accurate score 也仍存在距离
>  因此 estimated score 和 accurate score 的距离由扰动程度和原分布保留程度两者相关，而这两者是相互对立的

Yet another question remains: how do we choose an appropriate noise scale for the perturbation process? Larger noise can obviously cover more low density regions for better score estimation, but it over-corrupts the data and alters it significantly from the original distribution. Smaller noise, on the other hand, causes less corruption of the original data distribution, but does not cover the low density regions as well as we would like.
>  我们如何为扰动过程选择合适的噪声尺度？较大的噪声可以覆盖更多的低密度区域，从而更好地估计分数，但会过度破坏数据，导致与原始分布显著不同；较小的噪声对原始数据分布的破坏较小，但无法像期望的那样充分覆盖低密度区域

To achieve the best of both worlds, we use multiple scales of noise perturbations simultaneously [18, 19] . Suppose we always perturb the data with isotropic Gaussian noise, and let there be a total of $L$ increasing standard deviations $\sigma_1 < \sigma_2 < \cdots < \sigma_L$. 
>  我们可以同时使用多种尺度的噪声扰动
>  假设我们用 $L$ 个各向同性的高斯噪声 (所有维度上具有相同方差，且互不相关) 进行扰动，各个高斯噪声的标准差记作 $\sigma_i(i=1,2,\dots, L)$

We first perturb the data distribution $p(\mathbf x)$ with each of the Gaussian noise to obtain a noise-perturbed distribution

$$
p_{\sigma_{i}}(\mathbf x) = \int p(\mathbf y)\mathcal N(\mathbf x; \mathbf y, \sigma_i^2I)d\mathbf y
$$

Note that we can easily draw samples from $p_{\sigma_i}(\mathbf x)$ by sampling $\mathbf x\sim p(\mathbf x)$ and computing $\mathbf x + \sigma_i \mathbf z$, with $\mathbf z \sim \mathcal N(0, I)$.

>  对于每个高斯噪声，经过它扰动后的分布 $p_{\sigma_i}(\mathbf x)$ 的 (概率密度函数) 形式如上所示
>  (从公式来看，$p_{\sigma_i}(\mathbf x)$ 中样本 $\mathbf x$ 的生成过程可以看作 
>  1. 在原分布中采样样本 $\mathbf y\sim p(\mathbf x)$ 
>  2. 在均值为零的高斯分布中采样噪声 $\epsilon \sim \mathcal N(0, \sigma_i^2 I)$ 
>  3. 计算 $\mathbf x = \mathbf y + \epsilon$
>  过程很清晰，就是在原样本上添加了额外噪声)
>  要从扰动后的分布 $p_{\sigma_i}(\mathbf x)$ 中采样，我们就按照其生成过程，从原分布和噪声分布中分别采样 $\mathbf x \sim p(\mathbf x), \mathbf z \sim \mathcal N(0, I)$，然后计算 $\mathbf x + \sigma_i \mathbf z$ 即可 
>  (这里将从 $\mathcal N(0, \sigma_i^2I)$ 中采样的过程分为了先从 $\mathcal N(0, I)$ 中采样，再乘上 $\sigma_i$ 两步，这是合理的，因为从属于 $\mathcal N(0, I)$ 的样本，乘上 $\sigma_i$，就从属于 $\mathcal N(0, \sigma^2 I)$)
>  (这样的推导在数学上显然合理，因为数轴是连续的，但在计算机的有限精度条件下，这两种采样方式并不等价，因为数轴将会是离散的)

Next, we estimate the score function of each noise-perturbed distribution, $\nabla_{\mathbf x}\log p_{\sigma_i}(\mathbf x)$, by training a **Noise Conditional Score-Based Model** $\mathbf s_\theta(\mathbf x, i)$ (also called a Noise Conditional Score Network, or NCSN [18, 19, 21] , when parameterized with a neural network) with score matching, such that $\mathbf s_\theta(\mathbf x, i) \approx \nabla_{\mathbf x}\log p_{\sigma_i}(\mathbf x)$ for all $i=1,2,\cdots, L$.
>  对原数据进行了噪声扰动后，我们在扰动后得到的新数据上训练 score-based model，我们称之为条件于噪声的 score-based model，记作 $\mathbf s_\theta(\mathbf x, i)$
>  $\mathbf s_\theta(\mathbf x, i)$ 的训练目标是近似扰动后分布的得分函数 $\nabla_{\mathbf x} \log p_{\sigma_i}(\mathbf x)$
>  对于 $\sigma_i (i=1, 2, \dots, L)$，每个 $\sigma_i$ 都对应一个扰动分布 $p_{\sigma_i}(\mathbf x)$ 和一个 NCSN $\mathbf s_\theta(\mathbf x, i)$，我们用 $\theta$ 参数化所有的 NCSN，故学习到的参数 $\theta$ 最终将能够让模型能够拟合所有扰动分布的得分函数
>  (感觉这对 $\theta$ 的规模具有一定的要求，$\theta$ 需要先根据 $\mathbf x$，判断它最有可能是属于哪个扰动分布 $p_{\sigma_i}$，然后再代入它所拟合的对应的 $\nabla_{\mathbf x}\log p_{\sigma_i}$ 进行计算
>  又或者 $\theta$ 不是分别拟合一个个 $\nabla_{\mathbf x}\log p_{\sigma_i}$，而是先在各个扰动分布中提取公共特征，学会了原分布 $p(\mathbf x)$ 的 Pattern，进而拟合了原得分函数
>  具体的模式感觉还是依赖于优化过程和损失函数定义，第二种情况应该是理想的，如果损失函数的定义下，原分布的得分函数是最低点，则这样训练就是比较理想的)

![](https://yang-song.net/assets/img/score/multi_scale.jpg)

We apply multiple scales of Gaussian noise to perturb the data distribution (**first row**), and jointly estimate the score functions for all of them (**second row**).
>  图中展示了对数据分布应用不同尺度的噪声扰动后的结果，以及对这些分布协同估计的 score function (有多个扰动分布，但只训练一个 score-based model)

![](https://yang-song.net/assets/img/score/duoduo.jpg)

Perturbing an image with multiple scales of Gaussian noise.
>  图中展示了对一个真实的图像数据进行多尺度的噪声扰动得到的结果，可以看到，尺度越高，扰动后的像素点和原像素点的距离差距大的可能性越大，图像就更倾向于混乱的噪声

The training objective for $\mathbf s_\theta(\mathbf x, i)$ is a weighted sum of Fisher divergences for all noise scales. In particular, we use the objective below:

$$
\sum_{i=1}^L \lambda(i)\mathbb E_{p_{\sigma_i}(\mathbf x)}[\|\nabla_{\mathbf x}\log p_{\sigma_i}(\mathbf x)- \mathbf s_\theta(\mathbf x, i)\|_2^2]\tag{7}
$$

where $\lambda(i) \in \mathbb R > 0$ is a positive weighting function, often chosen to be $\lambda(i) = \sigma^2$. The objective (7) can be optimized with score matching, exactly as in optimizing the naive (unconditional) score-based model $\mathbf s_\theta(\mathbf x)$.

>  $\mathbf s_\theta(\mathbf x, i)$ 的训练目标是所有噪声尺度的 Fisher 散度加权和
>  权重函数通常选为 $\lambda(i) = \sigma^2$ (对于越高扰动的分布，权重越大)
>  该目标函数仍然可以用 score-matching 优化

After training our noise-conditional score-based model $\mathbf s_\theta(\mathbf x, i)$, we can produce samples from it by running Langevin dynamics for $i = L, L-1, \cdots, 1$ in sequence. This method is called **annealed Langevin dynamics** (defined by Algorithm 1 in  [18] , and improved by  [19, 34] ), since the noise scale $\sigma_i$ decreases (anneals) gradually over time.
>  训练好 noise-conditional score-based model $\mathbf s_\theta(\mathbf x, i)$ 之后，我们可以通过 annealed Langevin dynamics 来从中抽样

![](https://yang-song.net/assets/img/score/ald.gif)

Annealed Langevin dynamics combine a sequence of Langevin chains with gradually decreasing noise scales.

![](https://yang-song.net/assets/img/score/celeba_large.gif)

![](https://yang-song.net/assets/img/score/cifar10_large.gif)

Annealed Langevin dynamics for the Noise Conditional Score Network (NCSN) model (from ref. [18] ) trained on CelebA (**left**) and CIFAR-10 (**right**). We can start from unstructured noise, modify images according to the scores, and generate nice samples. The method achieved state-of-the-art Inception score on CIFAR-10 at its time.

Here are some practical recommendations for tuning score-based generative models with multiple noise scales:

- Choose $\sigma_1 < \sigma_2 < \cdots < \sigma_L$ as a [geometric progression](https://en.wikipedia.org/wiki/Geometric_progression#:~:text=In%20mathematics%2C%20a%20geometric%20progression,number%20called%20the%20common%20ratio.), with $\sigma_1$ being sufficiently small and $\sigma_L$ comparable to the maximum pairwise distance between all training data points [19] . $L$ is typically on the order of hundreds or thousands.
- Parameterize the score-based model $\mathbf s_\theta(\mathbf x, i)$ with U-Net skip connections [18, 20] .
- Apply exponential moving average on the weights of the score-based model when used at test time [19, 20] .

>  对于调优这类模型的一些建议:
>  - 将 $\sigma_1 < \cdots < \sigma_L$ 选择为几何级数，其中 $\sigma_1$ 足够小，$\sigma_L$ 和所有训练数据点之间的距离相当，$L$ 的大小通常在几百到几千之间
>  - 使用 U-Net skip connections 来参数化 $\mathbf s_\theta(\mathbf x, i)$
>  - 在测试时 (采样时)，对 score-based model 的权重使用指数移动平均 (简单移动平均对序列中所有数据点赋予相同权重，而指数移动平均的思想是越近的数据越重要，越近的数据点权重越高，故随着采样接近结束，Langevin dynamics 中的动态会越来越微弱) (这还是很类似退火思想的)
   
With such best practices, we are able to generate high quality image samples with comparable quality to GANs on various datasets, such as below:

![](https://yang-song.net/assets/img/score/ncsnv2.jpg)

Samples from the NCSNv2 [19] model. From left to right: FFHQ 256x256, LSUN bedroom 128x128, LSUN tower 128x128, LSUN church_outdoor 96x96, and CelebA 64x64.

## Score-based generative modeling with stochastic differential equations (SDEs)
As we already discussed, adding multiple noise scales is critical to the success of score-based generative models. By generalizing the number of noise scales to infinity [21] , we obtain not only **higher quality samples**, but also, among others, **exact log-likelihood computation**, and **controllable generation for inverse problem solving**.
>  多尺度的噪声规模对于 score-based model 至关重要
>  将噪声尺度泛化到无限，理论上可以达到精确的对数似然计算和可控的逆问题求解生成

In addition to this introduction, we have tutorials written in [Google Colab](https://colab.research.google.com/) to provide a step-by-step guide for training a toy model on MNIST. We also have more advanced code repositories that provide full-fledged implementations for large scale applications.

|Link|Description|
|---|---|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing) |Tutorial of score-based generative modeling with SDEs in JAX + FLAX|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dRR_0gNRmfLtPavX2APzUggBuXyjWW55?usp=sharing) |Load our pretrained checkpoints and play with sampling, likelihood computation, and controllable synthesis (JAX + FLAX)|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing) |Tutorial of score-based generative modeling with SDEs in PyTorch|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17lTrPLTt_0EDXa4hkbHmbAFQEkpRDZnh?usp=sharing) |Load our pretrained checkpoints and play with sampling, likelihood computation, and controllable synthesis (PyTorch)|
| [Code in JAX](https://github.com/yang-song/score_sde) |Score SDE codebase in JAX + FLAX|
| [Code in PyTorch](https://github.com/yang-song/score_sde_pytorch) |Score SDE codebase in PyTorch|

### Perturbing data with an SDE
When the number of noise scales approaches infinity, we essentially perturb the data distribution with continuously growing levels of noise. In this case, the noise perturbation procedure is a continuous-time [stochastic process](https://en.wikipedia.org/wiki/Stochastic_process#:~:text=A%20stochastic%20process%20is%20defined,measurable%20with%20respect%20to%20some), as demonstrated below
>  当噪声尺度的数量 $L$ 趋近于无限，我们实际上是以不断增长的噪声水平扰动数据分布
>  在这种情况下，噪声扰动过程是一个连续时间的随机过程

![](https://yang-song.net/assets/img/score/perturb_vp.gif)

Perturbing data to noise with a continuous-time stochastic process.

How can we represent a stochastic process in a concise way? Many stochastic processes ([diffusion processes](https://en.wikipedia.org/wiki/Diffusion_process) in particular) are solutions of stochastic differential equations (SDEs). In general, an SDE possesses the following form:

$$
\mathrm d \mathbf x = \mathbf f(\mathbf x , t)\mathrm dt + g(t)\mathrm d\mathbf w\tag{8}
$$

where $\mathbf f(\cdot, t): \mathbb R^d \mapsto \mathbb R^d$ is a vector-valued function called the drift coefficient, $g(t)\in \mathbb R$ is a real-valued function called the diffusion coefficient, $\mathbf w$ denotes a standard [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion), and $\mathrm d\mathbf w$ can be viewed as infinitesimal white noise.

>  一个随机过程可以用随机微分方程 (SDE) 进行表示
>  该方程包含了两个驱动随机变量随着时间变化的因素: 一个可预测的、确定性的趋势和一个不可预测的、随机性的波动

The solution of a stochastic differential equation is a continuous collection of random variables $\{x(t)\}_{t\in [0, T]}$. These random variables trace stochastic trajectories as the time index t grows from the start time 0 to the end time T. Let pt (x) denote the (marginal) probability density function of x (t). Here t∈[0, T] is analogous to i=1,2,⋯, L when we had a finite number of noise scales, and pt (x) is analogous to pσi (x). Clearly, p0 (x)=p (x) is the data distribution since no perturbation is applied to data at t=0. After perturbing p (x) with the stochastic process for a sufficiently long time T, pT (x) becomes close to a tractable noise distribution π(x), called a **prior distribution**. We note that pT (x) is analogous to pσL (x) in the case of finite noise scales, which corresponds to applying the largest noise perturbation σL to the data.

The SDE in [](https://yang-song.net/blog/2021/score/#mjx-eqn%3Asde)(8) is **hand designed**, similarly to how we hand-designed σ1<σ2<⋯<σL in the case of finite noise scales. There are numerous ways to add noise perturbations, and the choice of SDEs is not unique. For example, the following SDE

(9) dx=etdw

perturbs data with a Gaussian noise of mean zero and exponentially growing variance, which is analogous to perturbing data with N (0,σ12I), N (0,σ22I),⋯, N (0,σL2I) when σ1<σ2<⋯<σL is a [geometric progression](https://en.wikipedia.org/wiki/Geometric_progression#:~:text=In%20mathematics%2C%20a%20geometric%20progression,number%20called%20the%20common%20ratio.). Therefore, the SDE should be viewed as part of the model, much like {σ1,σ2,⋯,σL}. In 

[21]

, we provide three SDEs that generally work well for images: the Variance Exploding SDE (VE SDE), the Variance Preserving SDE (VP SDE), and the sub-VP SDE.

### Reversing the SDE for sample generation

Recall that with a finite number of noise scales, we can generate samples by reversing the perturbation process with **annealed Langevin dynamics**, i.e., sequentially sampling from each noise-perturbed distribution using Langevin dynamics. For infinite noise scales, we can analogously reverse the perturbation process for sample generation by using the reverse SDE.

![](https://yang-song.net/assets/img/score/denoise_vp.gif)

Generate data from noise by reversing the perturbation procedure.

Importantly, any SDE has a corresponding reverse SDE 

[35]

, whose closed form is given by

(10) dx=[f (x, t)−g2 (t)∇xlog⁡pt (x)]dt+g (t) dw.

Here dt represents a negative infinitesimal time step, since the SDE [](https://yang-song.net/blog/2021/score/#mjx-eqn%3Arsde)(10) needs to be solved backwards in time (from t=T to t=0). In order to compute the reverse SDE, we need to estimate ∇xlog⁡pt (x), which is exactly the **score function** of pt (x).

![](https://yang-song.net/assets/img/score/sde_schematic.jpg)

Solving a reverse SDE yields a score-based generative model. Transforming data to a simple noise distribution can be accomplished with an SDE. It can be reversed to generate samples from noise if we know the score of the distribution at each intermediate time step.

### Estimating the reverse SDE with score-based models and score matching

Solving the reverse SDE requires us to know the terminal distribution pT (x), and the score function ∇xlog⁡pt (x). By design, the former is close to the prior distribution π(x) which is fully tractable. In order to estimate ∇xlog⁡pt (x), we train a **Time-Dependent Score-Based Model** sθ(x, t), such that sθ(x, t)≈∇xlog⁡pt (x). This is analogous to the noise-conditional score-based model sθ(x, i) used for finite noise scales, trained such that sθ(x, i)≈∇xlog⁡pσi (x).

Our training objective for sθ(x, t) is a continuous weighted combination of Fisher divergences, given by

(11) Et∈U (0, T) Ept (x)[λ(t)‖∇xlog⁡pt (x)−sθ(x, t)‖22],

where U (0, T) denotes a uniform distribution over the time interval [0, T], and λ:R→R>0 is a positive weighting function. Typically we use λ(t)∝1/E[‖∇x (t) log⁡p (x (t)∣x (0))‖22] to balance the magnitude of different score matching losses across time.

As before, our weighted combination of Fisher divergences can be efficiently optimized with score matching methods, such as denoising score matching 

[17]

 and sliced score matching 

[31]

. Once our score-based model sθ(x, t) is trained to optimality, we can plug it into the expression of the reverse SDE in [](https://yang-song.net/blog/2021/score/#mjx-eqn%3Arsde)(10) to obtain an estimated reverse SDE.

(12) dx=[f (x, t)−g2 (t) sθ(x, t)]dt+g (t) dw.

We can start with x (T)∼π, and solve the above reverse SDE to obtain a sample x (0). Let us denote the distribution of x (0) obtained in such way as pθ. When the score-based model sθ(x, t) is well-trained, we have pθ≈p0, in which case x (0) is an approximate sample from the data distribution p0.

When λ(t)=g2 (t), we have an important connection between our weighted combination of Fisher divergences and the KL divergence from p0 to pθ under some regularity conditions 

[36]

:

KL⁡(p0 (x)‖   $p_\theta(\mathbf x)$)≤T2Et∈U (0, T) Ept (x) [λ(t)‖∇xlog⁡pt(x)−sθ(x,t)‖22](13)+KL⁡(pT‖π).

Due to this special connection to the KL divergence and the equivalence between minimizing KL divergences and maximizing likelihood for model training, we call λ(t)=g (t) 2 the **likelihood weighting function**. Using this likelihood weighting function, we can train score-based generative models to achieve very high likelihoods, comparable or even superior to state-of-the-art autoregressive models

[36]

.

### How to solve the reverse SDE

By solving the estimated reverse SDE with numerical SDE solvers, we can simulate the reverse stochastic process for sample generation. Perhaps the simplest numerical SDE solver is the [Euler-Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method). When applied to our estimated reverse SDE, it discretizes the SDE using finite time steps and small Gaussian noise. Specifically, it chooses a small negative time step Δt≈0, initializes t←T, and iterates the following procedure until t≈0:

Δx←[f (x, t)−g2 (t) sθ(x, t)]Δt+g (t)|Δt|ztx←x+Δxt←t+Δt,

Here zt∼N (0, I). The Euler-Maruyama method is qualitatively similar to Langevin dynamics—both update x by following score functions perturbed with Gaussian noise.

Aside from the Euler-Maruyama method, other numerical SDE solvers can be directly employed to solve the reverse SDE for sample generation, including, for example, [Milstein method](https://en.wikipedia.org/wiki/Milstein_method), and [stochastic Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_\(SDE\)). In 

[21]

, we provided a reverse diffusion solver similar to Euler-Maruyama, but more tailored for solving reverse-time SDEs. More recently, authors in 

[37]

 introduced adaptive step-size SDE solvers that can generate samples faster with better quality.

In addition, there are two special properties of our reverse SDE that allow for even more flexible sampling methods:

- We have an estimate of ∇xlog⁡pt (x) via our time-dependent score-based model sθ(x, t).
- We only care about sampling from each marginal distribution pt (x). Samples obtained at different time steps can have arbitrary correlations and do not have to form a particular trajectory sampled from the reverse SDE.

As a consequence of these two properties, we can apply MCMC approaches to fine-tune the trajectories obtained from numerical SDE solvers. Specifically, we propose **Predictor-Corrector samplers**. The **predictor** can be any numerical SDE solver that predicts x (t+Δt)∼pt+Δt (x) from an existing sample x (t)∼pt (x). The **corrector** can be any MCMC procedure that solely relies on the score function, such as Langevin dynamics and Hamiltonian Monte Carlo.

At each step of the Predictor-Corrector sampler, we first use the predictor to choose a proper step size Δt<0, and then predict x (t+Δt) based on the current sample x (t). Next, we run several corrector steps to improve the sample x (t+Δt) according to our score-based model sθ(x, t+Δt), so that x (t+Δt) becomes a higher-quality sample from pt+Δt (x).

With Predictor-Corrector methods and better architectures of score-based models, we can achieve **state-of-the-art** sample quality on CIFAR-10 (measured in FID 

[38]

 and Inception scores 

[12]

), outperforming the best GAN model to date (StyleGAN2 + ADA 

[39]

).

|Method|FID ↓|Inception score ↑|
|---|---|---|
|StyleGAN2 + ADA <br><br>[39]|2.92|9.83|
|Ours <br><br>[21]|**2.20**|**9.89**|

The sampling methods are also scalable for extremely high dimensional data. For example, it can successfully generate high fidelity images of resolution 1024×1024.

![](https://yang-song.net/assets/img/score/ffhq_1024.jpeg)

1024 x 1024 samples from a score-based model trained on the FFHQ dataset.

Some additional (uncurated) samples for other datasets (taken from this [GitHub repo](https://github.com/yang-song/score_sde)):

![](https://yang-song.net/assets/img/score/bedroom.jpeg)

256 x 256 samples on LSUN bedroom.

![](https://yang-song.net/assets/img/score/celebahq_256.jpg)

256 x 256 samples on CelebA-HQ.

### Probability flow ODE

Despite capable of generating high-quality samples, samplers based on Langevin MCMC and SDE solvers do not provide a way to compute the exact log-likelihood of score-based generative models. Below, we introduce a sampler based on ordinary differential equations (ODEs) that allow for exact likelihood computation.

In 

[21]

, we show t is possible to convert any SDE into an ordinary differential equation (ODE) without changing its marginal distributions {pt (x)}t∈[0, T]. Thus by solving this ODE, we can sample from the same distributions as the reverse SDE. The corresponding ODE of an SDE is named **probability flow ODE** 

[21]

, given by

(14) dx=[f (x, t)−12g2 (t)∇xlog⁡pt (x)]dt.

The following figure depicts trajectories of both SDEs and probability flow ODEs. Although ODE trajectories are noticeably smoother than SDE trajectories, they convert the same data distribution to the same prior distribution and vice versa, sharing the same set of marginal distributions {pt (x)}t∈[0, T]. In other words, trajectories obtained by solving the probability flow ODE have the same marginal distributions as the SDE trajectories.

![](https://yang-song.net/assets/img/score/teaser.jpg)

We can map data to a noise distribution (the prior) with an SDE, and reverse this SDE for generative modeling. We can also reverse the associated probability flow ODE, which yields a deterministic process that samples from the same distribution as the SDE. Both the reverse-time SDE and probability flow ODE can be obtained by estimating score functions.

This probability flow ODE formulation has several unique advantages.

When ∇xlog⁡pt (x) is replaced by its approximation sθ(x, t), the probability flow ODE becomes a special case of a neural ODE

[40]

. In particular, it is an example of continuous normalizing flows

[41]

, since the probability flow ODE converts a data distribution p0 (x) to a prior noise distribution pT (x) (since it shares the same marginal distributions as the SDE) and is fully invertible.

As such, the probability flow ODE inherits all properties of neural ODEs or continuous normalizing flows, including exact log-likelihood computation. Specifically, we can leverage the instantaneous change-of-variable formula (Theorem 1 in 

[40]

, Equation (4) in 

[41]

) to compute the unknown data density p0 from the known prior density pT with numerical ODE solvers.

In fact, our model achieves the **state-of-the-art** log-likelihoods on uniformly dequantized 4 CIFAR-10 images 

[21]

, **even without maximum likelihood training**.

|Method|Negative log-likelihood (bits/dim) ↓|
|---|---|
|RealNVP|3.49|
|iResNet|3.45|
|Glow|3.35|
|FFJORD|3.40|
|Flow++|3.29|
|Ours|**2.99**|

When training score-based models with the **likelihood weighting** we discussed before, and using **variational dequantization** to obtain likelihoods on discrete images, we can achieve comparable or even superior likelihood to the state-of-the-art autoregressive models (all without any data augmentation) 

[36]

.

|Method|Negative log-likelihood (bits/dim) ↓ on CIFAR-10|Negative log-likelihood (bits/dim) ↓ on ImageNet 32x32|
|---|---|---|
|Sparse Transformer|**2.80**|-|
|Image Transformer|2.90|3.77|
|Ours|2.83|**3.76**|

### Controllable generation for inverse problem solving

Score-based generative models are particularly suitable for solving inverse problems. At its core, inverse problems are same as Bayesian inference problems. Let x and y be two random variables, and suppose we know the forward process of generating y from x, represented by the transition probability distribution p (y∣x). The inverse problem is to compute p (x∣y). From Bayes’ rule, we have p (x∣y)=p (x) p (y∣x)/∫p (x) p (y∣x) dx. This expression can be greatly simplified by taking gradients with respect to x on both sides, leading to the following Bayes’ rule for score functions:

(15)∇xlog⁡p (x∣y)=∇xlog⁡p (x)+∇xlog⁡p (y∣x).

Through score matching, we can train a model to estimate the score function of the unconditional data distribution, i.e., sθ(x)≈∇xlog⁡p (x). This will allow us to easily compute the posterior score function ∇xlog⁡p (x∣y) from the known forward process p (y∣x) via equation [](https://yang-song.net/blog/2021/score/#mjx-eqn%3Ainverse_problem)(15), and sample from it with Langevin-type sampling 

[21]

.

A recent work from UT Austin 

[29]

 has demonstrated that score-based generative models can be applied to solving inverse problems in medical imaging, such as accelerating magnetic resonance imaging (MRI). Concurrently in 

[42]

, we demonstrated superior performance of score-based generative models not only on accelerated MRI, but also sparse-view computed tomography (CT). We were able to achieve comparable or even better performance than supervised or unrolled deep learning approaches, while being more robust to different measurement processes at test time.

Below we show some examples on solving inverse problems for computer vision.

![](https://yang-song.net/assets/img/score/class_cond.png)

Class-conditional generation with an unconditional time-dependent score-based model, and a pre-trained noise-conditional image classifier on CIFAR-10.

![](https://yang-song.net/assets/img/score/inpainting.png)

Image inpainting with a time-dependent score-based model trained on LSUN bedroom. The leftmost column is ground-truth. The second column shows masked images (y in our framework). The rest columns show different inpainted images, generated by solving the conditional reverse-time SDE.

![](https://yang-song.net/assets/img/score/colorization.png)

Image colorization with a time-dependent score-based model trained on LSUN church_outdoor and bedroom. The leftmost column is ground-truth. The second column shows gray-scale images (y in our framework). The rest columns show different colorizedimages, generated by solving the conditional reverse-time SDE.

![](https://yang-song.net/assets/img/score/lincoln.png)

We can even colorize gray-scale portrays of famous people in history (Abraham Lincoln) with a time-dependent score-based model trained on FFHQ. The image resolution is 1024 x 1024.

## Connection to diffusion models and others

I started working on score-based generative modeling since 2019, when I was trying hard to make score matching scalable for training deep energy-based models on high-dimensional datasets. My first attempt at this led to the method sliced score matching

[31]

. Despite the scalability of sliced score matching for training energy-based models, I found to my surprise that Langevin sampling from those models fails to produce reasonable samples even on the MNIST dataset. I started investigating this issue and discovered three crucial improvements that can lead to extremely good samples: (1) perturbing data with multiple scales of noise, and training score-based models for each noise scale; (2) using a U-Net architecture (we used RefineNet since it is a modern version of U-Nets) for the score-based model; (3) applying Langevin MCMC to each noise scale and chaining them together. With those methods, I was able to obtain the state-of-the-art Inception Score on CIFAR-10 in 

[18]

 (even better than the best GANs!), and generate high-fidelity image samples of resolution up to 256×256256×256 in 

[19]

.

The idea of perturbing data with multiple scales of noise is by no means unique to score-based generative models though. It has been previously used in, for example, [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing), annealed importance sampling

[43]

, diffusion probabilistic models

[44]

, infusion training

[45]

, and variational walkback

[46]

 for generative stochastic networks

[47]

. Out of all these works, diffusion probabilistic modeling is perhaps the closest to score-based generative modeling. Diffusion probabilistic models are hierachical latent variable models first proposed by [Jascha](http://www.sohldickstein.com/) and his colleagues 

[44]

 in 2015, which generate samples by learning a variational decoder to reverse a discrete diffusion process that perturbs data to noise. Without awareness of this work, score-based generative modeling was proposed and motivated independently from a very different perspective. Despite both perturbing data with multiple scales of noise, the connection between score-based generative modeling and diffusion probabilistic modeling seemed superficial at that time, since the former is trained by score matching and sampled by Langevin dynamics, while the latter is trained by the evidence lower bound (ELBO) and sampled with a learned decoder.

In 2020, [Jonathan Ho](http://www.jonathanho.me/) and colleagues 

[20]

 significantly improved the empirical performance of diffusion probabilistic models and first unveiled a deeper connection to score-based generative modeling. They showed that the ELBO used for training diffusion probabilistic models is essentially equivalent to the weighted combination of score matching objectives used in score-based generative modeling. Moreover, by parameterizing the decoder as a sequence of score-based models with a U-Net architecture, they demonstrated for the first time that diffusion probabilistic models can also generate high quality image samples comparable or superior to GANs.

Inspired by their work, we further investigated the relationship between diffusion models and score-based generative models in an ICLR 2021 paper 

[21]

. We found that the sampling method of diffusion probabilistic models can be integrated with annealed Langevin dynamics of score-based models to create a unified and more powerful sampler (the Predictor-Corrector sampler). By generalizing the number of noise scales to infinity, we further proved that score-based generative models and diffusion probabilistic models can both be viewed as discretizations to stochastic differential equations determined by score functions. This work bridges both score-based generative modeling and diffusion probabilistic modeling into a unified framework.

Collectively, these latest developments seem to indicate that both score-based generative modeling with multiple noise perturbations and diffusion probabilistic models are different perspectives of the same model family, much like how [wave mechanics](https://en.wikipedia.org/wiki/Wave_mechanics) and [matrix mechanics](https://en.wikipedia.org/wiki/Matrix_mechanics) are equivalent formulations of quantum mechanics in the history of physics 5 . The perspective of score matching and score-based models allows one to calculate log-likelihoods exactly, solve inverse problems naturally, and is directly connected to energy-based models, Schrödinger bridges and optimal transport

[48]

. The perspective of diffusion models is naturally connected to VAEs, lossy compression, and can be directly incorporated with variational probabilistic inference. This blog post focuses on the first perspective, but I highly recommend interested readers to learn about the alternative perspective of diffusion models as well (see [a great blog by Lilian Weng](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html)).

Many recent works on score-based generative models or diffusion probabilistic models have been deeply influenced by knowledge from both sides of research (see a [website](https://scorebasedgenerativemodeling.github.io/) curated by researchers at the University of Oxford). Despite this deep connection between score-based generative models and diffusion models, it is hard to come up with an umbrella term for the model family that they both belong to. Some colleagues in DeepMind propose to call them “Generative Diffusion Processes”. It remains to be seen if this will be adopted by the community in the future.

## Concluding remarks

This blog post gives a detailed introduction to score-based generative models. We demonstrate that this new paradigm of generative modeling is able to produce high quality samples, compute exact log-likelihoods, and perform controllable generation for inverse problem solving. It is a compilation of several papers we published in the past few years. Please visit them if you are interested in more details:

- [Yang Song*, Sahaj Garg*, Jiaxin Shi, and Stefano Ermon. _Sliced Score Matching: A Scalable Approach to Density and Score Estimation_. UAI 2019 (Oral)](https://arxiv.org/abs/1905.07088)
   
- [Yang Song, and Stefano Ermon. _Generative Modeling by Estimating Gradients of the Data Distribution_. NeurIPS 2019 (Oral)](https://arxiv.org/abs/1907.05600)
   
- [Yang Song, and Stefano Ermon. _Improved Techniques for Training Score-Based Generative Models_. NeurIPS 2020](https://arxiv.org/abs/2006.09011)
   
- [Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. _Score-Based Generative Modeling through Stochastic Differential Equations_. ICLR 2021 (Outstanding Paper Award)](https://arxiv.org/abs/2011.13456)
   
- [Yang Song*, Conor Durkan*, Iain Murray, and Stefano Ermon. _Maximum Likelihood Training of Score-Based Diffusion Models_. NeurIPS 2021 (Spotlight)](https://arxiv.org/abs/2101.09258)
   
- [Yang Song*, Liyue Shen*, Lei Xing, and Stefano Ermon. _Solving Inverse Problems in Medical Imaging with Score-Based Generative Models_. ICLR 2022](https://arxiv.org/abs/2111.08005)
   

For a list of works that have been influenced by score-based generative modeling, researchers at the University of Oxford have built a very useful (but necessarily incomplete) website: [https://scorebasedgenerativemodeling.github.io/](https://scorebasedgenerativemodeling.github.io/).

There are two major challenges of score-based generative models. First, the sampling speed is slow since it involves a large number of Langevin-type iterations. Second, it is inconvenient to work with discrete data distributions since scores are only defined on continuous distributions.

The first challenge can be partially solved by using numerical ODE solvers for the probability flow ODE with lower precision (a similar method, denoising diffusion implicit modeling, has been proposed in 

[49]

). It is also possible to learn a direct mapping from the latent space of probability flow ODEs to the image space, as shown in 

[50]

. However, all such methods to date result in worse sample quality.

The second challenge can be addressed by learning an autoencoder on discrete data and performing score-based generative modeling on its continuous latent space 

[28, 51]

. Jascha’s original work on diffusion models 

[44]

 also provides a discrete diffusion process for discrete data distributions, but its potential for large scale applications remains yet to be proven.

It is my conviction that these challenges will soon be solved with the joint efforts of the research community, and score-based generative models/ diffusion-based models will become one of the most useful tools for data generation, density estimation, inverse problem solving, and many other downstream tasks in machine learning.

### Footnotes

1. Hereafter we only consider probability density functions. Probability mass functions are similar. [↩](https://yang-song.net/blog/2021/score/#d-footnote-1)
2. Fisher divergence is typically between two distributions $p$ and $q$, defined as $\mathbb E_{p (\mathbf x)}[ \|\nabla_{\mathbf x}\log p (\mathbf x) - \nabla_{\mathbf x}\log q(\mathbf x)\|^2_2]$ . Here we slightly abuse the term as the name of a closely related expression for score-based models. [↩](https://yang-song.net/blog/2021/score/#d-footnote-2)
>  Fisher 散度衡量了两个分布 $p, q$ 之间的距离，定义为 $\mathbb E_{p (\mathbf x)}[ \|\nabla_{\mathbf x}\log p (\mathbf x) - \nabla_{\mathbf x}\log q(\mathbf x)\|^2_2]$，类似于 $p$ 和 $q$ 的得分函数之间的 (加权) 欧式距离 (将函数视作无限长的向量，将 $p(\mathbf x)$ 视作权重)

3. Commonly used score matching methods include denoising score matching [17] and sliced score matching [31] . Here is an introduction to [score matching and sliced score matching](https://yang-song.net/blog/2019/ssm/). [↩](https://yang-song.net/blog/2021/score/#d-footnote-3)
>  常用的得分匹配方法包括了去噪得分匹配和切片得分匹配

4. It is typical for normalizing flow models to convert discrete images to continuous ones by adding small uniform noise to them. [↩](https://yang-song.net/blog/2021/score/#d-footnote-4)
5. Goes without saying that the significance of score-based generative models/diffusion probabilistic models is in no way comparable to quantum mechanics. [↩](https://yang-song.net/blog/2021/score/#d-footnote-5)

### References
1. The neural autoregressive distribution estimator 
   Larochelle, H. and Murray, I., 2011. International Conference on Artificial Intelligence and Statistics, pp. 29--37.
2. Made: Masked autoencoder for distribution estimation 
   Germain, M., Gregor, K., Murray, I. and Larochelle, H., 2015. International Conference on Machine Learning, pp. 881--889.
3. Pixel recurrent neural networks 
   Van Oord, A., Kalchbrenner, N. and Kavukcuoglu, K., 2016. International Conference on Machine Learning, pp. 1747--1756.
4. NICE: Non-linear independent components estimation 
   Dinh, L., Krueger, D. and Bengio, Y., 2014. arXiv preprint arXiv: 1410.8516.
5. Density estimation using Real NVP 
   Dinh, L., Sohl-Dickstein, J. and Bengio, S., 2017. International Conference on Learning Representations.
6. A tutorial on energy-based learning 
   LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. and Huang, F., 2006. Predicting structured data, Vol 1 (0).
7. How to Train Your Energy-Based Models 
   Song, Y. and Kingma, D.P., 2021. arXiv preprint arXiv: 2101.03288.
8. Auto-encoding variational bayes 
   Kingma, D.P. and Welling, M., 2014. International Conference on Learning Representations.
9. Stochastic backpropagation and approximate inference in deep generative models 
   Rezende, D.J., Mohamed, S. and Wierstra, D., 2014. International conference on machine learning, pp. 1278--1286.
10. Learning in implicit generative models 
   Mohamed, S. and Lakshminarayanan, B., 2016. arXiv preprint arXiv: 1610.03483.
11. Generative adversarial nets 
   Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y., 2014. Advances in neural information processing systems, pp. 2672--2680.
12. Improved techniques for training gans 
   Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A. and Chen, X., 2016. Advances in Neural Information Processing Systems, pp. 2226--2234.
13. Unrolled Generative Adversarial Networks [[link]] (https://openreview.net/forum?id=BydrOIcle) 
   Metz, L., Poole, B., Pfau, D. and Sohl-Dickstein, J., 2017. 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview. net.
14. A Kernel Test of Goodness of Fit [[HTML]] (https://proceedings.mlr.press/v48/chwialkowski16.html) 
   Chwialkowski, K., Strathmann, H. and Gretton, A., 2016. Proceedings of The 33rd International Conference on Machine Learning, Vol 48, pp. 2606--2615. PMLR.
15. A kernelized Stein discrepancy for goodness-of-fit tests 
   Liu, Q., Lee, J. and Jordan, M., 2016. International conference on machine learning, pp. 276--284.
16. Estimation of non-normalized statistical models by score matching 
   Hyvarinen, A., 2005. Journal of Machine Learning Research, Vol 6 (Apr), pp. 695--709.
17. A connection between score matching and denoising autoencoders 
   Vincent, P., 2011. Neural computation, Vol 23 (7), pp. 1661--1674. MIT Press.
18. Generative Modeling by Estimating Gradients of the Data Distribution [[PDF]] (http://arxiv.org/pdf/1907.05600.pdf) 
   Song, Y. and Ermon, S., 2019. Advances in Neural Information Processing Systems, pp. 11895--11907.
19. Improved Techniques for Training Score-Based Generative Models [[PDF]] (http://arxiv.org/pdf/2006.09011.pdf) 
   Song, Y. and Ermon, S., 2020. Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.
20. Denoising diffusion probabilistic models 
   Ho, J., Jain, A. and Abbeel, P., 2020. arXiv preprint arXiv: 2006.11239.
21. Score-Based Generative Modeling through Stochastic Differential Equations [[link]] (https://openreview.net/forum?id=PxTIG12RRHS) 
   Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and Poole, B., 2021. International Conference on Learning Representations.
22. Diffusion models beat gans on image synthesis 
   Dhariwal, P. and Nichol, A., 2021. arXiv preprint arXiv: 2105.05233.
23. Cascaded Diffusion Models for High Fidelity Image Generation 
   Ho, J., Saharia, C., Chan, W., Fleet, D.J., Norouzi, M. and Salimans, T., 2021.
24. WaveGrad: Estimating Gradients for Waveform Generation [[link]] (https://openreview.net/forum?id=NsMLjcFaO8O) 
   Chen, N., Zhang, Y., Zen, H., Weiss, R.J., Norouzi, M. and Chan, W., 2021. International Conference on Learning Representations.
25. DiffWave: A Versatile Diffusion Model for Audio Synthesis [[link]] (https://openreview.net/forum?id=a-xFK8Ymz5J) 
   Kong, Z., Ping, W., Huang, J., Zhao, K. and Catanzaro, B., 2021. International Conference on Learning Representations.
26. Grad-tts: A diffusion probabilistic model for text-to-speech 
   Popov, V., Vovk, I., Gogoryan, V., Sadekova, T. and Kudinov, M., 2021. arXiv preprint arXiv: 2105.06337.
27. Learning Gradient Fields for Shape Generation 
   Cai, R., Yang, G., Averbuch-Elor, H., Hao, Z., Belongie, S., Snavely, N. and Hariharan, B., 2020. Proceedings of the European Conference on Computer Vision (ECCV).
28. Symbolic Music Generation with Diffusion Models 
   Mittal, G., Engel, J., Hawthorne, C. and Simon, I., 2021. arXiv preprint arXiv: 2103.16091.
29. Robust Compressed Sensing MRI with Deep Generative Priors 
   Jalal, A., Arvinte, M., Daras, G., Price, E., Dimakis, A.G. and Tamir, J.I., 2021. Advances in neural information processing systems.
30. Training products of experts by minimizing contrastive divergence 
   Hinton, G.E., 2002. Neural computation, Vol 14 (8), pp. 1771--1800. MIT Press.
31. Sliced score matching: A scalable approach to density and score estimation [[PDF]] (http://arxiv.org/pdf/1905.07088.pdf) 
   Song, Y., Garg, S., Shi, J. and Ermon, S., 2020. Uncertainty in Artificial Intelligence, pp. 574--584.
32. Correlation functions and computer simulations 
   Parisi, G., 1981. Nuclear Physics B, Vol 180 (3), pp. 378--384. Elsevier.
33. Representations of knowledge in complex systems 
   Grenander, U. and Miller, M.I., 1994. Journal of the Royal Statistical Society: Series B (Methodological), Vol 56 (4), pp. 549--581. Wiley Online Library.
34. Adversarial score matching and improved sampling for image generation [[link]] (https://openreview.net/forum?id=eLfqMl3z3lq) 
   Jolicoeur-Martineau, A., Piche-Taillefer, R., Mitliagkas, I. and Combes, R.T.d., 2021. International Conference on Learning Representations.
35. Reverse-time diffusion equation models 
   Anderson, B.D., 1982. Stochastic Processes and their Applications, Vol 12 (3), pp. 313--326. Elsevier.
36. Maximum Likelihood Training of Score-Based Diffusion Models 
   Song, Y., Durkan, C., Murray, I. and Ermon, S., 2021. Advances in Neural Information Processing Systems (NeurIPS).
37. Gotta Go Fast When Generating Data with Score-Based Models 
   Jolicoeur-Martineau, A., Li, K., Piche-Taillefer, R., Kachman, T. and Mitliagkas, I., 2021. arXiv preprint arXiv: 2105.14080.
38. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium 
   Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S., 2017. Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, {USA}, pp. 6626--6637.
39. Training Generative Adversarial Networks with Limited Data 
   Karras, T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J. and Aila, T., 2020. Proc. NeurIPS.
40. Neural Ordinary Differential Equations 
   Chen, T.Q., Rubanova, Y., Bettencourt, J. and Duvenaud, D., 2018. Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr{\'{e}}al, Canada, pp. 6572--6583.
41. Scalable Reversible Generative Models with Free-form Continuous Dynamics [[link]] (https://openreview.net/forum?id=rJxgknCcK7) 
   Grathwohl, W., Chen, R.T.Q., Bettencourt, J. and Duvenaud, D., 2019. International Conference on Learning Representations.
42. Solving Inverse Problems in Medical Imaging with Score-Based Generative Models [[PDF]] (http://arxiv.org/pdf/2111.08005.pdf) 
   Song, Y., Shen, L., Xing, L. and Ermon, S., 2022. International Conference on Learning Representations.
43. Annealed importance sampling 
   Neal, R.M., 2001. Statistics and computing, Vol 11 (2), pp. 125--139. Springer.
44. Deep unsupervised learning using nonequilibrium thermodynamics 
   Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. and Ganguli, S., 2015. International Conference on Machine Learning, pp. 2256--2265.
45. Learning to generate samples from noise through infusion training 
   Bordes, F., Honari, S. and Vincent, P., 2017. arXiv preprint arXiv: 1703.06975.
46. Variational walkback: Learning a transition operator as a stochastic recurrent net 
   Goyal, A., Ke, N.R., Ganguli, S. and Bengio, Y., 2017. arXiv preprint arXiv: 1711.02282.
47. GSNs: generative stochastic networks 
   Alain, G., Bengio, Y., Yao, L., Yosinski, J., Thibodeau-Laufer, E., Zhang, S. and Vincent, P., 2016. Information and Inference: A Journal of the IMA, Vol 5 (2), pp. 210--249. Oxford University Press.
48. Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling 
   De Bortoli, V., Thornton, J., Heng, J. and Doucet, A., 2021. Advances in Neural Information Processing Systems (NeurIPS).
49. Denoising Diffusion Implicit Models [[link]] (https://openreview.net/forum?id=St1giarCHLP) 
   Song, J., Meng, C. and Ermon, S., 2021. International Conference on Learning Representations.
50. Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed 
   Luhman, E. and Luhman, T., 2021. arXiv e-prints, pp. arXiv--2101.
51. Score-based Generative Modeling in Latent Space 
   Vahdat, A., Kreis, K. and Kautz, J., 2021. Advances in Neural Information Processing Systems (NeurIPS).