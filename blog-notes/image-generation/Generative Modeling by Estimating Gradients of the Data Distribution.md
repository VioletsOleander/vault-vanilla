---
completed: true
---
Site: https://yang-song.net/blog/2021/score/
Date: 5 May 2021

This blog post focuses on a promising new direction for generative modeling. We can learn score functions (gradients of log probability density functions) on a large number of noise-perturbed data distributions, then generate samples with Langevin-type sampling. The resulting generative models, often called _score-based generative models_, has several important advantages over existing model families: GAN-level sample quality without adversarial training, flexible model architectures, exact log-likelihood computation, and inverse problem solving without re-training models. In this blog post, we will show you in more detail the intuition, basic concepts, and potential applications of score-based generative models.
> 本文为生成式建模提出新的思路
> 我们可以在大量的添加噪声的数据分布上学习得分函数 (对数概率密度函数的梯度)，然后使用 Langevin 型采样来生成样本
> 基于该范式得到的生成式模型称为 score-based 生成式模型，其优势包括:
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
> 使用 score matching 的关键挑战在于所估计的 score function 在低密度区域是不准确的 (从这句话来看，score matching 应该是用数据集的经验分布替代真实分布，因此数据集没有过多覆盖到的样本区域的估计就是不准确的)
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

After training our noise-conditional score-based model $\mathbf s_\theta(\mathbf x, i)$, we can produce samples from it by running Langevin dynamics for $i = L, L-1, \cdots, 1$ in sequence. This method is called **annealed Langevin dynamics** (defined by Algorithm 1 in [18] , and improved by  [19, 34] ), since the noise scale $\sigma_i$ decreases (anneals) gradually over time.
>  训练好 noise-conditional score-based model $\mathbf s_\theta(\mathbf x, i)$ 之后，我们可以通过 annealed Langevin dynamics 来从中抽样

>  Q: 如果 $\mathbf s_\theta(\mathbf x, i)$ 训练得好的话，直接用 Langevin dynamics，直接根据 $\mathbf s_\theta(\mathbf x, 0)$ 采样理论上应该也能收敛？
>  A: 理论上可以，但现实就是随机采样的样本不一定落在 $\mathbf s_\theta(\mathbf x, 0)$ 准确估计的区间内，也就是 $\mathbf s_\theta(\mathbf x, 0)$ 总是训练不好的，因此，需要依赖 annealed Langevin dynamics，一步步得到能够落在 $\mathbf s_\theta(\mathbf x, 0)$ 准确估计的区间内的起始样本，再进行最后的 Langevin dynamics 转化，才能确保得到准确服从目标分布样本

![](https://yang-song.net/assets/img/score/ald.gif)

Annealed Langevin dynamics combine a sequence of Langevin chains with gradually decreasing noise scales.
>  annealed Langevin dynamics 结合了一系列噪声尺度逐渐下降的 Langevin chains

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
>  当噪声尺度的数量 $L$ 趋近于**无限**，我们实际上是以不断 (连续) 增长的噪声水平扰动数据分布
>  在这种情况下，噪声扰动过程是一个连续时间的随机过程

![](https://yang-song.net/assets/img/score/perturb_vp.gif)

Perturbing data to noise with a continuous-time stochastic process.

How can we represent a stochastic process in a concise way? Many stochastic processes ([diffusion processes](https://en.wikipedia.org/wiki/Diffusion_process) in particular) are solutions of stochastic differential equations (SDEs). In general, an SDE possesses the following form:

$$
\mathrm d \mathbf x = \mathbf f(\mathbf x , t)\mathrm dt + g(t)\mathrm d\mathbf w\tag{8}
$$

where $\mathbf f(\cdot, t): \mathbb R^d \mapsto \mathbb R^d$ is a vector-valued function called the drift coefficient, $g(t)\in \mathbb R$ is a real-valued function called the diffusion coefficient, $\mathbf w$ denotes a standard [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion), and $\mathrm d\mathbf w$ can be viewed as infinitesimal white noise.

>  一个随机过程可以用随机微分方程 (SDE) 进行表示
>  该方程包含了两个驱动随机变量随着时间变化的因素: 一个可预测的、确定性的趋势 (漂移) 和一个不可预测的、随机性的波动 (扩散)
>  方程中:
>  - $\mathrm d\mathbf x$ 表示随机变量 $\mathbf x$ 在一个无穷小实现步长 $\mathrm dt$ 内的微小变化
>  - $\mathbf f(\mathbf x, t)\mathrm d t$ 是确定性漂移项，其中向量值函数 $\mathbf f(\cdot, t): \mathbb R^d \mapsto \mathbb R^d$ 称为漂移系数，漂移系数取决于当前的 $\mathbf x$ 和时间 $t$，乘以 $\mathrm dt$ 得到了由这种确定性趋势引发的无穷小变化
>  - $g(t)\mathrm d\mathbf w$ 是随机扩散项，其中实值函数 $g(t)\in \mathbb R$ 称为扩散系数，$\mathbf w$ 表示一个标准的布朗运动，$\mathrm d\mathbf w$ 可以视作一个无穷小的白噪声

>  因为 (无限的) 噪声扰动过程可以视作一个连续时间的随机过程，故可以用 SDE 描述该过程
>  也就是说将尺度为 $\sigma_t$ 的噪声扰动视作了从 $0$ 到 $t$ 的逐步连续小扰动的结果

> [!info] 布朗运动
> 布朗运动指的是微小粒子在流体（液体或气体）中进行的**永不停息、无规则的折线运动**，本质是由于液体或气体中的**大量分子（原子）在不停地做无规则的、剧烈的热运动**，这些分子不断地、不平衡地撞击着悬浮在其中的微小粒子所造成的
> 
> 在数学上，布朗运动被抽象为一个**连续时间、连续状态的随机过程**，通常被称为**维纳过程（Wiener Process）**，并用 $W(t)$ 或 $B(t)$ 表示
> 它具有以下几个核心数学性质：
> - **连续性：** 粒子路径是连续的，没有突然的跳跃。
> - **独立增量：** 在不重叠的时间区间内，粒子的位置变化是相互独立的。例如，$W(t_2​)−W(t_1​)$（在时间 $[t_1​,t_2​]$ 内的变化）与 $W(t_4​)−W(t_3​)$（在时间 $[t_3​,t_4​]$ 内的变化，如果 $[t_1​,t_2​]$ 和 $[t_3​,t_4​]$ 不重叠）是独立的。
> - **平稳增量：** 增量（即位置变化）的统计性质只取决于时间间隔的长度，而不取决于起始时间点。
> - **正态分布的增量：** 对于任意 $t>s$，增量 $W(t)−W(s)$ 服从均值为 0、方差为 $t−s$ 的正态分布。也就是说，$W(t)−W(s)\sim \mathcal N(0,t−s)$。
> - **起点通常设为零：** 通常假设 $W(0)=0$。
> - **几乎处处不可微：** 尽管布朗运动的路径是连续的，但它在任何点都是不可微的，这意味着它的速度是无限的、无法定义的。这是一个非常反直觉但重要的数学性质，反映了其高度的随机性和不规则性。

The solution of a stochastic differential equation is a continuous collection of random variables $\{\mathbf x(t)\}_{t\in [0, T]}$. These random variables trace stochastic trajectories as the time index $t$ grows from the start time $0$ to the end time $T$. Let $p_t(\mathbf x)$ denote the (marginal) probability density function of $\mathbf x (t)$. 
>  随机微分方程的解是一组连续随机变量，每个时刻都对应一个随机变量 $\mathbf x(t)$
>  随机变量 $\mathbf x(t)$ 的概率密度函数记作 $p_t(\mathbf x)$

Here $t\in[0, T]$ is analogous to $i=1,2,\cdots, L$ when we had a finite number of noise scales, and $p_t(\mathbf x)$ is analogous to $p_{\sigma_i}(\mathbf x)$. Clearly, $p_0(\mathbf x) = p(\mathbf x)$ is the data distribution since no perturbation is applied to data at $t=0$. After perturbing $p (\mathbf x)$ with the stochastic process for a sufficiently long time $T$, $p_T (\mathbf x)$ becomes close to a tractable noise distribution $\pi(\mathbf x)$, called a **prior distribution**. 
>  这里不同的时间步 $t$ 可以对应不同的噪声尺度 $\sigma_i$，故时间步 $t$ 下的概率密度函数 $p_t(\mathbf x)$ 可以对应噪声尺度 $\sigma_i$ 下的概率密度函数 $p_{\sigma_i}(\mathbf x)$
>  $t=0$ 时，随机过程没有开始 (对应于没有对数据进行扰动)，故 $p_0(\mathbf x) = p(\mathbf x)$
>  在时间步 $T$ 足够大 (扰动尺度足够大) 的情况下，$p_T(\mathbf x)$ 将接近一个可解的噪声分布 $\pi(\mathbf x)$，称为先验分布

We note that $p_T (\mathbf x)$ is analogous to $p_{\sigma_L} (\mathbf x)$ in the case of finite noise scales, which corresponds to applying the largest noise perturbation $\sigma_L$ to the data.
>  在有限的噪声尺度下，$p_T(\mathbf x)$ 类似于 $p_{\sigma_L}(\mathbf x)$，对应于直接对数据应用尺度为 $\sigma_L$ 的噪声扰动

The SDE in (8) is **hand designed**, similarly to how we hand-designed $\sigma_1 < \sigma_2 < \cdots < \sigma_L$ in the case of finite noise scales. 
>  Eq 8 的 SDE 同样需要人为设计，类似于我们人为设计不同的噪声尺度 $\sigma_1, \dots, \sigma_L$

There are numerous ways to add noise perturbations, and the choice of SDEs is not unique. For example, the following SDE

$$
\mathrm d\mathbf x = e^t\mathrm d\mathbf w\tag{9}
$$

perturbs data with a Gaussian noise of mean zero and exponentially growing variance, which is analogous to perturbing data with $\mathcal N(0, \sigma_1^2 I), \mathcal N(0, \sigma_2^2 I),\cdots, \mathcal N(0, \sigma_L^2I)$ when $\sigma_1 < \sigma_2 < \cdots < \sigma_L$ is a [geometric progression](https://en.wikipedia.org/wiki/Geometric_progression#:~:text=In%20mathematics%2C%20a%20geometric%20progression,number%20called%20the%20common%20ratio.). 

>  添加噪声尺度的方法不同，对应的 SDE 自然也不同
>  例如 Eq 9 的 SDE 没有漂移项，仅以扩散项扰动数据，扩散项是一个方差指数增长的均值为零的高斯噪声，这对应于使用以几何级数增长的噪声尺度序列来扰动数据 (几何级数的通项公式形式为 $a_n = a r^{n-1}$)

Therefore, the SDE should be viewed as part of the model, much like $\{\sigma_1, \sigma_2, \dots, \sigma_L\}$. In [21] , we provide three SDEs that generally work well for images: the Variance Exploding SDE (VE SDE), the Variance Preserving SDE (VP SDE), and the sub-VP SDE.
>  因此，使用 SDE 时，SDE 也应视作模型的设计组成的一部分，就像我们在设计时选择噪声尺度序列一样
>  对于图像处理的效果较好的 SDE 包括: 方差爆炸 (VE) SDE (确保数据最终变为纯噪声)，方差保持 SDE (VP SDE) (保持数据的整体方差，缓慢将其转换为噪声)、sub-VP SDE

### Reversing the SDE for sample generation
Recall that with a finite number of noise scales, we can generate samples by reversing the perturbation process with **annealed Langevin dynamics**, i.e., sequentially sampling from each noise-perturbed distribution using Langevin dynamics. 
>  对于有限数量的噪声尺度，我们可以通过 annealed Langevin dynamics，即顺序地用 Langevin dynamics 从各个扰动分布中采样，来逆转扰动过程

For infinite noise scales, we can analogously reverse the perturbation process for sample generation by using the reverse SDE.
>  类似地，对于有限的噪声尺度，我们可以用逆向 SDE 来逆转 SDE 的扰动过程，从而生成样本

![](https://yang-song.net/assets/img/score/denoise_vp.gif)

Generate data from noise by reversing the perturbation procedure.

Importantly, any SDE has a corresponding reverse SDE [35] , whose closed form is given by

$$
\mathrm d\mathbf x = [\mathbf f(\mathbf x, t) - g^2(t)\nabla_{\mathbf x}\log p_t(\mathbf x)]\mathrm dt + g(t)\mathrm d\mathbf w\tag{10}
$$

Here $\mathrm dt$ represents a negative infinitesimal time step, since the SDE (10) needs to be solved backwards in time (from $t=T$ to $t=0$). In order to compute the reverse SDE, we need to estimate $\nabla_{\mathbf x}\log p_t(\mathbf x)$, which is exactly the **score function** of $p_t(\mathbf x)$.

>  任意的 SDE 都有一个对应的逆 SDE，其形式如上 
>  其中 $\mathrm d t$ 表示一个负的无穷小时间步，因为逆 SDE 需要从 $t=T$ 到 $t=0$ 逆时间求解
>  要求解逆 SDE，需要估计 $\nabla_{\mathbf x}\log p_t(\mathbf x)$，即每个时间步下，概率密度函数 $p_t(\mathbf x)$ 的得分函数

>  逆 SDE 的概念类似于描述了从最终的变换结果再逐步变换回去的随机过程，即从 A 到 B 的随机过程总是存在一个从 B 到 A 的随机过程，两个方向的随机过程都有各自的 SDE 表示

![](https://yang-song.net/assets/img/score/sde_schematic.jpg)

Solving a reverse SDE yields a score-based generative model. Transforming data to a simple noise distribution can be accomplished with an SDE. It can be reversed to generate samples from noise if we know the score of the distribution at each intermediate time step.
>  使用 SDE 将数据转换为噪声后，如果我们知道每一个中间时间步中分布的得分，我们就可以求解逆 SDE，从噪声生成样本

### Estimating the reverse SDE with score-based models and score matching
Solving the reverse SDE requires us to know the terminal distribution $p_T(\mathbf x)$, and the score function $\nabla_{\mathbf x}\log p_t(\mathbf x)$. By design, the former is close to the prior distribution $\pi(\mathbf x)$ which is fully tractable. 
>  求解 reverser SDE 要求知道终止分布 $p_T(\mathbf x)$ 和得分函数 $\nabla_{\mathbf x}\log p_t(\mathbf x)$

In order to estimate $\nabla_{\mathbf x}\log p_t(\mathbf x)$, we train a **Time-Dependent Score-Based Model** $\mathbf s_\theta(\mathbf x, t)$, such that $\mathbf s_\theta(\mathbf x, t) \approx \nabla_{\mathbf x}\log p_t(\mathbf x)$. This is analogous to the noise-conditional score-based model $\mathbf s_\theta(\mathbf x, i)$ used for finite noise scales, trained such that $\mathbf s_\theta(\mathbf x, i) \approx \nabla_{\mathbf x} \log p_{\sigma_i}(\mathbf x)$.
>  为了估计得分函数 $\nabla_{\mathbf x}\log p_t(\mathbf x)$，我们训练 Time-Dependent Score-Based Model $\mathbf s_\theta(\mathbf x ,t)$，类似于 noise-conditional score-based model $\mathbf s_\theta(\mathbf x, i)$

Our training objective for $\mathbf s_\theta(\mathbf x, t)$ is a continuous weighted combination of Fisher divergences, given by

$$
\mathbb E_{t\in \mathcal U(0, T)}\mathbb E_{p_t(\mathbf x)}[\lambda(t)\|\nabla_{\mathbf x}\log p_t(\mathbf x) - \mathbf s_\theta(\mathbf x, t)\|_2^2]\tag{11}
$$

where $\mathcal U(0, T)$ denotes a uniform distribution over the time interval $[0, T]$, and $\lambda: \mathbb R \mapsto \mathbb R^+$ is a positive weighting function. 

>  time-dependent score-based model $\mathbf s_\theta(\mathbf x, t)$ 的训练目标是 Fisher 散度的连续加权
>  其中 $\mathcal U(0, T)$ 表示 $[0, T]$ 上的连续分布，$\lambda: \mathbb R \mapsto \mathbb R^+$ 为正加权函数
>  (外层的期望在实际的离散时间步下应该可以替换为简单的求平均，内层就和 naive score matching 的散度项几乎一致，差异仅在多了一个权重项 $\lambda(t)$)

Typically we use $\lambda(t) \propto 1/\mathbb E[\|\nabla_{\mathbf x(t)}\log p(\mathbf x(t)| \mathbf x(0))\|_2^2]$  to balance the magnitude of different score matching losses across time.
> $\lambda(t)$ 通常定义为 $\lambda(t) \propto 1/\mathbb E[\|\nabla_{\mathbf x(t)}\log p(\mathbf x(t)| \mathbf x(0))\|_2^2]$ ，在该定义下，$t$ 越大，$\nabla_{\mathbf x(t)}\log p(\mathbf x(t))$ 的范数一般越小 (这里忽略了条件项 $\mathbf x(0)$)，因为数据已经高度随机化，故改变它的梯度一般较小，故 $\lambda(t)$ 在 $t$ 更大时更大 
> 这样的设计平衡了不同时间步下的 score matching 损失的程度 (或许是因为再 $t$ 比较大的时候，扰动分布的得分函数比较容易学习，故损失的绝对值比较小，因此需要增大权重来平衡)

As before, our weighted combination of Fisher divergences can be efficiently optimized with score matching methods, such as denoising score matching [17] and sliced score matching [31] . 
>  上述的加权 Fisher 散度目标仍然可以通过 score matching 方法优化，例如 denoising score matching 和 sliced score matching

Once our score-based model $\mathbf s_\theta(\mathbf x, t)$ is trained to optimality, we can plug it into the expression of the reverse SDE in (10) to obtain an estimated reverse SDE.

$$
\mathrm d\mathbf x = [\mathbf f(\mathbf x, t) - g^2(t)\mathbf s_\theta(\mathbf x, t)]\mathrm dt + g(t)\mathrm d\mathbf w\tag{12}
$$

We can start with $\mathbf x(T)\sim \pi$, and solve the above reverse SDE to obtain a sample $\mathbf x(0)$. 

>  训练好 score-based model $\mathbf s_\theta(\mathbf x, t)$ 之后，就可以将其替换掉 Eq 10 的 reverse SDE 中的 $\nabla_{\mathbf x}\log p_t(\mathbf x)$，得到 Eq 12
>  根据 Eq 12，我们可以从 $\mathbf x(T)\sim \pi$ 开始，求解 reverse SDE，获得样本 $\mathbf x(0)$

Let us denote the distribution of $\mathbf x(0)$ obtained in such way as $p_\theta$. When the score-based model $\mathbf s_\theta(\mathbf x, t)$ is well-trained, we have $p_\theta \approx p_0$, in which case $\mathbf x (0)$ is an approximate sample from the data distribution $p_0$.
>  我们将 $\mathbf x(0)$ 从属的分布记作 $p_\theta$，如果模型训练得好，则 $p_\theta \approx p_0$，$\mathbf x(0)$ 就近似是从数据分布 $p_0$ 中获取的样本

When $\lambda(t) = g^2(t)$, we have an important connection between our weighted combination of Fisher divergences and the KL divergence from $p_0$ to $p_\theta$ under some regularity conditions [36] :

$$
\begin{align}
\mathrm {KL}(p_0(\mathbf x)\| p_\theta(\mathbf x)) \le \frac T 2\mathbb E_{t\in \mathcal U(0,T)}\mathbb E_{p_t(\mathbf x)}[\lambda(t)\|\nabla_{\mathbf x}\log p_t(\mathbf x) - \mathbf s_\theta(\mathbf x, t)\|_2^2] \\+ \mathrm {KL}(p_T\|\pi)\tag{13}
\end{align}
$$

>  如果令 $\lambda(t) = g^2(t)$，则加权的 Fisher 散度和 $p_0, p_\theta$ 之间的 KL 散度在某些正则化条件下具有以上的关系

Due to this special connection to the KL divergence and the equivalence between minimizing KL divergences and maximizing likelihood for model training, we call $\lambda(t) = g(t)^2$ the **likelihood weighting function**. 
>  因为这一层关系，且考虑到优化 KL 散度等价于极大似然训练，我们称 $\lambda(t) = g(t)^2$ 为似然加权函数

Using this likelihood weighting function, we can train score-based generative models to achieve very high likelihoods, comparable or even superior to state-of-the-art autoregressive models [36] .
>  使用 $\lambda(t) = g(t)^2$ 时，训练出的 score-based generative 可以达到很高的似然，与 SOTA 的自回归模型相比拟

### How to solve the reverse SDE
By solving the estimated reverse SDE with numerical SDE solvers, we can simulate the reverse stochastic process for sample generation. Perhaps the simplest numerical SDE solver is the [Euler-Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method). When applied to our estimated reverse SDE, it discretizes the SDE using finite time steps and small Gaussian noise. 
>  生成样本需要求解 reverse SDE 过程，我们可以通过数值 SDE 求解器来求解
>  最简单的 SDE 求解器是 Euler-Maruyama 方法，该方法将用有限的时间步和小的高斯噪声离散化 SDE

Specifically, it chooses a small negative time step $\Delta t \approx 0$, initializes $t\leftarrow T$, and iterates the following procedure until $t\approx 0$:

$$
\begin{align}
\Delta \mathbf x &\leftarrow [\mathbf f(\mathbf x, t) - g^2(t)\mathbf s_\theta(\mathbf x, t)]\Delta t + g(t)\sqrt {|\Delta t|}\mathbf z_t\\
\mathbf x & \leftarrow \mathbf x + \Delta \mathbf x\\
t & \leftarrow  t + \Delta t
\end{align}
$$

Here $\mathbf z_t \sim \mathcal N(0, I)$. 

>  具体地说，它会选择一个小的离散时间步 $\Delta t \approx 0$，初始化 $t \leftarrow T$ (从扩散过程的末尾开始)，迭代以下过程直到 $t \approx 0$:
>  1. 计算 $\Delta \mathbf x$ ，公式中的 $[\mathbf f(\mathbf x, t) - g^2(t)\mathbf s_\theta(\mathbf x, t)]\Delta t$ 是漂移项，其中 $\mathbf f(\mathbf x, t)$ 是逆向 SDE 的漂移系数；公式中的 $g(t)\sqrt {|\Delta t|}\mathbf z_t$ 是噪声/扩散项，其中 $g(t)$ 为扩散系数，$\sqrt {|\Delta t|}$ 表示在 SDE 中，噪声尺度和时间步长的平方根成比例，$\mathbf z_t \sim \mathcal N(0, I)$ 表示小的噪声向量
>  2. 更新 $\mathbf x$ (加上 $\Delta \mathbf x$，获得新的，稍微去噪后的数据点)
>  3. 更新 $t$ (向 $t=0$ 迈进)

The Euler-Maruyama method is qualitatively similar to Langevin dynamics—both update $\mathbf x$ by following score functions perturbed with Gaussian noise.
>  Euler-Maruyama 方法在性质上和 Langevin dynamics 相似
>  Langevin dynamics 通过沿着得分函数的方向移动样本，并用高斯噪声进行扰动以更新样本
>  Euler-Maruyama 方法中，涉及得分函数的项作为扩散项引导样本，且同样用高斯噪声进行了扰动

Aside from the Euler-Maruyama method, other numerical SDE solvers can be directly employed to solve the reverse SDE for sample generation, including, for example, [Milstein method](https://en.wikipedia.org/wiki/Milstein_method), and [stochastic Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_\(SDE\)). 
>  除了 Euler-Maruyama，其他的数值 SDE solver 也可以直接被用于求解 reverse SDE，例如 Milstein method, stochastic Runge-Kutta methods

In  [21] , we provided a reverse diffusion solver similar to Euler-Maruyama, but more tailored for solving reverse-time SDEs. More recently, authors in [37] introduced adaptive step-size SDE solvers that can generate samples faster with better quality.
>  我们在 [21] 提出了类似 Euler-Maruyama 的 solver，更适用于求解  reverse-time SDE
>  [37] 引入了自适应步长 SDE solver，可以以更高质量更快地生成样本

In addition, there are two special properties of our reverse SDE that allow for even more flexible sampling methods:

- We have an estimate of $\nabla_{\mathbf x}\log p(\mathbf x)$ via our time-dependent score-based model $\mathbf s_\theta(\mathbf x, t)$.
- We only care about sampling from each marginal distribution $p_t(\mathbf x)$. Samples obtained at different time steps can have arbitrary correlations and do not have to form a particular trajectory sampled from the reverse SDE.

>  我们的 reverse SDE 具有两个特殊性质，允许更灵活的采样方法
>  - 我们有对 $\nabla_{\mathbf x}\log p(\mathbf x)$ 的估计
>  - 我们仅关心从每个边际分布 $p_t(\mathbf x)$ 的采样，在不同时间步获取的样本可以有任意的相关性，并不需要形成从反向 SDE 中采样的特定轨迹

As a consequence of these two properties, we can apply MCMC approaches to fine-tune the trajectories obtained from numerical SDE solvers. Specifically, we propose **Predictor-Corrector samplers**. The **predictor** can be any numerical SDE solver that predicts $\mathbf x(t + \Delta t) \sim p_{t + \Delta t}(\mathbf x)$ from an existing sample $\mathbf x(t) \sim p_t(\mathbf x)$. The **corrector** can be any MCMC procedure that solely relies on the score function, such as Langevin dynamics and Hamiltonian Monte Carlo.
>  因此，我们可以对从数值 SDE solvers 中获得的轨迹用 MCMC 方法微调
>  具体地说，我们提出了 Predictor-Corrector samplers
>  predictor 可以是任意的数值 SDE solver，从现有的样本 $\mathbf x(t) \sim p_t(\mathbf x)$ 预测下一个时刻的样本 $\mathbf x(t + \Delta t) \sim p_{t+\Delta t}(\mathbf x)$
>  corrector 可以是任意 MCMC 仅依赖于 score function 的 MCMC 过程，例如 Langevin dynamics 和 Hamiltonian MC

At each step of the Predictor-Corrector sampler, we first use the predictor to choose a proper step size $\Delta t < 0$, and then predict $\mathbf x(t + \Delta t)$ based on the current sample $\mathbf x(t)$. Next, we run several corrector steps to improve the sample $\mathbf x(t + \Delta t)$ according to our score-based model $\mathbf s_\theta(\mathbf x, t + \Delta t)$, so that $\mathbf x(t + \Delta t)$ becomes a higher-quality sample from $p_{t+\Delta t}(\mathbf x)$.
>  Predictor-Corrector sampler 的每一步中:
>  predictor 首先选择一个合适的步长 $\Delta t < 0$，然后基于当前样本 $\mathbf x(t)$ 预测 $\mathbf x(t + \Delta t)$
>  运行多次的 corrector steps ，基于 $\mathbf s_\theta(\mathbf x, t + \Delta t)$ 来提升样本 $\mathbf x(t + \Delta t)$，使得 $\mathbf x(t + \Delta t)$ 接近从 $p_{t + \Delta t}(\mathbf x)$ 采样的高质量样本

With Predictor-Corrector methods and better architectures of score-based models, we can achieve **state-of-the-art** sample quality on CIFAR-10 (measured in FID [38] and Inception scores [12] ), outperforming the best GAN model to date (StyleGAN2 + ADA [39] ).
>  Predictor-Corrector methods + 更好架构的 score-based model，可以在 CIFAR-10 上取得 SOTA 的样本质量

|        Method         |  FID ↓   | Inception score ↑ |
| :-------------------: | :------: | :---------------: |
| StyleGAN2 + ADA  [39] |   2.92   |       9.83        |
|      Ours  [21]       | **2.20** |     **9.89**      |

The sampling methods are also scalable for extremely high dimensional data. For example, it can successfully generate high fidelity images of resolution 1024×1024.
>  这一采样方法也可以拓展到非常高维的数据

![](https://yang-song.net/assets/img/score/ffhq_1024.jpeg)

1024 x 1024 samples from a score-based model trained on the FFHQ dataset.

Some additional (uncurated) samples for other datasets (taken from this [GitHub repo](https://github.com/yang-song/score_sde)):

![](https://yang-song.net/assets/img/score/bedroom.jpeg)

256 x 256 samples on LSUN bedroom.

![](https://yang-song.net/assets/img/score/celebahq_256.jpg)

256 x 256 samples on CelebA-HQ.

### Probability flow ODE
Despite capable of generating high-quality samples, samplers based on Langevin MCMC and SDE solvers do not provide a way to compute the exact log-likelihood of score-based generative models. 
>  虽然基于 Langevin MCMC 和 SDE solver 的采样器可以生成高质量样本，但它们无法计算样本的精确对数似然

Below, we introduce a sampler based on ordinary differential equations (ODEs) that allow for exact likelihood computation. In [21] , we show $t$ is possible to convert any SDE into an ordinary differential equation (ODE) without changing its marginal distributions $\{p_t(\mathbf x)\}_{t\in [0, T]}$. Thus by solving this ODE, we can sample from the same distributions as the reverse SDE. 
>  我们提出一个基于常微分方程 (ODE) 的采样器，以实现精确似然计算
>  我们可以将任意的 SDE 都转化为 ODE，转化后的 ODE 和 SDE 在任意时间步都具有相同的边际分布 $\{p_t(\mathbf x)\}_{t\in [0, T]}$
>  因此，通过求解 ODE，从 $t = T$ 开始，通过 ODE 演化到 $t = 0$ 时获得的样本和我们通过求解 reverse SDE 得到的样本从属于相同的分布

The corresponding ODE of an SDE is named **probability flow ODE** [21] , given by

$$
\mathrm d\mathbf x = \left[\mathbf f(\mathbf x, t) - \frac 1 2g^2(t)\nabla_{\mathbf x}\log p_t(\mathbf x)\right]\mathrm dt\tag{14}
$$

>  SDE 对应的这个 ODE 称为概率流 ODE，形式如 Eq 14，其中
>  - $\mathrm d \mathbf x$ 表示数据点 $\mathbf x$ 的微小变化
>  - $\mathbf f(\mathbf x, t)$ 是原始 SDE 中的漂移系数，它决定了数据在没有噪声情况下的趋势或方向
>  - $g^2(t)$ 是原始 SDE 中的扩散系数的平方，在原始 SDE 中控制噪声项的强度
>  - $\nabla_{\mathbf x}\log p_t(\mathbf x)$ 为得分函数
>  - $\mathrm d t$ 表示微小的时间步长
>  可以看到，概率流 ODE 中没有显式的随机噪声项

The following figure depicts trajectories of both SDEs and probability flow ODEs. Although ODE trajectories are noticeably smoother than SDE trajectories, they convert the same data distribution to the same prior distribution and vice versa, sharing the same set of marginal distributions $\{p_t(\mathbf x)\}_{t\in [0, T]}$. 
>  图中展示了 SDE 和 probability flow ODE 的轨迹
>  ODE 的轨迹比 SDE 的轨迹显著地更平滑，但即便轨迹不同，probability flow ODE 和 SDE 都将相同的数据分布转换为相同的先验分布，反之亦然，二者具有相同的一组边际分布 $\{p_t(\mathbf x)\}_{t\in [0, T]}$ 

In other words, trajectories obtained by solving the probability flow ODE have the same marginal distributions as the SDE trajectories.
>  换句话说，通过求解 probability flow ODE 获得的轨迹和求解 SDE 获得的轨迹具有相同的边际分布

![](https://yang-song.net/assets/img/score/teaser.jpg)

We can map data to a noise distribution (the prior) with an SDE, and reverse this SDE for generative modeling. We can also reverse the associated probability flow ODE, which yields a deterministic process that samples from the same distribution as the SDE. Both the reverse-time SDE and probability flow ODE can be obtained by estimating score functions.

This probability flow ODE formulation has several unique advantages.
>  相较于 SDE, probability flow ODE 的形式具有几个独特优势

When $\nabla_{\mathbf x}\log p_t(\mathbf x)$ is replaced by its approximation $\mathbf s_\theta(\mathbf x, t)$, the probability flow ODE becomes a special case of a neural ODE [40] . 
>  当我们用 score-based model 替换 probability ODE 中的 $\nabla_t \log p_t(\mathbf x)$ 时，probability flow ODE 就成为 neural ODE 的特例 (用神经网络来参数化微分方程的右侧)

> [! info] Neural ODE
> Neural ODE 是一类新的神经网络架构，它将深度学习的层堆叠概念推广到了连续深度
> Neural ODE 的核心思想是不将网络看作离散层的堆叠，而是一个连续的动力学系统，即隐藏状态 $\mathbf h(t)$ 随着一个连续的时间变量 $t$ 变化 (而不是随着离散的层数变化)
> 
> Neural ODE 中，神经网络用于建模隐藏状态 $\mathbf h(t)$ 随着时间 $t$ 变化的变化率，公式为
> $$ \frac {\mathrm d \mathbf h(t)}{\mathrm d t} = f(\mathbf h(t), t, \theta) $$
>  这个方程就是一个常微分方程 (ODE)

In particular, it is an example of continuous normalizing flows [41] , since the probability flow ODE converts a data distribution $p_0(\mathbf x)$ to a prior noise distribution $p_T(\mathbf x)$ (since it shares the same marginal distributions as the SDE) and is fully invertible.
>  进一步地，probability flow ODE 也是连续归一化流的特例，因为 probability flow ODE 也是将数据分布 $p_0(\mathbf x)$ 转化为一个先验噪声分布 $p_T(\mathbf x)$，且这个过程是完全可逆的

> [!info] Continuous Normalizing Flows
> 归一化流是一类生成模型，其思想是将一个简单的基础分布 (例如正态分布) (中的样本 $z_0$) 通过一系列**可逆且可微分**的映射 $f_1, f_2, \dots, f_k$ 转换为目标分布 (中的样本 $z_t$)
> 只要每个映射都是可逆且可微分的，其雅可比行列式就能计算，进而被转换后的数据点在目标分布下的概率密度/似然就是可计算的
> 
> 连续归一化流将离散的、堆叠的变换推广到连续，它不通过一系列离散的函数映射来转换分布，而是通过 ODE 来转换分布 (或者说定义分布变换)
> 其核心思想是将基础分布/样本 $z_0$ 到目标分布/样本 $z_t$ 的转换看作一个连续的时间演化过程，该过程由向量场 $v(z(t), t)$ 定义，$v(z(t), t)$ 描述了 $z(t)$ 在 $t$ 时刻的变化率
> 
> 用神经网络建模该向量场，就得到了 Neural ODE 模型
> $$ \frac {\mathrm d z(t)}{ \mathrm d t} = v(z(t), t) $$
> 
> 连续归一化流在变化下的概率密度是可计算的
> 根据瞬时变量变换定理，连续随机变量的对数概率密度的变化等于其变化率的雅可比矩阵的迹的负值
> $$\frac {\mathrm d \log p(z(t))}{\mathrm d t} = - \mathrm {tr}\left(\frac {\partial v(z(t), t)}{\partial z(t)}\right) $$
> 因此，为了计算最终数据 $z_T$ 的概率密度，我们根据上式，对时间 $0$ 到 $T$ 进行积分即可
> $$ \log p(z_T) = \log p_0(z_0) - \int_0^T \mathrm{tr}\left( \frac {\partial v(z(t), t)}{\partial z(t)}\right)  $$
> 实践中，这一积分可以由数值 ODE solver 计算

As such, the probability flow ODE inherits all properties of neural ODEs or continuous normalizing flows, including exact log-likelihood computation. Specifically, we can leverage the instantaneous change-of-variable formula (Theorem 1 in [40] , Equation (4) in [41] ) to compute the unknown data density $p_0$ from the known prior density $p_T$ with numerical ODE solvers.
>  因此，probability flow ODE 继承了 neural ODE 和连续规范化流的所有性质，包括了精确的对数似然计算
>  故我们可以利用瞬时变量变换定理计算，从先验密度 $p_T$ 计算未知数据密度 $p_0$

In fact, our model achieves the **state-of-the-art** log-likelihoods on uniformly dequantized $^4$ CIFAR-10 images [21] , **even without maximum likelihood training**.

| Method  | Negative log-likelihood (bits/dim) ↓ |
| :-----: | :----------------------------------: |
| RealNVP |                 3.49                 |
| iResNet |                 3.45                 |
|  Glow   |                 3.35                 |
| FFJORD  |                 3.40                 |
| Flow++  |                 3.29                 |
|  Ours   |               **2.99**               |

When training score-based models with the **likelihood weighting** we discussed before, and using **variational dequantization** to obtain likelihoods on discrete images, we can achieve comparable or even superior likelihood to the state-of-the-art autoregressive models (all without any data augmentation) [36] .

> [!info] Variational Dequantization
> 许多生成模型都基于连续概率分布来建模数据，目标是学习到真实的高维概率密度函数 $p(\mathbf x)$
> 但连续模型无法直接用于离散数据，原因包括:
> - 概率密度峰值问题，模型将倾向于在离散值上放置无限高的概率尖峰，例如一个像素值是 128，模型会给 128 非常高的密度，但 127.9，128.1 这些 “不存在” 的值概率密度则接近 0，分布会非常不自然
> - 梯度问题，离散值之间没有平滑的变化，无法计算梯度
> - 似然计算问题，概率质量函数不等同于概率密度函数，无法直接迁移似然计算方法
> 
>  去量化方法将离散的数据转化为连续的数据，便于模型处理
>  最简单的方法是均匀去量化，对每个离散数据点添加 $[0, 1)$ 内的均匀噪声，将每个离散值 “扩展为” 一个连续的区间，模型学习在这个连续的区间上分配概率密度
>  变分去量化的思想是不采用固定的均匀分布噪声，而是让模型学习一个最优的噪声分布来量化数据，它将添加到离散数据上的噪声视作一个隐变量 $u$，模型学习一个变分分布 $q(u\mid x)$ 来近似真实的条件分布 $p(u\mid x)$

|       Method       | Negative log-likelihood (bits/dim) ↓ on CIFAR-10 | Negative log-likelihood (bits/dim) ↓ on ImageNet 32x32 |
| :----------------: | :----------------------------------------------: | :----------------------------------------------------: |
| Sparse Transformer |                     **2.80**                     |                           -                            |
| Image Transformer  |                       2.90                       |                          3.77                          |
|        Ours        |                       2.83                       |                        **3.76**                        |

### Controllable generation for inverse problem solving
Score-based generative models are particularly suitable for solving inverse problems. At its core, inverse problems are same as Bayesian inference problems. Let $\mathbf x$ and $\mathbf y$ be two random variables, and suppose we know the forward process of generating $\mathbf y$ from $\mathbf x$, represented by the transition probability distribution $p(\mathbf y\mid \mathbf x)$. The inverse problem is to compute $p (\mathbf x \mid \mathbf y)$. From Bayes’ rule, we have $p (\mathbf x\mid \mathbf y)=p (\mathbf x) p (\mathbf y\mid \mathbf x)/\int p (\mathbf x) p (\mathbf y\mid \mathbf x) d\mathbf x$. 
>  score-based 模型很适合求解逆问题
>  逆问题的实质和贝叶斯推理问题相同，假设我们知道从随机变量 $\mathbf x$ 生成随机变量 $\mathbf y$ 的前向过程，用转移分布 $p(\mathbf y\mid \mathbf x)$ 来表示 (例如加噪，但不限于加噪)，对应的逆问题就是条件分布 $p(\mathbf x \mid\mathbf y)$
>  根据贝叶斯规则，有 $p(\mathbf x \mid \mathbf y) = p(\mathbf x) p(\mathbf y \mid \mathbf x) / \int p(\mathbf x)p(\mathbf y\mid \mathbf x) d\mathbf x$

This expression can be greatly simplified by taking gradients with respect to $\mathbf x$ on both sides, leading to the following Bayes’ rule for score functions:

$$
\nabla_{\mathbf x} \log p(\mathbf x\mid \mathbf y) = \nabla_{\mathbf x}\log p(\mathbf x) + \nabla_{\mathbf x}\log p(\mathbf y\mid \mathbf x)\tag{15}
$$

>  对贝叶斯公式两边取对数，然后求梯度，可以得到 Eq 15
>  Eq 15 描述了前向分布和逆向分布的得分函数之间的关系

Through score matching, we can train a model to estimate the score function of the unconditional data distribution, i.e., $\mathbf s_\theta \approx \nabla_{\mathbf x}\log p(\mathbf x)$. This will allow us to easily compute the posterior score function $\nabla_{\mathbf x}\log p(\mathbf x \mid \mathbf y)$ from the known forward process $p(\mathbf y \mid \mathbf x)$ via equation (15), and sample from it with Langevin-type sampling [21] . 
>  通过 score mathcing，我们可以训练近似数据分布得分函数的模型 $\mathbf s_\theta \approx \nabla_{\mathbf x}\log p(\mathbf x)$，进而可以轻易计算后验分布的得分函数 $\nabla_{\mathbf x}\log p(\mathbf x \mid \mathbf y)$，然后根据它进行 Langevin 采样

A recent work from UT Austin  [29] has demonstrated that score-based generative models can be applied to solving inverse problems in medical imaging, such as accelerating magnetic resonance imaging (MRI). Concurrently in [42] , we demonstrated superior performance of score-based generative models not only on accelerated MRI, but also sparse-view computed tomography (CT). We were able to achieve comparable or even better performance than supervised or unrolled deep learning approaches, while being more robust to different measurement processes at test time.

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
I started working on score-based generative modeling since 2019, when I was trying hard to make score matching scalable for training deep energy-based models on high-dimensional datasets. My first attempt at this led to the method sliced score matching [31] . Despite the scalability of sliced score matching for training energy-based models, I found to my surprise that Langevin sampling from those models fails to produce reasonable samples even on the MNIST dataset. I started investigating this issue and discovered three crucial improvements that can lead to extremely good samples: (1) perturbing data with multiple scales of noise, and training score-based models for each noise scale; (2) using a U-Net architecture (we used RefineNet since it is a modern version of U-Nets) for the score-based model; (3) applying Langevin MCMC to each noise scale and chaining them together. With those methods, I was able to obtain the state-of-the-art Inception Score on CIFAR-10 in [18] (even better than the best GANs!), and generate high-fidelity image samples of resolution up to 256×256 in [19] .
>  高质量样本生成的三个关键改进:
>  1. 对数据进行多尺度噪声扰动，为各个噪声尺度训练 score-based model
>  2. U-Net 架构
>  3. 对每个噪声尺度都应用 Langevin MCMC，然后将它们结合为一条链

The idea of perturbing data with multiple scales of noise is by no means unique to score-based generative models though. It has been previously used in, for example, [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing), annealed importance sampling [43] , diffusion probabilistic models [44] , infusion training [45] , and variational walkback [46] for generative stochastic networks [47] . Out of all these works, diffusion probabilistic modeling is perhaps the closest to score-based generative modeling. Diffusion probabilistic models are hierarchical latent variable models first proposed by [Jascha](http://www.sohldickstein.com/) and his colleagues [44] in 2015, which generate samples by learning a variational decoder to reverse a discrete diffusion process that perturbs data to noise. Without awareness of this work, score-based generative modeling was proposed and motivated independently from a very different perspective. Despite both perturbing data with multiple scales of noise, the connection between score-based generative modeling and diffusion probabilistic modeling seemed superficial at that time, since the former is trained by score matching and sampled by Langevin dynamics, while the latter is trained by the evidence lower bound (ELBO) and sampled with a learned decoder.

In 2020, [Jonathan Ho](http://www.jonathanho.me/) and colleagues  [20] significantly improved the empirical performance of diffusion probabilistic models and first unveiled a deeper connection to score-based generative modeling. They showed that the ELBO used for training diffusion probabilistic models is essentially equivalent to the weighted combination of score matching objectives used in score-based generative modeling. Moreover, by parameterizing the decoder as a sequence of score-based models with a U-Net architecture, they demonstrated for the first time that diffusion probabilistic models can also generate high quality image samples comparable or superior to GANs.
>  训练 diffusion 的 ELBO 本质上等价于训练 score-based model 的 score matching objective 的加权求和

Inspired by their work, we further investigated the relationship between diffusion models and score-based generative models in an ICLR 2021 paper  [21] . We found that the sampling method of diffusion probabilistic models can be integrated with annealed Langevin dynamics of score-based models to create a unified and more powerful sampler (the Predictor-Corrector sampler). By generalizing the number of noise scales to infinity, we further proved that score-based generative models and diffusion probabilistic models can both be viewed as discretizations to stochastic differential equations determined by score functions. This work bridges both score-based generative modeling and diffusion probabilistic modeling into a unified framework.
>  diffusion model 的采样方法可以与 annealed Langevin dynamics 结合，得到更统一的采样器: Predictor-Corrector sampler
>  将噪声尺度泛化到无限，score-based model 和 diffusion model 都可以视作由 score functions 决定的 SDE 的离散化，因此两类模型是统一的

Collectively, these latest developments seem to indicate that both score-based generative modeling with multiple noise perturbations and diffusion probabilistic models are different perspectives of the same model family, much like how [wave mechanics](https://en.wikipedia.org/wiki/Wave_mechanics) and [matrix mechanics](https://en.wikipedia.org/wiki/Matrix_mechanics) are equivalent formulations of quantum mechanics in the history of physics $^5$ . The perspective of score matching and score-based models allows one to calculate log-likelihoods exactly, solve inverse problems naturally, and is directly connected to energy-based models, Schrödinger bridges and optimal transport [48] . The perspective of diffusion models is naturally connected to VAEs, lossy compression, and can be directly incorporated with variational probabilistic inference. This blog post focuses on the first perspective, but I highly recommend interested readers to learn about the alternative perspective of diffusion models as well (see [a great blog by Lilian Weng](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html)).
>  从 score function 的角度出发，可以精确计算对数似然，自然地求解逆问题

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
>  score-based model 的两个缺点: 采样太慢，仅适用于连续分布

The first challenge can be partially solved by using numerical ODE solvers for the probability flow ODE with lower precision (a similar method, denoising diffusion implicit modeling, has been proposed in [49] ). It is also possible to learn a direct mapping from the latent space of probability flow ODEs to the image space, as shown in [50] . However, all such methods to date result in worse sample quality.

The second challenge can be addressed by learning an autoencoder on discrete data and performing score-based generative modeling on its continuous latent space [28, 51] . Jascha’s original work on diffusion models  [44] also provides a discrete diffusion process for discrete data distributions, but its potential for large scale applications remains yet to be proven.

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

# Supplementary
## 瞬时变量变换定理
考虑一个随机变量 $\mathbf z$ 在时间上的演化，由常微分方程 $\frac {\mathrm d \mathbf z(t)}{\mathrm t} = \mathbf v(\mathbf z(t), t)$ 定义，其中 $\mathbf v(\mathbf z(t), t)$ 为向量场，表示 $\mathbf z$ 在 $t$ 时刻的瞬时速度

记 $t$ 时刻，$\mathbf z(t)$ 服从的概率密度函数为 $p(\mathbf z, t)$，概率密度函数也是时间的函数，它随着时间的变化可以用连续性方程描述，形式为

$$
\frac {\partial p(\mathbf z, t)}{\partial t} + \nabla\cdot (p(\mathbf z, t)\mathbf v(\mathbf z, t)) = 0
$$

其中，$\frac {\partial p(\mathbf z, t)}{\partial t}$ 是概率密度函数对时间的偏导数，它表示在某个固定空间点 $\mathbf z$，概率密度函数随时间的变化率，如果为正，说明这个点上的概率密度在增加 (有更多 "概率粒子" 堆积起来)，如果为负，说明这个点上的概率密度在减少 (“概率粒子” 正在离开)

$\mathbf v(\mathbf z, t)$ 为速度场，表示在空间位置 $\mathbf z$ 和时间 $t$ 时，“概率粒子” 的瞬时速度 (就好像水流中每个位置的水都有流速)

$p(\mathbf z, t)\mathbf v(\mathbf z, t)$ 称为概率流密度 (通量密度=密度 x 速度，概率密度就类似 “概率粒子” 的密度，速度场表征了 “概率粒子” 的流速，通量密度是一个向量，表示每单位时间、每单位面积通过的物理量，且具有方向)，它表示单位时间内穿过单位面积的 “概率量” 

$\nabla$ 为散度运算符，散度的直观理解就是源和汇，其核心的物理意义就是衡量一个向量场在空间某一点的 “发散” 或 “汇聚” 程度
三维直角坐标系中，散度算子 $\nabla \cdot$ 作用于一个向量场 $\mathbf F = F_x \vec i + F_y \vec j + F_z \vec k$ ，得到散度场 (标量场)，散度的数学定义是: $\mathrm {div} \mathbf F = \nabla \cdot \mathbf F = \frac {\partial F_x}{\partial x} + \frac {\partial F_y}{\partial y} + \frac {\partial F_z}{\partial z}$ 
直观上看，定义中的每个加数都是向量场中某一点在特定方向上的偏导数，表示了场在这个方向的变化率
例如考虑 $x$ 方向，如果 $\frac {\partial F_x}{\partial x}$ 为正，说明分量 $F_x$ 沿着 $x$ 轴增加，意味着有更多 $x$ 方向的流量向正 $x$ 方向 “流出”，如果为 $\frac {\partial F_x}{\partial x}$ 为负，说明分量 $F_x$ 沿着 $x$ 轴减少，意味着有更多来自正 $x$ 方向的流量向负 $x$ 方向 “流入”，把这些变化率加起来，就是这个点的 “净流入” 或 “净流出” 量 (注意流入和流出都是针对 “正方向“ 定义的)
如果向量场中某一点的散度为正 ($\mathrm {div} > 0$)，说明这一点为源，它在向外发散，如果某一点的散度为负 ($\mathrm {div} < 0$)，说明这一点为汇，它在向内汇聚

散度描述了局部的性质，衡量了一个点 (单位体积) 上向量场的发散和汇聚，散度实质是通量的 “体密度”，即单位体积内的通量净流出量 (把散度看作源汇的密度，就像质量密度是单位体积内的质量一样)
净通量描述了整体的性质，衡量了整个封闭表面上有多少向量场穿过，是净流入还是净流出
因为一个封闭区域的净通量，一定是由这个区域内所有点的局部发散或汇聚行为累积而成的，故散度和净通量显然存在关系，它们的关系由高斯散度定理描述:

$$
\int \int_S \mathbf F\cdot d\mathbf S = \int\int\int_V (\nabla\cdot \mathbf F)dV
$$

其中，$\int\int_S \mathbf F\cdot d\mathbf S$ 表示向量场穿过封闭曲面 $\mathbf S$ 的净通量，$d\mathbf S$ 是一个向量面积元，其方向是曲面的外法线方向
$\int\int\int_V (\nabla\cdot \mathbf F) dV$ 表示向量场 $\mathbf F$ 的散度 $\nabla \cdot \mathbf F$ 在曲面所包围体积 $V$ 上的体积分

一个直观的例子: 在房间里点一只香烟，香烟这个 (房间内) 的点就是一个源，房间里其他的空气点不是源也不是汇 (流入和流出量平衡)，房间里的排气扇这个点就是一个汇
净通量将房间视作一个整体，考虑多少烟雾/空气净地穿过这个房间

回到连续性方程，在方程中，散度算子作用于速度场 (向量场) $\mathbf v(\mathbf z, t)$，也可以视作作用于通量密度场 (向量场) $p(\mathbf z, t)\mathbf v(\mathbf z, t)$
$\nabla \cdot (p(\mathbf z, t)\mathbf v(\mathbf z, t))$ 就计算了该点处通量密度 (概率流密度) 的散度，表征了在 $\mathbf z$ 这一点处，概率流密度的源/汇情况 (或者直接说成概率密度函数在这一个时刻下的瞬时变化情况)

连续性方程

$$\frac {\partial p (\mathbf z, t)}{\partial t}  =-  \nabla\cdot (p (\mathbf z, t)\mathbf v (\mathbf z, t))$$

告诉我们: 在固定点 $\mathbf z$ 上，概率密度函数的变化率严格等于概率流密度在该点处散度的负值 (源的概率密度减小，汇的概率密度增大，非源非汇，不变化)

连续性方程是对 “局部守恒” 的数学表达: 任何一个固定点上**守恒量**密度的变化，都必须通过该点的 “流” (通量) 的净流入和净流出表示，散度是 “流” 的净流入或流出的局部强度

回到瞬时变量变换定理，我们展开连续性方程中的散度项，得到:

$$
\frac {\partial p(\mathbf z, t)}{\partial t} + (\nabla p(\mathbf z, t))\cdot\mathbf v(\mathbf z, t)+ p(\mathbf z, t)\cdot(\nabla\cdot  \mathbf v(\mathbf z, t))= 0
$$

我们开始考虑对数概率密度函数 $\log p(\mathbf z(t), t)$ 对 $t$ 的全导数，根据链式法则:

$$
\frac {d}{dt}\log p(\mathbf z(t), t) = \frac {\partial \log p(\mathbf z(t), t)}{\partial t} + \sum_i\frac {\partial \log p(\mathbf z, t)}{\partial z_i}\frac {d z_i}{d t}
$$

其中 $\frac {dz_i}{dt} = v_i(\mathbf z, t)$，故

$$
\frac {d}{dt}\log p(\mathbf z(t), t) = \frac {\partial \log p(\mathbf z(t), t)}{\partial t} + \nabla_{\mathbf z}{\log p(\mathbf z, t)}\cdot \mathbf v(\mathbf z, t)
$$

因为 $\frac {\partial \log p}{\partial t} = \frac 1 p \frac {\partial p}{\partial t}$，以及 $\nabla \log p = \frac 1 p \nabla p$，故

$$
\begin{align}
\frac {d}{dt}\log p(\mathbf z(t), t) &= \frac 1 {p(\mathbf z, t)}\frac {\partial p(\mathbf z, t)}{\partial t} + \frac 1 {p(\mathbf z, t)}\nabla p(\mathbf z , t) \cdot\mathbf v(\mathbf z, t)\\
p(\mathbf z, t)\frac {d}{dt}\log p(\mathbf z(t), t)&=\frac {\partial p(\mathbf z, t)}{\partial t} + \nabla p(\mathbf z, t)\cdot \mathbf v(\mathbf z, t)
\end{align}
$$

根据连续性方程，等式右边可以替换为 $- p(\mathbf z, t)\cdot (\nabla \cdot \mathbf v(\mathbf z, t))$，故

$$
p (\mathbf z, t)\frac {d}{dt}\log p (\mathbf z (t), t) =-p(\mathbf z, t)\cdot (\nabla \cdot \mathbf v(\mathbf z, t))
$$

假设 $p(\mathbf z, t) \ne 0$，两边除以 $p(\mathbf z, t)$，得到

$$
\frac {d}{dt}\log p(\mathbf z(t), t) = -(\nabla \cdot \mathbf v (\mathbf z, t))
$$

其中等式右侧速度场的散度 $\nabla\cdot \mathbf v(\mathbf z, t)$ 定义为所有分量的偏导数之和

$$
\nabla \cdot\mathbf v(\mathbf z, t) = \sum_{i} \frac {\partial v_i(\mathbf z, t)}{\partial z_i}
$$

这对应于雅可比矩阵 $J = \frac {\partial \mathbf v(\mathbf z, t))}{\partial \mathbf z}$ 的迹

$$
\mathrm{tr}\left(\frac {\partial \mathbf v(\mathbf z, t)}{\partial \mathbf z}\right) = \mathrm tr(J) = \sum_{ii}J_{ii}=\sum_i \frac {\partial v_i(\mathbf z, t)}{\partial z_i} = \nabla\cdot \mathbf v(\mathbf z, t)
$$

因此最终有

$$
\frac {d}{dt}\log p(\mathbf z, t) = -\mathrm {tr}\left(\frac {\partial \mathbf v(\mathbf z, t)}{\partial \mathbf z}\right)
$$

这一公式将对数概率流场的变化率和速度场的散度联系了起来

得到了对数概率密度随时间的变化率，那么要得到某个时间点 $T$ 处的对数概率密度 $\log p(\mathbf z(T), T)$，我们可以对左右两边进行积分

$$
\int_0^T \frac {d}{dt}\log p(\mathbf z(t), t)dt = -\int_{0}^T \mathrm {tr}\left(\frac {\partial \mathbf v(\mathbf z(t), t)}{\partial \mathbf z(t)}\right)dt
$$

回忆一下微积分基本定理，一个关于 $t$ 的导函数的积分满足

$$
\int_0^T \frac {d}{dt}F(t)dt = F(T) - F(0)
$$

因此，我们得到

$$
\begin{align}
\log p(\mathbf z(T), T) - \log p(\mathbf z(0), 0) &= -\int_{0}^T \mathrm {tr}\left(\frac {\partial \mathbf v(\mathbf z(t), t)}{\partial \mathbf z(t)}\right)dt\\
\log p(\mathbf z(T), T) &= \log p(\mathbf z(0), 0)  - \int_0^T \mathrm {tr} \left(\frac {\partial \mathbf v(\mathbf z(t), t)}{\partial \mathbf z(t)}\right)
\end{align}
$$

因此，我们将目标样本在目标分布下的对数概率密度和初始样本在初始分布下的对数概率密度以及流动路径上速度场 $\mathbf v$ 的散度的积分联系了起来










