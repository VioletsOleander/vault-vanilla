---
completed:
---
Site: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
Date: 11 July 2021

So far, I’ve written about three types of generative models, [GAN](https://lilianweng.github.io/posts/2017-08-20-gan/), [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/), and [Flow-based](https://lilianweng.github.io/posts/2018-10-13-flow-models/) models. They have shown great success in generating high-quality samples, but each has some limitations of its own. GAN models are known for potentially unstable training and less diversity in generation due to their adversarial training nature. VAE relies on a surrogate loss. Flow models have to use specialized architectures to construct reversible transform.
>  各类生成模型的限制:
>  - GAN 训练不稳定，diversity 较低
>  - VAE 依赖替代损失
>  - Flow models 需要指定架构，确保转换可逆

Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).
>  diffusion model 定义 Markov 扩散过程，为数据加噪，模型学习逆扩散过程，从噪声去噪来采样数据
>  和 VAE, flow models 不同，diffusion model 的隐变量也是高维的 (和原数据维度相同)

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png)

Figure 1. Overview of different types of generative models.

# What are Diffusion Models?
Several diffusion-based generative models have been proposed with similar ideas underneath, including _diffusion probabilistic models_ ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)), _noise-conditioned score network_ (**NCSN**; [Yang & Ermon, 2019](https://arxiv.org/abs/1907.05600)), and _denoising diffusion probabilistic models_ (**DDPM**; [Ho et al. 2020](https://arxiv.org/abs/2006.11239)).

## Forward diffusion process
Given a data point sampled from a real data distribution $\mathbf x_0 \sim q(\mathbf x)$, let us define a _forward diffusion process_ in which we add small amount of Gaussian noise to the sample in $T$ steps, producing a sequence of noisy samples $\mathbf x_1, \dots, \mathbf x_T$. The step sizes are controlled by a variance schedule $\{\beta_t\in (0, 1)\}_{t=1}^T$.

$$
q(\mathbf x_t\mid \mathbf x_{t-1}) = \mathcal N(\mathbf x_t\mid \sqrt {1-\beta_t}\mathbf x_{t-1};\beta_t\mathbf I)\quad q(\mathbf x_{1:T}\mid \mathbf x_t) = \prod_{t=1}^T q(\mathbf x_t\mid \mathbf x_{t-1})
$$

>  给定真实数据 $\mathbf x_0\sim q(\mathbf x)$，diffusion 执行 $T$ 步的前向扩散，每步扩散为数据添加小量的高斯噪声
>  diffusion 过程产生了加噪样本序列 $\mathbf x_1, \dots, \mathbf x_T$
>  步长由方差调度参数 $\{\beta_t\in(0,1)\}_{t=1}^T$ 控制

>  推导
>  加噪过程:

$$
\begin{align}
q(\mathbf x_t \mid \mathbf x_{t-1}) &= \mathcal N(\mathbf x_t\mid \sqrt{1-\beta_t}\mathbf x_{t-1}; \beta_t \mathbf I)\\
&=\sqrt {1-\beta_t} \mathbf x_{t-1} + \mathcal N(\mathbf \epsilon \mid \mathbf 0; \beta_t\mathbf I)\\
&=\sqrt {1-\beta_t} \mathbf x_{t-1} + \sqrt \beta_t\cdot\mathcal N(\mathbf \epsilon \mid \mathbf 0; \mathbf I)\\
\end{align}
$$

> 轨迹的条件分布:

$$
\begin{align}
&\prod_{t=1}^T q(\mathbf x_t\mid \mathbf x_{t-1})\\
=& q(\mathbf x_1\mid \mathbf x_0)q(\mathbf x_2\mid \mathbf x_1) \cdots q(\mathbf x_T\mid\mathbf x_{T-1})\\
=& q(\mathbf x_1\mid \mathbf x_0)q(\mathbf x_2\mid \mathbf x_1, \mathbf x_0) \cdots q(\mathbf x_T\mid\mathbf x_{T-1}, \dots, \mathbf x_0)\tag{Markov Property}\\
=& \frac {q(\mathbf x_1, \mathbf x_0)}{q(\mathbf x_0)}\cdot \frac {q(\mathbf x_2, \mathbf x_1, \mathbf x_0)}{q(\mathbf x_1, \mathbf x_0)}\cdots \frac {q(\mathbf x_T,\dots, \mathbf x_0)}{q(\mathbf x_{T-1}, \dots, \mathbf x_0)}\\
=&\frac {q(\mathbf x_T, \dots, \mathbf x_0)}{q(\mathbf x_0)}\\
=& q(\mathbf x_{1:T}\mid \mathbf x_0)
\end{align}
$$

The data sample $\mathbf x_0$ gradually loses its distinguishable features as the step i $t$ becomes larger. Eventually when $T\to \infty$, $\mathbf x_T$ is equivalent to an isotropic Gaussian distribution.
>  样本 $\mathbf x_0$ 会在扩散过程逐渐失去其可辨识特征，若 $T\to \infty$，$\mathbf x_T$ 将服从各向同性的高斯分布

>  推导
>  假设对于所有的 $t \in [1, T]$，$\beta_t$ 都相等

$$
\begin{align}
q(\mathbf x_T\mid \mathbf x_{T-1}) &= \sqrt{1-\beta_t}\mathbf x_{T-1} + \sqrt \beta_t \cdot \mathcal N(\epsilon\mid \mathbf 0;\mathbf I)\\
&=\sqrt {1-\beta_t}(\sqrt{1-\beta_t}\mathbf x_{T-2} + \sqrt\beta_t\cdot \mathcal N(\epsilon \mid \mathbf 0;\mathbf I)) + \sqrt \beta_t\cdot\mathcal N(\epsilon_T\mid \mathbf 0; \mathbf I)\\
&=\cdots\\
&=(\sqrt {1-\beta_t})^{T}\mathbf x_0 + (\sqrt{1-\beta_t})^{T-1}\sqrt\beta_t\cdot \mathcal N(\epsilon\mid \mathbf 0;\mathbf I) +\cdots+\sqrt \beta_t \cdot \mathcal N(\epsilon\mid 0;\mathbf I)\\
&=(\sqrt {1-\beta_t})^{T}\mathbf x_0 + \mathcal N(\epsilon\mid \mathbf 0;(1-\beta_t)^{T-1}\beta_t \mathbf I) +\cdots+\sqrt \beta_t \cdot \mathcal N(\epsilon\mid \mathbf 0;\beta_t\mathbf I)\\
&=(\sqrt {1-\beta_t})^{T}\mathbf x_0 + \mathcal N(\epsilon\mid \mathbf 0;\left[(1-\beta_t)^{T-1}+\cdots+(1-\beta_t)^0\right]\beta_t \mathbf I) \\
\end{align}
$$

>  其中最后两个等式利用了正态分布的性质
>  因为 $\sum_{t=0}^{T-1} (1-\beta_t)^t=\frac {1-(1-\beta_t)^T}{\beta_t}$，则当 $T\to \infty$

$$
\begin{align}
q(\mathbf x_T\mid \mathbf x_{T-1}) &=(\sqrt {1-\beta_t})^{T}\mathbf x_0 + \mathcal N(\epsilon\mid \mathbf 0;(\sum_{t=0}^{T-1}(1-\beta_t)^t)\beta_t \mathbf I) \\
&=0\cdot \mathbf x_0 + \mathcal N(\epsilon\mid\mathbf 0;\mathbf I)\\
&=\mathcal N(\epsilon\mid\mathbf 0;\mathbf I)\\
\end{align}
$$

>  如果各个 $t \in [1, T]$ 的 $\beta_t$ 不相等，情况也是类似的，差异仅在于最后高斯分布的协方差矩阵不一定是 $\mathbf I$，而是 $\mathbf I$ 乘上一个常数 (相关推导在下面)

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)

Figure 2. The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source: [Ho et al. 2020](https://arxiv.org/abs/2006.11239) with a few additional annotations)

A nice property of the above process is that we can sample $\mathbf x_t$ at any arbitrary time step $t$ in a closed form using [reparameterization trick](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick). Let $\alpha_t = 1 - \beta_t$ and $\bar \alpha_t = \prod_{i=1}^t \alpha_i$:

$$
\begin{align}
\mathbf x_t &= \sqrt \alpha_t \mathbf x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1};\quad \text{where}\ \epsilon_{t-1},\epsilon_{t-2},\dots \sim \mathcal N(0, \mathbf I)\\
&=\sqrt \alpha_t(\sqrt \alpha_{t-1}\mathbf x_{t-2} + \sqrt {1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt {1-\alpha_{t}}\epsilon_{t-1}\\
&=\sqrt {\alpha_t\alpha_{t-1}}\mathbf x_{t-2} + \sqrt{\alpha_t - \alpha_{t}\alpha_{t-1}}\epsilon_{t-2}+\sqrt {1-\alpha_t}\epsilon_{t-1}\\
&=\sqrt {\alpha_t\alpha_{t-1}}\mathbf x_{t-2} + \sqrt {1-\alpha_t\alpha_{t-1}} \bar {\epsilon}_{t-2};\quad \text{where}\ \bar \epsilon_{t-2}\ \text{merges two Gaussians(*)}\\
&=\dots\\
&=\sqrt {\bar \alpha_t}\mathbf x_0 + \sqrt {1-\bar \alpha_t}\epsilon\\
q(\mathbf x_t\mid \mathbf x_0) &= \mathcal N(\mathbf x_t;\sqrt {\bar \alpha_t}\mathbf x_0;\sqrt {1-\bar \alpha_t}\mathbf I)
\end{align}
$$

(\*) Recall that when we merge two Gaussians with different variance, $\mathcal N(\mathbf 0, \sigma_1^2 \mathbf I)$ and $\mathcal N(\mathbf 0, \sigma_2^2\mathbf I)$, the new distribution is $\mathcal N(\mathbf 0, (\sigma_1^2 + \sigma_2^2)\mathbf I)$. Here the merged standard deviation is $\sqrt {(1-\alpha_t) + \alpha_t(1-\alpha_{t-1})} = \sqrt {1-\alpha_t\alpha_{t-1}}$.

>  采用 reparameterization trick，扩散过程中的每一个时间步上的样本 $\mathbf x_t$ 都可以闭式地被表示/采样

Usually, we can afford a larger update step when the sample gets noisier, so $\beta_1 < \beta_2 < \cdots < \beta_T$ and therefore $\bar \alpha_1 > \bar \alpha_2 > \cdots > \bar \alpha_T$.

### Connection with stochastic gradient Langevin dynamics
Langevin dynamics is a concept from physics, developed for statistically modeling molecular systems. Combined with stochastic gradient descent, _stochastic gradient Langevin dynamics_ ([Welling & Teh 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)) can produce samples from a probability density $p(\mathbf x)$ using only the gradients $\nabla_{\mathbf x}\log p(\mathbf x)$ in a Markov chain of updates:

$$
\mathbf x_t = \mathbf x_{t-1} +\frac \delta 2 \nabla_{\mathbf x}\log p(\mathbf x_{t-1}) + \sqrt \delta \epsilon_t; \quad \text{where}\ \epsilon_t \sim \mathcal N(\mathbf 0, \mathbf I)
$$

where $\delta$ is the step size. When $T\to \infty, \epsilon \to 0$, $\mathbf x_T$ equals to the true probability density $p(\mathbf x)$.

>  (stochastic gradient) Langevin dynamics 基于分布 $p(\mathbf x)$ 的得分函数 $\nabla_{\mathbf x}\log p(\mathbf x)$，构造 Markov 更新，来实现从分布中采样
>  更新的形式如上所示，其中 $\delta$ 为步长
>  随着时间步趋近于无穷，噪声项趋近于零，$\mathbf x_T$ 趋近于从真实分布 $p(\mathbf x)$ 中采样得到

Compared to standard SGD, stochastic gradient Langevin dynamics injects Gaussian noise into the parameter updates to avoid collapses into local minima.
>  (stochastic gradient) Langevin dynamics 和标准 SGD 的差异仅在于 (stochastic gradient) Langevin dynamics 在更新的时候还加入了随机高斯噪声，以避免陷入局部极小

>  如果不带噪声地看 Langevin dynamics，其本质就是梯度上升，样本前进的方向就是提高 $\log p(\mathbf x)$ 的方向

## Reverse diffusion process
If we can reverse the above process and sample from $q(\mathbf x_{t-1}\mid \mathbf x_t)$, we will be able to recreate the true sample from a Gaussian noise input, $\mathbf x_T\sim \mathcal N(\mathbf 0, \mathbf I)$. Note that if $\beta_t$ is small enough, $q(\mathbf x_{t-1}\mid \mathbf x_t)$ will also be Gaussian. 
>  前向扩散过程 Markov 式加噪，其定义已经在上面阐明
>  我们的目的是逆转前向扩散过程，从噪声 $\mathbf x_T\sim \mathcal N(\mathbf 0, \mathbf I)$ 去噪，恢复样本
>  这里依赖于一个事实: 在 $\beta_t$ 足够小的情况下，后验分布 $q(\mathbf x_{t-1}\mid \mathbf x_t)$ 也是高斯形式 (实际上后验的形式和 $\beta_t$ 的大小没有关联，它就是高斯的，但足够小的 $\beta_t$ 可以让扩散过程更接近连续时间上的 SDE，可以带来更好的平滑性)

Unfortunately, we cannot easily estimate $q(\mathbf x_{t−1}| \mathbf x_t)$ because it needs to use the entire dataset and therefore we need to learn a model $p_\theta$ to approximate these conditional probabilities in order to run the _reverse diffusion process_.

$$
p_\theta(\mathbf x_{0:T})=p(\mathbf x_T)\prod_{t=1}^T p_\theta(\mathbf x_{t-1}\mid \mathbf x_t)\quad p_\theta(\mathbf x_{t-1}\mid \mathbf x_t)=\mathcal N(\mathbf x_{t-1};\pmb \mu_{\theta}(\mathbf x_t, t),\pmb \Sigma_\theta(\mathbf x_t, t))
$$

>  形式确定了，我们的目标就是利用数据，学习/估计它的参数
>  我们用 $\theta$ 参数化各个后验高斯分布的均值和方差，此时各个 (近似的) 后验分布可以记作 $p_\theta(\mathbf x_{t-1}\mid \mathbf x_t) = \mathcal N(\mathbf x_{t-1};\pmb \mu_\theta(\mathbf x_t, t),\pmb \Sigma_\theta(\mathbf x_t, t))$

>  其实整个扩散过程可以看作描述所有时间步上随机变量的联合分布 $q$，该联合分布 $q$ 满足一定的约束: 由扩散过程定义的，两个连续的时间步之间的变量关系
>  我们想要学习这个联合分布，故用 $p_\theta$ 近似 $q$，因此是在参数化整个联合分布，而参数化的形式则是通过参数化一个个高斯型的 Markov 条件分布，整个联合分布由这些条件分布和初始的噪声分布相乘得到，就如上式所表示的

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-example.png)

Figure 3. An example of training a diffusion model for modeling a 2D swiss roll data. (Image source: [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585))

It is noteworthy that the reverse conditional probability is tractable when conditioned on $\mathbf { x } _ { 0 }$

$$
q ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { t - 1 } ; \tilde { \pmb { \mu } } ( \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) , \tilde { \beta } _ { t } \mathbf { I } )
$$

>  当额外添加了条件 $\mathbf x_0$ 后，后验分布是可计算的，其形式如上
>  以下是证明

Using Bayes' rule, we have:

$$
\begin{align}
q(\mathbf{x}_{t-1}\mid\mathbf x_{t}, \mathbf{x}_0) &= \frac {q(\mathbf x_{t},\mathbf x_{t-1}\mid \mathbf x_0)}{q(\mathbf x_{t}\mid \mathbf x_0)}\\
&=q (\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{q (\mathbf{x}_{t-1} | \mathbf{x}_0)}{q (\mathbf{x}_t | \mathbf{x}_0)} \\
&=q (\mathbf{x}_t | \mathbf{x}_{t-1}) \frac{q (\mathbf{x}_{t-1} | \mathbf{x}_0)}{q (\mathbf{x}_t | \mathbf{x}_0)} \\
&=\mathcal N(\mathbf x_t;\sqrt \alpha_t \mathbf x_{t-1},  ({1-\alpha_t}) \mathbf I) \frac{\mathcal N(\mathbf x_{t-1};\sqrt{\bar \alpha_{t-1}}\mathbf x_0,(1-\bar \alpha_{t-1})\mathbf I)}{\mathcal N(\mathbf x_t;\sqrt {\bar \alpha_t}\mathbf x_0,(1-\bar \alpha_t)\mathbf I)} \\
& \propto \exp \Big ( -\frac{1}{2} \Big ( \frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1 - \bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1 - \bar{\alpha}_t} \Big) \Big) \\
& = \exp \Big ( -\frac{1}{2} \Big ( \frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \mathbf{x}_{t-1} + \alpha_t \mathbf{x}_{t-1}^2}{\beta_t} + \frac{\mathbf{x}_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 \mathbf{x}_{t-1} + \bar{\alpha}_{t-1} \mathbf{x}_0^2}{1 - \bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt {\bar \alpha_t}\mathbf{x}_{0})^2}{1 - \bar{\alpha}_{t}} \Big) \Big) \\
& = \exp \Big ( -\frac{1}{2} \Big ( \Big ( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \Big) \mathbf{x}_{t-1}^2 - \Big ( \frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \Big) \mathbf{x}_{t-1} + C (\mathbf{x}_t, \mathbf{x}_0) \Big) \Big)
\end{align}
$$

where ${C } ( { \bf x } _ { t } , { \bf x } _ { 0 } )$ is some function not involving ${ \bf x } _ { t - 1 }$ and details are omitted.

>  首先利用贝叶斯定理，可以将后验分布的指数部分整理如上，其中 $C(\mathbf x_t, \mathbf x_0)$ 是不涉及 $\mathbf x_{t-1}$ 的函数

Following the standard Gaussian density function, the mean and variance can be parameterized as follows (recall that $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } ) } \end{array}$ ：

$$
\begin{align}
\tilde{\beta}_t &= 1 / \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right) = 1 / \left( \frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t (1 - \bar{\alpha}_{t-1})} \right) = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot {\beta}_t \\
\pmb {\tilde{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) &= \left( \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \right) / \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right) \\
\quad &= \left( \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \right) \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot {\beta}_t \\
\quad &= \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0
\end{align}
$$

>  根据上述整理的指数部分，我们可以写出 $\tilde \beta_t$ 的形式如上，以及 $\pmb {\tilde \mu_t}(\mathbf x_t, \mathbf x_0)$ 的形式如上

Thanks to the [nice property](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice), we can represent $\begin{array} { r } { \mathbf { x } _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \big ( \mathbf { x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \pmb { \epsilon } _ { t } \big ) } \end{array}$ and plug it into the above equation and obtain:

$$
\begin{align}
\tilde{\mu}_t &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t 
+ \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}} (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_t) \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\beta_t}{\sqrt \alpha_t(1 - \bar{\alpha}_t)}  \mathbf{x}_t - \frac {\beta_t}{\sqrt \alpha_t\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \\
&= \frac{\alpha_t(1 - \bar{\alpha}_{t-1})+\beta_t}{\sqrt\alpha_t(1 - \bar{\alpha}_t)} \mathbf{x}_t - \frac {\beta_t}{\sqrt \alpha_t\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \\
&= \frac{1}{\sqrt\alpha_t} \mathbf{x}_t - \frac {1-\alpha_t}{\sqrt \alpha_t\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \\
&= \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \right)
\end{align}
$$

>  根据之前的结论，我们将 $\mathbf x_0$ 表示为 $\frac {1}{\sqrt {\bar \alpha_t}}(\mathbf x_t - \sqrt {1-\bar \alpha_t}\epsilon_t)$，然后代入 $\pmb {\tilde \mu_t}$ 的表达式，得到上式

As demonstrated in Fig. 2., such a setup is very similar to [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) and thus we can use the variational lower bound to optimize the negative log-likelihood.
>  回到我们最初的设定，我们要基于数据近似参数 $\theta$，简单的思路就是极大似然法，最小化数据的负对数似然 $-\log p_\theta(\mathbf x_0)$
>  类似 VAE，我们可以通过优化负对数似然的变分上界来间接优化负对数似然，推导如下

>  真实联合分布 $q$ 和近似联合分布 $p_\theta$ 的形式都是由扩散过程定义的，我们在建模 $p_\theta$ 的时候已经确保了它的形式和 $q$ 的形式是一致的，因此剩下的工作就是通过数据拟合参数 $\theta$，让 $p_\theta$ 接近 $q$

$$
\begin{align}
- \log p_\theta(\mathbf{x}_0) &\leq - \log p_\theta(\mathbf{x}_0) + D_{\mathrm{KL}} \big( q(\mathbf{x}_{1:T} | \mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T} | \mathbf{x}_0) \big);\qquad\text{KL is non-negative} \\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \right] \\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \left[ \log \frac{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \right] \\
&= \mathbb{E}_q \left[ \log \frac{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] \\
\text{Let } L_{\mathrm{VLB}} &= \mathbb{E}_{q(\mathbf x_{0:T})} \left[ \log \frac{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0).
\end{align}
$$

>  推导
>  极大似然等价于最大化对数似然，等价于最小化负对数似然

$$
\max_\theta \mathbb E_{\mathbf x_0\sim q(\mathbf x_0)}[\log p_\theta(\mathbf x_0)] = \min_\theta \mathbb E_{\mathbf x_0\sim q(\mathbf x_0)}[-\log p_\theta(\mathbf x_0)]
$$

>  故很容易可以得到: 极大似然等价于最小化 KL 散度

$$
\begin{align}
&\max_\theta \mathbb E_{\mathbf x_0\sim q(\mathbf x_0)}[\log p_\theta(\mathbf x_0)]\\
=&\min_\theta \mathbb E_{\mathbf x_0\sim q(\mathbf x_0)}[\log\frac {1}{p_\theta(\mathbf x_0)}]\\
=&\min_\theta \mathbb E_{\mathbf x_0\sim q(\mathbf x_0)}[\log \frac {1}{p_\theta(\mathbf x_0)} + \log q(\mathbf x_0)]\\
=&\min_\theta \mathbb E_{\mathbf x_0\sim q(\mathbf x_0)}[\log \frac { q(\mathbf x_0)}{p_\theta(\mathbf x_0)} ]\\
=&\min_\theta D_{\mathrm{KL}}(q(\mathbf x_0)\|p_\theta(\mathbf x_0))
\end{align}
$$

>  根据我们参数化 $p_\theta$ 的形式 (我们是参数化了整个联合分布)，给定 $\mathbf x_{0:T} \sim q$，$p_\theta(\mathbf x_0)$ 的计算过程应该是

$$
\begin{align}
p_\theta(\mathbf x_0) &= \frac {p_\theta(\mathbf x_0,\mathbf x_{1:T})}{p_\theta(\mathbf x_{1:T}\mid \mathbf x_0)}\\
&= \frac {p_\theta(\mathbf x_{0:T})}{p_\theta(\mathbf x_{1:T}\mid \mathbf x_0)}
\end{align}
$$

>  可以看到，其中的条件分布 $p(\mathbf x_{1: T}\mid \mathbf x_0)$ 对于我们的参数化形式不太好计算，因为我们是按照后验分布参数化的 (毕竟我们要求解的就是后验分布)
>  因此，我们需要引入变分近似，用 $q(\mathbf x_{1: T}\mid \mathbf x_0)$ 替代 $p_\theta(\mathbf x_{1: T}\mid \mathbf x_0)$

$$
p_\theta(\mathbf x_0)^{\text{approx}} = \frac {p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}\mid \mathbf x_0)}
$$

>  我们可以继而优化 $p_\theta(\mathbf x_0)^{\text{approx}}$ 与 $q(\mathbf x_0)$ 之间的 KL 散度，来近似求解参数 $\theta$
>  接下来，我们探究 $p_\theta(\mathbf x_0)^{\text{approx}}$ 与 $q(\mathbf x_0)$ 之间的 KL 散度和 $p_\theta(\mathbf x_0)$ 与 $q(\mathbf x_0)$ 之间的 KL 散度的关系
>  给定一条轨迹 $\mathbf x_{0: T}$，我们有:

$$
\begin{align}
&\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_0)}\\
=&\log \frac {q(\mathbf x_0)p(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{0:T})}\\
=&\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_{0:T})} + \log p_\theta(\mathbf x_{1:T}\mid \mathbf x_0)\\
=&\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_{0:T})} + \log q(\mathbf x_{1:T}\mid \mathbf x_0)- \log q(\mathbf x_{1:T}\mid \mathbf x_0)+\log p_\theta(\mathbf x_{1:T}\mid \mathbf x_0)\\
=&\log \frac {q(\mathbf x_0)q(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{0:T})} - \log \frac {q(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{1:T}\mid \mathbf x_0)}\\
=&\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_{0})^{\text{approx}}} - \log \frac {q(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{1:T}\mid \mathbf x_0)}\\
\end{align}
$$

>  故

$$
\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_0)} = \log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_{0})^{\text{approx}}} - \log \frac {q(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{1:T}\mid \mathbf x_0)}\\
$$

>  两边都对轨迹 $\mathbf x_{0: T}\sim q$ 求期望，得到

$$
\begin{align}
\mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_0)}] &= \mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_{0})^{\text{approx}}}] - \mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{1:T}\mid \mathbf x_0)}]\\
D_{\mathrm{KL}}(q(\mathbf x_0)\|p_\theta(\mathbf x_0))&=\mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_{0})^{\text{approx}}}] - \mathbb E_{\mathbf x_0\sim q}[D_{\mathrm{KL}}(q(\mathbf x_{1:T}\mid \mathbf x_0)\|p_\theta(\mathbf x_{1:T}\mid \mathbf x_0))]
\end{align}
$$

>  因为 KL 散度的非负性，故 $D_{KL}(q(\mathbf x_0)\|p_\theta(\mathbf x_0))$ 的上界就是

$$
\begin{align}
&\mathbb E_{\mathbf x_{0: T}\sim q}[\log \frac {q(\mathbf x_0)}{p_\theta(\mathbf x_{0})^{\text{approx}}}]\\
=&\mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_{0:T})}{p_\theta(\mathbf x_{0:T})}]\\
=&D_{\mathrm{KL}}(q(\mathbf x_{0:T})\|p_\theta(\mathbf x_{0:T}))
\end{align}
$$

>  也可以把这个上界看作 $q(\mathbf x_0)$ 和 $p_\theta(\mathbf x_0)^{\text{approx}}$ 的 KL 散度
>  该上界和文中推导出的变分上界的关系是

$$
\begin{align}
&\mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_{0:T})}{p_\theta(\mathbf x_{0:T})}]\\
=&\mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_{1:T}\mid \mathbf x_0)q(\mathbf x_0)}{p_\theta(\mathbf x_{0:T})}]\\
=&\mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{0:T})}] + \mathbb E_{\mathbf x_{0:T}\sim q}[\log q(\mathbf x_0)]\\
=&\mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{0:T})}] - \mathbb E_{\mathbf x_{0}\sim q}[\log\frac {1}{ q(\mathbf x_0)}]\\
=&\mathbb E_{\mathbf x_{0:T}\sim q}[\log \frac {q(\mathbf x_{1:T}\mid \mathbf x_0)}{p_\theta(\mathbf x_{0:T})}] - \mathcal H[q(\mathbf x_0)]
\end{align}
$$

>  二者相差一个和优化参数 $\theta$ 无关的常数: 数据分布 $q(\mathbf x_0)$ 的熵
>  因此，这两个变分上界对于优化来说是完全等价的 (显然我推导的要更好理解一点)

It is also straightforward to get the same result using Jensen's inequality. Say we want to minimize the cross entropy as the learning objective,

$$
\begin{aligned}
L_\text{CE}
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big) \\
&\leq - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \\
&= \mathbb{E}_{q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] = L_\text{VLB}
\end{aligned}
$$

>  使用 Jensen's inequality 也可以得到相同的结果，推导如上

To convert each term in the equation to be analytically computable, the objective can be further rewritten to be a combination of several KL-divergence and entropy terms (See the detailed step-by-step process in Appendix B in [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)):
>  该变分上界可以进一步被分解为多个 KL 散度和熵项的结合，以便于实际计算，推导如下

$$
\begin{aligned}
L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
$$

Let’s label each component in the variational lower bound loss separately:

$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$

>  因此，$L_{\mathrm{VLB}}$ 可以写为 $L_T, \dots, L_0$ 的和，如上所示

Every KL term in $L_{\mathrm{VLB}}$ (except for $L_0$) compares two Gaussian distributions and therefore they can be computed in [closed form](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions). $L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $\mathbf x_T$ is a Gaussian noise. [Ho et al. 2020](https://arxiv.org/abs/2006.11239) models $L_0$ using a separate discrete decoder derived from $\mathcal N(\mathbf x_0;\pmb \mu_\theta(\mathbf x_1, 1), \pmb \Sigma_\theta(\mathbf x_1, 1))$.
>  除了 $L_0$ 外，每个 $L_t$ 都是关于两个高斯分布的 KL 散度，而多元高斯分布之间的 KL 散度可以被闭式计算
>  $L_T$ 可以在训练中被忽略，因为 $\mathbf x_T$ 是完全的高斯噪声 (by definition，故 $p_\theta(\mathbf x_T)$ 与 $\theta$ 实际无关)，$q$ 没有可学习参数
>  DDPM 中，$L_0$ 使用一个分别的离散编码器建模 ($\mathcal N(\mathbf x_0;\pmb \mu_\theta(\mathbf x_1, 1), \pmb \Sigma_\theta(\mathbf x_1, 1))$ 就是 $p_\theta(\mathbf x_0\mid \mathbf x_1)$ 在定义上的形式)

## Parameterization of $L_t$ for Training Loss
Recall that we need to learn a neural network to approximate the conditioned probability distributions in the reverse diffusion process, $p_\theta(\mathbf x_{t-1}\mid \mathbf x_t) = \mathcal N(\mathbf x_{t-1};\pmb \mu_\theta(\mathbf x_t, t), \pmb \Sigma_\theta(\mathbf x_t, t))$. 
>  根据定义，我们知道所有的后验分布 $p_\theta(\mathbf x_{t-1}\mid \mathbf x_t)$ 的形式都是 $\mathcal N(\mathbf x_{t-1};\pmb \mu_\theta(\mathbf x_t, t), \pmb \Sigma_\theta(\mathbf x_t, t))$

We would like to train $\pmb \mu_\theta$ to predict $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \Big ( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)$. 
>  根据之前的推导，我们知道，当 $\mathbf x_0$ 在条件中，所有的真实后验的形式都满足
>  $q ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { t - 1 } ; \tilde { \pmb { \mu } } ( \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) , \tilde { \beta } _ { t } \mathbf { I } )$，其中 $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \Big ( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)$
>  要最小化 $L_{t-1}$，就是要最小化 $q(\mathbf x_{t-1}\mid \mathbf x_t, \mathbf x_0)$ 和 $p_\theta(\mathbf x_{t-1}\mid \mathbf x_t)$ 之间的 KL 散度，因此，我们的训练目标 (之一) 就是让 $\pmb \mu_\theta$ 接近 $\pmb {\tilde \mu_t}$

Because $\mathbf x_t$ is available as input at training time, we can reparametrize the Gaussian noise term instead to make it predict $\epsilon_t$ from the input $\mathbf x_t$ at time step $t$:

$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= {\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$

The loss term $L_t$ is parameterized to minimize the difference from $\pmb {\tilde \mu}$ :

$$
\begin{aligned}
L_t 
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| \color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - \color{green}{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] 
\end{aligned}
$$

### Simplification
Empirically, [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) found that training the diffusion model works better with a simplified objective that ignores the weighting term:

The final simple objective is:

where C is a constant not depending on θ.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM-algo.png)

The training and sampling algorithms in DDPM (Image source: [Ho et al. 2020](https://arxiv.org/abs/2006.11239))
### Connection with noise-conditioned score networks (NCSN)
[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) proposed a score-based generative modeling method where samples are produced via [Langevin dynamics](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics) using gradients of the data distribution estimated with score matching. The score of each sample x’s density probability is defined as its gradient ∇xlog⁡q(x). A score network sθ:RD→RD is trained to estimate it, sθ(x)≈∇xlog⁡q(x).

To make it scalable with high-dimensional data in the deep learning setting, they proposed to use either _denoising score matching_ ([Vincent, 2011](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)) or _sliced score matching_ (use random projections; [Song et al., 2019](https://arxiv.org/abs/1905.07088)). Denosing score matching adds a pre-specified small noise to the data q(x~|x) and estimates q(x~) with score matching.

Recall that Langevin dynamics can sample data points from a probability density distribution using only the score ∇xlog⁡q(x) in an iterative process.

However, according to the manifold hypothesis, most of the data is expected to concentrate in a low dimensional manifold, even though the observed data might look only arbitrarily high-dimensional. It brings a negative effect on score estimation since the data points cannot cover the whole space. In regions where data density is low, the score estimation is less reliable. After adding a small Gaussian noise to make the perturbed data distribution cover the full space RD, the training of the score estimator network becomes more stable. [Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) improved it by perturbing the data with the noise of _different levels_ and train a noise-conditioned score network to _jointly_ estimate the scores of all the perturbed data at different noise levels.

The schedule of increasing noise levels resembles the forward diffusion process. If we use the diffusion process annotation, the score approximates sθ(xt,t)≈∇xtlog⁡q(xt). Given a Gaussian distribution x∼N(μ,σ2I), we can write the derivative of the logarithm of its density function as ∇xlog⁡p(x)=∇x(−12σ2(x−μ)2)=−x−μσ2=−ϵσ where ϵ∼N(0,I). [Recall](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice) that q(xt|x0)∼N(α¯tx0,(1−α¯t)I) and therefore,

sθ(xt,t)≈∇xtlog⁡q(xt)=Eq(x0)[∇xtlog⁡q(xt|x0)]=Eq(x0)[−ϵθ(xt,t)1−α¯t]=−ϵθ(xt,t)1−α¯t

## Parameterization of βt
The forward variances are set to be a sequence of linearly increasing constants in [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), from β1=10−4 to βT=0.02. They are relatively small compared to the normalized image pixel values between [−1,1]. Diffusion models in their experiments showed high-quality samples but still could not achieve competitive model log-likelihood as other generative models.

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed several improvement techniques to help diffusion models to obtain lower NLL. One of the improvements is to use a cosine-based variance schedule. The choice of the scheduling function can be arbitrary, as long as it provides a near-linear drop in the middle of the training process and subtle changes around t=0 and t=T.

βt=clip(1−α¯tα¯t−1,0.999)α¯t=f(t)f(0)where f(t)=cos⁡(t/T+s1+s⋅π2)2

where the small offset s is to prevent βt from being too small when close to t=0.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png)

Comparison of linear and cosine-based scheduling of β_t during training. (Image source: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))

## Parameterization of reverse process variance Σθ

[Ho et al. (2020)](https://arxiv.org/abs/2006.11239) chose to fix βt as constants instead of making them learnable and set Σθ(xt,t)=σt2I , where σt is not learned but set to βt or β~t=1−α¯t−11−α¯t⋅βt. Because they found that learning a diagonal variance Σθ leads to unstable training and poorer sample quality.

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed to learn Σθ(xt,t) as an interpolation between βt and β~t by model predicting a mixing vector v :

Σθ(xt,t)=exp⁡(vlog⁡βt+(1−v)log⁡β~t)

However, the simple objective Lsimple does not depend on Σθ . To add the dependency, they constructed a hybrid objective Lhybrid=Lsimple+λLVLB where λ=0.001 is small and stop gradient on μθ in the LVLB term such that LVLB only guides the learning of Σθ. Empirically they observed that LVLB is pretty challenging to optimize likely due to noisy gradients, so they proposed to use a time-averaging smoothed version of LVLB with importance sampling.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/improved-DDPM-nll.png)

Comparison of negative log-likelihood of improved DDPM with other likelihood-based generative models. NLL is reported in the unit of bits/dim. (Image source: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))

# Conditioned Generation

While training generative models on images with conditioning information such as ImageNet dataset, it is common to generate samples conditioned on class labels or a piece of descriptive text.

## Classifier Guided Diffusion

To explicit incorporate class information into the diffusion process, [Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233) trained a classifier fϕ(y|xt,t) on noisy image xt and use gradients ∇xlog⁡fϕ(y|xt) to guide the diffusion sampling process toward the conditioning information y (e.g. a target class label) by altering the noise prediction. [Recall](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#score) that ∇xtlog⁡q(xt)=−11−α¯tϵθ(xt,t) and we can write the score function for the joint distribution q(xt,y) as following,

∇xtlog⁡q(xt,y)=∇xtlog⁡q(xt)+∇xtlog⁡q(y|xt)≈−11−α¯tϵθ(xt,t)+∇xtlog⁡fϕ(y|xt)=−11−α¯t(ϵθ(xt,t)−1−α¯t∇xtlog⁡fϕ(y|xt))

Thus, a new classifier-guided predictor ϵ¯θ would take the form as following,

ϵ¯θ(xt,t)=ϵθ(xt,t)−1−α¯t∇xtlog⁡fϕ(y|xt)

To control the strength of the classifier guidance, we can add a weight w to the delta part,

ϵ¯θ(xt,t)=ϵθ(xt,t)−1−α¯tw∇xtlog⁡fϕ(y|xt)

The resulting _ablated diffusion model_ (**ADM**) and the one with additional classifier guidance (**ADM-G**) are able to achieve better results than SOTA generative models (e.g. BigGAN).

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/conditioned-DDPM.png)

The algorithms use guidance from a classifier to run conditioned generation with DDPM and DDIM. (Image source: [Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233)])

Additionally with some modifications on the U-Net architecture, [Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233) showed performance better than GAN with diffusion models. The architecture modifications include larger model depth/width, more attention heads, multi-resolution attention, BigGAN residual blocks for up/downsampling, residual connection rescale by 1/2 and adaptive group normalization (AdaGN).

## Classifier-Free Guidance

Without an independent classifier fϕ, it is still possible to run conditional diffusion steps by incorporating the scores from a conditional and an unconditional diffusion model ([Ho & Salimans, 2021](https://openreview.net/forum?id=qw8AKxfYbI)). Let unconditional denoising diffusion model pθ(x) parameterized through a score estimator ϵθ(xt,t) and the conditional model pθ(x|y) parameterized through ϵθ(xt,t,y). These two models can be learned via a single neural network. Precisely, a conditional diffusion model pθ(x|y) is trained on paired data (x,y), where the conditioning information y gets discarded periodically at random such that the model knows how to generate images unconditionally as well, i.e. ϵθ(xt,t)=ϵθ(xt,t,y=∅).

The gradient of an implicit classifier can be represented with conditional and unconditional score estimators. Once plugged into the classifier-guided modified score, the score contains no dependency on a separate classifier.

∇xtlog⁡p(y|xt)=∇xtlog⁡p(xt|y)−∇xtlog⁡p(xt)=−11−α¯t(ϵθ(xt,t,y)−ϵθ(xt,t))ϵ¯θ(xt,t,y)=ϵθ(xt,t,y)−1−α¯tw∇xtlog⁡p(y|xt)=ϵθ(xt,t,y)+w(ϵθ(xt,t,y)−ϵθ(xt,t))=(w+1)ϵθ(xt,t,y)−wϵθ(xt,t)

Their experiments showed that classifier-free guidance can achieve a good balance between FID (distinguish between synthetic and generated images) and IS (quality and diversity).

The guided diffusion model, GLIDE ([Nichol, Dhariwal & Ramesh, et al. 2022](https://arxiv.org/abs/2112.10741)), explored both guiding strategies, CLIP guidance and classifier-free guidance, and found that the latter is more preferred. They hypothesized that it is because CLIP guidance exploits the model with adversarial examples towards the CLIP model, rather than optimize the better matched images generation.

# Speed up Diffusion Models

It is very slow to generate a sample from DDPM by following the Markov chain of the reverse diffusion process, as T can be up to one or a few thousand steps. One data point from [Song et al. (2020)](https://arxiv.org/abs/2010.02502): “For example, it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on an Nvidia 2080 Ti GPU.”

## Fewer Sampling Steps & Distillation

One simple way is to run a strided sampling schedule ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)) by taking the sampling update every ⌈T/S⌉ steps to reduce the process from T to S steps. The new sampling schedule for generation is {τ1,…,τS} where τ1<τ2<⋯<τS∈[1,T] and S<T.

For another approach, let’s rewrite qσ(xt−1|xt,x0) to be parameterized by a desired standard deviation σt according to the [nice property](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice):

xt−1=α¯t−1x0+1−α¯t−1ϵt−1=α¯t−1x0+1−α¯t−1−σt2ϵt+σtϵ=α¯t−1(xt−1−α¯tϵθ(t)(xt)α¯t)+1−α¯t−1−σt2ϵθ(t)(xt)+σtϵqσ(xt−1|xt,x0)=N(xt−1;α¯t−1(xt−1−α¯tϵθ(t)(xt)α¯t)+1−α¯t−1−σt2ϵθ(t)(xt),σt2I)

where the model ϵθ(t)(.) predicts the ϵt from xt.

Recall that in q(xt−1|xt,x0)=N(xt−1;μ~(xt,x0),β~tI), therefore we have:

β~t=σt2=1−α¯t−11−α¯t⋅βt

Let σt2=η⋅β~t such that we can adjust η∈R+ as a hyperparameter to control the sampling stochasticity. The special case of η=0 makes the sampling process _deterministic_. Such a model is named the _denoising diffusion implicit model_ (**DDIM**; [Song et al., 2020](https://arxiv.org/abs/2010.02502)). DDIM has the same marginal noise distribution but deterministically maps noise back to the original data samples.

During generation, we don’t have to follow the whole chain t=1,…,T, but rather a subset of steps. Let’s denote s<t as two steps in this accelerated trajectory. The DDIM update step is:

qσ,s<t(xs|xt,x0)=N(xs;α¯s(xt−1−α¯tϵθ(t)(xt)α¯t)+1−α¯s−σt2ϵθ(t)(xt),σt2I)

While all the models are trained with T=1000 diffusion steps in the experiments, they observed that DDIM (η=0) can produce the best quality samples when S is small, while DDPM (η=1) performs much worse on small S. DDPM does perform better when we can afford to run the full reverse Markov diffusion steps (S=T=1000). With DDIM, it is possible to train the diffusion model up to any arbitrary number of forward steps but only sample from a subset of steps in the generative process.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDIM-results.png)

FID scores on CIFAR10 and CelebA datasets by diffusion models of different settings, including DDIM (η=0) and DDPM (σ^). (Image source: [Song et al., 2020](https://arxiv.org/abs/2010.02502))

Compared to DDPM, DDIM is able to:

1. Generate higher-quality samples using a much fewer number of steps.
2. Have “consistency” property since the generative process is deterministic, meaning that multiple samples conditioned on the same latent variable should have similar high-level features.
3. Because of the consistency, DDIM can do semantically meaningful interpolation in the latent variable.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/progressive-distillation.png)

Progressive distillation can reduce the diffusion sampling steps by half in each iteration. (Image source: [Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512))

**Progressive Distillation** ([Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512)) is a method for distilling trained deterministic samplers into new models of halved sampling steps. The student model is initialized from the teacher model and denoises towards a target where one student DDIM step matches 2 teacher steps, instead of using the original sample x0 as the denoise target. In every progressive distillation iteration, we can half the sampling steps.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/progressive-distillation-algo.png)

Comparison of Algorithm 1 (diffusion model training) and Algorithm 2 (progressive distillation) side-by-side, where the relative changes in progressive distillation are highlighted in green.  
(Image source: [Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512))

**Consistency Models** ([Song et al. 2023](https://arxiv.org/abs/2303.01469)) learns to map any intermediate noisy data points xt,t>0 on the diffusion sampling trajectory back to its origin x0 directly. It is named as _consistency_ model because of its _self-consistency_ property as any data points on the same trajectory is mapped to the same origin.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/consistency-models.png)

Consistency models learn to map any data point on the trajectory back to its origin. (Image source: [Song et al., 2023](https://arxiv.org/abs/2303.01469))

Given a trajectory {xt|t∈[ϵ,T]} , the _consistency function_ f is defined as f:(xt,t)↦xϵ and the equation f(xt,t)=f(xt′,t′)=xϵ holds true for all t,t′∈[ϵ,T]. When t=ϵ, f is an identify function. The model can be parameterized as follows, where cskip(t) and cout(t) functions are designed in a way that cskip(ϵ)=1,cout(ϵ)=0:

fθ(x,t)=cskip(t)x+cout(t)Fθ(x,t)

It is possible for the consistency model to generate samples in a single step, while still maintaining the flexibility of trading computation for better quality following a multi-step sampling process.

The paper introduced two ways to train consistency models:

1. **Consistency Distillation (CD)**: Distill a diffusion model into a consistency model by minimizing the difference between model outputs for pairs generated out of the same trajectory. This enables a much cheaper sampling evaluation. The consistency distillation loss is:
    
    LCDN(θ,θ−;ϕ)=E[λ(tn)d(fθ(xtn+1,tn+1),fθ−(x^tnϕ,tn)]x^tnϕ=xtn+1−(tn−tn+1)Φ(xtn+1,tn+1;ϕ)
    
    where
    
    - Φ(.;ϕ) is the update function of a one-step [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) solver;
    - n∼U[1,N−1], has an uniform distribution over 1,…,N−1;
    - The network parameters θ− is EMA version of θ which greatly stabilizes the training (just like in [DQN](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#deep-q-network) or [momentum](https://lilianweng.github.io/posts/2021-05-31-contrastive/#moco--moco-v2) contrastive learning);
    - d(.,.) is a positive distance metric function that satisfies ∀x,y:d(x,y)≥0 and d(x,y)=0 if and only if x=y such as ℓ2, ℓ1 or [LPIPS](https://arxiv.org/abs/1801.03924) (learned perceptual image patch similarity) distance;
    - λ(.)∈R+ is a positive weighting function and the paper sets λ(tn)=1.
2. **Consistency Training (CT)**: The other option is to train a consistency model independently. Note that in CD, a pre-trained score model sϕ(x,t) is used to approximate the ground truth score ∇log⁡pt(x) but in CT we need a way to estimate this score function and it turns out an unbiased estimator of ∇log⁡pt(x) exists as −xt−xt2. The CT loss is defined as follows:
    

LCTN(θ,θ−;ϕ)=E[λ(tn)d(fθ(x+tn+1z,tn+1),fθ−(x+tnz,tn)] where z∈N(0,I)

According to the experiments in the paper, they found,

- Heun ODE solver works better than Euler’s first-order solver, since higher order ODE solvers have smaller estimation errors with the same N.
- Among different options of the distance metric function d(.), the LPIPS metric works better than ℓ1 and ℓ2 distance.
- Smaller N leads to faster convergence but worse samples, whereas larger N leads to slower convergence but better samples upon convergence.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/consistency-models-exp.png)

Comparison of consistency models' performance under different configurations. The best configuration for CD is LPIPS distance metric, Heun ODE solver, and N=18. (Image source: [Song et al., 2023](https://arxiv.org/abs/2303.01469))

## Latent Variable Space

_Latent diffusion model_ (**LDM**; [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752)) runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by first trimming off pixel-level redundancy with autoencoder and then manipulating / generating semantic concepts with diffusion process on learned latent.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/image-distortion-rate.png)

The plot for tradeoff between compression rate and distortion, illustrating two-stage compressions - perceptual and semantic compression. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

The perceptual compression process relies on an autoencoder model. An encoder E is used to compress the input image x∈RH×W×3 to a smaller 2D latent vector z=E(x)∈Rh×w×c , where the downsampling rate f=H/h=W/w=2m,m∈N. Then an decoder D reconstructs the images from the latent vector, x~=D(z). The paper explored two types of regularization in autoencoder training to avoid arbitrarily high-variance in the latent spaces.

- _KL-reg_: A small KL penalty towards a standard normal distribution over the learned latent, similar to [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/).
- _VQ-reg_: Uses a vector quantization layer within the decoder, like [VQVAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vq-vae-and-vq-vae-2) but the quantization layer is absorbed by the decoder.

The diffusion and denoising processes happen on the latent vector z. The denoising model is a time-conditioned U-Net, augmented with the cross-attention mechanism to handle flexible conditioning information for image generation (e.g. class labels, semantic maps, blurred variants of an image). The design is equivalent to fuse representation of different modality into the model with a cross-attention mechanism. Each type of conditioning information is paired with a domain-specific encoder τθ to project the conditioning input y to an intermediate representation that can be mapped into cross-attention component, τθ(y)∈RM×dτ:

Attention(Q,K,V)=softmax(QK⊤d)⋅Vwhere Q=WQ(i)⋅φi(zi),K=WK(i)⋅τθ(y),V=WV(i)⋅τθ(y)and WQ(i)∈Rd×dϵi,WK(i),WV(i)∈Rd×dτ,φi(zi)∈RN×dϵi,τθ(y)∈RM×dτ

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/latent-diffusion-arch.png)

The architecture of the latent diffusion model (LDM). (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

# Scale up Generation Resolution and Quality

To generate high-quality images at high resolution, [Ho et al. (2021)](https://arxiv.org/abs/2106.15282) proposed to use a pipeline of multiple diffusion models at increasing resolutions. _Noise conditioning augmentation_ between pipeline models is crucial to the final image quality, which is to apply strong data augmentation to the conditioning input z of each super-resolution model pθ(x|z). The conditioning noise helps reduce compounding error in the pipeline setup. _U-net_ is a common choice of model architecture in diffusion modeling for high-resolution image generation.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/cascaded-diffusion.png)

A cascaded pipeline of multiple diffusion models at increasing resolutions. (Image source: [Ho et al. 2021](https://arxiv.org/abs/2106.15282)])

They found the most effective noise is to apply Gaussian noise at low resolution and Gaussian blur at high resolution. In addition, they also explored two forms of conditioning augmentation that require small modification to the training process. Note that conditioning noise is only applied to training but not at inference.

- Truncated conditioning augmentation stops the diffusion process early at step t>0 for low resolution.
- Non-truncated conditioning augmentation runs the full low resolution reverse process until step 0 but then corrupt it by zt∼q(xt|x0) and then feeds the corrupted zt s into the super-resolution model.

The two-stage diffusion model **unCLIP** ([Ramesh et al. 2022](https://arxiv.org/abs/2204.06125)) heavily utilizes the CLIP text encoder to produce text-guided images at high quality. Given a pretrained CLIP model c and paired training data for the diffusion model, (x,y), where x is an image and y is the corresponding caption, we can compute the CLIP text and image embedding, ct(y) and ci(x), respectively. The unCLIP learns two models in parallel:

- A prior model P(ci|y): outputs CLIP image embedding ci given the text y.
- A decoder P(x|ci,[y]): generates the image x given CLIP image embedding ci and optionally the original text y.

These two models enable conditional generation, because

P(x|y)=P(x,ci|y)⏟ci is deterministic given x=P(x|ci,y)P(ci|y)

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/unCLIP.png)

The architecture of unCLIP. (Image source: [Ramesh et al. 2022](https://arxiv.org/abs/2204.06125)])

unCLIP follows a two-stage image generation process:

1. Given a text y, a CLIP model is first used to generate a text embedding ct(y). Using CLIP latent space enables zero-shot image manipulation via text.
2. A diffusion or autoregressive prior P(ci|y) processes this CLIP text embedding to construct an image prior and then a diffusion decoder P(x|ci,[y]) generates an image, conditioned on the prior. This decoder can also generate image variations conditioned on an image input, preserving its style and semantics.

Instead of CLIP model, **Imagen** ([Saharia et al. 2022](https://arxiv.org/abs/2205.11487)) uses a pre-trained large LM (i.e. a frozen T5-XXL text encoder) to encode text for image generation. There is a general trend that larger model size can lead to better image quality and text-image alignment. They found that T5-XXL and CLIP text encoder achieve similar performance on MS-COCO, but human evaluation prefers T5-XXL on DrawBench (a collection of prompts covering 11 categories).

When applying classifier-free guidance, increasing w may lead to better image-text alignment but worse image fidelity. They found that it is due to train-test mismatch, that is to say, because training data x stays within the range [−1,1], the test data should be so too. Two thresholding strategies are introduced:

- Static thresholding: clip x prediction to [−1,1]
- Dynamic thresholding: at each sampling step, compute s as a certain percentile absolute pixel value; if s>1, clip the prediction to [−s,s] and divide by s.

Imagen modifies several designs in U-net to make it _efficient U-Net_.

- Shift model parameters from high resolution blocks to low resolution by adding more residual locks for the lower resolutions;
- Scale the skip connections by 1/2
- Reverse the order of downsampling (move it before convolutions) and upsampling operations (move it after convolution) in order to improve the speed of forward pass.

They found that noise conditioning augmentation, dynamic thresholding and efficient U-Net are critical for image quality, but scaling text encoder size is more important than U-Net size.

# Model Architecture

There are two common backbone architecture choices for diffusion models: U-Net and Transformer.

**U-Net** ([Ronneberger, et al. 2015](https://arxiv.org/abs/1505.04597)) consists of a downsampling stack and an upsampling stack.

- _Downsampling_: Each step consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a ReLU and a 2x2 max pooling with stride 2. At each downsampling step, the number of feature channels is doubled.
- _Upsampling_: Each step consists of an upsampling of the feature map followed by a 2x2 convolution and each halves the number of feature channels.
- _Shortcuts_: Shortcut connections result in a concatenation with the corresponding layers of the downsampling stack and provide the essential high-resolution features to the upsampling process.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/U-net.png)

The U-net architecture. Each blue square is a feature map with the number of channels labeled on top and the height x width dimension labeled on the left bottom side. The gray arrows mark the shortcut connections. (Image source: [Ronneberger, 2015](https://arxiv.org/abs/1505.04597))

To enable image generation conditioned on additional images for composition info like Canny edges, Hough lines, user scribbles, human post skeletons, segmentation maps, depths and normals, **ControlNet** ([Zhang et al. 2023](https://arxiv.org/abs/2302.05543) introduces architectural changes via adding a “sandwiched” zero convolution layers of a trainable copy of the original model weights into each encoder layer of the U-Net. Precisely, given a neural network block Fθ(.), ControlNet does the following:

1. First, freeze the original parameters θ of the original block
2. Clone it to be a copy with trainable parameters θc and an additional conditioning vector c.
3. Use two zero convolution layers, denoted as Zθz1(.;.) and Zθz2(.;.), which is 1x1 convo layers with both weights and biases initialized to be zeros, to connect these two blocks. Zero convolutions protect this back-bone by eliminating random noise as gradients in the initial training steps.
4. The final output is: yc=Fθ(x)+Zθz2(Fθc(x+Zθz1(c)))

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ControlNet.png)

The ControlNet architecture. (Image source: [Zhang et al. 2023](https://arxiv.org/abs/2302.05543))

**Diffusion Transformer** (**DiT**; [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) for diffusion modeling operates on latent patches, using the same design space of [LDM](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#ldm) (Latent Diffusion Model)]. DiT has the following setup:

1. Take the latent representation of an input z as input to DiT.
2. “Patchify” the noise latent of size I×I×C into patches of size p and convert it into a sequence of patches of size (I/p)2.
3. Then this sequence of tokens go through Transformer blocks. They are exploring three different designs for how to do generation conditioned on contextual information like timestep t or class label c. Among three designs, _adaLN (Adaptive layer norm)-Zero_ works out the best, better than in-context conditioning and cross-attention block. The scale and shift parameters, γ and β, are regressed from the sum of the embedding vectors of t and c. The dimension-wise scaling parameters α is also regressed and applied immediately prior to any residual connections within the DiT block.
4. The transformer decoder outputs noise predictions and an output diagonal covariance prediction.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DiT.png)

The Diffusion Transformer (DiT) architecture.  
(Image source: [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748))

Transformer architecture can be easily scaled up and it is well known for that. This is one of the biggest benefits of DiT as its performance scales up with more compute and larger DiT models are more compute efficient according to the experiments.

# Quick Summary

- **Pros**: Tractability and flexibility are two conflicting objectives in generative modeling. Tractable models can be analytically evaluated and cheaply fit data (e.g. via a Gaussian or Laplace), but they cannot easily describe the structure in rich datasets. Flexible models can fit arbitrary structures in data, but evaluating, training, or sampling from these models is usually expensive. Diffusion models are both analytically tractable and flexible
    
- **Cons**: Diffusion models rely on a long Markov chain of diffusion steps to generate samples, so it can be quite expensive in terms of time and compute. New methods have been proposed to make the process much faster, but the sampling is still slower than GAN.
    

# Citation

Cited as:

> Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

Or

```
@article{weng2021diffusion,
  title   = "What are diffusion models?",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io",
  year    = "2021",
  month   = "Jul",
  url     = "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/"
}
```

# References

[1] Jascha Sohl-Dickstein et al. [“Deep Unsupervised Learning using Nonequilibrium Thermodynamics.”](https://arxiv.org/abs/1503.03585) ICML 2015.

[2] Max Welling & Yee Whye Teh. [“Bayesian learning via stochastic gradient langevin dynamics.”](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) ICML 2011.

[3] Yang Song & Stefano Ermon. [“Generative modeling by estimating gradients of the data distribution.”](https://arxiv.org/abs/1907.05600) NeurIPS 2019.

[4] Yang Song & Stefano Ermon. [“Improved techniques for training score-based generative models.”](https://arxiv.org/abs/2006.09011) NeuriPS 2020.

[5] Jonathan Ho et al. [“Denoising diffusion probabilistic models.”](https://arxiv.org/abs/2006.11239) arxiv Preprint arxiv:2006.11239 (2020). [[code](https://github.com/hojonathanho/diffusion)]

[6] Jiaming Song et al. [“Denoising diffusion implicit models.”](https://arxiv.org/abs/2010.02502) arxiv Preprint arxiv:2010.02502 (2020). [[code](https://github.com/ermongroup/ddim)]

[7] Alex Nichol & Prafulla Dhariwal. [“Improved denoising diffusion probabilistic models”](https://arxiv.org/abs/2102.09672) arxiv Preprint arxiv:2102.09672 (2021). [[code](https://github.com/openai/improved-diffusion)]

[8] Prafula Dhariwal & Alex Nichol. [“Diffusion Models Beat GANs on Image Synthesis.”](https://arxiv.org/abs/2105.05233) arxiv Preprint arxiv:2105.05233 (2021). [[code](https://github.com/openai/guided-diffusion)]

[9] Jonathan Ho & Tim Salimans. [“Classifier-Free Diffusion Guidance.”](https://arxiv.org/abs/2207.12598) NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

[10] Yang Song, et al. [“Score-Based Generative Modeling through Stochastic Differential Equations.”](https://openreview.net/forum?id=PxTIG12RRHS) ICLR 2021.

[11] Alex Nichol, Prafulla Dhariwal & Aditya Ramesh, et al. [“GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.”](https://arxiv.org/abs/2112.10741) ICML 2022.

[12] Jonathan Ho, et al. [“Cascaded diffusion models for high fidelity image generation.”](https://arxiv.org/abs/2106.15282) J. Mach. Learn. Res. 23 (2022): 47-1.

[13] Aditya Ramesh et al. [“Hierarchical Text-Conditional Image Generation with CLIP Latents.”](https://arxiv.org/abs/2204.06125) arxiv Preprint arxiv:2204.06125 (2022).

[14] Chitwan Saharia & William Chan, et al. [“Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding.”](https://arxiv.org/abs/2205.11487) arxiv Preprint arxiv:2205.11487 (2022).

[15] Rombach & Blattmann, et al. [“High-Resolution Image Synthesis with Latent Diffusion Models.”](https://arxiv.org/abs/2112.10752) CVPR 2022.[code](https://github.com/CompVis/latent-diffusion)

[16] Song et al. [“Consistency Models”](https://arxiv.org/abs/2303.01469) arxiv Preprint arxiv:2303.01469 (2023)

[17] Salimans & Ho. [“Progressive Distillation for Fast Sampling of Diffusion Models”](https://arxiv.org/abs/2202.00512) ICLR 2022.

[18] Ronneberger, et al. [“U-Net: Convolutional Networks for Biomedical Image Segmentation”](https://arxiv.org/abs/1505.04597) MICCAI 2015.

[19] Peebles & Xie. [“Scalable diffusion models with transformers.”](https://arxiv.org/abs/2212.09748) ICCV 2023.

[20] Zhang et al. [“Adding Conditional Control to Text-to-Image Diffusion Models.”](https://arxiv.org/abs/2302.05543) arxiv Preprint arxiv:2302.05543 (2023).

- [Generative-Model](https://lilianweng.github.io/tags/generative-model/)