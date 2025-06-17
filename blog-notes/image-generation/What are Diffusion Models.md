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

>  既然目标是让 $\pmb \mu_\theta$ 接近 $\pmb {\tilde \mu}_t$，故我们可以将 $\pmb \mu_\theta(\mathbf x_t, t)$ 的形式定义为和 $\pmb {\tilde \mu}_t$ 相同的形式，如上所示
>  在该形式下，训练时的目标实际上就是让 $\epsilon_\theta(\mathbf x_t, t)$ 接近 $\pmb {\tilde \mu}_t$ 中的 $\epsilon_t$，也就是根据输入 $\mathbf x_t$ 预测噪声 $\epsilon_t$
>  (因此可以发现，扩散模型本质是在基于后验结果预测先验噪声)

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

>  $p_\theta(\mathbf x_{t-1}\mid \mathbf x_{t})$ 和 $q(\mathbf x_{t-1}\mid \mathbf x_{t}, \mathbf x_0)$ 的 KL 散度损失项 $L_{t-1}$ 的实际形式如上所示
>  在上面描述的对 $\pmb \mu_\theta(\mathbf x_t, t)$ 的参数化形式下，$L_{t-1}$ 可以重写为以上形式

### Simplification
Empirically, [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) found that training the diffusion model works better with a simplified objective that ignores the weighting term:

$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$

>  DDPM 在经验上发现忽略 $L_{t-1}$ 中的加权项，简化该目标，训练效果会更好
>  所有简化的的 $L_t (t=1, \dots, T)$ 加起来求平均，就得到了简化的最终目标 $L_t^{\text{simple}}$

The final simple objective is:

$$
L_\text{simple} = L_t^\text{simple} + C
$$

where $C$ is a constant not depending on $\theta$.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM-algo.png)

Figure 4. The training and sampling algorithms in DDPM (Image source: [Ho et al. 2020](https://arxiv.org/abs/2006.11239))

>  图中展示了 DDPM 的训练和采样算法
>  训练算法就是不断地对轨迹中的噪声进行预测以及相应的梯度优化
>  采样算法就是从纯噪声中采样一个样本，然后利用训练的模型基于后验样本预测先验噪声，然后基于后验分布采样上一个时间步的样本，如此往复直到 $\mathbf x_0$

### Connection with noise-conditioned score networks (NCSN)
[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) proposed a score-based generative modeling method where samples are produced via [Langevin dynamics](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics) using gradients of the data distribution estimated with score matching. The score of each sample $\mathbf x$ ’s density probability is defined as its gradient $\nabla_{\mathbf x}\log q(\mathbf x)$. A score network $s_\theta: \mathbb R^D \mapsto \mathbb R^D$ is trained to estimate it, $s_\theta \approx \nabla_{\mathbf x}\log q(\mathbf x)$.
>  score-based 生成式方法利用估计的数据分布的得分函数 $s_\theta$ ($s_\theta \approx \nabla_{\mathbf x}\log q(\mathbf x)$)，通过 Langevin dynamics 来采样

To make it scalable with high-dimensional data in the deep learning setting, they proposed to use either _denoising score matching_ ([Vincent, 2011](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)) or _sliced score matching_ (use random projections; [Song et al., 2019](https://arxiv.org/abs/1905.07088)). Denoising score matching adds a pre-specified small noise to the data $q(\tilde {\mathbf x}\mid \mathbf x)$ and estimates $q(\tilde {\mathbf x})$ with score matching.
>  为了让 score-based 方法可以拓展到高维数据，可以采用 denoising score matching 或 sliced score matching 来训练 $s_\theta$
>  其中 denoising score matching 的思路是为数据 $q(\tilde {\mathbf x}\mid \mathbf x)$ 添加小的噪声，然后再用 score matching 估计 $q(\tilde {\mathbf x})$

Recall that Langevin dynamics can sample data points from a probability density distribution using only the score $\nabla_{\mathbf x}\log q(\mathbf x)$ in an iterative process.

However, according to the manifold hypothesis, most of the data is expected to concentrate in a low dimensional manifold, even though the observed data might look only arbitrarily high-dimensional. It brings a negative effect on score estimation since the data points cannot cover the whole space. In regions where data density is low, the score estimation is less reliable. 
>  而根据流形假设，虽然观察到的数据 (数据集里的数据) 可能是任意高维的，但实际的大多数数据会集中在低维流形
>  因为观察到的数据点无法覆盖整个空间，故 score estimation 的准确率会受到影响，对数据密度低的区域的 score estimation 将会不可靠

After adding a small Gaussian noise to make the perturbed data distribution cover the full space $\mathbb R^D$, the training of the score estimator network becomes more stable. [Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) improved it by perturbing the data with the noise of _different levels_ and train a noise-conditioned score network to _jointly_ estimate the scores of all the perturbed data at different noise levels.
>  通过为数据分布添加小的高斯噪声，生成可以覆盖整个 $\mathbb R^D$ 空间的扰动的数据分布，可以让 score estimator 的训练更加稳定
>  Song 用不同尺度的噪声对数据进行扰动，训练了 noise-conditioned score network，该 network 估计了所有级别的扰动数据分布的 score function

The schedule of increasing noise levels resembles the forward diffusion process. If we use the diffusion process annotation, the score approximates $s_\theta(\mathbf x_t, t) \approx \nabla_{\mathbf x_t}\log q(\mathbf x_t)$. 
>  在 Song 的方法中，逐渐增加的噪声扰动级别就类似于一个前向扩散过程
>  如果我们采用扩散过程的符号表示，我们可以说 score-based model 在时间步 $t$ 估计了时间步 $t$ 下，(经过扩散的/经过扰动的) 数据的边际分布的得分函数，即 $s_\theta(\mathbf x_t, t) \approx \nabla_{\mathbf x_t}\log q(\mathbf x_t)$

Given a Gaussian distribution $\mathbf x \sim \mathcal N(\mu, \sigma^2 \mathbf I)$, we can write the derivative of the logarithm of its density function as $\nabla_{\mathbf{x}}\log p (\mathbf{x}) = \nabla_{\mathbf{x}} \Big (-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) = - \frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2} = - \frac{\boldsymbol{\epsilon}}{\sigma}$ where $\epsilon \sim \mathcal N(\mathbf 0, \mathbf I)$.
>  对于一个服从均值为 $\mu$，方差为 $\sigma^2\mathbf I$ 的高斯分布的数据 $\mathbf x$，其得分函数的形式，根据上述推导，可以简单记作 $-\frac {\epsilon}{\sigma}$，其中 $\epsilon \sim \mathcal N(\mathbf 0, \mathbf I)$

[Recall](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice) that $q (\mathbf{x}_t \vert \mathbf{x}_0) \sim \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$ and therefore

$$
\begin{align}
\mathbf{s}_\theta(\mathbf{x}_t, t) 
&\approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)\\
&=\nabla_{\mathbf x_t}\mathbb E_{q(\mathbf x_0)}[\log q(\mathbf x_t, \mathbf x_0)]\\
&=\nabla_{\mathbf x_t}\left(\mathbb E_{q(\mathbf x_0)}[\log q(\mathbf x_t\mid \mathbf x_0) + \log q(\mathbf x_0)]\right)\\
&=\nabla_{\mathbf x_t}\left(\mathbb E_{q(\mathbf x_0)}[\log q(\mathbf x_t\mid \mathbf x_0)] + \mathbb E_{q(\mathbf x_0)}[\log q(\mathbf x_0)]\right)\\
&= \nabla_{\mathbf{x}_t} \mathbb{E}_{q(\mathbf{x}_0)} [\log q(\mathbf{x}_t \vert \mathbf{x}_0)]\\
&= \mathbb{E}_{q(\mathbf{x}_0)} [\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \vert \mathbf{x}_0)]\\
&= \mathbb{E}_{q(\mathbf{x}_0)} \Big[ - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \Big]\\
&= - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
\end{align}
$$

>  在之前的扩散过程上的推导中，我们知道条件于 $\mathbf x_0$ 的 $\mathbf x_t$ 的后验分布 $q(\mathbf x_t\mid \mathbf x_0) \sim \mathcal N(\sqrt {\bar \alpha_t}\mathbf x_0, (1-\bar \alpha_t)\mathbf I)$，故 $q(\mathbf x_t)$ 的得分函数可以按照上述形式推导

>  因此可以看到，$\mathbf s_\theta(\mathbf x_t, t)$ 的目标形式同样是基于 $\mathbf x_t, t$ 预测噪声 $\epsilon_\theta$
>  也可以反过来说，扩散模型实际上也就是在学习数据的得分函数

## Parameterization of $\beta_t$ 
The forward variances are set to be a sequence of linearly increasing constants in [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. They are relatively small compared to the normalized image pixel values between $[−1,1]$. Diffusion models in their experiments showed high-quality samples but still could not achieve competitive model log-likelihood as other generative models.
>  DDPM 将前向方差设置为一系列线性递增的常数，从 $\beta_1 = 10^{-4}$ 到 $\beta_T = 0.02$
>  这些方差相对于规范化后的图片像素值 $[-1, 1]$ 都是较小的
>  DDPM 可以生成高质量的图像样本，但样本的似然值则不比其他生成式模型高

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed several improvement techniques to help diffusion models to obtain lower NLL (Negative Log Likelihood). One of the improvements is to use a cosine-based variance schedule. The choice of the scheduling function can be arbitrary, as long as it provides a near-linear drop in the middle of the training process and subtle changes around $t=0$ and $t=T$.

$$
\beta_t = \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999) \quad\bar{\alpha}_t = \frac{f(t)}{f(0)}\quad\text{where }f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)^2
$$

where the small offset $s$ is to prevent $\beta_t$ from being too small when close to $t=0$.

>  Nichol 为了提高扩散模型的样本似然，提出了一些改进方法
>  其中之一是使用基于余弦的方差调度，形式如上
>  其中调度函数 $f(t)$ 的选择可以任意，只要它可以在训练中途提供近线性的下降，以及在训练首尾 ($t=0, t=T$) 时改变微小

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png)

Figure 5. Comparison of linear and cosine-based scheduling of $\beta_t$ during training. (Image source: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))

## Parameterization of reverse process variance $\Sigma_\theta$
[Ho et al. (2020)](https://arxiv.org/abs/2006.11239) chose to fix $\beta_t$ as constants instead of making them learnable and set $\boldsymbol{\Sigma}_\theta (\mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$ , where $\sigma_t$ is not learned but set to $\beta_t$ or $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$ . Because they found that learning a diagonal variance $\Sigma_\theta$ leads to unstable training and poorer sample quality.
>  DDPM 将 $\beta_t$ 固定为常数，以及将 $\pmb \Sigma_\theta$ 设定为 $\sigma_t^2 \mathbf I$，其中 $\sigma_t$ 直接设定为 $\beta_t$ 或 $\tilde \beta_t$，不对协方差矩阵进行学习 (也就是直接基于先验分布的方差按固定形式计算后验分布的方差)
>  因为 DDPM 发现学习协方差矩阵会导致训练不稳定

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed to learn $\Sigma_\theta$ as an interpolation between $\beta_t$ and $\tilde \beta_t$ by model predicting a mixing vector $\mathbf v$ :

$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)
$$

However, the simple objective $L_{\text{simple}}$ does not depend on $\Sigma_\theta$ . To add the dependency, they constructed a hybrid objective $L_{\text{hybrid}} = L_{\text{simple}} + \lambda L_{\text{VLB}}$ where $\lambda = 0.001$ is small and stop gradient on $\pmb \mu_\theta$ in the $L_{VLB}$ term such that $L_{VLB}$ only guides the learning of $\Sigma_\theta$. 

>  Nichol 提出学习一个混合向量 $\mathbf v$，将 $\Sigma_\theta$ 表示为 $\beta_t, \tilde \beta_t$ 的混合
>  DDPM 中的 simple objective 忽略了 KL 散度中加权项，故和 $\Sigma_\theta$ 无关，Nichol 为了让目标函数和 $\Sigma_\theta$ 相关，构建了一个 hybrid objective
>  其中 $\lambda = 0.001$ 足够小，避免 $L_{\text{VLB}}$ 影响对 $\pmb \mu_\theta$ 的梯度，确保它仅引导对 $\Sigma_\theta$ 的学习

Empirically they observed that $L_{VLB}$ is pretty challenging to optimize likely due to noisy gradients, so they proposed to use a time-averaging smoothed version of $L_{VLB}$ with importance sampling.
>  Nichol 在经验上发现 $L_{\text{VLB}}$ 非常难以优化，故他们采用了重要性采样，构建了 time-averaging 平滑的 $L_{\text{VLB}}$

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/improved-DDPM-nll.png)

Figure 6. Comparison of negative log-likelihood of improved DDPM with other likelihood-based generative models. NLL is reported in the unit of bits/dim. (Image source: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))

# Conditioned Generation
While training generative models on images with conditioning information such as ImageNet dataset, it is common to generate samples conditioned on class labels or a piece of descriptive text.
>  我们继而考虑条件于类别标签以及描述文本片段来生成样本

## Classifier Guided Diffusion
To explicit incorporate class information into the diffusion process, [Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233) trained a classifier $f_\phi(y|\mathbf x_t,t)$ on noisy image $\mathbf x_t$ and use gradients $\nabla_{\mathbf x}\log f_\phi(y\mid \mathbf x_t)$ to guide the diffusion sampling process toward the conditioning information $y$ (e.g. a target class label) by altering the noise prediction.
>  Dhariwal 在加噪的图像 $\mathbf x_t$ 上训练了分类器 $f_\phi(y\mid \mathbf x_t, t)$，并用该分类器相对于 $\mathbf x_t$ 的梯度 $\nabla_{\mathbf x}\log f_\phi(y\mid \mathbf x_t)$ ，变更了噪声预测，来引导扩散采样过程向着条件信息 $y$ 前进

[Recall](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#score) that $\nabla_{\mathbf x_t} \log q(\mathbf x_t) = -\frac {1}{\sqrt {1-\bar \alpha_t}}\epsilon_\theta(\mathbf x_t, t)$ and we can write the score function for the joint distribution $q(\mathbf x_t,y)$ as following,

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y)
&= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y \vert \mathbf{x}_t) \\
&\approx - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t))
\end{aligned}
$$

>  根据上面的推导，数据的真实边际分布的得分函数的形式为 $\nabla_{\mathbf x_t} \log q(\mathbf x_t) = -\frac {1}{\sqrt {1-\bar \alpha_t}}\epsilon_\theta(\mathbf x_t, t)$
>  根据该形式，我们可以将数据和条件信息的联合分布 $q(\mathbf x_t, y)$ 的得分函数如上写出
>  其中第二步使用训练的分类器近似替代了条件信息条件于数据的真实后验分布

Thus, a new classifier-guided predictor $\bar \epsilon_\theta$ would take the form as following,

$$
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t)
$$

>  因此，为了让数据在采样过程中同时向着条件信息的方向移动，我们需要拟合的得分函数应该变更为 $\nabla_{\mathbf x_t}\log q(\mathbf x_t, y)$，进而根据上面的推导，实际的拟合目标就是一个变更后的噪声 $\bar \epsilon_\theta$ (classifier-guided predictor)，形式如上所示

To control the strength of the classifier guidance, we can add a weight $w$ to the delta part,

$$
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t)
$$

The resulting _ablated diffusion model_ (**ADM**) and the one with additional classifier guidance (**ADM-G**) are able to achieve better results than SOTA generative models (e.g. BigGAN).

>  要控制条件引导的程度，可以为额外的梯度添加权重 $w$
>  这样得到的模型可以取得比 SOTA 生成式模型更优的结果

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/conditioned-DDPM.png)

Figure 7. The algorithms use guidance from a classifier to run conditioned generation with DDPM and DDIM. (Image source: [Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233)])

Additionally with some modifications on the U-Net architecture, [Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233) showed performance better than GAN with diffusion models. The architecture modifications include larger model depth/width, more attention heads, multi-resolution attention, BigGAN residual blocks for up/downsampling, residual connection rescale by 1/2 and adaptive group normalization (AdaGN).
>  如果再对 U-Net 架构进行额外的修改，扩散模型的表现可以由于 GAN
>  架构上的修改包括扩大模型深度/宽度、增加 attention heads、使用多分辨率 attention、使用 BigGAN 残差块进行上采样和下采样、将残差连接缩放为原来的 1/2 以及适应性组规范化 (AdaGN)

## Classifier-Free Guidance
Without an independent classifier $f_\phi$, it is still possible to run conditional diffusion steps by incorporating the scores from a conditional and an unconditional diffusion model ([Ho & Salimans, 2021](https://openreview.net/forum?id=qw8AKxfYbI))
>  Ho 提出不必额外训练一个独立的分类器 $f_\phi$ 也可以进行条件化的扩散步骤，其方法是将一个有条件的扩散模型的 score 和无条件的扩散模型的 score 结合

Let unconditional denoising diffusion model $p_\theta(\mathbf x)$ parameterized through a score estimator $\epsilon_\theta(\mathbf x_t, t)$ and the conditional model $p_\theta(\mathbf x \mid y)$ parameterized through $\epsilon_\theta(\mathbf x, t  ,y)$. These two models can be learned via a single neural network. Precisely, a conditional diffusion model $p_\theta(\mathbf x|y)$ is trained on paired data $(\mathbf x,y)$, where the conditioning information $y$ gets discarded periodically at random such that the model knows how to generate images unconditionally as well, i.e. $\boldsymbol{\epsilon}_\theta (\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta (\mathbf{x}_t, t, y=\varnothing)$. 
>  我们记无条件去噪扩散模型 $p_\theta(\mathbf x)$ 由 score estimator $\epsilon_\theta(\mathbf x_t, t)$ 参数化，有条件扩散模型 $p_\theta(\mathbf x\mid y)$ 由 score estimator $\epsilon_\theta(\mathbf x, t y)$ 参数化
>  可以通过一个神经网络同时学习这两个模型，训练方法: 在数据对 $(\mathbf x, y)$ 上训练有条件的扩散模型 $p_\theta(\mathbf x \mid y)$ ，同时在训练过程中，周期性地丢弃随机条件信息 $y$，以确保模型同时学会如何在没有条件的情况下也能生成样本，即学会 $\epsilon_\theta(\mathbf x, t) = \epsilon_\theta(\mathbf x_t, t, y = \varnothing)$

The gradient of an implicit classifier can be represented with conditional and unconditional score estimators. Once plugged into the classifier-guided modified score, the score contains no dependency on a separate classifiers

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t)
&=\nabla_{\mathbf x_t}\log \frac {p(y,\mathbf x_t)}{p(\mathbf x_t)}\\
&=\nabla_{\mathbf x_t}\log \frac {p(\mathbf x_t\mid y)p(y)}{p(\mathbf x_t)}\\
&=\nabla_{\mathbf x_t}\log p(\mathbf x_t\mid y)+ \nabla_{\mathbf x_t}\log p(y)-\nabla_{\mathbf x_t}\log p(\mathbf x_t)\\
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}}\Big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) \\
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y)
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big) \\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$

>  可以基于有条件的和无条件的 score estimators 表示隐式的 classifier (即后验分布 $p(y\mid \mathbf x_t)$) 的得分函数，形式如上
>  由此，就不必训练一个额外的分类器 $f_\phi$ 来近似 $\nabla_{\mathbf x_t}\log p(y\mid \mathbf x_t)$，之后，再将如上计算的 $\nabla_{\mathbf x_t}\log p(y\mid \mathbf x_t)$ 按照之前的流程变更噪声形式即可

Their experiments showed that classifier-free guidance can achieve a good balance between FID (distinguish between synthetic and generated images) and IS (quality and diversity).
>  classifier-free guidance 可以在 FID 和 IS 之间取得较好的平衡

The guided diffusion model, GLIDE ([Nichol, Dhariwal & Ramesh, et al. 2022](https://arxiv.org/abs/2112.10741)), explored both guiding strategies, CLIP guidance and classifier-free guidance, and found that the latter is more preferred. They hypothesized that it is because CLIP guidance exploits the model with adversarial examples towards the CLIP model, rather than optimize the better matched images generation.
>  GLIDE 探索了 CLIP guidance 和 classifier-free guidance，发现后者更优秀

# Speed up Diffusion Models
It is very slow to generate a sample from DDPM by following the Markov chain of the reverse diffusion process, as $T$ can be up to one or a few thousand steps. One data point from [Song et al. (2020)](https://arxiv.org/abs/2010.02502): “For example, it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on an Nvidia 2080 Ti GPU.”
>  DDPM 依照逆扩散过程的 Markov chain 进行采样，因为时间步 $T$ 的数量可能达到数千，故采样的过程可能会非常慢
>  在 Nvidia 2080 Ti 上，从 DDPM 采样 50k 32x32 的图像需要 20h，而从 GAN 则少于 1 分钟

## Fewer Sampling Steps & Distillation
One simple way is to run a strided sampling schedule ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)) by taking the sampling update every $\lceil T/S  \rceil$ steps to reduce the process from $T$ to $S$ steps. The new sampling schedule for generation is $\{\tau_1, \dots, \tau_S\}$ where $\tau_1 < \tau_2 < \dots < \tau_S$ and $S<T$.
>  一个简单的加速采样的方式是进行跨步的采样，即只在逆扩散过程中，每 $\rceil T/ S \lceil$ 进行采样更新，以减少 $T$ 到 $S$ 的采样步数

For another approach, let’s rewrite $q_\sigma(\mathbf x_{t-1}\mid \mathbf x_t, \mathbf x_0)$ to be parameterized by a desired standard deviation $\sigma_t$ according to the [nice property](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice):

$$
\begin{aligned}
\mathbf{x}_{t-1} 
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 +  \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1} & \\
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon} & \\
&= \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t) + \sigma_t\boldsymbol{\epsilon} \\
q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I})
\end{aligned}
$$

where the model $\epsilon_\theta^{(t)}(\cdot)$ predicts the $\epsilon_t$ from $\mathbf x_t$.

>  上述的推导将后验分布 $q(\mathbf x_{t-1}\mid \mathbf x_t, \mathbf x_0)$ 进行了重写
>  其中，$\mathbf x_{t-1}$ 的推导中: 第二步的后两项加起来仍然是方差为 $1-\bar \alpha_{t-1}$ 的高斯噪声 (还利用了 $\epsilon_{t-1}$ 和 $\epsilon_t$ 的关系，这一关系在最初的 reparameterization 中，对 $\epsilon_t$ 的定义下有使用到，那个时候还写作 $\bar \epsilon$)，第三步将扩散模型估计的高斯噪声替换掉了前向扩散的真实噪声
>  根据 $\mathbf x_{t-1}$ 和 $\mathbf x_t$ 的关系的重写形式，可以重写后验分布 $q_\sigma(\mathbf x_{t-1}\mid \mathbf x_t, \mathbf x_0)$ (中的 $\tilde {\pmb \mu}_t$)，和之前的形式的主要差异就是在 $\tilde {\pmb \mu}_t$ 中引入了我们所期望的标准差 $\sigma_t$  (实际上，把 $\sigma_t$ 替换为 $\beta_t$，应该就是和之前一样的式子)

Recall that in $q (\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$, therefore we have:

$$
\tilde{\beta}_t = \sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t
$$

>  将重写的 $q_\sigma(\mathbf x_{t-1}\mid \mathbf x_t, \mathbf x_0)$ 和之前推导的形式 $q (\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$ 比较，可以确定 $\tilde \beta_t$ 和 $\sigma_t$ 的关系如上

Let $\sigma_t^2 = \eta \cdot \tilde{\beta}_t$ such that we can adjust $\eta \in \mathbb R^+$ as a hyperparameter to control the sampling stochasticity. The special case of $\eta = 0$ makes the sampling process _deterministic_. Such a model is named the _denoising diffusion implicit model_ (**DDIM**; [Song et al., 2020](https://arxiv.org/abs/2010.02502)). DDIM has the same marginal noise distribution but deterministically maps noise back to the original data samples.
>  我们令 $\sigma_t^2 = \eta\cdot \tilde \beta_t$，进而通过调节超参数 $\eta$ 来调节方差，进而调节采样随机性
>  (因此这里实际上就是改了一下逆扩散过程在定义上的方差，在之前，我们定义它直接等于前扩散过程的方差，在这里，我们将它定义为前扩散过程的方差乘上一个常数)
>  当 $\eta = 0$，采样就是确定性的，这样的模型称为 DDIM, DDIM 具有相同的边际噪声分布，但确定性地将噪声映射会原始数据样本

During generation, we don’t have to follow the whole chain $t=1,\dots, T$, but rather a subset of steps. Let’s denote $s<t$ as two steps in this accelerated trajectory. The DDIM update step is:

$$
q_{\sigma, s < t}(\mathbf{x}_s \vert \mathbf{x}_t, \mathbf{x}_0)
= \mathcal{N}(\mathbf{x}_s; \sqrt{\bar{\alpha}_s} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_s - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I})
$$

>  DDIM 的更新步骤和之前介绍的逆采样过程没有差别，但是因为我们通过超参数控制了方差，故可以认为采样的得到的结果是数个时间步之前的

While all the models are trained with $T=1000$ diffusion steps in the experiments, they observed that DDIM ($\eta = 0$) can produce the best quality samples when $S$ is small, while DDPM ($\eta = 1$) performs much worse on small $S$. DDPM does perform better when we can afford to run the full reverse Markov diffusion steps ($S=T=1000$). With DDIM, it is possible to train the diffusion model up to any arbitrary number of forward steps but only sample from a subset of steps in the generative process.
>  如果跳跃步数较多，$\eta=0$ 的表现最好 (但很显然样本的 diversity 会随之损坏)，$\eta = 1$ 的表现最差 ($\eta=1$ 就等于在原来的逆过程上跳步，显然效果会差)

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDIM-results.png)

Figure 8. FID scores on CIFAR10 and CelebA datasets by diffusion models of different settings, including DDIM ($\eta=0$) and DDPM ($\hat \sigma$). (Image source: [Song et al., 2020](https://arxiv.org/abs/2010.02502))

Compared to DDPM, DDIM is able to:

1. Generate higher-quality samples using a much fewer number of steps.
2. Have “consistency” property since the generative process is deterministic, meaning that multiple samples conditioned on the same latent variable should have similar high-level features.
3. Because of the consistency, DDIM can do semantically meaningful interpolation in the latent variable.

>  相较于 DDPM, DDIM:
>  - 可以用更少的步骤生成更高质量的样本
>  - 具有 "一致" 的性质，因为生成过程是确定性的，意味着条件于相同隐变量的多个样本应该具有相似的高级特征
>  - 因为该一致性，DDIM 可以在隐变量上进行语义上有意义的插值

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/progressive-distillation.png)
Figure 9. Progressive distillation can reduce the diffusion sampling steps by half in each iteration. (Image source: [Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512))

**Progressive Distillation** ([Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512)) is a method for distilling trained deterministic samplers into new models of halved sampling steps. The student model is initialized from the teacher model and denoises towards a target where one student DDIM step matches 2 teacher steps, instead of using the original sample $\mathbf x_0$ as the denoise target. In every progressive distillation iteration, we can half the sampling steps.
>  Progressive Distillation 是一个将训练好的确定性采样器蒸馏到一半的采样步的方法
>  其中学生模型初始化为教师模型，然后向着一个 student DDIM step 匹配两个 teacher DDIM step 的目标去噪，而不是使用原样本 $\mathbf x_0$ 作为去噪目标
>  每一次 progressive distillation 迭代都可以减半采样步

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/progressive-distillation-algo.png)

Figure 10. Comparison of Algorithm 1 (diffusion model training) and Algorithm 2 (progressive distillation) side-by-side, where the relative changes in progressive distillation are highlighted in green.  
(Image source: [Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512))

**Consistency Models** ([Song et al. 2023](https://arxiv.org/abs/2303.01469)) learns to map any intermediate noisy data points $\mathbf x_t, t > 0$ on the diffusion sampling trajectory back to its origin $\mathbf x_0$ directly. It is named as _consistency_ model because of its _self-consistency_ property as any data points on the same trajectory is mapped to the same origin.
>  Consistency Models 学习将扩散采样轨迹中的任意中间噪声数据点 $\mathbf x_t, t > 0$ 直接映射回其原始数据 $\mathbf x_0$
>  其 consistency 源自于它的 self-consistency 性质，即相同轨迹上的任意数据点都映射回相同的原始数据

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/consistency-models.png)

Figure 10. Consistency models learn to map any data point on the trajectory back to its origin. (Image source: [Song et al., 2023](https://arxiv.org/abs/2303.01469))

Given a trajectory $\{\mathbf x_t\mid t\in [\epsilon, T]\}$ , the _consistency function_ $f$ is defined as $f:(\mathbf x_t, t) \mapsto \mathbf x_\epsilon$ and the equation $f(\mathbf x_t, t) = f(\mathbf x_{t'}, t') = \mathbf x_{\epsilon}$ holds true for all $t, t' \in [\epsilon, T]$ When $t = \epsilon$, $f$ is an identify function. The model can be parameterized as follows, where $c_{skip}(t)$ and $c_{out}(t)$ functions are designed in a way that $c_{skip}(\epsilon)=1,c_{out}(\epsilon)=0$:

$$
f_\theta(\mathbf{x}, t) = c_\text{skip}(t)\mathbf{x} + c_\text{out}(t) F_\theta(\mathbf{x}, t)
$$

>  给定轨迹 $\{\mathbf x_t \mid t\in[\epsilon, T]\}$，一致性函数 $f$ 定义为 $f:(\mathbf x_t, t)\mapsto \mathbf x_\epsilon$
>  一致性函数满足方程 $f(\mathbf x_t, t) = f(\mathbf x_{t'}, t') = \mathbf x_\epsilon$ 对于所有的 $t, t' \in [\epsilon, T]$ 都保持为真，当 $t = \epsilon$，$f$ 就是恒等函数
>  一致性函数的形式可以按照残差和的形式定义，如上所示

It is possible for the consistency model to generate samples in a single step, while still maintaining the flexibility of trading computation for better quality following a multi-step sampling process.
>  consistency 可以单步生成样本，也可以多步生成样本，用更多计算换更好的质量

The paper introduced two ways to train consistency models:
>  训练 consistency model 有两种方式

1. **Consistency Distillation (CD)**: Distill a diffusion model into a consistency model by minimizing the difference between model outputs for pairs generated out of the same trajectory. This enables a much cheaper sampling evaluation. The consistency distillation loss is:

$$
\begin{aligned}
 \mathcal{L}^N_\text{CD} (\theta, \theta^-; \phi) &= \mathbb{E}
 [\lambda (t_n) d (f_\theta (\mathbf{x}_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat{\mathbf{x}}^\phi_{t_n}, t_n)] \\
 \hat{\mathbf{x}}^\phi_{t_n} &= \mathbf{x}_{t_{n+1}} - (t_n - t_{n+1}) \Phi (\mathbf{x}_{t_{n+1}}, t_{n+1}; \phi)
 \end{aligned}
$$

where
- $\Phi(\cdot; \phi)$ is the update function of a one-step [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) solver;
- $n∼\mathcal U[1,N−1]$, has an uniform distribution over $1,\dots,N−1$;
- The network parameters $\theta^-$ is EMA version of  $\theta$ which greatly stabilizes the training (just like in [DQN](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#deep-q-network) or [momentum](https://lilianweng.github.io/posts/2021-05-31-contrastive/#moco--moco-v2) contrastive learning);
- $d(.,.)$ is a positive distance metric function that satisfies $\forall \mathbf x, \mathbf y: d(\mathbf x, \mathbf y)\ge0$ and $d(\mathbf x, \mathbf y) = 0$ if and only if $\mathbf x = \mathbf y$ such as $\ell_2$, $\ell_1$ or [LPIPS](https://arxiv.org/abs/1801.03924) (learned perceptual image patch similarity) distance;
- $\lambda(\cdot)\in \mathbb R^+$ is a positive weighting function and the paper sets $\lambda(t_n) = 1$.

>  Consistency Distillation
>  将一个扩散模型蒸馏为 consistency model，优化目标是 consistency 对相同扩散轨迹中成对的样本的各自的输出之间的距离
>  损失函数的形式如上，其中
>  - $\Phi(\cdot;\phi)$ 为一步 ODE solver 的更新函数
>  - $n\sim \mathcal U[1, N-1]$ 为 $1, N-1$ 之间的均匀分布
>  - 网络参数 $\theta^-$ 是 $\theta$ 的指数移动平均版本，以稳定训练 (类似 DQN 中的目标网络参数)
>  - $d(\cdot, \cdot)$ 是一个正的距离度量函数，满足 $\forall \mathbf x, \mathbf y: d(\mathbf x, \mathbf y)\ge0$ and $d(\mathbf x, \mathbf y) = 0$ if and only if $\mathbf x = \mathbf y$，符合该性质的度量有 $\ell_2, \ell_1$ 等
>  - $\lambda(\cdot )\in \mathbb R^+$ 是一个正的加权函数

2. **Consistency Training (CT)**: The other option is to train a consistency model independently. Note that in CD, a pre-trained score model $s_\phi(\mathbf x, t)$ is used to approximate the ground truth score $\nabla \log p_t(\mathbf x)$ but in CT we need a way to estimate this score function and it turns out an unbiased estimator of $\nabla \log p_t(\mathbf x)$ exists as $-\frac {\mathbf x_t - \mathbf x}{t^2}$. The CT loss is defined as follows:

$$
\mathcal{L}^N_\text{CT} (\theta, \theta^-; \phi) = \mathbb{E}
[\lambda(t_n)d(f_\theta(\mathbf{x} + t_{n+1} \mathbf{z},\;t_{n+1}), f_{\theta^-}(\mathbf{x} + t_n \mathbf{z},\;t_n)]
\text{ where }\mathbf{z} \in \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

>  Consistency Training
>  独立训练一个 Consistency model
>  Consistency Distillation 中使用了预训练的 score model $s_\phi(\mathbf x, t)$ 来近似真实的得分 $\nabla \log p_t(\mathbf x)$，而 Consistency Training 则直接用 $-\frac {\mathbf x_t - \mathbf x}{t^2}$ (unbiased estimator) 来估计 $\nabla\log p_t(\mathbf x)$
>  Consistency Training 的损失定义如上

According to the experiments in the paper, they found,

- Heun ODE solver works better than Euler’s first-order solver, since higher order ODE solvers have smaller estimation errors with the same $N$.
- Among different options of the distance metric function $d(\cdot)$, the LPIPS metric works better than $\ell_1$ and $\ell_2$ distance.
- Smaller $N$ leads to faster convergence but worse samples, whereas larger $N$ leads to slower convergence but better samples upon convergence.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/consistency-models-exp.png)

Figure 12. Comparison of consistency models' performance under different configurations. The best configuration for CD is LPIPS distance metric, Heun ODE solver, and $N=18$. (Image source: [Song et al., 2023](https://arxiv.org/abs/2303.01469))

## Latent Variable Space
_Latent diffusion model_ (**LDM**; [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752)) runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by first trimming off pixel-level redundancy with autoencoder and then manipulating/generating semantic concepts with diffusion process on learned latent.
>  Latent diffusion model 在隐空间进行扩散过程，而不是在像素空间，这使得训练开销更小，推理速度更快
>  LDM 的灵感来源于一张图片的大多数像素都仅和一个感知细节相关，并且在压缩之后，这些语义和概念组合仍然可以保持
>  LDM 通过生成式建模方法大致将感知压缩和语义压缩分离: 首先用自动编码器去除像素级别的冗余信息，然后利用扩散过程在学习到的隐表示上操作或生成语义概念

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/image-distortion-rate.png)

Figure 13. The plot for tradeoff between compression rate and distortion, illustrating two-stage compressions - perceptual and semantic compression. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

The perceptual compression process relies on an autoencoder model. An encoder $\mathcal E$ is used to compress the input image $\mathbf x\in \mathbb R^{H\times W\times 3}$ to a smaller 2D latent vector $\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}$ , where the downsampling rate $f=H/h=W/w=2^m, m \in \mathbb{N}$. Then an decoder $\mathcal D$ reconstructs the images from the latent vector, $\tilde{\mathbf{x}} = \mathcal{D}(\mathbf{z})$. 

>  感知压缩过程通过自动编码器模型实现
>  encoder $\mathcal E$ 将输入图像 $\mathbf x \in \mathbb R^{H\times W \times 3}$ 压缩为一个二维隐向量 $\mathbf z = \mathcal E(\mathbf x)\in \mathbb R^{h\times w\times c}$，该压缩过程的下采样率为 $f = H/h = W/w = 2^m,m\in \mathbb N$
>  decoder $\mathcal D$ 从隐向量重构图像，$\tilde {\mathbf x} = \mathcal D(\mathbf z)$

The paper explored two types of regularization in autoencoder training to avoid arbitrarily high-variance in the latent spaces.

- _KL-reg_: A small KL penalty towards a standard normal distribution over the learned latent, similar to [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/).
- _VQ-reg_: Uses a vector quantization layer within the decoder, like [VQVAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vq-vae-and-vq-vae-2) but the quantization layer is absorbed by the decoder.

>  文中在 autoencoder 训练时，采用了两类规范化方式以避免隐空间的任意高方差
>  - KL-reg: 和标准正态分布之间的 KL 散度惩罚
>  - VQ-reg: 在 decoder 中采用向量量化层

The diffusion and denoising processes happen on the latent vector $\mathbf z$. The denoising model is a time-conditioned U-Net, augmented with the cross-attention mechanism to handle flexible conditioning information for image generation (e.g. class labels, semantic maps, blurred variants of an image). The design is equivalent to fuse representation of different modality into the model with a cross-attention mechanism. Each type of conditioning information is paired with a domain-specific encoder $\tau_\theta$ to project the conditioning input $y$ to an intermediate representation that can be mapped into cross-attention component, $\tau_\theta (y) \in \mathbb{R}^{M \times d_\tau}$:
>  实际的扩散和去噪过程都在隐向量 $\mathbf z$ 上进行
>  去噪模型是一个条件于时间的 U-Net，且采用了 cross-attention 机制来灵活处理条件信息 (例如类标签、语义图、图的模糊变体)
>  该设计等价于用 cross-attention 将不同模态的表征混合到模型中
>  每一类的条件信息对应一个领域特定的 encoder $\tau_\theta$，以将条件输入 $y$ 映射到中间表征: $\tau_\theta(y)\in \mathbb R^{M\times d_r}$

$$
\begin{aligned}
&\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\Big) \cdot \mathbf{V} \\
&\text{where }\mathbf{Q} = \mathbf{W}^{(i)}_Q \cdot \varphi_i(\mathbf{z}_i),\;
\mathbf{K} = \mathbf{W}^{(i)}_K \cdot \tau_\theta(y),\;
\mathbf{V} = \mathbf{W}^{(i)}_V \cdot \tau_\theta(y) \\
&\text{and }
\mathbf{W}^{(i)}_Q \in \mathbb{R}^{d \times d^i_\epsilon},\;
\mathbf{W}^{(i)}_K, \mathbf{W}^{(i)}_V \in \mathbb{R}^{d \times d_\tau},\;
\varphi_i(\mathbf{z}_i) \in \mathbb{R}^{N \times d^i_\epsilon},\;
\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}
\end{aligned}
$$

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/latent-diffusion-arch.png)

Figure 14. The architecture of the latent diffusion model (LDM). (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

# Scale up Generation Resolution and Quality
To generate high-quality images at high resolution, [Ho et al. (2021)](https://arxiv.org/abs/2106.15282) proposed to use a pipeline of multiple diffusion models at increasing resolutions. _Noise conditioning augmentation_ between pipeline models is crucial to the final image quality, which is to apply strong data augmentation to the conditioning input $\mathbf z$ of each super-resolution model $p_\theta(\mathbf x \mid \mathbf z)$. The conditioning noise helps reduce compounding error in the pipeline setup. _U-net_ is a common choice of model architecture in diffusion modeling for high-resolution image generation.
>  为了生成高分辨率下的高质量图像，Ho 提出使用由多个不同分辨率的 diffusion 模型组成的流水线
>  流水线中，不同模型之间的噪声条件增强对于最终的图片质量至关重要
>  Noise conditioning augmentation 为每个超分模型 $p_\theta(\mathbf x\mid \mathbf z)$ 的条件输入 $\mathbf z$ 应用强的数据增强，条件噪声可以帮助减少流水线启动时的计算误差
>  diffusion model 用于高分辨率图片生成时，常用的结构仍然是 U-Net

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/cascaded-diffusion.png)

Figure 15. A cascaded pipeline of multiple diffusion models at increasing resolutions. (Image source: [Ho et al. 2021](https://arxiv.org/abs/2106.15282)])

They found the most effective noise is to apply Gaussian noise at low resolution and Gaussian blur at high resolution. In addition, they also explored two forms of conditioning augmentation that require small modification to the training process. Note that conditioning noise is only applied to training but not at inference.

- Truncated conditioning augmentation stops the diffusion process early at step $t>0$ for low resolution.
- Non-truncated conditioning augmentation runs the full low resolution reverse process until step $0$ but then corrupt it by $\mathbf z_t \sim q(\mathbf x_t\mid \mathbf x_0)$ and then feeds the corrupted $\mathbf z_t$ s into the super-resolution model.

>  Ho 发现，在低分辨率下应用高斯噪声，在高分辨率下应用高斯模糊最为有效
>  此外，Ho 还探索了两种形式的条件增强
>   - Truncated conditioning augmentation: 低分辨率情况下，在较早的时间步直接停止逆扩散过程
>   - Non-truncated conditioning augmentation: 低分辨率下，运行完整的逆扩散过程，但会对 $\mathbf x_t$ 进行扰动: $\mathbf z_t \sim q(\mathbf x_t\mid \mathbf x_0)$，将扰动结果 $\mathbf z_t$ 交给超分模型
>  注意条件噪声仅在训练中应用，推理中不应用

The two-stage diffusion model **unCLIP** ([Ramesh et al. 2022](https://arxiv.org/abs/2204.06125)) heavily utilizes the CLIP text encoder to produce text-guided images at high quality. Given a pretrained CLIP model $\mathbf c$ and paired training data for the diffusion model, $(\mathbf x,y)$, where $\mathbf x$ is an image and $y$ is the corresponding caption, we can compute the CLIP text and image embedding, $\mathbf c^t(y)$ and $\mathbf c^i(x)$, respectively. The unCLIP learns two models in parallel:

- A prior model $P(\mathbf c^i|y)$: outputs CLIP image embedding $\mathbf c^i$ given the text $y$.
- A decoder $P(\mathbf x|\mathbf c^i,[y])$: generates the image $\mathbf x$ given CLIP image embedding $\mathbf c^i$ and optionally the original text $y$.

>  两阶段扩散模型 unCLIP 采用 CLIP text encoder 生成文本引导的图像
>  给定训练好的 CLIP 模型 $\mathbf c$ 和扩散模型成对的训练数据 $(\mathbf x, y)$ ($\mathbf x$ 为图像，$y$ 为对应的描述文字)，图像和文本的嵌入可以分别计算为 $\mathbf c^t(y), \mathbf c^i(x)$
>  unCLIP 并行地学习两个模型:
>  - 先验模型 $P(\mathbf c^i \mid y)$: 给定文本 $y$，输出 CLIP 图像嵌入 $\mathbf c^i$
>  - decoder $P(\mathbf x \mid \mathbf c^i, [y])$: 给定 CLIP 图像嵌入 $\mathbf c^i$ 以及可选的原始文本 $y$，生成图像 $\mathbf x$

These two models enable conditional generation, because

$$
\underbrace{P(\mathbf{x} \vert y) = P(\mathbf{x}, \mathbf{c}^i \vert y)}_{\mathbf{c}^i\text{ is deterministic given }\mathbf{x}} = P(\mathbf{x} \vert \mathbf{c}^i, y)P(\mathbf{c}^i \vert y)
$$


![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/unCLIP.png)

Figure 16. The architecture of unCLIP. (Image source: [Ramesh et al. 2022](https://arxiv.org/abs/2204.06125)])

unCLIP follows a two-stage image generation process:

1. Given a text $y$, a CLIP model is first used to generate a text embedding $\mathbf c^t(y)$. Using CLIP latent space enables zero-shot image manipulation via text.
2. A diffusion or autoregressive prior $P(\mathbf c^i|y)$ processes this CLIP text embedding to construct an image prior and then a diffusion decoder $P(\mathbf x|\mathbf c^i,[y])$ generates an image, conditioned on the prior. This decoder can also generate image variations conditioned on an image input, preserving its style and semantics.

>  unCLIP 采用两阶段的图像生成过程:
>  1. 给定文本 $y$，先用 CLIP 模型生成文本嵌入 $\mathbf c^t(y)$
>  2. 使用扩散模型或自回归先验 $P(\mathbf c^i)\mid y)$，基于 CLIP 文本嵌入构造图像先验，然后用扩散解码器 $P(\mathbf x\mid \mathbf c^i,[y])$ 生成条件于先验的图像

Instead of CLIP model, **Imagen** ([Saharia et al. 2022](https://arxiv.org/abs/2205.11487)) uses a pre-trained large LM (i.e. a frozen T5-XXL text encoder) to encode text for image generation. There is a general trend that larger model size can lead to better image quality and text-image alignment. They found that T5-XXL and CLIP text encoder achieve similar performance on MS-COCO, but human evaluation prefers T5-XXL on DrawBench (a collection of prompts covering 11 categories).

When applying classifier-free guidance, increasing $w$ may lead to better image-text alignment but worse image fidelity. They found that it is due to train-test mismatch, that is to say, because training data $\mathbf x$ stays within the range $[−1,1]$, the test data should be so too. Two thresholding strategies are introduced:

- Static thresholding: clip $\mathbf x$ prediction to $[−1,1]$
- Dynamic thresholding: at each sampling step, compute s as a certain percentile absolute pixel value; if $s>1$, clip the prediction to $[−s,s]$ and divide by $s$.

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

>  U-Net 由一个下采样栈和一个上采样栈组成
>  下采样: 每一步应用两个 3x3 卷积 + ReLU + 2x2 max pooling (strid 2)，每一步下采样后，特征通道数翻倍
>  上采样: 每一步应用上采样 + 2x2 卷积，减半特征通道数
>  shortcut: 为上采样过程提供高分辨率特征

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/U-net.png)

Figure 17. The U-net architecture. Each blue square is a feature map with the number of channels labeled on top and the height x width dimension labeled on the left bottom side. The gray arrows mark the shortcut connections. (Image source: [Ronneberger, 2015](https://arxiv.org/abs/1505.04597))

To enable image generation conditioned on additional images for composition info like Canny edges, Hough lines, user scribbles, human post skeletons, segmentation maps, depths and normals, **ControlNet** ([Zhang et al. 2023](https://arxiv.org/abs/2302.05543) introduces architectural changes via adding a “sandwiched” zero convolution layers of a trainable copy of the original model weights into each encoder layer of the U-Net. Precisely, given a neural network block $\mathcal F_\theta(\cdot)$, ControlNet does the following:

1. First, freeze the original parameters $\theta$ of the original block
2. Clone it to be a copy with trainable parameters $\theta_c$ and an additional conditioning vector $\mathbf c$.
3. Use two zero convolution layers, denoted as $\mathcal{Z}_{\theta_{z1}}(.;.)$ and $\mathcal{Z}_{\theta_{z2}}(.;.)$, which is 1x1 convo layers with both weights and biases initialized to be zeros, to connect these two blocks. Zero convolutions protect this back-bone by eliminating random noise as gradients in the initial training steps.
4. The final output is: $\mathbf{y}_c = \mathcal{F}_\theta(\mathbf{x}) + \mathcal{Z}_{\theta_{z2}}(\mathcal{F}_{\theta_c}(\mathbf{x} + \mathcal{Z}_{\theta_{z1}}(\mathbf{c})))$


![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ControlNet.png)

Figure 18. The ControlNet architecture. (Image source: [Zhang et al. 2023](https://arxiv.org/abs/2302.05543))

**Diffusion Transformer** (**DiT**; [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) for diffusion modeling operates on latent patches, using the same design space of [LDM](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#ldm) (Latent Diffusion Model)]. DiT has the following setup:

1. Take the latent representation of an input $\mathbf z$ as input to DiT.
2. “Patchify” the noise latent of size $I\times I \times C$ into patches of size p and convert it into a sequence of patches of size $(I/p)^2$.
3. Then this sequence of tokens go through Transformer blocks. They are exploring three different designs for how to do generation conditioned on contextual information like timestep $t$ or class label $c$. Among three designs, _adaLN (Adaptive layer norm)-Zero_ works out the best, better than in-context conditioning and cross-attention block. The scale and shift parameters, $\gamma$ and $\beta$, are regressed from the sum of the embedding vectors of $t$ and $c$. The dimension-wise scaling parameters $\alpha$ is also regressed and applied immediately prior to any residual connections within the DiT block.
4. The transformer decoder outputs noise predictions and an output diagonal covariance prediction.

>  DiT 基于 latent patch 进行扩散建模:
>  1. DiT 接收输入 $\mathbf z$ 的 latent 表示
>  2. 将大小为 $I\times I \times C$ 的 noise latent 划分为大小为 $p$ 的 patches，然后将其转化为大小为 $(I/p)^2$ 的 patch 序列
>  3. patch 序列喂给 Transformer 块
>  4. Transformer decoder 输出噪声预测和对角协方差预测

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DiT.png)

Figure 19. The Diffusion Transformer (DiT) architecture.  
(Image source: [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748))

Transformer architecture can be easily scaled up and it is well known for that. This is one of the biggest benefits of DiT as its performance scales up with more compute and larger DiT models are more compute efficient according to the experiments.
>  DiT 最大的优势是方便 scale up

# Quick Summary
- **Pros**: Tractability and flexibility are two conflicting objectives in generative modeling. Tractable models can be analytically evaluated and cheaply fit data (e.g. via a Gaussian or Laplace), but they cannot easily describe the structure in rich datasets. Flexible models can fit arbitrary structures in data, but evaluating, training, or sampling from these models is usually expensive. Diffusion models are both analytically tractable and flexible
- **Cons**: Diffusion models rely on a long Markov chain of diffusion steps to generate samples, so it can be quite expensive in terms of time and compute. New methods have been proposed to make the process much faster, but the sampling is still slower than GAN.

>  - 优势: diffusion 模型即 tractable 又 flexible
>  - 劣势: 依赖于长 Markov chain，采样速度远慢于 GAN

# References
[1] Jascha Sohl-Dickstein et al. [“Deep Unsupervised Learning using Nonequilibrium Thermodynamics.”](https://arxiv.org/abs/1503.03585) ICML 2015.
[2] Max Welling & Yee Whye Teh. [“Bayesian learning via stochastic gradient langevin dynamics.”](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) ICML 2011.
[3] Yang Song & Stefano Ermon. [“Generative modeling by estimating gradients of the data distribution.”](https://arxiv.org/abs/1907.05600) NeurIPS 2019.
[4] Yang Song & Stefano Ermon. [“Improved techniques for training score-based generative models.”](https://arxiv.org/abs/2006.09011) NeuriPS 2020.
[5] Jonathan Ho et al. [“Denoising diffusion probabilistic models.”](https://arxiv.org/abs/2006.11239) arxiv Preprint arxiv: 2006.11239 (2020). [[code](https://github.com/hojonathanho/diffusion)]
[6] Jiaming Song et al. [“Denoising diffusion implicit models.”](https://arxiv.org/abs/2010.02502) arxiv Preprint arxiv: 2010.02502 (2020). [[code](https://github.com/ermongroup/ddim)]
[7] Alex Nichol & Prafulla Dhariwal. [“Improved denoising diffusion probabilistic models”](https://arxiv.org/abs/2102.09672) arxiv Preprint arxiv: 2102.09672 (2021). [[code](https://github.com/openai/improved-diffusion)]
[8] Prafula Dhariwal & Alex Nichol. [“Diffusion Models Beat GANs on Image Synthesis.”](https://arxiv.org/abs/2105.05233) arxiv Preprint arxiv: 2105.05233 (2021). [[code](https://github.com/openai/guided-diffusion)]
[9] Jonathan Ho & Tim Salimans. [“Classifier-Free Diffusion Guidance.”](https://arxiv.org/abs/2207.12598) NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.
[10] Yang Song, et al. [“Score-Based Generative Modeling through Stochastic Differential Equations.”](https://openreview.net/forum?id=PxTIG12RRHS) ICLR 2021.
[11] Alex Nichol, Prafulla Dhariwal & Aditya Ramesh, et al. [“GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.”](https://arxiv.org/abs/2112.10741) ICML 2022.
[12] Jonathan Ho, et al. [“Cascaded diffusion models for high fidelity image generation.”](https://arxiv.org/abs/2106.15282) J. Mach. Learn. Res. 23 (2022): 47-1.
[13] Aditya Ramesh et al. [“Hierarchical Text-Conditional Image Generation with CLIP Latents.”](https://arxiv.org/abs/2204.06125) arxiv Preprint arxiv: 2204.06125 (2022).
[14] Chitwan Saharia & William Chan, et al. [“Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding.”](https://arxiv.org/abs/2205.11487) arxiv Preprint arxiv: 2205.11487 (2022).
[15] Rombach & Blattmann, et al. [“High-Resolution Image Synthesis with Latent Diffusion Models.”](https://arxiv.org/abs/2112.10752) CVPR 2022. [code](https://github.com/CompVis/latent-diffusion)
[16] Song et al. [“Consistency Models”](https://arxiv.org/abs/2303.01469) arxiv Preprint arxiv: 2303.01469 (2023)
[17] Salimans & Ho. [“Progressive Distillation for Fast Sampling of Diffusion Models”](https://arxiv.org/abs/2202.00512) ICLR 2022.
[18] Ronneberger, et al. [“U-Net: Convolutional Networks for Biomedical Image Segmentation”](https://arxiv.org/abs/1505.04597) MICCAI 2015.
[19] Peebles & Xie. [“Scalable diffusion models with transformers.”](https://arxiv.org/abs/2212.09748) ICCV 2023.
[20] Zhang et al. [“Adding Conditional Control to Text-to-Image Diffusion Models.”](https://arxiv.org/abs/2302.05543) arxiv Preprint arxiv: 2302.05543 (2023).
