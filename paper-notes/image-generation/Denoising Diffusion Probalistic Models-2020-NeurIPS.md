# Abstract
我们将展示使用扩散概率模型(diffusion probalistic models)可以生成高质量的合成图像(high quality image synthesis results)，扩散概率模型是一类隐变量模型，灵感来源于非平衡热力学(nonequalibrium thermodynamics)

我们最好的结果通过在一个加权变分界上训练得到(trained on a weight variational bound)，该界限是根据扩散概率模型和与Langevin动力学相匹配的去噪分数(denoising score)之间的新颖的联系(novel connection)设计的

我们的模型自然地支持(admit)一种渐进的有损压缩方案(progressive lossy decompression scheme)，且该方案可以解释为是自回归解码的一种泛化(a generalization of autogressive decoding)

在无条件的(unconditional)CIFAR-10数据集上，我们达到了9.46的Inception分数，以及SOTA的FID分数3.17，在256x256 LSUN上，我们得到了和ProgressiveGAN相同的样本质量(sample quality)
# 1 Introduction
近来，各种类型的深度生成式模型已经能在一系列的数据模态上生成高质量的样本，GANs、自回归模型、flows、VAEs已经可以用于合成非常好的图像和音频样本(image and audio samples)，同时，基于能量的建模(energy-based modeling)和分数匹配(socre matching)方法也取得了很大进展，可以生成与GANs可比的图像

本文将展示扩散模型中的进展，一个扩散概率模型(简称扩散模型)是一个参数化的马尔可夫链(paremeterized Markov chain)，使用变分推理训练，以在有限时间之后可以生成可以匹配数据的样本(produce samples matching the data after finite time)
该马尔可夫链上的转移(transition)是通过逆转一个扩散过程(reverse a diffusion process)学习得到的，在扩散过程中，马尔可夫链在采样的反方向逐渐为数据添加噪声，直到信号被破坏(until signal is destroyed)

当扩散过程是由小量的高斯噪声组成时(consist of small amouts of Gaussian noise)，我们足以将采样链转换(the sampling chain transitions)也设置成条件高斯(conditional Gaussians)，这允许我们使用特别简单的NN对其进行参数化

扩散模型是可以直观定义且高效训练的，但据我们所知，目前并没有工作展示扩散模型可以生成高质量样本(high quality samples)，我们将展示扩散模型是可以生成高质量样本的，有时会比其他类型的生成模型的结果更好，
另外，我们将展示对扩散模型特定的参数化可以使其在训练过程中等价于在多噪声级别上的去噪分数匹配(denoising score matching over multiple noise levels)，以及在采样过程中等价于退火的Langevin动力学

虽然模型的样本质量很高，但扩散模型相较于其他基于似然的模型并没有有竞争力的对数似然(但我们的模型的对数似然要优于基于能量和分数匹配的模型所报告的退火重要性抽样产生的大估计要好 large estimates annealed importance sampling)
我们发现，我们模型的大多数无损编码长度(the majority of our models' lossless codelengths)会被用于描述不易察觉的图像细节(be consumed to describe imperceptible image details)，
我们从有损压缩(lossy compression)的角度对这一现象进行了精细的分析，并展示扩散模型的采样过程是一种渐进式解码(progressive decoding)，这类似于比通常的自回归模型/过程要广泛地多的沿位序的自回归解码(autoregerssive decoding along a bit ordering that vastly generalized what is normally possible with autoregressive models)
# 2 Background
扩散模型是形式为$p_{\theta}(\mathbf x_0) := \int p_{\theta}(\mathbf x_{0:T})d\mathbf x_{1:T}$的隐变量模型，其中$\mathbf x_1, \dots, \mathbf x_T$是和数据$\mathbf x_0 \sim q(\mathbf x_0)$具有相同维度的隐变量，联合分布$p_{\theta}(\mathbf x_{0:T})$被称为逆向过程(reverse process)，它被定义为一个马尔可夫链，由学习到的高斯转换构成(learned Gaussian transitions)，马尔可夫链从$p(\mathbf x_T) = \mathcal N(\mathbf x_T; \mathbf 0, \mathbf I)$开始：
$$p_{\theta}(\mathbf x_{0:T}):=p(\mathbf x_T) \prod_{t=1}^T p_{\theta}(\mathbf x_{t-1}|\mathbf x_t),\quad  p_{\theta}(\mathbf x_{t-1}|\mathbf x_t):=\mathcal N(\mathbf x_{t-1};\boldsymbol \mu_{\theta}(\mathbf x_t,t),\boldsymbol \Sigma_{\theta}(\mathbf x_t,t))\tag{1}$$
将扩散模型和其他类型的隐变量模型相区分开的是它的近似后验(approximate posterior)$q(\mathbf x_{1:T}|\mathbf x_0)$，称其为前向过程(forward process)或扩散过程(diffusion process)，它是一个固定的(fixed)马尔可夫链，根据一个方差调度(variance schedule)$\beta_1,\dots,\beta_T$逐渐为数据添加高斯噪声：
$$q(\mathbf x_{1:T}|\mathbf x_0) :=\prod_{t=1}^T q(\mathbf x_t |\mathbf x_{t-1}),\quad q(\mathbf x_t|\mathbf x_{t-1}) :=\mathcal N(\mathbf x_t;\sqrt {1-\beta_t}\mathbf x_{t-1},\beta_t \mathbf I)\tag{2}$$
训练通过优化常规的负对数似然的变分界限进行：
$$\begin{align}
\mathbb E[-\log p_{\theta}(\mathbf x_0)] &\le \mathbb E_q\left[-\log \frac {p_{\theta}(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\right] \\&= \mathbb E_{q}\left[-\log p(\mathbf x_T)-\sum_{t\ge 1}\log \frac  {p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t-1})}\right]\\
&=:L
\end{align}\tag{3}$$
(
$$\begin{align}
&\mathbb E_{\mathbf x_0\sim q(\mathbf x_0)}[-\log p_{\theta}(\mathbf x_0)]\\
&=-\mathbb E_{\mathbf x_0\sim q(\mathbf x_0)}\left[\log  {p_{\theta}(\mathbf x_{0})}\right]\\
&=-\int d\mathbf x_0 q(\mathbf x_0)\log p_\theta(\mathbf x_0)\\
&=-\int d\mathbf x_0 q(\mathbf x_0)\log \int d\mathbf x_{1:T}{p_{\theta}(\mathbf x_{0:T})} \\
&=-\int d\mathbf x_0 q(\mathbf x_0)\log\int d\mathbf x_{1:T}q(\mathbf x_{1:T}|\mathbf x_0)\frac {p_{\theta}(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_{0})}\\
&\le-\int d\mathbf x_0 q(\mathbf x_0)\int d\mathbf x_{1:T}q(\mathbf x_{1:T}|\mathbf x_0)\log\frac {p_{\theta}(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_{0})}\\
&=-\int d\mathbf x_{0:T}q(\mathbf x_{0:T})\log\frac {p_{\theta}(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_{0})}\\
&=-\mathbb E_q\left[\log \frac  {p_{\theta}(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_{0})}\right]\\
&=-\mathbb E_q\left[\log  {p_{\theta}(\mathbf x_{0:T})}-\log{q(\mathbf x_{1:T}|\mathbf x_{0})}\right]\\
&=-\mathbb E_q\left[\log p(\mathbf x_T) {p_{\theta}(\mathbf x_{0:T-1}|\mathbf x_{T})}-\log{q(\mathbf x_{1:T}|\mathbf x_{0})}\right]\\
&=-\mathbb E_q\left[\log p(\mathbf x_T) +\log{p_{\theta}(\mathbf x_{0:T-1}|\mathbf x_{T})}-\log{q(\mathbf x_{1:T}|\mathbf x_{0})}\right]\\
&=-\mathbb E_q\left[\log p(\mathbf x_T) +\log \frac {p_{\theta}(\mathbf x_{0:T-1}|\mathbf x_{T})}{q(\mathbf x_{1:T}|\mathbf x_{0})}\right]\\
&=-\mathbb E_q\left[\log p(\mathbf x_T) +\sum_{t\ge 1}^T\log \frac {p_{\theta}(\mathbf x_{t-1}|\mathbf x_{t})}{q(\mathbf x_{t}|\mathbf x_{t-1})}\right]\\
&=\mathbb E_q\left[-\log p(\mathbf x_T) -\sum_{t\ge 1}^T\log \frac {p_{\theta}(\mathbf x_{t-1}|\mathbf x_{t})}{q(\mathbf x_{t}|\mathbf x_{t-1})}\right]\\
&=:L
\end{align}$$
)
前向过程的方差$\beta_t$可以通过重参数化来学习，或者作为超参数保持不变(held constant as hyperparameters)，并且逆过程的表达能力的一部分是由$p_\theta(\mathbf x_{t-1}|\mathbf x_t)$中高斯条件(Gaussian conditionals)的选择来保证的，因为当$\beta_t$很小时，两个过程具有相同的函数形式(functional form)

正向过程的一个显著特性(notable property)是它允许在任意的时间步$t$以封闭形式采样$\mathbf x_t$，我们记$\alpha_t :=1-\beta_t$，$\bar {\alpha_t} :=\prod_{s=1}^t \alpha_s$，我们有：
$$
q(\mathbf x_t|\mathbf x_0) = \mathcal N(\mathbf x_t;\sqrt {\bar \alpha_t}\mathbf x_0, (1-\bar \alpha_t)\mathbf I)\tag{4}
$$
(
每一步的前向扩散过程可以写为：
$$
\begin{align}
\mathbf x_t &= \sqrt {1-\beta_t}\mathbf x_{t-1} + \beta_t\epsilon_t\\
&=\alpha_t\mathbf x_{t-1} + (1-\alpha_t)\epsilon_t\\
&=\alpha_t\mathbf x_{t-1} + \epsilon
\end{align}
$$
其中$\epsilon_t \sim \mathcal N(0, \mathbf I), \epsilon \sim \mathcal N(0, (1-\alpha_t)\mathbf I)$
故：
$$
\begin{align}
\mathbf x_t &= \alpha_t \mathbf x_{t-1} + (1-\alpha_t)\epsilon_t\\
&=\alpha_t(\alpha_{t-1}\mathbf x_{t-2}+(1-\alpha_{t-1})\epsilon_{t-1}) + (1-\alpha_t)\epsilon_t\\
&=\alpha_t\alpha_{t-1}\mathbf x_{t-2} + \alpha_t(1-\alpha_{t-1})\epsilon_{t-1} +  (1-\alpha_t)\epsilon_t
\end{align}
$$
其中$\epsilon_{t-1}, \epsilon_t \sim \mathcal N(0, \mathbf I)$
当两个相互独立的正态分布相加时，方差和均值线性叠加，因此：
$$
\begin{align}
\mathbf x_t &= \alpha_t \mathbf x_{t-1} + (1-\alpha_t)\epsilon_t\\
&=\alpha_t(\alpha_{t-1}\mathbf x_{t-2}+(1-\alpha_{t-1})\epsilon_{t-1}) + (1-\alpha_t)\epsilon_t\\
&=\alpha_t\alpha_{t-1}\mathbf x_{t-2} + \alpha_t(1-\alpha_{t-1})\epsilon_{t-1} +  (1-\alpha_t)\epsilon_t\\
&=\alpha_t\alpha_{t-1}\mathbf x_{t-2}+\epsilon
\end{align}
$$
其中$\epsilon \sim \mathcal N(0, (1-\alpha_t\alpha_{t-1})\mathbf I)$
以此类推：
$$
\begin{align}
\mathbf x_t  &= \alpha_t\alpha_{t-1}\cdots\alpha_{1}\mathbf x_0 + \epsilon
\end{align}
$$
其中$\epsilon \sim \mathcal N(0, (1-\alpha_t\alpha_{t-1}\cdots \alpha_1)\mathbf I)$
因此$\mathbf x_t$的条件概率分布是：
$$
q(\mathbf x_t|\mathbf x_0) \sim \mathcal N(\bar {\alpha_t}\mathbf x_0,(1-\bar\alpha_t)\mathbf I)
$$
)

我们可以通过随机梯度下降来优化$L$中的随机项(random terms of $L$)，从而实现高效的训练
为了进一步改进，我们可以通过重写$L$来进一步降低方差，详细见[[#A Extended derivations|附录A]]：
$$
E_q\left[\underbrace {D_{KL}(q(\mathbf x_T|\mathbf x_0)|| p(\mathbf x_T)}_{L_T})+\sum_{t>1} \underbrace {D_{KL}(q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)||p_{\theta}(\mathbf x_{t-1}|\mathbf x_t))}_{L_{t-1}}-\underbrace {\log p_{\theta}(\mathbf x_0|\mathbf x_1)}_{L_0}\right]
$$
Eq.(5)使用KL散度直接比较$p_\theta(\mathbf x_{t-1}|\mathbf x_t)$与前向过程的后验概率(forward process posteriors)，当条件于$\mathbf x_0$时，这些后验概率是可解的(tractable)：
$$
\begin{align}
q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)&=\mathcal N(\mathbf x_{t-1};\tilde {\symbfit \mu}_t(\mathbf x_t, \mathbf x_0), \tilde \beta_t\mathbf I) \\
\text{where}\quad \tilde {\symbfit \mu}_t(\mathbf x_t, \mathbf x_0)&:= \frac {\sqrt {\bar \alpha_{t-1}}\beta_t}{1-\bar \alpha_t}\mathbf x_0 + \frac {\sqrt {\alpha_t}(1-\bar \alpha_{t-1})}{1-\bar \alpha_t}\mathbf x_t\quad\text{and}\quad \tilde \beta_t:= \frac {1-\bar \alpha_{t-1}}{1-\bar \alpha_t}\beta_t
\end{align}
$$
(
容易知道：
$$
\begin{align}
q(\mathbf x_t,\mathbf x_{t-1}| \mathbf x_0) &= q(\mathbf x_{t}|\mathbf x_{t-1},\mathbf x_0) \cdot q(\mathbf x_{t-1}|\mathbf x_0) \\
&= q(\mathbf x_{t}|\mathbf x_{t-1}) \cdot q(\mathbf x_{t-1}|\mathbf x_0) \\
\end{align}
$$
进而：
$$
\begin{align}
q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0) &=\frac {q(\mathbf x_t, \mathbf x_{t-1}|\mathbf x_0)}{q(\mathbf x_t|\mathbf x_0)}\\
&=\frac {q(\mathbf x_t|\mathbf x_{t-1})\cdot q(\mathbf x_{t-1}|\mathbf x_0)}{q(\mathbf x_t|\mathbf x_0)}\\
&=\frac {\mathcal N(\mathbf x_t;\sqrt{1-\beta_t}\mathbf x_{t-1},\beta_t\mathbf I)\cdot\mathcal N(\mathbf x_{t-1};\sqrt {\bar \alpha_{t-1}}\mathbf x_0,(1-\bar \alpha_{t-1})\mathbf I)}{\mathcal N(\mathbf x_t;\sqrt {\bar \alpha_{t}}\mathbf x_0,(1-\bar \alpha_t)\mathbf I)}
\end{align}
$$
两个关于同一个变量的高斯分布(正态分布)相除通常指的是两个概率密度函数(PDF)相除。假设我们有两个高斯分布，它们的数学表达式分别为：

$$f_1(x) = \frac{1}{\sqrt{2\pi\sigma_1^2}} e^{-\frac{(x-\mu_1)^2}{2\sigma_1^2}}$$
$$f_2(x) = \frac{1}{\sqrt{2\pi\sigma_2^2}} e^{-\frac{(x-\mu_2)^2}{2\sigma_2^2}}$$

其中，$\mu_1$和$\mu_2$是两个分布的均值，$\sigma_1$和$\sigma_2$是标准差。

将$f_1(x)$除以$f_2(x)$得到：

$$\frac{f_1(x)}{f_2(x)} = \frac{\frac{1}{\sqrt{2\pi\sigma_1^2}} e^{-\frac{(x-\mu_1)^2}{2\sigma_1^2}}}{\frac{1}{\sqrt{2\pi\sigma_2^2}} e^{-\frac{(x-\mu_2)^2}{2\sigma_2^2}}} = \frac{\sigma_2}{\sigma_1} e^{-\frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}}$$

进一步简化，我们得到：

$$\frac{f_1(x)}{f_2(x)} = \frac{\sigma_2}{\sigma_1} e^{-\frac{(\mu_1-\mu_2)^2}{2} \left(\frac{1}{\sigma_1^2} - \frac{1}{\sigma_2^2}\right)} e^{\frac{(\mu_1-\mu_2)x}{\sigma_1\sigma_2} \left(\frac{1}{\sigma_1^2} - \frac{1}{\sigma_2^2}\right)}$$

这个结果不是一个标准的概率密度函数，因为它不是关于$x$的对称函数，并且没有归一化(即积分不等于1)。但是，它确实给出了两个高斯分布相除的比率，这在统计学中，特别是在贝叶斯统计和信号处理等领域中非常有用。如果$\mu_1$和$\mu_2$相等，那么结果将简化为一个只依赖于$\sigma_1$和$\sigma_2$的比例因子。
)
因此，Eq.(5)中的所有KL散度都是高斯分布之间的比较，所以它们可以以Rao-Blackwell化的方式用封闭形式表达式来计算，而不是使用高方差的蒙特卡洛估计(high variance Monte Carlo estimates)
# A Extended derivations
本节对Eq.(5)进行推导，即扩散模型的减少方差的变分界限，推导源于Sohl-Dickstein el al.\[[[Deep Unsupervised Learning using Nonequilibrium Thermodynamics-2015-ICML|53]]\]：
$$
\begin{align}
L&=\mathbb E_q\left[-\log \frac {p_{\theta}(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\right]\\
&=\mathbb E_q\left[-\log p(\mathbf x_T) - \sum_{t\ge1}\log \frac {p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t-1})}\right]\\
&=\mathbb E_q\left[-\log p(\mathbf x_T)-\sum_{t>1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t-1})}-\log\frac {p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\
&=\mathbb E_q \left[-\log p(\mathbf x_T)- \sum_{t>1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{ q(\mathbf x_{t}|\mathbf x_{t-1},\mathbf x_0)} -\log\frac {p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\
&=\mathbb E_q \left[-\log p(\mathbf x_T)- \sum_{t>1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{\frac {q(\mathbf x_{t},\mathbf x_{t-1},\mathbf x_0)}{q(\mathbf x_{t-1},\mathbf x_0)} } -\log\frac {p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\
&=\mathbb E_q \left[-\log p(\mathbf x_T)- \sum_{t>1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{\frac {q(\mathbf x_{t-1}|\mathbf x_{t},\mathbf x_0)q(\mathbf x_t, \mathbf x_0)}{q(\mathbf x_{t-1},\mathbf x_0)} } -\log\frac {p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\
&=\mathbb E_q \left[-\log p(\mathbf x_T)- \sum_{t>1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)}\cdot \frac {q(\mathbf x_{t-1},\mathbf x_0)}{q(\mathbf x_t,\mathbf x_0)}-\log\frac {p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\
&=\mathbb E_q \left[-\log p(\mathbf x_T)- \sum_{t>1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)}+\sum_{t>1}\log \frac {q(\mathbf x_{t-1}|\mathbf x_0)}{q(\mathbf x_t|\mathbf x_0)}-\log\frac {p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\
&=\mathbb E_q \left[-\log p(\mathbf x_T)- \sum_{t>1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)}+\log \frac {q(\mathbf x_{1}|\mathbf x_0)}{q(\mathbf x_T|\mathbf x_0)}-\log\frac {p_\theta(\mathbf x_0|\mathbf x_1)}{q(\mathbf x_1|\mathbf x_0)}\right]\\
&=\mathbb E_q\left[-\log \frac {p(\mathbf x_T)}{q(\mathbf x_T |\mathbf x_0)}-\sum_{t>1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}-\log p_\theta(\mathbf x_0|\mathbf x_1)\right]\\
&=\mathbb E_q\left[D_{KL}(q(\mathbf x_T|\mathbf x_0)|| p(\mathbf x_T))+\sum_{t>1}D_{KL}(q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)||p_\theta(\mathbf x_{t-1}|\mathbf x_t))-\log p_\theta(\mathbf x_0|\mathbf x_1)\right]
\end{align}
$$

以下是$L$的另一种版本，它不容易估计(not tractable to estimate)，但对于我们在第4.3节的讨论是有用的
$$
\begin{align}
L&=\mathbb E_q\left[-\log p(\mathbf x_T) - \sum_{t\ge 1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t-1})}\right]\tag{23}\\
&=\mathbb E_q\left[-\log p(\mathbf x_T)-\sum_{t\ge 1}\log \frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t)}\cdot \frac {q(\mathbf x_{t-1})}{q(\mathbf x_t)}\right]\tag{24}\\
&=\mathbb E_q\left[-\log \frac {p(\mathbf x_T)}{q(\mathbf x_T)}-\sum_{t\ge 1}\log\frac {p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t)}-\log q(\mathbf x_0)\right]\tag{25}\\
&=D_{KL}(q(\mathbf x_T)||p(\mathbf x_T)) + \mathbb E_q\left[\sum_{t\ge 1}D_{KL}(q(\mathbf x_{t-1}|\mathbf x_t)||p_\theta(\mathbf x_{t-1}|\mathbf x_t))\right]+H(\mathbf x_0)\tag{26}
\end{align}
$$
