[VAE](<file:///D:\Learning\paper\2013-VAE-Auto-Encoding Variational Bayes.pdf>)
# Abstract
在连续的隐变量下(隐变量的后验分布不可解 intractable)，并且在大数据集上，我们应该如何在有向的概率模型(directed probabilistic models)上进行高效的推理和学习？
我们介绍一个可以拓展到大数据集的随机变分推理和学习算法，在一些可微条件下(under some mild differentiability conditions)，该算法在隐变量后验分布不可解时也是可行的

我们的贡献有两点：首先，我们说明对变分下界的重参数化可以产生一个可以直接用标准随机梯度方法优化的下界估计(lower bound estimator)；其次，我们说明对于一个独立同分布的数据集(其中每个数据点都和一个连续的隐变量有关)，可以通过拟合一个近似推理模型(approximate inference model)(也称为识别模型 recognition model)，使用之前所提到的下界估计(lower bound estimator)对不可解的后验分布进行高效的后验推理(posterior inference)

实验结果证明了我们的理论的优势
# 1 Introduction
我们该如何对具有后验分布不可解的连续隐变量和/或参数(continous latent variable and/or parameters)的有向概率模型进行高效的近似推理？变分贝叶斯方法包含了对不可解的后验分布的近似的优化过程

但不幸的时，常用的平均场(mean-field)方法需要期望相对于近似后验分布的解析解(analytical solubions of expectations w.r.t the approximate posterior)，而这在通常情况下是不可解的

我们说明对变分下界的重参数化可以导出一个对变分下界的简单的可微且无偏置的估计(differentiable unbiased estimator)，
这个SGVB(随机梯度变分贝叶斯 Stochastic Gradient Variational Bayes)估计可以用于任何带有连续隐变量和/或参数高效的近似后验分布推理(approximate posterior inference)，并且可以直接用标准的随机梯度上升方法优化

对于一个独立同分布的数据集且每个数据点都关联一个连续隐变量的情况(an i.i.d. dataset and continuous latent variables)，我们提出自动编码变分贝叶斯算法(Auto-Encoding VB AEVB)，AEVB算法中，我们可以高效地推理和学习，因为我们可以通过使用SGVB估计优化一个识别模型(recognition model)，这允许我们使用简单的祖先采样(ancestral sampling)进行非常高效的近似后验推理，进而允许我们高效地学习模型参数，而不需要每个数据点上都使用昂贵的迭代推理方法(iterative inference schemes)(例如MCMC)

最后所学习到的后验推理模型可以用于一系列任务，例如识别、去噪、表征和视觉任务，当我们使用神经网络表示识别模型时，我们得到的就是变分自动编码机(variational auto-encoder)
# 2 Method
本节的策略可以用于为一系列带有连续隐变量的有向图模型(directed graphical models with continuous latent variables)推导一个下界估计(lower bound estimator)(一个随机目标函数 a stochastic objective function)
我们将场景约束在每个数据点都带有隐变量的独立同分布数据集上，我们希望对(全局 global)参数进行最大似然(maximum likelihood ML)和最大后验(maximum a posteriori MAP)推理，并且为隐变量进行变分(variational)推理

该场景可以直观地延伸到我们也对全局参数也进行变分推理的场景，该条件下的算法可见附录，实验则留待后日工作

注意我们的方法可以在在线、非静止的设定下(online, non-stationary settings)应用，即流数据，为了简单，本文假设固定的数据集
## 2.1 Problem scenario
考虑包含了$N$个独立同分布的连续或离散变量$\mathbf x$的数据集$\mathbf X = \{\mathbf x^{(i)}\}_{i=1}^N$，
我们假设数据是由某个随机过程生成的，且和某个未观察到的连续随机变量$\mathbf z$有关，该过程由两步构成：
1. 从一个先验分布$p_{\theta^*}(\mathbf z)$中生成一个值$\mathbf z^{(i)}$
2. 从一个条件分布$p_{\theta^*}(\mathbf x | \mathbf z)$中生成一个值$\mathbf x^{(i)}$

我们假设先验分布$p_{\theta^*}(\mathbf z)$和条件分布$p_{\theta^*}(\mathbf x | \mathbf z)$来自于参数化的一族分布(parametric families of distributions)$p_{\theta}(\mathbf z)$和$p_{\theta}(\mathbf x  | \mathbf z)$，并且它们的概率密度函数相对于$\theta$和$\mathbf z$是几乎处处可微的

此时，该数据生成过程的许多细节是隐藏的，即真实的参数$\theta^*$以及隐变量$\mathbf z^{(i)}$的值对于我们是未知的

重要的是，我们不会进行对边际或后验概率分布进行常用的简化推断(common simplifying assumptions)，我们感兴趣于一个通用的算法，甚至可以在以下情况下可用：
1. Intractability 不可解性
    即边际似然的积分$p_{\theta}(\mathbf x) = \int p_{\theta}(\mathbf z)p_{\theta}(\mathbf x | \mathbf z)d\mathbf z$是不可解的
    (因此我们无法对$\mathbf x$的边际似然进行估值或微分)
    此时，真实的后验概率密度函数$p_{\theta}(\mathbf z | \mathbf x) = p_{\theta}(\mathbf x | \mathbf z)p_{\theta}(\mathbf z)/p_{\theta}(\mathbf x)$是不可解的
    (因此EM算法无法使用)
    同时，任意可行的平均场VB算法所需要的积分都是不可解的
    这些不可解性是十分常见的，在中等复杂的似然函数$p_{\theta}(\mathbf x | \mathbf z)$中就会出现(例如带有一个隐藏层的NN)
2. A large dataset 大数据集
    即数据集过大导致批量(batch)优化过于昂贵，因此只能使用小批量(minibatch)甚至单个数据点进行参数更新
    基于采样的方法，例如蒙特卡洛EM，此时就会太过缓慢，因为这些方法包括了需要遍历每个数据点的昂贵的采样循环(每个数据点都要采样)

我们感兴趣于在上述场景下的三个相关问题：
1. 对参数$\theta$的高效的近似ML或MAP估计
    参数$\theta$本身可能就是我们所感兴趣的，例如我们要分析某些自然过程
    参数$\theta$还允许我们模仿隐藏的随机过程(hidden random process)并生成类似于真实数据的人工数据
2. 在给定参数$\theta$的情况下，对于观察到的值$\mathbf x$，对隐藏变量$\mathbf z$进行高效的近似后验推理
    这对于编码(coding)和数据表示(data representation)任务很有用
3. 对变量$\mathbf x$的高效近似边际推理
    这允许我们执行任何需要$\mathbf x$的先验知识/分布的推理任务(inference task)，在CV中常见的应用是图像去噪(image denoising)、修复(inpainting)和超分辨率(super-resolution)

为了解决上述相关问题，我们引入一个识别模型$q_{\phi}(\mathbf z | \mathbf x)$，识别模型作为对不可解的真实后验分布$p_{\theta}(\mathbf z | \mathbf x)$的一个近似，
注意这和平均场变分推理中近似后验分布不同的地方在于$q_{\phi}(\mathbf z | \mathbf x)$并不一定要是可分解的(factorial)，且它的参数$\phi$并不是通过一些闭式期望(closed-form expectation)计算得到，在我们的方法中，我们共同地学习识别模型(recognition model)的参数$\phi$和生成模型(generative model)的参数$\theta$

从编码理论(coding theory)的角度来看，未观测到的变量$\mathbf z$可以解释为一个隐表示(latent representation)或编码(code)，
因此，在本文中，我们也将识别模型$q_{\phi}(\mathbf z | \mathbf x)$称为一个概率编码器(probabilistic encoder)，因为它在给定一个数据点$\mathbf x$的情况下产生一个分布(例如Gaussian)，这个分布是关于数据点$\mathbf x$可能生成于(been generated from)的编码$\mathbf z$的所有可能值的(over the possible values of the code $\mathbf z$)，
类似地，我们称$p_{\theta}(\mathbf x | \mathbf z)$为概率解码器(probabilistic decoder)，因为它在给定一个编码$\mathbf z$的情况下产生一个分布，这个分布是关于$\mathbf x$所有可能对应的值的(over the possible corresponding values of $\mathbf x$)
## 2.2 The variational bound
数据集的边际似然即所有独立(individual)数据点的边际似然的和，
即$\log p_{\theta}(\mathbf x^{(1)},\cdots,\mathbf x^{(N)}) = \sum_{i=1}^N \log p_{\theta}(\mathbf x^{(i)})$，和式中的每一项可以重写为：
$$\log p_{\theta}(\mathbf x^{(i)}) = D_{KL}(q_{\phi}(\mathbf z | \mathbf x^{(i)})||p_{\theta}(\mathbf z|\mathbf x^{(i)})) + \mathcal L(\theta,\phi;\mathbf x^{(i)})\tag{1}$$
RHS中的第一项是近似后验分布和真实后验分布之间的KL散度，第二项称为数据点$i$的边际似然的(变分)下界(lower bound)，它可以写为：
$$\log p_{\theta}(\mathbf x^{(i)}) \ge \mathcal L(\theta,\phi;\mathbf x^{(i)})=\mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}[-\log q_{\phi}(\mathbf z|\mathbf x^{(i)}) + \log p_{\theta}(\mathbf x^{(i)}, \mathbf z)]\tag{2}$$
也可以写为：
$$\mathcal L(\theta,\phi;\mathbf x^{(i)}) = -D_{KL}(q_{\phi}(\mathbf z | \mathbf x^{(i)})||p_{\theta}(\mathbf z)) + \mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}[\log p_{\theta}(\mathbf x^{(i)}|\mathbf z)]\tag{3}$$

(
逆推(2)到(1)：
$$\begin{align}
&D_{KL}(q_{\phi}(\mathbf z | \mathbf x^{(i)})||p_{\theta}(\mathbf z|\mathbf x^{(i)})) + \mathcal L(\theta,\phi;\mathbf x^{(i)})\\
=&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[\log \frac {q_{\phi}(\mathbf z | \mathbf x^{(i)})}{p_{\theta}(\mathbf z | \mathbf x^{(i)})}\right] + \mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[-\log q_{\phi}(\mathbf z | \mathbf x^{(i)}) + \log p_{\theta}(\mathbf x^{(i)}, \mathbf z)\right]\\
=&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[\log \frac {q_{\phi}(\mathbf z | \mathbf x^{(i)})}{p_{\theta}(\mathbf z | \mathbf x^{(i)})}-\log q_{\phi}(\mathbf z | \mathbf x^{(i)}) + \log p_{\theta}(\mathbf x^{(i)}, \mathbf z)\right]\\
=&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[\log  {q_{\phi}(\mathbf z | \mathbf x^{(i)})}-\log{p_{\theta}(\mathbf z | \mathbf x^{(i)})}-\log q_{\phi}(\mathbf z | \mathbf x^{(i)}) + \log p_{\theta}(\mathbf x^{(i)}, \mathbf z)\right]\\
=&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[-\log{p_{\theta}(\mathbf z | \mathbf x^{(i)})}+ \log p_{\theta}(\mathbf x^{(i)}, \mathbf z)\right]\\
=&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[\log \frac {p_{\theta}(\mathbf x^{(i)},\mathbf z)}{p_\theta{(\mathbf z | \mathbf x^{(i)})}}\right]\\
=&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[\log \frac {p_{\theta}(\mathbf x^{(i)},\mathbf z)}{\frac {p_\theta{(\mathbf x^{(i)},\mathbf z })}{p_{\theta}(\mathbf x^{(i)})} }\right]\\
=&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[\log \frac {p_{\theta}(\mathbf x^{(i)},\mathbf z)p_{\theta}(\mathbf x^{(i)})}{p_\theta{(\mathbf x^{(i)},\mathbf z }) }\right]\\
=&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}\left[\log  {p_{\theta}(\mathbf x^{(i)})}\right]\\
=&\log {p_{\theta}(\mathbf x^{(i)})}
\end{align}
$$
正推(2)到(3)：
$$
\begin{align}
\mathcal L(\theta,\phi;\mathbf x^{(i)})
&=
\mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}\left[-\log q_{\phi}(\mathbf z|\mathbf x^{(i)}) + \log p_{\theta}(\mathbf x^{(i)}, \mathbf z)\right]\\
&=
\mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}\left[-\log q_{\phi}(\mathbf z|\mathbf x^{(i)}) + \log p_{\theta}(\mathbf x^{(i)}|\mathbf z)  p_{\theta}(\mathbf z)\right]\\
&=
\mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}\left[-\log q_{\phi}(\mathbf z|\mathbf x^{(i)}) + \log p_{\theta}(\mathbf z)+ \log p_{\theta}(\mathbf x^{(i)}|\mathbf z) \right]\\
&=
\mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}\left[-\log \frac { q_{\phi}( \mathbf z | \mathbf x^{(i)}) }{p_{\theta}(\mathbf z )} + \log p_{\theta}(\mathbf x^{(i)}|\mathbf z)\right]\\
&=\mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}\left[-\log \frac { q_{\phi}( \mathbf z | \mathbf x^{(i)})}{p_{\theta}(\mathbf z )}\right] + \mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}\left[\log p_{\theta}(\mathbf x^{(i)}|\mathbf z)\right]\\
&=
-D_{KL}(q_{\phi}(\mathbf z | \mathbf x^{(i)}) || p_{\theta}(\mathbf z)) + \mathbb E_{q_{\phi}(\mathbf z|\mathbf x^{(i)})}\left[\log p_{\theta}(\mathbf x^{(i)}|\mathbf z)\right]
\end{align}
$$
)
我们希望相对变分参数$\phi$和生成参数$\theta$对下界$\mathcal L(\phi, \phi; \mathbf x^{(i)})$进行微分和优化，但是下界相对于$\phi$的梯度的求解有一点困难，对这类问题的朴素的蒙特卡洛梯度估计是：$\nabla_{\phi} \mathbb E_{q_{\phi}(\mathbf z)}[f(\mathbf z)] = \mathbb E_{q_{\phi}(\mathbf z)}[f(\mathbf z) \nabla_{q_{\phi}(\mathbf z)}{\log q_{\phi}(\mathbf z)}]\simeq \frac 1 L \sum_{l=1}^L[f(\mathbf z^{(l)}) \nabla_{q_{\phi}(\mathbf z^{(l)})}{\log q_{\phi}(\mathbf z^{(l)})}]$，其中$\mathbf z^{(l)}\sim q_{\phi}(\mathbf z | \mathbf x^{(i)})$，这种估计会有很大的方差，因此是不实际的
(
首先，考虑一个函数$f(x)$，它满足$\forall x, f(x) \ge 0$，则有：
$$\begin{align}
&\frac {df(x)} {dx}\\
=&\frac {de^{\log f(x)}}{dx}\\
=&\frac {de^{\log f(x)}}{d \log f(x)}\frac {d\log f(x)}{dx}\\
=&e^{\log f(x)}\frac {d\log f(x)}{dx}\\
=&f(x) \frac {d\log f(x)}{dx}
\end{align}$$
因此，对于一个概率分布$q_{\phi}(\mathbf z)$，它显然满足$\forall \mathbf z, q_{\phi}(\mathbf z) \ge 0$，则有：
$$\begin{align}
&\nabla_{\phi}q_{\phi}(\mathbf z)\\
=&\nabla_{\phi} e^{\log q_{\phi}(\mathbf z)}\\
=&\nabla_{\log q_{\phi}(\mathbf z)}e^{\log q_{\phi}(\mathbf z) }\cdot \nabla_{\phi}\log q_{\phi}(\mathbf z)\\
=&e^{\log q_{\phi}(\mathbf z) }\cdot \nabla_{\phi}\log q_{\phi}(\mathbf z) \\
=&q_{\phi}(\mathbf z) \cdot \nabla_{\phi}\log q_{\phi}(\mathbf z) 
\end{align}$$
因此，对于$\nabla_{\phi} \mathbb E_{q_{\phi}(\mathbf z)}[f(\mathbf z)]$，可以推断：
$$\begin{align}
&\nabla_{\phi} \mathbb E_{q_{\phi}(\mathbf z)}[f(\mathbf z)]\\
=&\nabla_{\phi} \int_{\mathbf z\sim q_{\phi}(\mathbf z)} q_{\phi}(\mathbf z)f(\mathbf z)d\mathbf z\\
=&\int_{\mathbf z\sim q_{\phi}(\mathbf z)} f(\mathbf z)\nabla_{\phi}q_{\phi}(\mathbf z)d\mathbf z\\
=&\int_{\mathbf z\sim q_{\phi}(\mathbf z)} f(\mathbf z)q_{\phi}(\mathbf z)\nabla_{\phi}\log q_{\phi}(\mathbf z)d\mathbf z\\
=&\int_{\mathbf z\sim q_{\phi}(\mathbf z)}q_{\phi}(\mathbf z) f(\mathbf z)\nabla_{\phi}\log q_{\phi}(\mathbf z)d\mathbf z\\
=&\mathbb E_{q_{\phi}(\mathbf z)}[f(\mathbf z)\nabla_{\phi}\log q_{\phi}(\mathbf z)]
\end{align}$$
)
## 2.3 The SGVB estimator and AEVB algorithm
本节，我们介绍对下界和下界相对于它的参数的导数更实用的估计，
回忆一下我们假设了一个形式为$q_{\phi}(\mathbf z | \mathbf x$)的近似后验分布
(本节的技巧也可以应用于$q_{\phi}(\mathbf z)$的情况，也就是不条件于$\mathbf x$的情况)

在一些简单的条件(mild conditions)下(这些条件具体见2.4部分)，对于一个近似后验分布$q_{\phi}(\mathbf z|\mathbf x)$，我们可以用和一个辅助的噪声变量$\boldsymbol {\epsilon}$相关的一个可微的变换(differentiable transformation)$g_{\phi}(\boldsymbol \epsilon, \mathbf x)$，来再参数化(reparameterize)随机变量$\tilde {\mathbf z}\sim q_{\phi}(\mathbf z | \mathbf x)$：
$$\tilde {\mathbf z} = g_{\phi}(\boldsymbol \epsilon,\mathbf x)\quad\text{with}\quad\boldsymbol \epsilon \sim p(\boldsymbol \epsilon)\tag{4}$$
2.4节给出了选择恰当的分布$p(\boldsymbol \epsilon)$和函数$g_{\phi}(\boldsymbol \epsilon, \mathbf x)$的一般策略，此时我们可以对某个函数$f(\mathbf z)$相较于$q_{\phi}(\mathbf z | \mathbf x)$的期望进行以下的蒙特卡洛估计：
$$\begin{align}
&\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}[f(\mathbf z)]\\
=& \mathbb E_{p(\boldsymbol \epsilon)}\left[f(g_{\phi}(\boldsymbol \epsilon,\mathbf x^{(i)}))\right]
\simeq\frac 1 L \sum_{l=1}^L f(g_{\phi}(\boldsymbol \epsilon,\mathbf x^{(i)}))\quad\text{where}\quad \boldsymbol \epsilon^{(l)}\sim p(\boldsymbol \epsilon)
\end{align}\tag{5}$$
我们将该技巧应用于变分下界(eq. (2))，得到我们通用的(generic)随机梯度变分贝叶斯估计(Stochastic Gradient Variational Bayes SGVB estimator)$\tilde {\mathcal L}^A(\theta, \phi; \mathbf x^{(i)})\simeq \mathcal L({\theta, \phi;\mathbf x^{(i)}})$：
$$\begin{align}
&\tilde {\mathcal L}^A(\theta, \phi; \mathbf x^{(i)})=\frac 1 L\sum_{l=1}^L\log p_{\theta}(\mathbf x^{(i)}, \mathbf z^{(i,l)}) -\log q_{\phi}(\mathbf z^{(i,l)}|\mathbf x^{(i)}) \\
&\text{where}\quad \mathbf z^{(i,l)} = g_{\phi}(\boldsymbol \epsilon ^{(i,l)},\mathbf x^{(i)})\quad \text{and}\quad \boldsymbol \epsilon^{(l)}\sim p(\boldsymbol \epsilon)
\end{align}\tag{6}$$

eq. (3)的KL散度$D_{KL}(q_{\phi}(\mathbf z | \mathbf x^{(i)})|| p_{\theta}(\mathbf z))$可以被解析式地积分(be integrated analytically)，因此只有期望的重构误差(expected reconstruction error)$\mathbb E_{q_{\phi}(\mathbf z | \mathbf x^{(i)})}[\log p_{\theta}(\mathbf x^{(i)}|\mathbf z)]$需要进行采样估计，因此，KL散度项可以被解释为对$\phi$的正则化，它鼓励近似后验分布接近先验分布$p_{\theta}(\mathbf z)$，因此，对应于eq. (3)，我们可以得到第二个版本的SGVB估计$\tilde {\mathcal L}^B(\theta, \phi; \mathbf x^{(i)})\simeq \mathcal L({\theta, \phi;\mathbf x^{(i)}})$，它一般比通用估计的方差要更小：
$$\begin{align}
&\tilde {\mathcal L}^B(\theta, \phi; \mathbf x^{(i)})= -D_{KL}(q_{\phi}(\mathbf z | \mathbf x^{(i)})||p_{\theta}(\mathbf z)) + \frac 1 L \sum_{l=1}^L(\log p_{\theta}(\mathbf x^{(i)}|\mathbf z^{(i,l)}))\\
&\text{where}\quad \mathbf z^{(i,l)} = g_{\phi}(\boldsymbol \epsilon ^{(i,l)},\mathbf x^{(i)})\quad \text{and}\quad \boldsymbol \epsilon^{(l)}\sim p(\boldsymbol \epsilon)
\end{align}\tag{7}$$
给定含有$N$个数据点中的$\mathbf X$中采样出的多个数据点(小批量)，我们可以基于小批量，构造整个数据集的边际似然下界的估计：
$$\mathcal L(\theta, \phi;\mathbf X)\simeq \tilde {\mathcal L}^M(\theta, \phi;\mathbf X^M) = \frac N M\sum_{i=1}^M\tilde {\mathcal L}(\theta, \phi; \mathbf x^{(i)})\tag{8}$$
其中小批量$\mathbf X^M = \{\mathbf x^{(i)}\}_{i=1}^M$即由从$\mathbf X$中随机抽取的$M$个数据点构成
在我们的实验中，我们发现估计每个数据点的边际似然下界需要的采样数量$L$可以设定为1，只要小批量大小$M$足够大，例如$M=100$

得到整个数据集的边际似然下界估计$\tilde {\mathcal L}(\theta,\phi; \mathbf X^M)$后，我们就可以计算导数$\nabla_{\theta,\phi}\tilde{\mathcal L}(\theta,\phi,\mathbf X^M)$，然后利用随机优化算法例如SGD或Adagrad进行优化，如Algorithm 1所示
![[VAE-Algoritm1.png]]

考虑eq. (7)所定义的目标函数，我们可以发现和自动编码器(auto-encoders)的联系，eq. (7)的第一项(近似后验分布和先验分布之间的KL散度)的作用是正则项(regularizer)，而第二项是一个期望的负重构损失(expected negative reconstruction error)，其中，
函数$g_{\phi}(\cdot)$的作用是将一个数据点$\mathbf x^{(i)}$和一个随机噪声向量$\boldsymbol \epsilon^{(l)}$映射到一个(和数据点$\mathbf x^{(i)}$相关的)近似后验分布中的隐变量样本$\mathbf z^{(i,l)} = g_{\phi}(\boldsymbol \epsilon^{(l)}, \mathbf x^{(i)})\quad\text{where}\quad \mathbf z^{(i,l)}\sim q_{\phi}(\mathbf z | \mathbf x^{(i)})$
随后，样本$\mathbf z^{(i,l)}$会作为函数$\log p_{\theta}(\mathbf x^{(i)}| \mathbf z^{(i,l)})$的输入，该函数定义了在给定$\mathbf z^{(i,l)}$的时候，数据点$\mathbf x^{(i)}$在生成式模型下的概率密度/质量，
因此，第二项在自动编码器的术语下可以称为是一个期望的负重构误差(reconstruction error)
## 2.4 The reparameterization trick
在从$q_{\phi}(\mathbf z | \mathbf x)$中生成样本时，我们采用了一种替代办法，即再参数化技巧，
再参数化技巧的实质很简单，令$\mathbf z$是一个连续变量，而$\mathbf z \sim q_{\phi}(\mathbf z | \mathbf x)$是某个条件分布，我们常常可以将随机变量(random variable)$\mathbf z$表示成一个确定的变量(deterministic variable)$\mathbf z = g_{\phi}(\boldsymbol \epsilon, \mathbf x)$，其中$\boldsymbol \epsilon$是一个辅助变量，从属于独立的边际分布$p(\boldsymbol \epsilon)$，而$g_{\phi}(\cdot)$是某个向量值函数，由$\phi$参数化

重参数化技巧使我们可以重写相对于$q_{\phi}(\mathbf z | \mathbf x)$的期望，以让对该期望的蒙特卡洛估计相对于$\phi$是可微的，证明如下：
给定确定的映射(deterministic mapping)$\mathbf z = g_{\phi}(\boldsymbol \epsilon, \mathbf x)$，
我们知道$q_{\phi}(\mathbf z | \mathbf x)\prod_i dz_i = p(\boldsymbol \epsilon)\prod_i d\epsilon_i$，
因此$^1$，$\int q_{\phi}(\mathbf z | \mathbf x) f(\mathbf z)d\mathbf z = \int p(\boldsymbol \epsilon)f(\mathbf z)d\boldsymbol \epsilon = \int p(\boldsymbol\epsilon)f(g_{\phi}(\boldsymbol \epsilon,\mathbf x))d\boldsymbol \epsilon$，随后，我们就可以对这个积分(期望)构造一个可微的估计(differentiable estimator)：$\int q_{\phi}(\mathbf z | \mathbf x)f(\mathbf z)d\mathbf z \simeq \frac 1 L \sum_{l=1}^L f(g_{\phi}(\mathbf x, \boldsymbol \epsilon^{(l)}))\quad{\text{where}}\quad \boldsymbol \epsilon^{(l)}\sim p(\boldsymbol \epsilon)$，
在2.3节，我们就应用了这一技巧，得到了对变分下界的可微估计

例如，随机变量$z$服从一元高斯分布$z\sim p(z|x)=\mathcal N(\mu, \sigma^2)$，此时，一个对$z$的合理的再参数化写为$z = \mu + \sigma \epsilon$，其中$\epsilon$是一个辅助的噪声变量，满足$\epsilon \sim \mathcal N(0,1)$，因此，
$\mathbb E_{\mathcal N(z;\mu,\sigma^2)}[f(z)] = \mathbb E_{\mathcal N(\epsilon;0,1)}[f(\mu + \sigma\epsilon)]\simeq \frac 1 L \sum_{l=1}^L f(\mu + \sigma\epsilon^{(l)})\quad \text{where} \quad \epsilon^{(l)} \sim \mathcal N(0,1)$

对于哪一类的$q_{\phi}(\mathbf z |\mathbf x)$，我们可以选择一个可微的变换$g_{\phi}(\cdot)$和辅助变量$\boldsymbol \epsilon \sim p (\boldsymbol \epsilon)$对其进行再参数化？三种基本的方法是：
1. $q_{\phi}(\mathbf z | \mathbf x)$有可解的逆累计分布函数(Tractable inverse CDF)
    在这种情况下，令$\boldsymbol \epsilon \in \mathcal U(\mathbf 0, \mathbf I)$，令$g_{\phi}(\boldsymbol \epsilon, \mathbf x)$为$q_{\phi}(\mathbf z | \mathbf x)$的逆CDF
    具有可解的逆累计分布函数的分布有：指数、柯西、逻辑、瑞利、帕累托、韦伯、倒数、戈姆珀茨、格贝尔和埃尔朗分布
2. $q_{\phi}(\mathbf z |\mathbf x)$类似于高斯分布(Analogous to the Gaussian example)
    对于任意的“位置-尺度”分布族("location-scale" family of distributions)，我们可以选择一个标准分布($\text{location}=0$, $\text{scale}=1$)作为辅助变量$\boldsymbol \epsilon$，然后令$g(\cdot) = \text{location} + \text{scale}\cdot \boldsymbol \epsilon$
    属于这类分布的例子有：拉普拉斯、椭圆、学生t、逻辑、均匀、三角和高斯分布
3. $q_{\phi}(\mathbf z |\mathbf x)$是组合形式(Composition)
    可以将随机变量表示为辅助变量的不同变换
    属于这类分布的例子有：对数正态(正态分布变量的指数)、伽马(指数分布变量的求和)、狄利克雷(伽马分布变量的加权和)、贝塔、卡方和F分布
如果$q_{\phi}(\mathbf z |\mathbf x)$三种条件都不满足，可以对逆CDF进行近似，近似的计算时间复杂度和PDF类似
---
$^1$注意对于微元(infinitesimals)，我们遵从标注惯例$d\mathbf z = \prod_i dz_i$
# 3 Example: Variational Auto-Encoder
本节中，我们给出一个示例，我们用NN作为概率编码器(probabilistic encoder)$q_{\phi}(\mathbf z |\mathbf x)$(对生成式模型$p_{\theta}(\mathbf x, \mathbf z)$的后验分布的近似)，其中参数$\phi$和$\theta$在AEVB算法下共同优化

令隐变量的先验分布(prior)是中心化的各向同性多元高斯分布(centered isotropic multivariate Gaussian)$p_{\theta}(\mathbf z) = \mathcal N(\mathbf z; \mathbf 0, \mathbf I)$，注意在这种情况下，隐变量的先验分布没有参数(lacks parameters)，
令$p_{\theta}(\mathbf x |\mathbf z)$是一个多元高斯分布(对于实数值数据 real-valued data)或伯努利分布(对于二进制数据 binary data)，其分布参数是由$\mathbf z$通过一个MLP(一个单隐藏层的全连接网络)计算得出的，注意在这种情况下，真实的后验分布$p_{\theta}(\mathbf z | \mathbf x)$是不可解的，
$q_{\phi}(\mathbf z |\mathbf x)$(对真实后验分布的近似)的形式有很大的自由度，我们假设真实的(且不可解)后验分布采取一个近似高斯形式(approximate Gaussian form)，且具有近似对角的协方差(approximately diagonal covariance)，在这种情况下，我们令变分近似后验分布(variational approximate posterior)是一个具有对角协方差结构(diagonal covariance structure)的多元高斯分布：
$$\log q_{\phi}(\mathbf z|\mathbf x^{(i)}) = \log \mathcal N(\mathbf z;\boldsymbol \mu^{(i)},\boldsymbol \sigma^{2(i)}\mathbf I)\tag{9}$$
近似后验分布的均值和标准差$\boldsymbol \mu^{(i)}$和$\boldsymbol \sigma^{(i)}$是MLP编码器的输出，即它们是关于数据点$\mathbf x^{(i)}$和变分参数$\phi$的非线性函数

如我们在2.4节所解释的，我们从近似后验分布中采样$\mathbf z^{(i,l)}\sim q_{\phi}(\mathbf z |\mathbf x^{(i)})$的方法是使用$\mathbf z^{(i,l)} = g_{\phi}(\mathbf x^{(i)}, \boldsymbol \epsilon^{(l)}) = \boldsymbol \mu^{(i)} + \boldsymbol \sigma^{(i)} \odot \boldsymbol \epsilon^{(l)}\quad \text{where}\quad \boldsymbol \epsilon^{(l)}\sim \mathcal N(\mathbf 0, \mathbf I)$，
(
随机采样是一种随机过程，它本身是不可导的，
使用再参数化技巧之前，我们需要对$\mathbf z^{(i,l)}$进行随机采样，此时，$\mathbf z^{(i,l)}$的生成相对于参数$\phi$(即$\boldsymbol \mu^{(i)}$和$\boldsymbol \sigma^{(i)}$)是带有随机性的，此时$\mathbf z^{(i,l)}$的梯度不能传递给参数$\phi$，
使用再参数化技巧之后，我们需要对$\boldsymbol \epsilon^{(l)}$进行随机采样，此时，$\mathbf z^{(i,l)}$的生成相对于参数$\phi$(即$\boldsymbol \mu^{(i)}$和$\boldsymbol \sigma^{(i)}$)是确定性的，随机性仅来自于$\boldsymbol \epsilon^{(l)}$，而它与参数$\phi$不相关，此时梯度可以顺利通过$\mathbf z^{(i,l)}$传递给参数$\phi$

在重参数化之前，$\mathbf z^{(i,l)}$不是直接关于$\phi$的函数，$\mathbf z^{(i,l)}$的概率$q_{\phi}(\mathbf z^{(i,l)} | \mathbf x^{i})$是直接关于$\phi$的函数,
在重参数化之后，$\mathbf z^{(i,l)}$是直接关于$\phi$的函数
)
在我们的示例模型中，隐变量的先验分布$p_{\theta}(\mathbf z)$和近似后验分布$q_{\phi}(\mathbf z|\mathbf x)$都是高斯分布，在这种情况下，我们使用 eq.(7)的估计，式中的KL散度项可以在不进行估计的情况下计算且微分，因此，最后得到的在数据点$\mathbf x^{(i)}$的估计是：
$$\begin{align}
&\tilde {\mathcal L}(\theta, \phi; \mathbf x^{(i)})\simeq \frac 1 2\sum_{j=1}^J(1+\log((\sigma_j^{(i)})^2)-(\mu_j^{(i)})^2-(\sigma_{j}^{(i)})^2) \\
&\quad\quad\quad\quad\quad\quad+ \frac 1 L \sum_{l=1}^L(\log p_{\theta}(\mathbf x^{(i)}|\mathbf z^{(i,l)}))\\
&\text{where}\quad \mathbf z^{(i,l)} = g_{\phi}(\boldsymbol \epsilon ^{(i,l)},\mathbf x^{(i)})\quad \text{and}\quad \boldsymbol \epsilon^{(l)}\sim p(\boldsymbol \epsilon)
\end{align}\tag{10}$$
其中解码项(decoding term)$\log p_{\theta}(\mathbf x^{(i)}| \mathbf z^{(i,l)})$是伯努利或高斯MLP，取决于我们建模的数据类型(即$\mathbf x$的数据类型)
# 4 Related work
据我们所知，wake-sleep算法是其他工作中唯一也适用于一般类别的连续隐变量模型(general class of continuous latent variable models)的在线学习方法，和我们的方法一样，wake-sleep算法也采用了一个识别模型来近似真实的后验分布，
wake-sleep算法的缺点在于它需要同时优化两个目标函数(concurrent optimization)，而同时优化这两个目标函数也不等价于优化边际似然(或它的界限)，
wake-sleep算法的优点在于它也适用于具有离散隐变量的模型，
wake-sleep算法和AEVB在每个数据点上的计算复杂度相同

最近，随机变分推理(stochastic variational inference)受到了越来越多的关注，[BJP12]引入了控制变量方案(control variate schemes)来减少我们在2.1节讨论的朴素(naive)梯度估计的方差，并将该方法应用于后验分布的指数族近似(exponential family approximations)；
在[RGB13]中，一些通用方法，即控制变量方案，被引入用于减少原始梯度估计的方差(original gradient estimator)；
[SK13]中使用了和本文中使用的类似的再参数化方法，在随机变分推理算法的高效版本中用于学习指数族近似分布的自然参数(natural parameters of exponential-family approximating distributions)

AEVB算法揭示了有向概率模型(directed probabilistic model)(用变分目标 variational objective 训练)和自动编码器之间的联系，
已经有人揭示了线性自动编码器和某种类别的生成式线性高斯模型(generative linear-Gaussian models)之间的联系，[Row98]中展示了PCA对应于具有先验$p(\mathbf z) = \mathcal N(0,\mathbf I)$，以及条件分布$p(\mathbf x|\mathbf z)=\mathcal N(\mathbf x; \mathbf W\mathbf z,\boldsymbol \epsilon\mathbf I)$的线性高斯模型的最大似然解，特别是在$\boldsymbol \epsilon$是无穷小的情况下

在自动编码器的相关最近工作中，[VLL+10]展示了未正则化(unregularized)的自动编码器的训练标准(training criterion)对应于最大化输入$X$和隐表示$Z$之间互信息的一个下界(见infomax原则[Lin89])，最大化互信息(w.r.t parameters)等价于最大化条件熵，而条件熵则是自动编码模型下数据的期望对数似然的下界[VLL+10]，即负的重构误差；
然而，这种重构标准(reconstruction criterion)本身并不足以学习有用的表示[BCV13]，[BCV13]提出正则化技术(regularization techniques)使自动编码器能够学习有用的表示，如去噪、收缩和稀疏自动编码器变体(denoising, contractive and sparse autoencoder variants)，
SGVB目标包含由变分界限(variational bound)决定的正则化项(见 eq.(10))，且不需要常见学习有用表示所需的麻烦的正则化超参数(regularization hyperparamters)；
相关的工作还有编码器-解码器架构，如预测稀疏分解(predictive sparse decomposition PSD)[KRL08]，我们从中获得了一些灵感；
同样相关的是最近引入的生成式随机网络(generative stochastic networks)[BTL13]，其中带噪声的自动编码器(noisy auto-encoders)学习了从数据分布中抽样的马尔可夫链的转移算子(transition operator)；
[SL10]中采用了一个识别模型，用于深度玻尔兹曼机的高效学习；
这些方法针对的是非归一化模型(unnormalized models)(即像玻尔兹曼机这样的无向模型 undirected models)或仅限于稀疏编码模型(sparse coding models)，我们提出的则是用于学习一般类别(a general class of)的有向概率模型的算法

最近提出的DARN方法[GMW13]，也使用自动编码结构学习有向概率模型，但他们的方法适用于二进制隐变量；
[RMW14]也在本文中描述的再参数化技巧的基础上，建立了自动编码器、有向概率模型和随机变分推理之间的联系，他们的工作是独立于我们进行的，为AEVB提供了另一个视角
# 5 Experiments
我们用来自MNIST和Frey Face数据集的图像训练生成式模型，并根据变分下界(variational lower bound)和估计边际似然(estimated marginal likelihood)和其他学习算法比较

生成式模型(generative model)(编码器 encoder)和变分近似(variational approximation)(解码器 decoder)的形式都来自我们在第三节中所描述的，
其中编码器和解码器具有相同数量的隐藏单元(hidden units)，
由于Frey Face数据是连续的，我们使用了一个具有高斯输出(Gaussian output)的解码器，和编码器一致，不同的地方在于我们通过对解码器的输出使用Sigmoid激活函数，将其均值限制在区间$(0,1)$内，

参数使用随机梯度上升更新，梯度通过对下界估计$\nabla_{\theta,\phi}\mathcal L(\theta, \phi; \mathbf X)$求微分计算得到(见Algorithm 1)，同时梯度会加上一项小的权重衰退项，对应于先验$p(\theta)=\mathcal N(0, \mathbf I)$，
优化该目标等价于近似最大后验估计(approximate MAP estimation)，其中似然的梯度通过下界的梯度近似

我们比较了AEVB和wake-sleep算法的性能，
我们采用相同的编码器(也称为识别模型)，所有的参数，包括变分参数和生成参数，都通过从$\mathcal N(0, 0.01)$中随机采样初始化，然后使用MAP准则(criterion)共同随机优化(jointly stochastically optimized)，
步长(stepsize)使用Adagrad[DHS10]进行调整，Adgrad全局步长参数(global stepsize parameters)会基于前几次迭代在训练集上的表现从$\{0.01, 0.02, 0.1\}$中选择，
小批量大小$M=100$，每个数据点采样数量$L=1$

**Likelihood lower bound**
在MNIST上，我们训练的生成模型(解码器)和对应的编码器(即识别模型)有500个隐藏单元，在Frey Face上则是200个(为了防止过拟合，因为这个数据集很小)，隐藏单元数量的选择基于自动编码机的先前文献，而不同算法的相对性能对于隐藏单元数量的选择并不十分敏感
Figure2展示了变分下界的比较结果
![[VAE-Fig2.png]]
有趣的是，过多的(superfluous)隐变量没有导致过拟合，这可以由变分界限的正则化性质解释(regularizing nature of the variational bound)

**Marginal likelihood**
对于非常低维度的隐空间，使用MCMC估计器估计学习到的生成模型的边际似然是可能的，
我们仍使用NN作为编码器和解码器，并使用100个隐藏单元和3个隐变量，
注意对于更高维度的隐空间，MCMC估计就会变得不可靠了

我们使用MNIST数据集，将AEVB与Wake-Sleep方法和具有混合蒙特卡洛(Hybrid Monte Carlo HMC)[DKPR87]采样器(sampler)的蒙特卡洛EM(MCEM)方法进行比较，我们也比较了这三个算法的收敛速度，
比较结果见Figure3，细节见附录
![[VAE-Fig3.png]]

**Visualisation of high-dimensional data**
如果我们选择一个非常低维度的隐空间(例如2D)，我们可以使用学习到的编码器(识别模型)，将高维度数据映射到低维度流形，附录A可视化了MNIST和Frey face数据集的2D隐流形(latent manifolds)
# 6 Conclusion
我们介绍了新颖的变分下界估计器：随机梯度变分贝叶斯(SGVB)，用于在连续隐变量下的高效近似推理，我们所提出的变分下界估计器可以用标准的随机梯度方法直接微分并优化，
对于独立同分布且每个数据点都具有连续隐变量的数据集，我们介绍了对其进行高效推理和学习的算法变分自动编码器(AEVB)，AEVB使用SGVB估计器学习近似推理模型(approximate inference model)，
实验结果反映了理论的优越性
# 7 Future work
因为SGVB估计器和AEVB算法可以应用于几乎所有带隐变量的学习和推理问题，未来的方向也有很多，包括：
1. 利用深度NN(例如CNN)作为编码器和解码器，学习层次化的生成式架构(hierarchical generative architectures)，使用AEVB共同训练
2. 时序模型(time-series models)，例如动态贝叶斯网络
3. 将SGVB应用于全局参数
4. 带有隐变量的有监督模型，可以用于学习复杂的噪声分布(noise distributions)
# A Visualisations
# B Solution of $-D_{KL}(q_{\phi}(\mathbf z) || p_{\theta}(\mathbf z))$, Gaussian Case
变分下界(variational lower bound)(即我们要最大化的目标函数)，包含了一个KL散度项，这个项一般可以解析式地积分，
我们在此给出当先验是$p_{\theta}(\mathbf z) = \mathcal N(\mathbf 0, \mathbf I)$以及后验近似$q_{\phi}(\mathbf z | \mathbf x^{(i)})$也是高斯分布时的解，令$J$是$\mathbf z$的维度，令$\boldsymbol \mu$和$\boldsymbol \sigma$表示在数据点$i$上评估的变分均值和标准差，令$\mu_j$和$\sigma_j$表示这些向量的第$j$个元素，则：
$$\begin{align}
\int q_{\phi}(\mathbf z) \log p_{\theta}(\mathbf z)d\mathbf z &= \int\mathcal N(\mathbf z; \boldsymbol \mu, \boldsymbol \sigma^2)\log \mathcal N(\mathbf z; \mathbf 0, \mathbf I)d\mathbf z\\
&=- \frac J 2\log(2\pi)- \frac 1 2\sum_{j=1}^J(\mu_j^2 + \sigma_j^2)
\end{align}$$
(
我们假设了近似后验分布的形式为多元高斯分布，而$J$维的多元高斯分布的概率密度函数写为：
$$q_{\phi}(\mathbf z |\mathbf x^{(i)}) =\mathcal N(\mathbf z;\boldsymbol \mu, \mathbf \Sigma) = \frac 1 {(2\pi)^{J/2}|\mathbf \Sigma|^{1/2}}\exp(-\frac 1 2(\mathbf z-\boldsymbol \mu)^T\mathbf \Sigma^{-1} (\mathbf z-\boldsymbol \mu))$$
取对数，得到：
$$\begin{align}
\log \mathcal N(\mathbf z;\boldsymbol \mu, \mathbf \Sigma) &= 
-\frac J 2 \log(2\pi) - \frac 1 2 \log |\mathbf \Sigma| -\frac 1 2(\mathbf z-\boldsymbol \mu)^T\mathbf \Sigma^{-1} (\mathbf z-\boldsymbol \mu)\\
&=-\frac J 2 \log(2\pi)-\frac 1 2(\log |\mathbf \Sigma| +(\mathbf z-\boldsymbol \mu)^T\mathbf \Sigma^{-1} (\mathbf z-\boldsymbol \mu ))
\end{align}$$
先验分布则是标准的多元高斯分布，容易得到：
$$p_{\theta}(\mathbf z) =\mathcal N(\mathbf z;\mathbf 0, \mathbf I) = \frac 1 {(2\pi)^{J/2}|\mathbf I|^{1/2}}\exp(-\frac 1 2\mathbf z^T\mathbf I^{-1} \mathbf z)$$
即：
$$\begin{align}
\log \mathcal N(\mathbf z;\boldsymbol \mu, \mathbf \Sigma)
&=-\frac J 2 \log(2\pi)-\frac 1 2(\mathbf z^T\mathbf z)
\end{align}$$
代入原积分表达式，得到：
$$\begin{align}
&\int (-\frac J 2\log(2\pi) - \frac 1 2 (\mathbf z^T\mathbf z))\mathcal N(\mathbf z; \boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&-\frac J 2\int\log(2\pi)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z - \frac 1 2\int (\mathbf z^T\mathbf z)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&\frac J 2\log (2\pi) - \frac 1 2\int (\mathbf z^T\mathbf z)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z
\end{align}$$
考虑项$- \frac 1 2\int (\mathbf z^T\mathbf z)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z$，我们将$\mathbf z^T\mathbf z$展开：
$$\begin{align}
&-\frac 1 2\int (\mathbf z^T\mathbf z)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&-\frac 1 2\int \sum_{j=1}^J(z_j^2)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z
\end{align}$$因为$\mathbf z$是多维的，且各维度独立，我们可以对每个维度单独进行积分：
$$\begin{align}
&-\frac 1 2\int (\mathbf z^T\mathbf z)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&-\frac 1 2\int \sum_{j=1}^J(z_j^2)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&-\frac 1 2\sum_{j=1}^J\int z_j^2 \mathcal N(z_j;\mu_j, \sigma_j^2)dz_j
\end{align}$$
对于一个正态分布，存在结论：
$$\begin{align}
Var[X] &= E[(X-E[X])^2] \\
&=E[X^2 - 2XE[X] + E^2[X]\\
&=E[X^2] - 2E^2[X] + E^2[X]\\
&=E[X^2] - E^2[X]\\
E[X^2] &=Var[X] + E^2[X] = \sigma^2 + \mu^2
\end{align}$$
因此容易得到：
$$\int z_j^2\mathcal N(z_j;\mu_j,  \sigma_j^2)dz_j = \mu_j^2 + \sigma_j^2$$
故：
$$\begin{align}
&-\frac 1 2\int (\mathbf z^T\mathbf z)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&-\frac 1 2\sum_{j=1}^J\int z_j^2 \mathcal N(z_j;\mu_j, \sigma_j^2)dz_j\\
=&-\frac 1 2\sum_{j=1}^J(\mu_j^2 + \sigma_j^2)
\end{align}$$
)
并且：
$$\begin{align}
\int q_{\phi}(\mathbf z) \log q_{\phi}(\mathbf z) d\mathbf z &= \int\mathcal N(\mathbf z;\boldsymbol \mu,\boldsymbol \sigma^2)\log\mathcal N(\mathbf z;\boldsymbol \mu,\boldsymbol \sigma^2)d\mathbf z\\
&=-\frac J 2\log(2\pi) - \frac 1 2\sum_{j=1}^J(1 + \log \sigma_j^2)
\end{align}$$
(
对数概率密度函数：
$$\begin{align}
\log \mathcal N(\mathbf z;\boldsymbol \mu, \mathbf \Sigma) &= 
-\frac J 2 \log(2\pi) - \frac 1 2 \log |\mathbf \Sigma| -\frac 1 2(\mathbf z-\boldsymbol \mu)^T\mathbf \Sigma^{-1} (\mathbf z-\boldsymbol \mu)\\
&=-\frac J 2 \log(2\pi)-\frac 1 2(\log |\mathbf \Sigma| +(\mathbf z-\boldsymbol \mu)^T\mathbf \Sigma^{-1} (\mathbf z-\boldsymbol \mu ))
\end{align}$$
后验近似分布写为$\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)$，其协方差矩阵是对角阵，因此$|\mathbf \Sigma | = \prod_{j=1}^J \sigma_j^2$，
故显然有：
$$\begin{align}
\log |\mathbf \Sigma | &= \sum_{j=1}^J\log \sigma_j^2\\
\end{align}$$
显然$\mathbf \Sigma^{-1}$也是对角阵，对角元素为$1 / \sigma_j^2$，故显然有：
$$(\mathbf z - \boldsymbol \mu)^T\mathbf \Sigma^{-1}(\mathbf z - \boldsymbol \mu) = \sum_{j=1}^J\frac {(z_j - \mu_j)^2}{\sigma_j^2}$$
代入原式，得到：
$$\begin{align}
\log \mathcal N(\mathbf z;\boldsymbol \mu, \mathbf \Sigma) &= 
-\frac J 2 \log(2\pi) - \frac 1 2 \log |\mathbf \Sigma| -\frac 1 2(\mathbf z-\boldsymbol \mu)^T\mathbf \Sigma^{-1} (\mathbf z-\boldsymbol \mu)\\
&=-\frac J 2 \log(2\pi)-\frac 1 2(\log |\mathbf \Sigma| +(\mathbf z-\boldsymbol \mu)^T\mathbf \Sigma^{-1} (\mathbf z-\boldsymbol \mu ))\\
&=-\frac J 2 \log(2\pi)-\frac 1 2(\sum_{j=1}^J\log \sigma_j^2 + \sum_{j=1}^J\frac {(z_j-\mu_j)^2}{\sigma_j^2})\\
&=-\frac J 2 \log(2\pi)-\frac 1 2\sum_{j=1}^J (\log \sigma_j^2 + \frac {(z_j-\mu_j)^2}{\sigma_j^2})\\
\end{align}$$
代入原积分表达式，得到：
$$\begin{align}
&\int (-\frac J 2 \log(2\pi)-\frac 1 2\sum_{j=1}^J \log \sigma_j^2 -\frac 1 2\sum_{j=1}^J \frac {(z_j-\mu_j)^2}{\sigma_j^2})\mathcal N(\mathbf z; \boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&\frac J 2\log (2\pi) - \frac 1 2\sum_{j=1}^J\log \sigma_j^2-\frac 1 2\int (\sum_{j=1}^J(z_j-\mu_j)^2/\sigma_j^2)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z
\end{align}$$
考虑项$- \frac 1 2\int (\sum_{j=1}^J(z_j-\mu_j)^2/\sigma_j^2)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z$，因为$\mathbf z$是多维的，且各维度独立，我们可以对每个维度单独进行积分：
$$\begin{align}
&-\frac 1 2\int (\sum_{j=1}^J(z_j-\mu_j)^2/\sigma_j^2)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&-\frac 1 2\sum_{j=1}^J\int(z_j-\mu_j)^2/\sigma_j^2 \mathcal N(z_j;\mu_j,\sigma_j^2)dz_j\\
=&-\frac 1 2\sum_{j=1}^J\frac 1 {\sigma_j^2}\int(z_j-\mu_j)^2 \mathcal N(z_j;\mu_j,\sigma_j^2)dz_j\\
\end{align}$$
对于一个正态分布，存在结论：
$$\begin{align}
E[(X-E[X])^2] &= Var[X]\\
&=\sigma^2
\end{align}$$
因此容易得到：
$$\begin{align}
&-\frac 1 2\int (\sum_{j=1}^J(z_j-\mu_j)^2/\sigma_j^2)\mathcal N(\mathbf z;\boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&-\frac 1 2\sum_{j=1}^J\frac 1 {\sigma_j^2}\int(z_j-\mu_j)^2 \mathcal N(z_j;\mu_j,\sigma_j^2)dz_j\\
=&-\frac 1 2\sum_{j=1}^J\frac 1 {\sigma_j^2}\sigma_j^2\\
=&-\frac 1 2\sum_{j=1}^J 1
\end{align}$$
代入原积分表达式，得到：
$$\begin{align}
&\int (-\frac J 2 \log(2\pi)-\frac 1 2\sum_{j=1}^J \log \sigma_j^2 -\frac 1 2\sum_{j=1}^J \frac {(z_j-\mu_j)^2}{\sigma_j^2})\mathcal N(\mathbf z; \boldsymbol \mu, \boldsymbol \sigma^2)d\mathbf z\\
=&\frac J 2\log (2\pi) - \frac 1 2\sum_{j=1}^J\log \sigma_j^2-\frac 1 2\sum_{j=1}^J 1\\
=&\frac J 2\log (2\pi) - \frac 1 2(\sum_{j=1}^J\log \sigma_j^2+\sum_{j=1}^J 1)\\
=&\frac J 2\log (2\pi) - \frac 1 2\sum_{j=1}^J(\log \sigma_j^2+ 1)\\
\end{align}$$
)
因此：
$$\begin{align}
-D_{KL}(q_{\phi}(\mathbf z) || p_{\theta}(\mathbf z) ) &=-\int q_{\phi}(\mathbf z)\log \frac {q_{\phi}(\mathbf z)}{p_{\theta}(\mathbf z)}d\mathbf z\\
&=\int q_{\phi}(\mathbf z)\log p_{\theta}(\mathbf z)d\mathbf z -\int q_{\phi}(\mathbf z) \log q_{\phi}(\mathbf z) d\mathbf z \\
&=\frac 1 2 \sum_{j=1}^J(\log \sigma_j^2 + 1 - u_j^2 - \sigma_j^2)
\end{align}$$
在识别模型$q_{\phi}(\mathbf z | \mathbf x)$中，$\boldsymbol \mu$和$\boldsymbol \sigma$是$\mathbf x$和变分参数$\phi$的函数
# C MLP's as probabilistic encoders and decoders
在变分自动编码机中，我们使用NN作为概率编码器和解码器，本文中，我们选择的是MLP，对于编码器，我们使用Gaussian输出的MLP，对于解码器，我们使用Gaussian或Bernoulli输出的MLP，取决于数据类型
## C.1 Bernoulli MLP as decoder
使用Bernoulli MLP作为解码器时，我们令$p_{\theta}(\mathbf x |\mathbf z)$为一个多元(multivariate)伯努利分布，而其概率由$\mathbf z$通过一个带单层隐藏层的全连接NN计算得到：
$$\begin{align}
&\log p_{\theta}(\mathbf x | \mathbf z) =\sum_{i=1}^D x_i\log y_i + (1-x_i)\log (1-y_i)\\
&\text{where }\mathbf y = f_{\sigma}(\mathbf W_2 \tanh(\mathbf W_1\mathbf z + \mathbf b_1) + \mathbf b_2)
\end{align}\tag{11}$$
其中$f_{\sigma}(\cdot)$是按元素运算的Sigmoid激活函数，其中$\theta = \{\mathbf W_1, \mathbf W_2, \mathbf b_1, \mathbf b_2\}$，即MLP的权重和偏置
## C.2 Gaussian MLP as encoder or decoder
使用Gaussian MLP作为编码器或解码器时，我们令$q_{\phi}(\mathbf z |\mathbf x)$或$p_{\theta}(\mathbf x | \mathbf z)$是多元高斯分布，且协方差矩阵是对角矩阵，例如：
$$\begin{align}
\log q_{\phi} (\mathbf z |\mathbf x) &= \log \mathcal N(\mathbf x;\boldsymbol \mu, \boldsymbol \sigma^2\mathbf I)\\
\text{where } \boldsymbol \mu &= \mathbf W_4\mathbf h + \mathbf b_4\\
\log \boldsymbol \sigma^2 &=\mathbf W_5\mathbf h + \mathbf b_5\\
\mathbf h&=\tanh(\mathbf W_3\mathbf x + \mathbf b_3)
\end{align}\tag{12}$$
其中变分参数$\phi = \{\mathbf W_3, \mathbf W_4, \mathbf W_5, \mathbf b_3, \mathbf b_4, \mathbf b_5\}$，即MLP的权重和偏置，当情况是解码器时，只需要将上式的$\mathbf z$和$\mathbf x$互换即可，此时MLP的权重和偏置代表参数$\theta$
# D Marginal likelihood estimator
我们对边际似然估计器进行推导，该估计器可以在采样空间(sampled space)的维度较小(小于5维)，且采取足够的(sufficient)样本的情况下，产出对边际似然较好的估计

令$p_{\theta}(\mathbf x, \mathbf z) = p_{\theta}(\mathbf z)p_{\theta}(\mathbf x | \mathbf z)$是我们要从中采样的生成式模型，对于一个给定的数据点$\mathbf x^{(i)}$，我们希望估计它的边际似然$p_{\theta}(\mathbf x^{(i)})$

估计过程包括三个阶段：
1. 使用基于梯度的MCMC(即混合蒙特卡洛，使用$\nabla_{\mathbf z} \log p_{\theta}(\mathbf z |\mathbf x) = \nabla_{\mathbf z}\log p_{\theta}(\mathbf z) + \nabla_{\mathbf z}\log p_{\theta}(\mathbf x |\mathbf z)$)，从后验中采样$L$个$\{\mathbf z^{(l)}\}$
2. 对这些样本$\{\mathbf z^{(l)}\}$拟合一个密度估计器$q(\mathbf z)$
3. 再从后验中采样$L$个新值，将这些样本，以及密度估计器$q(\mathbf z)$，放入以下估计器：
    $$p_{\theta}(\mathbf x^{(i)}) \simeq\left(\frac 1 L \sum_{l=1}^L\frac {q(\mathbf z^{(l)})}{p_{\theta}(\mathbf z)p_{\theta}(\mathbf x^{(i)}|\mathbf z^{(l)})}\right)^{-1}\quad\text{where }\mathbf z^{(l)}\sim p_{\theta}(\mathbf z | \mathbf x^{(i)})$$

推导过程：
$$\begin{align}
\frac 1 {p_{\theta}(\mathbf x^{(i)})}&= \frac {\int q(\mathbf z)d\mathbf z}{p_{\theta}(\mathbf x^{(i)})} = \frac {\int q(\mathbf z)\frac {p_{\theta}(\mathbf x^{(i)},\mathbf z)}{p_{\theta}{{(\mathbf x^{(i)}, \mathbf z)}}}d\mathbf z}{p_{\theta}(\mathbf x^{(i)})}\\
&=\int \frac {p_{\theta}(\mathbf x^{(i)}, \mathbf z)}{p_{\theta}(\mathbf x^{(i)})}\frac {q(\mathbf z)}{p_{\theta}(\mathbf x^{(i)},\mathbf z)}d\mathbf z\\
&=\int p_{\theta}(\mathbf z |\mathbf x^{(i)})\frac {q(\mathbf z)}{p_{\theta}(\mathbf x^{(i)},\mathbf z)}d\mathbf z\\
&\simeq \frac 1 L \sum_{l=1}^L \frac {q(\mathbf z^{(l)})}{p_{\theta}(\mathbf x^{(i)},\mathbf z^{(l)})}\quad \text{where}\quad \mathbf z^{(l)}\sim p_{\theta}(\mathbf z | \mathbf x^{(i)})\\
&= \frac 1 L \sum_{l=1}^L \frac {q(\mathbf z^{(l)})}{p_{\theta}(\mathbf z^{(l)})p_{\theta}(\mathbf x^{(i)}|\mathbf z^{(l)})}\quad \text{where}\quad \mathbf z^{(l)}\sim p_{\theta}(\mathbf z | \mathbf x^{(i)})\\
\end{align}$$
# E Monte Carlo EM
蒙特卡洛EM算法并没有采用编码器，相反，它使用隐变量后验分布的梯度$\nabla_{\mathbf z}\log p_{\theta}(\mathbf z | \mathbf x) = \nabla_{\mathbf z}\log p_{\theta}(\mathbf z) + \nabla_{\mathbf z} \log p_{\theta}(\mathbf x | \mathbf z)$对隐变量进行采样，蒙特卡洛EM过程包括10个HMC(混合蒙特卡洛)跃蛙步骤(leapfrog steps)，其步长(stepsize)会自动调节以使得接受率(acceptance)达到90%，然后使用获得的样本进行5次权重更新，
对于所有的算法，参数都是使用Adagrad步长(伴随退火调度 annealing schedule)进行更新的

边际似然使用训练集和测试集的前1000个数据点进行估计的，每个数据点都会从隐变量的后验分布中，使用包含4个蛙跃步骤的HMC采样50个值
# F Full VB
文中写到了我们可以同时对参数$\theta$和隐变量$\mathbf z$进行变分推理，而不仅仅对隐变量进行变分推理，我们在此对该例的估计器进行推导

令$p_{\alpha}(\theta)$是参数$\theta$的超先验分布(hyperprior)(超先验分布即先验分布参数的概率分布)，该超先验分布的参数是$\alpha$，此时边际似然可以写为：
$$\log p_{\alpha}(\mathbf X) = D_{KL}(q_{\phi}(\theta) || p_{\alpha}(\theta | \mathbf X)) + \mathcal L(\phi;\mathbf X)\tag{13}$$
其中第一个RHS项表示近似后验分布和真实后验分布之间的KL散度，第二个RHS项表示$\mathcal L(\phi; \mathbf X)$表示边际似然的变分下界：
$$\mathcal L(\phi; \mathbf X) = \int q_{\phi}(\theta)(\log p_{\theta}(\mathbf X) + \log p_{\alpha}(\theta) - \log q_{\phi}(\theta))d\theta\tag{14}$$
(
从 eq.(14)逆推 eq.(13)：
$$
\begin{align}
\log p_{\alpha}(\mathbf X) &= D_{KL}(q_{\phi}(\theta) || p_{\alpha}(\theta | \mathbf X)) + \mathcal L(\phi;\mathbf X)\\
&=\int q_{\phi}(\theta) \log \frac {q_{\phi}(\theta)}{p_{\alpha}(\theta|\mathbf X)}d\theta + \int q_{\phi}(\theta) \log \frac {p_{\theta}(\mathbf X)p_{\alpha}(\theta)}{q_{\phi}(\theta)}d\theta\\
&=\int q_{\phi}(\theta) \log \frac {q_{\phi}(\theta)}{p_{\alpha}(\theta | \mathbf X)} \frac {p_{\theta}(\mathbf X)p_{\alpha}(\theta)}{q_{\phi}(\theta)}d\theta\\
&=\int q_{\phi}(\theta) \log \frac {p_{\theta}(\mathbf X)p_{\alpha}(\theta)}{p_{\alpha}(\theta | \mathbf X)} d\theta\\
&=\int q_{\phi}(\theta) \log \frac {p_{\alpha}(\theta,\mathbf X)}{p_{\alpha}(\theta | \mathbf X)}d\theta \\
&=\int q_{\phi}(\theta) \log \frac {p_{\alpha}(\theta|\mathbf X)p_{\alpha}(\mathbf X)}{p_{\alpha}(\theta | \mathbf X)}d\theta \\
&=\int q_{\phi}(\theta)\log p_{\alpha}(\mathbf X)d\theta\\
&=\log p_{\alpha}(\mathbf X)
\end{align}$$
)
当近似后验分布等于真实后验分布时，变分下界等于边际似然，eq.(14)中项$\log p_{\theta}(\mathbf X)$由边际似然在单独的数据点上的和组成，即$\log p_{\theta}(\mathbf X) = \sum_{i=1}^N \log p_{\theta}(\mathbf x^{(i)})$，和式中的每一项都可以重写为：
$$\log p_{\theta}(\mathbf x^{(i)}) = D_{KL}(q_{\phi}(\mathbf z | \mathbf x^{(i)}) || p_{\theta}(\mathbf z | \mathbf x^{(i)})) + \mathcal L(\theta,\phi;\mathbf x^{(i)})\tag{15}$$
其中第一个RHS项是近似后验分布和真实后验分布之间的KL散度，第二个RHS项$\mathcal L(\theta, \phi; \mathbf x^{(i)})$是数据点$i$的边际似然的变分下界：
$$\mathcal L(\theta, \phi; \mathbf x^{(i)}) = \int q_{\phi}(\mathbf z | \mathbf x^{(i)}) \left(\log p_{\theta}(\mathbf x^{(i)}| \mathbf z)  + \log p_{\theta}(\mathbf z) - \log q_{\phi}(\mathbf z | \mathbf x^{(i)}) \right) d\mathbf z\tag{16}$$
(
从 eq.(16)逆推 eq.(15)，其中$\mathbf x^{(i)}$简略写为$\mathbf x$：
$$
\begin{align}
\log p_{\theta}(\mathbf x ) &= D_{KL}(q_{\phi}(\mathbf z | \mathbf x )|| p_{\theta}(\mathbf z | \mathbf x) )+ \mathcal L(\theta,\phi;\mathbf x)\\
&=\int q_{\phi}(\mathbf z | \mathbf x)\log\frac {q_{\phi}(\mathbf z | \mathbf x)}{p_{\theta}(\mathbf z | \mathbf x)}d\mathbf z + \mathcal L(\theta,\phi; \mathbf x)\\
&=\int q_{\phi}(\mathbf z | \mathbf x)\log\frac {q_{\phi}(\mathbf z | \mathbf x)}{p_{\theta}(\mathbf z | \mathbf x)}d\mathbf z + \int q_{\phi}(\mathbf z | \mathbf x) \log \frac {p_{\theta}(\mathbf x | \mathbf z)p_{\theta}(\mathbf z)}{q_{\phi}(\mathbf z | \mathbf x)}d\mathbf z\\
&=\int q_{\phi}(\mathbf z | \mathbf x) \log \frac {q_{\phi}(\mathbf z | \mathbf x)}{p_{\theta}(\mathbf z | \mathbf x)} \frac {p_{\theta}(\mathbf x, \mathbf z)}{q_{\phi}(\mathbf z | \mathbf x)}d\mathbf z\\
&=\int q_{\phi}(\mathbf z | \mathbf x)\log \frac {p_{\theta}(\mathbf x, \mathbf z)}{p_{\theta}(\mathbf z | \mathbf x)}d\mathbf z\\
&=\int q_{\phi}(\mathbf z | \mathbf x)\log p_{\theta}(\mathbf x)d\mathbf z\\
&=\log p_{\theta}(\mathbf x)
\end{align}
$$
)
eq .(16)和eq .(14)的RHS可以显式地写为三个期望的和，其中第二个和第三个成分有时可以解析式地解出，例如当$p_{\theta}(\mathbf x)$和$q_{\phi}(\mathbf z | \mathbf x)$都是高斯分布时，为了一般性，我们在此假设这些期望都是不可解的

在一定条件下(详细见文中)，对于选定的近似后验分布$q_{\phi}(\theta)$和$q_{\phi}(\mathbf z | \mathbf x)$，我们可以再参数化条件样本$\tilde {\mathbf z} \sim q_{\phi}(\mathbf z | \mathbf x)\tag{17}$为：
$$\tilde {\mathbf z } = g_{\phi}(\boldsymbol \epsilon, \mathbf x)\quad \text{with}\quad \boldsymbol \epsilon \sim p(\boldsymbol \epsilon)$$
其中我们选择先验分布$p(\boldsymbol \epsilon)$和函数$g_{\phi}(\boldsymbol \epsilon, \mathbf x)$使得以下式子成立：
$$\begin{align}
\mathcal L(\theta, \phi; \mathbf x^{(i)}) &= \int q_{\phi}(\mathbf z | \mathbf x)\left(\log p_{\theta}(\mathbf x^{(i)}|\mathbf z) + \log p_{\theta}(\mathbf z)- \log q_{\phi}(\mathbf z | \mathbf x)\right)d\mathbf z\\
&=\int p(\boldsymbol \epsilon)\left.\left(\log p_{\theta}(\mathbf x^{(i)}|\mathbf z) + \log p_{\theta}(\mathbf z)- \log q_{\phi}(\mathbf z | \mathbf x) \right)\right|_{\mathbf z=g_{\phi}(\boldsymbol \epsilon,\mathbf x^{(i)})}d\boldsymbol \epsilon
\end{align}\tag{18}$$
同样，对于近似后验分布$q_{\phi}(\theta)$可以做同样的事：
$$\tilde {\boldsymbol \theta} = h_{\boldsymbol \phi}(\boldsymbol \zeta)\quad \text{with}\quad \boldsymbol \zeta\sim p(\boldsymbol \zeta)\tag{19}$$
其中我们选择先验分布$p(\boldsymbol \zeta)$和函数$h_{\phi}(\boldsymbol \zeta)$使得以下式子成立：
$$\begin{align}
\mathcal L(\phi; \mathbf X) &= \int q_{\phi}(\boldsymbol \theta)\left(\log p_{\boldsymbol \theta}(\mathbf X) + \log p_{\boldsymbol \alpha}(\boldsymbol \theta)- \log q_{\boldsymbol \phi}(\boldsymbol \theta)\right)d\boldsymbol \theta\\
&=\int p(\boldsymbol \zeta)\left.\left(\log p_{\boldsymbol \theta}(\mathbf X) + \log p_{\boldsymbol \alpha}(\boldsymbol \theta)- \log q_{\boldsymbol \phi}(\boldsymbol \theta) \right)\right|_{\boldsymbol \theta=h_{\boldsymbol \phi}(\boldsymbol \zeta)}d\boldsymbol \zeta
\end{align}\tag{20}$$
为了让符号更简洁，我们引入一个符号简写$f_{\phi}(\mathbf x,\mathbf z, \boldsymbol \theta)$：
$$f_{\phi}(\mathbf x, \mathbf z , \boldsymbol \theta) = N\cdot (\log p_{\theta}(\mathbf x | \mathbf z) + \log p_{\theta}(\mathbf z)-\log q_{\phi}(\mathbf z | \mathbf x))+\log p_{\alpha}(\boldsymbol \theta)-\log q_{\phi}(\boldsymbol \theta)\tag{21}$$
根据eq .(20)和eq .(18)，在给定数据点$\mathbf x^{(i)}$下对变分下界的蒙特卡洛估计是：
$$\mathcal L(\boldsymbol \phi;\mathbf X) \simeq \frac 1 L\sum_{l=1}^L f_{\phi}(\mathbf x^{(l)}, g_{\phi}(\boldsymbol \epsilon^{(l)}, \mathbf x^{(l)}), h_{\phi}(\zeta^{(l)}))\tag{22}$$
其中$\boldsymbol \epsilon \sim p(\boldsymbol \epsilon)$，以及$\zeta^{(l)} \sim p(\boldsymbol \zeta)$，该估计仅依赖于从$p(\boldsymbol \epsilon$)和$p(\boldsymbol \zeta)$中的样本，该采样过程与$\phi$无关，因此，该估计可以相当于$\phi$微分，微分得到的随机梯度和随机优化方法例如SGD和Adagrad一同使用进行优化
![[VAE-Algorithm 2.png]]
## F.1 Example
令参数和隐变量的先验是中心的各项同性高斯分布，即$p_{\alpha}(\boldsymbol \theta) = \mathcal N(\boldsymbol \theta; \mathbf 0 , \mathbf I)$以及$p_{\theta}(\mathbf z) = \mathcal N(\mathbf z; \mathbf 0, \mathbf I)$，注意，此时先验分布是没有参数的，
同时，我们假设真实后验分布是近似高斯的，其协方差矩阵是近似对角的，此时，我们可以令变分近似后验分布是协方差矩阵为对角矩阵的多元高斯分布：
$$\begin{align}
\log q_{\phi}(\boldsymbol \theta) &= \log \mathcal N(\boldsymbol \theta;\boldsymbol \mu_{\theta}, \boldsymbol \sigma_{\theta}^2\mathbf I)\\
\log q_{\phi}(\mathbf z | \mathbf x)&=\log \mathcal N(\mathbf z; \boldsymbol \mu_{\mathbf z}, \boldsymbol \sigma_{\mathbf z}^2\mathbf I)
\end{align}\tag{23}$$
其中$\boldsymbol \mu_{\mathbf z}$和$\boldsymbol \sigma_{\mathbf z}$是关于$\mathbf x$的未指定形式的函数，
因为近似后验分布都是高斯分布，我们可以对其进行再参数化：
$$\begin{align}
q_{\phi}(\boldsymbol \theta)\quad&\text{as}\quad \tilde {\boldsymbol \theta} = \boldsymbol \mu_{\theta} + \boldsymbol \sigma_{\theta} \odot \boldsymbol \zeta\quad & \boldsymbol \zeta\sim \mathcal N(\mathbf 0, \mathbf I)\\
q_{\phi}(\mathbf z | \mathbf x)\quad&\text{as}\quad \tilde {\mathbf z} = \boldsymbol \mu_{\mathbf z} + \boldsymbol \sigma_{\mathbf z} \odot \boldsymbol \epsilon\quad & \boldsymbol \epsilon\sim \mathcal N(\mathbf 0, \mathbf I)\\
\end{align}\tag{23}$$

在本例中，我们可以构造一个方差更小的替代估计，因为$p_{\alpha}(\boldsymbol \theta), p_{\theta}(\mathbf z), q_{\phi}(\boldsymbol \theta), q_{\phi}(\mathbf z | \mathbf x)$都是高斯分布，因此$f_{\phi}$中相关的四个项都可以解析式地求解，得到的估计器就是：
$$\begin{align}
\mathcal L(\boldsymbol \phi;\mathbf X) 
&=\frac 1 L\sum_{l=1}^L N\left(\frac 1 2\sum_{j=1}^J \left(1 +\log (\sigma_{\mathbf z, j}^{(l)})^2-(\mu_{\mathbf z,j}^{(l)})^2-(\sigma_{\mathbf z, j}^{(l)})^2\right)+ \log p_{\theta}(\mathbf x^{(i)},\mathbf z^{(i)})\right)\\
&+\frac 1 2\sum_{j=1}^J\left (1+\log (\sigma_{\theta, j}^{(l)})^2-(\mu_{\theta,j}^{(l)})^2-(\sigma_{\theta, j}^{(l)})^2\right)
\end{align}\tag{24}$$
其中$\mu_j^{(i)}, \sigma_{j}^{(i)}$表示向量$\boldsymbol \mu^{(i)}$和$\boldsymbol \sigma^{(i)}$的第$j$个元素


