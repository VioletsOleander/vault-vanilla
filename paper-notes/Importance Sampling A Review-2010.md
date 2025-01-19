# Abstract 
We provide a short overview of Importance Sampling – a popular sampling tool used for Monte Carlo computing. We discuss its mathematical foundation and properties that determine its accuracy in Monte Carlo approximations. We review the fundamental developments in designing efficient IS for practical use. This includes parametric approximation with optimization based adaptation, sequential sampling with dynamic adaptation through resampling and population based approaches that make use of Markov chain sampling. 
> 重要性采样：蒙特卡洛计算的采样工具
> 对重要性采样的优化：在基于优化的适应上的参数近似，通过重采样来进行序列采样的动态适应，利用 Markov 链的基于群体的方法

keywords : Importance sampling, Monte Carlo approximation, sequential sampling, resampling, Markov chain sampling. 

# 1 Introduction 
Importance sampling (IS) refers to a collection of Monte Carlo methods where a mathematical expectation with respect to a target distribution is approximated by a weighted average of random draws from another distribution. 
> IS 指一系列 Monte Carlo 方法
> Monte Carlo 方法即相对于目标分布的数学期望是通过从另一个分布的随机采样的加权平均近似的

Together with Markov Chain Monte Carlo methods, IS has provided a foundation for simulation-based approaches to numerical integration since its introduction as a variance reduction technique in statistical physics (Hammersely and Morton 1954, Rosenbluth and Rosenbluth 1955). Nowadays, IS is used in a wide variety of application areas and there have been recent developments involving adaptive versions of the methodology. 
> IS 和 Markov Chain Monte Carlo 方法一起构成了基于模拟的数值积分计算的基础，它最早在统计物理学中被作为一个方差减少的方法被引入

The appeal of IS lies in a simple probability result. Let $p(x)$ be a probability density for a random variable $X$ and suppose we wish to compute an expectation $\mu_{f}=\mathbb{E}_{p}[f(X)]$ , with 

$$
\mu_{f}=\int f(x)p(x)d x.
$$ 
Then for any probability density $q(x)$ that satisfies $q(x)\,>\,0$ whenever $f(x)p(x)\neq0$ , one has 

$$
\mu_{f}=\mathbb{E}_{q}[w(X)f(X)]\tag{1}
$$ 

where $\begin{array}{r}{w(x)=\frac{p(x)}{q(x)}}\end{array}$ and now $\mathbb{E}_{q}[\cdot]$ denotes the expectation with respect to $q(x)$ . 
> 考虑随机变量 $X\sim p (x)$，我们希望计算期望 $\mu_f = \int f (x) p (x) dx$
> 考虑任意满足 $q (x)> 0$ 的分布 $q (x)$，我们可以将 $\mu_f$ 重写为 $\mu_f = \int q(x)f (x)\frac {p (x)}{q (x)} = \mathbb E_q[w (X) f (X)]$，其中 $w (x) = \frac {p (x)}{q (x)}$
> 此时，我们将考虑的是相对于 $q (x)$ 的期望

Therefore a sample of independent draws $x^{(1)},\cdot\cdot\cdot,x^{(m)}$ from $q(x)$ 
can be used to estimate $\mu_{f}$ by 

$$
\hat{\mu}_{f}=\frac{1}{m}\sum_{j=1}^{m}w(x^{(j)})f(x^{(j)}).
$$ 
> 此时，我们可以从 $q (x)$ 中采样 $x^{(1)}, \dots, x^{(m)}$ 来估计 $\hat \mu_f$

In many applications the density $p(x)$ is known only up to a normalizing constant. Here one has $w(x)\,=\,c w_{0}(x)$ where $w_{0}(x)$ can be computed exactly but the multiplicative constant $c$ is unknown. In this case one replaces $\hat{\mu}_{f}$ with the ratio estimate 
> 如果 $p(x)$ 只是比例上已知，即只知道 $p(x) = cp_0(x)$，其中 $p_0(x)$ 已知，而 $c$ 是未知的正规化常数
> 因此我们仅知道 $w(x) = cw_0(x)$，其中 $w_0(x)$ 已知，$c$ 未知
> 此时，我们将 $\hat \mu_f$ 替换为比值估计 $\tilde \mu_f = \sum_j w(x^j)f(x^j)/\sum_j w(x^j)$

$$
\tilde{\mu}_{f}=\frac{\sum_{j=1}^{m}w(x^{(j)})f(x^{(j)})}{\sum_{j=1}^{m}w\bigl(x^{(j)}\bigr)}.
$$ 
It follows from the strong law of large numbers that $\hat{\mu}_{f}\to\mu$ and ${\tilde{\mu}}_{f}\to\mu_{f}$ as $n\to \infty$ almost surely; see Geweke (1989). 
> 根据大数定律，随着 $m\to \infty$，$\hat \mu_f \to \mu$（平均值趋向于期望）
> 同理，随着 $m\to \infty, \tilde \mu_f \to \mu$（如果分子分母同时收敛于各自的期望，则分数也会收敛于期望的比值，显然分子收敛于 $mE_{q(x)}\left[f(x)\frac {p(x)}{q(x)}\right]= m\int f(x)p(x)dx = m\mu$，分母收敛于 $mE_{q(x)}\left[\frac {p(x)}{q(x)}\right]= m\int p(x)dx = m$）

Moreover a central limit theorem yields that $\sqrt{m}({\hat{\mu}}_{f}\,-\,\mu_{f})$ and $\sqrt{m}({\tilde{\mu}}_{f}\,-\,\mu_{f})$ normal with mean zero and respective variances ${\mathbb E}_{q}[(w(X)f(X)-\mu_{f})^{2}]$ and ${\mathbb E}_{q}[w(X)^{2}(f(X)-\mu_{f})^{2}]$ – whenever these quantities are finite. These asymptotic variances can be consistently estimated by reusing the sampled $x^{(j)}$ values as $\begin{array}{r}{\frac{1}{m}\sum_{j}[w(x^{(j)})f(x^{(j)})-\hat{\mu}_{f}]^{2}}\end{array}$ and ${\textstyle\sum_{j}}[w(x^{(j)})^{2}(f(x^{(j)})-$ $\begin{array}{r}{\tilde{\mu}_{f}\big)^{2}]/[\sum_{j}w\big(x^{(j)}\big)]^{2}}\end{array}$ respectively. 
> 根据中心极限定理，当样本数 $m\to \infty$，统计量 $\sqrt m (\hat \mu_f - \mu_f)$ 和 $\sqrt m (\tilde \mu_f - \mu_f)$ 的分布都会收敛到一个均值为0，方差分别为 $\mathbb E_q[(w (X) f (X)- \mu_f)^2]$ (随机变量 $w (X) f (X)$ 的方差) 和 ${\mathbb E}_{q}[w (X)^{2}(f (X)-\mu_{f})^{2}]$
> 这些极限值可以渐进地通过对应的统计量估计

(
关于比率估计 $\tilde \mu_f = \frac {\sum_{j} w (x^j)(f (x^j)}{\sum_j w (x^j)}$，我们需要将分子分母分开来看

分子 $S_m = \sum_j w(x^j) f(x^j)$ 可以看作由 $m$ 个独立同分布的随机变量 $w (X) f (X)$ 的求和，根据中心极限定理，当 $m\to \infty$，$S_m$ 的分布趋近于 $\mathcal N (E [w (X) f (X)], mVar[w (X) f (X)]) = \mathcal N(\mu_S, m\sigma_S^2)$，
其中 $\mu_S = \mu_f,\sigma_S^2 = Var[w (X) f (X)]$

分母 $T_m = \sum_j w (x^j)$ 也可以看作由 $m$ 个独立同分布的随机变量 $w (X)$ 的求和，根据中心极限定理，当 $m\to \infty$，$T_m$ 的分布趋近于 $\mathcal N (E[w (X)], mVar[w (X)]) = \mathcal N (\mu_T, m\sigma_T^2)$
其中 $\mu_T = 1$

同时，在 $m\to \infty$ 时，我们将 $T_m$ 近似看作常数 $m\mu_T$，此时将 $\tilde \mu_f$ 写为 $\tilde \mu_f = \frac {S_m}{m\mu_T} = \sum_j\frac {w(x^j)f(x^j)}{m\mu_T}$，因此 $\tilde \mu_f$ 可以视作 $m$ 个独立同分布的随机变量 $\frac 1 {m\mu_T}w (X) f (X)$ 的求和，根据中心极限定理，当 $m\to \infty$，$\tilde \mu_f$ 的分布趋近于 $\mathcal N (\frac {\mu_S}{m\mu_T}, \frac {m\sigma_S^2}{m^2\mu_T^2})$，其中 $\frac {\mu_S}{m\mu_T} = \mu_f，\frac {\sigma_S^2}{m\mu_T^2} = \frac {\sigma_S^2}{m}$

因此，$\sqrt m (\tilde u_f - \mu_f)$ 的分布在 $m\to \infty$ 时趋近于 $\mathcal N \left(0, \frac {\sigma_S^2}{\mu_T^2}\right)= \mathcal N(0, \sigma_S^2)$

考虑 $\sigma_S^2 = Var[w(X)f(X)] = E_q[(w(X)f(X))^2] - E_q^2[w(X)f(X)]$
因此 $\sigma_S^2 = E_q[(w (X) f (X))^2]- \mu_f^2$
进一步分解 $E_q[(w (X) f (X))^2]$:

$$
\begin{align}
&E_q[w(X)^2f(X)^2]\\
=&E_q[w(X)^2(f(X)-\mu_f+\mu_f)^2]\\
=&E_q[w(X)^2(f(X)-\mu_f)^2 + 2w(X)^2(f(X)-\mu_f)\mu_f + w(X)^2\mu_f^2]\\
=&E_q[w(X)^2(f(X)-\mu_f)^2] + E_q[2w(X)^2(f(X)-\mu_f)\mu_f] + E_q[w(X)^2\mu_f^2]
\end{align}
$$

其中 $E_q[2w (X)^2 (f (X) -\mu_f)\mu_f] = 2\mu_f E_q[w(X)^2 (f(X)- \mu_f)] = 2\mu_fE_q[w(X)^2f(X)]-2\mu_f^2 E_q[w(X)^2]$，因此，$E_q[2w (X)^2 (f (X)-\mu_f)\mu_f] + E_q[w (X)^2\mu_f^2] = 2\mu_f E_q[w (X)^2f (X)]-\mu_f^2 E_q[w (X)^2]$

考虑 $E_q[w (X)^2 (f (X)-\mu_f)]$:
当 $w (X), f (X)$ 相互独立时，二者的联合分布可以分解为二者的乘积，此时可以有 $E_q [w (X) f (X)] = E_q[w (X)]E_q[f (X)] = \mu_f$，因此，由于 $E_q[w (X)] = 1$，实际上我们有 $E_q[f (X)] = \mu_f$

故 $2\mu_f E_q[w (X)^2f (X)] - \mu^2_f E_q[w (X)^2] = 2\mu_f E_q[w (X)^2]E_q[f (X)] - \mu_f^2 E_q[w (X)^2]=\mu_f^2E_q[w(X)^2]$
要成立，需要假设 $E_q[w (X)^2] = 1$，这不一定成立，因此暂未证出
)

> [! 中心极限定理]
> 对于一个独立同分布的 $n$ 个样本组成的样本序列 $X_1,\dots, X_n$，当 $n\to \infty$，这些变量的样本均值 $\bar X_n = \frac 1 n\sum_{i=1}^n X_i$ 的分布将会趋向于正态分布
> 
> 假设每个样本 $X_i$ 都具有相同的期望 $\mu$ 和方差 $\sigma^2$，
> 则 $\bar X_n$ 的期望就为 $E[\frac 1 n \sum_{i=1}^n X_i]=\mu$，
> 方差为 $Var[\frac 1 n \sum_{i=1}^n X_i]= \frac 1 {n^2} Var\left[\sum_{i=1}^n X_i\right]= \frac 1 {n^2}\sum_{i=1}^n Var[X_i] = \frac {\sigma^2} n$
> 
> 中心极限定理表明：当 $n\to \infty$，$\bar X_n$ 的分布将会收敛到高斯分布 $\mathcal N (\mu, \frac {\sigma^2} n)$
> 因此， $\sqrt n (\bar X_n - \mu)$ 的分布将会收敛到高斯分布 $\mathcal N (0, \sigma^2)$

The approximation accuracy oﬀered by IS depends critically on the choice of the trial density $q(x)$ . 
> IS 的近似准确性很大程度依赖于试验密度 $q (x)$ 的选择

Suppose $f(x)=1$ for all $x$ , and consequently $\mu_{f}=1$ , but we still want to estimate this by using IS with a trial density $q(x)$ . In this case the variance of $\hat{\mu}_{f}$ is 

$$
V_{p}(\hat{\mu}_{f})=\mathbb{E}_{q}[(w(X)-1)^{2}]/m.
$$ 
> $\hat \mu_f = \frac 1 m\sum_{i=1}^m w (X_i) f (X_i)$
> $\mathbb V[\hat \mu_f] = \frac 1 {m^2}\mathbb V[\sum_{i=1}^m w (X_i) f (X_i)] = \frac 1{m^2} \sum_{i=1}^m\mathbb V[w (X_i) f (X_i)] = \frac 1 m\mathbb V[w (X_i) f (X_i)]$
> 因为 $f (x) = 1$ for all $x$，故
> $\mathbb V[\hat \mu_f] = \frac 1 m \mathbb [w (X_i)] = \mathbb E_q[(w (X)- 1)^2]/m$
> 可以看到，IS 估计的期望的方差和 $q (x)$ 是有关的

For IS to be accurate (with a limited number $m$ of draws) this variance must be small, which requires $q(x)$ be approximately proportional to $p(x)$ for most $x$ . 
> 如果 IS 需要在有限的 $m$ 次采样下尽量准确，该方差就要尽量小，这要求 $q (x)$ 应该对于大多数 $x$ 都近似与 $p (x)$ 成比例
> 此时，重要性权重 $w (x) = \frac {q (x)}{p (x)}$ 的值就会比较稳定，其方差就小

In a general Monte Carlo problem, where very little is known about the structural properties of the target density $p(x)$ , it could be challenging to identify a $q(x)$ that is easy to sample from and yet provides a good ap- proximation to $p(x)$ . Usually, this problem intensifies with the dimension of $x$ , as the relative volume of $x$ where $p(x)$ is high becomes extremely small. There are, however, special cases where a reasonable choice of $q(x)$ , or a class of such distributions, presents itself. This article provides an overview of these cases and the related IS algorithms. 
> 在通常的 Monte Carlo 问题中，我们对于目标密度 $p(x)$ 的结构化性质知之甚少，因此我们也很难决策出一个易于采样并且提供了对 $p(x)$ 的良好近似的 $q(x)$
> 这个问题随着 $x$ 的维度变高会更加严重，因为在高维空间中使得 $p(x)$ 取较高值的区域的相对体积会变得极其小，这意味着大部分的采样可能会落在 $p (x)$ 较低的区域，从而导致重要性权重的波动增加，进而增加了估计量的方差

In many applications theoretical properties of $p(x)$ are used to determine approximation within a family of $q(x)$ indexed by a low-dimensional vector-valued parameter. A final choice of $q(x)$ from the chosen family is made by numerically optimizing some pre-specified measure of efficiency. Evaluating this measure can itself require a pilot IS or a recursive scheme of IS – earning the overall technique the name Adaptive parametric Importance Sampling . This is discussed in detail in Section 2. 
> 许多应用中，我们会根据 $p(x)$ 的理论性质来在一个低维向量参数索引的 $q (x)$ 族中决定近似
> 从 $q(x)$ 家族中做出的选择是通过数值上优化某个预先指定的效率度量完成的，而评估该度量本身也可能需要一个最初的 IS (pilot IS) 或一个递归的 IS，故这整个计数被称为自适应参数化重要性采样
> 详见 Section 2

When $x$ is high-dimensional and possibly non-Euclidean, a parametric approximation to $p(x)$ is hard to obtain. In such cases one strategy is to break the task of approximating $p(x)$ into a series of low dimensional approximations. In many interesting Monte Carlo problems, $p(x)$ leads to a natural chain-like decomposition of $x$ allowing a sequential construction of $q(x)$ that takes advantage of this decomposition. The resulting IS, called Sequential Importance Sampling (SIS) is discussed in Section 3. In the absence of a natural decomposition, it is still possible to apply the SIS framework by extending the Monte Carlo problem to an augmented space. A specific implementation of this strategy, known as Annealed Importance Sampling is presented in Section 4. 
> 当 $x$ 是高维并且可能是非欧几何空间时，我们较难获得 $p (x)$ 的参数化近似
> 在这类情况下，一个策略是将近似 $p(x)$ 分解为一系列低维的近似，许多 Monte Carlo 问题中，$p (x)$ 可以导向 $x$ 的自然链式分解，进而允许我们利用该分解顺序构建 $q(x)$
> 该技术称为顺序重要性采样 (SIS)，详见 Section 3
> 即便没有自然的分解，我们也可以通过将 Monte Carlo 问题拓展到增广空间中来应用 SIS，该策略的一个特定实现就是 Annealed Importance，详见 Section 4

In Section 5, we review the use of resampling in SIS to adapt dynamically from an initial candidate $q(x)$ to the target $p(x)$ without requiring any numerical optimizations. This adaptability of SIS, which takes full advantage of its parallel computing structure, gives it a competitive edge against Monte Carlo methods that rely solely on Markov Chain sampling (MCS). 
> Section 5 回顾在 SIS 中使用重采样来动态地从初始候选分布 $q(x)$ 适应到目标分布 $p (x)$ ，而无需任何数值优化
> SIS 的这种适应性充分利用的它的并行计算结构，使得其相对于仅依赖于 Markov Chain 采样的 Monte Carlo 方法具有竞争优势

In Section 6, we end with a brief discussion of the current developments in IS research, especially its combination with MCS. 

# 2 Adaptive parametric Importance Sampling 
In Bayesian statistics and econometric applications, $p(x)$ often represents an un-normalized posterior density $p(x)\,=\,c g(x)$ over a Euclidean pa- rameter space, known only up to a multiplicative constant $c\ >\ 0$ . In many cases such a $p(x)$ can be asymptotically well approximated by a multivariate normal distribution with mean given by the posterior mode $\begin{array}{r}{\hat{x}=\operatorname*{argmin}_{x}[-\log p(x)]}\end{array}$ nd variance matrix given by the inverse of the Hessian of $-\log p(x)$ at $\hat x$ (see Section 4 of Ghosh et al. 2006 and the references therein). It is rather cheap to obtain stable and accurate ap- proximation to these quantities through standard optimization routines.
> 在贝叶斯统计中，$p (x)$ 一般表示一个欧式空间中的未规范化的后验密度 $p (x) = cg (x)$，其中我们仅知道 $c > 0$
> 许多情况下，这类 $p (x)$ 可以通过一个多元正态分布来渐进地很好近似，其中正态分布的均值由后验模式给出，即 $\hat x = \arg\min_x[-\log p (x)]$ (使得 $p (x)$ 最小的 $x$)，协方差矩阵由 $-\log p(x)$ 在 $\hat x$ 点出的海森矩阵的逆给出，并且，通过标准的优化例程（例如拟牛顿法、梯度下降法）获得这些量的是比较经济的

Hence the corresponding multivariate normal density serves as a good candidate for $q(x)$ . In practice, a multivariate Student density with similar characteristics may be preferred to the multivariate normal choice (see Geweke 1989 and Evans and Swartz 1995). This is because a multivariate Student density, with its heavy tails, provides a higher assurance of the finiteness ${\mathbb E}_{q}[w(X)^{2}]$ and thus that of the variance of $\tilde{\mu}_{f}$ . 
> 因此，该多元正态密度就是 $q (x)$ 的一个很好的候选
> 实践中，我们更偏好多元 Student 密度，因为多元 Student 密度具有厚尾（即较重的尾部，也就是在尾部的密度较高，更能捕获到远离均值的数据点），可以提供更高的保证使得 $\mathbb E_q[w (X)^2]$ 保持有限，进而使得 $\tilde \mu_f$ 的方差也是有限的（因为厚尾分布可以更好处理极端值，从而减少重要性权重 $w(X)$ 的波动，进而减少估计量的方差）

Oh and Berger (1992, and previously Kloek and van Dijk 1978) extended the above approach by reducing its dependence on the exact asymptotic approximation of $p(x)$ . They took $q(x)$ to be given by the density $q_{\lambda}(x)=t_{\nu}(x\mid\lambda)$ – the multivariate Student density with a fixed degrees of freedom $\nu$ and a location-scale $\lambda=(\mu,\Sigma)$ chosen to minimize 

$$
c v^{2}(\lambda)=\int\frac{p(x)^{2}}{t_{\nu}(x\mid\lambda)}d x-1.\tag{4}
$$

> Oh and Berger(1992)通过减少 $p(x)$ 对于精确渐进近似的依赖，进一步拓展该方法（渐进近似可能在小样本情况下不准确）
> 我们选择 $q (x)$ 为一个具有固定自由度 $\nu$ 的多元学生密度 $q_{\lambda}(x) = t_{\nu}(x\mid \lambda)$，其中位置尺度参数 $\lambda = (\mu,\Sigma)$ 的选择通过最小化 $cv^2 (\lambda)$ 实现

Note that $c v^{2}(\lambda)$ equals $\mathbb{E}_{q}[(w(X)-1)^{2}]$ and hence, as noted earlier, a small magnitude of this quantity ensures high accuracy in estimating $\mu_{f}$ for all $f(x)$ that are relatively ﬂat with respect to $p(x)$ . 
> 注意 $cv^2 (\lambda)$ 等于 $\mathbb E_q[(w (X)- 1)^2]$，也就是 $w (x)$ 在 $q (x)$ 下的方差，因此，最小化 $cv^2 (\lambda)$ 等价于最小化 $\mathbb E_q[(w (X) - 1)^2]$，以保证在估计所有相对于 $p (x)$ 较为平坦的 $f (x)$ 函数的 $\mu_f$ 时有较高的准确性
>
> $cv^2 (\lambda) = \mathbb E_q[(w (X)- 1)^2]$ 的推导如下：

$$
\begin{align}
&\mathbb E_q[(w(X)-1)^2]\\
=&\int q(x)(w(x)-1)^2dx\\
=&\int q(x)(w(x)^2 - 2w(x) + 1)dx\\
=&\int q(x)w(x)^2 - 2q(x)w(x) + q(x)dx\\
=&\int \frac {p(x)^2}{q(x)}dx -2\int p(x)dx + \int q(x)dx\\
=&\int \frac {p(x)^2}{q(x)}dx - 1
\end{align}
$$

Many authors take the related quantity ${\mathbb E}_{q}[w(X)^{2}]$ as a rule-of-thumb measure of efficiency of IS based on a trial density $q(x)$ ; see Liu (2001) for a discussion. 
> 许多作者将统计量 $\mathbb E_q[w (X)^2]$（也就是 $\int \frac {p (x)^2}{q (x)}dx$）作为基于试验密度 $q (x)$ 所进行的重要性采样的效率的衡量
> 当 $\mathbb E_q[w (X)^2]$ 较小时，重要性权重 $w (X)$ 方差较小，因此重要性采样的方差较小

In Oh and Berger (1993), the candidate set for $q(x)$ was further extended to a finite mixture of the form $\begin{array}{r}{q_{\lambda}(x)=\sum_{i=1}^{k}\pi_{i}t_{\nu}(x\mid\mu_{i},\Sigma_{i})}\end{array}$ with $\lambda=\{(\pi_{i},\mu_{i},\Sigma_{i})\}$ , to cover the case of multimodal posterior densities. Since the quantity in (4) cannot be computed in closed form or minimized analytically, Oh and Berger (1993) suggested the following approximate optimization. Start with an initial guess $\lambda^{\mathrm{init}}$ for $\lambda$ and sample $x^{(1)},\cdot\cdot\cdot,x^{(m)}$ from $q_{\lambda^{\mathrm{init}}}(x)$ . Compute $\begin{array}{r}{\lambda^{\mathrm{opt}}=\operatorname*{argmin}_{\lambda}\widehat{c v}^{2}(\lambda;\lambda^{\mathrm{init}})}\end{array}$ where 

$$
\begin{array}{r}{\widehat{c v}^{2}(\lambda;\lambda^{\prime})=\frac{\frac{1}{m}\sum_{j=1}^{m}\frac{[g(x^{(j)})/q_{\lambda^{\prime}}(x^{(j)})]^{2}}{[q_{\lambda}(x^{(j)}|)/q_{\lambda^{\prime}}(x^{(j)})]}}{[\frac{1}{m}\sum_{j=1}^{m}g\left(x^{(j)}\right)/q_{\lambda^{\prime}}\left(x^{(j)}\right)]^{2}}-1}\end{array}
$$

is an IS approximation to $c v^{2}(\lambda)$ based on the sample drawn from $q_{\lambda^{\mathrm{init}}}(x)$ . 
> Oh and Berger (1993) 进一步将 $q (x)$ 的候选集拓展到形式 $q_{\lambda}(x) = \sum_{i=1}^k \pi_i t_{\nu}(x\mid \mu_i, \Sigma_i)$ 的有限次混合，其中 $\lambda = \{(\pi_i, \mu_i,\Sigma_i)\}$，以覆盖多模态后验密度的情况
> 但该形式的 $q_{\lambda}(x)$ 在 (4) 中没有解析形式的最小化解，故需要采用近似优化求解，从初始猜测 $\lambda^{\text{init}}$ 开始，从 $q_{\lambda^{\text{init}}}(x)$ 中采样 $m$ 个样本，然后计算此时的 $\lambda^{\text{opt}} = \arg\min_\lambda \widehat {cv}^2(\lambda; \lambda^{\text{init}})$，其中

$$
\begin{array}{r}{\widehat{c v}^{2}(\lambda;\lambda^{\prime})=\frac{\frac{1}{m}\sum_{j=1}^{m}\frac{[g(x^{(j)})/q_{\lambda^{\prime}}(x^{(j)})]^{2}}{[q_{\lambda}(x^{(j)}|)/q_{\lambda^{\prime}}(x^{(j)})]}}{[\frac{1}{m}\sum_{j=1}^{m}g\left(x^{(j)}\right)/q_{\lambda^{\prime}}\left(x^{(j)}\right)]^{2}}-1}\end{array}
$$

> $\widehat {cv}^2(\lambda; \lambda^{\text{init}})$ 是一个在初始猜测值 $\lambda^{\text{init}}$ 下的基于从 $q_{\lambda^{\text{init}}}(x)$ 的采样数据的估计量，用于近似 $cv^2 (\lambda)$

A variation of the above idea was proposed in Richard and Zhang (2007). For a family of candidates $q_{\lambda}(x)$ , they suggested choosing $\lambda=\lambda^{\mathrm{opt}}$ where $(\alpha^{\mathrm{{opt}}},\lambda^{\mathrm{{opt}}})$ minimizes the pseudo divergence 

$$
d(\alpha,\lambda)=\int(\log g(x)-\alpha-\log q_{\lambda}(x))^{2}p(x)d x\tag{5}
$$

over $(\alpha,\lambda)$ . Note that if $p(x)\,=\,q_{\lambda_{0}}(x)$ for some $\lambda_{0}$ , then the above is minimized at $\left(-\log c,\lambda_{0}\right)$ . 
> Richard and Zhang (2007) 提出通过最小化伪散度来确定一族 $q_{\lambda}(x)$ 中最优的 $q_{\lambda^{\text{opt}}}(x)$
> 如果存在某个 $\lambda_0$ 使得 $p (x) = q_{\lambda_0}(x)$，则 $\lambda^{\text{opt}}$ 就是 $\lambda_0$

As in Oh and Berger (1993), (5) too has to be solved numerically. Richard and Zhang (2007) proposed the following iterative scheme for this. Start with an initial estimate of ${\lambda}={\lambda}^{(0)}$ . For $t=1,2,\cdot\cdot\cdot$ compute 

$$
(\alpha^{(t+1)},\lambda^{(t+1)})=\mathrm{argmin}_{\alpha,\lambda}\sum_{j=1}^{m}(\log g(x_{t}^{(j)})-\alpha-\log q_{\lambda}(x_{t}^{(j)}))^{2}\frac{g(x_{t}^{(j)})}{q_{\lambda^{(t)}}(x_{t}^{(j)})}\tag{6}
$$

where $x_{t}^{(j)}$ are drawn from $q_{\lambda^{(t)}}(x)$ . 
> (5) 同样不存在解析解，因此需要数值上的迭代优化，从初始估计 $\lambda = \lambda^{(0)}$ 开始，对于预设定的 $t$ 轮数，按照 (6) 进行迭代计算

The attractive feature of this program is that the minimization in (6) is a generalized, weighted least squares minimization problem for which global solutions are often easy to find. In particular, if $q_{\lambda}$ is chosen from an exponential family, then the above reduces to a simple least squares problem. 
> 该方法的优势在于 (6) 中的最小化是一个广义的加权最小二乘问题，因此容易找到全局最优解
> 特别地，如果 $q_{\lambda}$ 是属于指数族，则 (6) 中的最小化就是简单的最小二乘问题

A diﬀerent adaptive parametric IS was proposed by Owen and Zhou (2000) who combined IS with the method of control variates (see Hammersley and Handscomb 1964). They worked with a single choice of $q(x)$ but adapted their Monte Carlo method by optimizing over a parametric choice of the control variates. 
> Owen and Zhou (2000) 提出一个不同的自适应参数化 IS，他们将 IS 和控制变元的方法结合，他们的想法是固定 $q (x)$，转而通过对于控制变元的参数化选择的优化来调节 Monte Carlo 方法

In particular, they took $\begin{array}{r}{q(x)=\sum_{i=1}^{k}\alpha_{i}q_{i}(x)}\end{array}$ , with fixed densities $q_i$ ’s and a fixed probability vector $\alpha\,=\,\left(\alpha_{1},\cdot\cdot\cdot\,,\alpha_{k}\right)$ , but proposed to estimate $\mu_f$ by 

$$
\hat{\mu}_{f,\beta}=\frac{1}{m}\sum_{j=1}^{m}\frac{f(x^{(j)})p(x^{(j)})-\sum_{i=1}^{k}\beta_{i}q_{i}(x^{(j)})}{q(x^{(j)})}+\sum_{i=1}^{k}\beta_{i}
$$

with $\beta=\left(\beta_{1},\cdot\cdot\cdot,\beta_{k}\right)$ minimizing the asymptotic variance of $\hat{\mu}_{f,\beta}$ given by 

$$
\sigma^{2}(\beta)=\int{\bigg(\frac{f(x)p(x)-\sum_{i}{\beta_{i}q_{i}(x)}}{q(x)}-\mu_{f}+\sum_{i}{\beta_{i}\bigg)}^{2}q(x)d x}.
$$

A consistent estimate of an optimal $\beta$ can be found by minimizing 

$$
{\hat{\sigma}}^{2}(\beta,\beta_{0})=\sum_{j=1}^{m}\left({\frac{f(x^{(j)})p(x^{(j)})}{q(x^{(j)})}}-\sum_{i}\beta_{i}{\frac{q_{i}(x^{(j)})}{q(x)}}-\beta_{0}\right)^{2}
$$ 
through least squares methods. 
> 该方法令 $q (x) = \sum_{i=1}^k \alpha_i q_i (x)$，其中 $q_i (x)$ 和概率向量 $\alpha = (\alpha_1,\dots, \alpha_k)$ 都固定
> 该方法对于 $\mu_f$ 的估计 $\hat{\mu}_{f,\beta}$ 则和控制变元 $\beta = (\beta_1,\dots, \beta_k)$ 有关
> 控制变元的选择通过最小化 $\hat \mu_{f, \beta}$ 与 $\mu_f$ 之间的渐进方差得到
> 最小化过程同样是从初始猜测 $\beta_0$ 开始迭代式求解，每次迭代都是求解一个最小二乘问题

Note that $\hat{\mu}_{f,\beta}$ requires exact knowledge of $p(x)$ . When $p(x)=c g(x)$ with $c$ unknown, one can modify the estimate to become 

$$
\tilde{\mu}_{f,\beta}=\frac{\sum_{j=1}^{m}\frac{f(x^{(j)})g(x^{(j)})-\sum_{i=1}^{k}\beta_{i}q_{i}(x^{(j)})}{q(x^{(j)})}+\sum_{i=1}^{k}\beta_{i}}{\sum_{j=1}^{m}\frac{g(x^{(j)})-\sum_{i=1}^{k}\beta_{i}q_{i}(x^{(j)})}{q(x^{(j)})}+\sum_{i=1}^{k}\beta_{i}}.
$$

It is also possible to use two diﬀerent sets of $\beta$ in the numerator and the denominator above. This approach is particularly attractive when more than one $(p(x),f(x))$ are of interest, and at least one of the chosen $q_{i}(x)$ is expected to lead to an efficient IS for each pair (see Theorem 2 in Owen and Zhou 2000). 
> $\hat \mu_{f,\beta}$ 的估计需要确切地知道 $p (x)$，如果仅知道 $p (x) = c g (x)$，其中 $c$ 未知，可以用 $\tilde \mu_{f,\beta}$ 替代 $\hat \mu_{f,\beta}$

# 3 Sequential Importance Sampling 
In many Monte Carlo problems with a high-dimensional $x$ , the target den- sity $p (x)$ induces a chain-like decomposition of $x=\left (x_{1},\cdot\cdot\cdot, x_{d}\right)$ , paving he way for generating $x$ sequentially as $x_{[1: t]}\;=\;\left (x_{1},\cdot\cdot\cdot, x_{t}\right)$ , $1\,\leq\, t\,\leq d$  . Such decompositions occur naturally in state-space models (finance, signal-tracking), evolutionary models (molecular physics and biology, genetics) and others (see Section 3 of Liu 2001). 
> 在许多关于高维 $x$ 的 Monte Carlo 问题中，目标密度 $p (x)$ 可以针对 $x = (x_1,\dots, x_d)$ 进行链式分解，因此可以将 $x$ 的生成视作序列 $x_{[1: t]} = (x_1,\dots x_t), 1\le t \le d$ 的生成
> 这类分解在状态空间模型、进化模型中很常见

Writing $p (x)$ as 

$$
p({x})=p(x_{1})\prod_{t=2}^{d}p(x_{t}\mid x_{[1:t-1]})
$$

it is easy to see that an efficient IS can be built by using a $q (x)$ of the form 

$$
q({x})=q_{1}(x_{1})\prod_{t=2}^{d}q_{t}({x}_{t}\mid{x}_{[1:t-1]}),
$$

where $q_{t}\bigl (x_{t}\ \big|\ x_{[1: t-1]}\bigr)$ mimics  $p\big (x_{t}\mid x_{[1: t-1]}\big)$ well. 
> 将 $p (x)$ 链式分解后，我们可以同样使用链式分解的形式构建 $q (x)$
> 其中的每个条件概率分布 $q_t (x_t \mid x_{[1: t-1]})$ 都近似条件概率分布 $p (x_t\mid x_{[1: t-1]})$

For such a scheme, the importance weight $w (x)=p (x)/q (x)$ , too, can be computed sequentially as $w (x)=w_{d}$ where 

$$
w_{t}=w_{t-1}\frac{p(x_{t}\mid x_{[1:t-1]})}{q_{t}(x_{t}\mid x_{[1:t-1]})}\tag{8}
$$

and $w_{0}\,=\, 1$ . 
> 同样，重要性权重 $w (x) = p (x) / q (x)$ 也可以写为链式分解的形式
> 重要性权重 $w_0$ 初始值设定为 $1$，之后，随着 $t$ 增大，$w_{t-1}$ 通过式 (8) 计算得到 $w_t$

The sequence $w_{t}$ can be used to check on the ﬂy the importance of the sample being generated, and one can possibly discard a sample half-way if $w_{t}$ starts getting very small. We shall make this idea more precise in the next section. 
> $w_t$ 序列可以用于检查样本生成时的重要性
> 如果 $w_t$ 开始变得很小，我们可以考虑丢弃样本

To facilitate the construction of $q_{t}\big (x_{t}\ |\ x_{[1: t-1]}\big)$ , Liu (2001) presented the above sequential importance sampling (SIS) scheme in a slightly more general form. Liu introduced a sequence of auxiliary distributions $p_{t}\big (x_{[1: t]}\big)$ , $1\leq t\leq d$ , with $p_{d}(x_{[1: d]})=p (x)$ and rewrote the updating equation (8) as 

$$
w_{t}=w_{t-1}\frac{p_{t}(x_{[1:t]})}{p_{t-1}(x_{[1:t-1]})q_{t}(x_{t}\mid x_{[1:t-1]})}.
$$

The auxiliary densities $p_{t}\big (x_{[1: t]}\big)$ could be chosen to approximate the marginal densities $p (x_{[1: t]})$ with $p_{t}\bigl (x_{t}\mid x_{[1: t-1]}\bigr)$ serving as a guideline to constructing $q_{t}\bigl (x_{t}\mid x_{[1: t-1]}\bigr)$ . 
> Liu (2001) 引入一个辅助分布序列 $p_t (x_{[1: t]}), 1\le t\le d$，将 (8) 的更新规则重写
> 辅助分布 $p_t (x_{[1:t]})$ 应是对分布 $p (x_{[1:t]})$ 的近似，因此 $p_t (x_t\mid x_{[1: t-1]})$ 也作为 $p (x_t\mid x_{[1: t-1]})$ 的近似，用于构建 $q_t (x_t\mid x_{[1: t-1]})$

This general definition accommodates the possibility 
that there could be various ways of obtaining a good approximation. We shall illustrate this with two examples of historical and practical relevance. 
> 我们阐述两个利用近似构建 $q_t (x_t\mid x_{[1: t-1]})$

Consider the task of simulating a length- $d$ self-avoiding-walk (SAW, see Liu 2001) on the 2-dimensional integer lattice starting from $(0,0)$ . Here $x\,=\,\bigl (x_{1},\cdot\cdot\cdot\,, x_{d}\bigr)$ denotes a chain of $d$ integer coordinates $x_{t}\,=\,\left (i_{t}, j_{t}\right)$ , $1\leq t\leq d$ such that 

$$
(i_{t},j_{t})\in\{(i_{t-1}-1,j_{t-1}),(i_{t-1}+1,j_{t-1}),(i_{t-1},j_{t-1}-1),(i_{t-1},j_{t-1}+1)\}
$$ 

with $x_{t}\ \neq\ (0,0)$ a $x_{t}~\neq~x_{s}$ for any $1\;\leq\; s\;\neq\; t\;\leq\; d$ . 
> 考虑长度为 $d$ 的自回避游走，从 $(0,0)$ 开始，每一次在两个方向中任选其一向前或向后走一步，限制是不会走已经走过的点

Suppose the target distribution $p (x)$ is the uniform distribution over all length- $d$ SAWs. A reasonable choice of an auxiliary $p_{t}\big (x_{1[: t]}\big)$ in this case is the uniform distribution on $x_{[1: t]}$ . Bear in mind that $p_{t}(x_{[1: t]})\neq p (x_{[1: t]})$ . 
> 假设目标分布 $p (x)$ 是在所有长度为 $d$ 的自回避游走路径上的均匀分布
> 此时辅助分布 $p_t (x_{1[:t]})$ 可以选为 $x_{[1:t]}$ 上的均匀分布

By taking $q_{t}\bigl (x_{t}\ \bigm|\ x_{[1: t-1]}\bigr)\,=\, p_{t}\bigl (x_{t}\ \bigm|\ x_{[1: t-1]}\bigr)$ it is easy to see that given the first $t-1$ coordinates , one samples $x_t$ uniformly from the unoccupied $x_{[1: t-1]}$ neighbors of $x_{t-1}$ . 
Alternatively one can take $p_{t}\big (x_{[1: t]}\big)$ as the marginal distribution of ${x}_{[1: t]}$ under a uniform distribution on $x_{[1:t+1]}$ . In this case $q_{t}=p_{t}\big (x_{t}\ |\ x_{[1: t-1]}\big)$ leads to a two-step look ahead sampling of $x_{t}$ given $x_{[1: t-1]}$ where a neighbor of $x_{t-1}$ is selected with probability proportional to the number of unoccupied neighbors it currently has. 

In statistical missing data problems, for example, the observables $z_{t}$ , $1\leq t\leq n$ , are partitioned into $z_{t}=(x_{t}, y_{t})$ with only the $y_{t}$ components being actually observed. For maximum likelihood or Bayesian inference on such problems $p (x)$ often represents the conditional distribution $f (x\mid y)$ derived from a joint model $f (x, y)$ on the complete data $z$ . 
When $f (x, y)$ specifies independence or a simple chain structure across $z_{t}$ ’s, a useful choice of $p_{t}\bigl (x_{[1: t]}\bigr)$ is given by $f (x_{[1: t]}\ \ |\ \ y_{[1: t]})$ with $q_{t}(x_{t}\ \ |\ \ x_{[1: t-1]})\ =$ $p_{t}\bigl (x_{t}\ |\ x_{[1: t-1]}\bigr)\,=\, f\bigl (x_{t}\ |\ z_{[1: t-1]}, y_{t}\bigr)$ . The corresponding updates of $w_t$ can be written more compactly as $w_{t}=w_{t-1}f (y_{t}\ |\ z_{1: t-1})$ . Because the method fills in the missing components sequentially it is called sequential imputation . 

# 4 Annealed Importance Sampling 
The appealing feature of SIS is that it achieves an approximation to $p (x)$ through a series of simpler approximations of $p\big (x_{t}~\big|~x_{[1: t-1]}\big)$ by $q_{t}(x_{t}\textrm{|}$ $x_{[1: t-1]}$ ). 
> 序列重要性采样的特点是通过对一系列 $p (x_t\mid x_{[1: t-1]})$ 的近似达到了对 $p (x)$ 的近似

In Annealed Importance Sampling (AIS), Neal (2001) introduced a similar construction to handle cases where $x$ does not admit a natural chain-like decomposition. Like SIS, a sequence of distributions $p_{t}(x)$ , $0\leq$ $t\leq d$ is used, with $p_{d}(x)=p (x)$ . But each of these densities is defined on the same space on which $p (x)$ is defined. Here the sequence $\{p_{t}(x)\}_{t}$ forms a bridge of successive approximations from $p_{0}(x)$ to $p (x)\,=\, p_{d}(x)$ . The initial density $p_{0}(x)$ is taken to be diﬀuse and easy to sample from. It is required that at every step, sampling from $p_{t-1}(x)$ leads to an efficient IS for the immediate target $p_{t}(x)$ . This can be achieved for some $p (x)$ when one defines $p_{t}(x)\,=\, p_{0}(x)^{1-b_{t}}p (x)^{b_{t}}$ with $0\,=\, b_{0}\,<\, b_{1}\,<\,\cdots\,<\, b_{d}\,=\, 1$ . This gradual morphing of a diﬀuse $p_{0}(x)$ to a possibly well concentrated $p (x)$ is reminiscent of the cooling schedules applied in Simulated Annealing (SANN) for function optimization. In fact, Neal (2001) introduced AIS as an IS-augmented version of SANN fit for Monte Carlo approximations. 
> Neal (2001) 提出了退火重要性采样来处理 $x$ 不适合自然链式分解时的情况
> AIS 中同样使用了由近似分布 $p_t (x),0\le t\le d$ 构成的序列，并且令 $p_d (x) = p (x)$
> 和 SIS 不同的是序列中的每一个分布 $p_t (x)$ 都定义在和 $p (x)$ 相同的空间中，序列 $\{p_t (x)\}$ 实际上形成了从 $p_0 (x)$ 到 $p_d (x) = p (x)$ 对 $p (x)$ 的连续的近似
> 最初的密度 $p_0 (x)$ 设定为弥散的，并且易于从中采样
> 要求在每一步中，从 $p_{t-1}(x)$ 的采样都是对直接目标 $p_t (x)$ 的高效重要性采样
> 对于某些 $p (x)$，将 $p_t (x)$ 定义为 $p_t (x) = p_0 (x)^{1-b_t}p (x)^{b_t},0=b_0<b_1<\cdots<b_d = 1$ 可以达到这一目标
> 这种从弥散的 $p_0 (x)$ 到可能高度集中的 $p (x)$ 的逐渐变化让人联想到模拟退火 (SANN) 中应用于函数优化的冷却调度，实际上，Neal (2001) 最初将 AIS 介绍为 SANN 的一个重要性采样增强版本，用于 Monte Carlo 近似

In AIS, a random draw of $x$ is made by sequentially drawing $x_{(t)}$ , $0\leq t\leq d$ , and equating as follows. One starts by drawing $x =x_{(0)}$ from $p_{0}(x)$ and sets $w_{0}=1$ . Then, for $t=1,\cdots, d$ 

1. Sample $x_{(t)}$ from  $g_t (\cdot \mid x_{(t-1)})$
2. Set $w_t = w_{t-1}\frac {p_t (x_{(t-1)})}{p_{t-1}(x_{(t-1)})}$

where $g_{t}(x^{\prime}\mid x)$ is a transition kernel that leaves $p_{t}(x)$ invariant: 

$$
g_{t}(x^{\prime}\mid x)\geq0,\;\int g_{t}(x^{\prime}\mid x)d x^{\prime}=1,\;\;\int p_{t}(x)g_{t}(x^{\prime}\mid x)d x=p_{t}(x^{\prime}).\tag{9}
$$

> AIS 中，对于 $x$ 的随机采样通过对 $x_{(t)},0\le t\le d$ 的序列采样完成
> 首先从 $p_0 (x)$ 中采样出 $x = x_{(0)}$，然后令 $w_0 = 1$，之后，对于每一个时间步：
> 1. 从条件概率分布/转移核 $g_t (\cdot \mid x_{(t-1)})$ 中采样 $x_{(t)}$
> 2. 根据 $w_{t-1}$ 计算 $w_t$（更新权重）
> 其中转移核是 $p_t (x)$ 的不变量

Taking $\tilde{g}_{t}(x^{\prime}\mid x)=g_{t}(x\mid x^{\prime}) p_{t}(x^{\prime})/p_{t}(x)$ – the reversal of $g_{t}$ – it can be shown that $w_{d}$ gives the proper importance weight $p^{*}(x^{*})/q^{*}(x^{*})$ for the target density 

$$
p^*(x^*)=p_{d}(x_{(d)})\times{\tilde{g}}_{d}(x_{(d-1)}\ |\ x_{(d)})\times\cdot\cdot\cdot\times{\tilde{g}}_{1}(x_{(0)}\ |\ x_{(1)})
$$

on the augmented variable ${x}^{*}=\left (x_{(0)}, x_{(1)},\cdot\cdot\cdot, x_{(d)}\right)$ with 

$$
q^{*}(x^{*})=p_{0}(x_{(0)})\times g_{1}(x_{(1)}\mid x_{(0)})\times\cdot\cdot\times g_{d}(x_{(d)}\mid x_{(d-1)})
$$

as determined by the AIS sampling scheme. 
> 可以证明 $w_{d}$ 给出了针对增广变量 ${x}^{*}=\left (x_{(0)}, x_{(1)},\cdot\cdot\cdot, x_{(d)}\right)$ 的正确的重要性权重 $p^{*}(x^{*})/q^{*}(x^{*})$ 
> ($w_d$ 反映的就是路径 ${x}^{*}=\left (x_{(0)}, x_{(1)},\cdot\cdot\cdot, x_{(d)}\right)$ 的重要性权重，
> 目标分布 $p^*(x^*)$ 是定义在增广变量 $x^*$ 上的分布，它可以通过使用逆转转移核 $\tilde g_t$ 从 $p_d (x)$ 回溯计算得到，
> 采样分布 $q^*(x^*)$ 是实际采样过程中使用的分布，它通过从初始状态开始经过转移核 $g_t$ 转移得到)

The marginal distribution of $x_{(d)}$ determined by $p^{*}(x^{*})$ is simply $p_{d}\bigl (x_{(d)}\bigr)=p\bigl (x_{(d)}\bigr)$ . 

Note that $g_{t}(x^{\prime}\mid x)$ is left completely unspecified beyond the requisite invariance property (9). One can tap into the vast literature of Markov Chain Monte Carlo to construct a suitable transition kernel $g_{t}(x^{\prime}\mid x)$ . A simple choice is a few Metopolis or Gibbs updates of $x$ with $p_{t}(x)$ as the target density. AIS also oﬀers complete ﬂexibility in the choice of the number of steps $d$ and the intermediate densities $p_{d}(x)$ . This choice can have a major impact on the performance of the algorithm; see Lyman and Zuckerman (2007) and Godsill and Clapp (2001). 
> 转移核 $g_t (x'\mid x)$ 的构造参考合适的 Markov Chain Monte Carlo 方法
> 一个简单的选择是以 $p_t (x)$ 作为目标密度，进行几次 Metopolis 或 Gibbs 更新 (Metopolis-Hastings 算法和 Gibbs 采样都是常用的 MCMC 技术)
> AIS 允许我们自由选择中间步数 $d$ 和中间密度 $p_d (x)$ ，这些选择会很大程度影响算法表现

# 5 SIS with Resampling 
In all of the SIS implementations detailed above, the $m$ samples $x^{(1)},\cdot\cdot\cdot, x^{(m)}$ are drawn in a non-interactive parallel manner. In such schemes, most of the corresponding weights $w^{(j)}=w\big (x^{(j)}\big)$ are very small and contribute only a little to the computation of $\hat{\mu}_{f}$ or $\tilde{\mu}_{f}$ . This becomes particularly problematic when $x$ is high dimensional. 
> 之前讲述的 SIS 中，$m$ 个样本 $x^{(1)},\dots, x^{(m)}$ 是以非交互的并行形式采样的，此时大多数对应的权重 $w^{(j)} = w (x^{(j)})$ 都过于小，对 $\hat \mu_f$ 或 $\tilde \mu_f$ 的计算贡献很小，尤其当 $x$ 是高维度时

For example, in SIS, it can be shown that the weight sequence $w_{t}$ forms a martingale and hence its coefficient of variation $c v_{t}^{2}=\mathbb{V}\mathrm{ar}(w_{t})/\mathbb{E}(w_{t})^{2}$ explodes to infinity as $t$ increases (Kong et al. 1994). Consequently, a very small proportion of the final draws $x^{(j)}$ carry most of the weight – making the SIS estimate rather inefficient. 
> (Kong et al. 1994) 证明 SIS 中权重序列 $w_t$ 形成了一个鞅，因此其变异系数 $cv_t^2 = \mathbb {Var}(w_t)/\mathbb E (w_t)^2$ 随着 $t$ 增大而变得无限大
> 这导致最终抽取出的样本中仅有很少一部分携带大部分权重
> (马丁格尔 Martingale 是一个随机过程，其未来的期望值等于当前值，即序列 $X_t$ 满足 $\mathbb E[X_{t+1}\mid X_1,\dots, X_t] = X_t$；我们称满足该性质的 $X_t$ 为鞅，
> SIS 中，权重序列是一个鞅，即在已知目前所有权重的情况下，未来权重的期望值等于当前权重；
> 变异系数定义为权重的方差与期望值平方的比值，它反映了 $w_t$ 序列的离散程度，如果 $cv_t^2$ 很大，说明其方差相对于均值很大，则权重分布很不均匀；
> Kong 等人指出，随着 $t$ 增加，$cv_t^2$ 趋向于无穷大，即权重分布会变得极度不均匀，会有很少一部分的样本的权重非常大；
> 此时 SIS 估计就主要依赖于这几个少数样本，使得估计结果非常不稳定)

A simple fix of this, known as the enrichment method, was proposed by Wall and Erpenbeck (1959). Their idea was to grow all the $x_{[1: t]}^{(j)}$ , $j\leq1\leq m$ – each called a stream – simultaneously, and at intermediate check point $1\leq t_{1}\leq\cdot\cdot\leq t_{k}\leq d$ , replace the streams with small current weights $w_{t}^{(j)}$ with replicates of the streams with large current weights. A simple re-weighting of the resulting streams makes the whole process a valid IS scheme. Grassberger (1997) further suggested making the check points dynamic. In his Pruned-enriched Rosenbluth method (PERM) each current stream is either removed or replicated (or grown according to the original SIS scheme) based on its current weight being smaller than a lower cut-oﬀ $c_{t}$ or larger than an upper cut-oﬀ $C_{t}$ (or otherwise). Note that the total number of streams may not remain the same. 
> 一个简单的办法是富集方法，思路是同时增长所有的 $x_{[1: t]}^{(j)}, j\le 1\le m$，其中每一个都称为一个流，然后在每一个中间检查点 $1\le t_1\le\dots\le t_k \le d$ 用具有较大当前权重的流替换具有较小当前权重的流，对结果流进行简单的重新加权，使得整个过程成为一个有效的 IS 方案
> Grassberger (1997) 提出动态化检查点，在他的剪枝富集 Rosenbluth 方法中，对于每个当前流，如果其当前权重小于下限 $c_t$ ，则被移除，如果大于上限 $C_t$，则被复制，否则按原样保留，注意总流的数量可能不会保持不变

A related idea of replicating the “good” streams was explored by Gordon et al. (1993) for the special IS method known as the bootstrap filter (also particle filter ) for non-linear/non-Gaussian state space models. The setting here is similar to the missing data problem discussed in the pre- vious section with the following important diﬀerences: 
> Grodon (1993) 为非线性/高斯状态空间模型提出了近似的复制 “好的” 流的思路，称为引导滤波（或粒子滤波），它和数据缺失模型类似，但有以下两点不同

(A) the model on $f (y, x)$ is assumed to have the following Markov structure 
> $f (y, x)$ 上的模型被假设具有以下 Markov 结构

$$
f(y,x)=\prod_{t=1}^{n}[f_{\mathrm{state}}(x_{t}\mid x_{t-1})f_{\mathrm{obs}}(y_{t}\mid x_{t})]
$$

and 
(B) $f (x_{t}\mid x_{t-1}, y_{t})$ is not assumed to be easy to sample from, making the choice $q_{t}\bigl (x_{t}\ |\ x_{[1: t-1]}\bigr)\,=\, f\bigl (x_{t}\ |\ z_{[1: t-1]}, y_{t}\bigr)$ infeasible. The choice of $q_{t}\big (x_{t}\ |\ x_{t-1}\big)=f_{\mathrm{state}}\big (x_{t}\ |\ x_{t-1}\big)$ is assumed feasible, but it often leads to an extremely large $c v_{t}^{2}$ . 
Gordon et al. (1993) improved upon this through an extra resampling stage as follows. At stage $t$ , draw $x_{t}^{(\ast j)}$ , $1\leq j\leq m$ from $f_{\mathrm{state}}(x_{t}\mid x_{t-1}^{(j)})$ and weight each draw by $w_{t}^{(j)}\;\propto\; f_{\mathrm{obs}}\big (y_{t}\;\mid\; x_{t}^{(\ast j)}\big)$ . Resample from $\{x_{t}^{(*1)},\cdot\cdot\cdot\ ,x_{t}^{(*m)}\}$ with weights $w_{t}^{(j)}$ to produce the next stage draws $\{x_{t}^{(1)},\cdot\cdot\cdot, x_{t}^{(m)}\}$. At completion, each stream gets weighed equally $(w_{d}^{(j)}\,=\, 1/m)$ for evaluating the estimate $\tilde{\mu}_{f}$ . This estimate is guaranteed to converge to $\mu_{f}$ as the number of streams (also known as particles in state-space literature) tends to infinity. 
> $f (x_t\mid x_{t-1}, y)$ 并未被假定易于采样，因此选择 $q_t (x_t \mid x_{[1: t-1]}) = f (x_t\mid z_{[1: t-1]}, y_t)$ 是不可行的，选择 $q_t (x_t \mid x_{t-1}) = f_{\text{state}}(x_t\mid x_{t-1})$ 被认为是可行的，但常常会导致极其大的方差 $cv_T^2$
> 为此 Gordon (1993) 设计了一个额外的重采样阶段，在阶段 $t$，从 $f_{\mathrm{state}}(x_{t}\mid x_{t-1}^{(j)})$ 中抽取 $x_{t}^{(\ast j)}$，其中 $1\leq j\leq m$，然后根据权重 $w_{t}^{(j)}\;\propto\; f_{\mathrm{obs}}\big (y_{t}\;\mid\; x_{t}^{(\ast j)}\big)$ 对每次抽取进行加权。接着使用这些权重对 $\{x_{t}^{(*1)},\cdot\cdot\cdot\ ,x_{t}^{(*m)}\}$ 进行重采样，以产生下一轮的抽取 $\{x_{t}^{(1)},\cdot\cdot\cdot, x_{t}^{(m)}\}$。
> 在最终阶段，每个序列（或粒子）的权重相等 $(w_{d}^{(j)}\,=\, 1/m)$，用于评估估计值 $\tilde{\mu}_{f}$。随着序列数（即粒子数）趋于无穷大，这个估计值保证会收敛到 $\mu_{f}$

Liu and Chen (1998) introduced SIS with resampling (SISR) by combining across-stream resampling with a dynamic choice of check points. In SISR, a decision of resampling at state $t$ is made by checking the current coefficient of variation $c v_{t}^{2}$ of the weights $\{w_{t}^{(j)},\cdot\cdot\cdot, w_{t}^{(j)}\}$ against a specified cut-oﬀ $c_{t}$ (typically growing with $t$ at a polynomial rate). If $c v_{t}^{2}$  exceeds $c_{t}$ , then $x_{[1: t]}^{(j)}$ ’s are updated by resampling from $\{x_{[1: t]}^{(1)},\cdot\cdot\cdot, x_{[1: t]}^{(m)}\}$ with probability proportional to $w_{t}^{(j)}$ . Each resampled stream is then as- signed a weight $\textstyle\sum_{j}w_{t}^{(j)}/m$ . 
> Liu anc Chen (1998) 结合了跨流重采样和动态检查点选择，提出了 SISR
> SISR 中，状态 $t$ 下的重采样决策是通过比较当前权重 $\{w_T^{(j)}, \dots, w_t^{(j)}\}$ 的变异系数 $cv_t^2$ 和特定的阈值 $c_t$（一般以多项式速率随着 $t$ 增大）得到的
> 如果 $cv_t^2$ 超过 $c_t$，则从 $\{x_{[1: t]}^{(1)},\cdot\cdot\cdot, x_{[1: t]}^{(m)}\}$ 中按比例为 $w_{t}^{(j)}$ 的概率重新采样来更新 $x_{[1: t]}^{(j)}$，每条重采样的流随后被分配一个权重 $\textstyle\sum_{j}w_{t}^{(j)}/m$

A check on the coefficient of variation guards against unwanted pruning when all streams have similar weights. Chen et al. (2005) further modified SISR to allow resampling along a diﬀerent time measurement than the original stage index $t$ . 
> 对变异系数的检查可以防止当所有流的权重相似时不必要的修剪。Chen et al.(2005) 进一步修改了SISR，允许沿不同于原始阶段索引 $t$ 的时间度量进行重采样

The modified algo- rithm, called SIS with stopping time resampling (SISSTR) determines check points by applying a stopping rule on each stream separately. Once all streams have reached their first stop, they are pooled together and a resampling is done if the coefficient of variation of the current weights exceeds a pre-specified cut-oﬀ. The streams then grow in parallel until the next stop is reached by each, and so on. An interesting application of this was presented in Chen et al. (2005) to the coalescent model of Kingman (1982) where SISSTR remarkably improved a naive SIS due to Griffiths and Tavar´e (1994). 
> 改进后的算法称为停止时间顺序重采样（SISSTR），通过在每条流上单独应用停止规则来确定检查点。一旦所有流都到达它们的第一个停止点，就将它们汇集在一起，并在当前权重的变异系数超过预先设定的阈值时执行重采样。然后这些流并行增长直到每个流达到下一个停止点，并如此循环。
> 陈等人（2005）展示了一个有趣的应用，在金曼（Kingman，1982）的聚合模型中，SISSTR显著提高了Griffiths和Tavaré（1994）提出的朴素SIS的效果。

# 6 SIS and Markov Chain Sampling 
The introduction of resampling and the use of transition kernels have largely expanded the scope of SIS algorithms. The latter development has brought these algorithms closer to Monte Calro methods that use Markov chain sampling to generate a sequence of dependent draws from a target density by exploring it through appropriate transition kernels. 
> 重采样的引入以及转移核的应用极大地扩展了序列重要性抽样算法的应用范围
> 转移核的发展使得这些算法更接近于使用马尔可夫链抽样（Markov Chain Sampling, MCS）从目标分布中生成一系列依赖样本的蒙特卡罗方法

A consensus is rapidly emerging that much can be gained by combining these two approaches together. The AIS algorithm, in which Markov chain sampling (MCS) facilitates SIS on an artificially augmented space, gave the first formal exploration of such a combination. The Population Monte Carlo (PMC) algorithm of Capp´e et al. (2004) – inspired by a resampling enriched AIS due to Hukushima and Iba (2001) – went in the reverse di- rection. In PMC, a resampling based SIS facilitates MCS by adaptively choosing transition kernels that lead to most efficient exploration of the target distribution. Del Moral et al. (2006) proposed an extremely ﬂexible theoretical framework for an eﬀective symbiosis of IS and MCS in popula- tion based simulation methods for sequential Monte Carlo problems; see also Jasra et al. (2007) and Fernhead (2008). In these methods, a pool of draws is generated sequentially in an interactive, parallel manner. MCS guides local exploration by each stream in the pool and importance sam- pling enables the pool to decide how to efficiently redistribute its streams in the vast space it is trying to explore. Such conﬂuence of MCS and IS will be an important direction for future Monte Carlo research. 
> 人们越来越认同将这结合 SIS 和 MCS 可以带来许多益处
> 在人工增强的空间中利用马尔可夫链抽样促进 SIS 的适应性重要抽样（AIS, Adaptive Importance Sampling）算法是首次正式探索这种结合的方法
> 由 Cappé等人（2004）提出的群体蒙特卡罗（PMC, Population Monte Carlo）算法受到了 Hukushima 和 Iba（2001）提出的基于重采样的 AIS 的启发，其采样方向是相反的，在 PMC 中，基于重采样的 SIS 通过自适应地选择导致对目标分布最有效探索的转移核来促进 MCS
> Del Moral 等人（2006）提出了一个极其灵活的理论框架，用于在基于群体的模拟方法中有效地结合 IS 和 MCS 解决序贯蒙特卡罗问题；参见 Jasra 等人（2007）和 Fernhead（2008）。在这些方法中，一组样本是通过互动和平行的方式依次生成的。MCS 指导池中的每个流进行局部探索，而重要性抽样使整个池能够决定如何在其试图探索的广阔空间内有效地重新分配其流。MCS 与 IS 的这种融合将是未来蒙特卡罗研究的一个重要方向。

# Supplementary Reading
## A simple tutorial on Sampling Importance and Monte Carlo with Python codes
[original post](https://medium.com/@amir_masoud/a-simple-tutorial-on-sampling-importance-and-monte-carlo-with-python-codes-8ce809b91465)
### Introduction
In this post, I’m going to explain the importance sampling. Importance sampling is an approximation method instead of a sampling method. It shows up in machine learning topics as a trick. It is derived from a little mathematic transformation and is able to formulate the problem in another way.
> IS 实际是一种近似方法，而不是一种采样方法

The content of this post mainly originated from this fantastic YouTube [tutorial](https://www.youtube.com/watch?v=C3p2wI4RAi8&feature=emb_imp_woyt). You can find the Jupyter notebook for this post [here](https://gist.github.com/iamirmasoud/2b83de47ee93b9a27a7694495182164d). You can find the original post on my [website](http://www.sefidian.com/2022/08/01/sampling-importance-and-monte-carlo-simply-explained/).

### What is Monte Carlo?
It’s difficult to understand importance sampling in a vacuum. It will make a lot more sense if we motivate it from the perspective of **Monte Carlo** methods. Let’s dive right into these techniques are concerned with one extremely important test that shows up a ton in machine learning: **Calculating Expectations**. Without going into too much detail, expectations are very frequently the answers that many of our ML algorithms are seeking. They show up when fitting models, doing probability queries, summarizing model performance, and training a reinforcement learning agent.
> IS 要考虑的问题是：近似期望

Now mathematically that means we are interested in calculating this:

![](https://miro.medium.com/v2/resize:fit:376/0*TwSS6TnHf3esQNp-)

where _x_ is a continuous random vector. Boldface means it’s a vector and _p(x)_ is the probability density of _x._ _f(x)_ is just some scalar function of _x_ we happen to be interested in. 
Now you should think of this as the probability-weighted average of _f(x)_ over the entire space where _x_ lives to communicate that. It’s often written like this:

![](https://miro.medium.com/v2/resize:fit:672/0*pTF2DVyMXvR3ZdpE)

As an aside if _x_ were discrete then this integral would turn into a sum. None of the concepts really change for the discrete case. We’ll continue with the continuous case. 

Now the problem is sometimes, in fact, a lot of the time, _this integral is impossible to calculate exactly_. It’s because the dimension of _x_ is high so the space that lives within is exponentially huge and we have no hope of adding everything up within it. This is where Monte Carlo methods come in. _The idea is to merely approximate the expectation with an average._
> 基本思想就是用均值近似期望，计算均值的样本通过采样得到
> 即 $\mathbb E_p[f (x) ] \approx \underbrace{\frac 1 N \sum_{i=1}^N f (x_i)}_s,\quad x_i\sim p (x)$

This average says we need to collect _N_ samples of _x_ from the distribution _p_, plug those into _f_, and take their average. It turns out that as _N_ gets large this thing approaches our answer. For the sake of brevity, let’s call this sum _s_.

![](https://miro.medium.com/v2/resize:fit:984/0*yphpSA3Sq3msXKAe)

Let’s play with a one-dimensional example. Let’s say _p(x)_ and _f(x)_ are as below:

![](https://miro.medium.com/v2/resize:fit:1084/0*JRCQ0HXk9_QVDrjr)

In this case, things are simple enough such that we can calculate the answer exactly. To do that, we look at the product function and calculate the area under the purple curve.

![](https://miro.medium.com/v2/resize:fit:1112/0*5TN-zXDlAcTaemm5)

This is the exact value (0.5406). Please use your imagination and pretend we can’t actually calculate the area this way because in the general case it can be impossible. So instead, let’s approximate this area using the Monte Carlo method. First, we sample _x_’s from _p(x)_ and then plug those into _f(x)_ the average of these is an approximation to the integral.

![](https://miro.medium.com/v2/resize:fit:1174/0*TAVGW8sNz-8BE2Gp)

To emphasize, the core of Monte Carlo is to say the average of lines mapped on the y-axis (0.58) is an approximation to the area under the purple curve in the previous image (0.54). I’m not proving that but trust me it’s true and very important frankly. It’s more important than importance sampling!

One thing to point out is that _the value of average s is random_. If we were to resample everything we’d get a new value for _s_ each time. For example, with a different set of values for samples, we have another value for _s_ as follows:

![](https://miro.medium.com/v2/resize:fit:1172/0*vQNhIVDa4SC5Cdyb.png)

This means that **our sample average has its own distribution**. Let’s check that out. Let’s say the true expectation we’d like to estimate is as below:

![](https://miro.medium.com/v2/resize:fit:1082/0*VPLJJOKmZPwaW7sI.png)

Let’s say we repeat our estimate many times giving us many _s_’s. Now we’re in a better position to view the distribution of s.

![](https://miro.medium.com/v2/resize:fit:1116/0*CxwNc7VoRwMfYl6X.png)

Now to avoid confusion this distribution is just useful for a theoretical understanding of applications. We only get one sample of _s_. Keep that in mind as I will talk about this thing. The first thing to notice here is _the distribution is centered on the true expectation_. This is a critical property. When we have this property we say that **“our estimate is unbiased”**.
> 注意，根据采样计算得到的均值 $s$ 实质上是一个随机变量，$s$ 有自己的分布
> 容易知道，均值 $s$ 是对期望 $\mathbb E_p[f(x)]$ 的无偏估计，也就是说，$s$ 的期望就是 $\mathbb E_p[f (x)]$

Also looking at this distribution, we can see that the **variance** of _s_ something which tells you about the width of this distribution matters a lot.

![](https://miro.medium.com/v2/resize:fit:1110/0*SCE6mQax9bGQFzdP.png)

It gives us a sense of how off a single estimate is likely to be from the truth. Now it turns out the variance of _s_ is the variance of f(x) scaled down by _N_.
> $s$ 的方差 $\mathbb V_p[s] = \frac 1 {N} \mathbb V_p[f (x)]$

One other thing to notice. This distribution is a normal distribution. Where did that come from? Well, that’s the **Central Limit Theorem**. It’s a magical and super famous result and in this case, it says it doesn’t matter what _p(x)_ or _f(x)_ is, as _N_ _gets large_ this will get closer and closer to a normal distribution. In fact, we can summarize this by writing:
> $s$ 是 $N$ 个独立同分布的样本计算得到的均值，因此当 $N$ 趋近于无穷，根据中心极限定理，$s$ 服从的分布也趋近于高斯分布

![](https://miro.medium.com/v2/resize:fit:1086/0*0NtXzLTrcMYchw6J.png)

The Central Limit Theorem says:

> _The distribution of_ s _approaches the normal as_ N _gets large. The mean of that normal is the expectation we’re trying to calculate and its variance is the variance of f(x) scaled down by_ N.

### What is the importance sampling?
We start by introducing a _new distribution_ _q(x)_. As we’ll see this is something we get to choose and again we are interested in the same expectation as before. We had earlier noted this still involves _p(x)_. Now here’s a trick. The trick is to multiply the term within the integral by _q(x)/q(x)_ which is just equal to 1.

![](https://miro.medium.com/v2/resize:fit:1114/0*sT5VGX1lhe13WWjE.png)

We can do this without damaging anything as long as _q(x)_ is greater than 0 whenever _p(x)_ \* _f(x)_ is non-zero.

That gives us the following new integral.

![](https://miro.medium.com/v2/resize:fit:854/0*FfOC_c6EA0rd2Wqr.png)

Looking at how I’ve bracketed things we can see that it’s the **_probability-weighted average of a new function_** where the probability is given by _q(x)_ instead of _p_(x). So just like we did earlier, we can write that like this:
> 引入新分布 $q (x)$，将 $\mathbb E_p[f (x)]$ 重写为 $\mathbb E_q[\frac {p (x)}{q (x)}f (x)]$，此时我们将一个分布下的期望转化为了另一个分布下的期望，目标函数 $f (x)$ 额外乘上了权重系数 $\frac {p (x)}{q (x)}$

![](https://miro.medium.com/v2/resize:fit:854/0*_dyUb5A_gZZhsmKG.png)

This says:

> _The_ p(x) _probability weighted average of_ f(x) _is equal to the_ q(x) _probability weighted average of_ f(x) _times the ratio of p(x)/q(x) densities._

Now if we look at this and we recall the Monte Carlo, we can estimate this with samples from _q(x)_.

![](https://miro.medium.com/v2/resize:fit:1186/0*iBx0taBrVmpjpXYk.png)

What I mean by that is the original expectation we’re interested in is approximately equal to this new average:

![](https://miro.medium.com/v2/resize:fit:434/0*VbpV6psRExxbmvY_.png)

where the **_x_i_’s are sampled from _q_(x).** Let’s call this new average _r_.
> 此时，由于计算期望的目标分布变化，我们采样的目标分布也从 $p (x)$ 变化为 $q (x)$
> 即计算目标变为 $\underbrace{\frac 1 N \sum_{i=1}^N \frac {p (x_i)}{q (x_i)}f (x_i)}_r, \quad x_i\sim q(x)$

What’s the advantage of using this?

First, it’s unbiased just like in the previous case:
> $r$ 同样是 $\mathbb E_p[f (x)]$ 的无偏估计，也就是 $\mathbb E_q[r] = \mathbb E_p[f (x)]$

![](https://miro.medium.com/v2/resize:fit:370/0*GIqr_sELmHiXnVyq.png)

Second, it has a new possibly improved variance. That is the variance of _r_ which gives us a sense of how off from the truth a single sample of _r_ is likely to be is the variance of _f(x)_ times the density ratio where samples are generated according to _q(x)_ scaled down by _N_. The hope is **we can choose _q(x)_ such that this variance is less than the variance we dealt with earlier**. Now, to do that, it turns out _q(x)_ should say _x_’s are likely wherever the absolute value of _p(x)_ \* _f(x)_ is high.
> 此时考虑 $r$ 的方差 $\mathbb V_q[r] = \frac 1 {N} \mathbb V_q[\frac {p (x)}{q (x)}f (x)]$
> 我们希望通过适当的选择 $q (x)$，可以使得 $\mathbb V_q[r] < \mathbb V_p [s]$，
> 即 $\mathbb V_q[\frac {p (x)}{q (x)}f (x)] < \mathbb V_p [f (x)]$
> 为此，我们希望 $p (x) f (x)$ 的绝对值高时，$q (x)$ 也高

(

$$
\begin{align}
\mathbb V_q\left[\frac {p(x)}{q(x)}f(x)\right]&=\mathbb E_q\left[\frac {p(x)^2}{q(x)^2}f(x)^2\right] - \mathbb E_q\left[\frac {p(x)}{q(x)}f(x)\right]^2\\
&=\mathbb E_q\left[\frac {p(x)^2f(x)^2}{q(x)^2}\right] - \mathbb E_q\left[\frac {p(x)}{q(x)}f(x)\right]^2\\
\end{align}
$$

将 $\mathbb E_q[\frac {p (x)}{q (x)}f (x)] = \mathbb E_p[f (x)]$ 视作常数，我们需要考虑的是让 $\mathbb E_q[\frac {p (x)^2 f (x)^2}{q (x)^2}]$ 尽量小，以使得 $\mathbb V_q[r]$ 尽量小
因此 $q (x)$ 最好和 $p (x) f (x)$ 成比例，避免 $p (x) f (x)$ 较大时 $q (x)$ 较小，这会导致 $\mathbb E_q[\frac {p (x)^2 f (x)^2}{q (x)^2}]$ 在快速增大
)

![](https://miro.medium.com/v2/resize:fit:804/0*OWb2GS1wNnI-H_x6.png)

That’s a key result that I’m not proving but it kind of makes sense when you recognize that we are trying to estimate the area under the _p(x)_ times _f(x)_ curve got it.

It’s about time we do an example to help. Here is a little summary of the core ideas we’re demonstrating.

![](https://miro.medium.com/v2/resize:fit:1400/0*HHWcqtMAfxQLYFlX.png)

Let’s do an example where importance sampling will reduce the variance quite a bit in particular. Let’s say _p(x)_ and _f(x)_ are as follows:

![](https://miro.medium.com/v2/resize:fit:1140/0*sHS-5sB7c-Sd05S8.png)

As a point of reference, let’s calculate several estimates using Monte Carlo without importance sampling:

![](https://miro.medium.com/v2/resize:fit:1162/0*hnmf9jZP_WprBb2H.png)

![](https://miro.medium.com/v2/resize:fit:1174/0*vvgt7ooToswAeSOW.png)

![](https://miro.medium.com/v2/resize:fit:1184/0*f8WgKVXuFhJc45oL.png)

Notice how much our estimate is bouncing around. That is high variance. This is because _f(x) is large only in rare events under p(x)._ So a small set of samples have an outsized impact on the average.
> 示例中，$f (x)$ 仅在 $p (x)$ 较小时较大，即常常仅有少量的样本对整体均值有较大影响，因此通过采样计算的 $\frac 1 N \sum_{i=1}^N f (x_i), \quad x_i \sim p (x)$ 会有较大的方差

Now importance sampling can help us here because we can _make it sample the more important regions more frequently_. In particular, I’ll make _q(x)_ like the below:
> 重要性采样的思路是帮助我们更频繁地采样更重要的区域

![](https://miro.medium.com/v2/resize:fit:1126/0*We4DnxAXThED0YXf.png)

But now if we’d like to use _q_ samples we need to adjust the _f(x)_ function. So let’s do that by multiplying it by the density ratio which gives us this:
> 我们引入 $q (x)$，并调整 $f (x)$ 为 $\frac {p (x)}{q (x)} f (x)$

![](https://miro.medium.com/v2/resize:fit:1136/0*dOnLmxujeXuuASDi.png)

To declutter things a bit I’ll drop _p(x)_.

![](https://miro.medium.com/v2/resize:fit:1084/0*bSrtOr0B-8ueiB8O.png)

now after this adjustment, we can proceed just like we did with the plain Monte Carlo. That is we sample from _q(x)_ pass those into the density ratio adjusted _f(x)_ and then calculate their average. This is our new estimate (0.083):

![](https://miro.medium.com/v2/resize:fit:1144/0*LWtNIYFH6d5iWI5H.png)

The whole point is if we repeat this process we’ll get something that bounces around less (less variance) than if we sampled the previous way. That’s the whole idea the variance is reduced and with a single estimate, we expect to be less wrong.

### How to choose the new distribution q(x) in Importance Sampling?
How are we supposed to choose _q(x)_ in practice if its best answer depends on _p(x)_ and _f(x)_ which are supposedly difficult to work with?

That is indeed tricky and the answers aren’t very satisfying. The hope is that we’ll know a few things about _p(x)_ and _f(x)_ which will guide us. Maybe we know where _f(x)_ is high or maybe we can pick a _q(x) that_ merely approximates _p(x)_. it’s very much an art. Also fair warning, it’s very easy to do a terrible job at selecting _q(x)_, especially in high dimensions. The symptom of that will be _the density ratio will vary wildly over the samples_ and the majority of them will be very small. This means your average will be effectively determined by a small number of samples making it high variance not good.
> 均值有少量的样本决定就会导致不稳定

### Importance Sampling with Python code
Let’s recap what we’ve learned so far.

Consider a scenario you are trying to calculate an expectation of function _f(x)_, where _x ~ p(x)_, is subjected to some distribution. We have the following estimation of expectation:

![](https://miro.medium.com/v2/resize:fit:1392/0*G1G1TwyY3icGSeHK.png)

The Monte Carlo sampling method is to simply sample _x_ from the distribution _p(x)_ and take the average of all samples to get an estimation of the expectation. If _p(x)_ is very hard to sample from, we can estimate the expectation based on some known and easily sampled distribution _q(x)_. It comes from a simple transformation of the formula:

![](https://miro.medium.com/v2/resize:fit:1400/0*F44vtUzxfiHHbskN.png)

where _x_ is sampled from distribution _q(x)_ and _q(x)_ should not be 0. In this way, estimating the expectation is able to sample from another distribution _q(x)_, and _p(x)/q(x)_ is called _sampling ratio_ or _sampling weight_, which acts as a correction weight to offset the probability sampling from a different distribution.
> $p (x)/q (x)$ 称为采样比率/采样权重，acts as a correction weight to offset the probability sampling from a different distribution

Another thing we need to talk about is the variance of estimation:

![](https://miro.medium.com/v2/resize:fit:1088/0*Vywcie3GJyoyc1Z8.png)

where in this case, `X` is _f(x)p(x)/q(x)_ , so if _p(x)/q(x)_ is large, this will result in large variance, which we definitely hope to avoid. On the other hand, it is also possible to select proper _q(x)_ that results in an even smaller variance. Let’s get into a Python example.

#### Python Example
First, let’s define function _f(x)_ and sample distribution:

```
def f_x(x):  
    return 1/(1 + np.exp(-x))
```

The curve of _f(x)_ looks like:

![](https://miro.medium.com/v2/resize:fit:1400/0*UxaVXiVB2WMfEmHu.png)

Now let’s define the distribution of _p(x)_ and _q(x)_ :

```
def distribution(mu=0, sigma=1):  
    # return probability given a value  
    distribution = stats.norm(mu, sigma)  
    return distribution  
      
# pre-setting  
n = 1000mu_target = 3.5  
sigma_target = 1  
mu_appro = 3  
sigma_appro = 1p_x = distribution(mu_target, sigma_target)  
q_x = distribution(mu_appro, sigma_appro)
```

For simplicity reasons, here both _p(x) and q(x)_ are normal distributions, you can try to define some _p(x)_ that is very hard to sample from. In our first demonstration, let’s set two distributions close to each other with similar means (3 and 3.5) and the same sigma 1:

![](https://miro.medium.com/v2/resize:fit:1400/0*u46jBqLpyf7xB06L.png)

Now we are able to compute the true value sampled from distribution _p(x)_:

```
s = 0  
for i in range(n):  
    # draw a sample  
    x_i = np.random.normal(mu_target, sigma_target)  
    s += f_x(x_i)  
print("simulate value", s/n)
```

where we get an estimation of 0.954. Now let’s sample from _q(x)_ and see how it performs:

```
value_list = []  
for i in range(n):  
    # sample from different distribution  
    x_i = np.random.normal(mu_appro, sigma_appro)  
    value = f_x(x_i)*(p_x.pdf(x_i) / q_x.pdf(x_i))  
      
    value_list.append(value)
```

Notice that here _x_i_ is sampled from the approximate distribution _q(x)_, and we get an estimation of 0.949 and a variance of 0.304. See that we are able to get the estimate by sampling from a different distribution!

#### Comparison
The distribution _q(x)_ might be too similar to _p(x)_ that you probably doubt the ability of importance sampling, now let’s try another distribution:

```
# pre-setting  
n = 5000mu_target = 3.5  
sigma_target = 1  
mu_appro = 1  
sigma_appro = 1p_x = distribution(mu_target, sigma_target)  
q_x = distribution(mu_appro, sigma_appro)
```

with histogram:

![](https://miro.medium.com/v2/resize:fit:1400/0*FdKICy8hh7l8ksi2.png)

Here we set `n` to 5000, when the distribution is dissimilar, in general, we need more samples to approximate the value. This time we get an estimation value of 0.995, but a variance of 83.36.

The reason comes from _p(x)/q(x)_, as two distributions are too different from each other could result in a huge difference in this value, thus increasing the variance. The rule of thumb is to define _q(x)_ where _p(x)|f(x)|_ is large.

Here is the full code:

```python
import numpy as np  
import scipy.stats as stats  
import matplotlib.pyplot as plt  
import seaborn as sns  
def f_x(x):  
    return 1/(1 + np.exp(-x))  
def distribution(mu=0, sigma=1):  
    # return probability given a value  
    distribution = stats.norm(mu, sigma)  
    return distribution  
if __name__ == "__main__":  
    # pre-setting  
    n = 1000    mu_target = 3.5  
    sigma_target = 1  
    mu_appro = 3  
    sigma_appro = 1    
    p_x = distribution(mu_target, sigma_target)  
    q_x = distribution(mu_appro, sigma_appro)    
    plt.figure(figsize=[10, 4]) 
    sns.distplot([np.random.normal(mu_target, sigma_target) for _ in range(3000)], label="distribution $p(x)$")  
    sns.distplot([np.random.normal(mu_appro, sigma_appro) for _ in range(3000)], label="distribution $q(x)$")      
    plt.title("Distributions", size=16)  
    plt.legend()    # value  
    s = 0  
    for i in range(n):  
        # draw a sample  
        x_i = np.random.normal(mu_target, sigma_target)  
        s += f_x(x_i)  
    print("simulate value", s / n)    # calculate value sampling from a different distribution    value_list = []  
    for i in range(n):  
        # sample from different distribution  
        x_i = np.random.normal(mu_appro, sigma_appro)  
        value = f_x(x_i) * (p_x.pdf(x_i) / q_x.pdf(x_i))         
        value_list.append(value)    
        print("average {} variance {}".format(np.mean(value_list), np.var(value_list)))    # pre-setting different q(x)  
    n = 5000    
    mu_target = 3.5  
    sigma_target = 1  
    mu_appro = 1  
    sigma_appro = 1    
    p_x = distribution(mu_target, sigma_target)  
    q_x = distribution(mu_appro, sigma_appro)    
    plt.figure(figsize=[10, 4])    
    sns.distplot([np.random.normal(mu_target, sigma_target) for _ in range(3000)], label="distribution $p(x)$")  
    sns.distplot([np.random.normal(mu_appro, sigma_appro) for _ in range(3000)], label="distribution $q(x)$")    plt.title("Distributions", size=16)  
    plt.legend()    # calculate value sampling from a different distribution    value_list = []  
    # need larger steps  
    for i in range(n):  
        # sample from different distribution  
        x_i = np.random.normal(mu_appro, sigma_appro)  
        value = f_x(x_i) * (p_x.pdf(x_i) / q_x.pdf(x_i))
        value_list.append(value)    
        print("average {} variance {}".format(np.mean(value_list), np.var(value_list)))
```

### Wrap up
Importance sampling is likely to be useful when:

1. _p(x)_ is difficult or impossible to sample from.
2. We need to be able to evaluate _p(x)_. Meaning we can plug in an _x_ and get a value. In fact, that’s a little more than we need we actually only need. The ability to compute an unnormalized density but we’d have to tweak our procedure a bit. See sources if you’re curious.
3. _q(x)_ needs to be easy to evaluate and sample from since our estimate will ideally be made of many samples from it.
4. Lastly, and the hard part, is that you need to be able to choose _q(x)_ to be high where the absolute value of _p(x)_ times _f(x_) is high which is not necessarily an easy task.

One example where importance sampling is used in ML is policy-based reinforcement learning. Consider the case when you want to update your policy. You want to estimate the value functions of the new policy, but calculating the total rewards of taking an action can be costly because it requires considering all possible outcomes until the end of the time horizon after that action. However, if the new policy is relatively close to the old policy, you can calculate the total rewards based on the old policy instead and reweight them according to the new policy. The rewards from the old policy make up the proposal distribution.
> 考虑基于策略的 RL，我们要计算新策略下的 reward 期望时，需要采用 Monte Carlo 近似，如果新的策略近似于旧的策略，可以采用旧的策略作为 proposal distribution

Here is the summary of when Importance Sampling is used:

- _p(x)_ is difficult to sample from.
- We can evaluate _p(x)_.
- _q(x)_ is easy to evaluate and sample from.
- We can choose _q(x)_ to be high where $|p(x)f (x)|$ is high.