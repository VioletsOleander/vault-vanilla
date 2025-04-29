# Part 1 Supervised Learning
## CH1 Linear Regression
## CH2 Classification and logistic regression
## CH3 Generalized linear models
## CH4 Generative learning algorithms
## CH5 Kernel methods
## CH6 Support vector machines
# Part 2 Deep Learning
## CH7 Deep learning
# Part 3 Generalization and Regularization
## CH8 Generalizatoin
## CH9 Regularization and model selection
# Part 4 Unsupervised Learning
## CH 10 Clustering and the k-means algorithm
## CH 11 EM algorithms
### 11.1 EM for mixture of Guassians
给定训练集$\{x^{(1)},\dots, x^{(n)}\}$，在无监督设定下，我们没有标签数据

我们希望通过指定一个联合分布$p(x^{(i)},z^{(i)}) = p(x^{(i)} | z^{(i)})p(z^{(i)})$来建模数据，
其中$z^{(i)}\sim \text{Multinomial}(\phi)$($\phi_j = p(z^{(i)} = j), \phi_j \ge 0, \sum_{j=1}^k \phi_j = 1$)，
且$x^{(i)} | z^{(i)} = j\sim \mathcal N(u_j, \Sigma_j)$

我们令$k$是$z^{(i)}$可以取的值的数量，因此我们的模型假设了每个$x^{(i)}$都通过首先随机从$\{1,\dots,k\}$中选取$z^{(i)}$，然后从取决于$z^{(i)}$的$k$的高斯分布中的其中之一中采样出来，该模型成为混合高斯模型(mixture of Gaussians)

注意$z^{(i)}$是隐随机变量(latent random variables)，说明它们是隐藏的/未观测到的

我们的模型涉及到的参数有$\phi, \mu,\Sigma$，要估计这些参数，我们写出数据的似然
$$\begin{align}
\mathscr l(\phi,\mu,\Sigma) &= \sum_{i=1}^n\log p(x^{(i)};\phi,u,\Sigma)\\
&=\sum_{i=1}^n\log \sum_{z^{(i)}=1}^k p(x^{(i)}|z^{(i)};u,\Sigma) p(z^{(i)};\phi)
\end{align}$$
如果尝试将此式的导数设为零以求解，会发现无法求出闭式解

随机变量$z^{(i)}$用于表明每个$x^{(i)}$是从$k$个高斯分布中的哪一个中采样的，如果我们能确定$z^{(i)}$，则问题的求解就会变得容易，具体地说，我们就可以将问题直接写为
$$\begin{align}
\mathscr l(\phi,\mu,\Sigma)
&=\sum_{i=1}^n\log p(x^{(i)}|z^{(i)};u,\Sigma)+ \log p(z^{(i)};\phi)
\end{align}$$
最大化该式的参数就是
$$\begin{align}
\phi_j &= \frac 1 n\sum_{i=1}^n1\{z^{(i)}=j\}\\
\mu_j &=  \frac {\sum_{i=1}^n 1\{z^{(i)} = j\}x^{(i)}}{\sum_{i=1}^n 1\{z^{(i)} = j\}}\\
\Sigma_j &=\frac {\sum_{i=1}^n 1\{z^{(i)} = j\}(x^{(i)}-u_j)(x^{(i)}-u_j)^T}{\sum_{i=1}^n 1\{z^{(i)} = j\}}
\end{align}$$
该形式和高斯辨别分析中的参数估计几乎完全一致，在此处$z^{(i)}$起了扮演类别标签的作用，差异仅在于此时$z^{(i)}$属于多项式分布而非伯努利分布，且每个高斯分布的$\Sigma_j$都不同

EM算法是由两步主要步骤构成的算法，针对我们目前的问题，
在E-step中，算法尝试”猜测”$z^{(i)}$的值，在M-step中，我们假定上一步的猜测是正确的，然后根据似然函数最大化的规则，进行参数更新，即：
(E-step) 对于每个$i,j$，令
$$w^{(i)}_j = p(z^{(i)} = j | x^{(i)};\phi,\mu,\Sigma)$$
(M-step)更新参数
$$\begin{align}
\phi_j &= \frac 1 n\sum_{i=1}^nw_j^{(i)}\\
\mu_j &=  \frac {\sum_{i=1}^n w_j^{(i)}x^{(i)}}{\sum_{i=1}^n w_j^{(i)}}\\
\Sigma_j &=\frac {\sum_{i=1}^n w_j^{(i)}(x^{(i)}-u_j)(x^{(i)}-u_j)^T}{\sum_{i=1}^n w_j^{(i)}}
\end{align}$$

在E-step中，我们当前参数下，给定$x^{(i)}$，计算$z^{(i)}$的后验概率，即：
$$p(z^{(i)} = j|x^{(i)};\phi,\mu,\Sigma)=\frac {p(x^{(i)}|z^{(i)}=j;u,\Sigma)p(z^{(i)}=j;\phi)}{\sum_{l=1}^k p(x^{(i)}|z^{(i)}=l;u,\Sigma)p(z^{(i)}=l;\phi)}$$
其中，$p(x^{(i)} | z^{(i)} = j; u,\Sigma)$通过评估均值为$u_j$，协方差矩阵为$\Sigma_j$在点$x^{(i)}$处的概率密度得到，$p(z^{(i)}=j;\phi)$即$\phi_j$，E-step中计算的$w_{j}^{(i)}$表示我们对$z^{(i)}$值的“软 soft”猜测

可以比较在$z^{(i)}$已知时的更新策略和M-step中的更新策略，我们可以发现二者近乎是一致的，除了将指示函数$1\{z^{(i)} = j\}$换成了$w_{j}^{(i)}$

EM算法其实也和K-means算法类似，K-means中，我们对每个样本赋予了“硬 hard”的簇标签$c(i)$，而EM中我们则是使用“软 soft”标签$w_{j}^{(i)}$
和K-means相似，EM算法也容易仅收敛到局部最优，因此取多个不同的起始点是比较合理的

EM算法可以自然地解释为重复地去猜测未知的$z^{(i)}$
### 11.2 Jensen's inequality
令$f$为定义域是实数集的函数，则$f$是凸函数当且仅当$f''(x)\ge 0\quad\text{for all } x\in \mathbb R$若$f$接受向量作为输入，则$f$是凸函数当且仅当它的海森矩阵$H$是半正定的
($H\ge 0$)
若$f$对于所有$x$都有$f''(x) > 0$，则称$f$是严格凸的，向量情况下就是海森矩阵是正定矩阵($H > 0$)

Jensen不等式如下所述
**Theroem** $f$是凸函数，$X$是随机变量，则
$$E[f(X)] \ge f(E[X])$$
若$f$是严格凸函数，则$E[f(X)] = f(E[X])$当且仅当$X = E[X]$的概率是$1$，即$X$是常数
当$f$是凹函数时，不等式方向相反
### 11.3 General EM algorithms
给定训练集$\{x^{(1)},\dots, x^{(n)}\}$，内含$n$个独立的样本
我们有隐变量模型$p(x,z;\theta)$，其中$z$就是隐变量(为了简单，假设$z$只能取有限个值)，则$x$的概率密度函数可以通过对求边际密度函数得到：
$$p(x;\theta) = \sum_{z}p(x,z;\theta)\tag{11.1}$$
我们想通过极大化数据的对数似然来找到参数$\theta$，即：
$$\mathscr l(\theta)=\sum_{i=1}^n\log p(x^{(i)};\theta)\tag{11.2}$$
将目标函数以联合分布$p(x,z;\theta)$的形式重写得到：
$$\begin{align}
\mathscr l(\theta) &= \sum_{i=1}^n\log p(x^{(i)};\theta)\tag{11.3}\\
&=\sum_{i=1}^n\log\sum_{z^{(i)}}p(x^{(i)},z^{(i)};\theta)\tag{11.4}
\end{align}$$
对于该优化问题，我们难以找到显式的对参数的最大似然估计，因为它是一个明显的非凸优化问题
此处的$z^{(i)}$都是隐变量，往往在能观察到隐变量的情况下，就可以很容易地进行极大似然估计

EM算法给出了在该设定下进行极大似然估计的一种方法，EM算法的策略是反复地构造$\mathscr l$的下界(E-step)，然后优化该下界(M-step)

为了简化问题，我们先考虑只针对一个样本$x$最大化$\log p(x;\theta)$，写为：
$$\log p(x;\theta) = \log \sum_{z}p(x,z;\theta)\tag{11.5}$$
令$Q$为$z$的可能值上的一个分布，即$\sum_{z} Q(z) = 1, Q(z)\ge 0$

考虑以下式子：
$$\begin{align}
\log p(x;\theta) &=\log \sum_z p(x,z;\theta)\\
&=\log \sum_{z} Q(z) \frac {p(x,z;\theta)}{Q(z)}\tag{11.6}\\
&\ge \sum_{z}Q(z)\log \frac {p(x,z;\theta)}{Q(z)}\tag{11.7}
\end{align}$$
其中公式11.6到公式11.7的推导用到了Jensen不等式，具体地说，$f(x) = \log x$是一个凹函数，因为所有的$x\in \mathbb R^+$都满足$f''(x) = -1/x^2 <0$，而项$\sum_{z}Q(z) [\frac {p(x,z;\theta)}{Q(z)}]$可以视作$[p(x,z;\theta)/ Q(z)]$相对从属于分布$Q$的随机变量$z$的期望值，因此根据Jensen不等式，我们有：
$$f\left(E_{z\sim Q}\left[\frac {p(x,z;\theta)} {Q(z)}\right]\right) \ge E_{z\sim Q}\left[f(\frac {p(x,z;\theta)} {Q(z)})\right]$$
此时，对于任意分布$Q$，公式11.7都给出了$\log p(x;\theta)$的一个下界，
对于$Q$，我们有许多可能的选择，而我们希望在给定当前的参数$\theta$下，选择可以使得不等号$\ge$保持为等号的$Q$

而要令Jensen不等式保持为等号，我们知道这需要随机变量为常数，也就是说要求：
$$\frac {p(x,z;\theta)}{Q(z)} = c$$
其中$c$是某个不取决于$z$的常数
要满足该式，我们可以选择
$$Q(z) \propto p(x,z;\theta)$$

因为$Q$是一个分布，显然满足$\sum_z Q(z) = 1$，
因此可以设$Q(z) = kp(x,z;\theta)$，则$\sum_zQ(z) = k\sum_z p(x,z;\theta) = 1$，解出$k = \frac {1}{\sum_z p(x,z;\theta)}$，故：
$$\begin{align}
Q(z)&=\frac {p(x,z;\theta)}{\sum_{z}p(x,z;\theta)}\\
&=\frac {p(x,z;\theta)}{p(x;\theta)}\\
&=p(z|x;\theta)\tag{11.8}
\end{align}$$
因此，我们要选择的$Q$就是给定参数$\theta$时$z$相对$x$的后验概率

我们把$Q(z) = p(z|x;\theta)$代入公式11.7得到：
$$\begin{align}
\sum_z Q(z)\log \frac {p(x,z;\theta)}{Q(z)} 
&= \sum_zp(z|x;\theta)\log \frac{p(x,z;\theta)}{p(z|x;\theta)}\\
&=\sum_z p(z|x;\theta )\log \frac {p(z|x;\theta)p(x;\theta)}{p(z|x;\theta)}\\
&=\sum_zp(z|x;\theta)\log p(x;\theta)\\
&=\log p(x;\theta)\sum_z p(z|x;\theta)\\
&=\log p(x;\theta)
\end{align}$$
显然此时Jensen不等式等号成立

我们称公式11.7的表达式为置信下界(ELBO evidence lower bound)，记作：
$$\text{ELBO}(x;Q,\theta) = \sum_zQ(z)\log \frac {p(x,z;\theta)}{Q(z)}\tag{11.9}$$
因此公式11.7也可以重写为：
$$\forall Q,\theta,x,\quad \log p(x;\theta)\ge \text{ELBO}(x;Q,\theta)\tag{11.10}$$

直观上说，EM算法交替更新$Q$和$\theta$，根据
a) 固定$\theta$，将$Q$设为$p(z|x;\theta)$，以满足在当前的$\theta$下，$\text{ELBO}(x;Q,\theta) = \log p(x;\theta)$对所有$x$都成立
b) 固定$Q$，更新$\theta$以最大化$\text{ELBO}(x;Q,\theta)$

以上的讨论都基于我们要最大化单个样本的对数似然$\log p(x;\theta)$，我们可以拓展到多个样本的情况

我们有训练集$\{x^{(1)},\dots,x^{n}\}$，注意$Q$的最优选择是$p(z|x;\theta)$，它依赖于特定的样本$x$，因此，我们需要引入$n$个分布$Q_1,\dots,Q_n$，和样本$x^{(i)}$一一对应，对于每个样本$x^{(i)}$我们都可构建它的ELBO：
$$\log p(x^{(i)};\theta)\ge \text{ELBO}(x^{(i)};Q_i,\theta)=\sum_{z^{(i)}}Q_i(z^{(i)})\log\frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}$$
对所有样本求和，得到训练集对数似然的下界：
$$\begin{align}
\mathscr l(\theta) &\ge \sum_i \text{ELBO}(x^{(i)};Q_i,\theta)\\
&=\sum_{i}\sum_{z^{(i)}}Q_i(z^{(i)})\log\frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
\end{align}\tag{11.11}$$
对于任意的分布集合$Q_1,\dots,Q_n$，公式11.11都给出了$\mathscr l(\theta)$的一个下界，同时我们知道，当$Q_i$满足：
$$Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)};\theta)$$
时，等号成立
因此，我们简单地令$Q_i$为$z^{(i)}$在给定$x^{(i)}$和当前参数$\theta$时的后验分布
这就是算法的E-step，我们找到了对数似然$\mathscr l$的下界
然后，在算法的M-step，我们优化这一下界，更新参数$\theta$以最大化公式11.11

因此EM算法就是交替进行这两个步骤直到收敛：
E-step：对每个$i$，令
$$Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)};\theta)$$
M-step：令
$$\begin{align}
\theta &= \arg\max_{\theta} \sum_{i=1}^n \text{ELBO}(x^{(i)};Q_i,\theta)\\
&=\arg\max_{\theta}\sum_{i}\sum_{z^{(i)}}Q_i(z^{(i)})\log\frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
\end{align}\tag{11.12}$$
需要证明$\mathscr l (\theta^{(t)}) \le \mathscr l (\theta^{(t+1)})$，以说明EM算法是单调提升对数似然的
假设我们以$\theta^{(t)}$起始，在E-step中，我们选择$Q_i^{(t)}(z^{(i)}) = p(z^{(i)}|x^{(i)};\theta^{(t)})$，此时Jensen不等式等号成立，因此有：
$$\mathscr l(\theta^{(t)}) =\sum_{i=1}^n \text{ELBO}(x^{(i)};Q_i^{(t)},\theta^{(t)})\tag{11.13}$$
我们通过最大化公式11.13的RHS以得到$\theta^{(t+1)}$，因此有：
$$\begin{align}
\mathscr l(\theta^{(t+1)}) & \ge\sum_{i=1}^n \text{ELBO}(x^{(i)};Q_i^{(t)},\theta^{(t+1)})\\
&\ge\sum_{i=1}^n \text{ELBO}(x^{(i)};Q_i^{(t)},\theta^{(t)})\\
&=\mathscr l(\theta^{(t)})
\end{align}$$
因此，EM算法可以让对数似然单调递增，EM算法可以通过检查似然函数的增量是否小于预定的容许参数(tolerance parameter)以确定是否收敛

定义：
$$\text{ELBO}(Q,\theta) = \sum_{i=1}^n \text{ELBO}(x^{(i)};Q_i,\theta)= \sum_{i}\sum_{z^{(i)}}Q_i(z^{(i)})\log\frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\tag{11.14}$$
可以知道$\mathscr l(\theta) \ge \text{ELBO}(Q,\theta)$，EM算法可以视为对$\text{ELBO}(Q,\theta)$的交替优化算法，E-step更新$Q$以最大化$\text{ELBO}(Q,\theta)$，M-step更新$\theta$以最大化$\text{ELBO}(Q,\theta)$
#### 11.3.1 Other interpretation of ELBO
根据公式11.9，我们令$\text{ELBO}(x;Q,\theta) = \sum_z Q(z)\log \frac {p(x,z;\theta)}{Q(z)}$，ELBO实际上有多种形式，首先，我们可以将其写为：
$$\begin{align}
\text{ELBO}(x;Q,\theta) &=\mathbb E_{z\sim Q}[\log p(x,z;\theta)]- \mathbb E_{z\sim Q}[\log Q(z)]\\
&=\mathbb E_{z\sim Q}[\log p(x|z;\theta)p(z;\theta)] - \mathbb E_{z\sim Q}[\log Q(z)]\\
&=\mathbb E_{z\sim Q}[\log p(x|z;\theta)] -\mathbb E_{z\sim Q}[\log \frac {Q(z)}{p(z;\theta)}]\\
&=\mathbb E_{z\sim Q}[\log p(x|z;\theta)] - \sum_z Q(z)\log \frac {Q(z)}{p(z;\theta)}\\
&=\mathbb E_{z\sim Q}[\log p(x|z;\theta)] - D_{KL}(Q||p_z)
\end{align}\tag{11.15}$$
其中我们使用了$p_z$表示了$z$在联合分布$p(x,z;\theta)$下的边际分布，$D_{KL}$表示KL散度/相对熵：
$$D_{KL}(Q||p_z) = \sum_z Q(z)\log \frac {Q(z)}{p(z)}\tag{11.16}$$
在许多情况下，$z$的边际分布并不依赖于参数$\theta$，此时，我们可以看到相对$\theta$最大化ELBO等价于最大化公式11.15中的第一项，即最大化$x$条件于$z$的条件似然，这往往会比原问题更加简单

$\text{ELBO}(\cdot)$的另一种形式写为：
$$\begin{align}
\text{ELBO}(x;Q,\theta) &= \mathbb E_{z\sim Q}[\log p(x,z;\theta)]- \mathbb E_{z\sim Q}[\log Q(z)]\\
&=\mathbb E_{z\sim Q}[\log p(x;\theta)p(z|x;\theta)] - \mathbb E_{z\sim Q} [\log Q(z)] \\
&=\mathbb E_{z\sim Q}[\log p(x;\theta)] - \mathbb E_{z\sim Q}[\log \frac {Q(z)}{p(z|x;\theta)}]\\
&=\log p(x;\theta) - \sum_{z}Q(z)\log \frac {Q(z)}{p(z|x;\theta)}\\
&=\log p(x) - D_{KL}(Q||p_{z|x})
\end{align}\tag{11.17}$$
其中$p_{z|x}$即为$z$在参数$\theta$下给定$x$的条件分布，该形式说明了相对$Q$最大化ELBO，就是令$Q = p(z|x)$，这个结果我们在公式11.8已经给出过
### 11.4 Mixture of Gaussians revisited
回到之前拟合混合高斯分布的参数$\phi,\mu,\Sigma$的例子
E-step
和前面讨论的一样，在这一步我们需要确定隐变量$z^{(i)}$的分布，我们直接计算
$$w_j^{(i)} = Q_i(z^{(i)} = j ) = P(z^{(i)} = j | x^{(i)};\phi,\mu,\Sigma)$$
其中，$Q_i(z^{(i)} = j)$表示$z^{(i)}$在分布$Q_i$下取值$j$的概率
M-step
在这一步，我们相对我们的参数$\phi,\mu,\Sigma$最大化数据集的对数似然，即最大化：
$$\begin{align}
&\sum_{i=1}^n\sum_{z^{(i)}}Q_i(z^{(i)})\log \frac {p(x^{(i)},z^{(i)};\phi,\mu,\Sigma)}{Q_i(z^{(i)})}\\
=&\sum_{i=1}^n\sum_{j=1}^k Q_i(z^{(i)} = j)\log \frac {p(x^{(i)}|z^{(i)}=j;\mu,\Sigma)p(z^{(i)}=j;\phi)}{Q_i(z^{(i)}=j)}\\
=&\sum_{i=1}^n\sum_{j=1}^kw_j^{(i)}\log \frac {\frac 1{(2\pi)^{d/2}|\Sigma_j|^{1/2}}\exp(-\frac 1 2(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j))\cdot\phi_j}{w^{(i)}_j}
\end{align}$$
先考虑相对于参数$\mu_l$最大化该式，我们计算该式相对于$\mu_l$的导数：
$$\begin{align}
&\nabla_{\mu_l}\sum_{i=1}^n\sum_{j=1}^kw_j^{(i)}\log \frac {\frac 1{(2\pi)^{d/2}|\Sigma_j|^{1/2}}\exp(-\frac 1 2(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j))\cdot\phi_j}{w^{(i)}_j}\\
=&-\nabla_{\mu_l}\sum_{i=1}^n\sum_{j=1}^kw_j^{(i)}\frac 1 2(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)\\
=&\frac 1 2\sum_{i=1}^n w_l^{(i)}\nabla_{u_l}2\mu_l^T\Sigma_l^{-1}x^{(i)}-\mu_l^T\Sigma_l^{-1}\mu_l\\
=&\sum_{i=1}^nw_l^{(i)}(\Sigma_l^{-1}x^{(i)}-\Sigma_l^{-1}\mu_l)
\end{align}$$
将导数设为0，求解$\mu_l$，我们得到更新规则：
$$\mu_l:=\frac {\sum_{i=1}^nw^{(i)}_lx^{(i)}}{\sum_{i=1}^nw_l^{(i)}}$$
我们再考虑相对于参数$\phi_j$最大化该式，我们找出依赖于$\phi_j$的项，可以发现我们仅需要最大化：
$$\sum_{i=1}^n\sum_{j=1}^k w_j^{(i)}\log \phi_j$$
同时我们要注意到此时有一条额外的约束，即$\sum_{j=1}^k \phi_j = 1$，因为它们表示了概率$\phi_j = p(z^{(i)} = j;\phi)$，因此我们构造拉格朗日函数：
$$\mathcal L(\phi) = \sum_{i=1}^n\sum_{j=1}^k w_j^{(i)}\log \phi_j + \beta(\sum_{j=1}^k\phi_j -1)$$
其中$\beta$是拉格朗日乘子，我们计算导数，得到：
$$\frac {\partial}{\partial \phi_j} \mathcal L(\phi) = \sum_{i=1}^n\frac {w_j^{(i)}}{\phi_j}+\beta$$
令导数为零，我们解出：
$$\phi_j=\frac {\sum_{i=1}^nw_j^{(i)}}{-\beta}$$
即$\phi_j \propto \sum_{i=1}^n w_j^{(i)}$，使用约束$\sum_j \phi_j = 1$，我们可以轻松解出$-\beta = \sum_{i=1}^n\sum_{j=1}^k w_j^{(i)} = \sum_{i=1}^n 1 = n$，因此$\phi_j$的更新规则就是：
$$\phi_j :=\frac 1 n\sum_{i=1}^n w_j^{(i)}$$
最后我们考虑相对于参数$\Sigma_j$最大化该式，我们找到依赖于$\Sigma_j$的项，可以发现我们仅需最大化：
$$-\frac 1 2\sum_{i=1}^n w_j^{(i)}\left((x^{(i)}-\mu_j)\Sigma_j^{-1}(x^{(i)}-\mu_j)+\log|\Sigma_j|\right)$$
计算导数，我们得到：
$$\begin{align}
&\nabla_{\Sigma_j}\frac 1 2\sum_{i=1}^n w_j^{(i)}\left((x^{(i)}-\mu_j)\Sigma_j^{-1}(x^{(i)}-\mu_j)+\log|\Sigma_j|\right)\\
=&-\frac 1 2\sum_{i=1}^nw_j^{(i)}\nabla_{\Sigma_j}\left( (x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)+\log|\Sigma_j|\right)\\
=&\frac 1 2\sum_{i=1}^nw_j^{(i)}\left(\Sigma_j^{-T}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T\Sigma_j^{-T }-\Sigma_j^{-1}\right)\\
=&\frac 1 2\sum_{i=1}^nw_j^{(i)}\left(\Sigma_j^{-1}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T\Sigma_j^{-1}-\Sigma_j^{-1}\right)\\
\end{align}$$
令导数等于零，我们得到：
$$\begin{align}
\sum_{i=1}^n w_j^{(i)}\Sigma_j^{-1}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T\Sigma_j^{-1} &= \sum_{i=1}^nw_j^{(i)}\Sigma_j^{-1}\\
\sum_{i=1}^n w_j^{(i)}\Sigma_j^{-1}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T &= \sum_{i=1}^nw_j^{(i)}\\
\Sigma_j^{-1}\sum_{i=1}^n w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T &= \sum_{i=1}^nw_j^{(i)}\\
\sum_{i=1}^n w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T &= \Sigma_j\sum_{i=1}^nw_j^{(i)}\\
\frac {\sum_{i=1}^n w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\sum_{i=1}^nw_j^{(i)}} &= \Sigma_j\\
\end{align}$$
因此得到$\Sigma_j$的更新规则：
$$\Sigma_j:=\frac {\sum_{i=1}^n w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\sum_{i=1}^nw_j^{(i)}}$$
### 11.5 Variational inference and variational auto-encoder
不严谨地说，变分自动编码机一般指一族将EM算法拓展至由神经网络参数化的模型的算法，它延伸了变分推理(variational inference)技术以及一个“再参数化技巧(re-parameterization trick)”

变分自动编码机包含了如何将在非线性模型中将EM算法延伸至高维连续隐变量的中心思想

我们考虑用一个神经网络参数化$p(x,z;\theta)$，令$\theta$为网络$g(z;\theta)$的权重集合，网络$g(z;\theta)$将$z\in \mathbb R^k$映射到$\mathbb R^d$，令：
$$\begin{align}
&z\sim \mathcal N(0, I_{k\times k})\tag{11.18}\\
&x|z\sim \mathcal N(g(z;\theta),\sigma^2I_{d\times d})\tag{11.19}
\end{align}$$
其中$I_{k\times k}$是单位矩阵，$\sigma$是已知的标量

对于之前讨论的高斯混合模型，对固定的$\theta$，$Q(z)$的最优选择就是$Q(z) = p(z|x;\theta)$，即$z$的后验分布，这是可以解析式地计算得到的，但许多更复杂的模型中，例如公式11.19，计算准确的后验分布$p(z|x;\theta)$是不可解的

回忆公式11.10，ELBO对于任意的$Q$都会是$\log p(x;\theta)$的下界，此时我们无法解出准确的$Q$/后验分布使不等式等号成立，而是目标于找到对真实的后验分布的近似(approximation)
我们常常需要用某种特定的形式(假定$Q$的形式)以近似真实的后验分布，令$\mathcal Q$为我们所考虑的一族$Q$，我们希望在$\mathcal Q$中找到最接近真实后验分布的$Q$
我们回忆公式11.14对ELBO的定义：
$$\text{ELBO}(Q,\theta) = \sum_{i=1}^n \text{ELBO}(x^{(i)};Q_i,\theta)=\sum_i\sum_{z^{(i)}}Q_i(z^{(i)})\log \frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}$$
而EM算法可以视作是对$\text{ELBO}(Q,\theta)$的交替最大化，此时，在E-step，我们需要找到$Q\in \mathcal Q$以最大化ELBO：
$$\max_{Q\in \mathcal Q} \max_{\theta} \text{ELBO}(Q,\theta)\tag{11.20}$$

此时的一个问题是$Q$的什么形式(要对$Q$做什么样的结构化的假设)允许我们高效地最大化以上目标，当隐变量$z$是高维的离散变量时，一个常用的假设是平均场假设(mean field assumption)，即假设$Q_i(z)$的分布是坐标独立的(independent coordinates)，换句话说，$Q_i$可以被分解为$Q_i(z) = Q_i^1(z)\cdots Q_i^k(z_k)$
我们关注的是$z$是高维的连续变量的情况，因此，除了平均场假设外，我们还需要其他的技术

当$z\in \mathbb R^k$是连续隐变量时，我们需要做出若干决策以成功优化式11.20，
首先，我们需要给分布$Q_i$一个简单的(succinct)表示，因为$Q_i$是在无限个点上的分布，一个自然的选择是假设$Q_i$是高斯分布，带有特定的均值和方差，
我们进而希望$Q_i$的均值有一个简单的表示，注意到$Q_i$是用于近似$p(z^{(i)}|x^{(i)};\theta)$的，因此让$Q_i$的均值是某个关于$x^{(i)}$的函数是较为合理的
具体地说，令$q(\cdot;\phi),v(\cdot;\psi)$是两个从$d$维映射到$k$维的函数，分别由$\phi$和$\psi$参数化，我们假设：
$$Q_i = \mathcal N(q(x^{(i)};\phi),\text{diag}(v(x^{(i)};\psi))^2)\tag{11.21}$$
其中，$\text{diag}(w)$表示一个$k\times k$的矩阵，其对角线上的项来自于$w\in \mathbb R^k$，换句话说，分布$Q_i$被假设为坐标独立的多元高斯分布(Gaussian distribution with independent coordinates)，其均值和标准差由$q$和$v$管理

在变分自动编码机(VAE)中，$q$和$v$一般采用神经网络的形式($q,v$还可以共享参数)，$q,v$一般被称为编码器(encoder)(可以认为它将数据编码为隐式码 latent code)，而$g(z;\theta)$一般被称为解码器(decoder)

要注意，上述形式的$Q_i$在许多情况下往往不会是一个对真实后验分布的一个好的近似，但是为了优化的可行性，一些近似形式是必要的，也就是$Q_i$的形式需要满足一些其他的要求(公式11.21的形式是满足这些要求的)

在优化ELBO之前，我们先尝试对于一个固定的，形式为公式11.21的$Q$，以及固定的$\theta$，是否可以高效地评估ELBO的值，我们将ELBO重写为关于$\phi,\psi,\theta$的函数的形式：
$$\begin{align}
\text{ELBO}&=\sum_{i=1}^n \text E_{z^{(i)}\sim Q_i}\left[\log \frac {p(x^{(i)}, z^{(i)};\theta)}{Q_i(z^{(i)})}\right]\\
&\text{where }Q_i = \mathcal N(q(x^{(i)};\phi),\text{diag}(v(x^{(i)};\psi))^2)
\end{align}\tag{11.22}$$
为了能够评估期望中的$Q_i(z^{(i)})$的值，我们需要能够计算$Q_i$的概率密度，
为了能够评估期望$\text {E}_{z^{(i)}\sim Q_i}$，我们需要能够从分布$Q_i$中采样，以根据采样的样本对其进行经验估计，
而现在我们将$Q_i$的形式写为了$Q_i = \mathcal N(q(x^{(i)};\phi),\text{diag}(v(x^{(i)};\psi))^2)$，显然我们可以高效地完成二者

现在我们考虑优化ELBO，我们可以针对$\phi,\psi,\theta$进行梯度上升，而不用交替最大化(alternating maximization)，因为我们并不特别需要计算ELBO相对于每个变量的最大值，这需要很大的开销
(对于11.4节的高斯混合模型，计算最大值是解析上可行的，且是相对廉价的，因此我们在11.4节使用的是交替最大化)

在数学上，我们令$\eta$为学习率，则梯度上升步骤为：
$$\begin{align}
\theta&:=\theta + \eta \nabla_{\theta}\text{ELBO}(\phi,\psi,\theta)\\
\phi&:=\phi + \eta \nabla_{\theta}\text{ELBO}(\phi,\psi,\theta)\\
\psi&:=\psi + \eta \nabla_{\theta}\text{ELBO}(\phi,\psi,\theta)\\
\end{align}$$

相对于$\theta$计算梯度是较为简单的：
$$\begin{align}
\nabla_{\theta}\text{ELBO}(\phi,\psi,\theta) &=\nabla_{\theta}\sum_{i=1}^n \text E_{z^{(i)}\sim Q_i}\left [\log \frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right]\\
&=\nabla_{\theta}\sum_{i=1}^n\text E_{z^{(i)}\sim Q_i}[\log p(x^{(i)},z^{(i)};\theta)]\\
&=\sum_{i=1}^n\text {E}_{z^{(i)}\sim Q_i}[\nabla_{\theta}\log p(x^{(i)},z^{(i)};\theta)]
\end{align}\tag{11.23}$$
但相对于$\phi,\psi$计算梯度则较为棘手，因为采样分布$Q_i$决定于$\phi,\psi$
(抽象地说，我们遇到的问题可以简化为计算$\text E_{z\sim Q_{\phi}}[f(\phi)]$相对于$\phi$的梯度，我们知道，一般情况下，$\nabla \text E_{z\sim Q_{\phi}}[f(\phi)] \ne \text E_{z\sim Q_{\phi}}[\nabla f(\phi)]$，因为计算梯度时，$Q_{\phi}$对$\phi$的依赖也需要同时被考虑)

解决该问题的方法称为再参数化技巧(reparameterization trick)，我们将$z^{(i)}\sim Q_i = \mathcal N(q(x^{(i)};\phi),\text {diag}(v(x^{(i)};\psi))^2)$重写为一种等价的形式：
$$z^{(i)}= q(x^{(i)};\phi) + v(x^{(i)};\psi)\odot \xi^{(i)}\sim \mathcal N(0,I_{k\times k})\tag{11.24}$$
其中$x\odot y$表示两个相同维度向量的按元素乘法
我们利用了$x\sim N(\mu, \sigma^2)$等价于$x = \mu + \xi \sigma\text{ with } \sigma \sim N(0,1)$的事实，实际上公式11.24就是对随机变量$z^{(i)}\sim Q_i$的每一个维度都同时使用了这一事实

使用重参数化技巧将$z^{(i)}$的分布重写之后，我们得到：
$$\begin{align}
&\text E_{z^{(i)}\sim Q_i}\left[\log \frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right]\\
=&\text E_{\xi^{(i)}\sim \mathcal N(0,I_{k\times k})}\left[\log\frac {p(x^{(i)},q(x^{(i)};\phi) + v(x^{(i)};\psi)\odot\xi^{(i)};\theta)}{Q_i\left(q(x^{(i)};\phi)+v(x^{(i)};\psi)\odot \xi^{(i)}\right)}\right]
\end{align}\tag{11.25}$$
因此：
$$\begin{align}
&\nabla_{\phi}\text E_{z^{(i)}\sim Q_i}\left[\log \frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right]\\
=&\nabla_{\phi}\text E_{\xi^{(i)}\sim \mathcal N(0,I_{k\times k})}\left[\log\frac {p(x^{(i)},q(x^{(i)};\phi) + v(x^{(i)};\psi)\odot\xi^{(i)};\theta)}{Q_i\left(q(x^{(i)};\phi)+v(x^{(i)};\psi)\odot \xi^{(i)}\right)}\right]\\
=&\text E_{\xi^{(i)}\sim \mathcal N(0,I_{k\times k})}\left[\nabla_{\phi}\log\frac {p(x^{(i)},q(x^{(i)};\phi) + v(x^{(i)};\psi)\odot\xi^{(i)};\theta)}{Q_i\left(q(x^{(i)};\phi)+v(x^{(i)};\psi)\odot \xi^{(i)}\right)}\right]
\end{align}\tag{11.26}$$
此时，我们可以对$\xi^{(i)}$进行多次的采样，对公式11.26的RHS的期望值进行估计(在经验上，人们一般仅采样一次就进行估计，以追求最高的计算效率)

对ELBO相对于$\psi$的梯度的估计是类似的，最后得到的是：
$$\begin{align}
&\nabla_{\psi}\text E_{z^{(i)}\sim Q_i}\left[\log \frac {p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right]\\
=&\nabla_{\psi}\text E_{\xi^{(i)}\sim \mathcal N(0,I_{k\times k})}\left[\log\frac {p(x^{(i)},q(x^{(i)};\phi) + v(x^{(i)};\psi)\odot\xi^{(i)};\theta)}{Q_i\left(q(x^{(i)};\phi)+v(x^{(i)};\psi)\odot \xi^{(i)}\right)}\right]\\
=&\text E_{\xi^{(i)}\sim \mathcal N(0,I_{k\times k})}\left[\nabla_{\psi}\log\frac {p(x^{(i)},q(x^{(i)};\phi) + v(x^{(i)};\psi)\odot\xi^{(i)};\theta)}{Q_i\left(q(x^{(i)};\phi)+v(x^{(i)};\psi)\odot \xi^{(i)}\right)}\right]
\end{align}\tag{11.27}$$

得到了ELBO相对于$\theta,\phi,\psi$的梯度之后，就可以进行梯度上升，优化ELBO

实际中并不存在许多密度函数是解析上可计算的且可以重参数化的高维分布，能替代Gussian分布的分布选择是比较少的，具体参考Kingma and Welling
## CH 12 Principle component analysis
## CH 14 Self-supervised learning and foundation models
# Part 5 Reinforcement Learning and Control
## CH 15 Reinforcement learning
## CH 16 LQR, DDP and LQG
## CH 17 Policy Gradient(REINFORCE)

# Appendix
## Decision Trees
## Probability Theory Review and Reference

