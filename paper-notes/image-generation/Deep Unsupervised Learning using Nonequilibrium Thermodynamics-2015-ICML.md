# Abstract
机器学习的一个中心问题设计使用高度灵活的概率分布族(highly flexible families of probability distributions)对复杂的数据集进行建模，且要求这些概率分布在学习(learning)、采样(sampling)、推理(inference)和评估(evaluation)上仍然是解析上或计算上可行的(analytically or computationally tractable)

在本文中，我们开发来了一种可以同时具有灵活性(flexibility)和可解性(tractability)的方法，其中心思想，受非平衡统计物理的启发，是通过一个迭代式地前向扩散过程(iterative forward diffusion process)系统地且缓慢地破坏数据分布中地结构(destroy structure in a data distribution)，之后，我们学习一个逆向扩散过程(reverse diffusion process)以恢复(restore)数据中的结构，这便是一个高度灵活且可解(highly flexible and tractable)的数据生成式模型

该方法允许我们在快速地从具有数千个层或时间步(time steps)的深度生成式模型中采样、评估概率以对其进行学习，以及在学习到的模型下计算条件或后验概率
# 1 Introduction
历史上，概率模型需要从两个冲突的目标之间权衡：可解性和灵活性(tractability and flexibility)，
可解的模型可以被解析式地评估并且非常易于对数据进行拟合(例如Gaussian或Laplace)，但是它们不能充足地描述复杂数据集(rich datasets)中的结构，
另一方面，灵活的模型可以拟合任意数据中的结构，例如，我们可以将模型定义为任意的(非负)函数$\phi(\mathbf x)$，以产生灵活的分布$p(\mathbf x) = \frac {\phi(\mathbf x)} {Z}$，其中$Z$是规范化常数(normalization constant)，但是，对这个规范化常数的计算通常是不可解的，评估、训练、以及从这类灵活模型中采样通常需要非常昂贵的蒙特卡洛过程

存在许多解析式的方法缓解了这种权衡(但没有移除)，例如平均场理论和它的拓展，变分贝叶斯，对比散度，最小概率流，最小KL收缩，适当评分规则，分数匹配，伪似然，有环信念传播，
非参数方法也可以非常有效$^1$

---
$^1$非参数方法可以视为可解模型和灵活模型之间的平滑过渡(transition)，例如，一个非参数的Gaussian混合模型可以使用单个Gaussian表示少量的(a small amount of)数据，也可以将无限的数据(infinite data)表示为无限数量Gaussians的混合
## 1.1 Diffusion probabilistic models
我们将展示一个新颖的方式定义一种概率模型，它允许：
1. 模型结构上的极度灵活性(extreme flexibility)
2. 准确的采样(exact sampling)
3. 容易和其他分布相乘(easy multiplication with other distributions)，例如为了计算后验概率，以及
4. 模型对数似然(model log likelihood)，以及单个状态(individual states)的概率，都可以被廉价地评估

我们的模型使用马尔可夫链以逐渐地将一个分布转化为另一个分布，这种思想已经被应用于非平衡统计物理和序列蒙特卡洛中
我们构建一个生成式的Markov，通过一个扩散过程，将简单的已知的分布(例如Gaussian)转化为目标(数据)分布

我们没有使用这个Markov链以近似评估一个另外定义的模型，我们显式地将概率模型定义为Markov链的末端点(endpoint)，因为扩散过程中的每一步都有可以被解析式评估的概率(analytically evaluable probability)，因此完整的Markov链也可以被解析式评估

在该框架中学习涉及到估计对扩散过程的小扰动(small perturbations)进行估计，相较于用单个非解析式可规范化的(non-analytically-normalizable)潜在函数显式地描述整个分布，估计小的扰动是更可解的，进一步，因为对任意平滑的目标分布都存在(exists for any smooth target distribution)一个扩散过程，该方法可以捕获任意形式的数据分布(data distributions of arbitrary form)

我们将通过为二维的swiss roll，二元序列(binary sequence)，手写数字(MNIST)，以及数个自然图像数据集(CIFAR-10, bark, dead leaves)训练出带有高对数似然的模型来展示这类扩散概率模型的功能
## 1.2 Relationship to other work
唤醒-睡眠(wake-sleep)算法(Hinton, 1995; Dayan et al., 1995)引入了相互训练推理和生成式概率模型的概念(training inference and generative probablistic models against each other)，这种方法在近二十年中基本上未被探索，尽管有一些例外(Sminchisescu et al., 2006; Kavukcuoglu et al., 2010)，最近，发展这一想法的工作有了爆炸性的增长

(Kingma & Welling, 2013; Gregor et al., 2013; Rezende et al., 2014; Ozair & Bengio, 2014)中介绍了变分学习和推理算法，该算法允许相互训练一个灵活的生成式模型和隐变量的后验分布
这些论文中的变分界限(variational bound)与我们训练目标中使用的界限以及早期工作(Sminchisescu et al., 2006)中使用的界限相似，然而，我们的动机和模型形式都非常不同，而且与这些技术相比，我们的工作保留了以下差异和优势：
1. 我们使用来自物理学、准静态过程(quasi-static processes)和退火重要性抽样(annealed importance sampling)的思想来开发我们的框架，而不是来自变分贝叶斯方法的思想
2. 我们展示了如何轻松地将学到的分布与另一个概率分布相乘(例如，为了计算后验而与一个条件分布相乘)
3. 我们解决了在变分推理方法中训练推理模型可能特别具有挑战性的困难，该困难来自于推理和生成模型之间目标的不对称性(asymmetry in the objective between the inference and generative models)，我们将前向(推理)/forward(inference)过程限制为一个简单的函数形式(restrict the process to a simple functional form)，这样反向(生成)/reverse(generative)过程将具有相同的函数形式(have the same functional form)
4. 我们训练具有数千层(或时间步)/layers(or time steps)的模型，而不仅仅是少数几层
5. 我们提供了每层(或时间步)熵增量(entropy production)的上下界限(upper and lower bound)

除了上述技术以外，还有一些相关的技术用于训练概率模型，这些技术包括为生成式模型开发高度灵活的形式(highly flexible forms)，训练随机轨迹(stochastic trajectories)，或学习贝叶斯网络的逆(the reversal of a Baysian network)：
重加权的唤醒-睡眠(Bornschein & Bengio, 2015)拓展了原始唤醒-睡眠算法了并改进了学习规则；生成随机网络(Bengio & Thibodeau-Laufer, 2013; Yao et al., 2014)训练一个马尔可夫核，将其平衡分布与数据分布相匹配(match its equilibrium distribution to the data distribution)；神经自回归分布估计器(Larochelle & Murray, 2011)(及其循环(Uria et al., 2013a)和深度(Uria et al., 2013b)扩展)将联合分布分解为每个维度上的一系列可解的条件分布；对抗网络(Goodfellow et al., 2014)训练一个生成模型与分类器对抗，其中分类器试图区分生成样本和真实数据；在(Schmidhuber, 1992)中的一个类似的目标用于学习一个到具有边际独立单元(marginally independent units)的表示的双向映射；(Rippel & Adams, 2013; Dinh et al., 2014)学习到具有简单因子密度函数的潜在表示的双向映射；(Stuhlmuller et al., 2013)为贝叶斯网络学习了随机逆；条件高斯比例混合(MCGSMs Mistures of conditional Gaussian scale mixtures)(Theis et al., 2012)使用高斯比例混合(Gaussian scale mixtures)描述数据集，其参数取决于一个因果邻域序列(a sequence of causal neighborhoods)；此外，还有大量工作学习从简单的隐分布学习到数据分布的灵活生成式映射——早期的例子包括(MacKay, 1995)，其中神经网络被引入为生成模型，以及(Bishop et al., 1998)学习了从隐空间到数据空间的随机流形映射
我们将与对抗网络和MCGSMs进行实验比较

物理学中的相关思想包括Jarzynski等式(Jarzynski, 1997)，在机器学习中称为退火重要性抽样(AIS Annealed Importance Sampling)(Neal, 2001)，它使用一个马尔可夫链，该链缓慢地将一个分布转换为另一个分布，以计算正规化常数的一个比率(a ratio of normalizing constants)，(Burda et al., 2014)中展示了AIS也可以使用反向而非正向轨迹(reverse rather than forward trajectory)执行；
Langevin动力学(Langevin, 1908)是Fokker-Planck方程的随机实现(stochastic realization)，它展示了如何定义一个高斯扩散过程，该过程可以以任何目标分布作为其平衡(has any target distribution as its equilibrium)；在(Suykens & Vandewalle, 1995)中，Fokker-Planck方程被用于执行随机优化(stochastic optimization)；
最后，Kolmogorov前向和后向方程(forward and backward equations)(Feller, 1949)表明，对于许多前向扩散过程，反向扩散过程可以使用相同的功能形式来描述(be described using the same functional form)
# 2 Algorithm
我们的目标是定义一个前向(或推理)扩散过程，它将任意复杂的数据分布转化为一个简单的、可解的分布(converts any complex data distribution into a simple, tractable distribution)，然后学习该扩散过程的有限时间的逆(a finite-time reversal of this diffusion process)，该逆过程就定义了我们的生成式模型分布(generative model distribution)
![[Deep Unsupervised Learning using Nonequilibrium Thermodynamics-Fig1.png]]

我们首先描述前向的/推理扩散过程，然后展示如何训练逆向的/生成式扩散过程并将其用于评估概率，我们也为逆过程推导了熵界(entropy bounds)，并展示学习到的分布与任意的第二个分布相乘(例如，在对图像修复或为其去噪的时候会计算它的后验分布)
## 2.1 Forward Trajectory
我们将数据分布标记为$q(\mathbf x^{(0)})$，通过重复应用一个Markov扩散核(diffusion kernel)$T_{\pi}(\mathbf y|\mathbf y';\beta)$，数据分布会逐渐被转换为一个良好的分布(well behaved/analytically tractable distribution)$\pi(\mathbf y)$，其中$\beta$是扩散率(the diffusion rate)：
$$\begin{align}
\pi(\mathbf y) &= \int d\mathbf y' T_{\pi}(\mathbf y|\mathbf y';\beta)\pi(\mathbf y')\tag{1}\\
q(\mathbf x^{(t)}|\mathbf x^{(t-1)})&= T_{\pi}(\mathbf x^{(t)} |\mathbf x^{(t-1)};\beta_t)\tag{2}
\end{align}$$

前向轨迹(the forward trajectory)对应于从数据分布开始，然后执行$T$步的扩散，因此写为：
$$q(\mathbf x^{(0\dots T)}) = q(\mathbf x^{(0)})\prod_{t=1}^T q(\mathbf x^{(t)}|\mathbf x^{(t-1)})\tag{3}$$
在本文的实验中，$q(\mathbf x^{(t)} | \mathbf x^{(t-1)})$要么对应于最终得到具有单位协防差矩阵的高斯分布的高斯扩散过程(Gaussian diffusion)的高斯扩散核，要么对应于最终得到一个独立二项分布(independent binomial distribution)的二项式扩散过程(binomial diffusion)的二项式扩散核
## 2.2 Reverse Trajectory
我们将训练生成式分布以描述描述相同的轨迹，但方向是逆向：
$$\begin{align}
p(\mathbf x^{(T)}) &= \pi(\mathbf x^{(T)}) \tag{4}\\
p(\mathbf x^{(0\dots T)}) &= p(\mathbf x^{(T)})\prod_{t=1}^T p(\mathbf x^{(t-1)}|\mathbf x^{(t)})\tag{5}
\end{align}$$
对于高斯和二项式扩散，如果是连续扩散(小步长$\beta$的极限情况 limit of small step size $\beta$)，扩散过程的逆和前向过程具有相同的函数形式(identical functional form)(Feller, 1949)
因此，如果$q(\mathbf x^{(t)}|\mathbf x^{(t-1)})$是高斯/二项分布，且如果$\beta_t$很小，则$q(\mathbf x^{(t-1)}|\mathbf x^{(t)})$也将是高斯/二项分布，轨迹越长，就可以将扩散率$\beta$设定得越小

在学习时，只有高斯扩散核的均值和方差(mean and variance)，或二项式核的位翻转概率(bit flip probability)需要被估计，如Table App.1所示，$\mathbf f_{\mu}(\mathbf x^{(t)}, t)$以及$\mathbf f_{\Sigma}(\mathbf x^{(t)}, t)$即定义了逆Markov转换(Gaussian)的均值和方差的函数，以及$\mathbf f_b(\mathbf x^{(t)}, t)$即定义了二项分布的位翻转概率的函数，执行该算法的计算开销就是执行这些函数的开销乘上时间步的数量(the number of time-steps)

本文中的所有结果都使用的是MLP来定义这些函数，当然也可以用大量其他的回归或函数拟合方法，包括了非参数方法
## 2.3 Model Probability
生成式模型赋予给数据的概率是：
$$p(\mathbf x^{(0)}) = \int d\mathbf x^{(1\dots T)} p (\mathbf x^{(0\dots T)})\tag{6}$$
直观上，这个积分是难以求解的——但是借鉴自退火重要性抽样(Annealed Importance Sampling)和Jarzynski等式，我们可以评估前向和反向轨迹的相对概率，在前向轨迹上平均(evaluate the relative probability of the forward and reverse trajectories, averaged over forward trajectory)：
$$\begin{align}
p(\mathbf x^{(0)}) &= \int d\mathbf x^{(1\dots T)}p(\mathbf x^{(0\dots T)})\frac {q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})}{q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})}\tag{7}\\
&= \int d\mathbf x^{(1\dots T)}q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})\frac {p(\mathbf x^{(0\dots T)})}{q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})}\tag{8}\\
&= \int d\mathbf x^{(1\dots T)}q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})\frac {p(\mathbf x^{(T)})\prod_{t=1}^Tp(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})}\\
&= \int d\mathbf x^{(1\dots T)}q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})\frac {p(\mathbf x^{(T)})\prod_{t=1}^Tp(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{\prod_{t=1}^Tq(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\\
&= \int d\mathbf x^{(1\dots T)}q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\tag{9}\\
\end{align}$$
这可以通过在来自于前向轨迹$q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})$的样本上进行平均以快速评估(be evaluated rapidly by averaging over samples from the forward trajectory)

对于无穷小的$\beta$，可以令轨迹上的前向和反向分布相同(identical)，如果它们是相同的，则只需要$q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})$中的一个样本就可以准确评估上述积分，这对应于统计物理学中的准静态过程(quasi-static process)
(
等式(7)-(9)实际上可以视作对从属于分布$q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})$的$\mathbf x^{(1\dots T)}$的样本求函数$f(\mathbf x^{(1\dots T)})$的期望：
$$\begin{align}
p(\mathbf x^{(0)}) &= \int d \mathbf x^{(1\dots T)}q(\mathbf x^{(1\dots T)} | \mathbf x^{(0)})p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\\
&=\mathbb E_{\mathbf x^{(1\dots T)}\sim q(\mathbf x^{(1\dots T)}| \mathbf x^{(0)})}\left[p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right]\\
&=\mathbb E_{\mathbf x^{(1\dots T)}\sim q(\mathbf x^{(1\dots T)}| \mathbf x^{(0)})}\left[f(\mathbf x^{(1\dots T)})\right]\\
\end{align}$$
当$\beta$趋近于无穷小，对于$\forall t \in [1,T]$，扩散分布写为：
$$\begin{align}
&\lim_{\beta_t \to 0} q(\mathbf x^{(t)}|\mathbf x^{(t-1)})\\
=&\lim_{\beta_t \to 0} \mathcal N(\mathbf x^{(t)};\mathbf x^{(t-1)}\sqrt {1-\beta_t},\mathbf I\beta_t)\\
=&\mathcal N(\mathbf x^{(t)};\mathbf x^{(t-1)},\mathbf 0)
\end{align}$$
高斯分布在方差为$0$是没有定义的，因此上式不严谨，但是可以知道的是，高斯分布的概率密度曲线在方差趋近于$0$时会极其集中，也就是值几乎都分布在均值周围，不妨说，分布$\lim_{\beta_t \to 0} \mathcal N(\mathbf x^{(t)};\mathbf x^{(t-1)}\sqrt {1-\beta_t},\mathbf I\beta_t)$下，$\mathbf x^{(t+1)} = \mathbf x^{(t)}$的概率接近$1$，此时，扩散部分几乎不带有不确定性(几乎是完全确定的)，数据在扩散过程中是趋近于完全不变化的，可以近似将扩散分布视为恒等函数

因此，分布$q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)}) = \prod_{t=0}^{T-1}q(\mathbf x^{(t)}|\mathbf x^{(t+1)})$是由扩散过程中各个几乎不改变数据的扩散分布(恒等函数)相乘得到，该分布也是几乎不带有随机性的，此时样本$\mathbf x^{(1\dots T)}$其实也几乎不带有随机性，它取为$(\mathbf x^{(0)},\dots, \mathbf x^{(0)})$的概率值几乎为$1$，因此，对从属于分布$q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})$的$\mathbf x^{(1\dots T)}$的样本求函数$f(\mathbf x^{(1\dots T)})$的期望不需要积分，只需要采样一个样本即可，因为该期望不带有随机性
)
## 2.4 Training
训练即最大化模型对数似然：
$$\begin{align}
L &= \int d\mathbf x^{(0)} q(\mathbf x^{(0)})\log p(\mathbf x^{(0)})\tag{10}\\
&=\int d\mathbf x^{(0)} q(\mathbf x^{(0)})\log \left(\int d\mathbf x^{(1\dots T)}q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\tag{11}
\end{align}$$
根据Jensen不等式，它的下界是：
$$L\ge \int d \mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)})\log\left(p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\tag{12}$$
(
$$\begin{aligned}
L &=\int d\mathbf x^{(0)} q(\mathbf x^{(0)})\log \left(\int d\mathbf x^{(1\dots T)}q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\\
&=\int d\mathbf x^{(0)} q(\mathbf x^{(0)})\log \mathbb E_{\mathbf x^{(1\dots T)}\sim q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})}\left[p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right]\\
&\ge \int d\mathbf x^{(0)} q(\mathbf x^{(0)})\mathbb E_{\mathbf x^{(1\dots T)}\sim q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})}\left[\log \left(p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\right]\\
&= \int d\mathbf x^{(0)} q(\mathbf x^{(0)}) d\mathbf x^{(1\dots T)}{ q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})}\left[\log \left(p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\right]\\
&= \int d\mathbf x^{(0)}  d\mathbf x^{(1\dots T)}q(\mathbf x^{(0)}){ q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})}\left[\log \left(p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\right]\\
&= \int  d\mathbf x^{(0\dots T)}{ q(\mathbf x^{(0\dots T)})}\left[\log \left(p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\right]\\
\end{aligned}$$
)
令该下界为$K$，进一步对它进行[[#B Log Likelihood Lower Bound|化简]]：
$$\begin{align}
L&\ge K\tag{13}\\
K&=-\sum_{t=2}^T \int d\mathbf x^{(0)}d\mathbf x^{(t)}q(\mathbf x^{(0)}, \mathbf x^{(t)})D_{KL}\left(q(\mathbf x^{(t-1)}|\mathbf x^{(t)},\mathbf x^{(0)})||p(\mathbf x^{(t-1)}|\mathbf x^{(t)})\right)\\
&+H_q(\mathbf X^{(T)}|\mathbf X^{(0)})-H_q(\mathbf X^{(1)}|\mathbf X^{(0)})-H_p(\mathbf X^{(T)})\tag{14}
\end{align}$$
其中的熵和KL散度都可以解析式地计算，该界限的推导和变分贝叶斯方法中对数似然界的推导类似

如2.3节所述，如果前向和反向轨迹是相同的，即对应于一个准静态过程，则公式13中的不等号就写成等号

训练时我们的目标是找到可以最大化对数似然下界的逆Markov变换(reverse Markov transitions)：
$$\hat p(\mathbf x^{(t-1)}|\mathbf x^{(t)}) = \arg\max_{p(\mathbf x^{(t-1)}|\mathbf x^{(t)})} K\tag{15}$$
对于高斯扩散和二项扩散具体的估计目标(targets of estimation)见Table App.1

此时，估计一个概率分布的任务就被简化为对一些函数进行回归估计(performing regression on the functions)，这些函数即设定了一序列高斯分布的均值和方差(set the mean and covariance of a sequence of Gaussians)，或设定了一序列伯努利实验的状态翻转概率
### 2.4.1 Setting the diffusion rate $\beta_t$
在前向轨迹中$\beta_t$的选择对于训练模型的性能至关重要，在自适应重要性采样(AIS)中，合适的中间分布调度(schedule of intermediate distributions)可以大幅提高对数分割函数估计的准确性(the accuracy of the log partition function estimate)(Grosse et al., 2013)，在热力学中，在平衡分布之间移动时所采取的调度决定了会损失多少自由能(Spinney & Ford, 2013; Jarzynski, 2011)

对于高斯扩散，我们通过对$K$的梯度上升学习$^2$前向扩散调度(forward diffusion schedule)$\beta_{2\dots T}$，第一步的方差$\beta_1$会被固定为一个小的常数以防止过拟合，来自$q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})$的样本对于$\beta_{1\dots T}$的依赖通过使用“固定噪声(frozen noise)”来表明(made explicit)——正如(Kingma & Welling, 2013)中所述，噪声被视为一个额外的辅助变量(additional auxiliary variable)，在计算$K$相对于参数的偏导数时保持不变(held constant)

对于二项扩散，离散的状态空间使得我们不能用固定噪声进行梯度上升，我们选择前向扩散调度$\beta_{1\dots T}$的依据是每一步扩散都消除原信号中的$\frac 1 T$的信息，因此得到的扩散率是$\beta_t = (T-t+1)^{-1}$
(
二项前向扩散的形式是：
$$\mathcal B(\mathbf x^{(t)};\mathbf x^{(t-1)}(1-\beta_t)+0.5\beta_t)$$
当$\beta_t = \frac 1 {T-t+1}$，有：
$$\mathcal B(\mathbf x^{(t)};\frac {T-t} {T-t+1}\mathbf x^{(t-1)}+0.5\beta_t)$$
可以看到每一步扩散的状态翻转概率中，原来的状态$\mathbf x^{(t-1)}$被消去了$\frac 1 T$
当$t=0$时，扩散核的状态翻转概率为：
$$\frac {T}{T+1}\mathbf x^{(0)}+0.5\beta_0$$
当$t=1$时，扩散核的状态翻转概率为：
$$\begin{align}
&\frac {T-1}{T}\mathbf x^{(1)} + 0.5\beta_1\\
=&\frac {T-1}{T}(\frac {T}{T+1}\mathbf x^{(0)}+0.5\beta_0)+\beta_1\\
=&\frac {T-1}{T}\frac {T}{T+1}\mathbf x^{(0)}+\frac {T}{T+1}0.5\beta_0+\beta_1\\
=&\frac {T-1}{T+1}\mathbf x^{(0)}+\frac {T}{T+1}0.5\beta_0+0.5\beta_1\\
\end{align}$$
因此可以知道，对于$\forall t$，其扩散核的状态翻转概率为：
$$\begin{align}
&\frac {T-t}{T-t+1}\mathbf x^{(t-1)} + 0.5\beta_t\\
=&\frac{T-t}{T-t+1}(\frac {T-t+1}{T-t+2}\mathbf x^{(t-2)}+0.5\beta_{t-1})+0.5\beta_t\\
=&\frac {T-t}{T-t+2}\mathbf x^{(t-2)}+\frac {T-t}{T-t+1}0.5\beta_{t-1}+0.5\beta_t\\
=&\cdots\\
=&\frac {T-t}{T+1}\mathbf x^{(0)}+0.5\sum_{k=0}^{t-1}\frac {T-k-1}{T-k}\beta_k+0.5\beta_t
\end{align}$$
可以看到，经过$t+1$步扩散，$\mathbf x^{(t+1)}$保持为$\mathbf x^{(0)}$的概率仅有$\frac {T-t}{T+1}$，换句话说，$\mathbf x^{(t+1)}$中仅有$\frac {T-t}{T+1}$的信息来自于$\mathbf x^{(0)}$，显然，信息减少了$(T+1)-(T-t) = t+1$
)

---
$^2$最近的实验表明，使用与二项扩散相同的固定(fixed)$\beta_t$调度同样有效
## 2.5 Multiplying Distributions, and Computing Posteriors
需要计算后验的任务，例如信号去噪(signal denoising)或遗失值推理(inference of missing values)需要将模型概率分布$p(\mathbf x^{(0)})$和第二个分布相乘，或者和一个有界的正函数(bounded positive function)$r(\mathbf x^{(0)}$)相乘，以产生一个新的分布$\tilde p (\mathbf x^{(0)}) \propto p(\mathbf x^{(0)})r(\mathbf x^{(0)})$

相乘分布对于许多技术是昂贵且困难的，包括了VAE、GSNs、NADEs，以及大多数图模型，但在扩散模型下则是十分直接的，因为第二个分布可以被视为对扩散过程的每一步的小扰动(pertubation)，或常常可以精确地乘入每一步扩散(be multiplied into each diffusion step)

Figure 3,5展示了使用扩散模型对自然图像进行去噪和修复，下面将介绍在扩散概率模型的背景下如何相乘概率分布
![[Deep Unsupervised learning using nonequilibrium thermodynamics-Fig3.png]]
![[Deep Unsupervised learning using nonequilibrium thermodynamics-Fig5.png]]
### 2.5.1 Modified marginal distributions
首先，为了计算$\tilde p(\mathbf x^{(0)})$，我们将每个中间分布(intermediate distributions)乘上对应的函数$r(\mathbf x^{(t)})$，我们在分布/Markov转换的符号上添加波浪号以表示它属于一个被相应修改的轨迹，故$\tilde p(\mathbf x^{(0\dots T)})$就表示被修改的反向轨迹(modified reverse trajectory)，它从分布$\tilde p(\mathbf x^{(T)}) = \frac 1{\tilde Z_T} p(\mathbf x^{(T)})r(\mathbf x^{(T)})$开始，并通过一系列中间分布前进(proceed)：
$$\tilde p(\mathbf x^{(t)} = \frac 1{\tilde Z_t}p(\mathbf x^{(t)})r(\mathbf x^{(t)})\tag{16}$$
其中$\tilde Z_t$为第$t$个中间分布对应的规范化常数(normalizing constan)
### 2.5.2 Modified diffusion steps
反向扩散过程的Markov核$p(\mathbf x^{(t)}|\mathbf x^{(t+1)})$遵循如下平衡条件(equilibrium condition)：
$$p(\mathbf x^{(t)}) = \int d\mathbf x^{(t+1)}p(\mathbf x^{(t)}|\mathbf x^{(t+1)})p(\mathbf x^{(t+1)})\tag{17}$$
我们希望被扰动的Markov核$\tilde p(\mathbf x^{(t)}|\mathbf x^{(t+1)})$遵循被扰动分布的平衡条件：
$$
\begin{align}
\tilde p(\mathbf x^{(t)}) &= \int d\mathbf x^{(t+1)}\tilde p(\mathbf x^{(t)} |\mathbf x^{(t+1)})\tilde p(\mathbf x^{(t+1)})\tag{18}\\
\frac {p(\mathbf x^{(t)})r(\mathbf x^{(t)})}{\tilde Z_t}&=\int d\mathbf x^{(t+1)}\tilde p(\mathbf x^{(t)}|\mathbf x^{(t+1)})\frac{p(\mathbf x^{(t+1)})r(\mathbf x^{(t+1)})}{\tilde Z_{t+1}}\tag{19}\\
p(\mathbf x^{(t)})&=\int d \mathbf x^{(t+1)}\tilde p(\mathbf x^{(t)}|\mathbf x^{(t+1)})\frac{\tilde Z_t r(\mathbf x^{(t+1)})}{\tilde Z_{t+1}r(\mathbf x^{(t)})}p(\mathbf x^{(t+1)})\tag{20}
\end{align}$$
公式20被满足时，仅当：
$$\tilde p(\mathbf x^{(t)}|\mathbf x^{(t+1)}) = p(\mathbf x^{(t)}|\mathbf x^{(t+1)})\frac {\tilde Z_{t+1}r(\mathbf x^{(t)})}{\tilde Z_t r(\mathbf x^{(t+1)})}\tag{21}$$
公式21可能没有对应于一个规范化的概率分布，因此我们选择$\tilde p(\mathbf x^{(t)}|\mathbf x^{(t+1)})$为其对应的规范化的概率分布：
$$\tilde p(\mathbf x^{(t)}|\mathbf x^{(t+1)}) = \frac {1}{\tilde Z_t(\mathbf x^{(t+1)})}p(\mathbf x^{(t)}|\mathbf x^{(t+1)})r(\mathbf x^{(t)})\tag{22}$$
其中$\tilde Z_t(\mathbf x^{(t+1)})$是规范化常数(normalization constant)

对于高斯分布，每个扩散步骤通常相对于$r(\mathbf x^{(t)})$非常尖锐(sharply peaked)，这是由于其小的方差，这意味着$\frac {r(\mathbf x^{(t)})}{r(\mathbf x^{(t+1)})}$可以被视为对$p(\mathbf x^{(t)}|\mathbf x^{(t+1)})$的小扰动(small perturbation)，对高斯分布的小扰动会影响均值，但不会影响规范化常数，所以在这种情况下，方程21和22是等价的(见[[#C Perturbed Gaussian Transition|附录C]])
### 2.5.3 Applying $r(\mathbf x^{(t)})$
如果$r(\mathbf x{(t)})$是足够平滑的(sufficiently smooth)，则它可以被视为是对逆扩散核$p(\mathbf x^{(t)}|\mathbf x^{(t+1)})$的一个小扰动(smalll perturbation)，在这种情况下，$\tilde p(\mathbf x^{(t)}|\mathbf x^{(t+1)})$将和$p(\mathbf x^{(t)}|\mathbf x^{(t+1)})$具有相同的函数形式(identical functional form)，但对于高斯扩散核，其均值会被扰动，对于二项扩散核，其翻转概率(flip rate)会被扰动

被扰动的扩散核的形式见Table App.1，关于高斯扩散核扰动的推导见[[#C Perturbed Gaussian Transition|附录C]]

如果$r(\mathbf x^{(t)})$可以和一个高斯分布(或二项分布)闭式地相乘，它就可以直接和逆扩散核$p(\mathbf x^{(t)}|\mathbf x^{(t+1)})$直接相乘，这适用于$r(\mathbf x^{(t)})$是由相对于某些坐标子集由狄拉克$\delta$函数组成时的情况(consists of a delta function for some subset of corrdinates)，例如Figure 5中的图像修复示例
### 2.5.4 Choosing $r(\mathbf x^{(t)})$
通常，$r(\mathbf x^{(t)})$应该在轨迹的过程中变化得较慢(change slowly over the course of the trajectory)，在本文的实验中，我们选择让它保持恒定：
$$
r(\mathbf x^{(t)}) = r(\mathbf x^{(0)})\tag{23}
$$
另一个方便的选择是$r(\mathbf x^{(t)}) = r(\mathbf x^{(0)})^{\frac {T-t}{T}}$，在该选择下，$r(\mathbf x^{(t)})$对于逆扩散轨迹的起始点是没有贡献的，这保证了从$\tilde p(\mathbf x^{(T)})$抽取逆向轨迹的初始样本(initial sample)仍然是直接的(remains straightforward)
## 2.6 Entropy of Reverse Process
因为前向过程是已知的，我们可以推导逆向轨迹中每一步的条件熵的上界和下界，进而得到对数似然的上界和下界：
$$
\begin{align}
H_q(\mathbf X^{(t)}|\mathbf X^{(t-1)}) + H_q(\mathbf X^{(t-1)}|\mathbf X^{(0)}) - H_q(\mathbf X^{(t)}|\mathbf X^{(0)})\\
\le H_q(\mathbf X^{(t-1)}|\mathbf X^{(t)})\le H_q(\mathbf X^{(t)}|\mathbf X^{(t-1)})\tag{24}
\end{align}
$$
其中，上界和下界都仅依赖于$q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})$，并且可以解析式地计算，推导见[[#A Conditional Entropy Bounds Derivation|附录A]]
# 3 Experiments
我们在多种连续数据集和二进制数据集上训练了扩散概率模型，然后我们展示了从训练好的模型中进行采样和修复缺失数据的结果(sampling from the trained model and inpainting of missing data)，并和其他技术进行了比较，
在所有情况下，目标函数和梯度都是使用Theano(Bergstra & Breuleux，2010)计算的。模型训练采用了SFO(Sohl-Dickstein et al.，2014)，CIFAR-10除外，CIFAR-10的结果使用了该算法的开源实现，并使用RMSprop进行优化，Table 1中报告了我们模型在所有数据集上提供的对数似然的下界，利用Blocks(van Merrienboer et al., 2015)的算法参考实现可在 https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models 上找到
![[Deep Unsupervised learning using nonequilibrium thermodynamics-Table 1.png]]


---
$^3$该论文的一个早期版本在CIFAR-10上报告了更高的对数似然界(higher log likelihood bounds)，这些结果来自于学习了CIFAR-10数据集的8bit量化像素值的模型，此处报告的CIFAR-10上的对数似然界则是在对数据进行了预处理后得到的结果，预处理即为数据添加均匀噪声以移除像素量化(add uniform noise to remove pixel quantization)，由(Thesis et al., 2015)推荐
## 3.1 Toy Problems
### 3.1.1 Swiss roll
我们使用扩散概率模型构造一个二维的瑞士卷分布，我们使用了径向基网络(radial basis function)(径向基网络使用径向基函数作为激活函数)来生成 $\mathbf f_\mu (\mathbf x^{(t)},t)$ 和 $\mathbf f_\Sigma (\mathbf x^{(t)}, t)$，如Figure 1所示，可以看到瑞士卷分布被成功学习，更多细节见[[#D.1.1 Swiss Roll|Appendix D.1.1]]
### 3.1.2 Binary heartbeat distribution
我们在一个简单的长度为20的二进制序列上训练扩散概率模型，其中1每5个时间间隔(every 5th time bin)出现一次，其余的间隔都是0，我们使用MLP来生成逆向轨迹的伯努利率$\mathbf f_b(\mathbf x^{(t)}, t)$(序列中1出现的概率)
我们得到的在真实分布下的对数似然是$\log_2(\frac 1 5) = -2.322$比特每序列(bits per sequence，用于表示每个序列中包含的信息量)，如Figure 2和Table 1所示，学习效果几乎完美，更多细节请见[[#D.1.2|Appendix D.1.2]]
![[Deep Unsupervised learning using nonequilibrium thermodynamics-Figure 2.png]]

## 3.2 Images
我们在一些图像数据集上训练扩散概率模型，这些实验共享的多尺度卷积架构(multi-scale convolutional architecture)(在不同的尺度上应用卷积运算，以捕获不同级别的图像特征)在[[#D.2.1|Appendix D.2.1]]中描述，并在Figure D.1中进行了说明
### 3.2.1 Datasets
**MNIST** 为了能够与之前在简单数据集上的工作进行直接比较，我们在MNIST数字数据集(LeCun & Cortes, 1998)上进行了训练，与(Bengio et al.，2012; Bengio & Thibodeau-Laufer, 2013; Goodfellow et al.，2014)的对数似然相对值在Table 2中给出，MNIST训练得到的模型生成的样本在Figure App.1中给出
![[Deep Unsupervised learning using nonequilibrium thermodynamics-Figure App.1.png]]
我们的训练算法提供了对数似然的渐近一致的下界(asymptotically consistent lower bound on the log likelihood)，然而，大多数之前报告的关于连续MNIST对数似然的结果依赖于基于Parzen窗口的估计，这些估计是从模型样本计算得出的，因此，为了进行比较，我们使用(Goodfellow et al. 2014)发布的Parzen窗口代码估计MNIST对数似然

**CIFAR-10** 我们对CIFAR-10数据集(Krizhevsky & Hinton, 2009)的训练图像拟合了一个概率模型，训练模型的部分样本如Figure 3所示

**Dead Leaf Images** 枯叶图像(Jeulin, 1997; Lee et al., 2001)由层叠遮挡的圆组成(layered occluding circles)，这些圆是从多个尺度上的幂律分布中采样的(drawn from a power law distribution over scales)，它们具有可分析(analytically tractable)的结构，同时也捕捉了自然图像的许多统计复杂性(statistical complexities of natural images)，因此为自然图像模型提供了一个很好的测试案例，正如Table 2和Figure 4所示，我们在枯叶数据集上达到了SOTA的性能
![[Deep Unsupervised learning using nonequilibrium thermodynamics-Fig4.png]]

**Bark Texture Images** 我们在(Lazebnik et al., 2005)的树皮纹理图像(T01-T04)上训练了一个概率模型，在这个数据集上，我们展示了可以直接从后验分布中进行评估或生成，例如在Figure 5中使用模型后验样本填补一大片缺失数据(inpanting a large region of missing data using a sample from the model posterior)
# 4 Conclusion
我们介绍了用于建模概率分布(modeling probability distribution)的一种新颖的算法，该算法能够实现精确抽样和概率评估(exact sampling and evaluation of probabilities)，该算法也在包括具有挑战性的自然图像数据集在内的各种玩具和真实数据集上展示了其有效性
对于这些测试，我们都使用了相似的基本算法，表明了我们的方法能够准确建模各种分布，大多数现有的密度估计技术(density estimatino techniques)必须牺牲建模能力(modeling power)以保持可解性和效率(stay tractable and efficient)，而且抽样或评估通常代价极高(expensive)
我们算法的核心在于估计一个马尔可夫扩散链的逆(the reversal of a Markov diffusion chain)，该链将数据映射到噪声分布，随着步数的增加，每一步扩散的逆分布(reversal distribution of each diffusion step)变得简单且易于估计，因此，结果是该算法可以学习拟合任何数据分布(fit to any data distribution)，且仍然易于训练、精确抽样和评估(remains tractable to train, _exactly_ sample from, and evaluate)，并且在该算法下，操作条件和后验分布是直接的(straightforward to manipulate conditional and poterior distributions)
# A Conditional Entropy Bounds Derivation
逆向轨迹中的一步的条件熵$H_q(\mathbf X^{(t-1)}|\mathbf X^{(t)})$是：
$$
\begin{align}
H_q(\mathbf X^{(t-1)},\mathbf X^{(t)}) &=H_q(\mathbf X^{(t)}, \mathbf X^{(t-1)})\tag{25}\\
H_q(\mathbf X^{(t-1)}|\mathbf X^{(t)})+H_q(\mathbf X^{(t)})&=H_q(\mathbf X^{(t)}|\mathbf X^{(t-1)})+H_q(\mathbf X^{(t-1)})\tag{26}\\
H_q(\mathbf X^{(t-1)}|\mathbf X^{(t)})&=H_q(\mathbf X^{(t)}|\mathbf X^{(t-1)})+H_q(\mathbf X^{(t-1)})-H_q(\mathbf X^{(t)})\tag{27}
\end{align}
$$
通过观察到$\pi(\mathbf y)$就是最大熵分布(maximum entropy distribution)，我们可以构造熵变的上界(upper bound on the entropy change)，对于二项分布，这一点无疑是成立的，同时，当训练数据的方差为1时，这一点对于高斯分布也成立，
因此，对于高斯情况，训练数据必须要被缩放到具有单位范数，以使得以下等式成立，但不需要被白化(whiten)(白化是数据预处理的一种方法，它不仅缩放数据到单位范数，还通常包括旋转变换以对角化数据的协方差矩阵，使得数据具有单位方差和零均值，因此这里的意思是数据不需要进一步被标准化处理以使其具有单位方差和零均值)，上界的推导如下：
$$
\begin{align}
H_q(\mathbf X^{(t)})&\ge H_q(\mathbf X^{(t-1)})\tag{28}\\
H_q(\mathbf X^{(t-1)})-H_q(\mathbf X^{(t)})&\le 0\tag{29}\\
H_q(\mathbf X^{(t-1)}|\mathbf X^{(t)})&\le H_q(\mathbf X^{(t)}|\mathbf X^{(t-1)})\tag{30}
\end{align}
$$
通过观察到马尔可夫链中的额外步骤不会增加关于链中初始状态的信息(information available about the initial state in the chain)，因此不会减少初始状态的条件熵(the conditional entropy of the initial state)，我们可以建立熵差的一个下界：
$$
\begin{align}
H_q(\mathbf X^{(0)}|\mathbf X^{(t)})&\ge H_q(\mathbf X^{(0)}|\mathbf X^{(t-1)})\\
H_q(\mathbf X^{(t-1)})-H_q(\mathbf X^{(t)})&\ge H_q(\mathbf X^{(0)}|\mathbf X^{(t-1)})+H_q(\mathbf X^{(t-1)})-H_q(\mathbf X^{(0)}|\mathbf X^{(t)})-H_q(\mathbf X^{(t)})\\
H_q(\mathbf X^{(t-1)})-H_q(\mathbf X^{(t)})&\ge H_q(\mathbf X^{(0)},\mathbf X^{(t-1)})-H_q(\mathbf X^{(0)}, \mathbf X^{(t)})\\
H_q(\mathbf X^{(t-1)})-H_q(\mathbf X^{(t)})&\ge H_q(\mathbf X^{(t-1)}|\mathbf X^{(0)})-H_q(\mathbf X^{(t)}|\mathbf X^{(0)})\\
H_q(\mathbf X^{(t-1)}|\mathbf X^{(t)})&\ge H_q(\mathbf X^{(t)}|\mathbf X^{(t-1)})+H_q(\mathbf X^{(t-1)}|\mathbf X^{(0)})-H_q(\mathbf X^{(t)}|\mathbf X^{(0)})
\end{align}
$$
结合二者，我们就得到了单个步骤的条件熵的上下界：
$$
\begin{align}
H_q(\mathbf X^{(t)}|\mathbf X^{(t-1)})\ge H_q(\mathbf X^{(t-1)}|\mathbf X^{(t)})\ge \\
H_q(\mathbf X^{(t)}|\mathbf X^{(t-1)}) + H_q(\mathbf X^{(t-1)}|\mathbf X^{(0)}) - H_q(\mathbf X^{(t)}|\mathbf X^{(0)})\\
\end{align}
$$
其中上界和下界都仅依赖于条件的前向轨迹$q(\mathbf x^{(1\dots T)}|\mathbf x^{(0)})$，并且可以被解析式地计算
# B Log Likelihood Lower Bound
对数似然的下界是：
$$\begin{align}
L&\ge K\tag{37}\\
K&=\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \left(p(\mathbf x^{(T)})\prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\tag{38}
\end{align}$$
## B.1 Entropy of $p(\mathbf X^{(T)})$
从$K$中分离出$p(\mathbf X^{(T)})$，将其重写：
$$\begin{align}
K&=\int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)})\left(\log p(\mathbf x^{(T)}) + \log \prod_{t=1}^T\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)\\
&=\int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)})\left( \sum_{t=1}^T\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\right)+\int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)})\log p(\mathbf x^{(T)})\\
&=\int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)}) \sum_{t=1}^T\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}+\int d\mathbf x^{(T)} q(\mathbf x^{(T)})\log p(\mathbf x^{(T)})\\
&=\int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)}) \sum_{t=1}^T\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}+\int d\mathbf x^{(T)} q(\mathbf x^{(T)})\log \pi(\mathbf x^{(T)})\\
\end{align}$$
在设计上，$\pi(\mathbf x^{(t)})$相对于扩散核的交叉熵都是相等的常数，且都等于$p(\mathbf x^{(T)})$的熵：
$$K = \sum_{t=1}^T\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}-H_p(\mathbf X^{(T)})\tag{43}$$

(
$$\begin{align}
\int q(\mathbf x^{(t)})\log\frac {1}{\pi(\mathbf x^{(t)})}d\mathbf x^{(t)}=\int p(\mathbf x^{(T)})\log \frac {1}{p(\mathbf x^{(T)})}d\mathbf x^{(T)}
\end{align}$$
交叉熵衡量了两个概率分布的差异，熵衡量了一个概率分布所含的信息量，也就是在任意时刻$t$，近似分布$\pi (\mathbf x^{(t)})$和真实分布$q(\mathbf x^{(t)})$之间的差异都等于噪声分布$p(\mathbf x^{(T)})$的信息量
)
## B.2 Remove the edge effect at $t=0$
为了避免边缘效应(edge effect)，我们将逆轨迹的最后一步设定为和对应的前向扩散步骤相同：
$$p(\mathbf x^{(0)}|\mathbf x^{(1)}) = q(\mathbf x^{(1)}|\mathbf x^{(0)}) \frac {\pi(\mathbf x^{(0)})}{\pi (\mathbf x^{(1)})}=T_{\pi}\left(\mathbf x^{(0)}|\mathbf x^{(1)};\beta_1\right)$$
借此，移除第一个时间步在似然下界中的贡献(the contribution of the first time-step in the sum)：
$$\begin{align}
K &= \sum_{t=2}^T\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\\
&+\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(0)}|\mathbf x^{(1)})}{q(\mathbf x^{(1)}|\mathbf x^{(0)})}-H_p(\mathbf X^{(T)})\\
&=\sum_{t=2}^T\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\\
&+\int d\mathbf x^{(0)}d\mathbf x^{(1)}q(\mathbf x^{(0)},\mathbf x^{(1)})\log \frac {q(\mathbf x^{(1)}|\mathbf x^{(0)})\pi(\mathbf x^{(0)})}{q(\mathbf x^{(1)}|\mathbf x^{(0)})\pi (\mathbf x^{(1)})}-H_p(\mathbf X^{(T)})\tag{45}\\
&=\sum_{t=2}^T\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\\
&+\int d\mathbf x^{(0)}d\mathbf x^{(1)}q(\mathbf x^{(0)},\mathbf x^{(1)})\log \frac {\pi(\mathbf x^{(0)})}{\pi (\mathbf x^{(1)})}-H_p(\mathbf X^{(T)})\\
&=\sum_{t=2}^T\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\\
&+\int d\mathbf x^{(0)}d\mathbf x^{(1)}q(\mathbf x^{(0)},\mathbf x^{(1)})\log {\pi(\mathbf x^{(0)})}-\int d\mathbf x^{(0)}d\mathbf x^{(1)}q(\mathbf x^{(0)},\mathbf x^{(1)})\log{\pi (\mathbf x^{(1)})}\\&-H_p(\mathbf X^{(T)})\\
&=\sum_{t=2}^T\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})}\\
&+\int d\mathbf x^{(0)}q(\mathbf x^{(0)})\log {\pi(\mathbf x^{(0)})}-\int d\mathbf x^{(1)}q(\mathbf x^{(1)})\log{\pi (\mathbf x^{(1)})}\\&-H_p(\mathbf X^{(T)})\\
&=\sum_{t=2}^T\int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)})} -H_p(\mathbf X^{(T)})\tag{46}
\end{align}$$
其中的化简再次用到了“在设计上，$\pi(\mathbf x^{(t)})$相对于扩散核的交叉熵都是相等的常数，且都等于$p(\mathbf x^{(T)})$的熵”这一结论，即对于所有的$t$，都满足$-\int d\mathbf x^{(t)} q(\mathbf x^{(t)}) \log \pi (\mathbf x^{(t)}) = H_p(\mathbf X^{(T)})$
## B.3 Rewrite in terms of posterior $q(\mathbf x^{(t-1)}|\mathbf x^{(0)})$
因为前向轨迹是一个Markov过程：
$$K = \sum_{t=2}^T \int d\mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t)}|\mathbf x^{(t-1)},\mathbf x^{(0)})}-H_p(\mathbf X^{(T)})\tag{47}$$
使用贝叶斯规则，我们可以使用前向轨迹的后验和边际来重写该似然下界：
$$K = \sum_{t=2}^T \int d \mathbf x^{(0\dots T)}q(\mathbf x^{(0\dots T)})\log \left[\frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t-1)}|\mathbf x^{(t)},\mathbf x^{(0)})}\frac {q(\mathbf x^{(t-1)}|\mathbf x^{(0)})}{q(\mathbf x^{(t)}|\mathbf x^{(0)})}\right]-H_p(\mathbf X^{(T)})\tag{48}$$
## B.4 Rewrite in terms of KL divergences and entropies
我们可以发现一些项可以写成条件熵(conditional entropies)：
$$\begin{align}
K&=\sum_{t=2}^T \int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t-1)}|\mathbf x^{(t)},\mathbf x^{(0)})} \\
&+ \sum_{t=2}^T\int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)})\log\frac {q(\mathbf x^{(t-1)}|\mathbf x^{(0)})}{q(\mathbf x^{(t)}|\mathbf x^{(0)})}-H_p(\mathbf X^{(T)})\\
&=\sum_{t=2}^T \int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t-1)}|\mathbf x^{(t)},\mathbf x^{(0)})}\\
&+\sum_{t=2}^T\left[H_q(\mathbf X^{(t)}|\mathbf X^{(0)})-H_q(\mathbf X^{(t-1)}|\mathbf X^{(0)})\right]-H_p(\mathbf X^{(T)})\tag{49}\\
&=\sum_{t=2}^T \int d\mathbf x^{(0\dots T)} q(\mathbf x^{(0\dots T)})\log \frac {p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}{q(\mathbf x^{(t-1)}|\mathbf x^{(t)},\mathbf x^{(0)})}\\
&+H_q(\mathbf X^{(T)}|\mathbf X^{(0)})-H_q(\mathbf X^{(1)}|\mathbf X^{(0)})-H_p(\mathbf X^{(T)})\tag{50}\\
\end{align}$$
最后，我们将概率分布的对数比(log ratio of probability distributions)转换为KL散度：
$$\begin{align}
K&=-\sum_{k=2}^T \int d\mathbf x^{(0)}d\mathbf x^{(t-1)}d\mathbf x^{(t)}q(\mathbf x^{(0)},\mathbf x^{(t-1)},\mathbf x^{(t)})\log\frac{q(\mathbf x^{(t-1)}|\mathbf x^{(t)}, \mathbf x^{(0)})}{p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}\\
&+H_q(\mathbf X^{(T)}|\mathbf X^{(0)})-H_q(\mathbf X^{(1)}|\mathbf X^{(0)})-H_p(\mathbf X^{(T)})\\
&=-\sum_{k=2}^T \int d\mathbf x^{(0)}d\mathbf x^{(t-1)}d\mathbf x^{(t)}q(\mathbf x^{(0)},\mathbf x^{(t-1)},\mathbf x^{(t)})\log\frac{q(\mathbf x^{(t-1)}|\mathbf x^{(t)}, \mathbf x^{(0)})}{p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}\\
&+H_q(\mathbf X^{(T)}|\mathbf X^{(0)})-H_q(\mathbf X^{(1)}|\mathbf X^{(0)})-H_p(\mathbf X^{(T)})\\
&=-\sum_{k=2}^T \int d\mathbf x^{(0)}d\mathbf x^{(t-1)}d\mathbf x^{(t)}q(\mathbf x^{(0)},\mathbf x^{(t)})q(\mathbf x^{(t-1)}|\mathbf x^{(t)},\mathbf x^{(0)})\log\frac{q(\mathbf x^{(t-1)}|\mathbf x^{(t)}, \mathbf x^{(0)})}{p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}\\
&+H_q(\mathbf X^{(T)}|\mathbf X^{(0)})-H_q(\mathbf X^{(1)}|\mathbf X^{(0)})-H_p(\mathbf X^{(T)})\\
&=-\sum_{k=2}^T \int d\mathbf x^{(0)}d\mathbf x^{(t)}q(\mathbf x^{(0)},\mathbf x^{(t)})\left(d\mathbf x^{(t-1)}q(\mathbf x^{(t-1)}|\mathbf x^{(t)},\mathbf x^{(0)})\log\frac{q(\mathbf x^{(t-1)}|\mathbf x^{(t)}, \mathbf x^{(0)})}{p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}\right)\\
&+H_q(\mathbf X^{(T)}|\mathbf X^{(0)})-H_q(\mathbf X^{(1)}|\mathbf X^{(0)})-H_p(\mathbf X^{(T)})\\
&=-\sum_{k=2}^T \int d\mathbf x^{(0)}d\mathbf x^{(t)}q(\mathbf x^{(0)},\mathbf x^{(t)})D_{KL}\left(\left.{q(\mathbf x^{(t-1)}|\mathbf x^{(t)}, \mathbf x^{(0)})}\right|\left|{p(\mathbf x^{(t-1)}|\mathbf x^{(t)})}\right.\right)\\
&+H_q(\mathbf X^{(T)}|\mathbf X^{(0)})-H_q(\mathbf X^{(1)}|\mathbf X^{(0)})-H_p(\mathbf X^{(T)})
\end{align}$$
注意这些熵可以被解析式地计算，并且KL散度也可以在给定$\mathbf x^{(0)}$和$\mathbf x^{(t)}$时被解析式地计算
# C Perturbed Gaussian Transition
我们希望计算$\tilde p(\mathbf x^{(t-1)} |\mathbf x^{(t)})$，为了符号简单性，我们令$\mu = \mathbf f_{\mu}(\mathbf x^{(t)},t),\Sigma = \mathbf f_{\Sigma}(\mathbf x^{(t)},t)$，以及令$\mathbf y = \mathbf x^{(t-1)}$，我们有：
$$\begin{align}
\tilde p(\mathbf y | \mathbf x^{(t)}) &\propto p(\mathbf y | \mathbf x^{(t)})r(\mathbf y)\tag{52}\\
&=\mathcal N(\mathbf y;\mu,\Sigma)r(\mathbf y)\tag{53}
\end{align}$$
我们可以以能量函数的形式重写该式(in terms of energy functions)，其中$E_r(\mathbf y) = -\log r(\mathbf y)$：
$$\begin{align}
\tilde p(\mathbf y|\mathbf x^{(t)}) &\propto \exp[-E(\mathbf y)]\tag{54}\\
E(\mathbf y)&=\frac 1 2 (\mathbf y-\mu)^T\Sigma^{-1}(\mathbf y-\mu) + E_r(\mathbf y)\tag{55}
\end{align}$$
如果$E_r(\mathbf y)$相对于$\frac 1 2 (\mathbf y -\mu)^T\Sigma^{-1}(\mathbf y - \mu)$是平滑的，则我们可以使用它围绕$\mu$的Taylor展开对其近似，一个充分条件是$E_r(\mathbf y)$的海森矩阵(Hessian)的特征值在任何地方都比$\Sigma^{-1}$的特征值小得多(everywhere much smaller magnitude than)，则我们有：
$$E_r(\mathbf y) \approx E_r(\mu) + (\mathbf y  -\mu)^T\mathbf g\tag{56}$$
其中$\mathbf g = \left.\frac {\partial E_r(\mathbf y')}{\partial \mathbf y'}\right|_{\mathbf y'=\mu}$
(
$E_r(\mathbf y)$是一个定义在$\mathbb R^n$上的能量函数，$\mu$是$\mathbf y$的某个固定点(均值)，$\Sigma$是协防差矩阵，且$\Sigma^{-1}$存在，
如果$E_r(\mathbf y)$在$\mu$附近平滑，则$E_r(\mathbf y)$在$\mu$点附近是局部二次可微的，则$E_r(\mathbf y$)可以展开为：
$$E_r(y) = E_r(\mu) +\nabla E_r(\mu)^T(\mathbf y - \mu) + \frac 1 2(\mathbf y - \mu)^TH_r(\mu)(\mathbf y - \mu) + R_2(\mathbf y)$$
其中$H_r(\mu)$是$E_r(\mathbf y)$在$\mu$点的海森矩阵，$R_2(\mathbf y)$是二阶余项

如果$E_r(\mathbf y)$的海森矩阵所有特征值的绝对值都远小于$\Sigma^{-1}$的特征值的绝对值，说明$\frac 1 2 (\mathbf y - \mu)^TH_r(\mu)(\mathbf y - \mu)$相对于$\frac 1 2 (\mathbf y - \mu)\Sigma^{-1}(\mathbf y - \mu)$是一个小量，这意味着$E_r(\mathbf y)$的二阶项对于$E(\mathbf y)$的总体行为影响不大，此时，我们可以忽略二阶项和余项$R_2(\mathbf y)$，用一阶展开式近似$E_r(\mathbf y)$
)

我们将$E_r(\mathbf y)$的近似放入完整能量函数：
$$\begin{align}
E(\mathbf y) &\approx \frac 1 2(\mathbf y - \mu)^T \Sigma^{-1}(\mathbf y - \mu) + (\mathbf y-\mu)^T\mathbf g+ \text{constant}\tag{57}\\
&=\frac 1 2\mathbf y^T\Sigma^{-1}\mathbf y-\frac 1 2 \mathbf y^T\Sigma^{-1}\mu-\frac 1 2\mu^T\Sigma^{-1}\mathbf y+\frac 1 2\mu \Sigma^{-1}\mu\\
&+\mathbf y^T\mathbf g - \mu\mathbf g + \text{constant}\\
&=\frac 1 2\mathbf y^T\Sigma^{-1}\mathbf y-\frac 1 2 \mathbf y^T\Sigma^{-1}\mu-\frac 1 2 \mu^T\Sigma^{-1}\mathbf y+\text{constant}\\
&+\mathbf y^T\mathbf g-\text{constant}+\text{constant}\\
&=\frac 1 2 \mathbf y^T\Sigma^{-1}\mathbf y - \frac 1 2\mathbf y^T\Sigma^{-1}\mu - \frac 1 2\mu^T\Sigma^{-1}\mathbf y \\&+\frac 1 2\mathbf y^T\mathbf g +\frac 1 2 \mathbf g^T\mathbf y+\text{constant}\\
&=\frac 1 2 \mathbf y^T\Sigma^{-1}\mathbf y - \frac 1 2\mathbf y^T\Sigma^{-1}\mu - \frac 1 2\mu^T\Sigma^{-1}\mathbf y \\&+\frac 1 2\mathbf y^T\Sigma^{-1}\Sigma\mathbf g + \frac 1 2 \mathbf g^T\Sigma\Sigma^{-1}\mathbf y + \text{constant}\tag{58}\\
&=\frac 1 2(\mathbf y - \mu + \Sigma\mathbf g)^T\Sigma^{-1}(\mathbf y - \mu+\Sigma\mathbf g) + \text{constant}\tag{59}
\end{align}$$
这对应于一个高斯分布：
$$\tilde p(\mathbf y|\mathbf x^{(t)}) \approx \mathcal N(\mathbf y;\mu - \Sigma\mathbf g, \Sigma)\tag{60}$$
我们将其替换为原来的形式，得到：
$$\begin{align}
&\tilde p(\mathbf x^{(t-1)}|\mathbf x^{(t)})\\
\approx &\mathcal N\left(\mathcal x^{(t-1)};\mathbf f_\mu(\mathbf x^{(t)},t)+\mathbf f_{\Sigma}(\mathbf x^{(t)},t)\left.\frac {\partial \log r(\mathbf x^{(t-1)'})}{\partial \mathbf x^{(t-1)'}}\right|_{\mathbf x^{(t-1)'}=f_{\mu}(\mathbf x^{(t)},t)},\mathbf f_{\Sigma}(\mathbf x^{(t)},t)\right)
\end{align}\tag{61}$$
# D Experimental Details
## D.1 Toy Problems
### D.1.1 Swiss Roll
我们使用扩散模型构建二维瑞士卷分布的概率模型，其中生成式模型$p(\mathbf x^{(0\dots T)})$由40个时间步(time steps)的高斯扩散核组成，以单位协方差高斯分布初始化(initilaized at an identity-covariance Gaussian distribution)，我们使用一个隐藏层和16个隐藏单元的(规范化)径向基函数网络，用来生成均值函数$\mathbf f_\mu (\mathbf x^{(t)},t)$和对角协方差函数$\mathbf f_\Sigma(\mathbf x^{(t)},t)$，用于逆向轨迹，每个函数的顶层，即读出层(read-out)，针对每个时间步独立学习，但对于所有其他层的权重在所有时间步和两个函数之间共享，$\mathbf f_\Sigma(\mathbf x^{(t)},t)$的顶层输出会传递给Sigmoid函数以限制输出在0和1之间，如Figure 1所示，瑞士卷分布被成功学习
### D.1.2 Binary Heartbeat Distributiion
我们在长度为20的简单二进制序列上训练扩散概率模型，其中1每5个时间间隔出现一次，其余的间隔都是0，我们的生成式模型由2000个时间步的二项扩散组成，以和数据具有相同平均活动(the same mean activity as the data)的独立二项分布初始化，也就是$p(\mathbf x_i^{(T)}=1)=0.2$
我们使用具有Sigmoid非线性、20个输入单元和三层各有50个单元的隐藏层的多层感知器被训练来生成逆向轨迹的伯努利率$\mathbf f_b(\mathbf x^{(t)},t)$，顶层，即读出层，针对每个时间步独立学习，但所有其他层的权重在所有时间步共享，顶层输出通过一个Sigmoid函数传递，以限制它在0和1之间
如Figure 2所示，心跳分布被成功学习
## D.2 Images
### D.2.1 Architecture
![[Deep Unsupervised learning using nonequilibrium thermodynamics-FigD.1.png]]
**Readout** 在所有的情况下，我们使用卷积网络为每个图像像素$i$生成输出向量$\mathbf y_i \in \mathcal R^{2J}$，$\mathbf y_i$中的项(entries)被分成两个大小相同的子集$\mathbf y^\mu$和$\mathbf y^\Sigma$

**Temporal Dependence** 卷积输出$\mathbf y^\mu$被用作各像素的加权参数(per-pixel weighting coefficients)，用于在时间依赖的“凸起”函数上求和(in a sum over time-dependent "bump" functions)，为每个像素生成输出$\mathbf z_i^\mu \in \mathcal R$：
$$
\mathbf z_i^\mu = \sum_{j=1}^J\mathbf y_{ij}^\mu g_j(t)\tag{62}
$$
“凸起”函数写为：
$$
g_j(t) = \frac {\exp\left(-\frac 1 {2w^2}(t-\tau_j)^2\right)}{\sum_{k=1}^J\exp\left(-\frac 1 {2w^2}(t-\tau_k)^2\right)}\tag{63}
$$
其中$\tau_j \in (0, T)$为凸起中心(bump center)，$w$是凸起中心之间的间隔(spacing between bump centers)，$\mathbf z^\Sigma$以相同的方式生成，使用的是$\mathbf y^\Sigma$

对于所有的图像实验，使用的时间步数量都是$T=1000$，除了树皮数据集使用的是$T=500$

**Mean and Variance** 最后，这些输出会被结合以为每个像素$i$产生一个扩散均值和扩散方差预测(diffusion mean and variance prediction)：
$$
\begin{align}
\Sigma_{ii}&= \sigma(z_i^\Sigma + \sigma^{-1}(\beta_t))\tag{64}\\
\mu_i &= (x_i-z_i^\mu)(1-\Sigma_{ii})+z_i^\mu\tag{65}
\end{align}
$$
其中，$\Sigma$和$\mu$都是以对前向扩散核$T_\pi(\mathbf x^{(t)}|\mathbf x^{(t-1)};\beta_t)$的微扰来参数化的(parameterized as a pertubation around the forward diffusion kernel)， 而$z_i^\mu$是平衡分布的均值，可以从多次应用$p(\mathbf x^{(t-1)}| \mathbf x^{(t)})$所得到，$\Sigma$被限制为对角矩阵

**Multi-Scale Convolution** 我们希望实现通常通过池化网络(pooling networks)实现的目标——具体来说，我们希望发现并利用训练数据中的长距离和多尺度依赖性(long-range and multi-scale dependencies in training data)，由于网络输出是每个像素的系数向量(a vector of coefficients for every pixel)，我们可以生成全分辨率而非下采样的特征图(full resolution rather than down-sampled feature map)，因此，我们定义了多尺度卷积层，包括以下步骤：
1. 执行均值池化(mean pooling)以将图像下采样到多个尺度(downsample the image to multiple scales)，下采样以2的幂次进行
2. 在每个尺度上执行卷积(at each scale)
3. 将所有尺度上采样到全分辨率(upscale all scales to full resolution)，并叠加结果图像(sum the resulting images)
4. 执行逐点(pointwise)非线性变换，即一个软(soft)ReLU($\log[1+\exp(\cdot)]$)
前三个线性操作的组合类似于使用多尺度卷积核进行卷积，直到由上采样引入的阻塞伪影(up to blocking artifacts introduced by upsampling)，这种实现多尺度卷积的方法在(Barron et al.，2013)中有所描述

**Dense Layers** 密集层(作用于整个图像向量)和卷积核宽度为1的卷积层()分别作用于每个像素的特征向量)具有相同的形式，它们由线性变换组成，紧跟一个tanh非线性激活函数
