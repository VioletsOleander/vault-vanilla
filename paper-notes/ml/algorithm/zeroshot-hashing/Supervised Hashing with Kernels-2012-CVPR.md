# Abstract
# 1 Introduction
作者认为在完全利用监督信息的情况下，有监督哈希相较于无监督哈希可以取得更高的检索精度

作者认为先前工作的昂贵训练开销来源于它们使用的过于复杂的哈希目标函数

作者提出的目标函数利用了汉明距离和编码内积之间的代数等价性，巧妙地通过控制编码内积，以隐式且高效地对汉明距离进行优化
作者利用编码内积的可分离性质，设计了高效的贪心算法，按比特顺序地解目标哈希函数，对于线性不可分离的数据，则采用目标哈希函数的核形式
因此称该方法为Kernel-Based Supervised Hashing(KSH)
# 2 Kernel-Based Supervised Hashing
## 2.1 Hash Function with Kernels
给定数据集$\mathcal X = \{\symbfit x_1, \cdots, \symbfit x_n\}\subset \mathbb R^d$，哈希目标是寻找一组适当的哈希函数$h: \mathbb R^d\to \{1,-1\}^1$，其中的每一个都负责单个哈希比特的生成

使用核函数$\mathcal K: \mathbb R^d \times \mathbb R^d \to \mathbb R$以构建这样的哈希函数，核技巧在经验上和理论上被证明可以解决多数线性不可分的数据
利用核函数定义预测函数$f:\mathbb R^d \to \mathbb R$如下
$$f(\symbfit x) = \sum_{j=1}^m\mathcal K(\symbfit x_{(j)},\symbfit x)a_j-b\tag{1}$$
其中$\symbfit x_{(1)},\cdots, \symbfit x_{(m)}$为$m$个从$\mathcal X$中均匀采样的样本，$a_j\in\mathbb R$为系数，$b$为偏置，
$m$是固定的常数，满足$m \ll n$以保持哈希效率

基于$f$构造单个哈希比特的哈希函数$h(\symbfit x) = sgn(f(\symbfit x))$

哈希函数的一个重要设计准则是生成的哈希比特要尽可能包含多的信息，即需要满足$\sum_{i=1}^n h(\symbfit x_i) = 0$的平衡的哈希函数，在本文语境下，即要满足$b$是$\{\sum_{j=1}^m\mathcal K(\symbfit x_{(j)},\symbfit x_i)a_j\}_{i=1}^n$的平均值，即$b = \sum_{i=1}^n\sum_{j=1}^m\mathcal K(\symbfit x_{(j)},\symbfit x_i)a_j/n$，将等式(1)中的$b$替换，得到
$$f(\symbfit x)=\sum_{j=1}^m\left(\mathcal K(\symbfit x_{(j)},\symbfit x)-\frac 1 n\sum_{i=1}^n\mathcal K(\symbfit x_{(j)},\symbfit x_i)\right)a_j=\symbfit a^T{\bar {\symbfit k}(\symbfit x)}\tag{2}$$
其中$\symbfit a = [a_1,\cdots a_m]^T$，$\bar {\symbfit k}: \mathbb R^d \to \mathbb R^m$为一个向量映射，定义为
$$\bar{\symbfit k}(\symbfit x)=[\mathcal K(\symbfit x_{(1)},\symbfit x)-u_1,\cdots,\mathcal K(\symbfit x_{(m)},\symbfit x)-u_m]^T\tag{3}$$
其中$u_j = \sum_{i=1}^n \mathcal K(\symbfit x_{(j)},\symbfit x_i)/n$，可以预先计算

KLSH中，系数向量$\symbfit a$是从高斯分布中随机选取的，而由于哈希函数实际上由$\symbfit a$完全决定，因此本文中将利用有监督的信息以学习$\symbfit a$
## 2.2 Manipulating Code Inner Product
若需要$r$哈希比特，我们需要找到$r$个系数向量$\symbfit a_1,\cdots, \symbfit a_r$以构建$r$个哈希函数$\{h_k(\symbfit x) = sgn(\symbfit a_k^T \bar {\symbfit k}(\symbfit x))\}_{k=1}^r$

有监督信息一般以成对标签的形式给出，
集合$\mathcal M$包含了相似样本对，用标签$1$标记，
集合$\mathcal C$包含了不相似样本对，用标签$-1$标记
假设前$l$($m< l \ll n)$个样本$\mathcal X_l = \{\symbfit x_1,\cdots,\symbfit x_l\}$具有有监督信息，定义一个标签矩阵$S\in \mathbb R^{l\times l}$显式记录$\mathcal X_l$中的成对关系
$$S_{ij}=\begin{cases}
1&(\symbfit x_i,\symbfit x_j)\in\mathcal M\\
-1&(\symbfit x_i,\symbfit x_j)\in\mathcal C\\
0&otherwise
\end{cases}\tag{4}$$
标签为$0$表示相似/不相似关系未知或不确定

我们希望标签为$S_{ij} =1$的样本对之间的汉明距离是最小的，即$0$，
而标签为$S_{ij} = -1$的样本对之间的汉明距离是最大的，即哈希比特数$r$

汉明距离可以如下定义
$\mathcal D_h(\symbfit x_i,\symbfit x_j) = |\{k|h_k(\symbfit x_i)\ne h_k(\symbfit x_j),1\le k \le r\}|$
直接优化汉明距离较难，作者提出用内积代替

**Code Inner Product vs. Hamming Distances**
样本$\symbfit x$的$r$位哈希码写为$code_r(\symbfit x) = [h_1(\symbfit x),\cdots, h_r(\symbfit x)]\in \{1,-1\}^{1\times r}$，两个样本的哈希码的内积可以写为
$$\begin{align}
&code_r(\symbfit x_i)\circ code_r(\symbfit x_j)\\
=&|\{k|h_k(\symbfit x_i)= h_k(\symbfit x_j),1\le k \le r\}|\\
&-|\{k|h_k(\symbfit x_i)\ne h_k(\symbfit x_j),1\le k \le r\}|\\
=&r-2|\{k|h_k(\symbfit x_i)\ne h_k(\symbfit x_j),1\le k \le r\}|\\
=&r-2\mathcal D_h(\symbfit x_i,\symbfit x_j)
\end{align}\tag{5}$$
该式表明哈希码内积和汉明距离是一对一的，因此优化内积是等价优化汉明距离的

考虑$code_r(\symbfit x_i)\circ code_r(\symbfit x_j)\in[-r,r]$，而$S_{ij}\in[-1,1]$，
我们让$code_r(\symbfit x_i)\circ code_r(\symbfit x_j)\in[-r,r] / r$去拟合$S_{ij}$，
当$code_r(\symbfit x_i)\circ code_r(\symbfit x_j)\in[-r,r] / r = S_{ij} = 1$时，$\mathcal D_h(\symbfit x_i,\symbfit x_j) = 0$
当$code_r(\symbfit x_i)\circ code_r(\symbfit x_j)\in[-r,r] / r = S_{ij} = -1$时，$\mathcal D_h(\symbfit x_i,\symbfit x_j) = r$
因此这样拟合是合理的

我们提出最小二乘风格的目标函数$\mathcal Q$学习已标记数据$\mathcal X_l$的哈希码
$$\min_{H_l\in\{\pm 1\}^{l\times r}}\mathcal Q = \|\frac 1 rH_l H_l^T-S\|_F^2\tag{6}$$
其中$H_l = \begin{bmatrix}code_r(\symbfit x_1)\\\vdots\\code_r(\symbfit x_l)\end{bmatrix}$为$\mathcal X_l$的编码矩阵

令$h_k(\symbfit x) = sgn(f_k(\symbfit x)) = sgn(\symbfit a^T{\bar {\symbfit k}(\symbfit x)})=sgn({\bar {\symbfit k}^T(\symbfit x)}\symbfit a)$，$H_l$可以写为
$$H_l=\begin{bmatrix}h_1(\symbfit x_1),\cdots, h_r(\symbfit x_1)\\\cdots \\h_1(\symbfit x_l),\cdots, h_r(\symbfit x_l)\\
\end{bmatrix}=sgn(\bar K_l A)\tag{7}$$
其中$\bar K_l = [\bar {\symbfit k}(\symbfit x_1),\cdots, \bar {\symbfit k}(\symbfit x_l)]^T\in \mathbb R^{l\times m}$，$A = [\symbfit a_1,\cdots,\symbfit a_r]\in \mathbb R^{m\times r}$

将公式(6)中的$H_l$替换，得到目标函数的解析形式
$$\min_{A\in\mathbb R^{m\times r}}\mathcal Q(A) = \|\frac 1 r sgn(\bar K_lA)(sgn(\bar K_lA))^T-S\|_F^2\tag{8}$$

## 2.3 Greedy Optimization
利用矩阵乘法的可分离性质，重写$\mathcal Q$
$$\min_A\|\sum_{k=1}^r sgn(\bar K_l\symbfit a_k)(sgn(\bar K_l \symbfit a_k))^T - rS\|_F^2\tag{9}$$
其中$r$个向量$\symbfit a_k$，每个都决定了一个哈希函数，在和式中分离，
这启发我们用贪婪算法顺序求解$\symbfit a_k$，任意时刻，我们仅在之前解出的向量$\symbfit a_1^*,\cdots, \symbfit a_{k-1}^*$的条件下求解单个向量$\symbfit a_k$

>补充一个知识点
>有$1\times n$的向量$\symbfit v$，$n\times n$的矩阵$M$，则$$\begin{align}
&tr(\symbfit v\symbfit v^T M)\\
=&tr(\begin{bmatrix}v_1\\ \vdots \\ v_n \end{bmatrix}(\symbfit v^TM))\\
=&\sum_{i=1}^n v_i(\symbfit v^TM)_i
\end{align}$$
>考虑$$\begin{align}
&\symbfit v^T M \symbfit v\\
=&(\symbfit v^TM)\begin{bmatrix}v_1\\ \vdots \\v_n\end{bmatrix}\\
=&\sum_{i=1}^n(\symbfit v^TM)_i v_i
\end{align}$$
>因此有$$tr(\symbfit v\symbfit v^TM) = \symbfit v^TM\symbfit v$$

定义残差矩阵$R_{k-1} = rS - \sum_{t=1}^{k-1} sgn(\bar K_l \symbfit a_t^*)(sgn(\bar K_l \symbfit a_t^*))^T$，注意$R_0 = rS$
则$\symbfit a_k$可以通过最小化下式求解
$$\begin{align}
&\|sgn(\bar K_l\symbfit a_k)(sgn(\bar K_l\symbfit a_k))^T-R_{k-1}\|_F^2\\
=&\|\symbfit h_k\symbfit h_k^T-R_{k-1}\|_F^2\\
=&tr\left((\symbfit h_k\symbfit h_k^T-R_{k-1})(\symbfit h_k\symbfit h_k^T - R_{k-1})\right)\\
=&tr(\symbfit h_k\symbfit h_k^T\symbfit h_k\symbfit h_k^T - \symbfit h_k\symbfit h_k^TR_{k-1}-R_{k-1}\symbfit h_k\symbfit h_k^T + R_{k-1}R_{k-1} )\\
=&\symbfit h_k^T\symbfit h_ktr(\symbfit h_k\symbfit h_k^T)-2tr(\symbfit h_k\symbfit h_k^TR_{k-1}) + tr(R_{k-1}^2)\\
=&(\symbfit h_k^T\symbfit h_k)^2-2\symbfit h_k^TR_{k-1}\symbfit h_k+tr(R_{k-1}^2)\\
=&l^2-2\symbfit h_k^TR_{k-1}\symbfit h_k+tr(R_{k-1}^2)\\
=&-2\symbfit h_k^TR_{k-1}\symbfit h_k+const
\end{align}\tag{10}$$
丢弃常数项，得到
$$\begin{align}
g(\symbfit a_k) &= -\symbfit h_k^TR_{k-1}\symbfit h_k\\
&=-(sgn(\bar K_l \symbfit a_k))^TR_{k-1}sgn(\bar K_l\symbfit a_k)
\end{align}\tag{11}$$
而由于公式(10)是一定非负的，因此$g(\symbfit a_k)$是有下界的，但$g$不是平滑的也不是凸的，因此只能近似优化$g$

### 2.3.1 Spectral Relaxation
应用谱松弛技巧，丢弃$g$中的$sgn$函数，得到受限制的二次问题
$$\begin{align}
&\max_{\symbfit a_k}(\bar K_l\symbfit a_k)^TR_{k-1}(\bar K_l\symbfit a_k)\\
&s.t.(\bar K_l\symbfit a_k)^T(\bar K_l\symbfit a_k) = l
\end{align}\tag{12}$$
其中限制$(\bar K_l\symbfit a_k)^T(\bar K_l\symbfit a_k) = l$的意图在于让向量$\bar K_l \symbfit a_k$中的元素大致落在$[-1,1]$的范围内，以使得对松弛问题(12)的解和原问题(11)的解的范围是相似的

公式(12)是一个标准的广义的特征值问题$\bar K_l^T R_{k-1} \bar K_l \symbfit a = \lambda \bar K_l^T \bar K_l \symbfit a$
(或者写为$R_{k-1}\bar K_l \symbfit a = \lambda \bar K_l \symbfit a$)
因此解$\symbfit a_k$即为该特征值问题最大的特征值对应的特征向量，注意解$\symbfit a_k$要进行适当的放缩，放缩为$\symbfit a_k^0$以满足公式(12)中的约束

但当$l$较大时，例如$l \ge 5000$，谱松弛方法的解可能会因为放大的松弛误差导致相对最优解偏移较远，因此该方法得到的解仅仅会作为以下优化方法的初始化值

### 2.3.2 Sigmoid Smoothing
因为对公式(9)的优化的困难主要在于符号函数，我们将符号函数$sgn()$
替换为sigmoid形状的函数$\varphi(x) = 2/(1 + \exp(-x)) -1$
该函数足够平滑，并且在$|x|>6$时可以近似$sgn(x)$

因此，我们使用$g$的平滑替代$\tilde g$
$$\tilde g(\symbfit a_k)=-(\varphi(\bar K_l\symbfit a_k))^TR_{k-1} \varphi(\bar K_l\symbfit a_k)\tag{13}$$
$\tilde g$相对于$\symbfit a_k$的梯度是
$$\nabla\tilde g = -\bar K_l^T((R_{k-1}\symbfit b)\odot(\symbfit 1-\symbfit b\odot\symbfit b))\tag{14}$$
其中$\odot$表示Hadamard积，即按元素相乘，$\symbfit b = \varphi(\bar K_l\symbfit a_k)\in \mathbb R^l$

原来的函数$g$存在下界，因此它的平滑替代$\tilde g$也存在下界，结果上看，我们可以使用常规的梯度下降法优化$\tilde g$，注意平滑替代函数$\tilde g$也是非凸函数，因此直接寻找全局最小值是不现实的
为了快速收敛，采用谱松弛的解$\symbfit a_k^0$作为初始值，然后运用Nesterov's梯度方法

大多数情况下，我们可以得到一个局部最小值$\symbfit a_k^*$，使得$\tilde g(\symbfit a_k^*)$非常接近下界
# 3 Analysis
# 4 Experiments
## 4.1 CIFAR-10
## 4.2 Tiny-1M
# 5 Conclusions


