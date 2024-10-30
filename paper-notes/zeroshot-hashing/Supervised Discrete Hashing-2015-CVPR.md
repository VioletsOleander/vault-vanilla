# Abstract
# 1 Introduction
为了利用有监督的标签信息，我们从线性分类的角度出发构建哈希框架，学习到的哈希码应是对分类最优的
可以认为该工作将原数据非线性地转化到二进制空间，然后在该空间对数据进行分类
# 2 Supervised Discrete Hashing
$n$个样本$X = \{\symbfit x_i\}_{i=1}^n$，我们意图学习一系列哈希码$B = \{\symbfit b_i\}_{ i=1}^n \in \{-1,1\}^{L\times n}$
以保留样本间的语义相似度

为了利用标签信息，在线性分类的框架上考虑二进制码学习问题，我们希望学到的二进制码是对共同学习到的分类器是最优的，即我们的假设是好的二进制码同时是对分类理想的

采用如下多类别分类形式
$$\symbfit y = G(\symbfit b)=W^T\symbfit b=[\symbfit w_1^T\symbfit b,\cdots, \symbfit w_C^T\symbfit b]^T\tag{1}$$
其中$\symbfit w_k \in \mathbb R^{L\times 1},k=1,\cdots, C$，$\symbfit y \in \mathbb R^{C\times 1}$，$C$是类别数
$\symbfit y$为标签向量，其中最大的项即表明所属的类别

优化以下问题
$$\begin{align}
&\min_{B, W, F} \sum_{i=1}^n L(\symbfit y_i,W^T\symbfit b_i)+\lambda\|W\|_F^2\\
&s.t.\symbfit b_i=sgn(F(\symbfit x_i)),i=1,\cdots,n
\end{align}\tag{2}$$
其中$L(\cdot)$表示损失函数，$\lambda$为正则参数
$Y = \{\symbfit y_i\}_{i=1}^n \in \mathbb R^{C\times n}$表示真实值标签矩阵
哈希函数$H(\symbfit x) = sgn(F(\symbfit x))$将$\symbfit x$编码为$L$位的二进制向量

离散约束下，问题2是NP-hard问题
为了能保持离散约束的同时解该优化问题，将问题2重写为
$$\begin{align}
&\min_{B,W,F}\sum_{i=1}^nL(\symbfit y_i,W^T\symbfit b_i)+\lambda\|W\|^2+\nu\sum_{i=1}^n\|\symbfit b_i-F(\symbfit x_i)\|_2^2\\
&s.t.\symbfit b_i\in\{-1,+1\}^L
\end{align}\tag{3}$$
问题3的最后一项是二进制码$\symbfit b_i$和连续嵌入$F(\symbfit x_i)$的拟合误差，$\nu$为惩罚参数
理论上，足够大的$\nu$可以使得问题3任意接近问题2

问题3仍然高度非凸且难以解决，但在选择好合适的损失函数$L(\cdot)$后，可以迭代式逐个变量求解
## 2.1 Approximating $b_i$ by nonlinear embedding
设$F(\symbfit x)$的形式为
$$F(\symbfit x) = P^T\phi(\symbfit x)\tag{4}$$
其中$\phi(\symbfit x)$是一个$m$维的列向量，由RBF核映射得到，即$\phi(\symbfit x) = [\exp(\| \symbfit  x  - \symbfit a_1 \|_2^2/\sigma,\cdots,\exp(\| \symbfit  x  - \symbfit a_m \|_2^2/\sigma]^T$
其中$\{\symbfit a_j\}_{j=1}^m$为从训练集中随机选取的$m$个锚点，而$\sigma$为核带宽
$P\in\mathbb R^{m\times L}$

**F-Step**
固定公式3中的$B$，可以直接得到$P$
$$P = (\phi(X)\phi(X)^T)^{-1}\phi(X)B^T\tag{5}$$
注意该步骤是完全独立于损失函数$L(\cdot)$的

## 2.2 Joint learning with $l_2$ loss
公式3是灵活的，我们为该分类模型选取任意形式的损失，一个简单的选择是$l_2$损失
此时公式3重写为
$$\begin{align}
&\min_{B,W,F} \sum_{i=1}^n \|\symbfit y_i - W^T\symbfit b_i\|_2^2+\lambda\|W\|_F^2+\nu \sum_{i=1}^n\|\symbfit b_i - F(\symbfit x_i)\|_2^2\\
&s.t.\symbfit b_i\in\{\pm 1\}^L
\end{align}\tag{6}$$
即
$$\begin{align}
&\min_{B,W,F}  \|Y - W^TB\|_F^2+\lambda\|W\|_F^2+\nu \|B - F(X)\|_F^2\\
&s.t.B\in\{\pm 1\}^{L\times n}
\end{align}\tag{7}$$

**G-Step**
对于问题7，固定$B$，关于$W$就是一个带正则的最小二乘问题，闭式解为
$$W = (BB^T+\lambda I)^{-1}BY^T\tag{8}$$
**B-Step**
固定除$B$以外的变量，将问题7写为
$$\begin{align}
&\min_B \|Y-W^TB\|_F^2+\nu\|B-F(X)\|_F^2\\
&s.t.B\in\{\pm 1\}^{L\times n}
\end{align}\tag{9}$$
该问题是NP-hard问题，但当$B$的除某一行以外的所有其他行都固定时，我们可以得到$B$中的该行的闭式解，这意味着我们可以迭代式每次学习一个比特

将公式9重写为
$$\begin{align}
&\min_B\|Y\|_F^2-2tr(Y^TW^TB)+\|W^TB\|_F^2+\\
&\nu(\|B\|_F^2-2tr(B^TF(X))+\|F(X)\|_F^2)\\
&s.t. B\in \{\pm 1\}^{L\times n}
\end{align}\tag{10}$$
等价于
$$\begin{align}
&\min_B\|W^TB\|_F^2-2tr(B^TQ)\\
&s.t. B\in \{\pm 1\}^{L\times n}
\end{align}\tag{10}$$
其中$Q = WY + \nu F(X)$

我们采用离散循环坐标下降(DCC)方法学习$B$，即我们按位学习$B$，令$\symbfit z^T$为$B$的第$l$行，$l = 1,\cdots, L$，$B'$为除去第$l$行的$B$
类似地，令$\symbfit q^T$为$Q$的第$l$行，$Q'$为除去第$l$行的$Q$，令$\symbfit v^T$为$W$的第$l$行，$W'$为除去第$l$行的$W$，此时我们有
$$\begin{align}
\|W^TB\|_F^2&=tr(B^TWW^TB)\\
&=const+\|\symbfit z\symbfit v^T\|_F^2+2\symbfit v^TW'^TB'\symbfit z\\
&=const+2\symbfit v^TW'^TB'\symbfit z
\end{align}\tag{12}$$
注意$\|\symbfit z\symbfit v^T\|_F^2 = tr(\symbfit v\symbfit z^T\symbfit z \symbfit v^T) = n\symbfit v^T\symbfit v = const$
类似地，我们有
$$tr(B^TQ) = const + \symbfit q^T\symbfit z\tag{13}$$
结合公式11、12、13，我们有相关于$\symbfit z$的如下问题
$$\begin{align}
&\min_{\symbfit z}(\symbfit v^TW'^TB'-\symbfit q^T)\symbfit z\\
&s.t. \symbfit z\in \{\pm 1\}^{n}
\end{align}\tag{14}$$
该问题有最优解
$$\symbfit z = sgn(\symbfit q - B'^TW'^T\symbfit z)\tag{15}$$
公式15中我们可以知道，每一位$\symbfit z$的计算都在之前学习到的$L-1$位的基础上
实验中，全部的$L$位一般花费$tL$次迭代，$t = [2,5]$

## 2.3 Joing Learning with hinge loss
# 3 Experiments
## 3.1 Comparision between the $l_2$ loss and hinge loss
## 3.2 Discrete or not?
## 3.3 Retrieval on tiny natural images
## 3.4 MNIST: retrieval with hand-written digits
## 3.5 NUS-WIDE: retrieval with multiple labels
## 3.6 ImageNet: retrieval with high dimensional features
## 3.7 Classification with binary codes
# 4 Conclusions
