# Abstract
# 1 Introduction
Hashing with Graphs(AGH)利用锚点图将成对相似度矩阵转化为低秩的邻接矩阵，使得在大数据集上的计算可行性大大增加，但该方法的表现对选取的锚点敏感
DGH可以视为AGH的拓展，它将基于图的哈希转化为复杂的离散优化框架
# 2 Related Work
# 3 Problem Statement
$\mathcal D = \{(\mathbf x_i,\mathbf l_i)\}_{i=1}^N$为图像集，
其中$\mathbf x_i \in \mathbb R^{M}$为第$i$个图像的特征向量，$\mathbf l_i\in \{0,1\}^C$为对应标签向量
$N$为图片样本数量，$C$为类别数量

样本$\mathbf x_i, \mathbf x_j$之间的相似度$s_{ij}$的计算是
$$\begin{align}
s_{ij} &= 2\cos \langle \mathbf l_i, \mathbf l_j\rangle - 1\\
&=2 \frac {\mathbf l_i^T\mathbf l_j}{\|\mathbf l_i\|_2\cdot\|\mathbf l_j\|_2}-1\\
&=2\left(\frac {\mathbf l_i}{\|\mathbf l_i\|_2}\right)^T\left(\frac {\mathbf l_j}{\|\mathbf l_j\|_2}\right)-1\tag{1}
\end{align}$$
注：对于一个向量$\mathbf x$，它的二范数定义为$\|\mathbf x_i\| = \left(\sum_{i=1}^n x_i^2\right)^{\frac 1 2}$，即它的长度

我们令
$$G = \left[\frac {\mathbf l_1}{\|\mathbf l_1\|_2},\frac {\mathbf l_2}{\|\mathbf l_2\|_2},\cdots,\frac {\mathbf l_N}{\|\mathbf l_N\|_2}\right]^T\tag{2}$$
注意标签向量$\mathbf l_i$的长度都是$1$

因此成对相似度矩阵可以如下计算
$$S = 2GG^T-\mathbf 1_N\mathbf 1_N^T\tag{3}$$
其中$s_{ij}$的范围是$[-1,+1]$

目标是学习一系列可以在汉明空间保持基于标签计算的成对相似度的哈希函数
具体地说，是$K$个哈希函数$H(\cdot) = [h_1(\cdot),h_2(\cdot),\cdot, h_K(\cdot)]^T$，将每个图像$\mathbf x_i$嵌入一个$K$位的哈希空间，即$\mathbf b_i = H(\mathbf x_i)\in \{-1,+1\}^K$
因此整个数据集可以转化为$B = [\mathbf b_1, \mathbf b_2, \cdots, \mathbf b_N]^T\in \{-1,+1\}^{N\times K}$
# 4 Proposed Method
## 4.1 Similarity Preservation
给定两个图像$(\mathbf x_i, \mathbf x_j)$的各自的$K$位哈希码，它们的点积(范围是$[-K, +K]$)在理想情况下，应该和$(\mathbf x_i,\mathbf x_j)$之间的语义相似度$s_{ij}$成比例

因此通过如下优化问题让哈希码保持成对相似度
$$\begin{align}
&\min_{B}\|K\cdot S - BB^T\|_F^2\\
&s.t.\ B\in\{-1,+1\}^{N\times K},B^T\mathbf1_N = \mathbf 0_K,B^TB = N\cdot I_K\tag{4}
\end{align}$$
限制$B^T\mathbf 1_N = \mathbf 0_K$要求哈希码是平衡的(balanced)，即每一位有$50\%$的几率是$0$或$1$
限制$B^TB = N\cdot I_K$要求哈希码是不相关的(uncorrelated)

目标函数(4)有两个计算上的挑战
如何高效地构建$N\times N$的成对相似度矩阵$S$
如何高效地解决强约束下的离散优化问题

对于第一个挑战，作者用低秩的$G$(其中$C\ll N$)通过式$(3)$表示$S$，以减小存储开销
对于第二个挑战，多数现存方法将$B\in \{-1,+1\}^{N\times K}$松弛离散限制为$B\in\mathbb R^{N\times K}$，本文提出的方法通过辅助变量保持离散限制

## 4.2 Joint Learning
令$X = [\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_N]^T$，对于快速图像检索，使用线性哈希函数$P\in \mathbb R^{M\times K}$以生成哈希函数
$$B=sgn(XP)\tag{5}$$
通过拓展式(4)，以同时学习哈希函数$P$和哈希码$B$
$$\begin{align}
&\min_{B,P}\|K\cdot S-BB^T\|_F^2+\lambda\|sgn(XP)-B\|_F^2+\beta\|P\|_F^2\\
&s.t.\ B\in\{-1,+1\}^{N\times K},B^T\mathbf1_N = \mathbf 0_K,B^TB = N\cdot I_K\tag{6}
\end{align}$$
其中$\lambda$为一个权衡哈希码和哈希函数重要性的正的参数，$\beta$是一个非负的平滑因子(smoothing factor)以避免过拟合

符号函数$sgn(\cdot)$不可微，因此用$XP$替换$sgn(XP)$，因此要求$XP$中的每个元素自身接近$B$中对应的元素($-1$或$+1$)

为了进一步简化问题，引入一个辅助变量$Z$作为$B$的别名，即$Z = B$，重写(6)
$$\begin{align}
&\min_{B, P, Z}\|K\cdot S - BZ^T\|_F^2+\lambda\|XP-B\|_F^2+\beta\|P\|_F^2\\
&s.t.\begin{cases}B\in\{-1,+1\}^{N\times K}\\Z = B,Z^T\mathbf1_N = \mathbf 0_K,Z^TZ = N\cdot I_K \end{cases}\tag{7}
\end{align}$$

## 4.3 The Complete Optimization Problem
进一步地，去除限制$Z = B$，并将$Z$松弛对离散的$B$的实连续(real-valued continuous)近似，换句话说，$Z,B$不再要求严格相等，而是要互相近似，因此得到最终的优化目标
$$
\begin{align}
&\min_{B,P,Z}\mathcal O(P, B, Z)=\|K\cdot S- BZ^T\|_F^2\\
&+\lambda\|XP-B\|_F^2+\alpha\|B-Z\|_F^2+\beta\|P\|_F^2\\
&s.t.\begin{cases}B\in\{-1,+1\}^{N\times K}\\
Z \in \mathbb R^{N\times K},Z^T\mathbf1_N = \mathbf 0_K,Z^TZ = N\cdot I_K \end{cases}\tag{8}
\end{align}
$$
其中参数$\alpha$控制了$Z$对$B$的近似有多接近

在如上的框架下，我们可以同时得到哈希码和哈希函数，对于训练集外样本$X_{oos}$，它们的哈希码可以直接通过哈希函数映射得到
$$B_{oos} = sgn(X_{oos}P)\tag{9}$$
实际上仅仅是一个线性变换

## 4.4 Kernelization
非线性哈希函数可以拟合数据中更复杂的模式，因此本文的方法也可以通过核函数延伸至非线性哈希
给定一个非线性映射$\Phi: \mathbf x \in \mathbb R^{M} \to \Phi(\mathbf x)\in \mathbb R^D$，可以得到整个数据集的映射后的像
$\Phi(X)  = [\Phi(\mathbf x_1),\Phi(\mathbf x_2),\cdots, \Phi(\mathbf x_N)]^T\in \mathbb R^{N\times D}$

从数据集中随机选取$Q$个锚点，记为$\mathbf y_1, \mathbf y_2,\cdots ,\mathbf y_Q$，视$\Phi(\mathbf y_1), \Phi(\mathbf y_2),\cdots, \Phi(\mathbf y_Q)$为一组基向量(base vectors)，可以用于表示$\mathbb R^{D}$中的任意向量
这是一个常用的较有效的处理大数据的方法
我们得到$$
\begin{align}
\Phi(P) &= \Phi([\mathbf p_1,\mathbf p_2,\cdots, \mathbf p_K])\\
&=[\Phi(\mathbf y_1),\Phi(\mathbf y_2),\cdots, \Phi(\mathbf y_Q)]A\tag{10}
\end{align}$$
其中$A\in \mathbb R^{Q\times K}$，由此，等式(5)可以拓展为 $$\begin{align} B &=sgn(\Phi(X)\Phi(P))\\ &=sgn([\Phi(\mathbf x_1),\Phi(\mathbf x_2),\cdots, \Phi(\mathbf x_N)]^T\\ &\quad\ \times [\Phi(\mathbf y_1),\Phi(\mathbf y_2),\cdots,\Phi(\mathbf y_N)]A)\\ &=sgn((\Phi(\mathbf x_i)^T\Phi(\mathbf y_j)_{N\times Q}A)\tag{11} \end{align}$$ 令$\mathcal K: \mathbb R^D \times \mathbb R^D \to \mathbb R$表示非线性映射$\Phi$对应的核函数，则$\mathcal K_Q = (\Phi(\mathbf x_i)^T\Phi(\mathbf y_j))_{N\times Q}$表示核矩阵，则核化的$SCDH$算法为 $$\begin{align} &\min_{B, A, Z}\|K\cdot S-BZ^T\|_F^2+\lambda\|\mathcal K_QA-B\|_F^2\\ &\quad\quad +\alpha\|B- Z\|_F^2+\beta\|A\|_F^2\\ & s.t. \begin{cases}B\in\{-1,+1\}^{N\times K}\\Z\in\mathbb R^{N\times K}, Z^T\mathbf 1_N = \mathbf 0_K,Z^TZ = N\cdot I_K\end{cases}\tag{12} \end{align}$$简称该算法为$SCDH_K$ 

在选择了核函数$\mathcal K$和学习了矩阵$A$后，训练集外样本的编码可以如下计算得到
$$
\begin{align}
\mathbf b_{oos} 
&= sgn\left(\left[(\Phi(\mathbf x_{oos})^T\Phi(\mathbf y_j)_{1\times Q}A\right]^T\right)\\
&=sgn\left(\left[(\mathcal K(\mathbf x_{oos},\mathbf y_j)_{1\times Q}A\right]^T\right)\tag{13}
\end{align}$$
因为$Q\ll N$，上式的计算是十分高效的

# 5 Optimization
问题(8)包括三个变量需要优化$B, Z , P$，采用交替优化算法，在优化其中一个变量时，保持另一个变量不变，不断迭代直至收敛

## 5.1 Update P With B and Z Fixed
固定$B, Z$，关于$P$的目标函数写为
$$\min_P \mathcal O(P)=\lambda\|XP-B\|_F^2+\beta\|P\|_F^2\tag{14}$$
该问题实际上是一个带$L_2$正则最小二乘问题，令$\frac {\partial \mathcal O(P)}{\partial P} = 0$，直接得到闭式解
$$P = (X^TX+\frac {\beta}{\lambda}I_M)^{-1}X^TB\tag{15}$$

## 5.2 Update B With Z and P Fixed
固定$P, Z$，关于$B$的目标函数写为
$$\begin{align}
&\min_B \mathcal O(B) = \|K\cdot S - BZ^T\|_F^2+\lambda\|XP-B\|_F^2+\alpha \|B-Z\|_F^2\\
&s.t. B\in\{-1,+1\}^{N\times K}\tag{16}
\end{align}$$
等价于以下优化问题
$$\begin{align}
&\max_B tr(B^T\{K\cdot SZ+\lambda XP+\alpha Z\})\\
&s.t.B\in\{-1,+1\}^{N\times K}\tag{17}
\end{align}$$

为了解决这个问题，引入以下定理
定理1:
给定一个矩阵$C\in\mathbb R^{N\times K}$，优化问题
$$\max_B tr(BC^T)\quad s.t.B\in\{-1,+1\}^{N\times K}\tag{18}$$
有闭式解$B = sgn(C)$
证明:
根据迹函数的定义
$$tr(BC^T) =\sum_{i,j}b_{ij}c_{ij}\tag{19}$$
优化问题(18)等效于
$$\max_{b_{ij}} b_{ij}c_{ij}\quad s.t. b_{ij}\in\{-1,+1\}\tag{20}$$
对所有$b_{ij}(i\in\{1,2,\cdots, N\}, j\in\{1,2, \cdots, K\})$

显然，为了达到最大，每对$b_{ij}c_{ij}$都需要为正，即$b_{ij} = sgn(c_{ij})$，则定理得证

因此，式(16)的闭式解为
$$B = sgn(K\cdot SZ + \lambda XP+\alpha Z)\tag{21}$$

## 5.3 Update Z With B and P Fixed
固定$B, P$，关于$Z$的目标函数写为
$$
\begin{align}
&\min_Z \mathcal O(Z)=\|K\cdot S-BZ^T\|_F^2+\alpha\|B-Z\|_F^2\\
&s.t.Z\in\mathbb R^{N\times K}, Z^T\mathbf 1_N = \mathbf 0_K, Z^TZ=N\cdot I_K\tag{22}
\end{align}
$$
将其进一步简化为
$$\begin{align}
&\max_Z tr(Z^T\{K\cdot SB+\alpha B\})\\
&s.t.Z\in\mathbb R^{N\times K}, Z^T\mathbf 1_N = \mathbf 0_K, Z^TZ=N\cdot I_K\tag{23}
\end{align}$$
令$E = K\cdot SB + \alpha B$

为了解决这个问题，引入以下定理
定理2:
优化问题
$$\max_Ztr(Z^TE)\quad s.t.Z^T\mathbf 1_N = \mathbf 0_K, Z^TZ=N\cdot I_K\tag{24}$$
的闭式解为
$$Z = \sqrt N[U,\bar U][V,\bar V]^T\tag{25}$$
而矩阵$U = [\mathbf u_1,\mathbf u_2,\cdots, \mathbf u_K'], V = [\mathbf v_1,\mathbf v_2,\cdots, \mathbf v_K']$通过$JE$的奇异值分解得到
其中$J = I_N - \frac 1 N \mathbf 1_N \mathbf 1_N^T$，即$$JE = U\Sigma V^T=\sum_{k=1}^{K'}\sigma_k\mathbf u_k\mathbf v_k^T\tag{26}$$其中$\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{K'}\gt 0$
而矩阵$\bar U \in \mathbb R^{N\times (K-K')},\bar V\in \mathbb R^{K\times (K- K')}$则通过格兰-施密特正交化得到，以使得
$\bar U^T\bar U = I_{K-K'}, [U,\mathbf 1_N]^T\bar U = 0$
$\bar V^T \bar V = I_{K-K'},V^T\bar V=0$
若$K' = K$，则$\bar V, \bar U$为空
证明参考[[Discrete Graph Hashing-2014-NeurIPS]]

## 5.4 Computational Complexity
总的计算复杂度在每轮迭代线性于$N$

# 6 Experiments
## 6.1 Datasets
## 6.2 Evaluation
## 6.3 Setting
## 6.4 Results
## 6.5 Convergence Analysis
## 6.6 Hyperparameters of SCDH_K
$SCDH_K$(使用高斯核函数)有两个重要的超参数: 随机选取的锚点数量$Q$，核带宽$\sigma$
## 6.7 Ablation Study
## 6.8 Constraints vs Regularizers
## 6.9 Shallow vs Deep
## 6.10 Binary vs Real-Valued
## 6.11 Unseen Classes
## 6.12 Case Study
# 7 Conclusion




