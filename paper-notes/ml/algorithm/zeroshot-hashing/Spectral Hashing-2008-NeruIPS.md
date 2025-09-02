# Abstract
# 1 Introduction
# 2 Analysis: what makes a good code
平衡性：哈希码每一位都有 $50\%$ 的概率是 $-1$ 或 $1$，
不相关性：哈希空间满秩，各维度相互正交 (这是不同的位相互独立的松弛情况)

关联矩阵 $W\in R^{n\times n}$，$W_{ij} = \exp(-\|x_i-x_j\|^2/\epsilon^2)$
哈希空间中，相似邻居间的平均汉明距离写作 $\sum_{ij} W_{ij}\|y_i - y_j\|^2$

优化问题

$$\begin{aligned}minize:\sum_{ij}W_{ij}\|y_i-y_j\|^2\\
subject\ to:y_i\in\{-1,1\}^k\\
\sum_i y_i = 0\\
\frac 1 n\sum_i y_iy_i^T = I
\end{aligned}$$

若 $k=1$，该问题转换为图的平衡划分问题，其中 $-1$ 的一部分、$1$ 的一部分，分别称其为 $A,B$
而最小化 $\sum_{ij}W_{ij}\|y_i-y_j\|^2$ 等价于最小化 $\sum_{i \in A, j\in B}W_{ij} = cut(A, B)$，因此该问题等价于最小化 $cut(A,B)$，且要求 $|A| = |B|$，这是一个NP难问题

对于 $k$ 个位，也可以认为是找到 $k$ 个独立的平衡划分

## 2.1 Spectral Relaxation
哈希空间中的特征矩阵 $Y=\begin{bmatrix}y_1^T\\ \vdots \\ y_n^T\end{bmatrix}\in \mathbb R^{n\times k}$
度矩阵 $D = diag(\sum_j W_{1,j},\cdots, \sum_j W_{n,j})$

因为 $$\begin{aligned}
&trace(Y^TLY)\\
=&trace([y_1,\cdots,y_n]L\begin{bmatrix}y_1^T\\ \vdots \\ y_n^T\end{bmatrix})\\
=&trace(\begin{bmatrix}f_1^T\\ \vdots \\ f_n^T\end{bmatrix}L[f_1,\cdots,f_n])\\
=&trace(\begin{bmatrix}f_1^T\\ \vdots \\ f_n^T\end{bmatrix}[Lf_1,\cdots,Lf_n])\\
=&f_1^TLf_1+\cdots+f_n^TLf_n\\
=&\sum_{i=1}^nf_i^TLf_i\\
=&\sum_{i=1}^n\frac 1 2\sum_{jk}^nW_{jk}(f_{ij}-f_{ik})^2\\
=&\frac 1 2\sum_{jk}^nW_{jk}\sum_{i=1}^n(f_{ij}-f_{ik})^2\\
=&\frac 1 2\sum_{jk}^nW_{jk}\|y_j-y_k\|^2
\end{aligned}$$
故将优化问题重写为
$$\begin{aligned}
minimize:trace(Y^T(D-W)Y)\\
subject\ to:Y_{ij}\in\{-1,1\}\\
Y^T1=0\\
Y^TY=I
\end{aligned}$$
如果移除限制 $Y_{ij}\in\{-1,1\}$，则该问题就是简单的找到图拉普拉斯矩阵 $D-W$ 的 $k$ 个最小特征值对应的特征向量的问题 ($0$ 特征值对应的特征向量 $1$ 排除)

## 2.2 Out of Sample Extension
解出 2.1 中的松弛问题的 $k$ 个特征向量，再以阈值将其二元化即得到所要的哈希空间中的特征矩阵 $Y$

要求解样本外 (out-of-sample) 数据的哈希特征，我们假设数据点 $x_i\in R^d$ 是从概率分布 $p(x)$ 中采样得到的

分布中可以采样无穷多个样本，则优化问题中的求和替换成积分得

$$\begin{aligned}
minimize:\int\|y(x_1)-y(x_2)\|^2W(x_1,x_2)p(x_1)p(x_2)dx_1x_2\\
subject\ to:y(x)\in \{-1,1\}^k\\
\int y(x)p(x)dx=0\\
\int y(x)y(x)^Tp(x)dx=I
\end{aligned}$$

其中 $W(x_1,x_2)=\exp(-\|x_1-x_2\|^2/\epsilon^2)$
同样移除 $y(x)$ 的离散限制，则问题的解由特征向量变为加权 Laplace-Beltrami 算子的特征函数，即满足 $L_pf=\lambda f$ 的最小 $k$ 个特征函数 ($0$ 特征值对应的特征函数 $f(x)=1$ 排除)

事实上，如果经过正确的规范化，从从 $p(x)$ 采样 $n$ 个样本，计算其对应的离散拉普拉斯矩阵 $L$ 得到的特征向量在 $n\to\infty$ 时会向拉普拉斯算子 $L_p$ 的特征函数收敛

如果 $p(x)$ 是一个可分离分布 (separable distribution)，例如多维均匀分布 (multidimensional uniform distribution) $Pr(x) = \prod_iu_i(x_i)$ (其中 $u_i$ 为范围在 $[a_i,b_i]$ 内的均匀分布)，并且数据点之间的相似度定义为 $\exp(-\|x_i-x_j\|^2/\epsilon^2)$
则 $L_p$ 的特征函数存在外积形式，即若 $\Phi_i(x)$ 是 $L_p$ 在 $R^1$ 的特征函数，特征值为 $\lambda_i$，则 $\Phi_i(x_1)\Phi_j(x_2)\cdots\Phi_d(x_d)$ 是 $d$ 维问题的特征函数，其特征值为 $\lambda_i\lambda_j\cdots\lambda_d$

单维特征函数我们记为 $\Phi_k(x_1)$，或 $\Phi_k(x_2)$ 的形式，外积特征函数我们记为 $\Phi_k(x_1)\Phi_l(x_2)$ 的形式

如果我们通过对 $L_p$ 的 $k$ 个最小特征函数二值化以得到哈希码，即 $y(x) = sign(\Phi_k(x))$，若特征函数中的任意一个是外积特征函数，则哈希码中的一位是哈希码其他位的确定函数

给定数据集 $\{x_i\}$ 和哈希码长度 $k$，谱哈希算法步骤为：
- 使用 PCA 找到数据的主成分
- 沿着每个 PCA 方向，使用长方形近似 (rectangular approximation) 计算 $L_p$ 的 $k$ 个最小的单维解析特征函数
- 二值化特征函数得到哈希码

该算法的限制在于
1. 假设数据由多维均匀分布生成
2. 即使该算法避免了由外积特征函数引起的位之间的依赖，但不能保证不存在更高阶 (high-order) 的依赖

# 3 Results
# 4 Discussion
