# Abstract
# 1 Introduction
# 2 Overview
作者从基于锚点的标签预测和邻接矩阵设计两个方面入手以解决半监督学习的放缩性问题(scalability issue)

## 2.1 Anchor-Based Label Prediction
数据集$\mathcal X = \{\symbfit x_i\}_{i=1}^n\subset \mathbb R^d$，标签预测函数$f:\mathbb R^d \to \mathbb R$
在大规模的数据下，标签预测函数可以是锚点集的标签的加权平均，记锚点集$\mathcal U = \{\symbfit u_k\}_{k=1}^m \subset \mathbb R^d$，则经由锚点表现$f$：
$$f(\symbfit x_i) = \sum_{k=1}^mZ_{ik}f(\symbfit u_k)\tag{1}$$
其中$Z_{ik}$是样本自适应权重(sample-adaptive weights)

令$\boldsymbol f= [f(x_1),\cdots,f(x_n)]^T$，$\boldsymbol a = [f(u_1),\cdots,f(u_m)]^T$，重写上述等式
$$\boldsymbol f=Z\boldsymbol a, Z\in\mathbb R^{n\times m},m\ll n\tag
{2}$$
该式子将未知标签的解空间从较大的$\boldsymbol f$减小到较小的$\boldsymbol a$

锚点$\{u_k\}$取为k-means的簇中心而不是随机选择，作者认为k-means的簇中心具有更强的表示能力

## 2.2 Adjacency Matrix Design
kNN构图法在$x_i$是$x_j$的k最近邻或$x_j$是$x_i$的k最近邻的情况下连接点$v_i$和$v_j$，其时间复杂度是$O(kn^2)$，在$n$很大的情况下，该构图法可能是不可行的

## 2.3 Design Principles
对于大规模的问题，提出两个设计原则

Principle 1:
对$Z$施加非负和规范化限制，即$\sum_{k=1}^m Z_{ik} = 1$且$Z_{ik}\ge 0$，以保持通过回归得到的软(soft)标签的值域是一致的
流形假设(manifold assumption)表明连续的数据点应有相似的标签，独立的数据点则不太可能有相似的标签，因此在锚点$u_k$远离$x_i$时，令$Z_{ik}=0$，如此一来$x_i$的标签回归就是一个局部的加权平均和，并且$Z\in \mathbb R^{n\times m}$稀疏且非负

Principle 2:
要求邻接矩阵非负，即$W\ge 0$，那么图拉普拉斯矩阵$L =  W  - D$半正定，该性质保证了许多基于图半监督方法的全局最优解

Principle 3:
偏好稀疏的邻接矩阵$W$，以避免不相似点之间的错误的连接，有人指出全连接密集图表现要差于稀疏图
# 3 AnchorGraph: Large Graph Construction
## 3.1 Design of Z
目标在于设计回归矩阵$Z$，衡量样本$\mathcal X$与锚点$\mathcal U$的潜在关系，注意$\mathcal U$和$\mathcal X$不相交
根据Principle 1，我们希望对$x_i$，仅有相对其最近的$s(<m)$个锚点所对应的项$Z_{ik}$是非零的

有核函数$K_h()$和带宽$h$，定义
$$Z_{ik}=\frac {K_h(\symbfit x_i,\symbfit u_k)}{\sum_{k'\in\langle i\rangle}K_h(\symbfit x_i,\symbfit u_k')}\forall k\in\langle i\rangle\tag{3}$$
其中$\langle i\rangle \subset [1:m]$即保存了离$x_i$最近的$s$个锚点的索引的集合
我们采用高斯核函数$K_h(\symbfit x_i,\symbfit u_k) = \exp(-\|\symbfit x_i-\symbfit u_k\|^2/2h^2)$
但考虑核定义的权重对超参数$h$敏感，且缺乏解释，采用另一种方法

我们希望用数据点$x_i$最近邻的锚点的凸结合(convex combination)重构数据点
令$U = [u_1,\cdots, u_n]$，$U_{\langle i \rangle}\in \mathbb R^{d\times s}$为$x_i$的$s$个最近邻的锚点子矩阵

作者提出局部锚点嵌入(local anchor embedding/LAE)以优化凸的结合参数(coefficients)
$$\min_{\symbfit z_i\in\mathbb R^s} g(\symbfit z_i)=\frac 1 2\|\symbfit x_i-U_{\langle i\rangle}\symbfit z_i\|^2\ s.t.\ \symbfit 1^T\symbfit z_i=1,\symbfit z_i\ge0\tag{4}$$
对该问题的凸解集构成了一个多项式单纯形(multinomial simplex)
$$\mathbb S = \{\symbfit z\in \mathbb R^s: \symbfit 1^T\symbfit z=1,\symbfit z\ge0\}\tag{5}$$
对$z_i$的求解采用梯度下降，得到最优的权重向量后，设置
$$Z_{i,\langle i \rangle} =\symbfit z_i^T,|\langle i\rangle|=s,\symbfit z_i\in \mathbb R^s\tag{8}$$
并且$Z_{i,\bar {\langle i\rangle }} = 0$
最后得到的$Z$高度稀疏，其占用的内存大小约为$O(sn)$，满足Principle 1

## 3.2 Design of W
得到$Z$后，设计邻接矩阵$W$为
$$W = Z\Lambda^{-1}Z^T\tag{9}$$
其中$\Lambda\in \mathbb R^{m\times m}$为对角矩阵，$\Lambda_{kk}  = \sum_{i=1}^nZ_{ik}$
由于$Z$非负，$W$非负，满足Principle 2，
且由于$Z$稀疏，则当锚点被设为簇中心以使得多数不同簇间的数据点的最近锚点不同，经验上$W$稀疏，满足Principle 3

称以$W$为邻接矩阵所描述的图$G$为Anchor Graph
等式(9)是该文的核心发现，该式构造了一个非负且经验性稀疏的图邻接矩阵
Anchor Graph的图拉普拉斯矩阵为$L = D - W = I - Z\Lambda^{-1}Z^T$

理论上，等式(9)也可以由概率方法(probabilistic means)推导出
LAE算法中，我们从几何上的重构视角推导出$Z$，矩阵$Z$实际上揭示了在数据点和锚点之间的一个紧密的亲和性度量(a tight affinity measure)
概念上理解，即一个锚点$\symbfit u_k$对数据点$\symbfit x_i$的重构贡献的越多，它们之间的亲和性就越大

为了显式表达数据点和锚点之间的关系，引入二部图(bipartite graph)$B(\mathcal V, \mathcal U, \mathcal E)$
$V$为数据点集，点集$\mathcal U$包含了锚点$\{u_k\}_{k=1}^m$，集合$\mathcal E$包含了连接$V$和$\mathcal U$的边
当且仅当$Z_{ik} > 0$时，在$u_k, v_i$之间连接一个无向边




