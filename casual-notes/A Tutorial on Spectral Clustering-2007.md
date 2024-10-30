# 1 Introduction
# 2 Similarity graphs
聚类的直观目标是划分类使得类内的点相似而类间的点不相似
在图表示中，可以用点间的边权重大小衡量相似度
## 2.1 Graph notation
无向图$G = (V, E)$，$V = {v_1,\dots, v_n}$
加权邻接矩阵$W$，其中$w_{ij} \ge 0$，$w_{ij}= 0$说明$v_i,v_j$之间不直接相连

$v_i$的度$d_i = \sum_{j=1}^n w_{ij}$
度矩阵$D$定义为$$D = diag(d_1,\dots, d_n)$$
任意点集$A\subset V$，其补$\bar A$是$V\backslash A$

定义指示向量$\mathbb 1_A = [f_1,\dots, f_n]^T \in \mathbb R^n$，其中$f_i = 1$当且仅当$v_i \in A$，否则$f_i = 0$

对无交集的$A,B\subset V$，定义$links(A, B) = \sum_{i\in A, j\in B} w_{ij}$

衡量集合的大小，定义两个量
- $|A|$
	表示$A$中结点数
- $vol(A) = \sum_{i\in A} d_i = \sum_{v_i\in A}\sum_{j=1}^n w_{ij}$
	$cut(A) = links(A, \bar A), assoc(A) = links(A, A)$
	$cut(A) + assoc(A) = vol(A)$
	表示所有与$A$相连的边的权重和

## 2.2 Different similarity graphs
给定数据点$x_1,\cdots,x_n$和相似度矩阵$S$，有不同的构建相似图的方式，以建模数据点之间的局部临近关系

**The $\epsilon$-neighborhodd graph**
将距离/相似度小于$\epsilon$的点相连接，构建无权重图(unweighted graph)
适用于临近点之间的距离大部分都是一个量级(same scale)，对边添加权重为图带来更多的临近信息的情况

**$k$-nearest neighbor graphs**
如果$v_j$是$v_i$的$k$最近邻，则将$v_j$与$v_i$连接，这种构建方式会构建出有向图，因为$k$近邻关系不一定是对称的
有两种方法可以将该有向图转换为无向图，
一种方法即直接忽略边的方向，即当$v_j$是$v_i$的$k$最近邻或$v_i$是$v_j$的$k$最近邻时，$v_i$与$v_j$相连，得到的图一般称为$k$最近邻图(k-nearest neighbor graph)
另一种方法即当$v_j$是$v_i$的$k$最近邻且$v_i$是$v_j$的$k$最近邻时，$v_i$与$v_j$相连，得到的图一般称为互$k$最近邻图(mutual k-nearest neighbor graph)
得到图后，我们为相应的边附上的权重即其连接的两点之间的相似度

**The fully connected graph**
直接视$S$为相似图的邻接矩阵，构建相似图，因此图中所有结点都相连，边的权重为其连接的两点之间的相似度
相似图要求表征了点之间的局部临近关系(local neighborhood relationships)，因此该方法仅在相似度函数本身正确建模了局部临近关系时有效
这种类型的函数比如说有高斯相似度函数$s(x_i,x_j) = \exp(\frac {-\|x_i-x_j\|^2}{2\sigma^2})$，其中参数$\sigma$控制了临近关系的宽度(width of the neighborhoods)，和$\epsilon$邻近图中的$\epsilon$的作用相似

# 3 Graph Laplacians and their basic properties
## 3.1 The unnormalized graph Laplacian
定义未规范化的图拉普拉斯矩阵为$$L = D - W$$即度矩阵减去邻接矩阵
显然，$L$中的非对角线元素$l_{ij}=-w_{ij} \le 0(i\ne j)$，
$L$中的对角线元素$l_{ij} = d_{ij} - w_{ij}(i=j)$

$L$满足
(1) 对任意$f\in \mathbb R^n$，有
$$f^TLf = \frac 1 2\sum_{i,j=1}^nw_{ij}(f_i-f_j)^2$$
证明：
$$\begin{aligned}
f^TLf = f^T(D-W)f &= f^TDf-f^TWf\\
&= \sum_{i=1}^n d_if_i^2-\sum_{i,j=1}^n f_if_jw_{ij}\\
&= \frac 1 2\left(\sum_{i=1}^nd_if_i^2-2\sum_{i,j=1}^nf_if_jw_{ij} + \sum_{j=1}^nd_j f_j^2 \right)\\
&=\frac 1 2\sum_{i,j=1}^nw_{ij}(f_i - f_j)^2
\end{aligned}$$
(2) $L$为半正定矩阵

证明：由(1)得对任意$f\in \mathbb R^n$，有$f^TLf \ge 0$

(3) $L$的最小特征值为0，对应特征向量是$\mathbb 1$

证明：易知$L\mathbb 1 = 0$

(4) $L$有$n$个非负的实特征值$0 = \lambda_1 \le \lambda_2 \le \cdots \lambda_n$

未规范化的图拉普拉斯矩阵不依赖于临界矩阵$W$对角元素，其无论对角元素是多少，构造出的$L$都是相同的，即图拉普拉斯矩阵与自环无关

定理：
对于边权重非负的无向图$G$，其图拉普拉斯矩阵的0特征值的个数($k$)等于其连通分量($A_1,\dots,A_k$)的个数，特征值0的特征空间由连通分量的指示向量$\mathbb 1_{A_1},\dots, \mathbb 1_{A_k}$张开
证明：
若$k=1$，即0特征值仅一个，设其对应于特征向量$f$，有
$$0 = f^TLf = \sum_{i,j=1}^nw_{ij}(f_i-f_j)^2$$
因为对任意$i,j$，有$w_{ij} \ge 0$，故对任意$i,j$，有$w_{ij}(f_i -f_j)^2 = 0$
若$w_{ij} > 0$，即两个结点$v_i,v_j$相连，则要求$f_i = f_j$

因此，$f$中相连的结点对应的项$f_i$要求是相等的
若$G$中存在连通分量，令连通分量内结点对应的$f_i = 1$，其余结点对应的$f_j = 0$，构造该分量的指示向量$f$
考虑$Lf$中的第$i$项$$Lf_i =\langle L_i, f\rangle = \sum_{j = 1}^n l_{ij} f_j$$当$v_i$不在$f$所指示的连通分量中，
$f$中所有$v_i$和与$v_i$相邻的结点对应的项都等于0，其余都大于等于0
$L_i$中所有$v_i$和与$v_i$相邻的结点对应的项都大于0，其余都等于0
显然$\langle L_i, f\rangle = 0$
当$v_i$在$f$所指示的连通分量中，
$f$中所有$v_i$和与$v_i$相邻的结点对应的项都等于1，其余都等于0
$L_i$中所有$v_i$和与$v_i$相邻的结点对应的项都大于0，其余都等于0
$\langle L_i, f\rangle = d_{i} - \sum_{j=1}^nw_{ij} = 0$
故$$Lf = 0$$
即$G$中的连通分量的指示向量是$L$的0特征值对应的特征向量
显然$L$的0特征值对应的特征向量也应满足指示向量的线性组合的形式

对于不同的连通分量，因为它们不相交，因此其指示向量相互正交，因此有$k$个连通分量，就有$k$个对应的相互正交的特征向量，就对应$k$个独立的0特征值，构成了一个$k$维子空间
而由于所有的连通分量的并集是$V$，因此也不存在第$k+1$个独立的0特征值和对应的特征向量

因此，$G$的$k$个连通分量的指示向量即$L$的$k$个0特征值对应的特征向量，它们相互正交，共同构成了$L$的零空间

## 3.2 The normalized graph Laplacians
规范化的图拉普拉斯矩阵有两种形式
$$\begin{aligned}
L_{sym} &= D^{-\frac 1 2}L D^{-\frac 1 2}\\
&= D^{-\frac 1 2}(D - W)D^{-\frac 1 2}\\
&=I-D^{-\frac 1 2}WD^{-\frac 1 2}\\
L_{rw} &= D^{-1}L \\
&=D^{-1}(D-W)\\
&=I- D^{-1}W
\end{aligned}$$
性质：
(1) 对任意$f\in \mathbb R^n$，有
$$f^TL_{sym}f=\frac 1 2\sum_{i,j=1}^nw_{ij}\left(\frac {f_i}{\sqrt {d_i}}-\frac {f_j}{\sqrt {d_j}}\right)^2$$

证明：
$$\begin{aligned}
f^TL_{sym}f = f^T(I-D^{-\frac 12}WD^{-\frac 12})f &= f^Tf-f^TD^{-\frac 1 2}WD^{-\frac 1 2}f\\
&= \sum_{i=1}^nf_i^2-\sum_{i,j=1}^n \frac {w_{ij}} {\sqrt{d_id_j}} {f_if_j}
\end{aligned}$$
而
$$\begin{aligned}
&\frac 1 2\sum_{i,j=1}^nw_{ij}\left(\frac {f_i}{\sqrt {d_i}}-\frac {f_j}{\sqrt {d_j}}\right)^2\\
=&\frac 1 2\sum_{i,j=1}^nw_{ij}\left(\frac {f_i^2} {d_i}-\frac {2f_if_j}{\sqrt{d_id_j}}+\frac {f_j^2}{d_j}\right)\\
=&\frac 1 2\left(\sum_{i=1}^n\frac {f_i^2}{d_i}\sum_{j=1}^nw_{ij}
-\sum_{i,j=1}^nw_{ij}\frac {2f_if_j}{\sqrt{d_id_j}}
+\sum_{j=1}^n\frac {f_j^2}{d_j}\sum_{i=1}^nw_{ij}\right)\\
=&\frac 1 2\left(\sum_{i=1}^nf_i^2-\sum_{i,j=1}^nw_{ij}\frac {2f_if_j}{\sqrt{d_id_j}}+\sum_{j=1}^nf_j^2\right)\\
=&\sum_{i=1}^nf_i-\sum_{i,j=1}^n\frac {w_{ij}}{\sqrt{d_id_j}} f_if_j
\end{aligned}$$
故
$$f^TL_{sym}f=\frac 1 2\sum_{i,j=1}^nw_{ij}\left(\frac {f_i}{\sqrt {d_i}}-\frac {f_j}{\sqrt {d_j}}\right)^2$$
(2) $\lambda$是$L_{rw}$的特征值，对应的特征向量是$u$，当且仅当$\lambda$是$L_{sym}$的特征值，且对应的特征向量是$w = D^{\frac 1 2}u$

证明：
$\lambda$满足$L_{rw}u = \lambda u$，等式两边左乘$D^{\frac 1 2}$，得到$D^{\frac 1 2}L_{rw}u = \lambda D^{\frac 1 2}u$，即
$$D^{-\frac 1 2}Lu = \lambda D^{\frac 1 2}u$$
故
$$D^{-\frac 1 2}LD^{-\frac 1 2}w = \lambda w$$
即
$$L_{sym}w = \lambda w$$

(3) $\lambda$是$L_{rw}$的特征值，对应的特征向量是$u$，当且仅当$\lambda$和$u$满足$Lu = \lambda Du$

证明：
$\lambda$满足$L_{rw}u = \lambda u$，等式两边左乘$D$，得到$DL_{rw}u = \lambda Du$，即
$$Lu = \lambda Du$$

(4) 0是$L_{rw}$的特征值，对应的特征向量是$\mathbb 1$，0是$L_{sym}$的特征值，对应的特征向量是$D^{\frac 1 2}\mathbb 1$

证明：
$$\begin{aligned}
&L_{rw}\mathbb 1\\
=&D^{-1}L\mathbb 1\\
=&D^{-1}(L\mathbb1)\\
=&D^{-1}\mathbb 0\\
=&0
\end{aligned}$$

(5) $L_{sym}$和$L_{rw}$是半正定矩阵，有$n$个非负实值特征值，满足$0\le \lambda_1 \le \dots \le \lambda_n$

证明：
由(1)可知$L_{sym}$为半正定矩阵，再由(2)可知$L_{rw}$也为半正定矩阵 

定理：
对于边权重非负的无向图$G$，其(规范化的)图拉普拉斯矩阵的0特征值的个数($k$)等于其连通分量($A_1,\dots,A_k$)的个数，对于$L_{rw}$，对应的特征空间由连通分量的指示向量$\mathbb 1_{A_i}$张开，对于$L_{sym}$，其对应的特征空间由$D^{\frac 1 2}\mathbb 1_{A_i}$张开

证明：
对于$L$，有$$L\mathbb 1_{A_i} = 0$$
故
$$(D^{-1}L)\mathbb 1_{A_i}= L_{rw}\mathbb 1_{A_i} = 0$$
即$\mathbb 1_{A_i}$也是$L_{rw}$的0特征向量，且$\mathbb 1_{A_i}$之间相互正交，故其特征值0的特征空间也由$\mathbb 1_{A_i}$张开
由性质(2)可知，$L_{sym}$的特征空间由$D^{\frac 1 2}\mathbb 1_{A_i}$张开

# 4 Spectral Clustering Algorithms
设有$n$个数据点$x_1,\dots,x_n$，我们用$s_{ij} = s(x_i,x_j)$衡量其成对相似度，相似度非负，且对称，用$S$表示相似度矩阵

未规范化的谱聚类算法：
输入：相似度矩阵$S\in \mathbb R^{n\times n}$，需要构建的簇的数量$k$
- 利用相似度矩阵通过特定的方法构建相似图，令$W$为其带权邻接矩阵
- 计算未规范化的图拉普拉斯矩阵$L$
- 计算$L$的前$k$个特征向量$u_1,\dots,u_k$(对应最小的$k$个特征值)
- 构建$U = [u_1,\dots, u_k]\in \mathbb R^{n\times k}$
- 令$y_i\in \mathbb R^k$为$U$的第$i$行向量
- 以$y_i$作为点$v_i$在$\mathbb R^k$空间的特征向量，执行$k$均值聚类算法
输出：$k$个簇$C_1,\dots,C_k$

理解：
$L$的特征向量$f$均满足$$f^TLf=\lambda f^T f=\frac 1 2\sum_{i,j=1}^nw_{ij}\left( {f_i}-{f_j}\right)^2$$若$f$规范化为单位向量，则得到
$$f^TLf =\lambda=\frac 1 2\sum_{i,j=1}^nw_{ij}\left({f_i}-{f_j}\right)^2$$
故最小的$\lambda$对应使$\sum_{i,j=1}^nw_{ij}\left({f_i}-{f_j}\right)^2$最小特征向量$f$
视$f$为每个点$v_i$指派了一个值$f_i$，直观上说，对应特征向量最小的$f$，即是使得点之间的总体加权距离最小的一个总指派，它尽量在$v_i$与$v_j$之间的$w_{ij}$越大时(即$v_i$与$v_j$之间的相似图边权越大时)，令$f_i$与$f_j$越接近，因此$f$是一个最大化保持了原相似图中点之间的临近关系的一维向量
对于半正定矩阵$L$，其特征向量相互正交，因此$k$个特征向量正好构成了一个$k$维空间，其中的每一维度都尽量保持了原相似图中结点之间的临近关系信息，而各维度相互正交，也保证了不存在冗余的信息
因此，在新构建的$k$维空间中，点之间的临近关系信息被最大化保留，在这个空间执行聚类是合理的

根据使用的规范化图拉普拉斯矩阵的不同，规范化的谱聚类算法也有两种

规范化的谱聚类算法-Shi and Mailk(2000)：
输入：相似度矩阵$S\in \mathbb R^{n\times n}$，需要构建的簇的数量$k$
- 利用相似度矩阵通过特定的方法构建相似图，令$W$为其带权邻接矩阵
- 计算未规范化的图拉普拉斯矩阵$L$
- 通过解推广的特征问题$Lu=\lambda Du$，计算$L_{rw}$的前$k$个特征向量$u_1,\dots,u_k$(对应最小的$k$个特征值)
- 构建$U = [u_1,\dots, u_k]\in \mathbb R^{n\times k}$
- 以$y_i$作为点$v_i$在$\mathbb R^k$空间的特征向量，执行$k$均值聚类算法
输出：$k$个簇$C_1,\dots,C_k$

规范化的谱聚类算法-Ng,Jordan and Weiss(2002)
输入：相似度矩阵$S\in \mathbb R^{n\times n}$，需要构建的簇的数量$k$
- 利用相似度矩阵通过特定的方法构建相似图，令$W$为其带权邻接矩阵
- 计算规范化的图拉普拉斯矩阵$L_{sym}$
- 计算$L_{sym}$的前$k$个特征向量$u_1,\dots,u_k$(对应最小的$k$个特征值)
- 构建$U = [u_1,\dots, u_k]\in \mathbb R^{n\times k}$
- 将$U$的行做归一化(每行$L_2$范数为1)，即$u_{ij} = u_{ij} / (\sum_k u_{ik}^w)^{\frac 1 2}$
- 令$y_i\in \mathbb R^k$为$U$的第$i$行向量
- 以$y_i$作为点$v_i$在$\mathbb R^k$空间的特征向量，执行$k$均值聚类算法
输出：$k$个簇$C_1,\dots,C_k$

这三个算法的思路都是将数据点$v_i$的特征从$x_i$转换到$y_i\in \mathbb R^k$，这种转换强化了数据的簇属性(cluster-properties)，因此在新的特征空间能更容易发现簇

# 5 Graph cut point of view
聚类问题可以转化为图分割问题，在不同子集内的点，边的权重较小(相似度低)，在相同的子集内的点，边的权重较大(相似度高)
