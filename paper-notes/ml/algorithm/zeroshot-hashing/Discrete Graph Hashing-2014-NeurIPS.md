[Discrete Graph Hashing](<file:///D:\Learning\paper\2014-NeurIPS-Discrete Graph Hashing.pdf>)
[Supplemental](<file:///D:\Learning\paper\2014-NeurIPS-DGH-Supplemental.pdf>)
# Abstract
无监督哈希学习方法在哈希码长度增大时表现会迅速劣化，作者认为原因来自于为了得到离散哈希码的内部优化步骤

本文提出基于图的无监督哈希模型，优化方式为易解决的交替最大化算法(tractable，即具有最坏时间复杂度为多项式时间的算法，即P问题)
# 1 Introduction
本文聚焦紧凑哈希(compact hashing)，即哈希码长度较短(100 bits)

作者认为离散约束(discrete constraints)在学习过程中未受重视，导致无监督哈希模型在生成相对长一点的哈希码(100 bits)时表现不好
现有方法要么直接忽视了离散约束、要么先不管离散约束，解决松弛的优化问题(relaxed optimizations)，之后再将连续的哈希码向离散值归约(作者发现第二类方法在哈希码长度增大时表现会迅速劣化)

本文提出的优化过程在离散约束下进行，最终得到几乎平衡和不相关(balanced and uncorrelated)的哈希码
# 2 Discrete Graph Hashing
**Anchor Graphs**
Anchor数目$m\ll n$，对$x\in \mathcal X$，定义$z(x)$
$$z(x) = [\delta_1\exp (-\frac {\mathcal D^2(x,u_1)}t),\cdots,\delta_m\exp (-\frac {\mathcal D^2(x,u_m)}t)]^T/M,\quad\delta_j\in\{0,1\},t>0$$
其中$\delta_j=1$当且仅当锚点$u_j$是$x$的$s$最近邻($s\ll m$)之一，$\mathcal D()$为距离函数，例如$\mathcal l_2$距离
其中$M=\sum_{j=1}^m\delta_j\exp(-\frac {\mathcal D(x,u_j)}{t})$，故满足$$\|z(x)\|_1 = 1$$数据-锚点(data-to-anchor)亲和矩阵$Z(X)$为
$$Z(X) = [z(x_1),\cdots,z(x_n)]^T=\begin{bmatrix}z(x_1)^T\\ \vdots \\ z(x_n)^T\end{bmatrix}\in \mathbb R^{n\times m}$$
$Z(X)$高度稀疏，且$Z(X)_{ij} \ge 0$

记真实的亲和矩阵为$A^o\in \mathbb R^{n\times n}$
记$Z(X)$为$Z$
根据$Z$计算数据-数据(data-to-data)亲和矩阵/相似度矩阵$A$，作为$A^o$的近似
$$A = Z\Lambda^{-1}Z^T\in \mathbb R^{n\times n},\quad\Lambda=diag(Z^T1)\in \mathbb R^{m\times m}$$
$\Lambda$中对角线元素为$Z$中的对应列的元素和

故
$$\begin{aligned}
A &= Z\Lambda^{-1}Z^T\\
&=\begin{bmatrix}z(x_1)^T\\ \vdots \\ z(x_n)^T\end{bmatrix}\Lambda^{-1} [z(x_1),\cdots,z(x_n)]\\
&=\begin{bmatrix}z_1^T\\ \vdots \\ z_n^T\end{bmatrix}\Lambda ^{-1}[z_1,\cdots,z_n]\\
\end{aligned}$$
其中$\lambda_i = \sum_{j=1}^n Z_{ji}$

$A$的性质
(1) $A$是低秩半正定矩阵，秩不大于$m$
(2) $A$的行和、列和都是$1$
证明：
对于$A$的行元素和
$$\begin{aligned}
&A1\\
=&(Z\Lambda^{-1}Z^T)1\\
=&Z\Lambda^{-1}(Z^T1)\\
=&Z1\\
=&1\\
=&\begin{bmatrix}1\\ \vdots \\ 1\end{bmatrix}
\end{aligned}$$
对于$A$的列元素和
$$\begin{aligned}
&1^TA\\
=&1^T(Z\Lambda^{-1}Z^T)\\
=&(1^TZ)\Lambda^{-1}Z^T\\
=&1^TZ^T\\
=&1^T\\
=&[1,\cdots,1]
\end{aligned}$$


**Learning Model**
$$B = [b_1,\cdots,b_n]^T\in \{1,-1\}^{n\times r}$$
根据[[paper-notes/ml/algorithm/zeroshot-hashing/Spectral Hashing-2008-NeruIPS]]提出的标准的图哈希框架，学习目标是
$$\min_B\frac 1 2\sum_{i,j=1}^n\|b_i-b_j\|^2A_{ij}^o=tr(B^TL^oB),\ s.t.\ B\in\{\pm1\}^{n\times r},1^TB=0,B^TB=nI_r,(1)$$
其中$L^o$是基于真实亲和矩阵得到的图拉普拉斯矩阵
限制$1^TB$用于最大化每个哈希比特的信息，因为该限制令每个哈希比特都是对数据集$\mathcal X$的一个平衡划分(balanced partition)
限制$B^TB=nI_r$用于最小化哈希比特之间的依赖，因为该限制令$r$个哈希比特互不相关(uncorrelated)
该问题是NP hard问题，在[[paper-notes/ml/algorithm/zeroshot-hashing/Spectral Hashing-2008-NeruIPS]]中，作者丢弃了离散限制$B\in\{\pm 1\}^{n\times r}$，并作了数据均匀分布的假设，以解决一个松弛问题

本文使用Anchor graph的图拉普拉斯矩阵$L = I_n - A$代替$L^o$，故问题写为
$$\min_B tr(B^T(I_n-A)B),\ s.t.\ B\in\{\pm1\}^{n\times r},1^TB=0,B^TB=nI_r$$
而$$tr(B^T(I_n-A)B) = tr(B^TB-B^TAB)=tr(B^TB)-tr(B^TAB)$$
而$B^TB = nI_r$，故$tr(B^TB) = tr(nI_r) = nr$，为常数

故问题重写为
$$\max_Btr(B^TAB),\ s.t.\ B\in\{\pm1\}^{n\times r},1^TB=0,B^TB=nI_r\tag{2}$$
将$B$松弛为实数矩阵，之后通过阈值$0$将其离散化，但该方法在随着哈希码长度$r$增长，放大由松弛条件带来的错误时效果会劣化
本文希望直接解出离散的$B$

定义集合$\Omega = \{Y\in \mathbb R^{n\times r} | 1^TY = 0, Y^TY = nI_r\}$
通过松弛$1^TB = 0,B^TB=nI_r$两个条件，构建更广泛的图哈希框架
$$\max_B tr(B^TAB)-\frac \rho 2 dist(B,\Omega),\ s.t.\ B\in\{1,-1\}^{n\times r}\tag{3}$$
其中$dist(B,\Omega) = \min_{Y\in\Omega}\|B - Y\|_F^2$衡量了矩阵$B$到集合$\Omega$的距离
其中$\rho \ge 0$，为调节参数，控制了条件的松弛程度
如果$\rho$非常大，等价于要求$dist(B,\Omega) = 0$，即不松弛条件

松弛了条件，不一定有$B^TB = nI_r$，但仍然有$tr(B^TB) = tr(Y^TY) = nr$，故
$$\begin{aligned}
&\|B-Y\|_F^2\\
=&tr((B-Y)(B-Y)^T)\\
=&tr((B-Y)(B^T-Y^T))\\
=&tr(BB^T-BY^T-YB^T+YY^T)\\
=&tr(BB^T)+tr(YY^T)-tr(BY^T)-tr(YB^T)\\
=&2nr-2tr(B^TY)\\
=&2(nr-tr(B^TY))
\end{aligned}$$
故$$dist(B,\Omega) = \min_{Y\in\Omega}\|B - Y\|_F=\min_{Y\in\Omega}(2nr-2tr(B^TY))=2nr-2\max_{Y\in\Omega}tr(B^TY)$$
故
$$\begin{aligned}
&\max_B tr(B^TAB)-\frac \rho 2dist(B,\Omega)\\
=&\max_B\left(tr(B^TAB)+\rho \max_{Y\in \Omega}tr(B^TY)-\rho nr\right)\\
=&\max_B\left(tr(B^TAB)+\rho \max_{Y\in \Omega}tr(B^TY)\right)\\
=&\max_{B,Y}tr(B^TAB)+\rho\ tr(B^TY)\\
\end{aligned}$$
因此，问题进一步重写为
$$\begin{aligned}&\max_{B,Y}\mathcal Q(B,Y) = tr(B^TAB)+\rho\ tr(B^TY),\\
&s.t.\ B\in\{1,-1\}^{n\times r},Y\in\mathbb R^{n\times r},1^TY=0,Y^TY=nI_r
\end{aligned}\tag{4}$$
称该问题为离散图哈希(Discrete Graph Hashing/DGH)
同时施加$B\in \{-1,1\}^{n\times r}$和$B\in\Omega$的限制会使得图哈希问题计算上不可解(computational intractable)，DGH选择对$B$到$\Omega$的距离进行惩罚，但保留了对$B$的离散限制
DGH存在计算上可解的算法，可以得到近似平衡和不相关的哈希码

**Out-of-Sample Hashing**
哈希方法需要能为任意在训练数据集$\mathcal X$外的数据点$q\in \mathbb R^d$生成哈希码
在DGH中，我们通过最小化新数据点$q$和它的邻近数据点(根据亲和度矩阵$A$确定)之间的汉明距离得到其哈希码
$$b(q)\in \arg \min_{b(q)\in\{\pm1\}^r} \frac 1 2\sum_{i=1}^n\|b(q)-b_i^*\|^2A(q,x_i)=\arg\max_{b(q)\in\{\pm 1\}^r}\langle b(q),(B^*)^TZ\Lambda^{-1}z(q)\rangle$$
其中$B^* = [b_1^*,\cdots,b_n^*]^T$是训练集的解
其中$A(x,x')$定义为$A(x,x')=z^T(x)\Lambda^{-1}z(x')$

推导：
$$\begin{aligned}
&\sum_{i=1}^n\|b(q)-b_i^*\|^2A(q,x_i)\\
=&\sum_{i=1}^n\|b(q)-b_i^*\|^2z^T(q)\Lambda^{-1}z(x_i)\\
=&\sum_{i=1}^n z^T(q)\Lambda^{-1}\left(\|b(q)-b_i^*\|^2z(x_i)\right)\\
=&z^T(q)\Lambda^{-1}\sum_{i=1}^n\|b(q)-b_i^*\|^2z(x_i)\\
\end{aligned}$$
又因为
$$\begin{aligned}
&\sum_{i=1}^n\|b(q)-b_i^*\|^2z(x_i)\\
=&\sum_{i=1}^n(b(q)-b_i^*)^T(b(q)-b_i^*)z(x_i)\\
=&\sum_{i=1}^n(b^T(q)-(b_i^*)^T)(b(q)-b_i^*)z(x_i)\\
=&\sum_{i=1}^n\left(b^T(q)b(q)-b^T(q)b_i^*-(b_i^*)^Tb(q)+(b_i^*)^Tb_i^*\right)z(x_i)\\
=&\sum_{i=1}^n(\|b(q)\|+\|b_i^*\|-\langle b(q),b_i^*\rangle)z(x_i)\\
=&\sum_{i=1}^n2rz(x_i)-\langle b(q),b_i^*\rangle z(x_i)
\end{aligned}$$
故
$$\begin{aligned}
&\sum_{i=1}^n\|b(q)-b_i^*\|^2A(q,x_i)\\
=&z^T(q)\Lambda^{-1}\sum_{i=1}^n\|b(q)-b_i^*\|^2z(x_i)\\
=&z^T(q)\Lambda^{-1}\sum_{i=1}^n\left(2rz(x_i)-\langle b(q),b_i^*\rangle z(x_i)\right)\\
=&z^T(q)\Lambda^{-1}\sum_{i=1}^n 2rz(x_i)-z^T(q)\Lambda^{-1}\sum_{i=1}^n\langle b(q), b_i^*\rangle z(x_i)
\end{aligned}$$
其中被减数与$b(q)$无关，因此得
$$\begin{aligned}
&\arg\min_{b(q)\in \{\pm 1\}^r}\frac 1 2\sum_{i=1}^n\|b(q)-b_i^*\|^2A(q,x_i)\\
=&\arg\min_{b(q)\in \{\pm 1\}^r}\left( z^T(q)\Lambda^{-1}\sum_{i=1}^n 2rz(x_i)-z^T(q)\Lambda^{-1}\sum_{i=1}^n\langle b(q), b_i^*\rangle z(x_i)\right)\\
=&\arg\max_{b(q)\in \{\pm 1\}^r}z^T(q)\Lambda^{-1}\sum_{i=1}^n\langle b(q), b_i^*\rangle z(x_i)\\
\end{aligned}$$
而
$$\begin{aligned}
&\langle b(q), (B^*)^TZ\Lambda^{-1}z(q)\rangle\\
=&\left((B^*)^TZ\Lambda^{-1}z(q)\right)^Tb(q)\\
=&z^T(q)\Lambda^{-1}Z^TB^*b(q)\\
=&z^T(q)\Lambda^{-1}[z(x_1),\cdots,z(x_n)]\begin{bmatrix}(b_1^*)^T\\ \vdots \\ (b_n^*)^T\end{bmatrix}b(q)\\
=&z^T(q)\Lambda^{-1}[z(x_1),\cdots,z(x_n)]\begin{bmatrix}(b_1^*)^Tb(q)\\ \vdots \\ (b_n^*)^Tb(q)\end{bmatrix}\\
=&z^T(q)\Lambda^{-1}\sum_{i=1}^nz(x_i)(b_i^*)^Tb(q)\\
=&z^T(q)\Lambda^{-1}\sum_{i=1}^n\langle b_i^* , b(q) \rangle z(x_i)\\
\end{aligned}$$
故
$$\begin{aligned}
&\arg \max_{b(q)\in \{\pm1\}^r}\langle b(q), (B^*)^TZ\Lambda^{-1}z(q)\rangle\\
=&\arg \max_{b(q)\in \{\pm1\}^r}z^T(q)\Lambda^{-1}\sum_{i=1}^n\langle b_i^* , b(q) \rangle z(x_i)
\end{aligned}$$
因此我们有
$$ \arg \min_{b(q)\in\{\pm1\}^r} \frac 1 2\sum_{i=1}^n\|b(q)-b_i^*\|^2A(q,x_i)=\arg\max_{b(q)\in\{\pm 1\}^r}\langle b(q),(B^*)^TZ\Lambda^{-1}z(q)\rangle$$
而显然$$b^*(q) = \arg\max_{b(q)\in\{\pm 1\}^r}\langle b(q),(B^*)^TZ\Lambda^{-1}z(q)\rangle=sgn(Wz(q))$$其中$W = (B^*)^TZ\Lambda^{-1}\in \mathbb R^{r\times m}$，该矩阵在训练结束后即可计算得到，用于后续对新数据点$q$的哈希码的计算，该计算是十分高效的
# 3 Alternating Maximization
等式(4)的问题实际上是一个非线性的混合整数规划(nonlinear mixed-integer program)，包含离散变量$B$和连续变量$Y$，因此该问题通常是NP-hard的

因此提出可解决的交替最大化算法以优化问题(4)
算法交替地解决两个子问题，首先是B-子问题(B-subproblem)
$$\max_{B\in\{\pm 1\}^{n\times r}}f(B) = tr(B^TAB)+\rho tr(Y^TB)\tag{5}$$
然后是Y-子问题(Y-subproblem)
$$\max_{Y\in \mathbb R^{n\times r}}tr(B^TY),\ s.t.\ 1^TY=0,Y^TY=nI_r\tag{6}$$

## 3.1 B-Subproblem
对于B-子问题，提出一个迭代式的上升过程，称为有符号梯度方法(Signed Gradient Method)
>输入：$B^{(0)}\in\{1,-1\}^{n\times r},Y\in\Omega$
>$j = 0$
>repeat
>$B^{(j+1)} = sgn(\mathcal C(2AB^{(j)}) + \rho Y, B^{(j)}))$
>$j=j+1$
>until
>$B^{(j)}$收敛
>输出：$B = B^{(j)}$

在第$j$个迭代，定义在点$B^{(j)}$处线性化$f(B)$的局部函数$\hat f_j(B)$，以$\hat f_j(B)$替代$f(B)$进行离散优化，给定$B^{(j)}$，下一个点$B^{(j+1)}$通过$$B^{(j+1)} \in \arg \max_{B\in \{\pm 1\}^{n\times r}}\hat f_j(B^{(j)})=f(B^{(j)})+\langle \nabla f(B^{(j)}),B-B^{(j)}\rangle$$计算得到
由于$\nabla f(B^{(j)})$可能包含$0$项，$B^{(j+1)}$可能存在多解，为了避免该歧义性，引入函数$$\mathcal C(x,y)=\begin{cases}x,x\ne0\\y,x=0\end{cases}$$更新过程改为$$B^{(j+1)} = sgn\left(\mathcal C(\nabla f(B^{(j)},B^{(j)}) \right)=sgn\left(\mathcal C(2AB^{(j)}+\rho Y,B^{(j)})\right)$$因此$\nabla f(B^{(j)})$消失的项不会进行更新

由于$A$是半正定矩阵，$f$是凸函数，因此对任意$B$，有$f(B)\ge \hat f_j(B)$
故$f(B^{(j+1)}) \ge \hat f_j(B^{(j+1)}) \ge \hat f_j(B^{(j)}) = f(B^{(j)})$，
即保证该算法保证有$f(B^{(j+1)})\ge f(B^{(j)})$

## 3.2 Y-Subproblem
记中心化矩阵$J = I_n - \frac 1 n 11^T$，$J$的对角线元素为$\frac {n-1} n$，非对角线元素为$-\frac 1 n$
对$JB$进行奇异值分解得$$JB = U\Sigma V^T = \sum_{k=1}^{r'}\sigma_ku_kv_k^T$$其中$r' \le r$为$JB$的秩，$U = [u_1,\cdots, u_{r'}],V =[v_1,\cdots,v_{r'}]$
可以构造$\bar U \in \mathbb R^{n\times (r - r')},\bar V \in \mathbb R^{r\times(r-r')}$，
满足$\bar U^T \bar U = I_{r-r'},[U1]^T\bar U = 0$，$\bar V^T \bar V = I_{r-r'},V^T\bar V = 0$，可知
$$JB = [U\bar U]\begin{bmatrix}\Sigma&0\\0&0\end{bmatrix}[V\bar V]^T= \sum_{k=1}^{r'}\sigma_ku_kv_k^T$$
 
先给出结论：$$Y^* = \sqrt n [U \bar U][V \bar V]^T$$为Y-子问题的最优解
定义所有形式为$\sqrt n [U \bar U][V \bar V]^T$的矩阵为$\Phi(JB)$

证明(参考Supplemental)：
首先证明$Y$满足等式(6)中的条件
因为$1^TJ = 0$，故$1^TJB = 0$，又因为$U$为$JB$的列空间的一组标准正交基，
因此有$1^TU=0$
(证明：对于任意属于$JB$列空间的向量$f$，可以将其写为$\sum_{i=1}^{r'} c_i u_i,c_i\in \mathbb R$，
因为$1^TJB=0$，故对于任意属于$JB$列空间的向量$f$，有$1^Tf = 0$，即$1^T\sum_{i=1}^{r'}c_i u_i = \sum_{i=1}^{r'}c_i (1^T u_i) = 0$，
由于$c_i$的任意性，可以得出$\forall i\in[1,r'],1^Tu_i = 0$，即$1^TU=0$)

我们构造$\bar U$，使得$1^T\bar U = 0$，则现在有$1^T[U U^T] = 0$，因此有$1^TY^* = 0$
同时$(Y^*)^TY^* = n[V\bar V][U \bar U]^T[U \bar U][V \bar V]^T = nI_r$
因此$Y$满足两项条件

>补充两个相关知识
>von Neumann的迹不等式：
>$A, B$是$n\times n$的复矩阵，奇异值分别为$\alpha_1 \ge \cdots \ge \alpha_n,\beta_1 \ge \cdots \ge \beta_n$，则$$|tr(AB)|\le\sum_{i=1}^n\alpha_i\beta_i$$
>矩阵的内积：
>$A, B$为两个相同形状的矩阵，则
>$$\langle A, B\rangle = \sum_{ij}(A_{ij}\times B_{ij})=tr(A^TB)$$
>如果$A$是对称矩阵，即$A = A^T$，则$$\begin{align}
\langle A,B\rangle&=\sum_{ij}(A_{ij}\times B_{ij})\\
&=\sum_{ij}(A_{ji}\times B_{ji})\\
&=\sum_{ij}(A_{ij}\times B^T_{ij})\\
&=\langle A,B^T\rangle
\end{align}$$

考虑对任意的$Y\in \Omega$，
因为$1^TY = 0$，故$JY = Y - \frac 1 n 11^TY = Y$，
因为$Y^TY = nI_r$，因此$Y$的奇异值都为$\sqrt n$
故$$\langle B, Y \rangle = \langle B, JY \rangle = \langle JB, Y \rangle=tr((JB)^TY)\le\sqrt n\sum_{k=1}^{r'}\sigma_k$$
而对于$Y^*$，有
$$\begin{aligned}
\langle B, Y^* \rangle &=\langle JB, Y^* \rangle\\
&=\langle [U \bar U]\begin{bmatrix}\Sigma&0\\0&0\end{bmatrix}[V\bar V]^T,Y^*\rangle\\
&=tr([V \bar V]\begin{bmatrix}\Sigma&0\\0&0\end{bmatrix}^T[U\bar U]^TY^*)\\
&=\sqrt n\ tr([V\bar V] \begin{bmatrix}\Sigma&0\\0&0\end{bmatrix}^T[V\bar V]^T)\\
&=\sqrt n\langle \begin{bmatrix}\Sigma&0\\0&0\end{bmatrix}^T,I_r\rangle\\
&=\sqrt n \sum_{k=1}^{r'}\sigma_k
\end{aligned}$$
因此我们推出，对于任意$Y\in \Omega$，有
$$\begin{aligned}
tr(B^TY) = \langle B, Y\rangle &= \langle JB, Y \rangle\\
&\le \sqrt n \sum_{k=1}^{r'}\sigma_k\\
&=\langle B, Y^*\rangle\\
&=tr(B^TY^*\rangle
\end{aligned}$$
因此$Y^*$是Y-子问题的最优解

注意到
$$\begin{align}JJ &= (I_n-\frac 1n 11^T)(I_n-\frac 1n 1 1^T)\\ 
&= I_n-\frac 1 n11^T-\frac 1 n 11^T + \frac 1 {n^2}11^T11^T\\
&= I_n -\frac 2 n11^T +\frac 1 {n^2}1(1^T1)1^T\\
&= I_n -\frac 2 n11^T +\frac 1 {n}11^T\\
&=I_n-\frac 1 n 11^T\\
&=J
\end{align}$$
且$J$为对称矩阵，$J = J^T$

考虑$r\times r$大小的矩阵$B^TJB$：
$$\begin{aligned}
&B^TJB\\
=&B^TJJB\\
=&B^TJ^TJB\\
=&(JB)^T(JB)\\
	=&[V\bar V]\begin{bmatrix}\Sigma^2&0\\0&0\end{bmatrix}[V\bar V]^T
\end{aligned}$$
而$[V\bar V]$为$r\times r$的正交阵，显然这是$B^TJB$的一个特征值分解

因此，要计算$Y^*$，可以先对较小的$r\times r$的矩阵$B^TJB$进行特征值分解，得到$V, \bar V, \Sigma$，然后计算$U = JBV\Sigma^{-1}$，而$\bar U$则首先设为一个随机的矩阵，然后Gram-Schmidt正交化以得到满足要求的形式

当$r = r'$时，即$JB$列满秩时，可以知道$Y^*$为唯一的最优解
(注意当$r = r'$时，$\bar U,\bar V$都为$0$)

## 3.3 DGH Algorithm
总结该交替最大化算法，命名为DGH(Discrete Graph Hashing)

>输入：$B_0\in \{1, -1\}^{n\times r}$和$Y_0\in \Omega$
>$k=0$
>repeat
>$B_{k+1} = SGM(B_k, Y_k)$
>$Y_{k+1}\in \Phi(JB_{k+1})$
>$k = k+1$
>until
>$\mathcal Q(B_k,Y_k)$收敛
>输出：$B^* = B_k,Y^* = Y_k$

其中$SGM(\cdot,\cdot)$表示有符号梯度算法

DGH对任意合法的起始点$(B_0,Y_0)$都会收敛

**Initialization**
因为DGH算法处理的是离散的非凸优化，因此一个好的起始点选择十分重要，在此建议两个起始点

首先对$A$进行特征值分解$$A = P\Theta P^T=\sum_{k=1}^m\theta_kp_kp_k^T$$其中$\Theta = diag(\theta_1,\cdots,\theta_m)$，$P = [p_1,\cdots, p_m]$，
$\theta_1,\cdots,\theta_m$为非升序排列的特征值，$p_1,\cdots,p_m$为对应的规范化的特征向量
其中$\theta_1 = 1$，对应的特征向量为$p_1 = 1/\sqrt n$

第一个使用的起始点为$(Y_0 = \sqrt n H, B_0 = sgn(H))$，其中$H = [p_2, \cdots, p_{r+1}]\in \mathbb R^{n\times r}$

另外，$Y_0$也可以由$H$的列空间中的一系列正交向量构成，即$Y_0 = \sqrt n HR$，其中$R$为正交阵，$R\in \mathbb R^{r\times r}$
而$R$和$B_0$可以通过解一个新的离散优化问题得到
$$\max_{R,B_0}tr(R^TH^TAB_0),\ s.t.\ R\in\mathbb R^{r\times r},RR^T = I_r,B_0\in\{1,-1\}^{n\times r}\tag{8}$$

该问题来自于如下引理：
对任意正交矩阵$R\in \mathbb R^{r\times r}$和任意二元矩阵$B\in \{1, -1\}^{n\times r}$，
有$tr(B^TAB)\ge \frac 1 r tr^2(R^TH^TAB)$

该引理表明等式(8)可以解释为最大化$tr(B^TAB)$的一个下界
问题(8)的优化同样可以采用交替最大化算法解决，注意到$AH = H\bar \Theta$，其中$\bar \Theta = diag(\theta_2,\cdots, \theta_{r+1})$，因此等式(8)的目标等于$tr(R^T\bar \Theta H^TB_0)$
从$R^0 =I_r$开始，然后更新$B_0^{j+1} = sgn(H\bar \Theta R^j)$，然后更新$R^{j+1} = \bar U_j \bar V_j^T$，其中$\bar U_j, \bar V_j \in \mathbb R^{r\times r}$源于矩阵$\bar \Theta H^T B_0^j$的完全奇异值分解$\bar U_j \bar \Sigma_j \bar V_j^T$

经验上，第二种初始化方法的目标值好于第一种，因为它意在最大化目标值$\mathcal (B_0, Y_0)$的第一项的下界
# 4 Discussions

# 5 Experiments

# 6 Conclusion

# Appendix 1
**本文考虑的目标问题是**
$$\begin{aligned}&\max_{B,Y}tr(B^TAB)+\rho\ tr(B^TY),\\
&s.t.\ B\in\{1,-1\}^{n\times r},Y\in\mathbb R^{n\times r},1^TY=0,Y^TY=nI_r
\end{aligned}$$
其中$B$为哈希码矩阵，$A$为亲和度矩阵
通过迭代算法求解
(一) 固定$Y$，问题转化为
$$\max_{B\in\{\pm 1\}^{n\times r}}f(B) = tr(B^TAB)+\rho tr(Y^TB)$$
$B$的解为
$$B^{(j+1)} = sgn\left(\mathcal C(\nabla f(B^{(j)},B^{(j)}) \right)=sgn\left(\mathcal C(2AB^{(j)}+\rho Y,B^{(j)})\right)$$
(二) 固定$B$，问题转化为
$$\max_{Y\in \mathbb R^{n\times r}}tr(B^TY),\ s.t.\ 1^TY=0,Y^TY=nI_r$$
$Y$的解为
$$Y^* = \sqrt n [U \bar U][V \bar V]^T$$
其中$U,V$来自于$JB$的奇异值分解
**考虑问题是$F$范数的形式(1)**
$$\begin{align}
&\min_{B}\|BB^T-rA\|_F^2+\frac \rho 2 dist(B,\Omega)\\
&s.t.\ B\in\{1,-1\}^{n\times r},Y\in\mathbb R^{n\times r},1^TY=0,Y^TY=nI_r\end{align}$$
转化问题为
$$\begin{aligned}
&\min_{B} \|BB^T-rA\|_F^2+\frac \rho 2dist(B,\Omega)\\
=&\min_B \|BB^T-rA\|_F^2-\rho \max_{Y\in \Omega}tr(B^TY)\\
=&\min_B \|BB^T-rA\|_F^2-\rho\max_{Y\in \Omega}tr(B^TY)\\
=&\min_{B,Y}\|BB^T-rA\|_F^2-\rho\ tr(B^TY)\\
\end{aligned}$$
则
$$\begin{align}
&\min_{B,Y}\|BB^T-rA\|_F^2-\rho tr(B^TY)\\
=&\min_{B,Y}tr((BB^T-rA)(BB^T-rA)^T)-\rho tr(B^TY)\\
=&\min_{B,Y}tr((BB^T-rA)(BB^T-rA^T))-\rho tr(B^TY)\\
=&\min_{B,Y}tr(BB^TBB^T-rBB^TA^T-rABB^T+r^2AA^T)-\rho tr(B^TY)\\
=&\min_{B,Y}tr(BB^TBB^T)-rtr(BB^TA^T)-rtr(ABB^T)+r^2tr(AA^T)-\rho tr(B^TY)\\
=&\min_{B,Y}tr(BB^TBB^T)-2rtr(BB^TA)-\rho tr(B^TY)
\end{align}$$
其中
$$\begin{align}
&tr(BB^TA)\\
=&tr(ABB^T)\\
=&tr((AB)B^T)\\
=&\langle (AB)^T, B^T\rangle\\
=&\langle B^T, (AB)^T\rangle\\
=&\langle B, AB \rangle\\
=&tr(B^TAB)
\end{align}$$
考虑$tr((AB)B^T)$和$tr(B^T(AB))$，认为其等价
则
$$\begin{align}
&\min_{B,Y}\|BB^T-rA\|_F^2-\rho tr(B^TY)\\
=&\min_{B,Y}tr(BB^TBB^T)-2rtr(BB^TA)-\rho tr(B^TY)\\
=&\min_{B,Y}tr(BB^TBB^T)-(2rtr(B^TAB)+\rho tr(B^TY))
\end{align}$$
这是个四次优化问题，无法直接求解，在考虑问题是$F$范数的形式(2)中，引入了$B$的连续近似$Y$，以降低问题的难度

通过迭代算法求解
(一) 固定$Y$，问题转化为
$$\begin{align}
&\min_{B\in\{-1,1\}^{n\times r}}tr(BB^TBB^T)-2rtr(BB^TA)-\rho tr(B^TY)\\
\end{align}$$
以下为猜想
$f(B) = tr(BB^TBB^T) -2rtr(BB^TA)-\rho tr(B^TY)$
$\nabla f(B) =4B -2r(BA+B^TA)-\rho Y$
$B = sgn(4B-2r(B+B^T)A-\rho Y)$

(二) 固定$B$，问题转化为
$$\max_{Y\in \mathbb R^{n\times r}}tr(B^TY),\ s.t.\ 1^TY=0,Y^TY=nI_r$$
$Y$的解为
$$Y^* = \sqrt n [U \bar U][V \bar V]^T$$
其中$U,V$来自于$JB$的奇异值分解

**考虑问题是$F$范数的形式(2)**
$$\begin{align}
&\min_{B,Y}\|BY^T-rA\|_F^2,\\
&s.t.\ B\in\{1,-1\}^{n\times r},Y\in\mathbb R^{n\times r},1^TY=0,Y^TY=nI_r
\end{align}$$
则
$$\begin{align}
&\min_{B,Y}\|BY^T-rA\|_F^2\\
=&\min_{B,Y}tr((BY^T-rA)(BY^T-rA)^T)\\
=&\min_{B,Y}tr((BY^T-rA)(YB^T-rA^T))\\
=&\min_{B,Y}tr(BY^TYB^T-rBY^TA^T-rAYB^T+r^2AA^T)\\
=&\min_{B,Y}ntr(BB^T)-rtr(BY^TA^T)-rtr(AYB^T)+r^2tr(AA^T)\\
=&\min_{B,Y}n^2r-2rtr(BY^TA^T)+r^2tr(AA^T)\\
=&\max_{B,Y}tr(BY^TA^T)\\
=&\max_{B,Y}tr(BY^TA)
\end{align}$$
通过迭代算法求解
(一) 固定$Y$，得到
$$\begin{align}
&\max_{B}tr(BY^TA)\\
=&\max_B tr(B(Y^TA))\\
=&\max_B \langle B^T,Y^TA\rangle\\
=&\max_B \langle B,A^TY\rangle
\end{align}$$
故问题转化为
$$\max_B\langle B, A^TY\rangle,\ s.t.\ B\in\{-1,1\}^{n\times r}$$
则$B$的解为
$$B = sgn(A^TY)=sgn(AY)$$

(二) 固定$B$，得到
$$\begin{align}
&\max_{Y}tr(BY^TA)\\
=&\max_{Y} tr(A^TYB^T)\\
=&\max_Y \langle A,YB^T\rangle\\
=&\max_Y \langle A, BY^T\rangle\\
=&\max_Ytr(ABY^T)
\end{align}$$
则问题转化为
$$\max_Ytr(ABY^T),\ s.t.\ 1^TY = 0,Y^TY=nI_r$$
则$Y$的解为
$$Y^* = \sqrt n [U \bar U][V \bar V]^T$$
其中$U,V$来自于$JAB$的奇异值分解

# Appendix 2
形式一
$$\begin{align}
&\min_B tr(B^TLB)\\
=&\min_B tr(B^TDB)-tr(B^TAB)
\end{align}$$
形式二
$$\begin{align}
&\min_B \|BB^T-rA\|_F^2\\
=&\min_B tr(BB^TBB^T)-rtr(BB^TA^T)-rtr(ABB^T)+r^2tr(AA^T)\\
=&\min_B tr(BB^TBB^T)-2rtr(ABB^T)+r^2tr(AA^T)\\
=&\min_B tr(BB^TBB^T)-2rtr(B^TAB)
\end{align}$$
形式三
$$\begin{align}
&\min_{B,Z} \|BZ^T-rA\|_F^2\\
=&\min_{B,Z} tr(BZ^TZB^T)-rtr(BZ^TA^T)-rtr(AZB^T)+r^2tr(AA^T)\\
=&\min_{B,Z} tr(BB^T)-2rtr(AZB^T)+r^2tr(AA^T)\\
=&\min_{B,Z} tr(BB^T)-2rtr(AZB^T)\\
=&\min_{B,Z} tr(BB^T)-2rtr(B^TAZ)\\
\end{align}$$

推测：F范数和平均汉明距离在优化时的差异仅在权重上

# Appendix 3
$$\begin{align}
&\min_B  tr(B^T L B)\\
=&\min_B tr(B^TDB)-tr(B^TAB)\\
&s.t.B\in\{\pm 1\}^{n\times r},B^TB = nI_r,\symbfit 1^TB = \symbfit0^T\\\\
& with\ B = sgn(\Phi(X)W),let\ \Phi(X) = \Phi,\\
&let\ Z=\Phi W\ be\ the\ continous\ approximation\ of\ B\\\\
&\min_{W} tr(Z^TLZ)\\
=&\min_Wtr((\Phi W)^TL(\Phi W))\\
=&\min_Wtr(W^T\Phi^TL\Phi W)\\
=&\min_W tr(W^T\Phi^TD\Phi W)-tr(W^T\Phi^TA\Phi W)\\
&s.t.W^T\Phi^T\Phi W = nI_r,\symbfit 1^T\Phi W = \symbfit0^T\\
\\
&with\ A\ normalized,D =I\\
\\
&\max_Wtr(W^T\Phi^TA\Phi W)\\
\\
&let\ Z = P\Phi = \Phi W\\
\\
&\min_{W,P}tr(W^T\Phi^TAP\Phi)\\
&s.t.W^T\Phi^TP\Phi = nI_r,\symbfit 1^TP\Phi = \symbfit 0^T
\end{align}$$


类原型的初始化可以用类中心，类中心来自于特征空间，之后让哈希码向类中心靠拢，可以强化汉明空间和特征空间的联系

相似度矩阵的设计，让相似的项为1，不相似的项为-1，$rS$
相似的项，其哈希码完全一致，哈希码的内积是$r\times 1 = r$
不相似的项，其哈希码完全相反，但是，对于任何一个哈希码，和它完全相反的的哈希码有且仅有一个
类一和类二完全相反，类一和类三完全相反，则类二和类三完全相同

一个$r$位的哈希码，
和它$r$位都不同的哈希码仅存在1个，内积为$-r$
和它$r-1$位不同的哈希码存在$r$个，内积为$-r+2$
和它1位不同的哈希码存在$r$个，内积为$r-2$
和它完全相同的哈希码仅存在1个，内积为$r$

一个$r$维的离散空间中的任意一个向量，它和空间内其余向量的内积是离散值，
$[-r, -r+2, \cdots , r-2, r]$，一共有$r+1$种可能性

