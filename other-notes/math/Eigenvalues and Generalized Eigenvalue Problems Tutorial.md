# Abstract
# 1 Introduction
特征值问题中，一个矩阵的特征向量表示了该矩阵最重要和最具信息的向量
# 2 Introducing Eigenvalue and Generalized Eigenvalue Problems
## 2.1 Eigenvalue Problem
一个对称矩阵$A\in \mathbb R^{d\times d}$的特征值问题定义为
$$A\phi_i = \lambda_i\phi_i,\forall i\in\{1,\cdots,d\}\tag{1}$$
矩阵形式写为
$$A\Phi = \Phi\Lambda\tag{2}$$
其中$\Phi = [\phi_1,\cdots, \phi_d]\in \mathbb R^{d\times d}$，即$\Phi$的列由特征向量构成，$\phi_i\in \mathbb R^d$
$\Lambda = diag([\lambda_1,\cdots,\lambda_d]^T)\in \mathbb R^{d\times d}$，即$\Lambda$是特征值构成的对角矩阵，$\lambda_i \in \mathbb R$

对于特征值问题，矩阵$A$不要求对称，如果$A$对称，则它的特征向量相互正交，反之不然

公式二可以重写为
$$\begin{align}
A\Phi = \Phi \Lambda &\Rightarrow A\Phi\Phi^T = \Phi\Lambda\Phi^T\\
&\Rightarrow A = \Phi\Lambda\Phi^T = \Phi\Lambda \Phi^{-1}
\end{align}\tag{3}$$
其中因为$A$对称，故其特征向量相互正交，因此$\Phi$是正交矩阵，
故$\Phi^T = \Phi^{-1}$，且$\Phi\Phi^T = I$
注意对于正交矩阵$\Phi$，我们总有$\Phi^T\Phi = I$，但$\Phi\Phi^T = I$要在$\Phi$中的全部列向量都存在，即$\Phi$保持为方阵时才成立

公式三被称为特征分解\特征值分解\谱分解

## 2.2 Generalized Eigenvalue Problem
关于两个对称矩阵$A \in \mathbb R^{d\times d}, B \in \mathbb R^{d\times d}$的广义特征值问题定义为
$$A\phi_i = \lambda_iB\phi_i,\forall i\in\{1,\cdots, d\}\tag{4}$$
矩阵形式写为
$$A\Phi = B\Phi\Lambda\tag{5}$$
公式四和五的广义的特征值问题记为$(A, B)$，
$(A, B)$可以称为“对”(pair)，“对”的是顺序相关的，$\Phi,\Lambda$称为特征对$(A,B)$的广义的特征向量和特征值，$(\Phi,\Lambda)$或$(\phi_i,\lambda_i)$称为$(A,B)$的“特征对”(eigenpair)

而特征值问题就是广义的特征值问题在$B = I$时的特殊情况
# 3 Eigenvalue Optimization
## 3.1 Optimization Form 1
考虑如下关于$\phi \in \mathbb R^d$的优化问题
$$\begin{align}
&\max_{\phi}\phi^TA\phi\\
&s.t.\phi^T\phi = 1
\end{align}\tag{6}$$
其中$A\in \mathbb R^{d\times d}$

公式六作为一个带约束优化问题，它的拉格朗日函数为
$$\mathcal L = \phi^TA\phi-\lambda(\phi^T\phi-1)$$
其中$\lambda \in R$为拉格朗日乘子

拉格朗日式函数的导函数为
$$\frac {\partial \mathcal L}{\partial \phi} = 2A\phi - 2\lambda \phi$$
令其为零，得到
$$A\phi = \lambda\phi$$
因此满足该式的解就是矩阵$A$的特征向量，并且拉格朗日乘子是对应的特征向量

显然让拉格朗日函数的导数为零的解的数量和$A$的特征向量的数量相同，而其中可以最大化$\phi^TA\phi$的特征向量$\phi$显然是最大的特征值对应的特征向量，如果这是一个最小化问题，则解就是最小的特征值对应的特征向量

值得注意的是，公式六中的限制的常数不限定为1，而可以是任意常数，当我们取拉格朗日式相对于优化参数的导数时，任意常数相对于$\phi$的参数都将为零，因此不影响最后的求解

## 3.2 Optimization Form 2
考虑以下关于$\Phi \in \mathbb R^{d\times d}$的优化问题
$$\begin{align}&\max_{\Phi}tr(\Phi^TA\Phi)\\&s.t.\Phi^T\Phi =I\end{align}\tag{7}$$
其中$A\in \mathbb R^{d\times d}$

注意，考虑矩阵迹的性质，我们有
$tr(\Phi^T(A\Phi)) = \langle \Phi, A\Phi \rangle = \langle (A\Phi)^T, \Phi^T \rangle = tr(A\Phi\Phi^T)$
$tr((\Phi^TA)\Phi)) = \langle (\Phi^TA)^T, \Phi \rangle = \langle \Phi^T, \Phi^TA \rangle = tr(\Phi\Phi^TA)$
因此目标函数可以是$tr(\Phi^TA\Phi), tr(A\Phi\Phi^T),tr(\Phi\Phi^TA)$中的任意一个

公式七的拉格朗日函数为
$$\mathcal L =tr(\Phi^TA\Phi)-tr(\Lambda^T(\Phi^T\Phi-I))\tag{8}$$
其中$\Lambda \in \mathbb R^{d\times d}$为对角元素为拉格朗日乘子的对角矩阵

拉格朗日式函数相对参数的导函数为
$$\frac {\partial \mathcal L}{\partial \Phi} = 2A\Phi-2\Phi\Lambda$$
令其等于零，得到
$$A\Phi = \Phi \Lambda\tag{9}$$
显然公式九就是$A$的特征问题，满足公式九的$\Phi$即$A$的特征向量矩阵，而$\Lambda$即为$A$的特征值对角矩阵

如果公式七是最大化问题，则$\Lambda$中的特征值和$\Phi$中的特征向量从大到小排序，如果是最小化问题，则从小到大排序

## 3.3 Optimization Form 3
考虑以下关于$\phi \in \mathbb R^d$的优化问题
$$\begin{align}&\min_{\Phi}\|X-\phi\phi^TX\|_F^2\\&s.t.\phi^T\phi =1\end{align}\tag{10}$$
其中$X\in \mathbb R^{d\times n}$

> 矩阵的迹的循环不变性(cyclic invariance property)
> 有任意两个矩阵$A \in \mathbb R^{n \times d}, B \in \mathbb R^{d\times n}$
> 考虑二者乘积的迹$$\begin{align}&tr(AB)\\=&\sum_{i=1}^n (\sum_{j=1}^dA_{ij}B_{ji})\end{align}$$考虑颠倒顺序$$\begin{align}&tr(BA)\\=&\sum_{i=1}^d(\sum_{j=1}^n B_{ij}A_{ji})\end{align}$$显然有$$tr(AB) = tr(BA)$$

公式十中的目标函数可以重写为
$$\begin{align}
&\| X - \phi\phi^TX\|_F^2\\
=&tr((X-\phi\phi^TX)^T(X-\phi\phi^TX))\\
=&tr((X^T-X^T\phi\phi^T)(X-\phi\phi^TX))\\
=&tr(X^TX-2X^T\phi\phi^TX+X^T\phi\phi^T\phi\phi^TX)\\
=&tr(X^TX-X^T\phi\phi^TX)\\
=&tr(X^TX)-tr(X^T\phi\phi^TX)\\
=&tr(X^TX)-tr(XX^T\phi\phi^T)\\
=&tr(X^TX-XX^T\phi\phi^T)
\end{align}$$
则拉格朗日函数为
$$\begin{align}
\mathcal L =& \|X - \phi\phi^TX\|_F^2-\lambda(\phi^T\phi-1)\\
=&tr(X^TX)-tr(XX^T\phi\phi^T)-\lambda(\phi^T\phi-1)
\end{align}$$
它相对于优化参数的导函数为
$$\frac {\partial \mathcal L}{\partial \phi} = 2XX^T\phi-2\lambda\phi$$
令其为零，得到
$$XX^T\phi = \lambda \phi$$
我们令$A = XX^T \in \mathbb R^{d\times d}$，则该式进一步写为
$$A\phi = \lambda\phi$$
显然$A$是对称矩阵，该式即$A$的特征值问题

## 3.4 Optimization Form 4
考虑以下关于$\Phi \in \mathbb R^{d\times d}$的优化问题
$$\begin{align}&\max_{\Phi}\|X-\Phi\Phi^TX\|_F^2\\ 
&s.t.\Phi^T\Phi = I\end{align}\tag{11}$$
其中$X\in \mathbb R^{d\times n}$

目标函数同样可以重写为
$$\|X-\Phi\Phi^TX\|_F^2 = tr(X^TX-XX^T\Phi\Phi^T)$$
拉格朗日式函数
$$\begin{align}
\mathcal L =& \|X-\Phi\Phi^TX\|_F^2-tr(\Lambda^T(\Phi^T\Phi-I))\\
=&tr(X^TX)-tr(XX^T\Phi\Phi^T)-tr(\Lambda^T(\Phi^T\Phi-I))
\end{align}$$
其中$\Lambda \in \mathbb R^{d\times d}$为对角矩阵，包含了拉格朗日乘子

拉格朗日函数相对于优化参数的导函数为
$$\frac {\partial \mathcal L}{\partial \Phi} = 2XX^T\Phi - 2\Phi\Lambda$$
令其等于零，得到
$$XX^T\Phi = \Phi\Lambda$$
我们令$A = XX^T \in \mathbb R^{d\times d}$，则该式进一步写为
$$A\Phi = \Phi\Lambda$$
显然$A$是对称矩阵，该式即$A$的特征值问题

## 3.5 Optimization Form 5
考虑以下关于$\phi \in \mathbb R^{d}$的优化问题
$$\max_{\phi}\frac {\phi^TA\phi}{\phi^T\phi}\tag{12}$$
根据Rayleigh-Ritz quotient方法，该优化问题可以重写为
$$\begin{align}
&\max_{\phi}\phi^TA\phi\\
&s.t.\phi^T\phi = 1
\end{align}\tag{13}$$
此时和优化形式一一致

# 4 Generalized Eigenvalue Optimization
上一节介绍了最终引出特征值问题的优化形式，本节介绍最终引出广义特征值问题的优化形式
## 4.1 Optimization Form 1
考虑以下关于$\phi \in \mathbb R^{d}$的优化问题
$$\begin{align}
&\max_{\phi}\phi^TA\phi\\
&s.t.\phi^TB\phi = 1
\end{align}\tag{14}$$
其中$A\in \mathbb R^{d\times d}, B\in \mathbb R^{d\times d}$

拉格朗日函数为
$$\mathcal L = \phi^TA\phi-\lambda(\phi^TB\phi-1)$$
其中$\lambda \in \mathbb R$为拉格朗日乘子

拉格朗日函数相对于优化参数的导函数为
$$\frac {\partial \mathcal L}{\partial \phi} = 2A\phi-2\lambda B\phi$$
令其为零，得到
$$A\phi = \lambda B\phi$$
显然这是一个关于$(A, B)$的广义特征值问题，$\phi$为特征向量，$\lambda$为特征值

满足拉格朗日函数相对于其导数为零的$\phi$，即满足$A\phi = \lambda B\phi$，代入原式$\phi^TA\phi$可以得到$\phi^T\lambda B\phi = \lambda(\phi^TB\phi) = \lambda$，
因此，如果原问题是一个最大化问题，则解就是最大特征值对应的特征向量，
如果原问题是一个最小化问题，则解就是最小特征值对应的特征向量

比较公式十四和公式六可以知道公式六就是公式十四中$B  = I$的特殊情况

## 4.2 Optimization Form 2
考虑以下关于$\Phi \in \mathbb R^{d\times d}$的优化问题
$$\begin{align}&\max_{\Phi}tr(\Phi^TA\Phi)\\&s.t.\Phi^TB\Phi =I\end{align}\tag{15}$$
其中$A\in \mathbb R^{d\times d}, B\in \mathbb R^{d\times d}$
而根据矩阵迹的性质，目标函数可以是$tr(\Phi^TA\Phi)=tr(A\Phi \Phi^T)=tr(\Phi\Phi^TA)$中的任意一个

拉格朗日函数为
$$\mathcal L = tr(\Phi^TA\Phi)-tr(\Lambda^T(\Phi^TB\Phi- I))$$
其中$\Lambda \in \mathbb R^{d\times d}$为对角矩阵，对角线元素为对应的拉格朗日乘子

拉格朗日函数相对于优化参数的导函数为
$$\frac {\partial \mathcal L}{\partial \Phi} = 2A\Phi-2B\Phi\Lambda$$
令其为零，得到
$$A\Phi = B\Phi\Lambda$$
显然，这是一个关于$(A, B)$的广义特征值问题，其中$\Phi$为对应的特征向量矩阵，$\Lambda$为特征值矩阵

如果原问题是一个最大化问题，则$\Phi$中的特征向量按特征值大小从大到小排序，如果原问题是一个最小化问题，则$\Phi$中的特征向量按特征值大小从小到大排序，$\Lambda$中的特征值同理

## 4.3 Optimization Form 3
考虑以下关于$\phi \in \mathbb R^d$的优化问题
$$\max_{\phi}\frac {\phi^TA\phi}{\phi^TB\phi}\tag{16}$$
根据广义的Rayleigh-Ritz quotient方法，该优化问题可以重写为
$$\begin{align}
&\max_{\phi}\phi^TA\phi\\
& s.t. \phi^TB\phi = 1
\end{align}\tag{17}$$
该形式与优化形式一相同
# 5 Examples for the Optimization Problems
## 5.1 Examples for Eigenvalue Problem
### 5.1.1 Variance in Principle Component Analysis
PCA中，如果我们想将数据集投影到一个向量上(一维PCA子空间)，问题写为
$$\begin{align}
&\max_u u^TSu\\
&s.t.u^Tu = 1
\end{align}\tag{18}$$
其中$u$为投影向量，$S$为协方差矩阵
因此所求的$u$即为$S$最大特征值的特征向量

如果需要投影到由多个向量张成的PCA子空间上，问题写为
$$\begin{align}
&\max_U tr(U^TSU)\\
&s.t. U^TU = I
\end{align}\tag{19}$$
其中$U$的列向量张成了PCA子空间

### 5.1.2 Reconstruction in Pinciple Component Analysis
从另一个角度出发，PCA即拥有最小的重构损失的最优的线性投影
如果我们有一个PCA方向向量$u$，则数据集在该方向上的投影就是$u^TX$，而从这个方向重构的数据集就是$uu^TX$，要最小化原数据集和重构的数据集之间的误差，即$$\begin{align}
&\min_u \|X - uu^TX\|_F^2\\
&s.t. u^Tu = 1
\end{align}\tag{20}$$ 因此，所求解$u$即协方差矩阵$S  = XX^T$(其中$X$已经中心化过)的特征向量

如果要考虑多个PCA方向，则
$$\begin{align}
&\min_u \|X - UU^TX\|_F^2\\
&s.t. U^TU = I
\end{align}\tag{21}$$
其中所求解$U$的列向量就是协防差矩阵$S = XX^T$的特征向量
## 5.2 Examples for Generalized Eigenvalue Problem
### 5.2.1 Kernel Supervised Principle Component Analysis
核监督PCA(KSPCA)采用如下优化问题
$$\begin{align}
&\max_{\Theta} tr(\Theta^TK_xHK_yHK_x\Theta)\\
&s.t.\Theta^TK_x\Theta = I
\end{align}\tag{21}$$
其中$K_x,K_y$分别是训练数据和训练数据的标签的核矩阵，而$H = I - \frac 1 n 11^T$是中心化矩阵，所求解$\Theta$的列向量张成了核SPCA空间

根据公式十五，公式二十一的解为
$$K_xHK_yHK_x\Theta = K_x\Theta\Lambda\tag{23}$$
即关于$(K_xHK_yHK_x,K_x)$的广义特征值问题，$\Theta, \Lambda$是对应的特征向量矩阵和特征值矩阵

### 5.2.2 Fisher Discriminant Analysis
FDA中，目标在于最大化Fisher criterion
$$\max_w\frac {w^TS_B w}{w^TS_W w}\tag{24}$$
其中$w$是投影方向，$S_B,S_W$是类别之间和类别之内的scatter
$$S_B = \sum_{j=1}^c(u_i-u_t)(u_i-u_t)^T\tag{25}$$
$$S_W = \sum_{j=1}^c\sum_{i=1}^{n_j}(x_{j,i}-u_i)(x_{j,i}-u_i)^T\tag{26}$$
其中$c$是类别数量，$n_j$是第$j$个类别的样本数，$x_{j,i}$是第$j$个类的第$i$个数据点，$u_i$是第$i$个类别的均值向量，$u_t$为全体的均值向量

根据Rayleigh-Ritz quotient方法，将公式二十四中的问题重写为
$$\begin{align}
&\max_w w^TS_Bw\\
&s.t.w^TS_Ww=1
\end{align}\tag{27}$$
拉格朗日函数是
$$\mathcal L = w^TS_Bw-\lambda(w^TS_Ww-1)$$
拉格朗日函数相对参数的导函数是
$$\frac {\partial \mathcal L}{\partial w}=2S_Bw-2\lambda S_Ww$$
令其等于零得到
$$S_B w = \lambda S_W w$$
即一个关于$(S_B, S_W)$的广义特征值问题

解$w$是该问题最大特征值对应的特征向量
# 6 Solution to Eigenvalue Problem
考虑公式一
$$A\phi_i = \lambda_i\phi_i\Rightarrow (A-\lambda_iI)\phi_i = 0\tag{28}$$
这是一个线性方程组，根据克莱默法则(Cramer's rule)，一个线性方程组有非平凡解(non-trivial solution)当且仅当它的行列式等于零，因此
$$\det(A-\lambda_iI) = 0\tag{29}$$
公式29给出了一个$d$维多项式等式，有$d$个根/答案
注意如果$A$非满秩，有一些根将会是$0$，另外，如果$A$半正定，所有的根非负

公式29的根就是$A$的特征值，解出公式29的根后，我们将其代入公式28，找到它对应的特征向量$\phi_i \in \mathbb R^d$
注意我们解出的特征向量可以归一化，因为特征向量在于其方向而非数量级，该方向上的数量级信息存在于特征值中
# 7 Solution to Generalized Eigenvalue Problem
回忆公式16
$$\max_{\phi}\frac {\phi^TA\phi}{\phi^TB\phi}$$
令$\rho$为Rayleigh quotient
$$\rho(u;A,B)=\frac {u^TAu}{u^TBu},\forall u \ne 0\tag{30}$$
$\rho$在$\phi \ne 0$时为stationary当且仅当
$$(A-\lambda B)\phi = 0\tag{31}$$
对于一些标量$\lambda$成立
公式31也是一个线性方程组，这个线性方程组也可以从公式4中得到
$$A\phi_i = \lambda B\phi_i \Rightarrow (A-\lambda B)\phi_i = 0\tag{32}$$
同样，根据克莱默法则(Cramer's rule)，一个线性方程组有非平凡解(non-trivial solution)当且仅当它的行列式等于零，因此
$$\det(A-\lambda B) = 0\tag{33}$$
我们解公式33的根，但注意公式33的形式来自于公式4或公式16，仅考虑了一个特征向量$\phi$

如果要解公式5，存在两种解法
## 7.1 The Quick & Dirty Solution
考虑公式5
$$A\Phi = B\Phi \Lambda$$
如果$B$可逆，表达式可以左乘$B^{-1}$
$$B^{-1}A\Phi=\Phi\Lambda \Rightarrow C\Phi = \Phi\Lambda\tag{34}$$
其中$C = B^{-1}A$，这是一个关于$C$的特征值问题，可以根据公式29解

如果$B$不可逆，可以采用数值上的一些取巧，对$B$的主对角元素进行稍微修改，令其满秩
$$(B + \epsilon I)^{-1}A\Phi=\Phi\Lambda \Rightarrow C\Phi=\Phi \Lambda\tag{35}$$
其中$\epsilon$是一个非常小的正数，例如$\epsilon  = 10^{-5}$，可以使得$B$满秩即可

## 7.2 The Rigorous Solution
# 8 Conclusion
