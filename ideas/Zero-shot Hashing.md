**2024.2.28**
很多方法都在考虑 [[paper-notes/ml/algorithm/zeroshot-hashing/Spectral Hashing-2008-NeruIPS]] 提出的问题的各种近似优化形式，但 SH 中的 $\symbfit 1^TB$ 的条件，本义在于最大化每个哈希比特的信息，但实际上是一种原问题的一个强化，强化不相似样本间的距离差距，因为原损失在 $s_{ij}\ge 0$ 的条件下实际上仅存在尽量缩小相似样本之间的距离差距的趋势
因此不强求 $s_{ij} \ge 0$，不是考虑归一化 $S$，而是考虑中心化 $S$，
令 $\symbfit 1^TS = \symbfit 0^T$，而由于 $S = S^T$，$(\symbfit 1^TS)^T = S^T\symbfit 1 =S\symbfit 1=(\symbfit 0^T)^T=\symbfit 0$，
因此我们在中心化 $S$ 的列的同时，也中心化了 $S$ 的行，
从 $S\symbfit 1 = \symbfit 0 = 0\times\symbfit 1$ 容易得到 $\symbfit 1$ 此时属于 $S$ 的零空间

考虑中心化 $S$ 的列，令 $\symbfit 1^TS = \symbfit 0^T$，丢弃 $\symbfit 1^TB = \symbfit 0^T$ 的条件
已经知道

$$\begin{align}
&\min_B \|BB^T-rS\|_F^2\\
=&\min_B tr(BB^TBB^T)-rtr(BB^TS^T)-rtr(ABB^T)+r^2tr(SS^T)\\
=&\min_B tr(BB^TBB^T)-2rtr(SBB^T)+r^2tr(SS^T)\\
=&\min_B tr(BB^TBB^T)-2rtr(B^TSB)\\
&s.t.\symbfit 1^TS = \symbfit 0^T,B^TB = rI,B\in\{\pm 1\}^{n\times r}
\end{align}\tag{1}$$

而 $tr(BB^TBB^T)$ 在 $B^TB=I$ 条件下为常数，因此，公式 1 等价于

$$\begin{align}
&\max_Btr(B^TSB)\\
&s.t.\symbfit 1^TS = \symbfit 0^T,B^TB = rI,B\in\{\pm 1\}^{n\times r}
\end{align}\tag{2}$$

同时由于

$$\begin{aligned}
&tr(B^TLB)\\
=&\frac 1 2\sum_{jk}^ns_{jk}\|\symbfit b_j-\symbfit b_k\|^2
\end{aligned}$$

因此考虑用相似度矩阵 $S$ 作为邻接矩阵时，有

$$\begin{aligned}
&\min_B \frac 1 2\sum_{jk}^ns_{jk}\|\symbfit b_j-\symbfit b_k\|^2\\
=&\min_B tr(B^TLB)\\
=&\min _B tr(B^T(D-S)B)\\
=&\min_B tr(B^TDB)-tr(B^TSB)\\
&s.t.\symbfit 1^TS = \symbfit 0^T,B^TB = rI,B\in\{\pm 1\}^{n\times r}
\end{aligned}\tag{3}$$

而由于我们在中心化 $S$ 的列的同时，也中心化了 $S$ 的行，故满足 $D = 0$，因此公式 2 等价于 (事实上只要 $D$ 与 $B$ 无关，该等价就成立)

$$\begin{align}
&\max_Btr(B^TSB)\\
&s.t.\symbfit 1^TS = \symbfit 0^T,B^TB = rI,B\in\{\pm 1\}^{n\times r}
\end{align}\tag{4}$$

显然公式 4 和公式 2 等价，即 $F$ 范数形式的目标函数和平均哈希距离形式的目标函数在 $B^TB = rI$ 条件下是完全等价的

如果在这里丢弃 $B$ 的离散约束，则得到一个显然的关于 $S$ 的特征值问题，可以简单解出，SH 中采用的就是这类方法，KSH 则以松弛解为初始值，在此基础上进一步优化
直接采用松弛解的效果是不够理想的

考虑优化公式 4
参考 [[paper-notes/ml/algorithm/zeroshot-hashing/Supervised Hashing with Kernels-2012-CVPR]]，依旧假设哈希函数的形式为 $H(X) = sgn(\bar K_XP)$，即做核变换后再线性映射，并且列中心化 $\bar K_X$
注意 KSH 中没有关注哈希比特之间的不相关性，即 $B^TB = rI$ 约束，但 KSH 通过列中心化 $\bar K_X$，保证了 $\symbfit 1^T(\bar K_XP) = 0$，一定程度上支持了 $\symbfit 1^T B$ 的约束，
这种做法相当于显式地规定了每一位哈希函数的偏置量

**形式 1**
考虑直接以哈希函数为目标，将其嵌入公式 4

$$\begin{align}
&\max_P tr(sgn(\bar K_XP)^TSsgn(\bar K_XP))\\
&s.t.\symbfit 1^TS=\symbfit 0^T,sgn(\bar K_XP)^Tsgn(\bar K_XP) = rI
\end{align}$$

去除 $sgn(\cdot)$ 函数，得到

$$\begin{align}
&\max_P tr((\bar K_XP)^TS(\bar K_XP))\\
&s.t.\symbfit 1^TS=\symbfit 0^T,(\bar K_XP)^T(\bar K_XP) = rI
\end{align}$$

松弛问题存在闭式解，解满足 $(K_X^TS \bar K_X )P = (\bar K_X^T \bar K_X )P\Lambda$，
即 $((\bar K_X^T\bar K_X)^{-1}(K_X^TSK_X^T)-\Lambda)P = 0$，解出零空间即得到 $P$，
$P$ 中的列向量按特征值从大到小排列
如果 $\bar K_X^T\bar K_X$ 不可逆，则对其运用数值上的技巧，将其修改为 $(\bar K_X^T\bar K_X + \epsilon I)$，其中 $\epsilon$ 为极小的正数，例如 $\epsilon = 10^{-5}$

如此得到了松弛解

**形式 2**
借鉴原型的思想，同时希望引入一定的原特征空间信息，因此考虑用类中心初始化 $B$
考虑常用的哈希函数优化形式

$$\begin{align}
&\min_P\|sgn(\bar K_X P) -B\|_F^2+\alpha\|P\|_F^2\\
\end{align}$$

大多数工作选择丢弃 $sgn(\cdot)$ 函数，得到

$$\begin{align}
&\min_P\|\bar K_XP-B\|_F^2 + \alpha \|P\|_F^2\\
=&\min_Ptr((\bar K_XP-B)(P^T\bar K_X^T-B^T))+\alpha tr(PP^T)\\
=&\min_Ptr(\bar K_XPP^T\bar K_X^T-\bar K_XPB^T-BP^T\bar K_X^T + BB^T)+\alpha tr(PP^T)\\
=&\min_P tr(\bar K_XPP^T\bar K_X^T)-2tr(\bar K_XPB^T)+tr(BB^T)+\alpha tr(PP^T)\\
=&\min_P tr(\bar K_XPP^T\bar K_X^T)-2tr(BP^T\bar K_X^T)+\alpha tr(PP^T)\\
=&\min_P tr(P^T\bar K_X^T\bar K_XP)-2tr(BP^T\bar K_X^T)+\alpha tr(PP^T)
\end{align}\tag{10}$$

令公式 10 相对于 $P$ 的导数等于 0，得到

$$\begin{align}
2\bar K_X^T\bar K_XP-2\bar K_X^TB+2\alpha P&=0\\
\bar K_X^T\bar K_XP+\alpha P &=\bar K_X^TB\\
(\bar K_X^T\bar K_X+\alpha I)P &=\bar K_X^TB\\
P&=(\bar K_X^T\bar K_X^T+\alpha I)^{-1}\bar K_X^TB
\end{align}$$

**2024.3.1**
考虑一下 [[Strongly Constrained Discrete Hashing-2020-TIP|Strongly Constrained Discrete Hashing-2020-TIP
]]
$$\begin{align}
&\min_{B,Z}\mathcal\|BZ^T-rS\|_F^2+\alpha\|B-Z\|_F^2\\
&s.t.\begin{cases}B\in\{-1,+1\}^{n\times r}\\
Z \in \mathbb R^{n\times r},\mathbf1^TZ = \mathbf 0^T,Z^TZ = r\cdot I \end{cases}
\end{align}$$

目标函数可以写为

$$\begin{align}
&\min_{B,Z} tr((BZ^T-rS)(BZ^T-rS)^T)+\alpha tr((B-Z)(B-Z)^T)\\
=&\min_{B,Z} tr((BZ^T-rS)(ZB^T-rS))+\alpha tr((B-Z)(B^T-Z^T))\\
=&\min_{B,Z}tr(BZ^TZB^T)-2rtr(BZ^TS)+r^2tr(SS)\\
&+\alpha tr(BB^T)-2\alpha tr(BZ^T)+\alpha tr(ZZ^T)\\
=&\min_{B,Z} const-2rtr(Z^TSB)-2\alpha tr(Z^TB)\\
=&\max_{B,Z} rtr(B^TSZ)+\alpha tr(B^TZ)\\
=&\max_{B,Z} tr(B^T(rSZ + \alpha Z))\\
=&\max_{B,Z} tr(Z^T(rSB+\alpha B))
\end{align}$$

而 [[paper-notes/ml/algorithm/zeroshot-hashing/Discrete Graph Hashing-2014-NeurIPS]] 的目标函数则为

$$\begin{align}
&\max_{B,Z} tr(B^TSB)+\alpha tr(B^TZ)\\
=&\max_{B,Z} tr(B^T(SB+\alpha Z))\\
\end{align}$$

采用迭代优化
固定 $Z$，SCDH 关于 $B$ 的优化问题写为

$$\max_{B} tr(B^T(rSZ+\alpha Z))$$

最优解即 $B = sgn(rSZ + \alpha Z) = sgn((rS+\alpha I)Z)$
考虑 $Z^T(rS + \alpha I)(rS + \alpha I)Z$
我们希望 $(rS + \alpha I)(rS + \alpha I) =(rS+\alpha I)^2= I$，则一个解就是 $rS + \alpha I = I$，
那么 $rS = (1-\alpha) I$，即 $S = \frac {(1-\alpha)}{r} I$，显然不太现实
那么另一个考虑就是 $rS + \alpha I$ 是一个对称的正交矩阵，显然对称性是满足的
数值上看，将其展开 $(rS+\alpha I)(rS + \alpha I) = (r^2S^2 + 2r\alpha S + \alpha^2 I ) = I$，
则 $r^2 S^2 + 2r\alpha S = (1-\alpha^2)I$，不太好解

$$\begin{align}
&\min_{\alpha} \|(rS + \alpha I)^2-I\|_F^2\\
=&\min_{\alpha} \|r^2S^2+2r\alpha S+(\alpha^2-1) I\|_F^2\\
=&\min_{\alpha} tr((r^2S^2+2r\alpha S+(\alpha^2-1)I)^2)\\
=&\min_{\alpha} r^4tr(S^4)+2r^3\alpha tr(S^3)+r^2(\alpha^2-1) tr(S^2)\\
&+2r^3\alpha tr(S^3)+4r^2\alpha^2 tr(S^2)+2r\alpha(\alpha^2-1)tr(S)\\
&+r^2(\alpha^2-1)tr(S^2)+2r\alpha(\alpha^2-1)tr(S)+(\alpha^2-1)^2tr(I)\\
=&\min_{\alpha} \alpha(4r^3tr(S^3)) + (\alpha^3-\alpha)(4rtr(S))\\
&+(\alpha^2-1)(r^2tr(S^2)) + (\alpha^2-1)^2tr(I)\\
=&\min_{\alpha} A\alpha + B\alpha^2 + C\alpha^3 + D\alpha^4 + E
\end{align}$$

一个关于常数 $\alpha$ 的优化问题

考虑 $\symbfit 1^T(rSZ + \alpha Z)$
我们希望 $(r\symbfit 1^T SZ + \alpha\symbfit 1^T  Z) = \symbfit 0^T$，那么 $\symbfit 1^T SZ = \symbfit 0^T$，即 $\symbfit 1^TS = 0$ 即可

固定 $Z$，DGH 关于 $B$ 的优化问题写为

$$\max_{B} tr(B^T(SB+\alpha Z))$$

最优解即 $B = sgn(SB + \alpha Z)$
考虑 $(SB+\alpha Z)^T(SB + \alpha Z)$
我们希望 $(B^TS + \alpha Z^T)(SB + \alpha Z) = B^TSSB + \alpha^2 Z^TZ + \alpha B^TSZ + \alpha Z^TSB = rI$
即 $B^TS^2B + \alpha B^TSZ + \alpha Z^TSB + \alpha^2 r I = rI$，那么
$B^TS^2B + \alpha B^TSZ + \alpha Z^TSB = (r-r\alpha^2)I$，不太好解

考虑 $\symbfit 1^T(SB + \alpha Z)$
我们希望 $(r\symbfit 1^T SB + \alpha\symbfit 1^T  Z) = \symbfit 0^T$，那么 $\symbfit 1^T SB = \symbfit 0^T$，即 $\symbfit 1^TS = 0$ 即可

固定 $B$，SCDH 关于 $Z$ 的优化问题写为

$$\max_{Z}tr(Z^T(rSB+\alpha B))$$

固定 $B$，DGH 关于 $Z$ 的优化问题写为

$$\max_{Z}tr(Z^TB)$$

**2024.3.4**
激活函数平滑化
$f(x) = x$
$f(x) = sigmoid(x) = \frac {2}{e^{-x}+1} -1$
$f(x) = tanh(x)$
哈希函数
$H(X) = F(\bar K_X P)$

令 $\bar K_XP = Y$

$$\begin{align}
&tr(sgn(Y)^TStanh(Y)) + \alpha tr(sgn(Y)^Ttanh(Y))\\
&tr(sigmoid(Y)^TStanh(Y)) + \beta tr(tanh(Y)^Tsigmoid(Y))\\
&tr(Y^TSsigmoid(Y))+\gamma tr((sigmoid(Y)^TY)
\end{align}$$

**2024.3.5**
SCDH 在优化过程中引入了原特征空间信息 $X$，相似度信息 $S$，约束信息 $Z$

应不应该引入原特征空间信息？相似度信息是绝对正确的

$S_{tr} \rightarrow B_{tr}$
$B_{tr}\rightarrow X_{tr}$
$S_{tr}\rightarrow B_{tr} \rightarrow X_{tr}$

$X_{te} \rightarrow B_{te}$
$B_{te}\rightarrow S_{te}$
$X_{te}\rightarrow B_{te}\rightarrow S_{te}$

零样本学习中，重要的还是哈希函数，确切地说还是映射矩阵 $P$
$P = [\symbfit p_1,\dots,\symbfit p_r]$

两个约束都是对哈希码的有效性的约束，
$\bar K_X$ 不可控制，因此应该通过哈希函数尽量满足

$sgn(\bar K_X P) = B$
$sgn换成sigmoid$
$sigmoid(\bar K_X P) = B$ ? 增加了复杂性

要不考虑限制一下 $\bar K_X P$ 的范围，限制范围后感觉约束的迁移会更有效
干脆还是对 $\bar K_X$ 做一些正则，列中心化很合适?
或者特征空间/RKHS 每一个维度都除去均值？总之把数量级降下来，降到 $[-1,1]$


特征空间的数值是什么情况，有负值吗？ResNet-101 的输出没有负值

**2024.3.6**
考虑两种情况
1. 两个样本 $x_i,x_j$ 在特征空间相似，但是标签不同，即 $s_{ij} = -1$
2. 两个样本 $x_i,x_j$ 在特征空间不相似，但是标签相同，即 $s_{ij} = 1$
不妨考虑对满足这两种情况其一的 $s_{ij}$ 进行放缩

考虑一下原型的思想，为每一个类计算一个特征空间的类原型，简单点取就直接计算该类所有向量的均值
那么得到 $c$ 个类原型
计算类原型的时候可以计算一些统计量，比如标准差 $\sigma$

考虑概率与统计中的 $3\sigma$ 原则，如果特征向量落在了某个类原型的 $[-3\sigma,+3\sigma]$ 范围内，就认为特征空间上它与该类的其他样本是类似的，否则是不类似的
或者考虑统计该类所有向量与类原型的夹角余弦值，最后取中位数作为阈值

相似的程度也根据该样本的特征向量与类原型向量的夹角余弦值 $cos(\theta)$ 决定
$cos(\theta)$ 的取值范围是 $[-1,1]$，
越相似，越接近 1，如果标签不同，$s_{ij}$ 就越需要放缩，因此放缩倍率不妨规定为 $\frac {\alpha} {1-cos(\theta)}$，即 $s_{ij}$ 从 $-1$ 放缩到 $-\frac {\alpha} {1-cos(\theta)}$，$\alpha$ 是可调节参数
越不相似，越接近-1，如果标签相同，$s_{ij}$ 就越需要放缩，因此放缩倍率不妨规定为 $\frac {\alpha} {cos(\theta)+1}$，即 $s_{ij}$ 从 $1$ 放缩到 $\frac {\alpha} {cos(\theta)+1}$，$\alpha$ 是可调节参数

一个项 $s_{ij}$ 可能面临多个放缩倍率 (最多面临 $c$ 个)，选最大的那一个

感觉思路有点像 Hard sample 强化

这可以在特征空间做，也可以在执行了核函数之后的特征空间做

因为零样本学习是从特征空间出发到哈希空间的，而该方法可以学习能更好把握特征空间和标签空间的联系的哈希函数，因此对零样本学习方法有帮助

**2024.3.9**

$$\begin{align}
&\min_P\|\phi(\Phi(X)P) - B\|_F^2 + \|P\|_F^2\\
=&\min_P\|\phi(Z)-B\|_F^2 + tr(PP^T)\\
=&\min_Ptr((\phi(Z)-B)(\phi(Z)-B)^T)+tr(PP^T)\\
=&\min_Ptr(\phi(Z)\phi(Z)^T)-2tr(\phi(Z)B^T)+tr(BB^T)+tr(PP^T)\\
=&\min_Ptr(\phi(\Phi P)\phi(\Phi P)^T)-2tr(\phi(\Phi P)B^T)+tr(PP^T)\\
\end{align}$$

$f(x) = sigmoid(x)$
$f'(x) = f(x)(1-f(x))$

**2024.3.10**

$$
\begin{align}
f(P) = \max_{P}\sum_{ij}(\Phi P\circ B) - \lambda \|P\|_F^2\\
\end{align}
$$

$$\nabla_Pf(P) = \Phi^TB + 2\lambda P$$

$$g(x) = \frac {1}{1+e^{-x}}$$

$$\begin{align}
&\frac {\partial g(x)}{\partial x}\\
=&\frac {\partial \frac 1 y}{\partial y}\frac {\partial1+e^{-x}}{\partial x}\\
=&\frac {-1}{y^2}\times-e^{-x}\\
=&\frac {e^{-x}}{(1+e^{-x})^2}\\
=&(\frac {1}{1+e^{-x}})(1-\frac {1}{1+e^{-x}})\\
=&g(x)(1-g(x))
\end{align}$$

$$\begin{align}
f(P) = \max_{P}\sum_{ij}(2 * sigmoid(\Phi P) - 1)\circ B - \lambda \|P\|_1\\
\end{align}$$

考虑

$$\begin{align}
&\frac {\partial f(P)}{\partial P_{ij}}\\
=&\frac {\partial\sum_{ij}(2 * sigmoid(\Phi P)-1)\circ B}{\partial P_{ij} } - \frac {\partial \lambda\|P\|_1}{\partial P_{ij}}\\
\end{align}$$

考虑

$$\begin{align}
&\frac {\partial f(P)}{\partial P_{ij}}\\
=&\frac {\partial\sum_{ij}(2 * sigmoid(\Phi P)-1)\circ B}{\partial P_{ij} }\\
=&Tr\left[\left(\frac {\partial\sum_{ij}(2 * sigmoid(\Phi P)-1)\circ B}{\partial (2 * sigmoid(\Phi P)-1)}\right)^T\frac{{\partial(2 * sigmoid(\Phi P)-1)}}{\partial P_{ij}}\right]\\
=&Tr\left[\left(\frac {\partial \sum_{ij} U\circ B}{\partial U}\right)^T\frac{\partial U}{\partial P_{ij}}\right]\\
\end{align}$$

其中

$$\frac {\partial \sum_{ij} U\circ B}{\partial U}=B$$

$$\begin{align}
&\frac {\partial U}{\partial P_{ij}}\\
=& \frac {\partial (2*sigmoid(\Phi P)-1)}{\partial P_{ij}}\\
=& \frac {\partial (2*sigmoid(\Phi P))}{\partial P_{ij}}\\
=& 2\frac {\partial sigmoid(\Phi P)}{\partial P_{ij}}\\
=& 2\partial\begin{bmatrix}
0&\cdots &sigmoid(\Phi_{1,:}P_{:,j})&\cdots&0\\
0&\cdots &sigmoid(\Phi_{2,:}P_{:,j})&\cdots&0\\
\vdots&\vdots&\vdots&\vdots&\vdots\\
0&\cdots &sigmoid(\Phi_{n,:}P_{:,j})&\cdots&0\\
\end{bmatrix}/\partial P_{ij}\\
=& 2\begin{bmatrix}
0&\cdots &(sigmoid(\Phi_{1,:}P_{:,j}))(1-sigmoid(\Phi_{1,:}P_{:,j}))\Phi_{1,i}&\cdots&0\\
0&\cdots &(sigmoid(\Phi_{2,:}P_{:,j}))(1-sigmoid(\Phi_{2,:}P_{:,j}))\Phi_{2,i}&\cdots&0\\
\vdots&\vdots&\vdots&\vdots&\vdots\\
0&\cdots &(sigmoid(\Phi_{n,:}P_{:,j}))(1-sigmoid(\Phi_{n,:}P_{:,j}))\Phi_{n,i}&\cdots&0\\
\end{bmatrix}\\
\end{align}$$

故

$$\begin{align}
&\frac {\partial f(P)}{\partial P_{ij}}\\
=&Tr\left[\left(\frac {\partial \sum_{ij} U\circ B}{\partial U}\right)^T\frac{\partial U}{\partial P_{ij}}\right]\\
=&2Tr\left[B^T\begin{bmatrix}
0&\cdots &(sigmoid(\Phi_{1,:}P_{:,j}))(1-sigmoid(\Phi_{1,:}P_{:,j}))\Phi_{1,i}&\cdots&0\\
0&\cdots &(sigmoid(\Phi_{2,:}P_{:,j}))(1-sigmoid(\Phi_{2,:}P_{:,j}))\Phi_{2,i}&\cdots&0\\
\vdots&\vdots&\vdots&\vdots&\vdots\\
0&\cdots &(sigmoid(\Phi_{n,:}P_{:,j}))(1-sigmoid(\Phi_{n,:}P_{:,j}))\Phi_{n,i}&\cdots&0\\
\end{bmatrix}\\\right]\\
=&2 \langle B^T_{j,:},\begin{bmatrix}
(sigmoid(\Phi_{1,:}P_{:,j}))(1-sigmoid(\Phi_{1,:}P_{:,j}))\Phi_{1,i}\\
(sigmoid(\Phi_{2,:}P_{:,j}))(1-sigmoid(\Phi_{2,:}P_{:,j}))\Phi_{2,i}\\
\vdots\\
(sigmoid(\Phi_{n,:}P_{:,j}))(1-sigmoid(\Phi_{n,:}P_{:,j}))\Phi_{n,i}\\
\end{bmatrix}\rangle\\
=&2 \langle B_{:,j},\begin{bmatrix}
(sigmoid(\Phi_{1,:}P_{:,j}))(1-sigmoid(\Phi_{1,:}P_{:,j}))\Phi_{1,i}\\
(sigmoid(\Phi_{2,:}P_{:,j}))(1-sigmoid(\Phi_{2,:}P_{:,j}))\Phi_{2,i}\\
\vdots\\
(sigmoid(\Phi_{n,:}P_{:,j}))(1-sigmoid(\Phi_{n,:}P_{:,j}))\Phi_{n,i}\\
\end{bmatrix}\rangle\\
\end{align}$$

其中

$$\begin{align}
&\begin{bmatrix}
(sigmoid(\Phi_{1,:}P_{:,j}))(1-sigmoid(\Phi_{1,:}P_{:,j}))\Phi_{1,i}\\
(sigmoid(\Phi_{2,:}P_{:,j}))(1-sigmoid(\Phi_{2,:}P_{:,j}))\Phi_{2,i}\\
\vdots\\
(sigmoid(\Phi_{n,:}P_{:,j}))(1-sigmoid(\Phi_{n,:}P_{:,j}))\Phi_{n,i}\\
\end{bmatrix}\\
=&\begin{bmatrix}
(sigmoid(\Phi_{1,:}P_{:,j}))(1-sigmoid(\Phi_{1,:}P_{:,j}))\\
(sigmoid(\Phi_{2,:}P_{:,j}))(1-sigmoid(\Phi_{2,:}P_{:,j}))\\
\vdots\\
(sigmoid(\Phi_{n,:}P_{:,j}))(1-sigmoid(\Phi_{n,:}P_{:,j}))\\
\end{bmatrix}
\circ
\begin{bmatrix}
\Phi_{1,i}\\
\Phi_{2,i}\\
\vdots\\
\Phi_{n,i}
\end{bmatrix}\\
=&sigmoid(\Phi P_{:,j})\circ(1-sigmoid(\Phi P_{:,j}))\circ \Phi_{:,i}
\end{align}$$

显然雅可比矩阵

$$\begin{align}
&\frac {\partial f(P)}{\partial P}\\
=&\begin{bmatrix}
\frac {\partial f(P)}{\partial P_{11}}&\cdots&\frac {\partial f(P)}{\partial P_{1r}}\\
\vdots & \ddots & \vdots\\
\frac {\partial f(P)}{\partial P_{n1}}&\cdots&\frac {\partial f(P)}{\partial P_{nr}}\\
\end{bmatrix}
\end{align}$$

可以逐列计算

$$\begin{align}
&\begin{bmatrix}
\frac {\partial f(P)}{\partial P_{1j}}\\
\vdots \\
\frac {\partial f(P)}{\partial P_{nj}}\\
\end{bmatrix}\\
=&2\begin{bmatrix}
(sigmoid(\Phi P_{:,j})\circ(1-sigmoid(\Phi P_{:,j}))\circ \Phi_{:,1})^T\\
(sigmoid(\Phi P_{:,j})\circ(1-sigmoid(\Phi P_{:,j}))\circ \Phi_{:,2})^T\\
\vdots\\
(sigmoid(\Phi P_{:,j})\circ(1-sigmoid(\Phi P_{:,j}))\circ \Phi_{:,n})^T\\
\end{bmatrix}B_{:,j}\\
=&2\left(\begin{bmatrix}
(sigmoid(\Phi P_{:,j})\circ(1-sigmoid(\Phi P_{:,j}))^T\\
(sigmoid(\Phi P_{:,j})\circ(1-sigmoid(\Phi P_{:,j}))^T\\
\vdots\\
(sigmoid(\Phi P_{:,j})\circ(1-sigmoid(\Phi P_{:,j}))^T\\
\end{bmatrix}\circ\Phi^T\right)B_{:,j}
\end{align}$$

而
$$\frac {\partial \lambda\|P\|_1}{\partial P}=\lambda sgn(P) $$
**2024.3.15**
EigenDecomposition 精度不够，因为 $l$ 数量级太大
$l$的计算中，$SB$占了主导，而$S$中，$-1$项占了绝对主导

**2024.3.24**

$$\begin{align}
\min_{B,P,W,Z}\|K&\cdot S - BZ^T\|_F^2+\lambda\|\Phi(X)P-B\|_F^2+\alpha\|B-Z\|_F^2\\
&+\beta\|P\|_F^2+\gamma\|YW-B\|_F^2+\mu\|W\|_F^2\\
&+\eta\|(\Phi(X')P)B^T-K\cdot S'\|_F^2
\end{align}$$

Update P

$$\begin{align}
\min_{P}\lambda\|\Phi(X)P-B\|_F^2 +\eta \|(\Phi(X')P)B^T-K\cdot S' \|_F^2 + \beta\|P\|_F^2
\end{align}$$

$$\begin{align}
&\min_P\|(\Phi(X')PB^T -K \cdot S'\|_F^2\\
=&\min_PTr[(\Phi'PB^T-KS')(\Phi'PB^T-KS')^T]\\
=&\min_PTr[(\Phi'PB^T-rS')(BP^T\Phi'^T-rS'^T)]\\
=&\min_PTr[\Phi'PB^TBP^T\Phi'^T]-2rTr[\Phi'PB^TS'^T]\\
\end{align}$$

$$\begin{align}
&\min_P\lambda\|\Phi P-B\|_F^2 + \beta\|P\|_F^2\\
=&\min_P\lambda Tr[(\Phi P-B)(\Phi P-B)^T] + \beta Tr[PP^T]\\
=&\min_P\lambda Tr[(\Phi P-B)(P^T\Phi^T - B^T)]+\beta Tr[PP^T]\\
=&\min_P\lambda Tr[\Phi PP^T\Phi^T]-2\lambda Tr[\Phi PB^T] + \beta Tr[PP^T]\\
\end{align}$$

$$\begin{align}
&\min_P \lambda Tr[\Phi PP^T\Phi^T] - 2\lambda Tr[\Phi PB^T] \\
&+\eta Tr[\Phi' PB^TBP^T\Phi'^T] -2r\eta Tr[\Phi' P B^TS'^T]\\
&+\beta Tr[PP^T]\\
=&\min_P\lambda Tr[\Phi^T\Phi PP^T] - 2\lambda Tr[P^T\Phi^T B] + \beta Tr[PP^T]\\
&+\eta Tr[BP^T\Phi'^T\Phi'PB^T]-2r\eta Tr[P^T\Phi'^TS'B]
\end{align}$$

$$\nabla_P = 2\lambda\Phi^T\Phi P - 2\lambda \Phi^TB + 2\beta P + 2\eta\Phi'^T\Phi' PB^TB -2r\eta\Phi'^TS'B $$

如果把$B$换成$Z$，$Z$满足$Z^TZ = rI$
即

$$\begin{align}
\min_{P}\lambda\|\Phi(X)P-B\|_F^2 +\eta \|(\Phi(X')P)Z^T-K\cdot S' \|_F^2 + \beta\|P\|_F^2
\end{align}$$

$$\nabla_P = 2\lambda \Phi^T\Phi P - 2\lambda \Phi^TB + 2\beta P + 2r\eta \Phi'^T\Phi' P - 2r\eta \Phi'^TS'Z$$

令 $\nabla_P = 0$，得到

$$\begin{align}
&\lambda\Phi^T\Phi P + \beta P + r\eta \Phi'^T\Phi P= \lambda\Phi^TB + r\eta\Phi'^TS'Z\\
&(\lambda\Phi^T\Phi + \beta I + r\eta \Phi'^T\Phi')P=\lambda\Phi^TB + r\eta\Phi'^TS'Z\\
&P =(\frac {1}{r\eta}\Phi^T\Phi + \frac {1}{\lambda} \Phi'^T\Phi'+\frac {\beta}{r\lambda \eta} I )^{-1}(\frac {1}{r\eta}\Phi^TB+\frac 1 {\lambda}\Phi'^TS'Z)\\
&P=(\lambda\Phi^T\Phi+r\eta \Phi'^T\Phi' + \beta I)^{-1}(\lambda\Phi^TB + r\eta \Phi'^TS'Z)
\end{align}$$

Update W

Update B

Update Z

$$\begin{align}
&\min_Z\|rS - BZ^T\|_F^2 + \alpha \|B-Z\|_F^2 + \eta \|\Phi' PZ^T-rS'\|_F^2\\
=&\min_Z\|rS - BZ^T\|_F^2 + \alpha \|B-Z\|_F^2 + \eta \|rS'-\Phi' PZ^T\|_F^2\\
\end{align}$$

$$\begin{align}
&\min_Z\|rS - BZ^T\|_F^2\\
=&\min_ZTr[(rS-BZ^T)(rS-BZ^T)^T]\\
=&\min_ZTr[(rS-BZ^T)(rS-Z^TB)]\\
=&\min_Z-2rTr[BZ^TS^T] + Tr[BZ^TZ^TB]\\
=&\min_Z-2rTr[BZ^TS^T]\\
=&\min_Z-2rTr[Z^TS^TB]
\end{align}$$

$$\begin{align}
&\min_Z\eta \|rS-\Phi' PZ^T\|_F^2\\
=&\min_Z-2r\eta Tr[Z^TS'^T\Phi'P]
\end{align}$$

$$\min_Z\|B-Z\|_F^2 = \min_Z-2\alpha Tr[Z^TB]$$

故

$$\max_ZTr[Z^T(rSB + r\eta S'\Phi' P+\alpha B)]$$
