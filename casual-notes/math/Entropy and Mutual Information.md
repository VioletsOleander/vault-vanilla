# 1 Introduction
想象一下，Alice 和 Bob 分别住在多伦多和波士顿。Alice (多伦多) 只要不下大雪就去慢跑，Bob (波士顿) 从不去慢跑
注意到，Alice 的行为提供了多伦多天气的信息，Bob 的行为则没有提供信息，这是因为 Alice 的行为是随机的 (random)，且与多伦多的天气相关 (correlated with)，而鲍勃的行为是决定性的 (deterministic)
我们如何量化信息的概念 (quantify the notion of information)？

# 2 Entropy
**Definition** 一个概率质量函数 (pmf) 为 $p_X(x)$ 的离散随机变量 $X$ 的熵 (entropy) 定义为：

$$H(X) = -\sum_{x}p(x)\log p(x)=-\mathbb E[\log p(x)]$$

熵衡量了 $X$ 中期望的不确定性 (expected uncertainty)
(我们认为概率 $p(x)$ 越小，不确定性越大，而 $-\log p(x)$ 随着 $p(x)$ 减小而增大，因此用于衡量不确定性)
我们也说 $H(X)$ 近似等于我们平均可以从随机变量 $X$ 的一个实例 (instance) 中学习到的信息量
(从每个示例中得到的信息量就是 $-\log p(x)$，$H(X)$ 是它们的加权平均，我们认为概率越小的事情信息量越大)

注意 $\log$ 函数以什么为底数并不重要，改变底数也仅仅是对熵乘上了一个常数，我们通常以2为底数

## 2.1 Example
假设有随机变量：

$$X = \begin{cases}0,\quad\text{with prob }p\\1,\quad\text{with prob } 1-p\end{cases}$$

则 $X$ 的熵记为：

$$H(X) = -p\log p -(1-p)\log (1-p) = H(p)$$

注意到熵和随机变量取的值无关 (例如本例中的0, 1)，仅由概率分布 $p(x)$ 决定

## 2.2 Two variables
考虑两个随机变量 $X,Y$，服从联合分布，概率质量函数为 $p(x,y)$
**Definition** 随机变量 $X,Y$ 的联合熵 (joint entropy) 定义为：

$$H(X,Y) = -\sum_{x,y}p(x,y)\log p(x,y)$$

联合熵衡量了两个随机变量 $X,Y$ 放在一起时 (taken together) 的不确定程度

**Definition** 给定 $Y$，$X$ 的条件熵 (conditional entropy) 定义为：

$$H(X|Y) = -\sum_{x,y}p(x,y)\log p(x|y) = -\mathbb E[\log p(x|y)]$$

条件熵衡量了在我们知道了 $Y$ 的值以后，随机变量 $X$ 中还留有多少不确定性

Remark：可以把熵视作是针对概率分布的衡量统计量，概率分布越具有不确定性，依照该分布计算出的熵就越大
Remark：

$$\begin{align}
H(X|Y) &= -\mathbb E[\log p(x|y)]\\
&=-\sum_yp(y)\left(\sum_{x}p(x|y)\log p(x|y)\right)\\
&=-\sum_{x,y} p(x,y)\log p(x|y)
\end{align}$$

## 2.3 Properties
之前定义的各种熵有以下性质：
- Non negativity 非负性
	$H(X)\ge 0$，熵总是非负的，$H(X) = 0$ 当且仅当随机变量 $X$ 是确定的
	(证明：
	1 因为 $0\le p(x) \le 1$，故 $-\log p(x) \ge 0$，故 $H(X) = \mathbb E[-\log p(x)]\ge 0$
	2 当 $H(X) = 0$，可以知道一定存在 $p(x) = 1$，此时 $X$ 只能取 $x$ 一个值
	)
- Chain rule 链式法则
	可以按照如下方式分解联合熵：
    
$$H(X_1,X_2,\dots,X_n) = \sum_{i=1}^nH(X_i|X^{i-1})$$
    
	其中 $X^{i-1} = \{X_1,X_2,\dots,X_{i-1}\}$
	对于两个变量，链式法则写为：
	

$$\begin{align} H(X,Y) &= H(X|Y) + H(Y)\\ &=H(Y|X) + H(X) \end{align}$$
    
	注意通常 $H(X|Y) \ne H(Y|X)$
	(证明：仅证明两个变量的情况

$$\begin{align}
H(X,Y) &= -\sum_{x,y}p(x,y)\log p(x,y)\\
&=-\sum_{x,y}p(x,y)(\log p(x|y) + \log p(y))\\
&=-\sum_{x,y}p(x,y)\log p(x|y) - \sum_{x,y}p(x,y) \log p(y)\\
&=-\sum_{x,y}p(x,y)\log p(x|y) - \sum_{y}p(y) \log p(y)\\
&=H(X|Y)  + H(Y)
\end{align}$$
)
- Monotonicity 单调性
	条件于某个随机变量总是会让熵减少：
    
$$H(X|Y)\le H(X)$$

	也就是说有多的信息永远比没有好 (information never hurts)
	(证明：

$$\begin{align}
H(X)-H(X|Y) &=-\sum_{x}p(x)\log p(x) + \sum_{x,y}p(x,y)\log p(x|y)\\
&=-\sum_{x,y}p(x,y)\log p(x) + \sum_{x,y}p(x,y)\log p(x|y)\\
&=-\sum_{x,y}p(x,y) \left(\log p(x) - \log p(x|y)\right)\\
&=-\sum_{x,y}p(x,y) \left(\log \frac {p(x)}{p(x|y)}\right)\\
&=-\sum_{x,y}p(x,y) \left(\log \frac {p(x)p(y)}{p(x,y)}\right)\\
\end{align}$$

    定义随机变量 $Z$：$p\left(Z = \frac {p(x)p(y)}{p(x,y)}\right) = p(x,y)$
	则原式可以写为：

$$\begin{align}
H(X) - H(X|Y) = -\mathbb E[\log Z]
\end{align}$$

	根据 Jensen 不等式，有：

$$\begin{align}
-\mathbb E[\log Z]\ge -\log \mathbb E[Z] &= -\log \sum_{x,y}p(x,y) \frac {p(x)p(y)}{p(x,y)}\\
&=-\log\sum_{x,y} p(x)p(y)\\
&=-\log\sum_x\sum_yp(x)p(y)\\
&=-\log\sum_xp(x)\sum_yp(y)\\
&=-\log 1\\
&=0
\end{align}$$

	因此有

$$H(X)-H(X|Y) = -\mathbb E[\log Z] \ge 0$$

	证毕
	)
- Maximum entropy 最大熵
	令 $\mathcal X$ 为随机变量 $X$ 所有可能取值构成的集合，则

$$H(X) \le \log|\mathcal X|$$

	当 $X$ 服从均匀分布时，$H(X)$ 达到上界 $\log |\mathcal X|$
	(证明：

$$H(X) = -\sum_x p(x)\log p(x) = -\mathbb E[\log {p(x)}] = \mathbb E[\log \frac 1 {p(x)}]$$

	根据 Jensen 不等式，有：

$$\begin{align}
\mathbb E[\log \frac 1 {p(x)}]&\le \log \mathbb E[\frac 1 {p(x)}]\\
&=\log\sum_x p(x) \frac 1 {p(x)}\\
&=\log \sum_x 1\\
&=\log |\mathcal X|
\end{align}$$

	故 

$$H(X) = \mathbb E[\log \frac 1 {p(x)}]\le\log|\mathcal X|$$

	Jensen 不等式等号成立当且仅当 $\frac 1 {p(x)}$ 是常数，即 $X$ 服从均匀分布
	证毕
	)
- Non increasing under functions 在函数下不增加
	令 $X$ 是随机变量，$g(X)$ 是关于 $X$ 的确定的 (deterministic) 函数，有：

$$H(X)\ge H(g(X))$$

	等号成立当且仅当函数 $g$ 是可逆的 (invertible)
	(证明：
	根据链式法则，有：
$$H(X,g(X)) = H(X) + H(g(X)|X) = H(g(X)) + H(X|g(X))$$
	因为 $g$ 是关于 $X$ 的确定的函数，即分布 $p(g(x)|x)$ 不存在随机性，故 $H(g(X)|X) = 0$
	因此有：
$$H(X)-H(g(X) = H(X|g(X))\ge 0$$
	等号成立当且仅当 $H(X|g(X)) = 0$，即分布 $p(x|g(x))$ 不存在随机性，即给定 $g(X)$，我们可以确定地选择 $X$，也就是 $g$ 是可逆的
	证毕
	)
# 3 Continuous random variables
连续性随机变量各种熵的定义和离散型随机变量类似
**Definition** 有连续型随机变量 $X$，服从概率密度函数 $f(x)$，它的微分熵 (differential entropy) 定义为：
$$h(X) = -\int f(x)\log f(x)dx = -\mathbb E[\log f(x)]$$

**Definition** 考虑一对连续型随机变量 $X,Y$，服从联合概率密度函数 $f(x,y)$，则它们的联合熵定义为：
$$h(X,Y) = -\int\int f(x,y)\log f(x,y)dxdy$$
条件熵定义为：
$$h(X|Y) = -\int\int f(x,y)\log f(x|y)dxdy$$
## 3.1 Properties
一些在离散情况下成立的性质在连续情况下也成立，但也存在不成立的
- Non negativity doesn't hold 非负性不成立
	$h(X)$ 可以为负
	例如：考虑随机变量 $X$，在 $[a,b]$ 上均匀分布，则它的熵是：
$$h(X) = -\int_a^b\frac 1{b-a}\log \frac 1{b-a} dx = \log(b-a)$$
	若 $b-a<1$，则 $h(X)<0$
- Chain rule 链式法则成立
$$\begin{align}
h(X,Y) &= h(X) + h(Y|X)\\
&=h(Y) + h(X|Y)
\end{align}$$
- Monotonicity 单调性成立
$$h(X|Y)\le h(X)$$
- Maximum entropy
	对于一般的 (general) 概率密度函数 $f(x)$ 没有显式的计算熵的上界的公式，但对于幂有限 (power limited) 的函数则有
	考虑随机变量 $X\sim f(x)$，若有
$$\mathbb E[x^2] = \int x^2f(x)dx \le P$$
	则有
$$\max h(X) = \frac 1 2\log (2\pi eP)$$
	最大值在 $X\sim \mathcal N(0, P)$ 时达到
	(要证明，考虑使用标准拉格朗日乘子法求解问题 $\max h(f) = -\int f\log f dx\quad s.t.\ \mathbb E[x^2] = \int x^2 fdx \le P$)
- Non increasing under functions 在函数下不增加不一定成立
	因为我们不能保证 $h(X|g(X))\ge 0$

# 4 Mutual information
**Definition** 两个服从于联合概率质量函数 $p(x,y)$ 的离散型随机变量 $X,Y$ 之间的互信息 (mutual information) 定义为：

$$\begin{align}
I(X;Y) &= \sum_{x,y} p(x,y) \log \frac {p(x,y)}{p(x)p(y)}\\
&=H(X) - H(X|Y)\\
&=H(Y) - H(Y|X)\\
&=H(X) + H(Y) - H(X,Y)
\end{align}$$

证明：
$$\begin{align}
I(X;Y) &= \sum_{x,y} p(x,y) \log \frac {p(x,y)}{p(x)p(y)}\\
&=\sum_{x,y}p(x,y)\log \frac {p(x,y)}{p(y)} - \sum_{x,y}p(x,y) p(x)\\
&=\sum_{x,y}p(x,y)\log p(x|y)- \sum_{x,y}p(x,y) p(x)\\
&=- \sum_{x,y}p(x,y) p(x) +\sum_{x,y}p(x,y)\log p(x|y)\\
&=H(X)- H(X|Y)
\end{align}$$
由链式法则，我们知道：
$$H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$
因此有：
$$H(X)-H(X|Y) = H(Y) - H(Y|X)$$
同时有：
$$\begin{align}
2H(X,Y) &= H(X) + H(Y) + H(Y|X) + H(X|Y)\\
H(X) + H(Y) - H(X,Y) &=H(X,Y) - H(Y|X)- H(X|Y)\\
\end{align}$$
故：
$$\begin{align}
H(X) + H(Y) - H(X,Y) &=H(X)- H(X|Y)\\
H(X) + H(Y) - H(X,Y) &=H(Y)- H(Y|X)\\
\end{align}$$
证毕

**Definition** 两个服从于联合概率密度函数$f (x, y)$的连续型随机变量$X, Y$之间的互信息定义为：
$$I(X;Y) = \int\int f(x,y)\log\frac {f(x,y)}{f(x)f(y)}dxdy$$

可以认为互信息是$X$中包含的关于$Y$的信息，或是$Y$中包含的关于$X$的信息，或是$X$和$Y$共同具有的信息/不确定性 (uncertainty)
![[Entropy and mutual information-Fig1.png]]
## 4.1 Non-negativity of mutual information
对于连续性随机变量和离散型随机变量，都有 $I (X; Y)\ge 0$

**Jensen's inequality** 
一个函数在 $[a, b]$ 上是凸的，则对于 $\forall x_1, x_2\in [a, b]$，我们有：

$$f(\theta x_1 + (1-\theta)x_2)\le \theta f(x_1) + (1-\theta)f(x_2)$$

对于二次可微的函数，函数在 $[a, b]$ 上是凸的等价于 $f'' (x)\ge 0$ 对 $\forall x\in [a, b]$ 成立

**Lemma** Jensen 不等式说明了对于任意凸函数 $f (x)$，我们有

$$\mathbb E[f(x)]\ge f(\mathbb E[x])$$

**Relative entropy** 衡量两个概率分布之间的距离的一个自然的方式是使用相对熵 (relative entropy)，相对熵也称为 KL 散度

**Definition** 两个概率分布 $p (x)$ 和 $q (x)$ 之间的相对熵定义为：

$$D(p(x)||q(x)) = \sum_{x} p(x)\log\frac {p(x)}{q(x)}$$

Remark：

$$\begin{align}
D(p(x)||q(x))&= \sum_x p(x)\log \frac {p(x)}{q(x)}\\
&=-\sum_x p(x)\log q(x)+\sum_xp(x)\log p(x)\\
&=\underbrace{-\sum_x p(x)\log q(x)}_{\text{ cross entropy:}\  H(p, q)} - \underbrace{H(p)}_{\text{entropy} }
\end{align}$$

相对熵和互信息的关联是：

$$I(X;Y) = D(p(x,y)||p(x)p(y))$$

(
$$\begin{align}
I(X;Y) &= H(X) - H(X|Y)\\
&=-\sum_x p(x)\log p(x) + \sum_{x,y} p(x,y)\log p(x|y)\\
&=\sum_{x,y}p(x,y) \log \frac {p(x|y)}{p(x)}\\
&=\sum_{x,y} p(x,y)\log \frac {p(x,y)}{p(x)p(y)}\\
&=D(p(x,y) || p(x)p(y))
\end{align}$$
)

如果可以证明相对熵总是非负的，就可以说明互信息总是非负的

证明：相对熵的非负性
令$p (x)$和$q (x)$是两个任意的概率分布，考虑它们的相对熵：
$$\begin{align}
D(p(x)||q(x)) &= \sum_x p(x)\log \frac {p(x)}{q(x)}\\
&=-\sum_xp(x)\log \frac {q(x)}{p(x)}\\
&=-\mathbb E_{x\sim p(x)}\left[\log \frac {q(x)}{p(x)}\right]\\
&\ge-\log\left(\mathbb E_{x\sim p(x)}\left[\frac {q(x)}{p(x)}\right]\right)\\
&=-\log\left(\sum_x p(x)\frac {q(x)}{p(x)}\right)\\
&=-\log\left(\sum_x q(x)\right)\\
&=0
\end{align}$$
## 4.2 Conditional mutual information
**Definition** 令$X, Y, Z$服从概率质量函数为$p (x, y, z)$的联合分布，则给定$Z$，$X, Y$之间的条件互信息 (conditional mutual information) 定义为：
$$\begin{align}
I(X;Y|Z) &=-\sum_{x,y,z}p(x,y,z)\log\frac {p(x,y|z)}{p(x|z)p(y|z)}\\
&=H(X|Z)-H(X|YZ)\\
&=H(XZ)+H(YZ)-H(XYZ)-H(Z)
\end{align}$$
(证明：
$$\begin{align}
I(X;Y|Z) &=-\sum_{x,y,z}p(x,y,z)\log\frac {p(x,y|z)}{p(x|z)p(y|z)}\\
&=-\sum_{x,y,z}p(x,y,z) \log \frac {p(x,y|z)}{p(y|z)} -\sum_{x,y,z}p(x,y,z)\log \frac {1}{p(x|z)}\\
&=-\sum_{x,y,z}p(x,y,z)\log \frac {1}{p(x|z)}-\sum_{x,y,z}p(x,y,z) \log {p(x|y,z)} \\
&=-\sum_{x,z}p(x,z)\log \frac {1}{p(x|z)}-\sum_{x,y,z}p(x,y,z) \log {p(x|y,z)} \\
&=H(X|Z) - H(X|YZ)\\
&=H(X,Z) - H(Z)-(H(X,Y,Z)-H(Y,Z))\\
&=H(X,Z) + H(Y,Z)-H(X,Y,Z)-H(Z)
\end{align}$$
同理可以得到：
$$I(X;Y|Z) = H(Y|Z) - H(Y|XZ)$$
)
条件互信息衡量了在已知$Z$的情况下$X, Y$共同具有的信息/不确定性，或者说由$X, Y$共同具有，且$Z$不具有的信息/不确定性
## 4.3 Properties
- Chain rule 链式法则
	存在以下链式法则：
$$I(X;Y_1Y_2\dots Y_n)=\sum_{i=1}^nI(X;Y_i|Y^{i-1})$$
	其中$Y^{i-1} = \{Y_1, Y_2,\dots, Y_{i-1}\}$
	根据链式法则，我们可以推导出：
$$I(X;YZ) = I(X;Y) + I(X;Z|Y) = I(X,Z) + I(X;Y|Z)\tag{4.3.1}$$
- No monotonicity 非递增性
	条件可以减少两个变量间的互信息，也可以增大两个变量间的互信息，
	因此$I (X; Y|Z)\ge I (X; Y)$和$I (X; Y|Z)\le I (X; Y)$都有可能
	Increasing example 增大的例子：
	假设我们有$X, Y, Z$，使得$I (X; Z) = 0$(这意味着$X$和$Z$相互独立)，则公式 (4.3.1) 就写为：
$$I(X;Y) + I(X;Z|Y) = I(X;Y|Z)$$
	因此有：
$$I(X;Y|Z)-I(X;Y) =I(X;Z|Y)\ge 0$$
	即：
$$I(X;Y|Z)\ge I(X;Y)$$
	Decreasing example 减小的例子:
	假设我们有$X, Y, Z$，使得$I (X; Z|Y) = 0$，则公式 (4.3.1) 就写为：
$$I(X;Y) = I(X;Z)+I(X;Y|Z)$$
	因此有：
$$I(X;Y)-I(X;Y|Z)=I(X;Z)\ge 0$$
	即：
$$I(X;Y)\ge I(X;Y|Z)$$
# 5 Data processing inequality
考虑三个随机变量$X, Y, Z$，它们之间的关系形成了一个马尔可夫链：$X\rightarrow Y \rightarrow Z$，这个关系说明了$p (x, z|y) = p (x|y) p (z|y)$(也就是以$Y$为条件时，$X$和$Z$相互独立)，进一步说明了$I (X; Z|Y) = 0$，就像上面的减小的例子，因此此时有：
$$I(X;Y) =I(X;Z)+I(X;Y|Z)$$
即：
$$I(X;Y)\ge I(X;Z)$$
称该式为数据处理不等式 (data processing inequality)
即数据处理不能增加信号中含有的信息

