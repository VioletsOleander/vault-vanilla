> ryan

# Lecture 2
## 2.1 Convex Optimization Problem
A convex optimization problem is of the form:

$$
\operatorname*{min}_{x\in D}f(x)
$$

subject to

$$
\begin{array}{l}{{g_{i}(x)\leq0,\ i=1,...,m}}\\ {{h_{j}(x)=0,\ j=1,...,r}}\end{array}
$$

where $f$ and $g_{i}$ are all convex , and $h_{j}$ are affine. 
Any local minimizer of a convex optimization problem is a global minimizer.
> 凸优化问题定义为约束函数都为凸函数的最小化问题
> 凸优化问题中，局部最小点就是全局最小点
## 2.2 Convex Sets
### 2.2.1 Definitions
Definition 2.1 Convex set: 
a set $C\subseteq\mathbb{R}^{n}$ is a convex set if for any $x,y\in C$ , we have

$$
\begin{array}{r}{t x+(1-t)y\in C,\ f o r\ a l l\ 0\leq t\leq1}\end{array}
$$

> 凸集：包含任意两个点的凸组合的集合

Definition 2.2 Convex combination of $x_{1},...,x_{k}\in\mathbb{R}^{n}$ : 
any linear combination

$$
\theta_{1}x_{1}+...+\theta_{k}x_{k},{\mathrm{~}}w i t h{\mathrm{~}}\theta_{i}\geq0,{\mathrm{~}}a n d{\mathrm{~}}\sum_{i=1}^{k}\theta_{i}=1
$$

> 凸组合：任意满足系数和为 1，且各个系数非负的线性组合

Definition 2.3 Convex hull of set $C$ : 
all convex combinations of elements in $C$ . The convex hull is always convex.
> 凸包：集合 $C$ 的所有元素的凸组合构成的集合
> 显然凸包一定是凸集

Definition 2.4 Cone: 
a set $C\subseteq\mathbb{R}^{n}$ is a cone if for any $x\in C$ , we have $t x\in C$ for all $t\geq0$

> 锥：包含任意元素的非负放缩的集合 ($t\ge 0$)


Definition 2.5 Convex cone: 
a cone that is also convex, i.e.,

$$
x_{1},x_{2}\in C\implies t_{1}x_{1}+t_{2}x_{2}\in C\;f o r\;a l l\;t_{1},t_{2}\geq0
$$

> 凸锥：凸的锥
> 即包含任意两个元素的凸组合的任意放缩的集合

Definition 2.6 Conic combination of $x_{1},...,x_{k}\in\mathbb{R}$ : 
any linear combination

$$
\theta_{1}x_{1}+...+\theta_{k}x_{k},\ w i t h\ \theta_{i}\geq0
$$

> 锥组合：任意满足系数非负的线性组合

Definition 2.7 Conic hull of set $C$ : 
all conic combinations of elements in $C$ .
> 锥包：集合 $C$ 中所有的元素的锥组合构成的集合
### 2.2.2 Examples of convex sets
-  Empty set, point, line.
-  Norm ball: $\{x:\|x\|\leq r\}$ , for given norm $\lVert\cdot\rVert$ , radius $r$ .
-  Hyperplane: $\{x:a^{T}x=b\}$ , for given $a,b$ .
-  Halfspace: $\{x:a^{T}x\leq b\}$ .
-  Affine space: $\{x:A x=b\}$ , for given $A,b$ .
-  Polyhedron: $\{x:A x\leq b\}$ , where $\leq$ is interpreted componentwise. The set $\{x:A x\leq b,C x=d\}$ is also a polyhedron.
-  Simplex: special case of polyhedra, given by $\mathrm{conv}\{x_{0},...,x_{k}\}$ , where these points are affinely independent. The canonical example is the probability simplex,

$$
\mathrm{conv}\{e_{1},...,e_{n}\}=\{w:w\geq0,1^{T}w=1\}
$$

> 常见凸集：
> 空集、点、线
> 范数球（球心是原点）
> 超平面
> 半空间
> 仿射空间（仿射空间可以看作是没有原点的向量空间）
> 多面体
> 单纯形：单纯形是多面体的特例，单纯形定义为一组仿射无关的点的凸包
### 2.2.3 Examples of convex cones
-  Norm cone: $\{(x,t):\|x\|\leq t\}$ , for given norm $\lVert\cdot\rVert$ . It is called second-order cone under the $l_{2}$ norm $\left\|\cdot\right\|_{2}$ .
-  Normal cone: given any set $C$ and point $x\in C$ , the normal cone is

$$
{\mathcal{N}}_{C}(x)=\{g:g^{T}x\geq g^{T}y,{\mathrm{~for~all~}}y\in C\}
$$
    
    This is always a convex cone, regardless of $C$ .
-  Positive semidefinite cone:

$$
\mathbb{S}_{+}^{n}=\{X\in\mathbb{S}^{n}:X\succeq0\}
$$

    where $X\succeq0$ means that $X$ is positive semidefinite ( $\mathbb{S}^{n}$ is the set of $n\times n$ symmetric matrices).

> 常见凸锥：
> 范数锥，当范数是二范数时，称为二阶锥（证明：三角不等式）
> 法锥：给定任意集合 $C$，和 $C$ 中的一个点 $x$（ 一般是边界点），法锥定义为所有在 $x$ 处指向凸集外侧的法向量的集合 ($g^T (y-x)\le 0\text{ for all }y\in C$)
> 半正定锥：包含了所有 $\mathbb S^n$ 中的半正定矩阵的集合
### 2.2.4 Key properties of convex sets
- Separating hyperplane theorem : two disjoint conv e a separating between hyperplane them. Formally, if $C,D$ are nonempty convex sets with  $C\cap D=\emptyset$ , then there exists $a,b$ such that

$$
C\subseteq\{x:a^{T}x\leq b\},\ D\subseteq\{x:a^{T}x\geq b\}
$$

- Supporting hyperplane theorem : a boundary point of a convex set has a supporting hyperplane passing through it. Formally, if C is a nonempty convex set, and $x_{0}\in\mathrm{bd}(C)$ , then there exists $a$ such that

$$
C\subseteq\{x:a^{T}x\leq a^{T}x_{0}\}
$$

> 分离超平面定理：两个不相交的凸集之间存在一个分离超平面 $a^T x = b$
> 支撑超平面定理：
> 一个凸集 $C$ 的一个边界点 $x_0$ 存在一个穿过它的支撑超平面 $a^Tx = b$
> 显然 $x_0$ 满足 $a^Tx_0 = b$，而 $C$ 中的其他点 $x \in C$ 则都落在该支撑超平面的一边，即满足 $a^Tx < b = a^Tx_0$，则 $C$ 是该支撑超平面划分出的半空间的子集，即
> $C\subseteq\{x:a^{T}x\leq a^{T}x_{0}\}$
### 2.2.5 Operations preserving convexity
#### 2.2.5.1 Operations
-  Affine ages and preimages: if $f(x)=A x+b$ and $C$ is convex, then $f(C)=\{f(x):x\in C\}$ is convex, and if D is convex, then $f^{-1}(D)=\{x:f(x)\in D\}$ is convex. Compared to scaling and translation, this operation also has rotation and dimension reduction.

-  Perspective images and preimages: the perspective function is $P:\mathbb{R}^{n}\times\mathbb{R}_{++}\rightarrow\mathbb{R}^{n}$ (where $\mathbb{R}_{++}$ denotes positive reals),

$$
P(x,z)=x/z
$$

    for $z>0$ . If $C\subseteq\operatorname{dom}(P)$ is convex then so is $P(C)$ , and if $D$ is convex then so is $P^{-1}(D)$ .

-  Linear-fractional images and preimages: the perspective map composed with an affine function,

$$
f(x)={\frac{A x+b}{c^{T}x+d}}
$$

    is called a linear-fractional function, defined on $c^{T}x+d>0$ . If $C\subseteq\operatorname{dom}(f)$ is convex then so is $f(C)$ , and if D is convex then so is $f^{-1}(D)$ .

> 保持凸性的运算：
> 仿射变换（$f(x) = Ax + b$）及其逆变换
> 透视变换（$P(x, z) = x/z$）及其逆变换
> 线性分式变换及其逆变换
#### 2.2.5.2 Example: linear matrix inequality solution set
Given $A_{1},...,A_{k},B\in\mathbb{S}^{n}$ , a linear matrix inequality is of the form

$$
x_{1}A_{1}+x_{2}A_{2}+...+x_{k}A_{k}\preceq B
$$

for a variable $x\in\mathbb{R}^{k}$ . Let’s prove the set $C$ of points $x$ that satisfy the above inequality is convex.
> 关于 $\mathbb S^n$ 中的线性矩阵不等式的解集是凸集

Approach 1: directly verify that $x,y\,\in\,C\,\Rightarrow\,t x\,+\,(1\,-\,t)y\,\in\,C$ . This follows by checking that, for any $v$ ,

$$
v^{T}\left(B-\sum_{i=1}^{k}(t x_{i}+(1-t)y_{i})A_{i})\right)v\geq0
$$

Approach 2: let $f:\mathbb{R}^{k}\rightarrow\mathbb{S}^{n}$ , $\begin{array}{r}{f({ x})={ B}-\sum_{i=1}^{k}x_{i}A_{i}}\end{array}$ . Note that $C=f^{-1}(\mathbb{S}_{+}^{n})$ , affine preimage of convex set.
#### 2.2.5.3 Example: conditional probability set
Let $U,V$ be random variables over $\{1,...,n\}$ , $\{1,...,m\}$ . Let $C\subseteq\mathbb{R}^{n m}$ be a set of joint distributions for $U,V$ , i.e., each $p\in C$ defines joint probabilities

$$
p_{i j}=\mathbb{P}(U=i,V=j)
$$

Let $D\subseteq\mathbb{R}^{n m}$ contain corresponding conditional distributions , i.e., each $q\in D$ defines

$$
q_{i j}=\mathbb{P}(U=i|V=j)
$$

Assume $C$ is convex. Let’s prove that $D$ is convex. Write

$$
D=\left\{q\in\mathbb{R}^{n m}:q_{i j}={\frac{p_{i j}}{\sum_{k=1}^{n}p_{k j}}},{\mathrm{~for~some~}}p\in C\right\}=f(C)
$$

where $f$ is a linear-fractional function, hence $D$ is convex.
# 2.3 Convex Functions

# 2.3.1 Definitions

Definition 2.8 Convex function: $f$ : $\mathbb{R}^{n}\to\mathbb{R}$ such that the domain of function $f\,d o m(f)\subseteq\mathbb{R}^{n}$ is convex.

$$
f(t x+(1-t)y)\leq t f(x)+(1-t)f(y),\ f o r\ 0\leq t\leq1
$$

And all $x,y\in d o m(f)$

In other words, the function lies below the line segment joining $f(x)$ and $f(y)$

Definition 2.9 Concave function: opposite inequality of the definition above, so that

$$
f\ c o n c a v e\Leftrightarrow-f\ c o n v e x
$$

which is to say, f being concave implies -f being convex.

# Important modifiers:

• Strictly Convex : $f(t x+(1-t)y)<t f(x)+(1-t)f(y)$ , for $x\neq y$ and $0<t<1$ . In other words, f is convex and has greater curvature than a linear function. • Strongly Convex : With parameter $m>0$ , $f{\bigl(}{-}{\frac{m}{2}}||x||_{2}^{2}{\bigr)}$ ) is convex. In other words, f is at least as convex as a quadratic function.

Note : strongly convex implies strictly convex, which subsequently implies convex. In equation format:

$$
s t r o n g l y\ c o n v e x\Rightarrow s t r i c t l y\ c o n v e x\Rightarrow c o n v e x
$$

# 2.3.2 Examples of convex and concave functions

Univariate functions

(1) Exponential function: $e^{a x}$ is convex for any $a$ over $\mathbb{R}$ (2) Power function: $x^{a}$ is convex for $a\geq1$ or $a\leq0$ over $\mathbb{R}_{+}$ (nonnegative reals); $x^{a}$ is concave for $0\leq a\leq1$ over $\mathbb{R}_{+}$ (3) Logarithmic function: $l o g(x)$ is concave over $R_{++}$

• Affine function: $a^{T}x+b$ is both convex and concave. • Quadratic function: ${\scriptstyle{\frac{1}{2}}}x^{T}Q x+b^{T}x+c$ is convex provided that $Q\geq0$ (positive semidefinite) $\bullet$ Least squares loss: $||y-A x||_{2}^{2}$ is always convex (since $A^{T}A$ is always positive semidefinite) $\bullet$ $||x||$ is convex for any norm, for example: $l_{p}$ norms

$$
||x||_{p}=(\sum_{i=1}^{n}x_{p}^{i})^{1/p}\,f o r\,p\geq1,||x||_{\infty}=\operatorname*{max}_{i=1,\ldots n}|x_{i}|
$$

as well as operator (spectral) and trace (nuclear) norms

$$
||X||_{o p}=\sigma_{1}(X),||X||_{t r}=\sum_{i=1}^{r}\sigma_{r}(X)
$$

where $\sigma_{1}(X)\geq...\geq\sigma_{r}(X)\geq0$ are the singular values of the matrix X. 
• Indicator function: if $C$ is convex, then its indicator function

$$
I_{C}(x)={\left\{\begin{array}{l l}{0,x\in C}\\ {\infty,x\not\in C}\end{array}\right.}
$$

is convex

• Support function: for any set $C$ (convex or not), its support function

$$
I_{C}^{*}(x)=\operatorname*{max}_{y\in C}x^{T}y
$$

is convex

• Max function: $f(x)=m a x\{x_{1},...x_{n}\}$ is convex.

# 2.3.3 Key properties of convex functions

• A function is convex if and only if its restriction to any line is convex

• Epigraph characterization: a function $f$ is convex if and only if its epigraph

$$
e p i(f)=(x,t)\in d o m(f)\times\mathbb{R}:f(x)\leq t
$$

is a convex set.

• Convex sublevel sets: if $f$ is convex, then its sublevel sets

$$
x\in d o m(f):f(x)\leq t
$$

are convex, for all $t\in\mathbb R$ . The converse is not true.

• First-order characterization: if $f$ is diﬀerentiable, then $f$ is convex if and only if dom (f) is convex, and

$$
f(y)\geq f(x)+\nabla f(x)^{T}(y-x)
$$

for all $x,y\in d o m(f)$ . Therefore for a diﬀerentiable convex function $\nabla f(x)=0\Leftrightarrow x$ minimizes $f$ .

• Second-order characterization: if $f$ is twice diﬀerentiable, then $f$ is convex if and only if $d o m(f)$ is convex, and $\nabla^{2}f(x)\geq0$ for all $x\in d o m(f)$ .

• Jensen’s inequality: if $f$ is convex, and $X$ is a random variable supported on $d o m(f)$ , then $f(\mathbb{E}[\mathbb{X}])\leq$ $\mathbb{E}[f(x)]$ .

• Long-sum-exp function: $g(x)\,=\,l o g(\sum_{i=1}^{k}e^{a_{i}^{T}x+b_{i}})$ ) for fixed $a_{i},b_{i}$ . This is often called the soft max, since it smoothly approximates max $\scriptstyle\operatorname*{max}_{i=1,\ldots,k}\left(a_{i}^{T}x+b_{i}\right)$ ).

# 2.3.4 Operations preserving convexity

• Nonnegative linear combination: $f_{1},...f_{m}$ convex implies $a_{1}f_{1}+\ldots+a_{m}f_{m}$ is also convex for any $a_{1},...a_{m}\geq0$ .

• Pointwise max mization: if $f_{s}$ is convex for any $s\in S$ , then $f(x)=m a x_{s\in S}$ is also convex. Note : the set S is the number of functions $f_{x}$ , which can be infinite. 
• Partial minimization: if $g(x,y)$ is convex in , and $C$ is convex, then $f(x)=m i n_{y\in C}g(x,y)$ is convex. $x, y

$

• Affine composition: if $f$ is convex, then $g(x)=f(A x+b)$ is convex.

• General composition: suppose $f=h g$ , where $g:\mathbb{R}^{n}\rightarrow\mathbb{R},h:\mathbb{R}\rightarrow\mathbb{R},f:\mathbb{R}^{n}\rightarrow\mathbb{R}$ . Then: (1) $f$ is convex if $h$ is convex and nondecreasing, is convex $g$ (2) $f$ is convex if $h$ is convex and nonincreasing, is concave $g$ (3) $f$ is concave if $h$ is concave and nondecreasing, $g$ is concave (4) $f$ is convex if $h$ is convex and nonincreasing, is convex $g$ Note : To memorize this, think of the chain rule when $n=1$ :

$$
f^{\prime\prime}(x)=h^{\prime\prime}(g(x))g^{\prime}(x)^{2}+h^{\prime}(g(x))g^{\prime\prime}(x)
$$

• Vector composition: suppose that:

$$
\boldsymbol{f}(\boldsymbol{x})=\boldsymbol{h}(\boldsymbol{g}(\boldsymbol{x}))=\boldsymbol{h}(\boldsymbol{g}_{1}(\boldsymbol{x}),...,\boldsymbol{g}_{k}(\boldsymbol{x}))
$$

where $g:\mathbb{R}^{n}\rightarrow\mathbb{R}^{k},h:\mathbb{R}^{k}\rightarrow\mathbb{R},f:\mathbb{R}^{n}\rightarrow\mathbb{R}$ . Then: (1) $f$ is convex if h is convex and nondecreasing in each argument, $g$ is convex (2) $f$ is convex if $h$ is convex and nonincreasing in each argument, is concave $g$ (3) $f$ is concave if $h$ is concave and nondecreasing in each argument, $g$ is concave (4) $f$ is concave if $h$ is concave and nonincreasing in each argument, is convex $g$

