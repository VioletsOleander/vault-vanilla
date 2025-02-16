# 1 Four Fundamental Subspaces
对于任意一个 $m\times n$ 的实矩阵 $A\in\mathbb R^{m\times n}$，有四个与其相关的子空间
1. 列空间 (Column space)：$\text C(A)\in\mathbb R^m$
	矩阵的列空间，即矩阵的列向量张成的空间，是 $\mathbb R^m$ 的子空间
2. 行空间 (Row space)：$\text C(A^T)\in\mathbb R^n$
	矩阵的行空间，也矩阵的转置的列空间 (感觉写成 $\text R(A)$ 也挺直观的)，即矩阵的行向量张成的空间，是 $\mathbb R^n$ 的子空间
3. 零空间 (Null space)：$\text N(A)\in\mathbb R^n$
	矩阵的零空间，即 $Ax = 0$ 的所有解向量张成的空间，是 $\mathbb R^n$ 的子空间，
	容易知道，$\text N(A)$ 中所有的向量都与 A 的行空间中的所有向量正交，即 $\text N(A)$ 与 $\text C(A^T)$ 正交，
	对比左零空间的定义，可知 $Ax=0$ 这个式子是矩阵 $A$ 右乘一个向量得零，说明这个向量和矩阵的行空间正交，所以也可以叫它矩阵的右零空间
4. 左零空间 (Left nullspace)：$\text N(A^T)\in \mathbb R^m$
	矩阵的左零空间，即 $A^{T}x = 0$ 的所有解向量张成的空间，是 $\mathbb R^m$ 的子空间，
	$\text N(A^T)$ 中的所有向量都与 A 的列空间的所有向量正交，即 $\text N(A^T)$ 与 $\text C(A)$ 正交，
	之所以叫左零空间，是因为 $A^{T}x = 0$ 可以写成 $x^{T}A = 0$，即一个矩阵 $A$ 左乘一个向量得零，说明这个向量和矩阵的列空间正交

三秩相等定理：
对于矩阵 $A$，$A$ 的秩是 $r$，$A$ 的列秩是 $r(A)$，$A$ 的行秩/ $A^T$ 的列秩是 $r(A^T)$
则 $A$ 的列秩 $r(A)$ 等于 $A$ 的行秩 $r(A^T)$ 等于 $A$ 的秩，即 $r(A) = r(A^T)=r$

证明思路：
矩阵的秩是由最大非奇异子阵 (非奇异：行列式不为 $0$) 的大小决定的，而矩阵转置，行列式不变，则容易得到 $r(A^T) = r(A)$
要完整证明三秩相等，之后要再证明矩阵的秩等于其列秩 (利用行列式和极大无关组的性质来证)，即可得到三秩相等定理

四个子空间的维度 (正交基的个数)：
1. 列空间的维度：$\text {dim}[\text C(A)] = r(A)$
	列空间的维度即列秩数，也就是矩阵的秩
2. 行空间的维度：$\text {dim}[\text C(A^T)] = r(A^T)= r(A)$
	行空间的维度即行秩数，参考三秩相等定理，它等于列秩数/矩阵的秩
3. 零空间的维度：$\text{dim}[\text N(A)] = n - r(A) = n - \text{dim}[\text C(A^T)]$
	$A$ 的零空间和 $A$ 的行空间正交，容易知道二者的交集是空集，并集是全集，
	即 $\text N(A)\cup\text C(A^T) = \mathbb R^n$，
	因此零空间的维度即 $\mathbb R^n$ 的秩数 $n$ 减去行空间的维度/矩阵的秩，
	且两个空间的正交基的并集就是 $\mathbb R^n$ 的一组正交基
4. 左零空间的维度：$\text {dim}[\text N(A^T)] = m - r(A) = m - \text {dim}[\text C(A)]$
	$A$ 的左零空间和 $A$ 的列空间正交，容易知道二者的交集是空集，并集是全集，
	即 $\text N(A^T)\cup \text C(A) = \mathbb R^m$，
	因此左零空间的维度即 $\mathbb R^m$ 的秩数 $m$ 减去列空间的维度/矩阵的秩，
	且两个空间的正交基的并集就是 $\mathbb R^m$ 的一组正交基

# 2 SVD (Singular Value Decomposition)
对于任意一个大小为 $m\times n$，秩为 $r$ 的实矩阵 $A\in \mathbb R^{m\times n}$，有以下结论：

结论一：
对于实矩阵 $A$，有 $r(A) = r(A^{T}) = r(AA^{T})= r(A^{T}A)$

证明：
考虑两个式子 $Ax = 0$ 和 $A^TAx = 0$
对于 $\forall x_i \in \mathbb R^n$，若 $x_i$ 满足 $Ax_i = 0$，则显然有 $A^TAx_i = A^T(Ax_i) = 0$，
因此，满足 $Ax_i = 0$ 的 $x_i$ 一定满足 $A^TAx_i = 0$，
即 $Ax=0$ 的解集是 $A^TAx = 0$ 的解集的子集

对于 $\forall x_j \in \mathbb R^n$，若 $x_j$ 满足 $A^TAx_j = 0$，则 $x_j^TA^TAx_j = x_j^T(A^TAx_j) = 0$，则

$$\begin{align}
x_j^TA^TAx_j&=0\\
(x_j^TA^T)(Ax_j)&=0\\
(Ax_j)^T(Ax_j)&=0
\end{align}$$

因为 $(Ax_j)^T(Ax_j) \ge 0$，当且仅当 $Ax_j$ 的所有元素都为 $0$ 时，等号成立，故可以推出 $Ax_j$ 的所有元素都为 $0$，也就是 $Ax_j = 0$，
因此，满足 $A^TAx_j = 0$ 的 $x_j$ 一定满足 $Ax_j = 0$，
即 $A^TAx=0$ 的解集是 $Ax = 0$ 的解集的子集

两个集合相互为子集，则两个集合相等，也就是 $Ax = 0$ 和 $A^TAx = 0$ 的解集相等，即 $Ax = 0$ 和 $A^{T}Ax = 0$ 同解，故 $A$ 和 $A^TA$ 零空间相同

则 $A$ 和 $A^TA$ 的零空间的秩也相同，即 $n-r(A^{T}A) = n-r(A)$，由此得到 $A$ 和 $A^TA$ 的行空间的秩也相同，即 $r(A) = r(A^{T}A)$ ，证毕

事实上，因为 $A$ 和 $A^TA$ 的行空间都是 $\mathbb R^n$ 的子空间，且与各自的零空间正交，并互补，故由 $A$ 和 $A^TA$ 的零空间相同，可以得到 $A$ 和 $A^TA$ 的行空间相同

基于结论一的推论：
因为 $Ax = 0$ 和 $A^{T}Ax = 0$ 同解，则 $A$ 的零空间和 $A^TA$ 的零空间相同，即 $\text N(A) = \text N(A^TA)$。而行空间和零空间正交，那么 $A$ 的行空间和 $A^TA$ 的行空间相同，即 $\text R(A) = \text R(A^TA)$
(注意 $A^TA$ 是对称矩阵，因此 $A^TA$ 的行空间和列空间相同，零空间和左零空间相同)

在结论一中，用 $A^T$ 替换 $A$，得到 $A^Tx = 0$ 和 $AA^{T}x = 0$ 同解，则 $A^T$ 零空间/ $A$ 的左零空间和 $AA^T$ 的零空间/左零空间相同，即 $\text N(A^T) = \text N(AA^{T})$。左零空间和列空间正交，那么 $A$ 的列空间和 $AA^T$ 的列空间相同，即 $\text C(A) = \text C(AA^T)$
(注意 $AA^T$ 是对称矩阵，因此 $AA^T$ 的行空间和列空间相同，零空间和左零空间相同)

因此对于矩阵 $A$ 的四大子空间，我们都可以找到和其相同的子空间：
行空间：$\text C(A^T) = \text C(A^TA)$ / $\text R(A) = \text R(A^TA)$
零空间：$\text N(A) = \text N(A^TA)$
列空间：$\text C(A) = \text C(AA^T)$
左零空间：$\text N(A^T) = \text N(AA^T)$

## 2.1 $A^TA$
考虑矩阵 $A^TA\in \mathbb R^{n\times n}$，易知 $A^TA$ 的秩为 $r$

考虑 $A^{T}A$ 的性质：
- 对称性
	因为 $A^{T}A = (AA^{T})^T$，故 $A^{T}A$ 是实对称矩阵
- 半正定性
	对于任意非零 $n$ 元实向量 $x\in\mathbb R^n$，有 $x^{T}A^{T}Ax = (Ax)^{T}Ax = \|Ax\|^2\geqslant 0$，
	故 $A^{T}A$ 是半正定矩阵，$A^{T}A$ 的特征值都大于等于 $0$
	(因为对于 $A^{T}A$ 的特征向量 $p$ 来说，$p^{T}A^{T}Ap = \lambda p^{T}p = \lambda\|p\|^2 \geqslant 0$，则 $\lambda \geqslant 0$)

$A^{T}A$ 作为实对称矩阵，一定可以相似对角化，即 $A^{T}A$ 存在 $n$ 个相互正交的特征向量，并且由于 $A^{T}A$ 是半正定矩阵，其相应的特征值都大于等于 $0$，
因此对 $A^TA$ 进行特征值分解得到：

$$A^{T}A = V\Lambda V^T$$

而因为 $r(A^{T}A) = r(V\Lambda V^{T}) = r(\Lambda) = r$，可知 $A^{T}A$ 的 $n$ 个特征值中，$r$ 个大于 $0$，$n-r$ 个为 $0$ ($r(V\Lambda V^{T}) = r(\Lambda)$ 是因为 $V$ 是正交阵，满秩，可逆)

对于 $A^TA$ 的特征值大于 $0$ 的特征向量：
令它们为：$v_1,v_2,\cdots,v_r$，且令 $V_{1}=[v_1,v_2,\cdots,v_r]$
$V_1$ 的形状为 $n\times r$，$V_1$ 是正交阵
对于 $A^TA$ 的特征值等于 $0$ 的特征向量：
令它们为：$v_{r+1},v_{r+2},\cdots,v_n$，且令 $V_{2}=[v_{r+1},v_{r+2},\cdots,v_n]$
$V_2$ 的形状为 $n\times (n-r)$，$V_2$ 是正交阵
$V = [V_1,V_2]$

可知 $V_2$ 是 $A^{T}A$ 的零空间的一组标准正交基，也同时是 $A$ 的零空间的一组标准正交基
而 $V_1$ 与 $V_2$ 正交，
可知 $V_1$ 是 $A^{T}A$ 的行空间的一组标准正交基，也同时是 $A$ 的行空间的一组标准正交基

## 2.2 $AA^T$
同样的，考虑矩阵 $AA^T\in\mathbb R^{m\times n}$，易知它是一个秩为 $r$ 的半正定矩阵
对 $AA^T$ 也可以相似对角化：

$$AA^T=U\Lambda' U^T$$

而由于 $r(AA^T)=r(U\Lambda'U^T)=r(\Lambda')=r$，可以知道 $\Lambda'$ 中有 $r$ 个特征值大于 $0$，$n-r$ 个特征值为 $0$

对于特征值大于 $0$ 的特征向量：
令它们为：$[u_1,u_2,u_3,\cdots,u_r]$，且令 $U_1=[u_1,u_2,u_3,\cdots,u_r]$
$U_1$ 的形状为 $m\times r$，$U_1$ 是正交阵
对于特征值等于 $0$ 的特征向量：
令它们为：$[u_{r+1},u_{r+2},\cdots,u_m]$，且令 $U_2=[u_{r+1},u_{r+2},\cdots,u_m]$
$U_2$ 的形状为 $m\times(m-r)$，$U_2$ 是正交阵
$U=[U_1,U_2]$

可知 $U_2$ 是 $AA^{T}$ 的零空间的一组标准正交基，也同时是 $A^T$ 的零空间的一组标准正交基，也就是 $A$ 的左零空间的一组标准正交基
而 $U_1$ 与 $U_2$ 正交，
可知 $U_1$ 是 $AA^{T}$ 的行空间的一组标准正交基，也同时是 $A^T$ 的行空间的一组标准正交基，也就是 $A$ 的列空间的一组标准正交基

## 2.3 Singular Values
而 $V_1,U_1$ 之间，也就是 $A$ 的行空间的标准正交基和 $A$ 的列空间的标准正交基有什么关系？
$V_2,U_2$ 之间，也就是 $A$ 的零空间的标准正交基和 $A$ 的左零空间的标准正交基有什么关系？
$V,U$ 之间，也就是 $A^TA$ 的特征向量和 $AA^T$ 的特征向量之间有什么关系？

我们已经知道 $A^TA$ 的秩是 $r$，故对于其大于零的特征值，有对应的特征向量 $v_1,v_2,\cdots,v_r$ ，它们满足：

$$A^TAv_i=\lambda_iv_i\quad\lambda_i\ne0$$

将等式两边都左乘 $A$：

$$\begin{aligned}AA^TAv_i&=\lambda_iAv_i\quad(\lambda_i\ne0)\\(AA^T)(Av_i)&=\lambda_i(Av_i)\quad(\lambda_i\ne0)\end{aligned}$$

发现：
**对于 $A^TA$ 的特征向量 $v_i$ ($\lambda_i\ne0)$，$Av_i$ 是 $AA^T$ 的特征向量**
且对于 $v_i$，$A^TA$ 的特征值是 $\lambda_i$
对于 $Av_i$，$AA^T$ 的特征值也是 $\lambda_i$

对于其等于零的特征值，有对应的特征向量 $v_{r+1},v_{r+2},\cdots,v_n$，它们满足：

$$A^TAv_i=0$$

将等式两边都左乘 $A$：

$$\begin{aligned}AA^TAv_i&=0\\(AA^T)(Av_i)&=0\end{aligned}$$

和前面的形式很像，但没有同样的结论，这是因为 $A^TA$ 和 $A$ 的零空间是一样的，故：

$$A^{T}Av_{i}=0\ {\Rightarrow}\ Av_i=0$$

因此 $\lambda_i=0$ 时，$Av_i=0$，而零向量显然是 $AA^T$ 的零空间中的向量之一，在定义上，零向量不能作为特征向量，但我们能确定的是此时 $(AA^T)(Av_i)=0$ 是成立的

因此，对于 $A^TA$ 的特征值 $\lambda_i$，不论 $\lambda_i$ 是否为 $0$，等式 $(AA^T)(Av_i)=\lambda_i(Av_i)$ 都成立

这个结论也可以两种情况在一起推导，即对于 $A^TA$，有：

$$\begin{align}
A^{T}A&=V\Lambda V^{T}\\
(A^TA)V&=V\Lambda \\
A(A^TA)V &=AV\Lambda\\
(AA^T)(AV)&=(AV)\Lambda
\end{align}$$

其中 $V=[V_1,V_2]=[v_1,\cdots,v_r,v_{r+1},\cdots,v_n]$

同理，对于 $AA^T$，我们已经知道 $AA^T$ 的秩也是 $r$，故对于其大于零的特征值，有对应的特征向量 $u_1,u_2,\cdots,u_r$ ，它们满足：

$$AA^Tu_i=\lambda'_iu_i\quad\lambda'_i\ne0$$

将等式两边都左乘 $A^T$：

$$\begin{aligned}A^TAA^Tu_i&=\lambda'_iA^Tu_i\quad(\lambda'_i\ne0)\\(A^TA)(A^Tu_i)&=\lambda'_i(A^Tu_i)\quad(\lambda'_i\ne0)\end{aligned}$$

发现：
**对于 $AA^T$ 的特征向量 $u_i$ ($\lambda'_i\ne0)$，$A^Tu_i$ 是 $A^TA$ 的特征向量**
且对于 $u_i$，$AA^T$ 的特征值是 $\lambda'_i$
对于 $A^Tu_i$，$A^TA$ 的特征值也是 $\lambda'_i$

对于其等于零的特征值，有对应的特征向量 $u_{r+1},u_{r+2},\cdots,u_m$，它们满足：

$$AA^Tu_i=0$$

将等式两边都左乘 $A^T$：

$$\begin{aligned}A^TAA^Tu_i&=0\\(A^TA)(A^Tu_i)&=0\end{aligned}$$

和前面的形式很像，但没有同样的结论，这是因为 $AA^T$ 和 $A^T$ 的零空间是一样的，故：

$$AA^{T}u_{i}=0\ {\Rightarrow}\ A^Tu_i=0$$

因此 $\lambda'_i=0$ 时，$A^Tu_i=0$，而零向量显然是 $AA^T$ 的零空间中的向量之一，零向量不能作为特征向量，但我们能确定的是 $(A^TA)(A^Tu_i)=0$ 是成立的

因此，对于 $AA^T$ 的特征值 $\lambda_i'$，不论 $\lambda_i'$ 是否为 $0$，等式 $(A^TA)(A^Tu_i)=\lambda'_i(A^Tu_i)$ 都成立

这个结论也可以两种情况在一起推导，即对于 $AA^T$，有：

$$\begin{align}
AA^T&=U\Lambda' U^{T}\\
(AA^T)U&=U\Lambda' \\
A^T(AA^T)U &=A^TU\Lambda'\\
(A^TA)(A^TU)&=(A^TU)\Lambda
\end{align}$$

其中 $U=[U_1,U_2]=[u_1,\cdots,u_r,u_{r+1},\cdots,u_m]$

我们考虑上述推导得到的两个结论
首先，根据第一个结论，可以知道：**当特征值 $\lambda_i\ne0$，对于一个 $A^TA$ 的特征向量 $v_i$，一定有一个 $AA^T$ 的特征向量 $Av_i$ 与其对应，相应的特征值相等，即 $\lambda_i=\lambda_i'$**
我们将该特征向量 $Av_i$ 归一化，写为 $\sigma_iu_i(Av_i=\sigma_iu_i)$，其中 $\sigma_i$ 是 $Av_i$ 的长度，$\sigma_i\geqslant0$，$u_i$ 是单位向量，那么我们就有了 $v_i$ 到 $u_i$ 的一个对应关系如下：

$$v_{i}\ \rightarrow u_{i}\quad \lambda_i=\lambda'_i\ne0$$

左乘一个 $A$ 是确定的运算，因此**一个 $v_i$ 只能对应一个 $u_i$**

同理，根据第二个结论，可以知道：**当特征值 $\lambda_i' \ne 0$，对于一个 $AA^T$ 的特征向量 $u_i$，一定有一个 $A^TA$ 的特征向量 $A^Tu_i$ 与其对应，相应的特征值相等，即** $\lambda_i' = \lambda_i$
我们将该特征向量 $A^Tu_i$ 归一化，即 $A^Tu_i=\sigma'_iv_i$，我们类似地得到 $u_i$ 到 $v_i$ 的一个对应关系如下：

$$u_{i}\ \rightarrow \ v_{i}\quad\lambda'_i=\lambda_i\ne0$$

左乘一个 $A^T$ 是确定的运算，因此**一个 $u_i$ 只能对应一个 $v_i$**

我们进一步考虑这两个对应关系
先将 $v_i$ 映射成 $u_i$

$$Av_i=\sigma_iu_i\Rightarrow u_i=\frac{Av_i}{\sigma_i}$$

再从 $u_i$ 映射回来

$$A^T\sigma_iu_i=A^TAv_i=\lambda_iv_i\Rightarrow v_i=\frac{A^Tu_i}{\lambda_i/\sigma_i}$$

依旧得到 $v_i$
因此，我们得到了双向的对应关系如下：

$$v_i\leftrightarrow u_{i}\quad\lambda_i\ne0$$

这说明：
$A^TA$ 和 $AA^T$ 的那 $r$ 个大于 $0$ 的**特征值 $\lambda_1,\cdots,\lambda_r$ 是一样的**，相对应的**特征向量 $v_i$ 和 $u_i$ 也是一一对应的**，$v_i$ 左乘 $A$ 后就和 $u_i$ 平行，$u_i$ 左乘 $A^T$ 后就和 $v_i$ 平行
(**这其实也说明了 $A$ 的行空间的一组正交基和 $A$ 的列空间的一组正交基是对应的，可以相互转化**)
而 $A^TA$ 和 $AA^T$ 剩余的特征值也都是 $0$，其中 $A^TA$ 有 $n-r$ 个 $0$ 特征值，$AA^T$ 有 $m-r$ 个 $0$ 特征值

如果从 $u_i$ 开始映射到 $v_i$，可以得到：

$$A^Tu_i=\sigma'_iv_i\Rightarrow v_i=\frac{A^Tu_i}{\sigma'_i}$$

再映射回来：

$$A\sigma'_iv_i=AA^Tu_i=\lambda_iu_{i}\Rightarrow u_i=\frac{Av_i}{\lambda_i/\sigma'_i}$$

我们结合  $v_i=\frac{A^Tu_i}{\sigma'_i}$  和  $v_i=\frac{A^Tu_i}{\lambda_i/\sigma_i}$ ，容易得到：

$$\lambda_i=\sigma_i\sigma'_i$$

其中 

$$\sigma_i=\|Av_i\|\quad\sigma'_i=\|A^Tu_i\|$$ 
我们将 $\sigma_i, \sigma_i'$ 分别展开：

$$\begin{aligned}
\sigma_i&=\|Av_i\|=\sqrt{(Av_i)^T(Av_i)}\\
\sigma_i^2&=v_i^TA^TAv_i=v_i^T(A^TAv_i)=\lambda_iv_i^Tv_i=\lambda_i
\end{aligned}$$

$$\begin{aligned}
\sigma'_i&=\|A^Tu_i\|=\sqrt{(A^Tu_i)^T(A^Tu_i)}\\
\sigma_{i}'^{2}&=u_i^TAA^Tu_i=u_i^T(AA^Tu_i)=\lambda_iu_i^Tu_i=\lambda_i
\end{aligned}$$

进而有：

$$\sigma_i=\sigma'_i=\sqrt{\lambda_i}=\|Av_i\|=\|A^Tu_i\|$$

因此，$r$ 个 $A^TA$ 和 $AA^T$**共同的**非零的特征值 $\lambda_1,\cdots,\lambda_r$，对应了 $r$ 个同样非零的值 $\sigma_1=\sqrt{\lambda_1},\cdots,\sigma_r=\sqrt{\lambda_r}$，我们把它们称为 $A$ 和 $A^T$ 的奇异值 (singular value)
一般我们把它们进行从大到小排列，即：

$$\lambda_1\geqslant\cdots\geqslant\lambda_r\gt0$$
$$\sigma_1\geqslant\cdots\geqslant\sigma_r\gt0$$

## 2.4 Singular Value Decomposition
我们已经知道了 $A$ 的行空间的标准正交基 $V_1$ 和 $A$ 的列空间的标准正交基 $U_1$ 是可以相互转化的：

$$AV_1=A[v_1,\cdots,v_r]=[\sigma_1u_1,\cdots,\sigma_ru_r]$$

$V_1$ 中每一列是 $A^TA$ 的特征向量中对应 $r$ 个非零特征值的 $r$ 个单位特征向量
$V$ 中还有剩余的 $n-r$ 个对应的特征值为 $0$ 的特征向量
$U_1$ 中每一列是 $AA^T$ 的特征向量中对应 $r$ 个非零特征值的 $r$ 个单位特征向量
$U$ 中还有剩余的 $m-r$ 个对应的特征值为 $0$ 的特征向量

如果把等式左边的 $V_1$ 扩充为 $V$：

$$AV=A[v_1,\cdots,v_r,v_{r+1},\cdots,v_n]=[\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]$$

($A$ 和 $A^TA$ 的零空间相同，$A^TAV_2=0\Rightarrow AV_2=0$)

等式的右边是一个 $(m\times n)\times(n\times n)\rightarrow m\times n$ 的矩阵

我们构造一个 $m\times n$ 的矩阵 $\Sigma$：

$$\Sigma=
\begin{bmatrix} 
\sigma_1 & \cdots & \cdots & \cdots & \cdots &0\\ 
\vdots&\ddots& &&&\vdots\\ 
\vdots&&\sigma_r&&&\vdots\\
\vdots& & &0&&\vdots\\
\vdots& & & & \ddots&\vdots\\
0&\cdots & \cdots&\cdots &\cdots &0
\end{bmatrix}$$

$\Sigma$ 中从 $(1,1)$ 元到 $(r,r)$ 元排列着 $r$ 个奇异值 $\sigma$，其余元素都是 $0$

可以知道：

$$U\Sigma =[u_1,\cdots,u_r,u_{r+1},\cdots,u_m]\Sigma=[\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]$$

等式右边是一个 $(m\times m)\times (m\times n)\rightarrow m\times n$ 的矩阵

显然：

$$AV=U\Sigma$$

$V$ 是正交阵，因此：

$$A=U\Sigma V^T$$

**此即 $A$ 的奇异值分解**

式子中
$U:m\times m$，为标准正交阵，是 $\mathbb R^m$ 的一组标准正交基
其中的前 $r$ 列是 $A$ 的列空间的标准正交基，后 $m-r$ 列是 $A$ 的左零空间的标准正交基
$V:n\times n$，为标准正交阵，是 $\mathbb R^n$ 的一组标准正交基
其中的前 $r$ 列是 $A$ 的行空间的标准正交基，后 $n-r$ 列是 $A$ 的零空间的标准正交基
$\Sigma:m\times n$
其中有 $r\times r$ 子阵，子阵的对角线上是 $A$ 的奇异值 $\sigma$
一般 $\Sigma$ 中的奇异值都是从大到小排列：

$$\sigma_1\geqslant\cdots\geqslant\sigma_r\gt0$$

奇异值分解说明了：
任意一个形状的矩阵都可以被分解为左右两个正交阵，中间一个 $\Sigma$ 的形式
也就是说，对于一个 $n$ 维向量，任意一个线性变换 (左乘 $A$) 都可以看成三个过程：
- 旋转 (左乘 $V^T$)，或者说变基 (变为 $V^T$ 中的基)
- 放缩，前 $r$ 行的放缩系数分别是 $\sigma_i$，后 $m-r$ 行变为 $0$ (如果 $m<n$，会使得向量维数降低，如果 $m>n$，会使得向量维数升高)
- 旋转 (左乘 $U$)，或者说变基 (变为 $U$ 中的基)
最后得到经过线性变换后的 $m$ 维向量
(如果把放缩和第二次旋转结合，也可以说先旋转一次，变为 $V^T$ 中的基，然后对 $U$ 中的前 $r$ 个基放缩，放缩系数分别是 $\sigma_i$，并将 $U$ 的后 $m-r$ 个基消除，然后再旋转一次，变为 $U\Sigma$ 中的基，显然这次旋转的过程中，会导致向量的后 $m-r$ 元变为 $0$，同样，如果 $m<n$，会使得向量维数降低，如果 $m>n$，会使得向量维数升高)

如果把 

$$A = U\Sigma V^T$$

展开

$$A = [u_1,\cdots,u_r,u_{r+1},\cdots,u_m]\Sigma \begin{bmatrix}
v_1^T \\
\vdots \\
v_r^T \\
v_{r+1}^T \\
\vdots \\
v_n^T
\end{bmatrix} = [\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]\begin{bmatrix}
v_1^T \\
\vdots \\
v_r^T \\
v_{r+1}^T \\
\vdots \\
v_n^T
\end{bmatrix}$$

即

$$A = \sigma_1u_1v_1^T+\cdots+\sigma_ru_rv_r^T$$

$A$ 被分解为 $r$ 个秩为 $1$ 的矩阵的和

我们知道

$$\sigma_iu_i = Av_i\quad u_i=\frac{Av_i}{\sigma_i}$$

$$\sigma_iv_i = A^Tu_i\quad v_i=\frac{A^Tu_i}{\sigma_i}$$

那么将 

$$\sigma_iu_i = Av_i$$

代入容易得到

$$\begin{aligned}
A &= Av_iv_i^T+\cdots+Av_rv_r^T \\
&= A(v_iv_i^T+\cdots+v_rv_r^T) \\
& = A[v_i,\cdots,v_r]\begin{bmatrix}
v_i \\
\vdots \\
v_r
\end{bmatrix} \\
& = AV_1V_1^T
\end{aligned}$$

如果 $r(A) = n$ 我们可以得到

$$V_1V_1^T = E$$

同理，如果 $r(A) = m$ 我们可以得到

$$U_1^TU = E$$

当然这个结论也很显然
因为 $r(A) = n$ 时，$A^TA$ 满秩，可逆，无零特征值
$r(A) = m$ 时，$AA^T$ 满秩，可逆，无零特征值
但要注意在 $r(A)< m\ \ \&\&\ \  r(A) < n$ 时没有这个结论
但无论如何这个等式可以看作先将 $A$ 的行向量投影到 $V_1$ 中的每一列代表的基上，然后再用投影值乘以 $V_1^T$ 中的每一行代表的基上，线性组合复原 $A$ 的每一行
另一种对称的有关 $U$ 的情况同理

#  3 PCA (Principle Component Analysis)
## 3.1 Prerequisite
我们常常用一个矩阵表示数据集，例如，一个 $m\times n$ 的矩阵 $A$ 表示有 $m$ 个样本，每个样本用一个 $n$ 维的特征向量表示，也就是有 $n$ 个特征
如果需要做数据压缩，就希望能用更少的特征来表示样本，比如 $n$ 维的特征向量能不能压缩到 1 维，2 维，同时还保持了原来的大部分信息？

原来的一个样本用的是 $\mathbb R^n$ 空间的一个 $n$ 维向量表示，空间中有 $n$ 个相互正交的单位基向量，样本在每一个维度上的取值就是其特征向量在这个维度 (这个特征) 对应的单位基向量上的投影
比如当 $n=2$ ，空间中的两个单位基向量是 $e_1=[1,0]\quad e_2=[0,1]$
有样本 $x_1 = e_1 + 2e_2$，我们表示为 $x_1=[1,2]$
有样本 $x_2 = 2e_1 + e_2$，我们表示为 $x_2 = [2,1]$

如果想要降到 1 维，也就是只在 $\mathbb R^n$ 空间中选取了一个方向作为新的基向量，把每个样本的特征向量投影到这个方向，作为样本在这个方向上 (这个新特征上) 的取值

比如我们选取 $e = [\frac{\sqrt2}{2},\frac{\sqrt2}{2}]$ 作为新的基 (新的特征)，这个新的基 (新的特征) 是原来的基 (原来的特征) 的线性组合
将原来每个样本的特征向量投影到这个方向后，我们得到样本在这个新的特征上的取值 $x_1 = \frac{3\sqrt2}{2}$ $x_2= \frac{3\sqrt2}{2}$
我们将特征空间从 2 维压缩到了 1 维，只用一个数来表示样本
但这显然不是一个好的投影方向，$x_1$ 和 $x_2$ 在投影后失去了区分度了，也就是丢失了大部分信息 (可以区分不同样本的信息)

那么一个好的投影方向，就是在投影后仍然可以保持大部分信息，也就是保持住原来数据集中每个样本之间的区分度，在投影之后，原来离得近的样本可能会因为部分信息丢失失去区分度，但我们希望能尽量保持住样本之间的距离关系，原来离得远的样本还是离得比较远

我们用方差来描述一个总体离散程度，方差计算了样本离中心点距离的平方的平均值
总体 $X$ 的方差是：

$$Var(X) = E[(X-E[X])^2]$$

当把样本都投影到一个方向上，每个样本 $x_i$ 都用一个常数 $a_i$ 表示，我们得到一个总体 $(a_1,\cdots, a_m)$，这个方向上的样本方差就是：

$$Var = \frac{1}{m}\sum_{i = 1}^m (a_i - u)^2$$

其中 $u = \frac{1}{m}\sum_{i = 1}^{m}a_i$，即平均值
当均值为 $0$ 时，形式可以进一步简化：

$$Var = \frac{1}{m}\sum_{i = 1}^m a_i^2$$

方差衡量了这一批样本的离散程度，我们需要的投影方向就是可以使得投影后，方差最大的方向

如果仅仅只投影到一个方向上，方差就足够了。而如果要投影到两个及以上的方向，我们需要保证第二个方向和第一个方向是正交的，第三个方向和前两个方向都是正交的，以此类推
换句话说，我们要找到的是一组正交基，如果第二个方向和第一个方向不正交，说明第二个方向包含的部分信息可以由第一个方向表示，而我们希望的是第二个方向表示的信息是第一个方向不能表示的，同样，第三个方向的信息是前两个方向不能表示的

如果要衡量两个总体的线性相关程度，统计上用协方差：

$$Cov(x,y) = E[(x-E[x])(y-E[y])]$$

并且有：

$$Cov (x, x) = E[(x-E[x])^2]=Var (x)$$

如果两个总体的协方差为零，可以认为二者没有线性关联，详见[[#协方差与线性关联]]

把样本都投影到第一个方向上，每个样本 $x_i$ 都用一个常数 $a_i$ 表示
然后投影到第二个方向上，每个样本 $x_i$ 都用一个常数 $b_i$ 表示
此时，两个方向的协方差表示为：

$$Cov = \frac{1}{m}\sum_{i=1}^m (a_i-u_a)(b_i-u_b)$$

如果 $u_a = u_b = 0$，形式可以进一步简化：

$$Cov = \frac{1}{m}\sum_{i=1}^m a_ib_i$$

## 3.2 PCA
我们对问题的设置稍作总结
我们有一个数据集，用 $m\times n$ 的矩阵 $A$ 表示，$A$ 的每一行表示一个样本的 $n$ 维特征向量

$$A = \begin{bmatrix}
v_1^T \\
\vdots \\
v_m^T
\end{bmatrix}$$

我们令 $X = A^T$

$$X = [v_1,\cdots,v_m]$$

$X$ 的每一列表示一个样本的 $n$ 维特征向量

我们通过 PCA 需要满足：
(1) 找到一组正交基 $e_1,\cdots, e_n$ ($n$ 个相互正交的 $n$ 维单位向量)，将样本的特征向量都分别投影到这组正交基上

即我们需要找到一个矩阵 $P (n\times n)$

$$P = \begin{bmatrix}
e_1^T \\
\vdots\\
e_n^T
\end{bmatrix}$$

$$PA^T =PX= \begin{bmatrix}
e_1^T \\
\vdots\\
e_n^T
\end{bmatrix}
[v_1,\cdots,v_m] = \begin{bmatrix}
e_1^Tv_1 & \cdots &e_1^Tv_m \\
\vdots & \ddots &\vdots \\
e_n^Tv_1 & \cdots & e_n^Tv_m \\
\end{bmatrix} = Y(n \times m)$$

$Y$ 中的第 $i$ 列就是样本 $i$ 投影过后的新的特征向量，其中第一行是样本集的第一主成分，第二行是样本集的第二主成分

(2) 满足任意两个不同方向 $e_i, e_j$ 之间，样本的投影值 $e_i^Tv_1,\cdots, e_i^Tv_m$ 和 $e_j^Tv_1,\cdots, e_j^Tv_m$ 的协方差是 $0$

对 $Y$ 做归一化

$$Y' = Y - YH$$

其中其中 $H (m\times m)$ 是每个元素都是 $\frac 1 m$ 的矩阵

右乘 $H$ 将 $Y$ 的每一列都变为 $Y$ 中所有列的和的 $\frac 1 m$ (所有列的平均)，即 $YH$ 中的每一列的形式都是：

$$\begin{bmatrix}
\frac {e_1^T} m\sum_{i=1}^m v_i \\
\vdots \\
\frac {e_n^T} m\sum_{i=1}^m v_i 
\end{bmatrix}$$

因此，$Y - YH$ 即 $Y$ 的每一列都减去所有列的平均，即

$$Y-YH=\begin{bmatrix}
e_1^Tv_1-\frac {e_1^T} m\sum_{i=1}^m v_i & \cdots &e_1^Tv_m-\frac {e_1^T} m\sum_{i=1}^m v_i \\
\vdots & \ddots &\vdots \\
e_n^Tv_1-\frac {e_n^T} m\sum_{i=1}^m v_i & \cdots & e_n^Tv_m-\frac {e_n^T} m\sum_{i=1}^m v_i \\
\end{bmatrix}$$

令 $Y' = Y-YH$，有 

$$Y' = \begin{bmatrix}
e_1^T(v_1-\frac {1} m\sum_{i=1}^m v_i) & \cdots &e_1^T(v_m-\frac {1} m\sum_{i=1}^m v_i) \\
\vdots & \ddots &\vdots \\
e_n^T(v_1-\frac {1} m\sum_{i=1}^m v_i) & \cdots & e_n^T(v_m-\frac {1} m\sum_{i=1}^m v_i) \\
\end{bmatrix}$$

容易知道 $Y'$ 中的第 $i$ 列就是对样本 $i$ 新的特征向量进行归一化处理的结果

另外，因为 $Y' = Y-YH = PX - PXH = P (X-XH) = PX'$，故实际计算中往往把先把 $X$ 归一化得到 $X'$ 再直接计算 $Y'$

归一化也可以称作把特征进行中心化，即令每个特征的均值变为 $0$

可以把 $Y'$ 写得简洁一点

$$Y' (n\times m) = \begin{bmatrix}
a_1-u_a & \cdots & a_m-u_a \\
\vdots & \ddots & \vdots \\
n_1-u_n & \cdots & n_m-u_n
\end{bmatrix} = 
\begin{bmatrix}
(\vec a - \vec u_a)^T \\
\vdots \\
(\vec n - \vec u_n)^T
\end{bmatrix}$$

归一化处理的目的是方便我们计算协方差矩阵 $\frac 1 m C_{Y'}$

$$\begin{aligned}\frac 1 m C_{Y'}(n\times n) = \frac 1 m Y'Y'^T = \frac 1 m \begin{bmatrix}
(\vec a - \vec u_a)^T \\
\vdots \\
(\vec n - \vec u_n)^T
\end{bmatrix}
[\vec a-\vec u_a,\cdots,\vec n-\vec u_n]
\\ \\ =\begin{bmatrix}
Var_1&Cov_{12}&\cdots&Cov_{1n} \\
Cov_{21}&Var_2&&\vdots \\
\vdots& & \ddots&\vdots \\
Cov_{1n} & \cdots& \cdots& Var_n
\end{bmatrix}
\end{aligned}$$

我们希望协方差矩阵是对角阵，即任意两个主成分方向之间的协方差都是 $0$

$$\frac 1 m C_{Y'} =\begin{bmatrix}
Var_1&&& \\
&Var_2&& \\
& & \ddots& \\
& & & Var_n
\end{bmatrix}$$

(3) 满足 $e_1$ 是所有方向中，使得样本的投影值 (样本集的第一主成分) $e_1^Tv_1,\cdots, e_1^Tv_m$ 的方差是最大的那个方向，$e_2$ 是所有与 $e_1$ 正交的方向中，使得样本的投影值 (样本的第二主成分) $e_2^Tv_1,\cdots, e_2^Tv_m$ 的方差是最大的那个方向，依此类推

我们首先考虑满足构造一个满足前两个要求的投影矩阵
要满足前两个要求，我们需要找到正交阵 $P$，使得 $\frac 1 m C_{Y'}$ 为对角阵，也就是使得 

$$C_{Y'} = Y'Y'^T = PX'X'^{T}P^T$$

为对角阵，我们不妨将 $C_{Y'}$ 记作 $\Lambda$，即

$$PX'X'^{T}P^T = \Lambda$$

则

$$X'X'^{T} = P^T\Lambda P$$

容易知道 $X'X'^{T}$ 是对称矩阵，并且是半正定矩阵，因此 $X'X'^{T}$ 存在一组相互正交的 $n$ 维特征向量 $Q = [q_1,\cdots, q_n]$，可以对角化 $X'X'^{T}$，并且 $X'X'^{T}$ 的特征值都大于等于 $0$ ，即

$$X'X'^{T} = Q\Lambda Q^T$$

$$Q = P^T$$

一个很自然的想法是选择这些特征向量作为主成分方向

如果选择 $X'X'^{T}$ 的特征向量作为主成分方向，可以发现，我们可以满足第一个要求：主成分方向相互正交 (特征向量相互正交)，也可以满足第二个要求：不同主成分方向之间的协方差为 $0$ ($\Lambda = C_{Y'}$ 为对角矩阵)

并且，此时 $\Lambda$ 对角线上的特征值的 $\frac 1 m$ 就是对应的主成分方向上的方差。$\Lambda$ 中一般把特征值从大到小排列

$$\lambda_1\geqslant\cdots\lambda_n\geqslant 0 $$

特征值最大的特征向量即对应第一主成分方向，依此类推

我们进而考虑把 $X'X'^{T}$ 的特征向量作为主成分方向，是否能够满足第三个要求
比如把 $q_1$ 作为第一主成分方向的话，一定保证 $q_1$ 是所有方向中使得方差最大的那个方向吗？

容易知道

$$Var_1=\frac 1 m \sum_{i=1}^m (q_1^Tv_i-\frac 1 m\sum_{i = 1}^mq_1^Tv_i)^2=\frac 1 m \sum_{i=1}^m [q_1^T(v_i-\frac 1 m\sum_{i = 1}^mv_i)]^2$$

平方和的形式实际上可以认为在求一个向量的长度

$$Var_1 =\frac 1 m \|(q_1^TX')^T\|^2 = \frac 1 m \|X'^Tq_1\|^2 = \frac 1 m q_1^TX'X'^Tq_1 = \frac {\lambda_1} m $$

如果将 $q_1$ 换成 $\mathbb R^n$ 中的任意一个其他单位向量比如 $t$ ，因为 $q_1,\cdots, q_n$ 构成了 $\mathbb R^n$ 的一组正交基，我们可以把 $t$ 写为

$$t = c_1q_1+\cdots+c_nq_n$$

其中由 $\|t\| = 1$ 可得

$$c_1^2+\cdots+c_n^2 = 1$$

因此 

$$\begin{aligned}
Var_t &= \frac 1 mt^TX'X'^Tt \\
& = \frac 1 m \sum_{i = 1}^nc_iq_i^T
X'X'^T\sum_{i = 1}^nc_iq_i \\
& = \frac {\sum_{i=1}^n c_i\lambda_i} m \leqslant \frac {\lambda_1} m = Var_1
\end{aligned}$$

这可以说明 $q_1$ 就是所有方向中可以使得方差最大的方向

同理，可以证明，$q_2$ 就是所有和 $q_1$ 正交的方向中可以使得方差最大的方向，依此类推

至此，我们总结 PCA 的算法步骤
有 $m\times n$ 的矩阵 $A$
1. $X = A^T$
2. 归一化 $X' = X-XH$
3. 计算协方差矩阵 $C = \frac 1 m X'X'^{T}$
4. 对角化协方差矩阵 $\frac 1 m X'X'^{T} = \frac 1 m Q\Lambda Q^T$，$\Lambda$ 中特征值由大到小排列
5. $P = Q^T$，$Y = PX = Q^TX$

因此，PCA 的核心思想就是寻找一组正交基以对角化协方差矩阵，而这组正交基就是协方差矩阵的特征向量
要补充的两点
- 关于无偏估计
	这里推导中为了简化，计算样本方差和协方差中都是乘上 $\frac 1 m$，在实际中，为了得到总体方差的无偏估计，一般是乘上 $\frac 1 {m-1}$ (因为总体的均值是用样本的均值来估计的，因此自由度应该减一)
- 关于解释的方差
	容易知道，没有进行 PCA 之前，在原来的每个特征方向上也可以计算方差，所有的特征方向上的方差的和即 $\frac 1 {m-1}tr (X'X'^T)$ (tr 表示矩阵的迹)，显然
    
    $$\frac 1 {m-1}tr (X'X'^T) = \frac 1 {m-1} tr (\Lambda)$$
    
	我们知道对角矩阵 $\Lambda$ 中排列着每个主成分方向上的方差
	因此，进行 PCA 后，所有特征方向上的方差的和是不变的
	因此，每一个主成分 $q_i$ 解释的方差的比例就是它解释的方差的大小占所有的方向解释的方差的和的比例，即该特征值除以所有特征值的和
	$$explained\ variance\ of\ q_i = \frac {\lambda_i} {tr (\Lambda)}$$

## 3.3 Relation with SVD
$A$ 是原来的矩阵，$A' = A - HA$ 是进行**归一化之后的矩阵**，即每一行减去了所有行的均值
设 $r (A) = r$
$A = X^T, A' = X'^T$
容易知道 $A$ 和 $A'$ 的行空间相等，$X$ 和 $X'$ 的列空间相等，这四个子空间都相等
$R(A) = R(A') = C(X) = C(X')$

做 SVD，即对 $X'X'^T$ 进行对角化，即对 $A'^TA'$ 进行对角化

$$X'X'^T = Q\Lambda Q^T = A'^TA'$$

$Q$ 是我们需要的正交基，其中的每一列是一个基
其中的 $q_1,\cdots, q_r$ 是我们的主成分方向

而对 $A'$ 进行 SVD

$$A' = U\Sigma V^T$$

$\Sigma$ 中的奇异值 $\sigma_i$ 即 $\sqrt \lambda_i$
那么 $\sigma_i^2 = \lambda_i$ 即主成分方向 $q_i$ 上的样本方差的 $m-1$ 倍
因为 

$$样本方差 = \frac {\lambda_i} {m-1}$$

而 $\sigma_i$ 即主成分方向 $q_i$ 上的样本标准差的 $\sqrt{m-1}$ 倍
因为 

$$样本标准差 = \sqrt {\frac {\lambda_i} {m-1} } = \frac {\sigma_i} {\sqrt {m-1}}$$

我们容易知道 

$$Q=V$$
$$
A' = U\Sigma Q^T
$$
$$A' = [u_1,\cdots,u_m]\  \Sigma\  \begin{bmatrix}
q_1^T \\
\vdots \\
q_n^T
\end{bmatrix} = [\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]\ \begin{bmatrix}
q_1^T \\
\vdots \\
q_n^T
\end{bmatrix}$$
$$\begin{aligned}
A' &= \sigma_1u_1q_1^T+\cdots+\sigma_ru_rq_r^T \\
& = A'q_1q_1^T+\cdots+A'q_rq_r^T \\
\end{aligned}$$

注意到 $Aq_i$ 为一个列向量，其中的第 $j$ 行即第 $j$ 个样本投影到第 $i$ 个主成分方向上的投影值
这个等式依旧可以视为是利用投影后的投影值乘以对应的基，线性相加，复原原来的特征向量
即 $A'$ 中的第 $j$ 行等于 $a_jq_1^T+\cdots+r_jq_r^T$
其中 $a_j$ 到 $r_j$ 分别是第 $j$ 个样本从在第一个主成分方向上的取值到在第 $r$ 个主成分方向上的取值
$A'$ 写为 $r$ 个秩为 1 的矩阵的和的形式，可以认为与 $r$ 个主成分关联

而 $A'$ 是归一化之后的形式

$$A' = A-HA = (E-H)A$$
$$(E-H)A = (E-H)A(q_1q_1^T+\cdots+q_rq_r^T)$$

计算行列式可以知道 $E-H$ 是不可逆的
因此对于未做归一化的矩阵 $A$，没有相似的结论
因此 $A$ 的 PCA 是与 $A'$ 的 SVD 相关的，可以认为对 $A$ 做 PCA 就是对 $A'$ 做 SVD
