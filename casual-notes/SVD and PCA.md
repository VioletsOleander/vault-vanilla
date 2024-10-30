# 1 Four Fundamental Subspaces
对于任意一个$m\times n$的实矩阵$A\in\mathbb R^{m\times n}$，有四个与其相关的子空间
1. 列空间(Column space)：$\text C(A)\in\mathbb R^m$
	矩阵的列空间，即矩阵的列向量张成的空间，是$\mathbb R^m$的子空间
2. 行空间(Row space)：$\text C(A^T)\in\mathbb R^n$
	矩阵的行空间，也矩阵的转置的列空间(感觉写成$\text R(A)$也挺直观的)，即矩阵的行向量张成的空间，是$\mathbb R^n$的子空间
3. 零空间(Null space)：$\text N(A)\in\mathbb R^n$
	矩阵的零空间，即$Ax = 0$的所有解向量张成的空间，是$\mathbb R^n$的子空间，
	容易知道，$\text N(A)$中所有的向量都与A的行空间中的所有向量正交，即$\text N(A)$与$\text C(A^T)$正交，
	对比左零空间的定义，可知$Ax=0$这个式子是矩阵$A$右乘一个向量得零，说明这个向量和矩阵的行空间正交，所以也可以叫它矩阵的右零空间
4. 左零空间(Left nullspace)：$\text N(A^T)\in \mathbb R^m$
	矩阵的左零空间，即$A^{T}x = 0$的所有解向量张成的空间，是$\mathbb R^m$的子空间，
	$\text N(A^T)$中的所有向量都与A的列空间的所有向量正交，即$\text N(A^T)$与$\text C(A)$正交，
	之所以叫左零空间，是因为$A^{T}x = 0$可以写成$x^{T}A = 0$，即一个矩阵$A$左乘一个向量得零，说明这个向量和矩阵的列空间正交

三秩相等定理：
对于矩阵$A$，$A$的秩是$r$，$A$的列秩是$r(A)$，$A$的行秩/$A^T$的列秩是$r(A^T)$
则$A$的列秩$r(A)$等于$A$的行秩$r(A^T)$等于$A$的秩，即$r(A) = r(A^T)=r$
证明思路：
矩阵的秩是由最大非奇异子阵(非奇异：行列式不为$0$)的大小决定的，而矩阵转置，行列式不变，则容易得到$r(A^T) = r(A)$
要完整证明三秩相等，之后要再证明矩阵的秩等于其列秩(利用行列式和极大无关组的性质来证)，即可得到三秩相等定理

四个子空间的维度(正交基的个数)：
1. 列空间的维度：$\text {dim}[\text C(A)] = r(A)$
	列空间的维度即列秩数，也就是矩阵的秩
2. 行空间的维度：$\text {dim}[\text C(A^T)] = r(A^T)= r(A)$
	行空间的维度即行秩数，参考三秩相等定理，它等于列秩数/矩阵的秩
3. 零空间的维度：$\text{dim}[\text N(A)] = n - r(A) = n - \text{dim}[\text C(A^T)]$
	$A$的零空间和$A$的行空间正交，容易知道二者的交集是空集，并集是全集，
	即$\text N(A)\cup\text C(A^T) = \mathbb R^n$，
	因此零空间的维度即$\mathbb R^n$的秩数$n$减去行空间的维度/矩阵的秩，
	且两个空间的正交基的并集就是$\mathbb R^n$的一组正交基
4. 左零空间的维度：$\text {dim}[\text N(A^T)] = m - r(A) = m - \text {dim}[\text C(A)]$
	$A$的左零空间和$A$的列空间正交，容易知道二者的交集是空集，并集是全集，
	即$\text N(A^T)\cup \text C(A) = \mathbb R^m$，
	因此左零空间的维度即$\mathbb R^m$的秩数$m$减去列空间的维度/矩阵的秩，
	且两个空间的正交基的并集就是$\mathbb R^m$的一组正交基
# 2 SVD(Singular Value Decomposition)
对于任意一个大小为$m\times n$，秩为$r$的实矩阵$A\in \mathbb R^{m\times n}$，有以下结论：

结论1：
对于实矩阵$A$，有$r(A) = r(A^{T}) = r(AA^{T})= r(A^{T}A)$
证明：
考虑两个式子$Ax = 0$和$A^TAx = 0$
对于$\forall x_i \in \mathbb R^n$，若$x_i$满足$Ax_i = 0$，则显然有$A^TAx_i = A^T(Ax_i) = 0$，
因此，满足$Ax_i = 0$的$x_i$一定满足$A^TAx_i = 0$，
即$Ax=0$的解集是$A^TAx = 0$的解集的子集

对于$\forall x_j \in \mathbb R^n$，若$x_j$满足$A^TAx_j = 0$，则$x_j^TA^TAx_j = x_j^T(A^TAx_j) = 0$，则
$$\begin{align}
x_j^TA^TAx_j&=0\\
(x_j^TA^T)(Ax_j)&=0\\
(Ax_j)^T(Ax_j)&=0
\end{align}$$
因为$(Ax_j)^T(Ax_j) \ge 0$，当且仅当$Ax_j$的所有元素都为$0$时，等号成立，故可以推出$Ax_j$的所有元素都为$0$，也就是$Ax_j = 0$，
因此，满足$A^TAx_j = 0$的$x_j$一定满足$Ax_j = 0$，
即$A^TAx=0$的解集是$Ax = 0$的解集的子集

两个集合相互为子集，则两个集合相等，也就是$Ax = 0$和$A^TAx = 0$的解集相等，即$Ax = 0$和$A^{T}Ax = 0$同解，故$A$和$A^TA$零空间相同

则$A$和$A^TA$的零空间的秩也相同，即$n-r(A^{T}A) = n-r(A)$，由此得到$A$和$A^TA$的行空间的秩也相同，即$r(A) = r(A^{T}A)$

事实上，因为$A$和$A^TA$的行空间都是$\mathbb R^n$的子空间，且与各自的零空间正交，并互补，故由$A$和$A^TA$的零空间相同，可以得到$A$和$A^TA$的行空间相同

基于结论1的推论：
因为$Ax = 0$和$A^{T}Ax = 0$同解，则
$A$的零空间和$A^TA$的零空间相同，即$\text N(A) = \text N(A^TA)$；
行空间和零空间正交，那么$A$的行空间和$A^TA$的行空间相同，即$\text R(A) = \text R(A^TA)$；
(注意$A^TA$是对称矩阵，因此$A^TA$的行空间和列空间相同，零空间和左零空间相同)

在结论1中，用$A^T$替换$A$，得到$A^Tx = 0$和$AA^{T}x = 0$同解，则
$A^T$零空间/$A$的左零空间和$AA^T$的零空间/左零空间相同，即$\text N(A^T) = \text N(AA^{T})$；
左零空间和列空间正交，那么$A$的列空间和$AA^T$的列空间相同，即$\text C(A) = \text C(AA^T)$
(注意$AA^T$是对称矩阵，因此$AA^T$的行空间和列空间相同，零空间和左零空间相同)

因此对于矩阵$A$的四大子空间，我们都可以找到和其相同的子空间：
行空间：$\text C(A^T) = \text C(A^TA)$/$\text R(A) = \text R(A^TA)$
零空间：$\text N(A) = \text N(A^TA)$
列空间：$\text C(A) = \text C(AA^T)$
左零空间：$\text N(A^T) = \text N(AA^T)$
## 2.1 $A^TA$
考虑矩阵$A^TA\in \mathbb R^{n\times n}$，易知$A^TA$的秩为$r$

考虑$A^{T}A$的性质：
- 对称性
	因为$A^{T}A = (AA^{T})^T$，故$A^{T}A$是实对称矩阵
- 半正定性
	对于任意非零$n$元实向量$x\in\mathbb R^n$，有$x^{T}A^{T}Ax = (Ax)^{T}Ax = \|Ax\|^2\geqslant 0$，
	故$A^{T}A$是半正定矩阵，$A^{T}A$的特征值都大于等于$0$
	(因为对于$A^{T}A$的特征向量$p$来说，$p^{T}A^{T}Ap = \lambda p^{T}p = \lambda\|p\|^2 \geqslant 0$，则$\lambda \geqslant 0$)

$A^{T}A$作为实对称矩阵，一定可以相似对角化，即$A^{T}A$存在$n$个相互正交的特征向量，并且由于$A^{T}A$是半正定矩阵，其相应的特征值都大于等于$0$，
因此对$A^TA$进行特征值分解得到：
$$A^{T}A = V\Lambda V^T$$

而因为$r(A^{T}A) = r(V\Lambda V^{T}) = r(\Lambda) = r$，
可知$A^{T}A$的$n$个特征值中，$r$个大于$0$，$n-r$个为$0$
($r(V\Lambda V^{T}) = r(\Lambda)$是因为$V$是正交阵，满秩，可逆)

对于$A^TA$的特征值大于$0$的特征向量：
令它们为：$v_1,v_2,\cdots,v_r$，且令$V_{1}=[v_1,v_2,\cdots,v_r]$
$V_1$的形状为$n\times r$，$V_1$是正交阵
对于$A^TA$的特征值等于$0$的特征向量：
令它们为：$v_{r+1},v_{r+2},\cdots,v_n$，且令$V_{2}=[v_{r+1},v_{r+2},\cdots,v_n]$
$V_2$的形状为$n\times (n-r)$，$V_2$是正交阵
$V = [V_1,V_2]$

可知$V_2$是$A^{T}A$的零空间的一组标准正交基，也同时是$A$的零空间的一组标准正交基
而$V_1$与$V_2$正交，
可知$V_1$是$A^{T}A$的行空间的一组标准正交基，也同时是$A$的行空间的一组标准正交基
## 2.2 $AA^T$
同样的，考虑矩阵$AA^T\in\mathbb R^{m\times n}$，易知它是一个秩为$r$的半正定矩阵
对$AA^T$也可以相似对角化：$$AA^T=U\Lambda' U^T$$
而由于$r(AA^T)=r(U\Lambda'U^T)=r(\Lambda')=r$，可以知道$\Lambda'$中有$r$个特征值大于$0$，$n-r$个特征值为$0$

对于特征值大于$0$的特征向量：
令它们为：$[u_1,u_2,u_3,\cdots,u_r]$，且令$U_1=[u_1,u_2,u_3,\cdots,u_r]$
$U_1$的形状为$m\times r$，$U_1$是正交阵
对于特征值等于$0$的特征向量：
令它们为：$[u_{r+1},u_{r+2},\cdots,u_m]$，且令$U_2=[u_{r+1},u_{r+2},\cdots,u_m]$
$U_2$的形状为$m\times(m-r)$，$U_2$是正交阵
$U=[U_1,U_2]$

可知$U_2$是$AA^{T}$的零空间的一组标准正交基，也同时是$A^T$的零空间的一组标准正交基，也就是$A$的左零空间的一组标准正交基
而$U_1$与$U_2$正交，
可知$U_1$是$AA^{T}$的行空间的一组标准正交基，也同时是$A^T$的行空间的一组标准正交基，也就是$A$的列空间的一组标准正交基
## 2.3 Singular Values
而$V_1,U_1$之间，也就是$A$的行空间的标准正交基和$A$的列空间的标准正交基有什么关系？
$V_2,U_2$之间，也就是$A$的零空间的标准正交基和$A$的左零空间的标准正交基有什么关系？
$V,U$之间，也就是$A^TA$的特征向量和$AA^T$的特征向量之间有什么关系？

我们已经知道$A^TA$的秩是$r$，
故对于其大于零的特征值，有对应的特征向量$v_1,v_2,\cdots,v_r$
并且有：$$A^TAv_i=\lambda_iv_i\quad\lambda_i\ne0$$
将等式两边都左乘$A$：
$$\begin{aligned}AA^TAv_i&=\lambda_iAv_i\quad(\lambda_i\ne0)\\(AA^T)(Av_i)&=\lambda_i(Av_i)\quad(\lambda_i\ne0)\end{aligned}$$
发现：
**对于$A^TA$的特征向量$v_i$($\lambda_i\ne0)$，$Av_i$是$AA^T$的特征向量**
且对于$v_i$，$A^TA$的特征值是$\lambda_i$
对于$Av_i$，$AA^T$的特征值也是$\lambda_i$

对于其等于零的特征值，有对应的特征向量$v_{r+1},v_{r+2},\cdots,v_n$，
并且有：
$$A^TAv_i=0$$
将等式两边都左乘$A$：
$$\begin{aligned}AA^TAv_i&=0\\(AA^T)(Av_i)&=0\end{aligned}$$
和前面的形式很像，但没有同样的结论，
这是因为$A^TA$和$A$的零空间是一样的，容易知道：
$$A^{T}Av_{i}=0\ {\rightarrow}\ Av_i=0$$
因此$\lambda_i=0$时，$Av_i=0$，而零向量显然是$AA^T$的零空间中的向量之一，但我们能确定$(AA^T)(Av_i)=0$是成立的

因此，对于$A^TA$的特征值$\lambda_i$，不论$\lambda_i$是否为$0$，等式$(AA^T)(Av_i)=\lambda_i(Av_i)$都成立

这个结论可以两种情况在一起推导，即对于$A^TA$，有：
$$\begin{align}
A^{T}A&=V\Lambda V^{T}\\
(A^TA)V&=V\Lambda \\
A(A^TA)V &=AV\Lambda\\
(AA^T)(AV)&=(AV)\Lambda
\end{align}$$
其中$V=[V_1,V_2]=[v_1,\cdots,v_r,v_{r+1},\cdots,v_n]$

同理，对于$AA^T$，我们已经知道$AA^T$的秩也是$r$，
故对于其大于零的特征值，有对应的特征向量$u_1,u_2,\cdots,u_r$
并且有：
$$AA^Tu_i=\lambda'_iu_i\quad\lambda'_i\ne0$$
将等式两边都左乘$A^T$：
$$\begin{aligned}A^TAA^Tu_i&=\lambda'_iA^Tu_i\quad(\lambda'_i\ne0)\\(A^TA)(A^Tu_i)&=\lambda'_i(A^Tu_i)\quad(\lambda'_i\ne0)\end{aligned}$$
发现：
**对于$AA^T$的特征向量$u_i$($\lambda'_i\ne0)$，$A^Tu_i$是$A^TA$的特征向量**
且对于$u_i$，$AA^T$的特征值是$\lambda'_i$
对于$A^Tu_i$，$A^TA$的特征值也是$\lambda'_i$

对于其等于零的特征值，有对应的特征向量$u_{r+1},u_{r+2},\cdots,v_m$，
并且有：
$$AA^Tu_i=0$$
将等式两边都左乘$A^T$：
$$\begin{aligned}A^TAA^Tu_i&=0\\(A^TA)(A^Tu_i)&=0\end{aligned}$$
和前面的形式很像，但没有同样的结论，
这是因为$AA^T$和$A^T$的零空间是一样的，容易知道：
$$AA^{T}u_{i}=0\ {\rightarrow}\ A^Tu_i=0$$
因此$\lambda'_i=0$时，$A^Tu_i=0$，而零向量显然是$AA^T$的零空间中的向量之一，但我们能确定$(A^TA)(A^Tu_i)=0$是成立的

因此，对于$AA^T$的特征值$\lambda_i'$，不论$\lambda_i'$是否为$0$，等式$(A^TA)(A^Tu_i)=\lambda'_i(A^Tu_i)$都成立

这个结论可以两种情况在一起推导，即对于$AA^T$，有：
$$\begin{align}
AA^T&=U\Lambda' U^{T}\\
(AA^T)U&=U\Lambda' \\
A^T(AA^T)U &=A^TU\Lambda'\\
(A^TA)(A^TU)&=(A^TU)\Lambda
\end{align}$$
其中$U=[U_1,U_2]=[u_1,\cdots,u_r,u_{r+1},\cdots,u_m]$

从上述式子我们可以看出：
**当特征值$\lambda_i\ne0$，对于一个$A^TA$的特征向量$v_i$，一定有一个$AA^T$的特征向量$Av_i$与其对应，相应的特征值相等：$\lambda'_i=\lambda_i$**
将其归一化，写为$\sigma_iu_i(Av_i=\sigma_iu_i)$，其中$\sigma_i$是$Av_i$的长度，$\sigma_i\geqslant0$，$u_i$是单位向量，即：
$$v_{i}\ \rightarrow u_{i}\quad \lambda_i=\lambda'_i\ne0$$
左乘一个$A$是确定的运算，因此**一个$v_i$只能对应一个$u_i$**

同理，对于$AA^T$的特征向量，可以令$A^Tu_i=\sigma'_iv_i$，同样有：
$$u_{i}\ \rightarrow \ v_{i}\quad\lambda'_i=\lambda_i\ne0$$
左乘一个$A^T$是确定的运算，因此**一个$u_i$只能对应一个$v_i$**

并且：
先将$v_i$映射成$u_i$
$$Av_i=\sigma_iu_i\rightarrow u_i=\frac{Av_i}{\sigma_i}$$
再从$u_i$映射回来
$$A^T\sigma_iu_i=A^TAv_i=\lambda_iv_i\rightarrow v_i=\frac{A^Tu_i}{\lambda_i/\sigma_i}$$
依旧得到$v_i$
即：
$$v_i\leftrightarrow u_{i}\quad\lambda_i\ne0$$
这说明：
$A^TA$和$AA^T$的那$r$个大于$0$的**特征值$\lambda_1,\cdots,\lambda_r$是一样的**，相对应的**特征向量$v_i$和$u_i$也是一一对应的**，$v_i$左乘$A$后就和$u_i$平行，$u_i$左乘$A^T$后就和$v_i$平行
(**这其实也说明了$A$的行空间的一组正交基和$A$的列空间的一组正交基是对应的，可以相互转化**)
而$A^TA$和$AA^T$剩余的特征值也都是$0$，其中$A^TA$有$n-r$个$0$特征值，$AA^T$有$m-r$个$0$特征值

如果从$u_i$开始映射到$v_i$，可以得到：
$$A^Tu_i=\sigma'_iv_i\rightarrow v_i=\frac{A^Tu_i}{\sigma'_i}$$
再映射回来：
$$A\sigma'_iv_i=AA^Tu_i=\lambda_iu_{i}\rightarrow u_i=\frac{Av_i}{\lambda_i/\sigma'_i}$$
结合$$v_i=\frac{A^Tu_i}{\sigma'_i}$$和$$v_i=\frac{A^Tu_i}{\lambda_i/\sigma_i}$$容易得到：$$\lambda_i=\sigma_i\sigma'_i$$其中$$\sigma_i=\|Av_i\|\quad\sigma'_i=\|A^Tu_i\|$$进一步推导：
$$\begin{aligned}\sigma_i&=\|Av_i\|=\sqrt{(Av_i)^T(Av_i)}\\\sigma_i^2&=v_i^TA^TAv_i=\lambda_iv_i^Tv_i=\lambda_i\end{aligned}$$
$$\begin{aligned}\sigma'_i&=\|A^Tu_i\|=\sqrt{(A^Tu_i)^T(A^Tu_i)}\\\sigma_{i}'^{2}&=u_i^TAA^Tu_i=\lambda_iu_i^Tu_i=\lambda_i\end{aligned}$$
显然：$$\sigma_i=\sigma'_i=\sqrt{\lambda_i}=\|Av_i\|=\|A^Tu_i\|$$
$r$个$A^TA$和$AA^T$**共同的**非零的特征值$\lambda_1,\cdots,\lambda_r$，对应了$r$个同样非零的值$\sigma_1=\sqrt{\lambda_1},\cdots,\sigma_r=\sqrt{\lambda_r}$，我们把它们称为$A$和$A^T$的奇异值(singular value)
一般我们把它们进行从大到小排列，即：
$$\lambda_1\geqslant\cdots\geqslant\lambda_r\gt0$$
$$\sigma_1\geqslant\cdots\geqslant\sigma_r\gt0$$
## 2.4 Singular Value Decomposition
我们已经知道了$A$的行空间的标准正交基$V_1$和$A$的列空间的标准正交基$U_1$是可以相互转化的：
$$AV_1=A[v_1,\cdots,v_r]=[\sigma_1u_1,\cdots,\sigma_ru_r]$$
$V_1$中每一列是$A^TA$的特征向量中对应$r$个非零特征值的$r$个单位特征向量
$V$中还有剩余的$n-r$个对应的特征值为$0$的特征向量
$U_1$中每一列是$AA^T$的特征向量中对应$r$个非零特征值的$r$个单位特征向量
$U$中还有剩余的$m-r$个对应的特征值为$0$的特征向量

如果把等式左边的$V_1$扩充为$V$：
$$AV=A[v_1,\cdots,v_r,v_{r+1},\cdots,v_n]=[\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]$$
($A$和$A^TA$的零空间相同，因此有$A^TAV_2=0\rightarrow AV_2=0$)

等式的右边是一个$(m\times n)\times(n\times n)\rightarrow m\times n$的矩阵

我们构造一个$m\times n$的矩阵$\Sigma$：
$$\Sigma=
\begin{bmatrix} 
\sigma_1 & \cdots & \cdots & \cdots & \cdots &0\\ 
\vdots&\ddots& &&&\vdots\\ 
\vdots&&\sigma_r&&&\vdots\\
\vdots& & &0&&\vdots\\
\vdots& & & & \ddots&\vdots\\
0&\cdots & \cdots&\cdots &\cdots &0
\end{bmatrix}$$
$\Sigma$中从$(1,1)$元到$(r,r)$元排列着$r$个奇异值$\sigma$，其余元素都是$0$

可以知道：
$$U\Sigma =[u_1,\cdots,u_r,u_{r+1},\cdots,u_m]\Sigma=[\sigma_1u_1,\cdots,\sigma_ru_r,0,\cdots,0]$$
等式右边是一个$(m\times m)\times (m\times n)\rightarrow m\times n$的矩阵

显然：$$AV=U\Sigma$$
$V$是正交阵，因此：$$A=U\Sigma V^T$$
**此即$A$的奇异值分解**

式子中
$U:m\times m$，为标准正交阵，是$\mathbb R^m$的一组标准正交基
其中的前$r$列是$A$的列空间的标准正交基，后$m-r$列是$A$的左零空间的标准正交基
$V:n\times n$，为标准正交阵，是$\mathbb R^n$的一组标准正交基
其中的前$r$列是$A$的行空间的标准正交基，后$n-r$列是$A$的零空间的标准正交基
$\Sigma:m\times n$
其中有$r\times r$子阵，子阵的对角线上是$A$的奇异值$\sigma$
一般$\Sigma$中的奇异值都是从大到小排列：$$\sigma_1\geqslant\cdots\geqslant\sigma_r\gt0$$
奇异值分解说明了：
任意一个形状的矩阵都可以被分解为左右两个正交阵，中间一个$\Sigma$的形式
也就是说，对于一个$n$维向量，任意一个线性变换(左乘$A$)都可以看成三个过程：
- 旋转(左乘$V^T$)，或者说变基(变为$V^T$中的基)
- 放缩，前$r$行的放缩系数分别是$\sigma_i$，后$m-r$行变为$0$(如果$m<n$，会使得向量维数降低，如果$m>n$，会使得向量维数升高)
- 旋转(左乘$U$)，或者说变基(变为$U$中的基)
最后得到经过线性变换后的$m$维向量
(如果把放缩和第二次旋转结合，也可以说先旋转一次，变为$V^T$中的基，然后对$U$中的前$r$个基放缩，放缩系数分别是$\sigma_i$，并将$U$的后$m-r$个基消除，然后再旋转一次，变为$U\Sigma$中的基，显然这次旋转的过程中，会导致向量的后$m-r$元变为$0$，同样，如果$m<n$，会使得向量维数降低，如果$m>n$，会使得向量维数升高)

如果把$$A = U\Sigma V^T$$展开
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
$A$被分解为$r$个秩为$1$的矩阵的和

我们知道
$$\sigma_iu_i = Av_i\quad u_i=\frac{Av_i}{\sigma_i}$$
$$\sigma_iv_i = A^Tu_i\quad v_i=\frac{A^Tu_i}{\sigma_i}$$
那么将$$\sigma_iu_i = Av_i$$代入容易得到
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
如果$r(A) = n$我们可以得到
$$V_1V_1^T = E$$
同理，如果$r(A) = m$我们可以得到
$$U_1^TU = E$$
当然这个结论也很显然
因为$r(A) = n$时，$A^TA$满秩，可逆，无零特征值
$r(A) = m$时，$AA^T$满秩，可逆，无零特征值
但要注意在$r(A)< m\ \ \&\&\ \  r(A) < n$时没有这个结论
但无论如何这个等式可以看作先将$A$的行向量投影到$V_1$中的每一列代表的基上，然后再用投影值乘以$V_1^T$中的每一行代表的基上，线性组合复原$A$的每一行
另一种对称的有关$U$的情况同理