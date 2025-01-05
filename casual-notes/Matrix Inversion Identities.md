我们先推导两个恒等式，然后借助这两个恒等式推导矩阵逆定理 (Matrix Inversion Lemma)，或者称为 Sherman-Morrison-Woodbury 恒等式

Identity 1

$$
\begin{align}
(I + P)^{-1} &= (I+P)^{-1}(I+P-P)\\
&=(I+P)^{-1}(I+P) - (I+P)^{-1}P\\
&=I - (I+P)^{-1}P\tag{1}
\end{align}
$$

Identity 2

$$
\begin{align}
P + PQP &= P(I+QP) = (I+PQ)P\\
(I+PQ)^{-1}P &=P(I+QP)^{-1}\tag{2}
\end{align}
$$

Matrix Inversion Lemma: step 1
考虑可逆矩阵 $A$，以及矩阵 (一般为长方形矩阵) $B, C, D$，有：

$$
\begin{align}
(A + BCD)^{-1}&=\left(A\left[I + A^{-1}BCD\right]\right)^{-1}\\
&=\left[I + A^{-1}BCD\right]^{-1}A^{-1}\\
&=\left[I - (I + A^{-1}BCD)^{-1}A^{-1}BCD\right]A^{-1}\tag{Using (1)}\\
&=A^{-1} - (I+ A^{-1}BCD)^{-1}A^{-1}BCDA^{-1}
\end{align}
$$

Matrix Inversion Lemma: step 2
反复使用 (2)：

$$
\begin{align}
(A + BCD)^{-1} 
&= A^{-1}  - (I + A^{-1}BCD)^{-1}A^{-1}BCDA^{-1}\tag{3}\\
&= A^{-1}  - A^{-1}(I + BCDA^{-1})^{-1}BCDA^{-1}\tag{4}\\
&= A^{-1} - A^{-1}B(I+CDA^{-1}B)^{-1}CDA^{-1}\tag{5}\\
&=A^{-1} - A^{-1}BC(I + DA^{-1}BC)^{-1}DA^{-1}\tag{6}\\
&=A^{-1} - A^{-1}BCD(I + A^{-1}BCD)^{-1}A^{-1}\tag{7}\\
&=A^{-1} - A^{-1}BCDA^{-1}(I + BCDA^{-1})^{-1}\tag{8}\\
\end{align}
$$

Matrix Inversion Lemma: special case
如果 $C$ 可逆，则从 (5) 可以得到：

$$
\begin{align}
(A + BCD)^{-1}  &= A^{-1} - A^{-1}B(I + CDA^{-1}B)^{-1}CDA^{-1}\\
&=A^{-1} - A^{-1}B(C^{-1} + DA^{-1}B)^{-1}DA^{-1}\tag{9}
\end{align}
$$

这里，我们将 $(I + CDA^{-1}B)^{-1}C$ 替换为了 $(C^{-1} + DA^{-1}B)^{-1}$，推导如下：

如果 $C$ 可逆，有：

$$
\begin{align}
(I + CDA^{-1}B) &= C(C^{-1} + DA^{-1}B)\\
(I + CDA^{-1}B)^{-1} &= (C^{-1} + DA^{-1}B)^{-1}C^{-1}\\
(I + CDA^{-1}B)^{-1}C^{-1} &= (C^{-1} + DA^{-1}B)^{-1}\\
\end{align}
$$

Another related special case

$$
\begin{align}
(A + BCD)^{-1}BC &= A^{-1}(I + BCDA^{-1})^{-1}BC\\
&=A^{-1}B(I + CDA^{-1}B)^{-1}C\\
&=\text{and for invertible }C\tag{10}\\
&=A^{-1}B(C^{-1} + DA^{-1}B)^{-1}\tag{11}
\end{align}
$$




