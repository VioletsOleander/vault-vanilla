(1) 字典序法
首先计算 83674521 在字典序中的序号：

$$
7\times 7! + 2\times 6! + 4\times 5! + 4\times 4 ! + 2 \times 3! + 2\times 2! + 1\times 1! = 37313
$$

因此其中介数为 $(7244221)\uparrow$

因为 $2024 = 2 \times 6 ! + 4\times 5! + 4 \times 4! + 1\times 3! + 1\times 2 ! + 0\times 1!$，故 2024 的递增进位制数为 $(244110)\uparrow$

计算：

$$
\begin{align}
&&(7244221)&\uparrow \\
-& &(244110)&\uparrow\\
=& & (7000111)&\uparrow
\end{align}
$$

中介数 $(700111)\uparrow$ 对应的排列是：$81245673$

(2) 递增进位制数法
首先计算 83674521 映射到的中介数，
假设中介数为 $a_7\dots a_1$，根据定义：
$a_7 = 7,$
$a_6 = 4$,
$a_5 = 5$,
$a_4 = 2$,
$a_3 = 3$,
$a_2 = 2$,
$a_1 = 1$

因此中介数为 $(7452321)\uparrow$ 

计算：

$$
\begin{align}
&&(7452321)&\uparrow \\
-& &(244110)&\uparrow\\
=& & (7204211)&\uparrow
\end{align}
$$

中介数 $(7204211)\uparrow$ 对应的排列是：

$$
\begin{align}
&\underline 8\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\\

&\underline 8\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\underline 7\;\underline ~\;\underline ~\;\\

&\underline 8\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\underline 7\;\underline ~\;\underline 6\;\\

&\underline 8\;\underline 5\;\underline ~\;\underline ~\;\underline ~\;\underline 7\;\underline ~\;\underline 6\;\\

&\underline 8\;\underline 5\;\underline ~\;\underline 4\;\underline ~\;\underline 7\;\underline ~\;\underline 6\;\\

&\underline 8\;\underline 5\;\underline ~\;\underline 4\;\underline 3\;\underline 7\;\underline ~\;\underline 6\;\\

&\underline 8\;\underline 5\;\underline 2\;\underline 4\;\underline 3\;\underline 7\;\underline 1\;\underline 6\;\\
\end{align}
$$

即 85243716

(3) 递减进位制数法
将递增进位制数翻转，得到递减进位制数 $(1232547)\downarrow$

计算：

$$
\begin{align}
&&(1232547)&\downarrow \\
-& &(011442)&\downarrow\\
=& & (1221105)&\downarrow
\end{align}
$$

中介数 $(1221105)\downarrow$ 对应的排列是：

$$
\begin{align}
&\underline ~\;\underline ~\;\underline 8\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\\

&\underline ~\;\underline ~\;\underline 8\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\underline 7\;\\

&\underline ~\;\underline ~\;\underline 8\;\underline ~\;\underline ~\;\underline 6\;\underline ~\;\underline 7\;\\

&\underline ~\;\underline ~\;\underline 8\;\underline ~\;\underline 5\;\underline 6\;\underline ~\;\underline 7\;\\

&\underline ~\;\underline 4\;\underline 8\;\underline ~\;\underline 5\;\underline 6\;\underline ~\;\underline 7\;\\


&\underline 3\;\underline 4\;\underline 8\;\underline 2\;\underline 5\;\underline 6\;\underline ~\;\underline 7\;\\


&\underline 3\;\underline 4\;\underline 8\;\underline 2\;\underline 5\;\underline 6\;\underline 1\;\underline 7\;\\
\end{align}
$$

即 34825617

(4) 邻位对换法
给定排列 83674521
- 2的方向向左，背向2的方向中比2小的数字有1个， $b_2 = 1$
- 3的方向由 $b_2$ 为奇可以判断为向右，背向3的方向中比3小的数字有0个， $b_3 = 0$
- 4的方向由 $b_2 + b_3$ 为奇可以判断为向右，背向4的方向中比4小的数字有1个， $b_4 = 1$
- 5的方向由 $b_4$ 为奇可以判断为向右，背向5的方向中比5小的数字有2个， $b_5 = 2$
- 6的方向由 $b_4 + b_5$ 为奇可以判断为向右，背向6的方向中比6小的数字有1个， $b_6 = 1$
- 7的方向由 $b_6$ 为奇可以判断为向右，背向7的方向中比7小的数字有2个， $b_7 = 2$
- 8的方向由 $b_6 + b_7$ 为奇可以判断为向右，背向8的方向中比8小的数字有0个， $b_8 = 0$

因此中介数为 $(b_2b_3\dots b_8)\downarrow = (1012120)\downarrow$

计算：

$$
\begin{align}
&&(1012120)&\downarrow \\
-& &(011442)&\downarrow\\
=& & (1000135)&\downarrow
\end{align}
$$

8:  $b_7 + b_6 = 4$ 为偶，故8向左；$b_8 = 5$，故8为向左第6个空
7:  $b_6 = 1$ 为偶，故7向右；$b_7 = 3$，故7为向右第4个空
6:  $b_5+b_4 = 0$ 为偶，故6向左；$b_6 = 1$，故6为向左第2个空
5:  $b_4 = 0$ 为偶，故5向左；$b_5 = 0$，故5为向左第1个空
4:  $b_3+b_2 = 1$ 为奇，故4向右；$b_4 = 0$，故4为向右第1个空
3:  $b_2 = 1$ 为奇，故3向右；$b_3 = 0$，故3为向右第1个空
2:  $b_2 = 1$ 故2为向左第2个空

$$
\begin{align}
&\underline ~\;\underline ~\;\underline 8\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\underline ~\;\\

&\underline ~\;\underline ~\;\underline 8\;\underline ~\;\underline 7\;\underline ~\;\underline ~\;\underline ~\;\\


&\underline ~\;\underline ~\;\underline 8\;\underline ~\;\underline 7\;\underline ~\;\underline 6\;\underline ~\;\\

&\underline ~\;\underline ~\;\underline 8\;\underline ~\;\underline 7\;\underline ~\;\underline 6\;\underline 5\;\\

&\underline 4\;\underline ~\;\underline 8\;\underline ~\;\underline 7\;\underline ~\;\underline 6\;\underline 5\;\\

&\underline 4\;\underline 3\;\underline 8\;\underline 2\;\underline 7\;\underline ~\;\underline 6\;\underline 5\;\\

&\underline 4\;\underline 3\;\underline 8\;\underline 2\;\underline 7\;\underline 1\;\underline 6\;\underline 5\;\\
\end{align}
$$

即 43827165