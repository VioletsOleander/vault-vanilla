# Problem 1
(1) $F_n$ 满足递推关系 $F_n = F_{n-1} + F_{n-2}\quad(n\ge 2)$，将 $G_n - 3G_{n-1} + G_{n-2}$ 中的 $G_n$ 都替换为 $F_{2n}$，得到

$$
\begin{align}
&F_{2n} - 3F_{2n-2} + F_{2n-4}\\
=&F_{2n-1} + F_{2n-2}- 3F_{2n-2} + F_{2n-4}\\
=&F_{2n-1} - 2F_{2n-2} + F_{2n-4}\\
=&F_{2n-2} + F_{2n-3} - 2F_{2n-2} + F_{2n-4}\\
=&-F_{2n-2} + F_{2n-3} + F_{2n-4}\\
=&-F_{2n-2}  + F_{2n-2}\\
=&0
\end{align}
$$

因此 $G_n - 3G_{n-1} + G_{n-2} = 0$

(2) 根据递推关系 $G_n - 3G_{n-1} + G_{n-2} = 0$，可知特征多项式为

$$
C(x) = x^2 - 3x + 1
$$

特征方程 $C (x) = 0$ 有两个一重根

$$
\alpha_1 = \frac {3 + \sqrt 5}{2}, \alpha_2 = \frac {3-\sqrt 5}{2}
$$
因此 $G_n$ 的通项表达式写为

$$
G_n = A_1 \alpha_1^n + A_2 \alpha_2^n
$$

因为 $G_0 = F_0 = 0,G_1 = F_2 = 1$，故可以建立方程

$$
\begin{cases}
A_1 + A_2 = G_0 = 0\\
A_1\alpha_1 + A_2 \alpha_2 = G_1 =  1
\end{cases}
$$

解得 

$$\begin{cases}
A_1 =  \frac 1 {(\alpha_1 - \alpha_2)} = \frac {1}{\sqrt 5}\\
A_2 = -A_1 = -\frac {1}{\sqrt 5}
\end{cases}$$

因此

$$
G_n = \frac {1}{\sqrt 5}\left(\frac {3 + \sqrt 5}{2}\right)^n - \frac {1}{\sqrt 5}\left(\frac {3 - \sqrt 5}{2}\right)^n \quad (n\ge 2)
$$

故 $G_n$ 的母函数为

$$
\boxed{G(x) = \sum_{n=2}^\infty \frac {1}{\sqrt 5}\left( \left(\frac {3 + \sqrt 5}{2}\right)^n - \left(\frac {3 - \sqrt 5}{2}\right)^n \right)x^n}
$$

# Problem 2
根据母函数的形式，可以写出

$$
\begin{align}
G(x) &= \frac {1}{1-x + x^2}\\
(1- x + x^2)G(x)&= 1\\
G(x)- xG(x) + x^2G(x) &= 1\\
\end{align}
$$

令母函数 $G (x) = \sum_{n=0}^\infty a_n x^n$，则

$$
\begin{align}
\sum_{n=0}^\infty a_n x^n - \sum_{n=1}^\infty a_{n-1}x^{n} + \sum_{n=2}^\infty a_{n-2}x^n &= 1\\
a_0 + a_1 x - a_0x + \sum_{n=2}^\infty(a_n - a_{n-1}+a_{n-2})x^n &=1
\end{align}
$$

因此

$$
\begin{cases}
a_0 = 1\\
a_1 - a_0 = 0\\
a_n - a_{n-1}  + a_{n-2} = 0
\end{cases}
$$

因此 $a_n$ 满足的递推式为

$$
\boxed{a_n - a_{n-1} + a_{n-2} = 0\quad (n\ge 2)}
$$

并且 $\boxed {a_0 = 1, a_1 = 1}$

# Problem 3
设 $\{a_n\}$ 的母函数为 $G (x) = \sum_{n=0}^\infty a_n x^n$，则

$$
\begin{align}
G(x) &=\sum_{n=0}^\infty a_n x^n\\
&=\sum_{n=0}^\infty c \cdot (3x)^n + \sum_{n=0}^\infty b\cdot (-x)^n\\
&=\frac {c}{1-3x} + \frac {b}{1+x}\\
&=\frac {c(1+x) + b(1-3x)}{(1-3x)(1+x)}\\
&=\frac {(c-3b)x + (c + b)}{-3x^2 -2x + 1}\\
\end{align}
$$

因此

$$
\begin{align}
(-3x^2 - 2x + 1)G(x) &= (c-3b)x + (c + b)\\
-3x^2G(x) - 2xG(x) + G(x) &= (c-3b)x + (c +b )\\
-3\sum_{n=2}^\infty a_{n-2}x^n - 2\sum_{n=1}^\infty a_{n-1}x^n + \sum_{n=0}^\infty a_n x^n &= (c-3b)x + (c +b )\\
a_0 + a_1x-2a_0x + \sum_{n=2}^\infty (-3a_{n-2} -2a_{n-1} + a_n)x^n&= (c-3b)x + (c +b )\\
\end{align}
$$

故

$$
\begin{cases}
a_0 = c+b\\
a_1 - 2a_0 = c-3b\\
a_n -2a_{n-1} -3a_{n-2} = 0
\end{cases}
$$

因此递推式为

$$
\boxed{a_n - 2a_{n-1}  - 3a_{n-2} = 0\quad n\ge2}
$$

# Problem 4
令 $a_n = u_n + v_n$，其中 $v_n$ 是特解，$u_n$ 是通解
首先写出特征多项式

$$
C(x) = x^2 - 2x +1 
$$

可知特征方程有一个二重实根 $x_1 = 1$

非齐次项可以为 $5$，是关于 $n$ 的 $0$ 次多项式，因此特解可以写为

$$
v_n = A_0n^2
$$

将 $v_n$ 代入原递推关系式，得到

$$
\begin{align}
A_0n^2 - 2A_0(n-1)^2 + A_0(n-2)^2 &= 5\\
n^2-2(n-1)^2 + (n-2)^2&=\frac {5}{A_0}\\
n^2-2(n^2 -2n + 1) + n^2 - 4n + 4 &= \frac 5 {A_0}\\
2&=\frac 5 {A_0}\\
A_0 &= \frac {5}{2}
\end{align}
$$

因此

$$
v_n = \frac 5 2 n^2
$$

根据特征方程有一个二重实根 $x_1 = 1$，可以知道通解可以写为

$$
u_n = A_1 + A_2n
$$

因此

$$
a_n = v_n + u_n = \frac 5 2 n^2 + A_1 + A_2 n
$$

故代入初值得到

$$
\begin{cases}
A_1 = 1\\
A_2 =  - \frac 3 2
\end{cases}
$$

因此

$$
\boxed{a_n = \frac 5 2 n^2 - \frac {3}{2}n + 1}
$$

# Problem 5
假设子串 $AB$ 一次都不出现的字符串数量为 $a_n$
假设字符串以 $A$ 结尾，且子串 $AB$ 一次都不出现的字符串的数量为 $b_n$
假设字符串以 $B, C, D$ 结尾，且子串 $AB$ 一次都不出现的字符串的数量为 $c_n$

显然

$$
a_n = 3b_{n-1} + 4c_{n-1}
$$

容易知道

$$
\begin{align}
b_n &= b_{n-1} + c_{n-1}\\
c_n &=  2b_{n-1} + 3c_{n-1}
\end{align}
$$

将 $b_n - b_{n-1} = c_{n-1}$ 代入 $c_n = 2b_{n-1} + 3c_{n-1}$ 得到

$$
\begin{align}
b_{n+1} - b_n &= 2b_{n-1} + 3b_n - 3b_{n-1}\\
b_{n+1} - 4b_n + b_{n-1} &= 0
\end{align}
$$

即

$$
b_n - 4b_{n-1}+b_{n-2} = 0
$$

故特征多项式为

$$
C(x) = x^2 - 4x + 1
$$

可得特征方程存在两个实根

$$
\alpha_0 = 2 + \sqrt 3 , \alpha_1 = 2-\sqrt 3
$$

因此 $b_n$ 的形式为

$$
b_n = A_0\alpha_0^n + A_1\alpha_1^n
$$

容易知道 $b_n$ 的初值为 $b_0 = 0, b_1 = 1$，因此

$$
\begin{cases}
A_0 + A_1 = 0\\
A_0\alpha_0 + A_1\alpha_1 = 1
\end{cases}
$$

解得 $A_0 = \frac 1 {2\sqrt 3}, A_1 = -\frac {1}{2\sqrt 3}$

故

$$
b_n = \frac {1}{2\sqrt 3}\left(\left(2+\sqrt 3\right)^n - \left(2-\sqrt 3\right)^n\right)
$$

因为 $c_{n-1} = b_{n} - b_{n-1}$，故

$$
\begin{align}
a_n &= 3b_{n-1} + 4c_{n-1}\\
&=3b_{n-1} + 4b_{n} -4b_{n-1}\\
&=4b_{n}-b_{n-1}\\
&=\frac {2}{\sqrt 3}\left(\left(2+\sqrt 3\right)^{n+1} - \left(2-\sqrt 3\right)^{n+1}\right)\\
&\quad -\frac {1}{2\sqrt 3}\left(\left(2+\sqrt 3\right)^{n} - \left(2-\sqrt 3\right)^{n}\right)
\end{align}
$$

假设子串 $AB$ 至少出现一次的数目为 $x_n$，显然

$$
\begin{align}
x_n &= 4^n- a_n\\
&=\boxed{4^n - \frac {2}{\sqrt 3}\left(\left(2+\sqrt 3\right)^{n+1} - \left(2-\sqrt 3\right)^{n+1}\right)+\frac {1}{2\sqrt 3}\left(\left(2+\sqrt 3\right)^{n} - \left(2-\sqrt 3\right)^{n}\right)}
\end{align}
$$

# Problem 6
假设 $n$ 为奇数
先用标准汉诺塔方法将前 $n-1$ 从 $A$ 移动到 $B$，此时 $n-1$ 的位置固定
将 $n$ 移动到 $C$，此时 $n$ 的位置固定
用标准汉诺塔方法将前 $n-2$ 从 $B$ 移动到 $C$，此时 $n-2$ 的位置固定
用标准汉诺塔方法将前 $n-3$ 从 $C$ 移动到 $B$，此时 $n-3$ 的位置固定
以此类推

假设 $n$ 为偶数
先用标准汉诺塔方法将前 $n-1$ 从 $A$ 移动到 $C$，此时 $n-1$ 的位置固定
将 $n$ 移动到 $B$，此时 $n$ 的位置固定
用标准汉诺塔方法将前 $n-2$ 从 $C$ 移动到 $B$，此时 $n-2$ 的位置固定
用标准汉诺塔方法将前 $n-3$ 从 $B$ 移动到 $C$，此时 $n-3$ 的位置固定
以此类推

标准汉诺塔方法的通项公式为 $h_n = 2^n - 1$
因此假设移动次数为 $a_n$，有

$$
\begin{align}
a_n &= h_{n-1} + h_{n-2} + \cdots + h_1 + 1\\
&=\sum_{i=1}^{n-1}2^i - (n-1) + 1 \\
&=\sum_{i=1}^{n-1}2^i - n + 2 \\
&=2^n - 2 -n + 2\\
&=\boxed{2^n - n}
\end{align}
$$

# Problem 7
假设方案数为 $a_n$
假设不允许相同字母连续出现三次，但是结尾的最后两个字符是相同的方案数为 $b_n$
假设不允许相同字母连续出现三次，但是结尾的最后两个字符是不同的方案数为 $c_n$

显然

$$
a_n = (k-1)b_{n-1} + kc_{n-1}
$$

容易知道

$$
\begin{align}
b_{n} &= c_{n-1}\\
c_n &=(k-1)b_{n-1} + (k-1)c_{n-1}
\end{align}
$$

因此

$$
c_n = (k-1)(c_{n-1} + c_{n-2})
$$

特征多项式为

$$
C(x) = x^2 - (k-1)x - (k-1)
$$

特征方程的有两个根

$$
\alpha_0 = \frac {k-1 +\sqrt {k^2 + 2k-3}}{2},\alpha_1 = \frac {k-1-\sqrt {k^2 + 2k-3}}{2}
$$

因此 $c_n$ 可以写为

$$
c_n = A_0\alpha_0^n + A_1\alpha_1^n
$$

容易知道序列初值 $c_0 = 0, c_1 = 1$，因此

$$
\begin{cases}
A_0 + A_1 = 0\\
A_0\alpha_0 + A_1\alpha_1 = 1
\end{cases}
$$

解得

$$
A_0 = \frac {1}{\sqrt {k^2 + 2k-3}}, A_1 = -\frac 1 {\sqrt {k^2 +2k -3}}
$$

因此

$$
\begin{align}
c_n &= \frac 1{2^n\sqrt{k^2 + 2k-3}}\left(k-1 + \sqrt{k^2+2k-3}\right)^n\\
&\quad -\frac 1{2^n\sqrt{k^2 + 2k-3}}\left(k-1 - \sqrt{k^2+2k-3}\right)^n\\
&=\frac 1{2^n\sqrt{k^2 + 2k-3}}\left(\left(k-1 + \sqrt{k^2+2k-3}\right)^n-\left(k-1 - \sqrt{k^2+2k-3}\right)^n\right)
\end{align}
$$

因此

$$
\begin{align}
a_n &= (k-1)b_{n-1} + kc_{n-1}\\
&=(k-1)b_{n-1} + (k-1)c_{n-1} + c_{n-1}\\
&=c_n + c_{n-1}\\
&=\frac {c_{n+1}}{k-1}\\
&=\boxed{\frac 1{2^{n+1}(k-1)^{2/3}(k-2)^{1/2}}\left(\left(k-1 + \sqrt{k^2+2k-3}\right)^{n+1}-\left(k-1 - \sqrt{k^2+2k-3}\right)^{n+1}\right)}
\end{align}
$$

# Problem 8
令 $S_n = \sum_{k=1}^n k^4$，则 $S_{n-1} = \sum_{k=1}^{n-1}k^4$，容易知道

$$
S_n - S_{n-1} = n^4
$$

其特征多项式为

$$
C(x) = x- 1
$$

对它进行五次差分操作，可以得到一个齐次的递推关系，每进行一次差分操作，特征多项式就会乘上 $(x-1)$，因此最后得到的特征多项式为

$$
D(x) = (x-1)^6
$$

显然它有一个六重根 $\alpha = 1$

由此写出 $S_n$ 的通项表达式为

$$
S_n = A_0 + A_1n + A_2 n^2 + A_3 n^3 + A_4 n^4 + A_5 n^5
$$

代入初值，得到方程组

$$
\begin{cases}
A_0 = 0\\
A_0 + A_1 + A_2 + A_3 + A_4 + A_5 = 1\\
A_0 + 2A_1 + 2^2A_2 + 2^3A_3 + 2^4A_4 + 2^5A_5 = \sum_{k=1}^2 k^4\\
A_0 + 3A_1 + 3^2A_2 + 3^3A_3 + 3^4A_4 + 3^5A_5 = \sum_{k=1}^3 k^4\\
A_0 + 4A_1 + 4^2A_2 + 4^3A_3 + 4^4A_4 + 4^5A_5 =\sum_{k=1}^4 k^4 \\
A_0 + 5A_1 + 5^2A_2 + 5^3A_3 + 5^4A_4 + 5^5A_5 =\sum_{k=1}^5 k^4 \\
\end{cases}
$$

求解可得

$$
A_0 = 0, A_1 = -\frac{1}{30}, A_2 = 0, A_3 = \frac 1 3, A_4 = \frac{1}{2}, A_5 = \frac{1}{5}
$$

因此

$$
\boxed{S_n = -\frac 1 {30}n +\frac 1 3n^3 +  \frac 1 2n^4 +\frac 1 {5}n^5}
$$

# Problem 9
(1) 如果选择第一个元素，则剩下 $k-1$ 个元素的选择方法为 $f (n-2, k-1)$，如果不选择第一个元素，而选择方法为 $f (n-1, k)$，因此

$$
\boxed{f(n, k) = f(n-1, k) + f(n-2, k-1)}
$$

(2) 当 $n=1$ ，$f (1, 0)  = 1, f (1, 1) = 1, f (1, k) = 0\ (k > 1)$；
当 $n=2$ ，$f (2,0) = 1, f (2, 1) = 2, f (2, 2) = 0, f (2, k) = 0\ (k > 2)$；
当 $n=3$ ，$f (3, 0) = 1, f (3, 1) = 3, f (3, 2) = 1, f (3, 3) = 0 , f (3, k) =0\ (k > 3)$；
当 $n=4$ ，$f (4, 0) = 4, f (4, 1) = 4, f (4, 2) = 3, f (4, 3) = 2 , f(4,4) = 0,f (4, k) =0\ (k > 4)$；

因此猜测 $f (n, k) = \binom{n - k + 1}{k}$，故归纳假设为 $f (n, k) =  \binom{n-k + 1}{k}$

考虑 $f (n + 1, k)$，根据递推关系，可知

$$
\begin{align}
f(n+1, k) &= f(n, k) + f(n-1, k-1)\\
&=\binom{n-k+1}{k} + \binom{n-k + 1}{k-1}\\
&=\binom{n-k+1 + 1}{k}\\
&=\binom{(n+1)-k+1}{k}
\end{align}
$$

因此递推成立，故

$$
\boxed{f(n,k) = \binom{n-k+1}{k}}
$$

(3) 如果选择第一个元素，则剩下的 $k-1$ 个元素的选择方法为 $f (n-3, k-1)$，如果不选择第一个元素，则选择方法为 $f (n-1, k)$，因此

$$
\begin{align}
g(n,k) &= f(n-1,k) + f(n-3, k-1)\\
&=\boxed{\binom{n-k}{k} + \binom{n-k-1}{k-1}}
\end{align}
$$

# Problem 10
(1) 最后一块砖有三种选择：1. 一个1x1的正方形砖 2. 两个直角边为1的直角三角形砖 3. 一个斜边为2，两个直角边为1的直角三角形砖 (两种方向)，因此

$$
a_n = 2a_{n-1} + 2a_{n-2}
$$

特征多项式为

$$
C(x) = x^2 - 2x - 2
$$

有两个根

$$
\alpha_0 = 1 + \sqrt 3, \alpha_1 = 1-\sqrt 3
$$

因此

$$
a_n = A_0 \alpha_0^n + A_1 \alpha_1^n
$$

容易知道初始值 $a_1 = 2, a_2 = 6$，计算得 $a_0 = 1$，因此

$$
\begin{cases}
A_0 + A_1 = 1\\
A_0\alpha_0 + A_1\alpha_1 = 2
\end{cases}
$$

解得 

$$A_0 = \frac {\sqrt 3 +1} {2\sqrt 3}, A_1 =   \frac {\sqrt 3 -1} {2\sqrt 3}$$

因此

$$
\begin{align}
a_n &=  \frac {\sqrt 3 + 1}{2\sqrt 3}\left(1 + \sqrt 3\right)^n + \frac {\sqrt 3 - 1}{2\sqrt 3}\left(1-\sqrt 3\right)^n\\
&=  \frac {\ 1}{2\sqrt 3}\left(1 + \sqrt 3\right)^{n+1} - \frac {\ 1}{2\sqrt 3}\left(1-\sqrt 3\right)^{n+1}\\
&=\boxed{  \frac {\ 1}{2\sqrt 3}\left(\left(1 + \sqrt 3\right)^{n+1} - \left(1-\sqrt 3\right)^{n+1}\right)}\\
\end{align}
$$

(2) 设总砖数为 $b_n$
如果最后一块砖是 1x1 的方形砖，有 $a_{n-1}$ 种铺砖方案，这些方案的总砖数为 $b_{n-1}$，最后一块砖贡献的总砖数为 $a_{n-1}$
如果最后一块是两个小直角三角形，有 $a_{n-1}$ 种铺砖方案，总砖数为 $b_{n-1}$，最后一块砖贡献的数量为 $2a_{n-1}$
如果最后一块是一个大的直角三角形 + 两个小的直角三角形(第一种方向)，有 $a_{n-2}$ 种方案，数量为 $b_{n-2}$，最后一块砖贡献的数量为 $3a_{n-2}$
如果最后一块是一个大的直角三角形 + 两个小的直角三角形(第二种方向)，有 $a_{n-2}$ 种方案，数量为 $b_{n-2}$，最后一块砖贡献的数量为 $3a_{n-2}$

因此

$$
\begin{align}
b_n &= b_{n-1} + a_{n-1} + b_{n-1} + 2a_{n-1} + 2b_{n-2} + 6a_{n-2}\\
&=2b_{n-1} + 2b_{n-2} + 3a_{n-1} + 6a_{n-2}
\end{align}
$$

即

$$
\begin{align}
b_n - 2b_{n-1} + 2b_{n-2} &= 3a_{n-1} + 6a_{n-2}\\
\end{align}
$$

递推关系的特征多项式为

$$
C(x) = x^2 - 2x + 2x
$$

它的两个根 $\alpha_0, \alpha_1$ 在上一问已经求解

假设

$$
b_n= v_n + u_n
$$

其中 $v_n$ 为特解，$u_n$ 为通解

因为

$$
\begin{align}
3a_{n-1} + 6a_{n-2} &=  \frac { \sqrt 3}{2}\left(\left(1 + \sqrt 3\right)^{n} - \left(1-\sqrt 3\right)^{n}\right) + \sqrt 3 \left(\left(1 + \sqrt 3\right)^{n-1} - \left(1-\sqrt 3\right)^{n-1}\right)\\
&=\left(\frac {\sqrt 3} 2 + \frac {\sqrt 3}{1 + \sqrt 3}\right) \alpha_0^n -\left(\frac {\sqrt 3}{2} + \frac {\sqrt 3}{1-\sqrt 3}\right)\alpha_1^n
\end{align}
$$

因此 $v_n$ 可以写为

$$
v_n = A_0n\alpha_0^n + A_1n\alpha_1^n
$$

同时容易知道通解的形式为

$$
u_n = B_0\alpha_0^n + B_1\alpha_1^n
$$

因此

$$
b_n = (A_0n + B_0)\alpha_0^n + (A_1n + B_1)\alpha_1^n
$$

容易知道数列初值为 $b_1 = 3, b_2 = 18, b_3 = 54$，而 $b_0 = 3a_1 + 6a_0 - b_2 + 2b_1 = 0$

考虑将 $b_n$ 代回递推式中，得到

$$
\begin{align}
b_n - 2b_{n-1} +2b_{n-2}&=3a_{n-1} + 6a_{n-2}\\
2A_0\alpha_0^{n-1}-4A_0\alpha_0^{n-2} +2A_1\alpha_1^{n-1}-4A_1\alpha_1^{n-2} &=3a_{n-1} + 6a_{n-2}
\end{align}
$$

即

$$
\begin{align}
(\frac {2A_0}{\alpha_0}-\frac {4A_0}{\alpha_0^2})\alpha_0^n +(\frac {2A_1}{\alpha_1}-\frac {4A_1}{\alpha_1^2})\alpha_1^n  = \left(\frac {\sqrt 3} 2 + \frac {\sqrt 3}{1 + \sqrt 3}\right) \alpha_0^n -\left(\frac {\sqrt 3}{2} + \frac {\sqrt 3}{1-\sqrt 3}\right)\alpha_1^n\\
2A_0(\frac {1}{\alpha_0}-\frac {2}{\alpha_0^2})\alpha_0^n +2A_1(\frac {1}{\alpha_1}-\frac {2}{\alpha_1^2})\alpha_1^n  = \left(\frac {\sqrt 3} 2 + \frac {\sqrt 3}{1 + \sqrt 3}\right) \alpha_0^n -\left(\frac {\sqrt 3}{2} + \frac {\sqrt 3}{1-\sqrt 3}\right)\alpha_1^n\\
\end{align}
$$

因此

$$
\begin{cases}
2A_0(\frac 1 {\alpha_0} - \frac 2 {\alpha_0^2}) = \frac {\sqrt 3}2 + \frac {\sqrt 3}{1 + \sqrt 3}\\
2A_1(\frac 1 {\alpha_1} - \frac 2 {\alpha_1^2}) = \frac {\sqrt 3}2 + \frac {\sqrt 3}{1 - \sqrt 3}
\end{cases}
$$

解得

$$
A_0 = \frac {3(\sqrt 3  + 1)}{2}, A_1 = \frac {3(\sqrt 3 - 1)}{2}
$$

进而将 $b_0 = 0, b_1 = 3$ 代入 $b_n$ 的表达式，得到

$$
\begin{cases}
B_0 + B_1 = 0\\
(A_0 + B_0)\alpha_0 + (A_1 + B_1)\alpha_1 = 3
\end{cases}
$$

解得

$$
B_0 = -\frac {1}{8\sqrt 3}, B_1 = \frac 1 {8\sqrt 3}
$$

因此最后得到

$$
\begin{align}
b_n &= \left(A_0n + B_0\right)\alpha_0^n + \left(A_1n + B_1\right)\alpha_1^n\\
&= \left(\frac {3(\sqrt 3 + 1)n}{2} -\frac {1}{8\sqrt 3} \right)\alpha_0^n + \left(\frac {3(\sqrt 3 - 1)n}{2} + \frac {1}{8\sqrt 3}\right)\alpha_1^n\\
&= \boxed{\left(\frac {3(\sqrt 3 + 1)n}{2} -\frac {1}{8\sqrt 3} \right)\left(1 + \sqrt 3\right)^n + \left(\frac {3(\sqrt 3 - 1)n}{2} + \frac {1}{8\sqrt 3}\right)\left(1 - \sqrt 3\right)^n}\\
\end{align}
$$











