(1)  $a_n = n^3 (n \ge 0)$，$\{a_n\}$ 的母函数为 $A(x) = 1^3 x + 2^3 x^2 + \cdots + n^3 x^n = \sum_{n=0}^{\infty} n^3 x^n$

考虑两个数列 $\{b_n\}$ 和 $\{c_n\}$，如果 $b_n = nc_n$，我们有：

$$
\begin{align}
B(x) &= \sum_{n=0}^\infty b_nx^n\\
&=\sum_{n=0}^{\infty} nc_nx^n\\
&=x\cdot \sum_{n=0}^\infty c_n(x^n)'\\
&=x\cdot C'(x)
\end{align}
$$

考虑数列 $d_n \equiv 1$，$d_n$ 的母函数为 $D (x) = \sum_{n=0}^\infty x^n = \frac {1}{1-x}$

令数列 $c_n = n$，则 $c_n = nd_n$，故 

$$
\begin{align}
C (x) &= x\cdot D' (x)\\
&=x\cdot \left(\frac {1}{1-x}\right)'\\
&=\frac {x}{(1-x)^2}
\end{align}$$

令数列 $b_n = n^2$，则 $b_n = nc_n$，故

$$ \begin{align}
B (x) &= x\cdot C' (x)\\
&=x\cdot \left( \frac {x}{(1-x)^2}\right)'\\
&=x\cdot \frac {(1-x)^2 + 2x(1-x)}{(1-x)^4}\\
&=x\cdot \frac {x^2 - 2x + 1 - 2x^2 + 2x}{(1-x)^4}\\
&=x\cdot \frac {-x^2 + 1 }{(1-x)^4}\\
&=\frac {x(x+1) }{(1-x)^3}\\
\end{align}$$

$a_n = n^3$，则 $a_n = nb_n$，故

$$
\begin{align}
A (x) &= x\cdot B' (x)\\
&=x\cdot \left(\frac {x(x+1)}{(1-x)^3}\right)'\\
&=x\cdot \frac {(2x + 1)(1-x)^3  + 3x(x + 1)(1-x)^2}{(1-x)^6}\\
&=x\cdot \frac {(2x + 1)(1-x)  + 3x(x + 1)}{(1-x)^4}\\
&=x\cdot \frac {2x - 2x^2 + 1 - x  + 3x^2 + 3x}{(1-x)^4}\\
&=x\cdot \frac {x^2 + 4x +1}{(1-x)^4}\\
&=\frac {x(x^2 + 4x +1)}{(1-x)^4}\\
\end{align}$$

(2) $a_n = \binom{n+3}{3}(n\ge 0)$，$\{a_n\}$ 的母函数为 $A (x) = \sum_{n = 0}^\infty \binom{n+3}{3}x^n = \sum_{n=0}^\infty \frac {(n+3)(n+2)(n+1)}{6}x^n$

考虑

$$
\begin{align}
\frac 1 {1-x} &= \sum_{n=0}^\infty x^n\\
&=\sum_{n=0}^\infty \binom{n}{0} x^n
\end{align}$$

两边求导得到 

$$\begin{align}
\frac {1}{(1-x)^2} &= n\cdot \sum_{n=0}^\infty x^{n-1}\\
&=0 + 1x^0 + 2x^1 + \cdots\\
&=\sum_{n=0}^{\infty}(n+1)x^n\\
&=\sum_{n=0}^\infty \binom{n+1}{1}x^n
\end{align}$$

两边继续求导得到

$$
\begin{align}
\frac {1}{(1-x)^3} &= \frac {n(n-1)}{2} \sum_{n=0}^\infty x^{n-2}\\
&=\sum_{n=0}^\infty \frac {(n + 2)(n+1)}{2} x^n\\
&=\sum_{n=0}^\infty \binom{n+2}{2} x^n
\end{align}
$$

两边继续求导得到

$$
\begin{align}
\frac {1}{(1-x)^4} &= \frac {n(n-1)(n-2)}{6} \sum_{n=0}^\infty x^{n-3}\\
&=\sum_{n=0}^\infty \frac {(n + 3)(n+2)(n+1)}{6} x^n\\
&=\sum_{n=0}^\infty \binom{n+3}{3} x^n
\end{align}
$$

依次类推，可以得到

$$
\frac {1}{(1-x)^{k+1}} = \sum_{n=0}^\infty \binom{n + k}{k}x^n
$$

因此 

$$A (x) = \frac {1}{(1-x)^4}$$

(3) $a_n = \sum_{k=1}^{n+1} k ^3, A (x) = \sum_{n=0}^\infty (\sum_{k=1}^{n+1} k^3) x^n$

令 $b_n = n^3, B(x) = \frac {x (x^2 + 4x +1)}{(1-x)^4}$

则可以知道 $a_n = \sum_{k=1}^{n+1} b_k$

$$
\begin{align}
A(x) &= a_0 + a_1x + a_2 x^2 + \cdots \\
&= b_1 + (b_1 + b_2)x  + (b_1 + b_2 + b_3)x^2 + \cdots\\
&=b_1(1 + x + x^2 + \cdots) + b_2(x + x^2 + \cdots) + \cdots\\
&=b_1(\sum_{n=0}^\infty x^n) + b_2(\sum_{n=1}^\infty x^n) + b_3(\sum_{n=2}^\infty x^n) + \cdots\\
&=\frac {1}{1-x}\cdot b_1 + \frac {x}{1-x}\cdot b_2 + \frac {x^2}{1-x} b_3 + \cdots\\
&=\frac 1 {1-x} (b_1 + b_2 x + b_3 x^2 + \cdot)\\
&=\frac 1 {x(1-x)}(b_0  + b_1 x + b_2 x^2 + \cdot)\\
&=\frac 1 {x(1-x)}B(x)\\
&= \frac {B(x)}{x(1-x)}\\
&=\frac {(x^2 + 4x +1)}{(1-x)^5}
\end{align}
$$

(4) 

$$
\begin{align}
B(x) & = b_0 + b_1 x + b_2 x^2 + \cdots \\
&=a_0 + (a_1 - a_0)x + (a_2 - a_1)x^2 + \cdots\\
&=a_0(1-x) + a_1(x - x^2) + a_2(x^2 - x^3) + \cdots\\
&=(1-x)(a_0 + a_1x + a_2 x^2 + \cdots)\\
&=(1-x)A(x)\\
&=\frac {4-3x}{(1 + x - x^2)}
\end{align}
$$

