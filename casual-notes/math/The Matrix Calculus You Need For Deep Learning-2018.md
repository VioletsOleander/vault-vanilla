# 1 Introduction
# 2 Review: Scalar derivative rules
![[Matrix Calculus For DL-Fig1.png]]
将$\frac d {dx}$视为一个算子，它将一个一元函数映射为另一个一元函数，例如
$\frac d {dx} f(x)$将函数$f(x)$映射为它相对于变量$x$的导函数$f'(x)=\frac {df(x)}{dx}$

该算子是可分配的，并且可以提取常数，例如
$$\frac d {dx}9(x+x^2)=9\frac d{dx}(x+x^2) = 9(\frac d{dx}x + \frac d{dx}x^2)$$
# 3 Introduction to vector calculus and partial derivatives
对于多元函数，例如$f(x,y)$，我们分别计算该函数相对于各个变量的导数，
因此对于一个二元函数，我们会得到该函数分别对应其两个变量的两个偏导数(partial derivatives)

一元函数的导数算子是$\frac d {dx}$，多元函数的偏导数算子是$\frac \partial {\partial x}$
因此$\frac {\partial} {\partial x}f(x,y)$就是$f(x,y)$相对于变量$x$的偏导数函数，$\frac \partial {\partial y}f(x,y)$就是$f(x,y)$相对于变量$y$的偏导数函数

计算多元函数对于某个变量的偏导数时，将其他变量视为常数，问题就转换为计算一元函数对于其变量的导数

考虑$f(x,y) = 3x^2y$，则
$\frac \partial {\partial x} 3x^2y = 3y\frac \partial {\partial x}x^2 = 3y2x=6xy$
$\frac \partial {\partial x} 3x^2y = 3x^2\frac \partial {\partial x}y = 3x^2$

对于一个多元函数，它的梯度(gradient)定义为由它分别相对于其各个变量的偏导数所构成的向量
例如，$f(x,y)$的梯度就写为
$$\nabla f(x,y) = [\frac \partial {\partial x}f(x,y),\frac \partial {\partial y}f(x,y)]$$

目前为止，我们讨论的都是标量值函数
# 4 Matrix Calculus
对于两个接收相同自变量的多元函数$f(x,y) = 3x^2y$和$g(x,y) = 2x+y^2$，我们分别计算它们的梯度
$\nabla f(x,y) = [\frac \partial {\partial x}f(x,y),\frac \partial {\partial y}f(x,y)]=[6xy,3x^2]$
$\nabla g(x,y) = [\frac \partial {\partial x}g(x,y),\frac \partial {\partial y}g(x,y)]=[2,8y^7]$
得到两个梯度向量


我们将这两个梯度向量按行堆叠起来，成为一个矩阵，就得到了雅可比矩阵(Jacobian matrix)
$$J = \begin{bmatrix}\nabla f(x,y)\\ \nabla g(x,y)\end{bmatrix}=\begin{bmatrix}\frac \partial {\partial x}f(x,y)&\frac \partial {\partial y}f(x,y)\\
\frac \partial {\partial x}g(x,y)&\frac \partial {\partial y}g(x,y)\\\end{bmatrix} =
\begin{bmatrix}6xy&3x^2\\2&8y^7\end{bmatrix}$$
按行组织梯度向量的雅可比矩阵是分子布局(numerator layout)
按列组织梯度向量的雅可比矩阵是分母布局(denominator layout)
二者互为对方的转置

## 4.1 Generalization of the Jacobian
对于一个多元标量值函数，我们将它的多个自变量表示为用一个向量表示$f(x,y,z) \Rightarrow f(\mathbf x)$，而$\mathbf x_i$就表示向量$\mathbf x$的第$i$个元素
默认情况下，向量为列向量，形状是$n\times 1$，例如$\mathbf x = \begin{bmatrix}x_1\\x_2\\ \vdots\\ x_n\end{bmatrix}$

对$f(\mathbf x)$应用梯度算子，得到$\nabla f(\mathbf x) = [\frac \partial {\partial x_1}f(\mathbf x),\frac \partial {\partial x_2}f(\mathbf x),\cdots,\frac \partial {\partial x_n}f(\mathbf x) ]$
对于多元标量值函数，定义梯度算子的另一种形式$\frac \partial {\partial \mathbf x} = \nabla$，即$\frac \partial {\partial \mathbf x}f(\mathbf x) = \nabla f(\mathbf x)$


对应多个有共同自变量的多元标量值函数，例如$$\begin{align}&y_1 = f_1(\mathbf x)\\&y_2 = f_2(\mathbf x)\\ &\quad\quad\vdots \\&y_m = f_m(\mathbf x)\end{align}$$我们可以把它们组合为一个向量，记为$\mathbf y = \mathbf f(\mathbf x)$
$\mathbf y = \mathbf f(\mathbf x)$表示了一个由$m$个多元标量值函数组成的向量，可以认为$\mathbf y = \mathbf f(\mathbf x)$是一个多元向量值函数，其中$f_i$表示$\mathbf f$的第$i$个元素，是一个多元标量值函数，$f_i$的参数为$\mathbf x$，$\mathbf x$的长度是$n$

当$m=n$时，定义一个多元向量值恒等函数$\mathbf y = \mathbf f(\mathbf x) = \mathbf x$，
显然，它的形式为
$$\begin{align}&y_1 = f_1(\mathbf x)=x_1\\&y_2 = f_2(\mathbf x)=x_2\\ &\quad\quad\vdots \\&y_n = f_n(\mathbf x)=x_n\end{align}$$
即$f_i(\mathbf x) = x_i$

对于一个多元向量值函数，它的雅可比矩阵就是其中每个多元标量值函数相较于自变量$\mathbf x$的梯度向量堆叠而成的矩阵(一共$m$个梯度向量，梯度向量的长度是$n$)
$$\frac {\partial \mathbf y}{\partial \mathbf x} = \begin{bmatrix}
\nabla f_1(\mathbf x)\\ \nabla f_2(\mathbf x) \\ \vdots \\\nabla f_m(\mathbf x) 
\end{bmatrix}
=
\begin{bmatrix}
\frac \partial {\partial \mathbf x}f_1(\mathbf x)\\
\frac \partial {\partial \mathbf x}f_2(\mathbf x)\\
\vdots\\
\frac \partial {\partial \mathbf x}f_m(\mathbf x)\\
\end{bmatrix}
=\begin{bmatrix}
\frac \partial {\partial x_1} f_1(\mathbf x) &\frac \partial {\partial x_2} f_1(\mathbf x) &\cdots & \frac \partial {\partial x_n} f_1(\mathbf x)\\
\frac \partial {\partial x_1} f_2(\mathbf x) &\frac \partial {\partial x_2} f_2(\mathbf x) &\cdots & \frac \partial {\partial x_n} f_2(\mathbf x)\\
\vdots & \vdots&\cdots&\vdots\\
\frac \partial {\partial x_1} f_m(\mathbf x) &\frac \partial {\partial x_2} f_m(\mathbf x) &\cdots & \frac \partial {\partial x_n} f_m(\mathbf x)\\
\end{bmatrix}$$
每个$\frac \partial {\partial \mathbf x} f_i(\mathbf x)$都是一个长度为$n$的向量，因为$\frac \partial {\partial \mathbf x}$这个算子表示了对长度为$n$的向量$\mathbf x$计算偏导数，等效于梯度算子
雅可比矩阵有$n$列，因为有$n$个自变量
雅可比矩阵有$m$行，因为有$m$个等式/多元标量值函数

对于多元向量值函数，计算一个多元标量值函数的雅可比矩阵的算子也记为$\frac \partial {\partial \mathbf x}$，即$\frac \partial {\partial \mathbf x} \mathbf y = \frac {\partial \mathbf y} {\partial \mathbf x}$

雅可比矩阵的每一个元素都表示了一个偏导数

考虑多元向量值恒等函数$\mathbf f(\mathbf x) = \mathbf x$的雅可比行列式
$$\begin{align}
\frac {\partial \mathbf y}{\partial \mathbf x} = \begin{bmatrix}
\frac \partial {\partial \mathbf x} f_1(\mathbf x)\\
\frac \partial {\partial \mathbf x} f_2(\mathbf x)\\
\vdots\\
\frac \partial {\partial \mathbf x} f_n(\mathbf x)\\
\end{bmatrix} 
&= 
\begin{bmatrix}
\frac \partial {\partial x_1} f_1(\mathbf x) &\frac \partial {\partial x_2} f_1(\mathbf x) &\cdots & \frac \partial {\partial x_n} f_1(\mathbf x)\\
\frac \partial {\partial x_1} f_2(\mathbf x) &\frac \partial {\partial x_2} f_2(\mathbf x) &\cdots & \frac \partial {\partial x_n} f_2(\mathbf x)\\
\vdots & \vdots&\cdots&\vdots\\
\frac \partial {\partial x_1} f_m(\mathbf x) &\frac \partial {\partial x_2} f_m(\mathbf x) &\cdots & \frac \partial {\partial x_n} f_m(\mathbf x)\\
\end{bmatrix}\\
&=\begin{bmatrix}
\frac \partial {\partial x_1} x_1 &\frac \partial {\partial x_2} x_1 &\cdots & \frac \partial {\partial x_n} x_1\\
\frac \partial {\partial x_1} x_2 &\frac \partial {\partial x_2} x_2 &\cdots & \frac \partial {\partial x_n} x_2\\
\vdots & \vdots&\cdots&\vdots\\
\frac \partial {\partial x_1} x_n &\frac \partial {\partial x_2} x_n &\cdots & \frac \partial {\partial x_n} x_n\\
\end{bmatrix}\\
&=\begin{bmatrix}
1&0 & \cdots &0\\
0&1&\cdots &0\\
\vdots&\vdots & \ddots& \vdots\\
0 & 0 & \cdots & 1
\end{bmatrix}\\
&=I
\end{align}$$

## 4.2 Derivatives of vector element-wise binary operators
本节考虑按元素的二元向量运算(element-wise binary operations on vectors)，例如向量加减法$\mathbf w \pm \mathbf v$，向量比较$\mathbf w > \mathbf v, \mathbf w < \mathbf v$等

将按元素的二元向量运算表示为$\mathbf y = \mathbf f(\mathbf w) \bigcirc \mathbf g(\mathbf x)$
其中$|y | = n, | w| = |x| = n$，即按元素运算的运算元和结果向量的长度一定是相等的
其中$\bigcirc$表示任意的按元素运算符，例如加法运算符$+$

将$\mathbf y = \mathbf f(\mathbf w) \bigcirc \mathbf g(\mathbf x)$展开，就是
$$\mathbf y = \begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}
=
\begin{bmatrix}
f_1(\mathbf w)\\
f_2(\mathbf w)\\
\vdots\\
f_n(\mathbf w)
\end{bmatrix}
\bigcirc
\begin{bmatrix}
g_1(\mathbf x)\\
g_2(\mathbf x)\\
\vdots\\
g_n(\mathbf x)
\end{bmatrix}
=
\begin{bmatrix}
f_1(\mathbf w)\bigcirc g_1(\mathbf x)\\
f_2(\mathbf w)\bigcirc g_2(\mathbf x)\\
\vdots\\
f_n(\mathbf w)\bigcirc g_n(\mathbf x)
\end{bmatrix}$$

$\mathbf y = \mathbf f(\mathbf w) \bigcirc \mathbf g(\mathbf x)$由两个多元向量值函数运算得到

则$\mathbf y$相对于每一个向量自变量的偏导数就定义为$\mathbf y$相对于每一个向量自变量的雅可比矩阵
$\mathbf y$相对于$\mathbf w$的雅可比矩阵是
$$\begin{align}
J_{\mathbf w} = \frac {\partial \mathbf y}{\partial \mathbf w}=\begin{bmatrix}
\frac \partial {\partial\mathbf w} y_1\\
\vdots\\
\frac \partial {\partial \mathbf w}y_n\\
\end{bmatrix}
&=
\begin{bmatrix}
\frac \partial {\partial w_1} y_1 &\cdots & \frac \partial {\partial w_n} y_1\\
\vdots & \ddots&\vdots\\
\frac \partial {\partial w_1} y_n&\cdots & \frac \partial {\partial w_n} y_n\\
\end{bmatrix}\\
&=\begin{bmatrix}
\frac \partial {\partial w_1} (f_1(\mathbf w)\bigcirc g_1(\mathbf x)) &\cdots & \frac \partial {\partial w_n} (f_1(\mathbf w)\bigcirc g_1(\mathbf x))\\
\vdots & \ddots&\vdots\\
\frac \partial {\partial w_1} (f_n(\mathbf w)\bigcirc g_n(\mathbf x))&\cdots & \frac \partial {\partial w_n} (f_n(\mathbf w)\bigcirc g_n(\mathbf x))\\
\end{bmatrix}
\end{align}$$

$\mathbf y$相对于$\mathbf x$的雅克比矩阵是
$$\begin{align}
J_{\mathbf x} = \frac {\partial \mathbf y}{\partial \mathbf x}=\begin{bmatrix}
\frac \partial {\partial\mathbf x} y_1\\
\vdots\\
\frac \partial {\partial \mathbf x}y_n\\
\end{bmatrix}
&=
\begin{bmatrix}
\frac \partial {\partial x_1} y_1 &\cdots & \frac \partial {\partial x_n} y_1\\
\vdots & \ddots&\vdots\\
\frac \partial {\partial x_1} y_n&\cdots & \frac \partial {\partial x_n} y_n\\
\end{bmatrix}\\
&=\begin{bmatrix}
\frac \partial {\partial x_1} (f_1(\mathbf w)\bigcirc g_1(\mathbf x)) &\cdots & \frac \partial {\partial x_n} (f_1(\mathbf w)\bigcirc g_1(\mathbf x))\\
\vdots & \ddots&\vdots\\
\frac \partial {\partial x_1} (f_n(\mathbf w)\bigcirc g_n(\mathbf x))&\cdots & \frac \partial {\partial x_n} (f_n(\mathbf w)\bigcirc g_n(\mathbf x))\\
\end{bmatrix}
\end{align}$$

实际上，对于按元素运算，其雅可比矩阵往往是对角矩阵

考虑$J_{\mathbf w}$中的$i$行$j$列的一个元素$\frac \partial {\partial w_j} y_i=\frac \partial {\partial w_j}(f_i\bigcirc g_i)=\frac \partial {\partial w_j}(f_i(\mathbf w)\bigcirc g_i(\mathbf x))$，其中$i\ne j$，容易知道$\frac \partial {\partial w_j}(f_i(\mathbf w)\bigcirc g_i(\mathbf x))=(\frac \partial {\partial w_j} f_i(\mathbf w)) \bigcirc (\frac \partial {\partial w_j} g_i(\mathbf x))$
如果$f_i,g_i$相对于$w_j$是常数，或者说$f_i,g_i$不是$w_j$的函数，则显然有$\frac \partial {\partial w_j} f_i(\mathbf w) = \frac \partial {\partial w_j} g_i(\mathbf x) = 0$，则$(\frac \partial {\partial w_j} f_i(\mathbf w)) \bigcirc (\frac \partial {\partial w_j} g_i(\mathbf x))= 0 \bigcirc 0 = 0$

如果$\mathbf f,\mathbf g$也是按元素运算的函数，
则$\mathbf f(\mathbf w)$中的第$i$个元素，即$f_i(\mathbf w)$显然仅与$\mathbf w$中的第$i$个元素，即$w_i$有关
则$\mathbf g(\mathbf x)$中的第$i$个元素，即$g_i(\mathbf x)$显然仅与$\mathbf x$中的第$i$个元素，即$x_i$有关

那么显然在$i\ne j$的时候，$\frac {\partial } {\partial w_j} f_i(\mathbf w)= \frac {\partial } {\partial x_j}g_i(\mathbf x) = 0$，
则$(\frac \partial {\partial w_j} f_i(\mathbf w)) \bigcirc (\frac \partial {\partial w_j} g_i(\mathbf x))= 0 \bigcirc 0 = 0$
因为$f_i(\mathbf w)$仅与$w_i$有关，$g_i(\mathbf x)$仅与$x_i$有关，因此也可以将$f_i(\mathbf w),g_i(\mathbf x)$简写为$f_i(w_i),g_i(x_i)$

在这个条件下，满足
$$\begin{align}
J_{\mathbf w} = \frac {\partial \mathbf y}{\partial \mathbf w}=\begin{bmatrix}
\frac \partial {\partial\mathbf w} y_1\\
\vdots\\
\frac \partial {\partial \mathbf w}y_n\\
\end{bmatrix}
&=
\begin{bmatrix}
\frac \partial {\partial w_1} y_1 &\cdots & \frac \partial {\partial w_n} y_1\\
\vdots & \ddots&\vdots\\
\frac \partial {\partial w_1} y_n&\cdots & \frac \partial {\partial w_n} y_n\\
\end{bmatrix}\\
&=\begin{bmatrix}
\frac \partial {\partial w_1} (f_1(\mathbf w)\bigcirc g_1(\mathbf x)) &\cdots & \frac \partial {\partial w_n} (f_1(\mathbf w)\bigcirc g_1(\mathbf x))\\
\vdots & \ddots&\vdots\\
\frac \partial {\partial w_1} (f_n(\mathbf w)\bigcirc g_n(\mathbf x))&\cdots & \frac \partial {\partial w_n} (f_n(\mathbf w)\bigcirc g_n(\mathbf x))\\
\end{bmatrix}\\
&=\begin{bmatrix}
\frac \partial {\partial w_1} (f_1(w_1)\bigcirc g_1(x_1)) &\cdots & \frac \partial {\partial w_n} (f_1(w_1)\bigcirc g_1(x_1))\\
\vdots & \ddots&\vdots\\
\frac \partial {\partial w_1} (f_n(w_n)\bigcirc g_n(x_n))&\cdots & \frac \partial {\partial w_n} (f_n(w_n)\bigcirc g_n(x_n))\\
\end{bmatrix}\\
&=\begin{bmatrix}
\frac \partial {\partial w_1} (f_1(w_1)\bigcirc g_1(x_1)) &\cdots & 0\\
\vdots & \ddots&\vdots\\
0&\cdots & \frac \partial {\partial w_n} (f_n(w_n)\bigcirc g_n(x_n))\\
\end{bmatrix}
\end{align}$$

因此，可以直接将其写为
$$\frac {\partial \mathbf y}{\partial \mathbf w} = diag(\frac {\partial}{\partial w_1}(f_1(w_1)\bigcirc g_1(x_1)),\cdots, \frac {\partial}{\partial w_n}(f_n(w_n)\bigcirc g_n(x_n))$$
同理
$$\frac {\partial \mathbf y}{\partial \mathbf x} = diag(\frac {\partial}{\partial x_1}(f_1(w_1)\bigcirc g_1(x_1)),\cdots, \frac {\partial}{\partial x_n}(f_n(w_n)\bigcirc g_n(x_n))$$
该规则在$\mathbf f,\mathbf g$都是按元素运算的函数，且$\bigcirc$也是按元素进行的运算时成立

注意多元向量值函数的恒等函数$\mathbf f(\mathbf w) = \mathbf w$就是一个按元素运算的函数，
满足$f_i(\mathbf w) = w_i$

一些按元素二元向量运算的雅可比矩阵
![[Matrix Calculus for DL-Fig2.png]]

## 4.3 Derivatives involving scalar expansion
当对向量加减或乘除上一个常数时，我们其实可以将其视为隐式地将这个常数拓展成一个向量，然后对两个向量执行按元素的运算

例如，$\mathbf y = \mathbf x + z$实际上可以视为$\mathbf y = \mathbf f(\mathbf x) + \mathbf g(z)$，其中$\mathbf f(\mathbf x) = \mathbf x, \mathbf g(z) = \vec 1 z$
其中$z$是不决定于$\mathbf x$的任意常数

同样的，$\mathbf y = \mathbf x z$实际上可以视为$\mathbf y = \mathbf f(\mathbf x) \otimes \mathbf g(z)$，其中$\mathbf f(\mathbf x) = \mathbf x,\mathbf g(z) = \vec 1 z$
$\otimes$是按元素乘法

它们都满足$\frac {\partial \mathbf y} {\partial x} = diag(\cdots\frac {\partial }{\partial x_i}(f_i(x_i)\bigcirc g_i(z))\cdots)$

对于$\mathbf y = \mathbf x + z$，
$\frac {\partial }{\partial x_i}(f_i(x_i)+ g_i(z)) = \frac {\partial (x_i+ z)}{\partial x_i}=\frac {\partial x_i}{\partial x_i} + \frac {\partial z} {\partial x_i} = 1 + 0 = 1$
因此$\frac {\partial (\mathbf x + z)} {\partial\mathbf x} = diag(\vec 1) = I$
$\frac {\partial }{\partial z}(f_i(x_i)+ g_i(z)) = \frac {\partial (x_i+ z)}{\partial z}=\frac {\partial x_i}{\partial z} + \frac {\partial z} {\partial z} = 0 + 1 = 1$
同理$\frac {\partial (\mathbf x + z)} {\partial z} = \vec 1$

对于$\mathbf y = \mathbf xz$，
$\frac {\partial }{\partial x_i}(f_i(x_i)\otimes g_i(z)) = \frac {\partial (x_i\otimes z)}{\partial x_i}= \frac {\partial (x_i\times z)}{\partial x_i}=z\frac {\partial x_i}{\partial x_i} \times x_i\frac {\partial z} {\partial x_i} = z + 0 = z$
因此$\frac {\partial (\mathbf x z)} {\partial\mathbf x} = diag(\vec 1z) = Iz$
$\frac {\partial }{\partial z}(f_i(x_i)\otimes g_i(z)) = \frac {\partial (x_i\otimes z)}{\partial z}= \frac {\partial (x_i\times z)}{\partial z}=z\frac {\partial x_i}{\partial z} \times x_i\frac {\partial z} {\partial z} = 0+ x_i = x_i$
因此$\frac {\partial (\mathbf x z)} {\partial z} = \mathbf x$

## 4.4 Vector sum reduction
考虑$y = sum(\mathbf f(\mathbf x)) =sum(\begin{bmatrix}\vdots \\ f_i(\mathbf x)\\ \vdots\end{bmatrix}) =\sum_{i=1}^n f_i(\mathbf x)$

考虑$y$相对$\mathbf x$的梯度($1\times n$雅可比矩阵)
$$\begin{align}
\frac {\partial y}{\partial \mathbf x} &=[\frac {\partial y} {\partial x_1},\cdots,\frac {\partial y}{x_n}]\\
&=[\frac {\partial } {\partial x_1}\sum_if_i(\mathbf x),\cdots,\frac {\partial } {\partial x_n}\sum_i f_i(\mathbf x)]\\
&=[\sum_i\frac {\partial } {\partial x_1}f_i(\mathbf x),\cdots,\sum_i\frac {\partial } {\partial x_n} f_i(\mathbf x)]
\end{align}$$

若$\mathbf f(\mathbf x) = \mathbf x$，则$f_i(\mathbf x) = x_i$，则
$$\begin{align}
\frac {\partial y}{\partial \mathbf x} &=[\frac {\partial y} {\partial x_1},\cdots,\frac {\partial y}{x_n}]\\
&=[\sum_i\frac {\partial } {\partial x_1}f_i(\mathbf x),\cdots,\sum_i\frac {\partial } {\partial x_n} f_i(\mathbf x)]\\
&=[\sum_i\frac {\partial}{\partial x_1}x_i,\cdots,\sum_i\frac {\partial }{\partial x_n}x_i]\\
&=[1,\cdots,1]\\
&=\vec 1^T
\end{align}$$
若$\mathbf f(\mathbf x) = \mathbf xz$，则$f_i(\mathbf x) = zx_i$，则
$$\begin{align}
\frac {\partial y}{\partial \mathbf x} &=[\frac {\partial y} {\partial x_1},\cdots,\frac {\partial y}{x_n}]\\
&=[\sum_i\frac {\partial } {\partial x_1}f_i(\mathbf x),\cdots,\sum_i\frac {\partial } {\partial x_n} f_i(\mathbf x)]\\
&=[\sum_i\frac {\partial}{\partial x_1}zx_i,\cdots,\sum_i\frac {\partial }{\partial x_n}zx_i]\\
&=[z,\cdots,z]\\
&=\vec 1^Tz
\end{align}$$
而
$$\begin{align}
\frac {\partial y}{\partial z} 
&=\frac {\partial }{\partial z}\sum_i f_i(\mathbf x)\\
&=\sum_i\frac {\partial }{\partial z} f_i(\mathbf x)\\
&=\sum_i\frac {\partial }{\partial z} zx_i\\
&=\sum_i x_i\\
&=sum(\mathbf x)
\end{align}$$

## 4.5 The Chain Rules
链式法则在概念上属于分治算法(例如快排)，将复杂的/嵌套的表达式分解为更容易计算导数的子表达式
运用链式法则，我们独立计算每个子表达式的导数，最后将所有的中间结果结合得到最终的结果

### 4.5.1 Single-variable chain rule
单变量链式法则用于求一元标量值函数相对于其变量的导数
考虑嵌套函数$y = f(g(x))$，应用单变量链式法则求$y$相对于$x$的导数
$$\frac {dy}{dx} = \frac {dy}{du}\frac {du}{dx}$$
其中引入了中间变量$u = g(x)$

使用单变量链式法则的步骤：
1. 为嵌套的子表达式引入中间变量
2. 为中间变量计算它们相对于其参数的导数
3. 将所有中间变量的导数相乘
4. 替换上一步得到的结果中的中间变量为其原来的表示

通过上述步骤求$y = f(g(x)) = sin(x^2)$对于$x$的导数
1. 引入中间变量$u = x^2$用于代替子表达式$x^2$
	得到两个简单函数$u = x^2$和$y = sin(u)$
2. 计算导数
	$\frac {du}{dx} = 2x$
	$\frac {dy}{du} = cos(u)$
3. 结合
	$\frac {dy}{dx} = \frac {dy}{du}\frac {du}{dx} = cos(u)2x$
4. 替换
	$\frac {dy}{dx} = \frac {dy}{du}\frac {du}{dx} = cos(u)2x = cos(x^2)2x = 2xcos(x^2)$

单变量链式法则的应用场景是所有的中间函数都是单变量标量函数，
例如$y = y(u),u = u(x)$，即$x$对$y$的影响只有一条路径

利用链式法则展开深度嵌套的表达式可以理解为编译器将嵌套的函数调用如$f_4(f_3(f_2(f_1(x))))$展开为一串连续的函数调用如$f_1,f_2,f_3,f_4$的过程
函数$f_i$的结果将作为参数传递给$f_{i+1}$

例如计算$y = f(x) = ln(sin(x^3)^2)$对$x$的导数
1. 引入中间变量
	$u_1 = f_1(x) = x^3$
	$u_2 = f_2(u_1) = sin(u_1)$
	$u_3 = f_3(u_2) = u_2^2$
	$u_4 = f_4(u_3) = ln(u_3)$
2. 计算导数
	$\frac d {dx} u_1 = \frac d {dx} x^3 = 3x^2$
	$\frac d {du_1}  u_2= \frac d {d u_1} sin(u_1) = cos(u_1)$
	$\frac d {du_2} u_3= \frac d {d u_2} u_2^2 = 2u_2$
	$\frac d {du_3} u_4 = \frac d {du_3} ln(u_3) = \frac 1 {u_3}$
3. 结合四个中间结果
	$\frac {dy}{dx} = \frac {du_4}{dx} = \frac {du_4}{du_3}\frac {du_3}{du_2}\frac {du_2}{du_1}\frac {du_1}{dx} = \frac 1 {u_3} 2u_2 cos(u_1)3x^2$
4. 替换
	$\frac {dy}{dx} = \frac {6x^2cos(x^3)}{sin(x^3)}$

### 4.5.2 Single-variable total-derivative chain rule
单变量链式法则只能用于所有的中间函数都是单变量函数的情况

考虑函数$y = f(x) = x + x^2$
如果使用标量加法导数规则(scalar addition derivative rule)，可以立刻得到
$\frac {dy}{dx} = \frac d {dx} x + \frac d {dx} x^2 = 1 + 2x$

但如果考虑只是用链式规则，则该情况不符合单变量链式法则的使用场景
引入中间变量得到
$u_1(x) = x^2$
$u_2(x, u_1) = x + u_1$
显然$u_2$不是仅由$u_1$影响的单变量函数，而是由$u_1,x$共同影响的多变量函数

因为$u_2(x, u) = x + u_1$有多个参数，涉及到了偏导数，我们首先对我们得到的所有等式计算其所有的偏导数
$\frac {\partial u_1(x)}{\partial x} = \frac {\partial x^2}{\partial x} = 2x$
$\frac {\partial u_2(x, u_1)} {\partial u_1} = \frac {\partial}{\partial u_1}( x + u_1) = 0+ 1 =1$
$\frac {\partial u_2(x,u_1)}{\partial x} = \frac {\partial}{\partial x}( x + u_1)=0+1 = 1$
注意偏导数的一个关键假设就是在对$x$求偏导数时，函数中的其他所有变量都不随$x$的改变而改变，因而可以将它们视为常数
因此，这里的$\frac {\partial u_2(x,u_1)}{\partial x}$并不等于实际上$u_2$对$x$的导数，而是在假设了$u_1$与$x$无关情况下得到的一个偏导数

从数据流图来看，$x$有两条不同的路径影响$y$，一条直接影响，一条通过$u_1(x)$间接，可以由以下等式看出$x$的变化$\Delta x$如何影响$y$的值
$$\hat y = (x + \Delta x) + (x+\Delta x)^2$$
$$\Delta y = \hat y - y$$

全导数的思想在于，要计算$\frac {dy}{dx}$，需要将$\Delta x$对$\Delta y$的所有可能的贡献相加
全导数假设了函数$y$的所有参数，例如$u_1$都可能和$x$相关，即随着$x$变化而变化

函数$f(x) = u_2(x, u_1)$的全导数是
$$\frac {dy}{dx} = \frac {\partial f(x)}{\partial x}=\frac {\partial u_2(x, u_1)}{\partial x}=\frac {\partial u_2}{\partial x}\frac {\partial x}{\partial x}+\frac {\partial u_2}{\partial u_1}\frac {\partial u_1}{\partial x} = \frac {\partial u_2}{\partial x}+\frac {\partial u_2}{\partial u_1}\frac {\partial u_1}{\partial x}$$
如果$f(x) = u_2(x, u_1) = x + u_1$，我们使用上面的公式得到$\frac {dy}{dx} = 1 + 2x$，正确的答案

单变量全导数链式法则可以写为$$\frac {\partial f(x,u_1,\dots,u_n)}{\partial x} = \frac {\partial f}{\partial x}+\frac {\partial f}{\partial u_1}\frac {\partial u_1}{\partial x} + +\cdots+\frac {\partial f}{\partial u_n}\frac {\partial u_n}{\partial x}=\frac {\partial f}{\partial x}+\sum_{i=1}^n\frac {\partial f}{\partial u_i}\frac {\partial u_i}{\partial x}$$这里所有的导数都写为偏导数的形式，因为$f, u_i$都是多变量函数

考虑$f(x) = sin(x + x^2)$相对于$x$的全导数
首先引入中间变量
$u_1(x) = x^2$
$u_2(x, u_1) = x + u_1$
$u_3(u_2) = sin(u_2)$
计算导数
$\frac {\partial u_1}{\partial x} = 2x$
$\frac {\partial u_2}{\partial x} = \frac {\partial u_2}{\partial x} + \frac {\partial u_2}{\partial u_1}\frac {\partial u_1}{\partial x} = 1 + 1\times 2x=1+2x$
$\frac {\partial u_3}{\partial x} = \frac {\partial u_3}{\partial u_2}\frac {\partial u_2}{\partial x} = cos(u_2)\frac {\partial u_2}{\partial x} = cos(u_2)(1+2x) = cos(x+x^2)(1+2x)$

对于单变量全导数链式法则，如果再引入一个新的变量$u_{n+1}$作为$x$的别名，
即$u_{n+1} = x$，则原来的形式可以进一步写为
$$\frac {\partial f(u_1,\dots,u_{n+1})}{\partial x} = \sum_{i=1}^{n+1}\frac {\partial f}{\partial u_i}\frac {\partial u_i}{\partial x}$$

注意之所以称为单变量全导数链式法则，是因为事实上总体的函数还是$y = f(x)$的形式，仅有$x$一个参数，$x$只是通过不同的中间变量以不同的路径影响了$y$
通过单变量全导数链式法则是为了向量链式法则铺垫，
将$f(u_1,\dots,u_{n+1})$写作$f(\mathbf u)$，则可以得到$$\frac {\partial f(u_1,\dots,u_{n+1})}{\partial x} = \sum_{i=1}^{n+1}\frac {\partial f}{\partial u_i}\frac {\partial u_i}{\partial x}=\frac {\partial f}{\partial \mathbf u }\frac {\partial \mathbf u}{\partial x}$$我们将和式写成了两个向量$\frac {\partial f}{\partial \mathbf u}$和$\frac {\partial \mathbf u}{\partial x}$点积的形式

### 4.5.3 Vector chain rule
首先考虑一个单变量向量函数$\mathbf y =\mathbf f(x)$
$$\begin{bmatrix}y_1(x)\\y_2(x)\end{bmatrix}=
\begin{bmatrix}f_1(x)\\f_2(x)\end{bmatrix}=
\begin{bmatrix}ln(x^2)\\sin(3x)\end{bmatrix}$$
引入两个中间变量$g_1,g_2$，将函数写为$\mathbf y = \mathbf f(\mathbf g(x))$
$$\begin{align}
&\begin{bmatrix}
g_1(x)\\g_2(x)
\end{bmatrix}
=\begin{bmatrix}
x^2\\3x
\end{bmatrix}\\
&\begin{bmatrix}
f_1(\mathbf g)\\f_2(\mathbf g)
\end{bmatrix}=
\begin{bmatrix}
ln(g_1)\\
sin(g_2)
\end{bmatrix}
\end{align}$$
引入中间变量后得到的$\mathbf f(\mathbf g)$是一个多变量向量函数

向量$\mathbf y$相对于$x$的导数的计算涉及了单变量全导数链式法则，
我们手动将$\mathbf y$展开，分别对各个$y_i$，即$f_i$运用单变量全导数链式法则
$$\frac {\partial \mathbf y}{\partial x}=
\begin{bmatrix}
\frac {\partial f_1(\mathbf g)}{\partial x}\\
\frac {\partial f_2(\mathbf g)}{\partial x}
\end{bmatrix}=
\begin{bmatrix}
\frac {\partial f_1}{\partial g_1}\frac{\partial g_1}{\partial x}+
\frac {\partial f_1}{\partial g_2}\frac{\partial g_2}{\partial x}\\
\frac {\partial f_2}{\partial g_1}\frac{\partial g_1}{\partial x}+
\frac {\partial f_2}{\partial g_2}\frac{\partial g_2}{\partial x}\\
\end{bmatrix}=
\begin{bmatrix}
\frac 1 {g_1}2x+0\\
0+cos(g_2)3
\end{bmatrix}=
\begin{bmatrix}
\frac 2 x\\
3cos(3x)
\end{bmatrix}$$
其中
$$\begin{align}
&\begin{bmatrix}
\frac {\partial f_1}{\partial g_1}\frac{\partial g_1}{\partial x}+
\frac {\partial f_1}{\partial g_2}\frac{\partial g_2}{\partial x}\\
\frac {\partial f_2}{\partial g_1}\frac{\partial g_1}{\partial x}+
\frac {\partial f_2}{\partial g_2}\frac{\partial g_2}{\partial x}\\
\end{bmatrix}
\\
=&
\begin{bmatrix}
\frac {\partial f_1}{\partial g_1}&
\frac {\partial f_1}{\partial g_2}\\
\frac {\partial f_2}{\partial g_1}&
\frac {\partial f_2}{\partial g_2}\\
\end{bmatrix}
\begin{bmatrix}
\frac{\partial g_1}{\partial x}\\
\frac{\partial g_2}{\partial x}\\
\end{bmatrix}\\
=&
\begin{bmatrix}
\frac {\partial f_1}{\partial \mathbf g}\\
\frac {\partial f_2}{\partial \mathbf g}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial g_1}{\partial x}\\
\frac{\partial g_2}{\partial x}\\
\end{bmatrix}\\
=&\begin{bmatrix}
\frac {\partial f_1}{\partial \mathbf g}\\
\frac {\partial f_2}{\partial \mathbf g}
\end{bmatrix}
\frac {\partial \mathbf g}{\partial x}\\
=&\frac {\partial \mathbf f}{\partial \mathbf g}\frac {\partial\mathbf g}{\partial x}
\end{align}
$$
我们通过将$\mathbf f$相对$\mathbf g$的雅可比矩阵和$\mathbf g$相对$x$的雅可比矩阵相乘，得到了$\mathbf f$相对$x$的雅可比矩阵
事实上我们只是将按行依此使用单变量全导数链式法则整合为了矩阵计算的形式，由此得到向量链式法则

之前讨论了单变量的情况，现在进一步考虑多变量函数，即$\mathbf x$是个向量
当$\mathbf x$是个向量，形式依旧没有变化，只是最后得到的雅可比矩阵不再是单列的矩阵，而是多列的矩阵，第$i$列对应$x_i$
$$\begin{align}
&\frac {\partial}{\partial \mathbf x}\mathbf f(\mathbf g(\mathbf x))\\
=&\begin{bmatrix}
\frac {\partial \mathbf f(\mathbf g(\mathbf x))}{\partial x_1},\dots
\frac {\partial \mathbf f(\mathbf g(\mathbf x))}{\partial x_n}
\end{bmatrix}\\
=&\begin{bmatrix}
\frac {\partial \mathbf f}{\partial \mathbf g}\frac {\partial\mathbf g}{\partial x_1},\dots,\frac {\partial \mathbf f}{\partial \mathbf g}\frac {\partial\mathbf g}{\partial x_n}
\end{bmatrix}\\
=&\frac {\partial \mathbf f}{\partial \mathbf g}[\frac {\partial \mathbf g}{\partial x_1},\dots,\frac {\partial \mathbf g}{\partial x_n}]\\
=&\frac {\partial \mathbf f}{\partial \mathbf g}\frac {\partial \mathbf g}{\partial \mathbf x}\\
=&\begin{bmatrix}
\frac {\partial f_1}{\partial g_1} & \cdots & \frac {\partial f_1}{\partial g_k} \\
\vdots & \ddots & \vdots\\
\frac {\partial f_m}{\partial g_1} & \cdots & \frac {\partial f_m}{\partial g_k}
\end{bmatrix}
\begin{bmatrix}
\frac {\partial g_1}{\partial x_1} & \cdots & \frac {\partial g_1}{\partial x_n} \\
\vdots & \ddots & \vdots\\
\frac {\partial g_k}{\partial x_1} & \cdots & \frac {\partial g_k}{\partial x_n}
\end{bmatrix}
\end{align}$$
其中$m$为$|\mathbf f|$，$k$为$|\mathbf g|$，$n$为$|\mathbf x|$
此即向量链式法则的表现形式，和标量链式法则的表现形式是一致的

而神经网络处理的涉及是以向量为参数的函数，较少延伸到函数向量(一系列函数组织为一个向量)，例如神经元仿射函数是$sum(\mathbf w \otimes \mathbf x)$，激活函数是$max(0,\mathbf x)$

在之前提到，对于对向量$\mathbf x$进行按元素运算得到的结果$\mathbf w$相对于$\mathbf x$的雅可比矩阵是一个以$\frac {\partial w_i}{\partial x_i}$为对角线元素的对角矩阵，其余项为$0$，因为$w_i$仅仅是$x_i$的函数，与$x_j,j\ne i$无关
同样，如果$f_i$仅仅是$g_i$的函数，$g_i$仅仅是$x_i$的函数
$$\begin{align}
\frac {\partial \mathbf f}{\partial \mathbf g} = diag(\frac {\partial f_i}{\partial g_i})\\
\frac {\partial \mathbf g}{\partial \mathbf x} = diag(\frac {\partial g_i}{\partial x_i})\\
\end{align}$$
在这种情况下，向量链式法则简化为
$$\frac {\partial}{\partial \mathbf x}\mathbf f(\mathbf g(\mathbf x)) = diag(\frac {\partial f_i}{\partial g_i})diag(\frac {\partial g_i}{\partial x_i}) = diag(\frac {\partial f_i}{\partial g_i}\frac {\partial g_i}{\partial x_i})$$
即雅可比矩阵简化为对角线为对应标量链式法则结果的对角矩阵

# 5 The gradient of neuron activation
目前，我们已经可以计算一个典型的神经元激活相对于其参数$\mathbf w, b$的导数
$$activation(\mathbf x)=max(0,\mathbf w\cdot\mathbf x+b)$$
该式表示了一个全连接层和一个ReLU激活函数

考虑$\frac {\partial }{\partial \mathbf w}(\mathbf w\cdot \mathbf x+b)$和$\frac {\partial }{\partial b}(\mathbf w\cdot \mathbf x + b)$
向量点积实际上就是两个向量进行按元素乘法后再求和$\mathbf w\cdot \mathbf x = sum(\mathbf w \otimes \mathbf x) = \sum_{i}(w_ix_i)$
因此，引入一个中间变量$\mathbf u$，将点积运算写为
$\mathbf u = \mathbf w\otimes \mathbf x$
$y = sum(\mathbf u)$
分别计算各自的偏导数
$\frac {\partial \mathbf u}{\partial \mathbf w} = \frac {\partial }{\partial \mathbf w}(\mathbf w\otimes \mathbf x) = diag(\mathbf x)$
$\frac {\partial y}{\partial \mathbf u}  = \frac {\partial }{\partial \mathbf u}sum(\mathbf u) = \vec 1^T$
应用向量链式法则得到
$$\frac {\partial y}{\partial \mathbf w} = \frac {\partial y}{\partial \mathbf u}\frac {\partial\mathbf u }{\partial \mathbf w} = 1^Tdiag(\mathbf x)= \mathbf x^T$$

通过将点积运算写为标量计算的形式进行验证
$y = \mathbf  w\cdot \mathbf x = \sum_{i}(w_ix_i)$
$\frac {\partial y}{\partial w_j} = \frac {\partial }{\partial w_j}\sum_i(w_ix_i) = \sum_i \frac {\partial }{\partial w_j}(w_ix_i)= \frac {\partial }{\partial w_j}(w_jx_j) = x_j$
则
$\frac {\partial y}{\partial \mathbf w} = [x_1,\dots, x_n] = \mathbf x^T$

现在，令$y = \mathbf w \cdot \mathbf x + b$，可以得到
$\frac {\partial y}{\partial \mathbf w} = \frac {\partial }{\partial \mathbf w} (\mathbf w \cdot \mathbf x) + \frac {\partial}{\partial \mathbf w} b = \mathbf x^T + \vec 0^T = \mathbf x^T$
$\frac {\partial y}{\partial b} = \frac {\partial }{\partial b} (\mathbf w \cdot \mathbf x) + \frac {\partial}{\partial b} b = 0+1 = 1$

现在考虑激活函数$max(0,z)$，它的偏导数是一个分段函数
$\frac {\partial }{\partial z}max(0,z) = \begin{cases}0 & z\le 0  \\  \frac {dz}{dz} = 1 & z > 0\end{cases}$
如果参数是一个向量$max(0, \mathbf x)$，则按元素进行运算
$max(0, \mathbf x) = \begin{bmatrix}max(0, x_1)\\ max(0, x_2)\\ \vdots \\ max(0, x_n) \end{bmatrix}$
$$\frac {\partial }{\partial \mathbf x} max(0,\mathbf x) = \begin{bmatrix}
\frac {\partial }{\partial x_1}
max(0,x_1) & \cdots & \frac {\partial }{\partial x_n}max(0,x_1)\\
\vdots & \ddots &  \vdots \\
\frac {\partial }{\partial x_1}max(0,x_n) & \cdots & \frac {\partial }{\partial x_n}max(0,x_n)
\end{bmatrix}
= diag(\frac {\partial }{\partial x_i}max(0,x_i))$$
如果把$max(0,\mathbf x)$视为一个广播函数，也可以逐元素求偏导数
$$\frac {\partial }{\partial \mathbf x}max(0,\mathbf x) = \begin{bmatrix}\frac {\partial }{\partial x_1}max(0,x_1)\\\frac {\partial }{\partial x_2}max(0,x_2)\\ \vdots \\ \frac {\partial }{\partial x_n}max(0,x_n)\end{bmatrix}$$


现在考虑$activation(\mathbf x)$的偏导数，引入一个中间变量$z$
$z(\mathbf w, b, \mathbf w) = \mathbf w \cdot \mathbf x + b$
$activation(z) = max(0, z)$
分别求偏导数
$\frac {\partial } {\partial \mathbf w} z= \frac {\partial }{\partial \mathbf w}(\mathbf w \cdot \mathbf x + b) =\mathbf x^T$
$\frac {\partial } {\partial b} z= \frac {\partial }{\partial b}(\mathbf w \cdot \mathbf x + b) = 1$
$\frac {\partial }{\partial z} activation = \frac {\partial }{\partial z}max(0,z) = \begin{cases}0 & z\le 0  \\  \frac {dz}{dz} = 1 & z > 0\end{cases}$
利用向量链式法则
$$\begin{align}
\frac {\partial activation}{\partial \mathbf w} = \frac {\partial activation}{\partial z}\frac {\partial z}{\partial \mathbf w}\\
\end{align}$$
得到
$$\begin{align}
\frac {\partial activation}{\partial \mathbf w} = \frac {\partial activation}{\partial z}\frac {\partial z}{\partial \mathbf w}=\begin{cases}
0\frac {\partial z}{\partial \mathbf w} = \vec 0^T & z\le 0\\
1\frac {\partial z}{\partial \mathbf w} = \frac {\partial z}{\partial \mathbf w}=\mathbf x^T&z>0
\end{cases}
\end{align}$$
将$z = \mathbf w \cdot \mathbf x + b$替换回去，得到
$$\begin{align}
\frac {\partial activation}{\partial \mathbf w} = \begin{cases}
 \vec 0^T & \mathbf w\cdot \mathbf x+b\le 0\\
\mathbf x^T&\mathbf w\cdot \mathbf x +b>0
\end{cases}
\end{align}$$
同理，利用向量链式法则
$$\frac {\partial activation}{\partial b} = \frac {\partial activation}{\partial z}\frac {\partial z}{\partial b}$$
得到
$$\begin{align}
\frac {\partial activation}{\partial b} = \frac {\partial activation}{\partial z}\frac {\partial z}{\partial b}=
\begin{cases}
0\frac {\partial z}{\partial b} = 0 & \mathbf w\cdot \mathbf x+b\le 0\\
1\frac {\partial z}{\partial b} = 1 & \mathbf w \cdot \mathbf x +b>0
\end{cases} 
\end{align}$$
# 6 The gradient of the neural network loss function
训练神经网络时，我们需要计算损失相对于模型参数$\mathbf w, b$的偏导数
在训练时，我们会给出多个向量输入(如多张图像)和标量目标(如每张图像的类别)

令$\mathbf X = [\mathbf x_1,\mathbf x_2,\dots , \mathbf x_N]^T$，其中$N = |\mathbf X|$
令$\mathbf y = [target(\mathbf x_1),target(\mathbf x_2),\dots, target(\mathbf x_N)]^T$，其中$y_i$是一个标量

损失函数为
$$C(\mathbf w, b, \mathbf X, \mathbf y) = \frac 1 N\sum_{i=1}^N(y_i-activation(\mathbf x_i))^2=\frac 1 N\sum_{i=1}^N(y_i-max(0,\mathbf x_i\cdot \mathbf w+b))^2$$
我们引入一些中间变量，将其重写为
$u(\mathbf w, b, \mathbf x) = max(0, \mathbf w\cdot \mathbf x + b)$
$v(y, u) = y- u$
$C(\mathbf v) = \frac 1 N \sum_{i=1}^N v_i^2$

## 6.1 The gradient with respect to the weights
我们已经知道$$\begin{align}
\frac {\partial}{\partial \mathbf w} u(\mathbf w,b,\mathbf x)= \begin{cases}
 \vec 0^T & \mathbf w\cdot \mathbf x+b\le 0\\
\mathbf x^T&\mathbf w\cdot \mathbf x +b>0
\end{cases}
\end{align}$$且容易算出$$\frac {\partial v(y,u)}{\partial \mathbf w} = \frac{\partial}{\partial \mathbf w}(y-u) = \vec 0^T-\frac {\partial u }{\partial \mathbf w}=-\frac {\partial u}{\partial\mathbf w } = \begin{cases}
 \vec 0^T & \mathbf w\cdot \mathbf x+b\le 0\\
-\mathbf x^T&\mathbf w\cdot \mathbf x +b>0
\end{cases}$$
则容易得到
$$\begin{align}
\frac {\partial C(\mathbf v)}{\partial \mathbf w}
&=\frac {\partial }{\partial \mathbf w}\frac 1 N\sum_{i=1}^Nv_i^2\\
&=\frac 1 N\sum_{i=1}^N\frac {\partial }{\partial \mathbf w}v_i^2\\
&=\frac 1 N\sum_{i=1}^N\frac {\partial v_i^2}{\partial v_i}\frac {\partial v_i}{\partial \mathbf w}\\
&=\frac 1 N \sum_{i=1}^N2v_i\frac {\partial v_i}{\partial \mathbf w}\\
&=\frac 1 N \sum_{i=1}^N\begin{cases}
2v_i\vec 0^T=\vec 0^T & \mathbf w\cdot \mathbf x_i+b\le0
\\-2v_i\mathbf x_i^T & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\frac 1 N \sum_{i=1}^N\begin{cases}
\vec 0^T & \mathbf w\cdot \mathbf x_i+b\le0
\\-2(y_i-u_i)\mathbf x_i^T & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\frac 1 N \sum_{i=1}^N\begin{cases}
\vec 0^T & \mathbf w\cdot \mathbf x_i+b\le0
\\-2(y_i-max(0,\mathbf w\cdot\mathbf x_i+b))\mathbf x_i^T & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\frac 1 N \sum_{i=1}^N\begin{cases}
\vec 0^T & \mathbf w\cdot \mathbf x_i+b\le0
\\-2(y_i-(\mathbf w\cdot\mathbf x_i+b))\mathbf x_i^T & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\begin{cases}
\vec 0^T & \mathbf w\cdot \mathbf x_i+b\le0
\\\frac {-2} N\sum_{i=1}^N(y_i-(\mathbf w\cdot\mathbf x_i+b))\mathbf x_i^T & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\begin{cases}
\vec 0^T & \mathbf w\cdot \mathbf x_i+b\le0
\\\frac {2} N\sum_{i=1}^N(\mathbf w\cdot\mathbf x_i+b -y_i)\mathbf x_i^T & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}
\end{align}$$
引入一个名为误差项的变量$e_i = \mathbf w\cdot \mathbf x_i + b - y_i$，原式改写为$$\frac {\partial C}{\partial \mathbf w}=\begin{cases}
\vec 0^T & \mathbf w\cdot \mathbf x_i+b\le0
\\\frac {2} N\sum_{i=1}^N e_i\mathbf x_i^T & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}$$注意到对于成功激活的神经元(输入大于$0$)，损失函数相对于$\mathbf w$的梯度可以视为是对$X$中的所有$\mathbf x_i$的一个加权平均，而权重就是对应的误差项
因此误差项$e_i$越小，梯度在$\mathbf x_i$的方向的步长就越小，如果误差项$e_i$越大，则梯度在$\mathbf x_i$方向上的步长就越大，如果误差项$e_i$为负，则梯度在$\mathbf x_i$方向上的步长就为反向，这使得梯度总是指向了平均下可以使得损失$C$最大化的方向

为了使损失$C$最小，我们进行梯度下降，向负梯度方向更新
$$\mathbf w_{t+1} = \mathbf w_{t}-\eta\frac {\partial C}{\partial \mathbf w}$$

## 6.2 The derivative with respect to bias
要优化偏置项$b$，我们引入中间变量
$u(\mathbf w, b, \mathbf x) = max(0, \mathbf w\cdot \mathbf x + b)$
$v(y, u) = y- u$
$C(\mathbf v) = \frac 1 N \sum_{i=1}^N v_i^2$

我们已经知道$$\begin{align}
\frac {\partial}{\partial b} u(\mathbf w,b,\mathbf x)= \begin{cases}
 0& \mathbf w\cdot \mathbf x+b\le 0\\
1&\mathbf w\cdot \mathbf x +b>0
\end{cases}
\end{align}$$
且容易算出$$\frac {\partial v(y,u)}{\partial b} = \frac{\partial}{\partial b}(y-u) = 0-\frac {\partial u }{\partial b}=-\frac {\partial u}{\partial b } = \begin{cases}
0 & \mathbf w\cdot \mathbf x+b\le 0\\
-1&\mathbf w\cdot \mathbf x +b>0
\end{cases}$$
则容易得到
$$\begin{align}
\frac {\partial C(\mathbf v)}{\partial b}
&=\frac {\partial }{\partial b}\frac 1 N\sum_{i=1}^Nv_i^2\\
&=\frac 1 N\sum_{i=1}^N\frac {\partial }{\partial b}v_i^2\\
&=\frac 1 N\sum_{i=1}^N\frac {\partial v_i^2}{\partial v_i}\frac {\partial v_i}{\partial b}\\
&=\frac 1 N \sum_{i=1}^N2v_i\frac {\partial v_i}{\partial b}\\
&=\frac 1 N \sum_{i=1}^N\begin{cases}
0 & \mathbf w\cdot \mathbf x_i+b\le0
\\-2v_i & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\frac 1 N \sum_{i=1}^N\begin{cases}
0& \mathbf w\cdot \mathbf x_i+b\le0
\\-2(y_i-u_i)& \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\frac 1 N \sum_{i=1}^N\begin{cases}
0 & \mathbf w\cdot \mathbf x_i+b\le0
\\-2(y_i-max(0,\mathbf w\cdot\mathbf x_i+b)) & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\frac 1 N \sum_{i=1}^N\begin{cases}
0& \mathbf w\cdot \mathbf x_i+b\le0
\\2(\mathbf w\cdot\mathbf x_i+b -y_i) & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
&=\begin{cases}
0 & \mathbf w\cdot \mathbf x_i+b\le0
\\\frac {2} N\sum_{i=1}^N(\mathbf w\cdot\mathbf x_i+b -y_i) & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}\\
\end{align}$$
同样引入误差项，将原式改写为
$$\frac {\partial C}{\partial b} =
\begin{cases}
0 & \mathbf w\cdot \mathbf x_i+b\le0
\\\frac {2} N\sum_{i=1}^Ne_i & \mathbf w\cdot \mathbf x_i+b>0
\end{cases}$$
因此损失相对于偏置$b$的偏导数只能是误差项的平均或零
同样使用梯度下降更新$b$
$$b_{t+1}  = b_t-\eta\frac {\partial C}{\partial b}$$
事实上可以将$\mathbf w, b$结合为一个向量参数$\hat {\mathbf w} = [\mathbf w^T , b]^T$，
再对输入向量作一定修改$\hat {\mathbf x} = [\mathbf x^T , 1]^T$，
就可以将$\mathbf w \cdot \mathbf x + b$改写为$\hat {\mathbf w}\cdot \hat {\mathbf x}$

# 7 Summary
# 8 Matrix Calculus Reference
# 9 Notation
# 10 Resources

