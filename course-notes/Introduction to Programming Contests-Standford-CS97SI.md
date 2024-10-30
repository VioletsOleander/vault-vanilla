# Mathmatics 数学
## Algebra 代数
(1) Sum of Powers
平方和：
$$\sum_{k=1}^n k^2 = \frac 1 6 n(n+1)(2n+1)$$
证明(数学归纳法):
基线情况: 当n = 1时基线情况成立：$$1 = \frac 1 6\times2\times3$$
归纳假设: 假设当n = k时等式成立，即：$$\sum_{i=1}^k i^2 = \frac{1}{6}k(k+1)(2k+1)$$
归纳步骤: 我们需要证明当n = k + 1时等式也成立，即：$$\sum_{i=1}^{k+1} i^2 = \frac{1}{6}(k+1)(k+2)(2k+3)$$
我们可以将左侧的求和式展开：
$$\begin{aligned}\sum_{i=1}^{k+1} i^2 &= (k+1)^2 + \sum_{i=1}^{k} i^2 \\
&= (k+1)^2+\frac 1 6 k(k+1)(2k+1) \\
&= (k+1)(k+1+\frac 1 6 k(2k+1))\\
& = (k+1)(\frac 1 3k^2+\frac 7 6k + 1) \\
& = \frac 1 6(k+1)(2k^2+7k+6) \\
& = \frac 1 6(k+1)(k+2)(2k+3) 
\end{aligned}$$
因此，根据数学归纳法，我们证明了等式对于所有正整数n成立

立方和：
$$\sum_{i=1}^n k^3 = (\sum_{i=1}^nk)^2 = (\frac 1 2n(n+1))^2$$
证明思路依旧是数学归纳法

(2) Fast Exponentiation(快速幂次计算)
递归计算$a^n$
$$a^n = \begin{cases}1 \quad n=0 \\
a \quad n = 1 \\
(a^{n/2})^2 \quad n为偶数 \\
a(a^{(n-1)/2})^2 \quad n为奇数 
\end{cases}$$
代码：
```cpp
double pow(double a, int n){
	if(n == 0) return 1;
	if(n == 1) return a;
	double t = pow(a, n/2); \\递归调用
	return t * t * pow(a, n%2);
}
```
时间复杂度是$O(logn)$

(3) Linear Algebra
用高斯消元法解决线性方程组求解，矩阵转置，求矩阵的秩，计算矩阵的行列式
## Number Theory 数论
(1) Greatest Common Divisor(GCD) 最大公约数
1. gcd(a, 0) = a
2. Euclidean Algorithm(辗转相除法)
	反复使用gcd(a, b) = gcd(a, b-a)
	gcd(a, b) = gcd(a, b-a)是因为对于a, b之间任意的一个公因子d, 可以证明d也是a, b-a之间的公因子，反之亦然，因此a, b之间和a, b-a之间有相同的公因子，就有相同的GCD
	因此gcd(a, b) = gcd(a, b-a) = gcd(a, b-2a) = ... = gcd(a, b%a)
	这些公因子都可以写成$ax+by\quad x,y \in Z$的形式
 
	代码：
	```cpp
	int gcd(int a, int b){
		while(b){int r = a%b; a = b; b = r;}
		return a;
	}
	```
	时间复杂度是$O(log(a+b))$

(2) Congruence & Modulo Operation 
对于余$n$同余关系$x \equiv y(mod n)$：
	若$x^{-1}$是$x$余$n$的逆，则$xx^{-1} \equiv 1(mod n)$
	例如，因为$3\times 5 = 15, 15\  mod\  7 = 1$，故$5^{-1} \equiv 3(mod 7)$
	$x^{-1}$存在的条件是gcd(x, n) = 1，即x和n互质

(3) Chinese Remainder Theorem
有a, b, m, n，且gcd(m, n) = 1
求x使得$x\equiv a(mod\ m)$且$x\equiv  b(mod\ n)$

解决：
令$n^{-1}$为$n$的模$m$逆，即$n\times n^{-1} = 1(mod\ m)$
令$m^{-1}$为$m$的模$n$逆，即$m\times m^{-1} = 1(mod\ n)$
则解为：$x = ann^{-1} + bmm^{-1}$

## Combinatorics 组合数学
(1) Binomial Coefficients 二项式系数
$$\dbinom n k = C_n^k = \frac {n(n-1)(n-2)\cdots(n-k+1)}{k!}$$
在$(x+y)^n$中，项$x^ky^{n-k}$和$x^{n-k}y^k$的系数即为$C_n^k$

(2) Finonacci Sequence 斐波那契数列
$F_0 = 0, F_1  = 1$
$F_n = F_{n-1}+F_{n-2}\quad n\geqslant2$

$$\begin{bmatrix}
F_{n+1} \\
F_{n}
\end{bmatrix}
=\begin{bmatrix}
1 & 1\\
1 & 0
\end{bmatrix}
\begin{bmatrix}
F_n \\
F_{n-1}
\end{bmatrix}
= \begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix}^n
\begin{bmatrix}
F_1 \\
F_0
\end{bmatrix}$$
矩阵$$\begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix}$$
的两个特征值是$\lambda_1 = \frac {1 + \sqrt 5} 2,\lambda_2 = \frac {1 - \sqrt 5} 2$
因此可以做FFT快速计算矩阵的n次幂

## Geometry 几何
(1) 2D Vector Operations
对于一个二维向量(x, y)
将其逆时针旋转$\theta$：
$$\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}$$
即将$$\begin{bmatrix} 1 \\ 0\end{bmatrix},\begin{bmatrix}1 \\ 0 \end{bmatrix}$$两个基
变换为
$$\begin{bmatrix}\cos \theta \\ \sin \theta \end{bmatrix},
\begin{bmatrix}-\sin \theta \\ \cos \theta \end{bmatrix}$$

(2) Line-Line Intersection
两条直线$ax+by+c = 0, dx+ey+f = 0$
写成矩阵形式
$$\begin{bmatrix}
a & b \\
c & d \\
\end{bmatrix}\begin{bmatrix}
x \\
y
\end{bmatrix}= -
\begin{bmatrix}
c \\
f
\end{bmatrix}$$
解出$\begin{bmatrix} x \\ y \end{bmatrix}$即为两条直线的交点
当$ae = bd$两直线平行，无交点或有无限个交点

应用：Circumcircle of a Triangle 三角形的外接圆
有三个顶点ABC，需要找到圆心P，P到ABC的距离相等
可以写出AB的中垂线的方程，BC的中垂线的方程
解出两条直线的交点，即为点P

(3) Area of a Triangle
用叉乘计算

(4) Area of a Simple Polygon
将多边形分解为三角形，每个三角形面积用叉乘计算
# Appendix
## C++输入输出
### cin
(1) cin>>以空白字符(space, tab, newline)等为分隔符，通过分隔符划分输入，读入时会忽略分隔符
```cpp
char a; int b; float c; string d;
cin >> a >> b >> c >> d;
```

(2) cin.get()常用的四种重载形式:
```cpp
int cin.get()
istream& cin.get(char& var)
istream& cin.get(char* s, streamsize n)
istream& cin.get(char* s ,streamsize n, char delim)
```

读取一个字符：
```cpp
char a; a = cin.get();
char a; cin.get(a);
```
cin.get()不会忽略分隔符，如果读取成功，返回字符的ASCII码值，如果遇到文件结束符EOF(Windows上时Ctrl+z，linux上是Ctrl+d)，返回-1
cin.get(char& var)返回的是cin对象，因此可以链式操作: cin.get(a).get(b);

读取一行：
```cpp
char array[20]; cin.get(array, 20);
```
cin.get(char\* s, streamsize n)在读取一行时，遇到结束符停止读取，但不结束符进行处理，结束符仍留在缓冲区，可以通过delim指定结束符，默认是换行符
cin.get(char* s, streamsize n)只能读取入C风格的字符串

(3) cin.getline()的两种重载形式：
```cpp
istream& cin.getline(char* s, streamsize n);
istream& cin.getline(char* s, streamsize n, char delim);
```

读取一行：
```cpp
char array[20]; cin.getline(array, 20);
```
cin.getline(char\* s, streamsize count)在读取一行时，会对结束符进行读取，并将其去除，可以通过delim指定结束符
cin.getline(char* s, streamsize count)只能读取入C风格的字符串

(4) 如果上一次的输入操作在输入缓冲区中残留了数据，影响了下一次的输入操作，可以首先使用cin.clear()清空cin对象的状态标志位，然后用cin.ignore()清空输入缓冲区
```cpp
void cin.clear();
istream& cin.ignore(streamsize n, int delim = EOF);
```
cin.ignore(streamsize n)会跳过输入流中的n个字符，如果遇到了结束符，会提前结束，结束符可以通过delim指定，默认是EOF

### getline
getline()的两种重载形式：
```cpp
istream& getline(istream& is, string& str);
isteram& getline(istream& is, string& str, char delim);
```

读取一行：
```cpp
string str; geline(std::cin, str);
```
getline()在遇到结束符时，也会读取结束符，并将其替换为空字符，可以通过delim指定结束符，默认是换行符
getline()在遇到结束符，或遇到文件结束，或遇到输入达到最大限度会结束读取