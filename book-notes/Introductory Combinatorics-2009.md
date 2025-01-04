# 1 What Is Combinatorics? 
 Today, combinatorics is an important branch of mathematics. One of the reasons for the tremendous growth of combinatorics has been the major impact that computers have had and continue to have in our society. Because of their increasing speed, computers have been able to solve large-scale problems that previously would not have been possible. But computers do not function independently. They need to be programmed to perform. The bases for these programs often are combinatorial algorithms for the solutions of problems. 
 Analysis of these algorithms for efficiency with regard to running time and storage requirements demands more combinatorial thinking.
> 对于计算机算法效率 (运行时间和存储需求) 方面的考量需要组合数学思想

Combinatorics is concerned with arrangements of the objects of a set into patterns satisfying specified rules. 
> 组合数学关心的是将一个集合内的对象按照满足特定规则的方式排布

Two general types of problems occur repeatedly:

- *Existence of the arrangement.* If one wants to arrange the objects of a set so that certain conditions are fulfilled, it may not be at all obvious whether such an arrangement is possible. This is the most basic of questions. If the arrangement is not always possible, it is then appropriate to ask under what conditions, both necessary and sufficient, the desired arrangement can be achieved.
- *Enumemtion or classification of the arrangements.* If a .specified arrangement is possible, there may be several ways of achieving it. If so, one may want to count or to classify them into types. 
> 组合数学中两种常规的问题类型：
> - 排布的存在与否：满足某种要求的排布不一定存在，需要探究在什么情况下，可以达到理想的排布
> - 对于排布的计数或分类：若指定的排布存在，则达成这一排布有几种方法、可以分为几类

If the number of arrangements for a particular problem is small, the arrangements can be listed. It is important to understand the distinction between listing all the arrangements and determining their number. Once the arrangements are listed, they can be counted by setting up a one-to-one correspondence between them and the set of integers {I, 2, 3, ... , n} for some n. This is the way we count: one, two, three, ... . However, we shall be concerned primarily with techniques for determining the number of arrangements of a particular type without first listing them. 
> 可数的和仅知道数量是两个不同的概念

Of course the number of arrangements may be so large as to preclude listing them all. Two other combinatorial problems often occur.

- *Study of a known arrangement.* After one has done the (possibly difficult) work of constructing an arrangement satisfying certain specified conditions, its properties and structure can then be investigated.
- *Construction of an optimal arrangement.* If more than one arrangement is possible, one may want to determine an arrangement that satisfies some optimality criterion-that is, to find a "best" or "optimal" arrangement in some prescribed sense.
> 排布的数量过多，无法列出，此时出现的组合问题：
> - 对已知排布的研究
> - 构造一个最优的排布：从多个可行的排布中选出最优的排布

Thus, a general description of combinatorics might be that *combinatorics is concerned with the existence, enumemtion, analysis, and optimization of discrete structures.* In this book, discrete generally means "finite," although some discrete structures are infinite.
> 组合数学是关于离散结构的：
> - 存在
> - 枚举
> - 分析
> - 优化
> 本书中关心的离散结构就是“有限”

One of the principal tools of combinatorics for verifying discoveries is mathematical induction. Induction is a powerful procedure, and it is especially so in combinatorics. It is often easier to prove a stronger result than a weaker result with mathematical induction. Although it is necessary to verify more in the inductive step, the inductive hypothesis is stronger. Part of the art of mathematical induction is to find the right balance of hypotheses and conclusions to carry out the induction. We assume that the reader is familiar with induction; he or she will become more so as a result of working through this book.
> 主要的数学工具：数学归纳

The solutions of combinatorial problems can often be obtained using ad hoc arguments, possibly coupled with use of general theory. One cannot always fall back on application of formulas or known results. 
A typical solution of a combinatorial problem might encompass the following steps: (1) Set up a mathematical model, (2) study the model, (3) do some computation for small cases in order to develop some confidence and insight, and (4) use careful reasoning and ingenuity to finally obtain the solution of the problem. 
> 对于组合问题的典型求解步骤：
> - 设定数学模型
> - 研究模型
> - 对于小样例进行计算，探究规律
> - 推理得到问题的最终解

For counting problems, the inclusion-exclusion principle, the so-called pigeonhole principle, the methods of recurrence relations and generating functions, Burnside's theorem, and Polya's counting formula are all examples of general principles and methods that we will consider in later chapters. Often, however, cleverness is required to see that a particular method or formula can be applied and how to apply. Thus, experience in solving combinatorial problems is very important.

The implication is that with combinatorics, as with mathematics in general, the more problems one soives, the more likely one is able to solve the next problem.
> 求解组合数学问题需要熟练度

We now consider a few introductory examples of combinatorial problems. They vary from relatively simple problems (but whose solution requires ingenuity) to problems whose solutions were a major achievement in combinatorics. Some of these problems will be considered in more detail in subsequent chapters.
## 1.1 Example: Perfect Covers of Chessboards 
Consider an ordinary chessboard which is divided into 64 squares in 8 rows and 8 columns. Suppose there is available a supply of identically shaped dominoes, pieces which cover exactly two adjacent squares of the chessboard. 

Is it possible to arrange 32 dominoes on the chessboard so that no 2 dominoes overlap, every domino covers 2 squares, and all the squares of the chessboard are covered? We call such an arrangement a perfect cover or tiling of the chessboard by dominoes. 
> 完美覆盖：每个牌占两格，没有重叠，32个牌占满棋盘

This is an easy arrangement problem, and we can quickly construct many different perfect covers. It is difficult, but nonetheless possible, to count the number of different perfect covers. This number was found by Fischer in 1961 to be 12,988,816 = 24 X 172 X 532. 
> $8\times 8$ 棋盘的完美覆盖的个数如上

The ordinary chessboard can be replaced by a more general chessboard divided into $m\times n$ squares lying in m rows and n columns. A perfect cover need not exist now. Indeed, there is no perfect cover for the 3-by-3 board. For which values of m and n does the m-by-n chessboard have a perfect cover? It is not difficult to see that an m-by-n chessboard will have a perfect cover if and only if at least one of m and n is even or, equivalently, if and only if the number of squares of the chessboard is even. 
> $m\times n$ 的棋盘当且仅当至少 $m$ 或 $n$ 是偶数，或者说棋盘的方格数量是偶数时，存在完美覆盖

Fischer has derived general formulas involving trigonometric functions for the number of different perfect covers for the m-by-n chessboard. This problem is equivalent to a famous problem in molecular physics known as the dimer problem. It originated in the investigation of the absorption of diatomic atoms (dimers) on surfaces. The squares of the chessboard correspond to molecules, while the dominoes correspond to the dimers.

Consider once again the 8-by-8 chessboard and, with a pair of scissors, cut out two diagonally opposite corner squares, leaving a total of 62 squares. Is it possible to arrange 31 dominoes to obtain a perfect cover of this "pruned" board? Although the pruned board is very close to being the 8-by-8 chessboard, which has over 12 million perfect covers, it has no perfect cover.
> 移除对角的两个方格，留下 62 个格子
> 此时棋盘没有对应于 31 个牌的完美覆盖

The proof of this is an example of simple, but clever, combinatorial reasoning. In an ordinary 8-by-8 chessboard, usually the squares are alternately colored black and white, with 32 of the squares colored white and 32 of the squares colored black. If we cut out two diagonally opposite corner squares, we have removed two squares of the same color, say white. This leaves 32 black and 30 white squares. But each domino will cover one black and one white square, so that 31 nonoverlapping dominoes on the board cover 31 black and 31 white squares. We conclude that the pruned board has no perfect cover. 
> 移除的对角的两个方格是同一个颜色的，例如白色，故此时黑色比白色多两个格子，而每个牌覆盖一黑一白
> 因此 31 个不重叠的牌应该覆盖 31 黑和 31 白，和棋盘匹配不上，故不存在完美覆盖

More generally, we can take an m-by-n chessboard whose squares are alternately colored black and white and arbitrarily cut out some squares, leaving a pruned board of some type or other. When does a pruned board have a perfect cover? For a perfect cover to exist, the pruned board must have an equal number of black and white squares. But this is not sufficient, as the example in Figure 1.1 indicates.
> 存在完美覆盖的必要条件是黑格和白格数量相同，但是仍不是充分条件

Thus, we ask: What are necessary and sufficient conditions for a pruned board to have a perfect cover? We will return to this problem in Chapter 9 and will obtain a complete solution. There, a practical formulation of this problem is given in terms of assigning applicants to jobs for which they qualify.

There is another way to generalize the problem of a perfect cover of an m-by-n board by dominoes. Let b be a positive integer. In place of dominoes we now consider 1-by-b pieces that consist of b 1-by-1 squares joined side by side in a consecutive manner. These pieces are called b-ominoes. and they can cover b consecutive squares in a row or b consecutive squares in a column. In Figure 1.2, a 5-omino is illustrated. A 2-omino is simply a domino. A l-omino is also called a monomino. A perfect cover of an m-by-n board by b-ominoes is an arrangement of b-ominoes on the board so that (1) no two b-ominoes overlap, (2) every b-omino covers b squares of the board, and (3) all the squares of the board are covered. 
> 我们再考虑更大的牌：占连续的五个格

When does an m-by-n board have a perfect cover by b-ominoes? Since each square of the board is covered by exactly one b-omino, in order for there to be a perfect cover, b must be a factor of mn. Surely, a sufficient condition for the existence of a perfect cover is that b be a factor of m or b be a factor of n. For if b is a factor of m, we may perfectly cover the m-by-n board by arranging m/b b-ominoes in each of the n columns, while if b is a factor of n we may perfectly cover the board by arranging n/b b-ominoes in each of the m rows. 
> 完美覆盖存在的一个充分条件显然是 m 或 n 是 b 的倍数

Is this sufficient condition also necessary for there to be a perfect cover? Suppose for the moment that b is a prime number and that there is a perfect cover of the m-by-n board by b-ominoes. Then b is a factor of mn and, by a fundamental property of prime numbers, b is a factor of m or b is a factor of n. We conclude that, at least for the case of a prime number b, an m-by-n board can be perfectly covered by b-ominoes if and only if b is a factor of m or b is a factor of n. 

> 考虑 m 或 n 是 b 的倍数是不是完美覆盖存在的必要条件
> - 当 b 是质数：如果完美覆盖存在，首先知道 b 是 mn 的因数，则 b 一定是 m 的因数或者 n 的因数，此时 m 或 n 是 b 的倍数是完美覆盖存在的必要条件


In case b is not a prime number, we have to argue differently. So suppose we have the m-by-n board perfectly covered with b-ominoes. We want to show that either m or n has a remainder of 0 when divided by b. We divide m and n by b obtaining quotients p and q and remainders rand s, respectively:

$$
\begin{array}{r l r}{m}&{=}&{p b+r,\;\;\mathrm{where}\;\;\;\;\;\;0\leq r\leq b-1,}\\ {n}&{=}&{q b+s,\;\;\mathrm{where}\;\;\;\;\;\;0\leq s\leq b-1.}\end{array}
$$

If r = 0, then b is a factor of m. If s = 0, then b is a factor of n. By interchanging the two dimensions of the board, if necessary, we may assume that $r\le s$. We then want to show that r = 0.
> - 当 b 不是质数：我们将 m n 写为上述公式的形式，我们假设 $r\le s$ 
> 我们希望证明出 $r=0$ ，即 b 是 m 的因数，就证明了 m 或 n 是 b 的倍数是完美覆盖存在的必要条件

We now generalize the alternate black-white coloring used in the case of dominoes (b = 2) to b colors. We choose b colors, which we label as 1, 2, ... , b. We color a b-by-b board in the manner indicated in Figure 1.3, and we extend this coloring to an m-by-n board in the manner illustrated in Figure 1.4 for the case m = 10, n = 11, and b = 4.
> 此时，将棋盘中不同颜色的数量泛化到 b 种颜色，涂色顺序以行为主，依次涂色，详见 fig 1.4-1.5

Each b-omino of the perfect covering covers one square of each of the b colors. It follows that there must be the same number of squares of each color on the board. 
> 在已知达到了完美覆盖的情况下，每个牌覆盖棋盘的一个颜色的一个方格，共占 b 个不同颜色的方格，则可以知道棋盘中各个颜色的方格数量应该相同

We consider the board to be divided into three parts: the upper pb-by-n part, the lower left r-by-qb part, and the lower right r-by-s part. (For the 10-by-11 board in Figure 1.4, we would have the upper 8-by-11 part, the 2-by-8 part in the lower left, and the 2-by-3 part in the lower right.) In the upper part, each color occurs p times in each column and hence pn times all together. In the lower left part, each color occurs q times in each row and hence rq times overall. Since each color occurs the same number of times on the whole board, it now follows that each color occurs the same number of times in the lower right r-by-s part.
> 考虑将棋盘分为三个部分，$pb\times n, r\times qb, r\times s$
> 在 $pb \times n$ 部分，每一列中，每个颜色出现 $pb / b = p$ 次，因此该部分每个颜色一共出现了 $p\times n = pn$ 次
> 在 $r\times qb$ 部分，每一行中，每个颜色出现 $qb/b = q$ 次，因此该部分每个颜色一共出现了 $q\times r = rq$ 次
> 因为棋盘内各个颜色的方格数量是相同的，因此可以知道 $r\times s$ 部分中，各个颜色出现的次数也是相同的

How many times does color 1 (and, hence, each color) occur in the r-by-s part? Since $r\le s$, the nature of the coloring is such that color 1 occurs once in each row of the r-by-s part and hence r times in the r-by-s part. Let us now count the number of squares in the r-by-s part. On the one hand, there are rs squares; on the other hand, there are r squares of each of the b colors and so rb squares overall. Equating, we get rs = rb. If r =1= 0, we cancel to get s = b, contradicting $s\le b-1$. So r = 0, as desired. We summarize as follows:
 An m-by-n board has a perfect cover by b-ominoes if and only if b is a factor of m or b is a factor of n.
> 在 $r\times s$ 区域中，我们假设了 $r\le s$，并且已知 $r,s < b$
> 颜色1在每一行出现一次，因此区域内出现了 r 次，则共 b 个颜色，需要 rb 个格子，故得到 $rb = rs$ ，在条件下，该等式只有在 $r=0$ 时成立
> 综合两种情况，可以知道：当且仅当 m 或 n 是 b 的倍数的时候，一个 m-n 的棋盘可以被长度为 b 的棋牌完美覆盖

A striking reformulation of the preceding statement is the following: Call a perfect cover trivial if all the b-ominoes are horizontal or all the b-ominoes are vertical. Then an m-by-n board has a perfect cover by b-ominoes if and only if it has a trivial perfect cover. Note that this does not mean that the only perfect covers are the trivial ones. It does mean that if a perfect cover is possible, then a trivial perfect cover is also possible.
> 平凡的完美覆盖：棋牌只能都横向或者纵向
> 结论：当且仅当 m-n 的棋盘有平凡的完美覆盖，m-n 的棋盘存在完美覆盖，注意这不代表完美覆盖一定是平凡的，而是说明只要完美覆盖存在，则一定存在一个平凡的覆盖

We conclude this section with a domino-covering problem with an added feature.
Consider a 4-by-4 chessboard that is perfectly covered with 8 dominoes. Show that it is always possible to cut the board into two nonempty horizontal pieces or two nonempty vertical pieces without cutting through one of the 8 dominoes. The horizontal or vertical line of such a·cut is called a fault line of the perfect cover. Thus a horizontal fault line implies that the perfect cover of the 4-by-4 chessboard consists of a perfect cover of a k-by-4 board and a perfect cover of a (4 - k )-by-4 board for some k = 1,2, or 3. 
> 考虑一个 4-4 的棋盘，被大小为 8 的棋牌完美覆盖
> 考虑在棋盘中横着或者竖着划一刀，将棋盘分为两个非空的区域，且不划过任意一个棋牌，这一刀称为完美覆盖的 fault line
> fault line 将一个大的完美覆盖横着或者竖着分为了两个小的完美覆盖

Suppose there is a perfect cover of a 4-by-4 board such that none of the three horizontal lines and three vertical lines that cut the board into two nonempty pieces is a fault line. Let $x_1,x_2,x_3$ be, respectively, the number of dominoes that are cut by the horizontal lines (see Figure 1.5)
> 考虑反证法，假设存在一个完美覆盖，它不存在对应的 fault line
> 横着切有三个选择，假设它们切过的棋牌的数量分别是 $x_1,x_2,x_3$

Because there is no fault line, each of $x_1,x_2$ and $x_3$ is positive. A horizontal domino covers two squares in a row, while a vertical domino covers one square in each of two rows. From these facts we conclude successively that $x_1$ is even, $x_2$ is even, and $x_3$ is even. Hence,
> 因为不存在 fault line，显然 $x_1,x_2,x_3$ 不为零，且因为完美覆盖存在，容易推出 $x_{1}, x_{2},x_{3}$ 至少为 2，因此：

$$
x_{1}+ x_{2} + x_{3}\ge 2 + 2 + 2 = 6
$$

and there are at least 6 vertical dominoes in the perfect cover. In a similar way, we conclude that there are at least 6 horizontal dominoes. Since 12 > 8, we have a contradiction. Thus, it is impossible to cover perfectly a 4-by-4 board with dominoes without creating a fault line.
> 也就是说，为了满足三条线都不是 fault line，则至少需要 6 个竖直的牌，类似地，为了满足三条纵向的线也都不是 fault line，则也至少需要 6 个横着的牌，因此牌的总数至少要 12 个，推出矛盾
> 因此结论是：一个 4-4 棋盘上的完美覆盖一定存在 fault line
## 1.2 Example: Magic Squares 
Among the oldest and most popular forms of mathematical recreations are magic squares, which have intrigued many important historical people. A magic square of order n is an n-by-n array constructed out of the integers 1,2,3, ... , $n^2$ in such a way that the sum of the integers in each row, in each column, and in each of the two diagonals is the same numbers. The number s is called the magic sum of the magic square. 
> 幻方是由 $1,2,\dots, n^2$ 构成的 n-n 方阵，它使得方阵的每一行、每一列，以及两个对角线的和都是相同的
> 该和称为幻和

Examples of magic squares of orders 3 and 4 are

$$
\left[\begin{array}{r r r}{8}&{1}&{6}\\ {3}&{5}&{7}\\ {4}&{9}&{2}\end{array}\right]\mathrm{~and~}\left[\begin{array}{r r r r}{16}&{3}&{2}&{13}\\ {5}&{10}&{11}&{8}\\ {9}&{6}&{7}&{12}\\ {4}&{15}&{14}&{1}\end{array}\right]\mathrm{,}
$$ 
with magic sums 15 and 34, respectively. Benjamin Franklin constructed many magic squares with additional properties. The sum of all the integers in a magic square of order n is
> 一个 n 阶幻方中的全部整数的和：

$$
1+2+3+\cdot\cdot\cdot+n^{2}={\frac{n^{2}(n^{2}+1)}{2}},
$$ 
using the formula for the sum of numbers in an arithmetic progression (see Section 7.1). Since a magic square of order n has n rows each with magic sum s, we obtain the relation $ns = n^2(n^{2}+ 1)/2$. Thus, any two magic squares of order n have the same magic sum, namely,
> $n$ 阶幻方的幻和

$$
s = \frac {n(n^2+1)}{2}
$$

The combinatorial problem is to determine for which values of n there is a magic square of order n and to find general methods of construction. 
> 幻方中的组合问题在于对于哪些 n 值，n 阶幻方存在，以及找到构造幻方的通用方法

It is not difficult to verify that there can be no magic square of order 2 (the magic sum would have to be 5). But, for all other values of n, a magic square of order n can be constructed. There are many special methods of construction. 
> 除了不存在 2 阶幻方以外，其他所有的 n 值可以构造 n 阶幻方

We describe here a method found by de la Loubere in the seventeenth century for constructing magic squares of order n when n is odd. 
> 介绍一个当 n 为奇数的时候，构造 n 阶幻方的方法

First a 1 is placed in the middle square of the top row. The successive integers are then placed in their natural order along a diagonal line that slopes upward and to the right, with the following modifications:

(1) When the top row is reached, the next integer is put in the bottom row as if it came immediately above the top row.

(2) When the right-hand column is reached, the next integer is put in the left-hand column as if it had immediately succeeded the right-hand column.

(3) When a square that has already been filled is reached or when the top right-hand square is reached, the next integer is placed in the square immediately below the last square that was filled.

> 先把 1 放在最顶行的中间位置，之后，顺着数字顺序，依次向右上方向放数字，且遵循以下三个规则：
> 1. 如果达到最顶行，视最低行为最顶行的上端一行
> 2. 如果达到最右列，视最左列为最右列的右端一列
> 3. 若遇到了已经被填充的位置，或者达到了最右上角的位置，则下一个位置放在该位置的下面一格

The magic square of order 3 in (1.1), as well as the magic square

$$
\left[{\begin{array}{r r r r}{17}&{24}&{1}&{8}&{15}\\ {23}&{5}&{7}&{14}&{16}\\ {4}&{6}&{13}&{20}&{22}\\ {10}&{12}&{19}&{21}&{3}\\ {11}&{18}&{25}&{2}&{9}\end{array}}\right]
$$ 
of order 5, was constructed by using de la Loubere's method. Methods for constructing magic squares of even orders different from 2 and other methods for constructing magic squares of odd order can be found in a book by Rouse Ball. 

Three-dimensional analogs of magic squares have been considered. A magic cube of order n is an n-by-n-by-n cubical array constructed out of the integers $1,2,\dots, n^3$ in such a way that the sum s of the integers in the n cells of each of the following straight lines is the same:

(1) lines parallel to an edge of the cube;

(2) the two diagonals of each plane cross section;

(3) the four space diagonals.

> 三维的 n 阶幻方：形状为 n-n-n，包含了数字 $1,\dots, n^3$ ，满足
> 1. 和边平行的线的和等于幻和
> 2. 每个平面截面上的两个对角线上的和等于幻和
> 3. 四个空间对角线上的和等于幻和

The number s is called the magic sum of the magic cube and has the value $(n^{4}+ n) / 2$. 
(

$$
\begin{align}
1 + 2 + \dots + n^{3}&= \frac {(1+n^3)n^3}{2}\\
&=n^{2}s\\
s&=\frac {(1+n^3)n^3}{2n^2}\\
&=\frac {(n^4+n)}{2}
\end{align}
$$

)


We leave it as an easy exercise to show that there is no magic cube of order 2, and we verify that there is no magic cube of order 3.
> 我们验证不存在阶为 3 的三维幻方

Suppose that there is a magic cube of order 3. Its magic sum would then be 42. Consider any 3-by-3 plane cross section

$$
\left[\begin{array}{l l l}{a}&{b}&{c}\\ {x}&{y}&{z}\\ {d}&{e}&{f}\end{array}\right],
$$ 
with numbers as shown. Since the cube is magic. 

$$
\begin{array}{r c l}{a+y+f}&{=}&{42}\\ {b+y+e}&{=}&{42}\\ {c+y+d}&{=}&{42}\\ {a+b+c}&{=}&{42}\\ {d+e+f}&{=}&{42.}\end{array}
$$ 
Subtracting the sum of the last two equations from the sum of the first three, we get 3y = 42 and, hence, y = 14. But this means that 14 has to be the center of each plane cross section of the magic cube and, thus, would have to occupy seven different places. But it can occupy only one place, and we conclude that there is no magic cube of order 3. It is more difficult to show that there is no magic cube of order 4. A magic cube of order 8 is given in an article by Gardner.
> y = 14 必须是各个平面截面中心的元素，因此会占据 7 个不同的位置
> 但根据定义， 14只能占据1个位置，因此推出不存在阶为3的幻立方

Although magic squares continue to interest mathematicians, we will not study them further in this book
## 1.3 Example: The Four-Color Problem 
Consider a map on a plane or on the surface of a sphere where the countries are connected regions. To differentiate countries quickly, we must color them so that two countries that have a common boundary receive different colors (a corner does not count as a common boundary). What is the smallest number of colors necessary to guarantee that every map can be so colored? 
> 相互邻接的区域颜色需要不同，则如果要让任意一张地图都以该方式被上色，则需要的最少的颜色数量是多少

Some maps require four colors. That's easy to see. An example is the map in Figure 1.6. Since each pair of the four countries of this map has a common boundary, it is clear that four colors are necessary to color the map. 

It was proven by Heawood in 1890 that five colors are always enough to color any map. We give a proof of this fact in Chapter 12. It is not too difficult to show that it is impossible to have a map in the plane which has five countries, every pair of which has a boundary in common. Such a map, if it had existed, would have required five colors. But not having .five countries every two of which have a common boundary does not mean that four colors suffice. It might be that some map in the plane requires five colors for other more subtle reasons.
> 已证明出5个颜色足以涂色任意的一张地图
> 对于一个平面，它不可能有一张地图，地图中存在五个区域，其中的每一对都相互邻接，如果这样的地图存在，它就需要5个颜色

Now there are proofs that every planar map can be colored using only four colors, but they require extensive computer calculation.
> 已证明出任意的平面地图仅需要4中颜色就可以涂色
## 1.4 Example: The Problem of the 36 Officers 
Given 36 officers of 6 ranks and from 6 regiments, can they be arranged in a 6-by- 6 formation so that in each row and column there is one officer of each rank and one officer from each regiment? This problem, which was posed in the eighteenth century by the Swiss mathematician L. Euler as a problem in recreational mathematics, has important repercussions in statistics, especially in the design of experiments (see Chapter 10). 
> 每一行每一列都有 6 个不同的 rank 和 regiment

An officer can be designated by an ordered pair (i,j), where i denotes his rank (i = 1,2, ... ,6) and j denotes his regiment (j = 1,2, ... ,6). Thus, the problem asks the following question:
 Can the 36 ordered pairs (i, j) (i = 1,2, ... ,6; j = 1,2, ... ,6) be arranged in a 6-by-6 array so that in each row and each column the integers 1,2, ... ,6 occur in some order in the first positions and in some order in the second positions of the ordered pairs?

Such an array can be split into two 6-by-6 arrays, one corresponding to the first positions of the ordered pairs (the rank array) and the other to the second positions (the regiment array). Thus, the problem can be stated as follows:
 Do there exist two 6-by-6 arrays whose entries are taken from the integers 1,2, ... , 6 such that
 (1) in each row and in each column of these arrays the integers 1,2, ... ,6 occur in some order, and
 (2) when the two arrays are juxtaposed, all of the 36 ordered pairs (i, j) (i = 1,2, ... ,6;j = 1,2, ... ,6) occur?
> 问题重新表示为：
> 是否存在两个包含了元素 $1,\dots, 6$ 的 $6\times 6$ 的矩阵，满足
> 1. 各个矩阵中，每一行和每一列都会出现6个不同的数字，即是阶为6的拉丁方块
> 2. 两个矩阵放在一起时，36个有序对 $(i,j),\ (i=1,\dots, 6;j=1,\dots, 6)$ 都会出现，即两个拉丁方块正交

To make this concrete, suppose instead that there are 9 officers of 3 ranks and from 3 different regiments. Then a solution for the problem in this case is

$$
\begin{array}{r}{\left[\begin{array}{c c c}{1}&{2}&{3}\\ {3}&{1}&{2}\\ {2}&{3}&{1}\end{array}\right]\,,\ \ \ \ \left[\begin{array}{c c c}{1}&{2}&{3}\\ {2}&{3}&{1}\\ {3}&{1}&{2}\end{array}\right]\ \ \ \longrightarrow\ \ \ \left[\begin{array}{c c c}{(1,1)}&{(2,2)}&{(3,3)}\\ {(3,2)}&{(1,3)}&{(2,1)}\\ {(2,3)}&{(3,1)}&{(1,2)}\end{array}\right]\,.}\end{array}\tag{1.2}
$$ 
The preceding rank and regiment arrays are examples of Latin squares of order 3; each of the integers 1, 2, and 3 occurs once in each row and once in each column. 

The following are Latin squares of orders 2 and 4

$$
\left[\begin{array}{l l}{1}&{2}\\ {2}&{1}\end{array}\right]\mathrm{~and~}\left[\begin{array}{l l l l}{1}&{2}&{3}&{4}\\ {4}&{1}&{2}&{3}\\ {3}&{4}&{1}&{2}\\ {2}&{3}&{4}&{1}\end{array}\right].\tag{1.3}
$$ 
The two Latin squares of order 3 in (1.2) are called orthogonal because when they are juxtaposed, all of the 9 possible ordered pairs (i, j), with i = 1,2,3 and j = 1,2,3, result. We can thus rephrase Euler's question:
 Do there exist two orthogonal Latin squares of order 6?

Euler investigated the more general problem of orthogonal Latin squares of order n. It is easy to see that there is no pair of orthogonal Latin squares of order 2, since, besides the Latin square of order 2 given in (1.3), the only other one is

$$
\left[{\begin{array}{l l}{2}&{1}\\ {1}&{2}\end{array}}\right],
$$ 
and these are not orthogonal. Euler showed how to construct a pair of orthogonal Latin squares of order n whenever n is odd or has 4 as a factor. 
> 欧拉展示了当 n 是奇数或者以 4 为因子的时候，如何构造一对正交的 n 阶拉丁方块

Notice that this does not include n = 6. On the basis of many trials he concluded, but did not prove, that there is no pair of orthogonal Latin squares of order 6, and he conjectured that no such pair existed for any of integers 6, 10, 14, 18, ... ,4k + 2, . .. . By exhaustive enumeration, Tarrys in 1901 proved that Euler's conjecture was true for n = 6. Around 1960, three mathematician-statisticians, R. C. Bose, E. T. Parker, and S. S. Shrikhande, succeeded in proving that Euler's conjecture was false for all n > 6. 
That is, they showed how to construct a pair of orthogonal Latin squares of order n for every n of the form 4k +2, k = 2,3,4, .... This was a major achievement and put Euler's conjecture to rest. 
> 已经证明，除了 6 以外，对于任意 $n = 4k+2, k=2,3,4,\dots$ 阶的拉丁方，存在一对相互正交的拉丁方

Later we shall explore how to construct orthogonal Latin squares using finite number systems called finite fields and how they can be applied in experimental design.
## 1.5 Example: Shortest-Route Problem 
Consider a system of streets and intersections. A person wishes to travel from one intersection $A$ to another intersection $B$ .
In general, there are many available routes from $A$ to $B$ .The problem is to determine a route for which the distance traveled is as small as possible, a shortest route. 
> 最短路径问题是一个组合优化问题

This is an example of a combinatorial optimization problem. One possible way to solve this problem is to list in a systematic way all possible routes from $A$ to $B$ .It is not necessary to travel over any street more than once; thus, there is only a finite number of such routes. Then compute the distance traveled for each and select a shortestroute. This is not avery efficient procedure and, when the system is large, the amount of work may be too great to permit a solution in a reasonable amount of time. 

What is needed is an algorithm for determining a shortest route in which the work involved in carrying out the algorithm does not increase too rapidly as the system increases in size. In other words, the amount of work should be bounded by a polynomial function (as opposed to, say, an exponential function) of the size of the problem. In Section11.7 we describe such an algorithm This algorithm will actually find a shortest route from $A$ to every other intersection
> 需要复杂度和系统增长呈多项式关系的算法来求解最短路径

The problem of finding a shortest route between two intersections can be viewed abstractly. 
Let V be a finite set of objects called vertices (which correspond to the intersections and the ends of dead-end streets), and let E be a set of unordered pairs of vertices called edges (which correspond to the streets). Thus, some pairs of vertices are joined by edges, while others are not. The pair (V, E) is called a graph. 
A walk in the graph joining vertices x and y is a sequence of vertices such that the first vertex is x and the last vertex is y, and any two consecutive vertices are joined by an edge. Now associate with each edge a nonnegative real number, the length of the edge. The length of a walk is the sum of the lengths of the edges that join consecutive vertices of the walk. 

Given two vertices x and y, the shortest-route problem is to find a walk from x to y that has the smallest length. In the graph depicted in Figure 1.7, there are 6 vertices and 10 edges. The numbers on the edges denote their lengths. One walk joining x and y is x, a, b, d, y, and it has length 4. Another is x, b, d, y, and it has length 3. It is not difficult to see that the latter walk gives a shortest route joining x and y.

A graph is an example of a discrete structure which has been and continues to be extensively studied in combinatorics. The generality of the notion allows for its wide applicability in such diverse fields as psychology, sociology, chemistry, genetics, and communications science. Thus, the vertices of a graph might correspond to people, with two vertices joined by an edge if the corresponding people distrust each other; or the vertices might represent atoms, and the edges represent the bonds between atoms. You can probably imagine other ways in which graphs can be used to model phenomena. Some important concepts and properties of graphs are studied in Chapters 9, 11, and 12.
## 1.6 Example: Mutually Overlapping Circles 
Consider $n$ mutually overlapping circles $\gamma_{1},\gamma_{2},\ldots,\gamma_{n}$ in general position in the plane. 
By *mutually overlapping* we mean that each pair of the circles intersects in two distinct points (thus nonintersecting or tangent circles are not allowed). 
> 相互重叠的圆：每一对圆都在两个不同的点上相交

By *general position*, we mean that there do not exist three circles with a common point. The $n$ circles create a number of regions in the plane. The problem is to determine how many regions are so created. 
> 不存在三个具有共同点的圆
> 问题是探究 n 个圆创建了几个区域

Let $h_{n}$ equal the number of regions created. We easily compute that $h_{1}=2$ (the inside and outside of the circle $\gamma_{1}$ )， $h_{2}\,=\,4$ (the usual Venn diagram for two sets), and $h_{3}=8$ (the usual Ven n diagram for three sets). Since the numbers seem to be doubling, it is tempting now to think that $h_{4}=16$ However, a picture quickly reveals that $h_{4}=14$ (see Figure 1.8). 

One way to solve counting problems of this sort is to try to determine the change in the number of regions that occurs when we go from $n-1$ circles $\gamma_{1},\dots,\gamma_{n-1}$ to $\mathscr{n}$ circles $\gamma_{1},\dots,\gamma_{n-1},\gamma_{n}$ . In more formal language, we try to determine a recurrence relation for $h_{n}$ ；that is, express $h_{n}$ in terms of previous values. 
> 我们希望找到 $h_n$ 的递归关系，即 $h_n$ 和 $h_{1},\dots ,h_{n-1}$ 之间有什么关系

So assume that $n\geq2$ and that the $n-1$ mutually overlapping circles $\gamma_{1},\dots,\gamma_{n-1}$ have been drawn in the plane in general position creating $h_{n-1}$ regions. 
Then put in the nth circle $\gamma_{n}$ so that there are now $\mathscr{n}$ mutually overlapping circles in general position. Each of the first $n-1$ circles intersects the nth circle $\gamma_{n}$ in two points, and since the circles are in general position we obtain $2(n-1)$ distinct points $P_{1},P_{2},\dots,P_{2(n-1)}$ 
> 在平面上放上的第 n 个圆和之前的 n-1 个圆产生了 $2 (n-1)$ 个独立的交点

These $2(n-1)$ points divide $\gamma_{n}$ into $2(n-1)$ arcs: the arc between $P_{1}$ and $P_{2}$ ,the arc between $P_{2}$ and $P_{3},\,.\,.$ , the arc between $P_{2(n-1)-1}$ and $P_{2(n-1)}$ , and the arc between $P_{2(n-1)}$ and $P_{1}$ . Each of these $2(n-1)$ arcs divides a region formed by the first $n-1$ circles $\gamma_{1},\dots,\gamma_{n-1}$ into two, creating $2(n-1)$ more regions. Thus, $h_{n}$ satisfies the relation 
> 这些点将圆 n 划分为了 $2 (n-1)$ 个弧，每一段弧都将之前的 n-1 个圆划分出的各个区域分为两个区域，故创造出了 $2 (n-1)$ 个新区域，因此递推公式写为：

$$
h_{n}=h_{n-1}+2(n-1),\qquad(n\geq2).\tag{1.4}
$$ 
We can use the recurrence relation (1.4) to obtain a formula for $h_{n}$ in terms of the parameter $n$ .By iterating (1.4), we obtain 

$$
\begin{array}{r c l}

{h_{n}}&{=}&{h_{n-1}+2(n-1)}\\
{{h_{n}}}&{{=}}&{{h_{n-2}+2(n-2)+2(n-1)}}\\ {{h_{n}}}&{{=}}&{{h_{n-3}+2(n-3)+2(n-2)+2(n-1)}}\\ {{}}&{{\vdots}}&{{}}\\ {{h_{n}}}&{{=}}&{{h_{1}+2(1)+2(2)+\cdot\cdot\cdot+2(n-2)+2(n-1).}}\end{array}
$$ 
Since $h_{1}=2$ ,and $1+2+\cdot\cdot\cdot+(n-1)=n(n-1)/2$ ,we get 

$$
h_{n}=2+2\cdot{\frac{n(n-1)}{2}}=n^{2}-n+2,\qquad(n\geq2).
$$ 
This formula is also valid for $n=1$ ,since $h_{1}=2$ . A formal proof of this formula can now be given using mathematical induction. 
## 1.7 Example: The Game of Nim 
We close this introductory chapter by returning to the roots of combinatorics in recreational mathematics and investigating the ancient game of Nim. Its solution depends on parity, an important problem-solving concept in combinatorics. We used a simple parity argument in investigating perfect covers of chessboards when we showed that a board had to have an even number of squares to have a perfect cover with dominoes. 

Nim is a game played by two players with heaps of coins (or stones or beans). Suppose that there are $k\ge 1$ heaps of coins that contain, respectively, $n_1,\dots , n_k$ coins. The object of the game is to select the last coin. The rules of the game are as follows:

(1) The players alternate turns (let us call the player who makes the first move I and then call the other player II).

(2) Each player, when it is his or her turn, selects one of the heaps and removes at least one of the coins from the selected heap. (The player may take all of the coins from the selected heap, thereby leaving an empty heap, which is now "out of play.")

The game ends when all the heaps are empty. The last player to make a move-that is, the player who takes the last coin(s)-is the winner.

The variables in this game are the number k of heaps and the numbers $n_1,\dots, n_k$ of coins in the heaps. The combinatorial problem is to determine whether the first or second player wins and how that player should move in order to guarantee a win-a winning strategy.

Todevelop some understanding of $\mathrm{binom}$ ，we consider some special cases. 15Ifthere is initially only one heap, then player I wins by removing all the coins. Now suppose thatthere are $k=2$ heaps, with $_{n_{1}}$ and $n_{2}$ coins, respectively. Whether or not player I canwin dependsnot on the actualvalues of $n_{1}$ and $n_{2}$ but on whether or not they areequal. Supposethat $n_{1}\neq n_{2}$ .Player I can remove enough coins from the larger heap in order to leave two heaps of equal size for player II. Now player I, when it is her turn, can mimic player II's moves. Thus if player II takes $c$ coins from one of the heaps, then player I takes the same number $c$ of coins from the other heap. Such a strategy guarantees a win for player I. If $n_{1}=n_{2}$ ,thenplayer $\mathrm{II}$ can win by mimicking player I's moves. Thus, we have completely solved 2-heap Nim. An example of play in the2-heap game of Nim with heaps of sizes8and5, respectively, is 
> 对于两堆的 Nim 游戏，一个保证赢的策略就是每次轮到自己取，都让两队数量保持一致

$$
8,5\stackrel{\mathrm{~I~}}{\to}5,5\stackrel{\mathrm{II}}{\to}5,2\stackrel{\mathrm{~I~}}{\to}2,2\stackrel{\mathrm{II}}{\to}0,2\stackrel{\mathrm{~I~}}{\to}0,0.
$$ 
The preceding idea in solving 2-heap Nim, namely, moving in such a way as to leave two equal heaps, can be generalized toanynumber $k$ ofheaps. Theinsightone needs is provided by the conceptof the base 2 numeral of aninteger. Recallthateach positiveinteger $n$ can be expressed as a base 2 numeral by repeatedly removing the largestpower of 2 which does not exceed the number. For instance, toexpressthe decimal number 57 in base 2, weobservethat 

$$
\begin{array}{r l}&{2^{5}\leq57<2^{6},\quad57-2^{5}=25}\\ &{2^{4}\leq25<2^{5},\quad25-2^{4}=9}\\ &{2^{3}\leq9<2^{4},\quad9-2^{3}=1}\\ &{2^{0}\leq1<2^{1},\quad1-2^{0}=0.}\end{array}
$$ 
Thus, 

$$
57=2^{5}+2^{4}+2^{3}+2^{0},
$$ 
andthebase2numeralfor57is 111001.
> 将一个数转化为2进制数，也就是将一个 heap 划分为多个数量是 2 的指数幂的 subheap

Each digit in a base 2 numeral is either 0 or 1. The digit in the ith position, the one correspondingto $2^{i}$ ,is called the ith bit16 $(i\geq0)$ .We can think of each heap of coins as consistingof subheaps ofpowersof 2, according to its base numeral. Thus aheapof size 53 consists of subheaps ofsizes $2^{5},2^{4},2^{2}$ ，and $2^{0}$ .

In thecase of 2-heapNim, the total number of subheaps of each size is either 0,1, or 2. There is exactly one subheap of a particular size if and only if the two heaps have different sizes. Put another way, the total number of subheaps ofeachsizeisevenifandonlyifthetwoheapshavethe same size-—thatis, if and only if playerII canwin theNimgame. 
> 对于2 heap Nim，当且仅当两堆的数量不相同，会存在正好一个特定大小的 subheap
> 换句话说，各个大小的 subheap 的总数当且仅当两堆大小相同时是偶数，即当且仅当 player 2可以赢时，是偶数

Now consider a general Nim game with heaps of sizes $n_{1},n_{2},\ldots,n_{k}$ .Express each of thenumbers $n_{i}$ as base 2 numerals: 

$$
\begin{array}{r c l}{n_{1}}&{=}&{a_{s}\cdot\cdot\cdot a_{1}a_{0}}\\ {n_{2}}&{=}&{b_{s}\cdot\cdot\cdot b_{1}b_{0}}\\ &{\cdot\cdot\cdot}\\ {n_{k}}&{=}&{e_{s}\cdot\cdot\cdot e_{1}e_{0}.}\end{array}
$$ 
(By including leading 0s, we can assume that all of the heap sizes have base 2 numerals with the same number of digits.）We call aNim game balanced, providedthatthe number of subheapsofeachsizeiseven. Thus, aNimgameisbalancedifandonlyif 

$$
\begin{array}{c}{a_{s}+b_{s}+\cdot\cdot\cdot+e_{s}\mathrm{~is~even,}}\\ {\vdots}\\ {a_{i}+b_{i}+\cdot\cdot\cdot+e_{i}\mathrm{~is~even,}}\\ {\vdots}\\ {a_{0}+b_{0}+\cdot\cdot\cdot+e_{0}\mathrm{~is~even.}}\end{array}
$$ 
> 当且仅当各个大小的 subheap 的数量时偶数，我们称 Nim 是平衡的

A Nim game that isnotbalanced is called unbalanced. We saythattheith bit is balanced provided that the sum $a_{i}+b_{i}+\cdot\cdot\cdot+e_{i}$ iseven, and is unbalanced otherwise. Thus, a balanced game is one inwhich all bits arebalanced, while an unbalanced game isoneinwhichthereisatleastoneunbalancedbit. 
> 当且仅当第 i bit 的和是偶数，称第 i bit 是平衡的
> 平衡的 Nim 就是所有的 bit 都是平衡的
> 有一 bit 不平衡，Nim 就是不平衡的

We then have the following: 
 PlayerI can win in unbalanced Nim games,and player II can win in bal ancedNimgames. 
> 不平衡的 Nim 中，player 1 赢，平衡的 Nim 中，player 2 赢

To see this, we generalize the strategies used in 2-heapNim. Suppose theNim game is unbalanced. Let the largest unbalanced bit be the $j$ th bit. Then player I movesin such a way as to leave a balanced game for player II. She does this by selecting a heap whose jth bit is 1 and removing a number of coins from it so that the resulting game is balanced (see alsoExercise 32). No matter what player II does, she leaves for player I an unbalanced game again, and playerI once again balances it. Continuing like this ensuresplayer Iawin. If thegame starts outbalanced, thenplayer I'sfirstmove unbalancesit, and now player II adopts the strategy of balancing the game whenever itishermove. 
> 假设一个不平衡的 Nim，player 1找到最大的不平衡的 bit，然后移除一定数量，Nim 平衡，player 2之后一定会让 Nim 再次不平衡
> 依次往复，最后会是 player1赢
> 如果 Nim 开始平衡，player 2只需要总是想办法平衡 Nim 即可
# 2 Permutations and Combinations 
Most readers of this book will have had some experience with simple counting pro b- lems, so the concepts“permutation”and“combination”are probably familiar. But the experienced counter knows that even rather simple-looking problems can posed if- fi cul ties in their solutions. While it is generally true that in order to learn mathematics one must do mathematics, it is especially so here-the serious student should attempt to solve a large number of problems. 

In this chapter, we explorefour general principles and some of the counting formu- lasthat they imply. Each of these principlesgives a complementary principle, which we alsodiscuss. We conclude with an application of counting tofinite probability. 
> 本章介绍四个基本的原则
## 2.1 Four Basic Counting Principles 
The first principlel is very basic. It is one formulation of the principle that the whole is equal to the sum of its parts. 
> 第一个原则：全部等于部分的总和

Let $S$ be a set. A partition of $S$ is a collection $S_{1},S_{2},\ldots,S_{m}$ of subsets of $S$ such that each element of $S$ isinexactlyone ofthose subsets: 

$$
\begin{array}{r}{S=S_{1}\cup S_{2}\cup\cdot\cdot\cdot\cup S_{m},}\\ {\quad}\\ {S_{i}\cap S_{j}=\emptyset,\quad(i\neq j).}\end{array}
$$ 
Thus, thesets $S_{1},S_{2},\ldots,S_{m}$ are pairwise disjoint sets, and their union is $S$ .
> 集合的划分：互补相交，并集是全集

The subsets $S_{1},S_{2},\ldots,S_{m}$ are called the parts of the partition. We note that by this definition a part of a partition may be empty, but usually there is no advantage in considering partitions with one or more empty parts. 

The number of objects of a set $S$ isdenoted by $|S|$ and is sometimes called the size of $S$ 


**Addition Principle.**
Suppose thataset $S$ ispartitionedintopairwisedisjointparts $S_{1},S_{2},\ldots,S_{m}$ .The number of objects in $S$ can be determined by finding the number of objects in each of the parts, and adding the numbers so obtained: 

$$
|S|=|S_{1}|+|S_{2}|+\cdot\cdot\cdot+|S_{m}|.
$$ 
If thesets $S_{1},S_{2},\ldots,S_{m}$ areallowedto overlap, then a more profound principle, the inclusion-exclusion principle of Chapter 6, can be used to count the number of objects in $S$ 
> 集合的划分的所有 parts 的大小加起来等于集合的大小

In applying the addition principle, we usually define the parts descriptively. In otherwords, we breakup the problem into mutually exclusive cases that exhaust all possibilities. The art of applying the addition principle is to partition the set $S$ tobe countedinto“manageableparts"—thatis, partswhichwe canreadily count. Butthis statement needs to be qualified. If we partition $S$ into too many parts, then we may $S$ ha vc defeated ourselves. Forinstance, ifwepartition into parts each containing only oneelement, then applying the addition principle is the same as counting the number ofparts, and this is basically the same as listing all the objects of $S$ .Thus, amore appropriate description is that *the art of applying the ad*dition principle is to partition the setS into not too many manageable parts. 
> 加法原则的应用就是将全集划分为我们可以计数的各个 part，


Example.
Suppose we wish to find the number of different courses offered by the University of Wisconsin-Madison.We partition the courses according to the depart- ment in which they are listed.Provided there is no cross-listing (cross-listing occurs when the same course is listed by more than one department)，the number of courses offered by the University equals the sum ofthenumber·of courses offered by each department.口 

Another formulation of the addition principle in terms of choices is the following: *If an object can be selected from one pile in p ways and an object can be selected from a separate pilein $q$ ways,then the selection of one object chosen from either of the two piles can be made in $p+q$ ways.This formulation has an obvious generalization to more than two piles*
> 另一种加法原则的形式：
> 如果一个对象可以从一个 pile 以 p 中方式被选出，并且可以在另一个 pile 中以 q 中方式被选出，则从这两个 pile 选择出这个对象的方式就是 p+q 种

Example.
A student wishes to take either amathematics course orabiology course, butnotboth.If there are four mathematics courses and three biology courses for which the student has the necessary prerequisites,thenthe student can choose a course to takein $\mathbf{4}+\mathbf{3}=\mathbf{7}$ wa.ys

The second principle is alittlemore complicated.We state it for two sets,but it can also be generalized to any finite number of sets. 

**Multiplication Principle.**
Let $S$ beasetoforderedpairs $(a,b)$ ofobjects, wherethe first object a comes from a set of size $p$ ，and for each choice of object a there are $q$ choices for object $^{b}$ .Then the size of $S$ is $p\times q$ 

$$
|S|=p\times q.
$$ 
> 乘法原则：令 $S$ 是有序对 $(a, b)$ 的集合，其中第一个对象来自于一个大小为 p 的集合，对于第一个对象的每一个选择，第二个对象都有相应的 q 种选择，则集合 $S$ 的大小是 $p\times q$

The multiplication principle is actually a consequence of the addition principle. Let $a_{1},a_{2},\dotsc,a_{p}$ be the $_p$ different choices for the object $^{a}$ ：We partition $S$ into parts $S_{1},S_{2},\ldots,S_{p}$ where $S_{i}$ is the set of ordered pairs in $S$ with first object $a_{i}$ $(i=1,2,\dots,p)$ .The size of each $S_{i}$ is $q$ ；hence, by the addition principle 

$$
\begin{array}{r c l}{{|S|}}&{{=}}&{{|S_{1}|+|S_{2}|+\cdot\cdot\cdot+|S_{p}|}}\\ {{}}&{{=}}&{{q+q+\cdot\cdot\cdot+q\;\;\;(p\;q\cdot{\bf s})}}\\ {{}}&{{=}}&{{p\times q.}}\end{array}
$$ 
> 乘法原则可以由加法原则推导

Note how the basic fact—-multiplication of whole numbers is just repeated addition-- enters into the preceding derivation 

A second useful formulation of the multiplication principle is as follows: If the first taskhas $_p$ outcomes and, no matter what the outcome of thefirsttask, asecondtask has $q$ outcomes, then the two tasks performed consecutively have $p\times q$ outcomes. 
> 乘法原则的另一种形式：
> 如果第一个任务可以有 p 种结果，且无论其结果如何，第二个任务可以有 q 种结果，则一共有 $p\times q$ 种结果

Example.
A student is to take two courses.The first meets at any one of 3 hours in the morning, and the second at any one of 4 hours in the afternoon.The number of schedules that are possible for the student is $3\times4=12$ 

As already remarked,the multiplication principle can be generalized to three, four, or any finite number of sets. Rather than formulate it in terms of $n$ sets，wegive examplesfor $n=3$ and $n=4$ 

Example.
Chalk comes inthree different lengths,eight different colors，and four different diameters.How many different kinds of chalk are there? 

To determine a piece of chalk of a specific type, we carry out three different tasks (it does not matter in which order we take these tasks):Choose a length,Choose a color,Choose a diameter.By the multiplication principle,there are $3\times8\times4=96$ different kinds of chalk. 

Example.
The number of ways a man,woman,boy,and girl can be selected from five men,six women,two boys,and four girls is $5\times6\times2\times4=240$ 

The reason is that we have four different tasks to carry out: select a man (five ways),select a woman (six ways),select a boy (two ways),select a girl (four ways). If, in addition, we ask for the number of ways one person can be selected, the answer is $5+6+2+4=17$ .This follows from the addition principle for four piles.

Example.
Determine the number of positive integers that are factors of the number 

$$
3^{4}\times5^{2}\times11^{7}\times13^{8}.
$$ 
The numbers 3,5,11, and 13 are prime numbers. By the *fundamental theorem of arithmetic*,each factor is of the form 

$$
\mathbf{3}^{i}\times5^{j}\times11^{k}\times13^{l},
$$ 
where $0\leq i\leq4,\,0\leq j\leq2,\,0\leq k\leq7$ ，and $0\leq l\leq8$ ：There are five choices for $i$ threefor $j$ ，eightfor $k$ ,andninefor $l$ .By the multiplication principle,the number of factors is 

$$
5\times3\times8\times9=1080.
$$ 
In the multiplication principle the $q$ choices for object $b$ may vary with the choice of $^{a}$ .The only requirementis that therebe the same number $q$ ofchoices, notnecessarily thesamechoices. 
> 乘法原则中，第二个对象的 p 种选择可能会随着第一个对象的选择结果而改变，但唯一的要求就是数量仍保持 p 种，并不要求选择完全一样

Example.
How many two-digitnumbers have distinctand nonzerodigits? 

Atwo-digit number abc an be regarded as an ordered pair $(a,b)$ ，where $a$ isthe tensdigitand $^{b}$ istheunitsdigit. NeitherofthesedigitsisallowedtobeOinthe problem, and the two digits are to be different. There are nine choices for $^{a}$ ,namely $1,2,\ldots,9$ .Once $a$ ischosen, there are eight choices for $b$ .If $a=1$ ，theseeightchoices are $2,3,\ldots,9$ if $a=2$ ，the eight choices are $1,3,\ldots,9$ ,and so on. What is important for application of the multiplication principle is that the number of choices is always 8. The answer to the questions is, by the multiplication principle, $9\times8=72$ 

We can arrive at the answer 72 in another way. There are 90 two-digit numbers, $10,11,12,\ldots,99$ .Of these numbers, nine have a O, (namely, $10,20,\ldots,\,90)$ andnine have identical digits (namely, $11,22,\ldots,99)$ .Thus the number of two-digit numbers withdistinctandnonzerodigitsequals $\mathtt{90-9-9=72}$

The preceding example illustrates two ideas.One is that there maybe more than one way to arrive at the answer to a counting question.The other idea is that to find the number of objects in a set $A$ (in this case the set of two-digit numbers with distinct and non zero digits)it may beeasierto findthenumber ofobjectsinalargerset $U$ containing $S$ (thesetofalltwo-digit numbers in the preceding example)andthen subtract the number of objects of $U$ that do not belong to $A$ (thetwo-digitnumbers containing a 0 oridentical digits).Weformulate this idea as our thirdprinciple 

**Subtraction Principle.** 
Let $A$ bea setandlet $U$ be a larger set containing $A$ .Let 

$$
\overline{{A}}=U\setminus A=\{x\in U:x\notin A\}
$$ 
be the complement of $A$ in $U$ .
Then the number $|A|$ of objectsin $A$ isgivenbythe rule 

$$
|A|=|U|-|{\overline{{A}}}|.
$$ 

> 减法原则：一个集合的大小可以通过一个它的超集的大小减去它对应的补集的大小得到

In applying the subtraction principle, theset $U$ is usually some naturals etc on- si sting of all the objects under discussion (theso-called universal set). Usingthe subtraction principle makes sense onlyif itiseasier to count the number of objects in $U$ $\overline{{A}}$ $A$ andin than to count the number of objects in A
> 一般来说，超集会直接选择为全集


Example.
Computer passwords are to consist of a string of six symbols taken from thedigits $0,1,2,\ldots,9$ and the lowercase letters $a,b,c,\ldots,z$ .Howmanycomputer passwords have a repeated symbol? 

We want to count the number of objects in the set $A$ of computer passwords with a $U$ repeatedsymbol.Let be the set of all computer passwords,Taking the complement of $A$ in $U$ wegettheset $\overline{{A}}$ of computer passwords with norepeated symbol.By two applications of the multiplication principle,weget 

$$
\vert U\vert=36^{6}=2,176,782,336
$$ 
and 

$$
|{\overline{{A}}}|=36\cdot35\cdot34\cdot33\cdot32\cdot31=1,402,410,240.
$$ 
Therefore, 

$$
|A|=|U|-|\overline{{A}}|=2,176,782,336-1,402,410,240=774,372,096.
$$ 
We now formulate the final principle of this section. 

**Division Principle.**
Let $S$ be a finite set that is partitioned into $k$ partsinsucha way that each part contains the same number of objects. Then the number of parts in the partition is given by the rule 

$$
k={\frac{|S|}{\mathrm{number~of~object~in~a~part}}}.
$$ 
> 除法原则：
> 将一个有限集合划分为 k 个相同大小的 parts，则 parts 的数量 k 可以通过集合 $S$ 的大小除以各个 part 的大小得到

Thus,we can determine the number of parts if we know the number of objects in $S$ and the common value of the number of objects in theparts. 


Example.
There are 740 pigeons in a collection of pigeonholes.If each pigeonhole contains 5 pigeons,the number of pigeonholes equals 

$$
{\frac{740}{5}}=148.
$$ 
Moreprofound applications of the division principle will occurlater in this book. Now consider the next example 

Example.
You wish to give your Aunt Mollie a basket of fruit.In your refrigerator you have six oranges and nine apples.The only requirement is that there must be at least one piece of fruit in the basket(thatis,an empty basket of fruit is not allowed). How many different baskets of fruit are possible? 

One way to count the number of baskets is the following: First, ignorethere- quire ment that the basket cannot be empty. Wecan compensatefor thatlater. What distinguishes one basket of fruit from another is the number of oranges and number of apples in the basket. There are 7 choices for the number of oranges $(0,1,\ldots,6)$ and 10 choices for the number of apples $(0,1,\ldots,9)$ . By the multiplication principle, the number of different baskets is $7\times10=70$ Subtracting the empty basket, the answeris69. Notice that if we had not (temporarily) ignored the requirement that thebasketbe nonempty, then there would have been 9 or10 choices for the number of apples depending on whether ornotthenumber of orangeswasO, andwecould not have applied the multiplication principle directly. But an alternative solution is the following. Partition the nonempty baskets into two parts, $S_{1}$ and $S_{2}$ ，where $S_{1}$ consists of those baskets with no oranges and $S_{2}$ consists of those baskets with atleast one orange. The size of $S_{1}$ is9 $(1,2,\ldots,9$ apples) and the size of $S_{2}$ by theforegoing reasoningis $6\times10=60$ .The number of possible baskets of fruit is, bytheaddition principle,$9+60=69$

We made an implicit assumption in the preceding example which we should now bring into the open. It was assumed in the solution that the oranges were indistin- guis h able from one another (anorangeis an orange is an orange is...) andthatthe apples were indistinguishable from one another. Thus, what mattered in making up a basket of fruit was not which apples and which oranges went into it but only the number of each type of fruit. If we distinguished among the various oranges and the various apples (one orange is perfectly round, another is bruised, a third very juicy and soon), thenthenumber ofbasketswould belarger. Wewill returnto thisexample inSection3.5. 

Before continuing with more examples,we discuss some general ideas 
A great many counting problems can be classified as one of the following types: 

(1)Count the number of ordered arrangements or ordered selections of objects 
 (a) without repeating any object, 
 (b)with repetition of objects permitted (but perhaps limited) 

(2)Count the number of unordered arrangements or unordered selections of objects 
 (a) without repeating any object, 
 (b) with repetition of objects permitted (but perhaps limited) 

> 计数问题基本上可以分为以下几个类型：
> 1. 有序：不允许重复/允许重复
> 2. 无序：不允许重复/允许重复

Instead of distinguishing between non repetition and repetition of objects, itissome times more convenient to distinguish between selections from a set and a multiset A multiset is like a set except that its members neednot be distinct. For example wemighthave amultiset $M$ withthree $a$ 's, one $b$ ，two $c{\bf s}$ ，and four $d{\bf s}$ ，that is, 10 elements of 4 different types: 3oftype $^{a}$ ,1 of type $b$ ,2of type $c$ ，and4oftype $d$ We shall usually indicate a multiset by specifying thenumberof times different types of elements occur in it. Thus, $M$ shall be denoted by $\left\{3\cdot a,1\cdot b,2\cdot c,4\cdot d\right\}$ 3Thenumbers $3,1,2$ ,and4aretherepetitionnumbersofthemultiset $M$ .A set is a multiset that has allrepetitionnumbersequal to1.Toincludethelistedcase (b) when thereisnolimit onthenumber of timesanobjectof each typecanoccur (exceptfor thatimposed by the size of the arrangement), we allow infinite repetition numbers. Thus, a multiset in which $a$ and $c$ eachhave aninfiniterepetitionnumber and $b$ and $^{d}$ have repetition numbers2 and 4, respectively, isdenotedby $\{\infty\cdot a,2\cdot b,\infty\cdot c,4\cdot d\}$ .
> 有时可以将允许重复和不允许重复的区分以 multiset 和 set 的区分表示
> multiset 即允许成员重复的集合，一个集合就是所有成员的重复次数都是 1 的集合

Arrangements orselectionsin (1) in which order is taken into consideration are generally called permutations, whereas arrangements or selections in (2) in which order is irrelevant are generally called combinations. 
> 考虑顺序的排列或者选择一般称为 permutations/排列
> 顺序无关则称为 combinations/组合

In the next two sections we will develop some general formulas for the number of permutations and combinations of sets and multisets. But not all permutation and combination problems can be solved by using these formulas It is often necessary to return to the basic addition, multiplication, subtraction, and division principles 


Example.
How many odd numbers between 1000 and 9999 have distinct digits? 

A number between 1000 and 9999 is an ordered arrangement of four digits. Thus we are asked to count a certain collection of permutations. We have four choices to make: a units, a tens, a hundreds, and a thousands digit. Since the numbers we want tocountareodd，the units digit can be anyone of $1,3,5,7,9$ Thetensandthe hundreds digit can be anyone of $0,1,\ldots,9$ ，while thethousands digit canbe any one of $1,2,\ldots,9$ .Thus, there are five choices for the units digit. Since the digits are to bedistinct, we have eight choices for thethousands digit, whateverthe choice of the units digit. Then, there are eightchoices for thehundreds digit, whatever thefirst two choiceswere, and seven choices for the tens digit, whatever the first three choices were. Thus, by the multiplication principle, the answer to the question is $\mathbf{5\times8\times8\times7}=2240$

Suppose in the previous example we made the choices in a different order: First choose the thousands digit, thenthehundreds, tens, andunits. There are nine choices for the thousands digit, then nine choices for the hundreds digit (since we are allowed touseO), eight choices for the tens digit, but now the number of choices for the units digit (whichhastobeodd) depends on the previous choices. Ifwehadchosenno odddigits, the number of choices for the units digit would be 5; if we had chosen one odddigit, the number of choices for the units digit would be 4; andsoon. Thus, we cannot invoke the multiplication principle if we carryout our choices in the reverse order. There are two lessons to learn from this example. One is that as soon as your answer for the number of choices of one of the tasks is“itdepends" (or some such words), the multiplication principle cannot be applied. The second is that there may not be a fixed order in which the tasks have to betaken, and by changing the order a problem maybe more readily solved by the multiplication principle. Aruleofthumb to keep in mind is to make the most restrictive choice first. 
> 使用乘法原则时，顺序是很重要的，有时如果一个选择的数量依赖于前一个选择的具体值，乘法原则就无法进行
> 一个通用的指导是最先进行最受限制的选择

Example. How many integers between 0 and 10,000 have only one digit equal to 5? 

Let $S$ be the set of integers between 0 and 10,000 with only one digit equal to 5. 
First solution: We partition $S$ into the set $S_{1}$ of one-digit numbers in $S$ ，the set $S_{2}$ of two-digit numbers in $S$ ,the set $S_{3}$ of three-digit numbers in $S$ ,and the set $S_{4}$ of four-digit numbers in $S$ .There are nofive-digit numbersin $S$ .We clearly have 

$$
|S_{1}|=1.
$$ 
The numbers in $S_{2}$ naturally fall into two types:(1)the units digitis 5,and(2)the tens digitis 5.The number of the first type is 8（the tens digit cannot be 0 nor can it be 5).The number of the second type is 9 (the units digit cannot be 5).Hence, 

$$
\left|S_{2}\right|=8+9=17.
$$ 
Reasoning in a similar way,we obtain 

$$
|S_{3}|=8\times9+8\times9+9\times9=225
$$ 

$$
|S_{4}|=8\times9\times9+8\times9\times9+8\times9\times9+9\times9\times9=2673.
$$ 
Thus, 

$$
|S|=1+17+225+2673=2916.
$$ 
Second solution: By including leading zeros (e.g., think of 6 as 0006,25 as 0025,352 as 0352), we can regard each number in $S$ as a four-digit number. Now we partition $S$ intothesets $S_{1}^{\prime},S_{2}^{\prime},S_{3}^{\prime},S_{4}^{\prime}$ according to whether the 5 is in the first, second, third, or fourth position. Each of the four sets in the partition contains $9\times9\times9=729$ integers, and so the number of integers in $S$ equals 

$$
4\times729=2916.
$$ 

Example.How many different five-digit numbers can be constructed out of the digits 1, 1,1, 3,8? 

Here we are asked to count permutations of a multiset with three objects of one type,oneofanother,and one of a third.We really have only two choices to make: whichpositionis tobe occupied by the3(five choices)and then which position is to be occupied by the 8 (four choices). The remaining three places are occupied by 1s. By the multiplication principle,the answer is $5\times4=20$ 

If the five digits are 1,1,1,3,3,the answer is 10,half as many 

These examples clearly demonstrate that mastery of the addition and multi p lica tion principles is essential for becoming an expert counter. 
## 2.2 Permutations of Sets 
Let $\mathcal{T}$ be a positive integer. By an $\mathcal{T}$ -permutation of a set $S$ of $n$ elements,we understand an ordered arrangement of $\mathcal{T}$ of the $n$ elements. If $S\,=\,\{a,b,c\}$ ，then thethree1- permutations of $S$ are 
> 集合 $S$ 的 $n$ 个元素的 $r$ -排列表示的是对于 $n$ 个元素中的 $r$ 个元素的有序排列

$$
\begin{array}{l l l l l}{{\mathbfit{a}}}&{{}}&{{\mathbfit{b}}}&{{}}&{{\mathbfit{c},}}\end{array}
$$ 
the six 2-permutationsof $S$ are 

and the six 3-permutations of $S$ are 

Thereareno4-permutationsof $S$ since $S$ has fewer than four elements. 

We denote by $P(n,r)$ the number of $\mathscr{r}$ -permutationsof an $n$ -elementset.If $r>n$ then $P(n,r)=0$ .Clearly $P(n,1)=n$ for eachpositive integer $n$ .An $n$ -permutation of an $n$ -elementset $S$ will be more simply called a permutation of $S$ orapermutation of $\mathscr{n}$ elements. Thus,a permutation of a set $S$ can be thought of asalisting of the elementsof $S$ in some order.Previously we saw that $P(3,1)\,=\,3,P(3,2)\,=\,6$ and $P(3,3)=6$ 
> $P (n, r)$ 表示 n 个元素的 r 排列的数量

***Theorem 2.2.1***
For $n$ and $\boldsymbol{\mathscr{r}}$ positive integers with $r\leq n$ 

$$
P(n,r)=n\times(n-1)\times\cdots\times(n-r+1).
$$

> 定理：
> $n$ 个元素的 $r$ 排列的计算是：$P (n, r) = n\times (n-1)\times (n-r+1)$

**Proof.** In constructing an $\pmb{r}$ permutationofan $n$ -element set, wecan choose the first item in $n$ ways, the second item in $_{n-1}$ ways, whatever the choice of the first item, . · ·, and the $\pmb{r}$ th item in $n\!-\!\left(r\!-\!1\right)$ ways, whatever the choice of the first $r\!-\!1$ items. By the multiplication principle the $\pmb{r}$ itemscanbechosenin $n\times\left(n-1\right)\times\cdot\cdot\times\left(n-r+1\right)$ ways.
> 证明：
> 构造 $r$ 排布时，第一个元素有 $n$ 种选择，第二个元素有 $n-1$ 种选择，以此类推，最后一个元素有 $n-(r-1)$ 种选择，根据乘法原理，这 $r$ 个项的总的选择数量就是将它们乘起来

For a nonnegative integer $n$ ,we define $n!$ (read $n$ factorial) by 

$$
n!=n\times\left(n-1\right)\times\cdot\cdot\cdot\times2\times1,
$$ 
with the convention that $0!=1$ .We maythenwrite 

$$
P(n,r)={\frac{n!}{(n-r)!}}.
$$
> $P (n, r)$ 显然可以重写为 $n!/(n-r)!$

For $n\geq0$ ，wedefine $P(n,0)$ to be 1, and this agrees with the formula when $r=0$ 
> 我们定义 $n$ 个元素的 $0$ 排列是 1，也就是 $P (n, 0) = n!/n!=1$

The number of permutations of $n$ elementsis 

$$
P(n,n)={\frac{n!}{0!}}=n!.
$$ 
> $n$ 个元素的 $n$ 排列就是 $n!$


Example.
Thenumber of four-letter“words”thatcanbe formed byusingeach of theletters $a,b,c,d,e$ at most once is $P(5,4)$ ,and this equals $5!/(5-4)!=120.$ The numberoffive-letter words equals $P(5,5)$ , which is also 120.

Example.
Theso-called“15puzzle”consists of 15 sliding unit squares labeled with thenumbers1 through 15 and mounted in a 4-by-4 squareframeas shown inFigure 2.1.The challenge of the puzzle is to move from the initial position shown to any specified position.(That challenge is not the subject of this problem.)Byaposition, wemean an arrangement of the 15 numbered squares in theframewithone empty unit square.What is the number of positions in the puzzle(ignoring whether it is possible to move to the position from the initial one)? 

The problem is equivalent to determining the number of ways to assign the numbers $1,2,\dots,15$ to the 16 squares of a 4-by-4 grid,leaving one square empty.Since we can assign the number 16 to the empty square,the problem is also equivalent to determining thenumber ofassignments of thenumbers $1,2,\dots,16$ tothe16squares, andthisis $P(16,16)=16!$ 

Whatis the number of ways to assign the numbers $1,2,\ldots,15$ tothesquares of a6-by-6grid, leaving 21 squares empty? These assignments correspond tothe15- permutations of the 36 squares as follows: To an assignment of the numbers $1,2,\ldots,15$ to 15 of the squares, we associate the 15-permutation of the 36 squares obtained by putting the square labeled 1 first, the square labeled 2 second, andsoon. Hencethe totalnumberofassignmentsis $P(36,15)=36!/21!$

Example.
What is the number of ways to order the 26 letters of the alphabet so that no two of the vowels $a,e,i,o$ ,and $\boldsymbol{u}$ occur consecutively? 

The solution to this problem(like so many counting problems)is straightforward once we see how to do it.We think of two main tasks to be accomplished.The first task is to decide how to order the consonants among themselves.Thereare21 consonants,and so 21!permutations of the consonants.Since we cannot have two consecutive vowels in our final arrangement,the vowels must be in 5 of the 22spaces before,between,and after the consonants.Our second task is to put the vowels in theseplaces.There are 22 places for the $^{a}$ ，then21forthee,20forthei，19forthe $o$ ,and18forthe $\mathbfcal{U}$ ：Thatis,thesecondtaskcanbeaccomplishedin 

$$
P(22,5)={\frac{22!}{17!}}
$$ 
ways.
By the multiplication principle,we determine that the number of ordered ar- range ment s of the letters of the alphabet with no two vowels consecutive is 

$$
21!\times{\frac{22!}{17!}}.
$$ 
> 先排布好21个辅音字母，全排列 $21!$，此时21个辅音字母留出了22个空位用于排布元音字母，即22的5排列 $P (22,5)$


Example.
Howmanyseven-digitnumbersaretheresuchthatthedigitsaredis tinc t integers taken from $\{1,2,\ldots,9\}$ andsuchthatthedigits5and6donotappear consecutively in either order? 

Wewant to count certain 7-permutations of the set $\{1,2,\ldots,9\}$ ,andwepartition these 7-permutations into four types: (1) neither 5 nor 6 appears as a digit;(2) 5，but not 6, appears as a digit;(3) 6, but not 5, appears as a digit;(4) both 5 and 6 appear as digits.
> 先将排列分为4类：
> 5和6都没有出现: 7!
> 仅5出现: 7!
> 仅6出现: 7!
> 5和6都出现

The permutations of type(1)arethe7-permutations of $\{1,2,3,4,7,8,9\}$ ，andhence their number is $P(7,7)=7!=5040$ Thepermutations of type(2)can be counted as follows: Thedigitequal to 5 can be anyone of the seven digits.The remaining six digits area 6-permutationof $\{1,2,3,4,7,8,9\}$ .Hencethere are $7P(7,6)=7(7!)=35,280$ numbersoftype(2).In a similar way we see that there are35,280numbers of type(3).

To count thenumber of permutations of type(4),we partition the permutations of type(4)into three parts: 
> 此时仅需要考虑二者都出现的情况，我们将第四类的排布分为3种情况：
> 第一个数是5
> 最后一个数是5
> 5不是第一个也不是最后一个

**First digit equal to 5,and so second digit not equal to 6:** 

$$
\underline{{{\begin{array}{r l}{5}\end{array}}}}\neq\underline{{{\begin{array}{l}{6}\end{array}}}}\quad---\quad---\quad.
$$ 
There are five places for the6.The other five digits constitute a 5-permutationof the 7digits $\{1,2,3,4,7,8,9\}$ .Hence,thereare 

$$
5\times P(7,5)=\frac{5\times7!}{2!}=12,600
$$ 
numbers in this part. 


**Last digit equal to 5,and so next to last digit not equal to 6:** 
By an argument similar to the preceding,we conclude that there are also 12,600 numbers in this part. 


**A digit other than the first or last is equal to 5:** 
The place occupied by 5 is anyone of the five interior places.The place for the 6 can then be chosen in four ways.The remaining five digits constitute a 5-permutation of thesevendigits $\{1,2,3,4,7,8,9\}$ .Hence,thereare $5\times4\times P(7,5)=50,400$ numbers inthiscategory.Thus,thereare 

$$
2(12{,}600)\,\,+50{,}400{=}75{,}600
$$ 
numbers of types(4).

By the addition principle,the answer to the problemposed is 

$$
5040\,+2(35{,}280)\,+\!75{,}600=\!151{,}200.
$$ 
The solution just given was arrived at by partitioning the set of objects we wanted to count into manageable parts,parts the number of whose objects we could calculate, and thenusing the addition principle.
> 这种解法利用了加法原理


An alternative,and computationally easier, solutionis to use the subtraction principle as follows.Letus consider the entire collection $T$ ofseven-digitnumbers that can be formed by using distinct integers from $\{1,2,\ldots,9\}$ .Theset $T$ thencontains 

$$
\textstyle P(9,7)={\frac{9!}{2!}}=181,440
$$ 
numbers.Let $S$ consist of those numbersin $T$ in which 5 and 6 do not occur consecu- tively;so the complement $\overline{{S}}$ consists of those numbers in $T$ in which 5 and 6 do occur consecutively.We wish to determine the size of $S$ .Ifwecanfindthesizeof $\overline{{S}}$ ，then ourproblemis solved bythe subtraction principle.Howmanynumbers arethere in $\overline{{S}}?$ In $\overline{{S}}$ ,the digits 5 and 6 occur consecutively. There are six ways to position a 5 followedbya6,and six ways to position a 6 followed by a 5.The remaining digits constitutea5-permutationof $\{1,2,3,4,7,8,9\}$ .Sothenumberofnumbersin $\overline{{S}}$ is 

$$
2\times6\times P(7,5)=30,240.
$$ 
But then $S$ contains $181,440-30,240=151,200$ numbers. 
> 也可以利用减法原理求解

The permutations that we have just considered are more properly called *linear permutations*. We think of the objects asbeing arranged in a line. 
> 我们目前考虑的排列是线性排列，对象按照一行排列

If instead of arranging objects in a line, we arrange them in a circle, the number of permutations is smaller. Think of it this way: Suppose six children are marching in a circle. Inhow manydifferent ways can they form their circle? Since the children are moving, what matters are their positions relative to each other and not totheir environment. Thus, itisnatural to regard two circular permutations asbeing the sameprovidedone can be brought to the other by a rotation, thatis, by a circular shift. There are six linear permutations for each circular permutation. For example, the circular permutation arises from each of the linear permutations 

by regarding the lastdigitascomingbeforethe first digit. 
> 对于圆形排列，排列数量会大大减少，因为线性排列中不同的排列在圆形排列中是等价的（旋转）

Thus, thereisa6-to-1 correspondence between the linear permutations of six children and the circular per mutationsof the sixchildren. Therefore, tofind thenumberof circular permutations, we divide the number of linear permutationsby 6. Hence, thenumber of circular permutations of the six children equals ${\mathfrak{6!}}/{\mathfrak{6}}=5!$ 

***Theorem 2.2.2*** 
The number of circular $\mathscr{r}$ -permutations of a set of $n$ elements isgiven by 

$$
{\frac{P(n,r)}{r}}={\frac{n!}{r\cdot(n-r)!}}.
$$

In particular,the number of circular permutations of $n$ elementsis $(n-1)!$ 
> 定理：
> $n$ 个元素的圆形 $r$ 排列数量是：$P (n, r)/r = n!/(r\cdot (n-r)!)$
> 特别地，$n$ 个元素的圆形 $n$ 排列数量是：$(n-1)!$

**Proof.** A proof is essentially contained in the preceding paragraph and uses the divi- sionprinciple.Thesetoflinear $\pmb{r}$ -permutations can be partitioned into parts in such a way that twolinear $\pmb{r}$ permutations correspond to the same circular $\pmb{r}$ permutation if and only if they are in the samepart.Thus,the number of circular $\pmb{r}$ -permutations equals the number of parts.Since each part contains $\pmb{r}$ linear $\mathcal{T}$ -permutations,the number ofparts isgiven by 

$$
{\frac{P(n,r)}{r}}={\frac{n!}{r\cdot(n-r)!}}.
$$ 
> 使用除法原理进行证明，将线性的 $r$ 排列集合划分为多个大小相同的部分，每个部分内的线性排列在圆形排列下是等价的
> 此时圆形排列的数量就等于 parts 的数量，容易知道每个 part 包含 r 个线性排列，因此除以 r 即可

For emphasis,were mark that the preceding argument worked because each part containedthesamenumberof $\scriptstyle{\mathcal{T}}$ permutations so that we could apply the division principle to determine the number of parts.If,for example,we partition a set of 10 objects into parts of sizes 2,4, and 4, respectively, the number of parts cannot be obtained by dividing 10 by 2 or 4. 
> 强调一下只有 parts 大小相同才可以用除法原理

Another way to view the counting of circular permutations is the following: Supposewewishto count the number of circular permutations of $A,B,C,D,E$ ,and $F$ (the number of ways to seat $A,B,C,D,E$ and $F$ aroundatable).Sincewearefree to rotate the people,any circular permutation can be rotated so that $A$ isinafixed position;thinkofitas the“head"of the table: 

$$
\begin{array}{c c c c}{{}}&{{}}&{{A}}&{{}}\\ {{D}}&{{}}&{{}}&{{C}}\\ {{F}}&{{}}&{{}}&{{B}}\\ {{}}&{{}}&{{E}}&{{}}\end{array}
$$ 
Nowthat $A$ isfixed,the circular permutations of $A,B,C,D,E$ ,and $F$ canbeidentified with the linear permutations of $B,C,D,E$ ，and $F$ .(The preceding circular permut a- tion is identifiedwith thelinearpermutation $D F E B C$ ）Thereare5!linearpermuta- tionsof $B,C,D,E$ ，and $F$ ,andhence5!circular permutations of $A,B,C,D,E$ ,and $F$ 

This way of looking at circular permutations is also useful when the formula for circular permutations cannot be applied directly 
> 也可以将圆形全排列看作固定一个“头部”不变，剩余的元素做全排列


Example. 
Ten people, including two who do not wish to sit next to one another, are to be seated at around table.How many circular seating arrangements arethere? 
Wesolve this problemusing the subtraction principle.Let the 10 people be $P_{1},P_{2},P_{3},\dots,P_{10}$ ，where $P_{1}$ and $P_{2}$ are the two who do not wish to sittogether. Considers eating arrangements for 9 people $X,P_{3},\dots,P_{10}$ at around table.There are 8!such arrangements.Ifwereplace $X$ byeither $P_{1},P_{2}$ orby $P_{2},P_{1}$ ineachofthese arrangements,we obtaina seating arrangement for the 10 people in which $P_{1}$ and $P_{2}$ are next to one another.Hence using the subtraction principle,we see that the number of arrangements in which $P_{1}$ and $P_{2}$ are not together is ${\bf9!-2\times8!=7\times8!}$ 
> 解法1：减法原理

Another way to analyze this problem is the following:Firstseat $P_{1}$ atthe“head" ofthetable.Then $P_{2}$ cannot be on either side of $P_{1}$ .Thereare8choicesforthe person on $P_{1}$ 's left, 7 choices for the person on $P_{1}$ 'sright,and the remaining seats can be filled in 7! ways. Thus, the number of seating arrangements in which $P_{1}$ and $P_{2}$ are not together is 

$$
{\big.}8\times7\times7!=7\times8!.
$$ 
> 解法2：乘法原理

As we did before we discussed circular permutations,we will continue to use per- mutationtomean"linear permutation." 

Example. 
The number of ways to have 12 different markings on a rotating drum is $P(12,12)/12=11!$ 

Example.
What is the number of necklaces that can be made from 20 beads,eachof adifferentcolor? 

There are 20! permutations of the20beads. Since each necklace can be rotated without changing the arrangementof thebeads, thenumber of necklacesis atmost $20!/20=19!$ .Since a necklace can also be turned over without changing the arrange- mentof thebeads, thetotalnumber of necklaces, by the division principle, is $19!/2$ (镜面对称)

Circular permutations and necklaces are counted again in Chapter 14,in a more generalcontext. 
## 2.3 Combinations (Subsets) of Sets 
Let $S$ beasetof $n$ elements. A combination of a set $S$ is a term usually used to denote an unordered selection of the elements of $S$ .Theresultofsuchaselectionisasubset $A$ oftheelementsof $S$ $A\subseteq S$ ：Thusacombinationof $S$ isachoiceofasubsetof $S$ .Asaresult, the terms combination and subset are essentially interchangeable, and we shall generally use the more familiar subset rather than perhaps the more awkward combination, unless we want to emphasize the selection process. 
> 一个集合的组合是对该集合的无序的选取，选取出包含一定元素的子集，因此一个集合的组合就是该集合的一个子集选取
> “组合“和“子集“实际上可以互用

Now let $\mathcal{T}$ be a nonnegative integer. By an $\mathcal{r}$ -combinationofaset $S$ of $n$ elements, we understand an unordered selection of $\boldsymbol{\mathscr{r}}$ ofthe $n$ objectsof $S$ .Theresultofan $\boldsymbol{\mathscr{r}}$ -combinationisan $\mathscr{r}$ -subsetof $S$ ,a subset of $S$ consisting of $\mathcal{T}$ of the $n$ objectsof $S$ Again, we generally use ${\mathfrak{c}}_{\mathcal{r}}$ -subset"ratherthan ${\mathfrak{u}}_{r}$ combination." 
> n 个元素的 r 组合就是从中选出包含 r 个元素的子集

If $S=\{a,b,c,d\}$ ,then 

$$
\{a,b,c\},\{a,b,d\},\{a,c,d\},\{b,c,d\}
$$ 
are thefour 3-subsets of $S$ .We denoteby ${\binom{n}{r}}$ the number of $\mathcal{T}$ -subsetsofan $n$ -element set. Obviously, 
> 用 $\binom{n}{r}$ 表示一个 n 元素集合的 r 子集数量
> 显然当 r 大于 n，数量为0
> 当 n=0，数量也为0

$$
{\binom{n}{r}}=0\qquad{\mathrm{if~}}r>n.
$$ 
Also, 

$$
{\binom{0}{r}}=0\qquad{\mathrm{if~}}r>0.
$$ 
The following facts are readily seen to be true for each nonnegative integer $n$ 
$$
{\binom{n}{0}}=1,\ {\binom{n}{1}}=n,\ {\binom{n}{n}}=1.
$$ 
In particular, $\binom{0}{0}=1$ .The basic formula for the number of $\pmb{r}$ -subsets is given in the next theorem. 
> $\binom {n}{0} = 1, \binom{0}{0} = 1$


***Theorem 2.3.1*** 
For $0\leq r\leq n$ 

$$
P(n,r)=r!{\binom{n}{r}}.
$$ 
Hence, 

$$
{\binom{n}{r}}={\frac{n!}{r!(n.-r)!}}.
$$ 
> 定理：
> n 个元素的 r 排列数量等于 r 子集数量乘以 r 的全排列数量

Proof.
Let $S$ be an $n$ -elementset.Each $\pmb{r}$ permutation of $S$ arises in exactly one way as a result of carrying out the following two tasks: 
 (1)Choose $\mathcal{T}$ elementsfrom $S$ 
 (2)Arrange the chosen $\pmb{r}$ elements in some order. 

The number of ways to carry out the first task is,by definition,the number ${\binom{n}{r}}$ .The number of ways to carry out these condtaskis $P(r,r)=r!$ .By the multiplication principle,we have $P(n,r)\,=\,r!\,\left({n\atop r}\right)$ .We now use our formula $\begin{array}{r}{P(n,r)\,=\,\frac{n!}{(n-r)!}}\end{array}$ and obtain 

$$
{\binom{n}{r}}={\frac{P(n,r)}{r!}}={\frac{n!}{r!(n-r)!}}.
$$ 
> S 的每一个 r 排列是执行：1. 从 S 中选出 r 个元素 2. 对这 r 个元素进行排列，的结果，执行第一个任务的数量是 s 的 r-子集数量，执行第二个任务的数量是 r 的全排列数量

Example.
Twenty-five points are chosen in the plane so that no three of them are collinear.How many straight lines do they determine?How many triangles do they determine? 

Since no three of the pointslieon a line, every pair of points determines a unique straightline. Thus, the number of straight lines determined equals the number of 2-subsets of a25-element set, and this is given by 

$$
{\binom{25}{2}}={\frac{25!}{2!23!}}=300.
$$ 
Similarly, every three points determines a unique triangle, so that the number of triangles determined is given by 

$$
{\binom{25}{3}}={\frac{25!}{3!22!}}.
$$ 

Example.
There are 15 people enrolled in a mathematics course, but exactly 12 attend on any given day. The number of different ways that 12 students can be chosen 

$$
{\binom{15}{12}}={\frac{15!}{12!3!}}.
$$ 
If there are 25 seats in the classroom, the 12 students could seat themselves in $P(25,12)=25!/13!$ ways. Thus, thereare 

$$
{\binom{15}{12}}P(25,12)={\frac{15!25!}{12!3!13!}}
$$ 
ways in which an instructor might see the 12 students in the classroom. 

Example.
How many eight-letter words canbe constructed by using the 26letters of the alphabet if each word contains three,four,or five vowels?It is understood that there is no restriction on the number of times a letter can be used in a word. 

We count the number of words according to the number of vowels they contain and then use the addition principle 

First, consider words with three vowels. The three positions occupied by the vowels canbechosenin $\left(\begin{array}{l}{8}\\ {3}\end{array}\right)$ ways;the other five positions are occupied byconsonants.The vowel positions can then be completed in $5^{3}$ ways and the consonant positions in $21^{5}$ ways.Thus,the number of words with three vowels is 

$$
{\binom{8}{3}}5^{3}21^{5}={\frac{8!}{3!5!}}5^{3}21^{5}.
$$ 
In a similar way,we see that the number of words with four vowels is 

$$
{\binom{8}{4}}5^{4}21^{4}={\frac{8!}{4!4!}}5^{4}21^{4},
$$ 
and the number of words with five vowelsis 

$$
{\binom{8}{5}}5^{5}21^{3}={\frac{8!}{5!3!}}5^{5}21^{3}.
$$ 

Hence,the total number of words is 

$$
\frac{8!}{3!5!}5^{3}21^{5}+\frac{8!}{4!4!}5^{4}21^{4}+\frac{8!}{5!3!}5^{5}21^{3}.
$$ 

The following important property is immediate from Theorem 2.3.1: 

***Corollary 2.3.2***
For $0\leq r\leq n$ 

$$
{\binom{n}{r}}={\binom{n}{n-r}}.
$$ 
The numbers ${\binom{n}{r}}$ have many important and fascinating properties, and Chapter 5 is devoted to some of these. For the moment,we discuss only two basic properties. 
> 引理： $\binom {n}{r} = \binom{n}{n-r}$


***Theorem 2.3.3***
(Pascal's formula) For all integers $n$ and $k$ with $1\leq k\leq n-1$ 

$$
{\binom{n}{k}}={\binom{n-1}{k}}+{\binom{n-1}{k-1}}.
$$ 
> 定理：
> $\binom {n}{k} = \binom {n-1}{k} = \binom{n-1}{k-1}$

**Proof.** One way to prove this identity is to substitute the values of these numbers as given in Theorem 2.3.1 and then check that both sides are equal. Weleavethis straightforward verification to the reader. 

A combinatorial proof canbe obtained as follows: Let $S$ be a set of $n$ elements. Wedistinguishoneoftheelementsof $S$ and denote it by $\pmb{x}$ .Let $S\setminus\{x\}$ be the set obtainedfrom $S$ by removing the element $\pmb{x}$ .
We partition the set $X$ of $k$ -subsetsof $S$ intotwoparts, $A$ and $B$ .In $A$ weputallthose $k$ -subsetswhichdonotcontain $\boldsymbol{x}$ .In $B$ weputall the $k$ -subsets which do contain $x$ .The size of $X$ is $\textstyle|X|={\binom{n}{k}}$ ；hence, by the addition principle 

$$
{\binom{n}{k}}=|A|+|B|.
$$ 
The $k$ -subsets in $A$ are exactlythe $k$ -subsetsofthe set $S\backslash\{x\}$ of $n-1$ elements; thus, the size of $A$ is 

$$
|A|={\binom{n-1}{k}}.
$$ 
A $k$ -subsetin $B$ can alwaysbe obtainedby adjoining the element $\boldsymbol{\mathscr{x}}$ to a $\left(k-1\right)$ -subset $S\setminus\{x\}$ $B$ of .Hence, the size of satisfies 

$$
|B|={\binom{n-1}{k-1}}.
$$ 
Combining these facts, we obtain

$$
{\binom{n}{k}}={\binom{n-1}{k}}+{\binom{n-1}{k-1}}.
$$ 
> 证明：
> 证法1：直接代公式
> 证法2：
> 令集合 X 表示由集合 S 的所有 k-子集构成的集合，我们将集合 X 分为两个部分 A 和 B
> A 中放入的是所有不包含元素 x 的 k-子集，B 中放入的是所有包含元素 x 的 k-子集
> 显然，集合 X 的大小 $|X| = \binom{n}{k}$，因此根据加法原理，有 $|A| + |B| = |X| = \binom{n}{k}$
> 容易知道，A 中的 k-子集就是集合 $S\setminus \{x\}$ 的 k-子集，因此，A 的大小是 $|A| = \binom{n-1}{k}$
> 而集合 B 中的元素可以通过将 x 加入到集合 $S\setminus \{x\}$ 的 k-1子集得到，因此，B 的大小是 $|B| = \binom{n-1}{k-1}$
> 故得到 $\binom{n}{k} = \binom{n-1}{k} + \binom{n-1}{k-1}$

To illustrate the proof, let $n=5$ ， $k=3$ , and $S=\{x,a,b,c,d\}$ .Then the 3-subsets of $S$ in $A$ are

$$
\{a,b,c\},\{a,b,d\},\{a,c,d\},\{b,c,d\}.
$$ 
These are the 3-subsets of the set $\{a,b,c,d\}$ .The3-subsets $S$ in $B$ are 

$$
\{x,a,b\},\{x,a,c\},\{x,a,d\},\{x,b,c\},\{x,b,d\},\{x,c,d\}.
$$ 
Upon deletion of the element $_x$ in these 3-subsets,we obtain 

$$
\{a,b\},\{a,c\},\{a,d\},\{b,c\},\{b,d\},\{c,d\},
$$ 
the 2-subsetsof $\{a,b,c,d\}$ .Thus 

$$
{\binom{5}{3}}=10=4+6={\binom{4}{3}}+{\binom{4}{2}}.
$$ 
***Theorem 2.3.4*** 
For $n\geq0$ 

$$
{\binom{n}{0}}+{\binom{n}{1}}+{\binom{n}{2}}+\cdots+{\binom{n}{n}}=2^{n},
$$ 
and the common value equals the number of subsets of an $n$ -elementset. 
> 定理：
> $\binom{n}{0} + \binom{n}{1} + \dots = \binom{n}{n} = 2^n$
> 其中 $2^n$ 为 n 元素集合的子集数量

**Proof.** We prove this theorem by showing that both sides of the preceding equation count the number of subsets of an $n$ -elementset $S$ ,but in different ways. Firstwe observe that every subset of $S$ is an $\pmb{r}$ -subsetof $S$ for some $r=0,1,2,\ldots,n$ .Since ${\binom{n}{r}}$ equals the number of $\mathcal{T}$ -subsetsof $S$ ,it follows from the addition principle that 

$$
{\binom{n}{0}}+{\binom{n}{1}}+{\binom{n}{2}}+\cdot\cdot\cdot+{\binom{n}{n}}
$$ 
equals the number of subsets of $S$ 

We can also count the number of subsets of $S$ by breaking down the choice of a subset into $n$ tasks: Let the elements of $S$ be $x_{1},x_{2},\ldots,x_{n}$ . In choosing a subset of $S$ we have two choices to make for each of the $n$ elements: $\scriptstyle{\mathcal{X}}_{1}$ either goes into the subset or it doesn't, $x_{2}$ either goes into the subset or it doesn't,..·, $x_{n}$ either goes into the subset or it doesn't. Thus, by the multiplication principle, there are $2^{n}$ ways we can form a subset of $S$ .Wenow equate the two counts and complete the proof. 
> S 的每个子集都是一个 r-子集 ($r=0,1,\dots, n$)，因此左边等于右边

The proof of Theorem2.3.4 is aninstance of obtaining an identity by counting the objectsofaset (in this case the subsets of aset of $n$ elements) in two different ways andsetting theresults equalto one another. This technique of“double counting”is a. powerful one in combinatorics, and we will see several other applications of it. 

Example.
The number of 2-subsets of the set $\{1,2,\dots,n\}$ of the first $n$ positive integers is $\textstyle{\binom{n}{2}}$ .Partition the 2-subsets according to the largest integer they contain. For each $i=1,2,\dots,n$ thenumber of 2-subsets inwhich $i$ is the largest integer is $i-1$ (the other integer can be any of $1,2,\dots,i-1)$ .Equating the two counts,we obtain theidentity 

$$
0+1+2+\cdot\cdot\cdot+(n-1)={\binom{n}{2}}={\frac{n(n-1)}{2}}.
$$

## 2.4 Permutations of Multisets 
If $S$ is a multiset, an $\mathcal{T}$ -permutationof $S$ is an ordered arrangement of $\mathcal{T}$ of the objects of $S$. If thetotal number of objects of $S$ is $n$ (counting repetitions)，then an $n$ permutation of $S$ will also be called a permutation of $S$ .
> 如果 S 是一个多重集，则它的 r-排列是该集合中 r 个元素的有序排列
> 如果 S 包含的元素数量是 n（包括重复），则它的 n 排列称为该集合的一个排列

We first count the number of $\mathcal{T}$ -permutations of a multiset $S$ ,each of whose repetition number is infinite. 


***Theorem 2.4.1***
Let $S$ be a multiset with objects of $k$ differenttypes, whereeachobject hasaninfiniterepetitionnumber. Thenthenumberof $\pmb{r}$ -permutationsof of S is $k^{r}$ 
> 定理：
> 对于一个具有 k 个不同类型的对象的多重集 S，其中每个对象的重复次数都是无限，则 S 的 r 排列的数量是 $k^r$

**Proof.** In constructing an $\pmb{r}$ -permutationof $S$ ，we canchoosethefirstitemtobean object of anyone of the $k$ types. Similarly, the second item can bean object of anyone $k$ $S$ ofthe types, and so on. Since all repetition numbers of are infinite, the number $k$ of different choices for anyitemis always and itdoes notdepend on the choices of any previous items. By the multiplication principle, the $\mathscr{r}$ items can be chosenin $k^{r}$ ways.
> 证明：
> 一共选择 $r$ 次，每一次选择一个元素，都有 k 种类型可以选择，对于同一种类型，无论选择无限个中的哪一个，对于得到的排列都不会产生影响，因此一共有 $k^r$ 中不同的排列方式

An alternative phrasing of the theorem is: Thenumberof $\pmb{r}$ -permutationsof $k$ $k^{r}$ distinctobjects, each available in unlimited supply, equals .We also note that the conclusionofthetheoremremainstrueiftherepetitionnumbersofthe $k$ different types of objects of $S$ areallatleast $\pmb{r}$ .The assumption that the repetition numbers are infinite is a simple way of ensuring that we never run out of objects of any type. 
> 该定理也可以说为：k 个不同的对象的 r 排列数量（其中每个对象本身的数量都是无限个）等于 $k^r$
> 注意只要 k 个对象各自的重复此处至少是 r，该定理就是成立的

Example. 
What is the number of ternary numerals with at most four digits? 

The answer to this question is the number of 4-permutations of the multiset $\{\infty\,\cdot$ $0,\infty\cdot1,\infty\cdot2\}$ orofthemultiset $\{4\cdot0,4\cdot1,4\cdot2\}$ .By Theorem 2.4.1,this number equals $\mathbf{3^{4}}=\mathbf{81}$


We now count permutations of a multisetwith objects of $k$ different types,each with a finiterepetitionnumber. 

***Theorem 2.4.2***
Let $S$ be a multiset with objects of $k$ different types with finite repetition numbers $n_{1},n_{2},\ldots,n_{k}$ ，respectively. Let the size of $S$ be $n=n_{1}+n_{2}+\cdots+n_{k}$ Then the number of permutations of $S$ equals 

$$
{\frac{n!}{n_{1}!n_{2}!\cdot\cdot\cdot n_{k}!}}.
$$ 
> 定理：
> 令 $S$ 是一个包含了 k 个不同类型对象的多重集，每个对象的重复次数是有限的，分别是 $n_1, n_2,\dots, n_k$，令 $S$ 的大小为 $n = n_1 + n_2 + \dots + n_k$，则 $S$ 的排列数量等于 $n!/n_1! n_2!\dots n_k!$

**Proof.** We are given a multiset $S$ having objects of $k$ types, say $a_{1},a_{2},\dotsc,a_{k}$ ，with repetition numbers $n_{1},n_{2},\ldots,n_{k}$ ,respectively, for a total of $n=n_{1}+n_{2}+\cdot\cdot\cdot+n_{k}$ objects. We want to determine the number of permutations of these $n$ objects. We can think of it this way. There are $n$ places，and we want to put exactly one of the objects of $S$ in each of the places. We first decide which places are to be occupied by the $a_{1}\mathrm{{'s}}.$ Since there are $n_{1}\textbf{}a_{1}$ in $S$ ，we must choose a subset of $n_{1}$ places from the set of $n$ places. We can do this in $\scriptstyle{\binom{n}{n_{1}}}$ ways. We next decide which places are tobe occupied by the $a_{2}$ 's.There are $n-n_{1}$ places left, and we must choose $n_{2}$ of $\textstyle{\binom{n-n_{1}}{n_{2}}}$ $\scriptstyle{\binom{n-n_{1}-n_{2}}{n_{3}}}$ to choose the places for the $a_{3}$ 's.We continue like this, and invoke the multiplication principle and find thatthenumber ofpermutations of $S$ equals 

$$
{\binom{n}{n_{1}}}{\binom{n-n_{1}}{n_{2}}}{\binom{n-n_{1}-n_{2}}{n_{3}}}\cdot\cdot\cdot{\binom{n-n_{1}-n_{2}-\cdot\cdot\cdot-n_{k-1}}{n_{k}}}.
$$

> 我们需要决定 $S$ 中一共 $n$ 个对象的排列数量，可以认为有 n 个位置，我们要将每个 S 中的对象放在其中的一个位置
> 我们首先决定对象 $a_1$ 的位置，S 中有 $n_1$ 个 $a_1$，我们需要选择 $n_1$ 个位置被 $a_1$ 占据，一共有 $\binom{n}{n_1}$ 种方法，然后再从剩余的位置决定 $n_2$ 个 $a_2$ 的位置，一共有 $\binom{n-n_1}{n_2}$ 种方法，依次类推，就得到为所有的 $a_1,\dots ,a_k$ 可以决定的排列数量就是
> $\binom{n}{n_1}\binom{n-n_1}{n_2}\cdots \binom{n-n_1-\cdots-n_{k-1}}{n_k}$


Using Theorem 2.3.1, we see that this number equals 

$$
\begin{array}{r l r}&{}&{\frac{n!}{n_{1}!(n-n_{1})!}\frac{\left(n-n_{1}\right)!}{n_{2}!\left(n-n_{1}-n_{2}\right)!}\frac{\left(n-n_{1}-n_{2}\right)!}{n_{3}!\left(n-n_{1}-n_{2}-n_{3}\right)!}\cdot\cdot\cdot}\\ &{}&{\cdot\cdot\cdot\frac{\left(n-n_{1}-n_{2}-\cdot\cdot\cdot-n_{k-1}\right)!}{n_{k}!\left(n-n_{1}-n_{2}-\cdot\cdot\cdot-n_{k}\right)!},}\end{array}
$$ 
which, after cancellation, reduces to 

$$
\frac{n!}{n_{1}!n_{2}!n_{3}!\cdot\cdot\cdot n_{k}!0!}=\frac{n!}{n_{1}!n_{2}!n_{3}!\cdot\cdot\cdot n_{k}!}.
$$

> 展开之后化简，就得到 $n!/n_1! n_2!\dots n_k!$
> (不妨直接理解为所有 n 个元素全排列之后除去各个相同元素的全排列数量)


Example.
The number of permutations of the letters in the word MISSISSIPPI is 

$$
{\frac{11!}{1!4!4!2!}},
$$ 
since this number equals the number of permutations of the multiset $\{1\cdot M,4\cdot I,4$ $S,2\cdot P\}$

If the multiset $S$ has only two types, $\pmb{a}_{1}$ and $a_{2}$ ,of objects with repetition numbers $n_{1}$ and $n_{2}$ ，respectively, where $n\,=\,n_{1}+n_{2}$ ，then according to Theorem 2.4.2,the number of permutations of $S$ is 

$$
{\frac{n!}{n_{1}!n_{2}!}}={\frac{n!}{n_{1}!(n-n_{1})!}}={\binom{n}{n_{1}}}.
$$ 
Thus we may regard $\scriptstyle{\left({n\atop n_{1}}\right)}$ as the number of $n_{1}$ -subsets of a set of $n$ objects, and also as the number of permutations of a multiset with two types of objects with repetition numbers $n_{1}$ and $n-n_{1}$ ,respectively 

There is another interpretation of the numbers $\frac{n!}{n_{1}!n_{2}!\cdots n_{k}!}$ that occur in Theorem 2.4.2.This concerns the problem of partitioning a setof objects into parts of prescribed sizes where the parts now have labels assigned to them. To understand the implications ofthelastphrase,we offer the next example 


Example.
Consider a set of the four objects $\{a,b,c,d\}$ that is to be partitioned into two sets, each of size 2. If the parts are not labeled, then there are three different partitions: 

$$
\quad\{a,b\},\{c,d\};\quad\{a,c\},\{b,d\};\quad\{a,d\},\{b,c\}.
$$ 
Now suppose that the parts are labeled with different labels(e.g,.the colorsred and blue).Then the number of partitions is greater;indeed,there are six,since we can assign the labels red and blue to each part of a partition in two ways. For instance, for the particular partition $\{a,b\},\{c,d\}$ wehave 

$$
\mathrm{red}\text{box}\{a,b\},\mathrm{blue}\text{box}\{c,d\}
$$ 
and 

$$
\mathbf{blue~box}\{a,b\},\mathbf{red~box}\{c,d\}.
$$ 
In the general case,we can label the parts $B_{1},B_{2},\ldots,B_{k}$ (thinking of color 1,color 2，...,color $\pmb{k}$ ),andwealsothink of the parts as boxes.We then have the following result. 


***Theorem 2.4.3*** 
Let $n$ be a positive integer and let $n_{1},n_{2},\ldots,n_{k}$ be positive integers with $n=n_{1}+n_{2}+\cdot\cdot\cdot+n_{k}$ .The number of ways to partition a set of $n$ objects into $k$ labeled bozes in which Boz 1 contains $n_{1}$ objects,Bor 2 contains $\pmb{\mathscr{n}}_{2}$ objects, ..., Boc $k$ contains $n_{k}$ objects equals 

$$
{\frac{n!}{n_{1}!n_{2}!\cdot\cdot\cdot n_{k}!}}.
$$ 
If the bocesare not labeled,and $n_{1}=n_{2}=\cdots=n_{k}.$ ，then the number of partitions equals 

$$
{\frac{n!}{k!n_{1}!n_{2}!\cdots n_{k}!}}.
$$ 
> 如果 box 之间没有差异，且各个 box 的大小 $n_1 = n_2 = \dots = n_k$，则划分的数量还要再除以 $k!$

**Proof.** The proof is a direct application of the multiplication principle. We have to choose which objects go into which boxes, subjectto the size restrictions. We first choose $n_{1}$ objects for the first box, then $n_{2}$ of the remaining $n-n_{1}$ objects for the second box, then $\scriptstyle{\mathbf{\mathit{n}}_{3}}$ of the remaining $n-n_{1}-n_{2}$ objects for the third box,..., and finally $n_{-}n_{1}\_\dots\_n_{k-1}=n_{k}$ objects for the k th box. By the multiplication principle, the number of ways to make these choices is 

$$
{\binom{n}{n_{1}}}{\binom{n-n_{1}}{n_{2}}}{\binom{n-n_{1}-n_{2}}{n_{3}}}\cdot\cdot\cdot{\binom{n-n_{1}-n_{2}-\cdot\cdot\cdot-n_{k-1}}{n_{k}}}.
$$ 
As in the proof of Theorem 2.4.2, this gives 

$$
{\frac{n!}{n_{1}!n_{2}!\cdot\cdot\cdot n_{k}!}}.
$$ 
If boxes are not labeled and $n_{1}=n_{2}=\cdots=n_{k}$ ，then the result has to bedivided by $k!$ .This is so because, as in the preceding example, for each way of distributing the objects into the $k$ unlabeled boxes there are $k!$ ways in whichwe can nowattach thelabels $1,2,\ldots,k$ .Hence, using the division principle, we find that the number of partitions with unlabeled boxes is 

$$
{\frac{n!}{k!n_{1}!n_{2}!\cdot\cdot\cdot n_{k}!}}.
$$ 
The more difficult problem of counting partitions in which the sizes of the parts are not prescribed is studied in Section 8.2. 

We conclude this section with an example of a kind that we shall refer to many times in the remainder of the text.7 The example concerns non attacking rooks on a chessboard.Lest the reader be concerned that knowledge of chess is a prerequisite for therestof thebook,let us say at the outset that the only fact needed about the game of chess is that two rooks can attack one another if and only if they lie in the same rowor thesamecolumnof thechessboard.No other knowledge of chess is necessary (nor does it help!).Thus,a setof non attacking rooks on a chessboard simply means a collectionof“pieces"called rooks that occupy certain squares of theboard,and no two of the rooks lie in the same row or in the same column. 


Example.How many possibilities are therefor eight non attacking rooks on an 8-by-8 chessboard? 

Wegive each square on the board a pair $(i,j)$ of coordinates. The integer $_i$ desig nate s the row number ofthe square, and theinteger $j$ designates the column number ofthesquare. Thus, $_i$ and $j$ are integers between 1 and 8. Since the board is 8-by-8 and there are to be eight rooks on the board thatcannot attack one another, there must be exactly one rook in each row. Thus, the rooks must occupy eight squares withcoordinates 

$$
(1,j_{1}),(2,j_{2}),\ldots,(8,j_{8}).
$$ 
But there must also be exactly one rook in each column so that no two of the numbers ${j_{1},j_{2},\dots,j_{8}}$ canbeequal.Moreprecisely, 

$$
{j_{1},j_{2},\dots,j_{8}}
$$ 

must be a permutation of $\{1,2,\ldots,8\}$ .Conversely, if ${j_{1},j_{2},\dots,j_{8}}$ is apermutation of $\{1,2,\ldots,8\}$ ,then putting rooks in the squares with coordinates $(1,j_{1}),(2,j_{2}),\dots,(8,j_{8})$ we arrive at eight non attacking rooks on theboard. Thus, wehave a one-to-onecorre- sponde n ce between sets of 8 non attacking rooks on the 8-by-8 board and permutations of $\{1,2,\ldots,8\}$ .Sincethere are8! permutations of $\{1,2,\ldots,8\}$ ，thereare8! waysto place eightrooks on an 8-by-8board so thatthey arenonattacking 

We implicitly assumed in the preceding argument that the rooks were in dist in- guis h able from one another, thatis, they form a multiset of eight objects all of one type. Therefore, the only thing that mattered was which squares were occupied by rooks. If we have eight distinct rooks, say eight rooks each colored with one of eight differentcolors, then we have also to take into account which rook is in each of the eight occupied squares. Let us thus suppose that we have eight rooks of eight differ- entcolors. Havingdecided which eight squares aretobe occupied by the rooks (8! possibilities), we now haye also to decide what the color is of the rook in each of the occupied squares. As we look at the rooks from row 1 to row 8, we see a permutation of the eight colors. Hence, having decided which eight squares are to be occupied (8! possibilities), we then have to decide which permutation of the eight colors (8! permu tations) we shall assign. Thus, the number of ways tohave eight nonattacking rooks of eight different colors on an 8-by-8boardequals 

$$
8!8!=(8!)^{2}.
$$ 
Now suppose that, instead of rooks of eight different colors, wehave onered (R) rook,threeblue (B）rooks,andfour (Y) yellowrooks. It is assumed that rooks of the same color are indistinguishable from one another. 8 Now, as we look at the rooks fromrow1torow8, weseeapermutationofthecolorsofthemultiset 

$$
\{1\cdot R,3\cdot B,4\cdot Y\}.
$$ 
The number of permutations of this multiset equals,by Theorem 2.4.2 

$$
{\frac{8!}{1!3!4!}}.
$$ 
Thus,thenumber ofwaystoplace onered,threeblue,and four yellow rooks on an 8-by-8 board so that no rook can attack another equals 

$$
8!{\frac{8!}{1!3!4!}}={\frac{(8!)^{2}}{1!3!4!}}.
$$ 

The reasoning in the preceding example is quite general and leads immediately to thenext theorem. 


***Theorem2.4.4*** 
There aren rooks of $k$ colorswith $n_{1}$ rooks of the first color, $n_{2}$ rooks of the second color，.．：，and $n_{k}$ rooks of the kth color.The number of ways to arrange these rooks on ann-by-n board so that no rook can attack another equals 

$$
n!{\frac{n!}{n_{1}!n_{2}!\cdot\cdot\cdot n_{k}!}}={\frac{(n!)^{2}}{n_{1}!n_{2}!\cdot\cdot\cdot n_{k}!}}.
$$ 
> 定理：
> 有 n 个车，一共有 k 种颜色，其中 $n_1$ 个车的颜色是第一种，$n_2$ 个车的颜色是第二种，以此类推，则将这些车排布在 n-by-n 的棋牌上，使得每个车各占据一行一列的方法有 $n! \times n!/n_1! n_2!\dots n_k!$

Notethatif the rooks all have different colors $\mathbf{\dot{\mu}}_{k}=n$ and all $n_{\imath}=1$ )，theformula gives $(n!)^{2}$ as an answer. If the rooks are all colored the same ( $(k=1$ and $n_{1}=n_{\!}$ )，the formula gives $n!$ asananswer. 

Let $S$ bean $n$ -element multiset with repetition numbers equal to $n_{1},n_{2},\ldots,n_{k}$ SO that $n=n_{1}+n_{2}+\cdot\cdot\cdot+n_{k}$ .Theorem 2.4.2furnishes a simple formula for thenumber of $n$ permutationsof $S$ .If $r<n$ ,there is,in general,no simple formula for the number of $\mathcal{r}$ permutationsof $S.\,$ Nonetheless a solution can be obtained by the technique of generating functions, and we discuss this in Chapter 7. In certain cases, we can argue as in the next example. 


Example. 
Consider the multiset $S=\{3\cdot a,2\cdot b,4\cdot c\}$ of nine objects of three types. Find the number of 8-permutationsof $S$ 

The 8-permutations of $S$ can be partitioned into three parts: 
(i)8-permutations of $\{2\cdot a,2\cdot b,4\cdot c\}$ ,ofwhichthereare 

$$
{\frac{8!}{2!2!4!}}=420;
$$ 
(ii)8-permutations of $\{3\cdot a,1\cdot b,4\cdot c\}$ ,ofwhichthereare 

$$
{\frac{8!}{3!1!4!}}=280;
$$ 
(iii)8-permutations of $\{3\cdot a,2\cdot b,3\cdot c\}$ ,ofwhichthereare 

$$
{\frac{8!}{3!2!3!}}=560.
$$ 

Thus,the number of 8-permutations of $S$ is 

$$
420+280+560=1260.
$$ 
## 2.5 Combinations of Multisets 
If $S$ is a multiset, then an $\pmb{r}$ -combinationof $S$ is an unordered selection of $\mathcal{T}$ ofthe objectsof $S$ Thus, an $\mathscr{r}$ -combinationof $S$ (moreprecisely, the result of the selection) isitselfamultiset, asubmultisetof $S$ of size $\pmb{r}$ ，or，for short, an $\mathscr{r}$ -submultiset. If $S$ has $n$ objects, then there is only one $n$ -combinationof $S$ ，namely, $S$ itself. If $S$ contains objects of $k$ different types, then there are $k$ 1-combinations of $S$ .Unlikewhen discussing combinations of sets, we generally use combination rather than sub multiset.
> 一个多重集 S 的 r 组合是一个 S 中对象的无序组合，显然 r 组合本身也是一个多重集，或者称它为集合 S 中的大小为 r 的子多重集

 Example.
 Let $S=\{2\cdot a,1\cdot b,3\cdot c\}$ Then the 3-combinations of are 

$$
\begin{array}{r l}{\{2\cdot a,1\cdot b\},}&{\{2\cdot a,1\cdot c\},\quad\{1\cdot a,1\cdot b,1\cdot c\},}\\ &{}\\ {\{1\cdot a,2\cdot c\},}&{\{1\cdot b,2\cdot c\},\quad\{3\cdot c\}.}\end{array}
$$ 
We first count the number of $\mathcal{r}$ -combinations of a multiset all of whose repetition numbers are infinite(oratleast $\mathscr{r}$ 一 


***Theorem2.5.1***
Let $S$ be a multiset with objects of $k$ types,each with an infinite repetitionnumber.Then the numberof $\mathscr{r}$ -combinationsof $S$ equals 

$$
{\binom{r+k-1}{r}}={\binom{r+k-1}{k-1}}.
$$

> 定理：
> S 是一个包含了 k 个类型的多重集，每个类型对象的重复次数都是无限次，则S 的 r 组合的数量是 $\binom{r+k-1}{r} = \binom{r+k-1}{k-1}$


**Proof**. Let the $k$ types of objects of $S$ be $a_{1},a_{2},\dotsc,a_{k}$ sothat 

$$
{\cal S}=\{\infty\cdot a_{1},\infty\cdot a_{2},\ldots,\infty\cdot a_{k}\}.
$$ 
Any $r$ -combination of $S$ is of the form $\left\{x_{1}\cdot a_{1},x_{2}\cdot a_{2},\ldots,x_{k}\cdot a_{k}\right\}$ where $x_{1},x_{2},\ldots,x_{k}$ are nonnegative integers with $x_{1}+x_{2}+\dots+x_{k}=r$ ：Conversely,every sequence $x_{1},x_{2},\ldots,x_{k}$ of nonnegative integerswith $x_{1}+x_{2}+\cdot\cdot\cdot+x_{k}=r$ corresponds to an $\mathcal{r}$ -combinationof $S$ .Thus,the number of $\mathscr{r}$ -combinationsof $S$ equals the number of solutions of the equation 

$$
x_{1}+x_{2}+\dots+x_{k}=r,
$$ 
where $x_{1},x_{2},\ldots,x_{k}$ are nonnegative integers.We show that the number of these solutions equals the number of permutations of the multiset 

$$
T=\{r\cdot1,(k-1)\cdot*\}
$$ 
of $r+k-1$ objects of two different types. Given a permutation of $T$ ，the $k-1*\mathbf{\dot{s}}$ dividethe $\mathscr{r}$ 1s into $k$ groups.Let therebe $\scriptstyle{\mathcal{X}}_{1}$ 1s to theleftof thefirst $^*$ $\boldsymbol{x_{2}}$ 1sbetween the first and the second $*,\,\cdot\cdot\cdot$ and $\scriptstyle{\mathcal{X}}_{k}$ 1s to the right of the last $^*$ . Then $x_{1},x_{2},\ldots,x_{k}$ are non negative integers with $x_{1}+x_{2}+\cdot\cdot\cdot+x_{k}=r$ Conversely,given non negative integers $x_{1},x_{2},\ldots,x_{k}$ with $x_{1}\!+\!x_{2}\!+\!\cdot\cdot\!+\!x_{k}=r$ we can reverse the preceding steps and construct a permutation of $T$ 10 Thus, the number of $\pmb{r}$ -combinations of themultiset $S$ equals the number of permutations of the multiset T,whichby Theorem 2.4.2 is 

$$
{\frac{(r+k-1)!}{r!(k-1)!}}={\binom{r+k-1}{r}}.
$$ 
> 证明：
> S 的 r 组合满足从各个类型的对象中选出 $x_1, x_2,\dots ,x_k$ 个，其中 $x_1,\dots ,x_k$ 都是非负数，满足 $x_1 + x_2 + \dots + x_k = r$，很显然，任意的一个 $x_1,\dots ,x_k$ 序列都对应于一个 r 组合，因此，r 组合的数量实际上等于 $x_1+ \dots + x_k = r$ 的解的数量
> 而该式的解的数量等于多重集合 $T=\{r\cdot1, (k-1)\cdot*\}$ 的排列数
> 给定一个 $T$ 的排列，$k-1$ 个 $*$ 会将那 $r$ 个 1划分为 k 个组，设各组中 1 的数量分别是 $x_1, x_2,\dots ,x_k$，显然满足 $x_1 + x_2 + \dots + x_k = r$，同样，给定一个序列 $x_1 + x_2 + \dots + x_k = r$，其中 $x_i$ 都非负，我们可以根据这个序列构造一个 T 的排列，因此多重集 S 的 r 组合的数量等于多重集 T 的排列数量 $\frac{(r+k-1)!}{r! (k-1)!}= \binom{r+k-1}{r}$


Another way of phrasing Theorem 2.5.1 is as follows: The number of r-combinations of $k$ distinctobjects,each available in unlimited supply,equals 

$$
{\binom{r+k-1}{r}}.
$$ 
We note that Theorem 2.5.1remainstrue if the repetition numbers of the $k$ distinct objectsof $S$ areall at least $\mathcal{r}$ 
> 注意只要 S 中各个对象的重复数量至少是 r 次，该定理就成立


Example.
A bakery boasts eight varieties of doughnuts.If a box of doughnuts contains one dozen,how many different options are there for a box of doughnuts? 

It is assumed that the bakery has on hand a large number (at least 12)of each variety.Thisis a combination problem,sincewe assume the order of the doughnuts in a box is irrelevant for thepurchaser'spurpose.The number of different options for boxes equals the number of 12-combinations of a multiset with objects of 8 types,each having an infinite repetition number.ByTheorem 2.5.1,this number equals 

$$
{\binom{12+8-1}{12}}={\binom{19}{12}}.
$$ 

Example.
Whatisthenumberofnondecreasingsequencesoflength $\pmb{r}$ whoseterms aretakenfrom $1,2,\ldots,k?$ 

The non decreasing sequences to be counted can be obtained by first choosing an $\mathscr{r}\cdot$ -combination of the multiset 

$$
S=\{\infty\cdot1,\infty\cdot2,\ldots,\infty\cdot k\}
$$ 

and then arranging the elements in increasing order.Thus,the number of such se quencesequalsthenumberof $\mathscr{r}$ combinationsof $S$ ,and hence,by Theorem 2.5.1, equals 

$$
{\binom{r+k-1}{r}}.
$$ 
In the proof of Theorem 2.5.1，we defined a one-to-one correspondence between $\mathscr{r}$ combinations of a multiset $S$ withobjectsof $\pmb{k}$ different types and the non negative integral solutions of the equation 

$$
x_{1}+x_{2}+\dots+x_{k}=r.
$$ 
In this correspondence, $x_{i}$ represents the number of objects of the it h type that are usedinthe $\pmb{r}$ -combination.Putting restrictions on the number of times each type of object is to occur in the $\mathscr{r}$ -combination can be accomplished by putting restrictions on the $x_{i}$ .We give a first illustration of this in the next example 


Example.
Let $S$ bethemultiset $\left\{10\cdot a,10\cdot b,10\cdot c,10\cdot d\right\}$ with objects of four types, $a,b,c,$ and $^{d}$ What is the number of 10-combinationsof $S$ that have the property that eachof thefour types of objects occurs atleastonce? 

The answer is the number of positive integral solutions of 

$$
x_{1}+x_{2}+x_{3}+x_{4}=10,\qquad.
$$ 

where $\pmb{x}_{1}$ represents the number of $a$ 'sin a10-combination, $x_{2}$ the number of $b$ 's, $\boldsymbol{x_{3}}$ thenumberof $c{\bf\dot{s}}$ and $\pmb{x_{4}}$ thenumberof $d{\bf s}.$ Sincetherepetitionnumbersallequal 10,and 10 is the size of the combinations being counted,we canignore the repetition numbersof $S$ .We then perform the changes of variable: 

$$
y_{1}=x_{1}-1,\;y_{2}=x_{2}-1,\;y_{3}=x_{3}-1,\;y_{4}=x_{4}-1
$$ 
to get 

$$
{\bf y}_{1}+{\bf y}_{2}+{\bf y}_{3}+{\bf y}_{4}=6,
$$ 
wherethe $\mathbf{\nabla}y_{i}$ 's are tobe nonnegative.The number of nonnegative integral solutions of the new equation is,byTheorem2.5.1, 

$$
{\binom{6+4-1}{6}}={\binom{9}{6}}=84.
$$ 

Example. 
Continuing with the doughnut example following Theorem 2.5.1, we see thatthenumber of different optionsforboxes of doughnuts containing atleast one doughnut of each of the eight varieties equals 

$$
{\binom{4+8-1}{4}}={\binom{11}{4}}=330.
$$ 
General lower bounds onthenumber of times each type of objectoccurs in the combination alsocanbe handled bya change ofvariable.We illustrate this in the next example 


Example.
What is the number of integral solutions ofthe equation 

$$
x_{1}+x_{2}+x_{3}+x_{4}=20,
$$ 
in which 

$$
x_{1}\geq3,\;x_{2}\geq1,\;x_{3}\geq0\;\;\mathrm{and}\;\;x_{4}\geq5?
$$ 
We introduce the new variables 

$$
y_{1}=x_{1}-3,\;y_{2}=x_{2}-1,\;y_{3}=x_{3},\;y_{4}=x_{4}-5,
$$ 
and our equation becomes 

$$
y_{1}+y_{2}+y_{3}+y_{4}=11.
$$ 
The lowerboundsonthe $x_{i}$ 'saresatisfiedifandonlyifthe ${\mathbfit{y}}_{i}$ 'sarenonnegative.The number of non negative integral solutions of thenewequation,and hence the number of non negative solutions of the original equation,is 

$$
{\binom{11+4-1}{11}}={\binom{14}{11}}=364.
$$ 

It is more difficult to count the number of $\mathscr{r}$ -combinationsofamultiset 

$$
S=\left\{n_{1}\cdot a_{1},n_{2}\cdot a_{2},\ldots,n_{k}\cdot a_{k}\right\}
$$ 
with $k$ types of objects and general repetition numbers $n_{1},n_{2},\ldots,n_{k}$ .Thenumber of $\mathcal{T}$ combinationsof $S$ is the sameas thenumber of integral solutions of 

$$
x_{1}+x_{2}+\cdots+x_{k}=r,
$$ 
where 

$$
0\leq x_{1}\leq n_{1},\quad0\leq x_{2}\leq n_{2},\quad\ldots,\quad0\leq x_{k}\leq n_{k}.
$$ 
Wenowhaveupperboundsonthe $\boldsymbol{\mathscr{x}}_{i}$ 's,and these cannotbehandled inthe sameway aslowerbounds.In Chapter 6 we show how the inclusion-exclusion principle provides a satisfactory method for this case. 

## 2.6 Finite Probability 
In this section we give a brief and informal introduction to finite probability.ll As we will see,it all reduces to counting,and so the counting techniques discussed inthis chapter can be used to calculate probabilities. 
> 有限概率：和积分计算的连续概率相对

Thesettingforfiniteprobabilityisthis: There is an erperiment $\varepsilon$ whichwhen carried out results in one of a finite set of outcomes.We assume that each outcome is equally likely (that is,no outcome is more likelyto occur than any other);we say that the experiment is carried out randomly.Theset of all possible out comesiscalledthe sample space of the experiment and is denotedby $S$ .Thus $S$ is a finite set with,say, $n$ elements: 

$$
S=\{s_{1},s_{2},\ldots,s_{n}\}.
$$ 
When $\varepsilon$ is carried out, each $\mathscr{s}_{i}$ has a 1 in $n$ chance of occuring, and so we say that the probabilityoftheoutcome $s_{i}$ is $1/n$ ，written 

$$
\operatorname{Prob}(s_{i})={\frac{1}{n}},\quad(i=1,2,\ldots,n).
$$ 
Aneventisjustasubset $E$ of the sample space $S$ ,but it is usually given descriptively and not by actually listing all the outcomes in $E$ 
> 上面将的就是概率论的那一套


Example.
Consider the experiment $\varepsilon$ of tossing three coins,where each of the coins lands showing either Heads $(H)$ orTails $(T)$ .Since each coin can come up either $H$ on $T$ ,the sample space ofthis experiment is the set $S$ of consisting of the eight ordered pairs 

$$
\begin{array}{c}{{(H,H,H),(H,H,T),(H,T,H),(H,T,T),}}\\ {{(T,H,H),(T,H,T),(T,T,H),(T,T,T),}}\end{array}
$$ 
where,for instance, $(H,T,H)$ means that the first coin comes up as $H$ ，thesecond coin comes up as $T$ ,and the third coin comes up as $H$ .Let $E$ be the event thatat least two coins come up $H$ .Then 

$$
E=\{(H,H,H),(H,H,T),(H,T,H),(T,H,H)\},
$$ 
Since $E$ consists of four outcomes out of a possible eight outcomes,it is natural to assignto $E$ theprobability $4/8=1/2$ .This is made more precise in the next definition.

The probability of an event $E$ in an experiment with a sample space $S$ isdefined to be the proportion of outcomes in $S$ thatbelongto $E$ ；thus, 

$$
\operatorname{Prob}(E)={\frac{|E|}{|S|}}.
$$ 
By this definition,theprobability of an event $E$ satisfies 

$$
0\leq\operatorname{Prob}(E)\leq{\bar{1}},
$$ 
where $\mathbf{Prob}(E)=0$ if and only if $E$ is the empty event $\varnothing$ (the impossible event)and $\mathrm{Prob}(E)\,=\,1$ if andonlyif $E$ is the entire sample space $S$ (the guaranteed event) Thus to compute the probability of an event $E$ ，we have to make two counts: count the number of outcomes in the sample space $S$ and count the number of outcomes in the event $E$ 
> 要计算事件 E 的概率，需要进行两次计数：对样本空间内的结果数量进行计数、对事件的结果数量进行计数


Example.
We consider an ordinary deck of 52 cards with each card having one of 13 ranks $1,2,\dotsc,10,11,12,13$ and four suits Clubs (C),Diamonds (D),Hearts (H),and Spades (S).Usually, 11 is denoted as a Jack, 12 as a Queen,and 13 as a King.In addition,1 hastwo roles:either as a1(low;below the 2)or as anAce (high;above the King).12 Consider the experiment $\varepsilon$ of drawing a card at random.Thus the sample space $S$ is the set of 52 cards,each of which is assigned a probability of $1/52$ .Let $E$ be the event that the card drawn is a 5. Thus 

$$
E=\{(\mathbf{C},5),(\mathbf{D},5),(\mathbf{H},5),(\mathbf{S},5)\}.
$$ 
Since $|E|=4$ and $|S|=52,\,\mathrm{Prob}(E)=4/52=1/13.$ 

Example.Let $n$ be a positive integer.Suppose we choose a sequence $i_{1},i_{2},\dots,i_{n}$ ofintegersbetween1and $n$ atrandom.(1)What is the probability that the chosen sequence is a permutation of $1,2,\dots,n?$ (2)What is the probability that the sequence containsexactly $n-1$ different integers? 

Thesamplespace $S$ isthesetofallpossiblesequencesoflength $n$ eachofwhose termsisoneoftheintegers $1,2,\dots,n$ .Hence $|S|=n^{n}$ because there are $n$ choicesfor eachofthe $n$ terms. 

(1）Theevent $E$ that the sequence is a permutation satisfies $|E|=n!$ .Hence 

$$
\operatorname{Prob}(E)={\frac{n!}{n^{n}}}.
$$ 
(2) Let $F$ be the event that the sequence contains exactly $_{n-1}$ different integers.A sequencein $F$ contains one repeated integer,and exactly one of the integers $1,2,\ldots,n$ is missing in the sequence (so $n-2$ other integers occur in the sequence). There are $n$ choices for the repeated integer,and then $n-1$ choices for the missing integer. The places for the repeated integer can be chosen in $\textstyle{\binom{n}{2}}$ ways;theother $n-2$ integerscan be putin theremaining $n-2$ placesin $(n-2)$ !ways.Hence 

$$
|F|=n(n-1){\binom{n}{2}}(n-2)!={\frac{n!^{2}}{2!(n-2)!}},
$$ 
and 

$$
\operatorname{Prob}(F)={\frac{n!^{2}}{2!(n-2)!n^{n}}}.
$$ 
Example. Five identical rooks are placed at random in nonattacking positions on an 8-by-8board.What is the probability that the rooks are both in rows $1,2,3,4,5$ and in columns 4,5,6,7,8? 

Oursamplespace $S$ consist of all placements of five nonattacking rooks on the boardandso 

$$
|S|={\binom{8}{5}}^{2}\cdot5!={\frac{8!^{2}}{3!^{2}5!}}.
$$ 
Let $E$ be the event that thefive rooks are in the rows and columns prescribed above Then $E$ has size 5!, since there are 5! ways to place five nonattacking rooks on a 5-by-5 board.Hencewehave 

$$
\mathrm{Prob}(E)={\frac{5!^{2}3!^{2}}{8!^{2}}}={\frac{1}{3136}}.\qquad.
$$ 
Example.This is a multi part example relating to the card game Poker played with an ordinary deck of 52 cards.A poker hand consists of 5 cards.Ourexperiment

 $\mathcal{E}$ is to select a poker hand at random.Thus the sample space $S$ consistsofthe

 ${\binom{52}{5}}=2$ namely $1/2,598,960.$ 

(1)Let $E$ be the event that the poker hand is a full house;thatis,threecardsofone rank and two cards of a different rank(suitdoesn'tmatter).Tocomputethe probabilityof $E$ ,we need to calculate $|E|$ .Howdowedeterminethenumberof fullhouses?We use the multiplication principle thinking of four tasks: 

(a)Choose the rank with three cards. (b)Choose the three cards of that rank i.e.,their 3 suits. (c)Choose the rank with two cards. (d)Choose the two cards of that rank i.e.,their 2 suits. 

The number ofways of carrying thesetasks outis asfollows: 

(a) 13 

(b) $\textstyle{{\binom{4}{3}}=4}$ (c) 12 (after choice (a), 12 ranks remain) (d) ${\binom{4}{2}}=6$ 

Thus $|E|=13\cdot4\cdot12\cdot6=3,744$ and 

$$
\operatorname*{Pr}(E)={\frac{3,744}{2,598,960}}\approx0.0014.
$$ 
(2)Let $E$ be the eventthatthe poker·hand is a straight;that is,five cards of consecutive ranks(suitdoesn'tmatter),keeping in mind that the 1 is also the Ace.Tocompute $|E|$ ，wethinkoftwotasks: 

(a)Choose the five consecutive ranks. (b)Choose the suitofeachof theranks The number of ways of carrying out these two tasks is as follows (a)10 (the straights can begin with any of 1,2....,10) (b) $4^{5}$ (four possible suits for each rank) Thus $|E|=10\cdot4^{5}=10,240$ and 

$$
\operatorname*{Pr}(E)={\frac{10,240}{2,598,960}}\approx0.0039.
$$ 
(3)Let $E$ be the event that the poker hand isastraightfush;thatis,fivecards of consecutive ranks,all of the same suit.Using the reasoning in (b),we see that $\vert E\vert=10\cdot4=40$ and 

$$
\operatorname*{Pr}(E)={\frac{40}{2,598,960}}\approx0.0000154.
$$ 
(4)Let $E$ be the event that the poker hand consists of.eractly two pairs; that is,two cardsofonerank,two cards of a different rank,and one card of an additionally differentrank.Here we have to be a little careful since the first two mentioned ranks appear in the same way(as opposed to the full house,wheretherewere threecardsof onerankandtwo cardsof a differentrank).Tocompute $|E|$ in this case,we think of three tasks(not six ifwehad imitated(1)):· 

(a)Choosethetworanksoccuringinthetwopairs (b)Choose the two suits for each of these two ranks (c)Choose the remaining card. The number of ways of carrying out these three tasks is as follows: 

(a) $\begin{array}{l}{{{\binom{13}{2}}=78}}\\ {{{\binom{4}{2}}{\binom{4}{2}}=6\cdot6=36}}\end{array}$ (b) (c) 44 

Thus $|E|=78\cdot36\cdot44=123,552$ and 

$$
\operatorname*{Pr}(E)={\frac{123,552}{2,598,960}}\approx0.048,
$$ 
almosta $1\;\mathrm{in}\;20$ chance. 

(5）Let $E$ be the event that the poker hand contains at least one Ace.Here we use our subtraction principle. Let $\overline{{E}}=S\setminus E$ be the complementary event of a poker hand with no aces. Then $|\overline{{E}}|={\binom{48}{5}}=1$ 712,304.Thus $\left|E\right|=\left|S\right|-\left|\overline{{E}}\right|=$ $2,598,960-1,712,304=886,656$ and 

$$
{\begin{array}{r c l}{\operatorname*{Pr}(E)}&{=}&{{\cfrac{2,598,960-1,712,304}{2,598,960}}}\\ &{}&\\ &{=}&{1-{\cfrac{1,712,304}{2,598,960}}}\\ &{}&\\ &{=}&{{\cfrac{886,656}{2,598,960}}}\\ &{}&\\ &{\approx}&{0.34.}\end{array}}
$$ 
As we see in the calculation in (5), our subtraction principle in terms of probability becomes 

$$
\operatorname*{Pr}(E)=1-\operatorname*{Pr}({\overline{{E}}}),{\mathrm{~equivalently,~}}\operatorname*{Pr}({\overline{{E}}})=1-\operatorname*{Pr}(E).
$$ 
More probability calculations are given in the Exercises. 
# 3 The Pigeonhole Principle 
We consider in this chapter an important，butelementary,combinatorial principle that can be used to solve a variety of interesting problems,often with surprising conclusions.This principle is known under a variety of names,themostcommon of which are the pigeonhole principle,the Di rich let drawer principle,andtheshoebor principle.Formulated as a principle about pigeonholes,it says roughly that if a lot of pigeons fy into not too many pigeonholes,then at least one pigeonhole will be occupied by two or more pigeons.A moreprecise statementis given below. 
> 鸽笼原理=狄利特雷抽屉原理=鞋盒原理
> 大致的意思：很多鸽子飞入较少的鸽笼，则至少一个笼子会有2个或更多鸽子
## 3.1 Pigeonhole Principle: Simple Form 
The simplest form of the pigeonhole principle is the following fairly obvious assertion. 

***Theorem3.1.1***
If $n+1$ objectsare distributed into nbores,thenatleast one bor contains two or more of the objects. 
> 定理：
> $n+1$ 个物体放入 $n$ 个盒子，则至少有一个盒子包含两个或者更多物体

**Proof.** The proof is by contradiction.If each of the $n$ boxes contains at most one of theobjects,thenthetotal number of objectsis atmost $1+1+\cdot\cdot\cdot+1{\big(}n\ 1\mathbf{s}{\big)}=n$ Since we distribute $n+1$ objects,some box contains at least two of the objects. 
> 证明：
> 反证法，若每个盒子最多包含1个物体，则一共最多包含 n 个物体，少于 n+1，矛盾，因此如果有 n+1 个物体，至少有一个盒子包含大于1个物体

Notice that neither the pigeonhole principle nor itsproofgives any help in finding a box that contains two or more of the objects.They simply assert that if we examine each of theboxes,wewill come uponaboxthat contains more than one object.The pigeonhole principle merely guarantees the existence of such a box.Thus,whenever the pigeonhole principle is applied to prove the existence of an arrangement or some phenomenon,it will give no indication of how to construct the arrangement or find an instance of the phenomenon other than to examine all possibilities. 
> 鸽笼定理仅仅保证了存在性，没有其他更多信息