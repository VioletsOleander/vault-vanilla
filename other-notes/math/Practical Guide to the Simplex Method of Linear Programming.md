Marcel Oliver Revised: September 28, 2020 

# 1 The basic steps of the simplex algorithm 
## Step 1: Write the linear programming problem in standard form 
Linear programming (the name is historical, a more descriptive term would be linear optimization) refers to the problem of optimizing a linear objective function of several variables subject to a set of linear equality or inequality constraints. 
> 线性规划/线性优化指在一组线性等式/不等式的约束下，优化一个关于多个变量的线性目标函数

Every linear programming problem can be written in the following standard form . 

$$
{\mathrm{Minimize~}}\zeta=\pmb c^{T}{\boldsymbol{x}}\tag{1a}
$$

subject to 

$$
\begin{align}
A\pmb x = \pmb b\tag{1b}\\
\pmb x \ge \pmb 0\tag{1c}
\end{align}
$$

Here $\pmb{x}\,\in\,\mathbb{R}^{n}$ is a vector of $n$ unknowns, $A\,\in\,M(m\,\times\,n)$ with $n$ typically much larger than $m$ , $\pmb{c}\in\mathbb{R}^{n}$ the coeﬃcient vector of the objective function, and the expr $\pmb x\ge \pmb 0$ signiﬁes $x_{i}\geq0$ for $i=1,\dots,n$ . For simplicity, we assume that rank $\operatorname{rank}A=m$ , i.e., that the rows of A are linearly independent. 

>  所有的线性规划问题都可以写为如上的标准形式
>  $\pmb x \in \mathbb R^n$ 是包含 $n$ 个未知数的向量
>  $A\in M(m\times n)$ 为矩阵，一般 $n$ 会远大于 $m$，也就是列数会远大于行数，$m$ 就表示了 $A$ 定义的线性约束的数量
>  $\pmb c\in \mathbb R^n$ 为目标函数的系数向量
>  $\pmb x\ge \pmb 0$ 表示了 $\pmb x$ 的每个成分都非负，同样属于线性约束
>  我们假设 $A$ 的秩为 $m$，即 $A$ 的行线性独立，因此 $A$ 定义的线性约束也相互独立

Turning a problem into standard form involves the following steps. 
>  将线性规划问题写为标准形式的步骤为：

(i) Turn Maximization into minimization and write inequalities in standard order. 

This step is obvious. Multiply expressions, where appropriate, by $-1$ . 

>  将最大化转化为最小化
>  用标准顺序写下不等式

(ii) Introduce slack variables to turn inequality constraints into equality constraints with non-negative unknowns. 

Any equality of the form $a_{1}\,x_{1}+\cdot\cdot\cdot+a_{n}\,x_{n}\leq c$ can be replaced by $a_{1}\,x_{1}+\cdot\cdot\cdot+a_{n}\,x_{n}+s=c$ with $s\geq0$.

>  引入松弛变量，将不等式约束转为带有非负未知数的等式约束
>  任意形式为 $a_1x_1 + \dots + a_nx_n \le c$ 可以通过引入松弛变量 $s \ge 0$，转化为等式
>  $a_1x_1 + \dots + a_n x_n + s = c$

(iii) Replace variables which are not sign-constrained by diﬀerences. 

Any real number $x$ can be written as the diﬀerence of non-negative numbers $x=u-v$ with $u,v\geq0$ . 

>  将任意没有符号约束的变量替换为两个非负变量之差
>  任意实数 $x$ 都可以重写为两个非负实数 $u, v\ge 0$ 的差 $x = u - v$

Consider the following example. 

$$
{\mathrm{Maximize~}}z=x_{1}+2\,x_{2}+3\,x_{3}\tag{2a}
$$ 
subject to 

$$
\begin{align}
x_1 + x_2 - x_3 &= 1\tag{2b}\\
-2x_1 + x_2 + 2x_3 &\ge -5\tag{2c}\\
x_1 - x_2 &\le 4\tag{2d}\\
x_2 + x_3 &\le 5\tag{2e}\\
x_1 &\ge 0\tag{2f}\\
x_2 &\ge 0\tag{2g}
\end{align}
$$

Written in standard form, the problem becomes 

$$
{\mathrm{minimize~}}\zeta=-x_{1}-2\,x_{2}-3\,u+3\,v
$$ 
subject to 

$$
\begin{align}
x_1 + x_2 - u + v &= 1\\
2x_1 - x_2 - 2u + 2v + s_1 &= 5\\
x_1 - x_2 + s_2 &= 4\\
x_2 + u-v+s_3 &=5\\
x_1,x_2,u,v,s_1,s_2,s_3 &\ge 0
\end{align}
$$

## Step 2: Write the coefficients of the problem into a simplex tableau 
The coefficients of the linear system are collected in an augmented matrix as known from Gaussian elimination for systems of linear equations; the coeﬃcients of the objective function are written in a separate bottom row with a zero in the right hand column. 
>  此时等式约束构成了一个线性系统
>  我们将该线性系统的系数收集为一个增广矩阵
>  同时目标函数的系数也写在了最后一行，其最右边列的值记为 0

For our example, the initial tableau reads: 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/d58029865e13710dbe42d63b81079de610db213edc78fe711166c7cd58f32b87.jpg) 

In the following steps, the variables will be divided into $m$ basic variables and $n-m$ non-basic variables . We will act on the tableau by the rules of Gaussian elimination, where the pivots are always chosen from the columns corresponding to the basic variables. 
>  之后，我们将把 $n$ 个变量 ($n$ 个列) 划分为 $m$ 个基本变量和 $n-m$ 个非基本变量
>  我们将依据高斯消元法的规则对该表格进行操作，主元总是选定为基本变量的列

Before proceeding, we need to choose an initial set of basic variables which corresponds to a point in the feasible region of the linear programming problem. Such a choice may be non-obvious, but we shall defer this discussion for now. In our example, $x_{1}$ and $s_{1},\dots,s_{3}$ shall be chosen as the initial basic variables, indicated by gray columns in the tableau above. 
>  我们需要首先选定基本变量的初始集合
>  初始集合需要对应于线性规划问题的可行区域的一个点

## Step 3: Gaussian elimination 
For a given set of basic variables, we use Gaussian elimination to reduce the corresponding columns to a permutation of the identity matrix. This amounts to solving $A x=b$ in such a way that the values of the non basic variables are zero and the values for the basic variables are explicitly given by the entries in the right hand column of the fully reduced matrix. In addition, we eliminate the coeﬃcients of the objective function below each pivot. 
>  对于给定的一组基本变量，我们使用高斯消元法将相应的列化为单位矩阵的一个排列
>  这相当于解 $A\pmb x = \pmb b$，使得当非基本变量的值为零时，基本变量的值由完全简化矩阵右列中的条目显式给出
>  此外，我们在每个主元下方将目标函数的系数消去至 0

Our initial tableau is thus reduced to 

The solution expressed by the tableau is only admissible if all basic variables are non-negative, i.e., if the right hand column of the reduced tableau is free of negative entries. This is the case in this example. At the initial stage, however, negative entries may come up; this indicates that diﬀerent initial basic variables should have been chosen. At later stages in the process, the selection rules for the basic variables will guarantee that an initially feasible tableau will remain feasible throughout the process. 
>  注意此时只有在基本变量都是非负的，即右边列没有负元素的情况下，该解才是合法的
>  如果出现了负元素，说明应该选取不同的初始基本变量

## Step 4: Choose new basic variables 
If, at this stage, the objective function row has at least one negative entry, the cost can be lowered by making the corresponding variable basic. This new basic variable is called the entering variable . Correspondingly, one formerly basic variable has then to become non basic, this variable is called the leaving variable . We use the following standard selection rules. 
>  完成高斯消元后，如果目标函数一行至少存在一个负条目，我们可以将对应的变量设定为基本变量，以进一步降低目标函数
>  这个新的基本变量被称为进入变量
>  相应地，原来的某个基本变量需要变为非基本变量，该变量称为离开变量
>  选择进入变量和离开变量的规则是：

(i) The entering variable shall correspond to the column which has the most negative entry in the cost function row. If all cost function coeﬃcients are non-negative, the cost cannot be lowered and we have reached an optimum. The algorithm then terminates. 
>  进入变量所在的那一列在目标函数那一行应该有最小的负数值
>  如果目标函数行所有的系数都是非负的，则我们已经达到了最小值，算法停止

(ii) Once the entering variable is determined, the leaving variable shall be chosen as follows. Compute for each row the ratio of its right hand coeﬃcient to the corresponding coeﬃcient in the entering variable column. Select the row with the smallest ﬁnite positive ratio. The leaving variable is then determined by the column which currently owns the pivot in this row. If all coeﬃcients in the entering variable column are non-positive, the cost can be lowered in definitely, i.e., the linear programming problem does not have a ﬁnite solution. The algorithm then also terminates. 
>  确定了进入变量后，就需要选择离开变量
>  对于每一行，计算其右侧系数和进入变量列系数的比值，选择最小的有限正比值所对应的那一行，离开变量就是当前行的主元
>  如果进入变量那一列所有的系数都是非正的，则目标函数可以被无限降低，即线性规划问题没有有限解，算法停止

If entering and leaving variable can be found, go to Step 3 and iterate. 
>  确定了进入变量和离开变量后，我们回到 Step 3，执行高斯消元
>  然后继续迭代

Note that choosing the most negative coeﬃcient in rule (i) is only a heuristic for choosing a direction of fast decrease of the objective function. Rule (ii) ensures that the new set of basic variables remains feasible. 
>  rule (i) 中选择目标函数行最负的一列只是用于目标函数快速下降方向的一个启发式方法
>  rule (ii) 确保新的基本变量集合是可行的

Let us see how this applies to our problem. The previous tableau holds the most negative cost function coeﬃcient in column 3, thus $u$ shall be the entering variable (marked in boldface). The smallest positive ratio of right hand column to entering variable column is in row 3, as $\textstyle{\frac{3}{1}}<{\frac{5}{1}}$ . The pivot in this row points to $s_{2}$ as the leaving variable. Thus, after going through the Gaussian elimination once more, we arrive at 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/405d6cd34ac52938a1b5a404cefcab7f4e1cae9f4f827274d7491a83e7699d62.jpg) 

At this point, the new entering variable is $x_2$ corresponding the only negative entry in the last row. The leaving variable is $s_3$. After Gaussian elimination, we find

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/6c9d03419376c911b0c61b29cb3a39e8e5394f6aafe12a7965ce7fb729f25f90.jpg) 

Since there is no more negative entry in the last row, the cost cannot be lowered by choosing a diﬀerent set of basic variables; the termination condition applies. 
>  最后一行没有负项时，我们不能通过选择新的基本变量集合再降低目标函数，算法终止

## Step 5: Read off the solution 
The solution represented by the ﬁnal tableau has all non-basic variables set to zero, while the values for the basic variables can be can be read oﬀthe right hand column. The bottom right corner gives the negative of the objective function. 
>  最后读出解
>  最后的解中，所有非基本变量的值都设定为零，基本变量的值就是它们作为主元的那一行的右边系数
>  右底角就是目标函数的负值

In our example, the solution reads $\textstyle x_{1}={\frac{14}{3}}$ , $\textstyle x_{2}={\frac{2}{3}}$ , $\textstyle x_{3}=u={\frac{13}{3}}$ , $s_{1}=5$ , $v=s_{2}=s_{3}=0$ , which corresponds to $\zeta=-19$ , which can be independently checked by plugging the solution back into the objective function. 

As a further check, we note that the solution must satisfy (2b), (2d), and (2e) with equality and (2c) with a slack of 5. This can also be checked by direct computation. 

# 2 Initialization 
For some problem it is not obvious which set of variables form a feasible initial set of basic variables. For large problems, a trial-and-error approach is prohibitively expensive due the rapid growth of $\textstyle{\binom{n}{m}}$ , the number of possibilities to choose $m$ basic variables out of a total of n variables, as $m$ and $n$ become large. This problem can be overcome by adding a set of $m$ artiﬁcial variables which form a trivial set of basic variables and which are penalized by a large coeﬃcients $\omega$ in the objective function. This penalty will cause the artiﬁcial variables to become non-basic as the algorithm proceeds. 
We explain the method by example. For the problem 

$$
{\mathrm{minimize~}}z=x_{1}+2\,x_{2}+2\,x_{3}
$$ 

subject to 

$$
\begin{array}{c}{{x_{1}+x_{2}+2\,x_{3}+x_{4}=5\,,}}\\ {{{}}}\\ {{x_{1}+x_{2}+x_{3}-x_{4}=5\,,}}\\ {{{}}}\\ {{x_{1}+2\,x_{2}+2\,x_{3}-x_{4}=6\,,}}\\ {{{}}}\\ {{{\mathbfit{x}\geq\mathbf0\,,}}}\end{array}
$$ 

we set up a simplex tableau with three artiﬁcial variables which are initially basic: 

$$
{\begin{array}{r l}&{{\frac{\quad a_{1}\quad a_{2}\quad a_{3}\quad x_{1}\quad x_{2}\quad x_{3}\quad x_{4}~}{~}}}\\ &{{\begin{array}{c c c c c c c}{{\overline{{1}}}}&{{0}}&{{0}}&{{1}}&{{1}}&{{2}}&{{1}}&{{5}}\\ {{0}}&{{1}}&{{0}}&{{1}}&{{1}}&{{1}}&{{-1}}&{{5}}\\ {{0}}&{{0}}&{{1}}&{{1}}&{{2}}&{{2}}&{{-1}}&{{6}}\\ {{\omega}}&{{\omega}}&{{\omega}}&{{1}}&{{2}}&{{2}}&{{0}}&{{0}}\end{array}}}\end{array}}
$$ 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/25355d4d77e2a2ccda4f1e05cf7e5144b2726155cbf8e66ae3640b030a1b6f5d.jpg) 
We proceed as before by ﬁrst eliminating the nonzero entries below the pivots: 

Since, for $\omega$ large, $2-5\omega$ is the most negative coeﬃcient in the objective function row, $x_{3}$ will be entering and, since $\textstyle{\frac{5}{2}}<{\frac{6}{2}}<{\frac{5}{1}}$ , $a_{1}$ will be leaving. The Gaussian elimination step then yields 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/949374b5ec9e3916af4cbf070c0b28303578bb617c0d4a6316835304786d60d5.jpg) 

Now $x_{2}$ is entering, $a_{3}$ is leaving, and we obtain 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/399ee3acfdeb69922910cc7fbca5fab90bb9ce0f56fae0b23be59599c853d6f1.jpg) 

The new entering variable is $x_{1}$ while the criterion for the leaving variable is tied between $a_{2}$ and $x_{3}$ . Since we want the artiﬁcial variable to become nonbasic, we take $a_{2}$ to be leaving. (Choosing $x_{3}$ as the leaving variable would lead us to the same solution, albeit after a few extra steps.) We obtain 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/0bd78f04064d2e1f8e8b2d66d616d10e486bafb6116567b6791868ea011349f5.jpg) 

The termination condition is now satisﬁed, and we see that the solution is $z=6$ with $x_{1}=4$ , $x_{2}=1$ , $x_{3}=0$ , $x_{4}=0$ . 

We close with two remarks. 

• When using a computer to perform the simplex algorithm numerically, $\omega$ should be chosen large (one or two orders of magnitude larger than any of the other coeﬃcients in the problem) but not too large (to avoid loss of signiﬁcant digits in ﬂoating point arithmetic). 
• If not all artiﬁcial variables become non-basic, $\omega$ must be increased. If this happens for any value of $\omega$ , the feasible region is empty. 
• In the ﬁnal tableau, the penalty parameter $\omega$ can only appear in artiﬁcial variable columns. 

# 3 Duality 
The concept of duality is best motivated by an example. Consider the following transportation problem. Some good is available at location $A$ at no cost and may be transported to locations $B$ , $C$ , and $D$ according to the following directed graph: 


![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/7ead082d8b7ec017ebad2ee9d29f6c6610177bf3f6ead58fcb67b95c0667b973.jpg) 

On each of the unidirectional channels, the unit cost of transportation is $c_{j}$ for $j=1,\dots, 5$ . At each of the vertices $b_{\alpha}$ units of the good are sold, where $\alpha=B, C, D$ . How can the transport be done most eﬃciently? 

A ﬁrst, and arguably most obvious way of quantifying eﬃciency would be to state the question as a minimization problem for the total cost of transportation. If $x_{j}$ denotes the amount of good transported through channel $j$ , we arrive at the following linear programming problem: 

$$
{\mathrm{minimize~}}c_{1}\,x_{1}+\cdot\cdot\cdot+c_{5}\,x_{5}
$$ 

subject to 

$$
\begin{array}{c}{{x_{1}-x_{3}-x_{4}=b_{B}\,,}}\\ {{{}}}\\ {{x_{2}+x_{3}-x_{5}=b_{C}\,,}}\\ {{{}}}\\ {{x_{4}+x_{5}=b_{D}\,.}}\end{array}
$$ 

The three equality constraints state that nothing gets lost at nodes $B$ , $C$ , and $D$ except what is sold. 

There is, however, a second, seemingly equivalent way of characterizing eﬃciency of transportation. Instead of looking at minimizing the cost of transportation, we seek to maximize the income from selling the good. Letting $y_{\alpha}$ denote the unit price of the good at node $\alpha=A,\ldots, D$ with $y_{A}=0$ by assumption, the associated linear programming problem is the following: 

$$
{\mathrm{maximize}}\ y_{B}\,b_{B}+y_{C}\,b_{C}+y_{D}\,b_{D}
$$ 
subject to 

$$
\begin{array}{r l r}&{}&{y_{B}-y_{A}\leq c_{1}\,,}\\ &{}&{y_{C}-y_{A}\leq c_{2}\,,}\\ &{}&{y_{C}-y_{B}\leq c_{3}\,,}\\ &{}&{y_{D}-y_{B}\leq c_{4}\,,}\\ &{}&{y_{D}-y_{C}\leq c_{5}\,.}\end{array}
$$ 

The inequality constraints encode that, in a free market, we can only maintain a price diﬀerence that does not exceed the cost of transportation. If we charged a higher price, then “some local guy” would immediately be able to undercut our price by buying from us at one end of the channel, using the channel at the same ﬁxed channel cost, then selling at a price lower than ours at the high-price end of the channel. 

Setting 

$$
{\pmb x}=\left(\begin{array}{l}{x_{1}}\\ {\vdots}\\ {x_{5}}\end{array}\right)\,,\qquad{\pmb y}=\left(\begin{array}{l}{y_{B}}\\ {y_{C}}\\ {y_{D}}\end{array}\right)\,,\qquad\mathrm{and~}A=\left(\begin{array}{l l l l l}{1}&{0}&{-1}&{-1}&{0}\\ {0}&{1}&{1}&{0}&{-1}\\ {0}&{0}&{0}&{1}&{1}\end{array}\right)\,,
$$ 

we can write (5) as the abstract primal problem 

$$
{\begin{array}{r l}&{{\mathrm{minimize~}}\mathbf{c}^{T}\mathbf{\boldsymbol{x}}}\\ &{{\mathrm{subject~to~}}A\mathbf{\boldsymbol{x}}=\mathbf{\boldsymbol{b}},\mathbf{\boldsymbol{x}}\geq\mathbf{0}\,.}\end{array}}
$$ 

Likewise, (6) can be written as the dual problem 

$$
\begin{array}{r l}&{\mathrm{maximize}\ \pmb{y}^{T}\pmb{b}}\\ &{\mathrm{subject\to}\ \pmb{y}^{T}\pmb{A}\le\pmb{c}^{T}\,.}\end{array}
$$ 

We shall prove in the following that the minimal cost and the maximal income coincide, i.e., that the two problems are equivalent. 

Let us ﬁrst remark this problem is easily solved without the simplex algorithm: clearly, we should transport all goods sold at a particular location through the cheapest channel to that location. Thus, we might perform a simple search for the cheapest channel, something which can be done eﬃ- ciently by combinatorial algorithms such as Dijkstra’s algorithm [2]. The advantage of the linear programming perspective is that additional constraints such as channel capacity limits can be easily added. For the purpose of understanding the relationship between the primal and the dual problem, and for understanding the sign i can ce of the dual formulation, the simple present setting is entirely adequate. 
The unknowns $_{_{\pmb{x}}}$ in the primal formulation of the problem not only identify the vertex of the feasible region at which the optimum is reached, but they also act as sensitivity parameters with regard to small changes in the cost coeﬃcients $_\mathrm{c}$ . Indeed, when the linear programming problem is nondegenerate, i.e. has a unique optimal solution, changing the cost coeﬃcients from $_\mathrm{c}$ to $c+\Delta c$ with $|\Delta c|$ suﬃciently small will not make the optimal vertex jump to another corner of the feasible region, as the cost depends continuously on $_\mathrm{c}$ . Thus, the corresponding change in cost is $\Delta c^{T}x$ . If $x_{i}$ is nonbasic, the cost will not react at all to small changes in $c_{i}$ , whereas if $x_{i}$ is large, then the cost will be sensitive to changes in $c_{i}$ . This information is often important because it gives an indication where to best spend resources if the parameters of the problem—in the example above, the cost of transportation—are to be improved. 

Likewise, the solution vector $\textbf{\em y}$ to the dual problem provides the sensitivity of the total income to small changes in $^{b}$ . Here, $^{b}$ is representing the number of sales at the various vertices of the network; if the channels were capacity constrained, the channel limits were also represented as components of $^{b}$ . Thus, the dual problem is providing the answer to the question “if I were to invest in raising sales, where should I direct this investment to achieve the maximum increase in income?” 

The following theorems provide a mathematically precise statement on the equivalence of primal and dual problem. 

Theorem 1 (Weak duality) . Assume that x is a feasible vector for the primal problem (8) and $\textbf{\em y}$ is a feasible vector for the dual problem (9) . Then 

# (i) $\pmb{y}^{T}\pmb{b}\leq\pmb{c}^{T}\pmb{x}$ ; 

(ii) if (i) holds with equality, then $\mathbfit{w}$ and $\textbf{\em y}$ are optimal for their respective linear programming problems; (iii) the primal problem does not have a ﬁnite minimum if and only if the feasible region of the dual problem is empty; vice versa, the dual problem does not have a ﬁnite maximum if and only if the feasible region of the primal problem is empty. The proof is simple and shall be left as an exercise. To proceed, we say that $\mathbfit{w}$ is a basic feasible solution of $A{\pmb x}={\pmb b}$ , ${\pmb x}\geq0$ if it has at most $m$ nonzero components. We say that it is non degenerate if it has exactly $m$ nonzero components. If, in the course of performing the simplex algorithm, we hit a degenerate basic feasible solution, it is possible that the objective function row in the simplex tableau contains negative coeﬃcients, yet the cost cannot be lowered because the corresponding basic variable is already zero. This can lead to cycling and thus non-termination of the algorithm. We shall not consider the degenerate case further. 
When $_{_{\pmb{x}}}$ is a non degenerate solution to the primal problem (8), i.e., $_{_{\pmb{x}}}$ is non degenerate basic feasible and also optimal, then we can be assured that the simplex method terminates with all coeﬃcients in the objective function row non negative. (If they were not, we could immediately perform at least one more step of the algorithm with strict decrease in the cost.) In this situation, we can use the simplex algorithm as described to prove the following stronger form of the duality theorem. 

Theorem 2 (Strong duality) . The primal problem (8) has an optimal solution $\mathbfit{w}$ if and only if the dual problem (9) has an optimal solution $\textbf{\em y}$ ; in this case $\pmb{y}^{T}\pmb{b}=\pmb{c}^{T}\pmb{x}$ . 

Proof. We only give a proof in the case when the solution to the primal problem is non-degenerate. It is based on a careful examination of the termination condition of the simplex algorithm. Assume that $\mathbfcal{W}$ solves the primal problem. Without loss of generality, we can reorder the variables such that the ﬁrst $m$ variables are basic, i.e. 

$$
\pmb{x}=\binom{\pmb{x}_{B}}{\pmb{0}}
$$ 

and that the ﬁnal simplex tableau reads 

$$
\begin{array}{r}{\left(\begin{array}{l l}{\boldsymbol{I}}&{\boldsymbol{R}}\\ {\mathbf{0}^{T}}&{\mathbf{r}^{T}}\end{array}\bigg|\mathbf{x}_{B}\right)}\end{array}.
$$ 

The last row represents the objective function coeﬃcients and $z$ denotes the optimal value of the objective function. We note that the termination conditi of the simplex algorithm reads $\mathbfit{r}\geq\mathbf0$ . We now partition the initial matrix A and the coeﬃcients of the objective function $\mathbf{c}$ into their basic and nonbasic components, writing 

$$
A=\left(B\quad N\right)\qquad\mathrm{and}\qquad{\pmb c}=\left({\pmb c}_{N}\right)\,.
$$ 
Finally, it can be shown that the elementary row operations used in the Gaussian elimination steps of the simplex algorithm can be written as multip li cation by a matrix from the left, which we also partition into components compatible with the block matrix structure of (11), so that the transformation from the initial to the ﬁnal tableau can be written as 

$$
\begin{array}{r}{\left(\begin{array}{c c}{\boldsymbol{M}}&{\boldsymbol{\mathbf{v}}}\\ {\boldsymbol{\mathbf{u}}^{T}}&{\alpha}\end{array}\right)\left(\begin{array}{c c}{\boldsymbol{B}}&{\boldsymbol{N}}\\ {\boldsymbol{c}_{\boldsymbol{B}}^{T}}&{\boldsymbol{c}_{\boldsymbol{N}}^{T}}\end{array}\bigg|\begin{array}{c}{\boldsymbol{b}}\\ {0}\end{array}\right)=\left(\begin{array}{c c}{\boldsymbol{M}\boldsymbol{B}+\boldsymbol{\mathbf{v}}\boldsymbol{\mathbf{c}}_{\boldsymbol{B}}^{T}}&{\boldsymbol{M}\boldsymbol{N}+\boldsymbol{\mathbf{v}}\boldsymbol{\mathbf{c}}_{\boldsymbol{N}}^{T}}\\ {\boldsymbol{\mathbf{u}}^{T}\boldsymbol{B}+\alpha\boldsymbol{\mathbf{c}}_{\boldsymbol{B}}^{T}}&{\boldsymbol{\mathbf{u}}^{T}\boldsymbol{N}+\alpha\boldsymbol{\mathbf{c}}_{\boldsymbol{N}}^{T}}\end{array}\bigg|\begin{array}{c}{\boldsymbol{M}\boldsymbol{b}}\\ {\boldsymbol{\mathbf{u}}^{T}\boldsymbol{b}}\end{array}\right)\;.}\end{array}
$$ 

We now compare the right hand side of (13) with (11) to determine the coeﬃcients of the left hand matrix. First, we note that in the simplex algorithm, none of the Gaussian elimination steps on the equality constraints depend on the objective function coeﬃcients (other than the path taken from initial to ﬁnal tableau, which is not at issue here). This immediately implies that $\mathbf{\nabla}v=\mathbf{0}$ . Second, we observe that nowhere in the simplex algorithm do we ever rescale the objective function row. This immediately implies that $\alpha=1$ . This leaves us with the following set of matrix equalities: 

$$
\begin{array}{r}{M\boldsymbol{B}=\boldsymbol{I}\,,\qquad}\\ {M\pmb{b}=\pmb{x}_{B}\,,\qquad}\\ {\pmb{u}^{T}\boldsymbol{B}+\pmb{c}_{B}^{T}=\mathbf{0}^{T}\,,\qquad}\\ {\pmb{u}^{T}\boldsymbol{N}+\pmb{c}_{N}^{T}=\pmb{r}^{T}\,.}\end{array}
$$ 

so that $M=B^{-1}$ and $\pmb{u}^{T}=-\pmb{c}_{B}^{T}B^{-1}$ . We now claim that 

$$
\pmb{y}^{T}=\pmb{c}_{B}^{T}B^{-1}
$$ 

solves the dual problem. We compute 

$$
\begin{array}{r l}{\pmb{y}^{T}A=\pmb{c}_{B}^{T}B^{-1}\left(B}&{N\right)}\\ {=\left(\pmb{c}_{B}^{T}}&{\pmb{c}_{B}^{T}B^{-1}N\right)}\\ {=\left(\pmb{c}_{B}^{T}}&{\pmb{c}_{N}^{T}-\pmb{r}^{T}\right)}\\ {\le\left(\pmb{c}_{B}^{T}}&{\pmb{c}_{N}^{T}\right)=\pmb{c}^{T}\,.}\end{array}
$$ 

This shows that $\textbf{\em y}$ is feasible for the dual problem. Moreover, 

$$
\pmb{y}^{T}\pmb{b}=\pmb{c}_{B}^{T}B^{-1}\pmb{b}=\pmb{c}_{B}^{T}\pmb{x}_{B}=\pmb{c}^{T}\pmb{x}\,.
$$ 

Thus, by weak duality, $\textbf{\em y}$ is also optimal for the dual problem. 

The reverse implication of the theorem follows from the above by noting that the bi-dual is identical with the primal problem. 

# References 
[1] G.B. Dantzig, M.N. Thapa, “Linear Programming,” Springer, New York, 1997.

 [2] “Dijkstra’s algorithm.” Wikipedia, The Free Encyclopedia. April 4, 2012, 06:13 UTC. http://en.wikipedia.org/w/index.php?title=Dijkstra%27 s algorithm&oldid=485462965

 [3] D. Gale, Linear Programming and the simplex method , Notices of the AMS 54 (2007), 364–369. http://www.ams.org/notices/200703/fea-gale.pdf

 [4] P. Pedregal, “Introduction to optimization,” Springer, New York, 2004. 
