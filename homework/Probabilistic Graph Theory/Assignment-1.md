Student: 陈信杰
Student ID:  2024316427

# Problem 1

(1) The left hand side is $P (A, B\mid C) = P (A\mid C) P (B\mid C)$ , which indicates

$$
\begin{align}
P(A,B\mid C) &= \frac {P(A,B,C)}{P(C)}\\
&= P(A\mid C)P(B\mid C)\\
&=\frac {P(A,C)}{P(C)}\frac {P(B,C)}{P(C)},\\
\end{align}
$$

so the left hand side is equivalent to

$$
P(A,B,C)P(C) =  P(A,C)P(B,C).
$$

The right hand side is $P (A\mid B, C) = P (A\mid C)$ , which indicates

$$
\begin{align}
P(A\mid B,C) &= \frac {P(A,B,C)}{P(B,C)}\\
&=\frac {P(A,C)}{P(C)},\\
\end{align}
$$

so the right hand side is equivalent to 

$$
P (A, B, C) P (C) = P (A, C) P (B, C).
$$

Therefore, the left hand side is equivalent to the right hand side.

(2) left hand side: 

$$
P (A = a, B\mid C) = \frac {P(B, C,A=a)}{P(C)}
$$

right hand side: 

$$
P(B\mid C) = \frac {P(B, C)}{P(C)}
$$

Suppose lhs = rhs, then we can derive

$$
P(B, C) = P(B, C, A=a).\tag{1.1}
$$

Denote event $X$ as $X = \{B = b_i \cap C = c_j\} \text{ for all possible values } b_i, c_j \text{ that } B, C \text{ can take}.$
Denote event $Y$ as
$Y = \{A = a\}.$

Therefore, (1.1) indicates that $P (X) = P (X\cap Y)$ , which means $X \subseteq Y$.

Obviously, $X\subseteq Y$ is generally not true, so we can conclude that
$P (A = a, B\mid C) \ne P (B\mid C)$

(3) left hand side:

$$
\begin{align}
P(Z\mid X) &= \frac {P(Z,X)}{P(X)}\\
&=\frac {\sum_Y P(Z,X \mid Y)}{\sum_Y P(X\mid Y)}
\end{align}
$$

right hand side:

$$
\sum_Y P(Z\mid X, Y) = \sum_Y \frac {P(Z,X\mid Y)}{P(X\mid Y)}
$$

Suppose lhs = rhs, then we can derive

$$
\begin{align}
\frac {\sum_Y P(Z,X\mid Y)}{\sum_Y P(X\mid Y)}&=\sum_Y \frac {P(Z,X\mid Y)}{P(X\mid Y)},
\end{align}
$$

which is not always true.

Therefore, we can conclude $P (Z\mid X) \ne \sum_Y P (Z\mid X, Y)$

(4) The right hand side is

$$
\begin{align}
&\frac {P(X\mid Y)P(Y)}{\sum_Y P(X\mid Y)P(Y)}\\
=& \frac {P(X,Y)}{\sum_Y P(X,Y)}\\
=&\frac {P(X,Y)}{P(X)}\\
=&P(Y\mid X),
\end{align}
$$

which is equal to the left hand side.

# Problem 2
Denote the $K$ doors as $d_1, d_2, \dots, d_k$, and denote the event that we chose the bonus door as $A$. Obviously, we have $P (A) = \frac 1 k$ and $P (\bar A) = \frac {k-1} k$.

(1) If the host knew the location of the bonus
Denote the event that the host opened $d_j$ as $B_j$ (assume $d_j$ is not the bouns door).

Now, we are interested at the probability $P (A \mid B_j)$, according to the Baysian rule:

$$
\begin{align}
P(A\mid B_j) &= \frac {P(A, B_j)}{P(B_j)}\\
&=\frac {P(B_j \mid A)P(A)}{P(B_j, A) + P(B_j, \bar A)}\\
&=\frac {P(B_j \mid A)P(A)}{P(B_j\mid A)P(A) + P(B_j\mid \bar A)P(\bar A)}\\
&=\frac {\frac {1}{k-1}\cdot \frac 1 k}{\frac 1 {k-1}\cdot \frac 1 k + \frac 1 {k-2}\cdot \frac {k-1}{k}}\\
&=\frac 1 {1 + \frac {(k-1)^2}{k-2}}
\end{align}
$$

Obvisouly, $\frac {(k-1)^2}{k-2} > \frac {(k-1)^2}{k-1} = k-1$, thus $1 + \frac {(k-1)^2}{k-2} > 1 + k-1 = k$ , which indicates $P (A \mid B_j) < P (A)$.

Therefore, after the host opened a door, if we do not change the door, the probability that we won is $P (A \mid B_j) < \frac 1 k$.

If we change the door, the probability that we won is

$$
\begin{align}
&P(A)\cdot 0 + P(\bar A)\cdot  \frac {1}{k-2}\\
=& \frac {k-1}{k} \cdot \frac 1 {k-2}\\
>&\frac {k-1}{k} \cdot \frac 1 {k-1}\\
=&\frac 1 k
\end{align}
$$

Therefore, we should change a door.

(2) If the host did not konw the location of the bouns
If we did not change the door, the probability that we won is $P (A) = \frac 1 k$.

If we changed the door, the probability that we won is

$$
\begin{align}
&P(A)\cdot 0 + P(\bar A)\cdot \frac {1}{k-2}\\
=& \frac {k-1} k \cdot \frac 1 {k-2}\\
>&\frac {k-1}k \cdot \frac 1 {k-1}\\
=&\frac 1k
\end{align}
$$

Therefore, we should change a door.

# Problem 3
Following our commen sense, the result of each coin tossing is independent from each other, thus the probability that the 101-th experiment get "upward" should still be $0.5$.

However, it is stated that the coin is a "special coin". Therefore, considering not following our commen sense, we denote the event that the coin tossing get "upward" is $A$, and denote $P (A) = \theta \;(0\le \theta \le 1)$.

According to maximumu likelihood estimate, we should find a $\theta$ that

$$
\arg \max_{\theta}f(\theta) = \theta^{55}(1-\theta)^{45},
$$

which is equivalent to

$$
\arg \max_{\theta}\ln f(\theta) = 55\ln\theta +  45\ln(1-\theta).
$$

Thus

$$
\begin{align}
\frac {\partial \ln f(\theta)}{\partial\theta} = \frac {55}{\theta}- \frac {45}{1-\theta}.\tag{3.1}
\end{align}
$$

Let (3.1) equalts to $0$, we get

$$
\begin{align}
\frac {55}{\theta} &= \frac {45}{1-\theta}\\
45\theta &= 55 - 55\theta\\
\theta &=0.55
\end{align}
$$

Therefore, according to MLE, $\theta$ should be $0.55$.

# Problem 4
According to chain rule, we can decompose $P (X, Y, Z)$ as
$$
P(X,Y,Z) = P(X,Z\mid Y)P(Y) = P(Z\mid X, Y) P(X\mid Y) P(Y),\tag{4.1}
$$

or

$$
P(X,Y,Z) = P(Y,Z \mid X)P(X) = P(Z\mid X, Y) P(Y\mid X) P(X).\tag{4.2}
$$

We already konw

$$
P(X,Y,Z) = P(X)P(Y)P(Z\mid X ,Y).\tag{4.3}
$$

From (4.1)-(4.3), we can easily conclude that $X, Y$ are marginally independent.

(1) We aims to check that $P (X, Y\mid Z) = P (X\mid Z) P (Y\mid Z)$ does not hold.

If we equate LHS with RHS:

$$
\begin{align}
P(X,Y\mid Z) &= P(X\mid Z)P(Y\mid Z)\\
P(X\mid Y, Z)P(Y\mid Z) &= P(X\mid Z)P(Y\mid Z)\\
P(X\mid Y, Z) &= P(X\mid Z),
\end{align}
$$

which does not necessarily hold.

Therefore, $P (X, Y\mid Z) = P (X\mid Z) P (Y\mid Z)$ does not necessarily hold. Generally, given $Z$, $X$ and $Y$ are dependent.

(2) Equate (4.1) with (4.3) and equate (4.2) with (4.3), we can easily conclude $P (X, Y) = P (X) P (Y)$ , which means that $X, Y$ are marginally independent.

# Problem 5
Let $\mathcal{K}=(\mathcal{X},\mathcal{E})$ , and let $X\subset{\mathcal{X}}$ , we define the induced subgraph $\mathcal{K}[X]$ to be the graph $(X,{\mathcal{E}}^{\prime})$ where $\mathcal{E}^{\prime}$ are all the edges $X\rightleftharpoons Y\in{\mathcal{E}}$ such that $X, Y\in X$ . 

clique: A subgraph over $X$ is complete if every two nodes in $X$ are connected by some edge. The set $X$ is often called a clique. 

maximal clique: We say that a clique $X$ is maximal if for any superset of nodes $Y\supset X$ , $Y$ is not a clique. 

upward closure: We say that a subset of nodes $X \subset \mathcal X$ is upwardly closed in $\mathcal K$ if, for any $x\in X$, we have $\text{Boundary}_x \subset X$. We define the upward closure of $X$ to be the minimal upwardly closed subset $Y$ that contains $X$. 
($\text{Boundary}_x$ means the set that contains all the nodes that directly are neighboring node $x$ in graph $\mathcal K$)

# Problem 6
From the naive Baysian model, we can derive:
$P (C, X, Y) = P (X, Y\mid C) P (C) = P (X\mid C) P (Y\mid C) P (C)$.

Therefore,

$$
\begin{align}
P (X = 0 \mid Y = 1) &= \frac {P (X = 0, Y = 1)}{P(Y=1)}\\
&=\frac {\sum_C P(X=0,Y=1\mid C)P(C)}{\sum_CP(Y=1\mid C)P(C)}\\
&=\frac {\sum_C P(X=0\mid C)P(Y=1\mid C)P(C)}{\sum_CP(Y=1\mid C)P(C)}\\
&=\frac {0.6 * 0.5 * 0.3 + 0.4 * 0.6 * 0.5}{0.6* 0.3 + 0.4 * 0.5}\\
&=\frac {0.09 + 0.12}{0.18 + 0.2}\\
&=\frac {0.21}{0.38}
\end{align}
$$

# Problem 7
My reseach focus is about large language models. I choose this course because the probability graph theory introduced in this course is likely to assist my understanding about the underlying language modeling mechanism of large language models.