# The Idea
大 O 表示法是在P. Bachmann 于1892年出版的《Analytische Zahlentheorie》一书中引入的，他使用“$x\  \text{is}\  O(\frac n 2)$”来表达“$x \approx \frac n 2$”的概念

这种表示法非常适合比较算法效率 (algorithm efficiencies)，因为我们想要表达的是，给定算法的工作量增长 (the growth of effort of a given algorithm) 近似于一个标准函数的形状 (approximates the shape of a standard function)
# The Definitions
大 O ($O()$) 是五种标准渐近符号之一，在实践中，大 O 被用作算法工作量增长的紧密上界 (tight upper-bound on the growth of an algorithm's effort)(该工作量由函数 $f(n)$ 描述)，尽管它也可以是一个宽松的上界

为了更清楚地说明 $O()$ 作为紧密上界的角色，小 o ($o()$) 符号被用来描述一个不是紧密的上界

**Definition** (Big-O, $O()$)：令 $f(n)$ 和 $g(n)$ 是将正整数 (positive integers) 映射到正实数 (positive real numbers) 的函数，如果存在一个实常数 (real constant) $c >0$，以及存在一个整数常数 (integer contant) $n_0\ge 1$，使得 $f(n) \le c\times g(n)$ 对于所有满足 $n\ge n_0$ 的整数 $n$ 都成立，我们就说 $f(n)$ 是 $O(g(n))$ 的(或 $f(n) \in O(g(n))$)

**Definition** (Little-o, $o()$)：令 $f(n)$ 和 $g(n)$ 是将正整数映射到正实数的函数，如果对于任意实常数 $c >  0$，存在一个整数常数 $n_0\ge 1$，使得 $f(n) < c\times g(n)$ 对于所有满足 $n\ge n_0$ 的整数 $n$ 都成立，我们就说 $f(n)$ 是 $o(g(n))$ 的 (或 $f(n)\in o(g(n))$)

(
显然对于 $O()$ 的情况，$g(n)$ 只需要增长速率匹配 $f(n)$，就可以满足 $f(n)\in O(g(n))$，并不要求 $g(n)$ 严格大于 $f(n)$，只要找到一个足够大的常数 $c$ 放缩 $g(n)$ 使得 $f(n)\le c\times g(n)$ 即可，因此 $g(n)$ 是较为紧密的

对于 $o()$ 的情况，显然 $g(n)$ 需要在 $n$ 较大时严格并且远远大于 $f(n)$，才可以满足对于任意小的 $c>0$ 都有 $f(n)< c\times g(n)$，以使得 $f(n)\in o(g(n))$，因此 $g(n)$ 是较为宽松的
)

在 $f(n)$ 的另一边，我们可以方便地定义 $O()$ 和 $o()$ 的平行符号，以提供 $f(n)$ 增长的紧密和宽松下界，大 $\Omega$ ($Ω()$) 是紧密下界符号，小 $\omega$ ($ω()$)则描述了宽松下界

**Definition** (Big-Omega $\Omega()$)：令 $f(n)$ 和 $g(n)$ 是将正整数映射到正实数的函数，如果存在一个实常数 $c >0$，以及存在一个整数常数 $n_0\ge 1$，使得 $f(n) \ge c\times g(n)$ 对于所有满足 $n\ge n_0$ 的整数 $n$ 都成立，我们就说 $f(n)$ 是 $\Omega(g(n))$ 的 (或 $f(n) \in \Omega(g(n))$)

**Definition** (Little-Omega, $\omega()$)：令 $f(n)$ 和 $g(n)$ 是将正整数映射到正实数的函数，如果对于任意实常数 $c >  0$，存在一个整数常数 $n_0\ge 1$，使得 $f(n) > c\times g(n)$ 对于所有满足 $n\ge n_0$ 的整数 $n$ 都成立，我们就说 $f(n)$ 是 $\omega(g(n))$ 的 (或 $f(n)\in \omega(g(n))$)

(
显然对于 $\Omega()$ 的情况，$g(n)$ 只需要增长速率匹配 $f(n)$，就可以满足 $f(n)\in \Omega(g(n))$，并不要求 $g(n)$ 严格小于 $f(n)$，只要找到一个足够小的常数 $c$ 放缩 $g(n)$ 使得 $f(n)\ge c\times g(n)$ 即可，因此 $g(n)$ 是较为紧密的

对于 $\omega()$ 的情况，显然 $g(n)$ 需要在 $n$ 较大时严格并且远远小于 $f(n)$，才可以满足对于任意大的 $c>0$ 都有 $f(n)> c\times g(n)$，以使得 $f(n)\in \omega(g(n))$，因此 $g(n)$ 是较为宽松的
)

这些符号之间的关系如图所示：
![[Aysmptotic Notations.png]]
这些定义的限制总结于如下表格：

| Definition |  ? $c>0$  | ? $n_0\ge 1$ | $f(n)$ ? $c\cdot g(n)$ |
|:----------:|:---------:|:------------:|:----------------------:|
|   $O()$    | $\exists$ |  $\exists$   |         $\le$          |
|   $o()$    | $\forall$ |  $\exists$   |          $<$           |
| $\Omega()$ | $\exists$ |  $\exists$   |         $\ge$          |
| $\omega()$ | $\forall$ |  $\exists$   |          $>$           |

虽然 $\Omega()$ 和 $\omega()$ 不常用于描述算法，但我们可以基于它们 (尤其是 $\Omega()$) 来定义一种符号来描述 $O()$ 和 $\Omega()$ 的组合：大 $\Theta$ ($\Theta()$)
当我们说一个算法是 $\Theta(g(n))$ 时，我们是在说 $g(n)$ 既是算法工作量增长的紧密上界，也是下界

**Definition** (Big-Theta, $\Theta()$)：令 $f(n)$ 和 $g(n)$ 是将正整数映射到正实数的函数，当且仅当 $f(n)\in O(g(n))$ 且 $f(n) \in \Omega(g(n))$ 时，我们说 $f(n)$ 是 $\Theta(g(n))$ 的 (或 $f(n)\in \Theta(g(n))$)
# Appendix A
紧密上界：一个紧密上界是尽可能接近函数实际增长速率的上界估计，它不仅提供了一个上限，而且这个上限非常接近函数的实际值，尤其是在输入值较大时

例如一个算法的运行时间是 $T(n) = 2n^2 + 3n$，那么 $T(n)$ 的紧密上界可以是 $O(n^2)$，因为对于大的 $n$ 值，$3n$ 相对于 $2n^2$ 变得不重要，主要的增长贡献来自 $n^2$ 项

宽松上界：一个宽松上界是一个较为宽泛的上界估计，它可能远远大于函数的实际增长速率。这种估计通常更容易获得，但不如紧密上界精确

例如一个算法运行时间是 $T(n) = 2n^2 + 3n$，则它的一个宽松上界可以是 $O(n^2 + n)$ 甚至 $O(n^3)$，这些估计都正确，但它们没有 $O(n^2)$ 那样接近实际的增长速率