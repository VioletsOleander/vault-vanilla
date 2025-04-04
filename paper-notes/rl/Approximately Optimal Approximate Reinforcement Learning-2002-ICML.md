# Abstract 
In order to solve realistic reinforcement learning problems, it is critical that approximate algorithms be used. In this paper, we present the conservative policy iteration algorithm which finds an “approximately” optimal policy, given access to a restart distribution (which draws the next state from a particular distribution) and an approximate greedy policy chooser. 
>  本文提出 conservative policy iteration 算法，该算法在能够访问 restart distribution (从一个特定分布中抽取下一个状态) 的情况下找到近似最优的策略和一个近似贪心策略选择器

Crudely, the greedy policy chooser outputs a policy that usually chooses actions with the largest state-action values of the current policy, i.e. it outputs an “approximate” greedy policy. This greedy policy chooser can be implemented using standard value function approximation techniques. 
>  粗略地说，贪心策略选择器的输出策略的动作选择通常是当前策略中具有最大动作价值的动作
>  这一贪心策略选择器可以使用标准的价值函数近似技巧实现

 Under these assumptions, our algorithm: (1) is guaranteed to improve a performance metric (2) is guaranteed to terminate in a “small" number of timesteps and (3) returns an “approximately” optimal policy. The quantified statements of (2) and (3) depend on the quality of the greedy policy chooser, but not explicitly on the the size of the state space. 
 >  基于这些假设，我们的算法
 >  1. 保证提高策略的性能指标
 >  2. 保证在较少的时间步内终止
 >  3. 会返回一个近似最优策略
 >  关于 2, 3 的量化取决于贪心策略选择器的质量，但不明确依赖于状态空间的大小

# 1 Introduction 
The two standard approaches, greedy dynamic programming and policy gradient methods, have enjoyed many empirical successes on reinforcement learning problems. Unfortunately, both methods can fail to efficiently improve the policy. Approximate value function methods suffer from a lack of strong theoretical performance guarantees. We show how policy gradient techniques can require an unreasonably large number of samples in order to determine the gradient accurately. This is due to policy gradient method's fundamental intertwining of “exploration” and “exploitation". 
>  贪心动态规划和策略梯度方法这两大标准方法已经在许多 RL 问题上取得经验性的成功，但两种方法都无法高效改进策略
>  近似价值函数方法缺乏强的理论性能保证
>  我们将展示策略梯度方法为了准确确定梯度可能需要大量的样本，这是因为策略梯度方法本质上将 exploration 和 exploitation 交织在一起

In this paper, we consider a setting in which our algorithm is given access to a restart distribution and an “approximate” greedy policy chooser. Informally, the restart distribution allows the agent to obtain its next state from a fixed distribution of our choosing. Through a more uniform restart distribution, the agent can gain information about states that it wouldn't necessarily visit otherwise. Also informally, the greedy policy chooser is a “black box” that outputs a policy that on average chooses actions with large advantages with respect to the current policy, i.e. it provides an “approximate” greedy policy. This “black box" algorithm can be implemented using one of the well-studied regression algorithms for value functions (see [9, 3]). The quality of the resulting greedy policy chooser is then related to the quality of this “black box". 
>  本文中，我们考虑的设定是算法可以访问一个 restart distribution 和一个近似贪心策略选择器
>  restart distribution 允许 agent 从一个我们选择的固定分布中获取下一个状态，通过更均匀的 restart distribution, agent 可以获得一些原本可能不会访问的状态的信息 (exploration)
>  greedy policy chooser 是一个黑盒，它输出的策略在平均情况下会选择相较于当前策略具有较大优势的动作，即它提供了一个近似贪心策略，这一黑盒算法可以用价值函数的回归算法实现，故最终产生的 greedy policy chooser 的质量与这个黑盒的质量相关

Drawing upon the strengths of the standard approaches, we propose the conservative policy iteration algorithm. The key ingredients of this algorithm are: (1) the policy is improved in a more uniform manner over the state-space and (2) a more conservative policy update is performed in which the new policy is a mixture distribution of the current policy and the greedy policy. Crudely, the importance of (1) is to incorporate “exploration” and the importance of (2) is to avoid the pitfalls of greedy dynamic programming methods which can suffer from significant policy degradation by directly using “approximate” greedy policies. 
>  借鉴标准方法的优点，我们提出保守策略迭代算法，其关键要素包括
>  1. 策略在状态空间以更均匀的方式改进
>  2. 执行的策略更新更保守，新策略是当前策略和贪心策略的混合分布
>  粗略地说，1 的重要性在于引入了 exploration，2 的重要性在于避免贪心动态规划方法的缺陷，因为直接使用近似贪心策略可能导致策略显著退化

Our contribution is in proving that such an algorithm converges in a “small" number of steps and returns an "approximately” optimal policy, where the quantified claims do not explicitly depend on the the size of the state space. We first review the problems with standard approaches, then state our algorithm. 
>  我们的贡献在于证明了该算法可以以 "较少" 的步数收敛，并且返回 "近似" 最优的策略，而关于 "较少" 和 “近似” 的量化并不依赖于状态空间的大小
>  我们先回顾标准方法的问题，然后陈述我们的算法

# 2 Preliminaries 
A finite Markov decision process (MDP) is defined by the tuple $(S,D,A,\mathcal{R},\{P(s^{\prime};s,a)\})$ where: $S$ is a finite set of states, $D$ is the starting state distribution, $A$ is a finite set of actions, $\mathcal{R}$ is a reward function $\mathcal{R}:S\times A\rightarrow[0,R]$ , and $\left\{P(s^{\prime};s,a)\right\}$ are the transition probabilities, with $P(s^{\prime};s,a)$ giving the next-state distribution upon taking action $a$ in state $s$ 
>  以上定义了有限 MDP，其中 $D$ 是起始状态分布 

Although ultimately we desire an algorithm which uses only the given MDP $M$ , we assume access to a restart distribution, defined as follows: 

**Definition 2.1.** 
A $\mu$ restart distribution draws the next state from the distribution $\mu$ 

>  虽然我们最终希望算法仅使用给定的 MDP $M$，但我们假设算法可以访问一个 restart distribution，其定义如上
>  restart distribution $\mu$ 从分布 $\mu$ 中采样下一个状态

This restart distribution is a slightly weaker version of the generative model in [5]. As in [5], our assumption is considerably weaker than having knowledge of the full transition model. However, it is a much stronger assumption than having only “irreversible” experience, in which the agent must follow a single trajectory, with no ability to reset to obtain another trajectory from a state. If $\mu$ is chosen to be a relatively uniform distribution (not necessarily $D$ ),then this $\mu$ restart distribution can obviate the need for explicit exploration. 
>  restart distribution 是 [5] 中生成模型的一个稍弱的版本，与[5]一样，我们的假设比完全了解整个转移模型要弱得多
>  然而，该假设比仅具有 "不可逆" 经验的假设要强得多，在该情况下，智能体只能遵循单个轨迹，无法通过重置从某个状态获得另一条轨迹
>  如果 $\mu$ 选择为相对均匀的分布 (不一定是 $D$)，则 $\mu$ restart distribution 可以消除显式 exploration 的需求

The agent's decision making procedure is characterized by a stochastic policy $\pi(a;s)$ , which is the probability of taking action $a$ in state $s$ (where the semi-colon is used to distinguish the parameters from the random variables of the distribution). 
>  智能体的策略是随机策略 $\pi(a; s)$

We only consider the case where the goal of the agent is to maximize the $\gamma\cdot$ discounted average reward from the starting state distribution $D$ , though this has a similar solution to maximizing the average reward for processes that “mix” on a reasonable timescale [4]. 
>  我们只考虑 agent 的目标是从初始分布 $D$ 开始，最大化 $\gamma$ -discounted 平均奖励的情况

Given $0\leq\gamma<1$ , we define the value function for a given policy $\pi$ as 

$$
V_{\pi}(s)\equiv(1-\gamma)E\left[\sum_{t=0}^{\infty}\gamma^{t}\mathcal{R}(s_{t},a_{t})|\pi,s\right]
$$ 
where $s_{t}$ and $a_{t}$ are random variables for the state and action at time $t$ upon executing the policy $\pi$ from the starting state $s$ (see [7] for a formal definition of this expectation).  Note that we are using normalized values so $V_{\pi}(s)\in[0,R]$ . 

>  以上定义了从 $s$ 开始，遵循 $\pi$ 的价值函数
>  价值函数乘上了 $(1-\gamma)$ 进行规范化，确保 $V_\pi(s)\in[0, R]$

>  推导
>  $V_\pi(s)$ 在 $\forall t, \mathcal R(s_t, a_t) = 0$ 时取到最小值 $0$
>  $V_\pi(s)$ 在 $\forall t, \mathcal R(s_t, a_t) = R$ 时取到最大值

$$
\begin{align}
\max V_\pi(s) &= (1-\gamma)E\left[\sum_{t=0}^\infty \gamma^t R\mid \pi,s\right]\\
&=(1-\gamma)\sum_{t=0}^\infty \gamma^t R\\
&=(1-\gamma)(1 + \gamma + \gamma^2 + \cdots )R\\
&=(1-\gamma)\frac {\gamma^\infty -  1}{\gamma - 1} R\\
&=(1-\gamma)\frac {1}{1-\gamma}R\\
&=R
\end{align}
$$

>  最大值是 $R$

For a given policy $\pi$ , we define the state-action value as 

$$
Q_{\pi}(s,a)\equiv(1-\gamma)\mathcal{R}(s,a)+\gamma E_{s^{\prime}\sim P(s^{\prime};s,a)}\left[V_{\pi}(s^{\prime})\right]
$$ 
and as in [8] (much as in [1]), we define the advantage as 

$$
A_{\pi}(s,a)\equiv Q_{\pi}(s,a)-V_{\pi}(s)
$$ 
Again, both $Q_{\pi}(s,a)\in[0,R]$ and $A_{\pi}(s,a)\in[-R,R]$ due to normalization. 

>  以上定义了动作价值函数和优势函数
>  动作价值函数的范围仍然是 $Q_\pi(s, a)\in [0, R]$，故优势函数的范围是 $A_\pi(s, a) \in [-R, R]$

It is convenient to define the $\gamma$ -discounted future state distribution (as in [8]) for a starting state distribution $\mu$ as 

$$
d_{\pi,\mu}(s)\equiv(1-\gamma)\sum_{t=0}^{\infty}\gamma^{t}\operatorname*{Pr}(s_{t}=s;\pi,\mu)\tag{2.1}
$$ 
where the $1-\gamma$ is necessary for normalization. 

>  为起始状态分布 $\mu$ 和策略 $\pi$ 定义 $\gamma$ -discounted 未来状态分布如上
>  其中 $1-\gamma$ 是必要的，用于归一化

>  推导
>  我们要确保 $\sum_s d_{\pi, \mu}(s) = 1$

$$
\begin{align}
\sum_s d_{\pi,\mu}(s) &= (1-\gamma)\sum_s \sum_{t=0}^\infty \gamma^t P(s_t = s;\pi, \mu)\\
&=(1-\gamma)\sum_{t=0}^\infty \gamma^t \sum_s P(s_t = s;\pi,\mu)\\
&=(1-\gamma)\sum_{t=0}^\infty \gamma^t\\
&=(1-\gamma)(1 + \gamma + \gamma^2 + \cdots)\\
&=1
\end{align}
$$

>  其中第二个等号交换求和次序是因为级数非负且绝对收敛 (绝对收敛的双重级数可以交换求和次序)
>  第三个等号利用了 $\forall t\ge 0, \sum_s P(s_t = s;\pi,\mu) = 1$ 的性质

We abuse notation and write $d_{\pi,s}$ for the discounted future-state distribution with respect to the distribution which deterministically starts from state $s$ .
>  我们在这里滥用符号，使用 $d_{\pi, s}$ 标识与确定性地从状态 $s$ 开始的分布相关的 discounted future-state distribution

Note that $V_{\pi}(s)=E_{(a^{\prime},s^{\prime})\sim\pi d_{\pi,s}}\left[\mathcal{R}(s^{\prime},a^{\prime})\right]$ .
>  $V_\pi(s)$ 可以写为关于 $\pi d_{\pi, s}$ 求期望的形式

>  推导

$$
\begin{align}
V_\pi(s) &=  (1-\gamma) E_{\tau\sim \pi}\left[\sum_{t=0}^\infty \gamma^t \mathcal R(s_t, a_t)\mid s_0=s \right]\\
&=(1-\gamma)\sum_{t=0}^\infty \gamma^t E_{\tau\sim \pi}[\mathcal R(s_t, a_t)\mid s_0 = s]\\
&=(1-\gamma)\sum_{t=0}^\infty \gamma^t \sum_{s'}\sum_{a'}\mathcal R(s', a')P(s_t = s', a_t = a'\mid s_0 = s, \pi)\\
&=(1-\gamma)\sum_{t=0}^\infty \gamma^t \sum_{s'}\sum_{a'}\mathcal R(s', a')P(s_t=s'\mid s_0=s, \pi)\cdot \pi(a'\mid s')\\
&=\sum_{s'}\sum_{a'}\mathcal R(s', a')(1-\gamma)\sum_{t=0}^\infty \gamma^tP(s_t=s'\mid s_0=s, \pi)\cdot \pi(a'\mid s')\\
&=\sum_{s'}\sum_{a'}\mathcal R(s', a') d_{\pi,s}(s')\cdot \pi(a'\mid s')\\
&=E_{a',s'\sim \pi d_{\pi,s}}[\mathcal R(s', a')]
\end{align}
$$

>  其中第二个等式将求和移出期望，依旧利用了绝对收敛的双重级数可以交换求和次序的性质 (期望本质也是求和)
>  证明过程中通过求和顺序的移动构造出了 future-state distribution
>  最后得到的结论即状态价值函数可以写为对状态的 future-state distribution 和策略求期望的形式

This distribution is analogous to the stationary distribution in the undiscounted setting, since as $\gamma\to1$ ， $d_{\pi,s}$ tends to the stationary distribution for all $s$ , if one such exists. 
>  future-state distribution 类似于在无折扣条件下的稳态分布，因为当 $\gamma \to 1$，$\forall s, d_{\pi, s}$ 将趋向于其稳态分布，如果稳态分布存在的话

>  $\gamma\to 1$ 使得 $t \to \infty$ 时，$P(s_t; s_0,\pi)$ 的系数仍可以保持接近 1，随着 Markov Chain 趋于稳态，$P(s_t; s_0, \pi)$ 也将趋于 Markov Chain 的稳态分布 $\pi^*(s)$，之后便不再变化，故 $\pi^*(s)$ 将随着时间推移，在 $P(s_t; s_0,\pi)$ 中的比重越来越高，最后 $P(s_t; s_0,\pi)$ 就趋近于稳态分布
>  如果 $\gamma \not\to 1$，则当 $t\to \infty$ 时，或许 Markov Chain 尚未到达稳态，$P(s_t; s_0,\pi)$ 的权重就已经趋近于 0，故 $\sum_{t=0}^\infty \gamma^t P(s_t ;s_0,\pi)$ 的收敛值将还是主要由 $\gamma$ 本身的值决定

The goal of the agent is to maximize the discounted reward from the start state distribution $D$ 

$$
\eta_{D}(\pi)\equiv E_{s\sim D}\left[V_{\pi}(s)\right].
$$ 
Note that $\eta_{D}(\pi)~=~E_{(a,s)\sim\pi d_{\pi,D}}\left[\mathcal{R}(s,a)\right]$ .A well known result is that a policy exists which simultaneously maximizes $V_{\pi}(s)$ for all states. 

>  agent 的目标是最大化相对于初始状态分布 $D$ 的期望 discounted reward
>  利用 $V_\pi(s)$ 对状态的 future-state distribution 求期望的形式，$\eta_D(\pi)$ 可以写为 $E_{(a, s)\sim \pi d_{\pi, D}}[\mathcal R(s, a)]$
>  存在一种策略可以同时使所有状态的价值 $V_\pi(s)$ 达到最大值 (最优策略的定义)

# 3 The Problems with Current Methods 
We now examine in more detail the problems with approximate value function methods and policy gradient methods. There are three questions to which we desire answers to: 

(1) Is there some performance measure that is guaranteed to improve at every step? 
(2) How difficult is it to verify if a particular update improves this measure? 
(3) After a reasonable number of policy updates, what performance level is obtained? 

>  我们讨论近似价值函数方法和策略梯度方法的问题，我们讨论三个问题
>  1. 方法的每一步是否确保某个性能度量是单调提升的
>  2. 是否容易验证特定的更新会提高该度量
>  3. 在一定数量的策略更新后，策略会达到什么样的性能水平

We now argue that both greedy dynamic programming and policy gradient methods give unsatisfactory answers to these questions. Note that we did not ask is "What is the quality of the asymptotic policy?". We are only interested in policies that we can find in a reasonable amount of time. 
>  我们认为，贪心动态规划和策略梯度方法对于这些问题的答案都不尽如人意
>  注意，我们没有关心渐进策略的性能如何，我们只关心能在合理的时间内找到的策略

>  贪心动态规划指的就是经典的策略迭代方法
>  如果每一步都可以获取精确的价值函数 $V_\pi$，则该方法不会存在问题，但这的代价太高，故常用的是近似的价值函数估计，而这就会导致问题
>  本文对该方法在 section 3.1 进行了讨论

Understanding the problems with these current methods gives insight into our new algorithm, which addresses these three questions. 
>  理解这些当前方法的问题有助于深度了解我们提出的新算法，该算法旨在解决上述提出的三个问题

## 3.1 Approximate Value Function Methods 
Exact value function methods, such as policy iteration, typically work in an iterative manner. Given a policy $\pi$ , policy iteration calculates the state-action value $Q_{\pi}(s,a)$ , and then creates a new deterministic policy $\pi^{\prime}(a;s)$ such that $\pi^{\prime}(a;s)=1\ \text{iff}\ a\in\mathrm{argmax}_{a}Q_{\pi}(s,a)$ . This process is repeated until the state-action values converge to their optimal values. These exact value function methods have strong bounds showing how fast the values converge to optimal (see [7]). 
>  精确的价值函数方法，例如策略迭代，通常以迭代方式运行
>  给定策略 $\pi$，策略迭代方法先计算动作价值函数 $Q_\pi(s, a)$，然后基于 $\pi'(a, s) = 1\ \text{iff}\ a\in \arg\max_aQ_\pi(s, a)$ 构造新的确定性策略 $\pi'(a; s)$，该过程不断重复，直到动作价值函数收敛到其最优值
>  精确的价值函数方法具有严格的界，用以判断价值函数收敛到最优的速度

Approximate value function methods typically use approximate estimates of the state-action values in an exact method. These methods suffer from a paucity of theoretical results on the performance of a policy based on the approximate values. This leads to weak answers to all three questions. 
>  近似价值函数方法对精确方法中的动作价值进行近似估计
>  目前缺少关于基于近似价值得到策略的性能的理论上的性能保证，故这些方法也难以回答上述三个问题

Let us consider some function approximator $\widetilde{V}(s)$ with the $l_{\infty}$ -error: 

$$
\varepsilon=\operatorname*{max}_{s}|\widetilde{V}(s)-V_{\pi}(s)|
$$ 
where $\pi$ is some policy.

>  考虑某个函数近似器 $\tilde V(s)$，其 $l_\infty$ 误差定义如上，可以看到等于最大的绝对价值差，其中 $\pi$ 是目标策略

> [!info] $L_\infty$
> 函数 $f$ 的 $L_\infty$ 范数等于 $\max f(x)$，即函数的最大值
>  $l_\infty$ 表示了最坏情况下的误差

Let $\pi^{\prime}$ be a greedy policy based on this approximation. We have the following guarantee (see [3]) for all states $s$ 

$$
V_{\pi^{\prime}}(s)\geq V_{\pi}(s)-\frac{2\gamma\varepsilon}{1-\gamma}.\tag{3.1}
$$ 
>  令 $\pi'$ 是基于该近似价值 $\tilde V(s)$ 的贪心策略，我们对于所有的状态 $s$ 都有以上的界保证
>  可以看到，$\pi'$ 的状态价值将确保不小于 $\pi$ 的状态价值减去 $\frac {2\gamma}{1-\gamma} \epsilon$，其中 $\epsilon$ 是 $\tilde V(s)$ 的最大绝对估计偏移量
>  显然， $\epsilon$ 越大，$\pi'$ 更优的概率通常更低

In other words, the performance is guaranteed to not decrease by more than $\frac{2\gamma\varepsilon}{1-\gamma}$ . 
>  换句话说，性能的下降幅度不会超过 $\frac {2\gamma\epsilon}{1-\gamma}$
>  近似价值方法无法保证性能提升，故无法回答 Q1

>  作者认为，根据 Eq3.1，近似价值方法无法保证在每一步迭代单调提高策略，而是有可能导致策略变差 (如果 $\epsilon$ 比较大的话)，故这就是它的问题所在

Question 2 is not applicable since these methods don't guarantee improvement and a performance measure to check isn't well defined. 

For approximate methods, the time required to obtain some performance level is not well understood. Some convergence and asymptotic results exist (see [3]). 
>  对于近似方法，达到特性性能级别所需的时间也暂未有理论结果，存在部分收敛性和渐进结果

## 3.2 Policy Gradients Methods 
Direct policy gradient methods attempt to find a good policy among some restricted class of policies, by following the gradient of the future reward.
>  直接策略梯度方法尝试在一些受限的策略类 (由参数化的形式限制) 中找到一个良好的策略，通过计算未来奖励的梯度进行梯度上升优化

>  相较于策略迭代方法，直接的策略梯度方法直接对策略进行优化，这是其优势，但其劣势在于策略的具体形式由参数化的形式限制，故可探索的策略空间是有限的

Given some parameterized class $\{\pi_{\theta}|\theta\in\mathcal{R}^{m}\}$ ,these methods compute the gradient 


$$
\nabla\eta_{D}=\sum_{s,a}d_{\pi,D}(s)\nabla\pi(a;s)Q_{\pi}(s,a)\tag{3.2}
$$ 
(as shown in [8]). 

>  给定一类参数化的策略类 $\{\pi_\theta\mid \theta\in \mathcal R^m\}$，策略梯度方法按照上述公式计算梯度
>  (即目标函数是 $\mathbb E_{s\sim D}[V_\pi(s)]$ 时的策略梯度定理)

For policy gradient techniques, question 1 has the appealing answer that the performance measure of interest is guaranteed to improve under gradient ascent. 
>  策略梯度方法可以回答 Q1，我们关注的性能度量 (目标函数) 将在梯度上升下稳定提升

We now address question 2 by examining the situations in which estimating the gradient direction is difficult. We show that the lack of exploration in gradient methods translates into requiring a large number of samples in order to accurately estimate the gradient direction. 
>  我们通过分析在哪些情况下估计梯度方向是困难的，来回答 Q2
>  我们将说明，在梯度方法中缺乏探索性将导致需要大量的样本来准确估计梯度方向

![[pics/CPI-Fig3.1.png]]

Consider the simple MDP shown in Figure 3.1 (adapted from [10]). Under a policy that gives equal probability to all actions, the expected time to the goal from the left most state is $3(2^{n}-n-1)$ ,and with $n=50$ , the expected time to the goal is about $10^{15}$ .
>  考虑 Fig3.1 中的 MDP，在以相同概率随机选择左右的策略下，从最左侧的状态到最右侧的期望时间步数是 $3(2^n - n - 1)$

>  推导
>  该 MDP 中，状态空间是一个线性链，包含 $n$ 个状态 $0, 1, \dots, n-1$，在状态 $i$，智能体可以选择移动到状态 $i-1$ 和 $i+1$，其中状态 $0$ 表示处于目标状态，即最右端，状态 $n-1$ 表示处于初始状态，即最左边

>  (1) 定义递推关系
>  定义期望时间 $E_i$ 表示从状态 $i$ 到状态 $0$ 的期望时间步，建立如下递推关系
>  1. 处于状态 $0$ 时，$E_0 = 0$
>  2. 处于状态 $i(1\le i \le n-2)$ 时，智能体有 50% 可能移动到左边，有 50% 可能移动到右边，故存在递推关系 $E_i = 1 + 0.5 E_{i-1} + 0.5 E_{i+1}$
>  3. 处于状态 $n-1$ 时，智能体只能向右移动，故 $E_{n-1} = 1 + E_{n-2}$

>  (2) 求解递推关系
>  递推关系可以写为 $E_{i+1} - 2E_i + E_{i-1} = -2$，这是一个线性非齐次常系数递推方程，我们采用特解法对其进行求解
>  我们首先求其通解，该递推方程的特征方程是 $r^2 - r + 1 = 0$，它有一个重根 $r=1$，故特解数列的通项表达式可以写为 $u_i = A + B i$
>  非齐次项 $-2$ 可以写为 $1^i \cdot (-2)$，故特解数列的通项表达式可以写为 $v_i = Ci^2$

>  (3) 求解特解数列待定系数
>  首先利用递推式求解特解的待定系数，写出方程
>  $C(i+1)^2 - 2Ci^2 + C(i-1)^2 = -2$
>  求解得到 $C = -1$
>  此时，状态 $i(1\le 1\le n-2)$ 的通项公式为 $E_i = A + Bi - i^2$

>  (4) 求解通解数列待定系数
>  $E_1 = A + B - 1, E_{n-2} = A + B(n-2) - (n-2)^2$
>  $E_1 = 1 + 0.5 E_0 + 0.5 E_2 = 1 + 0.5 E_2$
>  $E_{n-2} = 1 + 0.5 E_{n-3} + 0.5 E_{n-1} \Rightarrow E_{n-2} = 3 + E_{n-3}$

$$
\begin{align}
E_1 = A + B -1 &= 1 + 0.5(A + 2B - 4)\\
A + B -1 &= 1 + 0.5A + B - 2\\
0.5A &= 0\\
A &= 0
\end{align}
$$

$$
\begin{align}
E_{n-2} = B(n-2) - (n-2)^2 &= 3 + B(n-3) - (n-3)^2\\
Bn - 2B -(n-2)^2 &= 3 + Bn - 3B -(n-3)^2\\
B &= 3 + (n-2)^2 - (n-3)^2\\
B &= 3 + (n-2 - n + 3) (n-2 + n-3)\\
B &= 3 + (2n-5)\\
B &= 2n-2
\end{align}
$$

>  故通项公式为 $E_i = (2n-2) i - i^2$
>  当 $i = n-1$，$E_{n-1} = 2(n-1)^2 - (n-1)^2 = (n-1)^2$
>  结果和文章表述不符，但是不确定是哪里有问题，或许是对其随机策略的定义理解有误

This MDP falls in the class of MDPs in which random actions are more likely than not to increase the distance to the goal state. For these classes of problems (see [11]), the expected time to reach the goal state using undirected exploration, i.e. random walk exploration, is exponential in the size of the state space. Thus, any "on-policy" method has to run for at least this long before any policy improvement can occur. In online value function methods, this problem is seen as a lack of exploration. 
>  上述 MDP 属于一类 MDP，这类 MDP 中，执行随机动作 (无方向探索) 相较于有方向探索更有可能增加达到目标状态的距离
>  对于这类问题，使用无方向探索 (即随机游走探索) 以达到目标状态的期望时间是指数于状态空间的大小的
>  因此，任意的同策略方法必须至少运行指数于状态空间大小的时间才能发生策略改进，在 online 价值函数方法中，该问题被视作缺乏探索性

>  上面的描述中，目标状态应该是指存在奖励的状态，作者的意思是同策略方法通常会从一个随机策略开始，利用该策略探索的同时不断改进该策略，其中策略的改进显然也依赖于其价值函数的改进，进而依赖于是否达到过目标状态
>  而随机游走探索达到目标状态所需要的时间步数通常指数于状态空间大小，因此，要改进一次策略 (通过改进价值函数)，就需要指数于状态空间大小的探索步数

Any sensible estimate of the gradient without reaching the goal state would be zero, and obtaining non-zero estimates requires exponential time with "on-policy" samples. 
>  任何在未达到目标状态之前对梯度的合理估计都应为零，而利用 on-policy 样本获得非零估计需要指数级的时间  (不妨理解为初始时 $Q_\pi(s, a)$ 都初始化为 0，如果没有达到能够获取奖励的目标状态，$Q_\pi(s, a)$ 将永远保持初始零值，进而策略梯度也保持零值)

Importance sampling methods do exist (see [6]), but are not feasible solutions for this class of problems. The reason is that if the agent could follow some "off-policy” trajectory to reach the goal state in a reasonable amount of time, the importance weights would have to be exponentially large. 
>  重要性采样方法对于这类问题不是可行的解决方案 (使用重要性采样的思路应该在于: on policy 下初始的随机策略即是行为策略也是目标策略，而使用随机策略难以探索到目标状态，故可以考虑使用某些更好的行为策略)
>  不可行的原因在于: 如果智能体能够遵循某种 off-policy 轨迹在合理的时间内达到目标状态，则重要性权重将会指数级增长

Note that a zero estimate is a rather accurate estimate of the gradient in terms of magnitude, but this provides no information about direction, which is the crucial quantity of interest. The analysis in [2] suggests a relatively small sample size is needed to accurately estimate the magnitude (within some $\varepsilon$ tolerance), though this does not imply an accurate direction if the gradient is small. Unfortunately, the magnitude of the gradient can be very small when the policy is far from optimal. 
>  零估计在幅度上对于梯度的估计是准确的，但无法提供方向信息
>  [2] 的分析表明，为了准确估计幅度 (在某个 $\epsilon$ 容差范围内)，只需要相对小的样本量，但如果梯度很小，估计的幅度正确不意味着估计的方向正确
>  当目前策略远离最优策略时，梯度的幅度可能会非常小，故估计梯度的方向正确性是难以保证的

![[pics/CPI-Fig3.2.png]]

Let us give an additional example demonstrating the problems for the simple two state MDP shown in Figure 3.2, which uses the common Gibbs table-lookup distributions, $\{\pi_{\theta}:\pi(a;s)\propto\exp(\theta_{s a})\}$ .Increasing the chance of a self-loop at $i$ decreases the stationary probability of $j$ , which hinders the learning at state $j$ . 
>  考虑 Fig3.2 中的两状态 MDP，其中策略使用常见的 Gibbs table-lookup distribution
>  提高在状态 $i$ 处的自循环概率会降低 $j$ 的稳态概率，进而阻碍了在状态 $j$ 的学习 (更可能留在状态 $i$，不容易探索到状态 $j$)

>  该 MDP 中，环境的 transition 是确定性的，故 Markov Chain 视角上的 transition = policy $\times$ dynamics transition 就等于 policy
>  故 policy 会定义该 Markov Chain 的稳态分布

Under an initial policy that has the stationary distribution $\rho(i)=.8$ and $\rho(j)=.2$ (using $\pi(a_{1};i)=.8$ and $\pi(a_{1};j)=.9)$ , learning at state $i$ reduces the learning at state $j$ leading to an an extremely fat plateau of improvement at 1 average reward shown in Figure 3.2. 
>  例如，在稳态分布为 $\rho(i)=0.8$ 和 $\rho(j)=0.2$ 的初始策略下 (使用 $\pi(a_{1};i)=0.8$ 和 $\pi(a_{1};j)=0.9$)，在状态 $i$ 的学习会减少在状态 $j$ 的学习，导致 Fig3.2 所示的平均奖励改进的极为平坦的区域 (一直在状态 $i$ 停留，平均奖励始终保持为 1)

>  上述稳态分布的计算应该有误，正确的计算如下
>   
$$
\begin{align}
\rho(i) \times 0.8 + \rho(j)\times 0.1 &= \rho(i)\\
\rho(j)\times0.9 + \rho(i)\times 0.2 &= \rho(j)\\
\rho(j) + \rho(i) &= 1
\end{align}
$$
>  
$$
\begin{align}
0.8\rho(i) + (1-\rho(i))\times 0.1 &= \rho(i)\\
0.7\rho(i) + 0.1 &= \rho(i)\\
0.3\rho(i)&=0.1\\
\rho(i) &= \frac 1 3
\end{align}
$$

>  故稳态分布应该为 $\rho(i) = \frac 1 3, \rho(j) = \frac 2 3$

![[pics/CPI-Fig3.3.png]]

Figure 3.3 shows that this problem is so severe that $\rho(j)$ drops as low as $10^{-7}$ from it's initial probability of .2. As in example 1, to obtain a nonzero estimate of the gradient it is necessary to visit state $j$ ：The situation could be even worse with a few extra states in a chain as in figure 3.1
>  Fig3.3 显示了这个问题如此严重，以至于 $\rho(j)$ 从初始概率 0.2 下降到低至 $10^{-7}$
>  与 example 1 类似，为了获得梯度的非零估计，必须访问状态 $j$：如果链中有几个额外的状态 (如 Fig3.1 所示)，情况可能会更糟

>  目前为止，一直讨论的是探索性的问题，探索性不够，策略将难以探索到具有更优奖励的目标状态，进而这些目标状态的动作价值可能仍保持初始的零值，从目标函数的视角来看，此时增大留在当前状态的概率反而是更有益的，可以稳定获得奖励，进而策略更新会不断增大留在当前状态的概率

Although asymptotically a good policy might be found, these results do not bode well for the answer to question 3, which is concerned with how fast such a policy can be found.
>  虽然渐进上，最终会找到一个优秀策略，但 Q3 关注的是合理数量的更新下，策略所能达到的水平，也就是关注找到一个优秀策略可以多快

These results suggest that in any reasonable number of steps, a gradient method could end up being trapped at plateaus where estimating the gradient direction has an unreasonably large sample complexity. 
 > 以上的结果显示，在任何合理的步数内，梯度方法都有可能被困在平面上，在该平面上，合理估计梯度方向将要求非常大的样本复杂度 (即需要非常多次采样才有可能探索到一个更优的目标状态)
 
Answering question 3 is crucial to understand how well gradient methods perform, and (to our knowledge) no such analysis exists. 

# 4 Approximately Optimal RL 
The fundamental problem with policy gradients is that $\eta_{D}$ , which is what we ultimately seek to optimize, is insensitive to policy improvement at unlikely states though policy improvement at these unlikely states might be necessary in order for the agent to achieve near optimal payoff. 
>  策略梯度的根本问题在于其目标函数 $\eta_D$ 对于在不太可能状态下的策略改进是不敏感的 (状态的概率低，它的价值在目标 $\eta_D$ 中占的权重就低)，而在这些不太可能状态下的策略改进可能对于智能体接近最优回报是必须的

We desire an alternative performance measure that does not down weight advantages at unlikely states or unlikely actions. A natural candidate for a performance measure is to weight the improvement from all states more uniformly (rather than by $D$ ),such as 

$$
\eta_{\mu}(\pi)\equiv E_{s\sim\mu}\left[V_{\pi}(s)\right]
$$ 
where $\mu$ is some “exploratory” restart distribution. 

>  我们希望有一个替代的性能度量，它不会低估不太可能状态或不太可能动作的优势
>  一个自然的选择是更均匀地加权来自所有状态的改进 (而不是根据 $D$ 加权)，例如对某个 “探索性” 的 restart distribution $\mu$ 求期望，记作 $\eta_\mu(\pi)$

Under our assumption of having access to a $\mu$ -restart distribution, we can obtain estimates of $\eta_{\mu}(\pi)$ 
>  在可以访问 $\mu$ 分布的假设下，可以获得对 $\eta_\mu(\pi)$ 的估计

Any optimal policy simultaneously maximizes both $\eta_{\mu}$ and $\eta_{D}$ . Unfortunately, the policy that maximizes $\eta_{\mu}$ within some restricted class of policies may have poor performance according to $\eta_{D}$ , So we must ensure that maximizing $\eta_{\mu}$ results in a good policy under $\eta_{D}$ 
>  任意的最优策略都将同时最大化 $\eta_\mu$ 和 $\eta_D$ (因为最优策略本质上是最大化 $V(s)\forall s$)
>  但是，在某个限制类内的最大化 $\eta_\mu$ 的策略则不一定会最大化 $\eta_D$，故我们需要确保最大化 $\eta_\mu$ 的策略在 $\eta_D$ 评估下也是优秀的策略

Greedy policy iteration updates the policy to some $\pi^{\prime}$ based on some approximate state-action values. Instead, let us consider the following more conservative update rule: 

$$
\pi_{\mathrm{new}}(a;s)=(1-\alpha)\pi(a;s)+\alpha\pi^{\prime}(a;s),\tag{4.1}
$$ 
for some $\pi^{\prime}$ and $\alpha\in[0,1]$ . 

>  贪心策略迭代基于近似的动作价值将策略贪心更新至 $\pi'$
>  我们考虑 Eq4.1 的保守更新规则，即不会完全更新到 $\pi'$，而是加权混合
>  $\alpha = 1$ 时，等价于贪心策略迭代

To guarantee improvement with $\alpha=1$ ， $\pi^{\prime}$ must choose a better action at every state, or else we could suffer the penalty shown in equation 3.1. 

In the remainder of this section, we describe a more conservative policy iteration scheme using $\alpha<1$ and state the main theorems of this paper. 
>  我们将描述 $\alpha < 1$ 的保守策略迭代方法

In subsection 4.1, we show that $\eta_{\mu}$ can improve under the much less stringent condition in which $\pi^{\prime}$ often, but not always, chooses greedy actions. In subsection 4.2, we assume access to a greedy policy chooser that outputs “approximately” greedy policies $\pi^{\prime}$ and then bound the performance of the policy found by our algorithm in terms of the quality of this greedy policy chooser. 
>  sec4.1 中，我们证明在更弱的条件下，即 $\pi'$ 经常，但并非总是选择贪心动作时，$\eta_\mu$ 可以提升
>  sec4.2 中，我们假设可以访问一个贪心策略选择器，它输出 “近似” 贪心策略 $\pi'$，然后根据该贪心策略选择器的质量来为我们的算法找到的策略表现定界

>  作者为了增强探索性，引入了 $\mu$ 替代 $D$，并且引入了保守更新
>  作者将说明，这样的更新可以提升策略性能

## 4.1 Policy Improvement 
A more reasonable situation is one in which we are able to improve the policy with some $\alpha>0$ using a $\pi^{\prime}$ that chooses better actions at most but not all states. 
>  在 Eq4.1 的保守更新下，当 $\pi'$ 会在大多数情况下但不是所有的情况下选择的是更好的动作时，选择某个 $\alpha > 0$ 可以改进策略

Let us define the policy advantage $\mathbb{A}_{\pi,\mu}\left(\pi^{\prime}\right)$ of some policy $\pi^{\prime}$ with respect to a policy $\pi$ and a distribution $\mu$ to be 

$$
\mathbb{A}_{\pi,\mu}(\pi^{\prime})\equiv E_{s\sim d_{\pi,\mu}}\left[E_{a\sim\pi^{\prime}(a;s)}\left[A_{\pi}(s,a)\right]\right].
$$ 
>  我们为某个策略 $\pi'$ 定义它相对于策略 $\pi$ 和分布 $\mu$ 的策略优势 $\mathbb A_{\pi, \mu}(\pi')$ 如上
>  该定义是一个期望，外层对 $s\sim d_{\pi,\mu}$ 求期望，内层对 $a\sim \pi'(a; s)$ 求期望，内部是 $\pi$ 定义的优势函数值 $A_\pi(s, a)$

>  注意，当 $\pi = \pi'$ 时

$$
\begin{align}
\mathbb A_{\pi,\mu}(\pi) &=E_{s\sim d_{\pi,\mu}}[E_{a\sim \pi(a;s)}[A_\pi(s,a)]]\\
&=E_{s\sim d_{\pi,\mu}}[E_{a\sim \pi(a;s)}[Q_{\pi}(s,a) - V_\pi(s)]]\\
&=E_{s\sim d_{\pi,\mu}}[E_{a\sim \pi(a;s)}[Q_{\pi}(s,a)] - V_\pi(s)]\\
&=E_{s\sim d_{\pi,\mu}}[E_{a\sim \pi(a;s)}[V_\pi(s) - V_\pi(s)]\\
&=0
\end{align}
$$

>  故对于给定状态 $s$，如果遵循 $\pi'$ 的动作选择将带来更大的价值 (仅遵循一次，之后继续遵循 $\pi$ 以计算价值)，$\mathbb A_{\pi,\mu}(\pi)$ 就将大于零

The policy advantage measures the degree to which $\pi^{\prime}$ is choosing actions with large advantages, with respect to the set of states visited under $\pi$ starting from a state $s\sim\mu$ . Note that a policy found by one step of policy improvement maximizes the policy advantage. 
>  策略优势度量了相对于从状态 $s\sim\mu$ 开始，在策略 $\pi$ 下访问的状态集合上，策略 $\pi'$ 选择具有较大优势动作的程度
>  注意，通过一步策略改进 (贪心) 找到的策略将最大化上述定义的策略优势 (因为贪心情况下，是确定性地选择 $\arg\max_a Q_\pi(s, a),\forall s$)

It is straightforward to show that $\begin{array}{r}{\frac{\partial\eta_{\mu}}{\partial\alpha}|_{\alpha=0}=\frac{1}{1-\gamma}\mathbb{A}_{\pi,\mu}}\end{array}$ (using equation 3.2), so the change in $\eta_{\mu}$ is: 

$$
\Delta\eta_{\mu}=\frac{\alpha}{1-\gamma}\mathbb{A}_{\pi,\mu}\left(\pi^{\prime}\right)+O(\alpha^{2}).\tag{4.2}
$$ 
>  可以证明 $\frac {\partial \eta_\mu}{\partial \alpha} \mid_{\alpha = 0} = \frac 1 {1-\gamma}\mathbb A_{\pi,\mu}$，故 $\eta_\mu$ 的改变量 $\Delta \eta_\mu$ 可以借助一阶 Taylor 展开进行近似，形式如上所示 (展开点 $\alpha = 0$ 处的梯度值 $\cdot$ $\alpha$ + 二阶余项 $O(\alpha^2)$ )

>  推导

$$
\begin{align}
&\frac {\partial \eta_{\mu}}{\partial \alpha}\mid_{\alpha = 0}\\
=&\frac {\partial \eta_\mu(\pi_{new})}{\partial \alpha}\mid_{\alpha = 0}\\
=&\frac {\partial E_{s\sim \mu}[V_{\pi_{new}}(s)]}{\partial \alpha}\mid_{\alpha = 0}\\
\end{align}
$$

>  考虑直接利用 Eq3.2 的策略梯度定理，只不过此时求梯度的对象是 $\alpha$ 而不是参数 $\theta$，但是定理仍然是适用的

$$
\begin{align}
&\frac {\partial \eta_\mu(\pi_{new})}{\partial \alpha}\mid_{\alpha=0}\\
=&\sum_{s,a}d_{\pi_{new},\mu}(s)\nabla_\alpha\pi_{new}(a;s)Q_{\pi_{new}}(s,a)\mid_{\alpha = 0}\\
=&\sum_{s,a}d_{\pi,\mu}(s)\nabla_\alpha\pi_{new}(a;s)\mid_{\alpha =0} Q_{\pi}(s,a)\\
=&\sum_{s,a}d_{\pi,\mu}(s)Q_{\pi}(s,a)\nabla_\alpha\pi_{new}(a;s)\mid_{\alpha =0} \\
=&\sum_{s,a}d_{\pi,\mu}(s) Q_{\pi}(s,a)\nabla_\alpha\left((1-\alpha)\pi + \alpha\pi'\right)\mid_{\alpha =0}\\
=&\sum_{s,a}d_{\pi,\mu}(s) Q_{\pi}(s,a)\nabla_\alpha\left(-\alpha\pi + \alpha\pi'\right)\mid_{\alpha =0}\\
=&\sum_{s,a}d_{\pi,\mu}(s) Q_{\pi}(s,a)\left(-\pi + \pi'\right)\mid_{\alpha =0}\\
=&\sum_{s,a}d_{\pi,\mu}(s) Q_{\pi}(s,a)\left(\pi'(a;s) - \pi(a;s)\right)\\
=&\sum_{s,a}d_{\pi,\mu}(s) \left(\pi'(a;s) Q_{\pi}(s,a)- \pi(a;s)Q_{\pi}(s,a)\right)\\
=&E_{s\sim d_{\pi,\mu}(s)} \left[\sum_a(\pi'(a;s) Q_{\pi}(s,a)- \pi(a;s)Q_{\pi}(s,a))\right]\\
=&E_{s\sim d_{\pi,\mu}(s)} \left[\sum_a\pi'(a;s) Q_{\pi}(s,a)- \sum_a\pi(a;s)Q_{\pi}(s,a)\right]\\
=&E_{s\sim d_{\pi,\mu}(s)} \left[E_{a\sim\pi'(a;s)}[ Q_{\pi}(s,a)]- E_{a\sim\pi(a;s)}[Q_{\pi}(s,a)]\right]\\
=&E_{s\sim d_{\pi,\mu}(s)} \left[E_{a\sim\pi'(a;s)}[ Q_{\pi}(s,a)]- V_\pi(s)\right]\\
=&E_{s\sim d_{\pi,\mu}(s)} \left[E_{a\sim\pi'(a;s)}[ Q_{\pi}(s,a) - V_\pi(s)] \right]\\
=&E_{s\sim d_{\pi,\mu}(s)} \left[E_{a\sim\pi'(a;s)}[ A_\pi(s,a)] \right]\\
=&\mathbb A_{\pi,\mu}(\pi')
\end{align}
$$

>  这里的系数 $\frac 1 {1-\gamma}$ 应该是来自于严谨形式的策略梯度定理中的系数 $(1 + \gamma + \gamma^2 + \cdots)$，因此这里假设了时间步无限

Hence, for sufficiently small $\alpha$ , policy improvement occurs if the policy advantage is positive, and at the other extreme of $\alpha=1$ , significant degradation could occur. We now connect these two regimes to determine how much policy improvement is possible. 
>  因此，对于足够小的 $\alpha$，如果策略优势 $\mathbb A_{\pi,\mu}(\pi')$ 为正，则策略就会改进 ($\Delta \eta_\mu > 0$)
>  而在 $\alpha = 1$ 的极端情况下，可能会出现显著退化 (参照 Eq3.1)
>  我们将这两个极端情况联系起来，以确定可能实现的策略改进程度

**Theorem 4.1.** Let $\mathbb A$ be the policy advantage of $\pi^{\prime}$ with respect to $\pi$ and $\mu$ and let $\varepsilon={}$ $\begin{array}{r}{\operatorname*{max}_{s}|E_{a\sim\pi^{\prime}(a;s)}\left[A_{\pi}(s,a)\right]|}\end{array}$ . For the update rule 4.1 and for all $\alpha\in[0,1]$ 

$$
\eta_{\mu}(\pi_{n e w})-\eta_{\mu}(\pi)\geq\frac{\alpha}{1-\gamma}(\mathbb{A}-\frac{2\alpha\gamma\varepsilon}{1-\gamma(1-\alpha)}).
$$

>  Theorem 4.1
>  令 $\mathbb A$ 为策略 $\pi'$ 相对于 $\pi, \mu$ 的策略优势 (遵照之前的定义，即 $\mathbb A_{\pi,\mu}(\pi')$)，令 $\epsilon = \max_s|E_{a\sim \pi'(a; s)}[A_\pi(s, a)]|$  ($\mathbb A$ 等价于对 $E_{a\sim \pi'(a; s)}[A_\pi(s, a)]$ 相对于 $s\sim d_{\pi,\mu}$ 求期望，故 $\epsilon$ 相当于期望内最大的那个项)
>  对于 Eq4.1 的更新规则，以及所有的更新系数参数 $\alpha \in [0, 1]$，更新后策略的目标 $\eta_\mu(\pi_{new})$ 相较于更新前策略的目标 $\eta_\mu(\pi)$ 的提升值满足以上的界

The proof of this theorem is given in the appendix. It is possible to construct a two state example showing this bound is tight for all $\alpha$ though we do not provide this example here. 
>  可以构造一个两状态示例，说明该界对于所有的 $\alpha$ 都是紧的

The first term is analogous to the first order increase specified in equation 4.2, and the second term is a penalty term. 
>  该界中，其第一个项类似于 Eq4.2 中的一阶增量 (应该是完全一致)，第二项是一个惩罚项 (类似于 Eq3.1 中的惩罚项，差异在于 $\epsilon$ 的定义不同，以及多乘上了 $\frac {\alpha}{1-\alpha}$)

Note that if $\alpha=1$ ,the bound reduces to 

$$
\eta_{\mu}(\pi_{\mathrm{new}})-\eta_{\mu}(\pi)\geq\frac{\mathbb{A}}{1-\gamma}-\frac{2\gamma\varepsilon}{1-\gamma}
$$ 
and the penalty term has the same form as that of greedy dynamic programming, where $\varepsilon$ ,as defined here, is analogous to the $l_{\infty}$ error in equation 3.1. 

>  当 $\alpha = 1$，该界的形式如上
>  注意到此时惩罚项的形式和贪心动态规划中的惩罚项 (Eq3.1) 具有相同的形式，其中 $\epsilon$ 定义为 $\epsilon = \max_s|E_{a\sim\pi'(a; s)}[A_\pi(s, a)]|= \max_s|E_{s\sim \pi'(a;s)}[Q_{\pi(s,a)}] - V_\pi(s)|$，也类似于 Eq3.1 中 $l_\infty$ 的定义

The following corollary shows that the greater the policy advantage the greater the guaranteed performance increase. 

**Corollary 4.2.** Let $R$ be the maximal possible reward and $\mathbb A$ be the policy advantage of $\pi^{\prime}$ with respect to $\pi$ and $\mu$. If $\mathbb{A}\geq0$ , then using $\begin{array}{r}{\alpha=\frac{(1-\gamma)\mathbb{A}}{4R}}\end{array}$ guarantees the following policy improvement: 

$$
\eta_{\mu}(\pi_{n e w})-\eta_{\mu}(\pi)\geq\frac{\mathbb{A}^{2}}{8R}.
$$ 
>  Corollary 4.2
>  令 $R$ 表示最大的可能奖励，$\mathbb A$ 表示 $\pi'$ 相较于 $\pi,\mu$ 的策略优势
>  若策略优势 $\mathbb A \ge 0$，则令更新系数 $\alpha = \frac {(1-\gamma)\mathbb A}{4R}$ 可以保证如上的策略提升幅度
>  可以看出，策略优势 $\mathbb A$ 越大，策略的提升幅度越大

>  根据作者在本节中的推导，他得出的结论是：当策略优势 $\mathbb A_{\pi,\mu}(\pi')\ge 0$，保守策略更新在 $\alpha = \frac {(1-\gamma)\mathbb A}{4R}$ 时保证可以提升策略的 $\eta_\mu$

Proof. Using the previous theorem, it is straightforward to show the change is bounded by $\textstyle{\frac{\alpha}{1-\gamma}}{\bigl(}\mathbb{A}{-}\alpha{\frac{2R}{1-\gamma}}{\bigr)}$ The corollary follows by choosing the $\alpha$ that maximizes this bound. 

>  推导
>  首先，根据 Theorem 4.1，策略提升满足的界为

$$
\begin{align}
\eta_\mu(\pi_{new}) - \eta_\mu(\pi) 
&\ge \frac {\alpha}{1-\gamma}\mathbb A - \frac \alpha {1-\gamma} \frac {2\alpha}{1-\gamma}\frac {\epsilon}{1-\alpha}\\
&\ge \frac {\alpha}{1-\gamma}\mathbb A - \frac \alpha {1-\gamma} \frac {2\alpha}{1-\gamma}\frac {\gamma}{1-\alpha}\max_s|E_{a\sim \pi'(a;s)}[A_\pi(s,a)]|
\end{align}
$$

>  考虑其中的最后一项

$$
\begin{align}
{\max_s|E_{a\sim \pi'(a;s)}[A_\pi(s,a)]|}
&=\max_s |E_{a\sim \pi'(a;s)}[Q_\pi(s,a)] - V_\pi(s)|\\
&=\max_s |\sum_a\pi'(s;a)[Q_\pi(s,a)] - V_\pi(s)|\\
\end{align}
$$

>  这里回忆 $\pi'$ 的定义，对于给定的 $s$，$\pi'(s; a)$ 的定义为

$$
\pi'(s;a) = \arg\max_a Q_\pi(s,a)
$$

>  故

$$
\begin{align}
&\max_s |\sum_a\pi'(s;a)[Q_\pi(s,a)] - V_\pi(s)|\\
=&\max_s|\max_a Q_\pi(s,a) - V_\pi(s)|\\
\le&\max_s|R - 0|\\
=&R\\
\le&\frac 1 \gamma R
\end{align}
$$

>  这里利用了 $V_\pi(s)\in [0, R],\forall s$ 以及 $Q_\pi(s, a)\in [0, R],\forall s, a$ 的性质
>  我们进而有

$$
\begin{align}
&\frac {\gamma}{1-\alpha}\max_s|E_{a\sim \pi'(a;s)}[A_\pi(s,a)]|\\
\le&\frac \gamma {1-\alpha}\frac 1 \gamma R\\
=&\frac 1 {1-\alpha}R\\
\end{align}
$$

>  故最后有

$$
\begin{align}
\eta_\mu(\pi_{new}) - \eta_\mu(\pi) 
&\ge \frac {\alpha}{1-\gamma}\mathbb A - \frac \alpha {1-\gamma} \frac {2\alpha}{1-\gamma}\frac {\gamma}{1-\alpha}\max_s|E_{a\sim \pi'(a;s)}[A_\pi(s,a)]|\\
&\ge \frac {\alpha}{1-\gamma}\mathbb A - \frac \alpha {1-\gamma} \frac {2\alpha}{1-\gamma}\frac 1 {1-\alpha}R\\
&\ge\frac \alpha {1-\gamma}\left(\mathbb A - \frac {2\alpha}{1-\alpha}\frac {R}{1-\gamma}\right)
\end{align}
$$

> 相较于作者推出的界，仍然多了系数 $\frac {1}{1-\alpha}$
> 并且实际上在作者推出的界中，令 $\alpha = \frac {(1-\gamma)\mathbb A}{4R}$ 也无法得到 $\frac {\mathbb A^2}{8R}$ 

## 4.2 Answering question 3 
We address question 3 by first addressing how fast we converge to some policy then bounding the quality of this policy. Naively, we expect our ability to obtain policies with large advantages to affect the speed of improvement and the quality of the final policy. Instead of explicitly suggesting algorithms that find policies with large policy advantages, we assume access to an $\varepsilon$ -greedy policy chooser that solves this problem. 
>  我们探讨 CPI 收敛到某个策略的速度，并对该策略的质量定界，以回答 Q3
>  直观上，我们希望能够获得较大优势的策略可以加速更新时的策略提升，并且提高最终策略的质量
>  我们不显示写出找到具有大策略优势的策略的算法，而是假设我们可以访问一个 $\epsilon$ -greedy 策略选择器，该策略选择器帮助我们解决该问题

Let us call this $\varepsilon$ good algorithm $G_{\varepsilon}(\pi,\mu)$ , which is defined as: 

**Definition4.3.** An $\varepsilon$ greedy policy chooser, $G_{\varepsilon}(\pi,\mu)$ ，is a function of a policy $\pi$ and a state distribution $\mu$ which returns a policy $\pi^{\prime}$ such that $\mathbb{A}_{\pi,\mu}(\pi^{\prime})~\ge~\mathrm{OPT}(\mathbb{A}_{\pi,\mu})-\varepsilon$ ，where $\operatorname{OPT}(\mathbb{A}_{\pi,\mu})\equiv$ $\operatorname*{max}_{\pi^{\prime}}\mathbb{A}_{\pi,\mu}(\pi^{\prime})$ 

>  我们将该 $\epsilon$ -greedy 策略选择器记作 $G_\epsilon(\pi,\mu)$，其定义如上
>  $G_\epsilon(\pi,\mu)$ 是关于策略 $\pi$ 和状态分布 $\mu$ 的函数，它返回 $\pi'$，$\pi'$ 的策略优势 $\mathbb A_{\pi,\mu}(\pi')$，满足上述的界，也就是仅比最佳的策略优势 (完全贪心情况下找到的 $\pi'$ 将达到最佳的策略优势) 低 $\epsilon$

In the discussion, we show that a regression algorithm that fits the advantages with an $\textstyle{\frac{\varepsilon}{2}}$ average error is sufficient to construct such a $G_{\varepsilon}$ 
>  我们将说明，以平均 $\frac \epsilon 2$ 的误差拟合优势的回归算法足以构建这样的贪心策略选择器 $G_\epsilon$ (即这样的回归算法足以拟合出满足上述界的新策略 $\pi'$)

The “breaking point” at which policy improvement is no longer guaranteed occurs when the greedy policy chooser is no longer guaranteed to return a policy with a positive policy advantage, i.e. when $\mathrm{OPT}(\mathbb{A}_{\pi,\mu})<\varepsilon$ . 
>  策略改进不再得到保证的 “临界点” 出现在贪心策略选择器不再保证返回具有正的策略优势的策略时出现，即在最佳的策略优势 $\text{OPT}(\mathbb A_{\pi,\mu}) < \epsilon$ 时

A crude outline of the Conservative Policy Iteration algorithm is: 

(1) Call $G_{\varepsilon}(\pi,\mu)$ to obtain some $\pi^{\prime}$ 
(2) Estimate the policy advantage $\mathbb{A}_{\pi,\mu}\left(\pi^{\prime}\right)$ 
(3) If the policy advantage is small (less than $\varepsilon$ ) STOP and return $\pi$ 
(4) Else, update the policy and go to (1). 

>  保守策略迭代算法的大致流程为
>  (1) 调用策略选择器 $G_\epsilon(\pi,\mu)$ 获得 $\pi'$
>  (2) 估计策略优势 $\mathbb A_{\pi,\mu}(\pi')$
>  (3) 如果策略优势很小 (小于 $\epsilon$)，则停止并返回 $\pi$
>  (4) 否则，更新策略，返回 (1)

where, for simplicity, we assume $\varepsilon$ is known. This algorithm ceases when an $\varepsilon$ small policy advantage (with respect to $\pi$ ) is obtained. By definition of the greedy policy chooser, it follows that the optimal policy advantage of $\pi$ is less than $2\varepsilon$ . 
>  为了简化起见，我们假设 $\epsilon$ 已知
>  该算法在获得一个策略优势小于 $\epsilon$ 的策略后停止，根据贪心策略选择器的定义，在停止时，任何比当前策略更好的策略相对于当前策略的策略优势都小于 $2\epsilon$ (因为停止时 $\mathbb A_{\pi,\mu}(\pi') < \epsilon$ ，故 $\text{OPT}(\mathbb A_{\pi,\mu}) < 2\epsilon$)

The full algorithm is specified in the next section. 

The following theorem shows that in polynomial time, the full algorithm finds a policy $\pi$ that is close to the “break point" of the greedy policy chooser. 
>  下面的定理表明，保守策略迭代的完整算法可以在多项式时间内找到一个策略 $\pi$，策略 $\pi$ 将接近贪心策略选择器的 “临界点”

**Theorem 4.4.** With probability at least $1-\delta$ ,conservative policy iteration: i) improves $\eta_{\mu}$ with every policy update, ii) ceases in at most $72\frac {R^2}{\epsilon^2}$ calls to $G_{\varepsilon}(\pi,\mu)$ ， and iii) returns a policy $\pi$ such that $\text{OPT}(\mathbb{A}_{\pi,\mu})<2\varepsilon$ 

>  Theorem 4.4
>  贪心策略迭代以至少 $1-\delta$ 的概率 
>  i) 在每一步策略更新提升 $\eta_\mu$ 
>  ii) 在最多调用 $G_\epsilon(\pi,\mu)$ $72\frac {R^2}{\epsilon^2}$ 次后停止
>  iii) 返回一个策略 $\pi$，满足 $\text{OPT}(\mathbb A_{\pi,\mu}) < 2\epsilon$

The proof is in the appendix. 

To complete the answer to question 3, we need to address the quality of the policy found by this algorithm. 
>  要完成对 Q3 的回答，我们需要讨论由 CPI 找到的策略的质量

Note that the bound on the time until our algorithm ceases does not depend on the restart distribution $\mu$ though the performance of the policy $\pi$ that we find does depend on $\mu$ since $\mathrm{OPT}(\mathbb{A}_{\pi,\mu})<2\varepsilon$ . 
>  注意，CPI 停止的时间界限不依赖于 restart distribution $\mu$，但找到的策略 $\pi$ 的性能依赖于 $\mu$，因为策略 $\pi$ 满足 $\text{OPT}(\mathbb A_{\pi,\mu} < 2\epsilon$)

Crudely, for a policy to have near optimal performance then all advantages must be small. Unfortunately, if $d_{\pi,\mu}$ is highly non-uniform, then a small optimal policy advantage does not necessarily imply that all advantages are small. The following corollary (of theorem 6.2) bounds the performance of interest, $\eta_{D}$ , for the policy found by the algorithm. We use the standard definition of the $l_{\infty}$ -norm, $\|{f}\|_{\infty}\equiv\operatorname*{max}_{s}~f(s)$ 
>  粗略地说，如果一个策略的性能接近最优，则所有其他策略相对于它的优势都必须很小
>  不幸的是，如果 $d_{\pi,\mu}$ 高度非均匀，则小的最优策略优势 ($\text{OPT}(\mathbb A_{\pi,\mu}$)) 并不一定意味着所有策略优势都小 (指初始分布为 $D$ 时)
>  以下的引理为算法找到的策略的性能相对于 $\eta_D$ 定界，我们使用标准的 $l_\infty$ 范数的定义，$\|f\|_\infty \equiv \max_s f(s)$

>  注意这里也提到了，$l_\infty$ 的标准定义直接记作 $\|f\|\equiv \max_sf(s)$

**Corollary 4.5.** Assume that for some policy $\pi$， $\text{OPT}(\mathbb{A}_{\pi,\mu})<\varepsilon$ . Let $\pi^{*}$ be an optimal policy. Then 

$$
\begin{array}{r}{\eta_{D}(\pi^{*})-\eta_{D}(\pi)\leq\displaystyle\frac{\varepsilon}{(1-\gamma)}\left\|\displaystyle\frac{d_{\pi^{*},D}}{d_{\pi,\mu}}\right\|_{\infty}}\ {\leq\displaystyle\frac{\varepsilon}{(1-\gamma)^{2}}\left\|\displaystyle\frac{d_{\pi^{*},D}}{\mu}\right\|_{\infty}.}\end{array}
$$ 
>  Corollary 4.5
>  假设对于某些策略 $\pi$，其最优策略优势 $\text{OPT}(\mathbb A_{\pi,\mu}) < \epsilon$
>  令 $\pi^*$ 为某个最优策略，则 $\eta_D(\pi^*) - \eta_D(\pi)$ 满足以上的界

The factor of $\left\|\frac{d_{\pi^{*},D}}{d_{\pi,\mu}}\right\|_{\infty}$ represents a mismatch between the distribution of states of the current policy and that of the optimal policy and elucidates the problem of using the given start-state distribution $D$ instead of a more uniform distribution. Essentially, a more uniform $d_{\pi,\mu}$ ensures that the advantages are small at states that an optimal policy visits (determined by $d_{\pi^{*},D}$ ). 
>  其中的因子 $\|\frac {d_{\pi^*, D}}{d_{\pi,\mu}}\|_\infty$ 表示了当前策略 $\pi$ 和最佳策略 $\pi^*$ 的状态分布之间的不匹配，并阐明了使用给定的起始分布 $D$ 而不是更均匀分布的问题
>  本质上，一个更加均匀的 $d_{\pi,\mu}$ 确保了在最优策略访问的状态 (由 $d_{\pi^*, D}$ 决定) 上的优势值较小 (如果初始分布选择 $D$，因子 $\|\frac {d_{\pi^*, D}}{d_{\pi,D}}\|_\infty$ 受到策略改变的影响较大，因为如果最优策略常访问的状态在初始策略下几乎不访问，则界就会很大，注意到这里的 $\infty$ 范数是取最大值，故 $d_{\pi,\mu}$ 更加均匀就能避免导致界过于大的情况)

The second inequality follows since $d_{\pi,\mu}(s)\geq(1-\gamma)\mu(s)$ , and it shows that a uniform measure prevents this mismatch from becoming arbitrarily large. 
>  第二个不等式源于 $d_{\pi,\mu}(s)\ge(1-\gamma)\mu(s)$ (根据 $d_{\pi,\mu}(s)$ 的定义推导即可)，它进一步说明了均匀的初始分布可以防止当前策略 $\pi$ 和最佳策略 $\pi^*$ 之间的状态分布之间的不匹配变得任意大

We now discuss and prove these theorems. 

# 5 Conservative Policy Iteration 
For simplicity, we assume knowledge of $\varepsilon$ . The conservative policy iteration algorithm is: 

(1) Call $G_{\varepsilon}(\pi,\mu)$ to obtain some $\pi^{\prime}$ 
(2) Use $\begin{array}{r}{O(\frac{R^{2}}{\varepsilon^{2}}\log\frac{R^{2}}{\delta\varepsilon^{2}})}\end{array}$ $\mu$ restarts to obtain an $\frac{\varepsilon}{3}$ accurate estimate $\hat{\mathbb A}$ of $\mathbb{A}_{\pi,\mu}\left(\pi^{\prime}\right)$ 
(3) If $\begin{array}{r}{\hat{\mathbb{A}}<\frac{2\varepsilon}{3}}\end{array}$ , STOP and return $\pi$ 
(4) If $\begin{array}{r}{\hat{\mathbb{A}}\ge\frac{2\varepsilon}{3}}\end{array}$ , then update policy $\pi$ according to equation 4.1 using $\frac{(1-\gamma)(\hat{\mathbb{A}}-\frac{\varepsilon}{3})}{4R}$ and return to step 1. 

>  简单起见，我们假设已知 $\epsilon$
>  保守策略迭代算法为
>  (1) 调用 $G_\epsilon(\pi,\mu)$ 获得某个 $\pi'$
>  (2) 执行 $O(\frac {R^2}{\epsilon^2}\log \frac {R^2}{\delta \epsilon^2})$ 次 $\mu$ restarts (应该是指执行这么多次的轨迹采样，用样本估计策略优势)，获得关于 $\mathbb A_{\pi,\mu}(\pi')$ 的 $\frac \epsilon 3$ 精确估计值 $\hat {\mathbb A}$
>  (3) 如果 $\hat {\mathbb A} < \frac {2\epsilon}{3}$，停止并返回 $\pi$
>  (4) 如果 $\hat {\mathbb A} \ge \frac {2\epsilon}{3}$，则根据 Eq4.1 使用 $\frac{(1-\gamma)(\hat{\mathbb{A}}-\frac{\varepsilon}{3})}{4R}$ 更新 $\pi$，然后返回第一步

where $\delta$ is the failure probability of the algorithm. Note that the estimation procedure of step (2) allows us to set the learning rate $\alpha$ 
>  其中 $\delta$ 是算法的失败概率
>  注意 step (2) 的估计过程允许我们设置学习率 $\alpha$

We now specify the estimation procedure of step (2) to obtain $\textstyle{\frac{\varepsilon}{3}}\cdot$ -accurate estimates of $\mathbb{A}_{\pi}\left(\pi^{\prime}\right)$ . It is straightforward to show that the policy advantage can be written as $\begin{array}{r c l}{\mathbb{A}_{\pi,\mu}\left(\pi^{\prime}\right)}&{=}&{E_{s\sim d_{\pi,\mu}}\left[\sum_{a}({\pi^{\prime}}(a;s)-\pi(a;s))Q_{\pi}(s,a)\right].}\end{array}$ We can obtain a nearly unbiased estimate $x_{i}$ of $\mathbb{A}_{\pi,\mu}\left(\pi^{\prime}\right)$ using one call to the $\mu$ -restart distribution (see definition 2.1). To obtain a sample $s$ from from $d_{\pi,\mu}$ , we obtain a trajectory from $s_{0}\sim\mu$ and accept the current state $s_{\tau}$ with probability $(1-\gamma)$ (see equation 2.1). 
>  我们详细说明 step (2) 获得 $\mathbb A_{\pi,\mu}(\pi')$ 的 $\frac \epsilon 3$ 精确估计的估计过程
>  首先，策略优势可以写为

$$
\mathbb A_{\pi,\mu}(\pi') = E_{s\sim d_{\pi,\mu}}\left[\sum_a(\pi'(a;s) - \pi(a;s))Q_\pi(s,a)\right]
$$

>  我们可以通过调用一次 $\mu$ (见 definition 2.1) 来获得 $\mathbb A_{\pi,\mu}(\pi')$ 的几乎无偏的估计值 $x_i$ (即直接从 $\mu$ 中采样 $s$)
>  为了从 $d_{\pi,\mu}$ 中获得样本 $s$，我们从 $s_0\sim \mu$ 开始生产轨迹，并且以概率 $(1-\gamma)$ 接收当前状态 $s_\tau$ (见 equation 2.1)

Then we chose an action $a$ from the uniform distribution, and continue the trajectory from $s$ to obtain a nearly unbiased estimate $\hat{Q}_{\pi}(s,a)$ of $Q_{\pi}(s,a)$ . Using importance sampling, the nearly unbiased estimate of the policy advantage from the $i$ -th sample is ${x_{i}=n_{a}\hat{Q}_{i}(s,a)(\pi^{\prime}(a;s)-\pi(a;s))}$ where $n_{a}$ is the number of actions. We assume that each trajectory is run sufficiently long such that the bias in $x_{i}$ is less than $\frac{\varepsilon}{6}$ 
>  然后，我们从均匀分布中选择一个动作 $a$，并从 $s$ ($s_\tau$) 继续轨迹以获得 $Q_\pi(s, a)$ 近乎无偏的估计 $\hat Q_\pi(s, a)$ (动作 $a$ 并非从 $\pi$ 中采样)
>  使用重要性采样，第 $i$ 次采样的策略优势的几乎无偏估计值为

$$
x_i = n_a\hat Q_i(s, a)(\pi'(a;s) - \pi(a;s))
$$

>  其中 $n_a$ 是动作的数量
>  我们假设每条轨迹运行得足够长，使得 $x_i$ 的变差小于 $\frac {\epsilon}{6}$

Since $\hat{Q}_{i}\in[0,R]$ , our samples satisfy $x_{i}\in$ $[-n_{a}R,n_{a}R]$ . Using Hoeffding's inequality for $k$ independent, identically distributed random variables, we have: 

$$
\mathrm{Pr}\left(|\mathbb{A}-\hat{\mathbb{A}}|>\Delta\right)\le2e^{-\frac{k\Delta^{2}}{2n_{a}^{2}R^{2}}},\tag{5.1}
$$ 
where ${\hat{\mathbb{A}}=\frac{1}{k}\sum_{i=1}^{k}x_{i}}$ (and $\mathbb A$ here is $\frac {\epsilon}{6}$ biased). Hence, the number of trajectories required to obtain an $\Delta$ -accurate sample with a fixed error rate is $\begin{array}{r}{O\left(\frac{n_{a}^{2}R^{2}}{\Delta^{2}}\right)}\end{array}$ 

>  因为 $\hat Q_i \in [0, R]$，我们的样本满足 $x_i \in [-n_a R, n_a R]$，使用 Hoeffding's inequality 对 $k$ 个独立同分布的随机变量进行分析，有 Eq 5.1 的不等式
>  Eq 5.1 说明了估计值相较于真实值的差异大于 $\Delta$ 的概率的界
>  其中 $\hat {\mathbb A} = \frac 1 k \sum_{i=1}^k x_i$，故为了以固定误差率获得一个 $\Delta$ -accurate 的样本，所需的轨迹数量为 $O(\frac {n_a^2R^2}{\Delta^2})$

The proof of theorem 4.4, which guarantees the soundness of conservative policy iteration, is straightforward and in the appendix. 

# 6 How Good is The Policy Found? 
Recall that the bound on the speed with which our algorithm ceases does not depend on the restart distribution used. In contrast, we now show that the quality of the resulting policy could strongly depend on this distribution. 
>  我们的算法的停止时间界并不依赖于使用的 restart distribution
>  相比之下，我们将证明所产生的策略的质量可能会强烈依赖于这个分布

The following lemma is useful: 

**Lemma 6.1.** For any policies $\tilde{\pi}$ and $\pi$ and any starting state distribution $\mu$ ， 

$$
\eta_{\mu}(\tilde{\pi})-\eta_{\mu}(\pi)=\frac{1}{1-\gamma}E_{(a,s)\sim\tilde{\pi}d_{\tilde{\pi},\mu}}[A_{\pi}(s,a)]
$$ 
>  Lemma 6.1
>  对于任意策略 $\tilde \pi$ 和 $\pi$ 以及任意起始状态分布 $\mu$，$\eta_\mu(\tilde \pi) - \eta_\mu(\pi)$ 满足上述等式

Proof. 
>  这里直接参照 TRPO 笔记中的证明，原文的证明比较抽象

This lemma elucidates a fundamental measure mismatch. The performance measure of interest, $\eta_{D}(\pi)$ changes in proportion the policy advantage $\mathbb{A}_{\pi,D}(\pi^{\prime})$ for small $\alpha$ (see equation 4.2), which is the average advantage under the state distribution $d_{\pi,D}$ . 
>  该引理阐明了一个基本的度量不匹配问题
>  我们感兴趣的度量 $\eta_D(\pi)$ 对于小的 $\alpha$ 值会按照策略优势 $\mathbb A_{\pi, D}(\pi')$ 成比例变化 (Eq4.2)，其中策略优势 $\mathbb A_{\pi, D}(\pi')$ 指在状态分布 $d_{\pi, D}$ 下的平均优势

However, for an optimal policy $\pi^{*}$ , the difference between $\eta_{D}(\pi^{*})$ and $\eta_{D}(\pi)$ is proportional to the average advantage under the state distribution $d_{\pi^{*},D}$ . 
>  但是，根据 Lemma 6.1，最优策略 $\pi^*$ 和 $\pi$ 的性能差异 $\eta_D(\pi^*), \eta_D(\pi)$ 则和状态分布 $d_{\pi^*, D}$ 下的平均优势成比例

Thus, even if the optimal policy advantage is small with respect to $\pi$ and $D$ , the advantages may not be small under $d_{\pi^{*},D}$ . This motivates the use of the more uniform distribution $\mu$ 
>  因此，即便相对于 $\pi, D$ 的最优策略优势很小，但这也是在 $d_{\pi, D}$ 下的结果，实际的，在 $d_{\pi^*, D}$ 下计算的优势可能并不小
>  这促使我们使用更均匀的分布 $\mu$

After termination, our algorithm returns a policy $\pi$ which has small policy advantage with respect to $\mu$ . We now quantify how far from optimal $\pi$ is, with respect to an arbitrary measure $\tilde{\mu}$ 
>  CPI 在结束后，返回的策略 $\pi$ 相对于 $\mu$ 有小的最优策略优势
>  我们量化 $\pi$ 相对于任意 $\tilde \mu$ 的最优策略优势

**Theorem 6.2.** Assume that for a policy $\pi$, $\text{OPT}(\mathbb{A}_{\pi,\mu})<\varepsilon$ .Let $\pi^{*}$ be an optimal policy. Then for any state distribution $\tilde{\mu}$ ， 

$$
{\eta_{\tilde{\mu}}(\pi^{*})-\eta_{\tilde{\mu}}(\pi)\leq\frac{\varepsilon}{(1-\gamma)}\left\lVert\frac{d_{\pi^{*},\tilde{\mu}}}{d_{\pi,\mu}}\right\rVert_{\infty}}\ {\leq\frac{\varepsilon}{(1-\gamma)^{2}}\left\lVert\frac{d_{\pi^{*},\tilde{\mu}}}{\mu}\right\rVert_{\infty}.}
$$ 

>  Theorem 6.2
>  有策略 $\pi$，满足 $\text{OPT}(\mathbb A_{\pi,\mu}) < \epsilon$，记最优策略为 $\pi^*$
>  对于任意状态分布 $\tilde \mu$，$\eta_{\tilde \mu}(\pi^*) - \eta_{\tilde \mu}(\pi)$ 满足如上的不等式

Proof. The optimal policy advantage is  $\mathrm{OPT}(\mathbb{A}_{\pi,\mu})=$ $\begin{array}{r}{\sum_{s}d_{\pi,\mu}(s)\operatorname*{max}_{a}A_{\pi}(s,a)}\end{array}$ . Therefore, 

$$
\begin{align}{\varepsilon}&{>}{\displaystyle\sum_{s}\frac{d_{\pi,\mu}(s)}{d_{\pi^{*},\mu}(s)}d_{\pi^{*},\tilde{\mu}}(s)\operatorname*{max}_{a}A_{\pi}(s,a)}\\ &{\geq}{\displaystyle\operatorname*{min}_{s}\left(\frac{d_{\pi,\mu}(s)}{d_{\pi^{*},\tilde{\mu}}(s)}\right)\sum_{s}d_{\pi^{*},\tilde{\mu}}(s)\operatorname*{max}_{a}A_{\pi}(s,a)}\\ &{\geq}{\displaystyle\left\|\frac{d_{\pi^{*},\tilde{\mu}}}{d_{\pi,\mu}}\right\|_{\infty}^{-1}\sum_{s,a}d_{\pi^{*},\tilde{\mu}}(s)\pi^{*}(a;s)A_{\pi}(s,a)}\\ &{=}{(1-\gamma)\left\|\frac{d_{\pi^{*},\tilde{\mu}}}{d_{\pi,\mu}}\right\|_{\infty}^{-1}(\eta_{\tilde{\mu}}(\pi^{*})-\eta_{\tilde{\mu}}(\pi))}\end{align}
$$ 
where the last step follows from lemma 6.1. The second inequality is due to $d_{\pi,\mu}(s)\leq(1-\gamma)\mu(s)$ 

Note that $\left\|{\frac{d_{\pi^{*},{\tilde{\mu}}}}{\mu}}\right\|_{\infty}$ is a measure of the mismatch in using $\mu$ rather than the future-state distribution of an optimal policy. 
>  其中 $\left\|{\frac{d_{\pi^{*},{\tilde{\mu}}}}{\mu}}\right\|_{\infty}$ 度量了使用 $\mu$ 而不是 $d_{\pi^*,\tilde \mu}$ 产生的不匹配程度

The interpretation of each factor of $\frac{1}{1-\gamma}$ is important. In particular, one factor of $\textstyle{\frac{1}{1-\gamma}}$ is due to the fact that difference between the performance of $\pi$ and optimal is ${\frac{1}{1-\gamma}}$ times the average advantage under $d_{\pi^{*},\tilde{\mu}}(s)$ (see lemma 6.1) and another factor of $\textstyle{\frac{1}{1-\gamma}}$ is due to the inherent non-uniformity of $d_{\pi,\mu}$ (since ${\dot{d}}_{\pi,\mu}(s)\leq(1-\gamma)\mu(s))$ 

# 7 Discussion 
We have provided an algorithm that finds an “approximately” optimal solution that is polynomial in the approximation parameter $\varepsilon$ ,but not in the size of the state space. 
>  我们提出了一个在关于近似参数 $\epsilon$ 的多项式时间内找到 “近似” 最优解的算法，该算法的收敛时间和状态空间大小无关 (但是其中的 Greedy Policy Chooser 的运行时间应该是和状态空间大小有关的)

We discuss a few related points. 

## 7.1 The Greedy Policy Chooser 
The ability to find a policy with a large policy advantage can be stated as a regression problem though we don't address the sample complexity of this problem. 
>  寻找具有更大策略优势的策略可以被表述为一个回归问题

Let us consider the error given by: 

$$
E_{s\sim d_{\pi,\mu}}\operatorname*{max}_{a}\left|A_{\pi}(s,a)-f_{\pi}(s,a)\right|.
$$ 
This loss is an average loss over the state space (though it is an $\infty$ -loss over actions). It is straightforward to see that if we can keep this error below $\textstyle{\frac{\varepsilon}{2}}$ , then we can construct an $\varepsilon$ -greedy policy chooser by choosing a greedy policy based on these approximation $f_{\pi}$ . 

>  考虑上述损失，它是在状态空间上的平均损失，且是动作空间上的 $\infty$ 损失
>  如果保持该误差在 $\frac \epsilon 2$ 以下，则就可以基于近似值 $f_\pi(s, a)$ 构造 $\epsilon$ -greedy 策略选择器

This $l_{1}$ condition for the regression problem is a much weaker constraint than minimizing an $l_{\infty}$ -error over the state-space, which is is the relevant error for greedy dynamic programming (see equation 3.1 and [3]). 
>  该回归损失对于状态空间上仅要求了 $l_1$ 条件，这是一个比 $l_\infty$ 弱许多的约束，而贪心动态规划要确保性能提升，就需要确保 $l_\infty$ 足够小 (Eq3.1)

Direct policy search methods could also be used to implement this greedy policy chooser. 



## 7.2 What about improving $\eta_{D}$ ？ 
Even though we ultimately seek to have good performance measure under $\eta_{D}$ , we show that it is important to improve the policy under a somewhat uniform measure. 
>  虽然我们最终要寻找 $\eta_D$ 最高的策略，我们也论证了在均匀的分布 $\mu$ 下提升策略的重要性

An important question is “Can we improve the policy according to both $\eta_{D}$ and $\eta_{\mu}$ at each update?" In general the answer is “no", but consider improving the performance under $\tilde{\mu}=(1-\beta)\mu+\beta D$ instead of just $\mu$ . This metric only slightly changes the quality of the asymptotic policy. 
>  考虑是否能在每步更新同时提升 $\eta_D, \eta_\mu$
>  一般来说，不能，但可以构造 $\tilde \mu =  (1-\beta)\mu + \beta D$，令策略基于 $\eta_{\tilde \mu}$ 提升
>  这样的构造仅略微改变策略的渐进质量

However by giving weight to $D$ ，the possibility of improving $\eta_{D}$ is allowed if the optimal policy has large advantages under $D$ , though we do not formalize this here. The only situation where joint improvement with $\eta_{D}$ is not possible is when $\mathrm{OPT}(\mathbb{A}_{\pi,D})$ is small. However, this is the problematic case where, under $D$ , the large advantages are not at states visited frequently. 
>  因为掺入了 $D$，故提高了提升 $\eta_D$ 的可能性

## 7.3 Implications of the mismatch 
The bounds we have presented directly show the importance of ensuring the agent starts in states where the optimal policy tends to visit. It also suggests that certain optimal policies are easier to learn in large state spaces — namely those optimal policies which tend to visit a significant fraction of the state space.
>  本文证明的界说明了确保智能体从最优策略倾向于访问的状态开始的重要性
>  也说明了在大的状态空间中，特定的最优策略更易于学习，具体地说，是倾向于访问状态空间中比较大的比例状态的最优策略

 An interesting suggestion for how to choose $\mu$ , is to use prior knowledge of which states an optimal policy tends to visit. 
 >  对 $\mu$ 的选择可以利用关于最优策略倾向于访问哪些状态的先验知识
