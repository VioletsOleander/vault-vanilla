# Abstract
Function approximation is essential to reinforcement learning, but the standard approach of approximating a value function and determining a policy from it has so far proven theoretically intractable. In this paper we explore an alternative approach in which the policy is explicitly represented by its own function approximator, independent of the value function, and is updated according to the gradient of expected reward with respect to the policy parameters. Williams's REINFORCE method and actor-critic methods are examples of this approach. 
>  函数近似是 RL 的关键，但目前先近似价值函数，根据近似价值函数确定策略的方法在理论上证明是不可解的
>  本文探索另一种方法，用与价值函数无关的函数近似器直接近似策略，根据策略参数相对于期望奖励的梯度来更新策略参数
>  REINFORCE 方法和 actor-critic 方法就是这种思路的实例

Our main new result is to show that the gradient can be written in a form suitable for estimation from experience aided by an approximate action-value or advantage function. Using this result, we prove for the first time that a version of policy iteration with arbitrary differentiable function approximation is convergent to a locally optimal policy.
>  我们的主要结果是证明了策略梯度可以以一种适合从经验中估计的形式表达，这种估计需要通过近似的动作价值函数或优势函数来辅助
>  利用这一结果，我们首次证明了一个新版本的 policy iteration 算法，它只要使用任意可微的函数近似器，就能够收敛到一个局部最优策略

# 0 Introduction
Large applications of reinforcement learning (RL) require the use of generalizing function approximators such neural networks, decision-trees, or instance-based methods. 
>  RL 的应用要求使用能够泛化的函数近似器，例如神经网络、决策树或基于实例的方法

The dominant approach for the last decade has been the value-function approach, in which all function approximation effort goes into estimating a value function, with the action-selection policy represented implicitly as the "greedy" policy with respect to the estimated values (e.g., as the policy that selects in each state the action with highest estimated value). 
>  过去十年的主流方法都是价值函数方法，即近似估计价值函数，策略则表示为对价值函数的贪心动作选择 (例如在每个状态下选择具有最高估计价值的策略)

The value-function approach has worked well in many applications, but has several limitations. First, it is oriented toward finding deterministic policies, whereas the optimal policy is often stochastic, selecting different actions with specific probabilities (e.g., see Singh, Jaakkola, and Jordan, 1994). Second, an arbitrarily small change in the estimated value of an action can cause it to be, or not be, selected. Such discontinuous changes have been identified as a key obstacle to establishing convergence assurances for algorithms following the value-function approach (Bertsekas and Tsitsiklis, 1996). 
>  价值函数方法在许多应用中表现良好，但存在几点限制
>  - 首先，其目标是找到确定性的策略，而最优策略通常是随机性的，即以特定的概率选择不同的动作
>  - 其次，对某个动作的估计价值进行微小的变化就有可能导致这个动作被选中或不被选中，这种不连续的变化被认为是阻碍价值函数方法的收敛保证的关键障碍

For example, Q-learning, Sarsa, and dynamic programming methods have all been shown unable to converge to any policy for simple MDPs and simple function approximators (Gordon, 1995, 1996; Baird, 1995; Tsitsiklis and van Roy, 1996; Bertsekas and Tsitsiklis, 1996). This can occur even if the best approximation is found at each step before changing the policy, and whether the notion of "best" is in the mean-squared-error sense or the slightly different senses of residual-gradient, temporal-difference, and dynamic-programming methods.
>  例如 Q-learning, Sarsa 和动态规划方法都被证明了无法在简单的 MDP 和简单的函数近似器收敛到任何策略
>  即便在每一步都找到最佳的 (价值) 近似，再改变策略，这种情况也可能会发生，无论 “最佳” 的定义均方误差意义上、还是残差梯度、时间差分、动态规划方法

In this paper we explore an alternative approach to function approximation in RL.
>  本文讨论 RL 中的另一种函数近似方法

Rather than approximating a value function and using that to compute a deterministic policy, we approximate a stochastic policy directly using an independent function approximator with its own parameters. 
>  与其近似价值函数，然后计算确定性策略，我们直接近似一个随机策略

For example, the policy might be represented by a neural network whose input is a representation of the state, whose output is action selection probabilities, and whose weights are the policy parameters. 
>  例如，策略可以由一个神经网络表示，其输入是状态的表示，输出是动作选择概率，权重是策略参数

Let $\theta$ denote the vector of policy parameters and $\rho$ the performance of the corresponding policy (e.g., the average reward per step). Then, in the policy gradient approach, the policy parameters are updated approximately proportional to the gradient.

$$
\Delta \theta \approx \alpha \frac{\partial\rho}{\partial\theta}, \tag{1}
$$

where $\alpha$ is a positive-definite step size. 

>  令 $\theta$ 表示策略参数向量，$\rho$ 表示策略的性能 (例如，每一步的平均奖励)
>  则在策略梯度方法中，策略函数按照与策略梯度 $\Delta \theta \approx \alpha \frac {\partial \rho}{\partial \theta}$ 近似成比例的方式更新
>  其中 $\alpha$ 是一个正定的学习率

If the above can be achieved, then $\theta$ can usually be assured to converge to a locally optimal policy in the performance measure $\rho$ . Unlike the value-function approach, here small changes in $\theta$ can cause only small changes in the policy and in the state-visitation distribution.
>  如果可以实现上述的更新，则 $\theta$ 通常保证会收敛到一个相对于性能度量 $\rho$ 的局部最优策略
>  和基于价值函数的方法不同，在基于策略梯度方法中，对参数 $\theta$ 的微小只会引起策略和状态访问分布的微小变化

In this paper we prove that an unbiased estimate of the gradient (1) can be obtained from experience using an approximate value function satisfying certain properties. Williams's (1988, 1992) REINFORCE algorithm also finds an unbiased estimate of the gradient, but without the assistance of a learned value function. REINFORCE learns much more slowly than RL methods using value functions and has received relatively little attention. 
>  本文中，我们证明可以通过使用满足某种性质的近似价值函数，从经验中获取对梯度 (1) 的无偏估计
>  REINFORCE 算法也找到了梯度的一个无偏估计，且不需要一个学习到的价值函数 (毕竟人家直接用的纯 Monte Carlo)，REINFORCE 算法的学习速度比使用价值函数的 RL 方法要慢得多，故受到的关注较少

Learning a value function and using it to reduce the variance of the gradient estimate appears to be essential for rapid learning. Jaakkola, Singh and Jordan (1995) proved a result very similar to ours for the special case of function approximation corresponding to tabular PQMDPs. Our result strengthens theirs and generalizes it to arbitrary differentiable function approximators. Konda and Tsitsiklis (in prep.) independently developed a very similar result to ours. See also Baxter and Bartlett (in prep.) and Marbach and Tsitsiklis (1998).
>  学习一个价值函数，并且用这个学习到的价值函数来减少梯度估计的方差似乎对于快速学习是至关重要的 (减少方差是因为采样少了，当然相应地偏差就大了，快速学习同样是因为采样少了)
>  Jaakkola 等人针对表格型 PQMDPs 的函数近似特殊情况证明了和我们非常相似的结果，我们的结果加强了他们的结论，将其推广到了任意可微的函数近似器

Our result also suggests a way of proving the convergence of a wide variety of algorithms based on "actor-critic" or policy-iteration architectures (e.g., Barto, Sutton, and Anderson, 1983; Sutton, 1984; Kimura and Kobayashi, 1998). In this paper we take the first step in this direction by proving for the first time that a version of policy iteration with general differentiable function approximation is convergent to a locally optimal policy. 
>  我们的结果也提供了一种证明大量的基于 actor-critic 或策略迭代架构的算法的收敛性的方法框架思路
>  在本文中，我们迈出第一步，首次证明了采用通用的可微函数近似的一类策略迭代方法可以收敛到局部最优的策略

Baird and Moore (1999) obtained a weaker but superficially similar result for their VAPS family of methods. Like policy-gradient methods, VAPS includes separately parameterized policy and value functions updated by gradient methods. However, VAPS methods do not climb the gradient of performance (expected long-term reward), but of a measure combining performance and value-function accuracy. As a result, VAPS does not converge to a locally optimal policy, except in the case that no weight is put upon value-function accuracy, in which case VAPS degenerates to REINFORCE. 
>  Baird 等人在他们的 VAPS 系列方法上得到了更弱，但表面上和我们相似的结果
>  类似于策略梯度方法，VAPS 也包含了分别参数化的策略函数和价值函数，并用梯度方法更新
>  但是 VAPS 方法并不是沿着性能 (期望的长期奖励) 的梯度更新，而是沿着结合了性能和价值函数准确性的梯度更新
>  因此，除了在不考虑价值函数准确性的情况下，VAPS 会退化到 REINFORCE 外，它并不能收敛到一个局部最优策略

Similarly, Gordon's (1995) fitted value iteration is also convergent and value-based, but does not find a locally optimal policy.

# 1 Policy Gradient Theorem
We consider the standard reinforcement learning framework (see, e.g., Sutton and Barto, 1998), in which a learning agent interacts with a Markov decision process (MDP). The state, action, and reward at each time $t \in \{0, 1, 2, \ldots \}$ are denoted $s_t \in \mathcal{S}$ , $a_t \in \mathcal{A}$ , and $r_t \in \mathcal{R}$ respectively. The environment's dynamics are characterized by state transition probabilities, $\mathcal{P}_{ss'}^a = Pr\left\{s_{t + 1} = s' \mid s_t = s, a_t = a\right\}$ , and expected rewards $\mathcal{P}_s^a = E\left\{r_{t + 1} \mid s_t = s, a_t = a\right\}$ , $\forall s, s' \in \mathcal{S}$ , $a \in \mathcal{A}$ . 

The agent's decision making procedure at each time is characterized by a policy, $\pi (s, a, \theta) = Pr\left\{a_t = a \mid s_t = s, \theta \right\}$ , $\forall s \in \mathcal{S}$ , $a \in \mathcal{A}$ , where $\theta \in \mathbb{R}^l$ , for $l < < |\mathcal{S}|$ , is a parameter vector. 

We assume that $\pi$ is differentiable with respect to its parameter, i.e., that $\frac{\partial \pi(s, a)}{\partial \theta}$ exists. We also usually write just $\pi (s, a)$ for $\pi (s, a, \theta)$ .
>  我们假设策略函数 $\pi$ 相对于它的参数 $\theta$ 是可微的，即 $\frac {\partial \pi(s, a)}{\partial \theta}$ 存在

With function approximation, two ways of formulating the agent's objective are useful. 
>  目标函数的构造有两种形式

One is the average reward formulation, in which policies are ranked according to their long-term expected reward per step, $\rho (\pi)$ :

$$
\rho (\pi) = \lim_{n\to \infty}\frac{1}{n} E\left\{r_1 + r_2 + \dots +r_n\mid \pi \right\} = \sum_s d^\pi (s)\sum_a\pi (s,a)R_s^\alpha ,
$$

where $d^{\pi}(s) = \lim_{t\to \infty}Pr\left\{s_t = s\big|s_0,\pi \right\}$ is the stationary distribution of states under $\pi$ , which we assume exists and is independent of $s_0$ for all policies. 

>  第一种形式是构造为平均奖励
>  在该构造中，评价一个策略的指标 $\rho(\pi)$ 定义为 Markov 过程达到稳态分布后，按照 $\pi$ 行动时，每一步获取的奖励的平均值
>  这里我们假设了任意策略都存在相应的稳态分布，并且从任意的起始状态出发都能达到稳态分布

 In the average reward formulation, the value of a state-action pair given a policy is defined as

$$
Q^{\pi}(s,a) = \sum_{t = 1}^{\infty}E\left\{r_t - \rho (\pi)\mid s_0 = s,s_0 = a,\pi \right\} ,\qquad \forall a\in S,a\in \mathcal{A}.
$$

The second formulation we cover is that in which there is a designated start state $s_0$ , and we care only about the long-term reward obtained from it. We will give our results only once, but they will apply to this formulation as well under the definitions

$$
\rho (\pi) = E\bigg\{\sum_{t = 1}^{\infty}\gamma^{t - 1}r_t\bigg|s_0,\pi \bigg\} \quad \mathrm{and}\quad Q^{\pi}(s,a) = E\bigg\{\sum_{k = 1}^{\infty}\gamma^{k - 1}r_{t + k}\bigg|s_t = s,a_t = a,\pi \bigg\} .
$$

where $\gamma \in [0,1]$ is a discount rate $(\gamma = 1$ is allowed only in episodic tasks). In this formulation, we define $d^{\pi}(s)$ as a discounted weighting of states encountered starting at $s_0$ and then following $\pi$ : $d^{\pi}(s) = \sum_{t = 0}^{\infty}\gamma^{t}Pr\left\{s_t = s\big|s_0,\pi \right\}$ .

>  第二种构造是从一个指定的起始状态 $s_0$ 出发，遵循策略 $\pi$ 获得的期望折扣奖励和
>  此处，$\rho(\pi)$ 就是我们熟悉的 $s_0$ 的状态价值函数，$Q^\pi(s, a)$ 就是我们熟悉的动作价值函数

Our first result concerns the gradient of the performance metric with respect to the policy parameter:

**Theorem 1 (Policy Gradient).** For any MDP, in either the average-reward or start-state formulations,

$$
\frac{\partial\rho}{\partial\theta} = \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a). \tag{2}
$$

Proof: See the appendix.

>  我们的第一个结果是策略梯度定理，即目标函数 $\rho$ 相对于策略参数 $\theta$ 的梯度形式

This way of expressing the gradient was first discussed for the average- reward formulation by Marbach and Tsitsiklis (1998), based on a related expression in terms of the state-value function due to Jaakkola, Singh, and Jordan (1995) and Cao and Chen (1997). We extend their results to the start- state formulation and provide simpler and more direct proofs. Williams's (1988, 1992) theory of REINFORCE algorithms can also be viewed as implying (2). 

In any event, the key aspect of both expressions for the gradient is that their are no terms of the form $\frac{\partial d^{\pi}(s)}{\partial\theta}$ : the effect of policy changes on the distribution of states does not appear. This is convenient for approximating the gradient by sampling. For example, if $s$ was sampled from the distribution obtained by following $\pi$ , then $\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a)$ would be an unbiased estimate of $\frac{\partial\rho}{\partial\theta}$ . Of course, $Q^{\pi}(s,a)$ is also not normally known and must be estimated. One approach is to use the actual returns, $R_{t} = \sum_{k = 1}^{\infty}r_{t + k} - \rho (\pi)$ (or $R_{t} = \sum_{k = 1}^{\infty}\gamma^{k - 1}r_{t + k}$ in the start-state formulation) as an approximation for each $Q^{\pi}(s_t,a_t)$ . This leads to Williams's episodic REINFORCE algorithm, $\Delta \theta_t\propto \frac{\partial\pi(s_t,a_t)}{\partial\theta} R_t\frac{1}{\pi(s_t,a_t)}$ (the $\frac{1}{\pi(s_t,a_t)}$ corrects for the oversampling of actions preferred by $\pi$ ), which is known to follow $\frac{\partial\rho}{\partial\theta}$ in expected value (Williams, 1988, 1992).
>  策略梯度定理的表达式中，关键的点在于它不包含形式为 $\frac {\partial d^\pi(s)}{\partial \theta}$ 的项: 策略变化对于状态分布的影响没有出现，这对于通过采样来估计梯度非常方便
>  策略梯度定理中，$Q^\pi(s, a)$ 通常是未知的，也需要被估计
>  一种估计方式是使用真实的回报，此即 REINFORCE 算法

# 2 Policy Gradient with Approximation
Now consider the case in which $Q^{\pi}$ is approximated by a learned function approximator. If the approximation is sufficiently good, we might hope to use it in place of $Q^{\pi}$ in (2) and still point roughly in the direction of the gradient. 
>  我们考虑 $Q^\pi$ 由一个学习到的函数近似器近似
>  如果近似得足够好，我们可以用近似值计算策略梯度，仍然可以大致得到梯度的方向

For example, Jaakkola, Singh, and Jordan (1995) proved that for the special case of function approximation arising in a tabular POMDP one could assure positive inner product with the gradient, which is sufficient to ensure improvement for moving in that direction. Here we extend their result to general function approximation and prove equality with the gradient.

Let $f_{w}: \mathcal{A} \times \mathcal{A} \to \mathfrak{R}$ be our approximation to $Q^{\pi}$ , with parameter $w$ . It is natural to learn $f_{w}$ by following $\pi$ and updating $w$ by a rule such as $\Delta w_t \propto \frac{\partial}{\partial w} [Q^{\pi}(s_t, a_t) - f_w(s_t, a_t)]^2 \propto [\hat{Q}^{\pi}(s_t, a_t) - f_w(s_t, a_t)]\frac{\partial f_w(s_t, a_t)}{\partial w}$ , where $\hat{Q}^{\pi}(s_t, a_t)$ is some unbiased estimator of $Q^{\pi}(s_t, a_t)$ , perhaps $R_t$ . When such a process has converged to a local optimum, then

$$
\sum_{s}d^{\pi}(s)\sum_{a}\pi (s,a)[Q^{\pi}(s,a) - f_{w}(s,a)]\frac{\partial f_{w}(s,a)}{\partial w} = 0. \tag{3}
$$

>  我们用函数 $f_w$ 作为动作价值函数的近似，它由 $w$ 参数化
>  学习 $f_w$ 的自然方式就是梯度更新，目标函数是预测值 $f_w(s, a)$ 和真实值 (的采样估计值) $\hat Q(s, a)$ 之间的期望均方误差
>  当 $f_w$ 收敛到局部最优时，将满足 Eq 3，这是最小二乘法在函数近似收敛时的正交条件: 损失和梯度正交

**Theorem 2 (Policy Gradient with Function Approximation).** If $f_{w}$ satisfies (3) and is compatible with the policy parameterization in the sense that

$$
\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial\pi(s,a)}{\partial\theta}\frac{1}{\pi(s,a)}, \tag{4}
$$

then

$$
\frac{\partial\rho}{\partial\theta} = \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} f_{w}(s,a). \tag{5}
$$

>  如果我们学习到的动作价值函数近似 $f_w$ 已经收敛，并且在平均意义上与真实的 $Q^\pi$ 误差最小化 (即满足 Eq 3)，并且 $f_w$ 和策略参数化兼容，即满足 Eq 4
>  此时我们可以保证，直接将 $f_w(s, a)$ 替换掉策略梯度定理中的动作价值函数，得到的是一个对目标梯度的无偏估计，即满足 Eq 5

>  Eq 4 (兼容函数近似) 的成立的最简单的情况:
>  假设策略和价值函数都是用线性函数近似，且共享特征 (或基函数)
>  策略: $\log \pi(s, a) = \theta^T\phi(s, a)$
>  价值函数: $f_w(s, a) = w^T\phi(s, a)$
>  其中 $\phi(s, a)$ 是一个特征向量
>  在这种情况下

$$
\frac {\partial f_w(s, a)}{\partial w} = \phi(s, a) = \frac {\partial \log \pi(s, a)}{\partial \theta}
$$

>  即兼容性条件成立
>  因此，在经典的 actor-critic 方法中，critic 和 actor 通常也会使用相同的基函数 (或特征) 来进行近似
>  深度学习时代，网络的复杂性使得严格保证兼容性变得困难，但深度学习算法在经验上还是取得了成功，原因大致有:
>  - 优势的使用，通过减去基线，减小了方差，故即便梯度估计是有偏的，但方向大致对，训练稳定，算法依旧可以收敛到一个局部最优
>  - 大数据量和大参数量

Proof: Combining (3) and (4) gives

$$
\sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta}\big[Q^{\pi}(s,a) - f_{w}(s,a)\big] = 0 \tag{6}
$$

which tells us that the error in $f_{w}(s, a)$ is orthogonal to the gradient of the policy parameterization. Because the expression above is zero, we can subtract it from the policy gradient theorem (2) to yield

以下是使用 `align` 环境重新排版后的公式：

$$
\begin{align}
\frac{\partial\rho}{\partial\theta} 
&= \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a) - \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta}\big[Q^{\pi}(s,a) - f_{w}(s,a)\big] \\
&= \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta}\big[Q^{\pi}(s,a) - Q^{\pi}(s,a) + f_{w}(s,a)\big] \\
&= \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} f_{w}(s,a).
\end{align}
\tag{Q.E.D.}
$$

# 3 Application to Deriving Algorithms and Advantages
Given a policy parameterization, Theorem 2 can be used to derive an appropriate form for the value-function parameterization. 
>  给定一个策略的参数化形式，可以根据定理 2 为价值函数的参数化推导一个合适的形式 (即让价值函数的参数化形式满足兼容性条件)

For example, consider a policy that is a Gibbs distribution in a linear combination of features:

$$
\pi (s,a) = \frac{e^{\theta^T\phi_{sb}}}{\sum_{b}e^{\theta^T\phi_{sb}}},\qquad \forall s\in \mathcal{S},s\in \mathcal{A},
$$

where each $\phi_{sa}$ is an $l$ - dimensional feature vector characterizing state- action pair $s, a$ . Meeting the compatibility condition (4) requires that

$$
\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial\pi(s,a)}{\partial\theta}\frac{1}{\pi(s,a)} = \phi_{sa} - \sum_b\pi (s,b)\phi_{sb},
$$

so that the natural parameterization of $f_w$ is

$$
f_{w}(s,a) = w^{T}\left[\phi_{sa} - \sum_{b}\pi (s,b)\phi_{sb}\right].
$$

In other words, $f_w$ must be linear in the same features as the policy, except normalized to be mean zero for each state. Other algorithms can easily be derived for a variety of nonlinear policy parameterizations, such as multi- layer backpropagation networks.

The careful reader will have noticed that the form given above for $f_w$ requires that it have zero mean for each state: $\sum_{a}\pi (s,a)f_{w}(s,a) = 0$ , $\forall s\in S$ . In this sense it is better to think of $f_w$ as an approximation of the advantage function, $A^\pi (s,a) = Q^\pi (s,a) - V^\pi (s)$ (much as in Baird, 1993), rather than of $Q^\pi$ .
>  可以发现上述的对 $f_w$ 的近似形式要求它在每个状态上的均值为零，从这个角度来看，将 $f_w$ 看作是对优势函数 $A^\pi(s,  a)  = Q^\pi(s, a) - V^\pi(s)$ 的近似更加合适

Our convergence requirement (3) is really that $f_w$ gets the relative value of the actions correct in each state, not the absolute value, nor the variation from state to state. Our results can be viewed as a justification for the special status of advantages as the target for value function approximation in RL. 
>  实际上，Eq 3 的收敛条件值要求 $f_w$ 在每个状态下正确地反映动作之间的相对价值，而不是绝对价值，也不是状态之间的变化
>  我们的结果可以看作是对 RL 中优势函数作为价值函数的近似目标的特殊地位的一种理论依据

In fact, our (2), (3), and (5), can all be generalized to include an arbitrary function of state added to the value function or its approximation. For example, (5) can be generalized to $\frac{\partial\rho}{\partial\theta} = \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta}\left[f_{w}(s,a) + v(s)\right]$ , where $v:S\to \mathbb{R}$ is an arbitrary function. (This follows immediately because $\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} = 0$ , $\forall s\in S$ .) 
>  实际上，Eq 2, 3, 5 都可以推广到包含任意关于状态的函数加到价值函数或其近似上
>  例如，Eq 5 可以泛化到 $f_w(s, a)$ 加上一个 $v(s)$，其中 $v(s)$ 是任意关于状态的函数 (这是因为 $\sum_a \frac {\partial \pi(s, a)}{\partial \theta} = \frac {\partial \sum_a \pi(s, a)}{\partial \theta} = 0$，其中 $\sum_a \pi(s, a) = 1$)

The choice of $v$ does not affect any of our theorems, but can substantially affect the variance of the gradient estimators. The issues here are entirely analogous to those in the use of reinforcement baselines in earlier work (e.g., Williams, 1992; Dayan, 1991; Sutton, 1984). In practice, $v$ should presumably be set to the best available approximation of $V^\pi$ . Our results establish that that approximation process can proceed without affecting the expected evolution of $f_w$ and $\pi$ .
>  函数 $v$ 的选择不会影响结论，但会显著影响梯度估计的方差
>  这里的问题和早期使用强化基线的问题完全一致，在实践中，$v$ 应该被设定为对 $V^\pi$ 的最好的可用近似
>  本文的结果也证明了: 学习和改进基线函数 $v$ 的过程可以独立于价值函数学习和策略学习进行，而不必担心引入偏差

# 4 Convergence of Policy Iteration with Function Approximation
Given Theorem 2, we can prove for the first time that a form of policy iteration with function approximation is convergent to a locally optimal policy.

**Theorem 3 (Policy Iteration with Function Approximation).** Let $\pi$ and $f_w$ be any differentiable function approximators for the policy and value function respectively that satisfy the compatibility condition (4) and for which $\max_{\theta ,s,a,i,j}\left|\frac{\partial^2\pi(s,a)}{\partial\theta_i\partial\theta_j}\right|< B< \infty$ . Let $\{\alpha_k\}_{k = 0}^{\infty}$ be any step- size sequence such that $\lim_{k\to \infty}\alpha_k = 0$ and $\sum_{k}\alpha_{k} = \infty$ . Then, for any MDP with bounded rewards, the sequence $\{d(\pi_k)f_{k = 0}^{\infty}\}$ defined by any $\theta_0$ , $\pi_k = \pi (\cdot ,\cdot ,\theta_k)$ , and

$$
\begin{array}{rcl}{w_k} & = & {w\mathrm{~such~that~}\sum_s d^{\pi_k}(s)\sum_a\pi_k(s,a)[Q^{\pi_k}(s,a) - f_w(s,a)]\frac{\partial f_w(s,a)}{\partial w} = 0}\\ {\theta_{k + 1}} & = & {\theta_k + \alpha_k\sum_s d^{\pi_k}(s)\sum_a\frac{\partial\pi_k(s,a)}{\partial\theta} f_{w_k}(s,a),} \end{array}
$$

converges such that $\lim_{k\to \infty}\frac{\partial\rho(\pi_k)}{\partial\theta} = 0$

Proof: Our Theorem 2 assures that the $\theta_k$ update is in the direction of the gradient. The bounds on $\frac{\partial^2\pi(s,a)}{\partial\theta_i\partial\theta_j}$ and on the MDP's rewards together assure us that $\frac{\partial^2\rho}{\partial\theta_i\partial\theta_j}$ is also bounded. These, together with the step-size requirements, are the necessary conditions to apply Proposition 3.5 from page 96 of Bertsekas and Tsitsiklis (1996), which assures convergence to a local optimum. Q.E.D.

# Appendix: Proof of Theorem 1
We prove the theorem first for the average-reward formulation and then for the start-state formulation.

$$
\begin{align}
\frac{\partial V^{\pi}(s)}{\partial\theta} &\stackrel{\mathrm{def}}{=}\frac{\partial}{\partial\theta}\sum_{a}\pi (s,a)Q^{\pi}(s,a) \quad \forall s\in \mathcal{S} \\
&= \sum_{a}\left[\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a) + \pi (s,a)\frac{\partial}{\partial\theta} Q^{\pi}(s,a)\right] \\
&= \sum_{a}\left[\frac{\partial\pi(s,a)}{\partial\theta}Q^{\pi}(s,a)+\pi(s,a)\frac{\partial}{\partial\theta}\left[\mathcal{R}_{s}^{a}-\rho(\pi)+\sum_{s^{\prime}}\mathcal{P}_{s s^{\prime}}^{a}V^{\pi}(s^{\prime})\right]\right] \\
&= \sum_{a}\left[\frac{\partial\pi(s,a)}{\partial\theta}Q^{\pi}(s,a)+\pi(s,a)\left[-\frac{\partial\rho}{\partial\theta}+\sum_{s^{\prime}}\mathcal{P}_{s s^{\prime}}^{a}\frac{\partial V^{\pi}(s^{\prime})}{\partial\theta}\right]\right]
\end{align}
$$

Therefore,

$$
\frac{\partial\rho}{\partial\theta} = \sum_{a}\left[\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a) + \pi (s,a)\sum_{s^{\prime}}\mathcal{P}_{ss^{\prime}}^{a}\frac{\partial V^{\pi}(s^{\prime})}{\partial\theta}\right] - \frac{\partial V^{\pi}(s)}{\partial\theta}
$$

Summing both sides over the stationary distribution $d^{\pi}$ ,

$$
\begin{array}{rcl}\sum_{s}d^{\pi}(s)\frac{\partial\rho}{\partial\theta} & = & \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a) + \sum_{s}d^{\pi}(s)\sum_{a}\pi (s,a)\sum_{s^{\prime}}\mathcal{P}_{ss^{\prime}}^{a}\frac{\partial V^{\pi}(s^{\prime})}{\partial\theta}\\ & & -\sum_{s}d^{\pi}(s)\frac{\partial V^{\pi}(s)}{\partial\theta}, \end{array}
$$

but since $d^{\pi}$ is stationary,

$$
\begin{array}{rcl}\sum_{s}d^{\pi}(s)\frac{\partial\rho}{\partial\theta} & = & \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a) + \sum_{s^{\prime}}d^{\pi}(s^{\prime})\frac{\partial V^{\pi}(s^{\prime})}{\partial\theta}\\ & & -\sum_{s}d^{\pi}(s)\frac{\partial V^{\pi}(s)}{\partial\theta}\\ \frac{\partial\rho}{\partial\theta} & = & \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a). \end{array}
$$

For the start-state formulation:

$$
\begin{array}{rcl}\frac{\partial V^{\pi}(s)}{\partial\theta} & \stackrel {\mathrm{def}}{=} & \frac{\partial}{\partial\theta}\sum_{a}\pi (s,a)Q^{\pi}(s,a)\quad \forall s\in \mathcal{S}\\ & = & \sum_{a}\left[\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a) + \pi (s,a)\frac{\partial}{\partial\theta} Q^{\pi}(s,a)\right]\\ & = & \sum_{a}\left[\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a) + \pi (s,a)\frac{\partial}{\partial\theta}\left[\mathcal{R}_s^a +\sum_{s'}\gamma \mathcal{P}_{ss'}^a V^{\pi}(s')\right]\right]\\ & = & \sum_{a}\left[\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a) + \pi (s,a)\sum_{s'}\gamma \mathcal{P}_{ss'}^a\frac{\partial}{\partial\theta} V^{\pi}(s')\right]\\ & = & \sum_{a}\sum_{k = 0}^{\infty}\gamma^k Pr(s\to x,k,\pi)\sum_{a}\frac{\partial\pi(x,a)}{\partial\theta} Q^{\pi}(x,a), \end{array} \tag{7}
$$

after several steps of unrolling (7), where $Pr(s\to x,k,\pi)$ is the probability of going from state $s$ to state $x$ in $k$ steps under policy $\pi$ . It is then immediate that

$$
\begin{array}{rcl}\frac{\partial\rho}{\partial\theta} & = & \frac{\partial}{\partial\theta} E\Big\{\sum_{t = 1}^{\infty}\gamma^{t - 1}r_t\Big|s_0,\pi \Big\} = \frac{\partial}{\partial\theta} V^{\pi}(s_0)\\ & = & \sum_{s}\sum_{k = 0}^{\infty}\gamma^k Pr(s_0\to s,k,\pi)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a)\\ & = & \sum_{s}d^{\pi}(s)\sum_{a}\frac{\partial\pi(s,a)}{\partial\theta} Q^{\pi}(s,a). \end{array} \tag{Q.E.D.}
$$
