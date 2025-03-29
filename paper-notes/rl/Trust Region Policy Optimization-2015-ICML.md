# Abstract 
We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm, called Trust Region Policy Optimization (TRPO). This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks. 
>  我们描述一个用于优化策略的迭代过程，该过程保证单调改进策略
>  通过对理论上正确的过程进行若干近似，我们提出一个实用算法，称为 TRPO，该算法类似于自然策略梯度方法，且对优化大规模非线性策略例如神经网络非常有效

Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input. Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters. 
>  试验证明了算法在各种任务中表现出 robust performance，包括学习模拟机器人游泳、跳跃、行走步态、使用屏幕图像作为输入玩 Atari 游戏
>  即便实践算法偏移了理论，但 TRPO 仍能实现单调改进，且超参数调优需求较少

# 1 Introduction 
Most algorithms for policy optimization can be classified into three broad categories: (1) policy iteration methods, which alternate between estimating the value function under the current policy and improving the policy (Bertsekas, 2005); (2) policy gradient methods, which use an estimator of the gradient of the expected return (total reward) obtained from sample trajectories (Peters & Schaal, 2008a) (and which, as we later discuss, have a close connection to policy iteration); and (3) derivative-free optimization methods, such as the cross-entropy method (CEM) and covariance matrix adaptation (CMA), which treat the return as a black box function to be optimized in terms of the policy parameters (Szita & Lorincz, 2006). 
>  多数策略优化算法可以分为三大类
>  (1) 策略迭代方法，交替进行当前策略下的价值函数估计和策略改进
>  (2) 策略梯度方法，利用从样本轨迹中获得的预期回报 (总奖励) 的梯度值 (优化策略)，策略梯度方法和策略迭代方法有密切联系
>  (3) 无导数优化方法，例如交叉熵方法和协方差矩阵适应法，它将回报视为一个黑盒函数，以策略参数的形式进行优化

General derivative-free stochastic optimization methods such as CEM and CMA are preferred on many problems, because they achieve good results while being simple to understand and implement. 
>  通用的无导数随机优化方法例如 CEM, CMA 因其效果和易于理解且实现在许多问题被使用

For example, while Tetris is a classic benchmark problem for approximate dynamic programming (ADP) methods, stochastic optimization methods are difficult to beat on this task (Gabillon et al., 2013). For continuous control problems, methods like CMA have been successful at learning control policies for challenging tasks like locomotion when provided with hand-engineered policy classes with low-dimensional parameterizations (Wampler & Popovic, 2009). 
>  例如在经典近似动态规划方法的基准问题——俄罗斯方块中，随机优化方法表现良好
>  对于连续控制问题，在提供了低维参数化的手工设计的策略类时，随机优化方法也可以实现学习复杂任务的控制策略，例如运动控制

The inability of ADP and gradient-based methods to consistently beat gradient-free random search is unsatisfying, since gradient-based optimization algorithms enjoy much better sample complexity guarantees than gradient-free methods (Nemirovski, 2005). Continuous gradient-based optimization has been very successful at learning function approximators for supervised learning tasks with huge numbers of parameters, and extending their success to reinforcement learning would allow for efficient training of complex and powerful policies. 
>  近似动态规划方法和基于梯度的方法无法优于无梯度的随机搜索方法 is unsatisfying，因为基于梯度的方法在样本复杂性保证上远优于无导数方法
>  连续的基于梯度的优化方法在处理具有大量参数的有监督学习任务时非常成功，将其拓展到 RL 领域将有助于高效训练复杂的策略

In this article, we first prove that minimizing a certain surrogate objective function guarantees policy improvement with non-trivial step sizes. Then we make a series of approximations to the theoretically-justified algorithm, yielding a practical algorithm, which we call trust region policy optimization (TRPO). We describe two variants of this algorithm: first, the single-path method, which can be applied in the model-free setting; second, the vine method, which requires the system to be restored to particular states, which is typically only possible in simulation. These algorithms are scalable and can optimize nonlinear policies with tens of thousands of parameters, which have previously posed a major challenge for model-free policy search (Deisenroth et al., 2013). 
>  本文中，我们证明最小化某个代理目标函数保证以非平凡的步长改进策略
>  我们继而对理论上成立的算法执行一系列近似，得到 TRPO 算法
>  我们描述该算法的两个变体，其一是 single-path 方法，可以在无模型情况下应用，其二是 vine 方法，它要求将系统恢复到特定状态，这通常只能在模拟环境中实现
>  算法可拓展，且可以优化具有数万个参数的非线性策略，而这是之前无模型策略搜索的重大挑战之一

In our experiments, we show that the same TRPO methods can learn complex policies for swimming, hopping, and walking, as well as playing Atari games directly from raw images. 

# 2 Preliminaries 
Consider an infinite-horizon discounted Markov decision process (MDP), defined by the tuple $(S,\mathcal{A},P,r,\rho_{0},\gamma)$ , where $\mathcal S$ is a finite set of states, $\mathcal{A}$ is a finite set of actions, $P:S\times{\mathcal{A}}\times{\mathcal{S}}\rightarrow\mathbb{R}$ is the transition probability distribution, $r:S\rightarrow\mathbb{R}$ is the reward function, $\rho_{0}:S\to\mathbb{R}$ is the distribution of the initial state $s_{0}$ , and $\gamma\in(0,1)$ is the discount factor. 
>  考虑一个无限期折扣 MDP，由 $(\mathcal S, \mathcal A, P, r, \rho_0, \gamma)$ 定义，状态集合和动作集合都是有限集，$P$ 为转移概率分布，$r$ 为奖励函数，奖励仅基于状态，$\rho_0$ 为初始状态 $s_0$ 的分布，$\gamma$ 为折扣因子

Let $\pi$ denote a stochastic policy $\pi:{\mathcal{S}}\times{\mathcal{A}}\to[0,1]$ , and let $\eta(\pi)$ denote its expected discounted reward: 

$$
\begin{align}
&\eta(\pi) = \mathbb E_{s_0, a_0, \dots}\left[\sum_{t=0}^\infty \gamma^t r(s_t)\right],\text{where}\\
&s_0\sim\rho_0(s_0), a_t\sim\pi(a_t\mid s_t), s_{t+1}\sim P(s_{t+1}\mid s_t, a_t)
\end{align}
$$

>  $\pi$ 表示某随机策略，$\eta(\pi)$ 表示从 $s_0\sim \rho_0(s_0)$ 开始，遵循 $\pi$ 的期望回报

We will use the following standard definitions of the state-action value function $Q_{\pi}$ , the value function $V_{\pi}$ , and the advantage function $A_{\pi}$ : 

$$
\begin{align}
&Q_\pi(s_t, a_t) = \mathbb E_{s_{t+1}, a_{t+1}, \dots}\left[\sum_{l=0}^\infty \gamma^l r(s_{t+l})\right]\\
&V_\pi(s_t) = \mathbb E_{a_t, s_{t+1}, \dots}\left[\sum_{l=0}^\infty \gamma^l r(s_{t+l})\right]\\
&A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s), \text{where}\\
&\quad a_t \sim \pi(a_t\mid s_t), s_{t+1}\sim P(s_{t+1}\mid s_t, a_t)\text{ for }t\ge 0  
\end{align}
$$

>  动作价值函数、状态价值函数、优势函数的标准定义如上
>  容易知道 $\eta(\pi) = \mathbb E_{s_0\sim \rho_0}[V_\pi(s_0)]$

The following useful identity expresses the expected return of another policy $\tilde{\pi}$ in terms of the advantage over $\pi$ , accumulated over timesteps (see Kakade & Langford (2002) or Appendix A for proof): 

$$
\eta(\tilde{\pi})=\eta(\pi)+\mathbb{E}_{s_{0},a_{0},\cdots\sim\tilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^{t}A_{\pi}(s_{t},a_{t})\right]\tag{1}
$$ 
where the notation $\mathbb{E}_{s_{0},a_{0},\cdots\sim\tilde{\pi}}\left[\ldots\right]$ indicates that actions are sampled $a_{t}\sim\tilde{\pi}(\cdot|s_{t})$ . 

>  我们可以用当前策略 $\pi$ 的优势函数 $A_\pi$ 和期望回报 $\eta(\pi)$ 表示另一个策略 $\tilde \pi$ 的期望回报 $\eta(\tilde \pi)$，形式如 Eq 1 所示
>  其中 $\mathbb E_{s_0, a_0, \dots \sim\tilde \pi}[...]$ 表示动作轨迹中的动作是从 $a_t\sim \tilde \pi(\cdot \mid s_t)$ 中采样得到

>  推导
>  记从 $s_0$ 开始的轨迹为 $\tau$

$$
\begin{align}
V_{\tilde \pi}(s_0) &= \mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty \gamma^tr(s_t)\right]\\
&=\mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty 
\gamma^t r(s_t)  - V_\pi(s_0) + V_\pi(s_0)\right]\\
&=\mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty \gamma^t r(s_t) - V_\pi(s_0)\right] + \mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[V_\pi(s_0)\right]\\
&=\mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty \gamma^t r(s_t) - V_\pi(s_0)\right] + V_\pi(s_0)\\
&=\mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty \gamma^t r(s_t) - \left(\sum_{t=0}^\infty \gamma^t V_\pi(s_t) - \sum_{t=1}^\infty \gamma^t V_\pi(s_t)\right)\right] + V_\pi(s_0)\\
&=\mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty \gamma^t r(s_t) + \left(\sum_{t=1}^\infty \gamma^t V_\pi(s_t) - \sum_{t=0}^\infty \gamma^t V_\pi(s_t)\right)\right] + V_\pi(s_0)\\
&=\mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty \gamma^t r(s_t) +\sum_{t=0}^\infty \gamma^t \left(\gamma V_\pi(s_{t+1}) - V_\pi(s_t)\right)\right] + V_\pi(s_0)\\
&=\mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty \gamma^t \left(r(s_t) + \gamma V_\pi(s_{t+1}) - V_\pi(s_t)\right)\right] + V_\pi(s_0)\\
&=V_\pi(s_0) + \mathbb E_{\tau\sim p_{\tilde \pi}(\tau)}\left[\sum_{t=0}^\infty \gamma^t A_\pi(s_t,a_t)\right] \\
&=V_\pi(s_0) + \mathbb E_{a_0,\dots,\sim\tilde \pi}\left[\sum_{t=0}^\infty \gamma^t A_\pi(s_t,a_t)\right] \\
\end{align}
$$

>  之后两边同时对 $s_0$ 求期望就得到了 Eq 1
>  在上面的推导中，虽然 $a_t$ 是根据 $\tilde \pi$ 采样得到，但是 $r(s_t) + \gamma V_\pi(s_{t+1}) - V_\pi(s_t)$ 直接替换为 $A_\pi(s_t, a_t)$ 也是没有问题的，$A_\pi(s_t, a_t)$ 是为给定的行动 $a_t$ 定义优势，并不在意该行动到底根据哪个策略采样

Let $\rho_{\pi}$ be the (unnormalized) discounted visitation frequencies 

$$
\rho_{\pi}(s)=P(s_{0}=s)+\gamma P(s_{1}=s)+\gamma^{2}P(s_{2}=s)+\ldots,
$$ 
where $s_{0}\sim\rho_{0}$ and the actions are chosen according to $\pi$ . 

>  我们令 $\rho_\pi(s)$ 表示 (未归一化的) 状态 $s$ (在轨迹中) 的访问频率/概率，其定义如上
>  其中初始状态 $s_0\sim \rho_0$，轨迹中的动作根据 $\pi$ 选取

We can rewrite Equation (1) with a sum over states instead of timesteps: 

$$
\begin{align}
\eta(\tilde \pi)&=\eta(\pi) + \sum_{t=0}^\infty \sum_s P(s_t=s\mid \tilde \pi)\sum_a \tilde \pi(a\mid s)\gamma^t A_\pi(s,a)\\
&=\eta(\pi) + \sum_{t=0}^\infty \sum_s \gamma^t P(s_t=s\mid \tilde \pi)\sum_a \tilde \pi(a\mid s) A_\pi(s,a)\\
&=\eta(\pi) + \sum_s \sum_{t=0}^\infty \gamma^t P(s_t = s\mid \tilde \pi)\sum_a \tilde \pi(a\mid s)A_\pi(s,a)\\
&=\eta(\pi) + \sum_s\rho_{\tilde \pi}(s)\sum_a\tilde \pi(a\mid s)A_\pi(s,a)\tag{2}
\end{align}
$$

>  我们进而重写 Eq 1，通过交换求和顺序，使其根据状态求和，而不是根据时间步求和

>  Eq 2 给出了新策略 $\tilde \pi$  (或者任意其他策略) 的目标函数和当前策略 $\pi$ 的目标函数的关系

This equation implies that any policy update $\pi\rightarrow\tilde{\pi}$ that has a nonnegative expected advantage at every state $s$ , i.e., $\begin{array}{r}{\sum_{a}\tilde{\pi}(a|s)A_{\pi}(s,a)\geq0}\end{array}$ , is guaranteed to increase the policy performance $\eta$ , or leave it constant in the case that the expected advantage is zero everywhere. 
>  Eq 2 表示了任意使策略 $\pi$ 变为 $\tilde \pi$ 的更新 $\pi \to \tilde \pi$ ，只要在每个状态 $s$ 上都有非负的期望优势，即 $\sum_a \tilde \pi(a\mid s) A_\pi(s, a)\ge0$ ($\pi$ 定义下的优势根据 $\tilde \pi$ 计算期望)，就保证会提高策略的表现 $\eta$
>  或者期望优势在所有状态都为零的情况下，就使 $\pi$ 保持不变

>  直观上看，$\tilde \pi$ 如果根据 $\pi$ 定义的优势函数 $A_\pi(s, a)$ 适当调整其动作分布，使得优势大的动作概率更高，使得优势为负的动作概率更低，就能使得期望非负，进而优化了原策略

This implies the classic result that the update performed by exact policy iteration, which uses the deterministic policy $\tilde{\pi}(s)=\arg\operatorname*{max}_{a}A_{\pi}(s,a)$ , improves the policy if there is at least one state-action pair with a positive advantage value and nonzero state visitation probability, otherwise the algorithm has converged to the optimal policy. 
>  这一结论实际上联系了经典的策略改进定理，即通过精确策略迭代 (使用确定性策略 $\tilde \pi(s) = \arg\max_a A_\pi(s, a)$ 作为下一个策略) 更新策略时，如果当前策略至少存在一个 state-action pair 具有非零的访问概率并且具有正优势值，则策略就会得到改进，否则算法已经收敛到最优策略 (收敛到最优策略后，所有的优势值应该都是 $\le 0$ 的，因为从状态 $s_t$ 出发一定能选择到最优的动作，故 $V_\pi(s_t) \ge Q_\pi(s_t, a_t)\ \  \forall a$)

However, in the approximate setting, it will typically be unavoidable, due to estimation and approximation error, that there will be some states $s$ for which the expected advantage is negative, that is, $\begin{array}{r}{\sum_{a}\tilde{\pi}(a|s)A_{\pi}(s,a)<0}\end{array}$ . 
>  然而，在近似设定下，由于估计和近似误差 (Monte Carlo 估计和价值函数近似的误差)，通常不可避免地会有一些状态，它在新策略 $\tilde \pi$ 下的期望优势为负，即 $\sum_a\tilde \pi(a\mid s) A_\pi(s, a)<0$

The complex dependency of $\rho_{\tilde{\pi}}(s)$ on $\tilde{\pi}$ makes Equation (2) difficult to optimize directly. Instead, we introduce the following local approximation to $\eta$ : 

$$
L_{\pi}(\tilde{\pi})=\eta(\pi)+\sum_{s}\rho_{\pi}(s)\sum_{a}\tilde{\pi}(a|s)A_{\pi}(s,a).\tag{3}
$$ 
Note that $L_{\pi}$ uses the visitation frequency $\rho_{\pi}$ rather than $\rho_{\tilde{\pi}}$ , ignoring changes in state visitation density due to changes in the policy. 

>  Eq 2 中，$\rho_{\tilde \pi}(s)$ 对 $\tilde \pi$ 的复杂依赖使得它难以直接优化
>  故我们引入对 $\eta$ 的局部近似，如 Eq 3 所示，也就是将新策略下的状态概率函数 $\rho_{\tilde \pi}(s)$ 直接替换为了旧策略下的状态概率函数 $\rho_\pi(s)$，即我们忽略了策略改变后，状态访问概率的变化

>  Eq 2 中，因为不可能基于新策略 $\tilde \pi$ 为状态进行采样，我们将 $\rho_{\tilde \pi}$ 替换为 $\rho_\pi$，也就是基于当前策略 $\pi$ 为状态进行采样，进行了近似

However, if we have a parameterized policy $\pi_{\theta}$ , where $\pi_{{\theta}}({a}|{s})$ is a differentiable function of the parameter vector $\theta$ , then $L_{\pi}$ matches $\eta$ to first order (see Kakade & Langford (2002)). That is, for any parameter value $\theta_{0}$ , 

$$
\begin{align}
L_{\pi_{\theta_0}} &= \eta(\pi_{\theta_0}),\\
\nabla_{\theta}L_{\pi_{\theta_0}}(\pi_{\theta})|_{\theta = \theta_0} &= \nabla_{\theta}\eta(\pi_\theta)|_{\theta = \theta_0}\tag{4}\\
\end{align}
$$

Equation (4) implies that a sufficiently small step $\pi_{{\theta}_{0}}\rightarrow\tilde{\pi}$ that improves $L_{\pi_{\theta_{old}}}$ will also improve $\eta$ , but does not give us any guidance on how big of a step to take. 

>  如果我们有一个参数化的策略 $\pi_\theta$，其中 $\pi_\theta(a\mid s)$ 是参数向量 $\theta$ 的可微函数，则 $L_\pi$ 在一阶上和 $\eta$ 匹配，即对于任意参数值 $\theta_0$ ，有 Eq 4 成立 ($L_\pi$ 和 $\eta$ 在 $\pi_{\theta_0}$ 点上相等，$L_\pi$ 和 $\eta$ 在 $\pi_{\theta_0}$ 点上的梯度相等)
>  Eq 4 表明，一个能够提高 $L_{\pi_{\theta_{old}}}$ 的足够小的 step $\pi_{\theta_0} \to \tilde \pi$ 也能够提高 $\eta$，但具体多小不清楚

>  $\eta$ 指代新策略 $\tilde \pi$ 的目标 $\eta(\tilde \pi)$，$L_\pi$ 指代我们对于该目标基于当前策略 $\pi$ 的近似 (Eq 3)
>  上面的讨论指出，虽然 Eq 3 的近似比较简单 (直接用 $\rho_\pi$ 替换了 $\rho_{\tilde \pi}$)，但近似结果 $L_\pi$ 仍和原目标 $\eta$ 保持一阶匹配的性质，因此，只要步长足够小，能够提高 $L_\pi$ 的参数更新也近似可以提高 $\eta$

To address this issue, Kakade & Langford (2002) proposed a policy updating scheme called conservative policy iteration, for which they could provide explicit lower bounds on the improvement of $\eta$ . 
>  为了确定步长的合适大小，Kakade & Langford 提出了称为保守策略迭代的策略更新方法，并且为该方法下，$\eta$ 的改进提供了明确的下界

To define the conservative policy iteration update, let $\pi_{\mathrm{old}}$ denote the current policy, and let $\pi^{\prime}=\arg\operatorname*{max}_{\pi^{\prime}}L_{\pi_{\mathrm{old}}}(\pi^{\prime})$ . The new policy $\pi_{\mathrm{new}}$ was defined to be the following mixture: 

$$
\pi_{\mathrm{new}}(a|s)=(1-\alpha)\pi_{\mathrm{old}}(a|s)+\alpha\pi^{\prime}(a|s).\tag{5}
$$ 
>  CPI 更新的定义如上，其中 $\pi_{old}$ 为当前策略，$\pi' = \arg\max_{\pi'} L_{\pi_{old}}(\pi')$ (最大化基于 $\pi_{old}$ 的近似目标的策略)，$\pi_{new}$ 为新策略，它是 $\pi_{old}, \pi'$ 的加权混合

Kakade and Langford derived the following lower bound: 

$$
{\eta({\pi_{\mathrm{new}}})\geq L_{\pi_{\mathrm{old}}}(\pi_{\mathrm{new}})-\frac{2\epsilon\gamma}{(1-\gamma)^{2}}\alpha^{2}}\ {\quad\mathrm{where}\ \epsilon=\operatorname*{max}_{s}\bigl|\mathbb{E}_{a\sim\pi^{\prime}(a|s)}\left[A_{\pi}(s,a)\right]\bigr|.}\tag{6}
$$ 
(We have modified it to make it slightly weaker but simpler.) 
Note, however, that so far this bound only applies to mixture policies generated by Equation (5). This policy class is unwieldy and restrictive in practice, and it is desirable for a practical policy update scheme to be applicable to all general stochastic policy classes. 

>  CPI 更新保证的下界 Eq 6 所示，该下界仅适用于 Eq 5 计算的混合策略，但该策略不是理想的

# 3 Monotonic Improvement Guarantee for General Stochastic Policies 
Equation (6), which applies to conservative policy iteration, implies that a policy update that improves the right-hand side is guaranteed to improve the true performance $\eta$ . 
>  Eq 6 表示了在 CPI 下，能够改进近似策略目标的更新是保证能够提高真实策略目标 $\eta$ 的

Our principal theoretical result is that the policy improvement bound in Equation (6) can be extended to general stochastic policies, rather than just mixture polices, by replacing $\alpha$ with a distance measure between $\pi$ and $\tilde{\pi}$ , and changing the constant $\epsilon$ appropriately. 
>  我们的主要理论结果是 Eq 6 中的策略改进界可以被拓展到随机策略，而不是混合策略
>  我们将 $\alpha$ 替换为了 $\pi, \tilde \pi$ 之间的距离度量，并且适当调整了常数 $\epsilon$

Since mixture policies are rarely used in practice, this result is crucial for extending the improvement guarantee to practical problems. The particular distance measure we use is the total variation divergence, which is defined by $\begin{array}{r}{D_{T V}(p\parallel q)=\frac{1}{2}\sum_{i}|p_{i}-q_{i}|}\end{array}$ for discrete probability distributions $p,q$ .
>  因为混合策略在实践中较少用，故我们的理论结果对于将该改进保证拓展到实际问题是十分重要的
>  我们使用的距离度量是全变异散度，对于离散分布 $p, q$，其定义为 $D_{TV}(p\|q) = \frac 1 2\sum_i |p_i - q_i|$

Define $D_{\mathrm{TV}}^{\mathrm{max}}(\pi,\tilde{\pi})$ as 

$$
D_{\mathrm{TV}}^{\mathrm{max}}(\pi,\tilde{\pi})=\operatorname*{max}_{s}D_{T V}(\pi(\cdot|s)\parallel\tilde{\pi}(\cdot|s)).\tag{7}
$$ 
>  我们将 $D_{TV}^{\max}(\pi, \tilde \pi)$ 定义为策略 $\pi, \tilde \pi$ 的 TV 散度相对于状态的最大值

**Theorem 1.** Let $\alpha=D_{\mathrm{TV}}^{\mathrm{max}}(\pi_{\mathrm{old}},\pi_{\mathrm{new}})$ . Then the following bound holds: 

$$
\begin{align}{\displaystyle{\eta(\pi_{\mathrm{new}})\geq L_{\pi_{\mathrm{old}}}(\pi_{\mathrm{new}})-\frac{4\epsilon\gamma}{(1-\gamma)^{2}}\alpha^{2}}}\\ {{\text{where}\ \epsilon=\operatorname*{max}_{s,a}|A_{\pi}(s,a)|}}\end{align}\tag{8}
$$

We provide two proofs in the appendix. The first proof extends Kakade and Langford’s result using the fact that the random variables from two distributions with total variation divergence less than $\alpha$ can be coupled, so that they are equal with probability $1-\alpha$ . The second proof uses perturbation theory. 

>  Theorem 1 给出了策略改进的界，它相对于 $\alpha = D_{TV}^{max}(\pi_{old}, \pi_{new})$ 定义
>  Appendix 提供了两种证明方法，第一个证明利用了两个 TV 散度小于 $\alpha$ 的分布的随机变量可以被耦合，故它们有 $1-\alpha$ 的概率相等
>  第二个证明利用了扰动理论

Next, we note the following relationship between the total variation divergence and the KL divergence (Pollard (2000), Ch. 3): $D_{T V}(p\parallel q)^{2}\leq D_{\mathrm{KL}}(p\parallel q)$ . 
>  TB 散度和 KL 散度的关系是 $D_{TV}(p\|q)^2 \le D_{KL}(p\|q)$

Let $\begin{array}{r}{D_{\mathrm{KL}}^{\mathrm{max}}(\pi,\tilde{\pi})=\operatorname*{max}_{s}D_{\mathrm{KL}}(\pi(\cdot|s)\parallel\tilde{\pi}(\cdot|s))}\end{array}$ . The following bound then follows directly from Theorem 1: 

$$
\begin{align}{\eta(\tilde{\pi})\geq L_{\pi}(\tilde{\pi})-C D_{\mathrm{KL}}^{\operatorname*{max}}(\pi,\tilde{\pi}),}\\ {\mathrm{where}\ C=\frac{4\epsilon\gamma}{(1-\gamma)^{2}}.}\end{align}\tag{9}
$$ 
>  令 $D_{KL}^{\max}(\pi, \tilde \pi) = \max_s D_{KL}(\pi(\cdot\mid s)\| \tilde \pi(\cdot\mid s))$，根据 Theorem 1，我们可以用 KL 散度表示界，如 Eq 9 所示

![[pics/TRPO-Algorithm1.png]]

Algorithm 1 describes an approximate policy iteration scheme based on the policy improvement bound in Equation (9). Note that for now, we assume exact evaluation of the advantage values $A_{\pi}$ . 
>  Algorithm 1 描述了基于 Eq 9 的策略提升界的近似策略迭代方法
>  注意目前我们尚且假设优势值 $A_\pi$ 是准确评估的

It follows from Equation (9) that Algorithm 1 is guaranteed to generate a monotonically improving sequence of policies $\eta(\pi_{0})\leq\eta(\pi_{1})\leq\eta(\pi_{2})\leq\ldots.$ To see this, let $M_{i}(\pi)=$ $L_{\pi_{i}}(\pi)-C D_{\mathrm{KL}}^{\operatorname*{max}}(\pi_{i},\pi)$ . Then 

$$
\begin{align}&{\eta(\pi_{i+1})\geq M_{i}(\pi_{i+1})\mathrm{~by~Equation~}(9)}\\ &{\eta(\pi_{i})=M_{i}(\pi_{i}),\mathrm{~therefore},}\\ &{\eta(\pi_{i+1})-\eta(\pi_{i})\geq M_{i}(\pi_{i+1})-M(\pi_{i}).}\tag{10}\end{align}
$$

>  根据 Eq 9 的策略提升界，可以知道 Algorithm 1 保证会生成单调提高的策略序列 $\eta(\pi_0)\le \eta(\pi_1)\le\eta(\pi_2)\le \dots$
>  令 $M_i(\pi)$ 为 Eq 9 的 RHS，根据 Eq 10，可以知道策略 $\pi_{i+1}, \pi_i$ 的目标 $\eta(\pi_{i+1}), \eta(\pi_i)$ 的差一定不小于二者的 $M_i$ 之差

Thus, by maximizing $M_{i}$ at each iteration, we guarantee that the true objective $\eta$ is non-decreasing. 
>  因此，Algorithm 1 通过在每次迭代最大化 $M_i$，保证了真实的目标 $\eta$ 是不减的

This algorithm is a type of minorization-maximization (MM) algorithm (Hunter & Lange, 2004), which is a class of methods that also includes expectation maximization. In the terminology of MM algorithms, $M_{i}$ is the surrogate function that minorizes $\eta$ with equality at $\pi_{i}$ . This algorithm is also reminiscent of proximal gradient methods and mirror descent. 
>  该算法是一类最小化-最大化算法，EM 算法也属于这类方法
>  在最小化-最大化算法中，$M_i$ 是一个代理函数，该函数总是不大于 $\eta$，并且在 $\pi_i$ 处与 $\eta$ 相等
>  该算法也类似于近端梯度下降法和镜像下降法

Trust region policy optimization, which we propose in the following section, is an approximation to Algorithm 1, which uses a constraint on the KL divergence rather than a penalty to robustly allow large updates. 
>  TRPO 算法是对 Algorithm 1 的近似，差异在于 TRPO 将 KL 散度用作约束，而不是作为惩罚项，以稳健地达到较大更新

# 4 Optimization of Parameterized Policies 
In the previous section, we considered the policy optimization problem independently of the parameterization of $\pi$ and under the assumption that the policy can be evaluated at all states. We now describe how to derive a practical algorithm from these theoretical foundations, under finite sample counts and arbitrary parameterizations. 
>  上一节中，我们独立于 $\pi$ 的参数化，基于可以在所有状态下评估策略的价值讨论了策略优化问题
>  我们现在描述如何根据这些理论基础推导出在有限样本和任意参数化下的实用算法

Since we consider parameterized policies $\pi_{{\theta}}({a}|{s})$ with parameter vector $\theta$ , we will overload our previous notation to use functions of $\theta$ rather than $\pi$ , e.g. $\eta(\theta):=\eta(\pi_{\theta})$ , $L_{\theta}(\tilde{\theta}):=L_{\pi_{\theta}}(\pi_{\tilde{\theta}})$ , and $D_{\mathrm{KL}}(\theta\parallel\tilde{\theta}):=D_{\mathrm{KL}}(\pi_{\theta}\parallel\pi_{\tilde{\theta}})$ . We will use $\theta_{\mathrm{old}}$ to denote the previous policy parameters that we want to improve upon. 
>  因为我们用参数向量 $\theta$ 参数化策略 $\pi_\theta(a\mid s)$，我们简化之前的符号，直接将各个关于策略 $\pi_\theta$ 的函数表示为关于参数 $\theta$ 的函数
>  我们用 $\theta_{old}$ 表示我们需要提升的旧策略参数

The preceding section showed that $\eta(\theta)~\geq~L_{\theta_{\mathrm{old}}}(\theta)~-$ $C D_{\mathrm{KL}}^{\mathrm{max}}(\theta_{\mathrm{old}},\theta)$ , with equality at $\theta=\theta_{\mathrm{old}}$ . Thus, by performing the following maximization, we are guaranteed to improve the true objective $\eta$ : 

$$
\begin{array}{r}{\underset{\theta}{\mathrm{maximize}}\left[L_{\theta_{\mathrm{old}}}(\theta)-C D_{\mathrm{KL}}^{\mathrm{max}}(\theta_{\mathrm{old}},\theta)\right].}\end{array}
$$ 
>  之前的部分证明了 $\eta(\theta) \ge L_{\theta_{old}}(\theta) - CD_{KL}^{\max}(\theta_{old}, \theta)$，其中等号在 $\theta = \theta_{old}$ 上取得
>  因此，我们只要以该下界为目标函数执行优化，就保证能够提高真实目标 $\eta$

In practice, if we used the penalty coefficient $C$ recommended by the theory above, the step sizes would be very small. 
>  但是在实践中，如果使用理论上的惩罚系数 $C$，更新步长会非常小

One way to take larger steps in a robust way is to use a constraint on the KL divergence between the new policy and the old policy, i.e., a trust region constraint: 

$$
\begin{align}
\operatorname*{maximize}_{\theta}L_{\theta_{\mathrm{old}}}(\theta)\\
\text{subject to}\ D_{KL}^{\max}(\theta_{old}, \theta)\le \delta
\end{align}\tag{11}
$$ 
>  一种以更稳健方式实现更大步长更新的方式是对新策略和旧策略之间的 KL 散度进行约束，即 Eq 11 的置信域约束

This problem imposes a constraint that the KL divergence is bounded at every point in the state space. While it is motivated by the theory, this problem is impractical to solve due to the large number of constraints. 
>  该问题为状态空间内的每一点都施加了 KL 散度的约束，虽然这一设定有理论依据，但因为约束的数量庞大，实际中难以求解 (不太可能遍历全部状态空间找最大)

Instead, we can use a heuristic approximation which considers the average KL divergence: 

$$
\begin{array}{r}{\overline{{D}}_{\mathrm{KL}}^{\rho}(\theta_{1},\theta_{2}):=\mathbb{E}_{s\sim\rho}\left[D_{\mathrm{KL}}(\pi_{\theta_{1}}(\cdot|s)\parallel\pi_{\theta_{2}}(\cdot|s))\right].}\end{array}
$$ 
>  我们使用启发式的近似，考虑平均 KL 散度

We therefore propose solving the following optimization problem to generate a policy update: 

$$
\begin{align}
&\operatorname*{maximize}_{\theta}L_{\theta_{\mathrm{old}}}(\theta)\\
&\text{subject to}\ \bar D_{KL}^{\rho_{\theta_{old}}}(\theta_{old}, \theta)\le \delta
\end{align}\tag{12}
$$

>  因此，我们通过求解以上优化问题来进行策略更新

Similar policy updates have been proposed in prior work (Bagnell & Schneider, 2003; Peters $\&$ Schaal, 2008b; Peters et al., 2010), and we compare our approach to prior methods in Section 7 and in the experiments in Section 8. Our experiments also show that this type of constrained update has similar empirical performance to the maximum KL divergence constraint in Equation (11). 
>  类似的策略更新方法已经在之前的工作中被提出
>  我们在第 7 节和第 8 节的实验部分将我们的方法与先前的方法进行了比较
>  我们的实验还表明，这种受约束的更新在经验性能上与方程 (11) 中的最大 KL 散度约束相似

# 5 Sample-Based Estimation of the Objective and Constraint 
The previous section proposed a constrained optimization problem on the policy parameters (Equation (12)), which optimizes an estimate of the expected total reward $\eta$ subject to a constraint on the change in the policy at each update. This section describes how the objective and constraint functions can be approximated using Monte Carlo simulation. 
>  上一节提出了一个关于策略参数的约束优化问题，它在每次更新时对策略的变化施加约束，优化期望总奖励 $\eta$ 的估计值
>  本节描述如何使用 Monte Carlo 近似该优化问题的目标和约束函数

We seek to solve the following optimization problem, obtained by expanding $L_{\theta_{\mathrm{old}}}$ in Equation (12): 

$$
\begin{align}{\underset{\theta}{\mathrm{maximize}}\displaystyle\sum_{s}\rho_{\theta_{\mathrm{old}}}(s)\sum_{a}\pi_{\theta}(a|s)A_{\theta_{\mathrm{old}}}(s,a)}\\{\mathrm{subject~to~}\overline{{D}}_{\mathrm{KL}}^{\rho_{\theta_{\mathrm{old}}}}\left(\theta_{\mathrm{old}},\theta\right)\leq\delta.}\end{align}\tag{13}
$$

>  我们首先将 Eq 12 中 $L_{\theta_{old}}$ 的期望 (相对于状态分布的期望) 展开，得到 Eq 13

We first replace $\begin{array}{r}{\sum_{s}\rho_{\theta_{\mathrm{old}}}(s)\left[\cdot\cdot\cdot\right]}\end{array}$ in the objective by the expectation $\frac{1}{1-\gamma}\mathbb{E}_{s\sim\rho_{\theta_{\mathrm{old}}}}\left[.~.~.\right]$ . Next, we replace the advantage values $A_{\theta_{\mathrm{old}}}$ by the $Q$ -values which only changes the objective by a constant. 
>  我们将目标中的求和替换为期望
>  然后用 $Q$ 值替换优势值 (这只会是目标函数值改变一个常数)

Last, we replace the sum over the actions by an importance sampling estimator. Using $q$ to denote the sampling distribution, the contribution of a single $s_{n}$ to the loss function is 

$$
\sum_{a}\pi_{{\theta}}(a|s_{n})A_{{\theta}_{\mathrm{old}}}(s_{n},a)=\mathbb{E}_{a\sim q}\left[\frac{\pi_{{\theta}}(a|s_{n})}{q(a|s_{n})}A_{{\theta}_{\mathrm{old}}}(s_{n},a)\right].
$$ 

>  最后将对动作的求和替换为重要性采样估计 (将 $\pi_\theta(a\mid s)$ 替换了，因为终究不可能基于更新后的参数 $\theta$ 采样)
>  用 $q$ 表示采样分布，单个状态 $s_n$ 对于目标函数的贡献如上

Our optimization problem in Equation (13) is exactly equivalent to the following one, written in terms of expectations: 

$$
\begin{align}
&\underset{\theta}{\operatorname*{maximize}}\mathbb{E}_{s\sim\rho_{\theta_{\mathrm{old}}},a\sim q}\left[\frac{\pi_{\theta}(a|s)}{q(a|s)}Q_{\theta_{\mathrm{old}}}(s,a)\right]\\
&\text{subject to}\ \mathbb E_{s\sim p_{\theta_{old}}}[D_{KL}(\pi_{\theta_{old}}(\cdot\mid s)\|\pi_{\theta}(\cdot\mid s))]\le \delta
\end{align}\tag{14}
$$

>  经过上述处理后 ($Q$ 值替换优势值，使用重要性采样)，可以得到 Eq 14
>  Eq 14 定义的优化问题和 Eq 13 是完全等价的，只不过是以期望形式表示

All that remains is to replace the expectations by sample averages and replace the $Q$ value by an empirical estimate. The following sections describe two different schemes for performing this estimation. 
>  之后，只需要用样本均值替换期望值，用经验估计替换 $Q$ 值即可
>  以下部分描述两种不同的执行此估计的方法

The first sampling scheme, which we call single path, is the one that is typically used for policy gradient estimation (Bartlett & Baxter, 2011), and is based on sampling individual trajectories. The second scheme, which we call vine, involves constructing a rollout set and then performing multiple actions from each state in the rollout set. This method has mostly been explored in the context of policy iteration methods (Lagoudakis & Parr, 2003; Gabillon et al., 2013). 
>  第一种是采样方法，我们称为单路径，它基于单独的轨迹样本，主要用于策略梯度估计
>  第二种方法，我们称为 vine，它涉及构造一个滚动集合，然后从滚动集合中的每个状态执行多个动作，主要用于策略迭代方法

## 5.1 Single Path 
In this estimation procedure, we collect a sequence of states by sampling $s_{0}~\sim~\rho_{0}$ and then simulating the policy $\pi_{\theta_{\mathrm{old}}}$ for some number of timesteps to generate a trajectory $s_{0},a_{0},s_{1},a_{1},\ldots,s_{T-1},a_{T-1},s_{T}$ . 
>  我们采样状态 $s_0\sim \rho_0$，然后执行策略 $\pi_{\theta_{old}}$ 一定时间步，生成轨迹

Hence, $q(a|s)=$ $\pi_{\theta_{\mathrm{old}}}(a|s)$ . $Q_{\theta_{\mathrm{old}}}(s,a)$ is computed at each state-action pair $(s_{t},a_{t})$ by taking the discounted sum of future rewards along the trajectory. 
>  故采样分布 $q(a\mid s)$ 就是 $\pi_{\theta_{old}}(a\mid s)$
>  价值函数 $Q_{\theta_{old}}(s, a)$ 基于观察到的回报计算

## 5.2 Vine 
In this estimation procedure, we first sample $s_{0}\sim\rho_{0}$ and simulate the policy $\pi_{\theta_{i}}$ to generate a number of trajectories. We then choose a subset of $N$ states along these trajectories, denoted $s_{1},s_{2},\ldots,s_{N}$ , which we call the “rollout set”. For each state $s_{n}$ in the rollout set, we sample $K$ actions according to $a_{n,k}\sim q(\cdot|s_{n})$ . Any choice of $q(\cdot|s_{n})$ with a support that includes the support of $\pi_{{\theta}_{i}}(\cdot|{s}_{n})$ will produce a consistent estimator. 
>  我们采样 $s_0 \sim \rho_0$，然后执行策略 $\pi_{\theta_i}$ 生成轨迹
>  然后我们从轨迹中选择一个包含 $N$ 个状态的子集，称为滚动集合，对于其中的每个状态 $s_n$，我们根据 $a_{n, k}\sim q(\cdot\mid s_n)$ 采样 $K$ 个动作
>  对于任意的采样分布 $q(\cdot\mid s_n)$，只要其支撑集包含了 $\pi_{\theta_i}(\cdot \mid s_n)$ 的支撑集 ($\pi_{\theta_i}(\cdot \mid s_n)$ 为零的地方，采样分布不能为零) 都可以产生一致的估计 (期望上看是相等的，即无偏差)

In practice, we found that $q(\cdot|s_{n})=\pi_{\theta_{i}}(\cdot|s_{n})$ works well on continuous problems, such as robotic locomotion, while the uniform distribution works well on discrete tasks, such as the Atari games, where it can sometimes achieve better exploration. 
>  实践中，我们发现在连续问题，例如机器人运动上，直接令 $q(\cdot \mid s_n) = \pi_{\theta_i}(\cdot \mid s_n)$ 的表现就很好，而在离散任务上，均匀分布则更好，因为它有时可以达到更好的探索

For each action $a_{n,k}$ sampled at each state $s_{n}$ , we estimate $\hat{Q}_{\theta_{i}}(s_{n},a_{n,k})$ by performing a rollout (i.e., a short trajectory) starting with state $s_{n}$ and action $a_{n,k}$ . We can greatly reduce the variance of the $Q$ -value differences between rollouts by using the same random number sequence for the noise in each of the $K$ rollouts, i.e., common random numbers. See (Bertsekas, 2005) for additional discussion on Monte Carlo estimation of $Q$ -values and ( Ng & Jordan, 2000) for a discussion of common random numbers in reinforcement learning. 
>  对于在每个状态 $s_n$ 下采样的动作 $a_{n, k}$，我们通过从状态 $s_n$ 和动作 $a_{n, k}$ 开始执行一个 rollout (即一段短轨迹) 来估计 $\hat Q_{\theta_i}(s_n, a_{n, k})$
>  通过在 $K$ 此 rollout 中使用相同的随机数序列生成噪声，我们可以大大降低不同 rollout 之间的 $Q$ -value 差异的方差，即我们采用了公共随机数方法

In small, finite action spaces, we can generate a rollout for every possible action from a given state. The contribution to $L_{\theta_{\mathrm{old}}}$ from a single state $s_{n}$ is as follows: 

$$
L_{n}(\theta)=\sum_{k=1}^{K}\pi_{\theta}(a_{k}|s_{n})\hat{Q}(s_{n},a_{k}),\tag{15}
$$

where the action space is $\mathcal{A}=\{a_{1},a_{2},\dotsb,a_{K}\}$ . 

>  在小的，有限的动作空间中，我们可以为每个给定的状态生成所有可能动作的 rollout，故此时单个状态 $s_n$ 对 $L_{\theta_{old}}$ 的贡献如 Eq 15 所示

In large or continuous state spaces, we can construct an estimator of the surrogate objective using importance sampling. The self-normalized estimator (Owen (2013), Chapter 9) of $L_{\theta_{\mathrm{old}}}$ obtained at a single state $s_{n}$ is 

$$
L_{n}(\theta)=\frac{\sum_{k=1}^{K}\frac{\pi_{\theta}(a_{n,k}|s_{n})}{\pi_{\theta_{\mathrm{old}}}(a_{n,k}|s_{n})}\hat{Q}(s_{n},a_{n,k})}{\sum_{k=1}^{K}\frac{\pi_{\theta}(a_{n,k}|s_{n})}{\pi_{\theta_{\mathrm{old}}}(a_{n,k}|s_{n})}},\tag{16}
$$

assuming that we performed $K$ actions $a_{n,1},a_{n,2},\ldots,a_{n,K}$ from state $s_{n}$ . This self-normalized estimator removes the need to use a baseline for the $Q$ -values (note that the gradient is unchanged by adding a constant to the $Q$ -values). Averaging over $s_{n}\sim\rho(\pi)$ , we obtain an estimator for $L_{\theta_{\mathrm{old}}}$ , as well as its gradient. 

>  在大状态空间或连续的状态空间中，我们可以用重要性采样构造代理目标的估计
>  单个 $s_n$ 对 $L_{\theta_{old}}$ 的贡献的自归一化的估计量如 Eq 16 所示
>  自归一化估计量消除了对 $Q$ -value 使用 baseline 的需求
>  对 Eq 16 在 $s_n \sim \rho(\pi)$ 上取平均，就得到了对 $L_{\theta_{old}}$ 的估计

The vine and single path methods are illustrated in Figure 1. We use the term vine, since the trajectories used for sampling can be likened to the stems of vines, which branch at various points (the rollout set) into several short offshoots (the rollout trajectories). 

![[pics/TRPO-Fig1.png]]

The benefit of the vine method over the single path method that is our local estimate of the objective has much lower variance given the same number of $Q$ -value samples in the surrogate objective. That is, the vine method gives much better estimates of the advantage values. The downside of the vine method is that we must perform far more calls to the simulator for each of these advantage estimates. Furthermore, the vine method requires us to generate multiple trajectories from each state in the rollout set, which limits this algorithm to settings where the system can be reset to an arbitrary state. In contrast, the single path algorithm requires no state resets and can be directly implemented on a physical system (Peters & Schaal, 2008b). 
>  vine 方法相较于 single path 方法的优势在于，在相同数量的 $Q$ -value 样本下，vine 方法对于目标函数的估计具有更小的方差，因此，vine 方法可以提供更优的优势值估计
>  vine 方法的劣势在于需要为每个优势值执行更多的采样，并且要求为 rollout set 中的每个状态生成多条轨迹，这限制了该算法只能应用在可以将系统重置回任意状态的场景，single path 则没有这方面限制，并且可以直接在物理系统上实现

# 6 Practical Algorithm 
Here we present two practical policy optimization algorithm based on the ideas above, which use either the single path or vine sampling scheme from the preceding section. 
>  我们展示基于上述思想的两个使用策略优化算法，其中一个使用 single path 采样方法，另一个使用 vine 采样方法

The algorithms repeatedly perform the following steps: 

1. Use the single path or vine procedures to collect a set of state-action pairs along with Monte Carlo estimates of their $Q$ -values. 
2. By averaging over samples, construct the estimated objective and constraint in Equation (14). 
3. Approximately solve this constrained optimization problem to update the policy’s parameter vector $\theta$ . We use the conjugate gradient algorithm followed by a line search, which is altogether only slightly more expensive than computing the gradient itself. See Appendix C for details. 

>  算法反复执行
>  1. 使用 single path 或 vine 收集一组状态-动作对和其 Q 值的 MC 估计
>  2. 基于样本平均值，构建 Eq 14 中的目标函数和约束
>  3. 近似求解该约束优化问题，更新策略参数 $\theta$，我们用共轭梯度算法并结合线性搜索，总体上比计算梯度本身更昂贵

With regard to (3), we construct the Fisher information matrix (FIM) by analytically computing the Hessian of the $\mathrm{KL}$ divergence, rather than using the covariance matrix of gradients. 
>  关于第三步的求解，我们通过解析计算 KL 散度 Hessian 来构造 FIM，而不是使用梯度的协方差矩阵

That is, we estimate $A_{ji}$ as $\begin{array}{r}{\frac{1}{N}\sum_{n=1}^{N}\frac{\partial^{2}}{\partial\theta_{i}\partial\theta_{j}}D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}(\cdot|s_{n})\parallel\pi_{\theta}(\cdot|s_{n}))}\end{array}$ , rather than $\begin{array}{r}{\frac{1}{N}\sum_{n=1}^{N}\frac{\partial}{\partial\theta_{i}}\log\pi_{\theta}(a_{n}|s_{n})\frac{\partial}{\partial\theta_{j}}\log\pi_{\theta}(a_{n}|s_{n})}\end{array}$ . The analytic estimator integrates over the action at each state $s_{n}$ and does not depend on the action $a_{n}$ that was sampled. As described in Appendix C, this analytic estimator has computational benefits in the large-scale setting, since it removes the need to store a dense Hessian or all policy gradients from a batch of trajectories. The rate of improvement in the policy is similar to the empirical FIM, as shown in the experiments. 

Let us briefly summarize the relationship between the theory from Section 3 and the practical algorithm we have described: 

- The theory justifies optimizing a surrogate objective with a penalty on KL divergence. However, the large penalty coefficient $C$ leads to prohibitively small steps, so we would like to decrease this coefficient. Empirically, it is hard to robustly choose the penalty coefficient, so we use a hard constraint instead of a penalty, with parameter $\delta$ (the bound on KL divergence). 
- The constraint on $D_{\mathrm{KL}}^{\mathrm{max}}(\theta_{\mathrm{old}},\theta)$ is hard for numerical optimization and estimation, so instead we constrain ${\overline{{D}}}_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)$ . 

>  让我们简要总结一下第3节中的理论与我们描述的实用算法之间的关系：
>  - 理论支持了一个带有 KL 散度惩罚的代理目标。然而，较大的惩罚系数 $C$ 会导致步长过小，因此我们希望减小这个系数。在实践中，很难稳健地选择惩罚系数，所以我们用一个硬约束代替惩罚，使用参数 $\delta$ 作为 KL 散度的上限。
>  - 对 $D_{\mathrm{KL}}^{\mathrm{max}}(\theta_{\mathrm{old}},\theta)$ 的约束在数值优化和估计中较为困难，因此我们转而约束 ${\overline{{D}}}_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)$。

Our theory ignores estimation error for the advantage function. Kakade & Langford (2002) consider this error in their derivation, and the same arguments would hold in the setting of this paper, but we omit them for simplicity. 
>  我们的理论忽略了优势函数的估计误差。Kakade & Langford (2002) 在其推导中考虑了这一误差，同样的论证也适用于本文的研究场景，但为了简化起见，我们省略了这部分内容。

# 7 Connections with Prior Work 
As mentioned in Section 4, our derivation results in a policy update that is related to several prior methods, providing a unifying perspective on a number of policy update schemes. The natural policy gradient (Kakade, 2002) can be obtained as a special case of the update in Equation (12) by using a linear approximation to $L$ and a quadratic approximation to the ${\overline{{D}}}_{\mathrm{KL}}$ constraint, resulting in the following problem: 

$$
\begin{align}
&\operatorname{maximize}\left[\nabla_{{\theta}}L_{{\theta}_{\mathrm{old}}}(\boldsymbol{\theta})\big\vert_{{\theta=\theta}_{\mathrm{old}}}\cdot({\theta-\theta}_{\mathrm{old}})\right]\\
&\text{subject to}\ \frac 1 2(\theta_{old} - \theta)^TA(\theta_{old} - \theta)\le \delta\\
&\text{where}\ A(\theta_{old})_{ij} =\\
&\frac{\partial}{\partial\theta_{i}}\frac{\partial}{\partial\theta_{j}}\mathbb{E}_{s\sim\rho_{\pi}}\left[D_{\mathrm{KL}}(\pi(\cdot|s,\theta_{\mathrm{old}})\parallel\pi(\cdot|s,\theta))\right]\big|_{\theta=\theta_{\mathrm{old}}}.
\end{align}
$$

The update is $\theta_{new} = \theta_{old} + \frac 1 \lambda A(\theta_{old})^{-1}\nabla_{\theta}L(\theta)|_{\theta=\theta_{old}}$ , where the step-size $\textstyle{\frac{1}{\lambda}}$ is typically treated as an algorithm parameter. This differs from our approach, which enforces the constraint at each update. Though this difference might seem subtle, our experiments demonstrate that it significantly improves the algorithm’s performance on larger problems. 

We can also obtain the standard policy gradient update by using an $\ell_{2}$ constraint or penalty: 

$$
\begin{align}&{\underset{\theta}{\mathrm{maximize}}\left[\nabla_{\theta}L_{\theta_{\mathrm{old}}}(\theta)\big|_{\theta=\theta_{\mathrm{old}}}\cdot(\theta-\theta_{\mathrm{old}})\right]}\\ &{\text{subject to}\ \frac{1}{2}\|\theta-\theta_{\mathrm{old}}\|^{2}\leq\delta.}\end{align}\tag{18}
$$

The policy iteration update can also be obtained by solving the unconstrained problem $\text{maximize}_\pi\  L_{\pi_{\mathrm{old}}}(\pi)$ , using $L$ as defined in Equation (3). 
>  直接求解 Eq 3 中定义的无约束优化问题 $\text{maximize}_\pi L_{\pi_{old}}(\pi)$ 等价于策略迭代更新

Several other methods employ an update similar to Equation (12). Relative entropy policy search (REPS) (Peters et al., 2010) constrains the state-action marginals $p(s,a)$ , while TRPO constrains the conditionals $p(a|s)$ . Unlike REPS, our approach does not require a costly nonlinear optimization in the inner loop. Levine and Abbeel (2014) also use a KL divergence constraint, but its purpose is to encourage the policy not to stray from regions where the estimated dynamics model is valid, while we do not attempt to estimate the system dynamics explicitly. Pirotta et al. (2013) also build on and generalize Kakade and Langford’s results, and they derive different algorithms from the ones here. 

# 8 Experiments 
We designed our experiments to investigate the following questions: 
1. What are the performance characteristics of the single path and vine sampling procedures? 
2. TRPO is related to prior methods (e.g. natural policy gradient) but makes several changes, most notably by using a fixed KL divergence rather than a fixed penalty coefficient. How does this affect the performance of the algorithm? 
3. Can TRPO be used to solve challenging large-scale problems? How does TRPO compare with other methods when applied to large-scale problems, with regard to final performance, computation time, and sample complexity? 

To answer (1) and (2), we compare the performance of the single path and vine variants of TRPO, several ablated variants, and a number of prior policy optimization algorithms. With regard to (3), we show that both the single path and vine algorithm can obtain high-quality locomotion controllers from scratch, which is considered to be a hard problem. We also show that these algorithms produce competitive results when learning policies for playing Atari games from images using convolutional neural networks with tens of thousands of parameters. 

## 8.1 Simulated Robotic Locomotion 
We conducted the robotic locomotion experiments using the MuJoCo simulator (Todorov et al., 2012). The three simulated robots are shown in Figure 2. The states of the robots are their generalized positions and velocities, and the controls are joint torques. Underactuation, high dimensionality, and non-smooth dynamics due to contacts make these tasks very challenging. The following models are included in our evaluation: 

1. Swimmer. 10-dimensional state space, linear reward for forward progress and a quadratic penalty on joint effort to produce the reward $\bar{r}(x,u)=\bar{v}_{x}-\mathrm{i}0^{-5}\|\bar{u}\|^{2}$ . The swimmer can propel itself forward by making an undulating motion. 
2. Hopper. 12-dimensional state space, same reward as the swimmer, with a bonus of $+1$ for being in a nonterminal state. We ended the episodes when the hopper fell over, which was defined by thresholds on the torso height and angle. 
3. Walker. 18-dimensional state space. For the walker, we added a penalty for strong impacts of the feet against the ground to encourage a smooth walk rather than a hopping gait. 

We used $\delta=0.01$ for all experiments. See Table 2 in the Appendix for more details on the experimental setup and parameters used. We used neural networks to represent the policy, with the architecture shown in Figure 3, and further details provided in Appendix D. To establish a standard baseline, we also included the classic cart-pole balancing problem, based on the formulation from Barto et al. (1983), using a linear policy with six parameters that is easy to optimize with derivative-free black-box optimization methods. 

The following algorithms were considered in the comparison: single path TRPO; vine TRPO; cross-entropy method (CEM), a gradient-free method (Szita & Lorincz, 2006); covariance matrix adaption (CMA), another gradient-free method (Hansen & Ostermeier, 1996); natural gradient, the classic natural policy gradient algorithm (Kakade, 2002), which differs from single path by the use of a fixed penalty coefficient (Lagrange multiplier) instead of the KL divergence constraint; empirical FIM, identical to single path, except that the FIM is estimated using the covariance matrix of the gradients rather than the analytic estimate; max KL, which was only tractable on the cart-pole problem, and uses the maximum KL divergence in Equation (11), rather than the average divergence, allowing us to evaluate the quality of this approximation. The parameters used in the experiments are provided in Appendix E. For the natural gradient method, we swept through the possible values of the stepsize in factors of three, and took the best value according to the final performance. 

Learning curves showing the total reward averaged across five runs of each algorithm are shown in Figure 4. Single path and vine TRPO solved all of the problems, yielding the best solutions. Natural gradient performed well on the two easier problems, but was unable to generate hopping and walking gaits that made forward progress. These results provide empirical evidence that constraining the KL divergence is a more robust way to choose step sizes and make fast, consistent progress, compared to using a fixed penalty. CEM and CMA are derivative-free algorithms, hence their sample complexity scales unfavorably with the number of parameters, and they performed poorly on the larger problems. The max $K L$ method learned somewhat more slowly than our final method, due to the more restrictive form of the constraint, but overall the result suggests that the average KL divergence constraint has a similar effect as the theatrically justified maximum KL divergence. Videos of the policies learned by TRPO may be viewed on the project website: http://sites.google.com/site/trpopaper/. 


![](https://cdn-mineru.openxlab.org.cn/extract/12df22f3-a5d8-45c1-9ba8-9013aa9dd3ec/c638110ef37093877307acf5506a8735a3e22ef2b1a93cc87ad012d6d364f67d.jpg) 
Figure 4. Learning curves for locomotion tasks, averaged across five runs of each algorithm with random initializations. Note that for the hopper and walker, a score of $^{-1}$ is achievable without any forward velocity, indicating a policy that simply learned balanced standing, but not walking. 
Note that TRPO learned all of the gaits with general purpose policies and simple reward functions, using minimal prior knowledge. This is in contrast with most prior methods for learning locomotion, which typically rely on hand-architected policy classes that explicitly encode notions of balance and stepping (Tedrake et al., 2004; Geng et al., 2006; Wampler & Popovic, 2009). 

## 8.2 Playing Games from Images 
To evaluate TRPO on a partially observed task with complex observations, we trained policies for playing Atari games, using raw images as input. The games require learning a variety of behaviors, such as dodging bullets and hitting balls with paddles. Aside from the high dimensionality, challenging elements of these games include delayed rewards (no immediate penalty is incurred when a life is lost in Breakout or Space Invaders); complex sequences of behavior ( $N^{*}$ bert requires a character to hop on 21 different platforms); and non-stationary image statistics (Enduro involves a changing and flickering background). 

We tested our algorithms on the same seven games reported on in (Mnih et al., 2013) and (Guo et al., 2014), which are made available through the Arcade Learning Environment (Bellemare et al., 2013) The images were preprocessed following the protocol in Mnih et al (2013), and the policy was represented by the convolutional neural network shown in Figure 3, with two convolutional layers with 16 channels and stride 2, followed by one fully-connected layer with 20 units, yielding 33,500 parameters. 

Table 1. Performance comparison for vision-based RL algorithms on the Atari domain. Our algorithms (bottom rows) were run once on each task, with the same architecture and parameters. Performance varies substantially from run to run (with different random initializations of the policy), but we could not obtain error statistics due to time constraints. 

<html><body><table><tr><td></td><td>B.Rider</td><td>Breakout</td><td>Enduro</td><td>Pong</td><td></td><td>Seaquest</td><td>S.Invaders</td></tr><tr><td>Random</td><td>354</td><td>1.2</td><td>0</td><td>-20.4</td><td>157</td><td>110</td><td>179</td></tr><tr><td>Human (Mnih et al., 2013)</td><td>7456</td><td>31.0</td><td>368</td><td>-3.0</td><td>18900</td><td>28010</td><td>3690</td></tr><tr><td>Deep Q Learning (Mnih et al., 2013)</td><td>4092</td><td>168.0</td><td>470</td><td>20.0</td><td>1952</td><td>1705</td><td>581</td></tr><tr><td>UCC-I (Guo et al., 2014)</td><td>5702</td><td>380</td><td>741</td><td>21</td><td>20025</td><td>2995</td><td>692</td></tr><tr><td>TRPO - single path</td><td>1425.2</td><td>10.8</td><td>534.6</td><td>20.9</td><td>1973.5</td><td>1908.6</td><td>568.4</td></tr><tr><td>TRPO-vine</td><td>859.5</td><td>34.2</td><td>430.8</td><td>20.9</td><td>7732.5</td><td>788.4</td><td>450.2</td></tr></table></body></html> 

The results of the vine and single path algorithms are summarized in Table 1, which also includes an expert human performance and two recent methods: deep $Q$ -learning (Mnih et al., 2013), and a combination of Monte-Carlo Tree Search with supervised training (Guo et al., 2014), called UCC-I. The 500 iterations of our algorithm took about 30 hours (with slight variation between games) on a 16-core computer. While our method only outperformed the prior methods on some of the games, it consistently achieved reasonable scores. Unlike the prior methods, our approach was not designed specifically for this task. The ability to apply the same policy search method to methods as diverse as robotic locomotion and image-based game playing demonstrates the generality of TRPO. 

the game-playing domain, we learned convolutional neural network policies that used raw images as inputs. This requires optimizing extremely high-dimensional policies, and only two prior methods report successful results on this task. 

Since the method we proposed is scalable and has strong theoretical foundations, we hope that it will serve as a jumping-off point for future work on training large, rich function approximators for a range of challenging problems. At the intersection of the two experimental domains we explored, there is the possibility of learning robotic control policies that use vision and raw sensory data as input, providing a unified scheme for training robotic controllers that perform both perception and control. The use of more sophisticated policies, including recurrent policies with hidden state, could further make it possible to roll state estimation and control into the same policy in the partially-observed setting. By combining our method with model learning, it would also be possible to substantially reduce its sample complexity, making it applicable to real-world settings where samples are expensive. 

# 9 Discussion 
We proposed and analyzed trust region methods for optimizing stochastic control policies. We proved monotonic improvement for an algorithm that repeatedly optimizes a local approximation to the expected return of the policy with a KL divergence penalty, and we showed that an approximation to this method that incorporates a KL divergence constraint achieves good empirical results on a range of challenging policy learning tasks, outperforming prior methods. 
>  我们提出并分析了优化随机控制策略的置信域方法
>  我们证明了一种算法的单调改进特性，该算法通过 KL 散度惩罚反复优化策略预期回报的局部近似值
>  我们展示了该算法的近似算法，它将 KL 散度惩罚改为 KL 散度约束，在一系列具有挑战性的任务中取得了良好结果，优于之前方法

Our analysis also provides a perspective that unifies policy gradient and policy iteration methods, and shows them to be special limiting cases of an algorithm that optimizes a certain objective subject to a trust region constraint. 

In the domain of robotic locomotion, we successfully learned controllers for swimming, walking and hopping in a physics simulator, using general purpose neural networks and minimally informative rewards. To our knowledge, no prior work has learned controllers from scratch for all of these tasks, using a generic policy search method and non-engineered, general-purpose policy representations. 

In the game-playing domain, we learned convolutional neural network policies that used raw images as inputs. This requires optimizing extremely high-dimensional policies, and only two prior methods report successful results on this task. Since the method we proposed is scalable and has strong theoretical foundations, we hope that it will serve as a jumping-off point for future work on training large, rich function approximators for a range of challenging problems. 

At the intersection of the two experimental domains we explored, there is the possibility of learning robotic control policies that use vision and raw sensory data as input, providing a unified scheme for training robotic controllers that perform both perception and control. 

The use of more sophisticated policies, including recurrent policies with hidden state, could further make it possible to roll state estimation and control into the same policy in the partially-observed setting. By combining our method with model learning, it would also be possible to substantially reduce its sample complexity, making it applicable to real-world settings where samples are expensive.

