# Abstract 
We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a “surrogate” objective function using stochastic gradient ascent. 
>  强化学习方法交替执行与环境交互采样数据和使用随机梯度上升优化替代目标函数

Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). 
>  标准策略梯度方法为每个数据样本执行一次梯度上升，我们提出支持多轮 minibatch 更新的目标函数，该方法称为近端策略优化 PPO
>  PPO 在某些方面具备置信域策略优化 TRPO 的特点，但更易于实现，更通用，且试验证明具有更好的样本复杂性

Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time. 
>  我们在一系列基准任务中测试了 PPO，包括模拟机器人运动、Atari 游戏
>  PPO 的表现优于其他在线策略梯度方法，总体上在样本复杂度、简洁性和运行时间之间取得了平衡

# 1 Introduction 
In recent years, several different approaches have been proposed for reinforcement learning with neural network function approximators. The leading contenders are deep $Q$ -learning [Mni+15], “vanilla” policy gradient methods [Mni + 16], and trust region / natural policy gradient methods [Sch+15b]. However, there is room for improvement in developing a method that is scalable (to large models and parallel implementations), data efficient, and robust (i.e., successful on a variety of problems without hyperparameter tuning). $Q$ -learning (with function approximation) fails on many simple problems $^{1}$ and is poorly understood, vanilla policy gradient methods have poor data efficiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks). 
>  近年来，DRL 领域的方法包括 deep Q-Learning、标准的策略梯度方法、置信域策略梯度方法
>  目前仍需要开发一个可拓展 (拓展到大规模模型和并行实现)、数据高效且 robust (在无需调节超参数的情况下在多种问题上取得成功) 的算法
>  deep Q-learning 在许多简单的问题上 (主要是连续控制问题) 失败，标准策略梯度方法数据效率较低且健壮性差 (因为是 on-policy)，置信域策略优化相对复杂，并且不兼容包含噪声 (如 dropout) 或参数共享 (策略函数和价值函数之间，或与辅助任务之间)

This paper seeks to improve the current state of affairs by introducing an algorithm that attains the data efficiency and reliable performance of TRPO, while using only first-order optimization. We propose a novel objective with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy. To optimize policies, we alternate between sampling data from the policy and performing several epochs of optimization on the sampled data. 
>  本文旨在引入一种算法来改善现状，该算法在仅使用一阶优化的情况下，达到了和 TRPO 相当的数据效率和可靠性能
>  我们提出了一种具有 clipped probability ratios 的新目标函数，该函数构成了对策略性能的悲观估计 (即下界)
>  优化策略时，我们在从策略中采样数据和基于采样数据执行多轮优化之间迭代

Our experiments compare the performance of various different versions of the surrogate objective, and find that the version with the clipped probability ratios performs best. We also compare PPO to several previous algorithms from the literature. On continuous control tasks, it performs better than the algorithms we compare against. On Atari, it performs significantly better (in terms of sample complexity) than A2C and similarly to ACER though it is much simpler. 
>  我们的试验比较了不同版本的代理目标函数的表现，发现了使用 clipped probability ratios 的版本表现最好
>  我们比较了 PPO 和现有算法，在连续控制任务中，PPO 的表现优于我们所对比的算法，在 Atari 游戏中，PPO 在样本复杂度方面显著优于 A2C，并且与 ACER 表现相当，并且它要简单得多

# 2 Background: Policy Optimization 
## 2.1 Policy Gradient Methods 
Policy gradient methods work by computing an estimator of the policy gradient and plugging it into a stochastic gradient ascent algorithm. 
>  策略梯度方法计算策略梯度的估计值，然后将估计值用于随机梯度上升以优化策略函数

The most commonly used gradient estimator has the form 

$$
\hat{g}=\hat{\mathbb{E}}_{t}\Big[\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid s_{t})\hat{A}_{t}\Big]\tag{1}
$$ 
where $\pi_{\theta}$ is a stochastic policy and $\hat{A}_{t}$ is an estimator of the advantage function at timestep $t$ . 

>  最常用的策略梯度估计的形式如上，其中 $\pi_{\theta}$ 是随机策略 (策略函数)，$\hat A_t$ 是时间步 $t$ 时的优势函数估计值

Here, the expectation $\hat{\mathbb{E}}_{t}[...]$ indicates the empirical average over a finite batch of samples, in an algorithm that alternates between sampling and optimization. 
>  期望 $\hat {\mathbb E}_t[...]$ 表示策略梯度的估计值是有限样本批量上的经验平均

Implementations that use automatic differentiation software work by constructing an objective function whose gradient is the policy gradient estimator; the estimator $\hat{g}$ is obtained by differentiating the objective 

$$
L^{P G}(\theta)=\mathbb{\hat{E}}_{t}\Big[\log\pi_{\theta}(a_{t}\mid s_{t})\hat{A}_{t}\Big].\tag{2}
$$ 
>  自动微分软件会基于目标函数自动计算上述形式的策略梯度估计值
>  策略梯度估计值 $\hat g$ 对应的目标函数形式如上
>  (可以看到，策略梯度中的 $\hat {\mathbb E}_t$ 实际上来自于目标函数中的 $\hat {\mathbb E}_t$，结合下文可以发现这里的 $\hat {\mathbb E}_t$ 应该是指对 minibatch 中，来自不同轨迹的各个时间步的策略梯度值的平均，就是典型的随机小批量梯度下降，minibatch 中的数据应该是不相关的)

While it is appealing to perform multiple steps of optimization on this loss $L^{P G}$ using the same trajectory, doing so is not well-justified, and empirically it often leads to destructively large policy updates (see Section 6.1; results are not shown but were similar or worse than the “no clipping or penalty” setting). 
>  虽然使用相同的轨迹对损失 $L^{PG}(\theta)$ 优化很有吸引力，但这在理论上缺乏充分依据，并且经验结果显示它通常会导致破坏性的大幅策略更新

## 2.2 Trust Region Methods 
In TRPO [Sch+15b], an objective function (the “surrogate” objective) is maximized subject to a constraint on the size of the policy update. Specifically, 

$$
\begin{align}
&\text{maximize}_{\theta} \quad\hat{\mathbb E}_t\left[\frac {\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat A_t\right]\tag{3}\\
&\text{subject to}\quad\hat {\mathbb E}_t[\mathrm{KL}[\pi_{\theta_{old}}(\cdot\mid s_t), \pi_{\theta}(\cdot\mid s_t)]] \le \delta\tag{4}
\end{align}
$$

Here, $\theta_{\mathrm{old}}$ is the vector of policy parameters before the update. This problem can efficiently be approximately solved using the conjugate gradient algorithm, after making a linear approximation to the objective and a quadratic approximation to the constraint. 

>  TRPO 中最大化目标函数时，存在一个约束限制了策略的更新幅度
>  其中，$\theta_{old}$ 是更新前的策略参数，通过对目标函数进行线性近似并且对约束进行二次近似后，可以使用共轭梯度算法高效地近似求解此问题

The theory justifying TRPO actually suggests using a penalty instead of a constraint, i.e., solving the unconstrained optimization problem 

$$
\underset{\theta}{\operatorname*{maximize}}\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}(a_{t}\mid s_{t})}{\pi_{\theta_{\mathrm{old}}}(a_{t}\mid s_{t})}\hat{A}_{t}-\beta\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_{t}),\pi_{\theta}(\cdot\mid s_{t})]\right]\tag{5}
$$ 
for some coefficient $\beta$ . 

>  支持 TRPO 的理论实际上建议使用惩罚项而不是使用约束，即求解 Eq 5 的无约束优化问题

This follows from the fact that a certain surrogate objective (which computes the max KL over states instead of the mean) forms a lower bound (i.e., a pessimistic bound) on the performance of the policy $\pi$ . 
>  这是由于形式如上的替代目标 (它计算状态上的最大 KL 散度，而不是平均值) 对策略 $\pi$ 的性能形成了一个下界

TRPO uses a hard constraint rather than a penalty because it is hard to choose a single value of $\beta$ that performs well across different problems—or even within a single problem, where the the characteristics change over the course of learning. 
>  TRPO 使用硬约束而不是惩罚项，因为很难为不同的问题 (甚至在一个问题内，其特性在学习过程中发生变化时) 选择一个单一的 $\beta$ 值使其都具有好的表现

Hence, to achieve our goal of a first-order algorithm that emulates the monotonic improvement of TRPO, experiments show that it is not sufficient to simply choose a fixed penalty coefficient $\beta$ and optimize the penalized objective Equation (5) with SGD; additional modifications are required. 
>  为此，为了在实践中，使用一阶优化算法模拟 TRPO 理论上的单调改进性质，我们经过试验，发现仅仅选择一个固定的惩罚系数 $\beta$ 并使用 SGA 优化目标函数 (5) 是不够的，需要额外的对 $\beta$ 的修改
>  (也就是理论确保了 Eq 5 是可以单调改进的，但实践中需要在优化过程中修改 $\beta$，否则无法达到理论上的特性)

# 3 Clipped Surrogate Objective 
Let $r_{t}(\theta)$ denote the probability ratio $\begin{array}{r}{r_{t}(\theta)=\frac{\pi_{\theta}(a_{t}\mid s_{t})}{\pi_{\theta_{\mathrm{old}}}(a_{t}\mid s_{t})}}\end{array}$ , so $r(\theta_{\mathrm{old}})=1$ . 
>  令 $r_t(\theta)$ 表示概率比率 $r_t(\theta) = \frac {\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$，故 $r(\theta_{old} ) = 1$

TRPO maximizes a “surrogate” objective 

$$
L^{C P I}(\theta)=\hat{\mathbb{E}}_{t}\bigg[\frac{\pi_{\theta}(a_{t}\mid s_{t})}{\pi_{\theta_{\mathrm{old}}}(a_{t}\mid s_{t})}\hat{A}_{t}\bigg]=\hat{\mathbb{E}}_{t}\Big[r_{t}(\theta)\hat{A}_{t}\Big].\tag{6}
$$ 
The superscript $C P I$ refers to conservative policy iteration [KL02], where this objective was proposed. 

>  TRPO 的目标函数进而可以写为 Eq 6 的形式
>  上标 CPI 指的是保守策略迭代 (即该方法叫做保守策略迭代，也就是把 TRPO 的约束直接丢弃了)

Without a constraint, maximization of $L^{C P I}$ would lead to an excessively large policy update; hence, we now consider how to modify the objective, to penalize changes to the policy that move $r_{t}(\theta)$ away from 1. 
>  没有约束时，最大化 $L^{CPI}$ 会导致策略更新过大，我们考虑如何修改该目标，以**惩罚使得 $r_t(\theta)$ 远离 1 的策略变化**

The main objective we propose is the following: 

$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\Big[\operatorname*{min}(r_{t}(\theta)\hat{A}_{t},\operatorname{clip}(r_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t})\Big]\tag{7}
$$ 
where epsilon is a hyperparameter, say, $\epsilon=0.2$ . The motivation for this objective is as follows. 

>  我们提出的主目标函数形式如上
>  其中 $\epsilon$ 为超参数

The first term inside the min is $L^{C P I}$ . The second term, $\mathrm{clip}(r_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t}$ , modifies the surrogate objective by clipping the probability ratio, which removes the incentive for moving $r_{t}$ outside of the interval $[1-\epsilon,1+\epsilon]$ . 
>  目标函数是一个 $\min$ 的期望 $\hat {\mathbb E}_t[\min(... , ...)]$
>  $\min$ 的第一项是 $L^{CPI}$，即没有约束的策略梯度目标
>  $\min$ 的第二项是 $\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t$，和 $L^{CPI}$ 相比，就是把 $r_t(\theta)$ 替换为了 $\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+ \epsilon)$，这一项通过裁剪概率比 $r_t(\theta)$ 来修改 $L^{CPI}$，通过裁剪，它移除了使 $r_t(\theta)$ 超出区间 $[1-\epsilon, 1 + \epsilon]$ 的激励 
 
Finally, we take the minimum of the clipped and unclipped objective, so the final objective is a lower bound (i.e., a pessimistic bound) on the unclipped objective. 
>  我们对裁切的目标和未裁切的目标取了更小值，故**最终的目标是未裁切目标的下界** 

>  考虑一下 $\mathrm{clip}$ 的影响，我们暂时去除 $\hat {\mathbb E}_t$，仅考虑一个样本
>  若 $\hat A_t < 0$，说明动作比基准差
>  - 当 $r_t(\theta)\in [1-\epsilon, 1 + \epsilon]$，没有影响，目标不变，为 $r_t(\theta)\hat A_t$
>  - 当 $r_t(\theta) > 1 + \epsilon$，没有影响，目标不变，为 $r_t(\theta)\hat A_t$
>  - 当 $r_t(\theta) < 1-\epsilon$，有影响，目标被裁剪为 $(1-\epsilon) \hat A_t$，比原目标小
>  若 $\hat A_t > 0$，说明动作比基准好
>  - 当 $r_t(\theta)\in [1-\epsilon, 1 + \epsilon]$，没有影响，目标不变，为 $r_t(\theta)\hat A_t$
>  - 当 $r_t(\theta) > 1 + \epsilon$，有影响，目标被裁剪为 $(1 + \epsilon)\hat A_t$，比原目标小
>  - 当 $r_t(\theta) < 1-\epsilon$，没有影响，目标不变，为 $r_t(\theta)\hat A_t$

With this scheme, we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse. 
>  通过该方案，我们在概率比变化会使目标函数改善时忽略它，并且在它使目标函数变差时将它纳入考虑

> (带基线的) 策略梯度方法的目标永远是增大 $\hat A_t$
> 当 $\hat A_t < 0$，当前动作低于基线
> 在 $r_t(\theta) > 1+\epsilon$ 时，目标不变。满足 $r_t(\theta) > 1+\epsilon$ 的 $\theta$ 所定义的策略中，当前动作出现的概率将更大，也就是满足 $r_t(\theta) > 1 + \epsilon$ 的 $\theta$ 将定义更差的策略
> 在 $r_t(\theta) < 1- \epsilon$ 时，目标被裁剪为 $(1-\epsilon) \hat A_t$，防止它更大。满足 $r_t(\theta) < 1-\epsilon$ 的 $\theta$ 所定义的策略中，当前动作出现的概率将更小，也就是满足 $r_t(\theta) < 1-\epsilon$ 的 $\theta$ 将定义更好的策略
> 从优化视角出发，此时的优化趋势应该是向 $r_t(\theta) < 1$ 的方向前进的，因此本质上 clipping 还是为了防止过激的优化，因为过激的优化的目标是让当前动作的概率尽可能变小，这会使得 $r_t(\theta)$ 尽可能小，如果没有 clipping，在 $r_t(\theta)$ 低于阈值 $1-\epsilon$ 时，目标函数 $L^{CPI}$ 仍然可以随着 $r_t(\theta)$ 的变小而更大，故仍然存在进一步的优化趋势，进行了 clipping 后，$r_t(\theta)$ 低于阈值后，目标函数不会再随着 $r_t(\theta)$ 的变小而更大了，进而也不会再存在进一步的优化趋势 (目标函数随 $\theta$ 的梯度/变化率为零)
> 因为 $\hat A_t$ 是 critic 给出的，critic 存在一定的 off-policy 情况 (优化 critic 的样本是 off-policy 收集的，从长期行为看，critic 将不会收敛到当前 policy 的 critic)，因此 actor 需要慎重考虑 $\hat A_t$ 的参考价值是否应该限制
> 作者这里的意思可能是想说过激的优化虽然看似会改善策略 (进而改善目标函数) 但实际上会破坏策略 (进而导致目标函数变差)，因此需要加以限制
> 当 $\hat A_t > 0$，当前动作高于基线，分析同上

Note that $L^{C L I P}(\theta)=L^{C P I}(\theta)$ to first order around $\theta_{\mathrm{old}}$ (i.e., where $r=1$ ), however, they become different as $\theta$ moves away from $\theta_{\mathrm{old}}$ . Figure $1$ plots a single term (i.e., a single $t$ ) in $L^{C L I P}$ ; note that the probability ratio $r$ is clipped at $1-\epsilon$ or $1+\epsilon$ depending on whether the advantage is positive or negative. 
>  $L^{CLIP}(\theta)$  (加了 clipping 的目标) 和 $L^{CPI}(\theta)$ (没有加 clipping 的目标) 在 $\theta_{old}$ 附近 (即 $r= 1$ 时) 是一阶近似相等的，当 $\theta$ 远离 $\theta_{old}$ 时，它们会变得不同
>  Figure 1 展示了 $L^{CLIP}$ 中的单独项 (单个 $t$) 关于 $r_t(\theta)$ 的变化曲线，注意到概率比 $r$ 根据优势是正值还是负值，在 $1-\epsilon$ 或 $1+\epsilon$ 处被裁剪

![[pics/PPO-Fig1.png]]

Figure 2 provides another source of intuition about the surrogate objective $L^{C L I P}$ . It shows how several objectives vary as we interpolate along the policy update direction, obtained by proximal policy optimization (the algorithm we will introduce shortly) on a continuous control problem. We can see that $L^{C L I P}$ is a lower bound on $L^{C P I}$ , with a penalty for having too large of a policy update. 
>  Fig 2 提供了关于替代目标 $L^{CLIP}$ 的另一种直观展示
>  Fig 2 展示了当我们沿着 PPO 算法在连续控制问题中计算得到的策略更新方向 (梯度方向) 进行插值时，不同的目标函数的变化情况
>  可以看到 $L^{CLIP}$ 是 $L^{CPI}$ 的一个下界，并且 $L^{CLIP}$ 对过大的策略更新幅度施加了惩罚  ($r_t(\theta)$ 超出阈值时，$L^{CLIP}$ 被 clipping 为与 $\theta$ 无关的值，相对于 $\theta$ 的梯度被 clipping 为零)

![[pics/PPO-Fig2.png]]

>  总感觉这图有问题，不理解红线为什么会下降
>  事实上，感觉 $L^{CLIP}$ 中，$\min$ 是多余的，因为 clipping 之后的 $r_t(\theta) \hat A_t$ 只可能更小，不可能更大

# 4 Adaptive KL Penalty Coefficient 
Another approach, which can be used as an alternative to the clipped surrogate objective, or in addition to it, is to use a penalty on KL divergence, and to adapt the penalty coefficient so that we achieve some target value of the KL divergence $d_{\mathrm{targ}}$ each policy update.
>  另一种可以作为 clipped surrogate objective 的替代方案，或者与其结合使用的方法是对 KL 散度施加惩罚，并调整惩罚系数，以便在每次策略更新时达到某个目标 KL 散度值 $d_{targ}$
 
In our experiments, we found that the KL penalty performed worse than the clipped surrogate objective, however, we’ve included it here because it’s an important baseline. 
>  我们的试验发现 KL 惩罚的表现差于 clipped surrogate objective，因此我们将它作为 baseline
 
In the simplest instantiation of this algorithm, we perform the following steps in each policy update: 
- Using several epochs of minibatch SGD, optimize the KL-penalized objective 

$$
{\cal L}^{K L P E N}(\theta)=\hat{\mathbb{E}}_{t}\bigg[\frac{\pi_{\theta}\left(a_{t}\mid s_{t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(a_{t}\mid s_{t}\right)}\hat{A}_{t}-\beta\mathrm{KL}\big[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_{t}),\pi_{\theta}(\cdot\mid s_{t})\big]\bigg]\tag{8}
$$ 
- Compute $d=\hat{\mathbb{E}}_{t}[\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_{t}),\pi_{\theta}(\cdot\mid s_{t})]]$
    - If $d < d_{targ}/1.5$, $\beta \leftarrow \beta /2$
    - If $d > d_{targg}/1.5$, $\beta \leftarrow \beta \times 2$

>  在算法的最简单实现中，我们在每次策略更新执行以下步骤：
>  - 使用多轮 minibatch SGD 优化 KL 惩罚的目标 Eq 8
>  - 计算 KL 散度 $d$，如果 $d < d_{targ}/1.5$，减少惩罚系数 $\beta$ (促使下一次更新的 $d$ 能够更大)，否则增大惩罚系数 $\beta$ (促使下一次更新的 $d$ 能够更小)

The updated $\beta$ is used for the next policy update. With this scheme, we occasionally see policy updates where the KL divergence is significantly different from $d_{\mathrm{targ}}$ , however, these are rare, and $\beta$ quickly adjusts. The parameters 1.5 and 2 above are chosen heuristically, but the algorithm is not very sensitive to them. The initial value of $\beta$ is a another hyperparameter but is not important in practice because the algorithm quickly adjusts it. 
>  更新后的 $\beta$ 值会用于下一次策略更新
>  在该方案下，我们偶尔会看到 KL 散度值显著和 $d_{targ}$ 不同的策略更新，但这种情况很少见，并且 $\beta$ 可以迅速调整
>  参数 1.5 和 2 是凭经验选择的，但算法对它们不敏感，$\beta$ 的初始值是另一个超参数，但在实际应用中不重要，因为算法会迅速调整它

# 5 Algorithm 
The surrogate losses from the previous sections can be computed and differentiated with a minor change to a typical policy gradient implementation. 
>  之前描述的 surrogate 的梯度的计算只需要对典型的策略梯度计算公式进行略微的修改即可

For implementations that use automatic differentiation, one simply constructs the loss $L^{C L I P}$ or $L^{K L P E N}$ instead of $L^{P G}$ , and one performs multiple steps of stochastic gradient ascent on this objective. 
>  如果使用自动微分，则我们需要做的仅仅是构造目标函数 $L^{CLIP}$ 或 $L^{KLPEN}$ 用以替代经典的目标函数 $L^{PG}$，然后针对该目标函数执行多步 SGA 即可

Most techniques for computing variance-reduced advantage-function estimators make use a learned state-value function $V(s)$ ; for example, generalized advantage estimation [Sch+15a], or the finite-horizon estimators in [Mni+16]. If using a neural network architecture that shares parameters between the policy and value function, we must use a loss function that combines the policy surrogate and a value function error term. 
>  计算 (用于减小方差的) 优势函数估计值的技术一般会使用一个学习到的状态价值函数 $V(s)$，例如广义优势估计或有限期估计
>  如果使用策略函数和价值函数共享参数的神经网络架构，我们就需要使用结合了策略的目标和价值的误差的损失项

This objective can further be augmented by adding an entropy bonus to ensure sufficient exploration, as suggested in past work [Wil92; Mni+16].
>  可以通过添加熵奖励项 (奖励策略的熵值) 来进一步增强该损失项，以确保足够的探索性

Combining these terms, we obtain the following objective, which is (approximately) maximized each iteration: 

$$
L_{t}^{C L I P+V F+S}(\theta)=\hat{\mathbb{E}}_{t}\big[L_{t}^{C L I P}(\theta)-c_{1}L_{t}^{V F}(\theta)+c_{2}S[\pi_{\theta}](s_{t})\big],\tag{9}
$$

where $c_{1},c_{2}$ are coefficients, and $S$ denotes an entropy bonus, and $L_{t}^{V F}$ is a squared-error loss $(V_{\theta}(s_{t})-V_{t}^{\mathrm{targ}})^{2}$ . 

>  结合这些项，我们得到如上目标函数，在每轮迭代中，我们近似地最大化该目标函数
>  其中 $c_1, c_2$ 为系数，$S$ 表示计算熵值，$L_t^{VF}$ 是价值函数的误差项，为均方误差损失 $(V_\theta(s_t) - V_t^{targ})^2$

One style of policy gradient implementation, popularized in [Mni+16] and well-suited for use with recurrent neural networks, runs the policy for $T$ timesteps (where $T$ is much less than the episode length), and uses the collected samples for an update. This style requires an advantage estimator that does not look beyond timestep $T$ . The estimator used by [Mni+16] is 

$$
{\hat{A}}_{t}=-V(s_{t})+\underbrace{r_{t}+\gamma r_{t+1}+\cdot\cdot\cdot+\gamma^{T-t-1}r_{T-1}+\gamma^{T-t}V(s_{T})}_{Q(s_t,a_t)的多步\text{Monte Carlo} 近似}\tag{10}
$$

where $t$ specifies the time index in $[0,T]$ , within a given length- $T$ trajectory segment. 

>  由 [Mni+16] 推广，并非常适合与 RNN 一起使用的一种策略梯度实现会执行 $T$ 个时间步的策略 ($T$ 小于回合长度)，然后使用收集到的样本执行一次更新 (即轨迹长度为 $T$)
>  这种方法用到的优势函数估计值会向前看 $T$ 个时间步 (或者更少)，其形式如上，其中 $t\in [0, T]$

Generalizing this choice, we can use a truncated version of generalized advantage estimation, which reduces to Equation (10) when $\lambda=1$ : 

$$
\begin{align}
&\hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + \cdots + (\gamma\lambda)^{T-t-1}\delta_{T-1}\tag{11}\\
&\text{where}\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\tag{12}
\end{align}
$$

>  将 Eq 10 推广，我们可以使用广义优势估计的截断版本，形式如上，在 $\lambda = 1$ 时，它等价于 Eq 10

>  推导

$$
\begin{align}
\hat A_t&= \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + \cdots + (\gamma\lambda)^{T-t-1}\delta_{T-1}\\
&=r_t + \gamma V(s_{t+1}) - V(s_t)\\
&\quad +\gamma\lambda (r_{t+1}+\gamma V(s_{t+2}) - V(s_{t+1}))\\
&\quad + \cdots\\
&\quad + (\gamma\lambda)^{T-t-1}(r_{T-1} + \gamma V(s_{T})-V(s_{T-1}))
\end{align}
$$

>  其中: 
>  - $r_t + \gamma V(s_{t+1}) - V(s_t)$ 是对 $A_t = Q(s_t, a_t) - V(s_t)$ 的 Monte Carlo 近似
>  - $r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1})$ 是对 $A_{t+1} = Q(s_{t+1}, a_{t+1}) - V(s_{t+1})$ 的 Monte Carlo 近似
>  - $\cdots$
>  - $r_{T-1} + \gamma V(s_T) - V(s_{T-1})$ 是对 $A_{T-1} = Q(s_{T-1}, a_{T-1}) - V(s_{T-1})$ 的 Monte Carlo 近似
>  各个近似按照时序，会乘上折扣因子 $\gamma\lambda$
>  - 如果 $\lambda = 1$，原式等价于 $- V(s_t) + r_t + \cdots + \gamma^{T-t-1}r_{T-1} + \gamma^{T-t}V(s_T)$，$A_t, \dots, A_{T-1}$ 仅基于 $\gamma^t$ 加权求和，或者直接视作使用多时间步近似 $Q(s_t, a_t)$
>  - 如果 $\lambda < 1$，则权重衰减更快，如果 $A_t, \dots, A_{T-1}$ 负数较多，即采样到的轨迹较差，应该会高估 $Q(s_t, a_t)$，反之如果轨迹较好，应该会低估 $Q(s_t, a_t)$
>  - 如果 $\lambda > 1$，则权重衰减更慢，如果 $A_t, \dots, A_{T-1}$ 正数较多，即采样到的轨迹较良好，应该会高估 $Q(s_t, a_t)$，反之如果轨迹较差，应该会低估 $Q(s_t, a_t)$
>  个人觉得应该要高估 $Q(s_t, a_t)$，因为当前的 critic 是存在一定滞后的

A proximal policy optimization (PPO) algorithm that uses fixed-length trajectory segments is shown below. Each iteration, each of $N$ (parallel) actors collect $T$ timesteps of data. Then we construct the surrogate loss on these $N T$ timesteps of data, and optimize it with minibatch SGD (or usually for better performance, Adam [KB14]), for $K$ epochs. 
>  近端策略优化算法的描述如下
>  PPO 算法使用固定长度的轨迹段，每次迭代中，$N$ 个并行的智能体各自收集 $T$ 个时间步的数据，然后我们基于总的 $NT$ 时间步的数据构造 surrogate loss，并使用 minibatch SGD 对其进行优化
>  算法一共执行 $K$ 个 epochs

![[pics/PPO-Algorithm1.png]]

# 6 Experiments 
## 6.1 Comparison of Surrogate Objectives 
First, we compare several different surrogate objectives under different hyperparameters. Here, we compare the surrogate objective $L^{C L I P}$ to several natural variations and ablated versions. 

No clipping or penalty:  $L_t(\theta) = r_t(\theta) \hat A_t$
Clipping: $L_t(\theta) = \min(r_t(\theta)\hat A_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1 + \epsilon)\hat A_t)$
KL penalty (fixed or adaptive) : $L_t(\theta) = r_t(\theta)\hat A_t - \beta\mathrm{KL}[\pi_{\theta_{old}}, \pi_\theta]$

>  我们在不同的参数下比较不同类型的 surrogate objectives，主要是 $L^{CLIP}$ 和它的变体

For the KL penalty, one can either use a fixed penalty coefficient $\beta$ or an adaptive coefficient as described in Section 4 using target KL value $d_{\mathrm{targ}}$ . 
>  KL 惩罚的 surrogate objective 可以使用固定的惩罚系数 $\beta$ 或者使用基于目标 KL 值 $d_{targ}$ 的自适应系数

Note that we also tried clipping in log space, but found the performance to be no better. 
>  我们尝试了在对数空间进行裁剪，但性能并未更好

Because we are searching over hyperparameters for each algorithm variant, we chose a computationally cheap benchmark to test the algorithms on. Namely, we used 7 simulated robotics tasks implemented in OpenAI Gym [Bro+16], which use the MuJoCo [TET12] physics engine. We do one million timesteps of training on each one. Besides the hyperparameters used for clipping ($\epsilon$) and the KL penalty $(\beta,d_{\mathrm{targ}})$ , which we search over, the other hyperparameters are provided in in Table 3. 
>  因为我们要为每种算法变体搜索超参数，故我们选择了一个计算成本较低的基准测试来测试这些算法
>  我们的基准测试包括了 OpenAI Gym 的 7 个模拟机器人任务，这些任务使用了 MuJoCo 物理引擎
>  我们在每个任务上进行了 100 万步的训练，用于 clipping 的超参数 $\epsilon$ 和用于 KL 惩罚项的超参数 $(\beta, d_{targ})$ 通过搜索得到，其他超参数的设定如 Table 3 所示

To represent the policy, we used a fully-connected MLP with two hidden layers of 64 units, and tanh nonlinearities, outputting the mean of a Gaussian distribution, with variable standard deviations, following [Sch+15b; Dua+16]. We don’t share parameters between the policy and value function (so coefficient $c_{1}$ is irrelevant), and we don’t use an entropy bonus. 
>  我们使用了一个带有两个 64 单元隐藏层的全连接 MLP 表示策略，使用 tanh 激活函数。策略函数输出高斯分布的均值，高斯分布的标准差为可变值
>  策略函数和价值函数之间不共享参数，因此系数 $c_1$ 无关紧要，并且我们不使用 entropy bonus

Each algorithm was run on all 7 environments, with 3 random seeds on each. We scored each run of the algorithm by computing the average total reward of the last 100 episodes. 
>  每个算法在 7 个环境运行，每个环境使用 3 个随机种子
>  每个算法的得分通过计算其最后 100 回合的平均总奖励得到

We shifted and scaled the scores for each environment so that the random policy gave a score of 0 and the best result was set to 1, and averaged over 21 runs to produce a single scalar for each algorithm setting. 
>  我们将算法在每个环境下得到的分数进行平移和缩放处理，使得随机策略得分为 0 ，最佳结果设为 1 (用一个随机策略在环境中跑一次，以它的得分为 0 基准，最佳算法的得分为 1 基准，线性放缩其他算法的得分)
>  对每个环境的分数执行方所后，我们计算每个算法的最终得分，即算法在 7 个环境一共 21 次运行的平均分数，最终每个算法有一个单一的标量值分数

The results are shown in Table 1. Note that the score is negative for the setting without clipping or penalties, because for one environment (half cheetah) it leads to a very negative score, which is worse than the initial random policy. 
>  结果如 Table 1 所示，注意在没有 clipping 和 KL 惩罚的情况下的算法分数为负值 (比随机策略还要差)，因为在我们的 half cheetah 环境中该算法得到了非常负面的得分，比初始随机策略还要差

![[pics/PPO-Table1.png]]

## 6.2 Comparison to Other Algorithms in the Continuous Domain 
Next, we compare PPO (with the “clipped” surrogate objective from Section 3) to several other methods from the literature, which are considered to be effective for continuous problems. We compared against tuned implementations of the following algorithms: trust region policy optimization [Sch+15b], cross-entropy method (CEM) [SL06], vanilla policy gradient with adaptive stepsize,  A2C [Mni+16], A2C with trust region [Wan+16]. 
>  我们将 PPO (with clipped surrogate objective) 和目前被认为是对于连续控制问题有效的其他方法比较，包括了 TRPO、交叉熵方法、带有自适应步长的朴素策略梯度、A2C、A2C with 置信域
>  其中自适应步长的朴素策略梯度即在每次更新后，基于新策略和旧策略的 KL 散度调节 Adam 的 stepsize 参数

A2C stands for advantage actor critic, and is a synchronous version of A3C, which we found to have the same or better performance than the asynchronous version. 
>  A2C 指 Advantage Actor Critic，它是 A3C 的同步版本，我们发现 A2C 的性能与异步版本相同或更优

![[pics/PPO-Fig3.png]]

For PPO, we used the hyperparameters from the previous section, with $\epsilon=0.2$ . We see that PPO outperforms the previous methods on almost all the continuous control environments. 
>  可以看到，PPO 在几乎所有连续控制场景中都优于之前的方法

## 6.3 Showcase in the Continuous Domain: Humanoid Running and Steering 
To showcase the performance of PPO on high-dimensional continuous control problems, we train on a set of problems involving a 3D humanoid, where the robot must run, steer, and get up off the ground, possibly while being pelted by cubes. 

The three tasks we test on are 
(1) RoboschoolHumanoid: forward locomotion only, 
(2) RoboschoolHumanoidFlagrun: position of target is randomly varied every 200 timesteps or whenever the goal is reached, 
(3) RoboschoolHumanoidFlagrunHarder, where the robot is pelted by cubes and needs to get up off the ground. 

See Figure 5 for still frames of a learned policy, and Figure 4 for learning curves on the three tasks. Hyperparameters are provided in Table 4. In concurrent work, Heess et al. [Hee+17] used the adaptive KL variant of PPO (Section 4) to learn locomotion policies for 3D robots. 

>  为了展示 PPO 在高维连续控制问题上的性能，我们在一组设计三维类人机器人的问题上进行训练，机器人需要跑、转向、从地面上站起，也可能被立方体击打
>  我们测试的三个任务是：
>  (1) 仅向前移动
>  (2) 目标位置每 200 个时间步或当目标达成时随机变化
>  (3) 机器人会被立方体击打，需要从地面站起

![[pics/PPO-Fig4.png]]
![[pics/PPO-Fig5.png]]

## 6.4 Comparison to Other Algorithms on the Atari Domain 
We also ran PPO on the Arcade Learning Environment [Bel+15] benchmark and compared against well-tuned implementations of A2C [Mni+16] and ACER [Wan+16]. For all three algorithms, we used the same policy network architecture as used in [Mni+16]. The hyperparameters for PPO are provided in Table 5. For the other two algorithms, we used hyperparameters that were tuned to maximize performance on this benchmark. 
>  我们在 ALE 基准上运行 PPO，将效果和 A2C 和 ACER 比较
>  对于这三种算法，我们都使用相同的策略网络架构

A table of results and learning curves for all 49 games is provided in Appendix B. We consider the following two scoring metrics: (1) average reward per episode over entire training period (which favors fast learning), and (2) average reward per episode over last 100 episodes of training (which favors final performance). Table 2 shows the number of games “won” by each algorithm, where we compute the victor by averaging the scoring metric across three trials. 
>  我们考虑以下两个评分指标
>  (1) 整个训练过程中每回合的平均奖励 (该指标有利于快速学习的算法，因为学得快的算法奖励涨得比较快，不容易在奖励较低的时段持续)
>  (2) 训练过程中最后 100 个回合的平均奖励 (该指标有利于最终表现优的算法)
>  我们计算三次试验的平均分数来选出胜者

![[pics/PPO-Table2.png]]

# 7 Conclusion 
We have introduced proximal policy optimization, a family of policy optimization methods that use multiple epochs of stochastic gradient ascent to perform each policy update. These methods have the stability and reliability of trust-region methods but are much simpler to implement, requiring only few lines of code change to a vanilla policy gradient implementation, applicable in more general settings (for example, when using a joint architecture for the policy and value function), and have better overall performance. 
>  我们介绍了 PPO，这是一类使用 **多轮 (multiple epochs)** 随机梯度上升来执行每次策略更新的策略优化方法
>  PPO 具有 TRPO 的稳定性和可靠性，且更易于实现，只需对标准的策略梯度实现进行少量代码修改即可
>  PPO 适用于更广泛的场景 (例如策略函数和价值函数共享架构时也可以使用 PPO)，并且整体性能更佳

>  实际上我觉得 PPO 主要在于提高了样本利用率，可以在更少的训练轮数下更快收敛，达到相同的性能需要收集更少的经验
>  或许给其他的策略梯度方法例如 A2C 更多的训练轮数 (或许要几倍于 PPO)，它们可以达到可能更优的表现，毕竟它们是完全 on-policy 的 (当然 critic 给出的评价肯定仍然是存在偏移的，从这一点的角度出发，对于策略的过激更新进行限制或许在 on-policy 情况下也是存在合理性的)，但鉴于效率问题，PPO 在有限的训练轮数下确实是更优的选择

# A Hyperparameters 
Tables ref to original paper

# B Performance on More Atari Games 
Here we include a comparison of PPO against A2C on a larger collection of 49 Atari games. Figure 6 shows the learning curves of each of three random seeds, while Table 6 shows the mean performance. 
