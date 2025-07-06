# Abstract
Dense process rewards have proven a more effective alternative to the sparse outcome-level rewards in the inference-time scaling of large language models (LLMs), particularly in tasks requiring complex multi-step reasoning.
> 在 LLM 的推理时拓展下，密集的过程奖励已经证明是稀疏的结果级奖励更加有效，尤其是对于要求多步复杂推理的任务 

While dense rewards also offer an appealing choice for the reinforcement learning (RL) of LLMs since their fine-grained rewards have the potential to address some inherent issues of outcome rewards, such as training efficiency and credit assignment, this potential remains largely unrealized. 
> 密集的过程奖励的细粒度性质有助于解决稀疏的结果奖励中的一些内在问题，例如训练效率和信用分配问题，但尚且没有工作实现这一潜力

> 稀疏的结果奖励下，只有 LLM 最终给出正确答案才会获得奖励，信用分配问题是指难以确定哪些中间步骤导致了结果的成功或失败
> 密集的过程奖励下，奖励信号更加集中和细粒度，频繁的奖励信号提供了更密集的梯度，加速了学习过程

This can be primarily attributed to the challenges of training process reward models (PRMs) online, where collecting high-quality process labels is prohibitively expensive, making them particularly vulnerable to reward hacking. 
> 这一潜力尚未实现的原因可以归结于在线训练过程奖励模型的挑战: 训练 PRM 需要大量的高质量过程标签，导致 PRM 容易受到奖励 hacking 的影响 (奖励 hacking 即模型找到捷径或不理想的策略骗取奖励，这些策略并不代表高质量的推理行为)

To address these challenges, we propose PRIME (Process Reinforcement through IMplicit rEwards), which enables online PRM updates using only policy rollouts and outcome labels through implicit process rewards. 
> 我们提出 PRIME (通过隐式奖励进行过程强化)，PRIME 允许仅使用策略运行和结果标签在线更新 PRM (不需要大量高质量过程标签)
> (隐式奖励即从最终的结果奖励中，推断或反向传播出对中间过程的奖励信号，故无需显式地标注每个中间步骤)

PRIME combines well with various advantage functions and forgoes the dedicated reward model training phase that existing approaches require, substantially reducing the development overhead. 
> PRIME 可以和各种优势函数良好结合，且无需 (现有方法都需要的) 专门的奖励模型训练阶段，故显著减少了成本

We demonstrate PRIME's effectiveness on competitional math and coding. Starting from Qwen2.5-Math-7B-Base, PRIME achieves a $15.1\%$ average improvement across several key reasoning benchmarks over the SFT model. Notably, our resulting model, Eurus-2-7B-PRIME, surpasses Qwen2.5-Math-7B-Instruct on seven reasoning benchmarks with $10\%$ of its training data.
> PRIME 在 Qwen2.5-Math-7B-Base 上，相较于 SFT 模型，在几个推理 benchmark 上达到了 15.1% 的性能提升
> Eurus-2-7B-PRIME 在仅使用了 10% 的 Qwen2.5-Math-7B-Instruct 的训练数据下，在几个推理 benchmark 上超越了 Qwen2.5-Math-7B-Instruct

# 1 Introduction
Dense process rewards, which provide feedback at each intermediate step rather than only the whole trajectory, have proven effective in inference-time scaling of large language models (LLMs) on challenging reasoning tasks (Uesato et al., 2022; Lightman et al., 2023; Wang et al., 2023; Yuan et al., 2024b). 
> 密集过程奖励在每个中间步都提供奖励，而不是仅仅对整个轨迹提供奖励
> 密集过程奖励对于 LLM 的推理时拓展是有效的

On the training side, they also present superiorities in the reinforcement learning (RL) of LLMs, particularly in improving training efficiency (Sutton & Barto, 2018) and credit assignment (Leike et al., 2018) compared with sparse outcome rewards. 
> 在 LLM 的训练期间，密集过程奖励相较于稀疏结果奖励也更有助于提高训练效率和信用分配

However, successful applications of dense rewards in RL for LLMs are limited (Setlur et al., 2024), as current industry-leading models primarily depend on verifiable outcome rewards and have not yet demonstrated meaningful progress with dense rewards (DeepSeek-AI et al., 2025; Team et al., 2025).
> 密集过程奖励在 LLM 的成功应用较少，当前的领先模型主要依赖于结果奖励

We identify the central challenge as how to acquire and utilize high-quality dense rewards at scale, which enables online process reward model (PRM) update efficiently. The reason is that, optimizing towards a static reward model eventually leads to overoptimization or reward hacking (Gao et al., 2022) due to distribution shift. 
> 我们认为关键的挑战在于如何大规模地获取和利用高质量的密集奖励来更新 PRM
> 为什么要在线更新 PRM: 原因在于，如果优化目标是一个静态的 (不会改变的) 奖励模型，由于分布偏移，最终会导致过度优化或奖励黑客

> RL 训练过程中，策略会不断改进，即策略分布本身会发生偏移，故奖励模型应该随之更新，否则新策略分布下的行为可能无法被准确评估
> 如果奖励模型固定，模型容易针对奖励模型过度优化，发展成骗奖励的策略

Ideally, this can be solved by improving the reward model online (Leike et al., 2018). However, acquiring dense process labels for training is prohibitively more expensive. Existing methods either need to build complicated human annotation pipelines (Ligttman et al., 2023) or rely on estimation-based methods, which require about $10\times$ more rollouts for each step than sampling only the response-level trajectories (Wang et al., 2023; Kazemnejad et al., 2024). Neither of them is scalable in online RL. Moreover, to the best of our knowledge, it remains underexplored how to incorporate dense rewards into RL for LLMs.
> 在线更新 PRM 可以解决过度优化和骗奖励的问题，但为训练获取密集的过程标签过于昂贵
> 现有的方法包括:
> -构建标记人工流水线
> -使用基于估计的方法，通常需要多 10 倍的策略运行才能在每一步获取有效信号
> 这两类方法都是不可拓展的，同时如何将密集奖励融入 LLM 的训练尚未有人探索

In this work, we propose Process Reinforcement through Implicit Rewards (PRIME), a scalable framework for enhancing reasoning capabilities via efficient reinforcement learning with dense token-level rewards. At its core, the framework employs recently proposed implicit process reward modeling (Yuan et al., 2024b) to train dense reward models with only outcome-level labels. This enables PRIME to perform online learning of reward signals using only outcome labels on policy rollouts, thereby fundamentally mitigating reward hacking while maintaining the same computational cost as traditional outcome reward models (ORMs). 
> PRIME 采用隐式过程奖励建模来仅仅通过输出级别的标签训练密集的 (token 级的) 奖励模型
> 故 PRIME 仅需要使用结果级的标签来训练密集奖励模型，因此可以在维持和传统的结果奖励模型相同计算开销的情况下，缓解骗奖励问题

Besides scalability, PRIME also (1) serves as a general method to fuse token-level dense rewards and sparse outcome rewards by calculating their returns separately before summing together, which is compatible with diverse RL algorithms (Williams, 1992; Kool et al., 2019; Shao et al., 2024; Ahmadian et al., 2024; Schulman et al., 2017); (2) eliminates the dedicated reward modeling stage, which is required by existing works, by simply initializing from the SFT model or even the base model $(\S 5.6)$ . 
> 除了可拓展性，PRIME 的优势还有:
> 1. 提供了一种通用方法来融合 token 级密集奖励和稀疏的结果奖励: 分别计算他们的回报，然后相加
> 2. 消除了现有方法中专门的奖励建模阶段: 直接从 SFT 模型或基础模型进行初始化 (无需为奖励模型收集大量的数据以及预训练)

In summary, starting from one single language model, the PRIME framework can efficiently accomplish the generation of dense rewards, the initialization and updating of reward models, as well as the reinforcement learning (RL) training of the policy model.
> PRIME 仅需要从单个语言模型开始，就能完成密集奖励的生成、奖励模型的初始化和更新、策略模型的 RL 训练

In experiments, we train Qwen2.5-Math-7B-Base (Yang et al., 2024b) with PRIME after a lightweight SFT warmup stage. Compared to RL using outcome rewards only, PRIME achieves a $2.5\times$ sample efficiency gain and a $6.9\%$ performance improvements on challenging math problems. 
> 我们在一个轻量的 SFT warmup 阶段后，用 PRIME 训练的 Qwen2.5-Math-7B-Base
> 相较于仅使用结果奖励的 RL 方法, PRIME 训练出的模型在数学问题上达到了 2.5x 的样本效率提升和 6.9% 的性能提升

![[pics/Process Reinforcement Through Implict Rewards-Fig1.png]]

As shown in Figure 1, through PRIME, we successfully achieve substantial improvement on key mathematical reasoning benchmarks over the SFT model, leading to $16.7\%$ improvement on average, and over $20\%$ on AMC&AIME competitions. Our final model Euras-2-7B-PRIME surpassed Qwen2.5-Math-7B-Instruct on five key mathematical benchmarks. 
> Eurus-2-7B-PRIME 与各个模型的对比见 Figure 1

Notably, this is achieved with only $10\%$ of the data used by Qwen-Math, as in Table 1.

| Model   | Eurus-2-7B-PRIME     | Qwen2.5-Math-7B-Instruct   |
| ----------| ------------------------| -----------------------------|
| Base Model | Qwen2.5-Math-7B     | Qwen2.5-Math-7B        |
| SFT Data  | 230K (open-source)    | 2.5M (open-source & in-house) |
| RM Data  | 0            | 618K (in-house)        |
| RM     | Eurus-2-7B-SFT      | Qwen2.5-Math-RM (72B)     |
| RL Data  | 150K queries × 4 samples | 66K queries × 32 samples   |

Table 1: The comparison of resource requirements between Euras2-7B-PRIME and Qwen2.5-Math-7B-Instruct. 

> Eurus-2-7B-PRIME 和 Qwen2.5-Math-7B-Instruct 都基于 Qwen2.5-Math-7B 训练
> 前者的 SFT 数据量为 230K，后者为 2.5M
> 前者的奖励模型为 Eurus-2-7B-SFT，后者的奖励模型为 72B 的专门训练的奖励模型

Our analysis shows that updating the PRM online is key to the success of PRIME (§5.1). We also show that PRIME could generally boost various RL algorithms, including RLOO (Ahmadian et al., 2024), REINFORCE (Williams, 1992), PPO (Schulman et al., 2017), and GRPO (Shao et al., 2024) (§5.4). In terms of the design choices of advantage estimate, we observe that Implicit PRMs are better to be used as reward models than value models (§5.5).
> PRM 的关键在于在线更新奖励模型
> PRIME 可以强化多个 RL 算法: RLOO, REINFORCE, PPO, GRPO
> 在奖励估计上，使用隐式 PRM 作为奖励模型比使用 value model 更优

# 2 Reinforcement Learning for LLMs and The Challenges of Incorporating Dense Rewards
Reinforcement Learning (RL) aims to learn an optimal policy $\pi_{\theta}$ that maximizes the expected cumulative discounted reward, namely return, when interacting with an environment. 
>  RL 的目标是学习在和环境交互时，能够最大化期望累积折扣奖励 (即回报) 的最优策略 $\pi_\theta$

In the context of autoregressive language modeling, state at step $t$ is the concatenation of prompt $\mathbf{x}$ and current response $\mathbf y_{<t}$ , and the action is the $t$ -th token or step $y_{t}$ .
>  在自回归语言建模的背景下，时间步 $t$ 的状态是 prompt $\mathbf x$ 和当前模型回复 $\mathbf y_{<t}$ 的拼接，时间步 $t$ 的动作是第 $t$ 个 token $y_t$

## 2.1 RL Preliminaries for LLMs
**Policy Gradient.** Policy gradient is a fundamental algorithm that directly optimizes this objective. Central to this approach is the advantage function $A_{t}$ , which quantifies how much better an action is compared to alternatives in a given state:

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\mathbf{x}\sim \mathcal{D},\mathbf{y}\sim \pi_{\theta}}\left[\sum_{t = 0}^{T}\nabla_{\theta}\log \pi_{\theta}(y_{t}|\mathbf{y}_{< t})A_{t}\right] \tag{1}
$$

where $(\mathbf{x},\mathbf{y})$ represents a pair of input and output. $\mathbf{x}$ is omitted for brevity. 

>  采用优势函数的策略梯度定理如 Eq 1 所示，优势函数衡量了在给定状态下，给定动作相较于其他动作 (的平均) 的相对好坏程度
>  Eq 1 中，$(\mathbf x, \mathbf y)$ 表示输入输出对，即状态 (可以看到 Eq 1 对 $\mathbf x, \mathbf y$ 求期望，也就是对状态求期望)，在策略的 condition 中，为了公式简洁，忽略了 $\mathbf x$

>  推导
>  我们的目标是找到一个最优策略，使其最大化累积的期望奖励，对于有限回合的任务，目标函数可以定义为从起始状态 $s_0$ 开始的期望总回报:

$$
\begin{align}
J(\theta) &= \mathbb E_{\tau \sim \pi_\theta}[G(\tau)]\\
&=\mathbb E_{\tau \sim \pi_\theta}[\sum_{t=0}^T \gamma^t R_t]\\
&=\sum_\tau P(\tau ; \theta) G(\tau)
\end{align}
$$

>  其中 $P(\tau ; \theta)$ 表示从 $s_0$ 开始的一条轨迹在策略 $\pi_\theta$ 下发生的概率:

$$
\begin{align}
P(\tau ; \theta) &= p(s_0) \prod_{t=0}^T \pi_\theta(a_t\mid s_t) p(s_{t+1}\mid s_t, a_t)\\
\end{align}
$$

>  目标函数相对于参数 $\theta$ 的梯度为:

$$
\begin{align}
\nabla_{\theta} J(\theta) &= \nabla_\theta \sum_\tau P(\tau ; \theta) G(\tau)\\
&=\sum_\tau\nabla_\theta P(\tau ; \theta) G(\tau)\\
&=\sum_\tau P(\tau ;\theta) \nabla_\theta \log P(\tau ; \theta) G(\tau)\\
&=\mathbb E_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau ; \theta)G(\tau)]\\
\end{align}
$$

>  其中 

$$
\begin{align}
\nabla_\theta \log P(\tau; \theta) &= \nabla_\theta \log p(s_0)\prod_{t=0}^T \pi_\theta(a_t\mid s_t) p(s_{t+1}\mid s_t, a_t)\\
&=\nabla_\theta \left(\log p(s_0) + \sum_{t=0}^T \log \pi_\theta(a_t\mid s_t) + \sum_{t=0}^T\log p(s_{t+1}\mid s_t, a_t)\right)\\
&=\nabla_\theta \log p(s_0) + \nabla_\theta \sum_{t=0}^T \log \pi_\theta(a_t\mid s_t) + \nabla_\theta \sum_{t=0}^T \log p(s_{t+1}\mid s_t, a_t)\\
&= \nabla_\theta \sum_{t=0}^T \log \pi_\theta(a_t\mid s_t)\\
&=  \sum_{t=0}^T \nabla_\theta\log \pi_\theta(a_t\mid s_t)\\
\end{align}
$$

>  故

$$
\begin{align}
\nabla_\theta J(\theta) &= \mathbb E_{\tau\sim \pi_\theta}[\nabla_\theta\log P(\tau; \theta)G(\tau)]\\
&=\mathbb E_{\tau\sim \pi_\theta} [\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t\mid s_t)G(\tau)]
\end{align}
$$

>  上述公式中的 $G(\tau)$ 是整条轨迹的回报，然而在时间步 $t$ 选择的动作 $a_t$ 只能影响时间步 $t$ 以后的回报，即因果性，$t$ 之前的回报和 $a_t$ 是无关的
>  我们将该梯度以另一种形式写出，对因果性进行推导:

$$
\begin{align}
\nabla_\theta J(\theta) 
&=\mathbb E_{\tau\sim \pi_\theta} [\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t\mid s_t)G(\tau)]\\
&=\sum_{t=0}^T \mathbb E_{\tau\sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a_t\mid s_t)G(\tau)]\\
&=\sum_{t=0}^T\mathbb E_{\tau \sim\pi_\theta}[\nabla_\theta\log \pi_\theta(a_t\mid s_t)(\sum_{i=0}^{t-1}\gamma^i R_i + \sum_{i=t}^T\gamma^i R_i)]\\
\end{align}
$$

>  我们考虑求和项中的每一个期望

$$
\mathbb E_{\tau\sim\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t\mid s_t)(\sum_{t=0}^{t-1}\gamma^i R_i + \sum_{i=t}^T \gamma^i R_i)]
$$

>  对于每一个时间步 $t$，在它之前的回报 $\sum_{t=0}^{t-1}\gamma^i R_i$ 是已经确定的，它不依赖于 $a_t$
>  我们可以证明，对于任意不依赖于 $a$ 的函数 $f(s)$，有

$$
\begin{align}
&\mathbb E_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a\mid s)f(s)]\\
=&\mathbb E_{s\sim d^{\pi_\theta}(s)}[\mathbb E_{a\sim \pi_\theta(\cdot\mid s)}[\nabla_\theta \log \pi_\theta(a\mid s)f(s)]]\\
=&\sum_{s} d^{\pi_\theta}(s) \sum_{a}\pi_\theta(a\mid s)[\nabla_\theta \log \pi_\theta(a\mid s)f(s)]\\
=&\sum_s d^{\pi_\theta}(s) f(s)\sum_a \pi_\theta(a\mid s)\nabla_\theta\log \pi_\theta(a\mid s)\\
=&\sum_s d^{\pi_\theta}(s) f(s)\sum_a \nabla_\theta \pi_\theta(a\mid s)\\
=&0
\end{align}
$$

>  注意这里的 $d^\pi_\theta$ 不一定指稳态分布，就是表示了策略 $\pi_\theta$ 下状态 $s$ 被访问的累积概率这个概念，不一定严格达到稳态
>  根据这个结论，求和项中的每一个期望可以进行简化

$$
\begin{align}
&\mathbb E_{\tau\sim\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t\mid s_t)(\sum_{t=0}^{t-1}\gamma^i R_i + \sum_{i=t}^T \gamma^i R_i)]\\
=&\mathbb E_{\tau\sim\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\sum_{i=t}^T \gamma^i R_i]\\
=&\mathbb E_{\tau\sim\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t\mid s_t) \gamma^t G_t]\\
\end{align}
$$

>  其中 $\gamma^t$ 是一个与 $\theta$ 无关的常数，最后如果所有时间步的 $\gamma^t$ 相加，实际上整个式子会乘上常数 $(1+\gamma + \gamma^2+\cdots +  \gamma^T)$ ，因为只是一个常数，故它不影响梯度的方向，只影响梯度的大小，且可以融合到学习率中，进而我们可以将它忽略
>  同样根据这个结论，求和项中的每一个期望可以引入与 $a_t$ 无关的任何函数，例如引入一个只依赖于状态的基线 $b(s_t)$ (通常是对状态价值函数 $V^{\pi_\theta}(s_t)$ 的近似)
>  故最终的策略梯度定理在实践中常常表示为

$$
\begin{align}
\nabla_\theta J(\theta) 
&=\sum_{t=0}^T\mathbb E_{\tau \sim\pi_\theta}[\nabla_\theta\log \pi_\theta(a_t\mid s_t)(\sum_{i=0}^{t-1}\gamma^i R_i + G_t)]\\
&=\sum_{t=0}^T\mathbb E_{\tau \sim\pi_\theta}[\nabla_\theta\log \pi_\theta(a_t\mid s_t) G_t]\\
&=\sum_{t=0}^T\mathbb E_{\tau \sim\pi_\theta}[\nabla_\theta\log \pi_\theta(a_t\mid s_t)  (G_t -b(s_t))]\\
&=\sum_{t=0}^T\mathbb E_{\tau \sim\pi_\theta}[\nabla_\theta\log \pi_\theta(a_t\mid s_t)  A_t]\\
&=\mathbb E_{\tau \sim\pi_\theta}[\sum_{t=0}^T\nabla_\theta\log \pi_\theta(a_t\mid s_t)  A_t]\\
\end{align}
$$

>  推导完毕

In practice, the advantage function is implemented as cumulative discounted rewards subtracting a baseline:

$$
A_{t} = \sum_{s = t}^{T}\gamma^{s -t}r(y_{s}) -b \tag{2}
$$

$\gamma \in [0,1]$ is a discount factor that optionally decays future rewards, and $r(y_{s})$ is the reward provided by the environment at time step $s$ with $x$ and $\mathbf{y}_{< s}$ being omitted in conditions. 

>  实践中优势函数通常定义为累积折扣奖励减去基线
>  Eq 2 中， $r(y_s)$ 是由环境在时间步 $s$ 提供的奖励，实际上 $\sum_{s=t}^T \gamma^{s-t}r(y_s) = G_s$

Eq. 2 is the general formula of the Monte-Carlo (MC) advantage estimate, which indicates that, the high-quality and dense reward at each step is crucial for RL. Different choices of $b$ include, e.g. directly using values Williams (1992), group average of rewards (Shao et al., 2024), and leave-one-out average of rewards (Ahmadian et al. (2024); Kool et al. (2019).
>  计算策略梯度的关键就是估计 Eq 2 中的优势值
>  基线 $b$ 也有许多不同的设定，例如状态价值函数的估计、奖励的组平均、奖励的留一平均

**Value Models.** Though the MC estimate is unbiased, it suffers from high variance because of the reliance on all future actions and rewards, which can be random and noisy. Value models, which predict expected accumulated rewards starting from a state, are adopted to help reduce the variance in advantage estimation, such as Generalized Advantage Estimation (GAE; Schulman et al., 2016): $A_{t}^{\mathrm{GAE}(\gamma ,\lambda)} = \sum_{s = 0}^{\infty}(\gamma \lambda)^{s}\delta_{t + s}$ , where $\delta_{t} = r(y_{t}) + \gamma V(\mathbf{y}_{< t + 1}) -V(\mathbf{y}_{< t})$ is the temporal difference (TD) error (Sutton, 1988), $V$ is a value model, and $\lambda$ controls the bias-variance tradeoff in advantage estimation. PPO (Schulman et al., 2017) is a representative of such actor-critic algorithms that explicitly train a value model along with the policy.
>  直接用 Monte Carlo 估计基线会导致方差过高，目前常采用的方式是另外训练一个价值模型估计基线，例如 Generalized Advantage Estimation
>  PPO 是采用 GAE 模型估计基线的代表性 actor-critic 算法

**Reward Sparsity.** Although dense rewards can be naturally integrated into the advantage function through Eq. 2, unfortunately, only outcome reward models (ORMs) are available in most practices of LLMs, i.e., only the final token bears a meaningful reward while intermediate tokens receive no rewards (Rafailov et al., 2023; Shao et al., 2024; DeepSeek-AI et al., 2025). In this bandit setting, $r(y_{t}) = 0$ for $t< T$ while $r(y_{T})$ can be non-zero, and Eq. 2 becomes $A = r(y_{T}) -b$ . 
>  Eq 2 可以很自然地和密集奖励进行结合，但目前大多数应用采用的都是最终结果奖励模型，即只有最终的 token 具有有意义的奖励，中间的 tokens 则没有奖励，也就是奖励模型的形式为 $r(y_t) = 0,\ \text{for}\ t < T,\ y_T \ne 0$，此时 Eq 2 的形式为 $A = r(y_T) -b$

This formulation, while simpler, can suffer from reward sparsity issues as the policy receives feedback only at the end of the entire generation. This may (1) encourage spurious solutions with incorrect processes but correct answers, (2) largely reduce sample efficiency in training, and (3) encounter the credit assignment problem (Sutton & Barto, 2018). These drawbacks could be further amplified on complicated tasks, which require more thinking and execution steps, urging the need of dense rewards (Uesato et al., 2022; Lightman et al., 2023). 
>  这样的奖励形式存在奖励稀疏性的问题，策略只有在整个生成过程的最后才收到反馈，这可能导致以下三个问题:
>  1. 鼓励模型给出虚假解，即过程错误，但结果正确
>  2. 显著降低训练的样本效率 (这一条感觉很关键)
>  3. 遇到信用分配问题
>  这些问题在遇到复杂任务时可能进一步加剧，因为复杂任务需要更多的思考和执行步骤

Some may consider employing a value model to mitigate the problem, as it predicts values at every step $t$ . However, previous work showed that value models may not be able to solve the reward sparsity issue effectively due to training challenges, despite the additional computation overhead (Shao et al., 2024; Ahmadian et al., 2024). We will also empirically validate this claim in §5.5.
>  有些人考虑用价值模型解决密集奖励的问题，因为价值模型可以在每一步 $t$ 都预测价值
>  但先前的工作表明价值模型可能不可以解决奖励稀疏性的问题，因为价值模型需要额外训练，且使用价值模型也有额外的计算开销
>  本工作也会通过试验验证这一观点

## 2.2 Key Challenges in Scalable Dense Rewards
The way to mitigate the reward sparsity problem is to adopt dense reward models, namely PRMs, which score model responses over each token or step. However, it is usually infeasible in practice to incorporate dense rewards into online RL because of three critical challenges in implementation.
>  缓解奖励稀疏性问题的方法就是采用密集奖励模型，即 PRM
>  PRM 在每一个 token 都为模型回复给出评分
>  但在实践中将密集奖励融入 online RL 存在三个关键挑战:

**C1. Process rewards are hard to define.** It is difficult to collect step-level labels since reasoning steps do not naturally occur in sequences. Although tokens are easily distinguishable, annotating labels for each token is too costly. Moreover, defining the absolute correctness of intermediate processes as dense rewards can be ambiguous, as some incorrect steps can also positively contribute to the final answer by pruning searching branches (OpenAI, 2024; DeepSeek-AI et al., 2025).
>  1. 难以定义过程奖励
>  收集 step-level 的标签非常困难，因为
>  -推理步骤不是简单地线性发生的 (思考过程可能是混乱的，涉及回溯、并行思考或突然的直觉飞跃，不像简单的分类任务一样有清晰的输入输出对)
>  -对每一个 token 标注它对 “推理步骤” 的贡献度或正确性是开销巨大的 (token 也不是一一对应到一个推理步骤，一个推理步骤可能涉及了多个 tokens，且一些语气词等 token 也并不一定和推理相关)
>  -即便愿意标注，由于我们希望奖励的是 “高层次的推理步骤”，这些步骤本身的界限模糊，奖励的定义故也是模糊的，例如一些步骤虽然不正确，但它们对于最终答案的得出也是具有剪枝的积极意义的 (知道什么是 “不正确的”，对于得到正确的答案也是有积极意义的)

**C2. PRM online updates are not scalable.** It is crucial to prevent reward overoptimization or reward hacking, which requires the reward model or value model to be updated online along with the policy model (Schulman et al., 2017; Gao et al., 2022). However, training PRMs often requires extensive nuanced step-level annotation, which is infeasible in online RL training. Therefore, this brings about considerable scalability and generalization concerns in dense rewards for RL.
>  2. PRM 的在线更新不是可拓展的
>  为了避免 LLM 过度优化奖励/骗奖励，奖励模型或价值模型也需要随着策略模型一起更新，而训练 PRM 需要大量的 step-level 标记，故 online 训练是不切实际的

**C3. Explicit reward modeling brings extra cost.** Training reward models require extensive annotation and broad data coverage to ensure a good balance between adaptability to the policy distribution and generalization to distribution shifts. Hence, the explicit training stage introduces a very costly data collection and an additional training overhead, especially for PRMs which typically require stepwise labels.
>  3. 显式的奖励模型带来的额外开销
>  好的奖励模型能够平衡对策略分布变化的适应和对策略分布变化的泛化，训练本身也是开销，训练好的开销就更大

Notably, a concurrent work shares similar conclusions and thus is impeded from incorporating PRMs into their large-scale RL training (DeepSeek-AI et al., 2025).

# 3 PRIME
To address the above challenges, we propose PRIME, a scalable online RL method with dense rewards. The key insight of PRIME is to apply implicit process rewards, which are derivable from the Implicit PRM that is trained with only outcome labels (Yuan et al., 2024b). This property enables us to update the PRMs online to avoid reward hacking. We then design a flexible framework to incorporate implicit process rewards with outcome rewards into any kind of MC advantage estimate. PRIME is illustrated in Figure 2 and Algorithm 1. Next, we will detail the implicit process rewards (§3.1) and how we leverage them to calculate advantages (§3.2), and introduce other techniques we used (§3.3).
>  PRIME 的关键思想是采用隐式的过程奖励，过程奖励通过 implicit PRM 给出，而 PRM 则仅仅通过输出标签训练

## 3.1 Enabling Scalable Reward Update With Implicit Reward Modeling
We consider dense rewards from the Implicit PRM because of the scalability. In short, Implicit PRM enables training an ORM with outcome labels only while repurposing it as a PRM at inference. The training stage is the same as standard ORM pipelines, with the only difference being representing the reward as $r_{\phi}(\mathbf{y})\coloneqq \beta \log \frac{\pi_{\phi}(\mathbf{y})}{\pi_{\mathrm{ref}}(\mathbf{y})}$ , where $\pi_{\phi}$ is the RM and $\pi_{\mathrm{ref}}$ is the reference model, both of which are causal LMs. At inference, the process rewards are obtained by:

$$
r_{\phi}(y_t)\coloneqq \beta \log \frac{\pi_{\phi}(y_t|\mathbf{y}_{< t})}{\pi_{\mathrm{ref}}(y_t|\mathbf{y}_{< t})} \tag{3}
$$

In PRIME, upon rollouts being generated and graded by the (ground truth) outcome verifier, we update the Implicit PRM online with on-policy rollouts and outcome supervision and then calculate token-level dense rewards to estimate advantages, which solves C1 and C2 mentioned in §2.2 respectively: (1) To prevent overoptimization and reward hacking, it is crucial to update reward models online. However, updating previous PRMs (Lightman et al., 2023) requires annotating step labels on the latest policy rollouts, which is neither efficient nor scalable during online RL. In contrast, the Implicit PRM only demands outcome labels to train due to its special reward representation, and thus it can be easily updated with policy rollouts and outcome labels or rewards, both of which have already been collected to update the policy model. (2) Unlike common PRMs that produce only step-level rewards, the Implicit PRM provides more fine-grained token-level rewards at no additional cost. This addresses the ambiguity in identifying steps in LLM responses while not introducing extra overhead, making it easy to combine with any RL algorithms for advantage estimation.

# Algorithm 1 Process Reinforcement through Implicit Rewards (PRIME)
Input Language model $\pi_{\theta_{\mathrm{init}}}$ ; outcome reward verifier $r_o$ ; dataset $\mathcal{D}$ ; sample number $K$ ; total iteration $N$ 1: Initialize policy model $\pi_{\theta}\leftarrow \pi_{\theta_{\mathrm{init}}},\pi_{\theta_{\mathrm{old}}}\leftarrow \pi_{\theta_{\mathrm{init}}}$ , implicit PRM $\pi_{\phi}\leftarrow \pi_{\theta_{\mathrm{init}}}$ , reference model $\pi_{\mathrm{ref}}\leftarrow \pi_{\theta_{\mathrm{init}}}$ 2: for iteration $= 1,\ldots ,\mathrm{N}$ do 3: Sample batch of prompts $B\sim D$ 4: Generate $K$ responses: $\{\mathbf{y}^1,\dots,\mathbf{y}^{K}\} \sim \pi_{\theta}(\cdot |\mathbf{x})$ for $\mathbf{x}\in B$ 5: Compute outcome rewards: $r_o(\mathbf{y}^{1:K})$ 6: Apply accuracy filter (8-3) on all prompts: $\mathcal{T}\gets \mathrm{Filter}(\mathbf{x},\mathbf{y}^{1:K},r_o(\mathbf{y}^{1:K}))$ for $\mathbf{x}\in B$ 7: Forward pass $\pi_{\phi},\pi_{\mathrm{ref}}$ on each $(\mathbf{x},\mathbf{y})\in \mathcal{T}$ to obatin implicit process reward $r_{\phi}(y_{tb})$ with Eq. 3 8: Update Implicit PRM $\pi_{\phi}$ by CE loss on $(\mathbf{x},\mathbf{y},r_{o}(\mathbf{y}))\in \mathcal{T}$ .. $\mathcal{L}_{\mathrm{CI}}(\phi) = -\mathbb{E}_{(\mathbf{x},\mathbf{y},r_o(\mathbf{y}))}\sim \mathcal{T}\left[r_o(\mathbf{y})\cdot \log \sigma (r_\phi (\mathbf{y})) + (1 -r_o(\mathbf{y}))\cdot \log (1 -\sigma (r_\phi (\mathbf{y})))\right]$ 9: Compute advantages $A$ with Eq. 5 10: Update policy $\pi_{\theta}$ by PPO loss in Eq. 6 11: Update old parameters: $\theta_{\mathrm{old}}\gets \theta$ 12: end for

Output Optimized policy model $\pi_{\theta}$

# 3.2 ADVANTAGE ESTIMATION AND POLICY UPDATE

Estimating advantages using Monte Carlo estimator with a leave-one-out baseline. After obtaining token-level dense rewards, we calculate advantages based on either MC estimators or GAE. To determine the advantage function in PRIME, we compare GAE with several MC estimators, including REINFORCE (Williams, 1992), RLOO (Ahmadian et al., 2024), and GRPO (Shao et al., 2024). Experimental details and results can be found in §5.4.

We find that MC estimators, despite being simpler, are strong enough to produce stable results. Therefore, we choose MC estimate as our advantage function and despite PRIME being compatible with any baseline estimation approaches, we instantiate it with a leave-one-out baseline from $K$ samples (Ahmadian et al., 2024) in this paper, as it performs better in the experiments:

$$
A^{i} = r(\mathbf{y}_{T}^{i}) -\frac{1}{K -1}\sum_{j\neq i}r(\mathbf{y}_{T}^{i}) \tag{4}
$$

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/998e9a355a5e7828fd34dddb55e09d7dfe976e03e4266a0abfe374a619795e26.jpg) 
Figure 2: Illustration of PRIME. PRIME follows that (1) initialize policy model and the Implicit PRM both with the reference model; (2) sample multiple responses for each prompt and filter with output accuracy; (3) obtain implicit process rewards by the Implicit PRM and update it using cross-entropy (CE) loss; (4) compute advantage and policy loss then update the policy model.

where $r(\mathbf{y}_T^i)$ denotes the reward of $i$ -th response at final step $T$ , $K$ is the number of samples for one prompt. The leave-one-out (LOO) baseline helps reduce variances.

More specifically, we use an Implicit PRM $\pi_{\phi}$ and an outcome verifier or reward model $r_o$ . We calculate the return of implicit process rewards and outcome rewards separately if both are available, since directly mixing their values may lead to numerical instability (Shao et al., 2024). For implicit process rewards, we perform a three-step process to calculate return: (1) Use the averaged implicit process rewards to calculate the leave-one-out baseline; (2) Normalize the process reward at step $t$ by subtracting the baseline; (3) Calculate the discounted return for each response. For outcome rewards, we directly adopt LOO without any modification. Finally, the advantage is set to the combination of

both returns:

$$
A_{t}^{i} = \underbrace{\sum_{s = t}^{|y^{i}|}\gamma^{s -t}}_{\mathrm{RLOO~with~implica~process~rewards}}\left[r_{\phi}(y_{s}^{i}) -\frac{1}{K -1}\sum_{j\neq i}r_{\phi}(\mathbf{y}^{j})\right]\underbrace{+r_{o}(\mathbf{y}^{i}) -\frac{1}{K -1}\sum_{j\neq i}r_{o}(\mathbf{y}^{j})}_{\mathrm{RLOO~with~outcome~rewards}} \tag{5}
$$

Updating policy with PPO clip surrogate loss. We adopt PPO clip surrogate loss for more stable policy updates:

$$
L_{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\left[\min \left(\frac{\pi_\theta(y_t|\mathbf{y}_{< t})}{\pi_{\theta_{\mathrm{old}}}(y_t|\mathbf{y}_{< t})} A_t,\mathrm{clip}\left(\frac{\pi_\theta(y_t|\mathbf{y}_{< t})}{\pi_{\theta_{\mathrm{old}}}(y_t|\mathbf{y}_{< t})},1 -\epsilon ,1 + \epsilon\right)A_t\right)\right] \tag{6}
$$

where $\epsilon$ is a clipping parameter. The loss prevents the updated policy from deviating too far from the original distribution, which is the prerequisite of importance sampling. The legitimacy of importance sampling then enables the reuse of rollouts sampled in previous steps, thus improving sampling efficiency.

# 3.3 OTHER TECHNIQUES

3.3 OTHER TECHNIQUESInitializing PRM with SFT/base model. In practice, we find that the starting policy model itself serves as a decent initialization of PRM, bypassing the PRM training stage. This solves C3 in §2.2 and even outperforms a dedicatedly trained PRM, as shown in § 5.1.

Online Prompt Filtering. As we sample multiple trajectories for each prompt, we introduce online prompt filtering which filters prompts within a certain accuracy range. This (1) preserves only the prompts within a certain median-level difficulty range (Yang et al., 2024b) and (2) balances data distribution for the Implicit PRM online training.

We present the ablation study results in Figure 3 using RLOO with outcome rewards only, from which we can see that the online prompt filter largely lowers the variance of RL training.

How PRIME addresses challenges in $\S 2.2$ .In summary, as illustrated in Figure 2 and Algorithm 1, PRIME adopts implicit process rewards for efficient PRM online update (C2), then inte grates token-level dense rewards with outcome rewards in MC advantage estimate (C1). The PRMs are directly initialized from SFT or base models, which foregoes explicit reward modeling (C3).

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/ec1cd1d0a4d17b2a55b1e7e436153ecfd58f3ed5eca697d1894641229944c98e.jpg) 
Figure 3: Impact of online prompt filtering on training rewards.

# 4 EXPERIMENTS

# 4.1 IMITATION WARMUP

We focus on mathematical and coding problems in this paper. For models, we start with Qwen2.5-Math-7B-Base (Yang et al., 2024b) for its great mathematical capabilities. We first performed supervised finetuning for RL preparation.

Data Construction. To construct the SFT dataset, we collect reasoning instructions from several open-source datasets. For completion, we employed LLaMA-3.1-70B-Instruct (Meta, 2024) to answer the instructions, with a system prompt requesting the model to perform action-centric chain-of-thought. We finally obtained 230K SFT data, the detailed sources and statistics can be found in § A.

SFT Results. After finetuning, the performance of our SFT model is reported in Figure 1. Compared to baselines, Eurus-2-7B-SFT lags Qwen2.5-Math-7B-Instruct on all mathematics benchmarks.

Table 2: Detailed results of PRIME and RLOO w/ outcome verifier (OV). At the same 240 steps, the model trained by PRIME is generally better than the model trained by outcome rewards. 

<table><tr><td>Method</td><td>Step</td><td>AIME 2024</td><td>AMC</td><td>MATH-500</td><td>MinervaMath</td><td>OlympiadBench</td><td>LeetCode</td><td>LiveCodeBench</td><td>Avg.</td></tr><tr><td>GPT-4o</td><td>-</td><td>9.3</td><td>45.8</td><td>76.4</td><td>36.8</td><td>43.3</td><td>58.9</td><td>48.8</td><td>45.6</td></tr><tr><td>Llama-3.1-70B-Inst.</td><td>-</td><td>20.0</td><td>37.3</td><td>65.0</td><td>37.1</td><td>30.5</td><td>35.0</td><td>34.4</td><td>37.0</td></tr><tr><td>Qwen2.5-Math-7B-Inst.</td><td>-</td><td>13.3</td><td>50.6</td><td>79.8</td><td>34.6</td><td>40.7</td><td>11.7</td><td>11.3</td><td>34.6</td></tr><tr><td>Eurus-2-7B-SFT</td><td>0</td><td>3.3</td><td>30.1</td><td>66.2</td><td>32.7</td><td>29.8</td><td>21.7</td><td>17.8</td><td>28.8</td></tr><tr><td>RLOO w/ OV Only</td><td>240</td><td>20.0</td><td>47.0</td><td>73.2</td><td>36.4</td><td>35.4</td><td>28.3</td><td>26.7</td><td>36.9</td></tr><tr><td></td><td>80</td><td>20.0</td><td>41.0</td><td>68.2</td><td>38.2</td><td>37.0</td><td>26.7</td><td>26.6</td><td>36.8</td></tr><tr><td></td><td>160</td><td>13.3</td><td>42.2</td><td>72.0</td><td>37.1</td><td>38.7</td><td>26.7</td><td>25.6</td><td>36.5</td></tr><tr><td>Eurus-2-7B-PRIME</td><td>240</td><td>20.0</td><td>50.6</td><td>78.2</td><td>39.3</td><td>40.3</td><td>31.1</td><td>27.5</td><td>41.0</td></tr><tr><td></td><td>320</td><td>16.7</td><td>51.8</td><td>77.8</td><td>39.7</td><td>41.5</td><td>36.1</td><td>28.5</td><td>41.7</td></tr><tr><td></td><td>392</td><td>26.7</td><td>57.8</td><td>79.2</td><td>38.6</td><td>42.1</td><td>33.3</td><td>28.6</td><td>43.9</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/24efbb5cf6752074048e0e43ef05b503c9635e1be35d055b9019780f9716a069.jpg) 
Figure 4: The effect of dense reward. We compare PRIME and RLOO with outcome verifier (OV). Dense rewards in PRIME lead to $2.5 \times$ sample efficiency and $6.9\%$ performance improvement. PRIME also substantially outperforms RLOO on downstream tasks.

# 4.2 RL SETTINGS

Rule-based Outcome Verifier. Consistent with recent research that adopts exact match with ground truth as unhackable rewards (Gao et al., 2024; Lambert et al., 2024; DeepSeek-AI et al., 2025), we define the rule-based ground truth outcome verifiers (OV) for math and coding as follows:

$$
r_{o}^{\mathrm{math}}(\mathbf{y}) = \left\{ \begin{array}{ll}1, & \mathrm{matched}\\ 0, & \mathrm{otherwise} \end{array} \right.r_{o}^{\mathrm{code}}(\mathbf{y}) = \frac{\sum\#\mathrm{passes}}{\sum\#\mathrm{testcases}}
$$

Hyperparameters. We use veRL (Sheng et al., 2024) to conduct experiments. By default, we initialize the Implicit PRM with SFT model and retain the SFT model for reference logprobs. For hyperparameters, we use a constant $5 \times 10^{-7}$ learning rate together with AdamW optimizer for policy model, and use a $10^{-6}$ learning rate for PRMs. Both policy and PRMs use a batch size of 256 and micro patchsize of 8. The rollout stage collects 256 prompts and samples 4 responses for each prompt. We set $\beta = 0.05$ for PRM training. We set KL coefficient to 0 in all experiments.

Evaluation Benchmarks. We evaluate on 7 reasoning benchmarks, focusing on competition-level mathematics and programming tasks, including AIME 2024 (Li et al., 2024), AMC (Li et al., 2024), MATH-500 (Hendrycks et al., 2021b), Minerva Math (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), LeetCode (Guo et al., 2024), and LiveCodeBench (v2) (Jain et al., 2024).

# 4.3 MAIN RESULTS

As shown in Figure 1 and Table 2, Eurus-2-7B-PRIME achieves substantial improvements on key reasoning benchmarks over the SFT version of the model, leading to $15.1\%$ improvement on average, and over $20\%$ on AMC and AIME competitions. Besides, Eurus-2-7B-PRIME achieves $26.7\%$ pass@1 on AIME 2024, surpassing GPT-4o, Llama-3.1-70B-Instruct, and Qwen2.5-Math-7B-Instruct, demonstrating its excellent reasoning ability.

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/7b9b725c4c4933decab5fd54d081d02cfae559aa4105730fce4022f2bbc18c2f.jpg) 
Figure 5: Comparison of different PRMs. Online PRM initialized from SFT model achieved the best results. Surprisingly, using PRMs trained on extra rollouts hurts the performance in both online and offline settings.

# 4.4 DENSE REWARDS V.S. SPARSE REWARDS

We first validate the effect of dense rewards compared to RLOO with outcome rewards only. We train this model for 240 steps. For PRIME, we use the same setting and train the model for 592 steps. We plot the training rewards measured by the outcome verifier and test accuracy in Figure 4. Compared with sparse reward, PRIME takes $40\%$ of the training steps to achieve the same training rewards as RLOO and improves the final rewards by $6.9\%$ , with lower variances. On downstream tasks, PRIME also consistently outperforms OV only setup. Detailed results are listed in Table 2.

# 5 ANALYSIS

# 5.1 DESIGN CHOICES FOR THE IMPLICIT PRM

The Implicit PRM is the key component of PRIME, and its design choices greatly affect RL. In this section, we explore two major factors: (1) the initialization model and (2) the update mechanism.

SFT model initializes a good PRM. Conventionally, we need to collect data to train RMs and PRMs, and then we can use them in RL. However, the Implicit PRM is a language model, so we can initialize it from any language model with the same tokenizer as the policy model. To investigate whether it is still necessary to train a PRM in advance, we conduct experiments with different PRM initialization strategies: with the SFT model itself and with a specially trained PRM. For the later one, we train EurusPRM from Eurus-2-7B-SFT with additional 500K data generated by Llama3.1 and Qwen2.5 series (data details in § B.5).

We report the experiment results in Figure 5. Surprisingly, directly using Eurus-2-7B-SFT to initialize the PRM greatly outperforms EurusPRM which was trained on more samples. We conjecture that initializing policy model and PRM from the same model largely alleviates the distribution shift issue, as the PRM is only trained on the online rollouts from the policy model.

Online PRM update is essential. To verify the effect of online PRM update, we pair the correct and wrong samples and calculate the PRM prediction accuracy using $r_{\phi}(y)$ . We report the PRM classification accuracy in Figure 6. The figure clearly shows that, online update mitigates overoptimization and reward

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/738f4e7cf2e1156c5772bfbd30a788c7e0af593cf771b8dd5a172e2437da303d.jpg) 
Figure 6: Impact of PRM online update. The offline PRM is gradually been overoptimized while online PRMs achieve higher accuracy throughout training.

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/a4a0a842fea12c622f59f87c900f4c24437daa5fdf94cc6d0bc099ed9352eda0.jpg) 
(a) Policy ref: We use the policy logprob as $\pi_{\mathrm{ref}}$ (b) SFT ref: We retain the initial policy to provide $\pi_{\mathrm{ref}}$ for for PRM. PRM and KL.

Figure 7: Comparison of different reference policy implementations. One uses the running policy's old logprobs as reference (policy ref) while the other uses the initial SFT model as the reference model (SFT ref).

hacking. The offline PRM, though starting with high accuracy, gradually drops during RL training procedure due to distribution shift. In contrast, online PRMs that are trained on policy rollouts show the reverse curve.

This is further validated with training rewards and downstream performance. To breakdown, Eurus-2-7B-SFT is both used as PRM initialization and the reference model in the main experiment, so the PRM is totally trained from scratch, which means the initial PRM outputs zero reward for all tokens. Therefore, Figure 4 also demonstrates the effect of online PRM update. For EurusPRM initialization, the online run outperforms the offline run as well in Figure 5.

# 5.2 REFERENCE MODEL CHOICE IS FLEXIBLE

We implement two variants of our algorithms to explore the effect of reference model of implicit PRM, one using the initial SFT model as the reference model (SFT ref) while the other using the running policy's old logprobs as reference (policy ref), as shown in Figure 7a. The policy ref simply adopts the old logprob of the policy model as $\pi_{\mathrm{ref}}$ while the SFT ref remains the initial SFT model for an additional $\pi_{\mathrm{ref}}$ calculation. We compare their performance in this section.

From the training rewards in Figure 8, we find the two strategies are close and have pros and cons in different aspects: The Q value calculated by implicit PRM is the expectation under the distribution of the reference model. So the updating policy could naturally serve as the reference. On the other hand, KL divergence calculation is only allowed when the initial SFT model is retained.

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/5bc4dcad7156674f1581566050c04bb257b44ba6b5232f819cf0d57905e7f8bc.jpg) 
Figure 8: Different reference model for PRM. We compare two reference model selection strategies for PRIME. Using the policy model as reference and using the initial SFT model as reference. Their rewards are similar.

# 5.3 SINGLE-FORWARD V.S. DOUBLE-FORWARD

Since our implicit PRM is concurrently updated in training, for each rollout stage, we can update the PRM before the policy model and use the updated PRM to re-calculate the process rewards, which

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/57949b2fd805246d2c44bdb6ef05f1493cae0976a6d322d16d702cf3f73ae96b.jpg) 
Figure 9: Single and double forward. While double forward methods obtain higher accuracy after online update, the two variants achieve similar rewards during training.

we call the double-forward setting. We investigate the impact of double-forward in both the training and test phases. Our default setting applies single-forward, which uses process rewards from old PRMs. We plot PRM accuracy on rollouts and training rewards in Figure 9.

Accordingly, we find that double-forward could increase PRM accuracy, but the training rewards remain close between the two methods.

# 5.4 PRIME WITH OTHER RL ALGORITHMS

As we stated before, PRIME is equally applicable to other RL algorithms beyond RLOO. In this section, we implement PRIME with REINFORCE (Williams, 1992), GRPO (Shao et al., 2024), and PPO (Schulman et al., 2017). Similarly to RLOO, we only modify the advantage estimation functions and leave the clip surrogate loss unchanged.

First of all, We compare different REINFORCE-like advantage estimators including REINFORCE, GRPO, and RLOO, toggling the existence of implicit process reward. To make different algorithms compatible with the compound of outcome verifier reward and process reward, we accordingly make adaptions similar to Eq. 5. For GRPO, we have

$$
A_{t}^{i} = \underbrace{\underbrace{r_{o}(\mathbf{y}^{i}) -\mathrm{mean}(r_{o}(\mathbf{y}^{j}))}_{\mathrm{GRPO~with~outcome~rewards}}}_{\mathrm{GRPO~with~outcome~rewards}} + \underbrace{\underbrace{\frac{|\mathbf{y}^{i}|}{s = t}\cdot[\frac{r_{\phi}(y_{s}^{i}) -\mathrm{mean}(\frac{r_{\phi}(\mathbf{y}^{j})}{|\mathbf{y}^{j}|})}_{\mathrm{GRPO~with~implicit~process~rewards}}}_{\mathrm{GRPO~with~implicit~process~rewards}}.} \tag{7}
$$

For REINFORCE, we have

$$
A_{t}^{i} = \underbrace{\underbrace{r_{o}\left(\mathbf{y}^{i}\right)}_{\mathrm{REINFORCE~with~outcome~rewards}}}_{\mathrm{REINFORCE~with~outcome~rewards}} + \underbrace{\underbrace{\sum_{s = t}^{\left|\mathbf{y}^{i}\right|}}_{s = t}s^{s = t}\cdot r_{\phi}(y_{s}^{i})}_{\mathrm{REINFORCE~with~implicit~process~rewards}}. \tag{8}
$$

From Figure 10 and Table 3, We show that PRIME boosts these algorithms on both efficiency and performance as it does with RLOO. PRIME contributes consistently regardless of the policy update method, making it a generic algorithm. It indicates that PRIME is a general plug-in for almost any RL algorithm for LLM, which largely extends the use cases of PRIME.

Moreover, the PPO variant of PRIME provides no performance gain, demonstrating that the additional computation cost from the critic model is redundant. This makes it possible to compensate for the expense of the process reward model by using REINFORCE-like algorithms with simpler advantage estimators. Finally, we choose the best-performing RLOO as the advantage estimator in our algorithm.

Table 3: Testset results of different RL algorithms. 

<table><tr><td>Method</td><td>Step</td><td>AIME 2024</td><td>AMC</td><td>MATH-500</td><td>MinervaMath</td><td>OlympiadBench</td><td>LeetCode</td><td>LiveCodeBench</td><td>Avg.</td></tr><tr><td>RLOO</td><td>240</td><td>20.0</td><td>47.0</td><td>73.2</td><td>36.4</td><td>35.4</td><td>28.3</td><td>26.7</td><td>36.9</td></tr><tr><td>RLOO w/ PRIME</td><td>240</td><td>20.0</td><td>47.0</td><td>73.2</td><td>39.3</td><td>40.3</td><td>31.1</td><td>27.5</td><td>41.0</td></tr><tr><td>REINFORCE</td><td>240</td><td>6.7</td><td>50.6</td><td>72.6</td><td>36.0</td><td>37.2</td><td>27.2</td><td>27.0</td><td>36.0</td></tr><tr><td>REINFORCE w/ PRIME</td><td>240</td><td>6.7</td><td>50.0</td><td>76.4</td><td>36.8</td><td>39.1</td><td>27.8</td><td>27.5</td><td>37.8</td></tr><tr><td>GRPO</td><td>240</td><td>10.0</td><td>44.6</td><td>73.2</td><td>37.5</td><td>36.6</td><td>25.0</td><td>22.8</td><td>36.1</td></tr><tr><td>GRPO w/ PRIME</td><td>240</td><td>16.7</td><td>47.0</td><td>75.0</td><td>34.9</td><td>38.2</td><td>28.9</td><td>25.9</td><td>37.8</td></tr><tr><td>PPO</td><td>240</td><td>10.0</td><td>41.0</td><td>73.6</td><td>36.0</td><td>36.3</td><td>28.3</td><td>25.7</td><td>35.8</td></tr><tr><td>PRIME as Value Model</td><td>240</td><td>16.7</td><td>44.6</td><td>72.6</td><td>34.6</td><td>35.7</td><td>27.8</td><td>24.6</td><td>36.6</td></tr><tr><td>PPO w/ PRIME</td><td>240</td><td>13.3</td><td>50.6</td><td>77.4</td><td>37.1</td><td>40.6</td><td>30.0</td><td>26.7</td><td>39.4</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/8cce4510f570b10dac814a8516887341b7b70844bb61da94d1a1d32374a08885.jpg) 
Figure 10: PRIME also benefits REINFORCE, GRPO, and PPO, achieving similar improvement as RLOO.

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/83db3cefc3de23df53db7dc8978abee616c84065db283803f418692bcfdb3e30.jpg) 
Figure 11: Comparison of value models and reward models. We show that value models, either the original PPO one or Implicit PRM, is substantially worse than reward models.

# 5.5 VALUE OR REWARD, HOW TO USE THE IMPLICIT PRM?

Besides using process rewards to estimate returns, we can also employ the Implicit PRM to predict values for advantage estimation in Eq. 2. Therefore, we compare four variants of MC estimate to determine the best way to incorporate dense supervision. Recall that the Implicit PRM has $\begin{array}{r}v_{\phi}(\mathbf{y}_{< t + 1}) = \sum_{i = 1}^{t}\beta \log \frac{\pi_{\phi}(y_{i}|\mathbf{y}_{< i})}{\pi_{\mathrm{ref}}(y_{i}|\mathbf{y}_{< i})} \end{array}$ with the process reward being $r_{\phi}(y_t) = v_{\phi}(\mathbf{y}_{< t + 1}) -v_{\phi}(\mathbf{y}_{< t})$ and we assume a ground-truth outcome verifier $r_{o}$ $\gamma = 1$ , then we represent the variants as follows:

(1) REINFORCE: $A_{t} = r_{o}(\mathbf{y})$

(2) On top of 
(1), using a linear-head value model $V$ to calculate the baseline: $A_{t} = r_{o}(\mathbf{y}) -V(\mathbf{y}_{< t})$ This is the original PPO in Figure 10 as we set $\gamma = 1$ and $\lambda = 1$

(3) On top of 
(1), using values from the Implicit PRM to serve as the baseline: $A_{t} = r_{o}(\mathbf{y}) -v_{\phi}(\mathbf{y}_{< t})$ . This is equivalent to PPO with its value model being replaced by values from the Implicit PRM when $\gamma = 1$ and $\lambda = 1$

(4) On top of 
(1), using process rewards from the Implicit PRM to calculate the return: $A_{t} = r_{o}(\mathbf{y}) + \sum_{s = t}^{T}r_{\phi}(y_{s})$ . This is the REINFORCE w/ PRIME in Figure 10.

Figure 11 reports the results. Comparing PPO and REINFORCE, we find that an additional value model does not benefit policy performance. Notably, using rewards from the Implicit PRM to calculate returns, which is the default setting in PRIME, greatly outperforms all three baselines, regardless of where the values come from. This indicates that PRMs work better than value models in RL for LLMs.

# 5.6 "ZERO" EXPERIMENTS

DeepSeek-AI et al. (2025) proposed DeepSeek-R1-Zero, which is directly trained from a base model with reinforcement learning. To further investigate the "Zero" setting, we also perform RL from

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/af3418985c594cbb2e19a4c4bcbeb646aae44aa0ac28e15a1e20ac045246a32b.jpg) 
Figure 12: "Zero" RL from Qwen2.5-Math-7B. RL from the base model converges way faster than the SFT model, surpassing the instruct version within 32 steps.

![](https://cdn-mineru.openxlab.org.cn/extract/fa0b3afc-ef0b-410f-9edd-282de09dbbfd/a64a2b56834cae1e43e2e7339c13ca3e69883e3c3ca641a7f01dfd232554ba72.jpg) 
Figure 13: "Zero" RL from Qwen2.5-32B-Base. RL from a 32B base model shows more promising gain, surpassing the instruct version within 16 steps.

Qwen2.5-Math-7B-Base and Qwen2.5-32B-Base (Yang et al., 2024a), skipping the SFT phase. We present the experimental results in Figure 12 and Figure 13. The observations are as follows:

(1) RL from base model is surprisingly efficient and effective. Comparing PRIME from Qwen2.5-Math-7B and Eurus-2-7B-SFT, the "Zero" setting converges much faster. This indicates that directly performing RL from a base model might be a strong alternative to the conventional SFT-RL pipeline. 
(2) Larger models benefit more. Comparing 7B and 32B models, we see that the 32B model gains more on both training rewards and test performance. This is aligned with the conclusion in DeepSeek-AI et al. (2025). 
(3) Saturation could be a potential issue. Although PRIME-Zero obtains impressive performance gain, we find it quickly saturated at a very early stage (about 50 steps), which hinders further improvement like in DeepSeek-AI et al. (2025). This is possibly attributed to the decrease of response diversity, and we leave this as future work.

# 6 RELATED WORK

RL for LLM Reasoning. In the context of LLMs, reinforcement learning has been widely used for aligning human preferences (Christiano et al., 2017; Ouyang et al., 2022; Cui et al., 2024), but the open-source community mostly adopt the data-driven imitation learning methods (Yuan et al., 2024a; Yue et al., 2024; Wei et al., 2024; Liu et al., 2024) to enhance the reasoning capabilities of LLMs. Over the past few months, the paradigm gradually shifted. OpenAI o1 (Jaech et al., 2024) first showed the tremendous potential of large-sacle RL for reasoning LLMs, and recent works have verified the scaling effect of the simple RL recipe with merely outcome rewards (DeepSeek-AI et al., 2025; Team

et al., 2025). Meanwhile, the role of dense rewards in RL remains underexplored, which is the main focus of PRIME.

Implicit Rewards. Implicit rewards are broadly adopted in LLM alignment (Rafailov et al., 2023; Chen et al., 2024b; Azar et al., 2024; Ethayarajh et al., 2024; Rosset et al., 2024; Chen et al., 2024a). Rafailov et al. (2024) first showed that optimizing DPO objective learns a Q function implicitly. Zhou et al. (2024) utilized implicit rewards in PPO, and showed that dense implicit rewards are better than sparse ones. Yuan et al. (2024b) further extended the conclusion to any loss function optimizing Eq. 3.

# 7 CONCLUSION

As the fuel of LLMs, data, will be depleted in the near future, we are entering a new era of search and exploration, which is exemplified by reinforcement learning (Sutton, 2019). This work develops PRIME, which produces and leverages dense rewards in online RL for LLM reasoning. Throughout the experiments, we validate that PRIME (1) greatly benefits sample efficiency and policy performance, (2) is easy to use with minimum cost, and (3) is a general method that works with broad RL algorithms together.

# REFERENCES

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human preferences. International Conference on Artificial Intelligence and Statistics, abs/2310.12036, 2024.

Changyu Chen, Zichen Liu, Chao Du, Tianyu Pang, Qian Liu, Arunesh Sinha, Pradeep Varakantham, and Min Lin. Bootstrapping language models with dpo implicit rewards. arXiv preprint arXiv:2406.09760, 2024a.

Huayu Chen, Guande He, Lifan Yuan, Ganqu Cui, Hang Su, and Jun Zhu. Noise contrastive alignment of language models with explicit rewards. arXiv preprint arXiv:2402.05369, 2024b.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with scaled ai feedback. In ICML, 2024.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng

Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaower Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Qu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zingang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zhih Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-rl: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948. Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. ICML, 2024. Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weiling Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, and Yi Wu. On designing effective rl reward at training time for llm reasoning. ArXiv, abs/2410.15115, 2024. Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, 2022. Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming-the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024. Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828-3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. acl-long.211. URL https://aclanthology.org/2024. acl-long.211/. Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021a. Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021b. Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai-ol system card. arXiv preprint arXiv:2412.16720, 2024. Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024. Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment. arXiv preprint arXiv:2410.01679, 2024. Wouter Koel, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free! In DeepRLStructPred@ICLR, 2019. URL https://api.semanticscholar.org/CorpusID:198489118.

Nathan Lambert, Jacob Daniel Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tatjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hanna Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. ArXiv, abs/2411.15124, 2024.

Jan Leike, David Krueger, Tom Everitt, Miljan Martio, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871, 2018.

Aitor Lewkowycz, Anders Andreassen, David Dothan, Ethan Dyer, Henryk Michalewski, Vinay Ramesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843-3857, 2022.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4math with 860k pairs of competition math problems and solutions. Hugging Face repository, 13:9, 2024.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Remi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson F. Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. arXiv preprint arXiv:2203.07814, 2022.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. ArXiv, abs/2303.20050, 2023.

Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Advancing frontier math reasoning with post-training and reward modeling. arXiv preprint arXiv:2412.15084, 2024.

Meta. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

OpenAI. Openai o1 system card. ArXiv, abs/2412.16720, 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744, 2022.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2023.

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From $r$ to $q^*$ : Your language model is secretly a q-function. arXiv preprint arXiv:2404.12358, 2024.

Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, and Tengyang Xie. Direct nash optimization: Teaching language models to self-improve with general preferences. ArXiv, abs/2404.03715, 2024.

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Amrith Sellur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024.

SkunkworksAI. reasoning-0.01, 2024.

Richard Sutton. The bitter lesson. Incomplete Ideas (blog), 13(1):38, 2019.

Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3: 9-44, 1988.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenthuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL https://qwenlm.qithub.io/blog/qwq-32b-preview/.

Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560, 2024.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Y.Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. ArXiv, abs/2312.08935, 2023.

Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering code generation with oss-instruct. In Forty-first International Conference on Machine Learning, 2024.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229-256, 1992.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jianxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024a.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement, 2024b. URL https://arxiv.org/abs/2409.12122.

Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and Maosong Sun. Advancing Ilm reasoning generalists with preference trees. ArXiv, 2024a.Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou, Zhiyuan Liu, and Hao Peng. Free process rewards without process labels, 2024b. URL https://arxiv.org/abs/2412.01981. Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023. Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhua Chen. Mammoth2: Scaling instructions from the web. ArXiv, abs/2405.03548, 2024. Kaiyan Zhang, Sihang Zeng, Ermo Hua, Ning Ding, Zhang-Ren Chen, Zhiyuan Ma, Haoxin Li, Ganqu Cui, Biqing Qi, Xuekai Zhu, Xingtai Lv, Hu Jinfang, Zhiyuan Liu, and Bowen Zhou. Ultramedical: Building specialized generalists in biomedicine, 2024. Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue. Opencodeinterpreter: Integrating code generation with execution and refinement. arXiv preprint arXiv:2402.14658, 2024. Zhanhui Zhou, Zhixuan Liu, Jie Liu, Zhichen Dong, Chao Yang, and Yu Qiao. Weak-to-strong search: Align large language models via searching over small language models. arXiv preprint arXiv:2405.19262, 2024.

Table 4: Actions in action-centric chain-of-thought reasoning framework. 

<table><tr><td>Action Name</td><td>Description</td></tr><tr><td>ASSESS</td><td>Analyze current situation, identify key elements and goals</td></tr><tr><td>ADVANCE</td><td>Move forward with reasoning -calculate, conclude, or form hypothesis</td></tr><tr><td>VERIFY</td><td>Check accuracy of current approach, look for errors</td></tr><tr><td>SIMPLIFY</td><td>Break complex problems into simpler parts</td></tr><tr><td>SYNTHESIZE</td><td>Combine multiple pieces of information into complete solution</td></tr><tr><td>PIVOT</td><td>Change strategy when current approach isn&#x27;t working</td></tr><tr><td>OUTPUT</td><td>Summarize thought process and present final answer</td></tr></table>

Table 5:Data statistics of SFT data. 

<table><tr><td>Task</td><td>Dataset</td><td>Size</td><td>Avg. Response Length</td><td>Source</td></tr><tr><td rowspan="4">Math</td><td>MathInstruct-MATH (Yue et al., 2023)</td><td>12715</td><td>964.01</td><td>https://huggingface.co/datasets/TIGER-Lab/MathInstruct</td></tr><tr><td>OpenMathIns-2-Aug-Math (Toshniwal et al., 2024)</td><td>15086</td><td>1202.25</td><td>https://huggingface.co/datasets/nvidia/OpenMathInstruct-2</td></tr><tr><td>Nichina (Li et al., 2024)</td><td>5584</td><td>131.61</td><td>https://huggingface.co/datasets/AI-MO/NunimaMathIns-OT</td></tr><tr><td>Reasoning-001 (Skunkworks, AI, 2024)</td><td>29831</td><td>116.49</td><td>https://huggingface.co/datasets/Slunkworks,Al/reasoning-0.01</td></tr><tr><td rowspan="3">Coding</td><td>Code-Feedback (Zheng et al., 2024)</td><td>805.16</td><td>828.16</td><td>https://huggingface.co/datasets/m-pvCode-Feedback</td></tr><tr><td>Magiccoder (Wei et al., 2024)</td><td>24480</td><td>1828.72</td><td>https://huggingface.co/datasets/ise-iiuc/Magiccoder-Evol-Instruct-110K</td></tr><tr><td>Magiccoder-OSS (Wei et al., 2024)</td><td>28980</td><td>1850.05</td><td>https://huggingface.co/datasets/ise-iiuc/Magiccoder-OSS-Instruct-75K</td></tr><tr><td>Biomedicine</td><td>UltraMedical,mc (Zhang et al., 2024)</td><td>35163</td><td>1891.06</td><td>https://huggingface.co/datasets/TinghuaC3/IUltraMedical</td></tr><tr><td>Total / Avg.</td><td>-</td><td>229763</td><td>1390.75</td><td>-</td></tr></table>

# A SFT DATA & TRAINING DETAILS

We first perform supervised finetuning on the base model to get a starter model for RL.

Action-centric chain-of-thought reasoning. We apply imitation learning (supervised finetuning) as a warmup stage to teach models to learn certain reasoning patterns. To this end, we first design an action-centric chain-of-thought reasoning framework. Table 4 shows the actions in the action-centric chain-of-thought reasoning framework. When the model generates answers, it conducts multi-step reasoning and chooses one of the 7 actions at each step. The response begins with the ASSESS action and ends with the OUTPUT action.

Construction of the SFT dataset. To construct the SFT dataset, we collect reasoning instructions from several open-source datasets. It is noteworthy that we did not include many datasets with ground-truth answers in SFT, even though they are of higher quality. However, we reserve them for later RL training. The reason is that we aim to use different datasets for SFT and RL to diversify the exploration in RL, and we consider ground-truth more essential in RL than in SFT. For completion, we employ LLaMA-3.1-70B-Instruct to answer the instructions, with a system prompt requesting the model to perform an action-centric chain-of-thought. Table 5 summarizes the key statistics of the datasets used for SFT. The datasets span mathematics, coding, and biomedicine. We finally obtain 230K SFT data and the average response length is 1390 tokens.

SFT Training. During the SFT phase, we conduct full parameter fine-tuning with a learning rate of 1e-05, utilizing the AdamW optimizer alongside a cosine annealing learning rate schedule and a warmup ratio of 0.1. The batch size was set to 96, with a fixed random seed of 42. The model was trained on 230K datasets for 3 epochs.

# B RL DATA PREPROCESSING

# B.1 RL DATA COLLECTION AND PREPROCESSING

We curate a high-quality RL training dataset of mathematics and coding problems with outcome verifiers (LaTeX answers for math and test cases for coding). For math, we source from NuminaMathCoT (Li et al., 2024), which contains about 860K math problems. The problems span from Chinese high school mathematics to International Mathematical Olympiad competition questions. For coding, we source from APPS (Hendrycks et al., 2021a), CodeContests (Li et al., 2022), TACO (Li et al., 2023), and Codeforces<sup>2</sup>. To further increase data quality, we conduct detailed cleaning and filtering. Finally, we retain 457k math problems and 27k coding problems.

# B.2 DATA FILTERING AND QUESTION-TYPE CLASSIFICATION

B.2 DATA FILTERING AND QUESTION-TYPE CLASSIFICATIONThe preprocessing pipeline employs a systematic rule-based approach to filter and classify mathematical problems to create a high-quality dataset with solvable problems, appropriate difficulty levels, and correct solutions. We exclude problems containing figures or diagrams since they require visual processing capabilities. We also remove proof questions due to difficulties in answer verification. Based on specific patterns, the remaining problems are classified into question-answering, multiple-choice, or fill-in-the-blank questions. Since fill-in-the-blank questions comprise less than 400 examples compared to the much larger set of multiple-choice questions, we focus solely on multiple-choice questions for further processing.

# B.3 CONVERTING TO DIRECT QUESTION-ANSWER FORMAT

We transform multiple-choice questions into a direct question-answer format through three sequential stages: rule-based filtering, LLM-based filtering, and LLM-based formatting.

We first identify and remove questions that inherently require multiple-choice options -specifically, those where comparing specific statements or properties is essential to the problem-solving process. These questions cannot be meaningfully converted to a direct question-answer format. The initial filtering employs simple rule-based pattern matching, searching for keywords like "following" and "statement" that typically indicate option-dependent problems.

Following the rule-based filtering, we employ Llama-3.1-8B-Instruct to perform a more nuanced classification of the remaining questions. Our pilot study revealed that while the LLM occasionally misclassifies questions, it tends to err on the conservative side -marking potentially convertible questions as requiring options rather than the reverse. Given our large dataset, we accepted this conservative approach to maintain quality.

For questions classified as convertible, we implement a two-phase reformatting process: 1) Question Reformatting: Removing choice indicators and restructuring the question to elicit direct answers. 2) Solution Reformatting: Converting multiple-choice solutions into step-by-step derivations, ensuring all final answers are presented in standard LaTeX boxed format. This systematic approach maintains mathematical rigor while creating a standardized format suitable for downstream applications.

# B.4 PROBLEM AND SOLUTION VALIDATION

The final stage involves merging all question-answer pairs and performing LLM-based comprehensive validation. We identify two key aspects in validation: solvability and correctness.

We leverage state-of-the-art mathematical reasoning models, including QwQ-32B-Preview (Team, 2024) and Qwen2.5-Math-72B-Instruct (Yang et al., 2024b), employing a self-consistency approach to determine problem solvability, and if solvable, verify the correctness of solutions provided in the original dataset.

To enhance validation accuracy, we first analyzed sample problems to identify characteristics of solvable and unsolvable cases and created synthetic unsolvable problems featuring missing conditions or logical contradictions. Based on these samples, we developed specialized prompts to improve the models' ability to distinguish solvability. Each problem undergoes five independent validation attempts, where the LLM: 1) Provides step-by-step solutions using LaTeX formatting. 2) Identifies unsolvability due to missing conditions or logical contradictions. 3) Generates complete reasoning traces for solvable problems. 4) Presents final answers in standardized LaTeX boxed format $(\backslash \mathrm{boxed}\{\ldots \})$ . 5) Document any impediments to solution completion.

We evaluate two key consistency measures across multiple validation attempts: 1) Status Consistency: agreement on problem solvability. 2) Answer Consistency: consistency of solutions across different attempts and agreement between generated solutions and ground truth. The final dataset retains only problems that demonstrate consistent solvability across validation attempts, agreement in solutions across multiple attempts, and alignment with ground truth answers. This rigorous validation process ensures the resulting dataset comprises well-defined, solvable problems with verified, accurate solutions.

Table 6: Data statistics of EurusPRM training dataset. 

<table><tr><td>Dataset</td><td>Generator Model</td><td>Num. Inst</td><td>Resp/Inst</td><td>Step-level/Response-level</td></tr><tr><td rowspan="4">UltraInteract</td><td>Llama-3.1-8B-Inst</td><td>20177</td><td>8</td><td>Response-level</td></tr><tr><td>Llama-3.1-8B-Base</td><td>13570</td><td>8</td><td>Response-level</td></tr><tr><td>Qwen2.5-72B-Inst</td><td>4758</td><td>8</td><td>Response-level</td></tr><tr><td>Qwen2.5-Math-7B-Base</td><td>25713</td><td>8</td><td>Response-level</td></tr><tr><td rowspan="2">Numina-SynMath</td><td>Llama-3.1-8B-Inst</td><td>4783</td><td>8</td><td>Response-level</td></tr><tr><td>Qwen2.5-Math-7B-Base</td><td>5806</td><td>8</td><td>Response-level</td></tr><tr><td rowspan="2">Numina-Olympiads</td><td>Llama-3.1-8B-Inst</td><td>2909</td><td>8</td><td>Response-level</td></tr><tr><td>Qwen2.5-Math-7B-Base</td><td>4739</td><td>8</td><td>Response-level</td></tr></table>

# B.5 PRM DATA

The dataset statistics of training EurusPRM are shown in Table 6.