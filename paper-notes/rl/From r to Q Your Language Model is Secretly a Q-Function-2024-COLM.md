# Abstract
Reinforcement Learning From Human Feedback (RLHF) has been critical to the success of the latest generation of generative AI models. In response to the complex nature of the classical RLHF pipeline, direct alignment algorithms such as Direct Preference Optimization (DPO) have emerged as an alternative approach. 

Although DPO solves the same objective as the standard RLHF setup, there is a mismatch between the two approaches. Standard RLHF deploys reinforcement learning in a specific token-level MDP, while DPO is derived as a bandit problem in which the whole response of the model is treated as a single arm.
>  虽然 DPO 和标准的 RLHF 求解的是相同的目标，但二者之间存在根本性的理论差异: 标准 RLHF 在 token-level 上的 MDP 上执行 RL，而 DPO 则可以视作一个多臂赌博机问题，其中模型的整个响应视作一个 “臂”，这简化了问题，但失去了 token 生成的细粒度、顺序性特征，使得模型或许无法理解序列生成中的因果性

In this work we rectify this difference. We theoretically show that we can derive DPO in the token-level MDP as a general inverse Q-learning algorithm, which satisfies the Bellman equation. Using our theoretical results, we provide three concrete empirical insights. First, we show that because of its token-level interpretation, DPO is able to perform some type of credit assignment. Next, we prove that under the token level formulation, classical search-based algorithms, such as MCTS, which have recently been applied to the language generation space, are equivalent to likelihood-based search on a DPO policy. Empirically we show that a simple beam search yields meaningful improvement over the base DPO policy. Finally, we show how the choice of reference policy causes implicit rewards to decline during training. 
>  本文意在解决这一差异，本文在理论上证明了 DPO 可以在 token-level 的 MDP 中被推导为广义的逆向 Q-learning 算法
>  本文在理论结果下提供了三个具体的经验性观点:
>  - 由于 DPO 可以被解释为 token-level，故可以执行某种类型的信用分配
>  - 在 token-level 的解释下，经典的基于搜索的算法，例如 MCTS ，等价于在 DPO 策略上基于似然的搜索
>  - 参考策略的选择对于训练时隐式奖励的下降有影响

We conclude by discussing applications of our work, including information elicitation in multi-turn dialogue, reasoning, agentic applications and end-to-end training of multi-model systems.

# 1 Introduction
Reinforcement Learning from Human Feedback (RLHF) has become the defacto method for aligning large language models (LLMs) with human intent due to its success in a wide range of applications from summarization (Shiennon et al., 2022) to instruction following (Ouyang et al., 2022). By learning a reward function from human-labeled comparisons, RLHF is able to capture complex objectives that are in-describable in practice. Following the success of (Ziegler et al., 2020), numerous works have considered new algorithms for training and sampling from large models in various domains using techniques from reinforcement learning (RL). 

In particular direct alignment methods, such as Direct Preference Optimization (DPO) (Rafailov et al., 2023) have gained traction in recent months because of their simplicity (Zhao et al., 2023a; Azar et al., 2023). Instead of learning a reward function and then using RL, direct alignment methods use the relationship between reward functions and policies in the contextual bandit setting to optimize both simultaneously. Similar ideas have since been applied to vision language (Zhao et al., 2023b) and image generation models (Lee et al., 2023).

While such direct alignment methods are purported to work the same as classical RLHF approaches that use policy gradient algorithms like PPO (Schulman et al., 2017), fundamental differences remain. For instance, classical RLHF methods optimize token-level value functions with a sparse reward at the terminal state. DPO on the other hand, operates only in a contextual bandits setting, treating the entire response as a single arm. 
>  虽然 DPO 这样的直接对齐方法声称和使用策略梯度算法的经典 RLHF 方法的工作方式相同，但二者实际存在根本差异
>  例如，经典的 RLHF 方法是使用仅在终止状态给出奖励的稀疏奖励函数，优化 token-level 的价值函数；DPO 则仅在上下文赌博机的设置中运行 (模型根据上下文，从一组预定义的 “臂” 中进行选择，并为选择的 “臂” 获得奖励) ，整个响应视作一个 “臂”  (模型根据完整的文本进行优化，而不是文本中的单个 token)

This is despite the fact that tokens are generated one at a time, and dense rewards are commonly known to be beneficial in the RL community. While direct alignment algorithms are interesting, at present it is unclear if they can be applied to sequences in the same way as the underlying RL algorithms used in typical RLHF pipelines.
>  DPO 的优化设定忽略了响应实际上是一个 token 接着一个 token 生成的，而众所周知，密集的奖励有利于模型的学习
>  因此，目前尚不清楚直接对齐算法能否有效处理语言生成中的序列性

>  然而 RLHF 不也是用的稀疏奖励吗？

In this work we rectify this difference by deriving DPO within the token-level MDP setting present in large language models using the usual form of binary preference-feedback. We then show that DPO training implicitly learns a token-level reward function, for which the language models logits define the optimal $Q$ function, or expected total future reward. We then demonstrate that DPO is able to flexibly model any possible dense reward function within the token MDP.
>  本文使用二元偏好反馈的常见设定，在 token-level MDP 的设定下推导 DPO，进而证明 DPO 训练会隐式地学习一个 token-level 奖励函数
>  对于该奖励函数，语言模型的 logits 定义了最优的 $Q$ function (即期望的总未来奖励)
>  我们进而证明 DPO 在 token MDP 下，可以灵活地建模所有可能的密集奖励函数

Empirically, we use our theoretical derivations to justify three practical insights which we believe to be of use to the community. First, we show that despite being derived as a contextual bandit, the implicit rewards of a DPO model have a per-token interpretation. Second, we demonstrate that likelihood search over a DPO model is analogous to searching over a reward function during decoding as done by contemporary works (Liu et al., 2023b; Feng et al., 2024). Finally, we identify the choice of initial policy and reference distribution as being important in determining the trajectory of implicit rewards during training.
>  基于我们的理论推导，我们提出三项实践性见解:
>  1. 尽管 DPO 没有明确被设计为提供 token 级别的奖励，但 DPO 可以解释为提供了 token 级别的隐式奖励
>  2. 对 DPO 模型在 decoding 时进行似然搜索 (从 DPO 训练的模型执行类似 beam search 的 decoding 策略) 类似于在 decoding 时搜索奖励函数
>  3. 我们确认了初始策略和参考分布的选择对于决定训练时的隐式奖励轨迹非常重要

# 2 Related Work
The problem of aligning policies with human intent using preference feedback has been a long studied problem in reinforcement learning (Akrour et al., 2011; Wilson et al., 2012). While the primary focus of RLHF was originally in control (Christiano et al., 2017), following the success of Ziegler et al. (2020) it has recently been broadly adopted by the language modeling (Ouyang et al., 2022; Nakano et al., 2021; Stiennon et al., 2022; Bai et al., 2022a) and even vision communities (Black et al., 2023a; Lee et al., 2023). 

Most works in RLHF optimize a learned reward function, used only at the end of generation, with a policy gradient style-method. Such approaches have been known to be unstable (Engstrom et al., 2020) and hard to scale, while at the same time theoretically existing at an unusual intersection between contextual bandits and RL. In response, several direct alignment methods (Rafailov et al., 2023; Azar et al., 2023; Zhao et al., 2023a) have been developed which simplify the RLHF pipeline by learning a policy from preference data without an intermediate reward function. Such methods however, derived solely as contextual bandits, leave several theoretical and practical questions unanswered which we seek to address.

>  DPO 目前都被推导为上下文赌博机问题，本文尝试解决 DPO 中的一些理论上的疑点

>  赌博机问题和 MDP 问题的差异在于:
>  1. 状态转换: 赌博机问题没有状态转换，agent 做出的动作不会影响环境，MDP 则不然 (这是最核心的差异)
>  2. 序列性: (正因为不存在状态转换) 赌博机问题侧重于独立的、重复的决策，不存在决策之间的内在序列依赖性，MDP 则明确建模了决策的序列性
>  因此，赌博机问题可以视作一个简化的特殊 MDP，这个 MDP 只有一个状态，并且每个动作执行后都会返回到这个唯一的状态

First, though direct alignment methods treat the LLM as a bandit, prior works have demonstrated that it is possible to use dense rewards Zelikman et al. (2022); Chan et al. (2024); Pan et al. (2023) or even approximate dynamic programming (Snell et al., 2022). Moreover, using the regret-model of preferences (Knox et al., 2023; 2024), Contrastive Preference Learning (Hejna et al., 2024) is able to use direct alignment for general MDPs, instead of the specific token MDP used in RLHF. Our work shows how DPO can be interpreted as optimizing a per-token reward function, which in practice is restricted to the family of optimal advantage functions.
>  首先，DPO 直接将 LLM 视为一个赌博机，那么 DPO 是否无法优化 token-level 的序列性信息 ?
>  我们将证明 DPO 可以被解释为优化一个 per-token 的奖励函数，并且该奖励函数在实践中被约束在最优的优势函数族中

Second, if DPO does not learn a reward function, can we still use its reward or value? Prior works have considered using best-of-K (Mudgal et al., 2023) or tree search (Liu et al., 2023b) for alignment with a value function Kim et al. (2022); Li et al. (2017) or discriminator (Yang & Klein, 2021). Using the implicit reward, we show that likelihood search results in a similar solution for direct alignment.
>  其次，DPO 没有直接学习奖励函数，那么我们还可以使用它 (隐式) 提供的奖励或价值 (来指导生成或理解模型的偏好) 吗？
>  我们将证明使用 (DPO 学习到的) 隐式奖励函数执行似然搜索可以在直接对齐中产生类似的解决方案
>  (也就是先用 DPO 训练一个模型，然后直接用基于似然的标准 decoding 方法，例如 beam search 进行 decoding 时，获得的输出实际上和通过明确使用独立的奖励/价值函数或判别器引导更复杂的搜索，例如 best-of-K 或树搜索所获得的结果一样好)

Our work builds on foundational knowledge in maximum entropy RL (Ziebart, 2010) and inverse RL (Ziebart et al., 2008; Ng et al., 1999; Cao et al., 2021). In particular, we leverage the mapping between $Q$ -functions and reward functions under a fixed policy as first done in inverse RL by Garg et al. (2022). 
>  本文的理论基于最大熵 RL 和 IRL 的理论
>  我们利用了 Grag 提出的，在固定的策略下，Q-functions 和奖励函数之间的映射关系

Related to our work, Nachum et al. (2017) uses similar derivations for reinforcement learning in control and Watson et al. (2023) does so for inverse RL from demonstration. Hejna & Sadigh (2024) exploit this relationship for RLHF. While related, these works still require an additional loop of reinforcement learning optimization, which we dispose of in our formulation of feedback learning for LLMs. In the LLM domain, Yu et al. (2024) uses pre-trained models as priors for Q-learning, while Cundy & Emon (2023) considers a similar formulation for imitation learning. In this work instead, we formulate preference-based learning as Q-learning.
>  一些类似的工作也将这一映射关系使用在控制、IRL、RLHF，但这些工作仍然需要额外的 RL 优化，我们的工作则不需要，我们将基于偏好的学习表述为 Q-learning

# 3 Preliminaries
In this section we first define the per-token MDP for large language models, and then describe how it relates to classic RLHF approaches and direct alignment algorithms, specifically DPO. We operate in the typical RLHF setting where we have a dataset $\mathcal{D} = \{(\mathbf{x}^{(i)},\mathbf{y}^{(i)})\}_{i = 1}^{N}$ of language prompts $\mathbf{x}$ and target answers $\mathbf{y}$ , which can each individually be broken down into a sequence of tokens, for example $\mathbf{x} = (x_0,\ldots ,x_m)$ , from a fixed discrete vocabulary $\mathcal{A}$ . Throughout this section we will use the $\mathbf{x}$ , $\mathbf{y}$ notation for the contextual bandit framing where the entire response $\mathbf{y}$ is the action, but will use state $\mathbf{s}$ and action $\mathbf{a}$ notation from RL literature for describing sequences at the token-level.

## 3.1 The Token-level MDP for Large Language Models
We define the token level MDP as a tuple $\mathcal{M} = (\mathcal{S},\mathcal{A},f,r,\rho_0)$ , where the state space $\mathcal{S}$ consists of all tokens generated so far (i.e. $\mathbf{s}_0 = \{x_0,\ldots ,x_m,y_0,\ldots ,y_t\}$ ) and the action space is the vocabulary of tokens $\mathcal{A}$ . The dynamics $f$ are the deterministic transition model between tokens $f(\mathbf{s},\mathbf{a}) = \mathbf{s}|\mathbf{a}$ , where $|$ is concatenation. The initial state distribution $\rho_0$ is a distribution over prompts $\mathbf{x}$ , where an initial state $\mathbf{s}_0$ is comprised of the tokens from $\mathbf{x}$ . 
>  token level MDP 定义为 $\mathcal M = (\mathcal S, \mathcal A, f, r, \rho_0)$
>  $\mathcal S$ 为状态空间，状态由目前为止所有的 tokens 构成
>  $\mathcal A$ 为动作空间，动作即词袋中的一个 token
>  $f$ 为环境动态/转移模型，转移模型是确定性的，定义为 $f(\mathbf s, \mathbf a) =  \mathbf s \mid \mathbf a$，即将状态和动作拼接
>  $\rho_0$ 为初始状态分布，即 prompt $\mathbf x$ 上的分布

In RLHF, the reward function is learned from human feedback over preferences between responses which we will denote using trajectories $\tau$ at the token level. As is typically done (Ziegler et al., 2020; Stiennon et al., 2022), we assume that preference trajectories start at the same state (initial prompt) and end in a terminal state (EOS token), from which future rewards are zero. 
>  我们假设偏好轨迹从相同状态 (初始 prompt) 开始，在终止状态 (EOS token) 结束，终止状态的未来奖励 (Q-value) 为零

In this token level MDP, the corresponding Bradley-Terry preference model Bradley & Terry (1952); Christiano et al. (2017) is

$$
p^* (\tau^w\succeq \tau^l) = \frac{\exp\left(\sum_{i = 1}^N r(\mathbf{s}_i^w,\mathbf{a}_i^w)\right)}{\exp\left(\sum_{i = 1}^N r(\mathbf{s}_i^w,\mathbf{a}_i^w)\right) + \exp\left(\sum_{i = 1}^M r(\mathbf{s}_i^l,\mathbf{a}_i^l)\right)}. \tag{1}
$$

which gives the probability that the "win" trajectory $\tau^w$ of length $N$ is preferred to the "loss" trajectory $\tau^l$ of length $M$ . 

>  在 token level 的 MDP 下，B-T 偏好模型中定义的偏好概率的形式如上
>  此时，一个轨迹 $\tau$ 的 “强度” 被定义为轨迹中所有 token 的 "强度" 的乘积，每一个 token 的 “强度” 定义为其奖励的指数
>  或者也可以理解为轨迹 $\tau$ 的 “强度” 定义为轨迹中所有 token 的奖励之和的指数

Now that we have defined the token level MDP, we can show how it relates to both classic and direct alignment RLHF methods.
>  我们接下来证明 token level MDP 和经典的 RLHF 以及直接对齐方法都相关

## 3.2 The Classical RLHF Methods
Most classical RLHF approaches (Ziegler et al., 2020; Bai et al., 2022b; Ouyang et al., 2022) first learn a reward function from human feedback on prompt and response pairs $(\mathbf{x},\mathbf{y}^w,\mathbf{y}^l)$ , then optimize it with a policy gradient-based method like PPO (Schulman et al., 2017) with an entropy-bonus using the following KL-constrained RL objective

$$
\max_{\pi_{\theta}}\mathbb{E}_{a_t\sim \pi_{\theta}(\cdot |s_t)}\left[\sum_{t = 0}^T (r(s_t,{a}_t) + \underbrace{\beta\log\pi_{\mathrm{ref}}({a}_t|s_t)}_{\mathrm{KL~penalty}}) + \beta \mathcal{H}(\pi_{\theta}){\Large\mid} s_0\sim \rho (s_0)\right] \tag{2}
$$

where $\pi_{\mathrm{ref}}$ is a reference policy, often resulting from supervised finetuning, from which the learned policy should not significantly deviate. 

>  经典的 RLHF 先从人类对 prompt, response pair $(\mathbf x, \mathbf y^w, \mathbf y^l)$ 的反馈学习奖励函数，然后使用基于策略梯度的方法，优化一个带有 KL 约束的 RL 目标函数 Eq2
>  Eq 2 中，外部应该是对整条轨迹 $\tau$ 求期望，或者说对 $a_0, a_1,\dots, a_T$ 的所有动作求期望，对 $a_t$ 求期望是简化写法

>  标准的 RLHF 目标，根据 DPO 中给出的形式，应该是

$$
\max_{\pi_{\theta}}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_{\theta}(y|x)}\big[r_{\phi}(x,y)\big] -\beta \mathbb{D}_{\mathrm{KL}}\big[\pi_{\theta}(y\mid x)\mid \pi_{\mathrm{ref}}(y\mid x)\big]
$$

>  这个形式的等价形式为

$$
\begin{align}
&\max_{\pi_\theta} \mathbb E_{\tau \sim \pi_\theta}[\sum_{t}r(s_t,a_t)  + \beta \log\pi_{\text{ref}}(a_t\mid s_t) - \beta \log\pi(a_t\mid s_t))]\\
=&\max_{\pi_\theta} \mathbb E_{\tau \sim \pi_\theta}[\sum_{t}r(s_t,a_t)  + \beta \log\pi_{\text{ref}}(a_t\mid s_t) - \beta\mathcal H(\pi(\cdot\mid s_t))]\\
\end{align}
$$

>  其中的等号是因为对轨迹求期望拆开来就是对 $\forall s_t, \forall a_t$ 求期望，故每个 $s_t$ 实际上都会遍历所有的 $\log \pi(a_t\mid s_t)$，进而可以写为 $\mathcal H(\pi(\cdot\mid s_t))$
>  这也符合标准的最大熵 RL 的目标形式 (将 $r(s_t, a_t) + \beta \log \pi_{\text{ref}}(a_t\mid s_t)$ 视为新的奖励函数)，即每一个时间步都会有熵惩罚项
>  而 Eq 2 中将熵放在了求和项后面，也就是只在轨迹末尾在具有熵惩罚项，这不符合标准的 RLHF 目标以及最大熵 RL 的目标，在这里姑且认为是作者的笔误

However, in classic RLHF methods the reward function is learned as a contextual bandit with the preference model

$$
p^* (\mathbf{y}^w\succeq \mathbf{y}^l) = \frac{\exp r(\mathbf{x},\mathbf{y}^w)}{\exp r(\mathbf{x},\mathbf{y}^w) + \exp r(\mathbf{x},\mathbf{y}^l)}
$$

and is thus only applied at the final timestep for the last action where $\mathbf{a}$ is EOS. 

>  在经典的 RLHF 中，奖励函数的形式是一个上下文赌博机，也就是定义于整个回应 $\mathbf y^w, \mathbf y^l$，形式如上
>  因此，实际上 RL 学习中，只有最后时间步的最后动作，即输出 token EOS 时，才会获得形式如上的奖励

In practice the actual reward used in the token-level PPO is

$$
r(\mathbf{s}_t,\mathbf{a}_t) = \left\{ \begin{array}{ll}\beta \log \pi_{\mathrm{ref}}(\mathbf{a}_t|\mathbf{s}_t), & \mathrm{if~}\mathbf{s}_{t + 1}\mathrm{~is~not~terminal}\\ r(\mathbf{x},\mathbf{y}) + \beta \log \pi_{\mathrm{ref}}(\mathbf{a}_t|\mathbf{s}_t), & \mathrm{if~}\mathbf{s}_{t + 1} = \mathbf{y}\mathrm{~is~terminal} \end{array} \right. \tag{3}
$$

in a maximum entropy formulation. 

> 因此，在实际中，token-level 的 PPO 优化使用的奖励函数定义如 Eq3
> 当输出 token 不是 EOS 时，奖励函数给出的奖励仅和熵惩罚有关，只有输出 token 是 EOS 时，奖励函数才给出基于偏好定义的奖励

This leads to an interesting contradiction where the reward function $r$ is treated like a bandit, but the actual RL value function and optimization is done per-token in practice.
>  因此，这实际上是一个矛盾，即奖励函数 $r$ 定义为一个赌博机，但实际的 RL 优化却按 per-token 的形式

## 3.3 Direct Preference Optimization
Unlike classical RLHF, DPO, as derived in Rafailov et al. (2023), stays entirely within the contextual bandits setting entirely and also uses the bandit-based preference model in section 3.2. To circumvent the need for an RL algorithm, DPO uses the well-known closed form solution to the KL-contextual bandit version of the RL problem posed in eq. (2) (Ziebart et al., 2008; Levine, 2018):

$$
\pi^{*}(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})}\pi_{\mathrm{ref}}(\mathbf{y}|\mathbf{x})e^{r(\mathbf{x},\mathbf{y})}
$$

where $\pi^{*}$ is the optimal policy and $Z(\mathbf{x})$ is the partition function that normalizes it. 

>  DPO 则完全在上下文赌博机的设定下优化，且同样使用了相同的偏好模型
>  DPO 使用了 KL 约束下的赌博机问题的闭式解形式，如上所示

DPO rearranges this equation to solve for reward as $r(\mathbf{x},\mathbf{y}) = \beta \log \pi^{*}(\mathbf{y}|\mathbf{x}) -\beta \log \pi_{\mathrm{ref}}(\mathbf{y}|\mathbf{x}) -Z(\mathbf{x})$ . Substituting this relationship into the standard binary cross-entropy loss function used for reward modeling yields the DPO loss equation as the partition function $Z(\mathbf{x})$ cancels from the Bradley Terry model.

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta};\pi_{\mathrm{ref}}) = -\mathbb{E}_{(\mathbf{x},\mathbf{y}^{w},\mathbf{y}^{l})\sim \mathcal{D}}\left[\log \sigma \left(\beta \log \frac{\pi_{\theta}(\mathbf{y}^{w}\mid\mathbf{x})}{\pi_{\mathrm{ref}}(\mathbf{y}^{w}\mid\mathbf{x})} -\beta \log \frac{\pi_{\theta}(\mathbf{y}^{l}\mid\mathbf{x})}{\pi_{\mathrm{ref}}(\mathbf{y}^{l}\mid\mathbf{x})}\right)\right] \tag{4}
$$

For brevity we use $\sigma$ to denote the logistic function. 

>  DPO 根据闭式解，将奖励函数重新以 $\pi^*$ 相关的形式表述，将新形式代换到奖励建模的标准交叉熵损失中，就得到了 DPO 损失，其中配分函数被消除

In the next section, we show how an alternative derivation of DPO can also cast its optimization within the token-level MDP.
>  我们将在下一节证明 DPO 的另一种形式的推导可以将其转化为 token-level MDP 下的优化形式

# 4 Theoretical Insights
In this section we explore how DPO can theoretically be cast into the token-level MDP, and explore the consequences of doing so. First, we provide a token level derivation of DPO under the assumptions in section 3.1. Next, we show that even in the token MDP, DPO is able to fit any reward function in the multi-step Bradley Terry preference model eq. (1). Ultimately, this shows that DPO can potentially be used for more sequential optimization tasks, like multi-turn interactions or even multi-modal generation.
>  我们探索如何将 DPO 在理论上转化为 token-level MDP 的形式
>  我们首先为 DPO 提供一个 token-level 的理论推导
>  然后我们将证明在 token-level MDP 的情况下，DPO 也可以拟合多步 B-T 模型定义下的任意奖励函数
>  这些结论说明了 DPO 可以用于更序列化的优化任务，例如多轮交互或多模态生成

## 4.1 DPO as a $Q$ -function in the Token Level MDP
**RL in the Token-level MDP.** While the original derivation of DPO relies on the fact that $Q^{*}(\mathbf{x},\mathbf{y}) = r(\mathbf{x},\mathbf{y})$ , this relationship does not hold in the token-level MDP. 

>  DPO 的原始推导依赖于一个事实: 最优动作价值函数等于即时奖励，即 $Q^*(\mathbf x, \mathbf y) = r(\mathbf x, \mathbf y)$
>  显然，在 token-level MDP 下，该关系不成立

>  即时奖励函数 $r(\mathbf x, \mathbf y )$ 是指在特定步骤中，从状态 $\mathbf x$ 采取动作 $\mathbf y$ 立即获得的奖励
>  最优动作价值函数 $Q^*(\mathbf x, \mathbf y)$ 是指在最优策略下，从状态 $\mathbf x$ 采取动作 $\mathbf y$ 之后，未来所有步骤的总折扣奖励
>  但原始的 DPO 的背景是一个赌博机，不存在后续的步骤，也没有未来的奖励可言，此时，选择一个动作获得的 “总未来奖励” 就是当下获得的即时奖励，且不仅仅是 $Q^*$，其他的 $Q$ 也是如此，因为动作 $\mathbf y$ 已经给出，故 $r(\mathbf x, \mathbf y)$ 是固定的，$Q$ 值的计算与具体的策略无关

To resolve this, we need to develop new mathematical results that will allow us to relate the reward function in the Token-level Bradley Terry model eq. (1) to the corresponding optimal policy $\pi^{*}$ . 
>  为此，我们需要推导出一个将 Eq 1 中 token-level BT 模型中的奖励函数和对应的最优策略 $\pi^*$ 关联的形式

In the general maximum entropy RL setting, the fixed point solution of eq. (2) is given by (Ziebart, 2010) as

$$
\pi^{*}(\mathbf{a}_{t}|\mathbf{s}_{t}) = e^{(Q^{*}(\mathbf{s}_{t},\mathbf{a}_{t}) -V^{*}(\mathbf{s}_{t})) / \beta} \tag{5}
$$

where $\pi^{*}(\mathbf{a}|\mathbf{s})$ is the optimal policy and $Q^{*}(\mathbf{s},\mathbf{a})$ is the optimal Q-function which models the total future reward from $(\mathbf{s},\mathbf{a})$ under $\pi^{*}$ .

>  在通用的最大熵 RL 设定下，对 Eq 2 的 fix point solution 形式为 Eq 5，其中 $\pi^*$ 为最优策略，$Q^*$ 为最优 Q-function

The optimal value function $V^{*}$ is a function of $Q^{*}$ ,

$$
V^{*}(\mathbf{s}_{t}) = \beta \log \sum_{\mathbf{a}\in \mathcal{A}}e^{Q^{*}(\mathbf{s}_{t},\mathbf{a}) / \beta} \tag{6}
$$

such that the policy $\pi^{*}$ integrates to one. 

>  Eq 5 中的最优价值函数 $V^*$ 则是 $Q^*$ 的函数，如 Eq 6

Unfortunately unlike in the bandits setting this relationship gives us no specific information about the reward function $r$ at a single state action pair since the optimal policy optimizes for total future returns as estimated by $Q$ . To do so, we will need to consider the relationship between $Q^{*}$ and $r$ .

>  Eq 5 给出的最优策略的形式和奖励函数 $r$ 无关，为了将最优策略和奖励函数联系起来，我们需要考虑 $Q^*$ 和 $r$ 的关系

**From $r$ to $Q^{*}$ .** The relationship between future returns and the current timestep is captured by the bellman equations which are satisfied by any valid Q-function. We write this below for the optimal policy $\pi^{*}$ under the reward $r$ with a KL divergence penalty:

$$
Q^{*}(\mathbf{s}_{t},\mathbf{a}_{t}) = \left\{ \begin{array}{ll}r(\mathbf{s}_{t},\mathbf{a}_{t}) + \beta \log \pi_{\mathrm{ref}}(\mathbf{a}_{t}|\mathbf{s}_{t}) + V^{*}(\mathbf{s}_{t + 1}), & \mathrm{if~}\mathbf{s}_{t + 1}\mathrm{~is~not~terminal}\\ r(\mathbf{s}_{t},\mathbf{a}_{t}) + \beta \log \pi_{\mathrm{ref}}(\mathbf{a}_{t}|\mathbf{s}_{t}), & \mathrm{if~}\mathbf{s}_{t + 1}\mathrm{~is~terminal} \end{array} \right. \tag{7}
$$

We can then rearrange the bellman equation for the optimal $Q$ -function in terms of the reward. This style of relationship was first explored by Garg et al. (2022) in imitation learning and later in Hejna & Sadigh (2024) for preference-based RL. However, these works require the use of a discount factor $\gamma < 1$ which is typically not used in RLHF. 

>  Eq 7 给出了经典的 Bellman 方程的形式，其中奖励函数被定义为 $r'(s_t, a_t) = r(s_t , a_t) + \beta \log \pi_{\text{ref}}(a_t\mid s_t)$
>  根据 Eq 7，我们可以将 $Q^*$ 写为由 $r$ 表示的形式，反之亦然，其他的一些工作也有类似的探索，但它们都使用了折扣因子 $\gamma < 1$，而通常在 RLHF 中不会使用 $\gamma$

In the appendix we prove the following Lemma which shows that this relationship is indeed one-to-one in the token MDP as well.

**Lemma 1.** Under mild assumptions, there is a bijection between reward functions $r(\mathbf{s}_t, \mathbf{a}_t)$ and corresponding optimal $Q$ -functions $Q^{*}(\mathbf{s}_t, \mathbf{a}_t)$ in the token MDP.

>  **Lemma 1**
>  在 token MDP 中，奖励函数 $r(s_t, a_t)$ 和对应的最优 Q-function 存在一一对应关系

This leads us to a rather interesting conclusion - that an LLM is always the optimal soft Q-functions for some reward function in the token MDP. 
>  在 DPO 中，我们知道 (在 token MDP 中) LLM (策略) 总是可以视作隐式地对应了一个奖励函数，在这里结合 Lemma 1，我们可以推导出 LLM (策略) 还总是对应于某个奖励函数的最优 Q-function

Consider any LLM which outputs logits $l_{\theta}$ and temperature parameter $\beta$ . As is common practice, we take the sampling policy $\pi$ to be the softmax over tokens modulated by temperature parameter $\beta$ -which is precisely eq. (5) where $Q^{*} = l_{\theta}$ because the value optimal function $V^{*}$ is precisely $\beta \log Z(\mathbf{s}_t)$ , normalizing the distribution. 
>  考虑任意一个输出 logits $l_\theta$，温度参数为 $\beta$ 的 LLM，我们通常会将模型的策略 (token 上的分布) 定义为 logits 除以温度参数 $\beta$ 的 softmax，而这个形式正好匹配 Eq 5，其中 $Q^*=l_\theta$，$V^* = \beta \log Z(s_t)$ 作为分布的规范化因子

The corresponding reward function may not be smooth or well-behaved. Notably, the logits have a free parameter due to the softmax. While this free-parameter results in the same optimal policy per later arguments, it means the sequence of values may not be smooth. 
>  但如果我们简单将原始 logits $l_\theta$ 视作 $Q^*$，那么隐含的，驱动这些 logits 的奖励函数可能不会那么平滑和明确定义
>  注意到这些 logits 实际上都有一个自由参数 (也就是对 logits 都加上或减去一个常数，softmax 的结果不会变)，logits 的自由参数不会改变策略，但会改变 $Q^*, V^*$ 函数的绝对值 (导致价值序列不平滑)，进而改变奖励函数

>  这里的不平滑是指 $Q, V$ 函数的决定值在训练或时间上的轨迹的不平滑，因此如果没有明确的约束，在训练的时候，因为输出的概率分布都相同，故这些 logits 可能会整体随意地上下浮动，这会给优化策略带来挑战

The question then becomes how to finetune the LLM such that it is the optimal Q-function for a reward function $r$ that aligns with human preferences. To do so, we will complete our derivation of DPO in the token MDP.
>  因此问题的核心在于如何微调 LLM 使得其最优 Q-function 对应的奖励函数 $r$ 和人类偏好对齐

**DPO learns our best estimate of $Q^{*}$** . Now that we have established a bijection between $r$ and $Q^{*}$ , we can derive a token-level version of DPO to align the implicit reward, induced by the $Q$ function represented by the language model, with that of the best estimate of reward, according to Bradley-Terry model in eq. (1). 
>  确立了奖励函数和最优 $Q$ 函数之间的双射关系后，我们推导 token-level 版本的 DPO

To do so, we need to represent the sum of rewards first in terms of the $Q$ -function $Q^{*}$ , and then in terms of the policy $\pi^{*}$ . We complete the first step by inverting the Bellman equation in eq. (7) and substituting it into the sum of rewards over a trajectory $\tau = \{\mathbf{s}_1, \mathbf{a}_1, \ldots , \mathbf{a}_{T -1}, \mathbf{s}_T\}$ .

$$
\begin{align}
\sum_{t = 0}^{T -1}r(\mathbf{s}_t,\mathbf{a}_t)&= \sum_{t = 0}^{T -1}\left(Q^* (\mathbf{s}_t,\mathbf{a}_t) -\beta \log \pi_{\mathrm{ref}}(\mathbf{a}_t|\mathbf{s}_t) -V^* (\mathbf{s}_{t + 1})\right) \\
&= Q^* (\mathbf{s}_0,\mathbf{a}_0) -\beta \log \pi_{\mathrm{ref}}(\mathbf{a}_0|\mathbf{s}_0) -\sum_{t = 1}^{T -1}Q^* (\mathbf{s}_t,\mathbf{a}_t) -V^* (\mathbf{s}_t) -\beta \log \pi_{\mathrm{ref}}(\mathbf{a}_t|\mathbf{s}_t) \end{align}
$$

The equality follows from $V^{*}(\mathbf{s}_{T}) = 0$ and re-arranging the sum to isolate $t = 0$ . 
>  我们的思路是先将奖励和用最优 $Q$ 函数 $Q^*$ 表示，进而用策略 $\pi^*$ 表示
>  我们首先根据 Eq 7，将每一个时间步的奖励 $r(s_t, a_t)$ 写为用 $Q^*, \pi_{\text{ref}}, V^*$ 表示的形式，进而整个轨迹的奖励和可以写为以上的形式 ($s_T$ 为终止状态，因此没有加上 $r(s_t, a_t)$)
>  其中第二个等号利用了 $V^*(s_T) = 0$，并重新组织了求和顺序

As $V^{*}$ is written entirely in terms of $Q^{*}$ and $\beta$ per eq. (6), we have expressed the sum of return over the sequence just in terms of $Q^{*}$ . 
>  因为 Eq 6 给出了 $Q^*$ 和 $V^*$ 的关系，故 $V^*$ 完全可以用 $Q^*$ 和 $\beta$ 来表示，因此我们已经将序列上的回报写为了仅和 $Q^*$ 有关的形式

Next, we exchange $Q^{*}$ for $\pi^{*}$ . We can log-linearize eq. (5) as $\beta \log \pi^{*}(\mathbf{a}_{t}|\mathbf{s}_{t}) = Q^{*}(\mathbf{s}_{t}, \mathbf{a}_{t}) -V^{*}(\mathbf{s}_{t})$ . This is equivalent to stating that the language model probabilities are just the softmax over $l_{\theta} = Q^{*}$ with temperature $\beta$ . Continuing from the above, with this substitution we get

$$
= Q^{*}(\mathbf{s}_{0},\mathbf{a}_{0}) -\beta \log \pi_{\mathrm{ref}}(\mathbf{a}_{0}|\mathbf{s}_{0}) + \sum_{t = 1}^{T -1}\beta \log \frac{\pi^{*}(\mathbf{a}_{t}|\mathbf{s}_{t})}{\pi_{\mathrm{ref}}(\mathbf{a}_{t}|\mathbf{s}_{t})} = V^{*}(\mathbf{s}_{0}) + \sum_{t = 0}^{T -1}\beta \log \frac{\pi^{*}(\mathbf{a}_{t}|\mathbf{s}_{t})}{\pi_{\mathrm{ref}}(\mathbf{a}_{t}|\mathbf{s}_{t})}
$$

where the final step results from adding and subtracting $V^{*}(\mathbf{s}_{0})$ and applying the substitution again. 

>  我们继而考虑用 $\pi^*$ 表示 $Q^*$，我们将 Eq 5 两边取对数，得到 $\beta \log \pi^*(a_t\mid s_t) = Q^*(s_t, a_t) - V^*(s_t)$，这等价于说 LLM 为 tokens 赋予的概率就是通过 logits $l_\theta = Q^*$ 和温度参数 $\beta$ 计算得到的
>  我们用 $\beta \log \pi^*(a_t\mid s_t) = Q^*(s_t, a_t) - V^*(s_t)$ 代入上面的式子，继而得到了更简单的形式如上

Now, this representation for the sum of rewards in terms of the optimal policy can be directly substituted into the preference model in eq. (1), where the $V^{*}(\mathbf{s}_{0})$ term will cancel just as $Z(\mathbf{x})$ did in the original DPO derivation assuming $\tau^{w}$ and $\tau^{l}$ start at the same state $\mathbf{s}_0$ , giving us the policy-induced preference model

$$
p_{\pi^*}\big(\tau^w\succeq \tau^l\big) = \sigma \left(\sum_{t = 0}^{N -1}\beta \log \frac{\pi^*\big(\mathbf{a}_t^w\big|\mathbf{s}_t^w\big)}{\pi_{\mathrm{ref}}(\mathbf{a}_t^w\big|\mathbf{s}_t^w)} -\sum_{t = 0}^{M -1}\beta \log \frac{\pi^*\big(\mathbf{a}_t^l\big|\mathbf{s}_t^l\big)}{\pi_{\mathrm{ref}}(\mathbf{a}_t^l\big|\mathbf{s}_t^l)}\right). \tag{8}
$$

>  至此，我们将奖励和表示为了和最优策略有关的形式，我们进而将它代入 Eq 1 中的偏好模型，其中 $V^*(s_0)$ 会被消掉，就像 $Z(x)$ 在最初的 DPO 推导中一样
>  假设轨迹 $\tau^w, \tau_l$ 从相同状态 $s_0$ 开始，那么我们就有以上策略引导的偏好模型

To derive the final DPO loss function, we can take the KL-divergence between the empirical preference model of our dataset $p_{\mathcal{D}}$ and the preference model implied by a learned policy $p_{\pi_{\theta}}$ , $\mathbb{D}_{\mathrm{KL}}(p_{\mathcal{D}}||p_{\pi_{\theta}})$ . This results in

$$
\mathcal{L}(\pi_{\theta},\mathcal{D}) = -\mathbb{E}_{(\tau_w,\tau_l)\sim \mathcal{D}}\left[\log \sigma \left(\left(\sum_{t = 0}^{N -1}\beta \log \frac{\pi^*(\mathbf{a}_t^w|\mathbf{s}_t^w)}{\pi_{\mathrm{ref}}(\mathbf{a}_t^w|\mathbf{s}_t^w)}\right) -\left(\sum_{t = 0}^{M -1}\beta \log \frac{\pi^*(\mathbf{a}_t^l|\mathbf{s}_t^l)}{\pi_{\mathrm{ref}}(\mathbf{a}_t^l|\mathbf{s}_t^l)}\right)\right)\right]\quad(9)
$$

In the next section we demonstrate that DPO can learn any dense reward function in the token-level MDP.

>  我们进而用 Eq 8 计算和数据集上的经验偏好模型 $p_{\mathcal D}$ 的 KL 散度，就得到了最终的 DPO 损失

>  推导

$$
\begin{align}
\mathrm {D_KL}(p_{\mathcal D} \| p_{\pi_\theta}) &= \mathbb E_{p_{\mathcal D}}\left[\log \frac {p_{\mathcal D}}{p_{\pi_\theta}}\right]\\
&= -\mathbb E_{p_{\mathcal D}}[\log p_{\pi_\theta}] + \mathbb E_{p_{\mathcal D}}[\log p_{\mathcal D}]\\
&=-\mathbb E_{(\tau_w, \tau_l)\sim \mathcal D}[\log p_{\pi_\theta}(\tau_w \succeq \tau_l)] + C
\end{align}
$$

>  推导完毕

## 4.2 Token-Level DPO Can Parameterize Any Dense Reward Function.
In the previous section we derived DPO using the bijection between reward functions and optimal $Q$ -functions uniquely available in the token-level MDP. 
>  在上一节中，我们用了 (token-level MDP 中才有的) 奖励函数和最优 $Q$ 函数之间的双射性质推导了 DPO

An alternative view of DPO casts it as restricting the learned reward function such that it belongs to the class optimal advantage functions $A^{*}(\mathbf{s},\mathbf{a}) = Q^{*}(\mathbf{s},\mathbf{a}) -V^{*}(\mathbf{s})$ from which an optimal policy is readily obtained per eq. (5). Here we show that this restriction does not limit the class of reward functions we can represent. 
>  有人认为 DPO 将奖励函数限制在了满足最优优势函数 $A^*(s, a) = Q^*(s, a) - V^*(s)$ 的形式 (参照 Eq 7 和上面的推导，DPO 就是这么将奖励函数代换为和 $Q, V$ 有关的形式的)，然后用它直接来塑造策略 (参照 Eq 5)
>  我们在这里证明尽管奖励函数的结构限制为最优优势函数的形式，但这并不会限制奖励函数的表达能力，即这并不会导致 DPO 无法表示某种类型的奖励函数

We begin by expanding the definition of equivalency used in Rafailov et al. (2023) to the broader class of potential-based reward shaping functions:

**Definition 1.** Two reward functions $r(\mathbf{s}_t,\mathbf{a}_t)$ and $r^{\prime}(\mathbf{s}_t,\mathbf{a}_t)$ are equivalent if there exists a potential function $\Phi (\mathbf{s})$ , such that $r^{\prime}(\mathbf{s}_t,\mathbf{a}_t) = r(\mathbf{s}_t,\mathbf{a}_t) + \Phi (\mathbf{s}_{t + 1}) -\Phi (\mathbf{s}_t)$
>  **Definition 1**
>  对于两个奖励函数 $r(s_t, a_t)$ 和 $r'(s_t, a_t)$ 如果存在一个势能函数 $\Phi(s)$，使得 $r'(s_t, a_t) = r(s_t, a_t) + \Phi(s_{t+1}) - \Phi(s_t)$，就称这两个奖励函数等价

In Ng et al. (1999)'s seminal work, the authors proved that two equivalent reward functions defined per definition 1 have the same optimal policy. By log-linearizing the optimal policy fixed point in eq. (5) and substituting in the Bellman equation from eq. (7) (Nadhum et al., 2017; Watson et al., 2023), we have

$$
\beta \log \frac{\pi^{*}(\mathbf{a}_{t}|\mathbf{s}_{t})}{\pi_{\mathrm{ref}}(\mathbf{a}_{t}|\mathbf{s}_{t})} = r(\mathbf{s}_{t},\mathbf{a}_{t}) + V^{*}(\mathbf{s}_{t + 1}) -V^{*}(\mathbf{s}_{t}). \tag{10}
$$

This is precisely the optimal advantage function, where $V^{*}$ directly follows the form of a potential shaping function. 

>  Ng (1999) 的工作证明了在 Definition 1 定义下，两个等价的奖励函数 (即它们之间只相差一个势函数的差值) 在对应的 MDP 中会产生相同的最优策略
>  我们利用 Eq 5 给出的最优策略形式，将其代入 Eq 7 中的 Bellman equation，得到 Eq 10
>  Eq 10 实际上就是最优优势函数的定义，也就是 DPO 优化的核心量，或者说 DPO 所定义的奖励函数的形式，而 Eq 10 中的 $V^*$ 实际上就扮演了势函数 $\Phi(s)$ 的角色
>  这意味着，DPO 优化的核心量实际上是一个和原始奖励函数 $r(s_t, a_t)$ 等价的新奖励函数，二者仅相差一个以 $V^*$ 为定义的奖励塑形项
>  而根据 Ng 的结论，这种变化不会影响最优策略，因此 DPO 虽然表面上限制了奖励函数的表达形式为某种优势函数，但仍然能找到与任意原始奖励函数相同的最优策略

>  推导

$$
\begin{align}
Q^*(s_t, a_t) &= r(s_t, a_t) + \beta \log\pi_{\text{ref}}(a_t\mid s_t)+ V^*(s_{t+1})\\
Q^*(s_t, a_t) - V^*(s_t)&=r(s_t, a_t) + \beta \log\pi_{\text{ref}}(a_t\mid s_t)+ V^*(s_{t+1}) -V^*(s_t)\\
\beta \log\pi^*(a_t\mid s_t) &=r(s_t, a_t) + \beta \log\pi_{\text{ref}}(a_t\mid s_t)+ V^*(s_{t+1}) -V^*(s_t)\\
\beta \log \frac {\pi^*(a_t\mid s_t)}{\pi_{\text{ref}}(a_t\mid s_t)} &= r(s_t,a_t) + V^*(s_{t+1})  - V^*(s_t)
\end{align}
$$

>  推导完毕

Watson et al. (2023) first used this derivation to arrive at a "coherent" reward function and follow-ups arrived at the same conclusion by noting that using the advantage as reward preserves the optimal policy (Knox et al., 2024; Hejna et al., 2024). Unlike prior works, however, we demonstrate that this re-parameterization also leads to the same exact preference distribution as $r$ .
>  其他研究者也发现了使用优势函数作为奖励函数的形式仍然可以保留最优策略这一结论
>  而 DPO 的独特贡献在于证明了这一重参数化形式不仅可以保留最优策略，并且能够保留人类偏好分布，即可以直接从偏好数据中优化

**Theorem 1.** Given a reference policy $\pi_{\mathrm{ref}}$ and a parameter $\beta >0$ all reward classes consistent with the Plackett-Luce (and Bradley-Terry) models in eq. (1) can be represented with the a re-parameterization of the form

$$
r(\mathbf{s},\mathbf{a}) = \beta \log \pi (\mathbf{a}|\mathbf{s}) -\beta \log \pi_{\mathit{ref}}(\mathbf{a}|\mathbf{s}) \tag{11}
$$

within the token MDP where $V^{*}(\mathbf{s}_t) = 0$ for all terminal states.

>  **Theorem 1**
>  给定参考策略 $\pi_{\text{ref}}$ 和参数 $\beta > 0$，在 token MDP 和所有终止状态的状态价值 $V^*(s_t) = 0$ 的条件下，所有和 P-L 模型一致的奖励类可以用 Eq 11 的重参数化形式表示
>  也就是说，这种特殊的奖励函数形式足以表示所有符合人类偏好的奖励，并且能够得到最优策略和偏好分布

**Proof.** Above we derived the invariance of the optimal policy under the re-parameterization. The preference model can be shown to be invariant by substituting and following the same steps used to arrive at eq. (8) in the last section, or by following Definition 1 from Watson et al. (2023).
>  **Proof**
>  1. 关于策略的不变性 (这样参数化以后，仍然能够得到最优策略): 在 Eq 10 我们用最大熵 RL 的最优策略的形式和 token MDP 的 Bellman equation 推导出了这种重参数化形式相较于最优策略的不变性
>  2. 关于偏好的不变性 (这样参数化以后，能够保持和原始奖励函数表示相同的偏好分布): 将 Eq 10 代入偏好模型，就会发现多余的项会上下消掉，因此得到的是和原来奖励函数相同的偏好分布

Interestingly, in practice, the potential function $\Phi (\mathbf{s}_t)$ represents the free parameter in the logits of the language model. An equal shift along all logits yields the same policy, but different Q-functions and corresponding rewards. The above Theorem proves that all of these are in the same equivalence class and induce the same set of preferences.
>  有趣的是，在实践中，势函数 $\Phi(s_t)$ 就表示了语言模型中 logits 的自由参数，而在所有 logits 上同时加上任意的势函数，仍然会得到相同的策略，但是会得到不同的 Q-function 以及对应的奖励函数
>  上面的定理就证明了所有的这些 logits 都属于相同的等价类，并且诱导出相同的偏好分布

Moreover, this Theorem implies that we can use DPO to learn the optimal policy for any per-token reward function, provided preference queries start at the same state and end at a terminal state. 
>  该定理还表明了我们可以用 DPO 学习任意 per-token 奖励函数的最优策略 (定理中没有限制 $r(s_t, a_t)$ 的形式，只是告诉我们它等价地对应到 $\beta \log \frac{\pi^{*}(\mathbf{a}_{t}|\mathbf{s}_{t})}{\pi_{\mathrm{ref}}(\mathbf{a}_{t}|\mathbf{s}_{t})}$)，只要 preference queries 都从相同的状态开始，并且在一个终止状态结束

In addition, DPO always fits an optimal advantage function for some reward which is responsible for credit assignment. Thus, the training data determines how close the learned advantage corresponds to that of the true reward. 
>  此外，这个定理也说明了 DPO 的拟合目标: 最优优势函数总是可以对应到一个用于信用分配的奖励函数
>  训练数据决定了这个学习到的优势函数能够如何对齐到真实的奖励函数
 
This is in contrast to methods that estimate the reward function and then additionally employ some policy improvement mechanism. Which algorithm performs better remains largely an open or empirical question.

The above derivations cast a language model as a Q function in the discrete token-level MDP. While this interpretation does not generally hold in continuous spaces, we can extend many of our results to other specially structured MDPs, like those present in diffusion. See Appendix B for more thorough treatment.
>  上述的推导将语言模型转化为离散的 token-level MDP 中的 Q-function，这一个解释虽然通常不在连续空间成立，但我们可以将这一结论拓展到任意的特殊结构 MDP，例如扩散 MDP

# 5 Practical Insights
In this section we discuss the empirical implications of our theoretical analysis. First, we qualitatively show that DPO can learn per-token credit assignment. Next, we use the derivations of the prior section to connect guided decoding and search-based algorithms, such as MCTS, to likelihood-based search on the DPO policy and empirically validate these results. Finally, (for the first time), we mathematically explain the phenomenon of decreasing likelihoods during DPO training, observed in the research and industry community.
>  本节将展示之前理论分析的经验性实证结果
>  首先，我们定性地展示 DPO 可以学习到 per-token 的信用分配；然后，我们使用之前的推导将引导式解码和基于搜索的算法，例如 MCTS，联系到在 DPO 优化后的策略上进行基于似然的搜索，并经验性验证这些结果

For all empirical evaluations we use the Pythia 2.8B model Biderman et al. (2023) and the Reddit TL;DR summarization dataset Stiennon et al. (2022). We use the default hyper-parameters from the original public DPO implementation, unless otherwise stated.

## 5.1 Does DPO Learn Credit Assignment?
In the previous section we outlined how the trained DPO policy represents an optimal Q-function for some reward that optimizes the preference equation. 
>  在之前的部分，我们证明了 DPO 训练将得到某个优化偏好的奖励函数所对应的最优 Q-function 的最优策略

In this section, we evaluate qualitatively if the DPO-trained model is able to learn credit assignment from trajectory feedback. We begin with a generic set of Reddit posts for the TL;DR test dataset, which we provide in Appendix C with additional examples.
>  我们在本节定性地评估 DPO 训练的模型是否可以通过轨迹反馈学习信用分配

In our representative example, the user discusses an employment negotiations situation. Two answers are shown in Figure 1. 

![](https://cdn-mineru.openxlab.org.cn/extract/59a4bfae-d140-4047-a800-2cfb72185b92/5d7f0a767bc15192892f914dfdac30cf2137ecd1bb317106d46f39fee4d24c1f.jpg) 

Figure 1: **Credit assignment in DPO based on answer-level feedback.** We provide two summaries to a Reddit post about a job interview. The left is the base response and on the right we have introduced errors in the salary range and the position level. Each token is colored corresponding to the DPO implicit reward as expressed in Eq. 11 (darker is higher), using the trained model. We see that the model correctly highlights the erroneous statements, without much change to the value of the other tokens, which indicates the ability to do credit assignment.

The base summary, which is correct is provided on the left. On the right we modify the summary by introducing a higher-level position and a corresponding higher salary. 

For each token in both answers we compute the DPO reward (equivalently the advantage function or "coherent" reward (Watson et al., 2023)), $r(\mathbf{s},\mathbf{a}) = \beta \log \pi_{\theta}(\mathbf{s}|\mathbf{a}) -\beta \log \pi_{\mathrm{ref}}(\mathbf{s}|\mathbf{a})$ , where $\pi_{\theta}$ as outlined in Theorem 1 (here $\pi_{\theta}$ is our DPO-trained model and $\pi_{\mathrm{ref}}$ is the SFT model). In Figure 1 each token is colored proportionally to this reward. We see that the model successfully identifies the tokens corresponding to the erroneous statements, while still maintaining comparable values for the rest, which is indicates that it can do credit assignment. 
>  我们为答案中的每个 token 计算 DPO 奖励: $r(s, a) = \beta \log \pi_\theta(s\mid a) - \beta \log \pi_{\text{ref}}(s\mid a)$
>  Figure 1 中的颜色深度和奖励成正比，可以看到 DPO 训练出来的模型可以成功识别对应于语句中错误的 tokens (给了较低的奖励)，其他正确的 tokens 则维持了正常的奖励

Moreover, we see that within the context of the first error ("250K" salary) the model still allocates reasonable values to the rest of the tokens and specifically identifies the second error "management position". This is a promising sign of the ability to do "stitching" Levine et al. (2020) i.e. a form of combinatorial generalization from offline data. If this is the case, our findings could be significant for the use of reinforcement learning and RLHF in LLMs, particularly for compositional tasks, such as code and reasoning. At the same time, in the recently introduced RewardBench Lambert et al. (2024), DPO models have demonstrated strong performance as classifiers on reasoning tasks. We believe these are encouraging results, which warrant further large-scale study beyond our qualitative observations.

## 5.2 Connecting Guided Decoding and Search to Likelihood-Based DPO Optimization
Recently Large Language Models have been combined with search algorithms during the inference stage Mudgal et al. (2024); Feng et al. (2024); Huang et al. (2024); Liu et al. (2023a), which have found to improve the quality of responses over standard next token decoding. 
>  最近地 LLM 在推理阶段结合了搜索算法，相较于标准的 next token decoding，提高了回应的质量

Following the standard literature, these methods rely on a (usually sparse) reward signal or model $r_{\theta}(\mathbf{s}_{\mathbf{t}},\mathbf{a}_t)$ which they use to train a separate value function $V_{\theta}(\mathbf{s}_t)$ . During inference time they deploy a graph-search algorithm in the token MDP as outlined in Section 3.1 to maximize the sum of rewards. 
>  这些方法通常会使用一个稀疏的奖励模型 $r_\theta(s_t, a_t)$ 来训练一个单独的价值函数 $V_\theta(s_t)$
>  在推理阶段，它们采用 token MDP 上的图搜索算法来最大化奖励和

Let us consider the search problem outlined in Eq. 2 with a partial expansion of length $K$ :

$$
\max_{\mathbf{a}_0,\ldots ,\mathbf{a}_K}r(\mathbf{s}_0,\mathbf{a}_0) + \beta \log \pi_{\mathrm{ref}}(\mathbf{s}_0,\mathbf{a}_0) + \ldots +r(\mathbf{s}_t,\mathbf{a}_t) + \beta \log \pi_{\mathrm{ref}}(\mathbf{s}_K,\mathbf{a}_K) + V^* (\mathbf{s}_{K + 1}) \quad (12)
$$

where $V^{*}$ is the optimal corresponding value function. 

>  考虑一个有限步骤的搜索问题，我们希望找到一个 $K$ 步的最佳动作序列，这个动作序列的价值使用上式来表示
>  可以看到，它等于每一步的即时奖励，以及第 $K+1$ 步到最终状态的最佳预期奖励

Now, if we directly substitute the reward representation from Eq. 10 into the above and considering a telescoping sum, with some standard algebra, we obtain that the above objective is equivalent to

$$
\max_{\mathbf{a}_0,\ldots ,\mathbf{a}_K} -V^* (\mathbf{s}_0) + \beta \log \pi^* (\mathbf{a}_0|\mathbf{s}_0) + \ldots +\beta \log \pi^* (\mathbf{a}_K|\mathbf{s}_K) \tag{13}
$$

where $\pi^{*}$ is the corresponding optimal policy.

>  推导
>  根据 Eq 10，我们可以得到

$$
\begin{align}
\beta \log \frac {\pi^*(s_t, a_t)}{\pi_{\text{ref}}(s_t,a_t)} &= r(s_t,a_t) + V^*(s_{t+1}) - V^*(s_{t})\\
r(s_t,a_t) + \beta \log \pi_{\text{ref}}(s_t,a_t) &= \beta \log \pi^*(s_t, a_t) - (V^*(s_{t+1}) - V^*(s_t))
\end{align}
$$

>  因此我们有

$$
\begin{align}
&r(s_0, a_0) + \beta \log \pi_{\text{ref}}(s_0, a_0) + \cdots + r(s_t, a_t) + \beta \log \pi_{\text{ref}}(s_K, a_K) + V^*(s_{K+1})\\
=&\sum_{t=0}^{K}[r(s_t,a_t) + \beta \log \pi_{\text{ref}}(s_t, a_t)] + V^*(s_{K+1})\\
=&\sum_{t=0}^K[\beta \log \pi^*(s_t, a_t) - V^*(s_{t+1}) + V^*(s_t)] + V^*(s_{K+1})\\
=&\sum_{t=0}^K\beta \log \pi^*(s_t, a_t) + V^*(s_0) 
\end{align}
$$

>  其中伸缩求和指的就是求和序列中相邻两项可以互相抵消的求和
>  其中 $V^*(s_0)$ 的符号和原文不一样，应该是作者笔误，而且该状态价值和最大化没有关系，因为也无所谓符号
>  可以看到最优的搜索序列也就是依照最优策略选择出的序列

Now, since the starting state is fixed (it's given by the problem) we have that a search algorithm based on the conservative reward function of the RLHF objective and the corresponding optimal value policy is equivalent to likelihood search on the corresponding optimal policy. 
>  因为起始状态 $s_0$ 是固定的 ($V^*(s_0)$ 项可以忽略)，故这个基于 RLHF 得到的奖励函数的搜索算法 (也就是 Eq 12) 等价于基于对应的最优策略的似然搜索问题 (也就是 Eq 13)
>  换句话说，因为作者已经将奖励函数和最优策略关联，故基于奖励的搜索可以转化为基于最优策略的搜索，进而也就是关于最优策略的极大似然问题
>  那么实际上，模型得到的最优策略已经包含了执行高效搜索有关的价值信息，也就是没必要再这样多此一举地搜索，简单的 beam search 即可

We empirically verify this property in Fig. 2, which shows the win rate of DPO models trained with three different $\beta$ values against the preferred summary in the test dataset. We see that a 5-beam search improves win-rates by 10-15% over the base policy (1-beam), which is comparable to the value-function guided search improvements reported in Mudgal et al. (2024). 
>  Fig2 中经验上地验证了这个性质
>  该试验的目的是对 DPO 进行 beam search 来评估模型表现，发现 5-beam 比 1-beam (贪心) 提高了 10-15% 的胜率，证明了 beam-search 找到了更好的路径，而这个提升幅度和之前工作报告的价值函数引导的搜索的提升幅度是可比的

Interestingly, we see performance degrade with higher number of beams. Increasing the number of beams also produces answer with exploding length, which is a sign of reward over-optimization Gao et al. (2023); Park et al. (2024); Rafailov et al. (2024) and would explain the degradation in performance. 
>  但 beam 数量太大时，效果反而下降，且生成的文本变得很长，这可能是奖励过度优化的迹象 (模型找到了最大化奖励的路径，但这往往不是真实的最优路径)

These observations are consistent with out formulation of beam search as a search over a learned reward function.
>  这些观察都表明: DPO 学习到了隐含的奖励函数，而 beam search 就等价于基于这个奖励函数进行搜索

![](https://cdn-mineru.openxlab.org.cn/extract/59a4bfae-d140-4047-a800-2cfb72185b92/273c4536a357eea69fa64f4c8ad91eae51696ffb70ec14a28f74e30d77c34d87.jpg) 

Figure 2: Model performance using beam search. Left: Win rate of the model generated summaries over the preferred summary on 256 held-out test prompts from the Reddit TL;DR dataset, as evaluated by GPT 4. Right: The average answer length based on number of beams. We see exploding verbosity with more than 5 beams, which also leads to lower model win rates, despite GPT4's well-know preference length bias.

These findings are consistent with the result of the recently proposed V-STaR algorithm Hosseini et al. (2024), which combines the approach of STaR Zelikman et al. (2022) with a DPO trained verifier. At inference time, the STaR model produces several candidate reasoning chains (plans) which are ranked by the DPO verifier likelihood. This can be seen as a form of likelihood based search as in Eq. 12, however instead of directly searching on the DPO model, it uses the STaR model as a proposal distribution. We hypothesize this is beneficial in preventing reward hacking, which is potentially an issue with deeper search as shown in Fig. 2.

## 5.3 Connections Between Proxy Tuning and Reinforcement Learning
Several recent works Mitchell et al. (2023); Liu et al. (2024a,b) have proposed an approach of inference-time model alignment through a proxy guidance model. These approaches start with a (unaligned) base model $\pi_{\mathrm{base}}$ and a proxy model $\pi_{\mathrm{proxy}}$ and a target distribution reference model $\pi_{\mathrm{ref}}$ . 
>  最近一些工作讨论模型的推理时对齐 (不在训练时对齐，而在推理时动态调整模型的输出分布，以符合人类偏好)
>  这些工作通过代理引导模型来进行推理时对齐，通常用一个对齐的代理模型 $\pi_{\text{proxy}}$ (小模型) 来对齐没有对齐的基础模型 $\pi_{\text{base}}$ (大模型)，还会使用到一个目标分布参考模型 $\pi_{\text{ref}}$

The inference time re-alignment of the base model is carried by re-weighting the conditional probabilities of each token:

$$
\pi (\mathbf{a}|\mathbf{s}_t)\propto \pi_{\mathrm{base}}(\mathbf{a}|\mathbf{s}_t)\left(\frac{\pi_{\mathrm{proxy}}(\mathbf{a}|\mathbf{s}_t)}{\pi_{\mathrm{ref}}(\mathbf{a}|\mathbf{s}_t)}\right)^\beta \tag{14}
$$

>  对齐方式是对基础模型在推理时的概率输出进行重新加权
>  如果 proxy 认为某个动作相较于 ref 更为有可能，该权重就会大于 1，反之同理，这其实是一种类似于引导采样的方法

Under our considerations from the prior chapter, then this becomes equivalent to

$$
\pi (\mathbf{a}|\mathbf{s}_t)\propto \pi_{\mathrm{base}}(\mathbf{a}|\mathbf{s}_t)\exp (\beta (Q^* (\mathbf{s}_t,\mathbf{a}) -V^* (\mathbf{s}_t))) \tag{15}
$$

where $\beta (Q^{*}(\mathbf{s}_{t},\mathbf{a}) -V^{*}(\mathbf{s}_{t})$ is the optimal implicit advantage from the proxy tuning model. That is our theoretical results allows us to tie the realignment approaches of Mitchell et al. (2023); Liu et al. (2024a,b) to recent works which explicitly train critic models Mudgal et al. (2024) for token-level decoding.

> 而如果代理模型是符合 DPO 训练的最优策略，为我们可以将 Eq 14 写为 Eq 15 (感觉还是有笔误)
>  因此推理时对齐实际上就是乘上了一个优势函数的指数，因此这些看似使用模型比率进行对齐的方法，在数学上等价于使用训练一个价值网络来引导对齐

## 5.4 Likelihoods should decrease when using DPO.
A surface level interpretation of DPO would lead one to believe it increases the likelihood of chosen responses, while decreasing the likelihood of rejected responses. This however, does not account for a well observed phenomena in which the likelihood of the chosen responses actually decrease over time (Pal et al., 2024). 
>  表面上看，DPO 优化会提高被选中的 response 的似然，降低被拒绝的 response 的似然
>  但在实际中观察到，在 DPO 训练中，被选中 response 的似然会随着时间而下降
>  这就是一个矛盾: DPO 旨在提高被选中 response 的似然，但实际观察中这个似然却会下降

This is illustrated on the left half of fig. 3, which we show that when performing SFT before DPO, the implicit rewards of both the chosen and rejected response decline, though the margin between them increases. However, given a MaxEnt RL framing, this phenomena may be expected.
>  Fig 3 中展示了在 DPO 之前执行 SFT 时 (也就是到底用不用 SFT 之后的模型作为 DPO 的起点)，chosen response 和 rejected response 的隐式奖励都会下降，不过二者之间的边距增加 (说明绝对值下降也没关系，模型依然可以正确比较，正确选择)
>  在最大熵 RL 的框架下，这实际上是所期望的 (降低绝对奖励来换取熵)

![](https://cdn-mineru.openxlab.org.cn/extract/59a4bfae-d140-4047-a800-2cfb72185b92/a210a48a87a2b3554ba8c9ce171e3ff2622311117d1f3cb0a3829c1fd1ed623e.jpg) 

Figure 3: The evolution of implicit rewards for DPO on TLDR (left) and CPL on the bin-picking dataset (right) during training. We see that when we start with SFT, reward values decrease, whereas starting without SFT causes implicit rewards to be positive for DPO and increase for CPL.

Consider the expected log ratio (or implicit reward) of a policy under the reference model, which is often measured during training. Algebraic manipulation yields the following relationship:

$$
\mathbb{E}_{\mathbf{a}\sim \pi_{\mathrm{ref}}(\cdot |\mathbf{s})}\left[\beta \log \frac{\pi(\mathbf{a}|\mathbf{s})}{\pi_{\mathrm{ref}}(\mathbf{a}|\mathbf{s})}\right] = -\beta \mathbb{D}_{\mathrm{KL}}\left(\pi_{\mathrm{ref}}(\cdot |\mathbf{s})||\pi (\cdot |\mathbf{s})\right) \tag{16}
$$

>  我们考虑策略的隐式奖励相对于参考模型的期望，容易知道它和 KL 散度相关，如 Eq 16

At the beginning of training when $\pi = \pi_{\mathrm{ref}}$ , the implicit rewards are trivially zero. However at the end of training, assuming $\pi_{\mathrm{ref}}\neq \pi^{*}$ , the KL divergence is necessarily positive, indicating that the implicit rewards must decrease in expectation to converge. This means that the average implicit rewards should go down when starting from the SFT model. 
>  在训练开始时，$\pi = \pi_{\text{ref}}$，隐式奖励都是零，其期望也是零，在训练结束后 $\pi_{\text{ref}} \ne \pi^*$，KL 散度就为正，隐式奖励的期望为负
>  这说明隐式奖励必须降低，模型才能收敛到最优策略
>  因此，如果从 SFT 模型开始，平均的隐式奖励必须下降

In fact, on the left side of fig. 3 we show that when one does not SFT before DPO, there is little discernible trend in the average implicit reward and the implicit rewards of the chosen responses remain above zero. In fact, this trend also holds for CPL Hejna et al. (2024) for the general MDP, where the implicit rewards actually increase if SFT is not used.
>  Fig 3 left 展示了在 DPO 之前不使用 SFT，则平均隐式奖励的下降趋势不明显，并且 chosen response 的奖励高于零
>  在 CPL 中，也观察到如果不使用 SFT，隐式奖励会增大

One might realize that the previous analysis does not necessitate that the implicit rewards of the chosen must decrease, just that the implicit rewards must decrease on average. However, in practice it is common place (and recommended by Rafailov et al. (2023)) to SFT on only the chosen responses to form $\pi_{\mathrm{ref}}$ . For this section only we will call this choice of reference $\tau_{\mathrm{ref}}^{w}$ . Substituting $\pi_{\mathrm{ref}}^{w}$ into eq. (16), we can see that when SFTing on the positive answers the implicit rewards of the chosen responses must go down because at convergence as $\mathbb{E}_{\pi_{\mathrm{ref}}^{w}}[\beta \log \pi^{*} -\beta \log \pi_{\mathrm{ref}}^{w}] = -\beta \mathbb{D}_{\mathrm{KL}}(\pi_{\mathrm{ref}}^{w}||\pi^{*})$
>  上述讨论说明了隐式奖励的平均值必须下降才能收敛，但没有说明 chosen response 的隐式奖励必须下降
>  但在实践中，只对 chosen response 进行微调是常见的操作，而如果参考模型是仅针对 chosen response 进行 SFT 的，那么 $\pi_{\text{ref}}$ 的分布也就集中在 chosen response 上，那么 chosen response 的隐式奖励下降就是合理的

**Based on this derivation and choice of $\pi_{\mathrm{ref}}^{w}$ , the likelihood of the chosen response should decrease in the process of DPO training.**

While choosing $\pi_{\mathrm{ref}} = \pi_{\mathrm{ref}}^{w}$ is done in practice (Rafailov et al., 2023), it does mean that DPO will decrease the likelihood of all data in favor of extrapolated responses, which could cause over-fitting. Moreover, now that we have provided a derivation of DPO in the token-level MDP, one might expect it to exhibit characteristics like an RL algorithm - namely that the implied $Q$ -function monotonically increases over time. However, this is not necessarily the case. Note that per analysis in Section section 3.1, DPO can be viewed as adjusting the reward (or advantage) from which the optimal policy is deterministically mapped within the token-level MDP. DPO does not train a policy to maximize reward, and thus we do not argue about whether its implied value functions should increase or decrease over time.
>  1. 使用 $\pi_{\text{ref}} = \pi_{\text{ref}}^w$ 导致的过拟合: $\pi_{\text{ref}}^w$ 是只在人类偏好的 chosen response 上进行过 SFT 的参考模型，DPO 视图让模型去学习一个比 $\pi_{\text{ref}}^w$ 更好的策略，这需要让 $\pi^*$ 偏离 $\pi_{\text{ref}}^w$ (尽管 $\pi_{\text{ref}}^w$ 已经非常擅长生成 chosen response)，这个偏离会在数学上导致 KL 散度增加，从而使训练数据的平均概率下降；而 DPO 训练模型时，不仅会学习 “好” 的例子，还会学习到一个奖励函数，然后根据这个奖励函数来生成响应，“推断响应” 就是指模型根据学到的奖励函数生成的响应，这些响应不在训练数据中出现；在推理过程中，模型遇到了新 prompt，就需要生成推断响应，如果模型过拟合了奖励，可能生成的推断响应虽然最大化了奖励，但实际上并不是想要的最好响应
>  2. DPO 和传统 RL 算法的区别: DPO 并不属于 RL 算法，RL 算法中，Q function 是随着时间单调增长的 (因为会学习到更好的策略)，但在 DPO 中则不是这样，因为 DPO 是在同时调整奖励 (优势)，从这个调整后的奖励来映射出最优策略，而不是训练一个最大化给定奖励的策略，因此，我们没有必要讨论是否隐式的价值函数是随着时间增长还是下降，因为这就不是 DPO 的训练目标，DPO 的训练目标是和数据集对齐，而对齐就会导致 KL 散度下降

# 6 Discussion
In this work we formulated the DPO optimization algorithm as learning an optimal Q function, which is represented by an LLM. This formulation and our results provide theoretical justification for empirically observed DPO training phenomena, which are not explained by the original bandit formulation. 
>  本文将 DPO 优化算法构建为学习一个最优 Q-function 的问题，其中 Q-function 可以用 LLM 策略表示

We further link and unify a family of proposed new LLM search algorithms by likelihood search under DPO and show comparable empirical gains by a simple 1-line code change to using beam search. 
>  我们进一步将 LLM 搜索算法联系到了 DPO 下的基于似然的搜索

Most importantly, we show qualitative early signs that DPO is able to learn credit assignment directly from feedback data. While larger-scale empirical exploration is necessary, we believe this an encouraging early sign. Our results indicate a number of promising future directions to explore:
>  我们定性地展示了 DPO 可以直接从反馈数据学习信用分配

**Learning intermediate reasoning from outcome feedback:** Recent works have shown promising results on that front Pang et al. (2024); Hwang et al. (2024).

**Multi-turn conversations:** Teaching language models to be an interactive conversationalists has been difficult, as RLHF is optimized as a single-turn bandit formulation. Moreover, classical methods, such as PPO, are not applicable in this setting. Recent work by Astukuri et al. (2024) has shown success in this domain using STaR and extending DPO to multi-turn conversational trees is a promising direction.

**Agentic LLMs:** LLM agents, such as WebGPT (Nakano et al., 2022) are equipped to take autonomous actions, such as browsing the Web and collecting information before providing an answer. The user then provides feedback based on the final output. Our derivations indicate that DPO training (on the full model trajectories) could learn optimal exploration behaviour. Recent works Song et al. (2024); Xi et al. (2024) shows promise in that direction.

**End-to-end training of generative AI systems:** Modern image generation systems, such as Dalle 3 Betker et al. (2023) use an LLM to produce high quality conditioning before calling a diffusion generation model. Also, recent long-form video generation models Hu et al. (2023); Gupta et al. (2023) combine transformer-based auto-regressive generations with a diffusion-based decoder. Such systems could potentially be optimized end-to-end with a hybrid version of DPO. We expand on these points in the Appendix.

We believe these are promising directions for future work.

# A Proof of Lemma 1
**Lemma 1.** For a fixed policy $\pi$ , there is a bijection between reward functions $r$ and corresponding optimal $Q$ -functions $(Q^{*})$ in the deterministic tree-structured LLM MDP.
>  **Lemma 1**
>  在确定性树状结构的 LLM MDP 中，给定一个固定的策略，奖励函数和对应的最优 Q-function 之间存在双射关系

>  确定性树状结构的 LLM MDP 指表示 LLM 生成过程的 Markov 决策过程，这个过程中，环境的转移是确定性的，也就是给定状态 (当前序列) 和动作 (新 token)，下一个状态是确定性的 (新序列 = 当前序列 + 新 token)，这个 MDP 过程是树状的，意味着从初始状态开始，每执行一个动作都会导向一个唯一的后续状态，并且没有循环
>  树状结构保证了 MDP 是有限时间步的，并且可以通过反向归纳法 (从终止状态向前) 求解

**Proof.** Let $Q^{*}$ denote the optimal $Q$ -function for reward $r$ . We prove the statement directly, starting with the injective case.

Assume there exists a reward function $r^{\prime}\neq r$ such that $Q_{r^{\prime}}^{*} = Q_{r}^{*}$ .Then, there must exist a state action pair such that $r^{\prime}(\mathbf{s}_t,\mathbf{a}_t)\neq r(\mathbf{s}_t,\mathbf{a}_t)$ .In fact, proceeding backwards from a leaf node (terminal state), there must be a first state action pair $(\mathbf{s}_t,\mathbf{a}_t)$ where $r^{\prime}(\mathbf{s}_t,\mathbf{a}_t)\neq r(\mathbf{s}_t,\mathbf{a}_t)$ . The $Q$ functions at this location are

$$
Q_{r^{\prime}}^{*}(\mathbf{s}_{t},\mathbf{a}_{t}) = r^{\prime}(\mathbf{s}_{t},\mathbf{a}_{t}) + V_{r^{\prime}}^{*}(\mathbf{s}_{t + 1}),\quad Q_{r}^{*}(\mathbf{s}_{t},\mathbf{a}_{t}) = r(\mathbf{s}_{t},\mathbf{a}_{t}) + V_{r}^{*}(\mathbf{s}_{t + 1})
$$

By the fact that this was the first location where the reward functions differed starting from a leaf node, we must have that $V_{r^{\prime}}^{*}(\mathbf{s}_{t + 1}) = V_{r}^{*}(\mathbf{s}_{t + 1})$ . This is because we can recursively. solve for the optimal policy, value, and $Q$ function using eq. (5) eq. (7), and eq. (6) from Ziebart et al. (2008). The rewards in all possible future states from $s,a$ are equal by virtue of this being the location of the first difference and thus the dynamic programming solution up to this point is the same. Thus, we can see that $Q_{r^{\prime}}^{*}(\mathbf{s}_t,\mathbf{a}_t)\neq Q_{r}^{*}(\mathbf{s}_t,\mathbf{a}_t)$ , completing this direction. 

>  先证明内射: 如果存在两个不同的奖励函数，则它们对应的最优 Q-function 也必然不同，即不同的奖励必然导致不同的最优 Q-function
>  证明方法为反证法，假设存在两个奖励函数 $r'\ne r$，它们具有相同的最优 Q-function $Q^*_{r'} = Q^*_r$
>  该假设成立的前提下，那么必然存在一个状态动作对 $(s_t, a_t)$，使得 $r'(s_t, a_t) \ne r(s_t, a_t)$，我们假设这个状态动作对是从一个叶节点 (终止状态) 开始向前回溯，第一次遇到的 $r'(s_t, a_t) \ne r(s_t, a_t)$ 的状态动作对
>  那么 $Q^*_{r'}$ 和 $Q_r^*$ 为状态动作对 $(s_t, a_t)$ 给出的 Q-value 分别为

$$
Q_{r'}^*(s_t, a_t) = r'(s_t, a_t) + V_{r'}^*(s_{t+1}), \qquad Q_r^*(s_t, a_t) = r(s_t, a_t) + V_r^*(s_{t+1})
$$

>  因为 $(s_t, a_t)$ 是从后往前回溯，$Q_{r'}^*$ 和 $Q_r^*$ 第一次出现分歧的节点，那么在这一点上，$V_{r'}^*(s_{t+1}) = V_{r}^*(s_{t+1})$ 仍然成立
>  那么将 $V_{r'}^*(s_{t+1}) = V_r^*(s_{t+1})$ 代入上面的式子，很容易得到 $Q_{r'}^*(s_t, a_t) \ne Q_r^*(s_t, a_t)$，进而和最初的假设矛盾，证明完毕

Note that this proof does not hold in general MDPs, only the token MDP where it is impossible to return to the same state after taking any number of actions.
>  注意，这个证明仅适用于特定的 MDP，即无环 MDP，因为我们在证明时使用了 “从叶子节点回溯” ，这利用了无环 MDP 中没有循环的特性，即无论再执行多少次动作，都永远不会回到相同的状态

The surjective direction is easier. For all $Q^{*}$ , we can compute a reward function $r(\mathbf{s}_t,\mathbf{a}_t) =$ $Q^{*}(\mathbf{s}_{t},\mathbf{a}_{t}) -V^{*}(\mathbf{s}_{t + 1})$ under deterministic dynamics. Thus, we can see that the mapping is surjective.
>  再证明满射: 任何最优的 $Q^*$ 函数都可以由某个奖励函数生成
>  证明: 在确定性动态下，方程 $r(s_t, a_t) = Q^*(s_t, a_t) - V^*(s_{t+1})$ 始终满足，故无论是何种的 $Q^*$ ，显然都存在到奖励函数 $r$ 的映射关系

# B Treatment of Diffusion Models
Conditional diffusion image generation models, such as Stable Diffusion 3 Esser et al. (2024) have also used a form of the DPO algorithm as outlined in Wallace et al. (2023). Our analysis can no longer be directly applied in that setting, since the generations are continuous. However, we could translate many of our results to that setting, if we consider a certain diffusion MDP. We outline our results below.
>  作者要考虑对扩散过程进行偏好微调的情况

## B.1 Diffusion MDP
We borrow the denoising MDP formulation from Black et al. (2023b); Fan et al. (2023). We again have the tuple $(S,A,f,r,\rho_0)$ , with the same formulation as in Section 3.1. At the same time consider a diffusion process with time index $t$ and $T$ total steps, conditioned on context $\mathbf {c}$ and image denoted by $\mathbf{x}_t$ . 
> 考虑一个去噪 MDP 过程，用 $(\mathcal S, \mathcal A, f, r ,\rho_0)$ 描述，它对应于一个时间步 $T$，条件于 context $\mathbf c$，以及图像 $\mathbf x_t$

Then we can map the diffusion generation process to an MDP in the following way

$$
\mathbf{s}_t = \left\{ \begin{array}{ll}(\pmb {c},T) & \mathrm{if~}t = 0\\ (\pmb {c},\pmb{x}_{T -t},T -t) & \mathrm{otherwise} \end{array} \right.
$$

That is the initial state consists of the prompt $\mathbf {c}$ and afterwards each state consists of the current denoised image $\mathbf{x}_t$ and time step $T -t$ .
> 扩散 MDP 中的状态定义如上，初始状态仅包含 prompt $\mathbf c$，后续的状态包含了 prompt $\mathbf c$ 和当前去噪的图像 $\mathbf x_t$

Notice that the time-steps in the MDP are inverse to the direction of the diffusion process (i.e. we start at noise and end at the final image). The action is just the next image iteration, from where the dynamics is also straightforward:

$$
\begin{array}{c}{\mathbf{a}_t\triangleq \mathbf{x}_{T -t + 1}}\\ {f(\mathbf{s}_t = (\mathbf{c},\mathbf{x}_{T -t},T -t),\mathbf{a}_t) = (\mathbf{c},\mathbf{a}_t,T -t -1)} \end{array}
$$
Notice that in this case the policy is stochastic, but the dynamics of the MDP is still deterministic.

>  动作就是下一张图像，环境动态是确定性的，也就是将下一个状态的图像替换为 $\mathbf a_t$

 Finally, the initial distribution, is just the distribution of prompts:

$$
\rho (\mathbf{s}_0)\triangleq (p(\pmb {c}),0)
$$

>  初始状态分布就是 prompts 的分布

## B.2 Theoretical Results for the Diffusion MDP
Given the above formulation, we can also prove that Lemma 1 also holds in the diffusion MDP.
>  在上述的构造下，Lemma 1 (最优 Q-function 和奖励函数存在一一对应关系) 在扩散 MDP 下仍然成立

**Lemma 2.** Under mild assumptions, there is a bijection between reward functions $r(\mathbf{s}_t,\mathbf{a}_t)$ and corresponding optimal $Q$ -functions $Q^{*}(\mathbf{s}_t,\mathbf{a}_t)$ in the diffusion MDP.
>  **Lemma 2**
>  在扩散 MDP 下，奖励函数 $r(s_t, a_t)$ 和对应的最优 Q-function $Q^*(s_t, a_t)$ 之间存在双射关系

**Proof.** Since the MDP still has deterministic dynamics, we have that Eq. 5-7 still hold. Now, given a reference policy $\pi_{\mathrm{ref}}$ , parameter $\beta$ and a critic $Q$ , we can trivially recover the unique reward function by inverting Eq. 7. 
>  **Proof**
>  因为 MDP 是确定性的，故 Eq 5-7 仍然成立
>  给定参考策略 $\pi_{\text{ref}}$ ，参数 $\beta$，critic $Q$，通过 Eq 7 ，我们可以表示出和 $Q$ 相关的，唯一的奖励函数

We will prove that given a reward function $r(\mathbf{s}_t,\mathbf{a}_t)$ we can recover a unique critic $Q$ .We work inductively in the diffusion MDP starting with $t = T$ , where we have $V^{*}(\mathbf{a}_{T}) = 0$ for all terminal states. We then have that

$$
\begin{align} 
& {Q^{*}(\mathbf{s}_{t -1},\mathbf{a}_{t -1}) = Q^{*}(\mathbf{s}_{t} = (\mathbf{c},\mathbf{x}_{T -t + 1},T -t + 1),\mathbf{a} = \mathbf{x}_{T -t}) =}\\ 
& {r(\mathbf{s}_{t} = (\mathbf{c},\mathbf{x}_{T -t + 1},T -t + 1),\mathbf{a} = \mathbf{x}_{T -t}) + \beta \log p_{ref}(\mathbf{x}_{T -t}|\mathbf{c},\mathbf{x}_{T -t + 1},T -t + 1) + }\\ 
& {\beta \log \int_{A}e^{Q^{*}(\mathbf{s}_{t} = (\mathbf{c},\mathbf{x}_{T -t},T -t),\mathbf{x}_{T -t -1}) / \beta}d\mathbf{x}_{T -t -1}} \end{align}
$$

where $\pi_{ref}$ is the reference backwards diffusion process.
>  我们接着证明给定奖励函数 $r(s_t, a_t)$，可以对应到唯一的 critic $Q$
>  我们从 $t= T$ 归纳式地进行推导
>  其中 $\pi_{ref}$ 是参考反扩散过程

In this case even though the state space is deterministic, our approach to the proof of Lemma 1 still holds by using backwards induction on the diffusion step $t$ . Notice, that from $V(\mathbf{s}_T = (\mathbf{c},\mathbf{x}_0,0)) = 0$ we can uniquely determine the critic values for all states at time step $T -1$ . Proceeding inductively backwards through time in the MDP/denoising process (forward in the diffusion process), we obtain the desired result.
>  通过对扩散时间步反向归纳，我们可以通过 $\pi_{ref}$ 和 $r$ 唯一地恢复所有状态-动作对的 Q-function

Given the proof of this Lemma, we can then directly apply the results of Section 4.2, including Theorem 1. Our results, also give us insights into the formulation of Wallace et al. (2023). In particular, by changing the sampling scheme of the intermediate diffusion the authors obtain two alternative formulations (Appendix S2 in Wallace et al. (2023)). Both of these schemes are suggested as empirical approximations in the formulation of the Diffusion-DPO algorithm, however in the view of Q-learning both of these are valid approaches to generating off-policy data. In fact, our interpretation of DPO allows for general off-policy data sampling and aggregation methods, which could yield a whole family of DPO algorithms in this domain. We leave the exploration of this direction for further work.
>  证明了扩散 MDP 中奖励函数和最优 Q-function 的一一对应性质，就可以利用 DPO 的结论，将奖励函数用策略表示

# C Reddit TL;DR Posts

# D End-to-End Training of Generative AI Systems
**End-to-end system:** Here, we present an outline for end-to-end training of generative AI systems from human feedback using DPO in multi-step MDP. We assume two models - a prompt refiner $\pi_{\theta}(\mathbf{z}|\mathbf{x})$ , which is a language model, generating discrete tokens in an autoregressive way, which, given a prompt $\mathbf{x}$ produces a refined prompt $\mathbf{z}$ . This prompt is then fed into an image-generation diffusion model $\pi_{\phi}(\mathbf{y}|\mathbf{z})$ , (which we parameterize as a denoising model $\epsilon_{\phi}$ ), which generates the final image. A real case example of that system is shown in Figure 4.

![](https://cdn-mineru.openxlab.org.cn/extract/59a4bfae-d140-4047-a800-2cfb72185b92/5a7746693bee0407a5f20e4ec592fb131e117c50e2303878b011309600c4ec03.jpg) 

Figure 4: Example of an end-to-end generative AI workflow. The user request an image of a Minotaur in the streets of New York. However, we see that the rejected image does not actually represent a Minotaur, but a bull. The prompt refiner generates a valid description and specifically includes the phrasing "the body of a man and the head of a bull", but the image generation model fails to follow that prompt. At the same time in the case of the chosen image (which does reflect the prompt) the prompt refiner generates a more descriptive text, which the image generator is able to follow more closely. While it is possible to train each component separately, joint training can in theory, optimize the refiner to generate prompts that the image generator can more closely realize, while also training the generator to directly produce more aligned images.

**Feedback generation:** During the feedback gathering stage, a user provides a query $\mathbf{x}$ and two refined prompts are sampled from $\mathbf{z}^1, \mathbf{z}^2 \sim \pi_{\theta}(\mathbf{x}|\mathbf{z})$ , which the user does not directly evaluate. Then, the image generation model generates two images $\mathbf{y}^i \sim \pi_{\phi}(\mathbf{y}|\mathbf{z}^i)$ for $i = 1,2$ . The user then provides a preference over the $\mathbf{y}^i$ to yield the preference pair $\{\mathbf{y}^w, \mathbf{z}^w \succ \mathbf{y}^l, \mathbf{z}^l |\mathbf{x}\}$

**Optimization:** Optimization is carried in a hybrid MDP, where the initial state is the prompt $\mathbf{x}$ and has the same form as the token MDP, as outlined in 3.1. When the EOS token is encountered in that MDP, at which point the transition dynamics switches to the diffusion denoising MDP introduced in Black et al. (2023a). Notice that this is still a valid MDP and all the conclusions of our derivations in the main section of our paper hold. We could then optimize this system end-to-end using a hybrid DPO objective, combining our results, with those presented in Wallace et al. (2023) we have:

$$
\begin{array}{r l} & {r_{\theta ,\phi}(\mathbf{x}^{w},\mathbf{y}^{w}) = \beta \sum_{i = 0}^{|\mathbf{z}^{w}|}\underset {\mathrm{prompt~refiner~MDP}}{\log}\frac{\pi_{\theta}(\mathbf{z}_{i}^{w}|\mathbf{x},\mathbf{z}_{< i}^{w})}{\pi_{\mathrm{ref}}(\mathbf{z}_{i}^{w}|\mathbf{x},\mathbf{z}_{< i}^{w})} +}\\ & {\qquad \gamma T\omega (\lambda_{t})\mathbb{E}_{t\sim \mathcal{U}(0,T)}\bigg[\underset {\mathrm{diffusion~MDP~objective}}{\underbrace{(\|e^{w} -\epsilon_{\mathrm{ref}}(\mathbf{y}_{t}^{w},\mathbf{z}^{w},t)\|_{2}^{2} -\|e^{w} -\epsilon_{\phi}(\mathbf{y}_{t}^{w},\mathbf{z}^{w},t)\|_{2}^{2})}}\bigg]} \end{array} \tag{17}
$$

where $\beta$ and $\gamma$ are two separate discounting factors for each modality. Here the diffusion objective follows directly from the derivation of Eq. 9 and the sampling scheme proposed in Wallace et al. (2023) (Eq. 12-14 in that paper). Notice here that the image generation model is conditioned on the corresponding refined prompt $\mathbf{z}^w$ . We can define $r(\mathbf{x}^l, \mathbf{y}^l)$ similarly and optimize the DPO objective:

$$
\mathcal{L}_{\mathrm{DPO}_{\theta ,\phi}} = -\mathbb{E}_{(\mathbf{x},\mathbf{y}^w,\mathbf{y}^l)\sim \mathcal{D}}\left[\log \sigma \left(r_{\theta ,\phi}(\mathbf{x}^w,\mathbf{y}^w) -r_{\theta ,\phi}(\mathbf{x}^l,\mathbf{y}^l)\right)\right] \tag{18}
$$

We demonstrate a particular real use of such a system in Figure 4. The user request an image of a Minotaur in the streets of New York. However, we see that the rejected image does not actually represent a Minotaur, but a bull. The prompt refiner generates a valid description and specifically includes the phrasing "the body of a man and the head of a bull", but the image generation model fails to follow that prompt. At the same time in the case of the chosen image (which does reflect the prompt) the prompt refiner generates a more descriptive text, which the image generator is able to follow more closely. While it is possible to train each component separately, joint training can in theory, optimize the refiner to generate prompts that the image generator can more closely realize, while also training the generator to directly produce more aligned images.

>  作者把 DPO 的思想拓展到任意的偏好拟合的场景

## D.1 Hybrid Video Generative Models
A recent line of work on long-form video generation Hu et al. (2023); Gupta et al. (2023) by combining auto-regressive transformer generation with a diffusion model decoding or uspcaling of the actual video frames to obtain temporally consistent and high-fidelity generations. We could deploy similar RLHF pipelines to the video-generation problem as well. It is also straightforward to extend the DPO joint optimization framework, presented in the previous section in Eq. 18 to this setting as well. Instead of textual prompt refiner tokens, the variables $\mathbf{z}$ would represent the latent token generations of the autoregressive component $\pi_{\theta}$ , which would be decoded into actual video frames $\mathbf{y}$ via the diffusion decoder $\pi_{\phi}$ . We believe this is an exciting direction to pursue for the emerging video generation technologies.

# E Beam Search Trends for PPO
We consider whether the beam search trends in Figure 2 hold for PPO, since PPO is known to exhibit reward over-optimization as well (Gao et al. (2023)). We use a PPO-tuned GPT-J-6B model released by CarperAI $^1$ , fine-tuned on the same TL;DR dataset. We use the same sampling parameters as with the DPO experiments ( $\tau = 1.0$ , $k = 50$ ) and generate 256 samples from test set prompts, with GPT-4 as the evaluator. We only report results for 1, 2, 5, and 7 beams; higher number of beams were tried but exhausted memory on an NVIDIA A40 GPU.

![](https://cdn-mineru.openxlab.org.cn/extract/59a4bfae-d140-4047-a800-2cfb72185b92/cc886991b919e1d01a3d65f7eedae9d3eea15a4c91ca91e4f00fe2ba32736bf5.jpg) 

Figure 5: PPO model performance using beam search. Left: Win-rate of model summaries over preferred summary. Right: Average answer length based on number of beams.

In Figure 5, we observe a similar over-optimization phenomenon as with DPO, with 2 beams both increasing winrate and decreasing sample length (even under the length-biased evaluator GPT-4). However, more than 2 beams leads to a decline in downstream performance and large uptick in sample length, similar behavior to DPO in the over-optimization regime. We find that the benefits of increasing the beam size are more limited when using this model.

# F TL;DR Sample Generations
## F.1 Samples Across Different Beam Searches
## F.2 Samples from DPO with and without SFT Pre-Training


# G Dataset and Hyperparameter Details
The TL;DR dataset contains 64,832 summary comparisons and is derived from the Webis TLDR dataset, with human feedback collected by OpenAI. Each example is scraped from Reddit and belongs to one of several "subreddits" (topic forums), with an associated title/post/human TL;DR (summary). Around $5\%$ is held out for validation.

Each TL;DR model was a pre-trained Pythia 2.8B model. We first SFT with a learning rate of $0.5 \times 10^{-6}$ and batch size of 128 for one epoch using 8 gradient accumulation steps on 4 NVIDIA A40 GPUs. All DPO models with the same setup and hyperparameters as the original paper (Rafailev et al., 2023). Specifically, we train for 1 epoch with a learning rate of $0.5 \times 10^{-6}$ and linear warmup of 150 steps. We use a batch size of 32 (example example is a positive and a negative) and clip gradient norms to 10. We use $\beta = 0.1$ for DPO.

All generation samples are performed with temperature 1.0 and max length 512 unless otherwise specified. Evaluations (winrates, lengths, etc) are computed with 256 samples from the held-out set in TL;DR.

# H GPT 4 Evaluation
We use the following prompt in all our GPT 4 evaluations. The order of the model sample and the reference response is randomized in each evaluation to avoid positional bias in the judge.

```
Which of the following summaries does a better job of summarizing the most important points in the given forum post?
Post: QUERY
Summary A: A
Summary B: B
FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:

Comparison: <one-sentence comparison and explanation> Preferred: <"A" or "B">
```

