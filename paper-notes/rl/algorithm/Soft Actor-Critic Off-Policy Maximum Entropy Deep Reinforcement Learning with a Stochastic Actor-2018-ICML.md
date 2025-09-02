# Abstract 
Model-free deep reinforcement learning (RL) algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties, which necessitate meticulous hyperparameter tuning. 
>  model-free RL 算法存在高样本复杂度和弱收敛性质两大问题，故需要进行细致的超参数调整

> [!info] 样本复杂度
> 样本复杂度衡量了一个算法需要多少样本才能达到一定的学习效果或性能水平
> RL 中，样本复杂度通常指智能体需要与环境进行多少次交互 (即获取多少次样本) 才能学会一个有效的策略

Both of these challenges severely limit the applicability of such methods to complex, real-world domains. In this paper, we propose soft actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework. In this framework, the actor aims to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible.
>  本文提出 soft actor-critic, 这是一个基于最大熵的异策略 actor-critic 框架
>  在该框架中，actor 的目标是在最大化期望奖励的同时也最大化熵值，即在完成成功完成任务的同时尽可能随机地行动

 Prior deep RL methods based on this framework have been formulated as Q-learning methods. By combining off-policy updates with a stable stochastic actor-critic formulation, our method achieves state-of-the-art performance on a range of continuous control benchmark tasks, outperforming prior on-policy and off-policy methods. 
 > 之前的基于该框架的 RL 方法已经被表述为 Q-Learning 方法
 > 我们的方法将异策略更新和稳定的随机 actor-critic 公式结合，在一系列连续控制任务中达到 SOTA，优于之前的同策略和异策略方法

Furthermore, we demonstrate that, in contrast to other off-policy algorithms, our approach is very stable, achieving very similar performance across different random seeds. 
>  此外，相较于其他异策略算法，我们的方法非常稳定，在不同随机种子下达到几乎相同的表现

# 1 Introduction 
Model-free deep reinforcement learning (RL) algorithms have been applied in a range of challenging domains, from games (Mnih et al., 2013; Silver et al., 2016) to robotic control (Schulman et al., 2015). The combination of RL and high-capacity function approximators such as neural networks holds the promise of automating a wide range of decision making and control tasks, but widespread adoption of these methods in real-world domains has been hampered by two major challenges. 
>  model-free 的深度 DL 算法存在两大限制

First, model-free deep RL methods are notoriously expensive in terms of their sample complexity. Even relatively simple tasks can require millions of steps of data collection, and complex behaviors with high-dimensional observations might need substantially more. Second, these methods are often brittle with respect to their hyperparameters: learning rates, exploration constants, and other settings must be set carefully for different problem settings to achieve good results. Both of these challenges severely limit the applicability of model-free deep RL to real-world tasks. 
>  其一，其样本复杂度太高，即便是相对简单的任务也需要数百万步数据收集
>  其二，对超参数敏感，例如 learning rate, exploration constant 等需要根据问题仔细设定

One cause for the poor sample efficiency of deep RL methods is on-policy learning: some of the most commonly used deep RL algorithms, such as TRPO (Schulman et al., 2015), PPO (Schulman et al., 2017b) or A3C (Mnih et al., 2016), require new samples to be collected for each gradient step. This quickly becomes extravagantly expensive, as the number of gradient steps and samples per step needed to learn an effective policy increases with task complexity. 
>  导致样本效率低的一大原因是同策略学习，例如 TRPO, PPO, A3C 都要求在每一个梯度步收集新样本
>  因为随着任务复杂性上升，所需要的梯度步数量以及每个梯度步需要的样本数量都需要提高，同策略方法的样本复杂度就会随着任务难度迅速提高

Off-policy algorithms aim to reuse past experience. This is not directly feasible with conventional policy gradient formulations, but is relatively straightforward for Q-learning based methods (Mnih et al., 2015). Unfortunately, the combination of off-policy learning and high-dimensional, nonlinear function approximation with neural networks presents a major challenge for stability and convergence (Bhatnagar et al., 2009). This challenge is further exacerbated in continuous state and action spaces, where a separate actor network is often used to perform the maximization in Q-learning. A commonly used algorithm in such settings, deep deterministic policy gradient (DDPG) (Lillicrap et al., 2015), provides for sample-efficient learning but is notoriously challenging to use due to its extreme brittleness and hyperparameter sensitivity (Duan et al., 2016; Henderson et al., 2017). 
>  异策略算法不能直接用于常规的策略梯度形式，但适用于基于 Q-Learning 的方法
>  但异策略学习使用神经网络作为近似器时，存在稳定性和收敛性的问题，在连续动作和状态空间的情况下，这类问题会更加严重
>  DDPG 使用一个分离的 actor 网络执行 Q-Learning 中的 maximization，其学习是参数高效的，但 DDPG 对超参数很敏感

>  因为 DDPG 使用确定性策略，缺乏探索 (可能把 DDPG 的行为策略定义得探索性更大一点会就可以解决这个问题)，故学习的泛化性较低，对超参数很敏感

We explore how to design an efficient and stable model-free deep RL algorithm for continuous state and action spaces. To that end, we draw on the maximum entropy framework, which augments the standard maximum reward reinforcement learning objective with an entropy maximization term (Ziebart et al., 2008; Toussaint, 2009; Rawlik et al., 2012; Fox et al., 2016; Haarnoja et al., 2017). 
>  我们探索为连续的状态和动作空间设计高效且稳定的 model-free deep RL 算法
>  为此，我们对标准的最大奖励 RL 目标进行改进，添加一个熵最大化项

Maximum entropy reinforcement learning alters the RL objective, though the original objective can be recovered using a temperature parameter (Haarnoja et al., 2017). More importantly, the maximum entropy formulation provides a substantial improvement in exploration and robustness: as discussed by Ziebart (2010), maximum entropy policies are robust in the face of model and estimation errors, and as demonstrated by (Haarnoja et al., 2017), they improve exploration by acquiring diverse behaviors. 
>  最大熵 RL 方法修改了 RL 目标，且可以通过 temperature 参数衡量权重
>  最大熵的引入可以提高探索和健壮性，策略对于模型 (环境) 和估计误差是健壮的

Prior work has proposed model-free deep RL algorithms that perform on-policy learning with entropy maximization (O’Donoghue et al., 2016), as well as off-policy methods based on soft Q-learning and its variants (Schulman et al., 2017a; Nachum et al., 2017a; Haarnoja et al., 2017). However, the on-policy variants suffer from poor sample complexity for the reasons discussed above, while the off-policy variants require complex approximate inference procedures in continuous action spaces. 
>  之前已经有工作提出同策略下的带有最大熵的 model-free RL 算法，以及基于 soft Q-Learning 的异策略方法及其变体
>  其中同策略方法样本复杂性高，soft Q-Learning 的异策略方法及其变体在连续动作空间需要执行复杂的近似推理 (以找到最大化 Q 的动作)

In this paper, we demonstrate that we can devise an off-policy maximum entropy actor-critic algorithm, which we call soft actor-critic (SAC), which provides for both sample-efficient learning and stability. 
>  本文设计了异策略最大熵 actor-critic 算法，我们称为 soft actor-critic，该算法是样本高效并且稳定的

This algorithm extends readily to very complex, high-dimensional tasks, such as the Humanoid benchmark (Duan et al., 2016) with 21 action dimensions, where off-policy methods such as DDPG typically struggle to obtain good results (Gu et al., 2016). SAC also avoids the complexity and potential instability associated with approximate inference in prior off-policy maximum entropy algorithms based on soft Q-learning (Haarnoja et al., 2017). 

We present a convergence proof for policy iteration in the maximum entropy framework, and then introduce a new algorithm based on an approximation to this procedure that can be practically implemented with deep neural networks, which we call soft actor-critic. 
>  我们为最大熵框架下的策略迭代的收敛性提供了证明，然后基于该迭代过程的近似构造 SAC 算法

We present empirical results that show that soft actor-critic attains a substantial improvement in both performance and sample efficiency over both off-policy and on-policy prior methods. We also compare to twin delayed deep deterministic (TD3) policy gradient algorithm (Fujimoto et al., 2018), which is a concurrent work that proposes a deterministic algorithm that substantially improves on DDPG. 
>  SAC 在样本效率和性能上都优于之前的同策略和异策略方法

# 2 Related Work 
Our soft actor-critic algorithm incorporates three key ingredients: an actor-critic architecture with separate policy and value function networks, an off-policy formulation that enables reuse of previously collected data for efficiency, and entropy maximization to enable stability and exploration. 
>  SAC 包括了三个关键要素: 
>  - actor-critic 结构，分离了策略和价值网络
>  - 一个 off-policy 公式，以高效复用之前收集的数据
>  - 最大化熵，以实现稳定性和探索

We review prior works that draw on some of these ideas in this section. Actor-critic algorithms are typically derived starting from policy iteration, which alternates between policy evaluation—computing the value function for a policy— and policy improvement—using the value function to obtain a better policy (Barto et al., 1983; Sutton & Barto, 1998). In large-scale reinforcement learning problems, it is typically impractical to run either of these steps to convergence, and instead the value function and policy are optimized jointly. In this case, the policy is referred to as the actor, and the value function as the critic. 
>  actor-critic 算法通常推导自策略迭代，策略迭代在策略评估——计算当前策略价值函数和策略改进——使用价值函数获得更优策略之间迭代
>  在大规模情况下，通常无法将这两个步骤收敛到最优解，价值函数和策略通常会被联合优化，在该情况下，策略称为 actor，价值函数称为 critic

Many actor-critic algorithms build on the standard, on-policy policy gradient formulation to update the actor (Peters & Schaal, 2008), and many of them also consider the entropy of the policy, but instead of maximizing the entropy, they use it as an regularizer (Schulman et al., 2017b; 2015; Mnih et al., 2016; Gruslys et al., 2017). On-policy training tends to improve stability but results in poor sample complexity. 
>  许多 actor-critic 算法基于标准的 on-policy 策略梯度公式构建，并且其中许多算法也会考虑策略的熵，但它们不是最大化熵，而是将其作为正则化项使用
>  on-policy 训练会提高稳定性，但样本复杂度高

There have been efforts to increase the sample efficiency while retaining robustness by incorporating off-policy samples and by using higher order variance reduction techniques (O’Donoghue et al., 2016; Gu et al., 2016). However, fully off-policy algorithms still attain better efficiency. 
>  为了在保持健壮性的同时提高样本效率，有方法尝试引入 off-policy 样本，同时使用高阶的方差减小技术，但是这类方法的样本效率仍然低于完全的 off-policy 算法

A particularly popular off-policy actor-critic method, DDPG (Lillicrap et al., 2015), which is a deep variant of the deterministic policy gradient (Silver et al., 2014) algorithm, uses a Q-function estimator to enable off-policy learning, and a deterministic actor that maximizes this Q-function. As such, this method can be viewed both as a deterministic actor-critic algorithm and an approximate Q-learning algorithm. 
>  一个特别流行的 off-policy actor-critic 方法 DDPG，属于确定性策略梯度算法的变体，使用 Q-function 估计器以支持 off-policy 学习，并且使用一个最大化该 Q-function 的确定性 actor
>  因此，DDPG 既可以被视作一个确定性的 actor-critic 算法，也可以被视作一个近似的 Q-learning 算法 (从 DDPG 的 actor 是在近似最大化 Q-function 的角度出发，确实可以将 DDPG 视作一个近似的 Q-learning 算法)

Unfortunately, the interplay between the deterministic actor network and the Q-function typically makes DDPG extremely difficult to stabilize and brittle to hyperparameter settings (Duan et al., 2016; Henderson et al., 2017).  As a consequence, it is difficult to extend DDPG to complex, high-dimensional tasks, and on-policy policy gradient methods still tend to produce the best results in such settings (Gu et al., 2016). 
>  不幸的是，确定性 actor 和 Q-function 的相互作用通常使得 DDPG 极难稳定，并且对超参数设置非常敏感
>  因此，将 DDPG 拓展到复杂、高维的任务很困难，而 on-policy 的策略梯度方法在这类问题上仍可以达到最好的结果

Our method instead combines off-policy actor-critic training with a stochastic actor, and further aims to maximize the entropy of this actor with an entropy maximization objective. We find that this actually results in a considerably more stable and scalable algorithm that, in practice, exceeds both the efficiency and final performance of DDPG. A similar method can be derived as a zero-step special case of stochastic value gradients (SVG(0)) (Heess et al., 2015). However, SVG(0) differs from our method in that it optimizes the standard maximum expected return objective, and it does not make use of a separate value network, which we found to make training more stable. 
>  我们的方法将 off-policy actor-critic 训练和 stochastic actor 结合，并且以最大化该 actor 的熵为目标
>  我们发现这会使算法更加稳定并且可拓展，在实践中，其效率和性能都超越了 DDPG
>  随机值梯度方法的零步特殊情况 (SVG(0)) 和我们的算法类似，不同之处在于 SVG(0) 的优化目标是标准的最大期望回报 (没有最大熵目标)，并且没有使用单独的价值网络 (我们发现使用单独的价值网络可以是训练更稳定)

Maximum entropy reinforcement learning optimizes policies to maximize both the expected return and the expected entropy of the policy. This framework has been used in many contexts, from inverse reinforcement learning (Ziebart et al., 2008) to optimal control (Todorov, 2008; Toussaint, 2009; Rawlik et al., 2012). 
>  最大熵 RL 优化策略以同时最大化策略的期望回报和期望熵，该框架已经用于许多情景，包括逆向 RL 和最优控制

In guided policy search (Levine & Koltun, 2013; Levine et al., 2016), the maximum entropy distribution is used to guide policy learning towards high-reward regions. 
>  在引导策略搜索中，最大熵分布被用于引导策略向高奖励区域改善

More recently, several papers have noted the connection between Q-learning and policy gradient methods in the framework of maximum entropy learning (O’Donoghue et al., 2016; Haarnoja et al., 2017; Nachum et al., 2017a; Schulman et al., 2017a). 
>  一些文章注意到在最大熵学习框架下，Q-learning 和策略梯度方法的联系

While most of the prior model-free works assume a discrete action space, Nachum et al. (2017b) approximate the maximum entropy distribution with a Gaussian and Haarnoja et al. (2017) with a sampling network trained to draw samples from the optimal policy. 
>  多数之前的 model-free 工作都假设了离散动作空间，而 Natchum 则使用 Gaussian 近似最大熵分布, Haarnoja 使用一个采样网络从最优策略中抽取样本

Although the soft Q-learning algorithm proposed by Haarnoja et al. (2017) has a value function and actor network, it is not a true actor-critic algorithm: the Q-function is estimating the optimal Q-function, and the actor does not directly affect the Q-function except through the data distribution. Hence, Haarnoja et al. (2017) motivates the actor network as an approximate sampler, rather than the actor in an actor-critic algorithm. 
>  Haarnoja 提出的 soft Q-learning 算法也有一个 value network 和 actor network，但该算法并不是 actor-critic 框架，其 value network 估计的是 optimal Q-function，而 actor 不会直接影响 Q-function
>  其 actor network 是作为一个近似采样器，而不是 actor-critic 算法中的 actor

Crucially, the convergence of this method hinges on how well this sampler approximates the true posterior. In contrast, we prove that our method converges to the optimal policy from a given policy class, regardless of the policy parameterization. 
>  关键在于，其方法的收敛程度取决于该采样器对真实后验的近似程度
>  相较之下，我们证明我们的方法能够从给定的策略类中收敛到最优策略，无论策略的参数化如何

Furthermore, these prior maximum entropy methods generally do not exceed the performance of state-of-the-art off-policy algorithms, such as DDPG, when learning from scratch, though they may have other benefits, such as improved exploration and ease of fine-tuning. In our experiments, we demonstrate that our soft actor-critic algorithm does in fact exceed the performance of prior state-of-the-art off-policy deep RL methods by a wide margin. 
>  另外，这些之前的最大熵方法的性能通常不会超越 sota 的 off-policy 算法，例如 DDPG 的性能

# 3 Preliminaries 
We first introduce notation and summarize the standard and maximum entropy reinforcement learning frameworks. 

## 3.1 Notation 
We address policy learning in continuous action spaces. We consider an infinite-horizon Markov decision process (MDP), defined by the tuple $(S,{\mathcal{A}},p,r)$ , where the state space $\mathcal S$ and the action space $\mathcal{A}$ are continuous, and the unknown state transition probability $p:\mathcal{S}\times\mathcal{S}\times\mathcal{A}\rightarrow$ $[0,\infty)$ represents the probability density of the next state $\mathbf{s}_{t+1}\in S$ given the current state $\mathbf{s}_{t}\in S$ and action $\mathbf{a}_{t}\in\mathcal{A}$ The environment emits a bounded reward $r:S\times A\rightarrow$ $[r_{\operatorname*{min}},r_{\operatorname*{max}}]$ on each transition. We will use $\rho_{\pi}(\mathbf{s}_{t})$ and $\rho_{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$ to denote the state and state-action marginals of the trajectory distribution induced by a policy $\pi(\mathbf{a}_{t}|\mathbf{s}_{t})$ . 
>  我们解决的是连续动作空间中的策略学习问题
>  我们考虑无限期 MDP, 其中状态空间 $\mathcal S$ 和动作空间 $\mathcal A$ 都是连续空间
>  环境会为每次转移给出有界的奖励 $r:\mathcal S \times \mathcal A \to [r_{\text{min}}, r_{\text{max}}]$
>  $\rho_\pi(\mathbf s_t)$ 和 $\rho_\pi(\mathbf s_t, \mathbf a_t)$ 表示遵循策略 $\pi(\mathbf a_t\mid \mathbf s_t)$ 的轨迹分布中的状态/状态-动作边际

## 3.2 Maximum Entropy Reinforcement Learning 
Standard RL maximizes the expected sum of rewards $\begin{array}{r}{\sum_{t}\mathbb{E}_{(\mathbf{s}_{t},\mathbf{a}_{t})\sim\rho_{\pi}}\left[r(\mathbf{s}_{t},\mathbf{a}_{t})\right]}\end{array}$ . 
>  标准 RL 的目标是最大化期望奖励和 $\sum_t \mathbb E_{(\mathbf s_t, \mathbf a_t)\sim \rho_\pi}[r(\mathbf s_t, \mathbf a_t)]$ (参照 CPI paper 中的推导)

We will consider a more general maximum entropy objective (see e.g. Ziebart (2010)), which favors stochastic policies by augmenting the objective with the expected entropy of the policy over $\rho_{\pi}(\mathbf{s}_{t})$ : 


$$
J(\pi)=\sum_{t=0}^{T}\mathbb{E}_{(\mathbf{s}_{t},\mathbf{a}_{t})\sim\rho_{\pi}}\left[r(\mathbf{s}_{t},\mathbf{a}_{t})+\alpha\mathcal{H}(\pi(\cdot|\mathbf{s}_{t}))\right].\tag{1}
$$

>  我们考虑一个更广义的最大熵目标，通过在目标中加上策略相对于 $\rho_\pi(\mathbf s_t)$ 的期望熵，以偏好随机策略

The temperature parameter $\alpha$ determines the relative importance of the entropy term against the reward, and thus controls the stochasticity of the optimal policy. 
>  温度参数 $\alpha$ 决定了熵项相对于奖励的相对重要性，从而控制最优策略的随机性

The maximum entropy objective differs from the standard maximum expected reward objective used in conventional reinforcement learning, though the conventional objective can be recovered in the limit as $\alpha\rightarrow0$ . For the rest of this paper, we will omit writing the temperature explicitly, as it can always be subsumed into the reward by scaling it by $\alpha^{-1}$ . 
>  最大熵目标和传统 RL 中使用的标准最大期望奖励目标不同，当 $\alpha \to 0$ 时，最大熵目标成为传统目标
>  本文其余部分将忽略显式写出温度参数 $\alpha$，因为可以通过将奖励缩放 $\alpha^{-1}$ 来隐式地包含它

This objective has a number of conceptual and practical advantages. First, the policy is incentivized to explore more widely, while giving up on clearly unpromising avenues. Second, the policy can capture multiple modes of near optimal behavior. In problem settings where multiple actions seem equally attractive, the policy will commit equal probability mass to those actions. Lastly, prior work has observed improved exploration with this objective (Haarnoja et al., 2017; Schulman et al., 2017a), and in our experiments, we observe that it considerably improves learning speed over state-of-art methods that optimize the conventional RL objective function. 
>  这一目标具有许多概念上和实践上的优势
>  首先，策略被激励去更广泛地探索，同时放弃明显无望的路径
>  其次，策略可以捕获接近最优行为的多个模式 (如果有多个行为可以达到相同的优势，最大熵目标会鼓励策略将概率分散给它们)，在多个动作具有相同吸引力的设定下，策略将会为这些动作分配相同的概率质量
>  最后，之前的工作发现该目标可以提高探索性，并且在我们的实验中，我们发现它可以显著提高学习速度 (相较于优化传统 RL 目标的 sota 方法)

We can extend the objective to infinite horizon problems by introducing a discount factor $\gamma$ to ensure that the sum of expected rewards and entropies is finite. Writing down the maximum entropy objective for the infinite horizon discounted case is more involved (Thomas, 2014) and is deferred to Appendix A. 
>  我们可以引入折扣因子 $\gamma$，确保期望奖励和熵的和是有限的，来将该目标拓展到无限期问题
>  写出无限期的折扣最大熵目标更为复杂，相关内容在附录中讨论

Prior methods have proposed directly solving for the optimal Q-function, from which the optimal policy can be recovered (Ziebart et al., 2008; Fox et al., 2016; Haarnoja et al., 2017). We will discuss how we can devise a soft actor-critic algorithm through a policy iteration formulation, where we instead evaluate the Q-function of the current policy and update the policy through an off-policy gradient update. Though such algorithms have previously been proposed for conventional reinforcement learning, our method is, to our knowledge, the first off-policy actor-critic method in the maximum entropy reinforcement learning framework. 
>  之前的方法提出直接求解最优 Q-function，基于最优 Q-function 获得最优策略
>  我们将讨论如何通过策略迭代公式设计一个 soft actor-critic 算法，该算法中，我们评估当前策略的 Q-function，然后通过 off-policy 的梯度更新来更新策略
>  尽管这类算法之前已经被提出用于传统 RL，我们的方法则是首个在最大熵 RL 框架下的 off-policy actor-critic 方法

# 4 From Soft Policy Iteration to Soft Actor-Critic 
Our off-policy soft actor-critic algorithm can be derived starting from a maximum entropy variant of the policy iteration method. 
>  我们的 off-policy soft actor-critic 算法可以从策略迭代方法的一个最大熵变体开始推导

We will first present this derivation, verify that the corresponding algorithm converges to the optimal policy from its density class, and then present a practical deep reinforcement learning algorithm based on this theory. 
>  我们首先展示该推导，验证对应的算法可以从最优策略收敛到其密度类
>  然后我们展示基于该理论的使用的 DRL 算法

## 4.1 Derivation of Soft Policy Iteration 
We will begin by deriving soft policy iteration, a general algorithm for learning optimal maximum entropy policies that alternates between policy evaluation and policy improvement in the maximum entropy framework. 
>  我们先推导 soft policy iteration
>  soft policy iteration 在最大熵的框架下，交替执行策略评估和策略改进，以学习最优的最大熵策略

Our derivation is based on a tabular setting, to enable theoretical analysis and convergence guarantees, and we extend this method into the general continuous setting in the next section. We will show that soft policy iteration converges to the optimal policy within a set of policies which might correspond, for instance, to a set of parameterized densities. 
>  推导基于表格设定，以便进行理论分析和确保收敛性
>  我们将证明 soft policy iteration 将在一组策略 (例如一组参数化的密度函数)中收敛到其中最优的策略

In the policy evaluation step of soft policy iteration, we wish to compute the value of a policy $\pi$ according to the maximum entropy objective in Equation 1.
>  soft policy iteration 的**策略评估步骤**的目标是基于 Eq 1 的最大熵目标计算策略 $\pi$ 的价值

For a fixed policy, the soft Q-value can be computed iteratively, starting from any function $Q:S\times{\mathcal{A}}\rightarrow\mathbb{R}$ and repeatedly applying a modified Bellman backup operator $\mathcal{T}^{\pi}$ given by 

$$
\begin{array}{r}{\mathcal{T}^{\pi}Q(\mathbf{s}_{t},\mathbf{a}_{t})\triangleq r(\mathbf{s}_{t},\mathbf{a}_{t})+\gamma\mathbb{E}_{\mathbf{s}_{t+1}\sim p}\left[V(\mathbf{s}_{t+1})\right],}\end{array}\tag{2}
$$ 
where 

$$
V(\mathbf{s}_{t})=\mathbb{E}_{\mathbf{a}_{t}\sim\pi}\left[Q(\mathbf{s}_{t},\mathbf{a}_{t})-\log\pi(\mathbf{a}_{t}|\mathbf{s}_{t})\right]\tag{3}
$$ 
is the soft state value function. 

>  对于给定的策略 $\pi$，其 soft Q-value 可以从任意初始函数 $Q:\mathcal S\times \mathcal A \to \mathbb R$ 开始迭代计算，即重复为 $Q$ 应用修改后的 Bellman backup operator $\mathcal T^\pi$
>  Bellman backup 算子 $\mathcal T^\pi$ 的定义如 Eq 2 所示，注意 Eq 2 中，状态价值函数 $V$ 指 soft 状态价值函数，其定义如 Eq 3 所示

>  状态价值函数的定义为 $V(\mathbf s_t) = \mathbb E_{\mathbf a_t\sim \pi}[Q(\mathbf s_t, \mathbf a_t)]$ ，soft 状态价值函数的定义是为每个 $Q(\mathbf s_t, \mathbf a_t)$ 减去了 $\log \pi(\mathbf a_t\mid \mathbf s_t)$，若 $Q(\mathbf s_t, \mathbf a_t)$ 相同， $\pi(\mathbf a_t\mid \mathbf s_t)$ 越低，$V(\mathbf s_t)$ 反而越高，并且没有上界
>  soft 状态价值函数的下界是原来的状态价值函数

We can obtain the soft value function for any policy $\pi$ by repeatedly applying $\mathcal{T}^{\pi}$ as formalized below. 
>  对于任意策略 $\pi$，我们通过反复应用 $\mathcal T^\pi$ 获得其 soft (动作) 价值函数

**Lemma 1 (Soft Policy Evaluation).** Consider the soft Bellman backup operator $\mathcal{T}^{\pi}$ in Equation 2 and a mapping $Q^{0}:S\times{\mathcal{A}}\rightarrow\mathbb{R}$ with $|{\mathcal{A}}|<\infty$ , and define $Q^{k+1}=T^{\pi}Q^{k}$ . Then the sequence $Q^{k}$ will converge to the soft $Q$ -value of $\pi$ as $k\rightarrow\infty$ . 

Proof. See Appendix B.1. 

>  Lemma 1
>  对 $Q^0: \mathcal S\times \mathcal A\to \mathbb R, |\mathcal A|< \infty$ 反复应用 $\mathcal T^\pi$，得到的 $Q^k$ 序列将随着 $k\to \infty$ 收敛到 $\pi$ 的 soft 动作价值函数

In the policy improvement step, we update the policy towards the exponential of the new Q-function. This particular choice of update can be guaranteed to result in an improved policy in terms of its soft value. 
>  soft policy iteration 的**策略改进步骤**中，我们将策略改进为其 soft 动作价值函数的指数，这一更新将确保改进后的策略的 soft 动作价值函数更优

Since in practice we prefer policies that are tractable, we will additionally restrict the policy to some set of policies $\Pi$, which can correspond, for example, to a parameterized family of distributions such as Gaussians. To account for the constraint that $\pi\in\Pi$ , we project the improved policy into the desired set of policies. 
>  实践中，我们偏好可解的策略，故我们将策略范围限制在某个策略集 $\Pi$ 中，$\Pi$ 可以对应于某个参数化的分布族，例如 Gaussians
>  因此，我们需要求解改进后的策略在策略集 $\Pi$ 中的投影作为其代替

While in principle we could choose any projection, it will turn out to be convenient to use the information projection defined in terms of the Kullback-Leibler divergence. 
>  我们使用 KL 散度定义的信息投影

In the other words, in the policy improvement step, for each state, we update the policy according to 

$$
\pi_{\mathrm{new}}=\arg\operatorname*{min}_{\pi^{\prime}\in\Pi}\mathrm{D}_{\mathrm{KL}}\left(\pi^{\prime}(\cdot|\mathbf{s}_{t})\|\frac{\exp\left(Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t},\cdot)\right)}{Z^{\pi_{\mathrm{old}}}(\mathbf{s}_{t})}\right).\tag{4}
$$ 

>  故我们在策略改进步骤中，对于每个状态，都根据 Eq 4 更新策略
>  Eq 4 找到 $\Pi$ 中和 $\frac {\exp (Q^{\pi_{old}}(\mathbf s_t,\cdot))}{Z^{\pi_{old}}(\mathbf s_t)}$ 的 KL 散度最小的策略作为 $\pi_{new}$，其中划分函数 $Z^{\pi_{old}}(\mathbf s_t)$ 将分布 $\exp(Q^{\pi_{old}}(\mathbf s_t,\cdot))$ 规范化

The partition function $Z^{\pi_{\mathrm{old}}}\left(\mathbf{s}_{t}\right)$ normalizes the distribution, and while it is intractable in general, it does not contribute to the gradient with respect to the new policy and can thus be ignored, as noted in the next section. 
>  划分函数通常是不可解的，但它不会对相对于新策略的梯度做出贡献，故可以忽略

For this projection, we can show that the new, projected policy has a higher value than the old policy with respect to the objective in Equation 1. We formalize this result in Lemma 2. 
>  可以证明投影得到的新策略的广义最大熵目标函数值 (Eq 1) 将更大 (即具有更优的 soft 动作价值函数)

**Lemma 2 (Soft Policy Improvement).** Let $\pi_{\mathrm{old}}\in\Pi$ and let $\pi_{\mathrm{new}}$ be the optimizer of the minimization problem defined in Equation 4. Then $Q^{\pi_{\mathrm{new}}}(\mathbf{s}_{t},\mathbf{a}_{t})\geq Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t},\mathbf{a}_{t})$ for all $\left(\mathbf{s}_{t},\mathbf{a}_{t}\right)\in\mathcal{S}\times\mathcal{A}$ with $|{\mathcal{A}}|<\infty$ . 

Proof. See Appendix B.2. 

>  Lemma 2
>  Eq 4 的解 $\pi_{new}$ 将保证 $Q^{\pi_{new}}(\mathbf s_t, \mathbf a_t) \ge Q^{\pi_{old}}(\mathbf s_t, \mathbf a_t)$ 对所有的 $(\mathbf s_t, \mathbf a_t) \in \mathcal S\times \mathcal A$ 成立 (其中 $|\mathcal A| < \infty$)

The full soft policy iteration algorithm alternates between the soft policy evaluation and the soft policy improvement steps, and it will provably converge to the optimal maximum entropy policy among the policies in $\Pi$ (Theorem 1). 
>  完整的 soft policy iteration 算法交替执行 soft policy evaluation 和 soft policy improvement，并且可以证明它将收敛到 $\Pi$ 中的最优最大熵策略

Although this algorithm will provably find the optimal solution, we can perform it in its exact form only in the tabular case. Therefore, we will next approximate the algorithm for continuous domains, where we need to rely on a function approximator to represent the Q-values, and running the two steps until convergence would be computationally too expensive. The approximation gives rise to a new practical algorithm, called soft actor-critic. 
>  虽然该算法可以找到最优解，但证明是基于表格化情况的假设
>  因此，我们需要将该算法进行近似处理，迁移到连续域
>  我们需要使用函数近似器表示 Q-values，且将 soft policy evaluation 和 soft policy improvement 运行至收敛的计算将过于昂贵，故也需要进行近似
>  执行近似后得到的使用算法称为 soft actor-critic

**Theorem 1 (Soft Policy Iteration).** Repeated application of soft policy evaluation and soft policy improvement from any $\pi\in\Pi$ converges to a policy $\pi^{*}$ such that ${Q^{\pi}}^{*}\left(\mathbf{s}_{t},\mathbf{a}_{t}\right)\geq$ $Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$ for all $\pi\in\Pi$ and $\left(\mathbf{s}_{t},\mathbf{a}_{t}\right)\in\mathcal{S}\times\mathcal{A}$ , assuming $|{\mathcal{A}}|<\infty$ . 

Proof. See Appendix B.3. 

>  Theorem 1
>  对任意 $\pi \in \Pi$ 反复应用 soft policy evaluation 和 soft policy improvement 将确保收敛到 $\pi^*$，满足 $Q^{\pi^*}(\mathbf s_t, \mathbf a_t) \ge Q^\pi(\mathbf s_t, \mathbf a_t)$ 对于所有的 $\pi \in \Pi$ 和 $(\mathbf s_t, \mathbf a_t)\in \mathcal S\times \mathcal A$ 成立 (其中 $|\mathcal A|<\infty$)

## 4.2. Soft Actor-Critic 
As discussed above, large continuous domains require us to derive a practical approximation to soft policy iteration. To that end, we will use function approximators for both the Q-function and the policy, and instead of running evaluation and improvement to convergence, alternate between optimizing both networks with stochastic gradient descent. 
>  要将 soft policy iteration 应用于具有大型连续动作域的问题，需要进行近似，故此时我们使用函数近似器表示 Q-function 和 policy (而不是理论情况下的表格)
>  并且我们不会执行 evaluation 和 improvement 至收敛，而是使用随机梯度下降交替优化两个网络

We will consider a parameterized state value function $V_{\psi}({\bf s}_{t})$ , soft Q-function $Q_{\theta}(\mathbf{s}_{t},\mathbf{a}_{t})$ , and a tractable policy $\pi_{\phi}(\mathbf{a}_{t}|\mathbf{s}_{t})$ . The parameters of these networks are $\psi,\theta$ , and $\phi$ . For example, the value functions can be modeled as expressive neural networks, and the policy as a Gaussian with mean and covariance given by neural networks. We will next derive update rules for these parameter vectors. 
>  我们参数化 soft 状态价值函数 $V_\psi(\mathbf s_t)$，soft 动作价值函数 $Q_\theta(\mathbf s_t, \mathbf a_t)$，以及策略 $\pi_\phi(\mathbf a_t\mid \mathbf s_t)$

The state value function approximates the soft value. There is no need in principle to include a separate function approximator for the state value, since it is related to the Q-function and policy according to Equation 3. 
>  实际上并不需要使用额外的函数近似器近似 soft 状态价值函数
>  根据 Eq 3, soft 状态价值函数可以基于 soft 动作价值函数和策略计算

This quantity can be estimated from a single action sample from the current policy without introducing a bias, but in practice, including a separate function approximator for the soft value can stabilize training and is convenient to train simultaneously with the other networks. 
>  根据 Eq 3，可以采样一个动作样本来获取对 soft 状态价值函数的无偏估计
>  但实践中倾向于使用一个分离的函数近似器表示 soft 状态价值函数，这可以稳定训练，并且同时训练该网络和其他网络也是方便的

The soft value function is trained to minimize the squared residual error 

$$
{J_{V}(\psi)=\mathbb{E}_{\mathbf{s}_{t}\sim\mathcal{D}}\left[\frac{1}{2}\left(V_{\psi}(\mathbf{s}_{t})-\mathbb{E}_{\mathbf{a}_{t}\sim\pi_{\phi}}\left[Q_{\theta}(\mathbf{s}_{t},\mathbf{a}_{t})-\log\pi_{\phi}(\mathbf{a}_{t}|\mathbf{s}_{t})\right]\right)^{2}\right]}\tag{5}
$$ 
where $\mathcal{D}$ is the distribution of previously sampled states and actions, or a replay buffer. 

>  soft 状态价值函数的损失函数 $J_V(\psi)$ 依据 Eq 3 定义，形式是均方误差

The gradient of Equation 5 can be estimated with an unbiased estimator 

$$
\hat{\nabla}_{\psi}J_{V}(\psi)=\nabla_{\psi}V_{\psi}(\mathbf{s}_{t})\left(V_{\psi}(\mathbf{s}_{t})-Q_{\theta}(\mathbf{s}_{t},\mathbf{a}_{t})+\log\pi_{\phi}(\mathbf{a}_{t}|\mathbf{s}_{t})\right),\tag{6}
$$

where the actions are sampled according to the current policy, instead of the replay buffer. 

>  $J_V(\psi)$ 梯度的无偏估计值使用样本计算，其中样本依据当前策略采样

The soft Q-function parameters can be trained to minimize the soft Bellman residual 

$$
J_{Q}(\theta)=\mathbb{E}_{(\mathbf{s}_{t},\mathbf{a}_{t})\sim\mathcal{D}}\left[\frac{1}{2}\left(Q_{\theta}(\mathbf{s}_{t},\mathbf{a}_{t})-\hat{Q}(\mathbf{s}_{t},\mathbf{a}_{t})\right)^{2}\right],\tag{7}
$$

with 

$$
\hat{Q}(\mathbf{s}_{t},\mathbf{a}_{t})=r(\mathbf{s}_{t},\mathbf{a}_{t})+\gamma\mathbb{E}_{\mathbf{s}_{t+1}\sim p}\left[V_{\bar{\psi}}(\mathbf{s}_{t+1})\right],\tag{8}
$$ 
>  soft 动作价值函数的损失函数 $J_Q(\theta)$ 的形式仍然是均方误差
>  其中目标值 $\hat Q(\mathbf s_t, \mathbf a_t)$ 依据单步 Bellman equation 定义

which again can be optimized with stochastic gradients 

$$
\hat{\nabla}_{\boldsymbol{\theta}}J_{Q}(\boldsymbol{\theta})=\nabla_{\boldsymbol{\theta}}Q_{\boldsymbol{\theta}}(\mathbf{a}_{t},\mathbf{s}_{t})\left(Q_{\boldsymbol{\theta}}(\mathbf{s}_{t},\mathbf{a}_{t})-r(\mathbf{s}_{t},\mathbf{a}_{t})-\gamma V_{\bar{\boldsymbol{\psi}}}(\mathbf{s}_{t+1})\right).\tag{9}
$$ 
>  $J_Q(\theta)$ 梯度的无偏估计值使用样本计算

The update makes use of a target value network $V_{\bar{\psi}}$ , where $\bar{\psi}$ can be an exponentially moving average of the value network weights, which has been shown to stabilize training (Mnih et al., 2015). Alternatively, we can update the target weights to match the current value function weights periodically (see Appendix E).
>  $J_Q(\theta)$ 的计算依赖于 soft 状态价值函数的目标网络 $V_{\bar \psi}$，其参数 $\bar \psi$ 可以是 soft 状态价值网络参数的指数移动均值，也可以定期更新目标网络的参数使其和当前 soft 状态价值网络的参数匹配

 Finally, the policy parameters can be learned by directly minimizing the expected KL-divergence in Equation 4: 

$$
J_\pi(\phi) = \mathbb E_{\mathbf s_t \sim \mathcal D}\left[\mathrm {D_{KL}}\left(\pi_\phi(\cdot\mid \mathbf s_t)\Big|\Big|\frac {\exp(Q_\theta(\mathbf s_t,\cdot))}{Z_\theta(\mathbf s_t)}\right)\right]\tag{10}
$$

>  策略网络的损失函数 $J_\pi(\phi)$ 的形式是 KL 散度 (的期望)

There are several options for minimizing $J_{\pi}$ . A typical solution for policy gradient methods is to use the likelihood ratio gradient estimator (Williams, 1992), which does not require backpropagating the gradient through the policy and the target density networks. 
>  有许多方法可以用于最小化 $J_\pi$ (最小化 KL 散度)
>  策略梯度方法的一个典型解决方案是使用似然比梯度估计器，该方法不需要通过策略网络和目标密度网络反向传递梯度

However, in our case, the target density is the Q-function, which is represented by a neural network an can be differentiated, and it is thus convenient to apply the reparameterization trick instead, resulting in a lower variance estimator. To that end, we reparametrize the policy using a neural network transformation 

$$
\mathbf{a}_{t}=f_{\phi}(\epsilon_{t};\mathbf{s}_{t}),\tag{11}
$$

where $\epsilon_{t}$ is an input noise vector, sampled from some fixed distribution, such as a spherical Gaussian. 

>  $J_\pi$ 中的目标密度是用可微分的神经网络表示的 Q-function，我们可以应用重参数化技巧，得到方差更低的估计器
>  我们将策略网络进行重参数化表示 (Eq 11)，其中 $\epsilon_t$ 为输入噪声向量，从某个固定分布采样得到，例如球状高斯分布

>  重参数化以前，我们通过策略直接对 $\mathbf a_t$ 采样，重参数化之后，我们对噪声 $\epsilon_t$ 采样，然后对噪声执行确定性的转换 $\mathbf a_t = f_\phi(\epsilon_t;\mathbf s_t)$ 以得到动作
>  故可以认为策略网络 $f_\phi$ 表示的是将噪声分布到理想策略的映射

We can now rewrite the objective in Equation 10 as 

$$
{J_{\pi}(\phi)=\mathbb{E}_{\mathbf{s}_{t}\sim\mathcal{D},\epsilon_{t}\sim\mathcal{N}}\left[\log\pi_{\phi}(f_{\phi}(\epsilon_{t};\mathbf{s}_{t})|\mathbf{s}_{t})-Q_{\theta}(\mathbf{s}_{t},f_{\phi}(\epsilon_{t};\mathbf{s}_{t}))\right],}\tag{12}
$$

where $\pi_{\phi}$ is defined implicitly in terms of $f_{\phi}$ , and we have noted that the partition function is independent of $\phi$ and can thus be omitted. 

>  基于重参数化的策略网络，$J_\pi(\phi)$ 写为 Eq 12 的形式

>  推导
>  先展开 Eq 10 的 KL 散度

$$
\begin{align}
J_\pi(\phi) &= \mathbb E_{\mathbf s_t \sim \mathcal D}\left[\mathrm {D_{KL}}\left(\pi_\phi(\cdot\mid \mathbf s_t)\Big|\Big|\frac {\exp(Q_\theta(\mathbf s_t,\cdot))}{Z_\theta(\mathbf s_t)}\right)\right]\\
&=\mathbb E_{\mathbf s_t\sim \mathcal D}\left[\mathbb E_{\mathbf a_t \sim \pi_\phi(\cdot\mid \mathbf s_t)}[\ln \pi_\phi(\mathbf a_t\mid \mathbf s_t) - Q_\theta(\mathbf s_t,\mathbf a_t) + \ln Z_\theta(\mathbf s_t)]\right]
\end{align}
$$

>  利用重参数化技巧，将 $\mathbf a_t$ 替换为 $f_\phi(\epsilon;\mathbf s_t)$，将对动作采样替换为对噪声 $\epsilon_t$ 采样

$$
\begin{align}
J_\pi(\phi) &=\mathbb E_{\mathbf s_t\sim \mathcal D}\left[\mathbb E_{\mathbf a_t \sim \pi_\phi(\cdot\mid \mathbf s_t)}[\ln \pi_\phi(\cdot\mid \mathbf s_t) - Q_\theta(\mathbf s_t,\mathbf a_t) + \ln Z_\theta(\mathbf s_t)]\right]\\
&=\mathbb E_{\mathbf s_t\sim \mathcal D}\left[\mathbb E_{\epsilon_t \sim\mathcal N}[\ln \pi_\phi(f_\phi(\epsilon_t;\mathbf s_t)\mid \mathbf s_t) - Q_\theta(\mathbf s_t,f_\phi(\epsilon_t;\mathbf s_t))+\ln Z_\theta(\mathbf s_t)]\right]
\end{align}
$$

>  因为 $Z_\theta(\mathbf s_t)$ 与 $\phi$ 无关，故可以进行忽略，最后得到

$$
J_\pi(\phi) = \mathbb E_{\mathbf s_t\sim \mathcal D, \epsilon_t\sim \mathcal N}[\ln \pi_\phi(f_\phi(\epsilon_t;\mathbf s_t)\mid \mathbf s_t) - Q_\theta(\mathbf s_t,f_\phi(\epsilon_t;\mathbf s_t))]
$$

>  推导完毕

We can approximate the gradient of Equation 12 with 

$$
{\hat{\nabla}_{\phi}J_{\pi}(\phi)=\nabla_{\phi}\log\pi_{\phi}(\mathbf{a}_{t}|\mathbf{s}_{t})}{+(\nabla_{\mathbf{a}_{t}}\log\pi_{\phi}(\mathbf{a}_{t}|\mathbf{s}_{t})-\nabla_{\mathbf{a}_{t}}Q(\mathbf{s}_{t},\mathbf{a}_{t}))\nabla_{\phi}f_{\phi}(\epsilon_{t};\mathbf{s}_{t})}\tag{13}
$$ 
where $\mathbf{a}_{t}$ is evaluated at $f_{\phi}(\epsilon_{t};\mathbf{s}_{t})$ . This unbiased gradient estimator extends the DDPG style policy gradients (Lillicrap et al., 2015) to any tractable stochastic policy. 

>  重参数化后，$J_\pi(\phi)$ 梯度的无偏估计值同样使用样本计算
>  该梯度估计将 DDPG 风格的策略梯度拓展至任何可解的随机策略

>  推导

$$
\begin{align}
&\nabla_\phi[\ln \pi_\phi(f_\phi(\epsilon_t;\mathbf s_t)\mid \mathbf s_t) - Q_\theta(\mathbf s_t,f_\phi(\epsilon_t;\mathbf s_t))]\\
=&\nabla_\phi\ln \pi_\phi(f_\phi(\epsilon_t;\mathbf s_t)\mid \mathbf s_t) - \nabla_\phi Q_\theta(\mathbf s_t,f_\phi(\epsilon_t;\mathbf s_t))\\
=&\nabla_\phi\ln \pi_\phi(\mathbf a_t\mid \mathbf s_t) - \nabla_\phi Q_\theta(\mathbf s_t,\mathbf a_t)\\
=&\nabla_{\mathbf a_t} \ln\pi_\phi(\mathbf a_t\mid \mathbf s_t)\cdot\nabla_\phi \mathbf a_t + \nabla_\phi\ln \pi_\phi(\mathbf a_t\mid \mathbf s_t) - \nabla_\phi Q_\theta(\mathbf s_t, \mathbf a_t)\\
=&\nabla_{\mathbf a_t} \ln\pi_\phi(\mathbf a_t\mid \mathbf s_t)\cdot\nabla_\phi \mathbf a_t + \nabla_\phi\ln \pi_\phi(\mathbf a_t\mid \mathbf s_t) - \nabla_{\mathbf a_t} Q_\theta(\mathbf s_t, \mathbf a_t)\cdot\nabla_\phi \mathbf a_t\\
=&\nabla_\phi\ln \pi_\phi (\mathbf a_t\mid \mathbf s_t)+(\nabla_{\mathbf a_t}\ln\pi_\phi(\mathbf a_t\mid \mathbf s_t)-\nabla_{\mathbf a_t}Q_\theta(\mathbf s_t,\mathbf a_t))\nabla_\phi\mathbf a_t
\end{align}
$$

>  推导完毕

Our algorithm also makes use of two Q-functions to mitigate positive bias in the policy improvement step that is known to degrade performance of value based methods (Hasselt, 2010; Fujimoto et al., 2018). In particular, we parameterize two Q-functions, with parameters $\theta_{i}$ , and train them independently to optimize $J_{Q}(\theta_{i})$ . We then use the minimum of the Q-functions for the value gradient in Equation 6 and policy gradient in Equation 13, as proposed by Fujimoto et al. (2018). Although our algorithm can learn challenging tasks, including a 21-dimensional Humanoid, using just a single Q-function, we found two Q-functions significantly speed up training, especially on harder tasks.
>  我们也利用了双 Q 函数，以缓解策略改进过程中的正偏差
>  具体地说，我们用 $\theta_i$ 参数化两个 Q-function，并独立训练它们以优化 $J_Q(\theta_i)$，并使用较小的 Q-function 计算 Eq 6 的价值梯度 (计算目标值) 和 Eq 13 中的策略梯度
>  我们发现使用双 Q-function 可以显著加快训练速度，尤其是在更困难的任务上

![[pics/SAC-Algorithm1.png]]

The complete algorithm is described in Algorithm 1. The method alternates between collecting experience from the environment with the current policy and updating the function approximators using the stochastic gradients from batches sampled from a replay buffer. In practice, we take a single environment step followed by one or several gradient steps (see Appendix D for all hyperparameter). 
>  完整算法见 Algorithm 1，该算法交替执行 1. 使用当前策略从环境中收集经验 2. 使用 replay buffer 中的 batches 计算随机梯度，更新函数估计器
>  实践中，我们执行一次 environment step，然后执行一次或多次 gradient step

Using off-policy data from a replay buffer is feasible because both value estimators and the policy can be trained entirely on off-policy data. The algorithm is agnostic to the parameterization of the policy, as long as it can be evaluated for any arbitrary state-action tuple. 
>  使用来自于 replay buffer 的 off-policy 数据是可行的，因为价值函数和策略都可以完全基于 off-policy 数据训练 
>  算法对于策略的参数化形式不可知，策略只要能为任意状态-动作对评估概率即可

>  off-policy 的原因在于重参数化了策略，故基于策略采样动作的流程是先基于高斯采样噪声，然后确定性计算动作
>  因此 SAC 本质也是采用了 DDPG/DPG 的思想，其策略是带有确定性性质的

# 5 Experiments 
The goal of our experimental evaluation is to understand how the sample complexity and stability of our method compares with prior off-policy and on-policy deep reinforcement learning algorithms. 

We compare our method to prior techniques on a range of challenging continuous control tasks from the OpenAI gym benchmark suite (Brockman et al., 2016) and also on the rllab implementation of the Humanoid task (Duan et al., 2016). Although the easier tasks can be solved by a wide range of different algorithms, the more complex benchmarks, such as the 21-dimensional Humanoid (rllab), are exceptionally difficult to solve with off-policy algorithms (Duan et al., 2016). 

The stability of the algorithm also plays a large role in performance: easier tasks make it more practical to tune hyperparameters to achieve good results, while the already narrow basins of effective hyperparameters become prohibitively small for the more sensitive algorithms on the hardest benchmarks, leading to poor performance (Gu et al., 2016). 

We compare our method to deep deterministic policy gradient (DDPG) (Lillicrap et al., 2015), an algorithm that is regarded as one of the more efficient off-policy deep RL methods (Duan et al., 2016); proximal policy optimization (PPO) (Schulman et al., 2017b), a stable and effective on-policy policy gradient algorithm; and soft Q-learning (SQL) (Haarnoja et al., 2017), a recent off-policy algorithm for learning maximum entropy policies. 

Our SQL implementation also includes two Q-functions, which we found to improve its performance in most environments. We additionally compare to twin delayed deep deterministic policy gradient algorithm (TD3) (Fujimoto et al., 2018), using the author-provided implementation. This is an extension to DDPG, proposed concurrently to our method, that first applied the double Q-learning trick to continuous control along with other improvements. We have included trust region path consistency learning (Trust-PCL) (Nachum et al., 2017b) and two other variants of SAC in Appendix E. We turned off the exploration noise for evaluation for DDPG and PPO. For maximum entropy algorithms, which do not explicitly inject exploration noise, we either evaluated with the exploration noise (SQL) or use the mean action (SAC). The source code of our SAC implementation and videos are available online. 

## 5.1 Comparative Evaluation 
Figure 1 shows the total average return of evaluation rollouts during training for DDPG, PPO, and TD3. We train five different instances of each algorithm with different random seeds, with each performing one evaluation rollout every 1000 environment steps. The solid curves corresponds to the mean and the shaded region to the minimum and maximum returns over the five trials. 

![[pics/SAC-Fig1.png]]

The results show that, overall, SAC performs comparably to the baseline methods on the easier tasks and outperforms them on the harder tasks with a large margin, both in terms of learning speed and the final performance.
>  结果显示，总体而言，在较简单的任务中，SAC 的表现与基线方法相当，并且在更难的任务上以较大的优势超过了这些方法，无论是学习速度还是最终性能都表现优异

For example, DDPG fails to make any progress on Ant-v1, Humanoidv1, and Humanoid (rllab), a result that is corroborated by prior work (Gu et al., 2016; Duan et al., 2016). SAC also learns considerably faster than PPO as a consequence of the large batch sizes PPO needs to learn stably on more high-dimensional and complex tasks. Another maximum entropy RL algorithm, SQL, can also learn all tasks, but it is slower than SAC and has worse asymptotic performance. 
>  例如，DDPG 在 Ant-v1、Humanoid-v1 和 Humanoid（rllab）任务上未能取得任何进展。由于 PPO 需要更大的批量大小才能在更高维度和更复杂的任务上稳定学习，因此 SAC 的学习速度明显快于 PPO。另一种最大熵强化学习算法 SQL 也可以完成所有任务，但其速度比 SAC 慢，并且最终性能较差。

The quantitative results attained by SAC in our experiments also compare very favorably to results reported by other methods in prior work (Duan et al., 2016; Gu et al., 2016; Henderson et al., 2017), indicating that both the sample efficiency and final performance of SAC on these benchmark tasks exceeds the state of the art. All hyperparameters used in this experiment for SAC are listed in Appendix D. 

## 5.2 Ablation Study 
The results in the previous section suggest that algorithms based on the maximum entropy principle can outperform conventional RL methods on challenging tasks such as the humanoid tasks. In this section, we further examine which particular components of SAC are important for good performance. We also examine how sensitive SAC is to some of the most important hyperparameters, namely reward scaling and target value update smoothing constant. 
>  之前的结果表明了基于最大熵的算法在困难任务上的表现会由于传统 RL 方法
>  本节执行消融试验，并且验证 SAC 对超参数的敏感度

**Stochastic vs. deterministic policy.** Soft actor-critic learns stochastic policies via a maximum entropy objective. The entropy appears in both the policy and value function. In the policy, it prevents premature convergence of the policy variance (Equation 10). In the value function, it encourages exploration by increasing the value of regions of state space that lead to high-entropy behavior (Equation 5). 
>  Stochastic vs deterministic policy
>  SAC 通过最大熵目标学习随机策略，策略和价值函数的定义中都包含了熵
>  在策略中，熵的引入防止策略方差过早收敛，在价值函数中，熵的引入增加了低概率行为的价值，故提高了能导致低概率行为的状态空间区域的价值

To compare how the stochasticity of the policy and entropy maximization affects the performance, we compare to a deterministic variant of SAC that does not maximize the entropy and that closely resembles DDPG, with the exception of having two Q-functions, using hard target updates, not having a separate target actor, and using fixed rather than learned exploration noise. 
>  为了探究策略的随机性和熵最大化对性能的影响，我们将 SAC 和其确定性变体比较
>  该变体不最大化熵，故和 DDPG 十分相似，和 DDPG 的差异在于使用了双 Q-function，目标网络的更新是 hard update，以及没有使用 target actor 网络

![[pics/SAC-Fig2.png]]

Figure 2 compares five individual runs with both variants, initialized with different random seeds. Soft actor-critic performs much more consistently, while the deterministic variant exhibits very high variability across seeds, indicating substantially worse stability. As evident from the figure, learning a stochastic policy with entropy maximization can drastically stabilize training. This becomes especially important with harder tasks, where tuning hyperparameters is challenging. 
>  Fig 2 显示了 SAC 的确定性变体的表现在不同随机种子下具有高方差，说明其稳定性较差，而 SAC 的表现更加一致
>  这说明学习带有最大熵的随机策略可以极大稳定训练过程，因此在面对更难的任务，超参数调节更加困难时，随机性非常重要

In this comparison, we updated the target value network weights with hard updates, by periodically overwriting the target network parameters to match the current value network (see Appendix E for a comparison of average performance on all benchmark tasks). 
>  在该比较中，目标网络的权重使用 hard update，即周期性用当前网络参数覆盖目标网络参数

![[pics/SAC-Fig3.png]]

**Policy evaluation.** Since SAC converges to stochastic policies, it is often beneficial to make the final policy deterministic at the end for best performance. For evaluation, we approximate the maximum a posteriori action by choosing the mean of the policy distribution. Figure 3(a) compares training returns to evaluation returns obtained with this strategy indicating that deterministic evaluation can yield better performance.
>  Policy evaluation
>  因为 SAC 会收敛到随机性策略，因此令最终的策略确定化可以带来更好的性能，评估时，我们用策略分布的平均值近似极大后验动作 (近似没有最大熵时，策略赋予概率最高的动作)
>  Fig 3 的结果表明确定化策略可以带来更好性能

It should be noted that all of the training curves depict the sum of rewards, which is different from the objective optimized by SAC and other maximum entropy RL algorithms, including SQL and Trust-PCL, which maximize also the entropy of the policy. 
>  注意，本文中所有的训练曲线展示的都是奖励和的变化情况，而最大熵方法的目标函数并不仅仅包含奖励和，同时还需要最大化熵

**Reward scale.** Soft actor-critic is particularly sensitive to the scaling of the reward signal, because it serves the role of the temperature of the energy-based optimal policy and thus controls its stochasticity. Larger reward magnitudes correspond to lower entries. 
>  Reward scale
>  SAC 对于奖励的尺度非常敏感，该尺度扮演的角色是基于能量 (熵) 的最优策略的温度因子，故控制了策略的随机性，更大的奖励尺度对应于更低的随机性

Figure 3(b) shows how learning performance changes when the reward scale is varied: For small reward magnitudes, the policy becomes nearly uniform, and consequently fails to exploit the reward signal, resulting in substantial degradation of performance. For large reward magnitudes, the model learns quickly at first, but the policy then becomes nearly deterministic, leading to poor local minima due to lack of adequate exploration. With the right reward scaling, the model balances exploration and exploitation, leading to faster learning and better asymptotic performance. 
>  Fig 3 中，奖励尺度较小时，策略会趋向于均匀策略，无法利用奖励信号，性能较差；奖励尺度较大时，模型在开始时学习较快，但会趋向于接近确定性，故会由于缺乏探索导致陷入局部最优；奖励尺度合适时，模型平衡了探索和利用，故具有高的学习效率和更好的渐进性能

In practice, we found reward scale to be the only hyperparameter that requires tuning, and its natural interpretation as the inverse of the temperature in the maximum entropy framework provides good intuition for how to adjust this parameter. 
>  实践中，我们发现奖励尺度是唯一需要调节的超参数
>  奖励尺度在最大熵框架下可以直观解释为温度参数的倒数

**Target network update.** It is common to use a separate target value network that slowly tracks the actual value function to improve stability. We use an exponentially moving average, with a smoothing constant $\tau$ , to update the target value network weights as common in the prior work (Lillicrap et al., 2015; Mnih et al., 2015). A value of one corresponds to a hard update where the weights are copied directly at every iteration and zero to not updating the target at all.
>  Target network update
>  使用独立的目标价值网络来缓慢跟踪实际的价值函数以提高稳定性是常见的做法
>  参照之前的工作，我们使用指数移动平均来更新目标网络的参数，其平滑常数为 $\tau$
>  $\tau = 1$ 时，对应于在每次迭代直接将网络参数拷贝给目标网络，$\tau = 0$ 时，对应于完全不更新目标网络

 In Figure 3(c), we compare the performance of SAC when $\tau$ varies. Large $\tau$ can lead to instabilities while small $\tau$ can make training slower. However, we found the range of suitable values of $\tau$ to be relatively wide and we used the same value (0.005) across all of the tasks. 
 >  Fig 3 比较了 $\tau$ 变化时 SAC 的性能变化
 >  $\tau$ 较大会导致不稳定性，$\tau$ 较小会导致训练缓慢，但合适的 $\tau$ 范围相对较宽，故我们在所有任务中都令 $\tau = 0.005$

In Figure 4 (Appendix E) we also compare to another variant of SAC, where instead of using exponentially moving average, we copy over the current network weights directly into the target network every 1000 gradient steps. We found this variant to benefit from taking more than one gradient step between the environment steps, which can improve performance but also increases the computational cost. 
>  Fig 4 中比较了使用 hard update 时，性能关于 $\tau$ 的变化
>  我们每经过 1000 梯度步，就将当前网络权重直接拷贝给目标网络
>  该变体在每次环境步中会执行多次梯度步，这提高了其表现，但也增加了计算成本

# 6 Conclusion 
We present soft actor-critic (SAC), an off-policy maximum entropy deep reinforcement learning algorithm that provides sample-efficient learning while retaining the benefits of entropy maximization and stability. 
>  SAC 是一个 off-policy 最大熵 DRL 算法，该算法的特点是 sample-efficient learning 以及最大熵带来的 stability

Our theoretical results derive soft policy iteration, which we show to converge to the optimal policy. 
> 我们在理论上推导了 soft policy iteration 可以收敛到最优策略

From this result, we can formulate a soft actor-critic algorithm, and we empirically show that it outperforms state-of-the-art model-free deep RL methods, including the off-policy DDPG algorithm and the on-policy PPO algorithm. In fact, the sample efficiency of this approach actually exceeds that of DDPG by a substantial margin. 
>  基于 soft policy iteration，我们构造 soft actor-critic 算法，其经验性结果显示了该算法优于 off-policy DDPG, on-policy PPO 等算法，并且 SAC 的样本效率实际上比 DDPG 高出很多 (SAC 学得比 DDPG 更快)

Our results suggest that stochastic, entropy maximizing reinforcement learning algorithms can provide a promising avenue for improved robustness and stability, and further exploration of maximum entropy methods, including methods that incorporate second order information (e.g., trust regions (Schulman et al., 2015)) or more expressive policy classes is an exciting avenue for future work. 
>  我们的结果说明了最大熵随机策略可以提高健壮性和稳定性

# A Maximum Entropy Objective 
The exact definition of the discounted maximum entropy objective is complicated by the fact that, when using a discount factor for policy gradient methods, we typically do not discount the state distribution, only the rewards. In that sense, discounted policy gradients typically do not optimize the true discounted objective. Instead, they optimize average reward, with the discount serving to reduce variance, as discussed by Thomas (2014). However, we can define the objective that is optimized under a discount factor as 

$$
J(\pi)=\sum_{t=0}^{\infty}\mathbb{E}_{(\mathbf{s}_{t},\mathbf{a}_{t})\sim\rho_{\pi}}\left[\sum_{l=t}^{\infty}\gamma^{l-t}\mathbb{E}_{\mathbf{s}_{l}\sim p,\mathbf{a}_{l}\sim\pi}\left[r(\mathbf{s}_{t},\mathbf{a}_{t})+\alpha\mathcal{H}(\pi(\cdot|\mathbf{s}_{t}))|\mathbf{s}_{t},\mathbf{a}_{t}\right]\right].
$$ 
This objective corresponds to maximizing the discounted expected reward and entropy for future states originating from every state-action tuple $\left(\mathbf{s}_{t},\mathbf{a}_{t}\right)$ weighted by its probability $\rho_{\pi}$ under the current policy. 

# B Proofs 
### B.1. Lemma 1 
Lemma 1 (Soft Policy Evaluation). Consider the soft Bellman backup operator $\mathcal{T}^{\pi}$ in Equation 2 and a mapping $Q^{0}:S\times{\mathcal{A}}\to\mathbb{R}$ with $|{\mathcal{A}}|<\infty$ , and define $Q^{k+1}=T^{\pi}Q^{k}$ . Then the sequence $Q^{k}$ will converge to the soft $Q$ -value of $\pi$ as $k\rightarrow\infty$ . 
Proof. Define the entropy augmented reward as $r_{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})\triangleq r(\mathbf{s}_{t},\mathbf{a}_{t})+\mathbb{E}_{\mathbf{s}_{t+1}\sim p}\left[\mathcal{H}\left(\pi(\cdot|\mathbf{s}_{t+1})\right)\right]$ and rewrite the update rule as 
$$
Q(\mathbf{s}_{t},\mathbf{a}_{t})\gets r_{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})+\gamma\mathbb{E}_{\mathbf{s}_{t+1}\sim p,\mathbf{a}_{t+1}\sim\pi}\left[Q(\mathbf{s}_{t+1},\mathbf{a}_{t+1})\right]
$$ 
and apply the standard convergence results for policy evaluation (Sutton & Barto, 1998). The assumption $|{\cal A}|<\infty$ is required to guarantee that the entropy augmented reward is bounded. 口 
### B.2. Lemma 2 
Lemma 2 (Soft Policy Improvement). Let $\pi_{\mathrm{old}}\in\Pi$ and let $\pi_{\mathrm{new}}$ be the optimizer of the minimization problem defined in Equation 4. Then $Q^{\pi_{\mathrm{new}}}(\mathbf{s}_{t},\mathbf{a}_{t})\geq Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t},\mathbf{a}_{t})$ for all $\left(\mathbf{s}_{t},\mathbf{a}_{t}\right)\in\mathcal{S}\times\mathcal{A}$ with $|{\mathcal{A}}|<\infty$ . 
Proof. Let $\pi_{\mathrm{old}}\in\Pi$ and let $Q^{\pi_{\mathrm{old}}}$ and $V^{\pi_{\mathrm{old}}}$ be the corresponding soft state-action value and soft state value, and let $\pi_{\mathrm{new}}$ be defined as 
$$
\begin{array}{r l}&{\pi_{\mathrm{new}}(\cdot|\mathbf{s}_{t})=\arg\operatorname*{min}_{\pi^{\prime}\in\Pi}\mathrm{D}_{\mathrm{KL}}\left(\pi^{\prime}(\cdot|\mathbf{s}_{t})\parallel\exp\left(Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t},\cdot)-\log Z^{\pi_{\mathrm{old}}}(\mathbf{s}_{t})\right)\right)}\ &{\qquad=\arg\operatorname*{min}_{\pi^{\prime}\in\Pi}J_{\pi_{\mathrm{old}}}(\pi^{\prime}(\cdot|\mathbf{s}_{t})).}\end{array}
$$ 
It must be the case that $J_{\pi_{\mathrm{old}}}\big(\pi_{\mathrm{new}}\big(\cdot|\mathbf{s}_{t}\big)\big)\leq J_{\pi_{\mathrm{old}}}\big(\pi_{\mathrm{old}}\big(\cdot|\mathbf{s}_{t}\big)\big)$ , since we can always choose $\pi_{\mathrm{new}}=\pi_{\mathrm{old}}\in\Pi$ . Hence Eat∼πnew [ $\begin{array}{r}{\log\pi_{\mathfrak{n e w}}(\mathbf{a}_{t}|\mathbf{s}_{t})-Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t},\mathbf{a}_{t})+\log Z^{\pi_{\mathrm{old}}}(\mathbf{s}_{t})]\leq\mathbb{E}_{\mathbf{a}_{t}\sim\pi_{\mathrm{old}}}\left[\log\pi_{\mathrm{old}}(\mathbf{a}_{t}|\mathbf{s}_{t})-Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t},\mathbf{a}_{t})+\log\pi_{\mathrm{old}}(\mathbf{a}_{t}|\mathbf{s}_{t})\right]}\end{array}$ Zπold(st) (17) 
and since partition function $Z^{\pi_{\mathrm{old}}}$ depends only on the state, the inequality reduces to 
$$
\begin{array}{r}{\mathbb{E}_{\mathbf{a}_{t}\sim\pi_{\mathrm{new}}}\left[Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t},\mathbf{a}_{t})-\log\pi_{\mathrm{new}}(\mathbf{a}_{t}|\mathbf{s}_{t})\right]\ge V^{\pi_{\mathrm{old}}}(\mathbf{s}_{t}).}\end{array}
$$ 
Next, consider the soft Bellman equation: 
$$
\begin{array}{r l}&{Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t},\mathbf{a}_{t})=r(\mathbf{s}_{t},\mathbf{a}_{t})+\gamma\mathbb{E}_{\mathbf{s}_{t+1}\sim p}\left[V^{\pi_{\mathrm{old}}}(\mathbf{s}_{t+1})\right]}\ &{\qquad\leq r(\mathbf{s}_{t},\mathbf{a}_{t})+\gamma\mathbb{E}_{\mathbf{s}_{t+1}\sim p}\left[\mathbb{E}_{\mathbf{a}_{t+1}\sim\pi_{\mathrm{new}}}\left[Q^{\pi_{\mathrm{old}}}(\mathbf{s}_{t+1},\mathbf{a}_{t+1})-\log\pi_{\mathrm{new}}(\mathbf{a}_{t+1}|\mathbf{s}_{t+1})\right]\right]}\ &{\qquad\vdots}\ &{\qquad\leq Q^{\pi_{\mathrm{new}}}(\mathbf{s}_{t},\mathbf{a}_{t}),}\end{array}
$$ 
where we have repeatedly expanded $Q^{\pi_{\mathrm{old}}}$ on the RHS by applying the soft Bellman equation and the bound in Equation 18. Convergence to $Q^{\pi_{\mathrm{new}}}$ follows from Lemma 1. 口 
### B.3. Theorem 1 
Theorem 1 (Soft Policy Iteration). Repeated application of soft policy evaluation and soft policy improvement to any $\pi\in\Pi$ converges to a policy $\pi^{*}$ such that $Q^{\pi^{*}}(\mathbf{s}_{t},\mathbf{a}_{t})\geq Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$ for all $\pi\in\Pi$ and $(\mathbf{s}_{t},\mathbf{a}_{t})\in\mathcal{S}\times\mathcal{A}$ , assuming $|{\mathcal{A}}|<\infty$ . 
Proof. Let $\pi_{i}$ be the policy at iteration $i$ . By Lemma 2, the sequence $Q^{\pi_{i}}$ is monotonically increasing. Since $Q^{\pi}$ is bounded above for $\pi\in\Pi$ (both the reward and entropy are bounded), the sequence converges to some $\pi^{*}$ . We will still need to show that $\pi^{*}$ is indeed optimal. At convergence, it must be case that $J_{\pi^{*}}(\pi^{*}(\cdot|\mathbf{s}_{t}))<J_{\pi^{*}}(\pi(\cdot|\mathbf{s}_{t}))$ for all $\pi\in\Pi$ , $\pi\neq\pi^{*}$ . Using the same iterative argument as in the proof of Lemma 2, we get $Q^{\pi^{*}}(\mathbf{s}_{t},\mathbf{a}_{t})>Q^{\pi}(\mathbf{s}_{t},\mathbf{a}_{t})$ for all $\left(\mathbf{s}_{t},\mathbf{a}_{t}\right)\in\mathcal{S}\times\mathcal{A}$ , that is, the soft value of any other policy in $\Pi$ is lower than that of the converged policy. Hence $\pi^{*}$ is optimal in $\Pi$ 
### C. Enforcing Action Bounds 
We use an unbounded Gaussian as the action distribution. However, in practice, the actions needs to be bounded to a finite interval. To that end, we apply an invertible squashing function (tanh) to the Gaussian samples, and employ the change of variables formula to compute the likelihoods of the bounded actions. In the other words, let $\mathbf{u}\in\mathbb{R}^{D}$ be a random variable and $\mu(\mathbf{u}|\mathbf{s})$ the corresponding density with infinite support. Then $\mathbf{a}=\operatorname{tanh}(\mathbf{u})$ , where tanh is applied elementwise, is a random variable with support in $(-1,1)$ with a density given by 
$$
\pi(\mathbf{a}|\mathbf{s})=\mu(\mathbf{u}|\mathbf{s})\left|\operatorname*{det}\left({\frac{\mathrm{d}\mathbf{a}}{\mathrm{d}\mathbf{u}}}\right)\right|^{-1}.
$$ 
Since the Jacobian $\mathrm{{d}\bf{a}/\mathrm{{d}\bf{u}=\mathrm{{diag}(1-t a n h^{2}(\bf{u}))}}}$ is diagonal, the log-likelihood has a simple form 
$$
\log\pi(\mathbf{a}|\mathbf{s})=\log\mu(\mathbf{u}|\mathbf{s})-\sum_{i=1}^{D}\log\left(1-\operatorname{tanh}^{2}(u_{i})\right),
$$ 
where $u_{i}$ is the $i^{\mathrm{th}}$ element of $\mathbf{u}$ . 
## D. Hyperparameters 
Table 1 lists the common SAC parameters used in the comparative evaluation in Figure 1 and Figure 4. Table 2 lists the reward scale parameter that was tuned for each environment. 
Table 1. SAC Hyperparameters 
<html><body><table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Shared</td><td></td></tr><tr><td>optimizer</td><td>Adam (Kingma & Ba, 2015)</td></tr><tr><td>learning rate discount ()</td><td>3·10-4</td></tr><tr><td>replay buffer size</td><td>0.99</td></tr><tr><td>number of hidden layers (all networks)</td><td>106</td></tr><tr><td>number of hidden units per layer</td><td>2</td></tr><tr><td>number of samples per minibatch</td><td>256</td></tr><tr><td></td><td>256</td></tr><tr><td>nonlinearity</td><td>ReLU</td></tr><tr><td>SAC</td><td></td></tr><tr><td>target smoothing coefficient (T)</td><td>0.005</td></tr><tr><td>target update interval</td><td>1</td></tr><tr><td>gradient steps</td><td></td></tr><tr><td>SAC (hard target update)</td><td></td></tr><tr><td>target smoothing coefficient (T)</td><td>1</td></tr><tr><td>targetupdateinterval</td><td>1000</td></tr><tr><td>gradient steps (except humanoids)</td><td>4</td></tr><tr><td>gradient steps (humanoids)</td><td>1</td></tr></table></body></html> 
Table 2. SAC Environment Specific Parameters 
<html><body><table><tr><td>Environment</td><td>Action Dimensions</td><td>RewardScale</td></tr><tr><td>Hopper-vl</td><td>3</td><td>5</td></tr><tr><td>Walker2d-v1</td><td>6</td><td>5</td></tr><tr><td>HalfCheetah-vl</td><td>6</td><td>5</td></tr><tr><td>Ant-v1</td><td>8</td><td>5</td></tr><tr><td>Humanoid-v1</td><td>17</td><td>20</td></tr><tr><td>Humanoid (rllab)</td><td>21</td><td>10</td></tr></table></body></html> 
## E. Additional Baseline Results 
Figure 4 compares SAC to Trust-PCL (Figure 4. Trust-PC fails to solve most of the task within the given number of environment steps, although it can eventually solve the easier tasks (Nachum et al., 2017b) if ran longer. The figure also includes two variants of SAC: a variant that periodically copies the target value network weights directly instead of using exponentially moving average, and a deterministic ablation which assumes a deterministic policy in the value update (Equation 6) and the policy update (Equation 13), and thus strongly resembles DDPG with the exception of having two Q-functions, using hard target updates, not having a separate target actor, and using fixed exploration noise rather than learned. Both of these methods can learn all of the tasks and they perform comparably to SAC on all but Humanoid (rllab) task, on which SAC is the fastest. 
![](https://cdn-mineru.openxlab.org.cn/extract/release/d6b67d8e-680f-4126-8e37-eeb47ffb8ca3/7a3e7abb0de833102a1c2b9e6fe295e39a38d168004b366e5f217dc384f32d0a.jpg) 
Figure 4. Training curves for additional baseline (Trust-PCL) and for two SAC variants. Soft actor-critic with hard target update (blue) differs from standard SAC in that it copies the value function network weights directly every 1000 iterations, instead of using exponentially smoothed average of the weights. The deterministic ablation (red) uses a deterministic policy with fixed Gaussian exploration noise, does not use a value function, drops the entropy terms in the actor and critic function updates, and uses hard target updates for the target Q-functions. It is equivalent to DDPG that uses two Q-functions, hard target updates, and removes the target actor. 