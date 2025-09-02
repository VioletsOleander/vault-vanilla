# Abstract
We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. 
>  我们将 Deep Q-Learning 的思想迁移到连续动作问题，我们提出一个 actor-critic, model-free 的基于确定策略梯度的算法，该算法可以在连续动作空间上运行

Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. 
>  我们的算法可以解决超过 20 个模拟物理任务，并且找到的策略能和具有完整领域动力学和其导数的规划算法找到的策略相比

We further demonstrate that for many of the tasks the algorithm can learn policies “end-to-end”: directly from raw pixel inputs. 
>  我们进一步展示我们的算法可以端到端地学习策略，即直接从原始像素输入中学习

# 1 Introduction
One of the primary goals of the field of artificial intelligence is to solve complex tasks from unprocessed, high-dimensional, sensory input. Recently, significant progress has been made by combining advances in deep learning for sensory processing (Krizhevsky et al., 2012) with reinforcement learning, resulting in the “Deep Q Network” (DQN) algorithm (Mnih et al., 2015) that is capable of human level performance on many Atari video games using unprocessed pixels for input. To do so, deep neural network function approximators were used to estimate the action-value function. 
>  DQN 中，深度网络被用于近似估计动作价值函数，直接使用未处理的像素作为输入

However, while DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have continuous (real valued) and high dimensional action spaces. DQN cannot be straightforwardly applied to continuous domains since it relies on a finding the action that maximizes the action-value function, which in the continuous valued case requires an iterative optimization process at every step. 
>  DQN 解决了具有高维观测空间的问题，但仅适用于离散和低维的动作空间
>  许多物理控制问题的动作空间都是高维且连续的，因为 DQN 需要找到最大化动作价值函数的动作，故难以直接应用到带有连续动作空间的问题，否则在每一步都需要迭代式优化来寻找最优动作

An obvious approach to adapting deep reinforcement learning methods such as DQN to continuous domains is to to simply discretize the action space. However, this has many limitations, most notably the curse of dimensionality: the number of actions increases exponentially with the number of degrees of freedom. For example, a 7 degree of freedom system (as in the human arm) with the coarsest discretization $a_{i}\in\lbrace{k,0,k\rbrace}$ for each joint leads to an action space with dimensionality: $3^{7}=2187$ . The situation is even worse for tasks that require fine control of actions as they require a correspondingly finer grained discretization, leading to an explosion of the number of discrete actions. Such large action spaces are difficult to explore efficiently, and thus successfully training DQN-like networks in this context is likely intractable. Additionally, naive discretization of action spaces needlessly throws away information about the structure of the action domain, which may be essential for solving many problems. 
>  将 DQN 迁移到连续领域的一个方法是离散化动作空间，但存在 curse of dimensionality 的问题: 离散后的动作数量随着动作空间自由度指数增长
>  过大的动作空间将难以高效探索，故在这种情况下难以成功训练类似 DQN 的网络
>  并且，朴素地离散化动作空间会丢失连续空间的结构，这可能是解决问题的关键信息

In this work we present a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces. Our work is based on the deterministic policy gradient (DPG) algorithm (Silver et al., 2014) (itself similar to NFQCA (Hafner & Riedmiller, 2011), and similar ideas can be found in (Prokhorov et al., 1997)). However, as we show below, a naive application of this actor-critic method with neural function approximators is unstable for challenging problems. 
>  本文提出一个 model-free, off-policy actor-critic 算法，使用深度网络学习在高维连续动作空间的策略
>  本文基于 DPG 算法，但要注意朴素地将该算法和深度网络结合是不稳定的

Here we combine the actor-critic approach with insights from the recent success of Deep Q Network (DQN) (Mnih et al., 2013; 2015). Prior to DQN, it was generally believed that learning value functions using large, non-linear function approximators was difficult and unstable. DQN is able to learn value functions using such function approximators in a stable and robust way due to two innovations: 1. the network is trained off-policy with samples from a replay buffer to minimize correlations between samples; 2. the network is trained with a target Q network to give consistent targets during temporal difference backups. In this work we make use of the same ideas, along with batch normalization (Ioffe & Szegedy, 2015), a recent advance in deep learning. 
>  我们将 actor-critic 方法和 DQN 的成功经验结合
>  在 DQN 之前，普遍观点是使用大型非线性函数近似器来学习价值函数是困难且不稳定的，DQN 可以用稳定且健壮的方式学习价值函数，这得益于两大创新: 1. 网络通过从 replay buffer 采样的方式训练，以最小化样本之间的关联 2. 网络通过目标 Q 网络进行训练，以在 TD 回溯更新的过程中提供一致的 target
>  本文中，我们利用了相同的想法，并结合了批量归一化

In order to evaluate our method we constructed a variety of challenging physical control problems that involve complex multi-joint movements, unstable and rich contact dynamics, and gait behavior. Among these are classic problems such as the cartpole swing-up problem, as well as many new domains. A long-standing challenge of robotic control is to learn an action policy directly from raw sensory input such as video. Accordingly, we place a fixed viewpoint camera in the simulator and attempted all tasks using both low-dimensional observations (e.g. joint angles) and directly from pixels. 
>  为了评估我们的方法，我们构建了一系列具有挑战性的物理控制问题，这些问题涉及复杂的多关节运动、不稳定且丰富的接触动力学以及步态行为。其中包括经典的倒立摆摆动问题等，以及许多新的领域。
>  机器人控制长期以来的一个挑战是从原始感官输入（如视频）直接学习动作策略。因此，我们在模拟器中放置了一个固定视角的摄像头，并尝试使用低维观测值（例如关节角度）和直接从像素中获取的方式来完成所有任务。

Our model-free approach which we call Deep DPG (DDPG) can learn competitive policies for all of our tasks using low-dimensional observations (e.g. cartesian coordinates or joint angles) using the same hyper-parameters and network structure. In many cases, we are also able to learn good policies directly from pixels, again keeping hyperparameters and network structure constant. 
>  DDPG 对于我们构建的所有任务，仅基于低维的观测 (例如坐标或关节角度) ，使用和之前方法相同的网络结构和超参数，都可以学习到优秀的策略
>  许多情况下，我们可以直接从像素中学习策略，且不需要改变超参数和网络结构

A key feature of the approach is its simplicity: it requires only a straightforward actor-critic architecture and learning algorithm with very few “moving parts”, making it easy to implement and scale to more difficult problems and larger networks. 
>  DDPG 很简单，它仅需一个 actor-critic 架构和一个学习算法，且 “可调参数” 非常少

For the physical control problems we compare our results to a baseline computed by a planner (Tassa et al., 2012) that has full access to the underlying simulated dynamics and its derivatives (see supplementary information). Interestingly, DDPG can sometimes find policies that exceed the performance of the planner, in some cases even when learning from pixels (the planner always plans over the underlying low-dimensional state space). 
>  对于物理控制问题，我们还将 DDPG 和一个 baseline planner 比较
>  该 planner 知道底层的环境动态和其导数，而 DDPG 可以在仅从像素学习的情况下找到优于 planner 的策略 (planner 始终在真实的低维状态空间进行规划决策)
>  (也就是说 DDPG 从像素中提取到了 underlying 的状态空间)

# 2 Background
We consider a standard reinforcement learning setup consisting of an agent interacting with an environment $E$ in discrete timesteps. At each timestep $t$ the agent receives an observation $x_{t}$ , takes an action $a_{t}$ and receives a scalar reward $r_{t}$ . 

In all the environments considered here the actions are real-valued ${a}_{t}\in\mathbb{R}^{N}$ . In general, the environment may be partially observed so that the entire history of the observation, action pairs $s_{t}=(x_{1},a_{1},...,a_{t-1},x_{t})$ may be required to describe the state. Here, we assumed the environment is fully-observed so ${s}_{t}={x}_{t}$ . 
>  我们考虑的环境中，动作都是实值向量 $a_t \in \mathbb R^N$，且观测等于状态 $s_t = x_t$

An agent’s behavior is defined by a policy, $\pi$ , which maps states to a probability distribution over the actions $\pi\colon S\to{\mathcal{P}}({\mathcal{A}})$ . The environment, $E$ , may also be stochastic. We model it as a Markov decision process with a state space $\mathcal S$ , action space $\mathcal {A}={\mathbb{R}}^{N}$ , an initial state distribution $p(s_{1})$ , transition dynamics $p(s_{t+1}|s_{t},a_{t})$ , and reward function $r(s_{t},a_{t})$ . 
>  定义策略、状态空间、动作空间、初始状态分布、转移动态、奖励函数

The return from a state is defined as the sum of discounted future reward $\begin{array}{r}{R_{t}=\sum_{i=t}^{T}\gamma^{(i-t)}r(s_{i},a_{i})}\end{array}$ with a discounting factor $\gamma\in[0,1]$ . Note that the return depends on the actions chosen, and therefore on the policy $\pi$ , and may be stochastic. The goal in reinforcement learning is to learn a policy which maximizes the expected return from the start distribution $J=\mathbb{E}_{r_{i},s_{i}\sim E,a_{i}\sim\pi}\left[R_{1}\right]$ . We denote the discounted state visitation distribution for a policy $\pi$ as $\rho^{\pi}$ . 
>  定义回报
>  学习目标是找到最大化初始状态的回报期望 $J = \mathbb E_{r_i, s_i\sim E, a_i\sim \pi}[R_1]$
>  记策略 $\pi$ 的折扣状态访问分布为 $\rho^\pi$

The action-value function is used in many reinforcement learning algorithms. It describes the expected return after taking an action $a_{t}$ in state $s_{t}$ and thereafter following policy $\pi$ : 

$$
Q^{\pi}(s_{t},a_{t})=\mathbb{E}_{r_{i\geq t},s_{i>t}\sim E,a_{i>t}\sim\pi}\left[R_{t}|s_{t},a_{t}\right]
$$ 
Many approaches in reinforcement learning make use of the recursive relationship known as the Bellman equation: 

$$
Q^{\pi}(s_{t},a_{t})=\mathbb{E}_{r_{t},s_{t+1}\sim E}\left[r(s_{t},a_{t})+\gamma\mathbb{E}_{a_{t+1}\sim\pi}\left[Q^{\pi}(s_{t+1},a_{t+1})\right]\right]
$$ 
>  定义了动作价值函数和相应的 Bellman equation

If the target policy is deterministic we can describe it as a function $\mu:\mathcal S\to A$ and avoid the inner expectation: 

$$
Q^{\mu}(s_{t},a_{t})=\mathbb{E}_{r_{t},s_{t+1}\sim E}\left[r(s_{t},a_{t})+\gamma Q^{\mu}(s_{t+1},\mu(s_{t+1}))\right]
$$ 
The expectation depends only on the environment. This means that it is possible to learn $Q^{\mu}$ off-policy, using transitions which are generated from a different stochastic behavior policy $\beta$ . 

>  如果目标策略 $\mu:\mathcal S\to \mathcal A$ 是确定性的，Bellman equation 中的对 $a_{t+1}\sim \pi$ 求期望可以去除，动作直接记为 $\mu(s_{t+1})$
>  此时该 Bellman equation 的期望仅依赖于环境，故可以使用由其他行为策略 $\beta$ 生成的轨迹，异策略学习动作价值函数 $Q^\mu$，

Q-learning (Watkins & Dayan, 1992), a commonly used off-policy algorithm, uses the greedy policy $\mu(s)=\mathrm{arg}\operatorname*{max}_{a}Q(s,a)$ . We consider function approximators parameterized by ${\theta^{Q}}$ , which we optimize by minimizing the loss: 

$$
L(\theta^{Q})=\mathbb{E}_{s_{t}\sim\rho^{\beta},a_{t}\sim\beta,r_{t}\sim E}\left[\left(Q(s_{t},a_{t}|\theta^{Q})-y_{t}\right)^{2}\right]\tag{4}
$$ 
where 
$$
y_{t}=r(s_{t},a_{t})+\gamma Q(s_{t+1},\mu(s_{t+1})|\theta^{Q}).\tag{5}
$$ 
While $y_{t}$ is also dependent on $\theta^{Q}$ , this is typically ignored. 

>  使用 $\theta^Q$ 参数化动作价值函数，并定义其损失如上
>  损失是 $Q(s_t, a_t\mid \theta^q)$ 和 $y_t$ 的均方误差 (对数据集求期望)，其中目标 $y_t$ 是 Bellman equation 的 RHS 的近似，目标 $y_t$ 虽然也依赖于 $\theta^Q$，但一般该依赖会被忽略

The use of large, non-linear function approximators for learning value or action-value functions has often been avoided in the past since theoretical performance guarantees are impossible, and practically learning tends to be unstable. Recently, (Mnih et al., 2013; 2015) adapted the Q-learning algorithm in order to make effective use of large neural networks as function approximators. Their algorithm was able to learn to play Atari games from pixels. In order to scale Q-learning they introduced two major changes: the use of a replay buffer, and a separate target network for calculating $y_{t}$ . We employ these in the context of DDPG and explain their implementation in the next section. 
>  在过去，由于无法提供理论上的性能保证，并且在实践中学习过程往往不稳定，故通常避免用 NN 来学习动作价值函数
>  而 DQN 则高效地利用了 NN 学习了动作价值函数，DQN 为了扩展 Q-Learning，引入了两个重大改变: 使用 replay buffer、使用目标网络单独计算 $y_t$，我们在 DDPG 中也采用了这两个方法

# 3 Algorithm
It is not possible to straightforwardly apply Q-learning to continuous action spaces, because in continuous spaces finding the greedy policy requires an optimization of $a_{t}$ at every timestep; this optimization is too slow to be practical with large, unconstrained function approximators and nontrivial action spaces. Instead, here we used an actor-critic approach based on the DPG algorithm (Silver et al., 2014). 
>  直接将 DQN 迁移到连续动作空间是不可能的，因为每一步都需要对动作 $a_t$ 进行优化
>  我们转而使用基于 DPG 算法的 actor-critic 方法

The DPG algorithm maintains a parameterized actor function $\mu(s|\theta^{\mu})$ which specifies the current policy by deterministically mapping states to a specific action. The critic $Q(s,a)$ is learned using the Bellman equation as in Q-learning. The actor is updated by following the applying the chain rule to the expected return from the start distribution $J$ with respect to the actor parameters: 

$$
\begin{align}
\nabla_{\theta^\mu}J &\approx\mathbb E_{s_t\sim \rho^\beta}[\nabla_{\theta^\mu} Q(s, a\mid\theta^Q)\mid_{s = s_t, a=\mu(s_t\mid\theta^\mu)}]\\
&=\mathbb E_{s_t\sim \rho^\beta}[\nabla_a Q(s, a\mid\theta^Q)\mid_{s=s_t,a=\mu(s_t\mid \theta^\mu)}\nabla_{\theta^\mu}\mu(s\mid \theta^\mu)\mid_{s=s_t}]
\end{align}\tag{6}
$$

Silver et al. (2014) proved that this is the policy gradient, the gradient of the policy’s performance. 

>  DPG 中，actor function $\mu(s\mid \theta^\mu)$ 将状态确定性映射到动作，critic function $Q(s, a)$ 和 Q-Learning 一样，用 Bellman equation 学习
>  actor 通过梯度方法更新，其目标函数是 (相对于起始分布) 的期望回报 $J$ 
>  (第一个 $\approx$ 应该是因为 $Q(\cdot, \cdot\mid \theta^Q)$ 仅仅是 $\mu(\cdot\mid\theta^\mu)$ 的价值函数的近似，第二个 $=$ 是应用了链式法则，$Q(\cdot, \cdot\mid \theta^Q)$ 和 $\theta^\mu$ 的关系中，$\mu(s\mid \theta^\mu)$ 是中间函数)

As with Q learning, introducing non-linear function approximators means that convergence is no longer guaranteed. However, such approximators appear essential in order to learn and generalize on large state spaces. NFQCA (Hafner & Riedmiller, 2011), which uses the same update rules as DPG but with neural network function approximators, uses batch learning for stability, which is intractable for large networks. A minibatch version of NFQCA which does not reset the policy at each update, as would be required to scale to large networks, is equivalent to the original DPG, which we compare to here. 
>  和 Q Learning 类似，为 DPG 引入非线性函数近似器后，就意味着收敛性不再有保证，但为了学习并泛化到大规模状态空间，引入非线性函数近似器是必要的
>  NFQCA 使用和 DPG 相同的更新规则，并使用了 NN 近似器，为了稳定性，使用了批量学习
>  NFQCA 的一个 minibatch 版本不会在每次更新时重置策略，等价于原始的 DPG (如果要拓展到大型网络，则需要重置策略)

![[pics/DDPG-Algorithm1.png]]

Our contribution here is to provide modifications to DPG, inspired by the success of DQN, which allow it to use neural network function approximators to learn in large state and action spaces online. We refer to our algorithm as Deep DPG (DDPG, Algorithm 1). 
>  我们的贡献是基于 DQN 的成功经验，对 DPG 进行了修改，使其能够使用 NN 在大规模状态和动作空间上学习，该算法称为 DDPG

One challenge when using neural networks for reinforcement learning is that most optimization algorithms assume that the samples are independently and identically distributed. Obviously, when the samples are generated from exploring sequentially in an environment this assumption no longer holds. Additionally, to make efficient use of hardware optimizations, it is essential to learn in minibatches, rather than online. 
>  使用 NN 进行 RL 的一个挑战在于大多数优化算法假设样本是独立同分布的，而当样本是来自于环境中的连续性探索时，独立同分布假设就不再成立
>  并且，为了高效地执行硬件优化，以 minibatches 的形式进行学习是必要的，而不是在线学习 (逐样本)

>  一点思考
>  独立同分布假设保证了数据集内的所有样本都是不相关的，故在一个 batch 内的样本都是不相关的，故其损失、梯度也是不相关的，对其梯度计算平均，就是对整个数据集上期望梯度进行了无偏估计
>  如果样本之间存在依赖，样本就不能视作单个数据点，假设对存在依赖的样本按照 batch 计算平均梯度，因为样本之间的关联，例如样本 a 是基于样本 b 的情况下再执行一次动作后得到的，那么样本 a 达成的损失值会和它的上一个状态 (样本 b)，甚至之前的多个状态存在关联，故对于样本 a 的损失计算的梯度是和样本 b 相关的，此时简单计算平均，就一定程度上重复计算了样本 b 导致的损失，从而引入了偏差
>  这种情况下，我们的函数将更倾向于朝着 “如果样本 a 此时的情况之前是样本 b，则要怎么样做才能使得分数更高”，而我们期望的实际上是 “无论样本 a 此时的情况之前是哪个状态，函数都能选择出较优的动作”，即泛化性能存在差异
>  但如果能够收集到无穷的样本，遍历所有可能性，理论上应该就没有问题 (这就凸显了 exploration 的重要性，它对于算法的泛化能力是十分重要的)

As in DQN, we used a replay buffer to address these issues. The replay buffer is a finite sized cache $\mathcal{R}$ . Transitions were sampled from the environment according to the exploration policy and the tuple $(s_{t},a_{t},r_{t},s_{t+1})$ was stored in the replay buffer. When the replay buffer was full the oldest samples were discarded. At each timestep the actor and critic are updated by sampling a minibatch uniformly from the buffer.  
>  和 DQN 中一样，我们使用 replay buffer 来解决这些问题
>  replay buffer 是大小固定的缓存 $\mathcal R$，其中存储了元组 $(s_t, a_t, r_t, s_{t+1})$，当 buffer 满时，最旧的样本会被丢弃
>  在每个时间步，actor 和 critic 从 buffer 中均匀采样一个 minibatch 用于更新

Because DDPG is an off-policy algorithm, the replay buffer can be large, allowing the algorithm to benefit from learning across a set of uncorrelated transitions.
>  因为 DDPG 是异策略算法，replay buffer 可以很大，便于算法尽量从一组不相关的样本中学习

>  这里原文称样本为 transition，还是很合理的，因为 agent 就是 learn from transitions，并且 transition 之间的相关性在概念上也很清晰

Directly implementing Q learning (equation 4) with neural networks proved to be unstable in many environments. Since the network $Q(s,a|\theta^{Q})$ being updated is also used in calculating the target value (equation 5), the Q update is prone to divergence. 
>  直接用 NN 进行 Eq4 的 Q-Learning 是不稳定的，因为存在自举，被更新的网络 $Q(s, a\mid \theta^Q)$ 也需要用于计算目标值 (Eq5)，自举导致的偏差传播容易让 $Q$ 发散

Our solution is similar to the target network used in (Mnih et al., 2013) but modified for actor-critic and using “soft” target updates, rather than directly copying the weights. We create a copy of the actor and critic networks, $Q^{\prime}(s,a|\theta^{Q^{\prime}})$ and $\mu^{\prime}(s|\theta^{\mu^{\prime}})$ respectively, that are used for calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks: $\theta^{\prime}\leftarrow\tau\theta+(1-$ $\tau)\theta^{\prime}$ with $\tau\ll1$ . This means that the target values are constrained to change slowly, greatly improving the stability of learning. 
>  我们将目标网络的思想用于 actor-critic，并且采用了 “软” 目标更新，而不是直接复制权重
>  我们分别为 actor 和 critic 创建拷贝网络 $Q'(s, a\mid \theta^{Q'}), \mu'(s\mid \theta^{\theta^{\mu'}})$，它们作为目标网络计算目标值，目标网络的权重将缓慢跟踪学习到的网络权重，即按照 $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ 更新 ($\tau \ll 1$)
>  这使得目标值的变化受到了限制，故大大提高了学习稳定性

This simple change moves the relatively unstable problem of learning the action-value function closer to the case of supervised learning, a problem for which robust solutions exist. We found that having both a target $\mu^{\prime}$ and $Q^{\prime}$ was required to have stable targets $y_{i}$ in order to consistently train the critic without divergence. This may slow learning, since the target network delays the propagation of value estimations. However, in practice we found this was greatly outweighed by the stability of learning. 
>  目标网络的使用将不稳定的自举学习向有监督学习的情况靠近 (即近似认为目标值和参数无关)
>  我们发现，要保持目标 $y_i$ 的稳定性，以稳定地训练 critic，避免发散，目标网络 $\mu', Q'$ 的使用是必须的
>  目标网络会减缓学习，因为它延迟了价值估计的传播 (目标网络的更新慢于原网络)，但实践中，稳定性优于学习速度

When learning from low dimensional feature vector observations, the different components of the observation may have different physical units (for example, positions versus velocities) and the ranges may vary across environments. This can make it difficult for the network to learn effectively and may make it difficult to find hyper-parameters which generalize across environments with different scales of state values. 
>  从低维特征向量观测中学习时，观测的不同部分/维度可能具有不同的物理单位 (例如位置和速度) ，以及可能在不同环境具有不同的范围，这会导致网络难以高效学习，并且难以找到能泛化到状态尺度不同的不同环境的超参数

One approach to this problem is to manually scale the features so they are in similar ranges across environments and units. We address this issue by adapting a recent technique from deep learning called batch normalization (Ioffe & Szegedy, 2015). This technique normalizes each dimension across the samples in a minibatch to have unit mean and variance. In addition, it maintains a running average of the mean and variance to use for normalization during testing (in our case, during exploration or evaluation).
>  我们的方法是手动缩放特征，使其在不同环境和单位中处于相似的范围，即采用了批量规范化
>  批量规范化将 minibatch 中的每一维度的值规范化到零均值和单位方差，并且维护一个均值和方差的 running average，该 running average 将在测试时用于规范化测试样本 (running average 近似表征了整个数据集的均值和方差)，在我们的情况下，测试时就是探索和评估时

 In deep networks, it is used to minimize covariance shift during training, by ensuring that each layer receives whitened input. In the low-dimensional case, we used batch normalization on the state input and all layers of the $\mu$ network and all layers of the $Q$ network prior to the action input (details of the networks are given in the supplementary material). With batch normalization, we were able to learn effectively across many different tasks with differing types of units, without needing to manually ensure the units were within a set range. 
 >  batch norm 通过确保每一层接收白化的输入，在训练时最小化协变量偏移
 >  在低维情况下，我们对状态输入以及 $\mu$ 网络的每一层、$Q$ 网络的每一层 (在动作输入之前) 都使用了 batch norm
 >  batch norm 确保了学习到的参数可以应用于多类任务

> [!info] Covariance Shift
> 协变量通常指输入特征，即用来预测目标变量的输入
> 协变量偏移即输入特征的分布发生了变化，例如在训练时，数据主要在 0-50 之间，但在测试时，数据主要在 100-150 之间
> 协变量偏移指训练时和测试时的输入特征的分布存在差异，这会导致模型无法泛化进而性能下降

> [!info] Whiten
> 白化是一类数据预处理技术，它通过线性变化，将数据转化为零均值和单位方差，且去除特征之间的相关性，白化后数据的协方差矩阵应该是单位阵，即特征之间是相互独立且标准化的

A major challenge of learning in continuous action spaces is exploration. An advantage of off-policies algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm. We constructed an exploration policy $\mu^{\prime}$ by adding noise sampled from a noise process $\mathcal{N}$ to our actor policy 

$$
\mu^{\prime}(s_{t})=\mu(s_{t}|\theta_{t}^{\mu})+\mathcal{N}\tag{7}
$$ 
$\mathcal{N}$ can be chosen to suit the environment. 

>  连续动作空间中，学习的一个挑战是探索
>  异策略算法的好处就是可以分离探索和学习算法 (因为行为策略和学习无关)
>  我们通过为 actor policy $\mu$ 添加噪声构造行为策略/探索策略 $\mu'$

As detailed in the supplementary materials we used an Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia (similar use of autocorrelated noise was introduced in (Wawrzynski, 2015)). 
>  我们使用了 Ornstein-Uhlenbeck 过程生成时序相关的探索，以提高带有惯性的物理控制问题的探索效率

# 4 Results

![](https://cdn-mineru.openxlab.org.cn/extract/release/82eb024c-6120-41e3-81bb-3b8df148eefc/95c195e1e78a7cd65861b10199ae7772665582c1569bdb6872ae3d0575ff1328.jpg) 

Figure 1: Example screenshots of a sample of environments we attempt to solve with DDPG. In order from the left: the cartpole swing-up task, a reaching task, a gasp and move task, a puck-hitting task, a monoped balancing task, two locomotion tasks and Torcs (driving simulator). We tackle all tasks using both low-dimensional feature vector and high-dimensional pixel inputs. Detailed descriptions of the environments are provided in the supplementary. Movies of some of the learned policies are available at https://goo.gl/J4PIAz. 

We constructed simulated physical environments of varying levels of difficulty to test our algorithm. This included classic reinforcement learning environments such as cartpole, as well as difficult, high dimensional tasks such as gripper, tasks involving contacts such as puck striking (canada) and locomotion tasks such as cheetah (Wawrzynski, 2009). In all domains but cheetah the actions were torques applied to the actuated joints. These environments were simulated using MuJoCo (Todorov et al., 2012). Figure 1 shows renderings of some of the environments used in the task (the supplementary contains details of the environments and you can view some of the learned policies at https://goo.gl/J4PIAz). 
>  我们构建了不同难度的模拟物理环境来测试我们的算法。这包括经典的强化学习环境，如 cartpole，以及复杂的高维任务，如机械手（gripper），涉及接触的任务，如冰球击打（加拿大）和运动任务，如猎豹（Wawrzynski，2009）。
>  在除了猎豹的所有领域中，动作是对驱动关节施加的扭矩。
>  这些环境使用 MuJoCo 进行模拟（Todorov 等，2012），图 1 显示了一些用于任务的环境渲染

In all tasks, we ran experiments using both a low-dimensional state description (such as joint angles and positions) and high-dimensional renderings of the environment. As in DQN (Mnih et al., 2013; 2015), in order to make the problems approximately fully observable in the high dimensional environment we used action repeats. For each timestep of the agent, we step the simulation 3 timesteps, repeating the agent’s action and rendering each time. Thus the observation reported to the agent contains 9 feature maps (the RGB of each of the 3 renderings) which allows the agent to infer velocities using the differences between frames. The frames were downsampled to 64x64 pixels and the 8-bit RGB values were converted to floating point scaled to [0, 1]. See supplementary information for details of our network structure and hyperparameters. 
>  每个任务都分别使用了低维特征向量 (例如关节角度、位置) 和高维的像素输入作为状态
>  为了让问题在高维环境中是近似全观测的，我们使用了 action repeats，即 agent 在每个时间步做出的动作会在之后的两个时间步重复做，环境则正常转移三个时间步，故对于 agent 来说，它的每个时间步的观测包含了 9 个特征图 (环境的每帧渲染是 RGB 图像)
>  特征图被下采样到 64x64，且 8 位 RGB 值被转换到 `[0, 1]` 之间

![](https://cdn-mineru.openxlab.org.cn/extract/release/82eb024c-6120-41e3-81bb-3b8df148eefc/9bc1a80a4cbe35e58332854780c37434eb88aba278a722a0cfdf42c6369c1bce.jpg) 

Figure 2: Performance curves for a selection of domains using variants of DPG: original DPG algorithm (minibatch NFQCA) with batch normalization (light grey), with target network (dark grey), with target networks and batch normalization (green), with target networks from pixel-only inputs (blue). Target networks are crucial. 

We evaluated the policy periodically during training by testing it without exploration noise. Figure 2 shows the performance curve for a selection of environments. We also report results with components of our algorithm (i.e. the target network or batch normalization) removed. In order to perform well across all tasks, both of these additions are necessary. In particular, learning without a target network, as in the original work with DPG, is very poor in many environments. 
>  策略会在训练时定期地被评估 (评估时不会给策略加噪声)
>  消融试验 (移除目标网络或 batch norm) 的结果也有报告，结果显示二者都是必须的，尤其是目标网络

Surprisingly, in some simpler tasks, learning policies from pixels is just as fast as learning using the low-dimensional state descriptor. This may be due to the action repeats making the problem simpler. It may also be that the convolutional layers provide an easily separable representation of state space, which is straightforward for the higher layers to learn on quickly. 

Table 1 summarizes DDPG’s performance across all of the environments (results are averaged over 5 replicas). We normalized the scores using two baselines. The first baseline is the mean return from a naive policy which samples actions from a uniform distribution over the valid action space. The second baseline is iLQG (Todorov & Li, 2005), a planning based solver with full access to the underlying physical model and its derivatives. We normalize scores so that the naive policy has a mean score of 0 and iLQG has a mean score of 1. DDPG is able to learn good policies on many of the tasks, and in many cases some of the replicas learn policies which are superior to those found by iLQG, even when learning directly from pixels. 

![](https://cdn-mineru.openxlab.org.cn/extract/release/82eb024c-6120-41e3-81bb-3b8df148eefc/e0d1e33e88ede88350bc308125d608a9b7af156488c8ccc66c72e4ddb23a07e8.jpg) 
Figure 3: Density plot showing estimated $\mathsf Q$ values versus observed returns sampled from test episodes on 5 replicas. In simple domains such as pendulum and cartpole the Q values are quite accurate. In more complex tasks, the Q estimates are less accurate, but can still be used to learn competent policies. Dotted line indicates unity, units are arbitrary. 

It can be challenging to learn accurate value estimates. Q-learning, for example, is prone to overestimating values (Hasselt, 2010). We examined DDPG’s estimates empirically by comparing the values estimated by $Q$ after training with the true returns seen on test episodes. Figure 3 shows that in simple tasks DDPG estimates returns accurately without systematic biases. For harder tasks the Q estimates are worse, but DDPG is still able learn good policies. 
>  准确估计价值函数具有挑战性，例如 Q-Learning 倾向于高估价值
>  我们将训练后的 Q 函数估计的价值和测试回合中的实际回报进行了经验比较
>  Figure 3 显示，在简单的任务中，DDPG 的价值估计准确，没有系统偏差，但困难任务上，价值估计效果差，但 DDPG 仍可以学习到良好策略

To demonstrate the generality of our approach we also include Torcs, a racing game where the actions are acceleration, braking and steering. Torcs has previously been used as a testbed in other policy learning approaches (Koutnik et al., 2014b). We used an identical network architecture and learning algorithm hyper-parameters to the physics tasks but altered the noise process for exploration because of the very different time scales involved. On both low-dimensional and from pixels, some replicas were able to learn reasonable policies that are able to complete a circuit around the track though other replicas failed to learn a sensible policy. 
>  为了展示我们方法的通用性，我们还引入了 Torcs，这是一款赛车游戏，其操作包括加速、刹车和转向
>  Torcs 之前已被用作其他策略学习方法的测试平台（Koutnik 等，2014b）
>  我们使用了和物理任务中相同的网络架构和学习算法超参数，但由于涉及的时间尺度非常不同，因此调整了探索的噪声过程
>  在低维数据以及从像素级别进行学习的情况下，一些实例能够学会合理的策略，可以完成赛道一圈的行驶，但其他实例未能学会有意义的策略

# 5 Related Work
The original DPG paper evaluated the algorithm with toy problems using tile-coding and linear function approximators. It demonstrated data efficiency advantages for off-policy DPG over both on- and off-policy stochastic actor critic. It also solved one more challenging task in which a multijointed octopus arm had to strike a target with any part of the limb. However, that paper did not demonstrate scaling the approach to large, high-dimensional observation spaces as we have here. 
>  原始的 DPG 论文通过使用平铺编码和线性函数逼近器来评估该算法在玩具问题上的表现。它展示了与 on-policy 和 off-policy 的 stochastic actor-critic 相比，off-policy 的 DPG 在数据效率方面的优势。此外，它还解决了一个更具挑战性的任务，即一个多关节章鱼手臂必须用肢体的任何部分击中目标。
>  然而，该论文并未展示将这种方法扩展到大规模、高维观测空间的能力，而这是我们在这里所实现的。

It has often been assumed that standard policy search methods such as those explored in the present work are simply too fragile to scale to difficult problems (Levine et al., 2015). Standard policy search is thought to be difficult because it deals simultaneously with complex environmental dynamics and a complex policy. Indeed, most past work with actor-critic and policy optimization approaches have had difficulty scaling up to more challenging problems (Deisenroth et al., 2013). Typically, this is due to instability in learning wherein progress on a problem is either destroyed by subsequent learning updates, or else learning is too slow to be practical. 
>  人们常常认为，像本研究中探讨的标准策略搜索方法（Levine 等人，2015）这样的方法在面对复杂问题时无法扩展。事实上，大多数过去的 actor-critic 和策略优化方法在扩展到更具挑战性的问题时都遇到了困难（Deisenroth 等人，2013）
>  通常，这是因为学习过程中的不稳定性导致问题上的进展要么被后续的学习更新破坏，要么学习速度太慢而无法实际应用。

Recent work with model-free policy search has demonstrated that it may not be as fragile as previously supposed. Wawrzynski (2009); Wawrzynski & Tanwani (2013) has trained stochastic policies in an actor-critic framework with a replay buffer. Concurrent with our work, Balduzzi & Ghifary (2015) extended the DPG algorithm with a “deviator” network which explicitly learns $\partial Q/\partial a$ . However, they only train on two low-dimensional domains. Heess et al. (2015) introduced SVG(0) which also uses a Q-critic but learns a stochastic policy. DPG can be considered the deterministic limit of SVG(0). The techniques we described here for scaling DPG are also applicable to stochastic policies by using the re-parametrization trick (Heess et al., 2015; Schulman et al., 2015a). 
>  最近关于无模型策略搜索的工作表明，它可能不像以前认为的那样脆弱
>  Wawrzynski 和 Tanwani（2013）在 actor-critic 框架中使用 replay buffer 训练了随机策略。Balduzzi 和 Ghifary（2015）通过引入一个“偏差网络”扩展了 DPG 算法，该网络显式地学习 $\partial Q/\partial a$。然而，他们仅在两个低维域上进行了训练。Heess 等人（2015）提出了 SVG (0)，它同样使用 Q-critic 但学习了一个随机策略。
>  DPG 可以被视为 SVG (0) 的确定性极限。我们在这里描述的用于扩展 DPG 的技术也可以通过重参数化技巧应用于随机策略（Heess 等，2015；Schulman 等，2015a）。

Another approach, trust region policy optimization (TRPO) (Schulman et al., 2015b), directly constructs stochastic neural network policies without decomposing problems into optimal control and supervised phases. This method produces near monotonic improvements in return by making carefully chosen updates to the policy parameters, constraining updates to prevent the new policy from diverging too far from the existing policy. This approach does not require learning an action-value function, and (perhaps as a result) appears to be significantly less data efficient. 
>  另一种方法是置信域策略优化（Trust Region Policy Optimization, TRPO）（Schulman 等，2015b），该方法通过对策略参数进行精心选择的更新，并限制更新以防止新策略与现有策略偏离过大，从而实现接近单调的回报提升。这种方法不需要学习动作价值函数，（或许正是因为这一点）似乎在数据效率方面显著低于其他方法。
>  (不知道作者到底是怎么理解 TRPO 的)

To combat the challenges of the actor-critic approach, recent work with guided policy search (GPS) algorithms (e.g., (Levine et al., 2015)) decomposes the problem into three phases that are relatively easy to solve: first, it uses full-state observations to create locally-linear approximations of the dynamics around one or more nominal trajectories, and then uses optimal control to find the locally-linear optimal policy along these trajectories; finally, it uses supervised learning to train a complex, non-linear policy (e.g. a deep neural network) to reproduce the state-to-action mapping of the optimized trajectories. 
>  为了解决 actor-critic 方法所面临的挑战，近期基于引导式策略搜索（GPS）算法的工作，例如，(Levine et al., 2015) 将问题分解为三个相对容易解决的阶段：首先，它利用完整的状态观测值，在一个或多个基准轨迹周围创建环境动态的局部线性近似；然后，使用最优控制来找到这些轨迹上的局部线性最优策略；最后，采用监督学习训练一个复杂的非线性策略（例如深度神经网络），以重现优化轨迹的状态到动作的映射关系。

This approach has several benefits, including data efficiency, and has been applied successfully to a variety of real-world robotic manipulation tasks using vision. In these tasks GPS uses a similar convolutional policy network to ours with 2 notable exceptions: 1. it uses a spatial softmax to reduce the dimensionality of visual features into a single $(x,y)$ coordinate for each feature map, and 2. the policy also receives direct low-dimensional state information about the configuration of the robot at the first fully connected layer in the network. Both likely increase the power and data efficiency of the algorithm and could easily be exploited within the DDPG framework. 
>  这种方法有几个优点，包括数据高效性，并且已经成功应用于多种基于视觉的真实世界机器人操作任务。在这些任务中，全局路径规划器（GPS）使用了一个类似于我们的卷积策略网络，但有两个显著的区别：1. 它使用空间软最大值（spatial softmax）将视觉特征的维度降低为每个特征图的一个单一的 $(x, y)$ 坐标；2. 策略在网络的第一个全连接层还接收关于机器人配置的直接低维状态信息。这两个特点可能增强了算法的性能和数据效率，也可以轻松地在深度确定性策略梯度（DDPG）框架中加以利用。

PILCO (Deisenroth & Rasmussen, 2011) uses Gaussian processes to learn a non-parametric, probabilistic model of the dynamics. Using this learned model, PILCO calculates analytic policy gradients and achieves impressive data efficiency in a number of control problems. However, due to the high computational demand, PILCO is “impractical for high-dimensional problems" (Wahlstrom et al., 2015). It seems that deep function approximators are the most promising approach for scaling reinforcement learning to large, high-dimensional domains. 
>  PILCO（Deisenroth & Rasmussen，2011）使用高斯过程来学习环境动态的非参数化、概率模型。利用这个学习到的模型，PILCO 计算出解析形式的策略梯度，并在多个控制问题中实现了令人印象深刻的样本效率。
>  然而，由于计算需求较高，PILCO 被认为“对高维问题不切实际”（Wahlstrom 等人，2015）。似乎深度函数逼近器是将强化学习扩展到大规模、高维域的最有前景的方法。

Wahlstrom et al. (2015) used a deep dynamical model network along with model predictive control to solve the pendulum swing-up task from pixel input. They trained a differentiable forward model and encoded the goal state into the learned latent space. They use model-predictive control over the learned model to find a policy for reaching the target. However, this approach is only applicable to domains with goal states that can be demonstrated to the algorithm. 
>  Wahlstrom 等人（2015）通过结合深度动态模型网络和模型预测控制解决了从像素输入的摆锤摆动任务。他们训练了一个可微分的前向模型，并将目标状态编码到学习到的潜在空间中。然后，他们通过对学习到的模型进行模型预测控制来找到达到目标状态的策略。然而，这种方法仅适用于那些可以向算法演示目标状态的领域。

Recently, evolutionary approaches have been used to learn competitive policies for Torcs from pixels using compressed weight parametrizations (Koutnik et al., 2014a) or unsupervised learning (Koutnik et al., 2014b) to reduce the dimensionality of the evolved weights. It is unclear how well these approaches generalize to other problems. 
>  最近，进化方法被用来从像素输入中为 Torcs 学习竞争性策略，使用压缩权重参数化（Koutnik 等人，2014a）或无监督学习（Koutnik 等人，2014b）来降低演化权重的维度。目前尚不清楚这些方法在其他问题上的泛化能力如何。

# 6 Conclusion
The work combines insights from recent advances in deep learning and reinforcement learning, resulting in an algorithm that robustly solves challenging problems across a variety of domains with continuous action spaces, even when using raw pixels for observations. As with most reinforcement learning algorithms, the use of non-linear function approximators nullifies any convergence guarantees; however, our experimental results demonstrate that stable learning without the need for any modifications between environments. 
>  该工作结合了 DQN 的成功经验，提出了解决连续动作空间上控制问题的算法
>  虽然使用 NN 在理论上没有收敛保证，但是试验结果证明了学习是稳定的

>  该工作可以总结为 inspiration from DQN, and apply it to DPG

Interestingly, all of our experiments used substantially fewer steps of experience than was used by DQN learning to find solutions in the Atari domain. Nearly all of the problems we looked at were solved within 2.5 million steps of experience (and usually far fewer), a factor of 20 fewer steps than DQN requires for good Atari solutions. This suggests that, given more simulation time, DDPG may solve even more difficult problems than those considered here. 
>  DDPG 的收敛步骤显著少于 DQN，大约所有问题在 250 万步解决，这比 DQN 少了 20 多倍

A few limitations to our approach remain. Most notably, as with most model-free reinforcement approaches, DDPG requires a large number of training episodes to find solutions. However, we believe that a robust model-free approach may be an important component of larger systems which may attack these limitations (Glascher et al., 2010). 

# Supplementary Information
# 7 Experiment Details
We used Adam (Kingma & Ba, 2014) for learning the neural network parameters with a learning rate of $10^{-4}$ and $10^{-3}$ for the actor and critic respectively. 

For $Q$ we included $L_{2}$ weight decay of $10^{-2}$ and used a discount factor of $\gamma=0.99$ . For the soft target updates we used $\tau=0.001$ . 

The neural networks used the rectified non-linearity (Glorot et al., 2011) for all hidden layers. The final output layer of the actor was a tanh layer, to bound the actions. 

The low-dimensional networks had 2 hidden layers with 400 and 300 units respectively $(\approx130{,}000$ parameters). Actions were not included until the 2nd hidden layer of $Q$ . When learning from pixels we used 3 convolutional layers (no pooling) with 32 filters at each layer. This was followed by two fully connected layers with 200 units $(\approx430{,}000$ parameters). 

The final layer weights and biases of both the actor and critic were initialized from a uniform distribution $[-3^{'}\times10^{-3^{'}},3^{\times}10^{-3}]$ and $[3\times10^{-4},3\times10^{-4}]$ for the low dimensional and pixel cases respectively. This was to ensure the initial outputs for the policy and value estimates were near zero. The other layers were initialized from uniform distributions $[-\frac{1}{\sqrt{f}},\frac{1}{\sqrt{f}}]$ where $f$ is the fan-in of the layer. The actions were not included until the fully-connected layers. We trained with minibatch sizes of 64 for the low dimensional problems and 16 on pixels. We used a replay buffer size of $10^{6}$ . 

For the exploration noise process we used temporally correlated noise in order to explore well in physical environments that have momentum. We used an Ornstein-Uhlenbeck process (Uhlenbeck $\&$ Ornstein, 1930) with $\theta=0.15$ and $\sigma=0.2$ . The Ornstein-Uhlenbeck process models the velocity of a Brownian particle with friction, which results in temporally correlated values centered around 0. 

# 8 Planning Algorithm
Our planner is implemented as a model-predictive controller (Tassa et al., 2012): at every time step we run a single iteration of trajectory optimization (using iLQG, (Todorov & Li, 2005)), starting from the true state of the system. Every single trajectory optimization is planned over a horizon between 250ms and $600\mathrm{ms}$ , and this planning horizon recedes as the simulation of the world unfolds, as is the case in model-predictive control. 

The iLQG iteration begins with an initial rollout of the previous policy, which determines the nominal trajectory. We use repeated samples of simulated dynamics to approximate a linear expansion of the dynamics around every step of the trajectory, as well as a quadratic expansion of the cost function. We use this sequence of locally-linear-quadratic models to integrate the value function backwards in time along the nominal trajectory. This back-pass results in a putative modification to the action sequence that will decrease the total cost. We perform a derivative-free line-search over this direction in the space of action sequences by integrating the dynamics forward (the forwardpass), and choose the best trajectory. We store this action sequence in order to warm-start the next iLQG iteration, and execute the first action in the simulator. This results in a new state, which is used as the initial state in the next iteration of trajectory optimization. 

# 9 Environment Details 
## 9.1 Torcs Environment 
For the Torcs environment we used a reward function which provides a positive reward at each step for the velocity of the car projected along the track direction and a penalty of $-1$ for collisions. Episodes were terminated if progress was not made along the track after 500 frames. 

## 9.2 MuJoCo Environments 
For physical control tasks we used reward functions which provide feedback at every step. In all tasks, the reward contained a small action cost. For all tasks that have a static goal state (e.g. pendulum swingup and reaching) we provide a smoothly varying reward based on distance to a goal state, and in some cases an additional positive reward when within a small radius of the target state. For grasping and manipulation tasks we used a reward with a term which encourages movement towards the payload and a second component which encourages moving the payload to the target. In locomotion tasks we reward forward action and penalize hard impacts to encourage smooth rather than hopping gaits (Schulman et al., 2015b). In addition, we used a negative reward and early termination for falls which were determined by simple threshholds on the height and torso angle (in the case of walker2d). 
