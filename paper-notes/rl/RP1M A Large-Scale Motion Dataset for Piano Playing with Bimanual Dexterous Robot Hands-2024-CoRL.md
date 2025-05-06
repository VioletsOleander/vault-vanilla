# Abstract
It has been a long-standing research goal to endow robot hands with human-level dexterity. Bimanual robot piano playing constitutes a task that combines challenges from dynamic tasks, such as generating fast while precise motions, with slower but contact-rich manipulation problems. 
>  双手机器人弹钢琴是一项结合了动态任务挑战 (例如生成快速而精确的动作) 和较慢单接触丰富的操作的问题

Although reinforcement learning-based approaches have shown promising results in single-task performance, these methods struggle in a multi-song setting. Our work aims to close this gap and, thereby, enable imitation learning approaches for robot piano playing at scale. 
>  基于 RL 的方法在单一任务上表现出色，但在多曲子的设定下则不行
>  我们的工作旨在弥合这一差距，实现大规模的机器人钢琴演奏模仿学习方法

To this end, we introduce the Robot Piano 1 Million (RP1M) dataset, containing bimanual robot piano playing motion data of more than one million trajectories. We formulate finger placements as an optimal transport problem, thus, enabling automatic annotation of vast amounts of unlabeled songs. 
>  为此，我们推出 Robot Piano 1 Million 数据集，它包含了超过 1 百万条机器人演奏钢琴的动作轨迹数据
>  我们将手指放置问题建模为最优传输问题，从而实现了对大量未标记歌曲的自动注释

Benchmarking existing imitation learning approaches shows that such approaches reach promising robot piano playing performance by leveraging RP1M. 
>  对现存的模仿学习方法进行的基准测试表明，使用 RP1M 数据集，这些方法可以达到令人满意的机器人钢琴演奏表现

Keywords: Bimanual dexterous robot hands, dataset for robot piano playing, imitation learning, robot learning at scale 

# 1 Introduction 
Empowering robots with human-level dexterity is notoriously challenging. Current robotics research on hand and arm motions focuses on manipulation and dynamic athletic tasks. Manipulation, such as grasping or reorienting [1], requires continuous application of acceptable forces at moderate speeds to various objects with distinct shapes and weight distributions. Environmental changes, like humidity or temperature, alter the already complex contact dynamics, which adds to the complexity of manipulation tasks. Dynamic tasks, like juggling [2] and table tennis [3] involve making and breaking contact, demanding high precision and tolerating less inaccuracy due to rarer contacts. High speeds in these tasks necessitate greater accelerations and introduce a precision-speed tradeoff. 
>  当前关于机器人手部和手臂的研究主要集中在操作任务和动态运动任务上
>  操作任务，例如抓取或重新定向物体，需要在中等的速度下对具有不同形状和重量分布的各种物体持续施加可接受的力，环境变化，例如湿度或温度会改变本已复杂的接触动力学，从而进一步增加操作任务的复杂性
>  动态任务，例如抛接球和乒乓球涉及接触与非接触的快速切换，由于接触的机会相对较少，这些任务要求高精度且容忍更少的误差，这些任务的高速特性还要求了更大的加速度，并引入了精度和速度指尖的权衡

Robot piano playing combines various aspects of dynamic and manipulation tasks: the agent is required to coordinate multiple fingers to precisely press keys for arbitrary songs, which is a high-dimensional and rich control task. At the same time, the finger motions have to be highly dynamic, especially for songs with fast rhythms. 
>  机器人弹钢琴结合了动态任务和操作任务的多个方面: agent 需要协调多个手指，并且精确地为任意曲子按下对应琴键，这是一个高维且复杂的控制任务
>  同时，手指动作需要是高度动态的，尤其是对于节奏较快的曲目

Well-practiced pianists can play arbitrary songs, but this level of generalization is extremely challenging for robots. In this work, we build the foundation to develop methods capable of achieving human-level bi-manual dexterity at the intersection of manipulation and dynamic tasks, while reaching such generalization capabilities in multi-task environments. 
>  对于机器人来说，演奏任意歌曲的泛化性极具挑战性
>  本工作中，我们构建了实现能够在操作任务和动作任务的交叉领域上达到人类水平双手灵巧性的方法的基础，同时具备在多任务环境下的泛化能力

While reinforcement learning (RL) is a promising direction, traditional RL approaches often struggle to achieve excellent performance in multi-task settings [4]. The advent of scalable imitation learning techniques [5] enables representing complex and multi-modal distributions. Such large models are most effective when trained on massive datasets that combine the state evolution with the corresponding action trajectories. 
>  传统 RL 方法在多任务设置下不容易达到优秀的性能，而可拓展的模仿学习方法则使得表示复杂，多模态的分布成为可能，这样的大型模型在训练于结合了状态演化和相应的动作轨迹上的数据集上将更加有效

So far, creating large datasets for robot piano play is problematic due to the time-consuming fingering annotations. Fingering annotations map which finger is supposed to press a particular piano key at each time step. With fingering information, the reward is less sparse, making the training significantly more effective. These labels usually require expert human annotators [6], preventing the agent from leveraging the large amounts of unlabeled music pieces on the internet [7]. Besides, human-labeled fingering may be infeasible for robots with morphologies different from human hands, such as different numbers of fingers or distinct hand dimensions. 
>  目前为止，为机器人钢琴演奏创建大型数据集仍然存在困难，因为指法标注非常耗时
>  指法标注指明了特定时间步中，那个手指应该按下那个琴键，有了手指信息后，奖励信号将更加密集，使得训练可以更加高效
>  指法标注通常需要专家标注员进行标注，并且，人类标注的指法对于手部形态和人类不同的机器人 (例如不同数量的手指或不同的手部尺寸) 可能是不可用的

In this paper, we propose the Robot Piano 1 Million dataset (RP1M). This dataset comprises the motion data of high-quality bi-manual robot piano play. In particular, we train RL agents for each of the 2k songs and roll out each policy 500 times with different random seeds. 
>  本文提出 RP1M，该数据集包含了高质量的双手机器人演奏动作数据，具体地说，我们为 2k 首曲子各自训练了 RL agent，然后让每个 agent 使用不同的随机种子演示了对应曲子 500 次

To enable the generation of RP1M, we introduce a method to learn the fingering automatically by formulating finger placement as an optimal transport (OT) problem [8, 9]. Intuitively, the fingers are placed in a way such that the correct keys are pressed while the overall moving distance of the fingers is minimized. 
>  为了构建 RP1M，我们将手指放置问题建模为最优传输问题，以自动为无指法标记的数据中构建指法标记，直观上说，手指的放置方式应该让正确的琴键被按下，并且手指的总移动距离应该最小化

Agents trained using our automatic fingering match the performance of agents trained with human-annotated fingering labels. Besides, our method is easy to implement with almost no extra training time. The automatic fingering also allows learning piano playing with different embodiments, such as robots with four fingers only. 
>  使用我们自动指法标记训练的 agent 与使用人工指法标记训练的 agent 表现相当
>  并且，我们的自动标记方法易于实现，几乎不需要额外的训练时间
>  自动指法标记也允许用不同的身体形态学习钢琴演奏，例如只有四个手指的机器人

With RP1M, it is now possible to train and test imitation learning approaches at scale. We benchmark various behavior cloning approaches and find that using RP1M, existing methods perform better in terms of generalization capabilities in multi-song piano play. 
>  有了 RP1M，现在可以大规模训练和测试模仿学习方法，我们测试了许多行为克隆方法，并发现使用 RP1M，现存的方法在演奏多首曲目是都有更强的泛化性


This work contributes in various ways: 

- To facilitate the research on dexterous robot hands, we release RP1M, a dataset of piano playing motions that includes more than $2{k}$ music pieces with expert trajectories generated by our agents. 
- We formulate fingering as an optimal transport problem, enabling the generation of vast amounts of robot piano data with RL, as well as allowing variations in the embodiment. 
- Using RP1M, we benchmark various approaches to robot piano playing, whereby existing imitation learning approaches reach promising results in motion synthesis on novel music pieces due to scaling with RP1M. 

>  本工作的贡献包括:
>  - 为了促进对灵巧机器人手的研究，我们发布了 RP1M 数据集，该数据集包含超过 2000 首音乐作品的钢琴演奏轨迹，演奏轨迹由我们训练的专家 agent 生成
>  - 我们将指法问题形式化为最优传输问题，使得通过强化学习生成大规模机器人演奏数据成为可能，并允许机器人形态上的变化
>  - 我们使用 RP1M 测试了机器人钢琴演奏的各种方法，归功于 RP1M 的规模，现存的模仿学习方法在新音乐作品的动作合成上取得了很好的结果

# 2 Related Work 
**Piano Playing with Robots** Piano playing with robot hands has been investigated for decades. It is a challenging task since bimanual robot hands should precisely press the right keys at the right time, especially considering its high-dimensional action space. Previous methods require specific robot designs [10, 11, 12, 13, 14, 15] or trajectory pre-programming [16, 17]. Recent methods enable piano playing with dexterous hands through planning [18] or RL [19] but are limited to simple music pieces. RoboPianist [4] introduces a benchmark for robot piano playing and demonstrates strong RL performance, but requires human fingering labels and performs worse in multi-task learning. Yang et al. [20] improves the policy training performance by considering the bionic constraints of the anthropomorphic robot hands. 
>  RoboPianist 使用 RL 实现机器人弹钢琴，并提出了一个机器人弹钢琴的基准，但其方法需要人类标记的指法标签，并且在多任务学习中表现较差
>  [20] 通过考虑了仿生机器人手的仿生约束，提高了策略训练表现

Human fingering informs the agent of the correspondence between fingers and pressed keys at each time step. These labels require expert annotators and are, therefore, expensive to acquire in practice. 
>  人类标记的指法告诉了 agent 每个时间步上手指和需要按压的键的对应关系
>  这些指法标记需要专家标记，因此获取成本较高

Several approaches learn fingering from human-annotated data with different machine learning methods [6, 21, 22]. Moryossef et al. [23] extract fingering from videos to acquire fingering labels cheaply. Ramoneda et al. [24] proposes to treat piano fingering as a sequential decision-making problem and use RL to calculate fingering but without considering the model of robot hands. Shi et al. [25] automatically acquires fingering via dynamic programming, but the solution is limited to simple tasks. Concurrent work [26] obtains fingering labels from YouTube videos and trains a diffusion policy to play hundreds of songs. 
>  一些方法利用不同的机器学习技术从人工标注的数据中学习指法
>  [23] 从视频中提取指法标记; [24] 提出将钢琴指法安排视作顺序决策问题，使用 RL 计算指法，但未考虑机器人手部模型; [25] 通过动态规划自动获取指法，但解决方案仅限于简单的任务; [26] 从 YouTube 视频中获取指法标签，并训练 diffusion 策略来演奏数百首歌曲

In our paper, we do not introduce a separate fingering model, instead, similar to human pianists, fingering is discovered automatically while playing the piano, hereby largely expanding the pool of usable data to train a generalist piano-playing agent. 
>  我们并不提出一个单独的指法模型，而是类似于人类演奏家，在演奏钢琴的过程中自动发现指法，从而大大扩展了可用于训练专用演奏 agent 的数据池

**Datasets for Dexterous Robot Hands** Most large-scale datasets of dexterous robot hands focus on grasping various objects. To get suitable grasp positions, some methods utilize planners [27, 28, 29], while others use learned grasping policies [30], or track grasping motions of humans and imitate these motions on a robot hand [31]. 
>  大多数大规模灵巧机器人手的数据集集中在抓取各种物体上
>  为了获取合适的抓取位置，一些方法使用 planners，其他方法使用学习到的抓取策略，或者跟踪人类的抓取动作并且在机器人手上模仿这些动作

Compared to the abundance of datasets for grasping, there exist relatively few datasets for object manipulation with dexterous robot hands. The D4RL benchmark [32] provides small sets of expert trajectories for four such tasks, consisting of human demonstrations and rollouts of trained policies. Zhao et al. [33] provides a small object manipulation dataset that utilizes a low-cost bimanual platform with simple parallel grippers. Chen et al. [34] collects offline datasets for two simulated bimanual manipulation tasks with dexterous hands. Furthermore, Fan et al. [35] proposes a large-scale dataset for bimanual hand-object manipulation with human hands rather than robot hands. 
>  相比之下，现存的灵巧物体操作相关的数据集则较少
>  D4RL 为四类灵巧操作任务提供了少量专家轨迹，包括了人类演示和训练好的策略的 rollout
>  [33] 利用低成本的双臂平台和简单的平行夹具提供了一个小型物体操作数据集
>  [34] 收集了两个模拟双臂操作任务的离线数据集
>  [35] 提出了使用人类手部收集的双手物体操作数据集

![[pics/RP1M-Table1.png]]

Table 1 summarizes the characteristics of these existing datasets. To the best of our knowledge, our RP1M dataset is the first large-scale dataset of dynamic, bimanual piano playing with dexterous robot hands. 

We further discuss related work on dexterous robot hands and generalist agents in Appendix A. 

# 3 Background 
**Task Setup** The simulated piano-playing environment is built upon RoboPianist [4]. It includes a robot piano-playing setup, an RL-based agent for playing piano with simulated robot hands, and a multi-task learner. To avoid confusion, we refer to these components as RoboPianist, RoboPianist-RL, and RoboPianist-MT, respectively. 
>  Task Setup
>  模拟演奏环境基于 RoboPianist 构建，它包含一个演奏设定、一个基于 RL 的演奏 agent、一个多任务学习者
>  我们将这些组件分别称为 RoboPianist, RoboPianist-RL, RoboPianist-MT

The piano playing environment features a full-size keyboard with 88 keys driven by linear springs, two Shadow robot hands [36], and a pseudo sustain pedal. 
>  演奏环境包含一个全尺寸钢琴，具有 88 个由线性弹簧驱动的键，两个 Shadow robot hands 和一个伪延音踏板

Sheet music is represented by Musical Instrument Digital Interface (MIDI) transcription. Each time step in the MIDI file specifies which piano keys to press (active keys). The goal of a piano-playing agent is to press active keys and avoid inactive keys under space and time constraints. This requires the agent to coordinate its fingers and place them properly in a highly dynamic scenario such that target keys, at not only the current time step but also the future time steps, can be pressed accurately and timely. 
>  乐谱由 MIDI 转录表示，MIDI 文件的每个时间步指定了需要按下的琴键 (active keys)
>  演奏 agent 的目标是在时间和空间限制下按下 active keys 并避免 inactive keys

The original RoboPianist uses MIDI files from the PIG dataset [6] which includes human fingering information annotated by experts. However, as mentioned earlier, this limits the agent to only play human-labeled music pieces, and the human annotation may not be suitable for robots due to the different morphologies. 
>  RoboPianist 使用 PIG 数据集的 MIDI 文件，它包含了由专家标注的人类指法信息
>  但是，如前所述，使用人类标注存在限制

The observation includes the state of the two robot hands, fingertip positions, piano sustain state, piano key states, and a goal vector, resulting in an 1144-dimensional observation space. The goal includes 10-step active keys and 10-step target sustain states obtained from the MIDI file, represented by a binary vector. 
>  observation 包含了两只机器手的状态、指尖位置、钢琴延音踏板状态、琴键状态、一个目标向量，故总的 observation 空间为 1144 维
>  目标向量包含了从 MIDI 文件中获取的未来 10 步 active keys 和延音踏板状态

RoboPianst further includes 10-step human-labeled fingering in the observation space but we remove this observation in our method since we do not need human-labeled fingering. 
>  RoboPianist 进一步将未来 10 步的人类标记的指法也加入 observation 空间，但我们移除了这一点，因为我们不需要人类标记的指法

For the action space, we remove the DoFs that do not exist in the human hand or are used in most songs, resulting in a 39-dimensional action space, consisting of the joint positions of the robot hands, the positions of forearms, and a sustain pedal. 
>  对于动作空间，我们移除了人类手上不存在的或者在大多数曲子中没有用到的自由度，将动作空间维度减少到 39 维，其中包括了机器手的关节位置、前臂位置和一个延音踏板

We evaluate the performance of the trained agent with an average F1 score calculated by $F_1 = 2\cdot\frac {precision\cdot recall}{precision + recall}$ . For piano playing, recall and precision measure the agent’s performance on pressing the active keys and avoiding inactive keys respectively [4]. 
>  我们通过计算平均 F1 score 来评估 agent 的性能

**Playing Piano with RL** We use RL to train specialist agents per song to control the bimanual dexterous robot hands to play the piano. We frame the piano playing task as a finite Markov Decision Process (MDP). 
>  Playing Piano with RL
>  我们使用 RL 训练每首歌的专用 agent，我们将演奏任务建模为 MDP

At time step $t$ , the agent $\pi_{\boldsymbol{\theta}}\big(a_{t}|\boldsymbol{s}_{t}\big)$ , parameterized by $\theta$ , receives state $s_{t}$ and takes action $a_{t}$ to interact with the environment and receives new state $s_{t+1}$ and reward ${r}_{t}$ . The state and action spaces are described above and the reward ${r}_{t}$ gives an immediate evaluation of the agent’s behavior. We will introduce reward terms used for training in Section 4.1.
>  状态和动作空间都由上所述，奖励 $r_t$ 将给出对 agent 行为的立即评估

The agent’s goal is to maximize the expected cumulative rewards over an episode of length $H$ , defined as $\begin{array}{r}{{\mathcal{J}}=\mathbb{E}_{\pi_{\theta}}\Big[\sum_{t=0}^{H}\gamma^{t}r_{t}\big(s_{t},a_{t}\big)\Big]}\end{array}$ , where $\gamma$ is a discount factor ranging from 0 to 1. 
>  agent 的目标是最大化长度为 $H$ 的回合的累积奖励

# 4 Large-Scale Motion Dataset Collection 
In this section, we describe our RP1M dataset in detail. We first introduce how to train a specialist piano-playing agent without human fingering labels. Removing the requirement of human fingering labels allows the agent to play any sheet music available on the Internet (under copyright license). We then analyze the performance of our specialist RL agent as well as the learned fingering. Lastly, we introduce our collected large-scale motion dataset, RP1M, which includes ${\sim}1\mathrm{m}$ expert trajectories for robot piano playing, covering ${\sim}2\mathrm{k}$ pieces of music. 

## 4.1 Piano Playing without Human Fingering Labels 
To mitigate the hard exploration problem posed by the sparse rewards, RoboPianist-RL adds dense reward signals by using human fingering labels. Fingering informs the agent of the “ground-truth” fingertip positions, and the agent minimizes the Euclidean distance between the current fingertip positions and the “ground-truth” positions. We now discuss our OT-based method to lift the requirement of human fingering. 
>  为了缓解稀疏奖励带来的 exploration 问题，RoboPianist-RL 使用人类指法标签添加了密集的奖励信号
>  指法标签告诉了 agent 真实的指尖位置，agent 最小化当前指尖和真实的指尖 z 指尖的欧式距离
>  我们讨论基于最优传输的方法，以消除对人类指法标记的需求

Although fingering is highly personalized, generally speaking, it helps pianists to press keys timely and efficiently. Motivated by this, apart from maximizing the key pressing rewards, we also aim to minimize the moving distances of fingers. Specifically, at time step $t$ , for the $i$ -th key $k^{i}$ to press, we use the $j$ -th finger $f^{j}$ to press this key such that the overall moving cost is minimized. 
>  在训练时，除了最大化按键奖励以外，我们还最小化手指的移动距离
>  具体地说，在时间步 $t$，对于要按下的第 $i$ 个键 $k^i$，按下该键的第 $j$ 个手指 $f^j$ 需要满足能够最小化总的移动开销

We define the minimized cumulative moving distance between fingers and target keys as $d_{t}^{\mathrm{OT}}\in\mathbb{R}^{+}$ , given by 

$$
\begin{align}
d_t^{OT} &= \min_{w_t}\sum_{(i,j)\in K_t\times F}w_t(k^i, f^j)\cdot c_t(k^i, f^j),\\
&s.t.\ i)\ \sum_{j\in F} w_t(k^i, f^j) = 1\quad \text{for}\ i\in K_t\\
&\quad\quad ii)\ \sum_{i\in K_t}w_t(k^i, f^j)\le 1\quad\text{for}\ j\in F\\
&\quad\quad iii)\ w_t(k^i,f^j)\in \{0,1\}\quad \text{for}\ (i,j)\in K_t\times F
\end{align}\tag{1}
$$

$K_{t}$ represents the set of keys to press at time step $t$ and $F$ represents the fingers of the robot hands. $c_{t}(k^{i},f^{j})$ represents the cost of moving finger $f^{j}$ to piano key $k^{i}$ at time step $t$ calculated by their Euclidean distance. $w_{t}(k^{i},f^{j})$ is a boolean weight. 

>  我们定义手指与目标键指尖的最小累积移动距离 $d_t^{OT}\in \mathbb R^+$ 如上
>  其中 $K_t$ 表示时间步 $t$ 时需要按下的 keys 的集合，$F$ 表示机器人的手指；$c_t(k^i, f^j)$ 表示将手指 $f^j$ 移动到琴键 $k^i$ 上的开销，通过欧式距离计算；$w_t(k^i, f^j)$ 为布尔权重

In our case, it enforces that each key in $K_{t}$ will be pressed by only one finger in $F$ , and each finger presses at most one key. 
>  我们约束 $K_t$ 中的每个键最多只能由 $F$ 中的一个手指按下

The constrained optimization problem in Eq. (1) is an optimal transport problem. Intuitively, it tries to find the best ”transport” strategy such that the overall cost of moving (a subset of) fingers $F$ to keys $K_{t}$ is minimized. We solve this optimization problem with a modified Jonker-Volgenant algorithm [37] from SciPy [38] and use the optimal combinations $(i^{*},j^{*})$ as the fingering for the agent. 
>  Eq 1 的约束优化问题是一个最优传输问题，它尝试找到最优的 “传输“ 策略 (手指放置策略)，使得移动 (部分) 手指 $F$ 到键 $K_t$ 的距离最小化
>  我们通过修改的 Jonker-Volgenant 算法求解该优化问题，并使用求解得到的最优组合 $(i^*, j^*)$ 作为 agent 的指法

The fingering is calculated on the fly based on the hands’ state, so during the RL training, the fingering adjusts according to the robot hands’ state. 
>  下一时刻的指法是根据当前手的状态实时计算的，故在 RL 训练过程中，指法会基于机器人手的状态而调整

We define a reward $r_{t}^{\mathrm{OT}}$ to encourage the agent to move the fingers close to the keys $K_{t}$ . which is defined as: 

$$
r_{t}^{\mathrm{OT}}=\left\{\begin{array}{l l}{\exp\bigl(c\cdot(d_{t}^{\mathrm{OT}}-0.01)^{2}\bigr)}&{\mathrm{if~}d_{t}^{\mathrm{OT}}\geq0.01,}\\ {1.0}&{\mathrm{if~}d_{t}^{\mathrm{OT}}<0.01.}\end{array}\right.
$$ 
>  我们定义奖励 $r_t^{OT}$ 以鼓励 agent 将手指移动到目标按键 $K_t$ 附近，其定义如上
>  如果最小累积移动距离 $d_t^{OT}$ 小于 $0.01$ ，奖励恒定为 $1$，否则，奖励值随着 $d_t^{OT}$ 增大而递减

$c$ is a constant scale value as used in Tassa et al. [39] and $d_{t}^{\mathrm{OT}}$ is the distance between fingers and target keys obtained by solving Eq. (1). $r_{t}^{\mathrm{OT}}$ increases exponentially as $d_{t}^{\mathrm{OT}}$ decreases and is set as 1 once $d_{t}^{\mathrm{OT}}$ is smaller than a pre-defined threshold (0.01). 
>  其中 $c$ 是一个常数标量值 (小于零)，$d_t^{OT}$ 是通过求解 Eq 1 得到的手指和目标按键指尖的距离
>  $r_t^{OT}$ 随着 $d_t^{OT}$ 减少而指数增长，当 $d_t^{OT}$ 小于预定的阈值，奖励就定为 1

The overall reward function is defined as: 

$$
{r}_{t}={r}_{t}^{\mathrm{{OT}}}+{r}_{t}^{\mathrm{{Press}}}+{r}_{t}^{\mathrm{{Sustain}}}+{\alpha}_{1}\cdot{r}_{t}^{\mathrm{{Collision}}}+{\alpha}_{2}\cdot{r}_{t}^{\mathrm{{Energy}}}
$$

$r^{\mathrm{Press}}$ and $r_{t}^{\mathrm{Sustain}}$ represent the reward for correctly pressing the target keys and the sustain pedal. $r_{t}^{\mathrm{Collision}}$ encourages the agent to avoid collision between forearms and $r_{t}^{\mathrm{Energy}}$ prefers energy-saving behaviors. $\alpha_{1}$ and $\alpha_{2}$ are coefficient terms, and $\alpha_{1}=0.5$ and $\alpha_{2}=5\cdot10^{-3}$ are adopted. 

>  总的奖励函数定义如上，它是各个奖励项的求和
>  其中 $r^{Press}, r^{Sustain}$ 表示正确按下目标按键和延音踏板的奖励，$r_t^{Collison}$ 鼓励 agent 避免前臂之间的碰撞，$r_t^{Enerty}$ 偏好更节约能量的行为
>  $\alpha_1, \alpha_2$ 为系数项

Our method is compatible with any RL methods, and we use DroQ [40] in our paper. 
>  我们的方法兼容各种 RL 方法，我们使用 DroQ

## 4.2 Analysis of Specialist RL Agents 
The performance of the specialist RL agents decides the quality of our dataset. In this section, we investigate the performance of our specialist RL agents. We are interested in i) how the proposed OT-based finger placement helps learning, ii) how the fingering discovered by the agent itself compares to human fingering labels, and iii) how our method transfers to other embodiments. 
>  专用 agent 的性能决定了我们数据集的质量
>  我们研究我们的专用 agent 的表现，具体在以下三个方面
>  i) 我们提出的基于最优传输的手指放置如何帮助学习
>  ii) agent 自身发现的手指拜访和人类标记的指法相比如何
>  iii) 我们的方法在其他形态上的迁移能力如何

![[pics/RP1M-Fig2.png]]

**Results** In Fig. 2, we compare our method with RoboPianist-RL both with and without human fingering. We use the same DroQ algorithm with the same hyperparameters for all experiments. 
>  Results
>  Fig2 中，我们将我们的方法和 RoboPianist-RL 分别在有无人类指法的情况下进行了比较，所有试验都使用 DroQ 算法，以及相同的超参数

RoboPianist-RL includes human fingering in its inputs, and the fingering information is also used in the reward function to force the agent to follow this fingering. Our method, marked as $O T$ , removes the fingering from the observation space and uses OT-based finger placement to guide the agent to discover its own fingering. We also include a baseline, called No Fingering, that removes the fingering entirely. 
>  RoboPianist-RL 的输入中有人类指法，且该指法信息也用在了奖励函数中以迫使 agent 遵循该指法
>  我们的方法将指法从 observation space 中移除 (输入中没有指法信息)，并使用基于最优传输的指法放置来引导 agent 发现自己的指法
>  我们还引入了完全去除指法的 baseline

The first two columns of Fig. 2 show that our method without human-annotated fingering matches RoboPianst-RL’s performance on two different songs. Our method outperforms the baseline without human fingering by a large margin, showing that the proposed OT-based finger placement boosts the agent learning. 
>  Fig2 展示了我们的方法匹配了具有人类指法输入的 RoboPianist-RL 的表现，并且比没有指法的 baseline 的性能高出很多，说明了我们提出的基于最优传输的指法放置帮助了 agent 的学习

The proposed method works well even on challenging songs. We test our method on Flight of the Bumblebee and achieve 0.79 F1 score after 3M training steps. To the best of our knowledge, we are the first to play the challenging song Flight of the Bumblebee with general-purpose bimanual dexterous robot hands. 
>  即便在具有挑战性的曲子上，我们的方法也能很好地工作

![[pics/RP1M-Fig3.png]]

**Analysis of the Learned Fingering** We now compare the fingering discovered by the agent itself and the human annotations. In Fig. 3, we visualize the sample trajectory of playing French Suite No.5 Sarabande and the corresponding fingering. We found that although the agent achieves strong performance for this song (the second plot in Fig. 2), our agent discovers different fingering compared to humans. 
>  Analysis of the Learned Fingering
>  我们比较 agent 自行发现的指法和人类标记的指法
>  Fig3 可视化了指法，我们发现，尽管 agent 在这首曲子上表现了强大的性能，但我们 agent 发现的指法与人类不同

For example, for the right hand, humans mainly use the middle and ring fingers, while our agent uses the thumb and first finger. Furthermore, in some cases, human annotations are not suitable for the robot hand due to different morphologies. For example, in the second time step of Fig. 3, the human uses the first finger and ring finger. However, due to the mechanical limitation of the robot hand, it can not press keys that far apart with these two fingers, thus mimicking human fingering will miss one key. Instead, our agent discovered to use the thumb and little finger, which satisfies the hardware limitation and accurately presses the target keys. 
>  例如，对于右手，人类主要使用中指或食指，agent 则主要使用大拇指和食指
>  此外，在一些情况下，由于机器人手的形态不同，人类标记并不适合机器人手
>  例如，在 Fig3 中的第二个时间步处，人类使用了食指和无名指，但由于机器人手部的机械限制，它无法用两个手指按下如此远距离的琴键，因此模仿人类的指法会漏掉一个音符，我们的 agent 发现了使用拇指和小指的方法，这符合硬件限制，并且准确按下了琴键

**Cross Emboidments** Labs usually have different robot platforms, thus having a method that works for different embodiments is highly desirable. We test our method on a different embodiment. To simplify the experiment, we disable the little finger of the Shadow robot hand and obtain a four-finger robot hand, which has a similar morphology to Allegro [41] and LEAP Hand [42]. We evaluate the modified robot hand on the song French Suite No.5 Sarabande (first 550 time steps), where our method achieves a 0.95 F1 score, similar to the 0.96 achieved with the original robot hands. In the bottom row of Fig. 3, we visualize the learned fingering with four-finger hands. The agent discovers different fingering compared to humans and the original hands but still accurately presses active keys, meaning our method is compatible with different embodiments. 
>  Cross Emboidments
>  我们在不同的形态上测试了我们的方法，为了简化试验，我们禁用了 Shadow robot hand 的小指，得到了一个四指机器人手
>  我们的方法下，修改后的机器人手达到的表现与原始机器人手的表现相近，agent 发现的指法与人类不同，但仍可以准确按下 active keys，这意味着我们的方法兼容不同的形态

## 4.3 RP1M Dataset 
To facilitate the research on dexterous robot hands, we collect and release a large-scale motion dataset for piano playing. Our dataset includes ${\sim}1{M}$ expert trajectories covering ${\sim}2{k}$ musical pieces. For each musical piece, we train an individual DroQ agent with the method introduced in Section 4.1 for 8 million environment steps and collect 500 expert trajectories with the trained agent. We chunk each sheet music every 550 time steps, corresponding to 27.5 seconds, so that each run has the same episode length. The sheet music used for training is from the PIG dataset [6] and a subset (1788 pieces) of the GiantMIDI-Piano dataset [7]. 
>  我们收集并发布了一个大规模的钢琴演奏动作数据集，数据集包括大约 100 万条轨迹，覆盖了大约 2000 首曲子
>  我们为每一首曲子单独训练一个 DroQ agent (在环境中运行 800 万步)，收集了 500 条轨迹
>  我们将每份乐谱每隔 550 时间步分割一次 (对应 27.5 秒)，这使得每次运行有相同的回合长度，用于训练的乐谱来自于 PIG 数据集和 GiantMIDI-Piano 数据集的一个子集

![[pics/RP1M-Fig4.png]]

In Fig. 4, we show the statistics of our collected motion dataset. The top plot shows the histogram of the pressed keys. We found that keys close to the center are more frequently pressed than keys at the corner. Also, white keys, taking $65.7\%$ , are more likely to be pressed than black keys. In the bottom left plot, 
>  Fig4 展示了我们收集到的动作数据的统计数据
>  顶部的图表是被按下琴键的分布直方图，我们发现靠近中心的键相较于靠近角落的键更经常被按下，此外，白色按键相较于黑色按键更经常被按下

we show the distribution of the number of active keys over all time steps. It roughly follows a Gaussian distribution, and $90.70\%$ musical pieces in our dataset include 1000-4000 active keys. 
>  左下角的图展示了所有时间步下活跃键数量的分布，它大致遵循高斯分布，并且数据集中 90.70% 的乐曲都包含了 1000-4000 个活跃的键

We also include the distribution of F1 scores of trained agents used for collecting data. We found most agents $(79.00\%)$ achieve F1 scores larger than 0.75, and $99.89\%$ of the agents’ F1 scores are larger than 0.5. The distribution of F1 scores reflects the quality of the collected dataset. We empirically found agents with F1 score $\geq0.75$ are capable of playing sheet music reasonably well with only minor errors. Agents with $\leq0.5$ F1 scores usually have notable errors due to the difficulty of songs or the mechanical limitations of the Shadow robot hand. We also include the F1 scores for each piece in our dataset so users can filter the dataset according to their needs. 
>  右下角的图展示了所有用于收集数据的 agents 的 F1 score 的分布
>  大多数 agent 的 F1 score 都大于 0.75，并且几乎全部 agents 的 F1 score 都大于 0.5
>  F1 score 的分布反映了所收集的数据的质量，我们经验性地发现 F1 score 高于 0.75 的 agent 能够以较少的错误合理地演奏乐谱，F1 score 低于 0.5 地 agent 通常会出现较为明显的错误
>  我们还在数据集中为每首曲子都提供了 F1 score，便于用户根据需要筛选数据集

# 5 Benchmarking Results 
The analysis in the previous section highlighted the diversity of highly dynamic piano-playing motions in the RP1M dataset. 

In this section, we assess the multi-task imitation learning performance of several widely used methods on our benchmark. To be specific, the objective is to train a single multi-task policy capable of playing various music pieces on the piano. We train the policy on a portion of the RP1M dataset and evaluate its in-distribution performance (F1 scores on songs included in the training data) and its generalization ability (F1 scores on songs not present in the training data). 
>  我们评估几种广泛使用的多任务模仿学习方法在我们的基准测试上的表现
>  具体地说，目标是训练一个多任务策略，能够在钢琴上演奏各种音乐作品
>  我们在 RP1M 数据集上的一部分训练该策略，并评估其分布内的性能 (在训练数据中的曲子上的 F1 score) 和其泛化能力 (不在训练数据中的曲子上的 F1 score)

**Baselines** We evaluated Behavior Cloning (BC) [43], Behavior Transformer (BeT) [44], Diffusion Policy [5] with U-Net (DP-U) [45] and with Transformer (DP-T) [46]. BC directly learns a policy by using supervised learning on observation-action pairs from expert demonstrations. BeT clusters continuous actions into discrete bins using ${k}$ -means, allowing it to model high-dimensional, continuous, multimodal action distributions as categorical distributions [44]. Diffusion Policy learns to model the action distribution by inverting a process that gradually adds noise to a sampled action sequence. We evaluated both the CNN-based (U-Net) Diffusion Policy (DP-U) and the Transformer-based Diffusion Policy (DP-T) with DDPM [47]. We use the same code and hyperparameters as Chi et al. [5]. Detailed descriptions of the baselines as well as hyperparameters are given in Appendix C.1. 
>  Baselines
>  我们评估了 BC, BeT, Diffusion Policy, DP-U, DP-T
>  BC 使用专家演示中的 observation-action ，通过有监督学习直接学习策略
>  BeT 使用 k-means 聚类将连续的动作聚类为离散的类别，以将高维、连续、多模态的动作分布建模为类别分布
>  Diffusion Policy 通过反转一个逐渐向采样的动作序列添加噪声的过程来学习建模动作分布，我们评估了基于 CNN (U-Net) 的 Diffusion Policy (DP-U) 和基于 Transformer 的 Diffusion Policy (DP-T)，并使用 DDPM，代码和超参数和 [5] 相同

**Experiment Setup** We train the policies on subsets of the RP1M dataset with different sizes: 12, 25, 50, 100, 150. We then evaluate the trained policies on both i) 12 in-distribution songs: music pieces that overlap with the training sets, and ii) 20 out-of-distribution (OOD) songs: music pieces that do not overlap with the training songs. The selected songs are very challenging and contain diverse motions and long horizons. In the experiment, we report zero-shot evaluation results without fine-tuning. We report the average F1 scores of each group of music pieces for policies trained with each baseline method. We list the selected songs for evaluation in Appendix C.2. 
>  Experiment Setup
>  我们在 RP1M 数据集的不同大小的子集上训练策略，然后在
>  i) 12 首分布内曲子 ii) 20 首分布外曲子
>  上评估策略的性能
>  所选出的曲子非常具有挑战性，包含了多样的动作和长的时间跨度
>  在实验中，我们报告的是未经微调的零样本结果，对于每个 baseline 训练的策略，我们报告每组曲子的评估 F1 score

![[pics/RP1M-Table2.png]]

**Discussion** We present the benchmarking performance of multi-task agents in Table 2. For the in-distribution evaluation, compared to F1 scores obtained with our RL specialist agents in Fig. 4, we notice a performance gap across all baselines. This gap widens as the data size increases. When trained on a smaller dataset with 12 training songs, DP-U performs comparably to BC-MLP and slightly outperforms DP-T, while BeT experiences a significant performance drop. 
>  Discussion
>  多任务 agent 的基准性能见 Table2
>  对于分布内评估，所有的 baseline 和专家 agent 都存在性能差距，随着数据集增大，差距会减小
>  训练在包含 12 首曲子的小数据集上时，DP-U 和 BC-MPL 表现相近，并略微优于 DP-T, BeT 的性能则显著下降

This decline may be attributed to hyperparameter choices, such as the number of action bins. Although we used the same number of action bins as the official implementation, the complexity of our tasks suggests that this configuration may be inadequate, and increasing the number of bins could improve performance. 
>  这一下降可能是由于超参数的选择，例如动作区间的数量

As the dataset size increases, we observe that Diffusion Policy outperforms the other baselines. DP-U and DP-T show performance drops of $15.77\%$ and $10.92\%$ , respectively, while BC-MLP suffers a more significant decline of $52.74\%$ . Similar performance gaps have been noted in previous work [4] and concurrent research [26] suggests a hierarchical policy structure, although it still lags behind RL specialists. This highlights the need for future research to address the performance gap between RL specialists and multi-task agents. 
>  随着数据集增大，我们观察到 Diffusion Policy 开始优于其他策略
>  但是专家 agent 的表现还是显著优于多任务 agent 的表现

In the zero-shot out-of-distribution evaluation, we find that performance improves for all evaluated baselines as the training data size increases. Specifically, the F1 scores for DP-U and DP-T rise from 0.181 to 0.256 and from 0.186 to 0.316, respectively, when the number of training songs is increased from 12 to 150. This suggests that larger datasets enhance the generalization capabilities of multi-task agents. We hope that releasing our large-scale RP1M dataset will contribute to the development of robust generalist piano-playing agents within the research community. 
>  我们发现所有评估的 baseline 都随着训练数据量增加而性能增加，这说明了更大的数据集可以提高多任务 agent 的泛化能力

# 6 Limitations & Conclusion 
**Limitations** Our paper has limitations in several aspects. Firstly, although our method lifts the requirement of human-annotated fingering, enabling RL training on diverse songs, our method still fails to achieve strong performance on challenging songs due to fast rhythms and mechanical limitations of the robot hands. Improving the RL method and hardware design could help address this. Secondly, the evaluation metric, F1 score, may not adequately capture musical performance and the position-based controller missing the target velocity would hinder the performance. Thirdly, our dataset includes only proprioceptive observations, whereas humans play piano using multi-modal inputs like vision, touch, and hearing; incorporating these could enhance the agent’s capabilities.
>  Limitations
>  其一，虽然我们的方法消除了对人类标记的指法的需求，从而能够在多样的曲目上进行 RL 训练，但由于机器人手的快节奏和机械限制，仍然在面对具有挑战性的曲子上无法取到高性能，改进 RL 方法和硬件设计可能有助于解决这个问题
>  其二，评估指标 F1 score 可能无法完全反映音乐表现，而基于位置的控制器未能达到目标速度可能会阻碍表现
>  其三，我们的数据集进包含本体感觉观测，而人类演奏时，包含了多模态输入，包括视觉、触觉、听觉，将这些输入结合可能会增强 agent 的能力

 Furthermore, there are several challenges to deploying the learned agent on a real-world robot. This includes the challenges of obtaining the state of the piano and the hands (e.g., tracking the precise fingertip positions), optimizing a precise position controller at high speed as well as the sim-to-real gap for the highly dynamic piano-playing task, etc. 
>  此外，部署到真实世界的机器人还存在许多挑战，包括了获取钢琴和手的状态的挑战 (例如，精确跟踪指尖位置)、优化高速的控制器的精确位置的挑战、以及 sim-to-real 的差距等等

Lastly, although we demonstrate better zero-shot generalization performance than RoboPianist-MT [4], there is still a gap between our best multi-task agent and RL specialists, which requires future investigation. 
>  最后，多任务 agent 与专家 agent 的能力仍然存在差距

**Conclusion** In this paper, we propose a large-scale motion dataset named RP1M for piano playing with bimanual dexterous robot hands. RP1M includes 1 million expert trajectories for playing $2\mathrm{k}$ musical pieces. To collect such a diverse dataset for piano playing, we lift the need for human-annotated fingering in the previous method by introducing a novel automatic fingering annotation approach based on optimal transport. On single songs, our method matches the baselines with human-annotated fingering and can be adopted across different embodiments. Furthermore, we benchmark various imitation learning approaches for multi-song playing. 
>  Conclusion
>  我们提出了一个名为 RP1M 的大规模运动数据集，用于带有双手机器人钢琴演奏
>  RP1M 包含演奏 2k 首音乐作品的 100 万条专家轨迹
>  为了收集数据集，我们通过引入一种基于最优传输的自动指法标注方法，消除了先前方法对人工标注指法的需求
>  在单首歌曲上，我们的方法与使用人工标注指法的基线相匹配，并且可以适用于不同的机器人形态
>  此外，我们评估了多种用于多首歌曲演奏的模仿学习方法

We report promising results in motion synthesis for novel music pieces when increasing the data size and identify the gap to achieving human-level piano-playing ability. We believe the RP1M dataset, with its scale and quality, forms a solid step towards empowering robots with human-level dexterity. 

# Appendix 
# A More Related Work 
**Dexterous Robot Hands** The research of dexterous robot hands aims to replicate the dexterity of human hands with robots. Many previous works [48, 49, 50, 51, 52, 53, 54, 55] use planning to compute a trajectory followed by a controller, thus require an accurate model of the robot hand. Closed-loop approaches have been developed by incorporating sensor feedback [56]. These methods also require an accurate model of the robot hand, which can be difficult to obtain in practice, especially considering the large number of active contacts between the hand and objects. 
>  Dexterous Robot Hands
>  许多先前的工作使用规划来计算轨迹，然后由控制器执行，故需要对机器人手的准确模型
>  [56] 引入了传感器反馈，开发了闭环方法
>  这些方法都需要对机器人手的精确建模，这在实际中难以获得，尤其是在考虑手和物体之间存在大量主动接触的情况下

Due to the difficulty of actually modeling the dynamics of the dexterous robot hand, recent methods resort to learning-based approaches, especially RL, which has achieved huge success in both robotics [57, 58, 1] and computer graphics [59]. To ease the training of dexterous robot hands with a large number of degrees of freedom (DoFs), demonstrations are commonly used [60, 61, 62, 63, 64]. Due to the advance of both RL algorithms and simulation, recent work shows impressive results on dexterous hand manipulation tasks without human demonstrations. Furthermore, the policy trained in the simulator can further be deployed on real dexterous robot hands via sim-to-real transfer [65, 66, 67, 30, 68, 69, 70]. 
>  近期方法转向基于学习的方法，尤其是 RL
>  为了简化具有大量自由度的机器人手训练过程，许多方法使用了演示
>  此外，通过 sim-to-real 迁移方法，许多模拟器上训练出的策略可以部署到实际机器人手上

**Generalist Agents** RL methods usually perform well on single tasks, however, as human beings, we can perform multiple tasks. Generalist agents are proposed to master a diverse set of tasks with a single agent [71, 57, 72, 73]. These methods typically resort to scalable models and large datasets [72, 74, 75, 76, 77]. 
>  通用 agent 旨在让一个 agent 掌握多样化的任务集，这些方法通常依赖于可拓展的模型和大型数据集

Recently, diffusion models have achieved many state-of-the-art results across image, video, and 3D content generation [78, 79, 80, 81, 82] In the context of robotics, diffusion models have been used as policy networks for imitation learning in both manipulation [5, 75, 83, 84] and locomotion tasks [85]. The same technique has also been investigated in multi-task learning [75, 84]. We investigate the application of diffusion policy in high-dimensional control tasks, that is, playing piano with bimanual dexterous robot hands. 
>  Diffusion 模型被用于模仿学习中的策略网络，应用于操作任务和运动任务，以及多任务学习
>  我们将 diffusion policy 应用于高维控制任务

# B RP1M Dataset Collection Details 
## B.1 Reward Formulation 
In Eq. (3) , we give the overall reward function used in our paper. We now give details of each term.

$r_{t}^{\mathrm{Press}}$ indicates whether the active keys are correctly pressed and inactive keys are not pressed. We use the same implementation as [4], given as: $\begin{array}{r}{r_{t}^{\mathrm{Press}}=0.5\cdot(\frac{1}{K}\sum_{t}^{K}g(||k_{s}^{i}-1||_{2}))+0.5\cdot(1-\mathbf 1_{\mathrm{fp}})}\end{array}$ . 

${ K}$ is the number of active keys, $k_{t}^{i}$ is the normalized key states with range [0, 1], where 0 means the $i$ -th key is not pressed and 1 means the key is pressed. $g$ is tolerance from Tassa et al. [39], which is similar to the one used in Equation (2). $\mathbf{1}_{\mathrm{fp}}$ indicates whether the inactive keys are pressed, which encourages the agent to avoid pressing keys that should not be pressed.

$r_{t}^{\mathrm{Sustain}}$ encourages the agent to press the pseudo sustain pedal at the right time, given as $r_{t}^{\mathrm{Sustain}}=g(s_{t}-s_{t}^{\mathrm{target}})$ . $s_{t}$ and $s_{t}^{\mathrm{target}}$ are the state of current and target sustain pedal respectively.

$r_{t}^{\mathrm{Collision}}$ penalizes the agent from collision, defined as $r_{t}^{\mathrm{Collision}}=1-\mathbf 1_{\mathrm{collision}}$ , where $\mathbf{1}_{\mathrm{collision}}$ is 1 if collision happens and 0 otherwise. 

$r_{t}^{\mathrm{Energy}}$ prioritizes energy-saving behavior. It is defined as   $r_t^{\text{Energy}} = |\tau_{\text{joints}}|^T|\mathbf v_{\text{joints}}|$ .  $\tau_{\text{joints}}$ and $\mathbf v_{\text{joints}}$ are joint torques and joint velocities respectively. 


## B.2 Training Details 
**Observation Space** Our 1144-dimensional observation space includes the proprioceptive state of dexterous robot hands and the piano as well as L-step goal states obtained from the MIDI file. In our case, we include the current goal and 10-step future goals in the observation space $({L}=11)$ . 

At each time step, an 89-dimensional binary vector is used to represent the goal, where 88 dimensions are for key states and the last dimension is for the sustain pedal. The dimension of each component in the observation space is given in Table 3. 

Table 3: Observation space. 

<html><body><center><table><tr><td>Observations</td><td>Dim</td></tr><tr><td>Piano goal state</td><td>L·88</td></tr><tr><td>Sustain goal state</td><td>L·1</td></tr><tr><td>Piano key joints</td><td>88</td></tr><tr><td>Piano sustain state</td><td>1</td></tr><tr><td>Fingertip position</td><td>3.10</td></tr><tr><td>Hand state</td><td>46</td></tr></table></center></body></html> 

**Training Algorithm & Hyperparameters** Although our proposed method is compatible with any reinforcement learning method, we choose the DroQ [40] as Zakka et al. [4] for fair comparison. DroQ is a model-free RL method, which uses Dropout and Layer normalization in the Q function to improve sample efficiency. We list the main hyperparameters used in our RL training in Table 4. 

Table 4: Hyperparameters used in our RL agent. 

<html><body><center><table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Training steps</td><td>8M</td></tr><tr><td>Episode length</td><td>550</td></tr><tr><td>Action repeat</td><td>1</td></tr><tr><td>Warm-up steps</td><td>5k</td></tr><tr><td>Buffer size</td><td>1M</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Update interval</td><td>2</td></tr><tr><td>Piano environment</td><td></td></tr><tr><td>Lookahead steps</td><td>10</td></tr><tr><td>Gravity compensation</td><td>True</td></tr><tr><td>Control timestep</td><td>0.05</td></tr><tr><td>Stretch factor</td><td>1.25</td></tr><tr><td>Trim slience</td><td>True</td></tr><tr><td>Agent</td><td></td></tr><tr><td>MLPs</td><td>[256, 256, 256]</td></tr><tr><td>Num. Q</td><td>2</td></tr><tr><td>Activation</td><td>GeLU</td></tr><tr><td>Dropout Rate</td><td>0.01</td></tr><tr><td>EMA momentum</td><td>0.05</td></tr><tr><td>Discount factor</td><td>0.88</td></tr><tr><td>Learnable temperature</td><td>True</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>3e-4</td></tr><tr><td>β1</td><td>0.9</td></tr><tr><td>β2</td><td>0.999</td></tr><tr><td>eps</td><td>1e-8</td></tr></table></center></body></html> 

## B.3 Computational Resources 
We train our RL agents on the LUMI cluster equipped with AMD MI250X GPUs, 64 cores AMD EPYC “Trento” CPUs, and 64 GBs DDR4 memory. Each agent takes 21 hours to train. The overall data collection cost is roughly 21 hours \* 2089 agents =43, 869 GPU hours. 

## B.4 MuJoCo XLA Implementation 
To speed up training, we re-implement the RoboPianist environment with MuJoCo XLA (MJX), which supports simulation in parallel with GPUs. MJX has a slow performance with complex scenes with many contacts. To improve the simulation performance, we made the following modifications: 
>  为了加快训练速度，我们使用 MuJoCo XLA (MJX) 重新实现了 RoboPianist 环境，该环境支持 GPU 并行模拟
>  但是 MJX 在处理包含大量接触点的复杂场景时性能较慢，为了提高模拟型嗯那个，我们进行了以下修改:

- We disable most of the contacts but only keep the contacts between fingers and piano keys as well as the contact between forearms. 
- Primitive contact types are used whenever possible. 
- The dimensionality of the contact space is set to 3. 
- The maximal contact points are set to 20. 
- We use Newton solver with iterations $=2$ and ls\_iterations $=6$ . 

>  - 我们仅用了大多数接触，进保留了手指和琴键以及前臂之间的接触
>  - 尽可能使用基本接触类型
>  - 接触空间维度设置为 3
>  - 最大接触点数设置为 20
>  - 使用 2 次迭代的牛顿求解器和 6 次迭代的线性搜索

After the above modifications, with 1024 parallel environments, the total steps per second is 159,376. 

![[pics/RP1M-Fig5.png]]

We use PPO implementation implemented with Jax to fully utilize the paralleled simulation. The PPO with MJX implementation is much faster than the DroQ implementation, which only takes 2 hours and 7 minutes for 40M environment steps on the Twinkle Twinkle Little Star song while as a comparison, DroQ needs roughly 21 hours for 8M environment steps. However, the PPO implementation fails to achieve a comparable F1 score as the DroQ implementation as shown in Fig. 5. Therefore, we use the DroQ implement with the CPU version of the RoboPianist environment. 
>  我们使用基于 Jax 实现的 PPO 算法，以充分利用并行仿真
>  采用 MJX 实现的 PPO 比 DroQ 实现快很多，但是 PPO 实现未能达到与 DroQ 相当的 F1 score，故我们最终选择 DroQ 在 RoboPianist 环境的 CPU 版本上训练

# C Multitask Benchmarking Details 
A single multi-task policy capable of playing various songs is highly desirable. However, playing different music pieces on the piano results in diverse behaviors, creating a complex action distribution, particularly for dexterous robot hands with a large number of degrees of freedom (DoFs).

This section introduces the baseline methods we have compared and the hyperparameters we have used. We also talk about the details of our multitask training and evaluation. 

## C.1 Baselines and Hyperparameters 
### C.1.1 Behavior Cloning 
Behavior Cloning (BC) [43] directly learns a policy using supervised learning on observation-action pairs from expert demonstrations, one of the simplest methods to acquire robotic skills. Due to its straightforward approach and proven efficacy, BC is popular across multiple fields. The method employs a Multi-Layer Perceptron (MLP) as the policy network. Given expert trajectories, the policy network learns to replicate expert behavior by minimizing the Mean Squared Error (MSE) between predicted and actual expert actions. 
>  Behavior Cloning 使用 MLP 作为策略网络，通过最小化动作预测的 MSE，以学习复制专家策略的行为

Despite its advantages, BC tends to perform poorly in generalizing to unseen states from the expert demonstrations. The MLP we used features three hidden layers, each with 512 units, followed by Layer Normalization and an Exponential Linear Unit (ELU) activation function to stabilize training and introduce non-linearity. 
>  BC 在推广到专家演示没有的状态时往往效果较差

Table 5: Hypermeters used in BC 

<html><body><center><table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch Size</td><td>1024</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning Rate</td><td>1e-4</td></tr><tr><td>Observation Horizon</td><td>1</td></tr><tr><td>Prediction Horizon</td><td>1</td></tr><tr><td>Action Horizon</td><td>1</td></tr></table></center></body></html> 

### C.1.2 BeT 
Behavior Transformers (BeT) [44] uses a transformer-decoder based backbone with a discrete action mode predictor coupled with a continuous action offset corrector to model continuous actions sequences. It clusters continuous actions into discrete bins using ${k}$ -means to model high-dimensional, continuous multi-modal action distributions as categorical distributions without learning complicated generative models. We adopted the implementation and hyperparameters from the Diffusion Policy codebase [5]. 
>  Behavior Transformers 将连续动作聚类到离散的区间中，将建模高维、连续多模态动作分布将末尾类别分布，避免学习复杂的生成模型

Table 6: Hyperparamerters used in BeT 

<html><body><center><table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch Size</td><td>512</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-4</td></tr><tr><td>Num of bins</td><td>64</td></tr><tr><td>MinGPT n_layer</td><td>8</td></tr><tr><td>MinGPT n_head</td><td>8</td></tr><tr><td>MinGPT n_embd</td><td>120</td></tr><tr><td>Observation Horizon</td><td>1</td></tr><tr><td>Prediction Horizon</td><td>1</td></tr><tr><td>Action Horizon</td><td>1</td></tr></table></center></body></html> 

### C.1.3 Diffusion Policy 
Diffusion models have achieved many state-of-the-art results across image, video, and 3D content generation [78, 79, 80, 81, 82]. In the context of robotics, diffusion models have been used as policy networks for imitation learning in both manipulation [5, 75, 83, 84] and locomotion tasks [85], showing remarkable performance across various robotic tasks. 
>  扩散模型已经被用于操作和运动任务的模仿学习中的策略网络

Diffusion Policy [5] proposed to learn an imitation learning policy with a conditional diffusion model. It models the action distribution by inverting a process that gradually adds noise to a sampled action sequence, conditioning on a state and a sampled noise vector. We evaluated both the U-Net-based Diffusion Policy (DP-U) and the transformer-based Diffusion Policy (DP-T). We build our diffusion policy training pipeline based on the original Diffusion Policy [5] codebase, which provides high-quality implementations. 
>  [5] 提出使用条件扩散模型来学习模仿学习策略

Table 7: Hyperparameters used in DP-U 

<html><body><center><table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch Size</td><td>1024</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-4</td></tr><tr><td>Weight Decay</td><td>1e-6</td></tr><tr><td>Diffusion Method</td><td>DDPM</td></tr><tr><td>Number of Diffusion Iterations</td><td>100</td></tr><tr><td>EMAPower</td><td>0.75</td></tr><tr><td>U-Net Hidden Layer Sizes</td><td>[256,512,1024]</td></tr><tr><td>Diffusion Step Embedding Dim.</td><td>256</td></tr><tr><td>Observation Horizon</td><td>1</td></tr><tr><td>Prediction Horizon</td><td>4</td></tr><tr><td>Action Horizon</td><td>4</td></tr></table></center></body></html> 

Table 8: Hyperparameters used in DP-T 

<html><body><center><table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch Size</td><td>1024</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-3</td></tr><tr><td>Weight Decay</td><td>1e-4</td></tr><tr><td>Diffusion Method</td><td>DDPM</td></tr><tr><td>EMA Power</td><td>0.75</td></tr><tr><td>n.layer</td><td>8</td></tr><tr><td>n_head</td><td>4</td></tr><tr><td>n_emb</td><td>156</td></tr><tr><td> p_drop_emb</td><td>0.0</td></tr><tr><td>p_drop_attn</td><td>0.3</td></tr><tr><td>Observation Horizon</td><td>1</td></tr><tr><td>Prediction Horizon</td><td>4</td></tr><tr><td>Action Horizon</td><td>4</td></tr></table></center></body></html> 

## C.2 Training and Evaluation 
We train the policies with 5 different sizes of expert data: 12, 25, 50, 100, and 150 songs, respectively. 

Subsequently, we assess the trained policies using two distinct categories of musical pieces. The first category, in-distribution songs, includes pieces that are part of the training datasets. Evaluating with in-distribution songs tests the multitasking abilities of the policies and checks if a policy can accurately recall the songs on which it was trained. 

The second group of songs for evaluation are out-of-distribution songs: those music pieces do not overlap with the training songs. The selected songs contain diverse motions and long horizons, making them challenging to play. This out-of-distribution evaluation measures the zero-shot generalization capabilities of the policies. 

Analogous to an experienced human pianist who can play new pieces at first sight, we aim to determine if it is feasible to develop a generalist agent capable of playing the piano under various conditions. 

Additionally, our framework is designed with flexibility in mind, allowing users to select songs not included in our dataset for either training data collection or evaluation. Furthermore, users have the option to assess their policies on specific segments of a song rather than the entire piece. 