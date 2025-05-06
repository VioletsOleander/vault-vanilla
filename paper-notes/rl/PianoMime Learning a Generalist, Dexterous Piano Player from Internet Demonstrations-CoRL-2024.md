# Abstract 
In this paper, we present PianoMime, a framework for training a piano-playing agent using Internet demonstrations. The Internet is a promising source of large-scale demonstrations for training our robot agents. In particular, in the case of piano playing, YouTube is full of videos of professional pianists playing a wide variety of songs. In our work, we leverage these demonstrations to train a generalist piano-playing agent capable of playing any song. 
>  PianoMime 是使用互联网演示来训练钢琴演奏 agent 的框架
>  互联网是训练机器人 agent 的大规模演示来源，例如 YouTube 有大量的专业钢琴家演奏各种曲子的视频
>  我们利用这些演示视频，训练了一个通用的钢琴演奏 agent，使其能够弹奏任何曲子

Our framework is divided into three parts: a data preparation phase to extract the informative features from the YouTube videos, a policy learning phase to train song-specific expert policies from the demonstrations, and a policy distillation phase to distill the policies into a single generalist agent. 
>  我们的框架分为三个部分: 
>  数据准备阶段用于从 YouTube 视频中提取有用的特征; 策略学习阶段用于训练特定曲子的专家策略; 策略蒸馏阶段将多个专家策略蒸馏为单个通用 agent

We explore different policy designs for representing the agent and evaluate the influence of the amount of training data on the agent’s ability to generalize to novel songs not present in the dataset. 

We show that we are able to learn a policy with up to $57\%$ F1 score on unseen songs. Project website: https://pianomime.github.io/ 
>  我们可以学习到在未见过的曲子上达到 57% F1 score 的策略

Keywords: Imitation Learning, Reinforcement Learning, Dexterous Manipulation, Learning from Observations 

# 1 Introduction 
The Internet is a promising source of large-scale data for training generalist robot agents. If properly exploited, it is full of demonstrations (video, text, audio) of humans solving an infinite number of tasks [1, 2, 3] that could inform our robot agents on how to behave. 

However, learning from these databases is challenging for several reasons. First, unlike teleoperation demonstrations, video data does not specify the actions that the robot is performing, which typically requires the use of reinforcement learning to induce the robot’s actions [4, 2, 5]. Second, videos typically show a human performing the task while the learned policy is applied to a robot. This often requires retargeting the human motion to the robot body [5, 6, 7]. Finally, as pointed out in [2], if we want to learn a generalist agent, we need to choose a task for which large databases are available and which allows for an unlimited variety of open-ended goals. 
>  从互联网数据学习具有挑战性
>  其一，与遥操作演示不同，视频数据不会明确指定 agent 需要执行的动作，通常需要使用 RL 来推导动作
>  其次，视频通常是人完成任务，我们需要将人类的动作重定位到机器人的身体上
>  最后，如果我们想学习一个通用代理，我们需要选择一个具有大量可用数据的任务，并且该任务允许无限多样化的开放目标

From opening doors [6] to manipulating ropes [8] or pick and place tasks [9, 10], previous work has successfully taught robot manipulation skills through observations. However, these approaches have been limited to robots with low dexterity or to a small variety of goals. 
>  之前的研究已经成功通过观察数据训练了机器人执行开门、操纵绳索、捡起或放下物体等任务
>  但这些方法都限制在低灵巧性的机器人，以及并不多样的目标

In this work, we focus on the task of learning a generalist piano player from Internet demonstrations. Piano-playing is a highly dexterous open-ended task [11]. Given two multi-fingered robot hands and a desired song, the goal of a piano-playing agent is to press the right keys, and only the right keys, at the right time. In addition, the task can be conditioned on arbitrary songs, allowing for large and high-dimensional goal conditioning. 
>  本工作专注于从互联网演示学习通用的钢琴演奏者
>  钢琴演奏是高度灵巧的开放式任务，该任务的目标是在正确的时间按下正确的琴键，且目标基于需要演奏的曲子而变化，故该任务实际上条件于是大规模且高维的目标

In addition, the Internet is full of videos of professional piano players performing a wide variety of songs. Interestingly, these pianists often record themselves from above, making it easy to observe their performances. In addition, they usually share the MIDI files of the song they are playing, making it easier to extract relevant information. 

![[pics/PianoMime-Fig1.png]]

To learn a generalist piano-playing agent from Internet data, we introduce PianoMime, a framework for training a single policy capable of playing any song (see Figure 1). In essence, the PianoMime agent is a goal-conditioned policy that generates actions in the configuration space, given the desired song to be played. At each time step, the agent receives a trajectory of keys to press as goal input. The policy then generates a trajectory of actions and executes them in chunks. 
>  PianoMime 是一个训练通用的演奏策略的训练框架
>  本质上，PianoMime agent 是一个条件于目标的策略，它在给定目标 (需要演奏的歌曲) 后，在配置空间中生成动作
>  在每个时间步，agent 会接收一串需要按下的琴键作为目标输入，然后策略生成一系列动作，并分块执行这些动作

**To train the agent**, we combine both reinforcement learning and imitation learning. We train individual song-specific expert policies using reinforcement learning in conjunction with YouTube demonstrations, and we distill all the expert policies into a single generalist behavior cloning policy. 
>  为了训练 agent，我们结合了 RL 和 IL
>  我们使用 RL 以及 YouTube 演示视频来训练每个曲子的专家策略，然后我们将所有的专家策略蒸馏为单个通用的行为克隆策略

![[pics/PianoMime-Fig2.png]]

**To represent the agent**, we perform ablations of different architectural design strategies to model the behavior cloning policy. We investigate the benefit of incorporating representation learning to enhance the geometric information of the goal input. In addition, we explore the effectiveness of a hierarchical policy that combines a high-level policy generating fingertip trajectories with a learned inverse model generating joint space actions (see Figure 2). We show that the learned agent is able to play arbitrary songs not included in the training dataset with about $56\%$ F1 score. 
>  在 agent 的表示上，我们执行了消融试验，探究用不同架构设计来建模行为克隆策略的效果
>  我们还探究了一种分层策略的有效性，它结合了一个生成指尖轨迹的高层策略和一个生成关节空间动作的策略 (通过逆模型学习)
>  结果显示，所学习的 agent 能够以约 56%的 F1 score 演奏训练数据集中未包含的任意歌曲

In summary, the main contribution of this work is a framework for training a generalist piano-playing agent using Internet demonstration data. To achieve this goal, we 
- Introduce a method for learning policies from Internet demonstrations by decoupling the human movement information from the task-related information. 
- Present a reinforcement learning approach that combines residual policy learning strategies [12, 13] with style reward-based strategies [5]. 
- Explore different policy architecture designs, introduce novel strategies to learn geometrically consistent latent features, and perform ablations on different architecture designs. 

>  本工作的贡献在于提出了使用互联网演示数据训练通用的钢琴演奏 agent 的框架，为此，我们
>  - 提出了一个从互联网演示数据学习策略的方法，该方法将人类移动信息和任务相关信息解耦
>  - 展示了一种结合了残差学习策略和基于风格奖励策略的 RL 方法
>  - 探索了不同的策略架构设计，提出了学习几何上一致的潜在特征的策略，并对不同架构进行了消融研究

Finally, we release the dataset and trained models as a benchmark for testing Internet-data-driven dexterous manipulation. 

# 2 Related Work 
**Robotic Piano Playing.** Several studies have investigated the development of robots capable of playing the piano. In [14], multi-target Inverse Kinematics (IK) and offline trajectory planning are used to position the fingers over the intended keys. In [15], a Reinforcement Learning (RL) agent is trained to control a single Allegro hand to play the piano using tactile sensor feedback. However, the piano pieces used in these studies are relatively simple. Subsequently, in [11], an RL agent is trained to control two Shadow hands to play complex piano pieces by designing a reward function that includes a fingering reward, a task reward, and an energy reward. In contrast to previous approaches, our approach exploits YouTube piano-playing videos, allowing for faster training and more accurate robot behavior. 
>  [14] 使用多目标逆运动学和离线轨迹规划来定位手指以覆盖目标琴键
>  [15] 利用触觉传感器反馈，训练 RL agent 控制单个 Allegro 手演奏钢琴
>  [11] 使涉及了指法奖励、任务奖励和能量奖励的奖励函数训练 RL agent 控制两个 Shadow hands 演奏曲子
>  我们利用演奏视频训练 agent

**Motion Retargeting and Reinforcement Learning.** Our work has similarities with motion retargeting [16], especially those works that combine motion retargeting with RL to learn control policies [17, 18, 5, 19, 6]. Given a mocap demonstration, it is common to use the demonstration either as a reward function [5, 19] or as a nominal behavior for residual policy learning [18, 6]. In our work, we extract not only the mocap information, but also task-related information (piano states), which allows the agent to balance between mimicking the demonstrations and solving the task. 
>  我们的工作与运动重定向有相似之处，尤其是哪些将运动重定向和 RL 结合以学习控制策略的工作
>  给定一个动作捕捉演示，通常会将演示用作奖励函数或者作为残差策略学习中的基准行为
>  在我们的动作中，我们不仅提取了动作捕捉信息，还提取了任务相关的信息 (钢琴状态)，使得 agent 在模仿演示和完成任务之间取得平衡

# 3 Method 
The PianoMime framework consists of three phases: data preparation, policy learning, and policy distillation. 
>  PianoMime 框架包括三个阶段: 数据准备、策略学习、策略蒸馏

In the **data preparation phase**, given the raw video demonstration, we extract the informative signals needed to train the policies. Specifically, we extract the fingertip trajectories and a MIDI file that informs us of the state of the piano at each instant. 
>  在数据准备阶段中，我们从原始的展示视频中提取用于训练策略所需的有用信号
>  具体地说，我们提取了指尖轨迹和一个 MIDI 文件，MIDI 文件存储了钢琴在每个时间点的状态

In the **policy learning phase**, we train song-specific policies via RL. This step is essential to generate the robot actions that are missing in the demonstrations. The policy is trained with two reward functions: a style reward and a task reward. The style reward aims to match the robot’s finger movements with those of the human in the demonstrations to preserve the human style, while the task reward encourages the robot to press the right keys at the right time. 
>  在策略学习阶段中，我们通过 RL 为每个曲子训练特定的策略
>  这一环节对于生成演示中没有的动作至关重要
>  策略基于两个奖励函数训练: 风格奖励和任务奖励，风格奖励旨在使机器人的手指动作和和人类的手指动作匹配，以保持人类的风格，任务奖励鼓励机器人在正确的时间按下正确的键

In the **policy distillation phase**, we train a single behavioral cloning policy to mimic all the song-specific policies. The goal of this phase is to train a single generalist policy that can play any song. We explore different policy designs and goal representation learning to improve the generalizability of the policy. 
>  在策略蒸馏阶段，我们训练一个行为克隆策略，模仿所有曲子的特定策略
>  该阶段的目标是训练一个通用的，可以演奏任何曲子的策略
>  我们探索了不同的策略设计和目标表示以提高策略的泛用性

## 3.1 Data Preparation: From raw data to human and piano state trajectories 
We generate the training dataset by web scraping. We download YouTube videos of professional piano artists playing different songs. In particular, we select YouTube channels that also upload MIDI files of the songs played. The MIDI files represent the trajectories of the piano’s state (keys pressed/unpressed) throughout the song. 
>  我们通过网络爬虫下载 YouTube 上的演奏视频来生成训练数据集
>  我们选择了有同时上传演奏曲子的 MIDI 文件的 YouTube 频道，MIDI 文件记录了钢琴在整个曲子下的状态 (按键被按下的情况)

We use the video to extract the movement of human pianists and the MIDI file to inform about the target state of the piano during the execution of the song. We choose the fingertip position as the key signal for the robot hand to mimic. While some dexterous tasks may require the use of the palm (e.g., grasping a bottle), we believe that mimicking the fingertip motion is sufficient for the piano-playing task. This also reduces the constraints on the robot, allowing it to adapt its embodiment more freely. 
>  除了下载 MIDI 文件以外，我们还从视频提取人类演奏家的动作
>  我们将指尖位置选为机器人应该模仿的关键信号，虽然某些灵巧任务可能需要使用手掌 (例如抓取瓶子)，我们认为模仿指尖动作对于钢琴演奏任务是足够的，这也减少了机器人的限制，使其可以自由调整其形态

To extract the fingertip motion from the videos, we use MediaPipe [20], an open-source framework for perception. Given a frame from the demonstration videos, MediaPipe outputs the skeleton of the hand. We find that the classic top-view recording in YouTube videos of piano playing is highly beneficial for obtaining an accurate estimate of fingertip positions. 
>  我们使用 MediaPipe (一个开源感知框架) 来提取视频中的指尖动作
>  MediaPipe 接收视频的一帧，输出手部骨架
>  我们发现，YouTube 视频上演奏视频的经典俯视视角对于获取指尖位置的准确估计非常有益

Note that since the videos are RGB, we lack the depth signal. Therefore, we predict the 3D fingertip positions based on the piano state. The detailed procedure is explained in Appendix A. 
>  因为视频是 RGB 格式，缺乏深度信号，故我们根据钢琴的状态预测 3D 的指尖位置

## 3.2 Policy Learning: Generating robot actions from observations 
In the data preparation phase, we extract two trajectories: a human fingertip trajectory $\tau_{\pmb x}$ and a piano state trajectory $\tau_{♪}$ . The human fingertip trajectory $\tau_{\pmb{x}}:(\pmb{x}_{0},\dots,\pmb{x}_{T})$ is a $T$ -step trajectory of the 3D fingertip positions of two hands $\boldsymbol{x}~\in~\mathbb{R}^{3\times10}$ (10 fingers). The piano state trajectory $\tau_{♪}:(♪_{1},\dots,♪_{T})$ is a $T$ -step trajectory of piano states $♪\in\mathbb{B}^{88}$ , represented by an 88-dimensional binary variable representing which keys should be pressed. 
>  在数据准备阶段，我们提取两条轨迹: 人类指尖轨迹 $\tau_{\pmb x}$ 和钢琴状态轨迹 $\tau_{♪}$
>  人类指尖轨迹 $\tau_{\pmb x}:(\pmb x_0, \dots, \pmb x_T)$ 是一个 $T$ 步轨迹，记录了两只手的手指尖的 3D 位置，故 $\pmb x\in \mathbb R^{3\times 10}$ (10 根手指)
>  钢琴状态轨迹 $\tau_♪ :(♪_1,\dots, ♪_T)$ 是一个 $T$ 步轨迹，其中 $♪ \in\mathbb B^{88}$，其每一维都是二元变量，表示是否应该按下对应的键

Given the ROBOPIANIST [11] environment, our goal is to learn a goal-conditioned policy $\pi_{\boldsymbol{\theta}}$ that plays the song defined by $\tau_{♪}$ while matching the fingertip movement given by $\tau_{\pmb x}$ . Note that satisfying both objectives jointly may be impossible. Perfectly tracking the fingertip trajectory $\tau_{x}$ might not lead to playing the song correctly. Although both trajectories are collected from the same source, errors in hand tracking and embodiment mismatches might lead to deviations, resulting in poor song performance. Therefore, we suggest using $\tau_{x}$ as a style guide behavior. 
>  给定 ROBOPIANIST 环境，我们的目标是学习一个目标条件策略 $\pi_{\pmb \theta}$，该策略能够演奏钢琴状态轨迹 $\tau_{♪}$ 定义的曲子，同时匹配人类指尖轨迹 $\tau_{\pmb x}$ 定义的指尖移动
>  注意，同时满足两个目标是不可能的，完美追踪指尖轨迹 $\tau_{\pmb x}$ 可能导致无法正确演奏曲子，虽然 $\tau_{\pmb x}, \tau_{♪}$ 都是从同一个曲子中收集达到，但手部追踪的误差和具身变现的不一致可能导致偏差，故完美匹配 $\tau_{\pmb x}$ 的策略的演奏效果并不好
>  因此，我们将 $\tau_{\pmb x}$ 作为一种风格引导行为

Similar to [11], we formulate the piano playing as an Markov Decision Process (MDP) with the horizon of the episode $H$ , which is the duration of the song to be played. The state observation is defined by the robot’s proprioception $\pmb s$ and the goal state $\pmb g_t$ . The goal state $\pmb g_t$ at time $t$ informs the desired piano key configurations in the future $\pmb{g}_{t}=({♪}_{t+1},\dots,{♪}_{t+L})$ , where $L$ is the lookahead horizon. As claimed in [11], to successfully learn how to play, the agent must be aware of several steps into the future to plan its actions. The action $\pmb a$ is defined as the desired configuration for both hands $\pmb q\in\mathbb{R}^{23\times2+1}$ , each with 23 joint angles and one dimension for the sustain pedal. 
>  我们将钢琴演奏构建为 MDP，其时间步数为 $H$，即要演奏曲子的长度
>  状态观测由机器人的本体感受 $\pmb s$ 和目标状态 $\pmb g_t$ 定义
>  $t$ 时刻的目标状态 $\pmb g_t$ 指示了未来一段时间内的钢琴键配置，即 $\pmb g_t = (♪_{t+1}, \dots, ♪_{t+L})$，其中 $L$ 为前瞻窗口长度
>  动作 $\pmb a$ 被定义为双手的关节以及延音踏板的配置 $\pmb q \in \mathbb R^{23\times 2 + 1}$ 的期望值

We propose to solve the reinforcement learning problem by combining residual policy learning [12, 13, 6] and style mimicking rewards [5, 19]. 
>  我们结合残差策略学习和风格模仿奖励来解决该 RL 问题

**Residual Policy Architecture.** Given the fingertip trajectory $\tau_{x}$ , we solve an IK [21] problem to obtain a trajectory of desired joint angles $\tau_{q}^{\mathrm{ik}}:(q_{0}^{\mathrm{ik}},\dots,q_{T}^{\mathrm{ik}})$ for the robot hands. 
>  残差策略架构
>  给定指尖轨迹 $\tau_{\pmb x}$，我们通过求解逆运动学 (IK) 问题获得机器人手部关节的期望轨迹 $\tau_{\pmb q}^{ik}: (\pmb q_0^{ik}, \dots, \pmb q_T^{ik})$

Then we represent the policy $\pi_{\boldsymbol{\theta}}(a|s,g_{t})=\pi_{\boldsymbol{\theta}}^{r}(a|s,g_{t})+q_{t+1}^{\mathrm{ik}}$ as a combination of a nominal behavior (given by the IK solution) and a residual policy $\pi_{\theta}^{r}$ . Given the target state at time $t$ , the nominal behavior is defined as the next desired joint angle $\pmb q_{t+1}^{\mathrm{ik}}$ . We then learn only the residual term around the nominal behavior.
>  然后，我们将策略 $\pi_{\pmb \theta}(\pmb a\mid \pmb s, \pmb g_t)$ 表示为 $\pi_{\pmb \theta}(\pmb a\mid \pmb s, \pmb g_t) = \pi_{\pmb \theta}^r(\pmb a\mid \pmb s, \pmb g_t) + \pmb q_{t+1}^{ik}$，即一个由 IK 解给出的名义行为 $\pmb q^{ik}_{t+1}$ 和一个残差策略 $\pi_{\pmb \theta}^r$ 的结合 (给定时间 $t$ 的目标状态，其名义行为定义为下一个时刻期望的关节角度向量 $\pmb q_{t+1}^{ik}$)
>  然后，我们仅学习残差策略

 In practice, we initialize the robot at $\pmb{{{{q}_{0}^{i k}}}}$ and roll both the goal state and the nominal behavior with a sliding window along $\tau_{♪}$ and $\tau_{q}^{\mathrm{ik}}$ respectively. 
>  在实践中，我们将机器人的初始化状态定为 $\pmb q_0^{ik}$，然后沿着轨迹 $\tau_♪$ 和 $\tau_{\pmb q}^{ik}$ 滑动窗口，更新目标状态和名义行为

**Style Mimicking Reward.** We also include a style-mimicking reward to preserve the human style in the trained robot actions. The reward function $r=r_{♪}+r_{\mathbf{\mathscr{x}}}$ consists of a task reward $r_♪$ and a style-mimicking reward $r_{x}$ . While the task reward $r_♪$ encourages the agent to press the correct keys, the style reward $r_{x}$ encourages the agent to move his fingertips similar to the demonstration $\tau_{x}$ . We provide further details in Appendix D. 
>  风格模仿奖励
>  我们引入风格模拟奖励，以在训练后的机器人动作中保留人类风格，故奖励函数定义为 $r = r_{♪} + r_{\pmb x}$，其中 $r_♪$ 为任务奖励，$r_{\pmb x}$ 为风格模拟奖励
>  任务奖励 $r_{\pmb x}$ 鼓励 agent 按下正确的琴键，风格模拟奖励 $r_♪$ 鼓励 agent 像演示一样移动指尖

## 3.3 Policy Distillation: Learning a generalist piano-playing agent 
In the policy learning phase, we train song-specific expert policies from which we roll out state and action trajectories $\tau_{\pmb s}:(\pmb s_{0},\ldots,\pmb s_{T})$ and $\tau_{\pmb{q}}:\left(\pmb{q}_{0},\ldots,\pmb{q}_{T}\right)$ . 
>  我们用策略学习阶段得到的策略演奏曲子，得到状态和动作轨迹 $\tau_{\pmb s}:(\pmb s_0, \dots, \pmb s_T), \tau_{\pmb q}: (\pmb q_0, \dots, \pmb q_T)$ 

Then we generate a dataset $\mathcal{D}:(\tau_{s}^{i},\tau_{q}^{i},\tau_{x}^{i},\tau_{♪}^{i})_{i=1}^{N}$ where $N$ is the number of songs learned. Given the dataset $\mathcal{D}$ , 
>  然后，我们构建数据集 $\mathcal D: (\tau_{\pmb s}^i, \tau_{\pmb q}^i, \tau_{\pmb x}^i, \tau_{♪}^i)_{i=1}^N$，其中 $N$ 是曲子数量

we apply Behavioral Cloning (BC) to learn a single generalist piano-playing agent $\pi_{\pmb{\theta}}\big(\pmb{q}_{t:t+L},\pmb{x}_{t:t+L}\vert\pmb{s}_{t},\pmb{♪}_{t:t+L}\big)$ , which outputs configuration space actions $\pmb{q}_{t:t+L}$ and fingertip movements $\pmb{x}_{t:t+L}$ conditioned on the current state $\pmb s_t$ and the future desired piano states $♪_{t:t+L}$ . 
>  给定数据集 $\mathcal D$，我们进行行为克隆，学习一个通用的演奏 agent $\pi_{\pmb \theta}(\pmb q_{t:t+L}, \pmb x_{t:t+L}\mid \pmb s_t, ♪_{t:t+L})$，该 agent 基于当前状态 $\pmb s_t$ 和未来期望的钢琴状态 $♪_{t:t+L}$ 输出配置空间的动作 $\pmb q_{t:t+L}$ 和指尖动作 $\pmb x_{t:t+L}$

We explore different strategies to represent and learn the behavioral cloning policy and improve its generalization capabilities. In particular, we explore (1) representation learning approaches to induce spatially informative features, (2) a hierarchical policy structure for sample-efficient training, and (3) expressive generative models [22, 23, 24] to capture the multimodality of the data. 
>  我们探讨了表示和学习行为克隆策略的不同方法，以提高其泛化能力
>  具体地说，我们研究了
>  (1) 表示学习方法以推导出空间信息特征
>  (2) 层次化策略结构以实现样本高效的训练
>  (3) 表达能力强的生成式模型以捕获数据的多模态性质

Also, inspired by current behavioral cloning approaches [22, 25], we train policies that output sequences of actions rather than single-step actions and execute them in chunks. 
>  此外，受到当前行为克隆方法 [22, 25] 的启发，我们让策略在训练时输出动作序列而非单步动作的策略，并将其分块执行

**Representation Learning.** We pre-train an observation encoder over the piano state $♪$ to learn spatially consistent latent features. We hypothesize that two piano states that are spatially close should lead to latent features that are close. Using these latent features as a target should lead to better generalization. 
>  表示学习
>  我们在钢琴状态 $♪$ 上与训练一个 observation 编码器，以学习空间一致的潜在特征，我们假设**两个在空间上接近的钢琴状态的潜在特征也会接近**，使用这些潜在特征作为目标应该能够实现更好的泛化能力

To obtain the observation encoder, we train an autoencoder with a reconstruction loss over a Signed Distance Field (SDF) defined on the piano state. Specifically, the encoder compresses the binary vector of the goal into a latent space, while the decoder predicts the SDF function value of a randomly sampled query point (the distance between the query point and the closest ”on” piano key). 
>  我们用基于定义在钢琴状态上的 SDF (符号距离场) 重构损失训练一个自动编码器，其中，encoder 将目标的二进制向量压缩到潜在空间，decoder 基于潜在向量预测随机采样的查询点的 SDF 函数值 (查询点和最近的 "on" 琴键之间的距离)

>  训练目标是 decoder 预测的距离能和真实的距离相近，由此在训练过程中，促进 encoder 将目标正确编码，尤其是目标的空间信息

For the BC policy, we concatenate L-timestep desired piano states and pass through the pre-trained observation encoder to obtain the latent goal representation. We provide more details in Appendix G. 
>  我们将 $L$ 时间步的期望钢琴状态拼接，传入预训练好的 encoder，获取其潜在目标表示，作为行为克隆策略的输入条件之一

**Hierarchical Policy.** We represent the piano-playing agent with a hierarchical policy. The high-level fingertip policy takes a sequence of desired future piano states $♪$ and outputs a trajectory of human fingertip positions $\pmb x$ . Then, a low-level inverse model takes the fingertip and piano state trajectories as input and outputs a trajectory of desired joint angles $\pmb q$ . 
>  分层策略
>  实际的策略表示为分层结构，高层的指尖策略接收未来钢琴状态轨迹 $♪$，输出预测的人类指尖位置 $\pmb x$ 的轨迹，低层的逆模型接收高层模型输出的指尖轨迹和钢琴状态轨迹作为输入，输出预测的关节角度 $\pmb q$ 的轨迹

![[pics/PianoMime-Fig2.png]]

On the one hand, while fingertip trajectory data is readily available from the Internet, obtaining low-level joint trajectories requires solving a computationally expensive RL problem. On the other hand, while the high-level mapping $(♪\mapsto\pmb{{x}})$ is complex and involves fingerings, the low-level mapping $(x\mapsto{\pmb q})$ ) is relatively simple, involving a task space to configuration space mapping. This decoupling allows us to train the more complex high-level mapping on large, cheap datasets, and the simpler low-level mapping on smaller, expensive datasets. We visualize the policy in Figure 2. 
>  一方面，指尖轨迹的数据可以轻松从互联网获取，而低级的关节轨迹则需要求解复杂的 RL 问题才可以获得
>  另一方面，高层的映射 $(♪ \mapsto \pmb x)$ 涉及指法，较为复杂，而底层映射 $(\pmb x\mapsto \pmb q)$ 涉及任务空间到配置空间的映射，则相对简单
>  这种解耦使得我们可以用大规模的数据训练复杂的高层映射，而使用小型的数据训练底层的数据

**Expressive Generative Models.** Considering that the human demonstration data of piano playing is highly multimodal, we explore the use of expressive generative models to better represent this multimodality. We compare the performance of different deep generative models based policies, such as Diffusion Policies [22] and Behavioral Transformer [23], as well as a deterministic policy. 
>  表现性生成模型
>  考虑到钢琴演奏的人类演示数据是多模态的，我们使用表示性生成模型来表示这种多模态特性
>  我们比较了基于不同深度生成模型的策略的表现，包括了扩散策略和行为 Transformer，以及确定策略

# 4 Experimental Results 
We divide the experimental evaluation into three parts. In the first part, we investigate the performance of our proposed framework in learning song-specific policies via RL. In the second part, we perform ablation studies on policy designs for learning a generalist piano-playing agent by distilling the previously learned policies via BC. Finally, in the third part, we study the influence of the amount of training data on the generalization capabilities. 

**Dataset and Evaluation Metrics** All experiments are performed on our collected dataset, which contains the notes and corresponding demonstration videos and fingertip trajectories of 60 piano songs from a Youtube channel, PianoX. To standardize the length of each task, each song is divided into several clips, each 30 seconds long (the dataset contains a total of 431 clips, 258K state-action pairs). 
>  数据集和评估指标
>  所有的试验均在我们收集的数据集上进行，它包含来自 YouTube 频道 PianoX 的 60 首曲子的乐谱、对应的演示视频和指尖轨迹
>  为了标准化每个任务的长度，每首歌曲被划分为若干个片段，每个片段 30s 长 (数据集一共包括 431 个片段，258K 个状态-动作对)

In addition, we select 12 unseen clips to investigate the generalization ability of the generalist policy. These clips consist of completely new songs that do not appear in the training dataset. We use the same evaluation metrics from RoboPianist [11], i.e., precision, recall, and F1 score (see Appendix B). 
>  此外，我们选择了 12 个未见过的片段来测试通用策略的泛化能力，这些片段由完全新的歌曲组成，未出现在训练集中
>  我们使用 precision, recall, F1 score 作为评估指标

We run each policy for the whole song and evaluate its performance. 

**Simulation Environment** Our experimental setup uses the ROBOPIANIST simulation environment [11], implemented in the Mujoco [26]. The agent predicts target joint angles at $20\mathrm{{Hz}}$ , and the targets are converted to torques using PD controllers running at ${500}\mathrm{Hz}$ . The notes of the songs are also discretized at a frequency of $20\mathrm{{Hz}}$ . 
>  模拟环境
>  我们使用 ROBOPIANIST 的模拟环境，agent 以 20Hz 的频率预测目标关节角度，目标关节角度会通过以 500Hz 运行的 PD 控制器被转换为扭矩
>  曲子的音符也会以 20Hz 的频率进行离散化

We use the same setup as [11] with two modifications: 1) The z-axis sliding joints attached to both forearms are enabled to allow more versatile hand movements. 2) We increase the proportional gain of the PD controller for the $\mathbf{\boldsymbol{x}}$ -axis sliding joints to allow faster horizontal movement, which we feel is essential for some fast-paced piano songs. 
>  我们的设置和 ROBOPIANIST 相同，但进行了两项修改
>  1) 启用了两个前臂的 z 轴滑动关节，以允许更灵活的手部动作
>  2) 增加了 x-轴滑动关节的 PD 控制器的比例增益，以实现更快的水平移动，我们认为这对于某些节奏较快的曲子非常重要

## 4.1 Evaluation on learning song-specific policies from demonstrations 
In this section, we evaluate the song-specific policy learning and aim to answer the following questions: (1) Does the integration of human demonstrations with RL help to achieve better performance? (2) Which elements of the learning algorithm are most important for good performance? 
>  本节，我们评估针对曲子的策略学习，并回答以下问题
>  (1) 人类演示和强化学习的结合是否助于达到更好的性能
>  (2) 学习算法的哪些要素对于良好的性能最重要

We use Proximal Policy Optimization (PPO) [27] because we found that it performs best compared to other RL algorithms. 

We compare our model to two baselines: 

**RoboPianist** [11] We use the RL method introduced in [11]. We keep the same reward functions as in the original work and manually label the fingering from the demonstration videos to provide the fingering reward. 

**Inverse Kinematics (IK)** [21] Given a fingertip trajectory demonstration $\tau_{\pmb x}$ , a Quadratic Programming-based IK solver [21] is used to compute a target, joint position trajectory and execute it open-loop. 

>  baseline 包括了 ROBOPIANIST 和 IK
>  ROBOPIANIST: 保留原始工作的奖励函数，其中指法奖励通过手动标记展示视频中的指法来提供
>  IK: 给定指尖轨迹演示 $\tau_{\pmb x}$，基于二次规划的 IK 求解器会被用于计算目标关节位置轨迹，并以开环方式执行

We select 10 clips from the collected dataset with different levels of difficulty. We individually train specialized policies for each of the 10 clips using both the baselines and our method. We then evaluate and compare their performance based on the obtained F1 score. 

![[pics/PianoMime-Fig3.png]]

**Performance.** As shown in Figure 3, our method consistently outperforms the RoboPianist baseline for all 10 clips, achieving an average F1 score of 0.94 compared to the baseline’s 0.74. We attribute this improvement to the incorporation of human priors, which narrows the RL search space to a favorable subspace, thereby encouraging the algorithm to converge on more optimal policies. 
>  我们的方法始终高于 baseline，平均 F1 score 为 0.94
>  我们将这一改进归因于人类先验的引入，它将 RL 的搜索空间缩小到了一个更有利的子空间，进而促进算法收敛到更优的策略

In addition, the IK method achieves an average F1 score of 0.70, only slightly lower than the baseline. This demonstrates the effectiveness of incorporating human priors, which provides a strong starting point for RL. 
>  此外，IK 方法的平均 F1 score 也仅略低于 ROBOPIANIST，这也说明了引入人类先验的有效性，人类先验为 RL 提供了良好的起点

We also observe that our method trains faster than the baseline. On an RTX 4090, the baseline took an average of 4 hours to train, while our method took an average of 2.5 hours. 
>  我们的方法也比 baseline 训练得更快

**Impact of Elements.** Our RL method has two main elements: a style-mimicking reward and a residual learning. We exclude each element individually to study their respective influences on policy performance (see Figure 3). 
>  我们的 RL 方法有两个主要元素: 风格模仿奖励和残差学习
>  我们分别排除每个元素，研究它们各自对策略性能的影响

We clearly observe the critical role of residual learning, which implies the benefit of using human demonstrations as nominal behavior. We observe a marginal performance increase of 0.03 when excluding the style-mimicking reward, but this also results in a larger discrepancy between the robot and human fingertip trajectories. Thus, the weight of the style-mimicking reward can be considered as a parameter that controls the human similarity of the learned robot actions. The ablation study for this weight is discussed in Appendix F. 
>  结果表明残差学习的作用最关键，这表明使用人类演示作为规范行为是有益的
>  我们发现排除风格模仿奖励，性能增加了 0.03，但这会导致机器人和人类的指尖轨迹差异更大
>  因此，风格模仿奖励的权重可以视作控制人类和机器人动作相似性的参数

![[pics/PianoMime-Fig4.png]]

**Hand Pose Visualization.** We provide Figure 3 and Figure 4 as an example of hand poses in different settings and provide attached videos on the website with further examples. 

In Figure 4, we exemplify that our policy places the hands in similar poses to the YouTube videos. We measure the distance between fingertips in YouTube videos and the robot in Appendix F. We observe that the IK nominal behavior leads the robot to place the fingers in positions similar to those in YouTube videos. The RL policy then slightly adapts the fingertip positions to press the keys correctly. 
>  我们的策略下，机器人的手姿态和视频中姿态更加相似
>  我们观察到 IK 基准行为使得机器人将手指放置在与 YouTube 视频中相似的位置，RL 策略则轻微调整了指尖位置，以正确按下琴键

Besides, we observe that the RoboPianist baseline sometimes presents visually unhuman-like motions. For example, in Figure 3 Left, the middle finger and the ring finger of the left robot hand are at relatively unhuman-like positions. 
>  此外，我们注意到 ROBOPIANIST baseline 有时会出现视觉上不自然的动作

## 4.2 Evaluation of model design strategies for policy distillation 
This section focuses on the evaluation of the policy distillation for playing different songs. We evaluate the influence of different policy design strategies on the agent’s performance. We aim to assess (1) the impact of integrating a pre-trained observation encoder to induce spatially consistent features, (2) the impact of a hierarchical design of the policy, and (3) the performance of different generative models on piano-playing data. 
>  本节评估策略蒸馏下，不同的策略设计对 agent 性能的影响
>  我们的目标是评估以下几点
>  (1) 集成与训练编码器以生成空间一致性特征的影响
>  (2) 策略的层次化设计的影响
>  (3) 不同的生成式模型的表现

**Proposed Models.** We propose two base policies, Two-stage Diff and Two-stage Diff-res policy. Both use hierarchical policies and goal representation learning, as described in Section 3.3. The only difference between them is that the low-level policy of Two-stage Diff predicts the target joints directly, while Two-stage Diff-res predicts the residual term of an IK solver. A detailed description of the policies can be found in Appendix I. 
>  我们提出了两种基础策略: Two-stage Diff 和 Two-stage Diff-res
>  二者都使用分层策略和目标表示学习，差异仅在于 Two-stage Diff 的低层策略直接预测目标关节，而 Two-stage Diff-res 则预测逆向运动学求解器的残差项

**Baselines.** We consider as baselines a Multi-task RL policy and a BC policy with MSE Loss from [11]. Additionally, we implement an Adversarial Inverse Reinforcement Learning (AIRL) baseline [28]. We provide further details of the models in Appendix I. 

**Ablation Models.** To analyze the impact of our policy design choices, we design three variants of our proposed model, i.e., w/o SDF: We train a policy that directly receives the 88-dimensional binary representation of the goal, without using the SDF observation encoder, to evaluate the impact of the goal’s representation learning, One-stage: We train an end-to-end diffusion policy to evaluate the impact of the hierarchical architecture, BeT: We train a two-stage Behavior-Transformer [23] to evaluate the impact of using diffusion models. 
>  我们设计了三种模型变体: 
>  w/o SDF: 直接接收目标的 88 维二进制表示，不使用 SDF observation encoder，以评估目标表示学习的影响
>  One-stage: 训练端到端的扩散策略，以评估分层架构的影响
>  BeT: 训练两阶段 Behavior-Tramsformer，以评估使用扩散模型的影响

![[pics/PianoMime-Table1.png]]

**Results.** As shown in Table 1, our methods (Two-stage Diff and Two-stage Diff-res) outperform the others on both training and test datasets. Multi-task RL and AIRL have higher precision on the test dataset, but this is because they barely press any keys. 
>  我们的方法 (Two-stage Diff, Two-stage Diff-res) 在训练集和测试集上都优于其他方法
>  Multi-task RL 和 AIRL 方法在测试集上的 precision 更改是因为它们几乎不按键

We observe a large improvement when using both diffusion policies instead of BeT, a hierarchical policy instead of an end-to-end policy and a slight improvement when using a pre-trained observation encoder, especially on the test dataset. 
>  我们观察到，使用扩散策略而不是 BeT 可以显著提高表现，使用层次化策略而不是端到端策略也可以显著提高表现，使用预训练的 observation encoder 可以略微提高表现，尤其是在测试集上

We also observe a slight performance improvement when the model predicts the residual term of IK (Two-stage Diff-res). 
>  我们观察到让模型预测 IK 的残差项可以略微提高表现

## 4.3 Evaluations on the impact of the data in the generalization 
In this section, we investigate the impact of scaling the training data on the generalization capabilities of the agent. We divide the experiments into two parts: 

(1) We evaluate the impact of scaling the training data on the performance of three policy designs (One-stage Diff, Two-stage Diff, and Two-stage Diff-res) evaluated on the test dataset by training them with different proportion of the dataset (see Figure 5 Top). 
(2) We evaluate the influence of a good high-level policy of Two-stage Diff by training different high-level policies on different proportions of the dataset (see Figure 5 Bottom). We provide additional results in Appendix M. 

>  (1) 我们通过使用不同比例的数据集来训练三种策略设计 (One-stage Diff, Two-stage Diff, Two-stage Diff-res) ，评估其在测试数据集上的性能变化
>  (2) 我们通过使用不同比例的数据集来训练不同的高级策略，评估 Two-stage Diff 中一个好的高层策略的影响

![[pics/PianoMime-Fig5.png]]

**Impact of scaling training data.** We observe that both Two-stage Diff and Two-stage Diff-res show consistent performance improvement when increasing the training data (Figure 5 Top). This trend implies that the two-stage policies have not yet reached their performance saturation with the given data and could potentially continue to benefit from additional training data in future works. 
>  Two-stage Diff, Two-stage Diff-res 都在训练数据增加时有持续的表现提升
>  这说明两阶段策略在当下的数据尚未达到性能饱和点，可能仍可以从更多训练数据受益

**Impact of high-level policy quality**. We further employ different combinations of the high-level and low-level policies of Two-stage Diff trained with different proportions of the dataset and assess their performance. 

In addition, we introduce a high-level oracle policy that outputs the ground-truth fingertip positions from the human demonstration videos. 
>  我们引入一个高阶的 oracle 策略，它直接输出人类展示视频中的真实指尖位置

The results (see Figure 5 Bottom) demonstrate that the overall performance of the policy is significantly influenced by the quality of the high-level policy. Low-level policies paired with Oracle high-level policies consistently outperform the ones paired with other high-level policies. Besides, we observe early performance convergence with increasing training data when paired with a low-quality high-level policy. 
>  策略的整体性能受到高级策略质量的显著影响，使用Orcale 高阶策略的策略优于使用其他高阶策略的策略
>  此外，我们观察到随着训练数据增加，性能会更快地收敛

## 4.4 Limitations 
**Inference Speed** One of the limitations is the inference speed. The models operate with an inference frequency of approximately $15\mathrm{{Hz}}$ on an RTX 4090 machine, which is lower than the standard real-time demand on hardware. Future works can employ faster diffusion models, e.g., DDIM [29], to speed up the inference. 
>  RTX 4090 上，模型的推理频率大约为 15Hz，这低于硬件上的标准实时需求
>  可以考虑采用更快的扩散模型，例如 DDIM 以加速推理过程

**Out-of-distribution Data** Most of the songs in our collected dataset are of modern style. When evaluating the model on the dataset from [11], which mainly contains classical songs, the performance degrades. It implies the model’s limited generalization across songs of different styles. Future work can collect more diverse training data to improve this aspect. 
>  我们收集的数据集中的曲子大多数为现代风格，在古典风格的曲子的数据集上评估时，性能有所下降
>  这表明模型在不同风格曲子上的泛化能力有限

**Acoustic Experience** Although the policy achieves up to $57\%$ F1-score on unseen songs, we found that higher accuracy is still necessary to make the song acoustically appealing and recognizable. Future work should focus on improving this accuracy to enhance the overall acoustic experience. 

# 5 Conclusion 
In this work, we present PianoMime, a framework for training a generalist robotic pianist using Internet video sources. 

The proposed framework is composed of three distinct phases: first, extract task-related and human motion-related trajectories from videos, second, train song-specific policies with reinforcement learning and finally, distill all the song-specific policies in a single generalist policy. 

We found that the resulting policy demonstrates an impressive generalization capability, achieving an average F1-score of $57\%$ on unseen songs. 

We believe that the findings for learning fine motor skills in piano playing can be applied to other tasks that require high dexterity and precision, and scenarios where robot data collection through teleoperation is challenging. 

# A Retargeting: From human hand to robot hand 
To retarget from the human hand to the robot hand, we follow a structured process. 
>  将人类的手重定位到机器人手遵循以下几个步骤

![[pics/PianoMime-Fig6.png]]

**Step 1: Homography Matrix Computation** Given a top-view piano demonstration video, we firstly choose $n$ different feature points on the piano. These points could be center points of specific keys, edges, or other identifiable parts of the keys that are easily recognizable (see Figure 6). 
>  1. 计算单应性矩阵
>  给定俯视视角的展示视频，我们首先在钢琴上选择 $n$ 个不同的特征点

Due to the uniform design of the pianos, these points represent the same physical positions in both the video and Mujoco. Given the chosen points, we follow the Eight-point Algorithm to compute the Homography Matrix $H$ that transforms the pixel coordinate in videos to the x-y coordinate in Mujoco (the z-axis is the vertical axis). 
>  由于钢琴设计是统一的，故这些特征点在 Mujoco 和视频中表示的是相同的物理位置
>  给定这些点，我们采用八点算法计算单应性矩阵 $H$，它将视频中的像素坐标转化为 Mujoco 中的 x-y 坐标

> [!info] Homography Matrix
>  Homography matrix (单应性矩阵)是一个 3×3 的矩阵，用于描述两个平面之间的投影变换关系
>  它能够将一个平面上的点映射到另一个平面上的对应点
> 
>  例如，在计算机视觉中，当我们有一张拍摄物体平面的图像，然后从另一个视角再拍摄一张同一物体平面的图像，homography matrix 可以将第一张图像中的点转换为第二张图像中对应的点


**Step 2: Transformation of Fingertip Trajectory** We then obtain the human fingertip trajectory with MediaPipe [20]. We collect the fingertips positions every 0.05 seconds. Then we transform the human fingertip trajectory within pixel coordinate into the Mujoco x-y 2D coordinate using the computed homography matrix $H$ . 
>  2. 转换指尖轨迹
>  我们使用 MediaPipe 获取人类指尖轨迹，我们隔 0.05 秒收集一次指尖位置，然后利用单应矩阵 $H$ 将人类指尖轨迹从像素坐标系转化到 Mujoco x-y 2D 坐标系

**Step 3: Heuristic Adjustment for Physical Alignment** We found that the transformed fingertip trajectory might not physically align with the notes, which means there might be no detected fingertip that physically locates at the keys to be pressed or the detected fingertip might locate at the border of the key (normally a human presses the middle point on the horizontal axis of the key). This misalignment could be due to the inaccuracy of the hand-tracking algorithm and the homography matrix. 
>  3. 物理对齐的启发式调整
>  我们发现，变换后的指尖轨迹可能无法在音符上物理对齐，这意味着检测到的指尖可能不会物理上位于需要按下的琴键上，或者位于琴键的边缘 (人类通常会在琴键的中轴点施加压力)
>  这种不对齐是由手部追踪算法和单应性矩阵的不准确性导致的

Therefore, we perform a simple heuristic adjustment on the trajectory to improve the physical alignment. Specifically, at each timestep of the video, we check whether there is any fingertip that physically locates at the key to be pressed. If there is, we adjust its y-axis value to the middle point of the corresponding key. Otherwise, we search within a small range, specifically the neighboring two keys, to find the nearest fingertip. If no fingertip is found in the range or the fingertip has been assigned to another key to be pressed, we then leave it. Otherwise, we adjust its y-axis value to the center of the corresponding key to ensure proper physical alignment. 
>  因此，我们为指尖轨迹进行简单的启发式调整，以改善物理对齐效果
>  我们在视频的每一个时间步检查是否有一个指尖位于需要按下的琴键上，如果有，我们将 y 轴的值调整为对应琴键的重点
>  否则，我们在一个小范围内 (相邻的两个琴键中) 搜索最近的指尖，如果没有找到，或者该指尖已经分配给另一个需要按下的琴键，就保持不变，否则将该指尖的 y 轴的值调整为该琴键的中心，以确保正确的物理对齐

**Step 4: ${z}$ -axis Value Assignment** Lastly, we assign the $z$ -axis value for the fingertips. For the fingertips that press keys, we set their ${z}$ -axis values to 0. For other fingertips, we set their z-axis value to $2\cdot h_{k e y}$ , where $h_{k e y}$ is the height of the keys in Mujoco. 
>  4. z 轴坐标赋值
>  我们最后为指尖赋 z 轴坐标值，对于按下琴键的指尖，我们将其 z 轴设定为 0，对于其他指尖，我们将 z 轴值设定为 $2\cdot h_{key}$，其中 $h_{key}$ 是 Mujoco 的琴键高度

# B Evaluation Metrics 
We use the same metrics from RoboPianist [11], i.e., Precision, Recall, and F1 score. Here we provide a detailed definitions of them: 

- True Positive (TP): Keys that should be pressed are pressed. 
- False Positive (FP): Keys that should not be pressed are pressed. 
- False Negative (FN): Keys that should be pressed are not pressed. 

$$
\mathrm{Precision}=\frac{T P}{T P+F P}
$$

$$
\mathrm{Recall}={\frac{T P}{T P+F N}}
$$ 
$$
\mathrm{F}1={\frac{2\cdot{\mathrm{Precision}}\cdot{\mathrm{Recall}}}{\mathrm{Precision}+\mathrm{Recall}}}
$$

Given the ground truth and the executed piano state trajectory, we calculate Precision, Recall, and F1 score for each timestep.  We then get the overall Precision, Recall, and F1 score by averaging them over timesteps. 
>  我们计算每个时间步的 Precision, Recall, F1 score，然后计算所有时间步的平均值

In this way, precision evaluates the robot’s capability of avoiding pressing the wrong keys, while recall evaluates the robot’s capability of pressing the correct keys. F1 score combines both of them. 

# C Implementation of Inverse Kinematics Solver 
The implementation of the IK solver is based on the approach of [21]. The solver addresses multiple tasks simultaneously by formulating an optimization problem and finding the optimal joint velocities that minimize the objective function. The optimization problem is given by: 

$$
\operatorname*{min}_{\dot{q}}\sum_{i}w_{i}\|J_{i}\dot{q}-K_{i}v_{i}\|^{2},
$$ 
where $w_{i}$ is the weight of each task, $K_{i}$ is the proportional gain and $v_{i}$ is the velocity residual. 

>  IK 求解器的实现基于 [21]，该求解器通过构建一个优化问题并找到最小化目标函数的最优关节速率来同时处理多个任务
>  优化问题的定义如上，其中 $w_i$ 是每个任务的权重，$K_i$ 是比例增益，$v_i$ 为速度残差

>  目标函数是一个关于关节速度 $\dot q$ 的函数，我们期望找到最小化目标函数的 $\dot q$
>  目标函数外层是一个对任务 $i$ 求和的和式，$w_i$ 是赋予每个任务的权重
>  $J_i$ 是 Jacobian 矩阵，它将关节速度 $\dot q$ 映射到任务空间的速度，因此 Jacobian 矩阵实际上描述了关节的运动是如何影响末端执行器在任务空间的运动
>  $K_i$ 为比例增益，$v_i$ 为速度残差，$K_iv_i$ 表示了我们期望末端执行器在任务空间达到的速度
>  $J_i\dot q - K_iv_i$ 表示了末端执行器实际速度和期望速度之间的差异，我们的目标是让末端执行器的实际运动尽可能接近期望的运动

We define a set of 10 tasks, each specifying the desired position of one of the robot’s fingertips. We do not specify the desired quaternions. All the weights $w_{i}$ are set to be equal. We use quadprog3 to solve the optimization problem with quadratic programming. The other parameters are listed in Table 2. 
>  我们定义了一组 10 个任务，每个任务指定机器人的一个指尖的预期位置

Table 2: The parameters of IK solver 

<html><body><center><table>
<tr><td>Parameter</td><td>Value</td></tr>
<tr><td>Gain</td><td>1.0</td></tr>
<tr><td>Limit Gain</td><td>0.05</td></tr>
<tr><td>Damping</td><td>1e-6</td></tr>
<tr><td>Levenberg-Marquardt Damping</td><td>1e-6</td></tr></table></center></body></html>

# D Detailed MDP Formulation of Song-specific Policy 
We present a detailed representation of the reward functions applied in our method in Table 3. 

![[pics/PianoMime-Table3.png]]

# E Training Details of Song-specific Policy 
We use PPO [27] (implemented by StableBaseline 3 [30]) to train the song-specific policy with residual RL(See Algorithm 1). All of the experiments are conducted using the same network architecture and tested using 3 different seeds. 

![[pics/PianoMime-Algorithm1.png]]

Both actor and critic networks are of the same architecture, containing 2 MLP hidden layers with 1024 and 256 nodes, respectively, and GELU [31] as activation functions. The detailed hyperparameters of the networks are listed in Table 6. 

Table 4: The observation space of song-specific agent. 

<html><body><center><table><tr><td>Observation</td><td>Unit</td><td>Size</td></tr><tr><td>Hand and Forearm Joint Positions</td><td>Rad</td><td>52</td></tr><tr><td>Hand and forearm Joint Velocities</td><td>Rad/s</td><td>52</td></tr><tr><td>Piano Key Joint Positions</td><td>Rad</td><td>88</td></tr><tr><td>Piano key Goal State</td><td>Discrete</td><td>88</td></tr><tr><td>Demonstrator Forearm and Fingertips Cartesian Positions</td><td>m</td><td>36</td></tr><tr><td>Prior control input u (solved by IK)</td><td>Rad</td><td>52</td></tr><tr><td>Sustain Pedal state</td><td>Discrete</td><td>1</td></tr></table></center></body></html> 


Table 5: The action space of song-specific agent. 

<html><body><center><table><tr><td>Action</td><td>Unit</td><td>Size</td></tr><tr><td>Target Joint Positions</td><td>Rad</td><td>46</td></tr><tr><td>Sustain Pedal</td><td>Discrete</td><td>1</td></tr></table></center></body></html> 


Table 6: The Hyperparameters of PPO 

<html><body><center><table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Initial Learning Rate</td><td>3e-4</td></tr><tr><td>Learning Rate Scheduler</td><td>Exponential Decay</td></tr><tr><td>Decay Rate</td><td>0.999</td></tr><tr><td>Actor Hidden Units</td><td>1024,256</td></tr><tr><td>Actor Activation</td><td>GELU</td></tr><tr><td>Critic Hidden Units</td><td>1024,256</td></tr><tr><td>Critic Activation</td><td>GELU</td></tr><tr><td>Discount Factor</td><td>0.99</td></tr><tr><td>Steps per Update</td><td>8192</td></tr><tr><td>GAE Lambda</td><td>0.95</td></tr><tr><td>Entropy Coefficient</td><td>0.0</td></tr><tr><td>Maximum Gradient Norm</td><td>0.5</td></tr><tr><td>Batch Size</td><td>1024</td></tr><tr><td> Number of Epochs per Iteration</td><td>10</td></tr><tr><td>Clip Range</td><td>0.2</td></tr><tr><td>Number of Iterations</td><td>2000</td></tr><tr><td>Optimizer</td><td>Adam</td></tr></table></center></body></html> 

# F Ablation Study for Weight of Style-mimicking Reward 
To numerically evaluate the human-likeness of the robot motion, we include an additional metric, $\Delta f t$ , which computes the average Euclidean distance between the robot fingertip positions and the human demonstrators for each timestep. 

We further make an ablation study to explore the impact of the weight of style-mimicking reward on $\Delta f t$ and F1 score, respectively (See Figure 7). 

The result indicates that when the weight of the mimic reward is zero, the F1 score is the highest, but the relative distance between the human fingertip positions and the robot’s fingertip positions is also the greatest. 

As we increase the influence of the mimic reward, the performance decreases, while the relative distance to the human fingertip positions also diminishes. 

This allows us to balance between improving performance and achieving a behavior more similar to the videos by adjusting the mimic reward. **The discrepancy is unavoidable since the robot’s embodiment differs from that of a human, and accurately playing the piano song might necessitate some deviation from human behavior.** 

![](https://cdn-mineru.openxlab.org.cn/extract/7f3ad8bb-6e34-4496-b9a0-fd278bc71392/bf60927c48d2f31897fd71498bd2f4258073e87a4652b2b3ba57a097c6c001c3.jpg) 

Figure 7: Impact of the weight of style-mimicking reward on $\Delta f t$ and F1 score 

# G Representation Learning of Goal 
We train an autoencoder to learn a geometrically continuous representation of the goal (See Figure 8 and Algorithm 2). 

![[pics/PianoMime-Fig8.png]]

During the training phase, the encoder $♪$ , encodes the original 88-dimensional binary representation of a goal piano state $♪_{t}$ into a **16-dimensional latent code $z$ .** 

The **positional encoding** of a randomly sampled 3D query coordinate $x$ is then concatenated with the latent code $z$ and passed through the decoder $\mathcal{D}$ . 

We use positional encoding here to represent the query coordinate more expressively. The decoder is trained to predict the SDF $f(x,♪_{t})$ . 

We define the SDF value of $x$ with respect to $♪_{t}$ as the Euclidean distance between the $x$ and the nearest key that is supposed to be pressed in $♪_{t}$ , mathematically expressed as: 

$$
\mathrm{SDF}({{x}},{♪}_{t})=\operatorname*{min}_{{{p}}\in\{p_{i}|{{♪}}_{t,i}=1\}}\|{{x}}-{{p}}\|,
$$ 
where $p_{i}$ represents the position of the $i$ -th key on the piano. The encoder and decoder are jointly optimized to minimize the reconstruction loss: 

$$
L(x,,{♪}_{t})=(\mathrm{SDF}(x,{♪}_{t})-\mathcal{D}(\mathcal{E}(v,x)))^{2}.
$$ 
![[pics/PianoMime-Algorithm2.png]]

We pre-train the autoencoder using the GiantMIDI dataset, which contains 10K piano MIDI files of 2,786 composers. 

The pre-trained encoder maps the $♪_{t}$ into the 16-dimensional latent code, which serves as the latent goal for behavioral cloning. 

The encoder network is composed of four 1D-convolutional layers, followed by a linear layer. Each successive 1D-convolutional layer has an increasing number of filters, specifically 2, 4, 8, and 16 filters, respectively. All convolutional layers utilize a kernel size of 3. The linear layer transforms the flattened output from the convolutional layers into a 16-dimensional latent code. The decoder network is an MLP with 2 hidden layers, each with 16 neurons. We train the autoencoder for 100 epochs with a learning rate of $1e-3$ . 

# H Training Details of Diffusion Model 
All the diffusion models utilized in this work, including One-stage Diff, the high-level and low-level policies of Two-stage Diff, Two-stage Diff-res, and Two-stage Diff w/o SDF, share the same network architecture. 

The network architecture is the same as the U-net diffusion policy in [22] and optimized with DDPM [32], except that we use temporal convolutional networks (TCNs) as the observation encoder, taking the concatenated goals (high-level policy) or fingertip positions (low-level policy) of several timesteps as input to extract the features on the temporal dimension. Each level of U-net is then conditioned by the outputs of TCNs through FiLM [33]. 

High-level policies take the goals over 10 timesteps and the current fingertip position as input and predict the human fingertip positions. In addition, we add a standard Gaussian noise on the current fingertip position during training to facilitate generalization. We further adjust the y-axis value of the fingertips pressing the keys in the predicted high-level trajectories to the midpoint of the keys. This adjustment ensures closer alignment with the data distribution of the training dataset. 

Low-level policies take the predicted fingertip positions, the goals over 4 timesteps, and the proprioception state as input to predict the robot’s actions. The proprioception state includes the robot joint positions and velocities, as well as the piano joint positions. We use 100 diffusion steps during training. To achieve high-quality results during inference, we find that at least 80 diffusion steps are required for high-level policies and 50 steps for low-level policies. 

Table 7: The Hyperparameters of DDPM 

<html><body><center><table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Initial Learning Rate</td><td>1e-4</td></tr><tr><td>Learning Rate Scheduler</td><td>Cosine</td></tr><tr><td>U-Net Filters Number</td><td>256,512, 1024</td></tr><tr><td>U-Net Kernel Size</td><td>５</td></tr><tr><td>TCN Filters Number</td><td>32,64</td></tr><tr><td>TCN Kernel Size</td><td>３</td></tr><tr><td>Diffusion Steps Number</td><td>100</td></tr><tr><td>Batch Size</td><td>256</td></tr><tr><td>Number of Iterations</td><td>800</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>EMA Exponential Factor</td><td>0.75</td></tr><tr><td>EMA Inverse Multiplicative Factor</td><td>1</td></tr></table></center></body></html> 

# I Policy Distillation Experiment 
**Two-stage Diff.** The model consists of a hierarchical policy with a pre-trained goal observation encoder, as described in Section 3.3. Note that the entire dataset is used for training the high-level policy, while only around $40\%$ of the collected clips (110K state-action pairs) are trained with RL and further used for training the low-level policy. The detailed network implementation is described in Appendix H. 

**Two-stage Diff w/o SDF.** We directly use the binary representation of the goal instead of the SDF embedding representation to condition the high-level and low-level policies. 

**Two-stage Diff-res** The model is close to Two-stage Diff, with slight changes. We employ an IK solver to compute the target joints given the fingertip positions predicted by the high-level policy. The low-level policy predicts a residual term around the IK solution instead of the robot’s actions. 

**Two-stage BeT.** We train both high-level and low-level policies with Behavior Transformer [23] instead of DDPM. The hyperparameter of Bet are listed in Table 8. 

**One-stage Diff.** We train a single diffusion model to predict the robot actions given the SDF embedding representation of goals and the proprioception state. 

**Multi-task RL.** We create a multi-task environment where for each episode a random song is sampled from the dataset. The observation and action space, as well as the reward function of the environment, follow the same settings as described in [11]. Consequently, we use Soft-Actor-Critic (SAC) [34] to train a single agent within the environment. Both the actor and critic networks are MLPs, each with 3 hidden layers, and each hidden layer contains 256 neurons. 

**BC-MSE.** We train a feedforward network to predict the robot action of the next timestep conditioned on the binary representation of goal and proprioception state with MSE loss. The feedforward network is an MLP with 3 hidden layers, each with 1024 neurons. 

**AIRL [28].** We use the same multi-task environment as Multi-task RL. We use an opensource implementation of AIRL based on PPO 5, where the actor and critic networks in PPO are MLPs, each consisting of three hidden layers with 256 neurons per layer. The reward and shaping term of the discriminator also use the same MLP architecture, with three hidden layers and 256 neurons in each layer. We collect expert state-action pairs by rolling out the song-specific policies, where the state consists of the song’s notes and proprioceptive state, and the action is the joint-space robot action. The collected data is then fed into the discriminator. 

Table 8: The Hyperparameters of Behavior Transformer 

<html><body><center><table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Initial Learning Rate</td><td>3e-4</td></tr><tr><td>Learning Rate Scheduler</td><td>Cosine</td></tr><tr><td>Number of Discretization Bins</td><td>64</td></tr><tr><td>Number of Transformer Heads</td><td>8</td></tr><tr><td> Number of Transformer Layers</td><td>8</td></tr><tr><td>Embedding Dimension</td><td>120</td></tr><tr><td>Batch Size</td><td>256</td></tr><tr><td>Number of Iterations</td><td>1200</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>EMA Exponential Factor</td><td>0.75</td></tr><tr><td>EMA Inverse Multiplicative Factor</td><td>1</td></tr></table></center></body></html> 

# J F1 Score of All Trained Song-Specific Policies 
Figure 10 shows the F1 score of all song-specific policies we trained. 

![](https://cdn-mineru.openxlab.org.cn/extract/7f3ad8bb-6e34-4496-b9a0-fd278bc71392/1d1598e79b6b61b98abc06e3a22247e811a0b06d0ba74ae08c7d239b9824fb38.jpg) 

Figure 10: F1 score of all 184 trained song-specific policies (descending order) 

# K Detailed Results on Test Dataset 
In Table 9 and Table 10, we show the Precision, Recall, and F1 score of each song in our collected test dataset and the Etude-12 dataset from [11], achieved by Two-stage Diff and Two-stage Diff-res, respectively. 

We observe an obvious performance degradation when testing on Etude-12 dataset. We suspect that the reason is due to **out-of-distribution data**, as the songs in the Etude-12 dataset are all classical, whereas our training and test dataset primarily consists of modern songs. 

Table 9: Quantitative results of each song in our collected test dataset 

<html><body><table><tr><td rowspan="2">Song Name</td><td colspan="3">Two-stage Diff</td><td colspan="3">Two-stage Diff-res</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Forester</td><td>0.81</td><td>0.70</td><td>0.68</td><td>0.79</td><td>0.71</td><td>0.67</td></tr><tr><td>Wednesday</td><td>0.66</td><td>0.57</td><td>0.58</td><td>0.67</td><td>0.54</td><td>0.55</td></tr><tr><td>Alone</td><td>0.80</td><td>0.62</td><td>0.66</td><td>0.83</td><td>0.65</td><td>0.67</td></tr><tr><td>Somewhere Only We Know</td><td>0.63</td><td>0.53</td><td>0.58</td><td>0.67</td><td>0.57</td><td>0.59</td></tr><tr><td>Eyes Closed</td><td>0.60</td><td>0.52</td><td>0.53</td><td>0.61</td><td>0.45</td><td>0.50</td></tr><tr><td>Pedro</td><td>0.70</td><td>0.58</td><td>0.60</td><td>0.67</td><td>0.56</td><td>0.47</td></tr><tr><td>Ohne Dich</td><td>0.73</td><td>0.55</td><td>0.58</td><td>0.75</td><td>0.56</td><td>0.62</td></tr><tr><td>Paradise</td><td>0.66</td><td>0.42</td><td>0.43</td><td>0.68</td><td>0.45</td><td>0.47</td></tr><tr><td>Hope</td><td>0.74</td><td>0.55</td><td>0.57</td><td>0.76</td><td>0.58</td><td>0.62</td></tr><tr><td>No Time To Die</td><td>0.77</td><td>0.53</td><td>0.55</td><td>0.79</td><td>0.57</td><td>0.60</td></tr><tr><td>The Spectre</td><td>0.64</td><td>0.52</td><td>0.54</td><td>0.67</td><td>0.50</td><td>0.52</td></tr><tr><td>Numb</td><td>0.55</td><td>0.44</td><td>0.45</td><td>0.57</td><td>0.47</td><td>0.48</td></tr><tr><td>Mean</td><td>0.69</td><td>0.54</td><td>0.56</td><td>0.71</td><td>0.55</td><td>0.57</td></tr></table></body></html> 

Table 10: Quantitative results of each song in the Etude-12 dataset 

<html><body><table><tr><td rowspan="2">Song Name</td><td colspan="3">Two-stage Diff</td><td colspan="3">Two-stage Diff-res</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>FrenchSuiteNo1Allemande</td><td>0.45</td><td>0.31</td><td>0.34</td><td>0.39</td><td>0.27</td><td>0.30</td></tr><tr><td>FrenchSuiteNo5Sarabande</td><td>0.29</td><td>0.23</td><td>0.24</td><td>0.24</td><td>0.18</td><td>0.19</td></tr><tr><td>PianoSonataD8451StMov</td><td>0.58</td><td>0.52</td><td>0.52</td><td>0.60</td><td>0.50</td><td>0.51</td></tr><tr><td>PartitaNo26</td><td>0.35</td><td>0.22</td><td>0.24</td><td>0.40</td><td>0.24</td><td>0.26</td></tr><tr><td>WaltzOp64No1</td><td>0.44</td><td>0.31</td><td>0.33</td><td>0.43</td><td>0.28</td><td>0.31</td></tr><tr><td>BagatelleOp3N04</td><td>0.45</td><td>0.30</td><td>0.33</td><td>0.45</td><td>0.28</td><td>0.32</td></tr><tr><td>KreislerianaOp16No8</td><td>0.43</td><td>0.34</td><td>0.36</td><td>0.49</td><td>0.34</td><td>0.36</td></tr><tr><td>FrenchSuiteNo5Gavotte</td><td>0.34</td><td>0.29</td><td>0.33</td><td>0.41</td><td>0.31</td><td>0.33</td></tr><tr><td>PianoSonataNo232NdMov</td><td>0.35</td><td>0.24</td><td>0.25</td><td>0.29</td><td>0.19</td><td>0.21</td></tr><tr><td>GolliwoggsCakewalk</td><td>0.60</td><td>0.43</td><td>0.45</td><td>0.57</td><td>0.40</td><td>0.42</td></tr><tr><td>PianoSonataNo21StMov</td><td>0.32</td><td>0.22</td><td>0.25</td><td>0.36</td><td>0.23</td><td>0.25</td></tr><tr><td> PianoSonataK279InCMajor1StMov</td><td>0.43</td><td>0.35</td><td>0.35</td><td>0.53</td><td>0.38</td><td>0.39</td></tr><tr><td>Mean</td><td>0.42</td><td>0.31</td><td>0.33</td><td>0.43</td><td>0.30</td><td>0.32</td></tr></table></body></html> 

# L Failure Cases 
For song-specific policies, because **the starting position of the hands is fixed to the middle of the piano**, we observe that some policies do not behave well at the beginning of the song. Particularly, when they are required to press the keys on the sides of the piano. 

For multi-song policies, especially for unseen songs, we observe that while the robot tends to press the desired keys, it sometimes wrongly presses the neighboring ones. This likely occurs because the model does not accurately learn the system dynamics. 

# M Extension: Evaluations on the impact of the data in the generalization 
In this section, we provide additional details on Section 4.3. We present the recall, precision, and F1 scores for the two experiments conducted in Section 4.3 in Figure 9. 

![](https://cdn-mineru.openxlab.org.cn/extract/7f3ad8bb-6e34-4496-b9a0-fd278bc71392/ed675c5ac1801d3eebae6b16c6831a40f22ed832dd64ca1086ac06ee801ca901.jpg) 

Figure 9: Precision, Recall, and F1 Score for policies trained with varying amounts of data volumes evaluated on the test dataset. Top: The models (One-Stage diffusion, Two-Stage Diffusion, and Two-Stage Diffusion-res) are trained with the same proportion of high-level and low-level datasets. Bottom: Two-stage diffusion models are trained with different proportions of high-level and low-level datasets. The x-axis represents the percentage of the low-level dataset utilized, while HL $\%$ indicates the percentage of the high-level dataset used. 

By observing the recall and precision, we can clearly observe that **increasing the dataset positively impacts both the precision and recall of the learned policy.** This indicates that the robot not only presses the proper keys more often (improves recall) but also avoids pressing the wrong keys equally often (improves precision). Thus, the observed improvement of the F1-score is led by both. 


