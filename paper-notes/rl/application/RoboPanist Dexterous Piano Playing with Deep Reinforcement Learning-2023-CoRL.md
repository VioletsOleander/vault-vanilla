# Abstract 
Replicating human-like dexterity in robot hands represents one of the largest open problems in robotics. Reinforcement learning is a promising approach that has achieved impressive progress in the last few years; however, the class of problems it has typically addressed corresponds to a rather narrow definition of dexterity as compared to human capabilities. To address this gap, we investigate piano-playing, a skill that challenges even the human limits of dexterity, as a means to test high-dimensional control, and which requires high spatial and temporal precision, and complex finger coordination and planning. 
>  用机器人手复刻人类手的灵巧性是机器人学中最大的开放问题
>  RL 在过去解决的问题在灵巧性方面的定义相较于人类能力更为狭窄，故我们研究了钢琴演奏这项技能，这项技能甚至挑战了人类灵巧性的极限，它是一个需要高度时空精确度和复杂手指协调和规划的高维控制问题

We introduce ROBOPIANIST, a system that enables simulated anthropomorphic hands to learn an extensive repertoire of 150 piano pieces where traditional model-based optimization struggles. We additionally introduce an open-sourced environment, benchmark of tasks, interpretable evaluation metrics, and open challenges for future study. Our website featuring videos, code, and datasets is available at https://kzakka.com/robopianist/. 
>  我们提出 ROBOPIANIST 系统，该系统使得模拟的人形手能够学习 150 首钢琴曲目，传统的基于模型的优化方法难以实现该任务
>  我们推出了一个开源环境、任务基准、可解释的评估指标，并且开放了未来的研究挑战

Keywords: high-dimensional control, bi-manual dexterity 

# 1 Introduction 
Despite decades-long research into replicating the dexterity of the human hand, high-dimensional control remains a grand challenge in robotics. This topic has inspired considerable research from both mechanical design [1, 2, 3] and control theoretic points of view [4, 5, 6, 7, 8]. 

Learning-based approaches have dominated the recent literature, demonstrating proficiency with in-hand cube orientation and manipulation [9, 10, 11] and have scaled to a wide variety of geometries [12, 13, 14, 15]. These tasks, however, correspond to a narrow set of dexterous behaviors relative to the breadth of human capabilities. In particular, most tasks are well-specified using a single goal state or termination condition, limiting the complexity of the solution space and often yielding unnatural-looking behaviors so long as they satisfy the goal state. 
>  基于学习的方法在近期文献中占据主导地位，这些方法已经可以熟练地控制手内立方体，并且已经拓展到更多的几何形状
>  但这些任务，相对于人类能力的广度而言，仅对应于狭窄的一组灵巧行为，特别是大多数任务使用单个目标状态或终止条件就能很好地定义，这限制了解空间的复杂性，并且通常会导致不自然的行为，只要它们满足目标状态即可

How can we bestow robots with artificial embodied intelligence that exhibits the same precision and agility as the human motor control system? 

In this work, we seek to challenge our methods with tasks commensurate with this complexity and with the goal of emergent human-like dexterous capabilities. To this end, we introduce a family of tasks where success exemplifies many of the properties that we seek in high-dimensional control policies. Our unique desiderata are (i) spatial and temporal precision, (ii) coordination, and (iii) planning. . 
>  我们试图用与人类运动控制的复杂程度相匹配的任务，并以实现类人的灵巧能力为目标来挑战我们的方法
>  为此，我们引入了一组任务，要成功实现该任务，需要具备我们在高维控制策略中所追求的许多特性，包括了 (i) 空间和时间精度 (ii) 协调性 (iii) 规划能力

We thus built an anthropomorphic simulated robot system, consisting of two robot hands situated at a piano, whose goal is to play a variety of piano pieces, i.e., correctly pressing sequences of keys on a keyboard, conditioned on sheet music, in the form of a Musical Instrument Digital Interface (MIDI) transcription (see Figure 1). 
>  我们构建了一个拟人化的模拟机器人系统，该系统由两个放置在钢琴上的机器人手组成，其目标是根据乐谱 (以 MIDI 转录的形式) 演奏各种钢琴曲目，即正确地按下键盘上的按键序列

> [!Info] MIDI
> MIDI (Musical Instrument Digital Interface) 即乐器数字接口，是一种用于电子乐器、计算机和其他音乐设备之间通信的标准协议和技术。
> 
> 历史背景
> MIDI 标准于 1983 年由一些电子乐器制造商共同制定，目的是解决不同品牌电子乐器之间无法相互连接和通信的问题，使得电子乐器可以像传统乐队一样协同演奏。
> 
> 基本概念
> - MIDI 信息：MIDI 设备之间传输的不是音频信号，而是一系列指令和数据，即 MIDI 信息。这些信息包括音符的开始和结束、音符的音高、音符的力度（即演奏的强度，决定了音符的响度）、音符的时值、控制变化（如音量、音调变化、踏板控制等）以及程序变更（用于选择不同的音色）等。
> - MIDI 通道：MIDI 通信使用 16 个独立的通道，每个通道可以传输不同的 MIDI 信息，这样就可以在一条 MIDI 线上同时控制多个乐器或设备，就像有 16 条独立的 “线路” 一样，实现复杂的音乐创作和编曲。
> 
> 硬件设备
> - MIDI 键盘：是一种带有琴键的控制器，能够发送 MIDI 信息。演奏者在键盘上演奏时，键盘会生成相应的音符信息，通过 MIDI 接口发送给其他设备，如音源模块、计算机等，进而产生声音。
> - MIDI 鼓垫：用于演奏打击乐部分。每个鼓垫对应不同的打击乐器音色，当演奏者敲击鼓垫时，会触发相应的 MIDI 音符信息，通过连接的设备产生打击乐声音。
> - MIDI 风控制器：模拟吹奏乐器的演奏方式，演奏者通过吹气和手指在控制器上的操作，生成 MIDI 信息，可用于控制合成器等设备发出类似管乐器的声音。
> - MIDI 接口：是 MIDI 设备之间进行通信的物理连接接口，有 DIN 接口和 USB 接口等类型。传统的 MIDI DIN 接口有 5 个针脚，用于发送和接收 MIDI 信号。随着技术的发展，USB 接口也成为了常见的 MIDI 接口类型，它具有传输速度快、连接方便等优点。

The robot hands exhibit high degrees of freedom (22 actuators per hand, for a total of 44), and are partially underactuated, akin to human hands. Controlling this system entails sequencing actions so that the hands are able to hit exactly the right notes at exactly the right times; simultaneously achieving multiple different goals, in this case, fingers on each hand hitting different notes without colliding; planning how to press keys in anticipation of how this would enable the hands to reach later notes under space and time constraints
>  这两支机器人手具有高的自由度 (每只手有 22 个执行元件，一共 44 个)，并且它们是部分欠驱动的，类似于人手
>  控制这个系统需要对动作进行排序，以便双手能在精确的时间击中正确的音符；需要同时实现多个不同的目标，例如每只手的手指按下不同的音符而不发生碰撞；需要规划如何按压琴键，在时间和空间限制下帮助双手到达后续的音符 (对应的琴键上)

> [!info] Partially Underactuated
> 在机械系统里，自由度指的是物体能够独立运动的参数数目，而驱动器/执行元件是为机械系统提供动力以实现运动的部件
> 所谓欠驱动，是指驱动器的数量少于系统运动的自由度数。部分欠驱动就是指机械系统中只有一部分自由度处于欠驱动状态，而不是所有的自由度都缺少足够的驱动器来直接控制。
> 
> 例子:
> 拿人手来说，人手有多个关节可以活动，比如手指的各个指关节可以弯曲、伸直，手腕也能旋转、弯曲等，这些不同的活动方式对应着很多自由度。但人手并不是每个自由度都有一个独立的肌肉来专门驱动，很多时候是多个肌肉协作来控制多个关节的运动，这就是部分欠驱动的一种体现。比如在握拳这个动作中，多个手指的关节同时运动，但并不是每个关节都有单独的肌肉发力来驱动，而是通过一组肌肉的协调收缩，带动手指各个关节依次弯曲，最终实现握拳的动作。

![[pics/RoboPianist-Fig1.png]]

We propose ROBOPIANIST, an end-to-end system that leverages deep reinforcement learning (RL) to synthesize policies capable of playing a diverse repertoire of musical pieces on the piano. We show that a combination of careful system design and human priors (in the form of fingering annotations) is crucial to its performance. Furthermore, we introduce ROBOPIANIST-REPERTOIRE-150, a benchmark of 150 songs, which allows us to comprehensively evaluate our proposed system and show that it surpasses a strong model-based approach by over $83\%$ . Finally, we demonstrate the effectiveness of multi-task imitation learning in training a single policy capable of playing multiple songs. 
>  ROBOPIANIST 是一个端到端的系统，基于 DRL 实现在钢琴上演奏多种音乐的策略，精心设计的系统和人类先验知识 (以指法注释的形式) 的结合对于其性能至关重要
>  我们还提出 ROBOPIANIST-REPETOIRE-150，一个包含 150 首曲子的数据集，使我们全面评估我们的系统，表明了它在 83% 的情况下由于基于强模型的方法
>  我们还展示了多任务模仿学习在训练一个能演奏多个歌曲的策略上的有效性

To facilitate further research and provide a challenging benchmark for high-dimensional control, we open source the piano-playing environment along with ROBOPIANIST-REPERTOIRE-150 at https://kzakka.com/robopianist/. 

# 2 Related Work 
We address related work within two primary areas: dexterous high-dimensional control, and robotic pianists. For a more comprehensive related work, please see Appendix A. 
>  相关工作被分类为两大类: 灵活的高维控制、机器人钢琴演奏者

**Dexterous Manipulation as a High-Dimensional Control Problem.** The vast majority of the manipulation literature uses lower-dimensional systems (i.e., single-arm, simple end-effectors), which circumvents challenges that arise in more complex systems. Specifically, only a handful of general-purpose policy optimization methods have been shown to work on high-dimensional hands, even for a single hand [10, 9, 12, 11, 16, 14, 17, 7], and of these, only a subset has demonstrated results in the real world [10, 9, 12, 11, 16]. Results with bi-manual hands are even rarer [15, 18]. In addition, the class of problems generally tackled in these settings corresponds to a definition of dexterity pertaining to traditional manipulation skills [19], such as re-orientation, relocation, manipulating simply-articulated objects (e.g., door-opening, ball throwing and catching), and using simple tools (e.g., hammer) [20, 21, 15, 11, 22, 12]. This effectively reduces the search space for controls to predominantly a single “basin-of-attraction" in behavior space per task. In contrast, our piano-playing task encompasses a more complex notion of a goal, extendable to arbitrary difficulty by only varying the musical score. 
>  大多数操纵相关的文献都是用低维系统 (例如单臂、简单末端执行器)，避开了更复杂系统上的挑战
>  并且，即便对于单个手而言，仅有少数几个通用目的的策略优化方法对于高维的手也是有效的，并且这些算法中只有部分算法在真实世界中展示了结果，带有双臂手的结果则更少
>  另外，这些文献解决的问题类别都对应于传统的控制技巧中的灵活性概念，例如重定向、重定位、操纵简单的铰接式的对象 (例如开门、接球和扔球)、使用简单的工具 (例如锤子)。这实际上将搜索空间限制在了每个任务在行为空间中的单个 “吸引域”
>  相较之下，弹钢琴的任务目标更加复杂，并且可以通过改变乐谱将难度变得任意复杂

**Robotic Piano Playing.** Robotic pianists have a rich history within the literature, with several works dedicated to the design of specialized hardware [23, 24, 25, 26, 27, 28], and/or customized controllers for playing back a song using pre-programmed commands [29, 30]. The works of Scholz [31], Yeon [32] use a dexterous hand to play the piano by leveraging a combination of inverse kinematics and offline trajectory planning. In Xu et al. [33], the authors formulate piano playing as an RL problem for a single Allegro hand on a miniature piano and leverage tactile sensor feedback. The piano playing tasks considered in these prior works are relatively simple (e.g., play up to six successive notes, or three successive chords with only two simultaneous keys pressed for each chord). On the other hand, ROBOPIANIST allows a general bi-manual controllable agent to emulate a pianist’s growing proficiency by providing a large range of musical pieces with graded difficulties. 
>  机器人钢琴演奏者有丰富的研究历史，有多项研究致力于专门硬件的设计和/或定制控制器的开发，用于通过预编程命令回放歌曲
>  Scholz[31] 和 Yeon[32] 的工作利用灵巧的手结合逆运动学和离线轨迹规划来弹奏钢琴。在 Xu 等人的研究[33]中，作者将钢琴演奏建模为一个单个 Allegro 手在迷你钢琴上的强化学习问题，并利用触觉传感器反馈。
>  这些先前工作的钢琴演奏任务相对简单 (例如，最多弹奏六个连续音符，或三个连续和弦，每个和弦仅同时按下两个键)，而 ROBOPIANIST 则允许一个通用的双手 agent 通过提供一系列难度分级的音乐作品来模拟钢琴家逐渐提高的熟练程度

# 3 Experimental Setup 
In this section, we introduce the simulated piano-playing environment as well as the musical suite used to train and evaluate our agent. 

**Simulation details.** We build our simulated piano-playing environment (depicted in Figure 1) using the open-source MuJoCo [34, 35] physics engine. The piano model is a full-size digital keyboard with 52 white keys and 36 black keys. We use a Kawai manual [36] as reference for the keys’ positioning and dimensions. Each key is modeled as a joint with a linear spring and is considered “active” when its joint position is within $0.5^{\circ}$ of its maximum range, at which point a synthesizer is used to generate a musical note. We also implement a mechanism to sustain the sound of any currently active note to mimic the mechanical effect of a sustain pedal on a real piano. The left and right hands are Shadow Dexterous Hand [37] models from MuJoCo Menagerie [38], which have been designed to closely reproduce the kinematics of the human hand. 
>  我们基于 MuJoCo 物理引擎构建模拟钢琴演奏环境
>  钢琴模型是一个全尺寸的数字硬盘，有 52 个白键和 36 个黑键，每个琴键被建模为一个带有线性弹簧的关节，并且当其关节位置是在其最大范围内的 $0.5^{\circ}$ 以内，就视为 “激活”，此时，合成器会生成一个音符
>  我们还实现了一个机制来维持当前激活音符的声音，以模仿真实钢琴上延音踏板的效果
>  左手和右手则是 MuJoCo Menagerie 中的 Shadow Dexterous Hand 模型，该模型已经经过设计以尽可能接近人类手部的运动学特性

**Musical representation.** We use the Musical Instrument Digital Interface (MIDI) standard to represent a musical piece as a sequence of time-stamped messages corresponding to note-on or note-off events. A message carries additional pieces of information such as the pitch of a note and its velocity. We convert the MIDI file into a time-indexed note trajectory (a.k.a, piano roll), where each note is represented as a one-hot vector of length 88. This trajectory is used as the goal representation for the agent, informing it which keys must be pressed at each time step. 
>  我们用 MIDI 标准将一首乐曲表示为一系列带有时间戳的消息，这些消息对应于音符的开启或关闭事件
>  此外，一条消息还带有额外的信息，例如音符的音高和其力度 (力度控制音量)
>  我们将 MIDI 文件转化为一个时间索引的音符轨迹 (即音乐软件中的 piano roll)，其中每个音符都被表示为长度为 88 的一维 one-hot 向量 (表示按下 88 个琴键中的哪一个)
>  该轨迹作为 agent 的目标表示，告知它在每个时间步需要按下哪个琴键

**Musical evaluation.** We use precision, recall, and F1 scores to evaluate the proficiency of our agent. These metrics are computed by comparing the state of the piano keys at every time step with the corresponding ground-truth state, averaged across all time steps. If at any given time there are keys that should be “on” and keys that should be “off”, precision measures how good the agent is at not hitting any of the “off” keys, while recall measures how good the agent is at hitting the “on” keys. The F1 score combines precision and recall into one metric, and ranges from 0 (if either precision or recall is 0) to 1 (perfect precision and recall). We primarily use the F1 score for our evaluations as it is a common heuristic accuracy score in the audio information retrieval literature [39], and we found empirically that it correlates with qualitative performance on our tasks. 
>  我们用精确率 ($\frac {TP}{TP + FP}$)、召回率 ($\frac {TP}{TP + FN}$)和 F1 score 来评估 agent 的熟练程度
>  这些指标是通过比较每个时间步下的钢琴键状态和真实状态，并对所有时间步计算平均值得到的
>  如果在任意给定的时间点，有应该是 “on” 的键和应该是 “off” 的键，精确率度量的就是 agent 不按下任何应该是 “off” 的键的表现，而召回率度量的是 agent 按下任何应该是 “on” 的键的表现
>  F1 score 结合了二者，范围从 0 (如果精确率或召回率为 0) 到 1 (精确率和召回率都是 1)
>  我们主要使用 F1 score 进行评估，因为它是在音频信息检索文献中常见的启发式准确率评分标准，我们通过试验发现它与我们任务的定性表现具有相关性

**MDP formulation.** We model piano-playing as a finite-horizon Markov Decision Process (MDP) defined by a tuple $(S,\mathcal{A},\rho,p,r,\gamma,H)$ where $\mathcal S\subset\mathbb{R}^{n}$ is the state space, $\mathcal{A}\subset\mathbb{R}^{m}$ is the action space, $\rho(\cdot)$ is the initial state distribution, $p(\cdot|s,a)$ governs the dynamics, $r:\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$ defines the rewards, $\gamma\in[0,1)$ is the discount factor, and $H$ is the horizon. The goal of an agent is to maximize its total expected discounted reward over the horizon: $\begin{array}{r}{\mathbb{E}\left[\sum_{t=0}^{H}\gamma^{t}r\big(s_{t},a_{t}\big)\right].}\end{array}$ 
>  我们将钢琴演奏建模为有限期 MDP，其中 $H$ 是时域长度 (最大时间步)
>  agent 的目标是最大化整个时域内的总期望折扣奖励 $\mathbb E\left[\sum_{t=0}^H \gamma^t r(s_t, a_t)\right]$

The agent’s observations consist of proprioceptive and goal state information. The proprioceptive state contains hand and keyboard joint positions. The goal state information contains a vector of key goal states obtained by indexing the piano roll at the current time step, as well as a discrete vector indicating which fingers of the hands should be used at that timestep.  
>  agent 的 observation 由本体感觉状态信息和目标状态信息组成
>  本地感觉状态包含了手部和键盘的关节位置信息；目标状态信息包含了两个向量，一个向量表示键的目标状态信息 (通过用当前时间步对 piano roll 索引得到)，另一个离散向量表示该时间步下应该用哪些手指按下对应的键

To successfully play the piano, the agent must be aware of at least a few seconds’ worth of its next goals in order to be able to plan appropriately. Thus the goal state is stacked for some lookahead horizon $L$ . performance, we discuss its design in Section 4.
>  要成功演奏钢琴，agent 必须至少了解未来几秒钟的目标，以适当地规划
>  因此，目标状态信息会在一定的 lookahead horizon $L$ 上堆叠

A detailed description of the observation space is given in Table 1.  
>  Table 1 描述了 agent 的 observation space

<html><center><body><table><tr><td>Observations</td><td>Unit</td><td>Size</td></tr><tr><td>Hand and forearm joints</td><td>rad</td><td>52</td></tr><tr><td>Forearm Cartesian position</td><td>m</td><td>6</td></tr><tr><td>Piano key joints</td><td>rad</td><td>88</td></tr><tr><td>Active fingers</td><td>discrete</td><td>L·10</td></tr><tr><td>Piano key goal state</td><td>discrete</td><td>L·88</td></tr></table></body></center></html>

Table 1: The agent’s observation space. $L$ corresponds to the lookahead horizon. 

>  Table 1 描述了 agent 的 observation 空间
>  其中前三列是本体感觉状态信息
>  - 第一列是手和前臂的关节角度，单位为弧度 (一个圆周角是 $2\pi$ 弧度)，一共有 52 个关节
>  - 第二列是前臂笛卡尔位置，单位为米，有 6 个维度应该是一条手臂的坐标 $(x, y, z)$ 对应其中的三个维度
>  - 第三列是钢琴键关节角度，单位为弧度，一共有 88 个关节 (88 个琴键)
> 后两列是目标状态信息

The agent’s action is 45 dimensional and consists of target joint angles for the hand with an additional scalar value for the sustain pedal. The agent predicts target joint angles at $20\mathrm{{Hz}}$ and the targets are converted to torques using PD controllers running at ${500}\mathrm{Hz}$ . 
>  agent 的动作是 45 维的，包括了 44 个手部关节的角度和 1 个额外的标量表示踏板持续值
>  agent 以 20 Hz (1 Hz 表示 1s 执行一次) 的频率预测目标关节角度，这些角度会被以 500 Hz 运行的 PD 控制器转换为扭矩

Since the reward function is crucial for learning performance, we discuss its design in Section 4.

**Fingering labels and dataset.** Piano fingering refers to the assignment of fingers to notes, e.g., “C4 played by the index finger of the right hand”. Sheet music will typically provide sparse fingering labels for the tricky sections of a piece to help guide pianists, and pianists will often develop their own fingering preferences for a given piece. Since fingering labels aren’t available in MIDI files by default, we used annotations from the PIG dataset [40] to create a corpus of 150 annotated MIDI files for use in the simulated environment. O
>  钢琴指法指为音符分配手指，例如 “右手食指演奏 C4”
>  乐谱通常会在作品的难点部分提供稀疏的指法标注以指导演奏家，演奏家通常会根据特定的作品发展自己的指法偏好
>  指法标记在 MIDI 文件中默认不包含，故我们使用 PIG 数据集中的注释，创建了一个包含 150 个带标注的 MIDI 文件的语料库，以在模拟环境中使用

verall this dataset we call REPERTOIRE-150 contains piano pieces from 24 Western composers spanning baroque, classical and romantic periods. The pieces vary in difficulty, ranging from relatively easy (e.g., Mozart’s Piano Sonata K 545 in C major) to significantly harder (e.g., Scriabin’s Piano Sonata No. 5) and range in length from tens of seconds to over 3 minutes. 
>  我们将该数据集称为 REPERTOIRE-150，它包含了 24 位西方作曲家的作品
>  这些作品难度各异，有非常简单的，有非常难的，时长从几十秒到超过 3 分钟不等

# 4 ROBOPIANIST System Design 
Our aim is to enable robots to exhibit sophisticated, high-dimensional control necessary for successfully performing challenging musical pieces. Mastery of the piano requires (i) spatial and temporal precision (hitting the right notes, at the right time), (ii) coordination (simultaneously achieving multiple different goals, in this case, fingers on each hand hitting different notes, without colliding), and (iii) planning (how a key is pressed should be conditioned on the expectation of how it would enable the policy to reach future notes). 
>  我们的目标是让机器人具备演奏高难度音乐作品所必须的复杂、高纬度控制能力
>  熟练演奏要求了以下几点
>  (i) 空间和时间精确度 (在准确的时间弹奏正确的音符)
>  (ii) 协调 (同时达到多个目标，即每只手的手指需要弹奏不同的音符，且不会冲突)
>  (iii) 规划 (演奏当前音符的方式还需要基于如何能弹奏到未来的音符的预期)

These behaviors do not emerge if we solely optimize with a sparse reward for pressing the right keys at the right times. The main challenge is exploration, which is further exacerbated by the high-dimensional nature of the control space. 
>  如果仅通过稀疏的奖励 agent 在正确的时间弹奏正确的琴键来进行优化，以上的行为都不会出现
>  主要的挑战在于 exploration, 而在高维控制空间中，这是困难的

![[pics/RoboPianist-Fig2.png]]

We overcome this challenge with careful system design and human priors, which we detail in this section. The main results are illustrated in Figure 2. We pick 3 songs in increasing difficulty from ROBOPIANIST-REPERTOIRE-150. We note that “Twinkle Twinkle” is the easiest while “Nocturne” is the hardest. We train for 5M samples, 3 seeds. We evaluate the F1 every 10K training steps for 1 episode (no stochasticity in the environment). 
>  为此，我们结合了系统设计和人类先验，主要的结果展示于 Fig2
>  我们训练了 500 万个样本, 3 个随机种子, 在每 10K 次训练步后评估一次一个回合的 F1 score (环境中没有随机性)

## 4.1 Human priors 
We found that the agent struggled to play the piano with a sparse reward signal due to the exploration challenge associated with the high-dimensional action space. To overcome this issue, we incorporated the fingering labels within the reward formulation (Table 2). When we remove this prior and only reward the agent for the key press, the agent’s F1 stays at zero and no substantial learning progress is made. We suspect that the benefit of fingering comes not only from helping the agent achieve the current goal, but facilitating key presses in subsequent timesteps. 
>  我们发现，由于高维动作空间带来的 exploration 挑战，agent 在稀疏的奖励信号下难以弹奏好钢琴
>  为了解决该问题，我们将指法标签也结合入奖励函数中，如果我们移除这一先验知识，仅根据按键行为奖励 agent, agent 的 F1 score 将在训练中始终保持为 0，不会出现显著的学习进展
>  我们推测，指法标记的好处不仅在于帮助 agent 达成当前目标，也有助于后续时间步的按键操作

![[pics/RoboPianist-Table2.png]]

Having the policy discover its own preferred fingering, like an experienced pianist, is an exciting direction for future research. 

## 4.2 Reward design 
We first include a reward proportional to how depressed the keys that should be active are. 
>  我们首先引入一个与应激活的按键的按下程度成比例的奖励: $0.5\cdot g(\|k_s - k_g\|_2)$，其中 $k_s$ 表示键的当前状态，$k_g$ 表示键的目标状态，$g$ 是一个函数，将距离转化为 $[0, 1]$ 范围内的标量

We then add a constant penalty if any inactive keys are pressed hard enough to produce sound. 
>  然后，我们在如果按下任意不应激活的按键，并且力度足够产生声音的情况下，添加一个常数惩罚项: $0.5\cdot (1 - \mathbf 1_{\{\text{false positive}\}})$
>  其中 $\mathbf 1_{\{\text{false positive}\}}$ 表示如果按响了任意不应激活的按键，即出现 false positive 时，就为 $1$，此时 $0.5\cdot (1 - \mathbf 1_{\{\text{false positive}\}}) = 0$，即 agent 将不会收到 $0.5$ 的奖励

This gives the agent some leeway to rest its fingers on inactive keys so long as they don’t produce sound. 
>  因为只有键被激活才会触发惩罚，故 agent 可以将手指暂放在不应激活的按键上，只要不发出声音

We found that giving a constant penalty regardless of the number of false positives was crucial for learning; otherwise, the agent would become too conservative and hover above the keyboard without pressing any keys. 
>  我们发现，无论 false positive 的数量如何，都给定一个恒定的惩罚，对于学习至关重要，否则，agent 将会过于保守，悬停在键盘上方，不按下任何键

In contrast, the smooth reward for pressing active keys plays an important role in exploration by providing a dense learning signal. 
>  相比之下，按下活跃按键对应的平滑奖励通过提供密集的学习信号，在 exploration 中起到关键作用

We introduce two additional shaping terms: (i) we encourage the fingers to be spatially close to the keys they need to press (as prescribed by the fingering labels) to help exploration, and 
>  我们引入了两个额外的项
>  (i) 我们鼓励手指在空间上接近它们将要按下的键 (指法标签描述了应该每个手指在各个时间步应该对应按下键)，以帮助 exploration
>  对应的项是 $g(\|p_f - p_k\|_2)$，其中 $p_f$ 表示手指的位置，$p_k$ 表示键的位置

(ii) we minimize energy expenditure, which reduces variance (across seeds) and erratic behavior control policies trained with RL are prone to generate. 
>  (ii) 我们最小化能量消耗，以减少随机种子之间的方差，并降低使用 RL 训练的控制策略容易产生的波动和异常行为
>  能量消耗项是 $|\tau_{\text{joints}}|^T|v_{\text{joints}}|$，其中 $\tau$ 表示关节扭矩，$v$ 为关节 (角) 速度，该项的权重系数是负数

> [!info] Energy Expenditure
> 能量消耗项涉及了关节扭矩和关节角速度
> 关节扭矩 $\tau_{\text{joints}}$ 是作用在关节上的力矩，单位通常是牛顿米 ($N\cdot m$)，扭矩的大小和方向决定了关节的旋转趋势
> 关节角速度 $v_{\text{joints}}$ 表示了关节旋转的快慢程度，单位为弧度每秒 $rad/s$，角速度的大小和方向反映了关节旋转的动态特性
> 
> 功率的定义是力 (或力矩) 与速度 (或角速度) 的乘积，单位为 $W$ 
> 对于每一个关节所做的旋转运动，其功率 $P = \tau \cdot v$
> 功率乘以时间就得到了消耗的能量，单位为 $J$

The total reward at a given time step is a weighted sum over the aforementioned components. A detailed description of the reward function can be found in Appendix B. 
>  在给定时间步下的总奖励是上述成分的加权和

## 4.3 Peeking into the future 
We observe additional improvements in performance and variance from including future goal states in the observation, i.e., increasing the lookahead horizon $L$ . Intuitively, this allows the policy to better plan for future notes – for example by placing the non-finger joints (e.g., the wrist) in a manner that allows more timely reaching of notes at the next timestep. 
>  我们观察到，在 agent 的 observation 中包含入未来的目标状态 (即增加 lookahead horizon $L$) 可以额外提高性能和降低方差
>  直观上看，未来的状态便于策略更好为未来音符规划——例如通过调节非手指关节 (手腕)，使得下一个时刻能更及时让手指达到目标音符

## 4.4 Constraining the action space 
To alleviate exploration even further, we explore disabling degrees of freedom [41] in the Shadow Hand that either do not exist in the human hand (e.g., the little finger being opposable) or are not strictly necessary for most songs. We additionally reduce the joint range of the thumb. While this speeds up learning considerably, we observe that with additional training time, the full action space eventually achieves similar F1 performance. 
>  为了进一步缓解 exploration 的困难，我们尝试在 Shadow Hand 模型中禁用一些自由度，这些自由度要么在人类手中不存在 (例如，小指可以对掌)，要么在大多数歌曲中不是严格必要的
>  我们还降低了拇指的关节活动范围
>  降低自由度显著加快了学习，但我们观察到，给定额外的训练时间，在完整的动作空间下也可以最终达到类似的 F1 score

# 5 Results 
In this section, we present our experimental findings on ROBOPIANIST-ETUDE-12, a subset of ROBOPIANIST-REPERTOIRE-150 consisting of 12 songs. 

The results on the full ROBOPIANIST-REPERTOIRE-150 can be found in Appendix C. We design our experiments to answer the following questions: 

(1) How does our method compare to a strong baseline in being able to play individual pieces? 
(2) How can we enable a single policy to learn to play more than one song?
(3) What effects do our design decisions have on the feasibility of acquiring highly complex, dexterous control policies? 

>  我们设计试验，以回答以下问题
>  (1) 在演奏单独的曲子时，我们的方法和 baseline 的比较结果如何
>  (2) 我们如何让单个策略可以演奏多首曲子
>  (3) 我们的设计决策对于获取高度复杂、灵活的控制策略有何帮助

## 5.1 Specialist Policy Learning 
For our policy optimizer, we use a state-of-the-art model-free RL algorithm DroQ [42], one of several regularized variants of the widely-used Soft-Actor-Critic [43] algorithm. We evaluate online predictive control (MPC) as a meaningful point of comparison. Specifically, we use the implementation from Howell et al. [44] that leverages the physics engine as a dynamics model, and which was shown to solve previously-considered-challenging dexterous manipulation tasks [9, 13, 11] in simulation. Amongst various planner options in [44], we found the most success with Predictive Sampling, a derivative-free sampling-based method. 
>  我们的策略优化器使用 DroQ，它是 SAC 的多个正则化变体之一
>  我们将 MPC 方法作为 baseline，它使用物理引擎作为动态模型，能够在模拟环境中解决具有挑战性的灵巧操纵任务，我们发现 MPC 在使用 Predictive Sampling (一种基于采样的无导数方法) 作为规划方法时最为有效

A detailed discussion of this choice of baseline and its implementation can be found in Appendix F. 

Our method uses 5 million samples to train for each song using the same set of hyperparameters, and the MPC baseline is run at one-tenth of real-time speed to give the planner adequate search time. 
>  我们使用 500 万个样本 (500 万次按键？500 万次演奏？) 来训练每一首曲子，所有曲子使用相同的超参数集合
>  MPC baseline 以现实时间的 1/10 运行，以为规划器提供足够搜索时间

![[pics/RoboPianist-Fig4.png]]

The quantitative results are shown in Figure 4. We observe that the ROBOPIANIST agent significantly outperforms the MPC baseline, achieving an average F1 score of 0.79 compared to 0.43 for MPC. 
>  定量结果见 Fig4
>  ROBOPIANIST agent 的平均 F1 score 为 0.79，MPC 为 0.43

We hypothesize that the main bottleneck for MPC is compute: the planner struggles with the large search space which means the quality of the solutions that can be found in the limited time budget is poor. 
>  我们推测 MPC 的主要瓶颈在于计算: 规划器难以在过大的搜索空间有效搜索，故在有限时间内找到的解的质量也有限

Qualitatively, our learned agent displays remarkably skilled piano behaviors such as (1) simultaneously controlling both hands to reach for notes on opposite ends of the keyboard, (2) playing chords by precisely and simultaneously hitting note triplets, and (3) playing trills by rapidly alternating between adjacent notes (see Figure 3). 
>  定性地说，我们学到的 agent 具有熟练的演奏行为，例如
>  (1) 同时控制双手触及键盘两端的音符
>  (2) 精确地同时按下三连音来演奏和弦
>  (3) 通过快速交替相邻音符演奏颤音

![[pics/RoboPianist-Fig3.png]]

We encourage the reader to listen to these policies on the supplementary website1. 

## 5.2 Multi-Song Policy Learning 
**Multitask RL** Ideally, a single agent should be able to learn how to play all songs. Secondly, given enough songs to practice on, we would like this agent to zero-shot generalize to new ones. 
>  理想情况下，单个 agent 应该可以演奏所有曲子，并且，如果练习了足够的曲子，agent 应该可以零样本泛化到新的曲子

To investigate whether this is possible, we create a multi-task environment where the number of tasks in the environment corresponds to the number of songs available in the training set. Note that in this multi-song setting, environments of increasing size are additive (i.e., a 2-song environment contains the same song as the 1-song environment plus an additional one). 
>  为了研究这是否可行，我们创建了一个多任务环境，环境中的任务数量对应于训练集中的曲子数量
>  需要注意的是，在这一设定下，环境是可以累加的 (即一个包含了两个曲子的环境可以由包含了一个曲子的环境再加上一个曲子得到)

>  应该可以理解为，为环境加上一首曲子就是 agent 在弹完了一首曲子之后，回合还没有结束，还需要弹下一首曲子

We use Für Elise as the base song and report the performance of a single agent trained on increasing amounts of training songs in Figure 5. 
>  我们用 Für Elise 作为基本环境，Fig5 报告了单一 agent 在不同数量曲子下训练的表现

![[pics/RoboPianist-Fig5.png]]

We observe that training on an increasing amount of songs is significantly harder than training specialist policies on individual songs. Indeed, the F1 score on Für Elise continues to drop as the number of training songs increases, from roughly 0.7 F1 for 1 song (i.e., the specialist) to almost 0 F1 for 16 songs. 
>  我们观察到，在更多的曲子上训练比针对单个曲子训练更加困难
>  随着训练曲子的数量增加，在 Für Elise 上的 F1 score 持续下降，从单个曲子的 0.7 到 16 个曲子的 0

We also evaluate the agent’s zero-shot performance (i.e., no fine-tuning) on ROBOPIANIST-ETUDE-12 (which does not have overlap with the training songs) to test our RL agent’s ability to generalize. We see, perhaps surprisingly, that multitask RL training fails to positively transfer on the test set regardless of the size of the pre-training tasks. 
>  我们还评估了 agent 的零样本表现，我们发现，无论预训练任务的大小如何，agent 都无法在测试集上实现正向迁移

As we increase the model size, the F1 score on the training song improves, suggesting that larger models can better capture the complexity of the piano playing task across multiple songs. 
>  随着我们增大模型大小，训练曲子上的 F1 score 会提升，这说明了更大的模型可以更好捕获多个歌曲演奏任务的复杂性

**Multitask Behavioral Cloning** Since multitask training with RL is challenging, we instead distill the specialist RL policies trained in Subsection 5.1 into a single multitask policy with Behavioral Cloning (BC) [45]. 
>  因为使用 RL 进行多任务训练具有挑战性，故我们转为通过行为克隆，将单个曲子的专家策略蒸馏为一个多任务策略

To do so, we collect 100 trajectories for each song in the ROBOPIANIST-REPERTOIRE-150, hold out 2 for validation, and use the ROBOPIANIST-ETUDE-12 trajectories for testing. 
>  为此，我们为数据集中的每个曲子都收集 100 条轨迹 (让各个曲子的专家策略演奏 100 次)，并保留其中的 2 条用于验证，使用 ROBOPIANIST-ETUDE-12  的轨迹进行测试

We then train a feed-forward neural network to predict the expert actions conditioned on the state and goal using a mean squared error loss. For a more direct comparison with multi-task RL, the dataset subsets use trajectories from the same songs used in the equivalently sized multi-task RL experiments. 
>  我们训练前馈网络来基于状态和目标，预测专家动作，使用 MSE 损失

![[pics/RoboPianist-Fig6.png]]

We observe in Figure 6a that as we increase the number of songs in the training dataset, the model’s ability to generalize improves, resulting in higher test F1 scores. 
>  Fig 6a 中，随着训练集曲子数量增加，模型的泛化能力会提高，测试 F1 score 会更高

We note that for a large model (hidden size of 1024), training on too few songs results in overfitting because there isn’t enough data. Using a smaller model alleviates this issue, as shown in the more detailed multitask BC results found in Appendix E, but smaller models are unable to perform well on multiple songs. 

Despite having better generalization performance than RL, zero-shot performance on ROBOPIANIST-ETUDE-12 falls far below the performance achieved by the specialist policy. 
>  零样本表现依旧远远低于专家策略的表现

Additionally, we investigate the effect of model size on the multitask BC performance. We train models with different hidden dimensions (fixing the number of hidden layers) on a dataset of 64 songs. As shown in Figure 6b, a smaller hidden dimension results in lower F1 performance which most likely indicates underfitting. 
>  Fig 6b 中，随着模型大小增大，模型的泛化能力会提高，说明了小模型会欠拟合

## 5.3 Further analysis 
In this section, we discuss the effect of certain hyperparameters on the performance of the RL agent. 

![[pics/RoboPianist-Fig7.png]]

**Control frequency and lookahead horizon:** Figure 7 illustrates the interplay between the control frequency (defined as the reciprocal of control timestep) and the lookahead horizon $L$ , and their effect on the F1 score. Too large of a control timestep can make it impossible to play faster songs, while a smaller control timestep increases the effective task horizon, thereby increasing the computational complexity of the task. 
>  Fig7 展示了控制频率 (定义为控制时间步长的倒数) 和前瞻范围 $L$ 之间的相互作用，及其对 F1 score 的影响
>  过大的控制时间步长可能导致无法演奏较快的曲子，过小的控制时间步长可以增加任务的有效时间范围，但会增大任务的计算复杂度 (在相同的一段时间内需要执行更多次的决策)

Lookahead on the other hand controls how far into the future the agent can see goal states. We observe that as the control timestep decreases, the lookahead must be increased to maintain the agent ability to see and reason about future goal states. 
>  前瞻范围控制了 agent 能够看到多远的未来目标状态
>  我们观察到，随着控制时间步长的减小，需要增大前瞻范围以维持 agent 能够看到并分析未来目标状态的能力 (因为实际的前瞻时间是前瞻范围和控制步长的乘积)

A control frequency of $20\mathrm{{Hz}}$ (0.05 seconds) is a sweet spot, with notably $100\mathrm{Hz}$ (0.01 seconds) drastically reducing the final score. At $100\mathrm{Hz}$ , the MDP becomes too long-horizon, which complicates exploration, and at $10\mathrm{{Hz}}$ , the discretization of the MIDI file becomes too coarse, which negatively impacts the timing of the notes. 
>  控制频率为 20Hz 较为理想，提高到 100Hz 会显著降低最终得分
>  在 100Hz 的控制频率下，整个 MDP 变得过于长远 (完成相同时间的曲子演奏需要执行更多的决策步骤)，导致 exploration 变得复杂
>  在 10 Hz 下，MIDI 文件的离散化会过于粗糙，对音符的时间准确性产生负面影响

>  从这段话可以推断出控制频率是和 MIDI 文件的离散化粒度相匹配的，会根据不同的控制频率以不同的粒度离散化 MIDI 文件

**Discount factor:** As shown in Figure 8, the discount factor has a significant effect on the F1 score. We sweep over a range of discount factors on two songs of varying difficulty. Notably, we find that discounts in the range 0.84 to 0.92 produce policies with roughly the same performance and high discount factors (e.g., 0.99, 0.96) result in lower F1 performance. 
>  折扣因子对于 F1 score 也有显著影响，过高的折扣因子的表现较差

Qualitatively, on Für Elise, we noticed that agents trained with higher discounts were often conservative, opting to skip entire note sub-segments. However, agents trained with lower discount factors were willing to risk making mistakes in the early stages of training and thus quickly learned how to correctly strike the notes and attain higher F1 scores. 
>  定性地说，在 Für Elise 中，我们注意到过高的折扣因子训练得到的 agent 会过于保守，倾向于跳过整个音符的小节
>  折扣因子较低时，agent 愿意在训练初期冒一定的错误风险，因此能够更快学会

# 6 Discussion 
**Limitations** While ROBOPIANIST produces agents that push the boundaries of bi-manual dexterous control, it does so in a simplified simulation of the real world. For example, the velocity of a note, which modulates the strength of the key press, is ignored in the current reward formulation. Thus, the dynamic markings of the composition are ignored. Furthermore, our RL training approach can be considered wasteful, in that we learn by attempting to play the entire piece at the start of every episode, rather than focusing on the parts of the song that need more practicing. Finally, our results highlight the challenges of multitask learning especially in the RL setting. 
>  agent 的训练是在现实世界中的简化模拟中实现的，例如，当前的奖励公式忽视了音符 (关节) 的角速度 (它表示按键的力度)，因此忽视了乐谱中的动态标记
>  另外，我们的 RL 训练方式有所浪费，我们在每次回合开始都让 agent 完整演奏整个曲目，而不是专注于需要更多练习的部分

**Conclusion** In this paper, we introduced ROBOPIANIST, which provides a simulation framework and suite of tasks in the form of a corpus of songs, together with a high-quality baseline and various axes of evaluation, for studying the challenging high-dimensional control problem of mastering piano-playing with two hands. 

Our results demonstrate the effectiveness of our approach in learning a broad repertoire of musical pieces, and highlight the importance of various design choices required for achieving this performance. 

There is an array of exciting future directions to explore with ROBOPIANIST including: leveraging human priors to accelerate learning (e.g., motion priors from YouTube), studying zero-shot generalization to new songs, incorporating multimodal data such as sound and touch. We believe that ROBOPIANIST serves as a valuable platform for the research community, enabling further advancements in high-dimensional control and dexterous manipulation. 

# A Extended Related Work 
We address related work within two primary areas: dexterous high-dimensional control, and robotic pianists. 

**Dexterous Manipulation and High-Dimensional Control** The vast majority of the control literature uses much lower-dimensional systems (i.e., single-arm, simple end-effectors) than high-dimensional dexterous hands. Specifically, only a handful of general-purpose policy optimization methods have been shown to work on high-dimensional hands, even for a single hand [10, 9, 12, 11, 16, 14, 17, 7], and of these, only a subset has demonstrated results in the real world [10, 9, 12, 11, 16]. Results with bi-manual hands are even rarer, even in simulation only [15, 18]. 

As a benchmark, perhaps the most distinguishing aspect of ROBOPIANIST is in the definition of “task success”. As an example, general manipulation tasks are commonly framed as the continual application of force/torque on an object for the purpose of a desired change in state (e.g., SE(3) pose and velocity). Gradations of dexterity are predominantly centered around the kinematic redundancy of the arm or the complexity of the end-effector, ranging from parallel jaw-grippers to anthropomorphic hands [46, 15]. A gamut of methods have been developed to accomplish such tasks, ranging from various combinations of model-based and model-free RL, imitation learning, hierarchical control, etc. [47, 10, 13, 12, 48, 49]. 
>  ROBOPIANIST 作为 banchmark，最为显著的特点在于对 “task success” 的定义
>  通用的操纵任务通常被描述为对物体施加连续的力/扭矩以实现期望的状态变化，对灵巧性的评判剧中在机械臂的动力学冗余和末端执行器的复杂性上

However, the class of problems generally tackled corresponds to a definition of dexterity pertaining to traditional manipulation skills [19], such as re-orientation, relocation, manipulating simply-articulated objects (e.g., door opening, ball throwing and catching), and using simple tools (e.g., hammer) [20, 21, 15, 11, 22]. The only other task suite that we know of that presents bi-manual tasks, the recent Bi-Dex [15] suite, presents a broad collection of tasks that fall under this category. 

While these works represent an important class of problems, we explore an alternative notion of dexterity and success. In particular, for most all the aforementioned suite of manipulation tasks, the “goal” state is some explicit, specific geometric function of the final states; for instance, an open/closed door, object re-oriented, nail hammered, etc. This effectively reduces the search space for controls to predominantly a single “basin-of-attraction" in behavior space per task. 
>  我们探索了另一种关于灵巧性和成功的概念
>  在之前的操纵任务中，“目标” 状态是最终状态的一些确定的几何函数，例如，打开/关闭的门、重新定位的物体、钉子被敲入等
>  这实际上将控制的搜索空间限制在了每个任务的行为空间中的单个 “吸引域” 中

In contrast, the ROBOPIANIST suite of tasks encompasses a more complex notion of a goal, which is encoded through a musical performance. In effect, this becomes a highly combinatorically variable sequence of goal states, extendable to arbitrary difficulty by only varying the musical score. “Success” is graded on accuracy over an entire episode; concretely, via a time-varying non-analytic output of the environment, i.e., the music. Thus, it is not a matter of the “final-state” that needs to satisfy certain termination/goal conditions, a criterion which is generally permissive of less robust execution through the rest of the episode, but rather the behavior of the policy throughout the episode needs to be precise and musical. 
>  ROBOPIANIST 任务的目标更加复杂，它实际上是一系列高度组合可变的目标状态序列，且可以由改变乐谱来拓展到任意难度
>  “成功” 基于对整个演奏的正确率定义，因此不是需要满足某些 “最终状态” 的中止/目标条件问题，这一基准通常允许整个回合可以执行地不够稳健；而是要求整个回合都需要精确且具有音乐性

Similarly, the literature on humanoid locomotion and more broadly, “character control", another important area of high-dimensional control, primarily features tasks involving the discovery of stable walking/running gaits [50, 51, 52], or the distillation of a finite set of whole-body movement priors [53, 54, 55], to use downstream for training a task-level policy. 

Task success is typically encoded via rewards for motion progress and/or reaching a terminal goal condition. It is well-documented that the endless pursuit of optimizing for these rewards can yield unrealistic yet “high-reward" behaviors. 

While works such as [53, 56] attempt to capture stylistic objectives via leveraging demonstration data, these reward functions are simply appended to the primary task objective. This scalarization of multiple objectives yields an arbitrarily subjective Pareto curve of optimal policies. In contrast, performing a piece of music entails both objectively measurable precision with regards to melodic and rhythmic accuracy, as well as a subjective measure of musicality. Mathematically, this translates as stylistic constraint satisfaction, paving the way for innovative algorithmic advances. 

>  大致意思就是说，任务 “成功” 的衡量应该复杂且实际比较好 (进而奖励函数的建模应该是多维的)，避免 agent 学习到的策略在数学上是优的，但却是不符合实际的

**Robotic Piano Playing** Robotic pianists have a rich history within the literature, with several works dedicated to the design of specialized hardware [23, 24, 25, 26, 27, 28], and/or customized controllers for playing back a song using pre-programmed commands (open-loop) [29, 30]. 

The work in [31] leverages a combination of inverse kinematics and trajectory stitching to play single keys and playback simple patterns and a song with a Shadow hand [37]. More recently, in [32], the author simulated robotic piano playing using offline motion planning with inverse kinematics for a 7-DoF robotic arm, along with an Iterative Closest Point-based heuristic for selecting fingering for a four-fingered Allegro hand. Each hand is simulated separately, and the audio results are combined post-hoc. Finally, in [33], the authors formulate piano playing as an RL problem for a single Allegro hand (four fingers) on a miniature piano, and additionally leverage tactile sensor feedback. However, the tasks considered are rather simplistic (e.g., play up to six successive notes, or three successive chords with only two simultaneous keys pressed for each chord). 

The ROBOPIANIST benchmark suite is designed to allow a general bi-manual controllable agent to emulate a pianist’s growing proficiency on the instrument by providing a curriculum of musical pieces, graded in difficulty. Leveraging two underactuated anthropomorphic hands as actuators provides a level of realism and exposes the challenge of mastering this suite of high-dimensional control problems. 

# B Detailed Reward Function 

![[pics/RoboPianist-Table2.png]]

# C Full Repertoire Results 
Fig9 ref to the original pdf

# D ROBOPIANIST Training details 
**Computing infrastructure and experiment running time** 
Our model-free RL codebase is implemented in JAX [57]. Experiments were performed on a Google Cloud n1-highmem-64 machine with an Intel Xeon E5-2696V3 Processor hardware with 32 cores $(2.3~\mathrm{GHz}$ base clock), 416 GB RAM and 4 Tesla K80 GPUs. 

Each “run”, i.e., the training and evaluation of a policy on one task with one seed, took an average of 5 hrs wall clock time. These run times are recorded while performing up to 8 runs in parallel. 
>  每次 "run" (使用单个随机种子，在单个任务上训练一个策略的总训练和验证时间) 需要 5h

**Network architecture** 
We use a regularized variant of clipped double Q-learning [58, 59], specifically DroQ [42], for the critic. Each Q-function is parameterized by a 3-layer multi-layer perceptron (MLP) with ReLU activations. Each linear layer is followed by dropout [60] with a rate of 0.01 and layer normalization [61]. The actor is implemented as a tanh-diagonal-Gaussian, and is also parameterized by a 3-layer MLP that outputs a mean and covariance. 
>  我们使用 clipped double Q-learning 的变体 DroQ
>  Q-function 使用三层的 MLP 建模 (ReLU 激活)，每层线性层后有 0.01 dropout 层和 layer norm 层
>  actor 被建模为 tanh 对角高斯分布，通过三层 MLP 参数化，网络输出均值和方差

Both actor and critic MLPs have hidden layers with 256 neurons and their weights are initialized with Xavier initialization [62], while their biases are initialized to zero. 

**Training and evaluation** 
We first collect 5000 seed observations with a uniform random policy, after which we sample actions using the RL policy. We then perform one gradient update every time we receive a new environment observation. We use the Adam [63] optimizer for neural network optimization. Evaluation happens in parallel in a background thread every 10000 steps. The latest policy checkpoint is rolled out by taking the mean of the output (i.e., no sampling). Since our environment is “fixed”, we perform only one rollout per evaluation. 
>  我们先用均匀随机策略收集 5000 个观测样本，之后用训练的 RL 策略收集样本
>  收到新的环境观测后，就会执行 Adam 梯度更新
>  每 10000 步执行一次评估
>  最后得到的策略不会执行采样，而是计算输出的平均值

**Reward formulation** 
The reward function for training the RL agent consists of three terms: 1) a key press term $r_{\mathrm{key}}$ , 2) a move finger to key term $r_{\mathrm{finger}}$ , and 3) an energy penalty term $r_{\text{energy}}$. 
>  奖励函数包含三项: 按键项 $r_{\text{key}}$, 手指到键的移动项 $r_{\text{finger}}$，能量惩罚项 $r_{\text{energy}}$

$r_{\mathrm{key}}$ encourages the policy to press the keys that need to be pressed and discourages it from pressing keys that shouldn’t be pressed. It is implemented as: 

$$
r_{\mathrm{key}}=0.5\cdot\left(\frac{1}{K}\sum_{i}^{K}g(||k_{s}^{i}-1||_{2})\right)+0.5\cdot(1-\mathbf{1}_{\{\mathsf{f a l s e p o s i t i v e}\}}),
$$ 
where $K$ is the number of keys that need to be pressed at the current timestep, $k_{s}$ is the normalized joint position of the key between 0 and 1, and $\mathbf 1_{\text{false positive}}$ is an indicator function that is 1 if any key that should not be pressed creates a sound.  $g$ is the tolerance function from the `dm_control` [52] library: it takes the L2 distance of $k_{s}$ and 1 and converts it into a bounded positive number between 0 and 1. We use the parameters `bounds=0.05` and `margin=0.5`

>  $r_{\text{key}}$ 中，$K$ 为当前时间步需要按下的按键数量，$k_s$ 为该按键归一化后的关节位置，在 0 到 1 之间
>  $g$ 是来自 `dm_control` [52] 库的容差函数：它将 L2 距离转换为介于 0 和 1 之间的有界正值

$r_{\mathrm{finger}}$ encourages the fingers that are active at the current timestep to move as close as possible to the keys they need to press. It is implemented as: 

$$
r_{\mathrm{finger}}=\frac{1}{K}\sum_{i}^{K}g(||p_{f}^{i}-p_{k}^{i}||_{2}),
$$ 
where $p_{f}$ is the Cartesian position of the finger and $p_{i}$ is the Cartesian position of a point centered at the surface of the key. $g$ for this reward is parameterized by `bounds=0.01` and ` margin=0.1 `

Finally, $r_{\text{energy}}$ penalizes high energy expenditure and is implemented as: 

$$
r_{\mathrm{energy}}=|\tau_{\mathrm{joints}}|^{\top}|\mathsf{v}_{\mathrm{joints}}|,
$$ 
where $\tau_{\mathrm{joints}}$ is a vector of joint torques and $\mathsf{v}_{\mathrm{joints}}$ is a vector of joint velocities. 

The final reward function sums up the aforementioned terms as follows: 

$$
r_{\mathrm{total}}=r_{\mathrm{key}}+r_{\mathrm{finger}}-0.005\cdot r_{\mathrm{energy}}
$$ 
**Other hyperparameters** 
For a comprehensive list of hyperparameters used for training the model-free RL policy, see Table 3. 

Table 3 ref to the original pdf

# E Multitask BC Results 
Fig10 ref to the original pdf

# F Baselines 
**Computing infrastructure and experiment running time** 
Our MPC codebase is implemented in $\mathrm{C}{+}{+}$ with MJPC [44]. Experiments were performed on a 2021 M1 Max Macbook Pro with 64 GB of RAM. 

**Algorithm** 
We use MPC with Predictive Sampling (PS) as the planner. PS is a derivative-free sampling-based algorithm that iteratively improves a nominal sequence of actions using random search. Concretely, $N$ candidates are created at every iteration by sampling from a Gaussian with the nominal as the mean and a fixed standard deviation $\sigma$ . The returns from the candidates are evaluated, after which the highest scoring candidate is set as the new nominal. The action sequences are represented with cubic splines to reduce the search space and smooth the trajectory. In our experiments, we used $N=10$ , $\sigma=0.05$ , and a spline dimension of 2. We plan over a horizon of 0.2 seconds, use a planning time step of 0.01 seconds and a physics time step of 0.005 seconds. 

**Cost formulation** 
The cost function for the MPC baseline consists of 2 terms: 1) a key press term $c_{\mathrm{key}}$ , 2) and a move finger to key term cfinger. 

The costs are implemented similarly to the model-free baseline, but don’t make use of the $g$ function, i.e., they solely consist in unbounded l2 distances. 

The total cost is thus: 

$$
c_{\mathrm{total}}=c_{\mathrm{key}}+c_{\mathrm{finger}}
$$ 
Note that we experimented with a control cost and an energy cost but they decreased performance so we disabled them. 

**Alternative baselines** 
We also tried the optimized derivative-based implementation of iLQG [64] also provided by [44], but this was not able to make substantial progress even at significantly slower than real-time speeds. iLQG is difficult to make real time because the action dimension is large and the algorithm theoretical complexity is $O(|A|^{3}\cdot H)$ . The piano task presents additional challenges due to the large number of contacts that are generated at every time step. This make computing derivatives for iLQG very expensive (particularly for our implementation which used finite-differencing to compute them). A possible solution would be to use analytical derivatives and differentiable collision detection. 

Besides online MPC, we could have used offline trajectory optimization to compute short reference trajectories for each finger press offline and then track these references online (in real time) using LQR. We note, however, that the (i) high dimensionality, (ii) complex sequence of goals adding many constraints, and (iii) overall temporal length (tens of seconds) of the trajectories pose challenges for this sort of approach. 
Figure 10: Detailed multi-task BC results. 