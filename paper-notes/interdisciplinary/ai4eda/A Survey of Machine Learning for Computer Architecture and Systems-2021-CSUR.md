# Abstract
很长一段时间，计算机架构和系统都为了机器学习模型的效率执行而优化，然而机器学习也可以转变计算机架构和系统的设计方式

本文详细回顾应用ML于计算机架构和系统设计的工作，
首先通过考虑ML技术在计算机架构/系统设计中的典型角色，
即：是为了快速的可预测的建模(fast predictive modeling)还是作为设计方法论(design methodology)，我们进行高层次的剖析分类

接着，我们总结计算机架构/系统设计中可以被ML技术解决的常见问题，以及为了解决这些问题采用的典型的ML技术
我们采取了数据中心可以被认为是一个仓库规模(warehouse-scale)的计算机的概念
# 1 Introduction
常规来说，计算机架构/系统设计是由人类专家基于直觉和启发执行的，一般基于启发的设计难以保证可拓展性(scalability)和最优性(optimality)
因此值得考虑利用ML执行自动化的、有效的计算机架构和系统设计
近年来，出现了应用ML技术于计算机架构和系统设计的迹象，这有着两重含义
1. 减少了人类专家的负担，以提升设计者的创造力
2. 正向反馈循环的闭环(the close of the positive feedback loop)，即为ML的计算机架构和系统和为计算机架构和系统的ML，形成促进两边提升的良性循环

现存的应用ML于计算机架构和系统设计的工作分为两类
1. 应用ML技术于快速和准确的系统建模(system modeling)
	包括了性能度量(performance metrics)和一些相关的标准(criteria)(例如功率消耗/power consumption，延迟/latency，吞吐率/throughput)
	系统设计的过程中，对系统行为进行快速准确的预测是必须的
	传统上，系统建模是通过循环-准确(cycle-accurate)和功能性虚拟平台(functional virtual platforms)，以及指令集模拟器(instruction set simulators)实现的，这些方法提供了准确的性能估计，但也带来了昂贵的计算开销，因此限制了大规模和复杂系统的可拓展性，同时，长的模拟时间占了设计过程的主导，而基于ML的方法则便于平衡开销和预测准确性
2. 应用ML技术作为设计方法论以直接加强计算机架构/系统设计
	ML技术善于提取特征(特征对于人类专家较难理解)，在没有显式编程的情况下做决策，以及积累经验，自动提升自己
	因此利用ML设计工具便于智能地探索设计空间(explore design space)，并且借助对工作负载(workloads)和系统之间的复杂的非线性关系的更好理解更好地管理资源
# 2 Different ML Techniques
ML的三大框架是：有监督学习，无监督学习，强化学习
## 2.1 Supervised Learning
有监督学习基于有标签的数据集，学习输入到输出的映射，以泛化到为未见过的输入做预测
- 回归即估计一个非独立变量和数个独立变量之间关系的过程，常见形式有线性/非线性回归，回归技术主要用于预测和因果关系推理(inference of causal relationships)
- SVM通过最大化边际(margins)以尝试找到能最好分离数据类的超平面，其一个变体为SVR(support vector regression)，用于执行回归任务，对新输入的预测或分类由它们相对于超平面的位置决定
- 决策树为具有代表性的逻辑学习方法(logical learning)，它用树结构建立回归或分类模型，最终得到的是带有决策结点和叶结点的数，每个决策结点表示一个特征，而该节点的分支表示特征的可能值，从根节点开始，输入实例顺序的经过结点和分支知道到达表示了分类结果或回归数值的叶节点
- ANN可以拟合大量函数，单层感知机常用于线性回归，复杂的DNN有多层，可以拟合非线性函数，DNN的变体利用不同的特定计算技巧，例如CNN利用卷积运算以利用空间特征(spatial)，RNN利用循环连接(recurrent connections)以从序列和历史中学习
- 集成学习使用多个模型，目标是达到比仅使用其中一个模型更好的效果，常见的种类有随机森林和梯度提升(gradient boosting)
不同的模型有对输入特征的不同偏好，SVM和ANN通常在特征为多维且连续是表现最好，而基于逻辑的系统则倾向于在处理离散/类别的特征表现更好

系统设计中，有监督学习通常用于性能建模(performance modeling)，布局预测(configuration prediction)，或从低层次的特征预测高层次的特征/行为
## 2.2 Unsupervised Learning
无监督学习是基于无标签的数据集找到之前位置的模式(perviously unkonwn patterns)的过程，两个流行的方法是聚类分析(clustering analysis)以及主成分分析(PCA)
- 聚类是基于对相似度的某种度量将数据对象聚集到不相交的簇的过程，聚类的目标在于合理地分类原始数据并且数据集中可能存在的结构和模式，最流行的聚类算法是k-means
- PCA实质上就是利用数据统计量的坐标轴转换(coordinate transformation)，PCA意图在于用少量的相互正交的轴保留原数据中的大部分变化性(variability)以降维
## 2.3 Reinforcement Learning
标准的RL中，一个智能体(agent)在一定的离散步数中和环境$\mathcal E$交互，每个时间步$t$，智能体从状态空间(state space)$\mathcal S$接受一个状态$s_t$，并根据它的策略/规则(policy)$\pi$从动作空间(action space)选择一个动作$a_t$，其中$\pi$就是一个从状态空间到动作空间的映射，作为回馈，智能体会接收下一个状态$s_{t+1}$和一个标量奖励(reward)$r_t:\mathcal S\times \mathcal A\rightarrow \mathbb R$，该过程一直持续到智能体达到一个终端状态(terminal state)，然后过程重新开始，返回量$R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$是时间步$t$时的奖励乘上一个折扣因子(discount factor)$\gamma \in (0,1]$的累计，智能体的目标在于最大化每个状态$s$的期望返回量

状态-动作(state-action)值$Q_{\pi}(s,a)= \mathbb E_{\pi}[R_t|s_t = s,a_t = a]$就是在策略$\pi$下，选定了状态$s$和动作$a$的期望返回值，类似地，状态(state)值$V_{\pi}(s) = \mathbb E_{\pi}[R_t|s_t = s]$即为在策略$\pi$下，选定了状态$s$的期望返回值

通常RL中有两个类型的方法：基于值(value-based)和基于策略(policy-based)
- 基于值的RL中，状态-动作函数$Q_{\pi}(s,a)$一般用表格方法(tabular approaches)或函数近似(function approximations)来近似
# 3 ML for Fast System Modeling
大多数现存的工作应用有监督学习以进行纯系统建模(pure system modeling)或高效设计空间探索(efficient design space exploration/DSE)
## 3.1 Sub-system Modeling and Performance Prediction
### 3.1.1 Memory Systems
存储系统中，基于ML的性能模型(performance model)被利用于帮助探索不同目标之间的权衡(trade-offs)

为了探索基于非易失性存储(non-volatile memoery/NVM)的高速缓存结构，[45]开发了ANN模型以从低层次特征(low-level features)，例如缓存关联性、容量、延迟(cache associativity, capacity and latency)预测高层次特征(high-level features)例如缓存读写的未命中(miss of cache read/write)和每周期的指令(instruction-per-cycle/IPC)

为了适应性地为不同的应用选取NVMs架构上的技术(architectural techniques)，Memory Cocktail Therapy[48]通过采取了梯度提升(gradient boosting)和带lasso的二次回归技术的轻量化的在线预测器(online predictor)估计生命周期(lifetime)、IPC和能量消耗(energy consumption)

为了优化吞吐量处理器(throughput processors)中的内存控制器布置(memeory controller placements)，[133]建立了CNN模型，接收内存控制器布置作为输入，预测吞吐量，这将优化过程加速了两个数量级

一些研究聚焦于内存访问模式的高效表示(efficient representations of memory acess patterns)，Block2Vec[42]通过训练DNN以学习到每个数据块的最优的向量表示，然后通过向量距离获取数据块相似度，以挖掘数据块相关性(data block correlations)，帮助了对缓存和数据预取(caching and data prefetching)的进一步优化
[209]使用GNN学习静态代码和其动态执行的融合表示(fused representation of static code and its dynamic execution)，这种联合的表示可以建模数据流(data flows)(例如数据预取/prefetching)和控制流(control flow)(例如分支预测/branch prediction)
### 3.1.2 Network-on-Chip(NoC)
NoC中，我们感兴趣的性能度量(performance metric)有延迟(latency)，能耗(energy consumption)和可靠性(reliability)

1 考虑延迟预测
[185]使用SVR模型预测基于网格(mesh-based)的NoC上的流量延迟(traffic flow latency)和平均信道等待时间(average channel waiting time)，这松弛(relax)了一些传统排队论(classical queuing theory)的一些假设(assumption)
除了显式预测延迟，一个轻量的基于硬件的ANN[212]预测交通热点(traffic hotspots)的存在，交通热点密集的交通堵塞(intensive traffic congestions)，会显著降低有效吞吐量，且隐式地暗示了NoC上的平均通讯(communication)延迟
该模型的输入特征是邻近的NoC路由器(routers)的缓存利用率(buffer utilization rate)，训练过后的预测器与主动交通热点防御算法(proactive hotspot-preventive routing algorithm)结合，以避免热点的形成

2 考虑估计能耗
学习到的预测器常用于节约NoC的动态/静态的能量，[51]在每个路由器上训练决策树，以预测链路利用率(link utilization)和交通方向(traffic direction)，预测结果和睡眠性链路存储单元(sleepy link storage units)结合，用于电源栅极/路由器(power-gate/routers)以及用于改变链路方向
[35]使用岭回归模型，预测缓存利用率，缓存利用率的改变(changes)，或一个结合了能耗和吞吐量的度量，基于该模型，路由器可以选择合适的电压/频率(which a router can select proper voltage/frequency)，
在光激性(photonic)NoCs，岭回归模型也应用于预测下一个时间窗口(time window)中，会进入每个路由器的数据包(packet)的数量[224]，基于该模型，可以知道对波长的合适的放缩倍数，以减少光子链路(pohtonic link)的静态能耗

3 考虑NoCs的可靠性(reliability)
[50]为每个链路训练了决策树以预测链路运行时的时序配合故障(timing fault)，基于该模型，通过使用强化的循环冗余校验及纠错码和松弛传输(relaxed transmission)，开发了一个主动的容错技术(proactive fault-tolerant)
## 3.2 System Modeling and Performance Prediction
正确且快速的系统性能估计对系统优化和设计是必须的
### 3.2.1 Graphic Processing Unit(GPU)
对于GPU建模，有两类预测：跨平台的预测(cross-platform predictions)和针对GPU的预测(GPU-specific predictions)
跨平台预测用于预先决定是否要将一个应用从CPU卸载至GPU，因为不是所有应用都可以从GPU执行中受益，而且移植过程(porting process)也有额外的开销
针对GPU的预测则用于估计关于帮助GPU设计空间探索的感兴趣的度量，以帮助解决设计空间的不规则性(design space irregularities)，和布置之间复杂的交互(complicated initeractions among configurations)

跨平台预测可以建模为二元分类问题，以识别潜在的GPU加速是否大于一定阈值，可以以动态指令配置文件(dynamic instruction profiles)作为输入，使用最近邻搜索或SVM[14]，或者以对源CPU代码的静态分析(即内存聚合/memeory coalescing，分支分歧/branch divergence， 内核大小可用并行性/kernel size available parallelism，指令密集度/instruction intensities)作为输入，训练随机森林模型[10]
[9]则利用了CPU单线程代码的静态和动态程序性质，训练了一个由100个基于回归的学习器，以预测GPU执行时间

针对GPU的预测中，
[90]接收GPU布局(configurations)和性能计数器(performance counters)作为输入特征，采用逐步线性回归(stepwise linear regression)模型预测GPU执行时间，该模型从许多的GPU参数中选出最重要的输入特征，因此在处理稀疏样本时也有高准确度；
[99]用NN预测器的集成以建模功率/吞吐量(power/throughput)；
[175]利用前代GPU的配置结果(profiling results from earlier-generation GPU)，训练一个集成了线性和非线性回归模型的模型，为未来代(later/future-generation)的GPU预测执行时间，该模型的速度1000倍快于循环-准确GPU模拟器(cycle-accurate GPU simulaors)；
[181]聚焦于内存中处理(processing-in-memory/PIM)辅助(assited)的GPU架构，将GPU核心分成两类：强力的GPU核但远离内存，辅助/简单的GPU核但靠近内存，然后训练逻辑回归模型，该模型以计算核的特点(kernel characteristics)作为输入特征，预测架构的核亲和性(architecture affinity of kernels)，目的在于正确辨别哪个计算核可以从PIM中获益，然后将它卸载到相应的辅助GPU核心，作者也建立了线性回归模型以预测每个核心的执行时间，基于这两个模型和计算核依赖信息(kernel dependency information)，可以建立一个并发核管理机制(concurrent kernel management mechanism)；
[239]聚焦于通用目的GPUs(GPGPUs)，对计算核相对于计算单元数量，引擎频率，内存频率的放缩行为(kernel scaling behaviour)进行建模，在训练中，有相似性能放缩行为(similar performance scaling behabiour)的核通过K-means聚类，当遇到新的核，会通过一个ANN分类器，将其分类到能最好描述它的放缩行为的簇里; 
[130]重新评估了对GPGPU的交通模式(traffic pattern)的推论(assumptions)，然后将CNN和一个服从t分布的随机邻近嵌入(t-distributed stochastic neighbor embedding)结合，以预测不同的交通模式
### 3.2.2 Single-Core Processor
在对单核处理器的性能预测建模中，早期的工作主要的目标是超标量处理器(superscalar)，[100]提出一个迭代式过程以用26个关键的微架构参数(key micro-architecture parameters)建立线性回归模型，以预测超线程处理器针对应用(application-specific)的每指令周期数(cycle-per-instruction/CPI)，他们在[101]中，使用9个关键的微架构参数，通过非线性回归构建预测模型(即由回归树生成的径向基函数网络/radial basis function network generated from regression trees)；[121]则使用三次样条回归(cubic splines)以预测针对应用的性能(每秒钟指令数量)和功率

之后的工作则聚焦于使用微架构参数和性能计数器对现存硬件(例如Intel，AMD和ARM处理器)的性能建模(performance modeling)，[57]构建了一个机械经验主义(mechanistic-empirical)的模型以预测三个Intel处理器的CPI，灵感来源于机械性建模(mechanistic modeling)，即模型内的未知参数通过回归得到，模型受益于机械性建模(即可解释性)和经验性建模(即容易实施/ease of implementation)；[268, 269]探索了对程序执行时间的跨平台预测的两种方法，其中Intel Core i7和AMD Phenom处理器的资料结果(profiling result)被用于预测目标ARM处理器的执行时间，[269]将特征空间的全局线性假设松弛为局部线性，然后应用约束下的局部稀疏线性回归(constrained locally sparse linear regression)，[268]则使用阶段级(phase-level)的性能特征，应用lasso回归模型
### 3.2.3 General Modeling and Performance Predicsion
回归是主流的使用微架构参数或其他特征来预测性能度量(performance metrics)的技术
对于常规的基于回归的模型，ANNs和非线性回归是常见的方式，以用于预测吞吐量(throughput)/延迟(latency)[83,110,122,124,244]，功率/能量[110,122]
也存在不同方法之间的比较，[123]比较了分段多项式回归(piecewise polynomial regression)和ANNs，分段多项式回归具有更好的可解释性，ANNs具有更好的泛化性，[177]将数个线性回归模型与不同的ANNs进行了比较，认为剪枝(pruned)的ANNs在要求更长训练时间的同时可以达到最好的准确率，[3]估计了多线程程序在目标硬件平台上的并行执行加速，发现高斯过程回归在该例中相较其他方法表现最好

越来越多最近的工作倾向于利用数据驱动(data-driven)的方法，
Ithemal[157]使用LSTM，利用了层次化的多规模(hierarchical multi-scale)RNN以预测基础块(basic block)(即没有分支或跳跃的指令序列/sequences of instructions with no branches and jumps)的吞吐量，评估结果显示Ithemal相较于解析式吞吐量估计器(analytical throughput estimators)更加准确，且速度一样快；
DiffTune[191]利用了Ithemal的变体，作为一个可微的替代(differentiable surrogate)以近似CPU模拟器，因为采用了可微函数，因此可以采用基于梯度的优化方法，以学习可以最小化x86基础块CPU模拟器误差的参数，学习到的参数最后会插回(plug back)原来的(original)模拟器；
[49]提供了一些关于基于学习的建模方法(learning-based modeling methods)的见解: 预测准确率的提升可能存在收益递减(diminishing returns)，多考虑领域知识会对系统优化有帮助，即使整体的准确率可能不会提升，因此，他们提出了生成式模型，通过生成更多的训练数据，以解决数据稀缺问题，然后应用了一个多阶段采样，以提升对最优配置点(optimal configuration point)的预测准确率

基于机器学习的预测型性能模型帮助我们进行高效的资源管理和快速的设计空间探索以提高吞吐量，
基于ANNs对IPC进行预测，[19]的资源分配方法[172]的任务调度方法可以选择能带来最优预测IPC的决策；
ESP[164]构建了带弹性网正则(elastic-net regularization)的回归模型，以预测应用干扰(application interference)(例如减速/slowdown)，该模型与调度器结合，以提升吞吐量；
MetaTune[198]是一个为卷积运算设计的基于的元学习(meta-learning)的开销(cost)模型，和搜索算法结合时，可以使得在编译时就可以进行高效的参数自动调节(auto-tuning of parameters)；
考虑非计算内核(the uncore)(即内存层次结构/memory hierarchies和NoCs)的快速设计空间探索，[200]使用带有受限制(restricted)四次样条的基于回归的模型以估计CPI，减少了四个数量级的设计空间探索时间

基于机器学习的预测型性能模型有利于性能和功率预算之间的调整(adaptations between performance and power budgets)，
[41]利用离线(off-line)多变量线性回归以预测IPC和/或不同架构布置(architecture configurations)的功率，通过动态并发截流(dynamic concurrency throttling)和动态电压和频率放缩(dynamic voltage and frequency scaling/DVFS)最大化了OpenMP程序的性能；
[12]应用硬件频率限制(frequency-limiting)技术以选择在给定功率约束下最优的硬件布置；为了有效针对不同优化目标应用DVFS，[102]提出可以采用受限制的多项式模型(constrained-posynominal)预测功耗，基于此设计策略；[140]则使用线性回归模型预测工作执行时间，基于此设计策略；
此外，进行智能电源管理(smart power management)则是一个更广泛的课题，LEO[165]采用了层次化的贝叶斯模型以预测性能和功率，将该模型与运行时能量优化(runtime energy optimization)结合，就可以找到性能-功耗的帕累托前沿(performance-power pareto fontier/帕累托前沿即最优解的集合)，并选择满足性能约束下，拥有最小能量的布局(configuration)；
CALOREE[163]进一步将电源管理任务拆分为两个抽象：一个用于性能预测的学习器和一个利用学习器的预测的适应性控制器
### 3.2.4 Data Center Performance Modeling and Prediction
数据中心主要用于传统的企业应用和云服务，许多工作采用ML技术以预测工作负载/资源相关的度量(workload/resource-related metrics)，以使能弹性资源供应(elastic resource provision)

常见的例子包括[253]使用SVR以预测工作完成时间(job completion time)；[196]利用自回归移动均值模型(autoregressive moving average/ARMA)预测即将到来的工作负载；[27]则利用自回归的集成移动均值模型(autoregressive integrated moving average/ARIMA)预测即将到来的工作负载；[109]利用隐式马尔科夫模型以描述工作负载模式中的变化(variations in workload patterns)；[70]使用轻量的统计学习算法估计工作负载的动态资源请求(dynamic resource demand of workloads)，[85]则使用MLP；[65]也使用MLP以预测数据中心的电能使用效率(power usage effectiveness)，该模型在Google数据中心已经被广泛地测试和验证过；[38]预测虚拟机行为(包括虚拟机声明周期/lifetimes，最大部署大小/maximumu deployment sizes，工作负载类别/workload classes)，目标在于健康/资源管理(health/resource management)和电力限制(power capping)，其中采用的ML模型包括随机森林和极限梯度提升树

除了工作负载/资源相关的度量，数据中心或云服务的可用性(availability)也需要关心，而其中最重要的任务就是预先预测磁盘故障(disk failure)，
根据SMART(Self-Monitoring, Analysis and Reporting Technology)性质(attributes)，借助ML来建立磁盘故障预测模型，[75]采用贝叶斯方法；[170]采用无监督聚类；[271]采用SVM和MLP；[126]采用分类与回归树(classification and regression trees/CART)；[248]采用RNN；[127]采用梯度提升回归树，上述提到的方法都依赖于离线训练，而在线随机森林[246]则可以在运行中随着新输入的数据变化，通过生成新的树学习新的信息，通过丢弃过时的树忘记旧的信息，结果上避免了磁盘故障预测问题中模型老化(aging)的问题
为了预测部分驱动故障(partial drive failure)，即磁盘故障或扇区故障，[146]探究了五种ML方法(CART，RF，SVM，NN和逻辑回归)，其中RF的表现一致地(consistently)高于其他方法；[249]结合了SMART性质和系统级别信号(system-level signals)以训练梯度提升树，这是一个在线预测模型，会根据磁盘在不久后(near future)的易错性(error-proneness)程度对磁盘进行排序
## 3.3 Performance Modeling in Chip Design and Design Automation
### 3.3.1 Analog Circuit Analysis
模拟电路设计通常是一个人工的过程，在布局前阶段(pre-layout phase)和布局后阶段(post-layout phase)之间需要多个试错(trial-and-error)迭代(iterations)
近年来，示意图上的(schematic)性能估计(即布局前)和布局后的模拟结果之间的差异逐渐增大，另一方面，源自示意图的解析式性能估计(analytical performance estimations)将随着设备放缩(device scaling)而不再正确，另一方面，虽然布局后模拟可以提供高准确率的估计，但它们是及其耗时的，并且已经是设计迭代时间(design iteration time)主要的瓶颈，因此ML被广泛用于快速的电路评估，以填补对ICs的性能预测

我们基于输入特征是从布局前信息提取还是从布局后信息提取对之前的工作进行分类讨论
首先是给定设计示意图(design schematics)时，我们可以在布局前阶段预测布局寄生效应(parasitics in layouts)，这可以帮助减小布局前和布局后模拟的差距，ParaGraph[190]构建GNN模型以预测依赖于布局的(layout-dependent)寄生效应(parasitics)和物理设备参数(physical device parameters)；MLParest[210]表明不基于图的方法(例如RF)在预测互联寄生效应(interconnect parasitics)时也十分有效，但是位置信息(placement information)的缺乏也可能导致预测中有较大的方差(variations)
其次就是给定电路示意图(circuit schematics)同时给定设备信息(device information)作为输入时，我们可以直接从布局前设计建模布局后性能，[5]提出了一个结合了贝叶斯共同学习框架(Bayesian co-learning framwork)和半监督学习的模型以预测能耗，整个电路的设计图被分划为多个块以建立块级别(block-level)的性能模型，基于块级别的模型上建立电路级别(circuit-level)的性能模型，通过将这两种低维度(low-dimensional)模型与大量的无标签的数据结合，假样本(pesudo samples)就可以在几乎没有开销的情况下被标记，最后，借助假样本和少量的有标签的数据，可以训练高维度的性能模型以将低级特征(low-level features)映射到电路级别的度量，该工作说明了在缺乏有标签样本的情况下建模性能模型也是可能的；一些基于贝叶斯方法的变体也在估计布局后性能中有好的表现，例如，[179]将SVM与贝叶斯回归结合以预测电路性能；[74]使用贝叶斯DNNs以比较电路设计
第三就是使用SPICE-like的模拟器进行布局后模拟(post-layout simulations)是十分耗时的，ML技术就可以用于快速估计布局设计性能[128]，为了更好地利用布局内部的结构化信息，[138]用3D图像表示中间的布局放置结果(intermediate layout placement results)，作为3D CNN模型的输入；[129]则将其用图编码(encoded as graphs)以训练一个自定义的GNN模型，以预测是否满足了某个设计规范(design specifications)
### 3.3.2 High-Level Synthesis(HLS)
高层次综合是一个从行为语言(behavioral languages)(例如C/C++/SystemC)到寄存器传输层次(register-transfer level)设计的自动化转换，这极大促进了包括FPGA(field-programmable gate arrays)或专用集成电路(applicaition-specific integrated circuits/ASICs)的硬件的发展
而HLS工具常常需要大量时间来综合(synthesize)每个设计，妨碍了设计者对设计空间的探索，因此考虑采用ML进行快速和准确的性能估计

在对HLS设计的性能估计中，对ML模型的输入特征主要从三个地方提取：HLS Directives，来自HLS front-ends的IRs，以及HLS reports
关于将脚本中的HLS指令作为输入特征的方法，[137]采用随机森林模型，预测各种设计度量，包括面积和有效延迟(area and effective latency)，[159]则预测了吞吐量和逻辑利用率(throughput and logic utilization)；[119]采用迁移学习方法以在不同的应用和综合选项(synthesis option)中迁移知识
关于以HLS前端生成的IR图作为输入特征的方法，[112]使用预特征化的区域模型(pre-characterized area models)对每个图中结点的资源需求进行计数，然后将其作为对ANNs的输入，以预测LUT路由使用(LUT routing usage)，寄存器重复(register duplication)和不可用(unavailable)的LUT；[223]利用GNNs以自动预测IR图中的算数操作到FPGAs上的不同资源的映射；为了预测实现后路由拥塞(post-implementation routing congestion)，[265]构建了一个将RTL实现后的路由拥塞度量和IRs中的操作联系起来的数据集，以训练可以定位源代码中产生高度拥塞区域的部分的ML模型
关于直接从HLS报告提取出信息用于输入特征的模型，[43]尝试了若干个ML模型(包括线性回归，ANN，梯度提升树)以预测实现后的资源利用率和时机(timing)；[148]应用集成学习，以准确估计吞吐量或区域吞吐量率(throughput-to-area ratio)；[136]采用了HLS报告和IR图作为特征，以预测功率

和FPGA/AISCs和CPU的异质性的平台的出现促进了硬件/软件协同设计(co-design)，进而促进了跨平台性能预测，[176]使用RF，基于从CPU执行中获得的程序计数器度量，预测FPGA循环计数(cycle conuts)和功耗，该方法在训练和测试是目标是同一个FPGA平台；[149]考虑不同的FPGA平台，使用ANNs以预测基于ARM处理器的目标FPGA上的应用的加速
### 3.3.3 Logic and Physical Synthesis
数字设计中，逻辑综合(logic synthesis)将RTL设计转化为优化的门电路级别表示(gate-level representations)，然后物理综合将这些设计网表(netlists)转化为物理布局，这两个阶段也需要大量时间才能生成最终的比特流/布局，因此考虑采用ML方法进行快速性能估计
逻辑综合中，[256]采用CNN模型，[258]采用基于LSTM的模型预测对特定设计采用不同的综合流(synthesis flow)的延迟和区域(delay and area)，输入就是以矩阵形式或时间序列表示的综合流
物理综合中，布线(routing)是一个受严格限制的复杂问题，EDA工具一般采用两步方法：全局布线(global routing/GR)和详细布线(detailed routing/DR)，GR工具粗粒度分配布线资源，提供布线计划以引导DR工具完成全部布线
通常，在GR后或在GR中就可以发现布线拥塞，而一个设计的可布线性(routability)则在DR后和设计规则检查(design rule checking/DRC)后确定
有许多工作研究了从较早的布局阶段就预测可布线性，以避免过度在布局和布线(placement and routing)之间往复的迭代

ASICs中，一些工作探究了通过估计设计规则违反(design rule violations/DRV)的数量以预测可布线性
[125]以GR的结果为输入，使用了数个ML模型，包括线性回归，ANN，决策树以预测DRVs的数量，final hold slack，功率，以及区域；[184]使用布局数据(placement data)和来自GR的拥塞地图(congestion maps from GR)作为输入特征，使用了非参数化的回归技术：多变量适应性回归样条(multivariate adaptive regression splines/MARS)，以预测布线资源的利用率(utilization of routing resource)和DRVs的数量；[28]只使用了布局信息，使用MARS和SVM以预测可布线性，[216]则使用MLP以预测DR的短期违反(short violations)；[247]用图像表示布局信息，用GR信息作为输入，使用全卷积网络(fully convolutional network/FCNs)以预测DRC热点(hotspots)的位置(locations)，[29]则使用布局数据，将预测工作建构成逐像素的二元分类任务，以预测GR拥塞地图；J-Net[131]是一个自定义的FCN模型，接受高像素的引脚图案(pin pattern)和低像素的布局信息作为输入特征，输出一个2D数组，表明每一项对应的块(tile)是否是一个DRC热点

FPGA中，[145]使用来自针脚计数和SLICEs的每区域线长(wirelength per area of SLICEs)的特征向量，用线性回归估计布线拥塞地图；[6,257]将布线拥塞预测问题构建成一个图像翻译问题(image translation problem)，使用条件GAN(conditional)，接受布局后图像作为输入，预测拥塞热图(heap maps)
# 4 ML as Design Methodology
本节介绍直接应用ML技术作为计算机架构/系统的设计方法论的工作
计算机架构和系统日渐复杂，也需要更多的人力进行设计和优化，因此，有人认为计算机架构和系统应该有设计和配置它们自己的能力(design and configure themselves)，根据工作负载的需要和用户指定的约束调整自己行为的能力，诊断故障的能力(diagnose failures)，从检测到的故障修复自己的能力等
许多在架构/系统设计的问题可以建构为组合优化问题(combinatorial optimization problem)或顺序决策制定问题(sequential desicion-making problem)，因此较为适合采用RL
## 4.1 Memory System Design
“内存墙(memory)”一直是冯诺依曼架构中的性能瓶颈，冯诺依曼架构中，计算一般比内存访问快数个数量级，为了解决这个问题，出现了层次化存储系统，也有对不同层次(level)的内存系统的优化
如今出现了基于ML的技术已设计更加智慧的内存系统
### 4.1.1 Cache
CPU和内存系统之间存在显著的延迟和带宽差异，因此一些研究聚焦于高效的缓存管理(cache management)，针对缓存优化的工作主要有两种：提高缓存置换策略(cache replacement policies)和设计智能的预取策略(prefetching policies)

关于缓存置换策略的工作
[96, 219]使用感知机学习以预测是否略过或重用最后一级缓存(last-level cache/LLC)中的参考程序块(referenced block)；[16]将缓存置换问题建模为一个马尔可夫决策过程，根据缓存行的预期命中数和平均命中数的差异决定是否替换缓存行；[207]训练了一个基于注意力的离线LSTM模型，从历史程序计数器(history program counters)中提取信息，然后将其用于建立一个在线的基于SVM的硬件预测器以服务缓存置换策略

关于智能预取策略的工作
[230]使用常规的基于表的预取器(table-based prefetcher)以提供预取建议(prefetching suggestions)，并且使用基于空间-温度局部性(spatio-temporal locality)训练的感知机以拒绝不必要的预取决定，改善了缓存污染问题(cache pollution problem)；类似地，[17]将基于感知机的预取过滤器(fliter)和常规的预期器结合，提高了预取的覆盖率(coverage)的同时也没有降低预取的正确率(accuracy)；[183]没有利用常用的空间-温度局部性，而是提出了基于上下文的内存预取器(context-based memory prefetcher)，利用了隐含了程序语义和数据结构内在的访问相关性的语义局部性(semantic locality that characterizes access correlations inherent to program semantics and data structures)，语义局部性由RL中的上下文盗贼模型(contextual bandit model)近似(approximate)；解释内存访问模式中的语义(semantics in memory access patterns)类似于NLP中的序列分析(sequence analysis)，因此一些工作使用基于LSTM的模型，视预取为回归问题[261]或分类问题[71]；但即便基于LSTM的模型有更好的表现，尤其在长的访问序列和噪声踪迹中(long access sequence and noise traces)，基于LSTM的模型也存在热身过长以及预测延迟的问题(long warm-up and prediction latency)，以及巨大的存储开销，[24]讨论了超参数是如何影响基于LSTM的预取器的性能，强调回顾大小(lookback size)(即内存访问历史窗口/memeory access history window)和LSTM模型大小会强烈影响预取器在不同级别的噪声水平或工作负载模式下的学习能力(learning ability)；为了容纳大的内存空间，[208]提出了层次化的序列模型，基于两个分离的基于注意力的LSTM层，以解耦(decouple)对页和偏移的预测(predictions of pages and offsets)，然而其相应的硬件实现对真正的处理器是不实际的
### 4.1.2 Memory Controller
智能的内存控制器可以显著提高内存带宽利用率(memory bandwidth utilization)
[84, 153]认为对于一个可以自我优化的，对工作负载改变具有适应性的内存控制器，可以建模为RL智能体，RL智能体总是选择具有最高的期望长期性能效益(即Q-values)(highest expected long-term performance benefits)的合法的DRAM命令(legal DRAM commands)；[169]在两大主要方面提升了该内存控制器，首先，不同动作(即合法的DRAM命令)的奖励(rewards of different actions)会根据遗传算法自动校准(calibrated)，以服务于不同的目标函数(例如能量，吞吐量等)，其次，采用了一个考虑了第一阶属性交互(first-order attribute interactions)的多因子方法(multi-factor method)以选择用于状态表示(state representations)的恰当的属性
由于这两项工作都使用了基于表的Q-learning且选择了受限的属性以表示状态，故应该关注其方法的可放缩性，且可以用更多富有信息的表示(informative representations)进一步提升性能
### 4.1.3 Others
一系列工作目标于内存系统的不同方面，[152]通过学习索引结构(index structure)加速了虚拟地址翻译(virtual address translations)，对所有测试虚拟地址的正确率接近100%，但该方法的推理时间长(long inference time)，实际的硬件实现留待之后的工作；[235]利用不同比特的非对称的传输开销(asymmetric transmission costs of different costs of different bits)，减少了互联网络中的数据移动能量(data movement energy in interconnects)，其中要被传输的数据块动态地被K-majority聚类算法聚集，以推演出节能的传输表达式(derive energy-efficient expressions for transmission)；针对NAND flash的垃圾收集机制中，[104]提出了基于RL的方法以减少长尾延迟(long-tail latency)，关键思想在于利用请求间隔(inter-request interval)(idle time)以动态决定要拷贝的页数或是否要执行擦除操作(erase operation)，其中决策由基于表的Q-learning制定，后续的工作[105]考虑了更细粒度的状态，并推出了Q-table缓存以在大量状态中管理关键状态
## 4.2 Branch Prediction
分支预测器(branch predictor)是现代处理器的支柱(mainstays)之一，它显著地提升了指令级别的并行(instruction-level parallelism)
随着流水线逐渐加深，误预测的代价也在增加(as piplines gradually deepen, the penalty of misprediction increases)

传统的分支预测器常常考虑有限的历史长度(limited history length)，而这可能会损害预测准确率，相比较下，基于感知机/MLP的预测器可以在合理的硬件预算下(with reasonable hardware budgets)，可以处理长历史，表现优于目前最好的不基于ML的预测器

[26]基于MLP，使用程序语料库(program corpus)和控制流图(control flow graphs)中的静态特征(static features)训练了一个静态(static)的分支预测器，以在编译时预测一个分支的方向；[95]则基于感知机训练了动态(dynamic)的分支预测器，它将分支地址进行哈希，以选择合适的感知机，然后计算相应的内积以决定是否选择该分支，该方法在线性可分的分支中效果优异，[91]通过应用前置流水线(ahead piplining)并基于路径历史(path history)选择分支，进一步提升了该方法的准确率，降低了延迟；为了在非线性可分的分支中也有高的准确率，[92]将基于感知机的预测推广为分段线性分支预测(piecewise linear branch prediction)；除了路径历史，[94]还利用了来自不同组织的分支历史记录(different organizations of branch histories)的多种类型的特征以提高总体的表现

考虑到分支预测器实际的硬件实现，SNAP[213]使用电流控制的数模转换器(current-steering digital-to-analog converters)将数字权重转化为模拟电流，将昂贵的数字点积计算替换为电流求和(summation)；它的优化版本采用了一些新的技术，例如使用了全局和每个分支的历史，可训练的缩放系数(scaling coefficients)，动态训练阈值(dynamic training threshold)等

[67]没有对是否采用某个分支进行二元决策，而是通过基于感知机的预测器，在位级别(bit level)直接预测一个间接分支的目标地址(the target address of an indirect branch)；[218]注意到目前基于感知机/MLP的预测器都取得了高的准确率，但有少量的静态分支指令会被系统性地误预测(a small amount of static branch instructions are systematically mispredicted)，作者将其称为hard-to-predict branches(H2Ps)，因此作者提出一个CNN辅助预测器(help predictor)对历史分支进行匹配，最终提高了条件分支中H2Ps的准确率
## 4.3 NoC Design
随着每个芯片上的核心数量的增加，NoC逐渐开始发挥着至关重要的作用，因为
它负责核心之间的通信(inter-core communication)以及核心和存储层次之间的数据移动(data movement between cores and memory hierarchies)

以下是几个值得关注的问题
第一，通信能量的规模化要慢于计算能量(communication energy scales slower than computation energy)[22]，说明有必要提高NoCs的效能(power efficiency)
第二，布线和交通流量控制的复杂性随着每个芯片上的核心数量的增加而增加，而工作负载的不断增加和其不规则性甚至加剧了这一问题
第三，随着晶体管不断地变小，NoCs变得更容易被不同类型的错误影响，因此可靠性成为关键问题
第四，一些非传统的NoC架构可能很有前景，而它们通常具有较大的设计空间以及复杂的设计约束，使得它们几乎不可能手动优化
在上述所有领域，基于ML的设计技术(design techniques)展示了它们的力量和魅力
### 4.3.1 Link Management and DVFS
功耗是NoCs的一个关键问题，而NoC中链路(links)通常消耗相当一部分网络功率
根据链路利用率的静态阈值(static threhold of link utilization)以决定打开/关闭链路时是降低功耗的一种微不足道的方法，它不能适应动态变化的工作负载

[201]使用多个ANN进行动态的链路管理，每个ANN负责NoC的一个区域，在给定每个区域的链路利用率的情况下，动态为每个时间间隔(time interval)计算一个阈值，根据阈值决定打开/关闭链路，该方法在低硬件开销(low hardware overheads)的情况下显著节省了能量，但会在路由(routing)中导致长延迟；为了满足一定的功率和热预算(power and thermal budgets)，[192]使用层次化的ANN以预测最优的NoC配置(configurations)(即链路带宽/link bandwidth、节点电压/node voltage、对节点的任务分配/task assignment to nodes)，其中局部ANN预测局部最优能耗，全局ANN利用局部ANN的预测，预测全局最优的NoC配置；为了节省动态功率，几项研究[60, 266]采用基于每个路由器的Q-learning智能体(per-router based Q-learning agents)，即一个离线训练的神经网络以为每个路由器选择最优的电压/频率水平(voltage/frequency levels)
### 4.3.2 Routing and Traffic Control
随着工作负载和它们的交通模式的多样性和不规则性的增加，基于学习(learning-based)的路由算法和交通控制方法展现了很好的效果
(1) 因为路由问题可以公式化为(formulated as)顺序的决策过程(sequential decision-making process)，一些研究应用了基于Q-learning的方法，即Q-routing算法[23]，它使用了传输时间(delivery time)的局部估计以最小化总的数据包传输时间(total packet delivery time)，能够处理不规则的网络拓扑，能够处理不规则的网络拓扑，并比传统的最短路径路由(shortest path routing)保持更高的网络负载(network load)；Q-routing被延伸到了多种场景，例如[118]将Q-routing与对偶(dual)RL结合，提高了学习速度和路由表现；[147]在动态NoCs(即网络结构/拓扑在运行时会动态变化)中求解数据包路由(packet routing)；[59]使用可重构的容错Q-routing(reconfigurable fault-tolerant Q-routing)处理无缓存(bufferless)NoCs中的不规则错误；[54]使用拥塞感知的非最小化Q-routing(congestion-aware non-minial Q-routing)，增强了在拥塞区域周围重路由(reroute)信息的能力
除了路由问题外，深度Q-网络在NoC仲裁策略(NoC arbitration policies)也有应用，[254, 255]中，智能体/仲裁器(arbiter)将特定输出端口(output port)授予具有最大Q值的输入缓冲(input buffer)，Q-网络在延迟和吞吐量方面有所提升，但由于深度网络的复杂性，直接的硬件实现还不现实，因此需要提炼出一个相对简单的电路实现
(2) 交通控制算法的目标是控制NoCs的拥塞，SPECTER NoC架构[46]，一种具有单周期多跳遍历(single-cycle multi-hop traversals)和自学习节流机制(self-learning throttling)的无缓存(bufferless)NoC，通过Q-learning控制向网络中的新的flits的注入(injection)，网络中的每个结点都根据它们的Q值独立选择是否提升，降低或保持它们的节流率，这显著提高了带宽分配的公平性(bandwidth allocation fairness)和网络吞吐量(network throughput)；[229]设计了一个基于ANN的准入控制器(admission controller)来确定标准NoC中每个节点的适当注入速率和控制策略
### 4.3.3 Reliability and Fault Tolerance
随着技术规模的缩小(technology scaling down)，晶体管和链路(transistors and links)更容易出现不同类型的错误，这表明可靠性是一个至关重要的问题，需要积极的容错技术来保证性能，[233]使用基于路由器(per-router-based)的Q-learning智能体来独立选择四种容错模式中的一种(one of the four fault-tolerant modes)，从而使端到端的数据包延迟和功耗最小化，这些智能体被预训练好，在运行时微调；后续的工作[234]中，这些纠错模式(error-correction modes)得到了扩展，并与各种多功能自适应信道配置(multi-function adaptive channel configurations)、重传设置(retransmission settings)和电源管理策略(power management strategies)相结合，显著优化了延迟、能量效率和平均故障时间(mean-time-to-failure)
### 4.3.4 General Design
随着每个芯片/系统核心数量的不断增加，核心异构性(heterogeneity)以及各种性能指标(performance target)的不断增加，要同时优化NoCs上的大量design knobs变得十分复杂
MLNoC[186]尝试自动化NoC设计，使用有监督学习，在多个优化目标下快速找到接近最优的NoC设计，MLNoC的训练数据是数以千计的现实世界的和合成的 SoC(System-on-Chip/片上系统)的设计，同时MLNoC也用真实世界的SoC设计进行评估，MLNoC的设计相较于人工优化的NoC设计具有更优越的性能

除了传统的2D网格NoCs，有一系列关于3D NoCs的研究，其中[44, 45]应用STAGE算法优化基于3D NoCs的小世界网络中的通信链接的垂直和平面布局(vertical and planar placement of communication links in small network based 3D NoCs)，STAGE算法在两个阶段之间反复交替，基础搜索阶段基于学到的评估函数尝试找到局部最优，元搜索阶段使用SVR学习评估函数；[97]将STAGE算法延伸至在异构性3D NoC系统中的多目标优化(multi-objective)，多目标优化共同考虑了GPU吞吐量、CPUs和LLCs之间的平均延迟、温度、能量；对于任意两个节点通过至少一个环(ring/loop)连接的无路由器NoCs(routerless NoCs)，[134]采用了深度RL框架，利用蒙特卡洛树搜索进行有效的设计空间探索以优化环路布置(loop placement)，并且如果仔细设计奖励函数(reward function)，可以严格执行设计约束(design constraints)
## 4.4 Resource Allocation or Management
资源分配或管理是计算机体系结构/系统和工作负载之间的协调(coordination)，因此，它的优化困难出现在双方日益增长的复杂性及其复杂的相互作用中
基于ML的方法可以根据动态的工作量或特定的约束(dynamic workloads or specified constraints)，迅速地调整策略
### 4.4.1 Power Management
基于ML的技术已被广泛应用于提高电源管理(power management)，主要有两个原因：第一，电力/能源消耗(power/energy consumption)可被视为运行时成本的一个度量(one metric of runtime costs)，其次，在某些情况下，可能存在对电力/能源的硬性或软性约束/预算(hard or soft constraint/budget)，因此必须提高功率效率

考虑到对系统的不同部分进行的电源管理，PACSL[1]使用命题规则(propositional rule)以调整对CPU核心和片上L2缓存的动态电压调节(dynamic voltage scaling/DVS)，在为每个部分独立应用DVS后，实现了功耗延时积(energy-delay product)的平均22%的提高；[238]coordinate an ANN controller with a proportional integral for uncore DVFS(非计算核心的动态电压与频率缩放)，ANN控制器可以用准备好的数据集离线预训练，也可以用自助学习(bootstrapped learning)在线训练；[182]利用了Q-learning
自适应地在通信功率和误码率(communication power and error bit rate)约束下，调整transmitters of 2.5D through-silicon interposer I/Os的输出电压摆动(output voltage swing)

在系统的层面，DVFS是最为流行的技术之一，[36]离线训练多项式逻辑回归分类器，在运行时进行查询，用于在任意功率上限(power cap)的条件下，准确地确定thread packing和DVFS的最佳操作点(optimal operating point)；GreenGPU[144]专注于具有CPU和GPU的异构系统，应用加权多数算法(weight majority algorithm)，对GPU核心和内存频率级别进行缩放，使其保持一致(scale frequency levels for both GPU cores and memory in a coordinated manner)；CHARSTAR[187]目标是联合优化单个核内的电源门控(power gating)和DVFS，核心的频率和配置(frequencies and configurations)通过轻量级的离线训练好的MLP预测器进行动态选择，以最小化能源消耗；[82]使用基于ML的分类器(例如树模型、梯度提升、KNN、MLP和SVM)，利用低层的硬件性能计数器(low-level hardware performance counters)，以预测最节能的资源设置(the most energy-efficient resource settings)(具体来说，就是调优插口分配/tuning socket allocation、 HyperThreads的使用、处理器DVFS)；[11]考虑到了DVFS时片上调节器的效率(on-chip regulator efficiency)造成的损失，尝试最小化在一个参数化的性能约束的能源消耗，而在线的控制策略则由基于table的Q-learning执行，该方法跨平台可移植，不需要对特定系统进行准确建模

一系列的研究利用RL在多核系统中进行动态电源管理，但随着系统规模增大(scale up)这些基于RL的方法经常遭遇状态空间爆炸(state space explosion)，有两种类型方法用于解决可扩展性问题(scalability issue)：(1)将RL与有监督学习结合， [103]提出半监督的基于RL的方法，复杂度与和兴数量呈线性，能够最大化吞吐量，同时确保功率约束和以协同的方式控制计算核心和非计算核心(control cores and uncores in synergy)；(2) 使用层次化Q-learning将时间复杂度减少到$O(nlogn)$，其中$n$表示核心的数量，[178]提出多层次Q-learning以选择目标功率模式(target power modes)，其中Q值由广义径向基函数近似；[32]则使用基于Table的分布式Q-learning进行DVFS，一个变体[33]额外考虑了到不同应用程序的优先级

一些能量管理策略针对特定的应用程序或平台，JouleGuard[79]是一个在能源预算下(energy budget)，使用系统资源(system resource)协调近似计算应用(coordinating approximate computing applications)的运行时控制系统(runtime control system)，它使用multi-arm bandit方法来确定最节能的系统配置(system configurations)，在此基础上确定应用配置(application configurations)以实现在能源预算范围最大化计算精度(compute accuracy)；[217]针对英特尔SkyLake处理器，提出应用了各种ML模型来动态时钟门控(clock-gating)未使用的资源的post-silicon CPU
### 4.4.2 Resource Management and Task Allocation
现代体系结构和系统已经变得如此复杂和多样化，使得优化性能或充分利用系统资源都不是一件容易的事情，且在具有特定要求或目标的各种工作量时，会进一步复杂化
因此需要开发更有效和自动化的资源管理和任务分配方法，其中基于ML的技术可以在探索大型设计空间的同时优化多个目标，并在精心设计后保持更好的可扩展性和可移植性

对于单核处理器，一个正则化的最大似然方法[53]基于运行时硬件计数器(runtime hardware counters)，为程序每个阶段的预测最好的硬件微架构配置(hardware micro-architectureal configuration)；对于多核处理器，一种基于统计机器学习(Statistical Machine Learning/SML)的方法[64]可以快速找到能同时优化运行时间和能源效率的配置，且由于该方法对应用程序和微体系架构的领域知识是不可知的，因此它是可移植的，能作为人类专家优化的替代；SML也可以作为一种整体方法应用于设计能层次化地在电路、平台和应用级别(circuit, platform, and application levels)优化性能的自演化系统(self-evolving)[20]；除了调优架构配置(tuning architectural configurations)外，动态的片上资源(on-chip resource)管理也对多核处理器至关重要，其中一个例子是动态缓存分区(cache partitioning)，为了响应不断变化的工作负载需求，[69]使用子种群(subpopulations algorithm)算法演化的RNN来动态划分(partition)L2缓存；[89]将在计算核心和非计算核心上的DVFS和LLC的动态划分结合，采用一种基于表的Q-learning协同优化(co-optimization)方法，达到了比单独使用二者时更低的功耗延时积(energy-delay products)

为了保证多核系统高效可靠的执行，任务分配应该考虑多个方面，如热和通讯问题(heat and communication issues)，针对处理器核心和NoC路由器的热相互作用(heat interaction)，[142]应用Q-learning，基于当前核心和路由器的温度执行任务分配，以最小化未来的最高温度(maximum temperature)；针对多芯片多核心系统的非统一和分层的片上/片外通信能力(non-uniform and hierarachical on/off-chip communication capability)，核心布局优化(core placement optimization)[241]利用深度确定性策略梯度(deep deterministic policy gradient/DDPG)[132]将计算映射到物理核，能够在对领域特定信息不可知的情况下工作

一些研究关注工作流(workflow)管理和一般(general)硬件资源分配(assignment)，SmartFlux[56]专注于数据密集型和连续处理(continuous processing)的工作流，它在多个ML模型(例如SVM、RF)的预测的帮助下，指导处理步骤的异步触发(guide asynchronous trigerring of processing steps)，其中ML模型的预测聚焦于是否要执行特定的步骤(execute certain steps)，以及为每一波数据决定相应的配置(decide corresponding configurations upon each wave of data)；给定目标DNN模型，部署场景(deployment scenarios)、平台约束(platform constraints)和优化目标(延迟/能量)，ConfuciuX[106]应用一个混合的两步(hybrid two-step)方案进行最优的硬件资源分配(即为每个DNN层分配的处理元素的数量和缓冲区大小)，其中REINFORCE[214]执行全局粗粒度搜索(global coarse-grained search)，然后执行用于细粒度调整的遗传算法(genetic algorithm for fine-grained tuning)；Apollo[98]是一个用于样本效率加速设备设计的通用架构探索框架(a general architecture exploration framwork for sample-efficient accelerator designs)，它利用基于ML的黑盒优化技术(例如贝叶斯优化)优化加速设备配置以满足用户指定的设计约束(use-specified design constraints)

在具有CPU和GPU的异构系统中，设备布置(device placement)指的是
将神经网络计算图中的节点(nodes in computational graphs of neural networks)映射到适当的硬件设备上，最开始时，[162]将计算操作(computational operations)手动分组，然后由REINFORCE分配给设备，REINFORCE以seq到seq的RNN模型作为参数化策略(paremeterized policy)；后来，[160]提出一个层次化的端到端模型使手动分组过程自动化；[66]引入了近端策略优化(proximal policy optimization/PPO)进一步优化训练速度；尽管上述方法带来了巨大进展，但它们是不可迁移的(not transferable)，对于每一个新的计算图，都要重新训练策略(policy)；通过将计算图的结构编码为静态的图嵌入(static graph embedding)[2]或可学习的图嵌入(learnable graph embedding)[270]，训练得到的布置策略(placement policy)对不可见神经网络展现了很好的泛化性
### 4.4.3 Scheduling
在经典的实时调度问题中，关键任务就是决定当前尚未被调度的作业(currently unscheduled jobs)被交由单个处理器执行的顺序，以优化整体的性能
而随着多核处理器成为主流，调度问题也逐渐变得复杂，一个主要原因是除了
性能以外，各种核心之间的平衡分配(balanced assignments among various cores)，响应时间公平性(response time fairness)等目标也开始需要纳入考虑
RL具备充分理解来自环境的反馈并动态调整策略的能力，因此成为实时调度的常用工具
为了优化作业被路由到单个CPU核心后的执行顺序(the execution order of jobs after they are routed to a single CPU core)，[236]提出了一种利用Q-routing的自适应调度策略(adaptive scheduling policy)，其中调度器利用路由器的Q-table来评估作业的优先级，并相应地决定作业的顺序，以最大化整体效用(overall utility)；在多核系统中，[58]提出了一种基于RL中基于值的时间差分方法(value-based temporal-difference method in RL)的自调整(self-tuning)调度算法，旨在最大化作为感兴趣的度量的任意加权和的成本函数(cost function that is an arbitrary weighted sum of metrics of interest)；[226]将算法改进为并行作业在线调度的通用方法(a general method for online scheduling of parallel jobs)，其中用参数化的模糊规则库来近似值函数(the value functions are approximated by a parameterized fuzzy rulebase)，此调度策略总是选择作业队列中具有最大值函数(value function)的作业执行，这可能会抢占(preempt)当前正在运行的作业，并将一些作业压缩到比它们理想情况下需要的更少的CPUs中(squeezes some jobs into fewer CPUs than they ideally require)，但目标是实现优化的长期效用(optimized long-term utility)
## 4.5 Data Center Management
随着数据中心的快速规模扩张，在一台机器中可能微不足道的问题变得越来越具有挑战性，更不用说固有的复杂问题了(inferently complicated)

早期的工作针对相对简单的资源分配(resource allocation)场景，即动态分配
不同数量的服务器(servers)给多个应用程序，
这个问题可以建模为RL问题，以服务水平效用函数(service-level utility function)作为奖励(rewards)：仲裁器(arbiter)依据本地值函数(local value functions)，其中本地值函数通过基于表格的方法[221]或函数近似[220]进行评估(estimated)，选择一个能获得最大总回报(maximum total return)的联合动作(joint action)；
为了更好地建模多个智能体之间的交互，[227]使用模糊(fuzzy)RL[227]的多智能体协调(multi-agent coordination)算法来解决内容交付网络(content delivery networks/CDN)中的动态内容分配(content allocation)，其中每次被请求的内容(each requested content)都被建模为一个智能体，在与其他智能体/内容协调(coordinate)的同时向需求量大的地区移动(area with a high demand)；
最近的一项创新[13]关注虚拟机到物理机的布置(placement)，以最大限度地降低物理机器中资源使用的峰值与均值之比(peak-to-average ratio of resource usage)，[13]对PPO和事后模仿学习(hindsight imitation learning)两种方法进行了评估
为了提高数据中心的性能和用户的体验质量(quality of expreience/QoE) ，基于 ML的技术已经在几个方面进行了探索：
(1) 高效调度作业以及高效诊断作业中的掉队者(stragglers within jobs)是重要的，针对数据中心的流量/交通优化(traffic optimization)(例如流量调度/flow scheduling，负载均衡/load balancing)，[30]开发了一个两级(two-level)RL系统: 其中外围系统(peripheral systems)通过DDPG进行训练，驻留在终端主机上(reside on end-hosts)并在本地进行即时(instant)流量优化决策，而中央系统(central system)通过策略梯度(policy gradient)进行训练，聚合全局流量信息，指导外围系统的行为，并为long flows作出交通优化决策；
[151]利用GNN来表示集群(cluster)信息和作业阶段(job stages)之间的依赖关系，这样基于RL的调度程序可以自动学习特定于工作负载(workload-specific)的调度策略，以调度具有复杂依赖关系的数据处理作业；
[267]将统计机器学习与元学习相结合，诊断数据中心规模的作业中的掉队者的成因(causes of stragglers at data-center-scale jobs)
(2) 部署智能的数据中心级别的缓存是重要的，DeepCache[171]采用LSTM编码器-解码器模型以预测未来的内容数量(content popularity)，预测结果和现存的缓存策略结合，以做出更智能的决策；
[211]将梯度提升机用于模拟一个松弛的Belady算法，该算法驱逐(evicts)下一个请求超出重用距离阈值(reuse distance threshold)对象，但被驱逐的对象不一定是未来最远的(farthest in the future)；
[242]是一个在线缓存替换(replacement)框架，利用DDPG预测对象优先级并进行相应的驱逐；
考虑到不基于历史的特征(non-history based features)，[232]构建了一个决策树来预测被请求的文件在将来是否只会被访问一次，这些只会被一次性访问的文件将直接发送给用户而不进入缓存，以避免缓存污染(pollution)
(3) 从工作负载角度来看，CDN或集群上的视频工作负载(video workloads)非常普遍，但它们的优化相当具有挑战性: 
首先，网络条件会随时间波动且各种QoE目标需要被同时平衡(be banalced simultaneously)
第二，只有粗略的(coarse)决策是可用的(available)，而当前的决策将对之后决策有长期影响，
这种情况自然地符合基于RL的技术，为了优化用户对流媒体视频(streaming videos)的QoE，自适应比特率(bitrate)算法被认为是内容提供者(content provider)使用的主要工具，该算法在客户端的视频播放器执行，根据网络条件为每个视频块(chunk)动态选择一个比特率；
[150, 252]应用asynchronous advantage actor-critic[166]，基于过去决策产生的性能(resulting performance from past decisions)为未来视频块选择合适的比特率；
当考虑混合CPU-GPU集群中的大规模视频工作负载时，性能往往会因为来自工作负载的不确定性和可变性，以及对异质(heterogeneous)资源的不平衡使用而下降，为了适应这种情况，[263]使用两个深层Q网络来构建一个两级任务(task)调度器，其中集群级(cluster-level)调度器为相互独立的视频任务(video tasks)选择合适的执行节点(execution nodes)，节点级(node-level)调度程序将相关的视频子任务分配给适当的计算单元(computing units)，该方案使调度模型能够根据集群环境(cluster environments)、视频任务的特性以及视频任务之间的依赖的运行时状态(runtime status)调整策略
## 4.6 Code Generation and Compiler
### 4.6.1 Code Generation
由于编程语言和自然语言之间语法(syntax)和语义(semantics)的相似性，代码生成或翻译的问题通常可以建模为一个NLP问题或一个神经机器翻译(nueral machine translation/NMT)问题
一个全面的调查[7]详细地对比了编程语言对自然语言，并讨论了这些相似性和差异是如何驱动不同的ML模型在代码中的设计与应用的

针对代码补全(code completion)，[188]探索了使用几个统计语言模型(statistical language model)(N-gram模型、RNN 和这两者的结合)选择有最高的概率并满足约束的可以补全程序的句子；
针对代码生成(code generation)，CLgen[40]用手写代码的语料库训练 LSTM模型，学习OpenCL程序的语义和结构，通过从学习到的模型中迭代抽样(iteratively sampling from the learned model)生成程序；
针对程序翻译(program translation)，基于NMT的技术被广泛应用于将程序从一种语言迁移到另一种语言，
例如，具有编码器-解码器结构的树到树(tree-to-tree)模型有效地将程序从Java转换成C#[31]；
序列到序列(seq2seq)模型可以将CUDA转换为OpenCL[111]；
Coda[63]不是在高级编程语言之间转换，而是将二进制可执行文件转换为相应的高级代码(high-level)，它采用了树到树编码器-解码器结构生成代码示意图(code sketch)，以及采用了集成的(ensembled)基于RNN的错误预测器，用于对生成的代码进行迭代错误校正(iterative error correction)
这些有监督的基于NMT的技术可能会遇到几个问题：难以泛化到比训练程序更长的(longer)程序，词汇量有限(limited size of vocabulary sets)，缺乏对齐的的输入输出数据(aligned input-output data)
TransCoder[120]充分依靠无监督的机器翻译，采用了transformer架构，并使用单语言源代码(monolingual source code)在C++、 Java和Python之间进行翻译
### 4.6.2 Compiler
编译器的复杂性随着计算机体系结构和工作负载的复杂性而增加
基于ML的技术可以从多个角度优化编译器，例如：指令调度(instruction schduling)、启发式编译器(compiler heuristics)、应用优化的顺序(the order to apply optimizations)、热路径识别(hot path identification)、自动向量化(auto-vectorization)，以及针对特定应用程序的编译(compilation for specific applications)

(1) 关于指令调度，一个调度相对于另一个调度的优先函数(perference function)可以通过RL中的时间差(temporal-difference)算法来计算[154]；
关于高度约束代码优化下的调度问题(under highly-constrained code optimization)，投影重参数化(projective reparameterization)[87]可以在指令间的数据依赖部分顺序(data-dependent partial orders over instructions)的约束条件下实现自动指令调度
(2) 为了改进启发式编译器，增强拓扑的神经演化(NEAT/Neuro-Evolution of Augmenting Topologies)[37]通过调节布置代价函数(tuning placement cost functions)提高指令启发式布置(instruction placement heuristics)；
为了避免人工特征工程，基于LSTM的模型[39]自动从原始代码(raw code)中学习启发式编译器，它为程序构造适当的嵌入(embedding)，同时学习优化过程(learn the optimization process)
(3) 关于选择应用不同的优化的适当的顺序，NEAT[114]可以自动为程序中的每个方法(method)生成有益的优化顺序(optimization orderings)
(4) 关于路径剖析(path profiling)，Crystal Ball[260]使用LSTM模型静态地识别热路径，即经常被执行的指令序列(sequences of instructions)，
由于Crystal Ball只依赖于IR，因此它避免了人工特征工程，并且独立于语言或平台
(5) 关于自动矢量化，[158]利用模仿学习(imitation learning)，模仿(mimic)基于超词级并行的向量化(superword-level parallelism based vectorization)提供的最优解[156]
(6) 关于针对特定应用程序的编译，有研究改善了为近似计算(approximate computing)或DNN应用程序的编译，
考虑编译近似计算，[55]提出了一种程序转换(transformation)方法，训练MLPs模仿(mimic)可近似的代码区域(regions of approximable code)，并最终用经过训练的MLPs代替原始代码；
[251]将这种算法转换(algorithmic transformation)扩展到了GPU
考虑编译DNNs，RELEASE[4]利用PPO搜索DNNs的最佳编译配置；
EGRL[107]在编译过程中优化DNN张量的内存放置(memory placement)，它结合了GNN、RL和进化搜索，找出到不同的板载内存组件(on-board memory compnents)(即SRAM、LLC和DRAM)的最佳映射(mapping)
## 4.7 Chip Design and Design Automation
### 4.7.1 Analog Design
与高度自动化的数字设计相比，模拟设计通常需要大量的人工和领域专业知识(domain expertise)，
首先，模拟电路有很大的设计空间来搜索适当的拓扑结构和设备大小
其次，目前并没有通用的用于优化或评估模拟设计的框架，以及设计规范(design specifications)通常根据具体情况而有所不同(vary case by case)

最近，机器学习技术被引入以加速模拟设计自动化
我们将按照模拟设计自上而下的流程(top-down flow)讨论这些研究：
在电路级别(circuit level)上，选择一个适当的电路拓扑以满足系统规范(system specifications)
然后在设备级别(device level)，根据不同的目标进行优化设备尺寸
这两个步骤组成了布局前设计(pre-layout designs)，而在设计好电路原理图(circuit schematics)后，就要在物理层面生成模拟布局

(1) 在电路层面，目前有自动生成二端口线性模拟电路(two-port linear analog circuits)的尝试[195]，设计规范(design specifications)由超网络编码(encoded by hypernetwork)[72]，为一个RNN模型生成权重，RNN模型被训练，然后用于选择电路元件(components)和它们的配置
(2) 在设备层面，RL和GNN的组合使自动晶体管尺寸选择(transistor sizing)成为可能[231] ，该工作能够推广到不同的电路拓扑(circuit topologies)或不同的技术结点(technology nodes)；
AutoCkt[205]迁移学习技术引入到深度RL中，用于自动大小选择(automatic sizing)，速度是传统遗传算法的40倍；
[194]全面讨论了如何通过深度学习和ANNs实现模拟集成电路(analog ICs)的自动大小选择和布局(automatic sizing and layout)
(3) 在物理层面，GeniusRoute[272]通过生成式神经网络指导自动化模拟电路布线(analog routing)，模拟布局和布线表示为图像，通过变分自动编码器(VAE)[71]来了解每个区域的路由可能性(routing likelihoods of each region)，GeniusRoute性能可以和人工布局(manual layout)相比，并能够推广到(generalize)其他不同功能的电路；
[139]将多目标贝叶斯优化应用于优化净加权参数的组合(combinations of net weighting parameters)，这可以显着改变平面图(floor plans)和布局解决方案(placement solutions)，以改善积木块电路(building block ciruits)的模拟布局
### 4.7.2 Digital Design
对于应用机器学习技术直接优化数字设计的研究，我们按照自顶向下的流程组织它们，即HLS(High Level Synthesis/高层次综合)、逻辑综合和物理综合

HLS设计中的设计空间探索通常涉及到在高级源代码中正确指定指令(杂注)(properly assigning directive/pragmas in high-level source code)，因为指令用于控制并行性(control parallelism)，调度(scheduling)和资源使用(resource usage)，会显著地影响HLS设计的质量
优化的目标往往是找到在不同目标之间的帕累托解(Pareto solutions between different objectives)，或者是满足预定义(pre-defined)的约束

在IR分析中，利用随机森林能够选择合适的循环展开因子(loop unrolling factors)，以优化执行延迟(execution latency)和资源使用的加权和[259]；
Prospector[155]使用贝叶斯方法优化指令布置(placement of directives)(循环展开/流水线 loop unrolling/pipelining，数组分区 array partitioning，函数内联 funtion inlining，分配 allocation) ，旨在找到FPGAs上的执行延迟和资源利用之间的帕累托解(Pareto solutions between execution latency and resource utilization)；
IronMan[243]目标于在保持延迟不变的前提下(keep latency unchanged)，找到不同资源之间的帕累托解(Pareto solutions between different resources)，它将GNNs与RL结合，在运算级别执行更细粒度的设计探索(conduct a finer-grained design exploration in the operation level)，通过优化资源杂注的赋值(assignments of resource pragma)，得到最优的资源分配策略

在逻辑综合中，RTL设计或逻辑网络(logic networks)用有向无环图表示，目标是在一定的约束条件下优化逻辑网络
LSOracle[173]使用MLP自动决定对于电路中不同的部分应该两个优化器中的哪一个；
逻辑优化可以表述为一个RL问题，状态是设计的当前状态，动作是两个有着相同I/O行为的DAGs之间的转换，优化目标使设计的面积或延迟最小化，该问题可以由策略梯度[73]或优势行为者-批评者[81]解决；
Q-ALS[180]的目的是近似逻辑综合，嵌入一个Q学习智能体来确定DAG中每个节点的最大可容忍误差(tolerable error)，以使得初级输出(primary outputs)的总错误率受到预先指定的约束的限制

在物理综合中，布局(placement)优化是一个热门的话题
(1) 为了优化时钟网络(clock networks)中的触发器放置(flip-flop placement)，[240]将一种改进的K-means算法用于分组后置触发器(post-placement flip flop)，并通过减少触发器和触发器的驱动器(drivers)之间的距离，且同时最小化原始布局结果的disruption，来重新定位这些簇；
优化时钟树综合(Clock Tree Synthesis/CTS) ；
[143]训练了一个回归模型，接受CTS前的布局图像以及CTS配置作为注入，预测CTS后指标(时钟功率 clock power，时钟线长 clock wirelength，和最大偏差 maximum skew)，这些指标再用作指导条件GAN训练的监督信息(supervisor to guide the trainning of a conditional GAN)，则训练好的生成器可以推荐CTS配置，从而优化时钟树
(2) 针对单元放置(cell placement)，[161]引入了深度RL方法来放置宏(内存单元)，在此之后，标准单元被强制方向(force-directed)放置
这种方法能够推广到未见得网络列表(netlists)，效果好于RePlace[34]，但速度慢几倍；
DREAMPlace[135]将解析式标准单元布局优化问题转化为神经网络训练问题，速度比RePlace快30倍，且效果不变差；
NVCell[189]是一个标准单元的自动布局生成器，用RL在放置后fix DRVs以及进行布线
(3) 基于ML的技术在许多设计自动化任务中展示了它们的通用性，如用稀疏贝叶斯学习进行后硅变化提取，和后硅时间调整，以减轻过程变化(process variation)所造成的影响[274]
# 5 Discussion and Potential Directions
在这一部分，我们讨论ML for 计算机体系结构和系统的潜力和局限性
讨论将跨越整个开发和部署栈，包括数据，算法，实现和目标，我们亦预期机器学习技术的应用可以成为硬件敏捷开发(hardware agile development)的推动力
## 5.1 Bridging Data Gaps
数据是机器学习的支柱，然而，完美的数据集有时是不可得的，
在这里，我们想仔细审查两点：小数据和大数据之间的差距，以及不完善(non-perfect)的数据
(1) 在一些EDA问题中，如物理综合中的布局和路由问题，模拟或评估是及其昂贵的，因此这方面的数据也是十分缺乏的
由于机器学习模型通常需要足够的数据来学习基本的统计规律并做出决策，小数据和大数据之间的差距经常限制了基于机器学习技术的能力
有许多想要弥补小数据和大数据差距的尝试：
在算法方面，能够处理小数据的算法有待开发，当前的技术有贝叶斯优化[108]，在小参数空间较为有效；以及主动学习(active learning)[206]，显著提高了样本效率(sample efficiency)
从数据方面来说，生成式方法可以用来生成合成数据[49]，缓解数据缺乏
(2) 关于不完善的数据，即使一些EDA工具可以产生大量的数据，这些数据并不总是被正确地标记，或不是以适合机器学习模型的形式呈现，在没有完美标记的训练数据的情况下，可能的替代方案是使用无监督学习，自监督学习[78] ，或结合监督与无监督技术[5]，与此同时，RL也是一个可以在运行中生成训练数据的解决方案
## 5.2 Developing Algorithms
我们仍然期待新的机器学习算法进一步在可扩展性、领域知识的可解释性等等方面改进系统建模和优化

**New ML Schemes** 经典的基于分析的方法通常采用自下而上或自上而下的过程，鼓励基于机器学习的技术提炼系统/架构的层次结构(distill hierarchical structures of systems/architecture)
一个例子是层次化RL[115] ，它具有灵活的目标规范(goal specifications)，并在具有稀疏反馈的复杂环境中学习目标导向的行为。这种模型可以实现更加灵活有效的多级设计和控制；
此外，许多系统优化包括多个智能体的参与(如NoC路由)，这类问题属于多智能体RL[264]的领域，这些智能体可以是完全合作的，完全竞争的，或二者混合，这使得系统优化更具多样性；
另一个有前途的方法是自监督学习[78] ，有利于提高模型的健壮性和减少数据稀缺性；
混合方法，即组合不同的机器学习技术或结合机器学习技术与启发式，例如RL可以与硬件资源分配的遗传算法相结合[106]

**Scalability** 系统的扩展对可放缩问题提出了挑战
在算法方面，多级技术有助于降低计算复杂度，如多级Q学习 for DVFS[32,33,178]；
一个隐含的解决方案是利用迁移学习： 预训练是一次性成本，可以在将来的每次使用中摊销，微调提供了在预训练模型提供的快速解决方案和需要更多时间的为特定任务训练的更好的模型之间的灵活性，几个例子[161,205,231]已经在第4.7节中讨论

**Domain Knowledge and Interpretability** 更好地利用领域知识可以帮助不同的系统问题选择更合适的模型，并提供更多的关于模型如何工作和为什么有效的直觉或解释
通过在内存访问模式/程序语言和自然语言之间进行语义类比，预取或代码生成问题可以建模为NLP问题，如4.1.1和第4.6.1节中讨论
通过对许多EDA问题中的图形表示进行类比，在这些问题中，数据本质上是以图形(例如，电路、逻辑网表 logic netlists或IR)表示的，可以利用GNN来解决问题[108]，在3.3节和4.7节中提供了几个例子
## 5.3 Improving Implementations and Deployments
为了充分利用基于ML的方法，我们需要考虑实际的实现、部署场景的适当选择以及部署后的模型维护(maintenance)

**Better Implementations** 为了更好实现基于机器学习的技术，可以从模型方面或软件/硬件协同设计方面进行改进[215]
从模型层次来看，网络剪枝(network pruning)和模型压缩减少了操作数量和模型大小[76]；权重量化通过降低操作/操作数(operatoins/operands)的精度提高了计算效率[86]
从协同设计的角度来看，已经用于DNN加速的策略也可以应用于ML for system

**Appropriate Scenarios: online vs. offline** 在部署ML for system designs时，考虑不同场景下的设计约束是至关重要的
一般来说，现有的工作分为两类
(1) 基于机器学习的技术要进行在线或运行时部署(无论训练阶段是在线或离线执行)，显然，模型的复杂性和运行时开销(runtime overhaed)通常受到特定约束的严格限制，例如，功率/能量、时间/延迟、面积(area)等
更进一步，如果需要进一步的在线训练/学习，则设计约束将更加严格，
一个有前景的方法是使用半在线(semi-online)学习模型，这已经被应用于解决一些经典的组合优化问题，如二分匹配(bipartite matching)[116]和缓存(caching)[117]
(2) 基于ML的技术离线应用(applied offline)，这通常指架构设计空间的探索这些问题利用基于机器学习的技术来指导系统实现，一旦设计阶段完成，机器学习模型将不会再被调用。因此，离线应用可以容忍相对较高的开销

**Model Maintenance** 在离线训练和在线部署的情况下，计算机体系结构领域使用的机器学习模型，与其他情况一样，需要定期维护和更新，以满足性能预期(performance expectations)，因为工作量会随着时间变化以及硬件会老化，这往往导致数据偏移或概念偏移(data drift or concept drift)[222]
为了主动避免机器学习模型的性能下降，可以在部署后阶段采取一些措施：
(1) 机器学习模型可以定期进行再训练(retrain)，也可以在关键性能指标(key performance indicators)低于特定阈值时进行再训练，
无视模型性能定期再训练是更为直接的方法，但这需要清楚地了解模型在自己的场景下应该更新的频率，如果再训练间隔过长，模型性能将下降
监测关键性能指标则依赖于一个能明确显示模型偏移的全面的度量面板(panel of measurements)，而这可能会引入额外的硬件/软件开销，以及不正确的度量选择往往违背这种方法的意图
(2) 在机器学习模型的再训练过程中，新收集的数据和以前的数据之间往往存在权衡，正确分配输入数据的重要性可以提高再训练的效果[25]
## 5.4 Supporting Non-homogeneous Tasks
基于机器学习的技术被认为既适用于当前的体系结构，也适用于新兴的体系结构，可以引领计算机体系结构和系统的长期进步

**Non-homogeneous Components** 计算机体系结构的设计和开发通常基于用途相似(similar purpose)的上一代架构，但通常依赖于上一代没有的下一代硬件组件(hardware components)，
例如使用新的设备节点和技术扩展(employment of new device nodes with technology scaling)，以及用基于NVM或PIM的组件替换存储系统中的常规组件，
除了来自不同代的组件导致的异质性，一个架构或系统还通常由来自库的标准部件和专用/定制硬件组件共同构成
因此机器学习辅助的架构/系统应该要有在不同代组件之间迁移的灵活性(flexibility to transfer among different-generation components)，且要同时支持标准件和专用件(standard and specialized components)

**Non-homogeneous Applications** 在计算机体系结构和系统设计中，有一些问题是共通的，而一些问题则可能会随着新的体系结构/系统和新的工作负载的出现而出现
(1) 对于设计领域，包括在硬件/软件/数据中心中的缓存(第4.1.1节和第4.5节) ，单核/多核/多核CPUs和异构系统的资源管理和任务分配(第4.4节) ，各种场景下的NoC设计(第4.4节)
(2) 对于新系统/工作负载引起的问题，迁移学习和元学习[174,225]可能有助于探索新的启发式或直接推导出设计方法论，例如，将元学习与RL[61]相结合可以训练一个“元”智能体，它被设计用来在少量的观察下就能适应特定的工作负载
## 5.5 Facilitating General Tool Design and Hardware Agile Development
尽管基于ML的建模显著降低了设计迭代过程中的评估成本，朝着硬件敏捷开发的到来迈出了很大的步伐，但仍有很长的路要走
在基于机器学习的设计方法论的前景中，一个最终的目标可能是完全自动化设计，其中应该包含两个核心能力：系统整体优化(holistic optimization in system-wise)，以及跨不同系统的轻松迁移(easy migration across different systems)，从而实现快速和敏捷的硬件设计

**Holistic Optimization** 由于最近的进步，机器学习技术已经越来越多地应用在在计算机系统设计和优化方面进行探索和开发[47]，需要进一步努力解决地问题可以是在高度受限的情况下(under highly constrained situations)的多目标优化，或者同时优化系统中的几个组件
我们设想了一个基于机器学习的系统性和整体性框架，它具有全景视野(panoramic vision)：它应该能够在协同(synergy)利用来自不同层次的系统的信息，以能够彻底描述系统行为以及它们本质上的层次抽象(intrinsically hierarchical abstractions)； 它还应该能够在不同的粒度做出决策，从而能够精确、全面地控制和改进系统

**Portable, Rapid, and Agile** 致力于便携、快速和敏捷的硬件设计，有两个方向：
(1) 系统/架构基于ML的技术之间设计良好的接口将促进跨平台的可移植性，因为机器学习模型可以在没有对目标域的明确描述的情况下也达到好的性能
(2) 基于ML的技术的进步或多或少地改变了设计自动化的工作流程，直接驱动了快速和敏捷的硬件设计，我们期望GNN更好地利用来自EDA领域的自然的图形数据；我们期望深度RL成为解决许多EDA优化问题的强大的通用工具，
尤其是在确切的启发式或目标是模糊的时候；我们期望这些基于机器学习的设计自动化工具可以提高设计师的工作效率，并在社区中蓬勃发展
# 6 Conclusion
如果没有强大的系统和强大的体系结构支持算法的大规模(at scale)运行，机器学习的进步将会受到阻碍，因此需要让机器学习改变计算机体系结构和系统设计的方式

现存的将机器学习应用于计算机体系结构/系统的工作大致分为两类：
基于机器学习的快速建模，包括性能指标或其他一些感兴趣的标准，
基于机器学习的设计方法论，直接利用机器学习作为设计工具
