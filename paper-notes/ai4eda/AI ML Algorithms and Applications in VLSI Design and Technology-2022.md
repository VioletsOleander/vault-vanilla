# Abstract
集成电路产业面临的一个明显挑战是研究与开发方法，以减少随着日益增长的制程变异(growing process variations)而来的设计复杂性(design complexity)和削减芯片制造周转时间(the turnaround time of chip manafacturing)

用于此类任务的传统方法主要是手工的，耗时的和资源密集的，相比之下，AI独特的学习策略提供了许多令人兴奋的自动化方法以处理在超大规模集成电路(VLSI vary-large-scale integraion)设计和测试方面的复杂的以及数据密集型的任务

在超大规模集成电路设计和制造中采用机器学习算法减少了理解和处理不同抽象层次内部和不同抽象层次之间的数据所需要的时间和精力，因此提高了集成电路的产量(yield)，缩短了制造周转时间

本文全面回顾在过去被引入到超大规模集成电路的设计和制造中人工智能/机器学习(AI/ML)自动化方法

此外，我们还讨论了人工智能/机器学习应用的未来范围，以彻底改变超大规模集成电路设计领域，以高速、高智能、高效率的实现为目标
# 1 Introduction
互补金属氧化物半导体(CMOS complementary metal-oxide semiconductor)晶体管在集成电路工业中引领了半导体器件(semiconductor devices)时代

CMOS技术微电子学一直占据着主导地位，自1960s以来，单芯片上晶体管数量呈指数级增长[1, 2]，经过多代技术的发展，晶体管的尺寸不断缩小(down scaling)，提高了这些器件的密度和性能，引领了微电子工业的巨大增长

现代超大规模集成电路(VLSI)技术使单个芯片上复杂数字系统(complex digital systmes)的实现成为可能

近年来，对于便携式电子设备的高需求显著提高了对具有先进功能的功率敏感(power-senstive)设计的需求，高度先进和可扩展的VLSI电路满足了不断增长的电子工业的需求

持续的设备规模缩小(downscaling)是集成电路技术进步主要的驱动力之一
目前，设备的规模正在缩小到亚3纳米级栅极结构及以上(sub-3-nm-gate regime and beyond)

CMOS技术的大规模缩放为设备工程师带来了许多挑战和新的机遇
随着晶体管尺寸(trainsistor simensions)的减小，半导体工艺复杂度(semiconductor process complexity)增加，当我们接近原子维度(atomic dimensions)时，简单的缩放最终会停止
虽然器件很小，但其性能的许多方面都在恶化，例如，泄漏增加(leaking increases)[4,5,6]；增益降低(gain decreases)；以及对制程变异的敏感性增加[7]
制程变异的增加显著地影响了电路的运行，导致了相同大小的晶体管的性能变化(variable performance)，它进一步影响电路的传播延迟(propagation delay)，电路的传播延迟成为随机变量，从而复杂化了定时关闭技术(timing closure techniques)以及强烈影响了芯片的产量[8]

在纳米级别(nanometer regime)中增长的制程变异是参数产量损失(parametric yield loss)的主要原因
多栅场效应晶体管(FETs Multi-gate field effect transistors)[9]比 CMOS 晶体管更能容忍制程变异，但是它们的性能参数(performance parameters)也受过度缩放的影响[10,11]

芯片的周转时间取决于电子设计自动化(EDA eletronic design automation)工具在克服设计约束(design constraints)方面的性能，EDA中传统的基于规则(rule-based)的方法论需要较长的时间为一系列设计约束产生一个最佳的解决方案
此外，在一定程度上，这些任务所使用的传统解决方案大部分是手工的；因此，它们是时序严格(time critical)和资源密集型(resource intensive)的，故而导致上市延迟(time-to-market delays)
此外，一旦数据被回馈，对于设计人员来说，要理解潜在的机制(即问题的根本原因)，并进行修复，是很困难和耗费时间的，这个困难会在制程和环境变异的影响下进一步增大[12,7]

近十年来，人工智能/机器学习策略在VLSI设计和技术得到了广泛的应用

VLSI-计算机辅助设计(CAD)工具在芯片设计流程的几个阶段都有应用，从设计入口(design entry)到完全自定义布局(full-custom layouts)，高度复杂的数字和模拟集成电路的设计和性能评估取决于CAD工具的能力
随着每个芯片晶体管数量的增加，VLSI-CAD工具的发展变得越来越具有挑战性
半导体和EDA领域中有很多结合AI/ML的机会，在各种VLSI设计和制造层次自动化过程，从而实现快速收敛(convergence)[13, 14]，AI算法被引导和设计，以通过高效、自动化的芯片制造(fabrication)解决方案实现相对较快的周转时间

这项工作尝试总结关于将AI/ML算法在不同的抽象水平上用于VLSI设计和建模的文献，本文提供了从电路建模到片上系统(SoC)设计的详细回顾，以及物理设计，测试和制造，我们还简要介绍了VLSI设计流和AI

论文结构安排如下：
第2节简要讨论了现有关于AI/ML-VLSI的综述文章，
第3节概述人工智能和机器学习，
第4节简要介绍VLSI设计和制造中的不同步骤，
第5节在不同抽象级别(器件级 device level、门级 gate level、电路级 circuit level、寄存器传输级 register-transfer level RTL、布局后模拟 post-layout simulation)详细介绍电路仿真(circuit simulation)中面向AI/ML-CAD的工作
第6节和第7节综述了架构级别和SoC级别的AI/ML算法
第8节和第9节分别讨论了在物理设计和制造(光刻 lithography、可靠性分析 reliability analysis、产量预测和管理 yield prediction and management)中提出的学习策略，
第10节报告了测试(testing)中提出的AI/ML方法，
第11节介绍了AI/ML-VLSI的训练数据来源，
第12节介绍了超大规模集成电路领域中AI/ML方法的挑战和机遇
# 2 Existing Reviews
AI对VLSI设计的影响于1985年由Robert首次展示[15]，他简要解释了在VLSI设计的不同层面上，AI技术在CAD工具中的范围和必要性，并简要介绍了现有的超大规模集成电路-人工智能(VLSI-AI)工具，并强调了将人工智能的扩展功能纳入CAD工具的重要性；
[16]和[17]简要介绍了在VLSI设计过程中引入人工智能的优势及其应用，其中[17]专注于AI在集成电路(IC)行业的应用，特别是在专家系统中；
不同的基于知识的(konwledge-based)系统，如设计自动化助理、NCR设计顾问(design advisor by NCR)和REDESIGN，正在VLSI行业中使用

[18]报道了针对模拟和数字VLSI电路以及基于知识的系统的神经网络实现；
[19]综述了物理设计(physical design)与数据分析(data analytics)的联合优化(joint optimization)以及ML

[20]回顾了物理设计中的ML；
[21]指出了在异步CAD/VLSI中基于ML的算法的机遇与挑战，他们提出开发一个基于ML的推荐(recommendation)工具，称为设计顾问(design advisor)，
设计顾问监控和记录不同设计者使用标准RTL、逻辑综合、放置布线工具的过程中采取的行动(actions)，然后通过运行训练引擎，为给定场景选择最佳操作(action)(说白了就是训练)
随后，设计顾问由电路设计者部署和使用以获得设计建议
这些设计顾问主要专注于异步CAD/ML工具；
[22]通过展示测试领域中的各种ML技术回顾了IC测试

[23]详细讨论了物理设计，产量预测，故障、功率和热分析(failure, power and thermal analysis) ，以及模拟设计领域中使用的各种ML方法；
[24]强调了ML在芯片设计中的应用，他们专注于微架构设计空间探索、功率分析、VLSI物理设计，以及模拟设计中基于ML的方法，以优化预测速度(prediction speed)和带出时间(tape-out time)，
他们提出了一种AI驱动的物理设计流程，该流程具有深度强化学习优化循环(DRL optimization loop)，可以自动探索高质量物理平面图(physical floorplans)、时序约束(timing constraints)和布局(placements)的设计空间，从而获得高质量的结果(results)、下游时钟树综合(clock-tree synthesis CTS)和路由步骤(routing steps)

EDA中的ML目前正受到研究人员和研究界的关注
在IC设计和制造中使用ML可以减少设计者在数据分析、优化设计流程上时间和精力，并缩短上市时间(time to market)[25]；
[26]在不同抽象层次上全面介绍了ML for CAD的最新技术，
该文还对ML在CAD中的使用进行了元研究(meta-study)，以捕捉对VLSI周期各个阶段(various levels of the VLSI cycle)合适的ML算法的总体趋势，根据元研究，与其他的抽象级别和算法相比，ML-CAD的趋势正在转向具有NN实现的物理设计(physical design)，
该文还讨论了在使用ML for CAD时面临的挑战，如组合优化问题、训练数据的有限可用性(limited availability)和实际的限制(practical limitations)；
[27]以表格形式总结了ML-CAD的工作，涵盖了数字/模拟设计流程中的许多抽象级别(abstraction levels)；
[28]详细回顾了GNNs for EDA，重点介绍了逻辑综合、物理设计和验证(verifications)，因为图形是一种表示电路、网络表(netlist)和布局(layour)的直观方式，GNN可以很容易地融入EDA以在不同级别解决组合优化问题，并提高QoR(结果质量 Quality of Results)[29]；
[30]综述了ML在布局和布线方面的应用，展示了在benchmark ISPD 2015数据集上的基准测试结果

[31]讨论了最近的在模拟和数字VLSI中的ML和DL技术，包括了物理设计；
[32]从ML的角度讨论了不同抽象级别的VLSI计算机辅助设计；
[33]讨论了RL在EDA中的应用、机遇和挑战，主要讨论了宏观芯片布局(macro chip placement)，模拟晶体管尺寸(analog transisotr sizing)和逻辑综合

上述综述对文献中提出的AI/ML方法进行了详细讨论，主要涵盖了数字VLSI设计流程的所有抽象级别

这篇综述总结了在不同抽象级别上用于VLSI设计和建模的AI/ML算法的文献，我们还讨论了在半导体设计流程中的不同级别的纳入自动化学习策略(automated learning strategies)的挑战、机遇和范围

图1中的树状图显示了本综述中不同章节(sections)中涵盖的设计抽象级别
图2给出了一个简洁的VLSI设计流程，其中包括工业中使用的传统商业CAD工具和研究人员提出的替代它的AI/ML技术
图6概述了文献中提出的用于VLSI电路仿真的AI/ML技术(用于估计电路性能参数，如晶体管特性 transistor characteristics、统计静态时序分析 SSTA statistical static timing analysis、泄漏功率 leakage power、功耗 power consumption和布局后行为 post-layout behaviour)
# 3 Brief on VLSI Design Flow
传统的数字集成电路设计流程有许多层次，如图2所示；
该流程图涵盖了一个通用的设计流程，包括全定制/半定制(full-custom/semi-custom)IC设计的前端和后端

设计规范(design specifications)抽象地描述了要设计的数字电路的功能、接口和总体架构，它们包括提供功能描述(functional descriptions)、时序规范(timing specifications)、传播延迟(propagation delays)、所需的封装类型(package type)和设计约束(design constraints)块状图(block diagrams)；
它们还充当设计工程师和供应商之间的协议

体系结构设计(architectural design)层次组成系统的基本体系结构，它包括诸多决策，例如精简指令集计算处理器/复杂指令集计算处理器(RISC/CISC)，算术逻辑单元(ALUs)和浮点单元(floating-point units)的数量
该级别的结果是包含了对子系统单元的功能性描述的微观体系结构规范(micro-architectural specification that contains the functional descriptions of subsystem units)
架构师可以基于这些描述估计设计性能和功率(estimate the design performance and power)

行为设计(behavioral design)层面提供了设计的功能描述(functional description of the design)，通常使用Verilog HDL/VHDL编写，
行为级别构成了对功能的高级描述(high-level description of the functionality)，隐藏底层的实现细节(underlying implmentation details)；
时序信息(timing information)在下一级，即RTL描述(寄存器传输级 register transfer level)中进行检查和验证
高级综合(HLS)工具可以将基于C/C++的系统规范(system specifications)自动转换为HDL

逻辑综合工具生成网络表(netlist)，即高级行为描述的门级描述(gate-level description for the high-level behavioral description)，逻辑综合工具确保
门级网表满足时序、面积和功率规范(timing, area, power specifications)；

逻辑验证(logic verification)通过测试台(testbench)/模拟(simulation)进行，
借助可测试性设计(design for testability)的形式验证(formal verification)和扫描插入(scan insertion)也在该阶段进行来检查RTL映射(mapping)[34]

接下来是系统划分(partitioning)，它将大型且复杂的系统划分为小型模块(modules)，然后就是元件平面布置/预布局(floor planning)、布置(placement)和布线(routing)，
平面规划器的主要功能是估计标准单元/模块的设计实施(cell/module design implementation)所需的芯片面积(chip area)，并负责改进设计性能，
布置和布线工具放置子模块(submodules)、门(gates)和触发器(flip-flops)

然后是CTS(时钟树综合)和重置布线(reset routing)，随后，执行每个块的布线

在布置和布线之后，就要进行布局验证(layout verification)以确定设计的布局是否符合电气/物理设计规则(electrical/physical design rules)和源示意图(source schematic)，这些过程是使用工具实现的，如设计规则检查(DRC design rule check)和电气规则检查(ERC electrical rule check)

布局后模拟(post-layour simulation)中，执行对寄生电阻和电容(parasitic resistance and capacitance)的提取和验证(extraction and verificaiton)

之后，芯片移动到签核阶段(sign-off stage)[35]，生成GDS-II文件，并将GDS-II文件发送到半导体铸造厂进行IC制造(fabrication)

IC制造涉及许多先进而复杂的技术，以及必须以最高的精度执行的物理和化学过程，它包括许多阶段，从晶圆制备(wafer preparation)到可靠性测试(reliability testing)
[36]对每个阶段进行了详细描述，
简而言之，硅晶体(silicon crystals)被培养生长(grown)并切片(sliced)以产生晶圆；晶圆必须打磨(polished)到近乎完美才能达到VLSI器件的极度小的尺寸(exiremely small dimensions)；

制造过程有多个步骤(step)，包括晶圆上各种材料的沉积(deposition)和扩散(diffusion)；布局数据(layout data)从GDS-II文件转换为光刻掩模(photolithographic mask)，每层一个(one for each layer)，掩模定义了晶圆上的哪些空间需要哪些材料的沉积、扩散或甚至去除(removed)；每个步骤使用一个掩模，要完成制造过程，一般需要使用几十个掩模；

光刻(Lithography)是涉及在IC的特定区域使用不同材料的掩模的准备和验证以及定义(preparation, verification, definition)的一个步骤，它是制造过程中的关键步骤，在不同的阶段(stages)都会重复多次
这也是受技术节点的缩放(downscaling of technology nodes)和过程变异(process variations)影响最大的步骤

在芯片被制造之后，晶圆会被切割(diced)，单独的芯片被分离(individual chips are separated)，随后，每个芯片都被包装和测试(packaged and tested)以验证设计规范(design specifications)和功能行为(functional behaviour)

后硅验证(post-silicon verification)是IC制造的最后一步，即在生产后检测和修复ICs和系统中的bug[37]
# 4 Brief on AI/ML algorithms
ML分为三种主要类型：有监督、无监督和强化学习
## 4.1 Supervised Learning
有监督学习进一步分为两类：分类和回归，有·监督学习的缺点是它需要大量的大量无偏置(unbiased)的标记的训练数据，在VLSI领域中，这较难获得
最流行的回归和分类算法包括线性回归、多项式回归和岭回归，决策树，随机森林，支持向量机，以及集成1学习[41, 42]
## 4.2 Unsupervised Learning
无监督学习被用于识别数据中的未知模式，聚类和通过主成分分析进行降维是主要的应用
常见聚类算法包括K近邻(KNN)、K均值聚类、层次聚类和凝聚层次聚类[44]
## 4.3 Semi-suprevised Learning
半监督学习是有监督和无监督方法之间的桥梁，一般在训练数据具有有限的标记样本和大量未标记的样本的情况下应用
它在自动化数据标记(automate data labeling)方面效果很好，好于单独的监督/无监督学习

在某些应用中，训练从有限的有标签数据开始，然后应用算法对未标记的数据打上伪标记，然后，应用有标签数据、有伪标签数据、无标签数据提高准确性[45，46]

在半监督学习的复杂应用中，需要付出很多努力以让它的两部分(有监督、无监督)都收敛
## 4.4 Reinforcement Learning
强化学习将情况映射到行动，以最大化数字奖励信号(signal)，RL注重于基于互动(interactions-based)的目标导向学习(goal-directed)[47]，强化学习是试图从经验中学习，并找到最大化奖励信号的最佳解决方案
## 4.5 Deep Learning
深度学习是ML的一个子集，特别适用于大数据处理
常用DNN包括深度置信网络、堆叠自动编码器(SAE stacked autoencoder)和深度卷积神经网络(DCNNs)[48]，其他流行的技术包括递归神经网络[51]，生成对抗性网络[52, 53]，和DRL(深度强化学习)[54]

在以下几节中，我们将从电路仿真(circuit simulation)开始，讨论AI/ML在VLSI设计和分析的不同抽象级别上的应用
# 5 AI at the Circuit Simulation
模拟(simulation)在IC器件建模中起着至关重要的作用

在纳米领域(nanometer regime)，由于过程和环境变异(process and environmental variations)的增加，通过模拟对设计好的电路的性能评估(performance evaluation)变得非常具有挑战性[55, 56, 57]

在设计周期(design cycle)的早期发现功能和电气性能变化(functional and electrical performance variatinos)可以提高IC产量(yield)，而这取决于模拟工具(simulaton tool)的能力

利用E-CAD工具中的AI/ML算法，可以减少设计工作(design effort)的同时，改进周转时间(turnaround time)和芯片性能
研究人员已经提出了针对的泄漏功率(leakage power)、总功率(total power)、动态功率(dynamic power)、传播延迟(propagation delay)和范围从栈级别晶体管模型(stack-level transistor models)到子系统级别(subsystem level)[58]的IR-drop估计的表征(characterization)的替代方法论(surrogate methodologies)

不同的AI/ML算法已被探索用于不同抽象级别的电路建模，包括线性回归、多项式回归、响应面建模(RSM response surface modeling)、SVM，集成技术、贝叶斯定理、ANNs和模式识别模型[59]
以下各个小节描述了针对VLSI设备/电路不同抽象级别提出的学习策略
## 5.1 Device Level
在电路和器件的晶体管级别(transistor level)的建模中，参数产量估计(parametric yield estimation)是主要的关注领域
统计感知(statistical-aware)的VLSI的参数产量估计并不是什么新鲜事，自1980s它就一直随着ML算法而发展

[60]提出了统计参数产量估计，用于确定MOS电路的总参数产量；
[61，62]为计算机辅助VLSI器件设计的统计设计分析(statistical design analysis)提出了一种响应面方法(RSM)，所提出的模型已成功应用于优化BiCMOS晶体管的设计；
关于RSM的全面综述，请参考[63]、[64]；
[65]提出了多元多项式回归(MPR multivariate polynomial regression MPR)方法，用于近似MOSFET在饱和状态下的早期电压(early voltage)和特性，他们考虑了一种曲线拟合方法，在MPR中使用最小二乘法来简化BSIM3和BSIM4方程[66]中的复杂性，以实际计算MOSFET特性

考虑到技术节点(techonlogy nodes)尺寸(dimensions)的急剧缩小，在设备级别(device level)对特性进行彻底分析是至关重要的，
由工艺(process)中的片内和片间变异(inter-die and intra-die varaitions)导致的晶体管行为的随机性会导致器件电流(device current)的指数级变化(exponential changes)，特别是在亚阈值附近(sub-threshold)[56]；
在估计工艺参数(process parameters)对设备的影响方面，统计采样技术(statistical sampling)比传统的基于角点(corner-based)的方法更有效[67]；
由统计采样技术生成的数据集非常适合用于学习策略；
用于分析不同技术节点的设备参数的AI/ML算法有助于优化设备参数并以非常高的计算速度估计参数产量，[68]实现了一种基于ML的Tikhonov正则化(TR)方法来分析在工艺(process)在GaN基高电子迁移率晶体管(HEMT high electron mobility transistors)中对$V_{TH}$的影响；
[69]提出了基于神经网络的对铁电场效应晶体管(FeFET ferroelectric field-effect transistor)的可变性分析(variability analysis)，将来自计量学(metrology)的极化图(polarization maps)形式的原始数据(raw data)作为输入，高/低阈值电压(high/low threshold voltage)、导通电流(on-state current)和亚阈值斜率(sub-threshold slope)被采样作为模型的输出
实验表明，与TCAD模拟相比，ML预测速度快106倍，准确率>98%；
[70]中提出了一种混合分析和DL辅助的MOSFET I-V(电流-电压 current-voltage)建模，其中为了对栅极长度(gate length)为12nm的GAAFET(栅极环绕晶体管 Gate-all-around transistor)技术的I-V特性进行建模，采用了具有18个神经元的3层神经网络

设计在7nm及以上的FinFET器件和电路的性能评估正变得具有挑战性，在制造之前准确估计这些设备的可靠性(reliability)也是另一个关心的问题[71]；
我们提出归纳式迁移学习(inductive transfer learning)[72，73]，作为一种基于现存技术节点(existing technology nodes)的知识对即将到来的(forthcoming)技术节点中的设备行为(device behavior)进行调查研究的技术

给定源域(source domain)$D_S$，以及对应的源任务(source task)$T_S$，目标域(target domain)$D_T$，以及目标任务(target task)$T_T$，迁移学习的目标是学习在$D_T$中的目标条件概率分布(target conditional probability distribution)$P(Y_T|X_T)$，学习的信息来自于$D_S$和$T_S$，其中$D_S\ne D_T$或$T_S\ne T_T$
在大多数情况下，可以假设有限数量的有标签的目标样本(labeled target examples)是可用的，其数量一般指数级小于有标签的源样本(labeled source examples)的数量
图4显示了关于开发一个使用迁移学习来分析即将到来的(upcoming)技术节点中的设备行为的学习系统(learning system)的建议方法
## 5.2 Gate Level
研究人员探索了AI/ML技术在门级电路设计和评估中的应用和发展

图5显示了在门级别的对统计感知电路模拟(statistical aware circuit simulation)的通用建模
RSM模型被广泛用于估计过程变异(process variations effects)对电路设计的影响，[74]详细分析了用RSMs估计工艺变化对电路设计的影响的发展；
[75]通过构建降维的(reduced dimensions)基于RSM的门延迟(gate-delay)模型，开发了一个统计的门内变异容忍单元库(library of statistical intra-gate variation tolerant cells)，这些被开发、优化好的的标准单元(standard cells)可用于芯片级(chip-level)优化，以实现关键路径的时序(timing of critical paths)；
[76, 77]中，RSM学习模型是通过统计的(statistical)实验设计(DoE design of experiment)和VLSI电路的门级库单元表征(gate-level library-cell characterization)的SSTA的自动选择算法的组合开发的，模型将阈值电压(threshold voltage)$V_{th}$和电流增益(current gain)$\beta$视为模型参数，以对功率、延迟和输出转换(output transitions)进行紧凑的晶体管模型表征(compact transistor characterization)；
[76]发现与蒙特卡洛(MC Monte Carlo)模拟相比，RSM和线性灵敏度(linear sensitivity)方法会将分析速度分别提高一个和两个数量级，尽管其代价是精度分别降低2%和7%；
[77]中，s-DoE在$3\sigma$分布的尾部平均误差为0.22%，而通过cadence encounter library characterizer(ELC)的敏感度分析的误差是它的十倍；

[78]还提出了一种变异感知(variation-aware)的统计实验设计方法(s-DoE)，用于预测静态随机存取存储器(SRAM)电路在制程变异性下(process variability)的参数产量(parametric yield)，他们的方法在$3\sigma$制程变异下比敏感度分析的精度要高大约两个数量级，同时CPU时间比MC模拟少10-100倍，文章中的案例研究证明了s-DoE在选择分布中感兴趣的区域(region of interest in the distribution)方面的优势，即在减少模拟次数(number of simulations)的同时提高准确率;
在类似的思路下，[79]为22nm的短接栅极(shorted-gate)和使用中心复合旋转设计的独立栅极FinFET(independent-gate FinFETs using a central composite rotatable disign)开发了基于RSM的泄漏分析模型(analytical leakage models)，用于估计制程变化时，FinFET标准单元中的泄漏电流(leakage curent)，他们的结果与在TCAD中使用2D横截面(cross-sections)进行的准MC模拟(quasi-MC simulations)结果一致

对模拟数据可能模式的探索(exploration of possible patterns in simulated data)和在电路设计的各个阶段数据的复用吸引了研究者的兴趣，
[80]提出一种健壮的查表方法(table-lookup method)用于估计门级别的电路泄漏功率(leakage power)，并且用贝叶斯接口(Bayesian interface BI)和NNs进行所有可能功率状态之间的切换，他们的模型使用NN进行使用模式识别，基于平均功耗值(average power consumption values)进行分类，分类目标是可能的状态(possible states)，其中心想法是使用电路可用的SPICE功率数据点的统计信息来表征(characterize)电路的状态转换模式(state transition patterns)和功耗值(power consumption values)之间的相关性，
这样的相关模式信息被进一步利用以预测电路的整个状态转换空间中的任何可见和不可预见的状态转换的功耗，
使用NN获得的估计误差总是表现出正态分布，且方差相较于基准曲线(benchmark curves)要小得多，
此外，估计误差随着簇的数量和NN复杂度的增加而减小，此外，训练和验证神经网络所需的时间与使用SPICE环境生成统计分布所需的计算时间相比可以忽略不计

[81]应用BI，提出了一种非线性分析时序模型(analytical timing model)用于体硅中(bulk silicon)、SOI技术以及非FinFET和FinFET技术中标准库单元的延迟和变化(slew of standard library cells)的统计表征，
模型采用了输出电容(capacitance)、输入变化率(slew rate)以及供电电压的有限组合作为输入
利用贝叶斯推理框架，他们使用来自于目标技术的一组极小的额外时序测量(an ultra-small set of additional timing measurements)，提取新的时序模型参数，实现模拟运行时的15倍的运行时加速且不影响精度，且精度比传统的查找表方法更好
他们使用ML来开发时序模型系数的先验(priors of timing model coefficients)，使用旧库(libraries)和稀疏采样来提供在目标技术中构建新库所需的额外数据点

多项式回归(polynomial regression)是另一个重要的分析式建模技术(analytical modeling technique)，
[82]提出了通过PR进行的统计泄漏估计(statistical leakage estimation)，在MCNC基准测试[83]上的实验结果表明，泄漏估计比[84]效率高五倍，和平均估计相比没有准确性损失，标准差大约在1%；
[85]提出了一个准确、低成本的Burr分布作为阈值电压(threshold voltages)相对均值$\pm 10$%变化时进行时延估计(delay estimation)的函数，样本是在90、45和22nm技术节点上生成的，来自MATLAB中的统计数据被应用于HSPICE仿真以获得时延变化，阈值电压和时延变化之间的关系被确定为四阶多项式方程，
Burr分布除了分布的均值和方差外，还将最大似然视为第三个参数，形成一个三参数的概率密度函数，因此Burr分布比正态分布多一个自由度，并且具有更低的误差分布

AI/ML预测算法也间歇性地应用于数字电路设计和模拟的制程-电压-温度(PVT process-voltage-temperature)变化感知的库单元表征(library-cell characterization)，
随着晶体管尺寸在深亚微米范围(deep sub-micrometer regime)内的急剧缩小(downscaling)，数字电路的准确性能建模(performance modeling)变得困难[87]、[88]；
为了解决深亚微米范围内数字电路性能建模的问题，Stillmaker等人提出了基于使用了在180纳米到7纳米技术节点范围内的预测技术模型(PTMs predictive technology models)的HSPICE模拟数据的多项式方程(polynomial equations)，用于拟合(curve-fitting)CMOS电路延迟、功率和能耗(enerty dissipation)测量(measurement)，
通过迭代式的功率、延迟和能耗测量实验(measurement experiments)，二阶和三阶多项式模型也被开发，模型达到了0.95的决定系数(coefficient of determination)(即R2score[91])；
在[92]和[89]中提出的缩放模型(scaling models)在比较不同技术节点和供应电压(supply voltages)下的器件(devices)时比经典缩放方法更准确

在[91]和[93]中分别报告了开发用于测量CMOS和FinFET数字逻辑单元(digital logic cells)泄露的PVT感知(process voltage temperature aware)的MPR和ANN模型；[91]还使用相同的MPR模型对总功率(total power)进行建模，开发的模型在HSPICE仿真下仅有小于1%的误差；
[94]报告了使用梯度提升(gradient boosting)算法进行PVT感知的泄漏功率(leakage power)和传播延迟(propagation delay)估计(PVT-aware estimationo)，与HSPICE仿真相比，在计算速度上提高了10000多倍，在估计误差上小于1%
这些特性化的库单元估计(characterized library-cell esitmations)可以用于估计复杂电路的整体泄漏功率和传播延迟，避免了传统编译器相对较长的仿真运行时间
[95]提出了使用回归算法估计基于MOSFET的数字电路(MOSFET-based digital circuits)的功耗；
基于PMOS的电阻负载反相器(RLI Resistive Load Inverter)、基于NMOS的RLI和基于CMOS的NAND门布局(gate layout)在90nm MOS技术中被用来创建数据集，提取的用于建模的特征向量包括电容、电阻、MOSFET的数量、它们各自的宽度和长度，以及各自布局的平均功耗，根据实验结果，Extra tree和多项式回归器在性能上优于线性、RF和DT回归器；
对复杂电路的分析需要基于GPU的电路分析，最近，在[96]中提出了一种工具，名为XT-PRAGGMA，它通过GPU加速的动态门级仿真和机器学习消除错误侵犯者(eliminate false aggressors)，并准确预测串扰引起的增量延迟(cross-talk induced delta delay)，与基于SPICE的仿真相比，它显示出1800倍的速度提升

在设计周期的早期阶段进行准确的产量估计(yield estimation)，可以对IC制造的成本和质量产生积极影响[97]、[98]，
对在扩展的PVT变异下(under expanding PVT variations)的设计在亚纳米尺度的VLSI电路的延迟和功率特性的全面分析，对于参数产量估计(parametric yield estimations)至关重要，如前所述，通过训练好的AI/ML算法，如PR、ANNs、GB和BI，可以进行非常准确的预测，其功率和延迟估计非常接近最可靠的HSPICE模型，将这些高效的ML模型整合到EDA工具中，用于库单元在晶体管级和门级的表征(characterization)，有助于在非常高的计算速度下对复杂VLSI电路进行性能评估(performance evaluation)，从而有助于产量分析，这些先进的计算EDA工具极大地改善了IC的周转时间(turnaround time)
## 5.3 Circuit Level
在制程变异下对VLSI电路进行统计特征描述(statistical characterization)是避免硅片重做(slicon re-spins)的关键，与门级类似，许多文献也报告了在电路级别对设计基于机器学习的替代模型的探索

[99]报告了使用NNs对VLSI电路的功率进行估计，训练好的NNs可以在不需要电路信息(如网络结构 net structures)的情况下，仅使用输入/输出(I/O)和单元数量(cell number)来估计功率，这种方法需要使用基准电路(bechmark circuit)的功耗估计结果来训练目标NN，有限的实验结果表明，该方法可以在特定网络结构(net structure)下以相当高的速度给出可接受的结果；
[100]讨论了一种基于内存活动计数器(memory activity counters)的新型功耗预测方法，该方法利用了功耗与潜在变量(potential varaibles)之间的统计关系，他们用于预测的机器学习模型包括支持向量回归(SVR)、遗传算法和NNs，他们表明，具有两层隐藏层和每层五个节点的NN是其中最佳的预测器，均方误差为0.047，他们提到：与硬件解决方案(hardware solution)相比，机器学习方法的成本更低、复杂性更小，运行时间更短；
[101]提出了一个高效的ANN模型，用于表征对泄漏功率的电压和温度感知的统计分析，训练好的晶体管级堆栈模型(transistor-level stack models)会用于电路泄漏估计，模型相较于蒙特卡洛统计泄漏估计，在运行时间上提高了100倍，并且在均值和标准偏差上误差小于1%和2%，模型的复杂度为$O(N)$，与现有的线性和二次模型(linear and quadratic models)[102, 103, 84, 104]相当

[105]提出了基于SVM的宏模型(macro models)，用于表征CMOS门的晶体管堆栈，与HSPICE计算相比，模型在估计泄漏功率上的平均运行速度加快了17倍；
[106]提出了一个混合替代模型(hybrid surrogate model)，该模型结合了ANN和SVM模型的预测，用于估计28纳米FDSOI技术中由于信号完整性感知路径延迟而导致的增量延迟(incremental delay due to the signal integrity aware path delay)，模型的最坏情况误差也小于10皮秒；
[107]中提出了一种使用RF对CMOS VLSI电路进行准确功率估计的方法，其性能优于NNs。在ISCAS'89基准电路(bechmark circuits)上的估计结果十分良好；
[108]提出了一种基于ResNet的，用于泄漏和延迟的快速高效优化的数字电路优化框架，在22nm金属栅高-K数字单元(Metal Gate High-K digital cells)上的结果表明，该方法相较于使用遗传算法，分别减少了36.7%和18.8%的延迟和泄漏
## 5.4 RTL Level
[109]详细讨论了制程变异性(process variability)对防护频带的(on guard bands)影响及其缓解措施(mitigation)；
[110]提出了一个监督学习模型，用于在RTL级别对位级静态时序误差(bit-level static timing error)进行预测，旨在减少在容错应用中的防护频带(guard band reduction in error-resilient applications)，
他们考虑了浮点流水线电路(floating-point pipelined circuits)，使用时序误差(timing error)特征化电路的行为，利用了Synopsys设计和Synopsys IC编译器分别作为前端和后端设计工具，
其中Synopsys Prime-Time被用于进行电压和温度缩放(scaling)，随后在Mentor Graphics ModelSim中进行后布局仿真(post-layout simulation)，并使用SDF回溯注释(back annotation)来提取位级时序误差信息，
在不同电压/温度角和未见过的工作负载(voltage/temperature corners and unseen workloads)下，逻辑回归模型的平均准确率达到95%，防护频带平均减少了10%；
[115]提出了在RTL级别的基于ML的功率估计技术，这些技术优于商业RTL工具[111, 112, 113, 114]，实验发现CNN的功率估计准确率要高于岭回归、梯度树提升和多层感知器；
[116]使用GNN从RTL模拟中进行平均功率估计，[117]提出了GRANNITE，在与传统的每周期门级仿真(per-cycle gate-level simulations)相比时，实现了超过18.7倍的速度提升

AI/ML策略可以扩展到电路和RTL级别，以构建宏单元模型(macrocell models)，用于参数产量(parametric yield)估计和优化，
使用ANN、CNN和深度学习技术构建的模型有助于复杂单元设计优化(complex cell design optimization)和功率延迟积(power-delay product)预测，因为它们对完整电路描述的依赖较小，
应用ML算法的一个关键瓶颈是为ML算法生成大数据，ML算法需要大量模拟数据才能准确建模输入输出之间的关系，这仅在某些数字电路级别(some levels)及其应用中是可能的，
生成对抗网络(GANs)的概念可以帮助解决这个问题，生成模型旨在估计训练数据的概率分布，并生成属于相同数据分布流形的样本，最近为回归任务提出的基于GAN的半监督方法架构加强了将GAN应用于数字电路回归任务的可能性，我们还需要探索不同的措施和技术来控制这些网络引入的量化误差(keep the quantization error introduced by these networks in check)
## 5.5 Post Layout Simulation
ML模型也促进了在重复的动态IR-drop(电压降)仿真中资源的有效利用，
[120]中提出的模型通过为违反电压降的单元实例构建小区域模型(small-region models for cells instances for IR-drop violations)，而不是为整个芯片构建全局模型，从而减少了训练时间，此外，ML模型还会在区域簇(regional clusters)上运行以提取所需特征并预测违反行为(violations)
在经过验证的工业设计上的实验(experiments on validated industry designs)表明，XGBoost模型在IR-drop预测性能优于CNNs，且每个ECO迭代只需要不到2分钟；
[121]开发了一种基于CNN的快速的设计独立的(design-independent)动态IR-drop估计技术，名为PowerNet；
[120]、[122]、[123]、[124]和[125]中则提出了设计依赖(design-dependent)的基于ML的IR-drop估计技术

[126]提出了一个名为Golden Timer eXtension(GTX)的基于ML的工具，用于签核时序分析(sign-off timing analysis)，使用该工具，他们尝试预测不同时序工具之间的时序裕度(timing slack)，以及跨多个技术节点的签核工具与实现工具之间的相关性(correlation between the sign-off tool and implementation tool across multiple technology nodes)，
由于STA签核(sign-off)对时序估计的不准确而导致产量不佳的问题(特别是在16纳米以下节点和低电压下)，可以使用支持先进工艺进行准确的时序校准(timing calibration)替代工具(surrogate tool)来改善

[127]讨论了在芯片设计和制造中的ML技术，特别是在22纳米以下工艺中解决制程变异对芯片制造的影响的ML技术，
作者讨论了用于pre-silicon HD、后硅变化提取 post silicon variation extraction、错误定位 bug localization的模式匹配技术技术，以及用于后硅时间调整 post-silicon time tuning学习技术；
[128]回顾了一些使用AI/ML方法的片上电网设计解决方案(on-chip power grid design solutions)，它详细讨论了使用概率、启发式和机器学习进行电网分析(power grid analysis)的方法，它进一步建议在设计阶段本身获得电网网络(power grid networks)的电迁移感知的老化预测(electromigration-aware aging prediction)是必要的

供电网络(PDNs power delivery networks)为IC的活性元件(active components)提供低噪声电源(low noise power)，随着供电电压(supply voltage)的降低，和电源电压(power supply voltage)的变化增加，系统的性能会受到影响，尤其是在更高的频率下，
电源噪声(power supply noise)的影响可以通过适当设计好的阻抗控制(impedance-controlled)的PDN来最小化，
PDN比率(PDN的实际阻抗与目标阻抗的比率 ratio)的增加会增加系统故障的概率，可以通过在板卡(board)和/或封装(package)上有效选择和放置解耦电容器(decoupling capacitors)将其最小化，[129]提出了一个快速的基于ML的surrogate-assited的元启发式(meta-heuristic)优化框架，用于解耦电容器的优化

此外，[130]提出了一个使用片上资源(on-chip resources)的低成本的基于机器学习的芯片性能预测框架，它能够预测芯片的最大工作频率(operating frequency)，用于速度分组(speed binning)，它的准确率相对于自动测试设备(Automatic Test equipment)超过90%，
在12纳米工业芯片上的实验结果表明，线性回归比XGBoost更适合用于该框架中，因为它具有更少的训练时间和模型尺寸，该方法还提出了一种传感器选择方法(sensor selection method)，以最小化片上传感器的面积开销(area overhead)；
[131]提出了RealMaps，这是一个用于实时估计全芯片热图(full-chip heatmaps)的框架，使用了LSTM-NN模型和现存的嵌入式温度传感器(embedded temperature sensors)和系统级使用信息(system-level utilization informatino)，通过2D空间离散余弦变换(DCT discrete cosine transform)来识别主要空间特征(dominant spatial features)的实验表明，只需要36个DCT系数(coeffieicnts)就能保持足够的准确性

图6总结了各项文献中提出的用于解决VLSI电路仿真的AI/ML算法

如前所述，AI和ML在电路仿真的各个阶段可以被整合到EDA工具和方法论中，以解决不同的统计/参数估计(statistical/parameter estimations)，包括漏电功率 leakage power、总功率 total power、传播延迟 propagation delay以及由于老化、产量和功耗引起的效应 effects induced due to aging, yield, and power consumption
# 6 AI in Architectures
随着AI/ML技术的发展，VLSI架构设计变得更加动态[132, 133]
高带宽高性能半导体设计的创新(innovations in high bandwidth and high performance semiconductor designs)和NN算法的进步为解决高级实时应用(real-time application)中的硬件实现(hardware implementation)挑战开辟了新的途径
过去几十年中，不同的架构激发了VLSI技术的进展，且大多数设计发展/改进都是由对边缘应用的需求(the need for edge applications)驱动的，这些应用需要高处理速度、高可靠性、低实现成本以及快速上市的时间窗口(time-to-market windows)

目前各项文献中提出的架构设计主要针对的应用领域包括图像处理和信号处理、语音处理、物联网(IoT)和汽车

本节将回顾存储器(memory)和脉动阵列(systolic array)中的VLSI架构修改(architectural modifications)，下一节则回顾在SoC级别的修改
## 6.1 Memory Systems
存储系统是计算系统的核心和主要组成部分之一，人们已经设计出了不同的可拓展的存储架构(scalable memory architectures)用于在各种物联网和嵌入式系统应用中实时处理机器学习算法

各种各样的AI应用都涉及了大数据集，并要求计算单元(computing unit)与存储器(memory)之间有更快的接口，为此，不同的存储架构被提出，以解决数据移动和处理问题：
[134]提出了在SRAM并行处理架构中的深度嵌入式计算(deep embedding of computation)，用于$256\times 256$图像中的模式识别，
他们的模型支持多行读取访问(multi-row read access)和模拟信号处理(analog signal processing)，而不降低系统性能，
他们的方法采用了两个模型：多行READ以及模拟绝对值差异和(SAD sum of absolute difference)计算，这种架构与传统架构不同的地方在于它不需要处理器和存储器之间的数据路径(data path)；
[132]提出了一个6T-SRAM阵列(array)，该阵列存储了一个ML分类器模型，这是一个超低能耗的用于图像分类的检测器，该阵列的原型(prototype)是一个$128\times 128$的SRAM阵列，以300MHz运行，在该阵列上运行的模型具有与在离散SRAM/数字乘累加(digital-MAC/MultiplyAccumulate)系统上运行的模型相当的准确性

[135]展示了一个健壮的深度存储内(deep-in-memory)ML分类器，该分类器使用随机梯度下降训练，基于使用了标准的16kB 6T-SRAM位元阵列(bit-cell array)的片上训练器(on-chip trainer)；
存储内计算(in-memory computing)是一种使用组装在阵列中的存储设备(memory device assembled in an array)来执行乘累加(MAC)操作的技术[136]，[133]研究了用深度存储内架构(deep in-memory architecture DIMA)替代常规的冯诺依曼架构，用于实现能量和延迟效率的ML系统级芯片(SoC)，
这种架构主要目标应用是需要进行大量的ML算法计算(computing heavy ML algorithms)的应用，如物联网和自动驾驶，
DIMA通过将传统的存储外围设备(memory periphery)植入(inplanting)计算硬件(computation hardware)，消除了分离计算和存储(seperate computation and memory)的需要，
该设计采用了具有不变的位元结构(changeless bit-cell structure)的6T SRAM，以保持存储密度不变(maintain the storage density)；
[137]中，在2T–1C配置(两个MoS2晶体管 MoS2 FETs和一个金属-绝缘体-金属电容器 metal-insulator-metal capacitor)中的MAC电路架构是人工神经网络中卷积运算的核心模块(core module)，该电路的存储部分类似于动态随机存取存储器(DRAM)，但由于MoS2晶体管的超低漏电流(ultralow leakage current)，具有更长的保持时间(retention time)；

[138]讨论了用于combined SVM训练和分类的并行数字(paralle digital)VLSI架构，在这个并行架构中，多层系统总线(multi-layer system bus)和多个分布式存储器(multiple distributed memories)充分利用了并行性；
在此之前，许多SVM在[139]、[140]和[141]中被开发和讨论，主要聚焦于90纳米技术节点(90-nm technology node)；
特别地，Wang等人在[138]中在45纳米节点上开发了SVM，使用商业GPU，它与数字硬件(digital hardware)上的传统SVM相比加速了29倍；

一部分计算任务可以在存储器内部执行，以解决数据移动问题，从而避免存储器访问瓶颈(memory access bottleneck)并显著加速应用性能，这类架构被称为存储内处理(processing-in-memory PIM)架构，
[142]提出了NNPIM，一种新颖的PIM架构，用于加速存储器内部的NN接口(NN's interface inside the memory)，其存储器架构结合了交叉存储器架构(crossbar memory architecture)以实现更快的操作，结合了优化技术以提高NN性能并减少能耗，以及结合了权重共享机制以减少NN的计算需求；
[143, 144]是一些重要的SOTA DRAM PIM架构

另一种正在发展的计算技术是近存储器处理(near-memory processing NMP)，NMP将存储器和逻辑芯片集成在3D存储包中(incorporates the memory and logic chips in 3D storage package)以提供高带宽；
[145]提出了一种用于训练DNN的近存储器架构，目的主要是加速DNN的训练而不是推理，其训练引擎NTX已经被用于大规模训练DNN(train the DNNs at scale)，
他们探索了RISC-V核心和NTX协处理器(coprocessor)，七倍减少了主处理器(main processor)上的开销，NTX与RISC-V处理器核心结合提供了一个共享的存储器空间，单周期可以访问128-KB紧耦合的数据内存(with single-cycle access on a 128-kB tightly coupled data memory)，
该架构采用混合存储立方体(hybrid memory cube)作为在数据中心训练DNN的存储器模块

[146]提出了一种通用的向量架构(general-purpose vector architecture)，用于将ML内核(kernel)迁移到近数据处理(near-data processing NDP)以实现高加速和低能耗，他们的架构在处理近数据(near-data)时，与高性能x86 baseline相比，在KNN上达到10x加速，在MLP上达到11x加速，在卷积上达到3x加速，
该工作还包括一个NDP内在函数库(intrinsics library)，支持基于大型向量的NDP架构验证(validating NDP architectures based on large vectors)；[147]提出了一个机器学习框架，为一个给定的应用，基于在给定工作负载下的应用性能排序，在基于HBM(High Bandwidth Memory 高带宽存储器)的NDP系统、基于HMC(Hybrid Memory Cube 混合存储立方体)的NDP系统和传统的基于DDR-4的NDP系统之中为应用预测合适的NSP系统

[148]为ML的存储内加速处理进行了K-means和KNN算法评估，提出了PRINS，这是一个采用电阻式内容可寻址存储器(resistive content addressable memory ReCAM)的系统，该架构既充当外存(storage)又充当大规模并行关联处理器(massively parallel associative processor)，
这种设计比冯·诺依曼架构模型在管理外存(storage)和主存(main memory)之间的瓶颈方面更有效，
这些算法在从主存获取数据的时间上要低于CPU、GPU和FPGA，
ReCAM比传统的CAM更有效，因为它实现了表达式的真值表(truth table of expression)的逐行执行，
与其它硬件相比，PRINS在K-means和KNN评估上功率效率和性能都更优

[149]对存储内计算(computing-in-memory CIM)在架构方面、尺寸、挑战和限制进行了调查；
[150]提出了一种健壮且面积效率(area efficient)高的CIM方法，使用6T代工厂位元(foundry bit-cells)，改善了点积计算的动态电压范围(dynamic voltage range)，能够承受位元$V_t$变化(variations)，并消除了任何读扰乱(read disturb)问题；
最新的在CIM芯片上的SOTA的工作在[151]；

根据他们的研究，基于SRAM的CIM解决方案可能是AI处理器的潜在选择，而不是基于NVM的(非易失性存储器)CIM，
基于NVM的CIM或记忆体设备(mem-risitive device)包括电阻随机存取存储器(RRAM resitive random acess memory)、磁阻RAM(magneto-restance RAM/MRAM)和相变存储器(phase-change memory PCM)[152, 153]

过去，[155]介绍了基于忆阻器(memristor-based)的DNN存储器内训练(training-in-memory)架构，命名为TIME，它减少了常规训练系统的计算时间，这种架构不仅支持干扰(interference)，还支持NN训练期间的反向传播和更新，它基于金属氧化物电阻(resistive)随机存取存储器，这提高了系统的性能和效率，
其主模块分为三个子阵列：全功能(full function)、缓存(buffer)和存储(memory)，全功能子阵列管理存储(memory)和训练操作，如干扰、反向传播和更新，存储子阵列管理数据存储(data storage)，缓存子阵列保存全功能子阵列的中间数据(intermediate data for the full-function array)，这种架构提高了深度强化学习和监督学习的能量效率(energy efficiency)

对硬件加速器(hardware accelerators)的全面调查超出了本文的范围。感兴趣的读者可以参考[156]，加速器和类似工作的综述[157, 158, 159]，然而，我们可以提供在架构级别为加快ML计算的不同设计方面的概述
## 6.2 Systolic Arrays
脉动阵列是数据流架构(data-flow architecture)的一个子集，它由几个相同的单元(cells)组成，每个单元都与最近的邻居单元局部连接，计算波(a wavefront of computation)在阵列中传播，其吞吐量与I/O带宽成正比

脉动阵列是细粒度和高度并行(highly concurrent)的架构，基于物联网的智能应用的进展已经指数级增加了对深度学习算法的需求，进而推动了基于脉动阵列架构的发展

[160, 161]提出了一个自动设计空间探索框架，用于在FPGA上实现基于CNN的脉动阵列架构(CNN-based systolic array architecture)，以实现高资源利用率和更高的速度，
他们利用分析模型(analytical model)提供深入的资源估计(resource estimation)和性能分析(performance analysis)，然而，FPGA上的脉动阵列实现受到深度神经网络稀疏性(sparsity)问题的很大影响；
研究人员早期就致力于解决这个问题，[162]提出了一种将稀疏卷积神经网络打包成更密集格式的方法，以便使用脉动阵列有效实现，然而，这些设计创建了不规则(irregular)的稀疏模型，未能利用脉动阵列的数据重用率特性(data-reuse rate feature)；
[163, 164]引入了结构化剪枝(structured pruning)，以克服与数据重用率相关的问题，产生与内存到脉动阵列的数据同步和节奏性流动(synchronous ans rhythmic flow of data)兼容(compatible)的DNN

进一步地，[165]提出了Eridanus，这是一种在将稀疏DNN模型实现在脉动阵列之前对零值进行结构化剪枝(structural pruning the zero-values)的方法，该方法检查所有滤波器之间的相关性(correlation among all the filters)，以提取局部密集块(locally-dense blocks)，其宽度(width)与目标脉动阵列的宽度相匹配，从而减少稀疏性问题；
类似地，对于FPGA平台上稀疏CNN模型的深度学习加速器的脉动阵列架构的优化是必要的，因为CNN滤波器矩阵中的零(zeros in the filter matrix of CNN)占用了计算单元(computation units)，导致效率次优，[166]提出了一种带有位图表示(bitmap representation)的稀疏矩阵打包方法，该方法压缩(condense)稀疏滤波器以减少脉动阵列加速器所需的计算

现存的文献中提出了许多对脉动阵列架构的修改，以解决特定应用问题，
[167]提出了一个Xilinx U50 Alveo FPGA卡上的脉动阵列作为MLP训练加速器，以解决在很短的时间内对大量流量日志(traffic logs)进行网络入侵/攻击检测，其每功耗的处理速度(processing speed per power consumption)比CPU快11.5倍，比GPU快21.4倍；
[168]提出一个结合了时序误差(timing error)预测和近似计算(approximate computing)的近似脉动阵列架构，以放宽(relaxing)MACs的时序约束(timing constraints)，该阵列在CIFAR-10图像分类上可以实现36%的能源减少，仅有1%的准确度损失；
[169]提出一种可重构(reconfigurable)的脉动环(systolic ring)架构，用于减少芯片上内存需求和功耗(on-chip memory requirement and power consumption)

矩阵乘法是大多数计算架构中的主要计算之一，
[170]提出了一种基于因式分解(factoring)和8进制乘法器(radix-8 multipliers)的新型脉动阵列，与提供相同功能的常规4进制(radix-4)设计相比，可以显著减少面积、延迟和功耗；
FusedGCN[171]是一种脉动架构，它计算三重矩阵乘法(triple matrix multiplication)以加速图卷积(graph convolutions)，它支持压缩的稀疏表示(compressed sparse representations)和不失去脉动架构规律性(regularity)的平铺计算(tiled computation)；
最近，[172]提出了一种基于对位传播加法器的部分因式分解(partial factoring of carry propagate adder)的混合累加器分解的脉动阵列(hybrid accumulator factored systolic array)，它在面积、延迟和功耗方面都有显著提高

加速设备的功能安全(functional safety)是另一个关键问题，由于GPU/TPU的制造缺陷(manufacturing defects)，而导致其加速的DNN在脉动阵列上的数据路径中显现出的故障(faults)可能导致功能安全违反(functional safety violation)，[173]对暴露于数据路径故障(faults in the data path)的DNN加速设备进行广泛的功能安全评估(functional safety assessment)

从SOTA工作方向来看，脉动阵列架构需要更加灵活，具有更多的数据流策略和多种数据传输模式，以应对未来深度神经网络深度的增加
# 7 AI at the SOC
为了整合深度学习，SoC架构已经进行了几项关键修改(modifications)，这些设计修改影响了通用SoC设计和包含了具有异构和大规模并行矩阵计算(heterogeneous and massive paralle matrix computations)的专用处理技术、创新的存储器架构以及高速数据连接的专用系统(specialized system)，

AI-SoC模型必须被压缩(compressed)，以确保它在移动、通信、汽车和物联网边缘应用中的受限的(constrained)存储架构下运行，
模型压缩可以通过控制剪枝(controlled pruning)，在不牺牲准确性的情况下进行，但是，功率、延迟和面积也是需要加以权衡的考量，因此，必须在内存和数据路径子系统(memory and datapath subsystems)上共同努力，谨慎选择架构修改

现场可编程门阵列(FPGA)是广泛使用的用于加速硬件上AI计算能力的可编程逻辑设备之一(programmable logic device)，由于其低成本、高能效(energy efficiency)、可重复使用性(reusability)和灵活性，FPGA成为了用于硬件加速器(hardware accelarators)的强大(robust)设备，而专用集成电路(ASIC)则最适合实现专用(specialized)应用

神经网络(NNs)受到生物学启发，执行并行计算，数字单元(digital units)，如DSP模型、浮点单元、算术逻辑单元(ALUs)和高速乘法器(high-speed multipliers)，可以有效地使用NN技术实现，NN对数字应用(digital applications)的基本优势在于，由于其操作时间几乎恒定(almost constant opeartion time)，无论电路中的位数如何增加，都可以高效地实现高速电路，利用NN计算中的并行性还提供了使用内部和片外存储器之间的平衡(balance between using internal and off-chip memory)

过去曾有许多用于SoC性能评估(performance evaluation)的机器学习和深度学习应用，
[174]使用LR开发了处理器的经验模型(empirical models)，以表征处理器响应(processor response)和微架构参数(micro-architectural parameters)之间的关系；
[175, 176]提出了通过回归分析(regression analysis)建立的功耗估计模型，用于准确预测微架构设计空间中微处理器应用的性能和功耗，
在[175]中提出的模型根据模型制定中模拟的样本数量(the number of samples simulated for the model formulation)，有效地对灵敏度(sensitivity)进行评估和建模，从大约220亿个设计点的设计空间中找到少于4000个足够样本，从而降低了仿真成本(simulation cost)，提高了分析效率(profiling efficiency)和性能，根据应用(application)的不同，50%-90%的预测实现了<10%的误差率(error rate)，报告的最大异常误差百分比(outlier error percent)约为20%-33%，
[176]使用层次聚类(hierarchical clustering)来确定十个考虑事件(considered events)中的最佳预测器(best predictors)，所提出的模型在应用于基于Intel XScale架构的PXA320移动处理器时，实际和估计功耗(estimated power consumptions)之间的平均估计误差约为4%

[177]对机器学习算法在未完全指定功能的逻辑综合中(logic synthesis of incompletely-specified functions)的应用进行了研究和比较分析

对于高速和能效(high-speed and energy-efficient)的计算系统，定期对SoC进行性能监控(performance monitoring)是必要的，然而，性能监控依赖于关键路径的准确采样(accurate sampling of critical paths)，而这些关键路径随着PVT(工艺、电压、温度)条件而大幅度变化，特别是在先进节点(advanced nodes)上，为了解决这个问题，[178]提出了一种基于机器学习的SoC实时(real-time)性能监控方法，该方法结合了未知的关键路径的物理寄生特性(physical parasitic characteristics)以及PVT变化

现存文献报告了几种针对特定应用的SoC架构，
[179]报告了用于多媒体内容分析(multimedia content analysis)的MLSoC(在TSMC 90纳米CMOS技术中实现)；
[180]提出了一个完整的端到端双引擎SoC(end-to-end dual-engine SoC)，用于面部分析，与SOTA系统相比，能效(energy efficiency)提高了超过2倍，这种效率来自于第一级(in the first level)中二元决策树的层次实现(hierarchical implementation of the Binary Decision Tree)，以及下一级中更耗电的(more power hungry)CNN，CNN仅当需要时才触发

机器效率监控(machine efficiency monitoring)对于实现高生产力、减少故障和降低成本至关重要，
[181]中提出了一个基于SoC的工具磨损监控(tool wear monitoring)系统，该系统结合了信号处理(signal processing)、深度学习和决策制定，系统中，从三轴加速度计(three-axial accelerometer)和MEMS麦克风收集到的数据经过传感器融合，结合使用相机在不同场景下测量工具侧磨损(tool flank wear)的数据，被输入到CNN以检测任何加工变化(machining variation)；
[182]提出了极端学习机(Extreme learning machine ELMs)，用于提高大型数据处理(large data processing)的计算效率和性能的NN架构；
[183]中提出了一种低成本的实时神经形态硬件系统(neuromorphic hardware system)，该系统本质是具有基于片上三元组(on-chip triplet-based)的奖励调节的尖峰时序依赖可塑性(R-STDP reward-modulated spike-timing-dependent plasticity)学习能力的极端学习机(ELM)

对SoC进行全面的时序分析(timing analysis)也是满足设计规范的必要条件，[184]对SoC物理设计(physical design)进行了基于集成学习的时序分析，它使用了许多具有不同参数设置的平面图文件(floor plan files with different parameter settings)，然后使用来自于Synopsys IC Compiler工具的松弛时间(slack time)作为标签，训练监督学习算法，其想法是在早期阶段基于预测结果对物理设计流(physical design flow)进行反馈，以便修改不当的楼层平面(improper floorplan)；
基于Bigram的多电压(mulit-voltage)感知的时序路径松弛分歧(timing path slack divergence)预测[185]采用了分类和回归树(CART)方法，实验结果显示，在预测单元延迟(cell delays)和端点时序松弛(endpoint timing slack)方面，其准确率达到了95%到97%

能够提供工业级芯片设计(industrial-quality chip design)的CAD工具必须能为最佳PPA(性能、功率、面积)进行调节，
[186]中提出了一种全面的方法，该方法涉及在线和离线(online and offline)机器学习方法共同工作，以调整工业设计流程(for industrial design flow tuning)，该工作突出了SynTunSys(STS)，这是一个在线系统，它优化设计(optimizes designs)并为一个推荐系统生成数据，该推荐系统执行离线训练和推荐；

[187]提出了在基线流水线RISC处理器(baseline pipelined RISC processor)中增加一个额外的ML流水线阶段(additional ML pipeline stage)，以将指令(instructions)分类为传播延迟类别(propagation delay classes)，并增强时间资源利用(temporal resource utilization)；
[188]中展示了在真实设计流程中部署基于ML的SoC设计的挑战，该工作强调了由于数据有限(limited data)、不充足的开源基准和数据集、基于EDA工具的数据生成以及合成数据生成所带来的挑战(insufficient open-source benchmarks and datasets, EDA tool-based data generation, and Synthetic data generation))

AI-SoC架构在紧密耦合(tightly coupled)的处理器和存储器架构方面的应用正处于开始阶段，要达到在边缘应用中模仿人脑的全能力，还有很长的路要走
# 8 AI in Physical design
超大规模集成电路(VLSI)物理设计存在许多需要多次迭代(iterations)才能收敛的组合问题(combinatorial problems)，半导体技术的发展(scaling)增加了这些设计问题的复杂性，包括复杂的设计规则(design rules)和面向制造设计(DFM design for manufacturing)的约束(constraints)，这使得实现最优解(optimal solution)变得具有挑战性[189]

传统上，这些问题/违规是手动(manually)检测和修复的，然而，传统的在先进节点上实现设计封闭(design closure to advanced nodes)的手动方法已经难以满足市场窗口，此外，在设计流程的后期阶段设计质量和制造过程对早期阶段的变化(changes)是极其敏感的，故早期阶段的设计变化会增加周转时间(turnaround time)并延缓设计封闭(design closure)，
因此，早期阶段有效设计(valid design)的预测至关重要，尤其是在当前技术节点(technology node)上

机器学习和模式匹配技术在物理设计的多个阶段提供了合理的抽象和结果质量(quality of results)，它们充当连接每个步骤的桥梁，并提供有价值的反馈以实现早期设计封闭(early design closure)

广义上，物理设计可以分为四个阶段：划分(partitioning)、布局规划和放置(floor planning and placement)、时钟树综合(clock tree synthesis)以及布线(routing)，我们将回顾研究人员在这些阶段提出的人工智能/机器学习方法
## 8.1 AI for Partitioning, Floor planning and Placement
划分是超大规模集成电路(VLSI)物理设计的主导领域之一，划分的主要目标是将复杂电路划分(divide)为子块(sub-blocks)，单独(individually)设计它们，然后分别组装(assemble)它们以降低设计复杂性(design complexity)

布局规划和放置是设计流程中对设计质量(design quality)和设计封闭(design closure)至关重要的其他阶段，
布局规划将划分中的逻辑描述和物理描述进行映射(maps the logic description from partioning and physical description)，以最小化芯片面积和延迟(minimize chip area and delay)，布局规划的目标是安排芯片的子块(arranging the chip's sub-blocks)，并决定I/O垫(I/O pads)、电源垫(power pads)以及电源(power)和时钟分配(clock distributions)的类型和位置(type and location)，
放置确定电路布局(circuit layour)中逻辑门(单元)(logic gates/cells)的物理位置(physical locations)，其解决方案在很大程度上影响后续的布线(routing)和后布线封闭(post-routing closure)，全局放置(global placement)、合法化(legalization)和详细放置(detailed placement)是放置的三个阶段，
全局放置提供了标准单元(standard cells)的大致位置，合法化基于全局放置解决方案移除任何设计规则违规和重叠(remove any design rule violations and overlaps)，详细放置逐步提高整体放置质量(incrementally improves the overall placement quality)[190]

在[191]中，芯片布局规划被建模为一个强化学习问题，它基于RL，构建了一个基于边的(edge-based)图卷积神经网络架构，能够学习丰富的且可迁移的芯片表示(chip representations)，该方法被用于设计下一代谷歌人工智能加速设备，并已显示出每新一代能够节省数千小时人力的潜力；
[192]提出一种基于机器学习的方法，在给定网表(netlist)、约束(constraints)的情况下，在布局规划阶段预测SRAMs后P&R(放置和布线 place and route)松弛度(post P&R slace of SRAMs)，在28nm代工厂FDSOI技术上的测试的布局上下文(floor context)，显示最坏情况下误差为224ps；
[193]提出用回归方法以快速评估在布局规划期间每个宏单元放置中(in each macro placement)的路由拥塞(routing congestion)和半周长线长(half-perimeter wire length)，他们探索了使用不同回归技术——LR、DTR(决策树回归器)、Booster DTR、NN和泊松回归的解决方案，其中，DTR表现更好；[194]提出的多芯片模块(MCM multi-chip module)有许多小芯片集成到一个封装(package)中，并由互连连接(joined by interconnects)，由于搜索空间稀疏，多芯片划分(multi-chip partitioning)十分困难，[195]提出了一种用于在MCM中划分的RL解决方案

关于放置，数据路径的高规律性(high regularity of data path)对于放置期间的紧凑布局设计(compact layout design)至关重要，然而，数据路径经常与其他电路混合(mixed with other circuits)，如随机逻辑(random logic)，对于嵌入许多数据路径的设计(designs with many embedded data paths)，适当提取和放置它们(extract and place them appropriately)以实现高质量放置至关重要，
现有的分析放置技术(analytical placement techniques)次优地处理它们[196]，由于技术限制，现代放置器(placers)未能有效处理数据路径，在这种情况下，ML发挥着关键作用
[197]提出PADE，用于混合了随机和数据路径电路的大规模设计中的数据路径自动提取的能力(automatic datapath extraction)，PADE通过分析全局放置网表(global placement netlist)提取有效特征，以预测数据路径的方向(direction)，PADE结合SVM和NN用于簇分类(cluster classification)和评估，在混合基准测试(hybrid benchmark)上的实验结果了该方法在半周长(half-perimeter)和Steiner树线长(Steiner tree wire lengths)上的改进；
[198]提出了一种基于连接向量(connection vector-based)和基于学习(learning-based)的数据路径逻辑提取策略(data path logic extraction strategies)，它将SVM和CNN用于提取，在MISPD 2011数据路径基准测试上的结果表明，SVM和CNN在分类数据路径(data path)和非数据路径(non-data path)部分方面表现相等

芯片放置(chip placement)是芯片设计周期中最耗时和最复杂的阶段之一，AI可以提供必要的手段来缩短芯片设计周期，形成硬件和AI之间的共生关系
为了减少芯片放置所需的时间，[199]提出了一种能够从过去的经验中学习并随着时间改进的方法，作者将放置作为RL问题，并训练了一个智能体，将芯片网表的节点(nodes of a chip netlist)放置在芯片画布(chip canvas)上，以在遵守放置密度(placement density)和路由拥塞(routing congestion)施加的约束的情况下，优化最终的PPA(功率、性能和面积)，该方法中，RL智能体(策略网络 policy network)顺序放置宏单元(macros)，一旦所有宏单元放置完毕，力导向方法(force-directed method)产生标准单元的粗略放置(rough placement of standard cells)，随着在更多芯片网表上获得经验，这个RL智能体在芯片放置上会变得更快、更好，该方法可以在不到6小时内生成放置(generate placement)，而最强的基线需要人类专家介入，整个过程可能需要数周；
[200]提出了量子(quantum)机器学习技术，为VLSI放置问题实现更快、更优同时错误率低的解决方案，它使用变分量子特征求解器(variational quantum Eigen solver VQE)[201]方法实现了完整的放置，在两个电路上进行了测试：一个玩具电路(包含八个门 gates)、一个取自MCNC基准测试套件(benchmark suite)[83]的名为“Apte”的电路；
[202]研究了用于放置和时序分析(placement and timing analysis)的GPU加速，通过利用机器学习技术与异构并行性(heterogeneous parallelism)，实现了百万门级设计(million-gate design)的500倍速度提升

放置和布线(placement and routing)是两个高度依赖的物理设计阶段，它们之间的紧密合作对于优化芯片布局非常推荐。传统的放置算法使用引脚延迟或通过线长模型估计可布性，由于制造约束的增加和复杂的标准单元布局，永远无法实现它们的目标[203]。在[204]中，提出了一个基于深度学习的模型（基于CNN），以快速分析详细路由器将遇到的路由难度，估计放置的可布性。在[205]中，提出了一个基于CNN的RL模型，用于详细放置，保持当前放置的最佳可布性。提出了一个通用的放置优化框架，以满足后布局PPA指标，并具有小的运行时间开销[206]。给定初始放置，无监督学习发现后路PPA改进的关键单元集群，从时序、功耗和拥塞分析中。随后进行了有向放置优化。该方法在5nm技术节点上的工业基准测试中得到验证。 在[207]中，提出了一个机器学习模型，用于预测各种物理布局因素对最小有效块级面积的敏感性，与常规设计技术共同优化（DTCO）和系统技术共同优化（STCO）方法相比，提供了100倍的速度提升。这项研究建议从他们在各种ML算法上的实验中，使用自举聚合和梯度提升技术进行块级面积敏感性预测。进一步地，[208]引用了MAGICAL（从网表到GDSII的全自动化模拟布局，包括自动布局约束生成、放置和布线），一个开源的VLSI放置引擎。Magical 1.0是开源的。[209]通过探索不同的楼层平面替代方案和放置风格，提出了自动化的楼层规划。 RL被提出作为IC物理设计的最佳解决方案，因为它不依赖于任何外部数据或先验知识进行训练，而可以根据代理通过设计空间探索产生的不寻常解决方案。一些RL方法用于放置优化[210, 211]。