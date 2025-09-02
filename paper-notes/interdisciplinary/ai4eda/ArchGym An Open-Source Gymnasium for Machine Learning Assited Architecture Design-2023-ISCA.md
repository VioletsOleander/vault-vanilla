# Abstract
使用ML进行设计空间探索有几大挑战：首先，难以确定最适合的ML算法；其次，在目前的方法之间评估性能和样本效率之间的权衡尚无定论(assessing the trade-offs between performance and sample efficiency across these mothods is inconclusive)；最后，缺乏一个整体的框架对各种方法进行公平、可复现和客观比较(fair, reproducible, and objective comparision)，因此阻碍了ML辅助的架构设计空间探索(architecture design space exploration)的进展

我们提出ArchGym，一个开源的gymnasium，一个易于拓展的框架，将各种搜索算法与架构模拟器联系(connect a diverse range of search algorithms to architecture simulators)

我们使用ArchGym，在定制内存控制器(custom memory controller)、深度神经网络加速设备(deep neural network acclerators)以及针对AR/VR工作负载的定制SoC上，评估了多种朴素的和领域特定的搜索算法，总共包括了超过21K次实验
结果表明，在样本数量不受限制的情况下，如果可以全面地调节超参数，ML算法都可以满足用户定义的目标规范(equally favorable to meet the user-defined target specification)，并不会有一种方法一定优于另一种方法
我们用“超参数抽奖(hyperparameter lottery)”来描述在提供了精心选择(meticulously selected)的超参数下，一个搜索算法能搜索到最优设计的相对可能的机会(relative probable chance)

此外，ArchGym中数据收集和聚合的方便性(the ease of data collection and aggregation)可以促进在ML辅助的架构设计空间探索方面的研究
在我们的代理开销模型(proximal cost model)中，ArchGym的RMSE为0.61%，说明ArchGym可将模拟时间减少2000倍
# 1 Introduction
摩尔定律的停滞和人们对更高计算效率的追求促进了对极致的领域特定的硬件定制(domain-specific hardware customization)，但该方向有几大挑战：首先，整个计算栈中数量巨大的设计参数(the immense number of design parameters across the compute stack)导致了搜索空间的组合爆炸；搜索空间内大量的不可行的设计点(infeasible design point)进一步复杂了优化；此外，应用环境的多样性以及计算栈中搜索空间的独特特征对传统优化方法的性能提出了挑战
为此，工业界和学界都转向ML驱动的优化以满足严格的领域特定要求(domain-specific requirements)，而尽管之前的工作已经展示了ML在设计中的优势，但可复现的基准的缺乏(the lack of reproducible baselines)阻碍了不同方法之间的客观公平的比较

首先，选择最合适的算法并估计超参数的作用时，它们的效果(efficacy)仍然是不能确定的
目前存在许多ML/启发式方法，从随机游走到RL都可以用于设计空间探索(DSE/design space expolration)，例如，目前有利用贝叶斯/数据驱动离线/RL优化方法进行DNN加速设备的参数探索，这些方法都相对于它们选择的基准有了高的性能提升，但目前尚不清楚这些改进是否实际是因为优化算法或超参数的选择(the choice of optimization algorithms or hyperparameters)
因此，为了确保可复现性和促进对ML辅助的架构DSE的发展，有系统的基准测试方法(systematic benchmarking methodology)是十分重要的

其次，随着模拟器成为架构创新的支柱(backbone)，我们越加需要在架构探索中处理准确率、速度和成本之间的权衡，
而取决于底层的建模细节(underlying modeling details)(例如，周期准确/cycle-accurate -> 事务级别的模拟器/transaction-level simulator -> 解析模型/analytical model -> 基于ML的代理模型/ML-based proxy model)，不同的模拟器估计出的准确率以及它们的性能估计速度之间的差异十分大，
其中，而解析模型或基于ML的代理模型由于抛弃了低层次的细节(low-level details)，一般都很灵活，但通常有较大预测误差，
此外，由于商业许可，从模拟器中采集的(collected from a simulator)样本数量可能有严格限制，
总的来说，这些约束表现出明显的性能与样本效率的权衡(performance vs. sample efficiency trade-offs)，影响了架构探索优化算法的选择，因此，如何在这些约束下系统地比较各种ML算法的有效性(effectiveness)是一个挑战

最后，将DSE的结果呈现为有意义的成果(例如数据集)对于深入了解设计空间是至关重要的(rendering the outcome of DSEs into meaningful artifacts such as datasets is critical for drawing insights about the design space)，
许多ML算法在不断出现，一些ML算法需要数据才能发挥作用，例如在RL领域中就已经出现许多算法，例如PPO、SAC、DQN、DDPG，用于解决一系列问题，同时也有采用离线(offline)RL方法的工作，以分摊数据收集的开销(amortize the cose of data collection)，
我们最后的目的是要确定如何分摊进行架构探索的搜索算法的开销(how to amortize the overhead of search algorithms for architecture exploration)，而目前尚无工作系统地研究如何在利用探索数据的同时保持对底层搜索算法的不可知(leverage exploration data while being agnostic to the underlying search algorithm)

我们提出ArchGym以解决以上挑战，ArchGym是一个开源gymnasium，用于分析和评估各种ML驱动的设计优化方法(methods for design optimization)，ArchGym强调在搜索算法和性能模型(performance models)之间使用相同的接口(interfaces)(例如架构模拟器或代理开销模型)，使得各种搜索算法可以有效映射(enabling effective mapping of variety of search algorithms)，此接口还支持了对用于比较的基准(baselines for comparision)和搜索算法的基准测试(benchmarking of search algorithms)的开发，
此外ArchGym提供了一个infrastructure，以可复现和可访问的方式收集和共享数据集(collect and share datasets in a reproducible and accessible manner)，推进对底层设计空间的理解

我们进行了超过21600次的实验，对应在四个结构设计空间探索问题上的大约15亿次模拟，四个结构设计空间探索问题分别是：
(1) DRAM内存控制器(DRAM memory controller)，
(2) DNN加速器(DNN accelerator)，
(3)系统芯片设计(SoC design)，
(4) DNN映射(DNN mapping)，
我们使用五种常用的搜索算法，并全面地扫描它们的相关超参数
评估结果显示，这些算法的最终表现有较大方差，例如，在不同的搜索算法分别针对DRAM内存控制器、DNN加速器、SoC设计时，我们观察到表现分别为90%，20%，40%的统计差异(statistical spread in performance)
包括离群值在内，每种算法都会产生至少一种配置(configuration)，在不同的设计空间中实现最佳目标(best objective arcoss different design spaces)

算法表现的差异主要是超参数选择的结果，而最优超参数的选择取决于ML算法的和底层领域的特点，但常用的超参数调节技术会带来更多复杂度，因此，对于一个架构DSE问题，选择最优的超参数仍是一个非平凡的问题(non-trivial)，就像赢得彩票一样，需要大量的资源
“超参数抽奖”就用于描述一个算法达到最优解(称满足所有用户定义的标准的结果为最优解，例如满足延迟$<L$)的相对机会

和常识相违背的一点是，我们的分析表明评估过的搜索方法在不同的设计空间探索问题中都是等价优秀的(equally favorable)

我们工作主要贡献总结为以下四点：
- 我们设计了用于ML辅助架构DSE的开源框架ArchGym，以对搜索算法进行系统的评估和客观的比较(systematic evaluatoin and objective comparision)
- 框架评估的结果表明，与常识不同的是，所有评估过的搜索算法对于架构DSE问题都是同等优秀的，没有某个算法(例如RL或贝叶斯方法或GA)必然是更好的
- 我们认为要公平地比较ML算法，需要将超参数优化的开销纳入考虑，例如对硬件模拟器样本的访问(acess to hardware simulator samples)，没有恰当的评估度量，算法的有效性是有误导性的
- 我们发布了一组精心挑选的数据集，这些数据集对于构建高保真代理开销模型(high-fidelity proxy cost models)非常有用，与传统的周期准确模拟器相比，这种代理开销模型的速度会快上多个数量级，缓解了架构探索中速度和精确度之间的权衡
- 基于增加数据集大小可以提高准确率的直觉，我们展示了通过ArchGym 增加多样性(adding diversity)，可以将均方根误差(average root mean square error)降低42倍
# 2 Background And Related Work


