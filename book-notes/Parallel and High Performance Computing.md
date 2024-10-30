# Introduction To Parallel Computing
## 1 Why parallel computing
并行计算是对于一个实例的多个操作的同时执行，并行计算的前提条件是并发性(concurrency)，或者称其为可能并行性，并发性意味着在系统资源可用的条件下，对实例的操作顺序可以是任意的
额外地，如果这些操作是可以同时发生的，可能并行性就成为了并行性

算法是解决计算问题的一系列解决步骤，算法可以是串行的，也可以是并行的

### 1.1 Why should you learn about parallel computing
由于微型化、时钟频率(机器指令被执行的速率，单位 MHz)、功率(单位 瓦特)、温度等的限制，处理器的串行处理性能存在上限

自2005年开始，CPU的单线程性能、CPU时钟频率、CPU功率的增长趋势放缓，CPU的核心数开始增加

处理器的理论性能与时钟频率和核心数的乘积成正比，显然提高CPU的理论性能可以从时钟频率入手，也可以从核心数入手(并行性)

如今常见的两种硬件层面的并行特性：
- 超线程(hyperthreading)
	由Intel在2002年首次引入
	它使得单个的CPU物理核心对于OS看来是两个逻辑核心，即将一个CPU物理核心虚拟成了两个CPU逻辑核心
	实现了超线程技术的CPU物理核心中，每个线程有自己的PC寄存器、指令寄存器、条件码寄存器，但是共享译码器和ALU(	如果各线程也拥有自己的译码器和ALU，那实际上就是两个物理核心了)
	当线程A的指令停顿，线程B可以利用空闲的译码器和ALU，以提高效率
	注意这里的线程概念和OS层面的线程概念完全不同，这里是硬件层次的概念
- 向量处理器(vector processors)
	大约在2000年出现于商业级处理器
	举例来说，一个位宽为256位的向量处理器(或向量单元)一次可以执行4个64位(双精度)指令或8个32位(单精度)指令

对于一个16核，采用了超线程技术以及采用了256位的向量单元的CPU，它的理论处理能力是：$16核 × 2超线程 × (256位宽的向量单元)/(64位双精度) = 128路并行$
一个只允许串行执行指令的应用程序在占用CPU运行时只能利用该CPU$1/128 = 0.8\%$的性能

#### 1.1.1 What are the potential benefits of parallel computing
- faster run time with more cmpute cores
- larger problem sizes with more compute nodes
- energy efficiency by doing more with less
	并行计算使得硬件更快完成任务从而进入睡眠模式节能
	并行计算使得我们可以通过并行多个较慢较节能的处理器达到较耗能的单个处理器的性能
	
	能耗估计：$$P = (N\ Processors)\times(R\ Watts/Processors)\times(T\ hours)$$其中$P$为能耗，$N$为处理器数量，$R$为热设计功率(硬件正常运行时的功率/所产生的最大热量)，$T$为运行时间
	
	例如，Intel 16核 Xeon E5-4660的热设计功率是120W，则用20个该处理器运行24小时，能耗为$$P = (20\ Processors) × (120\ W/Processors) × (24\ hours) = 57.60 kWhrs$$
	例如，NVIDIA Tesla V100 GPU的最大热设计功率为300W，则4个V100运行24小时，能耗为$$P = (4\ GPUs) × (300\ W/GPUs) × (24\ hrs) = 28.80 kWhrs$$
	GPU的功率一般高于CPU，但对于计算密集型的任务，少量的GPU的性能即可等同于大量CPU的性能，因此能耗反而是相对减少的
	但采用像GPU这样的设备来减少能耗依然需要一个前提，即任务是可并行的
- parallel computing can reduce costs
	对于大型HPC系统，能耗开销往往三倍于硬件成本开销

#### 1.1.2 Parallel computing cautions
首先让一个应用能理想运行，再去挖掘该应用的并行潜力，是否并行化取决于要投入的成本和成功后的收益

### 1.2 The fundamental laws of parallel computing
计算并行计算的加速程度(speedup)的一般公式是：
$$SpeedUp(N) = \frac {t_1} {t_N}$$
其中$t_1$是只用一个处理器解决问题所需要的时间，$t_N$是用$N$个处理器解决同样的问题所需要的时间
理想状态下的加速程度是线性的，即$SpeedUp(N) = N$
#### 1.2.1 The limit to parallel computing: Amdahl's Law
阿姆达尔在1967年提出了阿姆达尔定律，该定律描述了在处理器数量增加的情况下，解决一个固定大小的问题的加速程度/加速比(speedup)

阿姆达尔定律：
$$SpeedUp(N) = \frac 1 {S + \frac P N}$$
其中$P$是可以被并行化执行的代码所占的比例，$S$是只能串行执行的代码所占的比例，显然$P+S = 1$

阿姆达尔定律说明了无论我们并行的路数($N$)有多大，最后的加速程度仍然要受到只能串行执行的代码所占的比率($S$)的限制
因为很显然$$SpeedUp(N) = \frac 1 {S + \frac P N}$$作为$N$的函数，在$N\to \infty$时取到最大值，最大值为$1/S$，即加速程度的上限由$S$决定

**定义**：在固定问题的规模、增加计算资源的情况下，解决问题需要的时间的变化，这样的放缩称为强放缩(strong scaling)
强放缩能力即指在强放缩的情况下的加速能力，显然阿姆达尔定律就是描述强放缩能力的定律，它指出了强放缩能力的上限由$S$决定

#### 1.2.2 Breaking through the parrallel limit: Gustafson-Barsis's Law
阿姆达尔定律中隐含了一个假设：$P$是与$N$无关的，也就是说，问题的规模是固定的，不论处理器的数量有多大

但是古斯塔夫森和巴西斯在实验中发现这个假设并不符合实际，在实践中，大多数情况是：随着处理器数量的增加，问题的规模也随之增大，并且，往往问题规模的增大是在问题中的可以并行处理的部分，即$P$随着$N$增大而增大 

古斯塔夫森和巴西斯在实验数据的基础上，作出了一个假设：问题中可以并行处理的部分的规模随着处理器数量的增加而线性增加

在这一假设下，$P$与$N$呈线性相关，$S$与$N$无关，当有$N$个处理器时，问题的规模实际是$S + N\times P$
要解决这一问题，对于$N$个处理器，$S$部分仍然只能由$1$个处理器处理，需要$S$时间完成，而$N\times P$部分，可以用$N$个处理器并行处理，只需要$P$时间完成
对于$1$个处理器，$S$部分由一个处理器处理，需要$S$时间，$N\times P$部分仍由一个处理器处理，需要$P$时间
因此，加速比为：
$$\begin{aligned}
SpeedUp(N)&=\frac {S + N\times P}{S+P} \\
&= S+ N\times P \\
&=N - (N-1)S
\end{aligned}$$
这个公式也说明了，解决同样一个问题(只不过现在这个问题规模更大)，理论上使用$N$个处理器的效率会比仅使用$1$个处理器的效率快$S+N\times P$倍，这个倍数是随$N$线性增长的

如果要简单起见，也可以直接认为假设是：问题的规模随着处理器数量的增加而线性增加，比如，在仅有1个处理器时，需要处理1个问题，而在有$N$个处理器时，需要处理$N$个问题，因此$N$个处理器各自处理自己的问题，处理完$N$个问题需要的时间是$S + P = 1$，而这里要明确的是，如果让1个处理器处理$N$个问题，问题中的$S$部分是不需要重复做的，只需要完成一次即可(这也是加速比会小于$N$的原因)，因此，1个处理器处理$N$个问题所需要的时间是$S + N\times P$，这样计算，加速比仍然是$S + N\times P$，可以发现，相较于阿姆达尔定律固定了整个问题的规模，古斯塔夫森定律则是固定了平均到每个处理器的问题的规模，由此得到弱放缩的定义：
**定义**：在固定平均每个处理器的问题的规模、增加计算资源的情况下，解决问题所需要的时间的变化，这样的放缩称为弱放缩(weak scaling)

强放缩和弱放缩都很重要，它们对应的是不同的用户场景(user scenarios)

放缩性(scalability)常指是否存在更多的并行性(parallelism)以在软硬件中添加，以及更多的并行性带来的提升的幅度是否存在上限(an overall limit)
强放缩(阿姆达尔定律)的情况下，运行时间的放缩性(run-time scalability)显然比弱放缩(古斯塔夫森定律)的情况下要强

除运行时间的放缩性外，考虑内存的放缩性(memeory scalability)
复制性数组(replicated array，简称为$R$)指需要对每个处理器都要复制一份的数据集(dataset)
分布式数组(distributed array，简称为$D$)指需要分割和拆分到每个处理器上的数据集(dataset)

例如，在游戏模拟中，
100个角色信息可以平均划分给4个处理器处理，每个处理器处理25个，则角色信息数据就是分布式数组
1个游戏棋盘的信息则需要每个处理器都复制一份，则游戏棋盘信息数据就是复制性数组
如果是在弱放缩的情况下，问题的大小和处理器的数量成正比，
不妨设1个处理器时，问题大小为100MB，则两个处理器时，问题大小为200MB，四个处理器时，问题大小为400MB
![[PHPC-Fig1.6.png]]
如果数据是分布式数组，则符合古斯塔夫森定律讨论的弱放缩的情况，在这种情况下，总的内存需求和处理器数量是成线性增长的
如果数据不是分布式数组，则问题的大小随着处理器线性增大，但数据要为每个处理器都拷贝一份，总的内存需求和处理器数量成平方关系
因此分布式数组限制了放缩性，如果没有足够的内存，则无法利用多处理器的并行计算来提高效率

受限的运行时间放缩性(limited run-time scaling)意味着工作运行得慢
受限的内存放缩性(limited memeory scaling)意味着工作无法运行

计算密集型作业(computationally intensive job)的一种观点是，内存的每个字节都会在每个处理周期中都会被访问，因此运行时间是内存大小的函数，减少内存大小必然会减少运行时间，因此，并行性的最初重点应该是随着处理器数量的增长而减少内存大小

### 1.3 How does parallel computing work
![[PHPC-Fig1.7.png]]
作为开发者，我们直接负责的是应用软件层，
要将并行性引入一个算法，我们通过编程语言的选择和软件接口的使用以使用底层的硬件进行并行计算，同时我们需要决定如何将作业划分成一个个并行单元(parallel units)

开发者不负责编译器层、OS层和硬件层，但现有的硬件特性也会影响我们的决策和并行策略，开发者可以采用的并行方法(parallel approaches)有
- 基于进程的并行(Process-based parallelization)
- 基于线程的并行(Thread-based parallelization)
- 向量化(Vectorization)
- 流处理(Stream processing)
#### 1.3.1 Walking through a sample application
数据并行方法(data parallel approach)是最常用的并行计算应用策略(parallel computing application strategies)之一

对于一个计算问题，为了引入并行化，我们根据该问题其创建空间网格(spatial mesh)，在空间网格上执行计算
空间网格是规则的二维网格(regular two-dimensional/2D grid)，包含许多矩形元素或单元(rectangular elements or cells)

根据问题创建空间网格并为在网格上的并行计算做准备的步骤如下：
1. 将问题离散化(discretize)/拆分(break up)成多个更小的单元或元素(smaller cells or elements)
2. 定义要在网格(mesh)的每个元素上执行的计算核(a computation kernel)/操作(operation)
3. 在CPUs和GPUs上加入下列并行化层(layers of parallelization)以执行计算
	- 向量化(Vectorization)
		处理核一次处理多个数据单元
		(work on more than one unit of data at a time)
	- 线程(Threads)
		部署更多的计算路径以让更多处理核参与计算
		(deploy more than one compute pathway to engage more processing cores)
	- 进程(Process)
		将计算分配给多个程序实例，分散到单独的内存空间中
		(seperate program instances to spread out the calculation into seperate memeory spaces)
	- 将计算卸载至GPUs(Off-loading the calculation to GPUs)
		将数据发送给图形处理器进行计算
		(send the data to the graphics processor to calculate)

以一个空间区域(a region of space)的二维问题域(2D problem domain)为例
要根据喀拉喀托火山的二维图像(2D image)进行计算，用机器学习建模火山烟流、火山喷发和海啸等，而要得到实时的结果(real-time result)以辅助决策，计算速度就及其重要
![[PHPC-Fig1.8.png]]
因此，我们需要根据步骤引入并行计算：

**步骤一：将问题离散化成多个更小的单元或元素
(Step 1: Discretize the problem into smaller cells or elements)**
离散化(discretization)即指将问题域分解成多个片段(pieces)，在图像处理中，这些小片段一般是位图中的像素点(pixels in a bitmap image)，在计算领域(computational domain)，我们一般称其为单元或元素(cells or elements)
单元或元素的集合(collection)形成计算网格(computational mesh)，计算网格覆盖了我们需要进行模拟计算的空间领域(covers the spatial region for the simulation)
每个单元的数据值可以是整型、单精度浮点数、双精度浮点数

**步骤二：定义要在网格的每个元素上执行的计算核
(Step 2: Define a computational kernel, or operation, to conduct on each element of the meth)**
对该离散化数据的计算通常是某种形式的模版操作(stencil operation)
模板操作即利用模式来计算当前单元的新值，而模式则涉及到相邻单元(a pattern of adjacent cells)
常见的模式有
对相邻单元取平均值，用于图像的模糊化操作(blur operation)，
根据相邻单元求梯度，用于图像的边缘检测，锐利化图像的边缘(edge-dection, which sharpen the edges in a image)，
或者和解决用偏微分方程描述的物理系统相关的更复杂的操作等

模板操作(stencil operation)举例：
用涉及五个点的模板(five-point stencil)进行图像模糊(通过求模板涉及值的加权平均)
![[PHPC-Fig1.10.png]]
如果我们是根据RGB有色图像(color image)建立模型，则执行模板操作时，可以逐通道执行，例如逐通道模糊化
这里蕴含了偏微分方程中的“偏(partial)”的概念，例如在实际中，不同的颜色有不同的随时间和空间扩散和传播的速率(这可以是在图像上产生特殊效果，也可以描述如在显影过程中，真实的颜色如何在照片中渗出并融合)，它们是独立的，求解这些速率时，需要令其他两个颜色保持不变
在科学计算中，参与计算的变量可能是火山烟流的质量、速度，我们根据它们建立偏微分方程，利用并行计算求解

**步骤三：向量化以让处理核一次处理多个数据单元
(Step 3: Vectorization to work on more than one unit of data at a time)**
一些处理器拥有矢量运算(vector operations)的能力，能够一次处理多个数据单元
![[PHPC-Fig1.11.png]]
例如上图中的计算网格中，每个单元包含一个双精度浮点数，对于向量处理单元来说，在一条机器指令下，在一个时钟周期内，它可以同时对四个数据单元进行运算(with one instruction in one clock cycle)

**步骤四：多线程以部署更多的计算路径以让更多处理核参与计算
(Step 4: Threads to deploy more than one compute pathway to engage more processing cores)**
对于一个多核处理器，我们可以采用多线程让多个核同时进行计算工作
![[PHCP-Fig1.12.png]]
例如在上图中，处理器包含四个核，每个核是一次可以处理四个数据单元的向量处理单元，因此可以给每个核分配一个线程，则四个核在同一时间内，可以同时处理16个数据单元

**步骤五：多进程以将计算分配给多个程序实例，分散到多个内存空间
(Step 5: Processes to spread out the calculation to seperate memory spaces )**
一般单个计算机仅由一个CPU，在需要进一步提高并行处理性能时，我们需要多个个CPU，也就是多个计算机，在并行计算中，我们一般称其为结点(nodes)
我们可以将工作划分给不同的结点，每个结点运行各自的进程，因此每个进程的内存空间都是独立且相互分离的
![[PHCP-Fig1.13.png]]
例如在上图中，分配两个进程给两个结点，每个进程包含四个线程分配给四个核，每个核是一次可以处理四个数据单元的向量处理单元，则两个结点在同一时间内，可以同时处理32个数据单元

即现在的潜在加速为$32\times$
$$\begin{aligned}&2\ \text{desktops}\ (\text{nodes}) × 4\ \text{cores}\ × (256\ \text{bit-wide}\ \text{vector}\ \text{unit})/(64\text{-bit}\ \text{double})\\
=&32\times \text{potential}\ \text{speedup}\end{aligned}$$
如果对于相对高端的集群，潜在加速可以达到$4608\times$
$$\begin{aligned}&16\ \text{nodes} × 36\ \text{cores}\ × (512\ \text{bit-wide}\ \text{vector}\ \text{unit})/(64\text{-bit}\ \text{double})\\
=&4608\times \text{potential}\ \text{speedup}\end{aligned}$$

**第六步：将计算卸载至GPUs
(Off-loading the calculation to GPUs)**
GPUs使得我们可以利用流式多处理器(streaming multiprocessors)以加速计算

例如Nvidia Volta GPU拥有84个流式多处理器，每个流式多处理器拥有32个双精度核(double-precision cores)，因此同时可以有$84\times 32 = 2688$个双精度核工作
如果有一个16结点的集群，每个结点有一个Nvidia Volta GPU，则可以达到$16\times 2688=43008$路的并行计算

但要注意的是，实际的加速比(actual speedup)往往比计算出的潜在加速比(potential speedup)要低许多，我们需要对各个并行层进行协调尽可能达到高的加速比

#### 1.3.2 A hardware model for today's heterogeneous parallel systems
组成一个计算系统的基本硬件有：
- 动态随机存取存储器(DRAM/Dynamic Random Access Memory)
	用于存储数据和指令
- 计算核(computational core)，简称为核(core)
	进行算数运算(arithmetic operations)，包括加减乘除(add, subtract, multiply, divide)
	求值逻辑表达式(evaluate logic expressions)
	从DRAM中加载(load)数据，向DRAM中存储(store)数据
	(核从DRAM中加载数据和指令，进行运算后，将结果存入DRAM)
	现代CPUs一般具有多核，因此可以利用多核进行并行计算
- 加速器硬件(accelerator hardware)，其中最常见的是GPUs
	GPUs一般具有千个数量级的核，同时具有独立的存储空间

处理器、DRAM、加速器硬件的结合构成了一个计算结点(compute node)
计算节点可以是一个家用主机(home desktop)，也可以是超级计算机的一个机架(a rack in a supercomputer)
计算节点之间通过网络(networks)互联(interconnect)，每个独立的结点运行独立的OS实例用于管理所有的硬件资源(hardware resources)

这些构成计算系统的硬件的不同的组成方式形成了了不同类型的系统模型(models)
**分布式内存架构：一种跨节点并行方法
(Distributed Memory Architecture: A Cross-Node Parallel Method)**
分布式内存集群(distributed memory cluster)/分布式内存架构是并行计算的第一种也是最具可扩展性(scalable)的方法之一
![[PHCP-Fig1.15.png]]
在该架构中，每个CPU都拥有自己的本地内存(local memory)，CPU之间通过通信网络连接
该架构具有良好的可拓展性，可以几乎无限制添加新的结点

该架构中，整体的内存空间由每个结点的内存子空间组成，对于一个结点来说，访问结点外(off-node)的内存空间和访问结点(on-node)上的内存空间的行为是不同的，因此存在内存局部性(memeory locality)，
同时程序员需要显式访问不同的内存区域，且必须在应用程序开始时(at the outset of the application)管理内存空间的分区

**共享式内存架构：一种结点上并行方法
(Shared Memory Architecture: An On-Node Parallel Method)**
共享式内存架构中，两个CPU直接连接到同一共享内存(shared memory)上，如
![[PHCP-Fig1.16.png]]

该架构使得处理器共享相同的地址空间(address space)，因此简化了编程
但引入了潜在的内存冲突(potential memory conflicts)，可能导致正确性和性能问题(correctness and performance issues)

在多个CPU之间或在多核CPU上的多个处理核心之间同步(synchronize)内存的访问和内存的数据是复杂且昂贵的

在该架构中，CPUs和处理核心数量的增加不会扩大应用程序可用的内存空间(avaliable)，并且同步开销(synchronization costs)限制了该架构的可拓展性

**向量单元：单指令多操作
(Vector Units: Multiple Operations with One Instruction)**
提高CPU的时钟频率(clock frequency)以获得更高吞吐量(throughput)是很自然的做法，但这种方法最大的限制在于CPU的时钟频率越高，就需要越高的功率，产生更多的热量，而无论是安装了一定数量电源线的HPC超级计算中心，还是安装了有限容量电池的手机，设备都有功率限制(power limitations)
这个问题被称为功耗墙(the power wall)

除了提高时钟周期以外，另一种提高CPU吞吐量的想法是让CPU在一个周期内可以执行多个操作(more than one operation per cycle)
因此出现了向量化(vectorization)技术，相较于标量单元(scalar unit)，向量单元(vector unit)只需要稍微多一点点的功耗就能在一个周期内执行多个操作(比如处理多个数据)
并且，由于向量化技术可以减少执行时间，因此实际使用时往往可以减少总能耗(energy consumption)

向量长度(vector length)指向量单元的位宽，最常见的向量长度是256位，如果数据为64位双精度，则一次向量运算可以同时完成4个双精度运算
路数(lanes of a vector operation)指向量运算一次能处理几路的数据

例如
![[PHPC-Fig1.17.png]]图中，向量处理器一次从内存中取四个数据，进行四路运算，然后将四个结果存回内存

**加速器设备：一种专用的附加处理器
(Accelerator Device: A Special-Purpose Add-On Processor)**
加速器设备是为快速执行特定任务而设计的离散硬件，最常见的加速器设备是GPU，也可被称为通用图形处理单元(general-purpose graphics processing unit/GPGPU)
GPU包含许多小型处理核心，称为流式多处理器(streaming multiprocessors/SMs)
流式多处理器核心比CPU核心更简单，但大量的流式多处理器可以提供大量的算力

![[PHCP-Fig1.18.png]]
通常，CPU上会有一个小型集成的GPU，而大多数现代计算机也有一个独立的、离散的GPU通过外围组件接口(Peripheral Component Interface/PCI)总线连接到CPU，离散GPU通常比集成GPU更强大

总线引入了数据和指令的通信成本(communication cost for data and instructions)，在高端系统中，NVIDIA使用NVLink，AMD Radeon使用Infinity Fabric以降低通信成本，但是这一成本仍然是巨大的

**通用异构并行架构模型
(General Heterogeneous Parallel Architecture Model)**
将之前涉及到的架构结合入一个模型，即得到通用异构并行架构模型
![[PHCP-Fig1.19.png]]

例如上图中，存在跨节点并行(两个结点，结点之间通过网络连接)，存在结点内并行(每个结点内部有两个CPU，共享DRAM)，存在加速器设备(每个CPU都有集成GPU，且有一个CPU通过PCI总线和一个离散GPU连接)

对于一个结点内的两个CPU，尽管共享主内存，但它们通常访问的是不同的非统一内存访问区域(Non-Uniform Memory Access/NUMA regions)，这意味着访问第二个CPU的内存比访问它自己的内存更昂贵

#### 1.3.3 The application/software model for today's heterogeneous parallel systems
在并行计算的硬件模型之上的是OS，在OS之上的就是并行计算的软件模型，其中OS作为这两层之间的接口

程序员需要在源代码中显式地生成进程和线程(spawn process and threads)，将数据、作业、指令卸载(offload)到计算设备，对数据块进行操作等以使程序尽可能并行化运行，这类操作就是并行计算中的软件模型所包括的内容

常用的并行方法有：
- 基于进程的并行(Process-based parallelization)
- 基于线程的并行(Thread-based parallelization)
- 向量化(Vectorization)
- 流处理(Stream processing)

**基于进程的并行：消息传递
(Process-based Parallelization: Message Passing)**
消息传递方法(message passing approach)是为分布式内存架构而开发的，该方法使用显式消息(explicit messages)在进程之间移动数据，消息通过网络(network)或内存(memeory)传输

应用程序生成(spawn)的独立的进程(seperate processes)在消息传递中被称为ranks，其中每个进程都具有自己的内存空间和指令管道(memeory space and instruction pipeline)
![[PHCP-Fig1.20.png]]

进程由OS调度至处理器上执行，该操作在内核空间完成，用户没有权限干涉

1992年，许多消息传递库(message-passing library)合并为Message Passsing Interface(MPI)标准
现在几乎所有运行于多个结点的并行应用程序都使用了MPI库，MPI库也有许多不同的实现(different implementations)

> _Distributed computing versus parallel computing_
> 一些并行应用程序使用一种较低级别(low-level)的并行化方法，称为分布式计算
> 分布式计算式并行计算的子集，指一组的松散耦合(loosely-coupled)的进程，通过OS级别的调用(OS-level calls)进行协作以进行计算
>采用分布式计算的应用程序包括对等网络(peer-to-peer networks)、万维网(the World Wide Web)、互联网邮件(internet mail)、The Search for Extraterrestial Intelligence(SETI@home)等
>在分布式计算中，程序员使用远程过程调用(remote procedure call/RPC)或网络协议(a network protocal)以通过OS在一个单独的节点上创建进程，不同结点中的进程进而通过消息传递以交换信息，称为进程间通信(inter-process communication/IPC)，IPC也有许多变体(variaties)

**基于线程的并行：通过内存共享数据
(Thread-based Parallelization: Shared Data via Memory)**
基于线程的并行化在同一进程内生成多个独立的指令指针(instruction pointers)，线程之间共享进程内存
![[PHCP-Fig1.21.png]]
使用多线程时，程序员需要决定哪一部分的指令集和数据是独立且支持多线程的
OpenMP是常用的一个多线程库，提供了生成线程和在线程中划分作业的功能

多线程方法只能在一个结点中拓展(scaling within a single node)，因此只能提供适度的加速比(modest speedup)

**向量化：单指令多操作
(Vectorization: Multiple Operations with One Instruction)**
相比拓展HPC中心的计算资源，向量化方法更具成本效益，并且这种方法在手机等便携式设备(portable devices)上可能是绝对必要的

向量化中，处理器一次完成一个数据块的作业，一个数据块一般包含2-16个数据项
这种操作更正式的术语是单指令多数据(single instruction, multiple data/SIMD。SIMD)

编译器在分析时根据源代码中的编译指令(pragmas)引入向量化
程序员通过编译指令(pragmas and directives)引导编译器对部分代码进行向量化，因此向量化能力依赖于编译器能力
![[PHCP-Fig1.22.png]]


此外，在没有显式编译器标志(explicit complier flags)的情况下，生成的代码通常是针对功能最差的处理器，采用最短的向量长度，因此显著降低了向量化的有效性
有些机制(mechanisms)可以绕过编译器，但这些机制需要更多的编程工作，而且不可移植

**通过专用流处理器进行流处理
(Stream Processing Through Specialized Processors)**
流处理是一个数据流概念(dataflow concept)，在流处理中，数据流(a stream of data)由更简单的专用处理器处理
流处理技术常用于嵌入式计算、图像渲染、仿真计算

最常见的专用处理器是GPU，在计算时，CPU将数据通过PCI总线卸载到GPU进行运算，GPU将结果通过PCI总线返回
![[PHCP-FIg1.23.png]]

### 1.4 Categorizing parallel approaches
如果我们阅读更多关于并行计算的内容，我们会遇到SIMD(单指令，多数据)和MIMD(多指令，多数据)这样的缩写，这些术语源于迈克尔·弗林在1966年提出的计算机架构类别(categories of computer architecture)，现在已成为众所周知的弗林分类法(Flynn's Taxonomy)
这些类别有助于以不同的方式看待架构中的潜在并行化(potential parallelization in architecture)，该分类法是基于将指令和数据分解为串行操作或多操作(breaking up instructions and data into either serial or multiple operations)(Figure 1.24)
要注意的是，尽管该分类法很有用，但有些架构和算法并不适合整齐地归入一个类别(fit neatly within a category)
该分类法的有用之处在于帮助识别如SIMD这样的模式，SIMD在处理条件语句上可能存在潜在困难。这是因为每个数据项(data item)可能需要处于不同的代码块中，但线程必须执行相同的指令
![[PHPC-Fig1.24.png]]

当存在多个指令序列(instruction sequence)时，其类别就被称为多指令，单数据(MISD)，这不是一种常见的架构，最适合的例子是对同一数据进行冗余计算(redundant computation on the same data)，该架构一般用于实践高度容错的方法(fault-tolerant approaches)，例如航天器控制器，航天器处于高辐射环境中，通常需要运行每个计算的两个副本并比较两个的输出

向量化(vectorization)是SIMD的一个典型例子，其中，相同的指令在多个数据上执行(the same instruction is performed across multiple data)，SIMD的一个变体是单指令，多线程(SIMT)，通常用来描述GPU工作组(work groups)

最后一个类别在指令和数据上都有并行化，被称为MIMD，这个类别描述了多核并行架构(multi-core parallel architectures)，它构成了大多数大型并行系统的主体
### 1.5 Parallel strategies
![[PHPC-Fig1.25.png]]
在我们1.3.1节的初始示例中，我们查看了单元格或像素的数据并行，但数据并行也可以用于粒子和其他数据对象，数据并行是最常用也是最简单的并行策略，其本质上是每个进程执行相同的序(each process executes the same program)，但操作的是数据的独特子集(operates on a unique subset of data)，如Figure 1.25右上角所示，数据并行的优势在于，随着问题规模和处理器数量的增长，它能够很好地扩展(scales well)

另一种并行方法是任务并行(task parallelism)，任务并行包括了主控制器和工作线程(main controller with worker threads)、流水线(pipeline)或桶接力(bucket-brigade)策略，如Figure 1.25所示，
流水线方法在超标量(superscalar)处理器中使用，其中地址和整数计算(address and integer calculations)是使用单独的逻辑单元完成的，而不是浮点处理器，这就允许这些计算可以并行进行；桶接力策略使用每个处理器按顺序对数据进行操作和转换(operate on and transform the data in a sequence of operations)；在主处理器-工人处理器方法中，一个处理器调度并分发所有工人的任务，每个工人在返回之前完成的任务时检查下一个工作项(work item)
我们可以组合不同的并行策略，以暴露更大程度的并行性
### 1.6 Parallel speedup versus comparative speedups: Two different measures
我们将在这本书中展示许多比较性能数据(comparative performance numbers)和加速比(speedups)，通常，"加速比(speedup)"这个术语会被用于比较两种不同的运行时间(run times)，加速比是一个通用术语，用于许多上下文，例如量化优化的效果(quantify the effects of optimization)

我们定义两个不同的术语：
- 并行加速比(Parallel speedup) 
    我们实际上应该称之为串行-并行加速比(serial-to-parallel speelup)
    该加速比是相对于在标准平台上的标准串行运行的基线(relative to a baseline serial run on a standard platform)，通常是单个CPU，并行加速可以来源于在GPU上运行，或在计算机系统节点的所有核心上使用OpenMP或MPI
- 比较加速比(Comparative speedup)
    我们实际上应该称之为架构之间的比较加速比(comparative speedup between architectures)
    这通常是两个并行实现(two paralle implementations)之间的性能比较，或者是其他合理受限的硬件集之间的比较，例如，它可能是计算机节点上所有核心上的并行MPI实现与节点上的GPU(s)之间的比较

这两种性能比较类别代表了两个不同的目标，并行加速比是为了理解通过添加特定类型的并行性(add particular type of parallelism)可以获得多少加速比，但这不是架构之间的公平比较，例如，我们是将GPU运行时间与串行CPU运行时间进行比较，而不是多核CPU和GPU之间的比较，
在尝试比较多核CPU与节点上一个或多个GPU的性能时，采用架构之间的比较加速比更合适

近年来，一些人提出将要相互比较的两种架构规范化(normalize the two architectures)，即比较相似功率或能源需求(power or energy requirements)下架构的相对性能，而不是任意节点之间进行比较
由于存在太多的不同架构和以及可能的组合，我们建议在性能比较时，在括号内添加以下信息，以提供更多上下文：
- 如果相比较的两种架构都在2016年发布，且是最高端硬件(highest-end)，在每个项后添加(Best 2016)
    例如，并行加速比(Best 2016)和比较加速比(Best 2016)将表明比较是在特定年份(2016年)发布的最好硬件之间进行的，例如比较2016年的高端GPU和高端CPU
- 如果相比较的两种架构都在2016年发布，但不是最高端硬件(highest-end)，添加(Common 2016)或(2016)
    这对于拥有更主流部件的开发人员和用户会更加相关
- 如果GPU和CPU是在2016年的Mac笔记本电脑或台式机中发布的，添加(Mac 2016)，
    对于其他品牌在一段时间内(例如2016年)具有固定组件的类似情况，同样添加类似的注解
- 被比较的硬件发布年份可能存在不匹配时，添加(GPU 2016:CPU 2013)(本例中为2016年对比2013年)

由于CPU和GPU型号的激增，性能数字将必然更多地是定义较为模糊的比较，而不是一个明确定义的度量(well-defined metric)，但是，在更正式的场合，我们至少应该指出比较的性质，以便其他人对数字的含义有更好的了解，并对硬件供应商更公平
## 2 Planning for parallelization
本章重点介绍开发并行应用程序的工作流模型(workflow model)，如Figure 2.1所示
![[PHPC-Fig2.1.png]]
通常最好以小的增量(small increments)实现并行性，这样如果遇到问题，可以撤销(reversed)最后几次提交。这种模式适合敏捷项目管理技术

让我们想象一下，我们被分配了一个新项目，要加快并行化一个应用程序，该应用程序来自Figure 1.9中展示的空间网格(spatial mesh)(Krakatau火山示例)，这可以是一个图像检测算法，一个火山灰流的科学模拟，或一个由火山喷发产生的海啸波的模型，或者三者都是，我们可以采取哪些步骤来成功地进行并行化？

作为开始，我们需要一个项目计划，所以我们首先从这个工作流的步骤进行高层次概述(high-level overview)，然后，随着本章的进展，我们将更深入地探讨每个步骤，并聚焦于典型的并行项目特征上(typical for a parallel project)

> **Rapid development: The parallel workflow**
> 首先，我们需要为团队和应用程序为快速开发做好准备，因为我们已经有一个现有的串行应用程序，它在Figure 1.9的空间网格上工作，因此并行开发时可能会有很多小的更改和频繁的测试，以确保运算结果不变
> 代码准备包括设置版本控制、开发测试套件(developing a test suite)，并确保代码质量和可移植性
> 
>为了为开发周期设定阶段，我们需要确定可用的计算资源、我们的应用程序的需求和用于的性能要求，系统基准测试(system benchmarking)有助于确定计算资源的限制，而分析(profiling)有助于我们理解应用程序的需求及其最昂贵的计算内核(most expensive computational kernels)，计算内核指的是应用程序中既计算密集(computationally intensive)又概念上自包含的部分(conceptually self-contained)
>
>从内核分析(kernel profiles)中，我们将为例程的并行化和更改的实施进行计划，实现阶段(implementation stage)只有在例程被全部并行化并且代码保持可移植性和正确性时才完成，一旦满足这些要求，更改就可以提交到版本控制系统，在提交了增量更改之后，该过程将再次随着应用程序和内核分析开始
### 2.1 Approaching a new project: The preparation
Figure 2.2展示了准备步骤中推荐的组件(components)，这些对并行化项目(parallelization projects)都是特别重要的
![[PHCP-Fig2.2.png]]

在准备阶段，我们需要设置版本控制，为我们的应用程序开发测试套件，并清理现有代码，
版本控制允许我们跟踪对应用程序随时间所做的更改，同时允许我们快速撤销错误，并在以后可以追踪代码中的bug；测试套件允许我们在对代码进行每次更改时验证应用程序的正确性，与版本控制结合使用，辅助我们快速开发代码

有了版本控制和代码测试，我们现在可以开始清理代码的任务，好的代码要易于修改和扩展(modify and extend)，并且不会表现出不可预测的行为，我们可以通过模块化(modularity)和检查内存问题(memory issues)，确保良好的、干净的代码
模块化(modularity)意味着我们将内核实现为具有明确输入和输出的独立子程序或函数(independent subroutines or functions with well-defined input and output)，
内存问题(memory issues)可能包括内存泄漏(memory leaks)、越界内存访问(out-of-bounds memory access)和使用未初始化的内存(use of unintialized memory)

以可预测和高质量的代码开始我们的并行工作，可以促进快速进展和可预测的开发周期(predictable development cycles)

最后，我们要确保我们的代码是可移植的(portable)。这意味着多个编译器可以编译我们的代码，拥有并维护编译器可移植性(compiler portability)允许我们的应用程序可以针对除了我们当前可能考虑的平台之外的其他平台，此外，经验表明，开发可以在多个编译器工作下的代码有助于代码被提交到我们的代码版本历史之前发现bugs，随着高性能计算领域的快速变化，可移植性也允许我们的代码可以更快地适应变化

准备所花费的时间与实际实现并行化所花费的时间相媲美并不罕见，特别是对于复杂的代码，接下来，我们讨论项目准备的四个组成部分(the four component of project preparation)
#### 2.1.1 Version control: Creating a safety vault for your parallel code
在并行化过程中发生的许多变化中，我们可能会突然发现代码损坏了或返回了不同的结果，因此能够通过回退到一个正常工作的版本(backing up to a working version)来从这种情况中恢复是至关重要的
注意：在开始任何并行化工作之前，检查我们的应用程序是否有版本控制系统

例如，我们之间讨论的火山灰流模型从未有过任何版本控制，当我们深入挖掘时，我们发现实际上有四个版本的火山灰流代码在不同开发者的目录中，当有一个版本控制系统在运行时，我们可能想要审查团队用于日常操作的流程，也许团队认为切换到“拉取请求(pull request)”模型是个好主意，在这个模型中，更改在提交之前会由其他团队成员进行审查(posted for review by other team members before being commited)，或者团队可能觉得直接提交的“推送(push)”模型更适合并行化任务的快速、小的提交，在推送模型中，提交是直接对仓库进行的(directly to the repository)，无需审查，在我们没有版本控制的火山灰流应用的例子中，优先事项是建立一些东西来控制开发者之间代码的无序分歧

有许多版本控制系统可以选择，如果没有其他偏好，我们会建议使用Git，这是最常见的分布式版本控制系统，分布式版本控制系统(distributed version control system)是一种允许多个仓库数据库(multiple repository databases)的系统，而不是像集中式版本控制(centralized version control)中使用的单一中央系统(a single centralized system)，分布式版本控制对于开源项目和开发者在笔记本电脑上工作、在远程位置工作，或在其他他们没有连接到网络或靠近中央仓库的情况下工作是有利的，在今天的开发环境中，这是一个巨大的优势，但它的代价是带来了额外复杂性，集中式版本控制仍然流行，更适合企业环境，因为只有一个存放所有源代码信息的地方，集中式控制还为专有软件(proprietary software)提供了更好的安全性和保护

有许多好的书、博客和其他资源介绍如何使用Git，我们在本章末尾列出了一些。我们也在第17章列出了一些其他常见的版本控制系统，这些包括免费的分布式版本控制系统，如Mercurial和Git，商业系统如PerForce和ClearCase，以及用于集中式版本控制的CVS和SVN。无论你使用哪种系统，你和你的团队都应该频繁提交，以下场景在并行化任务中尤其常见：
- 我会在我添加下一个小更改后提交......
- 再做一个......然后突然之间代码就无法工作了
- 现在提交太迟了！
因此我们应该尝试通过定期提交来避免问题

提示：如果不希望在主仓库中有很多小的提交，我们可以使用一些版本控制系统(如Git)合并提交，或者可以仅为自己维护一个临时的版本控制系统

提交信息(commit message)是提交作者可以传达正在处理的任务以及为什么进行某些更改的地方，每个团队对于这些信息应该有多详细都有自己的偏好；我们建议在提交信息中使用尽可能多的细节(as much detail as possible)，这是通过今天的努力避免以后混淆的机会，一般来说，提交信息包括摘要(summary)和正文(body)，摘要提供了一个简短的声明，清楚地表明提交涵盖了哪些新变化，此外，如果我们使用了一个问题跟踪系统(issue tracking system)，摘要行将引用那个系统中的问题编号(reference an issue number from that system)，最后，正文包含了提交背后的大部分“为什么(why)”和“怎么做(how)”

> **Examples of commit message**
> 糟糕的提交信息：
> `Fixed a bug`
> 好的提交信息：
> `Fixed the race condition in the OpenMP version of the blur operator`
> 极好的提交信息：
```
[Iuuse # 21] Fixed the race condition in the OpenMP version of the blur operator.
* The race condition was causing non-reproducible results amongst GCC, Intel, and PGI compilers. To fix this, an OMP BARRIER was introduced to force threads to synchronize just before calculating the weighted sencil sum.
* Confirmed that the code builds and runs with GCC, Intel, and PGI compilers and produces consistent results.
```
>第一条信息并没有真正帮助任何人理解修复了什么bug，第二条信息有助于精确定位涉及模糊算子中竞争条件的问题的解决方案，最后一条信息引用了一个外部问题跟踪系统中的问题编号(issue number)(#21)，并在第一行提供了提交摘要(commit summary)，提交正文，即摘要下面的两个项目符号点(bullet points)，提供了更多关于具体需要什么以及为什么需要的详细信息，并表明我们在提交之前花时间测试了该版本
#### 2.1.2 Test suites: The first step to creating a robust, reliable application
测试套件(test suite)是一组问题(a set of problems)，它们通过测试应用程序的各部分来保证相关代码是有效的，除了最简单的代码之外，测试套件对于所有代码都是必需的
每次更改后，都应该进行测试，以确保我们得到的结果是一样的，这听起来很简单，但有些代码在不同的编译器和处理器数量下可能会得到略有不同的结果

**Understanding changes in results due to parallelism**
并行化过程本质上会改变操作的顺序，这会轻微地修改数值结果，但并行化中的错误也会导致数值误差
理解数值结果的改变原因在并行代码开发中至关重要，因为我们需要与单处理器运行进行比较，以确定我们的并行编码是否正确，我们将在第5.7节讨论减少数值误差的方法，以便并行化错误更加明显，届时我们将讨论全局求和的技术

对于我们的测试套件，我们需要一个工具来比较数值字段，并允许有一定的差异容差(compare numerical fields with small tolerance for differences)，近年来市场上出现了一些数值差异实用工具(numerical diff utilities)，其中两个这样的工具是：
- Numdiff，来自 https://www.nongnu.org/numdiff/
- ndiff，来自 https://www.math.utah.edu/~beebe/software/ndiff/

或者，如果我们的代码以HDF5或NetCDF文件的形式输出其状态，这些格式自带有实用工具，允许我们以不同的容差(tolerance)比较文件中存储的值
- HDF5是最初被称为分层数据格式(Hierarchical Data Format)的软件的第5版，现在称为HDF，它可以从HDF Group(https://www.hdfgroup.org/)免费获取，是输出大型数据文件(large data files)的常用格式
- NetCDF或网络通用数据格式(Network Common Data Form)是气候和地球科学界使用的另一种格式，NetCDF的当前版本是建立在HDF5之上的，可以在Unidata Program Center的网站(https://www.unidata.ucar.edu/software/netcdf/)找到这些库和数据格式

这两种文件格式都使用二进制数据(binary data)以提高速度和效率，二进制数据是数据的机器表示。这种格式对我们来说看起来就像天书，但HDF5有一些有用的实用工具允许我们查看其内部：h5ls实用工具可以列出文件中的对象(objects in the file)，例如所有数据数组的名称(names of all the data arrays)；h5dump实用工具转储(dumps)每个对象或数组中的数据
对于我们这里的目的，h5diff实用工具可以比较两个HDF文件，并报告超过数值容差(above a numeric tolerance)的差别

**Using CMake and CTest to automatically test your code**
近年来，出现了许多可用的测试系统，包括CTest、Google Test、pFUnit Test等，现在，让我们看看使用CTest和ndiff创建的系统

CTest是CMake系统的一个组件，CMake 是一个配置系统(configuration system)，它将生成的makefile适配到不同的系统和编译器，
使用CTest实施测试的过程相对容易，单独的测试可以被编写为任何命令序列(individual tests are written as any sequence of commands)，要将这些测试集成到CMake系统中，需要在 `CMakeLists.txt` 中添加以下内容：
- `enable_testing()`
- `add_test(<testname> <executable name> <arguments to executable>)`
然后使用 `make test`, `ctest` 来调用测试，或者可以使用 `ctest -R mpi` 来选择单独的测试(individual tests)，其中 `mpi` 是一个正则表达式，该命令会运行任何名称匹配正则表达式的测试

> **Example: CTest prerequisites**
>为了运行这个示例，需要安装MPI、CMake 和ndiff，对于MPI，我们将在 Mac上使用OpenMPI 4.0.0和CMake 3.13.3(包含 CTest)，在Ubuntu上使用较旧的版本
>我们将使用安装在Mac上的GCC编译器，版本为8，而不是默认的编译器，然后，使用包管理器安装OpenMPI、CMake 和 GCC，我们将在Mac上使用Homebrew，在Ubuntu Linux上使用Apt和Synaptic
>如果libopenmpi-dev的开发头文件与运行时(runtime)分开，请确保获取它们，ndiff是通过从 https://www.math.utah.edu/~beebe/software/ndiff/ 下载工具，然后运行 `./configure`、`make` 和 `make install` 手动安装的

让我们通过一个使用CTest系统创建测试，我们首先根据 Listing 2.1 制作两个源文件，为这个简单的测试系统创建应用程序
我们将使用计时器在串行和并行程序的输出中产生小的差异
Listing 2.1 Simple timing programs for demonstarting the testing system
`C Program, TimeIt.c`
```c
#include <unistd.h>
#include <stdio.h>
#include <time.h>
int main(int argc, char *argv[]) {
    struct timespec tstart, tstop, tresult;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    sleep(10);
    clock_gettime(CLOCK_MONOTONIC, &tstop);

    // timer has two values for resolution and to prevent overflows
    tresult.tv_sec = tstop.tv_sec - tstart.tv_sec;
    tresult.tv_usec = tstop.tv_nsec - tstart.tv_nsec;

    printf("Elapsed time is %f secs\n", (double)tresult.tv_sec + (double)tresult.tv_nsec*1.0e-9);
}
```
`MPI Program, MPITimeIT.c`
```c
#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
int main(int argc, char *argv[]) {
    int mtype;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mtype); // initializes MPI and gets processor rank
    double t1, t2;
    t1 = MPI_Wtime();
    sleep(10);
    t2 = MPI_Wtime();
    if (mtype == 0) printf( "Elapsed time is %f secs\n", t2 - t1); // print timing output from first processor
    MPI_Finalize(); // shuts down MPI
}
```

现在我们需要一个测试脚本来运行应用程序并产生一些不同的输出文件，运行测试脚本之后，我们应对输出进行数值比较
以下是一个名为 `mympiapp.ctest` 的示例：
`mympiapp.ctest`
```bash
# !/bin/sh
./TimeIt > run0.out # runs a serial test
mpirun -n 1 ./MPITimeIT > run1.out # runs the first MPI test on 1 processor
mpirun -n 2 ./MPITimeIT > run2.out # runs the second MPI test on 2 processors

ndiff --relative-error 1.0e-4 run1.out run2.out # compares the output for the two MPI jobs to get the test to fail
test1=$? # captures the status set by the ndiff command

ndiff --relative-error 1.0e-4 run0.out run2.out # compares the serial output to the 2 processor run
test2=$? # captures the status set by the ndiff command

exit "$(($test1+$test2))" # exits with cumulative status code so CTest can report pass or fail
```
这个测试首先在第5行上，使用0.1%的容差比较1个和2个处理器的并行作业的输出，然后在第7行上，将串行运行与2个处理器的并行作业进行比较，要让测试失败，我们尝试将容差降低到1.0e-5
CTest使用第9行上的退出代码来报告通过或失败，向测试套件添加许多CTest文件的最简单方法是使用一个循环，查找所有以 `.ctest` 结尾的文件，并将这些文件添加到CTest列表中
以下是一个包含创建两个应用程序的额外指令的 `CMakeLists.txt` 文件的示例：
`CMakeLists.txt`
```cmake
cmake_minimum_required (VERSION 3.0)
project(TimeIt)

enable_testing() // enables CTest functionality in CMake

find_package(MPI) // CMake built-in routine to find most MPI packages

// adds TimeIT and MPITimeIT build targets with their source code files
add_executable(TimeIt TimeIt.c) 

add_executable(MPITimeIt MPITimeIt.c) 

// Needs an include path to the mpi.h file and to the MPI library
target_include_directories(MPITimeIt PUBLIC. ${MPI_INCLUDE_PATH})
target_link_libraries(MPITimeIt ${MPI_LIBRARIES})

// gets all file with the extension .ctest and adds those to the test file list for CTest
file(GLOB TESTFILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.ctest")
foreach (TESTFILES ${TESTFILES})
    add_test (NAME ${TESTFILE} WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/${TESTFILE})
endforeach()

// a custom command, disclean, removes created files
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles CTestTestfile.cmake Makefile cmake_install.cmake)
```


请注意，这个示例中的 `MPIEXEC_EXECUTABLE` 和 `MPIEXEC_NUM_PROCESSES` 变量用于指定 `mpiexec` 命令和进程数量。`PASS_REGULAR_EXPRESSION` 属性用于定义测试通过的条件，这里它检查输出中是否包含特定的成功消息。`file(GLOB TEST_SCRIPTS "*.myctest")` 命令查找所有以 `.myctest` 结尾的文件，并将它们添加到测试套件中。
## 3 Performance limits and profiling
### 3.2
#### 3.2.4 Empirical measurement of bandwidth and flops
经验带宽(empirical bandwidth)用于衡量数据从内存装载到CPU的最快的时间
对于一个字节的数据，从寄存器到CPU需要一个时钟周期，从主存到CPU大约需要400个时钟周期

通过装载连续的数据，尽最大可能充分利用cache，可以得到CPU的最大可能数据传输速率(data transfer rate)，称其为内存带宽，
可以通过衡量对大型数组读写的时间以衡量带宽

有两种方法用于衡量内存带宽：STREAM Benchmark、roofline model(由Empirical Roofline Toolkit生成)
其中roofline model在一张图内包含了内存带宽和峰值浮点运算速率

STREAM基准测量程序(benchmark)衡量对大型数组进行读写的时间
包括四个操作：拷贝(copy)、放缩(scale)、加法(add)、三元(traid)
copy涉及0次算数操作
scale涉及1次算数操作
add涉及1次算数操作
traid涉及2次算数操作

如果计算可以复用cache中的数据，则就可以取得更高的浮点运算速率，如果假设所有要操作的数据都在寄存器或L1 cache中，则最大浮点运算速率完全由CPU时钟频率和CPU一个时钟周期内可以执行几次浮点运算决定，即通过计算得到的理论最大浮点运算速率(theoretical maximum flop rate)

roofline模型中，纵轴是每秒浮点运算次数(flops per second)，横轴是算数密度(arithmetic intensity)
算数密度高时，数据复用率高，速率受理论浮点运算速率制约
算数密度低时，数据复用率低，速率受内存速率制约

### 3.3
#### 3.3.1
Likwid(Like I Knew What I'm Doing)是一个Linux上的命令行工具
它使用硬件计数器衡量并报告系统参数，包括运行时间、时钟频率、功耗、内存读写统计信息等
因为Likwid会使用机器特有的寄存器(machine-specific registers/MSR)，因此要使用Likwid要先 `sudo modprobe msr` 启用MSR
## 4 Data design and performance models

## 5 Parallel algorithms and patterns
### 5.5 Spatial hashing: A highly-parallel algorithm
在第一章中，示例使用的是均匀大小的规则网格(uniform-size, regular grid)，本节探讨更复杂的计算网格

基于单元的自适应网格细化(Cell-based adaptive mesh refinement/AMR)是一类非结构化网格的技术
在基于单元的AMR中，单元数据数组(cell data array)是一维的，且数据可以是任意顺序，网格的位置信息存储在另一个数组中，里面包含了每个单元的大小和位置数据，因此数据是完全非结构化的
![[PHCP-Fig5.2.png]]

AMR技术可以划分为基于片、块、单元的方法(patch, block, cell-based)，基于片和块的方法使用由多个单元构成的不同大小的片或固定大小的块，以至少部分利用了这些单元组(groups of cells)的规则结构
基于单元的AMR则使用的是完全非结构化的数据，且数据的顺序任意

空间哈希技术(spatial hashing)技术的关键基于的是空间信息，哈希算法对于每一次查找操作的平均算法复杂度保持在$\Theta (1)$，所有的空间查询都可以由空间哈希完成，基本的原理在于将对象映射到规则排列的桶的网格中(grid of buckets)
![[PHCP-Fig5.3.png]]

如图5.3所示，桶的大小是根据要映射的对象的特征大小选择的(based on the characteristic size of objects to map)，
对于基于单元的AMR网格，桶的大小选择的是最小的单元大小
对于微粒或实体(particals or objects)，则如图5.3的右边所示，桶的大小选择则基于交互距离(interaction distance)，这个选择意味着只有紧邻的单元需要查询相互作用或碰撞计算
碰撞计算(collision computation)是空间哈希的一大应用领域，应用于光滑粒子流体力学、分子动力学和天体物理学，也适用于游戏引擎和计算机图形学，许多情况下，我们可以利用空间局部性，以减少计算开销

图5.3左边的AMR网格和非结构化的网格都可以被称为可微离散化数据(differential discretized data)，因为单元越小，梯度就越陡峭，就越利于更好求解物理现象，但单元格不能任意小，而存在限制，这种限制也防止了桶变得任意小
AMR网格和非结构化的网格在空间哈希时都会将它们单元的索引(indices)存在其所对应的所有的底层桶中

对于微粒和几何实体，微粒索引和实体标识符(indentifiers)也都会存在桶中，这保持了一种局部性，以防止计算开销随着问题规模增长，
例如，问题域在左上方增大，但右下方的空间哈希交互计算保持不变，这使得对于微粒计算的算法复杂度可以保持在$\Theta(N)$而非$\Theta(N^2)$

如下展示了微粒交互计算的伪代码，在内层循环中，只和邻近的位置进行计算
```
forall particles, ip, in NParticles{
	forall particles, jp, in Adjacent_Buckets{
		if (distance between particles < interaction_distance){
			perform collision or interaction calculation
		}
	}
}
```
#### 5.5.1 Using perfect hashing for spatial mesh operations
讨论一些使用完全哈希(perfect hashing)的方法，这些方法都依赖于完全哈希可以保证每个桶里只有一个项(entry)，避免了碰撞处理，即一个桶里可能有多个数据项的情况
我们研究四个最重要的空间哈希操作
- 邻元素查找(Neighbor finding)
	定位单元格两边的一个或两个邻居
- 重映射(Remapping)
	将另一个AMR网格映射回当前的网格
- 查表(Table lookup)
	在2D表格中定位间隔(intervals)以执行插值(interpolation)
- 排序(Sorting)
	对单元数据的1D或2D排序

**Neighbor finding using a spatial perfect hash**
在科学计算中，从一个单元(cell)中移出的物质必须移动到相邻的单元中，我们需要知道要移动到哪一个单元，以计算要移动的物质量，然后进行移动
在图像分析中，相邻单元的特征可以提供当前单元组成的重要特征

CLAMR中，AMR网格的规则是在一个单元的一个面上只能进行一次单级别的精细化跳跃(only a single-level jump in refinement across a face of a cell)，并且每个单元的每一边的邻近表仅包含一个邻近单元，且一定是左下方的单元，要找到其邻近的第二个单元，就使用其邻近的第一个单元的邻近表，例如单元 `ic` 的两个底部邻近单元是 `nbot[ic]` 和 `nrht[nbot[ic]]`
![[PHCP-Fig5.4.png]]

设定了邻近查找的规则后，问题就在于为每个单元设定邻近数组(neighbor arrays)

寻找相邻单元的可能算法之一，即最朴素的方法，就是直接遍历所有单元，确认是否邻近(通过每个单元的 `i` , `j` 和 `level` 变量确定)，朴素算法是$O(N^2)$的；同时还有基于树的方法，例如K-D树和四叉树(quadtree)(三维情况下就是三叉树)，这些是基于比较的算法，复杂度$O(Nlog N)$

K-D树将网格沿着x维对半分，然后沿着y维再对半分，重复此过程直到找到目标，该算法建立了K-D树，建树复杂度$O(NlogN)$，搜索复杂度$O(NlogN)$

四叉树每个父节点有四个子节点，每个子节点代表一个象限(quadrant)，以精确映射到基于单元的AMR网格的细分(subdivision)，完全(full)的四叉树从一个根结点/顶层开始，不断细分到AMR网格最精细的层次(finest level)，“截断的(truncated)”四叉树从网格的最粗粒度级别开始(coarest level of the mesh)，每一个粗粒度单元都映射到最精细层次的四叉树，四叉树算法是一种基于比较的算法，复杂度$O(NlogN)$

在一个面上仅跳跃一级(just one level jump across a face)的限制称为分级网格(graded mesh)，在基于单元的AMR中，分级网格很常见，但其他的四叉树应用，如天体物理学中的$n$体应用程序，会导致四叉树数据结构中出现更大的跳跃(much larger jumps)

细化的一级跳跃(one-level jump in refinement)使我们能够改进寻找邻居的算法设计(the algorithm design for finding neighbors)，我们可以从代表我们单元的叶子开始搜索，最多只需要沿着树向上两层就可以找到邻居
在搜索大小相似(similar size)的邻居时，搜索应该从树叶开始，并使用四叉树，
对于大型不规则对象(large irregular objects)的搜索，应使用K-D树，并且搜索应该从树根开始

在CPU上实现基于树的搜索算法是可行的，但树的结构和比较操作(the comparisions and tree construction)在GPU较难实现，因为在GPU中，无法进行工作组之外的比较(comparisions beyond the work group)

空间哈希进行邻居查找操作按阶段执行的算法步骤：
- 设定空间哈希的桶大小与基于单元的AMR网格的最精细级别相同(通过设定桶的大小为AMR网格中最精细的单元的大小，保证空间哈希中没有碰撞)
- 对于AMR网格中的每个单元，将单元格编号写入对应桶中
- 对于当前单元，在它的每一边计算与其相邻的更精细的单元的索引(the index for the finer cell one cell outside the current on each side)
- 根据索引，读取相应位置的桶中存储的值

图5.5展示了一个AMR网格和对应的哈希表
![[PHCP-Fig5.5.png]]
实际的算法实现在listing5.5，k-D算法可能需要几周甚至几个月的时间才能从CPU迁移到GPU，而空间哈希算法只需要不到一天，且空间哈希算法对于$N$个单元的平均复杂度只有$\Theta(N)$

CPU实现对哈希表的初始化见listing5.4，例程的输入包括1D数组`i` , `j` 和 `level` ，其中 `level` 是存储了每个单元的精细化层次(refinement level)，而 `i,j` 存储了各个单元的自己的精细化程度下，所在的行号和列号
listing5.4
```c
int *levtable = (int *)malloc(levmx+1)
for (int lev=0; lev<levmx+1; lev++){
	levtable[lev] = (int)pow(2,lev);
}

int jmaxsize = mesh_size*levtable[levmx];
int imaxsize = mesh_size*levtable[levmx];

int **hash = (int **)genmatrix(jmaxsize, imaxsize, sizeof(int));

for(int ic=0; ic<ncells; ic++){
	int lev = level[ic];
	for (int jj=j[ic]*levtable[levmx-lev];jj<(j[ic]+1)*levtable[levmx-lev]; jj++) {
		for (int ii=i[ic]*levtable[levmx-lev]; ii<(i[ic]+1)*levtable[levmx-lev]; ii++) {
			hash[jj][ii] = ic;
		}
	}
}
```

在GPU上使用OpenCL的实现对哈希表的初始化见listing5.5，和listing5.4相比，我们多定义了一个宏用于处理2D索引(indexing)，而最大的差异在于没有按照单元的循环，这是GPU代码的典型特征，即外部循环被移除，由内核启动来处理(handled by the kernel launch)，而单元格索引由每个线程通过调用内在函数 `get_global_id` 提供
listing5.5
```c
#define hashval(j,i) hash[(j)*imaxsize+(i)]

__kernel void hash_setup_kern(
	const uint isize,
	const uint mesh_size,
	const uint levmx,
	__global const int *levtable,
	__global const int *i,
	__global const int *j,
	__global const int *level,
	__global int *hash 88 ) {
	
	const uint ic = get_global_id(0); // each thread is a cell
	if (ic >= isize) return;

	int imaxsize = mesh_size*levtable[levmx];
	int lev = level[ic];
	int ii = i[ic];
	int jj = j[ic]; 
	int levdiff = levmx - lev;

	int iimin = ii *levtable[levdiff];
	int iimax = (ii+1)*levtable[levdiff];
	int jjmin = jj *levtable[levdiff];
	int jjmax = (jj+1)*levtable[levdiff];

	for(int jjj = jjmin; jjj < jjmax; jjj++){
		for(int iii = iimin; iii < iimax; iii++){
			hashval(jjj, iii) = ic; // set the hash table value to the thread ID(the cell number)
		}
	}
```

CPU上实现搜索邻居见listing5.6，逻辑就是循环遍历单元，然后查找哈希表寻找邻居，只需要在哈希表中对当前行号或列号在对应的方向加上适当的数，就能查找到邻居，对于左方或下方的邻居，只需要加一，对于右方或上方的邻居，就需要加上该单元在对应方向上实际占的最精细单元的数量
listing5.6
```c
for (int ic=0; ic<ncells; ic++){
	int ii = i[ic]; 
	int jj = j[ic]; 
	int lev = level[ic]; 
	int levmult = levtable[levmx-lev]; 
	int nlftval = hash[jj*levmult][MAX(ii*levmult - 1,0)]; 
	int nrhtval = hash[jj*levmult][MIN((ii+1)*levmult, imaxsize-1)]; 
	int nbotval = hash[MAX(jj*levmult-1,0)][ii*levmult]; 
	int ntopval = hash[MIN((jj+1)*levmult, jmaxsize-1)][ii *levmult];

	neigh2d[ic].left = nlftval;
	neigh2d[ic].right = nrhtval;
	neigh2d[ic].bot = nbotval;
	neigh2d[ic].top = ntopval;
}
```

在GPU上实现时，移除遍历单元的循环，替换为 `get_global_id()` 调用，
listing5.7
```c
#define hashval(j,i) hash[(j)*imaxsize+(i)]

__kernel void calc_neighbor2d_kern(
	const int isize,
	const uint mesh_size,
	const int levmx,
	__global const int *levtable,
	__global const int *i,
    __global const int *j,
    __global const int *level,
    __global const int *hash,
    __global struct neighbor2d *neigh2d
    ) {

	const uint ic = get_global_id(0);
	if (ic >= isize) return;

	int imaxsize = mesh_size*levtable[levmx];
	int jmaxsize = mesh_size*levtable[levmx];

	int ii = i[ic];
	int jj = j[ic]; 
	int lev = level[ic];
	int levmult = levtable[levmx-lev]; 
	
	int nlftval = hashval(jj*levmult, max(ii*levmult-1,0));
	int nrhtval = hashval(jj*levmult, min((ii+1)*levmult, imaxsize-1)); 
	int nbotval = hashval(max(jj*levmult-1,0), ii*levmult);
	int ntopval = hashval(min((jj+1)*levmult, jmaxsize-1), ii *levmult);
	neigh2d[ic].left = nlftval;
	neigh2d[ic].right = nrhtval;
	neigh2d[ic].bottom = nbotval;
	neigh2d[ic].top = ntopval;
}
```

**Remap calculations using a spatial perfect hash**
我们考虑将值从一个基于单元的AMR网格重映射到另一个基于单元的AMR网格，其中设置阶段(setup phase)和邻居查找一样，我们将每个单元的索引写入空间哈希桶内，重映射时，我们为源网格(source mesh)创建哈希表

在读阶段(read phase)时，用目标网格每个单元对应的单元号查询哈希表，在调整完单元的大小差异后，将源网格的单元值对应求和，填入目标网格中(queries the spatial hash for the cell numbers underlying each cell of the target mesh and sums up the values from the source mesh into the target mesh after adjusting for the size difference of the cells)

读阶段的CPU实现见listing5.8
listing5.8
```c
for(int jc = 0; jc < ncells_target; jc++) {
	int ii = mesh_target.i[jc];
	int jj = mesh_target.j[jc];
	int lev = mesh_target.level[jc];
	int lev_mod = two_to_the(levmx - lev);
	double value_sum = 0.0;
	
	for(int jjj = jj*lev_mod; jjj < (jj+1)*lev_mod; jjj++) {
		for(int iii = ii*lev_mod; iii < (ii+1)*lev_mod; iii++) {
			int ic = hash_table[jjj*i_max+iii];
			value_sum += value_source[ic] / (double)four_to_the( levmx-mesh_source.level[ic] );
		} 
	} 
	value_remap[jc] += value_sum;
}
```

**Table lookups using a spatial perfect hash**
从表格数据中查找值的操作提供了一种不同的局部性，可以通过空间哈希来利用，可以使用哈希来搜索两个轴(axes)上的插值区间(intervals for interpolation)
本例中，我们使用一个$51\times 32$的状态方程值查找表(lookup table of equatoin-of-state values)，两个轴分别是密度和温度，每个轴上的值之间使用相等的间距
用$n$表示轴的长度，$N$表示要执行的表查找(table lookups)的数量，在这项研究中，我们使用了三种算法：
- 第一种是从第一列和第一行开始的线性搜索(brute force)，对于每个数据查询，复杂度是$O(n)$，对于所有的 N，复杂度就是$O(N\times n)$，其中$n$表示每个轴的列数或行数
- 第二种是二分查找，它查看可能范围的中点值，并递归地缩小区间的位置(the location for the interval)，对于每个数据查询，复杂度是$O(logn)$
- 最后，我们使用哈希对每个轴的区间执行复杂度为$O(1)$的查找，我们测量了在单核CPU和GPU的上哈希算法的性能，测试代码在两个轴上搜索区间，并对表中的数据值进行简单的插值以得到结果

对于线性搜索，在单个轴上搜索区间时，只需要两次缓存加载(cache loads)，一共只需要四次缓存加载，二分查找也需要相同数量的缓存加载，因此考虑缓存加载，我们预计线性搜索和二分搜索的性能实际没有差异
哈希算法可以直接找到正确的区间，但也仍然需要一次缓存加载

将算法移植到GPU要复杂一些，我们首先看一下listing5.9中CPU上的哈希实现，代码循环遍历所有1600万个值，查找每个轴上的区间，然后对表中的数据进行插值以获得结果值，利用散列技术，我们可以通过一个简单的无条件的算术表达式(arithmetic expression with no conditionals)找到区间位置
listing5.9
```c
double dens_incr = (d_axis[50]-d_axis[0])/50.0;
double temp_incr = (t_axis[22]-t_axis[0])/22.0;

for(int i = 0; i < isize; i++){
	int tt = (temp[i] - t_axix[0])/temp_incr;
	int dd = (dens[i] - d_axix[0])/dens_incr;

	double xf = (dens[i] - d_axis[dd]) / 
				(d_axis[dd+1] - d_axis[dd]);
	double yf = (temp[i] - t_axis[tt]) /
				(t_axis[tt+1] - t_axis[tt]);
	value_array[i] = 
		xf * yf * data(dd+1,tt+1)
	+ (1.0-xf)* yf * data(dd, tt+1)
	+ xf *(1.0-yf) * data(dd+1,tt)
	+ (1.0-xf)*(1.0-yf)*data(dd, tt);
}
```
要将该代码移植到GPU，就和之前的例子一样，移除 `for` 循环，替换成对 `get_global_id` 的调用
但GPU每个工作组都包含一个共享的局部内存，可以存储大约4000个双精度浮点数，我们的表中有1173个浮点值，而两个坐标轴共有51+23个浮点值，因此可以放入工作组的局部内存中，以被工作组内的线程快速访问以及共享
listing5.10
```c
# define dataval(x,y) data[(x) + ((y) * xstride)]

__kernel void interpolate_kernel(
	const uint isize,
	const uint xaxis_size,
	const unit yaxis_size,
	const unit dsize,
	__global const double *xaxis_buffer,
	__global const double *yaxis_buffer,
	__local double *xaxis,
	__local double *yaxis,
	__local double *data,
	__global const double *x_array,
	__global const double *y_array,,
	__global double *value
	)
{
	const unit tid = get_local_id(0);
	const unit wgs = get_local_size(0);
	const unit gid = get_global_id(0);

	if (tid < xaxis_size)
		xaxis[tid] = xaxis_buffer[tid];
	if (tid < y_axis_size)
		yaxis[tid] = yaxis_buffer[tid];

	for (uint wid = tid; wid<d_size; wid+=wgs){
		data[wid] = data_buffer[wid];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	double x_incr = (xaxis[50]-xaxis[0])/50.0;
	double y_incr = (yaxis[22]-yaxis[0])/22.0;

	int xstride = 51;
	if (gid < isize) { 
		double xdata = x_array[gid];
		double ydata = y_array[gid];
		
		int is = (int)((xdata-xaxis[0])/x_incr);
		int js = (int)((ydata-yaxis[0])/y_incr);
		
		double xf = (xdata-xaxis[is])/ (xaxis[is+1]-xaxis[is]);
		double yf = (ydata-yaxis[js])/(yaxis[js+1]-yaxis[js]); 
		
		value[gid] = 
				xf * yf *dataval(is+1,js+1)
			+ (1.0-xf)* yf *dataval(is, js+1)
			+ xf *(1.0-yf)*dataval(is+1,js)
			+ (1.0-xf)*(1.0-yf)*dataval(is, js);
	}
}
```
代码中的第一部分用所有的线程协同地将数据装载入局部内存中，在进行插值计算前，需要一次同步，确保所有的数据都成功装载

**Sorting mesh data using a spatial perfect hash**
我们考虑对空间数据(spatial data)进行排序
我们可以用空间排序寻找最近邻、去重、简化范围搜索，进行图像输出等等一系列操作

我们考虑一维数据，最小单元大小是2.0，更大的单元格只允许是2的幂次，暂且先考虑仅有五种可能大小：2.0，4.0，8.0，16.0和32.0
每个单元的大小随机生成，单元之间随机排序

空间哈希排序的计算需要利用一些关于一维数据的信息，我们要知道$X$的最大值和最小值，以及要知道最小单元大小，依赖这些信息，我们计算可以保证完美哈希的桶索引(bucket index)：
$$b_k=\left[\frac {X_i-X_{min}}{\Delta_{min}}\right]$$
其中$b_k$就是要放置数据项的桶，$X_i$是单元的$x$轴坐标值，$X_{min}$是$X$的最小值，$\Delta_{min}$是任意两个邻近的$X$的距离的最小值

哈希排序的操作如图5.9所示
![[PHCP-Fig5.9.png]]
可以看到原数组中值之间最小的差是2.0，因此桶大小是2.0就可以保证不存在碰撞，而原数组中最小的值是0，因此桶位置的计算就通过$B_i = (X_i-0)/\Delta_{min} = X_i/2.0$得到
我们可以把值存在哈希表中，也可以把索引存在哈希表中

空间哈希排序算法的复杂度是$\Theta(N)$，而快排是$\Theta(NlogN)$，但空间哈希的使用要考虑问题的情况，以及需要更多内存空间

Listing 5.11: The spatial hash sort on the CPU
```c
uint hash_size= (uint) ((max_val - min_val) / min_diff);
int *hash = (int*)malloc(hash_size * sizeof(int));
memset(hash, -1, hash_size*sizeof(int));

for(uint i = 0; i < length; i++) {
	hash[(int)((arr[i]-min_val)/min_diff)] = i;
}

int count=0;
for(uint i = 0; i < hash_size; i++) {
	if(hash[i] >= 0) {
		sorted[count] = arr[hash[i]];
		count++;
	}
}
free(hash);
```

该算法的GPU实现需要在读阶段实现一个前缀和算法，以使得对排序后值的检索可以并行进行
#### 5.5.2 Using compact hashing for spatial mesh operations
在前一小节中，我们探索过用紧凑哈希进行近邻查找和重映射操作，其关键特点是我们不需要对每个空间哈希桶都写入，且可以通过处理碰撞来改进算法

紧凑哈希允许空间哈希被压缩，以使用更少的内存

**Neighbor finding with write optimization and compact hashing**
之间介绍的寻找最近邻的完美哈希算法在AMR网格的精细化级别数较少的时候效果很好，但当精细化级别有6级甚至更多时，一个最粗糙的单元需要写入至少64个哈希桶，一个最精细的单元只需要写入1个哈希桶，这会导致装载不平衡(load imbalance)以及并行执行时的线程发散(thread divergence)

线程发散即每个线程之间的工作量差异较大，导致需要等待最慢的(工作量最大的)线程的执行

我们可以通过图5.11的优化进一步改善完美哈希
![[PHCP-Fig5.11.png]]
第一项优化利用了邻居查询仅需要访问一个单元最外边的哈希桶的事实，因此不需要对内部的桶进行写入，进一步的分析发现只有边角和中点的哈希桶才会被访问，因此可以再减少写入次数，最后可以优化到每个单元仅需写入一个哈希桶，但在近邻查询时需要进行多次读，且要求哈希表中桶的初始值为-1表示该项无效(no entry)

因为此时要往哈希表中写的数据更少了，我们可以对哈希表进行压缩
哈希装载因子(hash load factor)定义为哈希表中有填充的项数除以哈希表大小，哈希表的大小乘子(size multiplier)则是哈希装载因子的倒数，例如大小乘子是$1.25$时，装载因子就是$0.8$

哈希稀疏性(hash sparsity)表示了哈希表中的空闲空间比例，稀疏的哈希表说明可以对其进行压缩

我们一般选择哈希装载因子$=0.333$，或者说大小乘子$=3$，这是因为在并行处理中，我们希望避免一个处理器慢于其他处理器的情况

图5.12展示了创建紧凑哈希的过程
![[PHCP-Fig5.12.png]]
在将完美哈希压缩为紧凑哈希的过程中，可能会发生冲突，就需要进行冲突处理，例如上图就利用了开放定址法(open addressing)解决冲突，即发生冲突时，寻找下一个可用的桶
存在一些其他的冲突解决方法，但它们常常需要在运行时分配内存，而在GPU中分配内存是比较困难的，因此可以直接使用开放定址法，即在已经分配好内存的哈希表中寻找下一个可用的桶

在开放定址法中，寻找下一个桶也有多种方法，包括：
- 线性探查(Linear probing)：顺序寻找下一个开放的桶，直到找到
- 二次探查(Quadratic probing)：增量是线性探查的平方，即$+1,+4,+9,\cdots$
- 二次哈希(Double hashing)：使用第二个哈希函数得到(确定的，且伪随机的)增量
在选择下一个位置时使用更复杂的模式可以避免哈希表中的聚集现象，即许多值聚集在一个部分，而其他部分是空的
我们使用二次探查，二次探查中，前几次的寻址可以在缓存中，因此速度更快，在查询/读取时，需要对查询值和存储值进行比较，如果不同，转下一个位置查询

我们计算写和读的次数来进行方法之间的比较，但要注意的是对具体值的写和读的次数需要调整为对缓存行的写和读的次数，比较才有实际意义，同时，优化了的哈希方法也有了更多的条件语句，因此，优化带来的性能提升可能不明显，在GPU上，由于优化可以减少线程发散，因此提升效果要更好

**Face neighbor finding for unstructured meshes**
目前为止，我们还没有讨论过关于非结构化网格的算法，因为很难保证可以为非结构化网格创建完美哈希

寻找多边形网格的邻接面(neighbor face)是一项昂贵的搜索过程，许多代码会选择存储邻接图(neighbor map)

Example: finding the face neighbor for an unstructured mesh展示了用空间哈希进行邻接面检索的操作，描述如下：
首先，找到每个面的中心点对应的哈希桶；其中，哈希桶有两个项，如果该面是朝向左上的，则单元的索引写入桶中的第一项，否则写入桶中的第二项；在搜索时，检查每个面中心的哈希桶中是否两项都有值，如果有，则这个哈希桶对应了一个邻接面，否则对应了一个朝外的面

我们用一次写和一次读实现了邻接面的查找

**Remaps with write optimizations and compact hashing**
我们将每个单元的单元索引只写入其对应的哈希桶范围的左下角的哈希桶，在读取过程中，如果没有找到值或该单元在输入网格(input mesh)中的级别不正确，我们寻找在输入网格中的单元如果在下一个更粗糙的级别会写入的哈希桶

图5.14显示了这种方法，其中输出网格(output mesh)中的单元1查询散列位置(0, 2)并找到一个–1，因此它会查找如果它在下一个更粗糙的级别会写入的散列位置：(0, 0)，并找到单元索引(cell index)1，然后将输出网格中的单元1的密度(density)设置为输入网格中的单元1的密度
对于在输出网格中的单元格9，它在查找散列位置(4, 4)，找到单元索引3，然后它在输入网格中查找单元3的级别，由于输入网格单元3的级别更精细，它还必须查询散列位置(6, 4), (4, 6), (6,6)，获得了单元索引9、4、7，前两个单元和单元3处于同一精细化水平，因此不需要再进一步，单元7处于更精细的级别，所以我们必须递归地继续查找到单元索引8、5和6

这就是空间哈希重映射的一个单次写，多次读的实现，代码见Listing5.12-5.14

Listing5.12 The setup phase for the single-write spatial hash remap on the CPU
```c
#define two_to_the(ishift) (1u << (ishift))

typedef struct {
	uint ncells; // number of cells in the mesh
	uint ibasesize; // number of coarse cells across the x-dimension
	unit levmax; // number of refinement levels in addition to the base mesh
	uint *dist;
	uint *i;
	uint *j;
	uint *level;
	double *values;
} cell_list;

cell_lsit icells, ocells;
// <.. lots of code too create mesh ...>

size_t hash_size = icells.ibasesize*two_to_the(icells.levmax) *
				   icells.ibasesize*two_to_the(icells.levmax)
int *hash = (int *) malloc(hash_size * sizeof(int));
uint i_max = icells.ibasesize * two_to_the(icells.levmax);
```
在写之前，要分配好哈希表，并全部写入哨兵值-1

Listing5.13 The write phase for the single-write spatial hash remap on the CPU
```c
for (uint i = 0; i < icells.ncells; i++){
	// the multiplier to convert between mesh levels
	uint lev_mod = two_to_the(icells.levmax - icells.level[i]);
	// computes the index for the ID hash table
	hash[((icells.j[i] * lev_mod) * i_max) + (icellls.i[i] * lev_mod)] = i;
}
```

Listing5.14 The read phase for the single-write spatial hash remap on the CPU
```c
for (uint i = 0; i < ocells.ncells; i++){
	uint io = ocells.i[i];
	uint jo = ocells.j[i];
	uint lev = ocells.level[i];

	uint lev_mod = two_to_the(ocells.levmax - lev);
	uint ii = io*lev_mod;
	uint ji = jo*lev_mod;

	uint key = ji*i_max + ii;
	int probe = hash[key];

	if (lev > ocells.levmax){lev = ocells.levmax;}

	while(probe < 0 && lev > 0){ // if a sentinel value is found, contiunes to coarser levels
		lev--;
		uint lev_diff = ocells.levmax - lev;
		ii >>= lev_diff;
		ii <<= lev_diff;
		ji >>= lev_diff;
		ji <<= lev_diff;
		key = ji*i_max + ii;
		probe = hash[key];
	}
	if(lev >= icells.ilevel[probe]){
		ocells.values[i] = icells.values[probe];
	}else { 
		ocells.values[i] = avg_sub_cells(icells, ji, ii, lev, hash);
	}
}
```

**Hierarchical hash technique for the remap operation**
另一种利用哈希进行重映射的方法是层次化的哈希以及“面包屑”技术(它使得我们不需要在开始时将哈希表初始化为哨兵值)

第一步是为网格的每一个精细化级别都分配一个哈希表，然后将单元格索引写入恰当级别的哈希表，然后递归向上(到更粗糙的级别)留下一个哨兵值，这使得查询可以知道在更精细化的哈希表中有值

如图5.15，对于输入网格中的单元格9，我们看到单元格索引被写入了中间级别的哈希表，然后再更粗糙级别的哈希表中的对应哈希桶中写入和哨兵值；输出网格中对于单元格9的读操作从最粗糙的哈希表开始，读到-1，知道要继续向下；它在中间级别的哈希表对应的4个哈希桶中找到三个单元格索引，以及一个-1，因此要继续向下，在最精细化的哈希表中又找到四个单元格索引；输出网格中其他的单元格都在最粗糙的哈希表中找到了单元格索引，因此直接将对应值赋值给输出网格的单元格

每个哈希表可以是完美哈希也可以是紧凑哈希，该方法也是递归结构，如果递归层数不深，也可以在GPUs上运行
### 5.6 Prefix sum(scan) pattern and its importance in parallel computing
前缀和运算，也称为扫描(scan)，是在不规则大小(irregular sizes)情况下常进行的运算，例如并行时，在不规则大小情况下的运算需要知道自己的起始位置，就需要用到前缀和运算

一个简单的例子就是每个处理器都有不同数量的项，要并行地将数据一起写入输出数组，或访问其他处理器/线程的数据，此时，使用前缀和，我们可以为每个处理器确定它在输出数组进行写入的起始位置

前缀和的计算可以包括(inclusive)当且所在的值，也可以不包括(exclusive)
![[PHCP-Fig5.16.png]]
如图5.16所示，不包括性的扫描得到的前缀和数组中，对应位置的元素指示了处理器/进程/线程在全局输出数组进行写入的起始索引，包括性的扫描中，则指示了结束索引

Listing5.15 The serial inclusive scan operation
```c
y[0] = x[0];
for (int i = 1; i < n; i++){
	y[i] = y[i-1] + x[i];
}
```
在扫描完成后，每个处理器/进程/线程在全局数组的写入位置就确定了，此时就可以利用前缀和数组进行并行写入操作
前缀和本身的计算是串行的，当前迭代(iteration)的计算依赖于上一个迭代的计算结果，但也存在将其并行化的方法
#### 5.6.1 Step-efficient parallel scan opeartion
步骤效率的算法使用尽可能最少的步骤(fewest number of steps)，但这不意味着操作数(number of operations)最少，因为每一步的操作数都有可能不同
(注意步骤之间是串行的)

前缀和运算可以通过基于树的归约模式并行化，如图5.17所示
![[PHCP-Fig5.17.png]]
该算法中，输出数组的每个元素没有等待前一个元素完成部分和的计算，而是将自己的值和自己的前一个元素的值进行求和，然后写入自己的位置，然后再做同样的操作，此时向前看两个位置，接着是先前看四个元素，以此类推

该算法的效果等价于做一次包括性扫描，该算法中，所有的处理器都是忙碌的，该算法只需要$O(\log_2n)$步，但工作量(amount of work)实际上大于串行算法
#### 5.6.2 Work-efficient parallel scan operation
工作效率的算法使用最少数量的操作(least number of operations)，它的步骤数不一定是最少的，因为每一步骤可能有不同数量的操作数

选择工作效率算法或步骤效率算法取决于并行进程可以同时存在的数量

我们对数据进行两次扫描(sweeps)，第一次扫描称为上扫(upsweep)，如图5.18所示
![[PHCP-Fig5.18.png]]

第二次扫描称为下扫(downsweep)，它先将数组的最后一个值设为0，然后从右到左再进行一次基于树的扫描，如图5.19所示
![[PHCP-Fig5.19.png]]
该算法的工作量少，但步骤数多，效果等价于不包括性扫描
可以看到该算法在上扫时，一开始只有一半的线程忙碌，最后仅剩一个线程忙率，在下扫/扫回时，一开始只有一个线程忙碌，到最后所有的线程都忙碌

该算法多出的额外的步骤使得可以复用之前计算的部分和结果，因此工作量是$O(N)$的

注意这两个算法都受CPU处理器数量或GPU工作组内线程数量的限制
#### 5.6.3 Parallel scan operations for large arrays
对于更大的数组，需要新的并行算法，如图5.20所示
![[PHCP-Fig5.20.png]]
该算法用到了三个GPU计算核，第一个计算核在每个工作组内进行一次归约求和，然后将这些部分和存到一个数组中，此时这个数组的大小已经远远小于了原来数组的大小(GPU中一个工作组内的线程数量可以高达1024)
第二个计算核遍历该数组，执行一次扫描/前缀和，得到的前缀和数组内存储着每个工作组的偏移(offset)
第三个计算核在每个工作组内进行扫描，利用该工作组的偏移量，每个线程得以计算正确的前缀和，最后写入输出数组中

并行的前缀和计算十分重要，因此该运算在GPU架构上是高度优化的
### 5.7 Parallel global sum: Addressing the problem of associativity
在有限精度算术中，改变加法的顺序会改变最终结果，即浮点加法不满足结合性

有限精度加法导致的问题例如：
- 当问题的规模逐渐扩大，最后加上的值对总体和的影响会逐渐变小，直至不改变和
- 当两个绝对值近乎一样，但符号不同的值相加，会发生灾难性消除(catastrophic cancellation)，导致结果的有效数位非常少，大部分数位是噪声
	例如运行 `x = 12.15692174374373 - 12.15692174374372` ，
	得 `x = 1.06581410364e–14`

并行运算会并行计算部分和，最后求和部分和得到全局和，其加法顺序和串行运算是不同的，因此得到的全局和也会不同，即全局和问题(issue)
求解全局和的模式(pattern)一般称为归约(reduction)

为了得到更精确的和，有几种方法：
一种基于排序的方法将数根据数量级(magnitude)从小到大排序，然后以串行顺序求和，以尽量避免对两个数量级相差较大的数求和导致结果的有效数位较少的问题

最简单的方法是使用更高的精度，如x86架构提供的 `long double` 类型(80bit)，但这种方法不具备可移植性

成对和(pairwise summation)是一种可以较快求解全局和的算法，时间复杂度$log_2n$但需要额外的空间

Kahan和是更高精度求解全局和最实用的方法，它使用额外的变量记录两个数量级相差较大的数中丢失的低位比特，作为修正项/补偿项
```js
function KahanSum(input)
    // Prepare the accumulator.
    var sum = 0.0
    // A running compensation for lost low-order bits.
    var c = 0.0
    // The array _input_ has elements indexed input[1] to input[input.length].
    for i = 1 to input.length do
        // c is zero the first time around.
        var y = input[i] - c
        // Alas, sum is big, y small, so low-order digits of y are lost.         
        var t = sum + y
        // (t - sum) cancels the high-order part of y;
        // subtracting y recovers negative (low part of y)
        c = (t - sum) - y
        // Algebraically, _c_ should always be zero. Beware
        // overly-aggressive optimizing compilers!
        sum = t
    // Next time around, the lost low part will be added to _y_ in a fresh attempt.
    next i

    return sum
```
Kahan和每次循环内执行了四次浮点运算，而非一次
# CPU: The parallel workhorse
## 6 Vectorization: FLOPs for free
大部分应用程序是内存受限(memeory bound)

### 6.1 Vectorization and single instruction, multiple data(SIMD) overview
SIMD中，一个向量指令 `add` 会替换指令队列中(instructoin queue)八个独立的标量指令 `add` ，减轻指令队列和cache的压力

SIMD中，可以指定生成的向量指令集(vector instruction set)
大多数编译器默认生成SSE2(Streaming SIMD Extensions)指令，可以运行于任意硬件，SSE2的双精度向量长度(vector length)最高只有二

AVX(Advanced Vector Extensions)向量位宽为256bit，双精度向量长度最高到四，可以运行于2011年后的任意硬件

### 6.2 Hardware trends for vectorization
AVX中，AMD支持融合乘加(fused multiply-add/FMA)向量指令
AVX2中，Intel支持融合乘加(fused multiply-add/FMA)向量指令
Intel和AMD自2018年起支持AVX512

### 6.3 Vectorization methods
#### 6.3.1 Optimized libraries provide performance for little effort
常用的高度优化的库有：
- BLAS(Basic Linear Algebra System)
- LAPACK(linear algebra package)
- SCALAPACK(scalable linear algebra package)
- FFT(Fast Fourier transform)
- Sparse Solvers
The Intel Math Kernel Library(MKL)实现了以上库的优化版本

#### 6.3.2 Auto-vectorization: The easy way to vectorization speedup(most of the time)
自动向量化即编译器对源码的向量化
注意如果设立向量化指令的额外开销高于向量化带来的表现提升，向量化反而会使程序更慢

自动向量化通过命令行的编译选项即可实现
对于GCC，即 `-ftree-vectorize` (`-O3` 默认包括)

C99标准添加了 `restrict` 关键字，用以指定指针指向的内存区域为专用区域
C++中则对应 `__restrict` 或 `__restrict__` 属性

对于GCC，`-fstrict-aliasing` 选项( `-O2` 默认包括)提示编译器代码中不存在别名使用(aliasing)
Aliasing即不同指针指向的内存区域有交叉，这会限制编译器生成向量化代码

推荐 `restrict` 和 `-fstrict-aliasing` 一起使用，`restrict` 可移植，对于不同的编译器则对应的选项有差异

#### 6.3.3 Teach the compiler through hints: Pragrams and directives
Pragma(编译指示/杂注)的形式为以 `#pragma` 开头的预处理语句 

条件块(conditional block)中存在除法时，`-fno-trapping-math` 提示编译器不必担心计算会抛出异常，以允许向量化，条件块中存在 `sqrt` 时，`-fno-math-errno` 提示编译器不必担心存在错误，以允许向量化

可以使用杂注( `private` 子句)提示编译器变量的值在循环之内不会保存，不存在流依赖(更好的方法是在循环体内声明并定义变量)
例如 `#pragma omp simd private(xspeed, yspeed)`

可以使用杂注注明归约(reduction)变量，杂注注明的归约变量会由OpenMP自动初始化为0，向量化后，每个向量路径(vector lane)都会有独自的归约变量拷贝，在 `for` 循环结束后再一起相加
例如 `#pragma omp simd reduction(+:summer)`

- Flow dependency(流依赖)：循环内的一个变量在写后被读(read-after-write)
- Anti-flow dependency(逆流依赖)：循环内的一个变量在读后被写(write-after-read)
- Output dependency(输出依赖)：循环内一个变量被写入多次

- Peel loop(剥离循环)：在主循环之前的循环，为未对齐的数据执行的循环，以保证主循环一定是对齐的
- Remainder loop(剩余循环)：在主循环之后的循环，处理由于太小而达不到全部向量长度的小部分数据

#### 6.3.4 Crappy loops, we got them: Use vector intrinsics
如果使用杂注仍然让编译器向量化程序，可以使用向量内部函数，但这是一个移植性较低的方法

最常用的内部函数集是Intel x86 vector intrinsics，可以运行于Intel和AMD的处理器上，且支持AVX向量指令

GCC vector extensions可以支持x86架构以外的其他架构，但仅限使用GCC编译器
如果指定的向量长度大于硬件支持的向量长度，编译器会生成结合短向量长度以达到大向量长度的指令

Fog vector class是一个开源C++向量类库，具有可移植性，并自动适应向量长度较短的旧硬件

注意数组的长度并不总是是向量长度的倍数，我们可以添加额外的零使数组长度满足倍数关系，也可以在另一部分处理余下的数

#### 6.3.5 Not for the faint of heart: Using assembler code for vectorization
直接使用向量汇编指令进行编程有最大概率达到最高效率，但可移植性受很大限制

汇编语言中使用到了 `ymm` 寄存器说明生成了向量指令，`zmm` 寄存器说明生成了AVX512向量指令，`xmm` 寄存器说明生成了标量指令或SSE向量指令

### 6.4 Programming style for better vectorization
- 在函数参数的指针类型声明中使用 `restrict` 属性
- 在最内部循环使用长度最长的数据结构
- 使用所需的最小数据类型( `short` vs `int` )
- 使用连续(contiguous)内存访问
- 使用Structures of Arrays(SOA)而不是Array of Structures(AOS)
- 在循环体内部定义局部变量
- 循环体内部的数组或变量要么只读要么只写
- 不要在循环体内部为不同的目的重用局部变量

### 6.5 Compiler flags relevant for vectorization for various compilers
不显式设置编译选项，编译器默认使用SSE2向量指令集
设置 `-march=native` / `-xHost`  / `-qarch=pwer9`，编译器使用本地向量指令集

Intel编译器中设置 `-axmic-axv512 -xcore-avx2` 使用AVX2指令集

对包含条件语句的循环向量化时，编译器会插入mask以只使用部分的向量结果
此时要通过编译选项指示编译器masked operations不会产生错误，如 `-fno-trapping-math -fno-math-errno`

### 6.6 OpenMP SIMD directives for better portability
C/C++的通用杂注格式是
```c
#pragma omp simd / Vectorizes the following loop or block of code
#pragma omp for simd / Threads and vectorizes the following loop
```

可以添加额外的子句加入更多信息
最常用的子句是 `private` 子句，该子句为每个向量通路创造一个独立的私有的变量，例如
```c
#pragma omp simd private(x)
for(int i=0; i<n; i++){
	x = array(i);
	y = sqrt(x)*x;
}
```
在循环体内定义变量可以使意图更明显从而替代简单的 `private` 子句
```c
double x=array(i);
```

`firstprivate` 子句为每个线程初始化变量，值为进入循环的值
`lastprivate` 子句为每个线程初始化变量， 值为循环后的值(等于逻辑上串行执行循环得到的值)

`reduction` 子句为每个通路创造并初始化私有变量，并在循环最后对每个通路的变量执行指定的运算(如求和)

`aligned` 子句告诉编译器数据以64byte为界对齐，数据对齐后就不必生成剥离循环，对齐的数据也可以更高效装载入向量寄存器
但在此之前要以对齐的方式分配好内存，可以使用函数如：
```c
void *memalign(size_t alignment, size_t size); 
int posix_memalign(void **memptr, size_t alignment, size_t size); 
void *aligned_alloc(size_t alignment, size_t size); 
void *aligned_malloc(size_t alignment, size_t size);
```

也可以使用属性：
```c
double x[100] __attribute__((aligned(64)));
```

`collapse` 子句告诉编译器将嵌套的循环结合为单个循环以向量化，参数指定了多少个循环参与结合：
```c
#pragma omp collapse(2)
for (int j=0; j<n; j++){ 
	for (int i=0; i<n; i++){ 
		x[j][i] = 0.0; 
	} 
}
```
循环要求使完美嵌套的，即只有最内层循环有语句

杂注也可以用于向量化函数，如
```c
#pragma omp declare simd
double pythagorean(double a, double b){ 
	return(sqrt(a*a + b*b)); 
}
```

## 7 OpenMP that performs
OpenMP(Open Multi-Processing)是一个共享内存(shared memory)编程标准
### 7.1 OpenMP Introduction
OpenMP为使用多线程的共享内存并行程序提供了一个标准且可移植的API，多数编译器支持OpenMP
#### 7.1.1 OpenMP concepts
松弛(relaxed)内存模型允许线程竞争情况(therad race conditions)存在
松弛(relaxed)的含义是变量在内存中的值不会立即更新(not updated immediately)

- 松弛内存模型(Relaxed memory model)
	变量在内存中的值或在所有处理器中的值不会立即更新
- 竞争情况(Race conditions)
	程序会由于贡献者的不同时间安排出现不同的运行结果的情况

- 私有变量(Private variable)
	OpenMP语义下指仅对其线程可见的本地私有变量
- 共享变量(Shared variable)
	OpenMP语义下指仅对所有线程可见的共享变量

- 作业共享(Work sharing)
	将作业划分给多个线程或进程
- 首次访问(First touch)
	对一个数组的首次访问将导致其内存被分配，分配位置将接近于首次访问它的线程的位置
	在首次访问之前，数组的内存仅存在于虚拟内存表的一个表项中，和虚拟内存想对应的物理内存将在首次访问后被分配

在NUMA模型中，一个处理器访问非本地的内存区域往往要花费原来的两倍时间

OpenMP使用松弛内存模型，因此线程间需要通过内存通信时，一般需要屏障(barrier)或刷新操作(flush operation)进行同步以防止线程竞争

OpenMP应用场景是单个计算节点，而不是具有分布式内存的多个计算节点模型

OpenMP的常用杂注：
- `#pragma omp parallel`
	为其后紧随的区域生成多线程
- `#pragma omp for
	为多线程等分作业
- `#pragma omp parallel for reduction` + `(+:sum)` / `(min:xmin)` / `(max:xmax)` 
	归约
- `#pragma omp barrier`
	创造一个停止点，以便进入下一个区域时所有的线程可以重新部署
- `#pragma omp masked`
	指定线性执行区域，仅由线程0执行，区域末端没有屏障
- `#pragma omp single`
	指定线性执行区域，仅由一个线程(不一定是线程0)执行，区域末端有屏障
- `#pragma omp critical or atomic`
	锁
#### 7.1.2 A simple OpenMP program
设置并行区域使用的线程数量的方式：
- 默认
	默认情况一般是结点支持的最大线程数量，也取决于编译器
- 环境变量
	由 `OMP_NUM_THREADS` 设定，例如 `$ export OMP_NUM_THREADS=16`
- 函数调用
	由OpenMP函数 `omp_set_threads` 设定，如 `omp_set_threads(16)`
- 杂注
	例如 `#pragma omp parallel num_threads(16)`

在并行区域之外定义的变量，默认是线程间的共享变量，在并行区域之内定义变量，则每个线程都会得到其自己的私有变量

### 7.2 Typical OpenMP use cases: Loop-level, high-level, and MPI plus OpenMP
OpenMP一般有三种使用场景：
- 循环层次(loop-level)的OpenMP
- 高层次(high-level)的OpenMP
- MPI和OpenMP
#### 7.2.1 Loop-level OpenMP for quick parallelization
应用程序只需要适当的提速且(单个结点上)内存资源充足是典型的循环级别OpenMP使用场景，它的特征有
- 适当并行
- 内存资源充足(低内存需求)
- 计算昂贵部分仅限于部分 `for` / `do` 循环

#### 7.2.2 High-level OpenMP for better parallel performance
高层次OpenMP从整个系统视角入手，包括内存、OS内核、硬件等
高层次OpenMP帮助我们消除大部分线程启动开销和线程同步开销

#### 7.2.3 MPI plus OpenMP for extreme scalability
仅在同一个NUMA区域内使用OpenMP，区域之间通过MPI通信

### 7.3 Examples of standard loop-level OpenMP
OpenMP依靠OS内核做内存处理，这是限制OpenMP表现的一个潜在因素

OpenMP采用松弛内存模型，每个线程仅有对内存的暂时视图

通过环境变量启用线程绑定
```shell
$ export OMP_PLACES=cores
$ export OMP_CPU_BIND=true
```
#### 7.3.1 Loop level OpenMP: Vector addition example
#### 7.3.2 Stream triad example
#### 7.3.3 Loop level OpenMP: Stencil example
#### 7.3.4 Performance of loop-level examples
#### 7.3.5 Reduction example of a global sum using OpenMP threading
```c
double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for(long i = 0; i< ncells; i++){
	sum+=var[i];
}
return sum;
```
归约操作会为每个线程都初始化一个 `sum` ，每个线程计算局部和，最后局部和相加得到全局和

#### 7.3.6 Potential loop-level OpenMP issuse
可以被OpenMP优化的循环必须是规范形式：
- 循环索引(index)变量必须是整数
- 循环索引不能在循环体内被更改
- 循环必须有标准退出条件(exit conditions)
- 循环迭代次数必须是可数的
- 循环不能有任何循环携带(loop-carried)的依赖
如果逆转循环顺序，或任意改变循环顺序，答案变化，则该循环内存在循环携带的依赖

- 细粒度并行(fine-grained parallelization)
	在并行计算中，多处理器或多线程之间需要经常性的同步
- 粗粒度并行(corse-grained parallelization)
	在并行计算中，多处理器或多线程之间不需要经常性的同步

### 7.4 Variable scope importance for correctness in OpenMP
通常来说，栈上的变量一般认为私有，堆上的变量一般认为公有
高层次OpenMP中最重要的是管理并行区域内被调用例程中的变量作用域

私有的变量最好在循环内声明

### 7.5 Function-level OpenMP: Making a whole function thread parallel
函数内声明的变量默认都为私有变量，而 `static` 修饰可以让变量的作用域扩展到文件级别，使变量在线程间公有，如 `static double *x;`

### 7.6 Improving parallel scalability with high-level OpenMP
高层次OpenMP的核心策略是通过最小化fork/join开销和内存延迟(latency)以在循环层次的OpenMP上进一步优化
(通过显式地在线程之间划分工作/显式地控制同步点)减小线程等待时间也是高层次OpenMP的策略之一
标准OpenMP采用典型的fork-join模型，而高层次OpenMP会使线程睡眠(dormant)以保持线程存活(alive)以减少开销

#### 7.6.1 How to implement high-level OpenMP
实现高层次OpenMP的步骤
- 基础实现(base implementation)
	实现循环级别OpenMP
- 减少线程启动(reduce thread start up)
	融合并行区域，将所有循环级别并行结构联合为一个更大的并行区域
- 同步(synchronization)
	为不需要同步的 `for` 循环加入 `nowait` 子句，并且人工为线程划分循环，以移除屏障
- 优化(optimize)
	若可能，让数组和变量对每个线程私有
- 代码纠错(code correctness)
	每一步后都要检查代码防止竞争情况出现

高层次OpenMP中，线程仅在程序执行前生成一次，执行中不用的线程保持睡眠(例如执行串行区域时)，需要使用时被唤醒

人工划分循环/数组的好处在于它通过不允许线程间共享内存中的相同的空间减少缓存抖动和竞争情况

#### 7.6.2 Example of implementing high-level OpenMP

### 7.7 Hybrid threading and vectorization with OpenMP
利用OpenMP多线程执行的循环可以通过加入 `simd` 子句同时将其向量化，即使用杂注 `#pragma omp parallel for simd`

### 7.8 Advanced examples using OpenMP
#### 7.8.1 Stencil example with a seperate pass for x and y directions
在两步(two-step)模板运算中，不同面上的数组会有不同的数据共享要求
例如
要求x面数组 `Xface[2][6]` 对每个线程都是私有的，因此采用 `double **xface = malloc(12 * sizeof(double));` 分配内存，每个线程都拥有自己独立的指针和相应指向的堆上内存
要求y面数组 `Yface[2][6]` 对所用线程公有，因此采用 `if(thread_id == 0) static double **y = malloc(12 * sizeof(double));` 分配内存，指针为静态变量，为所有线程共享

遵循首次接触原则，如果可能，让一个线程所要访问的数据/数组部分对其私有，可以提高内存局部性

使用OpenMP执行模板操作时，我们需要决定每个线程的内存是私有的还是公有的，例如将二维数组的计算按行划分给线程时，若在进行x轴上的模板计算，数据可以是对每个线程私有的，以加快速度，而在进行y轴上的模板计算时，由于需要访问相邻线程的数据，数据就要求是共享的

超线性加速比(super-linear speedup)发生于当作业在线程之间划分时，缓存性能也同时提升

#### 7.8.2 Kahan summation implementation with OpenMP threading
Kahan和存在循环携带的依赖

#### 7.8.3 Threaded implementation of the prefix scan algorithm
前缀和的多线程执行分为三个阶段
- 所有线程
	为各自的数据部分计算前缀和，起始偏移都为0
- 一个线程
	为每个线程计算全局的起始偏移
- 所有线程
	根据正确的起始偏移修正各自的前缀和

### 7.9 Threading tools essential for robust implementations
- Valgrind
	用于寻找未初始化的内存和线程的越界访问
- Call graph
	由cachegrind工具生成
- Allinea/ARM Map
	一个高层次的分析工具，用于得到关于线程启动和屏障的总体开销
- Intel Inspector
	探查线程竞争情况
#### 7.9.1 Using Allinea/ARM MAP to get a quick high-level profile of your application
#### 7.9.2 Finding your thread race conditions with Intel Inspector

### 7.10 Example of a task-based support algorithm
基于任务的并行方法将任务划分为独立的子任务然后将其发布给多个线程

可复现(reproducible)全局和的一个方法就是成对(pairwise)求和，该算法可以利用基于任务的并行方法，递归地对半划分数组，直到长度为1，然后自下到上求和，省去了基于数据的并行方法中人工确定线程的起始索引的工作
![[PHPC-Fig7.16.png]]

## 8 MPI: The parallel backbone
Message Passing Interface(MPI)标准允许程序访问额外的计算结点
消息传递(message passing)指进程间的消息传递

MPI标准目前最常用的两个实现是MPICH和OpenMPI

### 8.1 The basics for an MPI program
MPI是一个完全基于库(library-based)的语言
MPI程序总是以程序开头的 `MPI_INIT` 调用开始，并以程序结尾的 `MPI_Finalize` 调用结束(OpenMP不需要特殊的开始或结束命令，只需要在关键循环周围写上杂注)

#### 8.1.1 Basic MPI function calls for every MPI program
基础MPI函数调用包括 `MPI_INIT` 和 `MPI_Finalize` 
`main` 的参数应该传递给 `MPI_INIT` 
```c
#include<mpi.h>
int main(int argc, char *argv[]){
	MPI_INIT(&argc, &argv);
	MPI_Finalize();
	return 0;
}
```

一组可以互相交流的进程称为交流组(communicator)，其中每个进程有自己的进程号(process rank)
默认的交流组是 `MPI_COMM_WORLD` ，它由 `MPI_INIT` 设立

#### 8.1.2 Compiler wrappers for simpler MPI programs
MPI是一个库，但通过MPI编译器包装器(wrapper)，我们可以将其视为一个编译器

- mpicc
	C代码的包装器
- mpicxx/mpic++/mpiCC
	C++代码的包装器
- mpifort/mpif77/mpif90
	Fortran代码的包装器

#### 8.1.3 Using parallel startup commands
MPI并行进程的启动是一个复杂的操作，它通过启动命令执行
- `mpirun -n <nprocs>`
- `mpiexec -n <nprocs>`
- `aprun`
- `srun`

多数MPI启动命令通过 `-n` 选项指定进程数

#### 8.1.4 Minimum working example of an MPI program
```C
#include<mpi.h>
#include<stdio.h>

int main(int argc, char **argv){
	MPI_INIT(&argc, &argv);

	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // gets the rank number of the process
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // gets the number of ranks in the program determined by the mpirun command

	printf("Rank %d of %d\n", rank, nprocs);

	MPI_Finalize();
	return 0;
}
```

### 8.2 The send and receive commands for process-to-process communication
消息传递方法的核心就是点对点发送消息，或者说进程对进程发送消息
进程并行的关键在于协调工作(coordinate work)

传递消息的两端系统都需要由有各自的邮箱(mailbox)

信息由三元组构成：指向内存缓存区的指针、计数器、类型
类型数据在异质系统间(比如大端序机器和小端序机器)传递信息时很有用，起到指导类型转换的作用
信封包含了三项信息：信息的发送方、信息的接收方、信息标识符(identifier)，实际由三元组构成：rank、tag、comm group

相较于阻塞式(blocking)的发送和接收，异步(asynchronous)/立即(immediate)式的发送和接收往往更安全和快速

最基础的MPI接收和发送函数是 `MPI_Recv` 和 `MPI_Send` ，函数原型为
```c
MPI_Send(void *data, int count, MPI_Datatype datatype, int dest, int tag, MPI_COMM comm);
MPI_Recv(void *data, int count, MPI_Datatype datatype, int source, int tag, MPI_COMM comm, MPI_Status *status);
```
这两个调用都是阻塞式的，`MPI_Send` 只有在缓存已经被读过并且不再需要时返回，`MPI_Recv` 只有在缓存已经被装满时返回
采用阻塞式调用时，一定要小心死锁

`MPI_Sendrecv` 是一种集体通信调用(collective communication call)，它结合了 `MPI_Send` 和 `MPI_Recv` 的功能，二者具体的执行顺序将由MPI决定
使用集体通信调用可以将避免死锁和提高速度的任务交给MPI，是较好的实践

我们还可以使用立即式(immediate)调用，包括 `MPI_Isend` 和 `MPI_Irecv` ，立即式调用即会立即返回的调用，因此它们的执行是异步的或非阻塞的，异步的(asynchronous)意为调用不会等待工作完成才返回

使用异步调用时，要小心不要在工作完成前修改发送缓存(send buffer)和访问接收缓存(receive buffer)

`MPI_Send` 和 `MPI_Recv` 的变体还包括：
- `B` (buffered)
- `S` (synchronous)
- `R` (ready)
- `IB` (immediate buffered)
- `IS` (immediate synchronous)
- `IR` (immediate ready)

MPI为C预定义的常用数据类型包括：
- `MPI_CHAR` 
- `MPI_INT`
- `MPI_FLOAT`
- `MPI_DOUBLE`
- `MPI_PACKED` (泛用的单字节数据类型，常用于混合类型的数据)
- `MPI_BYTE` (泛用的单字节数据类型)

`MPI_PACKED` 和 `MPI_BYTE` 可以匹配任意其他类型，`MPI_BYTE` 可以用于避免异质通信中的数据类型转换，`MPI_PACKED` 一般用于 `MPI_PACK` 调用中

MPI还包含了一些通信完成测试调用，包括：
```c
int MPI_Test(MPI_Requeset *request, int *flag, MPI_Status *status);
int MPI_Testany(int count, MPI_Request requests[], int *flag, MPI_Status statuese[]);
int MPI_Testall(int count, MPI_Request requests[], int *flag, MPI_Status statuese[]);
int MPI_Testsome(int incount, MPI_Request requests[], int *outcount, int indeces[], MPI_Status statusse[]);
int MPI_Wait(MPI_Request requests[], MPI_Status *status);
int MPI_Waitany(int count, MPI_Request requests[], int *index, MPI_Status *status);
int MPI_Waitall(int count, MPI_Request requests[], int *index, MPI_Status *status);
int MPI_Waitsome(int incount, MPI_Request requests[], int *outcount, int indeces[], MPI_Status statusse[]);
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
```

### 8.3 Collective communication: A powerful component of MPI
集体通信调用对一个MPI通信组里的所有进程进行操作，要对其中的一部分子集进行操作，我们可以创建 `MPI_COMM_WORLD` 的子集
![[PHCP-Fig8.4.png]]`MPI_Barrier` 是唯一的不在数据上进行操作的集体通信调用
在通信组内的所有进程都需要调用集体通信调用，否则程序会进入死锁

#### 8.3.1 Using a barrier to synchronize timers
最简单的集体通信调用即 `MPI_Barrier` ，它用于同步同一个通信组内的所有进程
该调用会影响程序的运行速度，因此仅在必要的时候调用

#### 8.3.2 Using the broadcast to handle small file input
广播调用从一个进程向其他所有进程发送数据
`MPI_Bcast` 用于将进程从一个输入文件中读到的数据发送给所有其他进程
所有的进程一一打开同一个文件的操作是内在串行的，因此只由一个进程打开文件并将数据发送给所有其他进程是更好的实践

#### 8.3.3 Using a reduction to get a single value from across all process
MPI归约调用接收一个多维数组，返回一个标量，常用的有：
- `MPI_MAX` (maximum value in an array)
- `MPI_MIN` (minimum value in an array)
- `MPI_SUM` (sum of an aray)
- `MPI_MINLOC` (index of minimum value)
- `MPI_MAXLOC` (index of maximum value)

#### 8.3.4 Using gather to put order in debug printouts
收集操作 `MPI_Gather` 将所有进程的数据收集并存入一个数组
我们可以利用该操作使得控制台的输出变得有序，而不是随机顺序

#### 8.3.5 Using scatter and gather to send data out to processes for work
散播操作 `MPI_Scatter` 和收集操作相反，该操作将数据从一个进程发送到通信组内的所有进程，广播操作中，发送的数据是一样的，而散播操作中，对不同的进程发送的数据可以是不同的
散播操作常用于将一个数组内的数据分段划分给不同进程以划分工作

### 8.4 Data parallel examples
#### 8.4.1 Stream triad to measure bandwidth on the node
#### 8.4.2 Ghost cell exchanges in a two-dimensional(2D) mesh
幽灵单元是我们用于连接相邻处理器的网格的机制，它是在MPI中启用分布式内存并行的最重要方法之一

为了避免在主循环中出现条件结构，程序员倾向于在网格四周在添加一圈额外的单元格，然后再循环计算开始之前将其设为适当的值，因此称环绕计算网格的单元格为环单元格
领域边界(domain boundary)环即用于施加特定的边界条件的单元格

幽灵单元实际上是在边界单元外围的单元，它们用于存储其他处理器上的部分单元的值

环更新(halo updates)可以即指领域边界单元更新，也指幽灵单元更新

使用幽灵单元，我们避免每次需要临近处理器上的一个单元的值时就进行一次通信

在每个进程上进行内存分配时，额外分配环的内存
```c
double **x = malloc2D(jsize + 2*nhalo, isize + 2*nhalo);
```

如果需要交换对角单元(corner cell)，则水平方向的交换和竖直方向的交换之间需要一次同步，否则可以并行完成

在C中，列方向的数据不是连续的，需要利用 `MPI_Pack` 将其打包一起发送，而行方向的数据使用 `MPI_Isend` 即可

`MPI_Pack` 会将数据打包在一个不限类型的缓存内，发送至另一端再解包(unpack)

如果发送的数据都是同一类型的，可以使用数组赋值，构造一个发送数组，将要发送的数据按序填入，用 `MPI_Isend` 发送后，接收端同样按需将接收到的值按序赋值给幽灵单元，此时类型就是 `MPI_DOUBLE` ，而不是 `MPI_PACKED`

#### 8.4.3 Ghost cell exchanges in a three-dimensional(3D) stencil calculation

### 8.5 Advanced MPI functionality to simplify code and enable optimizations
MPI的高级功能包括
- MPI自定义数据类型(custom data types)
	从MPI的基础数据类型创建自定义数据类型
- 拓扑支持(topology support)
	支持基础的笛卡尔正则网格和更泛用的图拓扑
#### 8.5.1 Using custom MPI data types for performance and code simplification
MPI支持从基础MPI数据类型创建自定义MPI数据类型
MPI的数据类型创建函数有
- `MPI_Type_contigous` 
	将一块连续的数据创建为一个类型
- `MPI_Type_vector`
	Create a type out of blocks of strided data
- `MPI_Type_create_subarray`
	从一个大数组内创建一个长方形的子数组
- `MPI_Type_indexed` / `MPI_Type_create_hindexed`
	Creates an irregular set of indices described by a set of block lengths and displacements. The hindexed version expresses the displacements in bytes instead of a data type for more generality.
- `MPI_Type_create_struct`
	从一个结构体创建类型，类型大小包括了编译器做的对齐
![[PHCP-Fig8.6.png]]
自定义的数据类型在使用前必须初始化，涉及的例程有
- `MPI_Type_Commit`
	初始化一个自定义数据类型(包括必要的内存分配等)
- `MPI_Type_Free`
	释放自定义数据类型初始化时涉及的内存和数据结构项(entries)

创建自定义数据类型可以简化幽灵单元交换，避免调用 `MPI_Pack` ，这会避免一次额外的数据拷贝，因为数据会从自己在进程内的位置直接拷贝到MPI的发送缓存中

#### 8.5.2 Cartesian topology support in MPI
#### 8.5.3 Performance tests of ghost cell exchange variants
### 8.6 Hybrid MPI plus OpenMP for extreme scalability
MPI和OpenMP一起使用时，常常用OpenMP的threads替换MPI的ranks，对于需要使用上千个进程的MPI执行，用OpenMP的线程替换MPI进程可以减小内存使用

#### 8.6.1 The benefits of hybrid MPI plus OpenMP
在MPI代码上加上一层OpenMP并行层的抽象的好处可以是
- 结点之间通信的幽灵单元个数更少
- 对MPI缓存的内存需求更少
- 对NIC的竞争更少
- 基于树的通信时间更少
- 改进的负载均衡
- 访问所有的硬件组件
这些好处大部分来源于线程之间共享内存的特性

#### 8.6.2 MPI plus OpenMP example
要在MPI上执行OpenMP，第一步要将 `MPI_Init` 替换成 `MPI_Init_thread`，其原型为
```c
MPI_Init_thread(&argc, &argv, int thread_model required, int *thread_model_provided);
```
MPI标准定义了四个线程模型，提供了四种级别的线程安全
- `MPI_THREAD_SINGLE` 
	仅执行单线程(标准MPI)
- `MPI_THREAD_FUNNELED`
	多线程，但只有主线程执行MPI调用
- `MPI_THREAD_SERIALIZED`
	多线程，但一次只有一个线程执行MPI调用
- `MPI_THREAD_MULTIPLE`
	多线程，且多线程执行MPI调用
许多应用程序仅在主循环进行通信，因此 `MPI_THREAD_FUNNELED` 即可正常工作，注意如果有可能的话尽量使用较低级别的线程模型，因为MPI在更高级别的线程模型上会在MPI基础结构周围如发送和接收队列中添加互斥锁(mutex)和关键块(critical blocks)

OpenMP支持设定仿射，仿射会对进程、线程到特定硬件的调度设置优先级，也称为pinning和binding
对应的编译选项有 `--bind-to core` / `--bind-to socket` 和 `--bind-to hwthread`
也可以通过设置环境变量
`export OMP_PLACES=cores` 
`export OMP_CPU_BIND=ture`

# GPUs: Built to accelerate

## 9 GPU architectures and concepts
GPU加速系统的结构
![[PHCP-Fig9.1.png]]
我们使用OpenCL标准规定的术语介绍GPU，该标准为大部分GPU厂商遵循

#### 9.1 The CPU-GPU system as an acclerated computational platform
GPU在设计时聚焦于处理并行大的数据块(三角形或多边形等)，这是图像处理应用的要求

CPU可以在一个时钟周期处理数十个并行的线程或进程，而GPU则是数千个

最初为GPU编程设计的语言，如OpenGL，一般聚焦于图像操作，

而GPU从只能进行图像工作，拓展至也可以进行非图像工作，就属于通用图形计算单元(general-purpose graphics processing unit/GPGPU)的范畴

第一个取得广泛采用的GPGPU语言是NVIDIA GPUs的编程语言CUDA(Compute Unified Device Architecture)，CUDA于2007年首次发布
而主要的GPGPU的开放标准则是OpenCL(Open Computing Language)，由Apple领导的一系列厂商在2009年提出

基于指令(directive)的GPGPU语言包括OpenACC和OpenMP，它们允许开发者聚焦于开发，而不是将算法表示为图像运算的形式

##### 9.1.1 Integrated GPUs: An underused option on commodity-based systems
AMD的集成GPUs称为加速处理单元(Accelerated Processing Units/APUs)
AMD的GPU设计源于它们对ATI在2006年的收购

在AMD的APU中，CPU和GPU共享处理器内存，共享内存的一个优点在于它消除了PCI总线上的数据传输需求，GPU和CPU之间的数据传输速度常常是性能瓶颈之一

##### 9.1.2 Dedicated GPUs: The workhorse option
对于专用GPU，CPU通过PCI总线发送数据和指令给GPU令其工作，工作结束并且需要将结果写入文件时，GPU必须要将数据送回给CPU

#### 9.2 The GPU and the thread engine
图像处理器可以视为一个理想的线程引擎，线程引擎包括
- 近乎无限数量的线程
- 切换和启动线程零时间开销
- 通过自动工作组切换，隐藏内存访问延迟

硬件术语：
![[PHCP-Table 9.1.png]]
严格地说，NVIDIA没有向量硬件，或者说SIMD，而是通过一个warp(即一系列线程)执行单指令多线程模型(single instruction, multi-threaded/SIMT)对SIMD进行模拟
其他GPU也可以执行SIMT操作，NVIDIA的warp等价于OpenCL和AMD的subgroup

通常，GPU也有复制的硬件块以简化其硬件设计的拓展(scale)
![[PHCP-Table9.2.png]]
这是一种制造方便，但经常出现在规范列表(specification list)中

一个拥有多核CPU和两个GPU的计算节点
![[PHCP-Fig9.2.png]]
OpenCL中，计算设备任何可以支持OpenCL的可执行计算的设备，可以指CPU和GPU甚至FPGA，每个GPU都是一个计算设备

一个GPU由
- GPU RAM(也称为全局内存/global memeory)
- 工作负载分配器(Workload distributor)
- 计算单元/Compute units(CUs)(流多处理器/SMs in CUDA)
构成

计算单元(CUs)有自己的内部架构，也被称为微架构，从CPU取来的指令和数据首先由工作负载分配器处理，分配器会协调计算单元的指令执行和数据的移进移出
GPU的可实现性能(achievable performance)决定于
- 全局内存带宽(global memory bandwidth)
- 计算单元带宽(compute unit bandwidth)
- 计算单元数量

##### 9.2.1 The compute unit is the streaming multiprocessor(or subslice)
OpenCL标准规定的术语为compute units(CUs)/计算单元，而NVIDIA称其为streaming multiprocessors(SMs)/流多处理器，Intel称其为subslices/子切片

##### 9.2.2 Processing elements are the individual processors
每个计算单元都包含多个图像处理器(graphics processors)，OpenCL称为处理元素(processing elements/PEs)，NVIDIA称为CUDA核心或计算核心(CUDA cores or Compute Cores)，Intel称为执行单元(exucution units/EUs)，图像社区称为着色处理器(shader processors)

计算单元和处理元素的示意图
![[PHCP-Fig9.3.png]]
处理元素并不等效于CPU处理器，它们的设计更简单，聚焦于进行图形处理操作，但事实上图形所需的操作几乎包括了程序员在常规处理器上使用的所有算术运算

##### 9.2.3 Multiple data operations by each processing element
每个处理元素也可以执行单指令多数据操作，当然如果硬件不支持，也可以用多个处理元素进行类似的实现

##### 9.2.5 Calculating the peak theoretical flops for some leading GPUs
一些GPU的规格表
![[PHCP-Table9.3.png]]

对于NVIDIA和AMD的面向HPC市场的GPUs，一般执行一次双精度运算的时间可以执行两次单精度运算，即flop的比值接近1:2(精度的比值也是1:2)
Intel的集成显卡则是1:4
该比值可以通过计算FP64/FP32得到

理论峰值flops通过处理器个数(number of processors)乘以时钟频率(clock rate)乘以每个时钟周期执行的浮点数运算次数(floating-point operations per cycle)得到，其中每个时钟周期执行的浮点数运算次数指融合乘加运算(fused-multiply add/FMA)，即一个时钟周期执行两个浮点数运算(two operations in one cycle)
$$\begin{align}
&\text{Peak Theoretical Flops(GFlops/s)}\\
=&\text{Clock rate MHZ} \times\text{Compute Units}\times\text{Processing Units}\times \text{Flops/cycle}
\end{align}$$

#### 9.3 Characteristics of GPU memory spaces
![[PHCP-Fig9.4.png]]

GPU的内存空间包括：
- 私有内存/寄存器内存(Private memory/register memory)
	单个处理元素可以立即访问的存储空间，且仅限自己访问
- 本地内存(Local memory)
	单个计算单元和其上的所有处理元素可以访问存储空间，本地内存的大小大约为64-96KB
- 常定内存(Constant memory)
	可以被一个GPU上所有计算单元访问的且共享的只读存储空间
- 全局内存(Global memeory)
	可以被一个GPU上所有计算单元访问的内存空间
	
目前GPU的全局内存使用的是GDDR5，相较于CPU使用的DDR4、DDR5，GDDR5有更高的带宽，最新的GPU正尝试使用HBM2(High-Bandwidth Memory 2)，以获得更高的带宽且功耗更小

##### 9.3.1 Calculating theoretical peak memory bandwidth
GPU内存的带宽计算通过将内存时钟频率乘以内存事务(memory transactions)的宽度(单位bits)得到，如果使用了DDR(Double Data Rate)技术，则还要乘以2(因为在一个周期的开始和结束都会执行事务)
一些类型的DDR内存甚至可以一个周期内做更多的事物

常用GPU内存类型的规格：
![[PHCP-Table9.4.png]]
理论内存带宽计算：
$$\begin{align}
\text{Theoretical Bandwidth} = \text{Memory Clock Rate(GHz)}\times\text{Memory bus(bits)}\times\\
\text{1 byte/8 bits}\times\text{transaction multiplier}
\end{align}$$
其中transaction multiplier表示了一个时钟周期可以执行的事务次数

一些规格表会给出内存事务率，单位为Gbps，内存事务率等于时钟频率乘以每个时钟周期执行的事务数，因此理论内存带宽计算也可以是：
$$\begin{align}
\text{Theoretical Bandwidth} = \text{Memory Transaction Rate(Gbps)}\times\text{Memory bus(bits)}\times\\
\text{1 byte/8 bits}
\end{align}$$

##### 9.3.2 Measuring the GPU stream benchmark
Babel STREAM Benchmark用不同的编程语言衡量一系列硬件的带宽，包括用CUDA衡量NVIDIA GPU，以及用OpenCL、HIP、OpenACC、Kokkos、Raja、SYCL、OpanMP衡量相对应的GPU(以上都是可以分别用于NVIDIA，AMD，Intel GPU的对应GPU语言)

##### 9.3.3 Roofline performance model for GPUs
##### 9.3.4 Using the mixbench performance tool to choose the best GPU for a workload
mixbench工具用于画出不同GPU设备的表现差异
结果图中，纵轴是计算率(compute rate)，单位为GFlops/sec，横轴是内存带宽，单位为GB/sec
![[PHCP-Fig9.6.png]]
注意图中其实隐藏了第三个变量即算数密度，每个点的算数密度都是不同的，具体可以通过计算率/内存带宽得到

右上角的点表示了该GPU设备的峰值的计算速率和内存带宽

可以将多个GPU设备的峰值点在一个图中进行比较
![[PHCP-Fig9.7.png]]
其中的一条直线表示直线上的点的算数密度都是1 flop/word(1 flop/load)，这是大多数典型应用的算数密度，另一条虚线的算数密度是65 flop/word，这是矩阵乘法的算数密度

可以将设备的峰值点向直线/虚线对齐查看在对应算术密度下，设备的内存带宽和计算率
如果峰值点高于线，则对齐后的点代表了该应用在该设备上可达到的峰值性能(算数密度过低，内存带宽限制表现)，如果峰值点低于线，则对齐后的点代表了应用的峰值性能由设备性能约束(算数密度过高，核心计算能力限制表现)

#### 9.4 The PCI bus: CPU to GPU data transfer overhead
数据传输的速率会是GPU表现的一个限制，通过限制GPU和CPU互相传输的数据量以得到速度提升是常见的方法

当前的PCI总线版本称为PCI Express(PCIe)，PCIe标准经历过多次修正，从1.0到6.0
对PCI总线带宽的估计有两种方法
- 简易计算的理论峰值表现模型
- 微基准测试应用(micro-benchmark application)

##### 9.4.1 Theoretical bandwidth of the PCI bus
在专用GPU平台上，所有的GPU和CPU之间的数据通讯都发生在PCI总线上，因此总线是能严重影响应用的总体性能的硬件

一条PCI总线有多个通道(lane)，数据传输是多通道并行的

总线的理论带宽
$$\begin{align}
\text{Theoretical Bandwidth(GB/s)}=\text{Lanes}\times\text{TransferRate(GT/s)}\times\\
\text{OverheadFactor(Gb/GT)}\times\text{byte/8bits}
\end{align}$$
理论带宽的单位是$GB/s$
将通道数乘以每个通道的最大传输速率(maximum transfer rate)，将bits转化为bytes即可得到理论带宽
而由于传输速率的单位往往是$GT/s$(GigaTransfers per second)，一般来说，每一次传输(transfer)的数据量都是1比特，但往往还要乘上间接开销系数(overhead factor)，因为在数据传输过程中，往往伴随一个用于保证数据完整性的编码方案，这会降低传输效率，因此也称该项为间接开销系数
对于1.0设备，编码方案的开销有20%，因此间接开销系数为100%-20%=80%
对于3.0设备，编码方案的开销只有1.54%，因此实际的数据传输速率和Transfer rate的差异已经很小了

**PCIe Lanes**
Linux系统上，`lspci` 可以展示所有附加在主板上的外设

**Determining the maximum transfer rate**
PCIe总线每个通道的最大传输速率可以直接通过它的代数决定，代数即代表了要求的硬件性能，
PCIe规格由PCI特殊兴趣组(PCI Special Interest Group/PCI SIG)制定
![[PHCP-Table9.5.png]]

**Overhead rates**
Gen1和Gen2规定为了传输8个字节的有效数据需要传输10个字节
Gen3开始，为了传输128字节的有效数据需要传输130个字节
间接开销因子由此计算得到

**Reference data for PCIe theoretical peak bandwidth**

##### 9.4.2 A benchmark application for PCI bandwidth

## 10 GPU programming model
### 10.1 GPU programming abstractions: A common framework
- 图像运算具有大量并行性
- 无法在任务之间协调的运算
#### 10.1.1 Massive parallelism
要输出高帧率、高质量的图像，有大量的像素点、三角形、多边形需要处理和展示
而在数据上的运算是基本相同的(identical)，因此GPUs使用了对多个数据项仅应用一条指令的技术(apply a single instruction to multiple data items)

最常见的GPU编程抽象
![[PHCP-Fig10.1.png]]
其中包含了四个部分
- 数据分解(data decomposition)
- 提供块大小的(chunk-sized)工作以处理，同时有共享的本地存储
- 使用一条指令对多个数据项进行运算
- 向量化
从计算域(computational domain)开始，我们通过这四个部分逐渐将工作拆分

#### 10.1.2 Inability to coordinate among tasks
图像工作负载(workload)并不需要大量的运算之间的协调(corrdination)，因此如果需要在GPU上执行需要协调的操作，例如归约(reductions)，需要复杂的方案

#### 10.1.3 Terminology for GPU parallelism
![[PHCP-Table10.1.png]]
OpenCL是GPU编程语言的开放标准，因此使用OpenCL作为基本术语参考
OpenCL可以在所有的GPU硬件上运行
HIP(Heterogeneous Computing Interface for Portability)是CUDA的可移植(portable)衍生，由AMD为它们的GPUs开发，它的术语和CUDA基本相同
AMD HC(Heterogeneous Compute)编译器和来自微软的C++ AMP语言(目前已经不在活跃开发阶段)的术语基本相同

#### 10.1.4 Data decomposition into independent units of work: An NDRange of grid
GPUs绘制三角形、多边形、生成高帧率图形的运算是完全独立的(completely independent from each other)
因此，顶层的对计算工作的数据分解就要生成相互独立和不同步(independent and asychronous)的工作

GPUs在工作组(work group)等待内存加载时(stalls for memory loads)，会通过切换到另一个准备进行计算的工作组(work group)以隐藏延迟

如图所示，当子组等待内存读时，执行就会切换到其他子组
![[PHCP-Fig10.2.png]]

目前较新的GPUs的子组调度器的规格如下图
![[PHCP-Table10.2.png]]

数据分解操作中，我们将一个2D的计算域分解成多个2D的小块，OpenCL将NDRange划分为多个Work Group，CUDA称将grid划分为多个block
![[PHCP-Fig10.3.png]]
GPUs对于这些分解出来的工作组有两个假设
- 工作组之间完全相互独立且异步(independent and asychronous)
- 这些工作组可以访问全局和常量内存(global and constant memory)

每个工作组都是一个独立的工作单元，意味着这些工作组可以按任意顺序进行处理，提供了并行计算的可能

图中展示了1D、2D、3D的计算域的可能的数据分解方案
![[PHCP-Table10.3.png]]

注意，改变最快的块维度(fastest changing tile dimension)$T_x$，在追求最优性能时，其大小应该是缓存行长度、内存总线宽度、或子组大小的整数倍

对于需要邻居块信息的算法，最优的块大小需要考虑最小的表面重叠区域，因为表面重叠区域的数据需要被邻接的块也装载
![[PHCP-Fig10.4.png]]

#### 10.1.5 Work groups provide a right-sized chunk of work
在工作组内，工作组将工作划分给计算单元上的线程执行
OpenCL中，`CL_DEVICE_MAX_WORK_GROUP_SIZE` 表示硬件工作组大小
工作组的最大大小一般在256到1024之间

但在实际计算时，工作组大小一般会小于最大大小，那么组内的每个线程/工作项(work item)可用的内存资源就更多

工作组也可以划分为多个子组(subgroup/warp)
![[PHCP-Fig10.5.png]]
NVIDIA的wrap大小是32个threads，AMD的wavefront大小是64个work items
工作组的大小一定是子组的大小的整数倍

GPUs的工作组的典型特征是
- 循环处理每个子组(cycle through processing each subgroup)
- 拥有组内的本地内存(共享内存)
- 可以和一个工作组或一个子组同步

如果一个工作组内的多个线程需要相同的数据，可以通过将数据加载入工作组的本地内存来提升性能

#### 10.1.6 Subgroups, warps, or wavefronts execute in lockstep
GPUs可以使用一个指令对一系列数据进行处理，而不是对每个数据提供一个独立的指令，这可以减少需要处理的指令数量
在CPU上，该技术称为SIMD，所有的GPUs通过一组线程(a group of threads)对该技术进行模拟，称为SIMT(single instruction multiple-thread)

SIMT并不要求底层硬件是向量硬件
目前的SIMT操作按lockstep执行，即子组中的任意线程要经过一个分支语句时，子组中的所有线程都要执行分支语句的所有路径
![[PHCP-Fig10.6.png]]
SIMT由于只是模拟SIMD，因此并不严格要求仅仅只有一条指令，其灵活性在可以多于一条指令

对于GPUs，小部分的条件分支代码不会对总体的性能有很大影响，但如果一些线程要比其余线程执行的时间多出上千个周期，就是一个很严重的问题
而如果将所有执行长分支的线程都放置在一个子组内，就不容易导致线程分歧(thread divergence)

#### 10.1.7 Work item: The basic unit of operation
OpenCL中，运算的基础单位称为工作项，该工作项可以被映射到一个线程或一个处理核，取决于硬件实现

CUDA中，称其为线程，因为工作项在NVIDIA GPUs中就被映射到一个线程

一个工作项下还可以有一层并行抽象，即向量硬件单元，类似CPU中一个线程执行向量运算
![[PHCP-Fig10.7.png]]

#### 10.1.8 SIMD or vector hardware
一些GPUs有向量硬件单元，可以在做SIMT操作的同时执行SIMD操作

OpenCL语言和AMD语言有对向量操作进行暴露
但由于CUDA硬件并没有向量单元，因此CUDA语言中没有相同层次的支持，不过包含向量操作的OpenCL代码也可以运行在CUDA硬件上，但对应的操作是由CUDA硬件模拟的

### 10.2 The code structure for the GPU programming model
称CPU为主机(host)，称GPU为设备(device)

GPU编程模型将循环体从应用于函数中的数组长度(array range)或索引集(index set)中分离，循环体(loop body)创造了GPU计算核(kernel)，而索引集和参数被主机用于创建计算核调用(kernel call)

如图展示了从一个标准循环体到一个GPU计算核体(body)的转变，该例使用了OpenCL语法
![[PHCP-Fig10.8.png]]
CUDA语法是类似的，将 `get_global_id()` 替换为 `gid = blockIdx.x * blockDim.x + threadIdx.x` 即可

循环体转化为一个计算核并且将其与主机上的索引集绑定遵循四个步骤
1. 提取出并行核(extract the parallel kernel)
2. 将局部数据块映射到全局数据(map from the local data tile to global data)
3. 在主机上执行数据分解(decomposition)，将数据划分为块(blocks)
4. 分配所需的内存(memory)

#### 10.2.1 "Me" programming: The concept of a parallel kernel
在GPU计算核中，所有事都和自己相关(everything is relative to yourself)，例如
`c[i] = a[i] + scalar*b[i];`
在该表达式中，没有关于循环范围的信息，每个数据项仅知道要对自己做什么(what needs to be done to itself and itself only)
因此该编程模型可以称为是“我”编程模型("Me" programming model)，即仅关心自己的编程模型
而正因为仅关心自己，对每个数据元素的运算是完全相互独立的

考虑一个更复杂的模板算子
`xnew[j][i] = (x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i])/5.0`
虽然涉及到了两个索引，并且访问了邻近的数据值，但 `i,j` 决定时，该行代码就完全定义(defined)了

循环体和索引集的分离在C++中可以通过functor或lambda表达式实现(lambda表达式是未命名的局部函数，可以被赋值给变量，然后在局部使用或传递给一个例程)

lambda表达式由四个主要部分构成：
- 函数体(Lambda body)
- 参数(Arguments)
- 捕获闭包(Capture closure)
- 调用(Invocation)

Lambda表达式构建了像SYCL、Kokkos、Raja这样的C++语言中更自然地为GPUs生成代码的方式

#### 10.2.2 Thread indices: Mapping the local tile to the global world
在进行数据分解(data decomposition)时，我们会为每个工作组提供一些关于它在局部域或全局域的位置(where it is in the local and global domains)的信息

OpenCL中，我们可以得到以下信息
- 维度(Dimension)
	从计算核的调用中得到维度的数量，1D、2D或3D
- 全局信息(Global information)
	当前的局部工作项(local work item)所对应的全局索引
	每个全局维度的大小，即全局计算域的每个维度的大小
- 局部/块信息(Local/tile information)
	每个维度上的局部信息，包含了该维度上的块大小(tile size)，和该维度上的块索引
- 组信息(Group information)
	每个维度上的组信息，包含了该维度上的组数量，和该维度上的组索引

CUDA中，信息是类似的，但全局索引必须由局部线程索引和块索引共同计算出
`gid = blockIdx.x * blockDim.x + threadIdx.x`

下图表示了OpenCL和CUDA中获取索引的形式，OpenCL使用函数调用获取，而CUDA中使用显式计算获取
![[PHCP-Fig10.9.png]]
其中的 `get_group_id(0)` 这样的函数和 `blockIdx.x` 这样的变量在我们为GPU执行数据分解时就会被自动定义好

#### 10.2.3 Index sets
每个工作组的索引的范围应该是相同的，这可能需要通过对全局计算域做padding，使得它的大小是局部工作组大小的整数倍

利用C的算数计算的性质，计算padding后全局计算域的长度应该是多少
`global_work_size_x = ((global_size_x + local_work_size_x -1)/ local_work_size_x) * local_work_size_x`
如果 `global_size_x` 可以整除 `local_work_size_x` ，则该表达式的结果就是 `global_size_x`
如果 `global_size_x` 不能整除 `local_work_size_x` ，则该表达式的结果就是 `(global_size_x % local_work_size_x + 1) * local_work_size_x` ，等价于
`ceil(global_size_x/local_work_size_x) * locak_work_size_x` ，
即 `global_size_x` padding到可以整除 `local_work_size_x` 后的值

同时，为了避免越界读取，在计算核中要有边界检测，如
`if (gid > global_size_x) return;`
注意，避免越界读写在GPU计算核中是十分重要的，因为它们会导致随机的核瘫痪(kernel crashes)，同时不会出现错误信息(error message or output)

#### 10.2.4 How to address memeory resources in your GPU programming model
NVIDIA V100和AMD Radeon Instinct MI50都支持32G的RAM，一般的HPC CPU结点会带有128G的内存，因此一个带有4-6个GPUs的GPU计算节点会和CPU结点有相同大小的内存
在这种情况下，我们可以在GPU上使用和CPU上相同的内存分配策略，减少将数据来回传输的开销

一般情况下，GPU上的内存分配也要在CPU上进行，GPU和CPU的内存分配往往同时进行，然后数据在二者之间进行传输
但如果可能，应只对GPU做内存分配，以避免和CPU之间昂贵的数据传输，并且释放CPU上的内存空间

使用动态内存分配的算法在GPU上执行时需要改成静态内存分配

最先进的GPU可以将不规则的或打乱的多个内存访问(irregular or shuffled memory access)合并(coalesing)为单个、连续的缓存行大小的装载(coherent cache-line loads)

定义：合并的内存装载(coalesed memory loads)是将多个线程的分别的内存装载结合为单个缓存行大小的装载

GPU中，内存合并(memory coalescing)是在硬件层面由内存控制器(memory controller)完成的，执行内存装载合并得到的性能提升是巨大的
并且由于这是硬件上执行的，许多早期的GPU编程技巧就不再必要了

对于要重复使用的数据，将其装载到本地(共享的)内存可以提供额外的加速
在过去，这项优化对性能表现很重要，但GPUs上更好的cache性能已经使得它不再重要

取决于我们是否可以预测要使用多少的本地内存，有不同的本地内存使用策略，常见的两种内存使用/装载方法是网格方法(regular grid approach)和针对非结构化和适应性网格精细化(unstructured and adaptive mesh refinement)的不规则网格方法(irregular grid)
简单来说，就是要执行计算的区域是规则的还是不规则的，如果规则，则要使用多少内存/要装载多少个单元是可预测的，如果不规则，那就是不可预测的

一个典型的GPU应用一次一般装载128或256个单元(cells)，然后再装载周边的所需邻近单元(neighbor cells)
![[PHCP-Fig10.10.png]]
当邻近线程所使用的内存有重叠，且邻近线程所用的内存大小一致，即规则的/可预测的情况，
例如线程$i$需要第$i-1$和第$i+1$个值，那么邻近线程之间就存在共享的内存/值，在这种情况下，最好的方法就是做协作的内存装载(cooperative memory loads)，将需要的值从全局内存拷贝到本地(共享)内存，这可以显著提升执行效率

而不规则的计算网格会使得邻近线程所需要的内存值无法预测，这种情况下，一种方法就是将整个计算网格的一部分拷贝到本地内存中，然后每个线程按需求将所额外需要的邻近数据(neighbor data)装载入自己的寄存器中
### 10.3 Optimizing GPU resource usage
GPU编程的关键在于管理对于执行核(executing kernel)可用的(available)的有限的资源
意图使用超出可用量的资源会导致严重的性能下降

当前的一些GPUs的资源限制如下
![[PHCP-Table10.4.png]]

GPU编程者可以控制的最重要的参数就是工作组大小(work group size)

对于计算核(computatoinal kernel)来说，它们相较于图形核(graphics kernel)的复杂性主要在于对计算资源(compute resources)的大量要求(demand)，这通常称为内存压力(memory pressure)或寄存器压力(register pressure)
内存压力指计算核对内存资源的要求，寄存器压力则指计算核对寄存器资源的要求

减小工作组的大小，平均每个工作组就有更多可以利用的资源(如线程)，同时可以保证有更多的工作组用于上下文切换(context switching)
#### 10.3.1 How many registers does my kernel use?
使用 `nvcc` 编译命令行时，添加 `-Xptxas="-v"` 选项，或者在NVIDIA GPUs使用OpenCL编译命令行时，添加 `-cl-nv-verbose` 就可以查看计算核所使用的寄存器数量

#### 10.3.2 Occupancy: Making more work available for work group scheduling
当一个工作组因为内存延迟(memory latency)而停滞(stall)时，我们需要执行上下文切换，切换到其他可执行的工作组以隐藏延迟

GPUs有一个衡量标准称为占用率(occupancy)，占用率衡量了计算单元在计算中忙碌的程度，这个衡量是较为复杂的，因为它依赖于许多因素，例如所需求的内存和所使用的寄存器，精确的定义是
$$\begin{align}
&\text{Occupancy}=\\ 
&\text{Number of Active Threads/Maximum Number of Threads Per Compute Unit}
\end{align}$$
因为每个子组的线程数量是固定的，一个等价的定义基于subgroups/wavefronts/warps
$$\begin{align}
&\text{Occupancy}=\\ 
&\text{Number of Active Subgroups/Maximum Number of Subgroups Per Compute Unit}
\end{align}$$
活跃子组或线程的数量取决于先耗尽的子组或线程资源，常常一个工作组需要的寄存器数量或本地内存的大小会防止(prevent)另一个工作组开始工作

CUDA Occupancy Calculator用于计算占用率，NVIDIA编程指导也大量聚焦于最大化占用率，这是一个重要的指标，但是只要我们有足够的工作组以隐藏等待和延迟就可以达到高的占用率
### 10.4 Reduction pattern requires synchronization across work groups
目前为止，我们接触的计算循环(computational loops)都可以通过图10.8转换为计算核，即将 `for` 循环去除，用循环的计算体(computational body)创造GPU核
这个转换是十分简单且快速的，但也存在要代码难以转换为GPU计算核的情况

例如要做归约运算时，困难源于我们不能在工作组之间做协同的工作或比较(cooperative work or comparisions)，唯一的方式只能退出计算核，下图阐述了处理该情况的通用策略
![[PHCP-FIg10.11.png]]
实际中，数组的长度往往会达到数百万，
在第一步，我们计算每个工作组的和，将其存储在一个数组中(scratch array)，由此我们将要处理的数组长度减少为工作组的数量，或者说要处理的数组长度除去了工作组的长度(一般是512或1024)，此时，因为工作组之间无法交流(communicate)，我们退出第一个计算核，启动一个新的仅有一个工作组的计算核，然后遍历数组，分配工作给多个工作项(work item)求局部和，工作项之间可以通讯，因此最后在工作组内就能归约成一个全局和

在GPU上，上述工作需要多行代码，且需要两个计算核，而CPU上只需一行代码，虽然GPU的速度快于CPU，但需要更多的编程工作
同时，GPUs内的同步和比较也是较难执行的工作
### 10.5 Asychronous computing through queues(streams)
通过重叠数据传输和计算(overlapping data transfer and computation)，我们可以完全利用GPU，在GPU执行计算的同时，可以发生两次数据传输

GPU上工作的基本性质就是异步性(asychronous)，
工作在GPU上排队，通常仅在有对结果的请求或有同步请求时(a result or synchronization is requested)才被执行
![[PHCP-Fig10.12.png]]

我们可以使用多个独立且异步的队列对工作进行调度
![[PHCP-Fig10.13.png]]
使用多个队列以暴露数据传输和计算的重叠性(overlap)，创造并行的可能性
大多数的GPU语言支持某种形式的异步工作队列，OpenCL中，命令在队列中排队(commands are queued)，CUDA中，运算则在流中(operations are palced in streams)

如果我们有可以同时做以下三件事的GPU
- 将数据从主机拷贝到设备
- 核计算(kernel computataions)
- 将数据从设备拷贝回主机
则按照图10.13，将工作置于三个独立队列中，它们可以通过以下方式进行重叠
![[PHCP-Fig10.14.png]]
### 10. 6 Developing a plan to parallelize an application for GPUs
#### 10.6.1 Case1: 3D atmospheric simulation
假设有大气模拟程序，数据范围在$1024\times 1024 \times 1024$到$8192\times 8192 \times 8192$，x是垂直方向，y是水平方向，z是深度
我们可以考虑
- Option 1: Distribute data in a 1D fashion arcoss the z-dimension(depth)
	我们需要上万个工作组以在GPU上进行有效并行，参照表格9.3，我们有60-80个32位浮点计算单元，以进行大约2000路的并行计算/算数路径(arithmetic pathways)(每个计算单元运行一个工作组，一个工作组有32个工作线程，$60\times 32 = 1920, 80\times 32 = 2560$)，如果沿着z轴分布数据，我们得到1024到8192个工作组，对GPU并行来说，这个数量很少
	考虑每个工作组需要的资源，最小的情况是一个$1024\times 1024$的平面，加上任意需要的在幽灵单元中的邻近数据，假设两个方向都各需要一个幽灵单元，则我们需要$1024\times 1024\times 3 \times 8$个字节，即24MB的本地数据(local data)，参考表格10.4，GPUs的每个计算单元仅能存储64-96KB的本地数据，因此我们无法将数据预加载到本地内存中以加快计算
- Option 2: Distribute data in a 2D vertical columns arcoss y- and z-dimensions
	按两个维度分配数据，我们可以得到超过100万个潜在的工作组，因此我们有足够的独立的工作组，对每个工作组，我们会有1024到8192个单元，考虑4个方向都有邻近的幽灵单元，一个工作组需要最少$1024\times 5 \times 8$个字节，即40KB的本地数据，如果考虑更大的计算任务，其中每个单元包括多于一个的变量，GPUs计算单元的本地内存将不足
- Option 3: Distribute data in 3D cubes across x-, y-, and z-dimensions
	参考表格10.3，对于工作组，分配一个$4\times 4 \times 8$的单元块(cell tile)，考虑每个维度两边的近邻单元，一个工作组最少需要$6\times 6 \times 10 \times 8$字节即2.8KB的本地数据，在这种情况下，我们可以考虑在每个单元中放置多个变量
	
对于$1024\times 1024\times 1024$个单元，总的内存需求是$1024\times 1024\times 1024 \times 8$个字节即8GB，目前GPUs最多有32GB的RAM，因此这个问题可以在一个GPU上进行运算，但更大规模的问题可能需要512个GPUs，因此也需要考虑利用MPI进行分布式内存并行(distritubed memory parallelism)
#### 10.6.2 Case2: Unstructured mesh applilcation
本例中，我们的应用是一个3D的非结构化的网格，使用四面体(tetrahedral)或多边形(polygon)单元，数量范围是1到1千万
数据是一个包含了多边形和其相应数据例如空间坐标xyz的1D列表，此时，我们只能考虑1D数据分配

因为数据是非结构化的且存于1D数组中，我们直接按块大小(tile size)128分配数据(即每个工作组128分配到数据)，我们会得到8000到80000个工作组，数量足以让GPU互相切换以隐藏延迟，内存需求是$128\times 8$字节即1KB，这也允许我们在每个单元放置多个数据值
我们同样需要空间用于存储整数映射和邻近数组(integer mapping and neighbor arrays)以提供单元之间的关联性(connectivity)，邻近数据会最终被加载入每个线程的寄存器中，所以我们不必太担心本地内存存储过多线程的邻近数据导致空间不够
最大规模的网格有1千万个单元，因此需要80MB(每个单元8字节)，当然可能还需要空间存储表面，邻近，映射数组(face, neighbor, mapping arrays)，对于单个GPU来说，空间是完全足够的

为了更好的性能，我们可以通过使用数据分割库(data-partioning library)或使用可以保持在数组中互相邻近的数据在空间中也互相邻近的空间填充曲线(space-filling curve)以为非结构化的数据提供一定程度的局部性(locality)
## 11 Directive-based GPU programming
OpenMP发布于1997年，是优秀的基于指令的语言(directive-based)，发布之初，OpenMP主要聚焦于CPU
2011年，编译器厂商Cray，PGI，CAPS和GPU厂商NVIDIA联合发布了OpenACC标准，聚焦于GPU编程
OpenACC和OpenMP类似，也采用杂注，杂注指导编译器生成GPU代码
几年后，OpenMP Architecture Review Board(ARB)为OpenMP标准加入了支持GPU代码生成的杂注(从版本4.0开始)
### 11.1 Process to apply directives and pragmas for a GPU implementation
杂注即C/C++中给予编译器特殊指令的预处理语句(preprocess statements)，形式为
```c
#pragma acc <directive> [clause]
#pragma omp <directive> [clause]
```

在应用中使用OpenMP/OpenACC总体的步骤完全一样
![[PHCP-Fig11.1.png]]
总结为以下三个步骤
1. 将计算密集的工作移到GPU，当然这会导致GPU和CPU之间频繁的数据传输，因此会降低代码执行速度，但工作(work)需要先被移动至GPU
2. 减少GPU和CPU之间的数据移动，如果数据仅在GPU中使用，则将内存分配移动至GPU
3. 调节工作组大小、工作组数量以及其他计算核参数，以提升计算核性能
### 11.2 OpenACC: The easiest way to run on your GPU
首先需要做的是有一个可以工作的OpenACC编译器工具链，以下列出一些OpenACC编译器
- PGI
	这是商用编译器，但存在社区版本
- GCC
	版本7和8支持OpenACC 2.0a规范，版本9实现了大多数OpenACC 2.5规范
- Cray
	商用编译器，仅可以在Cray系统上使用，目前新版本已不再支持OpenACC

我们使用PGI(版本19.7)和CUDA(版本10.1)，PGI是所有编译器中最为成熟的

使用PGI编译器时，命令 `pgaccelinfo` 用于查看系统和环境信息，了解GPU设备的信息和特点
#### 11.2.1 Compiling OpenACC code
PGI编译器中，标志 `-acc -Mcuda` 用于启用OpenACC编译，表示 `Minfo=accel` 用于告诉编译器对加速器指令(accelerator directives)提供反馈(feedback)，标志
`-alias=ansi` 用于告诉编译器代码中不存在指针别名的情况，以促进并行计算核的生成，当然在源码中，为参数添加 `restrict` 属性也可以表明它指向的是没有重叠的内存区域
GCC编译器中，标志 `-fopenacc` 用于启用OpenACC编译，标志 `-fopt-info-optimized-omp` ，用于告诉编译器对加速器指令生成提供反馈

OpenACC编译器必须定义 `_OPENACC` 宏，因为OpenACC尚处于由许多编译器进行实现的阶段，我们可以根据该宏知道我们的编译器支持哪个版本的OpenACC，通过该宏实现条件语句(格式为 `-OPENACC == yyyymm` )也可以实现对新特性的条件编译

对应的版本日期如下
Version 1.0: 201111
Version 2.0: 201306
Version 2.5: 201510
Version 2.6: 201711
Version 2.7: 201811
Version 3.0: 201911
#### 11.2.2 Parallel compute regions in OpenACC for acclerating computations
指定要对一个代码块进行加速有两种选项，第一种是使用 `kernels` 杂注，让编译器对代码块进行自动并行化，代码块可以是包含了多个循环的大段代码
另一种是使用 `parallel loop` 杂注

**Using the kernels pragma to get auto-parallelization from the compiler**
`kernels` 杂注可以让编译器自动并行化代码块，常用于首先得到编译器对一部分代码的反馈信息
根据OpenACC 2.6标准，`kernels` 杂注的规范如下
```c
#pragma acc kernels [ data clause | kernel optimization | async clause | conditional ]
```
其中
```
data clauses - [ copy | copyin | copyout | create | no_create | present | deviceptr | attach | default(none|present) ] 
kernel optimization - [ num_gangs | num_workers | vector_length | device_type | self ] 
async clauses - [ async | wait ] 
conditional - [ if ]
```
对于我们希望并行化的代码块，我们在它的上面添加 `#pragma acc kernels` ，该杂注对紧随它的代码块生效

对于每一个 `for` 循环，OpenACC都会将其视为有一个 `#pragma acc loop auto` 在它之前，因此编译器会自行决定是否该循环可以并行化
编译器无法确定是否存在指针别名时，可能会判断存在循环依赖，通过对指针变量加上 `restrict` 属性可以解决这一问题
我们也可以通过为杂注添加额外的子句以告诉编译器该循环是可以直接并行的，默认的 `loop` 指令即 `loop auto` ，而根据OpenACC 2.6标准，`loop` 杂注的规范是
```c
#pragma acc loop [ auto | independent | seq | collapse | gang | worker | vector | tile | device_type | private | reduction ]
```
其中 `auto` 子句即让编译器自己进行分析，`seq` 子句则直接指定编译器生成顺序执行的代码，`independent` 子句则直接断言循环是可以并行化的，指定编译器生成并行化的代码
因此把杂注改为 `#pragma acc kernels loop independent` 也可以解决这一问题

**Try the parallel loop pragma for more control over parallelization**
使用 `parallel loop` 杂注则在格式上与其他并行语言例如OpenMP更一致，该杂注生成的代码也在不同编译器之间更具有一致性和可移植性(consistent and portable)，因为对于 `kernel` 杂注所要求的分析工作，每个编译器都有自己的执行方式，且分析工作并不一定充足

`parallel loop` 杂注实际上是两个指令，第一个是 `parallel` 指令，用于开辟一个并行区域，第二个是 `loop` 指令，用于将工作分配给并行的工作元素(work element)

`parallel` 和 `kernel` 指令接收几乎一样的子句，相较于 `kernel` 的子句，多出了 `reduction` , `private` 和 `firstprivate`
```c
#pragma acc parallel [clause]
data clauses - [ reduction | private | firstprivate | copy | copyin | copyout | create | no_create | present | deviceptr | attach | default(none|present) ] 
kernel optimization - [ num_gangs | num_workers | vector_length | device_type | self ] 
async clauses - [ async | wait ] 
conditional - [ if ]
```
而 `loop` 指令接收的子句已经在上一节提到过，值得一提的是，OpenACC对于处在并行区域，即 `parallel` 指令范围内的 `loop` 构造(construct)的默认子句是 `independent` 而不是 `auto` 
二者结合的构造 `parallel loop` 接收的子句和单个存在时接收的子句没有不同

使用 `parallel loop` ，我们就无需显式为变量添加 `restrict` 属性也可以正常并行化循环，因为此时 `loop` 指令的默认子句是 `independent` 

要执行归约操作时，可以对 `parallel loop` 杂注添加 `reduction` 子句，例如
```c
#include "mass_sum.h"
#define REAL_CELL 1
double mass_sum(int ncells, int* restrict celltype, double* restrict H, double* restrict dx, double* restrict dy){
	double summer = 0.0;
	#pragma acc parallel loop reduction(+:summer)
	for (int ic=0; ic<ncells ; ic++) {
		if (celltype[ic] == REAL_CELL) {
			summer += H[ic]*dx[ic]*dy[ic];
		}
	}
	return(summer);
}
```
`reduction` 子句的格式和OpenMP中的格式完全类似，`reduction` 子句还可以接收其他运算符，包括 `*` , `max` , `min` , `&` , `|` , `&&` , `||` 

如果在 `parallel` 区域中，希望不退出该区域，并顺序执行一个循环，我们可以使用 `serial` 指令：`#pragma acc serial`
#### 11.2.3 Using directives to reduce data movement between the CPU and GPU
OpenACC v2.6规范中，`data` 构造的规范是
```c
#pragma acc data [ copy | copyin | copyout | create | no_create | present | deviceptr | attach | default(none|present) ]
```
`data` 构造接受参数，参数用于表明要被拷贝或被操作的数据，如果是一个数组，需要注明范围，例如
`#pragma acc data copy(x[0:nsize])`
C/C++中，第一个数表示起始索引，第二个数表示长度(Fortran中为结束索引)

`data` 区域有两种变体，第一种是源于OpenACC v1.0标准的结构化数据区域(structured data region)，第二种是由OpenACC v2.0标准引入的动态数据区域(dynamic data region)
**Structured data region for simple blocks of code**
结构化数据区域作用于一个代码块，可以是由一个循环构成的自然代码块，也可以手动用大括号框起来具体的区间
`#pragma acc data create(a[0:nsize], b[0:nsize], c[0:nsize])` 用于指定在数据区域起始的时候创建数据，它们默认会在数据区域结束后被摧毁
`#pragma acc parallel loop present(a[0:nsize], b[0:nsize], c[0:nsize])`
用于表明数据已经被创建且存在，因此不需要额外拷贝
**Dynamic data region for a more flexible data scoping**
结构化的数据区域只能处理内存预分配的情况，即需要的内存需要在数据区域的开始就分配好
OpenACC v2.0加入了动态的(也称非结构化的)数据区域，为更复杂的数据管理情况服务，例如C++中的构造和析构函数
动态的数据区域不使用大括号表明数据区域范围，而使用 `enter` 和 `eixt` 子句
`#pragma acc enter data`
`#pragma acc exit data`

`#pragma acc enter data` 指令应该正好放在内存分配语句之后(after the allocation)，`#pragma acc exit data` 指令应该正好放在内存释放语句之前(before the deallocation)

在较大的动态内存区域中，我们需要额外的指令用于更新数据
`#pragma acc update [self(x) | device(x)]`
其中 `device` 参数表明位于设备上的数据需要更新，而 `self` 参数表明本地数据需要更新

建议尽量使用动态数据区域而不是结构化的数据区域

直接使用 `malloc` 会导致主机和设备上都被分配内存，如果我们只需要在设备上分配内存，可以使用 `acc_malloc` 函数，然后在计算区域内使用 `deviceptr` 子句，`acc_malloc` 函数对应的内存释放函数是 `acc_free`
示例见listing 11.7

listing 11.7只对1D数组有效，对于2D数组，由于 `deviceptr` 子句不接受描述符参数(descriptor argument)，因此计算核也需要修改，在展平成1D数组的数据上做2D数组的索引
#### 11.2.4 Optimizing the GPU kernels
一般来说，让更多的计算核运行于GPU上，且减少数据移动，会比直接优化计算核本身的影响更大
OpenACC编译器已经十分善于生成计算核，在此基础上进一步优化的潜在提升比较小

OpenACC定义了三个并行的层次(levels of parallelism)，包括
- Gang
	一个独立的共享资源的工作块(work block)，gang可以在组内(within group)同步，但不能跨组(arcoss groups)同步
	对于GPUs，gangs可以映射到CUDA线程块(thread blocks)或OpenCL工作组(work groups)
- Workers
	CUDA中的一个warp或OpenCL中一个工作组内的工作项(work items)
- Vector
	一个CPU上的SIMD向量，一个GPU上的SIMT工作组或warp，具有连续的内存引用(with contiguous memory reference)
![[PHCP-Fig11.3.png]]
可以设定特定的 `loop` 指令的并行层次，例如
`#pragma acc parallel loop vector`
`#pragma acc parallel loop gang`
`#pragma acc parallel loop gang vector`
外部循环必须是 `gang` 循环，内部循环应该是 `vector` 循环，二者之间可以有 `workers` 循环

`seq` 循环可以在任意并行层次

对于多数当前的GPUs，向量长度必须设置为32的倍数，因此向量长度也就是warp大小的整数倍，向量长度不能超过每块(block)的最大线程数量(一般是1024)
PGI编译器默认将向量长度设为128，对于一个loop，我们可以用 `vector_length(x)` 指令改变向量长度
改变向量长度的场景包括：内部循环的连续数据少于128，因此部分的向量长度无法利用，或需要将几个内部循环折叠，以利用更长的向量长度

可以使用 `num_workers` 子句修改 `worker` 的设置，当我们需要额外的一层并行或我们减小了向量长度时，增大 `worker` 数量是有益的，如果我们的代码需要在工作组内同步，则需要使用 `worker` 层次，但OpenACC并未提供同步命令，`worker` 层次也共享缓存和局部内存

最后的并行层次就是gangs，这一层并行是异步的，数量多的gangs利于GPU隐藏延迟，保持高的占用率
通常来说，编译器会将gangs的默认数量设为一个较大的值，因此用户无需再覆盖这个值，当然用户也可以用 `num_gangs` 子句对其进行设置

许多设定可能只对特定的硬件适合，我们在设定子句前添加 `device_type(type)` 子句，可以让该设定只对针对的硬件生效，`device type` 设定保持生效直到遇到下一个 `device type` 子句，例如
```c
#pragma acc parallel loop gang \
	device_type(acc_device_nvidia) vector_length(256) \
	device_type(acc_device_radeon) vector_length(64)
for (int j = 0; j < jmax; j++){
	#pragma acc loop vector
	for (int i = 0; i < imax; i++){
		<work>
	}
}
```
PGI v19.7的 `openacc.h` 头文件中包含了有效设备类型的列表，例如列表中其实没有 `acc_device_radeon` ，因此PGI编译器实际上不支持AMD Radeon设备，因此使用PGI编译器时，为了避免报错，还需要C预处理命令 `ifdef` 确保要使用的宏已经定义
以下是 `openacc.h` 的部分内容
```c
typedef enum{
	acc_device_none = 0,
	acc_device_default = 1,
	acc_device_host = 2,
	acc_device_not_host = 3,
	acc_device_nvidia = 4,
	acc_device_pgi_opencl = 7,
	acc_device_nvidia_opencl = 8,
	acc_device_opencl = 9,
	acc_device_current = 10
	} acc_device_t;
```

如果使用 `kernels` 指令，则格式略有不同，每个 `loop` 的注明并行类型的子句可以直接接受 `int` 类型的参数
```c
#pragma acc kernels loop gang
for(int j = 0; j < jmax; j++){
	#pragma acc loop vector(64)
	for (int i = 0; i < imax; i++){
		<work>
	}
}
```

循环可以用 `collapse(n)` 子句结合起来，该子句在有两个小的内部循环连续地遍历数据时十分有用，将这些循环结合可以让我们利用更长的向量长度
要 `collapse` 的循环必须是紧密嵌套的(tightly nested)(两个或多个循环在 `for` , `do` 语句之间没有额外的语句，或在循环的结尾之间没有额外的语句即紧密嵌套的)

结合两个循环以使用更长的向量的例子
```c
#pragma acc parallel loop collapse(2) vector(32)
for (int j=0; j<8; j++){ 
	for (int i=0; i<4; i++){ 
		<work> 
	} 
}
```

OpenACC v2.0加入了 `tile` 子句，我们可以借其指明tile大小，或传入 `*` ，让编译器自行选择
```c
#pragma acc parallel loop tile(*, *)
for (int j=0; j<jmax; j++){ 
	for (int i=0; i<imax; i++){ 
		<work> 
	} 
}
```

现在，考虑对模板计算的例子(stencil computation)进行优化，我们将执行计算的循环移动到GPU上，然后减少数据移动，另外，在CPU上，我们在循环的最后交换了指针，在GPU上，我们需要将新数据拷贝回原来的数组
示例见Listing11.8

在示例中加入 `collapse` 可以减少两个循环的开销
示例见Listing11.9

也可以尝试使用 `tile` 字句
示例见Listing11.10

但加入 `collapse` ，改变 `tile` 大小，改变向量长度等优化对实际执行时间的减少的相对量是很少的
#### 11.2.5 Summary of performance results for the stream triad
![[PHCP-Table11.2.png]]
对于stream triad基准测试，不同的计算核优化有不同的运行时间表现，可以看到，仅仅将计算核移动到GPU，如果GPU没有并行化循环，程序在GPU上顺序运行，运行速度非常慢，如果成功并行化循环，运行速度减少了，但数据移动的时间仍然很长；通过加入数据区域，减少了数据移动，可以看到运行时间显著减少
#### 11.2.6 Advanced OpenACC techniques
OpenACC还有许多其他特性方面处理复杂的代码

**Handling functions with the OpenACC routine directive**
OpenACC v1.0要求计算核内要使用的函数时是内联的，v2.0加入了 `routine` 指令，使得调用例程更加简单，该指令有两个版本
```c
#pragma acc routine [gang | worker | vector | seq | bind | no_host | device_type]
#pragma acc routine(name) [gang | worker | vector | seq | bind | no_host | device_type]
```
C/C++中， `routine` 指令应该在紧靠在函数原型或定义之前，而 `routine(name)` 则可以在函数被定义或使用的位置之前的任意位置

**Avoiding race conditions with OpenACC atomics**
许多多线程程序会有一个由多个线程更新的共享变量，这样的编程构造往往即是性能瓶颈也是潜在的竞争情况
OpenACC v2提供了原子操作以允许一次只有一个线程可以访问一个存储位置
```c
#pragma acc atomic [read | write | update | capture]
```
默认的子句是 `update` ，一个简答的用例是
```c
#pragma acc atomic
cnt++;
```

**Asynchronous oprations in OpenACC**
将OpenACC操作重叠有利于提升性能，重叠的操作(overlapping operations)即异步的，OpenACC使用 `async` 子句和 `wait` 指令支持异步操作
`async` 子句可以带有可选整数参数，添加到工作或数据指令中(work or data directive)，例如
`#pragma acc parallel loop async([<integer>])`

`wait` 可以是一条指令，也可以是添加到工作或数据指令中的子句
`async` 和 `wait` 的使用示例
```c
for(int n = 0; n < ntimes; ){
	#pragma acc parallel loop async
		<x face pass>
	#pragma acc parallel loop async
		<y face pass>
	#pragma acc wait
	#pragma acc parallel loop
		<Update cell from face fluxes>
}
```

**Unified memory to avoid managing data movement**
尽管统一的内存(unified memory)目前不是OpenACC标准的一部分，但有让系统管理内存移动的实验性开发(experimental developments)
对统一内存的实验性实现在CUDA和PGI OpenACC编译器中可用

在PGI编译器和最近的NVIDIA GPUs中使用 `-ta=tesla:managed` 标志，就可以尝试它们对统一内存的实验性实现，这会简化我们的代码，但影响目前是未知的

**Interoperability with CUDA libraries or kernels**
OpenACC提供了若干个指令和函数用于和CUDA库互操作(interoperate)
在调用库时，有必要告诉编译器要使用设备指针(device pointer)而不是主机数据(host data)，为此要使用 `host_data` 指令
```c
#pragma acc host_data use_device(x, y)
cublasDaxpy(n, 2.0, x, 1, y, 1);
```
当我们使用 `acc_malloc` 和 `cudaMalloc` 在设备上分配内存时，返回的指针就是已经在设备上的设备指针，我们使用 `deviceptr` 子句告诉编译器指针已经在设备上，如同Listing11.7中所示

GPU编程最常见的错误就是混乱了设备指针和主机指针
![[PHCP-Fig11.4.png]]
如图展示了三种可能的操作
在第一种情况中，我们使用 `malloc` 得到一个主机指针，使用 `present()` 子句将其转化为一个设备指针
在第二种情况中，我们使用 `acc_malloc` 或 `cudaMalloc` 直接得到设备指针，使用 `deviceptr()` 子句直接将该设备指针发送给GPU，不需要进行任何修改
最后一种情况中，我们使用 `host_data use_device()` 指令将一个设备指针找回主机，以便我们之后使用任何设备函数时，直接将 `x_dev` 作为参数传入

实践中，最好在指针名后添加 `_d` , `_h` 后缀以声明它们有效的场景

**Managing multiple devices in OpenACC**
OpenACC允许我们通过下列函数管理多个设备
- `int acc_get_num_devices(acc_device_t)`
- `acc_set_device_type() / acc_get_device_type()`
- `acc_set_device_num() / acc_get_device_num()`
### 11.3 OpenMP: The heavylight champ enters the world of accelerators
目前来说，OpenMP对加速设备的指令实现相较于OpenACC更不成熟，但在迅速发展，当前可用的为GPUs的实现(implementations)有
- Cray首先在2015年针对NVIDIA GPUs进行了OpenMP实现，目前Cray支持OpenMP v4.5
- IBM在Power 9处理器和NVIDIA GPUs上完全支持OpenMP v4.5
- Clang v7.0+在NVIDIA GPUs上支持OpenMP v4.5
- GCC v6+在AMD GPUs上支持OpenMP，v7+在NVIDIA GPUs上也提供支持
其中最成熟的两个实现是Cray和IBM，但只能在它们对应的系统上使用
GCC和Clang则还在发展中，可以使用的版本较少
#### 11.3.1 Compiling OpenMP code
CMake有OpenMP模块(module)，但该模块对OpenMP加速器指令没有显式支持，我们使用OpenMPAccel模块，以调用常规的OpenMP模块，且可以为加速设备添加需要的编译标志(flags)，OpenMPAccel模块也会检查当前OpenMP的版本，如果OpenMP的版本不是v4.0或更新，就会报错
主 `CmakeLists.txt` 文件的部分摘要见Listing11.12
#### 11.3.2 Generating parallel work on the GPU with OpenMP
OpenMP的设备并行抽象要比OpenACC的更为复杂，当然这也为调度工作(schduling work)提供了更大的灵活性
要为GPU生成并行工作，我们首先需要在每个循环前添加上以下指令
```c
#pragma omp target teams distribute parallel for simd
```
该指令的前三条子句用于启用硬件资源，其中 `target` 子句用于进入加速设备/GPU，`teams` 子句用于在GPU上创建更多工作组/创建许多团队，`distribute` 子句用于在每个工作组内启用更多线程/将工作分配给各个团队
后三条子句是并行工作子句，其中 `parallel` 子句复制每个线程上的工作，`for` 子句在每个团队内分散工作，`simd` 子句将工作分散给每个线程，为了可移植性，这三条子句都是必须的，因为每个编译器对如何分散(spread out)工作有不同的实现

对于一个有三层嵌套循环的计算核，我们可以根据以下指令来分散工作
```
k loop: #pragma omp target teams distribute
j loop: #pragma omp parallel for
i loop: #pragma omp simd
```
因为每个编译器都会以不同的方式分散工作，因此我们也需要该方案的一些变体
`simd` 循环一般是最内层的循环，遍历连续的内存位置

OpenMP v5.0引入了 `loop` 子句以简化工作分配，之后会介绍

目前所介绍的工作指令(work directives)还可以添加以下的子句：`private` , `firstprivate` , `lastprivate` , `shared` , `reduction` , `collapse` , `dist_schedule`
这些子句的功能都与OpenACC中的类似，而其中一个和OpenACC不同的主要差异是当进入并行工作区域时(entering a parallel work region)，数据被处理的默认方式(the default way that data is handled)
OpenACC一般会将所有必要的数组移动到加速设备
OpenMP则有两种可能
- 默认情况下，标量和静态分配的数组在执行前会被移动到加速设备
- 分配在堆区的数据则需要显式地被拷贝到加速设备(使用 `map` 子句)或从加速设备拷贝回来
#### 11.3.3 Creating data regions to control data movement to the GPU with OpenMP
将计算工作移动至GPU后，我们需要加入数据区域以管理和GPU之间的数据移动
OpenMP的数据移动指令和OpenACC中的类似，也有结构化的和动态的两个版本，指令的形式是
```
#pragma omp target data [map() | use_device_ptr()]
```

在GPU上创建一个结构化的数据区域示例
```c
#pragma omp target data map(to:a[0:nsize], b[0:nsize], c[0:nsize])
{
#pragma omp target teams distribute parallel for simd
	for(int i = 0; i < nsize; i++){
		a[i] = 1.0;
		b[i] = 2.0;
	}

	for(int k = 0; k < ntimes; k++){
		cpu_timer_start(&tstart);
		// stream triad loop
#pragma omp target teams distribute parallel for simd
		for(int i = 0; i < nsize; i++){
			c[i] = a[i] + scalar * b[i];
		}
		time_sum += cpu_timer_stop(tstart);
	}
}
```
示例中，工作指令被包含在了一个结构化数据区域内，因此如果数据不在GPU上，就会先被拷贝到GPU，然后一直被维护到代码块结束，再拷贝回来，因此不再需要每个并行工作循环都拷贝一次数据

但结构化的数据区域无法处理更通用的编程模式，OpenMP v4.5后加入了动态数据区域，也常称为非结构化的数据区域(unstructured)，指令的形式中含有 `enter` 和 `exit` 子句，以及一个 `map` 修饰符用于指定数据传输操作(例如 `to` 和 `from` )
示例如下
```
#pragma omp target enter data map([alloc | to]: array[[start]:[length]])
#pragma omp target exit data map([from | release | delete]: array[[start]:[length]])
```

在GPU上创建一个非结构化的数据区域示例
```c
#pragma omp target enter data map(to:a[0:nsize], b[0:nsize], c[0:nsize])

	struct timespec tstart;
	// initializing data and arrys
	double scalar = 3.0, time_sum = 0.0;
#pragma omp target teams distribute parallel for simd
	for(int i = 0; i < nsize; i++){
		a[i] = 1.0;
		b[i] = 2.0;
	}

	for(int k = 0; k < ntimes; k++){
		cpu_timer_start(&tstart);
		// stream triad loop
#pragma omp target teams distribute parallel for simd
		for(int i = 0; i < nsize; i++){
			c[i] = a[i] + scalar * b[i];
		}
		time_sum += cpu_timer_stop(tstart);
	}
#pragma omp target exit data map(from:a[0:nsize], b[0:nsize], c[0:nsize] )
```
示例中，我们将 `omp target data` 指令改为了 `omp target enter data` 指令，创建了非结构化的数据区域，数据在GPU上的作用域在遇到 `omp target exit data` 指令时结束，这些指令的效果和结构化数据区域是一样的
动态数据区域主要用在更复杂的数据管理场景，例如C++中的构造函数和析构函数

当CPU和GPU之间的来回数据传输是需要的，我们可以使用 `omp target update` 指令，语法为
```
#prgma omp target update [to | from] (array[start:length])
```

对于只在设备上使用的数据，我们可以通过直接在设备上分配数据，在退出数据区域时直接删除数据来进一步减少设备和主机之间的数据移动
OpenMP提供了函数调用来直接在设备上分配内存以及释放内存，需要添加头文件
```c
#include <omp.h>
double *a = omp_target_alloc(nsize*sizeof(double), omp_get_default_device());
omp_target_free(a, omp_get_default_device());
```
CUDA也提供了相应的例程，需要添加头文件
```c
#include <cuda_runtime.h>
cudaMalloc((void *)&a, nsize * sizeof(double));
cudaFree(a);
```

分配好内存后，我们需要在并行工作指令中添加子句将设备指针(device pointers)传递给设备上的计算核
```
#pragma omp target teams distribute parallel for is_device_ptr(a)
```

综合以上的示例如下
```c
double *a = omp_target_alloc(nsize*sizeof(double), omp_get_default_device());
double *b = omp_target_alloc(nsize*sizeof(double), omp_get_default_device());
double *c = omp_target_alloc(nsize*sizeof(double), omp_get_default_device()); 

struct timespec tstart;
// initializing data and arrays
double scalar = 3.0, time_sum = 0.0;

#pragma omp target teams distribute parallel for simd \ is_device_ptr(a, b, c)
	for (int i=0; i<nsize; i++) { 
		a[i] = 1.0;
		b[i] = 2.0;
	}
for (int k=0; k<ntimes; k++){
	cpu_timer_start(&tstart);
	// stream triad loop
#pragma omp target teams distribute parallel for simd \ is_device_ptr(a, b, c) 
	for (int i=0; i<nsize; i++){
		c[i] = a[i] + scalar*b[i];
	}
	time_sum += cpu_timer_stop(tstart);
}
printf("Average runtime for stream triad loop is %lf msecs\n", time_sum/ntimes); 

omp_target_free(a, omp_get_default_device());
omp_target_free(b, omp_get_default_device());
omp_target_free(c, omp_get_default_device());
```

OpenMP中，还可以使用 `omp declare target` 指令来在设备上分配数据，见如下示例
```c
#pragma omp declare target
	double *a, *b, *c;
#pragma omp end declare target

#pragma omp target
{
	a = malloc(nsize * sizeof(double));
	b = malloc(nsize * sizeof(double));
	c = malloc(nsize * sizeof(double));
}
<unchanged code>
#pragma omp target
{
	free(a);
	free(b);
	free(c);
}
```
示例中，我们首先用 `declare target` 声明了设备指针，然后在设备上分配内存，最后释放设备上的内存
#### 11.3.4 Optimizing OpenMP for GPUs
我们可以尝试一些方法来加快单个计算核的速度，但大部分最好让编译器进行优化，以提高可移植性

在OpenMP中优化模板计算核的示例如下
```c
double** restrict x = malloc2D(jmax, imax);
double** restrict xnew = malloc2D(jmax, imax);
#pragma omp target enter data map(to:x[0:jmax][0:imax], xnew[0:jmax][0:imax])

#pragma omp target teams
{
#pragma omp distribute parallel for simd
	for(int j = 0; j < jmax; j++){
		for(int i = 0; i  < imax; i++){
			xnew[j][i] = 0.0;
			x[j][i] = 5.0;
		}
	}
#pragma omp distribute parallel for simd
	for (int j = jmax/2 - 5; j < jmax/2 + 5; j++){
		for (int i = imax/2 - 5; i < imax/2 -1; i++){
			x[j][i] = 400.0;
		}
	}
} // omp target teams

for (int iter = 0; iter < niter; iter+=nburst){
	for (int ib = 0; ib < nburst; ib++){
		cpu_timer_start(&tstart_cpu);
#pragma omp target teams distribute parallel for simd
		for (int j = 1; j < jmax-1; j++){
			for (int i = 1; i < imax-1; i++){
				xnew[j][i]=(x[j][i]+ x[j][i-1]+x[j][i+1]+ x[j-1][i]+x[j+1][i])/5.0;
			}
		}
#pragma omp target teams distribute parallel for simd
		for (int j = 0; j < jmax; j++){
			for (int i = 0; i < imax; i++){
				x[j][i] = xnew[j][i];
			}
		}
		cpu_time += cpu_timer_stop(tstart_cpu);
	}
	printf("Iter %d\n",iter+nburst);
}
#pragma omp target exit data map(from:x[0:jmax][0:imax], xnew[0:jmax][0:imax])
free(x);
free(xnew);
```
此时我们还没进行优化，使用 `nvprof` 可以查看代码中各个部分所花费的时间，我们可以发现第三个计算核占了50%以上的计算时间

我们首先尝试用 `collapse` 将两个嵌套的循环合并为一个并行结构，如下所示
```c
#pragma omp target teams
{
#pragma omp distribute parallel for simd collapse(2)
	for(int j = 0; j < jmax; j++){
		for(int i = 0; i  < imax; i++){
			xnew[j][i] = 0.0;
			x[j][i] = 5.0;
		}
	}
#pragma omp distribute parallel for simd collapse(2)
	for (int j = jmax/2 - 5; j < jmax/2 + 5; j++){
		for (int i = imax/2 - 5; i < imax/2 -1; i++){
			x[j][i] = 400.0;
		}
	}
} // omp target teams

for (int iter = 0; iter < niter; iter+=nburst){
	for (int ib = 0; ib < nburst; ib++){
		cpu_timer_start(&tstart_cpu);
#pragma omp target teams distribute parallel for simd collapse(2)
		for (int j = 1; j < jmax-1; j++){
			for (int i = 1; i < imax-1; i++){
				xnew[j][i]=(x[j][i]+ x[j][i-1]+x[j][i+1]+ x[j-1][i]+x[j+1][i])/5.0;
			}
		}
#pragma omp target teams distribute parallel for simd collapse(2)
		for (int j = 0; j < jmax; j++){
			for (int i = 0; i < imax; i++){
				x[j][i] = xnew[j][i];
			}
		}
		cpu_time += cpu_timer_stop(tstart_cpu);
	}
	printf("Iter %d\n",iter+nburst);
}
```

让我们尝试另一种将并行工作指令分割到两个循环的方法，如下所示
```c
#pragma omp target teams
{
#pragma omp distribute
	for(int j = 0; j < jmax; j++){
#pragma omp parallel for simd
		for(int i = 0; i  < imax; i++){
			xnew[j][i] = 0.0;
			x[j][i] = 5.0;
		}
	}
#pragma omp distribute
	for (int j = jmax/2 - 5; j < jmax/2 + 5; j++){
#pragma omp parallel for simd
		for (int i = imax/2 - 5; i < imax/2 -1; i++){
			x[j][i] = 400.0;
		}
	}
} // omp target teams

for (int iter = 0; iter < niter; iter+=nburst){
	for (int ib = 0; ib < nburst; ib++){
		cpu_timer_start(&tstart_cpu);
#pragma omp target teams distribute 
		for (int j = 1; j < jmax-1; j++){
#pragma omp parallel for simd
			for (int i = 1; i < imax-1; i++){
				xnew[j][i]=(x[j][i]+ x[j][i-1]+x[j][i+1]+ x[j-1][i]+x[j+1][i])/5.0;
			}
		}
#pragma omp target teams distribute 
		for (int j = 0; j < jmax; j++){
#pragma omp parallel for simd
			for (int i = 0; i < imax; i++){
				x[j][i] = xnew[j][i];
			}
		}
		cpu_time += cpu_timer_stop(tstart_cpu);
	}
	printf("Iter %d\n",iter+nburst);
}
```

优化后的运行时间详见Table 11.3
Table 11.4展示了使用OpenMP，在Power 9处理器以及NVIDIA V100 GPU上使用IBM XL v16编译器运行stream triad的运行时间

对比可以发现，使用OpenMP和IBM XL编译器在简单的1D测试问题表现优秀，可以和OpenACC，PGI编译器持平，但在2D的模板计算测试问题则不然，迄今为止，人们关注的焦点是正确实现OpenMP的设备卸载标准(standard for device offloading)，期待随着更多编译器厂商提供对OpenMP的设备卸载支持
#### 11.3.5 Advanced OpenMP for GPUs
我们简要探讨围绕以下主题的高级OpenMP指令和子句
- 微调内核(fine-tuning kernels)
- 处理多种重要的程序结构(函数、扫描/scans，和对变量的共享访问)
- 重叠了数据移动和计算的异步操作
- 控制内存放置(memory placement)
- 处理复杂的数据结构
- 简化工作指令

**Controlling the GPU kernel parameters implemented by the OpenMP compiler**
考虑一些可以微调计算核性能的子句，可以将这些字句加入指令中，以改变编译器为GPU生成的计算核
- `num_teams` 决定了 `teams` 指令生成的团队数量
- `thread_limit` 定义了每个团队使用的线程数量
- `schedule` 或 `schedule(static, 1)` 指定以循环方式(round-robin)而不是以块的方式分发工作项(work items)，这可以帮助GPU上的内存装载合并(memory load coalescing)
- `simdlen` 指定向量长度或工作组线程数
在特定情况下使用这些子句会有帮助，但通常情况下，最好让编译器选择合适的参数

**Declaring an OpenMP device function**
当我们在设备上的并行区域内调用一个函数时，我们需要告诉编译器该函数也需要在设备上执行，因此需要使用 `declare target` 指令，语法和变量声明类似，例如
```c
#pragma omp declare target
int my_compute(<args>){
	<work>
}
```

**New scan reduction type**
OpenMP v5.0及以上提供了 `scan` 类型
```c
int run_sum = 0;
#pragma omp parallel for simd reduction(inscan,+: run_sum)
for (int i = 0; i < n; i++){
	run_sum += ncells[i];
	#pragma omp scan exclusive(run_sum)
	cell_start[i] = run_sum;
	#pragma omp scan inclusive(run_sum)
	cell_end[i] = run_sum;
}
```

**Preventing race conditions with OpenMP atomic**
在一个算法中，多个线程访问一个公共变量是很常见的现象，而这也常常是程序的性能瓶颈
许多编译器实现提供了原子指令(Atomics)，用于防止线程竞争，OpenMP同样提供了 `atomic` 指令，例如
```c
#pragma omp atomic
	i++;
```

**OpenMP's version of asynchronous operations**
我们在10.5节讨论了通过异步操作(asynchronous operations)重叠数据传输和计算的重要性，OpenMP也有相应的操作
我们在数据或工作指令(data or work directive)上使用 `nowait` 子句创建异步设备操作(asynchronous device operations)，我们也可以使用 `depend` 子句以指定一个新操作只有在之前的操作完成后才能开始

使用 `taskwait` 指令可以等待所有任务(tasks)的完成
```c
#pragma omp taskwait
```

**Accessing special memory spaces**
内存带宽(memory bandwidth)是限制性能最重要的因素之一，在基于杂注的语言中，我们一般难以直接控制内存的放置(placement of memory)以及相应的内存带宽(resulting memory bandwidth)

OpenMP 5.0后，我们可以通过 `allocate` 子句的修饰符(clause modifier) `allocator` 来指定特殊内存空间为目标(target spetical memory spaces)，例如共享内存(shared memory)以及高带宽内存(high-bandwidth memory)

`allocate` 子句接受一个可选的修饰符，如下
```c
allocate([allocator:] list)
```

我们可以使用下列函数以直接分配和释放内存：
```c
omp_alloc(size_t size, omp_allocator *allocator);
omp_free(void* ptr, const omp_allocator *allocator);
```

OpenMP 5.0为 `allocator` 规定了一些预定义的内存空间(predefined memory space)，如表所示：

一组函数可以用于定义新的内存allocators，其中两个主要的例程是：
```c
omp_init_allocator
omp_destory_allocator
```
分配新的allocator的函数接受一个预定义的空间参数(predefined space arguments)，以及allocator特性(traits)，例如该内存空间是否应该pinned 固定，aligned 对齐，private 私有，nearby 邻近，等等

**Deep copy support for transferring complex data structures**
OpenMP 5.0加入了 `declare mapper` 构造(construct)，用于执行深拷贝，深拷贝及在拷贝含有指针的数据结构时，会把指针指向的数据同时拷贝
该特性简化了具有复杂数据结构和类的程序向GPU迁移的困难

**Simplifying work distribution with the new loop directive**
OpenMP 5.0标准引入了更多灵活的工作指令(work directives)，其中之一就是 `loop` 指令，该指令的功能和OpenACC中的 `loop` 指令类似
`loop` 指令替代了 `distrubute parallel for simd` ，我们使用 `loop` 指令以告诉编译器该循环可以并发执行，但将实际的实现交给编译器完成

以下是一个在模板计算核(stencil kernel)中使用 `loop` 指令的例子
Listing 11.22 Using the new `loop` directive in OpenMP5.0
```c
#pragma omp target teams // launches work on the GPU with multiple teams
#pragma omp loop // the loop parallelized as independent work
	for (int j = 1; j < jmax-1; j++){
#pragma omp loop // the loop parallelized as independent work
		for (int i = 1; i < imax-1; i++){
			xnew[j][i]=(x[j][i]+x[j][i-1]+x[j][i+1]+ x[j-1][i]+x[j+1][i])/5.0;
		}
	}
```

`loop` 子句实际上是一个 `loop independent` 或 `concurrent` 子句，告诉编译器该循环的迭代之间不存在依赖，可以并行

`loop` 子句是一个描述性子句(decriptive clause)，它不直接告诉编译器如何去做，而是给编译器一个描述性信息

Prescriptive directives and clauses 规定性指令和子句：直接告诉编译器具体去做什么
Descriptive directives and clauses 描述性指令和子句：告诉编译器下一个循环构造(loop construct)的信息，给予编译器一定自由以自己生成最高效的实现

OpenMP一般使用的是规定性子句，这可以减少实现中的差异，提高可移植性，但在GPUs上，这会导致指令复杂且长，往往需要注意到硬件特有的功能(hardware-specific features)
描述性指令更接近OpenACC的哲学，这不需要我们了解硬件的细节，将为目标硬件生成正确且高效的代码的责任交给了编译器
## 12 GPU languages: Getting down to basics
本章介绍GPU的低级语言(lower-level languages)，我们称其为本地语言(native languages)，因为它们直接反应出目标GPU硬件的特性

相较于基于杂注的实现，这些语言对编译器的依赖更小，我们可以用这些语言对程序性能进行更细粒度的控制，这些语言和CH 11介绍的语言的差异就在于这些语言是基于GPU和CPU硬件的特性建立的，而OpenACC和OpenMP则是基于高级的抽象(high-level abstractions)，且依赖于编译器将其映射到不同的硬件上

GPU本地语言：CUDA、OpenCL、HIP，都需要一个分离的源代码(seperate source)用以创建GPU计算核，分离的源代码常常和CPU代码类似
要维护两个不同的源是这类语言的主要使用困难
如果本地语言仅支持一种类型的硬件，如果我们希望自己的代码能在其他供应商的GPU上也能跑，我们甚至需要维护更多源的变体(source variants)，
因此一些应用也用多个GPU语言和CPU语言实现了它们的算法

可以知道，可移植性对于GPU语言是很重要的，OpenCL是第一个开放标准语言(open-standard language)，可以在多种GPU甚至CPU上运行
AMD设计了HIP语言，以作为CUDA的更具可移植性的版本，HIP可以运行于AMD GPUs上，也支持其他供应商的GPUs

随着新语言的推出，本地语言和高级语言之间的差异逐渐变得模糊，SYCL语言，最初是在OpenCL之上(on top of)的一层C++层，是典型的具有更好可移植性的新语言，其他的新语言还有Kokkkos、RAJA等
SYCL支持对于GPU和CPU都只需要单个源(a single source for both CPU and GPU)

GPU语言之间的互操作性(interoperability)如图12.1所示
![[PHCP-Fig12.1.png]]
语言之间的互操作性正逐渐引起关注，因为HPC系统中配备的GPU正在呈现出多样性，因此SYCL正逐渐被更多人使用，
SYCL最初被开发的目的是在OpenCL之上提供一层更自然的C++层，SYCL突然兴起的原因是将其作为了OneAPI编程模型(programming model)的一部分，OneAPI编程模型则用于Aurora系统上的Intel GPUs

本章还会介绍一些性能可移植系统(performance portability systems)，如Kokkos和RAJA，它们最初被创建的目的是为了简化运行在大范围硬件(从CPUs到GPUs)上的困难，
它们工作在稍微高一点的抽象层次，但保证只需要单个源就可以在任意硬件上运行(promise a single source that will run everywhere)，
它们的发展源于美国能源部的一个重大努力，以支持将大型科学应用移植到更新的硬件上，
RAJA和Kokkos的目的是一次重写以创建一个单源代码库(one-time rewrite to create a single-source code base)，该代码库在硬件设计出现巨大变化(great change in hardware design)时都是可移植和可维护的
### 12.1 Features of a native GPU programming language
一个GPU语言必须要具备几个基本特征：
- 检测加速设备 Detecting the accelerator device
    GPU语言必须提供对加速设备的检测方式，并提供在加速设备之间进行选择的方式，
    一些语言在设备选择上会提供比其他语言更多的控制，
    即便是像CUDA这样的仅运行在NVIDIA GPU上的语言，也必须要有处理一个结点(node)上多个GPU的方法
- 支持编写设备内核 Support for writing device kernels
    GPU语言必须提供为GPUs或其他加速设备生成低层次指令(low-level instructions)的方式，
    GPUs提供的基本操作(basic operations)和CPUs几乎一样，因此内核语言(kernel language)和CPU语言也不应有太大不同，
    故，与其发明新的语言，最直接的方法是利用现存的语言和编译器生成新的指令集，
    因此，GPU语言就采用了特定版本的C或C++语言作为基础，
    CUDA最初基于C，现在基于C++，并对STL有一定支持，OpenCL基于C99标准，并发布了带有C++支持的新规范，
    GPU语言设计时还需要解决的问题是需要将主机和设备源代码(host and design source code)放在同一个文件中还是不同的文件中，
    但不论哪种方式，编译器都要识别主机和设备源代码(host and design sources)，且必须提供为不同硬件生成指令集的方式，
    编译器也必须决定何时(when)生成指令集，例如，OpenCL的策略是等待设备被选择，然后采用即时编译的方法(just-in-time JIT compiler approach)生成指令集
- 从主机调用设备内核的机制 Mechanism to call device kernels from the host
    有了设备代码之后，我们需要有从主机调用该代码的方式，
    执行该操作的语法几乎每个GPU语言都不同，但该操作的实际机制只比标准的子例程调用(subroutine call)复杂一点点
- 内存处理 Memory handling
    GPU语言必须有对内存分配(allocations)、内存释放(deallocations)以及在主机和设备之间来回移动数据的支持，
    为此最直接的方法就是为每个这类操作提供一个子例程调用，
    另一种方式是依赖编译器来检测何时移动数据，并让编译器在幕后帮我们完成，
    内存处理是GPU编程的主要部分，因此对于该功能硬件和软件上的创新还在继续
- 同步 Synchronization
    GPU语言必须提供一种机制用于指定CPU和GPU之间的同步需求，
    内核内(within kernels)也必须提供同步操作
- 流 Streams
    一个完整(complete)的GPU语言应该允许对异步操作流(asynchronous streams of operations)的调度以及对内核和内存传输操作(memory transfer operations)之间的显式依赖(explicit dependencies)的调度
实际上，原生(native)GPU语言的大多数部分和CPU代码并无太大不同，而上述几点往往是原生GPU语言之间的共性
### 12.2 CUDA and HIP GPU languages: The low-level performance option
我们首先了解两个低级(low-level)GPU语言：CUDA和HIP，这是两个最常用的GPU语言

计算同一设备架构(Compute Unified Device Architecture CUDA)是NVIDIA的专有语言，仅运行于NVIDIA GPUs，它最早于2008年发布，如今是主导的GPU原生编程语言
CUDA语言紧密反映了NVIDIA GPU的架构，CUDA并不是一个通用的加速器语言，但大多数加速器的概念是相似的，因此CUDA语言设计可以适用

AMD GPUs(前身为ATI)已经有过一系列生命周期短的GPU语言，而它们最终采用的是通过用HIP编译器”HIPifying“CUDA代码得到的一个类似CUDA的语言，这个流程是ROCm工具套件(suite of tools)的一部分，该套件提供了GPU语言之间的广泛的可移植性(extensive portability)
#### 12.2.1 Writing and building your first CUDA application
我们从构建和编译(build and compile)可以运行在GPU上的简单的CUDA应用开始，
我们使用流三元组(stream triad)的例子，即循环执行计算$\text C = \text A + \text{scalar}*\text B$

CUDA编译器会将常规的C++代码划分出来，传递给底层的C++编译器编译，然后自己编译剩余的CUDA代码，这两条路径得到的代码会再链接到一起形成一个单一的可执行文件

CUDA的每次发布都会和有限范围的编译器版本兼容(limited range of compiler versions)，例如CUDA v10.2兼容GCC至v8版本，因此编译器的版本不符合，就无法支持特定版本的CUDA语言

用于编译StreamTriad文件的Makefile见Listing 12.1
在Listing 12.1中第九行的模式规则(pattern rule)中，我们用NVIDIA NVCC编译器将后缀为 `.cu` 的文件转化为目标文件，然后在链接时链接了CUDA运行时库(runtime library)CUDART，我们可以在Makefile中指明特定的CUDA GPU架构和CUDA库文件的路径

模式规范(pattern rule)是向make utility(实用程序)指定的一个规范(specification)，它提供了如何将具有某一后缀模式的任意文件转化为具有另一后缀模式的文件的通用规则(general rule)

CMake构建系统也对CUDA有广泛的支持，我们会讨论旧风格(old-style)的支持以及新的现代CMake方法，旧风格的方法见Listing 12.2，它的优势在于对于具有旧版CMake系统的可移植性更高，并且可以自动检测NVIDIA GPU架构，因为自动检测硬件的特性十分方便，目前也推荐使用旧风格的CMake

Listing 12.2中第11行的可分离编译属性(即CUDA_SEPARABLE_COMPILATION)建议用在更健壮的通用开发构建系统(more robust build system for general development)(即设为ON，默认该属性是OFF)，将其OFF可以节省CUDA内核中的一些寄存器，为生成的代码进行一点小幅的优化，CUDA默认是为了性能，而不是为了更通用，更健壮的构建(因此默认是OFF)
(可分离编译属性设置为ON就是允许调用其他编译单元中的函数 calling functions in other compile units)
Listing 12.4中的第14行对NVIDIA GPU架构的自动检测十分便利，可以避免我们手动修改makefile

随着ver 3.0的发布，CMake的结构正在进行重大的修改，向“现代”CMake转变，“现代”风格的关键是一个更集成的系统(more integrated system)和对每个目标都应用属性(per target application of attributes)，在其对CUDA的支持中，这一点尤为明显
“现代”风格的CMake见Listing 12.3
现代风格的CMake要比旧风格更加简单，其关键是在Listing 12.3的第四行，我们将CUDA作为语言启用(enable CUDA as the language)，之后就不需要做太多额外的工作

在Listing 12.3的第9-10行，我们为设定标志(flags)来为特定的GPU架构编译，因为在现代风格的CMake中，尚没有自动检测GPU架构的方法
没有架构标志时，编译器会为sm_30 GPU设备生成代码并进行优化，为sm_30所生成的代码实际上可以在任何Kepler K40或更新型号的设备上运行，但不会为最新的架构进行优化，
我们可以为同一编译器(在同一次编译中)指定多个架构，但编译的速度将变慢，生成的可执行文件也会更大

我们也可以为CUDA设定可分离编译属性(separable compilation attribute)，但此时的语法略有不同，我们要将其应用于特定的目标(specific target)，具体见Listing 12.3的第15行

Listing 12.3第10行上的优化标志(optimization flaf) `-O3` 只会发送给编译常规C++代码的主机编译器(host compiler)，CUDA代码的默认优化级别就是 `-O3` ，且很少需要更改

总的来说，构建CUDA程序的过程正变得容易，且构建过程会继续发展变化，Clang也正在添加为编译CUDA代码的原生支持(native support)，为我们提供除NVIDIA编译器以外的另一个选择

现在我们可以关注源码，从Listing 12.4的GPU内核开始
Listing 12.4 CUDA version of stream triad: The kernel
```c
__global__ void StreamTriad(
    const int a,
    const double scalar,
    const double *a,
    const double *b,
          double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //gets cell index
    // Protect from going out-of-bounds
    if (i >= n) return;

    c[i] = a[i] + scalar * b[i] // stream triad body
}
```
正如GPU内核的典型特征，我们从计算块(computational block)中去除了 `for` 循环，只留下了第14行的循环体语句(loop body)，在第12行我们添加了条件语句方式数据访问越界，没有这层保护内核会在没有信息提示的情况下随机崩溃(randomly crash)
在第九行，我们从CUDA运行时(run time)设定的块和线程变量(block and thread variables)中计算得到了全局索引(global index)
我们对子例程添加了 `__global__` 属性，以告诉编译器这是一个会主机调用的GPU内核(called from the host)，同时在主机端，我们要设置好内存，并调用GPU内核(set up the memory and make the kernel call)，该过程展示于Listing 12.5
Listing 12.5 CUDA version of stream triad: Set up and tear down
```c
// allocate host memory and initialize
double *a = (double *)malloc(stream_array_size * sizeof(double));
double *b = (double *)malloc(stream_array_size * sizeof(double));
double *a = (double *)malloc(stream_array_size * sizeof(double));

// initialize arrays
for (int i = 0; i < stream_array_size; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
}

// allocate device memory. suffix of _d indicates a device pointer
double *a_d, *b_d, *c_d;
cudaMalloc(&a_d, stream_array_size * sizeof(double));
cudaMalloc(&b_d, stream_array_size * sizeof(double));
cudaMalloc(&c_d, stream_array_size * sizeof(double));

// setting block size and padding total grid size
//  to get even block sizes
int blocksize = 512;
int gridsize = (stream_array_size + blocksize - 1) / blocksize; // calculates number of blocks

< ... timing loop ... code shown below in listing 12.6 >

printf("Average runtime is %lf msecs data transfer is %lf msecs\n", tkernel_sum/NTIMES, (ttotal_sum - tkernel_sum)/NTIMES);

// Frees device memory
cudaFree(a_d);
cudaFree(b_d);
cudaFree(c_d);

// Frees host memory
free(a);
free(b);
free(c);
```
在以上代码中，我们首先在主机段分配了内存，并对其初始化，同时我们也需要在GPU上也有对应的存储空间，以便GPU在运算时存储这些数组数据，故我们使用 `cudaMalloc` 例程分配GPU内存
在47到49行，我们首先赋值了 `blocksize` ，块大小就是GPU工作组的大小，它的别名可以是tile size、block size或workgroup size，取决于使用的GPU编程语言(见Table 10.1)，然后我们计算了网格大小(grid size)，因为我们的数组大小不可能总是可以整除块大小，故我们进行了padding

> Example: Calculating block size for the GPU
> 在以下代码的第3行，我们计算块的数量(分数形式)，对于这个数组大小为1,000的示例，它有1.95个块，
> 如果使用整数算术，默认情况下它会被截断为1，所以我们必须将这些值中的每一个强制转换为浮点值，以进行浮点除法，我们实际上只需要将其中一个值进行强制转换，因为根据C/C++标准，编译器会对其他项的数据类型升级(promote)，但在编程惯例中，最好显式调用类型转换
>在第4和第5行使用了C语言 `ceil` 函数向上取整到下一个等于或大于当前浮点数的整数值，
>第6行的做法是加上比块大小少一的数，再执行整数除法并截断，也可以获得相同的结果，我们选择使用这个版本，因为整数形式不需要任何浮点运算，应该会更快
>`1 int stream_array_size = 1000`
>`2 int blocksize = 512`
>`3 float frac_blocks = (float) stream_array_size/(float)block_size;
>`>>>frac_blocksize = 1.95`
>`4 int nblocks = ceil(frac_blocks);
>`>>>nblocks = 2`
>或
>`5 int nblocks = ceil((float) stream_array_size/(float)blocksize);
>`6 int nblocks = (stream_array_size + blocksize - 1) / blocksize;`

现在，除了最后一个块外，所有块都有512个值，最后一个块的大小是512，但只包含488个数据项，在Listing 12.4的第12行的越界检查就是为了防止我们在这个部分填充(partially filled)的块上遇到麻烦，Listing 12.5中的最后几行释放了设备指针和主机指针，我们使用 `cudaFree` 函数来释放设备指针，以及使用C库函数 `free` 来释放主机指针

我们剩下的工作就是将内存复制到GPU，调用GPU内核，然后将内存复制回来，我们在一个计时循环中执行这些操作(见Listing 12.6)，这个循环可以执行多次以获得更好的测量结果，有时，由于初始化成本，第一次调用GPU会更慢，我们可以通过运行多个迭代来分摊这个成本，如果这样做还不够，也可以放弃第一次迭代的时间测量
Listing 12.6 CUDA version of stream triad: Kernel call and timing loop
```c
for (int k = 0; k < NTIMES; k++) {
    cpu_timer_start(&ttotal);
    cudaMemcpy(a_d, a, stream_array_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, stream_array_size * sizeof(double), cudaMemcpyHostToDevice);
    // cuda memcopy to device returns after buffer available
    cudaDeviceSynchronize();

    cpu_timer_start(&tkernel);
    StreamTriad<<<gridsize, blocksize>>>(stream_array_size, scalar, a_d, b_d, c_d);
    cudaDeviceSynchronize();
    tkernel_sum += cpu_timer_stop(tkernel);

    // cuda memcpy from device to host blocks for completion, so no seed for synchronize
    cudaMemcpy(c, c_d, stream_array_size * sizeof(double), cudaMemcpyDeviceToHost);
    ttotal_sum += cpu_timer_stop(ttotal);
    for (int i=0, icount=0; i<stream_array_size && icount < 10; i++){
    if (c[i] != 1.0 + 3.0*2.0) {
        printf("Error with result c[%d]=%lf on iter %d\n",i,c[i],k);
        icount++;
    } // if not correct, print error
} // result checking loop
} // timing for loop
```
计时循环内的模式由三步构成：
1. 将数据拷贝至GPU
2. 调用GPU内核对数据进行计算
3. 将数据拷贝回来

通过三重尖括号或角括号，可以很容易地发现对GPU内核的调用，如果我们忽略尖括号及尖括号内包含的变量，该行具有典型的C子程序调用语法(subroutine call syntax)：
`StreamTriad(stream_array, scalar, a_d, b_d, c_d);`
圆括号内的值是要传递给GPU内核的参数(arguments)

`<<<gridsize, blocksize>>>`
尖括号内的值则是要传递给CUDA编译器的参数，它指导CUDA编译器如何将问题为GPU划分为块，我们在Listing 12.2已经设定了块大小并计算和块的数量(或者说网格大小 grid size)以包含数组中的所有数据
在该例中，参数是一维的，我们也可以通过以下方式声明和设置这些参数，来拥有二维或三维数组，用于NxN矩阵
```
dim3 blocksize(16, 16); dim3 blocksize(8, 8, 8);
dim3 gridsize( (N + blocksize.x - 1)/ blocksize.x,
               (N + blocksize.y - 1)/ blocksize.y);
```

我们可以通过消除数据复制来加快内存传输(memory transfer)，
通常的内存分配(memory allocations)会被放置到分页内存中(be placed into pageable memory)，或者称为可以被按需移动的内存(be moved on demand)，而内存传输必须先将数据移动到固定内存(pinned memory)中，或者称为不能移动的内存，我们可以通过在固定内存而不是分页内存中分配数组来消除内存复制，
图9.8显示了我们可能获得的性能差异

CUDA为我们提供了一个函数调用 `cudaHostMalloc`，为我们实现了这一点。它是系统 `malloc` 例程的替代，其参数有轻微的变化，其中指针作为参数返回，如下所示：
```c
double *x_host = (double *)malloc(stream_array_size*sizeof(double));
cudaMallocHost((void*)&x_host, stream_array_size*sizeof(double));
```
使用固定内存也有缺点，如果一个应用使用了大量的固定内存，其他应用程序可用的交换内存空间就会不足
交换出(swap out)一个应用程序的内存并引入另一个应用程序的内存这种操作对用户来说是一个巨大的便利，这个过程被称为内存分页(memory paging)

**Definition**：在多用户、多应用程序操作系统中，内存分页是将内存页(memory pages)临时移动到磁盘(to disks)上，以便另一个进程可以进入内存的过程

内存分页是操作系统的一个重要进步，它让机器看起来好像拥有比实际上更多的内存，例如，它允许你在处理Word文档时临时启动Excel，而不必关闭原始应用程序，这是通过将数据先写入磁盘，然后在返回Word时再将其读回实现的，但这个操作代价很大，所以在高性能计算中，我们避免内存分页，因为它会带来严重的性能损失(severe performance penalty)

一些同时具有CPU和GPU的异构计算系统正在实现统一内存(unified memory)

**Definition**：统一内存是对CPU和GPU表现出单一地址空间的内存(the appearence of being a single address space)

我们已经看到，处理分离的CPU和GPU内存空间为编写GPU代码引入了许多复杂性，有了统一内存，GPU运行时系统(runtime system)会为我们处理这些，可能仍然要有两个单独的数组，但数据会被自动移动，在集成GPU上，有可能根本不需要移动内存
不过，我们仍建议编写程序时显式拷贝内存(with explicit memory copies)，以使我们的程序能够移植到没有统一内存的系统上，如果架构不需要，内存拷贝会被跳过
#### 12.2.2 A reduction kernel in CUDA: Life gets complicated
在较低级别的原生(lower level, native)GPU语言中，实现GPU线程之间的协作(cooperation among GPU threads)较为复杂

我们将通过一个简单的求和示例来看看如何处理这个问题，该示例需要两个独立的CUDA内核，它们分别在Listing 12.7-12.10中展示

以下Listing展示了第一步骤，我们在一个线程块内(within a thread block)对值进行求和，并将结果存储回归约暂存数组(reduction scratch array) `redscratch`
Listing 12.7: First pass of a sum reduction operation
`CUDA/SumReduction/SumReduction.cu (four parts)`
```c
__global__ void reduce_sum_stage1of2(
                const int isize, //0 Total number of cells
                double *array, //1
                double *blocksum, //2
                double *redscratch) //3
{
    extern __shared__ double spad[]; // scratchpad array in CUDA shared memory
    const unsigned int giX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tiX = threadId.x;
    const unsigned int group_id = blockIdx.x;

    spad[tiX] = 0.0;
    if (giX < isize) { // load memory into scratchpad array
        spad[tiX] = array[giX];
    }

    __syncthreads() // synchronize threads before using scratchpad data

    reduction_sum_within_block(spad); // sets reduction within thread block

    // write the local value back to an array
    //  the size of the number of groups
    if (tiX == 0){
        redscratch[groud_id] = spad[0];
        (*blocksum) = spad[0];
    }
}
```
我们首先通过让所有线程将它们各自的数据存储到CUDA共享内存(CUDA shared memory)中的一个暂存数组(scratchpad array)来开始第一轮处理，块中的所有线程(all thread in the block)都可以访问这个共享内存
访问共享内存仅需要一个或两个处理器周期(processor cycles)内被访问，而访问GPU主存(main GPU memory)则需要数百个周期，我们可以将共享内存想象为一个可编程的缓存(programmable cache)或者作为一块暂存内存(scratchpad memory)，
为了确保所有线程完成了存储操作，我们在第40行使用了一个同步调用(synchronization call)

由于块内的归约求和(reduction sum within the block)将在两个归约阶段中都被使用，我们把代码放在一个设备子例程(device subroutine)中，并在第42行调用它
设备子例程是应该由另一个设备子例程调用而不是由主机调用的子例程，在调用子例程之后，我们将得到的求和结果存储到我们将在第二阶段读取的更小的暂存数组中，我们还在第47行存储了结果，以防第二阶段可以被跳过，
因为我们无法访问其他线程块(other thread blocks)中的值，我们必须在另一个内核完成第二阶段的操作，
在这一阶段中，我们已将需要处理的数据长度除以了块大小(reduce the length of the data by our block size)

我们需要在两个阶段都需要对CUDA线程块(thread block)进行求和归约，所以我们将其编写为一个通用的设备例程(general device routine)，如下所示，注意下面列出的代码也可以很容易地修改为其他归约操作，并且对于HIP和OpenCL也只需要小的改动
Listing 12.8: Common sum reduction device kernel
`CUDA/SumReduction/SumRedution.cu (four parts)`
```c
#define MIN_REDUCE_SYNC_SIZE warpSize // CUDA defines warpSize to be 32

__device__ void reduction_sum_within_block(double *spad)
{
    const unsigned int tiX = threadIdx.x;
    const unsigned int ntX = blockDim.x;

    for (int offset = ntX >> 1; offset > MIN_REDUCE_SYNC_SIZE; offset >>=1) { // only use threads needed when greater than the warp size
        if (tiX < offset) {
            spad[tiX] = spad[tiX] + spad[tiX + offset];
        }
        __syncthreads(); // synchronizes between every level of the pass
    }
    if (tiX < MIN_REDUCE_SYNC_SIZE) {
        for (int offset = MIN_REDUCE_SYNC_SIZE; offset > 1; offset >>=1) {
            spad[tiX] = spad[tiX] + spad[tiX+offset];
            __syncthraeds(); // synchronizes between every level of the pass
        }
        spad[tiX] = spad[tiX] + spad[tiX + 1]; 
    }
}
```
该段代码定义了会被两遍(both passes)都调用的公用设备例程(common device routine)，它的功能是在线程块内执行求和归约，在例程前的 `__device__` 属性表明它在一个GPU内核中被调用，该例程的基本思想是如图12.2所示的，操作复杂度是在$O(\log n)$的成对归约树(pairwise reduction tree)
![[PHCP-Fig12.2.png]]

当工作集(working set)大于线程束(warp)大小时，我们在8-13行实现了一些微小的修改

相同的成对归约思想也可以用于全线程块(full-thread block)，在大多数GPU设备上全线程块包含的线程数量可以达到1024，尽管128到256更常用，如果我们的数组大小大于1024时，我们可以添加第二遍(add a second pass)，就如以下列表所示，它只使用一个单一的线程块
Listing 12.9 Second pass for reduction operation
`CUDA/SumReduction/SumReduction.cu (four parts)`
```c
__global__ void reduce_sum_stage2of2(
    const int isize,
    double *total_sum,
    double *redscratch)
{
    extern __shared__ double spad[];
    const unsigned int tiX = threadIdx.x;
    const unsigned int ntX = blockDim.x;

    int giX = tiX;

    spad[tiX] = 0.0;

    // load the sum from reduction scratch, redscratch
    if (tiX < isize) spad[tiX] = redscratch[giX]; // load values into srcatchpad array

    for (giX += ntX; giX < isize; giX += ntX) {
        spad[tiX] += redscratch[giX]; // loops by thread block-size increments to get all the data
    }

    __syncthreads(); // synchronizes when scratchpad array is filled

    reduction_sum_within_block(spad); // calls our common block reduction routine

    if (tiX == 0) {
        (*total_sum) = spad[0]; // one thread sets the total sum for return
    }
}
```
为了避免在处理大型数组时使用超过两个内核，我们使用一个线程块，并在67-69行上循环读取并将额外的数据求和到共享的暂存区(shared scratchpad)，我们使用单一线程块的原因就是我们可以在块内同步，从而避免了另一次内核调用的需要，
如果我们使用的线程块大小为128，并且有一个百万元素的数组，67-69行的循环将把大约60个值求和到共享内存的各个位置($1000000/128^2$)(数组大小在第一遍时减少了128倍，然后再求和到一个大小为128的暂存区)，如果我们使用更大的块大小(block size)，比如1024，我们可以将67-69行的循环从60次迭代减少到一次读取

在该循环之后，我们只需要调用之前使用过的公用的线程块归约例程(common thread block reduction)，得到的结果将是暂存数组中的第一个值

完成以上代码后，最后的部分就是从主机设置并调用(set up and call)这两个内核，如以下列表所示：
Listing 12.10 Host code for CUDA reduction
`CUDA/SumReduction/SumReduction.cu (four parts)`
```c
// calculates the block and grid sizes for CUDA kernels
size_t blocksize = 128;
size_t blocksizebytes = blocksize * sizeof(double);
size_t global_work_size = ((nsize + blocksize - 1) / blocksize) * blocksize;
size_t gridsize = global_work_size / blocksize;

// allocates device memory for the kernel
double *dev_x, *dev_total_sum, *dev_redscratch;
cudaMalloc(&dev_x, nsize*sizeof(double));
cudaMalloc(&dev_total_sum, 1*sizeof(double));
cudaMalloc(&dev_redscratch, gridsize*sizeof(double));

cudaMemcpy(dev_x, x, nsize*sizeof(double), cudaMemcpyHostToDevice); // copies the array to the GPU device

// calls the first pass of the reduction kernel
reduce_sum_stage1of2<<<gridsize, blocksize, blocksizebytes>>>(nsize, dev_x, dev_total_sum, dev_redscratch);

if (grid > 1) {
    // if needed, calls the second pass
    reduce_sum_stage2of2<<<1, blocksize, blocksizebytes>>>(nsize, dev_total_sum, dev_redscratch);
}

double total_sum;
cudaMemcpy(&total_sum, dev_total_sum, 1*sizeof(double), cudaMemcpyDeviceToHost);
printf("Result -- total sum %lf \n", total_sum);

cudaFree(dev_redscratch);
cudaFree(dev_total_sum);
cudaFree(dev_x);
```
主机代码中，我们首先在100-103行计算内核调用需要用到的大小，然后，我们必须为设备数组(device arrays)分配内存，
我们需要分配一个暂存数组，用于存储第一个内核计算得到的的每个块内的总和，因此我们在108行分配它，其大小即为网格大小，注意块数就是网格大小，
我们还需要一个共享的内存暂存数组，其大小是块大小，我们在112行和115行将其作为第三个参数传递给尖括号运算符(chevron operator)进入内核，第三个参数是可选参数，这是我们第一次看到它被使用

这个线程块归约示例就是一个对需要线程合作(thread cooperation)的内核的一般介绍，这个算法可以进一步优化，但沃恩也可以考虑使用一些库服务，如CUDA UnBound(CUB)、Thrust或其他GPU库
#### 12.2.3 Hipifying the CUDA code
CUDA代码只能在NVIDIA GPU上运行，但AMD也实现了一个类似的GPU语言，并将其命名为可移植异构接口(HIP Heterogeneous Interface for Portability)，它是AMD的Radeon Open Compute平台(ROCm)工具套件(suite of tools)的一部分，如果我们要使用HIP语言编程，我们可以调用hipcc编译器，该编译器在NVIDIA平台上使用NVCC，在AMD GPU上使用HCC

Listing 12.12 The HIP differences for the stream triad
`HIP/StreamTriad/StreamTriad.c`
```c
#include "hip/hip_runtime.h"
<...skipping...>
// allocate device memory, suffix of _d indicates a device pointer
double *a_d, *b_d, *c_d;
// cudaMalloc becomes hipMalloc
hipMalloc(&a_d, stream_array_size * sizeof(double));
hipMalloc(&b_d, stream_array_size * sizeof(double));
hipMalloc(&c_d, stream_array_size * sizeof(double));
<...skipping...>
for (int k = 0; k<NTIMES; k++) {
    cpu_timer_start(&ttoal);
    // copying arrray data from host to device
    // cudaMemcpy becomes hipMemcpy
    hipMemcpy(a_d, a, stream_array_size*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(b_d, a, stream_array_size*sizeof(double), hipMemcpyHostToDevice);
    // cuda memcopy to device returns after buffer available,
    // so synchronize to get accurate timing for kernel only
    hipDeviceSynchronize(); // cudaDeviceSynchronize becomes hipDeviceSynchronize

    cpu_timer_start(&tkernel);
    // launch stream triad kernel
    hipLaunchKernelGGL(StreamTriad, dim3(gridsize), dim3(blocksize), 0, 0, stream_array_size, scalar, a_d, b_d, c_d);
    // hipLaunchKernel is a more traditional syntax than the CUDA kernel launch

    // need to force completion to get timing 
    hipDeviceSynchronize();
    tkernel_sum += cpu_timer_stop(tkernel);

    // cuda memcpy from device to host blocks for completion
    // so no need fro synchronize
    hipMemcpy(c, c_d, stream_array_size*sizeof(double), hipMemcpyDeviceToHost);
<...skipping...>
}
<...skipping...>

// hipFree replaces cudaFree
hipFree(a_d);
hipFree(b_d);
hipFree(c_d);
```
要将CUDA源码转换为HIP源码，我们将源码内所有的 `cuda` 替换成 `hip` ，唯一一个显著的改变是内核启动调用(kernel launch call)，HIP使用的是比CUDA中使用的三重尖括号更传统的语法
### 12.3 OpenCL for a portable open source GPU language
由于对可移植(portable)GPU代码的巨大需求，一种新的GPU编程语言OpenCL在2008年应运而生，OpenCL是一个开放标准(open standard)的GPU语言，可以在NVIDIA和AMD/ATI显卡以及许多其他硬件设备上运行
OpenCL标准的建立由苹果公司领导，还有许多其他组织参与，OpenCL的一个优点是，你可以使用几乎所有的C语言甚至C++编译器来编译主机代码(host code)，对于GPU设备代码(device code)，OpenCL最初基于C99的一个子集，最近，OpenCL的2.1和2.2版本增加了对C++14的支持，但尚没有相应的实现(implementations)

OpenCL发布时引起了许多关注，因为终于有一种可以编写可移植GPU代码的方法，GIMP宣布它将支持OpenCL作为在许多硬件平台上提供GPU加速的一种方式，但现实是，许多人认为OpenCL太底层(low-level)、太冗长(verbose)，难以被广泛接受，它最终的角色可能是作为更高级语言的底层可移植性层(low-level portability layer)，但OpenCL作为一种在多种硬件设备上通用的语言的价值，已经通过它在嵌入式设备(embedded device)社区中的接受度得到了证明，特别是在现场可编程门阵列(FPGAs)中
OpenCL被认为冗长的一个原因是设备选择(device selection)更加复杂(也更强大)，OpenCL中，我们必须检测并选择我们将运行的设备(detect and select the device you will run on)，因此有可能仅仅是为了开始就需要一百行代码

几乎所有使用OpenCL的人都编写了一个库来处理底层问题，我们也不例外。我们的库叫做EZCL
几乎每个OpenCL调用都至少被一个轻量级包装层所包装以处理错误情况(be warpped with at least a light layer to handle the error conditions)，OpenCL库中，设备检测(device detection)、编译代码(compiling code)和错误处理(error handling)占用了大量的代码行

在我们的示例中，我们将使用我们EZCL库的简单版本(abbreviated version)，称为EZCL_Lite，以便可以看到实际的OpenCL调用，
EZCL_Lite例程(routines)用于选择设备(select the device)并对其进行设置以应用(set it up for application)，然后编译设备代码并处理错误
#### 12.3.1 Writing and building your first OpenCL application
用于构建OpenCL应用的Makefile见Listing 12.13
Makefile中，我们将 `DEVICE_DETECT_DEBUG` 标志(flag)设为1，以使其在构建时打印出可用的GPU设备的详细信息，在Makefile的第6行还增加了一个模式规则(pattern rule)，该规则将OpenCL源代码嵌入在运行时使用的程序中(embeds the OpenCL source into the program for use at run time)，其中的Perl脚本会将源代码转换为注释(`StreamTriad_kernel.inc`)，并将其作为第9行的依赖项，它会通过 `include` 语句被包含在 `StreamTriad.c` 文件中

`embed_source.pl` 是我们开发的，用于将OpenCL源代码直接链接到可执行文件中(link the OpenCL source directly into the executable)的工具(utility)(此工具的源代码参见本章示例)，
OpenCL代码的常见工作方式是有一个单独的源文件(a seperate source file)，该文件必须在运行时被定位(be located at run time)，一旦确定了设备，该文件就会被进行编译(be compiled once the device is known)，但使用单独的文件可能会造成找不到文件(not being able to be found)或获取错误版本的文件(getting the wrong version of the file)的问题，我们强烈建议将源代码嵌入到可执行文件中(embed the source into the executable)以避免这些问题

用于构建OpenCL应用的CMake文件见Listing 12.14
CMake在3.1版本中添加了对OpenCL的支持。我们在 `CMakelists.txt` 文件的第一行添加了这个版本要求，
我们还使用了CMake命令的`-DDEVICE_DETECT_DEBUG=1` 选项来开启设备检测的详细输出(the verbosity for the device detection)，以及设定了对OpenCL双精度的支持，在EZCL_Lite代码中，我们通过对OpenCL双精度的开启和关闭来设置OpenCL设备代码的即时编译(JIT)编译标志(set the just-in-time compile flag for the OpenCL device code)，
最后，我们在19-22行添加了一个自定义命令来将OpenCL设备源代码(device source)嵌入到可执行文件中

OpenCL内核的源代码在一个单独的文件中，称为 `StreamTriad_kernel.cl` ，如下所示
Listing 12.15 OpenCL kernel
`OpenCL/StreamTriad/StreamTriad_kernel.cl`
```c
// OpenCL kernel version of stream triad
__kernel void StreamTriad( // __kernel attribute indicates this is called from the host
    const int n,
    const double scalar,
    __global const double *a,
    __global const double *b,
    __global double *c)
{
    int i = get_global_id(); // get the thread index

    // Protect from going out-of-bounds
    if (i >= n) return;

    c[i] = a[i] + scalar*b[i];
}
```
OpenCL的内核代码与列表12.4中的CUDA内核代码几乎相同，只是 `__kernel` 替换了子例程声明(subroutine declaration)中的 `__global__` ，并且指针参数声明前添加了 `__global` ，以及获取线程索引(thread index)的方式不同，
此外，CUDA内核代码(kernel code)与主机的源代码(the source for the host)位于同一个 `.cu` 文件中，而OpenCL代码位于一个单独的 `.cl` 文件中，我们也可以将CUDA代码分离到它自己的 `.cu` 文件中，并将主机代码放入标准的C++源文件中，这就类似于我们为OpenCL应用程序使用的结构

注意：许多OpenCL和CUDA内核代码之间的差异都仅是表面的(superficial)

Listing 12.16展示了OpenCL主机端代码(host-size code)，我们的OpenCL流三元组(stream triad)有两种版本：没有错误检查(error checking)的 `StreamTriad_simple.c` 和有错误检查的 `StreamTriad.c` ，错误检查会增加许多行代码，因此此处展示的是简单版本
Listing 12.16 OpenCL version of stream triad: Set up and tear down
`OpenCL/StreamTriad/StreamTriad_simple.c`
```c
#include "StreamTriad_kernel.inc"
#ifdef __APPLE_CC__ // Apple has to be different
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif
#include "ezcl_lite.h" // Our EZCL_Lite support library
<...skipping code...>
cl_command_queue command_queue;
cl_context context;
iret = ezcl_devtype_init(CL_DEVICE_TYPE_GPU, &command_queue, &context); // Gets the GPU device
const char *defines = NULL;
cl_program program = ezcl_create_program_wsource(context, defines, StreamTriad_kernel_source); // Creates the program from the source
cl_kernel kernel_StreamTriad = clCreateKernel(program, "StreamTriad", &iret); // Compiles the StraemTriad kernel in the source

// allocate device memory, suffix of _d indicates a device pointer
size_t nsize = stream_array_size * sizeof(double);
cl_mem a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, nsize, NULL, &iret);
cl_mem b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, nsize, NULL, &iret);
cl_mem c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, nsize, NULL, &iret);

// setting work group size and padding
//   to get even number of workgroups
size_t local_work_size = 512;
size_t global_work_size = ( (stream_array_size + local_work_size - 1) / local_work_size) * local_work_size; // Work group size calculation is similar to CUDA
<...skipping code...>

clReleaseMemObject(a_d);
clReleaseMemObject(b_d);
clReleaseMemObject(c_d);

// cleans up kernel and device-related objects
clReleaseKernel(kernel_StreamTriad);
clRelaeseCommandQueue(command_queue);
clReleaseContext(context);
clReleaseProgram(program);
```
在程序的开始部分，我们在第34-37行遇到了一些和CUDA的差异，我们必须找到我们的GPU设备并编译我们的设备代码(device code)，而CUDA中则会在幕后(behind the scenes)为我们完成这些事，OpenCL代码中的两行调用了我们的EZCL_Lite例程来检测设备并创建程序对象(to detect the device and to create the program object)，我们之所以进行这些调用，是因为这些函数所需的代码量太长，无法在这里展示，这些例程的源代码长达数百行，尽管其中大部分是错误检查
其余的设置和清理(set up and tear down)代码遵循我们在CUDA代码中看到的相同模式，只是需要更多的清理工作(cleanup)，这同样与设备和程序源处理有关(device and program source handling)

调用OpenCL内核的代码部分见Listing 12.17
Listing 12.17: OpenCL version of stream triad: Kernel call and timing loop
`OpenCL/StreamTriad/StreamTriad_simple.c`
```c
for (int k = 0; k < NTIMES; k++) {
    cpu_timer_start(&ttotal);
    // copying array data from host to device
    // Memmory movement calls
    iret = clEnqueueWriteBuffer(command_queue, a_d, CL_FALSE, 0, nsize, &a[0], 0, NULL, NULL);
    iret = clEnqueueWriteBuffer(command_queue, b_d, CL_TRUE, 0, nsize, &b[0], 0, NULL, NULL);

    cpu_timer_start(&tkernel);
    // set stream triad kernel arguments
    iret = clSetKernelArg(kernel_StreamTriad, 0, sizeof(cl_int), (void *)&stream_array_size);
    iret = clSetKernelArg(kernel_StreamTriad, 1, sizeof(cl_double), (void *)&scalar);
    iret = clSetKernelArg(kernel_StreamTriad, 2, sizeof(cl_mem), (void *)&a_d);
    iret = clSetKernelArg(kernel_StreamTriad, 3, sizeof(cl_mem), (void *)&b_d)
    iret = clSetKernelArg(kernel_StreamTriad, 4, sizeof(cl_mem), (void *)&c_d))
    // call stream triad kernel
    clEnqueueNDRangeKernel(command_queue, kernel_StreamTriad, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    // need to force completion to get timing
    clEnqueueBarrier(command_queue);
    tkernel_sum += cpu_timer_stop(tkernel);
    // synchronization barrier
    iret = clEnqueueReadBuffer(command_queue, c_d, CL_TRUE, 0, nsize, c, 0, NULL, NULL);
    ttotal_sum += cpu_timer_stop(ttotal);
}
```
在第57-61行中，我们依照OpenCL的要求，对每个内核参数进行单独调用(a seperate call for every kernel argument)，如果我们还要检查每个调用的返回代码，将需要更过的代码行，这比CUDA版本要冗长得多，但两个版本之间仍有直接的对应关系(direct correspondence)，OpenCL只是在描述传递参数的操作时(operations to pass the arguments)更加冗长
除了设备检测(device detection)和程序编译(program compilation)之外，CUDA和OpenCL程序在操作方面是相似的，最大的区别仅是两种语言中使用的语法(syntax)

Listing 12.18展示了设备检测和创建程序调用的大致调用顺序(a rough call sequence for the device detection and the create program calls)，错误检查和对特殊情况的处理要求使得这些例程较为冗长，但对于这两个功能，拥有良好的错误处理(error handling)非常重要，我们需要编译器报告源代码中的错误，或者在它获取了错误的GPU设备时报告错误
Listing 12.18 OpenCL support library ezcl_lite
`OpenCL/StreamTriad/ezcl_lite.c`
```text
/* init and finish routine */
cl_int ezcl_devtype_init(cl_device_type device_type, cl_command_queue *command_queue, cl_context *context);
clGetPlatformIDs -- first to get number of platforms and allocate
clGetPlatformIDs -- now get platforms
Loop on number of platforms and
    clGetDeviceIDs -- once to get number of devices and allocate
    clGetDeviceIDs -- get devices
    check for double precision support -- clGetDeviceinfo
End loop
clCreateContext
clCreateCommandQueue

/* kernel and program routines */
cl_program ezcl_create_program_wsource(cl_context context, const char *defines, const char *source);
    clCreateProgramWithSource
    set a compile string (hardware specific options)
    clBuildProgram
    Check for error, if found
    clGetProgramBuildInfo
    and printout compile report
End error handling
```
许多语言都为OpenCL创建了接口，有C++、Python、Perl和Java版本，这些语言都为OpenCL创建了更高级别(higher-level)的接口，这些接口隐藏了C版本OpenCL中的一些细节，我们也强烈推荐使用我们的EZCL库或OpenCL的其他许多中间件库(middleware libraries)

自OpenCL v1.2以来，就有了一个非官方的C++版本，其实现仅仅是在C版本的OpenCL之上的一个薄层(thin layer)。尽管未能获得标准委员会的批准，但它完全可供开发者使用，它可在 https://github.com/KhronosGroup/OpenCL-CLHPP 上找到，
C++在OpenCL中的正式批准(formal approval)最近才发生，我们仍在等待其实现
#### 12.3.2 Reductions in OpenCL
OpenCL中的求和归约程序和CUDA中的也是类似的，Figure 12.3展示了二者的内核源代码中 `sum_within_block` 例程的差异，该设备内核是用于被另一个内核调用的
![[PHCP-Fig12.3.png]]
二者的差异始于声明上的属性(attributes on the declaration)，CUDA要求在声明上使用 `__device__` 属性，而OpenCL则不需要；对于参数，OpenCL中，传入的暂存数组(scratchpad array)需要一个 `__local` 属性，而CUDA不需要；另一个差异是获取本地线程索引(local thread index)和块大小(block/tile size)的语法(Figure 12.3中的第5行和第6行)；同步调用(synchronization call)的语法也不同；
例程的顶部需要通过宏定义定义一个线程束大小(warp size)来帮助在NVIDIA和AMD GPU之间实现可移植性，CUDA将其定义为一个线程束大小变量(warp-size variable)，而OpenCL则是通过编译器定义传入的(be passed in with a compiler define)

Figure 12.4中展示了两个内核中的第一个，即 `reduce_sum_stage1of2` ，该内核是从主机调用的(called from the host)，CUDA的 `__global__` 属性变成了OpenCL的 `__kernel` ，我们还必须为OpenCL的指针参数添加 `__global` 属性
![[PHCP-Fig12.4.png]]
下一个差异是值得注意的重要差异，在CUDA中，我们在内核代码中将共享内存中的scratchpad数组声明为 `extern __shared__` 变量，在主机端，这个共享内存空间的大小以字节为单位，在三重尖括号中的可选第三个参数(optional third argument)中给出；OpenCL中，该共享暂存数组则以 `__local` 属性作为参数列表中的最后一个参数传递(the last argument in the argument list)给内核，在主机端，该存储空间是在第四个内核参数的参数调用中指定的(specified in the set argument call for the fourth kernel argument)：
```c
clSetKernelArg(reduce_sum_1of2, 4, local_work_size*sizeof(cl_double), NULL);
```
该调用的第三个参数指定了该数组的大小

剩余的差异在于设置线程参数(thread parameters)和同步调用的语法

求和归约内核的第二遍的比较见Figure12.5
![[PHCP-Fig12.5.png]]
Figure12.5中所有的变更模式(change pattern)我们都已经在之前见过，可以看到仍然有内核声明和参数声明上的差异，局部的暂存数组(local scratch array)也与第一遍的内核有相同的差异，线程参数和同步也具有预期的相同差异

回顾Figure12.3-5中的三次比较，我们可以发现内核的主体(body)本质上是相同的

OpenCL中求和归约的主机端代码如Listing 12.19所示：
Listing 12.19 Host code for the OpenCL sum reduction
`OpenCL/SumReduction/SumReduction.c`
```c
cl_context context;
cl_command_queue command_queue;
ezcl_devtype_init(CL_DEVICE_TYPE_GPU, &command_queue, &context);

const char *defines = NULL;
cl_program program = ezcl_create_program_wsource(context, defines, SumReduction_kernel_source);
// two kernels to create from a single source
cl_kernel reduce_sum_1of2 = clCreateKernel(program, "reduce_sum_stage1of2_cl", &iret);
cl_kernel reduce_sum_2of2 = clCreateKernel(program, "reduce_sum_stage2of2_cl", &iret);

struct timespec tstart_cpu;
cpu_timer_start(&tstart_cpu);

size_t local_work_size = 128;
size_t global_work_size = ((nsize + local_work_size - 1) / local_work_size) * local_work_size;
size_t = nblocks = global_work_size / local_work_size;

cl_mem dev_x = clCreateBuffer(context, CL_MEM_READ_WRITE, nsize*sizeof(double), NULL, &iret);
cl_mem dev_total_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, 1*sizeof(double), NULL, &iret);
cl_mem dev_redscratch = clCreateBuffer(context, CL_MEM_READ_WRITE, nblocks*sizeof(double), NULL, &iret);

clEnqueueWriteBuffer(command_queue, dev_x, CL_TRUE, 0, nsize*sizeof(cl_double), &x[0], 0, NULL, NULL);

// calls first reduction pass
clSetKernelArg(reduce_sum_1of2, 0, sizeof(cl_int), (void *)&nsize);
clSetKernelArg(reduce_sum_1of2, 1, sizeof(cl_mem), (void *)&dev_x);
clSetKernelArg(reduce_sum_1of2, 2, sizeof(cl_mem), (void *)&dev_total_sum);
clSetKernelArg(reduce_sum_1of2, 3, sizeof(cl_mem), (void *)&dev_redscratch)
clSetKernelArg(reduce_sum_1of2, 3, local_work_size*sizeof(cl_double), NULL);

clEnqueueNDRangeKernel(command_queue, reduce_sum_1of2, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

// calls second reduction pass
if (nblocks > 1) {
    clSetKernelArg(reduce_sum_2of2, 0, sizeof(cl_int), (void *)&nblocks);
    clSetKernelArg(reduce_sum_2of2, 1, sizeof(cl_mem), (void *)&dev_total_sum);
    clSetKernelArg(reduce_sum_2of2, 2, sizeof(cl_mem), (void *)&dev_redscratch);
    clSetKernelArg(reduce_sum_2of2, 3, local_work_size*sizeof(cl_double), NULL);
}

double total_sum;

iret = clEnqueueReadBuffer(command_queue, dev_total_sum, CL_TRUE, 0, 1*sizeof(cl_double), &total_sum, 0, NULL, NULL);

printf("Result -- total sum %lf \n", total_sum);

clReleaseMemObject(dev_x);
clReleaseMemObject(dev_redscratch);
clReleaseMemObject(dev_total_sum);

clReleaseKernel(reduce_sum_1of2);
clReleaseKernel(reduce_sum_2of2);
clReleaseCommandQueue(command_queue);
clReleaseContext(context);
clReleaseProgram(program);
```
### 12.4 SYCL: An experimental C++ implementation goes mainstream
SYCL起始于2014年，作为OpenCL上的一个实验性的C++实现，开发者创建SYCL的目标是使SYCL成为C++语言的更自然扩展(more natural extension)，而不是像OpenCL给C语言带来的附加感(add-on feeling)，SYCL正在被开发成一个利用了OpenCL的可移植性和效率(portability and efficiency)的跨平台抽象层(cross-platform abstraction level)
当Intel宣布选择SYCL作为他们用于的能源部Aurora HPC系统的主要语言路径之一(major language pathways)时，SYCL就不再被认为是实验性的语言，Aurora系统将使用正在开发的新的Intel独立GPUs，Intel已经提出了一些对SYCL标准的补充，他们在其oneAPI开放编程系统(open programming system)中的Data Parallel C++ (DPCPP)编译器中对该补充标准进行了原型设计

我们将使用Intel的DPCPP版本的SYCL

Listing 12.21 Stream triad example for DPCPP version SYCL
`DPCPP/StreamTriad/StreamTriad.cc`
```cpp
#include <chrono>
#include "CL/sycl.hpp" // includes the SYCL header file

namespace Sycl = cl::sycl; // uses the SYCL namespace
using namespace std;

int main(int argc, char *argv[]) {
    chrono::high_resolution_clock::time_point t1, t2;

    size_t nsize = 10000;
    cout << "StreamTriad with "<< nsize << " elements" << endl;

    // host data
    vector<double> a(nsize, 1.0);
    vector<double> b(nsize, 2.0);
    vector<double> c(nsize, -1.0);

    t1 = chrono::high_resolution_clock::now();

    Sycl::queue Queue(Sycl::cpu_selector{}); // set up the device for CPU

    const double scalar = 3.0;

    // allocates the device buffer and sets to the host buffer
    Sycl::buffer<double,1> dev_a {a.data(), Sycl::range<1>(a.size())};
    Sycl::buffer<double,1> dev_b {b.data(), Sycl::range<1>(b.size())};
    Sycl::buffer<double,1> dev_c {c.data(), Sycl::range<1>(c.size())};

    // lambda for queue
    Queue.submit([&](Sycl::handler&CommandGroup) {
        // gets access to device arrays
        auto a = dev_a.get_access<Sycl::access::mode::read>(CommandGroup);
        auto b = dev_b.get_access<Sycl::access::mode::read>(CommandGroup);
        auto c = dev_c.get_access<Sycl::access::mode::write>(CommandGroup);

        // lambda for parallel for kernel
        CommandGroup.parallel_for<class StreamTraid>(Sycl::range<1>{nsize}, [=] (Sycl::id<1> it) {
            c[it] = a[it] + scalar * b[it];
        });
    });
    Queue.wait(); // waits for completion

    t2 = chrono::high_resolution_clock::now();

    double time1 = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
    cout << "Runtime is " << time1*1000.0 << " msecs" << endl;
}
```
第一个 `Sycl` 函数选择了一个设备，并创建了一个在该设备上工作的队列，我们请求的设备是CPU，对于具有统一内存(unified memory)的GPUs，该代码也可以工作
```cpp
Sycl::queue Queue(sycl::cpu_selector{});
```
要使用没有统一内存的GPUs，我们需要添加将数据从一个内存空间到另一个内存空间的显式拷贝(explicit copies)，默认选择器(default selector)优先寻找GPU，但会回退到CPU，如果我们只想选择GPU或CPU，我们也可以指定其他选择器，例如：
```cpp
Sycl::queue Queue(sycl::default_selector{}); // 使用默认设备
Sycl::queue Queue(sycl::gpu_selector{}); // 寻找GPU设备
Sycl::queue Queue(sycl::cpu_selector{}); // 寻找CPU设备
Sycl::queue Queue(sycl::host_selector{}); // 在主机(CPU)上运行
```
最后一个选项意味着它将像没有SYCL或OpenCL代码一样在主机上运行

设备和队列的设置比我们在OpenCL中做的要简单得多，首先我们需要使用SYCL缓冲区设置设备缓冲区(device buffer)：
```cpp
Sycl::buffer<double,1> dev_a { a.data(), Sycl::range<1>(a.size()) };
```
缓冲区的第一个参数是数据类型，第二个是数据的维度，然后我们给它变量名`dev_a`，变量的第一个参数是用来初始化设备数组的主机数据数组，第二个是使用的索引集(indec set)，在这种情况下，我们指定了一个从0到变量 `a` 大小的1维范围(1D range)

在第29行，我们遇到了第一个lambda表达式，用于为队列创建命令组处理程序(command group handler)：
```cpp
Queue.submit([&](Sycl::handler& CommandGroup)
```
我们在10.2.1节介绍了lambda，lambda捕获子句(capture clause) `[&]` 指定了按引用捕获(by reference)在例程中使用的外部变量(outside variables)，对于这个lambda表达式，捕获通过引用获取 `nsize` 、`scalar` 、`dev_a` 、`dev_b` 和 `dev_c` 以在lambda中使用
我们可以使用 `[&]` 捕获所有变量 ，也可以用下面的形式，指定将要捕获的每个变量，良好的编程实践会倾向于后者，但列表可能会很长：
```cpp
Queue.submit([&nsize, &scalar, &dev_a, &dev_b, &dev_c](Sycl::handler& CommandGroup)
```
在lambda的主体中，我们可以访问设备数组并为设备例程中的使用重命名它们，这相当于命令组处理程序(command group handler)的参数列表

然后我们为命令组创建第一个任务，即一个 `parallel_for`，`parallel_for`也用一个lambda定义：
```cpp
CommandGroup.parallel_for<class StreamTriad>(Sycl::range<1>{nsize},[=](Sycl::id<1> it)
```
lambda的名称是 `StreamTriad` ，然后我们告诉它我们将在一个从0到nsize的1D范围内操作，捕获子句 `[=]` 按值捕获变量 `a` 、`b` 和 `c` ，决定是按引用还是按值捕获是棘手的，但如果代码被推送到GPU，原始引用可能不在作用域内(out of scope)，不再有效，我们最后创建一个1D索引变量(index variable) `it` ，用于遍历范围(iterate over the range)
### 12.5 Higher-level languages for performance portability
到目前为止，我们已经看到CPU和GPU内核之间的差异并不是那么大，那么为什么不使用C++的多态性和模板来生成它们呢？这正是由能源部研究实验室开发的一些库所做的，这些项目启动的目的是为了便于将他们的许多代码移植到新的硬件架构上
Kokkos系统是由桑迪亚国家实验室创建的，并已经获得了广泛的关注，劳伦斯利弗莫尔国家实验室有一个名为RAJA的类似项目，这两个项目都已经成功地实现了单一源代码、多平台能力的目标(a single source, multiplatform capability)

这两种语言在很多方面与12.4节的SYCL语言有相似之处，实际上，它们在努力实现性能可移植性(performance portability)的过程中，从彼此那里借鉴了概念，它们各自提供了库，这些库是在较低级别的并行编程语言之上的(on top of)相当轻量级的层
#### 12.5.1 Kokkos: A performance portability ecosystem
Kokkos是一个为像OpenMP和CUDA这样的语言精心设计的抽象层(abstraction layer)，自2011年以来一直在开发中，
Kokkos具有以下命名的执行空间(execution spaces)，这些在Kokkos构建中通过相应的CMake标志(或使用Spack构建的选项)启用

| Kokkos execution spaces | CMake/Spack-enabled flags                  |
|:----------------------- |:------------------------------------------ |
| `Kokkos::Serial`        | `-DKokkos_ENABLE_SERIAL=On`(default is on) |
| `Kokkos::Threads`       | `-DKokkos_ENABLE_PTHREAD=On`               |
| `Kokkos::OpenMP`        | `-DKokkos_ENABLE_OpenMP=On`                |
| `Kokkos::Cuda`          | `-Dkokkos_ENABLE_CUDA=On`                  |
| `Kokkos::HPX`           | `-DKokkos_ENABLE_HPX=On`                   |
| `Kokkos::ROCm`          | `-Dkokkos_ENABLE_ROCm=On`                  |

对Kokkos构建添加CUDA选项就可以生成在NVIDIA GPUs上运行的代码，Kokkos可以处理许多其他的平台，有一些还在开发中

Listing 12.23是Kokkos的流三元运算例子，它和SYCL类似的是它也使用了C++ lambda来为CPU或GPU封装函数，Kokkos还支持使用仿函数(functors)来实现这一机制，但实践中使用lambda表达式更为简洁

Listing 12.23 Kokkos stream triad example
`Kokkos/StreamTriad/StreamTriad.cc`
```c++
#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv); {
        Kokkos::Timer timer;
        double time1;

        double scalar = 3.0;
        size_t nsize = 1000000;
        // declare arrays with Kokkos::View
        Kokkos::View<double *> a("a", nsize);
        Kokkos::View<double *> b("b", nsize);
        Kokkos::View<double *> c("c", nsize);

        cout << "StraemTriad with " << nsize <<" elements" << endl;
        
        Kokkos::parallel_for(nsize, KOKKOS_LAMBDA(int i) {
            a[i] = 1.0;
        });
        Kokkos::parallel_for(nsize, KOKKOS_LAMBDA(int i) {
            b[i] = 2.0;
        });

        timer.reset();

        Kokkos::parallel_for(nsize, KOKKOS_LAMBDA(const int i) {
            c[i] = a[i] + scalar * b[i];
        });

        time1 = timer.seconds();

        icount = 0;
        for(int i= 0; i< nsize && icount < 10; i++) {
            if (c[i] != 1.0 + 3.0 * 2) {
                cout << "Error with result c[" << i << "]=" << c[i] << endl; 
                icount++;
            }
        }
        if (icount == 0) {
            cout << "Program completed without error." << endl;             cout << "Runtime is " << time1*1000.0 << " msecs " << endl;
        }
    }
    Kokkos::finalize();
    return 0;
}
```
Kokkos程序始于 `Kokkos::initialize` 和 `Kokkos::finalize`这两个命令，这些命令启动了(start up)执行空间(execution space)所需的事物，例如线程(threads)
Kokkos的独特之处在于它将灵活的多维数组分配(multi-dimensional array allocations)封装为数据视图(data views)，这些视图可以根据目标架构进行切换(be switched depending on the target architecture)，换句话说，我们可以为CPU和GPU使用不同的数据顺序(data order)，我们在第14至16行使用了 `Kokkos::View` ，尽管此处仅用于一维数组，但 `Kokkos::View` 真正的价值体现在多维数组上，`Kokkos::View` 的一般语法是：
```cpp
View < double *** , Layout , MemorySpace > name (...);
```
内存空间(memory space)是模板的一个选项，各个执行空间有一个适合的默认值(a default appropriate for the execution space)，可用的内存空间有：
- `HostSpace`
- `CudaSpace`
- `CudaUVMSpace`

布局(layout)可以指定，且各个内存空间也有适用的默认值：
- 对于 `LayoutLeft`，最左边的索引为1(the leftmost index is stride 1)(`CudaSpace` 的默认值)
- 对于 `LayoutRight` ，最右边的索引为1(the rightmost index is stride 1)(`HostSpace` 的默认值)

内核(kernels)通过lambda语法，并需要在三种数据并行模式中指定其一：
- `parallel_for`
- `parallel_reduce`
- `parallel_scan`

在Listing 12.23的第20、23和29行，我们使用了 `parallel_for` 模式，`KOKKOS_LAMBDA` 宏替换了 `[=]` 或 `[&]` 捕获语法，Kokkos为我们自动处理了捕获，同时形式更易读
#### 12.5.2 RAJA for a more adaptable performance portability layer
RAJA性能可移植层(performance portability layer)的目标是在对现有的劳伦斯利弗莫尔国家实验室代码造成最小干扰的情况下实现可移植性，它比其他类似的系统在许多方面更简单、更容易采用，RAJA可以在以下支持下构建：
- `-DENABLE_OPENMP=On`(默认开启)
- `-DENABLE_TARGET_OPENMP=On` (默认关闭)
- `-DENABLE_CUDA=On` (默认关闭)
- `-DENABLE_TBB=On` (默认关闭)

RAJA还对CMake有很好的支持

RAJA版本的流三元组示例只需要进行一些简单的修改，如下所示的列表所示，RAJA还大量利用lambda表达式来提供它们对CPU和GPU的可移植性
Listing 12.25 Raja stream triad example
`Raja/StreamTriad/SteramTriad.cc`
```cpp
#include <chrono>
#include "RAJA/RAJA.hpp"

using namespace std;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[])) {
    chrono::high_resolution_clock::time_point t1, t2;
    cout<< "Running Raja Stream Triad\n";

    const int nsize = 1000000;

    // Allocate and initialize vector data
    double scalar = 3.0;
    double* a = new double[nsize];
    double* b = new double[nsize];
    double* c = new double[nsize];

    for (int i = 0; i< nsize; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    t1 = chrono::high_resolution_clock::now();

    RAJA::forall<RAJA::omp_parallel_for_exec>(
        RAJA::RangeSegment(0, nsize), [=](int i) {
            c[i] = a[i] + scalar * b[i];
        }
    );
    t2 = chrono::high_resolution_clock::now();

<...error checking ...>
    double time1 = chrono::duration_cast<chrono::duration<double>> (t2-t1).count();
    cout << "Runtime is " << time1*1000.0 << " msecs " << endl;
}
```
RAJA所需的更改仅包括了在第2行 `#include` RAJA头文件，以及将计算循环(computation loop)更改为 `Raja::forall`，可以看到，RAJA开发者仅需要一个低的门槛(low-entry threshold)就可以获得性能可移植性
要运行RAJA测试，我们包括了一个脚本，该脚本构建并安装RAJA，并使用RAJA构建流三元组代码并运行它，如下Listing所示：
Listing12.26 Integrated build and run script for Raja stream triad
`Raja/StreamTriad/Setup_Raja.sh`
```c
# !/bin/sh
export INSTALL_DIR=`pwd`/build/Raja
export Raja_DIR=${INSTALL_DIR}/share/raja/cmake // Raja_DIR points to Raja CMake tool

mkdir -p build/Raja_tmp && cd build/Raja_tmp
cmake ../../Raja_build -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j 8 install && cd .. && rm -rf Raja_tmp

cmake .. && make && ./StreamTriad // builds the stream triad code and runs it
```
## 13 GPU profiling and tools


# High performance computing
## 14 Affinity: Truce with the kernel
