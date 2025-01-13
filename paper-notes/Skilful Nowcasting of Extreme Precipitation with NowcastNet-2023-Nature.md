# Abstract
Extreme precipitation is a considerable contributor to meteorological disasters and there is a great need to mitigate its socioeconomic effects through skilful nowcasting that has high resolution, long lead times and local details. Current methods are subject to blur, dissipation, intensity or location errors, with physics-based numerical methods struggling to capture pivotal chaotic dynamics such as convective initiation and data-driven learning methods failing to obey intrinsic physical laws such as advective conservation. We present NowcastNet, a nonlinear nowcasting model for extreme precipitation that unifies physical-evolution schemes and conditional-learning methods into a neural-network framework with end-to-end forecast error optimization. On the basis of radar observations from the USA and China, our model produces physically plausible precipitation nowcasts with sharp multiscale patterns over regions of $2{,}048\,\mathsf{k m}\times2{,}048\,\mathsf{k m}$ and with lead times of up to 3 h. In a systematic evaluation by 62 professional meteorologists from across China, our model ranks first in $71\%$ of cases against the leading methods. NowcastNet provides skilful forecasts at light-to-heavy rain rates, particularly for extreme-precipitation events accompanied by advective or convective processes that were previously considered intractable. 
>  极端降水容易导致气候灾害，我们需要对其进行高精度、长提前期和具有局部细节的精确临近预报
>  目前的方法容易受到模糊、消散、密度或位置错误的影响，基于物理的数值方法难以捕获关键的混沌动态，例如对流启动，数据驱动的学习方法难以遵守固有的物理定律，例如平流守恒
>  本文提出 NowcastNet，一个用于预测极端降水的非线性临近预报模型
>  NowcastNet 将物理演化方案和条件学习方法统一到一个神经网络框架中，可以进行端到端的预测误差优化
>  NowcastNet 基于中国和美国的雷达观测数据学习
>  NowcastNet 可以生成物理上合理的清晰多尺度降水预报，预报覆盖 2048km $\times$ 2048 km 的区域，具有 3 小时的预测提前期
>  在中国各地的 62 位气象专家的系统评估中，NowcastNet 在 71% 的案例中排名第一
>  NowcastNet 可以提供从轻度到重度雨量的精确预报，特别是伴随平流或对流过程的极端降水事件，在 NowcastNet 之前，这类极端降水事件被认为是不可处理的

> [!info] 平流和对流
> 伴随有平流 (advection) 或对流 (convection) 过程的极端降水事件通常涉及到大气中水分和热量的快速移动，以及强烈的垂直空气运动，它们可以导致短时间内降下大量的雨水，造成洪水、山体滑坡等灾害
> 
> Advection (平流) 指风将水汽从一个地方携带到另一个地方的过程，在气象学中，这通常指的是**水平**方向上的水汽传输
> 当湿润空气被风吹向山脉或其他地形特征时，它被迫上升，在冷却过程中凝结形成云和降水，如果这种过程非常强烈，并且涉及大量水汽，就可能引发极端降水事件
> 
> Convection (对流) 指对流层内由于温度差异引起的热力驱动的**垂直**空气流动
> 对流中，暖湿空气/热空气因为比周围冷空气轻而上升，在上升过程中冷却并凝结成云滴，最终可能发展成为雷暴云或积雨云，进而产生降雨
> 强对流活动可以迅速形成并带来猛烈的短时暴雨

# Introduction
Nowcasting is defined by the World Meteorological Organization (WMO) as forecasting that yields local details across the mesoscale and small scale, over a period from the present up to 6 h ahead and which provides a detailed description of the present weather. Nowcasting is crucial in risk prevention and crisis management of extreme precipitation, commonly defined as the 95th percentile of the cumulative frequency distribution of daily precipitation. According to a recent report from the $\mathsf{W M O}$ , over the past 50 years, more than $34\%$ of all recorded disasters, $22\%$ of related deaths (1.01 million) and $57\%$ of related economic losses $\left({\tt U S \$2.84}\right.$ trillion) were consequences of extreme-precipitation events. 
>  临近预报 (Nowcasting) 被世界气象组织定义为在提供当前详细的天气描述下，可以提供中尺度和小尺度的局部细节的气象预测，时间范围是从当前时刻到未来 6 小时以内
>  极端降水一般定义为日降水量累计频率分布的第 95 百分位数 (也就是极端降水的降水量应该是日降水量统计值的最高的 5%，超过了 95% 的统计值)，临近预报可以用于预防极端降水带来的危害

Weather radar echoes provide cloud observations at sub-2-km spatial resolution and up to 5-min temporal resolution, which are ideal for precipitation nowcasting. The natural option for exploiting these data is numerical weather prediction, which produces precipitation forecasts based on solving coupled primitive equations of the atmosphere. However, these methods, even when implemented on a supercomputing platform, restrict the numerical weather prediction forecast update cycles to hours and the spatial resolution to the mesoscale, whereas extreme weather processes typically exhibit lifetimes of tens of minutes and individual features at the convective scale. 
>  天气雷达回波提供了亚 2km 空间尺度和 5min 时间尺度的云层观测，适用于临近降水预报
>  数值天气预报基于求解耦合的大气基本方程来利用这些数据进行降水预测
>  但即便在超算平台上，数值天气预测的更新周期也需要以小时计，同时空间尺度限制在中尺度，而极端天气的声明周期一般仅有几十分钟，同时其特征处于对流尺度

> [! info] 雷达回波
> 雷达回波（Radar Echo）是指由雷达发射的电磁波遇到目标物体（如雨滴、雪片、冰雹、云中的水汽粒子等）后反射回来的信号
> 
> 当雷达发出的电磁波束遇到降水粒子时，一部分能量会被反射回到雷达天线。雷达接收到这些反射回来的能量后，可以根据反射波的时间延迟计算出目标的距离，根据多普勒频移确定目标相对于雷达的速度，并根据反射强度估计降水的密度或强度。
> 
> 雷达回波强度通常以分贝单位（dBZ）表示，这反映了回波信号的功率水平，而不同的 dBZ 值对应于不同的降水强度。更高的 dBZ 值意味着更强的反射，这通常与更密集或更大的降水粒子相关联。

Alternative methods such as DARTS and pySTEPS are based on an advection scheme inspired solely by the continuity equation. These methods solve separately for the future states of the motion fields and the intensity residuals from composite radar observations and iteratively advect past radar fields to predict future fields. The advection scheme partially respects the physical conservation laws of precipitation evolution and is able to provide skilful extrapolations within 1 h, but it degrades quickly beyond that horizon, incurring high location error and losing small convective features. These errors accumulate in the auto regressive advection processes in uncontrolled ways, owing to existing advection implementations failing to incorporate nonlinear evolution simulations and end-to-end forecast error optimization. 
>  DARTS 和 pySTEPS 则基于仅由连续性方程启发的平流方案，这些方法基于组合雷达观测 (即多个雷达观测的组合)，分别求解运动场的未来状态和强度残差，并迭代式 (用预测的风场) 平流过去的雷达回波场 (指降水区域) 以预测未来的场
>  (这里假设了降水系统会随着风向一起移动)
>  平流方案部分遵守了降水演变的物理守恒定律，能够提供 1h 内的准确的外推，但超过该时间范围后性能迅速下降，出现高的位置误差并失去小的对流特征，因为现存平流方案没有考虑非线性演化模拟和端到端的预测误差优化，这些误差在自回归平流过程不受控制地累积

Deep-learning methods have been applied in recent years to weather nowcasting. These methods exploit large corpora of composite radar observations to train neural-network models in an end-to-end fashion, dispensing with explicit reference to the physical laws behind precipitation processes. They have proved useful for low-intensity rainfall as measured by per-grid-cell metrics such as the Critical Success Index (CSI). 
>  深度学习方法利用大量的组合雷达观测端到端地训练神经网络模型，无需显式参考降水过程背后的物理定律
>  深度学习方法在低强度的降水预测下表现良好，尤其是在逐网格单元的关键成功指数度量上

> [! info] 网格单元和关键成功指数
> 在气象预报和遥感中，通常会将研究区域划分为多个网格单元（grid cells），每个网格单元代表一个特定的地理区域，预报模型或观测数据会在这些网格上进行评估。
> 
> 关键成功指数（CSI）是一种用于评估天气预报准确性的统计度量，尤其适用于二分类事件（如降水/无降水），CSI 的计算公式如下：
> 
> $\text{CSI} = \frac{\text{命中数}}{\text{命中数} + \text{虚警数} + \text{漏报数}}$
> 
> - 命中数（Hits, H）：正确预报了发生的事件（预报有雨且实际下雨、预报无雨且实际无雨）。
> - 虚警数（False Alarms, FA）：错误地预报了事件的发生（预报有雨但实际无雨）。
> - 漏报数（Misses, M）：事件发生但未被预报到（预报无雨但实际下雨）。
> 
> CSI 的取值范围从 0 到 1，其中 1 表示完美预报 (预测全部命中)，而 0 表示预报完全失败 (没有预测命中)。较高的 CSI 值意味着更好的预报性能。

A large step forward in this setting has been the deep generative model of radar (DGMR) approach developed by DeepMind and the UK Met Office. This approach generates spatiotemporally consistent predictions with a lead time of up to $90\,\mathtt{m i n}$ , simultaneously capturing chaotic convective details and accounting for ensemble forecast uncertainty. In an expert evaluation by more than 50 meteorologists from the UK Met Office, DGMR ranked first in $89\%$ of cases against competing methods, including the advection-based method pySTEPS. 
>  DeepMind 和英国气象局开发的基于深度生成模型的雷达方法 (DGMR) 可以生成 90min 提前期的时空一致性预测，同时捕获混沌的对流细节，并考虑集合预报的不确定性 (即不仅仅给出一个最有可能的预测，而是给出一个分布，因此考虑到了不确定性，同时预测的概率值也提供了置信度)

Still, for extreme precipitation, DGMR may produce nowcasts with unnatural motion and intensity, high location error and large cloud dissipation at increasing lead times. These problems reflect the fact that radar echoes are only partial observations of the atmospheric system. Deep-learning models based purely on radar data analysis are hampered in their ability to capture the fuller range of physical phenomena underlying precipitation. We believe that physical knowledge of aspects of precipitation processes, including the conservation law of cloud transport and the log-normal distribution of rain rate, need to be embedded into data-driven models to make skilful nowcasting of extreme precipitation possible. 
>  但对于极端降水，DGMR 的预测会带有不自然的运动和强度 (不符合物理规律的降水移动模式和强度变化)、高的位置误差，以及随着提前期的增加，其预测的云层会快速消散 (模型没有正确模拟云层的持续时间和演变过程)
>  这些问题说明雷达回波只是对大气系统的部分观测，完全依赖雷达数据的深度学习模型难以捕获降水背后的完整物理现象信息
>  因此将降水过程背后的物理知识，包括云运输的守恒定律以及降雨率的对数正态分布需要被嵌入数据驱动的模型中

We present NowcastNet, a unified nowcasting model for extreme precipitation based on composite radar observations. It combines deep-learning methods with physical first principles, by means of a neural-network framework that implements neural evolution operators for modelling nonlinear processes and a physics-conditional mechanism for minimizing forecast error. This framework enables seamless integration of advective conservation into a learning model, successfully predicting long-lived mesoscale patterns and capturing short-lived convective details with lead times of up to 3 h. As we will show on the USA and China events corpora, the forecasts made by NowcastNet are judged by expert meteorologists to be more accurate and instructive than pySTEPS, DGMR or other deep-learning systems. 
>  NowcastNet 是基于组合雷达观测的统一极端降水临近预测模型，它通过一个神经网络框架结合了深度学习方法和物理第一性原理
>  NowcastNet 的神经网络框架实现了神经演化算子用以建模非线性过程，同时使用物理条件机制以最小化预测误差
>  该框架将平流守恒 (平流过程中，质量、能量、动量等物理量的守恒) 无缝集成到模型中，成功预测长期的中尺度模式并捕获短期的对流细节，将提前期提高到 3h

# NowcastNet 
Skilful nowcasting requires making use of both physical first principles and statistical-learning methods. NowcastNet provides such a unification using a neural-network framework, allowing end-to-end forecast error optimization. 

Our nowcasting algorithm (Fig. 1a) is a physics-conditional deep generative model that exploits radar-based estimates of surface precipitation to predict future radar fields $\widehat{\mathbf{x}}_{1:T}$ given past radar fields $\mathbf x_{-T_0:0}$ . The model includes a stochastic generative network parameterized by $\theta$ and a deterministic evolution network parameterized by $\phi$ . 
>  NowcastNet 的临近预测算法是基于物理条件的深度生成模型，它利用雷达估算的地表降水数据，给定过去的雷达场 $\mathbf x_{-T_0:0}$，预测未来的雷达场  $\widehat {\mathbf x}_{1:T}$
>  模型包括了随机性生成网络，参数为 $\theta$ 和确定性演化网络，参数为 $\phi$

The nowcasting procedure is based on physics-conditional generation from latent random vectors $\mathbf z$ , described by 

$$
P(\widehat{\mathbf{x}}_{1:T}|\mathbf{x}_{-T_{0}:0},\phi;\theta)\!=\!\int P(\widehat{\mathbf{x}}_{1:T}|\mathbf{x}_{-T_{0}:0},\phi(\mathbf{x}_{-T_{0}:0}),\mathbf{z};\theta)P(\mathbf{z})\mathrm{d}\mathbf{z}.\tag{1}
$$ 
The integration over latent Gaussian vectors $\mathbf z$ enables ensemble forecast with predictions skilfully capturing the pivotal chaotic dynamics.

>  临近预报过程基于从潜在随机向量 $\mathbf z$ 的物理条件生成
>  在潜在高斯向量 $\mathbf z$ 上的积分使得模型可以进行集合预报，以捕获关键的混沌动力学特征
>  (潜在高斯向量 $\mathbf z$ 从潜在空间中随机抽取得到，它代表了系统的可能的初始条件或系统状态，基于 $\mathbf z$ 积分为模型引入了不确定性
>  天气系统通常是高度非线性且混沌的，不同的 $\mathbf z$ 就会导致不同的演化路径，模型在数据中学习到基于各个 $\mathbf z$ 时，系统应该如何演化，从而模型具有了能够捕获系统一定的混沌演化的特性)

>  推导
>  随机性生成网络 (由 $\theta$ 参数化) 定义了给定确定性演化网络参数 $\phi$ 和过去雷达场 $\hat {\mathbf x}_{1:T}$ 下，未来雷达场 $\widehat {\mathbf x}_{-T_0:0}$ 的条件分布

$$
\begin{align}
P(\widehat {\mathbf x}_{1:T}\mid \mathbf x_{-T_0:0}, \phi ;\theta)
&=\int P(\hat {\mathbf x}_{1:T},\mathbf z\mid \mathbf x_{-T_0:0},\phi)d\mathbf z\\
&=\int P(\hat {\mathbf x}_{1:T}\mid \mathbf x_{-T_0:0},\phi,\mathbf z)P(\mathbf z\mid \mathbf x_{-T_0:0},\phi)d\mathbf z\\
&=\int P(\hat {\mathbf x}_{1:T}\mid \mathbf x_{-T_0:0},\phi,\mathbf z)P(\mathbf z)d\mathbf z\\
\end{align}
$$

>  这里利用了 $\mathbf z$ 独立于 $\phi, \mathbf x_{-T_0:0}$ 的性质

Although our work fits in a nascent thread of research on physics-informed neural networks, there are many challenges in the precipitation domain that are not readily accommodated by existing research. Most notably, the multiscale nature of atmospheric physics introduces emergent dependencies among several spatiotemporal scales and imposes inherent limits on atmospheric predictability. In particular, the convective processes are subject to chaotic error growth from uncertain initial conditions, limiting advection schemes to a spatial scale of $20\,\mathrm{km}$ and a lead time of 1 h (ref. 18). Naive combinations of neural networks and physical principles entangle the multiscale variability and corrupt the mesoscale and convective-scale patterns, creating undesirable confounding and uncontrolled errors. 
>  该工作属于物理学指导的神经网络这一领域
>  现存研究尚未解决降水领域的一些挑战，大气物理具有多尺度特性，故大气现象中，存在多个时空尺度之间的依赖关系，因此大气系统的可预测性较低
>  例如，对流过程容易受到不确定初始条件引起的混沌误差增长的影响 (也就是微小的初始条件差异会随着对流过程被不断放大，导致预测出现显著偏差)，故平流方案的空间尺度被限制在了 20km，提前时间限制为 1h
>  朴素地结合神经网络和物理定律会纠缠多尺度变异性 (不同尺度的信息被错误地混合在一起)，破坏中尺度和对流尺度模式，产生更多误差

> [! info] 大气现象的多尺度
> 大气现象涵盖了广泛的时空尺度，这些尺度互相关联和影响
> - 微尺度：如湍流和云滴碰撞过程，通常在米级或更小范围内发生。
> - 中尺度：如雷暴和局地降水系统，覆盖几十到几百公里。
> - 大尺度：如高压和低压系统、锋面等，涉及数百到数千公里的大气环流。

![[pics/NowcastNet-Fig1a.png]]

We address the multiscale problem by a new conditioning mechanism that the data-driven generative network $\theta$ boosts over the advection-based evolution network $\phi$ (Fig. 1a). The evolution network imposes compliance with the physics of precipitation, yielding physically plausible predictions $\mathbf x_{1:T}''$ for advective features at a scale of $20\,\mathrm{km}$ . 
>  我们通过新的条件机制解决大气预测的多尺度问题，NowcastNet 中，数据驱动的生成式网络 $\theta$ 在基于平流的演化网络 $\phi$ 上进行增强 (可以看到 eq(1) 中，$\phi$ 也是条件分布的条件之一)
>  演化网络确保符合降水物理规律，从而为 20km 尺度上的平流特征生成合理的预测 $\mathbf x_{1: T}''$ 

The nowcast decoder takes the nowcast encoder representations of past radar fields $\mathbf x_{-T_0:0}$ , along with the evolution network predictions $\mathbf x_{1:T}^{\prime\prime},$ and generates fine-grained predictions $\widehat{\mathbf{x}}_{1:T}$ from latent Gaussian vectors $\mathbf z$ that can capture convective features at a 1–2-km scale. Such a scale disentanglement mitigates error propagating upscale or downscale in the multiscale prediction framework. 
>  演化网络生成平流特征预测 $\mathbf x''_{1:T}$ 后，nowcast encoder 接受过去的雷达场 $\mathbf x_{-T_0:0}$ 以及演化网络的输出 $\mathbf x_{1: T}''$，输出对过去的雷达场 $\mathbf x_{-T_0:0}$ 的编码表示
>  nowcast decoder 接受 nowcast encoder 对 $\mathbf x_{-T_0:0}$ 的编码表示以及演化网络的输出 $\mathbf x_{1: T}''$，从潜在高斯向量 $\mathbf z$ 生成细粒度的未来雷达场预测 $\hat {\mathbf x}_{1: T}$，该预测可以捕获 1-2km 尺度上的对流特征
>  这样的尺度分离 (演化网络生成 20km 尺度的平流特征预测，生成网络生成 1-2km 尺度的对流特征预测) 缓解了多尺度预测框架中误差随着尺度向上传播或向下传播的过程

We use the spatially adaptive normalization technique to enable an adaptive evolution conditioning mechanism. In each forward pass, the mean and variance of every-decoder-layer activations are replaced by the spatially corresponding statistics computed from the evolution network predictions $\mathbf x_{1:T}^{\prime\prime}.$ As a result, NowcastNet adaptively combines mesoscale patterns governed by physical laws and convective-scale details revealed by radar observations, yielding skilful multiscale predictions with up to a 3-h lead time. 
>  我们使用空间适应性规范化技术实现适应性演化条件机制 (指生成网络如何条件于演化网络)
>  前向传播过程中，(Nowcast decoder 中) 每个解码层激活的均值和方差被根据演化网络输出 $\mathbf x_{1: T}''$ 计算得到的相应空间位置的统计信息替换 
>  (将演化网络输出的符合物理规律的平流层特征根据空间位置嵌入生成器网络)
>  因此，NowcastNet 可以适应性地结合符合物理规律的中尺度模式 (来自于演化网络的输出 $\mathbf x_{1: T}''$) 和由雷达观测揭示的对流尺度细节 (来自于输入雷达观测 $\mathbf x_{-T_0:0}$，提供具有 3h 提前时间的多尺度预测

Learning is framed as the training of a conditional generative adversarial network, given the pre-trained evolution network that encodes physical knowledge. A temporal discriminator is built on the nowcast decoder, taking as input the pyramid of features in several time windows and outputting whether the input is likely to be real radar or a fake field. The nowcast encoder and decoder are trained with an adversarial loss to generate convective details present in the radar observations but left out by the advection-based evolution network. 
>  NowcastNet 的学习被构建为训练条件 GAN 的框架
>  给定预训练好的，编码了物理知识的演化网络，我们在 nowcast decoder 上构建一个时间判别器 (生成网络作为条件 GAN 框架中的生成器)，它接受多个时间窗口的特征金字塔作为输入 (也就是 nowcast decoder 的输出)，判断该输入是真实的雷达场或假的雷达场
>  nowcast encoder 和 decoder 就基于对抗损失训练，训练目标就是生成能骗过判别器的输出，最后训练好的 nowcast encoder 和 decoder 最后就能够生成存在于雷达观测中但被基于平流的演化网络忽视的对流细节

Also, the generated nowcasts need to be spatially consistent with the radar observations. This is achieved by the pool regularization, which enforces consistency between spatial-pooled ensemble nowcasts and spatial-pooled observations. The pooling-level consistency is more tolerant of the spatial chaos in real fields and is capable of resolving the conflict between the generative network and the evolution network. 
>  另外，生成的临近预报需要和雷达观测在空间上一致 (也就是在空间分布上需要尽可能接近真实的雷达观测效果)，我们通过池化正则化实现 (池化用于捕获较大区域内的总体特征/主要模式，而忽略局部细节)
>  池化正则化强制空间池化后的集成临近预测和空间池化后的观测是一致的
>  池化级别的一致性更能容忍实际场中的空间混沌 (因为不要求局部细节的一致性)，并且可以解决生成网络和演化网络之间的冲突 (正则化约束了生成网络的生成在大尺度上需要和真实观测的空间布局一致，防止生成网络生成过于复杂，不符合真实规律的模式，这一约束促进了生成网络对演化网络提供的条件的重视，因为演化网络提供的条件是符合大尺度的物理规律的，重视演化网络提供的条件可以让生成网络有更高概率生成符合真实规律的模式)

# Evolution network 
NowcastNet enables multiscale nowcasting by conditioning the data-driven (stochastic) generative network $\theta$ on the advection-based (deterministic) evolution network $\phi$ . 

In atmospheric physics, the continuity equation is the fundamental conservation law governing the cloud transport and precipitation evolution. It has inspired a series of operational advection schemes, which model the precipitation evolution as a composition of advection by motion fields and addition by intensity residuals. 
>  大气物理学中，连续性方程是支配云层运输和降水演化的基本守恒定律
>  连续性方程启发了一系列操作性平流方案，它们将降水演化建模为由运动场引起的平流和强度残差引起的 (降水) 增加的组合
>  (强度残差指实际观察到的降水量和平流模型预测的降水量之间的差异，这部分差异不能用平流模型解释，可能是由局部的气象过程，例如对流、凝结与蒸发，以及地形效应、城市热岛效应、植被覆盖变化等引起的)

However, previous implementations of advection schemes, for example, pySTEPS, fall short in three respects: (1) their advection operation is not differentiable and thus cannot be embedded easily into an end-to-end neural framework for gradient-based optimization; (2) their steady-state assumption limits the implementations to linear regimes, failing to provide the nonlinear modelling capability crucial for precipitation simulations; and (3) their auto regressive nature prevents direct optimization of the forecast errors and errors arising from the estimation of the initial states, motion fields and intensity residuals will accumulate in an uncontrolled manner in the Lagrangian persistence model. 
>  之前的平流方案在三个方面存在不足：
>  1. 其平流操作不可微，不能直接嵌入到端到端的神经网络中
>  2. 其稳态假设 (假设系统处于稳态或近似线性状态) 将实现限制在线性区域，不能进行非线性建模
>  3. 其自回归性质阻止了直接优化预测误差，同时由初始状态、运动场和强度残差估计产生的误差将在 Lagrangian 持久模型中累积
>  (自回归即基于之前时间步的预测进行当前时间步的预测)

> [! info] 拉格朗日持久模型
> 拉格朗日持久模型（LPM, Lagrangian Persistence Model）基于拉格朗日观点，即假设一个质点（或观测点）随时间移动时，其属性（如降水量、温度等）保持不变 
> 
> 假设 $P (x, y, t)$ 表示在位置 $(x, y)$ 和时间 $t$ 的降水量，则拉格朗日持久模型可以表示为 $P (x + u \Delta t, y + v \Delta t, t + \Delta t) = P (x, y, t)$，其中 $u$ 和 $v$  分别是 $x$ 和 $y$ 方向的风速分量，$\Delta t$ 是时间步长。
> 这个公式表明，在时间 $t + \Delta t$，新的位置 $(x + u \Delta t, y + v \Delta t)$ 上的降水量等于当前时间 $t$ 中位置 $(x, y)$ 的降水量。
> 
> **优点**
> 拉格朗日持久模型相对简单，易于实现，适合实时预报，对于快速变化的天气现象（如雷暴、阵雨等），该模型能够捕捉到这些现象的主要特征，特别是在雷达观测数据丰富的区域。因此在短期内（如 0-2 小时）通常能提供较为准确的预报结果。
> 
> **缺点**
> 1. 拉格朗日持久模型属于自回归模型，模型中每个时间步的预测依赖于之前的预测结果，故误差都会在后续时间步中不断累积，这种累积是不受控制的，可能导致长期预测的准确性大幅下降。自回归模型缺乏一个有效的全局反馈机制来纠正长期预测中的误差，没有一个统一的方式来评估并修正整个序列的误差。
> 2. 拉格朗日持久模型假设属性在短时间内保持不变，忽略了复杂的非线性过程，缺乏非线性建模能力。
> 
> **实例**
> 雷暴预测：在一个典型的雷暴事件中，拉格朗日持久模型可以通过追踪云团的移动来预测未来几分钟内的降水分布。然而，由于雷暴的非线性和复杂性，长时间预测可能会出现较大误差，尤其是在没有高分辨率雷达数据的情况下。

![[pics/NowcastNet-Fig1b.png]]

We address these desiderata with our evolution network (Fig. 1b), which implements the 2D continuity equation through neural evolution schemes. On the basis of a new differentiable neural evolution operator, it learns the motion fields, intensity residuals and precipitation fields simultaneously by neural networks; moreover, it directly optimizes the forecast error throughout the time horizon by gradient-based back propagation. 
>  上述三点不足都由演化网络解决
>  演化网络通过神经演化方案实现了二维连续性方程，演化网络基于可微的神经演化算子同时学习运动场、强度残差和降水场
>  演化网络通过反向传播直接在整个时间范围内优化预测误差

![[pics/NowcastNet-Fig1c.png]]

Our physics-informed evolution network is built on a new differentiable neural evolution operator (Fig. 1c). The evolution operator takes the current radar field ${\mathbf x}_{0}$ as input and predicts the future radar fields $\mathbf x_{1:T}.$ At each time step, the radar field predicted at the last time step ${\bf x}_{t-1}^{\prime\prime}$ is evolved by one step of advection with the motion field $\mathbf{v}_{t}$ to obtain ${\bf x}_{t}^{\prime}$ and the intensity residual ${\mathbf s}_{t}$ is then added to yield ${\bf{x}}_{t}^{\prime\prime}$ . The operator makes all motion fields and intensity residuals learnable end to end by gradient-based optimization, which is unattainable by existing advection schemes. 
>  演化网络基于神经演化算子构建
>  神经演化算子接受当前雷达场 $\mathbf x_0$ 作为输入，预测未来的雷达场 $\mathbf x_{1:T}$
>  在每个时间步，上一个时间步预测的雷达场 $\mathbf x_{t-1}''$ 通过运动场 $\mathbf v_t$ 进一步平流演化，得到中间结果 $\mathbf x_t'$，再添加强度残差 $\mathbf s_t$ 得到 $\mathbf x_t''$
>  该算子中所有的运动场和强度残差都可以基于端到端的梯度优化学习，这是现有平流方案做不到的

When learning the operator with back propagation, we stop the gradients between each time step to block information interference. This mitigates the numerical instability arising from the underdetermined nature of the overall system, which has discontinuous interpolations in the evolution operator. 
>  通过反向传播学习该算子时，为了防止信息干扰，每个时间步之间的梯度会被停止传递，以缓解由系统的不确定性质引起的数值不稳定性，这种不稳定性在演化算子中表现为不连续插值

> [!info] 梯度截断
> 神经演化算子中，时间步之间的演化写为 
> 
> $$\mathbf x''_t = f(\mathbf x''_{t-1}, \mathbf v_t, \mathbf s_t)$$
> 
> 梯度截断的目的是防止不同时间步的预测结果之间的梯度传递，也就是确保
>  
> $$\frac {\partial \mathbf x''_t}{\partial \mathbf x''_{t-1}} = 0$$
> 
>  梯度截断阻止了时间步的

The evolution network augments with an encoder–decoder architecture that simultaneously predicts motion fields $\mathbf{v}_{1:T}$ and intensity residuals $\mathbf s_{1:T}$ at all future time steps based on past radar fields $\mathbf x_{-T_{0}:0},$ . Such a full dependency between the past and future time steps mitigates the non stationarity issue in sequence prediction. Also, the evolution encoder, motion decoder and intensity decoder are neural networks (Fig. 1b), enabling nonlinear evolution modelling, which previous advection schemes struggle to capture. 
>  演化网络还引入了编码器-解码器架构，基于过去的雷达场 $\mathbf x_{-T_0:0}$ 同时预测未来所有时间步的运动场 $\mathbf v_{1:T}$ 和强度残差 $\mathbf s_{1:T}$
>  过去和未来时间步的全依赖关系缓解了序列预测中的非平稳问题
>  演化编码器、运动解码器、强度解码器都是神经网络，以进行非线性演化建模

> [!info] 稳定过程和非平稳过程
> - 稳定过程：在统计学中，一个稳定过程是指其统计特性（如均值、方差、自相关函数等）不随时间变化的过程。
> - 非平稳过程：相反，非平稳过程的统计特性随时间变化。例如，在天气预报中，气象现象（如风暴、雷暴、阵雨等）可能会迅速演变，导致系统的统计特性在短时间内发生显著变化。

Learning of the evolution network is framed as directly optimizing the forecast error throughout the time horizon. The accumulated error arises in the evolution operator, measured by the sum of distances between evolved field ${\bf{x}}_{t}^{\prime\prime}$ and the observed radar ${\mathbf x}_{t}$ . Because each evolution step involves solving for both the motion field $\mathbf{v}_{t}$ and the intensity residual $\mathbf s_{t},$ to shortcut the gradient path for end-to-end optimization, we adopt the concept of residual learning and further calculate the sum of distances between the advected field ${\bf x}_{t}^{\prime}$ and the observed radar ${\mathbf x}_{t}$ . Combining the two sums of distances leads to the accumulation loss. 
>  演化网络的学习目标是直接优化整个时间范围内的预测误差
>  演化算子中，预测场 $\mathbf x_t''$ 和观测场 $\mathbf x_t$ 之间的距离的和作为累积误差
>  另外，每一步演化还涉及了求解运动场 $\mathbf v_t$ 和强度场 $\mathbf s_t$，为了缩短梯度路径以实现端到端优化，我们采用残差学习的思想，进一步计算了平流场 $\mathbf x_t'$ 和观测场 $\mathbf x_t$ 之间的距离的和，将二者结合作为累积损失

> [!info] 残差思想
> 残差学习表示为 
> 
> $$\mathbf y = f(\mathbf x) + \mathbf x$$
> 
> 此时 $\mathbf y$ 相对于 $\mathbf x$ 的梯度有两条传递路径
> 
> $$\frac {d \mathbf y}{d \mathbf x} = \frac {\partial \mathbf y}{\partial f(\mathbf x)}\frac {d f(\mathbf x)}{d \mathbf x} + \frac {\partial \mathbf y}{\partial \mathbf x}$$
> 
> 在演化网络中，将平流场和观测场之间的距离也加入到损失中，观测场 $\mathbf x_t$ 相对于平流场 $\mathbf x_t'$ 之间的梯度就有了两条路径，一条是经过预测场 $\mathbf x_t''$，一条是直接达到 $\mathbf x_t'$，因此虽然没有显式的残差结构，残差思想也是存在的

Furthermore, inspired in part by the continuity equation and in part by the fact that large precipitation patterns tend to be longer lived than small ones, we further design a motion-regularization term to make the motion fields smoother on the grids with heavier precipitation. Specifically, the spatial gradients of the motion fields $\mathbf{v}_{1:T}$ are computed by a Sobel filter and the gradient norm, weighted by rain rate, is used as the regularizer. 
>  受连续性方程启发，并考虑到大的降水模式往往比小降水模式持续时间更长
>  演化网络进一步添加了运动正则化项，使运动场在降水量大的网格上更加平滑 (大降水模式的运动场应该更加平滑和稳定，以反映其较长的持续时间和较大规模的空间分布)
>  我们用 Sobel 滤波器计算运动场 $\mathbf v_{1:T}$ 的空间梯度，使用降雨率加权的梯度范数作为正则化项 (这使得降雨量较大的地方的运动场的空间梯度倾向于更小，运动场也就倾向于更平滑)

# Evaluation settings 
We evaluate the forecasting skill and value of NowcastNet against state-of-the-art precipitation nowcasting models. pySTEPS, an advection-based method, has been widely adopted by meteorological centres worldwide for operational nowcasting. PredRNN, a data-driven neural network, has been deployed at the China Meteorological Administration. DGMR, an ensemble nowcasting method based on deep generative models with integrated domain knowledge, for example, spatiotemporal consistency of clouds and heavy-tailed distribution of rainfall, has shown the best forecasting skill and value in an expert evaluation held by the UK Met Office. 
>  NowcastNet 和 pySTEPS, PredRNN, DGMR 进行比较

All models are trained and tested on large radar corpora of the USA and China events, consisting of crops in fixed-length series extracted from the radar stream. An importance-sampling strategy is used to create datasets more representative of extreme-precipitation events.
>  训练数据集是中国和美国的大规模雷达数据集，数据集由从雷达流中提取的固定长度的序列组成
>  我们使用了重要性采样策略以创建更能表现极端降水事件的数据集

 In the USA corpus, we use the Multi-Radar Multi-Sensor (MRMS) dataset and all models are trained with radar observations for the years 2016–2020 and evaluated for the year 2021. In the China corpus, we use a private dataset provided by the China Meteorological Administration, with radar observations from September 2019 to March 2021 for training and from April 2021 to June 2021 for evaluation. Although the China corpus is smaller, the underlying weather system is more complex owing to geographical diversity. To avoid overfitting, we use a transfer learning strategy, in which all models are pre-trained on the USA training set and fine-tuned to China training set.
 >  为了避免过拟合，评估时采用迁移学习策略，所有模型在 USA 数据集的训练集上训练，在 China 数据集的训练集上微调

NowcastNet can produce high-resolution fields in seconds at inference time. We report two main quantitative metrics: the CSI with neighborhood that measures the location accuracy of nowcasts and the power spectral density (PSD) that measures the precipitation variability based on spectral characteristics of nowcasts compared with that of radar observations. 
>  评估指标包括两个：
>  1. 邻域内的 CSI 指数，它度量了临近预报的位置准确率 (预测的降水区域和实际观测的降水区域的重合度，引入邻域是放宽了对位置精度的要求)
>  2. 功率谱密度，它基于预测的谱特征度量了降水变异性，我们将预测的功率谱密度和雷达观测的功率谱密度进行比较 (功率谱密度衡量了信号在不同频率上的能量分布)

# Precipitation events 
![[pics/NowcastNet-Fig2.png]]

We investigate a precipitation event starting at 09:30 UTC on 11 December 2021 (Fig. 2), which was part of a tornado outbreak in eastern USA. First, several lines of intense storm developed across the Mississippi Valley and moved eastward; later, they converged to a convective fine line stretching along the associated cold front and sweeping from eastern Kentucky into Alabama. This precipitation event led to dozens of tornadoes, widespread rainstorms and straight-line winds reaching speeds of 78 mph. Prediction of the fine line, represented by the yellow line echo in the radar fields, is known to be very challenging. 
> 我们研究了一场始于 2021 年 12 月 11 日 UTC 时间 09:30 的降水事件（图 2），该事件是美国东部龙卷风爆发的一部分。首先，几条强烈的风暴线在密西西比河流域形成并向东移动；随后，它们汇聚成一条沿相关冷锋延伸的对流细线，从肯塔基州东部扫向阿拉巴马州。这场降水事件导致了数十次龙卷风、大范围暴雨和阵风，风速达到 78 英里/小时。预测这条对流细线（在雷达图中用黄色回波表示）一直被认为是非常具有挑战性的。

pySTEPS predicts future radar fields of good sharpness but incurs large location error and fails to keep the shape of the line echo at 1 h ahead. PredRNN only provides an outline trend but the predictions are too blurry, losing the multiscale patterns useful for meteorologists to make forecasts. DGMR is able to preserve the convective details but suffers from unnatural cloud dissipation, yielding large location errors and underestimated intensities. Worse still, the shapes of the line predicted by DGMR are excessively distorted. Throughout the 3-h event, NowcastNet is the only method able to accurately predict the movement of the fine line and preserve the envelope of the rain area. The line echo covers intense rainfall $(>\!32\,\mathsf{m m}\,\mathsf{h}^{-1})$ , for which NowcastNet achieves notably better CSI. NowcastNet also achieves the highest PSD at all wavelengths (that is, spatial scales), yielding sharp, consistent and multiscale nowcasts in reference to the ground truth. 
>  pySTEPS 预测的未来雷达场具有良好的清晰度，但存在较大的位置误差，并且无法保持 1 小时后的细线回波形状。
>  PredRNN 只提供大致的趋势，但预测结果过于模糊，失去了多尺度模式。
>  DGMR 能够保留对流细节，但存在不自然的云消散现象，导致较大的位置误差和低估的强度。更糟糕的是，DGMR 预测的细线形状过度扭曲。
>  在整个 3 小时的事件中，NowcastNet 是唯一能够准确预测细线移动并保留降雨区域轮廓的方法。
>  细线回波覆盖了强降雨（> 32 mm/h）区域，NowcastNet 在这类区域上的 CSI 表现尤为出色。NowcastNet 在所有波长（即空间尺度）上也实现了最高的 PSD，相对于真实情况，产生了清晰、一致且多尺度的临近预报。

![[pics/NowcastNet-Fig3.png]]

We investigate another precipitation event starting at 23:40 UTC on 14 May 2021 in the Jianghuai area of China (Fig. 3), for which several cities issued red rainstorm warnings. Three convective cells evolved differently. The first cell moved from the centre to the northeast, developing into a bow echo from a single-cell thunderstorm echo. The second cell was a squall line moving from the southwest to the middle, with the tail moving to the east. The third cell was in between and showed steady growth. 
>  我们研究了 2021 年 5 月 14 日世界协调时 23：40 开始于中国江淮地区的另一次降水事件（图 3），期间多个城市发布了暴雨红色预警。三个对流胞的发展情况各不相同。第一个对流胞从中部向东北方向移动，由单一对流雷暴回波发展成为一个弓形回波。第二个对流胞为一条飑线，从西南向中部移动，其尾部移向东边。第三个对流胞位于两者之间，并显示出稳定增长的趋势。

> [!info] 对流胞
> 对流胞（convective cell）是指大气中由对流活动形成的一种结构，它通常与热力不稳定性有关，一个典型的对流胞包括：
> 
> 1. 上升气流区：这是对流胞的核心部分，暖湿空气在此处上升，冷却并凝结成云滴或冰晶，从而形成云体。如果条件合适，这部分可以发展成为雷暴云。
> 
> 2. 下沉气流区：随着上升气流中的水汽凝结释放潜热，推动空气继续上升，最终云顶达到稳定层后，其中一些较冷、较干的空气开始下沉。这些下沉气流可以在地表附近造成强风，并且在接触到地面时向外扩散，形成所谓的“外流边界”。
> 
> 3. 降雨区：当云中的水滴变得足够大时，它们会因为重力作用而落下，形成降水。这可能伴随着上升气流区，也可能出现在下沉气流区前方，取决于风暴的具体结构。

Subject to noncompliance of physical conservation laws, PredRNN and DGMR suffer from fast dissipation and fail to predict the evolution of any convective cell at a 2-h lead time. pySTEPS predicts the direction of the three cells but fails to predict the specific location or the shape change. By contrast, NowcastNet yields plausible nowcasts for the evolutions of the three cells at a 3-h lead time. Although the nowcasts of the squall line and the growing cell are still not perfect, they are useful for meteorologists. Quantitative results of NowcastNet in terms of CSI neighborhood and PSD are substantially improved relative to the leading methods. 
>  在不遵守物理守恒定律的情况下，PredRNN 和 DGMR 存在快速耗散的问题，无法在 2 小时的提前时间内预测任何对流单元的演变。
>  pySTEPS 能够预测三个对流胞的方向，但无法准确预测具体位置或形状变化。
>  相比之下，NowcastNet 能够在 3 小时的提前时间内对这三个单元的演变提供合理的预测。尽管飑线和增长胞的预测仍不完美。
>  NowcastNet 在 CSI 邻域和 PSD 方面的定量结果相对于领先的方法有了显著的改进。

We inspect more weather events with extreme precipitation, convective initiation, light rainfall and typical processes in Extended Data Figs. 2–8 and Supplementary Figs. 2–5. High-resolution nowcasts of $2{,}048\,\mathsf{k m}\times2{,}048$ km are shown in Extended Data Figs. 9 and 10. 
>  我们进一步检查了更多极端降水、对流启动、小雨和典型过程的天气事件，详情见扩展数据图 2-8 和补充图 2-5。高分辨率的临近预报（覆盖范围为 2,048 公里×2,048 公里）展示在扩展数据图 9 和 10 中。

# Meteorologist evaluation 
We evaluate the forecast value of different models for extreme-precipitation events by the meteorologist evaluation protocol from the UK Met Office. For fairness, the China Meteorological Administration made a public invitation to senior meteorologists across China to participate in the evaluation. On the public website, experts can control the display of precipitation fields but the nowcasts of different models are shown anonymously and out of order. Finally, 62 expert meteorologists from the central and 23 provincial observatories completed the evaluation, each judging 15 test cases chosen randomly from the extreme-precipitation-event subsets. The USA and China subsets consist of 1,200 extreme events occurring over 93 days in 2021 and 50 days from April 2021 to June 2021, respectively. We note that, although judging the USA events by China meteorologists may incur some bias, we expect it to be relatively minor, as the global weather system shares underlying physical principles and the two countries share meteorological observations and technologies. 
>  我们通过英国气象局的气象学家评估协议来评估不同模型对极端降水事件的预报价值。为了公平起见，中国气象局公开邀请了中国各地的资深气象学家参与评估。在公共网站上，专家可以控制降水场的显示，但不同模型的即时预报是匿名且无序展示的。
>  最终，来自中央和 23 个省级观测站的 62 位专家气象学家完成了评估，每位专家随机评判了 15 个从极端降水事件子集中选择的测试案例。美国和中国的子集分别包含了 2021 年 93 天内发生的 1200 个极端事件和 2021 年 4 月至 6 月 50 天内发生的极端事件。
>  我们注意到，尽管由中国气象学家评判美国的事件可能会产生一些偏差，但我们预计这种偏差相对较小，因为全球天气系统共享基本的物理原理，并且两国共享气象观测和技术。

We augment the UK Met Office protocol by running two types of evaluation: posterior evaluation and prior evaluation. In the posterior evaluation, meteorologists were asked to objectively rank the forecasting value of the predictions of each model with reference to the future ground-truth observations. In the prior evaluation, meteorologists needed to subjectively rank the forecasting value given past radar series but without seeing the future ground truth. This protocol simulates the real scenario in which future observations are not accessible and meteorologists have to make an on-the-fly choice of which model is preferred for nowcasting. 
>  我们通过进行两种类型的评估来增强英国气象局的评估协议：后验评估和先验评估。在后验评估中，气象学家被要求根据未来的实际观测数据客观地对每个模型的预测价值进行排名。在先验评估中，气象学家需要在没有看到未来实际观测数据的情况下，仅根据过去的雷达序列主观地对预测价值进行排名。
>  这种协议模拟了实际情况下未来观测数据不可用时，气象学家必须即时选择哪个模型更适合进行即时预报的场景。

![[pics/NowcastNet-Fig4.png]]

The statistics of meteorologist evaluation are shown in Fig. 4a,b. In the posterior evaluation, NowcastNet was ranked as the first choice for $75.8\%$ of the USA events ([72.1, 79.3]) and for $67.2\%$ of the China events  ([63.1, 71.1]). In the prior evaluation, NowcastNet was ranked as the first choice for $71.9\%$ of the USA events ([66.6, 76.8]) and $64.4\%$ of the China events ([58.9, 69.7]). The numbers in brackets are $95\%$ confidence intervals. NowcastNet holds the highest meteorologist preference by providing skilful nowcasts that exhibit physical plausibility and multiscale features, whereas other models struggle. 
>  气象学家评估的统计结果如图 4a 和 4b 所示。在后验评估中，NowcastNet 被选为美国事件的首选模型的比例为 75.8%（[72.1, 79.3]），被选为中国事件的首选模型的比例为 67.2%（[63.1, 71.1]）。在先验评估中，NowcastNet 被选为美国事件的首选模型的比例为 71.9%（[66.6, 76.8]），被选为中国事件的首选模型的比例为 64.4%（[58.9, 69.7]）。括号中的数字是 95%的置信区间。NowcastNet 通过提供具有物理合理性和多尺度特征的准确即时预报，获得了最高的气象学家偏好，而其他模型则表现不佳。

# Quantitative evaluation 
We provide a quantitative evaluation based on the results for CSI neighborhood and PSD shown in Fig. 4c,d. The evaluation includes $\mathbf{U}\mathbf{\cdot}\mathbf{N e t}^{30}$ , a common baseline for precipitation nowcasting. Adopting the importance-sampling protocol of DGMR, we sample two subsets from the USA and China corpora, both representative of extreme-precipitation events. By CSI neighborhood, NowcastNet produces more accurate nowcasts at higher rain rate $(>\!\!16\,\mathsf{m m}\,\mathsf{h}^{-1})$ . By PSD, NowcastNet yields sharper nowcasts of more consistent variability in spectral characteristics to radar observations for a 3-h lead time. These quantities justify that NowcastNet is skilful for extreme-precipitation nowcasting, better able to predict precipitation patterns at both the mesoscale and the convective scale, while maintaining high accuracy of evolution prediction over a longer time period. 
>  我们基于图 4c 和 4d 中显示的 CSI 邻域和 PSD 结果提供了定量评估。评估包括一个常见的降水即时预报基准模型 U-Net。
>  我们采用 DGMR 的重要性采样方法，从美国和中国的数据集中分别抽取了两个子集，这两个子集都代表了极端降水事件。根据 CSI 邻域，NowcastNet 在较高降雨率（>16 mm/h）下生成了更准确的即时预报。根据 PSD，NowcastNet 在 3 小时的提前时间内生成了与雷达观测结果在频谱特性上更一致且更清晰的即时预报。
>  这些指标证明了 NowcastNet 在极端降水即时预报方面具有高技能，能够更好地预测中尺度和对流尺度的降水模式，同时在较长时间内保持高精度的演变预测。

In Supplementary Figs. 10–17, we provide further quantitative evaluations under both uniform-sampling and importance-sampling protocols. 
>  在补充图 10-17 中，我们进一步提供了在均匀采样和重要性采样方法下的定量评估。

# Conclusion 
Precipitation nowcasting is a leading long-term goal of meteorological science. Although progress has been made, numerical weather-prediction systems are at present unable to provide skilful nowcasts for extreme-precipitation events that are needed for weather-dependent policy making. 

Much of the inherent difficulty of nowcasting stems from the multiscale and multi physics problems arising in the atmosphere and the need to combine physical first principles with statistical-learning methods in a rigorous way. Our work addresses this challenge using an end-to-end optimization framework that combines physical-evolution schemes and conditional-learning methods. The resulting model, NowcastNet, provides physically plausible nowcasts with high resolution, long lead time and local details for extreme-precipitation events, for which existing methods struggle. 
>  临近预报的固有难度很大程度上来自于气象中的多尺度和多物理问题，这要求我们结合物理第一定律和统计学习方法
>  本工作提出一种端到端的优化框架，它结合了物理演化方案和条件学习方法
>  由此，模型可以为极端降水时间提供物理上合理的高分辨率、长提前时间和具有局部细节的临近预测

Much future work is needed to improve precipitation nowcasting skill. One direction is integration of more physical principles such as momentum conservation. Another direction is exploitation of more meteorological data such as satellite observations. We hope this work will inspire future research in these directions. 

# Methods 
Detailed explanations of the proposed model, as well as baselines, datasets and evaluations, are given here, with references to the Extended Data Figs. and Supplementary Information that add to the results provided in the main text. 

## Model details 
We describe NowcastNet with important details of the model architectures, the training methods and the hyperparameter tuning strategies. Ablation study of NowcastNet is available in Supplementary Information section A. 

**Evolution network.** The 2D continuity equation modified for precipitation evolution is 

$$
\frac{\partial\mathbf {x}}{\partial t}+(\mathbf v\cdot\nabla)\mathbf x=\mathbf {s}.\tag{2}
$$

Here $\mathbf x$ , $\mathbf v$ and $\mathbf s$ indicate radar fields of composite reflectivity, motion fields and intensity residual fields, respectively, and $\nabla$ denotes the gradient operator. 

>  降水演化的二维连续性方程如上所示，其中 $\mathbf x$ 是组合雷达反射率场，$\mathbf v$ 是 (平流)运动场，$\mathbf s$ 是强度残差场
>  $\nabla$ 是梯度算子
>  该式是流体连续方程到雷达反射率场的迁移，将流体连续方程中的密度场 $\rho$ 替换为了雷达反射率场 $\mathbf x$

> [!info] 雷达反射率场（Radar Reflectivity Field）
>  雷达反射率场描述了大气中水滴、冰晶等粒子对雷达波的散射强度。综合反射率（composite reflectivity）是这些散射强度的最大值，能够有效表示降水的强度和分布。现代气象雷达通过发射电磁波并接收其回波来测量雷达反射率场。
> 
> - 雷达反射率因子 $Z$：雷达反射率因子是降水目标物回波强度的单位，它量化了雷达波在遇到大气中的水滴、冰晶等粒子时的后向散射强度，其大小与降水目标物单位体积中降水粒子的大小、数量以及相态有关，单位为 $mm^6/m^3$。
> 
> - 综合反射率 $\mathbf{x}$：综合反射率是指在某个垂直剖面上所有高度层的雷达反射率因子 $Z$ 的最大值，它反映了该区域内最强烈的降水情况。
> 
> 雷达反射率场直接反映了大气中降水粒子的存在及其特性。具体来说：
> 
> - 高反射率值：对应于强降水区域，表明存在大量的大水滴或冰晶。
> - 低反射率值：对应于弱降水或无降水区域，表明存在的水滴或冰晶较少或较小。
> - 零反射率值：表示没有降水，即雷达波未遇到任何显著的散射体。
> 

The tendency term $({\bf v}\cdot\nabla){\bf x}$ reveals the mass leaving the system, which is the first-order approximation of the difference before and after the advection operation: 

$$
\frac{\mathbf{\boldsymbol{x}}(\mathbf{\boldsymbol{p}}+\Delta t\cdot\Delta\mathbf{\boldsymbol{v}},t+\Delta t)-\mathbf{\boldsymbol{x}}(\mathbf{\boldsymbol{p}},t)}{\Delta t},\tag{3}
$$

with $\mathbf p$ and $\mathbf t$ being the position and time, respectively. The residual field $\mathbf s$ shows the additive evolution mechanisms, such as the growth and decay of precipitation intensities. 

>  推导

$$
\begin{align}
\mathbf x(\mathbf p + \Delta t\cdot\mathbf v, t+\Delta t)
&=\mathbf x(x + \Delta x,y + \Delta y,t + \Delta t) \\
&\approx \mathbf x(x,y,t) + \frac {\partial \mathbf x}{\partial x}\Delta x + \frac {\partial \mathbf x}{\partial y}\Delta y + \frac {\partial \mathbf x}{\partial t}\Delta t\\
&=\mathbf x(\mathbf p, t) +\frac {\partial \mathbf x}{\partial x}\Delta x + \frac {\partial \mathbf x}{\partial y}\Delta y + \frac {\partial \mathbf x}{\partial t}\Delta t\\
\mathbf x(\mathbf p + \Delta t\cdot\mathbf v, t+\Delta t)-\mathbf x(\mathbf p, t)&=\frac {\partial \mathbf x}{\partial x}\Delta x + \frac {\partial \mathbf x}{\partial y}\Delta y + \frac {\partial \mathbf x}{\partial t}\Delta t\\
\frac {\mathbf x(\mathbf p + \Delta t\cdot \mathbf v, t+\Delta t)-\mathbf x(\mathbf p, t)}{\Delta t}&=\frac {\partial \mathbf x}{\partial x}\frac {\Delta x}{\Delta t} + \frac {\partial \mathbf x}{\partial y}\frac {\Delta y}{\Delta t} + \frac {\partial \mathbf x}{\partial t}\\
&=\frac {\partial \mathbf x}{\partial t}+\frac {\partial \mathbf x}{\partial x}v_x + \frac {\partial \mathbf x}{\partial y}v_y\\
&=\frac {\partial \mathbf x}{\partial t} + (\mathbf v \cdot \nabla)\mathbf x
\end{align}
$$

>  其中第二个 $\approx$ 使用了雷达观测场的一阶 Taylor 近似
>  因此，(2) 的 LHS 应该视作对 (3) 的近似，也就是对雷达观测场中，某个点在运动场的作用下，经过一段时间后，它的属性 (该点的综合反射率) 的变化率的近似
>  在拉格朗日质点模型中，质点的属性不随着质点的转移而变化，因此 (2) 的 RHS 应该是 0，但是 (2) 引入了残差项 $\mathbf s$，它表示了平流作用 (运动场 $\mathbf v$) 以外的影响，例如对流作用，因此虽然假设了平流作用 (运动场 $\mathbf v$) 下，质点的属性不会改变，但实际的改变率不是零

According to the continuity equation, the temporal evolution of precipitation can be modelled as a composition of advection by motion fields and addition by intensity residuals, which is the evolution operator we design for the evolution network. We use deep neural networks to simultaneously predict all these fields based on past radar observations, which enables nonlinear modelling capability for the complex precipitation evolution. 
>  根据连续性方程，降水的时序演变可以建模为由运动场引起的平流作用和强度残差叠加的组合
>  我们基于该连续性方程设计演化算子，使用深度神经网络基于过去的雷达观测同时预测所有的这些场，使得模型可以非线性建模复杂降水演化

![[pics/NowcastNet-extend fig1a.png]]

The evolution network (Fig. 1b) takes as input past radar observations $\mathbf x_{-T_{0}:0}$ and predicts future radar fields $\mathbf x_{1:T}^{\prime\prime}$ at a 20-km scale based on a nonlinear, learnable evolution scheme we propose specifically in this article. The architecture details are described in Extended Data Fig. 1a. The backbone of the evolution network is a two-path U-Net, which has a shared evolution encoder for learning context representations, a motion decoder for learning motion fields $\mathbf{v}_{1:T}$ and an intensity decoder for learning intensity residuals $\mathbf {s}_{1:T}.$ . The spectral normalization technique is applied in every convolution layer. In the skip connections of U-Net, all input and output fields are concatenated on the temporal dimension, that is, the channels in convolutional networks. 
>  演化网络接受过去的雷达观测 $\mathbf x_{-T_0:0}$ 作为输入，预测 20km 尺度的未来雷达场 $\mathbf x_{1: T}''$
>  其架构如图 fig1a，演化网络的主干是一个双路径 U-Net，包括一个共享的演化编码器，用于学习上下文表示、一个运动解码器，用于学习运动场 $\mathbf v_{1:T}$、一个强度解码器，用于学习强度残差 $\mathbf s_{1:T}$
>  每层卷积层都使用了谱规范化
>  U-Net 的跳跃连接中，所有的输入和输出在时间维度上拼接，即卷积网络中的通道维度 

The evolution operator (Fig. 1c) is at the core of the evolution network. We use the backward semi-Lagrangian scheme as the advection operator. Because $\mathbf{v}_{1:T}$ is learnable, we directly set it as the departure offset of the semi-Lagrangian scheme. Also, because $\mathbf s_{1:T}$ is learnable, we directly use it to model the growth or decay of precipitation intensities. We take precipitation rate instead of radar reflectivity as the unit of radar field $\mathbf {x}$ , as this modification will not influence the physical nature of the evolution process. 
>  演化算子是演化网络的核心，其中平流算子选为后向半拉格朗日方案，因为 $\mathbf v_{1:T}$ 是可学习的，我们直接将 $\mathbf v_{1:T}$ 设定为出发偏移量，同样因为 $\mathbf s_{1:T}$ 是可学习的，我们直接用它建模降水量强度的变化
>  我们使用降水率而不是雷达反射率作为雷达场 $\mathbf x$ 的单位，这样修改不会影响降水过程的物理本质，但可以让结果更直观

As applying bilinear interpolation for several steps will blur the precipitation fields, we opt for the nearest interpolation in the backward semi-Lagrangian scheme for computing ${\bf x}_{t}^{\prime}.$ . Yet, the nearest interpolation is not differentiable at $\mathbf{v}_{1:T}.$ We resolve this gradient difficulty by using bilinear interpolation (bili) to advect $(\mathbf{x}_{t}^{\prime})_{\mathrm{bili}}$ from $\mathbf{x}_{t-1}^{\prime\prime},\mathbf{v}_{1: T},$ , and use $(\mathbf{x}_{t}^{\prime})_{\mathrm{biii}}$ to compute the accumulation loss for optimizing the motion fields. Then we use the nearest interpolation to compute $\mathbf x_t'$ from $\mathbf{x}_{t-1}^{\prime\prime},\mathbf{v}_{1:T}$, and compute the evolved field $\mathbf x_t''$ from $\begin{array}{r}{{\bf x}_{t}^{\prime\prime}\!=\!{\bf x}_{t}^{\prime}+{\bf s}_{t}.}\end{array}$ . 
>  因为在连续的步骤中多次使用双线性插值会让降水场模糊化，我们使用最近邻插值从 $\mathbf x_{t-1}''$ 来计算 $\mathbf x_t'$ ，即计算速度场的平流效应下变化后的雷达场
>  (每次插值会引入一定程度的平滑效应，导致原始数据的细节逐渐丧失)
>  最近邻插值操作对于 $\mathbf v_{1:T}$ 不可微，故我们使用双线性插值，根据 $\mathbf x_{t-1}''$ 和 $\mathbf v_{1:T}$ 计算 $(\mathbf x_t')_{\mathrm {bili}}$，然后使用双线性插值计算累积损失以优化运动场
>  (也就是运动场的优化仍然是根据双线性插值进行，但传递给下一步的结果是由最近邻插值计算的)
>  因此，我们实际上使用最近邻插值从 $\mathbf x_{t-1}'', \mathbf v_{1:T}$ 计算 $\mathbf x_t'$，然后再计算 $\mathbf x_t'' = \mathbf x_t' + \mathbf s_t$

After each round of the evolution operator, we detach the gradient between two consecutive time steps because the overall system is under determined. Meanwhile, the successive interpolation operations will make end-to-end optimization unstable, and detaching the gradient (stop gradient in Fig. 1c) will markedly improve the numerical stability. 
>  在每一次演化算子应用后，我们断开两个连续时间步之间的梯度，因为整个系统是欠定的 (欠定指存在多种可能解)，另外，连续的插值操作会让端到端优化不稳定，故断开梯度会显著提高数值稳定性

The objective function for training the evolution network comprises two parts. The first part is the accumulation loss, which is the sum of the weighted $L_{1}$ distances between real observations and predicted fields: 

$$
J_{\mathrm{accu m}}=\sum_{t=1}^{T}\,\left(L_{\mathrm{wdis}}(\mathbf{x}_{t},\,(\mathbf{x}_{t}^{\prime})_{\mathrm{bili}})+L_{\mathrm{wdis}}(\mathbf{x}_{t},\mathbf{x}_{t}^{\prime\prime})\right).\tag{4}
$$

In particular, the weighted distance has the following form: 

$$
L_{\mathrm{wdis}}(\mathbf x_{t},\mathbf x_{t}^{\prime})=\|(\mathbf x_{t}-\mathbf x_{t}^{\prime})\odot\pmb{\mathrm{w}}(\mathbf x_{t})\|_{1},\tag{5}
$$ 
in which the pixel-wise weight $w(x)\,{=}\,\min(24,1\,{+}\,x)$ is taken from DGMR. 

>  训练演化网络的目标函数包括两部分，第一部分是累积误差，即真实观测场和预测场之间的加权 $L_1$ 距离
>  每个像素的权重 $w(x) = \min(24, 1+x)$

Because the rain rate approximately follows a log-normal distribution, it is necessary to add weight to balance different rainfall levels. Otherwise, neural networks will only fit light-to-medium precipitation taking dominant ratio in the data and heavy precipitation will not be accounted for sufficiently. We follow DGMR and use a weight proportional to the rain rate and clip it at 24 for robustness to spuriously large values in radar observations. 
>  因为降雨率近似服从对数正态分布 (这说明大多数数据点集中在轻度到中度的降水量范围内，重度降水量的频率较低)，故有必要添加权重以平衡不同的降水水平，否则神经网络会仅拟合轻度到中度的降水，因为它们在数据中占据主导比率
>  因此我们才用 DGMR 的加权机制，权重和降雨率成正比，同时限制权重最大值为 24，提高对于雷达观测中的异常大值的健壮性

The second part is the motion-regularization term in the form of gradient norm, which is motivated in part by the continuity equation and in part by the fact that large precipitation patterns tend to be longer lived than small ones: 

$$
J_{\mathrm{motion}}=\sum_{t=1}^{T}\;\big(\|\nabla\mathbf{v}_{t}^{1}\odot\sqrt{\mathbf{w}(\mathbf{x}_{t})}\,\|_{2}^{2}+\|\nabla\mathbf{v}_{t}^{2}\odot\sqrt{\mathbf{w}(\mathbf{x}_{t})}\,\|_{2}^{2}\big)\,,\tag{6}
$$ 
in which $\mathbf{v}_{t}^{1}$ and $\mathbf{v}_{t}^{2}$ are the two components of the motion fields. The gradient of the motion fields $\nabla\mathbf v$ is computed approximately with the Sobel filter: 

$$
\partial_{1}\boldsymbol{\mathbf{v}}\,{\approx}\,\left(\begin{array}{c c c}{1}&{0}&{-1}\\ {2}&{0}&{-2}\\ {1}&{0}&{-1}\end{array}\right)\,{*}\,\boldsymbol{\mathbf{v}},\qquad\partial_{2}\boldsymbol{\mathbf{v}}\,{\approx}\,\left(\begin{array}{c c c}{1}&{2}&{1}\\ {0}&{0}&{0}\\ {-1}&{-2}&{-1}\end{array}\right)\,{*}\,\boldsymbol{\mathbf{v}},
$$ 
in which $^*$ denotes the 2D convolution operator in the spatial dimension.

>  演化网络目标函数的第二项是运动正则化项，其形式是梯度范数
>  正则化的思路来自于连续性方程 (降水在运动过程中应该保持一定的连续性，避免不合理的突变或跳跃) 和大尺度降水模式通常比小尺度降水模式更加持久的事实 (降水量大的地方运动场应该更加平滑连贯，即梯度应该更小)
>  运动正则化项中，运动场的两个方向的梯度使用 Sobel 滤波器近似，权重矩阵 $\mathbf w(\mathbf x_t)$ 和降水量有关

Overall, the objective for training the evolution network (Fig. 1b) is 

$$
J_{\mathrm{evolucion}}\!=\!J_{\mathrm{accur}}\!+\!\lambda J_{\mathrm{motion}}.\tag{8}
$$ 
>  演化网络最后的目标函数如上所示

During training, we sample the radar fields with $256\times256$ spatial size as the input. On both the USA and China datasets, we fix input length $T_{0}\,{=}\,9$ and set output length $T\!=\!20$ for training and take the first 18 predicted fields for evaluation. Note that increasing $T_{0}$ does not provide substantial improvements and $T_{0}\!\ge\!4$ is sufficient. The tradeoff hyperparameter $\lambda$ is set as ${1}\times{10^{-2}}$ . We use the Adam optimizer with a batch size of 16 and an initial learning rate of ${1}\times{10}^{-3}$ , and train the evolution network for $3\times10^{5}$ iterations, during which we decay the learning rate to ${1}\times{10^{-4}}$ at the $2\times10^{5}{\mathrm{th}}$ iteration. 
>  训练时，雷达场根据 256x256 采样，输入序列长度固定为 $T_0= 9$，输出序列长度固定为 $T=20$，评估时，则选择前 $18$ 个输出
>  增大 $T_0$ 不会显著提高表现，$T_0\ge 4$ 就足够
>  正则化项系数 $\lambda$ 设定为 $1\times 10^{-2}$
>  优化器使用 Adam，初始学习率 $1\times 10^{-3}$，batch size 为 $16$ 
>  训练迭代数量 $3\times 10^5$，在第 $2\times 10^5$ 次迭代将学习率衰减到 $1\times 10^{-4}$

**Generative network.** Conditioning on the evolution network predictions $\mathbf x_{1:T}^{\prime\prime},$ , the generative network takes as input the past radar observations $\mathbf x_{-T_{0}:0}$ and generates from latent random vectors $\mathbf z$ for the final predicted precipitation fields $\widehat{\mathbf{x}}_{1:T}$ at a 1-2km scale. 
>  生成式网络接受过去雷达观测 $\mathbf x_{-T_0:0}$，条件于演化网络的预测 $\mathbf x_{1: T}''$，从隐随机向量 $\mathbf z$ 生成 1-2km 尺度的最终预测降水场 $\widehat {\mathbf x}_{1:T}$

The backbone of the generative network is a U-Net encoder–decoder structure, with architecture details shown in Extended Data Fig. 1b. The nowcast encoder has the identical structure as the evolution encoder (Extended Data Fig. 1a), which takes as input the concatenation of $\mathbf x_{-T_0:0}$ and $\mathbf x_{1:T}^{\prime\prime}$ . The nowcast decoder is a different convolutional network, which takes as input the contextual representations from the nowcast encoder, along with the transformation of the latent Gaussian vector $\mathbf z$ .  
>  生成式网络的主干是 U-Net 编解码器架构，其中编码器的架构和演化网络的编码器架构相同，它接受的输入是演化网络的预测 $\mathbf x_{1: T}''$ 和雷达观测 $\mathbf x_{-T_0:0}$ 的拼接，解码器结构则是另一个卷积网络结构，接受编码器输出的上下文表示以及潜在高斯向量 $\mathbf z$ 的转化作为输入


![[pics/NowcastNet-extend fig1c.png]]

The designs of D Block, S Block and Spatial Norm heavily used in the generative network are elaborated in Extended Data Fig. 1e. 

The noise projector transforms the latent Gaussian vector $\mathbf z$ to the same spatial size as the contextual representations from the nowcast encoder, as elaborated in Extended Data Fig. 1d. For each forward pass, each element of $\mathbf z$ is independently sampled from the standard Gaussian ${\mathcal{N}}(0,1)$ . Then $\mathbf z$ is transformed by the noise projector into a tensor with one-eighth the height and width of input radar observations. 
>  噪声投影器将潜在高斯向量 $\mathbf z$ 转化为和编码器输出的上下文表示相同的空间尺寸
>  每次前向传播，$\mathbf z$ 中的每个元素独立从标准高斯分布 $\mathcal N(0, 1)$ 中采样，然后被转化为宽度和高度分别为输入雷达观测的八分之一的 tensor

The physics-conditioning mechanism to fuse the generative network and the evolution network is implemented by applying the spatially adaptive normalization to each convolutional layer of the nowcast decoder (Extended Data Fig. 1b,e). First, each channel of the nowcast decoder is normalized by a parameter-free instance-normalization module. Then the evolution network predictions $\mathbf x_{1:T}^{\prime\prime}$ are resized to a compatible spatial size and then concatenated to the nowcast decoder at the corresponding layer through average pooling. Finally, a two-layer convolutional network transforms the resized predictions into new mean and variance for each channel of the nowcast decoder, ensuring not to distort the spatial-coherent features from the evolution network predictions $\mathbf x_{1:T}^{\prime\prime}$ . Through the physics-conditioning mechanism, the generative network is adaptively informed by the physical knowledge learned with the evolution network, while resolving the inherent conflict between physical-evolution and statistical-learning regimes. 
>  用于融合生成网络和进化网络的物理条件机制通过在 nowcast 解码器中的每个卷积层应用空间自适应规范化实现
>  首先，我们 nowcast 解码器的每个通道通过一个无参数的实例规范化模块进行规范化 (规范化每个特征图的均值和方差，使分布更加稳定)
>  然后，演化网络预测 $\mathbf x_{1: T}''$ 通过平均池化调整大小到匹配的空间尺寸
>  最后，通过两层的卷积网络将调整尺寸后的预测 $\mathbf x_{1: T}''$ 转化为新的均值和方差，匹配 nowcast 解码器的每个通道，同时保持不扭曲预测 $\mathbf x_{1: T}''$ 的空间一致性特征
>  (使用卷积而不是直接规范化是因为规范化是对每个特征图独立进行的，可能会破坏特征图之间的空间连贯性，卷积则更加灵活)
>  生成网络在物理条件机制中学习演化网络提供的物理机制，同时解决统计学习方法和物理演化的内在冲突

Conditioning on the evolution network predictions at a 20-km scale, the generative network is needed to further generate convective details at a 1-2km scale through training on a temporal discriminator $D$ (Extended Data Fig. 1c). The temporal discriminator takes as input real radar observations $\mathbf x_{1:T}$ and final predicted fields $\widehat{\mathbf{x}}_{1:T}$ and outputs scores of how likely they are being real or fake. At its first layer, the inputs are processed by 3D convolution layers with several kernel sizes at the temporal dimension from 4 to the full horizon. Then the multiscale features are concatenated and feed forwarded to subsequent convolutional layers with spectral normalization applied in each layer. 

>  演化网络的预测的尺度是 20-km，生成网络条件于该预测，需要进一步生成 1-2km 尺度的对流细节
>  为此，生成网络上需要同时训练一个时间判别器 $D$，时间判别器的输入是真实雷达观测 $\mathbf x_{1:T}$ 和生成网络生成的最终预测雷达场 $\widehat {\mathbf x}_{1:T}$，输出预测雷达场是真或假的概率
>  判别器的第一层使用了多个 3D 卷积核处理输入，这些卷积核在时间维度具有多种大小，从 4 到整个时间范围，之后，多尺度特征被拼接，传播给后续带有谱规范化的卷积层

The objective for training the temporal discriminator is 

$$
\begin{array}{r}{J_{\mathrm{disc}}=L_{\mathrm{ce}}(D(\mathbf{x}_{1:T}),\mathbf{1})+L_{\mathrm{ce}}(D(\widehat{\mathbf{x}}_{1:T}),0),}\end{array}\tag{9}
$$ 
with $L_{\mathrm{ce}}$ being the cross-entropy loss. 

>  时间判别器的训练目标如上，优化该目标就是要求判别器尽可能区分出真假

Within a two-player minimax game, the nowcast decoder of the generative network is trained to confuse the temporal discriminator by minimizing the adversarial loss modified by

$$
J_{\mathsf{a d v}}=L_{\mathsf{c e}}(D(\widehat{\mathbf{x}}_{1:T}),1).\tag{10}
$$ 
>  生成网络的 decoder 的训练目标是最小化对抗损失，也就是尽可能混淆判别器

The gradients back propagate through $\widehat{\mathbf {x}}_{1:T}$ , first to the nowcast decoder and then to the nowcast encoder of the generative network, leading it to predict realistic multiscale fields with convective-scale details. 
>  梯度通过 $\widehat {\mathbf x}_{1:T}$ 回传，首先传入 decoder，然后传入 encoder，随着梯度引导参数更新，生成网络最后倾向于生成带有对流级别细节的真实多尺度场

We take the idea of generative ensemble forecasting from DGMR and predict a group of precipitation fields $\widehat{\mathbf{x}}_{1:T}^{\mathbf{z}_{i}}$ from several latent inputs $\begin{array}{r}{{\bf z}_{1:k},}\end{array}$ with $k$ being the number of ensemble members. Then we aggregate the $k$ predictions $\widehat{\mathbf{x}}_{1:T}^{\mathbf{z}_{i}}$ and real fields $\mathbf x_{1:T}$ respectively by a max-pooling layer $Q$ in the spatial dimension, with kernel size and stride set as 5 and 2, correspondingly. On the basis of ensemble forecasts, the pool regularization is defined as the weighted distance between spatial-pooled observations and the mean of $k$ spatial-pooled predictions 

$$
J_{\mathrm{pool}}\!=\!L_{\mathrm{wdis}}\!\left(Q(\mathbf{x}_{1:T}),\frac{1}{k}\sum_{i=1}^{k}{Q(\widehat{\mathbf{x}}_{1:T}^{\mathbf{z}_{i}})}\right)\!.\tag{11}
$$ 
>  我们借鉴 DGMR 的生成式集成预测，从多个隐输入 $\mathbf z_{1:k}$ 预测一组降水场 $\widehat {\mathbf x}_{1: T}^{\mathbf z_i}$，其中 $k$ 是集成成员数量
>  我们通过最大池化层 $Q$ 在空间维度上聚合真实场 $\mathbf x_{1:T}$ 和 $k$ 个预测 $\widehat {\mathbf x}_{1: T}^{\mathbf z_i}$，对应的池化层的 kernel 大小和步长分别是 5 和 2
>  我们定义池化正则化为真实观测的空间池化结果和 $k$ 个预测的空间池化结果的平均值之间的加权距离

Overall, the objective for training the generative network (Fig. 1a) is 

$$
J_{\mathrm{generative}}=\beta J_{\mathrm{adv}}+\gamma J_{\mathrm{cool}}.\tag{12}
$$ 
>  生成式网络的最终目标就是对抗损失和池化正则化项的加权求和

We set the number of ensemble members as $k\!=\!4$ , adversarial loss weight $\beta\!=\!6$ and pool-regularization weight $\gamma\!=\!20.$ . Similar to the evolution network, we set input length $T_{0}\,{=}\,9$ and output length $T\!=\!20$ . 
> 集成成员数量 $k$ 设定为 $4$，对抗损失权重 $\beta$ 设定为 $6$，池化正则化权重 $\gamma$ 设定为 $20$
> 和演化网络的训练一样，生成网络的训练也设定为 $T_0 = 9, T = 20$  

We use the Adam optimizer with a batch size of 16 and an initial learning rate of $3\times{10^{-5}}$ for the nowcast encoder, the nowcast decoder and the temporal discriminator and train the generative network for $5\times10^{5}$ iterations. 
>  优化器为 Adam，批量大小 16 (和演化网络一样)，初始学习率 $3\times 10^{-5}$ (比演化网络低两个数量级)，训练 $5\times 10^5$ 次迭代 (比演化网络多，但数量级相同)

**Transfer learning.** NowcastNet is a foundational model for skilful precipitation nowcasting. A large-scale dataset will help NowcastNet be more apt at learning physical evolution and chaotic dynamics of the precipitation processes. Therefore, in countries or regions with intricate atmosphere processes but without sufficient radar observations, we use the transfer learning strategy, a de facto way to reusing knowledge from pre-trained foundational models. Given a pre-trained NowcastNet model, we use the objectives $J_{\mathrm{evolution}}$ and $J_{\mathrm{generative}}$ to fine-tune its evolution network and generative network through decoupled back propagation, which detaches the gradients between $J_{\mathrm{evolution}}$ and $J_{\mathrm{generative}}$ . As the physical knowledge behind the precipitation is universal and transferable across the world, we decrease the learning rate of the evolution network as one-tenth that for the generative network to avoid forgetting of physical knowledge. We pre-train a NowcastNet model on a large-scale dataset and fine-tune it to a small-scale dataset with the Adam optimizer, but only for $2\times10^{5}$ iterations. 
>  迁移学习
>  NowcastNet 是用于降水临近预测的基石模型，大规模数据集将帮助 NowcastNet 更好学习降水过程的物理演化和混沌动力学
>  因此，对于大气过程复杂但雷达观测数据不足的地区，我们使用迁移学习策略，给定预训练好的 NowcastNet 模型，我们用目标 $J_{\mathrm{evolution}}$ 和 $J_{\mathrm{generative}}$ ，通过解耦的反向传播微调演化网络和生成网络，也就是分离了 $J_{\mathrm {evolution}}$ 和 $J_{\mathrm{generative}}$ 的梯度，独立优化两个网络
>  因为降水背后的物理知识是普遍且可迁移的，我们将演化网络的学习率设为生成网络学习率的十分之一，避免灾难性遗忘物理知识，微调的迭代次数限制为 $2\times 10^5$

**Hyperparameter tuning.** We use the mean of CSI neighborhood (CSIN) over all prediction time steps at the rain levels of $16\,\mathrm{mm\,h^{-1}}$ , $32\,\mathrm{mm\,h^{-1}}$ and $64\,\mathrm{mm\,h^{-1}}$ when tuning the hyperparameters of the evolution network. We compute the criterion for hyperparameter tuning as the average of the quantities, $\frac{\mathsf{C S I N}_{16}+\mathsf{C S I N}_{32}+\mathsf{C S I N}_{64}}{3}$ . When tuning the hyperparameters of the generative network, we use the two main evaluation metrics, CSI neighbourhood and PSD. For each model with different hyperparameters, we first ensure that the PSD of the model is no worse than that of pySTEPS. Then we use the average CSI neighbourhood criterion $\frac{\mathsf{C S I N}_{16}+\mathsf{C S I N}_{32}+\mathsf{C S I N}_{64}}{3}$ to determine the final hyperparameters. 
>  超参数调节
>  调节演化网络超参数时，使用不同降雨等级下所有时间步的平均 CSI 邻域作为指标，调节生成网络超参数时，使用平均 CSI 邻域和 PSD 作为指标，我们先保证模型的 PSD 指标不低于 pySTEPS 决定初始超参数，然后使用平均 CSI 邻域指标决定最终超参数

## Baselines 
We describe the four baselines used in the comparative study. There is a rich literature of relevant work and we discuss them as further background in Supplementary Information section E. 

**DGMR.** DGMR is a state-of-the-art method for precipitation nowcasting, recognized by expert meteorologists. We genuinely reproduce it taking exactly the same architecture and training settings described in ref. 4 and the released model files available at https://github.com/deepmind/deepmind-research/tree/master/nowcasting, with the quantitative and qualitative results to match those reported in the original paper. We set the number $k$ of ensemble members as 4 during training, which is the same as NowcastNet. 
>  DGMR 
>  我们严格按照参考文献 4 中描述的架构和训练设置，以及在 https://github.com/deepmind/deepmind-research/tree/master/nowcasting 上发布的模型文件，真实再现了该方法，并且定量和定性结果与原始论文中的结果一致。在训练过程中，我们将集成成员的数量 $k$ 设为 4，与 NowcastNet 相同。

**PredRNN-V2.** We consider PredRNN-V2 (ref. 13), the latest version of PredRNN 37 with a four-layer convolutional-recurrent network, deployed at the China Meteorological Administration for operational nowcasting. We cut radar fields into $4\times4$ patches and unfold the patches as the channel dimension, which efficiently balances the computation cost and forecasting skill. Reverse scheduled sampling with an exponential increasing strategy is applied in the first $5\times10^{4}$ iterations. 
>  PredRNN-V2
>  我们考虑 PredRNN-V2（参考文献 13），这是 PredRNN 的最新版本，包含一个四层卷积递归网络，并在中国气象局用于业务性临近预报。我们将雷达场切割成 $4\times4$ 的块，并将这些块展开为通道维度，从而有效地平衡计算成本和预报技能。前 $5\times10^{4}$ 次迭代中应用了带有指数增加策略的逆序调度采样。

**U-Net.** We use the improved version proposed by Ravuri et al. , which adds a residual structure in each block of the vanilla U-Net, along with a loss weighted by precipitation intensity, and predicts all fields in a single forward pass. 
>  U-Net
>  我们使用 Ravuri 等人提出的改进版本，该版本在原始 U-Net 的每个块中添加了残差结构，并且使用了由降水量加权的损失函数，在一次前向传递中预测所有场。

**pySTEPS.** We use the pySTEPS implementation from ref. 9, following the default settings available at https://github.com/pySTEPS/pysteps. 
>  pySTEPS
>  我们使用参考文献 9 中的 pySTEPS 实现，遵循默认设置。

All deep-learning models, including NowcastNet, DGMR, PredRNN-V2 and U-Net, are trained on the USA dataset (years 2016–2020) by the Adam optimizer with a batch size of 16 for $5\times10^{5}$ iterations and transferred to the China dataset by fine-tuning for $2\times10^{5}$ iterations. For all models under evaluation, we establish a fair comparison by using the same weighting scheme $w(x)$ in the weighted distance $L_{\mathrm{wdis}}$ and the same sampling strategy of training data. Both the weighting scheme and the sampling strategy are taken from DGMR. 
>  所有深度学习模型，包括 NowcastNet、DGMR、PredRNN-V2 和 U-Net，均使用 Adam 优化器，批量大小为 16，在美国数据集（2016 年至 2020 年）上训练 $5\times10^{5}$ 次迭代，并通过微调 $2\times10^{5}$ 次迭代迁移到中国数据集。
>  在评估的所有模型中，我们通过在加权距离 $L_{\mathrm{wdis}}$ 中使用相同的权重方案 $w(x)$ 和相同的训练数据采样策略，建立了公平的比较。权重方案和采样策略均来自 DGMR。

# Datasets 
Two large-scale, high-resolution datasets of composite radar observations from the USA and China are used throughout the experiments. The evaluation metrics are described in Supplementary Information section B. More case studies of representative precipitation events and quantitative results of overall performance are available in Extended Data Figs. 2–8 and Supplementary Information sections C and D. 

**USA dataset.** The USA dataset consists of radar observations from the MRMS system , collected over the USA. The radar composites cover the area from $20\,^{\circ}\mathrm{N}$ to $55\,^{\circ}\mathrm{N}$ in the south–north direction and $130\,^{\circ}\mathrm{W}$ to $60\,^{\circ}\mathrm{W}$ in the east–west direction. The spatial grid of the composites is $3{,}500\times7{,}000$ , with a resolution of $0.01^{\circ}$ per grid. The missing values on the composites are assigned negative values, which can mask unconcerned positions during evaluation. We use radar observations collected for a 6-year time range from 2016 to 2021, in which the training set covers years 2016–2020 and the test set covers the year 2021. We follow the strategy used in ref. 4 such that the radar observations from the first day of each month in the training set are included in the validation set. To trade off computational cost and forecasting skill, we set the temporal resolution as 10 min and downscale the spatial size of radar fields to half of the original width and height, which will keep the most of the convective-scale details. We cap the rain rates at the value of $128\,\mathsf{m m\,h^{-1}}$ . 
>  美国数据集
>  美国数据集包含 MRMS 系统在美国收集的雷达观测数据。雷达合成图覆盖了南北方向从 $20\,^{\circ}\mathrm{N}$ 到 $55\,^{\circ}\mathrm{N}$，东西方向从 $130\,^{\circ}\mathrm{W}$ 到 $60\,^{\circ}\mathrm{W}$ 的区域。
>  合成图的空间网格为 $3{,}500\times7{,}000$，每个网格的分辨率为 $0.01^{\circ}$。合成图上的缺失值被赋予负值，这可以在评估过程中屏蔽无关位置。
>  我们使用 2016 年至 2021 年的 6 年时间范围内的雷达观测数据，其中训练集涵盖 2016 年至 2020 年，测试集涵盖 2021 年。
>  我们遵循参考文献 4 中使用的策略，将训练集中每个月第一天的雷达观测数据纳入验证集。
>  为了平衡计算成本和预报技能，我们将时间分辨率设为 10 分钟，并将雷达场的空间尺寸缩小到原宽和原高的二分之一，这样可以保留大部分对流尺度的细节。我们将降雨率限制在 $128\,\mathsf{mm/h}$。

**China dataset.** The China dataset includes radar observations collected over China by the China Meteorological Administration. The radar composites cover the area from $17^{\circ}$  N to $53^{\circ}$  N in the south–north direction and $96^{\circ}$  E to $132^{\circ}$  E in the east–west direction, with a coverage of the middle and east of China. The spatial grid of the composites is $3{,}584\times3{,}584$ , with a resolution of $0.01^{\circ}$ per grid. Similar to the USA dataset, the missing values are replaced by negative values. We use radar observations collected for a nearly 2-year time range from 1 September 2019 to 30 June 2021. Data from 1 September 2019 to 31 March 2021 are taken as the training set, whereas those from 1 April 2021 to 30 June 2021 are taken as the test set. We follow the strategy used in ref. 4 such that the radar observations from the first day of each month in the training set are included in the validation set. Notably, the test period covers the flood season when extreme precipitation and rainstorms are frequent in China. We set the temporal resolution, spatial size and rain-rate threshold exactly the same as the USA dataset. 
>  中国数据集
>  中国数据集包括由中国气象局在中国收集的雷达观测数据。雷达合成图覆盖了南北方向从 $17^{\circ}\mathrm{N}$ 到 $53^{\circ}\mathrm{N}$，东西方向从 $96^{\circ}\mathrm{E}$ 到 $132^{\circ}\mathrm{E}$ 的区域，覆盖了中国中部和东部。
>  合成图的空间网格为 $3{,}584\times3{,}584$，每个网格的分辨率为 $0.01^{\circ}$。与美国数据集类似，缺失值被替换为负值。
>  我们使用了从 2019 年 9 月 1 日至 2021 年 6 月 30 日近 2 年的雷达观测数据。2019 年 9 月 1 日至 2021 年 3 月 31 日的数据作为训练集，而 2021 年 4 月 1 日至 2021 年 6 月 30 日的数据作为测试集。
>  我们遵循参考文献 4 中使用的策略，将训练集中每个月第一天的雷达观测数据纳入验证集。
>  值得注意的是，测试期涵盖了中国频繁发生极端降水和暴雨的洪水季节。我们设置的时间分辨率、空间尺寸和雨强阈值与美国数据集完全相同。

**Data preparation.** We construct the training set and test set for each dataset using an importance-sampling strategy to increase the ratio of radar series with heavy precipitation. We first crop the full-frame series into smaller s patio temporal size. For the training set, we cut the series into crops of spatial size $256\times256$ and temporal size $270\,\mathrm{{min}}$ with offsets of 32 in the vertical and horizontal directions. For the test set, we cut the series into crops of spatial size $512\times512$ and temporal size 270 min with offsets of 32 in the vertical and horizontal directions. 
>  数据准备
>  我们使用重要性采样策略构建每个数据集的训练集和测试集，以增加包含强降水的雷达序列的比例。
>  我们首先将全帧序列裁剪成更小的空间和时间尺寸。对于训练集，我们将序列裁剪成空间尺寸为 $256\times256$ 和时间尺寸为 $270\,\mathrm{min}$ 的片段，垂直和水平方向的偏移量为 32 (偏移量指裁剪窗口移动的步长，如果步长是 256，就是窗口没有重叠)。对于测试集，我们将序列裁剪成空间尺寸为 $512\times512$ 和时间尺寸为 $270\,\mathrm{min}$ 的片段，垂直和水平方向的偏移量同样为 32。

Then we give each crop an acceptance probability, 
>  然后，我们为每个裁剪片段赋予一个接受概率：

$$
\mathrm{P r}({\mathbf  x}_{-T_{0}:T})=\sum_{t=-T_{0}}^{T}\|\mathbf{g}({\mathbf  x}_{t})\|_{1}\!+\epsilon,\tag{13}
$$

which is the sum of radar fields for all grids and all time steps on this crop, and $\epsilon$ is a small constant. As done in DGMR, for the training set, we set ${\bf g}(x)\!=\!1\!-\!{e}^{-x}$ on each grid with a valid value and $\mathbf{g}(x)=0$ on each grid with a missing value. 
>  这是该片段内所有网格和所有时间步的雷达场之和，$\epsilon$ 是一个小常数。如同 DGMR 所做的那样，对于训练集，我们在每个有有效值的网格上设置 ${\bf g}(x)=1-e^{-x}$，而在每个缺失值的网格上设置 $\mathbf{g}(x)=0$。

We use hierarchical sampling during training, by first sampling the full-frame series and then sampling the crop series. To evaluate the forecasting skill of different models on extreme-precipitation events, we define $\mathbf{g}(x)\!=\!x$ for the test set. The test set is sampled in advance and kept unchanged throughout evaluation. As our goal is skilful nowcasting of extreme precipitation, this importance-sampling strategy is biased towards weather events with a larger proportion of heavy precipitation. 
>  在训练过程中，我们使用层次化采样，首先采样全帧序列，然后采样片段序列。
>  为了评估不同模型在极端降水事件中的预报技能，测试集上 $\mathbf g(x)$ 定义为 $\mathbf{g}(x)=x$。测试集提前采样并在整个评估过程中保持不变。由于我们的目标是对极端降水进行临近预报，这种重要性采样策略偏向于包含大量强降水的天气事件。

We also use the uniform-sampling protocol such that all light-to-heavy precipitation can be equally evaluated. In this protocol, the crops in the test set are sampled uniformly from all spatial and temporal ranges. Because the uniformly sampled series usually have scarce precipitation, we enlarge the dataset size to 288,000 for the USA case and 120,000 for the China case, three times larger than the importance-sampled test datasets. The quantitative results under this protocol are available in Supplementary Figs. 10 and 11. 
>  我们还使用均匀采样方法，以确保轻度到重度降水都能得到平等评估。此时测试集中的片段从所有空间和时间范围内均匀采样。
>  由于均匀采样得到的序列通常降水稀少，我们将美国案例的数据集大小扩大到 288,000，将中国案例的数据集大小扩大到 120,000，分别是重要性采样测试数据集的三倍大。此方案下的定量结果可在补充图 10 和 11 中找到。

# Evaluation 
We perform a meteorologist evaluation as a cognitive assessment task and a quantitative evaluation using operational verification measures. 

**Meteorologist evaluation.** To construct the test subsets representative of extreme-precipitation events for expert meteorologist evaluation, we first sample a new test set that contains the crops with spatial size of $512\times512$ using the same strategy detailed in the previous section. After this test set is sampled, we rank the crops by the sum of rain rate on all grids with rate higher than a threshold of $20\,\mathrm{mm\,h^{-1}}$ . This is the threshold of heavy rainfall used in operational practice by the China Meteorological Administration. We take the top 1,200 events as the subset for expert meteorologist evaluation.
>  气象学家评估
>  为了构建代表极端降水事件的测试子集以供专家气象学家评估，我们首先使用上一节详细说明的相同策略，采样一个新的测试集，该测试集包含空间尺寸为 $512\times512$ 的片段。
>  在采样完该测试集后，我们按照所有网格上雨强高于 $20\,\mathrm{mm/h}$ 的雨强总和对采样得到的片段进行排名。这是中国气象局在实际操作中用于识别强降雨的阈值。我们选取排名前 1,200 的事件作为专家气象学家评估的子集。

Because the test events are fewer, we change the strategy to ranking all events by the proportion of grids with a rate higher than $20\,\mathrm{mm\,h^{-1}}$ , which include extreme precipitation with very high probability, while ensuring the temporal diversity. On all crops in this test subset, all models take as input the fields of spatial size $512\times512$ , and the central $384\times384$ area of the predicted fields are zoomed in to highlight the convective details. 
>  由于这样选取出的测试事件较少，我们将策略改为按超过 $20\,\mathrm{mm/h}$ 雨强的网格比例对所有事件进行排名，这样可以有较高概率包含极端降水事件，同时保证时间多样性
>  在这个测试子集的所有片段上，所有模型都以 $512\times512$ 的空间尺寸作为输入，预测场的中心 $384\times384$ 区域被放大以突出对流细节。

To enable a professional, transparent and fair meteorologist evaluation, the China Meteorological Administration issued a public announcement to all provincial meteorological observatories, inviting senior meteorologists to participate in the evaluation as volunteers. The announcement states the content, goal and how-to of the expert evaluation, and specifically clarifies that the evaluation results will only be used anonymously for the scientific research but not for the skill test of meteorologists or other purposes. Operationally, we build an anonymous website for the meteorologist evaluation. Each expert logs in to the website using an automatically generated user account with password protection to perform the evaluation anonymously, without being informed of any model information. 
>  为了使专业、透明和公平的气象学家评估成为可能，中国气象局向所有省级气象观测站发布了公开公告，邀请资深气象学家作为志愿者参与评估。公告详细说明了专家评估的内容、目标和方法，并明确指出评估结果仅匿名用于科学研究，不会用于气象学家的技术测试或其他目的。
>  在实际操作中，我们建立了一个匿名网站来进行气象学家评估。每位专家使用自动分配的用户账户和密码保护登录网站，匿名进行评估，不会被告知任何模型信息。

In the posterior evaluation, we show real radar observations in the past and future horizons and the model predictions anonymously in random order for each event, whereas in the prior evaluation, we only show the real radar observations in the past. Meteorologists can play the video, navigate the progress bar to deliberately observe cloud evolution or arbitrarily stop the video at a certain time step for a meticulous comparison of the forecasting skill and value of all models. 
>  在后验评估中，我们随机顺序展示每个事件的过去和未来的真实雷达观测以及模型预测，而在先验评估中，我们仅展示真实雷达观测的过去情况。气象学家可以播放视频，导航进度条以观察云的发展，或在某个特定时间步任意停止视频，以仔细对比所有模型的预报技能和价值。

**Quantitative evaluation.** Evaluation with commonly used quantitative metrics involves comparing the difference between ground truths and model predictions on the crops in the test set. Each model outputs 18 future frames of precipitation fields given nine past frames of radar observations, whereas pySTEPS is given four past frames. Similar to the evaluation protocol of DGMR, the input spatial size is set as $512\times512$ for computing the PSD metric and as $256\times256$ for computing the other metrics. We apply the central-cropping technique, which crops $64\times64$ grid cubes from the central area of the 18 predicted frames, along with the corresponding ground truths. The PSD metric is directly computed on the $512\times512$ precipitation fields, whereas the other metrics are computed between the predicted and ground-truth cubes. The central cropping can eliminate the boundary influence and reduce the computation cost. For methods with ensemble-forecasting ability, including NowcastNet, DGMR and pySTEPS, we set the number $k$ of ensemble members as 4 for computing specific quantitative measures. 
>  定量评估
>  使用常用的定量指标进行评估涉及比较测试集中片段的真实值和模型预测值之间的差异。每个模型根据九个过去的雷达观测帧输出 18 个未来帧的降水场， pySTEPS 则根据四个过去的帧进行预测。
>  类似于 DGMR 的评估协议，输入空间尺寸设置为 $512\times512$ 用于计算 PSD 指标，设置为 $256\times256$ 用于计算其他指标。
>  我们应用中心裁剪技术，从 18 个预测帧的中心区域裁剪出 $64\times64$ 的网格立方体，连同对应的真实值。PSD 指标直接在 $512\times512$ 的降水场上计算，而其他指标则在预测和真实值的立方体之间计算。中心裁剪可以消除边界影响并减少计算成本。
>  对于具有集合预报能力的方法，包括 NowcastNet、DGMR 和 pySTEPS，我们设置集合成员数量 $k$ 为 4，用于计算特定的定量指标。

# Data availability 
The processed radar data that support the findings of this study are available on the Tsinghua Cloud with the accession code ‘nowcast’; see https://cloud.tsinghua.edu.cn/d/b9fb38e5ee7a4dabb2a6. A smaller dataset with the code for exploratory analysis is available on Code Ocean at https://doi.org/10.24433/CO.0832447.v1. 

The MRMS data that support the training of the nowcasting models for the USA weather system are available with agreement from the NOAA at https://www.nssl.noaa.gov/projects/mrms or contact the MRMS data teams using mrms@noaa.gov. 

The radar data that support the training of the nowcasting models for the China weather system are available from the China Meteorological Administration but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are available from the authors on reasonable request and with permission of the China Meteorological Administration. Source data are provided with this paper. 

# Code availability 
We rely on PyTorch (https://pytorch.org) for deep model training and cartopy (https://scitools.org.uk/cartopy) for geospatial data processing. We use specialized open-source tools for pySTEPS (https://pysteps.github.io), DGMR (https://github.com/deepmind/deepmind-research/tree/master/nowcasting), PredRNN-V2 (https://github.com/thuml/predrnn-pytorch) and SPADE (https://github.com/NVlabs/SPADE). The code of NowcastNet and the pre-trained neural-network weights are available on Code Ocean (https://doi.org/10.24433/CO.0832447.v1). 

# Appendix
## A. Lagrangian and Eulerian specification of the flow field
In [classical field theories](https://en.wikipedia.org/wiki/Classical_field_theory "Classical field theory"), the **Lagrangian specification of the flow field** is a way of looking at fluid motion where the observer follows an individual [fluid parcel](https://en.wikipedia.org/wiki/Fluid_parcel "Fluid parcel") as it moves through space and time. Plotting the position of an individual parcel through time gives the [pathline](https://en.wikipedia.org/wiki/Streamlines,_streaklines,_and_pathlines "Streamlines, streaklines, and pathlines") of the parcel. This can be visualized as sitting in a boat and drifting down a river.
>  经典场论中，流体运动可以通过两种方式描述：拉格朗日描述法和欧拉描述法
>  拉格朗日描述法关注单个流体质点的运动轨迹，观察者跟随一个特定的流体质点随时间和空间移动，记录质点经过的路径，即迹线
>  可以想象为坐在一艘顺流而下的船上，观察并记录船的路径

The **Eulerian specification of the flow field** is a way of looking at fluid motion that focuses on specific locations in the space through which the fluid flows as time passes. This can be visualized by sitting on the bank of a river and watching the water pass the fixed location.
>  欧拉描述法关注固定的空间位置，观察随着时间推移该位置上的流体属性如何变化
>  可以想象为坐在河岸边观察水流经过面前的固定某个点

The Lagrangian and Eulerian specifications of the flow field are sometimes loosely denoted as the **Lagrangian and Eulerian frame of reference**. However, in general both the Lagrangian and Eulerian specification of the flow field can be applied in any observer's [frame of reference](https://en.wikipedia.org/wiki/Frame_of_reference "Frame of reference"), and in any [coordinate system](https://en.wikipedia.org/wiki/Coordinate_system "Coordinate system") used within the chosen frame of reference. The Lagrangian and Eulerian specifications are named after [Joseph-Louis Lagrange](https://en.wikipedia.org/wiki/Joseph-Louis_Lagrange "Joseph-Louis Lagrange") and [Leonhard Euler](https://en.wikipedia.org/wiki/Leonhard_Euler "Leonhard Euler"), respectively.
>  拉格朗日和欧拉描述法也可以称为拉格朗日和欧拉参照系，但一般拉格朗日和欧拉描述法都可以应用在任何观察者的参考系以及在选定参考系的任何坐标系统中

These specifications are reflected in [computational fluid dynamics](https://en.wikipedia.org/wiki/Computational_fluid_dynamics "Computational fluid dynamics"), where "Eulerian" simulations employ a fixed [mesh](https://en.wikipedia.org/wiki/Types_of_mesh "Types of mesh") while "Lagrangian" ones (such as [meshfree simulations](https://en.wikipedia.org/wiki/Meshfree_methods "Meshfree methods")) feature simulation nodes that may move following the [velocity field](https://en.wikipedia.org/wiki/Velocity_field "Velocity field").
>  在计算流体力学中，欧拉模拟使用固定的网格结构进行数值计算，每个网格单元代表一个固定的空间位置，拉格朗日模拟中节点随着速度场移动，不依赖于固定的网格结构，也被称为无网格方法

### A.1. History
[Leonhard Euler](https://en.wikipedia.org/wiki/Leonhard_Euler "Leonhard Euler") is credited of introducing both specifications in two publications written in 1755 and 1759. [Joseph-Louis Lagrange](https://en.wikipedia.org/wiki/Joseph-Louis_Lagrange "Joseph-Louis Lagrange") studied the equations of motion in connection to the [principle of least action](https://en.wikipedia.org/wiki/Action_principles "Action principles") in 1760, later in a treaty of fluid mechanics in 1781, and thirdly in his book _[Mécanique analytique](https://en.wikipedia.org/wiki/M%C3%A9canique_analytique "Mécanique analytique")_. In this book Lagrange starts with the Lagrangian specification but later converts them into the Eulerian specification. 

### A.2. Description
In the _Eulerian specification_ of a [field](https://en.wikipedia.org/wiki/Field_(physics) "Field (physics)"), the field is represented as a function of position $\mathbf x$ and time $t$. For example, the [flow velocity](https://en.wikipedia.org/wiki/Flow_velocity "Flow velocity") is represented by a function

$$
\mathbf u = (\mathbf x, t).
$$

>  欧拉描述法中，场表示为位置 $\mathbf x$ 和时间 $t$ 的函数，例如流速的表示如上

On the other hand, in the _Lagrangian specification_, individual fluid parcels are followed through time. The fluid parcels are labelled by some (time-independent) vector field $\mathbf x_0$. (Often, $\mathbf x_0$ is chosen to be the position of the center of mass of the parcels at some initial time $t_0$. It is chosen in this particular manner to account for the possible changes of the shape over time. Therefore, the center of mass is a good parameterization of the flow velocity $\mathbf u$ of the parcel.) In the Lagrangian description, the flow is described by a function

$$
\mathbf X(\mathbf x_0,t),
$$

giving the position of the particle labeled $\mathbf x_0$ at time $t$.

>  拉格朗日描述法中，我们随时间跟从单个流体质点，该流体质点用 $\mathbf x_0$ 标记，$\mathbf x_0$ 是与时间无关的向量场，一般选择为在初始时间 $t_0$ 时该质点的位置
>  拉格朗日描述将流体描述为关于 $\mathbf x_0$ 的 $t$ 的函数，也就是关心在时间 $t$ 时，用 $\mathbf x_0$ 标记的流体质点所在位置是哪里

The two specifications are related as follows:

$$
\mathbf u\left(\mathbf X(\mathbf x_0,t),t\right) = \frac {\partial \mathbf X}{\partial t}(\mathbf x_0,t),
$$

because both sides describe the velocity of the particle labeled $\mathbf x_0$ at time $t$.

>  二者可以通过以上方式转化，左边的 $\mathbf u\left(\mathbf X(\mathbf x_0, t), t\right)$ 使用欧拉描述法，给出了时空位置 $(\mathbf X(\mathbf x_0, t), t)$ 处的流速，右边的 $\frac {\partial \mathbf X}{\partial t}(\mathbf x_0, t)$ 使用拉格朗日描述法，给出了质点在 $\mathbf X(\mathbf x_0, t)$ 处的位置变化率，也就是该粒子的速度
>  两边都描述了粒子 $\mathbf x_0$ 在时间 $t$ 的速度，因此相等

Within a chosen coordinate system, $\mathbf x_0$ and $\mathbf x$ are referred to as the **Lagrangian coordinates** and **Eulerian coordinates** of the flow respectively.
>  在给定的坐标系统中 $\mathbf x_0$ 和 $\mathbf x$ 分别称为流体的拉格朗日坐标和欧拉坐标

### A.3. Material derivative
Main article: [Material derivative](https://en.wikipedia.org/wiki/Material_derivative "Material derivative")

The Lagrangian and Eulerian specifications of the [kinematics](https://en.wikipedia.org/wiki/Kinematics "Kinematics") and [dynamics](https://en.wikipedia.org/wiki/Dynamics_(physics) "Dynamics (physics)") of the flow field are related by the [material derivative](https://en.wikipedia.org/wiki/Material_derivative "Material derivative") (also called the Lagrangian derivative, convective derivative, substantial derivative, or particle derivative).
>  拉格朗日和欧拉描述下的流场的运动学和动力学通过材料导数关联

Suppose we have a flow field $\mathbf u$, and we are also given a generic field with Eulerian specification $\mathbf F(\mathbf x, t)$. Now one might ask about the total rate of change of $\mathbf F$ experienced by a specific flow parcel. This can be computed

$$
\frac {\mathrm D\mathbf F}{\mathrm D t} = \frac {\partial \mathbf F}{\partial t} + (\mathbf u \cdot \nabla)\mathbf F,
$$

 where $\nabla$ denotes the [nabla](https://en.wikipedia.org/wiki/Del "Del") operator with respect to $\mathbf x$, and the operator $\mathbf u \cdot \nabla$ is to be applied to each component of $\mathbf F$. This tells us that the total rate of change of the function $\mathbf F$ as the fluid parcels moves through a flow field described by its Eulerian specification $\mathbf u$ is equal to the sum of the local rate of change and the convective rate of change of $\mathbf F$. This is a consequence of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule "Chain rule") since we are differentiating the function $\mathbf F(\mathbf X(\mathbf x_0, t), t)$ with respect to $t$.

>  给定流场 $\mathbf u$，同时给定一个欧拉描述的一般场 $\mathbf F(\mathbf x, t)$，流体粒子经历的 $\mathbf F$ 的总变化率如上计算
>  其中 $\nabla$ 是相对于空间坐标 $\mathbf x$ 的梯度算子
>  该表达式说明了在流体粒子随着流场 $\mathbf u$ 运动下，$\mathbf F$ 的总变化率等于局部变化率和对流变化率的和
>  这实际上就是在对函数 $\mathbf F(\mathbf X(\mathbf x_0, t), t)$ 相对于 $t$ 微分时应用了链式法则，因为流体粒子的位置 $\mathbf X(\mathbf x_0, t)$ 本身也随着时间变化

[Conservation laws](https://en.wikipedia.org/wiki/Conservation_law "Conservation law") for a unit mass have a Lagrangian form, which together with mass conservation produce Eulerian conservation; on the contrary, when fluid particles can exchange a quantity (like energy or momentum), only Eulerian conservation laws exist. 
>  单位质量的守恒定律采取拉格朗日形式，因为它考虑的是单个粒子质量不变
>  整个流体域的质量守恒则采取欧拉形式，如果流体粒子之间可以交换物理量，例如能量或动量，只有欧拉形式存在，因为欧拉形式考虑的是整个流体系统内的质量守恒

## B. Material Derivative
In [continuum mechanics](https://en.wikipedia.org/wiki/Continuum_mechanics "Continuum mechanics"), the **material derivative** describes the time [rate of change](https://en.wikipedia.org/wiki/Derivative "Derivative") of some physical quantity (like [heat](https://en.wikipedia.org/wiki/Heat "Heat") or [momentum](https://en.wikipedia.org/wiki/Momentum "Momentum")) of a [material element](https://en.wikipedia.org/wiki/Material_element "Material element") that is subjected to a space-and-time-dependent [macroscopic velocity field](https://en.wikipedia.org/wiki/Flow_velocity "Flow velocity"). The material derivative can serve as a link between [Eulerian](https://en.wikipedia.org/wiki/Continuum_mechanics#Eulerian_description "Continuum mechanics") and [Lagrangian](https://en.wikipedia.org/wiki/Continuum_mechanics#Lagrangian_description "Continuum mechanics") descriptions of continuum [deformation](https://en.wikipedia.org/wiki/Deformation_(mechanics) "Deformation (mechanics)").
>  连续介质力学中，材料导数描述了一个受时空依赖的宏观速度场影响的物质元素的某个物理量的时间变化率
>  材料导数可以作为欧拉描述和拉格朗日描述之间的桥梁

For example, in [fluid dynamics](https://en.wikipedia.org/wiki/Fluid_dynamics "Fluid dynamics"), the velocity field is the [flow velocity](https://en.wikipedia.org/wiki/Flow_velocity "Flow velocity"), and the quantity of interest might be the [temperature](https://en.wikipedia.org/wiki/Temperature "Temperature") of the fluid. In this case, the material derivative then describes the temperature change of a certain [fluid parcel](https://en.wikipedia.org/wiki/Fluid_parcel "Fluid parcel") with time, as it flows along its [pathline](https://en.wikipedia.org/wiki/Streamlines,_streaklines,_and_pathlines "Streamlines, streaklines, and pathlines") (trajectory).

### B.1. Other names
There are many other names for the material derivative, including:

- **advective derivative**
- **convective derivative**
- **derivative following the motion**
- **hydrodynamic derivative**
- **Lagrangian derivative**
- **particle derivative**
- **substantial derivative**
- **substantive derivative**
- **Stokes derivative**
- **total derivative**, although the material derivative is actually a special case of the [total derivative](https://en.wikipedia.org/wiki/Total_derivative "Total derivative") 

### B.2. Definition
The material derivative is defined for any [tensor field](https://en.wikipedia.org/wiki/Tensor_field "Tensor field") $y$ that is _macroscopic_, with the sense that it depends only on position and time coordinates, $y = y(\mathbf x, t)$:

$$
\frac {\mathrm D y}{\mathrm D t} \equiv\frac {\partial y}{\partial t}+\mathbf u\cdot \nabla y,
$$

where $\nabla y$ is the [covariant derivative](https://en.wikipedia.org/wiki/Covariant_derivative "Covariant derivative") of the tensor, and $\mathbf u(\mathbf x, t)$ is the [flow velocity](https://en.wikipedia.org/wiki/Flow_velocity "Flow velocity"). 

>  材料导数适用于任何宏观张量场 $y$，宏观指 $y$ 仅依赖于位置和时间坐标，即 $y = y(\mathbf x, t)$
>  张量场 $y$ 的材料导数定义如上，其中 $\nabla y$ 是张量的协变导数，$\mathbf u(\mathbf x, t)$ 是流速场
>  $\frac {\partial y}{\partial t}$ 表示了张量场 $y$ 在固定位置处的时间变化率，对流项 $\mathbf u\cdot \Delta y$ 表示由于流体运动引起的张量场变化

Generally the convective derivative of the field $\mathbf u\cdot \nabla y$, the one that contains the covariant derivative of the field, can be interpreted both as involving the [streamline](https://en.wikipedia.org/wiki/Streamline_(fluid_dynamics) "Streamline (fluid dynamics)") [tensor derivative](https://en.wikipedia.org/wiki/Tensor_derivative_(continuum_mechanics) "Tensor derivative (continuum mechanics)") of the field $\mathbf u \cdot \nabla y$, or as involving the streamline [directional derivative](https://en.wikipedia.org/wiki/Directional_derivative "Directional derivative") of the field $(\mathbf u\cdot \nabla) y$, leading to the same result. Only this spatial term containing the flow velocity describes the transport of the field in the flow, while the other describes the intrinsic variation of the field, independent of the presence of any flow. Confusingly, sometimes the name "convective derivative" is used for the whole material derivative $D/Dt$, instead for only the spatial term $\mathbf u \cdot \nabla$. The effect of the time-independent terms in the definitions are for the scalar and tensor case respectively known as [advection](https://en.wikipedia.org/wiki/Advection "Advection") and convection.
>  通常，场的对流导数 $\mathbf u \cdot \nabla y$ 可以解释为场的流线张量导数，写为 $\mathbf u\cdot \nabla y$，或者解释为流线方向导数，写为 $(\mathbf u\cdot \nabla) y$，二者等价

## C. Semi-Lagrangian scheme
The **Semi-Lagrangian scheme** (SLS) is a [numerical method](https://en.wikipedia.org/wiki/Numerical_method "Numerical method") that is widely used in [numerical weather prediction](https://en.wikipedia.org/wiki/Numerical_weather_prediction "Numerical weather prediction") models for the integration of the equations governing atmospheric motion. A [Lagrangian](https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field "Lagrangian and Eulerian specification of the flow field") description of a system (such as the [atmosphere](https://en.wikipedia.org/wiki/Atmosphere "Atmosphere")) focuses on following individual air parcels along their trajectories as opposed to the [Eulerian](https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field "Lagrangian and Eulerian specification of the flow field") description, which considers the rate of change of system variables fixed at a particular point in space. A semi-Lagrangian scheme uses Eulerian framework but the discrete equations come from the Lagrangian perspective.
>  半拉格朗日方案是广泛用于数值天气预报模型中的数值方法，半拉格朗日方案使用欧拉框架，但离散方程从拉格朗日视角构建

### C.1. Some background
The Lagrangian rate of change of a quantity $F$ is given by

$$
\frac {\mathrm D F}{\mathrm D t} = \frac {\partial F}{\partial t} + (\mathbf v\cdot \nabla)F
$$

where $F$ can be a scalar or vector field and $\mathbf v$ is the velocity field. The first term on the right-hand side of the above equation is the _local_ or _Eulerian_ rate of change of $F$ and the second term is often called the _advection term_. Note that the Lagrangian rate of change is also known as the [material derivative](https://en.wikipedia.org/wiki/Material_derivative "Material derivative").

>  量 $F$ 的拉格朗日变化率 (即物质导数) 如上所示，$F$ 可以是标量场也可以是矢量场，$\mathbf v$ 是速度场
>  RHS 的第一项是 $F$ 的局部或欧拉变化率，第二项称为平流项

It can be shown that the equations governing atmospheric motion can be written in the Lagrangian form

$$
\frac {\mathrm D \mathbf V}{\mathrm D t} = \mathbf S(\mathbf V)
$$

where the components of the vector $\mathbf V$ are the (dependent) variables describing a parcel of air (such as velocity, pressure, temperature etc.) and the function $\mathbf S(\mathbf V)$ represents source and/or sink terms.

>  大气运动的控制方程可以以拉格朗日形式表示
>  其中向量 $\mathbf V$ 的成分是描述空气粒子的相互依赖的变量，例如速度、压力、温度
>  函数 $\mathbf S(\mathbf V)$ 表示源/汇项，即影响这些变量变化的因素，源项表示产生，汇项表示消耗

In a Lagrangian scheme, individual air parcels are traced but there are clearly certain drawbacks: the number of parcels can be very large indeed and it may often happen for a large number of parcels to cluster together, leaving relatively large regions of space completely empty. Such voids can cause computational problems, e.g. when calculating spatial derivatives of various quantities. There are ways round this, such as the technique known as [Smoothed Particle Hydrodynamics](https://en.wikipedia.org/wiki/Smoothed_Particle_Hydrodynamics "Smoothed Particle Hydrodynamics"), where a dependent variable is expressed in non-local form, i.e. as an integral of itself times a kernel function.
>  纯拉格朗日方案中，每个空气粒子都要追踪，需要追踪的粒子数量过多，同时空气粒子可能在某些区域聚集，空间中大量区域是空洞，会引起计算上的问题

Semi-Lagrangian schemes avoid the problem of having regions of space essentially free of parcels.

### C.2. The Semi-Lagrangian scheme
Semi-Lagrangian schemes use a regular (Eulerian) grid, just like finite difference methods. The idea is this: at every time step the point where a parcel originated from is calculated. An interpolation scheme is then utilized to estimate the value of the dependent variable at the grid points surrounding the point where the particle originated from. The references listed contain more details on how the Semi-Lagrangian scheme is applied.
>  半拉格朗日方案使用规则的欧拉网格进行计算，类似有限差分方法
>  其核心思想是：在每一步时间步，计算网格点上空气粒子的出发点 (该出发点不一定刚刚好在某个网格点上)，然后使用插值方案，利用该出发点周围网格点的值估计该出发点的值，进而估计到达网格点上的值

Semi-Lagrangian 方案的介绍详细见 'Atmospheric Modeling, Data Assimilation and Predictability' Chapter 3.3.3