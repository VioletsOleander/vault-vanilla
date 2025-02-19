# A Ablation Study of NowcastNet
NowcastNet is a physics-conditional deep generative model, comprised of the generative network and the evolution network. We provide an extensive ablation study to investigate the effect and behavior of different modules in NowcastNet, with a focus on the major designs of the evolution network and the physics-conditioning mechanism between the generative network and the evolution network. The ablation results in reference to the ground truth observations and PySTEPS are shown in Supplementary Fig. 1.

## A.1 Physics-conditioning mechanism
We first inspect the physics-conditioning mechanism in NowcastNet, which adaptively controls the integration of physical knowledge from the evolution network into the generative network. Besides the default conditioning mechanism in NowcastNet, we investigate another two natural alternatives: weak conditioning and no conditioning.
>  NowcastNet 中的物理条件机制适应性地控制了演化网络地物理知识对生成网络地融入，我们将物理条件机制和弱条件以及无条件进行比较

In the weak conditioning model, we remove from NowcastNet the evolution conditioning module (Fig. 1a), which enables fuller integration of the evolution network predictions to multiple layers of the generative network; Namely, we simply take the predictions as another input to the U-Net based nowcast decoder. In principle, U-Net employs shortcut connections from the bottom layers to the top layers to deliver multiscale features, making the generative network weakly conditioned on the evolution network. 
>  弱条件模型中，NowcastNet 的演化条件模块被移除，该模块在 NowcastNet 中负责将演化网络的预测进一步融合入生成网络的各个层，此时演化网络的输出仅仅是作为 Nowcast decoder 的另一个输入
>  Nowcast decoder 的结构基于 U-Net，U-Net 从底部层到顶部层都采用了残差连接以传递多尺度特征，故此时生成式解码器仍然是弱条件于演化网络的输出的

As shown in Supplementary Fig. 1, while the location of the predicted squall line by the weak conditioning model (3rd row) is relatively consistent with the original NowcastNet, the intensity of the squall line is not kept after two hours. Worse still, there are undesired extra noises in the predictions that are out of the scope of the evolution network predictions.
>  如 Supplementary Fig 1 所示，弱条件模型预测的 squall line 的位置和原模型是相对一致的，但它的强度在两小时后并未保持
>  另外，弱条件模型的预测中还有不在演化网络的预测范围内的额外噪声

In the no conditioning model, we remove the whole evolution network from NowcastNet, but keep all other training settings unchanged. As shown in Supplementary Fig. 1, the no conditioning model (4th row) exhibits a large tendency of dissipating the heavy precipitation and violating the cloud dynamics, thereby generating unusable predictions after one hour. This is a strong evidence that the physics-conditioning on the evolution network informed by physical knowledge is important to make NowcastNet skillful for extreme precipitation nowcasting.
>  无条件模型移除了整个演化网络，其他训练设定保持不变
>  如 Supplementary Fig 1 所示，无条件模型的预测表现了强降水的消散同时违反云动力学的强烈趋势，因此它对一个小时之后的预测就是没有用的
>  这说明了条件于基于物理知识的演化网络对于 NowcastNet 的极端降水预测能力是很重要的

## A.2 Major designs of evolution network
We inspect the major designs of the evolution network, and justify that each design is crucial to the skillful performance. As shown in Supplementary Fig. 1, by comparing NowcastNet (2nd row) and the evolution network (5th row), we can clearly observe that the evolution network can only produce mesoscale nowcasts at 20 km scale. This is due to its deterministic nature, although it is designed to be a nonlinear physical evolution scheme that is more powerful than the linear PySTEPS. We further boost the deterministic evolution network by the stochastic generative network, which makes the final NowcastNet able to generate convective details at 1-2 km scale.
>  如 Supplementary Fig 1 所示，比较 NowcastNet 和演化网络可以发现，演化网络只能生成 20 km 尺度的中尺度预测，因为它生成预测的本质是确定性的，虽然网络本身的设计目的是作为一个非线性物理演化方案
>  NowcastNet 基于演化网络，进一步使用随机性的生成网络，最终能够生成 1-2km 尺度的对流细节

We first consider a weaker version of the evolution network that does not adopt the accumulation loss. Instead, the weaker evolution network tries to predict the ground truth motion fields and intensity residues, by using as its objective the distance between ground truth $\hat {\pmb x}_{t+1}$ and the result of evolution on ground truth $\hat {\pmb x}_t$ which is not available at inference. Note that in our final accumulation loss, we adopt the distance between ground truth $\hat {\pmb x}_{t+1}$ and predicted field $\hat {\pmb x}_{t+1}''$, the result of evolution on previous prediction $\hat {\pmb x}_t''$ , which exactly complies with the autoregressive nature of the physical evolution process. The other training settings are exactly the same as the original evolution network.
>  考虑一个更弱的演化网络，它不使用累计误差训练，而是直接基于真实观测 $\hat {\pmb x}_t$，预测下一时刻图像 $\hat {\pmb x}_{t+1}$，该演化网络没有分别预测出强度场和运动场，再通过基于物理规律的演化进行自回归预测，我们期望它能够自己拟合出这一能力
>  其他的训练设定和原来的演化网络一致

As shown in Supplementary Fig. 1, the weaker evolution network (6th row) produces predictions with serious and unacceptable location bias. We also find that the location bias has moderate similarity with the PySTEPS predictions (10th row). In particular, the locations of the head and tail of the squall line predicted by the weaker evolution network and PySTEPS are fairly consistent. This is a strong evidence that PySTEPS lacks an end-to-end error optimization framework to explicitly control the accumulated error in the autoregressive evolution process.
>  如 Supplementary Fig 1 所示，更弱的演化网络的预测带有严重的位置偏差，同时这些位置偏差和 PySTEPS 的预测中的位置偏差存在一定的相似性，这表明 PySTEPS 缺乏端到端的误差优化框架以显示地控制自回归演化过程中累计误差

In contrast, the original evolution network (5th row) yields significantly better predictions of the motion and the shape of the squall line. The high nowcast skill stems from its capability of integrating the physical evolution scheme into a differentiable neural evolution operator (Fig. 1c) and empowering a fully neural framework with nonlinear simulations and end-to-end error optimization (Fig. 1b).
>  相交之下，原来的演化网络的运动场预测和 squall line 的形状都更好，这源于它将物理演化方法融入了可微的神经演化算子中，进而可以执行端到端的误差优化和非线性拟合

Second, we consider another weaker version of the evolution network by dropping the intensity-residual component. As shown in Supplementary Fig. 1, the motion-only evolution network (7th row) is better able to provide plausible location predictions consistent with the ground truth. This confirms the evolution network, benefiting from its particular designs (Fig. 1b), achieves essential improvements over PySTEPS.
>  再考虑另一种更弱的演化网络，它移除了强度残差成分，仅考虑了运动场
>  如 Supplementary Fig 1 所示，仅考虑运动场的演化网络的预测的位置和实际更一致，这表明了演化网络的设计使得它优于 PySTEPS

Third, we consider another weaker version of the evolution network by dropping the motion regularization term (8th row). This term is defined by an assumption in the motion field estimation that all neighbor points have similar motions, which is motivated in part by the continuity equation and in part by the fact that heavy precipitation patterns tend to be longer-lived and stabler in movement [2]. As shown in Supplementary Fig. 1 (8th row), the squall line predicted by this weaker version becomes unnaturally distorted after two hours. Note that the motion regularization term cannot be used independently because the accumulation loss to control the forecast error is the pillar objective of our physical evolution scheme.
>  再考虑另一种更弱的演化网络，它移除了运动正则化项，定义该项的理由是假设了临近点具有相似的运动场，该假设一部分源于连续性方程，一部分源于强降水模式往往在移动中更加持久且稳定
>  如 Supplementary Fig 1 所示，该网络预测的 squall line 在 2h 后不自然地扭曲

Finally, we inspect the last weaker version of the evolution network by eliminating the stop-gradient mechanism in the neural evolution operator (Fig. 1c), which is originally designed to mitigate the numerical instability. As shown in Supplementary Fig. 1, the weaker network without stopping the gradients (9th row) incurs serious location bias and unnatural dissipation. This verifies the need of the stop-gradient design for the neural evolution operator, which underlies the stable learnability of NowcastNet.
>  最后考虑一种更弱的演化网络，它移除了神经演化算子的梯度停止机制，这一机制最初设计于缓解数值稳定性
>  如 Supplementary Fig 1 所示，该网络的预测出现了位置偏差和不自然的消散，这说明了梯度停止机制的必要性，它是网络稳定学习能力的基础

# B Evaluation metrics
We first describe the four standard evaluation metrics used in this article.

**Critical Success Index (CSI)** [3] measures the accuracy of the binary decision induced by the predicted field whether the rain rate exceeds a particular threshold τ . It is computed by the ratio of the hit on the rain rate exceeding the threshold in the sum of hit, miss and false alarm on all grids [4]. It considers the precision and the recall simultaneously, and is a common choice for evaluation in precipitation nowcasting. It is noteworthy that during computing the CSI metric, the hit, miss and false alarm are counted over all the test set.
>  CSI (关键成功指数) 度量了预测场的二元决策的准确性，二元决策判断即降水量是否超过一定阈值 $\tau$
>  CSI 基于降雨量超过阈值的比率的命中网格数占命中网格数+漏报网格数+虚警网格数的比率计算，CSI 还同时考虑了精确度和召回率
>  CSI 是评估降水临时预测的常见指标，在计算 CSI 时，命中、漏报、虚警的数量基于整个测试集计算

**CSI-Neighborhood** [4] is the CSI metric calculated based on the neighborhood. The neighborhood methods evaluate the forecasts within a spatial window surrounding each grid, and can enable verification of how “close” the forecasts are, which are particularly suited for verifying high-resolution forecasts. CSI-Neighborhood calculates the counts of hit, miss and false alarm after max pooling with kernel size $\kappa$ and stride $\lfloor \kappa \rfloor$.
>  CSI-Neighborhood 是根据邻域计算的 CSI 指标，邻域方法评估每个网格周围的空间窗口中的预测，适用于高分辨率预测
>  CSI-Neighborhood 在进行最大池化操作后计算命中、漏报和虚警的数量

**Fractional Skill Score (FSS)** [5] is another metric based on the neighborhood, defined by a rain rate threshold τ . For every grid in the test dataset, we compute the fraction of surrounding grids within a spatial window that have exceeded the threshold. Then we sum the difference between the fractions of predictions and observations for all grids in the dataset, referred to as Fractional Brief Score (FBS). FSS is calculated from the normalized FBS, relying on τ to decide whether it rains on a local grid and use the Fractions Brier Score (FBS) to compare the forecasted and observed rain frequencies. Compared to CSI-Neighborhood that focuses on whether the prediction makes successful hits, FSS can further compare the ratio of grids that exceed the threshold in the spatial windows.
>  FSS (分数技巧评分) 同样是基于邻域的度量，并且由降雨率阈值 $\tau$ 定义
>  对于测试集中的每个网格，计算其周围网格超过 $\tau$ 的比率，并记录该比例值和对应的真实比例值的差异，将数据集中所有这样的差值相加，就得到了分数简要评分 FBS，FSS 就是规范化的 FBS
>  CSI-Neighborhood 聚焦于预测是否成功命中，FSS 则进一步比较了空间窗口中超过阈值的网格比率

**Power Spectral Density (PSD)** [6, 7] measures the power distribution over each spatial frequency, comparing the precipitation variability of forecasts to that of the observations. We adopt the implementation of PSD from the PySTEPS package [8]. Forecasts that have more minor differences with observations are less preferred.
>  功率谱密度 (PSD) 度量了每个空间频率上的功率分布，比较预测的降水变化性和真实的降水变化性
>  PSD 的实现来自于 PySTEPS 包，观测和预测的 PSD 差异越小越好

# C Additional precipitation events
We study additional cases covering both extreme and ordinary precipitation events in the USA and China, which are selected from the test set with the help of five chief forecasters at the China Meteorological Administration to be representative events in the periods for evaluation. Selected precipitation events of 2021 in the USA are shown in Extended Data Fig. 2–6 and Supplementary Fig. 2–5, including the event with tornado on March 25 1, the event with a massive squall line on May 4 2, the event with convective initiation and dissipation on August 14, the remnants of Hurricane Ida on September 1 3, the event with tornado outbreak on December 11 4, the event with widespread precipitation on April 11, the event with multiple supercells on April 24 5, the event with isolated cells on June 12, and the event with a linear storm system on July 14 6. Selected precipitation events from April to June 2021 in China are shown in Extended Data Fig. 7–8, including the hail event caused by a squall line on May 3 and the gale-warning event along with hail and lightning on June 30.
>  我们还研究了额外的具有代表性的极端和普通降水事件，它们都选自于测试集
>  2021 年美国选定的降水事件如扩展数据图 2-6 和补充图 2-5 所示，包括 3 月 25 日的龙卷风事件、5 月 4 日的大飑线事件、8 月 14 日的对流起始和消散事件、9 月 1 日飓风“艾达”残余事件、12 月 11 日的龙卷风爆发事件、4 月 11 日的大范围降水事件、4 月 24 日的多个超级单体事件、6 月 12 日的孤立单元事件以及 7 月 14 日的线性风暴系统事件。2021 年 4 月至 6 月期间中国选定的降水事件如扩展数据图 7-8 所示，包括 5 月 3 日由飑线引起的冰雹事件和 6 月 30 日伴随冰雹和闪电的大风预警事件。

We further demonstrate in Extended Data Fig. 9–10 the high-resolution precipitation nowcasts over a larger spatial region of 2048 km × 2048 km, corresponding to the USA and China precipitation events in Fig. 2–3 of the main text, respectively.

Videos of the selected precipitation events in the main text, the Extended Data and the Supplementary Information, and the corresponding three-hour nowcasts of different models are given in Supplementary Movie 1–13.

Showcases of the randomly-sampled events used in the meteorologist evaluation with the expert choices are shown in Supplementary Fig. 6–7 for the USA events and in Supplementary Fig. 8–9 for the China events.

# D Additional quantitative results
We then present additional quantitative results on the uniform-sampled test set in Supplementary Fig. 10–11, in which we include results of the evolution network termed as NowcastNet-evo. In addition to the quantitative results shown in the main text, we present additional results in the other metrics on the importance-sampled USA test set in Supplementary Fig. 12– 14, and that on the importance-sampled China test set in Supplementary Fig. 15–17.
>  我们还展示了在均与采样的测试集上、USA 测试集的重要性采样集合、China 测试集的重要性采样集合上的额外定量结果

We only present the results of NowcastNet with the strongest baselines, DGMR and PySTEPS. NowcastNet is comparable to DGMR at the rain rates below 8 mm/h but is preferable at higher rain rates. The quantitative results verify that NowcastNet is generally useful for light-to-heavy precipitation and particularly skillful for extreme precipitation. As the neighborhood size in these metrics gets larger, i.e., higher focus on the physical plausibility [9], the advantage of NowcastNet becomes more prominent.
>  在降雨率低于 8 毫米/小时的情况下，NowcastNet 与 DGMR 相当，但在更高的降雨率下更优
>  定量结果显示，NowcastNet 对于轻到重的降水总体上是有用的，并且对于极端降水特别有效。随着这些指标中的邻域大小增加，即对物理合理性的关注更高时，NowcastNet 的优势变得更加明显。

# E Related work
We elaborate a more extensive review on the development of precipitation nowcasting methods, and highlight their differences to NowcastNet.

In advection scheme-based methods, Lagrangian persistence is a basic paradigm [10], where the current state of the motion field is first estimated by optical flow methods and then used to advect the current state of radar reflectivity field for future time steps. Since the spatial scale of the precipitation patterns largely determines the temporal duration of its persistence [10], multiscale methods such as S-PROG [11] with Fourier decomposition or MAPLE [12] with wavelet decomposition are further proposed to predict the stochastic intensity evolution in the advection process. Another line of work considers the spectral method that solves for the motion and intensity fields from the continuity equation by Fourier transform [13, 14]. STEPS [15, 16] models both the stochastic intensity evolution and the stochastic perturbation of the motion fields, which is widely deployed as a strong precipitation nowcasting model. Many operational methods of the advection scheme are implemented in the popular open-source packages such as PySTEPS [8] or Rainymotion [17].
>  基于平流方案的方法中，拉格朗日持续模型是基本范式，该模型中首先通过光流方法预测动作场的当前状态，然后使用运动场对雷达反射场进行平流以得到未来时间步的预测
>  因为降水模式的空间尺度很大程度上决定了其持续的时间，故许多多尺度方法被提出以预测平流过程中随机强度的变化
>  另一类方法考虑谱方法，通过 Fourier 变换从连续方程中求解运动场和强度场
>  STEPS 同时建模了随机强度演化和对运动场的随机扰动，该模型被广泛部署，许多业务性的平流方法都基于开源包如 PySTEPS 实现

The first deep architecture designed for precipitation nowcasting is Convolutional LSTM [18], which unifies convolutional and recurrent structures for spatiotemporal prediction. This model was extensively improved later from multiple aspects, such as introducing the predictive coding structure [19], adding spatial-temporal highway [20, 21] or involving learnable feature deformations in the prediction process [22, 23]. These methods are used as common baselines for data-driven precipitation nowcasting [24–26]. UNet [27], a general multiscale backbone widely used in Artificial Intelligence for Science (AI4Science), is modified for precipitation nowcasting [28, 29]. Another series of methods, including MetNet and MetNet-V2 [30, 31], have further provided rainfall predictions of larger spatial range and longer lead times but the crucial multiscale patterns useful for meteorologists are lost significantly.
>  首个用于降水临近预测的深度学习架构是卷积 LSTM，该模型统一了卷积和循环结构，用于时空预测，该模型在未来的工作中从各种方面被改进
>  U-Net 是广泛用于 AI4Science 的通用多尺度主干网络，也被用于降水临近预测

Previous methods suffer inevitably from unnatural blur and dissipation in the predictions that lose most of convective-scale features. To mitigate this problem, a special loss from the computer vision area is adopted in [32] to enhance the field sharpness. Adversarial learning techniques are also involved to tackle the problem of blurry predictions [33, 34]. On extreme precipitation events, the stacked neural networks on different rain levels are used in [35], which can reach the same forecasting skill of advection scheme method S-PROG [11] on heavy precipitation in one-hour lead time. DGMR [9], a major step forward by DeepMind, explores generative models in precipitation nowcasting by integrating spatiotemporal consistency and log-normal distribution of rain rate, and enables probabilistic predictions and ensemble forecasting. Since DGMR is proved to be the state-of-the-art model in the expert examination held by the UK Met-Office, we consider it as a powerful competing method in this article.
>  之前的方法的预测都存在不自然的模糊和消散现象，失去了大多数对流尺度特征，为了解决该问题，[32] 采用 CV 领域的特殊损失以增强场的清晰度，[33, 34] 采用对抗学习方法以解决模糊预测的问题
>  [35] 使用了不同降雨等级上的堆叠神经网络以预测极端降水事件
>  DGMR 考虑了时空一致性和降雨率的对数正态分布，并且可以进行概率性预测和集成预测

Deep learning models inspired by physical principles are another class of related work. Physics-informed neural networks (PINN) [36] are created based on partial differential equations (PDE). PINN leverages the gradient-solving ability of neural networks to approximate the differential equation by taking it as the objective function. Neural ordinary differential equation (NODE) [37] adopts the numerical methods such as Runge-Kutta to replace the forward propagation in neural networks. NODE is a powerful parameterization tool for modeling continuous processes. The idea of integrating physical knowledge into deep learning was studied for the predictions of sea surface temperature fields [38]. Here, the method uses a deep model to predict motion field for a single time step forward from the past observations, and the temperature field is advected by the motion field; Such a process is repeated autoregressively. This method relies on the autoregressive solving process of the equation, incurring uncontrollable accumulation error. The method has not been shown skillful for extreme precipitation nowcasting.
>  由物理原则启发的深度学习模型是另一类相关工作
>  PINN 最初被创建用于求解偏微分方程，PINN 将偏微分方程作为其目标函数，使用神经网络的梯度求解能力对其进行近似
>  NODE (神经元常微分方程) 将前向传播替换为数值方法，建模连续过程
>  [38] 结合物理知识和 DL 研究海洋平面温度场，它使用深度模型基于之前的观测预测单个时间步的运动场，使用运动场平流温度场，该过程自回归地重复执行
>  这一方法依赖于方程地自回归求解，会导致不可控制地累计误差，在极端降水预测中，这类自回归方法尚未有太大作用