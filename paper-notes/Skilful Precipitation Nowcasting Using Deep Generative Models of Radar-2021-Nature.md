# Abstract
Precipitation nowcasting, the high-resolution forecasting of precipitation up to two hours ahead, supports the real-world socioeconomic needs of many sectors reliant on weather-dependent decision-making State-of-the-art operational nowcasting methods typically advect precipitation fields with radar-based wind estimates, and struggle to capture important non-linear events such as convective initiations. Recently introduced deep learning methods use radar to directly predict future rain rates, free of physical constraints. While they accurately predict low-intensity rainfall, their operational utility is limited because their lack of constraints produces blurry nowcasts at longer lead times, yielding poor performance on rarer medium-to-heavy rain events. 
>  降水临近预报指提前两个小时的高分辨率降水预测
>  先进的业务临近预报方法通常用基于雷达估计的风场平流降水场，难以捕获重要的非线性事件，例如对流启动
>  近来的深度学习方法使用雷达直接预测未来的降雨率，不受物理约束限制，它们可以准确预测低强度降雨，但由于缺乏约束，它们在较长的提前期时会产生模糊的预测，因此在中到大雨的降水事件预测上表现不佳

Here we present a deep generative model for the probabilistic nowcasting of precipitation from radar that addresses these challenges. Using statistical, economic and cognitive measures, we show that our method provides improved forecast quality, forecast consistency and forecast value. Our model produces realistic and spatiotemporally consistent predictions over regions up to $1,536\,\mathrm{km}\times1,280\,\mathrm{km}$ and with lead times from 5–90 min ahead. Using a systematic evaluation by more than 50 expert meteorologists, we show that our generative model ranked first for its accuracy and usefulness in $89\%$ of cases against two competitive methods. When verified quantitatively, these nowcasts are skillful without resorting to blurring. We show that generative nowcasting can provide probabilistic predictions that improve forecast value and support operational utility, and at resolutions and lead times where alternative methods struggle. 
>  本文提出一个深度生成模型，基于雷达进行概率降水临近预报
>  模型在 1536km x 1280km 的区域和 5-90min 的提前时间上达到了优秀的预测
>  试验验证了模型的有效性，其生成的概率预测可以提高预报价值并支持业务实用性，同时其预测的分辨率和提前期都领先于其他方法

# Introduction
The high-resolution forecasting of rainfall and hydrometeors zero to two hours into the future, known as precipitation nowcasting, is crucial for weather-dependent decision-making. Nowcasting informs the operations of a wide variety of sectors, including emergency services, energy management, retail, flood early-warning systems, air traffic control and marine services. For nowcasting to be useful in these applications the forecast must provide accurate predictions across multiple spatial and temporal scales, account for uncertainty and be verified probabilistically, and perform well on heavier precipitation events that are rarer, but more critically affect human life and economy. 
>  降水临近预测即对未来 0-2 个小时内的高分辨率降水预测
>  临近预报需要提供跨多个空间和时间尺度的准确预测，同时考虑到不确定性并提供概率验证，并且需要在罕见的强降水预测上也有良好表现

Ensemble numerical weather prediction (NWP) systems, which simulate coupled physical equations of the atmosphere to generate multiple realistic precipitation forecasts, are natural candidates for nowcasting as one can derive probabilistic forecasts and uncertainty estimates from the ensemble of future predictions. For precipitation at zero to two hours lead time, NWPs tend to provide poor forecasts as this is less than the time needed for model spin-up and due to difficulties in non-Gaussian data assimilation.
>  集成数值天气预报系统通过模拟大气的耦合物理方程以生成多个可能的降水预测
>  对于 0-2 小时提前时间的降水预报，NWP 的效果一般较差，因为该时间往往小于模型启动的时间，且非高斯数据同化存在挑战

 As a result, alternative methods that make predictions using composite radar observations have been used; radar data is now available (in the UK) every five minutes and at $1\,\mathsf{k m}\times1\,\mathsf{k m}$ grid resolution. Established probabilistic nowcasting methods, such as STEPS and PySTEPS, follow the NWP approach of using ensembles to account for uncertainty, but model precipitation following the advection equation with a radar source term. In these models, motion fields are estimated by optical flow, smoothness penalties are used to approximate an advection forecast, and stochastic perturbations are added to the motion field and intensity model. These stochastic simulations allow for ensemble nowcasts from which both probabilistic and deterministic forecasts can be derived and are applicable and consistent at multiple spatial scales, from the kilometre scale to the size of a catchment area. 
>  因此，出现了基于组合雷达观测进行预测的方法，在英国，雷达数据可以做到每五分钟更新一次，并且具有 1km x 1km 的网格分辨率
>  现存的概率临近预测方法，例如 STEPS 和 PySTEPS，遵循 NWP ，使用集成方法来考虑不确定性，通过基于雷达数据的平流方程建模降水
>  这些模型通过光流估计运动场，使用平滑惩罚以近似平流预测，且为运动场和强度模型添加随机扰动，这些随机模拟使得模型可以从集成预报中得出多个空间尺度 (从公里尺度到集水区尺度) 的概率性和确定性预报

> [!info] Spin-up Time
> NWP 模型在启动时需要一定时间来“预热”或稳定自身，模型在这个过程中逐渐调整初始状态。对于非常短期的预报，这段时间通常不足以让模型达到最佳性能，因此该期间生成的预报可能不准确

> [!info] Non-Gaussian Data Assimilation
> 数据同化指观测数据融入数值模型，以改善初始条件，提高预报精度的过程，传统的数据同化方法假设误差服从高斯分布，这不一定符合实际

Approaches based on deep learning have been developed that move beyond reliance on the advection equation. By training these models on large corpora of radar observations rather than relying on in-built physical assumptions, deep learning methods aim to better model traditionally difficult non-linear precipitation phenomena, such as convective initiation and heavy precipitation. This class of methods directly predicts precipitation rates at each grid location, and models have been developed for both deterministic and probabilistic forecasts. As a result of their direct optimization and fewer inductive biases, the forecast quality of deep learning methods—as measured by per-grid-cell metrics such as critical success index (CSI) at low precipitation levels (less than $2\,\mathsf{m m}\,\mathsf{h}^{-1}$) — has greatly improved. 
>  基于深度学习的方法不再依赖平流方程。基于深度学习的方法在大量雷达观测数据上训练模型，而不依赖内置物理假设，目标是更好模拟以往难以处理的非线性降水现象，例如平流启动和强降水
>  基于深度学习的方法直接预测每个网格的降水率，可以构建进行确定性预测的模型，也可以构建进行概率性预测的模型
>  深度学习模型直接优化降水误差，且相较于平流方法具有较少的归纳偏置，故大大改善了在低降水水平上 (小于 2mm/h) 的预测性能

As a number of authors have noted, forecasts issued by current deep learning systems express uncertainty at increasing lead times with blurrier precipitation fields, and may not include small-scale weather patterns that are important for improving forecast value. Furthermore, the focus in existing approaches on location-specific predictions, rather than probabilistic predictions of entire precipitation fields, limits their operational utility and usefulness, being unable to provide simultaneously consistent predictions across multiple spatial and temporal aggregations. The ability to make skilful probabilistic predictions is also known to provide greater economic and decision-making value than deterministic forecasts. 
>  当前深度学习模型发布的预测降水场随着提前期的增长会更加模糊 (更具不确定性)，并且可能不包含小尺度的气象模式
>  另外，现有方法主要关注特定位置的预测，而不是对整个降水场的概率性预测，进而不能同时提供多个时空上一致的预测
>  概率性预测的决策价值比确定性预测更高

Here we demonstrate improvements in the skill of probabilistic precipitation nowcasting that improves their value. To create these more skilful predictions, we develop an observations-driven approach for probabilistic nowcasting using deep generative models (DGMs). DGMs are statistical models that learn probability distributions of data and allow for easy generation of samples from their learned distributions. As generative models are fundamentally probabilistic, they have the ability to simulate many samples from the conditional distribution of future radar given historical radar, generating a collection of forecasts similar to ensemble methods. The ability of DGMs to both learn from observational data as well as represent uncertainty across multiple spatial and temporal scales makes them a powerful method for developing new types of operationally useful nowcasting. 
>  本文提出一种基于观测数据的概率性临近降水方法，采用了深度生成式模型
>  深度生成式模型学习数据的分布，基于其学习到的分布生成样本
>  生成式模型本质是概率性的，可以在条件于历史雷达观测的未来雷达观测的条件分布中模拟多个样本，生成一组预测，类似集成方法
>  生成式模型既可以从观测数据中学习，也可以表示多个时空尺度上的不确定性

These models can predict smaller-scale weather phenomena that are inherently difficult to predict due to underlying stochasticity, which is a critical issue for nowcasting research. DGMs predict the location of precipitation as accurately as systems tuned to this task while preserving spatiotemporal properties useful for decision-making. Importantly, they are judged by professional meteorologists as substantially more accurate and useful than PySTEPS or other deep learning systems. 
>  DGMs 可以预测由于潜在的随机性而难以预测的小尺度天气现象
>  DGMs 在预测降水位置的准确性方面和专门针对该任务优化的系统相当，同时还保留了对于决策有用的时空特性

# Generative models of radar 
Our nowcasting algorithm is a conditional generative model that predicts $N$ future radar fields given $M$ past, or contextual, radar fields, using radar-based estimates of surface precipitation $\mathbf{X}_{T}$ at a given time point $T$. Our model includes latent random vectors $\mathbf Z$ and parameters $\theta$, described by 

$$
\begin{array}{l}{P(\mathbf{X}_{M+1:M+N}\mid\mathbf{X}_{1:M})}\\ {\displaystyle\qquad=\int P(\mathbf{X}_{M+1:M+N}\mid\mathbf{Z},\mathbf{X}_{1:M},\mathbf {\theta})P(\mathbf {Z}\mid\mathbf{X}_{1:M})\mathrm{d}\mathbf{Z}.}\end{array}\tag{1}
$$

The integration over latent variables ensures that the model makes predictions that are spatially dependent. 

>  我们的算法是一个条件生成式模型，它在给定 $M$ 个过去的雷达观测场来预测未来的 $N$ 个雷达观测场
>  模型包括了潜在随机向量 $\mathbf Z$ 和参数 $\theta$，在潜在变量上的积分确保模型的预测在空间上是相关的

![[pics/DGMR-Fig1a.png]]

Learning is framed in the algorithmic framework of a conditional generative adversarial network $({\mathsf{G A N}})$ , specialized for the precipitation prediction problem. Four consecutive radar observations (the previous $20\,\mathsf{m i n}$ ) are used as context for a generator (Fig. 1a) that allows sampling of multiple realizations of future precipitation, each realization being 18 frames $(90\,\mathsf{m i n})$ . 
>  模型的学习过程在条件 GAN 的框架下进行
>  生成器接受 4 个连续的雷达观测帧 (过去 20 分钟，每个观测帧 5 分钟) 作为上下文，并采样出 18 个未来的雷达场 (未来 90 分钟)

Learning is driven by two loss functions and a regularization term, which guide parameter adjustment by comparing real radar observations to those generated by the model. The first loss is defined by a spatial discriminator, which is a convolutional neural network that aims to distinguish individual observed radar fields from generated fields, ensuring spatial consistency and discouraging blurry predictions. The second loss is defined by a temporal discriminator, which is a three-dimensional (3D) convolutional neural network that aims to distinguish observed and generated radar sequences, imposes temporal consistency and penalizes jumpy predictions.
>  模型的目标函数包括两个损失和一个正则化项，目标函数引导模型调节参数生成逼真的雷达观测
>  第一个损失由空间判别器定义，空间判别器为卷积网络，目的是区分单独的观测雷达场和生成雷达场，以确保生成雷达场的空间一致性，抑制模糊预测
>  第二个损失由时间判别器定义，时间判别器是三维卷积网络，目的是区分观测雷达场序列和生成雷达场序列，以确保生成雷达场序列的时间一致性，惩罚跳跃性预测

 These two discriminators share similar architectures to existing work in video generation. When used alone, these losses lead to accuracy on par with Eulerian persistence. To improve accuracy, we introduce a regularization term that penalizes deviations at the grid cell resolution between the real radar sequences and the model predictive mean (computed with multiple samples). This third term is important for the model to produce location-accurate predictions and improve performance. In the Supplementary Information, we show an ablation study supporting the necessity of each loss term. 
>  这两个判别器和现存的视频生成网络中的判别器结构类似
>  仅使用这两个损失时，得到的模型的预测准确率和欧拉持久模型相当，为了进一步提高准确率，我们引入正则化项，惩罚真实雷达序列和模型预测均值 (通过多个样本计算) 在网格分辨率上的偏差
>  正则化项对于模型生成位置准确的预测很重要，我们通过消融试验支持了每个损失项的必要性

Finally, we introduce a fully convolutional latent module for the generator, allowing for predictions over precipitation fields larger than the size used at training time, while maintaining spatiotemporal consistency. We refer to this DGM of rainfall as DGMR in the text. 
>  最后，我们为生成器引入了全卷积潜在模块，以支持在比训练时使用的降水场尺寸更大的降水场上训练，同时保持时空一致性
>  我们称用于降水预测的深度生成模型为 DGMR

> [!info] 欧拉持久模型
> 欧拉持久模型（Eulerian Persistence Model）是一种用于临近预报的简单预测方法，尤其是用来预测短时间内的天气变化，它假设未来的天气模式与当前观测到的模式相同，即认为大气中的物理量（如风速、温度、湿度或雷达反射率等）在短时间内不会发生显著变化。
> 具体来说，在降雨预报中，欧拉持久模型假设未来一段时间内的降水分布将保持不变，等于当前时刻的观测值。
> 
> **优点**：
   欧拉持久模型只是简单地复制了当前的观测数据作为未来的预测，不需要任何计算。该模型适用于非常短时间范围内的预报，例如 0 到 1 小时内的临近预报。在这个时间段内，大气条件通常不会有剧烈的变化，因此这种方法可以提供相对合理的预测。
> 
> **缺点**：
> 欧拉持久模型假设天气系统是静态的，忽略了天气系统的移动和发展。实际上，天气系统通常是动态的，会随着时间推移而发生变化。随着预报时间的延长，欧拉持久模型的准确性会迅速下降，因为它未能捕捉这些动态变化。并且它不能预测诸如雷暴形成、锋面过境等复杂天气现象的发展和演变。
> 
> 尽管有其局限性，欧拉持久模型常被用作评估更复杂预报模型性能的一个基准。如果一个新模型的表现无法超越欧拉持久模型，那么这个新模型可能并不具备实际应用价值。

The model is trained on a large corpus of precipitation events, which are $256\times256$ crops extracted from the radar stream, of length $110\,\mathrm{min}$ (22 frames). An importance-sampling scheme is used to create a dataset more representative of heavy precipitation (Methods). Throughout, all models are trained on radar observations for the UK for years 2016–2018 and evaluated on a test set from 2019. Analysis using a weekly train–test split of the data, as well as data of the USA, is reported in Extended Data Figs. 1–9 and the Supplementary Information. Once trained, this model allows fast full-resolution nowcasts to be produced, with a single prediction (using an NVIDIA V100 GPU) needing just over a second to generate. 
>  模型在一个大规模的降水事件数据集上训练，数据集中的降水事件是从雷达流中提取的 256x256 的裁剪区域，时间长度为 110min (22 帧)
>  我们采用重要性采样方法构建更能代表强降水的数据集

# Intercomparison case study 
We use a single case study to compare the nowcasting performance of the generative method DGMR to three strong baselines: PySTEPS, a widely used precipitation nowcasting system based on ensembles, considered to be state-of-the-art; UNet, a popular deep learning method for nowcasting; and an axial attention model, a radar-only implementation of MetNet. 
>  本小节进行一次案例研究，比较 DGMR 的生成式方法和 PySTEPS (基于集成的降水预测系统), UNet 和轴向注意力模型 MetNet

![[pics/DGMR-Fig1b.png]]

For a meteorologically challenging event, Figs. 1b, c and 4b shows the ground truth and predicted precipitation fields at $T\!+30$ , $T\!+60$ and $T\!+90$ min, quantitative scores on different verification metrics, and comparisons of expert meteorologist preferences among the competing methods. Two other cases are included in Extended Data Figs. 2 and 3. 
>  Figure 1b, c 展示了在 $T+30, T+60, T+90$ 时的真实降水场和预测降水场，以及不同指标下的定量评分
>  Figure 4b 展示了气象专家对各个方法的偏好比较

![[pics/DGMR-Fig4b.png]]

The event in Fig. 1 shows convective cells in eastern Scotland with intense showers over land. Maintaining such cells is difficult and a traditional method such as PySTEPS overestimates the rainfall intensity over time, which is not observed in reality and does not sufficiently cover the spatial extent of the rainfall. The UNet and axial attention models roughly predict the location of rain, but owing to aggressive blurring, over-predict areas of rain, miss intensity and fail to capture any small-scale structure. By comparison, DGMR preserves a good spatial envelope, represents the convection and maintains heavy rainfall in the early prediction, although with less accurate rates at $T+90$ min and at the edge of the radar than at previous time steps. When expert meteorologists judged these predictions against ground truth observations, they significantly preferred the generative nowcasts, with $93\%$ of meteorologists choosing it as their first choice (Fig. 4b). 
>  Fig1 展示了苏格兰东部的带有对流单元的强降水事件
>  传统方法例如 PySTEPS 难以维持这样的对流单元，随着时间推移会高估降水强度，同时也无法充分覆盖降水的空间范围
>  UNet 和轴向注意力模型粗略预测了降水的位置，但由于过高的模糊效应，它们高估了降水区域，低估了降水强度，无法捕获任意小尺度的结构
>  DGMR 维持了良好的空间范围，表示了对流，并在早期预测中维持了强降水 (没有模糊) 

The figures also include two common verification scores. These predictions are judged as significantly different by experts, but the scores do not provide this insight. This study highlights a limitation of using existing popular metrics to evaluate forecasts: while standard metrics implicitly assume that models, such as NWPs and advection-based systems, preserve the physical plausibility of forecasts, deep learning systems may outperform on certain metrics by failing to satisfy other needed characteristics of useful predictions. 
>  Figure 1 中的图还包括了两个常见的验证评分
>  模型的预测在专家评测中具有显著差异，但在评分中则没有，这说明了现存的预测评估指标存在局限性：标准的指标都隐式地假设了模型，例如基于平流的系统以及 NWP 维持了预报的物理合理性，深度学习模型虽然没有满足在这些指标上表现良好所需要的特性，但能够在其他指标上表现更好 (专家评测)

# Forecast skill evaluation 
We verify the performance of competing methods using a suite of metrics as is standard practice, as no single verification score can capture all desired properties of a forecast. We report the ${\mathsf{C S I}}$ to measure location accuracy of the forecast at various rain rates. We report the radially averaged power spectral density (PSD) to compare the precipitation variability of nowcasts to that of the radar observations. We report the continuous ranked probability score (CRPS) to determine how well the probabilistic forecast aligns with the ground truth. For CRPS, we show pooled versions, which are scores on neighbourhood aggregations that show whether a prediction is consistent across spatial scales. Details of these metrics, and results on other standard metrics, can be found in Extended Data Figs. 1–9 and the Supplementary Information. We report results here using data from the UK, and results consistent with these showing generalization of the method on data from the USA in Extended Data Figs. 1–9. 
>  我们使用一系列指标验证方法的性能，其中 CSI 用于衡量不同降雨率下的预报位置准确性，经向平均功率谱密度 (PSD) 用于比较预报和观测的降水变异性，连续排序概率评分 (CRPS) 用于决定概率预报和实际情况的匹配程度
>  我们使用 CRPS 的池化版本，在邻域的聚合上计算分数，以展示预测是否在跨空间尺度上和观测一致

![[pics/DGMR-Fig2a.png]]

Figure 2a shows that all three deep learning systems produce forecasts that are significantly more location-accurate than the PySTEPS baseline when compared using CSI. Using paired permutation tests with alternating weeks as independent units to assess statistical significance, we find that DGMR has significant skill compared to PySTEPS for all precipitation thresholds $(n\,{=}\,26,P\,{<}\,10^{-4})$ (Methods). 
>  根据 Figure 2a，所有的深度学习方法生成的预测在位置准确性方面都显著优于 PySETPS 方法

The PSD in Fig. 2b shows that both DGMR and PySTEPS match the observations in their spectral characteristics, but the axial attention and UNet models produce forecasts with medium- and small-scale precipitation variability that decreases with increasing lead time. As they produce blurred predictions, the effective resolution of the axial attention and UNet nowcasts is far less than the 1 km $\times1$ km resolution of the data. At $T\,{+}\,90\,\mathrm{min}$ , the effective resolution for UNet is $32\,\mathrm{km}$ and for axial attention is $16\,\mathsf{k m}$ , reducing the value of these nowcasts for meteorologists. 
>  根据 Figure 2b，DGMR 和 PySTEPS 的预测结果在频谱特性上都和观测结果匹配，而轴向注意力模型和 UNet 模型的预测在中小尺度上的降水预测的变异性随着提前时间的增加而减少
>  因为轴向注意力模型和 UNet 模型生成的预测较模糊，它们的预测的有效分辨率远远小于观测数据的 1km x 1km 分辨率，在 $T+90$ min 时，UNet 的有效分辨率是 32km，轴向注意力模型的有效分辨率是 32km

![[pics/DGMR-Fig3.png]]

For probabilistic verification, Fig. 3a, b shows the CRPS of the average and maximum precipitation rate aggregated over regions of increasing size. When measured at the grid-resolution level, DGMR, PySTEPS and axial attention perform similarly; we also show an axial attention model with improved performance obtained by rescaling its output probabilities (denoted ‘axial attention temp. opt.’). As the spatial aggregation is increased, DGMR and PySTEPS provide consistently strong performance, with DGMR performing better on maximum precipitation. The axial attention model is significantly poorer for larger aggregations and underperforms all other methods at scale four and above. Using alternating weeks as independent units, paired permutation tests show that the performance differences between DGMR and the axial attention temp. opt. are significant $(n\,{=}\,26,P\,{<}\,10^{-3})$ . 
>  Figure 3a, b 展示了在不同大小区域上聚合的平均和最大降水率
>  在网格分辨率级别上度量时，DGMR, PySTEPS 和轴向注意力模型的表现类似，轴向注意力的性能可以通过重新缩放其输出概率进行改进
>  随着空间聚合程度增阿达，DGMR 和 PySTEPS 的性能持续较优，DGMR 在最大降水率下表现更好，轴向注意力模型的表现则显著下降 

NWP and PySTEPS methods include post-processing that is used by default in their evaluation to improve reliability. We show a simple post-processing method for DGMR in Figs. 2 and 3 (denoted ‘recal’) (Methods), which further improves its skill scores over the uncalibrated approach. Post-processing improves the reliability diagrams and rank histogram to be as or more skilful than the baseline methods (Extended Data Fig. 4). 
>  Figure 2,3 中还展示了使用了后处理的 DGMR 的表现，后处理进一步提高了模型的评分

We also show evaluation on other metrics, performance on a data split over weeks rather than years, and evaluation recapitulating the inability of NWPs to make predictions at nowcasting timescales (Extended Data Figs. 4–6). We show results on a US dataset in Extended Data Figs. 7–9. 

Together, these results show that the generative approach verifies competitively compared to alternatives: it outperforms (on CSI) the incumbent STEPS nowcasting approach, provides probabilistic forecasts that are more location accurate, and preserves the statistical properties of precipitation across spatial and temporal scales without blurring whereas other deep learning methods do so at the expense of them. 
>  综合以上的结果表明，生成式方法比其他方法更有竞争力，它能提供更精确的位置概率预报，同时在不模糊的情况下保持降水在时空尺度上的统计特性

# Forecast value evaluation 
We use both economic and cognitive analyses to show that the improved skill of DGMR results in improved decision-making value. 

We report the relative economic value of the ensemble prediction to quantitatively evaluate the benefit of probabilistic predictions using a simple and widely used decision-analytic model; see the Supplementary Information for a description. Figure 4a shows that DGMR provides the highest economic value relative to the baseline methods (has highest peak and greater area under the curve). We use 20 member ensembles and show three accumulation levels used for weather warnings by Met Éireann (the Irish Meteorological service uses warnings defined directly in $\mathsf{m m}\,\mathsf{h}^{-1}$ ; https://www.met.ie/weather-warnings). This analysis shows the ability of the generative ensemble to capture uncertainty, and we show the improvement with samples in Extended Data Figs. 4 and 9, and postage stamp plots to visualize the ensemble variability in Supplementary Data 1–3. 
>  我们使用决策分析模型，量化地报告了集成预报的相对经济价值

Importantly, we ground this economic evaluation by directly assessing decision-making value using the judgments of expert meteorologists working in the 24/7 operational center at the Met Office (the UK’s national meteorology service). We conducted a two-phase experimental study to assess expert judgements of value, involving a panel of 56 experts. In phase 1, all meteorologists were asked to provide a ranked preference assessment on a set of nowcasts with the instruction that ‘preference is based on [their] opinion of accuracy and value’. Each meteorologist assessed a unique set of nowcasts, which, at the population level, allows for uncertainty characteristics and meteorologist idiosyncrasies to be averaged out in reporting the statistical effect. We randomly selected $20\%$ of meteorologists to participate in a phase 2 retrospective recall interview. 

Operational meteorologists seek utility in forecasts for critical events, safety and planning guidance. Therefore, to make meaningful statements of operational usefulness, our evaluation assessed nowcasts for high-intensity events, specifically medium rain (rates above $5\,\mathsf{m m}\,\mathsf{h}^{-1}$ ) and heavy rain (rates above $10\,\mathsf{m m\,h^{-1}}$ ). Meteorologists were asked to rank their preferences on a sample of 20 unique nowcasts (from a corpus of 2,126 events, being all high-intensity events in 2019). Data were presented in the form shown in Fig. 1b, c, showing clearly the initial context at $T\!+\!0$ min, the ground truth at $T\,{+}\,30\,\mathrm{min}$ , $T\,{+}\,60\,\mathrm{min}$ , and $T+90$ min, and nowcasts from PySTEPS, axial attention and DGMR. The identity of the methods in each panel was anonymized and their order randomized. See the Methods for further details of the protocol and of the ethics approval for human subjects research. 

The generative nowcasting approach was significantly preferred by meteorologists when asked to make judgments of accuracy and value of the nowcast, being their most preferred $89\%$ ( $95\%$ confidence interval (CI) [0.86, 0.92]) of the time for the $5\,\mathsf{m m}\,\mathsf{h}^{-1}$ nowcasts (Fig. 4c; $;P\!<\!10^{-4}\!)$ ), and $90\%$ ( $95\%$ CI [0.87, 0.92]) for the $10\,\mathsf{m m\,h^{-1}}$ nowcasts (Fig. 4d,  $P\!<\!10^{-4\cdot}$ . We compute the $P$ value assessing the binary decision whether meteorologists chose DGMR as their first choice using a permutation test with 10,000 resamplings. We indicate the Clopper–Pearson CI. This significant meteorologist preference is important as it is strong evidence that generative nowcasting can provide meteorologists with physical insight not provided by alternative methods, and provides a grounded verification of the economic value analysis in Fig. 4a. 

Meteorologists were not swayed by the visual realism of the predictions, and their responses in the subsequent structured interviews showed that they approached this task by making deliberate judgements of accuracy, location, extent, motion and rainfall intensity, and reasonable trade-offs between these factors (Supplementary Information, section C.6). In the phase 2 interviews, PySTEPS was described as “being too developmental which would be misleading”, that is, as having many “positional errors” and “much higher intensity compared with reality”. The axial attention model was described as “too bland”, that is, as being “blocky” and “unrealistic”, but had “good spatial extent”. Meteorologists described DGMR as having the “best envelope”, “representing the risk best”, as having “much higher detail compared to what \[expert meteorologists\] are used to at the moment”, and as capturing “both the size of convection cells and intensity the best”. In the cases where meteorologists chose PySTEPS or the axial attention as their first choice, they pointed out that DGMR showed decay in the intensity for heavy rainfall at $T+90$ min and had difficulty predicting isolated showers, which are important future improvements for the method. See the Supplementary Information for further reports from this phase of the meteorologist assessment. 

# Conclusion 
Skilful nowcasting is a long-standing problem of importance for much of weather-dependent decision-making. Our approach using deep generative models directly tackles this important problem, improves on existing solutions and provides the insight needed for real-world decision-makers. We showed—using statistical, economic and cognitive measures—that our approach to generative nowcasting provides improved forecast quality, forecast consistency and forecast value, providing fast and accurate short-term predictions at lead times where existing methods struggle. 
>  我们使用深度生成模型进行临近预报，统计上、经济上、认知上的度量展示了生成式模型可以提高预测质量、预测一致性和预测价值

Yet, there remain challenges for our approach to probabilistic nowcasting. As the meteorologist assessment demonstrated, our generative method provides skilful predictions compared to other solutions, but the prediction of heavy precipitation at long lead times remains difficult for all approaches. 
>  面对长提前时间的强降水事件，所有预测方法都效果不佳

Critically, our work reveals that standard verification metrics and expert judgments are not mutually indicative of value, highlighting the need for newer quantitative measurements that are better aligned with operational utility when evaluating models with few inductive biases and high capacity. Whereas existing practice focuses on quantitative improvements without concern for operational utility, we hope this work will serve as a foundation for new data, code and verification methods—as well as the greater integration of machine learning and environmental science in forecasting larger sets of environmental variables—that makes it possible to both provide competitive verification and operational utility. 
> 我们的工作还揭示了标准验证指标和专家评估并不一定一致，强调了需要新的定量度量方法

# Methods 
We provide additional details of the data, models and evaluation here, with references to extended data that add to the results provided in the main text. 

## Datasets 
A dataset of radar for the UK was used for all the experiments in the main text. Additional quantitative results on a US dataset are available in Supplementary Information section A. 

### UK dataset 
To train and evaluate nowcasting models over the UK, we use a collection of radar composites from the Met Office RadarNet network. This network comprises more than 15 operational, proprietary C-band dual polarization radars covering $99\%$ of the UK (see figure 1 in ref. ). We refer to ref. 11 for details about how radar reflectivity is post-processed to obtain the two-dimensional radar composite field, which includes orographic enhancement and mean field adjustment using rain gauges. Each grid cell in the $1536\times1,280$ composite represents the surface-level precipitation rate $\mathsf{i n}\,\mathsf{m m}\,\mathsf{h}^{-1}$ ) over a 1km x 1km region in the OSGB36 coordinate system. If a precipitation rate is missing (for example, because the location is not covered by any radar, or if a radar is out of order), the corresponding grid cell is assigned a negative value which is used to mask the grid cell at training and evaluation time. The radar composites are quantized in increments of $1/32\,\mathsf{m m\,h^{-1}}$ . 
>  在 1536 x 1280 的复合雷达图中，每个网格单元代表 OSGB36 坐标系统中 1km x 1km 区域的地表降水率 (单位为 mm/h)，如果某个区域的地表降水率观测缺失，相应的网格单元会被赋予一个负值，该负值作为掩码，用于在训练和评估时屏蔽该网格单元
>  雷达复合数据以 1/32 mm/h 的增量进行量化 (也就是降水量数据按照 1/32 mm/h 的单位离散化，例如某个网格单元的降水量为 0.5 mm/h，则它会被量化为最接近 1/32 mm/h 的整数倍，即 16/32 mm/h=0.5mm/h)

We use radar collected every five minutes between 1 January 2016 and 31 December 2019. We use the following data splits for model development. Fields from the first day of each month from 2016 to 2018 are assigned to the validation set. All other days from 2016 to 2018 are assigned to the training set. Finally, data from 2019 are used for the test set, preventing data leakage and testing for out of distribution generalization. For further experiments testing in-distribution performance using a different data split, see Supplementary Information section C. 
>  雷达数据收集时间段为 2016 年 1 月 1 日到 2019 年 12 月 31 日，每五分钟收集一次，2016 年到 2018 年每个月的第一天的数据被分配到验证集，其他数据分配到训练集
>  2019 年的数据作为测试集

### Training set preparation 
Most radar composites contain little to no rain. Supplementary Table 2 shows that approximately $89\%$ of grid cells contain no rain in the UK. Medium to heavy precipitation (using rain rate above $4\,\mathsf{m m\,h^{-1}}$ ) comprises fewer than $0.4\%$ of grid cells in the dataset. To account for this imbalanced distribution, the dataset is rebalanced to include more data with heavier precipitation radar observations, which allows the models to learn useful precipitation predictions. 
>  大多数复合雷达观测几乎不含降雨，数据集中近乎 89% 的网格单元没有降水
>  中到大雨 (降雨率高于 4mm/h) 的网格单元占比不到 0.4%
>  为此，我们对数据集进行了重新平衡，以包含更多具有较大降水的雷达观测

Each example in the dataset is a sequence of 24 radar observations of size $1,536\times1,280$ , representing two continuous hours of data. The maximum rain rate is capped at $128\,\mathsf{m m\,h^{-1}}$ , and sequences that are missing one or more radar observations are removed. $256\times256$ crops are extracted and an importance sampling scheme is used to reduce the number of examples containing little precipitation. We describe this importance sampling and the parameters used in Supplementary Information section A.1. After subsampling and removing entirely masked examples, the number of examples in the training set is roughly 1.5 million. 
>  数据集中每个样本为 24 个 1536 x 1280 的雷达观测组成的序列，表示连续两个小时的数据 (5 x 24 = 120min)，最大降雨速率被限制在 128mm/h，缺少一个以上雷达观测的序列会被移除
>  我们再从这些样本中提取出 256 x 256 的裁剪区域，并使用重要性采样方法来减少包含少量降水的样本数量
>  在经过子采样并移除了完全被掩蔽的样本后，训练集中的样本数量大约有 150 万个

## Model details and baselines 
Here, we describe the proposed method and the three baselines to which we compare performance. When applicable, we describe both the architectures of the models and the training methods. There is a wealth of prior work, and we survey them as additional background in Supplementary Information section E. 

### DGMR 

![[pics/DGMR-Extended Fig 1a.png]]

A high-level description of the model was given in the main text and in Fig. 1a, and we provide some insight into the design decisions here. 

**Architecture design.** 
The nowcasting model is a generator that is trained using two discriminators and an additional regularization term. Extended Data Fig. 1 shows a detailed schematic of the generative model and the discriminators. More precise descriptions of these architectures are given in Supplement B and corresponds to the code description; pseudocode is also available in the Supplementary Information. 
>  临近预报模型是一个生成器，使用两个判别器和一个额外的正则化项训练

The generator in Fig. 1a comprises the conditioning stack which processes past four radar fields that is used as context. Making effective use of such context is typically a challenge for conditional generative models, and this stack structure allows information from the context data to be used at multiple resolutions, and is used in other competitive video GAN models, for example, in ref. 26. This stack produces a context representation that is used as an input to the sampler. 
>  生成器包括一个条件栈，负责处理过去的四个雷达场作为上下文，这一栈结构使得上下文信息可以在多分辨率下被利用
>  条件栈生成的上下文表示作为采样器的输入

A latent conditioning stack takes samples from $N(0,1)$ Gaussian distribution, and reshapes into a second latent representation. 
>  除此以外，还有一个隐式条件栈，它接受来自标准高斯分布的样本作为输入，将其重塑为第二个潜在表示作为采样器的输入

The sampler is a recurrent network formed with convolutional gated recurrent units (GRUs) that uses the context and latent representations as inputs. The sampler makes predictions of 18 future radar fields (the next $90\,\mathrm{{min})}$ . 
>  采样器是一个循环神经网络，由卷积 GRUs 组成
>  采样器使用上下文和潜在表示作为输入，预测 18 个未来雷达场 (未来 90 分钟)

This architecture is both memory efficient and has had success in other forecasting applications. We also made comparisons with longer context using the past 6 or 8 frames, but this did not result in appreciable improvements. 
>  我们还尝试用过去的 6 或 8 帧提供更长的上下文，但效果没有显著提升

![[pics/DGMR-Extended Fig 1b.png]]

Two discriminators in Fig. 1b are used to allow for adversarial learning in space and time. The spatial and temporal discriminator share the same structure, except that the temporal discriminator uses 3D convolutions to account for the time dimension. Only 8 out of 18 lead times are used in the spatial discriminator, and a random $128\times128$ crop used for the temporal discriminator. These choices allow the models to fit within memory. 
>  Figure 1b 中的两个判别器用于在时间尺度和空间尺度上对抗学习
>  时空判别器的结构相同，差异仅在于时间判别器使用 3D 卷积以处理时间维度
>  空间判别器仅使用 18 个提前帧中的 8 个，时间判别器使用随机的 128 x 128 裁剪区域，这些选择是为了模型能够放进内存

We include a spatial attention block in the latent conditioning stack since it allows the model to be more robust across different types of regions and events, and provides an implicit regularization to prevent overfitting, particularly for the US dataset. 
>  我们在潜在条件栈中包含了一个空间注意力块，这使得模型在不同类型的区域和事件中更具健壮性，同时提供了一种隐式的正则化，防止过拟合

Both the generator and discriminators use spectrally normalized convolutions throughout, similar to ref. 35, since this is widely established to improve optimization. During model development, we initially found that including a batch normalization layer (without variance scaling) prior to the linear layer of the two discriminators improved training stability. The results presented use batch normalization, but we later were able to obtain nearly identical quantitative and qualitative results without it. 
>  生成器和判别器都使用了谱规范化的卷积，用于改善优化
>  在模型开发初期，我们发现在两个判别器中的线性层之前加入批量规范化层 (不进行方差缩放) 可以提高训练稳定性，但后来发现不使用批量规范化也可以得到相同的结果

**Objective function.** 
The generator is trained with losses from the two discriminators and a grid cell regularization term (denoted $\mathcal{L}_{R}(\theta))$ . The spatial discriminator $D_{\phi}$ has parameters $\phi$ , the temporal discriminator $T_{\psi}$ has parameters $\psi$ , and the generator $G_{\theta}$ has parameters $\theta$. We indicate the concatenation of two fields using the notation $\{\mathbf{X};G\}$ . The generator objective that is maximized is 

$$
\begin{align}{\mathcal{L}_{G}({\theta})=\mathbb{E}_{\mathbf{X}_{1:M+N}}[\mathbb{E}_{\mathbf Z}[D(G_{{\theta}}(\mathbf Z;\mathbf{X}_{1:M}))}{+\left.T(\{\mathbf{X}_{1:M};G_{{\theta}}(\mathbf Z;\mathbf{X}_{1:M})\})\right]-\lambda\mathcal{L}_{R}({\theta})];}\end{align}\tag{2}
$$ 
$$
\begin{align}{{\mathcal{L}_{{R}}(\theta)=\frac{1}{H W N}}}{||\mathrm{(\mathbb{E}_{\mathbf Z}[G_{\theta}(\mathbf Z;\mathbf{X}_{1:M})]-\mathbf{X}_{M+1:M+N}]})\odot w(\mathbf{X}_{M+1:M+N})||_{1}.}\end{align}\tag{3}
$$ 
>  目标函数包括判别器损失项 $\mathcal L_G(\theta)$ 和网格单元正则化项 $\mathcal L_R(\theta)$
>  空间判别器 $D_\phi$ 的参数为 $\phi$，时间判别器 $T_\psi$ 的参数为 $\psi$，生成器 $G_\theta$ 的参数为 $\theta$

>  损失 $\mathcal L_G(\theta)$ 是一个期望，外层期望是对全部序列数据 $\mathbf X_{1:M+N}$ 求期望
>  外层期望内由两项构成
>  第一项是对潜在变量 $\mathbf Z$ 的期望，期望内包括了 $D(G_\theta(\mathbf Z; \mathbf X_{1:M}))$，表示生成器接受潜在变量 $\mathbf Z$ 和输入序列 $\mathbf X_{1:M}$ 作为输入，其输出直接交给空间判别器，空间判别器输出判断其为真的分数；以及 $T(\{\mathbf X_{1: M}; G_{\theta}(\mathbf Z;\mathbf X_{1:M})\})$，表示时间判别器接受生成器的输出和原始输入序列 $\mathbf X_{1:M}$ 的拼接作为输入，输出判断生成器输出为真的分数
>  第二项是正则化项，它是一个在网格粒度上预测和真实观测之间距离的加权一范数，其中权重为 $w(\mathbf X_{M+1:M+N})$，$\mathbf X_{M+1:M+N}$ 表示真实的预测序列，因此是按照真实观测进行加权；距离为 $\mathbb E_{\mathbf Z}[G_\theta(\mathbf Z; \mathbf X_{1:M})] - \mathbf X_{M+1:M+N}$，也就是直接相减

>  要最大化目标函数，就是尽可能最大化 $D, T$ 的输出，即尽可能欺骗判别器，同时最小化正则化项，即最小化预测和真实观测之间的距离

We use Monte Carlo estimates for expectations over the latent $\mathbf Z$ in equations (2) and (3). These are calculated using six samples per input $\mathbf{X}_{1:M},$ which comprises $M\!=\!4$ radar observations. The grid cell regularizer ensures that the mean prediction remains close to the ground truth, and is averaged across all grid cells along the height $H$ , width $W$ and lead-time $N$ axes. It is weighted towards heavier rainfall targets using the function $w(y)=\operatorname*{max}(y+1,\,24)$ , which operate element-wise for input vectors, and is clipped at 24 for robustness to spuriously large values in the radar.
>  (2), (3) 中在潜在变量 $\mathbf Z$ 上的期望计算使用 Monte Carlo 估计 (用样本的平均值近似)，每个输入 $\mathbf X_{1:M}$ (包含 4 个连续的雷达观测，即 $M=4$) 采样 6 个潜在变量
>  网格正则化项跨高度 $H$，宽度 $W$ 和提前时间 $N$，在所有网格上平均，保证预测的 (加权) 均值接近真实值，加权使得较大的降雨目标权重更大，加权函数为 $w(y) = \max(y+1, 24)$ ($y$ 越大权重越大)，同时在 24 时截断，防止雷达中的异常大值降低方法的健壮性

The GAN spatial discriminator loss ${\mathcal{L}}_{\mathrm{D}}(\phi)$ and temporal discriminator loss ${\mathcal{L}}_{\mathrm{T}}(\psi)$ are minimized with respect to parameters $\phi$ and $\psi$ , respectively; $\mathrm {ReLU}(x)=\max(0,\,x)$ . The discriminator losses use a hinge loss formulation: 

$$
\begin{align}
\mathcal L_{D}(\phi) = \mathbb E_{\mathbf X_{1:M+N},\mathbf Z}[\mathrm{ReLU}(1-D_\phi(\mathbf X_{M+1:M+N}))\\ +\mathrm{ReLU}(1 + D_\phi(G(\mathbf Z;\mathbf X_{1:M})))]\tag{4}
\end{align}
$$

$$
\begin{align}
\mathcal L_T(\psi) = \mathbb E_{\mathbf X_{1:M+N},\mathbf Z}[\mathrm{ReLU}(1-T_\psi(\mathbf X_{1:M+N}))\\
+\mathrm{ReLU}(1+T_\psi(\{\mathbf X_{1:M};G(\mathbf Z;\mathbf X_{1:M})\}))]\tag{5}
\end{align}
$$

>  空间判别器和时间判别器相对于各自参数 $\phi, \psi$ 的损失如上，使用的是铰链损失
>  空间判别器的损失 $\mathcal L_D(\phi)$ 对输入序列 $\mathbf X_{1:M+N}$ 和潜在变量 $\mathbf Z$ 求期望，期望内是两个项的和，第一个项是 $1$ 减去真实场的判别分数 (ReLU 激活)，第二个项是 $1$ 加上预测场的判别分数 (ReLU) 激活，显然，要最小化该损失，真实场的判别分数需要越大越好，预测场的判别分数需要越小越好 (在 $\pm 1$ 的范围内，过大或过小会被 ReLU 限制)
>  时间判别器的损失同理

**Evaluation.** 
During evaluation, the generator architecture is the same, but unless otherwise noted, full radar observations of size $1,536\times1,280$ , and latent variables with height and width 1/32 of the radar observation size $48\times40\times8$ of independent draws from a normal distribution), are used as inputs to the conditioning stack and latent conditioning stack, respectively. In particular, the latent conditioning stack allows for spatiotemporally consistent predictions for much larger regions than those on which the generator is trained. 
>  评估时，传递给条件栈的输入是尺寸为 1536 x 1280 的完整雷达观测图，传递给潜在条件栈的输入是从正态分布中独立抽取的，尺寸为 48 x 40 x 8 的潜在变量 (宽高都为雷达观测的 1/32)
>  特别地，在评估时，潜在条件栈允许生成器在比训练时更大的区域上进行预测 (因为潜在变量的尺寸更大)

For operational purposes and decision-making, the most important aspect of a probabilistic prediction is its resolution. Specific applications will require different requirements on reliability that can often be addressed by post-processing and calibration. We develop one possible post-processing approach to improve the reliability of the generative nowcasts. At prediction time, the latent variables are samples from a Gaussian distribution with standard deviation 2 (rather than 1), relying on empirical insights on maintaining resolution while increasing sample diversity in generative models. In addition, for each realization we apply a stochastic perturbation to the input radar by multiplying a single constant drawn from a unit-mean gamma distribution $G(\alpha\,{=}\,5,\beta\,{=}\,5)$ to the entire input radar field. Extended Data Figures 4 (UK) and 9 (US) shows how the post-processing improves the reliability diagram and rank histogram compared to the uncorrected approach. 
>  概率预测最重要的方面是其分辨率，不同具体应用对可靠性的要求可以通过特定的后处理和校准来解决
>  我们提出一种后处理方法提高生成式临近预报的可靠性，在预测时，我们从标准差为 2 (而不是 1) 的高斯分布中抽取样本，这是为了保持分辨率的同时增加生成模型中样本多样性，另外，对于每次实现，我们通过为整个输入雷达场乘上一个从均值为 1 的伽马分布从采样得到的常数，对输入雷达场进行随机扰动

**Training.** The model is trained for $5\times10^{5}$ generator steps, with two discriminator steps per generator step. The learning rate for the generator is $5\times10^{-5}$ , and is $2\times10^{-4}$ for the discriminator and uses Adam optimizer with $\beta_{1}\!=\!0.0$ and $\beta_{2}\!=\!0.999.$ . The scaling parameter for the grid cell regularization is set to $\lambda\!=\!20$ , as this produced the best continuous ranked probability score results on the validation set. We train on 16 tensor processing unit cores (https://cloud.google.com/tpu) for one week on random crops of the training dataset of size $256\times256$ measurements using a batch size of 16 per training step. 
>  模型训练 $5\times 10^5$ 步，生成器的学习率为 $5\times 10^{-5}$，判别器的学习率为 $2\times 10^{-4}$，网格单元正则化项的缩放系数设定为 $\lambda = 20$
>  我们在 16 个 TPU 上训练了一周，对训练集使用了大小为 256x256 的随机裁剪，批量大小为 16

The Supplementary Information contains additional comparisons showing the contributions of the different loss components to overall performance. We evaluated the speed of sampling by comparing speed on both CPU (10 core AMD EPYC) and GPU (NVIDIA V100) hardware. We generate ten samples and report the median time: for CPU the median time per sample was 25.7 s, and 1.3 s for the GPU. 

### UNet baseline 
We use a UNet encoder–decoder model as strong baseline similarly to how it was used in related studies, but we make architectural and loss function changes that improve its performance at longer lead times and heavier precipitation. First, we replace all convolutional layers with residual blocks, as the latter provided a small but consistent improvement across all prediction thresholds. Second, rather than predicting only a single output and using autoregressive sampling during evaluation, the model predicts all frames in a single forward pass. This somewhat mitigates the excessive blurring found in ref. 5 and improves results on quantitative evaluation. 
>  UNet baseline 的架构和损失函数进行了改进，以提高其对于长提前时间和更强降水下的表现
>  我们首先将所有卷积层替换为残差块，同时将自回归预测改为一次输出所有帧，以缓解过度模糊的表现

Our architecture consists of six residual blocks, where each block doubles the number of channels of the latent representation followed by spatial down-sampling by a factor of two. The representation with the highest resolution has 32 channels which increases up to 1,024 channels. 

Similar to ref. 6, we use a loss weighted by precipitation intensity. Rather than weighting by precipitation bins, however, we reweight the loss directly by the precipitation to improve results on thresholds outside of the bins specified by ref. 6. Additionally, we truncate the maximum weight to $24\,\mathsf{m m}\,\mathsf{h}^{-1}$ as an error in reflectivity of observations leads to larger error in the precipitation values. We also found that including a mean squared error loss made predictions more sensitive to radar artefacts; as a result, the model is only trained with precipitation weighted mean average error loss. 
>  损失函数依旧由降水强度加权
>  我们将最大权重截断为 24 mm/h，因为观测反射率的误差会导致降水量的误差更大

The model is trained with batch size eight for $1\!\times\!10^{6}$ steps, with learning rate $2\times10^{-4}$ with weight decay, using the Adam optimizer with default exponential rates. We select a model using early stopping on the average area under the precision–recall curve on the validation set. The UNet baselines are trained with 4 frames of size $256\times256$ as context. 

### Axial attention baseline 
As a second strong deep learning-based baseline, we adapt the MetNet model, which is a combination of a convolutional long short-term memory (LSTM) encoder and an axial attention decoder, for radar-only nowcasting. MetNet was demonstrated to achieve strong results on short-term (up to 8 h) low precipitation forecasting using radar and satellite data of the continental USA, making per-grid-cell probabilistic predictions and factorizing spatial dependencies using alternating layers of axial attention. 
>  第二个 baseline 是 MetNet + LSTM 编码器和轴向注意力解码器
>  该模型在短期 (最长 8h) 的低降水量预报中取得了较优表现

We modified the axial attention encoder–decoder model to use radar observations only, as well as to cover the spatial and temporal extent of data in this study. We rescaled the targets of the model to improve its performance on forecasts of heavy precipitation events. After evaluation on both UK and US data, we observed that additional satellite or topographical data as well as the spatiotemporal embeddings did not provide statistically significant CSI improvement. An extended description of the model and its adaptations is provided in Supplementary Information section D. 

The only prediction method described in ref. 19 is the per-grid-cell distributional mode, and this is considered the default method for comparison. To ensure the strongest baseline model, we also evaluated other prediction approaches. We assessed using independent samples from the per-grid-cell marginal distributions, but this was not better than using the mode when assessed quantitatively and qualitatively. We also combined the marginal distributions with a Gaussian process copula, in order to incorporate spatiotemporal correlation similar to the stochastically perturbed parametrization tendencies (SPPT) scheme of ref. 40. We used kernels and correlation scales chosen to minimize spatiotemporally pooled CRPS metrics. The best performing was the product of a Gaussian kernel with $25\,\mathrm{km}$ spatial correlation scale, and an AR(1) kernel with 60 min temporal correlation scale. Results, however, were not highly sensitive to these choices. All settings resulted in samples that were not physically plausible, due to the stationary and unconditional correlation structure. These samples were also not favoured by external experts. Hence, we use the mode prediction throughout. 

### PySTEPS baseline 
We use the PySTEPS implementation from ref. 4 using the default configuration available at https://github.com/pySTEPS/pysteps. Refs. 3,4 provide more details of this approach. In our evaluation, unlike other models evaluated that use inputs of size $256\times256$ , PySTEPS is given the advantage of being fed inputs of size $512\times512$ , which was found to improve its performance. PySTEPS includes post-processing using probability matching to recalibrate its predictions and these are used in all results. 

## Performance evaluation 
We evaluate our model and baselines using commonly used quantitative verification measures, as well as qualitatively using a cognitive assessment task with expert meteorologists. Unless otherwise noted, models are trained on years 2016–2018 and evaluated on 2019 (that is, a yearly split). 

### Expert meteorologist study 
The expert meteorologist study described is a two-phase protocol consisting of a ranked comparison task followed by a retrospective recall interview. The study was submitted for ethical assessment to an independent ethics committee and received favourable review. Key elements of the protocol involved consent forms that clearly explained the task and time commitment, clear messaging on the ability to withdraw from the study at any point, and that the study was not an assessment of the meteorologist’s skills and would not affect their employment and role in any way. Meteorologists were not paid for participation, since involvement in these types of studies is considered part of the broader role of the meteorologist. The study was anonymized, and only the study lead had access to the assignment of experimental IDs. The study was restricted to meteorologists in guidance-related roles, that is, meteorologists whose role is to interpret weather forecasts, synthesize forecasts and provide interpretations, warnings and watches. Fifty-six meteorologists agreed to participate in the study. 

Phase 1 of the study, the rating assessment, involved each meteorologist receiving a unique form as part of their experimental evaluation. The axial attention mode prediction is used in the assessment, and this was selected as the most appropriate prediction during the pilot assessment of the protocol by the chief meteorologist. The phase 1 evaluation comprised an initial practice phase of three judgments for meteorologists to understand how to use the form and assign ratings, followed by an experimental phase that involved 20 trials that were different for every meteorologist, and a final case study phase in which all meteorologists rated the same three scenarios (the scenarios in Fig. 1a, and Extended Data Figs. 2 and 3); these three events were chosen by the chief meteorologist—who is independent of the research team and also did not take part in the study—as difficult events that would expose challenges for the nowcasting approaches we compare. Ten meteorologists participated in the subsequent retrospective recall interview. This interview involved an in-person interview in which experts were asked to explain the reasoning for their assigned rating and what aspects informed their decision-making. These interviews all used the same script for consistency, and these sessions were recorded with audio only. Once all the audio was transcribed, the recordings were deleted. 

The 20 trials of the experimental phase were split into two parts, each containing ten trials. The first ten trials comprised medium rain events (rainfall greater than $5\,\mathsf{m m}\,\mathsf{h}^{-1}$ ) and the second 10 trials comprised heavy rain events (rainfall greater than $10\,\mathsf{m m\,h^{-1}}$ ). 141 days from 2019 were chosen by the chief meteorologist as having medium-to-heavy precipitation events. From these dates, radar fields were chosen algorithmically according to the following procedure. First, we excluded from the crop selection procedure the $192\,\mathrm{km}$ that forms the image margins of each side of the radar field. Then, the crop over $256\,\mathrm{km}$ regions, containing the maximum fraction of grid cells above the given threshold, 5 or $10\,\mathsf{m m\,h^{-1}}$ , was selected from the radar image. If there was no precipitation in the frame above the given threshold, the selected crop was the one with the maximum average intensity. We use predictions without post-processing in the study. Each meteorologist assessed a unique set of predictions, which allows us to average over the uncertainty in predictions and individual preference to show statistical effect. 

Extended Data Figure 2 shows a high-intensity precipitation front with decay and Extended Data Fig. 3 shows a cyclonic circulation event (low-pressure area), both of which are difficult for current deep learning models to predict. These two cases were also assessed by all expert meteorologists as part of the evaluative study, and in both cases, meteorologists significantly preferred the generative approach ( ${}\langle n\!=\!56$ , $P\!<\!10^{-4})$ to competing methods. For the high-intensity precipitation front in Extended Data Fig. 2, meteorologists ranked first the generative approach in $73\%$ of cases. Meteorologists reported that DGMR has “decent accuracy with both the shape and intensity of the feature … but loses most of the signal for embedded convection by $T+90^{\prime\prime}$ . PySTEPS is “too extensive with convective cells and lacks the organisation seen in the observations”, and the axial attention model as “highlighting the worst areas” but “looks wrong”. 

For the cyclonic circulation in Extended Data Fig. 3, meteorologists ranked first the generative approach in $73\%$ of cases. Meteorologists reported that it was difficult to judge this case between DGMR and PySTEPS. When making their judgment, they chose DGMR since it has “best fit and rates overall”. DGMR “captures the extent of precipitation overall [in the] area, though slightly overdoes rain coverage between bands”, whereas PySTEPS “looks less spatially accurate as time goes on”. The axial attention model “highlights the area of heaviest rain although its structure is unrealistic and too binary”. We provide additional quotes in Supplementary Information section C.6. 

### Quantitative evaluation 
We evaluate all models using established metrics: CSI, CRPS, Pearson correlation coefficient, the relative economic value, and radially averaged PSD. These are described in Supplementary Information section F. 

To make evaluation computationally feasible, for all metrics except PSD, we evaluate the models on a subsampled test set, consisting of $512\times512$ crops drawn from the full radar images. We use an importance sampling scheme (described in Supplementary Information section A.1) to ensure that this subsampling does not unduly compromise the statistical efficiency of our estimators of the evaluation metrics. The subsampling reduces the size of the test set to 66,851 and Supplementary Information section C.3 shows that results obtained when evaluating CSI are not different when using the dataset with or without subsampling.
>  除了 PSD 外，我们在一个子采样的测试集上评估模型，该测试集由从完整雷达图像中抽取的 512 x 512 裁剪构成，子采样得到的测试集大小为 66851

All models except PySTEPS are given the centre $256\times256$ crop as input. PySTEPS is given the entire $512\times512$ crop as input as this improves its performance. The predictions are evaluated on the centre $64\times64$ grid cells, ensuring that models are not unfairly penalized by boundary effects. Our statistical significance tests use every other week of data in the test set (leaving $n\!=\!26$ weeks) as independent units. We test the null hypothesis that performance metrics are equal for the two models, against the two-sided alternative, using a paired permutation test with $10^{6}$ permutations. 
>  除 PySTEPS 以外，所有模型的输入都是 256 x 256 裁剪，PySETPS 的裁剪是 512 x 512 裁剪，因为这可以提高其性能
>  预测在中心 64 x 64 网格上评估，确保模型不会因为边界效应受到不公平的惩罚

Extended Data Figure 4 shows additional probabilistic metrics that measure the calibration of the evaluated methods. This figure shows a comparison of the relative economic value of the probabilistic methods, showing DGMR providing the best value. We also show how the uncertainty captured by the ensemble increases as the number of samples used is increased from 1 to 20. 

Extended Data Figure 5 compares the performance to that of an NWP, using the UKV deterministic forecast44, showing that NWPs are not competitive in this regime. See Supplementary Information section C.2 for further details of the NWP evaluation. 

To verify other generalization characteristics of our approach—as an alternative to the yearly data split that uses training data of 2016–2018 and tests on 2019—we also use a weekly split: where the training, validation and test sets comprise Thursday through Monday, Tuesday, and Wednesday, respectively. The sizes of the training and test datasets are 1.48 million and 36,106, respectively. Extended Data Figure 6 shows the same competitive verification performance of DGMR in this generalization test. 

To further assess the generalization of our method, we evaluate on a second dataset from the USA using the multi-radar multi-sensitivity (MRMS) dataset, which consists of radar composites for years $2017{-}2019^{45}$ . We use two years for training and one year for testing, and even with this more limited data source, our model still shows competitive performance relative to the other baselines. Extended Data Figs. 7–9 compares all methods on all metrics we have described, showing both the generalization and skilful performance on this second dataset. The Supplementary Information contains additional comparisons on performance with different initializations and performance of different loss function components. 