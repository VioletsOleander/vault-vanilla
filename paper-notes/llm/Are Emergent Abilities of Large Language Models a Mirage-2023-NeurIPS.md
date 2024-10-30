# Abstract
Recent work claims that large language models display emergent abilities, abilities not present in smaller-scale models that are present in larger-scale models. 
> 有工作说 LLM 有涌现能力，小模型没有

What makes emergent abilities intriguing is two-fold: their sharpness, transitioning seemingly instantaneously from not present to present, and their unpredictability, appearing at seemingly unforeseeable model scales. 
> 涌现能力两点吸引人：
> sharpness，突然出现
> unpredictability，不知道在什么 scale 中出现

Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, emergent abilities appear due the researcher’s choice of metric rather than due to fundamental changes in model behavior with scale. 
> 我们认为，对于特定任务和模型系列，分析固定的模型输出时，涌现能力出现的原因是研究者对 metric 的选择，而不是模型随着 scale 变化而行为上的 fundamental change

Specifically, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes in model performance. 
> 非线性或不连续的 metric 似乎会产生涌现能力
> 线性或连续的 metric 则产生平滑、可预测、线性的模型性能改变

We present our alternative explanation in a simple mathematical model, then test it in three complementary ways: we (1) make, test and confirm three predictions on the effect of metric choice using the InstructGPT/GPT-3 family on tasks with claimed emergent abilities, (2) make, test and confirm two predictions about metric choices in a meta-analysis of emergent abilities on BIG-Bench; and (3) show how to choose metrics to produce never-before-seen seemingly emergent abilities in multiple vision tasks across diverse deep networks. Via all three analyses, we provide evidence that alleged emergent abilities evaporate with different metrics or with better statistics, and may not be a fundamental property of scaling AI models.
> 我们用数学模型展示我们的解释，然后用三种互补的方式测试它
> 其一，使用 InstructGPT/GPT-3 家族，在声称有涌现能力的任务上，对于我们关于 metric choice 影响涌现能力的解释，做出了三个预测，并且测试了这三个预测，测试结果确认了我们的解释
> 其二，在 BIG-Bench 上，对于我们关于 metric choice 影响涌现能力的解释，做了两个预测，并通过元分析测试了这两个预测，测试结果确认了我们的解释
> 其三，展示如何选择 metric 以在多个视觉任务上产生涌现能力的效果
> 通过这三个分析，我们提供了证据，断言涌现能力随着不同的 metrics 或更好的 statistics 而消失，且涌现能力或许不是 scaling AI 模型的 fundamental 属性
# 1 Introduction
![[Are Emergent Abilities of Large Language Models a Mirage-Fig1.png]]

Emergent properties of complex systems have long been studied across disciplines, from physics to biology to mathematics. The idea of emergence was popularized by Nobel Prize-winning physicist P.W. Anderson’s “More Is Different” [1], which argues that as the complexity of a system increases, new properties may materialize that cannot be predicted even from a precise quantitative understanding of the system’s microscopic details. Recently, the idea of emergence gained significant attention in machine learning due to observations that large language models (LLMs) such as GPT [3], PaLM [6] and LaMDA [30] exhibit so-called “emergent abilities” [33, 8, 28, 3] (Fig. 1).
> 涌现能力：随着系统复杂性增加，新的属性会实质化/出现，它的出现无法预测

The term “emergent abilities of LLMs” was recently and crisply defined as “abilities that are not present in smaller-scale models but are present in large-scale models; thus they cannot be predicted by simply extrapolating the performance improvements on smaller-scale models” [33]. Such emergent abilities were first discovered in the GPT-3 family [3]. Subsequent work emphasized the discovery, writing that “\[although model\] performance is predictable at a general level, performance on a specific task can sometimes emerge quite unpredictably and abruptly at scale” [8]. These quotations collectively identify the two defining properties of emergent abilities in LLMs:

1. Sharpness, transitioning seemingly instantaneously from not present to present
2. Unpredictability, transitioning at seemingly unforeseeable model scales

> 先前的工作总结出涌现能力的两点特性: Sharpness, Unpredictability

These emergent abilities have garnered significant interest, raising questions such as: What controls which abilities will emerge? What controls when abilities will emerge? How can we make desirable abilities emerge faster, and ensure undesirable abilities never emerge? These questions are especially pertinent to AI safety and alignment, as emergent abilities forewarn that larger models might one day, without warning, acquire undesired mastery over dangerous capabilities [29, 10, 17, 18].

In this paper, we call into question the claim that LLMs possess emergent abilities, by which we specifically mean sharp and unpredictable changes in model outputs as a function of model scale on specific tasks. Our doubt stems from the observation that emergent abilities seem to appear only under metrics that nonlinearly or discontinuously scale any model’s per-token error rate. 
> 本文对于 LLM 的涌现能力提出质疑，本文针对的涌现能力即模型在特定任务上，随着模型 scale，模型的 output 出现 sharp and unpredictable 的改变
> 我们的质疑来源于我们观察到涌现能力似乎仅在非线性或不连续地 scale 任意模型地 per-token 错误率的 metric 上出现

For instance, as we later show, > 92% of emergent abilities on BIG-Bench tasks [28] (hand-annotated by [32]) appear under either of these two metrics:
> 例如，在 BIG-Bench 上，超过92%的涌现能力在以下两种 metrics 中出现:
> 多选: 正确选项的概率最高时为 1
> 完全字符串匹配: 输出字符串完全匹配目标时为 1

$$
\begin{align}
\text{Multiple Choice Grade}&:=\begin{cases}1\quad \text{if highest probability mass on correct option}\\ 0\quad \text{otherwise}\end{cases}
\\
\\
\text{Exact String Match}&:=\begin{cases}1\quad \text{if output string exactly matches target string}\\0\quad \text{otherwise}\end{cases}
\end{align}
$$

This raises the possibility of an alternative explanation for the origin of LLMs’ emergent abilities: sharp and unpredictable changes might be induced by the researcher’s choice of measurement, even though the model family’s per-token error rate changes smoothly, continuously and predictably with increasing scale. Specifically, our alternative posits that emergent abilities are a mirage caused primarily by the researcher choosing a metric that nonlinearly or discontinuously deforms per-token error rates, and secondarily by possessing too few test data to accuratly estimate the performance of smaller models, thereby causing smaller models to appear wholly unable to perform the task.
> 这为解释 LLM 的涌现能力提供了一个可替代的解释: sharp and unpredictable changes 可能是由研究者对于 measurement 的选择而引入的，而模型家族的 per-token 错误率则是随着 scale 平滑并且可预测地增长
> 我们认为涌现能力仅仅是一个 mirage，它主要是由于研究者选择了一个改变了 per-token 错误率的非线性的和不连续的 metric 导致，其次是由于处理了过少的测试数据以至于不能正确评估小模型的表现，导致小模型看起来完全不能执行任务

To communicate our alternative explanation, we present it as a simple mathematical model and demonstrate how it quantitatively reproduces the evidence offered in support of emergent abilities of LLMs. 
> 我们以数学模型的形式展示我们的解释，并展示它如何量化地重现支持 LLM 的涌现能力的 evidence

We then test our alternative explanation in three complementary ways:
> 我们用三个互补的方式测试我们的解释

1. We make, test and confirm three predictions based on our alternative hypotheses using the InstructGPT [24] / GPT-3 [3] model family. 
2. We meta-analyze published benchmarks [28, 33] to reveal that emergent abilities only appear for specific metrics, not for model families on particular tasks, and that changing the metric causes the emergence phenomenon to evaporate.
3. We induce never-before-seen, seemingly emergent abilities in multiple architectures across various vision tasks by intentionally changing the metrics used for evaluation.

> 1. 基于我们的解释，使用 InstructGPT/GPT-3家族，做出、测试并确定三个预测
> 2. 元分析已发布的 benchmarks，以揭示涌现能力仅针对特定 metrics 出现，而不是针对在特定任务上的模型家族，并且，改变 metric 会导致涌现现象消失
> 3. 我们在多个视觉任务上，通过刻意改变用于 evaluation 的 metrics，引入了从未见过的似是而非的涌现能力
# 2 Alternative Explanation for Emergent Abilities
How might smooth, continuous, predictable changes in model family performance appear sharp and unpredictable? The answer is that the researcher’s choice of a nonlinear or discontinuous metric can distort the model family’s performance to appear sharp and unpredictable.
> 研究者选择了非线性或不连续的 metric，使得模型的平滑、连续、可预测的 performance 变化看起来是 sharp 和 unpredictable 的


![[Are Emergent Abilities of Large Language Models a Mirage-Fig2.png]]

To expound, suppose that within a model family, the test loss falls smoothly, continuously and predictably with the number of model parameters. One reason to believe this is the phenomenon known as neural scaling laws: empirical observations that deep networks exhibit power law scaling in the test loss as a function of training dataset size, number of parameters or compute [13, 27, 11, 16, 9, 12, 15, 34, 14, 7, 26]. 
> 假设模型家族内，测试损失是随着模型参数的数量平滑、连续、可预测地下降的
> 该假设有一个支撑理由: neural scaling law
> neural scaling law: 经验性的观察发现，深度网络在测试损失上呈现出相对于训练数据集大小、需要计算的参数数量的幂律缩放 power law scaling

For concreteness, suppose we have a model family of different numbers of parameters $N>0$ and assume that each model’s per-token cross entropy falls as a power law with the number of parameters $N$ for constants $c>0$;  $\alpha < 0$ (Fig. 2A):
> 更具体地说，假设我们有一族参数数量 $N$ 不同的模型，假设每个模型的 per-token 交叉熵损失随着 $N$ 呈现幂律下降($c > 0, \alpha < 0$)

$$
\mathcal L_{CE}(N) = \left(\frac N c\right)^{\alpha}
$$

To be clear, we do not require this particular functional form to hold; rather, we use it for illustrative purposes.
> 但我们并不要求它满足这个特定的函数形式

 Let $V$ denote the set of possible tokens, $p\in \Delta^{|V|-1}$ denote the true but unknown probability distribution, and $\hat p_N\in \Delta^{|V|-1}$ denote the $N$ -parameter model’s predicted probability distribution. The per-token cross entropy as a function of number of parameters $N$ is:
> 令 $V$ 表示词袋，$p\in \Delta^{|V|-1}$ 表示真实的但是未知的概率分布，$\hat p_N \in \Delta^{|V|-1}$ 表示参数量 $N$ 的模型预测的概率分布
> 则 per-token 的交叉熵写为 $N$ 的函数是:

$$
\mathcal L_{CE}(N) :=-\sum_{v\in V}p(v)\log \hat p_N(v)
$$
In practice, $p$ is unknown, so we substitute a one-hot distribution of the observed token $v^*$:
> 在实践中，真实的概率分布 $p$ 是未知的，因此我们将它替换为观察到的 token $v^*$ 的 one-hot 分布

$$
\mathcal L_{CE}(N)= -\log \hat p_N(v^*)
$$

A model with $N$ parameters then has a per-token probability of selecting the correct token (Fig. 2B):
> 参数量为 $N$ 的模型会计算出 per-token 的概率，以选择正确的 token

$$
p(\text{single token correct}) = \exp(-\mathcal L_{CE}(N)) = \exp(-(N/c)^{\alpha})
$$
(
在 per-token 测试损失随着参数 $N$ 呈现单调幂律下降的前提下，$p(\text{single token correct})$ 会随着参数 $N$ 单调趋近于 $\exp(0) = 1$
)

Suppose the researcher then chooses a metric that requires selecting $L$ tokens correctly. For example, our task might be $L$ -digit integer addition, and a model’s output is scored 1 if all $L$ output digits exactly match all target digits with no additions, deletions or substitutions, 0 otherwise. 
> 假设研究者然后选择了需要正确选择 $L$ 个 token 的 metric
> 例如，任务是 $L$ 位整数加法，模型的输出需要 $L$ 个输出 digit 准确匹配目标 digit 才得到 score 1，否则 score 0

If the probability each token is correct is independent, the probability of scoring 1 is:
> 如果每个 token 的概率都是正确的且独立的，则 scoring 1的概率是:

$$
\text{Accuracy}(N) \approx p_N(\text{single token correct})^{\text{num. of tokens}} = \exp(-(N/c)^{\alpha})^L
$$

This choice of metric nonlinearly scales performance with increasing token sequence length. When plotting performance on a linear-log plot, one sees a sharp, unpredictable emergent ability on longer sequences (Fig. 2C) that closely matches claimed emergent abilities (inset). 
> 这个 metric 会在 token 序列长度增长时非线性地 scale performance，当在 linear-log 图上画出 performance 时，可以看到长序列上出现 sharp, unpredictable 的涌现能力，和之前研究声称的涌现能力很匹配

What happens if the researcher switches from a nonlinear metric like Accuracy, under which the per-token error rate scales geometrically in target length (App. A.3), to an approximately linear metric like Token Edit Distance, under which the per-token error rate scales quasi-linearly in target length (App. A.2)?
> 那么，此时我们考虑将非线性的 metric 例如 Accuracy (相对于 accuracy，per-token 的错误率是随着目标长度几何式 scale 的) 替换为一个近似线性的 metric，例如 Token Edit Distance (相对于 token edit distance，per-token 的错误率是随着目标长度近线性地 scale 的)

$$
\text{Token Edit Distance}(N)\approx L(1-p_N(\text{single token correct})) = L(1-\exp(-(N/c)^{\alpha})
$$

The linear metric reveals smooth, continuous, predictable changes in model performance (Fig. 2E). 
> 线性的 metric 就反映出了模型 performance 的改变是平滑、连续、可预测的

Similarly, if the researcher uses a discontinuous metric like Multiple Choice Grade, the researcher can find emergent abilities (Fig. 2D), but switching to a continuous metric like Brier Score removes the emergent ability (Fig. 2F).
> 类似地，如果研究者使用不连续的 metric，例如 Multiple Choice Grade，则会发现涌现能力，替换为连续的 metric 例如 Brier Score，则没有涌现能力

 In summary, sharp and unpredictable changes with increasing scale can be fully explained by three interpretable factors: (1) the researcher choosing a metric that nonlinearly or discontinuously scales the per-token error rate, (2) having insufficient resolution to estimate model performance in the smaller parameter regime, with resolution set by $1/\text{test dataset size}$, and (3)  insufficiently sampling the larger parameter regime.
 > 总结: 随着 scale 增大而出现的 sharp, unpredictable 的 change 可以用三个可以解释的因素来解释:
 > 1. 研究者选择了相对于 per-token 错误率非线性或不连续地 scale 的度量
 > 2. 对于小参数模型的 performance 的评估的分辨率 (数据集的精细程度) 不足
 > 3. 对于大参数模型的采样不足
# 3 Analyzing InstructGPT/GPT-3’s Emergent Arithmetic Abilities
Previous papers prominently claimed the GPT [3, 24] family displays emergent abilities at integer arithmetic tasks [8, 28, 33] (Fig. 2E). We chose these tasks as they were prominently presented [3, 8, 28, 33], and we focused on the GPT family due to it being publicly queryable.
> 之前的研究称 GPT 系列在整数算数任务中展现了涌现能力，同时因为 GPT 可以公开访问，我们选择 GPT 系列和算数任务进行研究

 As explained mathematically and visually in Sec. 2, our alternative explanation makes three predictions:
 
1. Changing the metric from a nonlinear or discontinuous metric (Fig. 2CD) to a linear or continuous metric (Fig. 2 EF) should reveal smooth, continuous, predictable performance improvement with model scale.
2. For nonlinear metrics, increasing the resolution of measured model performance by increasing the test dataset size should reveal smooth, continuous, predictable model improvements commensurate with the predictable nonlinear effect of the chosen metric.
3. Regardless of metric, increasing the target string length should predictably affect the model’s performance as a function of the length-1 target performance: approximately geometrically for accuracy and approximately quasilinearly for token edit distance.

> 我们的解释做了三个预测
> 1. 将 metric 从非线性或不连续改为线性或连续后，performance 应该相对于 model scale 展现出平滑、连续、可预测的提升
> 2. 对于非线性 metric，通过提高测试数据集的大小来提高 resolution of measured model performance 后，应该会揭示出平滑、连续、可预测的，以及和选定的 metric 的非线性效果一致的 model improvement
> 3. 不考虑 metric，提高目标 string 长度应该可以可预测地以按照 length-1 target performance 的函数的形式影响 model performance，即对于 accurady 是近似几何地，以及对于 token edit distance 是近似线性的

To test these predictions, we collected outputs from the InstructGPT/GPT-3 family on two tasks: 2-shot multiplication between two 2-digit integers and 2-shot addition between two 4-digit integers.
> 我们从 InstructGPT/GPT-3系列中收集对于两个任务: 对于两个 2-digit 整数的 2-shot 乘法和两个 4-digit 整数的 2-shot 加法的输出，以测试这些预测

![[Are Emergent Abilities of Large Language Models a Mirage-Fig3.png]]

**Prediction: Emergent Abilities Disappear With Different Metrics** 
On both arithmetic tasks, the GPT family displays emergent abilities if the target has 4 or 5 digits and if the metric is Accuracy (Fig. 3, top) [3, 8, 33]. However, if one changes from nonlinear Accuracy to linear Token Edit Distance while keeping the models’ outputs fixed, the family’s performance smoothly, continuously and predictably improves with increasing scale (Fig. 3, bottom). This confirms our first prediction and supports our alternative explanation that the source of emergent abilities is the researcher’s choice of metric, not changes in the model family’s outputs. We also observe that under Token Edit Distance, increasing the length of the target string from 1 to 5 predictably decreases the family’s performance in an approximately quasilinear manner, confirming the first half of our third prediction.
> 在两个算数任务上，对于同样的输出，metric 是 Accuracy 时，就会出现涌现能力，metric 时 Token Edit Distance 时，就会出现平滑、连续、可预测的 scale
> 这确认了我们的第一个预测，并支持了我们的解释：涌现能力的来源是研究者对于 metric 的选择，而不是模型系列输出的改变
> 同时可以看到，在 Token Edit Distance 下，增长目标 string 的长度从1到5会可预测地，以近似线性的形式降低模型系列的 performance

![[Are Emergent Abilities of Large Language Models a Mirage-Fig4.png]]

**Prediction: Emergent Abilities Disappear With Better Statistics** 
We next tested our second prediction: that even on nonlinear metrics such as accuracy, smaller models do not have zero accuracy, but rather have non-zero above-chance accuracy commensurate with choosing to use accuracy as the metric. In order to accurately measure models’ accuracy, we increased the resolution by generating additional test data, and found that on both arithmetic tasks, all models in the InstructGPT/GPT-3 family achieve above-chance accuracy (Fig. 4). This confirms our second prediction. We also observe that as the target string length increases, the accuracy falls approximately geometrically with the length of the target string, confirming the second half of our third prediction. These results additionally demonstrate that the researcher’s choice of metric has the effect that one should predict accuracy to have, i.e., geometric decay with the target length.
> 我们接着测试我们的第二个预测：即使是对于非线性的 metrics 例如 accuracy，小模型也并没有 zero accuracy，而是也有高于 50%的 accuracy
> 为了正确衡量模型的 accuracy，我们通过生成额外的测试数据提高 resolution，然后发现在算数任务上，所有的 GPT-3系列的模型达到了 50%以上的准确率
> 这确认了我们的第二个预测
> 我们也观察到随着目标 string 长度增加，accuracy 的下降近似和 target string 的长度成几何关系
# 4 Meta-Analysis of Claimed Emergent Abilities
Analyzing the GPT family is possible because the models are publicly queryable. However, other model families claimed to exhibit emergent abilities are not publicly queryable, nor are their generated outputs publicly available, meaning we are limited to analyzing the published results themselves [8, 33, 32]. 
> 其他声称有涌现能力的模型系列不可以公开访问，同时它们生成的结果也不是公开可用的，因此我们分析它们发布的结果

Our alternative explanation makes two predictions.

1. At the “population level” of Task-Metric-Model Family triplets, emergent abilities should appear predominantly on specific metrics, not task-model family pairs, and specifically with nonlinear and/or discontinuous metrics.
2. On individual Task-Metric-Model Family triplets that display an emergent ability, changing the metric to a linear and/or continuous metric should remove the emergent ability.
> 我们的解释做了两个预测：
> 1. 在任务-度量-模型系列三元组上，涌现能力应该主要在特定 metric 上出现，而不是任务-模型系列对上，并且主要应该在非线性或不连续的 metric 上
> 2. 在单个展现了涌现能力的任务-度量-模型系列三元组上，将度量转换为线性或连续的度量应该移除涌现能力

To test these predictions, we used to claimed emergent abilities on BIG-Bench [28, 33] due to the benchmark being pertinent and publicly available.

**Prediction: Emergent Abilities Should Appear with Metrics, not Task-Model Families** If emergent abilities are real, one should expect task-model family pairs to show emergence for all reasonable metrics. However, if our alternative explanation is correct, we should expect emergent abilities to appear only under certain metrics. 
> 预测：涌现能力应该随着 metric 出现，而不是随着任务-模型系列出现
> 如果涌现能力是真实的，我们应该可以看到任务-模型系列 pair 对于所有 reasonable 的 metric 都会出现
> 而如果我们的解释正确，我们应该可以看到涌现能力仅在特定 metric 下出现

To test this, we analyzed on which metrics emergent abilities appear. To determine whether a task-metric-model family triplet exhibits a possible emergent ability, we used a metric from previous work [28]. Letting $y_i\in \mathbb R$ denote model performance at model scales $x_i\in \mathbb R$, sorted such that $x_i < x_{i+1}$, the emergence score is:
> 我们在出现涌现能力的 metric 上进行了分析
> 为了决定是否 task-metric-model family 三元组展现了涌现能力，我们使用了以下 metric，其中 $y_i \in \mathbb R$ 表示模型在 scale $x_i \in \mathbb R$ 的表现，模型 scale 满足 $x_i < x_{i+1}$ ，因此 emergence score 写为:

$$
\text{Emergence Score}(\{(x_n,y_n)\}_{n=1}^N):=\frac {\text{sign}(\arg\max_i y_i-\arg\min_i y_i)(\max_i y_i -\min_i y_i)}{\sqrt{\text{Median}(\lbrace (y_i-y_{i-1})^2\rbrace_i)}}
$$

![[Are Emergent Abilities of Large Language Models a Mirage-Fig5.png]]

We found that most metrics used in BIG-Bench have *zero* task-model family pairs that exhibit emergent abilities: of the 39 preferred metrics in BIG-Bench, at most 5 display emergence (Fig. 5A). Many of the 5 are nonlinear and/or discontinuous, e.g., Exact String Match, Multiple Choice Grade, ROUGE-L-Sum (App. A.4). Notably, because BIG-Bench often scores models on tasks using multiple metrics, the lack of emergent abilities under other metrics suggests that emergent abilities do not appear when model outputs are scored using other metrics.
> 我们发现 BIG-Bench 使用的多数 metric 没有 task-model family pairs 展现出涌现能力，39个 metric 中仅有 5 个 metric 展现了涌现能力
> 这 5 个也多数是不连续或非线性的
> 并且由于 BIG-Bench 会使用多个 metric 在任务上 score model，模型在某个 metric 上缺乏涌现能力也表明了模型输出使用其他 metrics 也不会展现涌现能力

Because emergence score only *suggests* emergence, we also analyzed hand-annotated task-metric-model family triplets [32], which revealed emergent abilities appear with 4/39 metrics (Fig. 5B), and 2 metrics account for > 92% of claimed emergent abilities (Fig. 5C): Multiple Choice Grade and Exact String Match. Multiple Choice Grade is discontinuous, and Exact String Match is nonlinear.
> 因为 emergence score 仅仅 suggests emergence，我们也对手工标记的 task-metric-model family triplet 进行了分析，发现了涌现能力仅在 4/39个 metric 上出现，并且其中 2 个 metric 占据了 92% 声称的涌现能力: Multiple Choice Grade 和 Exact String Match

**Prediction: Changing Metric Removes Emergent Abilities** 
To test our second prediction, we focused on the LaMDA family [30] because its outputs are available through BIG-Bench. For our analysis, we identified tasks on which LaMDA displays emergent abilities with Multiple Choice Grade, then asked whether LaMDA still displays emergent abilities on the same tasks with a different BIG-Bench metric: Brier Score [2]. Brier Score is a strictly proper scoring rule for predictions of mutually exclusive outcomes; for a binary outcome, the Brier Score simplifies to the mean squared error between the outcome and its predicted probability mass. LaMDA’s emergent abilities on the discontinuous Multiple Choice Grade disappeared when we changed the metric to the continuous Brier Score (Fig. 6). These results support our alternative explanation that emergent abilities are induced by the chosen metric.
> 预测：改变 metric 移除涌现能力
> 我们选出 LaMDA 在 Multiple Choice Grade 上出现涌现能力的任务，然后查看 LaMDA 是否在相同任务，不同 metric (Brier Score)上展现出涌现能力
> Brier Score 对于二元的结果会简化为输出和预测概率质量之间的均方误差
> 切换到连续的 Brier Score 之后，LaMDA 的涌现能力消失了
> 这些结果支持了我们的解释：涌现能力来自于 metric 选择
# 5 Inducing Emergent Abilities in Networks on Vision Tasks
To demonstrate how emergent abilities can be induced by the researcher’s choice of metric, we show how to produce emergent abilities in deep networks of various architectures: fully connected, convolutional, self-attentional. We focus on vision tasks because abrupt transitions in vision models’ capabilities have not been observed to the best of our knowledge; this is one reason why emergence in large language models is considered so interesting. For the convolutional example, see App. B.
> 我们进一步展示如何在各种结构的深度网络中制造涌现能力
> 我们聚焦于视觉任务，因为目前没有研究提出视觉任务会出现涌现现象

![[Are Emergent Abilities of Large Language Models a Mirage-Fig7.png]]

**Emergent Reconstruction of CIFAR100 Natural Images by Nonlinear Autoencoders** 
> 非线性 AE 在 CIFAR100 上的涌现重构

We first induce an emergent ability to reconstruct images in shallow (i.e., single hidden layer) nonlinear autoencoders trained on CIFAR100 natural images [19]. To emphasize that the sharpness of the metric is responsible for emergent abilities, and to show that sharpness extends to metrics beyond Accuracy, we intentionally define a discontinuous metric that measures a network’s ability to reconstruct a dataset as the average number of test data with squared reconstruction error below threshold $c$:
> 我们首先在训练与 CIFAR100 的单隐藏层的非线性 AE 中引入涌现能力
> 为了强调 metric 的 sharpness 是涌现的来源，我们定义了一个非线性的 metric 来衡量网络重构数据集的能力
> 该 metric 定义为测试数据集中样本的平方重构损失小于阈值 $c$ 的平均数量

$$
\text{Reconstruction}_c(\lbrace x_n\rbrace_{n=1}^N) :=\frac 1N \sum_n\mathbb I[\|x_n -\hat x_n\|^2 < c]
$$

where $\mathbb I(\cdot)$ denotes an indicator variable and $\hat x_n$ is the autoencoder’s reconstruction of $x_n$. The autoencoder family displays smoothly decreasing squared reconstruction error as the number of bottleneck units increases (Fig. 7B). Under our newly defined Reconstructionc metric and for particular choices of $c$, the autoencoder family exhibits a sharp and seemingly unpredictable image reconstruction ability (Fig. 7C) that qualitatively matches published emergent abilities (Fig. 7A).
> Fig7中可以看到，在平常的均方重构误差 $\frac 1 N \sum_n \|x_n - \hat x_n \|^2$ 下，随着 AE 模型的bottleneck 单元的数量增长，误差平滑地下降
> 在我们定义的 metric 下和特定的 $c$ 下，AE 模型展现出了涌现能力

**Emergent Classification of Omniglot Characters by Autoregressive Transformers** 
> 自回归 Transformer 在多语言字符分类上的涌现

We next induce emergent abilities in Transformers [31] trained to autoregressively classify Omniglot handwritten characters [20], in a setup inspired by recent work [5]: Omniglot images are embedded by convolutional layers, then sequences of embedded image-image class label pairs are fed into decoder-only transformers. We measure image classification performance on sequences of length $L \in [1, 5]$, again via *subset accuracy*: 1 if all $L$ images are classified correctly (Fig. 8B), 0 otherwise. Causal transformers display a seemingly emergent ability to correctly classify Omniglot handwritten characters (Fig. 8C) that qualitatively matches published emergent abilities (Fig. 8A).
> 我们为自回归训练以分类多语言手写字符的 Transformer 引入涌现
> 多语言的图像先通过卷积层 embed，然后 embedded image-image 类别标签对序列会被 feed into decoder-only 的 transformer
> 我们在长度为 $L\in[1,5]$ 的序列上，通过 subset accuracy 衡量图像分类表现，如果全部 $L$ 个图像分类正确，score 1，否则 score 0
# 6 Related Work
Srivastava et al. [28] observed that while accuracy at a particular task can empirically appear sharp and unpredictable, cross entropy does not; the authors then hypothesized that emergent abilities may be partially attributed to the metric. Our paper converts their discussion into precise predictions, then quantitatively tests the predictions to reveal that: metric choice is likely wholly responsible for emergent abilities; well-known and widely-used metrics (including ones already used by [28]) capture graded improvements; emergent abilities do not appear only for tasks involving multiple steps, and indeed appear most commonly on the discontinuous Multiple Choice Grade; metric choice can be used to induce emergent abilities in a novel domain (vision) in diverse architectures and tasks.
> [28] 观察到 accuracy 在特定任务会出现 sharp 和 unpredictable，而 cross entropy 不会，其作者假设涌现能力或许部分会归因于 metric
> 我们将它们的讨论转化为精确的预测，并量化地测试这些预测，揭示了：metric 选择可能完全是涌现的源头

Caballero et al. [4] explain emergence by assuming a piece-wise power law functional form; under this view, emergent abilities are real, caused by a change in the governing power law. In contrast, our work suggests that emergent abilities are induced by the researcher, even under a single power law. Michaud et al. [25] posit that emergent abilities may be real under strong data assumptions.
> [4] 假设了一个分段幂律函数解释涌现，[4] 认为涌现是真实的，是由控制模型性能的幂律关系发生变化导致的
> 我们认为涌现由研究者引入，即便是在单一的幂律函数下
> [25] 认为强的数据假设下涌现会出现 (数据数量和质量高)
# 7 Discussion
Our paper presents an alternative explanation for claimed emergent abilities of large language models. For a fixed task and a fixed model family, the researcher can choose a metric to create an emergent ability or choose a metric to ablate an emergent ability. Ergo, *emergent abilities may be creations of the researcher’s choices, not a fundamental property of the model family on the specific task.* We emphasize that nothing in this paper should be interpreted as claiming that large language models cannot display emergent abilities; rather, our message is that previously claimed emergent abilities in [3, 8, 28, 33] might likely be a mirage induced by researcher analyses.
> 本文为 LLM 的涌现能力提出了另一个解释
> 对于固定的任务和固定的模型系列，研究者可以通过选择 metric 来创建涌现能力或消融涌现能力
> 因此涌现不是模型系列在特定任务上的基本属性
> 我们强调本文不意味着 LLM 不会展现涌现能力，而是说明了先前声称的涌现能力可能只是研究者引入的 mirage

Our paper has several implications. Firstly, a task and a metric are distinct and meaningful choices when constructing a benchmark. Secondly, when choosing metric(s), one should consider the metric’s effect on the per-token error rate and adapt their measuring process accordingly, e.g., if one chooses accuracy, one should make sure to have sufficient data to accurately measure accuracy to avoid the risk of drawing invalid scientific conclusions. Thirdly, when making claims about capabilities of large models, including proper controls is critical. In this particular setting, emergent abilities claims are possibly infected by a failure to control for multiple comparisons. In BIG-Bench alone, there are ≥ 220 tasks, ∼ 40 metrics per task, ~ 10 model families, for a total of ∼ 106 task-metric-model family triplets, meaning probability that no task-metric-model family triplet exhibits an emergent ability by random chance might be small. Fourthly, scientific progress can be hampered when models and their outputs are not made public for independent scientific investigation.
> 本文的含意：
> 构建 benchmark 时，任务和度量的选择都是有意义的
> 选择度量时，需要考虑它对于 per-token 的影响，并相应考虑合适的度量过程，例如如果选择了 accuracy，应该确保有足够数据衡量 accuracy，避免得到无效的结论
> 声称大模型的能力时，要考虑适当地控制变量
> 不公开模型和输出妨碍科学研究
# A Approximate Behavior of Metrics on Sequential Data
How do different metrics behave when used to measure autoregressive model outputs? Precisely answering this question is tricky and possibly analytically unsolvable, so we provide an approximate answer here.
> 这里提供对不同的度量用于衡量自回归模型输出时的行为的一个近似的分析

Notationally, we consider $N$ test data of length $L$ (here, length is measured in tokens) with targets denoted $t_n :=(t_{n1},t_{n2},\dots,t_{nL})$, the autoregressive model has a true-but-unknown per-token error probability of $\epsilon \in[0,1]$ and the model outputs prediction  $\hat t_n :=(\hat t_{n1}, \hat t_{n2}, \dots , \hat t_{nL})$. This assumes that the model’s per-token error probability is constant, which is empirically false, but modeling the complex dependencies of errors is beyond our scope.
> 我们考虑 $N$ 个长度为 $L$ 个 token 的测试数据，目标记为 $t_n := (t_{n1}, t_{n2}, \dots , t_{nL})$，模型的输出预测记为 $\hat t_n := (\hat t_{n1}, \hat t_{n2}, \dots, \hat t_{nL})$
> 自回归模型有一个真实的但是未知的 per-token 错误概率 $\epsilon \in [0,1]$，这里假设 per-token 错误概率是常数，这在经验上一般是错误的
## A.1 Per-Token Error Probability is Resolution-Limited
Note that because we have $N$ test data, each of length $L$, our resolution for viewing the per-token error probability $\epsilon$ is limited by $1/NL$. Here, resolution refers to “the smallest interval measurable by a scientific instrument; the resolving power.” To explain what resolution means via an example, suppose one wants to measure a coin’s probability of yielding heads. After a single coin flip, only two outcomes are possible $(H, T)$, so the resolution-limited probability of heads is either $0$ or $1$. After two coin flips, four outcomes are possible $(HH, HT, TH, TT)$, so the resolution-limited probability of heads is now one of $0,0.5,1$. After $F$ coin flips, we can only resolve the coin’s probability of yielding heads up to $1/F$. 
> 我们有 $N$ 个测试数据，每个长度为 $L$，我们用于评估 per-token 错误率 $\epsilon$  的 resolution 被限制为 $1/NL$
> 也就是说，根据测试数据集 $NL$ 个 token 统计 token 错误率时，错误率之间的差值的最小单位 (interval) 就是 $1 /{NL}$，即错误率只能是 $0, 1/NL, 2/NL, \dots , 1$

Consequently, we introduce a resolution-limited notation:
$$
\lfloor a\rceil_b :=a \text{ rounded to the nearest integer mutiple of } 1/b
$$
> 我们引入分辨率限制标记 $\lfloor a \rceil_b$，表示将 $a$ 归约到离它最近的 $1/b$ 的倍数
## A.2 Token Edit Distance
We first consider an adaptation of the Levenshtein (string edit) distance for models that function on tokens rather than characters, an adaptation we term the token edit distance. The token edit distance between two token sequences $t_n,\hat t_n$ is defined as the integer number of additions, deletions or substitutions necessary to transform $t_n$ into $\hat t_n$ (or vice versa).
>我们首先考虑将 Levenshtein (字符串编辑) 距离适应于基于 token 而非字符运作的模型，我们将这种适应称为 token 编辑距离
>两个 token 序列 $t_n, \hat t_n$ 之间的 token 编辑距离被定义为将 $t_n$ 转换为 $\hat t_n$ (或反之)所需的添加、删除或替换操作的整数次数

$$
\begin{align}
\text{Token Edit Distance}(t_n, \hat t_n) := &\text{Num Substitutions} + \text{Num. Additions} + \text{Num. Deletions}\\
=&\sum_{\mathscr l=1}^L \mathbb I[t_{n\mathscr l}\ne \hat t_{n\mathscr l}] + \text{Num. Additions} + \text{Num. Deletions}\\
\ge&\sum_{\mathscr l=1}^L \mathbb I[t_{n\mathscr l} \ne \hat t_{n\mathscr l}]
\end{align}
$$

The expected token edit distance is therefore:

$$
\begin{align}
\mathbb E[\text{Token Edit Distance}(t_n,\hat t_n)]
&\ge \mathbb E[\sum_{\mathscr l=1}^L\mathbb I[t_{n\mathscr l}\ne \hat t_{n\mathscr l}]]\\
&= \sum_{\mathscr l=1}^Lp(t_{n\mathscr l}\ne \hat t_{n\mathscr l})\\
&\approx L(1 - \epsilon)
\end{align}
$$

The resolution-limited expected token edit distance is therefore:

$$
\lfloor \mathbb E[\text{Token Edit Distance}(t_n,\hat t_n)]\rceil_{NL}\ge L(1-\lfloor \epsilon \rceil_{NL})
$$

From this, we see that the expected token edit distance scales approximately linearly with the resolution-limited per-token probability. The real rate is slightly higher than linear because additions and deletions contribute an additional non-negative cost, but modeling this requires a model of how likely the model is to overproduce or underproduce tokens, which is something we do not currently possess.
> 可以看到期望的 token edit distance 近似线性地随着 resolution-limited 的 per-token 概率 scale
## A.3 Accuracy
$$
\begin{align}
\text{Accuracy}(t_n, \hat t_n) &:= \mathbb I[\text{No additions}]\mathbb I[\text{No deletions}]\prod_{l=1}^L \mathbb I[t_{nl} = \hat t_{nl}]\\
&\approx \prod_{l=1}^L\mathbb I[t_{nl} = \hat t_{nl}]
\end{align}
$$

As with the Token Edit Distance (App. A.3), we ignore how likely the language model is to overproduce or underproduce tokens because we do not have a good model of this process. Continuing along,

$$
\begin{align}
\mathbb E[\log \text{Accuracy}] &= \sum_l \mathbb E[\log \mathbb I[t_{nl} = \hat t_{nl}]]\\
&\le \sum_l\log \mathbb E[\mathbb I[t_{nl} = \hat t_{nl}]]\\
&\approx L\log (1-\epsilon)
\end{align}
$$

Taking an approximation that would make most mathematicians cry:

$$
\begin{align}
\mathbb E[\text{Accuracy}]&\approx \exp(\mathbb E[\log \text{Accuracy}])\\
&=(1-\epsilon)^L
\end{align}
$$

This reveals that accuracy approximately falls geometrically with target token length. The resolution-limited expected accuracy is therefore:
> 这揭示了正确率近似随着的 $\epsilon$  的增加呈几何级数下降，即 $\frac {\partial (1-\epsilon)^L} {\partial \epsilon} = L(1-\epsilon)^{L-1}$

From this we can see that choosing a nonlinear metric like Accuracy is affected significantly more by limited resolution because Accuracy forces one to distinguish quantities that decay rapidly.