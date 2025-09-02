# Introduction
- 关注的问题：机器学习公平性(Machine Learning Fairness)
	The unfairness of ML software usually indicates the discriminatory treatment of different groups divided by sensitive features , e.g., sex. According to biased decisions made by ML software, one group (e.g., male) may have a particular advantage, i.e., with more opportunity to obtain “favorable” decisions1, called privileged, over the other group (e.g., female), called unprivileged
	不公平的机器学习模型即会进行区别对待的机器学习模型，对于通过敏感特征(如：性别)进行划分的两个群体(如：男女)，模型的决策会偏向于其中的一个群体，这个群体被称为有特权的(privileged)，而另一个群体被称为无特权的(unprivileged)，因为有偏向性，模型的决策也称为有偏差的(biased decision)
	
	fairness defect
	有人提出软件的公平性缺陷这一概念，即不公平的模型也认为是有缺陷的
	
	fariness is a core non-functional quality property of ML software
	performance optimization techniques might lead to unfairness
	公平性不是一个功能性(functional)的问题
	要最优化模型的表现，往往会导致不公平(unfairness)，要让模型公平，是在损失模型的表现的基础上的(因为不得不承认的是敏感的特征有时确实会带有利于模型正确决策的信息)
	
	Reduce discrimination
	说白了就是模型不能搞歧视
	比如一个预测被录取概率的模型，在其他条件(特征)都相等的情况下，男性应聘者和女性应聘者的录取概率应该是完全一样的，比如0.8，性别是一个敏感的特征，它不应该对模型的决策有任何影响，不然模型就是在搞歧视了，因此常常在训练中会把数据集中的敏感特征全部去掉，比如性别，比如种族，以此防止这些特征对模型的决策做出贡献，以此训练出较公平的模型
- 不公平的来源(the root of unfairness)
	ML software obtains its decision logic from training data. The behavior of ML software to a large extent is determined by the quality of training data, i.e., biased training data may lead to the trained ML software with statistical discrimination.
	有研究者认为来自于训练数据，带偏差的训练数据会训练出带偏差的模型(这里的偏差指的是在公平性上的偏差)
	
	前人也探究了通过优化训练数据集来减小模型的不公平性
	- 增大特征集的大小(Enlarge the size of feature sets)
	- 移除偏差样本和平衡数据分布
		remove biased labels 
		rebalance internal distributions
- 偏差的特征(biased features)
	the root cause of the unfairness could be biased features in training data. Here, we define biased features as features that are highly related to the sensitive feature in training data
	本文认为导致模型不公平的根本原因是训练数据集中带偏差的特征，即和敏感特征有紧密相关的特征，一般在训练的时候我们都会移除敏感特征防止它对模型的决策贡献，但是偏差特征可能也包含了一部分敏感特征的信息(比如两个特征是强线性相关的)，因此即使移除了敏感特征，它也可能通过偏差特征间接地对模型地决策做出贡献
	
	比如：居处区域和种族
- 优化偏差特征(Debug biased feature values)
	To obtain unbiased features, we try to debug feature values in training data, i.e., 
	(a) identify which features and which parts of them are biased
	(b) exclude the biased parts of such features to recover as much useful and unbiased information as possible
	为了防止上述的现象，本文提出了一种方法，流程即
	1. 首先找到带偏差的特征(和敏感特征相关性较强的特征) 
	2. 去除偏差特征中的偏差部分(和敏感特征相关的部分，或者说可以通过敏感特征计算出来的部分，我们认为这个部分就是偏差特征中携带的敏感特征的信息)，得到无偏差的特征
- 基于线性回归的训练数据debug(Linear-regression based Training Data Debugging (LTDD))
	(a) adopting Wald test with t-distribution to check whether the features contain significant bias 
	用假设检验确认偏差特征
	(b) removing the biased parts of training features by subtracting the estimated values from original feature values to construct fair ML software.
	移除带偏差的部分(减去)，然后训练模型
- 实验结果
	The results of our experiment show that LTDD can largely improve the fairness of ML software, with less or comparable damage to its performance
	LTDD可以很好地提升模型的公平性，同时对于模型的性能的损害也最小

# Approach

算法流程：
1. 确定偏差特征(Identify the biased features and estimate the biased parts of them)
	训练数据集中，共$n$个样本，每个样本由一个$d$维的特征向量决定，其中，$f_d$是敏感特征，$f_1,f_2,\cdots,f_{d-1}$是其余的特征，这一步要找出和$f_d$关联性较强的特征，即偏差特征
	
	在本文中，这种关联性指线性关联性，即两个特征的取值分布是否呈线性相关
	
	具体方法为：
	对于每一个其余特征，将它和敏感特征拟合线性回归模型：
	1. 假设：$$f_{non-sensentive} = a+bf_{senstive}+\epsilon$$
	2. 损失函数：$$Loss = \frac{1}{n}\sum\limits_{i=1}^{n}(f_{n_i}-\hat a-\hat bf_{s_i})^2$$$$Loss = \frac{1}{n}\|f_{n}-\hat b-\hat af_{s}\|_2^2$$
	3. 优化损失函数，可以梯度下降，也可以直接令，解出$a,b$：$$\frac{\partial Loss}{\partial \hat a}=0,\frac{\partial Loss}{\partial \hat b} = 0$$
	(简单线性回归同样可以从极大似然估计的角度解释，见[[#线性回归的极大似然解释]])
	
	进行假设检验(详细见[[#Appendix]])
	零假设$H_0$：斜率$b=0$，备择假设$H_1$：斜率$b\ne 0$
	显著性水平$\alpha=0.05$
	如果在0.05的显著性水平下，接受零假设，说明特征之间没有线性关系
	如果拒绝了零假设，说明有95%的把握认为特征之间有线性关系，将其标记为偏差特征
2. 移除偏差
	对于每一个偏差特征，做减法：$$f_{new}=f_{biased}-\hat a-\hat bf_{sensetive}$$
	即移去其中可以被敏感特征线性拟合的部分，只留下了不能拟合的部分(残差)作为新的特征
3. 得到新的训练数据集，以此拟合模型

# Experiment
## Experiment setups
- 数据集
	![[Training Data Debugging-Table 1.png|Table 1]]
- Baseline
	Fair-Smote
	Fairway
	Reweighting
	Disparate Impact Remover(DIR)
- 实验模型和验证设置
	拟合的模型主要是二分类模型，主要采用逻辑回归，还有朴素贝叶斯，SVM，代码中就是直接调用sklearn的库
	划分整体数据集：15%测试集，85%训练集
	重复实验100次
	![[Trainging Data Debugging-Figure 2.png|Figure 2]]
- Metrics
	- Fairness Metrics
		(1)Disparate Impact(DI)
		  差别性影响/不同影响(指在没有明显歧视意图的情况下，某些政策或规定对不同群体产生不同的负面影响)
		  It indicates the ratio of the probabilities of favorable results(favorable rates) obtained by the unprivileged ($x_s=0$ ) and privileged($x_s=1$) classes
		  $$DI = \frac{p(\hat y=1|x_s=0)}{p(\hat y=1|x_{s}= 1)}$$
		  $y=1$，被判定成正类是想要的结果(favorable result)，如被录用
		  这个指标表示的是两个由敏感特征划分的群体之间，被判定成正类的概率的比值
		  如敏感特征为性别，男性被模型认为录用的概率是0.8，女性是0.4，DI值就是0.5
		  DI值越接近1越好，说明越公平
		(2)Statistical Parity Difference(SPD)
		  统计平等差异
		  it represents the difference between the unprivileged and privileged classes to obtain favorable results
		  $$SPD = {p(\hat y=1|x_s=0)}-{p(\hat y=1|x_{s}= 1)}$$
		  由商变成了差，显然，SPD越接近0越好，说明没有差异很公平，否则说明存在差异
	- Performance Metrics
		![[Confusion Matrix.jpg|Confusion Matrix]]
		(1)ACC(Accuracy)
		预测正确的样本的比例
		$$ACC = \frac{TP+TN}{Total}$$
		(2)Recall
		在所有为正类的样本中，模型找出了其中多少样本，其所占的比例
		$$Recall = \frac{TP}{TP+FN}$$
		(3)False Alarm
		在所有为反类的样本中，模型找出了其中多少样本，其所占的比例
		$$False Alarm = \frac{FP}{FP+TN}$$
		
- Analysis methods(和baseline对比的方法)
	主要是rank-sum test和Cliff's delta，详见[[#威尔科克森秩和检验/曼-惠特尼 U 检验(Wilcoxon rank-sum test/Mann-Whitney U test)]]和[[#Cliff's delta]]
	作者认为$|\delta|<0.147$时，差异可以忽略，从$0.147 - 0.330$，差异较小，从$0.330 - 0.474$，差异中等，大于$0.474$，差异较大
	作者设定Win,Tie和Lose
	Win：rank-sum test零假设在$\alpha=0.05$的显著性水平下不成立，且$\delta>0.147$，说白了就是本文提出的方法的metric值大部分都相对baseline的更高
	Lose：rank-sum test零假设$\alpha=0.05$的显著性水平下不成立，且$\delta<-0.147$，说白了就是本文提出的方法的metric值大部分都相对baseline的更低
	Tie：其余情况认为时平手

## Experimental results
- 和什么都不做的原始模型比较Fariness和Performance![[Traning Data Debugging-Figure 3.png|Figure 3]]
	the more negative the difference observed, the more our method improves the fairness of the standard classifier
	结论：Our method can greatly improve the fairness of the original ML software and slightly damage its performance
	相较于原模型，应用LTDD可以极大提升公平性，同时对模型性能影响很小
- 和其他baseline方法比较Fariness![[Training Data Debugging-Table 2.png|Table 2]]
- 和其他baseline以及原始模型比较performance![[Traning Data Debugging-Table 3.png|Table 3]]
	Generally, all methods have losses to the performance of the original classifiers
	结论：Our method performs better than baselines in the improvement of fairness indicators with the performance damage less than or comparable to baselines.
	相较于其他方法，LTDD在提升fairness的程度都更大，同时对于模型性能的影响没有更差
- 改用其他的分类器模型，验证LTDD的泛化性
	详见Table 4，Table 5
	在其他的分类器上，结论依然不变
- 验证LTDD的可行性(Actionability)
	可行性使用favorable decision rate来衡量的
	也就是被预测为正例(favorable decision)的样本占样本总数的比例
	在实际中，正例(favorable decision)往往是占据资源的决策，比如一个决定是否批准给申请人贷款的模型，如果模型的判断结果是批准，就说明需要给申请人资源(贷款)
	
	作者希望，相较于原模型，使用过fairness method之后，模型在改善公平性的同时，不要大幅度提高正例的比例(favorable decision rate)，也就是说，希望方法是通过削减privileged group的正例数量，增加unprivileged group的正例数量来实现公平，而不是通过都提高了两个group的正例数量，只是一个提高的多，一个提高的少来实现公平，这显然会导致favorable decision rate上升，容易使用更多资源
	
	实验发现LTDD确实是如所期望的那么做的
	the way that LTDD improves fairness is to increase the favorable rate of non-privileged classes and to reduce the favorable rate of privileged classes
	the whole favorable rate is very close to the original value
	our method does not need more social resources to achieve in most cases
	![[Traning Data Debugging-Figure 4.png|Figure 4]]

# Discussion
- 关于biased feature所占的比例![[Training Data Debugging-Figure 5.png|Figure 5]]
- 关于特征修正后分布的变化![[Training Data Debugging-Figure 6.png|Figure 6]]

# Conclusion
本文关注的是机器学习模型的公平性问题
作者认为导致模型不公平的根本原因是训练集中偏差特征(biased feature)的存在
为了修正特征的偏差，作者提出LTDD(Linear-Regression based Training Data Debugging)，利用线性回归来判别偏差特征的存在，并修正偏差 
实验结果表明LTDD可以有效提高模型的公平性，同时对模型效果的影响较小

# Appendix

## 假设检验
对总体的参数提出一个假设值，然后利用样本信息推断这一假设是否成立
假设检验基于一个思想：小概率事件在一次抽样实验中是几乎不会发生的
在假设检验中，我们会提出两个假设：原假设$H_0$，和备择假设$H_1$，一般拒绝了$H_0$就意味着接受$H_1$
我们会假定原假设是成立的，然后在原假设成立的前提下，提出一个统计量，将事件(本次的抽样)与某个统计量的取值联系，这个统计量一般服从某种先验分布(正态总体的抽样分布)，因此这个统计量的取值落在了任意一个区间上的概率一般都是可以计算的
在原假设成立的前提下，我们可以计算该统计量的值，如果这个值落在了小概率区间(拒绝域)，说明这个统计量取这个值是一个小概率事件，也就是说，在原假设成立的前提下，本次的抽样的发生是一个小概率事件，也就是说这一般是不可能的，那么我们就认为原假设成立的前提是错误的，即原假设不成立，拒绝原假设，接受备择假设，否则，我们没有足够的理由拒绝原假设，就接受原假设

## 常用统计量的分布
1. 标准正态分布：$x \sim N(0,1)$
2. 卡方分布
	对于$n$个相互独立的，且均服从于标准正态分布的随机变量：$X_1,X_1,\cdots,X_{n},\  X_{i}\sim N(0,1)$
	称随机变量$$\chi^2=X_{1}^2+X_{2}^2+\cdots+X_{n}^2$$
	服从自由度为$n$的$\chi^2$分布
	即$\chi^{2}\sim \chi^2(2)$
3. $t$分布
	有随机变量$X\sim N(0,1)\quad Y\sim \chi^2(n)$，$X$和$Y$相互独立
	称随机变量：$$t = \frac{X}{\sqrt {Y/n}}$$
	服从自由度为$n$的$t$分布
	即$t \sim t(n)$
	当$n$充分大的时候，$t$分布近似于标准正态分布

## 正态总体的抽样分布
有总体$X\sim N(0,1)$，$X_1,X_2,\cdots,X_n$为总体$X$的简单随机样本(通过简单随机抽样的得到的样本)
样本均值：$$\bar{X} = \frac{1}{n}\sum_{i=1}^{n}X_i$$
样本方差：$$S^{2}= \frac{1}{n-1}\sum\limits_{i=1}^n(X_{i}-\bar{X})^2$$
有：
1. $$\frac{\bar{X}-\mu}{\sigma \sqrt{n}}\sim N(0,1)$$
2. $$\frac{n-1}{\sigma^{2}}S^{2}\sim \chi^2(n-1)$$
3. $$\frac{\bar{X}-\mu}{S/\sqrt{n}}\sim t(n-1)$$
## 上$\alpha$分位点
设随机变量$Z\sim N(0,1)$，对于$\alpha \in (0,1)$，实数$z_{\alpha}$满足：$$P\{Z > z_{\alpha}\} = \alpha$$
称$z_{\alpha}$为标准正态分布的上$\alpha$分位点

## 线性回归的极大似然解释
极大似然法(Maximum Likelihood Estimate)是用于对分布进行参数估计的朴素方法，对于线性回归的极大似然解释由高斯提出，主要的思想就是将$y$的生成过程看作是从一个概率分布进行采样的过程，然后利用极大似然法对该分布进行参数估计
我们假设(假设1）：$$y_i = a+bx_i+\epsilon$$
同时假设(假设2)：$$\epsilon \sim N(0,\sigma^2)$$
以及
假设3：$X$的分布任意
假设4：每次观测得到的$\epsilon$都是相互独立的，或者说$y_i$与$y_j$的生成过程是相互独立的
假设2即对残差(或者说噪声)的分布进行了假设，认为它从属于正态分布
那么由于正态分布的性质，容易得到：$$y_i\sim N(a+bx_i,\sigma^2)$$
也就是说，$y$从属于一个均值为$a+bx$，方差为$1$的正态分布，$y$的生成可以视为从这个分布中进行采样的形式
(事实上对于每一个$y_i$，很显然其分布的均值$a+bx_i$都不一样，因为我们同时假设了$y_i$之间是相互独立的，其实可以认为我们模拟了$n$个均值不同的正态分布，或者说$1$个多元正态分布，即直接从中采样出向量的正态分布，之后的推导将其视为一个多元正态分布)
对于这个分布，我们要估计它的参数$a$和$b$
利用极大似然法(最大化观测到的事件的概率)：$$Pr(\{y\}_{i=1}^{n})=\prod_{i=1}^n\frac{1}{\sigma\sqrt{2\pi}}exp\{-\frac{1}{2\sigma^2}(y_{i}-a-bx_i)^2\}$$
取对数：$$lnPr(\{y\}_{i=1}^{n})=\sum\limits_{i=1}^nln(\frac{1}{\sigma\sqrt{2\pi}}exp\{-\frac{1}{2\sigma^2}(y_{i}-a-bx_i)^2\})$$
要最大化上式,同样求导，令导数为零即可，可以得到可以使得似然最大化的参数$a,b,\sigma$
计算可以证明极大似然解和最小二乘解是完全一致的：$$\hat b = \frac{\sum\limits_{i=1}^n(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sum\limits_{i=1}^{n}(x_{i}-\bar{x})^2}=\frac{cov(X,Y)}{s_X^2}$$
$$\hat a = \bar y - \hat b \bar{x}$$
$$\hat {\sigma}^{2}= \frac{1}{n}\sum\limits_{i = 1}^n(y_{i}-(\hat a + \hat bx_{i}))^2$$
事实上这不是偶然，因为高斯在推导时考虑的问题就是：当$\epsilon$从属于什么分布时，可以从极大似然法得到最小二乘解
极大似然法给线性回归一种新的解释，现在$y_i$被视为一个随即变量，我们可以以此计算置信区间，进行假设检验等

### 线性回归的参数分布与推断
#### 参数的分布
线性回归的最小二乘解是：
$$\hat b = \frac{\sum\limits_{i=1}^n(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sum\limits_{i=1}^{n}(x_{i}-\bar{x})^2}=\frac{cov(X,Y)}{s_X^2}$$
$$\hat a = \bar y - \hat b \bar{x}$$
$$\hat {\sigma}^{2}= \frac{1}{n}\sum\limits_{i = 1}^n(y_{i}-(\hat a + \hat bx_{i}))^2$$
可以对等式进行重写，具体推导略过，思路是将$y_{i} = a+bx_{i}+\epsilon_i$代入，得到：
$$\hat b = b+\sum\limits_{i=1}^{n}\frac{x_{i}-\bar{x}}{ns_X^2}\epsilon_i$$
$$\hat a = a+\frac{1}{n}\sum\limits_{i=1}^n(1-\bar{x}\frac{x_i-\bar{x}}{s_X^2})\epsilon_i$$
也就是说，我们的最小二乘解可以写成一个确定的常数(即其期望)加上一个噪声(或者说残差)$\epsilon_i$的加权平均
而：$$\epsilon_{i}\sim N(0,\sigma^2)$$
可以认为$\epsilon_i$是采样得到的
注意我们不可能利用重写后的式子解出$\hat b,\hat a$，式子重写是将$y_{i} = a+bx_{i}+\epsilon_i$代入得到的，而其中的$a,b,\epsilon_i$都是未知的，重写后的式子是提供了我们另一种视角来看待我们得到的最小二乘解
由于正态分布的性质，计算可得：
$$\hat b \sim N(b,\frac{\sigma^2}{ns_X^2})$$
$$\hat a \sim N(a,\frac{\sigma^2}{n}(1+\frac{\bar{x}^2}{s_X^2}))$$
$$\frac{n\hat{\sigma}^2}{\sigma^{2}}\sim\chi_{n-2}^2$$
上述式子说明我们的最小二乘解或者说极大似然解也可以认为是采样得到的，注意到其实所谓的分布采样形式，或者说其中的不确定性都是从我们最开始的假设，也就是说$\epsilon \sim N(0,\sigma^2)$中推导得来的，而$y = a+bx+\epsilon$也是我们的假设，也就是说，在我们的假设下，我们有确定的$a,b$，不确定的(由采样得到的)$\epsilon$，从而由$x$“生成”了$y$，那么在$x$不变的情况下，对$\epsilon$进行重复多次的采样，可以得到不同的$y$(我们现在得到的$y$其实也就认为是进行一次采样观察到的结果)，然后进行极大似然参数估计，可以得到不同的$\hat a,\hat b$，这个过程可以视为对$\hat a,\hat b$进行采样的过程，服从于上述推导出的分布

#### 参数的标准差
有了参数的分布形式，我们可以知道它们的标准差(standard error/standar deviation)
$$se[\hat a] = \frac{\sigma}{s_{X}\sqrt{n}}$$
$$se[\hat b]=\frac{\sigma}{\sqrt{n}s_X}\sqrt{s_X^2+\bar{x}^2}$$
其中的$\sigma$我们换成对它的极大似然估计：
$$\hat{se}[\hat{a}]=\frac{\hat\sigma}{s_X\sqrt{n}}$$
$$\hat se[\hat b]=\frac{\hat\sigma}{\sqrt{n}s_X}\sqrt{s_X^2+\bar{x}^2}$$
可以计算：
$$\mathbb E[\hat{\sigma}^2]=\frac{n-2}{n}\sigma^2$$
也就是说：
$$\frac{n-2}{n}\hat\sigma^2$$才是对$\sigma^2$的无偏估计
因此我们应该代入的是无偏估计值，即：
$$\hat{se}[\hat{a}]=\frac{\hat\sigma}{s_X\sqrt{n-2}}$$
$$\hat se[\hat b]=\frac{\hat\sigma}{\sqrt{n-2}s_X}\sqrt{s_X^2+\bar{x}^2}$$
这是对我们估计的参数的标准差的无偏估计

#### 基于参数分布进行推断
我们主要关心斜率$b$的推断
我们已经知道：
$$\hat b \sim N(b,\frac{\sigma^2}{ns_X^2})$$
容易推出：
$$\frac{\hat b-b}{se[\hat b]}\sim N(0,1)$$
而标准正态分布的分布函数值可以通过查表得到，其中：
$$\Phi(1.96)=0.975,\Phi(-1.96)=0.025$$
因而：
$$\begin{aligned}&P(-1.96\leqslant\frac{\hat b-b}{se[\hat b]}\leqslant1.96)\\=&P(-z_{0.025}\leqslant\frac{\hat b-b}{se[\hat b]}\leqslant z_{0.025})\\=&P(b-1.96se[\hat b]\leqslant\hat b\leqslant b+1.96se[\hat b])\\=&P(\hat b-1.96se[\hat b]\leqslant b\leqslant \hat b+1.96se[\hat b])\\=&\Phi(1.96)-\Phi(-1.96)\\=&0.95\end{aligned}$$
得到的是$b$的$0.95$的置信区间
也就是说，进行$100$次抽样，计算$100$次这个区间(或者说把计算$\hat b$的整个过程直接看成一次抽样)，平均情况下，有$95$个区间会包含真实值$b$，也可以认为用本次采样计算出的这个区间来估计参数$b$的把握是$95\%$(有$95\%$的把握认为本次估计的区间包含真实值)

但是实际中，就如上一节提到，要得到真实的$se[\hat b]$，我们需要知道真实的$\sigma^2$，而我们往往只知道它的极大似然估计值，也就是$\hat \sigma^2$，因此我们往往知道的只是$\hat se[\hat b]$
因此我们需要知道的是$\frac{\hat b-b}{\hat se[\hat b]}$从属于什么分布
略去推导过程，直接给出：
$$\frac{\hat b-b}{\hat se[\hat b]}\sim t_{n-2}$$
$t_{n-2}$即自由度为$n-2$的$t$分布
事实上$a$也有相似的结论：
$$\frac{\hat a-a}{\hat se[\hat a]}\sim t_{n-2}$$

#### 基于参数推断进行假设检验
- $p$值($p-value$)
	简单理解，就是在原假设(零假设)正确的前提下，观测到样本数据的可能性，或者说，在原假设(零假设)正确的前提下，观测到原样本数据这一事件发生的概率
	
	一般我们计算出$p$值后，会将它与一个预设的概率阈值(显著性水平$\alpha$)进行比较，比如说$0.05$
	若$p>\alpha$，说明在$\alpha$的显著性水平下，不认为该事件是小概率事件，也就是说它是有概率发生的，因此应接受原假设，且有$p$的把握
	反之，则说明在$\alpha$这个显著性水平下，该事件是小概率事件，认为它在一次实验不会发生，应当拒绝原假设，但有$p$的可能性犯错误
- 对$b$进行$Wald\ \  t-test$(沃尔德$t$检验)
	1. 零假设$H_0$：斜率$b=0$，备择假设$H_1$：斜率$b\ne 0$，显著性水平$\alpha=0.05$
	2. 当零假设成立，即$b=0$，我们有：
		$$\frac{\hat b}{\hat se[\hat b]}\sim t_{n-2}$$
		事实上，当$n\geqslant45$时，可以将$t$分布近似为标准正态分布，方便计算：
		$$\frac{\hat b}{\hat se[\hat b]}\sim N(0,1)$$
		
	3. 我们已经知道，当上述分布成立时，我们有大概率事件(概率为0.95)：
		$$-1.96se[\hat b]\leqslant\hat b\leqslant 1.96se[\hat b]$$
		也可以写成：
		$$-1.96\leqslant\frac{\hat b-b}{se[\hat b]}\leqslant1.96$$
		即：
		$$|\frac{\hat b}{se[\hat b]}|\leqslant 1.96 = z_{0.025} = z_{\alpha/2}$$
		相应地有小概率事件(概率为0.05)：
		$$|\frac{\hat b}{se[\hat b]}|\gt 1.96 = z_{0.025} = z_{\alpha/2}$$
	4. 直接将值代入，检查是大概率事件成立还是小概率事件成立，即检查值落在了哪个区间里
	5. 如果值落在了拒绝域，即在原假设成立的前提下，小概率事件发生了，说明应该拒绝原假设，我们有0.95的把握是正确的，但有0.05的把握犯错，因为小概率事件的确有0.05的概率会发生
	6. 如果值没有落在拒绝域，说明在原假设成立的前提下，大概率事件发生了，在意料之中，我们没有足够的证据说明原假设是错误的，因此接受原假设
梳理一下，我们从最原始的观察的原数据开始，对其进行了线性回归，计算出了估计的斜率值，和它的标准差，然后将其和$t$分布联系到了一起，也就是说我们将观察到原始数据这一事件和一个概率分布联系到了一起，然后我们假设数据中不存在线性关联，在有这个假设的前提下，我们检验了我们观察到原始数据的概率，如果这是一个小概率事件(在假设原始数据不存在线性关联的前提下观测到了很有可能有线性关联的数据当然是个小概率事件)，说明假设错误，我们认为数据中存在线性关联，利用这种形式的假设检验，在我们断定两个特征是存在线性关联(拒绝原假设$b=0$)时，我们总可以说，我们在统计上有0.95的显著性是正确的

## 威尔科克森秩和检验/曼-惠特尼 U 检验(Wilcoxon rank-sum test/Mann-Whitney U test)

威尔克森秩和检验是检验两个总体的分布是否相同的的非参数统计显著性检验方法
非参数也就是说它没有对总体做出任何的假设，如正态假设，就如我们在进行Wald t-test时的那样
同时，非参数也说明我们进行的假设检验不是对一个参数的具体取值进行检验

在本文中，作者利用rank-sum test来检验他们提出的方法和其余baseline方法得到的度量值总体的分布之间的差异是否统计上显著，也就是说，他们的方法和其余的baseline方法得到的度量值总体的分布一样吗

依照原文的意思，作者应该是在每个数据集上都进行了100次的训练和测试，得到了100个度量值(metric)的值，也就是得到了一个度量值的总体，然后检验利用不同方法得到的总体，它们服从的分布是否相同

总的来说，rank-sum test的流程如下：
1. 我们有两个总体，第一个总体有$n_1$个样本观测值，第二个总体有$n_2$个样本观测值
2. 提出零假设$H_0$：两个总体服从的分布相同，备择假设$H_1$：两个总体服从的分布不同(可以更具体地说：二者的分布不同，分布1的值总体上比分布2的值更高(systematically larger)，或者反之)
	(Null hypothesis:  The two groups are sampled from populations with identical distributions.  Typically, that the sampled populations exhibit stochastic equality.
	Alternative hypothesis (two-sided): The two groups are sampled from populations with different distributions.  Typically, that one sampled population exhibits stochastic dominance.)
1. 将$N=n_1+n_2$个样本观测值放一起排序，由低到高标上序号/排位/秩(rank)，即值最小的样本的rank就是1，值最大的样本的rank就是N
2. 计算一个统计量：属于第一个总体的所有样本的秩的和，即：$$W=\sum\limits_{样本i属于第一个总体}rank_i$$
	比如属于第一个总体有3个样本，它们的秩分别为1，3，5，那么W = 1+3+5 = 9
	知道了W，也就知道了属于第二个总体的所有样本的秩的和，即：$$W' = \frac{N(N+1)}{2}-W$$
3. 这里不加证明地给出，在满足零假设的条件下，统计量W的期望值是：$$u_W=\frac{n_1(N+1)}{2}$$
	标准差是：$$\sigma_W=\sqrt{\frac{n_1n_2(N+1)}{12}}$$
	显然，如果统计量W离得$u_W$太远，就说明在$H_0$下，小概率事件发生了，应该拒绝$H_0$，这个“远”就是通过二者的距离是几倍的标准差来度量的
4. 但是如果不知道W具体的分布形式，一般也难以通过给定的显著性水平划分出拒绝域，事实上，统计量$$z = \frac{W-u_W}{\sigma_W}$$在零假设成立的前提下，所从属的分布是渐进正态的，也就是说随着$n_1,n_2$的增大，$z$的分布会逐渐趋于正态分布，我们可以由此构造拒绝域：$$P(|z|\geqslant1.96)=0.05$$
	z的值落在拒绝域中，也就是说W的取值过于离谱，小概率事件发生了，应该接受备择假设：二者的分布不同，如果$z$是正数，也就是$z\geqslant1.96$，说明总体1的rank值总体都比较高，也就是总体1的样本观测值总体都比较高，即分布1是systematically larger than分布2的

#### Cliff's delta
Cliff's delta统计量可以认为是一个对rank-sum test的补充，通过rank-sum test我们可以知道一个总体的分布在取值上一般整体上是要大于另一个总体的分布，表现为rank更高，而这个大的程度到底是多少(effect size)，可以用Cliff's delta来衡量
Statistics of effect size for the Mann–Whitney test report the degree to which one group has data with higher ranks than the other group.  They are related to the probability that a value from one group will be greater than a value from the other group

Cliff's delta的计算公式是：
$$Cliff's\ \ \delta= P(x_i>y_j)-P(x_i<y_j)$$
这个公式的意思是：$x_i$属于分布1，$y_j$属于分布2，从两个分布各自抽一个样本，计算$P(x_i>y_j)$和$P(x_i<y_j)$这两个事件的概率的差值
实际计算中使用频率代替概率，比如，总体1有3个样本，值是2，4，6，总体2有3个样本，值是1，3，5，那么两边两两配对，有9个样本对：$(2,1),(2,3),(2,5),(4,1),(4,3),(4,5),(6,1),(6,3),(6,5)$，其中，$x_i>y_j$的占了6个，$x_i<y_j$的占了3个，则$$Cliff's\ \ \delta=\frac{6-3}{9}=\frac{1}{3}$$
Cliff's delta的取值范围是-1到1，取到1时说明总体1的所有样本值都大于总体2，反之亦然
Cliff’s _delta_  ranges from –1 to 1, with 0 indicating stochastic equality of the two groups. 1 indicates that one group shows complete stochastic dominance over the other group, and a value of –1 indicates the complete stochastic domination of the other group

所以我们可以用Cliff's delta来衡量两个分布的差异程度，毕竟我们可以通过rank-sum test知道某一个分布systematically larger than另一个分布，但不知道larger的程度如何，Cliff's delta给了我们一种衡量手段，一般认为：
delta的取值(绝对值)在$0.11 - 0.28$差异是small，在$0.28 - 0.43$差异是medium，大于$0.43$是large

# References
- 统计学部分知识参考：
	《概率论与数理统计》大连理工大学出版社
- 线性回归部分知识参考：
	CMU课程：[Model Regression](https://www.stat.cmu.edu/~cshalizi/mreg/15/)部分讲义
	- Lecture 1: [Introduction to the course](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/01/lecture-01.pdf)
	- Lecture 4: [Simple linear regression models](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/04/lecture-04.pdf)
	- Lecture 6: [Estimating simple linear regression II](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf)
	- Lecture 8: [Inference in simple linear regression I](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/08/lecture-08.pdf)
	Princeton讲义: [Linear Regression via Maximization of the Likelihood](https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/mle-regression.pdf)
- 秩和检验部分参考：
	University of Florida课本第十四章: [Nonparametic Tests](https://users.stat.ufl.edu/~winner/sta3024/chapter14.pdf)
- Cliff's delta部分参考：
	网页: [Two-sample Mann–Whitney U Test](https://rcompanion.org/handbook/F_04.html)
	网页: [Cliff’s Delta](https://real-statistics.com/non-parametric-tests/mann-whitney-test/cliffs-delta/)