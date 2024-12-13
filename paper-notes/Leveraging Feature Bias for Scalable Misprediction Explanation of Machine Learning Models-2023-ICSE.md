# Introduction

问题领域：
Interpreting and debugging machine learning models
- Interpreting --> Explaining mispredictions 解释模型的误预测
- Debugging --> Upweight sampling 上采样：提升权重

意义：
Ensure the robustness of machine learning models
robustness: stability, generalization 
文中含义是指模型对特征值离群的样本也保持良好的预测性能
<br>
作者提出的两个模型(算法流程)：
- Explaining mispredictions: BGMD(Bias Guided Misprediction Diagnoser)
- Upweighting sampling: MAPS(Mispredicted Area UPweight Sampling)

# Detail

## Bias Guided Misprediction Diagnoser(**BGMD**)

### Misprediction Diagonoser:

#### Already Have:
$M$: a model
$D$: a set of samples(dataset)

#### Aim: 
Explaining the **mispredictions** of the model 解释模型的错误预测

#### How: 
Learn a rule: Misprediction Explanation(ME) rule

ME(Misprediction Explanation) rule --> a decision list based on features to identify a subdataset in which most of the instances are mispredicted by the trained model 

对单个样本：
- Given: a model, a sample
- Use: ME rule(based on features) 判断：该样本满足该规则吗？
- Get: **whether this sample will be mispredicted by the model**
		if the sample fits the rule:yes 满足-->该样本容易被模型误预测
		else:no 不满足-->该样本不容易被模型误预测


apply the rule to the whole dataset
get: a subset of samples which fit the rule 符合该规则的所有样本

对数据集：
- Given: a model, a dataset
- Use: ME rule(based on features)
- Get: a subdataset

assesment：
- precision
	82%：82% of the instances identified by the above-mentioned rule are mispredicted by the model
- recall
	46%: the identified instances contain 46% of all mispredicted instances.

essence: to learn a rule


### Bias Guided:

- EXPLAIN(sota)
	limitations: ME rules are deduced “blindly” from **all** features 从所有的特征推断规则(最后推断出的规则不一定含有所有特征)
	issues: **scalability & time** 规模增长，时间消耗
	<br>
- Feature Imbalance
	highly-skewed feature distribution 高度**不平衡**的特征分布
	ref Fig2
	
	minority vs majority:
	高度不平衡的特征分布导致大多数样本的特征值相近，少数样本的特征值离群，以此区分出了多数和少数，minority group就是特征值离群的那一部分样本
	eg Fig2
	
	The minority group of data is usually under-represented(欠表征) during model training,and as a result,the trained model is **biased towards the majority group**, causing the model to perform poorly on data containing under-represented features 包含离群特征值的样本数量太少，使得训练出来的模型在对测试集中(或实际部署中)遇到的同样包含离群特征值的样本的预测效果不理想(misprediction)
	
	biased towards the majority group(向多数群体偏置)：
	loss中，majority group占了主导，而对于minority group，其在loss中的占比太小，因此在梯度中的占比也比较小，也就意味着模型容易根据majority group的梯度进行优化而不是根据minority group的梯度进行优化，因此最后训练出来的模型显然对于majority group的分类能力较强或回归精度更高，而对于minority group也就是包含了离群特征值(under-represented feature)的样本的分类能力或回归精度更低
	
	model's performance:
	- majority group: better
	- minority group: worse
	
	模型的偏置(bias)本质是样本数量差异导致**在损失中所占比重**的差异
	<br>
- BGMD: Use the feature imbalance to generate ME rule
	two prior knowledge about data:
	- **feature imbalance**: data often exhibit highly-skewed feature distributions
	- trained models in many cases perform **poorly** on **subdataset with under-represented features** --> poorly on minority group
	
	poor performance --> mispredictions 
	the minority group is prone to be mispredicted
	模型对于这部分样本的预测性能不好
	
	BGMD: explain mispredictions from the perspective of feature imbalance

	Procedures of BGMD
	ref Algorithm 1
	1. Construct misprediction indication vector: $I(x)$
		given a model $M$
		$I(x) = 1$ if instance $x$ is mispredicted by $M$
		$I(x) = 0$ if instance $x$ is corrected predicted by $M$
		<br>
	2. Extract biased features
		ref Algorithm 2
		
		select a subset of features that the trained model is biased on: the model $M$ performs significantly better on the feature’s majority group than its minority group
		
		1. separates all data into mispredicted and correctly predicted group
		
		2. iterate each **feature** to evaluate whether there is a significant feature distribution difference(~~Mann-Whitney test, α < 0.05~~) between the mispredicted and correctly predicted groups 检查模型是否在该特征上产生了偏置(预测正确的样本在该特征上的取值分布和预测错误的样本在该特征上的取值分布是显著不同的)
		
		biased feature满足：分布是不平衡的，模型在以该特征为基准划分出的多数样本群体和少数样本群体上的预测表现是不平衡的(向多数样本群体偏置)
		<br>
	3. Generate atomic predicates(based on the extracted biased features)
		ref TABLE 2
		- for categorical features
		- for numerical features
			partition the set into k bins
		<br>
	4. Define rule Learning objective
		optimizing precision, recall and rule size simultaneously
		objective function: a linear combination of P, R and rule size
		ref formula(7)
		- Precision 
			the primary factor that identifies mispredictions instances **density**, i.e., reducing the number of correctly predicted instances in identified instances. 
			
		- recall 
			controls the **coverage** of all mispredicted instances, i.e., increasing the number of identified mispredicted instances.
			
		- rule size 
			mainly used for **accelerating convergence** and improving the explainability of generated rules
		<br>
	5. Rule learning
		iteratively adds previously generated atomic predicate into a decision list until the learned rules achieve the desired coverage(the target recall)
		using a standard sequential covering method
		
		the learned decision list(rule): a list of predicates
		 - if ($\phi 1$) then 1 
		 - else if ($\phi 2$) then 1 
		 - else 0 
		
		学习到的规则是基于偏置特征的，用规则判断就是基于样本在偏置特征上的取值判断

## Mispredicted Area uPweight Sampling(MAPS)

### Preliminaries:

#### Empirical Risk Minimization(ERM)
ref formula(8)
minimizing the **average** training loss 

#### Synthetic Minority Oversampling TEchnique(SMOTE)

##### Aim:
balance the number of instances between majority group and minority group(**oversampling**) 过采样

##### How:
synthesizing **artificial** instances in the minority group

hyperparameters: 
- K: neighbor numbers
- N: number of instances to create

Procedures of SMOTE:
1. Randomly selects an instance in the minority group.
2. Randomly selects any of its K nearest neighbors belonging to the same class and generates a temporary new instance $X_{temp}$ using **the average of selected K neighbors** 选K近邻，取平均
3. Randomly specifies a value $\lambda$ in the range $[0, 1]$
4. Generates and places a new instance on the vector between the original and $X_{temp}$, located $\lambda$ percent of the way from the original instance
$$X_{new} = X + \lambda (X_{temp} - X)$$

#### Just Train Twice(JTT)

**upweight sampling** 加权采样

the state-of-the-art technique which has been compared with several upweight and reweight methods

two-stages(train twice):
1. First Stage Training: 
	train a model $\hat{M}$ on training data and then constructs a misprediction indication vector I on the validation data 正常训练，然后定位验证集中被模型误预测的样本
	<br>
1. Second Stage Training：
	retrain a final model $M$ with validation data by upweighting all instances in the validation data that were mispredicted by the first trained model 在验证集进一步训练模型，提高被误预测样本的权重
	ref formula(9)

intuition: the first model mispredicted, the final model should pay more attention to them

problem: increasing the model’s weight only for mispredicted data instances can make the model overfit to them. This could also result in the previously correctly predicted instances to be mispredicted by the final model 过拟合，学习到了误预测样本的个性，而不是共性，学习到样本的个性会让模型损失泛化性

### Procedures of MAPS:

ref Algorithm 3
1. Mispredicted Area identification:
	- first trains a normal ML model $\hat{M}$
	- identifies groups of instances that **tend to** be mispredicted by BGMD
	<br>
2. Upweighting:
	upweighting the **identified instances**(instances that **tend to** be mispredicted), retrains a final model $M_{final}$
	ref formula(11)
	
	why not just upweight all the mispredicted samples but train such a BGMD to derive a rule and then locate the samples which fits the rule and then upweight those samples, it is clear that the choices by the rule are not guaranteed to be totally correct
	
	- overfitting(too specific, larger variance) --> Uncompromising performance for all data 希望是修复模型的偏置，而不希望模型从向一边偏置到向另一边偏置
	- interpretability <--> generalization(a general rule) 用通用的一个规则描述保证可解释性和泛化性
	
	需要的是被误预测样本的共性，而不需学习其个性
	用ME rule描述其共性, 共性是根据样本的特征来判断的
	满足该共性的样本是模型不容易正确预测，分类，识别的(hard samples)，因此提升该部分样本的权重，让模型pay more attention to those samples


MAPS: leverages the ME information generated by BGMD to improve the ML model’s performance on instances that contains under-represented features

## Evaluation

### ME rule generation technique comparison

- quiality: 
	precision and recall --> F1 score: $F1 = 2 * \frac{P*R}{P+R}$
	ref formula(12)
	$$F1 = 2 * \frac{P*R}{P+R} = 2* \frac{1}{\frac{1}{R} + \frac{1}{P}} $$
- efficiency: time

quality results ref to Table 3

note: 
BGMD and EXPLAIN take the target recall as a parameter
the generated rules prioritize improving the recall values, which hurts the precision score.

efficiency results ref to Fig 3

### Effectiveness of Mispredicted Area Upweight Sampling

ref to Table 5, Table 6

note: 
- it is important that these methods should not only improve the model’s performance on mispredicted data, but also ensure model’s performance on all data.

- SMOTE:
	biased towards the data that has been “duplicated” many times(the previous minority data) the new trained model performs worse on the majority group that performed well before the retraining This results in SMOTE gaining in recall but lowering the precision(~~mark~~)

### Impact of Upweight Value on MAPS

The higher the upweight value, the trained ML model pays more attention to the instances that identified by ME rule generation tools
ref Fig 4

### Discussion

#### Why BGMD works better?

- Focus only on useful features:
	deduces the ME rules only on biased features
	EXPLAIN deduces rules from all 29 features, BGMD only from 8 features, while the final rule from them both contains only 8 features
	<br>
- Make more attempts
	given the same computation time --> more granular predicates on features 因为特征数量少了，粒度就更细了

#### Why MAPS is a good method to fix models?

- Competitive performance
- Uncompromising performance for all data:
	JTT used a similar upweight sampling approach as MAPS, it reduced the performance of four models on all data
	
	making retrained model **pay more attention to the under-represented features instead of focusing more on particular mispredicted instances.**
- No extra computation
	SMOTE： adding more data means more computation during model training
- Model agnostic
	MAPS entirely focuses on identified data groups that are prone to be mispredicted

# Summary

作者观察到数据常常表现出特征分布极度不平衡的情况，并且这会导致训练出来的模型在遇到特征值离群的样本时预测表现不佳，产生误预测。
作者利用了特征分布不平衡这一现象，提出了BGMD，BGMD的目的是推断出一个能确定样本是否容易被误预测的规则，在推断时，BGMD首先确定了导致模型产生偏差的特征，然后依据这些偏置特征推断规则。
作者基于BGMD提出MAPS，MAPS的目的是对模型进行修补，强化模型在遇到特征值离群的样本时的预测性能，MAPS的步骤是首先用BGMD推断出规则，用该规则确定容易被模型误预测的样本，之后对这部分样本提升权重，重新训练得到新的模型。
实验证明BGMD相较于EXPLAIN，生成规则的速度很快，且规则可以有效解释模型的误预测，有较高的精确率和召回率。
实验证明MAPS相较于SMOTE，JTT，是更有效的上采样方法，提高了模型的健壮性。
BGMD和MAPS对模型没有局限性。