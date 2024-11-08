# 1 Introduction
## 1.1 Motivation
Most tasks require a person or an automated system to reason: to take the available information and reach conclusions, both about what might be true in the world and about how to act. For example, a doctor needs to take information about a patient — his symptoms, test results, personal characteristics (gender, weight) — and reach conclusions about what diseases he may have and what course of treatment to undertake. A mobile robot needs to synthesize data from its sonars, cameras, and other sensors to conclude where in the environment it is and how to move so as to reach its goal without hitting anything. A speech-recognition system needs to take a noisy acoustic signal and infer the words spoken that gave rise to it.
> 大多数任务需要推理：使用可得的信息推出结论

In this book, we describe a general framework that can be used to allow a computer system to answer questions of this type. In principle, one could write a special-purpose computer program for every domain one encounters and every type of question that one may wish to answer. The resulting system, although possibly quite successful at its particular task, is often very brittle: If our application changes, significant changes may be required to the program. Moreover, this general approach is quite limiting, in that it is hard to extract lessons from one successful solution and apply it to one which is very different.
> 本书描述让计算机系统回答这类问题的通用框架

We focus on a diferent approach, based on the concept of a declarative representation. In this approach, we construct, within the computer, a model of the system about which we would model like to reason. This model encodes our knowledge of how the system works in a computer-readable form. This representation can be manipulated by various algorithms that can answer questions based on the model. For example, a model for medical diagnosis might represent our knowledge about different diseases and how they relate to a variety of symptoms and test results. A reasoning algorithm can take this model, as well as observations relating to a particular patient, and answer questions relating to the patient’s diagnosis. 
> 我们聚焦基于 declarative representation 的方法，在计算机内构建关于需要推导的系统的模型，该模型将我们的知识编码为计算机可理解的表示
> 我们用算法运用这些表示来解决问题

**The key property of a declarative representation is the separation of knowledge and reasoning. The representation has its own clear semantics, separate from the algorithms that one can apply to it. Thus, we can develop a general suite of algorithms that apply any model within a broad class, whether in the domain of medical diagnosis or speech recognition. Conversely, we can improve our model for a specific application domain without having to modify our reasoning algorithms constantly.**
> declarative representation 的关键性质是知识和推理的分离
> 表示本身有自己的语义，和运用在它之上的算法分离
> 这使得我们可以开发更通用的算法，同时，可以在不常修改推理算法的情况下提高模型对于特定应用领域的能力 (提高表示)
 
Declarative representations, or model-based methods, are a fundamental component in many fields, and models come in many flavors. Our focus in this book is on models for complex systems that involve a significant amount of uncertainty. Uncertainty appears to be an inescapable aspect of most real-world applications. It is a consequence of several factors. We are often uncertain about the true state of the system because our observations about it are partial: only some aspects of the world are observed; for example, the patient’s true disease is often not directly observable, and his future prognosis is never observed. Our observations are also noisy — even those aspects that are observed are often observed with some error. The true state of the world is rarely determined with certainty by our limited observations, as most relationships are simply not deterministic, at least relative to our ability to model them. For example, there are few (if any) diseases where we have a clear, universally true relationship between the disease and its symptoms, and even fewer such relationships between the disease and its prognosis. Indeed, while it is not clear whether the universe (quantum mechanics aside) is deterministic when modeled at a suffciently fine level of granularity, it is quite clear that it is not deterministic relative to our current understanding of it. To summarize, uncertainty arises because of limitations in our ability to observe the world, limitations in our ability to model it, and possibly even because of innate nondeterminism.
> 我们聚焦包含大量 uncertainty 的系统
> uncertainty 来自于：我们的观察是片面 partial 的，我们的观察是存在错误 noisy 的，也就是来自于我们观察/建模世界的能力是有限的
> 或许世界内在也是不确定的

Because of this ubiquitous and fundamental uncertainty about the true state of world, we need to allow our reasoning system to consider different possibilities. One approach is simply to consider any state of the world that is possible. Unfortunately, it is only rarely the case that we can completely eliminate a state as being impossible given our observations. In our medical diagnosis example, there is usually a huge number of diseases that are possible given a particular set of observations. Most of them, however, are highly unlikely. If we simply list all of the possibilities, our answers will often be vacuous of meaningful content (e.g., “the patient can have any of the following 573 diseases”). **Thus, to obtain meaningful conclusions, we need to reason not just about what is possible, but also about what is probable.**
> uncertainty 的存在使得我们需要允许推理系统考虑概率
> 但为了得到更有意义的结果，在考虑可能性 possible 的同时，也需要考虑更可能性 probable

The calculus of *probability* theory (see section 2.1) provides us with a formal framework for considering multiple possible outcomes and their likelihood. It defines a set of mutually exclusive and exhaustive possibilities, and associates each of them with a probability — a number between 0 and 1, so that the total probability of all possibilities is 1. This framework allows us to consider options that are unlikely, yet not impossible, without reducing our conclusions to content-free lists of every possibility.
> 理论：概率微积分理论

**Furthermore, one finds that probabilistic models are very liberating. Where in a more rigid formalism we might find it necessary to enumerate every possibility, here we can often sweep a multitude of annoying exceptions and special cases under the “probabilistic rug,” by introducing outcomes that roughly correspond to “something unusual happens.”**
> 概率模型更具自由性，因为不需列举所有可能时间，通过引入概率模型，可以将许多特殊案例和异常情况归入“概率毯”下，即认为它们是“不寻常发生的事件”

In fact, as we discussed, this type of approximation is often inevitable, as we can only rarely (if ever) provide a deterministic specification of the behavior of a complex system. Probabilistic models allow us to make this fact explicit, and therefore often provide a model which is more faithful to reality.
> 我们极少可以确定规范复杂系统的行为，概率模型让我们表达出不确定性，因此更贴近现实
## 1.2 Structured Probabilistic Models
This book describes a general-purpose framework for constructing and using probabilistic models of complex systems. We begin by providing some intuition for the principles underlying this framework, and for the models it encompasses. This section requires some knowledge obasic concepts in probability theory; a reader unfamiliar with these concepts might wish to read section 2.1 first.
> 本书描述为复杂系统构建和使用概率模型的通用框架

Complex systems are characterized by the presence of multiple interrelated aspects, many of which relate to the reasoning task. For example, in our medical diagnosis application, there are multiple possible diseases that the patient might have, dozens or hundreds of symptoms and diagnostic tests, personal characteristics that often form predisposing factors for disease, and many more matters to consider. These domains can be characterized in terms of a set of random variable, where the value of each variable defines an important property of the world. For example, a particular disease, such as Flu, may be one variable in our domain, which takes on two values, for example, present or absent; a symptom, such as Fever, may be a variable in our domain, one that perhaps takes on continuous values. The set of possible variables and their values is an important design decision, and it depends strongly on the questions we may wish to answer about the domain.
> 领域中存在变量，可能变量的集合以及它们的值属于设计决策，它取决于我们对于领域需要回答的问题

Our task is to reason probabilistically about the values of one or more of the variables, possibly given observations about some others. In order to do so using principled probabilistic joint probability reasoning, we need to construct a joint distribution over the space of possible assignments to distribution some set of random variables $\mathcal X$. This type of model allows us to answer a broad range of interesting queries. For example, we can make the observation that a variable $X_i$ takes on the posterior specific value $x_i$, and ask, in the resulting posterior distribution, what the probability distribution distribution is over values of another variable $X_j$.
> 我们的任务是概率上分析这些变量的值 (可能给定关于其他变量的观测)
> 为此需要在构建随机变量集合的取值集合上构建联合分布
### 1.2.1 Probabilistic Graphical Models
Specifying a joint distribution over 64 possible values, as in example 1.1, already seems fairly daunting. When we consider the fact that a typical medical- diagnosis problem has dozens or even hundreds of relevant attributes, the problem appears completely intractable. This book describes the framework of probabilistic graphical models, which provides a mechanism for exploiting structure in complex distributions to describe them compactly, and in a way that allows them to be constructed and utilized effectively.
> 概率图模型框架提供了利用复杂分布中的结构来描述联合分布的方法

Probabilistic graphical models use a graph-based representation as the basis for compactly encoding a complex distribution over a high-dimensional space. In this graphical representation, illustrated in figure 1.1, the nodes (or ovals) correspond to the variables in our domain, and the edges correspond to direct probabilistic interactions between them. 
> 概率图中，节点对应于域中的变量，边对应于节点/变量之间的直接概率关系

![[Probabilistic Graph Models-Fig1.1.png]]
For example, figure 1.1a (top) illustrates one possible graph structure for our flu example. In this graph, we see that there is no direct interaction between Muscle Pain and Season, but both interact directly with Flu.

There is a dual perspective that one can use to interpret the structure of this graph. 
> 概率图的解释从两个方向出发

From one perspective, the graph is a compact representation of a set of independencies that hold in the distribution; these properties take the form $X$ is independent of $Y$ given $Z$, denoted $(X\perp Y | Z)$, for some subsets of variables $X,Y,Z$. 
> 其一：概率图给出了联合分布中的独立情况
> 其形式为 $X$ 在给定 $Z$ 的情况下独立于 $Y$，即 $(X\perp Y | Z)$

For example, our “target” distribution $P$ for the preceding example — the distribution encoding our beliefs about this particular situation — may satisfy the conditional independence $(Congestion \perp Season | Flu; Hayfever)$. This statement asserts that

$$
P(Congestion \mid Flu,Hayfever,Season) = P(Congestion\mid Flu,Hayfever)
$$
that is, if we are interested in the distribution over the patient having congestion, and we know whether he has the flu and whether he has hayfever, the season is no longer informative. Note that this assertion does not imply that Season is independent of Congestion; only that all of the information we may obtain from the season on the chances of having congestion we already obtain by knowing whether the patient has the flu and has hayfever. Figure 1.1a (middle) shows the set of independence assumptions associated with the graph in figure 1.1a (top). 
> 在条件 $Flu;Hayfever$ 下，$Congestion$ 和 $Season$ 独立
> 即在知道了 $Flu;Hayfever$ 的情况下，$Season$ 不会给 $Congestion$ 提供更多信息


The other perspective is that the graph defines a skeleton for compactly representing a highdimensional distribution: Rather than encode the probability of every possible assignment to all of the variables in our domain, we can “break up” the distribution into smaller factors, each over a much smaller space of possibilities. We can then define the overall joint distribution as a product of these factors.
> 其二：概率图定义了紧凑表示高维分布的框架
> 我们不是为域内的所有变量编码所有可能值的概率，而是将分布分解为小因子，每个因子都覆盖一个更小的概率空间
> 完整的联合分布定义为这些因子的积

 For example, figure 1.1(a-bottom) shows the factorization of the distribution associated with the graph in figure 1.1 (top). It asserts, for example, that the probability of the event “spring, no flu, hayfever, sinus congestion, muscle pain” can be obtained by multiplying five numbers: $P(Season = spring)$, $P(Flu = false \mid Seacon=sping)$, $P(hayfever=true\mid Seacon=spring)$, $P(congestion=true\mid hayfever=true,Flu=false)$, $P(Muscle Pain=true | Flu=False)$ This parameterization is significantly more compact, requiring only $3+4+4+4+2 = 17$ nonredundant parameters, as opposed to 63 nonredundant parameters for the original joint distribution (the 64th parameter is fully determined by the others, as the sum over all entries in the joint distribution must sum to 1). 
 The graph structure defines the factorization of a distribution $P$ associated with it — the set of factors and the variables that they encompass.

**It turns out that these two perspectives — the graph as a representation of a set of independencies, and the graph as a skeleton for factorizing a distribution — are, in a deep sense, equivalent. The independence properties of the distribution are precisely what allow it to be represented compactly in a factorized form. Conversely, a particular factorization of the distribution guarantees that certain independencies hold.**
> 这两个方面本质是等价的，分布的独立性使得完整分布可以用分解的形式被紧凑地表示，并且完整分布地特定分解也保证了一定的独立性是保持的

We describe two families of graphical representations of distributions. One, called Bayesian networks, uses a directed graph (where the edges have a source and a target), as shown in Markov network figure 1.1a (top). The second, called Markov networks, uses an undirected graph, as illustrated in figure 1.1b (top). It too can be viewed as defining a set of independence assertions (figure 1.1b [middle] or as encoding a compact factorization of the distribution (figure 1.1b [bottom]). Both representations provide the duality of independencies and factorization, but they differ in the set of independencies they can encode and in the factorization of the distribution that they induce.
> 我们将描述两类分布图表示
> 其一是贝叶斯网络，使用有向图
> 其二是 Markov 网络，使用无向图
> 两种表示都反映了独立性和可分解性之间的对偶关系
### 1.2.2 Representation, Inference, Learning
The graphical language exploits structure that appears present in many distributions that we want to encode in practice: the property that variables tend to interact directly only with very few others. Distributions that exhibit this type of structure can generally be encoded naturally and compactly using a graphical model.
> 图语言实质上描述了一个很常见的性质：变量通常会与其他一部分变量交互
> 具有这一性质的分布都可以用图表示

This framework has many advantages. 
First, it often allows the distribution to be written down tractably, even in cases where the explicit representation of the joint distribution is astronomically large. Importantly, the type of representation provided by this framework is *transparent*, in that a human expert can understand and evaluate its semantics and properties. This property is important for constructing models that provide an accurate reflection of our understanding of a domain. Models that are opaque can easily give rise to unexplained, and even undesirable, answers.
> 图模型表示清晰，且可以以可解的方式表示极巨大的联合分布

Second, as we show, the same structure often also allows the distribution to be used effectively for *inference* — answering queries using the distribution as our model of the world. In particular, we provide algorithms for computing the posterior probability of some variables given evidence on others. For example, we might observe that it is spring and the patient has muscle pain, and we wish to know how likely he is to have the flu, a query that can formally be written as $P(Flu = true \mid Season = spring; MusclePatin=true)$
. These inference algorithms work directly on the graph structure and are generally orders of magnitude faster than manipulating the joint distribution explicitly.
> 图模型便于推理

Third, this framework facilitates the effective construction of these models, whether by a human expert or automatically, by learning from data a model that provides a good approximation to our past experience. For example, we may have a set of patient records from a doctor’s office and wish to learn a probabilistic model encoding a distribution consistent with our aggregate data-driven experience. Probabilistic graphical models support a data-driven approach to model construction approach that is very effective in practice. In this approach, a human expert provides some rough guidelines on how to model a given domain. For example, the human usually specifies the attributes that the model should contain, often some of the main dependencies that it should encode, and perhaps other aspects. The details, however, are usually filled in automatically, by fitting the model to data. The models produced by this process are usually much better reflections of the domain than models that are purely hand-constructed. Moreover, they can sometimes reveal surprising connections between variables and provide novel insights about a domain.
> 图模型便于构建

**These three components — representation, inference, and learning — are critical components in constructing an intelligent system. We need a declarative representation that is a reasonable encoding of our world model. We need to be able to use this representation effectively to answer a broad range of questions that are of interest. And we need to be able to acquire this distribution, combining expert knowledge and accumulated data. Probabilistic graphical models are one of a small handful of frameworks that support all three capabilities for a broad range of problems.**
> 智能系统的三个成分：表示、推理、学习
> declarative representation 应该是世界的合理编码，我们使用表示来回答问题
# 2 Foundations 
In this chapter, we review some important background material regarding key concepts from probability theory, information theory, and graph theory. This material is included in a separate introductory chapter, since it forms the basis for much of the development in the remainder of the book. Other background material — such as discrete and continuous optimization, algorithmic complexity analysis, and basic algorithmic concepts — is more localized to particular topics in the book. Many of these concepts are presented in the appendix; others are presented in concept boxes in the appropriate places in the text. All of this material is intended to focus only on the minimal subset of ideas required to understand most of the discussion in the remainder of the book, rather than to provide a comprehensive overview of the field it surveys. We encourage the reader to explore additional sources for more details about these areas. 
## 2.1 Probability Theory 
The main focus of this book is on complex probability distributions. In this section we briefly review basic concepts from probability theory. 
### 2.1.1 Probability Distributions 
When we use the word “probability” in day-to-day life, we refer to a degree of confidence that an event of an uncertain nature will occur. For example, the weather report might say “there is a low probability of light rain in the afternoon.” Probability theory deals with the formal foundations for discussing such estimates and the rules they should obey. 
> 概率指对于一个不确定事件发生的信心

Before we discuss the representation of probability, we need to define what the events are to which we want to assign a probability. These events might be diferent outcomes of throwing a die, the outcome of a horse race, the weather configurations in California, or the possible failures of a piece of machinery. 
> 首先定义需要赋予概率的事件
#### 2.1.1.1 Event Spaces 
Formally, we define events by assuming that there is an agreed upon space of possible outcomes, which we denote by $\Omega$ . For example, if we consider dice, we might set $\Omega=\{1,2,3,4,5,6\}$ . In the case of a horse race, the space might be all possible orders of arrivals at the finish line, a much larger space. 
> 我们定义事件时，假设了存在一个包含所有可能结果的空间

In addition, we assume that there is a measurable events $\mathcal S$ to which we are willing to assign probabilities. Formally, each event $\alpha\in S$ is a subset of Ω . In our die example, the event $\{6\}$ represents the case where the die shows 6, and the event $\{1,3,5\}$ represents the case of an odd outcome. In the horse-race example, we might consider the event “Lucky Strike wins,” which contains all the outcomes in which the horse Lucky Strike is first. 
> 每一个事件都是空间 $\Omega$ 的子集

Probability theory requires that the event space satisfy three basic properties:

 • It contains the empty event $\varnothing$ , and the trivial event $\Omega$ .
 • It is closed under union. That is, if $\alpha,\beta\in S$ , then so is $\alpha\cup\beta$ .
 • It is closed under complementation. That is, if $\alpha\in\mathcal S$ , then so is $\Omega-\alpha$ . 

> 事件空间满足：
> 包含空集和本身
> 对于并集运算封闭
> 对于补集运算封闭

The requirement that the event space is closed under union and complementation implies that it is also closed under other Boolean operations, such as intersection and set diference. 
> 上述最后两条说明了事件空间在其他布尔运算下也封闭，例如交集和差集
#### 2.1.1.2 Probability Distributions 
***Definition 2.1*** 
A probability distribution $P$ over $(\Omega,S)$ is a mapping from events in $\mathcal S$ to real values that satisfies the following conditions: 

• $P(\alpha)\ge0$ for all $\alpha\in\mathcal S$ . 
• $P(\Omega)=1$ . 
• If $\alpha,\beta\in{\mathcal{S}}$ and $\alpha\cap\beta=\emptyset$ , then $P(\alpha\cup\beta)=P(\alpha)+P(\beta).$ . 

> 在事件空间和事件上的一个概率分布 $P$ 是一个将事件 $\mathcal S$ 映射到实数的映射，满足：
> 像都非负
> 空间本身的像是1
> 对于两个不相交事件，该映射保留了并集运算（映射为加法）

The first condition states that probabilities are not negative. The second states that the “trivial event,” which allows all possible outcomes, has the maximal possible probability of 1 . The third condition states that the probability that one of two mutually disjoint events will occur is the sum of the probabilities of each event. 
These two conditions imply many other conditions. Of particular interest are $P(\varnothing)=0$ , and $P(\alpha\cup\beta)=P(\alpha)+P(\beta)-P(\alpha\cap\beta)$ . 
> 以上的三个形式还可以推出其他的性质，如上所示
#### 2.1.1.3 Interpretations of Probability 
Before we continue to discuss probability distributions, we need to consider the interpretations that we might assign to them. Intuitively, the probability $P(\alpha)$ of an event $\alpha$ quantifies the degree of confidence that $\alpha$ will occur. If $P(\alpha)=1$ , we are certain that one of the outcomes in $\alpha$ occurs, and if $P(\alpha)\,=\,0$ , we consider all of them impossible. Other probability values represent options that lie between these two extremes. 

This description, however, does not provide an answer to what the numbers mean. There are two common interpretations for probabilities. 
> 对于概率，有两个常见的解释

The *frequentist* interpretation views probabilities as frequencies of events. More precisely, the probability of an event is the fraction of times the event occurs if we repeat the experiment indefinitely. For example, suppose we consider the outcome of a particular die roll. In this case, the statement $P(\alpha)=0.3$ , for $\alpha=\{1,3,5\}$ , states that if we repeatedly roll this die and record the outcome, then the fraction of times the outcomes in $\alpha$ will occur is 0.3 . More precisely, the limit of the sequence of fractions of outcomes in $\alpha$ in the first roll, the first two rolls, the first three rolls, . . . , the first $n$ rolls, . . . is 0.3

The frequentist interpretation gives probabilities a tangible semantics. When we discuss concrete physical systems (for example, dice, coin flips, and card games) we can envision how these frequencies are defined. It is also relatively straightforward to check that frequencies must satisfy the requirements of proper distributions. 
> 频率学派将概率视作事件的频率，即将概率定义为无限重复事件下，是事件发生的次数占实验次数的比值
> 频率学派的解释给了概率真实的语义

The frequentist interpretation fails, however, when we consider events such as “It will rain tomorrow afternoon.” Although the time span of “Tomorrow afternoon” is somewhat ill defined, we expect it to occur exactly once. It is not clear how we define the frequencies of such events. Several attempts have been made to define the probability for such an event by finding a *reference class* of similar events for which frequencies are well defined; however, none of them has proved entirely satisfactory. Thus, the frequentist approach does not provide a satisfactory interpretation for a statement such as “the probability of rain tomorrow afternoon is 0.3.” 

**An alternative interpretation views probabilities as subjective degrees of belief. Under this interpretation, the statement $P(\alpha)\,=\,0.3$ represents a subjective statement about one’s own degree of belief that the event $\alpha$ will come about.** Thus, the statement “the probability of rain tomorrow afternoon is 50 percent” tells us that in the opinion of the speaker, the chances of rain and no rain tomorrow afternoon are the same. Although tomorrow afternoon will occur only once, we can still have uncertainty about its outcome, and represent it using numbers (that is, probabilities). 
> 另一种解释将概率视作信念的主观程度，一个事件的概率表示人对于该事件会发生的信念的把握程度
> 概率被解释为人的主观感受

This description still does not resolve what exactly it means to hold a particular degree of belief. What stops a person from stating that the probability that Bush will win the election is 0.6 and the probability that he will lose is 0.8? The source of the problem is that we need to explain how subjective degrees of beliefs (something that is internal to each one of us) are reflected in our actions. 

This issue is a major concern in subjective probabilities. One possible way of attributing degrees of beliefs is by a betting game. Suppose you believe that $P(\alpha)=0.8$ . Then you would be willing to place a bet of \$1 against \$3 . To see this, note that with probability 0.8 you gain a dollar, and with probability 0.2 you lose \$3 , so on average this bet is a good deal with expected gain of 20 cents. In fact, you might be even tempted to place a bet of \$1 against \$4 . Under this bet the average gain is 0 , so you should not mind. However, you would not consider it worthwhile to place a bet \$1 against \$4 and 10 cents, since that would have negative expected gain. Thus, by finding which bets you are willing to place, we can assess your degrees of beliefs. 
> 但我们需要解释我们对于信念的主观程度如何反映在行动中
> 可以解释为如果期望收益大于0，则我们就可以接受这个概率

The key point of this mental game is the following. If you hold degrees of belief that do not satisfy the rule of probability, then by a clever construction we can find a series of bets that would result in a sure negative outcome for you. Thus, the argument goes, a rational person must hold degrees of belief that satisfy the rules of probability. 

In the remainder of the book we discuss probabilities, but we usually do not explicitly state their interpretation. Since both interpretations lead to the same mathematical rules, the technical definitions hold for both interpretations. 
### 2.1.2 Basic Concepts in Probability 
#### 2.1.2.1 Conditional Probability 
To use a concrete example, suppose we consider a distribution over a population of students taking a certain course. The space of outcomes is simply the set of all students in the population. Now, suppose that we want to reason about the students’ intelligence and their final grade. 
We can define the event $\alpha$ to denote “all students with grade A,” and the event $\beta$ to denote “all students with high intelligence.” Using our distribution, we can consider the probability of these events, as well as the probability of $\alpha\cap\beta$ (the set of intelligent students who got grade A). 
This, however, does not directly tell us how to update our beliefs given new evidence. Suppose we learn that a student has received the grade A; what does that tell us about her intelligence? 

This kind of question arises every time we want to use distributions to reason about the real world. More precisely, after learning that an event $\alpha$ is true, how do we change our probability about $\beta$ occurring? The answer is via the notion of conditional probability . 
Formally, the conditional probability of $\beta$ given $\alpha$ is defined as 

$$
P(\beta\mid\alpha)={\frac{P(\alpha\cap\beta)}{P(\alpha)}}
$$ 
That is, the probability that $\beta$ is true given that we know $\alpha$ is the relative proportion of outcomes satisfying $\beta$ among these that satisfy $\alpha$ . (Note that the conditional probability is not defined when $\begin{array}{r}{P(\alpha)=0.}\end{array}$ .) 
> 在给定 $\alpha$ 的情况下 $\beta$ 为真的概率定义为在满足了 $\alpha$ 的所有事件中，同时满足 $\beta$ 的相对比例
> 注意在 $P(\alpha) = 0$ 时，条件概率没有定义

The conditional probability given an event (say $\alpha$ ) satisfies the properties of definition 2.1 (see exercise 2.4), and thus it is a probability distribution by its own right. Hence, we can think of the conditioning operation as taking one distribution and returning another over the same probability space. 
> 条件概率的定义满足定义 2.1 对于概率的定义，因此条件概率本身也是一个概率分布
> 我们可以将条件运算视作输入一个概率分布，输出一个在相同概率空间的另一个概率分布
#### 2.1.2.2 Chain Rule and Bayes Rule 
From the definition of the conditional distribution, we immediately see that 

$$
P(\alpha\cap\beta)=P(\alpha)P(\beta\mid\alpha).
$$ 
This equality is known as the chain rule of conditional probabilities. More generally, if $\alpha_{1},.\,.\,.\,,\alpha_{k}$ are events, then we can write 

$$
P(\alpha_{1}\cap\ldots\cap\alpha_{k})=P(\alpha_{1})P(\alpha_{2}\mid\alpha_{1})\cdot\cdot\cdot P(\alpha_{k}\mid\alpha_{1}\cap\ldots\cap\alpha_{k-1}).
$$ 
In other words, we can express the probability of a combination of several events in terms of the probability of the first, the probability of the second given the first, and so on. It is important to notice that we can expand this expression using any order of events — the result will remain the same. 

Another immediate consequence of the definition of conditional probability is Bayes’ rule 

$$
P(\alpha\mid\beta)={\frac{P(\beta\mid\alpha)P(\alpha)}{P(\beta)}}.
$$ 
A more general conditional version of Bayes’ rule, where all our probabilities are conditioned on some background event $\gamma,$ , also holds: 

$$
P(\alpha\mid\beta\cap\gamma)={\frac{P(\beta\mid\alpha\cap\gamma)P(\alpha\mid\gamma)}{P(\beta\mid\gamma)}}.
$$ 
Bayes’ rule is important in that it allows us to compute the conditional probability $P(\alpha\mid\beta)$ from the “inverse” conditional probability $P(\beta\mid\alpha)$ . 


Example 2.1 
*Consider the student population, and let Smart denote smart students and GradeA denote students who got grade A. Assume we believe (perhaps based on estimates from past statistics) that $P(G r a d e A\mid S m a r t)\,=\,0.6$ , and now we learn that a particular student received grade A. Can we estimate the probability that the student is smart? According to Bayes’ rule, this depends on our prior probability for students being smart (before we learn anything about them) and the prior probability of students receiving high grades. For example, suppose that $P(S m a r t)\,=\,0.3$ and $P(G r a d e A)\;=\;0.2,$ , then we have that $P(S m a r t\mid G r a d e A)\;=\;0.6*0.3/0.2\;=\;0.9$ . That is, an A grade strongly suggests that the student is smart. On the other hand, if the test was easier and high grades were more common, say, $P(G r a d e A)~=~0.4$ then we would get that $P(S m a r t\mid G r a d e A)=0.6*0.3/0.4=0.45,$ , which is much less conclusive about the student.* 

Another classic example that shows the importance of this reasoning is in disease screening. To see this, consider the following hypothetical example (none of the mentioned figures are related to real statistics). 


Example 2.2 
*Suppose that a tuberculosis (TB) skin test is 95 percent accurate. That is, if the patient is TB-infected, then the test will be positive with probability 0 . 95 , and if the patient is not infected, then the test will be negative with probability 0.95 . Now suppose that a person gets a positive test result. What is the probability that he is infected? Naive reasoning suggests that if the test result is wrong 5 percent of the time, then the probability that the subject is infected is 0.95 . That is, 95 percent of subjects with positive results have TB.* 

*If we consider the problem by applying Bayes’ rule, we see that we need to consider the prior probability of TB infection, and the probability of getting positive test result. Suppose that 1 in 1000 of the subjects who get tested is infected. That is, $P(T B)=0.001$ . What is the probability of getting a positive t rom our description, we see that $0.001\cdot0.95$ positive result, and $0.999{\cdot}0.05$ · u $P(P o s i t i v e)=0.0509$ . Applying Bayes’ rule, we get that $P(T B\mid P o s i t i v e)=0.001\cdot0.95/0.0509\approx0.0187.$ Thus, although a subject with a positive test is much more probable to be TB-infected than is a random subject, fewer than 2 percent of these subjects are $T\!B$ -infected.* 
### 2.1.3 Random Variables and Joint Distributions 
#### 2.1.3.1 Motivation 
Our discussion of probability distributions deals with events. Formally, we can consider any event from the set of measurable events. The description of events is in terms of sets of outcomes. In many cases, however, it would be more natural to consider attributes of the outcome. For example, if we consider a patient, we might consider attributes such as “age,” “gender,” and “smoking history” that are relevant for assigning probability over possible diseases and symptoms. We would like then consider events such as “age $>55$ , heavy smoking history, and sufers from repeated cough.” 
> 我们之前考虑的是定义在事件上的概率分布，事件就是结果集合
> 我们现在考虑结果的属性，我们将事件定义为其属性满足特定要求的结果的集合

To use a concrete example, consider again a distribution over a population of students in a course. Suppose that we want to reason about the intelligence of students, their final grades, and so forth. We can use an event such as GradeA to denote the subset of students that received the grade A and use it in our formulation. However, this discussion becomes rather cumbersome if we also want to consider students with grade B, students with grade C, and so on. Instead, we would like to consider a way of directly referring to a student’s grade in a clean, mathematical way. 

The formal machinery for discussing attributes and their values in diferent outcomes are random variables . A random variable is a way of reporting an attribute of the outcome. For example, suppose we have a random variable Grade that reports the final grade of a student, then the statement $P(G r a d e=A)$ is another notation for $P(G r a d e A)$ . 
> 我们将定义了结果集合的属性称为随机变量，随机变量是报告一个结果的属性的一个方式，例如，分数>90，定义了结果 Grade=A
#### 2.1.3.2 What Is a Random Variable? 
Formally, a random variable, such as Grade , is defined by a function that associates with each outcome in $\Omega$ a value. For example, Grade is defined by a function $f_{G r a d e}$ that maps each person in $\Omega$ to his or her grade (say, one of A, B, or C). The event $G r a d e\,=\,A$ is a shorthand for the event $\{\omega\,\in\,\Omega\,:\,f_{G r a d e}(\omega)\,=\,A\}$ . In our example, we might also have a random variable Intelligence that (for simplicity) takes as values either “high” or “low.” In this case, the event “Intelligen  = high" refers, as can be expected, to the set of smart (high intelligence) students. 
> 随机变量定义为将事件集合 $\Omega$ 中的每个结果关联到一个值的函数
> 该函数输入一个事件，输出一个和该事件的性质相关联的值

Random variables can take diferent sets of values. We can think of categorical (or discrete ) random variables that take one of a few values, as in our two preceding examples. We can also talk about random variables that can take infinitely many values (for example, integer or real values), such as Height that denotes a student’s height. We use $V a l(X)$ to denote the set of values that a random variable $X$ can take. 
> 随机变量的取值可以是离散或者连续

In most of the discussion in this book we examine either categorical random variables or random variables that take real values. We will usually use uppercase roman letters $X,Y,Z$ to denote random variables. 
In discussing generic random variables, we often use a lowercase letter to refer to a value of a random variable. Thus, we use $x$ to refer to a generic value of $X$ . 
For example, in statements such as $P(X\,=\,x)\,\geq\,0$ for all $x\in\mathit{V a l}(X)$ . When we discuss categorical random variables, we use the notation $x^{1},\cdot\cdot\cdot,x^{k}$ , for $k\,=\,|\mathit{V a l}(X)|$ (the number of elements in $V a l(X))$ ), when we need to enumerate the specific values of $X$ , for example, in statements such as 

$$
\sum_{i=1}^{k}P(X=x^{i})=1.
$$ 
The distribution over such a variable is called a multinomial. 
> 多项式分布

In the case of a binary-valued random variable $X$ , where $V a l(X)=\{f a l s e,t r u e\}$ , we often use $x^{1}$ to denote the value true for $X$ , and x $x^{0}$ to denote the value false . The distribution of such a random variable is called a Bernoulli distribution . 
> 二值随机变量：伯努利分布

We also use boldface type to denote sets of random variables. Thus, $\pmb X,\pmb Y$ , or $\pmb Z$ are typically used to denote a set of random variables, while $\pmb x,\;\pmb y,\;\pmb z$ denote assignments of values to the variables in these sets. We extend the definition of $V a l(X)$ to refer to sets of variables in the obvious way. Thus, ${\pmb x}$ is always a member of $V a l(X)$ . 

For $Y\subseteq X$ , we use $\pmb x\langle \pmb Y\rangle$ to refer to to the assignment within ${\pmb x}$ to the variables in $Y$ . 
>表示 $\pmb Y$ 中的随机变量的赋值在 $\pmb x$ 内

For two assignments $\pmb x$ (to X)  and $\pmb y$ (to Y) , we say that $\pmb{x}\sim\pmb{y}$ if they agree on the their intersection, that is, $x\langle X\cap Y\rangle=y\langle X\cap Y\rangle$ 
> $\pmb x \sim \pmb y$ 表示两个随机变量的取值在 X, Y 交集内

In many cases, the notation $P(X=x)$ is redundant, since the fact that $x$ is a value of X is already reported by our choice of letter. Thus, in many texts on probability, the identity of a random variable is not explicitly mentioned, but can be inferred through the notation used for its value. Thus, we use $P(x)$ as a shorthand for $P(X=x)$ when the identity of the random variable is clear from the context. 
Another shorthand notation is that $\textstyle\sum x$ refers to a sum over all possible values that $X$ can take. Thus, the preceding statement will often appear as $\textstyle\sum_{x}P(x)=1$ . 
Finally, another standard notation has to do with conjunction. Rather than write $P((X=x)\cap(Y=y))$  , we write $P(X=x,Y=y)$ , or just $P(x,y)$ . 
#### 2.1.3.3 Marginal and Joint Distributions 
Once we define a random variable $X$ , we can consider the distribution over events that can be described using $X$ . This distribution is often referred to as the marginal distribution over the random variable $X$ . We denote this distribution by $P(X)$ . 
> 我们定义在随机变量上的边际分布为可以用随机变量描述的事件上的分布

Returning to our population example, consider the random variable Intelligence . The marginal distribution over Intelligence assigns probability to specific events such as $P(I n t e l l i g e n c e=h i g h)$ and $P(I n t e l l i g e n c e=l o w)$ , as well as to the trivial event $P(I n t e l l i g e n c e\in\{h i g h,l o w\})$ . Note that these probabilities are defined by the probability distribution over the original space. For concreteness, suppose that $P(I n t e l l i g e n c e=h i g h)=0.3$ , $P(I n t e l l i g e n c e=l o w)=0.7.$ . 

If we consider the random variable Grade , we can also define a marginal distribution. This is a distribution over all events that can be described in terms of the Grade variable. In our example, we have that $P(G r a d e=A)=0.25$ , $P(G r a d e=B)=0.37,$ , and $P(G r a d e=C)=0.38$ . 

It should be fairly obvious that the marginal distribution is a probability distribution satisfying the properties of definition 2.1. In fact, the only change is that we restrict our attention to the subsets of $\mathcal S$ that can be described with the random variable $X$ . 
> 边际分布满足定义 2.1 定义的概率分布，差别在于我们将注意力限制在了事件空间的子空间，即可以用随机变量 $\pmb X$ 描述的事件子集上

In many situations, we are interested in questions that involve the values of several random variables. For example, we might be interested in the event “ Intelligence $=h i g h$ and $G r a d e=A$ .” To discuss such events, we need to consider the joint distribution over these two random variables. 
In general, the joint distribution over a set $\mathcal{X}=\{X_{1},\ldots,X_{n}\}$ of random variables is denoted by $P(X_{1},\cdot\cdot\cdot,X_{n})$ and is the distribution that assigns probabilities to events that are specified in terms of these random variables. We use $\xi$ to refer to a full assignment to the variables in $\mathcal{X}$ , that is, $\xi\in V a l(\mathcal{X})$ 
> 涉及多个随机变量时，我们考虑联合分布，联合分布定义于随机变量集合上，或者说是给和这些随机变量相关的事件赋值概率的分布

The joint distribution of two random variables has to be consistent with the marginal distribution, in that $\begin{array}{r}{P(x)=\sum_{y}P(x,y)}\end{array}$ . This relationship is shown in figure 2.1, where we compute the marginal distribution over Grade by summing the probabilities along each row. Similarly, we find the marginal distribution over Intelligence by summing out along each column. 
The resulting sums are typically written in the row or column margins, whence the term “marginal distribution.” 

Suppose we have a joint distribution over the variables ${\mathcal X}\ =\ \{X_{1},.\,.\,.\,,X_{n}\}$ . The most fine-grained events we can discuss using these variables are ones of the form “ ${\mathit{X}}_{1}\,=\,{\mathit{x}}_{1}$ and $X_{2}=x_{2},\,.\,.\,.,$ and $X_{n}=x_{n}$ “ for a choice of values $x_{1},\ldots,x_{n}$ for all the variables. 
Moreover, any two such events must be either identical or disjoint, since they both assign values to all the variables in $\mathcal{X}$ . In addition, any event defined using variables in $\mathcal{X}$ must be a union of a set of such events. 
Thus, we are effectively working in a *canonical outcome space* : a space where each outcome corresponds to a joint assignment to $X_{1},\ldots, X_{n}$ . More precisely, all our probability computations remain the same whether we consider the original outcome space (for example, all students), or the canonical space (for example, all combinations of intelligence and grade). We use $\xi$ to denote these *atomic outcomes* : those assigning a value to each variable in $\mathcal{X}$ . 
> 标准的结果空间：空间中每一个结果对应于对随机变量 $X_1,\dots,X_n$ 的一个联合赋值，也就是将一个个的事件定义为了一个个的随机变量赋值
> 在原始的结果空间和在标准的结果空间的概率计算是等价的
> 标准的结果空间中，每个结果都是原子结果

For example, if we let $\mathcal{X}=\{I n t e l l i g e n c e, G r a d e\}$ , there are six atomic outcomes, shown in figure 2.1. The figure also shows one possible joint distribution over these six outcomes. 

Based on this discussion, from now on we will not explicitly specify the set of outcomes and measurable events, and instead implicitly assume the canonical outcome space. 
> 我们都假设使用标准的结果空间
#### 2.1.3.4 Conditional Probability 
The notion of conditional probability extends to induced distributions over random variables. For example, we use the notation P ( Intelligence | $G r a d e=A$ ) to denote the conditional distribution over the events describable by Intelligence given the knowledge that the student’s grade is A. 

Note that the conditional distribution over a random variable given an observation of the value of another one is not the same as the marginal distribution. In our example, P ( Intelligence = $h i g h)\;=\; 0.3$ , and $P (I n t e l l i g e n c e=h i g h\mid G r a d e= A)\;=\; 0.18/0.25\;=\; 0.72$ . Thus, clearly P ( Intelligence | $G r a d e=A_{\rangle}$ ) is diferent from the marginal distribution $P (I n t e l l i g e n c e)$ . The latter distribution represents our prior knowledge about students before learning anything else about a particular student, while the conditional distribution represents our more informed distribution after learning her grade. 

We will often use the notation $P (X\mid Y)$ to represent a set of conditional probabilty distributions. 
> $P (X\mid Y)$ 表示一个条件概率分布的集合

Intuitively, for each value of Y , this object assigns a probability over values of X using the conditional probability. This notation allows us to write the shorthand version of the chain rule: $P (X, Y)=P (X) P (Y\mid X)$ , which can be extended to multiple variables as 

$$
P (X_{1},\ldots, X_{k})=P (X_{1}) P (X_{2}\mid X_{1})\cdot\cdot\cdot P (X_{k}\mid X_{1},\ldots, X_{k-1}).
$$ 
Similarly, we can state Bayes’ rule in terms of conditional probability distributions: 

$$
P (X\mid Y)={\frac{P (X) P (Y\mid X)}{P (Y)}}.
$$ 
### 2.1.4 Independence and Conditional Independence 
#### 2.1.4.1 Independence 
As we mentioned, we usually expect $P (\alpha\mid\beta)$ to be diferent from $P (\alpha)$ . That is, learning that $\beta$ is true changes our probability over α . However, in some situations equality can occur, so that $P (\alpha\mid\beta)=P (\alpha)$ . That is, learning that $\beta$ occurs did not change our probability of $\alpha$ . 

***Definition 2.2*** 
We say that an event $\alpha$ is independent of event $\beta$ in $P$ , denoted $P\vDash (\alpha\perp\beta).$  if $P (\alpha\mid\beta)=$ $P (\alpha)$ or if $P (\beta)=0$ . 
> 在概率分布 $P$ 中，若 $\alpha$ 在 $\beta$ 下的条件概率分布等于 $\alpha$ 的边际分布，或者 $\beta$ 的概率是 0，则事件 $\alpha$ 独立于 $\beta$ ，记为 $P\vDash (\alpha \perp \beta)$

We can also provide an alternative definition for the concept of independence: 

***Proposition 2.1*** 
A distribution $P$ satifies $\alpha \perp \beta$ if and only if $P(\alpha \cap \beta) = P(\alpha) P(\beta)$

Proof 
Consider first the case where $P (\beta)\,=\, 0$ ; here, we also have $P (\alpha\cap\beta)\,=\, 0$ , and so the equivalence immediately hold 
When $P (\beta)\,\neq\, 0$ we can use the chain rule; we write $P (\alpha\cap\beta)=P (\alpha\mid\beta) P (\beta)$ . Since α is independent of β , we have that $P (\alpha\mid\beta)=P (\alpha)$ . Thus, $P (\alpha\cap\beta)=P (\alpha) P (\beta)$ . 
Conversely, suppose that $P (\alpha\cap\beta)=P (\alpha) P (\beta)$ . Then, by definition, we have that 

$$
P (\alpha\mid\beta)={\frac{P (\alpha\cap\beta)}{P (\beta)}}={\frac{P (\alpha) P (\beta)}{P (\beta)}}=P (\alpha).
$$ 
As an immediate consequence of this alternative definition, we see that independence is a symmetric notion. That is, $(\alpha\perp\beta)$ implies $(\beta\perp\alpha)$ . 
> 通过该引理，可以立刻知道独立性是对称的


Example 2.3 
*For example, suppose that we toss two coins, and let $\alpha$ be the event “the first toss results in a head” and $\beta$ the event “the second toss results in a head.” It is not hard to convince ourselves that we expect that these two events to be independent. Learning that $\beta$ is true would not change our probability of $\alpha$ . In this case, we see two diferent physical processes (that is, coin tosses) leading to the events, which makes it intuitive that the probabilities of the two are independent. In certain cases, the same process can lead to independent events. For example, consider the event $\alpha$ denoting “the die outcome is even” and the event $\beta$ denoting “the die outcome is 1 or $2.^{\prime\prime}$ It is easy to check that if the die is fair (each of the six possible outcomes has probability $\frac{1}{6}$ ), then these two events are independent.* 
#### 2.1.4.2 Conditional Independence 
**While independence is a useful property, it is not often that we encounter two independent events. A more common situation is when two events are independent given an additional event**. For example, suppose we want to reason about the chance that our student is accepted to graduate studies at Stanford or MIT. Denote by Stanford the event “admitted to Stanford” and by MIT the event “admitted to MIT.” In most reasonable distributions, these two events are not independent. If we learn that a student was admitted to Stanford, then our estimate of her probability of being accepted at MIT is now higher, since it is a sign that she is a promising student. 

Now, suppose that both universities base their decisions only on the student’s grade point average (GPA), and we know that our student has a GPA of A. In this case, we might argue that learning that the student was admitted to Stanford should not change the probability that she will be admitted to MIT: Her GPA already tells us the information relevant to her chances of admission to MIT, and finding out about her admission to Stanford does not change that. Formally, the statement is 

$P (M I T\mid S t a n f o r d, G r a d e A)=P (M I T\mid G r a d e A).$ In this case, we say that MIT is conditionally independent of Stanford given GradeA 
> 在给定条件下，两个事件独立，即条件独立
> 两个事件条件独立不表示两个事件独立，在不给定条件的情况下，两个事件可能不相互独立

***Definition 2.3 conditional independence*** 
We say that an event $\alpha$ is conditionally independent of event $\beta$ given event $\gamma$ in $P$ , denoted $P\vDash (\alpha\perp\beta\mid\gamma)$  , if $P(\alpha\mid\beta\cap\gamma)=P (\alpha\mid\gamma)$ or if $P (\beta\cap\gamma)=0$ . 
> 概率分布 $P$ 中，若事件 $\alpha$ 在同时给定 $\beta$ 和 $\gamma$ 的条件下发生的概率等于事件 $\alpha$ 在仅给定 $\gamma$ 的条件下发生的概率，或者事件 $\beta$ 和 $\gamma$ 同时发生的概率为 0，则称事件 $\alpha$ 在给定条件 $\gamma$ 的条件下条件独立于事件 $\beta$ ，记为 $P \vDash (\alpha \perp \beta \mid \gamma)$

It is easy to extend the arguments we have seen in the case of (unconditional) independencies to give an alternative definition. 

***Proposition 2.2*** 
$P$ satisfies $P\vDash (\alpha \perp \beta \mid \gamma)$ if and only if $P(\alpha \cap \beta \mid \gamma) = P(\alpha \mid \gamma)P(\beta \mid \gamma)$
> $P (\alpha , \beta \mid \gamma) = P (\alpha \mid \beta, \gamma) P (\beta \mid \gamma) = P (\alpha \mid \gamma) P (\beta \mid \gamma)$
#### 2.1.4.3 Independence of Random Variables 
Until now, we have focused on independence between events. Thus, we can say that two events, such as one toss landing heads and a second also landing heads, are independent. However, we would like to say that any pair of outcomes of the coin tosses is independent. To capture such statements, we can examine the generalization of independence to sets of random variables. 
> 我们将事件之间的独立性推广到随机变量之间的独立性

***Definition 2.4*** 
Let $X, Y, Z$ be sets of random variables. We say that $X$ is conditionally independent of $Y$ given $Z$ in a distribution $P$ if $P$ satisfies $(X=x\;\bot\; Y=y\;\vert\; Z=z)$ for all values $\pmb{x}\,\in\, V a l (\pmb{X})$ , $\pmb{y}\in V a l (\pmb{Y})$ , and $z\in V a l (Z)$ . The variables in the set Z are often said to be observed. if the $Z$ is empty, then instead of writing $(X\perp Y\mid\emptyset)$ , we write $(X\perp Y)$ and say that X and Y are marginally independent . 
> 给定随机变量 $X,Y,Z$ ，如果概率分布 $P$ 满足对于随机变量 $X, Y, Z$ 的所有取值 $x, y, z$ 都有 $(X = x \perp Y = y \mid Z = z)$ ，则称随机变量 $X$ 在给定 $Z$ 的条件下条件独立于 $Y$
> $Z$ 往往称为是被观察到的随机变量
> 如果 $Z$ 是空集，则直接称随机变量 X 和 Y 边际独立

Thus, an independence statement over random variables is a universal quantification over all possible values of the random variables. 
> 随机变量之间相互独立表示它们之间的所有可能取值都互不相关

The alternative characterization of conditional independence follows immediately: 

***Proposition 2.3*** 
The distribution $P$ satisfies $(\pmb X \perp \pmb Y\mid \pmb Z)$ if and only if $P(\pmb X \cap \pmb Y \mid \pmb Z) = P(\pmb X \mid \pmb Z)P(\pmb Y \mid \pmb Z)$

Suppose we learn about a conditional independence. Can we conclude other independence properties that must hold in the distribution? We have already seen one such example: 

- **Symmetry** : 

$$
(X\perp Y\mid Z)\Longrightarrow (Y\perp X\mid Z).\tag{2.7}
$$ 
There are several other properties that hold for conditional independence, and that often provide a very clean method for proving important properties about distributions. Some key properties are: 

- **Decomposition** : 

$$
(X\bot Y, W\mid Z)\Longrightarrow (X\bot Y\mid Z).\tag{2.8}
$$ 
> 证明：
> 由 $(X \perp Y, W \mid Z)$ 可知 $P(X,Y,W\mid Z)=P(X\mid Z)P(Y,W\mid Z)$
> 故 $\sum_W P (X, Y, W\mid Z) = \sum_W P (X\mid Z) P (Y, W\mid Z)=P(X\mid Z)\sum_W P(Y,W\mid Z)$
> 即 $P (X, Y\mid Z) = P (X\mid Z) P (Y\mid Z)$
> 故 $(X\perp Y\mid Z)$

- **Weak union** : 

$$
(X\perp Y, W\mid Z)\Longrightarrow (X\perp Y\mid Z, W).\tag{2.9}
$$ 
> 证明：
> $P (X, Y\mid Z, W) = P (X\mid Y, Z, W) P (Y\mid Z, W)$
> 由 $(X\perp Y, W\mid Z)$ 可知 $P (X\mid Y, Z, W) = P (X\mid Z)$
> 根据 decomposition，因为 $(X\perp Y, W \mid Z)$ ，故 $(X\perp W \mid Z)$
> 因此 $P (X\mid Z) = P (X\mid Z, W)$
> 故 $P (X, Y\mid Z, W) = P (X\mid Z, W) P (Y\mid Z, W)$
> 故 $(X\perp Y \mid Z, W)$

- **Contraction** : 

$$
(X\bot W\mid Z, Y)\,\&\, (X\bot Y\mid Z)\Longrightarrow (X\bot Y, W\mid Z).\tag{2.10}
$$ 
> 证明：
> $P (X, Y, W \mid Z) = P (X\mid Y, W, Z) P (Y, W\mid Z)$
> 由 $(X\perp W \mid Z, Y)$ 可知 $P (X, Y, W\mid Z) = P (X\mid Z, Y) P (Y, W\mid Z)$
> 由 $(X\perp Y \mid Z)$ 可知 $P (X, Y, W \mid Z) = P (X\mid Z) P (Y, W\mid Z)$
> 故 $(X\perp Y, W \mid Z)$

An additional important property does not hold in general, but it does hold in an important subclass of distributions. 

***Definition 2.5*** 
A distribution $P$ is said to be positive if for all events $\alpha\in{\mathcal{S}}$  such that $\alpha\neq\emptyset$ , we have that $P (\alpha)>0$ . 
> postive distribution: 只要事件不为空，发生的概率就大于0

For positive distributions, we also have the following property: 

- **Intersection** : For positive distributions, and for mutually disjoint sets $X, Y, Z, W$ : 

$$
(X\bot Y\mid Z, W)\,\&\, (X\bot W\mid Z, Y)\Longrightarrow (X\bot Y, W\mid Z).\tag{2.11}
$$ 
> 证明：
> 由 $P (X\perp Y \mid Z, W)$ 可知 $P (X\mid Y, Z, W) = P (X\mid Z, W)$
> 由 $P (X\perp W \mid Z, Y)$ 可知 $P (X\mid Y, Z, W) = P (X\mid Z, Y)$
> 故 $P (X\mid Z, W) = P (X\mid Z, Y)$
> 即 $\frac {P (X, W\mid Z)}{P (W\mid Z)} = \frac {P (X, Y\mid Z)}{P (Y\mid Z)}$（这一步要求 $P$ 是正分布）
> 故 $P (X, W\mid Z) P (Y\mid Z) = P (X, Y \mid Z) P (W\mid Z)$
> 同时对 $W$ 求和得到 $\sum_W P (X, W\mid Z) P (Y\mid Z) = \sum_W P (X, Y\mid Z) P (W\mid Z)$
> 即 $P (X\mid Z) P (Y\mid Z) = P (X, Y\mid Z)$
> 故 $(X\perp Y\mid Z)$
> 根据 constraction，容易知道 $(X\perp Y, W \mid Z)$

The proof of these properties is not difcult. 

For example, to prove Decomposition, assume that $(X\ \bot\ Y, W\ |\ Z)$ holds. Then, from the definition of conditional independence, we have that $P (X, Y, W\mid Z)\,=\, P (X\mid Z) P (Y, W\mid Z)$ . Now, using basic rules of probability and arithmetic, we can show 

$$
{\begin{array}{r c l}{P (X, Y\mid Z)}&{=}&{\displaystyle\sum_{w}P (X, Y, w\mid Z)}\\ &{=}&{\displaystyle\sum_{w}P (X\mid Z) P (Y, w\mid Z)}\\ &{=}&{\displaystyle P (X\mid Z)\sum_{w}P (Y, w\mid Z)}\\ &{=}&{P (X\mid Z) P (Y\mid Z).}\end{array}}
$$ 
The only property we used here is called “reasoning by cases” (see exercise 2.6). We conclude that $(X\perp Y\mid Z)$ . 
> reasoning by cases: 逐例分析，将一个随机变量的所有取值列出来分析
### 2.1.5 Querying a Distribution 
Our focus throughout this book is on using a joint probability distribution over multiple random variables to answer queries of interest. 
#### 2.1.5.1 Probability Queries 
Perhaps the most common query type is the probability query . Such a query consists of two parts: 

- **The evidence** : a subset $E$ of random variables in the model, and an instantiation $e$ to these variables; 
- **The query variable**: a subset of $Y$ of random variables in the network

Our task is to compute:

$$
P (Y\mid E=e),
$$ 
that is, the posterior probability distribution over the values $y$ of $Y$ , conditioned on the fact that $E=e$ . 
This expression can also be viewed as the marginal over $Y$ , in the distribution we obtain by conditioning on $e$ . 
> 概率查询：查询变量 $Y$ 在证据 $E$ 的某个特定值下的条件概率分布
#### 2.1.5.2 MAP Queries 
A second important type of task is that of finding a high-probability joint assignment to some subset of variables. 
> 找到对于某个变量子集最高概率的赋值

The simplest variant of this type of task is the $M A P$ query (also called most probable explanation (MPE) ), whose aim is to find the MAP assignment — the most likely assignment to all of the (non-evidence) variables. More precisely, if we let $W=\mathcal{X}-E$ , our task is to find the most likely assignment to the variables in W given the evidence $E=e$ : 

$$
\operatorname{MAP}(W\mid e)=\arg\operatorname*{max}_{w}P (w, e),\tag{2.12}
$$ 
where, in general, $\operatorname{arg\, max}_{x}f (x)$ represents the value of $x$ for which $f (x)$ is maximal. Note that there might be more than one assignment that has the highest posterior probability. In this case, we can either decide that the MAP task is to return the set of possible assignments, or to return an arbitrary member of that set. 
> MAP 查询：在给定证据 $E=e$ 的情况下，变量子集 $W$ 最有可能的取值 $w$，注意取值可以不止一个

It is important to understand the diference between MAP queries and probability queries. In a MAP query, we are finding the most likely joint assignment to $W$ . To find the most likely assignment to a single variable $A$ , we could simply compute $P (A\mid e)$ and then pick the most likely value. **However, the assignment where each variable individually picks its most likely value can be quite diferent from the most likely joint assignment to all variables simultaneously.** This phenomenon can occur even in the simplest case, where we have no evidence. 
> 一组随机变量最优可能的联合取值不同于各个随机变量各自最优可能的单独取值
#### 2.1.5.3 Marginal MAP Queries 
To motivate our second query type, let us return to the phenomenon demonstrated in example 2.4. 
Now, consider a medical diagnosis problem, where the most likely disease has multiple possible symptoms, each of which occurs with some probability, but not an overwhelming probability. On the other hand, a somewhat rarer disease might have only a few symptoms, each of which is very likely given the disease. As in our simple example, the MAP assignment to the data and the symptoms might be higher for the second disease than for the first one. The solution here is to look for the most likely assignment to the disease variable (s) only, rather than the most likely assignment to both the disease and symptom variables. This approach suggests the use of a more general query type. 

In the marginal MAP query, we have a subset of variables $Y$ that forms our query. The task is to find the most likely assignment to the variables in $Y$ given the evidence $E=e$ : 

$$
\operatorname{MAP}(Y\mid e)=\arg\operatorname*{max}_{y}P (y\mid e).
$$ 
If we let $Z=\mathcal{X}-Y-E$ , the marginal MAP task is to compute: 

$$
\operatorname{MAP}(Y\mid e)=\arg\operatorname*{max}_{Y}\sum_{Z}P (Y, Z\mid e).
$$ 
Thus, marginal MAP queries contain both summations and maximizations; in a way, it contains elements of both a conditional probability query and a MAP query. 

Note that example 2.4 shows that marginal MAP assignments are not monotonic: the most $\operatorname{MAP}(Y_{1}\mid e)$ might be completely diferent from the assignment to $Y_{1}$ in $\operatorname{MAP}(\{Y_{1}, Y_{2}\}\mid e)$  . 
Thus, in particular, we cannot use a MAP query to give us the correct answer to a marginal MAP query. 
### 2.1.6 Continuous Spaces 
In the previous section, we focused on random variables that have a finite set of possible values. In many situations, we also want to reason about continuous quantities such as weight, height, duration, or cost that take real numbers in $I\!\! R$ . 
> 我们还需要考虑取值是连续的随机变量

When dealing with probabilities over continuous random variables, we have to deal with some technical issues. For example, suppose that we want to reason about a random variable $X$ that can take values in the range between 0 and 1 . That is, $V a l (X)$ is the interval $[0,1]$ . Moreover, assume that we want to assign each number in this range equal probability. What would be the probability of a number $x$ ? Clearly, since each $x$ has the same probability, and there are infinite number of values, we must have that $P (X=x)=0$ . This problem appears even if we do not require uniform probability. 
#### 2.1.6.1 Probability Density Functions 
How do we define probability over a continuous random variable? We say that a function $p: I\!\! R\mapsto I\!\! R$ is a probability density function or $(P D F)$ for $X$ if it is a nonnegative integrable function such that 
> 概率密度函数定义为一个非负的可积函数，积分和是 1

$$
\int_{V a l (X)}p (x) d x=1.
$$ 
That is, the integral over the set of possible values of $X$ is 1. 
The PDF defines a distribution for $X$ as follows: 
for any $x$ in our event space: 

$$
P (X\leq a)=\intop_{-\infty}^{a}p (x) d x.
$$ 
The function $P$ is the cumulative distribution for $X$ . 
> $P$ 定义为随机变量 $X$  的累计分布

We can easily employ the rules of probability to see that by using the density function we can evaluate the probability of other events. For example, 

$$
P (a\leq X\leq b)=\intop_{a}^{b}p (x) d x.
$$ 
Intuitively, the value of a PDF $p (x)$ at a point $x$ is the incremental amount that $x$ adds to the cumulative distribution in the integration process. The higher the value of $p$ at and around $x$ , the more mass is added to the cumulative distribution as it passes $x$ . The simplest PDF is the uniform distribution. 
> 直觉上，$x$ 在概率密度函数上的取值 $p (x)$ 就是它在积分过程中对累计分布添加的增量，取值越高，增量越大

***Definition 2.6*** 
uniform distribution A variable $X$ has $^a$ uniform distribution over $[a, b]$ , denoted $X\sim\mathrm{Unif}[\mathrm{a},\mathrm{b}]$ if it has the PDF 

$$
p (x)={\left\{\begin{array}{l l}{{\frac{1}{b-a}}}&{b\geq x\geq a}\\ {0}&{{\mathrm{otherwise.}}}\end{array}\right.}
$$ 
Thus, the probability of any subinterval of $[a, b]$ is proportional its size relative to the size of $[a, b]$ . 
> 各个子区间的概率和它们的长度成比例

Note that, if $b-a<1$ , then the density can be greater than 1 . Although this looks unintuitive, this situation can occur even in a legal PDF, if the interval over which the value is greater than 1 is not too large. We have only to satisfy the constraint that the total area under the PDF is 1 . 

As a more complex example, consider the Gaussian distribution. 

***Definition 2.7*** 
A random variable $X$ has $a$ Gaussian distribution with mean $\mu$ and variance $\sigma^{2}$ , denoted $X\sim$ ${\mathcal{N}}\left (\mu;\sigma^{2}\right)$ , if it has the PDF 

$$
p (x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}}.
$$ 
$A$ standard Gaussian is one with mean 0 and variance 1 . 

A Gaussian distribution has a bell-like curve, where the mean parameter $\mu$ controls the location of the peak, that is, the value for which the Gaussian gets its maximum value. The variance parameter $\sigma^{2}$ determines how peaked the Gaussian is: the smaller the variance, the more peaked the Gaussian. 

Figure 2.2 shows the probability density function of a few diferent Gaussian distributions. 

More technically, the probability density function is specified as an exponential, where the expression in the exponent corresponds to the square of the number of standard deviations $\sigma$ that $x$ is away from the mean $\mu$ . The probability of $x$ decreases exponentially with the square of its deviation from the mean, as measured in units of its standard deviation. 
> 概率密度函数是指数函数，指数部分的表达式和随机变量相对于均值的距离和随机变量的方差有关
> $p (x)$ 随着它相对于均值的距离的平方指数下降，衡量单位是它的标准差
#### 2.1.6.2 Joint Density Functions 
The discussion of density functions for a single variable naturally extends for joint distributions of continuous random variables. 

***Definition 2.8***
Let $P$ be a joint distribution over continuous random variables $X_{1},\dots, X_{n}$ . A function $p (x_{1},\dots, x_{n})$ is a joint density function of $X_{1},\dots, X_{n}$ if 

- $p (x_{1},\cdot\cdot\cdot, x_{n})\geq0$ for all values $x_{1},\dots, x_{n}$ of $X_{1},\dots, X_{n}$ . 
- $p$ is an integrable function. 
- For any choice of $a_{1},\dotsc, a_{n}$ , and $b_{1},\dots, b_{n},$ 

$$P (a_{1}\leq X_{1}\leq b_{1},\ldots, a_{n}\leq X_{n}\leq b_{n})=\intop_{a_{1}}^{b_{1}}\cdot\cdot\cdot\intop_{a_{n}}^{b_{n}}p (x_{1},\ldots, x_{n}) d x_{1}\ldots d x_{n}.$$

> 联合概率密度函数：
> 定义域上非负
> 可积
> 可以分段积分 

Thus, a joint density specifies the probability of any joint event over the variables of interest. 

Both the uniform distribution and the Gaussian distribution have natural extensions to the multivariate case. The definition of a multivariate uniform distribution is straightforward. We defer the definition of the multivariate Gaussian to section 7.1. 

From the joint density we can derive the marginal density of any random variable by integrating out the other variables. Thus, for example, if $p (x, y)$ is the joint density of $X$ and $Y$ , then 

$$
p (x)=\intop_{-\infty}^{\infty}p (x, y) d y.
$$ 
To see ahy this equality holds, note that the event $a\leq X\leq b$ is, by definition, equal to the event “ ${}^{u}a\leq X\leq b$ and $-\infty\leq Y\leq\infty$ .” This rule is the direct analogue of marginalization for discrete variables. 
Note that, as with discrete probability distributions, we abuse notation a bit and use $p$ to denote both the joint density of $X$ and $Y$ and the marginal density of $X$ . 
In cases where the distinction is not clear, we use subscripts, so that $p_{X}$ will be the marginal density, of $X$ , and $p_{X, Y}$ the joint density. 
#### 2.1.6.3 Conditional Density Functions 
As with discrete random variables, we want to be able to describe conditional distributions of continuous variables. Suppose, for example, we want to define $P (Y\mid X=x)$ . Applying the definition of conditional distribution (equation (2.1)), we run into a problem, since $P (X=x)=$ 0 . Thus, the ratio of $P (Y, X=x)$ and $P (X=x)$ is undefined. 

To avoid this problem, we might consider conditioning on the event $x-\epsilon\,\leq\, X\,\leq\, x+\epsilon,$ , which can have a positive probability. Now, the conditional probability is well defined. Thus, we might consider the limit of this quantity when $\epsilon\rightarrow0$ . We define 

$$
P (Y\mid x)=\operatorname*{lim}_{\epsilon\to0}P (Y\mid x-\epsilon\leq X\leq x+\epsilon).
$$ 
When does this limit exist? If there is a continuous joint density function $p (x, y)$ , then we can derive the form for this term. To do so, consider some event on $Y$ , say $a\leq Y\leq b$ . Recall that 

$$
\begin{array}{r c l}{P (a\leq Y\leq b\mid x-\epsilon\leq X\leq x+\epsilon)}&{=}&{\displaystyle\frac{P (a\leq Y\leq b, x-\epsilon\leq X\leq x+\epsilon)}{P (x-\epsilon\leq X\leq x+\epsilon)}}\\ &{=}&{\displaystyle\frac{\int_{a}^{b}\int_{x-\epsilon}^{x+\epsilon}p (x^{\prime}, y) d y d x^{\prime}}{\int_{x-\epsilon}^{x+\epsilon}p (x^{\prime}) d x^{\prime}}.}\end{array}
$$ 
When $\epsilon$ is sufciently small, we can approximate 

$$
\int_{x-\epsilon}^{x+\epsilon}p (x^{\prime}) d x^{\prime}\approx2\epsilon p (x).
$$ 
Using a similar approximation for $p (x^{\prime}, y)$ , we get 

$$
\begin{array}{r c l}{P (a\leq Y\leq b\mid x-\epsilon\leq X\leq x+\epsilon)}&{\approx}&{\displaystyle\frac{\int_{a}^{b}2\epsilon p (x, y) d y}{2\epsilon p (x)}}\\ &{=}&{\displaystyle\int_{a}^{b}\frac{p (x, y)}{p (x)}d y.}\end{array}
$$ 
We conclude that $\textstyle{\frac{p (x, y)}{p (x)}}$ is the density of $P (Y\mid X=x)$ . 

***Definition 2.9*** 
Let $p (x, y)$ be the joint density of $X$ and $Y$ . The conditional density function of $Y$ given $X$ is defined as 
> 条件概率密度函数

$$
p (y\mid x)={\frac{p (x, y)}{p (x)}}
$$ 
When $p (x)=0$ , the conditional density is undefined. 
> 若 $p (x)$，则条件密度没有定义

The conditional density $p (y\mid x)$ characterizes the conditional distribution $P (Y\mid X=x)$ we defined earlier. 

The properties of joint distributions and conditional distributions carry over to joint and conditional density functions. In particular, we have the chain rule 

$$
p (x, y)=p (x) p (y\mid x)
$$ 
and Bayes’ rule 

$$
p (x\mid y)={\frac{p (x) p (y\mid x)}{p (y)}}.
$$ 
As a general statement, whenever we discuss joint distributions of continuous random variables, we discuss properties with respect to the joint density function instead of the joint distribution, as we do in the case of discrete variables. Of particular interest is the notion of (conditional) independence of continuous random variables. 

***Definition 2.10***
Let $X, Y$ , and $Z$ be sets of continuous random variables with joint density $p (X, Y, Z)$ . We say that $X$ is conditionally independent of $Y$ given $Z$ if 
$p(x\mid z) = p(x\mid y, z)$ for all $x, y, z$ such that $p (z) > 0$
### 2.1.7 Expectation and Variance 
#### 2.1.7.1 Expectation 
Let $X$ be a discrete random variable that takes numerical values; then the expectation of $X$ under the distribution $P$ is 

$$
E_{P}[X]=\sum_{x}x\cdot P (x).
$$ 
If $X$ is a continuous variable, then we use the density function 

$$
E_{P}[X]=\int x\cdot p (x) d x.
$$ 
For example, if we consider $X$ to be the outcome of rolling a fair die with probability $1/6$ for each outcome, then $\begin{array}{l}{\displaystyle{{\cal E}[X]\,=\, 1\,\cdot\,\frac{1}{6}\,+\, 2\,\cdot\,\frac{1}{6}\,+\,\cdot\,\cdot\,+\, 6\,\cdot\,\frac{1}{6}\,=\, 3.5}}\end{array}$ · · · · · · . 
On the other hand, if we consider a biased die where $P (X\,=\, 6)\,=\, 0.5$ and $P (X\,=\, x)\,=\, 0.1$ for $x\,<\, 6$ , then $\pmb{{\cal E}}[X]=1\cdot0.1+\cdot\cdot\cdot+5\cdot0.1+\cdot\cdot\cdot+6\cdot0.5=4.5$ · · · · · · · · · . 

Often we are interested in expectations of a function of a random variable (or several random variables). Thus, we might consider extending the definition to consider the expectation of a functional term such as $X^{2}+0.5X$ . 
Note, however, that any function $g$ of a set of random variables $X_{1},\ldots, X_{k}$ is essentially random variable $Y$ : For any outcome $\omega\in\Omega$ , we define the value of Y as $g (f_{X_{1}}(\omega),.\,.\,.\,, f_{X_{k}}(\omega))$ . 
> 对于随机变量集合定义的任意函数实际上定义了一个新的随机变量

Based on this discussion, we often define new random variables by a functional term. For example $Y=X^{2}$ , or $Y=e^{X}$ . We can also consider functions that map values of one or more categorical random variables to numerical values. One such function that we use quite often is the indicator function , which we denote ${\pmb 1}\{{\pmb X}={\pmb x}\}$  . This function takes value 1 when $X=x$ , and 0 otherwise. 

In addition, we often consider expectations of functions of random variables without bothering to name the random variables they define. For example $\pmb{{\cal E}}_{P}[X+Y]$ . Nonetheless, we should keep in mind that such a term does refer to an expectation of a random variable. 

We now turn to examine properties of the expectation of a random variable. 

First, as can be easily seen, the expectation of a random variable is a linear function in that random variable. Thus, 

$$
E[a\cdot X+b]=a E[X]+b.
$$

> 求期望是线性运算

A more complex situation is when we consider the expectation of a function of several random variables that have some joint behavior. An important property of expectation is that the expectation of a sum of two random variables is the sum of the expectations. 

***Proposition 2.4*** 
linearity of expectation 

$$
E[X+Y]=E[X]+E[Y].
$$ 
This property is called linearity of expectation. 
> 期望的线性性质，即便随机变量相互独立也满足

It is important to stress that this identity is true even when the variables are not independent. As we will see, this property is key in simplifying many seemingly complex problems. 

Finally, what can we say about the expectation of a product of two random variables? In general, very little: 


Example 2.5 
*Consider two random variables $X$ and $Y$ , each of which takes the value $+1$ with probability $1/2$ , and the value $-1$ wi probability $1/2$ . If $X$ and $Y$ are independent, then $E[X\cdot Y]=0$  . On the other hand, if X and $Y$ are correlated in that they always take the same value, then $E[X\cdot Y]=1.$* 


However, when $X$ and $Y$ are independent, then, as in our example, we can compute the expectation simply as a product of their individual expectations: 

***Proposition 2.5*** 
If $X$ and $Y$ are independent, then 
> 要求随机变量相互独立

$$
E[X\cdot Y]=E[X]\cdot E[Y].
$$ 
We often also use the expectation given some evidence. The conditional expectation of $X$ given $y$ is 
> 条件期望：和式中乘上的是条件概率

$$
E_{P}[X\mid\pmb{y}]=\sum_{x}x\cdot P (x\mid\pmb{y}).
$$ 
#### 2.1.7.2 Variance 
The expectation of $X$ tells us the mean value of $X$ . However, It does not indicate how far $X$ deviates from this value. A measure of this deviation is the variance of $X$ . 

$$
{\pmb V}\! a r_{P}[X]={\pmb E}_{P}\Big[\big (X-{\pmb E}_{P}[X]\big)^{2}\Big].
$$ 
Thus, the variance is the expectation of the squared difference between $X$ and its expected value. 
> 方差：随机变量相对于期望的距离平方的期望

It gives us an indication of the spread of values of $X$ around the expected value. 

An alternative formulation of the variance is 

$$
{V}\! a r[X]=E\big[X^{2}\big]-\big (E[X]\big)^{2}\,.\tag{2.16}
$$ 
(see exercise 2.11). Similar to the expectation, we can consider the expectation of a functions of random variables. 
> 方差同样只是定义于随机变量的一个函数

***Proposition 2.6*** 
If $X$ and $Y$ are independent, then 

$$
{V}a r[X+Y]=Va r[X]+Va r[Y].
$$ 
It is straightforward to show that the variance scales as a quadratic function of $X$ . In particular, we have: 

$$
\mathbb{W}a r[a\cdot X+b]=a^{2}\mathbb{W}a r[X].
$$ 
For this reason, we are often interested in the square root of the variance, which is called the standard deviation of the random variable. We define 

$$
\sigma_{X}={\sqrt{\mathbb{W}a r[X]}}.
$$ 
The intuition is that it is improbable to encounter values of $X$ that are farther than several standard deviations from the expected value of $X$ . Thus, $\sigma_{X}$ is a normalized measure of “distance” from the expected value of $X$ . 
As an example consider the Gaussian distribution of definition 2.7. 


***Proposition 2.7*** 
Let $X$ be a random variable with Gaussian distribution $N (\mu,\sigma^{2})$ , then $E[X]=\mu$ and $\mathbb{W}a r[X]=$ $\sigma^{2}$ . 

Thus, the parameters of the Gaussian distribution specify the expectation and the variance of the distribution. As we can see from the form of the distribution, the density of values of $X$ drops exponentially fast in the distance $\textstyle{\frac{x-\mu}{\sigma}}$ . 

Not all distributions show such a rapid decline in the probability of outcomes that are distant from the expectation. However, even for arbitrary distributions, one can show that there is a decline. 

***Theorem 2.1*** 
Chebyshev’s inequality 

$$
P (|X-\pmb{E}_{P}[X]|\ge t)\le\frac{Va r_{P}[X]}{t^{2}}.
$$ 
We can restate this inequality in terms of standard deviations: We write $t=k\sigma_{X}$ to get 

$$
P (|X-\pmb{E}_{P}[X]|\ge k\sigma_{X})\le\frac{1}{k^{2}}.
$$ 
Thus, for example, the probability of $X$ being more than two standard deviations away from $E[X]$ is less than $1/4$ . 

## 2.2 Graphs 
Perhaps the most pervasive concept in this book is the representation of a probability distribution using a graph as a data structure. In this section, we survey some of the basic concepts in graph theory used in the book. 
### 2.2.1 Nodes and Edges 
***Definition 2.11*** 
A graph is a data structure $\mathcal{K}$ consisting of a set of nodes and a set of edges. Throughout most this book, we will assume that the set of nodes is $\mathcal{X}=\{X_{1},\cdot\cdot\cdot, X_{n}\}$ . A pair of nodes $X_{i}, X_{j}$ can be connected by a directed edge $X_{i}\rightarrow X_{j}$ or an undirected edge $X_{i}{-}X_{j}$ . 
Thus, the set of edge  $\mathcal{E}$ is a set of pairs, where each pair is one of $X_{i}\rightarrow X_{j}$ → , $X_{j}\,\rightarrow\, X_{i}$ → , or $X_{i}{-}X_{j}$ for $X_{i}, X_{j}\in{\mathcal{X}}$  , $i<j$ . 

We assume throughout the book tha pair $X_{i}, X_{j}$ , at most one type of edge exists, we cannot have bot $X_{i}\,\rightarrow\, X_{j}$ and $X_{j}\,\rightarrow\, X_{i}$ nor can we have $X_{i}\rightarrow X_{j}$ and $X_{i}{-}X_{j}$ . the notation $X_{i}\gets X_{j}$ equivalent to $X_{j}\rightarrow X_{i}$ , and notation $X_{j}{-}X_{i}$ is equivalent to $X_{i}{-}X_{j}$ . We use $X_{i}\rightleftharpoons X_{j}$ to represent the case where $X_{i}$ and $X_{j}$ are connected via some edge, whether directed (in any direction) or undirected. 
> 我们假设本书中两点之间有向边和无向边不会同时存在

In many cases, we want to restrict attention to graphs that contain only edges of one kind or another. We say that a graph is directed if all edges are either $X_{i}\rightarrow X_{j}$ or $X_{j}\rightarrow X_{i}$ We usually denote directed graphs as $\mathcal{G}$ . We say that a graph is undirected if all edges are $X_{i}{-}X_{j}$ . We denote undirected graphs as H . We sometimes convert a general graph to an undirected graph by ignoring the directions on the edges. 


***Definition 2.11***
Given a graph $\mathcal{K}=(\mathcal{X},\mathcal{E})$ , its undirected version is a graph $\mathcal{H}=(\mathcal{X},\mathcal{E}^{\prime})$ where ${\mathcal{E}}^{\prime}=\{X{-}Y\ :$ $X\rightleftharpoons Y\in{\mathcal{E}}\}$ . 
> 忽视方向将有向图转化为无向图

Whenever we have that $X_{i}\,\rightarrow\, X_{j}\,\in\,{\mathcal{E}}$ , we say that $X_{j}$ is the child of $X_{i}$ in $\mathcal{K}$ , and that $X_{i}$ parent of $X_{j}$ in K . When have $X_{i}{-}X_{j}\,\in\,{\mathcal{E}}$ ∈E , we say th $X_{i}$ ighbor of $X_{j}$ n $\mathcal{K}$ (and vice versa). We a X and Y are adjacent wh r $X\rightleftharpoons Y\in{\mathcal{E}}$ ∈E . We use $\mathrm{Pa}_{X}$ to denote the parents of X , $\operatorname{Ch}_{X}$ to denote its children, and $\mathrm{Nb}_{X}$ to denote its neighbors. We defi the boundary of $X$ , denoted Boun ry $\cdot_{X}$ , to be $\mathrm{Pa}_{X}\cup\mathrm{Nb}_{X}$ ; for DAGs, this set is simply X ’s parents, and for undirected graphs X ’s neighbors. Figure 2.3 shows an example of a graph $\mathcal{K}$ . There we ve that $A$ is the only parent $C$ $F, I$ are the children o $C$ . The only neighbor of C is D , but its adjacent nodes are $A, D, F, I$ . The degree of a node X is the number of edges in which it participates. Its indegree is the number of directed edges $Y\rightarrow X$ . The degree of a graph is the maximal degree of a node in the graph. 
> 定义点的 boundary 为它的父节点和邻居的并集
> 对于有向无环图，boundary 就是它的父节点集合
> 对于无向图，boundary 就是邻居节点集合
> 图的度是图中节点度的最大值
### 2.2.2 Subgraphs 
In many cases, we want to consider only the part of the graph that is associated with a particular subset of the nodes. 

***Definition 2.12*** 
Let $\mathcal{K}=(\mathcal{X},\mathcal{E})$ , and let $X\subset{\mathcal{X}}$ , we define the induced subgraph $\mathcal{K}[X]$ to be the graph $(X,{\mathcal{E}}^{\prime})$ where $\mathcal{E}^{\prime}$ are all the edges $X\rightleftharpoons Y\in{\mathcal{E}}$ such that $X, Y\in X$ . 
> 导出子图：有结点子集定义，导出子图内包含了结点子集内所有互相连接的边

For example, figure 2.4a shows the induced subgraph $\mathcal{K}[C, D, I]$ . A type of subgraph that is often of particular interest is one that contains all possible edges. 

***Definition 2.13*** 
A subgraph over $X$ is complete if every two nodes in $X$ are connected by some edge. The set $X$ often called a clique ; we say that a clique $X$ is maximal if for any superset of nodes $Y\supset X$ , $Y$ is not a clique. Although the subset of nodes $X$ can be arbitrary, we are often interested in sets of nodes that preserve certain aspects of the graph structure. 
> 在一个节点集上的子图中，任意节点两两相连，称该子图是完全的，且该节点集是团
> 如果团的超集中没有团，则团是最大的

***Definition 2.14*** 
We say that a subset of nodes $X \subset \mathcal X$ is upwardly closed in $\mathcal K$ if, for any $x\in X$, we have $\text{Boundary}_x \subset X$. We define the upward closure of $X$ to be the minimal upwardly closed subset $Y$ that contains $X$. We define the upwardly closed subgraph of $X$ , denoted $\mathcal{K}^{+}[X]$ , to be the induced subgraph over $Y$ , $\mathcal{K}[Y]$ . 
> 对于一个节点集，如果它其中的任意节点的边界都在该节点集内，则该节点集是向上封闭的
> 一个节点集的上闭包就是包含了该节点集并且是向上封闭的最小节点集
> 一个节点集的向上封闭子图就是该节点集的上闭包的导出子图

For example, the set $A, B, C, D, E$ is the ard closure of the set $\{C\}$ in $\mathcal{K}$ . The upwardly closed subgraph of $\{C\}$ is shown in figure 2.4b. The upwardly closed subgraph of $\{C, D, I\}$ is shown in figure 2.4c. 
### 2.2.3 Paths and Trails 
Using the basic notion of edges, we can define diferent types of longer-range connections in the graph

***Definition 2.15*** 
We say that X1; : : : ; Xk form a path in the graph K = (X; E) if, for every i = 1; : : : ; k - 1, path we have that either Xi ! Xi+1 or Xi—Xi+1. A path is directed if, for at least one i, we have Xi ! Xi+1.
> 一组连续的节点相互按照排序两两连接（有向或无向，有向即 $X_i \rightarrow X_{i+1}$, 无向即 $X_i - X_{i+1}$），则形成了路径
> 如果其中至少一对连接是有向连接，路径就是有向路径


***Definition 2.16***
We say that $X_1,\dots, X_k$ form a trail in the graph K = (X; E) if, for every $i=1,\dots, k-1$, wehave that Xi ⇋ Xi+1.
> 迹 trail 比路径的要求更加宽松，路径要求连续的节点之间 $X_1,\dots ,X_k$ 的两两连接是按照排序的，即必须 $X_i \rightarrow X_{i+1}$ 或者 $X_i - X_{i+1}$，而迹则不要求一定要从小到大，只需要 $X_i, X_{i+1}$ 之间存在连接即可

In the graph K of figure 2.3, A; C; D; E; I is a path, and hence also a trail. On the other hand, A; C; F; G; D is a trail, which is not a path.


***Definition 2.17*** 
A graph is connected if for every Xi; Xj there is a trail between Xi and Xj.
> 连通图：任意两个节点之间都存在迹

We can now define longer-range relationships in the graph.

***Definition 2.18*** 
We say that X is an ancestor of Y in K = (X; E), and that Y is a descendant of X, if there ancestor descendant exists a directed path X1; : : : ; Xk with X1 = X and Xk = Y . We use DescendantsX to denote X’s descendants, AncestorsX to denote X’s ancestors, and NonDescendantsX to denote the set of nodes in X - DescendantsX.
> 如果图中存在一条从 $X$ 到 $Y$ 的有向路径，则称 $X$ 是 $Y$ 的祖先，$Y$ 是 $X$ 的后代，$X$ 的非后代就是节点集减去它的后代集

In our example graph K, we have that F; G; I are descendants of C. The ancestors of C are A, via the path A; C, and B, via the path B; E; D; C.

A final useful notion is that of an ordering of the nodes in a directed graph that is consistent with the directionality its edges.
> 有向图中，节点之间的顺序和边的指向是一致的

***Definition 2.19*** 
Let G = (X; E) be a graph. An ordering of the nodes X1; : : : ; Xn is a topological ordering relative to K if, whenever we have Xi ! Xj 2 E, then i < j.
> 一个节点序列，只要满足任意出现在序列中的两个节点的先后顺序不违反图中边指向的顺序，这个序列就是一个拓扑序列

Appendix A.3.1 presents an algorithm for finding such a topological ordering.
### 2.2.4 Cycles and Loops 
Note that, in general, we can have a cyclic path that leads from a node to itself, making that node its own descendant. 
> 可以让节点指向自己，则自己是自己的后代

***Definition 2.20*** cycle
A cycle in K is a directed path $X_1, \dots, X_k$ where X1 = Xk. A graph is acyclic if it contains nocycles.
> 环：图中的一条有向路径的起点和终点相同，就构成了一个环，如果图没有环，就是无环图

A directed acyclic graph (DAG) is one of the central concepts in this book, as DAGs are the basic graphical representation that underlies Bayesian networks. For some of this book, we also use acyclic graphs that are partially directed. The graph K of figure 2.3 is acyclic. However, if we add the undirected edge A—E to K, we have a path A; C; D; E; A from A to itself. Clearly, adding a directed edge E ! A would also lead to a cycle. Note that prohibiting cycles does not imply that there is no trail from a node to itself. For example, K contains several trails: C; D; E; I; C as well as C; D; G; F; C.
> 没有环不代表没有指向节点自己的迹

An acyclic graph containing both directed and undirected edges is called a partially directed PDAG acyclic graph or PDAG. The acyclicity requirement on a PDAG implies that the graph can be chain component decomposed into a directed graph of chain components, where the nodes within each chain component are connected to each other only with undirected edges. The acyclicity of a PDAG guarantees us that we can order the components so that all edges point from lower-numbered components to higher-numbered ones.
> 既有有向边又有无向边的无环图称为部分有向无环图
> 部分有向无环图可以看作是一个由 chain components 构成的有向图，而 chain components 内部的边都是无向边
> 我们可以对 chain components 进行排序（根据边的关系）

***Definition 2.21*** 
Let K be a PDAG over X. Let $K_1,\dots, K_\ell$ be a disjoint partition of X such that:

• the induced subgraph over Ki contains no directed edges;
• for any pair of nodes X 2 Ki and Y 2 Kj for i < j, an edge between X and Y can only be a directed edge X ! Y .

> 对于一个部分有向无环图，对于它的节点集，对于该节点集的一个不相交的划分，满足：
> 划分成分导出的子图不包含有向边
> 任意一个来自于两个不同划分成分的节点对，如果二者之间有边，只能是按划分成分需要由低指向高的有向边

Each component Ki is called a chain component.
> 此时每个划分成分 $K_i$ 都称为 chain 成分

Because of its chain structure, a PDAG is also called a chain graph.
> 部分有向无环图也可以称为链图

Note that when the PDAG is an undirected graph, the entire graph forms a single chain component. Conversely, when the PDAG is a directed graph (and therefore acyclic), each node in the graph is its own chain component
> 部分有向无环图完全无向时，整个图就是一个 chain component
> 完全有向时，各个节点成为 chain component

***Definition 2.22*** loop
A loop in $\mathcal{K}$ is a trail $X_{1},\ldots, X_{k}$ where $X_{1}=X_{k}$ . A graph is singly connected if it contains no loops. A node in a singly connected graph is called $^a$ leaf if it has exactly one adjacent node. A singly connected directed graph is also called $^a$ polytree . A singly connected undirected graph is called $^a$ forest ; if it is also connected, it is called $^a$ tree . 
> loop：一个两端都是相同节点的迹称为回路
> 没有回路的图称为单连通图
> 单连通图内的一个节点只有一个邻接节点时，称为叶子
> 单连通有向图称为 polytree
> 单连通无向图称为森林，如果该图是连通的，称为树

We can also define a notion of a forest, or of a tree, for directed graphs. 

***Definition 2.24***
A directed graph is a forest if each node has at most one parent. A directed forest is a tree if it is also connected. 
> 对于有向图，如果每个节点最多一个父节点，则称为有向森林
> 如果连通，就是树

Note that polytrees are very diferent from trees. For example, figure 2.5 shows a graph that is a polytree but is not a tree, because several nodes have more than one parent. As we will discuss later in the book, loops in the graph increase the computational cost of various tasks. 
> 对于有向图，polytree 中一个节点可以有多个父节点，树中只能有一个

We conclude this section with a final definition relating to loops in the graph. This definition will play an important role in evaluating the cost of reasoning using graph-based representations. 

***Definition 2.24*** 
$X_{1}{\mathrm{-}}X_{2}{\mathrm{-}}\cdot\cdot\cdot{\mathrm{-}}X_{k}{\mathrm{-}}X_{1}$ be a loop in the graph; $^a$ chord in the l p is an edge connecting $X_{i}$ and $X_{j}$ for two nonconsecutive nodes $X_{i}, X_{j}$ . An undirected graph H is said to be chordal if any loop $X_{1}{\mathrm{-}}X_{2}{\mathrm{-}}\cdot\cdot\cdot{\mathrm{-}}X_{k}{\mathrm{-}}X_{1}$ for $k\geq4$ has a chord. 
> 在一个回路中，一个 chord 是一个连接了回路中两个不相邻节点的边
> 如果一个无向图中的任意回路在>=4个不同节点时就一定有 chord，就是 chordal graph

Thus, for example, a loop $\mathit{A-B-C-D-A}$ (as in figure 1.1b) is nonchordal, but adding an edge $A{-}C$ would render it chordal. In other words, in a chordal graph, the longest “minimal loop” (one that has no shortcut) is a triangle. Thus, chordal graphs are often also called triangulated . 

We can extend the notion of chordal graphs to graphs that contain directed edges. 

***Definition 2.25***
A graph K is said to be chordal if its underlying undirected graph is chordal.
> 对于一个有向图，如果转化为无向图是 chordal，则它是 chordal

## 2.4 Exercises
**Exercise 2.5**
Let $\pmb X, \pmb Y, \pmb Z$ be three disjoint set of variables such that $\mathcal X = \pmb X \cup \pmb Y \cup \pmb Z$，Prove that $P\vDash (\pmb X \perp \pmb Y\mid \pmb Z)$ if and only if we can write $P$ in the form:

$$
P(\mathcal X) = \phi_1(\pmb X, \pmb Z)\phi_2(\pmb Y, \pmb Z)
$$

证明：
(1) $P\vDash (\pmb X \perp \pmb Y \mid \pmb Z)\Rightarrow P(\mathcal X) = \phi_1(\pmb X, \pmb Z)\phi_2(\pmb Y, \pmb Z)$
$P\vDash (\pmb X \perp \pmb Y \mid \pmb Z)$ 表明了 

$$
\begin{align}
P(\mathcal X) &= P(\pmb X, \pmb Y, \pmb Z)\\
&=P(\pmb X, \pmb Y\mid \pmb Z)P(\pmb Z)\\
&=P(\pmb X\mid \pmb Z)P(\pmb Y\mid \pmb Z)P(\pmb Z)\\
&=\phi_1(\pmb X, \pmb Z)\phi_2(\pmb Y, \pmb Z)P(\pmb Z)
\end{align}
$$

因为 $P (\pmb Z)$ 仅和 $\pmb Z$ 有关，它可以作为被吸收入 $\phi_1 (\pmb X, \pmb Z)$ 或 $\phi_2 (\pmb Y, \pmb Z)$，因此
$P(\mathcal X) = \phi_1(\pmb X, \pmb Z)\phi_2(\pmb Y, \pmb Z)$

(2) $P(\mathcal X) = \phi_1(\pmb X, \pmb Z)\phi_2(\pmb Y, \pmb Z)\Rightarrow P\vDash (\pmb X \perp \pmb Y \mid \pmb Z)$

$$
\begin{align}
P(\pmb X,\pmb Y\mid \pmb Z) &= \frac {P(\pmb X, \pmb Y, \pmb Z)}{P(\pmb Z)}\\
&=\frac {\phi_1(\pmb X,\pmb Z)\phi_2(\pmb Y, \pmb Z)}{P(\pmb Z)}\\
\end{align}
$$

显然等式右边可以分解为两个项的乘积，第一个项仅决定于 $\pmb X,\pmb Z$，第二个项仅决定于 $\pmb Y, \pmb Z$，也就是 $P (\pmb X, \pmb Y \mid \pmb Z) = P (\pmb X\mid \pmb Z) P (\pmb Y \mid \pmb Z)$，即 $P\vDash (\pmb X \perp \pmb Y \mid \pmb Z)$

# Part 1 Representation
# 3 The Baysian Network Representation
Our goal is to represent a joint distribution $P$ over some set of random variables $\mathcal X=\{X_1,\dots, X_n\}$. Even in the simplest case where these variables are binary-valued, a joint distribution requires the specification of $2^{n-1}$ numbers — the probabilities of the $2^n$ different assignments of values $x_1,\dots, x_n$. 
> 目标是表示出包含 $n$ 个变量的联合分布 $P$

For all but the smallest $n$, the explicit representation of the joint distribution is unmanageable from every perspective. Computationally, it is very expensive to manipulate and generally too large to store in memory. Cognitively, it is impossible to acquire so many numbers from a human expert; moreover, the numbers are very small and do not correspond to events that people can reasonably contemplate. Statistically, if we want to learn the distribution from data, we would need ridiculously large amounts of data to estimate this many parameters robustly. These problems were the main barrier to the adoption of probabilistic methods for expert systems until the development of the methodologies described in this book.
> 由于指数爆炸问题，不能直接表示每个事件的概率

In this chapter, we first show how independence properties in the distribution can be used to represent such high-dimensional distributions much more compactly. We then show how a combinatorial data structure — a directed acyclic graph — can provide us with a general purpose modeling language for exploiting this type of structure in our representation.
> 本章先讨论如何使用分布中的独立性质来紧凑表示高维分布
> 然后讨论有向无环图的使用 
## 3.1 Exploiting Independence Properties
The compact representations we explore in this chapter are based on two key ideas: the representation of independence properties of the distribution, and the use of an alternative parameterization that allows us to exploit these finer-grained independencies.
> 本章讨论的紧凑表示基于两个关键思想：
> 分布的独立表示性质、使用参数化以利用这些细粒度的独立性
### 3.1.1 Independent Random Variables 
To motivate our discussion, consider a simple setting where we know that each $X_{i}$ represents the outcome of a toss of coin $i$. In this case, we typically assume that the diferent coin tosses are marginally independent (definition 2.4), so that our distribution $P$ will satisfy $\left(X_{i}\perp X_{j}\right)$ for any $i, j$ . More generally (strictly more generally — see exercise 3.1), we assume that the distribution satisfies $(X\perp Y)$ for any disjoint subsets of the variables $X$ and $Y$. Therefore, we have that: 
> 我们让每个 $X_i$ 表示投掷硬币 $i$ 的结果，分布 $P$ 满足对于任意的 $i, j$， $X_i, X_j$ 之间边际独立
> 同时假设分布满足 $X\perp Y$

$$
P(X_{1},\cdot\cdot\cdot,X_{n})=P(X_{1})P(X_{2})\cdot\cdot\cdot P(X_{n}).
$$ 
If we use the standard parameterization of the joint distribution, this independence structure is obscured, and the representation of the distribution requires $2^{n}$ parameters. 
> 这个联合分布包含了 $2^n$ 个可能的事件，显然不能用 $2^n$ 个参数建模每个事件的概率

However, we can use a more natural set of parameters for specifying this distribution: If $\theta_{i}$ is the probability with which coin $i$ lands heads, the joint distribution $P$ can be specified using the $n$ parameters $\theta_{1},\ldots,\theta_{n}$ . These parameters implicitly specify the $2^{n}$ probabilities in the joint distribution. For example, the pro that all coin t eads is $\theta_{1}\cdot\theta_{2}\cdot...\cdot\theta_{n}$ . More generally, letting $\theta_{x_{i}}=\theta_{i}$ when $x_{i}=x_{i}^{1}$ and $\theta_{x_{i}}=1-\theta_{i}$ when $x_{i}=x_{i}^{0}$ , we can define: 
> 我们用一个参数就可以表示每个随机变量的两种取值的概率，因此用 $n$ 个参数就可以表示整个联合分布

$$
P(x_{1},.\,.\,.\,,x_{n})=\prod_{i}\theta_{x_{i}}.\tag{3.1}
$$ 
This representation is limited, and there are many distributions that we cannot capture by choosing values for $\theta_{1},\ldots,\theta_{n}$ . This fact is obvious not only from intuition, but also from a somewhat m al perspe.
The space of all $2^{n}-1$ dimensional subspace of I $I\!\!R^{2^{n}}\ -$ — the set { $\{(p_{1},.\,.\,,\dot{p_{2^{n}}})\,\in\,I\!\!R^{2^{n^{\prime}}}\,:\,\,p_{1}+.\,.\,.+p_{2^{n}}\,=\,1\}$ . On the other hand, the space of all joint distributions specified in a factorized way as in equation (3.1) is an $n$ -dimensional manifold in ${I\!\!R}^{2^{n}}$ . 
> 所有联合分布的空间是 $\mathbb R^{2^n}$ 的一个 $2^n -1$ 维子空间，而使用 $n$ 个参数可以建模的联合分布的空间仅是 $\mathbb R^{2^n}$ 空间的一个 $n$ 维流形，因此这类表示也是受限的

A key concept here is the notion of independent parameters — parameters whose values are not determined by others. For example, when specifying an arbitrary multinomial distribution over a $k$ dimensional sp e have $k-1$ independent parameters: the last probability is fully determined by the first $k-1$ − . In the case where we have an arbitrary istribution over $n$ binary random variables, the number of independent parameters is 2 $2^{n}\mathrm{~-~}1$ − . On the other hand, the number of independent parameters for distributions represented as $n$ independent binomial coin tosses is $n$ . Therefore, the two spaces of distributions cannot be the same. (While this argument might seem trivial in this simple case, it turns out to be an important tool for comparing the expressive power of diferent representations.) 
> 这里涉及的一个概念是独立参数，即值不依赖于其他参数的参数
> 例如，指定 $k$ 维空间的任意一个多项式分布需要 $k-1$ 个独立参数，表示最后一个概率的参数由前面的 $k-1$ 个参数决定
> 本例中，$n$ 个 binary 随机变量的联合分布空间需要 $2^n - 1$ 个独立参数，而 $n$ 个相互独立的伯努利试验仅需要 $n$ 个独立参数建模联合分布，因此这两个联合分布空间是不可能相同的
> 独立参数的数量可以用于比较两个表示的表示能力

As this simple example shows, certain families of distributions — in this case, the distributions generated by $n$ independent random variables — permit an alternative parameter iz ation that is substantially more compact than the naive representation as an explicit joint distribution. Of course, in most real-world applications, the random variables are not marginally independent. However, a generalization of this approach will be the basis for our solution. 
> 本例展示了对于特定的分布族可以用一个使用更少独立参数的表示替代原有的朴素且需要大量参数的表示，这就是我们的基本思想
### 3.1.2 The Conditional Parameterization 
Let us begin with a simple example that illustrates the basic intuition. Consider the problem faced by a company trying to hire a recent college graduate. The company’s goal is to hire intelligent employees, but there is no way to test intelligence directly. However, the company has access to the student’s SAT scores, which are informative but not fully indicative. 
Thus, our probability space is induced by the two random variables Intelligence $(I)$ and $S A T\ (S)$ . For simplicity, we assume that e of these takes two va s: $V a l(I)=\{i^{1},i^{0}\}$ present the values high intelligence ( $(i^{1})$ ) and low intelligence ( $(i^{0})$ ); similarly V $V a l(S)\,=\,\{s^{1},s^{0}\}$ , which also represent the values high (score) and low (score), respectively. 
> 我们的概率空间由两个随机变量 $I, S$ 导出 

Thus, our joint distribution in this case has four entries. For example, one possible joint distribution $P$ would be 
> 图见书，这里列出了联合分布 $P$ 中的所有可能结果及其概率

There is, however, an alternative, and even more natural way of representing the same joint distribution. Using the chain rule of conditional probabilities (see equation (2.5)), we have that 

$$
P(I,S)=P(I)P(S\mid I).
$$ 
Intuitively, we are representing the process in a way that is more compatible with causality. Var- ious factors (genetics, upbringing, . . . ) first determined (stochastically) the student’s intelligence. His performance on the SAT is determined (stochastically) by his intelligence. We note that the models we construct are not required to follow causal intuitions, but they often do. We return to this issue later on. 
> 用条件概率分布表示该联合概率分布，我们采用了一个更符合因果关系的表示

From a mathematical perspective, this equation leads to the following alternative way of representing the joint distribution. Instead of specifying the various joint entries $P(I,S)$ , we would specify it in the form of $P(I)$ and $P(S\mid I)$ . Thus, for example, we can represent the joint distribution of equation (3.2) using the following two tables, one representing the prior distribution over $I$ and the other the conditional probability distribution (CPD) of $S$ given $I$ : 
> 建模了因果关系之后，我们不再用一张大表表示整个联合分布，而是用了两个表，一个表表示 $I$ 上的先验分布，另一个表表示给定 $I$ 时，$S$ 的条件概率分布

The CPD $P(S\mid I)$ represents the probability that the student will succeed on his SATs in the two possible cases: the case where the student’s intelligence is low, and the case where it is high. The CPD asserts that a student of low intelligence is extremely unlikely to get a high SAT score $(P(s^{1}\mid i^{0})=0.05)$ ; on the other hand, a student of high intelligence is likely, but far from certain, to get a high SAT score $(P(s^{1}\mid i^{1})=0.8)$ . 

It is instructive to consider how we could parameterize this alternative representation. Here, we are using three binomial distributions, one for $P(I)$ , and two for $P(S\mid i^{0})$ and $P(S\mid i^{1})$ . Hence, we can parameterize this representation using three independent parameters, say $\theta_{i^{1}}$ , $\theta_{s^{1}|i^{1}}$ , and $\theta_{s^{1}|i^{0}}$ . Our representation of the joint distribution as a four-outcome multinomial also required three parameters. Thus, although the conditional representation is more natural than the explicit representation of the joint, it is not more compact. 
However, as we will soon see, the conditional parameterization provides a basis for our compact representations of more complex distributions. 
> 对于这个表示的参数化，我们使用了三个二项分布：$P (I), P (S\mid i^0), P (S\mid i^1)$，因此使用三个参数 $\theta_{i^1}, \theta_{s^1\mid i^1}, \theta_{s^1\mid i^1}$

Although we will only define Bayesian networks formally in section 3.2.2, it is instructive to see how this example would be represented as one. The Bayesian network, as shown in figure 3.1a, would have a node for each of the two random variables $I$ and $S$ , with an edge from $I$ to $S$ representing the direction of the dependence in this model. 
### 3.1.3 The Naive Bayes Model 
We now describe perhaps the simplest example where a conditional parameter iz ation is com- bined with conditional independence assumptions to produce a very compact representation of a high-dimensional probability distribution. Importantly, unlike the previous example of fully independent random variables, none of the variables in this distribution are (marginally) independent. 
> 本节描述如何结合条件参数化和条件独立假设以紧凑表示一个高维概率分布
#### 3.1.3.1 The Student Example 
Elaborating our example, we now assume that the company also has access to the student’s grade $G$ in some course. In this case, our probability space is the joint distribution over the three relevant random variables $I,\,S,$ , and $G$ . Assuming that $I$ and $S$ are as before, and that $G$ takes on three values $g^{1},g^{2},g^{3}$ , representing the grades $A,\,B$ , and $C$ , respectively, then the joint distribution has twelve entries. 

Before we even consider the speciﬁc numerical aspects of our distribution $P$ in this example, we can see that independence does not help us: for any reasonable $P$ , there are no indepen- dencies that hold. The student’s intelligence is clearly correlated both with his SAT score and with his grade. The SAT score and grade are also not independent: if we condition on the fact that the student received a high score on his SAT, the chances that he gets a high grade in his class are also likely to increase. Thus, we may assume that, for our particular distribution $P$ , $P(g^{1}\mid s^{1})>P(g^{1}\mid s^{0})$ . 
> 本例中的随机变量 $I, G, S$ 显然不满足独立性，但可以假设满足某种条件独立性

However, it is quite plausible that our distribution $P$ in this case satisﬁes a conditional independence property. If we know that the student has high intelligence, a high grade on the SAT no longer gives us information about the student’s performance in the class. More formally: 
> 已知 $i^1$ 之后，$s^1$ 就不会给出更多关于 $g$ 的信息

$$
P(g\mid i^{1},s^{1})=P(g\mid i^{1}).
$$ 
More generally, we may well assume that 

$$
P\vDash(S\perp G\mid I).\tag{3.4}
$$ 
Note that this independence statement holds only if we assume that the student’s intelligence is the only reason why his grade and SAT score might be correlated. In other words, it assumes that there are no correlations due to other factors, such as the student’s ability to take timed exams. These assumptions are also not “true” in any formal sense of the word, and they are often only approximations of our true beliefs. (See box 3. C for some further discussion.) 
> 该条件独立性的假设是 $I$ 是 $S$ 和 $G$ 在 $P$ 中展现出高度相关的唯一原因，即假设了 $S$ 和 $G$ 没有由于其他原因而出现的相关性

As in the case of marginal independence, conditional independence allows us to provide a compact speciﬁcation of the joint distribution. Again, the compact representation is based on a very natural alternative parameter iz ation. By simple probabilistic reasoning (as in equation (2.5)), we have that 

$$
P(I,S,G)=P(S,G\mid I)P(I).
$$ 
But now, the conditional independence assumption of equation (3.4) implies that 

$$
P(S,G\mid I)=P(S\mid I)P(G\mid I).
$$ 
Hence, we have that 

$$
P(I,S,G)=P(S\mid I)P(G\mid I)P(I).\tag{3.5}
$$

> 利用条件独立性，我们将联合分布再次表示成条件概率分布的乘积的形式

Thus, we have factorized the joint distribution $P(I,S,G)$ as a product of three conditional probability distributions (CPDs). This factorization immediately leads us to the desired alternative parameter iz ation. In order to specify fully a joint distribution satisfying equation (3.4), we need the following three CPDs: $P(I),\,P(S\mid I),$ , and $P(G\mid I)$ . The ﬁrst two might be the same as in equation (3.3). The latter might be 
> 之后，我们对分布 $P (I), P (S\mid I), P (G\mid I)$ 进行参数化

$$
\frac{I\;\;\parallel\;\;g^{1}\;\;\;\;\;g^{2}\;\;\;\;\;g^{3}}{i^{0}\;\parallel\;0.2\;\;\;\;0.34\;\;\;\;0.46\;\;}
$$ 
Together, these three CPDs fully specify the joint distribution (assuming the conditional inde- pendence of equation (3.4)). For example, 

$$
\begin{array}{l c l}{P(i^{1},s^{1},g^{2})}&{=}&{P(i^{1})P(s^{1}\mid i^{1})P(g^{2}\mid i^{1})}\\ &{=}&{0.3\cdot0.8\cdot0.17=0.0408.}\end{array}
$$ 
Once again, we note that this probabilistic model would be represented using the Bayesian network shown in ﬁgure 3.1b. 

In this case, the alternative parameter iz ation is more compact than the joint. We now have three binomial distributions — $P(I)$ , $P(S\mid i^{1})$ and $P(S\mid i^{0})$ , and two three-valued multino- mial distributions — $P(G\mid i^{1})$ and $P(G\mid i^{0})$ . Each of the binomials requires one independent parameter, and each three-valued multinomial requires two independent parameters, for a total of seven. By contrast, our joint distribution has twelve entries, so that eleven independent parameters are required to specify an arbitrary joint distribution over these three variables. 

It is important to note another advantage of this way of representing the joint: modularity. When we added the new variable $G$ , the joint distribution changed entirely. Had we used the explicit representation of the joint, we would have had to write down twelve new numbers. In the factored representation, we could reuse our local probability models for the variables $I$ and $S$ , and specify only the probability model for $G-$ the CPD $P(G\mid I)$ . This property will turn out to be invaluable in modeling real-world systems. 
> 这类表示还有一个优势就是模块性，当加入新的变量 $G$ 之后，虽然联合分布完全改变了，但我们仍可以复用一部分的局部概率分布，例如 $P (I)$
#### 3.1.3.2 The General Model 
![[Probabilistic Graph Models-Fig3.2.png]]

This example is an instance of a much more general model commonly called the naive Bayes model (also known as the Idiot Bayes model ). The naive Bayes model assumes that instances fall into one of a number of mutually exclusive and exhaustive classes . Thus, we have a class variable $C$ that takes on va es in some set $\{c^{1},\cdot\cdot\cdot,c^{k}\}$ . In our example, the class variable is the student’s intelligence I , and there are two classes of instances — students with high intelligence and students with low intelligence. 
> 上例使用的就是朴素贝叶斯模型，朴素贝叶斯模型假设实例落入一个互斥且完备的类集合中的一个类，记为一个类变量 $C$ 从 $\{c^1,\dots, c^k\}$ 中取值
> 上例中，类变量就是 $I$，有两类实例：IQ 高的学生和 IQ低的学生

The model also includes some number of features $X_{1},\dots,X_{n}$ whose values are typically observed. The naive Bayes assumption is that the features are conditionally independent given the instance’s class. In other words, within each class of instances, the diferent properties can be determined independently. 
> 朴素贝叶斯模型还包含了一定数量的特征 $X_1,\dots, X_n$，特征的值是被观测到的
> 朴素贝叶斯假设特征在给定了实例的类别的情况下是条件独立的，也就是在每类实例中，不同的属性都是独立决定的

More formally, we have that 

$$
(X_{i}\perp \pmb X_{-i}\mid C)\quad{\mathrm{~for~all~}}i,\tag{3.6}
$$ 
where $\pmb X_{-i}\;=\;\{X_{1},.\,.\,.\,,X_{n}\}\,-\,\{X_{i}\}$ . 

This model can be represented using the Bayesian network of ﬁgure 3.2. In this example, and later on in the book, we use a darker oval to represent variables that are always observed when the network is used. 

Based on these independence assumptions, we can show that the model factorizes as: 

$$
P(C,X_{1},\dots,X_{n})=P(C)\prod_{i=1}^{n}P(X_{i}\mid C).\tag{3.7}
$$ 
(See exercise 3.2.) Thus, in this model, we can represent the joint distribution using a small set of factors: a prior distribution $P(C)$ , specifying how likely an instance is to belong to diferent classes a priori, and a set of CPDs $P(X_{j}\mid C)$ , one for each of the $n$ ﬁnding variables. 
> 基于这些假设，我们将联合分布分解如上
> 因此，在朴素贝叶斯模型中，我们用一个先验分布 $P (C)$ (表示实例属于哪个类别的概率)，以及一系列条件概率分布 $P (X_j\mid C)$ 表示联合分布

These factors can be encoded using a very small number of parameters. For example, if all of the variables are binary, the number of independent parameters required to specify the distribution is $2n+1$ (see exercise 3.6). Thus, the number of parameters is linear in the number of variables, as opposed to exponential for the explicit representation of the joint. 
> 这种表示下，参数的数量和随机变量的数量是呈线性关系的
> 显式表示联合分布中，则是呈指数关系

**Box 3. A — Concept: The Naive Bayes Model.** 
The naive Bayes model, despite the strong as- sumptions that it makes, is often used in practice, because of its simplicity and the small number of parameters required. The model is generally used for classiﬁcation — deciding, based on the values of the evidence variables for $a$ given instance, the class to which the instance is most likely to belong.  We might also want to compute our conﬁdence in this decision, that is, the extent to which our model favors one class $c^{1}$ over another $c^{2}$ .
> 朴素贝叶斯模型常用于分类：给定一个实例的许多特征变量的值，决定它的类别
> 同时可以计算该决定相对于其他决定的置信度比例

 Both queries can be addressed by the following ratio: 

$$
{\frac{P(C=c^{1}\mid x_{1},\ldots,x_{n})}{P(C=c^{2}\mid x_{1},\ldots,x_{n})}}={\frac{P(C=c^{1})}{P(C=c^{2})}}\prod_{i=1}^{n}{\frac{P(x_{i}\mid C=c^{1})}{P(x_{i}\mid C=c^{2})}};\tag{3.8}
$$ 
see exercise 3.2). This formula is very natural, since it computes the posterior probability ratio of $c^{1}$ versus $c^{2}$ as a product of their prior probability ratio (the first term), multiplied by a set of terms $\scriptstyle{\frac{P(x_{i}|C=c^{1})}{P(x_{i}|C=c^{2})}}$ that measure the relative support of the finding $x_{i}$ for the two classes. 

This model was used in the early days of medical diagnosis , where the diferent values of the class variable represented diferent diseases that the patient could have. The evidence variables represented diferent symptoms, test results, and the like. Note that the model makes several strong assumptions that are not generally true, speciﬁcally that the patient can have at most one disease, and that, given the patient’s disease, the presence or absence of diferent symptoms, and the values of diferent tests, are all independent. This model was used for medical diagnosis because the small number of interpretable parameters made it easy to elicit from experts. For example, it is quite natural to ask of an expert physician what the probability is that a patient with pneumonia has high fever. Indeed, several early medical diagnosis systems were based on this technology, and some were shown to provide better diagnoses than those made by expert physicians. 

However, later experience showed that the strong assumptions underlying this model decrease its diagnostic accuracy. In particular, the model tends to overestimate the impact of certain evidence by “overcounting” it. For example, both hypertension (high blood pressure) and obesity are strong indicators of heart disease. However, because these two symptoms are themselves highly correlated, equation (3.8), which contains a multiplicative term for each of them, double-counts the evidence they provide about the disease. Indeed, some studies showed that the diagnostic performance of a naive Bayes model degraded as the number of features increased; this degradation was often traced to violations of the strong conditional independence assumption. This phenomenon led to the use of more complex Bayesian networks, with more realistic independence assumptions, for this application (see box 3. D). 
> 条件独立性假设在实际中可能产生的问题是高估某个证据的影响，例如实际中，在类别 $C$ 下，特征 $X_i, X_j$ 实际上是高度相关的 (交集大)，因此 $P (X_i, X_j\mid C)$ 会小于 $P (X_i\mid C) P (X_j\mid C)$，此时特征 $X_i, X_j$ 的出现会过度支持类别 $C$
> 随着变量的数量增大，朴素贝叶斯效果往往下降，往往因为对于条件独立性的违反变得更加严重了

Nevertheless, the naive Bayes model is still useful in a variety of applications, particularly in the context of models learned from data in domains with a large number of features and a rela- tively small number of instances, such as classifying documents into topics using the words in the documents as features; see box 17. E). 
## 3.2 Bayesian Networks 
Bayesian networks build on the same intuitions as the naive Bayes model by exploiting conditional independence properties of the distribution in order to allow a compact and natural representation. However, they are not restricted to representing distributions satisfying the strong independence assumptions implicit in the naive Bayes model. They allow us the ﬂexibility to tailor our representation of the distribution to the independence properties that appear reasonable in the current setting. 
> 贝叶斯网络的假设不像朴素贝叶斯一样简单，我们可以灵活定义贝叶斯网络中的条件独立性

The core of the Bayesian network representation is a directed acyclic graph (DAG) $\mathcal{G}$ , whose nodes are the random variables in our domain and whose edges correspond, intuitively, to direct inﬂuence of one node on another. This graph $\mathcal{G}$ can be viewed in two very diferent ways: 
> 贝叶斯网络使用 DAG 表示，节点是随机变量，边表示变量之间的影响关联
> 网络用分解的方式表示了联合分布，以及表示了分布之间的条件独立性

- as a data structure that provides the skeleton for representing a joint distribution compactly in a factorized way; 
- as a compact representation for a set of conditional independence assumptions about a distribution. 

As we will see, these two views are, in a strong sense, equivalent. 
### 3.2.1 The Student Example Revisited 
We begin our discussion with a simple toy example, which will accompany us, in various versions, throughout much of this book. 

![[Probabilistic Graph Theory-Fig3.3.png]]

#### 3.2.1.1 The Model 
Consider our student from before, but now consider a slightly more complex scenario. The student’s grade, in this case, depends not only on his intelligence but also on the difculty of the course, represented by a random variable $D$ whose domain is $V a l(D)=\{e a s y,h a r d\}$ . Our student asks his professor for a recommendation letter. The professor is absentminded and never remembers the names of her students. She can only look at his grade, and she writes her letter for him based on that information alone. The quality of her letter is a random variable $L$ , whose domain is $V a l(L)=\left\{s t r o n g,w e a k\right\}$ . The actual quality of the letter depends stochastically on the grade. (It can vary depending on how stressed the professor is and the quality of the cofee she had that morning.) 

We therefore have ﬁve random variables in this domain: the student’s intelligence $(I)_{i}$ , the course difculty $(D)$ , the grade $(G)$ , the student’s SAT score $(S)$ , and the quality of the recom- mendation letter $(L)$ . All of the variables except $G$ are binary-valued, and $G$ is ternary-valued. Hence, the joint distribution has 48 entries. 

As we saw in our simple illustrations of ﬁgure 3.1, a Bayesian network is represented using a directed graph whose nodes represent the random variables and whose edges represent direct inﬂuence of one variable on another. We can view the graph as encoding a generative sampling process executed by nature, where the value for each variable is selected by nature using a distribution that depends only on its parents. In other words, each variable is a stochastic function of its parents. 
> 我们将被贝叶斯网络视为它编码了自然执行的生成过程，其中每个随机变量的值都由自然通过一个分布选择，这个分布决定于该变量的父节点，即每个变量都是它的父节点的一个随机函数

Based on this intuition, perhaps the most natural network structure for the distribution in this example is the one presented in ﬁgure 3.3. The edges encode our intuition about the way the world works. The course difculty and the student’s intelligence are determined independently, and before any of the variables in the model. The student’s grade depends on both of these factors. The student’s SAT score depends only on his intelligence. The quality of the professor’s recommendation letter depends (by assumption) only on the student’s grade in the class. Intuitively, each variable in the model depends directly only on its parents in the network. We formalize this intuition later. 

The second component of the Bayesian network representation is a set of local probability models that represent the nature of the dependence of each variable on its parents. One such model, $P(I)$ , represents the distribution in the population of intelligent versus less intelligent student. Another, $P(D)$ , represents the distribution of difcult and easy classes. The distribution over the student’s grade is a conditional distribution $P(G\mid I,D)$ . It speciﬁes the distribution over the student’s grade, inasmuch as it depends on the student’s intelligence and the course difculty. Speciﬁcally, we would have a diferent distribution for each assignment of values $i,d$ . For example, we might believe that a smart student in an easy class is 90 percent likely to get an A, 8 percent likely to get a B, and 2 percent likely to get a C. Conversely, a smart student in a hard class may only be 50 percent likely to get an A. 
In general, each variable $X$ in the model is associated with a conditional probability distribution (CPD) that speciﬁes a distribution over the values of $X$ given each possible joint assignment of values to its parents in the model. For a node with no parents, the CPD is conditioned on the empty set of variables. Hence, the CPD turns into a marginal distribution, such as $P(D)$ or $P(I)$ . One possible choice of CPDs for this domain is shown in ﬁgure 3.4. The network structure together with its CPDs is a Bayesian network $\mathcal{B}$ ; we use $\mathcal{B}^{s t u d e n t}$ to refer to the Bayesian network for our student example. 
> 贝叶斯网络表示的第二个成分是一系列局部概率模型，它们表示了各个变量和其父变量之间的依赖关系
> 贝叶斯网络中的每个变量都从属于一个条件概率分布，每个父变量的变量条件于空集，因此其条件概率分布就是边际分布

How do we use this data structure to specify the joint distribution? Consider some particular state in this space, for example, $i^{1},d^{0},g^{2},s^{1},l^{0}$ . Intuitively, the probability of this event can be computed from the probabilities of the basic events that comprise it: the probability that the student is intelligent; the probability that the course is easy; the probability that a smart student gets a B in an easy class; the probability that a smart student gets a high score on his SAT; and the probability that a student who got a B in the class gets a weak letter. The total probability of this state is: 

$$
\begin{array}{l l l}{{P(i^{1},d^{0},g^{2},s^{1},l^{0})}}&{{=}}&{{P(i^{1})P(d^{0})P(g^{2}\mid i^{1},d^{0})P(s^{1}\mid i^{1})P(l^{0}\mid g^{2})}}\\ {{}}&{{=}}&{{0.3\cdot0.6\cdot0.08\cdot0.8\cdot0.4=0.004608.}}\end{array}
$$ 
Clearly, we can use the same process for any state in the joint probability space. In general, we will have that 

$$
P(I,D,G,S,L)=P(I)P(D)P(G\mid I,D)P(S\mid I)P(L\mid G).\tag{3.9}
$$ 
chain rule for Bayesian networks 

This equation is our ﬁrst example of the chain rule for Bayesian networks which we will deﬁne in a general setting in section 3.2.3.2. 
#### 3.2.1.2 Reasoning Patterns 
A joint distribution $P_{\mathcal{B}}$ speciﬁes (albeit implicitly) the probability $P_{\mathcal B}(Y\,=\,y\;\mid\;E\,=\,e)$ of any event $y$ given any observations $e$ , as discussed in section 2.1.3.3: We condition the joint distribution on the event $E=e$ by eliminating the entries in the joint inconsistent with our observation $e$ , and renormalizing the resulting entries to sum to 1; we compute the probability of the event $y$ by summing the probabilities of all of the entries in the resulting posterior distribution that are consistent with $y$ .
> 联合分布对于任何事件指定了条件概率分布 $P (Y = y \mid E = e)$，我们通过消除和观察 $e$ 不一致的项，然后将剩下的项规范化到和为 1，来条件于 $E = e$，通过对于剩下的项中全部和 $y$ 一致的项求和，得到事件 $y$ 的概率

To illustrate this process, let us consider our B student network and see how the probabilities of various events change as evidence is obtained. 

Consider a particular student, George, about whom we would like to reason using our model. We might ask how likely George is to get a strong recommendation $(l^{1})$ from his professor in Econ101. Knowing nothing else about George or Econ101, this probability is about 50.2 percent. More precisely, let $P_{\mathcal{B}^{s t u d e n t}}$ be the joint distribution deﬁned by the preceding BN; then we have that $P_{\mathcal{B}^{s t u d e n t}}(l^{1})\approx0.502$ . We now ﬁnd out that George is not so intelligent $(i^{0})$ ; the probability that he gets a strong letter from the professor of Econ101 goes down to around 38.9 percent; that is, $P_{\mathcal{B}^{s t u d e n t}}(l^{1}\mid i^{0})\approx0.389.$ . We now further discover that Econ101 is an easy class $(d^{0})$ . The probability that George gets a strong letter from the professor is now $P_{\mathcal{B}^{s t u d e n t}}(l^{1}\mid i^{0},d^{0})\approx0.513$ . Queries such as these, where we predict the “downstream” efects of various factors (such as George’s intelligence), are instances of causal reasoning or prediction . 
> 从贝叶斯网络的最顶端节点自上而下推理的过程/预测多个因素导致的效果的过程，就是因果推理或者预测

Now, consider a recruiter for Acme Consulting, trying to decide whether to hire George based on our previous model. A priori, the recruiter believes that George is 30 percent likely to be intelligent. He obtains George’s grade record for a particular class Econ101 and sees that George received a C in the class $(g^{3})$ . His probability that George has high intelligence goes down signiﬁcantly, to about 7.9 percent; that is, $P_{\mathcal{B}^{s t u d e n t}}(i^{1}\mid g^{3})\approx0.079$ . We note that the probability that the class is a difcult one also goes up, from 40 percent to 62.9 percent. 

Now, assume that the recruiter fortunately (for George) lost George’s transcript, and has only the recommendation letter from George’s professor in Econ101, which (not surprisingly) is evidential reasoning weak. The probability that George has high intelligence still goes down, but only to 14 percent: $P_{\mathcal{B}^{s t u d e n t}}(i^{1}\mid l^{0})\approx0.14$ . Note that if the recruiter has both the grade and the letter, we have the same probability as if he had only the grade: $P_{\mathcal{B}^{s t u d e n t}}(i^{1}\mid g^{3},l^{0})\approx0.079;$ ; we will revisit this issue. Queries such as this, where we reason from efects to causes, are instances of evidential reasoning or explanation . 
> 从效果推理导致它的原因的过程，就是证据推理或者解释

Finally, George submits his SAT scores to the recruiter, and astonishingly, his SAT score is high. The probability that George has high intelligence goes up dramatically, from 7.9 percent to 57.8 percent: $P_{\mathcal{B}^{s t u d e n t}}(i^{1}\mid g^{3},s^{1})\approx0.578$ . Intuitively, the reason that the high SAT score outweighs the poor grade is that students with low intelligence are extremely unlikely to get good scores on their SAT, whereas students with high intelligence can still get C’s. However, smart students are much more likely to get C’s in hard classes. Indeed, we see that the probability that Econ101 is a difcult class goes up from the 62.9 percent we saw before to around 76 percent. 

This last pattern of reasoning is a particularly interesting one. The information about the SAT gave us information about the student’s intelligence, which, in conjunction with the student’s grade in the course, told us something about the difculty of the course. In efect, we have one causal factor for the Grade variable — Intelligence — giving us information about another — Diffculty . 

Let us examine this pattern in its pure form. As we said, $P_{\mathcal{B}^{s t u d e n t}}(i^{1}\mid g^{3})\approx0.079$ other hand, if we now discover that Econ101 is a hard class, we have that $P_{\mathcal{B}^{s t u d e n t}}(i^{1}\mid g^{3},d^{1})\approx$ B | ≈ 0 . 11 . In efect, we have provided at least a partial explanation for George’s grade in Econ101. To take an even more striking example, if George gets a B in Econ 101, we have that $P_{\mathcal{B}^{s t u d e n t}}(i^{1}\mid$ $g^{2})\approx0.175$ . On the other hand, if Econ101 is a hard class, we get $P_{\mathcal{B}^{s t u d e n t}}(i^{1}\mid g^{2},d^{1})\approx0.34$ . In efect we have *explained away* the poor grade via the difculty of the class. 
**Explaining away is an instance of a general reasoning pattern called intercausal reasoning , where diferent causes of the same efect can interact. This type of reasoning is a very common pattern in human reasoning.** 
> 解释是因果关系间推理的一个实例，在因果关系间推理中，相同效果的不同导因会相互交互 （例如，在已知 Intelligence 高并且 Grade 低的情况下，就可以推理出 Difficulty 高的概率是较高的）
> 因果关系间推理也是人类推理的常见模式

For example, when we have fever and a sore throat, and are concerned about mononucleosis, we are greatly relieved to be told we have the ﬂu. Clearly, having the ﬂu does not prohibit us from having mononucleosis. Yet, having the ﬂu provides an alternative explanation of our symptoms, thereby reducing substantially the probability of mononucleosis. 

This intuition of providing an alternative explanation for the evidence can be made very precise. As shown in exercise 3.3, if the ﬂu deterministic ally causes the symptoms, the probability of mononucleosis goes down to its prior probability (the one prior to the observations of any symptoms). On the other hand, if the ﬂu might occur without causing these symptoms, the probability of mononucleosis goes down, but it still remains somewhat higher than its base level. Explaining away, however, is not the only form of intercausal reasoning. The inﬂuence can go in any direction. Consider, for example, a situation where someone is found dead and may have been murdered. The probabilities that a suspect has motive and opportunity both go up. If we now discover that the suspect has motive, the probability that he has opportunity goes up. (See exercise 3.4.) 

It is important to emphasize that, although our explanations used intuitive concepts such as cause and evidence, there is nothing mysterious about the probability computations we performed. They can be replicated simply by generating the joint distribution, as deﬁned in equation (3.9), and computing the probabilities of the various events directly from that. 
### 3.2.2 Basic Independencies in Bayesian Networks 
As we discussed, a Bayesian network graph $\mathcal{G}$ can be viewed in two ways. In the previous section, we showed, by example, how it can be used as a skeleton data structure to which we can attach local probability models that together deﬁne a joint distribution. 
In this section, we provide a formal semantics for a Bayesian network, starting from the perspective that the graph encodes a set of conditional independence assumptions. We begin by understanding, intuitively, the basic conditional independence assumptions that we want a directed graph to encode. We then formalize these desired assumptions in a deﬁnition. 
> 上一个部分我们展示了如何表现出贝叶斯网络架构中的局部条件概率分布，并结合它们定义了联合分布
> 本节介绍贝叶斯网络的形式语义
#### 3.2.2.1 Independencies in the Student Example 
In the Student example, we used the intuition that edges represent direct dependence. For example, we made intuitive statements such as “the professor’s recommendation letter depends only on the student’s grade in the class”; this statement was encoded in the graph by the fact that there are no direct edges into the $L$ node except from $G$ . This intuition, that “a node depends directly only on its parents,” lies at the heart of the semantics of Bayesian networks. 
> 我们在贝叶斯网络中使用边表示变量之间的独立性
> 当 $G$ 节点仅被 $L$ 节点出来的一条边指向时，表示该变量仅仅直接依赖于 $L$ 变量，这就是贝叶斯网络的语义的核心

We give formal semantics to this assertion using conditional independence statements. For example, the previous assertion can be stated formally as the assumption that $L$ is conditionally independent of all other nodes in the network given its parent $G$ : 

$$
(L\perp I,D,S\mid G).\tag{3.10}
$$

In other words, once we know the student’s grade, our beliefs about the quality of his recommendation letter are not inﬂuenced by information about any other variable. 
> 形式化地，我们将上一个声明表示为一个假设：在网络中给定它的父节点 $G$，$L$ 与所有其他的节点条件独立，也就是其他的任意节点都不会给出关于这个节点更多的信息

Similarly, to formalize our intuition that the student’s SAT score depends only on his intelligence, we can say that $S$ is conditionally independent of all other nodes in the network given its parent $I$ : 

$$
(S\perp D,G,L\mid I).\tag{3.11}
$$ 
Now, let us consider the $G$ node. Following the pattern blindly, we may be tempted to assert that $G$ is conditionally independent of all other variables in the network given its parents. However, this assumption is false both at an intuitive level and for the speciﬁc example distribution we used earlier. Assume, for example, that we condition on $i^{1},d^{1}$ ; that is, we have a smart student in a difcult class. In this setting, is $G$ independent of $L?$ Clearly, the answer is no: if we observe $l^{1}$ (the student got a strong letter), then our probability in $g^{1}$ (the student received an A in the course) should go up; that is, we would expect 

$$
P(g^{1}\mid i^{1},d^{1},l^{1})>P(g^{1}\mid i^{1},d^{1}).
$$ 
Indeed, if we examine our distribution, the latter probability is 0.5 (as speciﬁed in the CPD), whereas the former is a much higher 0.712 . 

Thus, we see that we do not expect a node to be conditionally independent of all other nodes given its parents. In particular, even given its parents, it can still depend on its descendants. 
> 但给定一个节点的父节点，该节点实际并不会条件独立于网络中的所有节点，它仍然会依赖于自己的子节点，

Can it depend on other nodes? For example, do we expect $G$ to depend on $S$ given $I$ and $D?$ Intuitively, the answer is no. Once we know, say, that the student has high intelligence, his SAT score gives us no additional information that is relevant toward predicting his grade. Thus, we would want the property that: 

$$
(G\perp S\mid I,D).\tag{3.12}
$$ 
It remains only to consider the variables $I$ and $D$ , which have no parents in the graph. 

Thus, in our search for independencies given a node’s parents, we are now looking for marginal independencies. As the preceding discussion shows, in our distribution $P_{\mathcal{B}^{s t u d e n t}}$ , $I$ is not independent of its descendants $G,\,L,$ or $S$ . Indeed, the only nondescendant of I is D . Indeed, we assumed implicitly that Intelligence and Difculty are independent. Thus, we expect that: 

$$
(I\perp D).\tag{3.13}
$$ 
This analysis might seem somewhat surprising in light of our earlier examples, where learning something about the course difculty drastically changed our beliefs about the student’s intelligence. In that situation, however, we were reasoning in the presence of information about the student’s grade. In other words, we were demonstrating the dependence of $I$ and $D$ given $G$ . This phenomenon is a very important one, and we will return to it. 
> 我们实际上假设了 $I$ 边际独立于 $D$，但回忆到在之前我们实际上观察到了 $I, D$ 在给定 $G$ 的情况下是不独立的，二者并不矛盾

For the va able $D$ , both $I$ and $S$ are nondescendants. Recall that, if $(I\perp D)$ then $(D\perp I)$ . The variable S increases our beliefs in the student’s intelligence, but knowing that the student is smart (or not) does not inﬂuence our beliefs in the difculty of the course. Thus, we have that 

$$
(D\perp I,S).\tag{3.14}
$$ 
We can see a pattern emerging. Our intuition tells us that the parents of a variable “shield” it from probabilistic inﬂuence that is causal in nature. In other words, once I know the value of the parents, no information relating directly or indirectly to its parents or other ancestors can inﬂuence my beliefs about it. However, information about its descendants can change my beliefs about it, via an evidential reasoning process. 
> 对于一个变量，一旦我们知道了其父变量的值，就没有与父变量或其他祖先直接相关或间接相关的信息可以影响我们对于该变量的概念
> 但关于该变量的子变量的信息则有影响（通过证据推理过程）
#### 3.2.2.2 Bayesian Network Semantics 
We are now ready to provide the formal deﬁnition of the semantics of a Bayesian network structure. We would like the formal deﬁnition to match the intuitions developed in our example. 

**Deﬁnition 3.1**
A Bayesian network structure G is a directed acyclic graph whose nodes represent random variables Bayesian network structure X1; : : : ; Xn. Let PaG Xi denote the parents of Xi in G, and NonDescendantsXi denote the variables in the graph that are not descendants of Xi. Then G encodes the following set of conditional local independence assumptions, called the local independencies, and denoted by I‘(G):

$$
(X_{i}\perp\mathrm{NonDS}_{X_{i}}\mid\mathrm{Pa}_{X_{i}}^{\mathcal{G}}).
$$ 
In other words, the local independencies state that each node $X_{i}$ is conditionally independent of its nondescendants given its parents. 
> 定义：贝叶斯网络是一个有向无环图，其中节点表示随机变量 $X_1,\dots, X_n$
> 贝叶斯网络编码了条件独立性假设，称为局部独立性，即对于网络中的每个变量 $X_i$，$X_i$ 在给定它的父节点的条件下和不是它的子孙节点的其他节点条件独立
> 我们记网络 $G$ 编码的条件独立性假设为 $\mathcal I_{\mathscr l}(\mathcal G)$

Returning to the Student network $G_{s t u d e n t},$ the local Markov independencies are precisely the ones dictated by our intuition, and speciﬁed in equation (3.10) – equation (3.14). 

Box 3. B — Case Study: The Genetics Example. One of the very earliest uses of a Bayesian net- work model (long before the general framework was deﬁned) is in the area of genetic pedigrees. In this setting, the local independencies are particularly intuitive. In this application, we want to model the transmission of a certain property, say blood type, from parent to child. The blood type of a person is an observable quantity that depends on her genetic makeup. Such properties are called phenotypes . The genetic makeup of a person is called genotype . 

To model this scenario properly, we need to introduce some background on genetics. The human genetic material consists of 22 pairs of autosomal chromosomes and a pair of the sex chromosomes (X and Y). Each chromosome contains a set of genetic material, consisting (among other things) of genes that determine a person’s properties. A region of the chromosome that is of interest is called $a$ locus ; a locus can have several variants, called alleles . 

For concreteness, we focus on autosomal chromosome pairs. In each autosomal pair, one chro- mosome is the paternal chromosome, inherited from the father, and the other is the maternal chromosome, inherited from the mother. For genes in an autosomal pair, a person has two copies of the gene, one on each copy of the chromosome. Thus, one of the gene’s alleles is inherited from the person’s mother, and the other from the person’s father. For example, the region containing the gene that encodes a person’s blood type is a locus. This gene comes in three variants, or alleles: $A$ , $B$ , and $O$ . Thus, a person’s genotype is denoted by an ordered pair, such as $\langle A,B\rangle$ ; with thr choices for each entry in the pair, there are 9 possible genotypes. The blood type phenotype is a function of both copies of the gene. For example, if the person has an $A$ allele and an $O$ allele, her observed blood type is “A.” If she has two $O$ alleles, her observed blood type is “O.” 

To represent this domain, we would have, for each person, two variables: one representing the person’s genotype, and the other her phenotype. We use the name $G(p)$ to represent person p ’s genotype, and $B(p)$ to represent her blood type. 

In this example, the independence assumptions arise immediately from the biology. Since the blood type is a function of the genotype, once we know the genotype of a person, additional evidence about other members of the family will not provide new information about the blood type. Similarly, the process of genetic inheritance implies independence assumption. Once we know the genotype of both parents, we know what each of them can pass on to the ofspring. Thus, learning new information about ancestors (or nondescendants) does not provide new information about the genotype of the ofspring. These are precisely the local independencies in the resulting network structure, shown for a simple family tree in ﬁgure 3.B.1. The intuition here is clear; for example, Bart’s blood type is correlated with that of his aunt Selma, but once we know Homer’s and Marge’s genotype, the two become independent. 

To deﬁne the probabilistic model fully, we need to specify the CPDs. There are three types of CPDs in this model: 

• The penetrance model $P(B(c)\mid G(c))$ , which describes the probability of diferent variants of a particular phenotype (say diferent blood types) given the person’s genotype. In the case of the blood type, this CPD is a deterministic function, but in other cases, the dependence can be more complex. • The transmission model $P(G(c)\mid G(p),G(m))$ , where $c$ is a person and $p,m$ her father and mother, respectively. Each parent is equally likely to transmit either of his or her two alleles to the child. • Genotype priors $P(G(c))$ , used when person c has no parents in the pedigree. These are the general genotype frequencies within the population. 

Our discussion of blood type is simpliﬁed for several reasons. First, some phenotypes, such as late-onset diseases, are not a deterministic function of the genotype. Rather, an individual with a particular genotype might be more likely to have the disease than an individual with other genotypes. Second, the genetic makeup of an individual is deﬁned by many genes. Some phenotypes might depend on multiple genes. In other settings, we might be interested in multiple phenotypes, which (naturally) implies a dependence on several genes. Finally, as we now discuss, the inheritance patterns of diferent genes are not independent of each other. 

Recall that each of the person’s autosomal chromosomes is inherited from one of her parents. However, each of the parents also has two copies of each autosomal chromosome. These two copies, within each parent, recombine to produce the chromosome that is transmitted to the child. Thus, the maternal chromosome inherited by Bart is a combination of the chromosomes inherited by his mother Marge from her mother Jackie and her father Clancy. The recombination process is stochastic, but only a handful recombination events take place within a chromosome in a single generation. Thus, if Bart inherited the allele for some locus from the chromosome his mother inherited from her mother Jackie, he is also much more likely to inherit Jackie’s copy for a nearby locus. Thus, to construct an appropriate model for multilocus inheritance, we must take into consideration the probability of a recombination taking place between pairs of adjacent loci. 

We can facilitate this modeling by introducing selector variables that capture the inheritance pattern along the chromosome. In particular, for each locus $\ell$ and each child $c_{z}$ , we have a variable $S(\ell,c,m)$ that takes the value 1 if the locus $\ell$ in c ’s maternal chromosome was inherited from c ’s maternal grandmother, and 2 if this locus was inherited from c ’s maternal grandfather. We have a similar selector variable $S(\ell,c,p)$ for c ’s paternal chromosome. We can now model correlations induced by low recombination frequency by correlating the variables $S(\ell,c,m)$ and $S(\ell^{\prime},c,m)$ for adjacent loci $\ell,\ell^{\prime}$ . 

This type of model has been used extensively for many applications. In genetic counseling and prediction, one takes a phenotype with known loci and a set of observed phenotype and genotype data for some individuals in the pedigree to infer the genotype and phenotype for another person in the pedigree (say, a planned child). The genetic data can consist of direct measurements of the relevant disease loci (for some individuals) or measurements of nearby loci, which are correlated with the disease loci. 

In linkage analysis, the task is a harder one: identifying the location of disease genes from pedigree data using some number of pedigrees where a large fraction of the individuals exhibit a disease phenotype. Here, the available data includes phenotype information for many individuals in the pedigree, as well as genotype information for loci whose location in the chromosome is known. Using the inheritance model, the researchers can evaluate the likelihood of these observations under diferent hypotheses about the location of the disease gene relative to the known loci. By repeated calculation of the probabilities in the network for diferent hypotheses, researchers can pinpoint the area that is “linked” to the disease. This much smaller region can then be used as the starting point for more detailed examination of genes in that area. This process is crucial, for it can allow the researchers to focus on a small area (for example, $1/10,000$ of the genome). 

As we will see in later chapters, the ability to describe the genetic inheritance process using a sparse Bayesian network provides us the capability to use sophisticated inference algorithms that allow us to reason about large pedigrees and multiple loci. It also allows us to use algorithms for model learning to obtain a deeper understanding of the genetic inheritance process, such as recombination rates in diferent regions or penetrance probabilities for diferent diseases. 
### 3.2.3 Graphs and Distributions 
The formal semantics of a Bayesian network graph is as a set of independence assertions. On the other hand, our Student BN was a graph annotated with CPDs, which deﬁned a joint distribution via the chain rule for Bayesian networks. In this section, we show that these two deﬁnitions are, in fact, equivalent. 
A distribution $P$ satisﬁes the local independencies associated with a graph $\mathcal{G}$ if and only if $P$ is representable as a set of CPDs associated with the graph $\mathcal{G}$ . We begin by formalizing the basic concepts. 
> 贝叶斯网络图的形式语义就是一系列独立假设
> 一个分布 $P$  满足和图 $G$ 相关的局部独立性当且仅当 $P$ 可以用和 $G$ 相关的一系列条件概率分布表示
#### 3.2.3.1 I-Maps 
We ﬁrst deﬁne the set of independencies associated with a distribution $P$ . 

**Deﬁnition 3.2**
Let $P$ be a distribution over $\mathcal{X}$ . e deﬁne $\mathcal{Z}(P)$ to be the set of independence assertions of the form $(X\perp Y\mid Z)$ that hold in P . 
> 定义：
> 令 $\mathcal I (P)$ 表示 $P$ 中满足的形式为 $(X\perp Y \mid Z)$ 的独立声明

We can now rewrite the statement that $^{a}P$ satisﬁes the local independencies associated with ${\mathcal{G}}"$ simply as ${\mathcal{T}}_{\ell}({\mathcal{G}})\subseteq{\mathcal{Z}}(P)$ . In this case, we say that $\mathcal{G}$ is an $I^{,}$ -map (independency map) for $P$ . However, it is useful to deﬁne this concept more broadly, since diferent variants of it will be used throughout the book. 
> 此时，$P$ 满足和 $G$ 相关的局部独立性可以简单记为 $\mathcal I_{\mathscr l}(G) \subseteq \mathcal I (P)$
> 我们称此时 $\mathcal G$ 是 $P$ 的一个 I-map（独立性映射）

**Deﬁnition 3.3** I-map 
Let K be any graph object associated with a set of independencies I(K). We say that K is anI-map for a set of independencies I if I(K) ⊆ I|I-map|
> 定义：
> 令 $\mathcal K$ 是和独立性集合 $\mathcal I (\mathcal K)$ 相关的任意图对象，对于一个独立性集合 $\mathcal I$，如果 $\mathcal I (\mathcal K) \subseteq \mathcal I$，则我们称 $\mathcal K$ 是 $\mathcal I$ 的一个 I-map 

We now say that $\mathcal{G}$ is an I-map for $P$ if $\mathcal{G}$ is an I-map fo $\mathcal{Z}(P)$ . 
> 因此，如果 $\mathcal G$ 是 $\mathcal I(P)$ 的一个 I-map ($\mathcal G$ 编码的独立性集合是 $P$ 编码的独立性集合的子集)，则 $\mathcal G$ 是 $P$ 的一个 I-map

As we can see from the direction of the inclusion, for G o be an I-map of $P$ , it is necessary that $\mathcal{G}$ does not m lead us regar ng independencies in P : any independence that $\mathcal{G}$ asserts must also hold in P . Conversely, P may have additional independencies that are not reﬂected in $\mathcal{G}$ . 
> 显然，如果 $\mathcal G$ 是 $P$ 的 I-map，则任意 $\mathcal G$ 中成立的独立性必须在 $P$ 中存在，另外 $P$ 可以有没有在 $\mathcal G$ 中反应的额外的独立性

Let us illustrate the concept of an I-map on a very simple example. 

Example 3.1
Consider a joint probability space over two independent random variables $X$ and $Y$ . There are three possibl over t o nodes: $\mathcal{G}_{\varnothing}$ , whi isconnected pair $\begin{array}{r l}{X}&{{}\;Y;\mathcal{G}_{X\rightarrow Y}}\end{array}$ G , which → has the edge $X\rightarrow Y$ → ; and G ${\mathcal{G}}_{Y\to X}$ → , which contains $Y\rightarrow X$ → . The graph $\mathcal{G}_{\varnothing}$ encodes the assumption that $(X\perp Y)$ . The latter two encode no independence assumptions. Consider the following two distributions: 

In the example on the left, $X$ and $Y$ are independent in $P$ ; for example, $P(x^{1})=0.48+0.12=$ 0 . 6 , $P(y^{1})\,=\,0.8,$ , and $P(x^{1},y^{1})\,=\,0.48\,=\,0.6\cdot0.8$ Thus, $(X\bot Y)\in{\mathcal{Z}}(P)$ , and we have that $\mathcal{G}_{\varnothing}$ an I-map of P . In fact, all three graphs are I-maps of P : ${\mathcal{T}}_{\ell}({\mathcal{G}}_{X\to Y})$ is empty, so that $P$ all the independenci s in it (si ilarly for ${\mathcal{G}}_{Y\rightarrow X.}$ ). In th example n the right, $(X\bot Y)\not\in{\mathcal{Z}}(P)$ ⊥ ̸∈I , so that $\mathcal{G}_{\varnothing}$ is not an I-map of P . Both other graphs are I-maps of P . 
#### 3.2.3.2 I-Map to Factorization 
A BN structure $\mathcal{G}$ encodes a set of conditional independence assumptions; every distribution for which G is an I-map must satisfy these assumptions. This property is the key to allowing the compact factorized representation that we saw in the Student example in section 3.2.1. The basic principle is the same as the one we used in the naive Bayes decomposition in section 3.1.3. 
> 一个贝叶斯网络结构 $\mathcal G$ 编码了一系列条件独立性假设，被该网络 I-map 的任意分布都必须满足这些条件独立性假设

Consider any distribution $P$ for which our Student BN $G_{s t u d e n t}$ is an I-map. We will de- compose the joint distribution and show that it factorizes into local probabilistic models, as in section 3.2.1. Consider the joint distribution $P(I,D,G,L,S)$ ; from the chain rule for probabil- ities (equation (2.5)), we can decompose this joint distribution in the following way: 

$$
P(I,D,G,L,S)=P(I)P(D\mid I)P(G\mid I,D)P(L\mid I,D,G)P(S\mid I,D,G,L).\tag{3.15}
$$ 
This transformation relies on no assumptions; it holds for any joint distribution $P$ . However, it is also not very helpful, since the conditional probabilities in the factorization on the right-hand side are neither natural nor compact. For example, the last factor requires the speciﬁcation of 24 conditional probabilities: $P(s^{1}\mid i,d,g,l)$ for every assignment of values $i,d,g,l$ . 
> 我们先对联合分布进行链式分解，链式分解不依赖于任何假设

This form, however, allows us to apply the conditional independence assumptions induced from the BN. Let us assume that $G_{s t u d e n t}$ is an I-map for our distribution $P$ . In particular, from equation (3.13), we have that $(D\perp I)\in{\mathcal{Z}}(P)$ . From that, we can conclude that $P(D\mid I)=$ $P(D)$ , allowing us to simplify the second factor on the right-hand side. Similarly, we know from equation (3.10) that $(L\ \bot\ I,D\ |\ G)\in{\mathcal{Z}}(P)$ . Hence, $P(L\mid I,D,G)=P(L\mid G)$ , allowing us to simplify the third term. Using equation (3.11) in a similar way, we obtain that 

$$
P(I,D,G,L,S)=P(I)P(D)P(G\mid I,D)P(L\mid G)P(S\mid I).\tag{3.16}
$$ 
This factorization is precisely the one we used in section 3.2.1. 
> 但链式分解允许我们使用条件独立性假设，得到更紧凑的表示

This result tells us that any entry in the joint distribution can be computed as a product of factors, one for each variable. Each factor represents a conditional probability of the variable given its parents in the network. This factorization applies to any distribution $P$ for which $G_{s t u d e n t}$ is an I-map. 
> 注意以上对于联合分布的分解对于任意被 $\mathcal G$ I-map 的分布都适用

We now state and prove this fundamental result more formally. 

**Deﬁnition 3.4** factorization
Let $\mathcal G$ be a BN graph over the variables $X_1,\dots,X_n$. We say that a distribution P over the samespace factorizes according to G if P can be expressed as a product factorization
> 定义：
> 令 $\mathcal G$ 是在变量 $X_1,\dots, X_n$ 上的贝叶斯图，如果在同一空间中的分布 $P$ 可以被表示成以下乘积，则称它根据 $\mathcal G$ 分解

$$
P(X_1,\dots,X_n) = \prod_{i=1}^n P(X_i\mid \text{Pa}_{X_i}^{\mathcal G})\tag{2.17}
$$

This equation is called the chain rule for Bayesian networks. The individual factors $P (X_i \mid \text{Pa}_{X_i}^{\mathcal G}$) Bayesian networks are called conditional probability distributions (CPDs) or local probabilistic models. chain rule for Bayesian networks 
> 该式被称为贝叶斯网络的链式法则，其中的独立因子 $P (X_i \mid \text{Pa}_{X_i}^{\mathcal G})$ 被称为条件概率分布或者局部概率模型

**Deﬁnition 3.5** Bayesian network 
Bayesian network is a pair B = (G; P ) where P factorizes over G, and where P is specified as a set of CPDs associated with G’s nodes. The distribution P is often annotated PB.
> 定义：
> 一个贝叶斯网络就是一个元组 $\mathcal B = (\mathcal G, P)$，其中 $P$ 按照 $\mathcal G$ 分解，并可以用和 $\mathcal G$ 相关的节点的一系列条件概率分布表示，$P$ 也可以记为 $P_{\mathcal B}$

We can now prove that the phenomenon we observed for Gstudent holds more generally

**Theorem 3.1** 
Let $\mathcal{G}$ be a BN str ture o er a set o rando variables $\mathcal{X}$ , and let $P$ e a joint distribution over the same space. If is an I-map for P , then P factorizes according to . 
> 定理：
> 令 $\mathcal G$ 是一个在随机变量集合 $\mathcal X$ 上的贝叶斯网络，令 $P$ 是相同空间上的一个联合分布，如果 $\mathcal G$ 是 $P$ 的一个 I-map， 则 $P$ 根据 $\mathcal G$ 分解

Proof 
Assume, without loss of generality, that $X_{1},\ldots,X_{n}$ is a topological ordering of the variables in $\mathcal{X}$ relative to $\mathcal{G}$ (see deﬁnition 2.19). As in our example, we ﬁrst use the chain rule for probabilities: 

$$
P(X_{1},\dots,X_{n})=\prod_{i=1}^{n}P(X_{i}\mid X_{1},\dots,X_{i-1}).
$$ 
No sider one of the fact $P_{\cdot}(X_{i}\mid X_{1},.\,.\,,X_{i-1})$ . As $\mathcal{G}$ is an map for $P$ , we have tha $\cdot X_{i}\perp$ escendants $X_{i}$ | $|\operatorname{Pa}_{X_{i}}^{\mathcal{G}}\}\in\mathcal{Z}(P)$ I . By assumption, all of $X_{i}$ ’s parents are in the set $X_{1},.\ldots,X_{i-1}$ . Furthermore, none of $X_{i}$ ’s descendants can possibly be in the set. Hence,

$$
\left\{X_{1},.\,.\,.\,,X_{i-1}\right\}=\mathrm{Pa}_{X_{i}}\cup Z
$$ 
where $Z\subseteq{\mathrm{NonDS}}_{X_{i}}$ . From the local independencies for $X_{i}$ and from the decom- position property (equation (2.8)) it follows that $(X_{i}\perp Z\mid\mathrm{Pa}_{X_{i}})$ . Hence, we have that 

$$
P(X_{i}\mid X_{1},.\,.\,,X_{i-1})=P(X_{i}\mid\mathrm{Pa}_{X_{i}}).
$$ 
Applying this transformation to all of the factors in the chain rule decomposition, the result follows.
> 证明：
> 先利用拓扑排序，对 $P$ 进行链式分解
> 因为 $\mathcal G$ 是 $P$ 的 I-map，则分解项 $P (X_i \mid X_1,\dots, X_{i-1})$ 中，除了 $X_i$ 的父节点都可以排除（它们条件独立于 $X_i$），得到 $P (X_i \mid \text{Pa}_{X_i})$
> 因此 $P$ 可以表示为式 2.17 的形式，也就是多个局部分布的乘积

Thus, the condition independence assumptions implied by a BN structure $\mathcal{G}$ allow us to factorize a distribution P for which $\mathcal{G}$ is an I-map into small CPDs. Note that the proof is con- structive, providing a precise algorithm for constructing the factorization given the distribution $P$ and the graph $\mathcal{G}$ . 
> 故只要 $\mathcal G$ 是 $P$ 的 I-map，$P$ 就可以按照 $\mathcal G$ 分解为多个条件概率分布的乘积，以上的证明也提供了如何构建该分解的步骤

The resulting factorized representation can be substantially ，more compact, particularly for sparse structures. 

Example 3.2 
In our Student example, the number of independent parameters is ﬁfteen: we have two binomial distributions $P(I)$ and $P(D)$ , with one independent parameter each; we have four multinomial distributions over $G$ — one for each assignment of values to $I$ and $D$ — each with two independent parameters; we have three binomial distributions over $L$ , each with one independent parameter; and similarly two binomial distributions over $S$ , each with an independent parameter. The speciﬁcation of the full joint distribution would require $48-1=47$ independent parameters. 

More generally, in a distribution over $n$ binary random variables, the speciﬁcation of the joint distribut n requires $2^{n}-1$ independen parameters. If the distribution factorizes according to a graph G where each has at most k parents, the total number of independent parameters required is less than n $n\cdot2^{k}$ · (see exercise 3.6). 
In many applications, we can assume a certain locality of inﬂuence between variables: although each variable is generally correlated with many of the others, it often depends directly on only a small number of other variables. Thus, in many cases, $k$ will be very small, even though $n$ is large. As a consequence, the number of parameters in the Bayesian network representation is typically exponentially smaller than the number of parameters of a joint distribution. This property is one of the main beneﬁts of the Bayesian network representation. 
> 在许多应用中，我们都可以假设变量之间的影响存在某种局部性，虽然每个变量会和许多其他变量相关，但它经常仅仅直接依赖于少部分的变量
> 因此，贝叶斯网络表示会指数级地减少表示联合分布所需要的参数量
#### 3.2.3.3 Factorization to I-Map 
Theorem 3.1 shows one direction of the fundamental connection between the conditional in- dependencies encoded by the BN structure and the factorization of the distribution into local probability models: that the conditional independencies imply factorization. The converse also holds: factorization according to $\mathcal{G}$ implies the associated conditional independencies. 
> 定理3.1展示了贝叶斯网络编码的条件独立性 imply 了分解
> 反过来其实也成立，根据 $\mathcal G$ 的分解 imply 了相关的条件独立性

**Theorem 3.2** 
Let $\mathcal{G}$ be a BN ructure over a set of ran m var bles $\mathcal{X}$ and let $P$ be a joint distribution over the same space. If $P$ factorizes according to G , then G is an I-map for P . 
> 定理：
> 令 $\mathcal G$ 是随机变量集合 $\mathcal X$ 上的贝叶斯网络，令 $P$ 是相同空间的一个联合分布，则如果 $P$ 根据 $\mathcal G$ 分解，则 $\mathcal G$ 是 $P$ 的一个 I-map

We illustrate this theorem by example, leaving the proof as an exercise (exercise 3.9). Let $P$ be so e distribution that factorizes according to $G_{s t u d e n t}$ . We need to show that $\mathcal{T}_{\ell}(G_{s t u d e n t})$ holds in P . Consider the indep dence assumption for the random variable $S-(S\perp D,G,L\mid I)$ . To prove that it holds for P , we need to show that 

$$
P(S\mid I,D,G,L)=P(S\mid I).
$$ 
By deﬁnition, 

$$
P(S\mid I,D,G,L)=\frac{P(S,I,D,G,L)}{P(I,D,G,L)}.
$$ 
By the chain rule for BNs equation (3.16), the numerator is equal to $P(I)P(D)P(G\mid I,D)P(L\mid$ $G)P(S\mid I)$ . By the process of marginalizing over a joint distribution, we have that the denominator is: 

$$
\begin{array}{r c l}{{P(I,D,G,L)}}&{{=}}&{{\displaystyle\sum_{S}P(I,D,G,L,S)}}\\ {{}}&{{=}}&{{\displaystyle\sum_{S}P(I)P(D)P(G\mid I,D)P(L\mid G)P(S\mid I)}}\\ {{}}&{{=}}&{{\displaystyle P(I)P(D)P(G\mid I,D)P(L\mid G)\sum_{S}P(S\mid I)}}\\ {{}}&{{=}}&{{P(I)P(D)P(G\mid I,D)P(L\mid G),}}\end{array}
$$ 
where the last step is a consequence of the fact that $P(S\mid I)$ is a distribution over values of $S$ , and therefore it sums to 1. We therefore have that 

$$
\begin{array}{r c l}{P(S\mid I,D,G,L)}&{=}&{\displaystyle\frac{P(S,I,D,G,L)}{P(I,D,G,L)}}\\ &{=}&{\displaystyle\frac{P(I)P(D)P(G\mid I,D)P(L\mid G)P(S\mid I)}{P(I)P(D)P(G\mid I,D)P(L\mid G)}}\\ &{=}&{P(S\mid I).}\end{array}
$$ 
Box 3. C — **Skill: Knowledge Engineering.** 
Our discussion of Bayesian network construction fo- cuses on the process of going from a given distribution to a Bayesian network. Real life is not like that. We have a vague model of the world, and we need to crystallize it into a network structure and parameters. This task breaks down into several components, each of which can be quite subtle. Unfortunately, modeling mistakes can have signiﬁcant consequences for the quality of the answers obtained from the network, or to the cost of using the network in practice. 
> 网络的错误建模往往会导致结果错误严重

**Picking variables** When we model a domain, there are many possible ways to describe the relevant entities and their attributes. Choosing which random variables to use in the model is often one of the hardest tasks, and this decision has implications throughout the model. A common problem is using ill-deﬁned variables. For example, deciding to include the variable Fever to describe a patient in a medical domain seems fairly innocuous. However, does this random variable relate to the internal temperature of the patient? To the thermometer reading (if one is taken by the medical staf)? Does it refer to the temperature of the patient at a speciﬁc moment (for example, the time of admission to the hospital) or to occurrence of a fever over a prolonged period? Clearly, each of these might be a reasonable attribute to model, but the interaction of Fever with other variables depends on the speciﬁc interpretation we use. 

As this example shows, we must be precise in deﬁning the variables in the model. The clarity test is a good way of evaluating whether they are sufciently well deﬁned. Assume that we are a million years after the events described in the domain; can an omniscient being, one who saw everything, determine the value of the variable? For example, consider a Weather variable with a value sunny. To be absolutely precise, we must deﬁne where we check the weather, at what time, and what fraction of the sky must be clear in order for it to be sunny. For a variable such as Heart-attack, we must specify how large the heart attack has to be, during what period of time it has to happen, and so on. By contrast, a variable such as Risk-of-heart-attack is meaningless, as even an omniscient being cannot evaluate whether a person had high risk or low risk, only whether the heart attack occurred or not. Introducing variables such as this confounds actual events and their probability. Note, however, that we can use a notion of “risk group,” as long as it is deﬁned in terms of clearly speciﬁed attributes such as age or lifestyle. 
> 检验模型中变量的定义：clarity test（清楚）

If we are not careful in our choice of variables, we will have a hard time making sure that evidence observed and conclusions made are coherent. 

Generally speaking, we want our model to contain variables that we can potentially observe or that we may want to query. However, sometimes we want to put in a hidden variable that is neither observed nor directly of interest. Why would we want to do that? Let us consider an example relating to a cholesterol test. Assume that, for the answers to be accurate, the subject has to have eaten nothing after 10:00 PM the previous evening. If the person eats (having no willpower), the results are consistently of. We do not really care about a Willpower variable, nor can we observe it. However, without it, all of the diferent cholesterol tests become correlated. To avoid graphs where all the tests are correlated, it is better to put in this additional hidden variable, rendering them conditionally independent given the true cholesterol level and the person’s willpower. 
> 一般来说，我们希望模型包含我们可以观察到或者我们想要查询的变量
> 但有时我们会包括一个我们观察不到，也不会查询的隐变量，这类隐变量的取值会对结果产生很大的影响

On the other hand, it is not necessary to add every variable that might be relevant. In our Student example, the student’s SAT score may be afected by whether he goes out for drinks on the night before the exam. Is this variable important to represent? The probabilities already account for the fact that he may achieve a poor score despite being intelligent. It might not be worthwhile to include this variable if it cannot be observed. 
> 另一方面，对于一个已有的变量，也不必要加入任何可能与它相关的变量作为隐变量，要考虑其重要性，该变量的影响可能已经在这个已有的变量的取值中表示了

It is also important to specify a reasonable domain of values for our variables. In particular, if our partition is not ﬁne enough, conditional independence assumptions may be false. For example, we might want to construct a model where we have a person’s cholesterol level, and two cholesterol tests that are conditionally independent given the person’s true cholesterol level. We might choose to deﬁne the value normal to correspond to levels up to 200, and high to levels above 200. But it may be the case that both tests are more likely to fail if the person’s cholesterol is marginal (200–240). In this case, the assumption of conditional independence given the value (high/normal) of the cholesterol test is false. It is only true if we add a marginal value. 
> 变量的合理取值范围同样重要，对于一个父变量，有时只有在合理的取值范围内，其自变量条件于它才会与其他变量有条件独立性，否则可能还是会与其他变量存在相关性

**Picking structure** As we saw, there are many structures that are consistent with the same set of independencies. One successful approach is to choose a structure that reﬂects the causal order and dependencies, so that causes are parents of the efect. Such structures tend to work well. Either because of some real locality of inﬂuence in the world, or because of the way people perceive the world, causal graphs tend to be sparser. It is important to stress that the causality is in the world, not in our inference process. For example, in an automobile insurance network, it is tempting to put Previous-accident as a parent of Good-driver, because that is how the insurance company thinks about the problem. This is not the causal order in the world, because being a bad driver causes previous (and future) accidents. In principle, there is nothing to prevent us from directing the edges in this way. However, a noncausal ordering often requires that we introduce many additional edges to account for induced dependencies (see section 3.4.1). 
> 常用的结构是反映了世界中的因果关系的结构
> 要注意我们指的因果关系是执因索果，而不是推理过程中的执果索引（不然就反过来了）

One common approach to constructing a structure is a backward construction process. We begin with a variable of interest, say Lung-Cancer. We then try to elicit a prior probability for that variable. If our expert responds that this probability is not determinable, because it depends on other factors, that is a good indication that these other factors should be added as parents for that variable (and as variables into the network). For example, we might conclude using this process that Lung-Cancer really should have Smoking as a parent, and (perhaps not as obvious) that Smoking should have Gender and Age as a parent. This approach, called extending the conversation , avoids probability estimates that result from an average over a heterogeneous population, and therefore leads to more precise probability estimates. 
> 构建一个结构的常用方法是反向构建过程
> 我们从一个感兴趣的变量开始，然后尝试得到该变量的先验概率，如果该变量的先验概率是不可决定的（因为它依赖于其他因素），则我们再将其他因素加入为该变量的父节点

When determining the structure, however, we must also keep in mind that approximations are inevitable. For many pairs of variables, we can construct a scenario where one depends on the other. For example, perhaps Difculty depends on Intelligence, because the professor is more likely to make a class difcult if intelligent students are registered. In general, **there are many weak inﬂuences that we might choose to model, but if we put in all of them, the network can become very complex.** Such networks are problematic from a representational perspective: they are hard to understand and hard to debug, and eliciting (or learning) parameters can get very difcult. Moreover, as reasoning in Bayesian networks depends strongly on their connectivity (see section 9.4), adding such edges can make the network too expensive to use. 
> 在决定结构时，我们需要知道近似是不可避免的
> 对于许多对的变量，我们可以构造出一个变量依赖于另一个变量的场景，一般情况下，我们可能会选择建模许多弱的依赖，但如果我们将它们都放在网络中，网络就会变得非常复杂
> 过于复杂的网络是有问题的，它们难以理解、debug 以及难以学习参数，同时边太多的网络的推理也会太昂贵

This ﬁnal consideration may lead us, in fact, to make approximations that we know to be wrong. For example, in networks for fault or medical diagnosis, the correct approach is usually to model each possible fault as a separate random variable, allowing for multiple failures. However, such networks might be too complex to perform efective inference in certain settings, and so we may sometimes resort to a single fault approximation , where we have a single random variable encoding the primary fault or disease. 
> 为了避免过于复杂的情况，我们可能需要进行近似，例如，仅仅建模主要的因素为随机变量

**Picking probabilities** One of the most challenging tasks in constructing a network manually is eliciting probabilities from people. This task is somewhat easier in the context of causal models, since the parameters tend to be natural and more interpretable. Nevertheless, people generally dislike committing to an exact estimate of probability. 

One approach is to elicit estimates qualitatively, using abstract terms such as “common,” “rare,” and “surprising,” and then assign these to numbers using a predeﬁned scale. This approach is fairly crude, and often can lead to misinterpretation. There are several approaches developed for assisting in eliciting probabilities from people. For example, one can visualize the probability of the event as an area (slice of a pie), or ask people how they would compare the probability in question to certain predeﬁned lotteries. Nevertheless, probability elicitation is a long, difcult process, and one whose outcomes are not always reliable: the elicitation method can often inﬂuence the results, and asking the same question using diferent phrasing can often lead to signiﬁcant diferences in the answer. For example, studies show that people’s estimates for an event such as “Death by disease” are signiﬁcantly lower than their estimates for this event when it is broken down into diferent possibilities such as “Death from cancer,” “Death from heart disease,” and so on. 
> 一种评估概率的方式是使用类似 “common”，“rare”，“suprising” 等词，然后使用预定义的范围为它们赋值概率，这个方法较粗略

How important is it that we get our probability estimates exactly right? In some cases, small errors have very little efect. For example, changing a conditional probability of 0.7 to 0.75 generally does not have a signiﬁcant efect. Other errors, however, can have a signiﬁcant efect: 
> 一些情况下，概率估计的些小偏差对于结果没有太大影响
> 但其他类别的概率误差则会有很大的影响

- **Zero probabilities** : A common mistake is to assign a probability of zero to an event that is extremely unlikely, but not impossible. The problem is that **one can never condition away a zero probability, no matter how much evidence we get. When an event is unlikely  but not impossible, giving it probability zero is guaranteed to lead to irrecoverable errors.** For example, in one of the early versions of the the Pathﬁnder system (box 3. D), 10 percent of the misdiagnoses were due to zero probability estimates given by the expert to events that were unlikely but not impossible. As a general rule, very few things (except deﬁnitions) have probability zero, and we must be careful in assigning zeros. 
> 0概率：一个常见的错误就是给一个非常罕见的事件赋值零概率，但这导致的问题是我们不能以零概率的变量为条件，因此除非一个事件不可能，赋值零概率会导致不可恢复的错误

- **Orders of magnitude**: Small diferences in very low probability events can make a large diference to the network conclusions. Thus, a (conditional) probability of $10^{-4}$ is very diferent from $10^{-5}$ . 
> 数量级：对于非常小的概率，虽然数量级差异不会在取值上呈现出太大差异，但是会对网络的推理产生很大影响

- **Relative values:** The qualitative behavior of the conclusions reached by the network — the value that has the highest probability — is fairly sensitive to the relative sizes of $P(x\mid y)$ for diferent values $y$ of $\mathrm{Pa}_{X}$ . For example, it is important that the network encode correctly that the probability of having a high fever is greater when the patient has pneumonia than when he has the ﬂu.  sensitivity analysis  medical diagnosis expert system  Pathﬁnder 
> 相对值：由网络推理出的行为（概率值最高的事件）对于 $P (x\mid y)$ 中不同 $y \in \text{Pa}_x$ 的相对值是较为敏感的

A very useful tool for estimating network parameters is sensitivity analysis , which allows us to determine the extent to which a given probability parameter afects the outcome. This process allows us to evaluate whether it is important to get a particular CPD entry right. It also helps us ﬁgure out which CPD entries are responsible for an answer to some query that does not match our intuitions. 
> 评估网络参数的一个有用工具是敏感性分析，这允许我们决定给定的概率影响结果的程度

Box 3. D — Case Study: Medical Diagnosis Systems. One of the earliest applications of Bayesian networks was to the task of medical diagnosis . In the 1980s, a very active area of research was the construction of expert systems — computer-based systems that replace or assist an expert in per- forming a complex task. One such task that was tackled in several ways was medical diagnosis. This task, more than many others, required a treatment of uncertainty, due to the complex, nondeter- ministic relationships between ﬁndings and diseases. Thus, it formed the basis for experimentation with various formalisms for uncertain reasoning. 

The Pathﬁnder expert system was designed by Heckerman and colleagues (Heckerman and Nath- wani 1992a; Heckerman et al. 1992; Heckerman and Nathwani 1992b) to help a pathologist diagnose diseases in lymph nodes. Ultimately, the model contained more than sixty diferent diseases and around a hundred diferent features. It evolved through several versions, including some based on non probabilistic formalisms, and several that used variants of Bayesian networks. Its diagnostic ability was evaluated over real pathological cases and compared to the diagnoses of pathological experts. 

One of the ﬁrst models used was a simple naive Bayes model, which was compared to the models based on alternative uncertainty formalisms, and judged to be superior in its diagnostic ability. It therefore formed the basis for subsequent development of the system. 

The same evaluation pointed out important problems in the way in which parameters were elicited from the expert. First, it was shown that 10 percent of the cases were diagnosed incorrectly, because the correct disease was ruled out by a ﬁnding that was unlikely, but not impossible, to manifest in that disease. Second, in the original construction, the expert estimated the probabilities P ( Finding | Disease ) by ﬁxing a single disease and evaluating the probabilities of all its ﬁndings. 

It was found that the expert was more comfortable considering a single ﬁnding and evaluating its probability across all diseases. This approach allows the expert to compare the relative values of the same ﬁnding across multiple diseases, as described in box 3.C. 

With these two lessons in mind, another version of Pathﬁnder — Pathﬁnder III — was con- structed, still using the naive Bayes model. Finally, Pathﬁnder IV used a full Bayesian network, with a single disease hypothesis but with dependencies between the features. Pathﬁnder IV was con- structed using a similarity network (see box 5. B), signiﬁcantly reducing the number of parameters that must be elicited. Pathﬁnder IV, viewed as a Bayesian network, had a total of around 75,000 parameters, but the use of similarity networks allowed the model to be constructed with fewer than 14,000 distinct parameters. Overall, the structure of Pathﬁnder IV took about 35 hours to deﬁne, and the parameters 40 hours. 

A comprehensive evaluation of the performance of the two models revealed some important insights. First, the Bayesian network performed as well or better on most cases than the naive Bayes model. In most of the cases where the Bayesian network performed better, the use of richer dependency models was a contributing factor. As expected, these models were useful because they address the strong conditional independence assumptions of the naive Bayes model, as described in box 3.A. Somewhat more surprising, they also helped in allowing the expert to condition the probabilities on relevant factors other than the disease, using the process of extending the conversation described in box 3. C, leading to more accurate elicited probabilities. Finally, the use of similarity networks led to more accurate models, for the smaller number of elicited parameters reduced irrelevant ﬂuctuations in parameter values (due to expert inconsistency) that can lead to spurious dependencies. 

Overall, the Bayesian network model agreed with the predictions of an expert pathologist in 50 / 53 cases, as compared with 47 / 53 cases for the naive Bayes model, with signiﬁcant therapeutic implications. A later evaluation showed that the diagnostic accuracy of Pathﬁnder IV was at least as good as that of the expert used to design the system. When used with less expert pathologists, the system signiﬁcantly improved the diagnostic accuracy of the physicians alone. Moreover, the system showed greater ability to identify important ﬁndings and to integrate these ﬁndings into a correct diagnosis. 

Unfortunately, multiple reasons prevent the widespread adoption of Bayesian networks as an aid for medical diagnosis, including legal liability issues for misdiagnoses and incompatibility with the physicians’ workﬂow. However, several such systems have been ﬁelded, with signiﬁcant success. Moreover, similar technology is being used successfully in a variety of other diagnosis applications (see box 23. C). 
## 3.3 Independencies in Graphs 
Dependencies and independencies are key properties of a distribution and are crucial for under- standing its behavior. As we will see, independence properties are also important for answering queries: they can be exploited to reduce substantially the computation cost of inference. There- fore, it is important that our representations make these properties clearly visible both to a user and to algorithms that manipulate the BN data structure. 

As we discussed, a graph structure $\mathcal{G}$ encodes a c ain set of cond onal independence assumptions $\mathcal{T}_{\ell}(\mathcal{G})$ . Knowing only that a distribution P factorizes over G , we can conclude that it satisﬁes $\mathcal{T}_{\ell}(\mathcal{G})$ . An imm iate question is whether there are other independencies that we can “rea of” directly from G . That is, are there other independencies that hold for every distribution P that factorizes over ? 
> 我们已经讨论过，一个图结构 $\mathcal G$ 编码了一些条件独立假设 $\mathcal I (\mathcal G)$，我们现在探究对于可以根据 $\mathcal G$ 分解的分布 $P$，是否存在其他独立性可以从图中得到
### 3.3.1 D-separation 
Our aim in this section is to understand when we can guara tee that an independence $(\boldsymbol{\textbf{X}}\perp$ $Y\mid Z)$ holds in a distribution associated with a BN structure G . To understand when a property is guaranteed to hold, it helps to consider its converse: “Can we imagine a case where it does not?” Thus, we focus our discussion on analyzing when it is possible that $X$ can inﬂuence $Y$ given $Z$ . If we construct an example where this inﬂuence occurs, then the converse property $(X\ \bot\ Y\ |\ Z)$ cannot f the distributions t ctorize over $\mathcal{G}$ , and hence the independence property ( $(X\perp Y\mid Z)$ cannot follow from I $\mathcal{T}_{\ell}(\mathcal{G})$ G . 
> 我们本节的目标是理解对于一个和 $\mathcal G$ 相关的分布，我们什么时候可以保证独立性 $(X \perp Y \mid Z)$ 存在

We therefore begin with an intuitive case analysis: Here, we try to understand when an observation regarding a variable $X$ can possibly change our beliefs about $Y$ , in the presence of evidence about the variables $Z$ . Although this analysis will be purely intuitive, we will show later that our conclusions are actually provably correct. 
> 我们尝试理解在给定 $Z$ 的情况下，什么时候关于变量 $X$ 的观测会可能地改变我们关于 $Y$ 的信念

**Direct connection** We begin with the simple case, when $X$ and $Y$ are directly connected via an edge, say $X\rightarrow Y$ . For any net rk st cture $\mathcal{G}$ that contains the edge $X\rightarrow Y$ , it is possible to construct a distribution where X and Y are correlated regardless of any evidence about any of the other variables in the network. In other words, if $X$ and $Y$ are directly connected, we can always get examples where they inﬂuence each other, regardless of $Z$ . 
> 当 $X, Y$ 直接通过一条边连接，显然二者是直接相关的，对于任意包含 $X\rightarrow Y$ 的网络 $\mathcal G$，我们总是构造出一个 X Y 相互关联的分布，无论网络中的其他变量如何，因此无论 $Z$ 是否给定，二者都会相互影响

In particular, assume that $V a l(X)\,=\,V a l(Y)$ ; we can simply set $X\,=\,Y$ . That, by itself, however, is not enough; if (given the evidence $Z$ ) $X$ deterministic ally takes some particular value, say 0 , then $X$ and $Y$ both deterministic ally take that value, and are uncorrelated. We therefore set the network so that $X$ is (for example) uniformly distributed, regardless of the values of any of its parents. This construction sufces to induce a correlation between $X$ and $Y$ , regardless of the evidence. 

**Indirect connection** Now consider the more complicated case when $X$ and $Y$ are not directly connected, but there is a trail between them in the graph. We begin by considering the simplest such case: a three-node network, where $X$ and $Y$ are not directly connected, but where there is a trail between them via $Z$ . It turns out that this simple case is the key to understanding the whole notion of indirect interaction in Bayesian networks. 

There are four cases where $X$ and $Y$ are connected via $Z$ , as shown in ﬁgure 3.5. The ﬁrst two correspond to causal chains (in either direction), the third to a common cause, and the fourth to a common efect. We analyze each in turn. 
> 考虑 XY 没有直接连接，但是之间有一条迹，
> 例如 $X, Y$ 没有直接连接，而是通过 $Z$ 连接，这样的连接一共有4种方式

![[Probabilistic Graph Theory-Fig3.5.png]]

**Indirect causal effect** (ﬁgure 3.5a). To gain intuition, let us return to the Student example, where we had a causal trail $I\to G\to L$ . Let us begin with the case where $G$ is not observed. Intuitively, if we observe that the student is intelligent, we are more inclined to believe that he gets an A, and therefore that his recommendation letter is strong. In other words, the probability of these latter events is higher conditioned on the observation that the student is intelligent.  

In fact, we saw precisely this behavior in the distribution of ﬁgure 3.4. Thus, in this case, we believe that $X$ can inﬂuence $Y$ via $Z$ . 

Now assume that $Z$ is observed, that is, $Z\in Z$ . As we saw in our analysis of the Student example, if we observe the student’s grade, then (as we assumed) his intelligence no longer inﬂuences his letter. In t, the local indep denc s f this network tell us that $(L\perp I\mid G)$ . Thus, we conclude that X cannot inﬂuence $Y$ via Z if Z is observed. 
> 1. 间接的因果影响：在 $Z$ 还未被观察到的情况下，$X$ 通过 $Z$ 影响 $Y$，即 $X\rightarrow Z \rightarrow Y$，当 $Z$ 已经被观察到，$X$ 和 $Y$ 条件独立

**Indirect evidential efect** (ﬁgure 3.5b). Returning to the Student example, we have a chain $I\,\rightarrow\, G\,\rightarrow\, L$ . We have already seen that observing a strong recommendation letter for the student changes our beliefs in his intelligence. Conversely, once the grade is observed, the letter gives no additional information about the student’s intelligence. Thus, our analysis in the case $Y\rightarrow Z\rightarrow X$ here is identical to the causal case: $X$ can inﬂuence $Y$ via $Z$ , but only if $Z$ is not observed. The similarity is not surprising, as dependence is a symmetrical notion. Speciﬁcally, if $(X\perp Y)$ does not hold, then $(Y\perp X)$ does not hold either. 
> 2. 间接的证据影响，和第一类相对称，依赖是一个对称的概念，也就是说如果 $X\perp Y$ 不成立的话，$Y\perp X$ 也是不成立的
> 因此，在 $Z$ 没有被观察到的情况下， $Y$ 或通过 $Z$ 影响 $X$，即 $Y \rightarrow Z \rightarrow X$

**Common cause** (ﬁgure 3.5c). This case is one that we have analyzed extensively, both within the simple naive Bayes model of section 3.1.3 and within our Student example. Our example has the student’s intelligence $I$ as a parent of his grade $G$ and his SAT score $S$ . As we discussed, $S$ and $G$ are correlated in this model, in that observing (say) a high SAT score gives us information about a student’s intelligence and hence helps us predict his grade. However, once we observe $I$ , this correlation disappears, and $S$ gives us no additional information about $G$ . Once again, for this network, this conclusion follows from the local independence assumption for the node $G$ (or for $S$ ). Thus, our conclusion here is identical to the previous two cases: $X$ can inﬂuence $Y$ via $Z$ if and only if $Z$ is not observed. 
> 3. 共同成因：当且仅当 $Z$ 没有被观察到时，$X$ 可以通过 $Z$ 影响 $Y$，当 $Z$ 被观察到时，$X, Y$ 之间条件独立

**Common efect** (ﬁgure 3.5d). In all of the three previous cases, we have seen a common pattern: $X$ can inﬂuence $Y$ via $Z$ if and only if $Z$ is not observed. Therefore, we might expect that this pattern is universal, and will continue through this last case. Somewhat surprisingly, this is not the case. Let us return to the Student example and consider $I$ and $D$ , which are parents of $G$ . When $G$ is not observed, we have that $I$ and $D$ are independent. In fact, this conclusion follows (once again) from the local independencies from the network. Thus, in this case, inﬂuence cannot “ﬂow” along the trail $X\rightarrow Z\leftarrow Y$ if the intermediate node $Z$ is not observed. 

On the other hand, consider the behavior when $Z$ is observed. In our discussion of the Student example, we analyzed precisely this case, which we called intercausal reasoning; we showed, for example, that the probability that the student has high intelligence goes down dramatically when we observe that his grade is a C $\scriptstyle (G\;=\; g^{3})$ ), but then goes up when we observe that the class is a difcult one $D=d^{1}$ . Thus, in presence of the evidence $G=g^{3}$ , we have that $I$ and $D$ are correlated. 
> 4. 共同影响：当 $Z$ 没有被观测到时，$X$ 和 $Y$ 是相互独立的，如果 $Z$ 被观测到，则 $X, Y$ 是相关的
> 并且，即便 $Z$ 没有被直接观测到，而是 $Z$ 的子孙节点被观测到，则 $X, Y$ 也会是相关的

Let us consider a variant of this last case. Assume that we do not observe the student’s grade, but we do observe that he received a weak recommendation letter $(L\,=\, l^{0}$ ). Intuitively, the same phenomenon happens. The weak letter is an indicator that he received a low grade, and therefore it sufces to correlate $I$ and $D$ . 

When inﬂuence can ﬂow from $X$ to $Y$ via $Z$ , we say that the trail $X\rightleftharpoons Z\rightleftharpoons Y$ is active . The results of our analysis for active two-edge trails are summarized thus: 
> 当影响可以从 $X$ 流经 $Y$ 到 $Z$，我们说迹 $X\rightleftharpoons Z \rightleftharpoons Y$ 是活跃的，我们总结对于双边迹的分析结果如下

- Causal trail $X\rightarrow Z\rightarrow Y$ : active if and only if $Z$ is not observed.
- Evidential trail $X\leftarrow Z\leftarrow Y$ : active if and only if $Z$ is not observed.
- Common cause $X\leftarrow Z\rightarrow Y$ : active if and only if $Z$ is not observed.
- Common efect $X\rightarrow Z\leftarrow Y$ : active if and only if either $Z$ or one of $Z$ ’s descendants is observed. 
> 因果迹：$X\rightarrow Z \rightarrow Y$，当仅当 $Z$ 没有被观察到时活跃
> 证据迹：$X\leftarrow Z\leftarrow Y$，当仅当 $Z$ 没有被观察到时活跃
> 共同原因：$X\leftarrow Z \rightarrow Y$，当仅当 $Z$ 没有被观察到时活跃
> 共同影响：$X\rightarrow Z \leftarrow Y$，当仅当 $Z$ 或者 $Z$ 的其中一个子孙被观察到时活跃

A structure where $X\rightarrow Z\leftarrow Y$ (as in ﬁgure 3.5d) is also called a $\nu$ -structure . 
> 结构为 $X \rightarrow Z \leftarrow Y$ 的结构（共同影响）称为 v-结构

It is useful to view probabilistic inﬂuence as a ﬂow in the graph. Our analysis here tells us when inﬂuence from $X$ can “ﬂow” through $Z$ to afect our beliefs about $Y$ . 

**General Case** Now co der case of a longer trail $X_{1}\,\,\rightleftharpoons\,\,\cdot\cdot\,\,\rightleftharpoons\,\, X_{n}$ . Intuitively, for inﬂuence to “ﬂow” from $X_{1}$ to $X_{n}$ , it needs to ﬂow through every single node on the trail. In other words, $X_{1}$ can inﬂuence $X_{n}$ if every two-edge trail $X_{i-1}\rightleftharpoons X_{i}\rightleftharpoons X_{i+1}$ along the trail allows inﬂuence to ﬂow. We can summarize this intuition in the following deﬁnition: 
> 考虑一个长的迹 $X_{1}\,\,\rightleftharpoons\,\,\cdot\cdot\,\,\rightleftharpoons\,\, X_{n}$，如果影响需要从 $X_1$ 流到 $X_n$ ，则它需要流过迹中的每一个节点

**Deﬁnition 3.6** active trail
Let $\mathcal{G}$ be a BN str $X_{1}\,\rightleftharpoons\,.\,.\,\rightleftharpoons\, X_{n}$ a rail in $\mathcal{G}$ . Let $Z$ be a subset of observed variables . The trail $X_{1}\rightleftharpoons...=X_{n}$ is active given Z if 

-  W never we have a $\nu$ -structure $X_{i-1}\to X_{i}\gets X_{i+1}$ , then $X_{i}$ or one of its descendants are in Z ;
- no other node along the trail is in $Z$ . 

> 定义：
> 对于贝叶斯网络中的一个迹 $X_1 \rightleftharpoons \dots \rightleftharpoons X_n$，令 $Z$ 表示观测到的变量的一个子集，给定 $Z$ 时，迹 $X_1 \rightleftharpoons \dots \rightleftharpoons X_n$ 在以下情况下活跃：
> - 对于其中的 v-结构 $X_{i-1}\rightarrow X_i \leftarrow X_{i+1}$，满足 $X_i$ 或 $X_i$ 的一个子孙在 $Z$ 中
> - 迹中没有其他的节点在 $Z$ 中

Note that if $X_{1}$ or $X_{n}$ are in $Z$ the trail is not active. 
> 注意如果 $X_1$ 或 $X_n$ 在 $Z$ 中，则这个迹也是不活跃的

In our Stude ve that $D\to G\leftarrow I\to S$ is not an active trail for $Z=\emptyset$ , because the v-structure $D\rightarrow G\leftarrow I$ → ← not activated. That same trail is active when $Z=\{L\}$ , because observing the descendant of G activates he v-structure. O and, when $Z=\{L, I\}$ , the trail is not active, because observing I blocks the trail $G\gets I\to S$ . 

What about graphs where there is more than one trail between two nodes? Our ﬂow intuition continues to carry through: one node can inﬂuence another if there is any trail along which inﬂuence can ﬂow. Putting these intuitions together, we obtain the notion of $d$ -separation , which provides us with a notion of separation between nodes in a directed graph (hence the term d-separation, for directed separation): 
> 对于两个节点之间存在多个迹的情况，则只要其中任意一条迹允许影响流过，则两个节点就是相关的

**Deﬁnition 3.7** 
Let $X, Y,$ $Z$ be three sets of nodes in $\mathcal{G}$ . We say that $X$ and $Y$ $\mathrm{d}$ para n $Z$ , d oted $\operatorname{d-sep}_{\mathcal{G}}(X; Y\mid Z).$ , if there is no active trail between any node $X\in X$ ∈ and $Y\in Y$ ∈ given Z . We use $\mathcal{Z}(\mathcal{G})$ to denote the set of independencies that correspond to d-separation: 

$$
{\mathcal{I}}({\mathcal{G}})=\{(X\perp Y\mid Z)\;:\;{\mathrm{d-sep}}_{\mathcal{G}}(X; Y\mid Z)\}.
$$ 
> 定义：
> 令 $X, Y, Z$ 是 $\mathcal G$ 中的三个节点集，如果给定 $Z$ 时，对于任意节点 $x \in X$ 和 $y \in Y$ 之间都不存在活跃的迹，则我们称 $X, Y$ 在给定 $Z$ 时是 d-seperation 的，记作 $\text{d-sep}_{\mathcal G}(X; Y\mid Z)$
> 因为不存在活跃的迹，故节点之间就是条件独立的，因此 $\mathcal I (\mathcal G)$ 实际上就是将图 $\mathcal G$ 中所有的 d-seperation 的节点表示为 $\perp$

This set is also called the set of global Markov independencies . The similarity between the nota- tion $\mathcal{Z}(\mathcal{G})$ and our notation $\mathcal{Z}(P)$ is not coincidental: As we discuss later, the in pendencies in $\mathcal{Z}(\mathcal{G})$ are precisely those that are guaranteed to hold for every distribution over G . 
> 集合 $\mathcal I (\mathcal G)$ 也称为全局的 Markov 独立集合
### 3.3.2 Soundness and Completeness 
So far, our deﬁnition of d-separation has been based on our intuitions regarding ﬂow of inﬂuence, and on our one example. As yet, we have no guarantee that this analysis is “correct.” Perhaps there is a distribution over the BN where $X$ can inﬂuence $Y$ despite the fact that all trails between them are blocked. 

Hence, the ﬁrst property we want to ensure for $\mathrm{d}$ -separation as a method for determining independence is soundness : if we ﬁnd that two nodes $X$ and $Y$ are d-separated given some $Z$ , then we are guaranteed that they are, in fact, conditionally independent given $Z$ . 
> 我们首先需要保证 d-seperation 这个概念的可靠性：如果我们在给定 $Z$ 的情况下找到两个节点 $X, Y$ 时 d-seperation 的，则我们可以保证这两个节点在给定 $Z$ 的情况下是条件独立的

**Theroem 3.3**
If a distribution $P$  factorizes according to $\mathcal G$, then $\mathcal I (\mathcal G)\subseteq \mathcal I (\mathcal P)$
> 定理：
> 如果一个分布 $P$ 根据 $\mathcal G$ 分解，则 $\mathcal I (\mathcal G)\subseteq \mathcal I (\mathcal P)$

In other words, any independence reported by $\mathrm{d}$ -separation is satisﬁed by the underlying dis- tribution. The proof of this theorem requires some additional machinery that we introduce in chapter 4, so we defer the proof to that chapter (see section 4.5.1.1). 
> 换句话说，任意由 d-seperation 表示的独立性都会被 underlying 的分布满足

A second desirable property is the complementary one — completeness : d-separation detects all possible independencies. More precisely, if we have that two variables $X$ and $Y$ are indepen- dent given $Z$ , then they are d-separated. A careful examination of the completeness property reveals that it is ill deﬁned, inasmuch as it does not specify the distribution in which $X$ and $Y$ are independent. 
> 我们还需要保证 d-seperation 这个概念的完整性：d-sepration 检测到所有可能的独立性，也就是说，如果两个节点 XY 在给定 Z 时条件独立，则它们一定是 d-seperation 的

To formalize this property, we ﬁrst deﬁne the following notion: 

**Deﬁnition 3.8** faithful 
A distribution $P$ is faithful to $\mathcal{G}$ i whenever $(X\perp Y\mid Z)\in{\mathcal{Z}}(P)$ , then $\operatorname{d-sep}_{\mathcal{G}}(X; Y\mid Z)$ . In other words, any independence in P is reﬂected in the d-separation properties of the graph. 
> 定义：
> 对于分布 P，只要 $(X\perp Y\mid Z) \in \mathcal I (P)$，就有 $\text{d-sep}_{\mathcal G}(X; Y\mid Z)$，则我们称分布 P 是忠实于图 $\mathcal G$ 的，换句话说，$P$ 中的任意独立性都在图 $\mathcal G$ 中的 d-sepration 中得到反映，也就是图中的条件独立性包含了 $P$ 中的条件独立性

We can now provide one candidate formalization of the completeness property is as follows: 

- For y distrib tion $P$ that fact zes ver $\mathcal{G}$ , e hav that $P$ is faithful to $\mathcal{G}$ ; that is, if $X$ and $Y$ are not d-se arated given Z in G , then X and $Y$ are dependent in all distributions P that factorize over G . 

> 我们现在将 d-seperation 的完整性描述如下：
> 对于任意根据 $\mathcal G$ 分解的分布 P，P 都忠实于 $\mathcal G$
> 也就是说，如果 XY 在给定 $\mathcal G$ 中的 Z 时不是 d-seperation 的，因为所有根据 $\mathcal G$ 分解的分布 $P$ 都忠实于 $\mathcal G$，则对于所有根据 $\mathcal G$ 分解的分布 $P$，都不存在 $(X\perp Y\mid Z)$，也就是 $X, Y$ 是相关的

This property is the obvious converse to our notion of soundness: If true, the two together would imply that, for any $P$ that factorizes over $\mathcal{G}$ , we have that $\mathcal{Z}(P)=\mathcal{Z}(\mathcal{G})$ . Unfortunately, this highly desirable property is easily shown to be false: Even if a distribution factorizes over $\mathcal{G}$ , it can still contain additional independencies that are not reﬂected in the structure. 
> 可靠性：$\mathcal G$ 中有 d-seperatoin，则根据 $\mathcal G$ 分解的 $P$ 中有对应的条件独立性
> 完整性：根据 $\mathcal G$ 分解的 $P$ 中存在条件独立性，则 $\mathcal G$ 中有对应的 d-seperation
> 可靠性和完整性同时成立时，就表示对于任意在 $\mathcal G$ 上分解的 $P$，有 $\mathcal I (P) = \mathcal I (\mathcal G)$，但这个性质往往不容易成立，往往在根据 $\mathcal G$ 分解的 $P$ 中会存在额外的独立性条件

Example 3.3 Consider a distribution $P$ over two variables $A$ and $B$ , where $A$ and $B$ are independent. One possible I-map for $P$ is the network $A\rightarrow B$ . For example, we can set the CPD for B to be 

This example clearly violates e ﬁrst candidate deﬁnition of completeness, because the graph $\mathcal{G}$ is an I-map for the distribution P , yet there are independencies that hold for this distribution but do not follow from $d$ -separation. In fact, these are not independencies that we can hope to discover by examining the network structure. 

Thus, the completeness property does not hold for this candidate deﬁnition of completeness. We therefore adopt a weak er yet still useful deﬁnition: 
> 我们考虑为 completeness 采用一个更弱的定义

- If $(X\perp Y\mid Z)$ i ll dis ibutions $P$ $\mathcal{G}$ , then $d{\mathfrak{-s e p}}_{\mathcal{G}}(X; Y\mid Z)$ . And the contrapositive: If X nd Y $Y$ are not d-separated given Z in $\mathcal{G}$ , then X and Y are dependent in some distribution P that factorizes over $\mathcal{G}$ . 
> 如果 $(X\perp Y \mid Z)$ 在所有根据 $\mathcal G$ 分解的分布 $P$ 中成立，则 $\text{d-sep}_{\mathcal G}(X; Y\mid Z)$ 成立，同时，如果在 $\mathcal G$ 中 $X, Y$ 在给定 $Z$ 时不是 d-seperated，则 $X, Y$ 在某个根据 $\mathcal G$ 分解的 $P$ 中是相关的

Using this deﬁnition, we can show: 

**Theorem 3.4** 
Let $\mathcal{G}$ be a BN s cture. If $X$ and $Y$ not d-separated gi en $Z$ in $\mathcal{G}$ , then $X$ and $Y$ are dependent given Z in some distribution P that factorizes over G . 
> 定理：
> 如果在 $\mathcal G$ 中 $X, Y$ 在给定 $Z$ 时不是 d-seperated 的（相互依赖），则 $X, Y$ 在*某个*根据 $\mathcal G$ 分解的分布 $P$ 中是相关的，也就是 $P$ 中没有额外的独立性

**Proof** The proof constructs a distribution $P$ that makes $X$ and $Y$ correlated. The construction is roughly as follows. As $X$ and $Y$ are not d-separated, there exists an active trail $U_{1},\dots, U_{k}$ between them. We deﬁne CPDs for the variables on the trail so as to make each pair $U_{i}, U_{i+1}$ correlated; in the case of a v-structure $U_{i}\,\to\, U_{i+1}\,\leftarrow\, U_{i+2}$ , we deﬁne the CPD of $U_{i+1}$ so as to ensure correlation, and also deﬁne the CPDs of the path to some downstream evidence node, in a way that guarantees that the downstream evidence activates the correlation between $U_{i}$ and $U_{i+2}$ . All other CPDs in the graph are chosen to be uniform, and thus the construction guarantees that inﬂuence only ﬂows along this single path, preventing cases where the inﬂuence of two (or more) paths cancel out. The details of the construction are quite technical and laborious, and we omit them. 
> 证明：构造一个 $X, Y$ 相关的，且根据 $\mathcal G$ 分解的分布 $P$

We can view the completeness result as telling us that our deﬁniti n of $\mathcal{Z}(\mathcal{G})$ is the maximal one. For any independence assertion that not a consequence f d-separation in $\mathcal{G}$ , we can always ﬁnd a counterexample distribution P that factorizes over G . In fact, this result can be strengthened signiﬁcantly: 
> 我们认为完整性告诉了我们 $\mathcal I (\mathcal G)$ 的定义是最大的，也就是对于任意不是 $\mathcal G$ 中的 d-seperation 的结果的独立性断言，我们都可以找到一个根据 $\mathcal G$ 分解的反例 $P$ ($P$ 中不存在不满足 d-seperation 结果的独立性断言)
> 我们可以强化这一结果

**Theorem 3.5** 
For almost all distributions $P$ that factorize over $\mathcal{G}$ , that is, for all distributions except for a set of measure zero in the space of CPD parameter iz at ions, we have that $\mathcal{Z}(P)=\mathcal{Z}(\mathcal{G})$ . 
> 定理：
> 对于根据 $\mathcal G$ 分解的*几乎全部*分布 $P$（也就是除了 CPD 参数化空间中测度为零的集合），我们有 $\mathcal I (\mathcal G) = \mathcal I (P)$

This result strengthens theorem 3.4 in two distinct ways: First, whereas theorem 3.4 shows that any dependency in the graph can be found in some distribution, this new result shows that there exists a single distribution that is faithful to the graph, that is, where all of the dependencies in the graph hold simultaneously. Second, not only does this property hold for a single distribution, but it also holds for almost all distributions that factorize over $\mathcal{G}$ . 
> 定理3.4表明了图中的任意依赖都可以在某个分布中找到，该定理说明了存在忠实于图的分布（分布中的独立性被图中的独立性包含），并且事实上对于几乎所有根据 $\mathcal G$ 分解的分布，这一点都是成立的

**Proof** At a high level, the proof is based on the following argument: Each conditional inde- pendence assertion is a set of polynomial equalities over the space of CPD parameters (see exercise 3.13). A basic property of polynomials is that a polynomial is either identically zero or it is nonzero almost everywhere (its set of roots has measure zero). Theorem 3.4 implies that polynomials corresponding to assertions outside $\mathcal{Z}(\mathcal{G})$ cannot be entically zero, because they have at least one counterexample. Thus, the set of distributions P , which exhibit any one of these “spurious” independence assertions, has measure zero. The set of distributions that do not satisfy $\mathcal{Z}(P)=\mathcal{Z}(\mathcal{G})$ is the union of these separate sets, one for each spurious independence assertion. The union of a ﬁnite number of sets of measure zero is a set of measure zero, proving the result. 
> 证明：每个条件独立性声明都是在 CPD 的参数空间中的一个多项式等式集合，而多项式的一个基本性质就是要么处处为零要么几乎都非零（即其根集合的测度为零）
> 定理3.4说明不存在于 $\mathcal I (\mathcal G)$ 中的独立性断言不能处处为零，因为它们至少存在一个反例，那么要为零的话，它们的测度就是零，因此展现出 “虚假的” 独立性的分布 $P$ 的测度是零
> 不满足 $\mathcal I (\mathcal G) = \mathcal I (P)$ 的分布构成的集合就是这些包含了 “虚假的” 独立性断言的分布的集合的并集，而有限个测度为零的集合的并集的测度仍然是零
> 因此除了测度为零的分布以外，都有 $\mathcal I (\mathcal G) = \mathcal I (P)$

**These results state that for almost all parameter iz at ions $P$ of the graph $\mathcal{G}$ (that is, for almost all possible choices of CPDs for the variables), the d-separation test precisely characterizes the independencies that hold for $P$ .** In other words, even if we have a dis bution $P$ that satisﬁes more independencies than $\mathcal{Z}(\mathcal{G})$ , a slight perturbation of the CPDs of P will almost always eliminate these “extra” independencies. This guarantee seems to state that such independencies are always accidental, and we will never encounter them in practice. However, as we illustrate in example 3.7, there are cases where our CPDs have certain local structure that is not accidental, and that implies these additional independencies that are not detected by $\mathrm{d}$ -separation. 
> 这个结果说明了对于几乎所有 $\mathcal G$ 参数化的 $P$ (对于 $\mathcal G$ 中的变量的几乎所有可能条件概率分布的选择)，d-seperation 测试可以精确地表征 $P$ 中存在的独立性
> 换句话说，即便我们有满足比 $\mathcal I (\mathcal G)$ 中更多独立性的分布 $P$，一个对 $P$ 中的条件概率分布的轻微扰动会几乎总是消除这些“额外”的福利性
### 3.3.3 An Algorithm for d-Separation 
The notion of $\mathrm{d}$ -separation allows us to infer independence properties of a distribution $P$ that factorizes over $\mathcal{G}$ simply by examining the connectivity of $\mathcal{G}$ . However, in order to be useful, we need to be able to determine d-separation efectively. Our deﬁnition gives us a constructive solution, but a very inefcient one: We can enumerate all trails between $X$ and $Y$ , and check each one to see whether it is active. The running time of this algorithm depends on the number of trails in the graph, which can be exponential in the size of the graph. 
> 要知道根据 $\mathcal G$ 分解的 $P$ 中存在的独立性，我们需要知道 $\mathcal G$ 中存在的 d-seperation
> 目前，我们只知道通过列举出 $X, Y$ 中所有的路径，然后检查二者之间是否存在活跃的路径，以确定二者是否为 d-seperation，该算法的时间依赖于图中的路径数量，往往和图的大小成指数比

Fortunately, there is a much more efcient algorithm that requires only linear time in the size of the graph. The algorithm has two phases. We begin by traversing the graph bottom up, from the leaves to the roots, marking all nodes that are in $Z$ or that have descendants in $Z$ . Intuitively, these nodes will serve to enable v-structures. In the second phase, we traverse breadth-ﬁrst from $X$ to $Y$ , stopping the traversal along a trail when we get to a blocked node. A node is blocked if: (a) it is the “middle” node in a v-structure and unmarked in phase I, or (b) is not such a node and is in $Z$ . If our breadth-ﬁrst search gets us from $X$ to $Y$ , then there is an active trail between them. 
> 存在时间和图大小成线性关系的算法，算法分两阶段：
> 第一阶段：从下至上，从叶子到根节点遍历图，标记所有在 $Z$ 中或者在 $Z$ 中有子孙的节点，这些节点会被用于 enable v-structure
> 第二阶段：从 $X, Y$ 广度优先遍历，对于遍历时沿着的一个迹，在达到一个 blocked 的节点时停止对这个迹的追踪
> 一个 blocked 的节点满足：它是一个 v-structure 中心的节点，并且在第一阶段没有被标记 or 它不是一个 v-structure 中心的节点，并且它在 $Z$ 中
> 如果广度优先遍历最终可以从 $X$ 到 $Y$，则说明 $X, Y$ 之间存在一条活跃路径

The precise algorithm is shown in algorithm 3.1. The ﬁrst phase is straightforward. The second phase is more subtle. For efciency, and to avoid inﬁnite loops, the algorithm must keep track of all nodes that have been visited, so as to avoid visiting them again. However, in graphs with loops (multiple trails between a pair of nodes), an intermediate node $Y$ might be involved in several trails, which may require diferent treatment within the algorithm: 
> 为了避免无限循环，算法必须维护 visited，防止重复访问
> 但是对于本身有 loop 的图（即一对节点之间存在多个 trial），有时一个中间节点 $Y$ 会在多个 trial 中被包含，此时还需要进一步修改算法

Example 3.4 
Consider the Bayesian network of ﬁgure 3.6, where our task is to ﬁnd all nodes reachable from $X$ . Assume tha $Y$ bserved, that is, $Y\in Z$ . Assume that the lgorithm ﬁrst encounters $Y$ via the direct edge $Y\rightarrow X$ → . Any extension of this t $Y$ nd hence the algo hm stops the traversal alon this trail. However, the trail $X\leftarrow Z\rightarrow Y\leftarrow W$ ← ← is not blocked by Y $Y$ . Thus, when we encounter $Y$ or the second time via the edge $Z\rightarrow Y$ → , we should not ignore it. Therefore, after the ﬁrst visit to Y , we can mark it as visited for the purpose of trails coming in from children of $Y$ , but not for the purpose of trails coming in from parents of $Y$ . 

In general, we see that, for each node $Y$ , we must keep track separately of whether it has been visited from the top and whether it has been visited from the bottom. Only when both directions have been explored is the node no longer useful for discovering new active trails. 

Based on this intuition, we can now show that the algorithm achieves the desired result: 

**Theorem 3.6** 
The algor hm Re hable $({\mathcal{G}}, X, Z)$ returns the set of all nodes reachable from $X$ via trails that are active in G given Z . 

The proof is left as an exercise (exercise 3.14). 
### 3.3.4 I-Equivalence 
The notion of $\mathcal{Z}(\mathcal{G})$ speciﬁes a set of conditional independence assertions that are associated with a graph. This notion allows us to abstract away the details of the graph structure, viewing it purely as a speciﬁcation of independence properties. In particular, one important implication of this perspective is the observation that very diferent BN structures can actually be equivalent, in that they encode precisely the same set of conditional independence assertions. Consider, for example, the three networks in ﬁgure 3.5a, (b), (c). All three of them encode precisely the same independence assumptions: $(X\perp Y\mid Z)$ . 
> $\mathcal I (\mathcal G)$ 指定了所有和图相关的条件独立断言，我们借助它抽象化图的结构，将图仅仅视作一系列独立性规定
> 因此，结构细节上不同的贝叶斯网络实质上是可以等价的，也就是编码了完全相同的独立性断言

**Deﬁnition 3.9** I-equivalence 
> 定义：
> 如果两个 $\mathcal X$ 上的图结构满足 $\mathcal I (\mathcal K_1) = \mathcal I (\mathcal K_2)$，则它们是 I-equivalent 的
> 定义于 $\mathcal X$ 上的所有图可以根据 I-equivalence 关系划分为互斥且完备的 I-equivalence 等价类，也就是由 I-equivalence 等价关系导出的等价类

Note that the v-structure network in ﬁgure $3.5\mathrm{d}$ induces a very diferent set of $\mathrm{d}$ -separation assertions, and hence it does not fall into the same I-equivalence class as the ﬁrst three. Its I-equivalence class contains only that single network. 
> v-structure 网络的 I-equivalence 往往只有它自己

I-equivalence of two graphs immediately implies that any distribution $P$ that can be factorized over one of these graphs can be factorized over the other. **Furthermore, there is no intrinsic property of $P$ that would allow us to associate it with one graph rather than an equivalent one. This observation has important implications with respect to our ability to determine the directionality of inﬂuence.** In particular, although we can determine, for a distribution $P (X, Y)$ , whether $X$ and $Y$ are correlated, there is nothing in the distribution that can help us determine whether the correct structure is $X\rightarrow Y$ or $Y\rightarrow X$ . We return to this point when we discuss the causal interpretation of Bayesian networks in chapter 21. 
> 两个图是 I-equivalent 时，表明可以根据其中任意一图分解的 P 也可以根据其他图分解
> 显然，我们也应该将 P 和一个 I-equivalent 关联，而不是关联仅仅一张图，这个思想会帮助我们决定影响的方向性
> 一般我们可以知道 $X, Y$ 是相关的，但分布中不会有其他的信息帮助我们决定是 $X\rightarrow Y$ 还是 $Y\rightarrow X$

The d-separation criterion allows us to test for I-equivalence using a very simple graph-based algorithm. We start by considering the trails in the networks. 

**Deﬁnition 3.10** skeleton 
The skeleton of a Bayesian network graph $\mathcal{G}$ over $\mathcal{X}$ is an undirected graph over $\mathcal{X}$ that contains an edge $\{X, Y\}$ for every edge $(X, Y)$ in G . 
> 定义：
> $\mathcal X$ 上的贝叶斯网络的 skeleton 是 $\mathcal G$ 导出的无向图

In the networks of ﬁgure 3.7, the networks (a) and (b) have the same skeleton. 

If two networks have a common skeleton, then the set of trails between two variables $X$ and $Y$ is same in both networks. If they do not have a common skeleton, we can ﬁnd a trail in one network that does not exist in the other and use this trail to ﬁnd a counterexample for the equivalence of the two networks. 
> 两个 skeleton 的图中，$X, Y$ 之间的 trail 集合是一样的，否则，可以找到一张图内存在但另一张中不存在的 trail

Ensuring that the two networks have the same trails is clearly not enough. For example, the networks in ﬁgure 3.5 all have the same skeleton. Yet, as the preceding discussion shows, the network of ﬁgure $3.5\mathrm{d}$ is not equivalent to the networks of ﬁgure 3.5a–(c). The diference, is of course, the v-structure in ﬁgure $3.5\mathrm{d}$ . Thus, it seems that if the two networks have the same skeleton and exactly the same set of v-structures, they are equivalent. Indeed, this property provides a sufcient condition for I-equivalence: 

**Theorem 3.7** 
> 定理：
> 对于 $\mathcal X$ 是上的两张图 $\mathcal G_1, \mathcal G_2$，如果二者具有相同的 skeleton，并且具有相同的 v-structure 集合，则二者是 I-equivalent 的 (充分条件)

The proof is left as an exercise (see exercise 3.16).

Unfortunately, this characterization is not an equivalence: there are graphs that are Iequivalent but do not have the same set of v-structures. As a counterexample, consider complete graphs over a set of variables. Recall that a complete graph is one to which we cannot add additional arcs without causing cycles. Such graphs encode the empty set of conditional in- dependence assertions. Thus, any two complete graphs are I-equivalent. Although they have the same skeleton, they invariably have diferent v-structures. Thus, by using the criterion on theorem 3.7, we can conclude (in certain cases) only that two networks are I-equivalent, but we cannot use it to guarantee that they are not. 
> 存在 v-structure 集合不同但是 I-equivalent 的图
> 考虑一个完全图（完全图即再加任意一个边就会导致环），完全图中的条件独立性断言是空集，因此任意两个完全图都是 I-equivalence，然而它们的 v-structure 可以不同

We can provide a stronger condition that does correspond exactly to I-equivalence. Intuitively, the ique dependence pattern that we want to associate with a v-structure $X\rightarrow Z\leftarrow Y$ is that X and Y are independent (conditionally on their parents), but dependent given Z . If there is a direct edge between $X$ and $Y$ , as there was in our example of the complete graph, the ﬁrst part of this pattern is eliminated. 

**Deﬁnition 3.11** immorality covering edge 
A $\nu$ -structure $X\rightarrow Z\leftarrow Y$ immorality if there is no direct edge between $X$ and $Y$ . If there is such an edge, it is called a covering edge for the $\nu$ -structure. 
> 定义：
> 如果在 $X, Y$ 之间没有直接的边，则 v-structure $X\rightarrow Z \leftarrow Y$ 就是一个 immorality，如果 $X, Y$ 之间存在直接连接的边，则该边被称为 v-structure 的 covering edge

Note that not every v-structure is an immorality, so that two networks with the same immoralities do not necessarily have the same v-structures. For example, two diferent complete directed graphs always have the same immoralities (none) but diferent v-structures. 
> 因为不是所有的 v-structure 都是 immorality，因此两个具有相同 immorality 的网络并不必要是具有相同 v-structure 的
> 例如，两个不同的完全有向图总是有相同的 immorality (None)，但是可以有不同的 v-structure

**Theorem 3.8** 
Let $\mathcal{G}_{1}$ and $\mathcal{G}_{2}$ be two graphs over $\mathcal{X}$ . Then $\mathcal{G}_{1}$ and $\mathcal{G}_{2}$ have the same skeleton and the same set of immoralities if and only if they are I-equivalent. 
> 定理：
> 令 $\mathcal G_1, \mathcal G_2$ 是 $\mathcal X$ 上的两个图，则当且仅当它们是 I-equivalent 时，$\mathcal G_1, \mathcal G_2$ 具有相同的 skeleton 和相同的 immorality 集合

The proof of this (more difcult) result is also left as an exercise (see exercise 3.17). 
We conclude with a ﬁnal characterization of I-equivalence in terms of local operations on the graph structure. 

**Deﬁnition 3.12** covered edge 
An edge $X\rightarrow Y$ in a graph $\mathcal G$ is said to be covered if $\text{Pa}_Y^{\mathcal G} = \text{Pa}_X^{\mathcal G} \cup \{X\}$
> 定义：
> 如果 $Y$ 的父节点就是 $X$ 的父节点加上 $X$，则边 $X\rightarrow Y$ 就是 covered edge

**Theorem 3.9** 
Two graphs $\mathcal{G}$ and ${\mathcal{G}}^{\prime}$ are $I_{\cdot}$ -equivalent if a d only if there exists a sequence of n orks $\mathcal{G}\;=$ $\mathcal{G}_{1},\ldots,\mathcal{G}_{k}=\mathcal{G}^{\prime}$ that are all I-equivalent to G such that the only diference between G $\mathcal{G}_{i}$ and $\mathcal{G}_{i+1}$ is a single reversal of a covered edge. 
> 定理：
> 当且仅当存在一个网络序列 $\mathcal G_1, \dots, \mathcal G_k = \mathcal G'$ 都是 $\mathcal G$ 的 I-equivalence，并且 $\mathcal G_{i+1}, \mathcal G_{i}$ 之间唯一的差异就是一个 covered edge 的反向时，$\mathcal G, \mathcal G'$ 是 I-equivalent

The proof of this theorem is left as an exercise (exercise 3.18). 
## 3.4 From Distributions to Graphs 
In the previous sections, we showed at, if $P$ factorizes ove $\mathcal{G}$ , we can derive a rich set of independence assertions that hold for P by simply examining G . This result immediately leads to the idea that we can use a graph as a way of revealing the structure in a distribution. In particular, we can test for independencies in $P$ by constructing a graph $\mathcal{G}$ that represents $P$ and testing d-separation in $\mathcal{G}$ . As we will see, having a graph that reveals the structure in P has other important consequences, in terms of reducing the number of parameters required to specify or learn the distribution, and in terms of the complexity of performing inference on the network. 
> 之前的部分中，我们展示了如果 $P$ 在 $\mathcal G$ 上分解，我们可以通过检查 $\mathcal G$ 就得到在 $P$ 中成立的一系列独立性断言，故我们可以用图表示分布的结构
> 特别地，我们可以通过构造一个表示 $P$ 的图，然后测试图中的 d-seperation 来揭示 $P$ 中的独立性
> 使用图表示分布可以帮助减少学习分布所需的参数，以及减少推理的复杂度

In this section, we examine the following question: Given a distribution $P$ , to what extent can we construct a graph $\mathcal{G}$ whose independencies are a reasonable surrogate for the independencies in $P\Lsh$ It is important to emphasize that we will never actually take a fully speciﬁed distribution $P$ and construct a graph $\mathcal{G}$ for it: As we discussed, a full joint distribution is much too large to represent explicitly. However, answering this question is an important conceptual exercise, which will help us later on when we try to understand the process of constructing a Bayesian network that represents our model of the world, whether manually or by learning from data. 
> 本节探讨以下问题：
> 给定分布 $P$，我们可以构造一个表示 $P$ 中独立性的图 $\mathcal G$ 到什么程度，注意我们当然不会为一个 fully specified 的分布构造一个非常大的图
### 3.4.1 Minimal I-Maps 
One approach to ﬁnding a graph that represents a distribution $P$ is simply to take any graph that is an I-map for $P$ . The problem with this naive approach is clear: As we saw in example 3.3, the complete graph is an I-map for any distribution, yet it does not reveal any of the independence structure in the distribution. However, examples such as this one are not very interesting. The graph that we used as an I-map is clearly and trivially unrepresentative of the distribution, in that there are edges that are obviously redundant. This intuition leads to the following deﬁnition, which we also deﬁne more broadly: 
> 最简单的方法是找到一个是 $P$ 的 I-map 的图，但也存在问题
> 例如，完全图是任意分布的 I-map（不存在依赖性），显然其中存在太多冗余边（依赖性）

**Deﬁnition 3.13** minimal I-map 
A graph $\mathcal{K}$ is $^a$ minimal I-map fo a set of independencies $\mathcal{T}$ if it is an I-map for $\mathcal{T}$ , and if the removal of even a single edge from renders it not an I-map. 
> 定义：
> 如果 $\mathcal K$ 是独立性集合 $\mathcal I$ 的 I-map，并且 $\mathcal K$ 移除任意一个边都会导致它不是 I-map，称 $\mathcal K$ 是 $\mathcal I$  的的 I-极小map（不存在多余的依赖性，删边就是增加独立性）
> minimal I-map 是在删边之后就会出现 $P$ 中没有的独立性，但这不代表着它本身就包含了 $P$ 中所有的独立性

This notion of an I-map applies to multiple types of graphs, both Bayesian networks and other types of graphs that we will encounter later on. Moreover, because it refers to a set of independencies $\mathcal{T}$ , i an be used to deﬁne an I-map for a distribution $P$ , by taking $\mathcal{Z}=\mathcal{Z}(P)$ , or to another graph ′ , by taking $\mathcal{Z}=\mathcal{Z}(K^{\prime})$ . 
> 最小 I-map 的概念不仅可以应用于贝叶斯网络，也可以应用于其他类型的图

Recall that deﬁnition 3.5 deﬁnes a Bayesian network to be a distribution $P$ that factorizes over $\mathcal{G}$ , thereby implying that $\mathcal{G}$ is an I-map for $P$ . is standard to restrict the deﬁnition even further, by requiring that $\mathcal{G}$ be a minimal I-map for P . 

How do we obtain a minimal I-map for the set of independencies induced by a given dis- tribution $P$ ? The proof of the factorization theorem (theorem 3.1) gives us a procedure, which is shown in algorithm 3.2. We assume we are given a predetermined variable ordering , say, $\{X_{1},\cdot\cdot\cdot, X_{n}\}$ . We n examine each variable $X_{i}$ , $i=1,\dots, n$ in turn. For each $X_{i}$ , we pick som inimal subset U of $\{X_{1},.\,.\,.\,, X_{i-1}\}$ to be $X_{i}$ ’s parents in $\mathcal{G}$ . More precisely, we requ that U satisfy $(X_{i}\mid\bot\;\{X_{1},\bot\;.\;.\;, X_{i-1}\}-U\mid U)$ , and that no no can be removed from U without violating this property. We then set U to be the parents of $X_{i}$ . 
> 考虑对于一个给定的分布 $P$，找到它的最小 I-map
> 参考 factorization 定理的证明过程，我们假设给定一个预定义的变量顺序 $\{X_1,\dots, X_n\}$，然后轮流检查变量 $X_i, i= 1,\dots, n$ ，
> 对于每一个 $X_i$，我们在 $\{X_1,\dots, X_{i-1}\}$ 中选择出某个最小的子集 $\pmb U$，用于表示 $\mathcal G$ 中 $X_i$ 的父变量集合，我们要求 $\pmb U$ 满足 $(X_i \perp \{X_1,\dots, X_{i-1}\}- \pmb U\mid \pmb U)$ （给定父变量，和其他变量条件独立），并且在不违反这一性质的前提下，没有节点可以从 $\pmb U$ 中被移除（$\pmb U$ 不能不包含 $X_i$ 的全部父变量，同时因为最小，$\pmb U$ 不能包含不是 $X_i$ 的父变量的节点，否则可以在不违反该性质的前提下被移除）
> 然后，我们将 $\pmb U$ 设定为 $X_i$ 的 parents

The proof of theorem 3.1 tells us that, if each node $X_{i}$ is independent of $X_{1},\dots, X_{i-1}$ given its parent $\mathcal{G}$ , then $P$ factorizes over $\mathcal{G}$ . We can then conclude from theore 3.2 that $\mathcal{G}$ is an I-map for P . By constructio $\mathcal{G}$ is minimal, so that $\mathcal{G}$ is a minimal I-map for P . 
> 定理3.1的证明告诉我们，如果每个节点 $X_i$ 在给定它在 $\mathcal G$ 中的 parents 的前提下独立于 $X_1,\dots, X_{i-1}$，则 $P$ 根据 $\mathcal G$ 分解，因此我们构造出的 $\mathcal G$ 是 $P$ 的一个 I-map
> 因为构造时 $\pmb U$ 是最小的，则构造出的 $\mathcal G$ 是 $P$ 的最小 I-map

Note that our choice of U may not be unique. Consider, for example, a case where two variables $A$ and $B$ are logically equivalent, that is, our distribution $P$ only gives positive probability to instantiations where $A$ and $B$ have the same value. Now, consider a node $C$ that is correlated with $A$ . Clearly, we can choose either $A$ or $B$ to be a parent of $C$ , but having chosen the one, we cannot choose the other without violating minimality. Hence, the minimal parent set $U$ in our construction is not necessarily unique. However, one can show that, if the distribution is positive (see deﬁnition 2.5), that is, if for any instantiation $\xi$ to all the network variables $\mathcal{X}$ we have that $P (\xi)>0$ , then the choice of parent set, given an ordering, unique. 
> $\pmb U$ 的选择不一定是唯一的
> 但如果 $P$ 是一个 positive 分布（只要事件不为空，概率就大于0），即对于所有网络变量 $\mathcal X$ 的任意实例化 $\xi$，我们都有 $P (\xi) > 0$，则可以证明给定一个顺序，对于 parent set 的选择是唯一的

![[Probabilistic Graph Theory-Algorithm3.2.png]]

Under this assumption, algorithm 3.2 can produce all minimal I-maps for P : Let be any min mal I-map for $P$ . If we give call Build-Minimal-I-Map with an orderi $\prec$ that is topological for G , then, due to the uniqueness argument, the algorithm must return G . 
> 在这一假设下，algorithm3.2可以为 $P$ 生成所有的最小 I-map：令 $\mathcal G$ 是 $P$ 的任意最小 I-map，如果我们为算法3.2给定一个 $\mathcal G$ 的拓扑排序 $\prec$，则算法一定会返回 $\mathcal G$
> 不同的拓扑排序会返回不同的 $\mathcal G$

At ﬁrst glance, the minimal I-map seems to be a reasonable candidate for capturing the structure in the distribution: It seems that if $\mathcal{G}$ is a minim I-map for a d ribution $P$ , then we should be able to “read of” all of the independencies in P directly from G . Unfortunately, this intuition is false. 
> 但即便 $\mathcal G$ 是 $P$ 的最小 I-map，我们也不能从 $\mathcal G$ 中“读出” $P$ 中全部的独立性，因为排序顺序实际上存在影响

Example 3.5 
Consider the distribution PBstudent, as defined in figure 3.4, and let us go through the process of constructing a minimal I-map for PBstudent. We note that the graph Gstudent precisely reflects the independencies in this distribution PBstudent (that is, I(PBstudent) = I(Gstudent)), so that we can use Gstudent to determine which independencies hold in PBstudent.

Our construction process for three diferent orderings. Throughout this process, it is important to remember that we are testing independencies relative to the distribution $P_{\mathcal{B}^{\mathrm{stunderit}}}$ . We can use $G_{\mathrm{stadium}}$ (ﬁgure 3.4) to guide our intuition about which independencies hold in $P_{\mathcal{B}^{\mathrm{stunderit}}}$ , but we can always resort to testing these independencies in the joint distribution $P_{\mathcal{B}^{\mathrm{stunderit}}}$ . 

The ﬁrst ordering is a very natural one: $D, I, S, G, L$ . We add one node at a time and see which of the possible edges from the preceding nodes are redundant. We start by adding $D$ , then $I$ . We an now remove the dge from $D$ to $I$ because this particular distribution satisﬁes $(I\perp D)$ , so $I$ is independent of D given its other parents (the empty set). Continuing on, we add S , but we n remove the edge from $D$ to $S$ be use ur distribution satisﬁes $(S\perp D\mid I)$ d G , but we can move the edge from S to G , because the distribution satisﬁes ( $\left (G\perp S\mid I, D\right)$ ⊥ | ) . Finally, we add L , but we can remove all edges from $D, I, S$ . Thus, our ﬁnal output is the graph in ﬁgure $3.8a_{!}$ , which is precisely our original network for this distribution. 
> 对于一个节点序列，我们一次加入一个节点，然后看哪些从前面节点到当前节点的边是多余的（多余的父节点）

Now, consider a somewhat less natural ordering: $L, S, G, I, D$ . In this case, the resulting I-map is not quite as natural or as sparse. To see this, let us consider the sequence of steps. We start by adding $L$ to the graph. Since it is the ﬁrst variable in the ordering, it must be a root. Next, we consider $S$ . The decision is whether to have $L$ as a parent of $S$ . Clearly, we need an edge from $L$ to $S$ , because the quality of the student’s letter is correlated with his SAT score in this distribution, and $S$ has no other parents that help render it independent of $L$ . Formally, we have th $(S\perp L)$ does not hold in the distribution. In the next iteration of the algorithm, we introduce G . Now, all possible s sets of $\{L, S\}$ are p tential rents set for $G$ . Clearly, $G$ depen nt on $L$ . Moreover, although G is independent of S given I , it is not independent of S given L . Hence, we must add the edge between $S$ and $G$ . Carrying out the procedure, we end up with the graph shown in ﬁgure $3.8b$ . 

Finally, consider the ordering: $L, D, S, I, G$ . In this case, a similar analysis results in the graph shown in ﬁgure $3.8c,$ which is almost a complete graph, missing only the edge from $S$ to $G$ , which we can remove because $G$ is independent of $S$ given $I$ . 

Note that the graphs in ﬁgure 3.8b, c really are minimal I-maps for this distribution. However, they fail to capture some or all of the independencies that hold in the distribution. Thus, they show that the fact that $\mathcal{G}$ i a minimal I-map for $P$ is far from a guarantee that $\mathcal{G}$ captures the independence structure in P . 
### 3.4.2 Perfect Maps 
We aim to ﬁnd a graph $\mathcal{G}$ that precisely captures the independencies in a given distribution $P$ . 
> 我们希望找到可以精确捕获给定分布 $P$ 中的独立性的图 $\mathcal G$

***Definition 3.14*** perfect map 
We say that a graph $\mathcal{K}$ $a$ perfect map (P-m ) for a set of independencies $\mathcal{T}$ if we have that $\mathcal{Z}(\mathcal{K})=\mathcal{Z}$ . We say that K is a perfect map for $P$ if $\mathcal{Z}(\mathcal{K})=\mathcal{Z}(P)$ .
> 如果我们有 $\mathcal I (\mathcal K) = \mathcal I$ ，则称图 $\mathcal K$ 是独立性集合 $\mathcal I$ 的完美 I-map，或者称为 p-map

If we obtain a grap $\mathcal{G}$ that is a P-m p for a distribution $P$ , then we can ( nition) read the independencies in P directly from G . By construction, our original graph $G_{s t u d e n t}$ is a P-map for $P_{\mathcal{B}^{s t u d e n t}}$ . 
> 如果我们得到一个分布 $P$ 的一个 p-map $\mathcal G$，则我们可以从 $\mathcal G$ 中读出 $P$ 中的所有的独立性

If our goal is to ﬁnd a perfect map for a distribution, an immediate question is whether every distribution has a perfect map. Unfortunately, the answer is no, and for several reasons. The ﬁrst type of counterexample involves regularity in the parameter iz ation of the distribution that cannot be captured in the graph structure. 
> 并不是每个分布都有 p-map，一个例子就是分布的参数化中的 regularity 有时不能被图结构捕获

Example 3.6 
Consider a joint distribution $P$ over 3 random variables $X, Y, Z$ such that: 

$$
P (x, y, z)=\left\{\begin{array}{l l}{{1/12\qquad}}&{{x\oplus y\oplus z=f a l s e}}\\ {{1/6\qquad}}&{{x\oplus y\oplus z=t r u e}}\end{array}\right.
$$ 
where $\oplus$ the XOR (exclusive OR unctio A sim e calc tion shows that $(X\bot Y)\in{\mathcal{Z}}(P)$ , and that Z is not independent of X given Y $Y$ or of $Y$ given X . Hence, one minimal $I_{\cdot}$ -map for this distribution is the network $X\rightarrow Z\leftarrow Y$ , using a deterministic XOR for the CPD of $Z$ . However, this network is not a perfect map; a preci ly analogous calculation shows that $(X\perp Z)\in{\mathcal{Z}}(P)$ , but this conclusion is not supported by a d-separation analysis. 

Thus, we see that deterministic relationships can lead to distributions that do not have a P-map. Additional examples arise as a consequence of other regularities in the CPD. 
> 确定性关系会让分布不存在 p-map
> 其他的 CPD 中的 regularity 也是如此

Example 3.7 
Consider a slight elaboration of our Student example. During his academic career, our student George has taken both Econ101 and CS102. The professors of both classes have written him letters, but the recruiter at Acme Consulting asks for only a single recommendation. George’s chance of getting the job depends on the quality of the letter he gives the recruiter. We thus have four random variables: $L1$ and $L2$ , corresponding to the quality of the recommendation letters for Econ101 and CS102 respectively; $C$ , whose value represents George’s choice of which letter to use; and $J$ , representing the event that George is hired by Acme Consulting. 

The obvious minimal $I^{,}$ -map for this distribution is shown in ﬁgure 3.9. Is this a perfect map? Clearly, it does not reﬂect independencies that are not at the variable level. In particular, we have that $(L1\perp J\mid C=2)$ . However, this limitation is not surprising; by deﬁnition, a BN structure makes independence assertions only at the level of variables. (We return to this issue in section 5.2.2.) However, our problems are not limited to these ﬁner-grained independencies. Some thought reveals that, in our arget distribution, we also have tha $(L1\;\bot\; L2\;\vert\; C, J)$ ! This independence is not implied by d-separation, because the $\nu$ -structure L $L1\,\rightarrow\, J\,\leftarrow\, L2$ → ← is enabl wever, we can convince ourselves that the independence holds using reasoning by cases. If $C=1$ , then there is no dependence of $J$ on $L2$ . Intuitively, the edge from $L2$ to $J$ disappears, eliminating the trail between $L1$ and $L2$ , so that $L1$ and $L2$ are independent in this case. A symmetric analysis applies in the case that $C\,=\, 2$ . Thus, in both cases, we have that $L1$ and $L2$ are independent. This independence assertion is not captured by our minimal I-map, which is therefore not a $P\cdot$ -map. 

A second class of distributions that do not have a perfect map are those for which the independence assumptions imposed by the structure of Bayesian networks is simply not appropriate. 
> 有时，贝叶斯网络 impose 的独立性假设仅仅是不适合于特定分布的，因此这类分布也不存在 p-map

Example 3.8
Consider a scenario where we have four students who get together in pairs to work on the homework for a class. For various reasons, only the following pairs meet: Alice and Bob; Bob and Charles; Charles and Debbie; and Debbie and Alice. (Alice and Charles just can’t stand each other, and Bob and Debbie had a relationship that ended badly.) The study pairs are shown in ﬁgure 3.10a. 

In this example, the professor accidentally misspoke in class, giving rise to a possible miscon- ception among the students in the class. Each of the students in the class may subsequently have ﬁgured out the problem, perhaps by thinking about the issue or reading the textbook. In subsequent study pairs, he or she may transmit this newfound understanding to his or her study partners. We therefore have four binary random variables, representing whether the student has the misconcep- tion or not. We assume tha or each $X\,\in\,\{A, B, C, D\}$ , $x^{1}$ denotes the case where the student has the misconception, and x $x^{0}$ denotes the case where he or she does not. 

Because Alice and Charles never speak to each other directly, we have that $A$ and $C$ are con- ditionally independent given $B$ and $D$ . Similarly, $B$ and $D$ are conditionally independent given $A$ and $C$ . Can we represent this distribution (with these independence properties) using a BN? One attempt is shown in ﬁgure 3.10b. Indeed, it encodes the independence assumption that $(A\ \perp\ C\ |\ \{B, D\})$ . However, it also implies that $B$ and $D$ are independent given only $A$ , but dependent given both $A$ and $C$ . Hence, it fails to provide a perfect map for our target dis- tribution. A second attempt, shown in ﬁgure 3.10c, is equally unsuccessful. It also implies that $(A\perp C\mid\{B, D\})$ , but it also implies that $B$ and $D$ are marginally independent. It is clear that all other candidate BN structures are also ﬂawed, so that this distribution does not have a perfect map. 
### 3.4.3 Finding Perfect Maps\*
Earlier we discussed an algorithm for ﬁnding minimal I-maps. We now consider an algorithm for ﬁnding a perfect map (P-map) of a distribution. Because the requirements from a P-map are stronger than the ones we require from an I-map, the algorithm will be more involved. 

Throughout the discussion in this section, we assume that $P$ has a P-map. In other words, there is an unknown DAG $\mathcal{G}^{*}$ that is map of $P$ . Since $\mathcal{G}^{*}$ is a P-map, we will interchangeably refer to independencies in P and in G $\mathcal{G}^{*}$ (since these are the same). We note that the algorithms we describe do fail when they are given a distribution that does not have a P-map. We discuss this issue in more detail later. 

Thus, our goal is to identify $\mathcal{G}^{\ast}$ from $P$ . One obvious difculty hat arises when we consider this goal is that $\mathcal{G}^{*}$ is, in general, not uniquely identiﬁable from P . A P-map of a distribution, if one exists, is generally not unique: As we saw, for example, in ﬁgure 3.5, multiple graphs can encode precisely the same independence assumptions. However, the P-map of a distribution is unique up to I-equivalence between networks. That is, a distribution $P$ can have many P-maps, but all of them are I-equivalent. 

If we require that a P-map construction algorithm return a single network, the output we get may be some arbitrary member of the I-equivalence class of $\mathcal{G}^{\ast}$ . A more correct answer would be to return the entire equivalence class, thus avoiding an arbitrary commitment to a possibly incorrect structure. Of course, we do not want our algorithm to return a (possibly very large) set of distinct networks as output. Thus, one of our tasks in this section is to develop a compact representation of an entire equivalence class of DAGs. As we will see later in the book, this representation plays a useful role in other contexts as well. 

This formulation of the problem points us toward a solution. Recall that, according to theorem 3.8, two DAGs are I-equivalent if they share the same skeleton and the same set of immoralities. Thus, we can construct the I-equivalence class for $\mathcal{G}^{\ast}$ by determi ng its skeleton and its immoralities from the independence properties of the given distribution P . We then use both of these components to build a representation of the equivalence class. 

#### 3.4.3.1 Identifying the Undirected Skeleton 

At this stage we want to construct an undirected graph $S$ that contains an edge $X{-}Y$ if $X$ and $Y$ are adjacent in $\mathcal{G}^{\ast}$ ; that is, if either $X\rightarrow Y$ or $Y\rightarrow X$ is an edge in $\mathcal{G}^{\ast}$ . 

The basic ea is to use independence queries of the form $(X\perp Y\mid U)$ for difer sets of variables U . This idea is based on the observation that if X and Y are adjacent in G $\mathcal{G}^{\ast}$ , we cannot separate them with any set of variables. 

Lemma 3.1 

$\mathcal{G}^{*}$ be a $P\cdot$ -map of a distribution $\mathcal{P}$ , an let $X$ and $Y$ be two v ables ch that $X\rightarrow Y$ is in $\mathcal{G}^{*}$ . Then, $P\not\models (X\bot Y\mid U)$ for any set U that does not include X and Y . 

Proof Assume that $X\,\rightarrow\, Y\,\in\,{\mathcal{G}}^{*}$ , and let $U$ be a set of riables. According to d- separation the il $X\rightarrow Y$ → cannot be blo ed by the evid $U$ $X$ and $Y$ are not d-separated by U . Since G $\mathcal{G}^{*}$ is a P-map of P , we have that $P\not\models (X\bot Y\mid U)$ ̸| ⊥ | . 

This lemma implies that if $X$ and $Y$ are adjacent in $\mathcal{G}^{*}$ that involve both of them would fail. Conversely, if X and $Y$ are not adjacent in $\mathcal{G}$ , we would hope to be able to ﬁnd a set of variables that makes these two variables conditionally independent. Indeed, as we now show, we can provide a precise characterization of such a set: 

Lemma 3.2 The proof is left as an exercise (exercise 3.19). Thus, if $X$ an $Y$ are not adjacent in $\mathcal{G}^{*}$ , then we can ﬁnd a set $U$ so that ${\mathcal{P}}\models (X\ \bot\ Y\ |\ U)$ ⊥ | . We call this set U a witness of their independence. Moreover, the lemma shows that we can ﬁnd a witn s of bounded size. Thus, if we assume that $\mathcal{G}^{\ast}$ has bounded indegree, say less than or equal to d , then we do not need to consider witness sets larger than d . 

 

With these tools in hand, we can now construct an algorithm for building a skeleton of $\mathcal{G}^{*}$ , shown in algorithm 3.3. For each pair of variables, we consider all potential witness sets and test for independence. If we ﬁnd a witness that separates the two variables, we record it (we will soon see why) and move on to the next pair of variables. If we do not ﬁnd a witness, then we conclu o variables are adjacent in $\mathcal{G}^{*}$ and add them to the skeleton. The list Witnesses $(X_{i}, X_{j},{\mathcal{H}}, d)$ H in line 4 speciﬁes the set of possible witness sets t t we consider for separating $X_{i}$ and $X_{j}$ . From our earlier discussion, if we assume a bound d on the indegree, then we can restrict attention to sets $U$ of size at most $d$ . Moreover, using the same analysis, we saw that we have a witness that consists either of the parents of $X_{i}$ or of the parents of $X_{j}$ . In the ﬁ case, we can restrict at ntion to sets $U\subseteq\dot{\mathrm{Nb}}_{X_{i}}^{\mathcal{H}}-\{X_{j}\}$ −{ } , where $\operatorname{Nb}_{X_{i}}^{\hat{\mathcal{H}}}$ are the neighbors of $X_{i}$ in the current graph H ; in the s ond, we c similarly restrict attention to ts $U\subseteq\mathrm{Nb}_{X_{j}}^{\mathcal{H}}-\{X_{i}\}$ . ally, w note that if U separates $X_{i}$ and $X_{j}$ , then also many of $U$ ’s supersets will separate $X_{i}$ and $X_{j}$ . Thus, we search the set of possible witnesses in order of increasing size. 

This algorithm ill over the correct skeleton given that $\mathcal{G}^{\ast}$ is a P-map of $P$ and has bounded indegree d . If P does not have a P-map, then the algorithm can fail; see exercise 3.22. This algorithm has complexity of $O (n^{d+2})$ since we consider $O (n^{2})$ pairs, and for each we perform $O ((n-2)^{d})$ independence tests. We greatly reduce the number of independence tests by ordering potential witnesses accordingly, and by aborting the inner loop once we ﬁnd a witness for a pair (after line 9). However, for pairs of variables that are directly connected in the skeleton, we still need to evaluate all potential witnesses. 
#### 3.4.3.2 Identifying Immoralities 

potential immorality 

Proposition 3.1 

At this stage we have reconstructed the undirected skeleton $S$ using Build-PMap-Skeleton . Now, we want to reconstruct edge direction. The main cue for learning about edge dir ions in $\mathcal{G}^{\ast}$ are immoralities. As shown in theorem 3.8, all DAGs in the equivalence class of G $\mathcal{G}^{\ast}$ share the same set of immoralities. Thus, our goal is to consider potential immoralities in the skeleton and for each one determine whether it is indeed an immorality. A triplet of variables $X, Z, Y$ is a potential immorality if the skeleton contains $X{-}Z{-}Y$ but does not contain an edge between $X$ an $Y$ . If such a triplet is indeed an immorality $\mathcal{G}^{\ast}$ , then $X$ an $Y$ cannot be independent given Z . Nor will they be independent given a set U that contains Z . More precisely, 

$\mathcal{G}^{\ast}$ $P\cdot$ map of a distribution $P$ , and let $X, Y$ nd $Z$ be variab s that form an immorality $X\rightarrow Z\leftarrow Y$ → ← . Then, $P\not\models (X\bot Y\mid U)$ ⊥ | for any set U that contains Z . 

Proof Let $U$ be set of ariables t at contains $Z$ nce $Z$ i bserved, the tr $X\rightarrow Z\leftarrow Y$ is active, and so X and Y $Y$ are not d-separated in G $\mathcal{G}^{\ast}$ . Since G $\mathcal{G}^{*}$ is a P-map of P , we have that $P^{*}\not\models (X\bot Y\mid U)$ ⊥ | . 

What happens in the complementary situation? Suppose $X{-}Z{-}Y$ in the skeleton, but is lity. at one of the followi three ases is in $\mathcal{G}^{*}$ : $X\rightarrow Z\rightarrow Y$ , $Y\,\rightarrow\, Z\,\rightarrow\, X$ → → , or $X\leftarrow Z\rightarrow Y$ ← → . In all three cases, X and Y are d-separated only if Z is observed. 

Proposition 3.2 

Let $\mathcal{G}^{*}$ be a $P\!\!\cdot\!\!$ map of a d $P$ nd let the triplet $X, Y, Z$ be a potential immorality in the n of G $\mathcal{G}^{\ast}$ , such that $X\rightarrow Z\leftarrow Y$ → ← is not in $\mathcal{G}^{*}$ . If U is such that $P\models (X\ \bot\ Y\ |\ U)$ ⊥ | , then $Z\in U$ . 

Proof Consider all three conﬁgurations of the trail $X\,\rightleftharpoons\, Z\,\rightleftharpoons\, Y$ . In all three, $Z$ must be served r to block the trail. Since $\mathcal{G}^{\ast}$ is a P-map of $P$ , we have that if $P\models (X\bot Y\mid$ ⊥ | $U$ ) , then $Z\in U$ ∈ . 

Combining these two results, we see that a potential immorality $X{-}Z{-}Y$ is an immorality if and only if $Z$ is not in the witness set (s) for $X$ and $Y$ . That is, if $X{-}Z{-}Y$ is an immorality, then proposition 3.1 shows that $Z$ is not in any witness set $U$ ; conversely, if $X{-}Z{-}Y$ is not an immorality, the $Z$ must be in every witness set $U$ . Thus, we can use the speciﬁc witness set $U_{X, Y}$ that we recorded for $X, Y$ in order to determine whether this triplet is an immorality or not: we simply check whether $Z\in U_{X, Y}$ . If $Z\notin U_{X, Y}$ , then we declare the triplet an immorality. Otherwise, we declare that it is not an immorality. The Mark-Immoralities procedure shown in algorithm 3.4 summarizes this process. 

#### 3.4.3.3 Representing Equivalence Classes 

Once we have the skeleton and identiﬁed the immoralities, we have a speciﬁcation of the equivalence class of $\mathcal{G}^{\ast}$ . example, to test if $\mathcal{G}$ is equivalent to $\mathcal{G}^{*}$ we can check whether it has the same skeleton as $\mathcal{G}^{\ast}$ and whether it agrees on the location of the immoralities. 

The description of an equivalence class using only the skeleton and the set of immoralities is somewhat unsatisfying. For example, we might want to know whether the fact that our network is in the equivalence class implies that there an a $X\rightarrow Y$ . Although the deﬁnition does tell us whether there is some edge between X and Y , it leaves the direction unresolved. In other cases, however, the direction of an edge is fully determined, for example, by the presence of an immorality. To encode both of these cases, we use a graph that allows both directed and undirected edges, as deﬁned in section 2.2. Indeed, as we show, the chain graph, or PDAG, representation (deﬁnition 2.21) provides precisely the right framework. 

Deﬁnition 3.15 class PDAG 

Let $\mathcal{G}$ be a AG. A chain graph $\mathcal{K}$ is $^a$ class f the equivalence ss of $\mathcal{G}$ i shares the sa skeleton as G , an ns a directed edge $X\rightarrow Y$ → if and only if all G ${\mathcal{G}}^{\prime}$ that are I-equivalent to G contain the edge $X\rightarrow Y$ . 

In other words, a class PDAG represents potential edge orientations in the equivalence classes. If the edge is directed, then all the members of the equivalence class agree on the orientation of the edge. If the edge is undirected, there are two DAGs in the equivalence class that disagree on the orientation of the edge. 

For example, the networks in ﬁgure 3.5a–(c) are I-equivalent. The class PDAG of this equiva- lence class is the graph $X{-}Z{-}Y$ , since both edges can be oriented in either direction in some member of the equivalence class. Note that, although both edges in this PDAG are undirected, not all joint orientations of these edges are in the equivalence class. As discussed earlier, setting the orientations $X\rightarrow Z\leftarrow Y$ results in the network of ﬁ re $3.5\mathrm{d}$ , which does not belong this equivalence class. More generally, if the class PDAG has k undirected edges, the equivalence class can contain at most $2^{k}$ networks, but the actual number can be much smaller. 

Can we efectively construct the class PDAG $\mathcal{K}$ for $\mathcal{G}^{\ast}$ from the reconstru d skeleton and immoralities? Clea , edges involved in immoralities must be directed in K . The obvious question is whether K can contain directed edges that are not involved in immoralities. In other words, can there be additional edges whose direction is necessarily the same in every member of the equivalence class? To understand this issue better, consider the following example: 

Example 3.9 

Consider the DAG of ﬁgure 3.11a. This DAG has a single immor $A\rightarrow C\leftarrow B$ s immorality implies that the class PDAG of this DAG must have the arcs A $A\,\rightarrow\, C$ and B $B\,\rightarrow\, C$ directed, as 

![](images/1a02c0c0fccb74370210e12e9cea0befa0aa0e6c62c4ef574dba3b71002430ec.jpg) 
Figure 3.11 Simple example of compelled edges in the representation of an equivalence class. (a) Original DAG $\mathcal{G}^{\ast}$ . (b) Skeleton of $\mathcal{G}^{*}$ annotated with immoralities. (c) a DAG that is not equivalent to $\mathcal{G}^{\ast}$ . 

shown in ﬁgure 3.11b. This PDAG representation suggests that the edge $C{-}D$ can assume either orientation. Note, however, that the DAG of ﬁgure 3.11c, where we orient the edge between $C$ and $D$ as $D\rightarrow C$ , contains additional immoralities (that is, $A\rightarrow C\leftarrow D$ and $B\rightarrow C\leftarrow D_{z}$ ). Thus, this DAG is not equivalent to our original DAG. 

In this example, there is only one possible orientation of $C{-}D$ that is consistent with the ﬁnding that $A{-}C{-}D$ is not an immorality. Thus, we conclude that the class PDAG for the DAG of ﬁgure 3.11a is simply the DAG itself. In other words, the equivalence class of this DAG is a singleton. 

As this example shows, a negative result in an immorality test also provides information about edge orientation. In particu , in y case where the PDAG $\mathcal{K}$ contai cture $X\rightarrow Y{-}Z$ and there is no edge from $X$ $Z$ we must orient the edge Y $Y\,\rightarrow\, Z$ → , for otherwise we would create an immorality X $X\rightarrow Y\leftarrow Z$ . 

Some thought reveals that there are other local conﬁgurations of edges where some ways of orienting edges are inconsistent, forcing a particular direction for an edge. Each such conﬁgu- ration can be viewed as a local constraint on edge orientation, give rise to a rule that can be used to orient more edges in the PDAG. Three such rules are shown in ﬁgure 3.12. 

Let us understand the intuition behind these rules. Rule R1 is precisely the one we discussed earlier. Rule R2 is derived from the standard acyclicity constraint: If we have the directed path $X\,\rightarrow\, Y\,\rightarrow\, Z$ , and an undirected edge $X{-}Z$ , we cannot direct the ed $X\leftarrow Z$ without creating a cycle. Hence, we can conclude that the edge must be directed X $X\rightarrow Z$ → . The third rule seems a little more complex, but it is also easily motivated. Assume, by contradiction, that we direct the edge $Z\rightarrow X$ . In this ca nnot direct the edge $X{-}Y_{1}$ $X\rightarrow Y_{1}$ without creati we must have $Y_{1}\rightarrow X$ → . Similarly, we must have $Y_{2}\rightarrow X$ . , in this case, Y $Y_{1}\,\rightarrow\, X\,\leftarrow\, Y_{2}$ → ← forms an i ity (a is no edge between $Y_{1}$ and $Y_{2}$ ), which contradicts the fact that the edges $X{-}Y_{1}$ and $X{-}Y_{2}$ are undirected in the original PDAG. 

These three rules can be applied constructively in an obvious way: A rule applies to a PDAG whenever the induced subgraph on a subset of variables exactly matches the graph on the left-hand side of the rule. In that case, we modify this subgraph to match the subgraph on the right-hand side of the rule. Note that, by applying one rule and orienting a previously undirected edge, we create a new graph. This might create a subgraph that matches the antecedent of a rule, enforcing the orientation of additional edges. This process, however, must terminate at 

 
Figure 3.12 Rules for orienting edges in PDAG. Each rule lists a conﬁguration of edges before and after an application of the rule. 

some point (since we are only adding orientations at each step, and the number of edges is ﬁnite). This implies that iterated application of this local constraint to the graph (a process known as constraint propagation ) is guaranteed to converge. 

 

Algorithm 3.5 implements this process. It builds an initial graph using Build-PMap-Skeleton and Mark-Immoralities , and then iteratively applies the three rules until convergence, that is, until we cannot ﬁnd a subgraph that matches a left-hand side of any of the rules. 

 
Figure 3.13 More complex example of compelled edges in the representation of an equivalence class. (a) Original DAG $\mathcal{G}^{\ast}$ . (b) S ton of $\mathcal{G}^{\ast}$ annotated with immoralities. (c) Complete PDAG represen- tation of the equivalence class of G $\mathcal{G}^{*}$ . 

Example 3.10 Consider the DAG shown in ﬁgure 3.13a. After checking for immoralities, we ﬁnd the graph shown in ﬁgure $3. l3b.$ Now, we can start applying the preceding rules. For example, consider the variables $B,\, E,$ , and $F$ . They induce a subgraph that matches the left-hand side of rule R1. Thus, we orient the edge between $E$ and $F$ to $E\rightarrow F$ . No onsider the variables $C,\,E.$ , and $F$ . T ir ind ced subgraph matches the left-hand side of rule R2, so we now orient the edge between C and F to $C\rightarrow F$ . stage, if we consider the variables $E,\, F,\,G.$ , we can apply the rule R1, orient e edge $F\rightarrow G$ → . (Alternatively, we could have arrived at the same orientation using $C,\, F$ , and $G$ .) The resulting PDAG is shown in ﬁgure 3.13c. 

It seems fairly obvious that this algorithm is guaranteed to be sound: Any edge that is oriented by this procedure is, indeed, directed in exactly the same way in all of the members of the equivalence class. Much more surprising is the fact that it is also complete: Repeated application of these three local rules is guaranteed to capture all edge orientations in the equivalence class, without the need for additional global constraints. More precisely, we can prove that this algorithm produces the correct class PDAG for the distribution $P$ : 

Theorem 3.10 Let P be a distribution that s a $P\!\!\cdot\!\!$ -map $\mathcal{G}^{\ast}$ , and let K be the PDAG returned by Build-PDAG $(\mathcal{X}, P)$ . Then, K is a class PDAG of G $\mathcal{G}^{\ast}$ . The proof of this theorem can be decomposed into several aspects of correctness. We have already established the correctness of the skeleton found by Build-PMap-Skeleton . Thus, it remains to show that the directionality of the edges is correct. Speciﬁcally, we need to establish three basic facts: 

• Acyclicity: The graph returned by Build-PDAG $\scriptstyle (\mathcal{X}, P)$ is acyclic. 

• Soundness: If $X\rightarrow Y\in{\mathcal{K}}$ , then $X\rightarrow Y$ appears in all DAGs in $\mathcal{G}^{*}$ ’s I-equivalence class.

 • Com f $X{-}Y\in{\mathcal{K}}$ , then we can ﬁnd a DAG $\mathcal{G}$ that is I-equivalent to $\mathcal{G}^{*}$ such that $X\rightarrow Y\in{\mathcal{G}}$ . 

The last condition establishes completeness, since there is no constraint on the direction of the arc. In other words, the same condition can be used to prove the existence of a graph with $X\rightarrow Y$ and of a graph with $Y\rightarrow X$ . Hence, it shows that either direction is possible within the equivalence class. We begin with the soundness of the procedure. 

Proposition 3.3 

Let $P$ b n that $P\!\!\cdot\!\!$ map $\mathcal{G}^{\ast}$ , and let K be the graph returned by Build- AG $(\mathcal{X}, P)$ . Then, if $X\rightarrow Y\in{\mathcal{K}}$ , then $X\rightarrow Y$ appears in all DAGs in the I-equivalence class of $\mathcal{G}^{\ast}$ . 

The proof is left as an exercise (exercise 3.23). Next, we consider the acyclicity of the graph. We start by proving a property of graphs returned by the procedure. (Note that, once we prove that the graph returned by the procedure is the correct PDAG, it will follow that this property also holds for class PDAGs in general.) 

Proposition 3.4 

Proposition 3.5 

$\mathcal{K}$ graph returned by Build-PDAG . Then, if $X\,\rightarrow\, Y\,\in\,{\mathcal{K}}$ and $Y{-}Z\;\in\;{\mathcal{K}},$ , then $X\rightarrow Z\in{\mathcal{K}}$ . 

The proof is left as an exercise (exercise 3.24). 

Let $\mathcal{K}$ be the chain graph returned by Build-PDAG . Then $\mathcal{K}$ is acyclic. 

Proof Suppose, by way of contradiction, that $\mathcal{K}$ contains a cycle. That is, there is a (partially) directed path $X_{1}\rightleftharpoons X_{2}\rightleftharpoons\ldots\rightleftharpoons X_{n}\rightleftharpoons X_{1}$ . Without loss of generality, assume that this path is the shortest cycle in $\mathcal{K}$ . We claim that the path c directed edge. To see that, suppose that the the triplet $X_{i}\,\rightarrow\, X_{i+1}{-}X_{i+2}$ → . Then, invoking proposit 4, we have that $X_{i}\,\rightarrow\, X_{i+2}\,\in\,{\mathcal{K}}$ → nd thus, we can construct a shorter path $X_{i+1}$ the edge $X_{i}\,\rightarrow\, X_{i+2}$ → . At this stage, we have a directed cycle $X_{1}\,\rightarrow\, X_{2}\,\rightarrow\,.\,.\,.\, X_{n}\,\rightarrow\, X_{1}$ → → → . Using proposition 3.3, we nclude that this cycle appears in any DAG in the quivalence class, and in rticular in G $\mathcal{G}^{\ast}$ . This conclusion contradicts the assumption that G $\mathcal{G}^{\ast}$ is acyclic. It follows that K is acyclic. 

The ﬁnal step is the completeness proof. Again, we start by examining a property of the graph $\mathcal{K}$ . 

Proposition 3.6 

The PDAG $\mathcal{K}$ returned by Build-PDAG is necessarily chordal. 

The proof is left as an exercise (exercise 3.25). 

This property a ws us to characterize the structure of the PDAG $\mathcal{K}$ returned by Build-PDAG . Recall that, since K is an undirected chain graph, we can partition X into chain components $K_{1},\dots, K_{\ell}$ , where each chain component contains variables that are connected by undirected edges (see deﬁnition 2.21). It turns out that, in an undirected chordal graph, we can orient any edge in any direction without creating an immorality. 

Let $\mathcal{K}$ be a undirected chordal graph over $\mathcal{X}$ , and let $X, Y\in{\mathcal{X}}$ . Then, there is a DAG $\mathcal{G}$ such that 

(a) The skeleton of $\mathcal{G}$ is $\mathcal{K}$ . (b) $\mathcal{G}$ does not contain immoralities. (c) $X\rightarrow Y\in{\mathcal{G}}$ . 

The proof of this proposition requires some additional machinery that we introduce in chapter 4, so we defer the proof to that chapter. 

Using this proposition, we see that we can orient edges in the chain component $K_{j}$ without introducing immoralities within the component. We still need to ensure that orienting an edge $X{-}Y$ within a component cannot introduce an immorality involving edges from outside the component. To s tuation cannot occur, suppose we orient the edge $X\,\rightarrow\, Y$ , and suppose that Z $Z\,\rightarrow\, Y\,\in\,{\mathcal{K}}$ → ∈K . eems poten l immorality. Ho pplying oposition 3.4, we se that s ce $Z\rightarrow Y$ → an $Y{-}X$ n K , then so must be $Z\rightarrow X$ → . Since $Z$ is a parent of both X and $Y$ , we have that $X\rightarrow Y\leftarrow Z$ → ← is not an immorality. This argument applies to any edge we orient within an undirected component, and thus no new immoralities are introduced. 

With these tools, we can complete the completeness proof of Build-PDAG . 

Le $P$ ribution that has a $P\!\!\cdot\!\!$ -map $\mathcal{G}^{\ast}$ , and t $\mathcal{K}$ be the gra returned by $(\mathcal{X}, P)$ . If $X{-}Y\in{\mathcal{K}}$ ∈K , then we can ﬁnd a DAG G that is I-equivalent to G $\mathcal{G}^{\ast}$ such that $X\rightarrow Y\in{\mathcal{G}}$ → ∈G . 

roof Suppose we have an undirected edge $X{-}Y\in{\mathcal{K}}$ . We w that there is a DAG $\mathcal{G}$ G that has the same skeleton and immoralities as K such that $X\rightarrow Y\in{\mathcal{G}}$ → ∈G . If can build such a graph $\mathcal{G}$ , then clearly it is in the I-equivalence class of $\mathcal{G}^{\ast}$ . 

The construction is simple. We start with the chain component that contains $X{-}Y$ , and use proposition 3.7 to orient the edges in the component so that $X\rightarrow Y$ is in the resulting DAG. Then, we use the same construction to orient all other chain components. Since the chain components are ordered and acyclic, and our orientation of each chain component is acyclic, the resulting directed graph is acyclic. Moreover, as shown, the new orientation in each component does not introduce immoralities. Thus, the resulting DAG has exactly the same skeleton and immoralities as $\mathcal{K}$ . 

## 3.5 Summary 
In this chapter, we discussed the issue of specifying a high-dimensional joint distribution com- pactly by exploiting its independence properties. We provided two complementary deﬁnitions of a Bayesian network. The ﬁrst is as a directed graph $\mathcal{G}$ , annotated with a set of conditional probability distributions $P (X_{i}\mid\mathrm{Pa}_{X_{i}})$ . The network together with the PDs deﬁne a di ribu- tion via the chain rule for Bayesian networks. In this case, we say that P factorizes over G . We also deﬁned the independence assumptions associated with the graph: the local independencies, the set of basic independence assumptions induced by the network structure; and the larger set of global independencies that are derived from the d-separation criterion. We showed the equivalence of these three f ndamental notions: $P$ facto es over $\mathcal{G}$ if and only if $P$ satisﬁes the local independencies of G , which holds if and only if P satisﬁes the global independencies derived from d-separation. This result shows the equivalence of our two views of a Bayesian network: as a scafolding for factoring a probability distribution $P$ , and as a representation of a set of independence assumptions that hold for $P$ . We also showed that the set of independen- cies derived from d-separation is a complete characterization of the independence properties that are implied by the graph structure alone, rather than by properties of a speciﬁc distribution over $\mathcal{G}$ . 
> 本章讨论了利用独立性质表示高维度联合分布的问题
> 我们提供了两个关于贝叶斯网络的互补定义，一个是有向图 $\mathcal G$ 和一系列条件概率分布 $P (X_i \mid \text{Pa}_{X_i})$，另一个是和网络相关的一系列独立性假设：局部独立性，即直接由网络结构推出的独立性假设集合；全局独立性，即由网络的 d-seperation 条件推出的独立性集合
> 我们说明了三个基本概念的等价：$P$ 根据 $\mathcal G$ 分解当且仅当 $P$ 满足 $\mathcal G$ 的局部独立性，which holds if and only if $P$ 满足从 d-seperation 中推导出的全局独立性
> 因此贝叶斯网络的两个定义是等价的：作为分解 $P$ 的 scaffolding 和作为 $P$ 中保持的一系列独立性假设的表示

We deﬁned a set of basic notions that use the characterization of a graph as a set of indepen- dencies. We deﬁned the notion of a minimal I-map and showed that almost every distribution has multiple minimal I-maps, but that a minimal I-map for $P$ does not necessarily capture all of the independence properties in $P$ . We then deﬁned a more stringent notion of a perfect map , and showed that not every distribution has a perfect map. We deﬁned $I^{,}$ -equivalence , which captures an independence-equivalence relationship between two graphs, one where they specify precisely the same set of independencies. 

Finally, we deﬁned the notion of a class PDAG , a partially directed graph that provides a compact representation for an entire I-equivalence class, and we provided an algorithm for constructing this graph. 

These deﬁnitions and results are fundamental properties of the Bayesian network represen- tation and its semantics. Some of the algorithms that we discussed are never used as is; for example, we never directly use the procedure to ﬁnd a minimal I-map given an explicit rep- resentation of the distribution. However, these results are crucial to understanding the cases where we can construct a Bayesian network that reﬂects our understanding of a given domain, and what the resulting network means. 

# 4 Undirected Graphical Models 
So far, we have dealt only with directed graphical models, or Bayesian networks. These models are useful because both the structure and the parameters provide a natural representation for many types of real-world domains. In this chapter, we turn our attention to another important class of graphical models, deﬁned on the basis of undirected graphs. 

As we will see, these models are useful in modeling a variety of phenomena where one cannot naturally ascribe a directionality to the interaction between variables. Furthermore, the undirected models also ofer a diferent and often simpler perspective on directed models, in terms of both the independence structure and the inference task. We also introduce a combined framework that allows both directed and undirected edges. We note that, unlike our results in the previous chapter, some of the results in this chapter require that we restrict attention to distributions over discrete state spaces. 

## 4.1 The Misconception Example 
To motivate our discussion of an alternative graphical representation, let us reexamine the Misconception example of section 3.4.2 (example 3.8). In this example, we have four students who get together in pairs to work on their homework for a class. The pairs that meet are shown via the edges in the undirected graph of ﬁgure 3.10a. 

As we discussed, we intuitively want to model a distribution that satisﬁes $(A\perp C\mid\{B,D\})$ and $(B\,\perp\,D\,\mid\,\{A,C\})$ , but no other independencies. As we showed, these independencies cannot be naturally captured in a Bayesian network: any Bayesian network I-map of such a distribution would necessarily have extraneous edges, and it would not capture at least one of the desired independence statements. More broadly, a Bayesian network requires that we ascribe a directionality to each inﬂuence. In this case, the interactions between the variables seem symmetrical, and we would like a model that allows us to represent these correlations without forcing a speciﬁc direction to the inﬂuence. 
> 变量之间的交互是对称的情况下，我们考虑用无向边建模，以不指定变量之间影响的特定方向

A representation that implements this intuition is an undirected graph. As in a Bayesian network, the nodes in the graph of a Markov network represent the variables, and the edges correspond to a notion of direct probabilistic interaction between the neighboring variables — an interaction that is not mediated by any other variable in the network. In this case, the graph of ﬁgure 3.10, which captures the interacting pairs, is precisely the Markov network structure that captures our intuitions for this example. As we will see, this similarity is not an accident. 
> 这对应的网络就是 Markov 网络，它和贝叶斯网络类似，节点表示变量，边表示变量之间的直接交互

The remaining question is how to parameterize this undirected graph. Because the interaction is not directed, there is no reason to use a standard CPD, where we represent the distribution over one node given others. Rather, we need a more symmetric parameter iz ation. Intuitively, what we want to capture is the afnities between related variables. For example, we might want to represent the fact that Alice and Bob are more likely to agree than to disagree. We associate with $A,B$ a general-purpose function, also called a factor : 
> 无向图中，影响是无向的，因此标准 CPD 的语义不符合
> 我们需要对称的参数化，我们需要捕获相关变量的亲近性
> 我们使用一种通用目的的函数来关联变量 $A, B$，称其为因子：

**Definition 4.1**
Let $D$ be a set of random variables. We deﬁne $a$ factor $\phi$ to be a function from $V a l(D)$ to I R . A factor is nonnegative if all its entries are nonnegative. The set of variables $_D$ is called the scope of the factor and denoted $Scope[φ]$. 
> 定义：
> $D$ 为随机变量集合，定义因子 $\phi$ 是从 $Val (D)$ 到 $\mathbb R$ 的函数，如果因子的所有项都非负，因子就是非负的
> 变量集合 $D$ 称为因子的作用域，记作 $Scope[\phi]$

Unless stated otherwise, we restrict attention to nonnegative factors. 
> 除非明确说明，我们都关注非负因子

In our example, we e a factor $\phi_{1}(A,B):V a l(A,B)\mapsto I\!\!R^{+}$ . The value associated with a particular assignment a, b denotes the afnity between these two values: the higher the value $\phi_{1}(a,b)$ , the more compatible these two values are. 
> 例如，我们有因子 $\phi_1 (A, B): Val (A, B)\mapsto \mathbb R^+$，$\mathbb R^+$ 中的值和特定的 $a, b$ 赋值关联，表示这两个值之间的亲密性，值越高，二者越相容

Figure 4.1a shows one possible compatibility factor for these variables. Note that this factor is not normalized; indeed, the entries are not even in $[0,1]$ . Roughly speaking, $\phi_{1}(A,B)$ asserts that it is more likely that Alice and Bob agree. It also adds more weight for the case where they are both right than for the case where they are both wrong. This factor function also has the property that $\phi_{1}(a^{1},b^{0})\,<\,\phi_{1}(a^{0},b^{1})$ . Thus, if they disagree, there is less weight for the case where Alice has the misconception but Bob does not than for the converse case. 

In a similar way, we deﬁne a compatibility factor for each other interacting pair: $\{B,C\}$ , $\{C,D\}$ , and $\{A,D\}$ . Figure 4.1 shows one possible choice of factors for all four pairs. For example, the factor over $C,D$ represents the compatibility of Charles and Debbie. It indicates that Charles and Debbie argue all the time, so that the most likely instantiations are those where they end up disagreeing. 
> 我们为每一个直接交互的变量对都定义相容因子

As in a Bayesian network, the parameter iz ation of the Markov network deﬁnes the local interactions between directly related variables. To deﬁne a global model, we need to combine these interactions. As in Bayesian networks, we combine the local models by multiplying them. Thus, we want $P(a,b,c,d)$ to be $\phi_{1}(a,b)\cdot\phi_{2}(b,c)\cdot\phi_{3}(c,d)\cdot\phi_{4}(d,a)$ .  In this case, however, we have no guarantees that the result of this process is a normalized joint distribution. Indeed, in this example, it deﬁnitely is not. Thus, we deﬁne the distribution by taking the product of  the local factors, and then normalizing it to deﬁne a legal distribution. Speciﬁcally, we deﬁne 

$$
P(a,b,c,d)=\frac{1}{Z}\phi_{1}(a,b)\cdot\phi_{2}(b,c)\cdot\phi_{3}(c,d)\cdot\phi_{4}(d,a),
$$ 
where 

$$
Z=\sum_{a,b,c,d}\phi_{1}(a,b)\cdot\phi_{2}(b,c)\cdot\phi_{3}(c,d)\cdot\phi_{4}(d,a)
$$ 
is a normalizing constant known as the partition function . 
> 定义了局部的交互之后，我们需要将这些交互结合以定义全局的模型
> 和贝叶斯网络中一样，我们通过将局部模型相乘来对它们进行结合，即 $P (a, b, c, d) = \phi_1 (a, b) \cdot \phi_2 (b, c)\cdot \phi_3 (c, d)\cdot \phi_4 (d, a)$
> 但我们不能保证得到规范化的联合分布，为此需要进行规范化，因此我们定义
> $P (a, b, c, d) = \frac 1 Z\phi_1 (a, b) \cdot \phi_2 (b, c)\cdot \phi_3 (c, d)\cdot \phi_4 (d, a)$，
> 其中 $Z$ 是规范化常数，$Z = \sum_{a, b, c, d}\phi_1 (a, b)\cdot\phi_2 (b, c)\cdot \phi_3 (c, d)\cdot \phi_4 (d, a)$，称其为划分函数

The term “partition” originates from the early history of Markov networks, which originated from the concept of Markov random ﬁeld (or MRF ) in statistical physics (see box 4.C); the “function” is because the value of $Z$ is a function of the parameters, a dependence that will play a signiﬁcant role in our discussion of learning. 
> $Z$ 也是其参数的函数，因此称为划分函数

In our example, the unnormalized measure (the simple product of the four factors) is shown in the next-to-last column in ﬁgure 4.2. For example, the entry corresponding to $a^{1},b^{1},c^{0},d^{1}$ is obtained by multiplying: 

$$
\begin{array}{r}{\phi_{1}(a^{1},b^{1})\cdot\phi_{2}(b^{1},c^{0})\cdot\phi_{3}(c^{0},d^{1})\cdot\phi_{4}(d^{1},a^{1})=10\cdot1\cdot100\cdot100=100,000.}\end{array}
$$ 
The last column shows the normalized distribution. 

We can use this joint distribution to answer queries, as usual. For example, by summing out $A,\,C$ , and $D$ , we obtain $P(b^{1})\approx0.732$ and $P(b^{0})\approx0.268$ ; that is, Bob is 26 percent likely to have the misconception. On the other hand, if we now observe that Charles does not have the misconception $(c^{0})$ , we obtain $P(b^{1}\mid c^{0})\approx0.06$ . 

The beneﬁt of this representation is that it allows us great ﬂexibility in representing inter- actions between variables. For example, if we want to change the nature of the interaction between $A$ and $B$ , we can simply modify the entries in that factor, without having to deal with normalization constraints and the interaction with other factors. The ﬂip side of this ﬂexibility, as we will see later, is that the efects of these changes are not always intuitively understandable. 
> 该表示使得我们在表示变量之间的交互时具有极大的灵活性，变量之间的因子可以任意定义，并且如果有两个变量的交互方式变化，可以直接修改其因子，不会影响其他因子

As in Bayesian networks, there is a tight connection between the factorization of the distribution and its independence properties. The key result here is stated in exercise 2.5: $P\models(X\ \bot\ Y\ |\ Z)$ if and only if we can write $P$ in the form $P(\mathcal{X})=\phi_{1}(\boldsymbol{X},Z)\phi_{2}(\boldsymbol{Y},Z)$ . In our example, the structure of the factors allows us to decompose the distribution in several ways; for example: 

$$
P(A,B,C,D)=\left[\frac{1}{Z}\phi_{1}(A,B)\phi_{2}(B,C)\right]\phi_{3}(C,D)\phi_{4}(A,D).
$$ 
From this decomposition, we can infer that $P\models(B\ \bot\ D\ |\ A,C)$  . We can similarly infer that $P\models(A\bot C\mid B,D)$ . 
> Markov 网络的分解和其表示的分布中的独立性同样存在联系：
> $P\vDash (\pmb X \perp \pmb Y\mid \pmb Z)$ 当且仅当我们可以将 $P$ 写为 $P (\mathcal X) = P(\pmb X, \pmb Y, \pmb Z) = \phi_1 (\pmb X, \pmb Z)\phi_2 (\pmb Y , \pmb Z)$

These are precisely the two independencies that we tried, unsuccessfully, to achieve using a Bayesian network, in example 3.8. Moreover, these properties correspond to our intuition of “paths of inﬂuence” in the graph, where we have that $B$ and $D$ are separated given $A,C$ , and that $A$ and $C$ are separated given $B,D$ . Indeed, as in a Bayesian network, independence properties of the distribution $P$ correspond directly to separation properties in the graph over which $P$ factorizes. 
> 和贝叶斯网络一样，分布 $P$ 分解的 Markov 网络中的分离性质直接对应了 $P$ 中的独立性质

## 4.2 Parameterization 
We begin our formal discussion by describing the parameter iz ation used in the class of undi- rected graphical models that are the focus of this chapter. In the next section, we make the connection to the graph structure and demonstrate how it captures the independence properties of the distribution. 

To represent a distribution, we need to associate the graph structure with a set of parameters, in the same way that CPDs were used to parameterize the directed graph structure. However, the parameter iz ation of Markov networks is not as intuitive as that of Bayesian networks, since the factors do not correspond either to probabilities or to conditional probabilities. As a con- sequence, the parameters are not intuitively understandable, making them hard to elicit from people. As we will see in chapter 20, they are also signiﬁcantly harder to estimate from data. 
> 要表示分布，我们需要将图结构和一组参数关联
> 有向图中，我们使用 CPDs 进行参数化
> 无向图中，因子并不直接对应概率分布或条件概率分布，因此参数并不是直觉上可以理解的

### 4.2.1 Factors 
A key issue in parameterizing a Markov network is that the representation is undirected, so that the parameteriz ation cannot be directed in nature. We therefore use factors, as deﬁned in deﬁnition 4.1. Note that a factor subsumes both the notion of a joint distribution and the notion of a CPD. A joint distribution over $_D$ is a factor over $_D$ : it speciﬁes a real number for every assignment of values of $_D$ . A conditional distribution $P(X\mid U)$ is a factor over $\{X\}\cup U$ . However, both CPDs and joint distributions must satisfy certain normalization constraints (for example, in a joint distribution the numbers must sum to 1), whereas there are no constraints on the parameters in a factor. 
>  Markov 网络表示是无向的，故参数化本质也应该无向，即因子
>  因子实际包含了联合分布和 CPD 的概念，$\pmb D$ 上的联合分布也是 $\pmb D$ 上的因子，条件概率分布 $P (X\mid \pmb U)$ 也是 $\{X\} \cup \pmb U$ 上的因子，区别仅在于规范化

As we discussed, we can view a factor as roughly describing the “compatibilities” between diferent values of the variables in its scope. We can now parameterize the graph by associating a set of a factors with it. One obvious idea might be to associate parameters directly with the edges in the graph. However, a simple calculation will convince us that this approach is insufcient to parameterize a full distribution. 

Example 4.1 
Consider a fully connected graph over $\mathcal{X}$ ; in this case, the graph speciﬁes no conditional ind en- dence assumptions, so that we should be able to specify an arbitrary joint distribution over X . If all of the variables are binary, each factor over an edge would have 4 parameters, and the total number of parameters in the graph would be $4{\binom{n}{2}}$ . However, the number of parameters required to specify a joint distribution over $n$ binary variables is $2^{n}-1$ . Thus, pairwise factors simply do not have enough parameters to encompass the space of joint distributions. More intuitively, such factors capture only the pairwise interactions, and not interactions that involve combinations of values of larger subsets of variables. 
> 考虑 $\mathcal X$ 上的完全连接图，该图没有指定条件独立性假设，因此我们可以指定 $\mathcal X$ 上的任意联合分布
> 如果变量都为2元，两个变量之间的因子需要的参数量是4，指定全部的因子的参数量是 $4\binom {n} {2}$，而制定 $n$ 个二元变量的任意联合分布的参数量实际应该为 $2^n - 1$
> 故仅仅使用成对的因子是不足以有足够的参数覆盖整个联合分布的空间的
> 直观上，这类因子仅捕获了成对的交互，不能表示更大子集的变量之间的组合交互

A more general representation can be obtained by allowing factors over arbitrary subsets of variables. To provide a formal deﬁnition, we ﬁrst introduce the following important operation on factors. 
> 通过允许因子包含任意数量的变量，可以得到更通用的表示

**Deﬁnition 4.2** factor product 
Let $X$ , $Y$ , and $Z$ be three disjoint sets of variables, and let $\phi_{1}(X,Y)$ and $\phi_{2}(Y,Z)$ be two factors. We deﬁne the factor product $\phi_{1}\times\phi_{2}$ to be a factor $\psi:V a l(X,Y,Z)\mapsto I\!\!R$ as follows: 

$$
\psi(\pmb X,\pmb Y,\pmb Z)=\phi_{1}(\pmb X,\pmb Y)\cdot\phi_{2}(\pmb Y,\pmb Z).
$$

> 定义：
> $\pmb X, \pmb Y, \pmb Z$ 是三个不相交变量集合，令 $\phi_1 (\pmb X, \pmb Y), \phi_2 (\pmb Y, \pmb Z)$ 为两个因子
> 定义因子积 $\phi_1 \times \phi_2$ 为因子 $\psi: Val (\pmb X, \pmb Y, \pmb Z)\to \mathbb R$ 如上

![[Probabilistic Graph Theory-Fig4.3.png]]

The key aspect to note about this deﬁnition is the fact that the two factors $\phi_{1}$ and $\phi_{2}$ are multiplied in a way that “matches up” the common part $Y$ . Figure 4.3 shows an example of the product of two factors. We have deliberately chosen factors that do not correspond either to probabilities or to conditional probabilities, in order to emphasize the generality of this operation. 
> 两个因子 $\phi_1, \phi_2$ 通过“匹配”共同部分 $\pmb Y$ 而相乘

As we have already observed, both CPDs and joint distributions are factors. Indeed, the chain rule for Bayesian networks deﬁnes the joint distribution factor as the product of the CPD factors. For example, when computing $P(A,B)=P(A)P(B\mid A)$ , e always multiply entries in the $P(A)$ and $P(B\mid A)$ tables that have the same value for A . 
> 在定义上，CPDs 和联合分布都是因子，贝叶斯网络的链式规则就是定义了联合分布因子可以作为 CPD 因子的乘积
> 例如，计算 $P (A, B) = P (A) P (B\mid A)$ 时，我们总是将 $P (A), P (B\mid A)$ 表中具有相同 $A$ 值的项相乘

Thus, letting $\phi_{X_{i}}(X_{i},\mathrm{Pa}_{X_{i}})$ represent $P(X_{i}\mid\mathrm{Pa}_{X_{i}})$ , we have that 
> 令 $\phi_{X_i}(X_i, \text{Pa}_{X_i})$ 表示 $P (X_i \mid \text{Pa}_{X_i})$，我们有

$$
P(X_{1},.\,.\,.\,,X_{n})=\prod_{i}\phi_{X_{i}}.
$$ 
### 4.2.2 Gibbs Distributions and Markov Networks 
We can now use the more general notion of factor product to deﬁne an undirected parameterization of a distribution. 
> 我们使用因子乘积的更通用概念来定义一个分布的无向参数化

**Deﬁnition 4.3**  Gibbs distribution 
A distribution $P_{\Phi}$ is $a$ Gibbs distribution parameterized by a set of factors $\Phi=\{\phi_{1}(D_{1}),.\,.\,.\,,\phi_{K}(D_{K})\}$ if it is deﬁned as follows: 
> 定义：
> 定义吉布斯分布 $P_{\Phi}$，由因子集合 $\Phi = \{\phi_1(\pmb D_1), \dots, \phi_K(\pmb D_K)\}$ 参数化，满足：

$$
P_{\Phi}(X_{1},.\,.\,.\,,X_{n})=\frac{1}{{Z}}\tilde{P}_{\Phi}(X_{1},.\,.\,.\,,X_{n}),
$$

where 

$$
\tilde{P}_{\Phi}(X_{1},\dots,X_{n})=\phi_{1}(\pmb D_{1})\times\phi_{2}(\pmb D_{2})\times\cdot\cdot\times\phi_{m}(\pmb D_{m})
$$ 
is an unnormalized measure and 

$$
Z=\sum_{X_{1},\ldots,X_{n}}\tilde{P}_{\Phi}(X_{1},\ldots,X_{n})
$$ 
is a normalizing constant called the partition function . 
> 即 $\tilde P_{\Phi}(X_1,\dots, X_n)$ 定义为 $\phi_1 (\pmb D_1), \dots, \phi_m (\pmb D_m)$ 的乘积，$Z$ 定义为 $\tilde P_{\Phi}$ 对于所有情况的求和（称为划分函数），$P_{\Phi}$ 定义为 $\tilde P_\Phi$ 规范化后得到的分布

It is tempting to think of the factors as representing the marginal probabilities of the variables in their scope. Thus, looking at any individual factor, we might be led to believe that the behavior of the distribution deﬁned by the Markov network as a whole corresponds to the behavior deﬁned by the factor. However, this intuition is overly simplistic. **A factor is only one contribution to the overall joint distribution. The distribution as a whole has to take into consideration the contributions from all of the factors involved.** 
> 概率图模型中，因子通常代表变量之间相互作用的方式，它们共同定义了一个联合概率分布
> 马尔可夫网络中的单个因子只是对整体联合分布的一个贡献，整个分布必须考虑到所有相关因子的贡献
> 虽然单个因子可能反映了其作用域内变量的某种形式的概率分布，但是整个马尔可夫网络的联合概率分布是由所有因子共同决定的，没有一个单独的因子可以完全描述整个系统的概率行为，因为每个因子都只描述了部分变量之间的交互作用，而忽略了其他因子的影响

Example 4.2
Consider the distribution of ﬁgure 4.2. The marginal distribution over $A,B$ , is  The most likely conﬁguration is the one where Alice and Bob disagree. By contrast, the highest entry in the factor $\phi_{1}(A,B)$ in ﬁgure 4.1 corresponds to the assignment $a^{0},b^{0}$ . The reason for the discrepancy is the inﬂuence of the other factors on the distribution. In particular, $\phi_{3}(C,D)$ asserts that Charles and Debbie disagree, whereas $\phi_{2}(B,C)$ and $\phi_{4}(D,A)$ assert that Bob and Charles agree and that Debbie and Alice agree. Taking just these factors into consideration, we would conclude that Alice and Bob are likely to disagree. In this case, the “strength” of these other factors is much stronger than that of the $\phi_{1}(A,B)$ factor, so that the inﬂuence of the latter is overwhelmed. 

We now want to relate the parameter iz ation of a Gibbs distribution to a graph structure. If our parameter iz ation contains a factor whose scope contains both $X$ and $Y$ , we are introducing a direct interaction between them. Intuitively, we would like these direct interactions to be represented in the graph structure. Thus, if our parameter iz ation contains such a factor, we would like the associated Markov network structure $\mathcal{H}$ to contain an edge between $X$ and $Y$ . 
> 我们准备将 Gibbs 分布的参数化和一个图结构联系
> 如果 Gibbs 分布的参数化中包含了一个因子，其作用域包含了 $X, Y$，说明二者是有直接联系的，直观上，就是在图中有一条边相连

**Deﬁnition 4.4** Markov network factorization
We say that a distribution $P_{\Phi}$ with $\Phi\:=\:\{\phi_{1}(D_{1}),.\,.\,.\,,\phi_{K}(D_{K})\}$ factorizes over a Markov network $\mathcal H$ if each $D_{k}$ $(k=1,.\,.\,.\,,K)$ is a complete subgraph of $\mathcal H$.
> 定义：
> 对于分布 $P_\Phi$，其因子集合为 $\{\phi_1(\pmb D_1), \dots, \phi_K(\pmb D_K)\}$，如果满足每个 $\pmb D_k(k = 1,\dots, K)$ 都是 $\mathcal H$ 的一个完全子图，则称 $P_\Phi$ 在 Markov 网络 $\mathcal H$ 上分解

The factors that parameterize a Markov network are often called clique potentials . 
> 参数化了 Markov 网络的因子 $\phi_i$ 常常称为团势函数（团即完全子图的节点构成的集合）

As we will see, if we associate factors only with complete subgraphs, as in this deﬁnition, we are not violating the independence assumptions induced by the network structure, as deﬁned later in this chapter. 
> 在定义4.4中，我们将因子仅和完全子图关联，这不会违反由网络结构推导出的独立性假设

Note that, because every complete subgraph is a subset of some (maximal) clique, we can reduce the number of factors in our parameter iz ation by allowing factors only for maximal cliques. 
> 每个完全子图都是某个最大团/最大完全子图的子集，可以通过仅留下表示最大团的因子来减少参数化中的因子数量

More precisely, let $C_{1},\ldots,C_{k}$ be the cliques in $\mathcal{H}$ . We can parameterize $P$ using a set of factors $\phi_{1}(C_{1}),.\,.\,.\,,\phi_{l}(C_{l})$ . Any factorization in terms of complete subgraphs can be converted into this form simply by assigning each factor to a clique that encompasses its scope and multiplying all of the factors assigned to each clique to produce a clique potential. In our Misconception example, we have four cliques: $\{A,B\},\,\{B,C\},\,\{C,D\}$ , and $\{A,D\}$ . Each of these cliques can have its own clique potential. One possible setting of the parameters in these clique potential is shown in ﬁgure 4.1. Figure 4.4 shows two examples of a Markov network and the (maximal) cliques in that network. 
> 具体地说，令 $\pmb C_1, \dots, \pmb C_k$ 为 $\mathcal H$ 中的团，我们使用因子集合 $\phi_1 (\pmb C_1), \dots, \phi_l (\pmb C_l)$ 来参数化 $P$
> 我们只需要将每个因子和包含了它的作用域内的变量的团关联，然后将各个团得到的因子相乘，就得到了这个团的团势函数

**Although it can be used without loss of generality, the parameter iz ation using maximal clique potentials generally obscures structure that is present in the original set of factors.** For example, consider the Gibbs distribution described in example 4.1. Here, we have a potential for every pair of variables, so the Markov network associated with this distribution is a single large clique containing all variables. If we associate a factor with this single clique, it would be exponentially large in the number of variables, whereas the original parameter iz ation in terms of edges requires only a quadratic number of parameters. See section 4.4.1.1 for further discussion. 
> 使用最大团势函数的参数化虽然不会失去一般性，但这通常容易掩盖原始因子集合中的结构
> 例如 example 4.1中，对于每一对变量，它们都可以有一个团势函数/因子，而完整的 Markov 网络则关联了完整的分布，网络本身也是一个团，故完整的分布也可以由单个团势函数/因子表示
> 但要用单个因子表示整个网络，其参数数量和变量数量呈指数关系，而原来的按照边的参数化仅和变量数量呈二次关系

Box 4.A — Concept: Pairwise Markov Networks. 
A subclass of Markov networks that arises in many contexts is that of pairwise Markov networks , representing distributions where all of the factors are over single variables or pairs of variables. More precisely, $a$ pairwise Markov network over $a$ graph $\mathcal{H}$ is associated with a set of node potentials $\{\phi(X_{i}):i=1,.\,.\,.\,,n\}$ and a set of edge potentials $\{\phi(X_{i},X_{j})\,:\,(X_{i},X_{j})\,\in\,\mathcal{H}\}$ . The overall distribution is (as always) the normalized product of all of the potentials (both node and edge). Pairwise MRFs are attractive because of their simplicity, and because interactions on edges are an important special case that often arises in practice (see, for example, box 4.C and box 4.B). 

A class of pairwise Markov networks that often arises, and that is commonly used as a benchmark for inference, is the class of networks structured in the form of a grid, as shown in ﬁgure 4.A.1. As we discuss in the inference chapters of this book, although these networks have a simple and compact representation, they pose a signiﬁcant challenge for inference algorithms. 

### 4.2.3 Reduced Markov Networks 
We end this section with one ﬁnal concept that will prove very useful in later sections. Consider the process of conditioning a distribution on some assignment $\mathbfit{u}$ to some subset of variables $U$ . Conditioning a distribution corresponds to eliminating all entries in the joint distribution that are inconsistent with the event $\pmb{U}=\pmb{u}$ , and renormalizing the remaining entries to sum to 1. Now, consider the case where our distribution has the form $P_{\Phi}$ for some set of factors $\Phi$ . Each entry in the unnormalized measure $\tilde{P}_{\Phi}$ is a product of entries from the factors $\Phi$ , one entry from each factor. If, in some factor, we have an entry that is inconsistent with $\pmb{U}=\pmb{u}$ , it will only contribute to entries in $\tilde{P}_{\Phi}$ that are also inconsistent with this event. Thus, we can eliminate all such entries from every factor in $\Phi$ . 
> 考虑将一个分布条件于对于某个变量子集 $\pmb U$ 的某个赋值 $\pmb u$，这对应于消除联合分布中所有和事件 $\pmb U = \pmb u$ 不一致的项，然后将剩余的项重新规范化到和为1
> 现在，考虑我们分布形式为 $P_\Phi$，因子集合为 $\Phi$，我们知道未规范化的度量 $\tilde P_\Phi$ 中的每个项都是因子集合 $\Phi$ 中每个因子的各个项的乘积，每个因子共享一个项
> 此时，如果在某个因子中，我们有某个项和事件 $\pmb U = \pmb u$ 不一致，显然它只会对于 $\tilde P_\Phi$ 中和该事件不一致的项做出贡献，因此，我们应该消除 $\Phi$ 中所有因子中所有和该事件不一致的项

More generally, we can deﬁne: 

**Deﬁnition 4.5** factor reduction 
Let $\phi(Y)$ be a factor, $\pmb{U}={\pmb u}$ an assignment for $U\subseteq Y$ . We deﬁne the reduction of the factor $\phi$ to the context $\pmb{U}=\pmb{u}$ , denoted $\phi[U=u]$ (and abbreviated $\phi[{\pmb u}])$ , to be a factor over scope $Y^{\prime}=Y-U$ , such that 

$$
\phi[{\pmb u}]({\pmb y}^{\prime})=\phi({\pmb y}^{\prime},{\pmb u}).
$$ 
For $U\not\subset Y$ , we deﬁne $\phi[{\pmb u}]$ to be $\phi[U^{\prime}=\pmb{u}^{\prime}]$ , wher $U^{\prime}=U\cap Y$ , and ${\pmb u}^{\prime}\,=\,{\pmb u}\langle{\pmb U}^{\prime}\rangle$ , where $u\langle U^{\prime}\rangle$ denotes the assignment in u to the variables in $U^{\prime}$ . 
> 定义：
> $\phi (\pmb Y)$ 为因子，$\pmb U = \pmb u$ 是对于 $\pmb U\subseteq \pmb Y$ 的赋值，定义因子 $\phi$ 对于上下文 $\pmb U = \pmb u$ 的简化为一个在作用域 $\pmb Y' = \pmb Y - \pmb U$ 上的因子，记作 $\phi[\pmb U = \pmb u]$ 或者 $\phi[\pmb u]$，满足 $\phi [\pmb u](\pmb y') = \phi (\pmb y', \pmb u)$
> 对于 $\pmb U \not\subset \pmb Y$，定义 $\phi[\pmb u]$ 为 $\phi[\pmb U' = \pmb u']$，其中 $\pmb U' = \pmb U \cap \pmb Y$，$\pmb u' = \pmb u \langle \pmb U' \rangle$（$\pmb u\langle \pmb U' \rangle$ 表示 $\pmb u$ 中对于 $\pmb U'$ 中的变量中的赋值）

Figure 4.5 illustrates this operation, reducing the of ﬁgure 4.3 to the context $C=c^{1}$ . 
> 直观上看，缩减就是把上下文对应的因子中的项都筛选了出来

Now, consider a product of factors. An entry in the product is consistent with $\mathbfit{u}$ if and only if it is a product of entries that are all consistent with $\mathbfit{u}$ . We can therefore deﬁne: 
> 考虑因子的积：积中的一项要和 $\pmb u$ 一致当且仅当求积的所有项都和 $\pmb u$ 一致

**Deﬁnition 4.6** reduced Gibbs distribution 
Let $P_{\Phi}$ be a Gibbs distribution parameterized by $\Phi=\{\phi_{1},.\,.\,.\,,\phi_{K}\}$ and let $\mathbfit{u}$ be a context. The reduced Gibbs distribution $P_{\Phi}[{\pmb u}]$ is the Gibbs distribution deﬁned by the set of factors $\Phi[{\pmb u}]=$ $\{\phi_{1}[{\pmb u}],.\,.\,.\,,\phi_{K}[{\pmb u}]\}$ . 
> 定义：
> $P_\Phi$ 为 Gibbs 分布，由 $\Phi = \{\phi_1, \dots, \phi_K\}$ 参数化，令 $\pmb u$ 为上下文
> 简化的 Gibbs 分布 $P_\Phi[\pmb u]$ 为由因子集合 $\Phi[\pmb u] = \{\phi_1[\pmb u], \dots, \phi_K[\pmb u]\}$ 参数化的 Gibbs 分布
> 直观上看，就是把每个因子都根据上下文筛选了一遍

Reducing the set of factors deﬁning $P_{\Phi}$ to some context $\mathbfit{u}$ corresponds directly to the opera- tion of conditioning $P_{\Phi}$ on the observation $\mathbfit{u}$ . More formally: 
> 将定义 $P_\Phi$ 的因子集合根据上下文 $\pmb u$ 简化对应于将 $P_\Phi$ 条件于观测 $\pmb u$

**Proposition 4.1** Let $P_{\Phi}(X)$ be a Gibbs distribution. Then $P_{\Phi}[{\pmb u}]=P_{\Phi}({\pmb W}\mid{\pmb u})$ where $W=X-U$ . 
> 命题：
> 令 $P_{\Phi}(\pmb X)$ 为 Gibbs 分布，则 $P_{\Phi}[{\pmb u}]=P_{\Phi}({\pmb W}\mid{\pmb u})$ where $\pmb W=\pmb X-\pmb U$ . 
> 也就是简化的分布等于条件概率分布

Thus, to condition a Gibbs distribution on a context $\mathbfit{u}$ , we simply reduce every one of its factors to that context. Intuitively, the renormalization step needed to account for $\mathbfit{u}$ is simply folded into the standard renormalization of any Gibbs distribution. 
> 因此，要将 Gibbs 分布条件于某上下文，我们只需要将其因子简化到该上下文
> 直观上，缩减后需要的重规范化实际上在缩减 Gibbs 分布的标准规范化过程中完成了

This result immediately provides us with a construction for the Markov network that we obtain when we condition the associated distribution on some observation $\mathbfit{u}$ . 
> 我们接着考虑为条件于 $\pmb u$ 得到的简化的 Gibbs 分布构造 Markov 网络

**Deﬁnition 4.7** reduced Markov network 
Let $\mathcal{H}$ be a Markov network over $X$ $\pmb{U}=\pmb{u}$ a context. The reduced M kov network $\mathcal{H}[\pmb{u}]$ is a ne ork over the nodes $W=X-U$ , where we have an edge $X{-}Y$ if there is an edge $X{-}Y$ in $\mathcal H$
> 定义：
> $\mathcal H$ 为 $\pmb X$ 上的 Markov 网络，$\pmb U = \pmb u$ 为上下文，定义缩减的 Markov 网络 $\mathcal H[\pmb u]$ 为节点 $\pmb W = \pmb X - \pmb U$ 上的 Markov 网络，如果 $\mathcal H$ 中边 $X-Y$ 存在，则缩减的 Markov 网络中该边也存在

**Proposition 4.2** 
Let PΦ(X) be a Gibbs distribution that factorizes over H, and U = u a context. Then PΦ[u] factorizes over H[u].
> 命题：
> $P_\Phi (\pmb X)$ 为在 $\mathcal H$ 上分解的 Gibbs 分布，$\pmb U = \pmb u$ 为上下文，则 $P_\Phi[\pmb u]$ 在 $\mathcal H[\pmb u]$ 上分解
> （$P_\Phi(\pmb X)$ 在 $\mathcal H$ 上分解说明 $\mathcal H$ 包含了 $\Phi$ 中所有因子对应的完全子图，给定上下文，$\Phi$ 中所有因子关于上下文的项被移除，对应于完全子图中关于上下文的边、节点被移除，但是根据定义4.7，$\mathcal H[\pmb u]$ 中的其他边都保留于 $\mathcal H$，因此对于 $P_\Phi[\pmb u]$ 的分解性质显然保留

Note the contrast to the efect of conditioning in a Bayesian network: Here, conditioning on a context $\mathbfit{u}$ only eliminates edges from the graph; in a Bayesian network, conditioning on evidence can activate a $\mathrm{v}$ -structure, creating new dependencies. We return to this issue in section 4.5.1.1. 
> Markov 网络和 Bayesian 网络在条件下略有不同，Markov 网路中，条件的影响就是移除边和点，Bayesian 网络中，条件可能会激活 v-structure，创建新的依赖

Example 4.3 
Consider, for example, the Markov network shown in ﬁgure 4.6a; as we will see, this network is the Markov network required to capture the distribution encoded by an extended version of our Student Bayesian network (see ﬁgure 9.8). Figure 4.6b shows the same Markov network reduced over a context of the form $G=g$ , and (c) shows the network reduced over a context of the form $G=g,S=s$ . As we can see, the network structures are considerably simpliﬁed. 

Box 4.B — Case Study: Markov Networks for Computer Vision. 
One important application area for Markov networks is computer vision. Markov networks, typically called MRFs in this vision com- munity, have been used for a wide variety of visual processing tasks, such as image segmentation, removal of blur or noise, stereo reconstruction, object recognition, and many more. 

In most of these applications, the network takes the structure of a pairwise MRF, where the variables correspond to pixels and the edges (factors) to interactions between adjacent pixels in the grid that represents the image; thus, each (interior) pixel has exactly four neighbors. The value space of the variables and the exact form of factors depend on the task. These models are usually formulated in terms of energies (negative log-potentials), so that values represent “penalties,” and a lower value corresponds to a higher-probability conﬁguration. 

In image denoising , for example, the task is to restore the “true” value of all of the pixels given possibly noisy pixel values. Here, we have a node potential for each pixel $X_{i}$ that penalizes large discrepancies from the observed pixel value $y_{i}$ . The edge potential encodes a preference for continuity between adjacent pixel values, penalizing cases where the inferred value for $X_{i}$ is too 

far from the inferred pixel value for one of its neighbors $X_{j}$ . However, it is important not to overpenalize true disparities (such as edges between objects or regions), leading to oversmoothing of the image. Thus, we bound the penalty, using, for example, some truncated norm, as described in box $4.D$ : $\epsilon(x_{i},x_{j})=\operatorname*{min}(c\|x_{i}-x_{j}\|_{p},\mathrm{dist}_{\mathrm{max}})$ (for $p\in\{1,2\}$ ). 

Slight variants of the same model are used in many other applications. For example, in stereo reconstruction , the goal is to reconstruct the depth disparity of each pixel in the image. Here, the values of the variables represent some discretized version of the depth dimension (usually more ﬁnely discretized for distances close to the camera and more coarsely discretized as the distance from the camera increases). The individual node potential for each pixel $X_{i}$ uses standard techniques from computer vision to estimate, from a pair of stereo images, the individual depth disparity of this pixel. The edge potentials, precisely as before, often use a truncated metric to enforce continuity of the depth estimates, with the truncation avoiding an over pen aliz ation of true depth disparities (for example, when one object is partially in front of the other). Here, it is also quite common to make the penalty inversely proportional to the image gradient between the two pixels, allowing a smaller penalty to be applied in cases where a large image gradient suggests an edge between the pixels, possibly corresponding to an occlusion boundary. 

In image segmentation , the task is to partition the image pixels into regions corresponding to distinct parts of the scene. There are diferent variants of the segmentation task, many of which can be formulated as a Markov network. In one formulation, known as multiclass segmentation, each var ble $X_{i}$ has a domain $\{1,\cdot\cdot\cdot,K\}$ , where the value of $X_{i}$ represents a region assignment for pixel i (for example, grass, water, sky, car). Since classifying every pixel can be computationally expensive, some state-of-the-art methods for image segmentation and other tasks ﬁrst oversegment the image into superpixels (or small coherent regions) and classify each region — all pixels within a region are assigned the same label. The oversegmented image induces a graph in which there is one node for each superpixel and an edge between two nodes if the superpixels are adjacent (share a boundary) in the underlying image. We can now deﬁne our distribution in terms of this graph. 

Features are extracted from the image for each pixel or superpixel. The appearance features depend on the speciﬁc task. In image segmentation, for example, features typically include statistics over color, texture, and location. Often the features are clustered or provided as input to local classiﬁers to reduce dimensionality. The features used in the model are then the soft cluster assign- ments or local classiﬁer outputs for each superpixel. The node potential for a pixel or superpixel is then a function of these features. We note that the factors used in deﬁning this model depend on the speciﬁc values of the pixels in the image, so that each image deﬁnes a diferent probability distribution over the segment labels for the pixels or superpixels. In efect, the model used here is $^a$ conditional random ﬁeld , a concept that we deﬁne more formally in section 4.6.1. 

The model contains an edge potential between every pair of neighboring superpixels $X_{i},X_{j}$ . Most simply, this potential encodes a contiguity preference, with a penalty of λ whenever $X_{i}\neq X_{j}$ . Again, we can improve the model by making the penalty depend on the presence of an image gradient between the two pixels. An even better model does more than penalize discontinuities. We can have nondefault values for other class pairs, allowing us to encode the fact that we more often ﬁnd tigers adjacent to vegetation than adjacent to water; we can even make the model depend on the relative pixel location, allowing us to encode the fact that we usually ﬁnd water below vegetation, cars over roads, and sky above everything. 

Figure 4.B.1 shows segmentation results in a model containing only potentials on single pixels (thereby labeling each of them independently) versus results obtained from a model also containing pairwise potentials. The diference in the quality of the results clearly illustrates the importance of modeling the correlations between the superpixels. 

## 4.3 Markov Network Independencies 
In section 4.1, we gave an intuitive justiﬁcation of why an undirected graph seemed to capture the types of interactions in the Misconception example. We now provide a formal presentation of the undirected graph as a representation of independence assertions. 

### 4.3.1 Basic Independencies 
As in the case of Bayesian networks, the graph structure in a Markov network can be viewed as encoding a set of independence assumptions. Intuitively, in Markov networks, probabilistic inﬂuence “ﬂows” along the undirected paths in the graph, but it is blocked if we condition on the intervening nodes. 
> 和 Bayesian 网络类似，Markov 网络结构也可以视为编码了独立性假设，直观上，Markov 网络中，概率影响通过无向路径流动，如果观察到了中间节点，则被堵塞

**Deﬁnition 4.8** active path 
Let $\mathcal{H}$ be a Markov network stru $X_{1}\!-\!\ldots\!-\!X_{k}$ a path in $\mathcal{H}$ . L $Z\subseteq\mathcal{X}$ of observed variables . The path $X_{1}\!-\!\ldots\!-\!X_{k}$ is active given Z if none of the $X_{i}{\mathit{\Sigma}}_{\mathit{S}},$ $i=1,\ldots,k,$ is in $Z$ . 
> 定义：
> $\mathcal H$ 为 Markov 网络，$X_1-\dots-X_k$ 为 $\mathcal H$ 中的路径，令 $\pmb Z\subseteq \mathcal X$ 为观察到的变量，给定 $\pmb Z$，如果 $X_i, i=1,\dots, k$ 都不在 $\pmb Z$ 中，则路径 $X_1-\dots-X_k$ 是活跃的

Using this notion, we define a notion of seperation in the graph.

**Deﬁnition 4.9** separation
We say that a set of nodes $Z$ s $X$ $Y$ in $\mathcal{H}$ , noted $\mathrm{sep}_{\mathcal{H}}(X;Y\mid Z).$ , if there is no active path betw n any node $X\in X$ and $Y\in Y$  given Z . We deﬁne the global independencies associated with H to be: 

$$
{\mathcal{I}}({\mathcal{H}})=\{(\pmb X\perp \pmb Y\mid \pmb Z)\;:\;\mathrm{sep}_{{\mathcal{H}}}(\pmb X;\pmb Y\mid \pmb Z)\}.
$$ 
> 定义：
> 节点集合 $\pmb Z, \pmb X, \pmb Y$，如果在给定 $\pmb Z$ 下，在 $X\in \pmb X$ 和 $Y\in\pmb Y$ 之间没有活跃路径，则称 $\pmb Z$ 分离 $\pmb X, \pmb Y$，记作 $\text{sep}_{\mathcal H}(\pmb X;\pmb Y\mid \pmb Z)$
> 定义 $\mathcal H$ 相关的全局独立性如上
> 直观上看，节点集 $\pmb X, \pmb Y$ 被 $\pmb Z$ 分离，就是在给定 $\pmb Z$ 下条件独立

As we will discuss, the i epend cies in $\mathcal{T}(\mathcal{H})$ are precisely those that are guaranteed to hold for every distribution P over H . In other word the separation criterion is sound for detecting independence properties in distributions over H 
> $\mathcal I (\mathcal H)$ 中的独立性保证对于所有在 $\mathcal H$ 上分解的 $P$ 成立，也就是图中的分离准则对于检测图上的分布中的独立性是可靠的

Note that the deﬁnition ation is monotonic in Z , that is, if $s e p_{\mathcal{H}}(X;Y\mid Z)$ , then $s e p_{\mathcal{H}}(X;Y\mid Z^{\prime})$ for any $Z^{\prime}\supset Z$ ⊃ . Thus, if we take separation as our deﬁnition of the inde- pendencies induced by the network structure, we are efectively restricting our ability to encode nonmonotonic independence relations. Recall that in the context of intercausal reasoning in Bayesian networks, nonmonotonic reasoning patterns are quite useful in many situations — for example, when two diseases are independent, but dependent given some common symp- tom. The nature of the separation property implies that such independence patterns cannot be expressed in the structure of a Markov network. We return to this issue in section 4.5. 
> 注意分离的定义是单调的，也就是如果 $sep_{\mathcal H}(\pmb X; \pmb Y\mid \pmb Z)$ 成立，则 $sep_{\mathcal H}(\pmb X; \pmb Y\mid \pmb Z')$ 对于任意 $\pmb Z' \supset \pmb Z$ 成立

As for Bayesian networks, we can show a connection between the independence properties implied by the Markov network structure, and the possibility of factorizing a distribution over the graph. As before, we can now state the analogue to both of our representation theorems for Bayesian networks, which assert the equivalence between the Gibbs factorization of a distribution $P$ over a graph $\mathcal{H}$ and the assertion that $\mathcal{H}$ is an I-map for $P$ , that is, that $P$ satisﬁes the Markov assumptions $\mathcal{Z}(\mathcal{H})$ . 

#### 4.3.1.1 Soundness 
We ﬁrst consider the analogue to theorem 3.2, which asserts that a Gibbs distribution satisﬁes the independencies associated with the graph. In other words, this result states the soundness of the separation criterion. 
> 首先讨论分离准则的可靠性，也就是在图上分解的分布会满足由图导出的独立性推断

**Theorem 4.1** 
Let $P$ be a distribution over $\mathcal{X}$ and $\mathcal{H}$ a Ma kov netw structure over $\mathcal{X}$ . If $P$ is a Gibbs distribution that factorizes over $\mathcal H$, then $\mathcal H$ is an I-map for $P$ .
> 定理：
> $P$ 为 $\mathcal X$ 上的分布，$\mathcal H$ 为 $\mathcal X$ 上的 Markov 网络，若 $P$ 是在 $\mathcal H$ 上分解的 Gibbs 分布，则 $\mathcal H$ 是 $P$ 的 I-map

Proof Let $X,Y,Z$ be any three disjoint subsets in $\mathcal{X}$ such that $Z$ separates $X$ and $Y$ in $\mathcal{H}$ . We want to show that $P\models(X\ \bot\ Y\ |\ Z)$，We start by considering the se w e $X\cup Y\cup Z=\mathcal{X}$ . As $Z$ $X$ from $Y$ , there direct betw $X$ and Y . Hence, any clique in H is fully contained e $X\cup Z$ ∪ in Y $Y\cup Z$ ∪ . Let I $\mathcal{T}_{X}$ be the indexes of the set of cliques that are contained in $X\cup Z$ ∪ , and let I $\mathcal{T}_{Y}$ be the indexes of the remaining cliques. We know that 
> 证明：
> $\pmb X, \pmb Y, \pmb Z$ 为 $\mathcal X$ 中任意三个不相交子集，满足 $\pmb Z$ 在 $\mathcal H$ 中分离 $\pmb X, \pmb Y$，我们希望得到 $P$ 满足 $(\pmb X\perp \pmb Y \mid \pmb Z)$，也就是 $P \vDash (\pmb X\perp \pmb Y \mid \pmb Z)$
> 考虑 $\pmb X \cup \pmb Y \cup \pmb Z = \mathcal X$，因为 $\pmb Z$ 分离了 $\pmb X, \pmb Y$，故 $\pmb X, \pmb Y$ 之间是不直接相连的，因此，$\mathcal H$ 中的任意团都完全包含在 $\pmb X \cup \pmb Z$ 或 $\pmb Y \cup \pmb Z$ 中，令 $\mathcal I_{\pmb X}$ 表示包含在 $\pmb X \cup \pmb Z$ 中的团的索引，令 $\mathcal I_{\pmb Y}$ 表示包含在 $\pmb Y \cup \pmb Z$ 中的团的索引
> 因为 $P$ 在 $\mathcal H$ 上分解，因此 $P$ 的因子都在 $\mathcal H$ 中有对应的团，故 $P$ 可以写为：

$$
P(X_{1},\ldots,X_{n})={\frac{1}{Z}}\prod_{i\in{\mathcal{I}}_{\pmb X}}\phi_{i}(\pmb D_{i})\cdot\prod_{i\in{\mathcal{I}}_{\pmb Y}}\phi_{i}(\pmb D_{i}).
$$ 
As we discussed, none of the factors in the ﬁrst product involve any variable in $Y$ , and none in the second product involve any variable in $X$ . Hence, we can rewrite this product in the form: 
> 显然，第一个乘积中，没有因子包含 $\pmb Y$ 中的任意变量（因为都是对应 $\pmb X \cup \pmb Z$ 中的团），同理，第二个乘积中，没有因子包含 $\pmb X$ 中的任意变量，因此我们将该乘积重写为：

$$
P(X_{1},\dots,X_{n})=\frac{1}{Z}f(\pmb X,\pmb Z)g(\pmb Y,\pmb Z).
$$ 
From this decomposition, the desired independence follows immediately (exercise 2.5). 
> 根据 exercise2.5 的结论，可以知道 $P\vDash (\pmb X\perp \pmb Y\mid \pmb Z)$

Now consider the case where $X\cup Y\cup Z\subset\mathcal{X}$ . L $U\,=\,\mathcal{X}\,-\,(X\cup Y\cup Z)$ . We can rtition U into two disjoint sets $U_{1}$ and $U_{2}$ such th $Z$ $X\cup U_{1}$ $Y\cup U_{2}$ in H . Using the preceding argument, we conclude that $P\models(X,U_{1}\bot Y,U_{2}\mid Z)$ . Using the decomposition property (equation (2.8)), we conclude that $P\models(X\bot Y\mid Z)$.
> 再考虑 $\pmb X \cup \pmb Y \cup \pmb Z \subset \mathcal X$，令 $\pmb U = \mathcal X - (\pmb X \cup \pmb Y \cup \pmb Z)$
> 我们可以将 $\pmb U$ 划分为两个不相交子集 $\pmb U_1, \pmb U_2$，使得 $\pmb Z$ 在 $\mathcal H$ 中分离 $\pmb X \cup \pmb U_1, \pmb Y \cup \pmb U_2$，因此有 $P \vDash (\pmb X, \pmb U_1 \perp \pmb Y, \pmb U_2 \mid \pmb Z )$
> 根据条件独立性的分解形式 (eq2.8)，我们容易得到 $P\vDash (\pmb X\perp \pmb Y\mid \pmb Z)$

The other direction (the analogue to theorem 3.1), which goes from the independence properties of a distribution to its factorization, is known as the Hammersley-Cliford theorem . Unlike for Bayesian networks, this direction does not hold in general. As we will show, it holds only under the additional assumption that $P$ is a positive distribution (see deﬁnition 2.5). 
> 我们证明了如果 $P$ 在 $\mathcal H$ 上分解，则 $\mathcal H$ 是 $P$ 的 I-map，但另一个方向，即如果 $\mathcal H$ 是 $P$ 的 I-map，则 $P$ 在 $\mathcal H$ 上分解，则在 Markov 网络中不一定成立
> 它仅在 $P$ 是正分布的假设下（即只要事件 $A$ 不为空，则 $P(A) > 0$）成立，该定理称为 Hammersley-Cliford theorem

**Theorem 4.2** Hammersley-Cliford theorem
Let $P$ be a sitive distribution over $\mathcal{X}$ , and $\mathcal{H}$ a Marko etwork graph over $\mathcal{X}$ . If $\mathcal{H}$ is an I-map for P , then P is a Gibbs distribution that factorizes over . 
> 定理：
>  $P$ 是 $\mathcal X$ 上的正分布，$\mathcal H$ 是 $\mathcal X$ 上的 Markov 网络，如果 $\mathcal H$ 是 $P$ 的 I-map，则 $P$ 就是可以在 $\mathcal H$ 上分解的 Gibbs 分布

To prove this result, we would need to use the independence assumptions to construct a set of factors over $\mathcal{H}$ that give rise to the distribution $P$ . In the ca of Bayesian networks, these factors were simply CPDs, which we could derive directly from P . As we have discussed, the correspondence between the factors in a Gibbs distribution and the distribution $P$ is much more indirect. The construction required here is therefore signiﬁcantly more subtle, and relies on concepts that we develop later in this chapter; hence, we defer the proof to section 4.4 (theorem 4.8). 
> 要证明该定理，我们需要使用 $\mathcal H$ 中的独立性假设来在 $\mathcal H$ 上构造出一个因子集合，通过该集合可以得到 $P$
> 在 Bayesian 网络中，这些因子就是 CPDs，CPDs 可以直接从图中读出，但 GIbbs 分布和分布 $P$ 中的因子的对应关系更不直接，该定理的证明推迟到 Theorem 4.8

This result shows that, **for positive distributions, the global independencies imply that the distribution factorizes according the network structure. Thus, for this class of distritions, we have at a distribution $P$ factorizes over a Markov network $\mathcal{H}$ if and only if H is an I-map of P** . The positivity assumption is necessary for this result to hold: 
> 定理4.2表明：对于正分布，其全局独立性表明了分布根据网络结构分解，对于这类分布，我们有分布 $P$ 在 Markov 网络 $\mathcal H$ 上分解当且仅当 $\mathcal H$ 是 $P$ 的 I-map

Example 4.4 
Consider a distribution $P$ over four binary random variables $X_{1},X_{2},X_{3},X_{4},$ which gives proba- bility $1/8$ to each of the following eight conﬁgurations, and probability zero to all others:  Let $\mathcal{H}$ be e graph $X_{1}{-}X_{2}{-}X_{3}{-}X_{4}{-}X_{1}$ . Then $P$ satisﬁes the global independencies with $\mathcal{H}$ ample, consider the independence $(X_{1}\perp X_{3}\mid X_{2},X_{4})$ . For the assignment $X_{2}=x_{2}^{1},X_{4}=x_{4}^{0}$ , we have that only assignments where $X_{1}=x_{1}^{1}$ receive positive probability. Thus, $P(x_{1}^{1}\mid x_{2}^{1},x_{4}^{0})=1$ | , and $X_{1}$ is trivially independent of $X_{3}$ in this conditional distribution. A similar analysis applies to all other cases, so that the global independencies hold. However, the distribution $P$ does not factorize according to $\mathcal{H}$ . The proof of this fact is left as an exercise (see exercise 4.1). 

#### 4.3.1.2 Completeness 
The preceding discussion shows the soundness of the separation condition as a criterion for detecting independencies in Markov networks: any distribution that factorizes over $\mathcal{G}$ satisﬁes the independence assertions implied by separation. The next obvious issue is the completeness of this criterion.
> 上一小节讨论了用分离条件作为准则检测 Markov 网络中的独立性的可靠性：**任意**在 $\mathcal G$ 上分解的分布都满足 $\mathcal G$ 中通过分离条件表明的独立性
> （可靠性：图中分离 --> 分布中独立，对于任意分解于图的分布成立）
> 本节讨论该准则的完备性

As for Bayesian networks, the strong version of completeness does not hold in this setting. In other words, it is not the case that ery pair of nodes $X$ d $Y$ that are not separated in $\mathcal{H}$ are dependent in every distribution $P$ which factorizes over H . However, as in theorem 3.3, we can use a weaker deﬁnition of completeness that does hold: 
> 对于贝叶斯网络，强完备性成立，也就是说，$\mathcal G$ 没有被分离 (d-seperation) 的两个节点 $X, Y$ 在分解于 $\mathcal G$ 上的分布 $P$ 中一定是依赖的，也就是图中的分离可以检测到分布中所有的独立性
> （完备性：图中不分离 --> 分布中不独立，对于任意分解于图的分布成立）
> 但 Markov 网络中，强完备性不成立，也就是说，$\mathcal H$ 中没有被分离的两个节点 $X, Y$ 在分解于 $\mathcal H$ 上的分布 $P$ 中不一定是依赖的，可能是独立的，也就是图中的分离不能检测出分布中所有的独立性
> Markov 网络中，我们可以定义一种较弱的完备性

**Theorem 4.3**
Let $\mathcal{H}$ be a Markov net rk structure. If $X$ an $Y$ are not separated iven $Z$ in $\mathcal{H}$ , then $X$ and $Y$ are dependent given Z in some distribution P that factorizes over H . 
> 定理：
> $\mathcal H$ 为 Markov 网络，如果在 $\mathcal H$ 中，$X, Y$ 在给定 $\pmb Z$ 时不分离，则 $X, Y$ 在**某个**分解于 $\mathcal H$ 上的分布 $P$ 中给定 $\pmb Z$ 时是依赖的

Proof e pro is a constructive one: we construct a distribution $P$ that factorizes over $\mathcal{H}$ where X and Y $Y$ are dependent. We assume, without loss of generality, that all variables are binary-valued. If this is not the case, we can treat them as binary-valued by restricting attention to two distinguished values for each variable. 
> 证明：
> 构造性证明，我们构造一个分解于 $\mathcal H$ 的分布 $P$，使得 $X, Y$ 在 $P$ 中给定 $\pmb Z$ 是依赖的
> 不失一般性，假设都为二值变量，如果变量不是二值变量，我们可以将注意限制到每个变量的两个不同值，将其视为二值变量

By assumption, $X$ and $Y$ $Z$ $\mathcal{H}$ ; hence, they must be connected by some unblocked trail. Let $X=U_{1}-U_{2}-\ldots-U_{k}=Y$ be some minimal trail in the graph such that, for all $i$ , $U_{i}\notin Z$ here we deﬁne a m trail in $\mathcal{H}$ to be a path with no shortcuts: thus, for any i and $j\neq i\pm1$ ̸ ± , there is no edge $U_{i}{-}U_{j}$ . We can always ﬁnd such a path: If we have a nonminimal path where we have $U_{i}{-}U_{j}$ for $j\,>\,i+1$ , we can always “shortcut” the original trail, converting it to one that goes directly from $U_{i}$ to $U_{j}$ . 
> $X, Y$ 在给定 $\pmb Z$ 时在 $\mathcal H$ 中不分离，因此二者必须通过某个未被堵塞的迹相连
> 不妨设 $X=U_1-U_2-\dots-U_k = Y$ 为图中满足该性质的极小的迹，满足 for all $i$, $U_i \not\in \pmb Z$
> 我们定义 $\mathcal H$ 中极小的迹为没有捷径的迹，即对于任意 $i$ 和 $j\ne i\pm 1$，不存在边 $U_i-U_j$
> 我们总可以找到这样一条极小的迹，如果有一条非极小的迹存在 $U_i - U_j,\quad j > i+1$，我们可以对这个迹取捷径，以此得到极小的迹

For any $i=1,\ldots,k-1$ , as there is an edge $U_{i}{-}U_{i+1}$ , i ollows that $U_{i},U_{i+1}$ must both appear in some clique $C_{i}$ . We pick some very large weight W , and for each i we deﬁne the clique potential $\phi_{i}(\boldsymbol{C}_{i})$ to assign weight $W$ if $U_{i}=U_{i+1}$ and weight 1 otherwise, regardless of the values of the other variables in the clique. Note that the cliques $C_{i}$ for $U_{i},U_{i+1}$ and $C_{j}$ for $U_{j},U_{j+1}$ must be diferent cliques: If $C_{i}=C_{j}$ , then $U_{j}$ is in the same clique as $U_{i}$ , and we have an edge $U_{i}{-}U_{j}$ , contradicting the minimality of the trail. Hence, we can deﬁne the clique potential for each clique $C_{i}$ separately. We deﬁne the clique potential for any other clique to be uniformly 1 . 
> 对于任意 $i=1,\dots, k-1$，存在边 $U_i - U_{i+1}$，因此 $U_i, U_{i+1}$ 一定都出现在某个团 $\pmb C_i$ 中
> 我们选定一个很大的权重 $W$，对于每个 $i$，我们定义团势能函数 $\phi_i (\pmb C_i)$，如果 $U_i = U_{i+1}$，赋予它权重 $W$，否则赋予权重 1，权重的赋予和团中其他的变量的值无关
> 注意 $U_i, U_{i+1}$ 的团 $\pmb C_i$ 和 $U_j, U_{j+1}$ 的团 $\pmb C_j$ 必须是不同的团，因为迹是极小的，因此 $U_i$ 和 $U_j$ 之间显然不可能有边，这允许我们为每个 $\pmb C_i$ 分别定义势能函数
> 我们将其他的团的势能都定义为 1

We now consider the distribution $P$ resulting from multiplying all of these clique potentials. Intuitively, the distribution $P(U_{1},.\,.\,,U_{k})$ is simply the distribution deﬁned by multiplying the pairwise factors for the pairs $U_{i},U_{i+1}$ , regardless of the other variables (including the ones in $Z)$ ). One can verify that, in $P(U_{1},\cdot\cdot\,,U_{k})$ , we have that $X=U_{1}$ and $Y=U_{k}$ are dependent. We leave the conclusion of this argument as an exercise (exercise 4.5). 
> 现在考虑将这些团势能函数全部相乘得到的分布 $P$
> 直观上，$P (U_1,\dots, U_k)$ 就是通过由 $U_i, U_{i+1}$ 定义的因子相乘得到的分布，与其他变量无关，包括 $\pmb Z$
> 可以验证 $P (U_1,\dots, U_k)$ 中，$X = U_1, Y = U_k$ 是相互依赖的

We can use the same argument as theorem 3.5 to conclude that, for almost all distributions $P$ that factorize over $\mathcal{H}$ (that is, for all distributions except for a set of measure zero in the space of factor parameter iz at ions) we have that $\mathcal{I}(P)=\mathcal{I}(\mathcal{H})$ . 
> 事实上，对于几乎所有在 $\mathcal H$ 上分解的分布 $P$（也就是除了在因子参数化空间中测度为零的一组分布），都有 $\mathcal I (P) = \mathcal I (\mathcal H)$

Once again, we can view this result as telling us that our deﬁnition of $\mathcal{T}(\mathcal{H})$ is t maximal one. For any independence assertion tha is not a consequen of separation in H , we can always ﬁnd a counterexample distribution P that factorizes over . 
> 可以认为该结果说明了我们对 $\mathcal I (\mathcal H)$ 的定义就是极大的，对于任意不是 $\mathcal H$ 分离的结果的独立性断言，我们总能在分解于 $\mathcal H$ 的分布中找到一个反例

### 4.3.2 Independencies Revisited 
When characterizing the independencies in a Bayesian network, we provided two deﬁnitions: the local independencies (each node is independent of its nondescendants given its parents), and the global independencies induced by d-separation. As we showed, these two sets of independencies are equivalent, in that one implies the other. 
> Bayesian 网络中的独立性被我们分为了两类：
> - 局部独立性：给定父变量，变量和所有非后继条件独立
> - 全局独立性：由 d-seperation 推导出的独立性
> 这两类独立性实际上等价，也就是可以根据局部独立性的定义推导出网络中的全局独立性，也可以根据全局独立性的定义推导出网络中的局部独立性

So far, our discussion for Markov networks provides only a global criterion. While the global criterion characterizes the entire set of independencies induced by the network structure, a local criterion is also valuable, since it allows us to focus on a smaller set of properties when examining the distribution, signiﬁcantly simplifying the process of ﬁnding an I-map for a distribution $P$ . 
> 我们目前仅讨论了 Markov 网络中的全局独立性

Thus, it is natural to ask whether we can provide a local deﬁnition of the independencies induced by a Markov network, analogously to the local independencies of Bayesian networks. Surprisingly, as we now show, in the context of Markov networks, there are three diferent possible deﬁnitions of the independencies associated with the network structure — two local ones and the global one in deﬁnition 4.9. While these deﬁnitions are related, they are equivalent only for positive distributions. 
> 类似贝叶斯网络，我们希望提供一个 Markov 网络中独立性的局部定义
> 对于 Markov 网络，存在三种可能的独立性定义，两个局部一个全局
> 这三个定义仅在分布是正分布时才等价

As we will see, nonpositive distributions allow for deterministic dependencies between the variables. Such deterministic interactions can “fool” local indepen- dence tests, allowing us to construct networks that are not I-maps of the distribution, yet the local independencies hold. 
> 非正分布允许变量间的确定性依赖，这会扰乱局部独立性测试，也就是可以借由它构建不是分布的 I-map 的网络 （不满足 $\mathcal I(\mathcal H) \subseteq \mathcal I(P)$），但分布的局部独立性在网络中成立

#### 4.3.2.1 Local Markov Assumptions 
The ﬁrst, and weakest, deﬁnition is based on the following intuition: Whenever two variables are directly connected, they have the potential of being directly correlated in a way that is not mediated by other variables. Conversely, when two variables are not directly linked, there must be some way of rendering them conditionally independent. Speciﬁcally, we can require that $X$ and $Y$ be independent given all other nodes in the graph. 
> 先介绍Markov 网络的第一个，也是最弱的独立性定义
> 它的直觉为：当两个变量直接相连，它们就有可能相互直接影响，而不经过中间变量；并且，如果两个变量没有直接相连，一定可以找到条件使二者条件独立，例如，我们可以要求给定图中所有其他节点，$X, Y$ 条件独立

**Deﬁnition 4.10** pairwise independencies 
Let $\mathcal{H}$ be a Markov network. We deﬁne the pairwise independencies associated with $\mathcal{H}$ to be: ${\mathcal{I}}_{p}({\mathcal{H}})=\{(X\perp Y\mid{\mathcal{X}}-\{X,Y\})\;:\;X{\mathrm{-}}Y\notin{\mathcal{H}}\}.$ 
> 定义：
> 令 $\mathcal H$ 为 Markov 网络，定义和 $\mathcal H$ 相关的成对独立性为：
> $\mathcal I_p (\mathcal H) = \{(X\perp Y \mid \mathcal X - \{X, Y\}): X-Y\not\in \mathcal H\}$

Using this deﬁnition, we can easily represent the independencies in our Misconception example using a Markov network: We simply connect the nodes up in exactly the same way as the interaction structure between the students. 

The second local deﬁnition is an undirected analogue to the local independencies associated with a Bayesian network. It is based on the intuition that we can block all inﬂuences on a node by conditioning on its immediate neighbors. 
> 第二个独立性定义类似贝叶斯网络中的局部独立性定义
> 它的直觉为：通过条件于某个节点的所有直接邻居，可以阻塞所有对该节点的影响

**Deﬁnition 4.11** Markov blanket 
For a given aph $\mathcal{H}$ , we deﬁne the Markov blanket of $X$ in $\mathcal{H}$ , denote $\mathrm{MB}_{\mathcal{H}}(X)$ , to be the neighbors of X in . We deﬁne the local independencies associated with to be: 
> 定义：
> 对于图 $\mathcal H$，定义 $\mathcal H$ 中 $X$ 的 Markov 毯为 $X$ 所有的邻居变量，记作 $\text{MB}_{\mathcal H}(X)$
> 然后，定义和 $\mathcal H$ 相关的局部独立性为：

$$
{\mathcal{I}}_{\ell}({\mathcal{H}})=\{(X\perp{\mathcal{X}}-\{X\}-\mathrm{MB}_{{\mathcal{H}}}(X)\mid\mathrm{MB}_{{\mathcal{H}}}(X))\;:\;X\in{\mathcal{X}}\}.
$$ 
In other words, the local independencies state that $X$ is independent of the rest of the nodes in the graph given its immediate neighbors. 
> 局部独立性的含义即 $X$ 在给定它的直接邻居节点/Markov 毯时，对 $\mathcal H$ 中剩余的所有节点都条件独立

We will show that these local independence assumptions hold for any distribution that factorizes over $\mathcal{H}$ , so that $X$ ’s Markov blanket in $\mathcal{H}$ truly does separate it from all other variables. 
> 这两个局部独立性假设对于任意分解于 $\mathcal H$ 的分布 $P$ 都成立
> 故 $X$ 在 $\mathcal H$ 中的 Markov blanket 确实在分布中将 $X$ 和其他变量分离
#### 4.3.2.2 Relationships between Markov Properties 
We have now presented three sets of independence assertions associated with a network struc- ture $\mathcal{H}$ . For general distributions, $\mathcal{T}_{p}(\mathcal{H})$ is strictly weaker than $\mathcal{T}_{\ell}(\mathcal{H})$ , which in turn is strictly weaker than $\mathcal{T}(\mathcal{H})$ . However, all three deﬁnitions are equivalent for positive distributions. 
> 目前为止我们讨论了三组和 $\mathcal H$ 相关的独立性断言
> 对于一般的分布，$\mathcal I_p(\mathcal H)$ 严格弱于 $\mathcal I_{\mathscr l}(\mathcal H)$，而 $\mathcal I_{\mathscr l}(\mathcal H)$ 严格弱于 $\mathcal I (\mathcal H)$
> 但对于正分布，这三个定义等价

**Proposition 4.3** 
For any Markov network H , and any distribution $P$ , we have that if $P\vDash\mathcal{Z}_{\ell}(\mathcal{H})$ I H then $P\vDash\mathcal{Z}_{p}(\mathcal{H})$ I H . The proof of this result is left as an exercise (exercise 4.8). 
> 命题：
> 对于任意 Markov 网络 $\mathcal H$ 和任意分布 $P$，如果 $P\vDash \mathcal I_{\mathscr l} (\mathcal H)$，则 $P\vDash \mathcal I_p (\mathcal H)$
>（如果在 $P$ 中存在独立性 $\mathcal I_{\mathscr l}(\mathcal H)$，则 $P$ 中存在独立性 $\mathcal I_p(\mathcal H)$）

**Proposition 4.4**
For any Markov network H , and any distribution $P$ , we have that if $P\vDash\mathcal{Z}(\mathcal{H})$ I H then $P\vDash\mathcal{Z}_{\ell}(\mathcal{H})$ I H . 
> 命题：
> 对于任意 Markov 网络 $\mathcal H$ 和任意分布 $P$，如果 $P\vDash \mathcal I (\mathcal H)$，则 $P\vDash \mathcal I_{\mathscr l}(\mathcal H)$
>（如果在 $P$ 中存在独立性 $\mathcal I(\mathcal H)$，则 $P$ 中存在独立性 $\mathcal I_{\mathscr l}(\mathcal H)$）

The proof of this result follows directly from the fact that if $X$ and $Y$ are not connected by an edge, then they are necessarily separated by all of the remaining nodes in the graph. 

The converse of these inclusion results holds only for positive distributions (see deﬁnition 2.5). More speciﬁcally, if we assume the intersection property (equation (2.11)), all three of the Markov conditions are equivalent. 
> 命题4.3和命题4.4的包含关系的反过来仅对于正分布成立，或者说对于具有 intersection 性质的分布成立

**Theorem 4.4** 
Let $P$ be a positive distribution. If $P$ satisﬁes $\mathcal{I}_{p}(\mathcal{H})$ , then $P$ satisﬁes $\mathcal{I}(\mathcal{H})$ . 
> 定理：
> $P$ 为正分布，如果 $P$ 满足 $\mathcal I_p (\mathcal H)$，则 $P$ 满足 $\mathcal I (\mathcal H)$

Proof We want to prove that for all disjoint sets $X,Y,Z$ : 

$$
s e p_{\mathcal{H}}(X;Y\mid Z)\Longrightarrow P\vDash(X\perp Y\mid Z).\tag{4.1}
$$ 
The proof proceeds by descending induction on the size of $Z$ . 
> 证明：
> 回忆一下 $\mathcal I (\mathcal H)$ 的定义为 ${\mathcal{I}}({\mathcal{H}})=\{(\pmb X\perp \pmb Y\mid \pmb Z)\;:\;\mathrm{sep}_{{\mathcal{H}}}(\pmb X;\pmb Y\mid \pmb Z)\}$，故我们需要证明在 $P$ 满足 $\mathcal I_p (\mathcal H)$ 的条件下，对于所有图中的 $sep_\mathcal H (\pmb X; \pmb Y \mid \pmb Z)$，在 $P$ 中成立 $(\pmb X \perp \pmb Y \mid \pmb Z)$
> 证明的思路是用归纳法，逐渐减少 $\pmb Z$ 的大小

The base case is $|Z|=n-2;$ ; equation (4.1) follows immediately fr the deﬁnition of $\mathcal{I}_{p}(\mathcal{H})$ . 
> 基例是 $|\pmb Z| = n-2$，此时 $\pmb X, \pmb Y, \pmb Z$ 满足 $sep_\mathcal H (\pmb X; \pmb Y\mid \pmb Z)$，显然，$\pmb X, \pmb Y$ 此时仅是两个节点，并且它们没有直接相连的边，满足 $(\pmb X \perp \pmb Y \mid \pmb Z)\in \mathcal I_p (\mathcal H)$，而 $P$ 满足 $\mathcal I_p (\mathcal H)$ ，故 $P\vDash (\pmb X \perp \pmb Y \mid \pmb Z)$

For the inductive step, assume that equation (4.1) holds for every $Z^{\prime}$ with size $|Z^{\prime}|=\dot{k}$ , and let Z be any set su $|Z|=k-1$ . We distinguish between two case 
> 归纳推理：
> 假设对于大小为 $k$ 的 $\pmb Z'$，式4.1成立，令 $\pmb Z$ 为任意满足 $|\pmb Z| = k-1$ 的集合，分为两类讨论

In the ﬁrst case, $X\cup Z\cup Y=\mathcal{X}$ ∪ ∪ X . As $|Z|<n-2$ , we hat either $|X|\geq2$ or $|Y|\geq2$ . Without loss of generality, assume that the latter holds; let $A\in Y$ ∈ and $Y^{\prime}=Y\!-\!\{A\}$ . From the t that $s e p_{\mathcal{H}}(X;Y\mid Z)$ , we also have that $s e p_{\mathcal{H}}(X;Y^{\prime}\mid Z)$ on one hand and $s e p_{\mathcal{H}}(\boldsymbol{X};A\mid$ $Z)$ on the other hand. As separation is monotonic, we also have $s e p_{\mathcal H}(X;Y^{\prime}\mid Z\cup\{A\})$ and $s e p_{\mathcal{H}}(X;A\mid Z\cup Y^{\prime})$ . The separating sets $Z\cup\{A\}$ and $Z\cup Y^{\prime}$ ∪ are ach at least size $|Z|+1=k$ in size, so that equation (4.1) applies, and we can conclude that P satisﬁes: 

$$
(X\bot Y^{\prime}\mid Z\cup\{A\})\quad\&\quad\ (X\bot A\mid Z\cup Y^{\prime}).
$$ 
Because $P$ is positive, we can apply the intersection property (equation (2.11)) and conclude that $P\models(X\ \bot\ Y^{\prime}\cup\{A\}\mid Z)$ , that is, $(X\perp Y\mid Z)$ . 
> 第一例：
> 当 $\pmb X \cup \pmb Z \cup \pmb Y = \mathcal X$，因为 $|\pmb Z | <n-2$，故 $|\pmb X | \ge 2$ 或 $|\pmb Y| \ge 2$
> 假设 $|\pmb Y | \ge 2$，令 $A\in \pmb Y$，$\pmb Y' = \pmb Y - \{A\}$，
> 因为 $sep_{\mathcal H}(\pmb X; \pmb Y \mid \pmb Z)$，故显然 $sep_{\mathcal H}(\pmb X; \pmb Y' \mid \pmb Z)$ 和 $sep_{\mathcal H}(\pmb X; A\mid \pmb Z)$ 也成立
> 因为 seperation 是单调的，故显然 $sep_{\mathcal H}(\pmb X; \pmb Y' \mid \pmb Z\cup \{A\})$ 和 $sep_{\mathcal H}(\pmb X; A\mid \pmb Z\cup \pmb Y')$ 也成立
> 而其中的分离集合 $\pmb Z \cup \{A\}$ 和 $\pmb Z \cup \pmb Y$ 的大小都至少是 $|\pmb Z| + 1 =k$，故式4.1成立，也就是  $P\vDash(\pmb X\perp \pmb Y' \mid \pmb Z\cup \{A\})$ 和 $P\vDash(\pmb X\perp A\mid \pmb Z\cup \pmb Y')$ 成立
> 根据正分布的 intersection 性质，我们可以得到
> $P\vDash (\pmb X\perp \pmb Y' \cup \{A\} \mid \pmb Z)$，也就是 $P\vDash (\pmb X\perp \pmb Y \mid \pmb Z)$

The second case is where $X\cup Y\cup Z\neq X$ . Here, we might have that both $X$ and $Y$ are singletons. This case requires a similar argument that uses the induction hypothesis and properties of independence. We leave it as an exercise (exercise 4.9). 
> 第二例的情况是 $\pmb X \cup \pmb Y\cup \pmb Z \ne \mathcal X$，但归纳推理是类似的，如果 $\pmb X, \pmb Y$ 都仅含单个节点，则二者的条件独立性直接包含在了 $\mathcal I_p (\mathcal H)$ 中，如果 $\pmb X, \pmb Y$ 包含多个节点，则推导和上面完全类似，同样满足 $\pmb Z \cup \{A\}$ 和 $\pmb Z \cup \pmb A$ 的大小至少是 $|\pmb Z | + 1 = k$

Our previous results entail that, for positive distributions, the three conditions are equivalent. 
> 根据定理4.4，对于正分布，三个独立性定义是等价的

**Corollary 4.1** 
The following three statements are equivalent for a positive distribution $P$ : 
> 引理：
> 以下三个表述对于正分布 $P$ 等价

1. $P\vDash\mathcal{I}_{\ell}(\mathcal{H})$ .
2. $P\vDash\mathcal{I}_{p}(\mathcal{H})$ .
3. $P\vDash\mathcal{I}(\mathcal{H})$ . 

This equivalence relies on the positivity assumption. In particular, for nonpositive distributions, we can provide examples of a distribution $P$ that satisﬁes one of these properties, but not the stronger one. 

Example 4.5 
Let $P$ be any distribution over ${\mathcal{X}}=\{X_{1},.\,.\,.\,,X_{n}\}.$ ; let $\mathcal{X}^{\prime}=\{X_{1}^{\prime},\cdot\cdot\cdot,X_{n}^{\prime}\}$ } . We now construct a distribution $P^{\prime}(\mathcal{X},\mathcal{X}^{\prime})$ whose arginal over $X_{1},\ldots,X_{n}$ is the same as $P$ , and where $X_{i}^{\prime}$ is deterministically equal to $X_{i}$ . t H be a Markov network over $\mathcal{X},\mathcal{X}^{\prime}$ that contains no edges other than $X_{i}{-}X_{i}^{\prime}$ . Then, in P $P^{\prime}$ ′ , $X_{i}$ is independent of the rest of the variables in the network given its neighbor $X_{i}^{\prime}$ , and similarly for $X_{i}^{\prime}$ thus, $\mathcal{H}$ isﬁes th local independencies for every node in the network. clearly $\mathcal{H}$ is not an I ap for $P^{\prime}$ ′ , since H makes many independence assertions regarding the $X_{i}$ ’s that do not hold in P (or in P $P^{\prime}$ ). 

Thus, for nonpositive distributions, the local independencies do not imply the global ones. A similar construction can be used to show that, for nonpositive distributions, the pairwise independencies do necessarily imply the local independencies. 
> 对于非负分布，满足图中的局部独立性并不表明它满足图中的全局独立性；满足图中的成对独立性并不表明它满足图中的局部独立性

Example 4.6
Let $P$ any tribution ove $\mathcal{X}=\{X_{1},\ldots,X_{n}\}$ , and now consider two auxiliary sets of vari- ables X $\mathcal{X}^{\prime}$ and X $\mathcal{X}^{\prime\prime}$ , and deﬁne X $\mathcal{X}^{\ast}=\mathcal{X}\cup\mathcal{X}^{\prime}\cup\mathcal{X}^{\prime\prime}$ X ∪X X . We now construct a distribution $P^{\prime}(\mathcal{X}^{*})$ whose marginal over $X_{1},\dots,X_{n}$ is the same as P , and where $X_{i}^{\prime}$ and $X_{i}^{\prime\prime}$ are both deterministic ally equal to $X_{i}$ . Let $\mathcal{H}$ be the empty Markov network over $\mathcal{X}^{*}$ . We argue that this empty network satisﬁes the pairwise assumptions for every pair of nodes in the network. For example, $X_{i}$ and $X_{i}^{\prime}$ are rendered independent ecause $\mathcal{X}^{*}\,-\,\{X_{i},X_{i}^{\prime}\}$ } contains $X_{i}^{\prime\prime}$ . Similarly, $X_{i}$ and $X_{j}$ are independent given $X_{i}^{\prime}$ . Thus, H satisﬁes the pairwise independencies, but not the local or global independencies. 
> Example 4.5和4.6都用了同一个 trick，就是引入确定性关系，使得构建出的分布出现在给定 $X'$ 时 $X$ 独立于所有其他变量这样的独立性，而这样的独立性在目前的 Markov 网络语义中显然是不包含的（目前的 Markov 网络仅用节点阻塞这样的方式表示独立性语义），故分布中的独立性超出了网络语义的表示范围，网络无法检测到分布中的全部独立性，导致完备性不成立
> 如果是正分布，分布中不存在概率为0的事件，也就是不存在确定性事件（概率为0的事件就是确定性事件取反），就不会有这样的问题

### 4.3.3 From Distributions to Graphs 
Based on our deeper understanding of the independence properties associated with a Markov network, we can now turn to the question of encoding the independencies in a given distribution $P$ using a graph structure. As for Bayesian networks, the notion of an I-map is not sufcient by itself: The complete graph implies no independence assumptions and is hence an I-map for any distribution. We therefore return to the notion of a minimal I-map, deﬁned in deﬁnition 3.13, which was deﬁned broadly enough to apply to Markov networks as well. 
> 本节讨论从分布到图，也就是用图编码给定分布中的独立性
> 对于贝叶斯网络，构建 I-map 本身也是不充分的，例如构建一个完全图，图结构不编码任何独立性信息，该图也是任何分布 I-map
> 我们故而考虑极小 I-map 的概念，也就是图中任意删除一条边都会导致图不再是 I-map，即多出额外的独立性

How can we construct a minimal I-map for a distribution $P$ ? Our discussion in section 4.3.2 immediately suggests two approaches for constructing a minimal I-map: one based on the pairwise Markov independencies, and the other based on the local independencies. 
> 考虑基于成对 Markov 独立性和局部独立性构建极小 I-map

In the ﬁrst approach, we consider the pairwise independencies. They assert that, if the edge $\{X,Y\}$ is not in $\mathcal{H}$ , then $X$ and $Y$ must be independent given all other nodes in the grap regardless of which other edges the graph contains. Thus, at the very least to guarantee that H is an I-map, we must add direct edges between all pairs of nodes X and Y such that 

$$
P\not\vDash(X\perp Y\mid{\mathcal{X}}-\{X,Y\}).\tag{4.2}
$$ 
We can now deﬁne $\mathcal{H}$ to include an edge $X{-}Y$ for all $X,Y$ for which equation (4.2) holds. 
> 对于成对独立性：
> 成对独立性即如果 $X-Y$ 不在 $\mathcal H$ 中，则 $X, Y$ 必须在给定图中所有其他节点的情况下条件独立，为此，在完全图的基础上，我们首先移除分布中存在成对独立性的变量之间直接相连的边
> 反过来说，就是在满足 $P\not \vDash (X\perp Y\mid \mathcal X - \{X, Y\})$ (即不存在成对独立性) 的变量 $X, Y$ 之间添加边

In the second approach, we use the local independencies and the notion of minimality. For each variable $X$ , we deﬁne the neighbors of $X$ to be a minimal set of nodes $Y$ that render $X$ independent of the rest of the nodes. More precisely, deﬁne: 
> 关于局部独立性：
> 我们首先为每个变量 $X$ 定义它的 Markov 毯，也就是可以让变量 $X$ 和其余节点条件独立的极小的节点集合

**Deﬁnition 4.12** Markov blanket 
A set $U$ is $a$ Markov blanket of $X$ in a distribution $P$ if $X\not\in U$ and if $U$ is a minimal set of nodes such that 

$$
(X\perp{\mathcal{X}}-\{X\}-U\mid U)\in{\mathcal{I}}(P).\tag{4.3}
$$ 
> 定义：
> 对于分布 $P$ 中的变量 $X$，它的 Markov 毯定义为一个节点集合 $\pmb U$，满足 $X\not\in \pmb U$，并且 $\pmb U$ 是满足 $(X\perp \mathcal X - \{X\} - \pmb U \mid \pmb U) \in \mathcal I (P)$ 的极小的节点集合

We then deﬁne a graph $\mathcal{H}$ by introducing an edge $\{X,Y\}$ for all $X$ and $Y\in{\mathrm{MB}}_{P}(X)$ . As deﬁned, this construction is not unique, since there may be several sets U satisfying equation (4.3). However, theorem 4.6 will show that there is only one such minimal set. 
> 因此，在图 $\mathcal H$ 中，对于所有的 $X$，我们需要为 $Y \in \text{MB}_P (X)$ 引入边 $X-Y$

In fact, we now show that any positive distribution $P$ has a unique minimal I-map, and that both of these constructions produce this I-map. 

We begin with the proof for the pairwise deﬁnition: 

**Theorem 4.5** 
Let $P$ be a positive distribution, and let $\mathcal{H}$ be deﬁned by roducing an edge $\{X,Y\}$ for all $X,Y$ for which equation (4.2) holds. Then the Markov network H is the unique minimal I-map for P . 
> 定理：
> $P$ 为正分布，$\mathcal H$ 通过为满足式4.2的所有 $\{X, Y\}$ 引入边得到，则 $\mathcal H$ 是 $P$ 的唯一极小 I-map

Proof The fact that $\mathcal{H}$ is an I-map for $P$ follows immediately from fact that $P$ , by construction, satisﬁes $\mathcal{T}_{p}(\mathcal{H})$ , and, therefore, by corollary 4.1, also s $\mathcal{Z}(\mathcal{H})$ The fact that it is minimal follows from the fact that if we eliminate some edge { $\{X,Y\}$ } from H , the graph wo d imply the pairwise independence $(X\perp Y\mid{\mathcal{X}}-\{X,Y\})$ , which w know to be false for P (otherwise, the edge would have been omitted in the construction of H ). The un enes f the minimal I-map also follows tr ally: By the same argument, an other I-map H for P must contain at least the edges in H and is therefore either equal to H or contains additional edges and is therefore not minimal. 
> 证明：
> 显然，在构造上，我们可以知道 $\mathcal H$ 是 $P$ 的 I-map，$P$ 满足 $\mathcal I_p (\mathcal H)$，故根据引理4.1，可以知道 $P$ 满足 $\mathcal I (\mathcal H)$
> 极小性：如果从图中移除某个边 $\{X, Y\}$，则 $\mathcal H$ 就会编码成对独立性 $(X\perp Y \mid \mathcal X - \{X, Y\})$，而在 $P$ 中该成对独立性是不成立的，否则 $X-Y$ 一开始就不会被加入，因此，$\mathcal H$ 是极小的 I-map
> 独立性：$P$ 的其他 I-map $\mathcal H'$ 至少需要包含和 $\mathcal H$ 同样的边，如果没有额外的边，就等于 $\mathcal H$，因此要保持极小性，就需要等于 $\mathcal H$

It remains to show that the second deﬁnition results in the same minimal I-map. 

**Theorem 4.6** 
Let $P$ be a positive distribution. For each node $X$ , let ${\mathrm{MB}}_{P}(X)$ be a minimal set of nodes $U$ satisfying equation (4.3). We deﬁne a grap $\mathcal{H}$ by introducing an ed e $\{X,Y\}$ for all $X$ and all $Y\in{\mathrm{MB}}_{P}(X)$ . Then the Markov network H is the unique minimal I-map for P . 
> 定理：
> $P$ 为正分布，$\mathcal H$ 通过为满足式4.3的所有 $\{X, Y\}\quad (Y \in \text{MB}_P(X))$ 引入边得到，则 $\mathcal H$ 是 $P$ 的唯一极小 I-map

The proof is left as an exercise (exercise 4.11). 

Both of the techniques for constructing a minimal I-map make the assumption that the distribution $P$ is positive. As we have shown, for nonpositive distributions, neither the pairwise independencies nor the local independencies imply the global one. Hence, for a nonpositive distribution $P$ , construc g a graph $\mathcal{H}$ su that $P$ satisﬁes the pairwise assumptions for $\mathcal{H}$ does not guarantee that H is an I-map for P . Indeed, we can easily demonstrate that both of these constructions break down for nonpositive distributions. 
> 定理4.5和4.6都基于前提条件：$P$ 为正分布
> 对于非负分布，成对独立性和局部独立性都不包含全局独立性，因此构造使得 $P$ 满足 $\mathcal H$ 编码的成对独立性和局部独立性的 Markov 网络不一定是 $P$ 的 I-map，也就是 $P$ 满足 $\mathcal I_p (\mathcal H)$ 和满足 $\mathcal I_{\mathscr l}(\mathcal H)$ 不一定表明 $P$ 满足 $\mathcal I (\mathcal H)$

Example 4.7
Consider a nonpositive distribution P over four binary variables A; B; C; D that assigns nonzero probability only to cases where all four variables take on exactly the same value; for example, we might have P (a1; b1; c1; d1) = 0:5 and P (a0; b0; c0; d0) = 0:5. The graph H shown in figure 4.7 is one possible output of applying the local independence I-map construction algorithm to P : For example, P j= (A ? C; D j B), and hence fBg is a legal choice for MBP (A). A similar analysis shows that this network satisfies the Markov blanket condition for all nodes. However, it is not an I-map for the distribution.

If we use the pairwise independence I-map construction algorithm for this distribution, the network constructed is the empty network. For example, the algorithm would not place an edge between $A$ and $B$ , because $P\models(A\bot B\mid C,D)$ ⊥ | . Exactly the same a alysis sh s that no edges will be placed into the graph. However, the resulting network is not an I-map for P . 
> Example 4.7同样用了确定性依赖的 trick，分布 $P$ 中存在确定性依赖，使得通过局部独立性算法和成对独立性算法构建出来的 Markov 网络 $\mathcal H$ 都会编码额外的独立性，使得 $\mathcal H$ 不是 $P$ 的 I-map

Both these examples show that deterministic relations between variables can lead to failure in the construction based on local and pairwise independence. Suppose that $A$ and $B$ are two variables that are identical to each other and that both $C$ and $D$ are variables that correlated to both $A$ and $B$ so that $(C\;\bot\;D\;|\;A,B)$ holds. Since $A$ is identical to $B$ , we have that both $(A,D\perp C\mid B)$ and $(B,D\perp C\mid A)$ hold. In other words, it su ces to observe ne of these two variables to capture the relevant information both have about C and separate C from $D$ . In this case the Markov blanket of $C$ is not uniquely deﬁned. This ambiguity leads to the failure of both local and pairwise constructions. Clearly, identical variables are only one way of getting such ambiguities in local independencies. Once we allow nonpositive distribution, other distributions can have similar problems. 
> 变量之间的确定性关系会使得通过分布中的局部或者成对独立性构造 I-map 是失败的
> 例如，假设 $A, B$ 确定性相等，$C, D$ 与它们都相关，满足 $(C\perp D \mid A, B)$，因为 $A = B$，因此有 $(A, D \perp C \mid B)$ 和 $(B, C\perp C \mid A)$，此时，只需观测到 $A, B$ 中的一个，就足以分离 $C, D$，此时 $C$ 的 Markov 毯就不是唯一的，这会导致通过局部独立性构建 I-map 失败

Having deﬁned the notion of a minimal I-map for a distribution $P$ , we can now ask to what extent it represents the independencies in $P$ . More formally, we can ask whether every distribution has a perfect map. Clearly, the answer is no, even for positive distributions: 
> 即便正分布也不一定有完备 I-map

Consider a distribution arising from a three-node Bayesian network with a $\nu$ -structure, for example, the distribution induced in the Student example over the nodes Intelligence, Difculty, and Grade (ﬁgure 3.3). In the Markov network for this distribution, we must clearly have an edge between $I$ and $G$ and between $D$ and $G$ . Can we omit the edge between $I$ and $D$ ? No, because we do not have that $(I\perp D\mid G)$ holds for the distribut on; rather, we ve the opposite: $I$ and $D$ are dependent given G . Therefore, the only minimal I-map for this P is the fully connected graph, which does not capture the marginal independence $(I\perp D)$ that holds in $P$ . 
> 考虑 v-structure $I\rightarrow G \leftarrow D$，将其建模为 Markov 网络时，我们需要添加边 $I-G$ 和 $G-D$，但也要添加边 $I-D$，因为 $(I\perp D\mid G)$ 在分布中不成立，因此 $I-D$ 不是成对独立的，反而是在给定 $G$ 之后，$I, D$ 相互依赖
> 因此，$P$ 的唯一极小 I-map 就是完全图，而完全图没有编码 $P$ 中的边际独立性 $(I\perp D)$

This example provides another counterexample to the strong version of completeness men- tioned earlier. The only distributions for which separation is a sound and complete criterion for determining conditional independence are those for which $\mathcal{H}$ is a perfect map. 
> 只有存在 perfect map 的分布，其 $\mathcal H$ 的分离准则才是可靠且完备的

## 4.4 Parameterization Revisited 
Now that we understand the semantics and independence properties of Markov networks, we revisit some alternative representations for the parameterization of a Markov network. 
### 4.4.1 Finer-Grained Parameterization 
#### 4.4.1.1 Factor Graphs 
A Markov network structure does not generally reveal all of the structure in a Gibbs parameterization. In particular, one cannot tell from the graph structure whether the factors in the parameterization involve maximal cliques or subsets thereof. Consider, for example, a Gibbs distribution $P$ over a fully connected pairwise Markov network; that is, $P$ is parameterized by a factor for each pair of variables $X,Y\in{\mathcal{X}}$ . The clique potential parameterization would utilize a factor whose scope is the entire graph, and which therefore uses an exponential number of parameters. On the other hand, as we discussed in section 4.2.1, the number of parameters in the pairwise parameterization is quadratic in the number of variables. Note that the com- plete Markov network is not redundant in terms of conditional independencies $\mathrm{~-~}P$ does not factorize over any smaller network. Thus, although the ﬁner-grained structure does not imply additional independencies in the distribution (see exercise 4.6), it is still very signiﬁcant. 
>马尔可夫网络结构一般并不能完全揭示 Gibbs 参数化中的所有结构
>特别地，从图结构中我们无法判断参数化中的因子是否涉及最大团或是这些最大团的子集。
>例如，考虑一个完全连通的成对马尔可夫网络上的 Gibbs 分布 $P$；也就是说，$P$ 由每一对变量 $X, Y\in \mathcal X$ 的因子参数化。最大团势参数化将利用一个作用在整个图上的因子，因此需要用到指数数量的参数。另一方面，成对参数化的参数数量与变量的数量呈二次关系。（也就是完全连通的 Markov 网络有两种参数化方式，这在 Markov 图中不能直接看出来）
>值得注意的是，完全连通的马尔可夫网络在条件独立性方面并不是冗余的 —— $P$ 不会在任何更小的网络上分解。因此，尽管更精细的结构并不会在分布中暗示额外的独立性（参见练习题4.6），但它仍然非常重要。

An alternative representation that makes this structure explicit is a factor graph . A factor graph is a graph containing two types of nodes: one type corresponds, as usual, to random variables; the other corresponds to factors over the variables. Formally: 
> 因子图包含两类节点：一类节点对应于随机变量，另一类节点对应于随机变量上的因子

**Deﬁnition 4.13** factor graph 
A factor graph $\mathcal{F}$ is an undirected graph containing two types of nodes: variable nodes (denoted as ovals) and factor nodes (denoted as squares). The graph only contains edges between variable nodes and factor nodes. A factor graph $\mathcal{F}$ is parameterized by a set of factors, where each factor node $V_{\phi}$ is associated with precisely one factor $\phi.$ , whose scope is the set of variables that are neighbors of $V_{\phi}$ in the graph. A distribution $P$ factorizes over $\mathcal{F}$ if it can be represented as a set of factors of this form. 
> 定义：
> 因子图为无向图 $\mathcal F$，包含两类节点：变量节点（椭圆形）、因子节点（方形）
> 图仅包含变量节点和因子节点之间的边
> 因子图 $\mathcal F$ 由一组因子参数化，每个因子节点 $V_\phi$ 正好和一个因子 $\phi$ 关联，其作用域就是和 $V_\phi$ 相邻的变量节点
> 分解于 $\mathcal f$ 的分布 $P$ 可以被图中的一组因子表示

Factor graphs make explicit the structure of the factors in the network. For example, in a fully connected pairwise Markov network, the factor graph would contain a factor node for each of the $\textstyle{\binom{n}{2}}$ pairs of nodes; the factor node for a pair $X_{i},X_{j}$ would be connected to $X_{i}$ and $X_{j}$ ; by contrast, a factor graph for a distribution with a single factor over $X_{1},\dots,X_{n}$ would have a single factor node connected to all of $X_{1},\ldots,X_{n}$ (see ﬁgure 4.8). Thus, although the Markov networks for these two distributions are identical, their factor graphs make explicit the diference in their factorization. 
> 因子图显式在网络中表示了因子的结构
> 两个分布的 Markov 网络相同时，其因子图可以不同，可以根据其因子图确认其因子分解的形式

#### 4.4.1.2 Log-Linear Models 
Although factor graphs make certain types of structure more explicit, they still encode factors as complete tables over the scope of the factor. As in Bayesian networks, factors can also exhibit a type of context-speciﬁc structure — patterns that involve particular values of the variables. These patterns are often more easily seen in terms of an alternative parameteriz ation of the factors that converts them into log-space. 
> 因子图依旧用表格的形式表示因子
> 在贝叶斯网络中，我们讨论了 CPD 的特定上下文中的结构，也就是在特定变量有特定赋值时的情况
> 在因子的另一种参数化形式：将其转化到对数空间时，这类模式也非常常见

More precisely, we can rewrite a factor $\phi(D)$ as 

$$\phi({\cal D})=\exp(-\epsilon({\cal D})),$$

where $\epsilon(D)\,=\,-\ln\phi(D)$ is often called an energy function . 
> 将因子 $\phi (\pmb D)$ 重写为 $\phi (\pmb D) = \exp (-\epsilon (\pmb D))$
> 其中 $\epsilon (\pmb D) = -\ln \phi (\pmb D)$ 称为能量函数（注意这要求 $\phi(\pmb D) \ne 0$）

The use of the word “energy” derives from statistical physics, where the probability of a physical state (for example, a conﬁguration of a set of electrons), depends inversely on its energy. 
> 在统计物理学中，一个物理状态（例如一组电子的一个配置）的概率与它的能量成反比（能量越低概率越高）
> 因此也就和能量函数的反成正比

In this logarithmic representation, we have 

$$
P(X_{1},.\,.\,,X_{n})\propto\exp\left[-\sum_{i=1}^{m}\epsilon_{i}(\pmb D_{i})\right].
$$ 
The logarithmic representation ensures that the probability distribution is positive. Moreover, the logarithmic parameters can take any value along the real line. 
> 对数表示：
> $P (X_1, \dots, X_n)\propto \prod_{i=1}^m \phi (\pmb D_i) = \prod_{i=1}^m \exp (-\epsilon_i (\pmb D_i)) = \exp\left[-\epsilon_i(\pmb D_i)\right]$
> 对数表示下，概率一定是正数

Any Markov network parameterized using positive factors can be converted to a logarithmic representation. 
> 任意使用正因子参数化的 Markov 网络都可以被转化为对数表示

Fig4.9:

$$
\epsilon_{1}(A,B)\qquad\qquad\epsilon_{2}(B,C)\qquad\qquad\epsilon_{3}(C,D)\qquad\qquad\epsilon_{4}(D,A)
$$ 

$$
\begin{array}{r}{\left|\begin{array}{l l l}{0}&{b^{0}}&{-3.4}\\ {0}&{b^{1}}&{-1.61}\\ {1}&{b^{0}}&{0}\\ {1}&{b^{1}}&{-2.3}\end{array}\right|\left|\begin{array}{c c c c}{b^{0}}&{c^{0}}&{-4.61}\\ {b^{0}}&{c^{1}}&{0}\\ {b^{1}}&{c^{0}}&{0}\\ {b^{1}}&{c^{1}}&{-4.61}\end{array}\right|\left|\begin{array}{c c c c}{c^{0}}&{d^{0}}&{0}\\ {c^{0}}&{d^{1}}&{-4.61}\\ {c^{1}}&{d^{0}}&{-4.61}\\ {c^{1}}&{d^{1}}&{0}\end{array}\right|\left|\begin{array}{c c c c}{d^{0}}&{a^{0}}&{-4.61}\\ {d^{0}}&{a^{1}}&{0}\\ {d^{1}}&{a^{0}}&{0}\\ {d^{1}}&{a^{1}}&{-4.61}\end{array}\right|}\end{array}
$$ 

Example 4.9 
Figure 4.9 shows the logarithmic representation of the clique potential parameters in ﬁgure 4.1. We can see that the “1” entries in the clique potentials translate into $"0"$ entries in the energy function. 

This representation makes certain types of structure in the potentials more apparent. For example, we can see that both $\epsilon_{2}(B,C)$ and $\epsilon_{4}(D,A)$ are constant multiples of an energy function that ascribes 1 to instantiations where the values of the two variables agree, and 0 to the instantiations where they do not. 

We can provide a general framework for capturing such structure using the following notion: 

**Deﬁnition 4.14** feature 
Let D be a subset of variables. We define a feature f(D) to be a function from Val(D) to R.
> 定义：
> $\pmb D$ 为变量子集，定义特征 $f (\pmb D)$ 为从 $Val (\pmb D)$ 到 $R$ 的函数 

A feature is simply a factor without the nonnegativity requirement. One type of feature of particular interest is the indicator feature that takes on value 1 for some values $\pmb{y}\in V a l(\pmb{D})$ and 0 otherwise. 
> 特征简单来说就是没有非负要求的因子
> 一类常用的特征是指示器特征，它对于特定的 $\pmb y \in Val (\pmb D)$ 取值为 1，否则取值为 0

Features provide us with an easy mechanism for specifying certain types of interactions more compactly. 

Example 4.10 
Consider a situation where $A_{1}$ and $A_{2}$ each have $\ell$ values $a^{1},\ldots,a^{\ell}$ . Assume that our distribution is such that we prefer situations where $A_{1}$ and $A_{2}$ take on the same value, but otherwise have no preference. Thus, our energy function might have the following form: 

$$
\epsilon(A_{1},A_{2})=\left\{\begin{array}{l l}{{-3\qquad\qquad A_{1}=A_{2}}}\\ {{0\qquad\qquad o t h e r w i s e}}\end{array}\right.
$$ 
Represented as a full factor, this clique potential requires $\ell^{2}$ values. However, it can also be represented as a log-linear function in terms of $a$ feature $f(A_{1},A_{2})$ that is an indicator function for the event $A_{1}=A_{2}$ . The energy function is then simply a constant multiple $-3$ of this feature. 

Thus, we can provide a more general deﬁnition for our notion of log-linear models: 

**Deﬁnition 4.15** log-linear model 
A distribution $P$ is a log-linear model over a Markov network $\mathcal{H}$ if it is associated with: 
- a set of features $\mathcal{F}=\{f_{1}(D_{1}),.\,.\,.\,,f_{k}(D_{k})\}$ , where each $D_{i}$ is a complete subgraph in $\mathcal{H}$ , 
- a set of weights $w_{1},\dots,w_{k}$ , 

such that 

$$
P(X_{1},\dots,X_{n})=\frac{1}{Z}\exp\left[-\sum_{i=1}^{k}w_{i}f_{i}(D_{i})\right].
$$ 
Note that we can have several features over the same scope, so that we can, in fact, represent a standard set of table potentials. (See exercise 4.13.) 
> 定义：
> Markov 网络 $\mathcal H$ 上的对数线性模型是一个分布 $P$，它满足：
> - 和一组特征 $\mathcal F = \{f_1 (\pmb D_1), \dots, f_k (\pmb D_k)\}$ 关联，其中每个 $\pmb D_i$ 都是 $\mathcal H$ 中的完全子图
> - 和一组权重 $w_1,\dots, w_k$ 关联
> 使得 $P (X_1, \dots, X_n) = \frac 1 Z \exp \left[ -\sum_{i=1}^k w_i f_i(\pmb D_i)\right]$

> $P (X_1, \dots, X_n) \propto \exp\left[-\sum_{i=1}^k w_if_i(\pmb D_i)\right] = \prod_{i=1}^k \exp \left[-w_if_i(\pmb D_i)\right]$
> 因此可以认为对数线性模型就对应于把因子设定为了 $\phi_i (\pmb D_i) = \exp \left[-w_i f_i (\pmb D_i)\right]$ 的形式，也就是将能量函数的形式设定为了 $w_i f_i (\pmb D_i)$

The log-linear model provides a much more compact representation for many distributions, especially in situations where variables have large domains such as text (such as box 4.E). 

#### 4.4.1.3 Discussion 
We now have three representations of the parameterization of a Markov network. The Markov network denotes a product over potentials on cliques. A factor graph denotes a product of factors. And a set of features denotes a product over feature weights. Clearly, each representation is ﬁner-grained than the previous one and as rich. A factor graph can describe the Gibbs distribution, and a set of features can describe all the entries in each of the factors of a factor graph. 
> 目前我们对 Markov 网络的参数化有了三种表示：Markov 网络表示多个团势能函数的乘积、因子图表示多个因子的乘积、一组特征表示特征的加权乘积
> 每个表示都比上一个更精细：因子图可以表示 Gibbs 分布、一组特征可以描述因子图中因子的所有项

Depending on the question of interest, diferent representations may be more appropriate. For example, a Markov network provides the right level of abstraction for discussing independence queries: The ﬁner-grained representations of factor graphs or log-linear models do not change the independence assertions made by the model. On the other hand, as we will see in later chapters, factor graphs are useful when we discuss inference, and features are useful when we discuss parameterizations, both for hand-coded models and for learning. 

**Box 4.C — Concept: Ising Models and Boltzmann Machines.** 
One of the earliest types of Markov network models is the Ising model , which ﬁrst arose in statistical physics as a model for the energy of a physical system involving a system of interacting atoms. In these systems, each atom is associ- ated with a binary-valued random variable $X_{i}\in\{+1,-1\}$ , whose value deﬁnes the direction of the atom’s spin. The energy function associated with the edges is deﬁned by a particularly simple parametric form: 
> 最早的一类 Markov 网络模型为 lsing 模型，出现在统计物理学，建模包含了交互的原子的物理系统的能量
> 系统中，每个原子和一个二值变量 $X_i \in \{+1, -1\}$，其值定义了原子的自旋方向
> 和其相关的能量函数和边关联，其参数化形式很简单：

$$
\epsilon_{i,j}(x_{i},x_{j})=w_{i,j}x_{i}x_{j}
$$ 
This energy is symmetric in $X_{i},X_{j}.$ ; it makes a contribution of $w_{i,j}$ to the energy function when $X_{i}=X_{j}$ (so both atoms have the same spin) and a contribution of $-w_{i,j}$ otherwise. 
> 能量在 $X_i, X_j$ 之间是对称的，如果两个原子的自旋方向相同，就贡献 $w_{ij}$ 的能量，否则就贡献 $-w_{ij}$ 的能量

Our model also contains a set of parameters $u_{i}$ that encode individual node potentials; these bias individual variables to have one spin or another. 
> 模型还包含一组参数 $u_i$ 来编码独立的节点势能，使得对应原子向某个特定的自旋方向偏置

As usual, the energy function deﬁnes the following distribution: 

$$
P(\xi)=\frac{1}{Z}\exp\left(-\sum_{i<j}w_{i,j}x_{i}x_{j}-\sum_{i}u_{i}x_{i}\right).\tag{4.4}
$$ 
As we can see, when $w_{i,j}\,>\,0$ the model prefers to align the spins of the two atoms; in this case, the interaction is called ferromagnetic . When $w_{i,j}<0$ the interaction is called anti ferromagnetic . When $w_{i,j}=0$ the atoms are non-interacting . 
> 能量函数定义的分布如上，仍然遵循能量越小，概率越大的形式
> 显然，模型中，$w_{ij} > 0$ 促使原子 $i, j$ 的自旋方向相同，$w_{ij} < 0$ 促使原子 $i, j$ 的自旋方向相反，$w_{ij} = 0$ 表示两个原子并未交互，二者的方向关系无所谓

Much work has gone into studying particular types of Ising models, attempting to answer a variety of questions, usually as the number of atoms goes to inﬁnity. For example, we might ask the probability of a conﬁguration in which a majority of the spins are $+1$ or $-1$ , versus the probability of more mixed conﬁgurations. The answer to this question depends heavily on the strength of the interaction between the variables; so, we can consider adapting this strength (by multiplying all weights by a temperature parameter ) and asking whether this change causes a phase transition in the probability of skewed versus mixed conﬁgurations. These questions, and many others, have been investigated extensively by physicists, and the answers are known (in some cases even analytically) for several cases. 
> 调节原子之间交互的强度：为权重乘上温度参数

Related to the Ising model is the Boltzmann distribution ; here, the variables are usually taken to have values $\{0,1\}$ , but still with the energy form of equation (4.4). Here, we get a nonzero contribution to the model from an edge $(X_{i},X_{j})$ only when $X_{i}=X_{j}=1,$ ; however, the resulting energy can still be reformulated in terms of an Ising model (exercise 4.12). 
> Blotzmann 分布中，变量一般取值 $\{0, 1\}$，但能量形式和式4.4仍相同
> 此时，只有在 $X_i = X_j = 1$ 时，才对系统能量有贡献，但该形式仍然可以重构为 Ising 模型

The popularity of the Boltzmann machine was primarily driven by its similarity to an activation model for neurons. To understand the relationship, we note that the probability distribution over each variable $X_{i}$ given an assignment to is neighbors is $\text{sigmoid}(z)$ where 

$$
z=-(\sum_{j}w_{i,j}x_{j})-w_{i}.
$$ 
This function is a sigmoid of a weighted combination of $X_{i}$ ’s neighbors, weighted by the strength and direction of the connections between them. This is the simplest but also most popular mathematical approximation of the function employed by a neuron in the brain. Thus, if we imagine a process by which the network continuously adapts its assignment by resampling the value of each variable as a stochastic function of its neighbors, then the “activation” probability of each variable resembles a neuron’s activity. This model is a very simple variant of a stochastic, recurrent neural network. 
> Boltzmann 机和神经元的激活模型很相似，考虑 Boltzmann 机中每个变量 $X_i$ 在给定它的邻居的赋值的情况下的分布，它实际为一个 sigmoid 函数 $\text{sigmoid}(z)$，其中 $z = -(\sum_j w_{ij}x_j)- w_i$ ( $X_i = 1$ 时相关的能量)，$z$ 实际上就是对和 $X_i$ 相连的邻居的加权求和再加上偏置
> ($X_i = 0$ 时，$P (X_i \mid \text{Pa}_{X_i}) = \frac 1 Z\exp (0) = 1$
> $X_i = 1$ 时，$P (X_i \mid \text{Pa}_{X_i}) = \frac 1 Z\exp \left(-\sum_jw_{ij}x_j -w_i\right)= \exp(z)$
> 显然，$Z = 1 + \exp (z)$，故 $P (X_i = 1\mid \text{Pa}_{X_i}) = \frac {\exp (z)}{1 + \exp (z)} = \text{sigmoid}(z)$)
> 该函数是对神经网络的最常用的近似数学形式

Box 4.D — Concept: Metric MRFs. 
One important class of MRFs comprises those used for la- beling . Here, we ve a graph of nodes $X_{1},\dots,X_{n}$ related by a set of edges $\mathcal{E}$ , and we wish to assign to each $X_{i}$ a label in the space $\mathcal{V}=\{v_{1},.\,.\,.\,,v_{K}\}$ . Each node, taken in isolation, has its preferences among the possible labels. However, we also want to impose a soft “smoothness” constraint over the graph, in that neighboring nodes should take “similar” values. 

We encode the individual node preferences as node potentials in a pairwise MRF and the smooth- ness preferences as edge potentials. For reasons that will become clear, it is traditional to encode these models in negative log-space, using energy functions. As our objective in these models is inevitably the MAP objective, we can also ignore the partition function, and simply consider the energy function: 

$$
E(x_{1},.\,.\,,x_{n})=\sum_{i}\epsilon_{i}(x_{i})+\sum_{(i,j)\in{\mathcal{E}}}\epsilon_{i,j}(x_{i},x_{j}).
$$ 
Our goal is then to minimize the energy: 

$$
\arg\operatorname*{min}_{x_{1},\ldots,x_{n}}E(x_{1},\ldots,x_{n}).
$$ 
We now need to provide a formal deﬁnition for the intuition of “smoothness” described earlier. There are many diferent types of conditions that we can impose; diferent conditions allow diferent methods to be applied. 

One of the simplest in this class of models is a slight variant of the Ising model , where we have that, for any $i,j$ : 

$$
\epsilon_{i,j}(x_{i},x_{j})=\left\{\begin{array}{l l}{{0}}&{{\qquad\qquad x_{i}=x_{j}}}\\ {{\lambda_{i,j}}}&{{\qquad\qquad x_{i}\neq x_{j},}}\end{array}\right.
$$ 
for $\lambda_{i,j}\geq0$ In this model, we obtain the lowest possible rwise energy (0) when two neighboring nodes $X_{i},X_{j}$ take the same value, and a higher energy $\lambda_{i,j}$ when they do not. 

This simple model has been generalized in many ways. The Potts model extends it to the setting of more than two labels. An even broader class contains models where we have a distance function on the labels, and where we prefer neighboring nodes to have labels that are a smaller distance apart. More precisely, a function $\mu\ :\ \mathcal{V}\times\mathcal{V}\mapsto[0,\infty)$ is $^a$ metric if it satisﬁes: 

- Reﬂexivity: $\mu(v_{k},v_{l})=0$ if and only if $k=l$ ;

- Symmetry: $\mu(v_{k},v_{l})=\mu(v_{l},v_{k}).$ ; 

$$
\begin{array}{c c c}{{a^{0}}}&{{b^{0}}}&{{-4.4}}\\ {{a^{0}}}&{{b^{1}}}&{{-1.61}}\\ {{a^{1}}}&{{b^{0}}}&{{-1}}\\ {{a^{1}}}&{{b^{1}}}&{{-2.3}}\end{array}\left|\begin{array}{c c c}{{b^{0}}}&{{c^{0}}}&{{-3.61}}\\ {{b^{0}}}&{{c^{1}}}&{{+1}}\\ {{b^{1}}}&{{c^{0}}}&{{0}}\\ {{b^{1}}}&{{c^{1}}}&{{-4.61}}\end{array}\right|
$$ 
We say that $\mu$ is a semimetric if it satisﬁes reﬂexivity and symmetry. We can now deﬁne a metric MRF (or a semimetric MRF ) by deﬁning $\epsilon_{i,j}(v_{k},v_{l})=\mu(v_{k},v_{l})$ for all $i,j$ , where $\mu$ is a metric (semimetric). We note that, as deﬁned, this model assumes that the distance metric used is the same for all pairs of variables. This assumption is made because it simpliﬁes notation, it often holds in practice, and it reduces the number of parameters that must be acquired. It is not required for the inference algorithms that we present in later chapters. Metric interactions arise in many applications, and play a particularly important role in computer vision (see box 4.B and box 13.B). For example, one common metric used is some form of truncated $p$ -norm (usually $p=1$ or $p=2.$ ): 

$$
\epsilon(x_{i},x_{j})=\operatorname*{min}(c\|x_{i}-x_{j}\|_{p},{\mathrm{dist}}_{\mathrm{max}}).
$$ 
### 4.4.2 Over parameterization 
Even if we use ﬁner-grained factors, and in some cases, even features, the Markov network parameterization is generally over parameterized. That is, for any given distribution, there are multiple choices of parameters to describe it in the model. Most obviously, if our graph is a single clique over $n$ binary variables $X_{1},\dots,X_{n}$ , then the network is associated with a clique potential that has $2^{n}$ parameters, whereas the joint distribution only has $2^{n}-1$ independent parameters. 
> 对于任意给定的分布，存在多个参数选择来在模型中描述这个分布

A more subtle point arises in the context of a nontrivial clique structure. Consider a pair of cliques $\{A,B\}$ and $\{B,C\}$ . The energy function $\epsilon_{1}(A,B)$ (or its corresponding clique potential) contains information not only about the interaction between $A$ and $B$ , but also about the distribution of the individual variables $A$ and $B$ . Similarly, $\epsilon_{2}(B,C)$ gives us information about the individual variables $B$ and $C$ . The information about $B$ can be placed in either of the two cliques, or its contribution can be split between them in arbitrary ways, resulting in many diferent ways of specifying the same distribution. 
> 例如考虑两对团 $\{A, B\}$ 和 $\{B, C\}$，能量函数 $\epsilon_1 (A, B)$ 包含了二者的交互，同时也包含了 $A, B$ 各自独立的分布，$\epsilon_2 (B, C)$ 也是如此，故关于 $B$ 的信息可以放在两个团其中之一，或者以任意的方式在二者之间分离，因此指定相同的分布就有了不同的方式

Example 4.11
Consider the energy functions 1(A; B) and 2(B; C) in figure 4.9. The pair of energy functions shown in figure 4.10 result in an equivalent distribution: Here, we have simply subtracted 1 from 1(A; B) and added 1 to 2(B; C) for all instantiations where B = b0. It is straightforward to check that this results in an identical distribution to that of ﬁgure 4.9. In inst ere $B\neq b^{0}$ the energy function returns exactly the same value as before. In cases where $B\,=\,b^{0}$ , the actual values of the energy functions have changed. However, because the sum of the energy functions on each instance is identical to the original sum, the probability of the instance will not change. 

Intuitively, the standard Markov network representation gives us too many places to account for the inﬂuence of variables in shared cliques. Thus, the same distribution can be represented as a Markov network (of a given structure) in inﬁnitely many ways. It is often useful to pick one of this inﬁnite set as our chosen parameterization for the distribution. 
> 直观上看，标准的 Markov 网络表示有近乎无限的方式表示在团中变量的影响，我们需要选择以一种对分布进行参数化

#### 4.4.2.1 Canonical Parameterization 
The canonical parameterization provides one very natural approach to avoiding this ambiguity in the parameterization of a Gibbs distribution $P$ . This canonical parameterization requires that the distribution $P$ be positive. It is most convenient to describe this parameterization using energy functions rather then clique potentials. For this reason, it is also useful to consider a log- transform of $P$ : For any assignment $\xi$ to $\mathcal{X}$ , we use $\ell(\xi)$ to denote $\ln P(\xi)$ . This transformation is well deﬁned because of our positivity assumption. 
> 规范参数化方式要求 $P$ 为正分布
> 一般使用能量函数而不是团势能来描述该参数化，故一般考虑 $P$ 的对数形式：
> 对于任意对 $\mathcal X$ 的赋值 $\xi$，我们使用 $\mathscr l (\xi)$ 来表示 $\ln P (\xi)$ (因为考虑对数形式，因此要求是正分布)

The canonical parameterization of a Gibbs distribution over $\mathcal{H}$ is deﬁned via set of energy functions over all cliques. Thus, for example, the Markov network in ﬁgure 4.4b would have energy functions for the two cliques $\{A,B,D\}$ and $\{B,C,D\}$ , energy functions for all possible pairs of variables except the pair $\{A,C\}$ (a total of ﬁve pairs), energy functions for all four singleton sets, and a constant energy function for the empty clique. 
> Gibbs 分布在 $\mathcal H$ 上的规范参数化通过在所有团上的一组能量函数定义（所有的团包括了：所有的完全子图、每个节点自己、以及为空团也定义一个常数能量函数）

At ﬁrst glance, it appears that we have only increased the number of parameters in the speciﬁcation. However, as we will see, this approach uniquely associates the interaction parameters for a subset of variables with that subset, avoiding the ambiguity described earlier. As a consequence, many of the parameters in this canonical parameterization are often zero. 
> 该方法唯一地将一个变量子集的交互参数和该变量子集关联
> 该方法中，许多参数常常是0

The canonical parameterization is deﬁned relative to a particular ﬁxed assignment $\xi^{*}\ =$ $\left(x_{1}^{*},\cdot\cdot\cdot,x_{n}^{*}\right)$ to the network variables $\mathcal{X}$ . is assignment can chosen arbitraly. For any subset of variables Z , and any assignment x to some subset of X that contain $Z$ , we deﬁne the assignment $\pmb{x}_{Z}$ to be $_{x\langle Z\rangle}$ , that is, the assignment in $_{_{x}}$ the variables in Z . Conversely, we deﬁne $\xi_{-Z}^{*}$ to be $\xi^{*}\langle\mathcal{X}-Z\rangle$ , that is, the assignment in $\xi^{*}$ to the variables outside Z . − can now construct an assignment $(x_{Z},\xi_{-Z}^{\ast})$ that keeps the assignments to the variables in Z − as speciﬁed in $_{_{x}}$ , and augments it using the default values in $\xi^{*}$ . 
> 规范参数化相对于对于网络变量 $\mathcal X$ 的特定赋值 $\xi^* = (x_1^*, \cdots, x_n^*)$ 定义，该赋值可以任意选择
> 对于任意的变量子集 $\pmb Z$，以及任意对某个包含 $\pmb Z$ 的 $\mathcal X$ 子集的赋值 $\pmb x$，定义赋值 $\pmb x_{\pmb Z}$ 为 $\pmb x\langle \pmb Z\rangle$，也就是 $\pmb x$ 中对于 $\pmb Z$ 的赋值，还定义 $\xi_{-\pmb Z}^*$ 为 $\xi^*\langle\mathcal X-\pmb Z\rangle$，即 $\xi^*$ 中对于 $\pmb Z$ 以外的变量的赋值
> 由此，我们构造赋值 $(\pmb x_{\pmb Z}, \xi^*_{-\pmb Z})$，即对于 $\pmb Z$ 中的变量，使用 $\pmb x$ 中指定的赋值，对于 $\pmb Z$ 外的变量，保持 $\xi^*$ 中的默认赋值

The canonical energy function for a clique $\pmb D$ is now deﬁned as follows: 

$$
\epsilon_{\pmb D}^{*}(\pmb d)=\sum_{\pmb Z\subseteq \pmb D}(-1)^{|\pmb D-\pmb Z|}\ell(\pmb d_{\pmb Z},\xi_{-\pmb Z}^{*}),\tag{4.8}
$$ 
where the sum is over all subsets of $D$ , including $D$ itself and the empty set $\varnothing$ . Note that all of the terms in the summation have a scope that is contained in D , which in turn is part of a clique, so that these energy functions are legal relative to our Markov network structure. 
> 团 $\pmb D$ 的规范能量函数定义为式4.8，其中的求和 $\sum_{\pmb Z\subseteq \pmb D}$ 是在 $\pmb D$ 的所有子集上求和，包括 $\pmb D$ 自己和空集

This formula performs an inclusion-exclusion computation. For a set $\{A,B,C\}$ , it ﬁrst subtracts out the inﬂuence of all of the pairs: $\{A,B\},\ \{B,C\}$ , and $\{C,A\}$ . However, this process oversubtracts the inﬂuence of the individual variables. Thus, their inﬂuence is added back in, to compensate. More generally, consider any subset of variables $\pmb Z\subseteq \pmb D$ . Intuitively, it make a “contribution” once for every subset $\pmb{U}\supseteq\pmb{Z}$ . Except for $\pmb U = \pmb D$ , the number of times that $\pmb Z$ appears is even — there is an even number of subsets $\pmb U \supseteq \pmb Z$ — and the number of times it appears with a positive sign is equal to the number of times it appears with a negative sign. Thus, we have effectively eliminated the net contribution of the subsets from the canonical energy function.
> 该公式执行的是包含-排斥计算（通过系数 $(-1)^{|\pmb D - \pmb Z|}$）
> 考虑任意变量子集 $\pmb Z \subseteq \pmb D$，直观上，它在每个 $\pmb U \supseteq \pmb Z$ 都对总能量函数做出一次贡献，除去 $\pmb U = \pmb D$，$\pmb Z$ 的超集的数量是偶数，其中一半的时候其系数是-1，另一半的时候其系数是+1，故 $\pmb Z$ 的净影响是在规范参数化中被完全消除的

Let us consider the effect of the canonical transformation on our Misconception network.

Let us choose $(a^{0},b^{0},c^{0},d^{0})$ as our arbitrary assignment on which to base the canonical parameterization. The resulting energy functions are shown in ﬁgure 4.11. For example, the energy value $\epsilon_{1}^{*}(a^{1},b^{1})$ was computed as follows:

$\begin{array}{r l}&{\ell(a^{1},b^{1},c^{0},d^{0})-\ell(a^{1},b^{0},c^{0},d^{0})-\ell(a^{0},b^{1},c^{0},d^{0})+\ell(a^{0},b^{0},c^{0},d^{0})=}\\ &{\phantom{a^{1}}-13.49--11.18--9.58+-3.18=4.09}\end{array}$ 

Note that many of the entries in the energy functions are zero. As discussed earlier, this phenomenon is fairly general, and occurs because we have accounted for the inﬂuence of small subsets of variables separately, leaving the larger factors to deal only with higher-order inﬂuences. We also note that these canonical parameters are not very intuitive, highlighting yet again the diffculties of constructing $^a$ reasonable parameter iz ation of a Markov network by hand. 

This canonical parameterization deﬁnes the same distribution as our original distribution $P$ : 
> 规范参数化定义了和原来的分布 $P$ 同样的分布

**Theorem 4.7**
Let $P$ be a positive Gibbs distribution over $\mathcal{H}$ , and let $\epsilon^{*}(\mathsf{D}_{i})$ for each clique $\mathsf{D}_{i}$ be deﬁned as speciﬁed in equation (4.8). Then 
> 定理：
> $P$ 为 $\mathcal H$ 上的正 Gibbs 分布，$\epsilon^*(\pmb D_i)$ 为每个团的规范能量函数 $\pmb D_i$，其定义如 (4.8)，则 $P$ 可以写为

$$
P(\xi)=\exp\left[\sum_{i}\epsilon_{\pmb{D}i}^{*}(\xi\langle\pmb{D}_{i}\rangle)\right].
$$ 
The proof for the case where $\mathcal{H}$ consists of a single clique is fairly simple, and it is left as an exercise (exercise 4.4). The general case follows from results in the next section. 
> 证明：
> 当 $\mathcal H$ 仅由一个团构成时，有

$$
\begin{align}
P(\xi) &=\exp\left[\epsilon_{\mathcal X}^*(\xi)\right]\\
&=\exp\left[\sum_{\pmb Z\subseteq \mathcal X}(-1)^{|\mathcal X-\pmb Z|}\ell(\xi_{\pmb Z},\xi_{-\pmb Z}^{*})\right]\\
&=\prod_{\pmb Z \subseteq \mathcal X}\exp\left[(-1)^{|\mathcal X-\pmb Z|}\ell(\xi_{\pmb Z},\xi_{-\pmb Z}^{*})\right]\\
&=\prod_{\pmb Z \subseteq \mathcal X}\exp\left[(-1)^{|\mathcal X-\pmb Z|}\ln P(\xi_{\pmb Z},\xi_{-\pmb Z}^{*})\right]\\
&=\prod_{\pmb Z \subseteq \mathcal X}\exp\left[\ln P(\xi_{\pmb Z},\xi_{-\pmb Z}^{*})^{(-1)^{{|\mathcal X - \pmb Z|}}}\right]\\
&=\prod_{\pmb Z \subseteq \mathcal X}\left[ P(\xi_{\pmb Z},\xi_{-\pmb Z}^{*})\right]^{(-1)^{{|\mathcal X - \pmb Z|}}}\\
&=P(\xi_{\mathcal X}, \xi^*_{-\mathcal X})\\
&=P(\xi)
\end{align}
$$

The canonical parameterization gives us the tools to prove the Hammersley-Clifford theorem, which we restate for convenience.

**Theorem 4.8**
Let $P$ be a positive distribution over $\mathcal{X}$ , a $\mathcal{H}$ a Markov network graph over $\mathcal{X}$ . If $\mathcal{H}$ is an $I\cdot$ -map for P , then P is a Gibbs distribution over . 
> 定理：
> $P$ 为 $\mathcal X$ 上的正分布，$\mathcal H$ 为 $\mathcal X$ 上的 Markov 网络，如果 $\mathcal H$ 是 $P$ 的 I-map，则 $P$ 是 $\mathcal X$ 上的 Gibbs 分布

Proof To prove this result, we need to show the existence of a Gibbs parameter iz ation for any distribution $P$ that satisﬁes the Markov assumptions associated with $\mathcal{H}$ . The proof is constructive, and simply uses the canonical parameterization shown earlier in this section. Given $P$ , we deﬁne an energy function for all subsets $\pmb D$ of nodes in the graph, regardless of whether they are cliques in the graph. This energy function is deﬁned exactly as in equation (4.8), relative to some speciﬁc ﬁxed assignment $\xi^{*}$ used to deﬁne the canonical parameterization. The distribution deﬁned using this set of energy functions is $P$ : the argument is identical to the proof of theorem 4.7, for the case where the graph consists of a single clique (see exercise 4.4). 
> 证明：
> 我们需要证明对于任意满足和 $\mathcal H$ 相关的 Markov 假设的分布 $P$ 都存在 Gibbs 参数化，故证明是构造性的，使用了规范参数化
> 给定 $P$，我们为所有图中的节点子集 $\pmb D$ 定义能量函数，无论该节点子集是否构成团，能量函数的定义遵循 (4.8)，也就是相对于某个特定的固定赋值 $\xi^*$ 来定义规范参数化
> 使用这一组能量函数定义的分布就是 $P$，该结论来源于定理4.7

It remains only to show that the resulting distribution is a Gibbs distribution over $\mathcal{H}$ . To show that, we need to show that the factors $\epsilon^{*}(\pmb D)$ are identically 0 whenever $\pmb D$ is not a clique in the graph, that is, whenever the nodes in $\pmb D$ do not form a fully connected subgraph.
> 接下来，我们需要证明这一组能量函数定义的分布也是 $\mathcal H$ 上的 Gibbs 分布
> 为此，我们需要证明因子 $\epsilon^*(\pmb D)$ 在 $\pmb D$ 不是图中的团的时候等于0

Assume that we have $X,Y\in \pmb D$ such that there is no edge between $X$ and $Y$ . 
> 假设 $\pmb D$ 中存在两个节点 $X, Y$ 之间没有边

For this proof, it helps to introduce the notation

$$
\sigma_{\pmb Z}[\pmb x]=(\pmb x_{\pmb Z},\xi_{-\pmb Z}^{\ast}).
$$ 
Plugging this notation into equation (4.8), we have that: 

$$
\epsilon_{\pmb D}^{*}(\pmb d)=\sum_{\pmb Z\subseteq \pmb D}(-1)^{|\pmb D-\pmb Z|}\ell(\sigma_{\pmb Z}[\pmb d]).
$$ 
> 引入记号 $\sigma_{\pmb Z}[\pmb d] = (\pmb d_{\pmb Z}, \xi^*_{-\pmb Z})$ ，以方便表示

We now rearrange the sum over subsets $\pmb Z$ into a sum over groups of subsets.  $\pmb W\subseteq \pmb D-\{X,Y\}$ ; then $\pmb W,\,\pmb W\cup\{X\},\,\pmb W\cup\{Y\}$ and $\pmb W\cup\{X,Y\}$ are all subsets of $\pmb Z$ . 
> 令 $\pmb W \subseteq \pmb D - \{X, Y\}$，然后将 $\pmb D$ 的全部子集 $\pmb Z$ 表示为四个组：$\pmb W,\pmb W \cup \{X\}, \pmb W\cup \{Y\}, \pmb W\cup \{X, Y\}$

Hence, we can rewrite the summation over subsets of $\pmb D$ as a summation over subsets of $\pmb D-\{X,Y\}$ : 
> 将和式重写为：

$$
\begin{align}
\epsilon_{\pmb D}^*(\pmb d) &= \sum_{\pmb W \subseteq \pmb D- \{X, Y\}}(-1)^{|\pmb D - \{X, Y\} - \pmb W|}\tag{4.9}\\
&\quad (\ell(\sigma_{\pmb W}[\pmb d]) - \ell(\sigma_{\pmb W \cup \{X\}}[\pmb d]) - \ell(\sigma_{\pmb W\cup \{Y\}}[\pmb d]) + \ell(\sigma_{\pmb W\cup \{X, Y\}}[\pmb d]))
\end{align}
$$

Now consider a speciﬁc subset $W$ in this sum, and let $u^{*}$ be $\xi^{*}\langle\mathcal{X}-D\rangle$ — the assignment to $\mathcal{X}-D$ in $\xi$ . We now have that: 
> 考虑某个特定的子集 $\pmb W$，令 $\pmb u^*$ 表示 $\xi^*\langle\mathcal X - \pmb D\rangle$，即 $\xi$ 中对于 $\mathcal X - \pmb D$ 的赋值，我们有：

$$
\begin{align}
\ell(\sigma_{\pmb W \cup \{X, Y\}}[\pmb d]) - \ell(\sigma_{\pmb W \cup \{X\}}[\pmb d]) &= \ln \frac {P(x, y, \pmb w, \pmb u^*)}{P(x, y^*,\pmb w, \pmb u^*)}\\
&=\ln\frac {P(y\mid x, \pmb w, \pmb u^*)P(x, \pmb w, \pmb u^*)}{P(y^*\mid x, \pmb w, \pmb u^*)P(x, \pmb w, \pmb u^*)}\\
&=\ln\frac {P(y\mid x^*, \pmb w, \pmb u^*)P(x, \pmb w, \pmb u^*)}{P(y^*\mid x^*, \pmb w, \pmb u^*)P(x, \pmb w, \pmb u^*)}\\
&=\ln\frac {P(y\mid x^*, \pmb w, \pmb u^*)P(x^*, \pmb w, \pmb u^*)}{P(y^*\mid x^*, \pmb w, \pmb u^*)P(x^*, \pmb w, \pmb u^*)}\\
&=\ln\frac {P(x^*, y,\pmb w, \pmb u^*)}{P(x^*,y^*, \pmb w, \pmb u^*)}\\
&=\ell(\sigma_{\pmb W \cup \{Y\}}[\pmb d])-\ell(\sigma_{\pmb W}[\pmb d])
\end{align}
$$

where the third equality is a consequence of the fact that $X$ and $Y$ are not connected directly by an edge, and hence we have that $P\models(X\ \bot\ Y\ |\ \mathcal{X}-\{X,Y\})$ . 
> 其中，第三个等式是因为 $X, Y$ 并没有直接相连，因此有 $P\vDash (X \perp Y \mid \mathcal X - \{X, Y\})$，故 $X$ 在条件中取任何值都无所谓；第四个等式是将 $\frac {P (x, \pmb w, \pmb u^*)}{P (x, \pmb w, \pmb u^*)}$ 替换为了 $\frac {P (x^*, \pmb w, \pmb u^*)}{P (x^*, \pmb w, \pmb u^*)}$，也就是用1替换1

Thus, we have that each term in the outside summation in equation (4.9) adds to zero, and hence the summation as a whole is also zero, as required. 
> 因此，我们可以得到式 (4.9) 中，对于存在不直接相连的两个点的团 $\pmb D$，和式中的四个项求和得到0，因此整个和式的结果也是0

For positive distributions, we have already shown that all three sets of Markov assumptions are equivalent; putting these results together with theorem 4.1 and theorem 4.2, we obtain that, **for positive distributions, all four conditions — factorization and the three types of Markov assumptions — are all equivalent.** 
> 我们已经知道，对于正分布，三个 Markov 假设是等价的，此时再加上定理4.8，我们可以知道对于正分布 $P$：$P$ 在 $\mathcal H$ 上分解和 $P$ 满足 $\mathcal H$ 上的三个 Markov 独立性假设这四个条件都是等价的

#### 4.4.2.2 Eliminating Redundancy 
An alternative approach to the issue of overparameterization is to try to eliminate it entirely. We can do so in the context of a feature-based representation, which is sufciently ﬁne-grained to allow us to eliminate redundancies without losing expressive power. The tools for detecting and eliminating redundancies come from linear algebra. 
> 在基于特征的表示下，可以借由线性代数移除多余的表示

We say that a set of features $f_{1},\ldots,f_{k}$ is linearly dependent if there are constants $\alpha_{0},\alpha_{1},.\cdot\cdot,\alpha_{k}$ , not all of which are 0 , so that for all $\xi$ 

$$
\alpha_{0}+\sum_{i}\alpha_{i}f_{i}(\xi)=0.
$$ 
> 对于一组特征 $f_1, \dots, f_k$ ，如果存在一组不全为0的常数 $\alpha_0, \alpha_1, \dots, \alpha_k$，满足对于所有的 $\xi$，都有 $\alpha_0 + \sum_i \alpha_i f_i (\xi) = 0$，则称这一组特征是线性相关的

This is the usual deﬁnition of linear dependencies in linear algebra, where we view each feature as a vector whose entries are the value of the feature in each of the possible instantiations. 

Example 4.13 
Consider again the Misconception example. We can encode the log-factors in example 4.9 as a set of features by introducing indicator features of the form: 

$$
f_{a,b}(A,B)=\left\{\begin{array}{l l}{{1\qquad}}&{{A=a,B=b}}\\ {{0\qquad}}&{{o t h e r w i s e.}}\end{array}\right.
$$ 
Thus, to represent $\epsilon_{1}(A,B)$ , we introduce four features that correspond to the four entries in the energy function. Since $A,B$ take on exactly one of these possible four values, we have that 

$$
f_{a^{0},b^{0}}(A,B)+f_{a^{0},b^{1}}(A,B)+f_{a^{1},b^{0}}(A,B)+f_{a^{1},b^{1}}(A,B)=1.
$$ 
Thus, this set of features is linearly dependent. 

Example 4.14 
Now consider also the features that capture $\epsilon_{2}(B,C)$ and their interplay with the features that capture $\epsilon_{1}(A,B)$ . We start by noting that the sum $f_{a^{0},b^{0}}(A,B)+f_{a^{1},b^{0}}(A,B)$ is equal to 1 when $B=b^{0}$ and 0 otherwise. Similarly, $f_{b^{0},c^{0}}(B,C)+f_{b^{0},c^{1}}(B,C)$ is also an indicator for $B=b^{0}$ . Thus we get that 

$$
f_{a^{0},b^{0}}(A,B)+f_{a^{1},b^{0}}(A,B)-f_{b^{0},c^{0}}(B,C)-f_{b^{0},c^{1}}(B,C)=0.
$$ 
And so these four features are linearly dependent. As we now show, linear dependencies imply non-unique parameterization. 

As we now show, linear dependencies imply non-unique parameterization.
> 线性相关性质暗示了不唯一的参数化

**Proposition 4.5** 
Let $f_{1},\ldots,f_{k}$ be a set of atures with weights $\pmb{w}=\{w_{1},.\,.\,.\,,w_{k}\}$ that form a log-linear representation of a distribution P . If there are coefcients $\alpha_{0},\alpha_{1},.\cdot\cdot,\alpha_{k}$ such that for all $\xi$ 

$$
\alpha_{0}+\sum_{i}\alpha_{i}f_{i}({\xi})=0\tag{4.10}
$$ 
then the log-linear model with weights $\pmb w^{\prime}=\{w_{1}+\alpha_{1},.\,.\,.\,,w_{k}+\alpha_{k}\}$ also represents $P$ . 
> 命题：
> 特征集合 $f_1, \dots, f_k$，权重 $\pmb w = \{w_1, \dots, w_k\}$，构成了分布 $P$ 的对数线性表示
> 如果存在系数 $\alpha_0, \alpha_1, \dots, \alpha_k$，使得对于所有的 $\xi$，有 $\alpha_0 + \sum_i \alpha_i f_i (\xi) = 0$，也就是特征之间存在线性相关，则权重为 $\pmb w' = \{w_1 + \alpha_1, \dots, w_k + \alpha_k\}$ 的对数线性模型也表示了 $P$

Proof Consider the distribution 
> 证明：
> 考虑权重 $\pmb w'$ 定义的对数线性模型，如下所示：

$$
P_{\pmb w^{\prime}}(\xi)\propto\exp\left\{-\sum_{i}(w_{i}+\alpha_{i})f_{i}(\xi)\right\}.
$$ 
Using equation (4.10) we see that
> 根据式 (4.10) ，我们有
> $-\sum_i (w_i + \alpha_i) f_i (\xi) = -\sum_iw_i f_i (\xi) - \sum_i \alpha_i f_i (\xi)$
> 即 $-\sum_i (w_i + \alpha_i) f_i (\xi) = \alpha_0 - \sum_i w_i f_i (\xi)$

$$
-\sum_{i}(w_{i}+\alpha_{i})f_{i}(\xi)=\alpha_{0}-\sum_{i}w_{i}f_{i}(\xi).
$$ 
Thus, 
> 带入即可得到 $P_{\pmb w'}(\xi)\propto \exp\left\{-\sum_i w_i f_i(\xi)\right\}\propto P_{\pmb w}(\xi)$

$$
P_{\pmb w^{\prime}}(\xi)\propto e^{\alpha_{0}}\exp\left\{-\sum_{i}w_{i}f_{i}(\xi)\right\}\propto P(\xi).
$$ 
We conclude that $P_{\pmb w^{\prime}}(\xi)=P(\xi)$ . 
> 因此 $\pmb w'$ 表示的分布和 $\pmb w$ 表示的分布实际上等价

Motivated by this result, we say that a set of linearly dependent features is redundant . A nonredundant set of features is one where the features are not linearly dependent on each other. In fact, if the set of features is nonredundant, then each set of weights describes a unique distribution. 
> 我们称一组线性相关的特征是冗余的，不冗余即一组特征线性无关，此时每一组权重都描述一个唯一的分布

**Proposition 4.6** 
Let $f_{1},\ldots,f_{k}$ be a set of nonredundant features, and let $v,w^{\prime}\in\mathit{I\!R}^{k}$ . If $\mathbf{\omega}\neq\mathbf{\omega}\mathbf{w}^{\prime}$ then $P_{w}\neq P_{w^{\prime}}$ . 
> 命题：
> $f_1, \dots, f_k$ 为一组不冗余的特征，令 $\pmb w, \pmb w' \in R^k$，如果 $\pmb w \ne \pmb w'$，则 $P_{\pmb w}\ne P_{\pmb w'}$

Example 4.15
Can we construct a nonredundant set of features for the Misconception example? We can determine the number of nonredundant features by building the $16\times16$ matrix of the values of the 16 features (four factors with four features each) in the 16 instances of the joint distribution. This matrix has rank of 9, which implies that a subset of 8 features will be a nonredundant subset. In fact, there are several such subsets. In particular, the canonical parameter iz ation shown in ﬁgure 4.11 has nine features of nonzero weight, which form a nonredundant parameter iz ation. The equivalence of the canonical parameter iz ation (theorem 4.7) implies that this set of features has the same expressive power as the original set of features. To verify this, we can show that adding any other feature will lead to a linear dependency. Consider, for example, the feature $f_{a^{1},b^{0}}$ . We can verify that 

$$
f_{a^{1},b^{0}}+f_{a^{1},b^{1}}-f_{a^{1}}=0.
$$ 
Similarly, consider the feature $f_{a^{0},b^{0}}$ . Again we can ﬁnd a linear dependency on other features: 

$$
f_{a^{0},b^{0}}+f_{a^{1}}+f_{b^{1}}-f_{a^{1},b^{1}}=1.
$$ 
Using similar arguments, we can show that adding any of the original features will lead to redundancy. Thus, this set of features can represent any parameter iz ation in the original model. 

## 4.5 Bayesian Networks and Markov Networks 
We have now described two graphical representation languages: Bayesian networks and Markov networks. Example 3.8 and example 4.8 show that these two representations are incomparable as a language for representing independencies: each can represent independence constraints that the other cannot. In this section, we strive to provide more insight about the relationship between these two representations. 

### 4.5.1 From Bayesian Networks to Markov Networks 
Let us begin by examining how we might take a distribution represented using one of these frameworks, and represent it in the other. One can view this endeavor from two diferent perspectives: Given a Bayesian network $\mathcal{B}$ , we can ask how to represent the distribution $P_{\mathcal{B}}$ as a parameterize Markov network; or, given graph G , we can ask how to represent the independencies in $\mathcal G$ using an undirected graph $\mathcal H$ . In other words, we might be interested in ﬁnding a minimal I-map for a distribution $P_{\mathcal{B}}$ , or a minimal I-map for the independencies $\mathcal{I}(\mathcal{G})$ . B We can see that these two questions are related, but each perspective ofers its own insights. 
> 考虑给定一个 Bayesian/Markov 网络表示的分布，如何将它用 Markov/Bayesian 网络表示
> 这个问题可以从两个角度考虑：给定一个贝叶斯网络 $\mathcal B$，如何将分布 $P_{\mathcal B}$ 表示为参数化的 Markov 网络，或给定图 $\mathcal G$，如何使用一个无向图 $\mathcal H$ 表示 $\mathcal G$ 中的独立性
> 换句话说，我们希望找到分布 $P_{\mathcal B}$ 的极小 I-map，或者找到 $\mathcal I (\mathcal G)$ 的极小 I-map

Let us begin by considering a distribution $P_{\mathcal{B}}$ , where $\mathcal{B}$ is a parameterized Bayesian network over a graph $\mathcal G$ . Importantly, the parameterization of B can also be viewed as a parameterization for a ribution: We simply take each CPD $P(X_{i}\mid\mathrm{Pa}_{X_{i}})$ and view it as a factor of scope $X_{i},\mathrm{Pa}_{X_{i}}$ . This factor satisﬁes additional normalization properties that are not generally true of all factors, but it is still a legal factor. This set of factors deﬁnes a Gibbs distribution, one whose partition function happens to be 1. 
> 考虑一个贝叶斯网络定义的分布 $P_{\mathcal B}$，其中 $\mathcal B$ 是在图 $\mathcal G$ 上参数化的贝叶斯网络
> 贝叶斯网络 $\mathcal B$ 的参数化也可以视作对一个 Gibbs 分布的参数化：将每个 CPD $P(X_i \mid \text{Pa}_{X_i})$ 视作在作用域 $X_i, \text{Pa}_{X_i}$ 上的因子
> 由此得到的一组因子定义了 Gibbs 分布，其划分函数恰好为 1 
> (因此实际上就是将 $P_{\mathcal B}$ 视为了一个 Gibbs 分布)
> (因为各个因子就是 CPD，故所有因子乘起来正好就是联合分布 $P_{\mathcal B}$ ，故 $\sum_{X_1,\dots, X_n} (\prod_{X_i} P (X_i\mid \text{Pa}_{X_i})) = \sum_{X_1,\dots, X_n} P_{\mathcal B}(X_1,\dots, X_n) = 1$，即划分函数为 1)

What is more important, **a Bayesian network conditioned on evidence $E=e$ also induces a Gibbs distribution: the one deﬁned by the original factors reduced to the context $E=e$ .** 
>更重要的是，一个条件于证据 $E=e$ 的贝叶斯网络也会诱导出一个吉布斯分布：这个分布即由原始因子定义的分布在上下文 $E=e$ 下简化后得到的分布

**Proposition 4.7** 
Let $\mathcal{B}$ be a Bayesian network over $\mathcal{X}$ and $\boldsymbol E\,=\,\boldsymbol e$ an observation. Let $W\,=\,\mathcal{X}\,-\,E$ . Then $P_{\mathcal{B}}(W\mid e)$ is a Gibbs distribution deﬁned by the factors $\Phi=\{\phi_{X_{i}}\}_{\mathcal{X}_{i}\in\mathcal{X}}$ , where 

$$
\phi_{X_{i}}=P_{\mathcal{B}}(X_{i}\mid\mathrm{Pa}_{X_{i}})[\boldsymbol{E}=\boldsymbol{e}].
$$ 
The partition function for this Giabbs distribution is $P(e)$ . 
> 命题：
> $\mathcal B$ 为 $\mathcal X$ 上的贝叶斯网络，$\pmb E = \pmb e$ 为观测
> 令 $\pmb W = \mathcal  X - \pmb E$，则分布 $P_{\mathcal B}(\pmb W \mid \pmb e)$ 是由因子 $\Phi = \{\phi_{X_i}\}_{\mathcal X_i \in \mathcal X}$ 定义的分布，其中的因子 $\phi_{X_i}$ 即 CPD 定义的因子（形式不改变，直接把 CPD 看作为因子）在上下文 $[\pmb E = \pmb e]$ 下简化后得到的因子

(证明：

$$
\begin{align}
P_{\Phi}[\pmb e] &= \frac 1 {Z'}\prod_{i=1}^K{\phi_{X_i}[\pmb e]}\\
&=\frac 1 {Z'}\prod_{i=1}^K P_{\mathcal B}(X_i \mid \text{Pa}_{X_i})[\pmb e]\\
&=\frac 1 {Z'} P_{\mathcal B}[\pmb e]\\
&=\frac 1 {Z'} P_{\mathcal B}(\pmb W, \pmb e)
\end{align}
$$

其中第三个等号来源于将每个因子根据上下文筛选和直接相乘再对乘积进行上下文筛选是一致的，其结果都等价于将原 Gibbs 分布在上下文下简化得到的 Gibbs 分布，因为因子乘积结果中和上下文一致的一项来源于每个因子中和上下文一致的各项相乘得到
其中第四个等号直接来源于定义，即简化的因子应该满足 $\phi' (\pmb w)[\pmb e] = \phi (\pmb w, \pmb e)$

因此，划分函数 $Z' = \sum_{\pmb w} P_{\mathcal B}(\pmb W, \pmb e) = P_{\mathcal B}(\pmb e)$
证毕 )

The proof follows directly from the deﬁnitions. This result allows us to view any Bayesian network conditioned as evidence as a Gibbs distribution, and to bring to bear techniques developed for analysis of Markov networks. 

What is the structure of the undirected graph that can serve as an I-map for a set of factors in a Bayesian network? In other words, what is the I-map for the Bayesian network structure $\mathcal{G}\mathrm{:}$ Going back to our construction, we see that we have created a factor for each family of $X_{i}$ , containing all the variables in the family. Thus, in the undirected I-map, we need to have an edge between $X_{i}$ and each of its parents, as well as between all of the parents of $X_{i}$ . This observation motivates the following deﬁnition: 
> 通过贝叶斯网络转化得到的无向图中，其变量之间的依赖性需要保持（不然就会引入额外的独立性），故显然，贝叶斯网络中直接相连的边应该在无向图中也保持，也就是 $X_i$ 和它的父变量之间的边

**Definition 4.16** moralized graph
The moral graph $\mathcal M[\mathcal G]$ of a Bayesian network structure $\mathcal G$ over $\mathcal X$ is the undirected graph over X that contains an undirected edge between X and Y if: (a) there is a directed edge between them (in either direction), or (b) X and Y are both parents of the same node.
> 定义：
> 贝叶斯网络结构 $\mathcal{G}$ 在变量集 $\mathcal{X}$ 上的道德图 $\mathcal{M}[\mathcal{G}]$ 是一个无向图，如果满足：
> (a) $\mathcal G$ 中存在从 $X$ 到 $Y$ 或从 $Y$ 到 $X$ 的有向边；或 
> (b) $\mathcal G$ 中 $X$ 和 $Y$ 都是同一个节点的父母节点
> 则 $\mathcal M[\mathcal G]$ 中 包含一条无向边连接变量 $X$ 和 $Y$ 

For example, ﬁgure 4.6a shows the moralized graph for the extended $\mathcal{B}^{s t u d e n t}$ network of ﬁgure 9.8. 

The preceding discussion shows the following result: 

**Corollary 4.2** 
Let $\mathcal{G}$ be Bayesian network structure Then for y distribution $P_{\mathcal{B}}$ such that $\mathcal{B}$ is a parameterization of G , we have that $\mathcal{M}[\mathcal{G}]$ is an I-map for $P_{\mathcal{B}}$ . 
> 引理：
> 有贝叶斯网络结构 $\mathcal G$，对于 $\mathcal G$ 的任意参数化 $\mathcal B$ 对应的分布 $P_{\mathcal B}$，$\mathcal M[\mathcal G]$ 是 $P_{\mathcal B}$ 的 I-map

One can also view the moralized graph construction purely from the perspective of the independencies encoded by a graph, avoiding completely the discussion of parameterizations of the network. 

**Proposition 4.9** 
Let $\mathcal{G}$ be any Bayesian network graph. The moralized graph $\mathcal{M}[\mathcal{G}]$ is a minimal $I_{\cdot}$ -map for $\mathcal{G}$ . 
> 命题：
> 有贝叶斯网络结构 $\mathcal G$，其道德图 $\mathcal M[\mathcal G]$ 是 $\mathcal G$ 的极小 I-map

Proof We ant to build a Markov network $\mathcal{H}$ such that ${\mathcal{I}}({\mathcal{H}})\subseteq{\mathcal{I}}({\mathcal{G}})$ , that is, that $\mathcal{H}$ is an I-map for G (see deﬁnition 3.3). We use the algorithm or constructing minimal maps based on the Markov independeies. Consider a node X in $\mathcal{X}$ : our task is to select as X ’s neighbors the smallest set of nodes $U$ that are needed to render X independent of all other nodes in the network. We deﬁne the arkov blank of $X$ in a Bayesian network $\mathcal{G}$ , denoted ${\mathrm{MB}}_{\mathcal{G}}(X)$ , to be the nodes consisting of X ’s parents, X ’s children, and other parents of X ’s children. We now need to show that ${\mathrm{MB}}_{\mathcal{G}}(X)$ d-separates $X$ from all other variables in $\mathcal{G}$ ; and that no subset of ${\mathrm{MB}}_{\mathcal{G}}(X)$ has that property. The proof uses straightforward graph-theoretic properties of trails, and it is left as an exercise (exercise 4.14). 
> 证明：
> 我们想要构建一个 Markov 网络 $\mathcal H$，$\mathcal H$ 是 $\mathcal G$ 的 I-map，也就是 $\mathcal I (\mathcal H) \subseteq \mathcal I (\mathcal G)$ (definition 3.3)
> 我们使用基于 Markov 独立性构建极小 I-map 的算法 (4.3.3节，主要思想就是从完全图开始，根据 Markov 独立性删去变量之间的边，这样构造可以保证图中不会有分布中没有的独立性)：
> 考虑 $\mathcal X$ 中的节点 $X$，在我们构建的 $\mathcal H$ 中，$X$ 的所有邻居节点应该构成它的 Markov 毯，即给定 $X$ 的 Markov 毯，$X$ 应该与其余所有节点独立，那么在有向图 $\mathcal G$ 中满足这一点的节点集应该包括 $X$ 的所有父节点、$X$ 的所有子节点以及 $X$ 的子节点的其他父节点，给定该节点集，$X$ 独立于所有其他节点，该节点集被我们定义为 $X$ 在贝叶斯网络 $\mathcal G$ 中的 Markov 毯，记作 $\text{MB}_{\mathcal G}(X)$
> 容易证明 $\text{MB}_{\mathcal G}(X)$ 将 $X$ 和 $\mathcal G$ 中的所有其他变量 d-seperate，并且没有 $\text{MB}_{\mathcal G}(X)$ 的子集满足这一性质
> 显然，$\mathcal M[\mathcal G]$ 就可以视作是根据 $\mathcal G$ 中的 Markov 独立性构建出的 Markov 网络 $\mathcal H$，显然有 $\mathcal I (\mathcal M[\mathcal G])\subseteq \mathcal I (\mathcal G)$，并且 $\mathcal M[\mathcal G]$ 删去任意一边，都会引入在 $\mathcal G$ 中不成立的独立性，故 $\mathcal M[\mathcal G]$ 是 $\mathcal G$ 的极小 I-map

Now, let us consider how “close” the moralized graph is to th original graph $\mathcal{G}$ . **Intuitively, the addition of the moralizing edges to the Markov network H leads to the loss of independence information implied by the graph structure.** For example, if our Bayesian network $\mathcal{G}$ has the form $X\rightarrow Z\leftarrow Y$ with no edge between $X$ and $Y$ , the Markov network $\mathcal{M}[\mathcal{G}]$ loses the information that X and Y are marginally independent (not given Z ). However, information is not always lost. Intuitively, moralization causes loss of information about independencies only when it introduces new edges into the graph. 
> $\mathcal M[\mathcal G]$ 通过保留 $\mathcal G$ 中的边，以及为 $\mathcal G$ 中的 v-structure 添加边得到，直观上，为 v-structure 添加的边会让 $\mathcal M[\mathcal G]$ 相较于 $\mathcal G$ 多了一些依赖性信息，也就是失去了一些独立性信息
> 考虑 v-structure $X\rightarrow Z \leftarrow Y$，在 $\mathcal M[\mathcal G]$ 中，$X, Y$ 会被相连，因此 $\mathcal M[\mathcal G]$ 失去了 $X, Y$ 边际独立的独立性信息
> 显然，$\mathcal M[\mathcal G]$ 只有在 $\mathcal G$ 中存在 v-structure 时，会引入新的边，导致失去部分独立性信息

We say that a Bayesian network $\mathcal{G}$ moral if it contains no immoralities (as in deﬁnition 3.11); that is, for any pair of variables $X,Y$ that share a child, there is a covering edge between $X$ and $Y$ . It is not difcult to show that: 
>如果一个贝叶斯网络 $\mathcal{G}$ 不包含任何 immorality (definition 3.11，也就是 v-structure)，我们称它是 moral
>也就是说，在 $\mathcal G$ 中，对于任意一对共享一个孩子的变量 $X$ 和 $Y$，在 $X$ 和 $Y$ 之间存在一条覆盖边

**Proposition 4.9**
If the directed graph G is moral, then its moralized graph M[G] is a perfect map of G.
> 命题：
> 如果有向图 $\mathcal G$ 是 moral，它的 moralized graph $\mathcal M[\mathcal G]$ 就是 $\mathcal G$ 的 perfect-map

Proof Let $\mathcal{H}=\mathcal{M}[\mathcal{G}]$ . We have already shown that ${\mathcal{I}}({\mathcal{H}})\subseteq{\mathcal{I}}({\mathcal{G}})$ , so it re opposite inclusion. Assume by contradiction that there is an independence $(X\ \bot\ Y\ |\ Z)\in$ $\mathcal{I}(\mathcal{G})$ which not in $\mathcal{I}(\mathcal{H})$ . Thus, there must exist some trail from X to $Y$ in H which is active given Z . Consider some such trail that is minimal, in the sense that it has no shortcuts. As $\mathcal{H}$ and $\mathcal{G}$ have precisely the same edges, the same trail must exist in $\mathcal{G}$ . As t cannot be active in $\mathcal{G}$ given Z , we conclude that it must contain a v-structure $X_{1}\rightarrow X_{2}\leftarrow X_{3}$ . However, because G is moralized, we also have some edge between $X_{1}$ and $X_{3}$ , contradicting the assumption that the trail is minimal. 
> 证明：
> 令 $\mathcal H = \mathcal M[\mathcal G]$，我们已知 $\mathcal I (\mathcal H) \subseteq \mathcal I (\mathcal G)$，故我们需要证明 $\mathcal I (\mathcal G) \subseteq \mathcal I (\mathcal H)$ 
> 使用反证法，假设存在 $(\pmb X\perp \pmb Y \mid \pmb Z) \in \mathcal I (\mathcal G)$ 且 $(\pmb X\perp \pmb Y \mid \pmb Z) \not \in \mathcal I (\mathcal H)$，也就是 $\mathcal H$ 中在给定 $\pmb Z$ 的条件下，在 $\pmb X, \pmb Y$ 之间存在活跃的迹
> 考虑这些活跃的迹中极小的一条，也就是没有捷径的一条迹，因为 $\mathcal H, \mathcal G$ 有完全相同的边，故该迹也存在于 $\mathcal G$ 中，因为 $(\pmb X \perp \pmb Y \mid \pmb Z) \in \mathcal I (\mathcal G)$，故该迹在给定 $\pmb Z$ 时一定不活跃，因此它一定包含 v-structure，这和 $\mathcal G$ 是 moral 的事实矛盾

Thus, a moral graph $\mathcal{G}$ can be converted to a Markov network without losing indepe ence assumptions. This conclusion is fairly intuitive, inasmuch as the only independencies in $\mathcal G$ that are not present in an undirected graph containing the same edges are those corresponding to v-structures. But if any v-structure can be short-cut, it induces no independencies that are not represented in the undirected graph. 
> 因此，moral graph $\mathcal G$ 可以在不损失独立性信息的情况下被转化为 Markov 网络
> 实际上，有向图 $\mathcal G$ 中不能在和它边相同的无向图中包含的唯一的独立性就是 v-structure 带来的独立性

We note, however, that very few directed graphs are moral. For example, assume that we have a v-structure $X\rightarrow Y\leftarrow Z$ , wh h is mora stence of an arc $X\rightarrow Z$ . If $Z$ has another paren $W$ , it o has a v-structure $X\rightarrow Z\leftarrow W$ → ← , which, to be moral, requires some edge between X and W . We return to this issue in section 4.5.3. 
> 但一般很少的有向图是 moral，例如一个包含 $X\rightarrow Y \leftarrow Z$ 的有向图，我们引入边 $X \rightarrow Z$ 使其 moral，但如果 $Z$ 有另一个父变量，则 v-structure $X \rightarrow Z \leftarrow W$ 又会存在

#### 4.5.1.1 Soundness of d-Separation 
The connection between Bayesian networks and Markov networks provides us with the tools for proving the soundness of the d-separation criterion in Bayesian networks. 
> 本节证明贝叶斯网络中 d-seperation 准则的可靠性

The idea behind the proof is to leverage the soundness of separation in undirected graphs, a result which (as we showed) is much easier to prove. Thus, we want to construct an undirected graph $\mathcal{H}$ such that active paths in $\mathcal{H}$ correspond to active paths in $\mathcal{G}$ . A moment of thought shows that the moralized graph is not the right construct, because there are paths in the undirected graph that orr ructu s in $\mathcal{G}$ that may or may not be active. For exa le, if our graph G is $X\rightarrow Z\leftarrow Y$ and Z is ot observed, d-separation tells us that $X$ and $Y$ are independent; but the moralized graph for G is the complete undirected graph, which does not have the same independence. 
> 证明的思路是利用无向图中 seperation 准则的可靠性
> 我们希望构造一个无向图 $\mathcal H$，使得 $\mathcal H$ 中的活跃路径和 $\mathcal G$ 中的活跃路径对应
> 直接使用 moralized graph $\mathcal M[\mathcal G]$ 并不合适，它无法建模 $\mathcal G$ 中的 v-structure $X\rightarrow Z \leftarrow Y$ 中 $Z$ 未被观察到时 $X, Y$ 的边际独立性 (该独立性是可以被 d-seperation 检测到的)

Therefore, to show the result, we ﬁrst want to eliminate v-structures that are not active, so as to remove such cases. To do so, we ﬁrst construct a subgraph where remove all barren nodes from the graph, thereby also removing all v-structures that do not have an observed descendant. The elimination of the barren nodes does not change the independence properties of the distribution over the remaining variables, but does eliminate paths in the graph involving v-structures that are not active. If we now consider only the subgraph, we can reduce d-separation to separation and utilize the soundness of separation to show the desired result. 
>我们首先希望消除那些不活跃的 v-structure
>为此，我们首先构建一个子图，该子图移除了所有空节点，这也会移除所有没有观测到的后代的 v-structure，移除空节点不会改变剩余变量上的分布的独立性质，但会消除涉及不活跃 v-structure 的路径
>现在只考虑这个子图，我们可以将有向图中的 d-seperation 简化为无向图中的 seperation 并利用 seperation 的可靠性来证明 d-seperation 的可靠性

>[!barren node]
>   
>在概率图模型（如贝叶斯网络或马尔可夫网络）中， “barren node”通常指的是那些在模型中不起作用或没有贡献的节点，具体来说：
>
>1. 没有下游节点：如果一个节点没有任何下游节点，那么它不会影响其他节点的状态。
>2. 状态不传播：如果一个节点的状态不能通过模型传播到其他节点，那么它也不会对整体的概率分布产生影响。
>
>在一个有效的概率图模型中，每个节点都应当对其周围的节点或者整个系统的概率分布有所贡献，如果某个节点完全孤立，或者它的状态无法通过模型传播到其他节点，那么这个节点就可以被认为是“barren node”
>
>对于一个 v-structure $A \rightarrow B \leftarrow C$，如果 $B$ 没有后续的下游节点，则节点 $B$ 的状态完全由节点 $A, C$ 决定，并且 $B$ 没有被观测到时，$B$ 不传递任何额外的信息给其他节点，故 $B$ 此时就是 barren node
>
>移除 barren node 不会对其他节点的分布产生影响，因为 barren node 本身就不影响任何其他节点

We ﬁrst use these intuitions to provide an alternative formulation for d-separation. Recall that in deﬁnition 2.14 we deﬁned the upward closure of a set of nodes $U$ in a graph to be $U\cup$ Ancestors U . Letting $U^{*}$ be the closure of a s $U$ , we can de the network induced over $U^{*}$ ; importantly, as all parents of every node in $U^{*}$ are also in U $U^{*}$ , we have all the variables mentioned in every CPD, so that the induced graph deﬁnes a coherent probability distribution. We let ${\mathcal{G}}^{+}[U]$ be the induced Bayesian network over $U$ and its ancestors. 
> 首先考虑为 d-seperation 提供另一种表述方式
> 在 definition 2.14 中，我们将节点集 $\pmb U$ 的上闭包定义为 $\pmb U \cup Ancestors_{\pmb U}$，我们将其记作 $\pmb U^*$
> 随后，我们定义在 $\pmb U^*$ 导出的贝叶斯网络，记作 $\mathcal G^+[\pmb U]$ 
> 因为 $\pmb U^*$ 中的任意节点的所有父节点都在 $\pmb U^*$ 中，因此导出的贝叶斯网络中任意 CPD 相关的变量都在 $\pmb U^*$ 中，导出的贝叶斯网络 $\mathcal G^+[\pmb U]$ 定义了一个一致的概率分布

**Proposition 4.10** 
Let $X,Y,Z$ three disjoint sets of nodes in a Bayesia twork $\mathcal{G}$ . Let $U=X\cup Y\cup Z,$ , and let G $\mathcal{G}^{\prime}=\mathcal{G}^{+}[\boldsymbol{U}]$ G be the induced Bayesian network over $U\cup$ ∪ Ancestors $U$ . Let H be the moralized graph $\mathcal{M}[\mathcal{G}^{\prime}]$ . Then $\operatorname{d-sep}_{\mathcal{G}}(X;Y\mid Z)$ if and only if $\mathrm{sep}_{\mathcal{H}}(X;Y\mid Z)$ . 
> 命题：
> $\pmb X, \pmb Y, \pmb Z$ 为 $\mathcal G$ 中三个不相交的节点集，记 $\pmb U = \pmb X \cup \pmb Y \cup \pmb Z$，记 $\pmb U$ 导出的贝叶斯网络为 $\mathcal G' = \mathcal G^+[\pmb U]$，记导出贝叶斯网络的 moral graph 为 $\mathcal H = \mathcal M[\mathcal G']$，则 $\text{d-sep}_{\mathcal G}(\pmb X; \pmb Y\mid \pmb Z)$ 存在当且仅当 $\text{sep}_{\mathcal H}(\pmb X; \pmb Y\mid \pmb Z)$ 存在
> 也就是贝叶斯网络 $\mathcal G$ 中的 d-seperation 和对应的导出的贝叶斯网络的 moral graph 的 seperation 等价

> 该思想类似于分类讨论
> 我们不能直接将 $\mathcal G$ 中的 d-seperation 和 $\mathcal M[\mathcal G]$ 中的 seperation 等价，因为 $\mathcal G$ 中的 v-structure 在不给定子节点时编码了条件独立，而 $\mathcal M[\mathcal G]$ 不能表示这一点，因为 $\mathcal M[\mathcal G]$ 无论子节点给定不给定都会将 v-structure 的父节点相连
> 因此，我们针对每个具体的 d-seperation 定义导出的网络 $\mathcal G'$，如果某个 v-structure 的子节点不给定，则 $\mathcal G'$ 中不会包含该子节点，也就是直接消除了 v-structure，故 $\mathcal M[\mathcal G']$ 自然对应编码了边际独立性

![[Probabilistic Graph Theory-Fig4.12.png]]

Example 4.16 
To gain some intuition for this result, consider the Bayesian network G of figure 4.12a (which extends our Student network). Consider the d-separation query d-sepG (D; I j L). In this case, U = fD; I; Lg, and hence the moralized graph M[G+[U]] is the graph shown in figure 4.12b, where we have introduced an undirected moralizing edge between D and I. In the resulting graph, D and I are not separated given L, exactly as we would have concluded using the d-separation procedure on the original graph. On the other hand, consider the d-separation query d-sepG (D; I j S; A). In this case, U = fD; I; S; Ag. Because D and I are not spouses in G+[U], the moralization process does not add an edge between them. The resulting moralized graph is shown in ﬁgure 4.12c. As we can see, we have that $\mathsf{s e p}_{\mathcal{M}[\mathcal{G}^{+}[\boldsymbol{U}]]}(D;I\mid S,A)$ , as desired. 

The proof for the general case is similar and is left as an exercise (exercise 4.15). 
With this result, the soundness of $\mathrm{d}$ -separation follows easily. We repeat the statement of theorem 3.3: 

**Theorem 4.9**
If a distribution $P_{\mathcal{B}}$ factorizes according to $\mathcal{G}$ , then $\mathcal{G}$ is an $I\cdot$ -map for $P$ . 
> 定理：
> 如果分布 $P_{\mathcal B}$ 根据 $\mathcal G$ 分解，则 $\mathcal G$ 是 $P$ 的 I-map
>(可靠性：$\mathcal G$ 中 d-seperation 导出的独立性在 $P$ 中一定存在)

Proof As in proposition 4.1 t $U=X\cup Y\cup Z$ , let $U^{*}=U\cup A n c e s t o r s_{U}$ , let $\mathcal{G}_{U^{\ast}}=\mathcal{G}^{+}[U]$ be the induced graph over $U^{*}$ , and let H be the moralized graph $\mathcal{M}[\mathcal{G}_{U^{*}}]$ . Let $P_{U^{*}}$ be the Ba an network distrib tion deﬁn ver ${\mathcal{G}}_{U^{*}}$ in the obvious way: the CPD for any variable in $U^{*}$ is the same as in B . Because $U^{*}$ is upwardly closed, all variables used in these CPDs are in $U^{*}$ . 

Now, consider an independence assertion $(X\,\perp\,Y\,\mid\,Z)\,\in\,{\mathcal{I}}({\mathcal{G}})$ ; we want to prove that $P_{\mathcal{B}}=(X\perp Y\mid Z)$ . By deﬁnition 3.7, if $(X\perp Y\mid Z)\in{\mathcal{I}}(\mathcal{G})$ , we have that $\mathit{d-s e p}_{\mathcal{G}}(X;Y\mid$ $Z)$ . It follows th $s e p_{\mathcal{H}}(X;Y\mid Z)$ , and hence that $(X\perp Y\mid Z)\in{\mathcal{Z}}({\mathcal{H}})$ . P $P_{U^{*}}$ is a Gibbs distribution over H , and hence, from theorem 4.1, $P_{U^{*}}\mid=(X\perp Y\mid Z)$ ⊥ | . e distribution $P_{U^{*}}(U^{*})$ is the same as $P_{\mathcal{B}}(U^{*})$ . Hence, it follows also that $P_{\mathcal{B}}\models(X\ \bot\ Y\ |\ Z)$  , B proving the desired result. 
> 证明：
> 令 $\pmb U = \pmb X \cup \pmb Y \cup \pmb Z$，记 $\pmb U^* = \pmb U \cup Ancestors_{\pmb U}$ ，记 $\mathcal G_{\pmb U^*} = \mathcal G^+[\pmb U]$，记 $\mathcal H = \mathcal M[\mathcal G_{\pmb U^*}]$，记 $P_{\pmb U^*}$ 为 $\mathcal G_{\pmb U^*}$ 上定义的分布，定义方式就是 $\mathcal G_{\pmb U^*}$ 中的 CPD 相乘得到，注意 $\mathcal G_{\pmb U^*}$ 中的 CPD 和 $\mathcal B$ 中的对应 CPD 都是一致的 ($\pmb U^*$ 向上封闭，因此包含了所有 CPD 相关的变量)
> 考虑独立性 $(\pmb X \perp \pmb Y \mid \pmb Z)\in \mathcal I (\mathcal G)$，我们要证明 $P_{\mathcal B}\vDash(\pmb X \perp \pmb Y \mid \pmb Z)$
> 根据 defintion 3.7，$(\pmb X \perp \pmb Y \mid \pmb Z)\in \mathcal I (\mathcal G)$ 等价于 $\text{d-sep}_{\mathcal G}(\pmb X; \pmb Y \mid \pmb Z)$ ，进而等价于 $\text{sep}_{\mathcal H}(\pmb X; \pmb Y \mid \pmb Z)$，进而等价于 $(\pmb X \perp \pmb Y \mid \pmb Z)\in \mathcal I (\mathcal H)$；而 $P_{\pmb U^*}$ 可以视作分解于 $\mathcal H$ 上的 Gibbs 分布，根据 Markov 网络的 seperation 的可靠性，有 $P_{\pmb U^*}\vDash (\pmb X \perp \pmb Y \mid \pmb Z)$，by exercise 3.8，导出网络中定义的联合分布和原网络中的联合分布是相同的，也就是 $P_{\pmb U^*}(\pmb U^*) = P_{\mathcal B}(\pmb U^*)$，因此 $P_{\pmb U^*}$ 中成立的独立性显然在 $P_{\mathcal B}$ 中也成立，即 $P_{\mathcal B}\vDash(\pmb X \perp \pmb Y \mid \pmb Z)$

> 该证明的思路：独立性在 $\mathcal G$ 中成立 -> 独立性在 $\mathcal H = \mathcal M[\mathcal G^+[\pmb U]]$ 中成立 -> 独立性在 $P_{\pmb U^*}$ 中成立 -> 独立性在 $P_{\mathcal B}$ 中成立

### 4.5.2 From Markov Networks to Bayesian Networks 
The previous section dealt with the conversion from a Bayesian network to a Markov network. We now consider the converse transformation: ﬁnding a Bayesian network that is a minimal I-map for a Markov network. It turns out that the transformation in this direction is signiﬁcantly more difcult, both conceptually and computationally. Indeed, the Bayesian network that is a minimal I-map for a Markov network might be considerably larger than the Markov network. 
> 上一节讨论了 Bayesian 网络到 Markov 网络的转换
> 本节讨论 Markov 网络到 Bayesian 网络的转换：对于给定的 Markov 网络，找到一个 Bayesian 网络是它的极小 I-map
> 事实上，作为一个 Markov 网络的极小 I-map 的 Bayesian 网络会比原网络大许多

![[Probabilistic Graph Theory-Fig4.13.png]]


Example 4.17 
Consider the Mark v networ ructure $\mathcal{H}_{\ell}$ of ﬁgure 4.13a, and assume that we want to ﬁnd a Bayesian network I-map for $\mathcal{H}_{\ell}$ . As we discussed in section 3.4.1, we can ﬁnd such an I-map by enumerating the nodes in X in some ordering, and deﬁne the parent set for each one in turn according to the independencies in the distribution. Assume we enumerate the nodes in the order $A,B,C,D,E,F$ . The process for $A$ and $B$ is obvious. Consider what happens when we add $C$ . We must, of course, introduce $A$ as a parent for $C$ . More interestingly, however, $C$ is not independent of $B$ given $A$ ; hence, we must also add $B$ as a parent for $C$ . Now, consider the node $D$ . One of its parents must be $B$ . As $D$ is not independent of $C$ given $B$ , we must add $C$ as a parent for $B$ . We do not need to add $A$ , as $D$ is independent of $A$ given $B$ and $C$ . Similarly, $E$ ’s parents must be $C$ and $D$ . Overall, the minimal Bayesian network $I_{\cdot}$ -map according to this ordering has the structure $\mathcal{G}_{\ell}$ shown in ﬁgure 4.13b. 

A quick examination of the structure $\mathcal{G}_{\ell}$ shows that we have added sever edges to the graph, resulting in a set of triangles crisscrossing the loop. In fact, the graph G $\mathcal{G}_{\ell}$ in ﬁgure 4.13b is chordal: all loops have been partitioned into triangles. 

One might hope that a diferent ordering might lead to fewer edges being introduced. Un- fortunately, this phenomenon is a general one: any Bayesian network I-map for this Markov network must add triangulating edges into the graph, so that the resulting graph is chordal (see deﬁnition 2.24). In fact, we can show the following property, which is even stronger: 
>人们可能会希望不同的节点顺序能够引入更少的边，不幸的是，这种现象是普遍存在的：对于这个马尔可夫网络的任何 I-map 贝叶斯网络都必须向图中添加三角化边，从而使所得的图成为弦图（参见定义2.24）
>事实上，我们可以证明以下性质，这一性质甚至更强：

**Theorem 4.10** 
Let H be a Markov network structure, and let G be any Bayesian network minimal I-map for H. Then G can have no immoralities (see definition 3.11).
> 定理：
> $\mathcal H$ 为 Markov 网络结构，$\mathcal G$ 为任意是 $\mathcal H$ 的极小 I-map 的贝叶斯网络结构，则 $\mathcal G$ 中不可能存在 v-structure (immorality)

Proof Let $X_1, \dots, X_n$ be a topological ordering for G. Assume, by contradiction, that there is some immorality Xi ! Xj Xk in G such that there is no edge between Xi and Xk; assume (without loss of generality) that i < k < j. 
Owing to minimality of the I-map G, if Xi is a parent of Xj, then Xi and Xj are not separated by Xj’s other parents. Thus, H necessarily contains one or more paths between Xi and $X_{j}$ that are not cut by $X_{k}$ (or by $X_{j}$ ’s other parents). Similarly, $\mathcal{H}$ necessarily contains one or more paths between $X_{k}$ and $X_{j}$ that are not cut by $X_{i}$ (or by $X_{j}$ ’s other parents). 
> 证明：
> 令 $X_1, \dots, X_n$ 为 $\mathcal G$ 的一个拓扑排序
> 假设 $\mathcal G$ 中存在 v-structure $X_i \rightarrow X_j \leftarrow X_k$，其中 $X_i , X_k$ 之间没有边，不失一般性，我们还假设 $i < k < j$
> 因为 $\mathcal G$ 是 $\mathcal H$ 的极小 I-map，故如果 $X_j$ 在 $\mathcal G$ 中是 $X_i$ 的父节点，则 $X_j$ 就不会被 $X_j$ 的其他父节点分离 (极小性意味着没有边是多余的，如果断开边 $X_j - X_i$，就会引入 $X_j, X_i$ 之间的独立性，因此边 $X_j - X_i$ 不能断开)；因此，在 $\mathcal H$ 中，$X_i, X_j$ 之间会有一条或者多条不被 $X_k$ 或者 $X_j$ 其他父变量的路径，类似地，对于 $X_k$ 也可以这样推理

Consider the parent set $U$ that was chosen for $X_{k}$ . By our previous argument, there are one or more paths in $\mathcal{H}$ between $X_{i}$ and $X_{k}$ via $X_{j}$ . As $i<k$ , and $X_{i}$ is ot a parent of $X_{k}$ (by our assumption), we have that $U$ must cut all of those paths. To do so, U must cut either all of the paths between $X_{i}$ and $X_{j}$ , or all of the paths between $X_{j}$ and $X_{k}$ : As long as there is at least one active path from $X_{i}$ to $X_{j}$ and one from $X_{j}$ to $X_{k}$ , there is an active path between $X_{i}$ and $X_{k}$ that is not cut by $U$ . Assume, without loss of generality, that $U$ cuts all paths between $X_{j}$ and $X_{k}$ (the other case is symmetrical). Now, consider the choice of parent set for $X_{j}$ , and recall that it is the (unique) minimal subset among $X_{1},\dots,X_{j-1}$ that separates $X_{j}$ from the others. In a Markov network, this set consists of all nodes in $X_{1},.\ldots,X_{j-1}$ that are the ﬁrst on some uncut path from $X_{j}$ . As $U$ separates $X_{k}$ from $X_{j}$ , it follows that $X_{k}$ cannot be the ﬁrst on any uncut path from $X_{j}$ , and therefore $X_{k}$ cannot be a parent of $X_{j}$ . This result provides the desired contradiction. 
>考虑为 $X_k$ 选择的父节点集合 $\pmb U$，根据我们之前的论点，在 $\mathcal{H}$ 中有从 $X_i$ 到 $X_k$ 经过 $X_j$ 的一条或多条路径，由于 $i < k$，且 $X_i$ 不是 $X_k$ 的父节点（根据我们的假设），那么 $\pmb U$ 必须切断所有这些路径，也就是说，$\pmb U$ 必须切断 $X_i$ 和 $X_j$ 之间或 $X_j$ 和 $X_k$ 之间的所有路径：只要从 $X_i$ 到 $X_j$ 和从 $X_j$ 到 $X_k$ 至少有一条激活路径，则存在一条未被 $\pmb U$ 切断的从 $X_i$ 到 $X_k$ 的激活路径
>不失一般性，假设 $\pmb U$ 切断了 $X_j$ 和 $X_k$ 之间的所有路径（另一情况是对称的）
>
>现在，考虑 $X_j$ 的父节点集合的选择，回想起来，它是 $X_1,\dots,X_{j-1}$ 中将 $X_j$ 与其他节点分开的唯一最小的子集
>在马尔可夫网络中，这个集合包含在 $X_1,\dots,X_{j-1}$ 中满足是某个未被切断路径上的第一个节点的所有节点，由于 $\pmb U$ 将 $X_k$ 与 $X_j$ 分隔开，因此可以得出结论 $X_k$ 不可能是任何未被切断路径上的 $X_j$ 的第一个节点，因此 $X_k$ 不可能是 $X_j$ 的父节点，这个结果提供了所需的矛盾

Because any nontriangulated loop of length at least 4 in a Bayesian network graph necessarily contains an immorality, we conclude: 
>因为在贝叶斯网络图中，任何长度至少为4的非三角环必然包含一个 immorality，我们得出结论：

**Corollary 4.3** 
Let H be a Markov network structure, and let $\mathcal{G}$ be any minimal I-map for $\mathcal{H}$ . Then $\mathcal{G}$ is necessarily chordal. 
> 引理：
> 令 $\mathcal{H}$ 是一个马尔可夫网络结构，并且令 $\mathcal{G}$ 是 $\mathcal{H}$ 的任何一个极小 I-map，那么 $\mathcal{G}$ 必然是弦图 ($\mathcal G$ 中的任意长度至少为4的环一定会被三角化，也就是包含 covering edge，否则就会构成 immorality/v-structure)

Thus, the process of turning a Markov network into a Bayesian network requires that we add enough edges to a graph to make it chordal. This process is called *triangulation* . As in the transformation from Bayesian networks to Markov networks, the addition of edges leads to the loss of independence information. For instance, in example 4.17, the Bayesian network $\mathcal{G}_{\ell}$ in ﬁgure 4.13b loses the information that $C$ and $D$ are independent given $A$ and $F$ . In the transformation from directed to undirected models, however, the edges added are only the ones that are, in some sense, implicitly there — the edges required by the fact that each factor in a Bayesian network involves an entire family (a node and its parents). By contrast, the transformation from Markov networks to Bayesian networks can lead to the introduction of a large number of edges, and, in many cases, to the creation of very large families (exercise 4.16). 
>因此，将马尔可夫网络转换为贝叶斯网络的过程需要向图中添加足够的边，使其成为弦图，这一过程称为三角化；正如从贝叶斯网络到马尔可夫网络的转换一样，添加边会导致独立性信息的丢失
>例如，在示例4.17中，图4.13b中的贝叶斯网络 $\mathcal{G}_{\ell}$ 失去了 $C$ 和 $D$ 在给定 $A$ 和 $F$ 时是独立的信息
>然而，在从有向模型到无向模型的转换过程中，添加的边只是那些在某种意义上原本就存在的边——即由贝叶斯网络中的每个因子涉及整个家族（一个节点及其父节点）的事实所要求的边；相比之下，从马尔可夫网络到贝叶斯网络的转换可能导致大量边的引入，并且在许多情况下，可能导致非常大的一族节点

### 4.5.3 Chordal Graphs 
We have seen that the conversion in either direction between Bayesian networks to Markov networks can lead to the addition of edges to the graph and to the loss of independence information implied by the graph structure. It is interesting to ask when a set of independence assumptions can be represented perfectly by both a Bayesian network and a Markov network. It turns out that this class is precisely the class of undirected chordal graphs. 
> 无向弦图中的独立性可以被贝叶斯网络完美表示，同时也可以被 Markov 网络完美表示

The proof of one direction is fairly straightforward, based on our earlier results. 

**Theorem 4.11** 
Let $\mathcal{H}$ b nonchordal Markov network. Then there is no Bayesian network $\mathcal{G}$ which is a perfect map for H (that is, such that $\mathcal{I}(\mathcal{H})=\mathcal{I}(\mathcal{G})$ ). 
> 定理：
> $\mathcal H$ 为 nonchordal Markov 网络，则不存在是 $\mathcal H$ 的 perfect map 的贝叶斯网络 $\mathcal G$，也就是不存在满足 $\mathcal I (\mathcal H) = \mathcal I (\mathcal G)$ 的贝叶斯网络 $\mathcal G$

Proof The roof s from the fact that the minimal I-map for $\mathcal{G}$ any I-map G for I $\mathcal{Z}(\mathcal{H})$ H must include edges that are not present in H . Because any additional edge eliminates independence assumptions, it is not possible for any Bayesian network $\mathcal{G}$ to precisely encode $\mathcal{I}(\mathcal{H})$ . 
> 证明：
> 首先 $\mathcal G$ 需要是 $\mathcal H$ 的极小 I-map，故 $\mathcal G$ 必须是弦图，因此 $\mathcal G$ 一定包含 $\mathcal H$ 中没有出现的边；因为任意额外的边会消除独立性，故 $\mathcal G$ 不可能完全编码 $\mathcal I (\mathcal H)$

To prove the other direction of this equivalence, we ﬁrst prove some important properties of chordal graphs. As we will see, chordal graphs and the properties we now show play a central role in the derivation of exact inference algorithms for graphical models. For the remainder of this discussion, we restrict attention to connected graphs; the extension to the general case is straightforward. The basic result we show is that we can decompose any connected chordal graph $\mathcal{H}$ into a *tree of cliques* — a tree whose nodes are the maximal cliques in ${\mathcal{H}}-s{\boldsymbol{0}}$ that the structure of the tree precisely encodes the independencies in H . (In the case of disconnected graphs, we obtain a forest of cliques, rather than a tree.) 
>为了证明这一等价关系的另一方向，我们首先证明弦图的一些重要性质，弦图及其现在展示的性质在精确推理算法的推导中起着核心作用
>在接下来的讨论中，我们将注意力限制在连通图上，推广到一般情况是直接的
>我们要展示的基本结果是：我们可以将任意连通弦图 $\mathcal{H}$ 分解成一个由团组成的树——这棵树的节点是由 ${\mathcal{H}}$ 中的极大团组成，而树的结构精确地编码了 $\mathcal{H}$ 中的独立性（在非连通图的情况下，我们得到的是一个由团组成的森林，而不是一棵树）

We begin by introducing some notation. Le $\mathcal{H}$ be connected undirected graph, and let $C_{1},\ldots,C_{k}$ be the set of maximal cliques in H . Let T $\mathcal{T}$ be any tree-structured graph whose nodes correspond to the maximal cliques $C_{1},\ldots,C_{k}$ . Let $C_{i},C_{j}$ be two cliques in the tree that direct edge; we deﬁne $S_{i,j}=C_{i}\cap C_{j}$ to be a sepset between $C_{i}$ and $C_{j}$ . Let $W_{<(i,j)}\ (W_{<(j,i)})$ ) be all of the variables that appear in any clique on the $C_{i}$ $(C_{j})$ edge us, each edge decomposes $\mathcal{X}$ into three disjoint sets: $W_{<(i,j)}-S_{i,j}$ , $W_{<(j,i)}-S_{i,j}$ − , and $\boldsymbol{S}_{i,j}$ . 
> 引入一些记号：$\mathcal H$ 表示连通无向图；$\pmb C_1, \dots, \pmb C_k$ 为 $\mathcal H$ 中的一系列极大团；$\mathcal T$ 为任意树，其节点对应于极大团；$\pmb C_i, \pmb C_j$ 为树中直接相连的两个团；定义 $\pmb S_{i, j} =  \pmb C_i \cap \pmb C_j$ 为 $\pmb C_i, \pmb C_j$ 之间的分离集 sepset； $\pmb W_{<(i,j)}\ (\pmb W_{<(j,i)})$  表示出现在 $\pmb C_i(\pmb C_j)$ 那一边的团中的任意节点，因此，一条边将 $\mathcal X$ 分为三个不相交集合 $\pmb W_{<(i, j)} - \pmb S_{ij} , \pmb W_{<(j, i)} - \pmb S_{ij}, \pmb S_{ij}$

**Deﬁnition 4.17** clique tree 
We say that a tree $\mathcal{T}$ is $^a$ clique tree for $\mathcal{H}$ if: 

-  each node corresponds to a clique in $\mathcal{H}$ , and each maximal clique in $\mathcal{H}$ is a node in $\mathcal{T}$ ; 
- each sepset $\boldsymbol{S}_{i,j}$ separates $W_{<(i,j)}$ and $W_{<(j,i)}$ in $\mathcal{H}$ . 

> 定义：
> 如果 $\mathcal T$ 满足：
> 每个节点对应于 $\mathcal H$ 中的团，且 $\mathcal H$ 中的每个极大团都对应于 $\mathcal T$ 中的一个节点
> 每个分离集 $\pmb S_{ij}$ 在 $\mathcal H$ 中分离 $\pmb W_{<(i, j)}, \pmb W_{<(j, i)}$
> 称 $\mathcal T$ 是 $\mathcal H$ 的团树

Note that this deﬁnition implies that each separator $\boldsymbol{S}_{i,j}$ renders its two sides conditionally independent in $\mathcal{H}$ . 
>该定义要求每个分离集 $\pmb S_{ij}$ 在 $\mathcal H$ 中分离 $\pmb W_{<(i, j)}, \pmb W_{<(j, i)}$，也就是二者要在 $\mathcal H$ 中条件独立


Example 4.18 
Consider the Bayesian network graph $\mathcal{G}_{\ell}$ in ﬁgure 4.13b. Since it contains no immoralities, its mo ized graph $\mathcal{H}_{\ell}^{\prime}$ is simply the same graph, but where all edges have been made undirected. As $\mathcal{G}_{\ell}$ is chordal, so is $\mathcal{H}_{\ell}^{\prime}$ . The clique tree for $\mathcal{H}_{\ell}^{\prime}$ is simply a chain $\{A,B,C\}\rightarrow\{B,C,D\}\rightarrow$ $\{C,D,E\}\,\rightarrow\,\{D,E,F\}$ , which clearly satisﬁes the separation requirements of the clique tree deﬁnition. 

**Theorem 4.12** 
Every undirected chordal graph $\mathcal{H}$ has a clique tree $\mathcal{T}$ . 
> 定理：
> 每个无向弦图都有团树

Proof We prove the theorem by induction on the number of nodes in the graph. The base case of a single node is trivial. Now, consider a chordal graph $\mathcal{H}$ of size $>1$ . If $\mathcal{H}$ consists of a single clique, then the theorem holds trivially. Therefore, consider the case where we have at least two nodes $X_{1},X_{2}$ that are not connected directly by an edge. Assume that $X_{1}$ and $X_{2}$ are connected, otherwise the inductive step holds trivially. Let $S$ be a minimal subset of nodes that separates $X_{1}$ and $X_{2}$ . 
> 证明跳过

The removal of the set $S$ breaks up the graph into at least two disconnected components — one containing $X_{1}$ , another containing $X_{2}$ , and perhaps additional ones. Let $W_{1},W_{2}$ be some partition of the variables in ${\mathcal{X}}-S$ into two disjoint components, such that $W_{i}$ encompasses the connected component containing $X_{i}$ . (The other connected components can be assigned to $W_{1}$ or $W_{2}$ arbitrarily.) We ﬁrst show that $S$ must be a complete subgraph. Let $Z_{1},Z_{2}$ be any two variables in $S$ . Due to the minimality of $S$ , each $Z_{i}$ must lie on a path between $X_{1}$ and $X_{2}$ that does not go through any other node in $S$ . (Otherwise, we could eliminate $Z_{i}$ from $S$ while still maintaining separation.) We can therefore construct a minimal path from $Z_{1}$ to $Z_{2}$ that goes only through nodes in $W_{1}$ by constructing a path from $Z_{1}$ to $X_{1}$ to $Z_{2}$ that goes only through $W_{1}$ , and by eliminating any shortcuts. We can similarly construct a minimal path from $Z_{1}$ to $Z_{2}$ that goes only through nodes in $W_{2}$ . The two paths together form a cycle of length $\geq4$ . Because of chordality, the cycle must have a chord, which, by construction, must be the edge $Z_{1}{-}Z_{2}$ . 

Now cons r the induc graph ${\mathcal{H}}_{1}\,=\,{\mathcal{H}}[W_{1}\cup S]$ . As $X_{2}\notin\mathcal{H}_{1}$ , this induced gr h is smaller than H . M over, H $\mathcal{H}_{1}$ is hordal, so we can apply the inductive hypothesis. Let T $\mathcal{T}_{1}$ be the clique tree for H $\mathcal{H}_{1}$ . Because S is a compl conne subgraph, it is eit a maximal ique or a subset of some maximal clique in H $\mathcal{H}_{1}$ . Let $C_{1}$ be some ue in $\mathcal{T}_{1}$ con ing $S$ (ther ay b ore than on such clique). We can imilarly deﬁne H $\mathcal{H}_{2}$ and $C_{2}$ for $X_{2}$ . If ne er $C_{1}$ or $C_{2}$ is equal $S$ , we nstruct a tree T $\mathcal{T}$ that contains the union of the cliques $\mathcal{T}_{1}$ d T $\mathcal{T}_{2}$ , and nnects $C_{1}$ a $C_{2}$ by edge therwise, with loss of generality, let $C_{1}=S$ ; we create T $\mathcal{T}$ by merging T $\mathcal{T}_{1}$ minus $C_{1}$ into T $\mathcal{T}_{2}$ , making all of $C_{1}$ ’s neighbors adjacent to $C_{2}$ instead. 

It remains to sho hat the resulting structure is a clique tree for $\mathcal{H}$ . First, we note at there is no clique in H that intersects both $W_{1}$ and $W_{2}$ ; hence, any aximal clique in H is a maximal ique in eith $\mathcal{H}_{1}$ or $\mathcal{H}_{2}$ (or both in e possible case of S ), so that all maxi l cliques in H appear in T . Thus, the nodes in T $\mathcal{T}$ are precisely the maximal cliques in H . Second, we need to show that any $\boldsymbol{S}_{i,j}$ separates $W_{<(i,j)}$ and $W_{<(j,i)}$ . Consider two variables $X\in W_{<(i,j)}$ and $Y\in W_{<(j,i)}$ . First, assume that $X,Y\in{\mathcal{H}}_{1}$ ; as all the nodes in $\mathcal{H}_{1}$ re on the T $\mathcal{T}_{1}$ side o tree, we also have that $S_{i,j}\subset\mathcal{H}_{1}$ . path bet two node $\mathcal{H}_{1}$ H t goes through $W_{2}$ can be shor t to go only through H $\mathcal{H}_{1}$ . Thus, if $\boldsymbol{S}_{i,j}$ separates $X,Y$ in H $\mathcal{H}_{1}$ , so separates them in H . The same argument applies for $X,Y\in{\mathcal{H}}_{2}$ . Now, sider $X\in W_{1}$ and $Y\in W_{2}$ . If $\boldsymbol{S}_{i,j}=\boldsymbol{S}$ , th esult follows from th fact at S separates $W_{1}$ and $W_{2}$ . Otherwise, a me th $\boldsymbol{S}_{i,j}$ in T $\mathcal{T}_{1}$ , on the path f m X to $C_{1}$ 1 . In this case, we have that $\boldsymbol{S}_{i,j}$ separates X from S , and S separates $\boldsymbol{S}_{i,j}$ from Y . The conclusion now follows from the transitivity of graph separation. 

We have therefore constructed a clique tree for $\mathcal{H}$ , proving the inductive claim. 

Using this result, we can show that the independe es in an undirected graph $\mathcal{H}$ can be captured perfectly in a Bayesian network if and only if is chordal. 

**Theorem 4.13**
Let H be a chordal Markov network. Then there is a Bayesian network G such that I(H) = I(G) 
> 定理：
> $\mathcal H$ 为弦图 Makrov 网络，则存在贝叶斯网络 $\mathcal G$ 使得 $\mathcal I (\mathcal H) = \mathcal I (\mathcal G)$

Proof let clique $C_{1}$ to be the root of the clique tree, and then order the cliques $C_{1},\ldots,C_{k}$ using any topological ordering, that is, where cliques closer to the root are ordered ﬁrst. We now order the nodes in the network in any ordering consistent with the clique ordering: if $X_{l}$ ﬁrst appears in $C_{i}$ and $X_{m}$ ﬁrst appears in $C_{j}$ , for $i<j$ , then $X_{l}$ must precede $X_{m}$ in the ordering. We now construct a Bayesian network using the procedure Build-Minimal-I-Map of algorithm 3.2 applied to the resulting node ordering $X_{1},\dots,X_{n}$ and to $\mathcal{I}(\mathcal{H})$ . 
> 证明跳过

Let $\mathcal{G}$ be the result , when $X_{i}$ is added to the graph, then $X_{i}$ pa nts are precisely $U_{i}=\mathrm{Nb}_{X_{i}}\cap\{X_{1},.\,.\,.\,,X_{i-1}\}$ ∩{ − } , where $\operatorname{Nb}_{X_{i}}$ of $X_{i}$ H . In er words, we want to show that $X_{i}$ is independent o $\{X_{1},.\,.\,.\,,X_{i-1}\}-U_{i}$ − } $U_{i}$ . et $C_{k}$ be the ﬁrs que in the clique ordering to which X i belongs. Then $U_{i}\subset C_{k}$ ⊂ . Let $C_{l}$ be the parent of $C_{k}$ in the rooted clique tree. According to our selected ordering, all of the variables in $C_{l}$ are ordered before any variab $C_{k}-C_{l}$ $S_{l,k}\subset\{X_{1},.\.\,.\,,X_{i-1}\}$ . M ver, from our choice of ordering, none of { $\{X_{1},.\,.\,.\,,X_{i-1}\}-U_{i}$ − } − are in any descendants of $C_{k}$ in the clique tree. Thus, they are all in $W_{<(l,k)}$ . From theorem 4.12, it follows that $S_{l,k}$ separates $X_{i}$ from all of $\{X_{1},.\,.\,.\,,X_{i-1}\}-U_{i}$ , and ence that $X_{i}$ is independent of all of $\{X_{1},.\,.\,.\,,X_{i-1}\}-U_{i}$ given $U_{i}$ . It follows that $\mathcal{G}$ and H have e same set of edges. Moreover, e note that all of $U_{i}$ are in $C_{k}$ , and h nce are connected in G . Therefore, $\mathcal{G}$ is moralized. As H is the moralized undirected graph of G , the result now follows from proposition 4.9. 

For example, the graph $\mathcal{G}_{\ell}$ of ﬁgure 4.13b, and its moralized network $\mathcal{H}_{\ell}^{\prime}$ encode precisely the same independencies. By contrast, as we discussed, there exists no Bayesian network that encodes precisely the independencies in the nonchordal network $\mathcal{H}_{\ell}$ of ﬁgure 4.13a. 

Thus, we have shown that chordal graphs are precisely the intersection between Markov networks and Bayesian networks, in that the independencies in a graph can be represented exactly in both types of models if and only if the graph is chordal. 
> 弦图就是 Markov 网络和 Bayesian 网络之间的交集，当且仅当图是弦图，其独立性才可以精确被两种网络表示

## 4.6 Partially Directed Models 
So far, we have presented two distinct types of graphical models, based on directed and undirected graphs. We can unify both representations by allowing models that incorporate both directed and undirected dependencies. We begin by describing the notion of conditional random ﬁeld , a Markov network with a directed dependency on some subset of variables. We then present a generalization of this framework to the class of chain graphs , an entire network in which undirected components depend on each other in a directed fashion. 
> 考虑部分有向图，即有的边有向，有的边无向
> 例如条件随机场，CRF 是一个 Markov 网络，其中部分节点带有有向的依赖

### 4.6.1 Conditional Random Fields 
So far, we have described the Markov network representation as encoding a joint distribution over $\mathcal{X}$ . The same undirected gra tation a parameter iz ation can also be us to encode a conditional distribution $P(Y\mid X)$ | , where $Y$ is a set of target variables and X is a (disjoint) set of observed variables . We will also see a directed analogue of this concept in section 5.6. In the case of Markov networks, this representation is generally called a conditional random ﬁeld (CRF). 
> Markov 网络表示可以用于编码 $\mathcal X$ 上的联合分布，实际上相同的图表示和参数化也可以用于编码条件分布 $P (\pmb Y \mid \pmb X)$，其中 $\pmb Y$ 表示目标变量，$\pmb X$ 表示观察变量，这类 Markov 网络称为 CRF

#### 4.6.1.1 CRF Representation and Semantics 
More formally, a CRF is an undirected graph whose nodes correspond to $Y\cup X$ . At a high level, this graph is parameterized in the same way as an ordinary Markov network, as a set of factors $\phi_{1}(D_{1}),.\,.\,.\,,\phi_{m}(D_{m})$ . (As before, these factors can also be encoded more compactly as a log-linear model; for uniformity of presentation, we view the log-linear model as encoding a set of factors.) However, rather than encoding the distribution $P(Y,X)$ , we view it as representing the conditional distribution $P(Y\mid X)$ . To have the network structure and parameteriztion correspond naturally to a conditional distribution, we want to avoid representing a probabilistic model over $X$ . We therefore disallow potentials that involve only variables in $X$ . 
> CRF 是一个节点和 $\pmb Y \cup \pmb X$ 相对的无向图，该图参数化的方式和普通 Markov 网络相同，即用一组因子 $\phi_1 (\pmb D_1), \dots, \phi_m (\pmb D_m)$ 参数化，这组因子也可以用 log-linear 模型更紧凑地表示
> 差异在于我们将 CRF 编码的分布视作条件分布 $P (\pmb Y \mid \pmb X)$ 而不是联合分布 $P (\pmb Y, \pmb X)$，为此，我们需要避免在 $\pmb X$ 上表示概率模型，故不允许仅包含 $\pmb X$ 中的变量的势能函数

**Deﬁnition 4.18** conditional random ﬁeld 
$A$ conditional random ﬁeld is an undirected graph $\mathcal{H}$ whose nodes correspond to $X\cup Y$ ; the network is annotated with a set of factors $\phi_{1}(D_{1}),.\,.\,.\,,\phi_{m}(D_{m})$ such that each $D_{i}\nsubseteq X$ . The network encodes a conditional distribution as follows: 

$$
\begin{align}
P(\pmb Y \mid \pmb X) &= \frac {1}{Z(\pmb X)} \tilde P(\pmb Y, \pmb X)\\
\tilde P(\pmb Y, \pmb X)&= \prod_{i=1}^m \phi_i(\pmb D_i)\\
 Z(\pmb X)&=\sum_{\pmb Y}\tilde P(\pmb Y, \pmb X)
\end{align}\tag{4.11}
$$

Two variables in $\mathcal{H}$ are connected by an (undirected) edge whenever they appear together in the scope of some factor. 
> 定义：
> CRF 为一个无向图 $\mathcal H$，其节点对应于 $\pmb X \cup \pmb Y$，该网络由一组因子 $\phi_1 (\pmb D_1), \dots, \phi_m (\pmb D_m)$ 标注，其中 $\pmb D_i \not \subseteq \pmb X$
> 该网络如式 (4.11) 编码了条件概率分布 $P (\pmb Y \mid \pmb X)$
> 在 $\mathcal{H}$ 中，当且仅当两个变量同时出现在某个因子的作用范围内时，它们由一条无向边连接

> CRF 中，计算 $\tilde P (\pmb Y, \pmb X)$ 时不需要考虑仅属于 $\pmb X$ 的团，这些团可以视作编码了 $\pmb X$ 的先验，而因为 $\pmb X$ 总是给定，故不需要考虑 $\pmb X$ 自己的先验，或者可以认为 $\pmb X$ 取任意 $\pmb x$ 值的概率是相等的，此时只需要考虑它和 $\pmb Y$ 的交互的影响，因为就算考虑了，当 $\pmb X$ 取任意 $\pmb x$ 值的先验概率相等时， $\tilde P (\pmb Y, \pmb X)$ 和 $Z (\pmb X)$ 在任意的 $\pmb x$ 值下也只是同时乘上一个固定的常数，最后还是被消掉
> 事实上将 $\tilde P (\pmb Y, \pmb X)$ 直接视为一个未规范化的 $\pmb Y$ 条件于 $\pmb X$ 的条件分布会更加直观 (也就是写为 $\tilde P (\pmb Y \mid \pmb X)$)，因为它只考虑了 $\pmb Y$ 自己内部的交互和 $\pmb Y$ 和 $\pmb X$ 之间的交互，规范化常数是 $Z (\pmb X)$ 是条件于 $\pmb X$ 下 $\pmb Y$ 的所有可能取值的分数总和，因此它是和 $\pmb Y$ 无关的常数，因为 $\pmb X$ 是前提条件，故和 $\pmb X$ 有关，就写为 $Z (\pmb X)$

The only diference between equation (4.11) and the (unconditional) Gibbs distribution of deﬁnition 4.3 is the diferent normalization used in the partition function $Z(X)$ . The deﬁnition of a CRF induces a diferent value for the partition function for every assignment ${x}$ to $X$ . This diference is denoted graphically by having the feature variables grayed out. 
> 式 (4.11) 和定义4.3中对于 (无条件) Gibbs 分布的定义的差异仅在于在分区函数 $Z (\pmb X)$ 计算的不同，CRF 的定义为 $\pmb X$ 的不同赋值 $\pmb x$ 都引入一个不同的分区函数值
> 在图中，我们会将特征变量 $\pmb X$ 涂灰

![[Probabilistic Graph Theory-Fig4.14.png]]

Example 4.19 
Consider a CRF over $\pmb Y = \{Y_1, \dots, Y_k\}$ and $\pmb X = \{X_1, \dots, X_k\}$, with an edge $Y_i - Y_{i+1}$ ($i = 1, \dots, k-1$) and an edge $Y_i - X_i$  ($i = 1,\dots, k$), as shown in figure 4.14a. The distribution represented by this network has the form: 

$$
\begin{array}{r c l}{P(\boldsymbol{Y}\mid\boldsymbol{X})}&{=}&{\displaystyle\frac{1}{Z(\boldsymbol{X})}\tilde{P}(\boldsymbol{Y},\boldsymbol{X})}\\ {\tilde{P}(\boldsymbol{Y},\boldsymbol{X})}&{=}&{\displaystyle\prod_{i=1}^{k-1}\phi(Y_{i},Y_{i+1})\prod_{i=1}^{k}\phi(Y_{i},X_{i})}\\ {Z(\boldsymbol{X})}&{=}&{\displaystyle\sum_{\boldsymbol{Y}}\tilde{P}(\boldsymbol{Y},\boldsymbol{X}).}\end{array}
$$

Note that, unlike the deﬁnition of a conditional Bayesian network, the structure of a CRF may still contain edges between variables in $X$ , which arise when two such variables appear together in a factor that also contains a target variable. However, these edges do not encode the structure of any distribution over $X$ , since the network explicitly does not encode any such distribution. 
> 条件随机场和条件贝叶斯网络在图表示中不同的地方在于 CRF 的图表示中允许出现 $\pmb X$ 中的变量相连，这种情况在相连的 $X_i, X_j$ 都和 $\pmb Y$ 中的某个变量相连，并且一同处于一个团中
> 因为 CRF 的分布在定义上是不考虑 $\pmb X$ 之内的任意分布的，因此这条边并没有编码 $\pmb X$ 上的分布中的任意结构

**The fact that we avoid encoding the distribution over the variables in $X$ is one of the main strengths of the CRF representation. This ﬂexibility allows us to incorporate into the model a rich set of observed variables whose dependencies may be quite complex or even poorly understood. It also allows us to include continuous variables whose distribution may not have a simple parametric form. This ﬂexibility allows us to use domain knowledge in order to deﬁne a rich set of features characterizing our domain, without worrying about modeling their joint distribution.** 
>避免对变量集 $\pmb X$ 上的分布进行编码，这是 CRF 表示的主要优点之一，这种灵活性使我们能够将一组丰富的观测变量纳入模型，无论这些变量之间的依赖关系有多复杂和难以理解。此外，这也使我们能够引入那些分布可能没有简单参数形式的连续变量。这种灵活性允许我们利用领域知识来定义丰富的特征集，而不必担心对其联合分布进行建模

For example, returning to the vision MRFs of box 4.B, rather than deﬁning a joint distribution over pixel values and their region assignment, we can deﬁne a conditional distribution over segment assignments given the pixel values. The use of a conditional distribution here allows us to avoid making a parametric assumption over the (continuous) pixel values. Even more important, we can use image-processing routines to deﬁne rich features, such as the presence or direction of an image gradient at a pixel. Such features can be highly informative in determining the region assignment of a pixel. However, the deﬁnition of such features usually relies on multiple pixels, and deﬁning a correct joint distribution or a set of independence assumptions over these features is far from trivial. The fact that we can condition on these features and avoid this whole issue allows us the ﬂexibility to include them in the model. See box 4.E for another example. 
>举例来说，回到第4.B节中的视觉马尔可夫随机场（MRF），我们不必定义像素值及其区域分配的联合分布，而是可以在给定像素值的情况下定义区域分配的条件分布。这里使用条件分布使我们能够避免对（连续）像素值做出参数假设
>更重要的是，我们可以使用图像处理技术来定义丰富的特征，例如像素处的图像梯度的存在或方向。这些特征在确定像素的区域分配时是非常有信息量的，但定义这样的特征通常依赖于多个像素，在这些特征上定义正确的联合分布或一套独立性假设远非易事。而我们能够通过条件于这些特征来避免这一问题，从而赋予我们在模型中包含它们的灵活性。参见第4.E节中的另一个示例
>( 直观地说，就是建模 $P (\pmb Y \mid \pmb X)$ 比建模 $P (\pmb Y, \pmb X)$ 更简单，条件分布考虑了 $\pmb Y$ 和 $\pmb X$ 之间的交互、$\pmb Y$ 内部的交互，而不需要考虑 $\pmb X$ 内部的交互；条件分布的语义就是如此，$\pmb X$ 是给定的，我们仅关心它对 $\pmb Y$ 的影响如何，这或许和 $\pmb X$ 内部的交互有关，但它对外的表现就是 $\pmb X$ 和 $\pmb Y$ 之间的交互，故我们只需要关注后者就行 )

#### 4.6.1.2 Directed and Undirected Dependencies 
A CRF deﬁnes a conditional distribution of $Y$ on $X$ ; thus, it can be viewed as a partially directed graph, where we have an undirected component over $Y$ , which has the variables in $X$ as parents. 
> CRF 定义了 $\pmb Y$ 条件于 $\pmb X$ 的条件分布，因为引入了有向的条件依赖，因此它可以被视作部分有向图，图中对应 $\pmb Y$ 的成分是无向的，但它们有 $\pmb X$ 对应的节点作为父变量

Example 4.20 naive Markov 
Consider a CRF over the binary-valued variables $\pmb X\;=\;\{X_{1},.\,.\,.\,,X_{k}\}$ and ${\pmb Y}\,=\,\{{Y}\}$ , and a pairwise potential between $Y$ and each $X_{i}$ ; this model is sometimes known as a naive Markov model, due to its similarity to the naive Bayes model. Assume that the pairwise potentials deﬁned via the following log-linear model 

$$
\phi_{i}(X_{i},Y)=\exp\left\{w_{i}{\pmb 1}\{X_{i}=1,Y=1\}\right\}.
$$ 
We also introduce a single-node potential $\phi_{0}(Y)=\exp\left\{w_{0}{\pmb 1}\{Y=1\}\right\}$ . 
> 根据 HC 定理，朴素 Markov 模型的团势能函数定义如上

Following equation (4.11), we now have: 

$$
\begin{array}{c c l}{{\displaystyle\tilde{P}(Y=1\mid x_{1},\ldots,x_{k})}}&{{=}}&{{\displaystyle\exp\left\{w_{0}+\sum_{i=1}^{k}w_{i}x_{i}\right\}}}\\ {{\displaystyle\tilde{P}(Y=0\mid x_{1},\ldots,x_{k})}}&{{=}}&{{\displaystyle\exp\left\{0\right\}=1.}}\end{array}
$$ 


In this case, we can show (exercise 5.16) that 

$$
P(Y=1\mid x_{1},\dots,x_{k})=\mathrm{sigmoid}\left(w_{0}+\sum_{i=1}^{k}w_{i}x_{i}\right),
$$ 
where 

$$
\mathrm{sigmoid}(z)=\frac{e^{z}}{1+e^{z}}
$$ 
> 推导：
> 根据式 (4.11)

$$
\begin{align}
\tilde P(Y\mid \pmb X) &={\phi_0(Y)\prod_i\phi_i(X_i,Y)}\\
\end{align}
$$

> 因此

$$\begin{align}\tilde P(Y=1\mid \pmb X) &={\exp \{w_0\}\prod_i\exp\{w_ix_i\}}\\
&=\exp\left\{w_0 + \sum_i w_i x_i\right\}\\

\tilde P(Y=0\mid \pmb X) &= {\exp \{0\}\prod_i\exp\{0\}}=1
\end{align}
$$

> 再将具体的值代入 $P (Y=y \mid \pmb X) = P (Y = y \mid \pmb X) / \sum_Y P (Y \mid \pmb X)$ 计算即可

is the sigmoid function. This conditional distribution $P(Y\mid X)$ is of great practical interest: It deﬁnes a CPD that is not structured as a table, but that is induced by a small set of parameters $w_{0},\ldots,w_{k}$ — parameters whose number is linear, rather than exponential, in the number of parents. This type of CPD, often called a logistic CPD , is a natural model for many real-world applications, inasmuch as it naturally aggregates the inﬂuence of diferent parents. We discuss this CPD in greater detail in section 5.4.2 as part of our general presentation of structured CPDs. 

The partially directed model for the CRF of example 4.19 is shown in ﬁgure 4.14b. We may be tempted to believe that we can construct an equivalent model that is fully directed, such as the one in ﬁgure $4.14\mathrm{c}$ . In particular, conditioned on any assignment $\mathbf {x}$ , the posterior distributions over $Y$ in the two models satisfy the same independence assignments (the ones deﬁned by the chain structure). However, the two models are not equivalent: In the Bayesian network, we have that $Y_{1}$ is independent of $X_{2}$ if we are not given $Y_{2}$ . By contrast, in the original CRF, the unnormalized marginal measure of $Y$ depends on the entire parameter iz ation of the chain, and speciﬁcally the values of all of the variables in $X$ . A sound conditional Bayesian network for this distribution would require edges from all of the variables in $X$ to each of the variables $Y_{i}$ , thereby losing much of the structure in the distribution. See also box 20.A for further discussion. 
> Fig 4.14 中 b 图和 c 图的独立性语义不是等价的，c 图中的贝叶斯网络中，$Y_1$ 在不给定 $Y_2$ 时是和 $X_2$ 相互独立的，而 b 图的 CRF 中 $Y_i$ 是依赖于所有的 $X_i$ 的，因此如果要建模为一个条件贝叶斯网络，我们需要将所有的 $X_i$ 向 $Y_i$ 引入一条有向边，这会导致失去所有 CRF 编码的 $Y_i$ 之间的独立性关系

Box 4.E — Case Study: CRFs for Text Analysis. One important use for the CRF framework is in the domain of text analysis. Various models have been proposed for diferent tasks, including part-of-speech labeling, identifying named entities (people, places, organizations, and so forth), and extracting structured information from the text (for example, extracting from a reference list the publication titles, authors, journals, years, and the like). Most of these models share a similar structure: We have a target variable for each word (or perhaps short phrase) in the document, which encodes the possible labels for that word. Each target variable is connected to a set of feature variables that capture properties relevant to the target distinction. These methods are very popular in text analysis, both because the structure of the networks is a good ﬁt for this domain, and because they produce state-of-the-art results for a broad range of natural-language processing problems. 

As a concrete example, consider the named entity recognition task, as described by Sutton and McCallum (2004, 2007). Entities often span multiple words, and the type of an entity may not be apparent from individual words; for example, “New York” is a location, but “New York Times” is an organization. The problem of extracting entities from a word sequence of length $T$ can be cast as a graphical model by introducing for each word, $X_{t},1\leq t\leq T$ , a target variable, $Y_{t}$ , which indicates the entity type of the word. The outcomes of $Y_{t}$ include B-Person, I-Person, B-Location, I-Location, B-Organization, I-Organization , and Other . In this so-called “BIO notation,” Other indicates that the word is not part of an entity, the B- outcomes indicate the beginning of a named entity phrase, and the I- outcomes indicate the inside or end of the named entity phrase. Having a distinguishing label for the beginning versus inside of an entity phrase allows the model to segment adjacent entities of the same type. 

linear-chain CRF 

hidden Markov model 

A common structure for this problem is $a$ linear-chain CRF often having two factors for each word: one factor $\phi_{t}^{1}(Y_{t},Y_{t+1})$ to represent the dependency between neighboring target variables, and another factor $\phi_{t}^{2}(Y_{t},X_{1},\dots.,X_{T})$ that represents the dependency between a target and its context in the word sequence. Note that the second factor can depend on arbitrary features of the entire input word sequence. We generally do not encode this model using table factors, but using a log-linear model. Thus, the factors are derived from a number of feature functions, such as . We note that, just as logistic CPDs are the conditional analog of the naive Bayes classiﬁer (example 4.20), the linear-chain CRF is the conditional analog of the hidden Markov model (HMM) that we present in section 6.2.3.1. 

A large number of features of the word $X_{t}$ and neighboring words are relevant to the named entity decision. These include features of the word itself: is it capitalized; does it appear in a list of common person names; does it appear in an atlas of location names; does it end with the character string “ton”; is it exactly the string “York”; is the following word “Times.” Also relevant are aggregate features of the entire word sequence, such as whether it contains more than two sports-related words, which might be an indicator that “New York” is an organization (sports team) rather than a location. In addition, including features that are conjunctions of all these features often increases accuracy. The total number of features can be quite large, often in the hundreds of thousands or more if conjunctions of word pairs are used as features. However, the features are sparse, meaning that most features are zero for most words. 

Note that the same feature variable can be connected to multiple target variables, so that $Y_{t}$ would typically be dependent on the identity of several words in a window around position t . These contextual features are often highly indicative: for example, “Mrs.” before a word and “spoke” after a word are both strong indicators that the word is a person. These context words would generally be used as a feature for multiple target variables. Thus, if we were using a simple naive-Bayes-style generative model, where each target variable is a parent of its associated feature, we either would have to deal with the fact that a context word has multiple parents or we would have to duplicate its occurrences (with one copy for each target variable for which it is in the context), and thereby overcount its contribution. 

Linear-chain CRFs frequently provide per-token accuracies in the high 90 percent range on many natural data sets. Per-ﬁeld precision and recall (where the entire phrase category and boundaries must be correct) are more often around 80–95 percent, depending on the data set. 

skip-chain CRF 

Although the linear-chain model is often efective, additional information can be incorporated into the model by augmenting the graphical structure. For example, often when a word occurs multiple times in the same document, it often has the same label. This knowledge can be incor- porated by including factors that connect identical words, resulting in a skip-chain CRF , as shown in ﬁgure 4.E.1a. The ﬁrst occurrence of the word “Green” has neighboring words that provide strong  KEY  B-PER Begin person name I-LOC Within location name I-PER Within person name OTH Not an entitiy B-LOC Begin location name  (a)  $B$ Begin noun phrase V Verb $I$ Within noun phrase IN Preposition $O$ Not a noun phrase PRP Possesive pronoun $N$ Noun DT Determiner (e.g., a, an, the) $A D J$ Adjective 

Figure 4.E.1 — Two models for text analysis based on a linear chain CRF Gray nodes indicate $_{X}$ and clear nodes $\mathbf{Y}$ . The annotations inside the $\mathbf{Y}$ are the true labels. (a) A skip chain CRF for named entity recognition, with connections between adjacent words and long-range connections between multiple occurrences of the same word. (b) A pair of coupled linear-chain CRFs that performs joint part-of-speech labeling and noun-phrase segmentation. Here, B indicates the beginning of a noun phrase, I other words in the noun phrase, and O words not in a noun phrase. The labels for the second chain are parts of speech. 

evidence that it is a Person ’s name; however, the second occurrence is much more ambiguous. By augmenting the original linear-chain CRF with an additional long-range factor that prefers its connected target variables to have the same value, the model is more likely to predict correctly that the second occurrence is also a Person . This example demonstrates another ﬂexibility of conditional models, which is that the graphical structure over $Y$ can easily depend on the value of the $X{\dot{s}}.$ . 

coupled HMM 

CRFs having a wide variety of model structures have been successfully applied to many difer- ent tasks. Joint inference of both part-of-speech labels and noun-phrase segmentation has been performed with two connected linear chains (somewhat analogous to a coupled hidden Markov mode , shown in ﬁgure 6.3). This structure is illustrated in ﬁgure 4.E.1b. 

### 4.6.2 Chain Graph Models\* 
We now present a more general framework that builds on the CRF representation and can be used to provide a general treatment of the independence assumptions made in these partially directed models. Recall from deﬁnition 2.21 that, in a partially directed acyclic graph (PDAG), the nodes can be disjointly partitioned into several chain components . An edge between two nodes in the same chain component must be undirected, while an edge between two nodes in diferent chain components must be directed. Thus, PDAGs are also called chain graphs . 

#### 4.6.2.1 Factorization 

chain graph model 

As in our other graphical representations, the str ture of a PDAG $\mathcal{K}$ can be used to deﬁne a factorization for a probability distribution over K . Intuitively, the factorization for PDAGs represents the distribution as a product of each of the chain components given its parents. Thus, we call such a representation a chain graph model . 

Intuitively, each chain component $K_{i}$ in the chain graph model is associated with a CRF that deﬁnes $\mathcal{P}(\boldsymbol{K}_{i}\mid\mathrm{Pa}_{\boldsymbol{K}_{i}})$ — the conditional distribution of $K_{i}$ given its parents the graph. More precisely, each is deﬁned via a set of factors that involve the variables in $K_{i}$ and their parents; the distribution $\mathcal{P}(K_{i}\mid\mathrm{Pa}_{K_{i}})$ deﬁned by using the factors associa ith $K_{i}$ to deﬁne a CRF whose target variables are $K_{i}$ and whose observable variables are $\mathrm{Pa}_{K_{i}}$ . 

To provide a formal deﬁnition, it helps to introduce the concept of a moralized PDAG. 

Deﬁnition 4.19 moralized graph Let $\mathcal{K}$ be DAG and $K_{1},\dots,K_{\ell}$ b its chain components. We deﬁne $\mathrm{Pa}_{K_{i}}$ to be the parents of nodes in $K_{i}$ i . The moralized graph of K is an undirected graph $\mathcal M[\mathcal K]$ ed by ﬁrst connecting, using undirected edges, any pair of nodes $X,Y\in\mathrm{Pa}_{K_{i}}$ for all $i=1,\ldots,\ell,$ , and then converting all directed edges into undirected edges. 

This deﬁnition generalizes our earlier notion of a moralized directed graph. In the case of directed graphs, each node is its own chain component, and hence we are simply adding undirected edges between the parents of each node. 

Example 4.21 

Figure 4.15 A chain graph $\mathcal{K}$ and its moralized version 

$D$ and $H$ (even though $D$ and $C,E$ are in the same chain component), since $D$ is not a parent of $I$ . 

We can now deﬁne the factorization of a chain graph: 

Deﬁnition 4.20 chain graph distribution 

Let $\mathcal{K}$ be a PDAG, and $K_{1},\dots,K_{\ell}$ be its chain components. $A$ chain graph distribution is deﬁned via a set of factors $\phi_{i}(D_{i})$ $(i=1,.\ldots,m),$ , such that each $D_{i}$ is a complete subgraph in the moralized graph $\mathcal{M}[\mathcal{K}^{+}[D_{i}]]$ . We associate $\phi_{i}(D_{i})$ with a single chain component $K_{j}$ , $D_{i}\subseteq K_{i}\cup\mathrm{Pa}_{K_{i}}$ and deﬁne $P(K_{i}\mid\mathrm{Pa}_{K_{i}})$ | as a CRF with these factors, and with $Y_{i}=K_{i}$ and $X_{i}=\mathrm{Pa}_{K_{i}}$ . We now deﬁne 

$$
P(\mathcal{X})=\prod_{i=1}^{\ell}P(K_{i}\mid\mathrm{Pa}_{K_{i}}).
$$ 

We say that a d tribution $P$ factorizes over $\mathcal{K}$ if it can be represented as a chain graph distribution over . 

 Example 4.22 

In the chain graph model deﬁned by the graph of ﬁgure 4.15, we require that the conditional distribution $P(C,D,E\mid A,B)$ factorize according to the graph of ﬁgure 4.16a. Speciﬁcally, we would have to deﬁne the conditional probability as a normalized product of factors: 

$$
{\frac{1}{Z(A,B)}}\phi_{1}(A,C)\phi_{2}(B,E)\phi_{3}(C,D)\phi_{4}(D,E).
$$ 

#### 4.6.2.2 Independencies in Chain Graphs 
boundary As for undirected graphs, there are three distinct interpretations for the independence properties induced by a PDAG. Recall that in a PDAG, we have both the notion of parents of $X$ (variables $Y$ such that $Y\rightarrow X$ is in the graph) and neighbors of $X$ (variable $Y$ such that $Y{-}X$ is in the graph). Recall that the union of these two sets is the boundary of X , denoted Boundary $X$ . Also recall, from deﬁnition 2.15, that the descendants of $X$ are those nodes $Y$ that can be reached using any directed path, where a directed path can involve both directed and undirected edges but must contain at least one edge directed from $X$ to $Y$ , and no edges directed from $Y$ to 

 
Figure 4.16 Example for deﬁnition of c-separation in a chain graph. (a) The Markov network $\mathcal{M}[\mathcal{K}^{+}[C,D,E]]$ . (b) The Markov network $\mathcal{M}[\mathcal{K}^{+}[C,D,E,I]]$ . 

$X$ . Thus, in the case of PDAGs, it follows that if $Y$ is a descendant of $X$ , then $Y$ must be in a “lower” chain component. 

Deﬁnition 4.21 

pairwise independencies For a PDAG $\mathcal{K}$ , we deﬁne the pairwise independencies associated with $\mathcal{K}$ to be: 

$$
\begin{array}{c}{{\mathcal{Z}_{p}(\mathcal{K})=\{(X\perp Y\mid\mathrm{(NonDSedant3}_{X}-\{X,Y\}))\;:}}\\ {{X,Y\ n o n{\cdot}a d j a c e n t,Y\in\mathrm{NonDSedant3}_{X}\}.}}\end{array}
$$ 

This deﬁnition generalizes the pairwise independencies for undirected graphs: in an undirected graph, nodes have no descendants, so NonDescendants $\v{x}=\v{x}$ . Similarly, it is not too hard to show that these independencies also hold in a directed graph. 

Deﬁnition 4.22 

local independencies For a PDAG $\mathcal{K}$ , we deﬁne the local independencies associated with $\mathcal{K}$ to be: 

$$
{\mathcal{Z}}_{\ell}(X)=\{(X\;\bot\;{\mathrm{NonDS}}_{X}-{\mathrm{Boundary}}_{X}\;|\;{\mathrm{boundary}}_{X})\;:\;X\in{\mathcal{X}}\}.
$$ 

This deﬁnition generalizes the deﬁnition of local independencies for both directed and undi- rected graphs. For directed graphs, NonDescendants $X$ is precisely the set of nondescendants, whereas Boundary $X$ is the set of parents. For undirected graphs, NonDescendants $X$ is $\mathcal{X}$ , whereas Boundary $\v{U}_{X}=\mathrm{Nb}_{X}$ . 

We deﬁne the global independencies in a PDAG using the deﬁnition of moral graph. Our deﬁnition follows the lines of proposition 4.10. 

Deﬁnition 4.23 c-separation 

Example 4.23 

Let $X,Y,Z\subset\mathcal{X}$ three disjoint set nd let $U=X\cup Y\cup Z$ . We sa $X$ from $Y$ given Z if $X$ is separated from $Y$ given Z in the undirected graph M ${\mathcal{M}}[{\mathcal{K}}^{+}[X\cup Y\cup Z]].$ K ∪ ∪ 

Consider again the PDAG of ﬁgure 4.15. Then $C$ is $c$ -separated from $E$ given $D,A$ , because $C$ and $E$ are s arated ven $D,A$ in the ndirected gr h $\mathcal{M}[\mathcal{K}^{+}[\{C,D,E\}]]$ , shown ﬁgure .16a. - ever, C is not c-separated from E given only D , since there is $^a$ path between C and E via $A,B$ . On the other hand, $C$ is not separated from $E$ given $D,A,I$ . The graph $\mathcal{M}[\mathcal{K}^{+}[\{C,D,E,I\}]]$ is shown in ﬁgure 4.16b. As we can see, the introduction of I into the set $U$ causes us to introduce a direct edge between $C$ and $E$ in order to moralize the graph. Thus, we cannot block the path between $C$ and $E$ using $D,A,I$ . 

This notion of c-separation clearly generalizes the notion of separation in undirected graphs, since the ancestors of a set $U$ n an undirected graph are simply the entire set of nodes $\mathcal{X}$ . It also generalizes the notion of d-separation in directed graphs, using the equivalent deﬁnition provided in proposition 4.10. Using the deﬁnition of c-separation, we can ﬁnally deﬁne the notion of global Markov independencies: 

Deﬁnition 4.24 global independencies Let $\mathcal{K}$ be a PDAG. We deﬁne the global independencies associated with $\mathcal{K}$ to be: ${\mathcal{Z}}(\mathcal{K})=\{(X\perp Y\mid Z)\;:\;X,Y,Z\subset{\mathcal{X}},X$ is $c$ -separated from $Y$ given $Z\}$ . 

As in the case of undirected models, these three criteria for independence are not equivalent for nonpositive distributions. The inclusions are the same: the global independencies imply the local independencies, which in turn imply the pairwise independencies. Because undirected models are a subclass of PDAGs, the same counterexamples used in section 4.3.3 show that the inclusions are strict for nonpositive distributions. For positive distributions, we again have that the three deﬁnitions are equivalent. 

We note that, as in the case of Bayesian networks, the parents of a chain component are always fully connected in $\mathcal{M}[\mathcal{K}[K_{i}\cup\mathrm{Pa}_{K_{i}}]]$ . Thus, while the structure over the parents helps factorize the distribution over the chain components containing the parents, it does not give rise to independence assertions in the conditional distribution over the child chain component. Importantly, however, it does give rise to structure in the form of the parameter iz ation of $P(K_{i}\mid\mathrm{Pa}_{K_{i}})$ , as we saw in example 4.20. 

As in the case of directed and undirected models, we have an equivalence between the requirement of factorization of a distribution and the requirement that it satisfy the indepen- dencies associated with the graph. Not surprisingly, since PDAGs generalize undirected graphs, this equivalence only holds for positive distributions: 

Theorem 4.14 A positive distribution $P$ factorizes over a PDAG K if and only if $P\vDash\mathcal{Z}(\mathcal{K})$ I K . We omit the proof. 

## 4.7 Summary and Discussion 
In this chapter, we introduced Markov networks , an alternative graphical modeling language for probability distributions, based on undirected graphs. 
> 本章介绍了描述概率分布的无向图语言，也就是 Markov 网络

We showed that Markov networks, like Bayesian networks, can be viewed as deﬁning a set of independence assumptions determined by the graph structure. In the case of undirected models, there are several possible deﬁnitions for the independence assumptions induced by the graph, which are equivalent for positive distributions. As in the case of Bayesian network, we also showed that the graph can be viewed as a data structure for specifying a probability distribution in a factored form. The factorization is deﬁned as a product of factors (general nonnegative functions) over cliques in the graph. We showed that, for positive distributions, the two characterizations of undirected graphs — as specifying a set of independence assumptions and as deﬁning a factorization — are equivalent. 
> Markov 网络也可以被视为编码了由图结构定义的一组独立性假设，这和贝叶斯网络是类似的
> Markov 网络的多个对独立性的定义对于正分布是等价的
> Markov 网络也可以视为以分解的形式指定了一个概率分布的数据结构，这和贝叶斯网络也是类似的。Markov 网络的分解定义为在图中的 cliques 上的 (非负) 因子的乘积
>对于正分布，无向图对于分布的两种表征——指定一组独立性假设和指定一种分布形式——是等价的

Markov networks also provide useful insight on Bayesian networks. In particular, we showed how a Bayesian network can be viewed as a Gibbs distribution. More importantly, the unnormalized measure we obtain by introducing evidence into a Bayesian network is also a Gibbs distribution, whose partition function is the probability of the evidence. This observation will play a critical role in providing a uniﬁed view of inference in graphical models. 
> Markov 网络也帮助我们理解贝叶斯网络
> 例如，我们可以将贝叶斯网络也视为定义了 Gibbs 分布，向贝叶斯网络引入 evidence 得到的未规范化的分布也是 Gibbs 分布，其划分函数就是 evidence 的概率
> 这会帮助我们为图模型的推理提供一个统一视角

We investigated the relationship between Bayesian networks and Markov networks and showed that the two represent diferent families of independence assumptions. The diference in these independence assumptions is a key factor in deciding which of the two representations to use in encoding a particular domain. There are domains where interactions have a natural directionality, often derived from causal intuitions. In this case, the independencies derived from the network structure directly reﬂect patterns such as intercausal reasoning. Markov networks represent only monotonic independence patterns: observing a variable can only serve to remove dependencies, not to activate them. Of course, we can encode a distribution with “causal” connections as a Gibbs distribution, and it will exhibit the same nonmonotonic independencies. However, these independencies will not be manifest in the network structure. 
> 贝叶斯网络和 Markov 网络各自表示不同的一族独立性假设，这些独立性假设的差异是我们决定具体要使用哪种模型解决问题的关键
> 一些领域中，交互具有自然的有向性，例如因果推理，此时网络结构中的独立性就表示了因果关系；Markov 网络则仅表示单调的独立性模式：观察到一个变量仅能移除依赖性，而不能激活依赖性 (对比于贝叶斯网络中的 v-structure)
> 我们当然可以将带有因果关系的分布编码为 Gibbs 分布，该分布也会展示出相同的非单调独立性，但对应的网络结构就无法完美反映这一关系

In other domains, the interactions are more symmetrical, and attempts to force a directionality give rise to models that are unintuitive and that often are incapable of capturing the independencies in the domain (see, for example, section 6.6). As a consequence, the use of undirected models has increased steadily, most notably in ﬁelds such as computer vision and natural language processing, where the acyclicity requirements of directed graphical models are often at odds with the nature of the model. The ﬂexibility of the undirected model also allows the distribution to be decomposed into factors over multiple overlapping “features” without having to worry about deﬁning a single normalized generating distribution for each variable. Conversely, this very ﬂexibility and the associated lack of clear semantics for the model parameters often make it difcult to elicit models from experts. Therefore, many recent applications use learning techniques to estimate parameters from data, avoiding the need to provide a precise semantic meaning for each of them. 
> 一些领域中，交互是对称的，添加有向性会使得图结构无法捕获领域中的一些独立性 (菱形结构)
> 在 CV 和 NLP 领域使用较多的是无向图，这些领域中，有向图的无环要求有时会和模型的自然性质冲突，而无向图的灵活性较高
> 使用无向图建模时，我们可以定义多个交叉的“特征”作为因子，进而表示分布，使用有向图则需要为每个变量定义一个规范化的条件分布
> 但灵活性意味着模型的参数缺乏清晰语义，使得专家不便于定义模型参数，因此我们开始使用学习方法，从数据中学习和估计参数，不再为它们确定详细的语义含义

Finally, the question of which class of models better encodes the properties of the distribution is only one factor in the selection of a representation. There are other important distinctions between these two classes of models, especially when it comes to learning from data. We return to these topics later in the book (see, for example, box 20.A). 

# 5 Local Probabilistic Models 
In chapter 3 and chapter 4, we discussed the representation of global properties of independence by graphs. These properties of independence allowed us to factorize a high-dimensional joint distribution into a product of lower-dimensional CPDs or factors. So far, we have mostly ignored the representation of these factors. In this chapter, we examine CPDs in more detail. We describe a range of representations and consider their implications in terms of additional regularities we can exploit. We have chosen to phrase our discussion in terms of CPDs, since they are more constrained than factors (because of the local normalization constraints). However, many of the representations we discuss in the context of CPDs can also be applied to factors. 
> 我们之前讨论了将高维的联合分布分解为低维的 CPDs 或因子的乘积，本章讨论 CPDs 的细节
## 5.1 Tabular CPDs 
When dealing with spaces composed solely of discrete-valued random variables, we can always resort to a tabular representation of CPDs, w re w code $P(X~\mid\operatorname{Pa}_{X})$ as a table that contains an entry for each joint assignment to X and $\mathrm{Pa}_{X}$ . For this table to be a proper CPD, we require that all the values are nonnegative, and that, for each value $\operatorname{pa}_{X}$ , we have 
> 当处理仅仅由离散值随机变量构成的空间时，我们总是可以用表格表示 CPDs，表格中的各个项包含了 $P (X\mid \text{Pa}_X)$ 的所有可能值，同时，我们要求

$$
\sum_{x\in V a l(X)}{\cal P}(x\mid\mathrm{pa}_{X})=1.\tag{5.1}
$$ 
It is clear that this representation is as general as possible. We can represent every possible discrete CPD using such a table. 

As we will also see, table-CPDs can be used in a natural way in inference algorithms that we discuss in chapter 9. These advantages often lead to the perception that table-CPDs , also known as conditional probability tables (CPTs), are an inherent part of the Bayesian network representation. 
> table-CPDs，也可以称为条件概率表格 CPTs

However, the tabular representation also has several signiﬁcant disadvantages. First, it is clear that if we consider random variables with inﬁnite domains (for example, random variables with continuous values), we cannot store each possible conditional probability in a table. But even in the discrete setting, we encounter difculties. The number of parameters needed to describe a table-CPD is the number of joint assignments to $X$ and $\mathrm{Pa}_{X}$ , that is, $|\,V a l(\mathrm{Pa}_{X})|\cdot|\,V a l(X)|$ . This number grows exponentially in the number of parents. Thus, for example, if we have 5 binary parents of a binary variable $X$ , we need specify $2^{5}=32$ values; if we have 10 parents, we need to specify $2^{10}=1,024$ values. 
> table-CPDs 需要的参数数量是  $|\,V a l(\mathrm{Pa}_{X})|\cdot|\,V a l(X)|$，这个值是随着 parents 的数量而指数增长的（$|Val(\text{Pa}_X)|$ 的值随着 $\text{Pa}_X$ 的数量指数增长）

Clearly, the tabular representation rapidly becomes large and unwieldy as the number of parents grows. This problem is a serious one in many settings. Consider a medical domain where a symptom, Fever , depends on 10 diseases. It would be quite tiresome to ask our expert 1,024 questions of the format: “What is the probability of high fever when the patient has disease $A$ , does not have disease $B$ , . . . ?” Clearly, our expert will lose patience with us at some point!  This example illustrates another problem with the tabular representation: it ignores structure within the CPD. If the CPD is such that there are no similarity between the various cases, that is, each combination of disease has drastically diferent probability of high fever, then the expert might be more patient. However, in this example, like many others, there is some regularity in the parameters for diferent values of the parents of $X$ . For example, it might be the case that, if the patient sufers from disease $A$ , then sh is certain to have high fever and thus $P(X\mid\mathrm{{pa}}_{X})$ is the same for all values $\operatorname{pa}_{X}$ in which A is true. Indeed, many of the representations we consider in this chapter attempt to describe such regularities explicitly and to exploit them in order to reduce the number of parameters needed to specify a CPD. 
> table-CPDs 忽视了 CPD 内的结构
> 本章我们考虑的许多表示都尝试显式描述 CPD 内的 regularity 以减少表示 CPS 所需要的参数

The key insight that allows us to avoid these problems is the following observation: **A CPD needs to specify a conditional probability $P(x\mid\mathrm{pa}_{X})$ for every assignment of values $\mathrm{pa}_{X}$ and $x$ , but it does not have to do so by listing each such value explicitly. We should view CPDs not as tables listing all of the conditional probabilities, but rather as functions that given $\mathrm{pa}_{x}$ and $x$ , return the conditional probability $P(x\mid\mathrm{pa}_{X})$ .** This implicit representation sufces in order to specify a well-deﬁned joint distribution as a BN. In the remainder of the chapter, we will explore some of the possible representations of such functions. 
> CPD 需要为所有的 $\text{pa}_X$ 和 $x$ 的赋值指定出 $P (x \mid \text{pa}_X)$，但并不一定要列举出它们的所有值，我们可以将 CPD 看作一个函数，给定 $\text{pa}_x, x$，返回条件概率 $P (x\mid \text{pa}_X)$
## 5.2 Deterministic CPDs 
### 5.2.1 Representation 
Perhaps the simplest type of nontabular CPD arises when a variable $X$ is a deterministic function of its parents $\mathrm{Pa}_{X}$ . That is, there is a function $f:V a l(\mathrm{Pa}_{X})\mapsto V a l(X)$ , such that 

$$
P(x\mid\operatorname{pa}_{X})={\left\{\begin{array}{l l}{1}&{\qquad x=f(\operatorname{pa}_{X})}\\ {0}&{\qquad{\mathrm{otherwise.}}}\end{array}\right.}
$$ 
> 当变量 $X$ 是 $\text{Pa}_X$ 的确定性函数时，我们得到最简单的非表格 CPD，也就是当 $x = f (\text{pa}_x)$ 时，$P (x\mid \text{pa}_X)$ 为1，否则都为0

For example, in the case of binary-valued variables, $X$ might be the “or” of its parents. In a continuous domain, we might want to assert in $P(X\mid Y,Z)$ that $X$ is equal to $Y+Z$ . 
> 例如，对于二值变量，$X$ 是它的父变量的 “或”，对于连续变量，$X = Y + Z$

Of course, the extent to which this representation is more compact than a table (that is, takes less space in the computer) depends on the expressive power that our BN modeling language offers us for specifying deterministic functions. For example, some languages might allow a vocabulary that includes only logical OR and AND of the parents, so that all other functions must be speciﬁed explicitly as a table. In a domain with continuous variables, a language might choose to allow only linear dependencies of the form $X=2Y+-3Z+1$ , and not arbitrary functions such as $X=\sin(y+e^{z})$ . 
> 但这类表示的 compact 程序取决于 BN 建模语言可以为我们提供的用于指定确定性函数的表示能力
> 例如可能只能建模逻辑的 OR/AND，因此其他的函数就必须用表格指定；例如只能建模线性关系，而不能是三角函数

Deterministic relations are useful in modeling many domains. In some cases, they occur naturally. Most obviously, when modeling constructed artifacts such as machines or electronic circuits, deterministic dependencies are often part of the device speciﬁcation. For example, the behavior of an OR gate in an electronic circuit (in the case of no faults) is that the gate output is a deterministic OR of the gate inputs. However, we can also ﬁnd deterministic dependencies in “natural” domains. 
> 确定性依赖有时本身就是规格指定的一部分，例如电路的 OR 门，有时我们也可以在 “natural” domains 中找到确定性依赖

Example 5.1 
Recall that the genotype of a person is determined by two copies of each gene, called alleles . Each allele can take on one of several values corresponding to diferent genetic tendencies. The person’s phenotype is often a deterministic function of these values. For example, the gene responsible for determining blood type has three values: $a,\,b,$ , and o . Letting $G_{1}$ and $G_{2}$ be variables representing the two alleles, and $T$ the variable representing the phenotypical blood type, then we have that: 

$$
T=\left\{\begin{array}{l l}{a l}\\ {a}\\ {}\\ {b}\\ {}\\ {o}\end{array}\right.
$$ 

Deterministic variables can also help simplify the dependencies in a complex model. 
> 确定性变量也可以帮助简化复杂模型中的依赖

Example 5.2 
When modeling a car, we might have four variables $T_{1},\dots,T_{4}$ , each corresponding to a ﬂat in one of the four tires. When one or more of these tires is ﬂat, there are several efects; for example, the steering may be afected, the ride can be rougher, and so forth. Naively, we can make all of the $T_{i}$ ’s parents of all of the afected variables — Steering, Ride, and so on. However, it can signiﬁcantly simplify the model to introduce a new variable Flat-Tire, which is the deterministic OR of $T_{1},\dots,T_{4}$ . We can then replace a complex dependency of Steering and Ride on $T_{1},\dots,T_{4}$ with a dependency on a single parent Flat-Tire, signiﬁcantly reducing their indegree. If these variables have other parents, the savings can be considerable. 
> good example
### 5.2.2 Independencies 
Aside from a more compact representation, we get an additional advantage from making the structure explicit.
Recall that conditional independence is a numerical property — it is deﬁned using equality of probabilities. However, the graphical structure in a BN makes certain properties of a distribution explicit, allowing us to deduce that some independencies hold without looking at the numbers. 
By making structure explicit in the CPD, we can do even more of the same. 
> 除了表示会更 compact，结构也会更加清晰


![[Probabilistic Graph Theory-Fig5.1.png]]

Example 5.3 
Consider the simple network structure in ﬁgure 5.1. If $C$ is a deterministic function of $A$ and $B$ , what new conditional independencies do we have? Suppose that we are given the values of $A$ and $B$ . Then, since $C$ is deterministic, we also know the value of $C$ . As a consequence, we have that $D$ and $E$ a independent. Thus, we conclude tha $(D\perp E\mid A,B)$ holds in the distribution. Note that, had C not been a deterministic function of A and B , this independence would not necessarily hold. Indeed, $d$ -separation would not deduce that $D$ and $E$ are independent given $A$ and $B$ . 
> 例如，当 $C$ 是 $A, B$ 的确定性函数时，知道了 $A, B$，实际上也知道了 $C$，因此就知道了 $D\perp E$，也就是对于该分布，$(D\perp E\mid A, B)$ 成立

Can we augment the d-separation procedure to discover independencies in cases such as this? Consider an independence assertion $(X\ \bot\ Y\ |\ Z)$ in our example, we are interested in the case where $Z=\{A,B\}$ . The variable C is not in Z and is therefore not considered observed. 

But when $A$ and $B$ are observed, then the value of $C$ is also known with certainty, so we can consider it as part of our observed set $Z$ . In our example, this simple modiﬁcation would sufce for inferring that $D$ and $E$ are independent given $A$ and $B$ . 

![[Probabilistic Graph Theory-Algorithm5.1.png]]

In other examples, however, we might need to continue this process. For example, if we had another variable $F$ that was a deterministic function of $C$ , then $F$ is also de facto observed when $C$ is observed, and hence when $A$ and $B$ are observed. Thus, $F$ should also be introduced into $Z$ . Thus, we have to extend $Z$ iteratively to contain all the variables that are determined by it. This discussion suggests the simple procedure shown in algorithm 5.1. 
> 我们可以稍微修改我们的 d-seperation 算法，例如 $F$ 是 $C$ 的确定性函数时，在 $C$ 被观测到时，$F$ 也应该认为被观测到，因此需要被加入被观测到的变量集合 $\pmb Z$，那么 d-seperation 算法就可以在存在确定性函数的情况下也可以检测到变量之间的条件独立性了

This algorithm provides a procedural deﬁnition for *deterministic separation* of $X$ from $Y$ given $Z$ . This deﬁnition is sound, in the same sense that d-separation is sound. 
> 该算法提供了 $\pmb X, \pmb Y$ 在给定 $\pmb Z$ 的情况下的 deterministic seperation 的一个过程定义，这个定义是可靠的，和 d-seperation 的可靠性含义一样，也就是 deterministic seperated 的变量保证是条件独立的

**Theorem 5.1** 
Let $\mathcal{G}$ be a net k stru re, and let $D,X,Y,Z$ les. If $X$ is deterministica separated $Y$ en Z (as deﬁned by $\cdot_{\mathrm{SEP}}(\mathcal{G},D,X,Y,Z))$ ), then for all distributions P such that $P\vDash\mathcal{Z}_{\ell}(\mathcal{G})$ | I G and where, for each $X\in D$ ∈ , $P(X\mid\mathrm{Pa}_{X})$ | is a deterministic CPD, we have that $P\models(X\bot Y\mid Z)$ ⊥ | . 
> 定理：
> $\mathcal G$ 是图，$\pmb D, \pmb X, \pmb Y, \pmb Z$ 是变量集合，如果 $\pmb X$ 在给定 $\pmb Z$ 时是和 $\pmb Y$ determinstic seperated（记为 $\text{DET-SEP}(\mathcal G, \pmb D, \pmb X, \pmb Y, \pmb Z)$），
> 则对于所有满足 $P\vDash \mathcal I_{\mathscr l}(\mathcal G)$ ，并且满足对于所有 $X\in \pmb D$，$P (X\mid \text{Pa}_X)$ 是一个确定性的 CPD 的分布 $P$ (保证分布的 $\pmb Z^+$ 至少包含图的 $\pmb Z^+$ )，则我们有 $P\vDash (\pmb X \perp \pmb Y \mid \pmb Z)$

The proof is straightforward and is left as an exercise (exercise 5.1). 

Does this procedure capture all of the independencies implied by the deterministic functions? As with d-separation, the answer must be qualiﬁed: Given only the graph structure and the set  of deterministic CPDs, we cannot ﬁnd additional independencies. 
> 事实上，仅仅给定图结构和确定性 CPDs 的集合，我们是找不到额外的独立性的

**Theorem 5.2** 
Let $\mathcal{G}$ be a network structure, and let $D,X,Y,Z$ be sets of variables. I $\boldsymbol{\mathbf{\ell}}_{\mathrm{DFT-SEP}}(\mathcal{G},D,X,Y,Z)$ returns false, then there is a distribution P such that $P\,\vDash\mathcal{Z}_{\ell}(\mathcal{G})$ I G and where, for each $X\in D$ ∈ , $P(X\mid\mathrm{Pa}_{X})$ is deterministic CPD, but we have that $P\not\models(X\bot Y\mid Z)$ ⊥ | . 
> 定理：
> $\mathcal G$ 是图，$\pmb D, \pmb X, \pmb Y, \pmb Z$ 是变量集合，如果 $\text{DET-SEP}(\mathcal G, \pmb D, \pmb X, \pmb Y, \pmb Z)$ 返回 false（也就是给定 $\pmb Z$ 时，$\pmb X, \pmb Y$ 不是 determinstic seperated），则存在满足 $P\vDash \mathcal I_{\mathscr l}(\mathcal G)$ ，并且满足对于所有 $X\in \pmb D$，$P (X\mid \text{Pa}_X)$ 是一个确定性的 CPD 的分布 $P$，满足 $P\not\models (\pmb X \perp \pmb Y \mid \pmb Z)$ 

Of course, the det-sep procedure detects independencies that are derived purely from the fact that a variable is a deterministic function of its parents. However, particular deterministic functions can imply additional independencies. 
> 对于一个变量是其父变量的确定性函数这一事实，det-sep 算法可以检测到从中（朴素地）推演出的独立性，但一些特定的确定性函数会暗示额外的独立性，例如 XOR

Example 5.4 
Consider the network of ﬁgure 5.2, where $C$ is the exclusive or of $A$ and $B$ . What additional independencies do we have here? In the case of XOR (although not for all other deterministic functions), the values of $C$ and $B$ fully determine that of $A$ . Therefore, we have that $(D\perp E\mid$ $B,C)$ holds in the distribution. 
Speciﬁc deterministic functions can also induce other independencies, ones that are more reﬁned than the variable-level independencies discussed in chapter 3. 
> 特定的确定性函数还会引导出其他的比变量级独立性更加精细的独立性，例如 OR

Example 5.5 
Consider the Bayesian network of ﬁgure 5.1, but where we also know that the deterministic function at $C$ is an OR. Assume we are given the evidence $A=a^{1}$ . Because $C$ is an OR of its parents, we immediately know that $C=c^{1}$ , regardless of the value of $B$ . Thus, we can conclude that $B$ and $D$ are now independent: In other words, we have that 

$$
P(D\mid B,a^{1})=P(D\mid a^{1}).
$$ 
On the other hand, if we are given $A=a^{0}$ , the value of $C$ is not determined, and it does depend on the value of $B$ . Hence, the corresponding statement conditioned on $a^{0}$ is false. 

Thus, deterministic variables can induce a form of independence that is diferent from the standard notion on which we have focused so far. Up to now, we have restricted attention to independence properties of the form $(X\ \bot\ Y\ |\ Z)$ which represent the assumption that $P(X\mid Y,Z)=P(X\mid Z)$ for all values of X , Y $Y$ and Z . Deterministic functions can imply a type of independence that only holds for particular values of some variables. 
> 因此，确定性函数实际上可以引导出我们之前讨论的标准概念之外的独立性，我们一直讨论的是形式为 $P (X\perp Y \mid Z)$ 的独立性，其表示了 $P (X\mid Y, Z) = P (X\mid Z)$ 对于 $X, Y, Z$ 的所有值都是成立的
> 但确定性函数可以暗示这类独立性仅对一些变量的特定值成立

***Definition 5.1*** context-speciﬁc independence 
Let $X,Y,Z$ be pairwise disjoint sets of variables, let $C$ be a set of variables (that might overlap with $X\cup Y\cup Z)$ , and let $c\in\mathit{V a l}(C)$ . We say that $X$ and $Y$ are contextually independent given Z and the context $^c$ denoted $(X\perp_{c}Y\mid Z,c)$ , if 
> 定义：
> 令 $\pmb X, \pmb Y, \pmb Z$ 是成对的不相交变量集合，令 $\pmb C$ 是变量集合（可以和 $\pmb X\cup \pmb Y\cup \pmb Z$ 相交），令 $c \in Val (\pmb C)$，如果

$$
P(\pmb X\mid \pmb Y,\pmb Z,c) = P(\pmb X \mid \pmb Z,c)\;\text{whenever}\;P(\pmb Y, \pmb Z, c)>0
$$

> 则我们称 $\pmb X, \pmb Y$ 在给定 $\pmb Z$ 和上下文 $c$ 时，contextually independent，记作 $(\pmb X \perp_c \pmb Y\mid \pmb Z, c)$

Independence statements of this form are called context-speciﬁc independencies (CSI). They arise in many forms in the context of deterministic dependencies. 
> 这种类型的独立性声明称为针对上下文的独立性 CSI，它们在确定性函数的上下文中会以多种形式出现

Example 5.6 
As we saw in example 5.5, we can have that some value of one parent $A$ can be enough to determine the value of the child $C$ . Thus, we have that $(C\perp_{c}B\mid a^{1})$ , and hence also that $(D\perp_{c}B\mid a^{1})$ . We can make additional conclusions if we use properties of the OR function. For example, if we know that $C=c^{0}$ , we can conclude that both $A=a^{0}$ and $B=b^{0}$ . Thus, in particular, we can conc th that $(A\ \bot_{c}\ B\ |\ c^{0})$ at $(D\perp_{c}E\mid c^{0})$ . Similarly, if we know that $C=c^{1}$ and $B=b^{0}$ , we can conclude that $A=a^{1}$ , and hence we have that $(D\perp_{c}E\mid b^{0},c^{1})$ . 

It is important to note that **context-speciﬁc independencies can also arise when we have tabular CPDs. However, in the case of tabular CPDs, the independencies would only become apparent if we examine the network parameters. By making the structure of the CPD explicit, we can use qualitative arguments to deduce these independencies.** 
> 但注意针对上下文的独立性在表格式 CPDs 中也会出现，并且仅会在我们检查网络参数时才会 become apparent，而通过让 CPD 的结构清晰，我们就可以用定性的论证来推导这些独立性
## 5.3 Context-Specific CPDs 
### 5.3.1 Representation 
Structure in CPDs does not arise only in the case of deterministic dependencies. A very common type of regularity arises when we have precisely the same efect in several contexts. 
> CPDs 中的结构不仅仅在确定性依赖中出现
> 当我们在多个上下文中有完全相同的影响时，一个非常常见的 regularity 类型就会出现

![[Probabilistic Graph Theory-Fig5.3.png]]

Example 5.7 
We augment our Student example to model the event that the student will be oered a job at Acme Consulting. Thus, we have a binary-valued variable J, whose value is j1 if the student is oered this job, and j0 otherwise. The probability of this event depends on the student’s SAT scores and the strength of his recommendation letter. We also have to represent the fact that our student might choose not to apply for a job at Acme Consulting. Thus, we have a binary variable Applied, whose value $(a^{1}$ or $a^{0}$ ) indicates whether the student applied or not. The structure of the augmented network is shown in ﬁgure 5.3. 

Now, we need to describe the CPD $P(J\mid A,S,L)$ . In our domain, even if the student does not apply, there is still a chance that Acme Consulting is sufciently desperate for employees to ofer him a job anyway. (This phenomenon was quite common during the days of the Internet Gold Rush.) In this case, however, the recruiter has no access to the student’s SAT scores or recommendation letters, and therefore the decision to make an ofer cannot depend on these variables. Thus, among the 8 values of the parents $A,S,L,$ , the four that have $A=a^{0}$ must induce an identical distribution over the variable $J$ . 

We can elaborate this model even further. Assume that our recruiter, knowing that SAT scores are a far more reliable indicator of the student’s intelligence than a recommendation letter, ﬁrst considers the SAT score. If it is high, he generates an ofer immediately. (As we said, Acme Consulting is somewhat desperate for employees.) If, on the other hand, the SAT score is low, he goes to the efort of obtaining the professor’s letter of recommendation, and makes his decision accordingly. In this case, we have yet more regularity in the CPD: $P(J\mid a^{1},s^{1},l^{1})=P(J\mid a^{1},s^{1},l^{0})$ . 

In this simple example, we have a CPD in which several values of $\mathrm{Pa}_{J}$ specify the same conditional probability over $J$ . In general, we often have CPDs where, for certain partial assign- ments $\mathbfit{u}$ to subsets $U\subset\mathrm{Pa}_{X}$ , the values of the remaining parents are not relevant. In such cases, several diferent distributions $P(X\mid\mathrm{{pa}}_{X})$ are identical. In this section, we discuss how we might capture this regularity in our CPD representation and what implications this structure has on conditional independence. There are many possible approaches for capturing functions over a scope $X$ that are constant over certain subsets of instantiations to $X$ . In this section, we present two common and useful choices: trees and rules. 
> 本例中，$\text{Pa}_J$ 的多个赋值都指向了在 $J$ 上的相同的条件概率分布
> 一般情况下，对于 $U\subset \text{Pa}_j$ 的部分赋值 $\pmb u$ 可以让 $\text{Pa}_J$ 中剩余的父变量的值和 CPDs 的形式无关，此时，多个分布 $P (X\mid \text{Pa}_X)$ 实质上是完全相同的
> 本节讨论如何捕获这类 regularity
> 在集合 $\pmb X$ 上捕获对于 $\pmb X$ 的特定子集实例保持为常数的函数有两种通用的方法：树和规则
#### 5.3.1.1 Tree-CPDs 
A very natural representation for capturing common elements in a CPD is via a tree , where the leaves of the tree represent diferent possible (conditional) distributions over $J$ , and where the path to each leaf dictates the contexts in which this distribution is used. 
> 在 CPDs 捕获共同元素的非常自然的一个方法就是树
> 树中的叶子表示 $J$ 上不同的可能条件概率分布，而到每一个叶子的路径表示分布被使用时所在的上下文

![[Probabilistic Graph Theory-Fig5.4.png]]

Example 5.8
Figure 5.4 shows a tree for the CPD of the variable $J$ in example 5.7. Given this tree, we ﬁnd $P(J\mid A,S,L)$ by traversing the tree from the root downward. At each internal node, we see a test on one of the attributes. For example, in the root node of our tree we see a test on the value $A$ . We then follow the arc that is labeled with the value $a$ , which is given in the current setting of th s. Assume, for example, that we are interested in $P(J\mid a^{1},s^{1},l^{0})$ . Thus, w have that $A\,=\,a^{1}$ , and we would follow the right-hand arc labeled a $a^{1}$ . The next test is over S . We have $S=s^{1}$ , and we would also follow the right-hand arc. We have now reached a leaf, which is annotated with $a$ particular distribution over $J$ : $P(j^{1})=0.9$ , and $P(j^{0})=0.1$ . This distribution is the one we use for $P(J\mid a^{1},s^{1},l^{0})$ . 
> Fig5.4展示了 $J$ 的 CPD 的一个树表示，给定这个树表示，我们通过从根向下的遍历得到一个 $P (J\mid A, S, L)$
> 在每一个内部节点，我们对一个属性进行测试

Formally, we use the following recursive deﬁnition of trees. 

**Deﬁnition 5.2** tree-CPD 
A tree-CPD representing a CPD for variable $X$ is a rooted tree; each t-node in the tree is either $^a$ leaf t-node or an interior t-node . Each leaf is labeled with a distribution $P(X)$ . Each interior t-node is labeled with some variable $Z\in\mathrm{Pa}_{X}$ . Each interior t-node set of outgoing arcs to its children, ch one associated with a unique variable assignment $Z=z_{i}$ for $z_{i}\in V a l(Z)$ . A branch $\beta$ through a tree-CPD is a path beginning at the root and proceeding to a leaf node. We assume that no branch contains two interior nodes labeled by the same variable. The parent context induced by branch $\beta$ is the set of variable assignments $Z=z_{i}$ encountered on the arcs along the branch. 
> 定义：
> 一个用于表示变量 $X$ 的 CPD 的 tree-CPD 是一个有根的树，树中的每个 t-node 可以是叶子 t-node 或者是内部 t-node
> 树的每一个叶子都被一个 $P (X)$ 分布 labeled，树的每一个内部节点都被 $Z\in \text{Pa}_X$ 的某一个变量 labeled
> 每一个内部节点都有一系列向外的通向子节点的边，每个边都和一个独立的变量赋值 $Z = z_i\; \text{for}\; z_i \in Val (Z)$
>
> tree-CPD 的一个分支 $\beta$ 是一个从根节点开始，到叶节点结束的路径，我们假设没有分支包含两个由相同的变量 label 的内部节点
> 由分支 $\beta$ 导出的 parent context 是一个包含了分支中路径经过的边的变量赋值 $Z=z_i$ 的集合

Note that, to avoid confusion, we use t-nodes and arcs for a tree-CPD, as opposed to our use of nodes and edges as the terminology in a BN. 

Example 5.9 
Consider again the tree in ﬁgure 5.4. There are four branches in this tree. One induces the parent context $\langle a^{0}\rangle$ , corresponding to the situation where the student did not apply for the job. A second induces the parent context $\langle a^{1},s^{1}\rangle$ , corresponding to an application with a gh . The two branches induce complete assignments to all the parents of J : ⟨ $\langle a^{1},s^{0},l^{1}\rangle$ ⟩ and $\langle a^{1},s^{0},l^{0}\rangle$ ⟨ ⟩ . Thus, this representation breaks down the conditional distribution of J given its parents into four parent contexts by grouping the possible assignments in $V a l(\mathrm{Pa}_{J})$ into subsets that have the same efect on $J$ . Note that now we need only 4 parameters to describe the behavior of $J$ , instead of 8 in the table representation. 
> Fig5.4一共有4个 branch，每个 branch 导出/代表了一个 parent context
> 树表示通过将 $Val(\text{Pa}_j$) 中的可能赋值根据对 $J$ 的相同影响划分为了多个子集，以此将 $J$ 在给定它的父变量时的条件概率分布分解为了4个 parent context
> 此时我们只需要4个参数就可以表示 $J$ 的所有条件概率分布，而不是8个参数

Regularities of this type occur in many domains. Some events can occur only in certain situations. For example, we can have a Wet variable, denoting whether we get wet; that variable would depend on the Raining variable, but only in the context where we are outside. Another type of example arises in cases where we have a sensor, for example, a thermometer; in general, the thermometer depends on the temperature, but not if it is broken. 

This type of regularity is very common in cases where a variable can depend on one of a large set of variables: it depends only on one, but we have uncertainty about the choice of variable on which it depends. 
> 这种类型的 regularity 在许多领域都会出现
> 这种类型的 regularity 在一个变量依赖于许多个变量的情况中非常常见：该变量仅依赖于其中的一个变量，但我们不知道是哪一个

Example 5.10  

![[Probabilistic Graph Theory-Fig5.5.png]]


Let us revisit example 3.7, where George had to decide whether to give the recruiter at Acme Consulting the letter from his professor in Computer Science 101 or his professor in Computer Science 102. George’s chances of getting a job can depend on the quality of both letters L1 and L2, and hence both are parents. However, depending on which choice $C$ George makes, the dependence will only be on one of the two. Figure 5.5a shows the network fragment, and $b$ shows the tree-CPD for the variable $J$ . (For simplicity, we have eliminated the dependence on $S$ and $A$ that we had in ﬁgure 5.4.) 

More formally, we deﬁne the following: 

**Deﬁnition 5.3** multiplexer CPD 
A CPD $P(Y\mid A,Z_{1},.\,.\,.\,,Z_{k})$ is said to be $a$ multiplexer CPD if $V a l(A)=\{1,.\,.\,.\,,k\}$ , and 

$$
P(Y\mid a,Z_{1},.\,.\,.\,,Z_{k})=\pmb 1\{Y=Z_{a}\},
$$ 
where $a$ is the value of $A$ . The variable $A$ is called the selector variable for the CPD. 
> 定义：
> 一个 CPD $P (Y\mid A, Z_1,\dots, Z_k)$ 如果满足 $Val (A) = \{1,\dots, k\}$，并且

$$
P(Y\mid a, Z_1,\dots, Z_k) = \pmb 1\{Y = Z_a\}
$$

> 其中 $a$ 是 $A$ 的一个值，则称该 CPD 为 multiplexer CPD，变量 $A$ 称为 CPD 的选择器变量 (条件概率分布 $P (Y\mid a, Z_1,\dots, Z_k)$ 中不存在不确定性，随机变量 $Y$ 取值为 $Z_a$ 的概率是1，其他都为0)

In other words, the value of the multiplexer variable is a copy of the value of one of its parents, $Z_{1},\ldots,Z_{k}$ . The role of $A$ is to select the parent who is being copied. Thus, we can think of a multiplexer CPD as a switch. 
> 换句话说，一个 multiplexer 变量的值是其中一个它的父变量 $Z_1,\dots, Z_k$ 的值的拷贝，$A$ 的作用是选择出需要拷贝的是哪一个父变量，因此，我们可以将 multiplerxer CPD 看作是一个开关

We can apply this deﬁnition to example 5.10 by introducing a new variable $L_{i}$ , which is a multiplexer of $L_{1}$ and $L_{2}$ , using $C$ as the selector. The variable $J$ now depends directly only on $L$ . The modiﬁed network is shown in ﬁgure 5.5c. 
> 我们对于 example 5.10 应用该定义，引入一个 multiplexer 变量 $L$，$C$ 作为它的选择器，此时变量 $J$ 直接依赖于 $L$

This type of model arises in many settings. For example, it can arise when we have diferent actions in our model; it is often the case that the set of parents for a variables varies considerably based on the action taken. Conﬁguration variables also result in such situations: depending on the speciﬁc conﬁguration of a physical system, the interactions between variables might difer (see box 5.A). 
> 这类模型在许多设定下会出现，一般地来说，就是取决于特定的配置/动作，变量之间的交互关系会变化

Another setting where this type of model is particularly useful is in dealing with uncertainty about correspondence between diferent objects. This problem arises, for example, in data association , where we obtain sensor measurements about real-world objects, but we are uncertain about which object gave rise to which sensor measurement. For example, we might get a blip on a radar screen without knowing which of of several airplanes the source of the signal. Such cases also arise in robotics (see box 15.A and box 19.D). Similar situations also arise in other applications, such as identity resolution : associating names mentioned in text to the real- correspondence variable world objects to which they refer (see box 6.D). We can model this type of situation using a correspondence variable $U$ that associates, with each sensor measurement, the identity $u$ of the object that gave rise to the measurement. The actual sensor measurement is then deﬁned using a multiplexer CPD that depends on the correspondence variable $U$ (which plays the role of the selector variable), and on the value of $A(u)$ for all $u$ from which the measurement could have been derived. The value of the measurement will be the value of $A(u)$ for $U=u$ , usually with some added noise due to measurement error. Box 12.D describes this problem in more detail and presents algorithms for dealing with the difcult inference problem it entails. 
> 这类模型的另一个设定在处理不同对象之间的对应性 correspondence 的不确定性时非常有用
> 这类问题可以出现在例如数据关联 data association 的场景中、标识识别 identity resolution 的场景中
> 对于这类场景，我们可以使用一个 correspondence 变量 $U$ 表示导致观察到的现象的父变量（原因），然后定义一个 multiplexer CPD，以 $U$ 作为选择器变量

Trees provide a very natural framework for representing context-speciﬁcity in the CPD. In particular, it turns out that people ﬁnd it very convenient to represent this type of structure using trees. Furthermore, the tree representation lends itself very well to automated learning algorithms that construct a tree automatically from a data set. 
> 树提供了一个表示 CPD 中的 contexte-specificity 的一个很自然的框架

Box 5.A — Case Study: Context-Speciﬁcity in Diagnostic Networks. A common setting where context-speciﬁc CPDs arise is in troubleshooting of physical systems, as described, for example, by Heckerman, Breese, and Rommelse (1995). In such networks, the context speciﬁcity is due to the presence of alternative conﬁgurations. For example, consider a network for diagnosis of faults in $^a$ printer, developed as part of a suite of troubleshooting networks for Microsoft’s Windows $95^{T M}$ op- erating system. This network, shown in ﬁgure 5.A.1a, models the fact that the printer can be hooked up either to the network via an Ethernet cable or to a local computer via a cable, and therefore depends on both the status of the local transport medium and the network transport medium. However, the status of the Ethernet cable only afects the printer’s output if the printer is hooked up to the network. The tree-CPD for the variable Printer-Output is shown in ﬁgure 5.A.1b. Even in this very simple network, this use of local structure in the CPD reduced the number of parameters required from 145 to 55. 

We return to the topic of Bayesian networks for troubleshooting in box 21.C and box 23.C. 
#### 5.3.1.2 Rule CPDs 
As we seen, trees are appealing for several reasons. However, trees are a global representation that captures the entire CPD in a single data structure. In many cases, it is easier to reason using a CPD if we break down the dependency structure into ﬁner-grained elements. A ﬁner- grained representation of context-speciﬁc dependencies is via rules . Roughly speaking, each rule corresponds to a single entry in the CPD of the variable. It speciﬁes a context in which the CPD entry applies and its numerical value. 
> 树是一个在单个数据结构内捕获整个 CPD 的全局表示
> 在许多情况下，我们需要将 CPD 的依赖结构分解为更细粒度的元素，而要对 context-specific 的依赖进行更细粒度的表示，就需要借助规则
> 每个规则都对应于变量的 CPD 中的单个 entry，它指定了 CPD 的上下文，以及它的数值

**Deﬁnition 5.4** rule scope 
$A$ rule $\rho$ i $a$ pair $\langle c;p\rangle$ where $^c$ is an assignment to some subset of variables $C$ , and $p\in[0,1]$ . We deﬁne C to be the scope of $\rho,$ , denoted Scope [ ρ ] . 
> 定义：
> 规则 $\rho$ 定义为一个 pair $\langle \pmb c; p\rangle$，其中 $\pmb c$ 是一个对某个变量子集 $\pmb C$ 的一个赋值，$p\in [0,1]$ ，我们定义 $\pmb C$ 是规则 $\rho$ 的作用域，记为 $\text{Scope}[\rho]$

This representation decomposes a tree-CPD into its most basic elements. 

Example 5.11 
Consider the tree of ﬁgure 5.4. There are eight entries in the CPD tree, such that each one corresponds to a branch in the tree and an assignment to the variable $J$ itself. Thus, the CPD deﬁnes eight rules: 

$$
\left\{\begin{array}{l l}{\rho_{1}\colon\langle a^{0},j^{0};0,8\rangle}\\ {\rho_{2}\colon\langle a^{0},j^{1};0,2\rangle}\\ {\rho_{3}\colon\langle a^{1},s^{0},j^{0};0,9\rangle}\\ {\rho_{4}\colon\langle a^{1},s^{0},l^{0},j^{1};0,1\rangle}\\ {\rho_{5}\colon\langle a^{1},s^{0},l^{1},j^{0};0,4\rangle}\\ {\rho_{6}\colon\langle a^{1},s^{0},l^{1},j^{1};0,6\rangle}\\ {\rho_{7}\colon\langle a^{1},s^{1},j^{0};0,1\rangle}\\ {\rho_{8}\colon\langle a^{1},s^{1},j^{1};0,9\rangle}\end{array}\right\}
$$ 

For example, the rule $\rho_{4}$ is derived by following the branch $a^{1},s^{0},l^{0}$ and then selecting the probability associated with the assignment $J=j^{1}$ . 

Although we can decompose any tree-CPD into its constituent rules, we wish to deﬁne rule- based CPDs as an independent notion. To deﬁne a coherent CPD from a set of rules, we need to make sure that each conditional distribution of the form $P(X\mid\mathrm{{pa}}_{X})$ is speciﬁed by precisely one rule. Thus, the rules in a CPD must be mutually exclusive and exhaustive. 
> 我们在上个例子中展示了我们可以将 tree-CPD 分解为组成它的多个规则
> 我们希望直接定义 rule-based CPD，为了从一个规则集合中定义 CPD，我们需要保证每个形式为 $P (X\mid \text{pa}_X)$ 都可以刚刚好被一个规则指定，因此，CPD 中的规则需要是互斥且完备的

**Definition 5.5**  rule-based CPD 
A rule-based CPD $P(X\mid\mathrm{Pa}_{X})$ is a set of rules $\mathcal{R}$ such that: 

• For each rule $\rho\in\mathcal R$ , we have that $Scope [ ρ |\subseteq\{X\}\cup\mathrm{Pa}_{X}$ . 
• For each assignment $(x,u)$ to $\{X\}\cup\mathrm{Pa}_{X}$ , we have precisely one rule $\langle c;p\rangle\in{\mathcal{R}}$ such that c is compatible with $(x,u)$ . In this case, we say that $P(X=x\mid\mathrm{Pa}_{X}={\pmb u})=p.$ . 
• The resulting CPD $P(X\mid U)$ is a legal CPD, in that $\sum_{x}P(x\mid u)=1.$ 

> 一个基于规则的 CPD $P (X\mid \text{Pa}_X)$ 是一个规则集合 $\mathcal R$，满足：
> - 对于每个规则 $\rho \in \mathcal R$，我们有 $\text{Scpoe}[\rho]\subseteq \{X\}\cup \text{Pa}_X$
> - 对于每个对 $\{X\}\cup \text{Pa}_X$ 的赋值 $(x, u)$，我们会恰好有一个规则 $\langle \pmb c, p\rangle \in \mathcal R$ ，满足 $\pmb c$ 和 $(x, \pmb u)$ 一致，此时，我们称 $P (X = x\mid \text{Pa}_X = \pmb u) = p$
> - 得到的 CPD $P (X\mid \pmb U)$ 满足 $\sum_x P (x\mid \pmb u) = 1$ 

The rule set in example 5.11 satisﬁes these conditions. Consider the following, more complex, example. 

Example 5.12 
Let $X$ be a variable with $\mathrm{Pa}_{X}=\{A,B,C\}$ , and assume that X ’s CPD is deﬁned via the following set of rules: 

This set of rules deﬁnes the following CPD: 

For example, the CPD entry $P(x^{0}\mid a^{0},b^{1},c^{1})$ is determined by the rule $\rho_{3}$ the CPD entry 0 . 2 ; we can verify that no other rule is compatible with the context $a^{0},b^{1},c^{1},x^{1}$ . We can also verify that each of the CPD entries is also compatible with precisely one context, and hence that the diferent contexts are mutually exclusive and exhaustive. 

Note that both CPD entries $P(x^{1}\mid a^{0},b^{1},c^{0})$ and $P(x^{0}\mid a^{0},b^{1},c^{0})$ are determined by $^a$ single rule $\rho_{9}$ . As the probabilities for the diferent contexts in this case must sum up to 1, this phenomenon is only possible when the rule deﬁnes a uniform distribution, as it does in this case. 

This perspective views rules as a decomposition of a CPD. We can also view a rule as a ﬁner-grained factorization of an entire distribution. 

**Proposition 5.1**
Let $\mathcal{B}$ be a Bayesian ne ork, and assume that each CPD $P(X\mid\mathrm{Pa}_{X})$ n $\mathcal{B}$ is represented as a set of rules $\mathcal{R}_{X}$ . Let R be the multiset deﬁned as $\uplus_{X\in\mathcal{X}}\mathcal{R}_{X}$ , where ⊎ denotes multiset join, which puts together all of the rule instances (including duplicates). Then, the probability of any instantiation $\xi$ to the network variables $\mathcal{X}$ can be computed as 

$$
P(\xi)=\prod_{\langle \pmb c;p\rangle\in{\mathcal R},\xi\sim \pmb c}p.
$$ 
The proof is left as an exercise (exercise 5.3). 
> 定理：
> 令 $\mathcal B$ 表示一个贝叶斯网络，假设 $\mathcal B$ 中的每一个 CPD 都可以被一个规则集合 $\mathcal R_X$ 表示
> 令 $\mathcal R$ 表示定义为 $\uplus_{X\in\mathcal{X}}\mathcal{R}_{X}$ 的多重集合，其中 $\uplus$ 表示多重集合的并集运算，因此 $\mathcal R$ 就是将所有的规则都放在一起的多重集合（包括重复）
> 则对于网络变量 $\mathcal X$ 的任意实例化 $\xi$ 的概率可以计算为 $P (\xi)=\prod_{\langle c; p\rangle\in{\mathcal R},\xi\sim c}p.$

The rule representation is more than a simple transformation of tree-CPDs. In particular, although every tree-CPDs can be represented compactly as a set of rules, the converse does not necessarily hold: not every rule-based CPD can be represented compactly as a tree. 
> 规则表示并不是 tree-CPDs 的简单转换，特别地，即便每个 tree-CPDs 都可以被表示为一个紧凑的规则集合，但不是每个规则集合都可以被表示为一个紧凑的 tree-CPD

Example 5.13 
Consider the rule-based CPD of example 5.12. In any rule set that is derived from a tree, one variable — the one at the root — appears in all rules. In the rule set $\mathcal{R}$ , none of the parent variables $A,B,C$ appears in all rules, and hence the rule set is not derived from a tree. If we try to represent it as a tree-CPD, we would have to select one of $A$ , $B$ , or $C$ to be the root. Say, for example, that we select A to be the root. In this case, rules that do not contain $A$ would necessarily correspond to more than one branch (one for $a^{1}$ and one for $a^{0}$ ). Thus, the transformation would result in more branches than rules. For example, ﬁgure 5.6 shows a minimal tree-CPD that represents the rule-based CPD of example 5.12. 
#### 5.3.1.3 Other Representations 
The tree and rule representations provide two possibilities for representing context-speciﬁc structure. We have focused on these two approaches as they have been demonstrated to be useful for representation, for inference, or for learning. However, other representations are also possible, and can also be used for these tasks. In general, if we abstract away from the details of these representations, we see that they both simply induce partitions of $V a l(\{X\}\cup\mathrm{Pa}_{X})$ , deﬁned by the branches in the tree on one hand or the rule contexts on the other. Each partition is associated with a diferent entry in $X$ ’s CPD. 
> 如果我们对这些表示进行抽象，我们可以认为这些表示都是简单地引入了对 $Val (\{X\}\cup \text{Pa}_X$) 的划分，在树中，它被定义为分支，在规则集中，它被定义为规则，每一个划分都对应于 $X$ 的 CPD 的一个 entry

This perspective allows us to understand the strengths and limitations of the diferent rep- resentations. In both trees and rules, all the partitions are described via an assignment to a subset of the variables. Thus, for example, we cannot represent the partition that contains only $a^{1},s^{1},l^{0}$ and $a^{1},s^{0},l^{1}$ , a partition that we might obtain if the recruiter lumped together candidates that had a high SAT score or a strong recommendation letter, but not both. As deﬁned, these representations also require that we either split on a variable (within a branch of the tree or within a rule) or ignore it entirely. In particular, this restriction does not allow us to capture dependencies that utilize a taxonomic hierarchy on some parent attribute, as described in box 5.B 
> 在树和规则中，划分都是通过对所有变量的一个子集的赋值得到的
> 定义上，这类表示要求我们要么在一个变量上做出划分 split，要么完全忽略它，这个限制导致我们不能捕获在一些 parent attribute 中使用了 taxonomic 层次的依赖

Of course, we can still represent distributions with these properties by simply having multiple tree branches or multiple rules that are associated with the same parameter iz ation. However, this solution both is less compact and fails to capture some aspects of the structure of the CPD. A very ﬂexible representation, that allows these structures, might use general logical formulas to describe partitions. This representation is very ﬂexible, and it can precisely capture any partition we might consider; however, the formulas might get fairly complex. Somewhat more restrictive is the use of a decision diagram , which allows diferent t-nodes in a tree to share children, avoiding duplication of subtrees where possible. This representation is more general than trees, in that any structure that can be represented compactly as a tree can be represented compactly as a decision diagram, but the converse does not hold. Decision diagrams are incomparable to rules, in that there are examples where each is more compact than the other. In general, diferent representations ofer diferent trade-ofs and might be appropriate for diferent applications. 
> 但是对于具有这些性质的分布，我们当然也可以用具有多个有相同参数的分支的树或者规则来表示，但显然不够 compact，并且不能捕获 CPD 中的一些结构

Box 5.B — Concept: Multinets and Similarity Networks. The multinet representation provides a more global approach to capturing context-speciﬁc independence. In its simple form, a multinet is a network centered on a single distinguished class variable $C$ , which is a root of the network. The multinet deﬁnes a separate network $\mathcal{B}_{c}$ for each value of $C$ , where the structure as well as the parameters can difer for these diferent networks. In most cases, a multinet deﬁnes a single network where every variable $X$ has as its parents $C$ , and all variables $Y$ in any of the networks $\mathcal{B}_{c}$ . However, the CPD of $X$ is such th in context $C\,=\,c_{i}$ , it depends y on $\dot{\mathrm{Pa}}_{X}^{\mathcal{B}_{c}}$ . In some cases, however, a subtlety arises, where Y $Y$ is a parent of X in $\mathcal{B}_{c^{1}}$ , and X is a parent of $Y$ in $\mathcal{B}_{c^{2}}$ . In this case, the Bayesian network induced by the multinet is cyclic; nevertheless, because of the context-speciﬁc independence properties of this network, it speciﬁes a coherent distribution. (See also exercise 5.2.) Although, in most cases, a multinet can be represented as a standard BN with context-speciﬁc CPDs, it is nevertheless useful, since it explicitly shows the independencies in $^a$ graphical form, making them easier to understand and elicit. 

A related representation, the similarity network , was developed as part of the Pathﬁnder system (see box $3.D$ ). In a similarity network, we deﬁne a network $\mathcal{B}_{S}$ for certain subsets of values $S\subset V a l(C)$ , which contains only those attributes re ant for distinguishing between the values in S . The underlying assumption is th variable X d not appear in e network $\mathcal{B}_{S}$ , then $P(X\mid C=c)$ is the same for all c $c\in S$ ∈ . More er, if X not ha $Y$ as a parent in this network, then $X$ is contextually independent of Y given C $C\,\in\,S$ ∈ and $X\mathit{\dot{s}}$ ’s ot pa nts in this network. A similarity network easily captures structure where the dependence of X on C is deﬁned in terms of a taxonomic hierarchy on $C$ . For example, we might have that our class variable is Disease. While Sore-throat depends on Disease, it does not have a diferent conditional distribution for every value $d$ of Disease. For example, we might partition diseases into diseases that do not cause sore throat and those that do, and the latter might be further split into difuse disease (causing soreness throughout the throat) and localized diseases (such as abscesses). Using this partition, we might have only three diferent conditional distributions for $P$ ( Sore-Throat $\mid D i s e a s e=d_{\mid}$ ) . Multinets facilitate elicitation both by focusing the expert’s attention on attributes that matter, and by reducing the number of distinct probabilities that must be elicited. 
### 5.3.2 Independencies 
In many of our preceding examples, we used phrases such as “In the case $a^{0}$ , where the student does not apply, the recruiter’s decision cannot depend on the variables $S$ and $L$ .” These phrases suggest that context-speciﬁc CPDs induce context-speciﬁc independence. In this section, we analyze the independencies induced by context-speciﬁc dependency models. 
> 针对上下文的 CPDs 会引入针对上下文的独立性，本节对其进行讨论

Consider a CPD $P(X\mid\mathrm{Pa}_{X})$ , where certain distributions over $X$ are shared across diferent instantiations of $\mathrm{Pa}_{X}$ . The structure of such a CPD allows us to infer certain independencies *locally* without having to consider any global aspects of the network. 
> 考虑一个 CPD $P (X\mid \text{Pa}_X)$，其中在 $X$ 上的特定分布被多个不同的 $\text{Pa}_X$ 实例共享
> 这类 CPD 的结构允许我们在不需要考虑网络全局的情况下局部地推理出特定独立性

Example 5.14 
Returning to example 5.7, we can see tha $(J\ \perp_{c}\ S,L\ \vert\ \ a^{0})$ : By the deﬁnition of the CPD, $P(J\mid a^{0},s,l)$ is the same for all values of s and l . Note that this equality holds regardless of the structure or the parameters of the network in which this CPD is embedded. Similarly, we have that $\left(J\perp_{c}L\mid a^{1},s^{1}\right)$ . 

In general, if we deﬁne $c$ to be the context associated with a branch in the tree-CPD for $X$ , then $X$ is independent of the remaining parents $(\mathrm{Pa}_{X}-S c o p e[c])$ given the context $c$ . However, there might be additional CSI statements that we can determine locally, conditioned on contexts that are not induced by complete branches. 
> 一般地，如果我们将 $\pmb c$ 定义为和 $X$ 的 tree-CPD 的一个分支相关的上下文，则 $X$ 在给定上下文 $\pmb c$ 的情况下，就和其余的 parents: $(\text{Pa}_X - Scope[\pmb c])$ 独立
> 但也存在我们根据局部的上下文就可以决定的 CSI statement，也就是不需要条件于完整的分支导出的上下文

Example 5.15 
Consider, the tree-CPD of ﬁgure $5.5b$ . Here, once George chooses to request a letter from one professor, his job prospects still depend on the quality of that professor’s letter, but not on that of the other. More precisely, we have that $(J\perp_{c}L_{2}\mid c^{1})$ ; note that $c^{1}$ is not the full assignment associated with a branch. 

Example 5.16 
More interestingly, consider again the tree of figure 5.4, and suppose we are given the context s1. Clearly, we should only consider branches that are consistent with this value. There are two such branches. One associated with the assignment $a^{0}$ and the other with the assignment $a^{1},s^{1}$ . We can immediately see that the choice between these two branches does not depend on the value of $L$ . Thus, we conclude that $(J\perp_{c}L\mid s^{1})$ holds in this case. 

We can generalize this line of reasoning by considering the rules compatible with a particular context $^c$ . Intuitively, if none of these rules mentions a particular parent $Y$ of $X$ , then $X$ is conditionally independent of $Y$ given $^c$ . More generally, we can deﬁne the notion of conditioning a rule on a context: 
> 考虑和某个特定的上下文 $\pmb c$ 相兼容的规则，直观地，如果其中没有规则提到 $X$ 的一个特定的 $Y$，则 $X$ 在给定 $\pmb c$ 的情况下条件独立于 $Y$

**Deﬁnition 5.6**  reduced rule 
Let $\rho\;=\;\left\langle c^{\prime};p\right\rangle$ be a rule and $C\,=\,c$ be a context. If $c^{\prime}$ is compatible ith $c$ , we say that $\rho\sim c$ . In this case, let $c^{\prime\prime}=c^{\prime}\langle S c o p e[c^{\prime}]-S c o p e[c]\rangle$ be the assignment in $c^{\prime}$ to the variables in $S c o p e[c^{\prime}]\mathrm{~-~}S c o p e[c]$ . We then deﬁne the reduced rule $\rho[{\pmb c}]\,=\,\langle{\pmb c}^{\prime\prime};p\rangle$ . For R a set of rules, we deﬁne the reduced rule set 

$$
{\mathcal{R}}[{\pmb{c}}]=\{\rho[{\pmb{c}}]\ :\ \rho\in{\mathcal{R}},\rho\sim{\pmb{c}}\}.
$$ 

> 定义：
> 令 $\rho = \langle \pmb c'; p\rangle$ 为一个规则，$\pmb C = \pmb c$ 为上下文，如果 $\pmb c'$ 和 $\pmb c$ 兼容，则我们说 $\rho \sim \pmb c$，此时，令 $\pmb c'' = \pmb c'\langle \text{Scope}[\pmb c']-\text{Scpoe}[\pmb c]\rangle$，也就是 $\pmb c'$ 对于处于 $\text{Scope}[\pmb c']-\text{Scope}[\pmb c]$ 中的变量的赋值
> 对于规则集合 $\mathcal R$，我们定义简化的规则 $\rho[\pmb c]=\langle \pmb c''; p\rangle$ 集合如上

Example 5.17 
In the rule set $\mathcal{R}$ of example 5.12, $\mathcal{R}[a^{1}]$ is the set 

$$
\begin{array}{r l}{\rho_{1}^{\prime}\colon\langle b^{1},x^{0};0.1\rangle}&{\qquad\rho_{2}\colon\langle b^{1},x^{1};0.9\rangle}\\ {\rho_{5}\colon\langle b^{0},c^{0},x^{0};0.3\rangle}&{\qquad\rho_{6}\colon\langle b^{0},c^{0},x^{1};0.7\rangle}\\ {\rho_{7}^{\prime}\colon\langle b^{0},c^{1},x^{0};0.4\rangle}&{\qquad\rho_{8}^{\prime}\colon\langle b^{0},c^{1},x^{1};0.6\rangle.}\end{array}
$$ 

Thus, we have left only the rules compatible with $a^{1}$ , and eliminated $a^{1}$ from the context in the rules where it appeared. 
> 我们从全部规则集合中过滤出仅和 $a^1$ 兼容的规则，并且将 $a^1$ 从过滤出的规则中的上下文中去除

**Proposition 5.2** 
Let $\mathcal{R}$ be the rul s in t ed CPD for a variable $X$ , and t $\mathcal{R}_{c}$ be the rules in $\mathcal{R}$ that are compatible with c . Let $Y\subseteq\operatorname{Pa}_{X}$ ⊆ be some subset of parents of X such that $Y\cap S c o p e[c]=\emptyset$ . If for every $\rho\in\mathcal{R}[c]$ , we have that $Y\cap S c o p e[\rho]=\emptyset$ , then $(X\perp_{c}Y\mid\mathrm{Pa}_{X}-Y,c)$ . 
> 引理：
> 令 $\mathcal R$ 是变量 $X$ 的 rule-based CPD 的规则集合，令 $\mathcal R_{\pmb c}$ 表示其中和 $\pmb c$ 兼容的规则，令 $Y\subseteq \text{Pa}_X$ 是 $X$ 的父变量的子集，满足 $Y\cap Scope[\pmb c] = \emptyset$
> 如果对于所有 $\rho \in \mathcal R[\pmb c]$，都满足 $Y\cap Scope[\rho] = \emptyset$，则满足 $(X\perp_{\pmb c} \pmb Y \mid \text{Pa}_X - \pmb Y, \pmb c)$

The proof is left as an exercise (exercise 5.4). 

This proposition speciﬁes a computational tool for deducing “local” CSI relations from the rule representation. We can check whether a variable $Y$ is being tested in the reduced rule set given a context in linear time in the number of rules. (See also exercise 5.6 for a similar procedure for trees.) 

This procedure, however, is incomplete in two ways. First, since the procedure does not examine the actual parameter values, it can miss additional independencies that are true for the speciﬁc parameter assignments. However, as in the case of completeness for d-separation in BNs, this violation only occurs in degenerate cases. (See exercise 5.7.) 

The more severe limitation of this procedure is that it only tests for independencies between $X$ and some of its parents given a context and the other parents. Are there are other, more global, implications of such CSI relations? 

Example 5.18 
Can we capture this intuition formally? Consider the dependence structure in the context $A=a^{0}$ . Intuitively, in this context, the edges $S\rightarrow J$ and $L\rightarrow J$ are both redundant, since we know that $(J\perp_{c}S,L\mid a^{0})$ . Thus, our intuition is that we should check for d-separation in the graph without this edge. Indeed, we can show that this is a sound check for CSI conditions. 

Deﬁnition 5.7 spurious edge 
$P(X~\mid\mathrm{Pa}_{X})$ be a CPD, let $Y\,\,\in\,\,\mathrm{Pa}_{X}$ , and let $^c$ be a context. We say that the edge $Y\rightarrow X$ urious in the contex $^c$ if $P(X\mid\mathrm{Pa}_{X})$ $(X\perp_{c}Y\mid\mathrm{Pa}_{X}-\{Y\},c^{\prime})$ , where $c^{\prime}=c\langle\mathrm{Pa}_{X}\rangle$ ⟨ ⟩ is the restriction of c to variables in $\mathrm{Pa}_{X}$ . 

If we represent CPDs with rules, then we can determine whether an edge is spurious by examin- ing the reduced rule s . L $\mathcal{R}$ be the rule-based CPD for $P(X\mid\mathrm{Pa}_{X})$ , then the edge $Y\rightarrow X$ is spurious in context c if Y does not appear in the reduced rule set $\mathcal{R}[c]$ . 


CSI-separation 
Now we can deﬁne CSI-separation , a variant of $\mathrm{d}$ -separation that takes CSI into account. This notion, deﬁned procedurally in algorithm 5.2, is straightforward: we use local considerations to remove spurious edges and then apply standard d-separation to the resulting graph. We say that 

$X$ is CSI-separated from $Y$ given $Z$ in the context $^c$ if CSI- $\cdot_{\mathrm{SEP}}(\mathcal{G},c,X,Y,Z)$ returns true . As an example, consider the network of example 5.7, in the context $A=a^{0}$ . In this case, we get that the arcs $S\rightarrow J$ and $L\rightarrow J$ are spurious, leading to the reduc grap in ﬁgure $5.7\mathrm{a}$ . As we can see, J and I are d-separated in the reduced graph, as are J and D . Thus, using CSI-sep , we get that $I$ and $J$ are $\mathrm{d}$ -separated given the context $a^{0}$ . Figure 5.7b shows the reduced graph in the context $s^{1}$ . 

It is not hard to show that CSI-separation provides a sound test for determining context- speciﬁc independence. 

Let $\mathcal{G}$ etwork structure, let $P$ b distribution such that $P\vDash\mathcal{Z}_{\ell}(\mathcal{G})$ I let $^c$ be a cont t, and let $X,Y,Z$ be sets of variables. If X is CSI-separated from Y given Z in the context c , then $P\models(X\ \bot_{c}Y\mid Z,c)$ . 

The proof is left as an exercise (exercise 5.8). 

Of course, we also want to know if CSI-separation is complete — that is, whether it discovers all the context-speciﬁc independencies in the distribution. At best, we can hope for the same type of qualiﬁed completeness that we had before: discovering all CSI assertions that are a direct consequence of the structural properties of the model, regardless of the particular choice of parameters. In this case, the structural properties consist of the graph structure (as usual) and the structure of the rule sets or trees. Unfortunately, even this weak notion of completeness does not hold in this case. 

Example 5.19 Consider the example of ﬁgure $5.5b$ and e con $C=c^{1}$ . I this context, the arc $L_{2}\,\rightarrow\,J$ is spurious. Thus, there is no path between $L_{1}$ and $L_{2}$ , even given J . Hence, CSI-sep will report that $L_{1}$ and $L_{2}$ are $d$ -separated given $J$ and the context $C\,=\,c^{1}$ . This case is shown in ﬁgure $5.8a$ . , we conclude that $\left(L_{1}\;\;\perp_{c}\;\;L_{2}\;\;|\;\;J,c^{1}\right)$ . Similarly, in the context $C\ =\ c^{2}$ , the arc $L_{1}\to J$ s, and we have that $L_{1}$ and $L_{2}$ are $d$ -separated given $J$ and $c^{2}$ , and hen that $(L_{1}\perp_{c}L_{2}\mid J,c^{2})$ ⊥ Thus, reason ng by cases, we nclude that once e of C , we have that $L_{1}$ and $L_{2}$ are always d-separated given J , and hence that $\left(L_{1}\perp L_{2}\mid J,C\right)$ ⊥ | . Can we get this conclusion using CSI-separation? Unfortunately, the answer is no. If we invoke CSI-separation with the empty context, then no edges are spurious and CSI-separation reduces to $d$ -separation. Since both $L_{1}$ and $L_{2}$ are parents of $J$ , we conclude that they are not separated given $J$ and $C$ . 

The problem here is that CSI-separation does not perform reasoning by cases. Of course, if we want to determine whether $X$ and $Y$ are independent given $Z$ and a context $^{c,}$ we can invoke CSI-separation on the context $c,z$ for each possible value of $Z$ , and see if $X$ and $Y$ are separated in all of these contexts. This procedure, however, is exponential in the number of variables of $Z$ . Thus, it is practical only for small evidence sets. Can we do better than reasoning by cases? The answer is that sometimes we cannot. See exercise 5.10 for a more detailed examination of this issue. 
## 5.4 Independence of Causal Inﬂuence 
In this section, we describe a very diferent type of structure in the local probability model. Consider a variable $Y$ whose distribution depends on some set of causes $X_{1},\ldots,X_{k}$ . In general, $Y$ can depend on its parents in arbitrary ways — the $X_{i}$ can interact with each other in complex ways, making the efect of each combination of values unrelated to any other combination. However, in many cases, the combined inﬂuence of the $X_{i}$ ’s on $Y$ is a simple combination of the inﬂuence of each of the $X_{i}$ ’s on $Y$ in isolation. In other words, each of the $X_{i}$ ’s inﬂuences $Y$ independently, and the inﬂuence of several of them is simply combined in some way. 
> 本节描述一个局部概率模型中一个非常不同的结构
> 考虑一个变量 $Y$ ，其分布依赖于某个集合 $X_1,\dots, X_k$，一般地 $Y$ 可以以任意方式依赖于其父变量，$X_i$ 之间可以以复杂的方式交互，使得不同的值组合的效果互不相同
> 但在许多情况下，$X_i$ 的结合对 $Y$ 的影响只是单独每个 $X_i$ 对于 $Y$ 的影响的结合

We begin by describing two very useful models of this type — the noisy-or model, and the class of generalized linear models . We then provide a general deﬁnition for this type of interaction. 
### 5.4.1 The Noisy-Or Model 
Let us begin by considering an example in which a diferent professor writes a recommendation letter for a student. Unlike our earlier example, this professor teaches a small seminar class, where she gets to know every student. The quality of her letter depends on two things: whether the student participated in class, for example, by asking good questions $(Q)$ ; and whether he wrote a good ﬁnal paper $(F)$ . Roughly speaking, each of these events is enough to cause the professor to write a good letter. However, the professor might fail to remember the student’s participation. On the other hand, she might not have been able to read the student’s handwriting, and hence may not appreciate the quality of his ﬁnal paper. Thus, there is some noise in the process. 

Let us consider each of the two causes in isolation. Assume that $P(l^{1}\;\mid\;q^{1},f^{0})\:=\:0.8,$ , that is, the professor is 80 percent likely to remember class participation. On the other hand, $P(l^{1}\mid q^{0},\bar{f^{1}})=0.9$ , that is, the student’s handwriting is readable in 90 percent of the cases. What happens if both occur: the student participates in class and writes a good ﬁnal paper? The key assumption is that these are two independent *causal mechanisms* for causing a strong letter, and that the letter is weak only if neither of them succeeded. The ﬁrst causal mechanism — class participation $q^{1}$ — fails with probability 0 . 2 . The second mechanism — a good ﬁnal paper $f^{1}\gets$ fails with probability 0 . 1 . If both $q^{1}$ and $f^{1}$ occurred, the probability that both mechanisms fail (independently) is $0.2\cdot0.1=0.02$ . Thus, we have that $\bar{P}(l^{0}\mid q^{1},f^{1})=0.02$  and $P(l^{1}\mid q^{1},f^{1})=0.98$ . In other words, our CPD for $P(L\mid Q,F)$ is: 
> 本例中，考虑了两个独立的因果机制，第一个因果机制 fail 的概率是0.2，第二个因果机制 fail 的概率是0.1，如果二者同时发生，则 fail 的概率应该是 $0.2*0.1 = 0.02$

This type of interaction between causes is called the noisy-or model. Note that we assumed that a student cannot end up with a strong letter if he neither participated in class nor wrote a good ﬁnal paper. We relax this assumption later on. 
> 这类 cause 之间的交互称为噪声或模型

![[Probabilistic Graph Theory-Fig5.9.png]]

An alternative way of understanding this interaction is by assuming that the letter-writing process can be represented by a more elaborate probabilistic model, as shown in ﬁgure 5.9. This ﬁgure represents the conditional distribution for the Letter variable given Questions and FinalPaper . It also uses two intermediate variables that reveal the associated causal mechanisms. The variable $Q^{\prime}$ is true if the professor remembers the student’s participation; the variable $F^{\prime}$ is true if the professor could read and appreciate the student’s high-quality ﬁnal paper. The letter is strong if and only if one of these events holds. We can verify that the conditional distribution $P(L\mid Q,F)$ induced by this netw k is precisely the one shown before. 

The probability that $Q$ causes L ( 0.8 in this example) is called the noise parameter , and denoted $\lambda_{Q}$ $\lambda_{Q}={\bar{P}}(q^{\prime1}\mid q^{1})$ . Similarly, we have a noise parameter $\lambda_{F}$ , which in this context is $\lambda_{F}=P(f^{\prime1}\mid f^{1})$ . 
> Fig5.9中，$Q$ 能够导致 $L$ 的概率称为噪声参数，记作 $\lambda_Q$，在 Fig5.9的分解下，我们可以知道 $\lambda_Q = P (q'^1 \mid q^1)$
> 类似地，我们有噪声参数 $\lambda_F = P (f'^1 \mid f^1)$
> 噪声参数编码了单独影响的情况下， cause 能导致结果的实际概率

We can also incorporate a *leak probabilit*y that represents the probability — say $0.0001\mathrm{~-~}$ that the professor would write a good recommendation letter for no good reason, simply because she is having a good day. We simply introduce another variable into the network to represent this event. This variable has no parents, and is true with probability $\lambda_{0}=0.0001$ . It is also a parent of the Letter variable, which remains a deterministic or. 
> 我们还引入泄露概率的概念，它表示无论如何 $L$ 都会成功的概率，例如 $0.00001$
> 我们为网络引入另一个变量来表示这一事件，该变量没有父变量，并且为真的概率是 $0.00001$，同时该变量是变量 $L$ 的父变量，关系是 determinstic or

The decomposition of this CPD clearly shows why this local probability model is called a noisy-or. The basic interaction of the efect with its causes is that of an OR, but there is some noise in the “efective value” of each cause. We can deﬁne this model in the more general setting: 
> 在 noisy-or 局部概率模型中，causes 之间的基本的交互就是或关系，并且在每个 cause 的“有效值”中存在一些噪音

Deﬁnition 5.8 noisy-or CPD
Let Y be a binary-valued random variable with k binary-valued parents $X_1,\dots ,X_k$. The CPD $P (Y \ X_1,\dots, X_k)$ is a noisy-or if there are k + 1 noise parameters $\lambda_0,\dots,\lambda_k$ such that noisy-or CPD
> 定义：
> 令 $Y$ 是二值随机变量，且有 $k$ 个二值的父变量 $X_1,\dots, X_k$
> 如果存在 $k+1$ 个噪声参数 $\lambda_0,\dots,\lambda_k$，使得

$$
\begin{align}
P(y^0\mid X_1,\dots,X_k) &= (1-\lambda_0)\prod_{i:X_i = x_i^1}(1-\lambda_i)\tag{5.2}\\
P(y^1\mid X_1,\dots,X_k)&= 1-[(1-\lambda_0)\prod_{i:X_i = x_i^1}(1-\lambda_i)]
\end{align}
$$

> 则CPD $P (Y\mid X_1,\dots, X_k)$ 称为一个噪声或

We note that, if we interpret $x_{i}^{1}$ as $1$ and $x_{i}^{0}$ as $0$ , we can rewrite equation (5.2) somewhat more compactly as: 

$$
P(y^{0}\mid x_{1},.\,.\,.,x_{k})=(1-\lambda_{0})\prod_{i=1}^{k}(1-\lambda_{i})^{x_{i}}.\tag{5.3}
$$ 
Although this transformation might seem cumbersome, it will turn out to be very useful in a variety of settings. 

![[Probabilistic Graph Theory-Fig5.10.png]]

Figure 5.10 shows a graph of the behavior of a special-case noisy or model, where all the variables have the same noise parameter $\lambda$ . The graph shows the probability of the child $Y$ in terms of the parameter $\lambda$ and the number of $X_{i}$ ’s that have the value true . 
> Fig5.10的情况对应于所有的变量都有相同的噪声参数 $\lambda$

The noisy-or model is applicable in a wide variety of settings, but perhaps the most obvious is in the medical domain. For example, as we discussed earlier, a symptom variable such as Fever usually has a very large number of parents, corresponding to diferent diseases that can cause the symptom. However, it is often a reasonable approximation to assume that the diferent diseases use diferent causal mechanisms, and that if any disease succeeds in activating its mechanism, the symptom is present. Hence, the noisy-or model is a reasonable approximation. 

Box 5.C — Concept: BN2O Networks. A class of networks that has received some attention in BN2O network the domain of medical diagnosis is the class of BN2O networks. 
> 在医疗诊断领域，BN2O 网络也十分常用

A BN2O network, illustrated in figure 5.C.1, is a two-layer Bayesian network, where the top layer corresponds to a set of causes, such as diseases, and the second to findings that might indicate these causes, such as symptoms or test results. All variables are binary-valued, and the variables in the second layer all have noisy-or models. 
> BN2O 网络是一个两层的贝叶斯网络，其中顶层对应于 causes，下一层则表示结果，所有的变量都是二值变量，且第二层的变量都有噪声或模型

Speciﬁcally, the CPD of $F_{i}$ is given by: 

$$
P(f_{i}^{0}\mid\mathrm{Pa}_{F_{i}})\;\;=\;\;(1-\lambda_{i,0})\prod_{D_{j}\in\mathrm{Pa}_{F_{i}}}(1-\lambda_{i,j})^{d_{j}}.
$$ 
These networks are conceptually very simple and require a small number of easy-to-understand parameters: Each edge denotes a causal association between a cause $d_{i}$ and a ﬁnding $f_{j}$ ; each is associated with a parameter $\lambda_{i,j}$ that encodes the probability that $d_{i}$ , in isolation, causes $f_{j}$ to manifest. Thus, these networks resemble a simple set of noisy rules, a similarity that greatly facilitates the knowledge-elicitation task. Although simple, BN2O networks are a reasonable ﬁrst approximation for a medical diagnosis network. 

BN2O networks also have another useful property. In Bayesian networks, observing a variable generally induces a correlation between all of its parents. In medical diagnosis networks, where ﬁndings can be caused by a large number of diseases, this phenomenon might lead to signiﬁcant complexity, both cognitively and in terms of inference. However, in medical diagnosis, most of the ﬁndings in any speciﬁc case are false — a patient generally only has a small handful of symptoms. As discussed in section 5.4.4, the parents of a noisy-or variable $F$ are conditionally independent given that we observe that $F$ is false. As a consequence, a BN2O network where we observe $F=f^{0}$ is equivalent to a network where $F$ disappears from the network entirely (see exercise 5.13). This observation can greatly reduce the cost of inference. 
> 贝叶斯网络中，观察到一个变量通常会引入该变量和它所有的父变量之间的关联
> 但噪声或变量 $F$ 的父变量在给定我们观察到 $F$ 为 false 时是条件独立的，因此在 BN2O 网络中，观察到 $F=f^0$ 等价于 $F$ 完全从网络中消失的网络，因此可以大大减少推理开销
### 5.4.2 Generalized Linear Models 
An apparently very diferent class of models that also satisfy independence of causal inﬂuence are the generalized linear models . Although there are many models of this type, in this section we focus on models that deﬁne probability distributions $P(Y\mid X_{1},.\,.\,,X_{k})$ where $Y$ takes on values in some discrete ﬁnite space. We ﬁrst discuss the case where Y and all of the $X_{i}$ ’s are binary-valued. We then extend the model to deal with the multinomial case. 
> 另一类也满足因果影响之间的独立性的模型是广义线性模型
> 我们讨论定义了概率分布 $P (Y\mid X_1,\dots, X_k)$ 的广义线性模型，其中 $Y$ 取离散值
#### 5.4.2.1 Binary-Valued Variables
Roughly speaking, our models in this case are a soft version of a linear threshold function. As a motivating example, we can think of applying this model in a medical setting: In practice, our body’s immune system is constantly ﬁghting of multiple invaders. Each of them adds to the burden, with some adding more than others. We can imagine that when the total burden passes some threshold, we begin to exhibit a fever and other symptoms of infection. That is, as the total burden increases, the probability of fever increases. This requires us to clarify two terms in this discussion. The ﬁrst is the “total burden” value and how it depends on the particular possible disease causes. The second is a speciﬁcation of how the probability of fever depends on the total burden. 

More generally, we examine a CPD of $Y$ given $X_{1},\ldots,X_{k}$ . We assume that the efect of the $X_{i}$ ’s on $Y$ can summari via a linear function $\begin{array}{r}{f(X_{1},\dots,X_{k})=\sum_{i=1}^{k}w_{i}X_{i}}\end{array}$ , where we again interpret $x_{i}^{1}$ as 1 and $x_{i}^{0}$ as 0 . In our example, this function will be the total burden on the immune system, and the $w_{i}$ coefcient describes how much burden is contributed by each disease cause. 
> 考虑给定 $X_1,\dots, X_k$ 时 $Y$ 的条件概率分布，我们假设 $X_i$  对于 $Y$ 的影响可以通过一个线性函数总结 $f (X_1,\dots, X_k) = \sum_{i=1}^k w_i X_i$，其中我们用 $x_i^1$ 表示 1，用 $x_i^0$ 表示 0
> 这可以认为是 $X_1,\dots, X_k$ 对于 $Y$ 的总贡献值，其中 $w_i$ 表示了各个 cause 贡献值的权重

The next question is how the probability of $Y~=~y^{1}$ depends on $f(X_{1},\cdot\cdot\cdot,X_{k})$ . In general, this probability undergoes a phase transition around some threshold value $\tau$ : when $f(X_{1},.\,.\,.\,,X_{k})\,\geq\,\tau$ , then $Y$ is very likely to be 1; when $f(X_{1},.\,.\,.\,,X_{k})\,<\tau$ , then $Y$ is very likely to 0. It is easier to inate $\tau$ by simply deﬁning $\begin{array}{r}{f(X_{1},\dots,X_{k})=w_{0}+\sum_{i=1}^{k}w_{i}X_{i}}\end{array}$ , so that $w_{0}$ takes the role of $-\tau$ . 
> 下一步我们要将贡献值转化为概率，即 $Y=y^1$ 的概率是如何依赖于 $f (X_1,\dots, X_k)$ 的
> 一般情况下，如果 $f (X_1,\dots, X_k )$ 超过某个阈值 $\tau$，则 $Y$ 就非常有可能为 1，当 $f (X_1,\dots, X_k) < \tau$，则 $Y$ 就非常有可能为 0
> 我们可以定义 $f (X_1,\dots, X_k) = w_0 + \sum_{i=1}^k w_i X_i$，其中 $w_0 = -\tau$，则此时 $f (X_1,\dots, X_k)$ 就可以直接与0比较

To provide a realistic model for immune system example and others, we do not use a hard threshold function to deﬁne the probability of $Y$ , but rather a smoother transition function. One common choice (although not the only one) is the sigmoid or logit function: 
> 为了提供一个更实际的模型，我们不会使用硬阈值函数来定义 $Y$ 的概率，而是使用更平滑的转换函数

$$
\mathrm{sigmoid}(z)=\frac{e^{z}}{1+e^{z}}.
$$ 

![[Probabilistic Graph Theory-Fig5.11.png]]

Figure 5.11a shows the sigmoid function. This function implies that the probability saturates to 1 when $f(X_{1},\dots,X_{k})$ is large, and saturates to 0 when $f(X_{1},\dots,X_{k})$ is small. 
And so, activation of another disease cause for a sick patient will not change the probability of fever by much, since it is already close to 1 . Similarly, if the patient is healthy, a minor burden on the immune system will not increase the probability of fever, since $f(X_{1},\cdot\cdot\cdot,X_{k})$ is far from the threshold. In the area of the phase transition, the behavior is close to linear. We can now deﬁne: 

**Deﬁnition 5.9** logistic CPD
Let $Y$ be a binary-valued random variable with $k$ parents $X_{1},\ldots,X_{k}$ that take on numerical values. The CPD $P(Y\mid X_{1},.\,.\,,X_{k})$ is a logistic CPD if there are $k+1$ weights $w_{0},w_{1},.\cdot\cdot,w_{k}$ such that: 

$$
P(y^{1}\mid X_{1},\ldots,X_{k})\ \ =\ \ \mathrm{sigmoid}(w_{0}+\sum_{i=1}^{k}w_{i}X_{i}).
$$ 

> 定义：
> 令 $Y$ 是二值随机变量，有 $k$ 个值为数值的父变量 $X_1,\dots, X_k$，如果存在 $k+1$ 个权重，满足

$$
P(y^1\mid X_1,\dots,X_k) = \text{sigmoid}(w_0 + \sum_{i=1}^k w_i X_i)
$$

> 则称 CPD $P (Y\mid X_1,\dots, X_k)$ 是逻辑 CPD

We have already encountered this CPD in example 4.20, where we saw that it can be derived by taking a naive Markov network and reformulating it as a conditional distribution. 

We can interpret the parameter $w_{i}$ in terms of its efect on the log-odds of $Y$ . In general, the odds ratio for a binary variable is the ratio of the probability of $y^{1}$ and the probability of $y^{0}$ . It is the same concept used when we say that the odds of some event (for example, a sports team winning the Super Bowl) are $^{u}2$ to 1.” Consider the odds ratio for the variable $Y$ , where we use $Z$ to represent $\begin{array}{r}{w_{0}+\sum_{i}w_{i}X_{i}}\end{array}$ : 
> 可以从 $Y$ 的对数几率的角度解释参数 $w_i$
> 二值变量的几率是 $y^1$ 的概率和 $y^0$ 的概率的比值
> 广义线性模型中，$Y$ 的几率如下，其中 $Z  = w_0 + \sum_i w_i X_i$

$$
O(X)={\frac{P(y^{1}\mid X_{1},\ldots,X_{k})}{P(y^{0}\mid X_{1},\ldots,X_{k})}}={\frac{e^{Z}/(1+e^{Z})}{1/(1+e^{Z})}}=e^{Z}.
$$

Now, consider the effect on this odds ratio as some variable $X_{j}$ changes its value from false to true . Let $X_{-j}$ be the variables in $X_{1},\ldots,X_{k}$ except for $X_{j}$ . Then: 

$$
\frac {O(X_{-j},x_j^1)}{O(X_{-j},x_j^0)} = \frac{\exp(w_0 + \sum_{i=\ne j}w_i X_i + w_j)}{\exp(w_0 + \sum_{i\ne j}w_i X_i)} = e^{w_j}
$$

Thus, $X_{j}=t r u e$ changes the odds ratio by a multiplicative factor of $e^{w_{j}}$ . A positive coefcient $w_{j}\,>\,0$ implies that $e^{w_{j}}\,>\,1$ so that the odds ratio increases, hence making $y^{1}$ more likely. Conversely, a negative coefcient $w_{j}\,<\,0$ implies that $e^{w_{j}}\,<\,1$ and hence the odds ratio decreases, making $y^{1}$ less likely. 
> 考虑当某个变量 $X_j$ 从 false 变为 true，前后几率的比值为 $e^{w_j}$
> 如果系数 $w_j > 0$，则 $e^{w_j} > 1$，说明几率增大，也就是 $y^1$ 更可能了
> 如果系数 $w_j < 0$，则 $e^{w_j} < 1$，说明几率减小，也就是 $y^0$ 更可能

Figure 5.11b shows a graph of the behavior of a special case of the logistic CPD model, where all the variables have the me weight $w$ . The graph shows $P(Y\mid X_{1},.\,.\,,X_{k})$ as a function of $w$ and the number of $X_{i}$ ’s that take the value true . The graph shows two cases: one where $w_{0}\,=\,0$ and the other where $w_{0}~=~-5$ . In the ﬁrst case, the probability starts out at 0.5, when none of the causes are in efect, and rapidly goes up to 1; the rate of increase is, as expected, much higher for high values of $w$ . 
> 可以看到 $w$ 越大，概率的增长速率也越大

It is interesting to compare this graph to the graph of ﬁgure 5.10b that shows the behavior of the noisy-or model with $\lambda_{0}=0.5$ . The graphs exhibit very similar behavior for $\lambda=w$ , showing that the incremental efect of a new cause is similar in both. However, the logistic CPD also allows for a negative inﬂuence of some $X_{i}$ on $Y$ by making $w_{i}$ negative. Furthermore, the parameterization of the logistic model also provides substantially more ﬂexibility in generating qualitatively diferent distributions. For example, as shown in ﬁgure 5.11c, setting $w_{0}$ to a diferent value allows us to obtain the threshold effect discussed earlier. Furthermore, as shown in ﬁgure 5.11d, we can adapt the scale of the parameters to obtain a sharper transition. However, the noisy-or model is cognitively very plausible in many settings. Furthermore, as we discuss, it has certain beneﬁts both in reasoning with the models and in learning the models from data. 
> 噪声或 CPD 模型在 $\lambda = w$ 时和逻辑 CPD 模型的行为非常相似
> 但逻辑 CPD 允许某个 $X_i$ 对 $Y$ 产生的是负面影响，只要让 $w_i$ 为负即可
> 逻辑 CPD 的参数化为生成性质上不同的分布也提供了更多灵活性，例如设定 $w_0$ 以改变阈值，以及 scale 参数以得到更陡峭的转变
#### 5.4.2.2 Multivalued Variables 
We can extend the logistic CPD to the case where $Y$ takes on multiple values $y^{1},\cdot\cdot\cdot,y^{m}$ . In this case, we can imagine that the diferent values of $Y$ are each supported in a diferent way by the $X_{i}$ ’s, where the support is again deﬁned via a linear function. The choice of $Y$ can be viewed as a soft version of “winner takes all,” where the $y^{i}$ that has the most support gets probability 1 and the others get probability 0. More precisely, we have: 
> 我们将逻辑 CPD 中的 $Y$ 拓展到多个值 $y^1,\dots, y^m$
> 可以认为每个 $X_i$ 都有不同的方式支持 $Y$ 的不同值，“支持”通过一个线性函数定义
> $Y$ 的取值就是具有支持最高的 $y^i$ 值

**Deﬁnition 5.10** multinomial logistic CPD 
Let $Y$ be an $m$ -valued random variable with $k$ parents $X_{1},\ldots,X_{k}$ that take on numerical values. The CPD $P(Y\mid X_{1},.\,.\,,X_{k})$ is $a$ multinomial logistic if for each $j=1,\dots,m,$ , there are $k+1$  weights $w_{j,0},w_{j,1},\cdot\cdot\cdot,w_{j,k}$ such that: 

$$
\begin{array}{r c l}{\ell_{j}(X_{1},\ldots,X_{k})}&{=}&{\displaystyle w_{j,0}+\sum_{i=1}^{k}w_{j,i}X_{i}}\\ {P(y^{j}\mid X_{1},\ldots,X_{k})}&{=}&{\displaystyle\frac{\exp\left(\ell_{j}(X_{1},\ldots,X_{k})\right)}{\sum_{j^{\prime}=1}^{m}\exp\left(\ell_{j^{\prime}}(X_{1},\ldots,X_{k})\right)}.}\end{array}
$$ 
> 定义：
> $Y$ 是一个 $m$ 值随机变量，有 $k$ 个父变量 $X_1,\dots, X_k$
> 如果对于每个 $j = 1,\dots, m$，都存在 $k+1$ 个权重 $w_{j, 0}, \dots, w_{j, k}$，使得

$$
\begin{array}{r c l}{\ell_{j}(X_{1},\ldots,X_{k})}&{=}&{\displaystyle w_{j,0}+\sum_{i=1}^{k}w_{j,i}X_{i}}\\ {P(y^{j}\mid X_{1},\ldots,X_{k})}&{=}&{\displaystyle\frac{\exp\left(\ell_{j}(X_{1},\ldots,X_{k})\right)}{\sum_{j^{\prime}=1}^{m}\exp\left(\ell_{j^{\prime}}(X_{1},\ldots,X_{k})\right)}.}\end{array}
$$

> 则称条件概率分布 $P (Y\mid X_1,\dots, X_k)$ 是一个多项式逻辑分布

Figure 5.12 shows one example of this model for the case of two parents and a three-valued child $Y$ . We note that one of the weights $w_{j,1},\cdot\cdot\cdot,w_{j,k}$ is redundant, as it can be folded into the bias term $w_{j,0}$ . 

We can also deal with the case where the parent variables $X_{i}$ take on more than two values. The approach taken is usually straightforward. If $X_{i}^{=}x_{i}^{1},\ldots,x_{i}^{m}$ , we deﬁne a new set of binary- valued variables $X_{i,1},.\cdot\cdot\cdot,X_{i,m},$ , where $X_{i,j}=x_{i,j}^{1}$ precisely when $X_{i}=j$ . 
> 如果父变量取多个值时，我们可以定义多个二值变量

Each of these new variables gets its own coefcient (or set of coefcients) in the logistic function. For example, if we have a binary-valued child $Y$ with an $m$ -valued parent $X$ , our logistic function would be parameterized using $m+1$ weights, $w_{0},w_{1},\dots,w_{m}$ , such that 

$$
P(y^{1}\mid X)\;\;=\;\;\mathrm{sigmoid}(w_{0}+\sum_{j=1}^{m}w_{j}I\{X=x^{j}\}).\tag{5.4}
$$ 
We note that, for any assignment to $X_{i}$ , precisely one of the weights $w_{1},.\,.\,.\,,w_{m}$ will make a contribution to the linear function. As a consequence, one of the weights is redundant, since it can be folded into the bias weight $w_{0}$
> 对于对 $X_i$ 的每一个赋值，仅仅只会有1个权重 $w_1,\dots, w_m$ 会对线性函数做出贡献

We noted before that we can view a binary-valued logistic CPD as a conditional version of a naive Markov model. We can generalize this observation to the nonbinary case, and show that the multinomial logit CPD is also a particular type of pairwise CRF (see exercise 5.16). 
### 5.4.3 The General Formulation 
Both of these models are specials case of a general class of local probability models, which satisfy a property called causal independence or independence of causal inﬂuence ( ICI ). These models all share the property that the inﬂuence of multiple causes can be decomposed into separate inﬂuences. We can deﬁne the resulting class of models more precisely as follows: 
> 之前介绍的两个广义线性模型都是局部概率模型的特例
> 局部概率模型满足一个称为因果独立性/因果影响独立性的特性，也就是多个 cause 的影响可以被分解为它们分别的影响

![[Probabilistic Graphical Models-Fig5.13.png]]

**Deﬁnition 5.11** 
Let $Y$ be a random variable with parents $X_{1},\ldots,X_{k}$ . The CPD $P(Y\mid X_{1},.\,.\,,X_{k})$ exhibits independence of causal inﬂuence if it is described via a network fragment of the structure shown in ﬁgure 5.13, where the CPD of $Z$ is a deterministic function $f$ . 
> 定义：
> 令 $Y$ 为父变量是 $X_1,\dots, X_k$ 的随机变量，如果CPD $P (Y\mid X_1,\dots, X_k)$ 可以通过5.13中的网络结构描述（其中 $Z$ 的 CPD 是一个确定性函数 $f$），则 $P (Y\mid X_1,\dots, X_k)$ 即展现了因果影响之间的独立性

Intuitively, each variable $X_{i}$ can be transformed separately using its own individual noise model. The resulting variables $Z_{i}$ are combined using some deterministic combination function. Finally, an additional stochastic choice can be applied to the result $Z$ , so that the ﬁnal value of $Y$ is not necessarily a deterministic function of the variables $Z_{i}$ ’s. The key here is that any stochastic parts of the model are applied independently to each of the $X_{i}$ ’s, so that there can be no interactions between them. The only interaction between the $X_{i}$ ’s occurs in the context of the function $f$ . 
> 每个变量 $X_i$ 通过自己独立的噪声模型被分别转换为变量 $Z_i$ ，然后经过某个确定性函数被结合为 $Z$，最终对 $Z$ 应用随机选择
> 故 $Y$ 的最终值并不必要是变量 $Z_i$ 的确定性函数
> 关键在于模型中每个随机部分都是独立应用于各个 $X_i$ 的，因此它们之间不存在交互，$X_i$ 之间唯一的交互存在于函数 $f$ 中

As stated, this deﬁnition is not particularly meaningful. Given an arbitrarily complex function $f$ , we can represent any CPD using the representation of ﬁgure 5.13. (See exercise 5.15.) It is possible to place various restrictions on the form of the function $f$ that would make the deﬁnition more meaningful. For our purposes, we provide a fairly stringent deﬁnition that fortunately turns out to capture the standard uses of ICI models. 

**Deﬁnition 5.12** 
We say that a deterministic binary function $x\diamond y$ is commutative if $x\diamond y=y\diamond x$ , and associative $i f\left(x\diamond y\right)\diamond z=x\diamond\left(y\diamond z\right)$ . We say that a function $f(x_{1},\dots,x_{k})$ is $a$ symmetric decomposable if there is a commutative associative function $x\diamond y$ such that $f(x_{1},.\,.\,.\,,x_{k})=x_{1}\diamond x_{2}\diamond$ $\cdot\cdot\cdot x_{k}$ . 
> 定义：
> 如果存在一个可交换且可结合的函数 $x\diamond y$ ，使得函数 $f (x_1,\dots, x_k) = x_1\diamond x_2\diamond  \cdots \diamond x_k$，则称 $f (x_1,\dots, x_k)$ 是对称可分解函数

**Deﬁnition 5.13** symmetric ICI 
We say that the CPD $P(Y\mid X_{1},.\,.\,,X_{k})$ exhibits symmetric ICI it is described via a network fragment of the structure shown in ﬁgure 5.13, where the CPD of Z is a deterministic symmetric decomposable function $f$ . The CPD exhibits fully symmetric ICI if the CPDs of the diferent $Z_{i}$ variables are identical. 
> 定义：
> 如果 CPD $P (Y\mid X_1,\dots, X_k)$ 可以被 fig5.13的结构描述，且其中 $Z$ 的 CPD 是一个对称可分解函数，则称 CPD 展现了对称的因果影响间的独立性
> 如果不同的 $Z_i$ 的 CPDs 都相同，则 CPD 展现了完全的对称 ICI

There are many instantiations of the symmetric ICI model, with diferent noise models — $P(Z_{i}\mid X_{i})$ — and diferent combination functions. Our noisy-or model uses the combination function OR and a simple noise model with binary variables. The generalized linear models use the $Z_{i}$ to produce $w_{i}X_{i}$ , and then summation as the combination function $f$ . The ﬁnal soft thresholding efect is accomplished in the distribution of $Y$ given $Z$ . 
> 不同的 ICI 模型的区别：不同的噪声模型 $P (Z_i\mid X_i)$ 和不同的结合函数

These types of models turn out to be very useful in practice, both because of their cognitive plausibility and because they provide a signiﬁcant reduction in the number of parameters required to represent the distribution. The number of parameters in the CPD is linear in the number of parents, as opposed to the usual exponential. 

Box 5.D — Case Study: Noisy Rule Models for Medical Diagnosis. As discussed in box 5.C, noisy rule interactions such as noisy-or are a simple yet plausible first approximation of models for noisy-max medical diagnosis. A generalization that is also useful in this setting is the noisy-max model. Like
noisy-max the application of the noisy-or model for diagnosis, the parents $X_{i}$ correspond to diferent diseases that the patient might have. In this case, however, the value space of the symptom variable $Y$ can be more reﬁned than simply $\{p r e s e n t,a b s e n t\}$ ; it can encode the severit f the symptom. Each $Z_{i}$ corresponds (intuitively) to the efect of the disease $X_{i}$ on the symptom Y in isolation, that is, the severity of the symptom in case only the disease $X_{i}$ is present. The value of $Z$ is the maximum of the diferent $Z_{i}$ ’s. 

Both noisy-or and noisy-max models have been used in several medical diagnosis networks. Two of the largest are the QMR-DT (Shwe et al. 1991) and CPCS (Pradhan et al. 1994) networks, both based on various versions of a knowledge-based system called QMR (Quick Medical Reference), compiled for diagnosis of internal medicine. QMR-DT is a BN2O network (see box 5.C) that contains more than ﬁve hundred signiﬁcant diseases, about four thousand associated ﬁndings, and more than forty thousand disease-ﬁnding associations. 

CPCS is a somewhat smaller network, containing close to ﬁve hundred variables and more than nine hundred edges. Unlike QMR-DT, the network contains not only diseases and ﬁndings but also variables for predisposing factors and intermediate physiological states. Thus, CPCS has at least four distinct layers. All variables representing diseases and intermediate states take on one of four values. A speciﬁcation of the network using full conditional probability tables would require close to 134 million parameters. However, the network is constructed using only noisy-or and noisy-max interactions, so that the number of actual parameters is only 8,254. Furthermore, most of the parameters were generated automatically from “frequency weights” in the original knowledge base. Thus, the number of parameters that were, in fact, elicited during the construction of the network is around 560. 

Finally, the symmetric ICI models allow certain decompositions of the CPD that can be exploited by probabilistic inference algorithms for computational gain, when the domain of the variables $Z_{i}$ and the variable $Z$ are reasonably small. 
### 5.4.4 Independencies 
As we have seen, structured CPDs often induce independence properties that go beyond those represented explicitly in the Bayesian network structure. Understanding these independencies can be useful for gaining insight into the properties of our distribution. Also, as we will see, the additional structure can be exploited for improving the performance of various probabilistic inference algorithms. 
> 我们知道有结构的 CPDs 可以引入贝叶斯网络结构中不能准确表示的独立性

The additional independence properties that arise in general ICI models $P(Y\mid X_{1},.\,.\,,X_{k})$ are more indirect than those we have seen in the context of deterministic CPDs or tree-CPDs. In particular, they do not manifest directly in terms of the original variables, but only if we decompose it by adding auxiliary variables. In particular, as we can easily see from ﬁgure 5.13, each $X_{i}$ is conditionally independent of $Y$ , and of the other $X_{j}$ ’s, given $Z_{i}$ . 
> 在通用 ICI 模型 $P (Y\mid X_1,\dots, X_k)$ 中出现的额外独立性相较于 determinstic CPDs 和 tree-CPDs 中的独立性要更不直接
> 它们并不关于原始变量展现出来，需要我们通过添加辅助变量进行分解，例如 fig5.13中，每个 $X_i$ 在给定 $Z_i$ 的情况下，条件独立于 $Y$ 和其他 $X_j$

We can obtain even more independencies by decomposing the CPD of $Z$ in various ways. For example, assume that $k=4$ , so that our CPD has the form $P(Y\mid X_{1},X_{2},X_{3},X_{4})$ . We can introduce two new variables $W_{1}$ and $W_{2}$ , such that: 

$$
\begin{array}{r c l}{{W_{1}}}&{{=}}&{{Z_{0}\diamond Z_{1}\diamond Z_{2}}}\\ {{W_{2}}}&{{=}}&{{Z_{3}\diamond Z_{4}}}\\ {{Z}}&{{=}}&{{W_{1}\diamond W_{2}}}\end{array}
$$ 
By the associativity of $\diamondsuit$ , the decomposed CPD is precisely equivalent to th riginal one. In this CPD, we can use the results of section 5.2 to conclude, for example, that $X_{4}$ is independent of $Y$ given $W_{2}$ . 
> 通过对 $Z$ 的 CPD 以多种方式分解，可以引入更多独立性

Although these independencies might appear somewhat artiﬁcial, it turns out that the associated decomposition of the network can be exploited by inference algorithms (see section 9.6.1). However, as we will see, they are only useful when the domain of the intermediate variables $(W_{1}$ and $W_{2}$ in our example) are small. This restriction should not be surprising given our earlier observation that any CPD can be decomposed in this way if we allow the $Z_{i}$ ’s and $Z$ to be arbitrarily complex. 

The independencies that we just saw are derived simply from the fact that the CPD of $Z$ is deterministic and symmetric. As in section 5.2, there are often additional independencies that are associated with the particular choice of deterministic function. 
> 上例中我们仅利用了 $Z$ 的 CPD 是确定且对称的性质推导出了新的独立性，我们可以看到之后会存在和特定的确定性函数相关的独立性

The best-known independence of this type is the one arising for noisy-or models: 

**Proposition 5.3**
$P(Y\mid X_{1},.\,.\,,X_{k})$ be a noisy-or CPD. Then for each $i\neq j$ , $X_{i}$ is independent of $X_{j}$ given $Y=y^{0}$ . 
> 引理：
> 对于噪声或 CPD $P (Y\mid X_1,\dots, X_k)$，当 $i\ne j$，$X_i$ 在给定 $Y = y^0$ 的情况下就独立于 $X_j$

The proof is left as an exercise (exercise 5.11). Note that this independence is not derived from the network structure via d-separation: Instantiating $Y$ enables the v-structure between $X_{i}$ and $X_{j}$ , and hence potentially renders them correlated. Furthermore, this independence is context- speciﬁc: it holds only for the speciﬁc value $Y=y^{0}$ . Other deterministic functions are associated with other context-speciﬁc independencies. 
> 该独立性不能从 d-seperation 中推导出来，而是和上下文 $Y = y^0$ 相关的
## 5.5 Continuous Variables 
So far, we have restricted attention to discrete variables with ﬁnitely many values. In many situations, some variables are best modeled as taking values in some continuous space. Examples include variables such as position, velocity, temperature, and pressure. Clearly, we cannot use a table representation in this case. One common solution is to circumvent the entire issue by discretizing all continuous variables. Unfortunately, this solution can be problematic in many cases. In order to get a reasonably accurate model, we often have to use a fairly ﬁne discretization, with tens or even hundreds of values. For example, when applying probabilistic models to a robot navigation task, a typical discretization granularity might be 15 centimeters for the $x$ and $y$ coordinates of the robot location. For a reasonably sized environment, each of these variables might have more than a thousand values, leading to more than a million discretized values for the robot’s position. CPDs of this magnitude are outside the range of most systems. 

**Furthermore, when we discretize a continuous variable we often lose much of the structure that characterizes it. It is not generally the case that each of the million values that deﬁnes a robot position can be associated with an arbitrary probability. Basic continuity assumptions that hold in almost all domains imply certain relationships that hold between probabilities associated with “nearby” discretized values of a continuous variable. However, such constraints are very hard to capture in a discrete distribution, where there is no notion that two values of the variable are “close” to each other.** 

Fortunately, nothing in our formulation of a Bayesian network requires that we restrict attention to discrete variables. Our only requirement is that the CPD $P(X\mid\mathrm{Pa}_{X})$ represent, for every assignment of values $\operatorname{pa}_{X}$ to $\mathrm{Pa}_{X}$ , a distribution over $X$. In this case, $X$ might be continuous, in which case the CPD would need to represent distributions over a continuum of values; we might also have some of $X$ ’s parents be continuous, so that the CPD would also need to represent a continuum of diferent probability distributions. However, as we now show, we can provide implicit representations for CPDs of this type, allowing us to apply all of the machinery we developed for the continuous case as well as for hybrid networks involving both discrete and continuous variables. 
> 我们的贝叶斯网络公式并不限定在离散变量，唯一的要求就是 CPD $P (X\mid \text{Pa}_X)$ 对于 $\text{Pa}_X$ 的每一个赋值 $\text{pa}_X$ 都定义了一个在 $X$ 上的分布

In this section, we describe how continuous variables can be integrated into the BN framework. We ﬁrst describe the purely continuous case, where the CPDs involve only continuous variables, both as parents and as children. We then examine the case of hybrid networks, which involve both discrete and continuous variables. 
> 本节讨论贝叶斯网络包含连续变量的情况，包括了仅仅包含连续变量以及既包含了连续变量也包含了离散变量

There are many possible models one could use for any of these cases; we brieﬂy describe only one prototypical example for each of them, focusing on the models that are most commonly used. Of course, there is an unlimited range of representations that we can use: any parametric representation for a CPD is eligible in principle. The only difculty, as far as representation is concerned, is in creating a language that allows for it. Other tasks, such as inference and learning, are a diferent issue. As we will see, these tasks can be difcult even for very simple hybrid models. 

The most commonly used parametric form for continuous density functions is the Gaussian distribution. We have already described the univariate Gaussian distribution in chapter 2. We now describe how it can be used within the context of a Bayesian network representation. 
> 连续密度函数最常用的参数化形式就是高斯分布

First, let us consider the problem of representing a dependency of a continuous variable $Y$ on a continuous parent $X$ . One simple solution is to decide to model the distribution of $Y$ as a Gaussian, whose parameters depend on the value of $X$ . In this case, we need to have a set of parameters for every one of the inﬁnitely many values $x\in V a l(X)$ . A commo olution is to decide that the mean of Y is a linear function of X , and that the variance of Y does not depend on $X$ . For example, we might have that 
> 考虑如何表示连续变量 $Y$ 依赖于连续变量 $X$
> 一个简单方法是将 $Y$ 的分布建模为高斯分布，其参数依赖于 $X$ 的值，例如 $Y$ 的均值是 $X$ 的线性函数，$Y$ 的方差不依赖于 $X$

$$
p(Y\mid x)=\!{\mathcal{N}}\left(-2x+0.9;1\right).
$$ 
Example 5.20 
Consider a vehicle (for example, a car) moving over time. For simplicity, assume that the vehicle is moving along a straight line, so that its position (measured in meters) at the t ’th second is described using a single variable $X^{(t)}$ . Let $V^{(t)}$ represent the velocity of the car at the k th second, measured in meters per second. Then, under ideal motion, we would have that $X^{(t+1)}=X^{(t)}+V^{(t)}-i f$ the car is at meter #510 along the road, and its current velocity is 15 meters/second, then we expect its position at the next second to be meter #525. However, there is invariably some stochasticity in the motion. Hence, it is much more realistic to assert that the car’s position $X^{(t+1)}$ is described using a Gaussian distribution whose mean is 525 and whose variance is 5 meters. 

This type of dependence is called a linear Gaussian model. It extends to multiple continuous parents in a straightforward way: 
> 这类依赖被称为线性高斯模型

**Deﬁnition 5.14** linear Gaussian CPD 
Let $Y$ be a continuous variable with continuous parents $X_{1},\ldots,X_{k}$ . We say that $Y$ has a linear Gaussian model if there are parameters $\beta_{0},\ldots,\beta_{k}$ and $\sigma^{2}$ such that 

$$
p(Y\mid x_{1},\ldots,x_{k})=\mathcal{N}\left(\beta_{0}+\beta_{1}x_{1}+\cdot\cdot\cdot+\beta_{k}x_{k};\sigma^{2}\right).
$$ 
In vector notation, 

$$
p(Y\mid\mathbf{\boldsymbol{x}})=\mathcal{N}\left(\beta_{0}+\beta^{T}\mathbf{\boldsymbol{x}};\sigma^{2}\right).
$$ 

> 定义：
> $Y$ 是连续变量，其父变量包括 $X_1,\dots, X_k$，如果存在参数 $\beta_0,\dots, \beta_k$ 以及 $\sigma^2$，使得

$$
p(Y\mid x_{1},\ldots,x_{k})=\mathcal{N}\left(\beta_{0}+\beta_{1}x_{1}+\cdot\cdot\cdot+\beta_{k}x_{k};\sigma^{2}\right).
$$

> 则 $Y$ 具有线性高斯模型

Viewed slightly diferently, this formulation says that $Y$ is a linear function of the variables $X_{1},\ldots,X_{k}$ , with the addition of Gaussian noise with mean 0 and variance $\sigma^{2}$ : 

$$
Y=\beta_{0}+\beta_{1}x_{1}+\cdot\cdot\cdot+\beta_{k}x_{k}+\epsilon,
$$ 
where $\epsilon$ is a Gaussian random variable with mean 0 and variance $\sigma^{2}$ , representing the noise in the system. 
> 该表示也可以理解为 $Y$ 是有关于 $X_1,\dots, X_k$ 的线性函数再加上均值为0，方差为 $\sigma^2$ 的高斯噪声 $\epsilon$

This simple model captures many interesting dependencies. However, there are certain facets of the situation that it might not capture. For example, the variance of the child variable $Y$ cannot depend on the actual values of the parents. In example 5.20, we might wish to construct a model in which there is more variance about a car’s future position if it is currently moving very quickly. The linear Gaussian model cannot capture this type of interaction. 

Of course, we can easily extend this model to have the mean and variance of $Y$ depend on the values of its parents in arbitrary way. For example, we can easily construct a richer representation where we allow the mean of $Y$ to be $\sin(x_{1})^{x_{2}}$ and its variance to be $(x_{3}/x_{4})^{2}$ . However, the linear Gaussian model is a very natural one, which is a useful approximation in many practical applications. Furthermore, as we will see in section 7.2, networks based on the linear Gaussian model provide us with an alternative representation for multivariate Gaussian distributions, one that directly reveals more of the underlying structure. 

Box 5.E — Case Study: Robot Motion and Sensors. 
One interesting application of hybrid mod- els is in the domain of robot localization . In this application, the robot must keep track of its location as it moves in an environment, and obtains sensor readings that depend on its location. This application is an example of a temporal model, a topic that will be discussed in detail in section 6.2; we also return to the robot example speciﬁcally in box 15.A. There are two main local probability models associated with this application. The ﬁrst speciﬁes the robot dynamics — the distribution over its position at the next time step $L^{\prime}$ given its current position $L$ and the action taken $A$ ; the second speciﬁes the robot sensor model — the distribution over its observed sensor reading $S$ at the current time given its current location $L$ . 

We describe one model for this application, as proposed by Fox et al. (1999) and Thrun et al. (2000). Here, the robot location $L$ is a three-dimensional vector containing its $X,Y$ coordinates and an angular orientation $\theta$ . The action $A$ speciﬁes a distance to travel and a rotation (ofset from the current $\theta_{-}$ ). The model uses the assumption that the errors in both translation and rotation are normally distributed with zero mean. Speciﬁcally, $P(L^{\prime}\mid L,A)$ is deﬁned as a product of two independent Gaussians with cut of tails, $P(\theta^{\prime}\mid\theta,A)$ and $P(X^{\prime},Y^{\prime}\mid X,Y,A)$ , whose variances 

are proportional to the length of the motion. The robot’s conditional distribution over $(X^{\prime},Y^{\prime})$ is $a$ banana-shaped cloud (see ﬁgure 5.E.1a, where the banana shape is due to the noise in the rotation. 

The sensor is generally some type of range sensor, either a sonar or a laser, which provides $a$ reading $D$ of the distance between the robot and the nearest obstacle along the direction of the sensor. There are two distinct cases to consider. If the sensor signal results from an obstacle in the map, then the resulting distribution is modeled by $a$ Gaussian distribution with mean at the distance to this obstacle. Letting $O_{L}$ be the distance to the closest obstacle to the position $L$ (along the sensor beam), we can deﬁne $P_{m}(D\mid L)=\mathcal{N}\left(o_{L};\sigma^{2}\right)$  , where the variance $\sigma^{2}$ represents the uncertainty of the measured distance, based on the accuracy of the world model and the accuracy of the sensor. Figure 5.E.1b shows an example of such a distribution for an ultrasound sensor and $a$ laser range ﬁnder. The laser sensor has a higher accuracy than the ultrasound sensor, as indicated by the smaller variance. 

The second case arises when the sensor beam is reﬂected by an obstacle not represented in the world model (for example, a dynamic obstacle, such as a person or a chair, which is not in the robot’s map). Assuming that these objects are equally distributed in the environment, the probability $P_{u}(D)$ of detecting an unknown obstacle at distance $D$ is independent of the location of the robot and can be modeled by an exponential distribution. This distribution results from the observation that a distance $d$ is measured if the sensor is not reﬂected by an obstacle at a shorter distance and is reﬂected at distance $d$ . An example exponential distribution is shown in ﬁgure 5.E.1c. 

Only one of these two cases can hold r a given measurement. Thus, $P(D\mid L)$ is a combi- nation of the tw distributions $P_{m}$ and $P_{u}$ . The combined probability $P(D\mid L)$ is based on the observation that d is measured in one of two cases: 

• The sensor beam in not reﬂected by an unknown obstacle before reaching distance $d$ , and is reﬂected by the known obstacle at distance $d$ (an event that happens only with some probability). 
• The beam is reﬂected neither by an unknown obstacle nor by the known obstacle before reaching distance $d$ , and it is reﬂected by an unknown obstacle at distance $d$ . 

Overall, the probability of sensor measurements is computed incrementally for the diferent distances starting at 0cm ; for each distance, we consider the probability that the sensor beam reaches the corresponding distance and is reﬂected either by the closest obstacle in the map (along the sensor beam) or by an unknown obstacle. Putting these diferent cases together, we obtain a single distribu- tion for $P(D\mid L)$ . This distribution is shown in ﬁgure 5.E.1d, along with an empirical distribution obtained from data pairs consisting of the distance $O_{L}$ to the closest obstacle on the map and the measured distance d during the typical operation of the robot. 
### 5.5.1 Hybrid Models 
We now turn our attention to models incorporating both discrete and continuous variables. We have to address two types of dependencies: a continuous variable with continuous and discrete parents, and a discrete variable with continuous and discrete parents. 
> 我们考虑同时包含离散和连续变量的模型
> 我们需要解决两类依赖：连续变量具有连续和离散的父变量、离散变量具有连续和离散的父变量

Let us ﬁrst consider the case of a continuous child $X$ . If we ignore the discrete parents of $X$ , we can simply represent the CPD of $X$ as a linear Gaussian of $X$ ’s continuous parents. 
The simplest way of making the continuous variable $X$ depend on a discrete variable $U$ is to deﬁne a diferent set of parameters for every value of the discrete parent. More precisely: 
> 考虑连续变量为子变量，最简单的方式是为离散父变量的每一个取值定义一组参数

**Deﬁnition 5.15**  conditional linear Gaussian CPD 
$X$ ontinuous variable, and let $U\,=\,\{U_{1},.\,.\,.\,,U_{m}\}$ be its discrete parents and $\textbf{\textit{Y}}=$ $\{Y_{1},\ldots,Y_{k}\}$ be its continuous parents. We say that X has $a$ conditional linear Gaussian (CLG) CPD if, for every value $\pmb u\in\mathit{V a l}(\pmb{U})$ , we have $a$ set of $k+1$ coefcients $a_{{\pmb u},0},\cdot\cdot\cdot,a_{{\pmb u},k}$ and $^a$ variance $\sigma_{u}^{2}$ such that 

$$
p(X\mid\mathbf{\boldsymbol{u}},\mathbf{\boldsymbol{y}})=\mathcal{N}\left(a_{\mathbf{\boldsymbol{u}},0}+\sum_{i=1}^{k}a_{\mathbf{\boldsymbol{u}},i}y_{i};\sigma_{\mathbf{\boldsymbol{u}}}^{2}\right)
$$ 
> 定义：
> $X$ 为连续变量，$\pmb U = \{U_1,\dots, U_m\}$ 为其离散父变量，$\pmb Y=\{Y_1,\dots, Y_k\}$ 为其连续父变量，如果对于每个值 $\pmb u \in Val (\pmb U)$，我们都存在 $k+1$ 个系数 $a_{\pmb u, 0},\dots, a_{\pmb u, k}$，以及一个方差 $\sigma_{\pmb u}^2$，使得

$$
p(X\mid\mathbf{\boldsymbol{u}},\mathbf{\boldsymbol{y}})=\mathcal{N}\left(a_{\mathbf{\boldsymbol{u}},0}+\sum_{i=1}^{k}a_{\mathbf{\boldsymbol{u}},i}y_{i};\sigma_{\mathbf{\boldsymbol{u}}}^{2}\right)
$$
> 则称 $X$ 具有条件线性高斯 CPD

If we restrict attention to this type of CPD, we get an interesting class of models. More precisely, we have: 

**Deﬁnition 5.16** CLG network 
A Bayesian network is called a CLG network if every discrete variable has only discrete parents and every continuous variable has a CLG CPD. 
> 定义：
> 如果贝叶斯网络中每个离散变量仅有离散父变量，且每个连续变量都有一个 CLG (continuous linear gaussian) CPD，则该网络称为一个 CLG 网络

Note that the conditional linear Gaussian model does not allow for continuous variables to have discrete children. A CLG model induces a joint distribution that has the form of a mixture — a weighted average — of Gaussians. The mixture contains one Gaussian component for each instantiation of the discrete network variables; the weight of the component is the probability of that instantiation. Thus, the number of mixture components is (in the worst case) exponential in the number of discrete network variables. 
> 根据定义，条件线性高斯模型不允许连续变量有离散的子变量
> CLG 模型将联合分布定义为了一个加权混合高斯分布，each instatiation of the discrete network variables 都贡献一个高斯成分，该高斯成分的权重就是该 instantiation 的概率
> 因此高斯成分的数量实际上会与离散变量的数量成指数比例

Finally, we address the case of a discrete child with a continuous parent. The simplest model is a threshold model. Assume we have a binary discrete variable $U$ with a continuous parent $Y$ . We may want to deﬁne: 
> 对于具有连续父变量的离散变量，最简单的方式就是 threshold 模型

$$
P(u^{1})=\left\{\begin{array}{l l}{0.9\qquad\qquad y\leq65}\\ {0.05\qquad\qquad\mathrm{otherwise}.}\end{array}\right.
$$ 
Such a model may be appropriate, for example, if $Y$ is the temperature (in Fahrenheit) and $U$ is the thermostat turning the heater on. 

The problem with the threshold model is that the change in probability is discontinuous as a function of $Y$ , which is both inconvenient from a mathematical perspective and implausible in many settings. However, we can address this problem by simply using the logistic model or its multinomial extension, as deﬁned in deﬁnition 5.9 or deﬁnition 5.10. 
> threshold 模型的问题在于概率的变化是连续父变量的不连续函数，一种办法是 logistic model 或者其多项式拓展

Figure 5.14 shows how a multinomial CPD can be used to model a simple sensor that has three values: low , medium and high . The probability of each of these values depends on the value of the continuous parent $Y$ . As discussed in section 5.4.2, we can easily accommodate a variety of noise models for the sensor: we can make it less reliable in borderline situations by making the transitions between regions more moderate. It is also fairly straightforward to generalize the model to allow the probabilities of the diferent values in each of the regions to be values other than 0 or 1. 

As for the conditional linear Gaussian CPD, we address the existence of discrete parents for $Y$ by simply introducing a separate set of parameters for each instantiation of the discrete parents. 
## 5.6 Conditional Bayesian Networks 
The previous sections all describe various compact representations of a CPD. Another very useful way of compactly representing a conditional probability distribution is via a Bayesian network fragment. We have already seen one very simple example of this idea: Our decomposition of the noisy-or CPD for the Letter variable, shown in ﬁgure 5.9. There, our decomposition used a Bayesian network to represent the internal model of the Letter variable. The network included explicit variables for the parents of the variable, as well as auxiliary variables that are not in the original network. This entire network represented the CPD for Letter . In this section, we generalize this idea to a much wider setting. 
> 之前的部分描述了 CPD 的多种紧凑表示
> 另一种紧凑表示 CPD 的方式是 Bayesian network fragment，例如我们的对噪声或模型的分解
> 即 fig5.9中，我们用 Bayesian network 表示变量 $L$ 的 internal model，这个网络包含了 $L$ 变量的直接父变量，以及原来网络中没有的辅助变量
> 本节我们将该思想拓展到更一般的设定下

Note that the network fragment in this example is not a full Bayesian network. In particular, it does not specify a probabilistic model — parents and a CPD — for the parent variables Questions and FinalPaper . This network fragment speciﬁes not a joint distribution over the variables in the fragment, but a conditional distribution of Letter given Questions and FinalPaper . More generally, we can deﬁne the following: 
> fig5.9中的 network fragment 并不是一个完全的 Bayesian network，network fragment 中的信息并没有帮助指定其中的父变量的条件概率分布
> network fragment 并没有指定 fragment 中所有变量的联合分布，而是仅仅指定了关于一个变量在给定其 parent variables 时的条件概率分布

**Definition 5.17** conditional Bayesian network
$A$ conditional network $\mathcal{B}$ over $Y$ given $X$ is deﬁned as a direc acyclic graph $\mathcal{G}$ whose nodes are $X\cup Y\cup Z$ , where $X,Y,Z$ ar disjoint. The variables in X are alled inputs, the variables in $Y$ outputs , and the variables in Z encapsulated . The variables in $X$ have no parents in $\mathcal{G}$ . The variables in $Z\cup Y$ are associated with a conditional probability distribution. The network deﬁnes a conditional distribution using a chain rule: 

$$
P_{\mathcal{B}}(Y,Z\mid X)=\prod_{X\in Y\cup Z}P(X\mid\mathrm{Pa}_{X}^{\mathcal{G}}).
$$ 
The distribution $P_{\mathcal{B}}(Y\mid X)$ is deﬁned as the marginal of $P_{\mathcal{B}}(Y,Z\mid X)$ : 

$$
P_{\mathcal{B}}(Y\mid X)=\sum_{Z}P_{\mathcal{B}}(Y,Z\mid X).
$$ 

> 定义：
> 给定 $\pmb X$，在 $\pmb Y$ 上的条件贝叶斯网络定义为一个有向无环图 $\mathcal G$，节点为 $\pmb X\cup \pmb Y \cup \pmb Z$，其中 $\pmb X, \pmb Y, \pmb Z$ 不相交
> $\pmb X$ 中的变量称为输入，$\pmb Y$ 中的称为输出
> 输入变量在图中没有父节点，$\pmb Z\cup \pmb Y$ 中的变量则都和一个条件概率分布相关联，即：

$$
P_{\mathcal{B}}(\pmb Y,\pmb Z\mid \pmb X)=\prod_{X\in \pmb Y\cup \pmb Z}P(X\mid\mathrm{Pa}_{X}^{\mathcal{G}}).
$$
> 分布 $P_{\mathcal B}(\pmb Y \mid \pmb X)$ 定义为 $P_{\mathcal B}(\pmb Y, \pmb Z\mid \pmb X)$ 的边际分布：

$$
P_{\mathcal{B}}(\pmb Y\mid \pmb X)=\sum_{\pmb Z}P_{\mathcal{B}}(\pmb Y,\pmb Z\mid \pmb X).
$$


The conditional random ﬁeld of section 4.6.1 is the undirected analogue of this deﬁnition. 
> 条件随机场就是该概念的无向图版本

The notion of a conditional BN turns out to be useful in many settings. In particular, we can use it to deﬁne an encapsulated CPD. 
> 条件贝叶斯网络可以用于定义 encapsulated CPD

**Definition 5.18**
Let $Y$ be a random variable with $k$ parents $X_{1},\ldots,X_{k}$ . The CPD $P(Y\mid X_{1},.\,.\,.\,,X_{k})$ is an encapsulated CPD if it is represented using a conditional Bayesian network over Y given $X_{1},\ldots,X_{k}$ . 
> 定义：
> 令 $Y$ 是具有 $k$ 个父变量 $X_1,\dots, X_k$ 的随机变量，如果条件概率分布可以使用一个在给定 $X_1,\dots, X_k$ 时在 $Y$ 上的条件贝叶斯网络表示，则 CPD $P (Y\mid X_1,\dots, X_k)$ 就是一个 encapsulated CPD

At some level, it is clear that the representation of an individual CPD for a variable $Y$ as a conditional Bayesian network ${\mathcal{B}}_{Y}$ es not add expressive power to the model. After we could simply take the network ${\mathcal{B}}_{Y}$ and “substitute it in” for the atomic CPD $P(Y\mid$ $\mathrm{Pa}_{Y}$ ) . 
One key advantage of the encapsulated representation over a more explicit model is that the encapsulation can simplify the model signiﬁcantly from a cognitive perspective. Consider again our noisy-or model. Externally, to the rest of the network, we can still view Letter as a single variable with its two parents: Questions and FinalPaper . All of the internal structure is encapsulated, so that, to the rest of the network, the variable can be viewed as any other variable. In particular, a knowledge engineer specifying the network does not have to ascribe meaning to the encapsulated variables. 
> 封装的 CPD 隐藏了内部的细节，使得我们可以直接将其视作一个原子 CPD 看待，外部的成分只需要关心它的输入和输出变量

The encapsulation advantage can be even more signiﬁcant when we want to describe a complex system where components are composed of other, lower-level, subsystems. When specifying a model for such a system, we would like to model each subsystem separately, without having to consider the internal model of its lower level components. 

In particular, consider a model for a physical device such as a computer; we might construct such a model for fault diagnosis purposes. When modeling the computer, we would like to avoid thinking about the detailed structure and fault models of its individual components, such as the hard drive, and within the hard drive the disk surfaces, the controller, and more, each of which has yet other components. By using an encapsulated CPD, we can decouple the model of the computer from the detailed model of the hard drive. We need only specify which global aspects of the computer state the hard drive behavior depends on, and which it inﬂuences. Furthermore, we can hierarchically compose encapsulated CPDs, modeling, in turn, the hard drive’s behavior in terms of its yet-lower-level components. 

In ﬁgure 5.15 we show a simple hierarchical model for a computer system. This high-level model for a computer, ﬁgure 5.15a, uses encapsulated CPDs for Power-Source , Motherboard , Hard- Drive , Printer , and more. The Hard-Drive CPD has inputs Temperature , Age and OS-Status , and the outputs Status and Full . Although the hard drive has a rich internal state, the only aspects of its state that inﬂuence objects outside the hard drive are whether it is working properly and whether it is full. The Temperature input of the hard drive in a computer is outside the probabilistic model and will be mapped to the Temperature parent of the Hard-Drive variable in the computer model. A similar mapping happens for other inputs. 

The Hard-Drive encapsulated network, ﬁgure 5.15b, in turn uses encapsulated CPDs for Controller , Surface , Drive-Mechanism , and more. The hierarchy can continue as necessary. In this case, the model for the variable Motor (in the Drive-Mechanism ) is “simple,” in that none of its CPDs are encapsulated. 

One obvious observation that can be derived from looking at this example is that an encapsulated CPD is often appropriate for more than one variable in the model. For example, the encapsulated CPD for the variable Surface1 in the hard drive is almost certainly the same as the CPDs for the variables Surface2 , Surface2 , and Surface4 . Thus, we can imagine creating a template of an encapsulated CPD, and reusing it multiple times, for several variables in the model. This idea forms the basis for a framework known as object-oriented Bayesian networks . 
## 5.7 Summary 
In this chapter, we have shown that our ability to represent structure in the distribution does not end at the level of the graph. In many cases, here is important structure within the CPDs that we wish to make explicit. 

In particular, we discussed several important types of discrete structured CPDs. 

- deterministic functions;
- asymmetric, or context speciﬁc, dependencies;
- cases where diferent inﬂuences combine independently within the CPD, including noisy-or, logistic functions, and more. 

In many cases, we showed that the additional structure provides not only a more compact parameter iz ation, but also additional independencies that are not visible at the level of the original graph. 
> 本章讨论了离散结构 CPD 的重要类型：
> - 确定性函数
> - 不对称 or 针对上下文
> - 不同影响独立地在 CDP 中结合，包括噪声或、logistic function

As we discussed, the idea of structured CPDs is critical in the case of continuous variables, where a table-based representation is clearly irrelevant. We discussed various representations for CPDs in hybrid (discrete/continuous) networks, of which the most common is the linear Gaussian representation. For this case, we showed some important connections between the linear Gaussian representation and multivariate Gaussian distributions. 
> 还讨论了混合网络中的 CPD 表示，最常见的是线性高斯表示

Finally, we discussed the notion of a conditional Bayesian network, which allows us to decompose a conditional probability distribution recursively, into another Bayesian network. 
> 条件贝叶斯网络：允许我们将条件概率分布递归地分解为另一个贝叶斯网络
# 6 Template-Based Representations 
## 6.1 Introduction 
A probabilistic graphical model (whether a Bayesian network or a Markov network) specifies a joint distribution over a fixed set $\mathcal{X}$ of random variables. This fixed distribution is then used in a variety of diferent situations. For example, a network for medical diagnosis can be applied to multiple patients, each with diferent symptoms and diseases. However, in this example, the diferent situations to which the network is applied all share the same general structure — all patients can be described by the same set of attributes, only the attributes’ values difer across patients. We call this type of model variable-based , since the focus of the representation is a set of random variables. 
> 一个概率图模型指定了随机变量集合 $\mathcal X$ 上的一个联合分布，该分布的对象也限制在随机变量集合 $\mathcal X$ 上，我们称这种模型是基于变量的，因为表示聚焦于随机变量集合

In many domains, however, the probabilistic model relates to a much more complex space than can be encoded as a fixed set of variables. In a temporal setting, we wish to represent distributions over systems whose state changes over time. For example, we may be monitoring a patient in an intensive care unit. In this setting, we obtain sensor readings at regular intervals — heart rate, blood pressure, EKG — and are interested in tracking the patient’s state over time. As another example, we may be interested in tracking a robot’s location as it moves in the world and gathers observations. Here, we want a single model to apply to trajectories of diferent lengths, or perhaps even infinite trajectories. 
> 在很多领域，我们往往不能将概率模型与一个固定的随机变量集合相关
> 例如在时序设定下，我们在一个状态随着时间变化的系统上表示分布，
> 又或者我们希望追踪一个机器人移动时的位置和它的观察
> 此时，我们希望对于不同的长度的轨迹，甚至无限长度的轨迹应用相同的一个模型

An even more complex setting arises in our Genetics example; here, each pedigree (family tree) consists of an entire set of individuals, all with their own properties. Our probabilistic model should encode a joint distribution over the properties of all of the family members. Clearly, we cannot define a single variable-based model that applies universally to this application: each family has a diferent family tree; the networks that represent the genetic inheritance process within the tree have diferent random variables, and diferent connectivities. Yet the mechanism by which genes are transmitted from parent to child is identical both for diferent individuals within a pedigree and across diferent pedigrees. 

In both of these examples, and in many others, we might hope to construct a single, com- pact model that provides a template for an entire class of distributions from the same type: trajectories of diferent lengths, or diferent pedigrees. In this chapter, we define representations that allow us to define distributions over richly structured spaces, consisting of multiple objects, interrelated in a variety of ways. These template-based representations have been used in two main settings. The first is temporal modeling, where the language of dynamic Bayesian networks allows us to construct a single compact model that captures the properties of the system dy- namics, and to produce distributions over diferent trajectories. The second involves domains such as the Genetics example, where we have multiple objects that are somehow related to each other. Here, various languages have been proposed that allow us to produce distributions over diferent worlds, each with its own set of individuals and set of relations between them. 
> 我们希望构建单个紧凑的模型，为一整类相同类型的分布（不同类型的轨迹、不同的血统）提供一个模板
> 本章介绍基于模板的表示，它允许我们在更丰富的，由多个相关的对象构成的结构空间之上定义分布，这类表示主要用于：时序建模（使用动态贝叶斯网络捕获系统动态，在不同的轨迹上生成不同分布）；基因分析

Once we consider higher-level representations that allow us to model objects, relations, and probabilistic statements about those entities, we open the door to very rich and expressive languages and to queries about concepts that are not even within the scope of a variable-based framework. For example, in the Genetics example, our space consists of multiple people with diferent types of relationships such as Mother , Father-of , and perhaps Married . In this type probability space, we can also express uncertainty about the identity of Michael’s father, or how many children Great-aunt Ethel had. Thus, we may wish to construct a probability distribution over a space consisting of distinct pedigree structures, which may even contain a varying set of objects. As we will see, this richer modeling language will allow us both to answer new types of queries, and to provide more informed answers to “traditional” queries. 

## 6.2 Temporal Models 
Our focus in this section is on modeling dynamic settings, where we are interested in reasoning about the state of the world as it evolves over time. We can model such settings in terms of a system state , whose value at time $t$ is a snapshot of the relevant attributes (hidden or observed) of the system at time $t$ . We assume that the system state is represented, as usual, as an assignment of value some s of random riables $\mathcal{X}$ . We use $X_{i}^{(t)}$ to represent the instantiation of the variable $X_{i}$ at time t . Note that $X_{i}$ itself is no longer a variable that takes a value; rather, it is a template variable . This template is instantiated at diferent points in time $t_{;}$ , and each $X_{i}^{(t)}$ is a variable that takes a value in $V a l(X_{i})$ . For a set of variables $X\subseteq\mathcal{X}$ , we use $X^{(t_{1}:t_{2})}\ (t_{1}<t_{2})$ to denote the set of variables $\{X^{(t)}:t\in[t_{1},t_{2}]\}$ . As usual, we use the notation $\mathbf{\boldsymbol{x}}^{(t:t^{\prime})}$ for an assignment of values to this set of variables. 
> 在动态设定下，我们对随着时间变化的状态进行分析，我们将其建模为一个系统状态，它在时间 $t$ 的值是系统此时相关属性（观察到的或隐藏的）的一个快照
> 我们假设系统状态通过对于一个随机变量集合 $\mathcal X$ 的一个赋值表示，我们使用 $X_i^{(t)}$ 表示随机变量 $X_i$ 在 $t$ 时的实例
> 注意，$X_i$ 本身已经不是一个直接取某个值的随机变量，而是一个模板变量，该模板在特定的时间点 $t$ 实例化，而每个实例 $X_i^{(t)}$ 是一个取 $Val (X_i)$ 中的某个值的随机变量
> 我们使用 $X^{(t_1:t_2)}$ 表示随机变量集合 $\{X^{(t)}: t\in [t_1, t_2]\}$，使用 $\pmb x^{(t: t')}$ 表示赋值

Each “possible world” in our probability space is now a trajectory : an assignment of values to each variable $X_{i}^{(t)}$ for each relevant time $t$ . Our goal therefore is to represent a joint distribution over such trajectories. Clearly, the space of possible trajectories is a very complex probability space, so representing such a distribution can be very difcult. We therefore make a series of simplifying assumptions that help make this representational problem more tractable. 
> 在我们的概率空间中，此时对于一个系统的观测实际是一个轨迹，包含了每个相关时刻 $t$ 上对于每个变量 $X_i^{(t)}$ 的赋值
> 我们需要表达该轨迹上的联合分布（概率空间是所有可能的轨迹），为此，需要做出一系列简化的假设

Example 6.1
Consider a vehicle localization task, where a moving car tries to track its current location using the data obtained from a, possibly faulty, sensor. The system state can be encoded (very simply) using the: Location — the car’s current location, Velocity — the car’s current velocity, Weather — the current weather, Failure — the failure status of the sensor, and Obs — the current observation. We have one such set of variables for every point t . 
A joint probability distribution over all of these sets defines a probability distribution over trajectories of the car. Using this distribution, we can answer a variety queries, such as: Given a sequence of observations about the car, where is it now? Where is it likely to be in ten minutes? Did it stop at the red light? 

### 6.2.1 Basic Assumptions 
Our first simplification is to discretize the timeline into a set of time slices : measurements of the system state taken at intervals that are regularly spaced with a predetermined time granularity $\Delta$ . Thus, we can now restrict our set of random variables to $\mathcal{X}^{(\bar{0})},\mathcal{X}^{(1)},...,$ where $\mathcal{X}^{(t)}$ are the ground random variables that represent the system state at time $t\!\cdot\!\Delta$ . For example, in the patient monitoring example, we might be interested in monitoring the patient’s state every second, so that $\Delta\,=\,1s e c$ . This assumption simplifies our problem from representing distributions over a continuum of random variables to representing distributions over countably many random variables, sampled at discrete intervals. 
> 第一个简化：将时间线离散化为时间片段集合
> 此时对于系统状态的度量按照规律的间隔观测，间隔是预定义的时间粒度 $\Delta$
> 此时的随机变量集合记为 $\mathcal X^{(0)},\dots, \mathcal X^{(t)}$，其中 $\mathcal X^{(t)}$ 即系统在时间 $t\cdot \Delta$ 时的基础随机变量集

Consider a distribution over trajectories sampled over a prefix of time $t\;=\;0,.\,.\,.\,,T\;-$ $P(\mathcal{X}^{(0)},\mathcal{X}^{(1)},\ldots,\mathcal{X}^{(T)})$ , often abbreviated $P(\bar{\mathcal{X}}^{(0:T)})$ . We can reparameterize the distribution using the chain rule for probabilities, in a direction consistent with time: 

$$
P(\mathcal{X}^{(0:T)})=P(\mathcal{X}^{(0)})\prod_{t=0}^{T-1}P(\mathcal{X}^{(t+1)}\mid\mathcal{X}^{(0:t)}).
$$

Thus, the distribution over trajectories is the product of conditional distributions, for the variables in each time slice given the preceding ones. 
> 考虑时间 $t=0,\dots, T$ 上的轨迹上的分布 $P (\mathcal X^{(0)}, \mathcal X^{(1)}, \dots, \mathcal X^{(T)})$，一般简写为 $P (\mathcal X^{(0:T)})$
> 我们使用链式法则，将该联合分布重参数化，方向沿着时间方向：

$$
P(\mathcal X^{(0:T)}) = P(\mathcal X^{(0)})\prod_{t=0}^{T-1}P(\mathcal X^{(t+1)}\mid \mathcal X^{(0:t)}).
$$

> 此时，我们将轨迹上的分布分解为了条件概率分布（给定先前的时间片的变量，当前时间片的变量从属于的分布）的乘积

We can considerably simplify this formulation by using our usual tool — conditional independence assumptions.  One very natural approach is to assume that the future is conditionally independent of the past given the present: 
> 显然，我们可以利用条件独立性假设简化该乘积，一种非常自然的方法就是假设在给定当下的条件下，未来条件独立于过去

**Definition 6.1** Markov assumption 
We hat a dynamic system over the template variables $\mathcal{X}$ satisfies the Markov assumption if, for all $t\geq0$ , 

$$
(\mathcal{X}^{(t+1)}\perp\mathcal{X}^{(0:(t-1))}\mid\mathcal{X}^{(t)}).
$$ 
Markov chain system Such systems are called Markov chain . 
> 定义：Markov assumption
> 对于模板变量 $\mathcal X$ 上的一个动态系统，如果对于所有的 $t\ge 0$，有

$$
(\mathcal X^{(t+1)}\perp \mathcal X^{(0:(t-1))}\mid \mathcal X^{(t)})
$$

> 我们称该系统满足 Markov 假设，并且该系统称为 Markov chain

Th kov a ions states that the variables in $\mathcal{X}^{(t+1)}$ cannot depend directly on variables in $\mathcal{X}^{(t^{\prime})}$ for t $t^{\prime}~<~t$ . If we were to draw our dependency model as an (infinite) Bayesian network, the Markov assumption would correspond to the constraint on the graph that there are no edges into $\mathcal{X}^{(t+1)}$ from variables in time slices $t\mathrm{~-~}1$ or earlier. 
> Markov 假设表明 $\mathcal X^{(t+1)}$ 中的变量不能直接依赖于 $\mathcal X^{(t')} (t' < t)$ 中的变量
> 如果要将我们的依赖模型表现为一个无限的 Bayesian 网络，Markov 假设对应于图中时间片 $t-1$ 以及以前的变量没有直接到 $\mathcal X^{(t+1)}$ 的边

Like many other conditional independence assumptions, the Markov assumption allows us to define a more compact representation of the distribution: 
> Markov 假设帮助我们简化了轨迹的联合分布表示

$$
P(\mathcal{X}^{(0)},\mathcal{X}^{(1)},\ldots,\mathcal{X}^{(T)})=P(\mathcal{X}^{(0)})\prod_{t=0}^{T-1}P(\mathcal{X}^{(t+1)}\mid\mathcal{X}^{(t)}).\tag{6.1}
$$

Like any conditional independence assumption, the Markov assumption may or may not be reasonable in a particular setting. 
> 在特定的设置下，Markov 假设可以是合理的，也可以是不合理的

Example 6.2 
time $t$ , because the previous locations give us information about the object’s direction of motion and speed. By adding Velocity, we make the Markov assumption closer to being satisfied. If, however, the driver is more likely to accelerate and decelerate sharply in certain types of weather (say heavy winds), then our $V,L$ model does not satisfy the Markov assumption relative to $V$ ; we can, again, make the model more Markovian by adding the Weather variable. Finally, in many cases, a sensor failure at one point is usually accompanied with a sensor failure at nearby time points, rendering nearby Obs variables correlated. By adding all of these variables into our state model, we define a state space whereby the Markov assumption is arguably a reasonable approximation. 
> 该例讲述了如何通过添加变量使得状态空间中 Markov 假设是近似成立的

Philosophically, one might argue whether, given a sufciently rich description of the world state, the past is independent of the future given the present. However, that question is not central to the use of the Markov assumption in practice. Rather, **we need only consider whether the Markov assumption is a sufciently reasonable approximation to the dependencies in our distribution. In most cases, if we use a reasonably rich state description, the approximation is quite reasonable.** In other cases, we can also define models that are semi-Markov , where the independence assumption is relaxed (see exercise 6.1). 
> 不考虑哲学，在实践中，我们仅仅需要考虑 Markov 假设是否对于我们分布中的依赖是一个合理的近似，在大多数情况下，如果我们使用丰富的状态描述，该近似是十分合理的
> 我们也可以定义半 Markov 的模型，松弛独立性假设

Because the process can continue indefinitely, equation (6.1) still leaves us with the task of acquiring an infinite set of conditional distributions, or a very large one, in the case of finite-horizon processes. Therefore, we usually make one last simplifying assumption: 
> 因为该过程会随着时间无限前进，(6.1) 中将会包含无限个条件概率分布，因此，我们需要再做出一个简化假设

**Definition 6.2**  stationary dynamical system
We say that a Markovian dynamic system is stationary (also called time invariant or homogeneous ) if $P(\dot{\mathcal{X}}^{(t+1)}\mid\mathcal{X}^{(t)})$ is the same for all $t$ . is case, we can represent the process using a transition model $P(\mathcal{X}^{\prime}\mid\mathcal{X})$ , so that, for any $t\geq0$ , 

$$
P(\mathcal{X}^{(t+1)}=\xi^{\prime}\mid\mathcal{X}^{(t)}=\xi)=\ P(\mathcal{X}^{\prime}=\xi^{\prime}\mid\mathcal{X}=\xi).
$$

> 定义：
> 如果 $P (\mathcal X^{(t+1)}\mid \mathcal X^{(t)})$ 对于全部的 $t$ 都相同，我们称 Markov chain 动态系统是平稳的/时间不变的/同质的
> 此时，我们用一个转移模型 $P (\mathcal X'\mid \mathcal X)$ 就可以表示 Markov 过程

### 6.2.2 Dynamic Bayesian Networks 
The Markov and stationarity assumptions described in the previous section allow us to represent the probability distribution over infinite trajectories very compactly: We need only represent the initial state distribution and the transition model $P(\mathcal{X}^{\prime}\mid\mathcal{X})$ . This transition model is a conditional probability distribution, which we can represent using a conditional Bayesian network, as described in section 5.6. 
> Markov 和平稳假设允许我们紧凑表示无限轨迹上的概率分布：我们仅需要表示初始分布和转移模型 $P (\mathcal X' \mid \mathcal X)$
> 转移模型是 CPD，我们可以用条件贝叶斯网络表示

Example 6.3 
Let us return to the setting of example 6.1. Here, we might want to represent the system dynamics using the model shown in figure 6.1a, the current observation depends on the car’s location (and the map, which is not explicitly modeled) and on the error status of the sensor. Bad weather makes the sensor more likely to fail. And the car’s location depends on the previous position and the velocity. All of the variables are interface variables except for Obs, since we assume that the sensor observation is generated at each time point independently given the other variables. 

This type of conditional Bayesian network is called a 2-time-slice Bayesian network (2-TBN) . 

![[Graphical Probabilistic Theory-Fig6.1.png]]


**Definition 6.3** 2-TBN interface variable 
A 2-time-slice Bayesian network (2-TBN) for a process over X is a conditional Bayesian network 2-TBN over X0 given XI, where XI ⊆ X is a set of interface variables.
> 定义： 2-时间片贝叶斯网络
> 对于 $\mathcal X$ 上的过程的 2时间片贝叶斯网络定义为在 $\mathcal X'$ 上的给定 $\mathcal X_I$ 的条件贝叶斯网络，其中 $\mathcal X_I \subseteq \mathcal X$ 是接口变量的集合（接口变量在不同的时间片上有不同的值，它们在时间上的延续使得我们可以从一个时间点推断到另一个时间点，因此它们是不同时间片之间的桥梁）
 
As a reminder, in a co tional Bayesian network, only the variabl $\mathcal{X}^{\prime}$ have parents or CPDs. 
> 注意条件贝叶斯网络仅定义了 $\mathcal X'$ 的条件概率分布，以及也仅 $\mathcal X'$ 有父变量
 
The interface var $\mathcal{X}_{I}$ are those variables whos alues at time t have a direct ef t on the variables at time $t+1$ . Thus nly the variables in X $\mathcal{X}_{I}$ can be parents of variables in X $\mathcal{X}^{\prime}$ . In our example, all variables except O are in the interface. 
> 接口变量即它们在时间 $t$ 的值会影响时间 $t+1$ 的变量值的变量，因此它们会作为2-TBN 中的父变量，非接口变量不会影响到 $\mathcal X'$

Overall, the 2-TBN represents the conditional distribution: 

$$
P(\mathcal{X}^{\prime}\mid\mathcal{X})=P(\mathcal{X}^{\prime}\mid\mathcal{X}_{I})=\prod_{i=1}^{n}P(X_{i}^{\prime}\mid\mathrm{Pa}_{X_{i}^{\prime}}).
$$ 
For each template variable $X_{i}$ , the CPD $P(X_{i}^{\prime}\mid\mathrm{Pa}_{X_{i}^{\prime}})$ is a template factor : it will be instantiated multiple times within the model, for multiple variables $X_{i}^{(t)}$ (and their parents). 

> 2-TBN 表示了 CPD：

$$
P(\mathcal X'\mid \mathcal X) = P(\mathcal X'\mid \mathcal X_I) = \prod_{i=1}^n P(X_i'\mid \text{Pa}_{X_i'})
$$

> 对于 $\mathcal X'$ 中的每个模板变量 $X_i'$，CPD $P (X_i' \mid \text{Pa}_{X_i'})$ 都是一个模板因子，在不同的时间点上，它会被实例化多次，实例化为 $X_i^{(t)}$ 和它们的父变量

Perhaps the simplest nontrivial example of a temporal model of this kind is the hidden Markov model (see section 6.2.3.1). It has a single state variable $S$ and a single observation variable $O$ . Viewed as a DBN, an HMM has the structure shown in figure 6.2. 
> 这类时序模型最简单的例子是隐 Markov 模型，HMM 有单个状态变量 $S$ 和单个观察变量 $O$

Example 6.4 
Consider a robot moving around in a grid. Most simply, the robot is the only aspect of the world that is changing, so that the state of the system $S$ is simply the robot’s position. Our transition model $P(S^{\prime}\mid S)$ then repre nts the probability that, if the robot is in some state (position) $s$ , it will move to another state s $s^{\prime}$ . Our task is to keep track of the robot’s location, using a noisy sensor (for example, a sonar) whose value depends on the robot’s location. The observation model $P(O\mid S)$ tells us the probability of making a particular sensor reading o given that the robot’s current position is $s$ . (See box 5. E for more details on the state transition and observation models in a real robot localization task.) 

In a 2-TBN, some of the edges are inter-time-slice edges , going between time slices, whereas others are intra-time-slice edges , connecting variables in the same time slice. Intuitively, our decision of how to relate two variables depends on how tight the coupling is between them. If the efect of one variable on the other is immediate — much shorter than the time granularity in the model — the inﬂuence would manifest (roughly) within a time slice. If the efect is slightly longer-term, the inﬂuence manifests from one time slice to the next. In our simple examples, the efect on the observations is almost immediate, and hence is modeled as an intra-time-slice edge, whereas other dependencies are inter-time-slice. In other examples, when time slices have a coarser granularity, more efects might be short relative to the length of the time slice, and so we might have other dependencies that are intra-time-slice. 
> 二时间片贝叶斯网络中，一些边属于时间片间边，即连接了两个时间片的节点，一些边属于时间片内边，连接了相同时间片内的节点
> 我们构建网络时，如何关联两个变量应取决于它们之间的联系有多紧密，如果二者之间的影响是直接的——比时间粒度更短，则影响可以在时间片内表示；如果影响是长期的，则可以在时间片之间表示

Many of the inter-time-slice edges are of the form $X\rightarrow X^{\prime}$ . Such edges are called persistence edges , and they represent the tendency of the variable X (for example, sensor failure) to persist over time with high probability. A variable $X$ for which we have an edge $X\rightarrow X^{\prime}$ in the 2-TBN is called a persistent variable . 
> 许多时间片间的边的形式是 $X\rightarrow X'$，这类边被称为持续边，表示变量 $X$ 以较高的概率随着时间持续的趋势
> 在 2-TBN 中，我们称具有形式为 $X\rightarrow X'$ 的变量为持续变量

Based on the stationarity property, a 2-TBN defines the probability distribution $P(\mathcal{X}^{(t+1)}\mid$ $\mathcal{X}^{(t)}$ ) for any $t$ . Given a distribution over the initial states, we can unroll the network over sequences of any length, to define a Bayesian network that induces a distribution over trajectories of that length. In these networks, all the copies of the variable $X_{i}^{(t)}$ for $t>0$ have the same dependency structure and the same CPD. Figure 6.1 demonstrates a transition model, initial state network, and a resulting unrolled DBN, for our car example. 
> 如果平稳性质成立，一个 2-TBN 实际上为任意 $t$ 定义了概率分布 $P (\mathcal X^{(t+1)}\mid \mathcal X^{(t)})$，此时，给定初始状态上的分布，我们可以展开任意长度的序列上的网络，得到针对该长度的轨迹上的贝叶斯网络
> 该网络中，所有 $X_i^{(t)}, t> 0$ 具有相同的依赖结构和相同的 CPD

**Definition 6.4** dynamic Bayesian network
$A$ dynamic Bayesian network (DBN) is a pair $\langle\mathcal{B}_{0},\mathcal{B}_{\rightarrow}\rangle$ , where ${\mathcal B}_{0}$ is a Bayesian network over $\mathcal{X}^{(0)}$ , representing the initial distribution over and $\mathcal{B}_{\rightarrow}$ is a 2-TBN for the process. For any desired time span $T\geq0$ , the distribution over X $\mathcal{X}^{(0:T)}$ is defined as $a$ unrolled Bayesian network , where, for any $i=1,\dots,n$ : 

• the structure and CPDs of X(0) i are the same as those for Xi in B0,
• the structure and CPD of X(t) i for t > 0 are the same as those for Xi0 in B!.

> 定义：
> 动态贝叶斯网络定义为 $\langle \mathcal B_0, \mathcal B_{\rightarrow} \rangle$，其中 $\mathcal B_0$ 是 $\mathcal X^{(0)}$ 上的贝叶斯网络，表示状态的初始分布，$\mathcal B_{\rightarrow}$ 是一个 2时间片贝叶斯网络
> 对于任意时间片 $T\ge 0$，$\mathcal X^{(0:T)}$ 上的分布定义为一个展开的贝叶斯网络，其中，对于任意 $i = 1,\dots , n$，满足：
> - $X_i^{(0)}$ 的结构和 CPDs 和 $X_i$ 在 $\mathcal B_0$ 中的结构和 CPDs 相同
> - $X_i^{(t)}, t> 0$ 的结构和 CPD 和 $X_i'$ 在 $\mathcal B_{\rightarrow}$ 中的结构和 CPD 相同

Thus, we can view a DBN as a compact representation from which we can generate an infinite set of Bayesian networks (one for every $T>0$ ). 
> 因此，我们可以将 DBN 视为可以生成无限个贝叶斯网络的紧凑表示

![[Graphical Probabilistic Theory-Fig6.3.png]]

Figure 6.3 shows two useful classes of DBNs that are constructed from HMMs. A factorial left, is a DBN whose 2-TBN has the s cture of a set of chains $X_{i}\ \rightarrow\ X_{i}^{\prime}$ $(i=1,.\,.\,.\,,n)$ ), with a single (always) observed variable Y $Y^{\prime}$ , which is a child of all the variables $X_{i}^{\prime}$ . This type of model is very useful in a variety of applications, for example, when several sources of sound are being heard simultaneously through a single microphone.
A coupled HMM ,  on the right, is also constructed from a set of chains $X_{i}$ , but now, each chain is an HMM with its private own observation variable $Y_{i}$ . The chains now interact directly via their state variables, with each chain afecting its adjacent chains. These models are also useful in a variety of applications. For example, consider monitoring the temperature in a building over time (for example, for fire alarms). Here, $X_{i}$ might be the true (hidden) state of the i th room, and $Y_{i}$ the value returned by the room’s own temperature sensor. In this case, we would expect to have interactions between the hidden states of adjacent rooms. 
> 两类常见的动态贝叶斯网络见 Fig6.3
> 左边是因子隐马尔可夫模型，该模型中，其 2-TBN 的结构是一系列的 $X_i \rightarrow X_i' ( i = 1, \dots, n)$ ，以及单个被观察到的变量 $Y'$，$Y'$ 是所有 $X_i'$ 的子变量
> 建模 factorial HMM 的情况例如：多个声源通过单个麦克风传出
> 右边是耦合隐马尔可夫模型，其结构同样包含一系列的 $X_i \rightarrow X_i' (i = 1,\dots , n)$，不同的是此时每个 $X_i \rightarrow X_i'$ 都构成一个隐马尔可夫模型，即都具有各自的观察变量 $Y_i$；并且，此时各条链还会直接通过状态变量交互，也就是每一条链都会影响其相邻的链

In DBNs, it is often the case that our observation pattern is constant over time. That is, we can partition the varia $\mathcal{X}$ into disjoint subsets $X$ and $^o$ , such that the variables in $X^{(t)}$ are always hidden and $O^{(t)}$ are always observed. For uniformity of presentation, we generally make this assumption; however, the algorithms we present also apply to the more general case. 
> DBN 建模中，我们的观察模式往往对于时间是不变的，也就是说，我们可以将 $\mathcal X$ 划分为两个不相交的集合 $X, O$，其中变量 $X^{(t)}$ 总是隐变量，变量 $O^{(t)}$ 总是被观察的变量
> 这是我们常常会做的假设

A DBN can enable fairly sophisticated reasoning patterns. 

Example 6.5
By explicitly encoding sensor failure, we allow the agent to reach the conclusion that the sensor has failed. Thus, for example, if we suddenly get a reading that tells us something unexpected, for example, the car is suddenly 15 feet to the left of where we thought it was 0.1 seconds ago, then in addition to considering the option that the car has suddenly teleported, we will also consider the option that the sensor has simply failed. Note that the model only considers options that are built into it. If we had no “sensor failure” variable, and had the sensor reading depend only on the current location, then the diferent sensor readings would be independent given the car’s trajectory, so that there would be no way to explain correlations of unexpected sensor readings except via the trajectory. Similarly, if the system knows (perhaps from a weather report or from prior observations) that it is raining, it will expect the sensor to be less accurate, and therefore be less likely to believe that the car is out of position. 

Box 6. A — Case Study: HMMs and Phylo-HMMs for Gene Finding. HMMs are a primary tool in algorithms that extract information from biological sequences. Key applications (among many) include: modeling families of related proteins within and between organisms, finding genes in DNA sequences, and modeling the correlation structure of the genetic variation between individuals in a population. We describe the second of these applications, as an illustration of the methods used. 

The DNA of an organism is composed of two paired helical strands consisting of a long sequence of nucleotides, each of which can take on one of four values — A, C, G, T; in the double helix structure, A is paired with T and C is paired with G, to form a base pair. The DNA sequence consists of multiple regions that can play diferent roles. Some regions are genes, whose DNA is transcribed into mRNA, some of which is subsequently translated into protein. In the translation process, triplets of base pairs, known as codons, are converted into amino acids. There are $4^{3}\,=\,64$ diferent codons, but only 20 diferent amino acids, so that the code is redundant. Not all transcribed regions are necessarily translated. Genes can contain exons, which are translated, and introns, which are spliced out during the translation process. The DNA thus consists of multiple genes that are separated by intergenic regions; and genes are themselves structured, consisting of multiple exons separated by introns. The sequences in each of these regions is characterized by certain statistical properties; for example, a region that produces protein has a very regular codon structure, where the codon triplets exhibit the usage statistics of the amino acids they produce. Moreover, boundaries between these regions are also often demarcated with sequence elements that help the cell determine where transcription should begin and end, and where translation ought to begin and end. Nevertheless, the signals in the sequence are not always clear, and therefore identifying the relevant sequence units (genes, exons, and more) is a difcult task. 

HMMs are a critical tool in this analysis. Here, we have a hidden state variable for each base pair, which denotes the type of region to which this base pair belongs. To satisfy the Markov assumption, one generally needs to refine the state space. For example, to capture the codon structure, we generally include diferent hidden states for the first, second, and third base pairs within a codon. This larger state space allows us to encode the fact that coding regions are sequences of triplets of base pairs, as well as encode the diferent statistical properties of these three positions. We can further refine the state space to include diferent statistics for codons in the first exon and in the last exon in the gene, which can exhibit diferent characteristics than exons in the middle of the gene. The observed state of the HMM naturally includes the base pair itself, with the observation model reﬂecting the diferent statistics of the nucleotide composition of the diferent regions. It can also include other forms of evidence, such as the extent to which measurements of mRNA taken from the cell have suggested that a particular region is transcribed. And, very importantly, it can contain evidence regarding the conservation of a base pair across other species. This last key piece of evidence derives from the fact that base pairs that play a functional role in the cell, such as those that code for protein, are much more likely to be conserved across related species; base pairs that are nonfunctional, such as most of those in the intergenic regions, evolve much more rapidly, since they are not subject to selective pressure. Thus, we can use conservation as evidence regarding the role of a particular base pair. 

One way of incorporating the evolutionary model more explicitly into the model is via a phylo- genetic HMM (of which we now present a simplified version). Here, we encode not a single DNA sequence, but the sequences of an entire phylogeny (or evolutionary tree) of related species. We let $X_{k,i}$ be the i th nucleotide for species $s_{k}$ . We also introduce a species-independent variable $Y_{i}$ denoting the functional role of the i th base pair (intergenic, intron, and so on). The base pair $X_{k,i}$ will depend on the corresponding base pair $X_{\ell,i}$ where $s_{\ell}$ is the ancestral species from which $s_{k}$ evolved. The parameters of this dependency will depend on the evolutionary distance between $s_{k}$ and $s_{l}$ (the extent to which $s_{k}$ has diverged) and on the rate at which a base pair playing a particular role evolves. For example, as we mentioned, a base pair in an intergenic region generally evolves much faster than one in a coding region. Moreover, the base pair in the third position in a codon also often evolves more rapidly, since this position encodes most of the redundancy between codons and amino acids, and so allows evolution without changing the amino acid composition. Thus, overall, we define $X_{k,i}\,\acute{s}$ parents in the model to be $Y_{i}$ (the type of region in which $X_{k,i}$ resides), $X_{k,i-1}$ (the previous nucleotide in species $s$ ) and $X_{\ell,i}$ (the i th nucleotide in the parent species $s_{\ell}$ ). This model captures both the correlations in the functional roles (as in the simple gene finding model) and the fact that evolution of a particular base pair can depend on the adjacent base pairs. This model allows us to combine information from multiple species in order to infer which are the regions that are functional, and to suggest a segmentation of the sequence into its constituent units. 

Overall, the structure of this model is roughly a set of trees connected by chains: For each i we have a tree over the variables $\{X_{k,i}\}_{s_{k}}$ , where the structure of the tree is that of the evolutionary tree; in addition, all of the $X_{k,i}$ are connected by chains to $X_{k,i+1}$ ; finally, we also have the variables $Y_{i}$ , which also form a chain and are parents of all of the $X_{k,i}$ . Unfortunately, the structure of this model is highly intractable for inference, and requires the use of approximate inference methods; see exercise 11.29. 

### 6.2.3 State-Observation Models 
An alternative way of thinking about a temporal process is as a state-observation model . In a state- observation model, we view the system as evolving naturally on its own, with our observations of it occurring in a separate process. This view separates out the system dynamics from our observation model, allowing us to consider each of them separately. It is particularly useful when our observations are obtained from a (usually noisy) sensor, so that it makes sense to model separately the dynamics of the system and our ability to sense it. 
> 另一种思考时序过程的方式是状态-观察模型
> 在状态-观察模型中，我们将模型视为自己自然演变，我们的观察发生在另一个过程中，因此，我们的观察模型和系统动态被分离开了，故我们可以分别独立考虑系统动态和观察
> 该模型在我们的观察具有噪声的情况下非常有用，此时分别建模系统动态和我们观察它的能力就十分合理

A state-observation model utilizes two independence assumptions: that the state variables evolve in a Markovian way, so that 

$$
(X^{(t+1)}\perp X^{(0:(t-1))}\mid X^{(t)});
$$

and that the observation variables at time $t$ are conditionally independent of the entire state sequence given the state variables at time $t$ : 

$$
(O^{(t)}\perp X^{(0:(t-1))},X^{(t+1:\infty)}\mid X^{(t)}).
$$

> 状态-观察模型采用了两个独立性假设：
> 1. 状态变量马尔可夫式演变，即 $(X^{(t+1)}\perp X^{(0: (t-1))}\mid X^{(t)})$
> 2. 给定 $t$ 时刻的状态变量，$t$ 时刻的观察变量条件独立于整个状态序列，即 $(O^{(t)}\perp X^{(0: (t-1))}, X^{(t+1:\infty)}\mid X^{(t)})$

We now view our probabilistic model as consisting of two components: the transition model , $P(X^{\prime}\mid X)$ , and the observation model , $P(O\mid X)$ . 
> 现在，我们将我们的概率模型视为由两个成分构成：转移模型 $P (X'\mid X)$ 和观察模型 $P (O\mid X)$

From the perspective of DBNs, this type of model corresponds to a 2-TBN structure where the observation variables $O^{\prime}$ are all leaves, and have parents only in $X^{\prime}$ . This type of situation arises quite naturally in a variety of real-world systems, where we do not have direct observations of the system state, but only access to a set of (generally noisy) sensors that depend on the state. The sensor observations do not directly efect the system dynamics, and therefore are naturally viewed as leaves. 
> 从 DBN 的角度来看，状态-观察模型对应于 2-TBN 结构中观察变量 $O'$ 都是叶子，且仅有在 $X'$ 的父变量
> 这类情况在现实系统中十分常见，我们往往不能直接观察系统状态，而是只能依访问依赖于系统状态（通常带噪声的）的传感器
> 而传感器的观察并不会直接影响系统动态，因此很自然地可以视为叶子

We note that we can convert any 2-TBN to a state-observation representation as follows: For any observed variable $Y$ that does not already satisfy the structural restrictions, we introduce a new variable $\tilde{Y}$ whose only parent is $Y$ , and that is deterministic ally equal to $Y$ . Then, we view $Y$ as being hidden, and we interpret our observations of $Y$ as observations on $\tilde{Y}$ . In efect, we construct $\tilde{Y}$ to be a perfectly reliable sensor of $Y$ . Note, however, that, while the resulting transformed network is probabilistic ally equivalent to the original, it does obscure structural independence properties of the network (for example, various independencies given that $Y$ is observed), which are now apparent only if we account for the deterministic dependency between $Y$ and $\tilde{Y}$ . 
> 我们可以将任意 2-TBN 转化为状态-观察模型：
> 对于任意不满足假设地观察到的变量 $Y$，我们引入新变量 $\tilde Y$ 作为其子变量，$\tilde Y$ 确定性地等于 $Y$
> 此时，我们可以将 $Y$ 视为隐变量，将对于 $Y$ 的观察解释为对于 $\tilde Y$ 的观察
> 这样得到的模型和原模型在概率上等价，但考虑独立性关系时需要更加小心，例如原模型中可能在 $Y$ 被观察到时存在条件独立性，现在我们应该将其解释为 $\tilde Y$ 被观察到，且考虑 $Y,\tilde Y$ 之间的确定性关系时，条件独立性存在

It is often convenient to view a temporal system as a state-observation model, both because it lends a certain uniformity of notation to a range of diferent systems, and because the state transition and observation models often induce diferent computational operations, and it is convenient to consider them separately. 

State-observation models encompass two important architectures that have been used in a wide variety of applications: hidden Markov models and linear dynamical systems. We now brieﬂy describe each of them. 
> 时序系统常常可以视为状态-观察模型，以保持符号一致性，并方便分别考虑对于状态转移模型和观察模型的计算
> 状态-观察模型有两个广泛使用的结构：HMM 和线性动态系统

#### 6.2.3.1 Hidden Markov Models 
A hidden Markov model , which we illustrated in figure 6.2, is the simplest example of a state- observation model. While an HMM is a special case of a simple DBN, it is often used to encode structure that is left implicit in the DBN representation. 
Specifically, the transition model $P(S^{\prime}\mid S)$ in an HMM is often assumed to be sparse, with many of the possible transitions having zero probability. In such cases, HMMs are often represented using a diferent graphical notation, which visualizes this sparse transition model. In this representation, the HMM transition model is encoded using a directed (generally cyclic) graph, whose nodes represent the diferent states of the system, that is, the values in $V a l(S)$ . We have a directed arc from $s$ to $s^{\prime}$ if it is possible to transition from $s$ to $s^{\prime}$ — that is, $P(s^{\prime}\mid s)>0$ . The edge from $s$ to $s^{\prime}$ can also be annotated with its associated transition probability $P(s^{\prime}\mid s)$ . 
> HMM 中的转移模型 $P (S'\mid S)$ 常常假设为稀疏的，即许多转移的概率是0
> 此时，HMM 可以采用另一类图表示，以可视化该稀疏转移模型
> 我们使用有向（通常有环）的图表示 HMM 转移模型，其中的节点表示系统的不同状态，也就是 $Val (S)$ 中的值，如果有从 $P (s' \mid s) > 0$，状态 $s$ 和 $s'$ 相连

Example 6.6
Consider an HMM with a state variable S that takes 4 values s1; s2; s3; s4, and with a transition where the rows correspond to states s and the columns to successor states $s^{\prime}$ (so that each row must sum to 1 ). The transition graph for this model is shown in figure 6.4. 

Importantly, the transition graph for an HMM is a very diferent entity from the graph encoding a graphical model. Here, the nodes in the graph are state , or possible values of the state variable; the directed edges represent possible transitions between the states, or entries in the CPD that have nonzero probability. Thus, the weights of the edges leaving a node must sum to 1. This graph representation can also be viewed as probabilistic finite-state automaton . Note that this graph-based representation does not encode the observation model of the HMM. In some cases, the observation model is deterministic, in that, for each $s$ , there is a single observation $O$ for which $P(o\mid s)=1$ (although the same observation can arise in multiple states). In this case, the observation is often annotated on the node associated with the state. 
> HMM 的状态转移模型的图表示可以视作一个概率有限状态自动机，其中每个节点的出边上的概率和应该是 1
> 该图表示没有编码 HMM 的观察模型
> 一些情况下，观察模型是确定的，即给定状态 $s$，具有确定的观察 $o$ ($P (o \mid s) = 1$)，此时可以直接在图上节点旁边标出观察 $o$

It turns out that HMMs, despite their simplicity, are an extremely useful architecture. For example, they are the primary architecture for speech recognition systems (see box 6. B) and for many problems related to analysis of biological sequences (see, for example, box 6. A). Moreover, these applications and others have inspired a variety of valuable generalizations of the basic HMM framework (see, for example, Exercises 6.2–6.5).  
> HMM 是语音识别系统的首选

Box 6. B — Case Study: HMMs for Speech Recognition. Hidden Markov models are currently the key technology in all speech- recognition systems. The HMM for speech is composed of three distinct layers: the language model , which generates sentences as sequences of words; the word model , where words are described as a sequence of phonemes; and the acoustic model , which shows the progression of the acoustic signal through a phoneme.  At the highest level, the language model represents a probability distribution over sequences of words in the language. Most simply, one can use $^a$ bigram model , which is a Markov model over words, defined via a probability distribution $P(W_{i}\mid W_{i-1})$ for each position i in the sentence. We can view this model as a Markov model where the state is the current word in the sequence. (Note that this model does not take into account the actual position in the sentence, so that $P(W_{i}\mid W_{i-1})$ is the same for all $i>1.$ .) $A$ somewhat richer model is the trigram model , where the states correspond to pairs of successive words in the sentence, so that our model defines a probability distribution $P(W_{i}\mid W_{i-1},W_{i-2})$ . Both of these distributions define a ridiculously naive model of language, since they only capture local correlations between neighboring words, with no attempt at modeling global coherence. Nevertheless, these models prove surprisingly hard to beat, probably because they are quite easy to train robustly from the (virtually unlimited amounts of) available training data, without the need for any manual labeling. 

The middle layer describes the composition of individual words in terms of phonemes — basic phonetic units corresponding to distinct sounds. These units vary not just on the basic sound uttered (“p” versus “b”), but also on whether the sound is breathy, aspirated, nasalized, and more. There is an international agreement on an International Phonetic Alphabet , which contains about 100 phonemes. Each word is modeled as a sequence of phonemes. Of course, a word can have multiple diferent pronunciations, in which case it corresponds to several such sequences. 

At the acoustic level, the acoustic signal is segmented into short time frames (around 10–25ms). A given phoneme lasts over a sequence of these partitions. The phoneme is also not homogenous. Diferent acoustics are associated with its beginning, its middle, and its end. We thus create an HMM for each phoneme, with its hidden variable corresponding to stages in the expression of the phoneme. HMMs for phonemes are usually quite simple, with three states, but can get more complicated, as in figure 6.B.1. The observation represents some set of features extracted from the acoustic signal; the feature vector is generally either discretized into a set of bins or treated as $^a$ continuous observation with a Gaussian or mixture of Gaussian distribution. 

Given these three models, we can put them all together to form a single huge hierarchical HMM that defines a joint probability distribution over a state space encompassing words, phonemes, and basic acoustic units. In a bigram model, the states in the space have the form $(w,i,j)$ , where $w$ is the current word, $i$ is a phoneme within that word, and $j$ is an acoustic position within that phoneme. The sequence of states corresponding to a word $w$ is governed by a word-HMM representing the distribution over pronunciations of $w$ . This word-HMM has a start-state and an end-state. When we exit from the end-state of the HMM for one word $w$ , we branch to the start-state of another word $w^{\prime}$ with probability $P(w^{\prime}\mid w)$ . Each sequence is thus a trajectory through acoustic HMMs of individual phonemes, transitioning from the end-state of one phoneme’s HMM to the start-state of the next phoneme’s HMM. 

A hierarchical HMM can be converted into a DBN, whose variables represent the states of the diferent levels of the hierarchy (word, phoneme, and intraphone state), along with some auxiliary variables to capture the “control architecture” of the hierarchical HMM; see exercise 6.5. The DBN formulation has the benefit of being a much more ﬂexible framework in which to introduce exten- sions to the model. One extension addresses the coarticulation problem , where the proximity of one phoneme changes the pronunciation of another. Thus, for example, the last phoneme in the word “don’t” sounds very diferent if the word after it is “go” or if it is “you.” Similarly, we often pronounce “going to” as “gonna.” The reason for coarticulation is the fact that a person’s speech articulators (such as the tongue or the lips) have some inertia and therefore do not always move all the way to where they are supposed to be. Within the DBN framework, we can easily solve this problem by introducing a dependency of the pronunciation model for one phoneme on the value of the preceding phoneme and the next one. Note that “previous” and “next” need to be interpreted with care: These are not the values of the phoneme variable at the previous or next states in the HMM, which are generally exactly the same as the current phoneme; rather, these are the values of the variables prior to the previous phoneme change, and following the next phoneme change. This extension gives rise to a non-Markovian model, which is more easily represented as a structured graphical model. Another extension that is facilitated by a DBN structure is the introduction of variables that denote states at which a transition between phonemes occurs. These variables can then be connected to observations that are indicative of such a transition, such as a significant change in the spectrum. Such features can also be incorporated into the standard HMM model, but it is diffcult to restrict the model so that these features affect only our beliefs in phoneme transitions. 

Finally, graphical model structure has also been used to model the structure in the Gaussian distribution over the acoustic signal features given the state. Here, two “traditional” models are: a diagonal Gaussian over the features, a model that generally loses many important correlations between the features; and a full covariance Gaussian, a model that requires many parameters and is hard to estimate from data (especially since the Gaussian is different for every state in the HMM). As we discuss in chapter 7, graphical models provide an intermediate point along the spectrum: we can use a Gaussian graphical model that captures the most important of the correlations between the features. The structure of this Gaussian can be learned from data, allowing a ﬂexible trade-off to be determined based on the available data. 

#### 6.2.3.2 Linear Dynamical Systems 
Another very useful temporal model is a linear dynamical system , which represents a system of one or more real-valued variables that evolve linearly over time, with some Gaussian noise. Such systems are also often called Kalman filters , after the algorithm used to perform tracking. 
> 另一类常用的时序模型是线性动态系统，它用于表示一个或者多个随着时间线性演化的实值变量（带有一点高斯噪声）
> 这类系统也常被称为 Kalman 滤波器

**A linear dynamical system can be viewed as a dynamic Bayesian network where the variables are all continuous and all of the dependencies are linear Gaussian.** 
> 线性动态系统可以视为一个 DBN，其中的变量都是连续变量，其中的依赖都是线性高斯

Linear dynamical systems are often used to model the dynamics of moving objects and to track their current positions given noisy measurements. (See also box 15.A.) 
> 线性动态系统常用于建模移动物体的动态，给定 noisy 的度量，追踪它们的当前位置

Example 6.7
Recall example 5.20, where we have a (vector) variable $X$ denoting a vehicle’s current position (in each relevant dimension) and $^a$ variable $V$ denoting its velocity (also in each dimension). As we discussed earlier, a first level approximation may a model where ${\cal P}(X^{\prime}\mid X,V)\;=\;$ ${\mathcal{N}}\left(X+V\Delta;\sigma_{X}^{2}\right)$  and $P(V^{\prime}\mid V)=\mathcal{N}\left(V;\sigma_{V}^{2}\right)$  (where ∆ , as before, is the length of our time slice). The observation — for example, a GPS signal measured from the car — is a noisy Gaussian measurement of $X$ . 

These systems and their extensions are at the heart of most target tracking systems, for example, tracking airplanes in an air traffc control system using radar data. 

Traditionally, linear dynamical systems have not been viewed from the perspective of factor- ized representations of the distribution. They are traditionally represented as a state-observation model, where the state and observation are both vector-valued random variables, and the transi- tion and observation models are encoded using matrices. More precisely, the model is generally defined via the following set of equations: 

$$
\begin{align}
{{P(X^{(t)}\mid X^{(t-1)})}}&={{\mathcal{N}\left(A X^{(t-1)};Q\right),}}\tag{6.3}\\ {{P(O^{(t)}\mid X^{(t)})}}&={{\mathcal{N}\left(H X^{(t)};R\right),}}\tag{6.4}\end{align}
$$

where: $X$ is an $n$ -vector of state variables, $O$ is an $m$ -vector of observation variables, $A$ is an $n\times n$ matrix defining the linear transition odel, $Q$ n $n\times n$ matrix defining the Gaussian noise associ ed with the system dynamics, H is an $n\times m$ × matrix defining the linear observation model, and R is an $m\times m$ matrix defining the Gaussian noise associated with the observations. 
> 线性动态系统通过 (6.3), (6.3) 定义，其中 $X$ 是长度为 $n$ 的状态向量，$O$ 是长度为 $n$ 的观察向量，矩阵 $A (n\times n)$ 定义了线性转移模型，矩阵 $Q (n\times n)$ 定义了和系统动态相关的高斯噪声，矩阵 $H (n\times n)$ 定义了线性观察模型，矩阵 $R (m\times m)$ 定义了和观察相关的高斯噪声

This type of model encodes independence structure implicitly, in the parameters of the matrices (see exercise 7.5). 
> 这类模型在矩阵的参数中隐式编码了独立性结构

There are many interesting generalizations of the basic linear dynamical system, which can also be placed within the DBN framework. For example, a nonlinear variant, often called an extended Kalman filter , is a system where the state and observation variables are still vectors of real numbers, but where the state transition and observation models can be nonlinear functions rather than linear matrix multiplications as in equation (6.3) and equation (6.4). Specifically, we usually write: 

$$
\begin{array}{r c l}{{P(X^{(t)}\mid X^{(t-1)})}}&{{=}}&{{f(X^{(t-1)},U^{(t-1)})}}\\ {{P(O^{(t)}\mid X^{(t)})}}&{{=}}&{{g(X^{(t)},W^{(t)}),}}\end{array}
$$

where $f$ and $g$ are deterministic nonlinear functions, and $U^{(t)},W^{(t)}$ are Gaussian random variables that explicitly encode the noise in the transition and observation models, respectively. 
> 线性动态系统可以进行拓展，例如非线性的变体，即拓展的 Kalman 滤波器
> 其中的状态和观察变量仍是实数向量，但状态转移模型和观察模型不再是简单矩阵乘法，而是非线性函数

In other words, rather than model the system in terms of stochastic CPDs, we use an equivalent representation that partitions the model into a deterministic function and a noise component. 
> 需要注意的是，非线性情况下，非线性函数是确定性函数，系统的不确定性来源于作为输入的高斯噪声的不确定性，这和直接之前用高斯分布建模条件概率分布是不同的
> 因此，我们没有用随机 CPDs 建模系统，而是将模型划分为一个确定性函数和一个噪声成分，这两个表示实际上是等价的

Another interesting class of models are systems where the continuous dynamics are linear, but that also include discrete variables. For example, in our tracking example, we might introduce a discrete variable that denotes the driver’s target lane in the freeway: the driver can stay in her current lane, or she can switch to a lane on the right or a lane on the left. Each of these discrete settings leads to diferent dynamics for the vehicle velocity, in both the lateral (across the road) and frontal velocity. 
> 系统还可以引入离散变量，系统的连续动态是线性的，但是不同的离散变量可以对应不同的连续动态

Systems that model such phenomena are called switching linear dynamical system (SLDS) . In such models, we system can switch between a set of discrete modes . While within a fixed mode, the system evolves using standard linear (or nonlinear) Gaussian dynamics, but the equations governing the dynamics are diferent in diferent modes. 
> 这类系统称为切换线性动态系统，系统在一系列离散模式中切换，在某个模式下，系统使用标准线性/非线性高斯动态来演化，不同的模式中，具体的动态（参数/形式）是不同的

We can view this type of system as a DBN including a discrete variable $D$ that encodes the mode, where $\mathrm{Pa}_{D^{\prime}}\,=\,\{D\}$ , and allowing D to be the parent of (some of) the continuous variables in the model, so that they use a conditional linear Gaussian CPDs. 
> 很显然，这就是条件线性高斯 CPDs

## 6.3 Template Variables and Template Factors 
Having seen one concrete example of a template-based model, we now describe a more general framework that provides the fundamental building blocks for a template-based model. This framework provides a more formal perspective on the temporal models of the previous section, and a sound foundation for the richer models in the remaining sections of this chapter. 

**Template Attributes** The key concept in the definition of the models we describe in this chapter is that of a *template* that is instantiated many times within the model. A template for a random variable allows us to encode models that contain multiple random variables with the same value space and the same semantics. For example, we can have a Blood-Type template, which has a particular value space (say, A , B , AB , or $O_{-}$ ) and is reused for multiple individuals within a pedigree. That is, when reasoning about a pedigree, as in box 3. B, we would want to have multiple instances of blood-type variables, such as Blood-Type ( Bart ) or Blood-Type ( Homer ) . We use the word attribute or template variable to distinguish a template, such as Blood-Type , from a specific random variable, such as Blood-Type ( Bart ) . 
> 一个随机变量的模板允许我们编码包含多个具有相同值空间和相同语义的随机变量的模型
> 随机变量模板实例化即得到随机变量
> 模板变量也可以称为属性

In a very diferent type of example, we can have a template attribute Location , which can be instantiated to produce random variables Location $(t)$ for a set of diferent time points $t$ . This type of model allows us to represent a joint distribution over a vehicle’s position at diferent points in time, as in the previous section. 

In these example, the template was a property of a single object — a person. More broadly, attributes may be properties of entire tuples of objects. For example, a student’s grade in a course is associated with a student-course pair; a person’s opinion of a book is associated with the person-book pair; the afnity between a regulatory protein in the cell and one of its gene targets is also associated with the pair. More specifically, in our Student example, we want to have a Grade template, which we can instantiate for diferent (student, course) pairs $s,c$ to produce multiple random variables $G r a d e(s,c)$ , such as $G r a d e(G e o r g e,C S l O I)$ .

Because many domains involve heterogeneous objects, such as courses and students, it is convenient to view the world as being composed of a set of objects . Most simply, objects can be divided into a set of mutually exclusive and exhaustive classes $\mathcal{Q}=\mathsf{Q}_{1},.\ldots,\mathsf{Q}_{k}$ . In the Student scenario, we might have a Student class and a Course class. 

Attributes have a tuple of arguments , each of which is associated with a particular class of objects. This class defines the set of objects that can be used to instantiate the argument in a given domain. For example, in our Grade template, we have one argument $S$ that can be instantiated with a “student” object, and another argument $C$ that can be instantiated with a “course” object. Template attributes thus provide us with a “generator” for random variables in a given probability space. 
> 属性具有一系列参数，每个参数都和特定类型的对象相关，类就是对象集合，它定义了参数在给定领域可以被实例化成什么对象
> 模板属性可以视为为我们在给定的概率空间提供了一个随机变量的“生成器”

**Definition 6.5** 
An attribute $A$ is a function ${\overline{{A(U_{1}}}},.\,.\,.\,,U_{k})$ , whose range is some set $V a l(A)$ and where each argu- ment $U_{i}$ is a typed logical variable associated with a particular class $\mathsf{Q}[U_{i}]$ . The tuple $U_{1},\dots,U_{k}$ is called the argument signature of the attribute $A$ , and denoted $\alpha(A)$ . 
> 定义：
> 属性 $A$ 是一个函数 $A (U_1,\dots, U_k)$，其值域是集合 $Val (A)$，其每个参数 $U_i$ 都是一个有类型的逻辑变量，和特定的类 $Q[U_i]$ 相关
> $U_1,\dots , U_k$ 称为属性 $A$ 的参数签名，记作 $\alpha (A)$

From here on, we assume without loss of generality that each logical variable $U_{i}$ is uniquely associated with a particular class $\mathsf{Q}[U_{i}]$ ; thus, any mention of a logical variable $U_{i}$ uniquely specifies the class over which it ranges. 
> 我们假设每个逻辑变量和唯一地和一个特定类 $Q[U_i]$ 关联

For example, the argument signature of the Grade attribute would have two logical variables $S,C$ , where $S$ is of class Student and $C$ is of class Course . We note that the classes associated with an attribute’s argument signature are not necessarily distinct. For example, we might have a binary-valued Cited attribute with argument signature $A_{1},A_{2}$ , where both are of type Article . We assume, for simplicity of presentation, that attribute names are uniquely defined; thus, for example, the attribute denoting the age for a person will be named diferently from the attribute denoting the age for a car. 

This last example demonstrates a basic concept in this framework: that of a relation . A relation is a property of a tuple of objects, which tells us whether the objects in this tuple satisfy a certain relationship with each other. For example, Took-Course is a relation over student-course object pairs $s,c,$ which is true is student $s$ took the course $c$ . As another example Mother is a relation between person-person pairs $p_{1},p_{2}$ , which is true if $p_{1}$ is the mother of $p_{2}$ . Relations are not restricted to involving only pairs of objects. For example, we can have a Go relation, which takes triples of objects — a person, a source location, and a destination location. 

At some level, a relation is simply a binary-valued attribute, as in our Cited example. However, this perspective obscures the fundamental property of a relation — that it relates a tuple of objects to each other. Thus, when we introduce the notion of probabilistic dependencies that can occur between related objects, the presence or absence of a relation between a pair of objects will play a central role in defining the probabilistic dependency model. 

**Instantiations** Given a set of template attributes, we can instantiate them in diferent ways, to produce probability spaces with multiple random variables of the same type. For example, we can consider a particular university, with a set of students and a set of courses, and use the notion of a template attribute to define a probability space that contains a random variable $G r a d e(s,c)$ for diferent (student, course) pairs $s,c$ . The resulting model encodes a joint distribution over the grades of multiple students in multiple courses. Similarly, in a temporal model, we can have the template attribute Location $(T)$ ; we can then select a set of relevant time points and generate a trajectory with specific random variables Location $.(t)$ 

To instantiate a set of template attributes to a particular setting, we need to define a set of objects for each class in our domain. For example, we may want to take a particular set of students and set of courses and define a model that contains a ground random variable Intelligence $(s)$ and $S A T(s)$ for every student object $s$ , a ground random variable Difculty $\cdot(c)$ for every course object $c,$ , and ground random variables $G r a d e(s,c)$ and Satisfaction $(s,c)$ for every valid pair of (student, course) objects. 

More formally, we now show how a set of template attributes can be used to generate an infinite set of probability spaces, each involving instantiations of the template attributes induced by some set of objects. We begin with a simple definition, deferring discussion of some of the more complicated extensions to section 6.6. 

**Definition 6.6** object skeleton
Let $\mathcal{Q}$ be a set of classes, and $\aleph$ a set of tem ttributes over $\mathcal{Q}$ . An object skeleton $\kappa$ specifies a fixed, finite set of objects ${\mathcal{O}}^{\kappa}[\mathsf{Q}]$ for every $\mathsf Q\in\mathcal Q$ . We also define ${\mathcal O}^{\kappa}[U_{1},.\,.\,,U_{k}]={\mathcal O}^{\kappa}[\mathsf{Q}[U_{1}]]\times.\,.\,.\times{\mathcal O}^{\kappa}[\mathsf{Q}[U_{k}]].$ By default, we define $\Gamma_{\kappa}[A]\,=\,{\mathcal{O}}^{\kappa}[\alpha(A)]$ to be the set of possible assignments to the logical variables in the argument signature of A . However, an object skeleton may also specify a subset of legal assignments. $\Gamma_{\kappa}[A]\subset\mathcal{O}^{\kappa}[\alpha(A)]$ . 
> 定义：
> $\mathcal Q$ 为类的集合，$\aleph$ 是 $\mathcal Q$ 上的属性的集合，一个对象框架 $\kappa$ 为每个类 $Q\in \mathcal Q$ 指定了一个固定的、有限的对象集合 $\mathcal O^k[Q]$
> 我们还定义 ${\mathcal O}^{\kappa}[U_{1},.\,.\,,U_{k}]={\mathcal O}^{\kappa}[\mathsf{Q}[U_{1}]]\times.\,.\,.\times{\mathcal O}^{\kappa}[\mathsf{Q}[U_{k}]].$ ，也就是 $\mathcal O^{\kappa}[U_1,\dots , U_k]$ 的成员是 $\mathcal O^{\kappa}[Q[U_i]]$ 的各自成员的所有可能组合
> 我们还定义 $\Gamma_{\kappa}[A] = \mathcal O^{\kappa}[\alpha (A)]$，即定义了 $A$ 的参数签名的逻辑变量集合的所有可能的赋值构成的集合，可以人为定义一些赋值不合法，此时 $\Gamma_{\kappa}[A] \subset \mathcal O^{\kappa}[\alpha (A)]$

We can now define the set of instantiations of the attributes: 

**Definition 6.7** ground random variable 
Let $\kappa$ be an object skeleton over $\mathcal{Q},\aleph$ . We define sets of ground random variables: 

$$
\begin{array}{r c l}{\mathcal{X}_{\kappa}[A]}&{=}&{\{A(\gamma)\ :\ \gamma\in\Gamma_{\kappa}[A]\}}\\ {\mathcal{X}_{\kappa}[\aleph]}&{=}&{\cup_{A\in\aleph}\mathcal{X}_{\kappa}[A].}\end{array}\tag{6.5}
$$ 
Note that we ar tation here, identifying an assignment $\gamma=\left\langle U_{1}\mapsto u_{1},.\,.\,.\,,U_{k}\mapsto u_{k}\right\rangle$ with the tuple ⟨ $\langle u_{1}\cdot\cdot\cdot,u_{k}\rangle$ ⟩ ; this abuse of notation is unambiguous in this context due to the ordering of the tuples. 
> 定义：
> 令 $\kappa$ 为 $\mathcal Q, \aleph$ 上的对象框架，定义 ground random variable 的集合
> $\mathcal X_{\kappa}[A] = \{A(\gamma): \gamma \in \Gamma_{\kappa}[A]\}$
> ${\mathcal{X}_{\kappa}[\aleph]}={\cup_{A\in\aleph}\mathcal{X}_{\kappa}[A].}$

The ability to specify a subset of ${\mathcal{O}}^{\kappa}[\alpha(A)]$ is useful in eliminating the need to consider random variables that do not really appear in the model. For example, in most cases, not every student takes every course, and so we would not want to include a Grade variable for every possible (student, course) pair at our university. See figure 6.5 as an example. 

Clearly, the set of random variables is diferent for diferent skeletons; hence the model is a template for an infinite set of probability distributions, each spanning a diferent set of objects that induces a diferent set of random variables. In a sense, this is similar to the situation we had in DBNs, where the same 2TBN could induce a distribution over diferent numbers of time slices. Here, however, the variation between the diferent instantiations of the template is significantly greater. 

Our discussion so far makes several important simplifying assumptions. First, we portrayed the skeleton as defining a set of objects for each of the classes. As we discuss in later sections, it can be important to allow the skeleton to provide additional background information about the set of possible worlds, such as some relationships that hold between objects (such as the structure of a family tree). Conversely, we may also want the skeleton to provide less information: In particular, the premise underlying equation (6.5) is that the set of objects is predefined by the skeleton. As we brieﬂy discuss in section 6.6.2, we may also want to deal with settings in which we have uncertainty over the number of objects in the domain. In this case, diferent possible worlds may have diferent sets of objects, so that a random variable such as $A(u)$ may be defined in some worlds (those that contain the object $u$ ) but not in others. Settings like this pose significant challenges, a discussion of which is outside the scope of this book. 
> (6.5) 的前提是对象的集合由对象框架 $\kappa$ 预定义

**Template Factors** The final component in a template-based probabilistic model is one that defines the actual probability distribution over a set of a ground random variables generated from a set of template attributes. Clearly, we want the specification of the model to be defined in a template-based way. Specifically, we would like to take a factor — whether an undirected factor or a CPD — and instantiate it to apply to multiple scopes in the domain. We have already seen one simple example of this notion: in a 2-TBN, we had a template CPD $P(X_{i}^{\prime}\mid\mathrm{Pa}_{X_{i}^{\prime}})$ , which we instantiated to apply to diferent scopes $X_{i}^{(t)},\mathrm{Pa}_{X_{i}^{(t)}}$ , by instantiating any occurrence $X_{j}^{\prime}$ to $X_{j}^{(t)}$ , and any occurrence $X_{j}$ to $X_{j}^{(t-1)}$ . In efect, there we had template variables of the form $X_{j}$ and $X_{j}^{\prime}$ as arguments to the CPD, and we instantiated them in diferent ways for diferent time points. 
> 一个基于模板的概率模型定义了在由模板属性集合生成的 ground random variable 集合之上的实际概率分布

We can now generalize this notion by defining a factor with arguments. Recall that a factor $\phi$ is a function from a tuple of random variables $X=S c o p e[\phi]$ to the reals; this function returns a number for each assignment ${x}$ to the variables $X$ . We can now define 
> 一个因子是 $\phi$ 是将随机变量 tuple $X = Scope[\phi]$ 的映射到实数的函数，该函数为每个对 $X$ 的赋值 $\pmb x$ 返回一个实数

**Definition 6.8** instantiated factor 
$A$ template factor is a function $\xi$ defined over a tuple of template attributes $A_{1},\ldots,A_{l},$ where each $A_{j}$ has a range $V a l(A)$ . It defines a mapping from $V a l(A_{1})\times.-.\times V a l(A_{l})$ to $I\!\!R$ . Given a tuple of random variables $X_{1},\ldots,X_{l}.$ , such that $V a l(X_{j})=V a l(A_{j})$ for all $j\,=\,1,\ldots,l.$ , we define $\xi(X_{1},.\,.\,.\,,X_{l})$ to be the instantiated factor from $X$ to $I\!\!R$ . 
> 定义：
> 模板因子是一个函数 $\xi$，定义于模板属性 tuple $A_1,\dots, A_l$，其中每个 $A_j$ 的值域为 $Val (A_j)$，该函数定义了从 $Val (A_1)\times Val (A_2)\times \dots \times Val (A_l)$ 到 $R$ 的映射
> 给定一个随机变量 tuple $X_1,\dots, X_l$，使得 $Val (X_j) = Val (A_j)$，我们定义 $\xi (X_1, \dots, X_l)$ 为从 $\pmb X$ 到 $R$ 的实例化因子

In the subsequent sections of this chapter, we use these notions to define various languages for encoding template-based probabilistic models. As we will see, some of these representational frameworks subsume and generalize on the DBN framework defined earlier. 

## 6.4 Directed Probabilistic Models for Object-Relational Domains 

Based on the framework described in the previous section, we now describe template-based representation languages that can encode directed probabilistic models. 

### 6.4.1 Plate Models 

plate model We begin our discussion by presenting the plate model , the simplest and best-established of the object-relational frameworks. Although restricted in several important ways, the plate modeling framework is perhaps the approach that has been most commonly used in practice, notably for encoding the assumptions made in various learning tasks. This framework also provides an excellent starting point for describing the key ideas of template-based languages and for motivating some of the extensions that have been pursued in richer languages. 

In the plate formalism, object types are called plates . The fact that multiple objects in the class share the same set of attributes and same probabilistic model is the basis for the use of the term “plate,” which suggests a stack of identical objects. We begin with some motivating examples and then describe the formal framework. 

#### 6.4.1.1 Examples 

Example 6.8 

Figure 6.6 Plate model for a set of coin tosses sampled from a single coin 

plate that all have the same domain $V a l(X)$ and are sampled from the same distribution. In a plate representation, we encode the fact that these variables are all generated from the same template by drawing only a single node $X(d)$ and enclosing it in a box denoting that $d$ ranges over $\mathcal{D}$ , so that we know that the box represents an entire “stack” of these identically distributed variables. This box is called a plate , with the analogy that it represents a stack of identical plates. 

In many cases, we want to explicitly encode the fact that these variables have an identical distribution. We therefore often explicitly add to the model the variable $\theta_{X}$ , which denotes the parameter iz ation of the CPD from which the variables $X(d)$ are sampled. Most simply, if the $X\mathit{\dot{s}}$ are coin tosses of a single (possibly biased) coin, $\theta_{X}$ would take on values in the range $[0,1]$ , and its value would denote the bias of the coin (the probability with which it comes up “Heads”). 

The idea of including the CPD parameters directly in the probabilistic model plays a central role in our discussion of learning later in this book (see section 17.3). For the moment, we note only that including the parameters directly in the model allows us to make explicit the fact that all of the variables $X(d)$ are sampled from the same CPD. By contrast, we could also have used a model where a variable $\theta_{X}(d)$ is included inside the plate, allowing us to encode the setting where each of the coin tosses was sampled from a diferent coin. We note that this transformation is equivalent to adding the coin ID $d$ as a parent to $X$ ; however, the explicit placement of $\theta_{X}$ within the plate makes the nature of the dependence more explicit. In this chapter, to reduce clutter, we use the convention that parameters not explicitly included in the model (or in the figure) are outside of all plates. 

Example 6.9 

ground Bayesian network 

Let us return to our Student example. We can have a Student plate that includes the attributes $I(S),G(S)$ . As shown in figure 6.7a, we can have $G(S)$ depend on $I(s)$ . In this model we have a set of (Intelligence, Grade) pairs, one for each student. The figure also shows the ground Bayesian network that would result from instantiating this model for two students. As we discussed, this model implicitly makes the assumption that the CPDs for $P(I(s))$ and for $P(G(s)\mid I(s))$ is the same for all students $s$ . Clearly, we can further enrich the model by introducing additional variables, such as an SAT-score variable for each student. 

Our examples thus far have included only a single type of object, and do not significantly expand the expressive power of our language beyond that of plain graphical models. The key benefit of the plate framework is that it allows for multiple plates that can overlap with each other in various ways. 

Example 6.10 

nested plate Assume we want to capture the fact that a course has multiple students in it, each with his or her own grade, and that this grade depends on the difculty of the course. Thus, we can introduce a second type of plate, labeled Course , where the Grade attribute is now associated with a (student, course) pair. There are several ways in which we can modify the model to include courses. In figure $6.7b,$ the Student plate is nested within the Course plate. The Difculty variable is enclosed within the Course plate, whereas Intelligence and Grade are enclosed within both plates. We thus have that Grade $(s,c)$ for a particular (student, course) pair $(s,c)$ depends on Difculty $(c)$ and on Intelligence $(s,c)$ . 

This formulation ignores an important facet of this problem. As illustrated in figure $6.7\mathrm{b}$ , it induces networks where the Intelligence variable is associated not with a student, but rather with a (student, course) pair. Thus, if we have the same student in two diferent courses, we would have two diferent variables corresponding to his intelligence, and these could take on diferent values. This formulation may make sense in some settings, where diferent notions of “intelligence” may be appropriate to diferent topics (for example, math versus art); however, it is clearly not a suitable model for all settings. Fortunately, the plate framework allows us to come up with a diferent formulation. 

Example 6.11 plate intersection 

Figure 6.7c shows a construction that avoids this limitation. Here, the Student plate and the Course plates intersect , so that the Intelligence attribute is now associated only with the Student plate, and Difculty with the Course plate; the Grade attribute is associated with the pair (comprising the intersection between the plates). The interpretation of this dependence is that, for any pair of (student, course) objects $s,c,$ the attribute Grade $(s,c)$ depends on Intelligence ( s ) and on Difculty $(c)$ . The figure also shows the network that results for two students both taking two courses. 

In these examples, we see that even simple plate representations can induce fairly complex ground Bayesian networks. Such networks model a rich network of interdependencies between diferent variables, allowing for paths of inﬂuence that one may not anticipate. 

Example 6.12 Consider the plate model of figure 6.7c, where we know that a student Jane took CS101 and got an A. This fact changes our belief about Jane’s intelligence and increases our belief that CS101 is an easy class. If we now observe that Jane got a C in Math 101, it decreases our beliefs that she is intelligent, and therefore should increase our beliefs that CS101 is an easy class. If we now observe that George got a C in CS101, our probability that George has high intelligence is significantly lower. Thus, our beliefs about George’s intelligence can be afected by the grades of other students in other classes. 

Figure 6.8 shows a ground Bayesian network induced by a more complex skeleton involving fifteen students and four courses. Somewhat surprisingly, the additional pieces of “weak” evidence regarding other students in other courses can accumulate to change our conclusions fairly radically: Considering only the evidence that relates directly to George’s grades in the two classes that he took, our posterior probability that George has high intelligence is 0 . 8 . If we consider our entire body of evidence about all students in all classes, this probability decreases from 0.8 to 0.25. When we examine the evidence more closely, this conclusion is quite intuitive. We note, for example, that of the students who took CS101, only George got a C. In fact, even Alice, who got a C in both of her other classes, got an A in CS101. This evidence suggests strongly that CS101 is not a difcult class, so that George’s grade of a C in CS101 is a very strong indicator that he does not have high intelligence. 

![](images/766ffe8ec48820114c23511ee14c1346ff86fadfddab2499b0f444ed282cc8b8.jpg) 
Figure 6.7 Plate models and induced ground Bayesian networks for a simplified Student example. (a) Single plate: Multiple independent samples from the same distribution. (b) Nested plates: Multiple courses, each with a separate set of students. (c) Intersecting plates: Multiple courses with overlapping sets of students. 

Thus, we obtain much more informed conclusions by defining probabilistic models that encompass all of the relevant evidence. 

As we can see, a plate model provides a language for encoding models with repeated  structure and shared parameters. As in the case of DBNs, the models are represented at the template level ; given a particular set of objects, they can then be instantiated to induce a ground Bayesian network over the random variables induced by these objects. Because there are infinitely many sets of objects, this template can induce an infinite set of ground networks. 

![](images/3c0885c0e2f292f6a2965711eed8b485ad88a0872be97ed433c701e388fa507d.jpg) 
Figure 6.8 Illustration of probabilistic interactions in the University domain. The ground network contains random variables for the Intelligence of fifteen students (right ovals), including George (denoted by the white oval), and the Difculty for four courses (left ovals), including CS101 and Econ101. There are also observed random variables for some subset of the (student, course) pairs. For clarity, these observed grades are not denoted as variables in the network, but rather as edges relating the relevant (student, course) pairs. Thus, for example George received an A in Econ101 but a C in CS101. Also shown are the final probabilities obtained by running inference over the resulting network. 

#### 6.4.1.2 Plate Models: Formal Framework 

We now provide a more formal description of the plate modeling language: its representation and its semantics. The plate formalism uses the basic object-relational framework described in section 6.4. As we mentioned earlier, plates correspond to object types. 

Each template attribute in the model is embedded in zero, one, or more plates (when plates intersect). If an attribute $A$ is embedded in a set of plates $\mathsf{Q}_{1},\ldots,\mathsf{Q}_{k}$ , we can view it as being associated with the argument signature $U_{1},\dots,U_{k}$ , where each logical variable $U_{i}$ ranges over the objects in the plate (class) $\mathsf{Q}_{i}$ . Recall that a plate model can also have attributes that are external to any plate; these are attributes for which there is always a single copy in the model. We can view this attribute as being associated with an argument signature of arity zero. 

In a plate model, the set of random variables induced by a template attribute $A$ is defined by the complete set of assignments: $\Gamma_{\kappa}[A]\,=\,{\mathcal{O}}^{\kappa}[\alpha(A)]$ . Thus, for example, we would have a Grade random variable for every (student, course) pair, whereas, intuitively, these variables are only defined in cases where the student has taken the course. We can take the values of such variables to be unobserved, and, if the model is well designed, its descendants in the probabilistic dependency graph will also be unobserved. In this case, the resulting random variables in the network will be barren, and can be dropped from the network without afecting the marginal distribution over the others. This solution, however, while providing the right semantics, is not particularly elegant. 

We now define the probabilistic dependency structures allowed in the plate framework. To provide a simple, well-specified dependency structure, plate models place strong restrictions on the types of dependencies allowed. For example, in example 6.11, if we define Intelligence to be a parent of Difculty (reﬂecting, say, our intuition that intelligent students may choose to take harder classes), the semantics of the ground model is not clear: for a ground random variable $D(c)$ , the model does not specify which specific $I(s)$ is the parent. To avoid this problem, plate models require that an attribute can only depend on attributes in the same plate. This requirement is precisely the intuition behind the notion of plate intersection: Attributes in the intersection of plates can depend on other attributes in any of the plates to which they belong. Formally, we have the following definition: 

Definition 6.9 plate model 

template parent 

parent argument signature 

$A$ plate model $\mathcal{M}_{P l a t e}$ defines, for each template attribute $A~\in~\aleph$ with argument signature $U_{1},\dots,U_{k}$ : • a set of template parents $\mathrm{Pa}_{A}=\{B_{1}(\pmb{U}_{1}),.\,.\,.\,,B_{l}(\pmb{U}_{l})\}$ such that for each $B_{i}(\pmb{U}_{i})$ , we e that $U_{i}\;\subseteq\;\{U_{1},.\,.\,.\,,U_{k}\}$ . The variables $U_{i}$ are the argument signature of the parent $B_{i}$ . • a template CPD $P(A\mid\mathrm{Pa}_{A})$ . 

This definition allows the Grade attribute $G r a d e (S, C)$ to depend on Intelligence $\cdot (S)$ , but not vice versa. Note that, as in Bayesian networks, this definition allows any form of CPD, with or without local structure. 

Note that the straightforward graphical representation of plates fails to make certain distinc- tions that are clear in the symbolic representation. 

Example 6.13 

Assume that our model contains an attribute $\mathit{C i t e d}(U_{1}, U_{2})$ , where $U_{1}, U_{2}$ are both in the Paper class. We might want the dependency model of this attribute to depend on properties of both papers, for example, $T o p i c (U_{1})$ , $T o p i c (U_{2})$ , or Review-Paper $\left (U_{1}\right)$ . To encode this dependency graphically, we first need to have two sets of attributes from the Paper class, one for $U_{1}$ and the other for $U_{2}$ . Moreover, we need to denote somehow which attributes of which of the two arguments are the parents. The symbolic representation makes these distinctions unambiguously. 

To instantiate the template parents and template CPDs, it helps to introduce some shorthand notation. $\gamma=\left\langle U_{1}\mapsto u_{1},.\,.\,.\,, U_{k}\mapsto u_{k}\right\rangle$ be some assignment to some set of logical vari- ables, and $B (U_{i_{1}},.\,.\,.\,, U_{i_{l}})$ be an attribute whose argument signature involves only a subset of these variables. We define $B (\gamma)$ to be the ground random variable $B (u_{i_{1}},.\,.\,.\,, u_{i_{l}})$ . 

The template-level plate model, when applied to a particular skeleton, defines a ground probabilistic model, in the form of a Bayesian network: 

Definition 6.10 ground Bayesian network 

lows. Let $A (U_{1},\dots, U_{k})$ be any template attribute in $\aleph$ . Then, for any assignment $\gamma\_=$ $\left\langle U_{1}\mapsto u_{1},.\,.\,, U_{k}\mapsto u_{k}\right\rangle\in\Gamma_{\kappa}[A].$ , we have a va $A (\gamma)$ in the ground network, with parents $B (\gamma)$ for all $B\in\mathrm{Pa}_{A}$ ∈ , and the instantiated CPD $P (A (\gamma)\mid\mathrm{Pa}_{A}(\gamma))$ | . 

Thus, in our example, we have that the network contains a set of ground random variables $G r a d e (s, c)$ , one for every student $s$ and every course $c$ . Each such variable depends on Intelligence $\cdot (s)$ and on Difculty $\left (c\right)$ . 

The ground net $\mathcal{B}_{\kappa}^{\mathcal{M}_{P l a t e}}$ specifies a well-defined joint distribution over $\mathcal{X}_{\kappa}[\aleph]$ , as required. The BN in figure 6.7b is precisely the network structure we would obtain from this defini- tion, using the plate model of figure 6.7 and the object skeleton $\mathcal{O}^{\kappa}[\mathsf{S t u d e n t}]=\{s_{1}, s_{2}\}$ and $\mathcal{O}^{\kappa}[\mathsf{C o u r s e}]=\{c_{1}, c_{2}\}$ . In general, despite the compact parameter iz ation (only one local prob- abilistic model for every attribute in the model), the resulting ground Bayesian network can be quite complex, and models a rich set of interactions. As we saw in example 6.12, the ability to incorporate all of the relevant evidence into the single network shown in the figure can significantly improve our ability to obtain meaningful conclusions even from weak indicators. 

The plate model is simple and easy to understand, but it is also highly limited in several ways. Most important is the first condition of definition 6.9, whereby $A (U_{1},\dots, U_{k})$ can only depend on attributes of the form $B (U_{i_{1}},.\,.\,.\,, U_{i_{l}})$ , where $U_{i_{1}},\ldots, U_{i_{l}}$ is a subtuple of $U_{1},\dots, U_{k}$ . This restriction significantly constrains our ability to encode a rich network of probabilistic dependencies between the objects in the domain. For example, in the Genetics domain, we cannot encode a dependence of Genotype $\left (U_{1}\right)$ on Genotype ( $\left (U_{2}\right)$ , where $U_{2}$ is (say) the mother of $U_{1}$ . Similarly, we cannot encode temporal models such as those described in section 6.2, where the car’s position at a point in time depends on its position at the previous time point. In the next section, we describe a more expressive representation that addresses these limitations. 

### 6.4.2 Probabilistic Relational Models 

As we discussed, the greatest limitation of the plate formalism is its restriction on the argument signature of an attribute’s parents. In particular, in our genetic inheritance example, we would like to have a model where Genotype $\prime (u)$ depends on Genotype $(u^{\prime})$ , where $u^{\prime}$ is the mother of $u$ . This type of dependency is not encodable within plate models, because it uses a logical variable in the attribute’s parent that is not used within the attribute itself. To allow such models, we must relax this restriction on plate models. However, relaxing this assumption without care can lead to nonsensical models. In particular, if we simply allow Genotype $(U)$ to depend on Genotype $\left (U^{\prime}\right)$ , we end up with a dependency model where every ground variable Genotype $(u)$ depends on every other such variable. Such models are intractably dense, and (more importantly) cyclic. What we really want is to allow a dependence of Genotype $(U)$ on Genotype $\left (U^{\prime}\right)$ , but only for those assignments to $U^{\prime}$ that correspond to $U$ ’s mother. We now describe one representation that allows such dependencies, and then discuss some of the subtleties that arise when we introduce this significant extension to our expressive power. 

#### 6.4.2.1 Contingent Dependencies 

To capture such situations, we introduce the notion of a contingent dependency , which specifies the context in which a particular dependency holds. A contingent dependency is defined in terms of a guard — a formula that must hold for the dependency to be applicable. 

Example 6.14 

Consider again our University example. As usual, we can define Grade $(S, C)$ for a student $S$ and a course $C$ to have the parents Difculty $\prime (C)$ and Intelligence $\cdot (S)$ . Now, however, we can make this dependency contingent on the guard Registered $\iota (S, C)$ . Here, the parent’s argument signature is the same as the child’s. More interestingly, contingent dependencies allow us to model the dependency of the student’s satisfaction in a course, Satisfaction $(S, C)$ , on the teaching ability of the professor who teaches the course. In this setting, we can make Teaching-Ability $\cdot (P)$ the parent faction $. (S, C)$ , where the dependency is contingent on the guard Registered $\I (S, C)\wedge$ Teaches $\cdot (P, C)$ . Note that here, we have more logical variables in the parents of Satisfaction $(S, C)$ than in the attribute itself: the attribute’s argument signature is $S, C,$ , whereas its parent argument signature is the tuple $S, C, P$ . 

We can also represent chains of dependencies within objects in the same class. 

Example 6.15 

For example, to encode temporal models, we could have Location $(U)$ depend on Location $(V)$ , contingent on the guard Precedes $\cdot (V, U)$ . In our Genetics example, for the attribute Genotype $(U)$ , we would define the template parents G $\cdot (V)$ and Genotype $(W)$ , the guard Mother $\cdot (V, U)\wedge$ Father $\mathopen{}\mathclose\bgroup\left (W, U\aftergroup\egroup\right)$ , and the parent signature $U, V, W$ . 

We now provide the formal definition underlying these examples: 

Definition 6.11 

contingent dependency model parent argument signature guard 

probabilistic relational model For a template attribute $A$ , we define $^a$ contingent dependency model as a tuple consisting of: 

• $A$ parent argument signature $\alpha (\operatorname{Pa}_{A})$ , which is a tuple of typed logical variables $U_{i}$ such that $\alpha (\mathrm{Pa}_{A})\supseteq\alpha (A)$ . • $A$ guard $\Gamma$ , which is a binary-valued formula defined in terms of a set of template attributes $\mathrm{Pa}_{A}^{\bar{\Gamma}}$ over the argument signature $\alpha (\operatorname{Pa}_{A})$ . • a set of template parents $\mathrm{Pa}_{A}=\{B_{1}(\pmb{U}_{1}),.\,.\,.\,, B_{l}(\pmb{U}_{l})\}$ such that for each $B_{i}(\pmb{U}_{i})$ , we have that $U_{i}\subseteq\alpha (\mathrm{Pa}_{A})$ . 

A probabilistic relational model (PRM) $\mathcal{M}_{P R M}$ defines, for each $A\in\mathbb{N}$ a contingent dependency model, as in definition 6.11, and a template CPD. The structure of the template CPD in this case is more complex, and we discuss it in detail in section 6.4.2.2. 

Intuitively, the template parents in a PRM, as in the plate model, define a template for the parent assignments in the ground network, which will correspond to specific assignments of the logical variables to objects of the appropriate type. In this setting, however, the set of logical variables in the parents is not necessarily a subset of the logical variables in the child. 

The ability to introduce new logical variables into the specification of an attribute’s par- ents gives us significant expressive power, but introduces some significant challenges. These challenges clearly manifest in the construction of the ground network. 

A PRM $\mathcal{M}_{P R M}$ and object skeleton $\kappa$ define a ground Bayesian network $\mathcal{B}_{\kappa}^{\mathcal{M}_{P R M}}$ as follows. Let $A (U_{1},\dots, U_{k})$ be any template attribute in ℵ . Then, for any assig $\gamma\,\in\,\Gamma_{\kappa}[A]$ we have $a$ variable $A (\gamma)$ in the ground network. This variable has, for any $B\,\in\,\mathrm{Pa}_{A}^{\Gamma}\cup\mathrm{Pa}_{A}$ ∈ and any assignment $\gamma^{\prime}$ to $\alpha (\mathrm{Pa}_{A})-\alpha (A).$ , the parent that is the instantiated variable $B (\gamma,\gamma^{\prime})$ . 

An important subtlety in this definition is that the attributes that appear in the guard are also parents of the ground variable. This requirement is necessary, because the values of the guard attributes determine whether there is a dependency on the parents or not, and hence they afect the probabilistic model. 

Using this definition for the model of example 6.14, we have that Satisfaction $\iota (s, c)$ has the parents: Teaching-Ability $\mathbf{\sigma}(p)$ , Registered $(s, c)$ , and Teaches $\cdot (p, c)$ for every professor $p$ . The guard in the contingent dependency is intended to encode the fact that the dependency on Teaching-Ability $(p)$ is only present for a subset of individuals $p$ , but it is not obvious how that fact afects the construction of our model. The situation is even more complex in example 6.15, where we have as parents of Genotype $(u)$ all of the variables of the form Father $\cdot (v, u)$ and Genotype ( v ) , for all person objects $v$ , and similarly for $M o t h e r (v, u)$ and Genotype $(v)$ . In both cases, the resulting ground network is very densely connected. In the Genetics network, it is also obviously cyclic. We will describe how to encode such dependencies correctly within the CPD of the ground network, and how to deal with the issues of potential cyclicity. 

#### 6.4.2.2 CPDs in the Ground Network 

As we just discussed, the ground network induced by a PRM can introduce a dependency of a variable on a set of parents that is not fixed in advance, and which may be arbitrarily large. How do we encode a probabilistic dependency model for such dependencies? 

Exploiting the Guard Structure The first key observation is that the notion of a contin- gent dependency is intended to specifically capture context-specific independence: In defini- tion 6.12, if the guard for a parent $B$ of $A$ is false for a particular assignment $(\gamma,\gamma^{\prime})$ , then there is no dependency of $A (\gamma)$ on $B (\gamma,\gamma^{\prime})$ . For example, unless Mother $\cdot (v, u)$ is true for a particular pair $(u, v)$ , we have no dependence of Genotype ( u ) on Genotype $(v)$ . Similarly, unless Registered $(s, c)\ \wedge$ Teaches $\!:\! (p, c)$ is true, there is no dependence of Satisfaction $(s, c)$ on Teaching-Ability $(p)$ . We can easily capture this type of context-specific independence in the CPD using a variant of the multiplexer CPD of definition 5.3. 

While this approach helps us specify the CPD in these networks of potentially unbounded indegree, it does not address the fundamental problem: the dense, and often cyclic, connectivity structure. A common solution to this problem is to assume that the guard predicates are properties of the basic relational structure in the domain, and are often fixed in advance. For example, in the tem oral setting, the Precedes relation is always fixed: time point $t-1$ always precedes time point t . Somewhat less obviously, in our Genetics example, it may be reasonable to assume that the pedigree is known in advance. 

relational skeleton 

We can encode this assumption by defining a relational skeleton $\kappa_{r}$ , which defines a certain set of facts (usually relationships between objects) that are given in advance, and are not part of the probabilistic model. In cases where the values of the attributes in the guards are specified as part of the relational skeleton, we can simply use that information to determine the set of parents that are active in the model, usually a very limited set. Thus, for example, if Registered and Teaches are part of the relational skeleton, then Satisfaction $(s, c)$ has the parent Teaching-Ability $\mathbf{\sigma}(p)$ only when Registered $\left (s, c\right)$ and $T e a c h e s (p, c)$ both hold. Similarly, in the Genetics example, if the pedigree structure is given in the skeleton, we would have that Genotype $(v)$ is a parent of Genotype ( u ) only if $M o t h e r (v, u)$ or Father $\cdot (v, u)$ are present in the skeleton. Moreover, we see that, assuming a legal pedigree, the resulting ground network in the Genetics domain is guaranteed to be acyclic. Indeed, the resulting model produces ground networks that are precisely of the same type demonstrated in box 3.B. The use of contingent dependencies allows us to exploit relations that are determined by our skeleton to produce greatly simplified models, and to make explicit the fact that the model is acyclic. 

relational uncertainty 

The situation becomes more complex, however, if the guard predicates are associated with a probabilistic model, and therefore are random variables in the domain. Because the guards are typically associated with relational structure, we refer to this type of uncertainty as relational uncertainty . Relational uncertainty introduces significant complexity into our model, as we now cannot use background knowledge (from our skeleton) to simplify the dependency structure in contingent dependencies. In this case, when the family tree is uncertain, we may indeed have that Genotype ( u ) can depend on every other variable Genotype ( v ) , a model that is cyclic and ill defined. However, if we restrict the distribution over the Mother and Father relations so as to ensure that only “reasonable” pedigrees get positive probability, we can still guarantee that our probabilistic model defines a coherent probability distribution. However, defining a probability distribution over the Mother and Father relations that is guaranteed to have this property is far from trivial; we return to this issue in section 6.6. 

Aggregating Dependencies By itself, the use of guards may not fully address the problem of defining a parameter iz ation for the CPDs in a PRM. Consider again a dependency of $A$ on an attribute $B$ that involves some set of logical variables $U^{\prime}$ that are not in $\alpha (A)$ . Even if we assume that we have a relational skeleton that fully determines the values of all the guards, there may be multiple assignments $\gamma^{\prime}$ to $U^{\prime}$ for which the guard holds, and hence multiple diferent ground parents $B (\gamma,\gamma^{\prime})$ — one for each distinct assignment $\gamma^{\prime}$ to $U^{\prime}$ . Even in our simple University example, there may be multiple instructors for a course $c_{i}$ , and therefore multiple ground variables Teaching-Ability $\mathbf{\sigma}(p)$ that are parents of a ground variable Satisfaction $(s, c)$ . In general, the number of possible instantiations of a given parent $B$ is not known in advance, and may not even be bounded. Thus, we need to define a mechanism for specifying a template-level local dependency model that allows a variable number of parents. Moreover, because the parents corresponding to diferent instantiations are interchangeable, the local dependency model must be symmetric. 

aggregator CPD 

There are many possible ways of specifying such a model. One approach is to use one of the symmetric local probability models that we saw in chapter 5. For example, we can use a noisy-or (section 5.4.1) or logistic model (see section 5.4.2), where all parents have the same parameter. An alternative approach is to define an aggregator CPD that uses certain aggregate statistics or summaries of the set of parents of a variable. (See exercise 6.7 for some analysis of the expressive power of such CPDs.) 

Ability are binary-valued, we might use a noisy-or model: Given a parameter iz ation for Satisfac- tion given a single Teaching-Ability, we can use the noisy-or model to define a general CPD for Satisfaction $(s, c)$ given any set of parents Teaching-Ability ( $p_{1}$ ) , . . . , Teaching-Ability $(p_{m})$ . Alter- natively, we can assume that the student’s satisfaction depends on the worst instructor and best instructor in the course. In this case, we might aggregate the teaching abilities using the min and max functions, and then use a CPD of our choosing to denote the student’s satisfaction as a function of the resulting pair of values. As another example, a student’s job prospects can depend on the average grade in all the courses she has taken. 

When designing such a combination rule, it is important to consider any possible boundary cases. On one side, in many settings the set of parents can be empty: a course may have no instructors (if it is a seminar composed entirely of invited lecturers); or a person may not have a parent in the pedigree. On the other side, we may also need to consider cases where the number of parents is large, in which case noisy-or and logistic models often become degenerate (see figure 5.10 and figure 5.11). 

The situation becomes even more complex when there are multiple distinct parents at the template level, each of which may result in a set of ground parents. For example, a student’s satisfaction in a course may depend both on the teaching ability of multiple instructors and on the quality of the design of the diferent problem sets in the course. We therefore need to address both the aggregation of each type of parent set (instructors or problem sets) as well as combining them into a single CPD. Thus, we need to define some way of combining a set of CPDs $\{P (X\mid Y_{i, 1},.\,.\,.\,, Y_{i, j_{i}})\quad:\quad i\,=\, 1,.\,.\,.\,, l\}$ , to a single joint CPD $\{P (X~|~}$ $Y_{1,1},.\,.\,.\,, Y_{1, j_{1}},.\,.\,.\,, Y_{l, 1},.\,.\,.\,, Y_{l, j_{l}})$ . Here, as before, there is no single right answer, and the particular choice is likely to depend heavily on the properties of the application. 

We note that the issue of multiple parents is distinct from the multiple parents that arise when we have relational uncertainty. In the case of relational uncertainty, we also have multiple parents for a variable in the ground network; yet, it may well be the case that, in any situation, at most one assignment to the logical variables in the parent signature will satisfy the guard condition. For example, even if we are uncertain about John’s paternity, we would like it to be the case that, in any world, there is a unique object $v$ for which Father $\cdot (v, J o h n)$ holds. 

(As we discuss in section 6.6, however, defining a probabilistic model that ensures this type of constraint can be far from trivial.) In this type of situation, the concept of a guard is again useful, since it allows us to avoid defining local dependency models with a variable number of parents in domains (such as genetic inheritance) where such situations do not actually arise. 

#### 6.4.2.3 Checking Acyclicity 

One important issue in relational dependency models is that the dependency structure of the ground network is, in general, not determined in advance, but rather induced from the model structure and the skeleton. How, then, can we guarantee that we obtain a coherent probability distribution? Most obviously, we can simply check, post hoc , that any particular ground network resulting from this process is acyclic. However, this approach is unsatisfying from a model design perspective. When constructing a model, whether by hand or using learning methods, we would like to have some guarantees that it will lead to coherent probability distributions. 

Thus, we would like to provide a test that we can execute on a mod $\mathcal{M}_{P R M}$ at the template level , and which will guarantee that ground distributions induced from M $\mathcal{M}_{P R M}$ will be coherent. 

![](images/aea016ac9cf2d4c81c74eab2678c07ca2c96ead2838bb25f607d051aa3d027da.jpg) 
Figure 6.9 Examples of dependency graphs: (a) Dependency graph for the University example. (b) Dependency graph for the Genetics example. 

One approach for doing so is to construct a template-level graph that encodes a set of potential dependencies that may happen at the ground level. The nodes in the graph are the template- level attributes; there is an edge from $B$ to $A$ if there is any possibility that a ground variable of type $B$ will inﬂuence one of type $A$ . 

Definition 6.13 template dependency graph 

$A$ template dependenc raph for a template depen ncy model $\mathcal{M}_{P R M}$ contains a node for each template-level attribute A , and a directed edge from B to A whenever there is an attribute of type $B$ in $\mathrm{Pa}_{A}^{\Gamma}\cup\mathrm{Pa}_{A}$ . 

This graph can easily be constructed from the definition of the dependency model. For example, the template dependency graph of our University model (example 6.14) is shown in figure $6.9\mathbf{a}$ . 

It is not difcult to show that if template dependency graph for el $\mathcal{M}_{P R M}$ is acyclic (as in this case), it is clear that any ground network generated from M $\mathcal{M}_{P R M}$ mus be acyclic (see exercise 6.8). However, a cycle in the t e dependency graph for M $\mathcal{M}_{P R M}$ does not imply that every ground network induced by M $\mathcal{M}_{P R M}$ is cyclic. (This is the case, however, for any nondegenerate instantiation of a plate model; see exercise 6.9.) Indeed, there are template models that, although cyclic at the template levels, reﬂect a natural dependency structure whose ground networks are guaranteed to be acyclic in practice. 

Example 6.17 Consider the template dependency graph of our Genetics domain, shown in figure $6.9b.$ . The template-level self-loop involving Genotype ( Person ) reﬂects a ground-level dependency of a person’s genotype on that of his or her parents. This type of dependency can only lead to cycles in the ground network if the pedigree is cyclic, that is, a person is his/her own ancestor. Because such cases (time travel aside) are impossible, this template model cannot result in cyclic ground networks for the skeletons that arise in practice . Intuitively, in this e, we have an (acyclic) ordering $\prec$ e objects (people) in the domain, which implies u $u^{\prime}$ can be nt of u only when u $u^{\prime}\prec u_{\downarrow}$ ≺ ; therefore, Genotype ( u ) can depend on Genotype $(u^{\prime})$ only when u $u^{\prime}\prec u$ ≺ . This ordering on objects is acyclic, and therefore so is the resulting dependency structure. 

The template dependency graph does not account for these constraints on the skeleton, and therefore we cannot conclude by examining the graph whether cycles can occur in ground networks for such ordered skeletons. However, exercise 6.10 discusses a richer form of the template dependency network that explicitly incorporates such constraints, and it is therefore able to determine that our Genetics model results in acyclic ground networks for any skeleton representing an acyclic pedigree. 

So far in our discussion of acyclicity, we have largely sidestepped the issue of relational uncertainty. As we discussed, in the case of relational uncertainty, the ground network contains many “potential” edges, only a few of which will ever be “active” simultaneously. In such cases, the resulting ground network may not even be acyclic, even though it may well define a coherent (and sparse) probability distribution for every relational structure. Indeed, as we discussed in example 6.17, there are models that are potentially cyclic but are guaranteed to be acyclic by virtue of specific constraints on the dependency structure. It is thus possible to guarantee the coherence of a cyclic model of this type by ensuring that there is no positive-probability assignment to the guards that actually induces a cyclic dependence between the attributes. 

## 6.5 Undirected Representation 

The previous sections describe template-based formalisms that use a directed graphical model as their foundation. One can define similar extensions for undirected graphical models. Many of the ideas underlying this extension are fairly similar to the directed case. However, the greater ﬂexibility of the undirected representation in avoiding local normalization requirements and acyclicity constraints can be particularly helpful in the context of these richer representations. Eliminating these requirements allows us to easily encode a much richer set of patterns about the relationships between objects in the domain; see, for example, box 6.C. In particular, as we discuss in section 6.6, these benefits can be very significant when we wish to define distributions over complex relational structures. 

The basic component in a template-based undirected model is some expression, written in terms of template-level attributes with logical variables as arguments, and associated with a template factor. For a given object skeleton, each possible assignment $\gamma$ to the logical variables in the expression induces a factor in the ground undirected network, all sharing the same parameters. As for variable-based undirected representations, one can parameterize a template- based undirected probabilistic model using full factors, or using features, as in a log-linear model. This decision is largely orthogonal to other issues. We choose to use log-linear features, which are the finest-grained representation and subsume table factors. 

Let us begin by revisiting our Misconception example in section 4.1. Now, assume that we are interested in defining a probabilistic model over an entire set of students, where some number of pairs study together. We define a binary predicate (relation) Study-Pair $\!\left (S_{1}, S_{2}\right)$ , which is true when two students $S_{1}, S_{2}$ study together, and a predicate (attribute) Misconception $(S)$ that encodes the level of understanding of a student $S$ . We can now define $a$ template feature $f_{M}$ , over pairs Misconception $\left (S_{1}\right)$ , Misconception $. (S_{2})$ , which takes value 1 whenever 

$$
[S t u d y–P a i r (S_{1}, S_{2})=t r u e\wedge M i s c o n c e p t i o n (S_{1})=M i s c o n c e p t i o n (S_{2})]=t r u e
$$ 

and has value $O$ otherwise. 

Definition 6.14 relational Markov network feature argument 

object skeleton 

Example 6.19 

Definition 6.15 ground Gibbs distribution 

$A$ ional Markov network $\mathcal{M}_{R M N}$ is defined in terms of a set $\Lambda$ of template features , where each $\lambda\in\Lambda$ ∈ comprises: 

• a real-valued template feature $f_{\lambda}$ whose arguments are $\aleph (\lambda)=\{A_{1}(\pmb{U}_{1}),.\,.\,.\,, A_{l}(U_{l})\},$ ; • a weight $w_{\lambda}\in\mathbb{R}$ . We define $\alpha (\lambda)$ so that for all $i$ , $U_{i}\subseteq\alpha (\lambda)$ . In example 6.18, we have that $\alpha (\lambda_{M})=\{S_{1}, S_{2}\}$ , both of type Student ; and $\aleph (\lambda_{M})=\{S t u d y{\mathrm{-}}P a i r (S_{1}, S_{2}), M i s c o n c e p t i o n (S_{1}), M i s c o n c e p t i o n (S_{2})\}.$ 

To specify a ground network using an RMN, we must provide an object skeleton $\kappa$ that defines ts ${\mathcal{O}}^{\kappa}[\mathsf{Q}]$ for each class $\mathsf Q$ . As before, we can also define a restricted set $\Gamma_{\kappa}[A]\,\subset\,\mathcal{O}^{\kappa}[\alpha (A)]$ ⊂O . Given a skeleton, we can now define a ground Gibbs distribution in the natural way: 

Continuing example 6.18, assume we are given a skeleton containing a particular set of students and the set of study pairs within this set. This model induces a Markov network where the ground random variables have the form Misconception $(s)$ for every student s in the domain. In this model, we have a feature $f_{M}$ for every triple of variables Misconception $\left (s_{1}\right)$ , Misconception $\left (s_{2}\right)$ , Study-Pair $\left (s_{1}, s_{2}\right)$ . As usual in log-linear models, features can be associated with a weight; in this example, we might choose $w_{M}\,=\, 10$ . In this case, the unnormalized measure for a given assignment to the ground variables would be $\exp (10K)$ , where $K$ is the number of pairs $s_{1}, s_{2}$ for which equation (6.6) holds. 

More formally: 

Given an RMN M and an object skeleton $\kappa$ , we can define a ground Gibbs distribution $P_{\kappa}^{\mathcal{M}_{R M N}}$ RMN as follows: 

• The variables in the network are $\mathcal{X}_{\kappa}[\aleph]$ (as in definition 6.7); • $P_{\kappa}^{\mathcal{M}_{R M N}}$ contains a term $\exp (w_{\lambda}\cdot f_{\lambda}(\gamma))$ for each feature template $\lambda\in\Lambda$ and each assignment $\gamma\in\Gamma_{\kappa}[\alpha (\lambda)]$ . 

As always, a (ground) Gibbs distribution defines a Markov network, where we connect every pair of variables that appear together in some factor. 

In the directed setting, the dense connectivity arising in ground networks raised several concerns: acyclicity, aggregation of dependencies, and computational cost of the resulting model. The first of these is obviously not a concern in undirected models. The other two, however, deserve some discussion. 

Although better hidden, the issue of aggregating the contribution of multiple assignments of a feature also arises in undirected model. Here, the definition of the Gibbs distribution dictates the form of the aggregation we use. In this case, each grounding of the feature defines a factor in the unnormalized measure, and they are combined by a product operation, or an addition in log-space. In other words, each occurrence of the feature has a log-linear contribution to the unnormalized density. Importantly, however, this type of aggregation may not be appropriate for every application. 

Example 6.20 Consider a model for “viral marketing” — a social network of individuals related by the Friends ${\bf\mathcal{(}}P, P^{\prime})$ relation, where the attribute of interest Gadget $\mathopen{}\mathclose\bgroup\left (P\aftergroup\egroup\right)$ is the purchase of some cool new gadget $G$ . We may want to construct a model where it is more likely that two friends either both own or both do not own $G$ . That is, we have a feature similar to $\lambda_{M}$ in example 6.18. In the log-linear model, the unnormalized probability that a person $p$ purchases $G$ grows log-linearly with the number $k_{p}$ of his friends who own $G$ . However, a more realistic model may involve a saturation efect, where the impact diminishes as $k_{p}$ grows; that is, the increase in probability of Gadget $\cdot (P)$ between $k_{p}=0$ and $k_{p}=1$ is greater than the increase in probability between $k_{p}=20$ and $k_{p}=21$ . 

Thus, in concrete applications, we may wish to extend the framework to allow for other forms of combination, or even simply to define auxiliary variables corresponding to relevant aggregates (for example, the value of $k_{p}$ in our example). 

The issue of dense connectivity is as much an issue in the undirected case as in the directed case. The typical solution is similar: If we have background knowledge in the form of a relational skeleton, we can significantly simplify the resulting model. Here, the operation is very simple: we simply reduce every one of the factors in our model using the evidence contained in the relational skeleton, producing a reduced Markov network, as in definition 4.7. In this network, we would eliminate any ground variables whose values are observed in the skeleton and instantiate their (fixed) values in any ground factors containing them. In many cases, this process can greatly simplify the resulting features, often making them degenerate. 

Returning to example 6.18, assume now that our skeleton specifies the instantiation of the relation Study-Pair, so that we know exactly which pairs of students study together and which do not. Now, consider the reduced Markov network obtained by conditioning on the skeleton. As all the variables Study-Pai $r (s_{1}, s_{2})$ are observed, they are all eliminated from the network. Moreover, for any pair of students $s_{1}, s_{2}$ for which Study-Pair $\cdot (s_{1}, s_{2})=f a l s o$ e, the feature $\lambda_{M}(s_{1}, s_{2})$ necessarily takes the value 0 , regardless of the values of the other variables in the feature. Because this ground feature is vacuous and has no impact on the distribution, it can be eliminated from the model. The resulting Markov network is much simpler, containing edges only between pairs of students who study together (according to the information in the relational skeleton). 

We note that we could have introduced the notion of a guarded dependency, as we did for PRMs. However, this component is far less useful here than it was in the directed case, where it also served a role in eliminating the need to aggregate parents that are not actually present in the network and in helping clarify the acyclicity in the model. Neither of these issues arises in the undirected framework, obviating the need for the additional notational complexity. 

Finally, we mention one subtlety that is specific to the undirected setting. An undirected model uses nonlocal factors, which can have a dramatic inﬂuence on the global probability measure of the model. Thus, the probability distribution defined by an undirected relational model is not modular: Introducing a new object into the domain can drastically change the distribution over the properties of existing objects, even when the newly introduced object seems to have no meaningful interactions with the previous objects. 

Let us return to our example 6.19, and assume that any pair of students study together with some probability $p_{\cdot}$ ; that is, we have an additional template feature over Study-Pair $\!\left (S_{1}, S_{2}\right)$ that takes the value $\log{p}$ when this binary attribute is true and $\log (1-p)$ otherwise. 

Assume that we have a probability distribution over the properties of some set of students $\mathcal{O}^{\kappa}[\mathsf{S t u d e n t}]\,=\,\{s_{1},.\,.\,.\,, s_{n}\}$ , and let us study how this distribution changes if we add a new student $s_{n+1}$ . Consider an assignment to the properties of $s_{1},\ldots, s_{n}$ in which m of the n students $s_{i}$ have Misconception $(s_{i})=1$ , whereas the remaining $n-m$ have Misconc tion $(s_{i})=0$ . We can now consider the following situations with respect to $s_{n+1}$ : he studies with k of the m students for whom Misconception $(s_{i})=1$ , with $\ell$ of the $n-m$ students for whom Misconception $\left (s_{i}\right)=0$ , and himself has Misconception $\iota (s_{n+1})=c$ (for $c\in\{0,1\}.$ ). The probability of each such event is 

$$
\binom{m}{k}\binom{(n-m)}{\ell}p^{\ell}(1-p)^{(n-m-\ell)})(10^{k c}\cdot10^{\ell (1-c)}),
$$ 

where the first two terms come from the factors over the Study-Pair $\cdot (S_{1}, S_{2})$ structure, and the final term comes from the template feature $\lambda_{M}$ . We want to compute the marginal distribution over our original variables (not involving $s_{n+1}$ ), to see whether introducing $s_{n+1}$ changes this distribution. Thus, we sum out over all of the preceding events, which (using simple algebra) is $(10p+(1-p))^{m}+(10p+(1-p))^{n-m}$ . 

This analysis shows that the assignments to our original variables are multiplied by very diferent terms, depending on the value $m$ . In particular, the probability of joint assignments where $m=0$ , so that all students agree, are multiplied by a factor of $(10p+(1-p))^{n}$ , whereas the probability of joint assignments where the students are equally divided in their opinion are multiplied by $(10p+(1-p))^{n/2}$ , an exponentially smaller factor. Thus, adding a new student, even one about whom we appear to know nothing, can drastically change the properties of our probability distribution. 

Thus, for undirected models, it can be problematic to construct (by learning or by hand) a template-based model for domains of a certain size, and apply it to models of a very diferent size. The impact of the domain size on the probability distribution varies, and therefore the implications regarding our ability to apply learning in this setting need to be evaluated per application. 

Box 6. C — Case Study: Collective Classification of Web Pages. One application that calls for interesting models of interobject relationships is the classification of a network of interlinked web- pages. One example is that of a university website, where webpages can be associated with students, faculty members, courses, projects, and more. We can associate each webpage w with a hidden variable $T (w)$ whose value denotes the type of the entity to which the webpage belongs. In a standard classification setting, we would use some learned classifier to label each webpage based on its features, such as the words on the page. However, we can obtain more information by also considering the interactions between the entities and the correlations they induce over their labels. For example, an examination of the data reveals that student pages are more likely to link to faculty webpages than to other student pages. 

One can capture this type of interaction both in a directed and in an undirected model. In a directed model, we might have a binary attribute $L i n k s (W_{1}, W_{2})$ that takes the value true if $W_{1}$ links to $W_{2}$ and false otherwise. We can then have $L i n k s (W_{1}, W_{2})$ depend on $T (W_{1})$ and $T (W_{2})$ , capturing the dependence of the link probability on the classes of the two linked pages. An alternative approach is to use an undirected model, where we directly introduce a pairwise template feature over $T (W_{1}), T (W_{2})$ for pairs $W_{1}, W_{2}$ such that $L i n k s (W_{1}, W_{2})$ . Here, we can give higher potentials to pairs of types that tend to link, for example, student-faculty, faculty-course, faculty-project, project-student, and more. 

A priori, both models appear to capture the basic structure of the domain. However, the directed model has some significant disadvantages in this setting. First, since the link structure of the webpages is known, the $L i n k s (W_{1}, W_{2})$ is always observed. Thus, we have an active $\nu$ -structure connecting every pair of webpages, whether they are linked or not. The computational disadvantages of this requirement are obvious. Less obvious but equally important is the fact that there are many more non-links than links, and so the signal from the absent links tends to overwhelm the signal that could derived from the links that are present. In an undirected model, the absent links are simply omitted from the model; we simply introduce a potential that correlates the topics of two webpages only if they are linked. Therefore, an undirected model generally achieves much better performance on this task. 

Another important advantage of the undirected model for this task is its ﬂexibility in incorporating a much richer set of interactions. For example, it is often the case that a faculty member has a section in her webpage where she lists courses that she teaches, and another section that lists students whom she advises. Thus, another useful correlation that we may wish to model is one between the types of two webpages that are both linked from a third, and whose links are in close proximity on the page. We can model this type of interaction using features of the form $\begin{array}{r}{C l o s e–L i n k s (W, W_{1}, W_{2})\wedge T (W_{1})=t_{1}\wedge T (W_{2})=t_{2}}\end{array}$ , where Close-Links $\mathfrak{s}(W, W_{1}, W_{2})$ is derived directly from the structure of the page. 

Finally, an extension of the same model can be used to label not only the entities (webpages) but also the links between them. For example, we might want to determine whether a student- professor $(s, p)$ pair with a link from s to $p$ represents an advising relationship, or whether a linked professor-course pair represents an instructor relationship. Once again, a standard classifier would make use of features such as words in the vicinity of the hyperlink. At the next level, we can use an extension of the model described earlier to classify jointly both the types of the entities and the types of the links that relate them. In a more interesting extension, a relational model can also utilize higher-level patterns; for example, using a template feature over triplets of template attributes $T (W)$ , we can encode the fact that students and their advisors belong to the same research group, or that students often serve as teaching assistants in courses that their advisors teach. 

## 6.6 Structural Uncertainty $\star$ 

structural uncertainty The object-relational probabilistic models we described allow us to encode a very rich family of distributions over possible worlds. In addition to encoding distributions over the attributes of objects, these approaches can allow us to encode structural uncertainty — a probabilistic model over the actual structure of the worlds, both the set of objects they contain and the relations 

between them. The diferent models we presented exhibit significant diferences in the types of structural uncertainty that they naturally encompass. In this section, we discuss some of the major issues that arise when representing structural uncertainty, and how these issues are handled by the diferent models. 

relational uncertainty object uncertainty 

There are two main types of structural uncertainty that we can consider: relational uncertainty , which models a distribution over the presence or absence of relations between objects; and object uncertainty , which models a distribution over the existence or number of actual objects in the domain. We discuss each in turn. 

### 6.6.1 Relational Uncertainty 

The template representations we have already developed already allow us to encode uncertainty about the relational structure. As in example 6.22, we can simply make the existence of a relationship a stochastic event. What types of probability distributions over relational structure can we encode using these representational tools? In example 6.22, each possible relation Study-Pair $\left (s_{1}, s_{2}\right)$ is selected independently, at random, with probability $p$ . Unfortunately, such graphs are not representative of most relational structures that we observe in real-world settings. 

Example 6.23 

Let us select an even simpler example, where the graph we are constructing is bipartite. Consider the relation Teaches $\cdot (P, C)$ , and assume that it takes the value true with probability 0 . 1 . Consider a skeleton that contains $l0$ professors and 20 courses. Then the expected number of courses per professor is 2, and the expected number of professors per course is 1. So far, everything appears quite reasonable. However, the probability that, in the resulting graph, a particular professor teaches $\ell$ courses is distributed binomially: ${\binom{20}{\ell}}0.1^{\ell}0.9^{20-\ell}$  . For example, the probability that any single professor teaches 5 or more courses is 4.3 percent, and the probability that at least one of them does is around 29 percent. This is much higher than is realistic in real-world graphs. The situation becomes much worse if we increase the number of professors and courses in our skeleton. 

Of course, we can add parents to this attribute. For example, we can let the presence of an edge depend on the research area of the professor and the topic of the course, so that this attribute is more likely to take the value true if the area and topic match. However, this solution does not address the fundamental problem: it is still the case that, given all of the research areas and topics, the relationship status for diferent pairs of objects (the edges in the relational graph) are chosen independently. 

In this example, we wish to model certain global constraints on the distribution over the graph: the fact that each faculty member tends to teach only a small subset of courses. Unfortunately, it is far from clear how to incorporate this constraint into a template-level generative (directed) model over attributes corresponding to the presence of individual relations. Indeed, consider even the simpler case where we wish to encode the prior knowledge that each course has exactly one instructor. This model induces a correlation among all of the binary random variables corresponding to diferent instantiations $T e a c h e s (p, c)$ for diferent professors $p$ and the same course $c$ : once we have $T e a c h e s (p, c)\:=\: t r u e,$ we must have Teaches $(p^{\prime}, c)=f a l s e$ for all $p^{\prime}\neq p$ . In order to incorporate this correlation, we would have to define a generative process that “selects” the relation variables $T e a c h e s (p, c)$ in some sequence, in a way that allows each Teaches $\cdot (p^{\prime}, c)$ to depend on all of the preceding variables Teaches $(p, c)$ . This induces dependency models with dense connectivity, an arbitrary number of parents per variable (in the ground network), and a fairly complex dependency structure. 

object-valued attribute 

An alternative approach is to use a diferent encoding for the course-instructor relationship. In logical languages, an alternative mechanism for relating objects to each other is via functions. A function , or object-valued attribute takes as argument a tuple of objects from a given set of classes, and returns a set of objects in another class. Thus, for example, rather than having a relation $M o t h e r (P_{1}, P_{2})$ , we might use a function Mother- $\mathsf{o f f}(P_{1})\mapsto$ Person that takes, as argument, a person-object $p_{1}$ , and returns the person object $p_{2}$ , which is $p_{1}$ ’s mother. In this case, the return-value of the function is just a single object, but, in general, we can define functions that return an entire set of objects. In our University example, the relation Teaches defines the function Courses-Of , which maps from professors to the courses they teach, and the function Instructor , which maps from courses to the professors that teach them. We note that these functions are inverses of each other: We have that a professor $p$ is in Instructor $\cdot (c)$ if and only if $c$ is in Courses- $\cdot O f (p)$ . 

As we can see, we can easily convert between set-valued functions and relations. Indeed, as long as the relational structure is fixed, the decision on which representation to use is largely a matter of convenience (or convention). However, once we introduce probabilistic models over the relational structure, the two representations lend themselves more naturally to quite diferent types of model. Thus, for example, if we encode the course-instructor relationship as a function from professors to courses, then rather than select pairwise relations at random, we might select, for any professor $p$ , the set of courses Courses- $O f (p)$ . We can define a distribution over sets in two components: a distribution over the size $\ell$ of the set, and a distribution that then selects $\ell$ distinct objects that will make up the set. 

Assume we want to define a probability distribution over the set Courses- $\cdot O f (p)$ of courses taught by a professor $p$ . We may first define a distribution over the number $\ell$ of courses c in Courses- $O f (p)$ . This distribution may depend on properties of the professor, such as her department or her level of seniority. Given the size $\ell.$ , we now have to select the actual set of $\ell$ courses taught by $p$ . We can define a model that selects $\ell$ courses independently from among the set of courses at the university. This choice can depend on properties of both the professor and the course. For example, if the professor’s specialization is in artificial intelligence, she is more likely to teach a course in that area than in operating systems. Thus, the probability of selecting c to be in Courses- $O f (p)$ depends both on Topic $\cdot (c)$ and on Research-Area $(p)$ . Importantly, since we have already chosen $\ell=|C o u r s e s{\cdot}O f (p)|$ , we need to ensure that we actuall select $\ell$ distinct courses, that is, we must sample from the courses without replacement. Thus, our ℓ sampling events for the diferent courses cannot be completely independent. 

While useful in certain settings, this model does not solve the fundamental problem. For example, although it allows us to enforce that every professor teaches between two and four courses, it still leaves open the possibility that a single course is taught by ten professors. We can, of course, consider a model that reverses the direction of the function, encoding a distribution over the instructors of each course rather than the courses taught by a professor, but this solution would simply raise the converse problem of the possibility that a single professor teaches a large number of classes. 

It follows from this discussion that it is difcult, in generative (directed) representations, to define distributions over relational structures that guarantee (or prefer) certain structural properties of the relation. For example, there is no natural way in which we can construct a probabilistic model that exhibits (a preference for) transitivity, that is, one satisfying that if $R (u, v)$ and $R (v, w)$ then (it is more likely that) $R (u, w)$ . 

These problems have a natural solution within the undirected framework. For example, a preference for transitivity can be encoded simply as a template feature that ascribes a high value to the (template) event 

$$
R (U, V)=t r u e,R (V, W)=t r u e,R (U, W)=t r u e.
$$ 

A (soft) constraint enforcing at most one instructor per course can be encoded similarly as a (very) low potential on the template event 

$$
T e a c h e s (P, C)=t r u e, T e a c h e s (P^{\prime}, C)=t r u e.
$$ 

A constraint enforcing at least one instructor per course cannot be encoded in the framework of relational Markov networks, which allow only features with a bounded set of arguments. However, it is not difcult to extend the language to include potentials over unbounded sets of variables, as long as these potentials have a compact, finite-size representation. For example, we could incorporate an aggregator feature that counts the number $t_{p}$ of objects $c$ such that Teaches $\cdot (p, c)$ , and introduce a potential over the value of $t_{p}$ . This extension would allow us to incorporate arbitrary preferences about the number of courses taught by a professor. At the same time, the model could also include potentials over the aggregator $i_{c}$ that counts the number of instructors $p$ for a course $c$ . Thus, we can simultaneously include global preferences on both sides of the relation between courses and professors. 

However, while this approach addresses the issue of expressing such constraints, it leaves unresolved the problem of the complexity of the resulting ground network. In all of these examples, the induced ground network is very densely connected, with a ground variable for every potential edge in the relational graph (for example, $R (u, v))$ , and a factor relating every pair or even every triple of these variables. In the latter examples, involving the aggregator, we have potentials whose scope is unbounded, containing all of the ground variables $R (u, v)$ . 

### 6.6.2 Object Uncertainty 

So far, we have focused our discussion on representing probabilistic models about the presence or absence of certain relations, given a set of base objects. One can also consider settings in which even the set of objects in the world is not predetermined, and so we wish to define a probability distribution over this set. 

Perhaps the most common setting in which this type of reasoning arises is in situations where diferent objects in our domain may be equal to each other. This situation arises quite often. For example, a single person can be student $\#34$ in CS101, student $\#57$ in Econ203, the eldest daughter of John and Mary, the girlfriend of Tom, and so on. 

One solution is to allow objects in the domain to correspond to diferent “names,” or ways of referring to an object, but explicitly reason about the probability that some of these names refer to the same object. But how do we model a distribution over equality relationships between the objects playing diferent roles in the model? 

The key insight is to introduce explicitly into the model the notion of a “reference” to an object, where the same object can be referred to in several diferent ways. That is, we include in the model objects that correspond to the diferent “references” to the object. Thus, for example, we could have a class of “person objects” and another class for “person reference objects.” We can use a relation-based representation in this setting, using a relation Refers- $t o (r, p)$ that is true whenever the reference $r$ refers to a person $p$ . However, we must also introduce uniqueness constraints to ensure that a reference $r$ refers to precisely a single person $p$ . Alternatively, a more natural approach is to use a function, or object-valued attribute, Referent $\cdot (r)$ , which designates the person to whom $r$ refers. This approach automatically enforces the uniqueness constraints, and it is thus perhaps more appropriate to this application. 

In either case, the relationship between references and the objects to which they refer is generally probabilistic and interacts probabilistic ally with other attributes in the domain. In particular, we would generally introduce factors that model the similarity of the properties of a “reference object” $r$ and those of the true object $p$ to which it refers. These attribute similarity potentials can be constructed to allow for noise and variation. For example, we can model the fact that a person whose name is “John Franklin Adams” may decide to go by “J.F. Adams” in one setting and “Frank Adams” in another, but is unlikely to go by the name “Peggy Smith.” We can also model the fact that a person may decide to “round down” his or her reported age in some settings (for example, social interactions) but not in others (for example, tax forms). The problem of determining the correspondence between references and the entities to which they refer is an instance of the correspondence problem, which is described in detail in box 12.D. $\mathrm{Box}\; 6.\mathrm{D}$ describes an application of this type of model to the problem of matching bibliographical citations. 

In an alternative approach, we might go one step further, we can eliminate any mention of the true underlying objects, and restrict the model only to object references. In this solution, the domain contains only “reference objects” (at least for some classes). Now, rather than mapping references to the object to which they refer, we simply allow for diferent references to “correspond” to each other. Specifically, we might include a binary predicate Same- $a s (\boldsymbol{r},\boldsymbol{r}^{\prime})$ , which asserts that $r$ and $r^{\prime}$ both refer to the same underlying object (not included as an object in the domain). 

To ensure that Same-As is consistent with the semantics of an equality relation, we need to introduce various constraints on its properties. (Because these constraints are standard axioms of equality, we can include them as part of the formalism rather than require each user to specify them.) First, using the ideas described in section 6.6.1, we can introduce undirected

 (hard) potentials to constrain the relation to satisfy: 

• Reﬂexivity — Same- $\cdot A s (r, r)$ ;

 • Symmetry — Same $\cdot A s (r, r^{\prime})$ if and only if Same- $\mathit{A s}(\mathit{r}^{\prime},\mathit{r})$ ;

 • Transitivity — Same- $\cdot A s (r, r^{\prime})$ and Same- $4s (r^{\prime}, r^{\prime\prime})$ implies Same- $\mathbf{\nabla}_{: A s}(r, r^{\prime\prime})$ . 

These conditions imply that the Same-As relation defines an equivalence relation on reference objects, and thus partitions them into mutually exclusive and exhaustive equivalence classes. Importantly, however, these constraints can only be encoded in an undirected model, and therefore this approach to dealing with equality only applies in that setting. In addition, we include in the model attribute similarity potentials, as before, which indicate the extent to which we expect attributes or predicates for two Same-As reference objects $r$ and $r^{\prime}$ to be similar to each other. This approach, applied to a set of named objects, tends to cluster them together into groups whose attributes are similar and that participate in relations with objects that are also in equivalent groups. 

There are, however, several problems with the reference-only solution. First, there is no natural place to put factors that should apply once per underlying entity. 

Example 6.25 

Suppose we are interested in inferring people’s gender from their names. We might have a potential saying that someone named “Alex” is more likely to be male than female. But if we make this a template factor on $\{N a m e (R)$ , Gender $\cdot (R)\}$ where $R$ ranges over references, then the factor will apply many times to people with many references. Thus, the probability that a person named “Alex” is male will increase exponentially with the number of references to that person. 

A related but more subtle problem is the dependence of the outcome of our inference on the number of references. 

Example 6.26 

Consider a very simple example where we have only references to one type of object and only the at- tribute $A$ , which takes values $1,2,3$ . For each pair of object references $r, r^{\prime}$ such that Same $\cdot A s (r, r^{\prime})$ holds, we have an attribute similarity potential relating $A (r)$ and $A (r^{\prime})$ : the cases of $A (r)=A (r^{\prime})$ have the highest weight $w$ ; $A (r)=1$ , $A (r^{\prime})=3$ has very low weight; and $A (r)=2$ , $A (r^{\prime})=1$ and $A (r)=2,\ A (r^{\prime})=3$ both have the same medium potential $q$ . Now, consider the graph of people related by the Same-As relation: since Same-As is an equivalence relation, the graph is $^a$ set of mutually exclusive and exhaustive partitions, each corresponding to a set of references that correspond to the same object. Now, assume we have a configuration of evidence where we observe $k_{i}$ references with $A (r)=i$ , for $i={1,2,3}$ . The most likely assignment relative to this model will have one cluster with all the $A (r)=1$ references, and another with all the $A (r)=3$ references. What about the references with $A (r)=2?$ 

Somewhat surprisingly, their disposition depends on the relative sizes of the clusters. To under- stand why, we first note that (assuming $w>1.$ ) there are only three solutions with reasonably high probability: three separate clusters; a $^{u}\!{\boldsymbol{I}}\!\!+\!\boldsymbol{Z}^{\prime}$ and a “3” cluster; and a “1” and a $^{u}{}{{+}}3^{\prime\prime}$ cluster. All other solutions have much lower probability, and the discrepancy decays exponentially with the size of the domain. Now, consider the case where $k_{2}=1$ , so that there only one $r^{*}$ with $A (r)=2$ . If we add $r^{*}$ to the “1” cluster, we introduce an attribute similarity potential between $A (r^{*})$ and all of the $A (r)$ ’s in the “1” cluster. This multiplies the overall probability of the configuration by $q^{k_{1}}$ . Similarly, if we add $r^{*}$ to the $"3"$ cluster, the probability of the configuration is multiplied by $q^{k_{3}}$ . Thus, if $q<1$ , the reference $r^{*}$ is more likely to be placed in the smaller of the two clusters; if $q>1$ , it is more likely to be placed in the larger cluster. As $k_{2}$ grows, the optimal solution may now be one where we put the $2s$ into their own, separate cluster; the benefit of doing so depends on the relative sizes of the diferent parameters $q, w, k_{1}, k_{2}, k_{3}$ . 

Thus, in this type of model, the resulting posterior is often highly peaked, and the probabilities of the diferent high-probability outcomes very sensitive to the parameters. By contrast, a model where each equivalence cluster is associated with a single actual object is a lot “smoother,” for the number of attribute similarity potentials induced by a cluster of references grows linearly, not quadratically, in the size of the cluster. 

Box 6. D — Case Study: Object Uncertainty and Citation Matching. Being able to browse the network of citations between academic works is a valuable tool for research. For instance, given one citation to a relevant publication, one might want a list of other papers that cite the same work. There are several services that attempt to construct such lists automatically by extracting citations from online papers. This task is difcult because the citations come in a wide variety of formats, and often contain errors — owing both to the original author and to the vagaries of the extraction process. For example, consider the two citations: 

Elston R, Stewart A. A General Model for the Genetic Analysis of Pedigree Data. Hum. Hered. 1971;21:523–542. Elston RC, Stewart J (1971): A general model for the analysis of pedigree data. Hum Hered 21523–542. 

These citations refer to the same paper, but the first one gives the wrong first initial for J. Stewart, and the second one omits the word “genetic” in the title. The colon between the journal volume and page numbers has also been lost in the second citation. A citation matching system must handle this kind of variation, but must also avoid lumping together distinct papers that have similar titles and author lists. 

Probabilistic object-relational models have proven to be an efective approach to this problem. One way to handle the inherent object uncertainty is to use a directed model with a Citation class, as well as Publication and Author classes. The set of observed Citation objects can be included in the object skeleton, but the number of Publication and Author objects is unknown. 

A directed object-relational model for this problem (based roughly on the model of Milch et al. (2004)) is shown in figure 6.D.1a. The model includes random variables for the sizes of the Author and Publication classes. The Citation class has an object-valued attribute PubCited (C), whose value is the Publication object that the citation refers to. The Publication class has a set-valued attribute Authors (P), indicating the set of authors on the publication. These attributes are given very simple CPDs: for PubCited (C), we use a uniform distribution over the set of Publication objects, and for Authors (P) we use a prior for the number of contributors along with a uniform selection distribution. 

To complete this model, we include string-valued attributes Name (A) and Title (P), whose CPDs encode prior distributions over name and title strings (for now, we ignore other attributes such as date and journal name). Finally, the Citation class has an attribute Text (C), containing the observed text of the citation. The citation text attribute depends on the title and author names of the publication it refers to; its CPD encodes the way citation strings are formatted, and the probabilities of various errors and abbreviations. 

Thus, given observed values for all the $T e x t (c_{i})$ attributes, our goal is to infer an assignment of values to the PubCited attributes — which induces a partition of the citations into coreferring groups. To get a sense of how this process works, consider the two preceding citations. One hypothesis, $H_{1}$ , is that the two citations $c_{1}$ and $c_{2}$ refer to a single publication $p_{1}$ , which has “genetic” in its title. An alternative, $H_{2}$ , is that there is an additional publication $p_{2}$ whose title is identical except for the omission of “genetic,” and $c_{2}$ refers to $p_{2}$ instead. $H_{1}$ obviously involves an unlikely event — a word being left out of a citation; this is reﬂected in the probability of $T e x t (c_{2})$ given $T i t l e (p_{1})$ . But the probability of $H_{2}$ involves an additional factor for $T i t l e (p_{2})$ , reﬂecting the prior probability of the string $^ Ḋ A Ḍ$ general model for the analysis of pedigree data” under our model of academic paper titles. Since there are so many possible titles, this probability will be extremely small, allowing $H_{1}$ to win out. As this example shows, probabilistic models of this form exhibit 

![](images/a889a20ae0773f2c02713db6882988b4fd0ab7b2b55184a9fd1292c414f89dfd.jpg) 
Figure 6.D.1 — Two template models for citation-matching (a) A directed model. (b) An undirected model instantiated for three citations. 

a built-in Ockham’s razor efect: the highest probability goes to hypotheses that do not include any more objects — and hence any more instantiated attributes — than necessary to explain the observed data. 

Another line of work (for example, Wellner et al. (2004)) tackle the citation-matching problem using undirected template models, whose ground instantiation is a CRF (as in section 4.6.1). As we saw in the main text, one approach is to eliminate the Author and Publication classes and simply reason about a relation $S a m e (C, C^{\prime})$ between citations (constrained to be an equivalence relation). Figure 6.D.1b shows an instantiation of such a model for three citations. For each pair of citations $C, C^{\prime}$ , there is an array of factors $\phi_{1},\ldots,\phi_{k}$ that look at various features of $T e x t (C)$ and $T e x t (C^{\prime})$ — whether they have same surname for the first author, whether their titles are within an edit distance of two, and so on — and relate these features to $S a m e (C_{1}, C_{2})$ . These factors encode preferences for and against coreference more explicitly than the factors in the directed model. 

However, as we have discussed, a reference-only model produces overly peaked posteriors that are very sensitive to parameters and to the number of mentions. Moreover, there are some examples where pairwise compatibility factors are insufcient for finding the right partition. For instance, suppose we have three references to people: “Jane,” which is clearly $a$ female’s given name; “Smith,” which is clearly a surname; and “Stanley,” which could be a surname or a male’s given name. Any pair of these references could refer to the same person: there could easily be a Jane Smith, a Stanley Smith, or a Jane Stanley. But it is unlikely that all three names corefer. Thus, a reasonable approach uses an undirected model that has explicit (hidden) variables for each entity and its attributes. The same potentials can be used as in the reference-only model. However, due to the use of undirected dependencies, we can allow the use of a much richer feature set, as described in box 4.E. 

Systems that use template-based probabilistic models can now achieve accuracies in the high 90s for identifying coreferent citations. Identifying multiple mentions of the same author is harder; accuracies vary considerably depending on the data set, but tend to be around 70 percent. These models are also useful for segmenting citations into fields such as the title, author names, journal, and date. This is done by treating the citation text not as a single attribute but as a sequence of tokens (words and punctuation marks), each of which has an associated variable indicating which field it belongs to. These “field” variables can be thought of as the state variables in a hidden Markov model in the directed setting, or a conditional random field in the undirected setting (as in box 4. E). The resulting model can segment ambiguous citations more accurately than one that treats each citation in isolation, because it prefers for segmentations of coreferring citations to be consistent. 

## 6.7 Summary 
The representation languages discussed in earlier chapters — Bayesian networks and Markov networks — allow us to write down a model that encodes a specific probability distribution over a fixed, finite set of random variables. In this chapter, we have provided a general frame- work for defining templates for fragments of the probabilistic model. These templates can be reused both within a single model, and across multiple models of diferent structures. Thus, a template-based representation language allows us to encode a potentially infinite set of distributions, over arbitrarily large probability spaces. The rich models that one can produce from such a representation can capture complex interactions between many interrelated objects, and thus utilize many pieces of evidence that we may otherwise ignore; as we have seen, these pieces of evidence can provide substantial improvements in the quality of our predictions. 
> 本章提供了定义概率模型模板的通用框架
> 这些模板可以在单个模型中复用，也可以跨模型复用
> 基于模板的表示允许我们在任意大的概率空间编码无限的分布

We described several diferent representation languages: one specialized to temporal representations, and several that allow the specification of models over general object-relational domains. In the latter category, we first described two directed representations: plate models, and probabilistic relational models. The latter allow a considerably richer set of dependencies to be encoded, but at the cost of both conceptual and computational complexity. We also described an undirected representation, which, by avoiding the need to guarantee acyclicity and coherent local probability models, avoids some of the complexities of the directed models. As we discussed, the ﬂexibility of undirected models is particularly valuable when we want to encode a probability distribution over richer representations, such as the structure of the relational graph. 

There are, of course, other ways to produce these large, richly structured models. Most obviously, for any given application, we can define a procedural method that can take a skeleton, and produce a concrete model for that specific set of objects (and possibly relations). For example, we can easily build a program that takes a pedigree and produces a Bayesian network for genetic inheritance over that pedigree. The benefit of the template-based representations that we have described here is that they provide a uniform, modular, declarative language for models of this type. Unlike specialized representations, such a language allows the template-based model to be modified easily, whether by hand or as part of an automated learning algorithm. Indeed, learning is perhaps one of the key advantages of the template-based representations. In particular, as we will discuss, the model is learned at the template level, allowing a model to be learned from a domain with one set of objects, and applied seamlessly to a domain with a completely diferent set of objects (see section 17.5.1.2 and section 18.6.2). 

In addition, by making objects and relations first-class citizens in the model, we have laid a foundation for the option of allowing probability distributions over probability spaces that are significantly richer than simply properties of objects. For example, as we saw, we can consider modeling uncertainty about the network of interrelationships between objects, and even about the actual set of objects included in our domain. These extensions raise many important and difcult questions regarding the appropriate type of distribution that one should use for such richly structured probability spaces. These questions become even more complex as we introduce more of the expressive power of relational languages, such as function symbols, quantifiers, and more. These issues are an active area of research. 

These representations also raise important questions regarding inference. At first glance, the problem appears straightforward: The semantics for each of our representation languages depends on instantiating the template-based model to produce a specific ground network; clearly, we can simply run standard inference algorithms on the resulting network. This approach is has been called knowledge-based model construction , because a knowledge-base (or skeleton) is used to construct a model. However, this approach is problematic, because the models produced by this process can pose a significant challenge to inference algorithms. First, the network produced by this process is often quite large — much larger than models that one can reasonably construct by hand. Second, such models are often quite densely connected, due to the multiple interactions between variables. Finally, structural uncertainty, both about the relations and about the presence of objects, also makes for densely connected models. On the other side, such models often have unique characteristics, such as multiple similar fragments across the network, or large amounts of context-specific independence, which could, perhaps, be exploited by an appropriate choice of inference algorithm. Chapter 15 presents some techniques for addressing the inference problems in temporal models. The question of inference in the models defined by the object-relational frameworks — and specifically of inference algorithms that exploit their special structure — is very much a topic of current work. 
# 7 Gaussian Network Models 
Although much of our presentation focuses on discrete variables, we mentioned in chapter 5 that the Bayesian network framework, and the associated results relating independencies to factorization of the distribution, also apply to continuous variables. The same statement holds for Markov networks. However, whereas table CPDs provide a general-purpose mechanism for describing any discrete distribution (albeit potentially not very compactly), the space of possible parameterizations in the case of continuous variables is essentially unbounded. 
> 贝叶斯网络和 Markov 网络对于分布的分解定理对于连续变量也是成立的
> 连续变量的可能的参数化空间是无界的

In this chapter, we focus on a type of continuous distribution that is of particular interest: the class of multivariate Gaussian distributions. Gaussians are a particularly simple subclass of distributions that make very strong assumptions, such as the exponential decay of the distribution away from its mean, and the linearity of interactions between variables. While these assumptions are often invalid, Gaussians are nevertheless a surprisingly good approximation for many real-world distributions. Moreover, the Gaussian distribution has been generalized in many ways, to nonlinear interactions, or mixtures of Gaussians; many of the tools developed for Gaussians can be extended to that setting, so that the study of Gaussian provides a good foundation for dealing with a broad class of distributions. 
> 本章介绍多元高斯分布
> 高斯分布对分布的结构做出了很强的假设，例如从分布均值向其他方向的指数衰减、变量交互的线性性质等
> 这些假设在现实问题不一定成立，但高斯分布同样存在拓展，例如拓展到非线性交互、混合高斯等

In the remainder of this chapter, we ﬁrst review the class of multivariate Gaussian distributions and some of its properties. We then discuss how a multivariate Gaussian can be encoded using probabilistic graphical models, both directed and undirected. 

## 7.1 Multivariate Gaussians 
### 7.1.1 Basic Parameterization 
We have already described the univariate Gaussian distribution in chapter 2. We now describe its generalization to the multivariate case. As we discuss, there are two diferent parameterizations for a joint Gaussian density, with quite diferent properties. 
> 对于联合高斯密度有两种参数化，二者的性质不同

The univariate Gaussian is deﬁned in terms of two parameters: a mean and a variance. In its most common representation, a multivariate Gaussian distribution over $X_{1},\dots,X_{n}$ is characterized by an $n$ -dimensional mean vector $\mu$ , and a symmetric $n\times n$ covariance matrix $\Sigma$ ; the density function is most often deﬁned as: 

$$
p({\pmb x})=\frac{1}{(2\pi)^{n/2}|{\Sigma}|^{1/2}}\exp\left[-\frac{1}{2}({\pmb x}-{\pmb\mu})^{T}{\Sigma}^{-1}({\pmb x}-{\pmb\mu})\right]\tag{7.1}
$$

where $|\Sigma|$ is the determinant of $\Sigma$ . 
>  $X_, \dots, X_n$ 上的多元高斯分布相关的参数是 $n$ 维均值向量 $\pmb \mu$ 和对称的协方差矩阵 $\Sigma$ ($\Sigma_{ij} = \text {cov} (X_i, X_j)$)，密度函数定义如上
>  其中 $|\Sigma|$ 为 $\Sigma$ 的行列式

We extend the notion of a standard Gaussian to the multidimensional case, deﬁning it to be a Gaussian whose mean is the all-zero vector 0 and whose covariance matrix is the identity matrix $I$ , which has 1 ’s on the diagonal and zeros elsewhere. The multidimensional standard Gaussian is simply a product of independent standard Gaussians for each of the dimensions. 
> 标准的多元高斯分布其均值向量为全零向量，协方差矩阵为单位矩阵 $I$
> 标准的多元高斯分布本质是由每一个维度的相互独立的标准单元高斯分布相乘得到的

In order for this equation to induce a well-deﬁned density (that integrates to 1), the matrix $\Sigma$ must be positive deﬁnite : for any $\mathbf{\Psi}^{\mathcal{X}}\in I\!\!R^{n}$ such that $\pmb{x}\neq0$ , we have that ${\pmb x}^{T}\Sigma{\pmb x}>0$ . Positive deﬁnite matrices are guaranteed to be nonsingular, and hence have nonzero determinant, a necessary requirement for the coherence of this deﬁnition. A somewhat more complex deﬁnition can be used to generalize the multivariate Gaussian to the case of a positive semi-deﬁnite covariance matrix: for any $\pmb{x}\in\mathbb{R}^{n}$ , we have that ${\pmb x}^{T}\Sigma{\pmb x}\geq0$ . This extension is useful, since it allows for singular covariance matrices, which arise in several applications. For the remainder of our discussion, we focus our attention on Gaussians with positive deﬁnite covariance matrices. 
>为了让式 7.1 是良定义的(积分值为1)，矩阵 $\Sigma$ 必须是正定的：对于任何 $\pmb x\in \mathbb{R}^{n}, \pmb{x}\neq0$，我们有 ${\pmb x}^{T}\Sigma{\pmb x}>0$ 。
>正定矩阵保证是非奇异的，因此具有非零行列式 (正定矩阵所有特征值为正数)，这是这个定义一致性的必要条件 (正定矩阵保证线性变化不会降维) 
>一个更复杂的定义将多元高斯分布推广到协方差矩阵半正定的情况：对于任何 $\pmb{x}\in\mathbb{R}^{n}$，我们有 ${\pmb x}^{T}\Sigma{\pmb x}\geq0$。这种扩展是有用的，因为它允许奇异的协方差矩阵 (半正定矩阵允许零特征值)，在许多应用中都会出现这种情况
>在我们讨论的剩余部分，我们将重点关注协方差矩阵为正定的高斯分布

Because positive deﬁnite matrices are invertible, one can also utilize an alternative parameterization, where the Gaussian is deﬁned in terms of its inverse covariance matrix $J=\Sigma^{-1}$ , called information matrix (or precision matrix). This representation induces an alternative form for the Gaussian density. Consider the expression in the exponent of equation (7.1): 

$$
\begin{array}{r c l}{{-\displaystyle\frac{1}{2}({\boldsymbol x}-{\boldsymbol\mu})^{T}\Sigma^{-1}({\boldsymbol x}-{\boldsymbol\mu})}}&{{=}}&{{-\displaystyle\frac{1}{2}({\boldsymbol x}-{\boldsymbol\mu})^{T}J({\boldsymbol x}-{\boldsymbol\mu})}}\\ {{}}&{{=}}&{{-\displaystyle\frac{1}{2}\left[{\boldsymbol x}^{T}J{\boldsymbol x}-2{\boldsymbol x}^{T}J{\boldsymbol\mu}+{\boldsymbol\mu}^{T}J{\boldsymbol\mu}\right].}}\end{array}
$$ 
The last term is constant, so we obtain: 

$$
p(\pmb{x})\propto\exp\left[-\frac12\pmb{x}^{T}J\pmb{x}+(J\pmb{\mu})^{T}\pmb{x}\right].\tag{7.2}
$$ 
This formulation of the Gaussian density is generally called the information form , and the vector $h=J\mu$ is called the potential vector . The information form deﬁnes a valid Gaussian density if and only if the information matrix is symmetric and positive deﬁnite, since $\Sigma$ is positive deﬁnite if and only if $\Sigma^{-1}$ is positive deﬁnite. The information form is useful in several settings, some of which are described here. 
> 另一种参数化利用了协方差矩阵的正定性质
> 因为正定矩阵一定有逆，我们定义信息/精度矩阵 $J = \Sigma^{-1}$ 为协方差矩阵的逆
> 我们将式 7.1 中的指数项中的 $\Sigma^{-1}$ 用 $J$ 替代，经过化简，就得到式 7.2
> 该形式的高斯密度称为信息形式，向量 $\pmb h = J\pmb \mu$ 称为势能向量
> 信息形式当且仅当信息矩阵 $J$ 是对称且正定时才定义有效的高斯分布 (正定的协方差矩阵就满足这一要求，$\Sigma$ 和 $\Sigma^{-1}$ 的正定性是一致的)

Intuitively, a multivariate Gaussian distribution speciﬁes a set of ellipsoidal contours around the mean vector $\mu$ . The contours are parallel, and each corresponds to some particular value of the density function. The shape of the ellipsoid, as well as the “steepness” of the contours, are determined by the covariance matrix $\Sigma$ . 
> 直观上，多元高斯分布指定了围绕均值向量 $\pmb \mu$ 的一组平行的椭圆等高线，每一条对应于密度函数的特定值
> 椭圆的形状和等高线的陡峭程度由协方差矩阵决定

Figure 7.1 shows two multivariate Gaussians, one where the covariances are zero, and one where they are positive. 
>图7.1展示了两个多元高斯分布，一个是协方差为零的情况，另一个是协方差为正的情况

As in the univariate case, the mean vector and covariance matrix correspond to the ﬁrst two moments of the normal distribution. In matrix notation, $\pmb{\mu}=\pmb{{\cal E}}[\pmb{X}]$ and $\Sigma=E[X X^{T}]-E[X]E[X]^{T}$ − . Breaking this expression down to the level of individual variables, we have that $\mu_{i}$ is the mean of $X_{i}$ , $\Sigma_{i,i}$ is the variance of $X_{i}$ , and $\Sigma_{i,j}\;=\;\Sigma_{j,i}$ (for $i\ne j)$ ) is the covariance between $X_{i}$ and $X_{j}$ : $\mathbf{C}o v[X_{i};X_{j}]=\mathbf{E}[X_{i}X_{j}]-\mathbf{E}[X_{i}]\mathbf{E}[X_{j}]$ . 
>正如单变量情况一样，均值向量和协方差矩阵对应于正态分布的前两阶矩，即 $\pmb{\mu}=\pmb{E}[\pmb{X}]$ 和 $\Sigma=\pmb E[\pmb X\pmb X^{T}]-\pmb E[\pmb X]\pmb E[\pmb X]^{T}$。将这个表达式分解到单个变量的层面，我们有：
>- $\mu_{i}$ 是 $X_{i}$ 的均值，
>- $\Sigma_{i,i}$ 是 $X_{i}$ 的方差，
>- $\Sigma_{i,j} = \Sigma_{j,i}$ （对于 $i \neq j$） 是 $X_{i}$ 和 $X_{j}$ 之间的协方差：$\mathbf{Cov}[X_{i};X_{j}]=\pmb{E}[X_{i}X_{j}]-\pmb{E}[X_{i}]\pmb{E}[X_{j}]$。
>
>简而言之，多元高斯分布的均值向量和协方差矩阵分别对应于各个变量的均值、方差以及不同变量之间的协方差

Example 7.1 
Consider a particular joint distribution p(X1; X2; X3) over three random variables. We can parameterize it via a mean vector $\mu$ and a covariance matrix $\Sigma$ : 

$$
\mu=\left(\begin{array}{r}{{1}}\\ {{-3}}\\ {{4}}\end{array}\right)\qquad\quad\Sigma=\left(\begin{array}{r r r}{{4}}&{{2}}&{{-2}}\\ {{2}}&{{5}}&{{-5}}\\ {{-2}}&{{-5}}&{{8}}\end{array}\right)
$$ 
As we can see, the covariances $\pmb{C}o\nu[X_{1};X_{3}]$ and $C o v[X_{2};X_{3}]$ are both negative. Thus, $X_{3}$ is negatively correlated with $X_{1}$ : when $X_{1}$ goes up, $X_{3}$ goes down (and similarly for $X_{3}$ and $X_{2}$ ). 
> $X_1, X_3$ 协方差为 0 -> $X_1, X_3$ 负相关 -> 总体趋势上，$X_1$ 上升，$X_3$ 下降，或者说二者远离各自均值的方向不同

> 协方差计算公式：

$$
\begin{align}
\text{cov}(X_1, X_2)&=E[(X_1 - E[X_1])(X_2 - E[X_2])]\\
&=E[X_1X_2 - X_1E[X_2] - X_2 E[X_1] + E[X_1]E[X_2]]\\
&=E[X_1X_2] - E[X_1]E[X_2]
\end{align}
$$

### 7.1.2 Operations on Gaussians 
There are two main operations that we wish to perform on a distribution: compute the marginal distribution over some subset of the variables $Y$ , and conditioning the distribution on some assignment of values $Z=z$ . It turns out that each of these operations is very easy to perform in one of the two ways of encoding a Gaussian, and not so easy in the other. 
> 对于分布我们常执行的两种计算：在某个变量子集 $\pmb Y$ 上计算边际分布、将分布条件于某个变量赋值 $\pmb Z = \mathbf z$ ，对于之前的两种参数化的高斯分布都容易完成

Marginalization is trivial to perform in the covariance form. Speciﬁcally, the marginal Gaussian distribution over any subset of the variables can simply be read from the mean and covariance matrix. For instance, in example 7.1, we can obtain the marginal Gaussian distribution over $X_{2}$ and $X_{3}$ by simply considering only the relevant entries in both the mean vector the covariance matrix. 
> 对于协方差的参数化形式，某个变量子集的边际分布可以直接从均值向量和协方差矩阵中读出来，只需要其中查看相关的 entries 即可

More generally, assume that we have a joint normal distribution over $\{X,Y\}$ where $X\,\in\,I\!\!R^{n}$ and $\pmb{Y}\,\in\,I\!\!R^{m}$ . Then we can decompose the mean and covariance of this joint distribution as follows: 

$$
p(\pmb X,\pmb Y)=\mathcal{N}\left({\left(\begin{array}{l}{\pmb \mu_{\pmb X}}\\ {\pmb \mu_{\pmb Y}}\end{array}\right)};\left[\begin{array}{l l}{\Sigma_{\pmb X \pmb X}}&{\Sigma_{\pmb X \pmb Y}}\\ {\Sigma_{\pmb Y \pmb X}}&{\Sigma_{\pmb Y \pmb Y}}\end{array}\right]\right)\tag{7.3}
$$ 
where  $\pmb \mu_{X}\in{I\!\!R}^{n},\,\pmb \mu_{Y}\in{I\!\!R}^{m},\,\Sigma_{X X}$ is a matrix of size $n\times n,\,\Sigma_{X Y}$ is a matrix of size $n\times m$ , $\Sigma_{Y X}=\Sigma_{X T}^{T}$ is a matrix of size $m\times n$ and $\Sigma_{Y Y}$ is a matrix of size $m\times m$   
> 一般地，如果我们有 $\pmb X \in \mathbb R^n, \pmb Y \in \mathbb R^m$ 上的联合高斯分布，我们可以按照式 7.3 分解该联合分布的均值和协方差

**Lemma 7.1** Let $\{X,Y\}$ ave a joint normal distribution deﬁned in equation (7.3). Then the marginal distri- bution over $Y$ is a normal distribution $\mathcal{N}\left(\mu_{Y};\Sigma_{Y Y}\right)$ . 
> 引理：
> $\pmb X, \pmb Y$ 服从式 7.3 定义的联合高斯分布，则 $\pmb Y$ 上的边际分布是高斯分布 $\mathcal N (\pmb \mu_{\pmb Y}, \Sigma_{\pmb Y\pmb Y}$)
> (高斯分布的一个基本性质就是高斯分布的边缘分布仍然是高斯分布)

The proof follows directly from the deﬁnitions (see exercise 7.1). 

On the other hand, conditioning a Gaussian on an observation $Z=z$ is very easy to perform in the information form. We simply assign the values $Z=z$ in equation (7.2). This process turns some of the quadratic terms into linear terms or even constant terms, and some of the linear terms into constant terms. The resulting expression, however, is still in the same form as in equation (7.2), albeit over a smaller subset of variables. 
> 将高斯分布条件于某个观测 $\pmb Z = \mathbf z$ 则在信息形式容易执行，我们在式 7.2 中直接赋值 $\pmb Z = \mathbf z$，这会将部二次项转化为线性项，将部分线性项转化为常数项
> 最后得到的表达式形式仍然是式 7.2，此时仅基于一个更小的变量子集

In summary, although the two representations both encode the same information, they have diferent computational properties. To marginalize a Gaussian over a subset of the variables, one essentially needs to compute their pairwise covariances, which is precisely generating the distribution in its covariance form. Similarly, to condition a Gaussian on an observation, one essentially needs to invert the covariance matrix to obtain the information form. For small matrices, inverting a matrix may be feasible, but in high-dimensional spaces, matrix inversion may be far too costly. 
> 小结：两种表示编码了相同信息，但计算性质不同
> 信息形式需要转置协方差矩阵，这在高维空间会过于昂贵

### 7.1.3 Independencies in Gaussians 
For multivariate Gaussians, independence is easy to determine directly from the parameters of the distribution. 
> 多元高斯分布可以直接从分布的参数中决定变量之间的独立性

**Theorem 7.1** 
Let $X=X_{1},...,X_{n}$ have a joint normal distribution $\mathcal{N}\left(\boldsymbol{\mu};\boldsymbol{\Sigma}\right)$ . Then $X_{i}$ and $X_{j}$ are independent if and only if $\Sigma_{i,j}=0$ . 
> 定理：
> $\pmb X  = X_1, \dots, X_2$ 有联合高斯分布 $\mathcal N (\pmb \mu; \Sigma)$，则当且仅当 $\Sigma_{ij} = 0$ ，$X_i, X_j$ 相互独立

The proof is left as an exercise (exercise 7.2). 

>证明：
>**必要性：** 如果 $X_i, X_j$ 相互独立，则 $\Sigma_{ij} = 0$
>假设 $X_i, X_j$ 相互独立。根据独立性的定义，随机变量 $X_i$ 和 $X_j$ 的联合概率密度函数等于它们各自边缘概率密度函数的乘积，即：

$$ p(X_i, X_j) = p(X_i) \cdot p(X_j) $$

>联合高斯分布中，协方差矩阵的非对角线元素表示了不同随机变量之间的线性相关性，如果两个随机变量相互独立，它们之间的线性相关性为零，故其协方差矩阵 $\Sigma$ 中对应于 $X_i$ 和 $X_j$ 之间的协方差项 $\Sigma_{ij}$ 必须为零。
>
>**充分性**：如果 $\Sigma_{ij} = 0$，则 $X_i, X_j$ 相互独立。
>假设 $\Sigma_{ij} = 0$。我们需要证明 $X_i$ 和 $X_j$ 相互独立。根据高斯分布的性质，如果 $\Sigma_{ij} = 0$，则 $X_i$ 和 $X_j$ 之间的线性相关性为零。
>在高斯分布的情况下，从两个随机变量之间没有线性关系是可以推出它们是统计上独立的。
>
>具体来说，对于联合高斯分布 $\mathcal{N}(\pmb \mu; \Sigma)$，如果协方差矩阵 $\Sigma$ 中的某个非对角线元素 $\Sigma_{ij} = 0$，那么 $X_i$ 和 $X_j$ 的联合分布可以分解为两个独立的边缘分布的乘积：

$$ p(X_i, X_j) = p(X_i) \cdot p(X_j) $$

>这是因为高斯分布的联合概率密度函数可以写成：

$$ p(\pmb X) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\pmb X - \pmb \mu)^T \Sigma^{-1} (\pmb X - \pmb \mu)\right) $$

>当 $\Sigma_{ij} = 0$ 时，协方差矩阵 $\Sigma$ 的逆矩阵 $\Sigma^{-1}$ 中对应的元素也为零，这使得联合概率密度函数可以分解为：

$$ p(X_i, X_j) = \frac{1}{(2\pi)^{1/2} \sqrt{\sigma_{ii}}} \exp\left(-\frac{1}{2} \frac{(X_i - \mu_i)^2}{\sigma_{ii}}\right) \cdot \frac{1}{(2\pi)^{1/2} \sqrt{\sigma_{jj}}} \exp\left(-\frac{1}{2} \frac{(X_j - \mu_j)^2}{\sigma_{jj}}\right) $$

>这表明 $X_i$ 和 $X_j$ 的联合分布确实可以分解为它们各自边缘分布的乘积，从而证明了它们的独立性。

Note that this property does not hold in general. In other words, if $p(X,Y)$ is not Gaussian, then it is possible that $C o v[X;Y]=0$ while $X$ and $Y$ are still dependent in $p$ . (See exercise 7.2.) 
> 在高斯分布中，线性不相关可以直接表示统计上的不相关，但在其他分布中不一定

At ﬁrst glance, it seems that conditional independencies are not quite as apparent as marginal independencies. However, it turns out that the independence structure in the distribution is apparent not in the covariance matrix, but in the information matrix. 
> 协方差矩阵容易看出边际独立性，信息矩阵容易看出条件独立性

**Theorem 7.2** 
Consider a Gaussian distribution $p(X_{1},.\,.\,.\,,X_{n})=\mathcal{N}\left(\pmb{\mu};\Sigma\right)$ , and let $J=\Sigma^{-1}$ be the information matrix. Then ${{J}_{i,j}}=0$ if and only if $\cdot p=(X_{i}\perp X_{j}\mid\mathcal{X}-\{X_{i},X_{j}\})$ ⊥ | X −{ } . 
> 定理：
> 考虑高斯分布 $p (X_1, \dots, X_n) = \mathcal N (\pmb \mu; \Sigma)$，$J = \Sigma^{-1}$ 为信息矩阵，则 $J_{ij} = 0$ 当且仅当 $p \vDash (X_1 \perp X_j \mid \mathcal X - \{X_1, X_j\}$)
> ($J_{ij} \ne 0$ 意味着 $X_j, X_i$ 之间存在直接的相互作用，因此给定其他所有变量后，二者仍然相互依赖；$J_{ij} = 0$ 意味着 $X_j, X_i$ 之间没有直接的相互作用，二者的相关性是通过其他变量间接产生的，因此二者在给定其他变量时条件独立)

The proof is left as an exercise (exercise 7.3). 

Example 7.2
Consider the covariance matrix of example 7.1. Simple algebraic operations allow us to compute its inverse: 

$$
J=\left(\begin{array}{c c c}{{0.3125}}&{{-0.125}}&{{0}}\\ {{-0.125}}&{{0.5833}}&{{0.3333}}\\ {{0}}&{{0.3333}}&{{0.3333}}\end{array}\right)
$$ 
As we can see, the entry in the matrix corresponding to $X_{1},X_{3}$ is zero, reﬂecting the fact that they are conditionally independent given $X_{2}$ . 

Theorem 7.2 asserts the fact that the information matrix captures independencies between pairs of variables, conditioned on all of the remaining variables in the model. These are precisely the same independencies as the pairwise Markov independencies of deﬁnition 4.10. Thus, we can view the information matrix $J$ for a Gaussian density $p$ as precisely capturing the pairwise Markov independencies in a Markov network representing $p$ . Because a Gaussian density is a positive distribution, we can now use theorem 4.5 to construct a Markov network that is a unique minimal I-map for $p$ : As stated in this theorem, the construction simply introduces an edge between $X_{i}$ and $X_{j}$ whenever $\left(X_{i}\perp X_{j}\mid{\mathcal{X}}-\{X_{i},X_{j}\}\right)$ does not hold in $p$ . But this latter condition holds precisely when $J_{i,j}\,\ne\,0$ ̸ . **Thus, we can ew the information matrix as directly deﬁning a minimal I-map Markov network for p , whereby nonzero entries correspond to edges in the network.** 
> 定理 7.2 表明信息矩阵 $J$ 捕获了成对变量之间的给定其他所有变量时的条件独立性，这恰好也就是成对 Markov 独立性的定义
> 因此，我们可以将一个高斯密度 $p$ 的信息矩阵 $J$ 视作精确地捕获了表示 $p$ 的 Markov 网络中的成对 Markov 独立性
> 因为高斯密度是正分布，因此我们可以根据定理 4.5 构造是 $p$ 的唯一极小 I-map 的 Markov 网络，构造时，我们为 $(X_i \perp X_j \mid \mathcal X - \{X_i, X_j\}$ 不成立的 $X_i, X_j$ 之间引入一条边，也就是等价于 $J_{ij} \ne 0$ 时为 $X_i, X_j$ 引入一条边。因此 $J$ 也可以视为直接定义了 $p$ 的极小 I-mapMarkov 网络，$J$ 中的非零项就对应于网络中的一条边

## 7.2 Gaussian Bayesian Networks 
We now show how we can deﬁne a continuous joint distribution using a Bayesian network. This representation is based on the linear Gaussian model , which we deﬁned in deﬁnition 5.14. Although this model can be used as a CPD within any network, it turns out that continuous networks deﬁned solely in terms of linear Gaussian CPDs are of particular interest: 
> 本小节展示使用贝叶斯网络定义连续联合分布

**Deﬁnition 7.1**  Gaussian Bayesian network 
We define a Gaussian Bayesian network to be a Bayesian network all of whose variables are continuous, and where all of the CPDs are linear Gaussians.
> 定义：
> 对于一个贝叶斯网络，如果它的所有变量都是连续变量，并且所有的 CPD 都是线性高斯模型，则该网络就是高斯贝叶斯网络

An important and surprising result is that linear Gaussian Bayesian networks are an alternative representation for the class of multivariate Gaussian distributions. . 
> 线性高斯贝叶斯网络实际上是一类多元高斯分布的替代性表示

This result has two parts. The ﬁrst is that a linear Gaussian network always deﬁnes a joint multivariate Gaussian distribution
> 线性高斯网络总是定义一个联合多元高斯分布：

**Theorem 7.3** 
Let $Y$ be the linear Gaussian of its parents $X_1, \dots, X_k$

$$
p(Y\mid\mathbf{\boldsymbol{x}})=\mathcal{N}\left(\beta_{0}+\pmb \beta^{T}\mathbf{\boldsymbol{x}};\sigma^{2}\right).
$$ 
Assume that $X_{1},\ldots,X_{k}$ are jointly Gaussian with distribution $\mathcal{N}\left(\boldsymbol{\mu};\boldsymbol{\Sigma}\right)$ . Then: 

- The distribution of $Y$ is a normal distribution $p(Y)=\mathcal{N}\left(\mu_{Y};\sigma_{Y}^{2}\right)$ where: 

$$
\begin{array}{r c l}{{\mu_{Y}}}&{{=}}&{{\beta_{0}+\pmb \beta^{T}\pmb \mu}}\\ {{\sigma_{Y}^{2}}}&{{=}}&{{\sigma^{2}+\pmb \beta^{T}\Sigma\pmb \beta.}}\end{array}
$$ 
- The joint distribution over $\{X,Y\}$ is a normal distribution where: 

$$
\pmb{C}o v[X_{i};Y]=\sum_{j=1}^{k}\beta_{j}\Sigma_{i,j}.
$$ 
> 定理：
> $Y$ 是其父变量 $X_1,\dots, X_k$ 的线性高斯模型，假设 $X_1, \dots, X_k$ 服从联合高斯分布 $\mathcal N (\pmb \mu , \Sigma)$，则：
>  $Y$ 也服从高斯分布，其均值和方差分别和 $\pmb \mu, \Sigma$ 相关
>  $\pmb X, Y$ 上的联合分布也是高斯分布，其中 $X_i, Y$ 的协方差等于 $\beta_j \Sigma_{ij}$ 对所有 $j$ 求和 ($\Sigma_{ij}$ 表示 $X_i, X_j$ 的协方差，$\beta_j$ 是 $X_j$ 和 $Y$ 相关的系数)

From this theorem, it follows easily by induction that if $\mathcal{B}$ is a linear Gaussian Bayesian network, then it deﬁnes a joint distribution that is jointly Gaussian. 
> 线性高斯网络的所有 CPD 都是线性高斯模型，因此根据该定理，线性高斯网络 $\mathcal B$ 就定义了一个联合的多元高斯分布

Example 7.3 Consider the linear Gaussian network $X_{1}\rightarrow X_{2}\rightarrow X_{3}$ , where 

$$
\begin{array}{r c l}{p(X_{1})}&{=}&{\mathcal{N}\left(1;4\right)}\\ {p(X_{2}\mid X_{1})}&{=}&{\mathcal{N}\left(0.5X_{1}-3.5;4\right)}\\ {p(X_{3}\mid X_{2})}&{=}&{\mathcal{N}\left(-X_{2}+1;3\right).}\end{array}
$$ 
Using the equations in theorem 7.3, we can compute the joint Gaussian distribution $p(X_{1},X_{2},X_{3})$ . For the mean, we have that: 

$$
\begin{array}{l l l}{{\mu_{2}}}&{{=}}&{{0.5\mu_{1}-3.5=0.5\cdot1-3.5=-3}}\\ {{\mu_{3}}}&{{=}}&{{(-1)\mu_{2}+1=(-1)\cdot(-3)+1=4.}}\end{array}
$$ 
The variance of $X_{2}$ and $X_{3}$ can be computed as: 

$$
\begin{array}{r c l}{{\Sigma_{22}}}&{{=}}&{{4+(1/2)^{2}\cdot4=5}}\\ {{\Sigma_{33}}}&{{=}}&{{3+(-1)^{2}\cdot5=8.}}\end{array}
$$

We see that the variance of the variable is a sum of two terms: the variance arising from its own Gaussian noise parameter, and the variance of its parent variables weighted by the strength of the dependence. 
> 可以看到，本例中，变量的方差是两项的和：一项是自己的高斯噪声参数，一项是其父变量的方差乘上依赖性计算得到的权重

Finally, we can compute the covariances as follows: 

$$
\begin{array}{l c l}{{\Sigma_{12}}}&{{=}}&{{(1/2)\cdot4=2}}\\ {{\Sigma_{23}}}&{{=}}&{{(-1)\cdot\Sigma_{22}=-5}}\\ {{\Sigma_{13}}}&{{=}}&{{(-1)\cdot\Sigma_{12}=-2.}}\end{array}
$$ 
The third equation shows that, although $X_{3}$ does not depend directly on $X_{1}$ , they have a nonzero covariance. Intuitively, this is clear: $X_{3}$ depends on $X_{2}$ , which depends on $X_{1}$ ; hence, we expect $X_{1}$ and $X_{3}$ to be correlated, a fact that is reﬂected in their covariance. As we can see, the covariance between $X_{1}$ and $X_{3}$ is the covariance between $X_{1}$ and $X_{2}$ , weighted by the strength of the dependence of $X_{3}$ on $X_{2}$ . 
> 本例中，可以看到 $X_1, X_3$ 之间的协方差等于 $X_1, X_2$ 之间的协方差乘上 $X_3, X_2$ 之间的依赖性作为权重

In general, putting these results together, we can see that the mean and covariance matrix for $p(X_{1},X_{2},X_{3})$ is precisely our covariance matrix of example 7.1. 

The converse to this theorem also holds: the result of conditioning is a normal distribution where there is a linear dependency on the conditioning variables. The expressions for converting a multivariate Gaussian to a linear Gaussian network appear complex, but they are based on simple algebra. They can be derived by taking the linear equations speciﬁed in theorem 7.3, and reformulating them as deﬁning the parameters $\beta_{i}$ in terms of the means and covariance matrix entries. 
> 该定理的逆命题同样成立：联合高斯分布下的条件分布也是高斯分布，并且是线性高斯模型
> 将多元高斯分布转换为线性高斯网络的公式看起来很复杂，但实际上它们基于简单的代数运算。这些公式可以通过定理7.3中指定的线性方程推导出来，并重新表述为通过均值和协方差矩阵元素来定义参数$\beta_{i}$。 

**Theorem 7.4** 
Let $\{X,Y\}$ have a joint normal distribution deﬁned in equation (7.3). Then the conditional density 

$$
p({Y}\mid\boldsymbol{X})=\mathcal{N}\left(\beta_{0}+\pmb \beta^{T}\boldsymbol{X};\sigma^{2}\right),
$$

is such that: 

$$
\begin{array}{r c l}{{\beta_{0}}}&{{=}}&{{\mu_{Y}-\Sigma_{Y \pmb X}\Sigma_{\pmb X \pmb X}^{-1}\mu_{\pmb X}}}\\ {{\pmb \beta}}&{{=}}&{{\Sigma_{\pmb X \pmb X}^{-1}\Sigma_{Y \pmb X}}}\\ {{\sigma^{2}}}&{{=}}&{{\Sigma_{Y Y}-\Sigma_{Y \pmb X}\Sigma_{\pmb X \pmb X}^{-1}\Sigma_{\pmb X Y}.}}\end{array}
$$ 
> 定理：
> $\{\pmb X, Y\}$ 服从式 7.3 定义的联合正态分布，则 $Y$ 条件于 $\pmb X$ 的条件分布也是正态分布，并且是线性高斯模型

This result allows us to take a joint Gaussian distribution and produce a Bayesian network, using an identical process to our construction of a minimal I-map in section 3.4.1. 
> 根据该定理，我们可以根据联合高斯分布写出分布中的各个条件概率分布，因此可以根据给定的独立性构建分布的极小 I-map

**Theorem 7.5** 
Let $\mathcal{X}=\{X_{1},\ldots,X_{n}\}$ , and let $p$ be a joint Gaussian distributio over $\mathcal{X}$ . Given any orderi $X_{1},\dots,X_{n}$ over X , we can construct a Bayesian network graph G and a Bayesian network B over G such that: 

1. $\mathrm{Pa}_{X_{i}}^{\mathcal{G}}\subseteq\{X_{1},\ldots,X_{i-1}\};$ ;
2. the CPD of $X_{i}$ in $\mathcal{B}$ is a linear Gaussian of its parents;
3. $\mathcal{G}$ is a minimal $I^{,}$ -map for $p$ . 

> 定理：
> $p$ 为 $\mathcal X = \{X_1, \dots, X_n\}$ 上的联合高斯分布，给定 $\mathcal X$ 上的任意顺序 $X_1, \dots, X_n$，我们可以构建贝叶斯网络图 $\mathcal G$ 和 $\mathcal G$ 上的贝叶斯网络 $\mathcal B$，满足：图中任意变量的父变量都在 $\mathcal X$ 中，$X_i$ 在 $\mathcal B$ 中的 CPD 是关于它的父变量的线性高斯模型，$\mathcal G$ 是 $p$ 的 minimal I-map

The proof is left as an exercise (exercise 7.4). 

As for the case of discrete networks, the minimal I-map is not unique: diferent choices of orderings over the variables will lead to diferent network structures. For example, the distribution in ﬁgure 7.1b can be represented either as the network where $X\rightarrow Y$ or as the network where $Y\rightarrow X$ . 
>至于离散网络的情况，最小的 I-map 不是唯一的：变量的不同排序将导致不同的网络结构。例如，图7.1b 中的分布可以用 $X\rightarrow Y$ 的网络或 $Y\rightarrow X$ 的网络来表示。

This equivalence between Gaussian distributions and linear Gaussian networks has important practical ramiﬁcations. On one hand, we can conclude that, for linear Gaussian networks, the joint distribution has a compact representation (one that is quadratic in the number of variables). Furthermore, the transformations from the network to the joint and back have a fairly simple and efciently computable closed form. Thus, we can easily convert one representation to another, using whichever is more convenient for the current task. Conversely, while the two representations are equivalent in their expressive power, there is not a one-to-one correspondence between their parameter iz at ions. In particular, although in the worst case, the linear Gaussian representation and the Gaussian representation have the same number of parameters (exercise 7.6), there are cases where one representation can be signiﬁcantly more compact than the other. 
>高斯分布与线性高斯网络之间的这种等价性具有重要的实际意义。一方面，我们可以得出结论，对于线性高斯网络，联合分布有一个紧凑的表示形式（即在变量数量上是二次的）
>此外，从网络到联合分布及其逆变换都有相对简单且可有效计算的闭式形式。因此，我们可以轻松地在这两种表示形式之间进行转换，使用对当前任务更方便的一种。另一方面，尽管这两种表示在表达能力上是等价的，但它们的参数化之间并没有一对一的对应关系
>特别是，虽然最坏情况下，线性高斯表示和高斯表示具有相同数量的参数（习题7.6），但在某些情况下，一种表示形式可以比另一种显著更紧凑。

Example 7.4 
Consider a linear Gaussian network structured as a chain: 

$$
X_{1}\rightarrow X_{2}\rightarrow\cdot\cdot\cdot\rightarrow X_{n}.
$$

Assuming the network parameterization is not degenerate (that is, the network is a minimal I-map of its distribution), we have that each pair of variables $X_{i},X_{j}$ are correlated. In this case, as shown in theorem 7.1, the covariance matrix would be dense — none of the entries would be zero. Thus, the representation of the covariance matrix would require a quadratic number of parameters. In the information matrix, however, for all $X_{i},X_{j}$ that are not neighbors in the chain, we have that $X_{i}$ and $X_{j}$ are conditionally independent given the rest of the variables in the network; hence, by theorem 7.2, ${{J}_{i,j}}\mathrm{~=~}0$ . Thus, the information matrix has most of the entries being zero; the only nonzero entries are on the tridiagonal (the entries $i,j$ for $j=i-1,i,i+1)$ . 

However, not all structure in a linear Gaussian network is represented in the information matrix. 

Example 7.5 In a v-structure $X\rightarrow Z\leftarrow Y$ , we have that $X$ and $Y$ are marginally independent, but not conditionally independent given Z . Thus, according to theorem 7.2, the $X,Y$ entry in the information matrix would not be 0 . Conversely, because the variables are marginally independent, the $X,Y$ entry in the covariance entry would be zero. 

Complicating the example somewhat, assume that $X$ and $Y$ also have a joint parent $W$ ; that is, the network is structured as a diamond. In this case, $X$ and $Y$ are still not independent given the remaining network variables $Z,W$ , and hence the $X,Y$ entry in the information matrix is nonzero. Conversely, they are also not marginally independent, and thus the $X,Y$ entry in the covariance matrix is also nonzero. 

These examples simply recapitulate, in the context of Gaussian networks, the fundamental diference in expressive power between Bayesian networks and Markov networks. 

## 7.3 Gaussian Markov Random Fields 
We now turn to the representation of multivariate Gaussian distributions via an undirected graphical model.  
> 本节讨论使用无向图模型表示多元高斯分布

We ﬁrst show how a Gaussian distribution can be viewed as an MRF. This formulation is derived almost immediately from the information form of the Gaussian. Consider again equation (7.2). We can break up the expression in the exponent into two types of terms: those that involve single variables $X_{i}$ and those that involve pairs of variables $X_{i},X_{j}$ . 
> 高斯分布可以被视作一个 MRF/CRF
> 考虑 eq7.2 ，我们指数中的表达式分为两项，一项仅和单个变量 $X_i$ 相关，另一项和一对变量 $X_i, X_j$ 相关

The terms that involve only the variable $X_{i}$ are:

$$
-\frac{1}{2}J_{i,i}x_{i}^{2}+h_{i}x_{i},
$$ 
where we recall that the potential vector $\pmb h=J\pmb \mu$ . 
> 仅和 $X_i$ 相关的项如上，其中 $\pmb h = J \pmb \mu$ 称为势能向量

The terms that involve the pair $X_{i},X_{j}$ are: 

$$
-\frac{1}{2}[J_{i,j}x_{i}x_{j}+J_{j,i}x_{j}x_{i}]=-J_{i,j}x_{i}x_{j},
$$ 
due to the symmetry of the information matrix. 
> 和 $X_i, X_j$ 相关的项如上

Thus, the information form immediately induces a pairwise Markov network, whose node potentials are derived from the potential vector and the diagonal elements of the information matrix, and whose edge potentials are derived from the of-diagonal entries of the information matrix. We also note that, when ${{J}_{i,j}}\mathrm{~=~}0$ , there is no edge between $X_{i}$ and $X_{j}$ in the model, corresponding directly to the independence assumption of the Markov network. 
> 因此，Gaussian 分布的信息形式直接导出了一个成对 Markov 网络，其节点势能来自于势能向量和信息矩阵的对角线元素，其边势能来自于信息矩阵的非对角线元素
> 当 $J_{i, j} = 0$，模型中 $X_i, X_j$ 之间没有边，直接对应于 Markov 网络的独立性假设

Thus, any Gaussian distribution can be represented as a pairwise Markov network with quadratic node and edge potentials. This Markov network is generally called a Gaussian Markov random ﬁeld (GMRF) .
> 因此，任意高斯分布可以被表示为带有二次节点和边势能的成对 Markov 网络，该 Markov 网络一般称为高斯 Markov 随机场

Conversely, consider any pairwise Markov network with quadratic node and edge potentials. Ignoring constant factors, which can be assimilated into the partition function, we can write the node and edge energy functions (log-potentials) as: 

$$
\begin{array}{r l}&{\ \ \ \epsilon_{i}(x_{i})=d_{0}^{i}+d_{1}^{i}x_{i}+d_{2}^{i}x_{i}^{2}}\\ &{\epsilon_{i,j}(x_{i},x_{j})=a_{00}^{i,j}+a_{01}^{i,j}x_{i}+a_{10}^{i,j}x_{j}+a_{11}^{i,j}x_{i}x_{j}+a_{02}^{i,j}x_{i}^{2}+a_{20}^{i,j}x_{j}^{2},}\end{array}\tag{7.6}
$$ 
where we used the log-linear notation of section 4.4.1.2. 
> 反过来说，考虑任意带有二次节点和边势能的成对 Markov 网络，忽略常数因子，我们可以将节点和边的能量函数 (势能函数的对数形式) 写为式 7.6 的形式

By aggregating like terms, we can reformulate any such set of potentials in the log-quadratic form: 

$$
p^{\prime}(\pmb{x})=\exp(-\frac{1}{2}\pmb{x}^{T}J\pmb{x}+\pmb h^{T}\pmb{x}),\tag{7.7}
$$

where we can assume without loss of generality that $J$ is symmetric. 
> 将能量函数中的常数消去 (这些常数本身也会被划分函数吸收)，再将系数统合，我们可以将该成对 Markov 网络的定义的联合分布写为式 7.7 的形式，其中 $J$ 是一个对称矩阵

This Markov network deﬁnes a valid Gaussian density if and only if $J$ is a positive deﬁnite matrix. If so, then $J$ is a legal information matrix, and we can take $\pmb h$ to be a potential vector, resulting in a distribution in the form of equation (7.2). 
> 因此，当且仅当 $J$ 是正定矩阵时，该 Markov 网络定义了一个有效的高斯分布，此时 $J$ 就是一个合法的信息矩阵，$\pmb h$ 就作为势能向量

However, unlike the case of Gaussian Bayesian networks, it is not the case that every set of quadratic node and edge potentials induces a legal Gaussian distribution. Indeed, the decom- position of equation (7.4) and equation (7.5) can be performed for any quadratic form, including one not corresponding to a positive deﬁnite matrix. For such matrices, the resulting function $\exp({\pmb x}^{T}A{\pmb x}+\bar{\pmb b}^{T}{\pmb x})$ will have an inﬁnite integral, and cannot be normalized to produce a valid density. 
> 虽然任意带有二次节点和边势能的 Markov 网络都可以写为式 7.7 的形式，但当 $J$ 不是正定矩阵时，$\exp (\pmb x^T J \pmb x + \pmb h^T \pmb x)$ 的积分将趋于无穷，因此不能被规范化以产生有效的概率分布

Unfortunately, **other than generating the entire information matrix and testing whether it is positive deﬁnite, there is no simple way to check whether the MRF is valid. In particular, there is no local test that can be applied to the network parameters that precisely characterizes valid Gaussian densities.** However, there are simple tests that are sufficient to induce a valid density. While these conditions are not necessary, they appear to cover many of the cases that occur in practice. 
>除了生成整个信息矩阵并测试其是否为正定矩阵外，没有简单的方法可以检查MRF是否有效，特别地，没有局部测试可以应用于网络参数以精确表征这些参数是否定义有效的高斯密度
>然而，存在一些简单的测试条件，这些条件对于引出一个有效的密度是充分的，虽然这些条件不是必要的，但它们涵盖了实际中出现的许多情况

We ﬁrst provide one very simple test that can be veriﬁed by direct examination of the information matrix. 

**Deﬁnition 7.2** diagonally dominant 
A quadratic MRF parameterized by J is said to be diagonally dominant if, for all i
> 定义：
> 一个由 $J$ 参数化的 MRF 如果满足对于所有 $i$ 都有 $\sum_{j\ne i} |J_{i, j}| < J_{i, i}$，则称 $J$ 是对角主导的

$$
\sum_{j\neq i}|J_{i,j}|<J_{i,i}.
$$ 
For example, the information matrix in example 7.2 is diagonally dominant; for instance, for $i=2$ we have: 

$$
|-0.125|+0.3333<0.5833.
$$ 
One can now show the following result: 

**Proposition 7.1** 
Let $\begin{array}{r}{p^{\prime}(x)=\exp(-\frac{1}{2}x^{T}J x+\overline{{h}}^{T}x)}\end{array}$ be a quadratic pairwise MRF. If $J$ is diagonally dominant, then $p^{\prime}$ deﬁnes a valid Gaussian MRF. 
> 命题：
> $p' (x) = \exp (-\frac 1 2 \pmb x^T J \pmb x + \pmb h^T \pmb x)$ 为二次成对 Markov 随机场，如果 $J$ 是对角主导的，则 $p'$ 定义了一个有效的高斯 Markov 随机场

The proof is straightforward algebra and is left as an exercise (exercise 7.8). 

The following condition is less easily veriﬁed, since it cannot be tested by simple examination of the information matrix. Rather, it checks whether the distribution can be written as a quadratic pairwise MRF whose node and edge potentials satisfy certain conditions. Speciﬁcally, recall that a Gaussian MRF consists of a set of node potentials, which are log-quadratic forms in $x_{i}$ , and a set of edge potentials, which are log-quadratic forms in $x_{i},x_{j}$ . We can state a condition in terms of the coeffcients for the nonlinear components of this parameterization: 

**Deﬁnition 7.3** pairwise normalizable 
A quadratic MRF parameterized as in equation (7.6) is said to be pairwise normalizable if: 

- for all i , $d_{2}^{i}>0$ ; 
- for all $i,j$ , the $2\times2$ matrix $\left(\begin{array}{c c}{{a_{02}^{i,j}}}&{{a_{11}^{i,j}/2}}\\ {{a_{11}^{i,j}/2}}&{{a_{20}^{i,j}}}\end{array}\right)$ is positive semideﬁnite. 

> 定义：
> 按照 eq 7.6 参数化的二次 Markov 随机场如果满足
> - $d_{2}^{i}>0$ 对于所有的 $i$ 成立，也就是节点势能中二次项的系数是正数
>- 对于所有的 $i, j$，矩阵 $\left(\begin{array}{c c}{{a_{02}^{i,j}}}&{{a_{11}^{i,j}/2}}\\ {{a_{11}^{i,j}/2}}&{{a_{20}^{i,j}}}\end{array}\right)$ 是半正定的
> 则称该 MRF 是成对可规范化的

Intuitively, this deﬁnition states that each edge potential, considered in isolation, is normalizable (hence the name “pairwise-normalizable”). 
> 该定义即声明了每个边势能在单独考虑时都是可规范化的

We can show the following result: 

**Proposition 7.2** 
Let $p^{\prime}(x)$ be a quadratic pairwise MRF, parameterized as in equation (7.6). If $p^{\prime}$ is pairwise normalizable, then it deﬁnes a valid Gaussian distribution. 
> 命题：
> $p' (x)$ 为二次成对 MRF，按照 eq 7.6 参数化，如果 $p'$ 是成对可规范化的，则它定义了有效的高斯分布

Once again, the proof follows from standard algebraic manipulations, and is left as an exercise (exercise 7.9). 

We note that, like the preceding conditions, this condition is sufficient but not necessary: 
> 上述两个条件都是充分条件而不是必要条件

Example 7.6 Consider the following information matrix: 

$$
\left(\begin{array}{l l l}{1}&{0.6}&{0.6}\\ {0.6}&{1}&{0.6}\\ {0.6}&{0.6}&{1}\end{array}\right)
$$ 
It is not difcult to show that this information matrix is positive deﬁnite, and hence deﬁnes a legal Gaussian distribution. However, it turns out that it is not possible to decompose this matrix into a set of three edge potentials, each of which is positive deﬁnite. 

Unfortunately, evaluating whether pairwise normalizability holds for a given MRF is not always trivial, since it can be the case that one parameterization is not pairwise normalizable, yet a diferent parameterization that induces precisely the same density function is pairwise normalizable. 

Example 7.7
Consider the information matrix of example 7.2, with a mean vector 0 . We can deﬁne this distribution using an MRF by simply choosing the node potential for $X_{i}$ to be $J_{i,i}x_{i}^{2}$ and the edge potential for $X_{i},X_{j}$ to be $2J_{i,j}x_{i}x_{j}$ . Clearly, the $X_{1},X_{2}$ edge does not deﬁne a nor- malizable density over $X_{1},X_{2}$ , and hence this MRF is not pairwise normalizable. However, as we discussed in the context of discrete MRFs, the MRF parameter iz ation is nonunique, and the same density can be induced using a continuum of diferent parameter iz at ions. In this case, one alternative parameter iz ation of the same density is to deﬁne all node potentials as $\epsilon_{i}(x_{i})=0.05x_{i}^{2}$ , and the edge potentials to be $\epsilon_{1,2}(x_{1},x_{2})=0.2625x_{1}^{2}+0.0033x_{2}^{2}-0.25x_{1}x_{2}.$ − , and $\epsilon_{2,3}(x_{2},x_{3})=0.53x_{2}^{2}+0.2833x_{3}^{2}+0.6666x_{2}x_{3}$ . Straightforward arithmetic shows that this set of potentials induces the information matrix of example 7.2. Moreover, we can show that this formulation is pairwise normalizable: The three node potentials are all positive, and the two edge potentials are both positive deﬁnite. (This latter fact can be shown either directly or as a conse- quence of the fact that each of the edge potentials is diagonally dominant, and hence also positive deﬁnite.) 

This example illustrates that the pairwise normalizability condition is easily checked for a speciﬁc MRF parameter iz ation. However, if our aim is to encode a particular Gaussian density as an MRF, we may have to actively search for a decomposition that satisﬁes the relevant constraints. If the information matrix is small enough to manipulate directly, this process is not difcult, but if the information matrix is large, ﬁnding an appropriate parameter ization may incur a nontrivial computational cost. 
>这个例子说明，对于特定的MRF参数化，成对归一化条件很容易进行检查。然而，如果我们希望将特定的高斯密度编码为MRF，则可能需要积极寻找满足相关约束的分解。如果信息矩阵足够小，可以直接操作，这个过程并不困难，但如果信息矩阵很大，找到适当的参数化可能会带来非 trivial 的计算成本。

## 7.4 Summary 
This chapter focused on the representation and independence properties of Gaussian networks. 
> 本章聚焦于高斯网络的表示和独立性质

We showed an equivalence of expressive power between three representational classes: multivariate Gaussians, linear Gaussian Bayesian networks, and Gaussian MRFs. In particular, any distribution that can be represented in one of those forms can also be represented in another. We provided closed-form formulas that allow us convert between the multivariate Gaussian representation and the linear Gaussian Bayesian network. The conversion for Markov networks is simpler in some sense, inasmuch as there is a direct mapping between the entries in the infor- mation (inverse covariance) matrix of the Gaussian and the quadratic forms that parameterize the edge potentials in the Markov network. 
> 本章介绍了三种表示类型的等价性：多元高斯分布、线性高斯贝叶斯网络、高斯 Markov 随机场，任意可以以其中一种形式表示的分布都可以用另一种形式表示
> 我们提供了在多元高斯表示和线性高斯贝叶斯网络之间转化的闭式公式，高斯分布的信息矩阵中的 entries 和 Markov 网络中参数化边势能的二次形式之间则存在直接映射

However, unlike the case of Bayesian networks, here we must take care, since not every quadratic parameterization of a pairwise Markov network induces a legal Gaussian distribution: The quadratic form that arises when we combine all the pairwise potentials may not have a ﬁnite integral, and therefore may not be normalizable. In general, there is no local way of determining whether a pairwise MRF with quadratic potentials is normalizable; however, we provided some easily checkable sufcient conditions that are often sufcient in practice. 
>然而，与贝叶斯网络不同的是，我们必须小心并不是每个二元马尔可夫随机场的二次参数化都能诱导出合法的高斯分布：我们结合所有成对势能时得到的二次形式可能没有有限的积分，因此可能无法归一化
>一般来说，没有局部方法可以确定一个具有二次势能的二元马尔可夫随机场是否可归一化；然而，我们提供了一些易于检查的充分条件，这些条件在实践中通常是足够的

The equivalence between the diferent representations is analogous to the equivalence of Bayesian networks, Markov networks, and discrete distributions: any discrete distribution can be encoded both as a Bayesian network and as a Markov network, and vice versa. However, as in the discrete case, this equivalence does not imply equivalence of expressive power with respect to independence assumptions. **In particular, the expressive power of the directed and undirected representations in terms of independence assumptions is exactly the same as in the discrete case: Directed models can encode the independencies associated with immoralities, whereas undirected models cannot; conversely, undirected models can encode a symmetric diamond, whereas directed models cannot.**
> 高斯分布的不同表示之间的等价性类似于贝叶斯网络、Markov 网络、离散分布之间的等价性：任意离散分布都可以被编码为贝叶斯网络和 Markov 网络，反之也成立
> 无论是离散情况还是连续情况，有向图和无向图之间关于独立性假设的表示能力都存在差异，有向模型可以编码和 immorality 相关的独立性，而无向模型不行，无向模型可以编码和对称菱形相关的独立性，而有向模型不行

 As we saw, the undirected models have a particularly elegant connection to the natural representation of the Gaussian distribution in terms of the information matrix; in particular, zeros in the information matrix for $p$ correspond precisely to missing edges in the minimal I-map Markov network for $p$ . 
> 无向模型和高斯分布的信息矩阵表示之间存在直接的联系：$p$ 的信息矩阵中为零的项直接对应于 $p$ 的极小 I-map 的 Markov 网络中缺少边

Finally, we note that the class of Gaussian distributions is highly restrictive, making strong assumptions that often do not hold in practice. Nevertheless, it is a very useful class, due to its compact representation and computational tractability (see section 14.2). Thus, in many cases, we may be willing to make the assumption that a distribution is Gaussian even when that is only a rough approximation. This approximation may happen a priori, in encoding a distribution as a Gaussian even when it is not. Or, in many cases, we perform the approximation as part of our inference process, representing intermediate results as a Gaussian, in order to keep the computation tractable. Indeed, as we will see, the Gaussian representation is ubiquitous in methods that perform inference in a broad range of continuous models. 
>最后，我们注意到高斯分布类是非常严格的，它常常做出在实际中不成立的强假设。尽管如此，由于其紧凑的表示和计算上的可处理性（见第14.2节），它是一个非常有用的类。因此，在许多情况下，即使这种假设只是一个粗略的近似，我们也可能愿意假设一个分布是高斯的。
>这种近似可以在先验知识中发生，即使在编码分布时它实际上并不是高斯分布 （也就是先验服从高斯，但实际分布不服从高斯）。或者，在许多情况下，我们在推理过程中进行近似，将中间结果表示为高斯分布，以便保持计算的可处理性。
>事实上，正如我们将看到的，高斯表示在广泛连续模型的推理方法中无处不在。

# Part 2 Inference
# 9  Exact Inference: Variable Elimination 
In this chapter, we discuss the problem of performing inference in graphical models. We show that the structure of the network, both the conditional independence assertions it makes and the associated factorization of the joint distribution, is critical to our ability to perform inference efectively, allowing tractable inference even in complex networks. 

conditional probability query 

Our focus in this chapter is on the most common query type: the conditional probability query , $P(Y\mid E=e)$ (see section 2.1.5). We have already seen several examples of conditional probability queries in chapter 3 and chapter 4; as we saw, such queries allow for many useful reasoning patterns, including explanation, prediction, intercausal reasoning, and many more. By the deﬁnition of conditional probability, we know that 

$$
P(Y\mid E=e)=\frac{P(Y,e)}{P(e)}.
$$ 

Each of the instantiations of the numerator is a probability expression $P(\pmb{y},e)$ , which can be computed by summing out all entries in the joint that correspond to assignments consistent with $\mathbfit{\Delta}_{\mathcal{Y},\mathbf{\Delta}e}$ . More precisely, let $W=\mathcal{X}-Y-E$ be the random variables that are neither query nor evidence. Then 

$$
P(\pmb{y},e)=\sum_{\pmb{w}}P(\pmb{y},e,\pmb{w}).
$$ 

Because $Y,E,W$ are all of the network variables, each term $P(\pmb{y},\pmb{e},\pmb{w})$ in the summation is simply an entry in the joint distribution. 

The probability $P(e)$ can also be computed directly by summing out the joint. However, it can also be computed as 

$$
P(e)=\sum_{y}P(y,e),
$$ 

renormalization which allows us to reuse our computation for equation (9.2). If we compute both equation (9.2) and equation (9.3), we can then divide each $P(\pmb{y},e)$ by $P(e)$ , to get the desired conditional probability $P(\pmb{y}\mid\pmb{e})$ Note that this process corresponds to taking the vector of marginal probabilities $P(\pmb{y}^{1},\pmb{e}),\dots,P(\pmb{y}^{k},\pmb{e})$ (where $k\,=\,|\mathit{V a l}(Y)|)$ and renormalizing the entries to sum to 1 . 

# 9.1 Analysis of Complexity 

In principle, a graphical model can be used to answer all of the query types described earlier. We simply generate the joint distribution and exhaustively sum out the joint (in the case of a conditional probability query), search for the most likely entry (in the case of a MAP query), or both (in the case of a marginal MAP query). However, this approach to the inference problem is not very satisfactory, since it returns us to the exponential blowup of the joint distribution that the graphical model representation was precisely designed to avoid. 

Unfortunately, we now show that exponential blowup of the inference task is (almost  certainly) unavoidable in the worst case: The problem of inference in graphical models is $\mathcal{N P}$ -hard, and therefore ly requires exponential time in the worst ca except in the unlikely event that P $\mathcal{P}=\mathcal{N P}$ NP ). Even worse, approximate inference is also NP -hard. Importantly, however, the story does not end with this negative result. In general, we care not about the worst case, but about the cases that we encounter in practice. As we show in the remainder of this part of the book, many real-world applications can be tackled very efectively using exact or approximate inference algorithms for graphical models. 

In our theoretical analysis, we focus our discussion on Bayesian networks. Because any Bayesian network can be encoded as a Markov network with no increase in its representation size, a hardness proof for inference in Bayesian networks immediately implies hardness of inference in Markov networks. 

# 9.1.1 Analysis of Exact Inference 

To address the question of the complexity of BN inference, we need to address the question of how we encode a Bayesian network. Without going into too much detail, we can assume that the encoding speciﬁes the DAG structure and the CPDs. For the following results, we assume the worst-case representation of a CPD as a full table of size $|V a l(\{X_{i}\}\cup\mathrm{Pa}_{X_{i}})|$ . 

As we discuss in appendix A.3.4, most analyses of complexity are stated in terms of decision problems. We therefore begin with a formulation of the inference problem as a decision prob- lem, and then discuss the numerical version. One natural decision version of the conditional probability task is the problem $B N–P r–D P,$ , deﬁned as follows: 

Given a $\mathcal{B}$ over $\mathcal{X}$ , a variable $X\in{\mathcal{X}}$ , and a value $x\in V a l(X)$ , decide whether $P_{\mathcal{B}}(X=x)>0$ . 

# Theorem 9.1 

Proof It is straightforward to prove that $B N–P r–D P$ is in $\mathcal{N P}$ : In the guessing phase, we full assignment $\xi$ to the network variables. In the veriﬁcation phase, we check whether X $X=x$ in $\xi;$ , and whether $P(\xi)\,>\,0$ . One of these guesses succeeds if and only if $P(X\,=\,x)\,>\,0$ . Computing $P(\xi)$ for a full assignment of the network variables requires only that we multiply the relevant entries in the factors, as per the chain rule for Bayesian networks, and hence can be done in linear time. 

To prove $\mathcal{N P}$ -hardness, we need to show that, if we can answer instances in BN-Pr-DP , we can use that as a subroutine to answer questions in a class of problems that is known to be $\mathcal{N P}$ -hard. We will use a reduction from the 3-SAT problem deﬁned in deﬁnition A.8. 

![](images/b75ddde8f0c0af7656fa1e96b577b56a7d34cae1b516a18d67fb0541decb04cc.jpg) 
Figure 9.1 An outline of the network structure used in the reduction of 3-SAT to Bayesian network inference. 

To show the reduction, we show the following: Given any 3-SAT formula $\phi$ , we can create a Bayesian network $\mathcal{B}_{\phi}$ with some distinguished variable $X$ , such that $\phi$ is satisﬁable if and only if $P_{\mathcal{B}_{\phi}}(X=x^{1})>0$ . Thus, if we can solve the Bayesian network inference problem in polynomial time, we can also solve the 3-SAT problem in polynomial time. To enable this conclusion, our BN $\mathcal{B}_{\phi}$ has to be constructible in time that is polynomial in the length of the formula $\phi$ . 

Consider a 3-SAT instance $\phi$ over the propositional variables $q_{1},\ldots,q_{n}$ . Figure 9.1 illustrates the structure of the network constr ted in this reduction. Our Bayesian network $\mathcal{B}_{\phi}$ has a node $Q_{k}$ for each propositional variable $q_{k}$ ; these variables are roots, with $P(q_{k}^{1})=0.5$ . It also has a no $C_{i}$ for each cl e $C_{i}$ . There is an edge from $Q_{k}$ to $C_{i}$ if $q_{k}$ or $\neg q_{k}$ is one of the literals in $C_{i}$ . The CPD for $C_{i}$ is deterministic, and chosen such that it exactly duplicates the behavior of the clause. Note that, because $C_{i}$ contains at most three variables, the CPD has at most eight distributions, and at most sixteen entries. 

We want to introduce a variable $X$ that has the value 1 if and only if all the $C_{i}$ ’s have the value 1 . We can achieve this requirement by having $C_{1},\ldots,C_{m}$ be parents of $X$ . This construction, however, has the property that $P(X\mid C_{1},.\,.\,,C_{m})$ is exponentially large when written as a table. To avoid this difculty, we introduce intermediate “AND” gates $A_{1},\dots,A_{m-2}$ , so that $A_{1}$ is the “AND” of $C_{1}$ and $C_{2}$ , $A_{2}$ is the “AND” of $A_{1}$ and $C_{3}$ , and so on. The last variable $X$ is the “AND” of $A_{m-2}$ and $C_{m}$ . This construction achieves the desired efect: $X$ has value 1 if and only if all the clauses are satisﬁed. Furthermore, in this construction, all variables have at most three (binary-valued) parents, so that the size of $\mathcal{B}_{\phi}$ is polynomial in the size of $\phi$ . It follows that $P_{\mathcal{B}_{\phi}}(x^{1}\mid q_{1},.\,.\,.\,,q_{n})=1$ if and only if $q_{1},\ldots,q_{n}$ is a satisfying assignment for $\phi$ . Because the prior probability of each possible assignment is $1/2^{n}$ , we get that the overall probability $P_{\mathcal{B}_{\phi}}(x^{1})$ is the number of satisfying assignments to $\phi$ , divided by $2^{n}$ . We can therefore test whether $\phi$ has a satisfying assignment simply by checking whether $P(x^{1})>0$ . 

This analysis shows that the decision problem associated with Bayesian network inference is $\mathcal{N P}$ -complete. However, the problem is originally a numerical problem. Precisely the same construction allows us to provide an analysis for the original problem formulation. We deﬁne the problem $B N!P r$ as follows: 

Given: a Bayesian network $\mathcal{B}$ over $\mathcal{X}$ , a variable $X\,\in\,{\mathcal{X}}$ , and a value $x\;\in\;V a l(X)$ , compute $P_{\mathcal{B}}(X=x)$ . 

Our task here is to compute the total probability of network instantiations that are consistent with $X=x$ . Or, in other words, to do a weighted count of instantiations, with the weight being the probability. An appropriate complexity class for counting problems is $\#\mathcal{P}$ : Whereas $\mathcal{N P}$ represents problems of deciding “are there any solutions that satisfy certain requirements,” $\#\mathcal{P}$ P represents problems that ask “how many solutions are there that satisfy certain requirements.” It is not surprising that we can relate the complexity of the BN inference problem to the counting class $\#\mathcal{P}$ : 

The problem BN-Pr is $\#\mathcal{P}$ -complete. We leave the proof as an exercise (exercise 9.1). 

# 9.1.2 Analysis of Approximate Inference 

Upon noting the hardness of exact inference, a natural question is whether we can circumvent the difculties by compromising, to some extent, on the accuracies of our answers. Indeed, in many applications we can tolerate some imprecision in the ﬁnal probabilities: it is often unlikely that a change in probability from 0 . 87 to 0 . 92 will change our course of action. Thus, we now explore the computational complexity of approximate inference. 

To analyze the approximate inference task formally, we must ﬁrst deﬁne a metric for evaluating the quality of our approximation. We can consider two perspectives on this issue, depending on how we choose to deﬁne our query. Consider ﬁrst our previous formulation of the conditional probabilit query task, wh e our goal is to compute the probability $P(Y\mid e)$ for some set of variables Y $Y$ and evidence e . The result of this type of query is a probability distribution over $Y$ . Given an approximate answer to this query, we can evaluate its quality using any of the distance metrics we deﬁne for probability distributions in appendix A.1.3.3. 

There is, however, another way of looking at this task, one that is somewhat simpler and will be very useful for analyzing its complexity. Consider a speciﬁc query $P(\pmb{y}\mid\pmb{e})$ , where we are focusing on one particular assignment $_{_y}$ . The approximate answer to this query is a number $\rho$ , whose accuracy we wish to evaluate relative to the correct probability. One way of evaluating the accuracy of an estimate is as simple as the diference between the approximate answer and the right one. 

$$
|P(\pmb{y}\mid e)-\rho|\leq\epsilon.
$$ 

This deﬁnition, although plausible, is somewhat weak. Consider, for example, a situation in which we are trying to compute the probability of a really rare disease, one whose true probability is, say, 0 . 00001 . In this case, an absolute error of 0 . 0001 is unacceptable, even though such an error may be an excellent approximation for an event whose probability is 0 . 3 . A stronger deﬁnition of accuracy takes into consideration the value of the probability that we are trying to estimate: 

Deﬁnition 9.2 relative error 

An estimate $\rho$ has relative error $\epsilon$ for $P(\pmb{y}\mid\pmb{e})$ if: 

$$
\frac{\rho}{1+\epsilon}\le P(\pmb{y}\mid e)\le\rho(1+\epsilon).
$$ 

Note that, unlike absolute error, relative error makes sense even for $\epsilon>1$ . For example, $\epsilon=4$ means that $P(\pmb{y}\mid e)$ is at least 20 percent of $\rho$ and at most 600 percent of $\rho$ . For probabilities, where low values are often very important, relative error appears much more relevant than absolute error. 

With these deﬁnitions, we can turn to answering the question of whether approximate in- ference is actually an easier problem. A priori, it seems as if the extra slack provided by the approximation might help. Unfortunately, this hope turns out to be unfounded. As we now show, approximate inference in Bayesian networks is also $\mathcal{N P}$ -hard. 

This result is straightforward for the case of relative error. 

Theorem 9.3 The following problem is -hard: 

Given a Bayesian network $\mathcal{B}$ ove $\mathcal{X}$ , a variable $X\in{\mathcal{X}}$ , and a value $x\in V a l(X)$ , ﬁnd a number $\rho$ that has relative error ϵ for $P_{\mathcal{B}}(X=x)$ . 

Proof The proof is obvious based on the original $\mathcal{N P}$ k inference (theorem 9.1). There, we proved that it is NP -hard to decide whethe $P_{\mathcal{B}}(x^{1})>0$ . Now, assume that we have an algorithm that returns an estimate $\rho$ to the same $P_{\mathcal{B}}(x^{1})$ , which B is guaranteed to have relative error $\epsilon$ for some $\epsilon>0$ . Then $\rho>0$ if and only if $P_{\mathcal{B}}(x^{1})>0$ . Thus, achieving this relative error is as $\mathcal{N P}$ -hard as the original problem. 

We can generalize this result to make $\epsilon(n)$ a function that grows with the input size $n$ . Thus, for example, we can deﬁne $\epsilon(n)=2^{2^{n}}$ and the theorem still holds. Thus, in a sense, this result is not so interesting as a statement about hardness of approximation. Rather, it tells us that relative error is too strong a notion of approximation to use in this context. 

What about absolute error? As we will see in section 12.1.2, the problem of just approximating $P(X\,=\,x)$ up to some ﬁxed absolute error $\epsilon$ has a randomized polynomial time algorithm. Therefore, the problem cannot be $\mathcal{N P}$ -hard unless $\mathcal{N P}=\mathcal{R P}$ . T sult is an improvement on the exact case, where even the task of computing $P(X=x)$ is -hard. 

Unfortunately, the good news is very limited in scope, in that it disappears once we introduce e. Speciﬁcally, it is $\mathcal{N P}$ -hard to ﬁnd an absolute approximation to $P(x\mid e)$ for any $\epsilon<1/2$ . 

Theorem 9.4 

Given a Ba netw $\mathcal{B}$ over $\mathcal{X}$ e $X\,\in\,{\mathcal{X}}$ , a value $x\,\in\,V a l(X)$ , and a observation $E=e$ for $E\subset{\mathcal{X}}$ ⊂X and $e\in V a l(E)$ ∈ , ﬁnd a number $\rho$ that has absolute error ϵ for $P_{\mathcal{B}}(X=x\mid e)$ . 

Proof The proof uses the same construction that we used before. Consider a formula $\phi$ , and nsider the analogous BN $\mathcal{B}$ , as described in theorem 9.1. Recall that our BN had a variable $Q_{i}$ for each propositional variable $q_{i}$ in our Boolean formula, a bunch of other intermediate variables, and then a variable $X$ whose value, given any assignment of values $q_{1}^{1},q_{1}^{0}$ to the $Q_{i}$ ’s, was the associated truth value of the formula. We now show that, given such an approximation algorithm, we can cide ether the formula is satis ble. We begin by computing $P(Q_{1}\mid x^{1})$ . We pick the value $v_{1}$ for $Q_{1}$ that is most likely given $x^{1}$ , and we instantiate it to this value. That , we generate a network $\mathcal{B}_{2}$ that does not contain $Q_{1}$ , and that represents the distribution $\mathcal{B}$ B conditioned on $Q_{1}\,=\,v_{1}$ . We repeat this process for $Q_{2},\ldots,Q_{n}$ . This results in some assignment $v_{1},\dots,v_{n}$ to the $Q_{i}$ ’s. We now prove that this is a satisfying assignment if and only if the original formula $\phi$ was satisﬁable. 

We begin with the easy case. If $\phi$ is not satisﬁable, then $v_{1},\dots,v_{n}$ can hardly be a satisfying assignment for it. Now, assume that $\phi$ is satisﬁable. We show that it also has a satisfying assignment with $Q_{1}\,=\,v_{1}$ . If $\phi$ is satisﬁable with both $Q_{1}\,=\,q_{1}^{1}$ and $Q_{1}\,=\,q_{1}^{0}$ , then this is obvious. Assume, however, that $\phi$ is satisﬁable, but not when $Q_{1}\,=\,v$ . Then necessarily, we will have that $P(Q_{1}=v\mid x^{1})$ is 0, and the probability of the complementary event i 1. If we have an approximation $\rho$ whose error is guaranteed to be $<1/2$ , then choosing the v that maximizes this probability is guaranteed to pick the $v$ whose probability is 1. Thus, in either case the formula has a satisfying assignment where $Q_{1}=v$ . 

We can continue in this fashion, proving by induction on $k$ that $\phi$ has a satisfying assignment with $Q_{1}=v_{1},.\,.\,.\,,Q_{k}=v_{k}$ . In the case where $\phi$ is satisﬁable, this process will terminate with a satisfying assignment. In the case where $\phi$ is not, it clearly will not terminate with a satisfying assignment. We can determine which is the case simply by checking whether the resulting assignment satisﬁes $\phi$ . This gives us a polynomial time process for deciding satisﬁability. 

Because $\epsilon=1/2$ corresponds to random guessing, this result is quite discouraging. It tells us that, in the case where we have evidence, approximate inference is no easier than exact inference, in the worst case. 

# 9.2 Variable Elimination: The Basic Ideas 

We begin our discussion of inference by discussing the principles underlying exact inference in graphical models. As we show, the same graphical structure that allows a compact represen- tation of complex distributions also help support inference. In particular, we can use dynamic programming techniques (as discussed in appendix A.3.3) to perform inference even for certain large and complex networks in a very reasonable time. We now provide the intuition underlying these algorithms, an intuition that is presented more formally in the remainder of this chapter. 

We begin by considering the inference task in a very simple network $A\,\rightarrow\,B\,\rightarrow\,C\,\rightarrow$ $D$ . We ﬁrst provide a phased computation, which uses results from the previous phase for the computation in the next phase. We then reformulate this process in terms of a global computation on the joint distribution. 

Assume that our ﬁrst goal is to compute the probability $P(B)$ , that is, the distribution over values $b$ of $B$ . Basic probabilistic reasoning (with no assumptions) tells us that 

$$
P(B)=\sum_{a}P(a)P(B\mid a).
$$ 

Fortunately, we have all the required numbers in our Bayesian network representation: each number $P(a)$ is in the CPD for $A$ , and each number $P(b\mid a)$ is in the CPD for $B$ . Note that if $A$ has $k$ values and $B$ has $m$ values, the number of basic arithmetic operations required is $O(k\times m)$ : to compute $P(b)$ , we m st multiply $P(b\mid a)$ $P(a)$ for each of the $k$ values of $A$ , and then add them u that is, $k$ multiplications and k $k-1$ − additions; this process must be repeated for each of the m values b . 

Now, assume we want to compute $P(C)$ . Using the same analysis, we have that 

$$
P(C)=\sum_{b}P(b)P(C\mid b).
$$ 

Again, the co itional probabilities $P(c\mid b)$ are known: they constitute the CPD for $C$ . The probability of B is not speciﬁed as part of the network parameters, but equation (9.4) shows us how it can be computed. Thus, we can compute $P(C)$ . We can continue the process in an analogous way, in order to compute $P(D)$ . 

Note that the structure of the network, and its efect on the parameter iz ation of the CPDs, is critical for our ability to perform this computation as described. Speciﬁcally, assume that $A$ had been a parent of $C$ . In this case, the CPD for $C$ would have included $A$ , and our computation of $P(B)$ would not have sufced for equation (9.5). 

Also note that this algorithm does not compute single values, but rather sets of values at a time. In particular equation (9.4) computes an entire distribution over all of the possible values of $B$ . All of these are then used in equation (9.5) to compute $P(C)$ . This property turns out to be critical for the performance of the general algorithm. 

Let us analyze the complexity of this process on a general chain. Assume that we have a chain with $n$ variables $X_{1}\,\rightarrow\,.\,.\,\rightarrow\,X_{n},$ each e in $k$ values. As described, the algorithm would compute $P(X_{i+1})$ from $P(X_{i})$ , for $i=1,\dots,n-1$ − . Each such step would consist of the following computation: 

$$
P(X_{i+1})=\sum_{x_{i}}{P(X_{i+1}\mid x_{i})P(x_{i})},
$$ 

where $P(X_{i})$ is computed in the previous step. The cost of each such step is $O(k^{2})$ : The distributi er $X_{i}$ has $k$ va s, and the CPD $P(X_{i+1}\mid X_{i})$ s $k^{2}$ values; we need to multiply $P(x_{i})$ , for value x , with each CPD entry $P(x_{i+1}\mid x_{i})$ $\,\!\,k^{2}$ multiplications), and then, for each value $x_{i+1}$ , sum up the co entries ( $(k\times(k-1)$ × − additions). We need to perform this process for every variable $X_{2},\ldots,X_{n}$ ; hence, the total cost is $O(n k^{2})$ . 

By comparison, consider the process of generating the entire joint and summing it out, which requires that we generate $k^{n}$ probabilities for the diferent events $x_{1},\dots,x_{n}$ . Hence, we have at least one example where, despite the exponential size of the joint distribution, we can do inference in linear time. 

Using this process, we have managed to do inference over the joint distribution without ever generating it explicitly. What is the basic insight that allows us to avoid the exhaustive enumeration? Let us reexamine this process in terms of the joint $P(A,B,C,D)$ . By the chain rule for Bayesian networks, the joint decomposes as 

$$
P(A)P(B\mid A)P(C\mid B)P(D\mid C)
$$ 

To compute $P(D)$ , we need to sum together all of the entries where $D=d^{1}$ , and to (separately) sum together all of the entries where $D\ =\ d^{2}$ . The exact computation that needs to be 

![](images/52a6137cf3ae5ad55699e70c707c9901c25b5aaae89da9334aabcbd8897f2c1a.jpg) 
Figure 9.2 Computing $P(D)$ by summing over the joint distribution for a chain $A\to B\to C\to$ $D$ ; all of the variables are binary valued. 

performed, for binary-valued variables $A,B,C,D$ , is shown in ﬁgure 9.2. 

Examining this summation, we see that it has a lot of structure. For example, the third and fourth terms in the ﬁrst two entries are both $P(c^{1}\mid b^{1})P(d^{1}\mid c^{1})$ . We can therefore modify the computation to ﬁrst compute 

$$
P(a^{1})P(b^{1}\mid a^{1})+P(a^{2})P(b^{1}\mid a^{2})
$$ 

and only then multiply by the common term. The same structure is repeated throughout the table. If we perform the same transformation, we get a new expression, as shown in ﬁgure 9.3. 

We now observe that certain terms are repeated several times in this expression. Speciﬁcally, $P(a^{1})P(b^{1}\mid a^{1})+P(a^{2})P(b^{1}\mid a^{2})$ and $P(a^{1})P(b^{2}\mid a^{1})+P(a^{2})P(b^{2}\mid a^{2})$ are each repeated four times. Thus, it seems clear that we can gain signiﬁcant computational savings by computing them once and then storing them. There are two such expressions, one for each value of $B$ . Thus, we e a function $\tau_{1}\ :\ \,V a l(B)\mapsto I\!\!R,$ , where $\tau_{1}(b^{1})$ is the ﬁrst of these two expressions, and $\tau_{1}(b^{2})$ is the second. Note that $\tau_{1}(B)$ corresponds exactly to $P(B)$ . 

The resulting expression, assuming $\tau_{1}(B)$ has been computed, is shown in ﬁgure 9.4. Examin- ing this new expression, we see that we once again can reverse the order of a sum and a product, resulting in the expression of ﬁgure 9.5. And, once again, we notice some shared expressions, that are better computed once and used multiple times. We deﬁne $\tau_{2}~:~V a l(C)\mapsto I\!\!R.$ . 

$$
\begin{array}{l l l}{{\tau_{2}(c^{1})}}&{{=}}&{{\tau_{1}(b^{1})P(c^{1}\mid b^{1})+\tau_{1}(b^{2})P(c^{1}\mid b^{2})}}\\ {{\tau_{2}(c^{2})}}&{{=}}&{{\tau_{1}(b^{1})P(c^{2}\mid b^{1})+\tau_{1}(b^{2})P(c^{2}\mid b^{2})}}\end{array}
$$ 

1. When $D$ is binary-valued, we can get away with doing only the ﬁrst of these computations. However, this trick does not carry over to the case of variables with more than two values or to the case where we have evidence. Therefore, our example will show the computation in its generality. 

$$
\begin{array}{c c}{{}}&{{(P(a^{1})P(b^{1}\mid a^{1})+P(a^{2})P(b^{1}\mid a^{2}))~~~P(c^{1}\mid b^{1})~~~P(d^{1}\mid c^{1})}}\\ {{+}}&{{(P(a^{1})P(b^{2}\mid a^{1})+P(a^{2})P(b^{2}\mid a^{2}))~~~P(c^{1}\mid b^{2})~~~P(d^{1}\mid c^{1})}}\\ {{+}}&{{(P(a^{1})P(b^{1}\mid a^{1})+P(a^{2})P(b^{1}\mid a^{2}))~~~P(c^{2}\mid b^{1})~~~P(d^{1}\mid c^{2})}}\\ {{+}}&{{(P(a^{1})P(b^{2}\mid a^{1})+P(a^{2})P(b^{2}\mid a^{2}))~~~P(c^{2}\mid b^{2})~~~P(d^{1}\mid c^{2})}}\\ {{}}&{{}}&{{}}\\ {{}}&{{(P(a^{1})P(b^{1}\mid a^{1})+P(a^{2})P(b^{1}\mid a^{2}))~~~P(c^{1}\mid b^{1})~~~P(d^{2}\mid c^{1})}}\\ {{+}}&{{(P(a^{1})P(b^{2}\mid a^{1})+P(a^{2})P(b^{2}\mid a^{2}))~~~P(c^{1}\mid b^{2})~~~P(d^{2}\mid c^{1})}}\\ {{+}}&{{(P(a^{1})P(b^{1}\mid a^{1})+P(a^{2})P(b^{1}\mid a^{2}))~~~P(c^{2}\mid b^{1})~~~P(d^{2}\mid c^{2})}}\\ {{+}}&{{(P(a^{1})P(b^{2}\mid a^{1})+P(a^{2})P(b^{2}\mid a^{2}))~~~P(c^{2}\mid b^{2})~~~P(d^{2}\mid c^{2})}}\end{array}
$$ 

Figure 9.3 The ﬁrst transformation on the sum of ﬁgure 9.2 

$$
\begin{array}{c c c}{\tau_{1}(b^{1})}&{P(c^{1}\mid b^{1})}&{P(d^{1}\mid c^{1})}\\ {+}&{\tau_{1}(b^{2})}&{P(c^{1}\mid b^{2})}&{P(d^{1}\mid c^{1})}\\ {+}&{\tau_{1}(b^{1})}&{P(c^{2}\mid b^{1})}&{P(d^{1}\mid c^{2})}\\ {+}&{\tau_{1}(b^{2})}&{P(c^{2}\mid b^{2})}&{P(d^{1}\mid c^{2})}\\ {}&{}&{}&{}\\ {\tau_{1}(b^{1})}&{P(c^{1}\mid b^{1})}&{P(d^{2}\mid c^{1})}\\ {+}&{\tau_{1}(b^{2})}&{P(c^{1}\mid b^{2})}&{P(d^{2}\mid c^{1})}\\ {+}&{\tau_{1}(b^{1})}&{P(c^{2}\mid b^{1})}&{P(d^{2}\mid c^{2})}\\ {+}&{\tau_{1}(b^{2})}&{P(c^{2}\mid b^{2})}&{P(d^{2}\mid c^{2})}\end{array}
$$ 

$$
{\begin{array}{l l l}{}&{(\tau_{1}(b^{1})P(c^{1}\mid b^{1})+\tau_{1}(b^{2})P(c^{1}\mid b^{2}))}&{P(d^{1}\mid c^{1})}\\ {+}&{(\tau_{1}(b^{1})P(c^{2}\mid b^{1})+\tau_{1}(b^{2})P(c^{2}\mid b^{2}))}&{P(d^{1}\mid c^{2})}\\ {}&{}&{}\\ {+}&{(\tau_{1}(b^{1})P(c^{1}\mid b^{1})+\tau_{1}(b^{2})P(c^{1}\mid b^{2}))}&{P(d^{2}\mid c^{1})}\\ {+}&{(\tau_{1}(b^{1})P(c^{2}\mid b^{1})+\tau_{1}(b^{2})P(c^{2}\mid b^{2}))}&{P(d^{2}\mid c^{2})}\end{array}}
$$ 

Figure 9.5 The third transformation on the sum of ﬁgure 9.2 

$$
{\begin{array}{r l}{\tau_{2}(c^{1})}&{P(d^{1}\mid c^{1})}\\ {+}&{\tau_{2}(c^{2})}&{P(d^{1}\mid c^{2})}\\ {\,}&{}\\ {\tau_{2}(c^{1})}&{P(d^{2}\mid c^{1})}\\ {+}&{\tau_{2}(c^{2})}&{P(d^{2}\mid c^{2})}\end{array}}
$$ 

Figure 9.6 The fourth transformation on the sum of ﬁgure 9.2 

The ﬁnal expression is shown in ﬁgure 9.6. 

Summarizing, we begin by computing $\tau_{1}(B)$ , which requires four multiplications and two additions. Using it, we can compute $\tau_{2}(C)$ , which also requires four multiplications and two additions. Finally, we can compute $P(D)$ , again, at the same cost. The total number of operations is therefore 18. By comparison, generating the joint distribution requires $16\cdot3=48$ multiplications (three for each of the 16 entries in the joint), and 14 additions (7 for each of $P(d^{1})$ and $P(d^{2}))$ . 

Written somewhat more compactly, the transformation we have performed takes the following steps: We want to compute 

$$
P(D)=\sum_{C}\sum_{B}\sum_{A}P(A)P(B\mid A)P(C\mid B)P(D\mid C).
$$ 

We push in the ﬁrst summation, resulting in 

$$
\sum_{C}P(D\mid C)\sum_{B}P(C\mid B)\sum_{A}P(A)P(B\mid A).
$$ 

We compute the product $\psi_{1}(A,B)=P(A)P(B\mid A)$ a d then sum out $A$ to obtain the func- $\begin{array}{r}{\tau_{1}(B)=\sum_{A}\psi_{1}(A,B)}\end{array}$ . Speciﬁcally, for each value b , we compute $\begin{array}{r l r}{\tau_{1}(b)=\sum_{A}\psi_{1}(A,b)=}\end{array}$ $\textstyle\sum_{A}P(A)P(b\mid A)$ . We then continue by computing: 

$$
\begin{array}{r c l}{{\psi_{2}(B,C)}}&{{=}}&{{\tau_{1}(B)P(C\mid B)}}\\ {{\tau_{2}(C)}}&{{=}}&{{\displaystyle\sum_{B}\psi_{2}(B,C).}}\end{array}
$$ 

This computation results in a new vector $\tau_{2}(C)$ , which we then proceed to use in the ﬁnal phase of computing $P(D)$ . 

dynamic programming 

# 

This procedure is performing dynamic programming (see appendix A.3.3); doing this sum- mation the naive way w uld h us compute every $\begin{array}{r}{P(b)=\sum_{A}P(A)P(b\mid A)}\end{array}$ | many times, once for every value of C and D . In general, in a chain of length $n$ , this internal summation would be computed exponentially many times. Dynamic programming “inverts” the order of computation — performing it inside out instead of outside in. Speciﬁcally, we perform the innermost summation ﬁrst, computing once and for all the values in $\tau_{1}(B)$ ; that allows us to compute $\tau_{2}(C)$ once and for all, and so on. 

To summarize, the two ideas that help us address the exponential blowup of the joint distribution are: 

• Because of the structure of the Bayesian network, some subexpressions in the joint depend only on a small number of variables.

 • By computing these expressions once and caching the results, we can avoid generating them exponentially many times. 

# 9.3 Variable Elimination 

factor To formalize the algorithm demonstrated in the previous section, we need to introduce some basic concepts. In chapter 4, we introduced the notion of a factor $\phi$ over a scope $S c o p e[\phi]=X$ , which is a function $\phi:V a l(X)\mapsto I\!\!R$ . The main steps in the algorithm described here can be viewed as a manipulation of factors. Importantly, by using the factor-based view, we can deﬁne the algorithm in a general form that applies equally to Bayesian networks and Markov networks. 

![](images/7910f7d1d1658bb966789c46e52c859b7c766f3e0c7732e16b190a3c9bd7e344.jpg) 
Figure 9.7 Example of factor marginalization: summing out $B$ . 

# 9.3.1 Basic Elimination 

# 9.3.1.1 Factor Marginalization 

The key operation that we are performing when computing the probability of some subset of variables is that of marginalizing out variables from a distribution. That is, we have a distribution over a of variables $\mathcal{X}$ , and we want to compute the marginal of that distribution over some subset X . We can view this computation as an operation on a factor: 

Deﬁnition 9.3 factor marginalization Let $X$ be a set of v iab s, and $Y\notin X$ a variable. Let $\phi(X,Y)$ e a factor. We deﬁne the factor marginalization of Y $Y$ in φ , denoted $\textstyle\sum_{Y}\phi$ , to be a factor $\psi$ over X such that: 

$$
\psi(X)=\sum_{Y}\phi(X,Y).
$$ 

This operation is also called summing out of $Y$ in $\psi$ . 

The key point in this deﬁnition is that we only sum up entries in the table where the values of $X$ match up. Figure 9.7 illustrates this process. 

The process of marginalizing a joint distribution $P(X,Y)$ onto $X$ in a Bayesian network is simply summing out the variables $Y$ in the factor corresponding to $P$ . If we sum out all variables, we get a factor consisting of a single number whose value is 1 . If we sum out all of the variables in the unnormalized distribution $\tilde{P}_{\Phi}$ deﬁned by the product of factors in a Markov network, we get the partition function. 

A key observation used in performing inference in graphical models is that the operations of factor product and summation behave precisely as do product and summation over numbers. , both operations are commutative, s $\phi_{1}\cdot\phi_{2}\,=\,\phi_{2}\,\cdot\,\phi_{1}$ and $\begin{array}{r l}{\sum_{\boldsymbol{X}}\sum_{\boldsymbol{Y}}\phi\;=}\end{array}$ P $\textstyle\sum_{Y}\sum_{X}\phi$ . Products are also associative, so that $\left(\phi_{1}\cdot\phi_{2}\right)\cdot\phi_{3}=\phi_{1}\cdot\left(\phi_{2}\cdot\phi_{3}\right)$ · · · · ) . Most importantly, 

![](images/cc1c1a63031b2a68f051ab83d7b0ebe34448d256f2e8f04485bd4d5665ca7276.jpg) 

we have a simple rule allowing us to exchange summation and product: If $X\not\in S c o p e[\phi_{1}]$ , then 

$$
\sum_{X}(\phi_{1}\cdot\phi_{2})=\phi_{1}\cdot\sum_{X}\phi_{2}.
$$ 

# 9.3.1.2 The Variable Elimination Algorithm 

The key to both of our examples in the last section is the application of equation (9.6). Speciﬁ- cally, in our chain example of section 9.2, we can write: 

$$
P(A,B,C,D)=\phi_{A}\cdot\phi_{B}\cdot\phi_{C}\cdot\phi_{D}.
$$ 

On the other hand, the marginal distribution over $D$ is 

$$
P(D)=\sum_{C}\sum_{B}\sum_{A}P(A,B,C,D).
$$ 

Applying equation (9.6), we can now conclude: 

$$
\begin{array}{r c l}{{P(D)}}&{{=}}&{{\displaystyle\sum_{C}\displaystyle\sum_{B}\displaystyle\sum_{A}\phi_{A}\cdot\phi_{B}\cdot\phi_{C}\cdot\phi_{D}}}\\ {{}}&{{=}}&{{\displaystyle\sum_{C}\displaystyle\sum_{B}\phi_{C}\cdot\phi_{D}\cdot\left(\displaystyle\sum_{A}\phi_{A}\cdot\phi_{B}\right)}}\\ {{}}&{{=}}&{{\displaystyle\sum_{C}\phi_{D}\cdot\left(\displaystyle\sum_{B}\phi_{C}\cdot\left(\displaystyle\sum_{A}\phi_{A}\cdot\phi_{B}\right)\right),}}\end{array}
$$ 

where the diferent transformations are justiﬁed by the limited scope of the CPD factors; for example, the second equality is justiﬁed by the fact that the scope of $\phi_{C}$ and $\phi_{D}$ does not contain $A$ . In general, any marginal probability computation involves taking the product of all the CPDs, and doing a summation on all the variables except the query variables. We can do these steps in any order we want, as long as we only do a summation on a variable $X$ after multiplying in all of the factors that involve $X$ . 

In general, we can view the task at hand as that of computing the value of an expression of the form: 

$$
\sum_{Z}\prod_{\phi\in\Phi}\phi.
$$ 

sum-product 

variable elimination 

We call this task the sum-product inference task. The key insight that allows the efective computation of this expression is the fact that the scope of the factors is limited, allowing us to “push in” some of the summations, performing them over the product of only a subset of factors. One simple instantiation of this algorithm is a procedure called sum-product variable elimination (VE), shown in algorithm 9.1. The basic idea in the algorithm is that we sum out variables one at a time. When we sum out any variable, we multiply all the factors that mention that variable, generating a product factor. Now, we sum out the variable from this combined factor, generating a new factor that we enter into our set of factors to be dealt with. 

Based on equation (9.6), the following result follows easily: 

Theorem 9.5 Let $X$ e set of variables, and let $\Phi$ be a set o hat for each $\phi\in\Phi$ , $S c o p e[\phi]\subseteq X$ . Let Y $Y\subset X$ ⊂ be a set of query variables, and let $Z=X\mathrm{~-~}Y$ − . Then for any ordering ≺ over Z , Sum-Product $\textstyle\mathcal{\mathrm{NE}}(\Phi,Z,\prec)$ returns a factor $\phi^{*}(Y)$ such that 

$$
\phi^{*}(Y)=\sum_{Z}\prod_{\phi\in\Phi}\phi.
$$ 

We can apply this algorithm to the task of computing the probability distribution $P_{\mathcal{B}}(Y)$ for a Bayesian network $\mathcal{B}$ . We simply instantiate $\Phi$ to consist of all of the CPDs: 

$$
\Phi=\{\phi_{X_{i}}\}_{i=1}^{n}
$$ 

$\phi_{X_{i}}\;=\;P(X_{i}\;\mid\;\mathrm{Pa}_{X_{i}})$ . We then apply the variable elimination algorithm to the set $\left\{Z_{1},.\,.\,.\,,Z_{m}\right\}=\mathcal{X}-Y$ (that is, we eliminate all the nonquery variables). 

We can also apply precisely the same algorithm to the task of computing conditional prob- abilities in a Markov network. We simply initialize the factors to be the clique potentials and 

![](images/6792aca268a0c20d90c2d5d08c61c86cf022b9cdd03881750691da39b9a34698.jpg) 
Figure 9.8 The Extended-Student Bayesian network 

run the elimination algorithm. As for Bayesian networks, we then apply the variable elimination algorithm the set $Z=\mathcal{X}-Y$ . T procedure returns an unnormalized factor over the query variables Y . The distribution over $Y$ can be obtained by normalizing the factor; the partition function is simply the normalizing constant. 

Let us demonstrate the procedure on a nontrivial example. Consider the network demonstrated in ﬁgure 9.8, which is an extension of our Student network. The chain rule for this network asserts that 

$$
\begin{array}{r c l}{{P(C,D,I,G,S,L,J,H)}}&{{=}}&{{P(C)P(D\mid C)P(I)P(G\mid I,D)P(S\mid I)}}\\ {{}}&{{}}&{{P(L\mid G)P(J\mid L,S)P(H\mid G,J)}}\\ {{}}&{{=}}&{{\phi_{C}(C)\phi_{D}(D,C)\phi_{I}(I)\phi_{G}(G,I,D)\phi_{S}(S,I)}}\\ {{}}&{{}}&{{\phi_{L}(L,G)\phi_{J}(J,L,S)\phi_{H}(H,G,J).}}\end{array}
$$ 

We will now apply the $V E$ algorithm to compute $P(J)$ . We will use the elimination ordering: $C,D,I,H,G,S,L$ : 

1. Eliminating $C$ : We compute the factors 

$$
\begin{array}{r c l}{\psi_{1}(C,D)}&{=}&{\phi_{C}(C)\cdot\phi_{D}(D,C)}\\ {\tau_{1}(D)}&{=}&{\displaystyle\sum_{C}\psi_{1}.}\end{array}
$$ 

2. Eliminating $D$ : Note that we have already eliminated one of the original factors that involve $D$ — $\phi_{D}(D,C)=P(D\mid C)$ . On the other hand, we introduced the factor $\tau_{1}(D)$ that involves $D$ . Hence, we now compute: 

$$
\begin{array}{r c l}{{\psi_{2}(G,I,D)}}&{{=}}&{{\phi_{G}(G,I,D)\cdot\tau_{1}(D)}}\\ {{\tau_{2}(G,I)}}&{{=}}&{{\displaystyle\sum_{D}\psi_{2}(G,I,D).}}\end{array}
$$ 

3. Eliminating $I$ : We compute the factors 

$$
\begin{array}{r c l}{{\psi_{3}(G,I,S)}}&{{=}}&{{\phi_{I}(I)\cdot\phi_{S}(S,I)\cdot\tau_{2}(G,I)}}\\ {{\tau_{3}(G,S)}}&{{=}}&{{\displaystyle\sum_{I}\psi_{3}(G,I,S).}}\end{array}
$$ 

4. Eliminating $H$ : We compute the factors 

$$
\begin{array}{r c l}{{\psi_{4}(G,J,H)}}&{{=}}&{{\phi_{H}(H,G,J)}}\\ {{\tau_{4}(G,J)}}&{{=}}&{{\displaystyle\sum_{H}\psi_{4}(G,J,H).}}\end{array}
$$ 

Note that $\tau_{4}\equiv1$ (all of its entries are exac : we are simply computing $\textstyle\sum_{H}P(H\mid G,J)$ | , which is a probability distribution for every $G,J$ , and hence sums to 1. A naive execution of this algorithm will end up generating this factor, which has no value. Generating it has no impact on the ﬁnal answer, but it does complicate the algorithm. In particular, the existence of this factor complicates our computation in the next step. 

5. Eliminating $G$ : We compute the factors 

$$
\begin{array}{r c l}{{\psi_{5}(G,J,L,S)}}&{{=}}&{{\tau_{4}(G,J)\cdot\tau_{3}(G,S)\cdot\phi_{L}(L,G)}}\\ {{\tau_{5}(J,L,S)}}&{{=}}&{{\displaystyle\sum_{G}\psi_{5}(G,J,L,S).}}\end{array}
$$ 

Note that, without the factor $\tau_{4}(G,J)$ , the results of this step would not have involved $J$ .

 6. Eliminating $S$ : We compute the factors 

$$
\begin{array}{r c l}{{\psi_{6}(J,L,S)}}&{{=}}&{{\tau_{5}(J,L,S)\cdot\phi_{J}(J,L,S)}}\\ {{\tau_{6}(J,L)}}&{{=}}&{{\displaystyle\sum_{S}\psi_{6}(J,L,S).}}\end{array}
$$ 

7. Eliminating $L$ : We compute the factors 

$$
\begin{array}{r c l}{{\psi_{7}(J,L)}}&{{=}}&{{\tau_{6}(J,L)}}\\ {{\tau_{7}(J)}}&{{=}}&{{\displaystyle\sum_{L}\psi_{7}(J,L).}}\end{array}
$$ 

We summarize these steps in table 9.1. 

Note that we can use any elimination ordering. For example, consider eliminating variables in the order $G,\,I,\,S,\,L,\,H,\,C,\,D$ . We would then get the behavior of table 9.2. The result, as before, is precisely $P(J)$ . However, note that this elimination ordering introduces factors with much larger scope. We return to this point later on. 

Table 9.1 A run of variable elimination for the query $P(J)$ 
![](images/2c1c21ab7dfc4373ffe9279eab5d5e65efd1f2db25ea799b991a723190af1dbf.jpg) 

Table 9.2 A diferent run of variable elimination for the query $P(J)$ 
![](images/508834ed8348cfb3fc2a217215d55ec8f80288833133b13d73c94e63c4902e90.jpg) 

# 9.3.1.3 Semantics of Factors 

It is interesting to consider the semantics of the intermediate factors generated as part of this computation. In many of the examples we have given, they correspond to marginal or conditional probabilities in the network. However, although these factors often correspond to such probabilities, this is not always the case. Consider, for example, the network of ﬁgure $9.9\mathrm{a}$ . The result of eliminating the variable $X$ is a factor 

$$
\tau(A,B,C)=\sum_{X}P(X)\cdot P(A\mid X)\cdot P(C\mid B,X).
$$ 

This factor does not correspond to any probability or conditional probability in this network. To understand why, consider the various options for the meaning of this factor. Clearly, it cannot be a conditional distribution where $B$ is on the left hand side of the conditioning bar (for example, $P(A,B,C))$ , as $P(B\mid A)$ has not yet been multiplied in. The mo candidate is $P(A,C\mid B)$ . However, this conjecture is also false. The obability $P(A\mid B)$ | relies he ily on the properties of the CPD $P(B\mid A)$ for example, if B is determi stically equal to A , $P(A\mid B)$ has a very diferent form than if B depends only very weakly on A . Since the CPD $P(B\mid A)$ was not taken into consideration when computing $\tau(A,B,C)$ , it cannot represent the conditional probability $P(A,C\mid B)$ . In general, we can verify that this factor 

![](images/290124f499eb4a94cf1d8d6d480e2e6161cb54b9d17a29b72e9a6affa0a50999.jpg) 
Figure 9.9 Understanding intermediate factors in variable elimination as conditional probabilities: (a) A Bayesian network where elimination does not lead to factors that have an interpretation as conditional probabilities. (b) A diferent Bayesian network where the resulting factor does correspond to a conditional probability. 

does not correspond to any conditional probability expression in this network. 

It is interesting to note, however, that the resulting factor does, in fact, correspond to a conditional probability $P(A,C\mid B)$ , but in a diferent network : the one shown in ﬁgure 9.9b, where all CPDs except for B are the same. In fact, this phenomenon is a general one (see exercise 9.2). 

# 9.3.2 Dealing with Evidence 

It remains only to consider how we would introduce evidence. For example, assume we observe the value $i^{1}$ (the student is intelligent) and $h^{0}$ (the student is unhappy). Our goal is to compute $P(J\mid i^{1},h^{0})$ . First, we reduce this problem to computing the unnormalized distribution $P(J,i^{1},h^{0})$ . From this intermediate result, we can compute the conditional probability as in equation (9.1), by renormalizing by the probability of the evidence $P(i^{1},h^{0})$ . 

factor reduction 

How do we compute $P(J,i^{1},h^{0})$ ? The key observation is proposition 4.7, which shows us how to view, as a Gibbs distribution, an unnormalized measure derived from introducing evidence into a Bayesian network. Thus, we can view this computation as summing out all of the entries in the reduced factor : $P[i^{1}h^{0}]$ whose scope is $\{C,D,G,L,S,J\}$ . This factor is no longer normalized, but it is still a valid factor. 

Based on this observation, we can now apply precisely the same sum-product variable elim- ination algorithm to the task of computing $P(\pmb{Y},\pmb{e})$ . We simply apply the algorithm to the set of factors in the network, reduced by $E=e$ , and eliminate the variables in $\mathcal{X}-Y-E$ . The returned factor $\phi^{*}(Y)$ is precisely $P(\pmb{Y},\pmb{e})$ . To obtain $P(Y\mid e)$ we simply renormalize $\phi^{*}(Y)$ by multiplying it by $\textstyle{\frac{1}{\alpha}}$ to obtain a legal distribution, where $\alpha$ is the sum over the entries in our unnormalized distribution, which represents the probability of the evidence. To summarize, the algorithm for computing conditional probabilities in a Bayesian or Markov network is shown in algorithm 9.2. 

We demonstrate this process on the example of computing $P(J,i^{1},h^{0})$ . We use the same 

![](images/92d14ca92532f8aa3e03f262653607230847c2cdddc7402a825367d5387c2108.jpg) 

elimination ordering that we used in table 9.1. The results are shown in table 9.3; the step num- bers correspond to the steps in table 9.1. It is interesting to note the diferences between the two runs of the algorithm. First, we notice that steps (3) and (4) disappear in the computation with evidence, since $I$ and $H$ do not need to be eliminated. More interestingly, by not eliminating $I$ , we avoid the step that correlates $G$ and $S$ . In this execution, $G$ and $S$ never appear together in the same factor; they are both eliminated, and only their end results are combined. Intuitively, $G$ and $S$ are conditionally independent given $I$ ; hence, observing $I$ renders them independent, so that we do not have to consider their joint distribution explicitly. Finally, we notice that $\phi_{I}[I=i^{1}]\,=\,P(i^{1})$ is a factor over an empty scope, which is simply a number. It can be multiplied into any factor at any point in the computation. We chose arbitrarily to incorporate it into step $(2^{\prime})$ . Note that if our goal is to compute a conditional probability given the evidence, and not the probability of the evidence itself, we can avoid multiplying in this factor entirely, since its efect will disappear in the renormalization step at the end. 

is deﬁned over the following set of variables: 

• For each factor $\phi_{c}\in\Phi$ with scope $X_{c},$ , we have a variable $\theta_{\pmb{x}_{c}}$ for every $\pmb{x}_{c}\in V a l(\pmb{X}_{c})$ . • For each variable $X_{i}$ and every value $x_{i}\in V a l(X_{i})$ , we have a binary-valued variable $\lambda_{x_{i}}$ 

In other words, the polynomial has one argument for each of the network parameters and for each possible assignment to a network variable. The polynomial $f_{\Phi}$ is now deﬁned as follows: 

$$
f_{\Phi}(\pmb\theta,\pmb\lambda)=\sum_{x_{1},...,x_{n}}\left(\prod_{\phi_{c}\in\Phi}\theta_{\pmb x_{c}}\cdot\prod_{i=1}^{n}\lambda_{x_{i}}\right).
$$ 

Evaluating the network polynomial is equivalent to the inference task. In particular, let $Y=y$ be an assignment to some subset of network variables; deﬁne an assignment $\lambda^{y}$ as follows: 

With this deﬁnition, we can now show (exercise 9.4a) that: 

$$
f_{\Phi}(\pmb\theta,\lambda^{y})=\tilde{P}_{\Phi}(Y=y\mid\pmb\theta).
$$ 

The derivatives of the network polynomial are also of signiﬁcant interest. We can show (exer- cise 9.4b) that 

$$
\frac{\partial f_{\Phi}(\pmb\theta,\pmb\lambda^{\pmb y})}{\partial\lambda_{x_{i}}}=\tilde{P}_{\Phi}(x_{i},\pmb y_{-i}\mid\pmb\theta),
$$ 

where $\pmb{y}_{-i}$ is the assignment in $_{_y}$ to all variables other than $X_{i}$ . We can also show that 

$$
{\frac{\partial f_{\Phi}(\pmb\theta,\lambda^{y})}{\partial\theta_{x_{c}}}}={\frac{{\tilde{P}}_{\Phi}(\pmb y,\pmb x_{c}\mid\pmb\theta)}{\theta_{x_{c}}}}~;
$$ 

sensitivity analysis 

this fact is proved in lemma 19.1. These derivatives can be used for various purposes, including retracting or modifying evidence in the network (exercise 9.4c), and sensitivity analysis — comput- ing the efect of changes in a network parameter on the answer to a particular probabilistic query (exercise 9.5). 

Of course, as deﬁned, the representation of the network polynomial is exponentially large in the number of variables in the network. However, we can use the algebraic operations performed in a run of variable elimination to deﬁne a network polynomial that has precisely the same complexity as the VE run. More interesting, we can also use the same structure to compute efciently all of the derivatives of the network polynomial, relative both to the $\lambda_{i}$ and the $\theta_{\pmb{x}_{c}}$ (see exercise 9.6). 

# 9.4 Complexity and Graph Structure: Variable Elimination 

From the examples we have seen, it is clear that the VE algorithm can be computationally much more efcient than a full enumeration of the joint. In this section, we analyze the complexity of the algorithm, and understand the source of the computational gains. 

We also note that, aside from the asymptotic analysis, a careful implementation of this algorithm can have signiﬁcant ramiﬁcations on performance; see box 10.A. 

# 9.4.1 Simple Analysis 

Let us begin with a simple analysis of the basic computational operations taken by algorithm 9.1. Assume we have $n$ random variables, and $m$ initial factors; in a Bayesian network, we have $m=n$ ; in a Markov network, we may have more factors than variables. For simplicity, assume we run the algorithm until all variables are eliminated. 

The algorithm consists of a set of elimination steps, where, in each step, the algorithm picks a variable $X_{i}$ , then multiplies all factors involving that variable. The result is a single large factor $\psi_{i}$ . The variable then gets summed out of $\psi_{i}$ , resulting in a new factor $\tau_{i}$ whose scope is the scope of $\psi_{i}$ minus $X_{i}$ . Thus, the work revolves around these factors that get created and processed. Let $N_{i}$ be the number of entries in the factor $\psi_{i}$ , and let $N_{\mathrm{max}}=\operatorname*{max}_{i}N_{i}$ . 

We begin by counting the number of multiplication steps. Here, we note that the total number of factors ever entered into the set of factors $\Phi$ is $m+n$ : the $m$ initial factors, plus the $n$ factors $\tau_{i}$ . Each of these factors $\phi$ is multiplied exactly once: when it is multiplied in line 3 of Sum-Product-Eliminate-Var to produce a large factor $\psi_{i}$ , it is also extracted from $\Phi$ . The cost of multiplying $\phi$ to produce $\psi_{i}$ is at most $N_{i}$ , since each entry of $\phi$ is multiplied into $\psi_{i}$ the total number of multiplication steps is at most $(n+m)N_{i}\leq$ $(n+m)N_{\mathrm{max}}\,=\,O(m N_{\mathrm{max}})$ . To analyze the number of addition steps, we note that the marginalization operation in line 4 touches each entry in $\psi_{i}$ exactly once. Thus, the cost of this operation is exactly $N_{i}$ ; we execute this operation once for each factor $\psi_{i}$ , so that the total number of additions is at most $n N_{\mathrm{max}}$ . Overall, the total amount of work required is $O(m N_{\mathrm{max}})$ . 

The source of the inevitable exponential blowup is the potentially exponential size of the factors $\psi_{i}$ . If each variable has no more than $v$ values, and a factor $\psi_{i}$ has a scope that contains $k_{i}$ variables, then $N_{i}\leq v^{k_{i}}$ . Thus, we see that the computational cost of the VE algorithm is dominated by the sizes of the intermediate factors generated, with an exponential growth in the number of variables in a factor. 

# 9.4.2 Graph-Theoretic Analysis 

Although the size of the factors created during the algorithm is clearly the dominant quantity in the complexity of the algorithm, it is not clear how it relates to the properties of our problem instance. In our case, the only aspect of the problem instance that afects the complexity of the algorithm is the structure of the underlying graph that induced the set of factors on which the algorithm was run. In this section, we reformulate our complexity analysis in terms of this graph structure. 

# 9.4.2.1 Factors and Undirected Graphs 

We begin with the observation that the algorithm does not care whether the graph that generated the factors is directed, undirected, or partly directed. The algorithm’s input is a set of factors $\Phi$ , and the only relevant aspect to the computation is the scope of the factors. Thus, it is easiest to view the algorithm as operating on an undirected graph $\mathcal{H}$ . 

More precisely, we can deﬁne the notion of an undirected graph associated with a set of factors: 

Let $\Phi$ be a set of factors. We deﬁne 

$$
S c o p e[\Phi]=\cup_{\phi\in\Phi}S c o p e[\phi]
$$ 

to be the set of all variables appearing in any of the factors in $\Phi$ . We deﬁne ${\mathcal{H}}_{\Phi}$ to be the undirected graph whose nodes correspond to the variables in Scope [Φ] and where we have an edge $X_{i}{-}X_{j}\in{\mathcal{H}}_{\Phi}$ if and only if there exists a factor $\phi\in\Phi$ such that $X_{i},X_{j}\in S c o p e[\phi]$ . 

In words, t irected graph ${\mathcal{H}}_{\Phi}$ introduces a fully connected subgraph over t scope of each factor $\phi\in\Phi$ ∈ , and hence is the minimal I-map for the distribution induced by Φ . We can now show that: 

Let $P$ be a distribution deﬁned by multiplying the factors in $\Phi$ and normalizing to deﬁne $^a$ distribution. Letting $X=S c o p e[\Phi]$ , 

$$
P(X)=\frac{1}{Z}\prod_{\phi\in\Phi}\phi,
$$ 

here $\begin{array}{r}{Z=\sum_{X}\prod_{\phi\in\Phi}\phi}\end{array}$ Q . Then ${\mathcal{H}}_{\Phi}$ is the minimal Markov network $I\cdot$ map for $P$ , and the factors ∈ $\Phi$ are a parameter iz ation of this network that deﬁnes the distribution P . 

The proof is left as an exercise (exercise 9.7). 

Note that, for a set of factors $\Phi$ deﬁned by a Bayesian netwo $\mathcal{G}$ , in the case without evidence, the undirected graph ${\mathcal{H}}_{\Phi}$ is precisely the moralized graph of G . In this case, the product of the factors is a normalized distribution, so the partition function of the resulting Markov network is simply 1 . Figure 4.6a shows the initial graph for our Student example. 

More interesting is the Markov network induced by a set of factors $\Phi[e]$ deﬁned by the reduction of the factors in a Bayesian network to some context $E=e$ . In this case, recall that the variables in $E$ are removed from the factors, so $X=S c o p e[\Phi_{e}]=\mathcal{X}-E$ . Furthermore, as we discussed, the unnormalized product of the factors is $P(X,e)$ , and the partition function of the resulting Markov network is precisely $P(e)$ . Figure 4.6b shows the initial graph for our Student example with evidence $G\;=\;g$ , and ﬁgure 4.6c shows the case with evidence $G=g,S=s$ . 

# 9.4.2.2 Elimination as Graph Transformation 

Now, consider the efect of a variable elimination step on the set of factors maintained by the algorithm and on the associated Markov network. When a variable $X$ is eliminated, several operations take place. First, we create a single factor $\psi$ that contains $X$ and all of the variables $Y$ with which it appears in factors. Then, we eliminate $X$ from $\psi$ , replacing it with a new factor $\tau$ that contains all of the variables $Y$ but does not contain $X$ . Let $\Phi_{X}$ be the resulting set of factors. 

ﬁll edge 

How does the graph ${\mathcal{H}}_{\Phi_{X}}$ from $\mathcal{H}_{\Phi}\mathfrak{H}$ The step of constructing $\psi$ generates edges between all of the variables Y $Y\in Y$ ∈ . Some of them were present in ${\mathcal{H}}_{\Phi}$ , whereas others are introduced due to the elimination step; edges that are introduced by an elimination step are called ﬁll edges . The step of eliminating $X$ from $\psi$ to construct $\tau$ has the efect of removing $X$ and all of its incident edges from the graph. 

![](images/31cdfc0084379ddb6de8422d0d10272e1cbef6d6f4f830d88863477cfed31aa5.jpg) 
Figure 9.10 Variable elimination as graph transformation in the Student example, using the elimi- nation order of table 9.1: (a) after eliminating $C$ ; (b) after eliminating $D$ ; (c) after eliminating $I$ . 

Consider again our Student network, in the case without evidence. As we said, ﬁgure 4.6a shows the original Markov network. Figure 9.10a shows the result of eliminating the variable $C$ . Note that there are no ﬁll edges introduced in this step. 

After an elimination step, the subsequent elimination steps use the new set of factors. In other words, they can be seen as operations over the new graph. Figure 9.10b and c show the graphs resulting from eliminating ﬁrst $D$ and then $I$ . Note that the step of eliminating $I$ results in a (new) ﬁll edge $G{-}S$ , induced by the factor $G,I,S$ . 

The computational steps of the algorithm are reﬂected in this series of graphs. Every factor that appears in one of the steps in the algorithm is reﬂected in the graph as a clique. In fact, we can summarize the computational cost using a single graph structure. 

# 9.4.2.3 The Induced Graph 

We deﬁne an undirected graph that is the union of all of the graphs resulting from the diferent steps of the variable elimination algorithm. 

Deﬁnition 9.5 induced graph Let $\Phi$ of factors over ${\mathcal X}\,=\,\{X_{1},.\,.\,.\,,X_{n}\}.$ , and $\prec\,b e$ an elimin on order for so subset $X\subseteq\mathcal{X}$ ⊆X . The induced graph $\mathcal{T}_{\Phi,\prec}$ is an undirected graph over X , where $X_{i}$ and $X_{j}$ are connected by an edge if they both appear in some intermediate factor $\psi$ generated by the VE algorithm using $\prec$ as an elimination ordering. For a Bayesian network graph $\mathcal{G}$ , we use $\mathcal{T}_{\mathcal{G},\prec}$ to denote the in ced graph for the factors $\Phi$ corresponding to the CPDs in $\mathcal{G}$ ; similarly, for a Markov network H we use $\mathcal{T}_{\mathcal{H},\prec}$ to denote the induced graph for the factors Φ corresponding to the potentials in H . The indu aph $\mathcal{T}_{\mathcal{G},\prec}$ for our Student example is show in ﬁgure 9.11a. We can see that the ﬁll edge $G{-}S$ , introduced in step (3) when we eliminated I , is the only ﬁll edge introduced. As we discussed, each factor $\psi$ used in the computation corresponds to a complete subgraph of the graph $\mathcal{T}_{\mathcal{G},\prec}$ and is therefore a clique in the graph. The connection between cliques in $\mathcal{T}_{\mathcal{G},\prec}$ and factors $\psi$ is, in fact, much tighter: 

![](images/d556a8c24eb9ce9fbc4270a465fb5393f5392fca8f908544832931ba7d517f40.jpg) 
(c) 

1. The scope of every factor generated during the variable elimination process is a clique in $\mathcal{T}_{\Phi,\prec}$ .

 2. Every maximal clique in $\mathcal{L}_{\Phi,-}$ is the scope of some intermediate factor in the computation. 

Proof We begin with the ﬁrst statement. Consider a factor $\psi(Y_{1},.\,.\,.\,,Y_{k})$ generated during the VE process. By the deﬁnition of the induced graph, there must be an edge between each $Y_{i}$ and $Y_{j}$ . Hence $Y_{1},\ldots,Y_{k}$ form a clique. 

To prove the second stateme consider some maximal clique $Y=\{Y_{1},.\,.\,.\,,Y_{k}\}$ . Assume, without loss of generality, that $Y_{1}$ is the ﬁrst of the varia s in Y in the ordering $\prec_{!}$ , and is therefore the ﬁrst among this set to be eliminated. Since Y $Y$ is a clique, there is an edge from $Y_{1}$ to each other $Y_{i}$ . Note that, once $Y_{1}$ is eliminated, it can appear in no more factors, so there can be no new edges added to it. Hence, the edges involving $Y_{1}$ were added prior to this point in the computation. The existence of an edge between $Y_{1}$ and $Y_{i}$ therefore implies that, at this point, there is a factor containing both $Y_{1}$ and $Y_{i}$ . When $Y_{1}$ is eliminated, all these factors must be multiplied. Therefore, the product step results in a factor $\psi$ that contains all of $Y_{1},Y_{2},.\,.\,.\,,Y_{k}$ . Note that this factor can contain no other variables; if it did, these variables would also have an edge to all of $Y_{1},\ldots,Y_{k}$ , so that $Y_{1},\ldots,Y_{k}$ would not constitute a maximal connected subgraph. 

Let us verify that the second property holds for our example. Figure 9.11b shows the maximal cliques in $\mathcal{T}_{\mathcal{G},\prec}$ : 

$$
\begin{array}{l c l}{{{\cal C}_{1}}}&{{=}}&{{\{{\cal C},{\cal D}\}}}\\ {{{\cal C}_{2}}}&{{=}}&{{\{{\cal D},{\cal I},{\cal G}\}}}\\ {{{\cal C}_{3}}}&{{=}}&{{\{{\cal I},{\cal G},{\cal S}\}}}\\ {{{\cal C}_{4}}}&{{=}}&{{\{{\cal G},{\cal J},{\cal L},{\cal S}\}}}\\ {{{\cal C}_{5}}}&{{=}}&{{\{{\cal G},{\cal H},{\cal J}\}.}}\end{array}
$$ 

Both these properties hold for this set of cliques. For example, $C_{3}$ corresponds to the factor $\psi$ generated in step (5). 

Thus, there is a direct correspondence between the maximal factors generated by our algorithm and maximal cliques in the induced graph. Importantly, the induced graph and the size of the maximal cliques within it depend strongly on the elimination ordering. Consider, for example, our other elimination ordering for the Student network. In this case, we can verify that our induced graph has a maximal clique over $G,I,D,L,J,H$ , a second over $S,I,D,L,J,H_{z}$ , and a third over $C,D,J$ ; indeed, the graph is missing only the edge between $S$ and $G$ , and some edges involving $C$ . In this case, the largest clique contains six variables, as opposed to four in our original ordering. Therefore, the cost of computation here is substantially more expensive. 

Deﬁnition 9.6 induced width 

tree-width 

We deﬁne the width of an induced graph to be the number of nodes in the largest clique in the graph minus 1. We deﬁne the induced width $w_{\mathcal{K},\prec}$ of an ordering $\prec$ tiv o a graph $\mathcal{K}$ (direct or undirected) to be the width of the ph $\mathcal{T}_{\mathcal{K},\prec}$ induced by applying VE to K using the ordering ≺ . We deﬁne the tree-width of a graph K to be its minimal induced width $\begin{array}{r}{w_{K}^{*}=\operatorname*{min}_{\prec}w(\mathcal{Z}_{K,\prec})}\end{array}$ I . ≺ K ≺ 

The minimal induced width of the graph $\mathcal{K}$ provides us a bound on the est performance we can hope for by applying VE to a probabilistic model that factorizes over K . 

# 9.4.3 Finding Elimination Orderings $\star$ 

How can we compute the minimal induced width of the graph, and the elimination ordering achieving that width? Unfortunately, there is no easy way to answer this question. 

Theorem 9.7 

It follows directly that ﬁnding the optimal elimination ordering is also $\mathcal{N P}$ -hard. Thus, we cannot easily tell by looking at a graph how computationally expensive inference on it will be. Note that this $\mathcal{N P}$ -completeness result is distinct from the $\mathcal{N P}$ -hardness of inference itself. That is, even if some oracle gives us the best elimination ordering, the induced width might still be large, and the inference task using that ordering can still require exponential time. 

However, as usual, $\mathcal{N P}$ -hardness is not the end of the story. There are several techniques that one can use to ﬁnd good elimination orderings. The ﬁrst uses an important graph-theoretic property of induced graphs, and the second uses heuristic ideas. 

# 9.4.3.1 Chordal Graphs 

chordal graph Recall from deﬁnition 2.24 that an undirected graph is chordal if it contains no cycle of length greater than three that has no “shortcut,” that is, every minimal loop in the graph is of length three. As we now show, somewhat surprisingly, the class of induced graphs is equivalent to the class of chordal graphs. We then show that this property can be used to provide one heuristic for constructing an elimination ordering. 

Theorem 9.8 Every induced graph is chordal. 

Proof Assume by contradiction that we have such a cycle $X_{1}{\mathrm{-}}X_{2}{\mathrm{-}}\ldots{\mathrm{-}}X_{k}{\mathrm{-}}X_{1}$ for $k>3$ , and assume without loss of generality that $X_{1}$ is the ﬁrst variable to be eliminated. As in the proof of theorem 9.6, no edge incident on $X_{1}$ is added after $X_{1}$ is eliminated; hence, both edges $X_{1}{-}X_{2}$ and $X_{1}{-}X_{k}$ must exist at this point. Therefore, the edge $X_{2}{-}X_{k}$ will be added at the same time, contradicting our assumption. 

verify that the graph re 9.11a is chordal. For example, the loop $H\rightarrow$ $G\rightarrow L\rightarrow J\rightarrow H$ is cut by the chord $G\rightarrow J$ . 

The converse of this theorem states that any chordal graph $\mathcal{H}$ is an induced graph for ome orderi One way of showing that is to show that there is an elimination ordering for H for which itself is the induced graph. 

# Theorem 9.9 

Any chordal graph $\mathcal{H}$ admits an elimination ordering that does not introduce any ﬁll edges into the graph. 

Proof We p ove this result by induction on the number of nodes in the tr . Le $\mathcal{H}$ be ordal graph with n nodes. As we showed in theorem 4.12, there is a clique tree T $\mathcal{T}$ for H . Let $C_{k}$ be a clique in the tree that is a leaf, that is, it has only a single other clique as a neighbor. Let $X_{i}$ be me variabl at is in $C_{k}$ but not in its n bor. Let ${\mathcal{H}}^{\prime}$ be the graph obtained by eliminating $X_{i}$ . Because $X_{i}$ belon nly to the clique $C_{k}$ , its neighbors are precisely $C_{k}-\{X_{i}\}$ . Because all of them are also in $C_{k}$ , they are connected to each other. Hence, eliminating $X_{i}$ introduces no ﬁll edges. Because ${\mathcal{H}}^{\prime}$ is also chordal, we can now apply the inductive hypothesis, proving the result. 

![](images/4735fc0cfa22bcaea45345a50f77cdc946bb8e6dd36596cd750aaac5fa42261c.jpg) 

# Example 9.2 

maximum cardinality 

Example 9.3 We can illustrate this construction on the graph of ﬁgure 9.11a. The maximal cliques in the induced graph are shown in $b,$ , and a clique tree for this graph is shown in c. One can easily verify that each sepset separates the two sides of the tree; for example, the sepset $\{G,S\}$ separates $C,I,D$ (on the left) from $L,J,H$ (on the right). The elimination ordering $C,D,I,H,G,S,L,J$ , an extension of the elimination in table 9.1 that generated this induced graph, is one ordering that might arise from the construction of theorem 9.9. For example, it ﬁrst eliminates $C,D$ , which are both in $^a$ leaf clique; it then eliminates $I$ , which is in a clique that is now a leaf, following the elimination of $C,D$ . Indeed, it is not hard to see that this ordering introduces no ﬁll edges. By contrast, the ordering in table 9.2 is not consistent with this construction, since it begins by eliminating the variables $G,I,S$ , none of which are in a leaf clique. Indeed, this elimination ordering introduces additional ﬁll edges, for example, the edge $H\rightarrow D$ . 

An alternative method for constructing an elimination ordering that introduces no ﬁll edges in a chordal graph is the Max-Cardinality algorithm, shown in algorithm 9.3. This method does not use the clique tree as its starting point, but rather operates directly on the graph. When applied to a chordal graph, it constructs an elimination ordering that eliminates cliques one at a time, starting from the leaves of the clique tree; and it does so without ever considering the clique tree structure explicitly. 

Consider applying Max-Cardinality to the chordal graph of ﬁgure 9.11. Assume that the ﬁrst node selected is $S$ . The second node selected must be one of $S$ ’s neighbors, say $J$ . The node that has the largest number of marked neighbors are now $G$ and $L$ , which are chosen subsequently. Now, the unmarked nodes that have the largest number of marked neighbors (two) are $H$ and $I$ . Assume we select $I$ . Then the next nodes selected are $D$ and $H$ , in any order. The last node to be selected is $C$ . One possible resulting ordering in which nodes are marked is thus $S,J,G,L,I,H,D,C$ . Importantly, the actual elimination ordering proceeds in reverse. Thus, we ﬁrst eliminate $C,D,$ , then $H$ , and so on. We can now see that this ordering always eliminates a variable from a clique that is a leaf clique at the time. For example, we ﬁrst eliminate $C,D$ from a leaf clique, then $H$ , then $G$ from the clique $\{G,I,D\}$ , which is now (following the elimination of $C,D)$ a leaf. 

As in this example, Max-Cardinality always produces an elimination ordering that is consistent with the construction of theorem 9.9. As a consequence, it follows that Max-Cardinality , when applied to a chordal graph, introduces no ﬁll edges. 

Theorem 9.10 Let H be a chordal graph. Let π be the ranking obtained by running Max-C dinality on $\mathcal{H}$ . Then Sum-Product-VE (algorithm 9.1), eliminating variables in order of increasing π , does not introduce any ﬁll edges. 

The proof is left as an exercise (exercise 9.8). 

The maximum cardinality search algorithm can also be used to construct an elimination ordering for a nonchordal graph. However, it turns out that the orderings produced by this method are generally not as good as those produced by various other algorithms, such as those described in what follows. 

triangulation 

polytree 

To summarize, we have shown that, if we construct a chordal graph that contains the graph ${\mathcal{H}}_{\Phi}$ corresponding to our s of factors $\Phi$ , we can use it as the basis for inference using $\Phi$ . The process of turning a graph H into a chordal graph is also called triangulation , since it ensures that the largest unbroken cycle in the graph is a triangle. Thus, we can reformulate our goal of ﬁnding an elimination ordering as that of triangulating a graph $\mathcal{H}$ so that the largest clique in the resulting graph is as small as possible. Of course, this insight only reformulates the problem: Inevitably, the problem of ﬁnding such a minimal triangulation is also $\mathcal{N P}$ -hard. Nevertheless, there are several graph-theoretic algorithms that address this precise problem and ofer diferent levels of performance guarantee; we discuss this task further in section 10.4.2. 

Box 9.B — Concept: Polytrees. One particularly simple class of chordal graphs is the class of Bayesian networks whose graph $\mathcal{G}$ is $a$ polytree . Recall from deﬁnition 2.22 that a polytree is a graph where there is at most one trail between every pair of nodes. 

Polytrees received a lot of attention in the early days of Bayesian networks, because the ﬁrst widely known inference algorithm for any type of Bayesian network was Pearl’s message passing algorithm for polytrees. This algorithm, a special case of the message passing algorithms described in subsequent chapters of this book, is particularly compelling in the case of polytree networks, since it consists of nodes passing messages directly to other nodes along edges in the graph. Moreover, the cost of this computation is linear in the size of the network (where the size of the network is measured as the total sizes of the CPDs in the network, not the number of nodes; see exercise 9.9). From the perspective of the results presented in this section, this simplicity is not surprising: In a polytree, any maximal clique is a family of some variable in the network, and the clique tree structure roughly follows the network topology. (We simply throw out families that do not correspond to a maximal clique, because they are subsumed by another clique.) 

Somewhat ironically, the compelling nature of the polytree algorithm gave rise to a long-standing misconception that there was a sharp tractability boundary between polytrees and other networks, in that inference was tractable only in polytrees and NP-hard in other networks. As we discuss in this chapter, this is not the case; rather, there is a continuum of complexity deﬁned by the size of the largest clique in the induced graph. 

# 9.4.3.2 Minimum Fill/Size/Weight Search 

An alternative approach for ﬁnding elimination orderings is based on a very straightforward intuition. Our goal is to construct an ordering that induces a “small” graph. While we cannot 

![](images/1f6e5c40cec9273d7551ce9744ddbb6d532cd17389d1373b9205426f560524a5.jpg) 

ﬁnd an ordering that achieves the global minimum, we can eliminate variables one at a time in a greedy way, so that each step tends to lead to a small blowup in size. 

The general algorithm is shown in algorithm 9.4. At each point, the algorithm evaluates each of the remaining variables in the network based on its heuristic cost function. Some common cost criteria that have been used for evaluating variables are: 

• Min-neighbors: The cost of a vertex is the number of neighbors it has in the current graph.

 • Min-weight: The cost of a vertex is the product of weights — domain cardinality — of its neighbors.

 • Min-ﬁll: - The cost of a vertex is the number of edges that need to be added to the graph due to its elimination.

 • Weighted-min-ﬁll: The cost of a vertex is the sum of weights of the edges that need to be added to the graph due to its elimination, where a weight of an edge is the product of weights of its constituent vertices. 

Intuitively, min-n ghbors and min-weight count the size or weight of the largest clique in $\mathcal{H}$ after eliminating X . Min-ﬁll and weighted-min-ﬁll count the number or weight of edges that would be introduced into $\mathcal{H}$ by eliminating $X$ . It can be shown (exercise 9.10) that none of these criteria is universally better than the others. 

This type of greedy search can be done either deterministic ally (as shown in algorithm 9.4), or stochastically. In the stochastic variant, at each step we select some number of low-scoring vertices, and then choose among them using their score (where lower-scoring vertices are selected with higher probability). In the stochastic variants, we run multiple iterations of the algorithm, and then select the ordering that leads to the most efcient elimination — the one where the sum of the sizes of the factors produced is smallest. 

Empirical results show that these heuristic algorithms perform surprisingly well in practice. Generally, Min-Fill and Weighted-Min-Fill tend to work better on more problems. Not surpris- ingly, Weighted-Min-Fill usually has the most signiﬁcant gains when there is some signiﬁcant variability in the sizes of the domains of the variables in the network. Box 9.C presents a case study comparing these algorithms on a suite of standard benchmark networks. 

Box 9.C — Case Study: Variable Elimination Orderings. Fishelson and Geiger (2003) performed a comprehensive case study of diferent heuristics for computing an elimination ordering, testing them on eight standard Bayesian network benchmarks, ranging from 24 nodes to more than 1,000. For each network, they compared both to the best elimination ordering known previously, obtained by an expensive process of simulated annealing search, and to the network obtained by a state- of-the-art Bayesian network package. They compared to stochastic versions of the four heuristics described in the text, running each of them for 1 minute or 10 minutes, and selecting the best network obtained in the diferent random runs. Maximum cardinality search was not used, since it is known to perform quite poorly in practice. 

The results, shown in ﬁgure 9.C.1, suggest several conclusions. First, we see that running the stochastic algorithms for longer improves the quality of the answer obtained, although usually not by a huge amount. We also see that diferent heuristics can result in orderings whose computational cost can vary in almost an order of magnitude. Overall, Min-Fill and Weighted-Min-Fill achieve the best performance, but they are not universally better. The best answer obtained by the greedy algorithms is generally very good; it is often signiﬁcantly better than the answer obtained by a deterministic state-of-the-art scheme, and it is usually quite close to the best-known ordering, even when the latter is obtained using much more expensive techniques. Because the computational cost of the heuristic ordering-selection algorithms is usually negligible relative to the running time of the inference itself, we conclude that for large networks it is worthwhile to run several heuristic algorithms in order to ﬁnd the best ordering obtained by any of them. 

# 9.5 Conditioning $\star$ 

# 

An alternative approach to inference is based on the idea of conditioning . The conditioning algorithm is based on the fact (illustrated in section 9.3.2), that observing the value of certain variables can simplify the variable elimination process. When a variable is not observed, we can use a case analysis to enumerate its possible values, perform the simpliﬁed VE computation, and then aggregate the results for the diferent values. As we will discuss, in terms of number of operations, the conditioning algorithm ofers no beneﬁt over the variable elimination al- gorithm. However, it ofers a continuum of time-space trade-ofs, which can be extremely important in cases where the factors created by variable elimination are too big to ﬁt in main memory. 

# 9.5.1 The Conditioning Algorithm 

The conditioning algorithm is easiest to explain in the context of a Markov network. Let $\Phi$ be a set of factors over $X$ and $P_{\Phi}$ be the associated distribution. We assume that any observations were already assimilated into $\Phi$ , so that our goal is to compute $P_{\Phi}(\pmb{Y})$ for some set of query variables $Y$ . For example, if we want to do inference in the Student network given the evidence $G=g$ , we would reduce the factors reduced to this context, giving rise to the network structure shown in ﬁgure $4.6\mathrm{b}$ . 

![](images/74a76c49f09b8e4965ade5a5c617045220d9a710e0e6ff76d697ba728455459e.jpg) 
Figure 9.C.1 — Comparison of algorithms for selecting variable elimination ordering. 

Computational cost of variable elimination inference in a range of benchmark networks, obtained by various algorithms for selecting an elimination ordering. The cost is measured as the size of the factors generated during the process of variable elimination. For each network, we see the cost of the best-known ordering, the ordering obtained by Hugin (a state-of-the-art Bayesian network package), and the ordering obtained by stochastic greedy search using four diferent search heuristics — Min-Neighbors, Min-Weight, Min-Fill, and Weighted-Min-Fill — run for 1 minute and for 10 minutes. 

![](images/b77a7a5be7063f947b9056ce4539581a1c9feb0289bdbc5864425b2c9b9fc106.jpg) 

The conditioning algorithm is based on the following simple derivation. Let $U\subseteq X$ be any set of variables. Then we have that: 

$$
\tilde{P}_{\Phi}(Y)=\sum_{\mathbf{\substack{u}\in\,}V a l(U)}\tilde{P}_{\Phi}(Y,\mathbf{\acute{u}}).
$$ 

The key observation is that each term $\tilde{P}_{\Phi}(Y,{\pmb u})$ can be computed by marginalizing out the variable in $X\mathrm{~-~}U\mathrm{~-~}Y$ in the unnormalized measure $\tilde{P}_{\Phi}[{\pmb u}]$ obtained by reducing $\mathbf{\bar{\mathit{P}}}_{\Phi}$ to the context u . As we have already discussed, the reduced measure is simply the measure deﬁned by reducing each of the factors to the context $\mathbfit{u}$ . The reduction process generally produces a simpler structure, with a reduced inference cost. 

We can use this formula to compute $P_{\Phi}(\pmb{Y})$ as follows: We construct a network $\mathcal{H}_{\Phi}[\pmb{u}]$ for each assignment $\mathbfit{u}$ ; these networks have identical structures, but diferent parameters. We run sum-product inference in each of them, to obtain a factor over the desired query set $Y$ . We then simply add up these factors to obtain $\tilde{P}_{\Phi}(Y)$ . We can also derive $P_{\Phi}(\pmb{Y})$ by renormalizing this factor to obtain a distribution. As usual, the normalizing constant is the partition function for $P_{\Phi}$ . However, applying equation (9.11) to the case of $Y=\emptyset$ , we conclude that 

$$
Z_{\Phi}=\sum_{\pmb{u}}Z_{\Phi[\pmb{u}]}.
$$ 

Thus, we can derive the overall partition function from the partition functions for the diferent subnetworks $\mathcal{H}_{\Phi[\pmb{u}]}$ . The ﬁnal algorithm is shown in algorithm 9.5. (We note tha Cond-Prob-VE was called without evidence, since we assumed for simplicity that our factors Φ have already been reduced with the evidence.) 

Assume that we want to compute $P(J)$ in the Student network with evidence $G=g^{1}$ , so that our initial graph would be the one shown in ﬁgure $4.6b$ . We can now perform inference by enumerating all of the assignments s to the variable $S$ . For each such assignment, we run inference on a graph structured as in ﬁgure 4.6c, with the factors reduced to the assignment $g^{1},s$ . In each such network we compute a factor over $J$ , and add them all up. Note that the reduced network contains two disconnected components, and so we might be tempted to run inference only on the component that contains $J$ . However, that procedure would not produce a correct answer: The value we get by summing out the variables in the second component multiplies our ﬁnal factor. Although this is $^a$ constant multiple for each value of $s$ , these values are generally diferent for the diferent values of $S$ . Because the factors are added before the ﬁnal renormalization, this constant inﬂuences the weight of one factor in the summation relative to the other. Thus, if we ignore this constant component, the answers we get from the $s^{1}$ computation and the $s^{0}$ computation would be weighted incorrectly. 

cutset conditioning 

Historically, owing to the initial popularity of the polytree algorithm, the conditioning ap- proach was mostly used in the case where the transformed network is a polytree. In this case, the algorithm is called cutset conditioning . 

# 9.5.2 Conditioning and Variable Elimination 

At ﬁrst glance, it might appear as if this process saves us considerable computational cost over the variable elimination algorithm. After all, we have reduced the computation to one that performs variable elimination in a much simpler network. The cost arises, of course, from the fact that, when we condition on $U$ , we need to perform variable elimination on the conditioned network multiple times, once for each assignment $u\in V a l(U)$ . e cost of this computation is $O(|V a l(\pmb{U})|)$ , which is exponential in the number of variables in U . Thus, we have not avoided the exponential blowup associated with the probabilistic inference process. In this section, we provide a formal complexity analysis of the conditioning algorithm, and compare it to the complexity of elimination. This analysis also reveals various interesting improvements to the basic conditioning algorithm, which can dramatically improve its performance in certain cases. 

To understand the operation of the conditioning algorithm, we return to the basic description of the probabilistic inference task. Consider our query $J$ in the Extended Student network. We know that: 

$$
p(J)=\sum_{C}\sum_{D}\sum_{I}\sum_{S}\sum_{G}\sum_{L}\sum_{H}P(C,D,I,S,G,L,H,J).
$$ 

Reordering this expression slightly, we have that: 

$$
p(J)=\sum_{g}\left[\sum_{C}\sum_{D}\sum_{I}\sum_{S}\sum_{L}\sum_{H}P(C,D,I,S,g,L,H,J)\right].
$$ 

The expression inside the parentheses is precisely the result of computing the probability of $J$ in the network $\mathcal{H}_{\Phi_{G=g}}$ , where $\Phi$ is the set of CPD factors in $\mathcal{B}$ . 

In other words, the conditioning algorithm is simply executing parts of the basic summation deﬁning the inference task by case analysis, enumerating the possible values of the conditioning 

![](images/1f965b24e2c88bafa776e8efd8693a4ca915d4e4dd83a37f5e25fceb4f5ca10f.jpg) 

variables. By contrast, variable elimination performs the same summation from the inside out, using dynamic programming to reuse computation. 

Indeed, if we simply did conditioning on all of the variables, the result would be an explicit summation of the entire joint distribution. In conditioning, however, we perform the condi- tioning step only on some of the variables, and use standard variable elimination — dynamic programming — to perform the rest of the summation, avoiding exponential blowup (at least over that part). 

In general, it follows that both algorithms are performing the same set of basic operations (sums and products). However, where the variable elimination algorithm uses the caching of dynamic programming to save redundant computation throughout the summation, conditioning uses a full enumeration of cases for some of the variables, and dynamic programming only at the end. 

From this argument, it follows that conditioning always performs no fewer steps than variable elimination. To understand why, consider the network of example 9.4 and assume that we are trying to compute $P(J)$ . The conditioned network ${\mathcal{H}}_{\Phi_{G=g}}$ has a set of factors most of which are identical to those in the original network. The exceptions are the reduced factors: $\phi_{L}[G=g](L)$ and $\phi_{H}[G=g](H,J)$ . For each of the three values $g$ of $G$ , we are performing variable elimination over these factors, eliminating all variables except for $G$ and $J$ . 

We can imagine “lumping” these three computations into one, by augmenting the scope of each factor with the variable $G$ . More precisely, we deﬁne a set of augmented factors $\phi^{+}$ as follows: The scope of the factor $\phi_{G}$ already contains $G$ , so $\phi_{G}^{+}(G,D,I)=\phi_{G}(G,D,I)$ . For the factor $\phi_{L}^{+}$ , we simply combine the three factors $\phi_{L,g}(L)$ , so that $\phi_{L}^{+}(L,g)=\phi_{L}[G=g](L)$ for all $g$ . Not surprisingly, the resulting factor $\phi_{L}^{+}(L,G)$ is simply our original CPD factor $\phi_{L}(L,G)$ . We deﬁne $\phi_{H}^{+}$ in the same way. The remaining factors are unrelated to $G$ . For each other variable $X$ over scope $Y$ , we simply deﬁne $\phi_{X}^{+}(Y,G)=\phi_{X}(Y)$ ; that is, the value of the factor does not depend on the value of $G$ . 

We can easily verify that, if we run variable elimination over he se of factors $\mathcal{F}_{X}^{+}$ for $X\,\in\,\{C,D,I,G,S,L,J,H\}$ , eliminating all variables except for J and G , we are perform- ing precisely the same computation as the three iterations of variable elimination for the three diferent conditioned networks $\mathcal{H}_{\Phi_{G=g}}$ : Factor entries involving diferent values $g$ of $G$ never in- 

![](images/accdf1e5efff758ee985d89c56f95ff4adca95ef4a0f4296d57d15b36c96605d.jpg) 

teract, and the computation performed for the entries where $G=g$ is precisely the computation performed in the network $\mathcal{H}_{\Phi_{G=g}}$ . 

Speciﬁcally, assume we are using the ordering $C,D,I,H,S,L$ to perform the elimination within each conditioned network $\mathcal{H}_{\Phi_{G=g}}$ . The steps of the computation are shown in table 9.4. Step (7) corresponds to the product of all of the remaining factors, which is the last step in variable elimination. The ﬁnal step in the conditioning algorithm, where we add together the results of the three computations, is precisely the same as eliminating $G$ from the resulting factor $\tau_{7}(G,J)$ . 

It is instructive to compare this execution to the one obtained by running variable elimination on the original set of factors, with the elimination ordering $C,D,I,H,S,L,G$ ; that is, we follow the ordering used within the conditioned networks for the variables other than $G,J$ , and then eliminate $G$ at the very end. In this process, shown in table 9.5, some of the factors involve $G$ , but others do not. In particular, step (1) in the elimination algorithm involves only $C,D$ , whereas in the conditioning algorithm, we are performing precisely the same computation over $C,D$ three times: once for each value $g$ of $G$ . In general, we can show: 

Let $\Phi$ be a set of factors, and $Y$ be a query. Let $U$ be a set of conditioning variables, and $Z=\mathcal{X}-Y-U$ . Let $\prec\,b e$ the elimination ordering over $Z$ used he variable elimination algorithm over e network $\mathcal{H}_{\Phi_{u}}$ in the onditioning algorithm. Let ≺ $\prec^{+}$ rdering that is with ≺ over the variables in Z , and where, for each variable $U\in U$ ∈ , we have that $Z\prec^{+}U$ ≺ . Then the number of operations performed by the conditi ng is no less than the number of operations performed by variable elimination with the ordering ≺ $\prec^{+}$ . 

We omit the proof of this theorem, which follows precisely the lines of our example. 

Thus, conditioning always requires no fewer computations than variable elimination with some particular ordering (which may or may not be a good one). In our example, the wasted computation from conditioning is negligible. In other cases, however, as we will discuss, we can end up with a large amount of redundant computation. In fact, in some cases, conditioning can be signiﬁcantly worse: 

![](images/fe534ffbae8d7f5e5015b2156bf131461f0a77a282c24967e6af874a2e613657.jpg) 
Figure 9.12 Networks where conditioning performs unnecessary computation 

to cut the single loop in the network. In this case, we would perform the entire elimination of the chain $A_{1}\to.\,.\,.\to A_{k-1}$ multiple times — once for every value of $A_{k}$ . 

Consider the network shown in ﬁgure 9.12b and assume that we wish to use cutset conditioning, where we cut every loop in the network. The most efcient way of doing so is to condition on every other $A_{i}$ variable, for example, $A_{2},A_{4},.\cdot\cdot,A_{k}$ (assuming for simplicity that $k$ is even). The cost of the conditioning algorithm in this case is exponential in $k$ , whereas the induced width of the network is 2 , and the cost of variable elimination is linear in $k$ . 

Given this discussion, one might wonder why anyone bothers with the conditioning algorithm. There are two main reasons. First, variable elimination gains its computational savings from caching factors computed as intermediate results. In complex networks, these factors can grow very large. In cases where memory is scarce, it might not be possible to keep these factors in memory, and the variable elimination computation becomes infeasible (or very costly due to constant thrashing to disk). On the other hand, conditioning does not require signiﬁcant amounts of memory: We run inference separately for each assignment $\mathbfit{u}$ to $U$ and simply accumulate the results. Overall, the computation requires space that is linear only in the size of the network. Thus, we can view the trade-of of conditioning versus variable elimination as a time-space trade-of. Conditioning saves space by not storing intermediate results in memory, but then it may cost additional time by having to repeat the computation to generate them. 

The second reason for using conditioning is that it forms the basis for a useful approximate inference algorithm. In particular, in certain cases, we can get a reasonable approximate solution by enumerating only some of the possible assignment $u\in V a l(U)$ . We return to this approach in section 12.5 

# 9.5.3 Graph-Theoretic Analysis 

As in the case of variable elimination, it helps to reformulate the complexity analysis of the conditioning algorithm in graph-theoretic terms. Assume that we choose to condition on a set $U$ , and perform variable elimination on the remaining variables. We can view each of these steps in terms of its efect on the graph structure. 

Let us begin with the step of conditioning the network on some variable $U$ . Once again, it is easiest to view this process in terms of its efect on an undirected graph. As we discussed, this step efectively introduces $U$ into every factor parameterizing the current graph. In graph- theoretic terms, we have introduced $U$ into every clique in the graph, or, more simply, introduced an edge between $U$ and every other node currently in the graph. 

When we ﬁnish the conditioning process, we perform elimination on the remaining variables. We have already analyzed the efect on the graph of eliminating a variable $X$ : When we eliminate $X$ , we add edges between all of the current neighbors of $X$ in the graph. We then remove $X$ from the graph. 

We can now deﬁne an induced graph for the conditioning algorithm. Unlike the graph for variable elimination, this graph has two types of ﬁll edges: those induced by conditioning steps, and those induced by the elimination steps for the remaining variables. 

Deﬁnition 9.7 conditioning induced graph 

# Example 9.7 

$\Phi$ be a set of factors over $\mathcal{X}=\{X_{1},\cdot\cdot\cdot,X_{n}\}$ $U\subset\mathcal X$ a set of conditioning variables, and $\prec\,b e$ ≺ be an elimination ering for some subset $X\subseteq\mathcal{X}-U$ ⊆X − . The induced graph $\mathcal{L}_{\Phi,-\langle,U|}$ is an undirected graph over X with the following edges: • $a$ conditioning edge between every variable $U\in U$ and every other variable $X\in{\mathcal{X}}$ ; • $a$ factor ed e between every pair of variables $X_{i},X_{j}\in X$ that both appear in some interme- diate factor ψ generated by the VE algorithm using ≺ as an elimination ordering. 

Consider the Student example of ﬁgure 9.8, where our query is $P(J)$ . Assume that (for some reason) we condition on the variable $L$ and perform elimination on the remaining variables using the ordering $C,D,I,H,G,S$ . The graph induced by this conditioning set and this elimination ordering is shown in ﬁgure 9.13, with the conditioning edges shown as dashed lines and the factor edges shown, as usual, by complete lines. The step of conditioning on $L$ causes the introduction of the edges between $L$ and all the other variables. The set of factors we have after the conditioning step immediately leads to the introduction of all the factor edges except for the edge $G{-}S_{i}$ ; this latter edge results from the elimination of $I$ . 

We can now use this graph to analyze the complexity of the conditioning algorithm. 

# Theorem 9.12 

![](images/fa7b2af60cfcabcf61bd5fd5e5bb2d44acccc45d95be4199c674e423ef6604a1.jpg) 

Figure 9.13 Induced graph for the Student example using both conditioning and elimination: we condition on $L$ and eliminate the remaining variables using the ordering $C,D,I,H,G,S$ . 

The proof is left as an exercise (exercise 9.12). 

This theorem provides another perspective on the trade-of between conditioning and elimi- nation in terms of their time complexity. Consider, as we did earlier, an algorithm that simply defers the elimination of the conditioning variables $U$ until the end. Consider the efect on the graph of the earlier steps of the elimination algorithm (those preceding the elimination of $U$ ). As variables are eliminated, certain edges might be added between the variables in $U$ and other variables (in particular, we add an dge between $X$ and $U\in U$ whenever they are both neigh- bors of some eliminated variable Y $Y$ ). However, conditioning adds edges between the variables $U$ and all other variables $X$ . Thus, conditioning always results in a graph that contains at least as many edges as the induced graph from elimination using this ordering. 

However, we can also use the same graph to precisely estimate the time-space trade-of provided by the conditioning algorithm. 

Consider an application of the co tioning algorithm to a set of factors $\Phi$ , where $U\subset\mathcal{X}$ is the ning variables, and ≺ is the elimination ordering used for the liminated variables $X\subseteq\mathcal{X}-U$ ⊆X − . The space complexity of the algorithm is $O(n\cdot v^{m_{f}})$ , where v is a bound on the domain size of any variable, and $m_{f}$ is the size of the largest clique in the graph using only factor edges. 

The proof is left as an exercise (exercise 9.13). 

By comparison, the asymptotic space complexity of variable elimination is the same as its time complexity: exponential in the size of the largest clique containing both types of edges. Thus, we see precisely that conditioning allows us to perform the computation using less space, at the cost (usually) of additional running time. 

# 9.5.4 Improved Conditioning 

As we discussed, in terms of the total operations performed, conditioning cannot be better than variable elimination. As we now show, conditioning, naively applied, can be signiﬁcantly worse. 

However, the insights gained from these examples can be used to improve the conditioning algorithm, reducing its cost signiﬁcantly in many cases. 

# 9.5.4.1 Alternating Conditioning and Elimination 

As we discussed, the main problem associated with conditioning is the fact that all computations are repeated for all values of the conditioning variables, even in cases where the diferent computations are, in fact, identical. This phenomenon arose in the network of example 9.5. 

It seems clear, in this example, that we uld prefer to eliminate the chain $A_{1}\to.\,.\,.\to A_{k-1}$ once and for all, before conditioning on $A_{k}$ . Having eliminated the chain, we would then end up with a much simpler network, involving factors only over $A_{k}$ , $B$ , $C$ , and $D$ , to which we can then apply conditioning. 

The perspective described in section 9.5.3 provides the foundation for implementing this idea. As we discussed, variable elimination works from the inside out, summing out variables in the innermost summation ﬁrst and caching the results. On the other hand, conditioning works from the outside in, performing the entire internal summation (using elimination) for each value of the conditioning variables, and only then summing the results. However, there is nothing that forces us to split our computation on the outermost summations before considering the inner ones. Speciﬁcally, we can eliminate one or more variables on the inside of the summation before conditioning on any variable on the outside. 

# Example 9.8 

Consider again the network of ﬁgure $9.l2a,$ , and assume that our goal is to compute $P(D)$ . We might formulate the expression as: 

$$
\sum_{A_{k}}\sum_{B}\sum_{C}\sum_{A_{1}}\cdot\cdot\cdot\sum_{A_{k-1}}P(A_{1},\cdot\cdot\cdot,A_{k},B,C,D).
$$ 

We can ﬁrst perform the internal summations on $A_{k-1},\ldots,A_{1}$ , resulting in a set of factors over the scope $A_{k},B,C,D$ . We can now condition this network (that is, the Markov network induced by the resulting set of factors) on $A_{k}$ , resulting in a set of simpliﬁed networks over $B,C,D$ (one for each value of $A_{k.}$ ). In each such network, we use variable elimination on $B$ and $C$ to compute $^a$ factor over $D$ , and aggregate the factors from the diferent networks, as in standard conditioning. 

In this example, we ﬁrst perform some elimination, then condition, and then elimination on the remaining network. Clearly, we can generalize this idea to deﬁne an algorithm that alternates the operations of elimination and conditioning arbitrarily. (See exercise 9.14.) 

# 9.5.4.2 Network Decomposition 

A second class of examples where we can signiﬁcantly improve the performance of condition- ing arises in networks where conditioning on some subset of variables splits the graph into independent pieces. 

# Example 9.9 

Consider the network of example 9.6, and assume that $k=16$ , and that we begin by conditioning on $A_{2}$ . After this step, the network is decomposed into two independent pieces. The standard conditioning algorithm would continue by conditioning further, say on $A_{3}$ . However, there is really no need to condition the top part of the network — the one associated with the variables $A_{1},B_{1},C_{1}$ on the variable $A_{3}$ : none of the factors mention $A_{3}$ , and we would be repeating exactly the same computation for each of its values. 

Clearly, having partitioned the network into two completely independent pieces, we can now perform the computation on each of them separately, and then combine the results. In particular, the conditioning variables used on one part would not be used at all to condition the other. More precisely, we can deﬁne an algorithm that checks, after each conditioning step, whether the resulting set of factors has been disconnected or not. If it has, it simply partitions them into two or more disjoint sets and calls the algorithm recursively on each subset. 

# 9.6 Inference with Structured CPDs $\star$ 

We have seen that BN inference exploits the network structure, in particular the conditional independence and the locality of inﬂuence. But when we discussed representation, we also allowed for the representation of ﬁner-grained structure within the CPDs. It turns out that a carefully designed inference algorithm can also exploit certain types of local CPD structure. We focus on two types of structure where this issue has been particularly well studied — independence of causal inﬂuence, and asymmetric dependencies — using each of them to illustrate a diferent type of method for exploiting local structure in variable elimination. We defer the discussion of inference in networks involving continuous variables to chapter 14. 

# 9.6.1 Independence of Causal Inﬂuence 

The earliest and simplest instance of exploiting local structure was for CPDs that exhibit inde- pendence of causal inﬂuence, such as noisy-or. 

# 9.6.1.1 Noisy-Or Decompositions 

Consider a simple network consisting of a binary variable $Y$ and its four binary parents $X_{1},X_{2},X_{3},X_{4}$ , where the CPD of $Y$ is a noisy-or. Our goal is to compute the probability of $Y$ . The operations required to execute this process, assuming we use an optimal ordering, is: 

• 4 multiplications for $P(X_{1})\cdot P(X_{2})

$ • 8 multiplications for $P(X_{1},X_{2})\cdot P(X_{3})

$ • 16 multiplications for $P(X_{1},X_{2},X_{3})\cdot P(X_{4})

$ • 32 multiplications for P $P(X_{1},X_{2},X_{3},X_{4})\cdot P(Y\mid X_{1},X_{2},X_{3},X_{4})$ 

The total is 60 multiplications, plus another 30 additions to sum out $X_{1},\dots,X_{4}$ , in order to reduce the resulting factor $P(X_{1},X_{2},X_{3},X_{4},Y)$ , of size 32, into the factor $P(Y)$ of size 2. 

However, we can exploit the structure of the CPD to substantially reduce the amount of computation. As we discussed in section 5.4.1, a noisy-or variable can be decomposed into a de- terministic OR of independent noise variables, resulting in the subnetwork shown in ﬁgure $9.14\mathrm{a}$ . This transformation, by itself, is not very helpful. The factor $P(Y\mid Z_{1},Z_{2},Z_{3},Z_{4})$ is still of size 32 if we represent it as a full factor, so we achieve no gains. 

The key idea is that the deterministic OR variable can be decomposed into various cascades of deterministic OR variables, each with a very small indegree. Figure $9.14\mathrm{b}$ shows a simple 

![](images/9803ea1ef758e8cb0e8659467d11eb294a68518011cc517ca2308a6ec2f2191f.jpg) 
Figure 9.14 Diferent decompositions for a noisy-or CPD: (a) The standard decomposition of a noisy- or. (b) A tree decomposition of the deterministic-or. (c) A tree-based decomposition of the noisy-or. (d) A chain-based decomposition of the noisy-or. 

decomposition of the deterministic OR as a tree. We can simplify this construction by eliminating the intermediate variables $Z_{i}$ , integrating the “noise” for each $X_{i}$ into the appropriate $O_{i}$ . In particular, $O_{1}$ would be the noisy-or of $X_{1}$ and $X_{2}$ , with the original noise parameters and a leak parameter of 0 . The resulting construction is shown in ﬁgure $9.14\mathrm{c}$ . 

We can now revisit the inference task in this apparently more complex network. An optimal ordering for variable elimination is $X_{1},X_{2},X_{3},X_{4},O_{1},O_{2}$ . The cost of performing elimination of $X_{1},X_{2}$ is: 

• 8 multiplications for $\psi_{1}(X_{1},X_{2},O_{1})=P(X_{1})\cdot P(O_{1}\mid X_{1},X_{2})

$ • 4 additions to sum o $X_{1}$ $\begin{array}{r}{\tau_{1}(X_{2},O_{1})=\sum_{X_{1}}\psi_{1}(X_{1},X_{2},O_{1})}\end{array}

$ • 4 multiplications for $\psi_{2}(X_{2},O_{1})=\tau_{1}(X_{2},O_{1})\cdot P(X_{2})$ ·

 • 2 additions for $\begin{array}{r}{\tau_{2}(O_{1})=\sum_{X_{2}}\psi_{2}(X_{2},O_{1})}\end{array}$ 

The cost for eliminating $X_{3},X_{4}$ is identical, as is the cost for subsequently eliminating $O_{1},O_{2}$ . Thus, the total number of operations is $3\cdot(8+4)=36$ multiplications and $3\cdot(4+2)=18$ additions. 

A diferent decomposition of the OR variable is as a simple cascade, where each $Z_{i}$ is consec- utively OR’ed with the previous intermediate result. This decomposition leads to the construction of ﬁgure $9.14\mathrm{d}$ . For this construction, an optimal elimination ordering is $X_{1},O_{1},X_{2},O_{2},X_{3},O_{3},X_{4}$ A simple analysis shows that it takes 4 multiplications and 2 additions to eliminate each of $X_{1},\dots,X_{4}$ , and 8 multiplications and 4 additions to eliminate each of $O_{1},O_{2},O_{3}$ . The total cost is $4\cdot4+3\cdot8=40$ multiplications and $4\cdot2+3\cdot4=20$ additions. 

# 9.6.1.2 The General Decomposition 

Clearly, the construction used in the preceding example is a general one that can be applied to more complex networks and other types of CPDs that have independence of causal inﬂuence. We take a variable whose CPD has independence of causal inﬂuence, and generate its decomposition into a set of independent noise models and a deterministic function, as in ﬁgure 5.13. 

We then cascade the computation of the deterministic function into a set of smaller steps. Given our assumption about the symmetry and associativity of the deterministic function in the deﬁnition of symmetric ICI (deﬁnition 5.13), any decomposition of the deterministic function results in the same answer. Speciﬁcally, consider a variable $Y$ with parents $X_{1},\ldots,X_{k}$ , whose deﬁnition 5.13. We can decompose $Y$ by introducing $k-1$ intermediate variables $O_{1},.\ldots,O_{k-1}$ , such that: 

• the variable $Z$ , and each of the $O_{i}$ ’s, has exactly two parents in $Z_{1},.\,.\,.\,,Z_{k},O_{1},.\,.\,.\,,O_{i-1}$ ;

 • the CPD of $Z$ and of $O_{i}$ is the deterministic $\diamondsuit$ of its two parents;

 • each $Z_{l}$ and each $O_{i}$ is a parent of at most one variable in $O_{1},.\,.\,.\,,O_{k-1},Z$ . 

These conditions ensure that $Z=Z_{1}\!\diamond Z_{2}\!\diamond...\diamond Z_{k}$ , but that this function is computed gradually, where the node corresponding to each intermediate result has an indegree of 2. 

We note that we can save some extraneous nodes, as in our example, by aggregating the noisy dependence of $Z_{i}$ on $X_{i}$ into the CPD where $Z_{i}$ is used. 

After executing this decomposition for every ICI variable in the network, we can simply apply variable elimination to the decomposed network with the smaller factors. As we saw, the complexity of the inference can go down substantially if we have smaller CPDs and thereby smaller factors. 

We note that the sizes of the intermediate factors depend not only on the number of variables in their scope, but also on the domains of these variables. For the case of noisy-or variables (as well as noisy-max, noisy-and, and so on), the domain size of these variables is ﬁxed and fairly small. However, in other cases, the domain might be quite large. In particular, in the case of generalized linear models, the domain of the intermediate variable $Z$ generally grows linearly with the number of parents. 

Example 9.10 Consider a variable $Y$ with $\mathrm{Pa}_{Y}=\{X_{1},.\,.\,.\,,X_{k}\}$ , where each $X_{i}$ is binary. Assu at Y ’s CPD is a generalized linear model, whose parameters are $w_{0}=0$ and $w_{i}=w$ for all $i>1$ . Then the domain of the intermediate variable $Z$ is $\{0,1,\ldots,k\}$ . In this case, th decomposition provides $^a$ trade-of: The size of the original CPD for $P(Y\mid X_{1},.\,.\,,X_{k})$ grows as $2^{k}$ ; the size of the factors in the decomposed network grow roughly as k $k^{3}$ . In diferent situations, one approach might be better than the other. 

Thus, the decomposition of symmetric ICI variables might not always be beneﬁcial. 

# 9.6.1.3 Global Structure 

Our decomposition of the function $f$ that deﬁnes the variable $Z$ can be done in many ways, all of which are equivalent in terms of their ﬁnal result. However, they are not equivalent from the perspective of computational cost. Even in our simple example, we saw that one decomposition can result in fewer operations than the other. The situation is signiﬁcantly more complicated when we take into consideration other dependencies in the network. 

Example 9.11 Consider the network of ﬁgure $9.14c,$ and assume that $X_{1}$ and $X_{2}$ have a joint parent $A$ . In this case, we eliminate $A$ ﬁrst, and end up with a factor over $X_{1},X_{2}$ . Aside from the $4+8\,=$ 12 multiplications and 4 additions required to compute this factor $\tau_{0}(X_{1},X_{2})$ , it now takes 8 multiplications to com te $\psi_{1}(X_{1},X_{2},O_{1})\,=\,\tau_{0}(X_{1},X_{2})\,\cdot\,P(O_{1}\mid X_{1},X_{2})\nonumber$ , and $4+2\,=\,6$ additions to sum out $X_{1}$ and $X_{2}$ in $\psi_{1}$ . The rest of the computation remains unchanged. Thus, the total number of operations required to eliminate all of $X_{1},\dots,X_{4}$ (after the elimination of $A$ ) is $8+12=20$ multiplications and $6+6=12$ additions. 

Conversely, assume that $X_{1}$ and $X_{3}$ have the joint parent $A$ . In this case, it still requires 12 multiplications and 4 additions to compute a factor $\tau_{0}(X_{1},X_{3})$ , but the remaining operations become signiﬁcantly more complex. In particular, it takes: 

• 8 multiplications for $\psi_{1}(X_{1},X_{2},X_{3})=\tau_{0}(X_{1},X_{3})\cdot{\cal P}(X_{2})$ • 16 multiplications for $\cdot\,\psi_{2}(X_{1},X_{2},X_{3},O_{1})=\psi_{1}(X_{1},X_{2},X_{3})\cdot P(O_{1}\mid X_{1},X_{2})$ ) • 8 additions fo $\begin{array}{r}{\tau_{2}(X_{3},O_{1})=\sum_{X_{1},X_{2}}\psi_{2}(X_{1},X_{2},X_{3},O_{1})}\end{array}$ 

The same number of operations is required to eliminate $X_{3}$ and $X_{4}$ . (Once these steps are completed, we can eliminate $O_{1},O_{2}$ as usual.) Thus, the total number of operations required to eliminate all of $X_{1},\dots,X_{4}$ (after the elimination of $A$ ) is $2\cdot(8+16)=48$ multiplications and $2\cdot8=16$ additions, considerably more than our previous case. 

Clearly, in the second network structure, had we done the decomposition of the noisy-or variable so as to make $X_{1}$ and $X_{3}$ parents of $O_{1}$ (and $X_{2},X_{4}$ parents of $O_{2}.$ ), we would get the same cost as we did in the ﬁrst case. However, in order to do that, we need to take into consideration the global structure of the network, and even the order in which other variables are eliminated, at the same time that we are determining how to decompose a particular variable with symmetric ICI. In particular, we should determine the structure of the decomposition at the same time that we are considering the elimination ordering for the network as a whole. 

# 9.6.1.4 Heterogeneous Factorization 

An alternative approach that achieves this goal uses a diferent factorization for a network — one that factorizes the joint distribution for the network into CPDs, as well as the CPDs of symmetric ICI variables into smaller components. This factorization is heterogeneous , in that some factors must be combined by product, whereas others need to be combined using the type of operation that corresponds to the symmetric ICI function in the corresponding CPD. One can then deﬁne a heterogeneous variable elimination algorithm that combines factors, using whichever operation is appropriate, and that eliminates variables. Using this construction, we can determine a global ordering for the operations that determines the order in which both local 

![](images/c2e513144c284e547fc260b8541b58ba035cec3923106790dc98c3f11d7fa58d.jpg) 
Figure 9.15 A Bayesian network with rule-based structure: (a) the network structure; (b) the CPD for the variable $D$ . 

factors and global factors are combined. Thus, in efect, the algorithm determines the order in which the components of an ICI CPD are “recombined” in a way that takes into consideration the structure of the factors created in a variable elimination algorithm. 

# 9.6.2Context-Speciﬁc Independence 

A second important type of local CPD structure is the context-speciﬁc independence, typically encoded in a CPD as trees or rules. As in the case of ICI, there are two main ways of exploiting this type of structure in the context of a variable elimination algorithm. One approach (exercise 9.15) uses a decomposition of the CPD, which is performed as a preprocessing step on the network structure; standard variable elimination can then be performed on the modiﬁed network. The second approach, which we now describe, modiﬁes the variable elimination algorithm itself to conduct its basic operations on structured factors. We can also exploit this structure within the context of a conditioning algorithm. 

# 9.6.2.1 Rule-Based Variable Elimination 

An alternative approach is to introduce the structure directly into the factors used in the variable elimination algorithm, allowing it to take advantage of the ﬁner-grained structure. It turns out that this approach is easier to understand and implement for CPDs and factors represented as rules, and hence we present the algorithm in this context. 

As speciﬁed in section 5.3.1.2, a rule-based CPD is described as a set of mutually exclusive and exhaustive rules, where each rule $\rho$ has the form $\langle c;p\rangle$ . As we already discussed, a tree-CPD and a tabular CPD can each be converted into a set of rules in the obvious way. 

# Example 9.12 

set of rules: 

$$
\left\{\begin{array}{c c}{{\rho_{1}}}&{{\langle b^{0},d^{0};1-q_{1}\rangle}}\\ {{\rho_{2}}}&{{\langle b^{0},d^{1};q_{1}\rangle}}\\ {{}}&{{}}\\ {{\rho_{3}}}&{{\langle a^{0},b^{1},d^{0};1-q_{2}\rangle}}\\ {{\rho_{4}}}&{{\langle a^{0},b^{1},d^{1};q_{2}\rangle}}\\ {{\rho_{5}}}&{{\langle a^{1},b^{1},d^{0};1-q_{3}\rangle}}\\ {{\rho_{6}}}&{{\langle a^{1},b^{1},d^{0};q_{3}\rangle}}\end{array}\right\}
$$ 

Assume that the CPD $P(E\mid A,B,C,D)$ is also associated with a set of rules. Our discussion will focus on rules involving the variable D , so we show only that part of the rule set: 

$$
\left\{\begin{array}{c c}{\rho_{7}}&{\langle a^{0},d^{0},e^{0};1-p_{1}\rangle}\\ {\rho_{8}}&{\langle a^{0},d^{0},e^{1};p_{1}\rangle}\\ {\rho_{9}}&{\langle a^{0},d^{1},e^{0};1-p_{2}\rangle}\\ {\rho_{10}}&{\langle a^{0},d^{1},e^{1};p_{2}\rangle}\\ {\rho_{11}}&{\langle a^{1},b^{0},c^{1},d^{0},e^{0};1-p_{4}\rangle}\\ {\rho_{12}}&{\langle a^{1},b^{0},c^{1},d^{0},e^{1};p_{4}\rangle}\\ {\rho_{13}}&{\langle a^{1},b^{0},c^{1},d^{1},e^{0};1-p_{5}\rangle}\\ {\rho_{14}}&{\langle a^{1},b^{0},c^{1},d^{1},e^{1};p_{5}\rangle}\end{array}\right\}
$$ 

Using this type of process, the entire distribution can be factorized into a multiset of rules ${\mathcal{R}},$ which is the union of all of the rules associated with he CPDs of the diferent v ables in the network. Then, the probability of any instantiation ξ to the network variables X can be computed as 

$$
P(\xi)=\prod_{\langle c;p\rangle\in\mathcal{R},\xi\sim c}p,
$$ 

where we recall that $\xi\sim c$ holds if the assignments $\xi$ and $^c$ are compatible, in that they assign the same values to those variables that are assigned values in both. 

Thus, as for the tabular CPDs, the distribution is deﬁned in terms of a product of smaller components. In this case, however, we have broken up the tables into their component rows. This deﬁnition immediately suggests that we can use similar ideas to those used in the table- based variable elimination algorithm. In particular, we can multiply rules with each other and sum out a variable by adding up rules that give diferent values to the variables but are the same otherwise. 

In general, we deﬁne the following two key operations: 

Deﬁnition 9.8 rule product 

Deﬁnition 9.9 rule sum 

This deﬁnition is signiﬁcantly more restricted than the product of tabular factors, since it requires that the two rules have precisely the same context. We return to this issue in a moment. 

After this operation, $Y$ is summed out in the context $^c$ . 

Both of these operations can only be applied in very restricted settings, that is, to sets of rules that satisfy certain stringent conditions. In order to make our set of rules amenable to the application of these operations, we might need to reﬁne some of our rules. We therefore deﬁne the following ﬁnal operation: 

Deﬁnition 9.10 rule split 

Let $\rho=\langle c;p\rangle$ be a rule, and let $Y$ be a variable. We deﬁne the rule split $S p l i t(\rho\angle Y)$ as follows: If $Y\in S c o p e[c].$ , then $S p l i t(\rho\angle Y)=\{\rho\}.$ ; otherwise, 

$$
S p l i t(\rho\angle Y)=\{\langle c,Y=y;p\rangle\ :\ y\in V a l(Y)\}.
$$ 

In general, the pu ose of rule splitti is to make the context of one rule $\rho=\langle c;p\rangle$ compatible with the context c $c^{\prime}$ of another rule $\rho^{\prime}$ . Naively, we might take all the variables in $S c o p e[c^{\prime}]\mathrm{~-~}$ Scope [ c ] and split $\rho$ recursively on each one of them. However, this process creates unnecessarily many rules. 

Example 9.13 Consider $\rho_{2}$ and $\rho_{14}$ in example 9.12, and assume we want to multiply them together. To do so, we need to split $\rho_{2}$ in order to produce a rule with an identical context. If we naively split $\rho_{2}$ on all three variables $A,C,E$ that appear in $\rho_{14}$ and not in $\rho_{2}$ , the result would be eight rules of the form: $\langle a,b^{0},c,d^{1},e;q_{1}\rangle$ , one for each combination of values $a,c,e$ . However, the only rule we really need in order to perform the rule produ operation is $\langle a^{1},b^{0},c^{1},d^{1},e^{1};q_{1}\rangle$ . 

Intuitively, having split $\rho_{2}$ on the variable A , it is wasteful to continue splitting the rule whose context is $a^{0}$ , since this rule (and any derived from it) will not participate in the desired rule product operation with $\rho_{14}$ . Thus, a more parsimonious split of $\rho_{14}$ that still generates this last rule is: 

$$
\left\{\begin{array}{l}{\langle a^{0},b^{0},d^{1};q_{1}\rangle}\\ {\langle a^{1},b^{0},c^{0},d^{1};q_{1}\rangle}\\ {\langle a^{1},b^{0},c^{1},d^{1},e^{0};q_{1}\rangle}\\ {\langle a^{1},b^{0},c^{1},d^{1},e^{1};q_{1}\rangle}\end{array}\right\}
$$ 

This new rule set is still a mutually exclusive and exhaustive partition of the space originally covered by $\rho_{2}$ , but contains only four rules rather than eight. 

In general, we can construct these more parsimonious splits using the recursive procedure shown in algorithm 9.6. This procedure gives precisely the desired result shown in the example. Rule splitting gives us the tool to take a set of rules and reﬁne them, allowing us to apply either the rule-product operation or the rule-sum operation. The elimination algorithm is shown in algorithm 9.7. Note that the ﬁgure only shows the procedure for eliminating a single variable $Y$ . The outer loop, which iteratively eliminates nonquery variables one at a time, is precisely the same as the Sum-Product-VE procedure in algorithm 9.1, except that it takes as input a set of rule factors rather than table factors. 

To understand the operation of the algorithm more concretely, consider the following example: 

# Example 9.14 

![](images/a37f7ee6a7bcd7687cdc05ac7e6cc56453d9c13f161dc249f38c374d1ef6bce4.jpg) 

The rules $\rho_{3}$ on the one hand, and $\rho_{7},\rho_{8}$ on the other, have compatible contexts, so we can choose to combine them. We begin by splitting $\rho_{3}$ and $\rho_{7}$ on each other’s context, which results in: 

$$
\left\{\begin{array}{l l}{\rho_{15}}&{\left\langle a^{0},b^{1},d^{0},e^{0};1-q_{2}\right\rangle}\\ {\rho_{16}}&{\left\langle a^{0},b^{1},d^{0},e^{1};1-q_{2}\right\rangle}\\ {}&{}\\ {\rho_{17}}&{\left\langle a^{0},b^{0},d^{0},e^{0};1-p_{1}\right\rangle}\\ {\rho_{18}}&{\left\langle a^{0},b^{1},d^{0},e^{0};1-p_{1}\right\rangle}\end{array}\right\}
$$ 

The contexts of $\rho_{15}$ and $\rho18$ match, so we can now apply rule product, replacing the pair by: 

$$
\left\{\begin{array}{c c}{{\rho_{19}}}&{{\langle a^{0},b^{1},d^{0},e^{0};(1-q_{2})(1-p_{1})\rangle}}\end{array}\right\}
$$ 

We can now split $\rho_{8}$ using the context of $\rho_{16}$ and multiply the matching rules together, obtaining 

$$
\left\{\begin{array}{c c}{{\rho_{20}}}&{{\langle a^{0},b^{0},d^{0},e^{1};p_{1}\rangle}}\\ {{\rho_{21}}}&{{\langle a^{0},b^{1},d^{0},e^{1};(1-q_{2})p_{1}\rangle}}\end{array}\right\}.
$$ 

The resulting rule set contains $\rho_{17},\rho_{19},\rho_{20},\rho_{21}$ in place of $\rho_{3},\rho_{7},\rho_{8}$ . 

We can apply a similar process to $\rho_{4}$ and $\rho_{9},\rho_{10}$ , which leads to their substitution by the rule set: 

$$
\left\{\begin{array}{c c}{\rho_{22}}&{\langle a^{0},b^{0},d^{1},e^{0};1-p_{2}\rangle}\\ {\rho_{23}}&{\langle a^{0},b^{1},d^{1},e^{0};q_{2}(1-p_{2})\rangle}\\ {\rho_{24}}&{\langle a^{0},b^{0},d^{1},e^{1};p_{2}\rangle}\\ {\rho_{25}}&{\langle a^{0},b^{1},d^{1},e^{1};q_{2}p_{2}\rangle}\end{array}\right\}.
$$ 

We can now eliminate $D$ in the context $a^{0},b^{1},e^{1}$ . only rules in $\mathcal{R}^{+}$ compatible with this context are $\rho_{21}$ and $\rho_{25}$ . We extract them fro $\mathcal{R}^{+}$ R and sum them; the resu ng rule $\langle a^{0},b^{1},e^{1};(1-q_{2})p_{1}+q_{2}p_{2}\rangle$ , is then inserted into R $\mathcal{R}^{-}$ . We can similarly eliminate D in the context $a^{0},b^{1},e^{0}$ . 

The process continues, with rules being split and multiplied. When $D$ has been eliminated in a set of mutually exclusive and exhaustive contexts, then we have exhausted all rules involving $D$ ; at this point, $\mathcal{R}^{+}$ is empty, and the process of eliminating $D$ terminates. 

![](images/4a2092097f97884ecb799b4eaa218ebaf41f0ebd052bdfe32c9bf039565165e2.jpg) 

A diferent way of understanding the algorithm is to consider its application to rule sets that originate from standard table-CPDs. It is not difcult to verify that the algorithm performs exactly the same set of operations as standard variable elimination. For example, the standard operation of factor product is simply the application of rule splitting on all of the rules that constitute the two tables, followed by a sequence of rule product operations on the resulting rule pairs. (See exercise 9.16.) 

To prove that the algorithm computes the correct result, we need to show that each operation performed in the context of the algorithm maintains a ce n correctness invariant. Let $\mathcal{R}$ be the current set of rules maintained by the algorithm, and W be the variables that have not yet been eliminated. Each operation must maintain the following condition: 

![](images/7b9b0b9c031c41e81c5756fcb425611eb807f37aab18593150e9f5b3a18b4ebd.jpg) 
Figure 9.16 Conditioning a Bayesian network whose CPDs have CSI: (a) conditioning on $a^{0}$ ; (b) conditioning on $a^{1}$ . 

The probability of a context $^c$ such that $S c o p e[c]\subseteq W$ can be obtained by multiplying all rules $\langle c^{\prime};p\rangle\in{\mathcal{R}}$ whose context is compatible with $^c$ . 

It is not difcult to show that the invariant holds initially, and that each step in the algorithm maintains it. Thus, the algorithm as a whole is correct. 

# 9.6.2.2 Conditioning 

We can also use other techniques for exploiting CSI in inference. In particular, we can generalize the notion of conditioning to this setting n an interesting way. Consider a network $\mathcal{B}$ , and assume that we condition it on a variable U . So far, we have assumed that the structure of the diferent conditioned networks, for the diferent values $u$ of $U$ , is the same. When the CPDs are tables, with no extra structure, this assumption generally holds. However, when the CPDs have CSI, we might be able to utilize the additional structure to simplify the conditioned networks considerably. 

Consider the network shown in ﬁgure 9.15, as described in example 9.12. Assume we condition this network on the variable $A$ . If we condition on $a^{0}$ , we see that the reduced CPD for $E$ no longer depends on $C$ . Thus, the conditioned Markov network for this set of factors is the one shown in ﬁgure 9.16a. By contrast, when we condition on $a^{1}$ , the reduced factors do not “lose” any variables aside from $A$ , and we obtain the conditioned Markov network shown in ﬁgure 9.16b. Note that the network in ﬁgure 9.16a is so simple that there is no point performing any further conditioning on it. Thus, we can continue the conditioning process for only one of the two branches of the computation — the one corresponding to $a^{1}$ . 

In general, we can extend the conditioning algorithm of section 9.5 to account for CSI in the CPDs or in the factors of a Markov network. Consider a single conditioning step on a variable $U$ . As we enumerate the diferent possible values $u$ of $U$ , we generate a possibly diferent conditioned network for each one. Depending on the structure of this network, we select which step to take next in the context of this particular network. In diferent networks, we might choose a diferent variable to use for the next conditioning step, or we might decide to stop the conditioning process for some networks altogether. 

We have presented two approaches to variable elimination in the case of local structure in the CPDs: preprocessing followed by standard variable elimination, and specialized variable elimination algorithms that use a factorization of the structured CPD. These approaches ofer diferent trade-ofs. On the one hand, the specialized variable elimination approach reveals more of the structure of the CPDs to the inference algorithm, allowing the algorithm more ﬂexibility in exploiting this structure. Thus, this approach can achieve lower computational cost than any ﬁxed decomposition scheme (see box 9.D). By comparison, the preprocessing approach embeds some of the structure within deterministic CPDs, a structure that most variable elimination algorithms do not fully exploit. 

On the other hand, specialized variable elimination schemes such as those for rules require the use of special-purpose variable elimination algorithms rather than of-the-shelf packages. Furthermore, the data structures for tables are signiﬁcantly more efcient than those for other types of factors such as rules. Although this diference seems to be an implementation issue, it turns out to be quite signiﬁcant in practice. One can somewhat address this limitation by the use of more sophisticated algorithms that exploit efcient table-based operations whenever possible (see exercise 9.18). 

Although the trade-ofs between these two approaches is not always clear, it is generally the case that, in networks with signiﬁcant amounts of local structure, it is valuable to design an inference scheme that exploits this structure for increased computational efciency. 

Box 9.D — Case Study: Inference with Local Structure. A natural question is the extent to which local structure can actually help speed up inference. 

In one experimental comparison by Zhang and Poole (1996), four algorithms were applied to frag- ments of the CPCS network (see box 5.D): standard variable elimination (with table representation of factors), the two decompositions illustrated in ﬁgure 9.14 for the case of noisy-or, and a special- purpose elimination algorithm that uses a heterogeneous factorization. The results show that in a network such as CPCS, which uses predominantly noisy-or and noisy-max CPDs, signiﬁcant gains in performance can be obtained. They results also showed that the two decomposition schemes (tree-based and chain-based) are largely equivalent in their performance, and the heterogeneous factorization outperforms both of them, due to its greater ﬂexibility in dynamically determining the elimination ordering during the course of the algorithm. 

For rule-based variable elimination, no large networks with extensive rule-based structure had been constructed. So, Poole and Zhang (2003) used a standard benchmark network, with 32 variables and 11,018 entries. Entries that were within 0 . 05 of each other were collaped, to construct a more compact rule-based representation, with a total of 5,834 distinct entries. As expected, there are a large number of cases where the use of rule-based inference provided signiﬁcant savings. However, there were also many cases where contextual independence does not provide signiﬁcant help, in which case the increased overhead of the rule-based inference dominates, and standard VE performs better. 

At a high level, the main conclusion is that table-based approaches are amenable to numerous optimizations, such as those described in box 10.A, which can improve the performance by an order of magnitude or even more. Such optimizations are harder to deﬁne for more complex data structures. Thus, it is only useful to consider algorithms that exploit local structure either when it is extensively present in the model, or when it has speciﬁc structure that can, itself, be exploited using specialized algorithms. 

# 9.7 Summary and Discussion 

In this chapter, we described the basic algorithms for exact inference in graphical models. As we saw, probability queries essentially require that we sum out an exponentially large joint distribution. The fundamental idea that allows us to avoid the exponential blowup in this task is the use of dynamic programming, where we perform the summation of the joint distribution from the inside out rather than from the outside in, and cache the intermediate results, thereby avoiding repeated computation. 

We presented an algorithm based on this insight, called variable elimination. The algorithm works using two fundamental operations over factors — multiplying factors and summing out variables in factors. We analyzed the computational complexity of this algorithm using the structural properties of the graph, showing that the key computational metric was the induced width of the graph. 

We also presented another algorithm, called conditioning, which performs some of the sum- mation operations from the outside in rather than from the inside out, and then uses variable elimination for the rest of the computation. Although the conditioning algorithm is never less expensive than variable elimination in terms of running time, it requires less storage space and hence provides a time-space trade-of for variable elimination. 

We showed that both variable elimination and conditioning can take advantage of local structure within the CPDs. Speciﬁcally, we presented methods for making use of CPDs with independence of causal inﬂuence, and of CPDs with context-speciﬁc independence. In both cases, techniques tend to fall into two categories: In one class of methods, we modify the network structure, adding auxiliary variables that reveal some of the structure inside the CPD and break up large factors. In the other, we modify the variable elimination algorithm directly to use structured factors rather than tables. 

Although exact inference is tractable for surprisingly many real-world graphical models, it is still limited by its worst-case exponential performance. There are many models that are simply too complex for exact inference. As one example, consider the $n\times n$ grid-structured pairwise Markov networks of box 4.A. It is not difcult to show that the minimal tree-width of this network is $n$ . Because these networks are often used to model pixels in an image, where $n\,=\,1,000$ is quite common, it is clear that exact inference is intractable for such networks. Another example is the family of networks that we obtain from the template model of example 6.11. Here, the moralized network, given the evidence, is a fully connected bipartite graph; if we have $n$ variables on one side and $m$ on the other, the minimal tree-width is $\operatorname*{min}(n,m)$ , which can be very large for many practical models. Although this example is obviously a toy domain, examples of similar structure arise often in practice. In later chapters, we will see many other examples where exact inference fails to scale up. Therefore, in chapter 11 and chapter 12 we discuss approximate inference methods that trade of the accuracy of the results for the ability to scale up to much larger models. 

One class of networks that poses great challenges to inference is the class of networks induced by template-based representations. These languages allow us to specify (or learn) very small, compact models, yet use them to construct arbitrarily large, and often densely connected, networks. Chapter 15 discusses some of the techniques that have been used to deal with dynamic Bayesian networks. 

Our focus in this chapter has been on inference in networks involving only discrete variables. The introduction of continuous variables into the network also adds a signiﬁcant challenge. Although the ideas that we described here are instrumental in constructing algorithms for this richer class of models, many additional ideas are required. We discuss the problems and the solutions in chapter 14. 

# 9.8 Relevant Literature 

The ﬁrst formal analysis of the computational complexity of probabilistic inference in Bayesian networks is due to Cooper (1990). 

peeling forward-backward algorithm 

nonserial dynamic programming 

Variants of the variable elimination algorithm were invented independently in multiple com- munities. One early variant is the peeling algorithm of Cannings et al. (1976, 1978), formulated for the analysis of genetic pedigrees. Another early variant is the forward-backward algorithm , which performs inference in hidden Markov models (Rabiner and Juang 1986). An even earlier variant of this algorithm was proposed as early as 1880, in the context of continuous models (Thiele 1880). Interestingly, the ﬁrst variable elimination algorithm for fully general models was invented as early as 1972 by Bertelé and Brioschi (1972), under the name nonserial dynamic programming . However, they did not present the algorithm in the setting of probabilistic inference in graph- structured models, and therefore it was many years before the connection to their work was recognized. Other early work with similar ideas but a very diferent application was done in the database community (Beeri et al. 1983). 

The general problem of probabilistic inference in graphical models was ﬁrst tackled by Kim and Pearl (1983), who proposed a local message passing algorithm in polytree-structured Bayesian networks. These ideas motivated the development of a wide variety of more general algorithms. One such trajectory includes the clique tree methods that we discuss at length in the next chapter (see also section 10.6). A second includes a specrum of other methods (for example, Shachter 1988; Shachter et al. 1990), culminating in the variable elimination algorithm, as presented here, ﬁrst described by Zhang and Poole (1994) and subsequently by Dechter (1999). Huang and Darwiche (1996) provide some useful tips on an efcient implementation of algorithms of this type. 

Dechter (1999) presents interesting connections between these algorithms and constraint- satisfaction algorithms, connections that have led to fruitful work in both communities. Other generalizations of the algorithm to settings other than pure probabilistic inference were described by Shenoy and Shafer (1990); Shafer and Shenoy (1990) and by Dawid (1992). The construction of the network polynomial was proposed by Darwiche (2003). 

The complexity analysis of the variable elimination algorithm is described by Bertelé and Brioschi (1972); Dechter (1999). The analysis is based on core concepts in graph theory that have been the subject of extensive theoretical analysis; see Golumbic (1980); Tarjan and Yannakakis (1984); Arnborg (1985) for an introduction to some of the key concepts and algorithms. 

Much work has been done on the problem of ﬁnding low-tree-width triangulations or (equiv- alently) elimination orderings. One of the earliest algorithms is the maximum cardinality search of Tarjan and Yannakakis (1984). Arnborg, Corneil, and Proskurowski (1987) show that the prob- lem of ﬁnding the minimal tree-width elimination ordering is $\mathcal{N P}$ -hard. Shoikhet and Geiger (1997) describe a relatively efcient algorithm for ﬁnding this optimal elimination ordering — one whose cost is approximately the same as the cost of inference with the resulting ordering. Becker and Geiger (2001) present an algorithm that ﬁnds a close-to-optimal ordering. Neverthe- less, most implementations use one of the standard heuristics. A good survey of these heuristic methods is presented by Kjærulf (1990), who also provides an extensive empirical comparison. Fishelson and Geiger (2003) suggest the use of stochastic search as a heuristic and provide another set of comprehensive experimental comparisons, focusing on the problem of genetic linkage analysis. Bodlaender, Koster, van den Eijkhof, and van der Gaag (2001) provide a series of simple preprocessing steps that can greatly reduce the cost of triangulation. 

The ﬁrst incarnation of the conditioning algorithm was presented by Pearl (1986a), in the context of cutset conditioning, where the conditioning variables cut all loops in the network, forming a polytree. Becker and Geiger (1994); Becker, Bar-Yehuda, and Geiger (1999) present a va- riety of algorithms for ﬁnding a small loop cutset. The general algorithm, under the name global conditioning , was presented by Shachter et al. (1994). They also demonstrated the equivalence of conditioning and variable elimination (or rather, the clique tree algorithm) in terms of the under- lying computations, and pointed out the time-space trade-ofs between these two approaches. These time-space trade-ofs were then placed in a comprehensive computational framework in the recursive conditioning method of Darwiche (2001b); Allen and Darwiche (2003a,b). Cutset algorithms have made a signiﬁcant impact on the application of genetic linkage analysis Schäfer (1996); Becker et al. (1998), which is particularly well suited to this type of method. 

The two noisy-or decomposition methods were described by Olesen, Kjærulf, Jensen, Falck, Andreassen, and Andersen (1989) and Heckerman and Breese (1996). An alternative approach that utilizes a heterogeneous factorization was described by Zhang and Poole (1996); this approach is more ﬂexible, but requires the use of a special-purpose inference algorithm. For the case of CPDs with context-speciﬁc independence, the decomposition approach was proposed by Boutilier, Friedman, Goldszmidt, and Koller (1996). The rule-based variable elimination algorithm was proposed by Poole and Zhang (2003). The trade-ofs here are similar to the case of the noisy-or methods. 

# 9.9 Exercises 

Exercise $\mathbf{9.1}\star$ 

Prove theorem 9.2. 

Exercise ${\bf9.2\star}$ 

Consider a factor produced as a product of some of the CPDs in a Bayesian network $\mathcal{B}$ : 

$$
\tau(W)=\prod_{i=1}^{k}P(Y_{i}\mid\mathrm{Pa}_{Y_{i}})
$$ 

where $W=\cup_{i=1}^{k}\big(\{Y_{i}\}\cup\mathrm{Pa}_{Y_{i}}\big)$ { } ∪ . 

a. Show that $\tau$ is a conditional probability in some network. More precisely, construct another Bayesian network $\mathcal{B}^{\prime}$ and a disjoint partition $W=Y\cup Z$ such that $\tau(\pmb{W})=\dot{P_{\mathcal{B}^{\prime}}}(Y\mid Z)$ . 

b. Conclude that all of the intermediate factors produced by the variable elimination algorithm are also conditional probabilities in some network. 

# Exercise 9.3 

Consider a modiﬁed variable elimination algorithm that is allowed to multiply all of the entries in a single factor by some arbitrary constant. (For example, it may choose to renormalize a factor to sum to 1.) If we run this algorithm on the factors resulting from a Bayesian network with evidence, which types of queries can we still obtain the right answer to, and which not? 

# Exercise $9.4\star$ 

This exercise shows basic properties of the network polynomial and its derivatives: 

a. Prove equation (9.8). b. Prove equation (9.9). 

evidence retraction c. Let $\pmb{Y}\,=\,\pmb{y}$ e assignment. For $Y_{i}\,\in\,Y$ , we now consider hat happens if we retract e observation $Y_{i}\,=\,y_{i}$ . More precisely, let ${\pmb{y}}_{-i}$ be the assignment in y to all variables other than Y $Y_{i}$ . − Show that 

$$
\begin{array}{r c l}{{P(\pmb{y}_{-i},Y_{i}=y_{i}^{\prime}\mid\theta)}}&{{=}}&{{\displaystyle\frac{\partial f_{\Phi}(\pmb{\theta},\pmb{\lambda}^{y})}{\lambda_{y_{i}^{\prime}}}}}\\ {{P(\pmb{y}_{-i}\mid\theta)}}&{{=}}&{{\displaystyle\sum_{y_{i}^{\prime}}\frac{\partial f_{\Phi}(\pmb{\theta},\pmb{\lambda}^{y})}{\lambda_{y_{i}^{\prime}}}.}}\end{array}
$$ 

# Exercise $9.5\star$ 

sensitivity analysis 

In this exercise, you will show how you can use the gradient of the probability of a Bayesian network to perform sensitivity analysis , that is, to compute the efect on a probability query of changing the parameters in a sing PD $P(X\mid U)$ . More precisely, let $\theta$ be one set of parameters for a $\mathcal{G}$ , wh we have that $\theta_{x\mid u}$ | is the parameter associated with the conditional probability entry $P(X\mid U)$ | . Let $\theta^{\prime}$ be another parameter assignment that is the same except that we replace the parameters $\theta_{x\mid u}$ with $\theta_{x\mid u}^{\prime}=\theta_{x\mid u}+\Delta_{x\mid u}$ . 

For an assignment $e$ (which may or may not involve variables in $X,U$ , compute the change $P(e:$ $\pmb{\theta})-P(e:\mathbf{\check{\theta}}^{\prime})$ in terms of $\Delta_{x\mid u}$ , and the network derivatives. 

# Exercise ${\bf9.6\star}$ 

Consider some run of variable elimination over the factors $\Phi$ , where all variables are eliminated. This run generates some set of intermediate factors $\tau_{i}(W_{i})$ . We can deﬁne a set of intermediate (arithmetic, not random) variables $v_{i k}$ corresponding to the diferent entries $\tau_{i}(w_{i}^{k})$ . 

a. Show how, for each variable $v_{i j}$ , we can write down an algebraic expression that deﬁnes $v_{i j}$ in terms of: the parameters $\lambda_{x_{i}}$ ; the parameters $\theta_{x_{c}}$ ; and variables $v_{j l}$ for $j<i$ . b. Use your answer to the previous part to deﬁne an alternative representation whose complexity is linear in the total size of the intermediate factors in the VE run. c. Show how the same representation can be used to compute all of the derivatives of the network polynomial; the complexity of your algorithm should be linear in the compact representation of the network polynomial that you derived in the previous part. (Hint: Consider the partial derivatives of the network polynomial relative to each $v_{i j}$ , and use the chain rule for derivatives.) 

# Exercise 9.7 

Prove proposition 9.1. 

# Exercise ${\bf9.8\star}$ 

Prove theorem 9.10, by showing that any ordering produced by the maximum cardinality search algorithm eliminates cliques one by one, starting from the leaves of the clique tree. 

# Exercise 9.9 

a. Show that variable elimination on polytrees can be performed in linear time, assuming that the local probability models are represented as full tables. Speciﬁcally, for any polytree, describe an elimination ordering, and show that the complexity of variable elimination with your ordering is linear in the size of the network. Note that the linear time bound here is in terms of the size of the CPTs in the network, so that the cost of the algorithm grows exponentially with the number of parents of a node. b. Extend your result from (1) to apply to cases where the CPDs satisfy independence of causal inﬂuence. Note that, in this case, the network representation is linear in the number of variables in the network, and the algorithm should be linear in that number. c. Now extend your result from (1) to apply to cases where the CPDs are tree-structured. In this case, the network representation is the sum of the sizes of the trees in the individual CPDs, and the algorithm should be linear in that number. 

# Exercise ${\bf9.10\star}$ 

Consider the four criteria described in connection with Greedy-Ordering of algorithm 9.4: Min-Neighbors, Min-Weight, Min-Fill, and Weighted-Min-Fill. Show that none of these criteria dominate the others; that is, for any pair, there is always a graph where the ordering produced by one of them is better than that produced by the other. As our measure of performance, use the computational cost of full variable elimination (that is, for computing the partition function). For each counterexample, deﬁne the structure of the graph and the cardinality of the variables, and show the ordering produced by each member of the pair. 

# Exercise $\mathbf{9.11}\star$ 

Let $\mathcal{H}$ be an undirected graph, and $\prec$ an Pro that $X{-}Y$ ll edg for all induce i $i\stackrel{.}{=}1,\ldots,k$ nd only if there is a path . $X{\mathrm{-}}Z_{1}{\mathrm{-}}\ldots Z_{k}{\mathrm{-}}Y$ in H such that $Z_{i}\prec X$ ≺ and $Z_{i}\prec Y$ ≺ 

# Exercise $\mathbf{9.12\star}$ 

Prove theorem 9.12. 

# Exercise ${\bf9.13\star}$ 

Prove theorem 9.13. 

# Exercise $9.14\star$ 

The standard conditioning algorithm ﬁrst conditions the network on the conditioning variables $U$ , splitting the computation into a set of computations, one for every instantiation $\mathbfit{u}$ to $U$ ; it then performs variable elimination on the remaining network. As we discussed in section 9.5.4.1, we can generalize conditioning so that it alternates conditioning steps and elimination in an arbitrary way. In this question, you will formulate such an algorithm and provide a graph-theoretic analysis of its complexity. 

et $\Phi$ be a set of factors over $\mathcal{X}$ , and let $_{X}$ be a set of non riab e a summ cedure $\sigma$ that each to be a X $X\,^{-}\!\in\,X$ ∈ e of operations, each of w appears in the sequence σ precisely once. The semantics of this procedure is that, h is either $\bar{e l i m}(X)$ or $c o n d(X)$ for some $X\in X$ ∈ , such going from left to right, we perform the operation described on the variables in sequence. For example, the summation procedure of example 9.5 would be written as: 

$$
e l i m(A_{k-1}),e l i m(A_{k-2}),.\,.\,.\,.\,e l i m(A_{1}),c o n d(A_{k}),e l i m(C),e l i m(B).
$$ 

a. Deﬁne an algorithm that takes a summation sequence as input and performs the operations in the order stated. Provide precise pseudo-code for the algorithm. b. Deﬁne the notion of an induced graph for this algorithm, and deﬁne the time and space complexity of the algorithm in terms of the induced graph. 

# Exercise $\mathbf{9.15\star}$ 

In section 9.6.1.1, we described an approach to decomposing noisy-or CPDs, aimed at reducing the cost of variable elimination. In this exercise, we derive a construction for CPD-trees in a similar spirit. 

a. Consider a variable $Y$ that has a binary-valued parent $A$ and four additional parents $X_{1},\dots,X_{4}$ . Assume that the CPD of $Y$ is structured as a tree whose ﬁrst split is $A$ , and where $Y$ depends only on $X_{1},X_{2}$ in the $A=a^{1}$ branch, and only on $X_{3},X_{4}$ in the $\ensuremath{\boldsymbol{A}}^{\textup{\scriptsize{\bar{\ }}}}=\ensuremath{\boldsymbol{a}}^{0}$ branch. Deﬁne two new variables, $Y_{a^{1}}$ and $Y_{a^{0}}$ , which represent the value that $Y$ would take if $A$ were to have the value $a^{1}$ , and the value that $Y$ would take if $A$ were to have the value $a^{0}$ . Deﬁne a new model for $Y$ that is deﬁned in terms of these new variables. Your model should precisely specify the CPDs for $Y_{a^{1}}$ , $Y_{a^{0}}$ , and $Y$ in terms of $Y\mathbf{\dot{s}}$ original CPD. 

b. Deﬁne a general procedure that recursively decomposes a tree-CPD using the same principles. 

# Exercise 9.16 

In this exercise, we show that rule-based variable elimination performs exactly the same operations as table-based variable elimination, when applied to rules generated from table-CPDs. Consider two table fac $\phi(X),\phi^{\prime}(Y)$ . Let $\mathcal{R}$ be the set of constituent rules for $\phi(X)$ and $\mathcal{R}^{\prime}$ the set of constituent rules for $\phi(Y)$ . 

a. Show that the operation of multiplying $\boldsymbol{\phi}\cdot\boldsymbol{\phi}^{\prime}$ can be implemented as a series of rule splits on $\mathcal{R}\cup\mathcal{R}^{\prime}$ , followed by a series of rule products. b. ow that the operation of summing out $Y\in X$ in $\phi$ can be implemented as a series of rule sums in . 

# Exercise $9.17\star$ 

Prove that each step in the algorithm of algorithm 9.7 maintains the program-correctness invariant de- scribed in the text: Let $\mathcal{R}$ be the current set of rules maintained by the algorithm, and $W$ be the variables that have not yet been eliminated. The invariant is that: 

ility of a context $^c$ such that $S c o p e[c]\subseteq W$ can be obtained by multiplying all rules $\langle\pmb{c}^{\prime};\hat{p}\rangle\in\mathcal{R}$ whose context is compatible with c . 

# Exercise ${\bf9.18\star\star}$ 

Consider an alternative factorization of a Bayesian network where each factor is a hybrid between a rule and a table, called a confactor . Like a rule, a confactor associated with a context $^{c;}$ however, rather than a single number, each confactor contains not a single number, but a standard table-based factor. For example, the CPD of ﬁgure 5.4a would have a confactor, associated with the middle branch, whose context is $a^{1},{\bar{s}}^{0}$ , and whose associated table is 

$$
\begin{array}{l l}{{l^{0},j^{0}}}&{{0.9}}\\ {{l^{0},j^{1}}}&{{0.1}}\\ {{l^{1},j^{0}}}&{{0.4}}\\ {{l^{1},j^{1}}}&{{0.6}}\end{array}
$$ 

Extend the rule splitting algorithm of algorithm 9.6 and the rule-based variable elimination algorithm of algorithm 9.7 to operate on confactors rather than rules. Your algorithm should use the efcient table-based data structures and operations when possible, resorting to the explicit partition of tables into rules only when absolutely necessary. 

# Exercise ${\bf9.19\star\star}$ 

We have shown that the sum-product variable elimination algorithm is sound, in that it returns the same answer as ﬁrst multiplying all the factors, and then summing out the nonquery variables. Exercise 13.3 asks for a similar argument for max-product. One can prove similar results for other pairs of operations, such as max-sum. Rather than prove the same result for each pair of operations we encounter, we now provide a generalized variable elimination algorithm from which these special cases, as well as others, follow directly. This general algorithm is based on the following result, which is stated in terms of a pair of abstract operators: generalized combination of two factors, denoted $\phi_{1}\otimes\phi_{2}$ N φ ; and generalized marginaliz on of a factor $\phi$ over a subset $W$ , denoted $\Lambda_{W}(\phi)$ . We deﬁne our generalized variable elimination algorithm in direct analogy to the sum-product a rithm of algorithm 9.1, replacing factor product with N and summation for variable elimination with Λ . 

We now show that if these two operators satisfy certain conditions, the variable elimination algorithm for these two operations is sound: 

Commutativity of combination: For any factors $\phi_{1},\phi_{2}$ : 

$$
\phi_{1}\bigotimes\phi_{2}=\phi_{2}\bigotimes\phi_{1}.
$$ 

Associativity of combination: For any factors $\phi_{1},\phi_{2},\phi_{3}$ : 

$$
\phi_{1}\bigotimes(\phi_{2}\bigotimes\phi_{3})=(\phi_{1}\bigotimes\phi_{2})\bigotimes\phi_{3}.
$$ 

Consonance of marginalization: If $\phi$ is a factor of scope $W$ , and $Y,Z$ are disjoint subsets of $W$ , then: 

$$
\Lambda_{Y}(\Lambda_{Z}(\phi))=\Lambda_{(Y\cup Z)}(\phi).
$$ 

Marginalization over combination: If $\phi_{1}$ is a factor of scope $W$ and $Y\cap W=\emptyset$ , then: 

$$
\Lambda_{Y}(\phi_{1}\bigotimes\phi_{2})=\phi_{1}\bigotimes\Lambda_{Y}(\phi_{2}).
$$ 

Show that if $\otimes$ and $\Lambda$ satisfy the preceding axioms, t n we obtain a theorem analogous to th rem 9.5. That is, the algorithm, when applied to a set of factors Φ and a set of variables to be eliminated Z , returns a factor 

$$
\phi^{*}(Y)=\Lambda z(\bigotimes_{\phi\in\Phi}\phi).
$$ 

# Exercise ${\bf9.20\star\star}$ 

You are taking the ﬁnal exam for a course on computational complexity theory. Being somewhat too theoretical, your professor has insidiously sneaked in some unsolvable problems and has told you that exactly $K$ of the $N$ problems have a solution. Out of generosity, the professor has also given you a probability distribution over the solvability of the $N$ problems. 

To f malize the scenario, let $\mathcal{X}=\{X_{1},.\,.\,.\,,X_{N}\}$ variables correspo ing to the N questions in the exam where $V a l(X_{i})\,=\,\bar{\mathrm{\left\{0(unsolveeable),1(solveeable)\right\}}}$ { } . Fur ermore, let B be a Bayesian network parameterizing a probability d strib ion over X (that is, problem i may be easily used to solve problem $j$ so that the probabilities that i and $j$ are solvable are not independent in general). 

a. We begin by describing a method for computing the probability of a question being solvable. That is we want to compute $\breve{P}(X_{i}=1,\mathrm{PSbubble}(\hat{\mathcal{X}})=\breve{K})$ ) where 

$$
\operatorname{PSIM}({\mathcal{X}})=\sum_{i}\mathbf{1}\{X_{i}=1\}
$$ 

is the number of solvable problems assigned by the professor. 

To this end, we deﬁne an extended factor $\phi$ as a “regular” factor $\psi$ and an index so that it deﬁnes a ction $\phi(X,L):V a l(X)\times\{0,\dot{.}\,.\,,N\}\mapsto I\!\!R$ where $X=S c o p e[\phi]$ . A projection of such a factor $[\phi]_{l}$ l is a regular factor $\psi:V a l(X)\mapsto I\!\!R,$ 7→ , such that $\psi(X)=\phi(X,{\bar{l}})$ . 

Provide a deﬁnition of factor combination and factor marginalization for these extended factors such that 

$$
P(X_{i},\mathrm{PSimize}(\mathcal{X})=K)=\left[\sum_{\mathcal{X}-\{X_{i}\}}\prod_{\phi\in\Phi}\phi\right]_{K},
$$ 

where each $\phi\in\Phi$ is an extended factor corresponding to some CPD of the Bayesian network, deﬁned as follows: 

$$
\begin{array}{r}{\phi_{X_{i}}\big(\{X_{i}\}\cup\mathbf{Pa}_{X_{i}},k\big)=\left\{\begin{array}{l l}{P(X_{i}\mid\mathrm{Pa}_{X_{i}})}&{\mathrm{if~}X_{i}=k}\\ {0}&{\mathrm{otherwise}}\end{array}\right.}\end{array}
$$ 

b. Show that your operations satisfy the condition of exercise 9.19 so that you can compute equation (9.16) use the generalized variable elimination algorithm. 

c. Realistically, you will have time to work on exactly $M$ problems $1\leq M\leq N)$ . Obviously, your goal is to maximize the expected number of solvable problems that you attempt. (Luckily for you, every solvable problem that you attempt you will solve correctly, and you neither gain nor lose credit for working on an unsolvable problem.) Let $\mathbf{Y}$ be a subset of $\mathcal{X}$ indicating exactly $M$ problems you choose to work on, and let 

$$
\operatorname{Correct}(X,Y)=\sum_{X_{i}\in Y}X_{i}
$$ 

be the number of solvable problems that you attempt. The expected number of problems you solve is 

$$
\pmb{\mathscr{E}}_{P_{\mathcal{B}}}[\mathrm{Correct}(\mathcal{X},Y)\mid\mathrm{PSbubble}(\mathcal{X})=K].
$$ 

Using your generalized variable elimination algorithm, provide an efcient algorithm for computing this expectation. 

d. Your goal is to ﬁnd $\mathbf{Y}$ that optimizes equation (9.17). Provide a simple example showing that: 

$$
\arg\operatorname*{max}_{Y:\vert Y\vert=M}E_{P_{B}}[\mathrm{Correct}(\mathcal{X},Y)]\neq\arg\operatorname*{max}_{Y:\vert Y\vert=M}E_{P_{B}}[\mathrm{Correct}(\mathcal{X},Y)\ \vert\ \mathrm{possible}(\mathcal{X})=K
$$ 

e. Give an efcient algorithm for ﬁnding 

$$
\arg\operatorname*{max}_{Y:\lvert Y\rvert=M}E_{P_{\mathcal{B}}}\bigl[\mathrm{Correct}(\mathcal{X},Y)\ \vert\ \mathrm{PSbubble}(\mathcal{X})=K\bigr].
$$ 

(Hint: Use linearity of expectations.) 