# Summary
The formulation of conditional probability models for finite systems of spatially interacting random variables is examined. A simple alternative proof of the Hammersley-Clifford theorem is presented and the theorem is then used to construct specific spatial schemes on and off the lattice. Particular emphasis is placed upon practical applications of the models in plant ecology when the variates are binary or Gaussian. Some aspects of infinite lattice Gaussian processes are discussed. Methods of statistical analysis for lattice schemes are proposed, including a very fe xi ble coding technique. The methods are illustrated by two numerical examples. Itis maintained throughout that the conditional probability approach to the specification and analysis of spatial interaction is more attractive than the alternative joint probability approach. 
> 本文：
> 构建了有限空间内交互的随机变量系统的条件概率模型
> 为 HC 定理提出了另一种证明，并探讨了使用 HC 定理在晶格和非晶格上构建具体的空间模式（晶格：规则排列的点阵）
> 特别探讨了变量是二元和高斯时的应用
> 探讨了无限晶格上的高斯过程
> 提出了适用于晶格方案的统计分析方法，包括一个灵活的编码技术
> 认为条件概率模型对于空间交互的指定和分析比联合概率方法更有效

# 1 Introduction
In this paper, we examine some stochastic models which may be used to describe certain types of spatial processes. Potential applications of the models occur in plant ecology and the paper concludes with two detailed numerical examples in this area. At a formal level, we shall largely be concerned with a rather arbitrary system, consisting of a finite set of sites, each site having associated with it a univariate random variable. In most ecological applications, the sites will represent points or regions in the Euclidean plane and will often be subject to a rigid lattice structure. For example, Cochran (1936) discusses the incidence of spotted wilt over a rectangular array of tomato plants. The disease is transmitted by insects and, after an initial period of time, we should clearly expect to observe clusters of infected plants. The formulation of spatial stochastic models will be considered in Sections 2-5 of the paper. Once having set up a model to describe a particular situation, we should then hope tobe able to estimate any unknown parameters and to test the goodness-of-fit of the model on the basis of observation. We shall discuss the statistical analysis of lattice schemes in Sections 6 and 7. 
> 本文探讨应用于描述特定类型空间过程的随机模型
> 考虑有有限个站点构成的任意系统，每个站点关联一个单变量随机变量，site 可以表示欧式空间中的点或区域

We begin by making some general comments on thetypesof spatial systems which we shall, and shall not, be discussing. Firstly, we shall not be concerned here with any random distribution which maybe associated with the locations of the sites themselves. Indeed, when setting up models in practice, we shall require quite specific information on the relative positions of sites, in order toassess the likely interdependence between the associated random variables. Secondly, although, as in Cochran's example above, the system may, in reality, have developed continuously through time, we shall always assume that observation on it is only available at an isolated instant; hence, we shall not be concerned here with the setting up of spatial- temporal schemes. This has the important consequence that our models will not be mechanistic and must be seen as merely attempts at describing the “here and now" of a wider process. In many practical situations, this is a reasonable standpoint, since we can only observe the variables at a single point in time (for example, the yields of fruit trees in an orchard) but, in other cases, a spatial-temporal approach may be more appropriate. In fact, the states of the tomato plants, in Cochran's example, were observed at three separate points in time and it is probably most profitable to use a classical temporal autoregression to analyse the system. A similar comment applies to the hop plants data of Freeman (i953). Ideally, even when dealing with a process at a single instant of time, we should first set up an intuitively plausible spatial-temporal model and then derive the resulting instantaneous spatial structure. This can sometimes be done if we are prepared to assume stationarity in both time and space (see Bartlett, 197la) but, unfortunately, such an assumption is unlikely to be realistic in our context. However, when this approach is justifiable, it is of course helpful to check that our spatial models are consistent with it; forsimple examples, see Besag (1972a). Otherwise, regarding the transient spatial structure of a spatial-temporal process, this is almost always intractable and hence there exists a need to set up and examine purely spatial schemes without recourse to temporal considerations. 
> 仅考虑空间方案，不考虑时序

The following examples areintended as typical illustrations of the spatial situations we shall have in mind. They are classified according to the nature of 
(a) the system of sites (regular or irregular), 
(b) the individual sites (points or regions) and 
(c) the associated random variables (discrete or continuous) 

1.1. A regular lattice of point sites with discrete variables commonly occurs under experimental conditionsin plant ecology. Examples include the pattern of infection in anarrayof plants (Cochran, 1936, as describedabove; Freeman, 1953, onthe incidence of nettlehead virus in hop plants) and the presence or absence of mature plants seeded on a lattice and subsequently under severe competition for survival (data kindly supplied by Dr E. D. Ford, Institute of Tree Biology, Edinburgh, relates to dwarf French marigolds on a triangular lattice of side 2cm). Often, asabove, the dataarebinary. 

1.2. A regularlattice ofpoint sites with continuous variables commonly occurs in agricultural experiments, where individual plant yields are measured (Mead, 1966, 1967,1968, on competition models; Batch el or and Reed, 1918, onfruittrees). Itis often reasonable to assume that the variate s have a multivariate normal distribution. 

1.3. A regular lattice of regions with discrete variables arises in sampling an irregularly distributed population when a rectangular grid is placed over an area and counts are made of the number of individuals in each quadrat (Professor P. Greig- Smithon Carexarenaria, inBartlett, 1971b; Gleaves, 1973, onPlantagolanceolata；Clarke, 1946, and Feller, 1957, p. 150, on flying-bomb hits in South London during WorldWarH; Matui, 1968, on the locations of farms and villages in an area of Japan). In plant ecology, the quadrats are often so small that few contain more than a single plant and it is then reasonable to reduce the data to a binary (presence/ absence) form. 

1.4. A regular lattice of regions with continuous variables typically occurs in field trials where aggregate yields are measured (Mercer and Hall, 1911, on wheat plots). Multivariate normality is often a reasonable assumption 

1.5. Irregular point sites with discrete variables arise in sampling natural plant populations. Examples include the presence or absence of infection in individuals and the variety of plant at each site in a multi-species community. 

1.6. Irregular point sites with continuous variables again occur in sampling natural plant populations (Brown, 1965, on tree diameters in pine forests; Mead, 1971, on competition models). 

1.7. Irregular regions with discrete orcontinuous variableshave applications particularly in a geographical context, with regions defined by administrative boundaries (O'Sullivan, 1969, and Ord, 1974, on aspects of the economy of Eire) 
> 这里讨论了一些什么情况下用什么具体建模的实例，都是植物学相关

It has previously been stated that in thepractical construction of spatial models, we shall require precise information concerning the relative positions of the various sites. Where the sites are regions, rather than points, the data are by definition, aggregate data and the assumption of single, uniquely defined locations for each of the associated variables is clearly open to criticism. For example, quadrat counts (Section 1.3) are usually used to examine spatial pattern rather than spatial inter- action. Further comments will appear in Section 5. 
> 要求空间方案中，站点之间的确切相对位置要知道，以方便建模依赖性和独立性

Combinations of the above situations may occur. For example, in competition experiments where yields are measured, “missing observations" may be due to intense competition and should then be specifically accounted for by the introduction of mixed distributions. We shall not contemplate such situations here. 

# 2 Conditional Probability Approach to Spatial Process
There appear to be two main approaches to the specification of spatial stochastic processes. These stem from the non-equivalent definitions of a“nearest-neighbour' system, Originally due to Whittle (1963) and Bartlett (1955 Section 2.2, 1967, 1968), respectively. 
> 指定空间随机过程存在两大主流方法，差异源于“最近邻”系统的不同定义 (Whittle and Bartlett)

Suppose, for definiteness, that we temporarily restrict attention to a rectangular lattice with sites labelled by integer pairs $(i,j)$ and with anassociatedset $\{X_{i,j}\}$ finiteness or otherwise of the lattice. 
> 我们考虑长方形晶格，其站点用 $(i, j)$ 标记，晶格与随机变量集合 $\{X_{i, j}\}$ 相关

Then Whittle's basic definition requires that the joint probability distribution of the variates should be of the product form 
> Whittle 的定义要求联合分布应该是以下的乘积形式
> 可以看到 (2.1) 中，整体的联合分布定义为所有 $\mathcal Q_{i, j}$ 的乘积，而 $\mathcal Q_{i, j}$ 依赖于 $X_{i, j}$ 本身和其上下左右的四个随机变量

$$
\prod_{i,j}\mathcal{Q}_{i,j}(x_{i,j};\,x_{i-1,j},x_{i+1,j},x_{i,j-1},x_{i,j+1}),\tag{2.1}
$$ 
where $x_{i,j}$ is a value of the random variable, $X_{i,j}.$ .

On the other hand, Bartlett's definition requires that the conditional probability distribution of $X_{i,j},$ given all other site values, should depend only upon the values at the four nearest sites to $(i,j),$ $x_{i-1,j},\;x_{i+1,j},\,x_{i,j-1}$ ${x_{i,j+1}}$ may be said to have rather more intuitive appeal, this is marred by a number of disadvantages. 
> Barlett 的定义则是从条件概率的角度出发，要求给定所有站点的值，$X_{i, j}$ 的条件概率分布仅依赖于与它直接邻近的四个站点

Firstly, there is no obvious method of deducing the joint probability structure associated with a conditional probability model. Secondly, the conditional proba. bility structure itself is subject to some unobvious and highly restrictive consistency conditions. When these are enforced, it can be shown (Brook, 1964) that the con- ditional probability formulation is degenerate with respect to (2.1). Thirdly, it has been remarked by Whittle (1963) that the natural specification of an equilibrium process in statistical mechanics is in terms of the joint distribution rather than the conditional distribution of thevariables. 
> 讨论了目前认为的条件概率方法的一些问题和限制

These problems were partially investigated in a previous paper (Besag, 1972a) The constraints on the conditional probability structure were identified for homospatial models, given the nature of the variables. Had these models failed to retain anypractical appeal, then there would have been little further scopefor discussion. 

However, this is not the case. For example, with binary variables, the conditional probability formulation necessarily generates just that basic model (the Ising model of ferro magnetism) which has been at the centre of so much work in statistical mechanics. Thus, although this model may classically be formulated in terms of joint probabilities, it is generated in a natural way through basic conditional probability assumptions. This fact may also be related to the problem of degeneracy. There is surely no indignity in studying a subclass of schemes provided that subclass is of interest in its own right. However, we go further. Suppose we consider wider classes of conditional probability models in which the conditional distribution of $X_{i,j}$ is allowed to depend upon the values at more remote sites. We can build up a hierarchy of models, more and more general, which eventually will include the scheme (2.1) and any particular generalization of it. Thatis, we extend the concept of first-, second- andhigher-orderMarkov chainsinone dimension to therealm of spatial processes. There is then no longer any degeneracy associated with the conditional probability models. This is the approach takenin thepresentpaper. It has been made possible by the advent of the celebrated Hammersley-Clifford theorem which, sadly, has remained unpublished by its authors. 
> 作者认为条件概率模型并非不可用：
> 例如对于二元变量，条件概率构建可以必要地产生基础模型，例如 Ising 模型，虽然该模型传统上是用联合概率表示的，但通过基本的条件概率模型也可以自然地生成该模型
> 条件概率模型可能会退化到一个子类问题，但研究子类问题是有价值的
> 通过考虑 $X_{i, j}$ 的条件概率依赖于更远的站点，可以构建层次化模型，拓展到更广泛的模型，甚至包括式 (2.1) 及其任何特例
> 在空间过程中，该拓展类似于将 Markov 链拓展到更高阶的概念
> 因此，条件概率模型就不再与任何退化问题相关，此即本文的方法，它依赖于 HC 定理（该定理允许从条件概率分布推导出联合概率分布，并且证明了马尔可夫随机场可以表示为吉布斯分布）

Finally, in this section, we examine the problems and implications of deriving the joint probability structure associated with the site variables, given their individual conditional distributions. We nolonger restrict attention to“nearest-neighbour" models nor even to lattice schemes but instead consider a fairly arbitary system of sites. Suppose then that we are concerned with a finite collection of random variables, $X_{1},...,X_{n},$ which are associated with sites labelled ${1,...,n,}$ respectively. For each site, $P(x_{i}\vert\,x_{1},...,x_{i-1},x_{i+1},...,x_{n}),$ the conditional distribution of $X_{i},$ given all other site values, is specified and we require the joint probability distribution of the variables. Our terminology will be appropriate to discrete variables but the arguments equally extend to the continuous case. 
> 本节最后也会讨论使用条件概率分布推导联合分布的问题和影响，并且将考虑的模型范围拓展到任意的站点系统，即对于有限的随机变量集合 $X_1, \dots, X_n$，在给定每个 $X_i$ 的条件概率分布（给定所有其他变量）时，如何推导整体的联合分布
> 方法可以拓展到连续变量

We make the following important assumption: if $\pmb{x_{1}},...,\pmb{x_{n}}$ can individually occur at the sites ${1,...,n,}$ respectively, then they can occur together. Formally, if $P(x_{i})\!>\!0$ for each $i_{!}$ then $P(x_{1},...,x_{n})\!>\!0$ .This is called thepositivity condition byHammersley and Clifford (i97l) and will be assumed throughout the present paper. It is usually satisfied in practice. We define the sample space $\Omega$ to be theset of all possible realizations $\pmb{x}=(x_{1},...,x_{n})$ of the system. That is, $\Omega=\{\mathbf{x}\colon P(\mathbf{x})\!>\!0\}.$ 
> 重要假设：$P$ 为正分布
> 定义 $\Omega$ 为样本空间 $\Omega = \{\pmb x: P (\pmb x) > 0\}$

It then follows that for any two given realizations $\mathbf{x}$ and $\mathbf{y}\!\in\!\Omega$ 

$$
{\frac{P(\mathbf{x})}{P(\mathbf{y})}}=\prod_{i=1}^{n}{\frac{P(x_{i}|x_{1},...,x_{i-1},y_{i+1},...,y_{n})}{P(y_{i}|x_{1},...,x_{i-1},y_{i+1},...,y_{n})}}.\tag{2.2}
$$ 
The proof of this result resembles that of equation (6) in Besag (1972a). Clearly, we may write 

$$
P(\mathbf x)=P(x_{n}|x_{1},...,x_{n-1})P(x_{1},...,x_{n-1});\tag{2.3}
$$ 
however, $P(x_{1},...,x_{n-1})$ cannot be factorized in a useful way since, for example, $P(x_{n-1}|x_{1},...,x_{n-2})$ is not easily obtained from the given conditional distributions. 

Nevertheless, we can introduce $y_{n}$ write 

$$
P(\mathbf x)\!=\!\frac{P(x_{n}|x_{1},...,x_{n-1})}{P(y_{n}|x_{1},...,x_{n-1})}P(x_{1},...,x_{n-1},y_{n})
$$ 

> 这一步来源于：
> $P (x_1,\dots, x_{n-1}) P (y_n \mid x_1,\dots, x_{n-1}) = P (x_1,\dots, x_{n-1}, y_n)$
> $P (x_1,\dots, x_{n-1}) = \frac {P (x_1, \dots, x_{n-1}, y_n)}{P (y_n\mid x_1, \dots, x_{n-1})}$

and now operate on ${x_{n-1}}$ in $P(x_{1},...,x_{n-1},y_{n})$ . This yields 

$$
P(x_{1},...,x_{n-1},y_{n})={\frac{P(x_{n-1}|x_{1},...,x_{n-2},y_{n})}{P(y_{n-1}|x_{1},...,x_{n-2},y_{n})}}P(x_{1},...,x_{n-2},y_{n-1},y_{n}),
$$ 
> 这一步来源于：
> $P (x_1, \dots, x_{n-1}, y_n) = P (x_{n-1}\mid x_1, \dots, x_{n-2}, y_n) P (x_1,\dots, x_{n-2}, y_n)$
> 而 $P (x_1,\dots, x_{n-2}, y_n) P (y_{n-1}\mid x_1, \dots, x_{n-2}, y_n) = P (x_1,\dots, x_{n-2}, y_{n-1}, y_n)$
> 将 $P (x_1, \dots, x_{n-2}, y_n)$ 替换为 $\frac {P (x_1,\dots, x_{n-2}, y_{n-1}, y_n)}{P (y_{n-1}\mid x_1, \dots, x_{n-2}, y_n)}$ 即可

after the similar introduction of $y_{n-1}$ .Continuing the reduction process, we eventually arrive at equation (2.2) which clearly determines the joint probability structure of the system in terms of the given conditional probability distributions. We require the positiv it y condition merely toensure thateach term in the denominator of (2.2) is non-zero. 
> 通过该手段，不断引入 $y_i$，我们最终可以得到式 (2.2)，也就是用给定的条件概率决定了系统的联合概率结构
> 正分布的假设就是防止分母为0

Equation (2.2) highlights the two fundamental difficulties concerning the specification of a system through its conditional probability structure. Firstly, the labelling of individual sites in the system being arbitrary implies that many factorizations of $P(\mathbf{x})/P(\mathbf{y})$ are possible. All of these must, of course, be equivalent and this, in turn, implies the existence of severe restrictions on the available functional forms of the conditional probability distributions in order to achieve a mathematically consistent joint probability structure. This problem has been investigated by Levy (1948), Brook (1964), Spitzer (1971), Hammersley and Clifford (1971) and Besag (1972a) and we discuss it in detail in the next section. Secondly, whilst expressions for the relative probabilities of two realizations may be fairly straightforward, those for absolute probabilities, in general, involve an extremely awkward normalizing function with the consequence that direct approaches to statistical inference through the likelihood function are rarely possible. We shall have to negotiate this problem in Section 6 of thepaper. 
> 式 (2.2) 也指出了两大困难：
> 系统中各站点的标签是任意的，故 $P (\pmb x)/ P (\pmb y)$ 存在多种可能分解，这些分解需要等价，这对条件概率分布的函数形式施加了限制，因为要保证数学上一致的联合概率结构
> 两个赋值/实现的相对概率 $P (\pmb x)/P (\pmb y)$ 的表示是直接的，但绝对概率则涉及到复杂的归一化函数，这使得通过似然函数进行直接的统计推断方法很少可行

# 3 Markov Fields and The Hammersly-Clifford Theroem
In this section, we examine the constraints on the functional form of the conditional probability distribution available at each of the sites. We restate a theorem of Hammersley and Clifford (1971) and give a simple alternative proof. This theorem, which has received considerable attention recently, is essential to the construction of valid spatial schemes through the conditional probability approach. We begin by describing the problem more precisely. Our definitions will closely follow those of Hammersley and Clifford. 
> 本节探讨每个站点的条件概率分布的函数形式的限制，并对 HC 定理给出一种证明
> HC 定理对于通过条件概率方法来构建有效的空间方案是必要的

The first definition determines the set of neighbours for each site. Thus, site $j(\neq i)$ is said to be a neighbour of site $_i$ if and onlyif thefunctional formof $P(x_{i}|\,x_{1},...,x_{i-1},x_{i+1},...,x_{n})$ is dependent upon the variable ${x}_{j}$ .
> 首先定义每个站点的邻居集合：
> 有站点 $i$，站点 $j (\ne i)$ 当且仅当 $x_i$ 的给定其余全部节点的条件概率分布 $P (x_i \mid x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n)$ 的函数形式是依赖于变量 $x_j$ 时，站点 $j$ 是站点 $i$ 的邻居（也就是给定邻居节点，该节点条件独立于其他所有节点）

As the simplest example, suppose that $X_{1},...,X_{n}$ is a Markov chain. Then it is easily shown that site $i\left(2\!\leqslant\!i\!\leqslant\!n\!-\!1\right)$ has neighbours $_{i-1}$ and $_{i+1}$ whilst the sites 1 and ${n}$ have the single neighbours 2 and ${n-1}$ , respectively. 
> 例如，Markov $X_1,\dots, X_n$ 中，节点 $i (2\le i \le n-1)$ 的邻居节点是 $i-1, i+1$，首尾的两个节点则仅有一个邻居
> 对于一个节点，给定其邻居之后，它就与其他剩余节点条件独立

For a more interesting spatial example, suppose the sites form a finite rectangular lattice and are now conveniently labelled by integer pairs $(i,j)$ .Then, if $P(x_{i,j}|$ all other site values) depends only upon $x_{i,j},\,x_{i-1,j},\,x_{i+1,j},$ $x_{i,j-1}$ and ${x_{i,j+1}}$ for each internal site $(i,j)$ , we have a so-called “nearest-neighbour' lattice scheme. In such a case, each internal site $(i,j)$ has four neighbours, namely $(i\!-\!1,j),(i\!+\!1,j),(i,j\!-\!1)$ and $(i,j+1)$ .(There is a slight inconsistency in the usage of the word “neighbour" here: this will be resolved in later sections by introducing the term“first-order"scheme.) 
> 又例如，对于一个点阵，其内部的站点的邻居就是其上下左右的四个站点

Any system of ${n}$ sites, each with specified neighbours, clearly generates a class ofvalid stochastic schemes. We call any member of this class a Markov field. Our aim is to be able to identify the class in any given situation.
> 对于任意包含 $n$ 个站点的系统，如果其中的每个站点都指定了邻居，它就生成了一类有效的随机方案，我们称其为 Markov 场
> 我们的目标是可以在任意给定的情况下都能识别出 Markov 场

Any set of sites which either consists of a single site or else in which every site is aneighbourofeveryothersiteinthesetiscalledaclique. Thus, inthe"nearest- neighbour"situation describedabove, there are. cliques of theform $\scriptstyle\{(i,j)\}.$ $\{(i{-}1,j),(i,j)\}$ and $\{(i,j{-}1),(i,j)\}$ over the entire lattice, possibly with adjustments at the boundary. The definition of a clique is crucial to the construction of valid Markov fields. 
> 对于一个站点集合，如果它仅包含单个站点，或者集合内的所有站点都是集合中任意其他站点的邻居，该集合被称为一个团
> 因此，在上述描述的邻居情况下，整个点阵上存在形式为 $\{(i,j)\}$、$\{(i-1,j),(i,j)\}$ 和 $\{(i,j-1),(i,j)\}$ 的团，边界处可能需要进行调整
> 团的定义对于构造有效的马尔可夫场至关重要。

We now make two assumptions, again following Hammers ley and Clifford. Firstly, we suppose that there are only a finite number of values available at each site although we shall relax this condition later in the section. Secondly, we assumethat thevalue zero is available at each site. If this is originally untrue, it can always be sub segue ntl y brought about by re-indexing the values taken at the offending sites, a procedure which will be illustrated in Section 4.3. 
>我们遵循 Hammersley 和 Clifford 的做法，作出两个假设
>首先，我们假定在每个站点上只存在有限数量的值
>其次，我们假设在每个站点上可以取零值。如果这个假设最初不成立，总是可以通过重新索引违规站点上的值来实现这一点，这种方法将在第 4.3 节中加以说明

This second assumption, whichis therefore made for purely technical reasons, ensuresthat, under the positivity con- dition, an entire realization of zeros is possible. That is, $P(\mathbf{0})>0$ and we may legitimately define 
>第二个假设纯粹是为了技术原因而作出的，它确保在正分布的条件下，我们可以对全部的站点赋零值，也就是说有 $P(\mathbf{0})\!>\!0$
>那么，我们可以合法地为任意 $\mathbf x \in \Omega$ 定义：

$$
\mathcal Q(\mathbf{x})\!\equiv\!\ln\left\{P(\mathbf{x})/P(\mathbf{0})\right\}\tag{3.1}
$$ 
for any ${\mathbf{x}}\in\Omega$ . 

Lastly given any ${\bf x}\in{\Omega}$ , we write ${\bf x}_{i}$ for the realization 

$$
(x_{1},...,x_{i-1},0,x_{i+1},...,x_{n}).
$$

> 对于任意给定的赋值 $\mathbf x \in \Omega$，我们定义 $\mathbf x_i$ 为 $(x_1, \dots, x_{i-1}, 0, x_{i+1}, \dots, x_n)$，也就是把 $\mathbf x$ 中的第 $i$ 项替换为了 $0$

The problem to which Hammersley and Clifford addressed themselves may now be stated as follows: given the neighbours of each site, what is the most general form which $\mathcal Q({\pmb x})$ may takein order to give a valid probability structure to the system? 
>HC 定理要解决的问题陈述如下：
>给定每个站点的邻居，为了给系统赋予有效的概率结构，$\mathcal Q({\pmb x})$ 可以采取的最一般形式是什么？

Since

$$
\begin{align}
&\exp(\mathcal Q(\mathbf x) - \mathcal Q(\mathbf x_i))\\
=&\exp\left(\ln\{P(\mathbf x)/P(\mathbf 0)\} - \ln\{P(\mathbf x_i)/P(\mathbf 0)\}\right)\\
=&\exp(\ln\{P(\mathbf x)/P(\mathbf x_i)\})\\
=&P(\mathbf x)/P(\mathbf x_i)\\
=&\frac {P(x_i\mid x_1, \dots, x_{i-1}, x_{i+1},\dots, x_n)P(x_1,\dots,x_{i-1}, x_{i+1},\dots,x_n)}
{P(0\mid x_1,\dots, x_{i-1},x_{i+1},\dots, x_n)P(x_1,\dots, x_{i-1},x_{i+1}, \dots, x_n)}\\
=&\frac {P(x_i\mid x_1, \dots, x_{i-1}, x_{i+1},\dots, x_n)}
{P(0\mid x_1,\dots, x_{i-1},x_{i+1},\dots, x_n)}\\
=& {P(x_i\mid x_1, \dots, x_{i-1}, x_{i+1},\dots, x_n)}/
{P(0\mid x_1,\dots, x_{i-1},x_{i+1},\dots, x_n)}\tag{3.2}
\end{align}
$$

the solution to this problem immediately gives the most general form which may be taken by the conditional probability distribution at each site. 
> 根据 (3.2)，我们可以知道，确认了 $\mathcal Q (\mathbf x)$ 的形式之后，我们就可以确定每个站点上条件概率的比值，进而确定每个站点的条件概率的形式

In dealing with the rather general situation described above, the Hammersley- Clifford theorem superseded the comparatively pedestrian results which had been obtained for “nearest-neighbour" systems on the ${k}$ -dimensional finite cubic lattice (Spitzer, 1971; Besag, 1972a). However, the original method of proof is circuitous and requires the development of an operational calculus (the “blackening algebra").
>在处理上述较为一般的情况时，Hammersley-Clifford 定理超越了之前对于在 $k$ 维有限立方格上的“最近邻”系统所获得的相对简单的结果（Spitzer, 1971；Besag, 1972a）
>然而，原始的 HC 定理的证明方法相当迂回

A simple alternative statement and proof of the theorem rest upon the observation that for any probability distribution $P(\mathbf {x})$ , subject to the above conditions, there exists an expansion of $Q({\bf x}).$ , unique on $\Omega$ and of the form 
> 此处提出 HC 定理的另一种证明
> 观察到，对于任意满足上述条件（正分布、可以取 $\mathbf x = \mathbf 0$）的分布 $P (\mathbf x)$，它对应的 $\mathcal Q(\mathbf x)$ 存在于在 $\Omega$ 上的唯一的展开，形式为：
> (证明见[[#Deduction for $ mathcal Q ( mathbf x)$ 's expansion|附录(待完成)]])

$$
\begin{align}
\mathcal Q(\mathbf x) &=\sum_{1\le i \le n}x_i G_i(x_i) + \sum_{1\le i<j\le n}x_ix_j G_{ij}(x_i, x_j)\\
&+\sum_{1\le i< j < k \le n}x_ix_jx_k G_{ijk}(x_i, x_j, x_k) + \dots\\
&+x_1x_2\dots x_n G_{1,2,\dots, n}(x_1, x_2, \dots, x_n).
\end{align}\tag{3.3}
$$

For example, we have 

$$
x_{i}\,G_{i}(x_{i})\!\equiv\!\mathcal Q(0,...,0,x_{i},0,...,0)\!-\!\mathcal Q(\mathbf 0),
$$ 
with analogous difference formula for the higher order ${G}$ -functions.

With the above notation, Hammersley and Clifford's result may be stated in the following manner:  *for any $\begin{array}{r}{1\!\leqslant\!i\!<\!j\!<\ldots\!<\!s\!\leqslant\!n,}\end{array}$ the function $G_{i,j,...,s}\,i n$ (3.3) may be non-null if and only if the sites $i,j,...,s$ form a clique.* 
> 此时，HC 定理可以表述为：
> 对于任意 $1\le i < j < \dots < s \le n$，式 (3.3) 中对应的函数 $G_{i, j,\dots, s}$ 当且仅当站点 $i, j,\dots, s$ 构成一个团时，该函数非空/非零

Subject to this restriction, the ${G}$ functions maybe chosen arbitrarily. Thus, given the neighbours of each site, we can immediately write down the most general form for $\mathcal Q({\bf x})$ and hence for the conditional distributions. We shall see examples of this later on. 
> 而当函数 $G$ 非空时，其形式可以任意选择
> 因此，给定每个站点的邻居，我们可以立即根据式 (3.3) 写出 $\mathcal Q (\mathbf x)$ 最一般的形式（仅和每个团有关），进而写出条件分布

*Proof of theorem.* It follows from equation (3.2) that, for any $\mathbf{x}\!\in\!\Omega,$ $\mathcal Q(\mathbf{x})\!-\!\mathcal Q(\mathbf{x}_{i})$ can only depend upon ${x_{i}}$ itself and the values at sites which are neighbours of site $i.$ 
> 证明：
> 根据式 (3.2)，可知对于任意的 $\mathbf x\in \Omega$， $\mathcal Q (\mathbf x) - \mathcal Q (\mathbf x_i)$ 的值 
> ($\mathcal Q(\mathbf x) - \mathcal Q(\mathbf x_i) = \ln\frac {P (x_i\mid x_1, \dots, x_{i-1}, x_{i+1},\dots, x_n)}{P (0\mid x_1,\dots, x_{i-1}, x_{i+1},\dots, x_n)}$)
> 仅依赖于 $x_i$ 本身的值和 $i$ 的邻居站点的取值

Without loss of generality, we shall only consider site. 1 in detail. We then have, from equation (3.3), 
> 以站点 1 为例，考虑 $\mathcal Q (\mathbf x) - \mathcal Q (\mathbf x_1)$，根据式 (3.3)，有：

$$
\begin{align}
\mathcal Q(\mathbf x) - \mathcal Q(\mathbf x_1) &= x_1\Big\{G_1(x_1) + \sum_{2\le j\le n}x_j G_{1,j}(x_1, x_j) + \sum_{2\le j < k \le n}x_jx_k G_{1,j,k}(x_1, x_j, x_k)\}\\
&+\dots+x_2x_3\dots x_n G_{1, 2, \dots, n}(x_1, x_2, \dots, x_n)\Big\}.
\end{align}
$$

Now suppose site $l(\neq1)$ is not a neighbour of site 1. Then $\mathcal Q(\mathbf{x})\!-\!\mathcal Q(\mathbf{x}_1)$ must be independentof $x_{l}$ for all $\mathbf{x}\!\in\!\Omega$ .Putting $x_{i}=0$ for ${i\neq1}$ or $l$, we immediately see that $G_{1,l}(x_{1},x_{l})=0$ on ${\Omega}$ .Similarly, by other suitable choices of $\mathbf{x},$ it is easily seen successively that all $3$ -, $4$ -, ..., ${n}-$ variable ${G}$ functions involving both ${x_{1}}$ and $x_{l}$ must be null.
> 考虑一个不是站点 1 的邻居的站点 $l (\ne 1)$
> 则对于任意的 $\mathbf x \in \Omega$，应该满足 $\mathcal Q (\mathbf x) - \mathcal Q (\mathbf x_1)$ 独立于 $x_l$，也就是 $x_l$ 的取值应该不影响 $\mathcal Q(\mathbf x) - \mathcal Q(\mathbf x_1)$ 的取值
> 因此，根据上文的 $\mathcal Q (\mathbf x) - \mathcal Q (\mathbf x_1)$ 的展开形式，容易知道其中的 $G_{1, l}(x_1, x_l) = 0$ 在 $\Omega$ 上恒成立，同理，任意带有 $x_1, x_l$ 的高阶 $G$ 函数也应该为空

The analogous result holds for any pair of sites which are not neighbours of each and hence, in general, $G_{i,j,\dots,s}$ can only be non-null if the sites $i,j,...,s$ forms a clique.
> 同理，对于任意的站点对 $x_i, x_j$，都应该满足二者不是邻居时，带有 $x_i, x_j$ 的 $G$ 函数应该为空
> 因此，一般地说， $G$ 函数 $G_{i, j, \dots, s}$ 当且仅当它涉及的站点 $i, j, \dots, s$ 两两都是邻居，也就是构成了一个团时，才非空

On the other hand, any set of ${G}$ -functions gives rise to a valid probability distribution $P(\mathbf{x})$ which satisfies the positivity condition. Also since $\mathcal Q(\mathbf{x})\!-\!\mathcal Q(\mathbf{x}_{i})$ depends only upon ${x}_{l}$ if there is a non-null ${G}$ -function involving both $x_{i}$ and $x_l$ it follows that the same is true of $P(x_{i}|\,x_{1},...,x_{i-1},x_{i+1},...,x_{n})$ .This completes the proof. 
>另一方面，任何一组 $G$ 函数都可以生成一个满足正性条件的有效概率分布 $P(\mathbf{x})$ 
>此外，由于 $\mathcal Q(\mathbf{x}) - \mathcal Q(\mathbf{x}_{i})$ 仅依赖于满足为 $x_i$ 的邻居站点的 ${x}_{l}$，这对于条件概率 $P(x_{i}|x_{1},...,x_{i-1},x_{i+1},...,x_{n})$ 来说也成立
>证毕

We now consider some simple extensions of the theorem. Suppose firstly that the variate s can take a den umer ably infinite set of values. Then the theorem still holds if in the second part, we impose the added restriction that the $G$ -functionsbe chosen such that $\Sigma \exp \mathcal Q({\bf x})$ is finite, where the summation is over all $\mathbf x\in{{\Omega}}$ 
> 考虑 HC 定理的拓展：
> 我们假定变量此时可以取无限个值，此时，要让 HC 定理仍然成立，我们需要为能选取的 $G$ 函数施加限制：选取的 $G$ 函数应该使得 $\sum_{\mathbf x\in \Omega} \exp \mathcal Q (\mathbf x)$ 是有限的

Similarly, if the variates each have absolutely continuous distributions and we interpret $P(\pmb{x})$ and allied quantities as probability densities, the theorem holds provided we ensure that exp $Q(\mathbf{x})$ is integrable over all $\mathbf{x}.$ These additional requirements must not be taken lightly, as we shall see by examples in Section 4. 
>类似地，如果每个变量都有连续分布，并且我们将  $P(\mathbf {x})$ 和相关量解释为概率密度函数，此时，要让 HC 定理仍然成立，需要确保 $\exp \mathcal Q(\mathbf{x})$ 在所有 $\mathbf{x} \in \Omega$ 上可积，也就是要求 $\int_{\mathbf x \in \Omega} \exp \mathcal Q (\mathbf x)$ 是有限的

Finally, we may consider the case of multivariate rather than univariate site variables. In particular, suppose that the random vector at site $\pmb{i}$ has $\nu_{i}$ components. Then we may replace that site by $\nu_{i}$ notional sites, each of which is associated with a single component of the random vector. An appropriate system of neighbours may then be constructed and the univariate theorem be applied in the usual way. We shall not consider the multi- variate situation any further in the present paper. 
>最后，我们考虑多元而非单一变量的情况：
>特别地，假设在站点 $i$ 的随机向量有 $v_i$ 个分量，则我们可以用 $v_i$ 个虚拟站点替换该站点，每个虚拟站点与随机向量的一个分量相关联，然后构造适当的邻居系统，并按常规方式应用单变量情况下的定理
>在本文中，我们不再进一步考虑多元情况。

As a straightforward corollary to the theorem, it may easily be established that for any given Markov field 

$$
\begin{align}
P(X_i = x_i, X_j = x_j,\dots, X_s = x_s \mid \text{all other site values})
\end{align}
$$

depends only upon $x_{i},x_{j},...,x_{s}$ and the values at sites neighbouring sites $i,j,\dots,s.$ In the Hammersley-Clifford terminology, the local and global Markovian properties are equivalent. 
> 该定理的一个直接的引理就是：
> 对于任意给定的 Markov 场，条件概率分布 $P(X_i = x_i, X_j = x_j,\dots, X_s = x_s \mid \text{all other site values}$) 仅依赖于 $x_i, x_j, \dots, x_s$ 的值以及 $i, j,\dots, s$ 的所有邻居站点的值 
> ( 该条件概率分布可以视作联合概率分布 $P (\mathbf x)$ 的一个边际分布，而根据 HC 定理，联合概率分布 $P (\mathbf x)$ 仅和所有的团相关，不构成团的 $G$ 函数都为空，故显然该边际分布也仅决定于它相关的团，也就是 $i, j,\dots, s$ 和它们的邻居节点 )
> 在 HC 定理成立的条件下，局部和全局的 Markov 性质是等价的

In practice, we shall usualy find that the sites occur in a finite region of Euclidean space and that they often fall naturally into two sets: those which are internal to the field, it is quite likely that we are able to make reasonable assumptions concerning the conditional distribution associated with each of the internal sites but that problems arise at the boundary of the system. Such problems may usually be by-passed by considering the joint distribution of the internal site variables conditional upon fixed (observed) boundary values. We need then only specify the neighbours and associated conditional probability structure for each of the internal sites in order to define uniquely the above joint distribution. This is a particularly useful approach for lattice systems. 
> 实际中，我们会发现站点通常出现在欧几里得空间的一个有限区域内，并且它们往往会自然地分成两组：内部的站点和边界上的站点
> 我们一般可以为每个内部站点的条件分布做出合理的假设，但这些假设在系统的边界处则一般不成立
> 该问题通常可以通过考虑内部站点变量在固定的（观测到的）边界值条件下的联合分布来绕过，因此，我们只需要为每个内部站点指定邻居及其相关的条件概率结构，即可唯一定义上述联合分布
> 该方法对点阵系统尤其有用

The positivity condition remains as yet unconquered and it would be of con- side r able theoretical interest to learn the effect of its relaxation. Ontheotherhand, it is probably fair to say that the result would be of little practical significance in the analysis of spatial interaction with given site locations 
>分布的正性条件这一要求目前仍未解决，了解其松弛的效果在理论上会非常有趣
>另一方面，可以说，放宽这一条件在给定站点位置的空间交互分析中的实际意义不大

Finally, we note that, for discrete variables, a further proof of the Hammersley- Clifford theorem has been given by Grimmett (1973). This is apparently based upon the Mobius inversion theorem (Rota, 1964). Other references on the specification of Markov fields include Averintsev (1970), Preston (1973) and Sherman (1973). 
>最后，我们注意到，对于离散变量，Grimmett (1973) 给出了 Hammersley-Clifford 定理的另一个证明。这显然是基于 Möbius 反演定理（Rota, 1964）
>有关马尔可夫场规范化的其他参考文献包括 Averbintsev (1970)、Preston (1973) 和 Sherman (1973)

# 4 Some Spatial Schemes Associated with The Exponential Family
In thenext twosections, w ebecome more specific in our discussion of spatial schemes. The present section deals with a particular subclass of Markov fields and with some of the models which are generated by it, whilst Sections 5.1, 5.2 and 5.3 are more concerned with practical aspects of conditional probability models. In Section 5.4, the simultaneous autoregressive approach (Mead, 1971; Ord, 1974) to finite spatial systems is discussed, again from the conditional probability viewpoint. Finally, in Section 5.5, stationary auto-normal models on the infinite regular lattice aredefined and compared with the stationary simultaneous autoregressions of Whittle (1954). 

In the remainder of this paper, we shall use the function $\pmb{p_{i}(.)}$ to denote the con- ditional probability distribution (or density function) of $X_{i}$ given all other site values. Thus $\pmb{p_{i}(.)}$ is a function of $\pmb{x_{i}}$ and of the values at sites neighbouring site i. Wherever possible, the arguments of $\mathfrak{p}_{\mathfrak{i}}(.)$ will be omitted. 

## 4.1. Auto-models 

Given $\pmb{n}$ sites, labelled $1,..., n,$ and theset ofneighbours for each, wehave seen in Section 3 how the Hammers ley-Clifford theorem generates the class of valid pro ba- bility distributions associated with the site variables $X_{1},..., X_{n}$ .Within this general framework, we shall in Section 4.2 consider particular schemes for which $Q ({\bf x})$ is well defined and has the representation 

$$
Q(\mathbf{x})=\!\!\sum_{1\leqslant i\leqslant n}\!\!x_{i}\,G_{i}(x_{i})\!+\!\!\sum_{1\leqslant i<j\leqslant n}\!\!\beta_{i,j}\,x_{i}\,x_{j},
$$ 

where $\beta_{i, j}=0$ unlesssites $i$ and $j$ are neighbours of each other. Such schemes will be termedauto-models. 

In order to motivate this definition, it is convenient to consider the wider formu lationbelow. Suppose we make the following assumptions. 
Assumption 1. The probability structure of the system is dependent only upon contributions from cliques containing no more than two sites. That is, when well defined, the expansion (3.3) becomes 

$$
\begin{array}{r}{\mathcal{Q}(\mathbf{x})=\underset{\substack{1\leqslant i\leqslant n}}{\sum}x_{i}\,G_{i}(x_{i})+\underset{\substack{1\leqslant i<j\leqslant n}}{\sum}x_{i}\,x_{j}\,G_{i,j}(x_{i},x_{j}),}\end{array}
$$ 

where $G_{i, j}\!\!\left (.\right)\!\!=\! 0$ unless sites $_i$ and $j$ are neighbours 

Assumption 2. The conditional probability distribution associated with each of the sites belongs to the exponential family of distributions (Kendall and Stuart, 1961, p. 12). That is, for each $i,$ 

$$
\ln p_{i}(x_{i};\,...)=A_{i}(.)\,B_{i}(x_{i})\!+\!C_{i}(x_{i})\!+\!D_{i}(.),
$$ 

where the functions $\pmb{{\cal B}_{i}}$ and $c_{i}$ are of specified form and $A_{i}$ and $D_{i}$ are functions of the values at sites neighbouring site $i.$ Avalidchoiceof $\pmb{A}_{i}$ determines the type of dependence upon neighbouring site values and $\pmb{{\mathscr{D}}_{\pmb{\mathscr{i}}}}$ is then the appropriate normalizing function. 

It is shown in Section 4.3 that as a direct consequence of Assumptions 1 and 2. $\pmb{A}_{i}$ mustsatisfy 

$$
A_{i}(\cdot)\!\equiv\!\alpha_{i}\!+\!\sum_{j=1}^{n}\beta_{i,j}\,B_{j}(x_{j}),
$$ 

where $\beta_{j, i}{\equiv}\beta_{i, j}$ and $\beta_{i, j}=0$ unless sites $i$ and $^j$ are neighbours of each other. Hence, it follows, when appropriate, that $G_{i, j}$ in equation (4.2) has the form 

$$
G_{i,j}(x_{i},x_{j})\!\equiv\!\beta_{i,j}H_{i}(x_{i})\,H_{j}(x_{j}),
$$ 

where $x_{i}H_{i}(x_{i})=B_{i}(x_{i})\!-\! B_{i}\! (0)$ . Thus we generate the class of auto-models by making the additional requirement that, for each $i,$ the function $B_{\ell}$ is linear in $\pmb{x_{i}}$ 

Superficially, auto-models might appear to form quite a useful subclass of Markov fields. Assumption 1 is not only satisfied for any rectangular lattice “nearest- neighbour"scheme but can alsobe taken as a fairly natural starting point in much wider lattice and non-lattice situations. Further, the linearity of $\pmb{{\cal B}_{i}}$ issatisfiedbythe most common members of the exponential family. However, the assumptions are, in fact, so restrictive, as seen through equation (4.4), that they often produce models which, in the end result, are devoid of any intuitive appeal at all. In Section 4.2, a rangeof auto-models has been included and hopefully illustrates both endsof the spectrum. Practical applications of twoof the modelswillbe discussed inlater sections. 

It is clear that, in terms of equation (4.1), auto-models have conditional prob- ability structure satisfying 

$$
p_{i}(x_{i};\ldots)/p_{i}(0;\ldots)=\exp\Big[x_{i}\Big\{G_{i}(x_{i})+\sum_{j=1}^{n}\beta_{i,j}\,x_{j}\Big\}\Big],
$$ 

where again $\beta_{j,\pmb{i}}\!\!\equiv\!\beta_{\pmb{i},\pmb{j}}$ and $\beta_{i, j}\!=\! 0$ unless sites $\pmb{i}$ and $^j$ are neighbours of each other The models can further be classified according to the form which $\pmb{p_{i}(.)}$ takes and this leads to the introduction of terms such as auto-normal, auto-logistic and auto-binomial to describe specific spatial schemes. 

In the subsequent discussion, it will be assumed, unless otherwise stated, that any parameters $\beta_{i, j}$ are at least subject to the conditions following equation (4.6). Ranges of summation will be omitted wherever possible and these should then be apparent by comparison with equation (4.1) or (4.6). 
## 4.2. Some Specific Auto-models 
### 4.2.1. Binaryschemes 
For any finite system of binary (zero-one) variables, the only occasions upon which agivennon-null $G$ -functioncan contribute to $Q ({\pmb x})$ in the expansion (3.3) are those upon which each of its arguments is unity. We may therefore replace all non-null ${G}$ -functions by single arbitrary parameters, without any loss of generality, and this leads to the multivariate logistic models of Cox (1972). 
> 对于仅含有二元 (0,1) 变量的有限系统，其中任意一个给定的非空 $G$ 函数可以对展开式 (3.3) 中的 $Q (\pmb x)$ 做出贡献的情况就是所有的相关 $x_i$ 都取为 1，因此 $G$ 函数实际的贡献情况仅有零和一个固定的常数，我们进而可以在不失一般性的情况下将 $G$ 函数替换为任意的单个参数
> 由此就得到了多元的逻辑斯蒂模型

One would hope in practice that only a fairly limited number of non-zero parameters need to be included. In particular, if the only non-zero parameters are those associated with cliques consisting of single sites and of pairs of sites, we have an auto-logistic model for which wemaywrite 

$$
\begin{array}{r}{Q(\mathbf{x})=\sum\alpha_{i}x_{i}\!+\!\sum\!\sum\beta_{i,j}\,x_{i}\,x_{j}.}\end{array}
$$ 
It follows that 

$$
p_{i}(.)=\exp{\{x_{i}(\alpha_{i}+\sum\beta_{i,j}\,x_{j})\}}/\{1+\exp{(\alpha_{i}+\sum\beta_{i,j}\,x_{j})}\},
$$ 
analogous to a classical logistic model (Cox, 1970, Chapter 1), except that here the explanatory variables are themselves observations on the process. 

### 4.2.2. Gaussian schemes 

In many practical situations, especially those arising in plant ecology, it is reason- able to assume that the joint distribution of the site variables (plant yields), possibly after suitable transformation, is multivariate normal. It is evident that any such scheme is an auto-normal scheme: In particular, we shall consider schemes for which 

$$
\begin{array}{r}{p_{i}(.)=(2\pi\sigma^{2})^{-\frac{1}{2}}\exp{[-\frac{1}{2}\sigma^{-2}\{x_{i}-\mu_{i}-\sum\beta_{i,j}(x_{j}-\mu_{j})\}^{2}]}.}\end{array}
$$ 

Using the factorization (2.2) or otherwise, this leads to the joint density function, 

$$
P(\mathbf{x})=(2\pi\sigma^{2})^{-\frac{1}{2}n}\big|\,\mathbf{B}\big|^{\frac{1}{2}}\exp\{-\frac{1}{2}\sigma^{-2}(\mathbf{x}-\boldsymbol{\upmu})^{\mathrm{T}}\,\mathbf{B}(\mathbf{x}-\boldsymbol{\upmu})\},
$$ 

where $\pmb{\upmu}$ is the $\pmb{n}\!\times\! 1$ vector of arbitrary finite means, $\pmb{\mu_{i:}}$ and $\pmb{\mathfrak{B}}$ is the ${\pmb n}\times{\pmb n}$ matrix $(i, j)$ $\begin{array}{r l}{-\beta_{i, j}.}\end{array}$ Clearly $\mathbf{B}$ is symmetric but of course we also require $\mathbf{B}$ to be positive definite in order for the formulation to be valid. 

At this point, it is perhaps worth indicating the distinction between the process (4.9) defined above, for which 

$$
E(X_{i}|{\mathrm{~all~other~site~values}})=\mu_{i}\!+\!\sum\beta_{i,j}(x_{j}\!-\!\mu_{j}),
$$ 

and the process defined by the set of $\pmb{n}$ simultaneous autoregressive equations, typically 

$$
\begin{array}{r}{X_{i}=\mu_{i}\!+\!\sum\beta_{i,j}(X_{j}\!-\!\mu_{j})\!+\!\varepsilon_{i},}\end{array}
$$ 

where $\pmb{\varepsilon_{1}},...,\pmb{\varepsilon_{n}}$ are independent Gaussian variates, each with zero mean and variance $\pmb{\sigma^{2}}$ . In contrast to equation (4.10), the latter process has joint probability density function, 

$$
P(\mathbf{x})=(2\pi\sigma^{2})^{-\frac{1}{4}n}\big|\,\mathbf{B}\big|\exp\{-\frac{1}{2}\sigma^{-2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}}\,\mathbf{B}^{\mathrm{T}}\,\mathbf{B}(\mathbf{x}-\boldsymbol{\mu})\},
$$ 

where B is defined as before. Also, it is no longer necessary that $\beta_{j, i}{\equiv}\beta_{i, j},$ Onlythat $\mathbf{B}$ should be non-singular. Further aspects of simultaneous autoregressive schemes will be discussed in Sections 5.4 and 5.5. 
### 4.2.3. Auto-binomial schemes 

Supposethat $X_{i}$ has a conditional binomial distribution with associated “sample size” $\pmb{m_{i}}$ and “probability of success" $\theta_{i}$ which is dependent upon the neighbouring site values. Then $H_{i}(x_{i})\!\equiv\! 1$ and, under Assumption 1, the odds of “success" to "failure" must satisfy 

$$
\begin{array}{r}{\ln\{\theta_{i}/(1-\theta_{i})\}=\alpha_{i}\!+\!\sum\beta_{i,j}\,x_{j}.}\end{array}
$$ 

When $m_{i}=1$ for all $i,$ . we again have the auto-logistic model. 

### 4.2.4. Auto-Poisson schemes 

Supposethat $X_{i}$ has a conditional Poisson distribution with mean $\pmb{\mu_{i}}$ dependent upon the neighbouring site values. Again $\pmb{H_{i}}(x_{i})\!\!=\! 1$ and, under Assumption 1, $\pmb{\mu_{i}}$ is subject to the form 

$$
\begin{array}{r}{\mu_{i}=\exp{(\alpha_{i}\!+\!\sum{\beta_{i,j}}x_{j})}.}\end{array}
$$ 

Further, since the range of $X_{i}$ is infinite, we must ensure that exp $Q ({\bf x})$ is summable over $\mathbf{x}$ .We show below that this requires the further restriction $\beta_{i, j}\!\leqslant\! 0$ for all $_i$ and $j.$ Wehave 

$$
\begin{array}{r}{Q(\mathbf{x})=\sum\{\alpha_{i}x_{i}\!-\!\ln{(x!)}\}\!+\!\sum\!\sum\beta_{i,j}x_{i}x_{j}.}\end{array}
$$ 

Clearly exp $Q ({\bf x})$ must be summable when each $\beta_{i, j}=0$ so the same holds when each $\beta_{i, j}\!\leqslant\! 0$ ：To show the necessity of the condition, we consider the distribution of the pair of variates $(X_{\mathfrak{z}}, X_{\mathfrak{z}})$ given that all other site values are equal to zero. The odds of the realization $(x_{1}, x_{2})$ to the realization $(0,0)$ are then 

$$
\exp Q(x_{1},x_{2},0,...,0)=\exp{(\alpha_{1}x_{1}+\alpha_{2}x_{2}+\beta_{1,2}x_{1}x_{2})}/{(x_{1}!\,x_{2}!)},
$$ 

fornon-negative integers $\pmb{x_{1}}$ and $\pmb{x_{2}}$ .We certainly require that the sum of this quantity over all $x_{1}$ and $\pmb{x_{2}}$ converges and this is only true when $\beta_{\mathbf{1},\mathbf{\hat{z}}}\!\leqslant\! 0$ .Similarly, we require $\beta_{i, j}\!\leqslant\! 0$ for all $i$ and $j.$ This restriction is severe and necessarily implies a “competitive' rather than“co-operative" interaction between auto-Poisson variates. 

### 4.2.5. Auto-exponential schemes 

Suppose that $X_{i}$ has a conditional negative exponential distribution with mean $\pmb{\mu_{i}}$ dependent upon the values at sites neighbouring site $i.$ Once more $H_{i}(x_{i})\!\equiv\! 1$ and, under Assumption 1, $\pmb{\mu_{i}}$ must take the form $(\alpha_{i}\!+\!\sum\beta_{i, j}\, x_{j})^{-1},$ .The scheme is valid provided $\scriptstyle\alpha_{i}>0$ and $\beta_{i, j}\!\geqslant\! 0$ but the conditional probability structure appears to lack any form of intuitive appeal. Analogous statements hold for all gamma-type distributions. 

## 4.3. Proof of Equation (4.4) 

In order to establish the result (4.4) under Conditions 1 and 2, we begin by assumingthatln $\pmb{p_{i}}(0;...)$ is well behaved, relaxing this condition later. For con- venience, weshall write $\pmb{A}_{i}$ and $D_{i}$ of equation (4.3) as functions of 

$$
(x_{1},...,x_{i-1},0,x_{i+1},...,x_{n})
$$ 

although in reality they depend only upon the values at sites neighbouring site i. SinceIn $p_{i}(0;...)$ iswellbehaved, $Q (\mathbf{x})$ is well defined (under the positivity condition) and has the representation (4.2) according to Assumption 1. Equations (4.2) and (4.3) may now be related through equation (3.2). Putting $x_{j}=0$ for all $j\!\neq\! i,$ weobtain, foreach $i,$ 
$$
x_{i}\,G_{i}(x_{i})=A_{i}(0)\left\{B_{i}(x_{i})\!-\!B_{i}(0)\right\}+C_{i}(x_{i})-C_{i}(0).
$$ 

Now suppose sites 1 and 2 are neighbours of each other. Putting $x_{j}=0$ for $_{j\geqslant3}$ and again using equation (3.2) to link (4.2) and (4.3), we obtain, for $i\,{=}\, 1$ 

$x_{1}G_{1}(x_{1})+x_{1}x_{2}G_{1,2}(x_{1}, x_{2})=A_{1}(0, x_{2}, 0,..., 0)\,\{B_{1}(x_{1})-B_{1}(0)\}+C_{1}(x_{1})-C_{1}(0)$ and, for $\pmb{i}=\pmb{2};$ 

$$
x_{2}\,G_{2}(x_{2})+x_{1}\,x_{2}\,G_{1,2}(x_{1},x_{2})=A_{2}(x_{1},0,...,0)\,\{B_{2}(x_{2})-B_{2}(0)\}+C_{2}(x_{2})-C_{2}(0).
$$ 

Combining these two equations with (4.14), we deduce that 

$$
x_{1}\,x_{2}\,G_{1,2}(x_{1},x_{2})=\beta_{1,2}\{B_{1}(x_{1})\!-\!B_{1}(0)\}\,\{B_{2}(x_{2})\!-\!B_{2}(0)\},
$$ 

where $\beta_{\mathbf{1},\mathbf{2}}$ is a constant. More generally, if sites $\pmb{i}$ and $^j$ are neighbours and $i\!<\! j,$ 

$$
x_{i}\,x_{j}\,G_{i,j}(x_{i},x_{j})=\beta_{i,j}\{B_{i}(x_{i})\!-\!B_{i}(0)\}\{B_{j}(x_{j})\!-\!B_{j}(0)\}.
$$ 

The result (4.4) is easily deduced from (4.14) and (4.15) 

The condition that ln $\pmb{p_{i}}(0;...)$ is well behaved is not satisfied by all members of the exponential family. However, in cases where In $\pmb{p_{i}}(0;...)$ degenerates as, for example, with most gamma distributions, we may use a simple transformation on the $X_{i}{\bf\ddot{s}}$ to affirm that (4.4) still holds. Suppose, without loss of generality, that $_{0<p_{i}(1;\ldots)<\infty}$ and in that case let $Y_{i}=\ln{X_{i}}$ at each site. Then the conditional probability structure of theprocess $\{Y_{i}\}$ also lies within the exponential family of distributions but there is no degeneracy associated with the value $Y_{i}=0$ .Theprevious arguments may then be applied to show that $\pmb{A}_{i}$ still satisfies equation (4.4) 

# 5 Some Two-dimentional Spatial Schemes and Their Applications
## 5.1. Finite Lattice Schemes 
In practice, the construction of conditional probability models on a finite regular lattice is simplified by the existence of a fairly natural hierarchy inthe choice of neighbours for each site. For simplicity, and because it occurs most frequently in practice, we shall primarily discuss the rectangular lattice with sites defined by integer pairs $(i, j)$ over some finite region. Where the notation becomes a little unwieldy, the reader may find it helpful to sketch and label the sites appropriately. The simplest model which allows for local stochastic interaction between the variates $X_{i, j}$ is then the first-order Markov scheme (or“nearest-neighbour"model) in which each interior site $(i, j)$ is deemed to have four neighbours, namely $(i\!-\! 1,\! j),\: (i\!+\! 1,\! j),\: (i,\! j\!-\! 1)$ and $(i, j+1)$ .If, as suggested in Section 3, we now interpret $Q ({\pmb x})$ as being concerned with the distribution of theinternal sitevariables conditional upon given boundary values, the representation (3.3) in the Hammersley-Clifford theorem can be written 

$$
\begin{array}{r l}&{Q(\mathbf{x})=\sum x_{i,j}\,\phi_{i,j}(x_{i,j})\!+\!\sum x_{i,j}\,x_{i+1,j}\,\psi_{1,i,j}(x_{i,j},x_{i+1,j})}\\ &{\qquad\qquad\qquad+\sum x_{i,j}\,x_{i,j+1}\,\psi_{2,i,j}(x_{i,j},x_{i,j+1}),}\end{array}
$$ 

where $\{\phi_{i, j}\},\;\{\psi_{1, i, j}\}$ and $\{\psi_{\mathbf{2}, i, j}\}$ are arbitrary sets of functions, subject to the sum- mabilityof $Q ({\bf x})$ , and the ranges of summation in (5.1) are such that each clique, involving at least one site internal to the system, contributes a single term to the representation. Writing $(x, t, t^{\prime}, u, u^{\prime})$ for the partial realization 
$$
(x_{i,j},x_{i-1,j},x_{i+1,j},x_{i,j-1},x_{i,j+1}),
$$ 

the conditional probability structure at the site $(i, j)$ isgivenby 

$$
\begin{array}{r}{p_{i,j}(x;\,t,t^{\prime},u,u^{\prime})=\exp\{f_{i,j}(x;\,t,t^{\prime},u,u^{\prime})\}/{\sum}\exp\{f_{i,j}(z;\,t,t^{\prime},u,u^{\prime})\},}\end{array}
$$ 

where 

$$
f_{i,j}(.)=x\{\phi_{i,j}(x)+t\psi_{1,i-1,j}(t,x)+t^{\prime}\psi_{1,i,j}(x,t^{\prime})+u\psi_{2,i,j-1}(u,x)+u^{\prime}\psi_{2,i,j}(x,u^{\prime})\}
$$ 

and the summation, or integration in the case of continuous variates, extends over all values $z,$ possibleat $(i, j)$ . In any given practical situation, the $\phi\cdot$ ， $\psi_{\mathbf{1}^{-}}$ and $\psi_{\pmb{\mathscr{s}}}$ -functions can then be chosen to give an appropriate distributional form for $\pmb{p_{i, j}(.)}$ . For the scheme to be spatially homogeneous, these functions must be independent of position $(i, j)$ on the lattice. We then have the special case discussed by Besag (1972a). If, further, $\begin{array}{r}{\pmb{\psi_{1}}=\pmb{\psi_{2}},}\end{array}$ the scheme is said to be isotropic. 

The idea of a first-order scheme may easily be extended to produce higher-order schemes. Thus a second-order scheme allows $(i, j)$ tohave the additional neighbours $(i{-}1, j{-}1),$ $(i\!+\! 1, j\!+\! 1),$ $(i{-}1, j{+}1)$ and $(i\!+\! 1,\! j\!-\! 1),$ whilstathird-orderscheme further includes the sites $(i\!-\! 2,\! j), (i\!+\! 2,\! j), (i,\! j\!-\! 2)$ and $(i, j+2)$ .To obtain $Q ({\pmb x})$ we merely add a contributory term for each clique whichinvolves atleast one siteinternal to the system. For example, a homogeneous second-order scheme has 

$$
\begin{array}{r l}&{Q(\mathbf{x})=\sum x_{i,j}\,\phi(.)+\sum x_{i,j}\,x_{i+1,j}\,\psi_{1}(.)+\sum x_{i,j}\,x_{i,j+1}\,\psi_{2}(.)}\\ &{\qquad\qquad\qquad+\sum x_{i,j}\,x_{i+1,j+1}\,\psi_{3}(.)+\sum x_{i,j}\,x_{i+1,j-1}\,\psi_{4}(.)}\\ &{\qquad\qquad\qquad+\sum x_{i,j}\,x_{i+1,j}\,x_{i,j+1}\,\xi_{1}(.)+\sum x_{i,j}\,x_{i+1,j}\,x_{i+1,j+1}\,\xi_{2}(.)}\\ &{\qquad\qquad\qquad+\sum x_{i,j}\,x_{i+1,j}\,x_{i+1,j-1}\,\xi_{3}(.)+\sum x_{i,j}\,x_{i,j+1}\,x_{i+1,j+1}\,\xi_{4}(.)}\\ &{\qquad\qquad\qquad+\sum x_{i,j}\,x_{i+1,j}\,x_{i,j+1}\,x_{i+1,j+1}\,\delta(.),}\end{array}
$$ 

where the arguments of each function are its individual multipliers; thus 

$$
\delta(.){\equiv}\,\delta(x_{i,j},x_{i+1,j},x_{i,j+1},x_{i+1,j+1})
$$ 

and so on. In specific examples, the apparent complexity of the expressions may be very much reduced. However, it is felt that, unless the variables are Gaussian, third- and higher-order schemes will almost always be too unwieldy to be of much practical use. 

First- and second-order schemes may easilybe constructed for other lattice systems in two or more dimensions. Amongst these, the plane triangle lattice is of particular interest, firstly because it frequently occurs in practice andsecondly because a first-order scheme on a triangular lattice, for which each internal site has six neighbours, is likely to be more realistic than the corresponding scheme on a rectangular lattice. 

## 5.2 Specific Finite Lattices Schemes
### 5.2.1. Binary data 

It is clear from equation (5.1) that the homogeneous first-order scheme for zero-one variables on a rectangular lattice is given by 

$$
\begin{array}{r}{Q(\mathbf{x})=\alpha\sum x_{i,j}+\beta_{1}\sum x_{i,j}\,x_{i+1,j}+\beta_{2}\sum x_{i,j}\,x_{i,j+1},}\end{array}
$$ 
where $\alpha,\beta_{1}$ and $\beta_{\mathfrak{z}}$ are arbitrary parameters. This leads to the conditional probability structure 

$$
p_{i,j}(x;\,t,t^{\prime},u,u^{\prime})=\frac{\exp{\{x\{\alpha+\beta_{1}(t+t^{\prime})+\beta_{2}(u+u^{\prime})\}}\}}{1+\exp{\{\alpha+\beta_{1}(t+t^{\prime})+\beta_{2}(u+u^{\prime})\}}},
$$ 

in the notation of Section 5.1. The scheme is necessarily auto-logistic 

For the second-order scheme, there are cliques of sizes three and four and there is no longer any need for the scheme to be auto-logistic. Thus, if we additionally $(v, v^{\prime}, w, w^{\prime})$ $(x_{i-1, j-1}, x_{i+1, j+1}, x_{i-1, j+1}, x_{i+1, j-1}),$ that $\pmb{p_{i, j}(.)}$ is now given by an expression similar to (5.3) but with the terms in curly brackets $\{\}$ replacedby 

$$
\begin{array}{r l}&{\alpha\!+\!\beta_{1}(t\!+\!t^{\prime})\!+\!\beta_{2}(u\!+\!u^{\prime})\!+\!\gamma_{1}(v\!+\!v^{\prime})\!+\!\gamma_{2}(w\!+\!w^{\prime})}\\ &{\phantom{x x x x x x x x x x x x x x x x x x x x x x x x x}+\xi_{1}(t u\!+\!u^{\prime}w\!+\!w^{\prime}t^{\prime})\!+\!\xi_{2}(t v\!+\!v^{\prime}u^{\prime}\!+\!u t^{\prime})\!+\!\xi_{3}(t w\!+\!w^{\prime}u\!+\!u^{\prime}t^{\prime})}\\ &{\phantom{x x x x x x x x x x x x x x}+\xi_{4}(t u^{\prime}\!+\!u v\!+\!v^{\prime}t^{\prime})\!+\!\eta(t w\!+\!t^{\prime}u^{\prime}v^{\prime}\!+\!t u^{\prime}w\!+\!t^{\prime}u w^{\prime}).}\end{array}
$$ 

The scheme is only auto-logistic if the $\xi_{-}$ and $\pmb{\eta}$ parameters are all zero. 

Incidentally, .this is a convenient point at which to mention the first-order binary scheme on a triangular lattice, for this can be thought of as a scheme on a rectangular latticein which $(i, j)$ has the six neighbours $(i\!-\! 1,\! j),\;(i\!+\! 1,\! j),\;(i,\! j\!-\! 1),\;(i,\! j\!+\! 1),$ $(i{-}1, j{-}1)$ and $(i\!+\! 1, j\!+\! 1)$ . The homogeneous first-order scheme is thus obtained from (5.4) by putting $\gamma_{\tt2}=\xi_{\tt1}=\xi_{\tt3}=\eta=0$ .The scheme is auto-logistic only if, in addition, $\pmb{\xi_{2}}=\pmb{\xi_{4}}=\pmb{0}.$ 

Regarding applications of the rectangular lattice models, we shall, in Section 7.1 analyse Gleaves's Plantago lanceolata data using the first- and second-order isotropic auto-logistic schemes. However, none of the sets of data, cited in Section 1, appears to provide a convincing demonstration of low-order auto-logistic behaviour. It is hoped that more “appropriate' sets of data will become available in the future. A number of remarks are made in this context. Firstly, in order to carry out a detailed statistical analysis of spatial interaction, rather than merely test for independence or estimate the parameters of a model, it is usually the case that fairly extensive data are required. For example, spatial models have been fitted to Greig-Smith's data by Bartlett (1971b), using the spectral approximation technique of Bartlett and Besag (1969), and by Besag (1972c), using the coding technique of Section 6. The respective models are similar, though not equivalent, and each appears to give a fairly satisfactory fit. However, the last statement should be viewed with some scepticism since the goodness-of-fit tests available for such a small system $({\pmb24}\times{\pmb24})$ arevery weak. This will be illustrated by the more detailed analysis of Gleaves's data. 

Secondly, it is stressed that the lower-order homogeneous schemes, under dis- cussion here, have been specifically designed with local stochastic interaction in mind; in particular, it is unreasonable to apply them in situations where there is evidence of gross heterogeneity over the lattice. For example, the hop plant data of Freeman (1953) display a fairly clear dichotomy between the upper and lower halves of the lattice, the former being relatively disease free (Bartlett, 1974). Thirdly, the use of lattice schemes on Greig-Smith's and Gleaves's data is, of course, an artifice: as remarked in Section 1, these examples are really concerned with spatial pattern rather than spatial interaction. Furthermore, as is well known, the size of quadrat used when collecting such data can profoundly influence the results of the subsequent statistical analysis. Incidentally, from a numerical viewpoint, it is most efficient to arrange the quadrat size so that O's and 1's occur with approximately equal frequency. An alternative procedure might be to adopt some sort of nested analysis (Greig-Smith, 1964). 
The criticisms above are not intended to paint a particularly gloomy picture but merely to point out some limitations of the models. It is maintained that auto-logistic analyses can be useful in practice; the models, having once been established, are easy to interpret and, even when rejected, can aid an understanding of the data and of the underlying spatial situation. 

### 5.2.2. Gaussian variables 

Ithas already been stated in Section 2.2 that auto-normal schemes are of relevance to many ecological situations. For a finite rectangular lattice system, two homo- gene o us schemes are of particular practical interest. They are thefirst-orderscheme forwhich $X_{i, j},$ givenallother sitevalues, is normally distributed with mean 

$$
\alpha+\beta_{1}(x_{i-1,j}+x_{i+1,j})+\beta_{2}(x_{i,j-1}+x_{i,j+1})
$$ 

and constant variance $\pmb{\sigma^{2}}$ and the second-order scheme for which $X_{i, j},$ given all other site values, is normally distributed with mean 

$$
\begin{array}{r l}&{\alpha+\beta_{1}(x_{i-1,j}+x_{i+1,j})+\beta_{2}(x_{i,j-1}+x_{i,j+1})}\\ &{\quad+\gamma_{1}(x_{i-1,j-1}+x_{i+1,j+1})+\gamma_{2}(x_{i-1,j+1}+x_{i+1,j-1})}\end{array}
$$ 

andconstantvariance, $\pmb{\sigma^{2}}$ .Such schemes can, for example, be used for the analysis of crop yields in uniformity trials when, perhaps through local fluctuations in soil fertility or the influence of competition, it is no longer reasonable to assume statistical independence. This is illustrated in Section 7, using the classical wheat plots data of MercerandHall (1911). 

In more general experimental situations, it is possible to setup in homogeneous auto-normal schemes to account for stochastic interaction between the variables. For example, one can replace $\pmb{\alpha}$ in the expressions (5.5) and (5,6) by $\propto_{i, j},$ allowingthisto depend deterministic ally upon the treatment combination at $(i, j)$ ,intheusualway. Such schemes can still be analysed by the coding methods which will be discussed in Section 6. It is suggested that there is a need for further research here, particularly into the use of specially constructed experimental designs which take advantage of both themodel and the codinganalysis. 

At this point, it is perhaps worth while anticipating the results of Section 5.4 in order tore-emphasize the distinction between thepresent approach and thatbased upon simultaneous auto regressive schemes. Removing means for simplicity, suppose we consider the schemedefinedby the equations, 

$$
X_{i,j}=\beta_{1}\,X_{i-1,j}+\beta_{1}^{\prime}\,X_{i+1,j}+\beta_{2}\,X_{i,j-1}+\beta_{2}^{\prime}\,X_{i,j+1}+\varepsilon_{i,j}\,
$$ 

over some finite region, with appropriate adjustments attheboundary of thesystem, wherethe $\boldsymbol{\varepsilon_{i, j}}\,{\bf\hat{s}}$ are independent Gaussian error variates with common variance. The analogous scheme on a finite triangular lattice has been examined by Mead (1967). It might well be assumed that, at least when $\beta_{1}^{\prime}=\beta_{1}$ and ${\beta}_{\mathfrak{z}}^{\prime}={\beta}_{\mathfrak{z}},$ the conditional expectation structure of the process (5.7) would tally with the expression (5.5), putting ${\pmb{\alpha}}={\bf0}$ .However, this is not at all the case: in fact, the process (5.7) has conditional expectation structure defined by 
$$
\begin{array}{r l r}{\lefteqn{=(\beta_{\bf1}\!+\!\beta_{\bf1}^{\prime})\,(x_{i-1,j}\!+\!x_{i+1,j})\!+\!(\beta_{\bf2}\!+\!\beta_{\bf2}^{\prime})\,(x_{i,j-1}\!+\!x_{i,j+1})}}\\ &{}&{-(\beta_{\bf1}\,\beta_{\bf2}^{\prime}\!+\!\beta_{\bf1}^{\prime}\,\beta_{\bf2})\,(x_{i-1,j-1}\!+\!x_{i+1,j+1})\!-\!(\beta_{\bf1}\,\beta_{\bf2}\!+\!\beta_{\bf1}^{\prime}\,\beta_{\bf2}^{\prime})\,(x_{i-1,j+1}\!+\!x_{i+1,j-1})}\\ &{}&{-\,\beta_{\bf1}\,\beta_{\bf1}^{\prime}(x_{i-2,j}\!+\!x_{i+2,j})\!-\!\beta_{\bf2}\,\beta_{\bf2}^{\prime}(x_{i,j-2}\!+\!x_{i,j+2}),}&{\quad{\mathrm{(1)}}}\end{array}
$$ 

consistent with a special case (since there are only four independent $\beta\cdot$ parameters rather than six) in the class of third-order auto-normal schemes. The peculiar con- d it ional expectation structure arises because of the bilateral nature of the auto- regression; that is, in contrast with the unilateral time series situation, $\pmb{\varepsilon}_{\pmb{i},\pmb{j}}$ isnot independent of the remaining right-hand-side variables in (5.7). Some previous comments concerning the conditional probability structure of simultaneously defined schemes have been made by Bartlett (1971b), Besag (1972a) and Moran (1973a, b) 

## 5.3. Non-lattice Systems 

We now turn to the construction of models for which there are a finite number of irregularly distributed, but co-planar, sites. Asstated inSection1, we shall onlybe concerned here with the distribution of the site variables $X_{i}\, (i=1,..., n),$ giventhe knowledge of their respective locations, and not with an investigation of the spatial pattern associated with the sites themselves. The first problem is in the choice of neighbours for each site. If the sites comprise a finite system of closed irregular regions in the form of a mosaic, such as counties or states in a country, it will usually be natural to include as neighbours of a given site $i,$ those sites to which it is adjacent. In addition, it may be felt necessary to include more remote sites whose influence is, nevertheless, felt tobe of direct consequence to the site i variable. 

Alternatively, if the sites constitute a finite set of irregularly distributed points in the plane, a rather more arbitrary criterion of neighbourhood must be adopted. However, the situationcan bereduced to the preceding one if we canfind an intuitively plausible method of defining appropriate territories for each site. One possibility is to construct the Voronyi polygons (or Dirichlet cells) for the system. The polygon of site iis defined by the union of those points in the plane which lie nearer to site i than to any other site. This formulation clearly produces a unique set of non-overlapping convex territories, often capable of a crude physical or biological interpretation. It appears to have been first used in practice by Brown (1965) in a study of local stand density in pine forests. Brown interpreted the polygon of any particular tree as defining the “area potentially available' to it. If, in general, two sites are deemed to be neighbours only when their polygons are adjacent, it is evident that each internal site must have atleast three neighbours and that cliques of more than four sites cannot occur. With this definition, Brown's pine trees each have approximately six neigh- bours, as might commonly be expected in situations where competitive influences tend to produce a naturally or artificially imposed regularity on the pattern of sites. A slight, but artificial, reduction in complexity occurs if we further stipulate that in order for two sites to be neighbours, the line joining them must pass through their common polygon side. Cliques can then contain no more than three members. 

Mead (1971) and Ord (1974) have each used the Voronyi polygons of a system to setupandexaminesimultaneousautoregressiveschemessuchas (4.12). 
Whatever the eventual choice of the neighbourhood criterion, we may derive the most general form for the available conditional probability structure in any particular situation by applying theHammersley-Clifford theorem. Some specific schemes have been given in Section 4.2. In particular, we discuss the use of the auto-normal scheme (4.9). The first task is to reduce the dimensionality of the parameter space by relating the $\pmb{\mu_{i}}\,\mathbf{\hat{s}}$ and $\beta_{\pmb{i},\pmb{j}}\,\pmb{\dot{s}}$ in some intuitively reasonable way. In the case of point sites, suppose the Voronyi polygons are constructed and that $d_{i, j}$ represents the distance between neighbouring sites $\pmb{i}$ and $j$ whilst $l_{i, j}$ represents the length of their common polygon side. It is then often feasible to relate each $\pmb{\mu_{i}}$ and non-zero $\beta_{i, j}$ to the corresponding $d_{i, j}$ and $l_{i, j}.$ The symmetry property of the $\beta_{i, j}\,\mathbf{\dot{s}}$ arises naturally. Specific suggestions for use in the scheme (4.12) have been made by Mead (1971) in the context of plant competition models. Analogous suggestions are made by Ord (1974) in a geographical context. These suggestions could equally be implemented in the case of conditional probability models. 

## 5.4. Simultaneous Auto regressive Schemes 

Atvarious stages, reference has been made to the simultaneous autoregressive schemes (4.12) and (5.7). We now determine their associated conditional probability structure since this is a facet of the models which has occasionally been misunderstood in the past. In fact, it is convenient to widen the formulation somewhat by considering schemes of the form 

$$
\sum b_{i,j}X_{j}\!=\!Z_{i}
$$ 

for $i=1,..., n,$ or, in matrix notation, ${\bf B X}={\bf Z},$ where $\pmb{\mathbb{B}}$ is an ${\pmb n}\!\times\!{\pmb n}$ non-singular matrix and $\mathbf{z}$ is a vector ofindependent continuous random variables. In practice, the matrix ${\bf\nabla}\cdot{\bf B}$ will often be fairly sparse. We neither demand that the $z_{i}\mathbf{\dot{s}}$ are identically distri- buted nor that they are Gaussian. Let $\pmb{f}\!\!\left (.\right)$ denote the density function of $Z_{i}$ Then $X_{\pmb{v}}..., X_{\pmb{n}}$ have joint density, 

$$
P(\mathbf{x})=\left\|\mathbf{B}\right\|f_{1}(\mathbf{b_{1}^{T}}\mathbf{x})f_{2}(\mathbf{b_{2}^{T}}\mathbf{x})\ldots f_{n}(\mathbf{b_{n}^{T}}\mathbf{x}),
$$ 

where $\mathbf{b}_{i}^{\mathbf{T}}$ denotestheith. rowofB. The conditional probability structure at site $\pmb{i}$ is then immediately obtainable from equation (3.2) or an analogue thereof. In particular, the result (5.8) is easily deduced. 

More generally, suppose we say that site $\scriptstyle{j\neq k}$ is acquainted with site $\pmb{k}$ if and only if, for some $i,$ $\pmb{b}_{\pmb{i},\pmb{j}}\!\neq\! 0$ and $\pmb{b_{i, k}\!\neq\! 0}$ ; that is, if and only if at least one of the equations (5.9) depends upon both $X_{j}$ and $X_{k}$ .Then it is easily seen that the conditional distribution of $X_{k}$ can at most depend upon the values at sites acquainted with site $\pmb{k}$ That is, the neighbours of any site are included in its acquaintances. 

In agivenpractical situation, the setsof acquaintances and neighbours of a site may well be identical but this is not necessarily so. Suppose, for example, we consider theprocess 

$$
X_{i,j}=\beta_{1}\,X_{i-1,j}+\beta_{2}\,X_{i,j-1}+\beta_{3}\,X_{i-1,j-1}+\varepsilon_{i,j},
$$ 

defined, for convenience, over a ${\pmb{p}}\!\times\!{\pmb{q}}$ finite rectangular torus lattice, where the $\pmb{\varepsilon_{i, j}}\cdot\mathbf{\tilde{s}}$ are independent Gaussian variables with zero means and common variances. Then

 $(i, j)$ has acquaintances $(i\!-\! 1, j),\: (i\!+\! 1, j),\: (i, j\!-\! 1),\: (i, j\!+\! 1),$ $(i{-}1, j{+}1)$ $(i\!+\! 1, j\!-\! 1)

$ $(i{-}1, j{+}2)$ and $\mathbf{\Phi}(i\!+\! 1, j\!-\! 2)$ provided $\beta_{\mathbf{1}},\,\beta_{\mathbf{2}}$ and $\beta_{\mathfrak{s}}$ are non-zero. In general, these sites will also constitute the set of neighbours of $(i, j)$ . However, suppose $\beta_{\mathfrak{s}}=\beta_{\mathfrak{s}}\beta_{\mathfrak{s}}$ then the sites $(i{-}1, j{+}1)$ and $(i\!+\! 1,\! j\!-\! 1)$ arenolonger neighboursof $(i, j)$ . In fact, we shall find in Section 6 that this result provides a useful approach to problems of statistical inference, for it enables unilateral approximations to first-order auto- normal schemes to be constructed. 
## 5.5. Stationary Auto-normal Processes on an Infinite Lattice 

We define a stationary Gaussianprocess $\{X_{i, j}\colon i, j=0,\pm1,\ldots\}$ tobeafinite-order auto-normal process on the infinite rectangular lattice if it has autocovariance generating function (a.c.g.f.) equal to 

$$
K(1\!-\!\sum\!\Sigma\,b_{k,l}z_{1}^{k}z_{2}^{l})^{-1}\!,\quad1\!\le\!\!\!\!\!\!\!\!\sum\!\!\!\!\!
$$ 

where (i) only a finite number of the real coefficients $\pmb{b}_{\pmb{k}\pmb{l}}$ are non-zero, (ii) $\begin{array}{r}{\pmb{b}_{\pmb{0},\pmb{0}}=\pmb{0},}\end{array}$ $\scriptstyle\pmb{b}_{-\pmb{k},-\pmb{l}}=\pmb{b}_{\pmb{k},\pmb{l}}$ $\Sigma\Sigma b_{k, l}z_{1}^{k}z_{2}^{l}\!<\! 1$ $\left|z_{1}\right|=\left|\tilde{z}_{2}\right|=1.$ $\pmb{K}$ $X_{i, j}^{'}$ $-\infty$ $+\infty$ unless otherwise stated. 

The existence of such processes was demonstrated by Rosanov (1967) and in certain special cases by Moran (1973a, b). Moran (1973b) included a simplified account of some of Rosanov's paper and we shall use this below to discuss the structure of the schemes. Firstly, however, we reintroduce the concept of neighbour- hood. That is, for a stationary Gaussian process with a.c.g.f. of the form (5.11), we definethesite $(i{-}k_{, j{-}l})$ tobe a neighbour of $(i, j)$ ifandonly if $\pmb{b}_{\pmb{k},\pmb{l}}\!\!\neq\! 0$ .Wenow show that this accordswith our finite system definition. 

It follows from equation (5.11) that, provided $|r|{+}|s|{>}0,$ 

$$
\rho_{r,s}=\Sigma\Sigma b_{k,l}\rho_{r-k,s-l},
$$ 

where $\pmb{\rho_{r, s}}$ denotes the autocorrelation of lags $\pmb{r}$ and $\pmb{\mathscr{s}}$ in $_i$ and $j,$ respectively. Now let $\{\varepsilon_{i, j}\colon\stackrel{.}{i, j}=0,\pm\, 1,\ldots\}$ be a doubly infinite set of variates defined by 

$$
X_{i,j}=\alpha\!+\!\sum\!\sum b_{k,l}X_{i-k,j-l}\!+\!\varepsilon_{i,j},
$$ 

where $\alpha=(1\!-\!\Sigma\Sigma b_{k,\! l})\mu$ and $\pmb{\mu}=\pmb{E}(\pmb{X_{i, j}})$ . Then the $\pmb{\varepsilon_{i, j}}\mathbf{\dot{s}}$ are stationary Gaussian variables with zeromeans and commonvariances $\pmb{\sigma^{2}}$ ,say. Also the equations (5.12) imply that $\pmb{\varepsilon}_{i, j}$ and $X_{i^{\prime}, j^{\prime}}$ are uncorrelated provided $\scriptstyle{\left|i-i^{\prime}\right|+\left|j-j^{\prime}\right|>0}$ This result together with (5.13), implies the following: given the values at any finite set of sites which includes the neighbours of $(i, j), X_{i, j}$ has conditional mean $\alpha\!+\!\Sigma\Sigma b_{k, l}x_{i-k, j-l}$ and conditional variance $\pmb{\sigma^{2}}$ independent of the actual surroundingvalues. 

Thus, wehave confirmed that the present criterion of neighbourhood is consistent with that for finite systems and that the properties of stationary, infinite lattice auto-normal schemes are in accordance with those of the homogeneous, finite lattice schemes. In particular, we may define first-, second- and higher-order schemes analogous to those appearing in Section 5.2.2. 

Finally, we make some remarks concerning the infinite lattice schemes proposed by Whittle (1954). Removing means for simplicity, Whittle considered simultane- ously defined stationary processes in the class 

$$
\begin{array}{r}{\Sigma\Sigma\,a_{k,l}\,X_{i-k,j-l}=Z_{i,j},}\end{array}
$$ 

where $\{Z_{i, j}\colon i, j=0,\pm1,\ldots\}$ is a doubly infinite set of independent Gaussian variates, each with zero mean and variance $\pmb{v},$ The scheme (5.14) has a.c.g.f. 

$$
v(\textstyle\sum\sum a_{k,l}z_{1}^{k}\,z_{2}^{l})^{-1}(\textstyle\sum\sum a_{k,l}z_{1}^{-k}\,z_{2}^{-l})^{-1}.
$$ 
If thenumberofnon-zero coefficients $\pmb{a}_{k, l}$ is finite, we shall refer to (5.14) as being a “finite-order" Whittle scheme. It is clear from (5.15) that any such scheme has a finite-order auto-normal representation. The converse is in general untrue: for example, eventhefirst-orderauto-normal scheme does not have a finite-order simultaneous auto regressive representation unless $\beta_{1}$ Or $\beta_{\mathbf{\hat{z}}}=\mathbf{0}$ .One is therefore led topose the following question: whenusing finite-order schemesinthe statistical analysis of spatial data, are there a priori reasons for restricting attention to the par- ticular finite-order schemes generated by (5.14) or should the wider range of auto- normal models be considered? Note that when the number of sites is finite, thereis. for Gaussian variates, a complete, but somewhat artificial, correspondence between the classes of simultaneous and conditional probability models 

A further point which is relevant, whether the number of sites is finite or infinite, is illustrated by the following example. The most general bilateral scheme used by Whittle in examining the wheat plot data of Mercer and Hall (1911) was the infinite latticeanalogueof (5.7), namely 

$$
X_{i,j}=\beta_{1}\,X_{i-1,j}+\beta_{1}^{\prime}\,X_{i+1,j}+\beta_{2}\,X_{i,j-1}+\beta_{2}^{\prime}\,X_{i,j+1}+Z_{i,j}.
$$ 

Firstly, this process again has the rather peculiar conditional expectation structure (5.8), but secondly, as noted by Whittle, there is anambiguity inthe identity of parameters. That is, if we interchange $\beta_{1}$ and $\beta_{1}^{\prime}$ and also $\beta_{\mathfrak{z}}$ and $\beta_{\mathfrak{z}}^{\prime},$ we obtain a processwith the identical probability structure. For thescheme (4.12), thesameholds true if we interchange $\pmb{\mathbb{B}}$ and $\mathbf{B^{T}}$ This seems rather unsatisfactory. In the time series situation, the problem does not arise if one invokes the usual assumption that past influences future, not vice versa. With spatial schemes, the problem can be overcome if we are content to examine merely the conditional probability structure oftheprocess, givenby equation (5.8) in the present context. It is suggested that such considera- tions againsupport theuseof the conditional probability approach to spatial systems. A further comment appears in Section 7 of the paper. 

We note that first- and second-order stationary auto-normal schemes on the infinite lattice were first proposed by Lévy (1948) but that, as remarked by Moran (1973a), existence was assumed without formal justification. Moran himself concen- t rates almost exclusively on the first-orderscheme. 

# 6 Statistical Analysis of Lattice Systems
In this section, we propose some methods of parameter estimation and some goodness-of-fit tests applicable to spatial Markov schemes defined over a rectangular lattice. The methods may be extended to other regular lattice systems, notably the triangular lattice, and, in part, to some non-lattice situations. In practice, it would appear that, amongst lattice schemes, it is the ones of first and second order which are of most interest and it is these upon which we shall concentrate. It has already been established in Section 2 that, generally speaking, a direct approach to statistical inference through maximum likelihood is intractable because oftheextremely awkward natureof thenormalizing function. We therefore seek alternative techniques. The exceptional case occurs when thevariateshave anauto-normalstructure, forwhich thenormalizing function may often be evaluated numerically without toomuch effort, eveninsomenon-lattice situations. Each of the methods will be illustrated in Section 7 ofthepaper. 
## 6.1. Coding Methods on the Rectangular Lattice 

We assume, in the notation of Section 4, that the conditional distributions, $\pmb{p_{i, j}(.)}$ ,are of a given functional form but collectively contain a number of unknown parameters whosevalues are tobe estimatedon thebasisof a single realization, $\mathbf{x},$ ofthesystem. Coding methods of parameter estimation were introduced by Be sag (1972c), in the context of binary data, but they are equally available in more general situations. 

In order to fit a first-order scheme, we begin by labelling the interior sites of the lattice, alternately $\times$ and ., as shown in Fig. 1. It is then immediately clear that, 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/2ff574b59c0c68cee6116bc776c0f0427d4347616537877b91b5cc74a6d5d8ec.jpg) 
FIG. 1. Coding pattern for a first-order scheme. 

according to the first-order Markov assumption, the variables associated with the $\times$ sites, given the observed values at all other sites, are mutually independent. This results inthe simple conditional likelihood, 

$$
\begin{array}{r}{\prod p_{i,j}(x_{i,j};\,x_{i-1,j},x_{i+1,j},x_{i,j-1},x_{i,j+1}),}\end{array}
$$ 

forthe $\times$ sitevalues, the product being taken over all $\times$ sites. Conditional maximum- likelihood estimates of the unknown parameters can then be obtained in the usual way. Alternative estimates may be obtained by maximizing the likelihood function for the 

.site values conditional upon the remainder (or, that is, using a unit shift in the coding pattern). The two procedures are likely tobehighly dependent but, neverthe less, it is reasonable, in practice, to carry out both and then combine the results appropriately. 

-" In order to estimate the parameters of a second-order scheme, we may code the internal sites as shown in Fig. 2. Again considering the joint distribution of the $\times$ Ssite variables given the . site values, we may obtain conditional maximum-likelihood 

![](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/a93e826f56a7673c00de3646917c7dea2ab6e61ad1cfc019f028befc65ff7c97.jpg) 
FIG. 2. Coding pattern for a second-order scheme. 

estimates of the parameters. By performing shifts of the entire coding framework over the lattice, four sets of estimates are available and these may then be combined appropriately. 

Using the coding methods, we may easily construct likelihood-ratio tests to examine thegoodnessof fitof particular schemes. Here, we stress three points. Firstly, it is highly desirable that thewider classof schemes againstwhichwe test is one which has intuitive spatial appeal, otherwise the test islikely tobe weak. This is, of course, an obvious comment but one which, in the limited statistical work on spatial analysis, has sometimes been neglected. Secondly, the two maximized likeli- hoods we obtain must be strictly comparable. For example, if the fit of a scheme of first order is being examined against one of second order, the resulting likelihood- ratio testwill onlybevalid if both the schemeshavebeen fitted tothe same set of data-—-that is, using the Fig. 2 coding in each case. Thirdly, there will be more than one test available (under shifts in coding) and these should be considered collectively. Whilst precise combination of the results may not be possible, they can usuallybe amalgamated in some conservative way. These points will be illustrated in Section 7. 
The efficiency of coding techniques can to a limited extent be investigated following the methods of Ord (1974). Also of relevance are the papers by Ogawara

 (1951), Williams (1952) andHannan (1955a, b) andsome comments by Placket t

 (1960, p. 121), all on coding methods for Markov chains. The coding techniques will not, in general, be fully efficient but their great advantage lies in their simplicity and flexibility. Some results will be reported elsewhere but further investigation of the techniques is still required. 

## 6.2. Unilateral Approximations on the Rectangular Lattice 

An alternative estimation procedure for homogeneous first-order spatial schemes involves the construction of a simpler process which has approximately the required probability structure but which is much easier to handle. The approach is similar (equivalent for stationary auto-normal schemes) to that of Bartlett and Besag (1969). We begin by defining the set of predecessors of any site $(i, j)$ in the positive quadrant to consist of those sites $(k, l)$ on the lattice which satisfy either (i) $\imath\!<\! j$ or (ii) $\imath\!=\! j$ and $\pmb{k}\!<\!i.$ We may then generate a unilateral stochastic process $\{X_{i, j}\colon i{>}0, j{>}0\}$ in the positive quadrant by specifying the distribution of each variable $X_{i, j}$ conditional upon the values at sites which are predecessors of $(i, j)$ . In practice, we shall allow the distribution of $X_{i, j}$ to depend only on a limited number of predecessor values. Such a process is a natural extension of a classical one-dimensional finite auto- regressive time series into two dimensions and is well defined if sufficient initial values are given. Special cases of such schemes have been discussed by Bartlett and Besag (1969), Bartlett (1971b) and Besag (1972b). By a judicious choice of the unilateral scheme, we may obtain a reasonable approximation to a given first-order spatial scheme. The more predecessor values we allow $X_{i, j}$ to depend upon, the better the approximation can be made. The great advantage of a unilateral scheme is that its likelihood function is easily written down and parameter estimation may be effected by straightforward maximum likelihood. 

As the simplest general illustration, we consider unilateral processes of the form 

$$
P(x_{i,j}|{\mathrm{~all~precondition}})=q(x_{i,j};\,x_{i-1,j},x_{i,j-1}).
$$ 

The joint probability distribution of the variables $X_{i, j}\left (1\!\leqslant\! i\!\leqslant\! m, 1\!\leqslant\! j\!\leqslant\! n\right)$ isgivenby 

$$
\prod_{i=1}^{m}\prod_{j=1}^{n}q(x_{i,j};\,x_{i-1,j},x_{i,j-1})
$$ 

and, hence, for any interior site $(i, j)$ we have, in the notation of Section 4, the bilateral structure 

$$
\frac{p_{i,j}(x;\ldots)}{p_{i,j}(x^{*};\ldots)}\!=\!\frac{q(x;\,t,u)q(t^{\prime};\,x,w^{\prime})q(u^{\prime};\,w,x)}{q(x^{*};\,t,u)q(t^{\prime};\,x^{*},w^{\prime})q(u^{\prime};\,w,x^{*})}.
$$ 

That is, the conditional distribution, $P (x_{i, j}|$ all other site values), depends not only upon $x_{i-1, j},\, x_{i+1, j},\, x_{i, j-1}$ and $x_{i, j+1}$ but also upon $\pmb{x_{i-1, j+1}}$ and $\pmb{x_{i+1, j-1}}$ .Nevertheless, the primary dependence is upon the former set of values and, by a suitable choice of ${\pmb q}(.),$ we may use the unilateral process as an approximation to a given homogeneous first-order spatial scheme. For a better approximation, we may consider unilateral processesof theform, 
$$
P(x_{i,j}|{\mathrm{~all~precessor}})=q(x_{i,j};\,x_{i-1,j},x_{i,j-1},x_{i+1,j-1})
$$ 

and so on. The method will be illustrated for an auto-normal scheme in Section 7. 

## 6.3. Maximum-likelihood Estimation for Auto-normalSchemes 

We begin by considering the estimation of the parameters in an auto-normal scheme of the form (4.9) but subject to the restriction $\pmb{\upmu}=\pmb{0}$ .We assume thatthe dimensionality of the parameter space is reduced through $\mathbf{B}$ having a particular structure and that $\pmb{\sigma^{2}}$ is both unknown and independent of the $\beta_{i, j}\,\mathrm{\bf\dot{s}}$ For a given realization $\pmb{x}_{:}$ the corresponding likelihood function is then equal to 

$$
(2\pi\sigma^{2})^{-{\frac{1}{2}}n}\left|\left.\mathbf{B}\right|^{\frac{1}{2}}\exp{(-{\frac{1}{2}}\sigma^{-2}\mathbf{x}^{\mathrm{T}}\mathbf{B}\mathbf{x})}.\right.
$$ 

It follows that the maximum-likelihood estimate of $\pmb{\sigma^{2}}$ will be given by 

$$
{\hat{\pmb{\sigma}}}^{\pmb{\hat{\imath}}}={\pmb{n}}^{-1}{\bf x}{\hat{\bf B}}{\bf x},
$$ 

once ${\hat{\pmb{\mathscr{B}}}},$ the maximum-likelihood estimate of ${\pmb{\mathfrak{B}}},$ has been found. Substituting (6.2) into (6.1), we find that $\pmb{\hat{\mathbf{B}}}$ may be obtained by minimizing 

$$
-n^{-1}\ln\left|\mathbf{B}\right|\!+\!\ln\left(\mathbf{x}^{\mathbf{T}}\mathbf{B}\mathbf{x}\right)\!.
$$ 

The problem of implementing maximum-likelihood estimation therefore rests upon the evaluation of thedeterminant, $|{\pmb B}|$ .We now examine how this relates to existing research into simultaneous autoregressions. 

Supposethen thatwe temporarily abandon the auto-normalmodel aboveand decide instead to fit a simultaneous scheme of the form (4.12), again subject to $\pmb{\upmu}=\pmb{0}$ andwith $\pmb{\mathfrak{B}}$ having the same structure as in (6.1). Provided (6.1) is valid so is the present, but different, scheme. The likelihood function now becomes 

$$
(2\pi\sigma^{2})^{-\frac{1}{2}n}\left|\mathrm{\bfB}\right|\exp\left(-\mathrm{\bfB}\sigma^{-2}\mathrm{\bfX}^{\mathrm{T}}\mathrm{\bfB}^{\mathrm{T}}\mathrm{\bfB}\mathrm{\bfX}\right)
$$ 

and the new estimate of $\pmb{\mathfrak{B}}$ must be found by minimizing 

$$
-2n^{-1}\ln\left|\mathbf{B}\right|\!+\!\ln(\mathbf{x}^{\mathrm{{T}}}\mathbf{B}^{\mathrm{{T}}}\mathbf{B}\mathbf{x}).
$$ 

Again the only real difficulty centres upon the evaluation of the determinant $|\pmb{\mathscr{B}}|,\mathbf{a}$ point which we may, in a sense, now turn to advantage. Suppose that we wish to ft the auto-normal scheme associated with (6.1) to a given set of data. Then it follows that we may use existing approaches to fitting simultaneous autoregressive schemes provided that these can cope with the likelihood function (6.4). Indeed with minor modifications, we may use any existing computer programs. It is probably fair to say that thus far the simultaneous and conditional probability schools have tended to suggest the same structure for B in a given problem. This, together with the previous remarks, implies that it would be relatively straightforward to conduct a useful comparative investigation of the two approaches for some given sets of data. 

As regards minimizing (6.5), computational progress has been made by Mead (1967) on small (triangular) lattices and by Ord (1974) in non-lattice situations where the number of sites is fairly limited (about 40 or less) and there are only one or two unknown parameters determining B. The reader is referred to their papers for further details. As regardslarge lattices for which we may sometimes view thedata as being a partial realization of a stationary Gaussian infinite lattice process, wemay use thesemi-analytical result ofWhittle (1954) which is summarizedbelow. 
Whittle showed, for the simultaneous autoregression (5.14), that, given a partial realization of the process over $\pmb{n}$ sites, theterm $n^{-1}|{\bf n}|{\bf B}|$ in (6.5) canbe approximated $z_{1}^{0}z_{2}^{0}$ 

$$
\begin{array}{r}{\ln(\Sigma\Sigma\,a_{k,l}z_{1}^{k}z_{2}^{l}).}\end{array}
$$ 

There is no complication if the variates have equal, but non-zero, means. Thus, in order to fit a particular auto-normal scheme of the form (5.11) from a partial realiza tionoftheprocessover $\pmb{n}$ sites, we needtominimize (6.3), where $\bar{n^{-1}}|{\bf n}|{\bf B}|$ is. the absolute term in the power series expansion of 

$$
\ln(1\!-\!\sum\!\Sigma b_{k,l}z_{1}^{k}z_{2}^{l})
$$ 

and where, neglecting boundary effects. 

$$
\mathbf{x}^{\mathrm{T}}\,\mathbf{B}\mathbf{x}=C_{0,0}\!-\!\sum\!\sum b_{k,l}C_{k,l}
$$ 

and $C_{k, l}$ denotes the empirical autocovariance of lags $k$ and. $\imath$ in $i$ and $j,$ respectively (cf. Whittle, 1954). 

For example, with the first-order scheme, analogous to (5.5), we minimize 

$$
\begin{array}{r}{-\Lambda(\mathfrak{g})\!+\!\ln{(C_{0,0}\!-\!2\beta_{1}C_{1,0}\!-\!2\beta_{2}C_{0,1})},}\end{array}
$$ 

where $\Lambda ({\pmb\beta})$ is the absolute term in the power series expansion of 

$$
\ln\{1\!-\!\beta_{1}(z_{1}\!+\!z_{1}^{-1})\!-\!\beta_{2}(z_{2}\!+\!z_{2}^{-1})\}.
$$ 

With the second-order scheme, analogous to (5.6), we minimize 

$$
\begin{array}{r}{-\Lambda(\mathfrak{g},\gamma)\!+\!\ln{(C_{0,0}\!-\!2\beta_{1}C_{1,0}\!-\!2\beta_{2}C_{0,1}\!-\!2\gamma_{1}C_{1,1}\!-\!2\gamma_{2}C_{1,-1})},}\end{array}
$$ 

where $\Lambda (\pmb{\upbeta},\pmb{\upgamma})$ is the absolute term in thepower series expansionof 

$$
\ln\{1\!-\!\beta_{1}(z_{1}\!+\!z_{1}^{-1})\!-\!\beta_{2}(z_{2}\!+\!z_{2}^{-1})\!-\!\gamma_{1}(z_{1}\,z_{2}\!+\!z_{1}^{-1}z_{2}^{-1})\!-\!\gamma_{2}(z_{1}\,z_{2}^{-1}\!+\!z_{1}^{-1}z_{2})\!\}.
$$ 

The absolute terms can easily be evaluated for given parameter values by appropriate numerical Fourier inversion. The expression (6.3) may be minimized by, for example, the Newton-Raphson technique. Convergence, in the limited work thus far, has been extremely rapid. A numerical example is included in Section 7. 

Finally, we note an analogybetween thefitting of stationary auto-normal schemes in the analysis of spatial data and the fitting of auto regressive schemes in classical time-series analysis. That is, considering a particular scheme in the class (5.11), suppose that the corresponding auto correlations are denoted by $\pmb{\rho}_{\pmb{k},\pmb{r}}$ Then the effect of large-sample maximum-likelihood estimation is to ensure perfect agreement between $\pmb{\rho_{k, l}}$ and the corresponding sample autocorrelation $\pmb{r_{k, l}}$ whenever $\pmb{b}_{\pmb{k},\pmb{\ell}}\!\neq\! 0$ in $\pmb{\rho_{1,0}}=\pmb{r_{1,0}}$ and $\pmb{\rho_{0,1}}=\pmb{r_{0,1}}$ : For the second-order scheme, we additionally fit $\rho_{1,1}=r_{1,1}$ and $\pmb{\rho_{1,-1}}=\pmb{r_{1,-1}}$ autoregressions. This may suggest that the auto-normal schemes are, in fact, a more natural extension of classical temporal autoregressions tospatial situations. 
# 7 Numerical Examples
## 7.1. Auto-logistic Analysis of Plantago lanceolata Data 

Observations on Plantago lanceolata weremade over an apparently homogeneous area of lead-zinc tailings in defunct mine workings, Treloggan, Flintshire. The sampling frame consisted of a transect $10\times940$ withgridsize $2\,{\tt c m}\times2\,{\tt c m}$ .Counts weremadeof the number of seedlings and number of adults in each of the 9,400 quadrats. As the grid size is rather small, the analysis below is based upon the pooled dataandonly presence/absenceofplantsin eachquadratisconsidered. Thelatter simplification results in little loss of information. The data were kindly collected by Dr J. T. Gleaves of the Department of Botany, University of Liverpool. 

Welet $x_{i, j}=0/1$ denote absence/presence of Plantago lanceolata in the $(i, j)$ th quadrat. There is no reason to expect asymmetry in the system and we shall therefore only consider auto-logistic schemes of theform 

(a) isotropicfirst-order scheme for which 

$$
P(x_{i,j}|\;\mathrm{all~other~values})=\frac{\exp{\{(\alpha+\beta y_{i,j})\,x_{i,j}\}}}{1+\exp{(\alpha+\beta y_{i,j})}},
$$ 

$y_{i, j}=x_{i-1, j}+x_{i+1, j}+x_{i, j-1}+x_{i, j+1},

$ (b) isotropicsecond-orderschemefor which 

$$
P(x_{i,j}|{\mathrm{~all~other~values}})={\frac{\exp\left\{\left(\alpha+\beta y_{i,j}+\gamma z_{i,j}\right)x_{i,j}\right\}}{1+\exp\left(\alpha+\beta y_{i,j}+\gamma z_{i,j}\right)}},
$$ 

where, in addition, $z_{i, j}=x_{i-1, j-1}\!+x_{i+1, j+1}\!+x_{i-1, j+1}\!+x_{i+1, j-1}.$ 

Note that a full isotropic second-order scheme would involve two further parameters, corresponding to cliques of triples and quadruples, respectively. A further comment appearslater. 

Parameter estimates can be obtained for schemes (a) and (b) using the coding techniques described in Section 6.1. For scheme (a), estimates for $\pmb{\alpha}$ and $\beta_{;}$ under Fig. 1 codings, are given in Table 1. The respective observed and expected frequencies 

![TABLE 1 Auto-logistic analysis of Plant ago lance ol at a data: parameter estimates for scheme (a) under Fig. 1 codings ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/eb266c6efb85652c2992321994ac84a7a4357e73c830e4c45af7e9b88bd18fd2.jpg) 

appear in Tables 2 and 3, and these may be used to conduct simple chi-squared goodness-of-fit tests for the scheme. The resulting statistics, each on 3 degrees of freedom (d.fr.), are 4.84 and 2.90, respectively, suggesting a satisfactory fit. However, interpreting these tests as likelihood-ratio tests, we see that the wider hypothesis, against which we are examining scheme (a), is itself rather specialized (and un- attractive) sinceit still assumes independence between the columns within thebody of thetables. Thus, the above typeof testis notrecommended although its use may be unavoidablewherethereisashortageofdata. 
![TABLE 2 Auto-logistic analysis of Plant ago lance ol at a data:observed and expected frequencies forfirstanalysisinTable1 ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/0fae7fc6726ad85bb339058994c74c4214d2e7d488cadab54d6354e4fe2359c0.jpg) 

![Auto-logistic analysis of Plantago lanceolata data: observed and expected frequencies forsecondanalysisinTable1 ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/5e43b038a72a76751e2109a1c1d975aa9da40a3824c4dbe58a9fbc666a8c7376.jpg) 

For scheme (b), the parameter estimates, under Fig. 2 codings, are given in Table 4. Scheme (a) may also be fitted under these codings and this is done in Table 5. Hence, we may examine the goodness of fit of scheme (a) within the class (b) in each of the four cases, using the usual likelihood-ratio test. The resulting statistics, each on 1 d.fr., are 49.9, 60.6, 49.4 and 48·6! It is now clear that scheme (a) is hopelessly inadequate in describing the system and that we must be wary of non-significant results in the type of test previously described. Incidentally, it is of interest to note the correspondence between Tables 1 and 5. 

A typical set of observed and expected frequencies under scheme (b) isrecorded in Table 6 and may be used to produce a simple chi-squared test for scheme (b) itself. The results corresponding to each of the four analyses are given in Table 7 and, combining these conservatively, leads totherejection of scheme (b) atthe5per cent level. Thus, despite the preceding remarks, it is still. quite possible for the weaker formof testtobeuseful inpractice. 

It might be of interest to fit the full second-orderorhigher-order schemes to the data but, once fitted, a large number of cells would be found empty or nearly empty in the contingency tables, resulting in the invalidity of the usual distributional assumptions concerning goodness-of-fit tests. Furthermore, the objections (see Section 5.2.1) to Markov lattice models for quadrat schemes suggest that a more detailed analysis on the present lines is unlikely to be particularly helpful. 
![TABLE4 Auto-logistic analysis of Plant ago lance ol at a data: parameterestimatesforscheme $(b)$ under Fig.2 codings ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/5b335f3e2d529b91c3a80d98ab28d1b98155da580bba6ea4492cff95ffffedc0.jpg) 

![TABLE 5 Auto-logistic analysis of Plant ago lance ol at a data: parameter estimates for scheme(a) under Fig.2 codings ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/0532dfa8886727d7897e8912b4aa5c1bc2912e75c97eb92ec7ece848de74bb59.jpg) 

![Auto-logistic analysis of Plantago lanceolata data: observed and expected frequencies forfirstanalysis inTable4 ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/318a0af930ed7df20baf14fa266ffb4c72130cc6b74f613b5d4e69f8b7dfaaa2.jpg) 
![TABLE7 Auto-logistic analysis of Plantago lanceolata data: goodness-of-fit tests for scheme (b) ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/bd4194aadebaa590ca22749511d40d5624b4eb202b9accb14ce332d9a36f16f0.jpg) 

## 7.2. Auto-normal Analysis of Mercer and Hall Wheat Plots 

Mercer and Hall (1911) present the results of a uniformity trial concerning 500 wheat plots, each $\mathbf{11\, f t\times10{\cdot}82\, i}$ ft, arranged as a $20\times25$ rectangular array. Two measurements, grain yield and straw yield, were made on each plot. Whittle (1954) analysed the grain yields, fitting various stationary normal autoregressions, as briefly described in Section 6.3 of the present paper. We shall analyse the same set of data but on the basis of the homogeneous first- and second-order schemes, (5.5) and

 (5.6). 

### 7.2.1. Coding methods 

In Tables 8 and 9 we record the parameter estimates for the schemes (5.5) and

 (5.6), respectively, using the coding techniques. The various analyses, within each table, refer to shifts in coding pattern, as previously described. Scheme (5.5) is also fitted under the Fig. 2 codings (Table 10) in order to test for the significance of the parameters $\mathbf{\gamma_{1}}$ and $\gamma_{\tt z}$ ：A typical analysis of variance is given in Table 11. Over the four coding shifts, the respective $\pmb{F}$ ratios for the combined effect of $\gamma_{1}$ and $\gamma_{\tt z}$ are 0·9 

![TABLE 8 Auto-normalanalysisofwheatplotsdata: parameter estimates for scheme (5.5) under Fig.1 codings ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/40b8edfaf8c727a25d761d1855fd0e74970ef737323a2c23fa3817f038d3145d.jpg) 

(2 and 103 d.fr.), 0:06 (2 and 94 d.fr.), 1.1 (2 and 103 d.fr.) and 1·2 (2 and 94 d.fr.). Each of these statistics suggests that the first-order scheme provides an adequate description $F_{\frac{2}{2},\frac{}{94}}$ distribution). However, for further comments concerning the model and the para- meterestimates, seeSection7.2.4. 
![Auto-normal analysis of wheat plots data: parameter estimates for scheme (5.6) under Fig.2codings ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/16f7132034208df2f1fd47efdaf479152b464a1f2f221d0864bf04efed98d31c.jpg) 

![Auto-normalanalysisofwheatplotsdata: parameter estimates for scheme (5.5)under Fig.2codings ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/1661858bd444f1088ae48e2d6292c389e2fc83bd2acd05255aafbd3c7554f865.jpg) 

![Auto-normal analysis ofwheatplots data:first analysis of variance underFig.2 codings ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/41d2b63cbec9a988c4151d5293df3637b859d1bbb830c4f94ca26d0be1085673.jpg) 

### 7.2.2. Unilateral approximations to the first-order scheme 

If we treat the data as a partial realization from the stationary infinite lattice version of (5.5), we have an a.c.g.f. proportional to 

$$
\{1\!-\!\beta_{1}(z_{1}\!+\!z_{1}^{-1})\!-\!\beta_{2}(z_{2}\!+\!z_{2}^{-1})\}^{-1}.
$$ 

As a first unilateral approximation, we may use the stationary auto regression 

$$
X_{i,j}=b_{\mathbf{1}}\,X_{i-1,j}\!+b_{\mathbf{2}}\,X_{i,j-1}\!+\!Z_{i,j},
$$ 
where $\{Z_{i, j}\colon i, j=0,\,\pm\, 1,\ldots\}$ is a doubly infinite set of independent Gaussian variates, each with zero mean and equal variance. The scheme (7.1) has a.c.g.f. proportional to 

$$
\{1+b_{1}^{2}+b_{2}^{2}-b_{1}(z_{1}+z_{1}^{-1})-b_{2}(z_{2}+z_{2}^{-1})+b_{1}b_{2}(z_{1}\,z_{2}^{-1}+z_{1}^{-1}\,z_{2})\}^{-1},
$$ 

which is clearly a first approximation to (7.1). Fitting the scheme (7.2) results in parameter estimates $\pmb{\hat{b}_{1}=0\cdot488}$ and $\hat{b}_{\hat{\mathbf{z}}}=0\!\cdot\!\dot{2}\! 02$ and an estimated a.c.g.f. proportional to 

$$
\{1-0\cdot382(z_{1}+z_{1}^{-1})-0\cdot158(z_{2}+z_{2}^{-1})+0\cdot077(z_{1}\,z_{2}^{-1}+z_{1}^{-1}\,z_{2})\}^{-1}.
$$ 

The values 0-382 and 0.158 may therefore be interpreted as crude estimates of $\beta_{1}$ and $\beta_{\pmb{2}}$ in (7.1). However, these estimates are somewhat arbitrarily formed and little confidence should be placed in them. For example, there is no real reason why $\hat{b}_{1}$ and $\pmb{\hat{b}_{2}}$ themselves should not be used to estimate $\beta_{1}$ and $\beta_{\pmb{\mathscr{s}}}$ ; the fact is that $\beta_{1}$ and $\beta_{\pmb{\mathscr{z}}}$ are really too large for the first approximation to be of much use. Thus, we consider the second unilateral approximation. 

The distribution of $X_{i, j},$ conditional upon all other values, in the scheme (7.2) $\pmb{x_{i+1, j-1}}$ $\pmb{x_{i-1, j+1}}$ by modifying the scheme to 

$$
X_{i,j}=b_{1}\,X_{i-1,j}+b_{2}\,X_{i,j-1}+b_{1}\,b_{2}\,X_{i+1,j-1}+Z_{i,j},
$$ 

with a.c.g.f. proportional to 

$$
\{1+b_{1}^{2}+b_{2}^{2}+b_{1}^{2}b_{2}^{2}-b_{1}(1-b_{2}^{2})\,(z_{1}+z_{1}^{-1})-b_{2}(z_{2}+z_{2}^{-1})+b_{1}^{2}b_{2}(z_{1}^{2}\,z_{2}^{-1}+z_{1}^{-2}\,z_{2})\}^{-1}.
$$ 

Fitting this scheme gives parameter estimates $\hat{b}_{1}=0\cdot483$ and $\hat{b}_{\mathbf{\hat{z}}}=\mathbf{0}\mathbf{\cdot}\mathbf{150}$ andan estimated a.c.g.f. proportional to 

$$
\{1-0\cdot374(z_{1}+z_{2}^{-1})-0\cdot119(z_{2}+z_{2}^{-1})+0\cdot028(z_{1}^{\tt a}\,z_{2}^{-1}+z_{1}^{-\tt a}\,z_{2})\}^{-1}.
$$ 

The values 0·374 and 0·119 may therefore be interpreted as better estimates of $\beta_{1}$ and $\beta_{\pmb{\mathscr{s}}}$ $X_{i+\pmb{2}, j-1}$ sequence of unilateral approximations to the scheme (5.5) may be generated (although there is probably little point in going further than the third one). These unilateral schemes also have the advantage that their analytical correlation structure is available in a region of the plane (Besag, 1972b). However, in more general situations, the approximation technique is likely to be rather cumbersome and is not particularly recommended. 

Whittle (1954) also fits the unilateral scheme (7.2) to the Mercer and Hall data, but in its own right rather than as an approximation. Whittle notes that the fit is "surprisingly" good in comparison with some of his bilateral autoregressions. This might now be explained by interpreting the scheme as an approximation to an auto- normalscheme. 

### 7.2.3. Adaptation of Whittle's method 

Under the assumption of stationarity and neglecting edge effects, we may fit the schemes (5.5) and (5.6) as outlined in Section 6.3. Parameter estimation can be carried out iteratively using the Newton-Raphson technique, with the normalizing function. $\Lambda_{;}$ being evaluated by numerical integration at each stage. The results are given in Table 12. The likelihood-ratio test for the scheme (5.5) within the class (5.6) gives a chi-squared statistic 2.69 on 2 d.fr. and again (see Section 7.2.1) the first-order auto- normal scheme appears satisfactory. 
![Auto-normal analysis of wheat plots data: parameter estimates using Whittle's method for stationary schemes ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/9227f7f4865ad87da7c49e9f4cc0e979651a7592e512030813f1faa744bb095a.jpg) 

### 7.2.4. More realistic models 

Concerning the Mercer and Hall data, it was pointed out by Whittle (1954) that the simple simultaneous autoregression (5.16) does not really reflect the observed correlation structure. This is also true for the first- and second-order (stationary) auto-normal schemes. The disparity between the observed and fitted correlograms can easily be seen from Tables 13, 14 and 15. The entries in Table 13 have been 

![TABLE13 Observed auto correlations for the wheat plots data ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/53d94dcb9dc65af8f330cfe055255ab6ebb60036a58f53cd9f5e358e664fd285.jpg) 

![TABLE14 Fittedautocorrelationsforthefirst-orderschemeinTable12 ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/5227fba9571503ce52130c071c51ddcdcde33dc8095e6a7573f33a79a679d297.jpg) 
![TABLE15 Fitted auto correlations for the second-order scheme in Table12 ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/0958011a70e35d41462981174fe95f8b9c527636b66e5363bb158b1ed5b8cfd8.jpg) 

copied from Whittle's paper. Whittle gives a number of possible explanations of the observed correlogram behaviour. Amongst these is the fact that the data are inte- grated observations of growth over plots rather than point observations. As with quadrat counts, this renders a simple Markov assumption somewhat dubious. A further suggestion (Patankar, 1954) is that the process is non-stationary, but we leave this for the moment and treat the observed correlogram at face value. The question then is: how can we reproduce a fitted correlation structure which tallies with Table 13, without'the associated scheme becoming too artificial? The answer may well lie in the use of a third-order auto-normal scheme, for then $\pmb{r_{2,0}}$ and $\pmb{r_{0,2}}$ canbefitted exactly and it is here that the trouble really seems to be. Note that, with the inclusion of third-order terms, the second-order terms may now have a significant role to play. The results of fitting the third-order scheme will be reported in due course. 

In the Biometrika paper immediately following Whittle's, Patankar (1954) also examined some spatial aspects of the Mercer and Hall data but, notably, only after removing a significant linear trend running from West to East. Thus, in Table 16 we give a typical auto-normal analysis of variance, constructed by the coding method 

![IABLEI6 Auto-normal analysis of wheat plots with the inclusion of a linear trend term:first analysisofvarianceunderFig.2codings ](https://cdn-xlab-data.openxlab.org.cn/pdf/0d128e46-f6b2-4b48-ad43-116803537f77.pdf/251365e1f41c732b4ec8542f3a405c7a1e0b0d3fd827879783e8e156a8e9fafe.jpg) 

but including trend removal. The estimates of the parameters are only slightly changed, as might be expected in such a situation, and the overall conclusions con- cerning influence of diagonally nearest plots remains unchanged 

Thirdly, we note a disconcerting property of some of the coding fits. That is, on a number of occasions, they individually give parameter estimates which are in- consistent with a stationary auto-normalscheme. Forexample, withthefirst-order scheme (5.5), thesecondanalysisinTable8producesestimatesof $\beta_{1}$ and $\beta_{\pmb{2}}$ whosesum exceeds 0·5. Similar inconsistencies occur even after the removal of trend. Whilst it is sohappensthat, for each model fitted, the meanestimates are feasible, the in- dividual values again suggest that the models are not entirely appropriate. 
Summarizing, it cannot be claimed that the present auto-normal schemes have been successful in reflecting the overall probabilistic structure of the wheat plots process. Further analysis is required, although it is felt to be perhaps more important to examine a range of examples rather than to concentrate in too much detail upon asinglesetof classicaldata. 

# 8 Concluding Remarks
In the preceding sections, an attempt has been made to establish that a con ditional probability approach to spatial processes is not only feasible but is also desirable. It has been suggested, firstly, that the conditional probability approach has greater intuitive appeal to the practising statistician than the alternative joint probability approach; secondly, that the existence of the Hammersley-Clifford theorem has almost entirely removed any consistency problems and, further, can easilybe used as a tool for the construction of conditional probability models in many situations; thirdly, that the basic lattice models under the conditional prob- ability approach yield naturally to a very simple parameter estimation procedure (the coding technique) and, atleast for binary and Gaussian variate s, tostraight- forward goodness-of-ft tests. For Gaussian variates, maximum likelihood appears equally available for both simultaneous and conditional probability models of similar complexity. As regards the joint probability approach, it is not clear to the present author how, outside the Gaussian situation, the models are to be used in practice. How, for example, would Gleaves's binary data be analysed? 

On the other hand, the two examples discussed in Section 7 of the paper are far from convincing in demonstrating that simple conditional probability schemes provide satisfactory models for spatial processes. It is felt to be pertinent that, in each case, the data were derived from regions of the plane rather than point sites. There is clearly a need for more practical analyses to be undertaken. Some alternative suggestions on the specification of lattice models for aggregated data would alsobe of greatinterest. 
# Discussion of Mr. Besag's Paper
Professor D. R. Cox (Imperial College) : The paper is original, lucid and comprehensive. The topic is important and notoriously difficult. It is a pleasure to congratulate Mr Besag. 

Statistical subjects canbe characterized qualitatively by their statistical analysis to stochastic model ratio. This is rather low in the present subject and therefore the emphasis in the present paper on models for the analysis of data is very welcome. Never- theless understanding of the conditional models may be helped by relating them to temporal-spatial models, and in particular to their stationary distributions. It would be interesting to know what general connections can be established between Mr Be sag's auto-models and stationary distributions ofsimple temporal-spatial processes. 

Mr Be sag remarks on the possible advantages of a triangular rather than a rectangular lattice. Has this been tried numerically on examples of aggregated responses, where a comparison with a square lattice couldbemade? The physical meaning of a first-order scheme is more appealing for a triangular lattice than for a square one. There is possibly a qualitative connection with results on the location of points in sampling for a mean (Daleniusetal., 1961). 

Experimental design aspects are mentioned briefly in the paper. The link here is, I think, with the method of Papadakis (Bartlett, 1938). In this the treatment effects are estimated after adjustment by analysis of covariance on the residuals on neighbouring plots. The one-dimensional version of this has been related by Atkinson (1969) to an autoregressive process and Mr Besag's discussion probably provides a framework for the two-dimensional theory. 

The sections of the paper on the coding method are of particularly general interest as illustrating one more techniquefor simplifying complicated likelihoods. There aremany outstanding questions; aqualitative explanation of thehigh efficiency in the one- dimensional case (Hannan, 1958) might throw light on the two-dimensional behaviour. 

I propose a cordial vote of thanks to Mr Besag for his excellent paper. 

Dr A. G. HAwKEs (University of Durham): Like many people who have proposed or seconded votes of thanks at meetings of this Society Iam distinguished by the fact that I know little about the subject of the paper. Of course I have seen papers previously on the analysis of distributions on lattices. Having received the distinct impression that they contained rather nasty, messy mathematics and unpleasant·computation, since I had no desperate need to understand them, I put them on one side and, apart from a brief look, tended not towork through them carefully. Therefore I am extremely grateful toMrBesag forpresenting thispaperwith his elegantgeneral treatmentof distributions on lattices or, indeed, for any multivariate distribution at alland his interesting general results. 

In addition, he gives a simple and flexible class of automodels, a simple method of analysisandsomenicepracticalexamples. 

# Appendix
## Deduction for $\mathcal Q (\mathbf x)$ 's expansion
TODO:
推导：

$$
\begin{align}
\mathcal Q(\mathbf x) &=\sum_{1\le i \le n}x_i G_i(x_i) + \sum_{1\le i<j\le n}x_ix_j G_{ij}(x_i, x_j)\\
&+\sum_{1\le i< j < k \le n}x_ix_jx_k G_{ijk}(x_i, x_j, x_k) + \dots\\
&+x_1x_2\dots x_n G_{1,2,\dots, n}(x_1, x_2, \dots, x_n).
\end{align}
$$

其中：

$$
x_{i}\,G_{i}(x_{i})\!\equiv\!\mathcal Q(0,...,0,x_{i},0,...,0)\!-\!\mathcal Q(\mathbf 0),
$$ 
高阶的 $G$ 函数的定义类似，
例如将 $x_ix_j G_{ij}(x_i, x_j)$ 定义为 $\mathcal Q (0,\dots,  x_i, \dots, x_j, \dots, 0)- \mathcal Q (\mathbf 0)$
也就是左边是仅 $x_i, x_j$ 取值来源于 $\mathbf x$，其余都固定取为 $0$ 的 $\mathcal Q$ 函数，右边是 $\mathcal Q (\mathbf 0)$

两边取指数：

$$
\begin{align}
\exp\left\{(\mathcal Q(\mathbf x)\right\} &=\exp\left\{\sum_{1\le i \le n}x_i G_i(x_i)\right\} \cdot 
\exp\left\{\sum_{1\le i<j\le n}x_ix_j G_{ij}(x_i, x_j)\right\}\\
&\cdot\exp\left\{\sum_{1\le i< j < k \le n}x_ix_jx_k G_{ijk}(x_i, x_j, x_k)\right\} + \dots\\
&\cdot\exp\left\{x_1x_2\dots x_n G_{1,2,\dots, n}(x_1, x_2, \dots, x_n)\right\}\\
\end{align}
$$

考虑等式右边第一项：

$$
\begin{align}
\exp\left\{\sum_{1\le i\le n}x_i G_i(x_i)\right\}
&=\prod_{1\le i\le n}\exp\left\{x_i G_i(x_i)\right\}\\
&=\prod_{1\le i\le n}\exp\left\{\mathcal Q(0, \dots, 0,x_i,0,\dots, 0)-\mathcal Q(\mathbf 0)\right\}\\
&=\prod_{1\le i \le n}\frac {P(x_i\mid 0, \dots,0,0,\dots 0)}{P(0\mid 0,\dots, 0,0,\dots,0)}
\end{align}
$$


考虑等式右边第二项：

$$
\begin{align}
\exp\left\{\sum_{1\le i<j\le n}x_ix_j G_{ij}(x_i,x_j)\right\}
&=\prod_{1\le i<j\le n}\exp\left\{x_ix_j G_{ij}(x_i,x_j)\right\}\\
&=\prod_{1\le i<j\le n}\exp\left\{\mathcal Q(0, \dots, x_i,\dots,x_j,\dots, 0)-\mathcal Q(\mathbf 0)\right\}\\
&=\prod_{1\le i<j \le n}\frac {P(x_i,x_j\mid 0, \dots,0,\dots,0,\dots 0)}{P(0,0\mid 0,\dots, 0,\dots,0,\dots,0)}
\end{align}
$$

