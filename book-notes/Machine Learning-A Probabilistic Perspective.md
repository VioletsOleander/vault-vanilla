# 11 Mixture models and the EM algorithm 
## 11.1 Latent variable models 
In Chapter 10 we showed how graphical models can be used to deﬁne high-dimensional joint probability distributions. The basic idea is to model dependence between two variables by adding an edge between them in the graph. (Technically the graph represents conditional independence, but you get the point.) 

An alternative approach is to assume that the observed variables are correlated because they arise from a hidden common “cause”. Model with hidden variables are also known as latent variable models or LVM s. As we will see in this chapter, such models are harder to ﬁt than models with no latent variables. However, they can have signiﬁcant advantages, for two main reasons. First, LVMs often have fewer parameters than models that directly represent correlation in the visible space. This is illustrated in Figure 11.1. If all nodes (including H) are binary and all CPDs are tabular, the model on the left has 17 free parameters, whereas the model on the right has 59 free parameters. 

Second, the hidden variables in an LVM can serve as a bottleneck , which computes a compressed representation of the data. This forms the basis of unsupervised learning, as we will see. Figure 11.2 illustrates some generic LVM structures that can be used for this purpose. In general there are  $L$ latent variables,  $z_{i1},.\,.\,.\,,z_{I L}$ , and  $D$ visible variables,  $x_{i1},\cdot\cdot\cdot,x_{i D}$ , where usually  $D\gg L$ . If we have  $L>1$ , there are atent factors contributing to each observation, so we have a many-to-many mapping. If $L=1$ , we we only have a single latent variable; in this case,  $z_{i}$ is usually discrete, and we have a one-to-many mapping. We can also have a many-to-one mapping, representing different competing factors or causes for each observed variable; such models form the basis of probabilistic matrix factorization, discussed in Section 27.6.2. Finally, we can have a one-to-one mapping, which can be represented as $\mathbf{z}_{i}\rightarrow\mathbf{x}_{i}$ . By allowing $\mathbf{z}_{i}$ and/or $\mathbf{x}_{i}$ to be vector-v is representati n subsume all the others. Depending on the form of the likelihood $p(\mathbf{x}_{i}|\mathbf{z}_{i})$ | and the prior $p\big(\mathbf{z}_{i}\big)$ , we can generate a variety of different models, as summarized in Table 11.1. 

## 11.2 Mixture models 
The simplest form of LVM is $z_{i}\in\{1,.\,.\,.\,,K\}$ , representing a disc l use a discrete prior for this, $p(z_{i})=\mathrm{Cat}(\pi)$ . For the likelihood, we use $p(\mathbf{x}_{i}|z_{i}=k)=p_{k}(\mathbf{x}_{i})$ | , 

![](images/0d0a19065964329cdbcbd7f6e74fcdf7a65f2b08f0c4db499f4cd63c5d166c61.jpg) 
Figure 11.1 A DGM with and without hidden variables. The leaves represent medical symptoms. The roots represent primary causes, such as smoking, diet and exercise. The hidden variable can represent mediating factors, such as heart disease, which might not be directly visible. 

![](images/3cc12b17008724ea8a564ea7b22c49275e450c79ad4deb4aa9833581cc035e06.jpg) 
Figure 11.2 A latent variable model represented as a DGM. (a) Many-to-many. (b) One-to-many. (c) Many-to-one. (d) One-to-one. 

where  $p_{k}$ is the  $k$ ’th base distribution for the observations; this can be of any type. The overall model is known as a mixture model , since we are mixing together the  $K$ base distributions as follows: 

$$
p(\mathbf{x}_{i}|\pmb{\theta})=\sum_{k=1}^{K}\pi_{k}p_{k}(\mathbf{x}_{i}|\pmb{\theta})
$$ 

This is a convex combination of the  $p_{k}\,{\bf\dot{s}}$ , since we are taking a weighted sum, where the mixing weights  $\pi_{k}$ satisfy  $0\leq\pi_{k}\leq1$ and $\textstyle\sum_{k=1}^{K}\pi_{k}=1$ . We give some examples below. 

![](images/85cfcd6ce906fb8f1dab6dbe7cda53db4293840cd6c62fe525a40e9d71b91b0d.jpg) 

Table 11.1 Summary of some popular directed latent variable models. Here “Prod” means product, so “Prod. Discrete” in the likelihood means a factore of the form  $\begin{array}{r}{\prod_{j}\mathrm{Cat}(x_{i j}|\mathbf{z}_{i})}\end{array}$  , and “Prod. analysis”. “ICA” stands for “independent components analysis”. Gaussian” means a factored distribution of the form  $\begin{array}{r}{\prod_{j}\mathcal{N}(x_{i j}|\mathbf{z}_{i})}\end{array}$   . “PCA” stands for “principal components 

### 11.2.1 Mixtures of Gaussians 
The most widely used mixture model is the mixture of Gaussians (MOG), also called a Gaussian mixture model or GMM . In this model, each base distribution in the mixture is a multivariate Gaussian with mean $\pmb{\mu}_{k}$ and covariance matrix $\pmb \Sigma_{k}$ . Thus the model has the form 

$$
p(\mathbf{x}_{i}|\pmb{\theta})=\sum_{k=1}^{K}\pi_{k}\mathcal{N}(\mathbf{x}_{i}|\pmb{\mu}_{k},\pmb{\Sigma}_{k})\tag{11.2}
$$

>  高斯混合模型中，每个基本分布是多个多元高斯分布的混合，形式如上
>  其中 $\pi_k$ 表示 $p(z=k)$，满足 $\pi_k \ge 0,\sum_{k}\pi_k = 1$

>  推导

$$
\begin{align}
p(\mathbf x_i\mid \pmb \theta) &= \sum_{k=1}^K p(\mathbf x_i,z=k\mid \pmb\theta)\\
&=\sum_{k=1}^Kp(z=k\mid \pmb \theta)p(\mathbf x_i\mid z=k,\pmb \theta)\\
&=\sum_{k=1}^Kp(z=k)p(\mathbf x_i\mid z=k,\pmb \theta)\\
&=\sum_{k=1}^K\pi_k\mathcal N(\mathbf x_i\mid \pmb \mu_k,\pmb \Sigma_k)
\end{align}
$$

>  推导完毕

Figure 11.3 shows a mixture of 3 Gaussians in 2D. Each mixture component is represented by a different set of elliptical contours. Given a sufficiently large number of mixture components, a GMM can be used to approximate any density deﬁned on $\mathbb{R}^{D}$ . 
>  给定充分大的混合成分，高斯混合模型可以用于近似 $\mathbb R^D$ 中的任意概率密度函数

### 11.2.2 Mixture of multinoullis 
We can use mixture models to deﬁne density models on many kinds of data. For example, suppose our data consist of  $D$ -dimensional bit vectors. In this case, an appropriate class- conditional density is a product of Bernoullis: 

$$
p(\mathbf{x}_{i}|z_{i}=k,\pmb\theta)=\prod_{j=1}^{D}\mathrm{Ber}(x_{i j}|\mu_{j k})=\prod_{j=1}^{D}\mu_{j k}^{x_{i j}}(1-\mu_{j k})^{1-x_{i j}}
$$ 

where  $\mu_{j k}$ is the probability that bit  $j$ turns on in cluster  $k$ . 

The latent variables do not have to any meaning, we might simply introduce latent variables in order to make the model more powerful. For example, one can show (Exercise 11.8) that the mean and covariance of the mixture distribution are given by 

$$
\begin{array}{r c l}{\mathbb{E}\left[{\bf x}\right]}&{=}&{\displaystyle\sum_{k}\pi_{k}{\pmb\mu}_{k}}\\ {\mathrm{cov}\left[{\bf x}\right]}&{=}&{\displaystyle\sum_{k}\pi_{k}[{\pmb\Sigma}_{k}+{\pmb\mu}_{k}{\pmb\mu}_{k}^{T}]-\mathbb{E}\left[{\bf x}\right]\mathbb{E}\left[{\bf x}\right]^{T}}\end{array}
$$ 

where  $\Sigma_{k}\,=\,\mathrm{diag}(\mu_{j k}(1\,-\,\mu_{j k}))$ . So although the component distributions are factorized, the joint distribution is not. Thus the mixture distribution can capture correlations between variables, unlike a single product-of-Bernoullis model. 

### 11.2.3 Using mixture models for clustering 

There are two main applications of mixture models. The ﬁrst is to use them as a black-box density model, $p(\mathbf{x}_{i})$ . This can be useful for a variety of tasks, such as data compression, outlier detection, and creating generative classiﬁers, where we model each class-conditional density $p(\mathbf{x}|y=c)$ by a mixture distribution (see Section 14.7.3). 

The second, and more common, application of mixture models is to use them for clustering. We discuss this topic in detail in Chapter 25, but the basic idea is simple. We ﬁrst ﬁt the mixture model, and then co pute  $p(z_{i}=k|\mathbf{x}_{i},\pmb\theta)$ , which represents the poste or probabi ty that point $i$ belongs to cluster k . This is known as the responsibility of cluster k for point i , and can be computed using Bayes rule as follows: 

$$
\begin{array}{r l r}{r_{i k}\triangleq p(z_{i}=k|\mathbf{x}_{i},\pmb{\theta})}&{=}&{\frac{p(z_{i}=k|\pmb{\theta})p(\mathbf{x}_{i}|z_{i}=k,\pmb{\theta})}{\sum_{k^{\prime}=1}^{K}p(z_{i}=k^{\prime}|\pmb{\theta})p(\mathbf{x}_{i}|z_{i}=k^{\prime},\pmb{\theta})}}\end{array}
$$ 

This procedure is called soft clustering , and is identical to the computations performed when using a generative classiﬁer. The difference between the two models only arises at training time: in the mixture case, we never observe $z_{i}$ , whereas with a generative classiﬁer, we do observe  $y_{i}$ (which plays the role of  $z_{i}$ ). 

We can represent the amount of uncertainty in the cluster assignment by using $1-\operatorname*{max}_{k}r_{i k}$ . Assuming this is small, it may be reasonable to compute a hard clustering using the MAP estimate, given by 

$$
z_{i}^{*}=\arg\operatorname*{max}_{k}r_{i k}=\arg\operatorname*{max}_{k}\log p(\mathbf{x}_{i}|z_{i}=k,\pmb{\theta})+\log p(\mathbf{z}_{i}=k|\pmb{\theta})
$$ 

![](images/e3a75af5056f1ebf5b460a8c17f69b5d66d8afb1059274e6b0effa17e83dad2a.jpg) 
Figure 11.4 (a) Some yeast gene expression data plotted as a time series. (c) Visualizing the 16 cluster centers produced by K-means. Figure generated by kmeansYeastDemo . 

![](images/df9661f06494f533c5d483eec040d70b478d8259fcca34b3d9e9223ac0010bdc.jpg) 
Figure 11.5 We ﬁt a mixture of 10 Bernoullis to the binarized MNIST digit data. We show the MLE for the corresponding cluster means,  $\mu_{k}$ . The numbers on top of each image represent the mixing weights $\hat{\pi}_{k}$ . No labels were used when training the model. Figure generated by mixBerMnistEM . 

Hard clustering using a GMM is illustrated in Figure 1.8, where we cluster some data rep- resenting the height and weight of people. The colors represent the hard assignments. Note that the identity of the labels (colors) used is immaterial; we are free to rename all the clusters, without affecting the partitioning of the data; this is called label switching . 

Another example is shown in Figure 11.4. Here the data vectors  $\mathbf{x}_{i}~\in~\mathbb{R}^{7}$  represent the expression levels of different genes at 7 different time points. We clustered them using a GMM. We see that there are several kinds of genes, such as those whose expression level goes up monotonically over time (in response to a given stimulus), those whose expression level goes down monotonically, and those with more complex response patterns. We have clustered the series into  $K=16$ groups. (See Section 11.5 for details on how to choose  $K$ .) For example, we can represent each cluster by a prototype or centroid . This is shown in Figure 11.4 (b). 

As an example of clustering binary data, consider a binarized version of the MNIST handwrit- ten digit dataset (see Figure 1.5 (a)), where we ignore the class labels. We can ﬁt a mixture of Bernoullis to this, using  $K=10$ , and then visualize the resulting centroids, $\hat{\pmb{\mu}}_{k}$ , as shown in Figure 11.5. We see that the method correctly discovered some of the digit classes, but overall the results aren’t great: it has created multiple clusters for some digits, and no clusters for others. There are several possible reasons for these “errors”: 

• The model is very simple and does not capture the relevant visual characteristics of a digit. For example, each pixel is treated independently, and there is no notion of shape or a stroke. 

• Although we think there should be 10 clusters, some of the digits actually exhibit a fair degree of visual variety. For example, there are two ways of writing 7’s (with and without the cross bar). Figure 1.5 (a) illustrates some of the range in writing les. Thus we need  $K\gg10$ clusters to adequately model this data. However, if we set K to be large, there is nothing in the model or algorithm preventing the extra clusters from being used to create multiple versions of the same digit, and indeed this is what happens. We can use model selection to prevent too many clusters from being chosen but what looks visually appealing and what makes a good density estimator may be quite different.

 • The likelihood function is not convex, so we may be stuck in a local optimum, as we explain 

This example is typical of mixture modeling, and goes to show one must be very cautious trying to “interpret” any clusters that are discovered by the method. (Adding a little bit of supervision, or using informative priors, can help a lot.) 

### 11.2.4 Mixtures of experts 

Section 14.7.3 described how to use mixture models in the context of generative classiﬁers. We can also use them to create discriminative models for classiﬁcation and regression. For example, consider the data in Figure 11.6 (a). It seems like a good model would be three different linear regression functions, each applying to a different part of the input space. We can model this by allowing the mixing weights and the mixture densities to be input-dependent: 

$$
\begin{array}{r c l}{p(y_{i}|\mathbf{x}_{i},z_{i}=k,\pmb{\theta})}&{=}&{\mathcal{N}(y_{i}|\mathbf{w}_{k}^{T}\mathbf{x}_{i},\sigma_{k}^{2})}\\ {p(z_{i}|\mathbf{x}_{i},\pmb{\theta})}&{=}&{\mathrm{Cat}(z_{i}|\mathcal{S}(\mathbf{V}^{T}\mathbf{x}_{i}))}\end{array}
$$ 

See Figure 11.7 (a) for the DGM. 

This model is called a mixture of experts or MoE (Jordan and Jacobs 1994). The idea is that each submodel is considered to be an “expert” in a certain region of input space. The function $p(z_{i}\,=\,k|\mathbf{x}_{i},\theta)$ is called a gating function , and decides which expert to use, depending on the input values. For example, Figure 11.6 (b) shows how the three experts have “carved up” the 1d input space, Figure 11.6 (a) shows the predictions of each expert individually (in this case, the experts are just linear regression models), and Figure 11.6 (c) shows the overall prediction of the model, obtained using 

$$
p(y_{i}|\mathbf{x}_{i},\pmb\theta)=\sum_{k}p(z_{i}=k|\mathbf{x}_{i},\pmb\theta)p(y_{i}|\mathbf{x}_{i},z_{i}=k,\pmb\theta)
$$ 

We discuss how to ﬁt this model in Section 11.4.3. 

![](images/a2db863404fe6287d54cfeb1c1d196574cfd53dd81f556bbabe3b83cb0e2345f.jpg) 
Figure 11.6 (a) Some data ﬁt with three separate regression lines. (b) Gating functions for three different “experts”. (c) The conditionally weighted average of the three expert predictions. Figure generated by mixexpDemo . 

![](images/458dcb8452e4887fc9b97970225e5fee7343a4ef6261827f2158c19e2ddf233b.jpg) 
Figure 11.7 (a) A mixture of experts. (b) A hierarchical mixture of experts. 

![](images/51ca109d0a3c8109558d734479065c9c58c3bf0dce86d7a63f14cd6cae3c5f3a.jpg) 
Figure 11.8 (a) Some data from a simple forwards model. (b) Some data from the inverse model, ﬁt with a mixture of 3 linear regressions. Training points are color coded by their responsibilities. (c) The predictive mean (red cross) and mode (black square). Based on Figures 5.20 and 5.21 of (Bishop 2006b). Figure generated by mix exp Demo One To Many . 

It should be clear that we can “plug in” any model for the expert. For example, we can use neural networks (Chapter 16) to represent both the gating functions and the experts. The result is known as a mixture density network . Such models are slower to train, but can be more ﬂexible than mixtures of experts. See (Bishop 1994) for details. 

It is also possible to make each expert be itself a mixture of experts. This gives rise to a model known as the hierarchical mixture of experts . See Figure 11.7 (b) for the DGM, and Section 16.2.6 for further details. 

#### 11.2.4.1 Application to inverse problems 

Mixtures of experts are useful in solving inverse problems . These are problems where we have to invert a many-to-one mapping. A typical example is in robotics, where the location of the end effector (hand) $\mathbf{y}$ is uniquely determined by the joint angles of the motors, x . However, for any given location y , there are many settings of the joints $\mathbf{x}$ that can produce it. Thus the inverse mapping  $\mathbf{x}=f^{-1}(\mathbf{y})$ is not unique. Another example is kinematic tracking of people from video (Bo et al. 2008), where the mapping from image appearance to pose is not unique, due to self occlusion, etc. 

![](images/fd5adbf79e7d4273a1a6836e17e0e2f2fa2c8de0fd0e4964d943aef4efdaa1e2.jpg) 
Figure 11.9 A LVM represented as a DGM. Left: Model is unrolled for  $N$ examples. Right: same model using plate notation. 

A simpler example, for illustration purposes, is shown in Figure 11.8 (a). We see that this deﬁnes a function,  $y\ =\ f(x)$ , since for every value  $x$ along the horizontal axis, there is a unique response  $y$ . This is sometimes called the forwards model . Now consider the problem of computing  $x=f^{-1}(y)$ . The corresponding inverse model is shown in Figure 11.8 (b); this is obtained by simply interchanging the  $x$ and  $y$ axes. Now we see that for some values along the horizontal axis, there are multiple possible outputs, so the inverse is not uniquely deﬁned. For example, if  $y=0.8$ , then  $x$ could be 0.2 or 0.8. Consequently, the predictive distribution, $p(x|y,\theta)$ is multimodal. 

We can ﬁt a mixture of linear experts to this data. Figure 11.8 (b) shows the prediction of each expert, and Figure 11.8 (c) shows (a plugin approximation to) the posterior predictive mode and mean. Note that the posterior mean does not yield good predictions. In fact, any model which is trained to minimize mean squared error — even if the model is a ﬂexible nonlinear model, such as neural network — will work poorly on inverse problems such as this. However, the posterior mode, where the mode is input dependent, provides a reasonable approximation. 

## 11.3 Parameter estimation for mixture models 

We have seen how to compute the posterior over the hidden variables given the observed variables, assuming the parameters are known. In this section, we discuss how to learn the parameters. 

In Section 10.4.2, we showed that when we have complete data and a factored prior, the posterior over the parameters also factorizes, making computation very simple. Unfortunately this is no longer true if we have hidden variables and/or missing data. The reason is apparent from looking at Figure 11.9. If the  $z_{i}$ were observed, then by d-sepa ion, we see that  $\begin{array}{r}{\mathbf{\nabla}\theta_{z}\perp\mathbf{\nabla}\theta_{x}|\mathcal{D},}\end{array}$ , and hence the posterior will factorize. But since, in an LVM, the $z_{i}$ are hidden, the parameters are no longer independent, and the posterior does not factorize, making it much harder to 

![](images/035f0082dfb9e64957a5682200903600cca9dccd8d76d175826e35f39ce16d03.jpg) 
Figure 11.10 Left:  $N=200$ data points sampled from a mixture of 2 Gaussians in 1d, with  $\pi_{k}\,=\,0.5$ , $\sigma_{k}\,=\,5$ ,  $\mu_{1}=-10$ and  $\mu_{2}=10$ . Right: Likelihood surface  $p(\mathcal{D}|\mu_{1},\mu_{2})$ , with all other parameters set to their true values. We see the two symmetric modes, reﬂecting the un ident i ability of the parameters. Figure generated by mix Gauss Li k Surface Demo . 

compute. This also complicates the computation of MAP and ML estimates, as we discus below. 

#### 11.3.1 Unidentiability 

The main problem with computing  $p(\pmb{\theta}|\mathcal{D})$ for an LVM is that the posterior may have multiple modes. To see why, consider a GMM. If the  $z_{i}$ were all observed, we would have a unimodal posterior for the parameters: 

$$
p(\pmb\theta|\mathcal D)=\mathrm{Dir}(\pmb\pi|\mathcal D)\prod_{k=1}^{K}\mathrm{NIMW}(\pmb\mu_{k},\pmb\Sigma_{k}|\mathcal D)
$$ 

Consequently we can easily ﬁnd the globally optimal MAP estimate (and hence globally optimal MLE). 

But now suppose the  $z_{i}$ ’s are hidden. In this case, for each of the possible ways of “ﬁlling in” the  $z_{i}$ ’s, we get a different unimodal likelihood. Thus when we marginalize out over the  $z_{i}$ ’s, we get a multi-modal posterior for  $p(\pmb{\theta}|\mathcal{D})$ .  These modes correspond to different labelings of the clusters. This is illust Figure 11.10 (b), where we plot the likelihood function,  $p(\mathcal{D}|\mu_{1},\mu_{2})$ , for a 2D GMM with $K=2$ for the data is shown in Figure 11.10 (a). We see two peaks, one corresponding to the case where  $\mu_{1}=-10$ ,  $\mu_{2}=10$ , and the other to the case where  $\mu_{1}=10$ , $\mu_{2}\,=\,-10$ . We say the parameters are not identiﬁable , since there is not a unique MLE. Therefore there cannot be a unique MAP estimate (assuming the prior does not rule out certain labelings), and hence the posterior must be multimodal. The question of how many modes there are in the parameter posterior is hard to answer. There are  $K!$ possible labelings, but some of the peaks might get merged. Nevertheless, there can be an exponential number, since ﬁnding the optimal MLE for a GMM is NP-hard (Aloise et al. 2009; Drineas et al. 2004). 

Un ident i ability can cause a problem for Bayesian inference. For example, suppose we draw some samples from the posterior,  $\pmb\theta^{(s)}\,\sim\,\dot{p}(\pmb\theta|\mathcal{D})$ , and then average them, to try to approximate the posterior mean,  $\begin{array}{r}{\overline{{\pmb\theta}}\,=\,\frac{1}{S}\sum_{s=1}^{S}\pmb\theta^{(s)}}\end{array}$ 
 . (This kind of Monte Carlo approach is explained in more detail in Chapter 24.) If the samples come from different modes, the average will be meaningless. Note, however, that it is reasonable to average the posterior predictive distributions,  $\begin{array}{r}{\bar{p(\mathbf{x})}\approx\frac{1}{S}\sum_{s=1}^{S}p(\mathbf{x}|\pmb{\theta}^{(s)})}\end{array}$ 
 , since the likelihood function is invariant to which mode the parameters came from. 

A variety of solutions have been proposed to the un ident i ability problem. These solutions depend on the details of the model and the inference algorithm that is used. For example, see (Stephens 2000) for an approach to handling un ident i ability in mixture models using MCMC. 

The approach we will adopt in this chapter is much simpler: we just compute a single local mode, i.e., we perform approximate MAP estimation. (We say “approximate” since ﬁnding the globally optimal MLE, and hence MAP estimate, is NP-hard, at least for mixture models (Aloise et al. 2009).) This is by far the most common approach, because of its simplicity. It is also a reasonable approximation, at least if the sample size is sufficiently large. To see why, consider Figure 11.9 (a). We see that there are  $N$ latent variables, each of which gets to “see” one data point each. However, there are only two latent parameters, each of which gets to see  $N$ data points. So the posterior uncertainty about the parameters is typically much less than the posterior uncertainty about the latent variables. This justiﬁes the common strategy of computing  $p\big(z_{i}|\mathbf{x}_{i},\hat{\pmb\theta}\big)$ , but not bothering to compute  $p(\pmb{\theta}|\mathcal{D})$ . In Section 5.6, we will study hierarchical Bayesian models, which essentially put structure on top of the parameters. In such models, it is important to model $p(\pmb{\theta}|\mathcal{D})$ , so that the parameters can send information between themselves. If we used a point estimate, this would not be possible. 

### 11.3.2 Computing a MAP estimate is non-convex 

In the previous sections, we have argued, rather heuristically, that the likelihood function has multiple modes, and hence that ﬁnding an MAP or ML estimate will be hard. In this section, we show this result by more algebraic means, which sheds some additional insight into the problem. Our presentation is based in part on (Rennie 2004). 

Consider the log-likelihood for an LVM: 

$$
\log p(\mathcal{D}|\pmb{\theta})=\sum_{i}\log\left[\sum_{\mathbf{z}_{i}}p(\mathbf{x}_{i},\mathbf{z}_{i}|\pmb{\theta})\right]
$$ 

Unfortunately, this objective is hard to maximize. since we cannot push the log inside the sum. This precludes certain algebraic simplications, but does not prove the problem is hard. 

Now suppose the joint probability distribution  $p(\mathbf{z}_{i},\mathbf{x}_{i}|\pmb{\theta})$ is in the exponential family, which means it can be written as follows: 

$$
p(\mathbf{x},\mathbf{z}|\theta)=\frac{1}{Z(\theta)}\exp[\theta^{T}\phi(\mathbf{x},\mathbf{z})]
$$ 

where  $\phi(\mathbf{x},\mathbf{z})$ are the sufficient statistics, and  $Z(\theta)$ is the normalization constant (see Sec- tion 9.2 for more details). It can be shown (Exercise 9.2) that the MVN is in the exponential family, as are nearly all of the distributions we have encountered so far, including Dirichlet, multinomial, Gamma, Wishart, etc. (The Student distribution is a notable exception.) Further- more, mixtures of exponential families are also in the exponential family, providing the mixing indicator variables are observed (Exercise 11.11). 

With this assumption, the complete data log likelihood can be written as follows: 

$$
\ell_{c}(\pmb\theta)=\sum_{i}\log p(\mathbf x_{i},\mathbf z_{i}|\pmb\theta)=\pmb\theta^{T}(\sum_{i}\phi(\mathbf x_{i},\mathbf z_{i}))-N Z(\pmb\theta)
$$ 

The ﬁrst term is clearly linear in  $\theta$ . One can show that  $Z(\theta)$ is a convex function (Boyd and Vandenberghe 2004), so the overall objective is concave (due to the minus sign), and hence has a unique maximum. 

Now consider what happens when we have missing data. The observed data log likelihood is given by 

$$
\ell(\pmb\theta)=\sum_{i}\log\sum_{\mathbf z_{i}}p(\mathbf x_{i},\mathbf z_{i}|\pmb\theta)=\sum_{i}\log\left[\sum_{\mathbf z_{i}}e^{\pmb\theta^{T}\pmb\phi(\mathbf z_{i},\mathbf x_{i})}\right]-N\log Z(\pmb\theta)
$$ 

One can show that the log-sum-exp function is convex (Boyd and Vandenberghe 2004), and we know that  $Z(\theta)$ is convex. However, the difference of two convex functions is not, in general, convex. So the objective is neither convex nor concave, and has local optima. 

The disadvantage of non-convex functions is that it is usually hard to ﬁnd their global op- timum. Most optimization algorithms will only ﬁnd a local optimum; which one they ﬁnd depends on where they start. There are some algorithms, such as simulated annealing (Sec- tion 24.6.1) or genetic algorithms, that claim to always ﬁnd the global optimum, but this is only under unrealistic assumptions (e.g., if they are allowed to be cooled “inﬁnitely slowly”, or al- lowed to run “inﬁnitely long”). In practice, we will run a local optimizer, perhaps using multiple random restarts to increase out chance of ﬁnding a “good” local optimum. Of course, careful initialization can help a lot, too. We give examples of how to do this on a case-by-case basis. 

Note that a convex method for ﬁtting mixtures of Gaussians has been proposed. The idea is to assign one cluster per data point, and select from amongst them, using a convex  $\ell_{1}$ -type penalty, rather than trying to optimize the locations of the cluster centers. See (Lashkari and Golland 2007) for details. This is essentially an unsupervised version of the approach used in sparse kernel logistic regression, which we will discuss in Section 14.3.2. Note, however, that the $\ell_{1}$ penalty, although convex, is not necessarily a good way to promote sparsity, as discussed in Chapter 13. In fact, as we will see in that Chapter, some of the best sparsity-promoting methods use non-convex penalties, and use EM to optimie them! The moral of the story is: do not be afraid of non-convexity. 

## 11.4 The EM algorithm 
For many models in machine learning and statistics, computing the ML or MAP parameter estimate is easy provided we observe all the values of all the relevant random variables, i.e., if we have complete data. However, if we have missing data and/or latent variables, then computing the ML/MAP estimate becomes hard. 
> 在数据不完整的情况下，计算极大似然和极大后验估计是较难的

One approach is to use a generic gradient-based optimizer to ﬁnd a local minimum of the negative log likelihood or NLL , given by 

$$
\mathrm{NLL}\triangleq-\frac{1}{N}\log p(\mathcal D|\pmb\theta)\tag{11.16}
$$ 
However, we often have to enforce constraints, such as the fact that covariance matrices must be positive deﬁnite, mixing weights must sum to one, etc., which can be tricky (see Exercise 11.5). In such cases, it is often much simpler (but not always faster) to use an algorithm called expectation maximization , or EM for short (Dempster et al. 1977; Meng and van Dyk 1997; McLachlan and Krishnan 1997). This is a simple iterative algorithm, often with closed-form updates at each step. Furthermore, the algorithm automatically enforce the required constraints. 

> 数据不完整情况下，仍然可以使用基于梯度的方法，寻找负对数似然的局部极小值，但这一般需要施加一定约束，例如协方差矩阵半正定，混合权重和为 1 等
> EM 算法相较下则更加简单，EM 算法每步迭代都是闭式的，同时该算法自动施加了所需要的约束

EM exploits the fact that if the data were fully observed, then the ML/ MAP estimate would be easy to compute. In particular, EM is an iterative algorithm which alternates between inferring the missing values given the parameters (E step), and then optimizing the parameters given the “ﬁlled in” data (M step). We give the details below, followed by several examples. We end with a more theoretical discussion, where we put the algorithm in a larger context. See Table 11.2 for a summary of the applications of EM in this book. 
>  EM 算法在给定参数推理缺失值和给定完全数据优化参数之间迭代

### 11.4.1 Basic idea 
Let  $\mathbf{x}_{i}$ be the visible or observed variables in case  $i$ , and let  $\mathbf{z}_{i}$ be the hidden or missing variables. The goal is to maximize the log likelihood of the observed data: 

$$
\ell (\pmb\theta)=\sum_{i=1}^{N}\log p (\mathbf x_{i}|\pmb\theta)=\sum_{i=1}^{N}\log\left[\sum_{\mathbf z_{i}}p (\mathbf x_{i},\mathbf z_{i}|\pmb\theta)\right]\tag{11.17}
$$ 
Unfortunately this is hard to optimize, since the log cannot be pushed inside the sum. 

>  带有缺失变量时，对数似然的形式如上，它需要对缺失变量的取值求和

EM gets around this problem as follows. Deﬁne the complete data log likelihood to be 

$$
\ell_{c}(\pmb\theta)\triangleq\sum_{i=1}^{N}\log p (\mathbf x_{i},\mathbf z_{i}|\pmb\theta)\tag{11.18}
$$ 
This cannot be computed, since  $\mathbf{z}_{i}$ is unknown. So let us deﬁne the expected complete data log likelihood as follows: 

$$
Q (\pmb{\theta},\pmb{\theta}^{t-1})=\mathbb{E}\left[\ell_{c}(\pmb{\theta})\big|\mathcal{D},\pmb{\theta}^{t-1}\right]\tag{11.19}
$$ 
where $t$ is the current iteration number.  $Q$ is called the auxiliary function . The expectation is taken parameters,  $\pmb{\theta}^{t-1}$ , and the observed data  $\mathcal{D}$ . 

>  定义期望对数似然 $Q$ 如上，将 $Q$ 称为辅助函数
>  期望是相对于参数 $\pmb \theta^{t-1}$ 和观测数据 $\mathcal D$ 所取

The goal of the E step is to compute $Q (\pmb \theta,\pmb \theta^{t-1})$ , or rather, the terms inside of it which the MLE depends on; these are known as the expected sufficient statistics or ESS. In the ${M}$ step , we optimize the Q function wrt $\theta$ : 

$$
\pmb\theta^{t}=\arg\operatorname*{max}_{\pmb \theta}Q (\pmb \theta,\pmb \theta^{t-1})\tag{11.20}
$$ 
>  E 步的目标是计算辅助函数 $Q$，或者说计算期望充分统计量，继而计算期望对数似然
>  M 步的目标是相对于 $\pmb \theta$ 优化辅助函数

To perform MAP estimation, we modify the M step as follows: 

$$
\pmb \theta^{t}=\mathop{\mathrm{argmax}}_{\pmb \theta} Q (\pmb \theta,\pmb \theta^{t-1})+\log p (\pmb \theta)\tag{11.21}
$$

The E step remains unchanged. 

>  如果执行 MAP 估计，则 M 步额外添加一项，E 步不变

In Section 11.4.7 we show that the EM algorithm monotonically increases the log likelihood of the observed data (plus the log prior, if doing MAP estimation), or it stays the same. So if the objective ever goes down, there must be a bug in our math or our code. (This is a surprisingly useful debugging tool!) 
>  EM 算法中观测数据的对数似然单调不减，如果 EM 迭代中，目标函数降低了，说明编码或推导有问题

Below we explain how to perform the E and M steps for several simple models, that should make things clearer. 

### 11.4.2 EM for GMMs 
In this section, we discuss how to ﬁt a mixture of Gaussians using EM. Fitting other kinds of mixture models requires a straightforward modiﬁcation — see Exercise 11.3. We assume the number of mixture components,  $K$ , is known (see Section 11.5 for discussion of this point). 

#### 11.4.2.1 Auxiliary function 
The expected complete data log likelihood is given by 

$$
\begin{array}{r c l}{Q (\pmb \theta,\pmb \theta^{(t-1)})}&{\triangleq}&{\mathbb{E}\left[\displaystyle\sum_{i}\log p (\mathbf{x}_{i}, z_{i}|\pmb \theta)\right]}\\ &{=}&{\displaystyle\sum_{i}\mathbb{E}\left[\log\left[\displaystyle\prod_{k=1}^{K}(\pi_{k}p (\mathbf{x}_{i}|\pmb \theta_{k}))^{\mathbb{I}(z_{i}=k)}\right]\right]}\\ &{=}&{\displaystyle\sum_{i}\sum_{k}\mathbb{E}\left[\mathbb{I}(z_{i}=k)\right]\log\left[\pi_{k}p (\mathbf{x}_{i}|\pmb \theta_{k})\right]}\\ &{=}&{\displaystyle\sum_{i}\sum_{k}p (z_{i}=k|\mathbf{x}_{i},\pmb \theta^{t-1})\log[\pi_{k}p (\mathbf{x}_{i}|\pmb \theta_{k})]}\\ &{=}&{\displaystyle\sum_{i}\sum_{k}r_{i k}\log\pi_{k}+\sum_{i}\sum_{k}r_{i k}\log p (\mathbf{x}_{i}|\pmb \theta_{k})}\end{array}
$$ 
where  $r_{i k}\,\triangleq\, p (z_{i}\,=\, k|\mathbf{x}_{i},\pmb{\theta}^{(t-1)})$ is the responsibility that cluster  $k$ takes for data point  $i$ . This is computed in the E step, described below. 

#### 11.4.2.2 E step 
The E step has the following simple form, which is the same for any mixture model: 

$$
\begin{array}{r c l}{r_{i k}}&{=}&{\displaystyle\frac{\pi_{k}p (\mathbf{x}_{i}|\pmb{\theta}_{k}^{(t-1)})}{\sum_{k^{\prime}}\pi_{k^{\prime}}p (\mathbf{x}_{i}|\pmb{\theta}_{k^{\prime}}^{(t-1)})}}\end{array}
$$

#### 11.4.2.3 M step 
In the M step, we optimize $Q$ wrt  $\pi$ and the $\theta_{k}$ . For  $\pi$ , we obviously have 

$$
\pi_{k}\;\;\;=\;\;\;\frac{1}{N}\sum_{i}r_{i k}=\frac{r_{k}}{N}
$$ 
where $k$ . 

To derive the M step for the  $\pmb{\mu}_{k}$ and  $\Sigma_{k}$ terms, we look at the parts of  $Q$ that depend on  $\pmb{\mu}_{k}$ and $\Sigma_{k}$ . We see that the result is 

$$
\begin{array}{c c l}{\ell (\pmb{\mu}_{k},\pmb{\Sigma}_{k})}&{=}&{\displaystyle\sum_{k}\sum_{i}r_{i k}\log p (\mathbf{x}_{i}|\pmb{\theta}_{k})}\\ &{=}&{\displaystyle-\frac{1}{2}\sum_{i}r_{i k}\left[\log|\pmb{\Sigma}_{k}|+(\mathbf{x}_{i}-\pmb{\mu}_{k})^{T}\pmb{\Sigma}_{k}^{-1}(\mathbf{x}_{i}-\pmb{\mu}_{k})\right]}\end{array}
$$ 
This is just a weighted version of the standard problem of computing the MLEs of an MVN (see Section 4.1.3). One can show (Exercise 11.2) that the new parameter estimates are given by 

$$
\begin{array}{c c l}{\pmb{\mu}_{k}}&{=}&{\displaystyle\frac{\sum_{i}r_{i k}\mathbf{x}_{i}}{r_{k}}}\\ {\pmb{\Sigma}_{k}}&{=}&{\displaystyle\frac{\sum_{i}r_{i k}(\mathbf{x}_{i}-\pmb{\mu}_{k})(\mathbf{x}_{i}-\pmb{\mu}_{k})^{T}}{r_{k}}=\frac{\sum_{i}r_{i k}\mathbf{x}_{i}\mathbf{x}_{i}^{T}}{r_{k}}-\pmb{\mu}_{k}\pmb{\mu}_{k}^{T}}\end{array}
$$ 
These equations make intuitive sense: the mean of cluster  $k$ is just the weighted average of all points assigned to cluster  $k$ , and the covariance is proportional to the weighted empirical scatter matrix. 

After computing the new estimates, we set  $\pmb{\theta}^{t}=(\pi_{k},\mu_{k},\Sigma_{k})$ for $k=1:K$ , and go to the next E step. 

#### 11.4.2.4 Example 

mple of the algo action is shown in Figure 11.11. We start with  $\mu_{1}=(-1,1)$ , $\pmb{\Sigma}_{1}=\mathbf{I},$ ,  $\mu_{2}=(1,-1)$ , $\pmb{\Sigma}_{2}=\mathbf{I}$ . We color code points such that blue points come from cluster 1 and red points from cluster 2. More precisely, we set the color to 

$$
{\mathrm{coker}}(i)=r_{i1}{\mathrm{blue}}+r_{i2}{\mathrm{red}}
$$ 

so ambiguous points appear purple. After 20 iterations, the algorithm has converged on a good clustering. (The data was standardized, by removing the mean and dividing by the standard deviation, before processing. This often helps convergence.) 

#### 11.4.2.5 K-means algorithm 

There is a popular variant of the EM algorithm for GMMs known as the K-means algorithm , which we now discuss. Consider a GMM in which we make the following assumptions:  $\pmb{\Sigma}_{k}=$ $\sigma^{2}\mathbf{I}_{D}$ is ﬁxed, and  $\pi_{k}\,=\, 1/K$ is ﬁxed, so only the cluster centers,  $\pmb{\mu}_{k}\,\in\,\mathbb{R}^{\bar{D}}$ , have to be estimated. Now consider the following delta-function approximation to the posterior computed during the E step: 

$$
p (z_{i}=k|\mathbf{x}_{i},\pmb\theta)\approx\mathbb{I}(k=z_{i}^{*})
$$ 

where  $z_{i}*=\operatorname{argmax}_{k}p (z_{i}=k|\mathbf{x}_{i},\boldsymbol{\theta})$ . This is sometimes called hard EM , since we are making a hard assignment of points to clusters. Since we assumed an equal spherical covariance matrix for each cluster, the most probable cluster for  $\mathbf{x}_{i}$ can be computed by ﬁnding the nearest prototype: 

$$
z_{i}^{*}=\arg\operatorname*{min}_{k}||\mathbf{x}_{i}-\pmb{\mu}_{k}||_{2}^{2}
$$ 

Hence in each E step, we must ﬁnd the Euclidean distance between $N$ data points and $K$ cluster centers, which takes  $O (N K D)$ time. However, this can be sped up using various techniques, such as applying the triangle inequality to avoid some redundant computations (Elkan 2003). Given the hard cluster assignments, the M step updates each cluster center by computing the mean of all points assigned to it: 

$$
\boldsymbol{\mu}_{k}=\frac{1}{N_{k}}\sum_{i: z_{i}=k}\mathbf{x}_{i}
$$ 

See Algorithm 5 for the pseudo-code. 

![](images/d6bca854b5212bf817c23265184af843dcaa51e7e6464665ad225e37ba3d9703.jpg) 
Figure 11.11 Illustration of the EM for a GMM applied to the Old Faithful data. (a) Initial (random) values of the parameters. (b) Posterior responsibility of each point computed in the ﬁrst E step. The degree of redness indicates the degree to which the point belongs to the red cluster, and similarly for blue; this purple points have a roughly uniform posterior over clusters. (c) We show the updated parameters after the ﬁrst M step. (d) After 3 iterations. (e) After 5 iterations. (f) After 16 iterations. Based on (Bishop 2006a) Figure 9.8. Figure generated by mix Gauss Demo Faithful . 

Algorithm 11.1: K-means algorithm 

![](images/ef52f0fcb24456bfc6c573e45132062bf03b2f31c4ab17ee1d5b44f929c0f308.jpg) 

![](images/0f65d9d0882d5174dd65f9f190ce88a4e27d74fa896192f8933f835545451dcb.jpg) 
Figure 11.12 An image compressed using vector quantization with a codebook of size  $K$ . (a)  $K=2$ . (b) $K=4$ . Figure generated by vqDemo . 

#### 11.4.2.6 Vector quantization 

Since K-means is not a proper EM algorithm, it is not maximizing likelihood. Instead, it can be interpreted as a greedy algorithm for approximately minimizing a loss function related to data compression, as we now explain. 

Suppose we want to perform lossy compression of some real-valued vectors, $\mathbf{x}_{i}\in\mathbb{R}^{D}$ . A very simple approach to this is to use vector quantization or VQ . The basic idea is to replace each real-valued ve r  $\mathbf{x}_{i}\in\mathbb{R}^{D}$  crete symbol  $z_{i}\in\{1,.\,.\,.\,, K\}$ , which is an index into a codebook of K prototypes, $\pmb{\mu}_{k}\in\mathbb{R}^{D}$ ∈ . Each data vector is encoded by using the index of the most similar prototype, where similarity is measured in terms of Euclidean distance: 

$$
\begin{array}{r c l}{\mathrm{enched}(\mathbf{x}_{i})}&{=}&{\arg\operatorname*{min}_{k}||\mathbf{x}_{i}-\pmb{\mu}_{k}||^{2}}\end{array}
$$ 

We can deﬁne a cost function that measures the quality of a codebook by computing the reconstruction error or distortion it induces: 

$$
J (\pmb{\mu},\mathbf{z}|K,\mathbf{X})\triangleq\frac{1}{N}\sum_{i=1}^{N}||\mathbf{x}_{i}-\mathrm{decode}(\mathrm{encode}(\mathbf{x}_{i}))||^{2}=\frac{1}{N}\sum_{i=1}^{N}||\mathbf{x}_{i}-\pmb{\mu}_{z_{i}}||^{2}
$$ 

where $\operatorname{decade}(k)=\mu_{k}$ . The K-means algorithm can be thought of as a simple iterative scheme for minimizing this objective. 

Of course, we can achieve zero distortion if we assign one prototype to every data vector, but that takes  $O (N D C)$ space, where  $N$ is the number of real-valued data vectors, each of length  $D$ , and  $C$ is the number of bits needed to represent a real-valued scalar (the quantization accuracy). However, in many data sets, we see similar vectors repeatedly, so rather than storing them many times, we can store them once and then create pointers to them. Hence we can reduce the space requirement to  $O (N\log_{2}K+K D C)$ : the  $O (N\log_{2}K)$ term arises because each of the  $N$ data vectors needs to specify which of the  $K$ codewords it is using (the pointers); and the  $O (K D C)$ term arises because we have to store each codebook entry, each of which is a  $D$ -dimensional vector. Typically the ﬁrst term dominates the second, so we can approximate the rate of the encoding scheme (number of bits needed per object) as  $O (\log_{2}K)$ , which is typically much less than  $O (D C)$ . 

One application of VQ is to image compr Consider the $N=200\times320=64,000$ pixel image in Figure 11.12; this is gray-scale, so $D=1$ . If we use one byte to represent each pixel (a gray-scale intensity of 0 to 255), then  $C=8$ , so we need  $N C=512,000$ bits to represent the image. For the compressed image, we need  $N\log_{2}K+K C$ bits. For  $K=4$ , this is about $128\mathrm{kb}$ , a factor of 4 compression. For  $K=8$ , this is about $192\mathrm{kb}$ , a factor of 2.6 compression, at negligible perceptual loss (see Figure 11.12 (b)). Greater compression could be achieved if we modelled spatial correlation between the pixels, e.g., if we encoded 5x5 blocks (as used by JPEG). This is because the residual errors (differences from the model’s predictions) would be smaller, and would take fewer bits to encode. 

#### 11.4.2.7 Initialization and avoiding local minima 

Both K-means and EM need to be initialized. It is common to pick $K$ data points at random, and to make these be the initial cluster centers. Or we can pick the centers sequentially so as to try to “cover” the data. That is, we pick the initial point uniformly at random. Then each subsequent point is picked from the remaining points with probability proportional to its squared distance to the points’s closest cluster center. This is known as farthest point clustering (Gonzales 1985), or k-means $^{++}$ (Arthur and Vassilvitskii 2007; Bahmani et al. 2012). Surprisingly, this simple trick can be shown to guarantee that the distortion is never more than  $O (\log K)$ worse than optimal

 (Arthur and Vassilvitskii 2007). 

An heuristic that is commonly used in the speech recognition community is to incrementally

 “grow” GMMs: we initially give each cluster a score based on its mixture weight; after each round of training, we consider splitting the cluster with the highest score into two, with the new centroids being random perturbations of the original centroid, and the new scores being half of the old scores. If a new cluster has too small a score, or too narrow a variance, it is removed. We continue in this way until the desired number of clusters is reached. See (Figueiredo and Jain 2002) for a similar incremental approach. 

#### 11.4.2.8 MAP estimation 

As usual, the MLE may overﬁt. The overﬁtting problem is particularly severe in the case of GMMs. To understand the problem, suppose for simplicity that  $\pmb{\Sigma}_{k}=\sigma_{k}^{2}I$ , and that $K=2$ . It is possible to get an inﬁnite likelihood by assigning one of the centers, say  $\mu_{2}$ , to a single data point, say  $\mathbf{x}_{1}$ , since then the 1st term makes the following contribution to the likelihood: 

$$
\mathcal{N}(\mathbf{x}_{1}|\mu_{2},\sigma_{2}^{2}I)=\frac{1}{\sqrt{2\pi\sigma_{2}^{2}}}e^{0}
$$ 

![](images/bf21a05b1e5c77a873531dd5454c1feb20967874ecb1539932d71b1a08b1c418.jpg) 
Figure 11.13 (a) Illustration of how singularities can arise in the likelihood function of GMMs. Based on (Bishop 2006a) Figure 9.7. Figure generated by mix Gauss Singularity . (b) Illustration of the beneﬁt of MAP estimation vs ML estimation when ﬁtting a Gaussian mixture model. We plot the fraction of times (out of 5 random trials) each method encounters numerical problems vs the dimensionality of the problem, for $N=100$ samples. Solid red (upper curve): MLE. Dotted black (lower curve): MAP. Figure generated by mixGaussMLvsMAP . 

Hence we can drive this term to inﬁnity by letting  $\sigma_{2}\rightarrow0$ , as shown in Figure 11.13 (a). We will call this the “collapsing variance problem”. 

An easy solution to this is to perform MAP estimation. The new auxiliary function is the expected complete data log-likelihood plus the log prior: 

$$
\mathcal{V}^{\prime}(\pmb{\theta},\pmb{\theta}^{o l d})=\left[\sum_{i}\sum_{k}r_{i k}\log\pi_{i k}+\sum_{i}\sum_{k}r_{i k}\log p (\mathbf{x}_{i}|\pmb{\theta}_{k})\right]+\log p (\pmb{\pi})+\sum_{k}\log p (\pmb{\pi}_{i}|\pmb{\theta}_{k})
$$ 

Note that the E step remains unchanged, but the M step needs to be modiﬁed, as we now explain. 

For the prior on the mixture weights, it is natural to use a Dirichlet prior,  $\pi\sim\operatorname{Dir}(\alpha)$ , since this is conjugate to the categorical distribution. The MAP estimate is given by 

$$
\begin{array}{c c c}{\pi_{k}}&{=}&{\displaystyle{\frac{r_{k}+\alpha_{k}-1}{N+\sum_{k}\alpha_{k}-K}}}\end{array}
$$ 

If we use a uniform prior,  $\alpha_{k}=1$ , this reduces to Equation 11.28. 

The prior on the parameters of the class conditional densities, $p (\pmb{\theta}_{k})$ , depends on the form of the class conditional densities. We discuss the case of GMMs below, and leave MAP estimation for mixtures of Bernoullis to Exercise 11.3. 

For simplicity, let us consider a conjugate prior of the form 

$$
p (\pmb{\mu}_{k},\pmb{\Sigma}_{k})=\mathrm{NIW}(\pmb{\mu}_{k},\pmb{\Sigma}_{k}|\mathbf{m}_{0},\kappa_{0},\nu_{0},\mathbf{S}_{0})
$$ 

From Section 4.6.3, the MAP estimate is given by 

$$
\begin{array}{c c l}{\hat{\mu}_{k}}&{=}&{\displaystyle\frac{r_{k}\overline {{\mathbf x}} _{k}+\kappa_{0}\mathbf m_{0}}{r_{k}+\kappa_{0}}}\\ &{\overline {{\mathbf x}} _{k}}&{\triangleq}&{\displaystyle\frac{\sum_{i}r_{i k}\mathbf x_{i}}{r_{k}}}\\ {\hat{\mathbf X}_{k}}&{=}&{\displaystyle\frac{\mathbf S_{0}+\mathbf S_{k}+\frac{\kappa_{0}r_{k}}{\kappa_{0}+r_{k}}(\overline {{\mathbf x}} _{k}-\mathbf m_{0})(\overline {{\mathbf x}} _{k}-\mathbf m_{0})^{T}}{\nu_{0}+r_{k}+D+2}}\\ {\mathbf S_{k}}&{\triangleq}&{\displaystyle\sum_{i}r_{i k}(\mathbf x_{i}-\overline {{\mathbf x}} _{k})(\mathbf x_{i}-\overline {{\mathbf x}} _{k})^{T}}\end{array}
$$ 

We now illustrate the beneﬁts of using MAP estimation instead of ML estimation in the context of GMMs. We apply EM to some synthetic data in  $D$ dimensions, using either ML or MAP estimation. We count the trial as a “failure” if there are numerical issues involving singular matrices. For each dimensionality, we conduct 5 random trials. The results are illustrated in Figure 11.13 (b) using  $N=100$ . We see that as soon as  $D$ becomes even moderately large, ML estimation crashes and burns, whereas MAP estimation never encounters numerical problems. 

When using MAP estimation, we need to specify the hyper-parameters. Here we mention some simple heuristics for setting them (Fraley and Raftery 2007, p163). We can set  $\kappa_{\mathrm{0}}\,=\, 0$ , so that the  $\pmb{\mu}_{k}$ are unregularized, since the numerical problems only arise from  $\Sigma_{k}$ . In this case, the MAP estimates simplify to  $\hat{\pmb{\mu}}_{k}=\overline {{\mathbf{x}} }_{k}$ and  $\begin{array}{r}{\hat{\Sigma}_{k}\overset{\cdot}{=}\frac{\mathbf{S}_{0}+\mathbf{S}_{k}}{\nu_{0}+r_{k}+D+2}\overset{\cdot}{}}\end{array}$ , which is not quite so scary-looking. 

Now we discuss how to set  $\mathbf{S}_{0}$ . One possibility is to use 

$$
\mathbf{S}_{0}=\frac{1}{K^{1/D}}\mathrm{diag}(s_{1}^{2},.\,.\,.\,, s_{D}^{2})
$$ 

$\begin{array}{r}{s_{j}\,=\, (1/N)\sum_{i=1}^{N}(x_{i j}\,-\,\overline {{x}} _{j})^{2}}\end{array}$ − is the pooled variance for dimension  $j$ . (The reason for the $\frac{1}{K^{1/D}}$ term is that the resulting volume of each ellipsoid is then given by  $|\mathbf{S}_{0}|\ =$ $\textstyle{\frac{1}{K}}|\mathrm{diag}(s_{1}^{2},.\,.\,.\,, s_{D}^{2})|$  | | .) The parameter  $\nu_{0}$ controls how strongly we believe this prior. The weakest prior we can use, while still being proper, is to set  $\nu_{0}=D+2$ , so this is a common choice. 
# 17 Markov and hidden Markov models 
## 17.1 Introduction 
In this chapter, we discuss probabilistic models for sequences of observations, $X_{1},\dots,X_{T}$ , of arbitrary length $T$ . Such models have applications in computational biology, natural language processing, time series forecasting, etc. We focus on the case where we the observations occur at discrete “time steps”, although “time” may also refer to locations within a sequence. 
> 本章讨论关于任意长度 $T$ 的观测序列 $X_1,\dots, X_T$ 的概率模型，我们专注于离散的“时间步”
## 17.2 Markov models 
Recall from Section 10.2.2 that the basic idea behind a Markov chain is to assume that $X_{t}$ captures all the relevant information for predicting the future (i.e., we assume it is a sufficient statistic). 
> Markov 链的基本思想就是 $X_i$ 已经捕获了需要预测未来的所有相关信息/充足统计量

If we assume discrete time steps, we can write the joint distribution as follows: 

$$
p(X_{1:T})=p(X_{1})p(X_{2}|X_{1})p(X_{3}|X_{2})\dots=p(X_{1})\prod_{t=2}^{T}p(X_{t}|X_{t-1})\tag{17.1}
$$ 
This is called a Markov chain or Markov model . 
> 对 $P (X_{1:T})$ 的按照时间步顺序的链式分解就是 Markov 模型（链式分解时，除了上一个时间步的父变量，其余父变量都假设与当前变量条件独立）

If we assume the transition function $p(X_{t}|X_{t-1})$ is independent of time, then the chain is called homogeneous , stationary , or time-invariant . This is an example of parameter tying , since the same parameter is shared by multiple variables. This assumption allows us to model an arbitrary number of variables using a fixed number of parameters; such models are called stochastic processes . 
> 如果假设转移函数 $p (X_t\mid X_{t-1})$ 独立于时间 $t$，则模型称为一致的/静止的/时间不变的，此时参数是共享的，因此可以用固定数量的参数建模任意数量的变量
> 这种类型的模型称为随机过程

If we assume that the observed variables are discrete, so $X_{t}\,\in\,\{1,\ldots,K\}$ , this is called a discrete-state or finite-state Markov chain. We will make this assumption throughout the rest of this section. 
> 如果随机变量为离散型，则模型称为离散状态/有限状态 Markov 链
### 17.2.1 Transition matrix 
When $X_{t}$ is discrete, so $X_{t}\;\in\;\{1,.\,.\,.\,,K\}$ , the conditional distribution $p(X_{t}|X_{t-1})$ written as a K $K\times K$ × matrix, known as the tra sition matrix A , where $A_{i j}~=~p(X_{t}~=~$ $j|X_{t-1}=i)$ is the probability of going from state i to state $j$ . Each row of the matrix sums to one, $\textstyle\sum_{j}A_{i j}=1$ , so this is called a stochastic matrix . 
> $X_t$ 离散时，条件分布 $p (X_t\mid X_{t-1})$ 可以写为 $K\times K$ 矩阵的 $A$，称其为转移矩阵
> $A_{ij} = p (X_t = j \mid X_{t-1} = i)$ 即从状态 $i$ 到 $j$ 的概率
> 矩阵的每一行求和 $\sum_j A_{ij} = \sum_j p (X_t = j\mid X_{t-1} = i) = 1$
> 因此矩阵 $A$ 也成为随机矩阵

![[ML A Prob Perspective-FIg17.1.png]]

A stationary, finite-state Markov chain is equivalent to a stochastic automaton . It is common to visualize such automata by drawing a directed graph, where nodes represent states and arrows represent legal transitions, i.e., non-zero elements of A . This is known as a state transition diagram . The weights associated with the arcs are the probabilities. For example, the following 2-state chain 
> 静态、有限状态的 Markov 链等价于一个随机自动机
> 可以用状态转移图表示，其中节点表示状态，边表示转移


$$
\mathbf A = \begin{pmatrix}
1-\alpha&\alpha\\
\beta&1-\beta
\end{pmatrix}\tag{17.2}
$$

is illustrated in Figure 17.1 (left). The following 3-state chain 

$$
\mathbf{A}={\left(\begin{array}{l l l}{A_{11}}&{A_{12}}&{0}\\ {0}&{A_{22}}&{A_{23}}\\ {0}&{0}&{1}\end{array}\right)}\tag{17.3}
$$ 
is illustrated in Figure $17.1(\mathrm{right})$ . This is called a left-to-right transition matrix , and is commonly used in speech recognition (Section 17.6.2). 

The $A_{i j}$ element of the transition matrix specifies the probability of getting from $i$ to $j$ in one step. The $n$ -step transition matrix $\mathbf{A}(n)$ is defined as 
> 定义 $n$ 步转移矩阵 $\mathbf A (n)$，$A_{ij}(n) = p (X_{t+n} = j \mid X_t = i)$，即 $X_t$ 经过 $n$ 个时间步之后状态从 $i$ 转移到 $j$ 的概率

$$
A_{i j}(n)\triangleq p(X_{t+n}=j|X_{t}=i)\tag{17.4}
$$ 
which is the probability of getting from $i$ to $j$ in exactly $n$ steps. Obviously $\mathbf{A}(1)=\mathbf{A}$ . The Chapman-Kolmogorov equations state that 

$$
A_{i j}(m+n)=\sum_{k=1}^{K}A_{i k}(m)A_{k j}(n)
$$ 
In words, the probability of getting from $i$ to $j$ in $m+n$ steps is just the probability of getting from $i$ to $k$ in $m$ steps, and then from $k$ to $j$ in $n$ steps, summed up over all $k$ . 
> 先走 $m$ 步转移到状态 $k$ ，再走 $n$ 步转移到状态 $j$，对所有的 $k$ 求和

We can write the above as a matrix multiplication 
> 因此可以直接写为矩阵乘法

$$
\mathbf{A}(m+n)=\mathbf{A}(m)\mathbf{A}(n)
$$ 
Hence 

$$
\mathbf{A}(n)=\mathbf{A}\,\mathbf{A}(n-1)=\mathbf{A}\,\mathbf{A}\,\mathbf{A}(n-2)=\cdot\cdot\cdot=\mathbf{A}^{n}
$$ 
Thus we can simulate multiple steps of a Markov chain by “powering up” the transition matrix. 
> 因此对转移矩阵乘上幂次就可以表示 Markov 链上的多步转移

```
SAYS IT’S NOT IN THE CARDS LEGENDARY RECONNAISSANCE BY ROLLIE DEMOCRACIES UNSUSTAINABLE COULD STRIKE REDLINING VISITS TO PROFIT BOOKING WAIT HERE AT MADISON SQUARE GARDEN COUNTY COURTHOUSE WHERE HEHAD BEEN DONE IN THREE ALREADY IN ANY WAY IN WHICH A TEACHER 
```

Table 17.1 Example output from an 4 -gram word model, trained using backoff smoothing on the Broadcast News corpus. The first 4 words are specified by hand, the model generates the 5th word, and then the results are fed back into the model.
### 17.2.2 Application: Language modeling 
One important application of Markov models is to make statistical language models , which are probability distributions over sequences of words. We define the state space to be all the words in English (or some other language). The marginal probabilities $p(X_{t}=k)$ are called unigram statistics . If we use a first-order Markov model, then $p(X_{t}=k|X_{t-1}=j)$ is called a bigram model . If we use a second-order Markov model, then $p(X_{t}\,=\,k|X_{t-1}\,=\,j,X_{t-2}\,=\,i)$ is called a trigram model . And so on. 
In general these are called $\mathbf{n}$ -gram models . 
> Markov 模型可用作统计语言模型（在单词序列上的概率分布）
> 定义状态空间为英语中所有词，边际概率 $p(X_t = k)$ 称为 unigram 统计量，如果使用一阶 Markov 模型，则 $p(X_{t}=k|X_{t-1}=j)$ 称为 bigram 模型，使用二阶 Markov 模型，则 $p(X_{t}\,=\,k|X_{t-1}\,=\,j,X_{t-2}\,=\,i)$ 称为 trigram 模型
> 统称为 n-gram 模型

 For example, Figure 17.2 shows 1-gram and 2-grams counts for the letters $\{a,\cdot\cdot\cdot,z,-\}$ (where - represents space) estimated from Darwin’s On The Origin Of Species . 
 
 Language models can be used for several things, such as the following:
> 语言模型可以用于：
 
- Sentence completion A language model can predict the next word given the previous words in a sentence. This can be used to reduce the amount of typing required, which is particularly important for disabled users (see e.g., David Mackay’s Dasher system 1 ), or uses of mobile devices.
> 句子补全：给定上下文预测下一个词

 - Data compression Any density model can be used to define an encoding scheme, by assigning short codewords to more probable strings. The more accurate the predictive model, the fewer the number of bits it requires to store the data.
> 数据压缩：为概率更高的 string 赋予更短的 codeword

 - Text classification Any density model can be used as a class-conditional density and hence turned into a (generative) classifier. Note that using a 0-gram class-conditional density (i.e., only unigram statistics) would be equivalent to a naive Bayes classifier (see Section 3.5).
 > 文本分类：generative classifier

 - Automatic essay writing One can sample from $p(x_{1:t})$ to generate artificial text. This is one way of assessing the quality of the model. In Table 17.1, we give an example of text generated from a 4-gram model, trained on a corpus with 400 million words. ((Tomas et al. 2011) describes a much better language model, based on recurrent neural networks, which generates much more semantically plausible text.) 
 > 自动散文撰写：从 $p (x_{1:t})$ 采样来生成文本
#### 17.2.2.1 MLE for Markov language models 
We now discuss a simple way to estimate the transition matrix from training data. 
> 介绍根据训练数据估计转移矩阵的方式

The probability of any particular sequence of length $T$ is given by 
> 任意长度为 $T$ 的特定序列的概率

$$
\begin{array}{r c l}{p(x_{1:T}|\pmb{\theta})}&{=}&{\pi(x_{1})A(x_{1},x_{2})\dots A(x_{T-1},x_{T})}\\ &{=}&{\displaystyle\prod_{j=1}^{K}(\pi_{j})^{\mathbb{I}(x_{1}=j)}\prod_{t=2}^{T}\prod_{j=1}^{K}\prod_{k=1}^{K}(A_{j k})^{\mathbb{I}(x_{t}=k,x_{t-1}=j)}}\end{array}
$$ 
Hence the log-likelihoo of a set of sequences $\mathcal{D}=\left(\mathbf{x}_{1},\cdot\cdot\cdot,\mathbf{x}_{N}\right)$ , where $\mathbf{x}_{i}=\left(x_{i1},.\,.\,,x_{i,T_{i}}\right)$ is a sequence of length $T_{i}$ , is given by 

$$
\begin{array}{l l l}{\log p(\mathcal{D}|\pmb{\theta})}&{=}&{\displaystyle\sum_{i=1}^{N}\log p(\mathbf{x}_{i}|\pmb{\theta})=\sum_{j}N_{j}^{1}\log\pi_{j}+\sum_{j}\sum_{k}N_{j k}\log A_{j k}}\end{array}
$$ 
where we define the following counts: 

$$
N_{j}^{1}\triangleq\sum_{i=1}^{N}\mathbb{I}(x_{i1}=j),\ \ N_{j k}\triangleq\sum_{i=1}^{N}\sum_{t=1}^{T_{i}-1}\mathbb{I}(x_{i,t}=j,x_{i,t+1}=k)
$$ 
Hence we can write the MLE as the normalized counts: 

$$
\hat{\pi}_{j}=\frac{N_{j}^{1}}{\sum_{j}{N_{j}^{1}}},\,\,\,\hat{A}_{j k}=\frac{N_{j k}}{\sum_{k}{N_{j k}}}
$$ 
These results can be extended in a straightforward way to higher order Markov models. 

However, the problem of zero-counts becomes very acute whenever the number of states $K$ , and/or the order of the chain, $n$ , is large. An n-gram models has $O(K^{n})$ parameters. If we have $K\sim50,000$ words in our vocabulary, then a bi-gram model will have about 2.5 billion free parameters, corresponding to all possible word pairs. It is very unlikely we will see all of these in our training data. However, we do not want to predict that a particular word string is totally impossible just because we happen not to have seen it in our training text — that would be a severe form of overfitting. 

A simple solution to this is to use add-one smoothing, where we simply add one to all the empirical counts before normalizing. The Bayesian justification for this is given in Section 3.3.4.1. However add-one smoothing assumes all n-grams are equally likely, which is not very realistic. A more sophisticated Bayesian approach is discussed in Section 17.2.2.2. 

An alternative to using smart priors is to gather lots and lots of data. For example, Google has fit n-gram models (for $n=1:5$ ) based on one trillion words extracted from the web. Their data, which is over 100GB when uncompressed, is publically available. An example of their data, for a set of 4-grams, is shown below. 

serve as the incoming 92 serve as the incubator 99 serve as the independent 794 serve as the index 223 serve as the indication 72 serve as the indicator 120 serve as the indicators 45 serve as the indispensable 111 serve as the indispensible 40 serve as the individual 234 

Although such an approach, based on “brute force and ignorance”, can be successful, it is rather unsatisfying, since it is clear that this is not how humans learn (see e.g., (Tenenbaum and $\mathrm{Xe}~2000)$ ). A more refined Bayesian approach, that needs much less data, is described in Section 17.2.2.2. 

#### 17.2.2.2 Empirical Bayes version of deleted interpolation 

A common heuristic used to fix the sparse data problem is called deleted interpolation (Chen and Goodman 1996). This defines the transition matrix as a convex combination of the bigram frequencies $f_{j k}=N_{j k}/N_{j}$ and the unigram frequencies $f_{k}=N_{k}/N$ : 

$$
A_{j k}=(1-\lambda)f_{j k}+\lambda f_{k}
$$ 

The term $\lambda$ is usually set by cross validation. There is also a closely related technique called backoff smoothing ; the idea is that if $f_{j k}$ is too small, we “back off” to a more reliable estimate, namely $f_{k}$ . 

We will now show that the deleted interpolation heuristic is an approximation to the predic- tions made by a simple hierarchical Bayesian model. Our presentation follows (McKay and Peto 1995). First, let us use an independent Dirichlet prior on each row of the transition matrix: 

$$
\mathbf{A}_{j}\sim\mathrm{Dir}\bigl(\alpha_{0}m_{1},\dots,\alpha_{0}m_{K}\bigr)=\mathrm{Dir}\bigl(\alpha_{0}\mathbf{m}\bigr)=\mathrm{Dir}(\alpha)
$$ 

ere $\mathbf{A}_{j}$ is row $j$ of the tr $\mathbf{m}$ the pri $\textstyle\sum_{k}m_{k}=1)$ ) and $\alpha_{0}$ is the prior strength. We will use the same prior for each row: see Figure 17.3. 

The posterior is given by $\mathbf{A}_{j}\sim\operatorname{Dir}(\alpha+\mathbf{N}_{j})$ ∼ , where $\mathbf{N}_{j}\,=\,\bigl(N_{j1},.\,.\,,N_{j K}\bigr)$ is the vector that records the number of times we have transitioned out of state j to each of the other states. From Equation 3.51, the posterior predictive density is 

$$
\rho(X_{t+1}=k|X_{t}=j,\mathcal{D})=\overline{{A}}_{j k}=\frac{N_{j k}+\alpha m_{k}}{N_{j}+\alpha_{0}}=\frac{f_{j k}N_{j}+\alpha m_{k}}{N_{j}+\alpha_{0}}=(1-\lambda_{j})f_{j k}-\alpha m_{k}=0
$$ 

$$
\lambda_{j}=\frac{\alpha}{N_{j}+\alpha_{0}}
$$ 

This is very similar to Equation 17.13 but not identical. The main difference is that the Bayesian model uses a context-dependent weight $\lambda_{j}$ to combine $m_{k}$ with the empirical frequency $f_{j k}$ , rather than a fixed weight $\lambda$ . This is like adaptive deleted interpolation. Furthermore, rather than backing off to the empirical marginal frequencies $f_{k}$ , we back off to the model parameter $m_{k}$ . 

The only remaining question is: what values should we use for $\alpha$ and $\mathbf{m?}$ Let’s use empirical Bayes. Since we assume each row of the transition matrix is a priori independent given $_{\alpha}$ , the marginal likelihood for our Markov model is found by applying Equation 5.24 to each row: 

$$
p(\mathcal{D}|\alpha)=\prod_{j}\frac{B(\mathbf{N}_{j}+\alpha)}{B(\alpha)}
$$ 

where $\mathbf{N}_{j}\,=\,\bigl(N_{j1},.\,.\,,N_{j K}\bigr)$ are the counts for leaving state $j$ and $B(\alpha)$ is the generalized beta function. 

We can fit this using the methods discussed in (Minka 2000e). However, we can also use the following approximation (McKay and Peto 1995, p12): 

$$
m_{k}\propto|\{j:N_{j k}>0\}|
$$ 

This says that the prior probability of word $k$ is given by the number of different contexts in which it occurs, rather than the number of times it occurs. To justify the reasonableness of this result, Mackay and Peto (McKay and Peto 1995) give the following example. 

![](images/e3b94bd6a3b1100f07d78aff151d05e93c3d772c123919c61fea738449ecfdda.jpg) 
Figure 17.3 A Markov chain in which we put a different Dirichlet prior on every row of the transition matrix A , but the hyperparameters of the Dirichlet are shared. 

Imagine, you see, that the language, you see, has, you see, a frequently occuring couplet ’you see’, you see, in which the second word of the couplet, see, follows the first word, you, with very high probability, you see. Then the marginal statistics, you see, are going to become hugely dominated, you see, by the words you and see, with equal frequency, you see. 

If we use the standard smoothing formula, Equation 17.13, then P (you | novel) and P (see | novel), for some novel context word not seen before, would turn out to be the same, since the marginal frequencies of ’you’ and ’see’ are the same (11 times each). However, this seems unreasonable. ’You’ appears in many contexts, so P (you | novel) should be high, but ’see’ only follows ’you’, so P (see | novel) should be low. If use the Bayes formula Equation 17.15, we will get this effect for free, since we back off to $m_{k}$ not $f_{k}$ , and $m_{k}$ will be large for ’you’ and small for ’see’ by Equation 17.18. 

Unfortunately, although elegant, this Bayesian model does not beat the state-of-the-art lan- guage model, known as interpolated Kneser-Ney (Kneser and Ney 1995; Chen and Goodman 1998). However, in (Teh 2006), it was shown how one can build a non-parametric Bayesian model which outperforms interpolated Kneser-Ney, by using variable-length contexts. In (Wood et al. 2009), this method was extended to create the “sequence memoizer”, which is currently (2010) the best-performing language model. 

#### 17.2.2.3 Handling out-of-vocabulary words 

While the above smoothing methods handle the case where the counts are small or even zero, none of them deal with the case where the test set may contain a completely novel word. In particular, they all assume that the words in the vocabulary (i.e., the state space of $X_{t.}$ ) is fixed and known (typically it is the set of unique words in the training data, or in some dictionary). 

![](images/ddd9a9f0ad5402b48a6c1931b4bd070cc6710d45b4061e1254dd0dff4e2b82ac.jpg) 
Figure 17.4 Some Markov chains. (a) A 3-state aperiodic chain. (b) A reducible 4-state chain. 

Even if all $\overline{{A}}_{j k}$ ’s are non-zero, none of these models will predict a novel word outside of this set, and hence will assign zero probability to a test sentence with an unfamiliar word. (Unfamiliar words are bound to occur, because the set of words is an open class. For example, the set of proper nouns (names of people and places) is unbounded.) 

A standard heuristic to solve this problem is to replace all novel words with the special symbol unk , which stands for “unknown”. A certain amount of probability mass is held aside for this event. 

A more principled solution would be to use a Dirichlet process, which can generate a countably infinite state space, as the amount of data increases (see Section 25.2.2). If all novel words are “accepted” as genuine words, then the system has no predictive power, since any misspelling will be considered a new word. So the novel word has to be seen frequently enough to warrant being added to the vocabulary. See e.g., (Friedman and Singer 1999; Griffiths and Tenenbaum 2001) for details. 