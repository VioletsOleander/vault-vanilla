---
completed: true
---
Site: https://friedmanroy.github.io/blog/2023/AIS/
Date: Nov 9 2023

Suppose you can’t find your keys. You know you left them in your apartment somewhere, but don’t remember where. This happens pretty often, so you have a keep note of the possible places you may have left your keys. You want to find out the probability that the keys are in a specific room of your apartment. Let’s call this room $R$ - mathematically speaking, you want to calculate:

$$P(\text{keys} \in R) = \int \mathbf{1}[x \in R] \cdot p_{\text{keys}}(x) dx\tag{1}$$

where $x \in \mathbb{R}^2$ is a two-dimensional coordinate where the keys may have been forgotten and $p_{\text{keys}}(x)$ is the PDF for the keys to be in the location $x$. The function $\mathbf{1}[x \in R]$ equals 1 if $x \in R$, otherwise it is 0. The above can be rewritten:

$$P(\text{keys in } R) = E_{x \sim p_{\text{keys}}}[\mathbf{1}[x \in R]]\tag{2}$$

So… how can we calculate (or approximate) this expectation? Which room is most probable to contain your lost keys?

While this example is a bit silly, the problem can be abstracted to fit many different situations. In this post, I’m going to show how it can be solved using Annealed Importance Sampling (AIS). Honestly speaking, when the data is 2-dimensional there are better ways to do this, but 2D allows for simple (intuitive!) visualizations, so let’s stick with our somewhat wonky example.

# Problem Statement
Let’s define the problem again, just a bit more generally.

We have some distribution over the domain $\mathcal{X}$:

$$p(x) = \frac{1}{Z}\tilde{p}(x)\tag{3}$$

where for a given $x$ we know how to calculate $\tilde{p}(x)$. In this example, I’m going to assume the normalizing constant $Z$ isn’t known. This setting matches a situation where you keep track of where the keys were left in the past and have a non-parametric formulation for the density $\tilde{p}(x)$, in which case $Z$ is hard to calculate.

I will also assume that there’s a function $f: \mathcal{X} \rightarrow \mathbb{R}$ and we want (for whatever reason) to calculate:

$$\bar{f} = E_{x \sim p(x)}[f(x)]\tag{4}$$

In our example, $p(x) = p_{\text{keys}}(x)$ and $f(x) = \mathbf{1}[x \in R]$.

The question is: how can we calculate $\bar{f}$? And as a secondary goal: is there a way to estimate $Z$ simultaneously, so we have access to the full (normalized) distribution?

>  我们要解决的情况:
>  已知未规范化的分布 $\tilde p(x)$，未知规范化常数 $Z$，想要计算定义于随机变量 $x$ 上的函数相对于 $p(x)$ 的期望值 $\bar f$

### First Approach
One way to find $\bar{f}$ and $Z$ is using _importance sampling_ (IS). In IS, a simple distribution $q(x)$ is chosen and the expectation in equation (4) is approximated according to (see [A.1](https://friedmanroy.github.io/blog/2023/AIS/#a1-importance-sampling) for more details):

$$
\begin{equation} \tilde{w}(x)=\frac{\tilde{p}(x)}{q(x)}\qquad\overline{f}\approx \frac{\sum_{x_i\sim q}\tilde{w}(x_i)f(x_i)}{\sum_{x_i\sim q}\tilde{w}(x_i)} \end{equation}
$$

The number $\tilde{w}(x)$ defines the **relative importance** of $x$ under our distribution of interest $\tilde{p}(x)$ and the simple distribution $q(x)$, which is why $\tilde{w}(x)$ are called _importance weights_.

>  估计规范化常数 $Z$ (以及 $\bar f$) 的一个常见方法就是重要性采样: 从简单分布 $q(x)$ 中采样，计算重要性权重 $\tilde w(x)$ 对来自于采样分布的样本加权

IS also let’s us approximate the **normalization constant** $Z$, using only the importance weights: $Z = \mathbb{E}_q[\tilde{w}(x)] \approx \frac{1}{N}\sum_{x_i \sim q}^N \tilde{w}(x_i)$.

While incredibly simple and easy to use, IS is actually pretty hard to calibrate. Here, calibration means choosing a distribution $q(x)$ that is similar in some sense to $p(x)$. The best we can do, after all, is $q(x)=p(x)$. In that case, all of the importance weights will be equal to 1 and we would get a perfect approximation of $\bar{f}$ and $Z$. No, usually a much simpler distribution $q(x)$ is chosen, and if $q(x) \gg p(x)$ in some region of space then many samples from $q(x)$ will end up with very low importance weights $\tilde{w}(x)$. In such a situation, an **enormous number of samples** has to be used in order to get a sound approximation.
>  IS 易于使用，但非常难以校准
>  校准的意思是选择一个近似于目标分布 $p(x)$ 的采样分布 $q(x)$
>  对采样分布 $q(x)$ 的选择是一个非常 tricky 的过程，如果在某个区域 $q(x) \gg p(x)$，那么该区域中的样本都将具有非常低的重要性权重 $\tilde w(x)$，进而需要极大量的样本才能得到可靠的估计

### Another Way
If we are solely interested in estimating the expectation in equation (4), then another alternative is available - as long as we have some method for producing samples from $p(x)$ using only the unnormalized function $\tilde{p}(x)$. If that is the case, then $M$ points $x_1, \cdots, x_M$ can be sampled and used to get an unbiased approximation of the expectation:

$$\bar{f} \approx \frac{1}{M}\sum_{i:x_i \sim p}^M f(x_i)\tag{6}$$

To this end, Markov chain Monte Carlo (MCMC) methods can be used, such as [Langevin dynamics](https://friedmanroy.github.io/blog/2022/Langevin/), in order to sample from the distribution. Many of these MCMC methods only require access to the gradient of the log of the distribution, $\nabla \log p(x) = \nabla \log \tilde{p}(x)$, so not knowing the normalization constant isn’t a problem. However, this doesn’t give us any estimate of $Z$ and many times it’s also difficult to tune an MCMC sampler.

>  如果我们不关心计算规范化常数 $Z$，仅关心计算某个函数相对于目标分布的期望，可以采用 MCMC 方法 (仅关心利用 $\tilde p(x)$ 获取服从 $p(x)$ 的样本)
>  例如经典的 MCMC 方法: Langevin dynamics，只需要目标分布的得分函数就可以执行，而 $\nabla \log p(x) = \nabla\log \tilde p(x)$，故规范化常数在此时并不成为问题
>  但 MCMC 方法无法用以估计规范化常数，且许多情况下 MCMC sampler 也难以调节

### Combining IS with MCMC
At it’s core, AIS is a way to combine the importance weights in IS with an MCMC approach. The idea is relatively simple: start with a sample from a simple distribution $q(x)$ and use MCMC iterations to get this sample closer to the distribution of interest $p(x)$. At the same time, we can also keep track of the relative importance of the sample, getting better calibrated importance weights.
>  AIS 本质上将 IS 中的重要性权重结合到了 MCMC 方法中，其思想是: 从来自于简单分布 $q(x)$ 的样本开始，使用 MCMC 方法将该样本逐渐向服从目标分布 $p(x)$ 靠近，但是在 MCMC 迭代的同时**追踪该样本的相对重要性 (重要性权重)**
>  这样，我们即借助 MCMC 方法获取了来自于更加校准的分布的样本，也保持了样本的重要性权重

That’s the main intuition behind AIS. Don’t worry if it’s still unclear, you have a bit more to read which I hope will clarify things.

# Annealing the Importance Distribution
As in IS, we begin by choosing:

$$q(x) = \frac{1}{Z_0}\tilde{q}(x)\tag{7}$$

which is easy to sample from and whose normalization constant, $Z_0$, is known.

How are we going to get this sample closer to $p(x)$? We’re going to define a series of intermediate distributions that gradually get closer and closer to $p(x)$. For now I’ll define the $T$ intermediate distributions as:

$$
\begin{aligned} \pi_t(x)&=\tilde{q}(x)^{1-\beta(t)}\cdot\tilde{p}(x)^{\beta(t)}\\ \beta(t)&=t/T \end{aligned}
$$

where $p(x)=\tilde{p}(x)/Z_T$ is the distribution we’re actually interested in. 

Notice that $\beta(0)=0$ and $\beta(T)=1$, so:

$$
\begin{align} \pi_0(x)&=\tilde{q}(x)\\ \pi_T(x)&=\tilde{p}(x) \end{align}
$$

Furthermore, the values of $\beta(t)$ gradually move from 0 to 1, so for each $t$ the function $\pi_t(x)$ is an unnormalized distribution somewhere between the two distributions $\tilde{q}(x)$ and $\tilde{p}(x)$. These intermediate distributions will allow a smooth transition from the simple distribution to the complex.

>  AIS 和 IS 一样，从一个简单的分布 $q(x)$ 开始
>  之后，我们需要定义一系列逐渐接近目标分布 $p(x)$ 的中间分布 $\pi_t(x)$
>  其中，$\beta(t)= \frac {t}{T}\in[0, 1]$，随着 $t$ 增大而增大，故随着 $t$ 增大，$\pi_t(x)$ 中，初始分布 $\tilde q(x)$ 的成分减小，目标分布 $\tilde p(x)$ 的成分将逐渐变多
>  在该定义下，中间分布 $\pi_t(x)$ 始终在是 $\tilde q(x)$ 和 $\tilde p(x)$ 之间的未规范化分布

If we use many iteration $T$, then the difference between each $\pi_t(x)$ and $\pi_{t+1}(x)$ will be very small, such that a sample from $\pi_t(x)$ is almost (but not quite) a valid sample from $\pi_{t+1}(x)$. Accordingly, we can use a relatively lightweight MCMC approach to get a sample from $\pi_{t+1}(x)$ starting from the $\pi_t(x)$ sample. And we can do this for all $t$, starting from the initial simple distribution $\pi_0(x)$.

>  迭代数量/中间分布数量 $T$ 足够多时，$\pi_t(x)$ 和 $\pi_{t+1}(x)$ 之间的差异就会较小，因此二者之间的 MCMC 过程也会更加轻量

At the same time, the importance weights for $\pi_{t+1}(x)$ given the $\pi_t(x)$ “proposal distribution” are $w_t = \frac{\pi_{t+1}(x)}{\pi_t(x)}$. We essentially want to get the importance weights for the whole chain $\pi_0(x) \rightarrow \pi_1(x) \rightarrow \cdots \rightarrow \pi_T(x)$, so we will multiply the time-based importance weights along the way. Ultimately, given a chain of $x_0, \cdots, x_{T-1}$ the importance weight of the whole chain will be given by:

$$w(x_0, \cdots, x_T) = Z_0 \frac{\pi_1(x_0)}{\pi_0(x_0)} \cdot \frac{\pi_2(x_1)}{\pi_1(x_1)} \cdots \frac{\pi_T(x_{T-1})}{\pi_{T-1}(x_{T-1})}\tag{10}$$

Notice that for all the intermediate $t$ 's that are not equal to 0 or $T$, the unnormalized distribution $\pi_t(x)$ always appears in the numerator _and_ denominator once, meaning that we don’t need to estimate the normalizing coefficients $Z_t$ as they cancel out.

>  将 $\pi_{t+1}(x)$ 视作目标分布，将 $\pi_t(x)$ 视作采样分布，来自于采样分布的样本相对于目标分布的重要性权重 $w_t = \frac {\pi_{t+1}(x)}{\pi_t(x)}$
>  那么整个样本轨迹 $x_0, \dots, x_{T-1}$ 的重要性权重的计算就如上所示

Putting all of this together, the AIS algorithm proceeds as follows (see appendix [A.2](https://friedmanroy.github.io/blog/2023/AIS/#a2-ais-importance-weights**) for something a bit more formal):

> 1. sample $x_0 \sim q(x)$
> 2. set $w_0 = Z_0$
> 3. for $t=1, \cdots, T$:
> 4. $\qquad$ set $w_t = w_{t-1} \cdot \frac{\pi_t(x_{t-1})}{\pi_{t-1}(x_{t-1})}$
> 5. $\qquad$ sample $x_t \sim \pi_t(x)$ starting from $x_{t-1}$

That’s it.

### Small Notes
For this post, I chose a particular (“standard”) way to define the intermediate distributions $\pi_t(x)$. However, any set of intermediate distributions can be chosen, as long as the unnormalized form of each of them can be calculated and the change is gradual enough.
>  中间分布可以按照任意形式定义，只要它们自己的规范化常数可以被计算，并且中间分布之间的变化是平缓的

Additionally to that, while $\pi_t(x) = \pi_0^{1-\beta(t)}(x)\pi_T^{\beta(t)}(x)$ is the definition most often used in practice, $\beta(t)$ is usually _not_ just linear in $t$. There are many options for the scheduling/annealing of $\beta(t)$, where different heuristics are taken into account in the definition of the schedule.

### Examples and Visualizations
Implementation details
In all of the following examples, I’m using Langevin dynamics or the Metropolis corrected version ( [called MALA](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)) with a single step as the MCMC algorithm between intermediate distributions. Moreover, I always used $q (x)=\mathcal{N}(x;\ 0, I)$ as the proposal distribution.

To be honest, this would not work in any real application - a single Langevin step _doesn’t_ sample from the distribution (you usually need many more steps). Luckily, for these visualizations a single step _was_ enough and conveys the message equally well, so I’d rather keep the simpler approach for now.

The first example is really simple - the target and proposal distributions are both Gaussian:

![AIS from one Gaussian to another, non-isotropic Gaussian.](https://github.com/friedmanroy/friedmanroy.github.io/blob/master/assets/blog_figs/AIS/two_gaussians.gif?raw=true)

Figure 1: a really simple example of using AIS to anneal between a standard normal to another (non-isotropic) Gaussian. Brighter values indicate regions with higher probability, and the two black dots are the samples across the intermediate distributions. Notice how the intermediate distributions "pull" the two samples after them, finally reaching the target distribution.

An important advantage of AIS is that it anneals between a simple distribution, slowly morphing into the more complicated distribution. If properly calibrated, this allows it to sample from all modes:

![AIS from one Gaussian to another, non-isotropic Gaussian.](https://github.com/friedmanroy/friedmanroy.github.io/blob/master/assets/blog_figs/AIS/1to3_gaussians.gif?raw=true)

Figure 2: AIS from one Gaussian to a mixture of 3 Gaussians. When the proposal distribution is properly set, the annealing process ensures that all modes are properly "covered".

Of course, AIS can be used to sample from much more complex distributions:

![AIS from one Gaussian to another, non-isotropic Gaussian.](https://github.com/friedmanroy/friedmanroy.github.io/blob/master/assets/blog_figs/AIS/spiral.gif?raw=true)

Figure 3: Notice how AIS doesn't "waste" samples in regions with practically 0 density towards the end.

# Importance Weights
The mathematical trick of AIS is the way we defined the weights, $w_T$ (see [A.2](https://friedmanroy.github.io/blog/2023/AIS/#a2-ais-importance-weights) for more details regarding the definition). Like in regular importance sampling, the weights are defined in such a way that:

$$E_{x_0 \sim q}[w_T] = Z_T$$

So, we can use $M$ samples $x_T^{(1)}, \cdots, x_T^{(M)}$ and importance weights $w_T^{(1)}, \cdots, w_T^{(M)}$ created using the AIS algorithm to estimate the expectation from equation (4):

$$\bar{f} \approx \hat{f} = \frac{\sum_i^M w_T^{(i)} f(x_T^{(i)})}{\sum_i^M w_T^{(i)}}$$

In fact, this $\hat{f}$ is an unbiased estimator for $\bar{f}$!

### Calculations in Log-Space
If you’ve ever dealt with probabilistic machine learning, you probably already know that multiplying many (possible very small) probabilities is a recipe for disaster. This is also true here.

Recall:

$$
\begin{equation} w_T=Z_0\cdot\frac{\pi_1(x_0)}{\pi_0(x_0)}\cdot\frac{\pi_2(x_1)}{\pi_1(x_1)}\cdots\frac{\pi_T(x_{T-1})}{\pi_{T-1}(x_{T-1})} \end{equation}
$$

In almost all practical use cases, the values $\pi_i (x)$ are going to be very small numbers. So, $w_T$ is the product of many small numbers. If $T$ is very large, it is almost guaranteed that the precision of our computers won’t be able to handle the small numbers and eventually we’ll end up with $w_T=0/0$.

>  计算重要性权重时，我们是在将多个很小的数值相乘
>  显然，如果 $T$ 足够大，计算机的精度将无法保证这样的计算最终是正确的，我们往往最后会得到 $w_t = 0/0$

Instead, the importance weights are usually calculated in _log-space_, which modifies the update for the importance weight into:

$$
\begin{equation} \log w_t=\log w_{t-1}+\log \pi_t(x_{t-1})-\log\pi_{t-1}(x_{t-1}) \end{equation}
$$

The log-weights can then be averaged to get an estimate of $\log Z_t$ … well, almost.

>  因此我们需要在对数空间进行计算，重要性权重的对数形式的计算如上所示

Averaging out the log-weights gives us $\mathbb{E}_{x_0 \sim q(x)}[\log w_T]$, however by Jensen’s inequality:

$$
\begin{equation} \mathbb{E}_{x_0\sim q}[\log w_T] \le \log \mathbb{E}_{x_0\sim q}[w_T]=\log Z_T \end{equation}
$$

>  但是，虽然 $w_t$ 的期望是 $Z_t$，$\log w_t$ 的期望却不是 $\log Z_T$，故此时不能简单地计算 $\log w_t$ 的平均值
>  根据 Jensen's inequality，$\log w_t$ 的期望实际上和 $\log Z_T$ 之间满足上述不等关系

So, when we use the log of importance weights, it’s important to remember that they only provide us with a _stochastic lower bound_ (The lower bound is stochastic because we only get an estimate of $Z_T$ when the number of samples is finite. This makes things a bit hard sometimes: the variance of the estimator can sometimes push the estimate to be larger than the true value, even though it's a lower bound!) of the normalization constant. When $T$ is very large, it can be shown that the variance of the estimator tends to 0, meaning the lower bound becomes tight.
>  因此我们实际上是在估计 $Z_T$ 的一个下界，该估计的方差有时可能会比真实值还要大
>  当 $T$ 足够大的时候，这个下界将是一个紧的下界，该估计的方差将趋近于 0

Bottom line is: the number of intermediate distributions $T$ should be quite large and carefully calibrated.
>  因此，我们必须确保时间步数量 $T$ 足够大

### Reversing the Annealing
There is a silver lining to the above. If we reverse the AIS procedure, that is start at $\pi_T(x)$ and anneal to $\pi_0(x)$, then we can generate a _stochastic upper bound_ of $Z_T$.

Keeping the same notation as above, let $w_T$ be the importance weights of the regular AIS and $m_0$ be the importance weights of the reverse annealing. Then:

$$
\begin{align} \mathbb{E}_{x_T\sim p}[\log m_0]&\le \log \mathbb{E}_{x_T\sim p}[m_0]=\log\frac{1}{Z_T}\\ \Leftrightarrow \log Z_T&\ge - \mathbb{E}_{x_T\sim p}[\log m_0] \end{align}
$$

The only problem, which you may have noticed, is that the reverse procedure needs to start from samples out of $p(x)$, our target distribution. Fortunately, such samples were produced by the forward procedure of AIS (This method for finding both the stochastic lower and upper bounds is called _bidirectional Monte Carlo_!).

# Finding Your Keys
Back to our somewhat contrived problem.

Here’s your apartment and the PDF for $p_\text{key}(x)$ representing the distribution of probable key placements:

![AIS from one Gaussian to another, non-isotropic Gaussian.](https://github.com/friedmanroy/friedmanroy.github.io/blob/master/assets/blog_figs/AIS/room_distribution.png?raw=true)

Figure 4: The floor plan with the density of finding the keys at each point in space (brighter is higher density). It's impossible to find the keys outside the house or in the walls, so the darkest blue in this image should be treated as essentially 0 density.

Your place is really big!

As you can see, there are rooms more likely and less likely to contain the keys and there are regions where it would be almost impossible to find the keys (all the places with the darkest shade of blue). Such places are, for instance, outside the house, in the walls or in the middle of a hallway.

Conveniently, the rooms are numbered. We want to estimate, given this (unnormalized) PDF the probability that the keys are in a room, say room 7:

$$
\begin{equation} P(\text{keys}\in R_7)=? \end{equation}
$$

Well, let’s use AIS to calculate the importance weights. Here’s the compulsory animation:

![AIS from one Gaussian to another, non-isotropic Gaussian.](https://github.com/friedmanroy/friedmanroy.github.io/blob/master/assets/blog_figs/AIS/keys.gif?raw=true)

Figure 5: Running AIS on the floor plan. The points towards the end really look as if they were sampled from the correct distribution, even though it's such a weird one. Also, note that I ran this algorithm for many more iterations than the previous ones - this helped the sampling procedure, but could probably be done with less iterations.

More implementation details
Unlike the previous animations, for these trajectories I actually used 100 samples and am only showing 30 (otherwise everything would be full of moving black dots). Also, notice that towards the end of the AIS procedure the particles get “stuck”; this is because I used Metropolis-Hastings acceptance steps (If you are unfamiliar with this term, don't sweat it. Basically, I used a method that rejects sampled points that aren't from the distribution I'm trying to sample from.) and most of the sampling steps towards the end were rejected, because of the really small densities at the edges of the rooms.

Also, the annealing for this animation was a bit tricky to set. Because the density outside the house is basically constant (and equal to 0), if the annealing isn’t carefully adjusted points have a tendency of getting stuck there. My solution was to also anneal the impossibility of being in those regions, just in a much slower pace than the other parts of the distribution (If you've ever heard of _log-barriers_ in optimization, then I think it's basically the same concept..)

Using the importance weights accumulated during this sampling procedure, we can now calculate the probability of the keys being in any one of the rooms, for instance room 7:

$$
\begin{align} P(\text{keys}\in R_7)&=\mathbb{E}_x\left[\textbf{1}[x\in R_7]\right]\\ &\approx\frac{\sum_i w_T^{(i)}\cdot \textbf{1}[x\in R_7]}{\sum_i w^{(i)}_T} \end{align}
$$

Using this formula to calculate the probabilities of the keys being in each of the rooms, we get:

![AIS from one Gaussian to another, non-isotropic Gaussian.](https://github.com/friedmanroy/friedmanroy.github.io/blob/master/assets/blog_figs/AIS/key_probabilities.png?raw=true)

Figure 6: The same floor plan, only with the probabilities of the keys being in any of the rooms overlayed on top. Brighter rooms have higher probability.

And there you have it! You should probably check in either room 9 or 6 and only then search in the other rooms.

# **Practical Applications of AIS**
While I believe the example in this post is good for visualization and intuition, it’s pretty silly (as I already mentioned). In 2D, rejection sampling probably achieves the same results with much less fuss.

The more common use for AIS that I’ve seen around is as a method for _Bayesian inference_ (e.g., [Neal, 2001](https://www.cs.toronto.edu/~radford/ftp/ais-long.pdf)).

Suppose we have some prior distribution $p(\theta; \varphi)$ parameterized by $\varphi$ and a likelihood $p(x|\theta)$. Bayesian inference is, at its core, all about calculating the posterior distribution and the evidence function:

$$
\overbrace{p(\theta|x;\varphi)}^{\text{posterior}} = \frac{p(\theta)\cdot p(x|\theta)}{\underbrace{p(x;\varphi)}_{\text{evidence}}}
$$

For most distributions in the real world this is really really hard. As a consequence, using MCMC methods for sampling from the posterior (or _posterior sampling_) is very common. However, such methods don’t allow for calculation of the evidence, which is one of the primary ways models are selected in Bayesian statistics.

AIS offers an elegant solution both to posterior sampling and evidence estimation. Let’s define our proposal and target distributions once more, adjusted for Bayesian inference:

$$
\begin{equation} \pi_0(\theta)=p(\theta;\ \varphi)\qquad\ \ \ \ \ \ \ \ \pi_T(\theta)=p(\theta;\varphi)\cdot p(x\vert\ \theta) \end{equation}
$$

As you have probably already noticed, $\pi_T(\theta)$ is the **unnormalized version** of the posterior. The normalization constant of $\pi_T(\theta)$ is **exactly the evidence**. We only need to choose an annealing schedule between the proposal and target distributions. Taking inspiration from our earlier annealing schedule, we can use (for example):

$$
\begin{equation} \pi_t(\theta)=p(\theta;\varphi)\cdot p(x\vert\theta)^{\beta(t)} \end{equation}
$$

where $\beta(0)=0$ and $\beta(T)=1$.

That’s it. If $T$ is large enough, then we can be sure that the samples procured from the AIS algorithm will be i.i.d. from the posterior. Moreover, the weights $w_T^{(i)}$ can be used to estimate the evidence:

$$
\begin{equation} p(x;\varphi)\approx \frac{1}{M}\sum_i w_T^{(i)} \end{equation}
$$

And there you have it! Instead of simply sampling from the posterior, you can get an estimate for the evidence at the same time (As long as you don't use a batched method on many data points $x$ like they do in Bayesian neural networks, I don't think this will work there (although variants do exist).).

# **Conclusion**
You now (maybe) know what annealed importance sampling is and how to use it. My main hope was to give some intuition into what happens in the background when you use AIS. I find the concept of sampling by starting at a simple distribution and moving to a more complex one really cool, especially when it is treated in such a clear and direct manner.

# Appendix

## A.1 Importance Sampling
We know how to calculate $\tilde{p}(x)$, but don’t know how to sample from it. The simplest solution for calculating $\overline{f}$ and $Z$ is through what is called _importance sampling_.

Start by choosing a simpler distribution $q(x)$ whose normalization is completely known _and_ is easy to sample from. (Also, you have to make sure that the support of $p(x)$ is contained in the support for $q(x)$!). Then:

$$
\begin{align} \mathbb{E}_{x\sim p}[f(x)]&=\intop p(x)f(x)dx\\ &=\intop \frac{p(x)}{q(x)}f(x)q(x)dx\\ &=\mathbb{E}_{x\sim q}\left[\frac{p(x)}{q(x)}\cdot f(x)\right] \end{align}
$$

Using $q(x)$, we somehow magically moved the difficulty of sampling $x$ from $p(x)$ to the much simpler operation of sampling $x$ from $q(x)$! The expectation can now be approximated using a finite number of samples. Let $w(x) = p(x)/q(x)$ and generate $M$ samples from the distribution $q(x)$ such that:

$$E_{x \sim p}[f(x)] \approx \frac{1}{M}\sum_{i:x_i \sim q}^M w(x_i) \cdot f(x_i)\tag{27}$$

>  使用重要性采样时，对于任意函数的 $f(x)$，我们使用来自于 $q(x)$ 的样本，计算加权平均，估计它相对于 $p(x)$ 的期望，其中权重就是重要性权重
>  换句话说，我们是在估计函数 $\frac {p(x)}{q(x)}f(x)$ 相对于 $q(x)$ 的期望，用该期望近似我们的目标期望 ($f(x)$ 相对于 $p(x)$ 的期望)
>  此时我们完全不需要在 $p(x)$ 中采样，在 $q(x)$ 中采样，然后计算重要性权重即可

But there’s a problem: we don’t really know how to calculate $p(x)$ (since we don’t know $Z$), only $\tilde{p}(x)$. Fortunately, we can also estimate $Z$ for the same price! Denote $\tilde{w}(x) = \tilde{p}(x)/q(x)$, then:

$$
\begin{align} Z&=\intop \tilde{p}(x)dx=\intop\frac{\tilde{p}(x)}{q(x)}q(x)dx\\ &=\intop \tilde{w}(x)q(x)dx\\ &=\mathbb{E}_{x\sim q}\left[\tilde{w}(x)\right]\\ &\approx \frac{1}{M}\sum_{i:\ x_i\sim q}^M\tilde{w}(x_i) \end{align}
$$

> 考虑回未规范分布的问题， $\tilde p(x)$ 的规范化常数可以视作恒等函数 $f\equiv 1$ 相对于 $\tilde p(x)$ 的期望，即 $Z = \int \tilde p(x) dx$
> 使用重要性采样，我们估计函数 $\frac {\tilde p(x)}{q(x)}f = \frac {\tilde p(x)}{q(x)}=\tilde w(x)$ 相对于 $q(x)$ 的期望，用它近似 $f$ 相对于 $\tilde p(x)$ 的期望，即近似 $Z$
> 因此 $Z$ 的估计值就是 $\tilde w(x)$ 的平均值

So, our estimate of $\overline{f}$ is given by:

$$E_{x \sim p}[f(x)] \approx \frac{\sum_i \tilde{w}(x_i) \cdot f(x_i)}{\sum_i \tilde{w}(x_i)}\tag{32}$$

The $w(x)$ (and their unnormalized versions) are called _importance weights_ as for each $x_i$ they capture the relative importance between $p(x_i)$ and $q(x_i)$.

>  估计了 $Z$，我们将 $f(x)$ 相对于 $\tilde p(x)$ 的期望 (也由重要性采样估计) 除以 $Z$ 的估计值，就估计了 $f(x)$ 相对于 $p(x)$ 的期望

At the limit $M \rightarrow \infty$, the above approximation becomes accurate. Unfortunately, when $M$ is finite, this estimation is biased and in many cases can be very misspecified.
>  样本数量趋于极限，估计就是准确的

## A.2 AIS Importance Weights
To properly understand the construction of the importance weights in AIS, we are going to need to be more precise than my explanation in the main body of text.

>  我们有 target distribution $p(x) = \pi_T(x)/ Z_T$ 和 proposal distribution $q(x) = \pi_0(x)/Z_0$
>  在 target distribution 和 proposal distribution 之间，有 $T-1$ 个 intermediate unnormalized distributions $\pi_t(x),(1\le t\le T-1)$ 
>  相邻的 intermediate distribution 之间由 transition operator $\mathcal T_t(x\to x')$ 执行转换

What do I mean by “invariant transition operators”? Well, these will be our sampling algorithms, so Langevin dynamics on the $t$ -th distribution, $\pi_t(x)$. The “invariant” part just means that this transition operator maintains _detailed balance_ with respect to the distribution $\pi_t(x)$:

$$
\begin{equation} \mathcal{T}_t(x\rightarrow x')\frac{\pi_t(x)}{Z_t}=\mathcal{T}_t(x'\rightarrow x)\frac{\pi_t(x')}{Z_t} \end{equation}
$$

As long as $\mathcal{T}_t(x \rightarrow x')$ has this property for every possible pair of $x$ and $x'$, it can be used in AIS.

>  transition operator $\mathcal T_t(x \to x')$ 要满足的约束就是需要细致平衡条件，定义如上 (本质就是维持稳态分布)

Now, recall that the sampling procedure in AIS was carried out as follows:

 - sample $x_0 \sim \pi_0$
 - generate $x_1$ using $\mathcal{T}_1(x_0 \rightarrow x_1)$
 - ...
 - generate $x_T$ using $\mathcal{T}_T(x_{T-1} \rightarrow x_T)$

This procedure describes a (non-homogeneous) Markov chain, with transition probabilities determined according to $\mathcal{T}_t$.

>  AIS 的采样过程就是由多个 transition operator $\mathcal T_t$ 定义的非一致性 Markov chain

In the scope of this Markov chain, we can talk about the forward joint probability (starting at $x_0$ and moving to $x_T$) and the reverse joint probability (starting at $x_T$ and going back). At its root, AIS is just importance sampling with the reverse joint as the target and the forward as the proposal. Mathematically, define:

$$
\begin{align} \pi(x_0,\cdots,x_T)&=\pi_T(x_T)\cdot\mathcal{T}_T(x_T\rightarrow x_{T-1})\cdots \mathcal{T}_1(x_1\rightarrow x_0)\\ q(x_0,\cdots,x_T)&=q(x_0)\cdot\mathcal{T}_1(x_0\rightarrow x_1)\cdots \mathcal{T}_T(x_{T-1}\rightarrow x_T) \end{align}
$$

>  我们为这个 Markov chain 定义前向联合分布和逆向联合分布，如上所示
>  AIS 本质可以视作以前向联合分布 $q(x_0, \cdots, x_T) = q(x_0) \mathcal T_1(x_0 \to x_1) \cdots \mathcal T_T(x_{T-1} \to x_T)$ 为 proposal distribution，以逆向联合分布 $\pi(x_0, \cdots, x_T)  = \pi_T(x_T) \cdot \mathcal T_T(x_T\to x_{T-1}) \cdots \mathcal T_1(x_1 \to x_0)$ 为 target distribution 的重要性采样 (采样对象是整个轨迹)

Of course, we never actually observe $\mathcal{T}_t(x_t \rightarrow x_{t-1})$, only the opposite direction. How can we fix this? Well, using detailed balance:

$$\mathcal{T}_t(x_t \rightarrow x_{t-1}) = \frac{\pi_t(x_{t-1})}{\pi_t(x_t)} \cdot \mathcal{T}_t(x_{t-1} \rightarrow x_t)\tag{36}$$

>  我们实际上并未观察到逆向的转移分布，但可以根据细致平衡条件，依赖前向转移分布和两个稳态分布计算出逆向的转移分布

This neat property allows us to write the full form of the importance weights (Getting to the last line requires rearranging the terms and using the equation above for the reverse transition, but this post is already pretty long and I don't think adding that math particularly helps here... Everything cancels out nicely and we get the form for the importance weights as in the main text!):

$$
\begin{align} 
w=&\frac{\pi(x_0,\cdots,x_T)}{q(x_0,\cdots,x_T)}\\ 
&=\frac{\pi_T(x_T)}{q(x_0)}\cdot\frac{\mathcal{T}_T(x_T\rightarrow x_{T-1})\cdots \mathcal{T}_1(x_1\rightarrow x_0)}{\mathcal{T}_1(x_0\rightarrow x_1)\cdots \mathcal{T}_T(x_{T-1}\rightarrow x_T)}\\ 
&=Z_0\cdot \frac{\pi_1(x_0)}{\pi_0(x_0)}\cdot\frac{\pi_2(x_1)}{\pi_1(x_1)}\cdots\frac{\pi_T(x_{T-1})}{\pi_{T-1}(x_{T-1})} 
\end{align}
$$

>  根据 Eq 36，我们就可以展开对重要性权重 $w = \pi(x_0, \cdots, x_T) / q(x_0, \cdots, x_T)$ 的计算，如上所示

These importance weights are exactly the same as those defined in the main body of text, but their motivation is maybe clear now?

The important point is that the proposal distribution creates a path from $x_0$ to $x_T$ while the “true target distribution” is the path from $x_T$ to $x_0$. So the importance weighting is now the forward path $\stackrel{\rightarrow}{\mathcal{T}}(x_0 \rightarrow x_T)$ as a simpler alternative to the reverse path $\stackrel{\leftarrow}{\mathcal{T}}(x_T \rightarrow x_0)$.

To hammer this point home, the normalization constant for $\pi_T(x)$ can be found by taking the expectation with regards to the forward paths:

$$
\begin{equation} Z_T=\mathbb{E}_{\stackrel{\rightarrow}{\mathcal{T}}(x_0\rightarrow x_T)}\left[w\right]=\mathbb{E}_{\stackrel{\rightarrow}{\mathcal{T}}(x_0\rightarrow x_T)}\left[\frac{\pi_T (x_T)\stackrel{\leftarrow}{\mathcal{T}}(x_T\rightarrow x_0)}{q (x_0)\stackrel{\rightarrow}{\mathcal{T}}(x_0\rightarrow x_T)}\right] \end{equation}
$$

> 类似地，对重要性权重根据 proposal distribution (前向轨迹分布) 求期望，就可以得到真正的目标分布 $\pi_T(x)$ 的规范化常数 $Z_T$

>  推导
>  我们知道

$$
\begin{align}
Z_T &= \int \pi_T(x_T) dx_T\\
\end{align}
$$

>  其中 $\pi_T(x_T)$ 可以写为

$$
\begin{align}
\pi_T(x_T) 
&=\int \pi_T(x_T)\mathcal T_T(x_T\to x_{T-1})\cdots \mathcal T_1(x_1 \to x_0)dx_0\cdots x_{T-1}\\ 
&=\int \pi(x_0, \dots, x_T)dx_0\cdots x_{T-1}
\end{align}
$$

>  故

$$
Z_T = \int \pi(x_0, \dots, x_T)dx_0\cdots x_T
$$

>  也就是对整个路径在路径分布 $\pi(x_0, \dots, x_T)$ (该路径分布以 $\pi_T$ 为起点) 上的积分
>  因此为了求解 $Z_T$，AIS 将整个路径分布视作 target，将以 $\pi_0$ 为起点的路径分布作为 proposal，执行重要性采样 (不光是计算 $Z_T$，任何函数相对于 $\pi_T$ 的期望都可以这样计算，和普通的 IS 是一样的)

>  因此，AIS 和 IS 的差异就在于 target 和 proposal 的形式
>  AIS 的 target 和 proposal 被显式定义为了路径分布，如果我们的转移步数足够多，则 target 和 proposal 的差异将比普通的 IS 中 (将 $\pi_T$ 作为 target，将 $\pi_0$ 作为 proposal) 直接比较 $\pi_T$ 和 $\pi_0$ 的差异要小得多 

That was… probably hard to follow. Hopefully I got some of the message across - there is a Markov chain that goes from $q(x)$ to $\pi_T(x)$ and the reverse of it. If you understood that, and are comfortable with importance sampling, then you’re fine. It’ll sink in if you think about it a bit more.

This is a neat mathematical trick, though. Theoretically, it is no different than standard importance sampling, we just defined weird proposal and target distributions. Transforming the a simple distribution to something close to the target, though, that’s the core of it.

If you read this far, well, I commend you. Good luck using AIS!