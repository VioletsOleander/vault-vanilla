# Abstract
A central problem in machine learning involves modeling complex data-sets using highly flexible families of probability distributions in which learning, sampling, inference, and evaluation are still analytically or computationally tractable. Here, we develop an approach that simultaneously achieve both flexibility and tractability. The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data. This approach allows us to rapidly learn, sample from, and evaluate probabilities in deep generative models with thousands of layers or time steps, as well as to compute conditional or posterior probabilities under the learned model. We additionally release an open source reference implementation of the algorithm.
# 1 Introduction
Historically, probabilistic models suffer from a tradeoff between two conflicting objectives: *tractability* and *flexibility*. Models that are *tractable* can be analytically evaluated and easily fit to data(e.g. a Gaussian or Laplace). However, these models are unable to aptly describe structure in rich datasets. On the other hand, models that are *flexible* can be modeled to fit sturcture in arbitarty data. For example, we can define models in terms of any (non-negative) function $\phi(\mathbf x)$ yielding the flexible distribution $p(\mathbf x) = \frac {\phi(\mathbf x)}{Z}$, where $Z$ is a normalization constant. However, computing this normalization constant is generally intractable. Evaluating, training, or drawing samples from such flexible models typically requires a very expensive Monte Carlo process.

A variety of analytic approximations exist which ameliorate, but do not remove, this tradeoff-for instance mean field theory and its expansion (T, 1982; Tanaka, 1998), variational Bayes (Jordan et al., 1999), contrastive divergence (Welling & Hinton, 2002; Hinton, 2002), minimum probability flow (Sohl-Dickstein et al. 2011b;a), minimum KL contraction (Lyu, 2011), proper scoring rules (Gneiting & Raftery, 2007; Parry et al., 2012), score matching (Hyvarinen, 2005), pseudolikelihood (Besag, 1975), Ioopy belief propagation (Murphy et al., 1999), and many, many more. Non-parametric methods (Gershman &  Blei, 2012) can also be very effective $^1$.

---
$^1$ Non-parametric methods can be seen as transitioning smoothly between tractable and flexible models. For instance, a non-parametric Gaussian mixture model will represent a small amount of data using a single Gaussian, but may represent infinite data as a mixture of an infinite number of Gaussians.
## 1.1 Diffusion probabilistic models
We present a novel way to define probabilistic models that allows:
1. extreme flexibility in model structure,
2. exact sampling,
3. easy multiplication with other distributions, e.g. in order to compte a poserior, and
4. the model log likelihood, and the probability of individual states, to be cheaply evaluated.

Our method uses a Markov chain to gradually convert one distribution into another, and idea used in non-equilibrium statistical physics (Jarzynski, 1997) and sequential Monte Carlo (Neal, 2001). We build a generative Markov chain which converts a simple known distribution (e.g. a Gaussian) into a target (data) distribution using a diffusion process. Rather than use this Markov chain to approximately evaluate a model which hash been otherwise defined, we explicitly define the probabilistic model as the endpoint of the Markov chain. Since each step in the diffusion chain has an analytically evaluatable probability, the full chain can also be analytically evaulated.

Learning in this framework involves estimating small pertubations to a diffusion process. Estimating small perturbations is more tractable than explicitly describing the full distributions with a single, non-analytically-normalizable, potential function. Furthermore, since a diffusion process exists for any smooth target distribution, this method can capture data distributions of arbitrary form.

We demonstrate the utility of these *diffusion probabilistic models* by traning high log likelihood models for a two-dimensional swiss roll, binary sequence, handwritten digit (MNIST), and several natural image (CIFAR-10, bark, and dead leaves) datasets.
## 1.2 Relationship to other work
The wake-sleep algorithm (Hinton, 1995; Dayan et al., 1995) introduced the idea of training inference and generative probabilistic models against each other. This approach remained largely unexplored for nearly two decades, though with some excptions (Sminchisescu et al., 2006; Kavukcuoglu et al., 2010). There has been a recent explosion of work developing this idea. In (Kingma & Welling, 2013; Gregor et al., 2013; Rezende et al., 2014; Ozair & Bengio, 2014) variational learning and inference algorithms were developed which allow a flexible generative model and posterior distribution over latent variables to be directly trained against each other.

The variational bound in these papers is similar to the one used in our training objectives and in the eariler work of (Sminchisescu et al., 2006). However, our motivation and model form are both quite different, and the present work retains the following differences and advantages relative to these techniques:
1. We develop our framework using ideas from physics, quasi-static processes, and annealed importance sampling rather than variational Bayesian methods.
2. We thow how to easily multiply the learned distribution with another probability distribution (eg with a conditional distributino in order to compute a posterier)
3. We address the difficulty that training the inference model can prove particularly chanllenging in variational inference methods, due to the asymmetry in the objective between the inference and generative models. We restrict the forward (inference) process to a simple functional form, in such way that the reverse (generative) process will have the same functinoal form.
4. We train models with thousands of layers (or time steps), rather than only a handful of layers.
5. We provide upper and lower bounds on the entropy production in each layer (or time step).

There are a number of related techniques for training probabilistic models (summarized below) that develop highly flexible forms for generative mdoels, train stochastic trajectories, or learn the reversal of a Baysian network. Reweighted wake-sleep (Bornschein & Bengio, 2015) develops extensions and improved learning rules for the original wake-sleep algorithm. Generative stocahstic networks (Bengio & Thiboduau-Laufer, 2013; Yao et al., 2014) train a Markov kernel to match its equilibrium distribution to the data distribution. Neural autoregressive distribution estimators (Larochelle & Murray, 2011) (and their recurrent (Uria et al., 2013 a) and deep (Uria et al., 2013b) extensions) decompose a joint distribution into a sequence of tractable conditional distributions over each dimension. Adversarial networks (Goodfellow et al., 2014) train a generative model against a classifier which attempts to distinguish generated samples from true data. A similar objective in (Schmidhuber, 1992) learns a two-way mapping to a representation with marginally independent units. In (Rippel & Adams, 2013; Dinh et al., 2014) bijective deterministic maps are learned to a latent representation with a simple factorial density function. In (Stuhlmuller et al., 2013) stochastic inverses are leaned for Bayesian networks. Mixtures of conditional Gaussian scale mixtures (MCGSMs) (Theis et al., 2012) describe a dataset using Gaussian scale mixtures, with parameters which depend on a sequence of causal neighbors. There is additionally significant work leaning flexible generative mappings from simple latent distributions to data distributions - early examples including (MacKay, 1995) where neural networks are introduced as generative models, and (Bishop et al., 1998) where a stochastic manifold mapping is learned from a latent space to the data space. We will compare expreimentally against adversarial networks and MCGSMs.

Related ideas from physics include the Jarzynski equality (Jarzynsik, 1997), known in machine learning as annealed Importance Sampling (AIS) (Neal, 2001), which uses a Markov chain which slowly converts one disctirubtioninto another to compute a ratio of normalizing constants. In (Burda et al., 2014) it is shown that AIS can also be performed using the reverse rather than forward trajectory. Langevin dynamics (Langevin, 1908), which are the stochastic realization of the Fokker-Planck equation, show how to define a Gaussian diffusion process which has any target distributino as its equilibrium. In (Suykens & Vandewalle, 1995) the Fokker-Planck equation is used to perform stochastic optimization. Finally, the Kolmogorov forward and bacward equation (Feller, 1949) show that for many forward diffusion processes, the reverse diffusion processes can be described using the same functional form.
# 2 Algorithm
Our goal is to define a forward (or inference) diffusion process which converts any complex data distribution into a simple, tractable, distribution, and then learn a finite-time reversal of this diffusion process which defines our generative model distribution (See Figure 1). We first describe the forward, inference diffusion process. We then show how the reverse, generative diffusion process can be trained and used to evaluate probabilities. We also derive entropy bounds for the reverse process, and show how the learned distributions can be multiplied by any second distribution (e.g. as would be done to compute a posterior when inpainting or denoising an image).
## 2.1 Forward Trajectory
We label the data distribution $q(\mathbf x^{(0)})$. The data distribution is gradually converted into a well behaved (analytically tractable) distribution $\pi (\mathbf y)$ by repeated application of a Markov diffusion kernel $T_{\pi}(\mathbf y\mid \mathbf y';\beta)$ for $\pi (\mathbf y)$, where $\beta$ is the diffusion rate,

$$
\begin{align}
\pi(\mathbf y)&= \int d\mathbf y' T_{\pi}(\mathbf y \mid \mathbf y';\beta)\pi(\mathbf y')\tag{1}\\
q(\mathbf x^{(t)}\mid \mathbf x^{(t-1)}) &=T_{\pi}(\mathbf x^{(t)}|\mathbf x^{(t-1)};\beta_t).\tag{2}
\end{align}
$$

The forward trajectory, corresponding to starting at the data distribution and performing $T$ steps of diffusion, is thus

$$
q(\mathbf x^{(0\dots T)}) = q(\mathbf x^{(0)})\prod_{t=1}^Tq(\mathbf x^{(t)}\mid \mathbf x^{(t-1)})\tag{3}
$$

For the experiments shown below, $q(\mathbf x^{(t)}|\mathbf x^{(t-1)})$ corresponds to either Gaussian diffusion into a Gaussian distribution with identity-covariance, or binomial diffusion into an independent binomial distribution. Table App. 1 gives the diffusion kernels for both Gaussian and binomial distributions.
## 2.2 Reverse Trajectory
The generative distribution will be trained to describe the same trajectory, but in reverse,

$$
\begin{align}
p(\mathbf x^{(T)}) &= \pi(\mathbf x^{{(T)}})\tag{4}\\
p(\mathbf x^{(0\dots T)}) &= p(\mathbf x^{(T)})\prod_{t=1}^{T}p(\mathbf x^{(t-1)}\mid \mathbf x^{(t)})\tag{5}
\end{align}
$$

For both Gaussian and binomial diffusion, for continuous diffusion (limit of small step size $\beta$ ) the reversal of the diffusion process has the identical functional form as the forward process (Feller, 1949). Since $q(\mathbf x^{(t)}\mid \mathbf x^{(t-1)})$ is a Gaussian (binomial) distribution, and if $\beta_t$ is small, then $q (\mathbf x^{(t-1)}|\mathbf x^{(t)})$ will also be a Gaussian (binomial) distribution. The longer the trajectory the smaller the diffusion rate $\beta$ can be made.

During leaning only the mean and covariance for a Gaussian diffusion kernel, or the bit filp probability for a binomial kernel, need be estimated. As shown in Table App.1, $\mathbf f_{\mu}(\mathbf x^{(t)}, t)$ and $\mathbf f_{\Sigma}(\mathbf x^{(t)}, t)$ are functions defining the mean and covariance of the reverse Markov transitions for a Gaussian, and $\mathbf f_b (\mathbf x^{(t)}, t)$ is a function providing the bit flip probability for a binomial distribution. The computational cost of running this algorithm is the cost of these functions, times the number of time-steps. For all results in this paper, multi-layer perceptrons are used to define these functions. A wide range of regression or function fitting techniques would be applicable however, including nonparameteric methods.
# 2.3 Model Probability
The probability the generative model assigns to the data is

$$
p(\mathbf x^{(0)}) = \int d\mathbf x^{(1\dots T)}p(\mathbf x^{(0\dots T)}).\tag{6}
$$

Naively this integral is intractable - but taking a cue from annealed imporatance sampling and the Jarzynski equality, we instead evaluate the relative probability of the forward and reverse trajectories, averaged over forward trajectories,

$$
\begin{align}
p(\mathbf x^{(0)}) &=\int d\mathbf x^{(1\dots T)} p(\mathbf x^{(0\dots T)})\frac {q(\mathbf x^{(1\dots T)}\mid \mathbf x^{(0)})}{q(\mathbf x^{(1\dots T)}\mid\mathbf x^{(0)})} \tag{7}\\
&=\int d\mathbf x^{(1\dots T)} q(\mathbf x^{(1\dots T)}\mid \mathbf x^{(0)}) \frac {p(\mathbf x^{(0\dots T)})}{q(\mathbf x^{(1\dots T)}\mid\mathbf x^{(0)})}\tag{8}\\
&=\int d\mathbf x^{(1\dots T)} q(\mathbf x^{(1\dots T)}\mid \mathbf x^{(0)}) p(\mathbf x^{(T)})\prod_{t=1}^{T}\frac {p(\mathbf x^{(t-1)}\mid\mathbf x^{(t)})}{q(\mathbf x^{(t)}\mid \mathbf x^{(t-1)})}.\tag{9}
\end{align}
$$

This can be evaluated rapidly by averageing over samples from the forward trajectory $q (\mathbf x^{(1\dots T)}\mid \mathbf x^{(0)})$. For infinitesimal $\beta$ the forward and reverse distribution over trajectories can be made identical (see Section 2.2). If they are identical then only a *single* sample from $q (\mathbf x^{(1\dots T)}\mid \mathbf x^{(0)})$ is required to exactly evaluate the above integral, as can be seen by substitution. This corresponds to the case of a quasi-static process in statistical physics (Spinney & Ford, 2013; Jarzynski, 2011).
## 2.4 Training

