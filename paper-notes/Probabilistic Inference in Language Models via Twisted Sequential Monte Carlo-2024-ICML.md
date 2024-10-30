# Abstract  

Numerous capability and safety techniques of Large Language Models (LLMs), including RLHF, automated red-teaming, prompt engineer- ing, and infilling, can be cast as sampling from an unnormalized target distribution defined by a given reward or potential function over the full se- quence. In this work, we leverage the rich toolkit of Sequential Monte Carlo (SMC) for these prob- abilistic inference problems. In particular, we use learned  twist functions  to estimate the expected fu- ture value of the potential at each timestep, which enables us to focus inference-time computation on promising partial sequences. We propose a novel contrastive method for learning the twist functions, and establish connections with the rich literature of soft reinforcement learning. As a complementary application of our twisted SMC framework, we present methods for evaluating the accuracy of language model inference techniques using novel  bidirectional  SMC bounds on the log partition function. These bounds can be used to estimate the KL divergence between the infer- ence and target distributions in both directions. We apply our inference evaluation techniques to show that twisted SMC is effective for sampling undesirable outputs from a pretrained model (a useful component of harmlessness training and automated red-teaming), generating reviews with varied sentiment, and performing infilling tasks.  

# 1. Introduction  

A wide range of language model learning and inference tasks can be viewed as steering a model’s generations to satisfy a specified property. In particular, traditional rein- forcement learning from human feedback (RLHF) pipelines ( Ziegler et al. ,  2019 ;  Stiennon et al. ,  2020 ;  Ouyang et al. , 2022 ;  Bai et al. ,  2022 ;  Rafailov et al. ,  2023 ) may be viewed as targeting an unnormalized target modulated by a terminal reward function which reflects human feedback ( Korbak et al. ,  2022b ). Red-teaming techniques such as prompt- engineering and infilling may seek target outputs with low reward or (high probability of) undesirable responses ( Zou et al. ,  2023 ;  Perez et al. ,  2022 ). In reasoning tasks, we may seek to target outputs which are likely to be deemed valid by a ‘verifier’ ( Cobbe et al. ,  2021 ;  Anil et al. ,  2021 ;  Dohan et al. ,  2022 ;  Hu et al. ,  2023 ). Specific properties of the generated responses might also be enforced ( Khalifa et al. , 2020 ;  Yang & Klein ,  2021 ;  Lew et al. ,  2023 ).  

We view the above tasks as instances of  probabilistic in- ference : sampling from a target unnormalized density and estimating its intractable (log) normalization constant. Con- sider a pre ed base model $p_{0}\big(\mathbf{s}_{1:T}|\mathbf{s}_{0}\big)$  which generates responses $\mathbf{s}_{1:T}$  of maximum length  T  based on a variable- length prompt $\mathbf{s}_{0}$ . We consider defining the target distri- bution of interest using the base model modulated by a potential function $\phi(\mathbf{s}_{1:T})$  which evaluates full sequences,  

$$
\sigma(\mathbf{s}_{1:T}|\mathbf{s}_{0}):=\frac{1}{\mathcal{Z}_{\sigma}(\mathbf{s}_{0})}p_{0}(\mathbf{s}_{1:T}|\mathbf{s}_{0})\phi(\mathbf{s}_{1:T}),
$$  

$$
\mathcal{Z}_{\sigma}(\mathbf{s}_{0}):=\sum_{\mathbf{s}_{1:T}}\tilde{\sigma}\bigl(\mathbf{s}_{1:T}|\mathbf{s}_{0}\bigr)=\sum_{\mathbf{s}_{1:T}}p_{0}\bigl(\mathbf{s}_{1:T}|\mathbf{s}_{0}\bigr)\phi(\mathbf{s}_{1:T}),
$$  

where   $\tilde{\sigma}(\mathbf{s}_{1:T}|\mathbf{s}_{0})$  denotes the unnormalized density. We refer to  Z $\mathcal{Z}_{\sigma}(\mathbf{s}_{0})$  as the normalization constant or partition function, which is intractable due to the summation over $\mathbf{s}_{1:T}$ . We drop dependence on $\mathbf{S}_{0}$  to avoid clutter, but note that each prompt induces a different partition function. In the context of the aforementioned applications,   $\phi(\mathbf{s}_{1:T})$  may be derived from a human preference model (for RLHF), an indication of bad behavior (for automated red-teaming), or a verifier’s prediction of correctness (for reasoning tasks). We refer to  Table 5  or  Korbak et al.  ( 2022b );  Dohan et al.  ( 2022 ); Phan et al.  ( 2023 );  Hu et al.  ( 2023 ) for further examples and discussion of probabilistic inference in language models.  

Twisted Sequential Monte Carlo in Language Models In this work, we leverage tools from (twisted) Sequen- tial Monte Carlo (SMC) ( Doucet et al. ,  2001 ;  Del Moral et al. ,  2006 ;  Briers et al. ,  2010 ;  Chopin et al. ,  2020 ) to perform and evaluate inference in the language mod- eling setting ( Sec. 3 ). A particular challenge in sam- pling from  Eq. (1)  is that the target distribution   $\sigma(\mathbf{s}_{1:T})$ is non-causal. In order to sample tokens sequentially, one needs to infer the marginal distribution   $\begin{array}{r l}{\sigma(\mathbf{s}_{1:t})}&{{}=}\end{array}$ $\begin{array}{r}{\sum_{\mathbf{s}_{t+1:T}}\sigma\big(\mathbf{s}_{1:T}\big)\propto\sum_{\mathbf{s}_{t+1:T}}p_{0}\big(\mathbf{s}_{t+1:T}\big|\mathbf{s}_{1:t}\big)\phi\big(\mathbf{s}_{1:T}\big)}\end{array}$ P , which involves an intractable marginalization. To address this problem, we propose to learn  twist functions   $\psi_{t}(\mathbf{s}_{1:t})$  which modulate the base model such that $p_{0}(\mathbf{s}_{1:t})\psi_{t}(\mathbf{s}_{1:t})$  matches the target marginals   $\sigma(\mathbf{s}_{1:t})$ , up to normalization. The twist functions can be used to focus each step of language model generation on promising partial sequences.  

Evaluating Inference in Language Modeling Sampling from the target distribution is closely intertwined with bounding the log partition function. Similarly to variational inference or traditional RLHF objectives ( Korbak et al. , 2022b ), SMC algorithms yield lower bounds on $\log\mathcal{Z}_{\sigma}$ , where tighter bounds typically coincide with more accu- rate target sampling. However,  upper  bounds may often be obtained when an exact target sample is available ( Grosse et al. ,  2015 ;  2016 ;  Brekelmans et al. ,  2022 ). The difference between upper and lower bounds on $\log\mathcal{Z}_{\sigma}$  in fact yields an upper bound on the symmetrized KL divergence between inference samples and the target distribution ( Grosse et al. , 2016 ). For these reasons, we argue in  Sec. 5  that log par- tition function estimates are a powerful tool for evaluating language model inference techniques.  

Contributions Our probabilistic inference perspective leads to the following contributions:  

•  Twisted Sequential Monte Carlo for Language Model- ing : We view  twisted  SMC as a general framework for sampling and evaluation of language models. While twisted SMC is well-known and  Lew et al.  ( 2023 ) consider SMC with fixed, few-step-ahead target infor- mation in the language modeling setting, we propose to  learn  intermediate twist functions for target distribu- tions defined by terminal potential only. •  Contrastive Twist Learning : We develop probabilis- tic methods for learning intermediate twist functions, presenting a novel  contrastive twist learning  (CTL) method inspired by energy-based modeling and den- sity ratio estimation in  Sec. 4.1 . Further, we adapt existing twisted SMC methods ( Lawson et al. ,  2018 ; 2022 ;  Lioutas et al. ,  2022 ) to the language modeling setting, and highlight connections with inference tech- niques inspired by (soft) reinforcement learning (RL). •  Evaluating Inference in Language Models : Finally, we demonstrate that twisted SMC provides a rich set of tools for evaluating language model fine-tuning or controlled generation techniques. We propose a novel SMC upper bound on $\log\mathcal{Z}_{\sigma}$  which is applica- ble when an exact target sample is available and may be of independent interest. We leverage these bounds to evaluate the quality of inference by measuring the KL divergence to the target   $\sigma(\mathbf{s}_{1:T})$  in  both  directions,  

which can be used to diagnose mode-dropping behav- ior of methods such as proximal policy optimization (PPO) ( Schulman et al. ,  2017 ) which optimize a mode- seeking divergence.  

We proceed to describe background on importance sampling and SMC in  Sec. 2 , before presenting our framework for twisted SMC in the language modeling setting in  Sec. 3 . We propose methods to learn the twist functions in  Sec. 4  and methods to evaluate inference in  Sec. 5 . Our experimental results in  Sec. 7  showcase the ability of twisted SMC to im- prove controlled generation and lend insights into inference quality in existing methods.  

# 2. Background  

Suppose we are given access to an unnormalized density $\tilde{\sigma}(\mathbf{s}_{1:T})$  which can be efficiently evaluated. We focus on estimation of the partition function or normalization con- stant   $\begin{array}{r}{\mathcal{Z}_{\sigma}:=\sum_{\mathbf{s}_{1:T}}\tilde{\sigma}\big(\mathbf{s}_{1:T}\big)}\end{array}$ , since unbiased estimators with low variance yield approximate sampling techniques which closely approximate the target distribution ( Finke ,  2015 ; Maddison et al. ,  2017 ). We review simple importance sam- pling (SIS) and SMC techniques in this section.  

# 2.1. Simple Importance Sampling  

Simple importance sampling (SIS) provides an unbiased estimator of   $\mathcal{Z}_{\sigma}$  by calculating im ce weights for any normalized proposal distribution $\bar{q}(\mathbf{s}_{1:T})$ ,  

$$
w(\mathbf{s}_{1:T}^{i}):=\frac{\tilde{\sigma}\left(\mathbf{s}_{1:T}^{i}\right)}{q(\mathbf{s}_{1:T}^{i})}\,,
$$  

which is unbiased since   $\mathcal{Z}_{\sigma}=\mathbb{E}_{q(\mathbf{s}_{1:T})}[w(\mathbf{s}_{1:T})]$ . The im- portance weights also yield an an unbiased  K -sample esti- mator of the partition function,  

$$
\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SIS}}:=\frac{1}{K}\,\sum_{i=1}^{K}w\big(\mathbf{s}_{1:T}^{i}\big)\,,\qquad\mathbf{s}_{1:T}^{i}\sim q\big(\mathbf{s}_{1:T}\big)\,.
$$  

By normalizing the weights in  Eq. (2)  over   $K$  samples from $q(\mathbf{\dot{s}}_{1:T})$ , we can obtain (biased) estimators of expectations under   $\dot{\boldsymbol{\sigma}}(\mathbf{s}_{1:T})$ ,  

$$
\mathbb{E}_{\sigma(\mathbf{s}_{1:T})}\left[f\bigl(\mathbf{s}_{1:T}\bigr)\right]\approx\sum_{k=1}^{K}\frac{w\bigl(\mathbf{s}_{1:T}^{k}\bigr)}{\sum_{j=1}^{K}w\bigl(\mathbf{s}_{1:T}^{j}\bigr)}f\bigl(\mathbf{s}_{1:T}^{k}\bigr)
$$  

or select an approximate target sample $\mathbf{s}_{1:T}^{\sigma}$   from a categori- cal distribution with the self-normalized importance weights  

$$
\mathbf{s}_{1:T}^{\sigma}\gets\mathbf{s}_{1:T}^{\omega},\qquad\quad\omega\sim\mathsf{c a t}\left(\left\{\frac{w\left(\mathbf{s}_{1:T}^{i}\right)}{\sum_{j=1}^{K}w\left(\mathbf{s}_{1:T}^{j}\right)}\right\}_{i=1}^{K}\right).
$$  

The quality of the approximations in  Eq. (3) -( 5 ) depends crucially on how well the proposal $q(\mathbf{s}_{1:T})$  (which may be learned,  Sec. 3.2 ) matches the target   $\sigma(\mathbf{s}_{1:T})$ . While we discuss evaluation methods in  Sec. 5 , note that if inference is exact (i.e.,   $q(\mathbf{s}_{1:T})=\sigma(\mathbf{s}_{1:T}))$ , then the variance of the importance weights is zero, as $w(\mathbf{s}_{1:T})=\mathcal{Z}_{\sigma}$  for all $\mathbf{s}_{1:T}$  .  

![](images/c24709c3d2167f3fe288f1b4731f93bd102925fd33f0df0612cc44e286ce3357.jpg)  

![](images/420a0a0e769ae28930d394837225a011212dd88cbf3696e3f0015efbdb864c29.jpg)  
(a) Simple Importance Sampling  

Figure 2: Illustrative example of SIS and (Twisted) SMC for sampling book reviews conditioned on positive sentiment   $\phi(\mathbf{s}_{1:T})$ . SIS only performs resampling after observing the entire sequence, while SMC can kill or clone partial sequences $\mathbf{s}_{1:t}$  based on incremental impor- tance weights induced by twist functions $\psi_{t}(\mathbf{s}_{1:t})$ . Green/red indicate high/low importance weights at each incremental step of SMC, or at the final step of SIS. For SMC with the base model proposal $p_{0}$  and the optimal twists, the incremental weights ${\psi}_{t}^{*}/\psi_{t-1}^{*}$   ( Alg. 1  or  Eq. (6) ) − are directly correlated with sentiment.  

dex random variables from the sampling procedure   $s\sim$ $q_{\mathrm{SMC}}(\pmb{S})$  in  Alg. 1 . Assuming resampling at every step,  

# 2.2. Sequential Monte Carlo  

SMC improves inference by decomposing it into easier subproblems involving a set of unnormalized intermediate target distributions   $\{\tilde{\pi}_{t}(\mathbf{s}_{1:t})\}_{t=1}^{T}$ } . A key observation is that as long as   $\pi_{T}(\mathbf{s}_{1:T})~=~\sigma(\mathbf{s}_{1:T})$ , we obtain an unbiased estimate of the partition function   $\mathcal{Z}_{T}=\mathcal{Z}_{\sigma}$ , regardless of the intermediate $\pi_{t}$  and proposal   $q$ .  

$$
\mathcal{Z}_{\sigma}=\mathbb{E}\bigg[\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}\bigg]=\mathbb{E}_{q_{\mathrm{SMC}}(S)}\Bigg[\prod_{t=1}^{T}\frac{1}{K}\sum_{k=1}^{K}w_{t}\big(\mathbf{s}_{1:t}^{k}\big)\Bigg].
$$  

To see that $\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}$ Z is unbiased, we can view  Eq. (8) as performing simple importance sampling   $\begin{array}{r l}{\mathcal{Z}_{\sigma}}&{{}=}\end{array}$ $\mathbb{E}_{q_{\mathrm{SMC}}(S)}\left[\frac{\tilde{\sigma}_{\mathrm{SMC}}(S)}{q_{\mathrm{SMC}}(S)}\right]$ h i in the extended state space, for appropri- ate definitions of $\sigma_{\mathrm{SMC}}(S)$  and $q_{\mathrm{SMC}}(\pmb{S})$  detailed in  App. F or ( Andrieu et al. ,  2010 ;  Maddison et al. ,  2017 ). Intu- itively, we may view the average incremental importance weights at each step as estimating the partition function ratio $\begin{array}{r}{\bar{\mathcal{Z}_{t}}\bar{/\mathcal{Z}}_{t-1}\approx\frac{1}{K}\sum_{k=1}^{K}w_{t}(\mathbf{s}_{1:t}^{k})}\end{array}$ .  Eq. (8)  composes interme- diate partition function ratio estimators to obtain an estimate $\begin{array}{r}{\mathcal{Z}_{T}=\mathcal{Z}_{\sigma}=\prod_{t=1}^{T}\mathcal{Z}_{t}/\mathcal{Z}_{t-1}}\end{array}$ , with   $\mathcal{Z}_{0}=1$ .  

We begin by defining the  incremental  importance weights  

$$
w_{t}(\mathbf{s}_{1:t}):=\frac{\tilde{\pi}_{t}(\mathbf{s}_{1:t})}{\tilde{\pi}_{t-1}(\mathbf{s}_{1:t-1})q(s_{t}\vert\mathbf{s}_{1:t-1})}\,.
$$  

where   $\tilde{\pi}_{t}$  is the unnormalized density of   $\pi_{t}=\tilde{\pi}_{t}/\mathcal{Z}_{t}$ Z .  

SMC maintains a set of $K$  partial sequences, by first sam- pling from the proposal   $q(s_{t}^{k}|\mathbf{s}_{1:t-1}^{k})$   |  in each index $k$ . Op- − tional resampling steps may be performed to clone se- quences with high incremental importance weights using  

With no resampling, SMC reduces to SIS with target $\sigma(\mathbf{s}_{1:T})=\pi_{T}(\mathbf{s}_{1:T})$  and proposal   $q(\mathbf{s}_{1:T})$ . Using the final- step SMC weights, we may estimate expectations or draw approximate samples $\mathbf{s}_{1:T}^{\sigma}$   as in  Eq. (4) -( 5 ).  

$$
\mathbf{s}_{1:t}^{k}\gets\mathbf{s}_{1:t}^{\omega_{t}^{k}},\quad\quad\omega_{t}^{k}\sim\mathsf{c a t}\left(\left\{\frac{w_{t}(\mathbf{s}_{1:t}^{i})}{\sum_{j=1}^{K}w_{t}(\mathbf{s}_{1:t}^{j})}\right\}_{i=1}^{K}\right),
$$  

similarly to  Eq. (5) . Since resampling is performed  with replacement, sequences with high weights may be cloned multiple times. The resulting $\mathbf{s}_{1:t}^{\omega_{t}^{k}}$   are used as prefixes for the next step of proposal sampling in index $k$  (see  Alg. 1 ).  

We can show that SMC yields an unbiased estimator $\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}$ Z of the normalization constant $\mathcal{Z}_{\sigma}$ , by considering the ex- tended state space   ${\cal S}\;:=\;\{s_{t}^{k},\omega_{t}^{k}\}_{k,t=1}^{\bar{K},T}$   of token and in-  

Fig. 2  illustrates the key advantage of SMC resampling over SIS. While a suboptimal   $q(\mathbf{s}_{1:T})$  may produce sequences with low probability under the target   $\sigma(\mathbf{s}_{1:T})$ , SMC resam- pling with well-chosen intermediate targets   $\pi_{t}$  clones the most promising partial sequences $\mathbf{s}_{1:t}$  at step $t$ . Since later sampling proceeds from these prefixes, we expect to obtain final sequences which better cover the high-probability re- gions of the target distribution. We discuss techniques to evaluate the quality of SMC or SIS sampling in  Sec. 5 .  

# 3. Twisted Sequential Monte Carlo for Language Modeling  

A key design choice in the SMC procedure above is the in- termediate targets   $\{\pi_{t}\}_{t=1}^{T-1}$ , where we assume   $\pi_{T}(\mathbf{s}_{1:T})=$ $\sigma(\mathbf{s}_{1:T})$  is always the target distribution. In state-space models with observation likelihoods or environments with intermediate rewards,  filtering  SMC considers target infor- mation collected from times $\tau\leq t$  to define $\pi_{t}$ . ( Chopin et al. ,  2020 ). Previous work on SMC for language models ( Lew et al. ,  2023 ) has considered per-token or few-step- ahead statistics to define tractable intermediate $\pi_{t}$ . However, we are often interested in target distributions which are de- termined by a terminal  potential   $\phi(\mathbf{s}_{1:T})$  only, as in  Eq. (1)  

In such settings,  twisted  SMC methods ( Briers et al. ,  2010 ; Whiteley & Lee ,  2014 ;  Lawson et al. ,  2022 ) consider the full  target information (until time   $T$ ) to define   $\{\pi_{t}\}_{t=1}^{T-1}$ . In other words, our desired intermediate targets are the true marginals $\sigma(\mathbf{s}_{1:t})$  of the target distribution. Intuitively, note that in order to exactly sample   $\mathbf{s}_{1:T}\,\sim\,\sigma(\mathbf{s}_{1:T})$ , we need to ensure partial sequences are distributed according to the intermediate marginals   $\mathbf{s}_{1:t}~\sim~\sigma(\mathbf{s}_{1:t})$ ec. 3.1 , we will represent the intermediate targets  { $\{\pi_{t}\}_{t=1}^{T-1}$ }   using  twist functions $\psi_{t}\colon\mathbf{s}_{1:t}\to\mathbb{R}$  which modulate the base model to (approximately) match the target marginals, thereby sum- marizing future information relevant to sampling at time $t$ .  

# 3.1. Twist Functions  

We represent the intermediate target distributions   $\{\pi_{t}\}_{t=1}^{T-1}$ for SMC sampling using the following general form.  

Definition 3.1  (  Twisted (Intermediate) Targets  ) .  Using approximate twist functions   $\{\psi_{t}\}_{t=1}^{T-1}$   and the final target $\phi_{i}$ , we define the twisted intermediate target distributions  

$$
\begin{array}{r}{\pi_{t}(\mathbf{s}_{1:t})=\left\{\frac{1}{\mathcal{Z}_{t}^{\psi}}\;p_{0}(\mathbf{s}_{1:t})\;\psi_{t}(\mathbf{s}_{1:t})\qquad t\neq T\right.}\\ {\left.\frac{1}{\mathcal{Z}_{\sigma}}\;p_{0}(\mathbf{s}_{1:T})\;\phi(\mathbf{s}_{1:T})\qquad t=T\right.}\end{array}
$$  

For an arbitrary proposal $q$  and the unnormalized targets in Eq. (9) , the incremental importance weights are given by  

$$
w_{t}(\mathbf{s}_{1:t})=\frac{p_{0}(s_{t}|\mathbf{s}_{1:t-1})}{q(s_{t}|\mathbf{s}_{1:t-1})}\frac{\psi_{t}(\mathbf{s}_{1:t})}{\psi_{t-1}(\mathbf{s}_{1:t-1})}.
$$  

While uninformed twist functions $\psi_{t}$  may result in $\pi_{t}(\mathbf{s}_{1:t})$  

which are no closer to the target marginal $\sigma(\mathbf{s}_{1:t})$  than the base model $p_{0}\big(\mathbf{s}_{1:t}\big)$  (for example, in early stages of learn- ing), the crucial fact is that our final target distribution in Eq. (9)  reflects the target  potential $\phi(\mathbf{s}_{1:T})$ . As in  Sec. 2.2 , this ensures that, regardless of the intermediate twists, our resulting importance sampling estimators will be unbiased.  

Finally, the optimal twists   $\psi_{t}^{*}(\mathbf{s}_{1:t})$  recover the intermediate marginals $\pi_{t}^{*}(\mathbf{s}_{1:t})=\sigma(\mathbf{s}_{1:t})$  of the target distribution. We state the sense in which   $\pi_{t}^{*}$   and   $\psi_{t}^{*}$   are optimal in  App. A.1 , and prove the following proposition in  App. B Prop. B.1 .  

Proposition 3.2  ( Optimal Twists ) .  For a given target dis- tribution   $\sigma(\mathbf{s}_{1:T})$  in  Eq.  (1) , the optimal twist functions $\psi_{t}^{*}(\mathbf{s}_{1:t})$  (in regions where   $p_{0}(\mathbf{s}_{1:t})>0,$ ) correspond to  

$$
\pi_{t}^{*}(\mathbf{s}_{1:t})=\sigma(\mathbf{s}_{1:t})=\frac{1}{\mathcal{Z}_{t}^{\psi^{*}}}\;p_{0}(\mathbf{s}_{1:t})\;\psi_{t}^{*}(\mathbf{s}_{1:t}).
$$  

Up to a constant independent of $\mathbf{s}_{1:t}$ , the optimal twists are  

$$
\psi_{t}^{*}(\mathbf{s}_{1:t})\propto\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi(\mathbf{s}_{1:T}).
$$  

and satisfy the recursion  

$$
\psi_{t}^{*}(\mathbf{s}_{1:t})\propto\sum_{s_{t+1}}p_{0}(s_{t+1}\,|\,\mathbf{s}_{1:t})\psi_{t}^{*}(\mathbf{s}_{1:t+1}).
$$  

Since the optimal twist functions are unavailable due to the need to marginalize over future timesteps, we consider learn- ing approximate twist functions using methods in  Sec. 4 .  

# 3.2. Proposal Distribution  

For a given set of targets $\{\pi_{t}\}_{t=1}^{T}$ , the importance weights in  Eq. (10)  depend crucially on the choice of proposal.  

Base Model as Proposal The most straightforward choice of proposal is the base pre-trained model, $q=p_{0}$ . While we demonstrate in  Sec. 7  that SMC resampling with learned twists and the base model proposal can closely approximate the target distribution, this may require large   $K$ . We can achieve greater efficiency using better choices of proposal.  

Twist-Induced Proposal For given targets   $\{\pi_{t}\}_{t=1}^{T}$ , the optimal proposal minimizes the variance of the importance weights ( App. A.1 ). In the language model setting with a ter- minal  potential  only, we will in fact be able to sample from the optimal proposal for the one-step importance weights.  

Proposition 3.3. (Twist-Induced Proposal).  For a given set of intermediate twisted targets $\pi_{t}(\mathbf{s}_{1:t})$  in  Eq.  (9) , the proposal which minimizes the variance of the one-step in- cremental importance weights   $w_{t}$  is given by  

$$
\begin{array}{r l}&{q_{t}^{\pi}\big(\boldsymbol{s}_{t}|\mathbf{s}_{1:t-1}\big)\propto\frac{\pi_{t}\big(\mathbf{s}_{1:t}\big)}{\pi_{t-1}\big(\mathbf{s}_{1:t-1}\big)}}\\ &{\qquad\qquad\qquad=\frac{1}{Z_{t}^{\pi}\big(\mathbf{s}_{1:t-1}\big)}p_{0}\big(\boldsymbol{s}_{t}|\mathbf{s}_{1:t-1}\big)\psi_{t}\big(\mathbf{s}_{1:t}\big).}\end{array}
$$  

See  App. A.2  for proof. For   $t<T$ , we can construct a pa- rameterization of $\psi_{t}(\mathbf{s}_{1:t})$  such that the proposal is tractable to sample in transformer architectures, where the normal- ization   $\begin{array}{r}{Z_{t}^{\pi}(\mathbf{s}_{1:t-1})=\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}(\mathbf{s}_{1:t})}\end{array}$  P  sums over the discrete vocabulary of ne s $s_{t}\in\mathcal V$ . However, for the final timestep, note that $\phi(\mathbf{s}_{1:T})$  may require calls to a different neural network such as a reward model or classifier. We thus consider an approximate   $\psi_{T}(\mathbf{s}_{1:T})\approx\phi(\mathbf{s}_{1:T})$  for the proposal   $q_{T}(s_{T}|\mathbf{s}_{1:T-1})\propto p_{0}(s_{T}|\mathbf{s}_{1:T-1})\psi_{T}(\mathbf{s}_{1:T})$ the final step. With slight abuse of notation, we let $q^{\pi}(\mathbf{s}_{1:T})$ denote this tractable proposal over full sequences,  

$$
q^{\pi}(\mathbf{s}_{1:T}):=\Big(\prod_{t=1}^{T-1}q_{t}^{\pi}\big(s_{t}\vert\mathbf{s}_{1:t-1}\big)\Big)\;q_{T}\big(s_{T}\vert\mathbf{s}_{1:T-1}\big)\,.
$$  

Using this proposal, the incremental weights become  

$$
w_{t}(\mathbf{s}_{1:t})=\left\{\begin{array}{l l}{\displaystyle\frac{\sum_{s_{t}}p_{0}\left(s_{t}\left|\mathbf{s}_{1:t-1}\right)\psi_{t}\left(\mathbf{s}_{1:t}\right)}{\psi_{t-1}\left(\mathbf{s}_{1:t-1}\right)}}&{t<T}\\ {\displaystyle\frac{\sum_{s_{T}}p_{0}\left(s_{T}\left|\mathbf{s}_{1:T-1}\right)\psi_{T}\left(\mathbf{s}_{1:T}\right)}{\psi_{T-1}\left(\mathbf{s}_{1:T-1}\right)}\frac{\phi\left(\mathbf{s}_{1:T}\right)}{\psi_{T}\left(\mathbf{s}_{1:T}\right)}}&{t=T}\end{array}\right.,
$$  

which are independent of $s_{t}$  for $t<T$  

Variational Proposal As noted in  Sec. 2.1 , SMC with no resampling steps reduces to SIS with the full target distri- bution $\sigma(\mathbf{s}_{1:T})$ . Policy gradient methods ( Schulman et al. , 2017 ;  Parshakova et al. ,  2019 ;  Korbak et al. ,  2022a ;  Go et al. ,  2023 ) which directly learn a tractable approximation $q(\mathbf{s}_{1:T})$  to the target distribution may thus be viewed as a particularly simple instance of SMC, or inference more gen- erally (see  Korbak et al.  ( 2022b )). We may also evaluate these inference methods using our proposed tools in  Sec. 5 See  Table 1  and  App. E  for detailed losses and discussion.  

Finally, note that a separate proposal $q$  might also be learned alongside the twisting targets   $\bar{\{\pi_{t}\}_{t=1}^{T-1}}$ . This may be useful to approximate the variance-minimizing proposal for multi- step or adaptive resampling ( Prop. A.5 ) beyond the tractable optimal one-step proposal in  Prop. 3.3 . We discuss training losses based on multi-step importance weights in  App. C.1  

# 3.3. Conditional Target Distributions  

More generally, we may consider  conditional  target distri- butions, obtained by conditioning on an observation random variable $o_{T}$ . This mirrors the standard setting of SMC in state-space models ( Doucet et al. ,  2001 ;  Briers et al. ,  2010 ; Gu et al. ,  2015 ;  Maddison et al. ,  2017 ;  Lawson et al. ,  2022 ), with further discussion in  App. B.2 .  

De ing   $\phi(\mathbf{s}_{1:T},o_{T})=\sigma(o_{T}\vert\mathbf{s}_{1:T})$  as a prob del of $o_{T}$  , our target distribution is the posterior $\sigma(\mathbf{s}_{1:T}|o_{T})$ ,  

$$
\sigma(\mathbf{s}_{1:T}|o_{T})=\frac{1}{\mathcal{Z}_{\sigma}(o_{T})}p_{0}(\mathbf{s}_{1:T})\sigma(o_{T}|\mathbf{s}_{1:T})\;,
$$  

where the partition function   $\begin{array}{r l r}{\mathcal{Z}_{\sigma}(o_{T})}&{{}=}&{\sigma(o_{T})\quad=}\end{array}$ $\begin{array}{r}{\sum_{\mathbf{s}_{1:T}}p_{0}(\mathbf{s}_{1:T})\sigma(o_{T}\vert\mathbf{s}_{1:T})}\end{array}$  is the marginal of the given   $o_{T}$ . In this setting,  Prop. 3.2  suggests that the optimal twists, which match the marginals $\sigma(\mathbf{s}_{1:t}|o_{T})$ , correspond to the conditional likelihood of $o_{T}$  given $\mathbf{s}_{1:t}$ ,  

$$
\begin{array}{r l}&{\psi_{t}^{*}(\mathbf{s}_{1:t},o_{T})\propto\displaystyle\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi(\mathbf{s}_{1:T},o_{T})}\\ &{\qquad\qquad\qquad=\sigma(o_{T}|\mathbf{s}_{1:t})\;,}\end{array}
$$  

since $\begin{array}{r}{\sigma(o_{T}|\mathbf{s}_{1:t})~=~\sum_{\mathbf{s}_{t+1:T}}\sigma(o_{T},\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\end{array}$   P . We can proceed to construct intermediate target distributions and proposals as in the previous sections, where   $\psi_{t}(\mathbf{s}_{1:t},o_{T})$  and even   $q_{t}(s_{t}|\mathbf{s}_{1:t-1},o_{T})$  may be conditioned on a particular value of $o_{T}$  .  

To recover the unconditional setting, we can fix a binary observational variable   $\sigma(o_{T}=1|\mathbf{s}_{1:T}):=\phi(\mathbf{s}_{1:T})$  ( Levine , 2018 ) and omit explicit conditioning, showing that condi- tional twist learning generalizes our previous exposition.  

Exact Target Sampling on Simulated Data Assum- ing   $\sigma\big(o_{T}|\mathbf{s}_{1:T}\big)$  is tractable to sample, we may obtain an exact sample from the target posterior for simulated $o_{T}$  using ancestral sampling. In particular, by sampling $\mathbf{s}_{1:T},o_{T}\sim p_{0}(\mathbf{s}_{1:T})\sigma(o_{T}\vert\mathbf{s}_{1:T})$ , we obtain  the joint distribution, which also factorizes as $\sigma(o_{T},\mathbf{s}_{1:T})=$ $\sigma(o_{T})\sigma(\mathbf{s}_{1:T}|o_{T})$ . Using the latter factorization, we may interpret $\mathbf{s}_{1:T}$  as an exact sample from the target posterior for the given $o_{T}$ .  

We refer to this as the Bidirectional Monte Carlo (BDMC) trick ( Grosse et al. ,  2015 ;  2016 ), and will use it to draw exact samples for training in  Sec. 4.1.2  or evaluation in  Sec. 5 .  

# 3.4. Connections with Reinforcement Learning  

Twisted SMC shares close connections with (soft) rein- forcement learning ( Levine ,  2018 ;  Pich e et al. ,  2018 ;  Law- son et al. ,  2018 ;  Heng et al. ,  2020 ), which we develop with detailed discussion in  App. B.3  and  App. D . In par- ticular, we consider language modeling as a Markov De- cision Process (MDP) with states   $x_{t}:=\mathrm{\bf~s}_{1:t-1}$ , actions $a_{t}:=\,s_{t}$ , and deterministic transitions   $p(x_{t+1}|x_{t},a_{t})\;=\;$ $\delta(\mathbf{s}_{1:t}~=~\mathsf{c o n c a t}\big(s_{t},\mathbf{s}_{1:t-1}\big)\big)$ . We describe two differ- ent definitions of the reward function in relation to the potential  function $\phi(\mathbf{s}_{1:T})$  below. In  App. B.1 , we further extend our SMC framework to capture settings with interme- diate potentials $\phi_{t}\big(\mathbf{s}_{1:t}\big)$  or rewards over partial sequences.  

Base Model Policy Evaluation Viewing the final potential $\phi(\mathbf{s}_{1:T})$  as the reward function, the optimality condition   $\begin{array}{r}{\psi_{t}^{*}(\mathbf{s}_{1:t})~=~\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi(\mathbf{s}_{1:T})}\end{array}$  P  |  in Eq. (12)  corresponds to exact  policy evaluation  of the future reward under the  fixed  base model policy   $p_{0}\big(\mathbf{s}_{t+1:T}\big|\mathbf{s}_{1:t}\big)$ Mudgal et al.  ( 2023 ) adopt this perspective for controlled decoding, and refer to the twist functions as ‘prefix scorers’.  

Soft RL with KL Regularization Alternatively, we may consider the soft or KL-regularized RL target distributions commonly used in language modeling ( Levine ,  2018 ;  Ko- rbak et al. ,  2022b ) as a special case of our twisted SMC framework. For a regularization strength $\beta$ , define the ter- minal potential as  

$$
\phi(\mathbf{s}_{1:T})=e^{\beta r(\mathbf{s}_{1:T})}.
$$  

In this case, the intermediate twist functions in  Def. 3.1  cor- respond to state-action   $Q$ -values,   $\psi_{t}(\mathbf{s}_{1:t})=e^{\beta Q(s_{t},\mathbf{s}_{1:t-1})}$ ( App. B.3 ). In particular, consider the recursion for the optimal twists in  Eq. (13) . Taking the logarithm of both sides and recalling the definition of the soft value function $V^{*}(\mathbf{s}_{1:t})$  ( Levine ,  2018 ), we obtain  

$$
Q^{*}(s_{t},\mathbf{s}_{1:t-1})=\frac{1}{\beta}\log\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})e^{\beta Q^{*}(s_{t+1},\mathbf{s}_{1:t})},
$$  

which is a soft Bellman recursion with no intermediate reward. From the soft RL perspective, the twist functions are analogous to a critic, while the proposal plays the role of an actor ( Levine ,  2018 ;  Haarnoja et al. ,  2018 ). We provide detailed discussion of the soft RL case in  App. B.3 , and review RL-inspired losses for twist learning in  App. C.1 .  

Benefits of the Probabilistic Perspective While soft RL is a natural special case of our framework which gives intu- ition for the role of the twist functions, our approach allows for general target distributions without reference to RL ob- jectives and suggests principled probabilistic resampling using SMC. Further, we develop twist learning techniques inspired by density ratio estimation, including our novel CTL method or the SIXO objective from ( Lawson et al. , 2022 ), which are more naturally motivated from a proba- bilistic perspective. Finally, we leverage our probabilistic perspective to propose novel language model evaluation techniques inspired by Bidirectional Monte Carlo ( Grosse et al.  ( 2015 ;  2016 )) in  Sec. 5 .  

# 4. Learning the Twist Functions  

We next consider methods to learn twist functions $\psi_{t}^{\theta}$   param- eterized by neural networks, presenting a novel  contrastive twist learning  (CTL) approach in  Sec. 4.1 . We summarize twist learning methods from related work in  Sec. 4.2 .  

# 4.1. Contrastive Twist Learning  

To match our approximate   $\pi_{t}^{\theta}$   to the target marginals, we propose to minimize $T$  separate KL divergences,  

$$
\underset{\pmb{\theta}}{\mathrm{min}}\,\mathcal{L}_{\mathrm{CTL}}(\pmb{\theta}):=\underset{\pmb{\theta}}{\mathrm{min}}\sum_{t=1}^{T}D_{\mathrm{KL}}\big(\sigma(\mathbf{s}_{1:t})\,\big|\,\pi_{t}^{\pmb{\theta}}(\mathbf{s}_{1:t})\big)\,.
$$  

While other divergences could be used to learn   $\pi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$ , we argue that the mass-covering behavior of  Eq. (21)  is a desirable property for twist learning. Since we separately match each $\sigma(\mathbf{s}_{1:t})$ , our hope is that suboptimal learning in early timesteps does not lead to aggressive pruning of partial sequences that would achieve high final target likelihood.  

Using  Eq. (9) , the gradient of  Eq. (21)  at each $t$  becomes  

$$
\begin{array}{r}{\mathbb{E}_{\sigma(\mathbf{s}_{1:t})}\bigl[\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\bigr]-\mathbb{E}_{\pi_{t}^{\theta}(\mathbf{s}_{1:t})}\bigl[\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\bigr],}\end{array}
$$  

which allows us to learn from exact target samples of $\sigma(\mathbf{s}_{1:t})$ in the first term when they are available.  

We note the similarity of the objective in  Eq. (21)  and gradi- ent in  Eq. (22)  to maximum likelihood training of energy- based models (EBM)s. Due to the form of the gradient update, we refer to this method as  contrastive twist learning (CTL). We proceed to describe approximate techniques for positive sampling (first term) and negative sampling (second term) in the next subsections.  

# 4.1.1. A PPROXIMATE  N EGATIVE  S AMPLING  

A common challenge in energy-based modeling is that the second term in  Eq. (22)  involves sampling from the target $\pi_{t}$  with intractable normalization constant   $\mathbf{\bar{\mathcal{Z}}}_{t}^{\psi}$ . We proceed to estimate the expectation using SIS as in  Eq. (4) , using a proposal $q(\mathbf{s}_{1:t})$  such as the base model or the twist-induced proposal from  Sec. 3.2 . Note that SMC resampling with learned intermediate twist functions could also be used.  

# 4.1.2. (A PPROXIMATE ) P OSITIVE  S AMPLING  

In contrast to traditional EBM settings, we do not necessar- ily have exact samples available from a ‘data’ distribution. We describe several settings related to availability of positive samples, which are explored in our experiments in  Sec. 7 .  

Exact Target Samples If exact posterior samples are available, for example using the BDMC trick in  Sec. 3.3 , we may use them directly in the gradient update in  Eq. (22)  

Rejection Sampling Rejection sampling can yield ex- act target samples $\mathbf{s}_{1:T}^{\sigma}$   when an upper bound on the like- lihood ratio $\begin{array}{r l r}{\frac{\tilde{\sigma}\left(\mathbf{s}_{1:T}\right)}{q\left(\mathbf{s}_{1:T}\right)}\mathbf{\Omega}}&{\le}&{M}\end{array}$  is known. In cases where the target   $\tilde{\sigma}(\mathbf{s}_{1:T})$  is defined by thresholding or an indi- $p_{0}\bigl(\mathbf{s}_{1:T}\bigr)\mathbb{I}\bigl(\mathbf{s}_{1:t}\ \in\ \mathcal{C}\bigr)$  or j ribution $p_{0}(\mathbf{s}_{1:T})\sigma(o_{T}|\mathbf{s}_{1:T})$  | ,  learly take $M\,=\,1$  for the base model proposal $p_{0}(\mathbf{s}_{1:T})$ . If the base model yields posterior samples in reasonable time, we can obtain exact  

  
Table 1: Losses for twist (top) and proposal (bottom) learning, where   $\pi_{s}(\cdot)$  indicates an arbitrary sampling distribution.  

samples for training using rejection sampling, and use our twist learning procedures to greatly improve sampling effi- ciency at generation time.  

While an improved proposal $q$  should more efficiently draw samples meeting the target conditions, exact rejection sam- pling would require estimation of   $M$ . Approximate or quasi rejection sampling might be used in this case, as analysed in  Eikema et al.  ( 2022 ).  

Approximate Positive Sampling using SIS or SMC In cases where exact samples are unavailable and rejection sam- pling is inefficient or inexact, we leverage SMC sampling with twist tar s   $\{\pi_{t}^{\theta}\}_{t=1}^{T}$   }   and any proposal   $q(\mathbf{s}_{1:T})$  to first draw a set of  K  full sequences $\mathbf{s}_{1:T}^{1:K}$   . As in  Eq. (4) , we can use the normalized SMC weights since the last resampling step to estimate the expected gradient in the first term of Eq. (22) . Without resampling, we recover SIS estimation.  

While both our approximate positive and negative sampling for estimating the expectations in  Eq. (22)  rely on SMC or SIS weights (often with the same proposal), the crucial distinction is that weights for  positive  sampling are based on the  true target potential   $\phi(\mathbf{s}_{1:T})$  over  full  sequences.  

Truncation to Partial Sequences For an exact positive sample, we use its truncation to a partial sequence of length $t$  (which corresponds to a sample from the desired marginal $\sigma_{t}$ ) to perform the gradient update in  Eq. (22) . For approx- imate positive sampling, we use the same set of   $K$  final weights to estimate the expected gradient at each timestep.  

# 4.2. Twist Learning Methods from Related Work  

We briefly describe alternative approaches for twist learning, with detailed discussion in  App. C  and a summary of the loss functions for methods used in our experiments in  Table 1 .  

Soft Q-Learning (RL) Enforcing the recursion in  Eq. (13) using a squared error loss is analogous to soft   $Q$ -learning in the RL literature (see  Eq. (20) ), and has been used for twisted SMC in  Lioutas et al.  ( 2022 ).  Mudgal et al. ( 2023 ) derive a similar squared-error loss, viewing $\phi(\mathbf{s}_{1:T})$ as the reward. Finally, we interpret path consistency losses ( Nachum et al. ,  2017 ), which were derived in the soft RL setting and have been used for language modeling in  Guo et al.  ( 2021 );  Hu et al.  ( 2023 ), from an importance sampling perspective in  App. C.1  and  E.1  

SIXO The SIXO loss proposed by  Lawson et al.  ( 2022 ) learns twist functions using a binary classification task to distinguish samples from the target marginal   $\sigma(\mathbf{s}_{1:t}|o_{T})$  and base model $p_{0}\big(\mathbf{s}_{1:t}\big)$  at each step, which corresponds to noise contrastive estimation ( Gutmann & Hyv arinen ,  2010 ) for learning energy-based models. See  App. C.3 .  

FUDGE Yang & Klein  ( 2021 ) learn twists by construct- ing a binary classification task to instead learn the condi- tional likelihood $\sigma\big(o_{T}|\mathbf{s}_{1:t}\big)$  ( Eq. (18) ). This may be viewed as enforcing the $T\!-\!t$  −  step optimality equation in  Eq. (12)  or Eq. (18) , where rollouts should be obtained using the base model $p_{0}\big(\mathbf{s}_{t+1:T}\big|\mathbf{s}_{1:t}\big)$  (see  Table 1  or  App. C.4 ).  Mudgal et al.  ( 2023 );  Deng & Raffel  ( 2023 ) similarly propose to enforce the $T-t$  step optimality condition using a squared- error loss, $\begin{array}{r}{\sum_{t}\mathbb{E}_{p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}[(\phi(\mathbf{s}_{1:T})-\psi_{t}(\mathbf{s}_{1:t}))^{2}]}\end{array}$ . |  

5. Evaluating Inference in Language Models Our SMC framework yields a rich set of tools for evaluating inference techniques in language models, using well-studied quantities such as the log partition function $\log\mathcal{Z}_{\sigma}$ divergence to the target distribution. Remarkably, with ac- cess to a single exact sample from the target distribution, we show in  Prop. 5.1  that we can obtain  upper  bounds on $\log\mathcal{Z}_{\sigma}$ in addition to lower bounds. These bounds can tightly sand- wich $\log\mathcal{Z}_{\sigma}$  with increasing   $K$ , thereby ensuring reliable conclusions regarding inference quality.  

# 5.1. Applications of $\log\mathcal{Z}_{\sigma}$  Estimation  

Evaluating Fine-Tuned Models To motivate this section and present an important application of our SMC methods, consider evaluating how well a given   $q(\mathbf{s}_{1:T})$  matches a target distribution for controlled generation or fine-tuning. Assume that $q$  is tractable to sample and evaluate. To cal- culate the KL divergence to $\sigma$  in either direction, we also require an estimate of the  log  partition function $\log\mathcal{Z}_{\sigma}$ ,  

$$
\begin{array}{r l}&{D_{\mathrm{KL}}\big(q(\mathbf{s}_{1:T})\,\|\,\sigma(\mathbf{s}_{1:T})\big)=\mathbb{E}_{q}\left[\log\frac{q\big(\mathbf{s}_{1:T}\big)}{p_{0}\left(\mathbf{s}_{1:T}\right)\phi\left(\mathbf{s}_{1:T}\right)}\right]+\log\mathcal{Z}_{\sigma}}\\ &{D_{\mathrm{KL}}\big(\sigma(\mathbf{s}_{1:T})\,\|\,q(\mathbf{s}_{1:T})\big)=\mathbb{E}_{\sigma}\left[\log\frac{p_{0}\left(\mathbf{s}_{1:T}\right)\phi\left(\mathbf{s}_{1:T}\right)}{q\left(\mathbf{s}_{1:T}\right)}\right]-\log\mathcal{Z}_{\sigma}}\\ &{\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\mathcal{Q}3)}\end{array}
$$  

For $D_{\mathrm{KL}}(\sigma\parallel q)$ , note that we also require samples from the target $\sigma$ , as may be readily available using the BDMC trick when $\sigma$  is defined as a Bayesian posterior ( Sec. 3.3 ). In such cases, we argue that SMC can be used to accurately bound the value of $\log\mathcal{Z}_{\sigma}$  and estimate each KL divergence above. Estimation of $D_{\mathrm{KL}}(\sigma\parallel q)$  may be particularly important to diagnose mode-dropping in inference techniques such as PPO which optimize the mode-seeking   $D_{\mathrm{KL}}(q\,\|\,\sigma)$  during fine-tuning ( Korbak et al. ,  2022b ).  

Evaluating Twisted SMC Sampling After running SIS or SMC with   $K$  samples, we can sample a single index as in  Eq. (5)  to return a single approximate target sample $\mathbf{s}_{1:T}^{\sigma}$   . However, the marginal distribution of this sample, which we denote as   $\mathbf{s}_{1:T}^{\sigma}\sim q_{\scriptscriptstyle\mathrm{SMC}}(\mathbf{s}_{1:T})$   ∼ , is not t table due to the need to sum over all possible sets of  K  samples. Nevertheless, we will show below that the tightness of our $\log\mathcal{Z}_{\sigma}$  Prop. 5.1  provides upper bo es   $D_{\mathrm{KL}}\big(q_{\mathrm{SMC}}\big(\mathbf{s}_{1:T}\big)\parallel\boldsymbol{\sigma}\big(\mathbf{s}_{1:T}\big)\big)$ or $D_{\mathrm{KL}}\big(\sigma(\mathbf{s}_{1:T})\parallel q_{\mathrm{SMC}}(\mathbf{s}_{1:T})\big)$  ∥ , respectively.  

Alternatively, we can also use the single-sample KL di- vergences in  Eq. (23)  for the twist-induced proposal $q^{\pi}$   in Eq. (15)  to evaluate a set of twist functions $\psi_{t}$  ( Sec. 7.2 ).  

# 5.2. Bidirectional SMC Bounds on $\log\mathcal{Z}_{\sigma}$  

Given the importance of   $\log\mathcal{Z}_{\sigma}$  estimation as motivated above, we propose a  bidirectional SMC  stochastic upper bound which is novel (to the best of our knowledge), and may be of interest outside of the language modeling setting.  

Recall from  Sec. 2.2  that SMC admits an interpretation as SIS in an extended state space   $S:=\{s_{t}^{k},\omega_{t}^{k}\}_{k=1,t=1}^{K,\hat{T}}$     which includes all tokens and resampling indices. We derive lower and upper bounds on $\log\mathcal{Z}_{\sigma}$  in  Prop. 5.1  below, with proof and detailed description of the extended state space target $\sigma_{\mathrm{SMC}}(S)$  and proposal   $q_{\mathrm{SMC}}(\pmb{S})$  distributions in  App. F .  

Proposition 5.1. (Bidirectional SMC Bounds)  The log partition function $\log\mathcal{Z}_{\sigma}$  of a target distribution $\sigma(\mathbf{s}_{1:T})$ can be lower and upper bounded by  

$$
\begin{array}{r l}&{\mathbb{E}_{q_{\mathtt{s m c}}(S)}\left[\log\prod_{t=1}^{T}\frac{1}{K}\sum_{i=1}^{K}w_{t}\big(\mathbf{s}_{1:t}^{i}\big)\right]\leq\log\mathcal{Z}_{\sigma}}\\ &{\quad\quad\log\mathcal{Z}_{\sigma}\leq\mathbb{E}_{\sigma_{\mathtt{s m c}}(S)}\left[\log\prod_{t=1}^{T}\frac{1}{K}\sum_{i=1}^{K}w_{t}\big(\mathbf{s}_{1:t}^{i}\big)\right].}\end{array}
$$  

The gap in the lower bound is $D_{\mathrm{KL}}(q_{\mathrm{SMC}}(S)\,||\,\sigma_{\mathrm{SMC}}(S))$ , and the gap in the upper bound is $D_{\mathrm{KL}}(\sigma_{\mathrm{SMC}}(S)\parallel q_{\mathrm{SMC}}(S))$ .  

See  App. F  for a detailed discussion and derivations. The proof proceeds by adapting a general approach for extended state space log partition function bounds from  Brekelmans et al.  ( 2022 ) using the probabilistic interpretation of SMC from  Andrieu et al.  ( 2010 );  Maddison et al.  ( 2017 ). With no resampling, the SIS case recovers the Importance Weighted Autoencoder (IWAE) lower ( Burda et al. ,  2015 ) and upper ( Sobolev & Vetrov ,  2019 ;  Brekelmans et al. ,  2022 ) bounds.  

Sampling from $\sigma_{\mathrm{SMC}}$  for SMC Upper Bounds We now discuss sampling from   $\sigma_{\mathrm{SMC}}(S)$  for the expectation in the upper bound, which requires a single,  exact  sample from the target distribution $\sigma(\mathbf{s}_{1:T})$ . This sample may be obtained, for example, using the BDMC trick in  Sec. 3.3 . Note that Sec. 2.2  and  Alg. 1  describe sampling from $q_{\mathrm{SMC}}(\pmb{S})$ , which is used for the expectation in the lower bound.  

Sampling from   $\sigma_{\mathrm{SMC}}(S)$  differs from sampling from $q_{\mathrm{SMC}}(\pmb{S})$  by its treatment of the exact target sample. In particular, the partial sequence corresponding to the exact target sample is guaranteed to be cloned once at each re- sampling step. In other indices, resampling proceeds as in Sec. 2.2 , where the exact sample may be cloned additional times based on its incremental importance weights. Finally, we sample   $K-1$  next tokens from the proposal, while the value of the remaining chain is fixed by the exact target sample. See  App. F  and  Alg. 2  for detailed discussion.  

Tightness of the Bidirectional Bounds Since the bounds in  Prop. 5.1  become exact as   $K\rightarrow\infty$ for any proposal ( Burda et al. ,  2015 ;  Maddison et al. ,  2017 ), we can use SMC or IWAE with large $K$  to sandwich the  log  partition function when   $\sigma$  samples are available.  

For a given   $K$ , the gap in the extended state space $\log\mathcal{Z}_{\sigma}$  bounds in  Prop. 5.1  provides further insight into the quality of twisted SMC sampling via the dis- tribution of the marginal sample   $\mathbf{s}_{1:T}^{\sigma}$   ( Sec. 5.1 ). In particular, the data processing inequality suggests that $D_{\mathrm{KL}}\big(q_{\mathrm{SMC}}(\mathbf{s}_{1:T})\parallel\boldsymbol{\sigma}(\mathbf{s}_{1:T})\big)\leq D_{\mathrm{KL}}\big(q_{\mathrm{SMC}}(S)\parallel\boldsymbol{\sigma}_{\mathrm{SMC}}(S)\big)$ a $\begin{array}{r}{\mathrm{id}\,D_{\mathrm{KL}}\big(\sigma(\mathbf{s}_{1:T})\,\|\,q_{\mathrm{SMC}}(\mathbf{s}_{1:T})\big)\leq D_{\mathrm{KL}}\big(\sigma_{\mathrm{SMC}}(S)\,\|\,q_{\mathrm{SMC}}(S)\big)}\end{array}$ ) ( Grosse et al. ,  2015 ;  2016 ). Thus, if the difference between upper and lower b nds on   $\log\mathcal{Z}_{\sigma}$  is small, then we can conclude that the  K -sample SMC or SIS procedures in Sec. 2.2  yield a single approximate sample   $\mathbf{s}_{1:T}^{\sigma}$   whose distribution   $q_{\mathrm{SMC}}(\mathbf{s}_{1:T})$  is close to the target   $\sigma(\mathbf{s}_{1:T})$  in symmetrized KL divergence.  

# 6. Related Work  

In the previous sections, we have discussed related work as it fit within our SMC framework for language modeling. Note that  Lew et al.  ( 2023 ) consider SMC sampling for language models, but do not learn twist functions or proposals.  

Decoding from language models to obtain diverse ( Holtz- man et al. ,  2019 ;  Vilnis et al. ,  2023 ) or controlled generation ( Zhang et al. ,  2023 ;  Dathathri et al. ,  2019 ;  Krause et al. , 2020 ;  Yang & Klein ,  2021 ;  Guo et al. ,  2021 ;  Qin et al. , 2022 ;  Snell et al. ,  2022 ;  Hu et al. ,  2023 ) is an active area of research. Our SMC resampling approach may be viewed as a principled  probabilistic  extension of best-of- $.K$  decoding methods.  Mudgal et al.  ( 2023 ) propose a $K$ -way  arg max decoding scheme based on ‘prefix scorers’ $\psi_{t}$  learned us- ing  Eq. (13) , but also consider using these twists as logits for softmax sampling in the proposal. However, neither of these decoding schemes are aligned with our proposed SMC framework, as we discuss in detail in  App. D . For example, greedy  arg max  decoding with respect to the op- timal twists in  Prop. 3.2  does not yield samples from the target distribution $\sigma(\mathbf{s}_{1:T})$ .  

Finally, RL-based methods such as PPO maintain both a policy or proposal network and value network or advantage estimator during training. From the soft RL perspective in Sec. 3.4  and  App. B.3 , the soft values play a similar role as our twist functions for SMC resampling.  Liu et al.  ( 2023 ) consider using Monte Carlo Tree Search (MCTS) based on PPO value estimates to improve decoding, while  Chaffin et al.  ( 2022 ) consider discriminator-driven MCTS.  

# 7. Experiments  

We now illustrate empirically how our framework can be used to evaluate inference through $\log\mathcal{Z}_{\sigma}$  bounds and KL divergences between the sampling and target distributions, providing meaningful quantitative comparison between vari- ous learning methods. We consider a range of tasks through- out this section, including toxic story generation (as an example of uncovering rare undesirable behavior), gen- erating reviews with varied sentiment, and infilling. For the toxicity and infilling tasks, we consider the TinySto- ries model ( Eldan & Li ,   $2023)^{4}$   as a small-scale model where the generation is coherent, and use the prompt of ‘Once upon a time, there was a’. For the toxicity task, we elicit responses judged to be toxic by the classifier from  Corr ea   $(2023)^{5}$ . For the sentiment task, we con- sider the GPT2-Medium 6   model and a classifier trained on Amazon reviews. Our code is available at  https: //github.com/Silent-Zebra/twisted-smc-lm  .  

# 7.1. Comparing SIS and SMC for $\log\mathcal{Z}_{\sigma}$  Estimation  

We first use our   $\log\mathcal{Z}_{\sigma}$  bounds to test how twisted SMC can improve upon SIS and efficiently sample rare events. We consider the task of toxic story generation. The target $\sigma(\mathbf{s}_{1:T})\,\propto\,p_{0}(\mathbf{s}_{1:T})\mathbb{I}[\mathbf{s}_{1:T}\,\in\,\mathcal{C}]$  where $\mathcal{C}:=$ $\left\{\mathbf{s}_{1:T}\;|r(\mathbf{s}_{1:T})\leq\eta\right\}$ {  | } , $r(\mathbf{s}_{1:T})$  is the non-toxic l and the threshold $\eta=-5$  −  corresponds to a greater than 99% chance of being toxic. Rejection sampling under   $p_{0}$  yields exact  

  
Figure 3:  Compa n of SIS (IWAE) and SMC bo ds on $\log\mathcal{Z}_{\sigma}$ for base proposal $p_{0}$  and twist-induced proposal $q^{\pi}$ , with twists learned with CTL. With the twist-induced proposal, both SIS and SMC bounds are tight; with the base proposal, resampling with learned twists is needed. Resampling based on ESS instead of every-step resampling yields similar results.  

samples for $\log\mathcal{Z}_{\sigma}$  UB estimation, but can require hundreds of thousands of samples. Thus, this setting also allows us to test the effectiveness of approximate positive sampling for twist training when target samples are rare.  

Fig. 3  demonstrates that training twists with CTL and ap- proximate positive sampling can significantly improve log partition function estimation and sampling efficiency. We first note that both upper and lower bounds tighten as   $K$ increases, as expected, for both SIS and SMC. Using   $p_{0}$  as proposal, the SIS LB (orange) generally fails to draw any samples meeting the threshold. By contrast, SMC resam- pling (red) with $p_{0}$  proposal eventually achieves  tight $\log\mathcal{Z}_{\sigma}$ upper and lower bounds, yielding near-exact target samples (small KL divergence between the distribution over samples and the target distribution) by the reasoning in  Sec. 5 .  

However, both SMC and SIS with the twist-induced pro- posal achieve tight estimation and near-exact sampling of the target toxic outputs with orders of magnitude lower   $K$ Resampling does not appear to help or hurt these bounds, as the effect of the twists has been incorporated in the proposal $q^{\pi}$   in  Eq. (15) . Thus, we conclude that using the twist- induced proposal can provide significant efficiency gains over base model sampling.  

# 7.2. Evaluating Twist-Induced or Variational Proposals  

We next leverage our   $\log\mathcal{Z}_{\sigma}$  bounds t single- sample inference using   $D_{\mathrm{KL}}(q\,\|\,\sigma)$  and $D_{\mathrm{KL}}(\sigma\,\|\,q)$  ∥ , as in Sec. 5.1 . Across settings, we consider two SIS proposal- learning methods: PPO ( Schulman et al. ,  2017 ) which min- imizes $D_{\mathrm{KL}}(q\,\|\,\sigma)$  during optimization, and distributional policy gradient (DPG), which minimizes $D_{\mathrm{KL}}(\sigma\parallel q)$  ( Par- shakova et al. ,  2019 ) (see  App. E ).  

Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo 
  
Table 2: Toxicity ( Sec. 7.2.1 ) Table 3: Sentiment ( Sec. 7.2.2 ) Table 4: Infilling ( Sec. 7.2.3 )  

We consider four twist learning methods, including CTL and RL from  Sec. 4 , SIXO ( Lawson et al. ,  2022 ), and FUDGE ( Yang & Klein ,  2021 ) (see  App. C ). For each, we measure KL divergences involving the twist-induced proposal $q^{\pi}$ . Thus,  these experiments showcase two comple- mentary applications of SMC : as a novel inference method yielding a tractable   $q^{\pi}$ , and as an evaluation method for any other inference method (such as PPO) using   $K$ -sample bounds on $\log\mathcal{Z}_{\sigma}$  

# 7.2.1. GENERATING TOXIC STORIES  

We consider toxic story generation as in  Sec. 7.1 , but us- ing a target   $\sigma({\bf s}_{1:T})~\propto~p_{0}({\bf s}_{1:T})p(a~=~1|{\bf s}_{1:T})$ , where $p(a\,=\,1|\mathbf{s}_{1:T})$  denotes the probability of the text being judged as toxic by a classifier. Compared to the threshold- ing target, this task provides a smoother gradient signal for learning (see  App. G.3 ) but still allows for exact sampling via rejection sampling. We train using approximate posi- tive sampling, but provide an ablation with exact positive sampling results in  App. H.3 .  

We report KL divergences in  Table 2 . We observe that PPO learns the best proposal with respect to   $D_{\mathrm{KL}}(q\,\|\,\sigma)$  while our CTL method performs best with respect to $D_{\mathrm{KL}}(\sigma\parallel q)$ , which is consistent with the divergences minimized dur- ing training. Finally, in  App. H.1  we provide a quali- tative example of a toxic story generated with CTL for $\sigma(\mathbf{s}_{1:T})\,\propto\,p_{0}(\mathbf{s}_{1:T})p(a\,=\,1|\mathbf{s}_{1:T})^{\beta}$   with   $\beta\,=\,10$ , a case where no exact samples are available.  

# 7.2.2. G ENERATION WITH  V ARIED  S ENTIMENT  

For the sentiment setting described earlier, we consider a prompt ‘I boug and target $\sigma(\mathbf{s}_{1:T})\propto p_{0}(\mathbf{s}_{1:T})p(a=$ $1|\mathbf{s}_{1:T})$ , where $a=1$  indicates a 1-star review and exact samples are available by rejection sampling. We train using approximate positive sampling (see  App. H.3  for compari- son with exact). While all methods achieve low KL diver- gences in  Table 3 , CTL performs best for both directions.  

# 7.2.3. I NFILLING  

In this section, we demonstrate a  conditional twist function parameter iz ation, where   $\psi_{t}^{\boldsymbol\theta}(\mathbf{s}_{1:t},o_{T})$  takes input $o_{T}$   which identifies the target distribution   $\sigma(\mathbf{s}_{1:T}|o_{T})$  as in  Sec. 3.3 We consider an infilling task ( Lew et al. ,  2023 ;  Hu et al. , 2023 ), where the observation variables   $o_{T}\;:=\;\mathbf{s}_{T+1:T+c}$ correspond to continuation tokens, and their likelihood $\sigma(o_{T}|\mathbf{s}_{1:T})\,:=\,p_{0}\bigl(\mathbf{s}_{T+1:T+c}|\mathbf{s}_{1:T}\bigr)$  is evaluated under the base model, given generated $\mathbf{s}_{1:T}$  . The target distribution corresponds  posterio $\sigma(\mathbf{s}_{1:T}|o_{T})$ . Instead of train- ing separate  { $\{\psi_{t}^{\theta}\}$   }  for eac $o_{T}$ ortize learning of a conditional twist network $\psi_{t}^{\boldsymbol\theta}(\mathbf{s}_{1:t},o_{T})$ .  

A second distinctive feature of this setting is that we  train from exact posterior or target samples, which are readily available using the BDMC trick in  Sec. 3.3 . In particular, we may sample sequences of length $T+c$  from the base model   ${\bf s}_{1:T+c}\,\sim\,p_{0}({\bf s}_{1:T+c})\,=\,\sigma({\bf s}_{1:T},o_{T}\,=\,{\bf s}_{T+1:T+c})$ , and interpret the prefix ${\bf s}_{1:T}\sim\sigma\big({\bf s}_{1:T}|o_{T}={\bf s}_{T+1:T+c}\big)$  ∼  | )  as a target sample. Note that we do not explicitly control the continuations tokens $o_{T}$  defining the tasks. We evaluate av- erage KL divergences over 2000 different   ${\cal O}_{T}={\bf s}_{T+1:T+c}$ , with $T=15$  and   $c=10$ , and report results in  Table 4 .  

We find that DPG performs best for both directions of the KL divergence in this setting, likely due to its ability to lever- age exact positive samples by minimizing   $D_{\mathrm{KL}}(\sigma_{o_{T}}\parallel q_{o_{T}})$ . While CTL also learns from exact positive samples, it re- quires approximate negative sampling and only performs comparably to SIXO, which uses exact positive samples and performs exact negative sampling under   $p_{0}$ . Finally, PPO trains from   $q_{o_{T}}$  samples only, and performs relatively poorly with respect to $D_{\mathrm{KL}}(\sigma_{o_{T}}\parallel q_{o_{T}})$ . We show qualitative results in  App. H.1  to correlate KL divergence results with sample quality.  

Using our KL divergence evaluation methods, we conclude DPG may be preferable when exact target samples are avail- able ( Sec. 7.2.3 ,  App. H.3 ), while CTL may be preferable with approximate positive sampling ( Sec. 7.2.1 ,  Sec. 7.2.2 ).  

# 8. Conclusion  

In this work, we have presented twisted SMC as a principled probabilistic inference framework for solving numerous ca- pability and safety tasks in LLMs. After discussing different design choices for twisted SMC and their relation to related work, we proposed a novel contrastive method for twist learning. Furthermore, we have proposed novel bidirec- tional SMC bounds for evaluating LLM inference methods. We demonstrated the effectiveness of our methods quanti- tatively and qualitatively in both sampling and evaluation across a variety of experimental settings.  

# Acknowledgments  

AM and RG acknowledge support from the Canada CIFAR AI Chairs program and from Open Philanthropy. SZ thanks Juhan Bae for helping debug memory issues in the code. Resources used in this research were provided, in part, by the Province of Ontario, the Government of Canada, and companies sponsoring the Vector Institute. We thank the anonymous reviewers for helpful comments on earlier ver- sions of this paper.  

# Impact Statement  

This paper is motivated by the social consequences of re- cent advances in the field of machine learning. Controlled generation from language models has the potential to im- prove safety through better steering of generation to human preferences, more efficient automated red-teaming, and the ability to estimate or bound probabilities of rare behaviors. Any such work is inherently a double-edged sword; the same techniques used to generate samples from a harmless distribution of text could, with a single sign change, be repurposed for generating samples from a harmful distri- bution of text. Thus, better controlled generation (in our framework, better sampling from target distributions) can provide benefits in the hands of responsible users but can also magnify harms in the hands of malevolent users (who have access to model weights).  

Overall, we believe the potential positive social benefits of our work in evaluation and steering language model output towards desired target distributions outweigh the potential negatives stemming primarily from misuse.  

# Appendix  

# A. Proofs  

In this section, we present the sense in which the target marginals correspond to the  optimal  intermediate distributions in twisted SMC. We defer proof of  Prop. 3.2  from the main text to slightly more general version in  App. B.1 Prop. B.1 , although  Prop. A.4  provides the analogous statement in terms of the intermediate target distributions   $\pi_{t}^{*}(\mathbf{s}_{1:t})=\sigma(\mathbf{s}_{1:t})$ instead of the optimal twists $\psi_{t}^{*}$   .  

We also prove  Prop. 3.3  from the main text in  App. A.2  and derive the gradient of the CTL loss ( Eq. (22) ) in  App. A.3 .  

# A.1. Proof for Optimal Intermediate Target Distributions  

In order to achieve sampling from the full joint distribution   $\sigma(\mathbf{s}_{1:T})$ , each intermediate target   $\sigma(\mathbf{s}_{1:t})$  must match the intermediate marginal   $\sigma(\mathbf{s}_{1:t})$ . To formalize this notion, we provide the following definition of optimality, justified by the fact that it yields an exact partition function estimator.  

To do so, we will consider the multi-step importance weights  

$$
\scriptstyle\varepsilon:t+c-1}(\mathbf{s}_{1:t+c-1})\,=\,\prod_{\tau=t}^{t+c-1}w_{\tau}(\mathbf{s}_{1:\tau})\,=\,\prod_{\tau=t}^{t+c-1}\frac{\tilde{\pi}_{\tau}(\mathbf{s}_{1:\tau})}{\tilde{\pi}_{\tau-1}(\mathbf{s}_{1:\tau-1})q(s_{\tau}|\mathbf{s}_{1:\tau-1})}\,=\,\frac{\tilde{\pi}_{t+c-1}(\mathbf{s}_{1:t+c-1})}{\tilde{\pi}_{t-1}(\mathbf{s}_{1:t-1})q(\mathbf{s}_{t:t+c-1}|\mathbf{s}_{1:\tau-1})},
$$  

( $c$ -Step SMC Weights)  

using a telescoping cancellation in the final equality. The one-step weights correspond to   $c=1$ , denoted simply as   $w_{t}$ .  

Definition A.1  ( Optimal Twisted SMC Sampling ) .  For a given target distribution $\begin{array}{r}{\sigma(\mathbf{s}_{1:T})\propto p_{0}(\mathbf{s}_{1:T})\phi(\mathbf{s}_{1:T}),}\end{array}$ , we ref to a twisted SMC procedure,   $\mathrm{SMC}(\{\pi_{t}\}_{t=1}^{T},q,K)$  or   $\mathbf{SMC}(p_{0},\{\psi_{t}\}_{t=1}^{T},q,K)$  (with   $\pi_{T}=\sigma$  or $\psi_{T}=\phi_{,}$ ),  $a s$  optimal  if $c$ -step importance weights $w_{t:t+c-1}(\mathbf{s}_{1:t+c-1})=\mathcal{Z}_{t+c-1}^{\psi}/\mathcal{Z}_{t-1}^{\psi}$   for all $1\leq t\leq T$  and   $0\leq c\leq T-t+1$ .  

Note, that the role of $\psi_{t}$  and $\mathcal{Z}_{t}^{\psi}$   is specified in  Def. 3.1 . We assume $\pi_{T}=\sigma$  for the goal of estimating   $\mathcal{Z}_{\sigma}$ , and show below that an optimal twisted SMC procedure yields an exact partition function estimator.  

Proposition A.2  ( Optimal SMC yields Exact Partition Function Estimation ) .  For any optimal twisted SMC procedure, the resulting estimator of the partition function   $\mathcal{Z}_{\sigma}$  has zero bias and zero variance.  

Proof.  As in  Footnote 1  or  App. F Alg. 2 , consider   $\{t_{r}\}_{r=1}^{R}$   index timesteps where resampling occurs and fix   $t_{0}=0$  and $t_{R}=T$ . The SMC estimator of   $\mathcal{Z}_{\sigma}=\mathcal{Z}_{T}^{\psi}$   becomes $\begin{array}{r}{\hat{\mathcal{Z}}_{\sigma}^{\mathtt{S M C}}=\prod_{r=1}^{R}\frac{1}{K}\sum_{i=1}^{K}\Bigl(\prod_{t=t_{r-1}+1}^{t_{r}}w_{t}\left(\mathbf{s}_{1:t}^{i}\right)\Bigr)}\end{array}$ P Q     for   $S\sim q_{\mathrm{SMC}}(S)$ . Using the optimality definition in  Def. A.1 , we have $w_{t}(\mathbf{s}_{1:t})\,=\,\mathcal{Z}_{t}^{\psi}/\mathcal{Z}_{t-1}^{\psi}$  Z   for all partial sequences   $\mathbf{s}_{1:t}$ . Noting the − telescoping multiplicative cancellation and the fact that   $w_{t}(\mathbf{s}_{1:t}^{i})=\mathcal{Z}_{t}^{\psi}/\mathcal{Z}_{t-1}^{\psi}$  Z Z   is constant with respect to indices $i\in[1,K]$ , − we have the following estimator for a single run of an optimal SMC procedure,  

$$
\hat{\mathcal{Z}}_{\sigma}^{\mathrm{{sc}}}=\prod_{r=1}^{R}\frac{1}{K}\sum_{i=1}^{K}\left(\prod_{t=t_{r-1}+1}^{t_{r}}w_{t}\Big(\mathbf{s}_{1:t}^{i}\Big)\right)=\prod_{r=1}^{R}\frac{\mathcal{Z}_{t_{r}}^{\psi}}{\mathcal{Z}_{t_{r-1}}^{\psi}}=\frac{\mathcal{Z}_{t_{R}}^{\psi}}{\mathcal{Z}_{t_{0}}^{\psi}}=\frac{\mathcal{Z}_{T}^{\psi}}{\mathcal{Z}_{0}^{\psi}}=\mathcal{Z}_{\sigma}
$$  

as desired, assuming   $\mathcal{Z}_{0}^{\psi}=1$ . Since $\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}=\mathcal{Z}_{\sigma}$ Z  Z  is independent of   $S$ , we conclude $\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}$ Z has zero bias and zero variance.  

Not define optimality in  Def. A.1  using the condition that   $w_{t:t+c-1}(\mathbf{s}_{1:t+c-1})=\mathrm{cons}$ t  for all   $1\leq t\leq T$ $0\leq c\leq T-t+1$  ≤  ≤  − imilar derivations as ab yield $\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}=\mathrm{const}$ Z . As we will show in  App. F , $\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}$ Z is unbiased with $\mathbb{E}[\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}]=\bar{\mathcal{Z}}_{\sigma}$ Z  Z . We thus conclude that   $\hat{\mathcal{Z}}_{\sigma}^{\mathrm{SMC}}=\mathcal{Z}_{\sigma}.$ Z  Z  with zero variance, and thus  Prop. A.2  holds.  

With this notion of optimality in mind, we demonstrate the following necessary and sufficient conditions.  

Proposition A.3  ( Optimality Conditions ) .  The following conditions are necessary and sufficient for twisted SMC optimality,  

$$
\begin{array}{r l r l r l}&{(i):}&{\quad}&{\pi_{t}^{*}(\mathbf{s}_{1:t})=\sigma(\mathbf{s}_{1:t})}&{\quad}&{\forall\quad1\le t\le T}\\ &{(i i):}&{\quad}&{q_{t}^{*}(s_{t}|\mathbf{s}_{1:t-1})=\sigma(s_{t}|\mathbf{s}_{1:t-1})}&{\quad}&{\forall\quad1\le t\le T\,.}\end{array}
$$  

Proof.  (Necessary) Optimal Twisted  ${\mathit{S M C}}\implies(i),(i i)$ ⇒ :  We begin by writing the marginalization of the unnormalized density   $\tilde{\pi}_{t+c}^{*}$   over prefixes of length   $t$  as  

$$
\mathbf{\Psi}_{t+c}^{*}(\mathbf{s}_{1:t})=\sum_{\mathbf{s}_{t+1:t+c}}\tilde{\pi}_{t+c}^{*}(\mathbf{s}_{1:t+c})=\sum_{\mathbf{s}_{t+1:t+c}}p_{0}(\mathbf{s}_{1:t+c})\psi_{t+c}(\mathbf{s}_{1:t+c})=p_{0}(\mathbf{s}_{1:t})\sum_{\mathbf{s}_{t+1:t+c}}p_{0}(\mathbf{s}_{t+1:t+c}|\mathbf{s}_{1:t})\psi_{t+c}(\mathbf{s}_{t+1:t+c}),
$$  

The normalization constant of   $\tilde{\pi}_{t+c}^{*}(\mathbf{s}_{1:t})$  can easily be seen to be   $\mathcal{Z}_{t+c}^{\psi^{*}}$   after summing over   $\mathbf{s}_{1:t}$   above, which yields   $\pi_{t+c}^{*}(\mathbf{s}_{1:t})\ =\ \tilde{\pi}_{t+c}^{*}(\mathbf{s}_{1:t})/\mathcal{Z}_{t+c}^{\psi^{*}}$ . We now factorize the   $c$ -step incremental importance weights (at step   $t+1$ , see  Eq. ( $c$ -Step SMC Weights) ) using the above identities, which imply that   $\tilde{\pi}_{t+c}^{*}(\mathbf{s}_{1:t+c})\ =\ \mathcal{Z}_{t+c}^{\psi^{*}}\pi_{t+c}^{*}(\mathbf{s}_{1:t+c})\ =$  Z $\mathcal{Z}_{t+c}^{\psi^{*}}\pi_{t+c}^{*}(\mathbf{s}_{1:t})\pi_{t+c}^{*}(\mathbf{s}_{t+1:t+c}|\mathbf{s}_{1:t})$  and  

$$
w_{t+1:t+c}({\mathbf{s}}_{1:t+c})=\frac{\tilde{\pi}_{t+c}^{*}({\mathbf{s}}_{1:t+c})}{\tilde{\pi}_{t}^{*}({\mathbf{s}}_{1:t})q^{*}({\mathbf{s}}_{t+1:t+c}\vert{\mathbf{s}}_{1:t})}=\frac{\mathcal{Z}_{t+c}^{\psi^{*}}}{\mathcal{Z}_{t}^{\psi^{*}}}\frac{\pi_{t+c}^{*}({\mathbf{s}}_{1:t})}{\pi_{t}^{*}({\mathbf{s}}_{1:t})}\frac{\pi_{t+c}^{*}({\mathbf{s}}_{t+1:t+c}\vert{\mathbf{s}}_{1:t})}{q^{*}({\mathbf{s}}_{t+1:t+c}\vert{\mathbf{s}}_{1:t})}
$$  

In order to have $w_{t+1:t+c}(\mathbf{s}_{1:t+c})=\mathcal{Z}_{t+c}^{\psi^{*}}/\mathcal{Z}_{t}^{\psi^{*}}$ Z in general, we thus require $\pi_{t+c}^{*}(\mathbf{s}_{1:t})=\pi_{t}^{*}(\mathbf{s}_{1:t})$  and $\pi_{t+c}^{*}(\mathbf{s}_{t+1:t+c}|\mathbf{s}_{1:t})=$ $q^{*}(\mathbf{s}_{t+1:t+c}|\mathbf{s}_{1:t})$  for all  t  and $c\leq T-t$ .  

(Sufficient)   $(i),(i i)\implies$ ⇒ Optimal Twisted SMC:  Consider the incremental importance weights using   $(i)$  and   $(i i)$ ,  

$$
w_{t}(\mathbf{s}_{1:t})=\frac{\tilde{\pi}_{t}^{*}\left(\mathbf{s}_{1:t}\right)}{\tilde{\pi}_{t-1}^{*}\left(\mathbf{s}_{1:t-1}\right)q_{t}^{*}\left(s_{t}\vert\mathbf{s}_{1:t-1}\right)}=\frac{\mathcal{Z}_{t}^{\psi}\sigma(\mathbf{s}_{1:t})}{\mathcal{Z}_{t-1}^{\psi}\sigma(\mathbf{s}_{1:t-1})\sigma(s_{t}\vert\mathbf{s}_{1:t-1})}=\frac{\mathcal{Z}_{t}^{\psi}}{\mathcal{Z}_{t-1}^{\psi}}
$$  

which matches the optimality definition in  Def. A.1  

Proposition A.4  ( Optimal Intermediate Target Distributions ) .  For a given target distribution $\sigma(\mathbf{s}_{1:T})$  ( Eq.  (31) ), the following conditions are equivalent, and are necessary for optimality of a twisted SMC procedure involving   $\{\pi_{t}^{*}\}_{t=1}^{T}$ } ,  

$$
\begin{array}{r l r l}{(i):}&{}&{\pi_{t}^{*}(\mathbf{s}_{1:t})=\displaystyle\sum_{s_{t+1}}\pi_{t+1}^{*}(\mathbf{s}_{1:t+1})}&{\quad}&{\forall\quad1\le t\le T-1\,,}\\ {(i i):}&{}&{\pi_{t}^{*}(\mathbf{s}_{1:t})=\displaystyle\sum_{\mathbf{s}_{t+1:t+c}}\pi_{t+c}^{*}(\mathbf{s}_{1:t+c})}&{\quad}&{\forall\quad1\le t\le T-1,\,1\le c\le T-t\,,}\\ {(i i i):}&{}&{\pi_{t}^{*}(\mathbf{s}_{1:t})=\sigma(\mathbf{s}_{1:t})}&{\quad}&{\forall\quad1\le t\le T\,.}\end{array}
$$  

Conditions   $(i)$  and   $(i i i)$  directly correspond to the recursions for the optimal twist functions given in  Prop. 3.2  and  Prop. B.1 .  

Proof.   $(i)\iff(i i)$ :  It is clear that   $(i i)\implies(i)$  as a special case for   $c=1$ . To show   $(i)\implies(i i)$ , we have  

$$
\pi_{t}^{*}(\mathbf{s}_{1:t})=\sum_{s_{t+1}}\pi_{t+1}^{*}(\mathbf{s}_{1:t+1})=\sum_{s_{t+1}}\sum_{s_{t+2}}\pi_{t+2}^{*}(\mathbf{s}_{1:t+2})=\ldots=\sum_{\mathbf{s}_{t+1:t+c}}\pi_{t+c}^{*}(\mathbf{s}_{1:t+c}).
$$  

$(i)\implies(i i i):]$ ⇒  Recursively applying   $(i)$  until time $T$  suggests that  

$$
\pi_{t}^{*}({\bf s}_{1:t})=\sum_{s_{t+1}}\pi_{t+1}^{*}({\bf s}_{1:t+1})=\sum_{s_{t+1}}\sum_{s_{t+2}}\pi_{t+2}^{*}({\bf s}_{1:t+2})=\ldots=\sum_{{\bf s}_{t+1:T}}\pi_{T}^{*}({\bf s}_{1:T})=\sum_{{\bf s}_{t+1:T}}\sigma({\bf s}_{1:T})=\sigma({\bf s}_{1:t}).
$$  

$(i i i)\implies(i):$ ⇒  The target marginals clearly satisfy the recursion  

$$
\sigma(\mathbf{s}_{1:t}):=\sum_{\mathbf{s}_{t+1:T}}\sigma(\mathbf{s}_{1:T})=\sum_{s_{t+1}}\sum_{\mathbf{s}_{t+2:T}}\sigma(\mathbf{s}_{1:T})=\sum_{s_{t+1}}\sigma(\mathbf{s}_{1:t+1}).
$$  

# A.2. Proof of Twist-Induced Proposal  

Proposition 3.3. (Twist-Induced Proposal).  For a given set of intermediate twisted targets $\pi_{t}(\mathbf{s}_{1:t})$  in  Eq.  (9) , the proposal which minimizes the variance of the one-step incremental importance weights $w_{t}$  is given by  

$$
\begin{array}{r l}&{q_{t}^{\pi}\bigl(s_{t}|\mathbf{s}_{1:t-1}\bigr)\propto\frac{\pi_{t}\bigl(\mathbf{s}_{1:t}\bigr)}{\pi_{t-1}\bigl(\mathbf{s}_{1:t-1}\bigr)}}\\ &{\qquad\qquad\qquad=\frac{1}{Z_{t}^{\pi}\bigl(\mathbf{s}_{1:t-1}\bigr)}p_{0}\bigl(s_{t}\bigl|\mathbf{s}_{1:t-1}\bigr)\psi_{t}\bigl(\mathbf{s}_{1:t}\bigr).}\end{array}
$$  

Proof.  We seek to minimize the variance of the resulting importance weights, subject to a constraint on the proposal probabilities summing to 1. Introducing a Lagrange multiplier   $\bf{\dot{\lambda}}(s_{1:t-1})$ , we have  

$$
\begin{array}{r}{\tilde{\mathbf{e}}_{q(s_{t}|\mathbf{s}_{1:t-1})}\bigg[\bigg(\frac{\tilde{\pi}_{t}(\mathbf{s}_{1:t})}{\tilde{\pi}_{t-1}(\mathbf{s}_{1:t-1})q(s_{t}|\mathbf{s}_{1:t-1})}\bigg)^{2}\bigg]-\bigg(\mathbb{E}_{q(s_{t}|\mathbf{s}_{1:t-1})}\bigg[\bigg(\frac{\tilde{\pi}_{t}(\mathbf{s}_{1:t})}{\tilde{\pi}_{t-1}(\mathbf{s}_{1:t-1})q(s_{t}|\mathbf{s}_{1:t-1})}\bigg)\bigg]\bigg)^{2}+\lambda(\mathbf{s}_{1:t-1})\bigg(\sum_{s_{t}}q(s_{t}|\mathbf{s}_{1:t-1})\bigg)^{2}\bigg],}\end{array}
$$  

Taking $\begin{array}{r}{\frac{\delta}{\delta q}(\cdot)=0}\end{array}$  implies  

$$
\begin{array}{r}{=\left(\frac{\tilde{\pi}_{t}\left(\mathbf{s}_{1:t}\right)}{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}\right)q\left(s_{t}\left\vert\mathbf{s}_{1:t-1}\right.\right)}\right)^{2}-2q(s_{t}\left\vert\mathbf{s}_{1:t-1}\right)\left(\frac{\tilde{\pi}_{t}\left(\mathbf{s}_{1:t}\right)}{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}\right)q\left(s_{t}\left\vert\mathbf{s}_{1:t-1}\right.\right)}\right)\frac{\tilde{\pi}_{t}\left(\mathbf{s}_{1:t}\right)}{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}\right)q\left(s_{t}\left\vert\mathbf{s}_{1:t-1}\right.\right)^{2}}+\lambda_{1}^{2}-\frac{\lambda_{2}}{\lambda_{3}}.}\end{array}
$$  

where the derivative in the second term is zero since the   $q(s_{t}|\mathbf{s}_{1:t-1})$  cancel. Finally, we have  

$$
\begin{array}{r l}{\frac{\tilde{\pi}_{t}\left(\mathbf{s}_{1:t}\right)^{2}}{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}\right)^{2}q\left(s_{t}|\mathbf{s}_{1:t-1}\right)^{2}}=\lambda(\mathbf{s}_{1:t-1})}&{}\\ {q^{*}(s_{t}|\mathbf{s}_{1:t-1})=\frac{1}{\sqrt{\lambda(\mathbf{s}_{1:t-1})}}\frac{\tilde{\pi}_{t}\left(\mathbf{s}_{1:t}\right)}{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}\right)}}&{=\frac{1}{Z_{t}^{\pi}\left(\mathbf{s}_{1:t-1}\right)}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}(\mathbf{s}_{1:t})}\end{array}
$$  

where   $Z_{t}^{\pi}(\mathbf{s}_{1:t-1})$  (or $\lambda$ ) is chosen to enforce normalization. −  

We focused on the one-step twist-induced proposal in  Prop. 3.3 . However, this proposal is  not optimal  for resampling every $c$  steps (as would also occur, for example, with adaptive resampling).  

Proposition A.5  ( Multi-Step Twist Induced Proposal (Generalization of  Prop. 3.3 ) ) .  For resampling $c$ -steps ahead, the optimal proposal (over $\mathbf{s}_{t+1:t+c-1,}$ ) which minimizes the variance of the importance weights   $w_{t:t+c-1}(\mathbf{s}_{1:t+c-1})$  is given by  

$$
q^{\pi}(\mathbf{s}_{t:t+c-1}|\mathbf{s}_{1:t-1})=\frac{p_{0}(\mathbf{s}_{t:t+c-1}|\mathbf{s}_{1:t-1})\psi_{t+c-1}(\mathbf{s}_{1:t+c-1})}{\sum_{\mathbf{s}_{t:t+c-1}}p_{0}(\mathbf{s}_{t:t+c-1}|\mathbf{s}_{1:t-1})\psi_{t+c-1}(\mathbf{s}_{1:t+c-1})}.
$$  

The proof follows the same reasoning as in the proof of  Prop. 3.3  above, using the multistep weights   $w_{t:t+c-1}(\mathbf{s}_{1:t+c-1})=$ $\frac{\tilde{\pi}_{t+c-1}\big(\mathbf{s}_{1:t+c-1}\big)}{\tilde{\pi}_{t-1}\big(\mathbf{s}_{1:t-1}\big)q\big(\mathbf{s}_{t:t+c-1}\big|\mathbf{s}_{1:t-1}\big)}$   from  Eq. ( $c$ -Step SMC Weights) . − − − | −  

Note that the denominator is not usually tractable for   $c>1$  in language modeling applications.  

# A.3. Derivation of CTL Gradient  

Lemma A.6  ( Derivation of CTL Gradient ) .  For the  CTL  loss   $\begin{array}{r}{\underset{\pmb{\theta}}{\mathrm{min}}\,\mathcal{L}_{C T L}(\pmb{\theta}){:=}\underset{\pmb{\theta}}{\mathrm{min}}\sum_{t=1}^{T}D_{\mathrm{KL}}\big(\sigma(\mathbf{s}_{1:t})\bigm|\bigm|\pi_{t}^{\pmb{\theta}}\big(\mathbf{s}_{1:t}\big)\big)}\end{array}$   P    , the (negative) gradient with respect to the parameters $\pmb{\theta}$  is given by  

$$
-\nabla_{\theta}\mathcal{L}_{C T L}(\theta)=\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t})}\Big[\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big]-\mathbb{E}_{\pi_{t}^{\theta}(\mathbf{s}_{1:t})}\Big[\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big]
$$  

Proof.  Consider expanding the form of $\pi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$  using  Eq. (9) , noting $\mathcal{Z}_{t}^{\psi}$   is independent of $\mathbf{s}_{1:t}$ . t Taking the gradient with respect to   $\pmb{\theta}$  using the log derivative identity  ∇ $\nabla_{\pmb{\theta}}f(\pmb{\theta})=f(\pmb{\theta})\nabla_{\pmb{\theta}}\log\breve{f}(\pmb{\theta})$ ∇ , we have  

$$
\begin{array}{r l}&{-\nabla_{\theta}\mathcal{L}_{\mathrm{CTL}}(\theta)=-\nabla_{\theta}\left(\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t})}\left[\log\sigma(\mathbf{s}_{1:t})-\log p_{0}(\mathbf{s}_{1:t})-\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\right]+\log\displaystyle\sum_{\mathbf{s}_{1:t}}p_{0}(\mathbf{s}_{1:t})\psi_{t}^{\theta}(\mathbf{s}_{1:t})\right)}\\ &{\qquad\qquad\qquad=\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t})}\Big[\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big]-\displaystyle\sum_{t=1}^{T}\sum_{\mathbf{s}_{1:t}}\frac{p_{0}(\mathbf{s}_{1:t})\psi_{t}^{\theta}(\mathbf{s}_{1:t})}{\sum_{\mathbf{s}_{1:t}}p_{0}(\mathbf{s}_{1:t})\psi_{t}^{\theta}(\mathbf{s}_{1:t})}\nabla_{\theta}\Big(\log p_{0}(\mathbf{s}_{1:t})+\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big)}\\ &{\qquad\qquad\qquad=\displaystyle\sum_{t=1}^{T}\biggl(\mathbb{E}_{\sigma(\mathbf{s}_{1:t})}\Big[\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big]-\mathbb{E}_{\pi_{t}^{\theta}(\mathbf{s}_{1:t})}\Big[\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big]\biggr)}\end{array}
$$  

# B. SMC with Intermediate Potentials and Connection with Soft Reinforcement Learning  

In the main text, we focused on settings where the target distribution is defined by a  potential $\phi(\mathbf{s}_{1:T})$  depending on full sequences only, as in  Eq. (1) . This setting highlights the need for (learned) twist functions to summarize the future expected value of the potential in the absence of intermediate target information.  

In this appendix, we generalize our exposition to show how our twisted SMC framework can accommodate settings with intermediate  potentials , which is evocative of connections with soft reinforcement learning ( Levine ,  2018 ). We leverage intuition from soft RL while introducing our general probabilistic interpretation, by using $(\stackrel{\mathrm{sRL})}{=}$  to instantiate the soft RL special case. In particular, soft RL will correspond to the terminal potential  

$$
\phi_{t}\big(\mathbf{s}_{1:t}\big)\overset{\mathrm{(sRL)}}{=}e^{\beta\ r_{t}\left(\mathbf{s}_{1:t}\right)}
$$  

which corresponds to   $\phi(\mathbf{s}_{1:T})=e^{\beta r_{T}(\mathbf{s}_{1:T})}$   if the  potential  is given at the final step only (as in RLHF,  Korbak et al.  ( 2022b )). However, we defer detailed discussion of soft RL to  App. B.3 . See  Table 5  for several examples of intermediate  potentials .  

Finally, we formalize a notion of conditional target distributions and twist functions in  App. B.2 , which generalizes the exposition in the main text and captures our conditional twist learning experiments in  Sec. 7.2.3 .  

# B.1. Twisted SMC with Intermediate Potentials  

To generalize the exposition in the main text, we might consider defining the target as  

$$
\sigma(\mathbf{s}_{1:T}):=\frac{1}{\mathcal{Z}_{\sigma}}p_{0}(\mathbf{s}_{1:T})\left(\prod_{t=1}^{T}\phi_{t}(\mathbf{s}_{1:t})\right)\overset{\scriptscriptstyle(\mathrm{sRL})}{=}\frac{1}{\mathcal{Z}_{\sigma}}p_{0}(\mathbf{s}_{1:T})e^{\beta\sum_{t=1}^{T}r_{t}(\mathbf{s}_{1:t})}
$$  

where  Eq. (1)  and the main text exposition corresponds to   $\phi_{t}\bigl(\mathbf{s}_{1:t}\bigr)=1$  for $t<T$ .  

Optimal Twists with Intermediate  Potentials Using  Eq. (31) , the marginal distribution $\begin{array}{r}{\sigma(\mathbf{s}_{1:t})=\sum_{\mathbf{s}_{t+1:T}}\sigma(\mathbf{s}_{1:T})}\end{array}$ P  over $t$  tokens becomes  

$$
\begin{array}{c}{\displaystyle\sigma(\mathbf{s}_{1:t})=\!\frac{1}{\mathcal{Z}_{\sigma}}p_{0}(\mathbf{s}_{1:t})\left(\prod_{\tau=1}^{t}\phi_{\tau}(\mathbf{s}_{1:\tau})\right)\left(\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\prod_{\tau=t+1}^{T}\phi_{\tau}(\mathbf{s}_{1:\tau})\right)}\\ {\displaystyle(\stackrel{\mathrm{sRL}}{=}\frac{1}{\mathcal{Z}_{\sigma}}p_{0}(\mathbf{s}_{1:t})e^{\beta\stackrel{\sum_{\tau=1}^{t}r_{\tau}(\mathbf{s}_{1:\tau})}{\sum_{\mathbf{s}_{t+1:T}}}}\left(\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})e^{\beta\stackrel{\sum_{\tau=t+1}^{T}r_{\tau}(\mathbf{s}_{1:\tau})}{\sum_{\tau=t+1}^{T}r_{\tau}(\mathbf{s}_{1:\tau})}}\right)}\end{array}
$$  

As in  Prop. 3.2 , the goal of the optimal twist functions is to facilitate sampling from the intermediate marginals $\sigma(\mathbf{s}_{1:t})$  of the target distribution $\sigma(\mathbf{s}_{1:T})$ .  

We consider two different quantities involved in defining the optimal twists, which differ in their treatment of the intermediate reward. For the soft RL setting, this corresponds to the natural distinction between   $Q$ -values and (soft) value functions $V_{t}$ .  

$$
\boldsymbol{\sigma}(\mathbf{s}_{1:t})=\!\frac{1}{\mathcal{Z}_{\sigma}}p_{0}(\mathbf{s}_{1:t})\left(\prod_{\tau=1}^{t-1}\phi_{\tau}(\mathbf{s}_{1:\tau})\right)\phi_{t}(\mathbf{s}_{1:t})\Big(\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\prod_{\tau=t+1}^{T}\phi_{\tau}(\mathbf{s}_{1:\tau})\Big)
$$  

$$
\stackrel{\mathrm{(sRL)}}{=}\frac{1}{\mathcal{Z}_{\sigma}}p_{0}(\mathbf{s}_{1:t})\left(e^{\beta\underset{\tau=1}{\overset{t-1}{\sum}}r_{\tau}(\mathbf{s}_{1:\tau})}\right)e^{\beta\underset{\tau=1}{\overset{t}{\sum}}p_{t}(\mathbf{s}_{1:t})}\bigg(\underbrace{\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})e^{\beta\underset{\tau=t+1}{\overset{T}{\sum}}r_{\tau}(\mathbf{s}_{1:\tau})}}_{\Phi_{t}^{\ast}(\mathbf{s}_{1:t}):\propto e^{\beta V_{t}^{\ast}(\mathbf{s}_{1:t})}=}\bigg)
$$  

where   $\propto$ means ‘defined to be proportional to’ and $Q_{t}^{*}(s_{t},\mathbf{s}_{1:t-1})=r_{t}(\mathbf{s}_{1:t})+V_{t}^{*}(\mathbf{s}_{1:t})$  in RL notation. See  App. B.3  for − detailed derivations in the soft RL special case. In general,   $\varPhi_{t}$  captures the expectation of  future  potentials  from   $t+1:T$ , analogous to the (soft) value function. The twists   $\psi_{t}$  play a role analogous to a   $Q$ -value, estimating both the immediate   $\phi_{t}$ and future value $\varPhi_{t}$ . In particular,  

$$
\psi_{t}^{*}(\mathbf{s}_{1:t})\propto\phi_{t}(\mathbf{s}_{1:t})\varPhi_{t}^{*}(\mathbf{s}_{1:t})\qquad\mathrm{where}\quad\varPhi_{t}^{*}(\mathbf{s}_{1:t}):\propto\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\prod_{\tau=t+1}^{T}\phi_{\tau}(\mathbf{s}_{1:\tau})
$$  

We continue to refer to $\psi_{t}$  as the  twist functions  and focus on probabilistic interpretations based on   $\psi_{t}$  instead of $\varPhi_{t}^{*}$   (see App. B.4  for additional discussion).  

To show that this notation is consistent with the main text, consider the optimal twists $\psi_{t}^{*}(\mathbf{s}_{1:t})=\phi_{t}(\mathbf{s}_{1:t})\varPhi_{t}^{*}(\mathbf{s}_{1:t})$  with no intermediate  potentials , $\phi_{t}(\mathbf{s}_{1:t})=1$  for   $t<T$ . For $t<T$ , $\psi_{t}^{*}(\mathbf{s}_{1:t})=\varPhi_{t}^{*}(\mathbf{s}_{1:t})$  reflect the future expected  potential  and for $t=T$ , the terminal potential is   $\psi_{T}^{*}(\mathbf{s}_{1:T})=\phi_{T}(\mathbf{s}_{1:T})$ , with no future potentials after step $T$ , i.e. $\varPhi_{T}=1$ .  

Building on  Eq. (32) -( 33 ) above, the following generalization of  Prop. 3.2  defines the ‘optimal’ twists so as to obtain the intermediate target marginals $\sigma(\mathbf{s}_{1:t})$  (see  Prop. A.4 ).  

Proposition B.1  ( Optimal Twists ) .  For a given target distribution   $\sigma(\mathbf{s}_{1:T})$  in  Eq.  (31) , the optimal twist functions yield intermediate $\{\pi_{t}\}_{t=1}^{T-1}$   which match the target marginals. In regions where   $p_{0}\bigl(\mathbf{s}_{1:t}\bigr)>0,$ , the optimal twists are given by  

$$
\mathbf{\Phi}_{1:t})=\sigma(\mathbf{s}_{1:t})=\frac{1}{\mathcal{Z}_{t}^{\psi^{\star}}}\ p_{0}(\mathbf{s}_{1:t})\left(\prod_{\tau=1}^{t-1}\phi_{\tau}(\mathbf{s}_{1:\tau})\right)\psi_{t}^{*}(\mathbf{s}_{1:t})\quad=\frac{1}{\mathcal{Z}_{t}^{\phi^{\star}}}\ p_{0}(\mathbf{s}_{1:t})\left(\prod_{\tau=1}^{t-1}\phi_{\tau}(\mathbf{s}_{1:\tau})\right)\phi_{t}(\mathbf{s}_{1:t})\Phi_{1:t}(\mathbf{s}_{1:t})
$$  

Up to a constant $c_{t}$  independent of $\mathbf{s}_{1:t}$ , the optimal twists $\psi_{t}^{*}$   are given by  

$$
\psi_{t}^{*}(\mathbf{s}_{1:t})=c_{t}\;\phi_{t}(\mathbf{s}_{1:t})\sum_{\mathbf{s}_{t+1:T}}p_{0}\big(\mathbf{s}_{t+1:T}\vert\mathbf{s}_{1:t}\big)\prod_{\tau=t+1}^{T}\phi_{\tau}\big(\mathbf{s}_{1:\tau}\big)
$$  

where   $c_{t}$  is absorbed into the normalization constant   $\boldsymbol{\mathcal{Z}}_{t}^{\boldsymbol{\psi}^{*}}$ . The optimal twists satisfy the recursion  

$$
\psi_{t}^{*}(\mathbf{s}_{1:t})=\frac{\mathcal{Z}_{t}^{\psi^{*}}}{\mathcal{Z}_{t+1}^{\psi^{*}}}\phi_{t}(\mathbf{s}_{1:t})\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})\psi_{t+1}^{*}(\mathbf{s}_{1:t+1}).
$$  

Remark B.2  ( Equivalence Class of   $\psi_{t}$  a $\varPhi_{t}$  Note that any rescaling of   $\psi_{t}\leftarrow c_{t}\bar{\psi}_{t}$  by a constant with respect to $\mathbf{s}_{1:t}$ will yield the same intermediate marginals $\pi_{t}(\mathbf{s}_{1:t})$ , due to the normalization  stant  Z $\mathcal{Z}_{t}^{\psi}$   which scales with $\psi_{t}$ . This defines an equivalent class in the space of functions. The same statement holds for $\varPhi_{t}$ . We express results such as  Eq. (36)  using proportionality   $\propto$ . We define $\psi_{t}$  and $\varPhi_{t}$  as the members of their equivalent classes whose normalization   $\mathcal{Z}_{t}^{\psi}$   and $\mathcal{Z}_{t}^{\phi}$   are equal. Thus, we have $\psi_{t}\big(\mathbf{s}_{1:t}\big)=\phi_{t}\big(\mathbf{s}_{1:t}\big)\varPhi_{t}\big(\mathbf{s}_{1:t}\big)$ .  

Proof.  Substituting  Eq. (36)  into  Eq. (35) , we obtain the desired marginal  Eq. (32) ,  

$$
\pi_{t}^{*}(\mathbf{s}_{1:t})=\frac{c_{t}}{\mathcal{Z}_{t}^{\psi^{*}}}\;p_{0}(\mathbf{s}_{1:t})\;\prod_{\tau=1}^{t}\phi_{\tau}(\mathbf{s}_{1:\tau})\left(\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\prod_{\tau=t+1}^{T}\phi_{\tau}(\mathbf{s}_{1:\tau})\right)=\sigma(\mathbf{s}_{1:t})
$$  

where the final equality follows from absorbing the constant $c_{t}$  into   $\boldsymbol{\mathcal{Z}}_{t}^{\boldsymbol{\psi}^{*}}$ , with $\begin{array}{r}{\frac{1}{\mathcal{Z}_{\sigma}}=\frac{c_{t}}{{\mathcal{Z}_{t}^{\psi}}^{*}}}\end{array}$ and   $\mathcal{Z}_{\sigma}$  which normalizes $\tilde{\sigma}(\mathbf{s}_{1:t})$ . We will now use $\begin{array}{r}{c_{t}=\frac{\mathcal{Z}_{t}^{\psi^{*}}}{\mathcal{Z}_{\sigma}}}\end{array}$   to show the recursion in  Eq. (37) . Note that  Eq. (36)  implies  

$$
\begin{array}{r l}&{\psi_{t}^{*}(\mathbf{s}_{1:t})=c_{t}\,\phi_{t}(\mathbf{s}_{1:t})\,\displaystyle\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})\underbrace{\biggl(\phi_{t+1}(\mathbf{s}_{1:t+1})\,\displaystyle\sum_{\mathbf{s}_{t+2:T}}p_{0}(\mathbf{s}_{t+2:T}|\mathbf{s}_{1:t+1})\,\prod_{\tau=t+2}^{T}\phi_{\tau}(\mathbf{s}_{1:\tau})\biggr)}_{\frac{1}{c_{t+1}}\,\psi_{t+1}^{*}(\mathbf{s}_{1:t+1})}}\\ &{\quad\quad=\frac{\mathcal{Z}_{t}^{\psi^{*}}}{\mathcal{Z}_{t+1}^{\psi^{*}}}\phi_{t}(\mathbf{s}_{1:t})\displaystyle\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})\psi_{t+1}^{*}(\mathbf{s}_{1:t+1})}\end{array}
$$  

where the second line follows from $\begin{array}{r}{\frac{c_{t}}{c_{t+1}}=\frac{{\mathcal{Z}_{t}^{\psi}}^{*}/{\mathcal{Z}_{\sigma}}}{{\mathcal{Z}_{t+1}^{\psi}}^{*}/{\mathcal{Z}_{\sigma}}}}\end{array}$  . This demonstrates  Eq. (37) .  

This leads to the following definition of the intermediate twisting targets (we defer the soft RL special case to  App. B.3 ).  

Definition B.3  ( Twisted Intermediate Targets  ) .  Using approximate twist functions   $\{\psi_{t}\}_{t=1}^{T-1}$   , we define the twisted intermediate target distributions  

$$
\pi_{t}(\mathbf{s}_{1:t})=\left\{\begin{array}{l l}{\displaystyle\frac{1}{\mathcal{Z}_{t}^{\psi}}\ p_{0}(\mathbf{s}_{1:t})\left(\prod_{\tau=1}^{t-1}\phi_{\tau}(\mathbf{s}_{1:\tau})\right)\,\psi_{t}(\mathbf{s}_{1:t})}&{\quad(t<T)}\\ {\displaystyle\frac{1}{\mathcal{Z}_{\sigma}}p_{0}(\mathbf{s}_{1:T})\prod_{t=1}^{T}\phi_{t}(\mathbf{s}_{1:t})}&{\quad(t=T)}\end{array}\right.
$$  

One-Step Twist-Induced Proposal Using  Prop. 3.3  and  Def. B.3  and noting that   $\phi_{t-1}(\mathbf{s}_{1:t-1})$  is independent of $s_{t}$ , we have the optimal one-step proposal  

$$
\begin{array}{r l}&{q_{t}^{\pi}(s_{t}|\mathbf{s}_{1:t-1})\propto\frac{\pi_{t}(\mathbf{s}_{1:t})}{\pi_{t-1}(\mathbf{s}_{1:t-1})}=\frac{\mathcal{Z}_{t-1}^{\psi}}{\mathcal{Z}_{t}^{\psi}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\frac{\phi_{t-1}(\mathbf{s}_{1:t-1})\psi_{t}(\mathbf{s}_{1:t})}{\psi_{t-1}(\mathbf{s}_{1:t-1})}}\\ &{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=:\frac{1}{Z_{t}^{\pi}(\mathbf{s}_{1:t-1})}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}(\mathbf{s}_{1:t})}\\ &{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=\frac{p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}(\mathbf{s}_{1:t})}{\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}(\mathbf{s}_{1:t})}}\end{array}
$$  

where in the second line, we absorb terms which depend only on $\mathbf{S}_{1:t-1}$  (and not $s_{t}$ ) into the normalization. In the soft RL special case, we have $q_{t}^{\pi}(s_{t}|\mathbf{s}_{1:t-1})\propto p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta Q_{t}(s_{t},\mathbf{s}_{1:t-1})}$ | −  ∝ | −  (see  Eq. (Twist-Induced Proposal (soft RL))  below).  

# B.2. Conditional Twisted SMC  

To formalize our notion of conditional twists in the infilling experiments ( Sec. 7.2.3 ), we generalize our above framework to explicitly depend on ‘observation’ random variables $\{o_{t}\}_{t=1}^{T}$ . This matches the common setting of SMC in state-space models ( Briers et al. ,  2010 ;  Gu et al. ,  2015 ;  Lawson et al. ,  2022 ;  Chopin et al. ,  2020 ). Our derivations in this section also emphasize that the optimal twist functions in  Prop. B.1  learn functions proportional to  conditional likelihoods  of the future observation variables given the current sequence (see  Eq. (40)  below)). We recover the unconditional targets in the main text for fixed   $o_{T}=1$ .  

Consider a target distribution   $\sigma(\mathbf{s}_{1:T}|o_{1:T})$  con servation random variables   $\pmb{o}_{1:T}:=\{o_{t}\}_{t=1}^{T}$ . We define a probabilistic model over observations $\sigma(o_{t}|\mathbf{s}_{1:t})=\phi_{t}(o_{t},\mathbf{s}_{1:t})$ |  as the intermediate  potential ,   which yields the target posterior  

$$
\sigma(\mathbf{s}_{1:T}|o_{1:T})=\frac{p_{0}(\mathbf{s}_{1:T})\left(\prod_{t=1}^{T}\sigma(o_{t}|\mathbf{s}_{1:t})\right)}{\sum_{\mathbf{s}_{1:T}}p_{0}(\mathbf{s}_{1:T})\left(\prod_{t=1}^{T}\sigma(o_{t}|\mathbf{s}_{1:t})\right)}=\frac{1}{\mathcal{Z}_{\sigma}(o_{1:T})}p_{0}(\mathbf{s}_{1:T})\left(\prod_{t=1}^{T}\phi_{t}(o_{t},\mathbf{s}_{1:t})\right)=\frac{p_{0}(\mathbf{s}_{1:T})\sigma(o_{1:T})}{\sigma(o_{1:T})}
$$  

where we interpret   $\begin{array}{r}{\sigma(o_{1:T}|\mathbf{s}_{1:T})=\prod_{t=1}^{T}\sigma(o_{t}|\mathbf{s}_{1:t})}\end{array}$  and   $\mathcal{Z}_{\sigma}(o_{1:T})=\sigma(\mathbf{s}_{1:T})$  to make the Bayesian posterior explicit in the last equality. Note, we now seek to estimate a different partition function   $\mathcal{Z}_{\sigma}(o_{1:T})$  for each set of observation variables.  

Using our infilling experiments in  Sec. 7.2.3  as an example, consider (a sequence of) subsequent tokens   ${\cal O}_{T}={\bf s}_{T+1:T+c}$  as observation variables, where the observation model is simply the base language model   $\sigma(o_{T}|\mathbf{s}_{1:T}):=p_{0}(\mathbf{s}_{T+1:T+c}|\mathbf{s}_{1:T})$ .  

Using  Eq. (38) , the intermediate marginals become  

$$
\begin{array}{r l}{\sigma(\mathbf{s}_{1:t}|\boldsymbol{o}_{1:T})=}&{\displaystyle\sum_{\mathbf{s}_{t+1:T}}\sigma(\mathbf{s}_{1:T}|\boldsymbol{o}_{1:T})}\\ &{\quad=\displaystyle\sum_{\mathbf{s}_{t+1:T}}\frac{1}{\sigma(\boldsymbol{o}_{1:T})}p_{0}(\mathbf{s}_{1:t})p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\biggl(\prod_{t=1}^{T}\sigma(\boldsymbol{o}_{t}|\mathbf{s}_{1:t})\biggr)}\\ &{\quad=\frac{1}{\mathcal{Z}_{\sigma}(\boldsymbol{o}_{1:T})}p_{0}(\mathbf{s}_{1:t})\biggl(\displaystyle\prod_{\tau=1}^{t}\phi_{\tau}(\boldsymbol{o}_{\tau},\mathbf{s}_{1:\tau})\biggr)\displaystyle\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\biggl(\prod_{\tau=t+1}^{T}\phi_{\tau}(\boldsymbol{o}_{\tau},\mathbf{s}_{1:\tau})\biggr)}\\ &{\quad=\frac{1}{\mathcal{Z}_{\sigma}(\boldsymbol{o}_{1:T})}p_{0}(\mathbf{s}_{1:t})\biggl(\displaystyle\prod_{\tau=1}^{t}\phi_{\tau}(\boldsymbol{o}_{\tau},\mathbf{s}_{1:\tau})\biggr)\sigma\bigl(\boldsymbol{o}_{t+1:T}|\mathbf{s}_{1:t}\bigr)\,,}\end{array}
$$  

noting that $\begin{array}{r}{\sigma(o_{t+1:T}|\mathbf{s}_{1:t})=\sum_{\mathbf{s}_{t+1:T}}\sigma(o_{t+1:T},\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\end{array}$  matches the second to last line.  

The optimal twists take a similar form as  Prop. B.1 , but now as a function of the future observation or conditioning information. Furt  the optimal twists is proportional to the c nal likelihoods (e.g.,   $\sigma(o_{t+1:T}|\mathbf{s}_{1:t}))$  of future observations given $\mathbf{s}_{1:t}$ , which marginalize over future tokens (e.g., $\mathbf{s}_{t+1:T}.$  ),  

$$
\begin{array}{c}{{\displaystyle\varPhi_{t}^{*}\big({\bf s}_{1:t},o_{t+1:T}\big)\overset{o_{t+1:T}}{\propto}\sigma\big(o_{t+1:T}\vert{\bf s}_{1:t}\big)=\sum_{{\bf s}_{t+1:T}}p_{0}\big({\bf s}_{t+1:T}\vert{\bf s}_{1:t}\big)\Big(\prod_{\tau=t+1}^{T}\phi_{\tau}\big(o_{\tau},{\bf s}_{1:\tau}\big)\Big)\,,}}\\ {{\displaystyle\psi_{t}^{*}\big({\bf s}_{1:t},o_{t:T}\big)\overset{o_{t:T}}{\propto}\sigma\big(o_{t:T}\vert{\bf s}_{1:t}\big)=\sum_{{\bf s}_{t+1:T}}p_{0}\big({\bf s}_{t+1:T}\vert{\bf s}_{1:t}\big)\Big(\prod_{\tau=t}^{T}\phi_{\tau}\big(o_{\tau},{\bf s}_{1:\tau}\big)\Big)\,,}}\end{array}
$$  

where   $f(x,o)\stackrel{o}{\propto}g(x,o)$ ∝  denotes proportionality up to a constant which depends on   $^o$  only:   $\exists c(o)\colon f(x,o)=c(o)g(x,o)$ These equations can be confirmed by comparing  Prop. B.1  with the last two lines in  Eq. (39) .  

The intermediate marginals over partial sequences can finally be rewritten as either  

$$
\begin{array}{r l}&{\sigma\big(\mathbf{s}_{1:t}\big|\pmb{o}_{1:T}\big)\overset{\scriptscriptstyle{o_{1:T}}}{\propto}p_{0}\big(\mathbf{s}_{1:t}\big)\Bigg(\displaystyle\prod_{\tau=1}^{t}\phi_{\tau}\big(o_{\tau},\mathbf{s}_{1:\tau}\big)\Bigg)\varPhi_{t}^{*}\big(\mathbf{s}_{1:t},\pmb{o}_{t+1:T}\big)\,,}\\ &{\qquad\qquad\qquad=p_{0}\big(\mathbf{s}_{1:t}\big)\Bigg(\displaystyle\prod_{t=1}^{t-1}\phi_{\tau}\big(o_{\tau},\mathbf{s}_{1:\tau}\big)\Bigg)\psi_{t}^{*}\big(\mathbf{s}_{1:t},\pmb{o}_{t:T}\big)\,.}\end{array}
$$  

We discuss the choice of parameter iz ation using   $\psi_{t}$  versus $\varPhi_{t}$  in  App. B.4  

The conditional twist learning formulation matches the setting of  Lawson et al.  ( 2022 ), to which we refer the reader for additional discussion. We use this conditional perspective to derive classification losses for twist learning in  App. C.3 - C.4 .  

Unconditional Targets as a Special Case In cases where we are only learning twists for a single set of conditioning information such as a single classifier label or a reward model, note that we can omit explicit conditioning information in

 $\psi_{t}(\mathbf{s}_{1:t},o_{t})$  and consider setting   $\{o_{t}=1\}_{t=1}^{T}$ . With terminal  potential  only as in the main text, we write   $\sigma(o_{T}=1|\mathbf{s}_{1:T})=

$ $\phi(\mathbf{s}_{1:T})$  and the overall target distribution as   $\sigma({\bf s}_{1:T})=\sigma({\bf s}_{1:T}|o_{T}=1)\propto p_{0}({\bf s}_{1:T})\phi_{T}({\bf s}_{1:T})$ . Thus, the formulation in Eq. (38) - Eq. (40)  strictly generalizes our exposition in the main text and  App. B.1 . With intermediate  potentials , we set $\begin{array}{r}{\dot{\sigma(o_{1:T}=\mathbf{1}|\mathbf{s}_{1:T})}=\prod_{t=1}^{T}\dot{\phi_{t}}(\mathbf{s}_{1:t})}\end{array}$ .  

Our notation also matches the exposition in  Levine  ( 2018 ) for the soft RL case with a binary observation or ‘optimality’ random variable $\sigma(o_{t}=1|\mathbf{s}_{1:t-1},s_{t})=e^{\beta r_{t}(\mathbf{s}_{1:t-1},s_{t})}$ | , where the reward is a function of the state $x_{t}=\mathbf{s}_{1:t-1}$  and action − $a_{t}=s_{t}$  pair (see the MDP interpretation in  App. B.3 ).  

# B.3. Connection with Soft Reinforcement Learning  

In this section, we more explicitly describe the soft reinforcement learning setting ( Levine ,  2018 ) commonly used in RLHF ( Korbak et al. ,  2022b ) as a special case of our probabilistic framework. Again, we use notation $(\stackrel{\mathrm{sRL})}{=}$  to indicate that the expressions in this section correspond to a particular instance of our SMC framework where   $\phi(\mathbf{s}_{1:T})=e^{\beta r(\mathbf{s}_{1:T})}$ .  

Summary of Soft RL Notation To summarize the below derivations, we state the relevant assignments for the soft RL case. We focus on the optimal case for simplicity, but note that approximate versions play identical roles,  

$$
=e^{\beta\ r_{t}(\mathbf{s}_{1:t})}\quad\quad\psi_{t}^{*}(\mathbf{s}_{1:t})=e^{\beta r_{t}(\mathbf{s}_{1:t})+\beta V_{t}^{*}(\mathbf{s}_{1:t})}=e^{\beta Q_{t}^{*}(s_{t},\mathbf{s}_{1:t-1})}\quad\quad\varPhi_{t}^{*}(\mathbf{s}_{1:t})=e^{\beta V_{t}^{*}(\mathbf{s}_{1:t})}\mathrm{~(T w i s t\to~\mu~)~},
$$  

where   $\psi_{t}^{*}(\mathbf{s}_{1:t})=\phi_{t}(\mathbf{s}_{1:t})\varPhi_{t}^{*}(\mathbf{s}_{1:t})$  or $Q_{t}^{*}(s_{t},\mathbf{s}_{1:t-1})=r_{t}(\mathbf{s}_{1:t})+V_{t}^{*}(\mathbf{s}_{1:t})$ . In the other direction, we have −  

$$
r_{t}(\mathbf{s}_{1:t})=\frac{1}{\beta}\log\phi_{t}(\mathbf{s}_{1:t})\qquad Q_{t}^{*}(s_{t},\mathbf{s}_{1:t-1})=\frac{1}{\beta}\log\psi_{t}^{*}(\mathbf{s}_{1:t})\qquad V_{t}^{*}(\mathbf{s}_{1:t})=\frac{1}{\beta}\log\varPhi_{t}^{*}(\mathbf{s}_{1:t})
$$  

MDP Interpretation To draw connections with soft RL, we view language model controlled decoding as a MDP, where the prom wn from an initial state d $\mathbf{s}_{0}\,\sim\,\nu_{0}$ , an action policy   $\pi(a_{t}|x_{t})\,=\,q(s_{t}|\mathbf{s}_{1:t-1})$  selects the next token $a_{t}~=~s_{t}$  given a partial sequence $x_{t}\;=\;\mathbf{s}_{1:t-1}$  as the state, and deterministic environment transitions − $P(x_{t+1}\ =\ \mathbf{s}_{1:t}|a_{t}\ =\ s_{t},x_{t}\ =\ \mathbf{s}_{1:t-1})\ =\ \delta(x_{t}\ =\ \mathrm{{const}}(s_{t},\mathbf{s}_{1:t-1}))$  append the selected token to update the state. Discounting may also be included without difficulty. The reward is given by $r_{t}(\mathbf{s}_{1:t})$ .  

Final Target Distribution We define the target distribution as the solution to the following variational optimization which solves the regularized MDP described above,  

$$
\sigma(\mathbf{s}_{1:T})\stackrel{(\mathbf{s}_{1:T})}{=}\frac{1}{\mathcal{Z}_{\sigma}}p_{0}(\mathbf{s}_{1:T})e^{\beta\underset{t=1}{\overset{T}{\sum}}r_{t}(\mathbf{s}_{1:t})}=\underset{q(\mathbf{s}_{1:T})}{\arg\operatorname*{max}}\,\mathbb{E}_{q(\mathbf{s}_{1:T})}\Big[\underset{t=1}{\overset{T}{\sum}}r_{t}(\mathbf{s}_{1:t})\Big]-\frac{1}{\beta}D_{\mathrm{KL}}(q(\mathbf{s}_{1:T})\,||\,p_{0}(\mathbf{s}_{1:T}))
$$  

which corresponds to the choice $\phi_{t}\bigl(\mathbf{s}_{1:t}\bigr)=e^{\beta\ r_{t}\left(\mathbf{s}_{1:t}\right)}$   as in  Eq. (Twist to Soft RL) . The soft value is defined as the maximum value of the above optimization for optimal $q^{*}(\mathbf{s}_{1:T})$ , and corresponds to the scaled log partition function  

$$
:=\frac{1}{\beta}\log\mathcal{Z}_{\sigma}=\frac{1}{\beta}\log\sum_{\mathbf{s}_{1:T}}p_{0}(\mathbf{s}_{1:T})e^{\beta\sum_{t=1}^{T}r_{t}(\mathbf{s}_{1:t})}=\operatorname*{max}_{q(\mathbf{s}_{1:T})}\mathbb{E}_{q(\mathbf{s}_{1:T})}\Big[\sum_{t=1}^{T}r_{t}(\mathbf{s}_{1:t})\Big]-\frac{1}{\beta}D_{\mathrm{KL}}(q(\mathbf{s}_{1:T})\,\|\,p_{0}(\mathbf{s}_{1:T})).
$$  

which can be confirmed by substituting   $q(\mathbf{s}_{1:T})=\sigma(\mathbf{s}_{1:T})$  from  Eq. (42)  into the maximization on the right side of  Eq. (43) . Although we omit the dependence of   $\mathcal{Z}_{\sigma}(\mathbf{s}_{0})$  on the prompt   $\mathbf{S}_{0}$  for notational simplicity (see  Eq. (1) ), note that $V_{0}^{\ast}:=V^{\ast}(\mathbf{s}_{0})$ naturally corresponds to the soft value of the prompt as the initial state in the MDP.  

Optimal Intermediate Marginals and Soft Value Decomposing the maximization in  Eq. (43)  into optimizations over each $q(s_{t+1}|\mathbf{s}_{1:t})$ , we define the intermediate soft value $V_{t}^{*}(\mathbf{s}_{1:t})$  as the maximum of the expected future regularized reward  

$$
\begin{array}{r l r}{\lefteqn{=\frac{1}{\beta}\log\phi_{t}^{*}(\mathbf{s}_{1:t})\stackrel{\mathrm{(SR)}}{=}\frac{1}{\beta}\log\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\ e^{\beta\sum_{\tau=t+1}^{T}r_{\tau}(s_{1:\tau})}}}&{}&{\mathrm{(Optimal~Intensity~sf~Sof~t~)}}\\ &{}&{=\underset{q(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}{\operatorname*{max}}\ \mathbb{E}_{q(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\Big[\sum_{\tau=t+1}^{T}r_{\tau}(\mathbf{s}_{1:\tau})\Big]-\frac{1}{\beta}D_{\mathrm{KL}}(q(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\,||\,p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t}))}\\ &{}&{=\underset{q(\mathbf{s}_{t+1}|\mathbf{s}_{1:t})}{\operatorname*{max}}\ \mathbb{E}_{q(\mathbf{s}_{t+1}|\mathbf{s}_{1:t})}\Big[r_{t+1}(\mathbf{s}_{1:t+1})+V_{t+1}^{*}(\mathbf{s}_{1:t+1})\Big]-\frac{1}{\beta}D_{\mathrm{KL}}(q(\mathbf{s}_{t+1}|\mathbf{s}_{1:t})\,||\,p_{0}(\mathbf{s}_{t+1}|\mathbf{s}_{1:t})).}\end{array}
$$  

where, in the third line, we isolate the optimization over $q(s_{t}|\mathbf{s}_{1:t-1})$  by (i) assuming optimality at $\tau<t$  and (ii) substituting the optimal value   $V_{t+1}^{*}(\mathbf{s}_{1:t+1})\;=\;\operatorname*{max}_{q(\mathbf{s}_{t+2:T}\mid\mathbf{s}_{1:t+1})}[\ldots]$  of the maximization over   $q(\mathbf{s}_{t+2:T}|\mathbf{s}_{1:t+1})$  (i.e. recursively  | applying the second line).  

The optimal intermediate marginal can be written using either   $V_{t}^{*}(\mathbf{s}_{1:t})$  or   $Q_{t}^{*}\big(s_{t},\mathbf{s}_{1:t-1}\big)$  form (as in  Eq. (33)  above, or by − substituting the optimal   $V_{t}^{\ast}$   or   $Q_{t}^{*}$   into the twist targets below).  

Twisted Intermediate Targets We state the approximate twisting targets for  both $V_{t}$  or   $Q_{t}$  parameter iz at ions in order to make connections with soft RL losses in  App. C . For approximate   $V_{t}(\mathbf{s}_{1:t})$  or   $Q_{t}\big(s_{t},\mathbf{s}_{1:t-1}\big)$ , we have  

$$
\begin{array}{r l r}{\pi_{t}(\mathbf{s}_{1:t})\stackrel{\mathrm{(sRL)}}{=}\displaystyle\frac{1}{\mathcal{Z}_{t}^{V}}p_{0}(\mathbf{s}_{1:t})e^{\beta\displaystyle\sum_{\tau=1}^{t-1}r_{\tau}(\mathbf{s}_{1:\tau})}e^{\beta r_{t}(\mathbf{s}_{1:t})+\beta V_{t}(\mathbf{s}_{1:t})}}&{\qquad(t<T)}\\ {=\displaystyle\frac{1}{\mathcal{Z}_{t}^{Q}}p_{0}(\mathbf{s}_{1:t})e^{\beta\displaystyle\sum_{\tau=1}^{t-1}r_{\tau}(\mathbf{s}_{1:\tau})}e^{\beta Q_{t}(s_{t},\mathbf{s}_{1:t-1})}}&{\qquad(t<T)}\end{array}
$$  

where the final twisting target is given by  Eq. (42)  and the optimal   $Q$ -values are defined as  

$$
Q_{t}^{*}(s_{t},\mathbf{s}_{1:t-1})=r_{t}(\mathbf{s}_{1:t})+V_{t}^{*}(\mathbf{s}_{1:t})
$$  

One-Step Proposal Finally, the optimal one-step proposal (e.g. in $V_{t}$  form) can be derived either (i) as the twist-induced proposal from  Eq. (Twist Targets (Soft RL V) )  and  Prop. B.1  or (ii) as the solution to the one-step optimization in the third line of  Eq. (Optimal Intermediate Soft Value) . As in  Eq. (Twist-Induced Proposal  $(\psi)$  ) ,  

$$
q_{t}^{\pi}(s_{t}|\mathbf{s}_{1:t-1})\overset{\mathrm{(sRL)}}{=}\frac{p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta(r_{t}(\mathbf{s}_{1:t})+V_{t}(\mathbf{s}_{1:t}))}}{\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta(r_{t}(\mathbf{s}_{1:t})+V_{t}(\mathbf{s}_{1:t}))}}\propto p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta Q_{t}(s_{t},\mathbf{s}_{1:t-1})}.
$$  

(Twist-Induced Proposal (soft RL))  

We define the one-step log normalization constant induced by an approximate   $V_{t}$  or   $Q_{t}$  as   $V_{V_{t}}$  or   $V_{Q_{t}}$ , respectively,  

$$
V_{t}(\mathbf{s}_{1:t-1}):=\frac{1}{\beta}\log\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta(r_{t}(\mathbf{s}_{1:t})+V_{t}(\mathbf{s}_{1:t}))}\qquad V_{Q_{t}}(\mathbf{s}_{1:t-1}):=\frac{1}{\beta}\log\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})
$$  

such that, for example, $q_{t}^{\pi}(s_{t}|\mathbf{s}_{1:t-1})=p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta Q_{t}(s_{t},\mathbf{s}_{1:t-1})-\beta V_{Q_{t}}(\mathbf{s}_{1:t-1})}.$  

RLHF Minimizes   $D_{\mathbf{KL}}(q\parallel\sigma)$ Note that, for a given suboptimal   $q(\mathbf{s}_{1:T})$ , the value of the variational optimization in Eq. (42)  is a lower bound on the (scaled) log partition function $\begin{array}{r}{V_{0}^{*}=\frac{1}{\beta}\log\mathcal{Z}_{\sigma}}\end{array}$ . Similarly to the standard Evidence Lower Bound, the gap in this lower bound is given by the KL divergence  

$$
\frac{1}{\beta}\log\mathcal{Z}_{\sigma}=\underbrace{\frac{1}{\beta}D_{\mathrm{KL}}(q(\mathbf{s}_{1:T})\,||\,\sigma(\mathbf{s}_{1:T}))}_{\mathrm{ELBO}\,\mathrm{gap}\,(\geq0)}+\biggl(\mathbb{E}_{q(\mathbf{s}_{1:T})}\Big[\sum_{t=1}^{T}r_{t}(\mathbf{s}_{1:t})\Big]-\frac{1}{\beta}D_{\mathrm{KL}}(q(\mathbf{s}_{1:T})\,||\,p_{0}(\mathbf{s}_{1:T}))\biggr)
$$  

In this sense, we consider soft RL or policy gradient methods such as PPO which optimize  Eq. (42)  as targeting $\sigma(\mathbf{s}_{1:T})$  by minimizing   $D_{\mathrm{KL}}\big(q(\mathbf{s}_{1:T})\,\|\,\sigma(\mathbf{s}_{1:T})\big)$  ( Korbak et al. ,  2022b ).  

# B.4. Remarks on Parameter iz ation  

While the twisting targets ( Eq. (Twist Targets $(\psi)$  ) ) and twist-induced proposal ( Eq. (Twist-Induced Proposal  $(\psi)$  ) ) may equivalently be parameterized using approximate   $\varPhi_{t}$ , we focus on the $\psi_{t}$  parameter iz ation to match the main text. In particular, recall that the optimal twists satisfy   $\psi_{t}^{*}(\mathbf{s}_{1:t})=\phi_{t}(\mathbf{s}_{1:t})\varPhi_{t}^{*}(\mathbf{s}_{1:t})$  for all $t$ . With no intermediate  potential   $\phi_{t}=1$ for   $t<T$ ), our approximate twists estimate   $\begin{array}{r}{\psi_{t}(\mathbf{s}_{1:t})\approx\varPhi_{t}^{*}(\mathbf{s}_{1:t})\propto\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi_{T}(\mathbf{s}_{1:T})}\end{array}$ P  for   $t<T$ . In this section, we describe how the presence of intermediate potentials may affect the choice of twist parameter iz ation.  

The twist-induced proposal may not be tractable to evaluate at the final timestep, since it may be costly to evaluate the terminal   $\phi_{T}\big(\mathbf{s}_{1:T}\big)$ r all   $s_{T}\,\in\,\mathcal{V}$  given a context $\mathbf{S}_{1:T-1}$  (as described in  Sec. 2 ). Thus, we learn an $\psi_{T}(\mathbf{s}_{1:T})\approx\phi_{T}(\mathbf{s}_{1:T})$  ≈   osal sampling, which can be easily evaluated over  |V|  next tokens. The final $\pi_{T}(\mathbf{s}_{1:T})=\sigma(\mathbf{s}_{1:T})$  is defined using $\phi(\mathbf{s}_{1:T})$  in order to preserve unbiased estimation. However, after sampling the proposal according to   $\psi_{T}$  , we only need to evaluate $\phi(\mathbf{s}_{1:T})$  over $K$  full sequences to calculate the importance weights at the final step ( Eq. (16) ). See  Intermediate Potential Tractable over   $K$  Sequences Only  paragraph below.  

Intermediate  Potentials  Tractable over   $|\mathcal{V}|$  Sequences However, in settings where the intermediate  potentials   $\phi_{t}(\mathbf{s}_{1:t})$ are  tractable to calculate for all $s_{t}\in\mathcal V$  given $\mathbf{S}_{1:t-1}$  (e.g. using an indicator function or forward pass in a transformer architecture, as in  Table 5 ), it may be useful to use a $\varPhi_{t}$  parameter iz ation of the twist targets and twist-induced proposal. This allows us to use the  exact  immediate  potentials $\phi_{t}\big(\mathbf{s}_{1:t}\big)$  alongside an estimated $\varPhi_{t}^{\theta}$   , instead of an approximate $\psi_{t}^{\pmb\theta}\approx\phi_{t}\varPhi_{t}^{*}$   ≈ which estimates both the immediate $\phi_{t}$  and future expected value of  potentials $\varPhi_{t}^{*}$   . Using notation established in  Eq. (33) and  Prop. B.1 , the twisting targets in  Eq. (Twist Targets  $(\psi)$  )  can be rewritten using a   $\varPhi_{t}^{\theta}$   parameter iz ation  

$$
\pi_{t}^{\theta}(\mathbf{s}_{1:t})=\frac{1}{\mathcal{Z}_{t}^{\psi}}\ p_{0}(\mathbf{s}_{1:t})\left(\prod_{\tau=1}^{t-1}\phi_{\tau}(\mathbf{s}_{1:\tau})\right)\,\phi_{t}(\mathbf{s}_{1:t})\varPhi_{t}^{\theta}(\mathbf{s}_{1:t})\qquad(t<T)
$$  

with   $\pi_{T}(\mathbf{s}_{1:T})\,=\,\sigma(\mathbf{s}_{1:T})$  as before. The twist-induced proposal $q_{t}^{\pi}(s_{t}|\mathbf{s}_{1:t-1})\,\propto\,p_{0}(s_{t}|\mathbf{s}_{1:t-1})\phi_{t}(\mathbf{s}_{1:t})\varPhi_{t}^{\theta}(\mathbf{s}_{1:t})$ |  ∝ |  and its − − normalization constant are tractable in this case, by evaluating both the given $\phi_{t}\big(\mathbf{s}_{1:t}\big)$  and parameterized $\varPhi_{t}^{\theta}(\mathbf{s}_{1:t})$  in a single forward pass and normalizing over the discrete vocabulary of next tokens.  

Intermediate  Potentials  Tractable over $K$  Sequences Only In cases where the intermediate  potentials  are difficult to evaluate, we would like to limit evaluation of   $\phi_{t}(\mathbf{s}_{1:t})$  to only   $K$  partial sequences. In this case, parameterizing the twisted targets $\pi_{t}$  using $\psi_{t}^{\theta}$   or $Q_{t}^{\theta}$   ( Eq. (Twist Targets $(\psi)$  ) ,  Eq. (Twist Targets (Soft RL Q) ) ) instead of   $\varPhi_{t}^{\theta}$   or   $V_{t}^{\theta}$   may be preferable to ensure a tractable twist-induced proposal. Separate parameter iz at ions of the proposal (using $\psi_{t}^{\pmb{\xi}}$ ) and targets $(\phi_{t}\varPhi_{t}^{\theta})$ ) might also be considered.  

In the case of the final timestep described above or in  Sec. 3.2 , note that we use a learned $\psi_{T}^{\pmb{\xi}}$   to parameterize a tractable variational propo $q_{T}\big(s_{T}|\mathbf{s}_{1:T-1}\big)$ . In this case, we have no fut  value   $\varPhi_{T}(\mathbf{s}_{1:T})=1$  and only need to evaluate the terminal potential $\phi(\mathbf{s}_{1:T})$  for calculating importance weights over $K$  sequences.  

# C. Twist Learning Losses  

In this section, we describe various methods for twist learning beyond our proposed contrastive twist learning (CTL) procedure from  Sec. 4 . In  App. C.1 , we first describe several losses from the soft RL literature from a probabilistic perspective, building closely on our developments in  App. B.1 . We then proceed to describe SIXO ( Lawson et al. ,  2022 ) and FUDGE ( Yang & Klein ,  2021 ) in  App. C.3 - C.4 .  

We emphasize losses found in related work or used as experimental baselines using equation tags (e.g.  Eq. (SIXO) ), where equations  Eq. (RL Baseline) ,  Eq. (SIXO) ,  Eq. (FUDGE)  are used in our experiments. We consider settings with intermediate potentials in  App. C.1 , but focus on the ( $\phi_{t}=1$  for   $t<T$ ) setting in the remainder of the section, as in the main text.  

# C.1. Soft Q-Learning (RL) and Path Consistency Losses from Log Importance Weights  

From the probabilistic perspective of the SMC log importance weights, we can derive several losses for twist learning, including soft Q-learning and path consistency learning (PCL) ( Nachum et al. ,  2017 ) losses from the soft RL literature.  

A general principle for deriving loss functions would be to minimize the variance of the (log) importance weights under some sampling distribution $\pi_{s}$ , which leads to constant importance weights at optimality. To draw connections with previous work, we also consider minimizing the square of the log weights, which at optimality, ensures that   $\log w=0$  and $w=1$  are equal to a  particular  constant. We will proceed to parameterize the twist functions using parameters   $\pmb{\theta}$  and consider loss terms which minimize the variance or square of $c$ -step log weights at time $t$ ,  

$$
\mathcal{L}_{\log\mathrm{Var}}^{(t,c)}(\pmb{\theta}):=\mathrm{Var}_{\pi_{s}}\bigg[\sum_{\tau=t}^{t+c-1}\log w_{\tau}(\mathbf{s}_{1:\tau})\bigg]\qquad\qquad\mathcal{L}_{\log\mathrm{cons}}^{(t,c)}(\pmb{\theta}):=\mathbb{E}_{\pi_{s}}\bigg[\bigg(\sum_{\tau=t}^{t+c-1}\log w_{\tau}(\mathbf{s}_{1:\tau})\bigg)^{2}\bigg].
$$  

$\mathcal{L}_{\mathrm{log\,cons}}^{(t,c)}(\pmb{\theta})$  indicates ‘consistency’ in  log -weight space for   $c$ -step-ahead weights at time   $t$  (see  Eq. ( $c$ -Step SMC Weights) ). We will consider various choices of parameter iz ation and proposal in the following subsections. For example, let $\mathcal{L}_{\log\mathrm{cons}}^{(t,c)}(\pmb{\theta};\{\psi_{t},q_{t}^{\pi}\})$     ote the log-consistency loss correspondin ting targets parameterized by   $\psi_{t}^{\theta}$   and the twist induced proposal $q_{t}^{\pi}$   (note, our notation for the one-step weights $w_{t}(\mathbf{s}_{1:t})$  does not make these choices explicit).  

For reference, we derive the log importance weights with intermediate potentials and arbitrary   $q$  as  

$$
\log w_{t}(\mathbf{s}_{1:t})=\log\frac{\tilde{\pi}_{t}\bigl(\mathbf{s}_{1:t}\bigr)}{\tilde{\pi}_{t-1}\bigl(\mathbf{s}_{1:t-1}\bigr)q\bigl(s_{t}\bigl|\mathbf{s}_{1:t-1}\bigr)}=\log\frac{p_{0}\bigl(\mathbf{s}_{1:t}\bigr)\biggl(\prod_{r=1}^{t-1}\phi_{\tau}\bigl(\mathbf{s}_{1:\tau}\bigr)\biggr)\,\psi_{t}\bigl(\mathbf{s}_{1:t}\bigr)}{p_{0}\bigl(\mathbf{s}_{1:t-1}\bigr)\biggl(\prod_{r=1}^{t-2}\phi_{\tau}\bigl(\mathbf{s}_{1:\tau}\bigr)\biggr)\,\psi_{t-1}\bigl(\mathbf{s}_{1:t-1}\bigr)q\bigl(s_{t}\bigl|\mathbf{s}_{1:t-1}\bigr)}
$$  

$$
\implies\quad\log w_{t}(\mathbf{s}_{1:t})=\log\phi_{t-1}(\mathbf{s}_{1:t-1})+\log\psi_{t}(\mathbf{s}_{1:t})-\log\psi_{t-1}(\mathbf{s}_{1:t-1})-\log\frac{q(s_{t}|\mathbf{s}_{1:t-1})}{p_{0}(s_{t}|\mathbf{s}_{1:t-1})}
$$  

Various special cases arise from choices of twist parameter iz at ions and proposals in the following subsections.  

# C.1.1. S OFT  Q-L EARNING AND  RL B ASELINE  

For single-step log-weights, the   $\psi$ - parameter iz ation  of the targets ( Eq. (Twist Targets $(\psi)$  ) ,  Eq. (Twist Targets (Soft RL Q) ) ), and the  twist-induced proposal  ( Eq. (Twist-Induced Proposal  $(\psi)$  ) ,  Eq. (Twist-Induced Proposal (soft RL)) ), we have  

$$
\begin{array}{r l r}{\lefteqn{\mathbf{s}_{1:t})=\log\phi_{t-1}(\mathbf{s}_{1:t-1})+\log\psi_{t}(\mathbf{s}_{1:t})-\log\psi_{t-1}(\mathbf{s}_{1:t-1})-\left(\log\frac{p_{0}(s_{t}|\mathbf{s}_{1:t-1})}{\widetilde{p_{0}}(s_{t}|\mathbf{s}_{1:t-1})}+\log\psi_{t}(\mathbf{s}_{1:t})-\log\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\right)}}\\ &{}&{=\log\phi_{t-1}(\mathbf{s}_{1:t-1})+\log\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}(\mathbf{s}_{1:t})-\log\psi_{t-1}(\mathbf{s}_{1:t-1})~~~~~~}\end{array}
$$  

where the second term $\begin{array}{r}{\log Z_{t}^{\pi}(\mathbf{s}_{1:t-1})=\log\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}(\mathbf{s}_{1:t})}\end{array}$  normalizes the twist-induced proposal ( Eq. (14) ).  

Minimizing the sum of  one-step log consistency losses  (i.e. squared log weights in  Eq. (48) ) will yield the familiar soft $Q$ -learning loss (e.g.  Lioutas et al.  ( 2022 ) Eq. (4)-(5)). Adjusting indexing from  Eq. (48)  and introducing a stop-gradient within   $\log Z_{t}^{\pi}(\mathbf{s}_{1:t-1})$ , we have  

$$
\begin{array}{r l r}{\lefteqn{\mathrm{s_{0:FQ}}(\theta):=\operatorname*{min}_{\theta}\sum_{t=1}^{T}\mathcal{L}_{\log\mathrm{cons}}^{(t+1,1)}(\theta;\{\psi_{t},q_{t}^{\pi}\})}}&{}&{(\mathrm{Sof~Q})}\\ &{}&{=\operatorname*{min}_{\theta}\sum_{t=1}^{T}\mathbb{E}_{\pi_{s}(\cdot)}\Big[\Big(\log\phi_{t}(\mathbf{s}_{1:t})+\log\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})\mathrm{sg}\big(\psi_{t+1}^{\theta}(\mathbf{s}_{1:t+1})\big)-\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big)^{2}}\\ &{}&{\overset{(\mathrm{sBL})}{=}\operatorname*{min}_{\theta}\sum_{t=1}^{T}\mathbb{E}_{\pi_{s}(\cdot)}\Big[\Big(\beta r_{t}(\mathbf{s}_{1:t})+\log\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})e^{\beta\mathrm{sg}\big(Q_{t}^{\theta}(s_{t+1},\mathbf{s}_{1:t})\big)}-\beta Q_{t}^{\theta}(s_{t},\mathbf{s}_{1:t-1})\Big)^{2}\Big]}\end{array}
$$  

In the final line, we rewrite the loss for the soft RL special case,   $\phi_{t}\mathbf{\left(s_{1:t}\right)}~=~e^{\beta r_{t}\left(\mathbf{s}_{1:t}\right)}$   using the substitutions in Eq. (Twist to Soft RL) . Note that the  log -normalization term is analogous to an induced soft value   $V_{Q_{t}^{\theta}}(\mathbf{s}_{1:t-1})\;=\;$ $\begin{array}{r}{\frac{1}{\beta}\log\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta Q_{t}^{\theta}(s_{t},\mathbf{s}_{1:t-1})}}\end{array}$  P , so that each squared error loss has the form   $\mathbb{E}[\beta^{2}(r_{t}+V_{t}-Q_{t})^{2}]$ . Hence, we refer to this loss as  Soft $Q$ -learning  loss.  

The  log -normalization term, which arises from normalizing the twist-induced proposal, is analogous to the ‘target’ value in deep $Q$ -learning.  Lioutas et al.  ( 2022 ) consider the soft-Q learning loss to SMC sampling in self-driving applications where interaction with the environment is expensive.  Lawson et al.  ( 2018 ) adopt a similar loss function (using a parameter iz ation of the value   $V_{t}^{\pmb{\theta}}$ ) in the setting of state-space models with tractable intermediate rewards.  

RL Baseline with no Intermediate Reward The soft Q-learning loss in  Eq. (Soft Q Learning)  simplifies nicely in the case of no intermediate rewards, as in the main text  $(\phi_{t}(\mathbf{s}_{1:t})=1$  for $t<T$  and   $\varPhi_{T}=1)$ ).  

Written in terms of twist functions, we separate the terms at $t<T$  and   $t=T$  for purposes of exposition  

$$
\begin{array}{r l}&{\underset{\pmb{\theta}}{\operatorname*{min}}\,\mathcal{L}_{\mathrm{{RL}}}(\pmb{\theta}):=\underset{\pmb{\theta}}{\operatorname*{min}}\displaystyle\sum_{t=1}^{T}\mathcal{L}_{\mathrm{{log}}}^{(t+1,1)}(\pmb{\theta};\{\psi_{t},q_{t}^{\pi},\phi_{t}=1\})}\\ &{=\underset{\pmb{\theta}}{\operatorname*{min}}\displaystyle\sum_{t=1}^{T-1}\mathbb{E}_{\pi_{s}(\cdot)}\Big[\Big(\log\displaystyle\sum_{s_{t+1}}p_{0}\big(s_{t+1}|\mathbf{s}_{1:t}\big)\mathbf{s}\mathbf{g}\big(\psi_{t+1}^{\theta}(\mathbf{s}_{1:t+1})\big)-\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big)^{2}\Big]+\mathbb{E}_{\pi_{s}(\cdot)}\Big[\Big(\log\phi(\mathbf{s}_{1:T})-\log\psi_{t}^{\theta}(\mathbf{s}_{1:t+1})\Big)^{2}\Big],}\end{array}
$$  

For intermediate timesteps, note that  Eq. (RL Baseline)  enforces the recursion $\begin{array}{r}{\psi_{t-1}^{\theta}(\mathbf{s}_{1:t-1})=\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\theta}(\mathbf{s}_{1:t})}\end{array}$ in  Eq. (13)  of the main text, albeit in log space. In  App. C.2  below, we consider the one-step squared error loss enforcing this recursion directly (without logarithms), i.e. $\begin{array}{r}{\mathbb{E}_{\pi_{s}}[(\psi_{t-1}^{\theta}(\mathbf{s}_{1:t-1})-\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\theta}(\mathbf{s}_{1:t}))^{2}]}\end{array}$  , −  

C.1.2. P ATH  C ONSISTENCY  L EARNING  ( FOR  T WIST  L EARNING )  

Using the  value parameter iz ation  of the targets ( $\varPhi_{t}$  or   $V_{t}$ , see  Eq. (Twist Targets  $(\varPhi)$  ) ,  Eq. (Twist Targets (Soft RL V) ) ), the one-step log consistency loss with arbitrary proposal   $q$  recovers the path-consistency loss (PCL) from  Nachum et al.  ( 2017 ).  

Switching to a $\varPhi_{t}^{\theta}$   parameter iz ation of the twisting targets, we substitute   $\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})=\phi_{t}(\mathbf{s}_{1:t})\varPhi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$  into the log importance weights in  Eq. (48) . The log-consistency loss becomes,  

$$
\begin{array}{r l}&{\underset{\theta}{\operatorname*{min}}\,\mathcal{L}_{\mathrm{PC}}(\theta):=\underset{\theta}{\operatorname*{min}}\displaystyle\sum_{t=1}^{T}\mathcal{L}_{\mathrm{log}\,\mathrm{cons}}^{(t,1)}(\theta;\{\phi_{t},\mathrm{any}\;q\})}\\ &{\qquad\qquad\qquad=\underset{\theta}{\operatorname*{min}}\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\pi_{s}}\Bigg[\bigg(\log\phi_{t}(\mathbf{s}_{1:t})+\log\phi_{t}^{\theta}(\mathbf{s}_{1:t})-\log\phi_{t-1}^{\theta}(\mathbf{s}_{1:t-1})-\log\frac{q(s_{t}\vert\mathbf{s}_{1:t-1})}{p_{0}(s_{t}\vert\mathbf{s}_{1:t-1})}\bigg)^{2}\Bigg]}\\ &{\overset{\mathrm{(sBL)}}{=}\underset{\theta}{\operatorname*{min}}\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\pi_{s}}\Bigg[\bigg(\beta\Big(r_{t}(\mathbf{s}_{1:t})+V_{t}^{\theta}(\mathbf{s}_{1:t})-V_{t-1}^{\theta}(\mathbf{s}_{1:t-1})\Big)-\log\frac{q(s_{t}\vert\mathbf{s}_{1:t-1})}{p_{0}(s_{t}\vert\mathbf{s}_{1:t-1})}\bigg)^{2}\Bigg]}\end{array}
$$  

In particular, substituting the soft RL  potential  terms from  Eq. (Twist to Soft RL) ,  Eq. (PCL)  recovers the path consistency loss from  Nachum et al.  ( 2017 ). Note that we derived PCL from an importance sampling perspective, whereas PCL was originally derived by enforcing KKT conditions of the soft RL problem.  

We might also consider multi-step losses for various $c$ . Minimizing the square of the multi-step log weights with arbitrary $q$ recovers the multi-step PCL loss ( Nachum et al. ,  2017 ),  

$$
\begin{array}{r l r}{\lefteqn{\check{\mathbf{\rho}}_{\mathrm{CL}}^{(t,c)}(\pmb{\theta}):=\operatorname*{min}_{\pmb{\theta}}\mathcal{L}_{\mathrm{CL}}^{(t,c)}(\pmb{\theta}_{t};\{\phi_{t},\mathrm{any}\,q\})}}&{}&{(\mathrm{mult})}\\ &{=\operatorname*{min}_{\pmb{\theta}}\mathbb{E}_{\pi_{s}}\left[\left(\sum_{\tau=t}^{t+c}\log\phi_{\tau}(\mathbf{s}_{1:\tau})+\log\phi_{t+c}^{\theta}(\mathbf{s}_{1:t+c})-\log\phi_{t-1}^{\theta}(\mathbf{s}_{1:t-1})-\sum_{\tau=t}^{t+c}\log\frac{q\left(s_{\tau}\left\vert\mathbf{s}_{1:\tau-1}\right\vert\right)}{p_{0}\left(s_{\tau}\left\vert\mathbf{s}_{1:\tau-1}\right.\right)}\right)^{2}\right]}\\ &{}&{=\operatorname*{min}_{\pmb{\theta}}\mathbb{E}_{\pi_{s}}\left[\left(\sum_{\tau=t-1}^{t+c-1}\log\phi_{\tau}(\mathbf{s}_{1:\tau})+\log\psi_{t+c}^{\theta}(\mathbf{s}_{1:t+c})-\log\psi_{t-1}^{\theta}(\mathbf{s}_{1:t-1})-\sum_{\tau=t}^{t+c}\log\frac{q\left(s_{\tau}\left\vert\mathbf{s}_{1:\tau-1}\right.\right)}{p_{0}\left(s_{\tau}\left\vert\mathbf{s}_{1:\tau-1}\right.\right)}\right)^{2}\right]}\\ &{}&{\stackrel{(\mathrm{BL})}{=}\operatorname*{min}_{\pmb{\theta}}\mathbb{E}_{\pi_{s}}\left[\left(\beta\sum_{\tau=t}^{t+c}\gamma_{\tau}(\mathbf{s}_{1:\tau})+\beta\,V_{t+c}^{\theta}(\mathbf{s}_{1:t+c})-\beta\,V_{t-1}^{\theta}(\mathbf{s}_{1:t-1})-\sum_{\tau=t}^{t+c}\log\frac{q\left(s_{\tau}\left\vert\mathbf{s}_{1:\tau-1}\right.\right)}{p_{0}\left(s_{\tau}\left\vert\mathbf{s}_{1:\tau-1}\right.\right)}\right)^{2}\right]}\end{array}
$$  

where we write the   $\psi_{t}^{\theta}$   parameter iz ation in  Eq. (50)  explicitly for use in  App. D.1 . While PCL considers learned a proposal or policy $q$  with the goal of approximating the solution of a regularized MDP, we leave joint learning of proposals $\{q^{\pmb{\xi}}(s_{t}|\mathbf{s}_{1:t-1})\}_{t=1}^{T}$   and SMC target twists   $\{\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})\}_{t=1}^{T}$ }   or   $\{V_{t}^{\pmb\theta}(\mathbf{s}_{1:t})\}_{t=1}^{T}$ }   to future work.  

In  App. E , we describe using PCL to learn the proposal  only  ( Guo et al. ,  2021 ), with the values   $V_{Q_{t}}(\mathbf{s}_{1:t})$  induced from learned proposal twists   $Q_{t}^{\pmb{\xi}}(s_{t+1},\mathbf{s}_{1:t})$  which define   $\{q_{Q_{t}}^{\pmb{\xi}}(s_{t+1}|\mathbf{s}_{1:t})\}_{t=0}^{T-1}$   (in similar fashion to  Eq. (Twist-Induced Proposal (soft RL)) , but without reference to twisting targets).  

# C.2. Controlled Decoding Losses via Optimal Twist Identities ( Mudgal et al. ,  2023 )  

In  Prop. B.1  (or  Prop. 3.2  and  Eq. (13)  in the main text), we noted that the optimal twists satisfy the following relationships  

$$
\begin{array}{r l r l}{\left(\mathbf{s}_{1:t}\right)=c_{t}\;\phi_{t}(\mathbf{s}_{1:t})\displaystyle\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\displaystyle\prod_{\tau=t+1}^{T}\phi_{\tau}(\mathbf{s}_{1:\tau})}&&{=\displaystyle\frac{c_{t}}{c_{t+1}}\phi_{t}(\mathbf{s}_{1:t})\displaystyle\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})\psi_{t+1}^{*}}\\ {\left(\phi_{\equiv}\right)_{c_{t}}\displaystyle\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi(\mathbf{s}_{1:T})}&&{\stackrel{(\phi_{\equiv}=1)}{=}\displaystyle\frac{c_{t}}{c_{t+1}}\displaystyle\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})\psi_{t+1}^{*}(\mathbf{s}_{1:t+1}|\mathbf{s}_{1:t})}\end{array}
$$  

We proceed to describe two ‘controlled decoding’ (CD) losses from  Mudgal et al.  ( 2023 ) as using a  squared error  loss to enforce the optimality conditions in  Eq. (51) , for settings with no intermediate  potentials   $(\phi_{t}(\mathbf{s}_{1:t})=1$  for   $t<T$ ).  Mudgal et al.  ( 2023 ) also propose two ways to use the learned ‘twists’ at inference time, which we discuss in relation to our proposed SMC framework in  App. D.1 .  

CD-Q The CD-Q loss from  Mudgal et al.  ( 2023 ) corresponds to minimizing the one-step recursion in  Eq. (51)  using the expected squared error under a (possibly off-policy) sampling distribution $\pi_{s}$ . Assuming  no intermediate reward  and an additional squared error loss to approximate the terminal potential $\psi_{T}^{\theta}(\mathbf{s}_{1:T})\approx\phi(\mathbf{s}_{1:T})$  ≈ , we have  

$$
\mathbf{\Sigma}_{\mathrm{cap-}\mathrm{Q}}(\pmb{\theta}):=\operatorname*{min}_{\pmb{\theta}}\sum_{t=1}^{T-1}\mathbb{E}_{\pi_{s}(\cdot)}\Big[\Big(\sum_{s_{t+1}}p_{0}(s_{t+1}|\mathbf{s}_{1:t})\psi_{t+1}^{\theta}(\mathbf{s}_{1:t+1})-\psi_{t}^{\theta}(\mathbf{s}_{1:t})\Big)^{2}\Big]+\mathbb{E}_{\pi_{s}(\cdot)}\Big[\big(\phi(\mathbf{s}_{1:T})-\psi_{T}^{\theta}(\mathbf{s}_{1:t})\big)^{2}\Big]
$$  

Eq. (CD-Q)  enforces the same optimality condition as the  Eq. (RL Baseline)  loss (i.e. $\begin{array}{r l}{\psi_{t}^{\boldsymbol\theta}(\mathbf{s}_{1:t})}&{{}=}\end{array}$ $\begin{array}{r}{\sum_{s_{t+1}}p_{0}\big(s_{t+1}|\mathbf{s}_{1:t}\big)\psi_{t+1}^{\pmb\theta}\big(\mathbf{s}_{1:t+1}\big)\big)}\end{array}$ ), without log scaling of each term inside the squared error. At optimality, we have zero-variance one-step importance weights  $(w(\mathbf{s}_{1:t})=1$  in  Eq. (10) ) for the twist-induced proposal.  

At optimality, we in fact also have $\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})=\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi_{T}(\mathbf{s}_{1:T})$ P  (as in  Eq. (51)  and the proof of  Prop. B.1 ).  

CD-FUDGE While we might naively like to consider the loss $\begin{array}{r}{\mathbb{E}_{\pi_{s}(\cdot)}\bigl[\bigl(\psi_{t}^{\pmb{\theta}}(\mathbf{s}_{1:t})-\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi(\mathbf{s}_{1:T})\bigr)^{2}\bigr]}\end{array}$   P   to enforce  Prop. 3.2  or  Eq. (51) , note that marginalization over multiple steps is not tractable in general.  

Instead, the CD-FUDGE loss 9   defined as  

$$
\underset{\pmb{\theta}}{\operatorname*{min}}\,\mathcal{L}_{\mathrm{{CD-FVDGE}}}(\pmb{\theta}):=\underset{\pmb{\theta}}{\operatorname*{min}}\sum_{t=1}^{T}\mathbb{E}_{\pi_{s}(\mathbf{s}_{1:t})}\bigg[\mathbb{E}_{p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\bigg[\Big(\psi_{t}^{\pmb{\theta}}(\mathbf{s}_{1:t})-\phi(\mathbf{s}_{1:T})\Big)^{2}\Big]\bigg]
$$  

can be shown to have the same gradient as the desired (but intractable) squared error loss above ( Mudgal et al. ,  2023 ).  

Si minimizer of the expected squared error (under   $p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t}))$  to a single function   $\psi_{t}^{\boldsymbol\theta}(\mathbf{s}_{1:t})$  (which is independent of $\mathbf{s}_{t+1:T})$  ) is given by the conditional expectation ( Banerjee et al. ,  2005 ), we can also see that  Eq. (CD-FUDGE)  has the desired minimum $\begin{array}{r}{\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})=\sum_{\mathbf{s}_{t+1:T}}p_{0}\big(\mathbf{s}_{t+1:T}\big|\mathbf{s}_{1:t}\big)\phi\big(\mathbf{s}_{1:T}\big)}\end{array}$  | . Note, it is crucial that the inner expectation samples rollouts under the base model $p_{0}\big(\mathbf{s}_{t+1:T}\big|\mathbf{s}_{1:t}\big)$  |  to obta sired conditional expectation as the min mizer. While it appears that any prefix sampling distribution can be used, $\pi_{s}=p_{0}$  allows for losses to be calculated at all  t  in a single sampling run.  

Mudgal et al.  ( 2023 ) also propose two decoding-time usages for the learned twist functions   $\psi_{t}^{\theta}$ : stochastic token-by-token sampling and argmax decoding of partial sequences. We discuss their inconsistencies with our SMC framework in  App. D .  

CD-FUDGE for $\log{\psi_{t}^{\theta}}$ We can also compare  Eq. (CD-FUDGE)  with the multi-step PCL loss in  Eq. (50) , choosing   $\phi_{t}=1$ for   $t<T$  and the proposal equal to the base model   $q=p_{0}$  so that the proposal terms cancel. Noting that $\psi_{T}(\mathbf{s}_{1:T})=\phi(\mathbf{s}_{1:T})$ is fixed to the exact terminal  potential  and choosing the $c=T-t+1$ -step PCL loss for each $t$ , note that  Eq. (50)  would reduce to $\begin{array}{r}{\sum_{t}\mathbb{E}[\left(\log\phi(\mathbf{s}_{1:T})+0-\log\psi_{t}^{\boldsymbol{\theta}}(\mathbf{s}_{1:t})-0\right)^{2}]}\end{array}$    .  Deng & Raffel  ( 2023 ) optimize this loss with reweighting of terms based on timestep (higher weight for $t\approx T$  ≈ ).  Eq. (CD-FUDGE)  optimizes the squared error of the difference  without log scaling of each term , under appropriate sampling of rollouts.   10  

# C.3. SIXO: Smoothing Inference with Twisted Objectives ( Lawson et al. ,  2022 )  

Lawson et al.  ( 2022 ) adopt a noise-contrastive estimation loss ( Gutmann & Hyv arinen ,  2010 ) to learn the target twist functions using binary classification. For state space models,  Lawson et al.  ( 2022 ) adopt our setting in  App. B.2  with observation variables $o_{t}$  emitted based on the sampling state $\mathbf{s}_{1:t}$  (or simply $s_{t}$ ) and a known likelihood $\phi_{t}\big(o_{t},s_{t}\big)=\sigma\big(o_{t}|s_{t}\big)$ . As discussed in  App. B.4 , in these settings with easily evaluable intermediate  potentials , it may be preferable to parameterize $\varPhi_{t}^{\theta}(\mathbf{s}_{1:t},\mathbf{o}_{t+1:T})$  as in  Eq. (Twist Targets  $(\varPhi)$  ) .  Lawson et al.  ( 2022 ) indeed use this parameter iz ation (see their Eq. 5).  

Recall from  Eq. (39)  that the optimal twists or future values amount to conditional likelihoods,  

$$
\begin{array}{r}{\Phi_{t}^{*}\bigl(\mathbf{s}_{1:t},\boldsymbol{o}_{t+1:T}\bigr)\stackrel{\boldsymbol{o}_{t+1:T}}{\propto}\boldsymbol{\sigma}\bigl(\boldsymbol{o}_{t+1:T}\bigl|\mathbf{s}_{1:t}\bigr)\,,\qquad\qquad\psi_{t}^{*}\bigl(\mathbf{s}_{1:t},\boldsymbol{o}_{t:T}\bigr)\stackrel{\boldsymbol{o}_{t:T}}{\propto}\boldsymbol{\sigma}\bigl(\boldsymbol{o}_{t:T}\bigl|\mathbf{s}_{1:t}\bigr)\,,}\end{array}
$$  

where $\stackrel{o}{\propto}$ ∝ denotes proportionality up to a constant which depends on   $^o$  only. Using Bayes rule, we have  

$$
(o_{t+1:T}|\mathbf{s}_{1:t})=\frac{\sigma(\mathbf{s}_{1:t}|o_{t+1:T})\sigma(o_{t+1:T})}{p_{0}(\mathbf{s}_{1:t})}\overset{o_{t+1:T}}{\propto}\frac{\sigma(\mathbf{s}_{1:t}|o_{t+1:T})}{p_{0}(\mathbf{s}_{1:t})}\,,\qquad\sigma(o_{t:T}|\mathbf{s}_{1:t})\overset{o_{t:T}}{\propto}\frac{\sigma(\mathbf{s}_{1:t}|o_{t:T})}{p_{0}(\mathbf{s}_{1:t})}
$$  

noting that   $\sigma(o_{t+1:T})$  and $p_{0}\bigl(\mathbf{s}_{1:t}\bigr)$  are marginals of $\sigma(\mathbf{s}_{1:t},\mathbf{*}_{t+1:T})$  by definition. The above reasoning suggests that we may learn the twists, or likelihood ratio   $\begin{array}{r}{\varPhi_{t}^{*}(\mathbf{s}_{1:t},\mathbf{o}_{t+1:T})\propto\sigma(\mathbf{o}_{t+1:T}|\mathbf{s}_{1:t})\propto\sigma(\mathbf{s}_{1:t}|\mathbf{o}_{t+1:T})/p_{0}(\mathbf{s}_{1:t}).}\end{array}$  ∝   |  ∝ | , using a classifier which seeks to disting ish samples from $\sigma(\mathbf{s}_{1:t}|o_{t+1:T})$ |  and $p_{0}\big(\mathbf{s}_{1:t}\big)$  ( Gutmann & Hyv arinen ,  2010 ;  Lawson et al. ,  2022 ). at each  t , we classify the event   $y\,=\,1$ , indicating that   $\mathbf{s}_{1:t}\,\sim\,\sigma(\mathbf{s}_{1:t}|\pmb{o}_{t+1:T})$ , or $y\,=\,0$ , indicating that $\mathbf{s}_{1:t}\sim p_{0}(\mathbf{s}_{1:t})$ .  

Consider a given   $\scriptstyle{o_{t+1:T}}$ , which can be either   $\mathbf{\sigma}_{o_{t+1:T}}=\mathbf{1}$  in the unconditional case or $\pmb{o}_{t+1:T}\sim\pi_{s}\big(\pmb{o}_{t+1:T}\big)$  drawn from a behavioral policy as discussed below. The SIXO loss becomes  

$$
\begin{array}{r l}&{\mathfrak{x}_{0}(o_{1:T};\theta)=\displaystyle\sum_{t=1}^{T-1}\mathbb{E}_{\sigma(\mathbf{s}_{1:t}|o_{t+1:T})}\Bigl[\log\mathrm{s}\,\mathrm{s}\mathfrak{g m o i d}\bigl(\log\phi_{t}^{\theta}(\mathbf{s}_{1:t},\mathbf{o}_{t+1:T})\bigr)\Bigr]+\mathbb{E}_{p_{0}(\mathbf{s}_{1:t})}\Bigl[\log\left(1-\mathrm{s}\,\mathrm{s}\mathfrak{g m o i d}\bigl(\log\phi_{t}^{\theta}(\mathbf{s}_{1:t},\mathbf{o}_{t:T})\bigr)\right)\Bigr]}\\ &{\quad\quad\quad=\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t}|o_{t:T})}\Bigl[\log\mathrm{s}\,\mathrm{s}\mathfrak{g m o i d}\bigl(\log\psi_{t}^{\theta}(\mathbf{s}_{1:t},\mathbf{o}_{t:T})\bigr)\Bigr]+\mathbb{E}_{p_{0}(\mathbf{s}_{1:t})}\Bigl[\log\left(1-\mathrm{s}\,\mathrm{s}\mathfrak{g m o i d}\bigl(\log\psi_{t}^{\theta}(\mathbf{s}_{1:t},\mathbf{o}_{t:T})\bigr)\right)\Bigr]}\\ &{\quad\quad\quad=\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t}|o_{t:T})}\biggl[\log\frac{\psi_{t}^{\theta}(\mathbf{s}_{1:t},\mathbf{o}_{t:T})}{1+\psi_{t}^{\theta}(\mathbf{s}_{1:t},\mathbf{o}_{t:T})}\biggr]+\mathbb{E}_{p_{0}(\mathbf{s}_{1:t})}\biggl[\log\frac{1}{1+\psi_{t}^{\theta}(\mathbf{s}_{1:t},\mathbf{o}_{t:T})}\biggr]}\end{array}
$$  

Note that we can perform approximate positive sampling as in  Sec. 4  to estimate expectations in the first term.  

Exact Conditional Sampling However, we can also use the BDMC trick in  Sec. 3.3  to obtain exact target samples for general observation variables. In order to facilitate tractable sampling, we optimize the  Eq. (SIXO)  loss over a sampling distribution $\pi_{s}(o_{1:T})=\sigma(o_{1:T})$  for all   $t$ , such that the objective becomes  

$$
_{1:T})[\mathcal{L}_{\mathrm{SXO}}(o_{1:T};\theta)]=\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t},o_{t+1:T})}\bigg[\log\frac{\psi_{t}^{\theta}(\mathbf{s}_{1:t},o_{t:T})}{1+\psi_{t}^{\theta}(\mathbf{s}_{1:t},o_{t:T})}\bigg]+\mathbb{E}_{p_{0}(\mathbf{s}_{1:t})\sigma(o_{t+1:T})}\bigg[\log\frac{1}{1+\psi_{t}^{\theta}(\mathbf{s}_{1:t},o_{t:T})}\bigg].
$$  

With this choice,  te that we may sample once from $\begin{array}{r}{\sigma(\mathbf{s}_{1:T},\mathbf{o}_{1:T})=\prod_{t=1}^{T}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\sigma(o_{t}|\mathbf{s}_{1:t})}\end{array}$ | − |  using ancestral sampling and use the appropriate truncations for positive sampling terms involving   $\sigma(\mathbf{s}_{1:t},\mathbf{*}_{t+1:T})$ . By shuffling observation variables across a batch of  K  samples, we may obtain samples from the product of marginals   $p_{0}(\mathbf{s}_{1:T})\sigma(o_{1:T})$  or   $p_{0}(\mathbf{s}_{1:t})\sigma(\pmb{o}_{t+1:T})$ in the negative sampling term.  

In the main text, note that we condition on   $o_{T}=1$  or $o_{T}=\mathbf{s}_{T+1:T+c}$  (for infilling).  

Gradient and Comparison with CTL Proceeding with the $\psi_{t}^{\theta}$   parameter iz ation for th target $\sigma(\mathbf{s}_{1:T}|o_{T})=\sigma(\mathbf{s}_{1:T})$ with fixed $o_{T}$  and unconditional twists   $\psi_{t}^{\boldsymbol\theta}(\mathbf{s}_{1:t})$ , the gradient of  Eq. (SIXO)  with respect to  θ  is  

$$
\begin{array}{r l}&{\displaystyle\mathrm{co}(\pmb\theta)=\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t})}\bigg[\nabla_{\pmb\theta}\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})-\frac{\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})}{1+\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})}\nabla_{\pmb\theta}\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})\bigg]-\mathbb{E}_{p_{0}(\mathbf{s}_{1:t})}\bigg[\frac{\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})}{1+\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})}\nabla_{\pmb\theta}\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})\bigg]}\\ &{\quad\quad=\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t})}\bigg[\frac{1}{1+\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})}\nabla_{\pmb\theta}\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})\bigg]-\mathbb{E}_{p_{0}(\mathbf{s}_{1:t})}\bigg[\frac{\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})}{1+\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})}\nabla_{\pmb\theta}\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})\bigg]}\end{array}
$$  

The SIXO gradient is superficially similar to our CTL gradient in  Sec. 4.1 , in that it involves $\nabla_{\theta}\log{\psi_{t}^{\pmb{\theta}}}$   under positive and negatives samples. However, viewing $\tilde{\pi}_{t}^{\pmb\theta}\big(\mathbf{s}_{1:t}\big)=p_{0}\big(\mathbf{s}_{1:t}\big)\psi_{t}^{\pmb\theta}\big(\mathbf{s}_{1:t}\big)$  as the unnormalized density of our intermediate twisting target, we can see that the second term in the  SIXO  update includes $\tilde{\pi}_{t}^{\theta}(\mathbf{s}_{1:t})$ . Rewriting to highlight differences with our CTL gradient, we have  

$$
\begin{array}{r l r}&{\mathcal{L}_{\mathrm{SICO}}=\displaystyle\sum_{t=1}^{T}\biggl(\displaystyle\sum_{\mathbf{s}_{1:t}}\sigma(\mathbf{s}_{1:t})\frac{1}{1+\psi_{t}^{\theta}(\mathbf{s}_{1:t})}\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})-\displaystyle\sum_{\mathbf{s}_{1:t}}\tilde{\pi}_{t}^{\theta}(\mathbf{s}_{1:t})\frac{1}{1+\psi_{t}^{\theta}(\mathbf{s}_{1:t})}\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\biggr)}&\\ &{\theta\mathcal{L}_{\mathrm{CTL}}=\displaystyle\sum_{t=1}^{T}\biggl(\displaystyle\sum_{\mathbf{s}_{1:t}}\sigma(\mathbf{s}_{1:t})}&{\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})-\displaystyle\sum_{\mathbf{s}_{1:t}}\tilde{\pi}_{t}^{\theta}(\mathbf{s}_{1:t})}&{\frac{1}{\mathcal{Z}_{t}^{\psi}}}&{\nabla_{\theta}\log\psi_{t}^{\theta}(\mathbf{s}_{1:t})\biggr)}&\end{array}
$$  

To compare the two, first note that the positive sampling gradient in SIXO is scaled by a factor of $\frac{1}{1+\psi_{t}^{\theta}\left(\mathbf{s}_{1:t}\right)}$  factor (which reflects the mis classification probability under $\psi_{t}^{\pmb\theta}.$ ). For the negative sampling terms, note that $\tilde{\pi}_{t}^{\theta}(\mathbf{s}_{1:t})$  is divided by a factor of $\frac{1}{1+\psi_{t}^{\theta}\left(\mathbf{s}_{1:t}\right)}$  in the SIXO gradient, instead of the true normalization constant   $\mathcal{Z}_{t}^{\psi}$   for the gradient of our CTL loss  Eq. (22) .  

# C.4. FUDGE: Future Discriminators ( Yang & Klein ,  2021 )  

In contrast to SIXO, the FUDGE method from  Yang & Klein  ( 2021 ) seeks to directly learn a discriminative classifier to match the conditional likelihood   $\psi_{t}^{*}(\mathbf{s}_{1:t},o_{T})\propto\sigma(o_{T}|\mathbf{s}_{1:t})$  ∝   |  or   $\psi_{t}^{*}(\mathbf{s}_{1:t},\mathbf{o}_{t:T})\propto\sigma(\mathbf{o}_{t:T}|\mathbf{s}_{1:t})$  ∝   |  (see  App. B.2 ).  

As before, we define the joint distribution   $\sigma({\bf s}_{1:T},o_{T})\;=\;p_{0}({\bf s}_{1:T})\sigma(o_{T}|{\bf s}_{1:T})$  with   $\phi\bigl(\mathbf{s}_{1:T},o_{T}\bigr)\,=\,\sigma\bigl(o_{T}|\mathbf{s}_{1:T}\bigr)$ . From Eq. (52)  above or  App. B.2 Eq. (40) , we have  

$$
\psi_{t}^{*}(\mathbf{s}_{1:t},o_{T})\propto\sigma(o_{T}|\mathbf{s}_{1:t}):=\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\sigma(o_{T}|\mathbf{s}_{1:T})
$$  

Yang & Klein  ( 2021 ) consider training a ‘future discriminator’ $\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t},o_{T})\approx\sigma(o_{T}|\mathbf{s}_{1:t})$  ≈   | which, in  Eq. (54 nalizes over future tokens to predict the expected probability that a full sequence with prefix $\mathbf{s}_{1:t}$  emits $o_{T}$  (e.g., let $o_{T}=a$  be the probability of a classifier for class   $a$ , or the probability that $\mathbf{S}_{1:T}$  satisfies a desired attribute indicated by a boolean   $o_{T}=1$ ). In similar fashion to SIXO in the previous section, we define a binary random variable $y$  such that  

$$
\sigma(y|\mathbf{s}_{1:t},o_{T})=\left\{\!\!\begin{array}{l l}{\sigma(o_{T}|\mathbf{s}_{1:t})\quad\quad}&{y=1}\\ {1-\sigma(o_{T}|\mathbf{s}_{1:t})\quad}&{y=0}\end{array}\!\!\right.\quad\quad\quad p_{\psi_{t}^{\theta}}(y|\mathbf{s}_{1:t},o_{T})=\left\{\!\!\begin{array}{l l}{\psi_{t}^{\theta}(\mathbf{s}_{1:t},o_{T})\quad}&{y=1}\\ {1-\psi_{t}^{\theta}(\mathbf{s}_{1:t},o_{T})\quad}&{y=0}\end{array}\!\!\right.
$$  

where we directly parameterize   $p_{\psi_{t}^{\theta}}(y|\mathbf{s}_{1:t},o_{T})=\psi_{t}^{\theta}(\mathbf{s}_{1:t},o_{T})$  to be a probability distribution (e.g. using a sigmoid or softmax activation). For a given observation random variable   $o_{T}$  and partial sequence   $\mathbf{s}_{1:t}$ , we can define the FUDGE loss  

$$
\begin{array}{r l}{(\mathbf{s}_{1:t},o_{T};\theta):=\displaystyle\sum_{t=1}^{T}D_{\mathbf{k}1}\Big(\sigma(y|\mathbf{s}_{1:t},o_{T})\left\|\,p_{\psi_{t}^{\theta}}(y|\mathbf{s}_{1:t},o_{T})\right)}&{}\\ {=\displaystyle\sum_{t=1}^{T}-\Big[\sigma(y=1|\mathbf{s}_{1:t},o_{T})\log p_{\psi_{t}^{\theta}}(y=1|\mathbf{s}_{1:t},o_{T})+\sigma(y=0|\mathbf{s}_{1:t},o_{T})\log p_{\psi_{t}^{\theta}}(y=0|\mathbf{s}_{1:t},o_{T})\Big]}&{}\\ {=\displaystyle\sum_{t=1}^{T}-\mathbb{E}_{p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\bigg[\sigma(o_{T}|\mathbf{s}_{1:T})\log\psi_{t}^{\theta}(\mathbf{s}_{1:t},o_{T})+\Big(1-\sigma(o_{T}|\mathbf{s}_{1:T})\Big)\log\Big(1-\psi_{t}^{\theta}(\mathbf{s}_{1:t},o_{T})\Big)\bigg].}\end{array}
$$  

d to the third line, we have used the fact t $\sigma(y\;=\;1|{\bf s}_{1:t},o_{T})\;=\;\sigma(o_{T}|{\bf s}_{1:t})\;=\;$ $\begin{array}{r}{\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\sigma(o_{T}|\mathbf{s}_{1:T})}\end{array}$ P  from  Eq. (54)  and  Eq. (55) . At the optimum, $p_{\psi_{t}^{\theta}}(y=1|\mathbf{s}_{1:t},o_{T})=\sigma(y=1|\mathbf{s}_{1:t},o_{T})$ implies $\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t},o_{T})=\sigma(o_{T}|\mathbf{s}_{1:t})$ | , as desired.  

While sampling may be done using an arbitrary distribution over prefixes $\mathbf{s}_{1:t}$  and observation   $o_{T}$  ,  Eq. (FUDGE)  requires mpled under the base model   $p_{0}\big(\mathbf{s}_{t+1:T}\big|\mathbf{s}_{1:t}\big)$  in order to ensure sampling from the appropriate distribution $\sigma(y=1|\mathbf{s}_{1:t},o_{T})$ | . This restriction is similar to what we required in  Eq. (CD-FUDGE) , although the loss in  Eq. (FUDGE)  is based on cross entropy classification rather than a squared error. We discuss the choices made in our experiments below.  

Yang & Klein  ( 2021 ) Setting In the original FUDGE paper,  Yang & Klein  ( 2021 ) consider learning from a dataset of labelled examples   $(\mathbf{s}_{1:T},o_{T})$  or   $(\mathbf{s}_{1:t},o_{T})$  for a binary observation variable   $o_{T}=1$  which defines the target distribution.  

Unconditional Twist Setting For the unconditional twist experiments in  Sec. 7.2.1 - 7.2.2 , we sample under the base model proposal $\pi_{s}(\mathbf{s}_{1:t})=p_{0}(\mathbf{s}_{1:t})$  where the target distribution conditions on $o_{T}=1$  and $\sigma(o_{T}=1|\mathbf{s}_{1:T})=\phi(\mathbf{s}_{1:T})=\sigma(y=$ $1|\mathbf{s}_{1:T},o_{T}=1\rangle$ ) . In particular, we optimize  

$$
\underset{\theta}{\operatorname*{min}}\sum_{t=1}^{T}\mathbb{E}_{p_{0}(\mathbf{s}_{1:t})}[\mathcal{L}_{\mathrm{FIDGE}}(\mathbf{s}_{1:t},o_{T}=1;\pmb{\theta})]
$$  

Conditional Twist Setting For conditional twist learning, we can consider amortizing learning the twists   $\psi_{t}(\mathbf{s}_{1:t},o_{T})$ over some distribution of observation variables   $\pi_{s}(\mathbf{s}_{1:t},o_{T})$ . In particular, in our infilling experiments in  Sec. 7.2.3 , we consider sampling under the following joint distribution,  

$$
\pi_{s}\big(\mathbf{s}_{1:t},o_{T}\big)=p_{0}\big(\mathbf{s}_{1:t}\big)\sigma\big(o_{T}\,|\,\mathbf{s}_{1:t}\big)\,,
$$  

which we can sample from by first sampling from $p_{0}(\mathbf{s}_{1:T})\sigma(o_{T}\mid\mathbf{s}_{1:T})$  and then dropping   $\mathbf{s}_{t+1:T}$  subsequence. Therefore, the overall objective becomes  

$$
\begin{array}{r l}&{\underset{\pmb{\theta}}{\operatorname*{min}}\,\mathbb{E}_{\pi_{s}(\mathbf{s}_{1:t},o_{T})}[\mathcal{L}_{\mathrm{FUVG}}(\mathbf{s}_{1:t},o_{T};\pmb{\theta})]}\\ &{\qquad=\underset{\pmb{\theta}}{\operatorname*{min}}\sum_{t=1}^{T}-\mathbb{E}_{p_{0}(\mathbf{s}_{1:T})\sigma(o_{T}\mid\mathbf{s}_{1:t})}\bigg[\sigma(o_{T}|\mathbf{s}_{1:T})\log\psi_{t}^{\pmb{\theta}}(\mathbf{s}_{1:t},o_{T})+\left(1-\sigma(o_{T}|\mathbf{s}_{1:T})\right)\log\left(1-\psi_{t}^{\pmb{\theta}}(\mathbf{s}_{1:t},o_{T})\right)\bigg].}\end{array}
$$  

expectation $p_{0}(\mathbf{s}_{1:T})$  the expectation under   $p_{0}\big(\mathbf{s}_{t+1:T}\big|\mathbf{s}_{1:t}\big)$  from  Eq. (FUDGE at rollout of $\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t}$  |  used to sample from $p_{0}(\mathbf{s}_{1:T})$  should be independent of the rollout used to sample from $\sigma\big(o_{T}|\mathbf{s}_{1:t}\big)$  | .  

# D. Decoding Strategies using Learned Twists from  Mudgal et al.  ( 2023 ) D.1. Proposal Sampling in  Mudgal et al.  ( 2023 )  

As noted in  App. C.2  (and in $\mathcal{L}^{*}(\pmb{\theta})$  in  Mudgal et al.  ( 2023 )), the CD losses can be seen as enforcing the optimality conditions  

$$
\psi_{t}^{\mathrm{cd}*}({\bf s}_{1:t})=\sum_{{\bf s}_{t+1:T}}p_{0}({\bf s}_{t+1:T}|{\bf s}_{1:t})\phi({\bf s}_{1:T}),\qquad\qquad\forall t.
$$  

In RL terms, we interpret the twists   $\psi_{t}^{\mathrm{cd}*}$ as performing  policy evaluation  of the expected  unregularized  ‘reward’   $\phi(\mathbf{s}_{1:T})$ under a fixed policy $p_{0}(\mathbf{s}_{1:T})$ . The notation of  Mudgal et al.  ( 2023 ) (their Eq. (1), (5), our  Eq. (57) ) indeed corresponds to  

$$
\phi\bigl(\mathbf{s}_{1:T}\bigr)=:r_{\mathrm{cd}}\bigl(\mathbf{s}_{1:T}\bigr).
$$  

However,  Mudgal et al.  ( 2023 ) propose to use the learned twist functions   $\psi_{t}^{\theta}$   to perform one-step sampling as  

$$
q_{t}^{\mathrm{cd}}(s_{t}|\mathbf{s}_{1:t-1})\propto p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta\ \psi_{t}^{\theta}(\mathbf{s}_{1:t})}
$$  

We proceed to explain that this scheme  does not correspond to sampling from the twist-induced proposal  under two different definitions of the target $\sigma(\mathbf{s}_{1:T})$  (or potential   $\phi(\mathbf{s}_{1:T})_{\lambda}$ ) in our SMC framework.  

Comparison with Our   $\phi(\mathbf{s}_{1:T})\,=\,r_{\mathbf{c}\mathbf{d}}(\mathbf{s}_{1:T})$  Case: As we have argued above, the CD-Q and CD-FUDGE may be viewed as learning twist values $\psi_{t}^{\theta}$   for a terminal  potential   $\phi(\mathbf{s}_{1:T})=r_{\mathrm{cd}}(\mathbf{s}_{1:T})$ . However, our twist-induced proposal which minimizes the variance of the one-step importance weights with these SMC targets   $\{\pi_{t}^{\theta}\}$ }  would yield  

$$
q_{t}^{\pi}(s_{t}|\mathbf{s}_{1:t-1})\propto p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\theta}(\mathbf{s}_{1:t}),
$$  

which, compared to  Eq. (CD proposal)  does not exponentiate or scale $\psi_{t}^{\theta}$   and is directly proportional to the expected $r_{\mathrm{cd}}$ .  

Comparison with Our   $\phi(\mathbf{s}_{1:T})\,=\,e^{\beta r_{\mathbf{cd}}(\mathbf{s}_{1:T})}$   Case (Soft RL): The stochastic sampling in  Eq. (CD proposal)  is also reminiscent of the twist-induced proposal in the soft RL case of our framework where, in contrast to  Eq. (CD reward) , the target is defined via   $\phi(\mathbf{s}_{1:T})=e^{\beta r_{\mathrm{cd}}\left(\mathbf{s}_{1:T}\right)}$ . As in  App. B.3 ,  

$$
q_{t}^{\pi}(s_{t}|\mathbf{s}_{1:t-1})\propto p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta\ V_{t}^{\theta}(\mathbf{s}_{1:t})}
$$  

We proceed to write both   $q_{t}^{\mathrm{cd}}$   and   $q_{t}^{\pi}$   as the solution to a variational optimization,  highlighting similarities in blue , but noting the  different definitions of $\phi$  in terms of $r_{c d}$ . We assume no intermediate  potential  or reward, and consider the optimal twists to emphasize the role of $r_{\mathrm{cd}}$ . Using  Mudgal et al.  ( 2023 ) Eq. 2 and Thm 2.1 (for CD) and  Eq. (Optimal Intermediate Soft Value)  (for soft RL), we have  

$$
q_{t}^{\mathrm{cd}^{*}}(s_{t}|\mathbf{s}_{1:t-1})=\operatorname*{arg\,max}_{q(s_{t}|\mathbf{s}_{1:t-1})}\mathbb{E}_{q(s_{t}|\mathbf{s}_{1:t-1})}\Big[\underbrace{\mathbb{E}_{p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\big[r_{\mathrm{cd}}(\mathbf{s}_{1:T})\big]}_{\psi_{t}^{\mathrm{cd}^{*}}(\mathbf{s}_{1:t})\mathrm{~(for~}\phi=r_{\mathrm{cd}})}\Big]-\frac{1}{\beta}D_{\mathrm{KL}}(q(s_{t}|\mathbf{s}_{1:t-1})\ ||\ p_{0}(s_{t}|\mathbf{s}_{1:t-1})).
$$  

(CD proposal optimization)  

$$
q_{t}^{\pi^{*}}(s_{t}|\mathbf{s}_{1:t-1})=\underbrace{\arg\operatorname*{max}}_{q(s_{t}|\mathbf{s}_{1:t-1})}\mathbb{E}_{q(s_{t}|\mathbf{s}_{1:t-1})}\bigg[\underbrace{\frac{1}{\beta}\log\mathbb{E}_{p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\Big[e^{\beta r_{\mathrm{sd}}(\mathbf{s}_{1:T})}\Big]}_{V_{t}^{\prime}(\mathbf{s}_{1:t})\mathrm{~(for~}\phi=e^{\beta r_{\mathrm{sd}}})}\bigg]-\frac{1}{\beta}D_{\mathrm{KL}}(q(s_{t}|\mathbf{s}_{1:t-1})\,||\,p_{0}(\phi=e^{\beta r_{\mathrm{sd}}}))=0
$$  

(Soft RL proposal optimization)  

The second terms of  Eq. (CD proposal optimization)  and  Eq. (Soft RL proposal optimization)  match and correspond to one-step KL divergence regularization of the policy   $q_{t}\big(s_{t}|\mathbf{s}_{1:t-1}\big)$ . However, the expectation terms differ as we now discuss.  

Soft Values Account for Future Regularization Using  Eq. (Optimal Intermediate Soft Value)  to expand the definition of the soft value function, we see that  Eq. (Soft RL proposal optimization)  also implicitly contains an expected terminal reward,  

$$
\mathbf{s}_{1:t})=\frac{1}{\beta}\log\mathbb{E}_{p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}e^{\beta r_{\mathrm{cd}}(\mathbf{s}_{1:T})}=\operatorname*{max}_{q(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\mathbb{E}_{q(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})}\big[r_{\mathrm{cd}}(\mathbf{s}_{1:T})\big]-\frac{1}{\beta}D_{\mathrm{KL}}(q(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})|\mathbf{s}_{1:t}),
$$  

As   $\beta\,\rightarrow\,0$  in  Eq. (58) , this optimization strictly  $q(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\,=\,p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})$ , and the soft value function recovers the expected reward under the base model $\mathbb{E}_{p_{0}\left(\mathbf{s}_{t+1:T}\left\vert\mathbf{s}_{1:t}\right.\right)}\big[r_{\mathrm{cd}}\big(\mathbf{s}_{1:T}\big)\big]$ , which appears t term  Eq. (CD proposal  | optimization) . On the other hand, the second term in  Eq. (CD proposal optimization)  uses $\beta>0$  for optimization of the prop $q(s_{t}|\mathbf{s}_{1:t-1})$  at the current step. This inconsistency in  Eq. (CD proposal optimization)  (using $\beta=0$  in the first term and $\beta>0$  in the second term) arises from the fact that  Eq. (CD proposal optimization)  does not consider the effect of  future regularization, while the MDP formulation in  Eq. (Soft RL proposal optimization)  does so via the optimization in  Eq. (58) and the log-mean-exp form of the soft value function   $V_{t}^{\ast}$ .  

On  Mudgal et al.  ( 2023 )’s One-Step Proposal and SMC Interpretation As noted in  Eq. (57) , the twists learned by Mudgal et al.  ( 2023 ) correspond to policy evaluation for the reward   $r_{\mathrm{cd}}$  under the base model   $p_{0}$ . However, we have argued that the one-step proposal in  Eq. (CD proposal)  (which considers one-step KL regularization of   $q_{t}^{\mathrm{cd}}$   to $p_{0,}$ ) does not immediately fit within our SMC framework. In particular, it is not apparent that the composition of one-step proposals $\begin{array}{r}{q^{\mathrm{cd}}(\mathbf{s}_{1:T})=\overline{{\prod_{\tau=1}^{t}q_{\tau}^{\mathrm{cd}}(s_{\tau}|\mathbf{s}_{1:\tau-1})}}}\end{array}$  samples from the marginals   $\sigma(\mathbf{s}_{1:t})$  of a natural target distribution $\sigma(\mathbf{s}_{1:T})$  at optimality. −  

Flexible Inference-Time   $\beta$  Scaling The experiments in  Mudgal et al.  ( 2023 ) evaluate tradeoff curves between expected reward and $D_{\mathrm{KL}}\big(q^{\mathrm{cd}}\big(\mathbf{s}_{1:T}\big)\parallel p_{0}\big(\mathbf{s}_{1:T}\big)\big)$    for various values of regularization strength $\beta$ . Since the twists learned by  Mudgal et al.  ( 2023 ) in  Eq. (57)  do not depend on $\beta$ , sampling according to  Eq. (CD proposal)  or  Eq. (CD proposal optimization) has the benefit of allowing flexible tempering or $\beta$ -scaling at inference time without additional learning.  

Such tradeoff curves are also natural from the perspective of soft-RL (c.f.  Eq. (42)  and  Eq. (46) ). While  Eq. (58)  appears to require separate twist-learning procedures for each $\beta$ , flexible inference-time   $\beta$  scaling could be achieved with a single training run in our framework by learning a conditional twist network $\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t},\beta)$  which considers   $\beta$  in its input and training loss, or adapting the methods of ( Bae et al. ,  2022 ) proposed in the context of rate-distortion optimization.  

Comparison with  Khanov et al.  ( 2024 ) Khanov et al.  ( 2024 ) consider softmax decoding similar to  Eq. (Twist-Ind. proposal  $(\phi=r_{\mathrm{cd}})$ ) . However, instead of   $V_{t}^{\theta}(\mathbf{s}_{1:t})$  as the logit, they use a reward model $r_{T}\mathbf{\Psi}(\mathbf{s}_{1:T})$  which is trained from full sequences  $(\phi(\mathbf{s}_{1:T})=e^{\beta r_{T}(\mathbf{s}_{1:T})})$ , but applied to partial sequences without modification,   $r_{T}(\mathbf{s}_{1:t})$ . This clearly does not correspond to a twist or soft value function $\begin{array}{r}{\bar{V_{t}^{*}}(\bar{\mathbf{s}_{1:t}})=\frac{1}{\beta}\log\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})e^{\beta r_{T}(\mathbf{s}_{1:T})}\neq r_{T}(\mathbf{s}_{1:t}),}\end{array}$ .  

# D.2. Blockwise Greedy Decoding in  Mudgal et al.  ( 2023 )  

As an alternative use of the twist functions at inference time and a generalization of best-of- $K$  decoding to partial sequences, Mudgal et al.  ( 2023 ) also consider a ‘blockwise’ decoding scheme using the learned twist functions   $\psi_{t}^{\theta}$ . In particular, for   $K$ partial completions of length et al.  ( 2023 ) propose to choose $M$  (from a prefix $\mathbf{s}_{1:t}$ ), sampled from the base model, $\mathbf{s}_{t+1:t+M}^{(k)}\sim p_{0}(\mathbf{s}_{t+1:t+M}|\mathbf{s}_{1:t})$   ,  Mudgal  

$$
\mathbf{s}_{t+1:t+M}^{\omega}=\mathop{\arg\operatorname*{max}}_{k}\psi_{t+M}^{\theta}\bigl(\mathbf{s}_{1:t+M}^{k}\bigr)
$$  

and proceed with sampling   $K$  further continuations with prefix $\mathbf{s}_{1:t+M}^{\omega}$   until the next resampling step or an end-of-string token is reached. The  arg max  selection strategy may seem natural from the unregularized RL (as $\beta\to\infty$ ) or expected future reward perspective in  App. D.1 , but does not yield samples from   $\sigma(\mathbf{s}_{1:T})$  with the corresponding optimal twists.  

Our SMC framework instead would advocate  probabilistic  resampling based on the approximate twist functions using the ( $c-$  or $M$ -step) importance weights in  Sec. 3  in order to match the desired target distribution.  

Finally,  Khanov et al.  ( 2024 ) also consider  arg max  decoding of next tokens using the unmodified $r_{T}(\mathbf{s}_{1:t})$  described above.  

# E. Proposal Learning Methods  

We next describe  ethods for learning variational policies or proposa $\begin{array}{r}{\underline{q}^{\pmb{\xi}}(\mathbf{s}_{1:T})\,=\,\prod_{t=1}^{T}q_{t}^{\pmb{\xi}}\bigl(s_{t}\vert\mathbf{s}_{1:t-1}\bigr)}\end{array}$ | ram by $\xi$ , which can be used for SMC sampling with intermediate targets $\pi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$  and learned twists $\psi_{t}^{\boldsymbol\theta}(\mathbf{s}_{1:t})$  or $V_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$ parameterized by  θ . Alternatively, such proposals may be used directly in the IWAE bounds on   $\log\mathcal{Z}_{\sigma}$ , which rely on simple importance sampling over full sequences as in  Sec. 2.1  and  do not require the definition of intermediate targets $\pi_{t}$ .  

In  App. E.3 , we provide a detailed description of the DPG policy gradient method, which can be interpreted as a maximum likelihood objective for a sequential energy-based model. To distinguish this EBM approach from our CTL method for twist learning, we emphasize issues which can arise from naive use of a proposal-learning objective to define intermediate twisting targets for SMC in  App. E.3.1 .  

# E.1. Path Consistency Learning for Controlled Generation  

Guo et al.  ( 2021 ) consider learning   $Q$ -values to obtain a fine-tuned variational policy which can be directly used as a sampling distribution for controlled generation. Building on the path consistency learning (PCL) loss in  Nachum et al. ( 2017 ) and  App. C.1.2 ,  Guo et al.  ( 2021 ) consider parameterizing the proposal using $Q_{t}^{\boldsymbol{\xi}}(\bar{s_{t}},\bar{\mathbf{s}}_{1:t-1})$ ,  

$$
q_{t}^{\pmb\xi}(s_{t}|\mathbf{s}_{1:t-1})=p_{0}(s_{t}|\mathbf{s}_{1:t-1})e^{\beta Q_{t}^{\pmb\xi}(s_{t},\mathbf{s}_{1:t-1})-\beta V_{Q^{\pmb\xi}}(\mathbf{s}_{1:t-1})}
$$  

where   $\begin{array}{r}{V_{Q\pmb{\varepsilon}}\big(\mathbf{s}_{1:t-1}\big)=\frac{1}{\beta}\log\sum_{s_{t}}p_{0}\big(s_{t}\big\vert\mathbf{s}_{1:t-1}\big)e^{\beta Q_{t}^{\pmb{\varepsilon}}}}\end{array}$  enforces normalization.  

Guo et al.  ( 2021 ) define the targets using $\bar{Q}_{t}^{\pmb{\xi}}(s_{t},\mathbf{s}_{1:t-1})$ , a slowly-updated target network based on   $Q_{t}^{\xi}$ . Using the implied − form of the soft value $\begin{array}{r}{\bar{V}({\bf s}_{1:t-1}):=\frac{1}{\beta}\log\sum_{s_{t}}p_{0}\bigl(s_{t}\vert{\bf s}_{1:t-1}\bigr)e^{\beta\bar{Q}_{t}^{\xi}\left(s_{t},{\bf s}_{1:t-1}\right)}}\end{array}$ , the single-step PCL loss becomes  

$$
_{-^{0}}(\pmb{\xi})=\operatorname*{min}_{\pmb{\xi}}\sum_{t=1}^{T}\mathbb{E}_{\pi_{\mathfrak{s}}(\mathbf{s}_{1:t})}\bigg[\Big(r_{t}(\mathbf{s}_{1:t})+\mathfrak{s g}(\bar{V}_{t}(\mathbf{s}_{1:t}))-\mathfrak{s g}(\bar{V}_{t-1}(\mathbf{s}_{1:t-1}))-Q_{t}^{\pmb{\xi}}(s_{t},\mathbf{s}_{1:t-1})+V_{Q}\mathfrak{s}(\mathbf{s}_{1:t-1})\Big)\bigg].
$$  

where $\mathsf{s g}(\cdot)$  indicates stop gradient. Building on the interpretation in  App. C.1 , we view $\bar{V}_{t}(\mathbf{s}_{1:t})$  and $\bar{V}_{t-1}(\mathbf{s}_{1:t-1})$  as the − − twisting targets, with a learned proposal parameterized by   $Q_{t}^{\xi}$   as in  Eq. (60)  (or  App. B.4 ). While the loss in  Eq. (61) is similar in practice to the soft Q-learning loss in  App. C.1.1 , we emphasize that the latter is motivated from the SMC perspective with the twisting targets as the primary object of interest and flexibility in the choice of proposal. By contrast, Guo et al.  ( 2021 ) are interested in learning a proposal policy and do not consider, for example, resampling according to $\bar{V}_{t}$ .  

Guo et al.  ( 2021 );  Nachum et al.  ( 2017 ) also consider ‘multi-step’ PCL losses ( Eq. (multi-step PCL) ) which use observed reward during rollouts of length $\lambda$  to limit reliance on estimated intermediate values $\bar{V}_{t}(\mathbf{s}_{1:t})$ . The objective in $\mathrm{Nu}$  et al. ( 2023 ) also corresponds to a PCL objective.  

# E.2. Policy Gradient Methods  

Traditional RLHF pipelines use a policy gradient method such as PPO to optimize the objective in  Eq. (42) ,  

$$
\mathcal{L}_{\mathrm{ELBO}}(\pmb{\xi})=\operatorname*{max}_{\pmb{\xi}}\,\mathbb{E}_{q^{\pmb{\xi}}(\mathbf{s}_{1:T})}[r_{T}(\mathbf{s}_{1:T})]-\frac{1}{\beta}\,D_{\mathrm{KL}}\big(q^{\pmb{\xi}}(\mathbf{s}_{1:T})\,\big\|\,p_{0}(\mathbf{s}_{1:T})\big)=\operatorname*{min}_{\pmb{\xi}}D_{\mathrm{KL}}\big(q^{\pmb{\xi}}(\mathbf{s}_{1:T})\,\big\|\,\sigma(\mathbf{s}_{1:T})\big)
$$  

where $\begin{array}{r}{r_{T}\bigl(\mathbf{s}_{1:T}\bigr)=\frac{1}{\beta}\log\phi\bigl(\mathbf{s}_{1:T}\bigr)}\end{array}$  corresponds to our final twist. As in  Eq. (46) , the gap in this optimization is the mode- seeking KL divergence   $D_{\mathrm{KL}}\big(q^{\pmb{\xi}}(\mathbf{s}_{1:T})\,\big|\big|\,\sigma(\mathbf{s}_{1:T})\big)$ .  

Notably, this objective does not make use of exact target samples from   $\sigma(\mathbf{s}_{1:T})$  when they are available. Further, the mode-seeking behavior has been shown to reduce diversity of fine-tuned models ( Stiennon et al. ,  2020 ;  Go et al. ,  2023 ). To combat this,  Go et al.  ( 2023 ) derive policy gradient methods to optimize arbitrary $f$ -divergences   $D_{f}\big(q^{\pmb{\xi}}(\mathbf{s}_{1:T})\,\big|\big|\,\sigma(\mathbf{s}_{1:T})\big)$    between the learned variational policy $q^{\pmb\xi}$   and target $\sigma$ .  

# E.3. Policy Gradient with Mass-Covering / Maximum Likelihood KL Divergence  

We focus on the case of minimizing the mass-covering  KL  divergence $D_{\mathrm{KL}}\big(\sigma(\mathbf{s}_{1:T})\bigparallel q^{\pmb{\xi}}(\mathbf{s}_{1:T})\big)$    to train $q_{\xi}$ , which constitutes the distributional policy gradients ( DPG ) method for language model finetuning ( Parshakova et al. ,  2019 ;  Khalifa et al. ,  2020 ; Korbak et al. ,  2022a ;  Go et al. ,  2023 ) and has been used to learn SMC proposals in state-space models in ( Gu et al. ,  2015 ).  

In particular, the gradient of $D_{\mathrm{KL}}\big(\sigma({\bf s}_{1:T})\,\big\|\,q^{\pmb{\xi}}({\bf s}_{1:T})\big)=\mathbb{E}_{\sigma({\bf s}_{1:T})}[\log\sigma({\bf s}_{1:T})-\log q^{\pmb{\xi}}({\bf s}_{1:T})]\,.$  is  

$$
\begin{array}{r}{\nabla_{\pmb{\xi}}D_{\mathtt{K L}}\big(\sigma(\mathbf{s}_{1:T})\,\big\|\,q^{\pmb{\xi}}(\mathbf{s}_{1:T})\big)=-\mathbb{E}_{\sigma(\mathbf{s}_{1:T})}\big[\nabla_{\pmb{\xi}}\log q^{\pmb{\xi}}(\mathbf{s}_{1:T})\big]=-\mathbb{E}_{q^{\pmb{\xi}}(\mathbf{s}_{1:T})}\Bigg[\frac{\sigma(\mathbf{s}_{1:T})}{q^{\pmb{\xi}}(\mathbf{s}_{1:T})}\nabla_{\pmb{\xi}}\log q^{\pmb{\xi}}(\mathbf{s}_{1:T})}\\ {=-\mathbb{E}_{q^{\pmb{\xi}}(\mathbf{s}_{1:T})}\Bigg[\frac{1}{Z_{\sigma}}\frac{\widetilde{\sigma}(\mathbf{s}_{1:T})}{q^{\pmb{\xi}}(\mathbf{s}_{1:T})}\nabla_{\pmb{\xi}}\log q^{\pmb{\xi}}(\mathbf{s}_{1:T})\Bigg].}\end{array}
$$  

We recognize the importance weights $\begin{array}{r}{w(\mathbf{s}_{1:T})=\frac{\tilde{\sigma}(\mathbf{s}_{1:T})}{q^{\pmb{\xi}}(\mathbf{s}_{1:T})}}\end{array}$   from  Eq. (3) .  Go et al.  ( 2023 ) consider estimating  Eq. (63)  using a moving average estimate of the partition function $\hat{Z}_{\sigma}$  

$$
\nabla_{\pmb{\xi}}D_{\mathrm{KL}}\big(\sigma(\mathbf{s}_{1:T})\,\big\|\,q^{\pmb{\xi}}(\mathbf{s}_{1:T})\big)\ \approx\ \sum_{k=1}^{K}\frac{1}{\hat{Z}_{\sigma}}w(\mathbf{s}_{1:T}^{(k)})\nabla_{\pmb{\xi}}\log q^{\pmb{\xi}}(\mathbf{s}_{1:T}^{(k)}),
$$  

Alternatively, the expectation may thus be estimated using SIS with the variational policy $q^{\pmb{\xi}}(\mathbf{s}_{1:T})$ . Using self-normalized importance sampling (SNIS) to estimate  Eq. (63)  as in  Eq. (5)  corresponds to $\begin{array}{r}{\hat{Z}_{\sigma}=\sum_{j=1}^{K}w(\mathbf{s}_{1:T}^{(k)})}\end{array}$ , with  

$$
\nabla_{\pmb{\xi}}D_{\mathrm{KL}}\big(\sigma(\mathbf{s}_{1:T})\,\big\|\,q^{\pmb{\xi}}(\mathbf{s}_{1:T})\big)\ \approx\ \sum_{k=1}^{K}\frac{w(\mathbf{s}_{1:T}^{(k)})}{\sum_{j=1}^{K}w(\mathbf{s}_{1:T}^{(j)})}\nabla_{\pmb{\xi}}\log q^{\pmb{\xi}}(\mathbf{s}_{1:T}^{(k)}).
$$  

We use this gradient for DPG proposal learning in the main text experiments, although we use the parameter iz ation described in  Eq. (DPG)  below.  

DPG as Sequential Maximum Likelihood Objective We now show  Eq. (64)  is equivalent to a sequential maximum likelihood EBM objective. Consider minimizing the KL divergence,  

$$
D_{\mathrm{KL}}\big(\boldsymbol{\sigma}(\mathbf{s}_{1:T})\parallel q^{\pmb{\xi}}(\mathbf{s}_{1:T})\big)=\sum_{t=1}^{T}\mathbb{E}_{\boldsymbol{\sigma}(\mathbf{s}_{1:t-1})}D_{\mathrm{KL}}\Big(\boldsymbol{\sigma}(s_{t}|\mathbf{s}_{1:t-1})\Big\|\,q_{t}^{\pmb{\xi}}(s_{t}|\mathbf{s}_{1:t-1})\Big)
$$  

While this is reminscent of the twist-induced proposal in  Prop. 3.3 , we emphasize distinctions between energy-based learning of the proposal (DPG) versus energy-based learning of twist functions (CTL) in  App. E.3.1 .  

The gradient of  Eq. (EBM proposal learning)  becomes  

$$
\nabla_{\pmb{\xi}}D_{\mathrm{KL}}\Big(\sigma(\mathbf{s}_{1:T})\,\Big|\,\Big|\,q^{\pmb{\xi}}(\mathbf{s}_{1:T})\Big)=\sum_{t=1}^{T}\mathbb{E}_{\sigma(\mathbf{s}_{1:t-1})}\Big[\mathbb{E}_{\sigma(s_{t}|\mathbf{s}_{1:t-1})}\big[\nabla_{\pmb{\xi}}\log\psi_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t})\big]-\mathbb{E}_{q_{t}^{\pmb{\xi}}(s_{t}|\mathbf{s}_{1:t-1})}\big[\nabla_{\pmb{\xi}}\log\psi_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t-1})\big]\Big].
$$  

Starting from  Eq. (64) , we now seek to recover  Eq. (66) . Using  Eq. (65) , we can write  

$$
\begin{array}{r l}&{\quad\log q^{\pmb{\xi}}(\mathbf{s}_{1:T}^{(k)})=\displaystyle\sum_{t=1}^{T}\Big(\log p_{0}\big({s}_{t}^{(k)}|\mathbf{s}_{1:t-1}^{(k)}\big)+\log\psi_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t}^{(k)})-\log\displaystyle\sum_{s_{t}}p_{0}\big(s_{t}|\mathbf{s}_{1:t-1}^{(k)}\big)\psi_{t}^{\pmb{\xi}}\big(s_{t},\mathbf{s}_{1:t-1}^{(k)}\big)\Big)}\\ &{\quad^{\prime}\mathbf{\varepsilon}_{\pmb{\xi}}\log q^{\pmb{\xi}}(\mathbf{s}_{1:T}^{(k)})=\displaystyle\sum_{t=1}^{T}\biggl(\nabla_{\pmb{\xi}}\log\psi_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t}^{(k)})-\mathbb{E}_{q_{t}^{\pmb{\xi}}(s_{t}|\mathbf{s}_{1:t-1}^{(k)})}\Big[\nabla_{\pmb{\xi}}\log\psi_{t}^{\pmb{\xi}}\big(s_{t},\mathbf{s}_{1:t-1}^{(k)}\big)\Big]\biggr)}\end{array}
$$  

Substituting into  Eq. (64) , we recover  

$$
\begin{array}{r}{\displaystyle\kappa_{\mathrm{KL}}\left(\boldsymbol{\sigma}(\mathbf{s}_{1:T})\,\big\|\,q^{\xi}(\mathbf{s}_{1:T})\right)\;\approx\;\displaystyle\sum_{k=1}^{K}\frac{w(\mathbf{s}_{1:T}^{(k)})}{\sum_{j=1}^{K}w(\mathbf{s}_{1:T}^{(k)})}\sum_{t=1}^{T}\Bigl(\nabla_{\xi}\log\psi_{t}^{\xi}(\mathbf{s}_{1:t}^{(k)})-\mathbb{E}_{q_{t}^{\xi}(s_{t}|\mathbf{s}_{1:t-1}^{(k)})}\Bigl[\nabla_{\xi}\log\psi_{t}^{\xi}(s_{t},T)\Bigr]\,.}\end{array}
$$  

which is an SNIS estimate of the maximum likelihood EBM gradient in  Eq. (66) , as desired. Note that the expectation over $q_{t}^{\pmb{\xi}}(s_{t}|\mathbf{s}_{1:t-1}^{(k)})$ |  can be calculated exactly.  

Comparison with CTL Objective The gradient in  Eq. (DPG)  above appears similar to our CTL objective and gradient in Sec. 4.1 . However, the DPG loss in  Eq. (EBM proposal learning)  is a single (joint) KL divergence over the entire sequence, whereas CTL optimizes $T$  separate KL divergences for each intermediate marginal.  

For the DPG gradient in  Eq. (66) , negative sampling is performed using a ‘positive’ prefix $\mathbf{s}_{1:t-1}^{(k)}\sim\sigma(\mathbf{s}_{1:t-1})$ −   −  and an  exact ‘negative’ sample from the one-step-ahead $q_{t}^{\pmb{\xi}}(s_{t}|\mathbf{s}_{1:t-1}^{(k)})$  ( Eq. (65) , which we have assumed to be tractable). In practice, we − obtain the prefixes using the truncation of exact samples or approximate positive sampling with the final target weights as in Eq. (DPG) . By contrast, the CTL gradient in  Eq. (22)  involves  approximate  negative sampling under each $\pi_{t}(\mathbf{s}_{1:t})$ .  

E.3.1. N AIVE  U SE OF  P ROPOSAL  L EARNING TO DEFINE  T WISTED  SMC T ARGETS  

While we have shown in  Prop. 3.3  how one-step proposals   $\{q_{t}^{\pi}(s_{t}|\mathbf{s}_{1:t-1})\}_{t=1}^{T}$ −   can be induced from a given set of twist functions   $\{\psi_{t}(\mathbf{s}_{1:t})\}_{t=1}^{T}$   or target distributions   $\{\pi_{t}(\mathbf{s}_{1:t})\}_{t=1}^{T}$ , we now emphasize that moving the other direction (inducing intermediate twisting targets from a proposal learning scheme parameterized by   $\{\psi_{t}^{\pmb\xi}\}_{t=1}^{T})$   } ) does not yield the correct intermediate targets for resampling ( App. A.1 ), even at optimality in the proposal learning objective.  

We focus our arguments on learning with the EBM maximum likelihood objective in  Eq. (EBM proposal learning)  as an example. The proposal energies $\psi_{t}^{\mathbf{\tilde{\xi}}}(\mathbf{s}_{1:t})$  appear to play a role analogous to the twist function $\psi_{t}(\mathbf{s}_{1:t})$  in the one-step proposal induced from twist targets   $\{\pi_{t}\}_{t=1}^{T}$   in  Sec. 3 .  

However, we proceed to show in  Prop. E.2  that naive use of $\psi_{t}^{\pmb{\xi}}$   to define twisting targets using  11  

$$
\begin{array}{r}{\pi_{t}^{\pmb{\xi}}\big(\mathbf{s}_{1:t}\big)=\left\{\begin{array}{l l}{\frac{1}{\mathcal{Z}_{t}^{\psi}}\ p_{0}\big(\mathbf{s}_{1:t}\big)\ \psi_{t}^{\pmb{\xi}}\big(\mathbf{s}_{1:t}\big)\quad}&{t\neq T}\\ {\frac{1}{\mathcal{Z}_{\sigma}}\ p_{0}\big(\mathbf{s}_{1:T}\big)\ \phi\big(\mathbf{s}_{1:T}\big)\quad}&{t=T}\end{array}\right.}\end{array}
$$  

need not lead to an SMC procedure for which   $\pi_{t}^{\pmb\xi}({\bf s}_{1:t})=\sigma({\bf s}_{1:t})$ , even if   $q_{t}^{\pmb{\xi}}(s_{t}|\mathbf{s}_{1:t-1})=\sigma(s_{t}|\mathbf{s}_{1:t-1})$ | |  for all   $t$ . We thus − − argue that $\psi_{t}^{\pmb{\xi}}$   learned using  Eq. (EBM proposal learning)  should not be used  as target twists in  Eq. (67) , since they do not yield the optimal interemdiate target distributions at optimality ( App. A.1 ).  

We begin by showing a simple lemma for the one-step conditionals in  Eq. (EBM proposal learning) .  

Lemma E.1.  Any twist induced proposal   $q_{t}^{\pmb{\xi}}(s_{t}|\mathbf{s}_{1:t-1})$ |  (induced by $\psi_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t}),$ ) is invariant to rescaling   $\psi_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t})$  by an − arbitrary constant $c(\mathbf{s}_{1:t-1})$  with respect to $\mathbf{S}_{1:t-1}$ ,  

$$
\psi_{t}^{\pmb{\xi}c}(\mathbf{s}_{1:t}):=c(\mathbf{s}_{1:t-1})\psi_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t})
$$  

Proof.  

$$
\mathbf{\xi}_{\cdot1})=\frac{p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\mathbf{\xi}_{t}}(\mathbf{s}_{1:t})}{\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\mathbf{\xi}_{t}}(\mathbf{s}_{1:t})}=\frac{p_{0}(s_{t}|\mathbf{s}_{1:t-1})c(\mathbf{s}_{1:t-1})\psi_{t}^{\mathbf{\xi}}(\mathbf{s}_{1:t})}{\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})c(\mathbf{s}_{1:t-1})\psi_{t}^{\mathbf{\xi}}(\mathbf{s}_{1:t})}=\frac{p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\mathbf{\xi}}(\mathbf{s}_{1:t})}{\sum_{s_{t}}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\mathbf{\xi}}(\mathbf{s}_{1:t})}=q_{0},
$$  

Proposition E.2.  There exist   $\{\psi_{t}^{\pmb\xi}^{*}\}_{t=1}^{T}$   }   such that  $(i)\,q_{t}^{\pmb{\xi}^{*}}(s_{t}|\mathbf{s}_{1:t-1})=\sigma\bigl(s_{t}\bigl|\mathbf{s}_{1:t-1}\bigr)$ | − | −  and (ii) the SMC targets   $\{\pi_{t}^{\pmb{\xi}*}(\mathbf{s}_{1:t})\}_{t=1}^{T}$ } induced by $\{\psi_{t}^{\pmb\xi}^{*}\}_{t=1}^{T}$   }   via  Eq.  (67)  are different from   $\sigma(\mathbf{s}_{1:t})$ .  

Proof.  To satisfy condition (i) of the current proposition, we define  

$$
\psi_{\tau}^{\pmb{\xi}^{*}}(\mathbf{s}_{1:\tau}):=\left\{\begin{array}{r l r}{\sum_{\mathbf{s}_{\tau+1:T}}p_{0}(\mathbf{s}_{\tau+1:T}|\mathbf{s}_{1:\tau})\phi(\mathbf{s}_{1:T})}&{}&{\tau\neq t}\\ {c(\mathbf{s}_{1:t-1})\;\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi(\mathbf{s}_{1:T})}&{}&{\tau=t}\end{array}\right.
$$  

which for all   $\tau$ , yields optimal proposals:  ( $(i)\;q^{\xi*}(s_{\tau}|\mathbf{s}_{1:\tau-1})=\sigma(s_{\tau}|\mathbf{s}_{1:\tau-1})\propto p_{0}(s_{\tau}|\mathbf{s}_{1:\tau-1})\psi_{\tau}^{\xi*}(\mathbf{s}_{1:\tau})$ | |  ∝ |  via  Lemma E.1 . − − − However, it is clear that   $c(\mathbf{s}_{1:t-1})\neq1$  can break the necessary condition for optimality of SMC sampling that $\pi_{t}(\mathbf{s}_{1:t})=$ $\sigma(\mathbf{s}_{1:t})$  ( Prop. A.4 ). In particular, consider  

$$
\begin{array}{r l}&{\pi_{t}^{\pmb{\xi}^{*}}({\bf s}_{1:t})=\frac{1}{\mathcal{Z}_{t}^{\psi}}\,p_{0}({\bf s}_{1:t})\,\psi_{t}^{\pmb{\xi}^{*}}({\bf s}_{1:t})=\frac{1}{\mathcal{Z}_{t}^{\psi}}\,c({\bf s}_{1:t-1})p_{0}({\bf s}_{1:t})\displaystyle\sum_{{\bf s}_{t+1:T}}p_{0}({\bf s}_{t+1:T}|{\bf s}_{1:t})\phi({\bf s}_{1:T})}\\ &{\phantom{m m m m m m m m m}=\frac{1}{\mathcal{Z}_{t}^{\psi}}c({\bf s}_{1:t-1})\tilde{\sigma}({\bf s}_{1:t})\neq\sigma({\bf s}_{1:t})}\end{array}
$$  

for   $c(\mathbf{s}_{1:t-1})\neq1$ , which introduces an additional factor which depends on $\mathbf{s}_{1:t}$ . Thus, the twist target   $\pi_{t}^{\pmb{\xi}^{*}}(\mathbf{s}_{1:t})$  induced from $\psi_{t}^{\pmb{\xi}^{*}}(\mathbf{s}_{1:t})$  in  Eq. (69)  is not equal to the desired marginal   $\sigma(\mathbf{s}_{1:t})$ , despite the fact that all proposals are optimal.  

We indeed observed experimentally that resampling based on  Eq. (67)  after training using  Eq. (EBM proposal learning) could lead to  worse  SMC   $\log\mathcal{Z}_{\sigma}$  bounds than simply calculating the SIS or IWAE bound with $\begin{array}{r}{\dot{\prod}_{t=1}^{T}q_{t}^{\pmb{\xi}}\dot{\left(s_{t}\middle|\mathbf{\dot{s}}_{1:t-1}\right)}}\end{array}$ | . −  

Optimality in CTL Objective implies Optimal Twisted SMC In contrast to  Prop. E.2 , note that the global optimum of our CTL objective $\begin{array}{r}{\operatorname*{min}\sum_{t=1}^{T}D_{\mathrm{KL}}\Big(\sigma\big(\mathbf{s}_{1:t}\big)\,\Big\|\,\pi_{t}^{\psi}\big(\mathbf{s}_{1:t}\big)\Big)}\end{array}$  (which occurs for the optimal twists   $\{\psi_{t}^{*}\}_{t=1}^{T-1}$   }   in  Prop. 3.2 ), results in both the twist-induced proposal $q_{t}^{\pi^{*}}(s_{t}|\mathbf{s}_{1:t-1})=\sigma(s_{t}|\mathbf{s}_{1:t-1})$ | |  and the twisting targets   $\pi_{t}^{*}(\mathbf{s}_{1:t})=\sigma(\mathbf{s}_{1:t})$  satisfying the − − necessary and sufficient conditions for optimality outlined in  App. A.1 Prop. A.3 .  

E.3.2. SMC  WITH  N ORMALIZED  T ARGETS  I NDUCED BY  L EARNED  P ROPOSAL  L EADS TO  U NIFORM  W EIGHTS The issue in  Prop. E.2  arises from the degree of freedom   $c(\mathbf{s}_{1:t-1})$  in the normalization constant of the one-step proposal. To avoid this, we can instead define  normalized  twisted intermediate targets using  

$$
\tilde{\pi}_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t})=\left\{\begin{array}{l l}{p_{0}(\mathbf{s}_{1:t})\,\displaystyle\prod_{\tau=1}^{t}\frac{\psi_{\tau}^{\pmb{\xi}}(\mathbf{s}_{1:\tau})}{Z_{\tau}^{\pmb{\xi}}(\mathbf{s}_{1:\tau-1})}}&{=\displaystyle\prod_{\tau=1}^{t}q_{\tau}^{\pmb{\xi}}\big(s_{\tau}|\mathbf{s}_{1:\tau-1}\big)\qquad t\neq T}\\ {p_{0}(\mathbf{s}_{1:T})\,\,\phi\big(\mathbf{s}_{1:T}\big)}&{t=T}\end{array}\right.
$$  

where $Z_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t-1})$  arises from the proposal $\begin{array}{r}{q_{t}^{\pmb\xi}\big(s_{t}|\mathbf{s}_{1:t-1}\big):=\frac{1}{Z_{t}^{\pmb\xi}\left(\mathbf{s}_{1:t-1}\right)}p_{0}\big(s_{t}|\mathbf{s}_{1:t-1}\big)\psi_{t}^{\pmb\xi}\big(\mathbf{s}_{1:t}\big)}\end{array}$  learned according to  Eq. (EBM − proposal learning) .  

Crucially, $\tilde{\pi}_{t}^{\pmb\xi}$   in    (71)  are automatically normalized for   $t\neq T$ , as the product of normalized proposals. In this case, SMC resampling with $q^{\pmb\xi}$   or the twist-induced proposal yields uniform resampling weights,  

$$
\begin{array}{r l}&{\gamma):\ w_{t}(\mathbf{s}_{1:t})=\frac{\tilde{\pi}_{t}^{\xi}(\mathbf{s}_{1:t})}{\tilde{\pi}_{t-1}^{\xi}(\mathbf{s}_{1:t-1})q^{\xi}(s_{t}|\mathbf{s}_{1:t-1})}=\frac{p_{0}(\mathbf{s}_{1:t})\displaystyle\prod_{r=1}^{t}\frac{\psi_{t}^{\xi}(\mathbf{s}_{1:r})}{Z_{t}^{\xi}(\mathbf{s}_{1:r-1})}}{p_{0}(\mathbf{s}_{1:t-1})\left(\displaystyle\prod_{r=1}^{t-1}\frac{\psi_{t}^{\xi}(\mathbf{s}_{1:r-1})}{Z_{t}^{\xi}(\mathbf{s}_{1:r-1})}\right)\frac{1}{Z_{t}^{\xi}(\mathbf{s}_{1:t-1})}p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\xi}(\mathbf{s}_{1:t})}}\end{array}
$$  

Although we were able to construct well-behaved intermediate twisting targets from a proposal-learning scheme $q_{t}^{\pmb{\xi}}(s_{t}|\bar{\mathbf{s}_{1:t-1}})\propto p_{0}(s_{t}|\mathbf{s}_{1:t-1})\psi_{t}^{\pmb{\xi}}(\mathbf{s}_{1:t})$ | −  ∝ − ,  Eq. (72)  shows that this  does not lead to meaningful intermediate SMC resampling . In other words, for  t < T , the marginal distributions of SMC samples   $\mathbf{s}_{1:t}^{k}$   with this scheme are simply   $q^{\pmb{\xi}}(\mathbf{s}_{1:t})$ , the same as we would obtain with no resampling (SIS/IWAE).  

# F. Bidirectional SMC  

In this section, we recall the extended state-space probabilistic interpretation of SMC from ( Maddison et al. ,  2017 ;  Andrieu et al. ,  2010 ). The idea is to define an unnormalized target distribution   $\sigma_{\mathrm{SMC}}(S)$  and normalized proposal   $q_{\mathrm{SMC}}(\pmb{S})$  over an extended state space   $S$  containing all random variables relevant to SMC sampling and importance weighting with   $K$ sequences of length $T$ . Defining $\tilde{\boldsymbol{\sigma}}_{\mathrm{SMC}}(\boldsymbol{S})$  such that its  rmalization constant matches $\mathcal{Z}_{\sigma}$ , we can use simple imp ance sampling (SIS) in this extended state space to show that  K -sequence SMC sampling yields an unbiased estimator of  Z $\mathcal{Z}_{\sigma}$ , for example   $\mathcal{Z}_{\sigma}=\mathbb{E}_{q_{\mathrm{SMC}}(S)}[\frac{\tilde{\sigma}_{\mathrm{SMC}}(S)}{q_{\mathrm{SMC}}(S)}]$  (as in  Eq. (8) ). Our end goal is to use this probabilistic interpretation to derive the lower and upper bounds on   $\log\mathcal{Z}_{\sigma}$  in  Prop. 5.1 , following  Brekelmans et al.  ( 2022 ) App. A.  

We define the extended state space proposal and target distributions below, noting that our bounds will require sampling from normalized $\sigma_{\mathrm{SMC}}(S)$  or $q_{\mathrm{SMC}}(\pmb{S})$ , and evaluating $\tilde{\boldsymbol{\sigma}}_{\mathrm{SMC}}(\boldsymbol{S})$  and   $q_{\mathrm{SMC}}(\pmb{S})$ . We summarize the algorithm for sampling $\sigma_{\mathrm{SMC}}(S)$  in  Alg. 2 , using concatenation notation for simplicity instead of the probabilistic interpretation using index histories in the text.  

Single-Sequence Target and Proposal We construct our importance sampling bounds with the goal of estimating the (log) partition function and sampling from a target distribution   $\sigma(\mathbf{s}_{1:T})=\tilde{\sigma}(\mathbf{s}_{1:T})/\mathcal{Z}_{\sigma}$ Z . We leverage a sequence of intermediate tributions,   $\begin{array}{r}{\{\pi_{t}(\mathbf{s}_{1:t})\,=\,\frac{1}{\mathcal{Z}_{t}}\tilde{\pi}_{t}\bigl(\mathbf{s}_{1:t}\bigr)\}_{t=1}^{T}}\end{array}$   over partial sequences, with the final target   $\pi_{T}(\mathbf{s}_{1:T})\,=\,\sigma(\mathbf{s}_{1:T})$  and Z t $\mathcal{Z}_{T}=\mathcal{Z}_{\sigma}$ Z  Z . We assume $\tilde{\pi}_{0}(\mathbf{s}_{0})=1$  for $\mathcal{Z}_{0}=1$ . Finally, our bounds and sampling procedures also depend on a given set of proposal distribution  { $\{q(s_{t}\,|\,\mathbf{s}_{1:t-1})\}_{t=1}^{T}$  | − } , as in  Sec. 2.2 .  

Extended State Space Random Variables Consider an extended state space $s$  containing $K T$  tokens $\{s_{t}^{k}\}_{t=1,k=1}^{T,K}$     with $s_{t}^{k}\in\mathcal{V}$ ∈V  and $K T$  indexing random variables   $\{\omega_{t}^{k}\}_{t=1,k=1}^{T,K}$   with $\omega_{t}^{k}\in[1,K]$ ∈ , to represent the results of resampling ( Eq. (7) ),  

$$
\boldsymbol{S}:=\left\{s_{t}^{k},\omega_{t}^{k}\right\}_{t=1,k=1}^{T,K}
$$  

For ease of notation (and similarly to  Maddison et al.  ( 2017 );  Andrieu et al.  ( 2010 )), we call attention to our use of recursive backtracking index operations to collect sequences $\{\mathbf{s}_{1:t}\}$  based on the results of resamplin $\{\omega_{t}^{k}\}$   } . We use  lists  of index histories to construct sequences of tokens, with two recursive definitions of histories. Letting  +  indicate appending of lists,  

$$
\begin{array}{r l}{\pmb{h}_{0}^{k}:=[]\ \ \forall k,\quad}&{\pmb{h}_{t}^{k}:=\pmb{h}_{t-1}^{\omega_{t}^{k}}+[\omega_{t}^{k}]}\\ {\bar{\pmb{h}}_{0}^{k}:=[]\ \ \forall k,\quad}&{\bar{\pmb{h}}_{t}^{k}:=\pmb{h}_{t-1}^{k}+[k]}\end{array}
$$  

(Index Notation)  

For example, the history $h_{t-1}^{k}$   will be used to construct prefix sequences $\mathbf{s}_{1:t-1}^{h_{t-1}^{k}}$   (i.e. lists of tokens) for sampling a next − − token $s_{t}^{k}$   . We denote sequences of tokens with the index history in the superscript and also expand the definition for clarity,  

$$
\begin{array}{r l r}{\mathbf{s}_{1:t}^{h_{t}^{k}}:=\mathbf{s}_{1:t-1}^{h_{t-1}^{\omega_{t}^{k}}}+[s_{t}^{\omega_{t}^{k}}]}&{}&{=[s_{1}^{h_{t-1}^{\omega_{t}^{k}}[1]},\dots,s_{t-1}^{h_{t-1}^{\omega_{t}^{k}}[t-1]},s_{t}^{\omega_{t}^{k}}]=[s_{1}^{\omega_{1}^{\omega_{t}^{k}}},\dots,s_{t-2}^{\omega_{t}^{\omega_{t}^{k}}},s_{t-1}^{\omega_{t-1}^{\omega_{t}^{k}}},s_{t}^{\omega_{t}^{k}}]}\\ {\bar{\mathbf{s}}_{1:t}^{h_{t}^{k}}:=\mathbf{s}_{1:t-1}^{h_{t-1}^{k}}+[s_{t}^{k}]}&{}&{\mathrm{(Sedsucccurlyeqno~NOtt)}}\end{array}
$$  

In the second line, we define $\mathbf{s}_{1:t}^{\bar{h}_{t}^{k}}$   as a sequence of length $t$  which concatenates the prefix $\mathbf{s}_{1:t}^{h_{t-1}^{k}}$ with next token   $s_{t}^{k}$   . The notation   $\mathbf{s}_{1:t}^{\bar{h}_{t}^{k}}$   represents partial sequences  before  resampling. By contrast, we will use the notation $\mathbf{s}_{1:t}^{h_{t}^{k}}$   in the first line of Eq. (Sequence Notations)  to refer to sequences  after  resampling.  

Consider the sequence   $\mathbf{s}_{1:t}^{\bar{h}_{t}^{i}}$   in a particular index   $i\in[1,K]$  before  resampling. Resampling at time   $t$  may result in choosing $\omega_{t}^{k}=i$  for some   $k$ . Using the first line, we see that $\begin{array}{r}{\mathbf{s}_{1:t}^{h_{t}^{k}}=\mathbf{s}_{1:t-1}^{h_{t}^{\omega_{t}^{k}}}+[s_{t}^{\omega_{t}^{k}}]=\mathbf{s}_{1:t-1}^{h_{t-1}^{i}}+[s_{t}^{i}]}\end{array}$  for those indices such that $\omega_{t}^{k}=i$ . − − Indeed, this matches the definition of   $\mathbf{s}_{1:t}^{\bar{h}_{t}^{i}}=\mathbf{s}_{1:t-1}^{h_{t-1}^{i}}+[s_{t}^{i}]$  in the second line (before resampling). Thus, the indexing notation − in  Eq. (Sequence Notations)  reflects resampling or cloning of sequences $\mathbf{s}_{1:t}^{\bar{h}_{t}^{i}}$   into the indices such that $\omega_{t}^{k}=i$ , which yields prefixes $\mathbf{s}_{1:t}^{h_{t}^{k}}$   for the next step of sampling  $(t+1)$  in each index $k\in[1,K]$ .  

![](images/698a4fb6dc55e64ceefe11fd5e867fb5f8a987c91114005f6c85650bfdc84636.jpg)  

$^T=2$  

![](images/82812a13de7e7fc43d36a2ea7411140d7c1f9684baf6e5446088e973024f2674.jpg)  

${\cal T}=2$  

Figure 4: Graphical Models for extended state- space proposal and target distributions which result in the bidirectional SMC bounds. We show density evaluation in the proposal and target for a fixed set of   $\{s_{t}^{k},\omega_{t}^{k}\}_{k=1,t=1}^{3,\hat{2}}$   . We let the size of the circles reflect the (hypothetical) importance weights of sequences $\dot{\mathbf{s}}_{1:t}^{\bar{h}_{t}^{k}}$   and $\omega_{t}^{k}$ reflect the (hypothetical) results of resampling with these weights. In   $(b)$ , we assume fixed $j_{T+1}=j_{3}=1$  as in the text, with   $\omega_{2}^{1}=2$ .  

Algorithm 2  (Twisted) SMC Target Sampling  $(\sigma_{\mathrm{SMC}})$ (blue indicates changes from SMC proposal algorithm; $\mathfrak{s}_{1:T}$  is an exact posterior sample)  

$$
\begin{array}{r}{\mathbf{SMC}\mathbf{-TARGET}\Big(p_{0},q,\{\psi_{t}\}_{t=1}^{T-1},\phi,K,\{t_{r}\}_{t=1}^{R-1},t_{0}=0,t_{R}=T,\mathfrak{s}_{1:T}\Big)}\end{array}
$$  

$s_{t}^{k}\gets\mathfrak{s}_{t}$  

$$
\begin{array}{r l}&{\mathbf{s}_{1:t}^{k}\leftarrow\mathsf{c o n c a t}\big(\mathbf{s}_{1:t-1}^{k},s_{t}^{k}\big)}\\ &{\mathbf{if}\ t<T\,\mathbf{th}\mathbf{en}}\\ &{\quad w_{t}^{k}\leftarrow\frac{p_{0}\big(\mathbf{s}_{t}^{k}\,\big|\,\mathbf{s}_{1:t-1}^{k}\big)}{q\big(\mathbf{s}_{t}^{k}\,\big|\,\mathbf{s}_{1:t-1}^{k}\big)}\frac{\psi_{t}\big(\mathbf{s}_{1:t}^{k}\big)}{\psi_{t-1}\big(\mathbf{s}_{1:t-1}^{k}\big)}}\\ &{\mathbf{else}}\\ &{\quad w_{t}^{k}\leftarrow\frac{p_{0}\big(\mathbf{s}_{t}^{k}\,\big|\,\mathbf{s}_{1:t-1}^{k}\big)}{q\big(\mathbf{s}_{t}^{k}\,\big|\,\mathbf{s}_{1:t-1}^{k}\big)}\frac{\phi\big(\mathbf{s}_{1:t}^{k}\big)}{\psi_{t-1}\big(\mathbf{s}_{1:t-1}^{k}\big)}}\end{array}
$$  

$$
\begin{array}{r}{\mathbf{\Sigma}_{\omega_{t}^{k}}^{\sim}\sim\mathsf{c a t}\left(\left\{\frac{\prod_{t=t_{r-1}+1}^{t_{r}}w_{t}^{i}}{\sum_{j=1}^{K}\prod_{t=t_{r-1}+1}^{t_{r}}w_{t}^{j}}\right\}_{i=1}^{K}\right)}\\ {\mathbf{s}_{1:t}^{k}\gets\mathbf{s}_{1:t}^{\omega_{t}^{k}}}\end{array}
$$  

return $\begin{array}{r l}&{\dot{\left\{\mathbf{s}_{1:T}^{k},\prod_{t=t_{R-1}+1}^{T}w_{t}^{k}\right\}}_{k=1}^{K}}\\ &{\dot{\mathcal{Z}}_{\sigma}^{\scriptscriptstyle\mathrm{SMC}}=\prod_{r=1}^{R}\frac{1}{K}\sum_{k=1}^{K}\prod_{t=t_{r-1}+1}^{t_{r}}w_{t}^{k}}\end{array}$  

Extended State Space Proposal Distribution Sampling from the extended state space proposal corresponds to the procedure described in  Sec. 2.2  and Alg. 1, which we write as 12  

$$
\begin{array}{r}{q_{\mathtt{S M C}}\Big(\{s_{t}^{k},\omega_{t}^{k}\}_{t=1,k=1}^{T,K}\Big):=\overset{T}{\underset{k=1}{\prod}}\left[\overset{K}{\underset{k=1}{\prod}}q\Big(s_{t}^{k}\,\bigg|\,\mathbf{s}_{1:t-1}^{h_{t-1}}\Big)\underset{k=1}{\prod}\,q\big(\omega_{t}^{k}\,\big|\,\mathbf{s}_{1:t}^{1:K}\big)\right]\qquad\qquad\qquad\mathrm{(SMC~Exponential)}}\\ {\mathrm{where}\;\forall\,k,\quad q\big(\omega_{t}^{k}=i\,\big|\,\mathbf{s}_{1:t}^{1:K}\big):=\frac{\bar{\pi}_{t}\Big(\mathbf{s}_{1:t}^{\bar{h}_{t}^{k}}\Big)}{\bar{\pi}_{t-1}\Big(\mathbf{s}_{1:t-1}^{h_{t-1}^{k}}\Big)q\Big(s_{t}^{k}\,\big|\,\mathbf{s}_{1:t-1}^{h_{t-1}^{k}}\Big)}=\frac{w_{t}\Big(\mathbf{s}_{1:t}^{\bar{h}_{t}^{k}}\Big)}{\sum_{k=1}^{K}w_{t}\Big(\mathbf{s}_{1:t}^{\bar{h}_{t}^{k}}\Big)}}\\ {\mathrm{where}\;\forall\,k,\quad q\big(\omega_{t}^{k}=i\,\big|\,\mathbf{s}_{1:t}^{1:K}\big):=\frac{\tilde{\pi}_{t}\Big(\mathbf{s}_{1:t}^{\bar{h}_{t}^{k}}\Big)}{\sum_{k=1}^{K}\,\frac{\tilde{\pi}_{t}\Big(\mathbf{s}_{1:t-1}^{\bar{h}_{t-1}^{k}}\Big)}{\tilde{\pi}_{t-1}\Big(\mathbf{s}_{1:t-1}^{\bar{h}_{t-1}^{k}}\Big)q\Big(s_{t}^{k}\,\big|\,\mathbf{s}_{1:t-1}^{h_{t-1}^{k}}\Big)}=\frac{w_{t}\Big(\mathbf{s}_{1:t}^{\bar{h}_{t}^{k}}\Big)}{\sum_{k=1}^{K}w_{t}\Big(\mathbf{s}_{1:t}^{\bar{h}_{t}^{k}}\Big)}}\end{array}
$$  

To recount the description above, note that the next token $s_{t}^{i}$   in index $i$  is sampled from the proposal, conditioned on the prefix   $\mathbf{s}_{1:t-1}^{h_{t-1}^{i}}$ . We concatenate these tokens $\mathbf{s}_{1:t}^{\bar{h}_{t}^{i}}=\mathbf{s}_{1:t-1}^{h_{t-1}^{i}}+[s_{t}^{i}]$  (  Eq. (Sequence Notations) ) and calculate importance − − weights. We perform resampling in each index $k$  according to   $q(\omega_{t}^{k}|\mathbf{s}_{1:t}^{1:K})$   | , or SNIS with the calculated weights (as in Eq. (7) ). Finally, after resampling, we clone the sequence in the chosen index $\omega_{t}^{k}$   into index   $k$  and proceed to sample   $s_{t+1}^{k}$ with an prefix defined by the indices $\pmb{h}_{t}^{k}=\pmb{h}_{t-1}^{\omega_{t}^{k}}+[\omega_{t}^{k}]$ .  

Worked Example:  To make this more concrete, we provide a worked example of the procedure in  Fig. 4  (a). At step $t=1$ , we resample the token   $s_{t=1}^{k=2}$   twice (for indices $k=1,3)$ ), with   $\omega_{1}^{1}=\omega_{1}^{3}=2$  (and in index  2 , set   $\omega_{1}^{2}=3$  to sample $s_{1.}^{3.}$ ). We record the prefix history as, for example,   $h_{1}^{1}=h_{1}^{3}=[\omega_{1}^{1}]=[2]$ , which corresponds to $\mathbf{s}_{1}^{h_{1}^{1}}=s_{1}^{2}$ .  

At step 2 in (a), we proceed to sample   $s_{2}^{1}\,\sim\,q\bigl(s_{2}|\mathbf{s}_{1}^{h_{1}^{1}}\,=\,[s_{1}^{2}]\bigr)$    (and similarly   $s_{2}^{3}\,\sim\,q(s_{2}|\mathbf{s}_{1}^{\mathbf{h}_{1}^{3}}\,=\,[s_{1}^{2}]))$   ), whereas   $s_{2}^{2}\sim$   ∼ $q\big(s_{2}|\mathbf{s}_{1}^{h_{1}^{1}}\,=\,[s_{1}^{3}]\big)$ . We next evaluate the importance weights over three concatenated sequences:   $\mathbf{s}_{1}^{\bar{h}_{1}^{1}}\,=\,[s_{1}^{2}]+[s_{2}^{1}]$ , $\mathbf{s}_{1}^{\bar{h}_{1}^{2}}=[s_{1}^{3}]+[s_{2}^{2}]$ , and   $\mathbf{s}_{1}^{\bar{h}_{1}^{3}}=[s_{1}^{2}]+[s_{2}^{3}]$ , emphasizing that $s_{2}^{k}$   is the final token in each index. Shown in the red circles, we proceed to resample   $\omega_{2}^{1}=2,\omega_{2}^{2}=3$ ,  and $\omega_{2}^{3}=2$  at step $t=2$ .  

Finally, we need to backtrack to obtain the history of the indices for the sequence to be cloned in resampling. Namely, for index  1  where $\omega_{t=2}^{k=1}=2$ , we concatenate   $h_{1}^{\omega_{2}^{1}}+[\omega_{2}^{1}]=h_{1}^{2}+[2]=[3,2]=:h_{2}^{1}$   (i.e. the history for time 2, index 1). This list of indices specifies the prefix   $\mathbf{s}_{1:2}^{h_{2}^{1}}=[s_{1}^{3},s_{2}^{2}]$  at step $t=3$ , index $k=1$ . Similar reasoning applies for other indices.  

Extended State Space Target We are finally ready to specify the extended state space target distribution. The crucial difference is to identify a single sequence $\mathbf{s}_{1:T}^{h_{T}^{1}}$   of length $T$  (the choice of index 1 is arbitrary). This sequence $\mathbf{s}_{1:T}^{h_{T}^{1}}$   will be evaluated under the unnormalized target distribution   $\tilde{\pi}_{T}(\mathbf{s}_{1:T})=\tilde{\sigma}(\mathbf{s}_{1:T})$  or exactly sampled from the target $\mathbf{s}_{1:T}^{h_{T}^{1}}\sim\sigma(\mathbf{s}_{1:T})$   ∼ in the extended state space target distribution.  

In particular, we begin by sampling a full s quence of indices $\{j_{t}\}_{t=1}^{T}$   uniformly at random $\mathrm{Pr}(j_{1},j_{2},...j_{T})=(1/K)^{T}$ . Setting   $\omega_{T}^{1}=j_{T}$  , we let   $\bar{\omega}_{t-1}^{j_{t}}\bar{=}j_{t-1}$  for all  t . This implies the following, −  

$$
\begin{array}{r l}{\omega_{T}^{1}=j_{T},\;\omega_{t-1}^{j_{t}}=j_{t-1}\quad}&{\Longrightarrow\quad h_{T}^{1}=[j_{1},j_{2},...j_{T}],\qquad h_{t-1}^{j_{t}}=[j_{1},j_{2},...j_{t-1}],}\\ &{\mathrm{\quad~and\quad}\quad\bar{h}_{t}^{j_{t}}=h_{t}^{j_{t+1}}}\end{array}
$$  

To show these identities, note that   $\omega_{t-1}^{j_{t}}=j_{t-1}$  and  Eq. (Index Notation)  imply   $h_{t-1}^{j_{t}}=h_{t-2}^{\omega_{t-1}^{j_{t}}}+[\omega_{t-1}^{j_{t}}]=h_{t-2}^{j_{t-1}}+[j_{t-1}]=$ − − − − − − − $\bar{h}_{t-1}^{j_{t-1}}$   , which matches  Eq. (76) . Applying this recursion again yields   $h_{t-1}^{j_{t}}=h_{t-3}^{j_{t-2}}+[j_{t-2},j_{t-1}]...=[j_{1},j_{2},...j_{t-1}]$ . − − − − − − Taken together, these notations allow us to interleave a true target sample in particular indices   $\{j_{t}\}$ , guaranteeing that at least one target samples appears at each step.  

The extended state space target distribution differs from   $q_{\mathrm{SMC}}$  in its handling of this sequence, which identified as $\mathbf{s}_{1:T}^{h_{T}^{1}}$   with prefixes   $\mathbf{s}_{1:t-1}^{h_{t}^{j_{t}}}$   using  Eq. (75) . Noting that sampling   $\{j_{t}\}_{t=1}^{T}$   amounts to specifying a particular set of   $\omega_{t}^{k}$   as in  Eq. (75) -( 76 ), −  

$$
\mathbf{\Phi}_{\operatorname{SLC}}\left(\{s_{t}^{k},\omega_{t}^{k}\}_{t=1,k=1}^{T,K}\right)=\underbrace{\mathbb{P}_{T}(j_{1},j_{2},\dots j_{T})}_{\left(\frac{1}{K}\right)^{T}}\tilde{\pi}_{T}\left(\mathbf{s}_{1:T}^{h_{T}^{1}}\right)\prod_{t=1}^{T}\left[\prod_{\underset{k\neq j_{t}}{k=1}}^{K}q\left(s_{t}^{k}\left|\mathbf{s}_{1:t-1}^{h_{t-1}^{k}}\right)\prod_{\underset{k\neq j_{t+1}}{k=1}}^{K}q\left(\omega_{t}^{k}\left|\mathbf{s}_{1:t}^{1:K}\right)\right].
$$  

(SMC Extended Target)  

Note, the normalization constant of   $\tilde{\boldsymbol{\sigma}}_{\mathrm{SMC}}(\boldsymbol{S})$  is equal to   $\mathcal{Z}_{\sigma}$  since only $\tilde{\pi}_{T}(\mathbf{s}_{1:T})=\tilde{\sigma}(\mathbf{s}_{1:T})$  is unnormalized.  

To describe ancestral sampling from  Eq. (SMC Extended Target) , we first sample   $\{j_{t}\}_{t=1}^{T}$   uniformly as above, and place an exact target sequence in indices $\mathbf{s}_{1:T}^{h_{T}^{1}}$   (or, equivalently, sequentially sample $s_{t}^{j_{t}}\sim\pi_{t}(s_{t}|\mathbf{s}_{1:t-1}^{h_{t-1}^{j_{t}}})$   − . At each step, the remaining $K-1$  indices   $k\neq j_{t}$  are sampled from the proposal. For resampling, we fix index $j_{t}$  to hold the exact sample and resample the remaining $K-1$  −  indices. Note that the resampling eights   $q\mathopen{}\mathclose\bgroup\left(\bar{\omega_{t}^{k}}\aftergroup\egroup\right|\mathbf{s}_{1:t}^{1:K}\aftergroup\egroup\right)$    in  Eq. (74)  include  the exact sample, which may be cloned additional times into indices other than $j_{t}$  if its importance weights are high. The procedure above simply ensures that  at least  one exact sequence is sampled. See  Alg. 2  for the pseudocode of the algorithm.  

Note that  Maddison et al.  ( 2017 , Alg. 2) presents a different SMC extended state space target distribution than ours. In their work, $j_{1}=1$  and they sample $\mathbf{j}_{2:T+1}$ , while in ours $j_{T+1}=1$  and we sample $\mathbf{j}_{1:T}$ . However, both targets result in the same log partition function bounds.  

Worked Example:  In  Fig. 2  (c), we use blue circles and arrows to highlight the exact-sample indices $h_{T}^{1}=[j_{1},j_{2}]=[3,2]$ and the target sequence $\mathbf{s}_{1:T}^{h_{T}^{1}}=[s_{1}^{3},s_{2}^{2}]$ . Using the recursion   $\omega_{t-1}^{j_{t}}=j_{t-1}$  with $j_{T+1}=j_{3}=1$  fixed, we may also express − − $h_{T}^{1}=[j_{1},j_{2}]=[3,2]=[\dot{\omega_{1}^{2}},\omega_{2}^{1}]$ . At step 2, note the target sequence is sampled/evaluated an additional time in index 3.  

Importance Weights in the Extended State Space Assume we are given a fixed set of   $\{s_{t}^{k},\omega_{t}^{k}\}_{t=1,k=1}^{T,K}$   , which may be sampled from either $\sigma_{\mathrm{SMC}}(S)$  or $q_{\mathrm{SMC}}(\pmb{S})$ . We proceed to show that the unnormalized importance weights in the extended state space simplify as follows.  

Lemma F.1.  For the extended state space target $\tilde{\sigma}_{\mathrm{SMC}}$  and proposal $q_{\mathrm{SMC}}$  above, the simple importance weights in the extended state space become  

$$
\langle s_{t}^{k},\omega_{t}^{k}\rangle_{t=1,k=1}^{T,K}\Big)=\prod_{t=1}^{T}\frac{1}{K}\sum_{k=1}^{K}\frac{\tilde{\pi}_{t}\Big(\bar{s}_{1:t}^{\bar{h}_{t}^{k}}\Big)}{\tilde{\pi}_{t-1}\Big(\bar{s}_{1:t-1}^{\bar{h}_{t-1}^{k}}\Big)q\Big(s_{t}^{k}\;\Big|\;s_{1:t-1}^{\bar{h}_{t-1}^{k}}\Big)}=\prod_{t=1}^{T}\frac{1}{K}\sum_{k=1}^{K}w_{t}\Big(\bar{s}_{1:t}^{\bar{h}_{t}^{k}}\Big)=:\prod_{t=1}^{T}\frac{1}{K}\sum_{k=1}^{K}w_{t}\Big(s_{1:t-1}^{\bar{h}_{t-1}^{k}}\Big)
$$  

which can be used to obtain unbiased   $\mathcal{Z}_{\sigma}$  estimators ( Eq.  (8) ) or bounds on $\log\mathcal{Z}_{\sigma}$ Prop. 5.1 , with proof below).  

Proof.  To evaluate the importance weights (with the goal of estimating   $\mathcal{Z}_{\sigma}$ ), we consider  

$$
\begin{array}{r l r}{\lefteqn{\frac{\tilde{\sigma}_{\mathsf{S M C}}}{q_{\mathsf{S M C}}}\left(\{s_{t}^{k},\omega_{t}^{k}\}_{t=1,k=1}^{T,K}\right)=\frac{\left(\frac{1}{K}\right)^{T}\tilde{\pi}_{T}\left(\mathbf{s}_{1:T}^{h_{T}^{1}}\right)\prod_{t=1}^{T}\left[\prod_{k=1}^{K}q\left(s_{t}^{k}\;\middle|\;\mathbf{s}_{1:t-1}^{h_{t-1}^{k}}\right)\prod_{k\neq\bar{j}_{t+1}}^{K}q\left(\omega_{t}^{k}\;\middle|\;\mathbf{s}_{1:t}^{1:K}\right)\right]}{\prod_{t=1}^{T}\left[\prod_{k=1}^{K}q\left(s_{t}^{k}\;\middle|\;\mathbf{s}_{1:t-1}^{h_{t-1}^{k}}\right)\prod_{k=1}^{K}q\left(\omega_{t}^{k}\;\middle|\;\mathbf{s}_{1:t}^{1:K}\right)\right]}}\\ &{}&{\overset{(1)}{=}\left(\frac{1}{K}\right)^{T}\tilde{\pi}_{T}\left(\mathbf{s}_{1:T}^{h_{T}^{1}}\right)\prod_{t=1}^{T}\frac{1}{q\left(s_{t}^{j_{t}}\;\middle|\;\mathbf{s}_{1:t-1}^{h_{t-1}^{j_{t}^{k}}}\right)q\left(\omega_{t}^{j_{t+1}}\;\middle|\;\mathbf{s}_{1:t}^{1:K}\right)}}\end{array}
$$  

where in  (1) , note that terms in the denominator cancel except for the indices   $[0,j_{1},...j_{T}]=h_{T}^{1}$ . Recalling that $\omega_{t}^{j_{t+1}}=j_{t}$ from  Eq. (76) , we expand the resampling weights   $q(j_{t}|\mathbf{s}_{1:t}^{1:K})$  for the sequence indexed by   $s_{t}^{j_{t}}$ ,   $\bar{\mathbf{s}}_{1:t-1}^{j_{t}}$ , and $\mathbf{s}_{1:t-1}^{\bar{h}_{t}^{j_{t}}}$ , − −  

$$
\mathbf{\Xi}_{\stackrel{(2)}{=}}^{\underline{{(2)}}}\left(\frac{1}{K}\right)^{T}\tilde{\pi}_{T}\!\left(\mathbf{s}_{1:T}^{h_{T}^{1}}\right)\prod_{t=1}^{T}\frac{\displaystyle\sum_{k=1}^{K}\ \frac{\tilde{\pi}_{t}\left(\mathbf{s}_{1:t}^{h_{k}^{t}}\right)}{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}^{h_{t}^{t}}\right)q\left(s_{t}^{k}\bigg|\mathbf{s}_{1:t-1}^{h_{t-1}^{t}}\right)}}{\displaystyle\underbrace{q\!\left(s_{t}^{j_{t}}\!\!-\!\!s_{1:t-1}^{h_{t-1}^{j_{t}^{t}}}\right)}_{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}^{h_{t-1}^{j_{t}^{t}}}\right)q\left(s_{t}^{j_{t}^{j_{t}^{t}}}\!\!\right)}\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!
$$  

Finally, we obtain a telescoping cancellation of $\tilde{\pi}_{t}$  terms using the indexing identities in  Eq. (75) -( 76 ). In particular, since $\bar{\pmb h}_{t}^{j_{t}}=\pmb h_{t}^{j_{t+1}}$ and   $\bar{h}_{t-1}^{j_{t-1}}=h_{t-1}^{j_{t}}$   with   $\bar{h}_{T}^{j_{T}}=h_{T}^{1}$ , we can simplify the terms in  Eq. (80)  as − −  

$$
\Big)\prod_{t=1}^{T}\frac{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}^{h_{t-1}^{j_{t}}}\right)}{\tilde{\pi}_{t}\left(\mathbf{s}_{1:t}^{h_{t}^{j_{t}}}\right)}=\tilde{\pi}_{T}\left(\mathbf{s}_{1:T}^{h_{T}^{j_{T}}}\right)\prod_{t=1}^{T}\frac{\tilde{\pi}_{t-1}\left(\mathbf{s}_{1:t-1}^{h_{t-1}^{j_{t}}}\right)}{\tilde{\pi}_{t}\left(\mathbf{s}_{1:t}^{h_{t}^{j_{t}}}\right)}=\tilde{\pi}_{T}\left(\mathbf{s}_{1:T}^{h_{T}^{j_{T}}}\right)\prod_{T=1}^{\tilde{\pi}_{T-1}\left(\mathbf{s}_{1:T-1}^{h_{T}^{j_{T}}}\right)}\frac{\tilde{\pi}_{T-1}\left(\mathbf{s}_{1:T-1}^{h_{T}^{j_{T}}}\right)}{\tilde{\pi}_{T-1}\left(\mathbf{s}_{1:T}^{h_{T}^{j_{T}}}\right)}\tilde{\pi}_{T-1}\left(\mathbf{s}_{1:T-1}^{h_{T}^{j_{T}}}\right)\cdots\frac{1}{\tilde{\pi}_{T-1}\left(\mathbf{s}_{1:T-1}^{h_{T}^{j_{T}}}\right)}
$$  

using the assumption that $\tilde{\pi}_{0}(\cdot)=1$ · . Simplifying from  Eq. (80) , the final unnormalized importance weights become  

$$
\Big(\{s_{t}^{k},\omega_{t}^{k}\}_{t=1,k=1}^{T,K}\Big)=\prod_{t=1}^{T}\frac{1}{K}\sum_{k=1}^{K}\frac{\tilde{\pi}_{t}\Big(\mathbf{s}_{1:t}^{k}\Big)}{\tilde{\pi}_{t-1}\Big(\mathbf{s}_{1:t-1}^{k,k}\Big)q\Big(s_{t}^{k}\,\Big|\,\mathbf{s}_{1:t-1}^{k,k}\Big)}=\prod_{t=1}^{T}\frac{1}{K}\sum_{k=1}^{K}w_{t}\Big(\mathbf{s}_{1:t}^{k}\Big)=:\prod_{t=1}^{T}\frac{1}{K}\sum_{k=1}^{K}w_{t}(\mathbf{s}_{1:t-1}^{k,k}).
$$  

as desired, where we abbreviate the importance weights as $w_{t}(\mathbf{s}_{1:t}^{k})$  for simplicity of notation. Note that we also obtain an unbiased estimate of the partition function via  

$$
\mathcal{Z}_{\sigma}=\mathbb{E}_{q_{\mathrm{SMC}}(S)}\bigg[\frac{\tilde{\sigma}_{\mathrm{SMC}}(S)}{q_{\mathrm{SMC}}(S)}\bigg]=\mathbb{E}_{q_{\mathrm{SMC}}(S)}\left[\prod_{t=1}^{T}\frac{1}{K}\sum_{k=1}^{K}w_{t}\bigg(\mathbf{s}_{1:t}^{k}\bigg)\right]
$$  

Proposition 5.1. (Bidirectional SMC Bounds)  The log partition function $\log\mathcal{Z}_{\sigma}$  of a target distribution $\sigma(\mathbf{s}_{1:T})$  can be lower and upper bounded by  

$$
\begin{array}{r l}&{\mathbb{E}_{q_{\mathtt{s m c}}(s)}\left[\log\prod_{t=1}^{T}\frac{1}{K}\sum_{i=1}^{K}w_{t}\big(\mathbf{s}_{1:t}^{i}\big)\right]\leq\log\mathcal{Z}_{\sigma}}\\ &{\quad\quad\log\mathcal{Z}_{\sigma}\leq\mathbb{E}_{\sigma_{\mathtt{s m c}}(s)}\left[\log\prod_{t=1}^{T}\frac{1}{K}\sum_{i=1}^{K}w_{t}\big(\mathbf{s}_{1:t}^{i}\big)\right].}\end{array}
$$  

The gap in the lower bound is $D_{\mathrm{KL}}(q_{\mathrm{SMC}}(S)\,||\,\sigma_{\mathrm{SMC}}(S))$ , and the gap in the upper bound is   $D_{\mathrm{KL}}(\sigma_{\mathrm{SMC}}(S)\,||\,q_{\mathrm{SMC}}(S))$ .  

Proof.  The proof follows directly from  Brekelmans et al.  ( 2022 ) App. A, where it is shown that for   $\sigma_{\mathrm{ext}}(\mathbf{\boldsymbol{S}}),q_{\mathrm{ext}}(\mathbf{\boldsymbol{S}})$  such that   $\mathcal{Z}_{\sigma}=\mathbb{E}_{q_{\mathrm{ext}}(S)}\big[\frac{\tilde{\sigma}_{\mathrm{ext}}(S)}{q_{\mathrm{ext}}(S)}\big]$ , we can construct lower and upper bounds on   $\log\mathcal{Z}_{\sigma}$  

$$
\begin{array}{r l}&{\relax_{\mathrm{ext}}(\boldsymbol{S})\parallel\sigma_{\mathrm{ext}}(\boldsymbol{S}))+\mathbb{E}_{q_{\mathrm{ext}}(\boldsymbol{S})}\biggl[\log\frac{\tilde{\sigma}_{\mathrm{ext}}(\boldsymbol{S})}{q_{\mathrm{ext}}(\boldsymbol{S})}\biggr]=\log\mathcal{Z}_{\sigma}=\mathbb{E}_{\sigma_{\mathrm{ext}}(\boldsymbol{S})}\biggl[\log\frac{\tilde{\sigma}_{\mathrm{ext}}(\boldsymbol{S})}{q_{\mathrm{ext}}(\boldsymbol{S})}\biggr]-D_{\mathrm{KL}}(\sigma_{\mathrm{ext}}(\boldsymbol{S})\parallel q_{\mathrm{ext}}(\boldsymbol{S}))}\\ &{\ \ \ \ \ \ \ \ \mathbb{E}_{q_{\mathrm{ext}}(\boldsymbol{S})}\biggl[\log\frac{\tilde{\sigma}_{\mathrm{ext}}(\boldsymbol{S})}{q_{\mathrm{ext}}(\boldsymbol{S})}\biggr]\leq\log\mathcal{Z}_{\sigma}\leq\mathbb{E}_{\sigma_{\mathrm{ext}}(\boldsymbol{S})}\biggl[\log\frac{\tilde{\sigma}_{\mathrm{ext}}(\boldsymbol{S})}{q_{\mathrm{ext}}(\boldsymbol{S})}\biggr]}\end{array}
$$  

where the gap in the lower and upper bounds are   $D_{\mathrm{KL}}(q_{\mathrm{ext}}(\boldsymbol{S})\,\|\,\sigma_{\mathrm{ext}}(\boldsymbol{S}))$  and   $D_{\mathrm{KL}}(\sigma_{\mathrm{ext}}(S)\,||\,q_{\mathrm{ext}}(S))$ , respectively.  

Substituting our SMC probabilistic interpretation in  Eq. (SMC Extended Proposal)  and  Eq. (SMC Extended Target) , along with the importance weights in  Lemma F.1 , into  Eq. (83)  yields the desired bounds in  Eq. (24) .  

IWAE as a Special Case of our SMC Probabilistic Interpretation Note that we recover IWAE (or SIS over   $K$  samples) from SMC with no intermediate resampling. In particular, this corresponds to $\omega_{t}^{k}=k$  for all   $t<T$ , with importance weighting from resampling occurring at the final step $\textstyle\prod_{k=1}^{K}q(\omega_{T}^{k}|\mathbf{s}_{1:T}^{1:K})$   | . This yields the $1/K$  average inside the log in the IWAE bounds (i.e., SMC with only one resampling step at $t=T$ ). While the importance weights are crucial to construct the bound, note that ‘resampling’ is not necessary at the final step and we may return all   $K$  samples along with their weights.  

Viewing IWAE as a special case of our SMC probabilistic interpretation is complementary to the interpretations in  Domke & Sheldon  ( 2018 );  Brekelmans et al.  ( 2022 ) and also provides upper bounds ( Sobolev & Vetrov ,  2019 ).  

# G. Additional Experiment Details  

# G.1. Common Details Across Experiments  

For all experiments, we use the Adam optimizer with $\beta_{1},\beta_{2}=\{0.9,0.999\}$ . We use custom implementations of SMC. For PPO, we use the HuggingFace TRL PPO Trainer ( https://github.com/huggingface/trl/blob/main/trl/trainer/ ppo trainer.py ), modified slightly to accomodate our custom twist parameter iz at ions, as described below. For other methods, we use Optax (Flax) and custom loss functions. We use HuggingFace models ( https://huggingface.co/ models ) for the base $p_{0}$  models and build custom layers on top of those.  

For the twist $\psi_{t}^{\boldsymbol\theta}(\mathbf{s}_{1:t})$ , we always parameterize $\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$  for numerical stability. We choose random normal initializations centered at mean 0, with low variance,   such that   $\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})\approx0,\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})\approx1$  ≈  ≈  at the beginning of training, which means the initial sequences generated by the twist-induced proposal approximately come from the base model $p_{0}$ . All methods are initialized using the same random seeds, and thus start from the same parameter values. See  App. G.2  for additional discussion of choices for the twist parameter iz ation.  

For methods that directly learn a proposal (DPG and PPO), we could directly finetune a language model that outputs   $q(\mathbf{s}_{1:t})$ . However, in order to ensure consistency in terms of model capacity and ease of learning compared to our twisted proposals, we instead have these proposal learning methods output a modifier   $\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$  which is added to the base model log probability   $\log p_{0}(\mathbf{s}_{1:t})$ . Note that using random normal initializations centered at mean 0 with low variance, this scheme results in initial $q$  samples coming approximately from   $p_{0}$ .  

For methods that can make use of exact posterior samples, when we have access to them ( Sec. 7.2.3 ,  App. H.3 ), we use them. This is straightforward for methods like DPG, SIXO, and our CTL (unless we have only a single sample, as we discuss for infilling in  App. G.4  ). For our RL twist learning, we found the best empirical performance training on a combination of $q$ and exact $\sigma$  samples when they were available (as opposed to just $q$  otherwise), and use those results. Similarly, for FUDGE, when exact $\sigma$  samples are available, we use them together with $p_{0}$  samples.  

It is not straightforward to compare PPO versus other methods, because of the inner loop in PPO that repeats several clipped gradient steps on a given set of samples. This means that, for a constant number of samples, PPO makes more gradient updates than other methods, while for a constant number of gradient updates, PPO sees fewer samples. Ultimately we decided to normalize based on the number of samples seen; we consider each outer step (including a full PPO inner loop, in our experiments, 4 gradient steps) as a single “gradient update.” We make this choice since sampling is the main bottleneck in terms of computational cost, and the number of inner PPO steps is a hyperparameter which we did not tune.  

All of our experiments were run on a single GPU, usually on an NVIDIA A40 with 48G memory. All experiments took no longer than 9 wall-clock hours to run for a single learning method, with infilling ( Sec. 7.2.3 ) experiments taking longest; most other experiments took no longer than 4 hours.  

# G.2. Choices of Twist Parameter iz ation  

The choice of parameter iz ation for the twist   $\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$  is a design decision, independent of our overall framework. While one could keep an entirely separate model for each   $\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$ , this is likely to be memory-inefficient and learn slowly. Instead, we use a shared parameter iz ation across $\mathbf{s}_{1:t}$ , in the same way that the base language model uses a single architecture to output probability distributions over tokens at each time step $t$ . We lay out parameter iz ation choices we considered below.  

# G.2.1. L INEAR  H EAD  

The simplest choice is to replace the linear head of the base language model with a new linear head, keep the base model fixed, and only train the linear head. This parameter iz ation incurs very little additional computation cost compared to just using the base language model. However, we found this to be capacity constrained in our experiments, achieving worse KL divergences than other parameter iz at ions.  

# G.2.2. MLP H EAD  

Instead of a linear head, we consider a 3-layer fully connected neural network (MLP) with ReLU non-linearities as a head on top of the base language model. The base model is still kept fixed; only the MLP head is trained. This incurs more computational cost than a linear head ( App. G.2.1 ), but the additional cost is still small relative to the cost of a forward pass through the base transformer model. We found this to generally perform well in our experiments, so we use it for the toxicity threshold experiment in  Sec. 7.1  and sentiment in  Sec. 7.2.2 .  

# G.2.3. S EPARATE  T RANSFORMER FOR THE  T WIST  

We can also consider an entirely separate transformer that outputs only the twist value. That is, we copy the base model, and repurpose it to output a twist value   $\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$  instead of logits for next-token probabilities. We then train the entire network end-to-end. This is significantly more computationally costly than the former approaches, and does not always do better than just an MLP head ( App. G.2.2 ), so we generally do not recommend using this. Still, we found it to perform well in toxicity classification in  Sec. 7.2.1 , so we use it there.  

# G.2.4. S EPARATE  T RANSFORMER FOR THE  T WIST ,  WITH  MLP H EAD  

This is similar to  App. G.2.3 , except we also replace the final linear head with a MLP head as in  App. G.2.2 . The model outputs   $\log\psi_{t}^{\pmb\theta}(\mathbf{s}_{1:t})$  and is trained end-to-end. This is the most computationally costly approach outlined here, and is unnecessary for most of our settings. However, in infilling with 15 generated tokens ( Sec. 7.2.3 ) we found this parameter iz ation to perform materially better than all others, particularly with DPG ( App. E.3 ), so we use it for all infilling experiments.  

With both this parameter iz ation and  App. G.2.3 , we increase computation time by a factor of around 2 on the forward pass, and significantly increase memory and time usage on the backwards pass during training (though sampling is still the main time bottleneck). Whether this parameter iz ation is worth the potential gain in performance depends on the desired use case. We emphasize that our overall framework is independent of the choice of parameter iz ation.  

# G.3. Comments on Our Choices of Experiment Settings  

Our settings and evaluation metrics in  Sec. 7  are chosen to highlight our scientific findings. In particular, the toxicity threshold experiment in  Sec. 7.1  demonstrates the improvement of SMC over SIS with the base model with CTL learned twists. In order to highlight this distinction, we have chosen a setting where it is  extremely  difficult to draw samples satisfying the threshold using the base model $p_{0}$  (see SIS/IWAE LB line in  Fig. 3 ).  

However, twist-learning in the toxicity threshold setting presents challenges. For approximate positive sampling and a thresholded target, all importance weights will be 0 if none of our   $K$  samples meet the threshold. As noted above, sampling from $p_{0}$ , or the SMC/twisted proposal for   $\psi_{t}^{\boldsymbol\theta}(\mathbf{s}_{1:t})\approx1$  ≈  at initialization, is extremely unlikely to draw samples meeting the threshold (i.e., within the support of the target) in the setting of  Sec. 7.1 . As a result, initial iterations of twist learning receive no learning signal until a thresholded positive sample is drawn from the base model.  

To avoid this difficulty for baselines comparisons in  Sec. 7.2 , we instead focused on settings with   $\phi(\mathbf{s}_{1:T})$  given by probabilities. Nevertheless, we note that there are no fundamental differences between the settings considered in  Sec. 7.1 and  Sec. 7.2 . Thus, we  y also evaluate single-sample $D_{\mathrm{KL}}(\sigma\parallel q)$  and $D_{\mathrm{KL}}(q\,\|\,\sigma)$  in the setting of  Sec. 7.1 , or plot $\log\mathcal{Z}_{\sigma}$ bounds as a function of  K  in for the settings in  Sec. 7.2 .  

# G.4. Experiment-Specific Details  

Details for SIS and SMC Comparison ( Sec. 7.1 ) We generate 10 output tokens, and train twists using  Sec. 4.1  with approximate positive sampling as discussed in  Sec. 4.1.2 .  

Note that u $\sigma(\mathbf{s}_{1:T})\propto p_{0}(\mathbf{s}_{1:T})\mathbb{I}[\mathbf{s}_{1:T}\in\mathcal{C}]$   $\mathcal{C}:=\{\mathbf{s}_{1:T}\ |r(\mathbf{s}_{1:T})\leq\eta\}$ ctly runs into numerica calcul $\log\mathcal{Z}_{\sigma}$  when $\mathbf{s}_{1:T}\notin\mathcal{C}$ ∈C  and $\mathbb{I}[\mathbf{s}_{1:T}\in\mathcal{C}]=0$  ∈C e $\epsilon+\mathbb{I}[\mathbf{s}_{1:T}\in\mathcal{C}]$  ∈C  everywhere instead of  I $\mathbb{I}[\mathbf{s}_{1:T}\in\mathcal{C}]$   C , where  ϵ $\epsilon=10^{-16}$ . In  Fig. 3 , this yields a SIS/IWAE $\log\mathcal{Z}_{\sigma}$ $\mathbf{LB}\approx-36$  when no samples are drawn that fall in the set .  

We use an MLP head to parameterize the twist, as in  App. G.2.2 , with 768 hidden units per layer, matching the TinyStories model’s embedding dimension. We use a batch size (number of SMC particles/samples) of 1000, with a learning rate of 0.0001, and train using CTL for a total of 5000 gradient updates. We did not tune hyperparameters because we found this setting to work well, and we are not comparing across different learning methods.  

For each point on each line on  Fig. 3 , we run SIS or SMC 20 times, each with a different randomly selected true posterior sample for the upper bounds. The line shows the average value across these 20 runs, while the shaded area shows $95\%$ confidence intervals. See also  App. G.1  for details common across experiments.  

Details for Toxicity ( Sec. 7.2.1 ) We generate 20 output tokens. We parameterize the twist with a separate network as in App. G.2.3 . We use a batch size (number of SMC particles/samples) of 100, and train for a total of 2048 gradient updates. For each learning method, we used a coarse grid search over learning rates between 0.000001 and 0.001, using the best one found, which was usually 0.00003 or 0.0001. We run each learning method over 5 different random seeds, reporting the average KL divergence and $95\%$  confidence intervals over these 5 seeds.  

For each KL divergence e we first get sandwich bounds on   $\log\mathcal{Z}_{\sigma}$  as laid out in  Sec. 5 , using the learned twists for the twisted proposal with $K=500$  samples. We find SIS/IWAE and SMC bounds to be similarly tight, so use SIS/IWAE for simplicity. We do this 4 times, providing 4 upper bound estimates and 4 lower bound estimates, and take the average midpoint as the $\log\mathcal{Z}_{\sigma}$  estimate for each exper We then take the median (across all learning methods and seeds) of these estimates, and use that as our estimate of $\log\mathcal{Z}_{\sigma}$  Z . This is then used as a common value for the KL divergence across all methods and seeds, which controls for po ble noise in $\log\mathcal{Z}_{\sigma}$  bounds and en  fair comparison across methods. We generally have tight bounds (upper boun $\approx$ er bound), which suggest our $\log\mathcal{Z}_{\sigma}$  Z  estimates are generally accurate, but note that any inaccuracies in estimating $\log\mathcal{Z}_{\sigma}$  Z  would only affect the absolute values of the KL divergences, not the relative differences among different learning methods.  

We estimate expectations in  Eq. (23)  with 2000 samples from   $q$  and 2000 exact posterior samples for $\sigma$ . With 2000 samples, our estimates have $95\%$  confidence intervals generally between 0.05 and 0.10, suggesting that our estimates of expectations are unlikely to be off by more than 0.10. The exact posterior samples were collected offline; such a large number of samples takes several hours to collect, and in practical settings, we would likely only be able to collect a much smaller number of samples. All our methods still apply with fewer exact posterior samples, but the variance in estimates will be higher. See also  App. G.1  for details common across experiments.  

Details for Sentiment ( Sec. 7.2.2 ) We generate 10 output tokens. We parameterize the twist using an MLP head ( App. G.2.2 ), with 1024 hidden units per layer, matching the GPT2Medium model’s embedding dimension. Other details are the same as for toxicity above. Collecting exact posterior samples is less time consuming in this case (less than an hour). See  App. G.1  for common experimental details.  

Details for Infilling ( Sec. 7.2.3 ) We parameterize the twist using a separate transformer with an MLP head ( App. G.2.4 ), with 768 hidden units per layer (matching the TinyStories model’s embedding dimension). We make the following adjustments to the forward pass of the language model for the conditional twist setting. Instead of taking in only   $\mathbf{s}_{1:T}$ , the model takes in both $\mathbf{s}_{1:T}$  and $\mathbf{s}_{T+1:T+c}$  and passes each separately through the body (everything except the head). Thus, $\mathbf{s}_{T+1:T+c}$  can be seen as a second prompt. For $\mathbf{s}_{T+1:T+c}$ , we take the embeddings produced after the last conditioning token  has been processed, broadcast it across time steps $1:T$ , and pass that as additional input to the MLP head $s_{T+c}$ (concatenated with embeddings for $\mathbf{s}_{1:T}$  at each   $t\in1...T)$ ). This allows the MLP head to produce different output depending on the conditioning tokens.  

Since we are in the conditional target distribution setting ( Sec. 3.3 ), with   $o_{T}\;=\;\mathbf{s}_{T+1:T+c}$ , to compare across learn- $\begin{array}{r}{\mathbb{E}_{o_{T}}[D_{\mathrm{KL}}(q_{o_{T}}\parallel\sigma_{o_{T}})]\ :=\ \mathbb{E}_{o_{T}}[D_{\mathrm{KL}}(q(\mathbf{s}_{1:T}|o_{T})\parallel\sigma(\mathbf{s}_{1:T}|o_{T}))]}\end{array}$  and $\begin{array}{r}{\mathbb{E}_{o_{T}}[D_{\mathrm{KL}}\big(\sigma_{o_{T}}\parallel q_{o_{T}}\big)]:=\mathbb{E}_{o_{T}}[D_{\mathrm{KL}}\big(\sigma\big(\mathbf{s}_{1:T}|o_{T}\big)\parallel q\big(\mathbf{s}_{1:T}|o_{T}\big)\big)]}\end{array}$  ∥  |  ∥  |  where $\mathbb{E}_{o_{T}}[\cdot]:=\mathbb{E}_{p_{0}(\mathbf{s}_{T+1:T+c})}[\cdot]$  for infilling. Note that,  

$$
\begin{array}{r l}&{\mathbb{E}_{o_{T}}[D_{\mathrm{KL}}(q(\mathbf{s}_{1:T}|o_{T})\,\|\,\sigma(\mathbf{s}_{1:T}|o_{T}))]=\mathbb{E}_{o_{T}}\bigg[\mathbb{E}_{q(\mathbf{s}_{1:T}|o_{T})}\bigg[\log\frac{q\big(\mathbf{s}_{1:T}|o_{T}\big)}{p_{0}(\mathbf{s}_{1:T})\phi(\mathbf{s}_{1:T},o_{T})}\bigg]\bigg]+\mathbb{E}_{o_{T}}[\log\mathcal{Z}_{\sigma}(\mathbf{s}_{1:T}|o_{T})]}\\ &{\mathbb{E}_{o_{T}}[D_{\mathrm{KL}}(\sigma(\mathbf{s}_{1:T}|o_{T})\,\|\,q(\mathbf{s}_{1:T}|o_{T}))]=\mathbb{E}_{o_{T}}\bigg[\mathbb{E}_{\sigma(\mathbf{s}_{1:T}|o_{T})}\bigg[\log\frac{p_{0}(\mathbf{s}_{1:T})\phi(\mathbf{s}_{1:T},o_{T})}{q(\mathbf{s}_{1:T}|o_{T})}\bigg]\bigg]-\mathbb{E}_{o_{T}}[\log\mathcal{Z}_{\sigma}(\mathbf{s}_{1:T}|o_{T})]}\end{array}
$$  

where for a fixed   $o_{T}$ ,   $\begin{array}{r}{\mathbb{E}_{q(\mathbf{s}_{1:T}|o_{T})}\biggl[\log\frac{q(\mathbf{s}_{1:T}|o_{T})}{p_{0}(\mathbf{s}_{1:T})\phi(\mathbf{s}_{1:T},o_{T})}\biggr]}\end{array}$ h i and   $\begin{array}{r}{\mathbb{E}_{\sigma(\mathbf{s}_{1:T}|o_{T})}\!\left[\log\frac{p_{0}(\mathbf{s}_{1:T})\phi(\mathbf{s}_{1:T},o_{T})}{q(\mathbf{s}_{1:T}|o_{T})}\right]}\end{array}$ h i may be evaluated as before, similar to the unconditional setting. In particular, for our experiments, we use 1-sample estimates of these expectations, as we have a single exact sa $\sigma(\mathbf{s}_{1:T}|o_{T})$  by the BD e choose to draw a single sample from the conditional proposal $q(\mathbf{s}_{1:T}|o_{T})$  | . We average this over 2000 $2000\;o_{T}\sim p_{0}\big(\mathbf{s}_{T+1:T+c}\big)$  ∼ , approximating the outer expectation, giving us a 2000-sample estimate of 1-sample estimates for the first term in the right hand side of both equations above. With 2000 samples, our estimates have $95\%$  confidence intervals generally between 0.20 and 0.30.  

Note that $\mathbb{E}_{o_{T}}[\log\mathcal{Z}_{\sigma}(o_{T})]$  is independent of the learning method or proposal $q$ , unlike the first term we discussed above. Thus, in order to save computation and provide us with a more accurate estimate of $\mathbb{E}_{o_{T}}[\log\mathcal{Z}_{\sigma}(o_{T})]$ , we estimate this term only once. Specifically, we consider only the learning method with the lowest KL divergence (DPG), and use SIS/IWAE bounds. For each   $o_{T}$ , we estimate $\log\mathcal{Z}_{\sigma}(o_{T})$  with $K=500$  samples, which gives us relatively tight sandwich bounds, ag point as our e  average this over  $1000\,o_{T}\sim p_{0}\big(\mathbf{s}_{T+1:T+c}\big)$ , giving us a 1000-sample estimate of $\mathbb{E}_{o_{T}}[\log\mathcal{Z}_{\sigma}(o_{T})]$  Z , where each $\log\mathcal{Z}_{\sigma}(o_{T})$  Z  is itself estimated via 500 samples.  

For negative sampling with contrastive twist learning (CTL) in this setting, we need at least 2 negative samples per set of conditioning tokens   ${\cal O}_{T}={\bf s}_{T+1:T+c}$  to perform SIS reweighting; this is in contrast with other twist learning methods which can generate a single negative sample per $o_{T}$ . For the positive sample, we can use our single exact sample directly, or we can run the SMC upper bound sampling procedure (“Sampling from   $\sigma_{\mathrm{SMC}}$  for SMC Upper Bounds” section in  Sec. 5.2 ) generate more approximate $\sigma$  samples using the given exact sample. We find the latter to generally perform slightly better than the former, so adopt that for our infilling experiments.  

We use a fixed batch size of 100 across all methods for training twists. To clarify the meaning of this batch size, for methods other than CTL, we have 100 draws of exact   $\sigma$  samples, each for a different set of conditioning tokens   ${\cal O}_{T}={\bf s}_{T+1:T+c}$ , so we train over 100 different $o_{T}$  at a time using 1 negative sample per $o_{T}$  . For CTL, since we need at least 2 negative samples per   $o_{T}$ , we split the batch size of 100 across the number of different   $o_{T}$  and the number of negative samples per $o_{T}$ , as an additional hyperparameter. We use $25~o_{T}$  with 4 negative samples per   $o_{T}$  for the experiments in  Sec. 7.2.3  and  

Table 6:  Qualitative Results - Reviews Very Likely to be of a Particular Rating 
![](images/165cbc7a9cb2d3be9b9a8575b0be9c5a8fabaa41f37f6a8483e4639dc8ee58d1.jpg)  

Table 7:  Qualitative Results - Infilling Examples 
![](images/e08133009bdab681008438ad11d26de62c6d44f6359a011bf85a36512491130e.jpg)  

$10\:o_{T}$  with 10 negative samples per $o_{T}$  for the experiments in  App. H.2 . Controlling for batch size in this way is arguably disadvantageous for CTL compared to other learning methods, as it learns on a smaller number of   $o_{T}$ , but this controls for memory requirements, and we feel is more fair than controlling for the number of   $o_{T}$  seen but allowing more negative samples for CTL relative to other methods. We train for a total of 5500 gradient updates. For each method, we used a coarse grid search over learning rates between 0.000001 and 0.001, using the best one found, which was usually 0.0001 or 0.00003. We run each learning method over 5 different random seeds, reporting the average KL divergence and $95\%$  confidence intervals over these 5 seeds. See also  App. G.1  for details common across experiments.  

# H. Additional Experimental Results  

# H.1. Qualitative Results  

Toxicity Controlled Generation when No Exact Posterior Samples are Available In  Sec. 7.2.1  we targeted   $\sigma(\mathbf{s}_{1:T})\propto$ $p_{0}\big(\mathbf{s}_{1:T}\big)e^{\beta\log p\left(a|\mathbf{s}_{1:T}\right)}$   with   $\beta=1$ . We can also target $\beta>1$ ; higher $\beta$  produces a more peaked distribution of text that is more l  be of class $a$ . However, for $\beta\neq1$  we can no longer generate exact posterior samples and thus cannot upper bound $\log\mathcal{Z}_{\sigma}$  Z . Our twist learning ( Sec. 4.1 ) with approximate positive sampling ( Sec. 4.1.2 ) can le aningful twists in this setting, which we illustrate with a qualitative example of a story (200 tokens upper limit) and $\beta=10$ :  

“Once upon a time, there was a little girl named Lily. She had a big thumb that she liked to suck on. One day, Lily went to the park to play with her friends. She was having so much fun until her thumb got stuck in her shoe. She tried to pull it out, but it hurt too much. Lily started to cry and her friends tried to help her, but they couldn’t get her thumb out either. She was scared and didn’t know what to do. Her friends tried to help her, but they couldn’t get it out either. Sadly, Lily had to go to the hospital and get a big bandage on her thumb. She couldn’t play with her friends anymore. From that day on, Lily never went to the park again.”  

The story is coherent and follows the general style of the TinyStories base model, while having a high probability  $(\approx88\%)$ ) of being toxic according to the toxicity classifier, likely due to the presence of negative words such as ‘suck’, ‘hurt’, ‘cry’, and ‘scared’. This supports the ability of our methods to control outputs based on the chosen posterior distribution.  

Sentiment Controlled Generation when No Exact Posterior Samples are Available As above, we also consider $\sigma(\mathbf{s}_{1:T})\propto p_{0}(\mathbf{s}_{1:T})e^{\beta\log p\left(a\vert\mathbf{s}_{1:T}\right)}$ , where   $\beta>1$ , except now $p(a|\mathbf{s}_{1:T})$  is based on the sentiment classifier in  Sec. 7.2.2 . In Table 6  we provide qualitative examples showing 20 tokens produced with twisted SMC with 500 particles, for $\beta=100$ , using twists trained with  Sec. 4.1 . These illustrate our framework’s ability to learn reviews that embody each rating class.  

Infilling In  Table 7  we compare qualitative results on an example set of conditioning tokens for DPG, SIXO, and CTL (in that order, to reflect increasing KL divergence). The qualitative results correlate with the quantitative measures of KL divergence; the lowest KL divergence (DPG) corresponds to infilled tokens that respect grammar and the topic. SIXO, which has higher KL divergence, fails to respect grammar. CTL generates incorrect grammar and is less on-topic, corresponding to the highest KL divergence among these methods.  

Table 8: KL Divergences (averaged over conditioning tokens drawn from the base model) for Infilling Experiments ( Sec. 7.2.3  ) with 2 Output Tokens and 1 Conditioning Token  

![](images/5d8a07e54ebbad820f2227fd29b4f16ccc72b6ff86253656b61a5dfa28502a76.jpg)  

# H.2. Infilling with Fewer Tokens  

We consider the same setting as  Sec. 7.2.3  but only generating 2 tokens, conditioned on 1 token. We show KL divergence evaluations in  Table 8 . Our evaluation reveals interesting differences among learning methods, even in this easier setting where most methods achieve low KL divergence in both directions. DPG and RL learns best, while FUDGE learns notably slower. PPO suffers on $D_{\mathrm{KL}}(\sigma\parallel q)$ , though this may be unsurprising since PPO does not make use of exact $\sigma$  samples.  

# H.3. Approximate vs. Exact Posterior Sampling  

In our toxicity and sentiment experiments, we train using approximate $\sigma$  samples to reflect the more common real-world setting where the amount of exact samples needed for training are not available. However, here we run an additional ablation experiment for insight into the effect of positive versus approximate sampling. We use rejection sampling ( Sec. 4.1.2 ) to generate exact posterior samples for training. This is much slower than generating approximate samples, so is not a practical strategy for training; we investigate this solely for understanding.  

We provide a comparison of KL divergences (evaluated the same way as in the main paper) when training using exact versus approximate $\sigma$  samples for a selection of methods that performed well in our previous experiments and are able to make use of $\sigma$  samples. Toxicity ( Sec. 7.2.1 ) results are in  Table 9  and sentiment ( Sec. 7.2.2 ) results are in  Table 10 . The first two columns of KL divergences are for exact   $\sigma$  samples. The next two are for training on the same number of samples, but using approximate positive sampling ( Sec. 4.1.2 ). Overall, for a constant number of samples, having exact $\sigma$  samples improves performance for most methods. Note however that there is an additional time cost required for rejection sampling to generate exact samples, so the exact   $\sigma$  training requires significantly more wall-clock time for any given number of samples.  

We also plot the single-sample KL divergence in both directions as a function of training time for exact vs. approximate sampling, on toxicity and sentiment experiments, in  Fig. 5 . The approximate sampling results match those in the main paper (with different colors). The exact $\sigma$  sample results cut off earlier because the time cost required for rejection sampling reduces the number of gradient updates that can be made for a given amount of wall-clock time.  

Table 9: KL Div. for Toxicity Experiments ( Sec. 7.2.1 ), comparing exact $\sigma$  samples versus approximate positive sampling. 
![](images/2273a2046096df452eacd82fdb37123aa141b35e01e691b0989ccf9a075e005b.jpg)  

Table 10: KL Div. for Sentiment Experiments ( Sec. 7.2.2 ), comparing exact   $\sigma$  samples versus approximate positive sampling. 
![](images/5676b537fb962acfcc9a6b1e73ad7394d55ae3a64c5e0e614f498144bca62fb6.jpg)  

![](images/af9109802d202627b488a43f565ecac7f0db72e81d74f1a9a65a69748e6b3c1a.jpg)  
(a) Toxicity ( Sec. 7.2.1 )  

![](images/da105b77c54adea07f3c6ea24ee4833f6c67339fbbb97d239ae3cf69677f5cdc.jpg)  
(b) Sentiment ( Sec. 7.2.2 )  

Figure 5: Training comparison for Exact versus Approximate   $\sigma$  (positive) sampling, as described in  App. H.3 . Having access to exact target samples makes learning lead to lower KL divergences in a more reliable manner.  