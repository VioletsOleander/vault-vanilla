#  2 Multi-arm Bandits 
The most important feature distinguishing reinforcement learning from other types of learning is that it uses training information that evaluates the actions taken rather than instructs by giving correct actions. This is what creates the need for active exploration, for an explicit trial-and-error search for good behavior. Purely evaluative feedback indicates how good the action taken is, but not whether it is the best or the worst action possible. Evaluative feedback is the basis of methods for function optimization, including evolutionary methods. Purely instructive feedback, on the other hand, indicates the correct action to take, independently of the action actually taken. This kind of feedback is the basis of supervised learning, which includes large parts of pattern classification, artificial neural networks, and system identification. In their pure forms, these two kinds of feedback are quite distinct: evaluative feedback depends entirely on the action taken, whereas instructive feedback is independent of the action taken. There are also interesting intermediate cases in which evaluation and instruction blend together. 
In this chapter we study the evaluative aspect of reinforcement learning in a simplified setting, one that does not involve learning to act in more than one situation. This nonassociative setting is the one in which most prior work involving evaluative feedback has been done, and it avoids much of the complexity of the full reinforcement learning problem. Studying this case will enable us to see most clearly how evaluative feedback differs from, and yet can be combined with, instructive feedback. 
The particular nonassociative, evaluative feedback problem that we explore is a simple version of the $n$ -armed bandit problem. We use this problem to introduce a number of basic learning methods which we extend in later chapters to apply to the full reinforcement learning problem. At the end of this chapter, we take a step closer to the full reinforcement learning problem by discussing what happens when the bandit problem becomes associative, that is, when actions are taken in more than one situation. 
## 2.1 An $n$ -Armed Bandit Problem 
Consider the following learning problem. You are faced repeatedly with a choice among $n$ different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to maximize the expected total reward over some time period, for example, over 1000 action selections, or time steps. 
This is the original form of the $n$ -armed bandit problem, so named by analogy to a slot machine, or “one-armed bandit,” except that it has $n$ levers instead of one. Each action selection is like a play of one of the slot machine’s levers, and the rewards are the payoffs for hitting the jackpot. Through repeated action selections you are to maximize your winnings by concentrating your actions on the best levers. Another analogy is that of a doctor choosing between experimental treatments for a series of seriously ill patients. Each action selection is a treatment selection, and each reward is the survival or well-being of the patient. Today the term “ $^{n}$ -armed bandit problem” is sometimes used for a generalization of the problem described above, but in this book we use it to refer just to this simple case. 
In our $n$ -armed bandit problem, each action has an expected or mean reward given that that action is selected; let us call this the value of that action. If you knew the value of each action, then it would be trivial to solve the $n$ -armed bandit problem: you would always select the action with highest value. We assume that you do not know the action values with certainty, although you may have estimates. 
If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest. We call this a greedy action. If you select a greedy action, we say that you are exploiting your current knowledge of the values of the actions. If instead you select one of the nongreedy actions, then we say you are exploring, because this enables you to improve your estimate of the nongreedy action’s value. Exploitation is the right thing to do to maximize the expected reward on the one step, but exploration may produce the greater total reward in the long run. For example, suppose the greedy action’s value is known with certainty, while several other actions are estimated to be nearly as good but with substantial uncertainty. The uncertainty is such that at least one of these other actions probably is actually better than the greedy action, but you don’t know which one. If you have many time steps ahead on which to make action selections, then it may be better to explore the nongreedy actions and discover which of them are better than the greedy action. Reward is lower in the short run, during exploration, but higher in the long run because after you have discovered the better actions, you can exploit them many times. Because it is not possible both to explore and to exploit with any single action selection, one often refers to the “conflict” between exploration and exploitation. 
In any specific case, whether it is better to explore or exploit depends in a complex way on the precise values of the estimates, uncertainties, and the number of remaining steps. There are many sophisticated methods for balancing exploration and exploitation for particular mathematical formulations of the $n$ -armed bandit and related problems. However, most of these methods make strong assumptions about stationarity and prior knowledge that are either violated or impossible to verify in applications and in the full reinforcement learning problem that we consider in subsequent chapters. The guarantees of optimality or bounded loss for these methods are of little comfort when the assumptions of their theory do not apply. 
In this book we do not worry about balancing exploration and exploitation in a sophisticated way; we worry only about balancing them at all. In this chapter we present several simple balancing methods for the $n$ -armed bandit problem and show that they work much better than methods that always exploit. The need to balance exploration and exploitation is a distinctive challenge that arises in reinforcement learning; the simplicity of the $n$ -armed bandit problem enables us to show this in a particularly clear form. 
## 2.2 Action-Value Methods 
We begin by looking more closely at some simple methods for estimating the values of actions and for using the estimates to make action selection decisions. In this chapter, we denote the true (actual) value of action $a$ as $q(a)$ , and the estimated value on the $t$ th time step as $Q_{t}(a)$ . Recall that the true value of an action is the mean reward received when that action is selected. One natural way to estimate this is by averaging the rewards actually received when the action was selected. In other words, if by the $t$ th time step action $a$ has been chosen $N_{t}(a)$ times prior to $t$ , yielding rewards $R_{1},R_{2},\ldots,R_{N_{t}(a)}$ , then its value is estimated to be 
$$
Q_{t}(a)={\frac{R_{1}+R_{2}+\cdot\cdot\cdot+R_{N_{t}(a)}}{N_{t}(a)}}.
$$ 
If $N_{t}(a)=0$ , then we define $Q_{t}(a)$ instead as some default value, such as $Q_{1}(a)=0$ . As $N_{t}(a)\to\infty$ , by the law of large numbers, $Q_{t}(a)$ converges to $q(a)$ . We call this the sample-average method for estimating action values because each estimate is a simple average of the sample of relevant rewards. Of course this is just one way to estimate action values, and not necessarily the best one. Nevertheless, for now let us stay with this simple estimation method and turn to the question of how the estimates might be used to select actions. 
The simplest action selection rule is to select the action (or one of the actions) with highest estimated action value, that is, to select at step $t$ one of the greedy actions, $A_{t}^{*}$ , for which $Q_{t}(A_{t}^{*})=\operatorname*{max}_{a}Q_{t}(a)$ . This greedy action selection method can be written as 
$$
A_{t}=\operatorname*{argmax}_{a}Q_{t}(a),
$$ 
where $\mathrm{arg}\mathrm{max}_{a}$ denotes the value of $a$ at which the expression that follows is maximized (with ties broken arbitrarily). Greedy action selection always exploits current knowledge to maximize immediate reward; it spends no time at all sampling apparently inferior actions to see if they might really be better. A simple alternative is to behave greedily most of the time, but every once in a while, say with small probability $\boldsymbol{\varepsilon}$ , instead to select randomly from amongst all the actions with equal probability independently of the actionvalue estimates. We call methods using this near-greedy action selection rule $\varepsilon$ -greedy methods. An advantage of these methods is that, in the limit as the number of plays increases, every action will be sampled an infinite number of times, guaranteeing that $N_{t}(a)\to\infty$ for all $a$ , and thus ensuring that all the $Q_{t}(a)$ converge to $q(a)$ . This of course implies that the probability of selecting the optimal action converges to greater than $1-\varepsilon$ , that is, to near certainty. These are just asymptotic guarantees, however, and say little about the practical effectiveness of the methods. 
To roughly assess the relative effectiveness of the greedy and $\varepsilon$ -greedy methods, we compared them numerically on a suite of test problems. This was a set of 2000 randomly generated $n$ -armed bandit tasks with $n=10$ . For each bandit, the action values, $q(a)$ , $a=1,\ldots,10$ , were selected according to a normal (Gaussian) distribution with mean 0 and variance 1. On $t$ th time step with a given bandit, the actual reward $R_{t}$ was the $q(A_{t})$ for the bandit (where $A_{t}$ was the action selected) plus a normally distributed noise term that was mean 0 and variance 1. Averaging over bandits, we can plot the performance and behavior of various methods as they improve with experience over 1000 steps, as in Figure 2.1. We call this suite of test tasks the 10-armed testbed. 
Figure 2.1 compares a greedy method with two $\boldsymbol{\varepsilon}$ -greedy methods ( $\varepsilon=0.01$ and $\varepsilon=0.1$ ), as described above, on the 10-armed testbed. Both methods formed their action-value estimates using the sample-average technique. The upper graph shows the increase in expected reward with experience. The greedy method improved slightly faster than the other methods at the very beginning, but then leveled off at a lower level. It achieved a reward per step of only about 1, compared with the best possible of about 1.55 on this testbed. The greedy method performs significantly worse in the long run because it often gets stuck performing suboptimal actions. The lower graph shows that the greedy method found the optimal action in only approximately one-third of the tasks. In the other two-thirds, its initial samples of the optimal action were disappointing, and it never returned to it. The $\varepsilon$ -greedy methods eventually perform better because they continue to explore, and to improve their chances of recognizing the optimal action. The $\varepsilon=0.1$ method explores more, and usually finds the optimal action earlier, but never selects it more than $91\%$ of the time. The $\varepsilon~=~0.01$ method improves more slowly, but eventually performs better than the $\varepsilon=0.1$ method on both performance measures. It is also possible to reduce $\varepsilon$ over time to try to get the best of both high and low values. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/86e8e8e3a9eb957248009ff0844849e2b415bea455948172748a2c84217f73ab.jpg) 
Figure 2.1: Average performance of $\varepsilon$ -greedy action-value methods on the 10-armed testbed. These data are averages over 2000 tasks. All methods used sample averages as their action-value estimates. The detailed structure at the beginning of these curves depends on how actions are selected when multiple actions have the same maximal action value. Here such ties were broken randomly. An alternative that has a similar effect is to add a very small amount of randomness to each of the initial action values, so that ties effectively never happen. 
The advantage of $\boldsymbol{\varepsilon}$ -greedy over greedy methods depends on the task. For example, suppose the reward variance had been larger, say 10 instead of 1. With noisier rewards it takes more exploration to find the optimal action, and $\varepsilon$ -greedy methods should fare even better relative to the greedy method. On the other hand, if the reward variances were zero, then the greedy method would know the true value of each action after trying it once. In this case the greedy method might actually perform best because it would soon find the optimal action and then never explore. But even in the deterministic case, there is a large advantage to exploring if we weaken some of the other assumptions. For example, suppose the bandit task were nonstationary, that is, that the true values of the actions changed over time. In this case exploration is needed even in the deterministic case to make sure one of the nongreedy actions has not changed to become better than the greedy one. As we will see in the next few chapters, effective nonstationarity is the case most commonly encountered in reinforcement learning. Even if the underlying task is stationary and deterministic, the learner faces a set of banditlike decision tasks each of which changes over time due to the learning process itself. Reinforcement learning requires a balance between exploration and exploitation. 
## 2.3 Incremental Implementation 
The action-value methods we have discussed so far all estimate action values as sample averages of observed rewards. The obvious implementation is to 
maintain, for each action $a$ , a record of all the rewards that have followed the selection of that action. Then, when the estimate of the value of action a is needed at time $t$ , it can be computed according to (2.1), which we repeat here: 
$$
Q_{t}(a)={\frac{R_{1}+R_{2}+\cdot\cdot\cdot+R_{N_{t}(a)}}{N_{t}(a)}},
$$ 
where here $R_{1},\ldots,R_{N_{t}(a)}$ are all the rewards received following all selections of action $a$ prior to play $t$ . A problem with this straightforward implementation is that its memory and computational requirements grow over time without bound. That is, each additional reward following a selection of action $a$ requires more memory to store it and results in more computation being required to determine $Q_{t}(a)$ . 
As you might suspect, this is not really necessary. It is easy to devise incremental update formulas for computing averages with small, constant computation required to process each new reward. For some action, let $Q_{k}$ denote the estimate for its $k$ th reward, that is, the average of its first $k-1$ rewards. Given this average and a $k$ th reward for the action, $R_{k}$ , then the average of all $k$ rewards can be computed by 
$$
{\begin{array}{r l}{Q_{k+1}}&{=}&{{\cfrac{1}{k}}\displaystyle\sum_{i=1}^{k}R_{i}}\ &{=}&{{\cfrac{1}{k}}\left(R_{k}+\displaystyle\sum_{i=1}^{k-1}R_{i}\right)}\ &{=}&{{\cfrac{1}{k}}\left(R_{k}+(k-1)Q_{k}+Q_{k}-Q_{k}\right)}\ &{=}&{{\cfrac{1}{k}}\left(R_{k}+k Q_{k}-Q_{k}\right)}\ &{=}&{Q_{k}+{\cfrac{1}{k}}\left[R_{k}-Q_{k}\right],}\end{array}}
$$ 
which holds even for $k=1$ , obtaining ${\cal Q}_{2}=R_{1}$ for arbitrary $Q_{1}$ . This implementation requires memory only for $Q_{k}$ and $k$ , and only the small computation (2.3) for each new reward. 
The update rule (2.3) is of a form that occurs frequently throughout this book. The general form is 
$$
N e w E s t i m a t e\leftarrow O l d E s t i m a t e+S t e p S i z e\left[T a r g e t-O l d E s t i m a t e\right].
$$ 
The expression $\lfloor\mathit{T a r g e t}-\mathit{O l d E s t i m a t e}\rfloor$ is an error in the estimate. It is reduced by taking a step toward the “Target.” The target is presumed to indicate a desirable direction in which to move, though it may be noisy. In the case above, for example, the target is the $k$ th reward. 
Note that the step-size parameter $(S t e p S i z e)$ used in the incremental method described above changes from time step to time step. In processing the $k$ th reward for action $a$ , that method uses a step-size parameter of $\frac{1}{k}$ . In this book we denote the step-size parameter by the symbol $\alpha$ or, more generally, by $\alpha_{t}(a)$ . We sometimes use the informal shorthand $\begin{array}{r}{\alpha=\frac{1}{k}}\end{array}$ to refer to this case, leaving the dependence of $k$ on the action implicit. 
## 2.4 Tracking a Nonstationary Problem 
The averaging methods discussed so far are appropriate in a stationary environment, but not if the bandit is changing over time. As noted earlier, we often encounter reinforcement learning problems that are effectively nonstationary. In such cases it makes sense to weight recent rewards more heavily than long-past ones. One of the most popular ways of doing this is to use a constant step-size parameter. For example, the incremental update rule (2.3) for updating an average $Q_{k}$ of the $k-1$ past rewards is modified to be 

$$
Q_{k+1}=Q_{k}+\alpha\Big[R_{k}-Q_{k}\Big],\tag{2.5}
$$ 
where the step-size parameter $\alpha\in(0,1]$ is constant. 

>  求平均的方法适合于静态环境，但不适合动态环境
>  动态环境中，需要为最近的奖励赋予相对于更早的奖励更多的权重，一种实现方法是使用常数步长参数

This results in $\mathit{Q}_{k+1}$ being a weighted average of past rewards and the initial estimate $Q_{1}$ : 

$$\begin{align*}
Q_{k+1} &= Q_{k} + \alpha [R_{k} - Q_{k}] \\
        &= \alpha R_{k} + (1 - \alpha) Q_{k} \\
        &= \alpha R_{k} + (1 - \alpha) [\alpha R_{k-1} + (1 - \alpha) Q_{k-1}] \\
        &= \alpha R_{k} + (1 - \alpha) \alpha R_{k-1} + (1 - \alpha)^{2} Q_{k-1} \\
        &= \alpha R_{k} + (1 - \alpha) \alpha R_{k-1} + (1 - \alpha)^{2} \alpha R_{k-2} + \cdots + (1 - \alpha)^{k-1} \alpha R_{1} + (1 - \alpha)^{k} Q_{1} \\
        &= (1 - \alpha)^{k} Q_{1} + \sum_{i=1}^{k} \alpha (1 - \alpha)^{k-i} R_{i}.
\end{align*}$$

We call this a weighted average because the sum of the weights is $(1-\alpha)^{k}+$ $\textstyle\sum_{i=1}^{k}\alpha(1-\alpha)^{k-i}=1$ , as you can check yourself. Note that the weight, $\alpha(1-$ $\alpha)^{k-i}$ , given to the reward $R_{i}$ depends on how many rewards ago, $k-i$ , it was observed. The quantity $1-\alpha$ is less than 1, and thus the weight given to $R_{i}$ decreases as the number of intervening rewards increases. In fact, the weight decays exponentially according to the exponent on $1-\alpha$ . (If $1-\alpha=0$ , then all the weight goes on the very last reward, $R_{k}$ , because of the convention that $0^{0}=1$ .) Accordingly, this is sometimes called an exponential, recency-weighted average. 

>  其中所有权重的和 $(1-\alpha)^k + \sum_{i=1}^k \alpha(1-\alpha)^{k-i}=1$
>  $R_i$ 的权重将随着奖励数量的增多而指数性下降，该方法也称为指数近期加权平均

Sometimes it is convenient to vary the step-size parameter from step to step. Let $\alpha_{k}(a)$ denote the step-size parameter used to process the reward received after the $k$ th selection of action $a$ . As we have noted, the choice $\begin{array}{r}{\alpha_{k}(a)=\frac{1}{k}}\end{array}$ results in the sample-average method, which is guaranteed to converge to the true action values by the law of large numbers. But of course convergence is not guaranteed for all choices of the sequence $\{\alpha_{k}(a)\}$ . A well-known result in stochastic approximation theory gives us the conditions required to assure convergence with probability 1: 

$$
\sum_{k=1}^{\infty}\alpha_{k}(a)=\infty\qquad\mathrm{~and~}\qquad\sum_{k=1}^{\infty}\alpha_{k}^{2}(a)<\infty.\tag{2.7}
$$ 
The first condition is required to guarantee that the steps are large enough to eventually overcome any initial conditions or random fluctuations. The second condition guarantees that eventually the steps become small enough to assure convergence. 

>  随着步数的前进，我们也可以更改步长参数
>  要确保收敛，步长参数需要满足以上条件，第一个条件是确保步长足够大，使得可以最终克服初始随机状态，第二个条件确保步长足够小以确保收敛

Note that both convergence conditions are met for the sample-average case, $\begin{array}{r}{\alpha_{k}(a)=\frac{1}{k}}\end{array}$ , but not for the case of constant step-size parameter, $\alpha_{k}(a)=\alpha$ . In the latter case, the second condition is not met, indicating that the estimates never completely converge but continue to vary in response to the most recently received rewards. As we mentioned above, this is actually desirable in a nonstationary environment, and problems that are effectively nonstationary are the norm in reinforcement learning. In addition, sequences of step-size parameters that meet the conditions (2.7) often converge very slowly or need considerable tuning in order to obtain a satisfactory convergence rate. Although sequences of step-size parameters that meet these convergence conditions are often used in theoretical work, they are seldom used in applications and empirical research. 
>  需要注意的是，样本平均情况下，$\alpha_k(\alpha) = \frac 1 k$，这两个收敛条件均得以满足，但对于固定步长参数的情况，$\alpha_k(\alpha)=\alpha$，则不满足。在后者中，第二个条件未能满足，这表明估计值不会完全收敛，而是会持续根据最近收到的奖励发生变化。正如我们之前所提到的，在非平稳环境中，这种情况实际上是可取的，而在强化学习中，实际上是非平稳的问题才是常态。此外，满足条件（2.7）的步长参数序列往往收敛非常缓慢，或者需要大量的调整才能获得令人满意的收敛速度。尽管在理论工作中经常使用满足这些收敛条件的步长参数序列，但在实际应用和经验研究中却很少使用。

## 2.5 Optimistic Initial Values 
All the methods we have discussed so far are dependent to some extent on the initial action-value estimates, $Q_{1}(a)$ . In the language of statistics, these methods are biased by their initial estimates. For the sample-average methods, the bias disappears once all actions have been selected at least once, but for methods with constant $\alpha$ , the bias is permanent, though decreasing over time as given by (2.6). In practice, this kind of bias is usually not a problem, and can sometimes be very helpful. The downside is that the initial estimates become, in effect, a set of parameters that must be picked by the user, if only to set them all to zero. The upside is that they provide an easy way to supply some prior knowledge about what level of rewards can be expected. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/144d8b770300d621fa0945502a26785c54da88a1bcba270edad712cf170d5c45.jpg) 
Figure 2.2: The effect of optimistic initial action-value estimates on the 10- armed testbed. Both methods used a constant step-size parameter, $\alpha=0.1$ . 
Initial action values can also be used as a simple way of encouraging exploration. Suppose that instead of setting the initial action values to zero, as we did in the 10-armed testbed, we set them all to $+5$ . Recall that the $q(a)$ in this problem are selected from a normal distribution with mean 0 and variance 1. An initial estimate of $+5$ is thus wildly optimistic. But this optimism encourages action-value methods to explore. Whichever actions are initially selected, the reward is less than the starting estimates; the learner switches to other actions, being “disappointed” with the rewards it is receiving. The result is that all actions are tried several times before the value estimates converge. The system does a fair amount of exploration even if greedy actions are selected all the time. 
Figure 2.2 shows the performance on the 10-armed bandit testbed of a greedy method using $Q_{1}(a)=+5$ , for all $a$ . For comparison, also shown is an $\varepsilon$ -greedy method with $Q_{1}(a)=0$ . Initially, the optimistic method performs worse because it explores more, but eventually it performs better because its exploration decreases with time. We call this technique for encouraging exploration optimistic initial values. We regard it as a simple trick that can be quite effective on stationary problems, but it is far from being a generally useful approach to encouraging exploration. For example, it is not well suited to nonstationary problems because its drive for exploration is inherently temporary. If the task changes, creating a renewed need for exploration, this method cannot help. Indeed, any method that focuses on the initial state in any special way is unlikely to help with the general nonstationary case. The beginning of time occurs only once, and thus we should not focus on it too much. This criticism applies as well to the sample-average methods, which also treat the beginning of time as a special event, averaging all subsequent rewards with equal weights. Nevertheless, all of these methods are very simple, and one of them or some simple combination of them is often adequate in practice. In the rest of this book we make frequent use of several of these simple exploration techniques. 
## 2.6 Upper-Confidence-Bound Action Selection 
Exploration is needed because the estimates of the action values are uncertain. The greedy actions are those that look best at present, but some of the other actions may actually be better. $\boldsymbol{\varepsilon}$ -greedy action selection forces the non-greedy actions to be tried, but indiscriminately, with no preference for those that are nearly greedy or particularly uncertain. It would be better to select among the non-greedy actions according to their potential for actually being optimal, taking into account both how close their estimates are to being maximal and the uncertainties in those estimates. One effective way of doing this is to select actions as 
$$
A_{t}=\underset{a}{\operatorname{\argmax}}\left[Q_{t}(a)+c\sqrt{\frac{\ln t}{N_{t}(a)}}\right],
$$ 
where $\ln t$ denotes the natural logarithm of $t$ (the number that $e\approx2.71828$ would have to be raised to in order to equal $t$ ), and the number $c>0$ controls the degree of exploration. If $N_{t}(a)=0$ , then $a$ is considered to be a maximizing action. 
The idea of this upper confidence bound (UCB) action selection is that the square-root term is a measure of the uncertainty or variance in the estimate of $a$ ’s value. The quantity being max’ed over is thus a sort of upper bound on the possible true value of action $a$ , with the $c$ parameter determining the confidence level. Each time $a$ is selected the uncertainty is presumably reduced; $N_{t}(a)$ is incremented and, as it appears in the denominator of the uncertainty term, the term is decreased. On the other hand, each time an action other $a$ is selected $t$ is increased; as it appears in the numerator the uncertainty estimate is increased. The use of the natural logarithm means that the increase gets smaller over time, but is unbounded; all actions will eventually be selected, but as time goes by it will be a longer wait, and thus a lower selection frequency, for actions with a lower value estimate or that have already been selected more times. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/91eb41f2d367fedff5b2e9ac78a40390abe8c8f96b628767b41d28f0bae238b0.jpg) 
Figure 2.3: Average performance of UCB action selection on the 10-armed testbed. As shown, UCB generally performs better that $\varepsilon$ -greedy action selection, except in the first $n$ plays, when it selects randomly among the as-yetunplayed actions. UCB with $c=1$ would perform even better but would not show the prominent spike in performance on the 11th play. Can you think of an explanation of this spike? 
Results with UCB on the 10-armed testbed are shown in Figure 2.3. UCB will often perform well, as shown here, but is more difficult than $\varepsilon$ -greedy to extend beyond bandits to the more general reinforcement learning settings considered in the rest of this book. One difficulty is in dealing with nonstationary problems; something more complex than the methods presented in Section 2.4 would be needed. Another difficulty is dealing with large state spaces, particularly function approximation as developed in Part III of this book. In these more advanced settings there is currently no known practical way of utilizing the idea of UCB action selection. 
## 2.7 Gradient Bandits 
So far in this chapter we have considered methods that estimate action values and use those estimates to select actions. This is often a good approach, but it is not the only one possible. In this section we consider learning a numerical preference $H_{t}(a)$ for each action $a$ . The larger the preference, the more often that action is taken, but the preference has no interpretation in terms of reward. Only the relative preference of one action over another is important; if we add 1000 to all the preferences there is no affect on the action probabilities, which are determined according to a soft-max distribution (i.e., Gibbs or Boltzmann distribution) as follows: 
$$
\operatorname*{Pr}\{A_{t}=a\}={\frac{e^{H_{t}(a))}}{\sum_{b=1}^{n}e^{H_{t}(b)}}}=\pi_{t}(a),
$$ 
where here we have also introduced a useful new notation $\pi_{t}(a)$ for the probability of taking action $a$ at time $t$ . Initially all preferences are the same (e.g., $H_{1}(a)=0,\forall a,$ so that all actions have an equal probability of being selected. 
There is a natural learning algorithm for this setting based on the idea of stochastic gradient ascent. On each step, after selecting the action $A_{t}$ and receiving the reward $R_{t}$ , the preferences are updated by: 
$$
\begin{array}{r l r}&{H_{t+1}(A_{t})=H_{t}(A_{t})+\alpha\big(R_{t}-\bar{R}_{t}\big)\big(1-\pi_{t}(A_{t})\big),}&{\quad\mathrm{and}}\ &{\quad H_{t+1}(a)=H_{t}(a)-\alpha\big(R_{t}-\bar{R}_{t}\big)\pi_{t}(a),}&{\quad\forall a\ne A_{t},}\end{array}
$$ 
where $\alpha>0$ is a step-size parameter, and $R_{t}\in\mathbb{R}$ is the average of all the rewards up through and including time $t$ , which can be computed incrementally as described in Section 2.3 (or Section 2.4 if the problem is nonstationary). The $R_{t}$ term serves as a baseline with which the reward is compared. If the reward is higher than the baseline, then the probability of taking $A_{t}$ in the future is increased, and if the reward is below baseline, then probability is decreased. The non-selected actions move in the opposite direction. 
Figure 2.4 shows results with the gradient-bandit algorithm on a variant of the 10-armed testbed in which the true expected rewards were selected according to a normal distribution with a mean of +4 instead of zero (and with unit variance as before). This shifting up of all the rewards has absolutely no affect on the gradient-bandit algorithm because of the reward baseline term, which instantaneously adapts to the new level. But if the baseline were omitted (that is, if $R_{t}$ was taken to be constant zero in (2.10)), then performance would be significantly degraded, as shown in the figure. 
One can gain a deeper insight into this algorithm by understanding it as a stochastic approximation to gradient ascent. In exact gradient ascent, each preference $H_{t}(a)$ would be incrementing proportional to the increment’s effect on performance: 
$$
H_{t+1}(a)=H_{t}(a)+\alpha\frac{\partial\mathbb{E}\left[R_{t}\right]}{\partial H_{t}(a)},
$$ 
where the measure of performance here is the expected reward: 
$$
\mathbb{E}[R_{t}]=\sum_{b}\pi_{t}(b)q(b).
$$ 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/b4cb97cc9f719102b2aa107b2aec65400bf5dfdb0196cc56c8eb0011381a4b6a.jpg) 
Figure 2.4: Average performance of the gradient-bandit algorithm with and without a reward baseline on the 10-armed testbed with $\mathbb{E}[q(a)]=4$ . 
Of course, it is not possible to implement gradient ascent exactly in our case because by assumption we do not know the $q(b)$ , but in fact the updates of our algorithm (2.10) are equal to (2.11) in expected value, making the algorithm an instance of stochastic gradient ascent. 
The calculations showing this require only beginning calculus, but take several steps. If you are mathematically inclined, then you will enjoy the rest of this section in which we go through these steps. First we take a closer look at the exact performance gradient: 
$$
\begin{array}{l}{\displaystyle\frac{\partial\mathbb{E}[R_{t}]}{\partial H_{t}(a)}=\frac{\partial}{\partial H_{t}(a)}\left[\sum_{b}\pi_{t}(b)q(b)\right]}\ {\displaystyle=\sum_{b}q(b)\frac{\partial\pi_{t}(b)}{\partial H_{t}(a)}}\ {\displaystyle=\sum_{b}\big(q(b)-X_{t}\big)\frac{\partial\pi_{t}(b)}{\partial H_{t}(a)},}\end{array}
$$ 
where $X_{t}$ can be any scalar that does not depend on $b$ . We can include it here because the gradient sums to zero over the all the actions, b ∂∂Hπtt((ba)) . As $H_{t}(a)$ is changed, some actions’ probabilities go up and some down, but the sum of the changes must be zero because the sum of the probabilities must 
remain one. 
$$
=\sum_{b}\pi_{t}(b)\left(q(b)-X_{t}\right)\frac{\partial\pi_{t}(b)}{\partial H_{t}(a)}/\pi_{t}(b)
$$ 
The equation is now in the form of an expectation, summing over all possible values $b$ of the random variable $A_{t}$ , then multiplying by the probability of taking those values. Thus: 
$$
\begin{array}{r l}&{=\mathbb{E}\bigg[\big(q(A_{t})-X_{t}\big)\frac{\partial\pi_{t}(A_{t})}{\partial H_{t}(a)}/\pi_{t}(A_{t})\bigg]}\ &{=\mathbb{E}\bigg[\big(R_{t}-\bar{R}_{t}\big)\frac{\partial\pi_{t}(A_{t})}{\partial H_{t}(a)}/\pi_{t}(A_{t})\bigg],}\end{array}
$$ 
where here we have chosen $X_{t}\:=\:R_{t}$ and substituted $R_{t}$ for $q(A_{t})$ , which is permitted because $\mathbb{E}[R_{t}]\:=\:q(A_{t})$ and because all the other factors are nonrandom. Shortly we will establish that $\begin{array}{r}{\frac{\partial\pi_{t}(b)}{\partial H_{t}(a)}=\pi_{t}(b)\left(\mathbb{I}_{a=b}-\pi_{t}(a)\right)}\end{array}$ , where $\mathbb{I}_{a=b}$ is defined to be 1 if $a=b$ , else $0$ . Assuming that for now we have 
$$
\begin{array}{r l}&{=\mathbb{E}\left[\left(R_{t}-\bar{R}_{t}\right)\pi_{t}(A_{t})\left(\mathbb{I}_{a=A_{t}}-\pi_{t}(a)\right)/\pi_{t}(A_{t})\right]}\ &{=\mathbb{E}\left[\left(R_{t}-\bar{R}_{t}\right)\left(\mathbb{I}_{a=A_{t}}-\pi_{t}(a)\right)\right].}\end{array}
$$ 
Recall that our plan has been to write the performance gradient as an expectation of something that we can sample on each step, as we have just done, and then update on each step proportional to the sample. Substituting a sample of the expectation above for the performance gradient in (2.11) yields: 
$$
H_{t+1}(a)=H_{t}(a)+\alpha{\bigl(}R_{t}-{\bar{R}}_{t}{\bigr)}{\bigl(}\mathbb{I}_{a-A_{t}}-\pi_{t}(a){\bigr)},\qquad\forall a,
$$ 
which you will recognize as being equivalent to our original algorithm (2.10). 
Thus it remains only to show that $\begin{array}{r}{\frac{\partial\pi_{t}(b)}{\partial H_{t}(a)}=\pi_{t}(b)\big(\mathbb{I}_{a=b}-\pi_{t}(a)\big)}\end{array}$ , as we assumed earlier. Recall the standard quotient rule for derivatives: 
$$
{\frac{\partial}{\partial x}}\left[{\frac{f(x)}{g(x)}}\right]={\frac{{\frac{\partial f(x)}{\partial x}}g(x)-f(x){\frac{\partial g(x)}{\partial x}}}{g(x)^{2}}}.
$$ 
Using this, we can write 
$$
\begin{array}{r l}{\lefteqn{\frac{\partial\pi_{\ell}(k)}{\partial H_{\ell}(a)}=\frac{\partial}{\partial H_{\ell}(a)}\pi_{\ell}(b)}}\ &{=\frac{\partial}{\partial H_{\ell}(a)}\left[\sum_{\sum_{n=1}^{n}\in H_{\ell}(k)}^{\ell\lfloor(k)\ell\rfloor}\right]}\ &{=\frac{\frac{\partial H_{\ell}(a)}{\partial H_{\ell}(a)}\sum_{n=1}^{n}e^{H_{\ell}(a)}-e^{H_{\ell}(a)}\left[\theta\right]^{2}}{\left(\sum_{n=1}^{n}e^{H_{\ell}(a)}\right)^{2}}}\ &{=\frac{\frac{\prod_{a=0}\nu_{\ell}H_{\ell}(a)}{\sum_{n=1}^{n}e^{H_{\ell}(a)}\sum_{n=1}^{n}e^{H_{\ell}(a)}-e^{H_{\ell}(b)}e^{H_{\ell}(a)}}}{\left(\sum_{n=1}^{n}e^{H_{\ell}(b)}\right)^{2}}}\ &{=\frac{\prod_{a=0}e^{H_{\ell}(b)}\left(\sum_{n=1}^{n}e^{H_{\ell}(b)}-e^{H_{\ell}(b)}e^{H_{\ell}(a)}\right)}{\sum_{n=1}e^{H_{\ell}(b)}\left(\sum_{n=1}^{n}e^{H_{\ell}(b)}\right)^{2}}}\ &{=\frac{\prod_{a=0}e^{H_{\ell}(b)}}{\sum_{n=1}e^{H_{\ell}(b)}\left(\sum_{n=1}^{n}e^{H_{\ell}(b)}+\left(\sum_{n=1}^{n}e^{H_{\ell}(b)}\right)^{2}\right)}}\ &{=\mathbb{I}_{a\to\infty}e_{\ell}(b)\left(\sum_{n=\ell}\nu_{\ell}(a)\right).}\end{array}
$$ 
(by the quotient rule) 
We have just shown that the expected update of the gradient-bandit algorithm is equal to the gradient of expected reward, and thus that the algorithm is an instance of stochastic gradient ascent. This assures us that the algorithm has robust convergence properties. 
Note that we did not require anything of the reward baseline other than that it not depend on the selected action. For example, we could have set is to zero, or to 1000, and the algorithm would still have been an instance of stochastic gradient ascent. The choice of the baseline does not affect the expected update of the algorithm, but it does affect the variance of the update and thus the rate of convergence (as shown, e.g., in Figure 2.4). Choosing it as the average of the rewards may not be the very best, but it is simple and works well in practice. 
## 2.8 Associative Search (Contextual Bandits) 
So far in this chapter we have considered only nonassociative tasks, in which there is no need to associate different actions with different situations. In these tasks the learner either tries to find a single best action when the task is stationary, or tries to track the best action as it changes over time when the task is nonstationary. However, in a general reinforcement learning task there is more than one situation, and the goal is to learn a policy: a mapping from situations to the actions that are best in those situations. To set the stage for the full problem, we briefly discuss the simplest way in which nonassociative tasks extend to the associative setting. 
As an example, suppose there are several different $n$ -armed bandit tasks, and that on each play you confront one of these chosen at random. Thus, the bandit task changes randomly from play to play. This would appear to you as a single, nonstationary $n$ -armed bandit task whose true action values change randomly from play to play. You could try using one of the methods described in this chapter that can handle nonstationarity, but unless the true action values change slowly, these methods will not work very well. Now suppose, however, that when a bandit task is selected for you, you are given some distinctive clue about its identity (but not its action values). Maybe you are facing an actual slot machine that changes the color of its display as it changes its action values. Now you can learn a policy associating each task, signaled by the color you see, with the best action to take when facing that task—for instance, if red, play arm 1; if green, play arm 2. With the right policy you can usually do much better than you could in the absence of any information distinguishing one bandit task from another. 
This is an example of an associative search task, so called because it involves both trial-and-error learning in the form of search for the best actions and association of these actions with the situations in which they are best.2 Associative search tasks are intermediate between the $n$ -armed bandit problem and the full reinforcement learning problem. They are like the full reinforcement learning problem in that they involve learning a policy, but like our version of the $n$ -armed bandit problem in that each action affects only the immediate reward. If actions are allowed to affect the next situation as well as the reward, then we have the full reinforcement learning problem. We present this problem in the next chapter and consider its ramifications throughout the rest of the book. 
## 2.9 Summary 
We have presented in this chapter several simple ways of balancing exploration and exploitation. The $\boldsymbol{\varepsilon}$ -greedy methods choose randomly a small fraction of the time, whereas UCB methods choose deterministically but achieve exploration by subtly favoring at each step the actions that have so far received fewer samples. Gradient-bandit algorithms estimate not action values, but action preferences, and favor the more preferred actions in a graded, probabalistic manner using a soft-max distribution. The simple expedient of initializing estimates optimistically causes even greedy methods to explore significantly. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/912f29eb3df2c09d80f2742ae0a6daf43e3736829ce0457164843dfc35c7cb21.jpg) 
Figure 2.5: A parameter study of the various bandit algorithms presented in this chapter. Each point is the average reward obtained over 1000 steps with a particular algorithm at a particular setting of its parameter. 
It is natural to ask which of these methods is best. Although this is a difficult question to answer in general, we can certainly run them all on the 10-armed testbed that we have used throughout this chapter and compare their performances. A complication is that they all have a parameter; to get a meaningful comparison we will have to consider their performance as a function of their parameter. Our graphs so far have shown the course of learning over time for each algorithm and parameter setting, but it would be too visually confusing to show such a learning curve for each algorithm and parameter value. Instead we summarize a complete learning curve by its average value over the 1000 steps; this value is proportional to the area under the learning curves we have shown up to now. Figure 2.5 shows this measure for the various bandit algorithms from this chapter, each as a function of its own parameter shown on a single scale on the x-axis. Note that the parameter values are varied by factors of two and presented on a log scale. Note also the characteristic inverted-U shapes of each algorithm’s performance; all the algorithms perform best at an intermediate value of their parameter, neither too large nor too big. In assessing an method, we should attend not just to how well it does at its best parameter setting, but also to how sensitive it is to its parameter value. All of these algorithms are fairly insensitive, performing well over a range of parameter values varying by about an order of magnitude. Overall, on this problem, UCB seems to perform best. 
Despite their simplicity, in our opinion the methods presented in this chapter can fairly be considered the state of the art. There are more sophisticated methods, but their complexity and assumptions make them impractical for the full reinforcement learning problem that is our real focus. Starting in Chapter 5 we present learning methods for solving the full reinforcement learning problem that use in part the simple methods explored in this chapter. 
Although the simple methods explored in this chapter may be the best we can do at present, they are far from a fully satisfactory solution to the problem of balancing exploration and exploitation. 
The classical solution to balancing exploration and exploitation in $n$ -armed bandit problems is to compute special functions called Gittins indices. These provide an optimal solution to a certain kind of bandit problem more general than that considered here but that assumes the prior distribution of possible problems is known. Unfortunately, neither the theory nor the computational tractability of this method appear to generalize to the full reinforcement learning problem that we consider in the rest of the book. 
There is also a well-known algorithm for computing the Bayes optimal way to balance exploration and exploitation. This method is computationally intractable when done exactly, but there may be efficient ways to approximate it. In this method we assume that we know the distribution of problem instances, that is, the probability of each possible set of true action values. Given any action selection, we can then compute the probability of each possible immediate reward and the resultant posterior probability distribution over action values. This evolving distribution becomes the information state of the problem. Given a horizon, say 1000 plays, one can consider all possible actions, all possible resulting rewards, all possible next actions, all next rewards, and so on for all 1000 plays. Given the assumptions, the rewards and probabilities of each possible chain of events can be determined, and one need only pick the best. But the tree of possibilities grows extremely rapidly; even if there are only two actions and two rewards, the tree will have $2^{2000}$ leaves. This approach effectively turns the bandit problem into an instance of the full reinforcement learning problem. In the end, we may be able to use reinforcement learning methods to approximate this optimal solution. But that is a topic for current research and beyond the scope of this book. 
# 3 Finite Markov Decision Processes 
In this chapter we introduce the problem that we try to solve in the rest of the book. For us, this problem defines the field of reinforcement learning: any method that is suited to solving this problem we consider to be a reinforcement learning method. 
Our objective in this chapter is to describe the reinforcement learning problem in a broad sense. We try to convey the wide range of possible applications that can be framed as reinforcement learning tasks. We also describe mathematically idealized forms of the reinforcement learning problem for which precise theoretical statements can be made. We introduce key elements of the problem’s mathematical structure, such as value functions and Bellman equations. As in all of artificial intelligence, there is a tension between breadth of applicability and mathematical tractability. In this chapter we introduce this tension and discuss some of the trade-offs and challenges that it implies. 

## 3.1 The Agent–Environment Interface 
The reinforcement learning problem is meant to be a straightforward framing of the problem of learning from interaction to achieve a goal. The learner and decision-maker is called the agent. The thing it interacts with, comprising everything outside the agent, is called the environment. These interact continually, the agent selecting actions and the environment responding to those actions and presenting new situations to the agent.1 The environment also gives rise to rewards, special numerical values that the agent tries to maximize over time. A complete specification of an environment defines a task, one instance of the reinforcement learning problem. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/676517caabf785249e0b9ec8a44c3b55f5606549c89604f62e7d5edddc5a6c3a.jpg) 
Figure 3.1: The agent–environment interaction in reinforcement learning. 
More specifically, the agent and environment interact at each of a sequence of discrete time steps, $t=0,1,2,3,\ldots$ .2 At each time step $t$ , the agent receives some representation of the environment’s state, $S_{t}\in\mathcal{S}$ , where S is the set of possible states, and on that basis selects an action, $A_{t}\in\mathcal{A}(S_{t})$ , where $\mathcal{A}(S_{t})$ is the set of actions available in state $S_{t}$ . One time step later, in part as a consequence of its action, the agent receives a numerical reward, $R_{t+1}\in$ $\mathcal{R}\subset\mathbb{R}$ , and finds itself in a new state, $S_{t+1}$ .3 Figure 3.1 diagrams the agent– environment interaction. 
At each time step, the agent implements a mapping from states to probabilities of selecting each possible action. This mapping is called the agent’s policy and is denoted $\pi_{t}$ , where $\pi_{t}(a|s)$ is the probability that $A_{t}=a$ if $S_{t}=s$ . Reinforcement learning methods specify how the agent changes its policy as a result of its experience. The agent’s goal, roughly speaking, is to maximize the total amount of reward it receives over the long run. 
This framework is abstract and flexible and can be applied to many different problems in many different ways. For example, the time steps need not refer to fixed intervals of real time; they can refer to arbitrary successive stages of decision-making and acting. The actions can be low-level controls, such as the voltages applied to the motors of a robot arm, or high-level decisions, such as whether or not to have lunch or to go to graduate school. Similarly, the states can take a wide variety of forms. They can be completely determined by low-level sensations, such as direct sensor readings, or they can be more high-level and abstract, such as symbolic descriptions of objects in a room. Some of what makes up a state could be based on memory of past sensations or even be entirely mental or subjective. For example, an agent could be in the state of not being sure where an object is, or of having just been surprised in some clearly defined sense. Similarly, some actions might be totally mental or computational. For example, some actions might control what an agent chooses to think about, or where it focuses its attention. In general, actions can be any decisions we want to learn how to make, and the states can be anything we can know that might be useful in making them. 
In particular, the boundary between agent and environment is not often the same as the physical boundary of a robot’s or animal’s body. Usually, the boundary is drawn closer to the agent than that. For example, the motors and mechanical linkages of a robot and its sensing hardware should usually be considered parts of the environment rather than parts of the agent. Similarly, if we apply the framework to a person or animal, the muscles, skeleton, and sensory organs should be considered part of the environment. Rewards, too, presumably are computed inside the physical bodies of natural and artificial learning systems, but are considered external to the agent. 
The general rule we follow is that anything that cannot be changed arbitrarily by the agent is considered to be outside of it and thus part of its environment. We do not assume that everything in the environment is unknown to the agent. For example, the agent often knows quite a bit about how its rewards are computed as a function of its actions and the states in which they are taken. But we always consider the reward computation to be external to the agent because it defines the task facing the agent and thus must be beyond its ability to change arbitrarily. In fact, in some cases the agent may know everything about how its environment works and still face a difficult reinforcement learning task, just as we may know exactly how a puzzle like Rubik’s cube works, but still be unable to solve it. The agent– environment boundary represents the limit of the agent’s absolute control, not of its knowledge. 
The agent–environment boundary can be located at different places for different purposes. In a complicated robot, many different agents may be operating at once, each with its own boundary. For example, one agent may make high-level decisions which form part of the states faced by a lower-level agent that implements the high-level decisions. In practice, the agent–environment boundary is determined once one has selected particular states, actions, and rewards, and thus has identified a specific decision-making task of interest. 
The reinforcement learning framework is a considerable abstraction of the problem of goal-directed learning from interaction. It proposes that whatever the details of the sensory, memory, and control apparatus, and whatever objective one is trying to achieve, any problem of learning goal-directed behavior can be reduced to three signals passing back and forth between an agent and its environment: one signal to represent the choices made by the agent (the actions), one signal to represent the basis on which the choices are made (the states), and one signal to define the agent’s goal (the rewards). This framework may not be sufficient to represent all decision-learning problems usefully, but it has proved to be widely useful and applicable. 
Of course, the particular states and actions vary greatly from task to task, and how they are represented can strongly affect performance. In reinforcement learning, as in other kinds of learning, such representational choices are at present more art than science. In this book we offer some advice and examples regarding good ways of representing states and actions, but our primary focus is on general principles for learning how to behave once the representations have been selected. 
Example 3.1: Bioreactor Suppose reinforcement learning is being applied to determine moment-by-moment temperatures and stirring rates for a bioreactor (a large vat of nutrients and bacteria used to produce useful chemicals). The actions in such an application might be target temperatures and target stirring rates that are passed to lower-level control systems that, in turn, directly activate heating elements and motors to attain the targets. The states are likely to be thermocouple and other sensory readings, perhaps filtered and delayed, plus symbolic inputs representing the ingredients in the vat and the target chemical. The rewards might be moment-by-moment measures of the rate at which the useful chemical is produced by the bioreactor. Notice that here each state is a list, or vector, of sensor readings and symbolic inputs, and each action is a vector consisting of a target temperature and a stirring rate. It is typical of reinforcement learning tasks to have states and actions with such structured representations. Rewards, on the other hand, are always single numbers. 
Example 3.2: Pick-and-Place Robot Consider using reinforcement learning to control the motion of a robot arm in a repetitive pick-and-place task. If we want to learn movements that are fast and smooth, the learning agent will have to control the motors directly and have low-latency information about the current positions and velocities of the mechanical linkages. The actions in this case might be the voltages applied to each motor at each joint, and the states might be the latest readings of joint angles and velocities. The reward might be $+1$ for each object successfully picked up and placed. To encourage smooth movements, on each time step a small, negative reward can be given as a function of the moment-to-moment “jerkiness” of the motion. 
Example 3.3: Recycling Robot A mobile robot has the job of collecting empty soda cans in an office environment. It has sensors for detecting cans, and an arm and gripper that can pick them up and place them in an onboard bin; it runs on a rechargeable battery. The robot’s control system has components for interpreting sensory information, for navigating, and for controlling the arm and gripper. High-level decisions about how to search for cans are made by a reinforcement learning agent based on the current charge level of the battery. This agent has to decide whether the robot should (1) actively search for a can for a certain period of time, (2) remain stationary and wait for someone to bring it a can, or (3) head back to its home base to recharge its battery. This decision has to be made either periodically or whenever certain events occur, such as finding an empty can. The agent therefore has three actions, and its state is determined by the state of the battery. The rewards might be zero most of the time, but then become positive when the robot secures an empty can, or large and negative if the battery runs all the way down. In this example, the reinforcement learning agent is not the entire robot. The states it monitors describe conditions within the robot itself, not conditions of the robot’s external environment. The agent’s environment therefore includes the rest of the robot, which might contain other complex decision-making systems, as well as the robot’s external environment.  

## 3.2 Goals and Rewards 
In reinforcement learning, the purpose or goal of the agent is formalized in terms of a special reward signal passing from the environment to the agent. At each time step, the reward is a simple number, $R_{t}\in\mathbb R$ . Informally, the agent’s goal is to maximize the total amount of reward it receives. This means maximizing not immediate reward, but cumulative reward in the long run. We can clearly state this informal idea as the reward hypothesis: 
That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward). 
The use of a reward signal to formalize the idea of a goal is one of the most distinctive features of reinforcement learning. 
Although formulating goals in terms of reward signals might at first appear limiting, in practice it has proved to be flexible and widely applicable. The best way to see this is to consider examples of how it has been, or could be, used. For example, to make a robot learn to walk, researchers have provided reward on each time step proportional to the robot’s forward motion. In making a robot learn how to escape from a maze, the reward is often $-1$ for every time step that passes prior to escape; this encourages the agent to escape as quickly as possible. To make a robot learn to find and collect empty soda cans for recycling, one might give it a reward of zero most of the time, and then a reward of $+1$ for each can collected. One might also want to give the robot negative rewards when it bumps into things or when somebody yells at it. For an agent to learn to play checkers or chess, the natural rewards are $+1$ for winning, $-1$ for losing, and 0 for drawing and for all nonterminal positions. 
You can see what is happening in all of these examples. The agent always learns to maximize its reward. If we want it to do something for us, we must provide rewards to it in such a way that in maximizing them the agent will also achieve our goals. It is thus critical that the rewards we set up truly indicate what we want accomplished. In particular, the reward signal is not the place to impart to the agent prior knowledge about how to achieve what we want it to do.4 For example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such taking its opponent’s pieces or gaining control of the center of the board. If achieving these sorts of subgoals were rewarded, then the agent might find a way to achieve them without achieving the real goal. For example, it might find a way to take the opponent’s pieces even at the cost of losing the game. The reward signal is your way of communicating to the robot what you want it to achieve, not how you want it achieved. 
Newcomers to reinforcement learning are sometimes surprised that the rewards—which define of the goal of learning—are computed in the environment rather than in the agent. Certainly most ultimate goals for animals are recognized by computations occurring inside their bodies, for example, by sensors for recognizing food, hunger, pain, and pleasure. Nevertheless, as we discussed in the previous section, one can redraw the agent–environment interface in such a way that these parts of the body are considered to be outside of the agent (and thus part of the agent’s environment). For example, if the goal concerns a robot’s internal energy reservoirs, then these are considered to be part of the environment; if the goal concerns the positions of the robot’s limbs, then these too are considered to be part of the environment—that is, the agent’s boundary is drawn at the interface between the limbs and their control systems. These things are considered internal to the robot but external to the learning agent. For our purposes, it is convenient to place the boundary of the learning agent not at the limit of its physical body, but at the limit of 
its control. 
The reason we do this is that the agent’s ultimate goal should be something over which it has imperfect control: it should not be able, for example, to simply decree that the reward has been received in the same way that it might arbitrarily change its actions. Therefore, we place the reward source outside of the agent. This does not preclude the agent from defining for itself a kind of internal reward, or a sequence of internal rewards. Indeed, this is exactly what many reinforcement learning methods do. 

## 3.3 Returns 
So far we have discussed the objective of learning informally. We have said that the agent’s goal is to maximize the cumulative reward it receives in the long run. How might this be defined formally? If the sequence of rewards received after time step $t$ is denoted $R_{t+1},R_{t+2},R_{t+3},...$ , then what precise aspect of this sequence do we wish to maximize? In general, we seek to maximize the expected return, where the return $G_{t}$ is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards: 

$$
G_{t}=R_{t+1}+R_{t+2}+R_{t+3}+\cdot\cdot\cdot+R_{T},
$$ 
where $T$ is a final time step. This approach makes sense in applications in which there is a natural notion of final time step, that is, when the agent– environment interaction breaks naturally into subsequences, which we call episodes,5 such as plays of a game, trips through a maze, or any sort of repeated interactions. Each episode ends in a special state called the terminal state, followed by a reset to a standard starting state or to a sample from a standard distribution of starting states. Tasks with episodes of this kind are called episodic tasks. In episodic tasks we sometimes need to distinguish the set of all nonterminal states, denoted S, from the set of all states plus the terminal state, denoted $\mathcal{S}^{+}$ . 
On the other hand, in many cases the agent–environment interaction does not break naturally into identifiable episodes, but goes on continually without limit. For example, this would be the natural way to formulate a continual process-control task, or an application to a robot with a long life span. We call these continuing tasks. The return formulation (3.1) is problematic for continuing tasks because the final time step would be $T=\infty$ , and the return, which is what we are trying to maximize, could itself easily be infinite. (For example, suppose the agent receives a reward of $+1$ at each time step.) Thus, in this book we usually use a definition of return that is slightly more complex conceptually but much simpler mathematically. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/6d463a9c341f2fa798a155f1216e02f0eff9446cb432003076828d01f351f57b.jpg) 
Figure 3.2: The pole-balancing task. 
The additional concept that we need is that of discounting. According to this approach, the agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized. In particular, it chooses $A_{t}$ to maximize the expected discounted return: 
$$
G_{t}=R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\cdot\cdot\cdot=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1},
$$ 
where $\gamma$ is a parameter, $0\leq\gamma\leq1$ , called the discount rate. 
The discount rate determines the present value of future rewards: a reward received $k$ time steps in the future is worth only $\gamma^{k-1}$ times what it would be worth if it were received immediately. If $\gamma<1$ , the infinite sum has a finite value as long as the reward sequence $\{R_{k}\}$ is bounded. If $\gamma=0$ , the agent is “myopic” in being concerned only with maximizing immediate rewards: its objective in this case is to learn how to choose $A_{t}$ so as to maximize only $R_{t+1}$ . If each of the agent’s actions happened to influence only the immediate reward, not future rewards as well, then a myopic agent could maximize (3.2) by separately maximizing each immediate reward. But in general, acting to maximize immediate reward can reduce access to future rewards so that the return may actually be reduced. As $\gamma$ approaches 1, the objective takes future rewards into account more strongly: the agent becomes more farsighted. 
Example 3.4: Pole-Balancing Figure 3.2 shows a task that served as an early illustration of reinforcement learning. The objective here is to apply forces to a cart moving along a track so as to keep a pole hinged to the cart from falling over. A failure is said to occur if the pole falls past a given angle from vertical or if the cart runs off the track. The pole is reset to vertical after each failure. This task could be treated as episodic, where the natural episodes are the repeated attempts to balance the pole. The reward in this case could be $+1$ for every time step on which failure did not occur, so that the return at each time would be the number of steps until failure. Alternatively, we could treat pole-balancing as a continuing task, using discounting. In this case the reward would be $-1$ on each failure and zero at all other times. The return at each time would then be related to $-\gamma^{K}$ , where $K$ is the number of time steps before failure. In either case, the return is maximized by keeping the pole balanced for as long as possible. 

## 3.4 Unified Notation for Episodic and Continuing Tasks 
In the preceding section we described two kinds of reinforcement learning tasks, one in which the agent–environment interaction naturally breaks down into a sequence of separate episodes (episodic tasks), and one in which it does not (continuing tasks). The former case is mathematically easier because each action affects only the finite number of rewards subsequently received during the episode. In this book we consider sometimes one kind of problem and sometimes the other, but often both. It is therefore useful to establish one notation that enables us to talk precisely about both cases simultaneously. 
To be precise about episodic tasks requires some additional notation. Rather than one long sequence of time steps, we need to consider a series of episodes, each of which consists of a finite sequence of time steps. We number the time steps of each episode starting anew from zero. Therefore, we have to refer not just to $S_{t}$ , the state representation at time $t$ , but to $S_{t,i}$ , the state representation at time $t$ of episode $i$ (and similarly for $A_{t,i}$ , $R_{t,i}$ , $\boldsymbol{\pi}_{t,i}$ , $T_{i}$ , etc.). However, it turns out that, when we discuss episodic tasks we will almost never have to distinguish between different episodes. We will almost always be considering a particular single episode, or stating something that is true for all episodes. Accordingly, in practice we will almost always abuse notation slightly by dropping the explicit reference to episode number. That is, we will write $S_{t}$ to refer to $S_{t,i}$ , and so on. 
We need one other convention to obtain a single notation that covers both episodic and continuing tasks. We have defined the return as a sum over a finite number of terms in one case (3.1) and as a sum over an infinite number of terms in the other (3.2). These can be unified by considering episode termination to be the entering of a special absorbing state that transitions only to itself and that generates only rewards of zero. For example, consider the state transition 
diagram 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/6894782f0790589b5ad37302228b879381bb6c96779b17f923f4a53da790c8ef.jpg) 
Here the solid square represents the special absorbing state corresponding to the end of an episode. Starting from $S_{0}$ , we get the reward sequence $+1,+1,+1,0,0,0,\ldots$ . Summing these, we get the same return whether we sum over the first $T$ rewards (here $T=3$ ) or over the full infinite sequence. This remains true even if we introduce discounting. Thus, we can define the return, in general, according to (3.2), using the convention of omitting episode numbers when they are not needed, and including the possibility that $\gamma=1$ if the sum remains defined (e.g., because all episodes terminate). Alternatively, we can also write the return as 
$$
G_{t}=\sum_{k=0}^{T-t-1}\gamma^{k}R_{t+k+1},
$$ 
including the possibility that $T=\infty$ or $\gamma=1$ (but not both $^6$ ). We use these conventions throughout the rest of the book to simplify notation and to express the close parallels between episodic and continuing tasks. 
## 3.5 The Markov Property\*
In the reinforcement learning framework, the agent makes its decisions as a function of a signal from the environment called the environment’s state. In this section we discuss what is required of the state signal, and what kind of information we should and should not expect it to provide. In particular, we formally define a property of environments and their state signals that is of particular interest, called the Markov property. 
In this book, by “the state” we mean whatever information is available to the agent. We assume that the state is given by some preprocessing system that is nominally part of the environment. We do not address the issues of constructing, changing, or learning the state signal in this book. We take this approach not because we consider state representation to be unimportant, but in order to focus fully on the decision-making issues. In other words, our main concern is not with designing the state signal, but with deciding what action to take as a function of whatever state signal is available. 
Certainly the state signal should include immediate sensations such as sensory measurements, but it can contain much more than that. State representations can be highly processed versions of original sensations, or they can be complex structures built up over time from the sequence of sensations. For example, we can move our eyes over a scene, with only a tiny spot corresponding to the fovea visible in detail at any one time, yet build up a rich and detailed representation of a scene. Or, more obviously, we can look at an object, then look away, and know that it is still there. We can hear the word “yes” and consider ourselves to be in totally different states depending on the question that came before and which is no longer audible. At a more mundane level, a control system can measure position at two different times to produce a state representation including information about velocity. In all of these cases the state is constructed and maintained on the basis of immediate sensations together with the previous state or some other memory of past sensations. In this book, we do not explore how that is done, but certainly it can be and has been done. There is no reason to restrict the state representation to immediate sensations; in typical applications we should expect the state representation to be able to inform the agent of more than that. 
On the other hand, the state signal should not be expected to inform the agent of everything about the environment, or even everything that would be useful to it in making decisions. If the agent is playing blackjack, we should not expect it to know what the next card in the deck is. If the agent is answering the phone, we should not expect it to know in advance who the caller is. If the agent is a paramedic called to a road accident, we should not expect it to know immediately the internal injuries of an unconscious victim. In all of these cases there is hidden state information in the environment, and that information would be useful if the agent knew it, but the agent cannot know it because it has never received any relevant sensations. In short, we don’t fault an agent for not knowing something that matters, but only for having known something and then forgotten it! 
What we would like, ideally, is a state signal that summarizes past sensations compactly, yet in such a way that all relevant information is retained. This normally requires more than the immediate sensations, but never more than the complete history of all past sensations. A state signal that succeeds in retaining all relevant information is said to be Markov, or to have the Markov property (we define this formally below). For example, a checkers position—the current configuration of all the pieces on the board—would serve as a Markov state because it summarizes everything important about the complete sequence of positions that led to it. Much of the information about the sequence is lost, but all that really matters for the future of the game is retained. Similarly, the current position and velocity of a cannonball is all that matters for its future flight. It doesn’t matter how that position and velocity came about. This is sometimes also referred to as an “independence of path” property because all that matters is in the current state signal; its meaning is independent of the “path,” or history, of signals that have led up to it. 
We now formally define the Markov property for the reinforcement learning problem. To keep the mathematics simple, we assume here that there are a finite number of states and reward values. This enables us to work in terms of sums and probabilities rather than integrals and probability densities, but the argument can easily be extended to include continuous states and rewards. Consider how a general environment might respond at time $t+1$ to the action taken at time $t$ . In the most general, causal case this response may depend on everything that has happened earlier. In this case the dynamics can be defined only by specifying the complete probability distribution: 
$$
\operatorname*{Pr}\{R_{t+1}=r,S_{t+1}=s^{\prime}\mid S_{0},A_{0},R_{1},\ldots,S_{t-1},A_{t-1},R_{t},S_{t},A_{t}\},
$$ 
for all $r,s^{\prime}$ , and all possible values of the past events: $S_{0}$ , $A_{0}$ , $R_{1}$ , ..., $S_{t-1}$ , $A_{t-1}$ , $R_{t}$ , $S_{t}$ , $A_{t}$ . If the state signal has the Markov property, on the other hand, then the environment’s response at $t+1$ depends only on the state and action representations at $t$ , in which case the environment’s dynamics can be defined by specifying only 
$$
p(s^{\prime},r|s,a)=\operatorname*{Pr}\{R_{t+1}=r,S_{t+1}=s^{\prime}\mid S_{t},A_{t}\},
$$ 
for all $r$ , $s^{\prime}$ , $S_{t}$ , and $A_{t}$ . In other words, a state signal has the Markov property, and is a Markov state, if and only if (3.5) is equal to (3.4) for all $s^{\prime}$ , $r$ , and histories, $S_{0}$ , $A_{0}$ , $R_{1}$ , ..., $S_{t-1}$ , $A_{t-1}$ , $R_{t}$ , $S_{t}$ , $A_{t}$ . In this case, the environment and task as a whole are also said to have the Markov property. 
If an environment has the Markov property, then its one-step dynamics (3.5) enable us to predict the next state and expected next reward given the current state and action. One can show that, by iterating this equation, one can predict all future states and expected rewards from knowledge only of the current state as well as would be possible given the complete history up to the current time. It also follows that Markov states provide the best possible basis for choosing actions. That is, the best policy for choosing actions as a function of a Markov state is just as good as the best policy for choosing actions as a function of complete histories. 
Even when the state signal is non-Markov, it is still appropriate to think of the state in reinforcement learning as an approximation to a Markov state. 
In particular, we always want the state to be a good basis for predicting future rewards and for selecting actions. In cases in which a model of the environment is learned (see Chapter 8), we also want the state to be a good basis for predicting subsequent states. Markov states provide an unsurpassed basis for doing all of these things. To the extent that the state approaches the ability of Markov states in these ways, one will obtain better performance from reinforcement learning systems. For all of these reasons, it is useful to think of the state at each time step as an approximation to a Markov state, although one should remember that it may not fully satisfy the Markov property. 
The Markov property is important in reinforcement learning because decisions and values are assumed to be a function only of the current state. In order for these to be effective and informative, the state representation must be informative. All of the theory presented in this book assumes Markov state signals. This means that not all the theory strictly applies to cases in which the Markov property does not strictly apply. However, the theory developed for the Markov case still helps us to understand the behavior of the algorithms, and the algorithms can be successfully applied to many tasks with states that are not strictly Markov. A full understanding of the theory of the Markov case is an essential foundation for extending it to the more complex and realistic non-Markov case. Finally, we note that the assumption of Markov state representations is not unique to reinforcement learning but is also present in most if not all other approaches to artificial intelligence. 
Example 3.5: Pole-Balancing State In the pole-balancing task introduced earlier, a state signal would be Markov if it specified exactly, or made it possible to reconstruct exactly, the position and velocity of the cart along the track, the angle between the cart and the pole, and the rate at which this angle is changing (the angular velocity). In an idealized cart–pole system, this information would be sufficient to exactly predict the future behavior of the cart and pole, given the actions taken by the controller. In practice, however, it is never possible to know this information exactly because any real sensor would introduce some distortion and delay in its measurements. Furthermore, in any real cart–pole system there are always other effects, such as the bending of the pole, the temperatures of the wheel and pole bearings, and various forms of backlash, that slightly affect the behavior of the system. These factors would cause violations of the Markov property if the state signal were only the positions and velocities of the cart and the pole. 
However, often the positions and velocities serve quite well as states. Some early studies of learning to solve the pole-balancing task used a coarse state signal that divided cart positions into three regions: right, left, and middle (and similar rough quantizations of the other three intrinsic state variables). This distinctly non-Markov state was sufficient to allow the task to be solved easily by reinforcement learning methods. In fact, this coarse representation may have facilitated rapid learning by forcing the learning agent to ignore fine distinctions that would not have been useful in solving the task. ■ 
Example 3.6: Draw Poker In draw poker, each player is dealt a hand of five cards. There is a round of betting, in which each player exchanges some of his cards for new ones, and then there is a final round of betting. At each round, each player must match or exceed the highest bets of the other players, or else drop out (fold). After the second round of betting, the player with the best hand who has not folded is the winner and collects all the bets. 
The state signal in draw poker is different for each player. Each player knows the cards in his own hand, but can only guess at those in the other players’ hands. A common mistake is to think that a Markov state signal should include the contents of all the players’ hands and the cards remaining in the deck. In a fair game, however, we assume that the players are in principle unable to determine these things from their past observations. If a player did know them, then she could predict some future events (such as the cards one could exchange for) better than by remembering all past observations. 
In addition to knowledge of one’s own cards, the state in draw poker should include the bets and the numbers of cards drawn by the other players. For example, if one of the other players drew three new cards, you may suspect he retained a pair and adjust your guess of the strength of his hand accordingly. The players’ bets also influence your assessment of their hands. In fact, much of your past history with these particular players is part of the Markov state. Does Ellen like to bluff, or does she play conservatively? Does her face or demeanor provide clues to the strength of her hand? How does Joe’s play change when it is late at night, or when he has already won a lot of money? 

Although everything ever observed about the other players may have an effect on the probabilities that they are holding various kinds of hands, in practice this is far too much to remember and analyze, and most of it will have no clear effect on one’s predictions and decisions. Very good poker players are adept at remembering just the key clues, and at sizing up new players quickly, but no one remembers everything that is relevant. As a result, the state representations people use to make their poker decisions are undoubtedly nonMarkov, and the decisions themselves are presumably imperfect. Nevertheless, people still make very good decisions in such tasks. We conclude that the inability to have access to a perfect Markov state representation is probably not a severe problem for a reinforcement learning agent. ■ 

## 3.6 Markov Decision Processes 
A reinforcement learning task that satisfies the Markov property is called a Markov decision process, or MDP. If the state and action spaces are finite, then it is called a finite Markov decision process (finite MDP). Finite MDPs are particularly important to the theory of reinforcement learning. We treat them extensively throughout this book; they are all you need to understand 90% of modern reinforcement learning. 
>  满足 Markov 性质的 RL 任务称为 MDP，如果状态和动作空间有限，就称为有限 MDP

A particular finite MDP is defined by its state and action sets and by the one-step dynamics of the environment. Given any state and action $s$ and $a$ , the probability of each possible pair of next state and reward, $s^{\prime},r$ , is denoted 

$$
p(s^{\prime},r|s,a)=\operatorname*{Pr}\{S_{t+1}=s^{\prime},R_{t+1}=r\mid S_{t}=s,A_{t}=a\}.\tag{3.6}
$$ 
These quantities completely specify the dynamics of a finite MDP. Most of the theory we present in the rest of this book implicitly assumes the environment is a finite MDP. 

>  一个有限 Markov 决策过程的定义包括其状态集、动作集和环境的一步动态
>  环境的一步动态指给定状态和动作 $s, a$，下一个状态和回报 $s', r$ 二者的联合条件分布，如 3.6 所示

Given the dynamics as specified by (3.6), one can compute anything else one might want to know about the environment, such as the expected rewards for state–action pairs, 
>  给定 3.6 的环境动态，可以计算任意关于环境的信息

$$
r(s,a)=\mathbb{E}[R_{t+1}\mid S_{t}=s,A_{t}=a]=\sum_{r\in\mathcal{R}}r\sum_{s^{\prime}\in\mathcal{S}}p(s^{\prime},r|s,a),\tag{3.7}
$$

>  例如给定状态-动作对，期望的回报如上 (先计算回报的边际分布，然后计算回报的期望)

the state-transition probabilities, 

$$
p(s^{\prime}|s,a)=\operatorname*{Pr}\{S_{t+1}=s^{\prime}\mid S_{t}=s,A_{t}=a\}=\sum_{r\in\mathcal{R}}p(s^{\prime},r|s,a),\tag{3.8}
$$

>  状态转移概率如上 (下一个状态 $S'$ 的边际分布)

and the expected rewards for state–action–next-state triples, 

$$
r(s,a,s^{\prime})=\mathbb{E}[R_{t+1}\mid S_{t}=s,A_{t}=a,S_{t+1}=s^{\prime}]=\frac{\sum_{r\in\mathcal{R}}r p(s^{\prime},r|s,a)}{p(s^{\prime}|s,a)}.\tag{3.9}
$$ 
>  状态-动作-下一个状态三元组的期望回报如上 (先计算回报的边际条件分布，然后计算期望)

In the first edition of this book, the dynamics were expressed exclusively in terms of the latter two quantities, which were denote $\mathcal{P}_{s s^{\prime}}^{a}$ and $\mathcal{R}_{s s^{\prime}}^{a}$ respectively. One weakness of that notation is that it still did not fully characterize the dynamics of the rewards, giving only their expectations. Another weakness is the excess of subscripts and superscripts. In this edition we will predominantly use the explicit notation of (3.6), while sometimes referring directly to the transition probabilities (3.8). 

Example 3.7: Recycling Robot MDP The recycling robot (Example 3.3) can be turned into a simple example of an MDP by simplifying it and providing some more details. (Our aim is to produce a simple example, not a particularly realistic one.) Recall that the agent makes a decision at times determined by external events (or by other parts of the robot’s control system). At each such time the robot decides whether it should (1) actively search for a can, (2) remain stationary and wait for someone to bring it a can, or (3) go back to home base to recharge its battery. Suppose the environment works as follows. The best way to find cans is to actively search for them, but this runs down the robot’s battery, whereas waiting does not. Whenever the robot is searching, the possibility exists that its battery will become depleted. In this case the robot must shut down and wait to be rescued (producing a low reward). 

The agent makes its decisions solely as a function of the energy level of the battery. It can distinguish two levels, high and low, so that the state set is $\mathcal{S}=\{\mathtt{h i g h},\mathtt{l o w}\}$ . Let us call the possible decisions—the agent’s actions— wait, search, and recharge. When the energy level is high, recharging would always be foolish, so we do not include it in the action set for this state. The agent’s action sets are 

$$
\begin{array}{r c l}{{\mathcal{A}(\mathrm{high})}}&{{=}}&{{\{\mathrm{search},\mathrm{wait}\}}}\ {{\mathcal{A}(\mathrm{1ow})}}&{{=}}&{{\{\mathrm{search},\mathrm{wait},\mathrm{recharge}\}.}}\end{array}
$$ 
If the energy level is high, then a period of active search can always be completed without risk of depleting the battery. A period of searching that begins with a high energy level leaves the energy level high with probability $\alpha$ and reduces it to low with probability $1-\alpha$ . On the other hand, a period of searching undertaken when the energy level is low leaves it low with probability $\beta$ and depletes the battery with probability $1-\beta$ . In the latter case, the robot must be rescued, and the battery is then recharged back to high. Each can collected by the robot counts as a unit reward, whereas a reward of $^{-3}$ results whenever the robot has to be rescued. Let $r_{\tt s e a r c h}$ and $r_{\tt w a i t}$ , with $r_{\tt s e a r c h}>r_{\tt w a i t}$ , respectively denote the expected number of cans the robot will collect (and hence the expected reward) while searching and while waiting. Finally, to keep things simple, suppose that no cans can be collected during a run home for recharging, and that no cans can be collected on a step in which the battery is depleted. This system is then a finite MDP, and we can write down the transition probabilities and the expected rewards, as in Table 3.1. 

A transition graph is a useful way to summarize the dynamics of a finite MDP. Figure 3.3 shows the transition graph for the recycling robot example. There are two kinds of nodes: state nodes and action nodes. There is a state node for each possible state (a large open circle labeled by the name of the state), and an action node for each state–action pair (a small solid circle labeled 
<html><body><table><tr><td>S</td><td>s'</td><td>a</td><td>p(s'ls,a)</td><td>r(s,a, s)</td></tr><tr><td>high</td><td>high</td><td>search</td><td></td><td>rsearch</td></tr><tr><td>high</td><td>low</td><td>search</td><td>1-Q</td><td>rsearch</td></tr><tr><td>low</td><td>high</td><td>search</td><td>1-β</td><td>-3</td></tr><tr><td>low</td><td>low</td><td>search</td><td>β</td><td>Tsearch</td></tr><tr><td>high</td><td>high</td><td>wait</td><td>1</td><td>rwait</td></tr><tr><td>high</td><td>low</td><td>wait</td><td>0</td><td>rwait</td></tr><tr><td>low</td><td>high</td><td>wait</td><td>0</td><td>rwait</td></tr><tr><td>low</td><td>low</td><td>wait</td><td>1</td><td>rwait</td></tr><tr><td>low</td><td>high</td><td>recharge</td><td>1</td><td>0</td></tr><tr><td>low</td><td>low</td><td>recharge</td><td>0</td><td>0.</td></tr></table></body></html> 
Table 3.1: Transition probabilities and expected rewards for the finite MDP of the recycling robot example. There is a row for each possible combination of current state, $s$ , next state, $s^{\prime}$ , and action possible in the current state, $a\in\mathcal{A}(s)$ . 
by the name of the action and connected by a line to the state node). Starting in state $s$ and taking action $a$ moves you along the line from state node $s$ to action node $(s,a)$ . Then the environment responds with a transition to the next state’s node via one of the arrows leaving action node $(s,a)$ . Each arrow corresponds to a triple $(s,s^{\prime},a)$ ，where $s^{\prime}$ is the next state, and we label the arrow with the transition probability, $p(s^{\prime}|s,a)$ , and the expected reward for that transition, $r(s,a,s^{\prime})$ . Note that the transition probabilities labeling the arrows leaving an action node always sum to 1. ■ 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/73fa2d3755140c75137d981971bbb5b951b99dbe7501d02401822ab555b1b60b.jpg) 
Figure 3.3: Transition graph for the recycling robot example 

## 3.7 Value Functions 
Almost all reinforcement learning algorithms involve estimating value functions—functions of states (or of state–action pairs) that estimate how good it is for the agent to be in a given state (or how good it is to perform a given action in a given state). The notion of “how good” here is defined in terms of future rewards that can be expected, or, to be precise, in terms of expected return. Of course the rewards the agent can expect to receive in the future depend on what actions it will take. Accordingly, value functions are defined with respect to particular policies. 
>  几乎所有 RL 算法都涉及估计价值函数，价值函数是关于状态/状态-动作对的函数，评估了智能体在给定状态/在给定状态并执行给定动作的“优异”程度
>  “优异”程度由未来的期望奖励定义，即期望回报
>  智能体在未来期望收到的奖励取决于它执行的动作，故价值函数是相对于特定策略而定义

Recall that a policy, $\pi$ , is a mapping from each state, $s\in\mathcal{S}$ , and action, $a\in$ ${\mathcal{A}}(s)$ , to the probability $\pi({a}|{s})$ of taking action $a$ when in state $s$ . Informally, the value of a state $s$ under a policy $\pi$ , denoted $v_{\pi}(s)$ , is the expected return when starting in $s$ and following $\pi$ thereafter. 
>  策略 $\pi$ 是从状态 $s\in \mathcal S$ 和动作 $a\in \mathcal A(s)$ 到概率 $\pi(a\mid s)$ 的映射，表示了在状态 $s$ 下执行动作 $a$ 的概率
>  状态 $s$ 在策略 $\pi$ 下的价值记作 $v_\pi(s)$，定义为从状态 $s$ 开始，后续执行策略 $\pi$ 所能收到的期望回报

For MDPs, we can define $v_{\pi}(s)$ formally as 

$$
v_\pi(s) = \mathbb E_\pi\left[G_t \mid S_t= s\right] = \mathbb E_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t + k +1}\mid S_t =s \right]\tag{3.10}
$$

where $\mathbb{E}_{\pi}[\cdot]$ denotes the expected value of a random variable given that the agent follows policy $\pi$ , and $t$ is any time step. Note that the value of the terminal state, if any, is always zero. We call the function $v_{\pi}$ the state-value function for policy $\pi$ . 

>  对于 MDP，策略 $\pi$ 的状态价值函数 $v_\pi$ 的形式定义如上
>  终止状态的状态价值函数为零

Similarly, we define the value of taking action $a$ in state $s$ under a policy $\pi$ , denoted $q_{\pi}(s,a)$ , as the expected return starting from $s$ , taking the action $a$ , and thereafter following policy $\pi$ : 

$$
q_\pi(s, a) = \mathbb E_\pi\left[G_t\mid S_t =s, A_t = a\right] = \mathbb E_\pi\left[\sum_{k=0}^\infty R_{t+k+1}\mid S_t = s, A_t = a\right]\tag{3.11}
$$

We call $q_{\pi}$ the action-value function for policy $\pi$ . 

>  在策略 $\pi$ ，在状态 $s$ 执行动作 $a$ 的价值记作 $q_\pi(s, a)$，定义为从状态 $s$ 开始，执行动作 $a$，然后遵循策略 $\pi$ 所能得到的期望回报
>  $q_\pi(s, a)$ 称为策略 $\pi$ 的动作价值函数

The value functions $v_{\pi}$ and $q_{\pi}$ can be estimated from experience. For example, if an agent follows policy $\pi$ and maintains an average, for each state encountered, of the actual returns that have followed that state, then the average will converge to the state’s value, $v_{\pi}(s)$ , as the number of times that state is encountered approaches infinity. If separate averages are kept for each action taken in a state, then these averages will similarly converge to the action values, $q_{\pi}(s,a)$ . 
>  价值函数 $v_\pi, q_\pi$ 都可以从实践中估计
>  例如智能体遵循策略 $\pi$，在实践中记录每次遇到状态 $s$ 得到的实际回报，计算平均值，随着次数增加，该均值会收敛到该状态的真实价值 $v_\pi(s)$
>  同样，智能体在实践中记录每个状态下执行每个动作得到的实际回报的均值，均值最后会收敛到真实值 $q_\pi(s, a)$

We call estimation methods of this kind Monte Carlo methods because they involve averaging over many random samples of actual returns. These kinds of methods are presented in Chapter 5. Of course, if there are very many states, then it may not be practical to keep separate averages for each state individually. Instead, the agent would have to maintain $v_{\pi}$ and $q_{\pi}$ as parameterized functions and adjust the parameters to better match the observed returns. This can also produce accurate estimates, although much depends on the nature of the parameterized function approximator (Chapter 9). 
>  这类估计方法就是 Monte Carlo 方法，它使用均值估计期望
>  为每个状态和每个状态-动作对分别维护均值过于昂贵，智能体可以用参数化的函数表示 $v_\pi, q_\pi$，通过调节参数以匹配观察，这虽然依赖于参数化函数估计器的性质，但也可以产生正确的估计

A fundamental property of value functions used throughout reinforcement learning and dynamic programming is that they satisfy particular recursive relationships. For any policy $\pi$ and any state $s$ , the following consistency condition holds between the value of $s$ and the value of its possible successor states: 
>  价值函数的一个基本性质是满足特定的递归关系
>  对于任意的策略 $\pi$ 和任意的状态 $s$， $s$ 和它可能的后续状态满足以下一致条件

$$
\begin{align}
v_\pi(s)&=\mathbb E_\pi\left[G_t \mid S_t = s\right]\\
&=\mathbb E_\pi\left[\sum_{k=0}^\infty\gamma^{k}R_{t+k+1}\mid S_t = s\right]\\
&=\mathbb E_\pi \left[R_{t+1} + \gamma\cdot\sum_{k=0}^\infty\gamma^k R_{t+k+2}\mid S_t = s\right]\\
&=\sum_{a}\pi(a\mid s)\sum_{s'}\sum_r p(s', r\mid s, a)\left[r + \gamma\mathbb E_{\pi}\left[\sum_{k=0}^\infty \gamma^k R_{t+k+2}\mid S_{t+1} = s' \right]\right]\\
&=\sum_a\pi(a\mid s)\sum_{s', r}p(s',r\mid s, a)[r + \gamma v_\pi(s')]\qquad (3.12)
\end{align}
$$

>  其中第四个等号将对于 $\pi$ 的期望按序拆分，首先根据当前状态选择一个可能动作，然后根据动作-状态对和环境动态选择可能的后续状态-奖励对，此时 $R_{t+1}$ 已经确定为 $r$，后续则对 $R_{t+2}$ 之后的奖励，基于状态 $s'$ 按照 $\pi$ 取期望
>  因此状态价值和后续的状态价值构成了以上递归关系

where it is implicit that the actions, $a$ , are taken from the set ${\mathcal{A}}(s)$ , the next states, $s^{\prime}$ , are taken from the set $\mathcal S$ (or from $\mathcal{S}^{+}$ in the case of an episodic problem), and the rewards, $r$ , are taken from the set $\mathcal{R}$ . 

Note also how in the last equation we have merged the two sums, one over all the values of $s^{\prime}$ and the other over all values of $r$ , into one sum over all possible values of both. We will use this kind of merged sum often to simplify formulas. 

Note how the final expression can be read very easily as an expected value. It is really a sum over all values of the three variables, $a$ , $s^{\prime}$ , and $r$ . For each triple, we compute its probability, $\pi(a|s)p(s^{\prime},r|s,a)$ , weight the quantity in brackets by that probability, then sum over all possibilities to get an expected value. 
>  最后的表达式可以容易地解读为一个期望值，它实际上是对三个变量 $a, s', r$ 的所有取值的求和，对于每个三元组，我们计算其概率 $\pi(a\mid s) p(s', r\mid s, a)$，使用概率对 $r + \gamma v_\pi(s')$ 加权，然后求加权平均和得到期望

Equation (3.12) is the Bellman equation for $v_{\pi}$ . It expresses a relationship between the value of a state and the values of its successor states. Think of looking ahead from one state to its possible successor states, as suggested by Figure 3.4a. Each open circle represents a state and each solid circle represents a state–action pair. Starting from state $s$ , the root node at the top, the agent could take any of some set of actions—three are shown in Figure 3.4a. From each of these, the environment could respond with one of several next states, $s^{\prime}$ , along with a reward, $r$ . The Bellman equation (3.12) averages over all the possibilities, weighting each by its probability of occurring. It states that the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way. 
>  Eq 3.12 是状态价值函数 $v_\pi$ 的 Bellman 方程，该方程表示了当前状态的价值和其后继状态的价值之间的关系
>  Bellman 方程遍历所有的可能性 (给定 $s$，可能发生的动作 $a$，以及后续可能的状态和奖励) 用概率作为权重求加权平均和。该方程说明了起始状态的价值必须等于下一个状态的期望 (折扣) 价值加上对应的期望奖励

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/651d493c3b2c94886e64b88c82b45730a0f68d34b2d6a793a212e548141a086e.jpg) 

Figure 3.4: Backup diagrams for (a) $v_{\pi}$ and (b) $q_{\pi}$ . 

The value function $v_{\pi}$ is the unique solution to its Bellman equation. We show in subsequent chapters how this Bellman equation forms the basis of a number of ways to compute, approximate, and learn $v_{\pi}$ . 
>  价值函数 $v_\pi$ 是其 Bellman 方程的唯一解

We call diagrams like those shown in Figure 3.4 backup diagrams because they diagram relationships that form the basis of the update or backup operations that are at the heart of reinforcement learning methods. These operations transfer value information back to a state (or a state–action pair) from its successor states (or state–action pairs). We use backup diagrams throughout the book to provide graphical summaries of the algorithms we discuss. (Note that unlike transition graphs, the state nodes of backup diagrams do not necessarily represent distinct states; for example, a state might be its own successor. We also omit explicit arrowheads because time always flows downward in a backup diagram.) 
>  如 Figure 3.4 的图称为回溯更新图，这类图描绘了形成 RL 学习方法基础的更新或回溯更新操作，这些操作将价值信息从后继状态/状态-动作对传递回先导状态/状态-动作对
>  注意与转移图不同，回溯更新图中的节点不一定表示不同的状态，因为一个状态的后继状态也可能是它自己

Example 3.8: Gridworld Figure 3.5a uses a rectangular grid to illustrate value functions for a simple finite MDP. The cells of the grid correspond to the states of the environment. At each cell, four actions are possible: north, south, east, and west, which deterministically cause the agent to move one cell in the respective direction on the grid. Actions that would take the agent off the grid leave its location unchanged, but also result in a reward of $-1$ . Other actions result in a reward of 0, except those that move the agent out of the special states A and B. From state A, all four actions yield a reward of $+10$ and take the agent to $\mathrm{A}^{\prime}$ . From state B, all actions yield a reward of $+5$ and take the agent to $\mathrm{B^{\prime}}$ . 

Suppose the agent selects all four actions with equal probability in all states. Figure 3.5b shows the value function, $v_{\pi}$ , for this policy, for the discounted reward case with $\gamma=0.9$ . This value function was computed by solving the system of equations (3.12). Notice the negative values near the lower edge; these are the result of the high probability of hitting the edge of the grid there under the random policy. State A is the best state to be in under this policy, but its expected return is less than 10, its immediate reward, because from A the agent is taken to $\mathrm{A}^{\prime}$ , from which it is likely to run into the edge of the grid. State B, on the other hand, is valued more than 5, its immediate reward, because from B the agent is taken to $\mathrm{B^{\prime}}$ , which has a positive value. From $\mathrm{B^{\prime}}$ the expected penalty (negative reward) for possibly running into an edge is more than compensated for by the expected gain for possibly stumbling onto A or B. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/d2a736ee309cc3ea7eea418bf42f988a546c36028d2a401689187c0f4e80637b.jpg) 

Figure 3.5: Grid example: (a) exceptional reward dynamics; (b) state-value function for the equiprobable random policy. 

Example 3.9: Golf To formulate playing a hole of golf as a reinforcement learning task, we count a penalty (negative reward) of $-1$ for each stroke until we hit the ball into the hole. The state is the location of the ball. The value of a state is the negative of the number of strokes to the hole from that location. Our actions are how we aim and swing at the ball, of course, and which club we select. Let us take the former as given and consider just the choice of club, which we assume is either a putter or a driver. The upper part of Figure 3.6 shows a possible state-value function, $v_{\mathrm{putt}}(s)$ , for the policy that always uses the putter. The terminal state in-the-hole has a value of 0. From anywhere on the green we assume we can make a putt; these states have value $-1$ . Off the green we cannot reach the hole by putting, and the value is greater. If we can reach the green from a state by putting, then that state must have value one less than the green’s value, that is, $-2$ . For simplicity, let us assume we can putt very precisely and deterministically, but with a limited range. This gives us the sharp contour line labeled $-2$ in the figure; all locations between that line and the green require exactly two strokes to complete the hole. Similarly, any location within putting range of the $-2$ contour line must have a value of $^{-3}$ , and so on to get all the contour lines shown in the figure. Putting doesn’t get us out of sand traps, so they have a value of $-\infty$ . Overall, it takes us six strokes to get from the tee to the hole by putting. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/2f79d9d17a82f91bb1a7e4a2d59b816fbb775b7dbbdf3fc3dca4e9253cc88ed2.jpg) 

Figure 3.6: A golf example: the state-value function for putting (above) and the optimal action-value function for using the driver (below). 

## 3.8 Optimal Value Functions 
Solving a reinforcement learning task means, roughly, finding a policy that achieves a lot of reward over the long run. For finite MDPs, we can precisely define an optimal policy in the following way. Value functions define a partial ordering over policies. A policy $\pi$ is defined to be better than or equal to a policy $\pi^{\prime}$ if its expected return is greater than or equal to that of $\pi^{\prime}$ for all states. In other words, $\pi\geq\pi^{\prime}$ if and only if $v_{\pi}(s)\geq v_{\pi^{\prime}}(s)$ for all $s\in\mathcal{S}$ . There is always at least one policy that is better than or equal to all other policies. This is an optimal policy. Although there may be more than one, we denote all the optimal policies by $\pi_{*}$ . They share the same state-value function, called the optimal state-value function, denoted $v_{*}$ , and defined as 

$$
v_{*}(s)=\operatorname*{max}_{\pi}v_{\pi}(s),\tag{3.13}
$$ 
for all $s\in\mathcal{S}$ . 

>  求解一个 RL 问题大致意味着找到一个可以在长期执行下获得大量奖励的策略
>  对于有限 MDP，可以确切地定义一个最优策略：价值函数定义了策略之间的一个偏序关系，对于所有的状态，如果一个策略 $\pi$ 的价值不小于另一个策略 $\pi'$ 的价值，就称策略 $\pi$ 相对于 $\pi'$ 是更优的或者至少是等价的，即 $\pi \ge \pi'$ 当且仅当 $v_\pi(s)\ge v_{\pi'}(s)\quad \forall s \in \mathcal S$
>  有限 MDP 中，总是存在至少一个策略，它大于等于所有其他策略，该策略即最优策略，最优策略也不一定仅有一个，我们记所有最优策略为 $\pi_*$
>  所有最优策略在价值函数层面等价，或者说它们共享相同的状态价值函数，最优策略的状态价值函数记作 $v_*(s)$，定义如上

Optimal policies also share the same optimal action-value function, denoted $q_{*}$ , and defined as 

$$
q_{*}(s,a)=\operatorname*{max}_{\pi}q_{\pi}(s,a),\tag{3.14}
$$ 
for all $s\in\mathcal{S}$ and $a\in\mathcal{A}(s)$ . For the state–action pair $(s,a)$ , this function gives the expected return for taking action $a$ in state $s$ and thereafter following an optimal policy. 

>  最优策略同样共享相同的动作价值函数，记作 $q_*$，定义如上
>  该价值函数给出了在状态 $s$ 执行动作 $a$，之后遵循一个最优策略所能得到的期望回报

Thus, we can write $q_{*}$ in terms of $v_{*}$ as follows: 

$$
\begin{align}
q_{*}(s,a)=\mathbb{E}[R_{t+1}+\gamma v_{*}(S_{t+1})\mid S_{t}=s,A_{t}=a].\tag{3.15}
\end{align}
$$ 
>  因此，最优动作价值函数和下一个状态的最优状态价值函数存在如上的关系 (这里的求期望实际上就是在对后继状态求期望)

Example 3.10: Optimal Value Functions for Golf The lower part of Figure 3.6 shows the contours of a possible optimal action-value function $q_{*}(s,\mathtt{d r i v e r})$ . These are the values of each state if we first play a stroke with the driver and afterward select either the driver or the putter, whichever is better. The driver enables us to hit the ball farther, but with less accuracy. We can reach the hole in one shot using the driver only if we are already very close; thus the $-1$ contour for $q_{*}(s,\mathtt{d r i v e r})$ covers only a small portion of the green. If we have two strokes, however, then we can reach the hole from much farther away, as shown by the $-2$ contour. In this case we don’t have to drive all the way to within the small $-1$ contour, but only to anywhere on the green; from there we can use the putter. The optimal action-value function gives the values after committing to a particular first action, in this case, to the driver, but afterward using whichever actions are best. The $^{-3}$ contour is still farther out and includes the starting tee. From the tee, the best sequence of actions is two drives and one putt, sinking the ball in three strokes. 

Because $v_{*}$ is the value function for a policy, it must satisfy the self-consistency condition given by the Bellman equation for state values (3.12). Because it is the optimal value function, however, $v_{*}$ ’s consistency condition can be written in a special form without reference to any specific policy. This is the Bellman equation for $v_{*}$ , or the Bellman optimality equation. 
>  最优状态价值函数 $v_*$ 也是某个最优策略的价值函数，故显然也满足 Bellman 方程给出的自一致条件
>  $v_*$ 的一致性条件可以进一步写为不参照任意特定策略的特殊形式，即 $v_*$ 的 Bellman 方程，我们称为 Bellman 最优方程

Intuitively, the Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state: 

$$
\begin{align}
v_*(s) &= \max_{a\in \mathcal A(s)}q_{\pi_*}(s, a)\\
&=\max_a \mathbb E_{\pi^*}[G_t\mid S_t = s, A_t = a]\\
&=\max_a \mathbb E_{\pi^*}\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1}\mid S_t = s, A_t = a\right]\\
&=\max_a \mathbb E_{\pi^*}\left[R_{t+1} + \gamma\cdot\sum_{k=0}^\infty \gamma^{k} R_{t+k+2}\mid S_t = s, A_t = a\right]\\
&=\max_a \mathbb E\left[R_{t+1} + \gamma v_*(S_{t+1})\mid S_t = s, A_t = a\right]\tag{3.16}\\
&=\max_a \sum_{s',r}p(s',r'\mid s, a)[r + \gamma v_*(s')]\tag{3.17}
\end{align}
$$

The last two equations are two forms of the Bellman optimality equation for $v_{*}$ . 

>  直观上，Bellman 最优方程表明了最优策略下的状态价值一定等于从该状态采取最佳行动的期望回报

The Bellman optimality equation for $q_{*}$ is 

$$
\begin{align}
q_*(s, a) 
&=\mathbb E_{\pi^*}\left[G_t\mid S_t = s, A_t = a\right]\\
&=\mathbb E_{\pi^*}\left[\sum_{k=0}^\infty \gamma^kR_{t+k+1}\mid S_t = s, A_t = a\right]\\
&=\mathbb E_{\pi^*}\left[R_{t+1} + \gamma\cdot \sum_{k=0}^\infty\gamma^k R_{t+k+2}\mid S_t = s, A_t = a\right]\\
&=\mathbb E_{}\left[R_{t+1} + \gamma v_*(S_{t+1})\mid S_t = s, A_t = a\right]\\
&= \mathbb E\left[R_{t+1} + \gamma\max_{a'} q_*(S_{t+1}, a')\mid S_t = s, A_t = a\right]\\
&=\sum_{s',r}p(s',r\mid s, a)[r + \gamma \max_{a'}q_*(s', a')]
\end{align}
$$

>  最优动作价值函数 $q_*$ 的 Bellman 最优方程如上

The backup diagrams in Figure 3.7 show graphically the spans of future states and actions considered in the Bellman optimality equations for $v_{*}$ and $q_{*}$ . These are the same as the backup diagrams for $v_{\pi}$ and $q_{\pi}$ except that arcs have been added at the agent’s choice points to represent that the maximum over that choice is taken rather than the expected value given some policy. Figure 3.7a graphically represents the Bellman optimality equation (3.17). 
>  Figure 3.7 展示了 Bellman 最优方程所考虑的未来状态和动作，图形和之前是相同的，差异在于为智能体的选择点添加了弧线，以表示智能体仅选择达到最大值的动作，而不是计算期望值

For finite MDPs, the Bellman optimality equation (3.17) has a unique solution independent of the policy. The Bellman optimality equation is actually a system of equations, one for each state, so if there are $N$ states, then there are $N$ equations in $N$ unknowns. If the dynamics of the environment are known $(p(s^{\prime},r|s,a))$ , then in principle one can solve this system of equations for $v_{*}$ using any one of a variety of methods for solving systems of nonlinear equations. One can solve a related set of equations for $q_{*}$ . 
>  对于有限 Markov 决策过程，Bellman 最优方程有独立于策略的唯一解
>  Bellman 最优方程本质是一个方程组，每个状态贡献一个方程，如果有 $N$ 个状态，就有 $N$ 个未知数的 $N$ 个方程，如果环境动态 $p(s', r\mid s, a)$ 已知，则原则上可以求解该方程组以得到 $v_*$， $q_*$ 同理

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/b3b0bf249c33bdbaa61d946b8261f51a2cc318c1da400a346e20c36791969953.jpg) 


Figure 3.7: Backup diagrams for (a) $v_{*}$ and (b) $q_{*}$ 

Once one has $v_{*}$ , it is relatively easy to determine an optimal policy. For each state $s$ , there will be one or more actions at which the maximum is obtained in the Bellman optimality equation. Any policy that assigns nonzero probability only to these actions is an optimal policy. 
>  解出最优状态价值函数 $v_*$ 之后，就容易决定最优策略：对于每个状态 $s$，会存在一组动作使得 Bellman 最优方程 $\max_a \mathbb E\left[R_{t+1} + \gamma v_*(S_{t+1})\mid S_t = s, A_t = a\right]$ 中的期望取到最大，任意对这组动作赋予非零概率值，对其余动作赋予零概率值的策略就都是最优策略

You can think of this as a one-step search. If you have the optimal value function, $v_{*}$ , then the actions that appear best after a one-step search will be optimal actions. Another way of saying this is that any policy that is greedy with respect to the optimal evaluation function $v_{*}$ is an optimal policy. 
>  这个过程可以视作一步搜索，已知最优状态价值函数 $v_*$ 之后，在一步搜索之后最优的动作就是最优策略会执行的动作
>  换句话说，任意相对于最优状态价值函数 $v_*$ 是贪心的策略都是最优策略

The term greedy is used in computer science to describe any search or decision procedure that selects alternatives based only on local or immediate considerations, without considering the possibility that such a selection may prevent future access to even better alternatives. Consequently, it describes policies that select actions based only on their short-term consequences. The beauty of $v_{*}$ is that if one uses it to evaluate the short-term consequences of actions—specifically, the one-step consequences—then a greedy policy is actually optimal in the long-term sense in which we are interested because $v_{*}$ already takes into account the reward consequences of all possible future behavior. By means of $v_{*}$ , the optimal expected long-term return is turned into a quantity that is locally and immediately available for each state. Hence, a one-step-ahead search yields the long-term optimal actions. 
>  最优状态状态价值函数 $v_*$ 已经考虑了所有未来可能行为的奖励后果，因此基于 $v_*$ 进行贪心决策 (基于一步搜索的结果进行决策) 就能得到长期上最优的行动。$v_*$ 将最优的长期回报转化为可以局部地基于每个状态立即获得的量

Having $q_{*}$ makes choosing optimal actions still easier. With $q_{*}$ , the agent does not even have to do a one-step-ahead search: for any state $s$ , it can simply find any action that maximizes $q_{*}(s,a)$ . The action-value function effectively caches the results of all one-step-ahead searches. It provides the optimal expected long-term return as a value that is locally and immediately available for each state–action pair. Hence, at the cost of representing a function of state–action pairs, instead of just of states, the optimal action-value function allows optimal actions to be selected without having to know anything about possible successor states and their values, that is, without having to know anything about the environment’s dynamics. 
>  解出 $q_*$ 之后，仍然可以容易选择最优行动，对于任意状态 $s$ ，智能体只需要找到能够最大化 $q_*(s, a)$ 的动作
>  动作价值函数有效存储了所有一步搜索 (即执行一次动作) 的结果，提供了局部地基于每个状态-动作对就可以知道的最优期望长期回报值
>  因此基于最优动作价值函数可以在不需要知道可能的后继状态和其价值的情况下进行判断，也就是不需要知道环境动态

Example 3.11: Bellman Optimality Equations for the Recycling Robot Using (3.17), we can explicitly give the Bellman optimality equation for the recycling robot example. To make things more compact, we abbreviate the states high and low, and the actions search, wait, and recharge respectively by h, l, s, w, and re. Since there are only two states, the Bellman optimality equation consists of two equations. The equation for $v_{*}(\mathtt{h})$ can be written as follows: 

$$
\begin{array}{r l}{\mathrm{~\psi_*(h)~}=}&{\operatorname*{max}\left\{\begin{array}{l l}{p(\mathrm{h}|\mathbf{h},\mathbf{s})[r(\mathbf{h},\mathbf{s},\mathbf{h})+\gamma v_{*}(\mathbf{h})]+p(\mathrm{\mathbb{1}}|\mathbf{h},\mathbf{s})[r(\mathbf{h},\mathbf{s},\mathbf{1})+\gamma v_{*}(\mathbf{1})],}\ {p(\mathrm{h}|\mathbf{h},\mathbf{v})[r(\mathbf{h},\mathbf{v},\mathbf{h})+\gamma v_{*}(\mathbf{h})]+p(\mathrm{\mathbb{1}}|\mathbf{h},\mathbf{v})[r(\mathbf{h},\mathbf{w},\mathbf{1})+\gamma v_{*}(\mathbf{1})]}\end{array}\right\}}\ {=}&{\operatorname*{max}\left\{\begin{array}{l l}{\alpha[r_{\mathbf{s}}+\gamma v_{*}(\mathbf{h})]+(1-\alpha)[r_{\mathbf{s}}+\gamma v_{*}(\mathbf{1})],}\ {1[r_{\mathbf{s}}+\gamma v_{*}(\mathbf{h})]+0[r_{\mathbf{s}}+\gamma v_{*}(\mathbf{1})]}\end{array}\right\}}\ {=}&{\operatorname*{max}\left\{\begin{array}{l l}{r_{\mathbf{s}}+\gamma[\alpha v_{*}(\mathbf{h})+(1-\alpha)v_{*}(\mathbf{1})],}\ {r_{\mathbf{u}}+\gamma v_{*}(\mathbf{h})}\end{array}\right\}.}\end{array}
$$ 
Following the same procedure for $v_{*}(1)$ yields the equation 

$$
v_{*}(1)=\operatorname*{max}\left\{\begin{array}{l}{\beta r_{\mathrm{s}}-3(1-\beta)+\gamma[(1-\beta)v_{*}(\mathrm{h})+\beta v_{*}(1)]}\ {r_{\mathrm{u}}+\gamma v_{*}(1),}\ {\gamma v_{*}(\mathrm{h})}\end{array}\right\}.
$$ 
For any choice of $r_{\mathrm{s}}$ , $r_{\mathrm{{w}}}$ , $\alpha$ , $\beta$ , and $\gamma$ , with $0\leq\gamma<1$ , $0\leq\alpha,\beta\leq1$ , there is exactly one pair of numbers, $v_{*}(\mathtt{h})$ and $v_{*}(1)$ , that simultaneously satisfy these two nonlinear equations. 

Example 3.12: Solving the Gridworld Suppose we solve the Bellman equation for $v_{*}$ for the simple grid task introduced in Example 3.8 and shown again in Figure 3.8a. Recall that state A is followed by a reward of $+10$ and transition to state $\mathrm{A}^{\prime}$ , while state B is followed by a reward of $+5$ and transition to state $\mathrm{B^{\prime}}$ . Figure 3.8b shows the optimal value function, and Figure 3.8c shows the corresponding optimal policies. Where there are multiple arrows in a cell, any of the corresponding actions is optimal. 

Explicitly solving the Bellman optimality equation provides one route to finding an optimal policy, and thus to solving the reinforcement learning problem. However, this solution is rarely directly useful. It is akin to an exhaustive search, looking ahead at all possibilities, computing their probabilities of occurrence and their desirabilities in terms of expected rewards. This solution relies on at least three assumptions that are rarely true in practice: (1) we accurately know the dynamics of the environment; (2) we have enough computational resources to complete the computation of the solution; and (3) the Markov property. For the kinds of tasks in which we are interested, one is generally not able to implement this solution exactly because various combinations of these assumptions are violated. For example, although the first and third assumptions present no problems for the game of backgammon, the second is a major impediment. Since the game has about $10^{20}$ states, it would take thousands of years on today’s fastest computers to solve the Bellman equation for $v_{*}$ , and the same is true for finding $q_{*}$ . In reinforcement learning one typically has to settle for approximate solutions. 
>  获取最优策略的一种方式就是显式求解 Bellman 最优方程，但 Bellman 最优方程的求解要求遍历所有的可能性，且依赖于实践中较少成立的三个假设
>  1. 确切知道环境动态
>  2. 具有足够计算资源求解
>  3. 满足 Markov 性质
>  状态集合的大小往往是指数级别，因此确切求解 Bellman 最优方程往往不现实

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/70982c51cdbfc121197ae7db3f3f084106132d14fae4c14289a72c075209a8ab.jpg) 

Figure 3.8: Optimal solutions to the gridworld example. 

Many different decision-making methods can be viewed as ways of approximately solving the Bellman optimality equation. For example, heuristic search methods can be viewed as expanding the right-hand side of (3.17) several times, up to some depth, forming a “tree” of possibilities, and then using a heuristic evaluation function to approximate $v_{*}$ at the “leaf” nodes. (Heuristic search methods such as A∗ are almost always based on the episodic case.) The methods of dynamic programming can be related even more closely to the Bellman optimality equation. Many reinforcement learning methods can be clearly understood as approximately solving the Bellman optimality equation, using actual experienced transitions in place of knowledge of the expected transitions. We consider a variety of such methods in the following chapters. 
>  许多决策过程可以视作近似求解 Bellman 最优方程，例如启发式搜索可以认为是多次展开 Eq 3.17 的 RHS，构成一个树，在叶子节点近似 $v_*$，又例如动态规划，以及许多 RL 方法

## 3.9 Optimality and Approximation 
We have defined optimal value functions and optimal policies. Clearly, an agent that learns an optimal policy has done very well, but in practice this rarely happens. For the kinds of tasks in which we are interested, optimal policies can be generated only with extreme computational cost. A well-defined notion of optimality organizes the approach to learning we describe in this book and provides a way to understand the theoretical properties of various learning algorithms, but it is an ideal that agents can only approximate to varying degrees. As we discussed above, even if we have a complete and accurate model of the environment’s dynamics, it is usually not possible to simply compute an optimal policy by solving the Bellman optimality equation. For example, board games such as chess are a tiny fraction of human experience, yet large, custom-designed computers still cannot compute the optimal moves. A critical aspect of the problem facing the agent is always the computational power available to it, in particular, the amount of computation it can perform in a single time step. 
The memory available is also an important constraint. A large amount of memory is often required to build up approximations of value functions, policies, and models. In tasks with small, finite state sets, it is possible to form these approximations using arrays or tables with one entry for each state (or state–action pair). This we call the tabular case, and the corresponding methods we call tabular methods. In many cases of practical interest, however, there are far more states than could possibly be entries in a table. In these cases the functions must be approximated, using some sort of more compact parameterized function representation. 
Our framing of the reinforcement learning problem forces us to settle for approximations. However, it also presents us with some unique opportunities for achieving useful approximations. For example, in approximating optimal behavior, there may be many states that the agent faces with such a low probability that selecting suboptimal actions for them has little impact on the amount of reward the agent receives. Tesauro’s backgammon player, for example, plays with exceptional skill even though it might make very bad decisions on board configurations that never occur in games against experts. In fact, it is possible that TD-Gammon makes bad decisions for a large fraction of the game’s state set. The on-line nature of reinforcement learning makes it possible to approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less effort for infrequently encountered states. This is one key property that distinguishes reinforcement learning from other approaches to approximately solving MDPs. 

## 3.10 Summary 
Let us summarize the elements of the reinforcement learning problem that we have presented in this chapter. Reinforcement learning is about learning from interaction how to behave in order to achieve a goal. The reinforcement learning agent and its environment interact over a sequence of discrete time steps. The specification of their interface defines a particular task: the actions are the choices made by the agent; the states are the basis for making the choices; and the rewards are the basis for evaluating the choices. Everything inside the agent is completely known and controllable by the agent; everything outside is incompletely controllable but may or may not be completely known. A policy is a stochastic rule by which the agent selects actions as a function of states. The agent’s objective is to maximize the amount of reward it receives over time. 
The return is the function of future rewards that the agent seeks to maximize. It has several different definitions depending upon the nature of the task and whether one wishes to discount delayed reward. The undiscounted formulation is appropriate for episodic tasks, in which the agent–environment interaction breaks naturally into episodes; the discounted formulation is appropriate for continuing tasks, in which the interaction does not naturally break into episodes but continues without limit. 
An environment satisfies the Markov property if its state signal compactly summarizes the past without degrading the ability to predict the future. This is rarely exactly true, but often nearly so; the state signal should be chosen or constructed so that the Markov property holds as nearly as possible. In this book we assume that this has already been done and focus on the decisionmaking problem: how to decide what to do as a function of whatever state signal is available. If the Markov property does hold, then the environment is called a Markov decision process (MDP). A finite $M D P$ is an MDP with finite state and action sets. Most of the current theory of reinforcement learning is restricted to finite MDPs, but the methods and ideas apply more generally. 
A policy’s value functions assign to each state, or state–action pair, the expected return from that state, or state–action pair, given that the agent uses the policy. The optimal value functions assign to each state, or state–action pair, the largest expected return achievable by any policy. A policy whose value functions are optimal is an optimal policy. Whereas the optimal value functions for states and state–action pairs are unique for a given MDP, there can be many optimal policies. Any policy that is greedy with respect to the optimal value functions must be an optimal policy. The Bellman optimality equations are special consistency condition that the optimal value functions must satisfy and that can, in principle, be solved for the optimal value functions, from which an optimal policy can be determined with relative ease. 
A reinforcement learning problem can be posed in a variety of different ways depending on assumptions about the level of knowledge initially available to the agent. In problems of complete knowledge, the agent has a complete and accurate model of the environment’s dynamics. If the environment is an MDP, then such a model consists of the one-step transition probabilities and expected rewards for all states and their allowable actions. In problems of incomplete knowledge, a complete and perfect model of the environment is not available. 
Even if the agent has a complete and accurate environment model, the agent is typically unable to perform enough computation per time step to fully use it. The memory available is also an important constraint. Memory may be required to build up accurate approximations of value functions, policies, and models. In most cases of practical interest there are far more states than could possibly be entries in a table, and approximations must be made. 
A well-defined notion of optimality organizes the approach to learning we describe in this book and provides a way to understand the theoretical properties of various learning algorithms, but it is an ideal that reinforcement learning agents can only approximate to varying degrees. In reinforcement learning we are very much concerned with cases in which optimal solutions cannot be found but must be approximated in some way. 

# 4 Dynamic Programming 
The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP). Classical DP algorithms are of limited utility in reinforcement learning both because of their assumption of a perfect model and because of their great computational expense, but they are still important theoretically. DP provides an essential foundation for the understanding of the methods presented in the rest of this book. In fact, all of these methods can be viewed as attempts to achieve much the same effect as DP, only with less computation and without assuming a perfect model of the environment. 
>  动态规划指在环境被完美建模为 MDP 问题下用于计算最优策略的一类算法
>  经典的动态规划算法需要完美的模型，并且计算开销较大，因此应用有限，但具有理论价值
>  之后介绍的方法都可以视作以更少的计算或者不假设环境具有完美模型的情况下，尝试达到和 DP 相同的效果

Starting with this chapter, we usually assume that the environment is a finite MDP. That is, we assume that its state, action, and reward sets, $\mathcal S$, ${\mathcal{A}}(s)$ , and $\mathcal{R}$ , for $s\in\mathcal{S}$ , are finite, and that its dynamics are given by a set of probabilities $p(s^{\prime},r|s,a)$ , for all $s\in\mathcal S$ , $a\in\mathcal A(s)$ , $r\in\mathcal R$ , and $s^{\prime}\in\mathcal{S}^{+}$ ( $\mathcal{S}^{+}$ is $\mathcal S$ plus a terminal state if the problem is episodic). Although DP ideas can be applied to problems with continuous state and action spaces, exact solutions are possible only in special cases. A common way of obtaining approximate solutions for tasks with continuous states and actions is to quantize the state and action spaces and then apply finite-state DP methods. The methods we explore in Chapter 9 are applicable to continuous problems and are a significant extension of that approach. 
>  我们假设环境是有限 MDP，即其状态集合、动作集合、奖励集合 $\mathcal S, \mathcal A(s)\ \text{for}\ s\in \mathcal S, \mathcal R$ 都是有限集合。MDP 的动态由一组概率 $p(s', r\mid s, a)\ \text{for all}\ a\in \mathcal S, a\in \mathcal A(s), r\in \mathcal R, s' \in \mathcal S^+$ ($\mathcal S^+$ 为 $\mathcal S$ 再加上一个终止状态)
>  DP 一般情况下仅限于有限 MDP，对于连续的状态和动作空间，一般会进行量化近似

The key idea of DP, and of reinforcement learning generally, is the use of value functions to organize and structure the search for good policies. In this chapter we show how DP can be used to compute the value functions defined in Chapter 3. As discussed there, we can easily obtain optimal policies once we have found the optimal value functions, $v_{*}$ or $q_{*}$ , which satisfy the Bellman optimality equations: 
>  DP 的关键思想是用价值函数来组织和结构化对策略的搜索，即我们用 DP 计算最优价值函数，根据最优价值函数定义策略

>  最优状态价值函数和最优动作价值函数对于所有的 $s\in \mathcal S, a\in \mathcal A(s), s' \in \mathcal S^+$ 满足 Bellman 最优方程，如下所示：

$$
\begin{align}
v_*(s)&=\max_a \mathbb E\left[R_{t+1}+\gamma v_*(S_{t+1})\mid S_t = s, A_t = a\right]\\
&=\max_a\sum_{s',r}p(s',r\mid s, a)[r + \gamma v_*(s')]\tag{4.1}
\end{align}
$$

or 

$$
\begin{align}
q_*(s, a)&= \mathbb E\left[R_{t+1} + \gamma \max_a'q_*(S_{t+1}, a')\mid S_t = s, A_t = a\right]\\
&=\sum_{s',r}p(s',r\mid s, a)[r + \gamma\max_{a'}q_*(s',a')]\tag{4.2}
\end{align}
$$

for all $s\in\mathcal S$ , $a\in\mathcal{A}(s)$ , and $s^{\prime}\in\mathcal{S}^{+}$ . 

As we shall see, DP algorithms are obtained by turning Bellman equations such as these into assignments, that is, into update rules for improving approximations of the desired value functions. 
>  DP 算法将 Bellman 方程转化为赋值，即转化为改进所需的价值函数 (的近似) 的更新规则

## 4.1 Policy Evaluation 
First we consider how to compute the state-value function $v_{\pi}$ for an arbitrary policy $\pi$ . This is called policy evaluation in the DP literature. We also refer to it as the prediction problem. Recall from Chapter 3 that, for all $s\in\mathcal{S}$ , 

$$
\begin{align}
v_\pi(s) &=\mathbb E_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots\mid S_t = s]\\
&=\mathbb E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})\mid S_t = s]\tag{4.3}\\
&=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s, a)[r + \gamma v_\pi(s')]\tag{4.4}
\end{align}
$$

where $\pi(a|s)$ is the probability of taking action $a$ in state $s$ under policy $\pi$ , and the expectations are subscripted by $\pi$ to indicate that they are conditional on $\pi$ being followed. The existence and uniqueness of $v_{\pi}$ are guaranteed as long as either $\gamma<1$ or eventual termination is guaranteed from all states under the policy $\pi$ . 
>  考虑为任意策略 $\pi$ 计算其状态价值函数 $v_\pi$，DP 中称为策略评估 (注意策略评估的目标是计算策略的状态价值函数)
>  我们知道 $v_\pi$ 满足 Eq 4.3 和 Eq 4.4 描述的 Bellman 方程，只要折扣因子 $\gamma < 1$ 或者在策略 $\pi$ 下，从任意状态开始都可以达到终止状态，$v_\pi$ 的存在性和唯一性就可以保证 
>  ($v_\pi$ 的计算展开来实际上是一个级数，如果 $\gamma< 1$，即便没有确定的终止状态，该级数也会收敛，故 $v_\pi$ 的存在性可以保证；而如果能在有限步终止，则 $v_\pi$ 就不是无穷级数，故 $v_\pi$ 的存在性也可以保证，即便此时 $\gamma$ 不一定 $<1$)

If the environment’s dynamics are completely known, then (4.4) is a system of $|\mathcal S|$ simultaneous linear equations in $|\mathcal S|$ unknowns (the $v_{\pi}(s)$ , $s\in\mathcal{S}$ ). In principle, its solution is a straightforward, if tedious, computation. For our purposes, iterative solution methods are most suitable. 
>  如果环境动态完全已知，则 (4.4) 就是包含了 $|\mathcal S|$ 个未知数 (这些未知数是 $v_\pi(s), s\in \mathcal S$) 和 $|\mathcal S|$ 个线性方程的方程组
>  原则上，求解改线性方程组较为繁琐但直接，而迭代解法是较为合适的解法

Consider a sequence of approximate value functions $v_{0},v_{1},v_{2},\ldots.$ , each mapping $\mathcal{S}^{+}$ to $\mathbb{R}$ . The initial approximation, $v_{0}$ , is chosen arbitrarily (except that the terminal state, if any, must be given value $0$ ), and each successive approximation is obtained by using the Bellman equation for $v_{\pi}$ (3.12) as an update rule: 

$$
\begin{align}
v_{k+1}(s)&=\mathbb E[R_{t+1} + \gamma v_k(S_{t+1})\mid S_t = a]\\
&=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s, a)[r + \gamma v_k(s')]\tag{4.5}
\end{align}
$$

for all $s\in\mathcal{S}$ . 

>  考虑一个近似价值函数序列 $v_0, v_1, v_2, \dots$，每个价值函数都是从 $\mathcal S^+$ 到 $\mathbb R$ 的映射
>  初始的近似函数 $v_0$ 可以任意选择 (但注意其中终止状态的价值需要为 0)，而序列中的每个后继近似值可以根据 $v_\pi$ 的 Bellman 方程作为更新规则而计算，如 Eq 4.5 所示
>  Eq 4.5 的更新规则适用于所有 $\mathcal s \in \mathcal S$

Clearly, $v_{k}=v_{\pi}$ is a fixed point for this update rule because the Bellman equation for $v_{\pi}$ assures us of equality in this case. Indeed, the sequence $\{v_{k}\}$ can be shown in general to converge to $v_{\pi}$ as $k\rightarrow\infty$ under the same conditions that guarantee the existence of $v_{\pi}$ . This algorithm is called iterative policy evaluation. 
>  显然，在该更新规则下，$v_k = v_\pi$ 是一个固定点 (即将 $v_\pi$ 代入 Eq 4.5 的 RHS，得到的 LHS 仍然是 $v_\pi$，故更新收敛)，$v_\pi$ 的 Bellman equation 确保了这一性质成立
>  可以证明，随着 $k\to \infty$，在保证 $v_\pi$ 存在的相同条件下，序列 $\{v_k\}$ 将收敛到 $v_\pi$，这一算法就称为迭代式策略评估

To produce each successive approximation, $v_{k+1}$ from $v_{k}$ , iterative policy evaluation applies the same operation to each state $s$ : it replaces the old value of $s$ with a new value obtained from the old values of the successor states of $s$ , and the expected immediate rewards, along all the one-step transitions possible under the policy being evaluated. We call this kind of operation a full backup. Each iteration of iterative policy evaluation backs up the value of every state once to produce the new approximate value function $v_{k+1}$ . 
>  迭代式策略评估执行连续的价值函数估计，基于旧估计 $v_k$ 生成新估计 $v_{k+1}$
>  迭代式策略评估为每个状态 $s$ 执行相同的操作：将 $s$ 的旧价值替换为根据 $s$ 的后继状态的旧价值以及期望的中间奖励计算得到的新价值的期望值 (期望基于策略 $\pi$ 和环境的一步动态计算)，这类操作称为全回溯更新
>  迭代式策略评估的每次迭代都会被每个状态的价值执行一次全回溯更新，进而得到新的近似价值函数 $v_{k+1}$

There are several different kinds of full backups, depending on whether a state (as here) or a state–action pair is being backed up, and depending on the precise way the estimated values of the successor states are combined. All the backups done in DP algorithms are called full backups because they are based on all possible next states rather than on a sample next state.
>  全回溯更新操作有多种类型，取决于是回溯更新一个状态 (评估状态价值函数) 还是状态-动作对 (评估动作价值函数)，还取决于后继状态的估计价值是如何结合的
>  DP 算法中做的所有的回溯更新操作都是全回溯更新，因为它们基于所有可能的后继状态，而不是一个 (采样得到的) 样本后继状态

 The nature of a backup can be expressed in an equation, as above, or in a backup diagram like those introduced in Chapter 3. For example, Figure 3.4a is the backup diagram corresponding to the full backup used in iterative policy evaluation. 
 >  回溯更新的性质可以用上述的 Bellman 方程解释 (即 Eq 4.5)，或者用回溯更新图解释
 >  迭代式策略评估对应于 Figure 3.4a 的回溯更新图

To write a sequential computer program to implement iterative policy evaluation, as given by (4.5), you would have to use two arrays, one for the old values, $v_{k}(s)$ , and one for the new values, $v_{k+1}(s)$ . This way, the new values can be computed one by one from the old values without the old values being changed. Of course it is easier to use one array and update the values “in place,” that is, with each new backed-up value immediately overwriting the old one. Then, depending on the order in which the states are backed up, sometimes new values are used instead of old ones on the right-hand side of (4.5). This slightly different algorithm also converges to $v_{\pi}$ ; in fact, it usually converges faster than the two-array version, as you might expect, since it uses new data as soon as they are available. 
>  要实现执行 Eq 4.5 的迭代式策略评估的顺序执行的程序，需要用到两个数组，一个存储旧价值 $v_k(s)$，一个存储新价值 $v_{k+1}(s)$，每一次迭代根据一个数组的值更新另一个数组的值
>  实际上也可以仅使用一个数组，直接原地更新价值，取决于状态被回溯更新的顺序，有时 Eq 4.5 的 RHS 可能会用到新价值而不是旧价值，但该算法仍然可以证明收敛，并且一般收敛更快，因为它一有新数据就立即使用

We think of the backups as being done in a sweep through the state space. For the in-place algorithm, the order in which states are backed up during the sweep has a significant influence on the rate of convergence. We usually have the in-place version in mind when we think of DP algorithms. 
>  我们使用原地修改的版本，遍历整个状态空间执行回溯更新操作，实际上遍历过程中的顺序对于算法的收敛速率有很大影响

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/208c5870260790cbf9bca7dc6c715f4c4b7c3dfd157f032f5e63f0850b44ec8c.jpg) 

Figure 4.1: Iterative policy evaluation. 

Another implementation point concerns the termination of the algorithm. Formally, iterative policy evaluation converges only in the limit, but in practice it must be halted short of this. A typical stopping condition for iterative policy evaluation is to test the quantity max $s\in\mathcal{S}$ $|v_{k+1}(s)-v_{k}(s)|$ after each sweep and stop when it is sufficiently small. Figure 4.1 gives a complete algorithm for iterative policy evaluation with this stopping criterion. 
>  另一个实现上的要点涉及算法的终止，形式上，策略迭代算法仅在极限情况下收敛，实践中，一般的停止条件是在每次迭代后测试 $\max_{s\in \mathcal S}|v_{k+1}(s) - v_k(s)|$ 的值是否足够小

Example 4.1 Consider the $4\times4$ gridworld shown below. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/300aea38dffe36886d8d2e0f8392fc2ee231a9bad468e7e673a7b5e4608630a4.jpg) 

The nonterminal states are $\mathcal{S}=\{1,2,\ldots,14\}$ . There are four actions possible in each state, ${\mathcal{A}}={\mathfrak{f}}\mathfrak{u}\mathfrak{p}$ , down, right, left}, which deterministically cause the corresponding state transitions, except that actions that would take the agent off the grid in fact leave the state unchanged. Thus, for instance, $p(6|5,\mathtt{r i g h t})=1$ , $p(10|5,\mathrm{right})=0$ , and $p(7|7,\mathtt{r i g h t})=1$ . This is an undiscounted, episodic task. The reward is $-1$ on all transitions until the terminal state is reached. The terminal state is shaded in the figure (although it is shown in two places, it is formally one state). The expected reward function is thus $r(s,a,s^{\prime})=-1$ for all states $s,s^{\prime}$ and actions $a$ . Suppose the agent follows the equiprobable random policy (all actions equally likely). The left side of Figure 4.2 shows the sequence of value functions $\{v_{k}\}$ computed by iterative policy evaluation. The final estimate is in fact $v_{\pi}$ , which in this case gives for each state the negation of the expected number of steps from that state until termination. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/f2d3f8357a59914825a4820d30e3c09de9e7bfc3935df37fa238caadf7a2f7bf.jpg) 

Figure 4.2: Convergence of iterative policy evaluation on a small gridworld. The left column is the sequence of approximations of the state-value function for the random policy (all actions equal). The right column is the sequence of greedy policies corresponding to the value function estimates (arrows are shown for all actions achieving the maximum). The last policy is guaranteed only to be an improvement over the random policy, but in this case it, and all policies after the third iteration, are optimal. 

## 4.2 Policy Improvement 
Our reason for computing the value function for a policy is to help find better policies. Suppose we have determined the value function $v_{\pi}$ for an arbitrary deterministic policy $\pi$ . For some state $s$ we would like to know whether or not we should change the policy to deterministically choose an action $a\neq\pi(s)$ . We know how good it is to follow the current policy from $s$ —that is $v_{\pi}(s)$ —but would it be better or worse to change to the new policy? 
>  为特定策略计算价值的一个原因是帮助找到更优的策略
>  假设我们已经为策略 $\pi$ 确定了其价值 $v_\pi$ ，此时，对于某个状态 $s$，我们希望知道我们是否应该改变当前策略，以确定性地选择动作 $a\ne \pi(s)$ (选择不是根据当前策略得到的动作)。我们已经知道了从 $s$ 开始，遵循 $\pi$ 能够获得的期望奖励 $v_\pi(s)$，但我们要探究的是如果换成一个新策略，期望奖励是否会更好还是更坏

One way to answer this question is to consider selecting $a$ in $s$ and thereafter following the existing policy, $\pi$ . The value of this way of behaving is 

$$
\begin{align}
q_\pi(s, a)&=\mathbb E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})\mid S_t = s, A_t = a]\tag{4.6}\\
&=\sum_{s',r}p(s',r\mid s, a)[r + \gamma v_\pi(s')]
\end{align}
$$

The key criterion is whether this is greater than or less than $v_{\pi}(s)$ . If it is greater—that is, if it is better to select $a$ once in $s$ and thereafter follow $\pi$ than it would be to follow $\pi$ all the time—then one would expect it to be better still to select $a$ every time $s$ is encountered, and that the new policy would in fact be a better one overall. 

>  要回答这一问题，我们不妨考虑 $\pi$ 的动作价值函数，即先确定性选择一个动作 $a$，然后再根据 $\pi$ 进行行为，考察这样能获得多少期望奖励
>  我们需要比较 $q_\pi(s, a)$ 是否会高于 $v_\pi(s)$，如果高，说明我们最好先选择 $a$，然后再遵循 $\pi$，并且，在之后遇到了 $s$ 时，也最好直接选择 $a$，这样得到的新策略显然会优于旧策略

That this is true is a special case of a general result called the policy improvement theorem. Let $\pi$ and $\pi^{\prime}$ be any pair of deterministic policies such that, for all $s\in\mathcal{S}$ , 

$$
q_{\pi}(s,\pi^{\prime}(s))\geq v_{\pi}(s).\tag{4.7}
$$ 
Then the policy $\pi^{\prime}$ must be as good as, or better than, $\pi$ . That is, it must obtain greater or equal expected return from all states $s\in\mathcal{S}$ : 

$$
v_{\pi^{\prime}}(s)\geq v_{\pi}(s).\tag{4.8}
$$

Moreover, if there is strict inequality of (4.7) at any state, then there must be strict inequality of (4.8) at at least one state. 

>  以上的讨论实际上就是策略提升定理的一个一般性结果
>  令 $\pi$ 和 $\pi'$ 是任意的一对确定性策略，二者对于任意的 $s\in \mathcal S$ 满足 Eq (4.7)，即在遇到任意状态 $s$ 时，根据 $\pi'$ 选择动作得到的价值会优于根据 $\pi$ 选择动作得到的价值。那么，策略 $\pi'$ 一定是一个不差于 $\pi$ 的策略，即对于所有的状态 $s\in \mathcal S$，都有 Eq 4.8 成立
>  并且，如果 Eq 4.7 在任意一个状态是严格的不等式，则 Eq 4.8 在至少一个状态下也是严格的不等式

This result applies in particular to the two policies that we considered in the previous paragraph, an original deterministic policy, $\pi$ , and a changed policy, $\pi^{\prime}$ , that is identical to $\pi$ except that $\pi^{\prime}(s)=a\neq\pi(s)$ . Obviously, (4.7) holds at all states other than $s$ . Thus, if $q_{\pi}(s,a)>v_{\pi}(s)$ , then the changed policy is indeed better than $\pi$ . 
>  显然，我们之前讨论的两个策略：原来的策略 $\pi$ 和修改后的策略 $\pi'$ 满足策略提升定理，其中修改后的策略定义为： $\pi'(s) = a \ne \pi(s)$，其他情况都与 $\pi$ 一致
>  显然，Eq 4.7 对于所有状态都成立，并且对于状态 $s$ 有严格的不等式成立，故修改后的状态就会优于 $\pi$

The idea behind the proof of the policy improvement theorem is easy to understand. Starting from (4.7), we keep expanding the $q_{\pi}$ side and reapplying (4.7) until we get $v_{\pi^{\prime}}(s)$ : 

$$
\begin{align}
v_\pi(s)&\le q_\pi(s, \pi'(s))\\
&=\mathbb E_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1})\mid S_t = s]\\
&\le\mathbb E_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1}))\mid S_t = s]\\
&=\mathbb E_{\pi'}[R_{t+1} + \gamma \mathbb E_{\pi'}[R_{t+2} + \gamma v_{\pi}(S_{t+2})]\mid S_t=s]\\
&=\mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_{\pi}(S_{t+2})]\mid S_t=s]\\
&\le \mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2q_\pi(S_{t+2}, \pi'(S_{t+2}))\mid S_t = s]\\
&\le \mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2\mathbb E_{\pi'}[R_{t+3} + \gamma v_\pi(S_{t+3})]\mid S_t = s]\\
&=\mathbb E_{\pi'}[R_{t+1}+ \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3v_\pi(S_{t+3})\mid S_t = s]\\
&\quad\vdots\\
&\le \mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4}\cdots\mid S_t = s]\\
&=v_{\pi'}(s)
\end{align}
$$

>  策略提升定理的证明思路是从 Eq 4.7 开始，不断地展开 $q_\pi$，然后重新应用 Eq 4.7，最后得到 $v_{\pi'}(s)$

So far we have seen how, given a policy and its value function, we can easily evaluate a change in the policy at a single state to a particular action. It is a natural extension to consider changes at all states and to all possible actions, selecting at each state the action that appears best according to $q_{\pi}(s,a)$ . 
>  我们目前考虑了给定策略和其价值函数的情况下，对策略进行“在特定状态将选定一个特定动作“这样的改变所能带来的影响
>  我们进一步考虑在所有状态下遍历所有的可能动作，选择出能够具有最大 $q_\pi(s, a)$ 的动作

In other words, to consider the new greedy policy, $\pi^{\prime}$ , given by 

$$
\begin{align}
\pi'(s)&=\arg\max_a q_\pi(s, a)\\
&=\arg\max_a \mathbb E[R_{t+1} + \gamma v_\pi(S_{t+1})\mid S_t = s, A_t = a]\tag{4.9}\\
&=\arg\max_a \sum_{s', r}p(s', r \mid s, a)[r + \gamma v_\pi(s')]
\end{align}
$$

where $\mathrm{arg}\operatorname*{max}_{a}$ denotes the value of $a$ at which the expression that follows is maximized (with ties broken arbitrarily). The greedy policy takes the action that looks best in the short term—after one step of lookahead—according to $v_{\pi}$ . By construction, the greedy policy meets the conditions of the policy improvement theorem (4.7), so we know that it is as good as, or better than, the original policy. The process of making a new policy that improves on an original policy, by making it greedy with respect to the value function of the original policy, is called policy improvement. 

>  换句话说，我们考虑如 Eq 4.9 的贪心策略，它选择动作的依据是原策略的动作价值函数 $q_\pi(s, a)$
>  该贪心策略选择短期 (经过一步探查) 来看最优的 (根据 $v_\pi$) 动作，容易证明，该贪心策略满足策略改进定理的条件 (Eq 4.7)，因此该策略不会差于原策略
>  基于原策略的价值函数定义贪心的新策略的这一过程称为策略改进

Suppose the new greedy policy, $\pi^{\prime}$ , is as good as, but not better than, the old policy $\pi$ . Then $v_{\pi}=v_{\pi^{\prime}}$ , and from (4.9) it follows that for all $s\in\mathcal{S}$ : 
>  假设随着策略的不断改进，新策略 $\pi'$ 已经不能再基于原策略 $\pi$ 而提升，而是保持一样好的水平，即 $v_\pi = v_{\pi'}$，则将 $v_{\pi'}$ 代入 Eq 4.9 的 LHS 和 RHS，可以得到对于所有的 $s\in \mathcal S$，都有以下式子成立：

$$
\begin{align}
v_{\pi'}(s)&=\max_a\mathbb E[R_{t+1} + \gamma v_{\pi'}(S_{t+1})\mid S_t = s,A_t = a]\\
&=\max_a\sum_{s',r}p(s',r\mid s, a)[r+\gamma v_{\pi'}(s')]
\end{align}
$$

But this is the same as the Bellman optimality equation (4.1), and therefore, $v_{\pi^{\prime}}$ must be $v_{*}$ , and both $\pi$ and $\pi^{\prime}$ must be optimal policies. Policy improvement thus must give us a strictly better policy except when the original policy is already optimal. 
>  可以看出，该式就是 Bellman 最优方程，因此，$v_{\pi'}(s)$ 就是 Bellman 最优方程的解，即 $v_ {\pi'}$ 就是 $v_*$
>  因此，策略改进过程总是可以给出一个更好的策略，除非原始策略已经是最优策略

So far in this section we have considered the special case of deterministic policies. In the general case, a stochastic policy $\pi$ specifies probabilities, $\pi(\boldsymbol{a}|\boldsymbol{s})$ , for taking each action, $a$ , in each state, $s$ . We will not go through the details, but in fact all the ideas of this section extend easily to stochastic policies. In particular, the policy improvement theorem carries through as stated for the stochastic case, under the natural definition: 

$$
q_{\pi}(s,\pi^{\prime}(s))=\sum_{a}\pi^{\prime}(a|s)q_{\pi}(s,a).
$$ 
>  目前为止，我们考虑的都是确定性策略，再一般情况下，随机性策略 $\pi$ 指定的是在每个状态 $s$ 下执行各个动作 $a$ 的概率 $\pi(a\mid s)$
>  可以证明，之前讨论的所有概念都可以拓展到随机性策略，特别地，策略改进定律也可以应用到随机性策略的情况，其中 $q_\pi(s, \pi'(s))$ 的定义如上

In addition, if there are ties in policy improvement steps such as (4.9)—that is, if there are several actions at which the maximum is achieved—then in the stochastic case we need not select a single action from among them. Instead, each maximizing action can be given a portion of the probability of being selected in the new greedy policy. Any apportioning scheme is allowed as long as all submaximal actions are given zero probability. 
>  此外，如果在策略改进步骤中出现了平局，即有多个动作都可以达到最大值，则在随机策略的情况下，我们不需要一定从中选出一个，只需要令所有次优的动作的概率为零即可

The last row of Figure 4.2 shows an example of policy improvement for stochastic policies. Here the original policy, $\pi$ , is the equiprobable random policy, and the new policy, $\pi^{\prime}$ , is greedy with respect to $v_{\pi}$ . The value function $v_{\pi}$ is shown in the bottom-left diagram and the set of possible $\pi^{\prime}$ is shown in the bottom-right diagram. The states with multiple arrows in the $\pi^{\prime}$ diagram are those in which several actions achieve the maximum in (4.9); any apportionment of probability among these actions is permitted. The value function of any such policy, $v_{\pi^{\prime}}(s)$ , can be seen by inspection to be either $-1$ , $-2$ , or $^{-3}$ at all states, $s\in\mathcal{S}$ , whereas $v_{\pi}(s)$ is at most $-14$ . Thus, $v_{\pi^{\prime}}(s)\geq v_{\pi}(s)$ , for all $s\in\mathcal S$ , illustrating policy improvement. Although in this case the new policy $\pi^{\prime}$ happens to be optimal, in general only an improvement is guaranteed. 

## 4.3 Policy Iteration 
Once a policy, $\pi$ , has been improved using $v_{\pi}$ to yield a better policy, $\pi^{\prime}$ , we can then compute $v_{\pi^{\prime}}$ and improve it again to yield an even better $\pi^{\prime\prime}$ . We can thus obtain a sequence of monotonically improving policies and value functions: 

$$
\pi_{0}\stackrel{\mathrm{\tiny~E}}{\longrightarrow}v_{\pi_{0}}\stackrel{\mathrm{\tiny~I}}{\longrightarrow}\pi_{1}\stackrel{\mathrm{\tiny~E}}{\longrightarrow}v_{\pi_{1}}\stackrel{\mathrm{\tiny~I}}{\longrightarrow}\pi_{2}\stackrel{\mathrm{\tiny~E}}{\longrightarrow}\cdot\cdot\cdot\stackrel{\mathrm{\tiny~I}}{\longrightarrow}\pi_{*}\stackrel{\mathrm{\tiny~E}}{\longrightarrow}v_{*},
$$ 
where $\xrightarrow{\textrm{E}}$ denotes a policy evaluation and $\xrightarrow{\mathrm{~I~}}$ denotes a policy improvement. 

>  当一个策略 $\pi$ 根据 $v_\pi$ 被改进为更好的策略 $\pi'$ 后，我们需要重新计算 $v_{\pi'}$，以进一步改进 $\pi'$ 得到更好的 $\pi''$
>  因此，我们可以得到一个单调提升的策略和价值函数序列如上，其中 $\stackrel{\mathrm E}{\rightarrow}$ 表示策略评估，$\stackrel{I}{\rightarrow}$ 表示策略提升

Each policy is guaranteed to be a strict improvement over the previous one (unless it is already optimal). Because a finite MDP has only a finite number of policies, this process must converge to an optimal policy and optimal value function in a finite number of iterations. 
>  序列中，每个策略都保证比前一个策略有严格的提升，除非它已经是最优策略
>  因为有限 MDP 只有有限数量个策略，故该过程在有限数量的迭代中一定会收敛到一个最优策略和最优价值函数


![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/1267a494db94feb9dd11f824b3c14be757f29f81edfbf4d283a38589bdddcb31.jpg) 

Figure 4.3: Policy iteration (using iterative policy evaluation) for $v_{*}$ . This algorithm has a subtle bug, in that it may never terminate if the policy continually switches between two or more policies that are equally good. The bug can be fixed by adding additional flags, but it makes the pseudocode so ugly that it is not worth it. :-) 

This way of finding an optimal policy is called policy iteration. A complete algorithm is given in Figure 4.3. Note that each policy evaluation, itself an iterative computation, is started with the value function for the previous policy. This typically results in a great increase in the speed of convergence of policy evaluation (presumably because the value function changes little from one policy to the next). 
>  根据策略评估和策略改进找到最优策略的方式称为策略迭代，完整的算法如 Fig 4.3 所示
>  注意每次迭代中的策略评估本身也是一个迭代过程，并且都会以之前的策略的价值函数作为起点 (而不是随机初始化一个价值函数)，这一般会大大加快策略评估的收敛速度 (大致是因为之前策略和当前策略的价值函数的差异一般并不大)

Policy iteration often converges in surprisingly few iterations. This is illustrated by the example in Figure 4.2. The bottom-left diagram shows the value function for the equiprobable random policy, and the bottom-right diagram shows a greedy policy for this value function. The policy improvement theorem assures us that these policies are better than the original random policy. In this case, however, these policies are not just better, but optimal, proceeding to the terminal states in the minimum number of steps. In this example, policy iteration would find the optimal policy after just one iteration. 
>  策略迭代往往可以在几个迭代下就收敛

Example 4.2: Jack’s Car Rental 
Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he rents it out and is credited $\$10$ by the national company. If he is out of cars at that location, then the business is lost. Cars become available for renting the day after they are returned. To help ensure that cars are available where they are needed, Jack can move them between the two locations overnight, at a cost of $\$2$ per car moved. We assume that the number of cars requested and returned at each location are Poisson random variables, meaning that the probability that the number is $n$ is $\textstyle{\frac{\lambda^{n}}{n!}}e^{-\lambda}$ , where $\lambda$ is the expected number. Suppose $\lambda$ is 3 and 4 for rental requests at the first and second locations and 3 and 2 for returns. To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional cars are returned to the nationwide company, and thus disappear from the problem) and a maximum of five cars can be moved from one location to the other in one night. We take the discount rate to be $\gamma=0.9$ and formulate this as a continuing finite MDP, where the time steps are days, the state is the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight. Figure 4.4 shows the sequence of policies found by policy iteration starting from the policy that never moves any cars. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/047a1d185674055acd4d84601e4a2b12c3f9af7726b32f424df9dccfb211e0b7.jpg) 

Figure 4.4: The sequence of policies found by policy iteration on Jack’s car rental problem, and the final state-value function. The first five diagrams show, for each number of cars at each location at the end of the day, the number of cars to be moved from the first location to the second (negative numbers indicate transfers from the second location to the first). Each successive policy is a strict improvement over the previous policy, and the last policy is optimal. 

## 4.4 Value Iteration 
One drawback to policy iteration is that each of its iterations involves policy evaluation, which may itself be a protracted iterative computation requiring multiple sweeps through the state set. If policy evaluation is done iteratively, then convergence exactly to $v_{\pi}$ occurs only in the limit. Must we wait for exact convergence, or can we stop short of that? The example in Figure 4.2 certainly suggests that it may be possible to truncate policy evaluation. In that example, policy evaluation iterations beyond the first three have no effect on the corresponding greedy policy. 
>  策略迭代的缺点是它的每次迭代都涉及策略评估，策略评估本身是一个冗长的计算，需要多次遍历状态集合，并且往往只有在极限情况下才能收敛到 $v_\pi$
>  实践中，我们并不需要等到策略评估完全收敛，而是可以提前停止

In fact, the policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One important special case is when policy evaluation is stopped after just one sweep (one backup of each state). This algorithm is called value iteration. It can be written as a particularly simple backup operation that combines the policy improvement and truncated policy evaluation steps: 

$$
\begin{align}
v_{k+1}(s)&=\max_a\mathbb E[R_{t+1} + \gamma v_k(S_{t+1})\mid S_t = s, A_t = a]\tag{4.10}\\
&=\max_a\sum_{s',r}p(s',r\mid s, a)[r + \gamma v_k(s')]
\end{align}
$$

for all $s\in\mathcal{S}$ . For arbitrary $v_{0}$ , the sequence $\{v_{k}\}$ can be shown to converge to $v_{*}$ under the same conditions that guarantee the existence of $v_{*}$ . 

>  实践中，策略迭代中的策略评估可以通过多种方式截断，同时不失去策略迭代的收敛保证
>  一个重要的特例就是在策略评估遍历了一轮 (每个状态一次回溯更新) 之后就停止，该算法称为价值迭代，价值迭代的更新公式如 Eq 4.10 所示
>  对于任意的 $v_0$，可以证明序列 $\{v_k\}$ 会在 $v_*$ 存在的相同条件下收敛到 $v_*$

>  (4.10) 包括了
>  1. 一步的策略评估更新，得到新的状态价值函数 $v_k'(s)=\mathbb E_\pi[R_{t+1} + \gamma v_k(S_{t+1})\mid S_t = s]$
>  2. 根据 $v_k'(s)$ 计算动作价值函数 $q_k'(s, a) = \mathbb E_\pi[R_{t+1} + \gamma v_k(S_{t+1})\mid S_t = s, A_t = a]$
>  3. 策略改进，得到新策略 $\pi'$，动作选择依据是 $\arg\max_a q_k'(s, a) = \arg\max_a \mathbb E[R_{t+1} + \gamma v_k(S_{t+1})\mid S_t = a, A_t = a]$
>  4. 直接让 $\max_a q_k'(s, a)$ 作为新的策略的状态价值函数 $v_{k+1}(s)$，这等价于让 $v_k(s)$ 作为 $\pi'$ 的价值函数的初始值，对它执行一步的策略评估更新，即 $v_{k+1}(s) = \mathbb E_{\pi'}[R_{t+1} +\gamma v_k(S_{t+1})\mid S_t=  s]= \max_a \mathbb E[R_{t+1} + \gamma v_k(S_{t+1})\mid S_t = s, A_t = a]$

Another way of understanding value iteration is by reference to the Bellman optimality equation (4.1). Note that value iteration is obtained simply by turning the Bellman optimality equation into an update rule. Also note how the value iteration backup is identical to the policy evaluation backup (4.5) except that it requires the maximum to be taken over all actions. Another way of seeing this close relationship is to compare the backup diagrams for these algorithms: Figure 3.4a shows the backup diagram for policy evaluation and Figure 3.7a shows the backup diagram for value iteration. These two are the natural backup operations for computing $v_{\pi}$ and $v_{*}$ . 
>  另一个理解价值迭代的方式是 Bellman 最优方程，即将 Bellman 最优方程转化为一个更新规则
>  价值迭代和策略评估非常相似，差异仅在于它要求取所有动作中的最大值

Finally, let us consider how value iteration terminates. Like policy evaluation, value iteration formally requires an infinite number of iterations to converge exactly to $v_{*}$ . In practice, we stop once the value function changes by only a small amount in a sweep. Figure 4.5 gives a complete value iteration algorithm with this kind of termination condition. 
>  和策略评估一样，价值迭代形式上要求无穷多次的迭代以准确收敛到 $v_*$，实践中，当价值函数在一次遍历中修改的幅度较小时，我们就停止

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/6fce07fdd317278d43251337471be073a3aad6e43a28c6d58415899b2892bcb7.jpg) 

Figure 4.5: Value iteration. 

Value iteration effectively combines, in each of its sweeps, one sweep of policy evaluation and one sweep of policy improvement. Faster convergence is often achieved by interposing multiple policy evaluation sweeps between each policy improvement sweep. In general, the entire class of truncated policy iteration algorithms can be thought of as sequences of sweeps, some of which use policy evaluation backups and some of which use value iteration backups. Since the max operation in (4.10) is the only difference between these backups, this just means that the max operation is added to some sweeps of policy evaluation. All of these algorithms converge to an optimal policy for discounted finite MDPs. 
>  价值迭代在其每次遍历中结合了一遍策略评估和一遍策略改进
>  通常，在每次策略改进遍历之间插入多次的策略评估遍历可以加快收敛

Example 4.3: Gambler’s Problem A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips. If the coin comes up heads, he wins as many dollars as he has staked on that flip; if it is tails, he loses his stake. The game ends when the gambler wins by reaching his goal of $\$100$ , or loses by running out of money. On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars. This problem can be formulated as an undiscounted, episodic, finite MDP. The state is the gambler’s capital, $s\in\{1,2,\ldots,99\}$ and the actions are stakes, $a\in\{0,1,\ldots,\operatorname*{min}(s,100-s)\}$ . The reward is zero on all transitions except those on which the gambler reaches his goal, when it is $+1$ . The state-value function then gives the probability of winning from each state. A policy is a mapping from levels of capital to stakes. The optimal policy maximizes the probability of reaching the goal. Let $p_{h}$ denote the probability of the coin coming up heads. If $p_{h}$ is known, then the entire problem is known and it can be solved, for instance, by value iteration. Figure 4.6 shows the change in the value function over successive sweeps of value iteration, and the final policy found, for the case of $p_{h}=0.4$ . This policy is optimal, but not unique. In fact, there is a whole family of optimal policies, all corresponding to ties for the argmax action selection with respect to the optimal value function. Can you guess what the entire family looks like?  

## 4.5 Asynchronous Dynamic Programming 
A major drawback to the DP methods that we have discussed so far is that they involve operations over the entire state set of the MDP, that is, they require sweeps of the state set. If the state set is very large, then even a single sweep can be prohibitively expensive. For example, the game of backgammon has over $10^{20}$ states. Even if we could perform the value iteration backup on a million states per second, it would take over a thousand years to complete a single sweep. 

Asynchronous DP algorithms are in-place iterative DP algorithms that are not organized in terms of systematic sweeps of the state set. These algorithms back up the values of states in any order whatsoever, using whatever values of other states happen to be available. The values of some states may be backed up several times before the values of others are backed up once. To converge correctly, however, an asynchronous algorithm must continue to backup the values of all the states: it can’t ignore any state after some point in the computation. Asynchronous DP algorithms allow great flexibility in selecting states to which backup operations are applied. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/e1b8ccfe1c2958b76f442927c3265826bd4e6502ce5e9cb16cb16164091410a5.jpg) 
Figure 4.6: The solution to the gambler’s problem for $p_{h}=0.4$ . The upper graph shows the value function found by successive sweeps of value iteration. The lower graph shows the final policy. 

For example, one version of asynchronous value iteration backs up the value, in place, of only one state, $s_{k}$ , on each step, $k$ , using the value iteration backup (4.10). If $0\leq\gamma<1$ , asymptotic convergence to $v_{*}$ is guaranteed given only that all states occur in the sequence $\{s_{k}\}$ an infinite number of times (the sequence could even be stochastic). (In the undiscounted episodic case, it is possible that there are some orderings of backups that do not result in convergence, but it is relatively easy to avoid these.) Similarly, it is possible to intermix policy evaluation and value iteration backups to produce a kind of asynchronous truncated policy iteration. Although the details of this and other more unusual DP algorithms are beyond the scope of this book, it is clear that a few different backups form building blocks that can be used flexibly in a wide variety of sweepless DP algorithms. 

Of course, avoiding sweeps does not necessarily mean that we can get away with less computation. It just means that an algorithm does not need to get locked into any hopelessly long sweep before it can make progress improving a policy. We can try to take advantage of this flexibility by selecting the states to which we apply backups so as to improve the algorithm’s rate of progress. We can try to order the backups to let value information propagate from state to state in an efficient way. Some states may not need their values backed up as often as others. We might even try to skip backing up some states entirely if they are not relevant to optimal behavior. Some ideas for doing this are discussed in Chapter 8. 

Asynchronous algorithms also make it easier to intermix computation with real-time interaction. To solve a given MDP, we can run an iterative DP algorithm at the same time that an agent is actually experiencing the MDP. The agent’s experience can be used to determine the states to which the DP algorithm applies its backups. At the same time, the latest value and policy information from the DP algorithm can guide the agent’s decision-making. For example, we can apply backups to states as the agent visits them. This makes it possible to focus the DP algorithm’s backups onto parts of the state set that are most relevant to the agent. This kind of focusing is a repeated theme in reinforcement learning. 

## 4.6 Generalized Policy Iteration 
Policy iteration consists of two simultaneous, interacting processes, one making the value function consistent with the current policy (policy evaluation), and the other making the policy greedy with respect to the current value function (policy improvement). In policy iteration, these two processes alternate, each completing before the other begins, but this is not really necessary. In value iteration, for example, only a single iteration of policy evaluation is performed in between each policy improvement. 
>  策略迭代包括了两个同时进行且互相交互的过程，一个是策略评估，使得价值函数与当前策略一致，一个是策略改进，使得策略对于当前的价值函数贪心
>  策略迭代中，这两个过程交替进行，每个过程在上个过程结束后再开始，但这并不是必要的，例如，可以在每次策略改进之间仅执行一次策略评估迭代

In asynchronous DP methods, the evaluation and improvement processes are interleaved at an even finer grain. In some cases a single state is updated in one process before returning to the other. As long as both processes continue to update all states, the ultimate result is typically the same—convergence to the optimal value function and an optimal policy. 
>  在异步 DP 方法中，评估和改进过程甚至以更细的粒度交错进行，某些情况下，一个过程仅更新一个状态，就交替到另一个过程
>  只要两个过程继续更新所有的状态，最终的结果是一致的——收敛到最优价值函数和最优策略

We use the term generalized policy iteration (GPI) to refer to the general idea of letting policy evaluation and policy improvement processes interact, independent of the granularity and other details of the two processes. Almost all reinforcement learning methods are well described as GPI. That is, all have identifiable policies and value functions, with the policy always being improved with respect to the value function and the value function always being driven toward the value function for the policy. This overall schema for GPI is illustrated in Figure 4.7. 
>  我们称让策略评估和策略改进过程相互交互的思想为广义策略迭代
>  几乎所有的强化学习方法都可以描述为广义策略迭代，也就是说，这些方法都具有可识别的策略和价值函数，并且在学习过程中，策略总是基于价值函数进行改进，同时价值函数总是趋向于该策略的真实价值函数

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/4e89034c6f9aa89953587a4d766b70f850e719e91dc8ef09671ee055937ce40e.jpg) 

Figure 4.7: Generalized policy iteration: Value and policy functions interact until they are optimal and thus consistent with each other. 

It is easy to see that if both the evaluation process and the improvement process stabilize, that is, no longer produce changes, then the value function and policy must be optimal. The value function stabilizes only when it is consistent with the current policy, and the policy stabilizes only when it is greedy with respect to the current value function. Thus, both processes stabilize only when a policy has been found that is greedy with respect to its own evaluation function. This implies that the Bellman optimality equation (4.1) holds, and thus that the policy and the value function are optimal. 
>  容易看出，如果策略评估和策略改进过程都趋于稳定，即不再变化，则价值函数和策略必须是最优的
>  策略评估过程仅在价值函数和当前策略一致时稳定，而策略改进仅在当前策略是相对于价值贪心的时候稳定，因此，两个过程仅在策略是相对于其真实价值函数贪心时稳定，这一条件实际上意味着 Bellman 最优方程成立，故此时策略和价值函数都是最优的

The evaluation and improvement processes in GPI can be viewed as both competing and cooperating. They compete in the sense that they pull in opposing directions. Making the policy greedy with respect to the value function typically makes the value function incorrect for the changed policy, and making the value function consistent with the policy typically causes that policy no longer to be greedy. In the long run, however, these two processes interact to find a single joint solution: the optimal value function and an optimal policy. 
>  广义策略迭代中的评估和改进过程可以视作即竞争又合作
>  从某种意义上说，它们是相互对立的。使策略相对于价值函数变得贪婪通常会使价值函数对于改变后的策略变得不准确，而使价值函数与策略一致通常会导致该策略不再具有贪婪性。
>  然而，从长远来看，这两个过程相互作用以找到一个联合解决方案：最优价值函数和最优策略。

One might also think of the interaction between the evaluation and improvement processes in GPI in terms of two constraints or goals—for example, as two lines in two-dimensional space: 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/6658a5d9925f65aa919223b8bd8781a6151e38e68c06a53664b2680257aae27c.jpg) 

Although the real geometry is much more complicated than this, the diagram suggests what happens in the real case. Each process drives the value function or policy toward one of the lines representing a solution to one of the two goals. The goals interact because the two lines are not orthogonal. Driving directly toward one goal causes some movement away from the other goal. Inevitably, however, the joint process is brought closer to the overall goal of optimality. The arrows in this diagram correspond to the behavior of policy iteration in that each takes the system all the way to achieving one of the two goals completely. In GPI one could also take smaller, incomplete steps toward each goal. In either case, the two processes together achieve the overall goal of optimality even though neither is attempting to achieve it directly. 

>  可以将GPI中的评估和改进过程之间的交互视为两个约束或目标——例如，二维空间中的两条线
>  尽管实际情况要复杂得多，但该图示说明了实际发生的情况。每个过程都会使价值函数或策略朝向代表其中一个目标的某条线移动。由于这两条线不是正交的，因此这些目标会相互影响。直接朝一个目标移动会导致在某种程度上远离另一个目标。然而，最终联合过程会更接近整体优化的目标。
>  该图中的箭头对应于策略迭代的行为，即每次迭代都会使系统完全实现其中一个目标。在 GPI 中，也可以朝着每个目标采取较小的、不完全的步骤。无论哪种情况，两个过程共同实现了整体的优化目标，尽管它们都没有直接尝试实现这一目标。

## 4.7 Efficiency of Dynamic Programming 
DP may not be practical for very large problems, but compared with other methods for solving MDPs, DP methods are actually quite efficient. If we ignore a few technical details, then the (worst case) time DP methods take to find an optimal policy is polynomial in the number of states and actions. If $n$ and $m$ denote the number of states and actions, this means that a DP method takes a number of computational operations that is less than some polynomial function of $n$ and $m$ . A DP method is guaranteed to find an optimal policy in polynomial time even though the total number of (deterministic) policies is $m^{n}$ . In this sense, DP is exponentially faster than any direct search in policy space could be, because direct search would have to exhaustively examine each policy to provide the same guarantee. Linear programming methods can also be used to solve MDPs, and in some cases their worst-case convergence guarantees are better than those of DP methods. But linear programming methods become impractical at a much smaller number of states than do DP methods (by a factor of about 100). For the largest problems, only DP methods are feasible. 

DP is sometimes thought to be of limited applicability because of the curse of dimensionality (Bellman, 1957a), the fact that the number of states often grows exponentially with the number of state variables. Large state sets do create difficulties, but these are inherent difficulties of the problem, not of DP as a solution method. In fact, DP is comparatively better suited to handling large state spaces than competing methods such as direct search and linear programming. 

In practice, DP methods can be used with today’s computers to solve MDPs with millions of states. Both policy iteration and value iteration are widely used, and it is not clear which, if either, is better in general. In practice, these methods usually converge much faster than their theoretical worst-case run times, particularly if they are started with good initial value functions or policies. 

On problems with large state spaces, asynchronous DP methods are often preferred. To complete even one sweep of a synchronous method requires computation and memory for every state. For some problems, even this much memory and computation is impractical, yet the problem is still potentially solvable because only a relatively few states occur along optimal solution trajectories. Asynchronous methods and other variations of GPI can be applied in such cases and may find good or optimal policies much faster than synchronous methods can. 
## 4.8 Summary 
In this chapter we have become familiar with the basic ideas and algorithms of dynamic programming as they relate to solving finite MDPs. Policy evaluation refers to the (typically) iterative computation of the value functions for a given policy. Policy improvement refers to the computation of an improved policy given the value function for that policy. Putting these two computations together, we obtain policy iteration and value iteration, the two most popular DP methods. Either of these can be used to reliably compute optimal policies and value functions for finite MDPs given complete knowledge of the MDP. 

Classical DP methods operate in sweeps through the state set, performing a full backup operation on each state. Each backup updates the value of one state based on the values of all possible successor states and their probabilities of occurring. Full backups are closely related to Bellman equations: they are little more than these equations turned into assignment statements. When the backups no longer result in any changes in value, convergence has occurred to values that satisfy the corresponding Bellman equation. Just as there are four primary value functions ( $v_{\pi}$ , $v_{*}$ , $q_{\pi}$ , and $q_{*}$ ), there are four corresponding Bellman equations and four corresponding full backups. An intuitive view of the operation of backups is given by backup diagrams. 

Insight into DP methods and, in fact, into almost all reinforcement learning methods, can be gained by viewing them as generalized policy iteration (GPI). GPI is the general idea of two interacting processes revolving around an approximate policy and an approximate value function. One process takes the policy as given and performs some form of policy evaluation, changing the value function to be more like the true value function for the policy. The other process takes the value function as given and performs some form of policy improvement, changing the policy to make it better, assuming that the value function is its value function. Although each process changes the basis for the other, overall they work together to find a joint solution: a policy and value function that are unchanged by either process and, consequently, are optimal. In some cases, GPI can be proved to converge, most notably for the classical DP methods that we have presented in this chapter. In other cases convergence has not been proved, but still the idea of GPI improves our understanding of the methods. 

It is not necessary to perform DP methods in complete sweeps through the state set. Asynchronous $D P$ methods are in-place iterative methods that back up states in an arbitrary order, perhaps stochastically determined and using out-of-date information. Many of these methods can be viewed as fine-grained forms of GPI. 

Finally, we note one last special property of DP methods. All of them update estimates of the values of states based on estimates of the values of successor states. That is, they update estimates on the basis of other estimates. We call this general idea bootstrapping. Many reinforcement learning methods perform bootstrapping, even those that do not require, as DP requires, a complete and accurate model of the environment. In the next chapter we explore reinforcement learning methods that do not require a model and do not bootstrap. In the chapter after that we explore methods that do not require a model but do bootstrap. These key features and properties are separable, yet can be mixed in interesting combinations. 

# 5 Monte Carlo Methods 
In this chapter we consider our first learning methods for estimating value functions and discovering optimal policies. Unlike the previous chapter, here we do not assume complete knowledge of the environment. Monte Carlo methods require only experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. 
>  我们开始考虑估计价值函数并发现最优策略的学习方法
>  和之前的章节不同的时，此时我们不再假设我们知道环境动态，MC 方法仅需要经验——从实际或模拟与环境交互中得到的状态、动作和奖励样本序列

Learning from actual experience is striking because it requires no prior knowledge of the environment’s dynamics, yet can still attain optimal behavior. Learning from simulated experience is also powerful. Although a model is required, the model need only generate sample transitions, not the complete probability distributions of all possible transitions that is required for dynamic programming (DP). In surprisingly many cases it is easy to generate experience sampled according to the desired probability distributions, but infeasible to obtain the distributions in explicit form. 
>  从实际经验中学习是非常引人注目的，因为它不需要任何关于环境动态的先验知识，但仍可以达到最优行为。从模拟经验中学习也非常强大。尽管需要一个模型，但该模型只需要生成样本转换，而不是动态规划（DP）所需的完整概率分布。
>  要知道，在许多的情况下，根据所需概率分布生成经验是很容易的，但以显式形式获得这些分布则是不可行的。

Monte Carlo methods are ways of solving the reinforcement learning problem based on averaging sample returns. To ensure that well-defined returns are available, here we define Monte Carlo methods only for episodic tasks. That is, we assume experience is divided into episodes, and that all episodes eventually terminate no matter what actions are selected. Only on the completion of an episode are value estimates and policies changed. Monte Carlo methods can thus be incremental in an episode-by-episode sense, but not in a step-by-step (online) sense. The term “Monte Carlo” is often used more broadly for any estimation method whose operation involves a significant random component. Here we use it specifically for methods based on averaging complete returns (as opposed to methods that learn from partial returns, considered in the next chapter). 
>  为了确保可以获得定义明确的回报，我们在这里仅为回合制任务定义 MC 方法，即我们假设经验按照回合划分，并且所有的回合最终都会终止。只有在一个回合终止后，价值估计和策略才会进行更新
>  进而，MC 方法可以执行逐回合的更新，但不是逐步的更新 (online)
>  在这里，MC 方法特指基于完成回报的平均值进行估计的方法 (和下一章的基于部分回报的方法比较)

Monte Carlo methods sample and average returns for each state–action pair much like the bandit methods we explored in Chapter 2 sample and average rewards for each action. The main difference is that now there are multiple states, each acting like a different bandit problem (like an associative-search or contextual bandit) and that the different bandit problems are interrelated. That is, the return after taking an action in one state depends on the actions taken in later states in the same episode. Because all the action selections are undergoing learning, the problem becomes nonstationary from the point of view of the earlier state. 
>  MC 方法为每个状态-动作对进行采样，并且计算平均回报
>  MC 方法中，在某个状态执行一个动作之后得到的回报还依赖于在同一回合中之后的状态下采取的动作，从早期状态的角度来看，这是一个不平稳的过程 (方差大)

To handle the nonstationarity, we adapt the idea of general policy iteration (GPI) developed in Chapter 4 for DP. Whereas there we computed value functions from knowledge of the MDP, here we learn value functions from sample returns with the MDP. The value functions and corresponding policies still interact to attain optimality in essentially the same way (GPI). As in the DP chapter, first we consider the prediction problem (the computation of $v_{\pi}$ and $q_{\pi}$ for a fixed arbitrary policy $\pi$ ) then policy improvement, and, finally, the control problem and its solution by GPI. Each of these ideas taken from DP is extended to the Monte Carlo case in which only sample experience is available. 
>  为了处理不平稳性，我们借鉴广义策略迭代的思想。在 GPI 中，我们通过已知的 MDP 来计算价值函数，而在这里，我们则通过 MDP 的样本回报来学习价值函数。价值函数和相应的策略仍然以基本相同的方式互动以达到最优。
>  与 DP 章节一样，我们首先考虑预测问题（即计算固定任意策略π下的 $v_{\pi}$ 和 $q_{\pi}$），然后是策略改进，最后是控制问题及其通过 GPI 的解决方案。
>  我们在在蒙特卡洛情况下 (即只有样本经验可用时) 扩展这些来自 DP 的想法

## 5.1 Monte Carlo Prediction 
We begin by considering Monte Carlo methods for learning the state-value function for a given policy. Recall that the value of a state is the expected return—expected cumulative future discounted reward—starting from that state. An obvious way to estimate it from experience, then, is simply to average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. This idea underlies all Monte Carlo methods. 
>  考虑为给定策略使用 MC 方法学习状态价值函数
>  回想一下，一个状态的价值是从该状态开始的期望回报——即预期的未来折扣奖励之和。那么，从经验中估计它的明显方法就是简单地对访问该状态后观察到的回报进行平均。随着更多的回报被观察到，这个平均值应该会收敛到期望值。这一想法构成了所有蒙特卡罗方法的基础。

In particular, suppose we wish to estimate $v_{\pi}(s)$ , the value of a state $s$ under policy $\pi$ , given a set of episodes obtained by following $\pi$ and passing through $s$ . Each occurrence of state $s$ in an episode is called a visit to $s$ . Of course, $s$ may be visited multiple times in the same episode; let us call the first time it is visited in an episode the first visit to $s$ . The first-visit MC method estimates $v_{\pi}(s)$ as the average of the returns following first visits to $s$ , whereas the every-visit MC method averages the returns following all visits to $s$ . 
>  给定一系列回合记录，这些回合是遵循策略 $\pi$ 得到的，我们此时希望估计 $v_\pi(s)$
>  我们称一个回合中每次 $s$ 的出现为对 $s$ 的一次访问，回合中第一次 $s$ 的出现就是对 $s$ 的第一次访问
>  首次访问 MC 方法使用对 $s$ 的首次访问后的平均奖励来估计 $v_\pi(s)$，每次访问 MC 方法使用对所有访问 $s$ 的平均奖励

These two Monte Carlo (MC) methods are very similar but have slightly different theoretical properties. First-visit MC has been most widely studied, dating back to the 1940s, and is the one we focus on in this chapter. Every-visit MC extends more naturally to function approximation and eligibility traces, as discussed in Chapters 9 and 7. First-visit MC is shown in procedural form in Figure 5.1. 
>  这两个 MC 方法在理论性质上有细微差异，首次访问 MC 方法自 20 世纪 40 年代以来就被广泛研究，并且是本章中我们关注的方法。每次访问 MC 方法更自然地扩展到函数逼近和资格迹。
>  首次访问 MC 方法的形式如图 5.1 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/f45243eb69f213f07a09acac2a7e0ed01f23ea6606f1d417fc0f6390f31b18fb.jpg) 

Figure 5.1: The first-visit MC method for estimating $v_{\pi}$ . Note that we use a capital letter $V$ for the approximate value function because, after initialization, it soon becomes a random variable. 

Both first-visit MC and every-visit MC converge to $v_{\pi}(s)$ as the number of visits (or first visits) to $s$ goes to infinity. This is easy to see for the case of first-visit MC. In this case each return is an independent, identically distributed estimate of $v_{\pi}(s)$ with finite variance. By the law of large numbers the sequence of averages of these estimates converges to their expected value. Each average is itself an unbiased estimate, and the standard deviation of its error falls as $1/\sqrt{n}$ , where $n$ is the number of returns averaged. Every-visit MC is less straightforward, but its estimates also converge asymptotically to $v_{\pi}(s)$ (Singh and Sutton, 1996). 
>  首次访问蒙特卡洛和每次访问蒙特卡洛当状态 $s$ 的访问次数（或首次访问次数）趋向于无穷大时，都会收敛到 $v_{\pi}(s)$。
>  对于首次访问蒙特卡洛来说，这一点很容易理解。在这种情况下，每个回报都是 $v_{\pi}(s)$ 的一个独立且同分布的估计，具有有限的方差。根据大数定律，这些估计值的平均序列会收敛到它们的期望值。每个平均值本身也是一个无偏估计，其误差的标准偏差随 $n$ 的增加而减少，其中 $n$ 是样本数量。
>  每次访问蒙特卡洛则不那么直接，但它的估计值最终也会收敛到 $v_{\pi}(s)$（Singh 和 Sutton，1996）。

The use of Monte Carlo methods is best illustrated through an example. 

Example 5.1: Blackjack The object of the popular casino card game of blackjack is to obtain cards the sum of whose numerical values is as great as possible without exceeding 21. All face cards count as 10, and an ace can count either as 1 or as 11. We consider the version in which each player competes independently against the dealer. The game begins with two cards dealt to both dealer and player. One of the dealer’s cards is face up and the other is face down. If the player has 21 immediately (an ace and a 10-card), it is called a natural. He then wins unless the dealer also has a natural, in which case the game is a draw. If the player does not have a natural, then he can request additional cards, one by one $(h i t s)$ , until he either stops $(s t i c k s)$ or exceeds 21 (goes bust). If he goes bust, he loses; if he sticks, then it becomes the dealer’s turn. The dealer hits or sticks according to a fixed strategy without choice: he sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes bust, then the player wins; otherwise, the outcome—win, lose, or draw—is determined by whose final sum is closer to 21. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/8bcf013cb2ce2d0ac9fc986c41db7f490be813b32f2c5e9754a1f1555477c19a.jpg) 
Figure 5.2: Approximate state-value functions for the blackjack policy that sticks only on 20 or 21, computed by Monte Carlo policy evaluation. 

Playing blackjack is naturally formulated as an episodic finite MDP. Each game of blackjack is an episode. Rewards of $+1$ , $-1$ , and 0 are given for winning, losing, and drawing, respectively. All rewards within a game are zero, and we do not discount ( $\gamma=1$ ); therefore these terminal rewards are also the returns. The player’s actions are to hit or to stick. The states depend on the player’s cards and the dealer’s showing card. We assume that cards are dealt from an infinite deck (i.e., with replacement) so that there is no advantage to keeping track of the cards already dealt. If the player holds an ace that he could count as 11 without going bust, then the ace is said to be usable. In this case it is always counted as 11 because counting it as 1 would make the sum 11 or less, in which case there is no decision to be made because, obviously, the player should always hit. Thus, the player makes decisions on the basis of three variables: his current sum (12–21), the dealer’s one showing card (ace–10), and whether or not he holds a usable ace. This makes for a total of 200 states. 

Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits. To find the state-value function for this policy by a Monte Carlo approach, one simulates many blackjack games using the policy and averages the returns following each state. Note that in this task the same state never recurs within one episode, so there is no difference between first-visit and every-visit MC methods. In this way, we obtained the estimates of the state-value function shown in Figure 5.2. The estimates for states with a usable ace are less certain and less regular because these states are less common. In any event, after 500,000 games the value function is very well approximated. 

Although we have complete knowledge of the environment in this task, it would not be easy to apply DP methods to compute the value function. DP methods require the distribution of next events—in particular, they require the quantities $p(s^{\prime},r|s,a)$ —and it is not easy to determine these for blackjack. For example, suppose the player’s sum is 14 and he chooses to stick. What is his expected reward as a function of the dealer’s showing card? All of these expected rewards and transition probabilities must be computed before DP can be applied, and such computations are often complex and error-prone. In contrast, generating the sample games required by Monte Carlo methods is easy. This is the case surprisingly often; the ability of Monte Carlo methods to work with sample episodes alone can be a significant advantage even when one has complete knowledge of the environment’s dynamics. ■ 

Can we generalize the idea of backup diagrams to Monte Carlo algorithms? The general idea of a backup diagram is to show at the top the root node to be updated and to show below all the transitions and leaf nodes whose rewards and estimated values contribute to the update. For Monte Carlo estimation of $v_{\pi}$ , the root is a state node, and below it is the entire trajectory of transitions along a particular single episode, ending at the terminal state, as in Figure 5.3. Whereas the DP diagram (Figure 3.4a) shows all possible transitions, the Monte Carlo diagram shows only those sampled on the one episode. Whereas the DP diagram includes only one-step transitions, the Monte Carlo diagram goes all the way to the end of the episode. These differences in the diagrams accurately reflect the fundamental differences between the algorithms. 
>  回溯更新图的基本思想是顶部为要更新的根节点，以树状展示所有的转移和叶节点，这些节点的奖励和估计价值对根节点的更新有贡献
>  对于 $v_\pi$ 的 MC 估计来说，根节点是状态节点，下面是一次回合的完整轨迹直到终止状态如图5.3所示。

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/e8557c789ffd07662c5098933a3b829db540c363e45a7c40fae4f65298a935b5.jpg) 

Figure 5.3: The backup diagram for Monte Carlo estimation of $v_{\pi}$ . 

An important fact about Monte Carlo methods is that the estimates for each state are independent. The estimate for one state does not build upon the estimate of any other state, as is the case in DP. In other words, Monte Carlo methods do not bootstrap as we defined it in the previous chapter. 
>  关于蒙特卡洛方法的一个重要事实是，每个状态的估计是独立的。一个状态的估计不会像在 DP 中那样基于任何其他状态的估计。换句话说，蒙特卡洛方法不像我们在前一章定义的那样进行自举。

In particular, note that the computational expense of estimating the value of a single state is independent of the number of states. This can make Monte Carlo methods particularly attractive when one requires the value of only one or a subset of states. One can generate many sample episodes starting from the states of interest, averaging returns from only these states ignoring all others. This is a third advantage Monte Carlo methods can have over DP methods (after the ability to learn from actual experience and from simulated experience). 
>  特别注意，估计单个状态价值的计算开销与状态数量无关。这使得蒙特卡洛方法在只需要一个或一组状态的价值时特别有吸引力。可以从感兴趣的那些状态生成许多样本回合，仅从这些状态计算回报并忽略所有其他状态。
>  这是蒙特卡洛方法相对于 DP 方法的第三个优势（除了能够从实际经验和模拟经验中学习之外）。

Example 5.2: Soap Bubble 
Suppose a wire frame forming a closed loop is dunked in soapy water to form a soap surface or bubble conforming at its edges to the wire frame. If the geometry of the wire frame is irregular but known, how can you compute the shape of the surface? The shape has the property that the total force on each point exerted by neighboring points is zero (or else the shape would change). This means that the surface’s height at any point is the average of its heights at points in a small circle around that point. In addition, the surface must meet at its boundaries with the wire frame. The usual approach to problems of this kind is to put a grid over the area covered by the surface and solve for its height at the grid points by an iterative computation. Grid points at the boundary are forced to the wire frame, and all others are adjusted toward the average of the heights of their four nearest neighbors. This process then iterates, much like DP’s iterative policy evaluation, and ultimately converges to a close approximation to the desired surface. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/040944d4b51f2661fe443a08ce3c7f0f6d7eab9a1a9235baf74521a1903a2e6f.jpg) 
A bubble on a wire loop 

This is similar to the kind of problem for which Monte Carlo methods were originally designed. Instead of the iterative computation described above, imagine standing on the surface and taking a random walk, stepping randomly from grid point to neighboring grid point, with equal probability, until you reach the boundary. It turns out that the expected value of the height at the boundary is a close approximation to the height of the desired surface at the starting point (in fact, it is exactly the value computed by the iterative method described above). Thus, one can closely approximate the height of the surface at a point by simply averaging the boundary heights of many walks started at the point. If one is interested in only the value at one point, or any fixed small set of points, then this Monte Carlo method can be far more efficient than the iterative method based on local consistency. ■ 

## 5.2 Monte Carlo Estimation of Action Values 
If a model is not available, then it is particularly useful to estimate action values (the values of state–action pairs) rather than state values. With a model, state values alone are sufficient to determine a policy; one simply looks ahead one step and chooses whichever action leads to the best combination of reward and next state, as we did in the chapter on DP. Without a model, however, state values alone are not sufficient. One must explicitly estimate the value of each action in order for the values to be useful in suggesting a policy. Thus, one of our primary goals for Monte Carlo methods is to estimate $q_{*}$ . To achieve this, we first consider the policy evaluation problem for action values. 
>  如果没有模型 (没有环境动态)，则估计动作价值比估计状态价值更加重要
>  有模型时，状态价值足以用于决定一个策略，我们只需要向前看一步，选择能够带来最大回报的动作
>  没有模型时，必须明确估计每个动作的值
>  因此 MC 方法的主要目标其实是估计 $q_*$，为此，我们需要考虑动作价值函数的策略评估问题

The policy evaluation problem for action values is to estimate $q_{\pi}(s,a)$ , the expected return when starting in state $s$ , taking action $a$ , and thereafter following policy $\pi$ . The Monte Carlo methods for this are essentially the same as just presented for state values, except now we talk about visits to a state–action pair rather than to a state. 
>  动作价值的策略评估问题即估计策略 $\pi$ 的动作价值函数 $q_\pi(s, a)$，它表示从 $s$ 开始，执行 $a$，然后遵循 $\pi$ 能得到的期望奖励
>  MC 方法对该函数的评估本质还是采样，求均值，此时的样本是状态-动作对，而不仅仅是状态

A state–action pair $s,a$ is said to be visited in an episode if ever the state $s$ is visited and action $a$ is taken in it. The every-visit MC method estimates the value of a state–action pair as the average of the returns that have followed visits all the visits to it. The first-visit MC method averages the returns following the first time in each episode that the state was visited and the action was selected. These methods converge quadratically, as before, to the true expected values as the number of visits to each state–action pair approaches infinity. 
>  first-visit MC 方法使用每个回合首次对 $(s, a)$ 的访问得到的回报的均值作为估计，every-visit MC 方法使用所有对 $(s, a)$ 的访问得到的回报的均值作为估计
>  随着对 $(s, a)$ 的访问次数趋近于无穷大，估计值二次收敛到真实值

The only complication is that many state–action pairs may never be visited. If $\pi$ is a deterministic policy, then in following $\pi$ one will observe returns only for one of the actions from each state. With no returns to average, the Monte Carlo estimates of the other actions will not improve with experience. This is a serious problem because the purpose of learning action values is to help in choosing among the actions available in each state. To compare alternatives we need to estimate the value of all the actions from each state, not just the one we currently favor. 
>  一个问题是许多 $s, a$ 对可能永远不会被访问
>  如果 $\pi$ 是确定性策略，则遵循 $\pi$，对于每个状态 $s$，我们只能观察到它和一个动作 $a$ 得到的回报

This is the general problem of maintaining exploration, as discussed in the context of the $n$ -armed bandit problem in Chapter 2. For policy evaluation to work for action values, we must assure continual exploration. One way to do this is by specifying that the episodes start in a state–action pair, and that every pair has a nonzero probability of being selected as the start. This guarantees that all state–action pairs will be visited an infinite number of times in the limit of an infinite number of episodes. We call this the assumption of exploring starts. 
>  该问题属于 maintain exploration 的范围，我们必须确保持续的探索，以估计动作价值函数
>  一种方法是指定回合直接从状态-动作对开始，在开始时每个状态-动作对都有非零的概率被选中，在回合数趋近于无穷时，可以确保所有状态-动作对被访问
>  我们称之为探索开始假设

The assumption of exploring starts is sometimes useful, but of course it cannot be relied upon in general, particularly when learning directly from actual interaction with an environment. In that case the starting conditions are unlikely to be so helpful. The most common alternative approach to assuring that all state–action pairs are encountered is to consider only policies that are stochastic with a nonzero probability of selecting all actions in each state. We discuss two important variants of this approach in later sections. For now, we retain the assumption of exploring starts and complete the presentation of a full Monte Carlo control method. 
>  但在从和环境的真实交互中学习时，探索开始假设则没用，因为难以遍历所有起始状态
>  此时，最常考虑的方法是使用一个随机策略，它在每个状态选择每个动作都有非零的概率

## 5.3 Monte Carlo Control 
We are now ready to consider how Monte Carlo estimation can be used in control, that is, to approximate optimal policies. The overall idea is to proceed according to the same pattern as in the DP chapter, that is, according to the idea of generalized policy iteration (GPI). In GPI one maintains both an approximate policy and an approximate value function. The value function is repeatedly altered to more closely approximate the value function for the current policy, and the policy is repeatedly improved with respect to the current value function: 
>  我们考虑用 MC 估计最优策略
>  总体的思路和 DP 章节类似，使用广义策略迭代的思想，维护一个近似策略和近似价值函数，价值函数在策略评估中不断迭代，以接近当前策略的真实价值函数，策略相对于当前价值函数不断改进

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/0a4ec9a2da660666c698d56a07ac6f1913e8fce5b71c0196ce73073ba3c50d97.jpg) 

These two kinds of changes work against each other to some extent, as each creates a moving target for the other, but together they cause both policy and value function to approach optimality. 
>  这两个目标在某种程度上对抗，但整体上看它们会使得策略和价值函数都趋向于最优

To begin, let us consider a Monte Carlo version of classical policy iteration. In this method, we perform alternating complete steps of policy evaluation and policy improvement, beginning with an arbitrary policy $\pi_{0}$ and ending with the optimal policy and optimal action-value function: 

$$
\pi_{0}\stackrel{\mathrm{\tiny~E}}{\longrightarrow}q_{\pi_{0}}\stackrel{\mathrm{\tiny~I}}{\longrightarrow}\pi_{1}\stackrel{\mathrm{\tiny~E}}{\longrightarrow}q_{\pi_{1}}\stackrel{\mathrm{\tiny~I}}{\longrightarrow}\pi_{2}\stackrel{\mathrm{\tiny~E}}{\longrightarrow}\cdot\cdot\cdot\stackrel{\mathrm{\tiny~I}}{\longrightarrow}\pi_{*}\stackrel{\mathrm{\tiny~E}}{\longrightarrow}q_{*},
$$ 
where $\xrightarrow{\textrm{E}}$ denotes a complete policy evaluation and $\xrightarrow{\mathrm{~I~}}$ denotes a complete policy improvement. Policy evaluation is done exactly as described in the preceding section. Many episodes are experienced, with the approximate action-value function approaching the true function asymptotically. 
>  策略迭代中，策略评估使用 MC 执行

For the moment, let us assume that we do indeed observe an infinite number of episodes and that, in addition, the episodes are generated with exploring starts. Under these assumptions, the Monte Carlo methods will compute each $q_{\pi_{k}}$ exactly, for arbitrary $\pi_{k}$ . 
>  我们假设有无限个回合的样本，这些回合基于探索开始假设，则 MC 方法将准确地为任意 $\pi_k$ 计算出 $q_{\pi_k}$

Policy improvement is done by making the policy greedy with respect to the current value function. In this case we have an action-value function, and therefore no model is needed to construct the greedy policy. For any action-value function $q$ , the corresponding greedy policy is the one that, for each $s\in\mathcal{S}$ , deterministically chooses an action with maximal action-value: 

$$
\pi(s)=\arg\operatorname*{max}_{a}q(s,a).\tag{5.1}
$$ 
>  策略改进让策略相对于当前价值函数贪心，注意此时价值函数是动作价值函数，故此时的策略改进不需要环境动态，(回忆起如果价值函数是状态价值函数时，我们需要用环境动态先计算出动作价值函数，再进行选择，此时可以认为 MC 估计出的动作价值函数中已经包含了环境动态的信息) 我们直接根据 $q(s, a)$ 确定性地选择出 $\arg\max_a q(s, a)$

Policy improvement then can be done by constructing each $\pi_{k+1}$ as the greedy policy with respect to $q_{\pi_{k}}$ . The policy improvement theorem (Section 4.2) then applies to $\pi_{k}$ and $\pi_{k+1}$ because, for all $s\in\mathcal{S}$ , 

$$
\begin{align}
q_{\pi_k}(s, \pi_{k+1}(s)) &= q_{\pi_k}(s, \arg\max_a q_{\pi_k}(s, a))\\
&=\max_a q_{\pi_k}(s, a)\\
&\ge q_{\pi_k}(s, \pi_k(s))\\
&=v_{\pi_k}(s)
\end{align}
$$

>  此时，策略改进定理仍然是成立的

As we discussed in the previous chapter, the theorem assures us that each $\pi_{k+1}$ is uniformly better than $\pi_{k}$ , or just as good as $\pi_{k}$ , in which case they are both optimal policies. This in turn assures us that the overall process converges to the optimal policy and optimal value function. In this way Monte Carlo methods can be used to find optimal policies given only sample episodes and no other knowledge of the environment’s dynamics. 
>  该定理确保了 $\pi_{k+1}$ 会严格优于 $\pi_k$，除非已经是最优，因此整个策略迭代过程也保证是收敛的
>  因此，在没有对环境动态的知识下，仅给定回合样本，MC 方法也可以用于找到最优策略
>  (和 DP 的差异仅在于策略评估环节中，DP 利用环境动态计算状态价值函数，进而计算动作价值函数，MC 则使用样本直接估计动作价值函数)

We made two unlikely assumptions above in order to easily obtain this guarantee of convergence for the Monte Carlo method. One was that the episodes have exploring starts, and the other was that policy evaluation could be done with an infinite number of episodes. To obtain a practical algorithm we will have to remove both assumptions. We postpone consideration of the first assumption until later in this chapter. 
>  但我们做了两个假设，其一是回合都具有探索性起始，其二是策略评估可以用无限的回合样本执行
>  实践中需要移除这两个假设

For now we focus on the assumption that policy evaluation operates on an infinite number of episodes. This assumption is relatively easy to remove. In fact, the same issue arises even in classical DP methods such as iterative policy evaluation, which also converge only asymptotically to the true value function. 
>  现在我们假设策略评估仅涉及有限数量的回合
>  实际上，经典 DP 中的迭代式策略评估也有类似的情况，回忆起 DP 中的策略评估是在渐进情况下收敛到真实值

In both DP and Monte Carlo cases there are two ways to solve the problem. One is to hold firm to the idea of approximating $q_{\pi_{k}}$ in each policy evaluation. Measurements and assumptions are made to obtain bounds on the magnitude and probability of error in the estimates, and then sufficient steps are taken during each policy evaluation to assure that these bounds are sufficiently small. This approach can probably be made completely satisfactory in the sense of guaranteeing correct convergence up to some level of approximation. However, it is also likely to require far too many episodes to be useful in practice on any but the smallest problems. 
>  这样的问题有两种方式解决，一种是在每一步中尽可能近似，使得误差足够小，这种方法可以在某种意义上确保收敛到一定的近似水平，但每一步需要的回合数太多，不现实

The second approach to avoiding the infinite number of episodes nominally required for policy evaluation is to forgo trying to complete policy evaluation before returning to policy improvement. On each evaluation step we move the value function toward $q_{\pi_{k}}$ , but we do not expect to actually get close except over many steps. We used this idea when we first introduced the idea of GPI in Section 4.6. One extreme form of the idea is value iteration, in which only one iteration of iterative policy evaluation is performed between each step of policy improvement. The in-place version of value iteration is even more extreme; there we alternate between improvement and evaluation steps for single states. 
>  第二种是不尽可能近似，执行一定步数后就进行策略改进
>  这类方法的一个极限情况就是价值迭代，其中迭代式策略评估仅执行一次迭代，原地的价值迭代是更极端的版本，我们在其中的每一步都在策略改进和评估之间切换

For Monte Carlo policy evaluation it is natural to alternate between evaluation and improvement on an episode-by-episode basis. After each episode, the observed returns are used for policy evaluation, and then the policy is improved at all the states visited in the episode. A complete simple algorithm along these lines is given in Figure 5.4. We call this algorithm Monte Carlo $E S$ , for Monte Carlo with Exploring Starts. 
>  MC 策略评估中，可以以回合为单位，切换策略评估和策略改进，即每次策略评估仅使用一个回合的数据，对该回合中访问的所有状态执行策略评估，然后对该回合中访问的所有状态执行策略改进
>  该算法称为 MC ES，即具有探索起点的 MC 方法

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/ebd59b706cc7d8e1ae02c0b95ad319dad4dea9bffae343b62fc1823a18f32b21.jpg) 

Figure 5.4: Monte Carlo ES: A Monte Carlo control algorithm assuming exploring starts and that episodes always terminate for all policies. 

In Monte Carlo ES, all the returns for each state–action pair are accumulated and averaged, irrespective of what policy was in force when they were observed. It is easy to see that Monte Carlo ES cannot converge to any suboptimal policy. If it did, then the value function would eventually converge to the value function for that policy, and that in turn would cause the policy to change. Stability is achieved only when both the policy and the value function are optimal. Convergence to this optimal fixed point seems inevitable as the changes to the action-value function decrease over time, but has not yet been formally proved. In our opinion, this is one of the most fundamental open theoretical questions in reinforcement learning (for a partial solution, see Tsitsiklis, 2002). 
>  MC ES 不会收敛到任意次优的策略，如果有，则价值函数会最终收敛到该策略的价值函数，而这会进而让该策略被改进，MC ES 过程仅在策略和价值函数都达到最优时稳定
>  MC ES 到这一固定点的收敛性直观容易看出，因为价值函数的改变将逐渐减小，但尚未被证明

Example 5.3: Solving Blackjack It is straightforward to apply Monte Carlo ES to blackjack. Since the episodes are all simulated games, it is easy to arrange for exploring starts that include all possibilities. In this case one simply picks the dealer’s cards, the player’s sum, and whether or not the player has a usable ace, all at random with equal probability. As the initial policy we use the policy evaluated in the previous blackjack example, that which sticks only on 20 or 21. The initial action-value function can be zero for all state–action pairs. Figure 5.5 shows the optimal policy for blackjack found by Monte Carlo ES. This policy is the same as the “basic” strategy of Thorp (1966) with the sole exception of the leftmost notch in the policy for a usable ace, which is not present in Thorp’s strategy. We are uncertain of the reason for this discrepancy, but confident that what is shown here is indeed the optimal policy for the version of blackjack we have described. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/1e8f8ba32e94330fa12daebe16c26e17939cac58db0b3be3bf04dcef5a6d64f7.jpg) 

Figure 5.5: The optimal policy and state-value function for blackjack, found by Monte Carlo ES (Figure 5.4). The state-value function shown was computed from the action-value function found by Monte Carlo ES. 

## 5.4 Monte Carlo in Control without Exploring Starts 
How can we avoid the unlikely assumption of exploring starts? The only general way to ensure that all actions are selected infinitely often is for the agent to continue to select them. There are two approaches to ensuring this, resulting in what we call on-policy methods and off-policy methods. On-policy methods attempt to evaluate or improve the policy that is used to make decisions, whereas off-policy methods evaluate or improve a policy different from that used to generate the data. The Monte Carlo ES method developed above is an example of an on-policy method. In this section we show how an on-policy Monte Carlo control method can be designed that does not use the unrealistic assumption of exploring starts. Off-policy methods are considered in the next section. 
>  考虑避免探索起点假设
>  为了确保所有动作可能被无限次选择，有两种方法：同策略和异策略
>  同策略方法试图评估和改进用于决策的策略，异策略方法中，生成数据的和评估、改进用于决策的策略则不同
>  MC ES 属于同策略方法

In on-policy control methods the policy is generally soft, meaning that $\pi(a|s)>0$ for all $s\in\mathcal{S}$ and all $a\in\mathcal{A}(s)$ , but gradually shifted closer and closer to a deterministic optimal policy. Many of the methods discussed in Chapter 2 provide mechanisms for this. The on-policy method we present in this section uses $\boldsymbol{\varepsilon}$ -greedy policies, meaning that most of the time they choose an action that has maximal estimated action value, but with probability $\boldsymbol{\varepsilon}$ they instead select an action at random. That is, all nongreedy actions are given the minimal probability of selection, $\frac{\epsilon}{|\mathcal{A}(s)|}$ , and the remaining bulk of the probability, $\begin{array}{r}{1-\varepsilon+\frac{\epsilon}{|\mathcal{A}(s)|}}\end{array}$ , is given to the greedy action. The $\varepsilon$ -greedy policies are examples of $\boldsymbol{\varepsilon}$ -soft policies, defined as policies for which $\pi(a|s)\geq{\frac{\epsilon}{|{\mathcal{A}}(s)|}}$ for all states and actions, for some $\varepsilon>0$ . Among $\varepsilon$ -soft policies, $\varepsilon\cdot$ -greedy policies are in some sense those that are closest to greedy. 
>  同策略控制方法中，策略通常是 soft 的，即对于所有的 $s\in \mathcal S$ 和 $a\in \mathcal A(s)$ 都有 $\pi(a\mid s) > 0$
>  我们在本节讨论 $\epsilon$ -greedy 策略，该策略在大多数时候会选择最大化动作价值函数的动作，但会有 $\epsilon$ 的概率随机选择一个动作，也就是说其他非贪心的动作会有 $\frac {\epsilon}{|\mathcal A(s)|}$ 的概率被选中，贪心动作会有 $1-\epsilon + \frac {\epsilon}{|\mathcal A(s)|}$ 的概率被选中
>  $\epsilon$ -greedy 策略属于 $\epsilon$ soft 策略，$\epsilon$ -soft 策略定义为对于所有状态和任意动作，都有 $\pi(a\mid s)\ge \frac {\epsilon}{|\mathcal A(s)|}$ 的概率被选中 ($\epsilon > 0$)，$\epsilon$ -greedy 是 $\epsilon$ -soft 策略中最接近贪心的策略

The overall idea of on-policy Monte Carlo control is still that of GPI. As in Monte Carlo $\mathrm{ES}$ , we use first-visit MC methods to estimate the action-value function for the current policy. Without the assumption of exploring starts, however, we cannot simply improve the policy by making it greedy with respect to the current value function, because that would prevent further exploration of nongreedy actions. Fortunately, GPI does not require that the policy be taken all the way to a greedy policy, only that it be moved toward a greedy policy. In our on-policy method we will move it only to an $\varepsilon$ -greedy policy. For any $\boldsymbol{\varepsilon}$ -soft policy, $\pi$ , any $\boldsymbol{\varepsilon}$ -greedy policy with respect to $q_{\pi}$ is guaranteed to be better than or equal to $\pi$ . 
>  同策略 MC 的思想仍然是 GPI，我们使用首次访问 MC 方法为当前策略估计动作价值函数，但在没有探索性开始的假设下，我们在策略改进时不能简单让下一个策略对当前价值函数贪心，这会防止对非贪心策略的进一步探索
>  GPI 并不要求策略改进时完全使用贪心，而是可以朝向贪心的方向改进，在同策略方法中，我们在策略改进时将策略改进为 $\epsilon$ -greedy 策略
>  对于任意的 $\epsilon$ -soft 策略 $\pi$，任意相对于 $q_\pi$ 的 $\epsilon$ -greedy 策略都保证不差于 $\pi$

That any $\varepsilon$ -greedy policy with respect to $q_{\pi}$ is an improvement over any $\varepsilon$ -soft policy $\pi$ is assured by the policy improvement theorem. 
>  任意相对于 $q_\pi$ 的 $\epsilon$ -贪心策略相对于任意的 $\epsilon$ -soft 策略 $\pi$ 都保证是其提升，这一结论仍然由策略改进定理给出，证明如下

Let $\pi^{\prime}$ be the $\varepsilon$ -greedy policy. The conditions of the policy improvement theorem apply because for any $s\in\mathcal{S}$ : 

$$
\begin{align}
q_\pi(s, \pi'(s)) &=\sum_a\pi'(a\mid s)q_\pi(s, a)\\
&=\frac {\epsilon}{|\mathcal A(s)|}\sum_a q_\pi(s, a) + (1-\epsilon)\max_a q_\pi(s, a)\\
&\ge\frac {\epsilon}{|\mathcal A(s)|}\sum_a q_\pi(s, a) + (1-\epsilon)\sum_a\frac {\pi(a\mid s) - \frac {\epsilon}{|\mathcal A(s)|}}{1-\epsilon}q_\pi(s, a)\\
\end{align}\tag{5.2}
$$

(the sum is a weighted average with nonnegative weights summing to $1$ , and as such it must be less than or equal to the largest number averaged) 

$$
\begin{align}
&=\frac {\epsilon}{|\mathcal A(s)|}\sum_a q_\pi(s, a) - \frac {\epsilon}{|\mathcal A(s)|}\sum_{a}q_\pi(s, a) + \sum_a \pi(a\mid s)q_\pi(s, a)\\
&=v_\pi(s)
\end{align}
$$

Thus, by the policy improvement theorem, $\pi^{\prime}\geq\pi$ (i.e., $v_{\pi^{\prime}}(s)\geq v_{\pi}(s)$ , for all $s\in{\mathcal{S}}$ ). 
>  根据策略改进定理，我们进而知道对于所有的 $s\in \mathcal S$，都有 $\pi' \ge \pi$

We now prove that equality can hold only when both $\pi^{\prime}$ and $\pi$ are optimal among the $\varepsilon$ -soft policies, that is, when they are better than or equal to all other $\boldsymbol{\varepsilon}$ -soft policies. 
>  我们进而证明只有在 $\pi', \pi$ 都是 $\epsilon$ -soft 策略中最优的那个时，等号成立

Consider a new environment that is just like the original environment, except with the requirement that policies be $\boldsymbol{\varepsilon}$ -soft “moved inside” the environment. The new environment has the same action and state set as the original and behaves as follows. If in state $s$ and taking action $a$ , then with probability $1-\varepsilon$ the new environment behaves exactly like the old environment. With probability $\boldsymbol{\varepsilon}$ it repicks the action at random, with equal probabilities, and then behaves like the old environment with the new, random action. The best one can do in this new environment with general policies is the same as the best one could do in the original environment with $\varepsilon$ -soft policies. Let $\widetilde{v}_{*}$ and $\widetilde{q}_{*}$ denote the optimal value functions for the new environment. Then a policy $\pi$ is optimal among $\boldsymbol{\varepsilon}$ -soft policies if and only if $v_{\pi}=\widetilde{v}_{*}$ . 
>  考虑一个新环境，在该环境中，如果 $s$ 执行了 $a$，则有 $1-\epsilon$ 的概率新环境的行为和旧环境一致，有 $\epsilon$ 的概率新环境重新随机按照均匀分布选择一个动作
>  在新环境中，旧环境最优的策略会变为 $\epsilon$ -soft 的形式
>  令 $\tilde v_*$ 和 $\tilde q_*$ 表示新环境中最优的价值函数，则一个 $\epsilon$ -soft 策略 $\pi$ 当且仅当 $v_\pi = \tilde v_*$ 时，它是最优的

From the definition of $\widetilde{v}_{*}$ we know that it is the unique solution to 

$$\begin{align*}
\tilde{v}_*(s) &= (1-\varepsilon)\max_a \tilde{q}_*(s, a) + \frac{\varepsilon}{|\mathcal{A}(s)|}\sum_a \tilde{q}_*(s, a) \\
&= (1-\varepsilon)\max_a \sum_{s', r} p (s', r | s, a) \Big[r + \gamma \tilde{v}_*(s')\Big] \\
&\quad + \frac{\varepsilon}{|\mathcal{A}(s)|}\sum_a \sum_{s', r} p (s', r | s, a) \Big[r + \gamma \tilde{v}_*(s')\Big].
\end{align*}$$

>  在该环境下，$\tilde v_*(s)$ 和 $\tilde q_*(s, a)$ 满足如上的 Bellman equation

When equality holds and the $\boldsymbol{\varepsilon}$ -soft policy $\pi$ is no longer improved, then we also know, from (5.2), that 

$$\begin{align*}
v_{\pi}(s) &= (1-\varepsilon)\operatorname*{max}_{a}q_{\pi}(s, a) + \frac{\epsilon}{|\mathcal{A}(s)|}\sum_{a}q_{\pi}(s, a) \\
&= (1-\varepsilon)\operatorname*{max}_{a}\sum_{s', r}p (s', r|s, a)\Big[r+\gamma v_{\pi}(s')\Big] \\
&\quad + \frac{\epsilon}{|\mathcal{A}(s)|}\sum_{a}\sum_{s', r}p (s', r|s, a)\Big[r+\gamma v_{\pi}(s')\Big].
\end{align*}$$

However, this equation is the same as the previous one, except for the substitution of $v_{\pi}$ for $\widetilde{v}_{*}$ . Since $\widetilde{v}_{*}$ is the unique solution, it must be that $v_{\pi}=\widetilde{v}_{*}$ . 

>  而如果策略改进过程稳定了，根据 Eq 5.2，$\pi$ 的状态价值函数和动作价值函数之间也满足相同形式的方程
>  因此，策略改进过程的固定点就是最优的 $\epsilon$ -soft 策略

In essence, we have shown in the last few pages that policy iteration works for $\varepsilon$ -soft policies. Using the natural notion of greedy policy for $\boldsymbol{\varepsilon}$ -soft policies, one is assured of improvement on every step, except when the best policy has been found among the $\boldsymbol{\varepsilon}$ -soft policies. 
>  我们上述的讨论就是说明了策略迭代对于 $\epsilon$ -soft 策略也是有效的

This analysis is independent of how the action-value functions are determined at each stage, but it does assume that they are computed exactly. This brings us to roughly the same point as in the previous section. Now we only achieve the best policy among the $\boldsymbol{\varepsilon}$ -soft policies, but on the other hand, we have eliminated the assumption of exploring starts. 
>  分析过程独立于动作价值函数的具体定义，但假设了动作价值函数会被准确计算，该过程能够带来 $\epsilon$ -soft 策略中的最优策略，但帮助我们消除了探索性开始的假设

The complete algorithm is given in Figure 5.6. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/0f7e0adc617635bdd42b15f6c97b3be9c1b1839f780fcc6d7fb0013a3cfea8c3.jpg) 

Figure 5.6: An on-policy first-visit MC control algorithm for $\boldsymbol{\varepsilon}$ -soft policies. 

## 5.5 Off-policy Prediction via Importance Sampling 
So far we have considered methods for estimating the value functions for a policy given an infinite supply of episodes generated using that policy. Suppose now that all we have are episodes generated from a different policy. That is, suppose we wish to estimate $v_{\pi}$ or $q_{\pi}$ , but all we have are episodes following another policy $\mu$ , where $\mu\neq\pi$ . We call $\pi$ the target policy because learning its value function is the target of the learning process, and we call $\mu$ the behavior policy because it is the policy controlling the agent and generating behavior. The overall problem is called off-policy learning because it is learning about a policy given only experience “off” (not following) that policy. 

In order to use episodes from $\mu$ to estimate values for $\pi$ , we must require that every action taken under $\pi$ is also taken, at least occasionally, under $\mu$ . That is, we require that $\pi(a|s)>0$ implies $\mu(a|s)>0$ . This is called the assumption of coverage. It follows from coverage that $\mu$ must be stochastic in states where it is not identical to $\pi$ . The target policy $\pi$ , on the other hand, may be deterministic, and, in fact, this is a case of particular interest. Typically the target policy is the deterministic greedy policy with respect to the current action-value function estimate. This policy we hope becomes a deterministic optimal policy while the behavior policy remains stochastic and more exploratory, for example, an $\varepsilon$ -greedy policy. 

Importance sampling is a general technique for estimating expected values under one distribution given samples from another. We apply this technique to off-policy learning by weighting returns according to the relative probability of their trajectories occurring under the target and behavior policies, called the importance-sampling ratio. Given a starting state $S_{t}$ , the probability of the subsequent state–action trajectory, $A_{t},S_{t+1},A_{t+1},...,S_{T}$ , occurring under any policy $\pi$ is 

$$
\prod_{k=t}^{T-1}\pi(A_{k}|S_{k})p(S_{k+1}|S_{k},A_{k}),
$$ 
where $p$ is the state-transition probability function defined by (3.8). Thus, the relative probability of the trajectory under the target and behavior policies (the importance-sampling ratio) is 

$$
\rho_{t}^{T}=\frac{\prod_{k=t}^{T-1}\pi(A_{k}|S_{k})p(S_{k+1}|S_{k},A_{k})}{\prod_{k=t}^{T-1}\mu(A_{k}|S_{k})p(S_{k+1}|S_{k},A_{k})}=\prod_{k=t}^{T-1}\frac{\pi(A_{k}|S_{k})}{\mu(A_{k}|S_{k})}.
$$ 
Note that although the trajectory probabilities depend on the MDP’s transition probabilities, which are generally unknown, all the transition probabilities cancel and drop out. The importance sampling ratio ends up depending only on the two policies and not at all on the MDP. 
Now we are ready to give a Monte Carto algorithm that uses a batch of observed episodes following policy $\mu$ to estimate $v_{\pi}(s)$ . It is convenient here to number time steps in a way that increases across episode boundaries. That is, if the first episode of the batch ends in a terminal state at time 100, then the next episode begins at time $t=101$ . This enables us to use time-step numbers to refer to particular steps in particular episodes. In particular, we can define the set of all time steps in which state $s$ is visited, denoted $\Im(s)$ . This is for an every-visit method; for a first-visit method, ${\mathfrak{T}}(s)$ would only include time steps that were first visits to $s$ within their episode. Also, let $T(t)$ denote the first time of termination following time $t$ , and $G_{t}$ denote the return after $t$ up through $T(t)$ . Then $\{G_{t}\}_{t\in\mathcal{T}(s)}$ are the returns that pertain to state $s$ , and $\{\rho_{t}^{T(t)}\}_{t\in\mathcal{T}(s)}$ are the corresponding importance-sampling ratios. To estimate $v_{\pi}(s)$ , we simply scale the returns by the ratios and average the results: 
$$
V(s)=\frac{\sum_{t\in\mathbb{T}(s)}\rho_{t}^{T(t)}G_{t}}{|\mathbb{T}(s)|}.
$$ 
When importance sampling is done as a simple average in this way it is called ordinary importance sampling. 
An important alternative is weighted importance sampling, which uses a weighted average, defined as 

$$
V(s)=\frac{\sum_{t\in\mathbb{T}(s)}\rho_{t}^{T(t)}G_{t}}{\sum_{t\in\mathbb{T}(s)}\rho_{t}^{T(t)}},
$$ 
or zero if the denominator is zero. To understand these two varieties of importance sampling, consider their estimates after observing a single return. In the weighted-average estimate, the ratio ρtT(t) for the single return cancels in the numerator and denominator, so that the estimate is equal to the observed return independent of the ratio (assuming the ratio is nonzero). Given that this return was the only one observed, this is a reasonable estimate, but of course its expectation is $v_{\mu}(s)$ rather than $v_{\pi}(s)$ , and in this statistical sense it is biased. In contrast, the simple average (5.4) is always $v_{\pi}(s)$ in expectation (it is unbiased), but it can be extreme. Suppose the ratio were ten, indicating that the trajectory observed is ten times as likely under the target policy as under the behavior policy. In this case the ordinary importance-sampling estimate would be ten times the observed return. That is, it would be quite far from the observed return even though the episode’s trajectory is considered very representative of the target policy. 
Formally, the difference between the two kinds of importance sampling is expressed in their variances. The variance of the ordinary importancesampling estimator is in general unbounded because the variance of the ratios is unbounded, whereas in the weighted estimator the largest weight on any single return is one. In fact, assuming bounded returns, the variance of the weighted importance-sampling estimator converges to zero even if the variance of the ratios themselves is infinite (Precup, Sutton, and Dasgupta 2001). In practice, the weighted estimator usually has dramatically lower variance and is strongly preferred. A complete every-visit MC algorithm for off-policy policy evaluation using weighted importance sampling is given at the end of the next section in Figure 5.9. 

Example 5.4: Off-policy Estimation of a Blackjack State Value 
We applied both ordinary and weighted importance-sampling methods to estimate the value of a single blackjack state from off-policy data. Recall that one of the advantages of Monte Carlo methods is that they can be used to evaluate a single state without forming estimates for any other states. In this example, we evaluated the state in which the dealer is showing a deuce, the sum of the player’s cards is 13, and the player has a usable ace (that is, the player holds an ace and a deuce, or equivalently three aces). The data was generated by starting in this state then choosing to hit or stick at random with equal probability (the behavior policy). The target policy was to stick only on a sum of 20 or 21, as in Example 5.1. The value of this state under the target policy is approximately −0.27726 (this was determined by separately generating one-hundred million episodes using the target policy and averaging their returns). Both off-policy methods closely approximated this value after 1000 off-policy episodes using the random policy. Figure 5.7 shows the mean squared error (estimated from 100 independent runs) for each method as a function of number of episodes. The weighted importance-sampling method has much lower overall error in this example, as is typical in practice. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/20ecff6b36b9bb246243021220824893024a5dc1b49baf224ee43355f79f06f4.jpg) 
Figure 5.7: Weighted importance sampling produces lower error estimates of the value of a single blackjack state from off-policy episodes (see Example 5.4). 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/fd96f5df9e53f368ab3a92b1eb40d3f05cdc66d073a89742863df9966b2aabb6.jpg) 
Figure 5.8: Ordinary importance sampling produces surprisingly unstable estimates on the one-state MDP shown inset (Example 5.5). The correct estimate here is 1, and, even though this is the expected value of a sample return (after importance sampling), the variance of the samples is infinite, and the estimates do not convergence to this value. These results are for off-policy first-visit MC. 

Example 5.5: Infinite Variance 
The estimates of ordinary importance sampling will typically have infinite variance, and thus unsatisfactory convergence properties, whenever the scaled returns have infinite variance—and this can easily happen in off-policy learning when trajectories contain loops. A simple example is shown inset in Figure 5.8. There is only one nonterminal state $s$ and two actions, end and back. The end action causes a deterministic transition to termination, whereas the back action transitions, with probability 0.9, back to $s$ or, with probability 0.1, on to termination. The rewards are $+1$ on the latter transition and otherwise zero. Consider the target policy that always selects back. All episodes under this policy consist of some number (possibly zero) of transitions back to $s$ followed by termination with a reward and return of $+1$ . Thus the value of $s$ under the target policy is thus 1. Suppose we are estimating this value from off-policy data using the behavior policy that selects end and back with equal probability. The lower part of Figure 5.8 shows ten independent runs of the first-visit MC algorithm using ordinary importance sampling. Even after millions of episodes, the estimates fail to converge to the correct value of 1. In contrast, the weighted importance-sampling algorithm would give an estimate of exactly 1 everafter the first episode that was consistent with the target policy (i.e., that ended with the back action). This is clear because that algorithm produces a weighted average of the returns consistent with the target policy, all of which would be exactly 1. 

We can verify that the variance of the importance-sampling-scaled returns is infinite in this example by a simple calculation. The variance of any random variable $X$ is the expected value of the deviation from its mean $X$ , which can be written 
$$
\operatorname{Var}[X]=\operatorname{\mathbb{E}}\left[\left(X-{\bar{X}}\right)^{2}\right]=\operatorname{\mathbb{E}}\left[X^{2}-2X{\bar{X}}+{\bar{X}}^{2}\right]=\operatorname{\mathbb{E}}\left[X^{2}\right]-{\bar{X}}^{2}.
$$ 
Thus, if the mean is finite, as it is in our case, the variance is infinite if and only if the expectation of the square of the random variable is infinite. Thus, we need only show that the expected square of the importance-sampling-scaled return is infinite: 
$$
\mathbb{E}\left[\left(\prod_{t=0}^{T-1}\frac{\pi(A_{t}|S_{t})}{\mu(A_{t}|S_{t})}G_{0}\right)^{2}\right].
$$ 
To compute this expectation, we break it down into cases based on episode length and termination. First note that, for any episode ending with the end action, the importance sampling ratio is zero, because the target policy would never take this action; these episodes thus contribute nothing to the expectation (the quantity in parenthesis will be zero) and can be ignored. We need only consider episodes that involve some number (possibly zero) of back actions that transition back to the nonterminal state, followed by a back action transitioning to termination. All of these episodes have a return of 1, so the $G_{0}$ factor can be ignored. To get the expected square we need only consider each length of episode, multiplying the probability of the episode’s occurrence by the square of its importance-sampling ratio, and add these up: 
(the length 1 episode) 
$$
{\begin{array}{r l}&{={\frac{1}{2}}\cdot0.1\left({\frac{1}{0.5}}\right)^{2}}\ &{\quad+{\frac{1}{2}}\cdot0.9\cdot{\frac{1}{2}}\cdot0.1\left({\frac{1}{0.5}}{\frac{1}{0.5}}\right)^{2}}\ &{\quad+{\frac{1}{2}}\cdot0.9\cdot{\frac{1}{2}}\cdot0.9\cdot{\frac{1}{2}}\cdot0.1\left({\frac{1}{0.5}}{\frac{1}{0.5}}{\frac{1}{0.0}}_{5}^{2}\right)^{2}}\ &{\quad+\dots}\ &{=0.1{\underset{k=0}{\overset{\infty}{\sum}}}0.9^{k}\cdot2^{k}\cdot2}\ &{\quad=0.2{\underset{k=0}{\overset{\infty}{\sum}}}1.8^{k}}\ &{\quad={\underset{\infty}{\overset{\infty}{\sum}}}1.8^{k}}\end{array}}
$$ 
(the length 2 episode) 
(the length 3 episode) 
## 5.6 Incremental Implementation 
Monte Carlo prediction methods can be implemented incrementally, on an episode-by-episode basis, using extensions of the techniques described in Chapter 2. Whereas in Chapter 2 we averaged rewards, in Monte Carlo methods we average returns. In all other respects exactly the same methods as used in Chapter 2 can be used for on-policy Monte Carlo methods. For off-policy Monte Carlo methods, we need to separately consider those that use ordinary importance sampling and those that use weighted importance sampling. 

In ordinary importance sampling, the returns are scaled by the importance sampling ratio ρt (5.3), then simply averaged. For these methods we can again use the incremental methods of Chapter 2, but using the scaled returns in place of the rewards of that chapter. This leaves the case of off-policy methods using weighted importance sampling. Here we have to form a weighted average of the returns, and a slightly different incremental algorithm is required. 

Suppose we have a sequence of returns $G_{1},G_{2},\dots,G_{n-1}$ , all starting in the same state and each with a corresponding random weight $W_{i}$ (e.g., $W_{i}=\rho_{t}^{T(t)}$ ρT (t)). We wish to form the estimate 

$$
V_{n}={\frac{\sum_{k=1}^{n-1}W_{k}G_{k}}{\sum_{k=1}^{n-1}W_{k}}},\qquadn\geq2,
$$ 
and keep it up-to-date as we obtain a single additional return $G_{n}$ . In addition to keeping track of $V_{n}$ , we must maintain for each state the cumulative sum $C_{n}$ of the weights given to the first $n$ returns. The update rule for $V_{n}$ is 

$$
V_{n+1}=V_{n}+{\frac{W_{n}}{C_{n}}}\Big[G_{n}-V_{n}\Big],\qquadn\ge1,
$$ 
and 
$$
C_{n+1}=C_{n}+W_{n+1},
$$ 
where $C_{0}=0$ (and $V_{1}$ is arbitrary and thus need not be specified). Figure 5.9 gives a complete episode-by-episode incremental algorithm for Monte Carlo policy evaluation. The algorithm is nominally for the off-policy case, using weighted importance sampling, but applies as well to the on-policy case just by choosing the target and behavior policies as the same. 
o
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/c3789a12f13b300342e48dbae08e322ddc014ef666b3b4c584c73075d737d317.jpg) 

Figure 5.9: An incremental every-visit MC policy-evaluation algorithm, using weighted importance sampling. The approximation $Q$ converges to $q_{\pi}$ (for all encountered state–action pairs) even though all actions are selected according to a potentially different policy, $\mu$ . In the on-policy case ( $\pi=\mu$ ), $W$ is always 1. 
## 5.7 Off-Policy Monte Carlo Control 
We are now ready to present an example of the second class of learning control methods we consider in this book: off-policy methods. Recall that the distinguishing feature of on-policy methods is that they estimate the value of a policy while using it for control. In off-policy methods these two functions are separated. The policy used to generate behavior, called the behavior policy, may in fact be unrelated to the policy that is evaluated and improved, called the target policy. An advantage of this separation is that the target policy may be deterministic (e.g., greedy), while the behavior policy can continue to sample all possible actions. 

Off-policy Monte Carlo control methods use one of the techniques presented in the preceding two sections. They follow the behavior policy while learning about and improving the target policy. These techniques requires that the behavior policy has a nonzero probability of selecting all actions that might be selected by the target policy (coverage). To explore all possibilities, we require that the behavior policy be soft (i.e., that it select all actions in all states with nonzero probability). 

Figure 5.10 shows an off-policy Monte Carlo method, based on GPI and weighted importance sampling, for estimating $q_{*}$ . The target policy $\pi$ is the greedy policy with respect to $Q$ , which is an estimate of $q_{\pi}$ . The behavior policy $\mu$ can be anything, but in order to assure convergence of $\pi$ to the optimal policy, an infinite number of returns must be obtained for each pair of state and action. This can be assured by choosing $\mu$ to be $\boldsymbol{\varepsilon}$ -soft. 
A potential problem is that this method learns only from the tails of episodes, after the last nongreedy action. If nongreedy actions are frequent, then learning will be slow, particularly for states appearing in the early portions of long episodes. Potentially, this could greatly slow learning. There has been insufficient experience with off-policy Monte Carlo methods to assess how serious this problem is. If it is serious, the most important way to address it is probably by incorporating temporal-difference learning, the algorithmic idea developed in the next chapter. Alternatively, if $\gamma$ is less than 1, then the idea developed in the next section may also help significantly. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/eec714aad93b30c23662168d8535d7fdc4280f8033273ebda1683853d4cbd335.jpg) 
Figure 5.10: An off-policy every-visit MC control algorithm, using weighted importance sampling. The policy $\pi$ converges to optimal at all encountered states even though actions are selected according to a different soft policy $\mu$ , which may change between or even within episodes. 

## 5.8 Importance Sampling on Truncated Returns \*
So far our off-policy methods have formed importance-sampling ratios for returns considered as unitary wholes. This is clearly the right thing for a Monte Carlo method to do in the absence of discounting (i.e., if $\gamma\:=\:1$ ), but if $\gamma<1$ then there may be something better. Consider the case where episodes are long and $\gamma$ is significantly less than 1. For concreteness, say that episodes last 100 steps and that $\gamma=0$ . The return from time 0 will then be $G_{0}=R_{1}$ , and its importance sampling ratio will be a product of 100 factors, ${\frac{\pi(A_{0}|S_{0})}{\mu(A_{0}|S_{0})}}{\frac{\pi(A_{1}|S_{1})}{\mu(A_{1}|S_{1})}}\cdot\cdot\cdot{\frac{\pi(A_{99}|S_{99})}{\mu(A_{99}|S_{99})}}$ . In ordinary importance sampling, the return will be scaled by the entire product, but it is really only necessary to scale by the first factor, by $\frac{\pi(A_{0}|S_{0})}{\mu(A_{0}|S_{0})}$ . The other 99 factors ${\frac{\pi(A_{1}|S_{1})}{\mu(A_{1}|S_{1})}}\cdot\cdot\cdot\frac{\pi(A_{99}|S_{99})}{\mu(A_{99}|S_{99})}$ are irrelevant because after the first reward the return has already been determined. These later factors are all independent of the return and of expected value 1; they do not change the expected update, but they add enormously to its variance. In some cases they could even make the variance infinite. Let us now consider an idea for avoiding this large extraneous variance. 
The essence of the idea is to think of discounting as determining a probability of termination or, equivalently, a degree of partial termination. For any $\gamma\in[0,1)$ , we can think of the return $G_{0}$ as partly terminating in one step, to the degree $1-\gamma$ , producing a return of just the first reward, $R_{1}$ , and as partly terminating after two steps, to the degree $(1-\gamma)\gamma$ , producing a return of $R_{1}+R_{2}$ , and so on. The latter degree corresponds to terminating on the second step, $1-\gamma$ , and not having already terminated on the first step, $\gamma$ . The degree of termination on the third step is thus $(1-\gamma)\gamma^{2}$ , with the $\gamma^{2}$ reflecting that termination did not occur on either of the first two steps. The partial returns here are called flat partial returns: 
$$
\bar{G}_{t}^{h}=R_{t+1}+R_{t+2}+\cdot\cdot\cdot+R_{h},\qquad0\leq t<h\leq T,
$$ 
where “flat” denotes the absence of discounting, and “partial” denotes that these returns do not extend all the way to termination but instead stop at $h$ , called the horizon (and $^{\prime}I^{\prime}$ is the time of termination of the episode). The conventional full return $G_{t}$ can be viewed as a sum of flat partial returns as suggested above as follows: 
$$
\begin{array}{r l}{G_{t}=}&{R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\cdots+\gamma^{T-t-1}R_{T}}\ {\quad}&{\quad=(1-\gamma)R_{t+1}}\ {\quad\quad}&{\quad+(1-\gamma)\gamma\left(R_{t+1}+R_{t+2}\right)}\ {\quad\quad}&{\quad+(1-\gamma)\gamma^{2}\left(R_{t+1}+R_{t+2}+R_{t+3}\right)}\ {\quad\quad}&{\quad\vdots}\ {\quad}&{\quad\quad+(1-\gamma)\gamma^{T-t-2}\left(R_{t+1}+R_{t+2}+\cdots+R_{T-1}\right)}\ {\quad\quad}&{\quad\quad+\gamma^{T-t-1}\left(R_{t+1}+R_{t+2}+\cdots+R_{T}\right)}\ {\quad\quad}&{\quad\quad=\gamma^{T-t-1}\bar{G}_{t}^{T}+(1-\gamma)\displaystyle\sum_{h=t+1}^{T-1}\bar{G}_{t}^{h-t-1}\bar{G}_{t}^{h}}\end{array}
$$ 
Now we need to scale the flat partial returns by an importance sampling ratio that is similarly truncated. As $G_{t}^{h}$ only involves rewards up to a horizon $h$ , we only need the ratio of the probabilities up to $h$ . We define an ordinary importance-sampling estimator, analogous to (5.4), as 
$$
V(s)=\frac{\sum_{t\in\Im(s)}\left(\gamma^{T(t)-t-1}\rho_{t}^{T(t)}\bar{G}_{t}^{T(t)}+(1-\gamma)\sum_{h=t+1}^{T(t)-1}\gamma^{h-t-1}\rho_{t}^{h}\bar{G}_{t}^{h}\right)}{|\mathcal{T}(s)|},
$$ 
and a weighted importance-sampling estimator, analogous to (5.5), as 
$$
V(s)=\frac{\sum_{t\in\Upsilon(s)}\left(\gamma^{T(t)-t-1}\rho_{t}^{T(t)}\bar{G}_{t}^{T(t)}+(1-\gamma)\sum_{h=t+1}^{T(t)-1}\gamma^{h-t-1}\rho_{t}^{h}\bar{G}_{t}^{h}\right)}{\sum_{t\in\Upsilon(s)}\left(\gamma^{T(t)-t-1}\rho_{t}^{T(t)}+(1-\gamma)\sum_{h=t+1}^{T(t)-1}\gamma^{h-t-1}\rho_{t}^{h}\right)}.
$$ 
## 5.9 Summary 
The Monte Carlo methods presented in this chapter learn value functions and optimal policies from experience in the form of sample episodes. This gives them at least three kinds of advantages over DP methods. First, they can be used to learn optimal behavior directly from interaction with the environment, with no model of the environment’s dynamics. Second, they can be used with simulation or sample models. For surprisingly many applications it is easy to simulate sample episodes even though it is difficult to construct the kind of explicit model of transition probabilities required by DP methods. Third, it is easy and efficient to focus Monte Carlo methods on a small subset of the states. A region of special interest can be accurately evaluated without going to the expense of accurately evaluating the rest of the state set (we explore this further in Chapter 8). 

A fourth advantage of Monte Carlo methods, which we discuss later in the book, is that they may be less harmed by violations of the Markov property. This is because they do not update their value estimates on the basis of the value estimates of successor states. In other words, it is because they do not bootstrap. 

In designing Monte Carlo control methods we have followed the overall schema of generalized policy iteration (GPI) introduced in Chapter 4. GPI involves interacting processes of policy evaluation and policy improvement. Monte Carlo methods provide an alternative policy evaluation process. Rather than use a model to compute the value of each state, they simply average many returns that start in the state. Because a state’s value is the expected return, this average can become a good approximation to the value. In control methods we are particularly interested in approximating action-value functions, because these can be used to improve the policy without requiring a model of the environment’s transition dynamics. Monte Carlo methods intermix policy evaluation and policy improvement steps on an episode-by-episode basis, and can be incrementally implemented on an episode-by-episode basis. 

Maintaining sufficient exploration is an issue in Monte Carlo control methods. It is not enough just to select the actions currently estimated to be best, because then no returns will be obtained for alternative actions, and it may never be learned that they are actually better. One approach is to ignore this problem by assuming that episodes begin with state–action pairs randomly selected to cover all possibilities. Such exploring starts can sometimes be arranged in applications with simulated episodes, but are unlikely in learning from real experience. In on-policy methods, the agent commits to always exploring and tries to find the best policy that still explores. In off-policy methods, the agent also explores, but learns a deterministic optimal policy that may be unrelated to the policy followed. 

Off-policy Monte Carlo prediction refers to learning the value function of a target policy from data generated by a different behavior policy. Such learning methods are all based on some form of importance sampling, that is, on weighting returns by the ratio of the probabilities of taking the observed actions under the two policies. Ordinary importance sampling uses a simple average of the weighted returns, whereas weighted importance sampling uses a weighted average. Ordinary importance sampling produces unbiased estimates, but has larger, possibly infinite, variance, whereas weighted importance sampling always has finite variance and are preferred in practice. Despite their conceptual simplicity, off-policy Monte Carlo methods for both prediction and control remain unsettled and a subject of ongoing research. 

The Monte Carlo methods treated in this chapter differ from the DP methods treated in the previous chapter in two major ways. First, they operate on sample experience, and thus can be used for direct learning without a model. Second, they do not bootstrap. That is, they do not update their value estimates on the basis of other value estimates. These two differences are not tightly linked, and can be separated. In the next chapter we consider methods that learn from experience, like Monte Carlo methods, but also bootstrap, like DP methods. 

# 6 Temporal-Difference Learning 
If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be temporal-difference (TD) learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap). The relationship between TD, DP, and Monte Carlo methods is a recurring theme in the theory of reinforcement learning. This chapter is the beginning of our exploration of it. Before we are done, we will see that these ideas and methods blend into each other and can be combined in many ways. In particular, in Chapter 7 we introduce the TD( $\lambda$ ) algorithm, which seamlessly integrates TD and Monte Carlo methods. 
>  TD 是 DP 和 MC 思想的结合
>  类似 MC 方法，TD 可以直接从经验学习，不需要环境动态模型，类似 DP 方法，TD 基于另一个估计更新当前估计，而不是等待最终的结果 (也就是 DP 和 TD 都自举)

As usual, we start by focusing on the policy evaluation or prediction problem, that of estimating the value function $v_{\pi}$ for a given policy $\pi$ . 

For the control problem (finding an optimal policy), DP, TD, and Monte Carlo methods all use some variation of generalized policy iteration (GPI). The differences in the methods are primarily differences in their approaches to the prediction problem. 
>  对于控制问题 (寻找最优策略)，DP, TD, MC 都使用 GPI 的变体，它们的差异主要体现在它们对预测问题的求解，即如何进行策略评估 (计算价值函数)

## 6.1 TD Prediction 
Both TD and Monte Carlo methods use experience to solve the prediction problem. Given some experience following a policy $\pi$ , both methods update their estimate $\upsilon$ of $v_{\pi}$ for the nonterminal states $S_{t}$ occurring in that experience. 
>  TD 和 MC 都使用经验来求解预测问题
>  给定遵循策略 $\pi$ 得到的经验，TD 和 MC 根据经验中出现的非终止状态 $S_t$ 更新它们对 $v_\pi$ 的预测

Roughly speaking, Monte Carlo methods wait until the return following the visit is known, then use that return as a target for $V(S_{t})$ . A simple every-visit Monte Carlo method suitable for nonstationary environments is 

$$
V(S_{t})\leftarrow V(S_{t})+\alpha\Big[G_{t}-V(S_{t})\Big],\tag{6.1}
$$ 
where $G_{t}$ is the actual return following time $t$ , and $\alpha$ is a constant stepsize parameter (c.f., Equation 2.4). Let us call this method constant- $\alpha$ MC. 

>  MC 方法会等待到某个状态访问的最终回报已知，一个 every-visit MC 方法对于目标 $V(S_t)$ 的更新如上，其中 $G_t$ 是实际回报，$\alpha$ 是步长参数
>  该方法称为 constant- $\alpha$ MC

Whereas Monte Carlo methods must wait until the end of the episode to determine the increment to $V(S_{t})$ (only then is $G_{t}$ known), TD methods need wait only until the next time step. At time $t+1$ they immediately form a target and make a useful update using the observed reward $R_{t+1}$ and the estimate $V(S_{t+1})$ . The simplest TD method, known as $T D({{0}})$ , is 

$$
V(S_{t})\leftarrow V(S_{t})+\alpha\Big[R_{t+1}+\gamma V(S_{t+1})-V(S_{t})\Big].\tag{6.2}
$$

In effect, the target for the Monte Carlo update is $G_{t}$ , whereas the target for the TD update is $R_{t+1}+\gamma V(S_{t+1})$ . 

>  TD 方法不会等到结束，TD(0) 仅等待一个时间步，然后用估计 $V(S_{t+1})$ 和奖励 $R_{t+1}$ 近似 $G_t$，然后执行更新
>  MC 更新的目标是 $G_t$，TD 更新的目标是 $R_{t+1} + \gamma V_{S_{t+1}}$

Because the TD method bases its update in part on an existing estimate, we say that it is a bootstrapping method, like DP. 
>  TD 方法基于现存估计更新，故它和 DP 一样是自举方法

We know from Chapter 3 that 

$$\begin{align*}
v_{\pi}(s) &= \mathbb{E}_{\pi}\big[G_{t}\mid S_{t}=s\big]\tag{6.2} \\
           &= \mathbb{E}_{\pi}\bigg[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}\biggm\vert S_{t}=s\bigg] \\
           &= \mathbb{E}_{\pi}\bigg[R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+2}\biggm\vert S_{t}=s\bigg] \\
           &= \mathbb{E}_{\pi}\big[R_{t+1}+\gamma v_{\pi}\big (S_{t+1}\bigm)\mid S_{t}=s\big].\tag{6.4}
\end{align*}$$

Roughly speaking, Monte Carlo methods use an estimate of (6.3) as a target, whereas DP methods use an estimate of (6.4) as a target. The Monte Carlo target is an estimate because the expected value in (6.3) is not known; a sample return is used in place of the real expected return. The DP target is an estimate not because of the expected values, which are assumed to be completely provided by a model of the environment, but because $v_{\pi}(S_{t+1})$ is not known and the current estimate, $V(S_{t+1})$ , is used instead. The TD target is an estimate for both reasons: it samples the expected values in (6.4) and it uses the current estimate $V$ instead of the true $v_{\pi}$ . Thus, TD methods combine the sampling of Monte Carlo with the bootstrapping of DP. As we shall see, with care and imagination this can take us a long way toward obtaining the advantages of both Monte Carlo and DP methods. 

>  MC 方法使用对 Eq 6.3 的估计作为目标，DP 方法使用对 Eq 6.4 的估计作为目标
>  MC 方法的“估计”来源于用样本回报估计实际的期望回报
>  DP 方法的“估计”来源于 $v_\pi(S_{t+1})$ 未知，使用 $V(S_{t+1})$ 代替，而不是来源于期望，DP 会根据环境动态计算出期望
>  TD 则二者的“估计”兼有，TD 在 Eq 6.4 使用样本替代期望，并且用 $V$ 替代 $v_\pi$
>  因此，TD 结合了 MC 的采样和 DP 的自举

Figure 6.1 specifies TD(0) completely in procedural form, and Figure 6.2 shows its backup diagram. The value estimate for the state node at the top of the backup diagram is updated on the basis of the one sample transition from it to the immediately following state. 
>  TD(0) 中，当前状态节点的价值估计根据一个后继状态的样本更新

We refer to TD and Monte Carlo updates as sample backups because they involve looking ahead to a sample successor state (or state–action pair), using the value of the successor and the reward along the way to compute a backed-up value, and then changing the value of the original state (or state–action pair) accordingly. Sample backups differ from the full backups of DP methods in that they are based on a single sample successor rather than on a complete distribution of all possible successors. 
>  TD 和 MC 更新都称为样本回溯更新，因为涉及到了采样
>  DP 则是全回溯更新，基于所有可能后继的完整分布更新

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/413ef04ea3450200d58dfa1bd00f41ffe32b5b1af3f732ab342a84a8f1b75dec.jpg) 

Figure 6.1: Tabular $\mathrm{TD}(0)$ for estimating $v_{\pi}$ . 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/b9fa09f5cd8ea5846133c75b64d7e948999edce39f6c0d5e8c22267958663759.jpg) 

Figure 6.2: The backup diagram for TD(0). 

Example 6.1: Driving Home Each day as you drive home from work, you try to predict how long it will take to get home. When you leave your office, you note the time, the day of week, and anything else that might be relevant. Say on this Friday you are leaving at exactly 6 o’clock, and you estimate that it will take 30 minutes to get home. As you reach your car it is 6:05, and you notice it is starting to rain. Traffic is often slower in the rain, so you re-estimate that it will take 35 minutes from then, or a total of 40 minutes. Fifteen minutes later you have completed the highway portion of your journey in good time. As you exit onto a secondary road you cut your estimate of total travel time to 35 minutes. Unfortunately, at this point you get stuck behind a slow truck, and the road is too narrow to pass. You end up having to follow the truck until you turn onto the side street where you live at 6:40. Three minutes later you are home. The sequence of states, times, and predictions is 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/d9f0c8e91d1617dbd2c33e64be11472709b1635a66967c631bb3f8420bac8fac.jpg) 
Figure 6.3: Changes recommended by Monte Carlo methods in the driving home example. 

thus as follows: 
<html><body><table><tr><td></td><td>ElapsedTime</td><td>Predicted</td><td>Predicted TotalTime</td></tr><tr><td>leaving office, friday at 6</td><td>minutes) 0</td><td>TimetoGo 30</td><td>30</td></tr><tr><td>reach car, raining</td><td>5</td><td>35</td><td>40</td></tr><tr><td>exiting highway</td><td>20</td><td>15</td><td>35</td></tr><tr><td>2ndary road, behind truck</td><td>30</td><td>10</td><td>40</td></tr><tr><td>entering home street</td><td>40</td><td>3</td><td>43</td></tr><tr><td>arrive home</td><td>43</td><td>0</td><td>43</td></tr></table></body></html> 

The rewards in this example are the elapsed times on each leg of the journey.1 We are not discounting ( $\gamma=1$ ), and thus the return for each state is the actual time to go from that state. The value of each state is the expected time to go. The second column of numbers gives the current estimated value for each state encountered. 

A simple way to view the operation of Monte Carlo methods is to plot the predicted total time (the last column) over the sequence, as in Figure 6.3. The arrows show the changes in predictions recommended by the constant- $\alpha$ MC method (6.1), for $\alpha=1$ . These are exactly the errors between the estimated value (predicted time to go) in each state and the actual return (actual time to go). For example, when you exited the highway you thought it would take only 15 minutes more to get home, but in fact it took 23 minutes. Equation 6.1 applies at this point and determines an increment in the estimate of time to go after exiting the highway. The error, $G_{t}-V(S_{t})$ , at this time is eight minutes. Suppose the step-size parameter, $\alpha$ , is $1/2$ . Then the predicted time to go after exiting the highway would be revised upward by four minutes as a result of this experience. This is probably too large a change in this case; the truck was probably just an unlucky break. In any event, the change can only be made off-line, that is, after you have reached home. Only at this point do you know any of the actual returns. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/ea54344e4da5144d417a0d924fa8716273e58c224e80dfff5a474423e3b35d1c.jpg) 

Figure 6.4: Changes recommended by TD methods in the driving home example. 

Is it necessary to wait until the final outcome is known before learning can begin? Suppose on another day you again estimate when leaving your office that it will take 30 minutes to drive home, but then you become stuck in a massive traffic jam. Twenty-five minutes after leaving the office you are still bumper-to-bumper on the highway. You now estimate that it will take another 25 minutes to get home, for a total of 50 minutes. As you wait in traffic, you already know that your initial estimate of 30 minutes was too optimistic. Must you wait until you get home before increasing your estimate for the initial state? According to the Monte Carlo approach you must, because you don’t yet know the true return. 

According to a TD approach, on the other hand, you would learn immediately, shifting your initial estimate from 30 minutes toward 50. In fact, each estimate would be shifted toward the estimate that immediately follows it. Returning to our first day of driving, Figure 6.4 shows the same predictions as Figure 6.3, except with the changes recommended by the TD rule (6.2) (these are the changes made by the rule if $\alpha=1$ ). Each error is proportional to the change over time of the prediction, that is, to the temporal differences in predictions. 

Besides giving you something to do while waiting in traffic, there are several computational reasons why it is advantageous to learn based on your current predictions rather than waiting until termination when you know the actual return. We briefly discuss some of these next. 

## 6.2 Advantages of TD Prediction Methods 
TD methods learn their estimates in part on the basis of other estimates. They learn a guess from a guess—they bootstrap. Is this a good thing to do? What advantages do TD methods have over Monte Carlo and DP methods? Developing and answering such questions will take the rest of this book and more. In this section we briefly anticipate some of the answers. 

Obviously, TD methods have an advantage over DP methods in that they do not require a model of the environment, of its reward and next-state probability distributions. 
>  TD 优于 DP 的点在于它不需要环境动态

The next most obvious advantage of TD methods over Monte Carlo methods is that they are naturally implemented in an on-line, fully incremental fashion. With Monte Carlo methods one must wait until the end of an episode, because only then is the return known, whereas with TD methods one need wait only one time step. Surprisingly often this turns out to be a critical consideration. Some applications have very long episodes, so that delaying all learning until an episode’s end is too slow. Other applications are continuing tasks and have no episodes at all. Finally, as we noted in the previous chapter, some Monte Carlo methods must ignore or discount episodes on which experimental actions are taken, which can greatly slow learning. TD methods are much less susceptible to these problems because they learn from each transition regardless of what subsequent actions are taken. 
>  TD 优于 MC 的点在于 TD 可以自然地实现为在线的、增量更新的形式
>  MC 必须等待回合的结束，TD 只需要等待一个时间步，要知道一些应用的回合很长，一些应用根本没有回合的概念

But are TD methods sound? Certainly it is convenient to learn one guess from the next, without waiting for an actual outcome, but can we still guarantee convergence to the correct answer? Happily, the answer is yes. For any fixed policy $\pi$ , the TD algorithm described above has been proved to converge to $v_{\pi}$ , in the mean for a constant step-size parameter if it is sufficiently small, and with probability 1 if the step-size parameter decreases according to the usual stochastic approximation conditions (2.7). Most convergence proofs apply only to the table-based case of the algorithm presented above (6.2), but some also apply to the case of general linear function approximation. These results are discussed in a more general setting in the next two chapters. 
>  TD 方法仍然保证可以正确收敛，对于任意固定策略 $\pi$，TD 保证可以收敛到 $v_\pi$

If both TD and Monte Carlo methods converge asymptotically to the correct predictions, then a natural next question is “Which gets there first?” In other words, which method learns faster? Which makes the more efficient use of limited data? At the current time this is an open question in the sense that no one has been able to prove mathematically that one method converges faster than the other. In fact, it is not even clear what is the most appropriate formal way to phrase this question! In practice, however, TD methods have usually been found to converge faster than constant- $\alpha$ MC methods on stochastic tasks, as illustrated in the following example. 
>  虽然没有理论证明，在实践中经常发现 TD 方法收敛得比 constant- $\alpha$ MC 方法更快

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/916e7702e165ca38798dd7c2fbe745414520c99c9d092c65b4629a4817ada7e7.jpg) 

Figure 6.5: A small Markov process for generating random walks. 

Example 6.2: Random Walk In this example we empirically compare the prediction abilities of TD(0) and constant- $\alpha$ MC applied to the small Markov process shown in Figure 6.5. All episodes start in the center state, C, and proceed either left or right by one state on each step, with equal probability. This behavior is presumably due to the combined effect of a fixed policy and an environment’s state-transition probabilities, but we do not care which; we are concerned only with predicting returns however they are generated. Episodes terminate either on the extreme left or the extreme right. When an episode terminates on the right a reward of $+1$ occurs; all other rewards are zero. For example, a typical walk might consist of the following state and reward sequence: ${\mathsf{C}},0,\mathsf{B},0,{\mathsf{C}},0,\mathsf{D},0,\mathsf{E},1$ . Because this task is undiscounted and episodic, the true value of each state is the probability of terminating on the right if starting from that state. Thus, the true value of the center state is $v_{\pi}(\mathsf{C})=0.5$ . The true values of all the states, $\mathsf{A}$ through $\mathsf{E}$ , are $\frac{1}{6},\frac{2}{6},\frac{3}{6},\frac{4}{6}$ , and $\frac{5}{6}$ . Figure 6.6 shows the values learned by TD(0) approaching the true values as more episodes are experienced. Averaging over many episode sequences, Figure 6.7 shows the average error in the predictions found by $\mathrm{TD}(0)$ and constant- $\alpha$ MC, for a variety of values of $\alpha$ , as a function of number of episodes. In all cases the approximate value function was initialized to the intermediate value $V(s)=0.5$ , for all $s$ . The TD method is consistently better than the MC method on this task over this number of episodes. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/69ffc61068a6bac9350710804b3a91d9d7ac27d482d899b9bea6240067ae8258.jpg) 

Figure 6.6: Values learned by TD(0) after various numbers of episodes. The final estimate is about as close as the estimates ever get to the true values. With a constant step-size parameter ( $\alpha=0.1$ in this example), the values fluctuate indefinitely in response to the outcomes of the most recent episodes. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/c3e083119cfb43564d3c45dcd45fb0a25114594b67d9102555911660a8d743a3.jpg) 

Figure 6.7: Learning curves for TD(0) and constant- $\alpha$ MC methods, for various values of $\alpha$ , on the prediction problem for the random walk. The performance measure shown is the root mean-squared (RMS) error between the value function learned and the true value function, averaged over the five states. These data are averages over 100 different sequences of episodes. 

## 6.3 Optimality of TD(0) 
Suppose there is available only a finite amount of experience, say 10 episodes or 100 time steps. In this case, a common approach with incremental learning methods is to present the experience repeatedly until the method converges upon an answer. 
>  假设经验的数量是有限的，例如只有 10 个回合或 100 个时间步
>  此时，常用的方法是反复使用这些经验进行增量学习，直到收敛

Given an approximate value function, $V$ , the increments specified by (6.1) or (6.2) are computed for every time step $t$ at which a nonterminal state is visited, but the value function is changed only once, by the sum of all the increments. Then all the available experience is processed again with the new value function to produce a new overall increment, and so on, until the value function converges. We call this batch updating because updates are made only after processing each complete batch of training data. 
>  给定近似价值函数 $V$，我们计算经验中所有时间步 $t$ 的增量 (TD 误差)，然后用累积的 TD 误差更新一次价值函数
>  然后，我们再重复这一过程，不断反复利用经验更新价值函数，直到收敛
>  这称为批量更新，因为仅在处理完每个批量的训练数据之后才执行更新

Under batch updating, TD (0) converges deterministically to a single answer independent of the step-size parameter, $\alpha$ , as long as $\alpha$ is chosen to be sufficiently small. The constant- $\alpha$ MC method also converges deterministically under the same conditions, but to a different answer. 
>  使用批量更新的 TD(0) 会确定性地收敛到独立于步长参数 $\alpha$ 的一个结果，只要 $\alpha$ 足够小。同一条件下，constant- $\alpha$ MC 方法也会确定性收敛到另一个结果

Understanding these two answers will help us understand the difference between the two methods. Under normal updating the methods do not move all the way to their respective batch answers, but in some sense they take steps in these directions. Before trying to understand the two answers in general, for all possible tasks, we first look at a few examples. 

Example 6.3 Random walk under batch updating. Batch-updating versions of TD(0) and constant- $\alpha$ MC were applied as follows to the random walk prediction example (Example 6.2). After each new episode, all episodes seen so far were treated as a batch. They were repeatedly presented to the algorithm, either TD(0) or constant- $\alpha$ MC, with $\alpha$ sufficiently small that the value function converged. The resulting value function was then compared with $v_{\pi}$ , and the average root mean-squared error across the five states (and across 100 independent repetitions of the whole experiment) was plotted to obtain the learning curves shown in Figure 6.8. Note that the batch TD method was consistently better than the batch Monte Carlo method. ■ 

Under batch training, constant- $\alpha$ MC converges to values, $V(s)$ , that are sample averages of the actual returns experienced after visiting each state $s$ . These are optimal estimates in the sense that they minimize the mean-squared error from the actual returns in the training set. In this sense it is surprising that the batch TD method was able to perform better according to the root mean-squared error measure shown in Figure 6.8. How is it that batch TD was able to perform better than this optimal method? The answer is that the Monte Carlo method is optimal only in a limited way, and that TD is optimal in a way that is more relevant to predicting returns. But first let’s develop our intuitions about different kinds of optimality through another example. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/e60a2bccb7ef3c50db1536cea2907ca6158128643ea071c5baf77830430be356.jpg) 
Figure 6.8: Performance of TD(0) and constant- $\alpha$ MC under batch training on the random walk task. 
Example 6.4: You are the Predictor Place yourself now in the role of the predictor of returns for an unknown Markov reward process. Suppose you observe the following eight episodes: 
This means that the first episode started in state A, transitioned to $\textsf{B}$ with a reward of 0, and then terminated from $\textsf{B}$ with a reward of 0. The other seven episodes were even shorter, starting from $\textsf{B}$ and terminating immediately. Given this batch of data, what would you say are the optimal predictions, the best values for the estimates $V(\mathsf{A})$ and $V(\mathsf{B})$ ? Everyone would probably agree that the optimal value for $V(\mathsf{B})$ is $\textstyle{\frac{3}{4}}$ , because six out of the eight times in state $\textsf{B}$ the process terminated immediately with a return of 1, and the other two times in $\textsf{B}$ the process terminated immediately with a return of $0$ . 
But what is the optimal value for the estimate $V(\mathsf{A})$ given this data? Here there are two reasonable answers. One is to observe that $100\%$ of the times the process was in state A it traversed immediately to $\textsf{B}$ (with a reward of $0$ ); and since we have already decided that $\textsf{B}$ has value $\textstyle{\frac{3}{4}}$ , therefore A must have value $\textstyle{\frac{3}{4}}$ as well. One way of viewing this answer is that it is based on first modeling the Markov process, in this case as 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/847a1858d6224fcf718e70acc682108805ad180b761842bffbb6e26b06a8b533.jpg) 
and then computing the correct estimates given the model, which indeed in this case gives $\begin{array}{r}{V(\mathsf{A})=\frac{3}{4}}\end{array}$ . This is also the answer that batch TD(0) gives. 
The other reasonable answer is simply to observe that we have seen A once and the return that followed it was 0; we therefore estimate $V(\mathsf{A})$ as 0. This is the answer that batch Monte Carlo methods give. Notice that it is also the answer that gives minimum squared error on the training data. In fact, it gives zero error on the data. But still we expect the first answer to be better. If the process is Markov, we expect that the first answer will produce lower error on future data, even though the Monte Carlo answer is better on the existing data. 

The above example illustrates a general difference between the estimates found by batch TD(0) and batch Monte Carlo methods. Batch Monte Carlo methods always find the estimates that minimize mean-squared error on the training set, whereas batch TD(0) always finds the estimates that would be exactly correct for the maximum-likelihood model of the Markov process.
>  批量 MC 方法一般收敛到最小化训练集均方误差的估计，批量 TD(0) 方法一般收敛到 Markov 过程的准确极大似然模型
 
 In general, the maximum-likelihood estimate of a parameter is the parameter value whose probability of generating the data is greatest. In this case, the maximum-likelihood estimate is the model of the Markov process formed in the obvious way from the observed episodes: the estimated transition probability from $i$ to $j$ is the fraction of observed transitions from $i$ that went to $j$ , and the associated expected reward is the average of the rewards observed on those transitions. Given this model, we can compute the estimate of the value function that would be exactly correct if the model were exactly correct. This is called the certainty-equivalence estimate because it is equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated. In general, batch TD(0) converges to the certainty equivalence estimate. 
 
This helps explain why TD methods converge more quickly than Monte Carlo methods. In batch form, TD(0) is faster than Monte Carlo methods because it computes the true certainty-equivalence estimate. This explains the advantage of TD(0) shown in the batch results on the random walk task (Figure 6.8). The relationship to the certainty-equivalence estimate may also explain in part the speed advantage of nonbatch TD(0) (e.g., Figure 6.7). Although the nonbatch methods do not achieve either the certainty-equivalence or the minimum squared-error estimates, they can be understood as moving roughly in these directions. Nonbatch TD(0) may be faster than constant- $\alpha$ MC because it is moving toward a better estimate, even though it is not getting all the way there. At the current time nothing more definite can be said about the relative efficiency of on-line TD and Monte Carlo methods. 

Finally, it is worth noting that although the certainty-equivalence estimate is in some sense an optimal solution, it is almost never feasible to compute it directly. If $N$ is the number of states, then just forming the maximum likelihood estimate of the process may require $N^{2}$ memory, and computing the corresponding value function requires on the order of $N^{3}$ computational steps if done conventionally. In these terms it is indeed striking that TD methods can approximate the same solution using memory no more than $N$ and repeated computations over the training set. On tasks with large state spaces, TD methods may be the only feasible way of approximating the certainty equivalence solution. 

## 6.4 Sarsa: On-Policy TD Control 
We turn now to the use of TD prediction methods for the control problem. As usual, we follow the pattern of generalized policy iteration (GPI), only this time using TD methods for the evaluation or prediction part. As with Monte Carlo methods, we face the need to trade off exploration and exploitation, and again approaches fall into two main classes: on-policy and off-policy. In this section we present an on-policy TD control method. 
>  我们考虑控制问题，仍然遵循 GPI 的模式，此时我们用 TD 方法进行策略评估

The first step is to learn an action-value function rather than a state-value function. In particular, for an on-policy method we must estimate $q_{\pi}(s,a)$ for the current behavior policy $\pi$ and for all states $s$ and actions $a$ . This can be done using essentially the same TD method described above for learning $v_{\pi}$ . 
>  我们需要学习动作价值函数而非状态价值函数，在同策略方法下，我们需要为当前的行为策略 $\pi$ 估计 $q_\pi(s, a)$，我们使用之前描述的学习 $v_\pi$ 的相同 TD 方法对其进行估计

Recall that an episode consists of an alternating sequence of states and state–action pairs: 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/e26bddff3eb36635e810576e47d8e0d12884ad99547f43ba420351f352149b62.jpg) 


In the previous section we considered transitions from state to state and learned the values of states. Now we consider transitions from state–action pair to state–action pair, and learn the value of state–action pairs. Formally these cases are identical: they are both Markov chains with a reward process. 
>  在之前，我们考虑的是从状态到状态的转移，以学习状态的价值
>  现在，我们考虑从状态-动作对到状态-动作对的转移，以学习状态-动作对的价值
>  形式上，二者都是相同的，都属于带有奖励的 Markov 过程

The theorems assuring the convergence of state values under TD(0) also apply to the corresponding algorithm for action values: 

$$
Q(S_{t},A_{t})\xleftarrow{}Q(S_{t},A_{t})+\alpha\Big[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_{t},A_{t})\Big].\tag{6.5}
$$

This update is done after every transition from a nonterminal state $S_{t}$ . If $S_{t+1}$ is terminal, then $Q(S_{t+1},A_{t+1})$ is defined as zero. This rule uses every element of the quintuple of events, $(S_{t},A_{t},R_{t+1},S_{t+1},A_{t+1})$ , that make up a transition from one state–action pair to the next. This quintuple gives rise to the name Sarsa for the algorithm. 

>  使用 TD(0) 对动作价值函数的更新如 Eq 6.5 所示
>  更新在每次转移后执行，如果 $S_{t+1}$ 是终止状态，则其动作价值定义为 0
>  这一更新规则需要利用五元组 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$，该五元组表示了从一个状态-动作对到下一个状态-动作对的转移

It is straightforward to design an on-policy control algorithm based on the Sarsa prediction method. As in all on-policy methods, we continually estimate $q_{\pi}$ for the behavior policy $\pi$ , and at the same time change $\pi$ toward greediness with respect to $q_{\pi}$ . The general form of the Sarsa control algorithm is given in Figure 6.9. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/3d4bf63e45060b80b69296d2a01f1aaa7df62aa502d00cbbdc93a01ce7e3a229.jpg) 

Figure 6.9: Sarsa: An on-policy TD control algorithm. 

The convergence properties of the Sarsa algorithm depend on the nature of the policy’s dependence on $q$ . For example, one could use $\varepsilon$ -greedy or $\varepsilon$ - soft policies. According to Satinder Singh (personal communication), Sarsa converges with probability 1 to an optimal policy and action-value function as long as all state–action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy (which can be arranged, for example, with $\boldsymbol{\varepsilon}$ -greedy policies by setting $\varepsilon=1/t$ ), but this result has not yet been published in the literature. 

Example 6.5: Windy Gridworld Figure 6.10 shows a standard gridworld, with start and goal states, but with one difference: there is a crosswind upward through the middle of the grid. The actions are the standard four—up, down, right, and left—but in the middle region the resultant next states are shifted upward by a “wind,” the strength of which varies from column to column. The strength of the wind is given below each column, in number of cells shifted upward. For example, if you are one cell to the right of the goal, then the action left takes you to the cell just above the goal. Let us treat this as an undiscounted episodic task, with constant rewards of $-1$ until the goal state is reached. Figure 6.11 shows the result of applying $\boldsymbol{\varepsilon}$ -greedy Sarsa to this task, with $\varepsilon=0.1$ , $\alpha=0.5$ , and the initial values $Q(s,a)=0$ for all $s,a$ . The increasing slope of the graph shows that the goal is reached more and more quickly over time. By 8000 time steps, the greedy policy (shown inset) was long since optimal; continued $\boldsymbol{\varepsilon}$ -greedy exploration kept the average episode length at about 17 steps, two more than the minimum of 15. Note that Monte Carlo methods cannot easily be used on this task because termination is not guaranteed for all policies. If a policy was ever found that caused the agent to stay in the same state, then the next episode would never end. Step-by-step learning methods such as Sarsa do not have this problem because they quickly learn during the episode that such policies are poor, and switch to something else. ■ 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/01042f47e5c5223191effb5d0817df6973e47ab237eeca3de3164158e505e62a.jpg) 

Figure 6.10: Gridworld in which movement is altered by a location-dependent, upward “wind.” 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/37e0976a6c9674cd806294334446e2021936161c931c11b539d864a052687a4a.jpg) 

Figure 6.11: Results of Sarsa applied to the windy gridworld. 

## 6.5 Q-Learning: Off-Policy TD Control 
One of the most important breakthroughs in reinforcement learning was the development of an off-policy TD control algorithm known as $Q$ -learning (Watkins, 1989). Its simplest form, one-step Q-learning, is defined by 

$$
Q(S_{t},A_{t})\xleftarrow{}Q(S_{t},A_{t})+\alpha\Big[R_{t+1}+\gamma\operatorname*{max}_{a}Q(S_{t+1},a)-Q(S_{t},A_{t})\Big].\tag{6.6}
$$ 
In this case, the learned action-value function, $Q$ , directly approximates $q_{*}$ , the optimal action-value function, independent of the policy being followed. This dramatically simplifies the analysis of the algorithm and enabled early convergence proofs. The policy still has an effect in that it determines which state–action pairs are visited and updated. However, all that is required for correct convergence is that all pairs continue to be updated. 
>  异策略的 TD 控制算法称为 Q-Learning，一步 Q Learning 的更新如 Eq 6.6 所示
>  按照 Eq 6.6 学习的动作价值函数 $Q$ 是对 $q_*$ 的直接近似，独立于使用什么策略
>  行为策略仍然存在影响，它会决定哪个状态-动作对被访问且更新，但 Q-Learning 对于收敛的要求仅仅是所有的状态-动作对都会被继续更新

As we observed in Chapter 5, this is a minimal requirement in the sense that any method guaranteed to find optimal behavior in the general case must require it. Under this assumption and a variant of the usual stochastic approximation conditions on the sequence of step-size parameters, $Q$ has been shown to converge with probability 1 to $q_{*}$ . The Q-learning algorithm is shown in procedural form in Figure 6.12. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/2507a1a1262ca99d5c9d36e277ef8fd7af6a5e504142fb45fded9a54f0ef00e5.jpg) 

Figure 6.12: Q-learning: An off-policy TD control algorithm. 

What is the backup diagram for Q-learning? The rule (6.6) updates a state–action pair, so the top node, the root of the backup, must be a small, filled action node. The backup is also from action nodes, maximizing over all those actions possible in the next state. Thus the bottom nodes of the backup diagram should be all these action nodes. Finally, remember that we indicate taking the maximum of these “next action” nodes with an arc across them (Figure 3.7). Can you guess now what the diagram is? If so, please do make a guess before turning to the answer in Figure 6.14. 
>  Q-Learning 的回溯更新图如下所示

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/9d6f0f31458d46881c0482bfc17198eb64865ee818f16843d0239ab80bd15d5f.jpg) 

Figure 6.14: The backup diagram for Q-learning. 

Example 6.6: Cliff Walking This gridworld example compares Sarsa and Q-learning, highlighting the difference between on-policy (Sarsa) and offpolicy (Q-learning) methods. Consider the gridworld shown in the upper part of Figure 6.13. This is a standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down, right, and left. Reward is $-1$ on all transitions except those into the the region marked “The Cliff.” Stepping into this region incurs a reward of $-100$ and sends the agent instantly back to the start. The lower part of the figure shows the performance of the Sarsa and Q-learning methods with $\varepsilon$ -greedy action selection, $\varepsilon=0.1$ . After an initial transient, Q-learning learns values for the optimal policy, that which travels right along the edge of the cliff. Unfortunately, this results in its occasionally falling off the cliff because of the $\varepsilon$ -greedy action selection. Sarsa, on the other hand, takes the action selection into account and learns the longer but safer path through the upper part of the grid. Although Qlearning actually learns the values of the optimal policy, its on-line performance is worse than that of Sarsa, which learns the roundabout policy. Of course, if $\boldsymbol{\varepsilon}$ were gradually reduced, then both methods would asymptotically converge to the optimal policy. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/25ded17a7073872d279b0833e9902adb37cfe4657499cf5f4a23f4e761b39bcd.jpg) 

Figure 6.13: The cliff-walking task. The results are from a single run, but smoothed. 


## 6.6 Games, Afterstates, and Other Special Cases 
In this book we try to present a uniform approach to a wide class of tasks, but of course there are always exceptional tasks that are better treated in a specialized way. For example, our general approach involves learning an action-value function, but in Chapter 1 we presented a TD method for learning to play tic-tac-toe that learned something much more like a state-value function. If we look closely at that example, it becomes apparent that the function learned there is neither an action-value function nor a state-value function in the usual sense. A conventional state-value function evaluates states in which the agent has the option of selecting an action, but the state-value function used in tic-tac-toe evaluates board positions after the agent has made its move. Let us call these afterstates, and value functions over these, afterstate value functions. Afterstates are useful when we have knowledge of an initial part of the environment’s dynamics but not necessarily of the full dynamics. For example, in games we typically know the immediate effects of our moves. We know for each possible chess move what the resulting position will be, but not how our opponent will reply. Afterstate value functions are a natural way to take advantage of this kind of knowledge and thereby produce a more efficient learning method. 

The reason it is more efficient to design algorithms in terms of afterstates is apparent from the tic-tac-toe example. A conventional action-value function would map from positions and moves to an estimate of the value. But many position–move pairs produce the same resulting position, as in this example: 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/14decc69bc475dfffc50d75bcc2bac195a83b6f04e70e2765406a0e154c0066c.jpg) 

In such cases the position–move pairs are different but produce the same “afterposition,” and thus must have the same value. A conventional action-value function would have to separately assess both pairs, whereas an afterstate value function would immediately assess both equally. Any learning about the position–move pair on the left would immediately transfer to the pair on the right. 

Afterstates arise in many tasks, not just games. For example, in queuing tasks there are actions such as assigning customers to servers, rejecting customers, or discarding information. In such cases the actions are in fact defined in terms of their immediate effects, which are completely known. For example, in the access-control queuing example described in the previous section, a more efficient learning method could be obtained by breaking the environment’s dynamics into the immediate effect of the action, which is deterministic and completely known, and the unknown random processes having to do with the arrival and departure of customers. The afterstates would be the number of free servers after the action but before the random processes had produced the next conventional state. Learning an afterstate value function over the afterstates would enable all actions that produced the same number of free servers to share experience. This should result in a significant reduction in learning time. 

It is impossible to describe all the possible kinds of specialized problems and corresponding specialized learning algorithms. However, the principles developed in this book should apply widely. For example, afterstate methods are still aptly described in terms of generalized policy iteration, with a policy and (afterstate) value function interacting in essentially the same way. In many cases one will still face the choice between on-policy and off-policy methods for managing the need for persistent exploration. 

## 6.7 Summary 
In this chapter we introduced a new kind of learning method, temporaldifference (TD) learning, and showed how it can be applied to the reinforcement learning problem. As usual, we divided the overall problem into a prediction problem and a control problem. TD methods are alternatives to Monte Carlo methods for solving the prediction problem. In both cases, the extension to the control problem is via the idea of generalized policy iteration (GPI) that we abstracted from dynamic programming. This is the idea that approximate policy and value functions should interact in such a way that they both move toward their optimal values. 
One of the two processes making up GPI drives the value function to accurately predict returns for the current policy; this is the prediction problem. The other process drives the policy to improve locally (e.g., to be $\boldsymbol{\varepsilon}$ -greedy) with respect to the current value function. When the first process is based on experience, a complication arises concerning maintaining sufficient exploration. We have grouped the TD control methods according to whether they deal with this complication by using an on-policy or off-policy approach. Sarsa and actor–critic methods are on-policy methods, and Q-learning and R-learning 
are off-policy methods. 
The methods presented in this chapter are today the most widely used reinforcement learning methods. This is probably due to their great simplicity: they can be applied on-line, with a minimal amount of computation, to experience generated from interaction with an environment; they can be expressed nearly completely by single equations that can be implemented with small computer programs. In the next few chapters we extend these algorithms, making them slightly more complicated and significantly more powerful. All the new algorithms will retain the essence of those introduced here: they will be able to process experience on-line, with relatively little computation, and they will be driven by TD errors. The special cases of TD methods introduced in the present chapter should rightly be called one-step, tabular, modelfree TD methods. In the next three chapters we extend them to multistep forms (a link to Monte Carlo methods), forms using function approximation rather than tables (a link to artificial neural networks), and forms that include a model of the environment (a link to planning and dynamic programming). 
Finally, in this chapter we have discussed TD methods entirely within the context of reinforcement learning problems, but TD methods are actually more general than this. They are general methods for learning to make longterm predictions about dynamical systems. For example, TD methods may be relevant to predicting financial data, life spans, election outcomes, weather patterns, animal behavior, demands on power stations, or customer purchases. It was only when TD methods were analyzed as pure prediction methods, independent of their use in reinforcement learning, that their theoretical properties first came to be well understood. Even so, these other potential applications of TD learning methods have not yet been extensively explored. 

# 7 Eligibility Traces 
Eligibility traces are one of the basic mechanisms of reinforcement learning. For example, in the popular TD( $\lambda$ ) algorithm, the $\lambda$ refers to the use of an eligibility trace. Almost any temporal-difference (TD) method, such as Q-learning or Sarsa, can be combined with eligibility traces to obtain a more general method that may learn more efficiently. 
>  资格迹是 RL 的基本机制之一
>  TD ($\lambda$) 算法中，$\lambda$ 指的是使用资格迹，几乎所有的 TD 算法，例如 Q-Learning 或 Sarsa，都可以和资格迹结合，得到更通用和高效的方法

There are two ways to view eligibility traces. The more theoretical view, which we emphasize here, is that they are a bridge from TD to Monte Carlo methods. When TD methods are augmented with eligibility traces, they produce a family of methods spanning a spectrum that has Monte Carlo methods at one end and one-step TD methods at the other. In between are intermediate methods that are often better than either extreme method. In this sense eligibility traces unify TD and Monte Carlo methods in a valuable and revealing way. 
>  有两种方法理解资格迹，从理论的角度，资格迹是 TD 方法和 MC 方法之间的桥梁，TD 方法结合了资格迹之后可以得到一系列方法，MC 方法和一步 TD 方法分别位于这一系列方法的两端，两端之间的就是中间方法，通常比任意一个极端的方法要好
>  从这个意义上说，资格迹统一了 MC 和 TD

The other way to view eligibility traces is more mechanistic. From this perspective, an eligibility trace is a temporary record of the occurrence of an event, such as the visiting of a state or the taking of an action. The trace marks the memory parameters associated with the event as eligible for undergoing learning changes. When a TD error occurs, only the eligible states or actions are assigned credit or blame for the error. Thus, eligibility traces help bridge the gap between events and training information. Like TD methods themselves, eligibility traces are a basic mechanism for temporal credit assignment. 
>  从另一种更加机械的角度看，资格迹是对事件发生的临时记录，例如访问某个状态或执行某个动作的记录
>  迹标记了和事件相关的记忆参数，出现 TD 误差时，只有被标记了有资格的事件 (状态或动作) 会被认为是导致误差的来源，因此资格迹有助于弥合事件和训练信息之间的差距

For reasons that will become apparent shortly, the more theoretical view of eligibility traces is called the forward view, and the more mechanistic view is called the backward view. The forward view is most useful for understanding what is computed by methods using eligibility traces, whereas the backward view is more appropriate for developing intuition about the algorithms themselves. In this chapter we present both views and then establish senses in which they are equivalent, that is, in which they describe the same algorithms from two points of view. As usual, we first consider the prediction problem and then the control problem. That is, we first consider how eligibility traces are used to help in predicting returns as a function of state for a fixed policy (i.e., in estimating $v_{\pi}$ ). Only after exploring the two views of eligibility traces within this prediction setting do we extend the ideas to action values and control methods. 
>  关于资格迹更为理论的观点是前向视角，更为机械的观点是后向视角
>  我们首先考虑预测问题，然后考虑控制问题，即我们首先考虑资格迹是如何被用于帮助特定策略 $\pi$ 估计其状态价值函数 $v_\pi$ 的，然后拓展到动作价值函数和控制方法

## 7.1 n-Step TD Prediction 
What is the space of methods lying between Monte Carlo and TD methods? Consider estimating $v_{\pi}$ from sample episodes generated using $\pi$ . Monte Carlo methods perform a backup for each state based on the entire sequence of observed rewards from that state until the end of the episode. The backup of simple TD methods, on the other hand, is based on just the one next reward, using the value of the state one step later as a proxy for the remaining rewards. One kind of intermediate method, then, would perform a backup based on an intermediate number of rewards: more than one, but less than all of them until termination. For example, a two-step backup would be based on the first two rewards and the estimated value of the state two steps later. Similarly, we could have three-step backups, four-step backups, and so on. Figure 7.1 diagrams the spectrum of $n$ -step backups for $v_{\pi}$ , with the one-step, simple TD backup on the left and the up-until-termination Monte Carlo backup on the right. 
>  考虑从使用 $\pi$ 生成的回合中估计 $v_\pi$，MC 方法基于从当前状态到回合结束观察到的整个奖励序列进行一次回溯更新，而简单 TD 方法的回溯更新则仅基于下一个奖励，使用后续状态的价值作为剩余奖励的替代执行回溯更新
>  那么，一种中间方法就基于中间数量 (多于 1 个，但少于直到终止前的全部) 的奖励进行一次回溯更新

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/1f78ce19c3f770c140c1a709e501420c3f40790afac764b17d3199b6f4787ef8.jpg) 

Figure 7.1: The spectrum ranging from the one-step backups of simple TD methods to the up-until-termination backups of Monte Carlo methods. In between are the $n$ -step backups, based on $n$ steps of real rewards and the estimated value of the $n$ th next state, all appropriately discounted. 

The methods that use $n$ -step backups are still TD methods because they still change an earlier estimate based on how it differs from a later estimate. Now the later estimate is not one step later, but $n$ steps later. Methods in which the temporal difference extends over $n$ steps are called $n$ -step TD methods. The TD methods introduced in the previous chapter all use one-step backups, and henceforth we call them one-step TD methods. 
>  使用 n 步回溯更新的方法仍然属于 TD 方法，因为它们仍然基于之后的估计值来该表之前的估计值
>  而此时之后的估计不再是一步之后，而是 n 步之后，时间差拓展到 n 步的 TD 称为 n 步 TD 方法

More formally, consider the backup applied to state $S_{t}$ as a result of the state–reward sequence, $S_{t},R_{t+1},S_{t+1},R_{t+2},...,R_{T},S_{T}$ (omitting the actions for simplicity). We know that in Monte Carlo backups the estimate of $v_{\pi}(S_{t})$ is updated in the direction of the complete return: 

$$
G_{t}=R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\cdot\cdot\cdot+\gamma^{T-t-1}R_{T},
$$ 
where $T$ is the last time step of the episode. 

>  在 MC 回溯更新中，$v_\pi(S_t)$ 的估计按照完整回报更新，公式如上，其中 $T$ 指回合的最后一个时间步

Let us call this quantity the target of the backup. Whereas in Monte Carlo backups the target is the return, in one-step backups the target is the first reward plus the discounted estimated value of the next state: 

$$
R_{t+1}+\gamma V_{t}(S_{t+1}),
$$ 
where $V_{t}:\mathcal{S}\rightarrow\mathbb{R}$ here is the estimate at time $t$ of $v_{\pi}$ , in which case it makes sense that $\gamma V_{t}(S_{t+1})$ should take the place of the remaining terms $\gamma R_{t+2}+$ $\gamma^{2}R_{t+3}+\dots+\gamma^{T-t-1}R_{T}$ , as we discussed in the previous chapter. Our point now is that this idea makes just as much sense after two steps as it does after one. 

>  我们称 $G_t$ 为回溯更新的目标，在 MC 中，目标就是实际的回报，在一步更新中，目标就是第一个奖励加上乘以折扣的下一个状态的估计价值，其中 $V_t$ 指 $t$ 时刻对 $v_\pi$ 的估计

The target for a two-step backup might be 

$$
R_{t+1}+\gamma R_{t+2}+\gamma^{2}V_{t}(S_{t+2}),
$$ 
where now $\gamma^{2}V_{t}(S_{t+2})$ corrects for the absence of the terms $\gamma^{2}R_{t+3}+\gamma^{3}R_{t+4}+$ $\cdot\cdot\cdot+\gamma^{T-t-1}R_{T}$ . Similarly, the target for an arbitrary $n$ -step backup might be 

$$
R_{t+1}+\gamma R_{t+2}+\gamma^{2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^{n}V_{t}(S_{t+n}),\quad\forall n\geq1.\tag{7.1}
$$ 
All of these can be considered approximate returns, truncated after $n$ steps and then corrected for the remaining missing terms, in the above case by $V_{t}(S_{t+n})$ . 

>  对于任意的 n 步回溯更新，其目标形式为 Eq 7.1 所示，这些目标都可以视作近似回报，使用 $V_t(S_{t+n})$ 近似了 $n$ 步之后的回报

Notationally, it is useful to retain full generality for the correction term. We define the general $n$ -step return as 

$$
G_{t}^{t+n}(c)=R_{t+1}+\gamma R_{t+2}+\cdot\cdot\cdot+\gamma^{n-1}R_{h}+\gamma^{n}c,
$$ 
for any $n\geq1$ and any scalar correction $c\in\mathbb R$ . The time $h=t+n$ is called the horizon of the $n$ -step return. 

>  符号上来说，保留校正项的完全一般性是有用的。对于任意的 $n \geq 1$ 和任意的标量校正 $c \in \mathbb{R}$，我们定义一般的 $n$ 步回报如上。时间 $h = t + n$ 被称为 $n$ 步回报的视界 (horizon)。

If the episode ends before the horizon is reached, then the truncation in an $n$ -step return effectively occurs at the episode’s end, resulting in the conventional complete return. In other words, if $h\geq T$ , then $G_{t}^{h}(c)=G_{t}$ . Thus, the last $n$ $n$ -step returns of an episode are always complete returns, and an infinite-step return is always a complete return. This definition enables us to treat Monte Carlo methods as the special case of infinite-step targets. All of this is consistent with the tricks for treating episodic and continuing tasks equivalently that we introduced in Section 3.4. There we chose to treat the terminal state as a state that always transitions to itself with zero reward. Under this trick, all $n$ -step returns that last up to or past termination have the same value as the complete return. 
>  如果 $h\ge T$，则 $G_t^h(c) = G_t$
>  我们将终止状态视作一个总是以零奖励转移到其自身的状态，那么所有持续到或者超过终止状态的 n 步回报在定义上总是得到和完整回报相同的值

An $n$ -step backup is defined to be a backup toward the $n$ -step return. In the tabular, state-value case, the $n$ -step backup at time $t$ produces the following increment $\Delta_{t}(S_{t})$ in the estimated value $V(S_{t})$ : 

$$
\Delta_{t}(S_{t})=\alpha\Big[G_{t}^{t+n}(V_{t}(S_{t+n}))-V_{t}(S_{t})\Big],\tag{7.2}
$$ 
where $\alpha$ is a positive step-size parameter, as usual. The increments to the estimated values of the other states are defined to be zero ( $\Delta_{t}(s)=0,\forall s\neq S_{t}$ ). 

>  n 步回溯更新定义为基于 n 步回报的回溯更新，基于之前的定义，n 步回溯中，TD 目标定义为 $G_t^{t+n}(V_t(S_{t+n}))$，TD 目标可以进而用于定义增量，其中 $\alpha$ 为正的步长因子
>  注意，如果 $t$ 时刻的状态 $S_t$ 不是 $s$，则 $\Delta_t(s)$ 定义为零

We define the $n$ -step backup in terms of an increment, rather than as a direct update rule as we did in the previous chapter, in order to allow different ways of making the updates. In on-line updating, the updates are made during the episode, as soon as the increment is computed. In this case we write 

$$
V_{t+1}(s)=V_{t}(s)+\Delta_{t}(s),\qquad\forall s\in\mathbb{S}.\tag{7.3}
$$ 
On-line updating is what we have implicitly assumed in most of the previous two chapters. 

>  我们根据 TD 目标计算增量，进而执行更新，在在线更新的设定下，我们计算完增量就直接进行更新

In off-line updating, on the other hand, the increments are accumulated “on the side” and are not used to change value estimates until the end of the episode. In this case, the approximate values $V_{t}(s),\forall s\in\mathcal{S}$ , do not change during an episode and can be denoted simpty $V(s)$ . At the end of the episode, the new value (for the next episode) is obtained by summing all the increments during the episode. That is, for an episode starting at time step 0 and terminating at step $^{\prime}T^{\prime}$ , for all $s\in\mathcal{S}$ : 

$$
\begin{align}
V_{t+1}(s)&=V_t(s), \quad \forall t < T\\
V_T(s)&=V_{T-1}(s) + \sum_{t=0}^{T-1}\Delta_t(s)
\end{align}\tag{7.4}
$$

with of course $V_{0}$ of the next episode being the $V_{T}$ of this one. You may recall how in Section 6.3 we carried this idea one step further, deferring the increments until they could be summed over a whole set of episodes, in batch updating. 

>  在离线更新的设定下，增量则会先累积，直到回合结束之前，在回合内的所有时间步上，近似价值函数 $V_t(s)$ 都不会更新
>  在回合结束后，价值函数根据回合内的全部增量更新，如 Eq 7.4 所示
>  下一个回合基于更新后的价值函数继续迭代
>  (这和之前的差异就是是要走完一个回合再更新价值函数还是每走一步就更新一次价值函数)
>  甚至还可以进一步推迟更新，在经过适当的几个回合之后再进行更新

For any value function $v:\mathcal{S}\rightarrow\mathbb{R}$ , the expected value of the $n$ -step return using $\upsilon$ is guaranteed to be a better estimate of $v_{\pi}$ than $v$ is, in a worst-state sense. That is, the worst error under the new estimate is guaranteed to be less than or equal to $\gamma^{n}$ times the worst error under $\upsilon$ : 

$$
\operatorname*{max}_{s}\left|\mathbb{E}_{\boldsymbol{\pi}}\big[G_{t}^{t+n}({v}(S_{t+n}))\big|S_{t}=s\big]-v_{\pi}(s)\right|\le\gamma^{n}\operatorname*{max}_{s}\left|{v}(s)-v_{\pi}(s)\right|,
$$

for all $n\geq1$ . This is called the error reduction property of $n$ -step returns. Because of the error reduction property, one can show formally that on-line and off-line TD prediction methods using $n$ -step backups converge to the correct predictions under appropriate technical conditions. The $n$ -step TD methods thus form a family of valid methods, with one-step TD methods and Monte Carlo methods as extreme members. 

>  对于任意的状态价值函数 $v$ ( $v_\pi$ 的近似)，使用 $v$ 的 $n$ 步回报的期望值在最差的状态下都保证是对 $v_\pi$ 的更好的估计
>  也就是说，对于所有的 $n\ge 1$ ($n$ 是 TD 步数)，新估计的最坏误差都保证不高于当前的最坏误差的 $\gamma^n$ 倍
>  (直观上看，上述公式的 LHS 中包含了更多的真实观察到的奖励，故会更接近真实值)
>  这被称为 $n$ 步回报的误差减少性质，因为误差减少性质的存在，我们可以形式地证明使用 $n$ 步回溯更新的在线和离线 TD 方法都会收敛到正确的结果 (即真实值 $v_\pi$)

Nevertheless, $n$ -step TD methods are rarely used because they are inconvenient to implement. Computing $n$ -step returns requires waiting $n$ steps to observe the resultant rewards and states. For large $n$ , this can become problematic, particularly in control applications. The significance of $n$ -step TD methods is primarily for theory and for understanding related methods that are more conveniently implemented. In the next few sections we use the idea of $n$ -step TD methods to explain and justify eligibility trace methods. 
>  实践中，$n$ 步 TD 方法较少使用，因为它们不易于实现，计算 $n$ 步回报要求等待 $n$ 步来观察结果的奖励和状态，对于较大的 $n$，这可能会成为问题

Example 7.1: $n$ -step TD Methods on the Random Walk Consider using $n$ -step TD methods on the random walk task described in Example 6.2 and shown in Figure 6.5. Suppose the first episode progressed directly from the center state, $\mathsf{C}$ , to the right, through D and $\mathsf{E}$ , and then terminated on the right with a return of 1. Recall that the estimated values of all the states started at an intermediate value, $V_{0}(s)=0.5$ . As a result of this experience, a onestep method would change only the estimate for the last state, $V(\mathsf E)$ , which would be incremented toward 1, the observed return. A two-step method, on the other hand, would increment the values of the two states preceding termination: $V(\mathsf{D})$ and $V(\mathsf E)$ both would be incremented toward 1. A threestep method, or any $n$ -step method for $n>2$ , would increment the values of all three of the visited states toward 1, all by the same amount. Which $n$ is better? Figure 7.2 shows the results of a simple empirical assessment for a larger random walk process, with 19 states (and with a $-1$ outcome on the left, all values initialized to $0$ ). Results are shown for on-line and off-line $n$ -step TD methods with a range of values for $n$ and $\alpha$ . The performance measure for each algorithm and parameter setting, shown on the vertical axis, is the square-root of the average squared error between its predictions at the end of the episodenfor the 19 states and their true values, then averaged over the first 10 episodes and 100 repetitions of the whole experiment (the same sets of walks were used for all methods). First note that the on-line methods generally worked best on this task, both reaching lower levels of absolute error and doing so over a larger range of the step-size parameter $\alpha$ (in fact, all the off-line methods were unstable for $\alpha$ much above 0.3). Second, note that methods with an intermediate value of $n$ worked best. This illustrates how the generalization of TD and Monte Carlo methods to $n$ -step methods can potentially perform better than either of the two extreme methods. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/fbad28c4a0c8459e92007e5f9bfa7b7a7d832bd2664703a44c5370e2d40d4bde.jpg) 
Figure 7.2: Performance of $n$ -step TD methods as a function of $\alpha$ , for various values of $n$ , on a 19-state random walk task (Example 7.1). 

## 7.2 The Forward View of TD($\lambda$) 
Backups can be done not just toward any $n$ -step return, but toward any average of $n$ -step returns. For example, a backup can be done toward a target that is half of a two-step return and half of a four-step return: $\begin{array}{r}{\frac{1}{2}G_{t}^{t+2}(V_{t}(S_{t+2}))+}\end{array}$ ${\textstyle\frac{1}{2}}G_{t}^{t+4}(V_{t}(S_{t+4}))$ . Any set of returns can be averaged in this way, even an infinite set, as long as the weights on the component returns are positive and sum to 1. The composite return possesses an error reduction property similar to that of individual $n$ -step returns (7.5) and thus can be used to construct backups with guaranteed convergence properties. Averaging produces a substantial new range of algorithms. For example, one could average one-step and infinite-step returns to obtain another way of interrelating TD and Monte Carlo methods. In principle, one could even average experience-based backups with DP backups to get a simple combination of experience-based and model-based methods (see Chapter 8). 
>  回溯更新不仅限于使用任意的 $n$ 步回报，也可以使用 $n$ 步回报的平均，例如，可以使用一半的两步回报和一半的四步回报进行回溯更新
>  这样的复合回报具有类似于单个 $n$ 步回报同样的误差减少特性，因此可以用于构建具有收敛性保证的回溯更新算法
>  这样的平均计算衍生了许多新算法，例如，可以通过平均一步回报和无限步回报来相互关联 MC 和 TD
>  原则上，可以将基于经验的回溯更新和基于 DP 的回溯更新结合，以结合基于经验的和基于模型的方法

A backup that averages simpler component backups is called a complex backup. The backup diagram for a complex backup consists of the backup diagrams for each of the component backups with a horizontal line above them and the weighting fractions below. For example, the complex backup for the case mentioned at the start of this section, mixing half of a two-step backup and half of a four-step backup, has the diagram: 
>  平均了较简单的回溯更新成分的回溯更新成为复杂回溯更新
>  复杂回溯更新的回溯更新图由每个成员回溯更新的回溯更新图组成，其上方有一条横线，下方是加权分数

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/78cd9456ee57655ba8319c6588a525bdfc0a5b2223e1b17bf8a40c89ddfb5706.jpg) 

The TD( $\lambda$ ) algorithm can be understood as one particular way of averaging $n$ -step backups. This average contains all the $n$ -step backups, each weighted proportional to $\lambda^{n-1}$ , where $\lambda\in[0,1]$ , and normalized by a factor of $1-\lambda$ to ensure that the weights sum to $1$ (see Figure 7.3). The resulting backup is toward a return, called the $\lambda$ -return, defined by 

$$
L_{t}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t}^{t+n}(V_{t}(S_{t+n})).
$$

>  TD($\lambda$) 算法可以理解为 $n$ 步回溯更新的一种特定的加权方式，它包含了所有的 $n$ 步回溯更新值 (每个回溯更新值都是对真实回报 $G_t$ 的一个近似)，其中每个值以正比于 $\lambda^{n-1}$ 的权重加权，其中 $\lambda\in [0, 1]$，并且整体乘以因子 $(1-\lambda)$ 进行规范化，确保权重值和为 1
>  最后得到的回溯更新值称为 $\lambda$ 回报，定义如上

>  推导：权重值和为 1
>  容易知道 $\sum_{n=1}^\infty \lambda^{n-1}$ 是一个从 $n=1$ 开始的无穷等比数列，根据等比数列的求和公式，容易知道 $\sum_{n=1}^\infty \lambda^{n-1} = \frac {\lambda^\infty - \lambda^0}{\lambda-1} = \frac {-1}{\lambda - 1}$，故 $(1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} = 1$

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/73d15475f5fb6d6be49a71befda7b8f213fcbc4ea4c4dc874177b9161635965c.jpg) 

Figure 7.3: The backup diagram for $\mathrm{TD}(\lambda)$ . If $\lambda=0$ , then the overall backup reduces to its first component, the one-step TD backup, whereas if $\lambda=1$ , then the overall backup reduces to its last component, the Monte Carlo backup. 

Figure 7.4 further illustrates the weighting on the sequence of $n$ -step returns in the $\lambda$ -return. The one-step return is given the largest weight, $1-\lambda$ ; the two-step return is given the next largest weight, $(1-\lambda)\lambda$ ; the three-step return is given the weight $(1-\lambda)\lambda^{2}$ ; and so on. The weight fades by $\lambda$ with each additional step. After a terminal state has been reached, all subsequent $n$ -step returns are equal to $G_{t}$ . If we want, we can separate these post-termination terms from the main sum, yielding 

$$
L_{t}~=~(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t}^{t+n}(V_{t}(S_{t+n}))~+~\lambda^{T-t-1}G_{t},\tag{7.6}
$$ 
as indicated in the figures. This equation makes it clearer what happens when $\lambda=1$ . In this case the main sum goes to zero, and the remaining term reduces to the conventional return, $G_{t}$ . Thus, for $\lambda=1$ , backing up according to the $\lambda$ -return is the same as the Monte Carlo algorithm that we called constant- $\alpha$ MC (6.1) in the previous chapter. On the other hand, if $\lambda\:=\:0$ , then the $\lambda$ -return reduces to $G_{t}^{t+1}(V_{t}(S_{t+1}))$ , the one-step return. Thus, for $\lambda\:=\:0$ , backing up according to the $\lambda$ -return is the same as the one-step TD method, TD(0) from the previous chapter (6.2). 

>  $\lambda$ 回报中，$n$ 步回报序列的权重如 Figure 7.4 所示，可以看到一步回报具有最大的权重 $1-\lambda$，之后权重以 $\lambda$ 的比率衰减
>  考虑到终止时间步 $T$，我们进一步完善定义，将终止时间步上的回报 $G_t$ 分离出来，最后得到的 $\lambda$ 回报如 Eq 7.6 所示
>  根据 Eq 7.6，当 $\lambda = 1$ 时， $\lambda$ 回报就是 MC 回报 $G_t$，此时 TD($\lambda$) 就等价于 MC 算法，当 $\lambda = 0$ 时，$\lambda$ 回报就是 1 步回报 $G_t^{t+n}(V_t(S_{t+n}))$ ，此时 TD($\lambda$) 就等价于一步 TD 方法，或者称为 TD(0)

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/f33c47933fcecfbf54f757dfac77c9767948f8f63b8db154090c1ea4339569be.jpg) 
Figure 7.4: Weighting given in the $\lambda$ -return to each of the $n$ -step returns. 

We define the $\lambda$ -return algorithm as the method that performs backups towards the $\lambda$ -return as target. On each step, $t$ , it computes an increment, $\Delta_{t}(S_{t})$ , to the value of the state occurring on that step: 

$$
\Delta_{t}(S_{t})=\alpha\Big[L_{t}-V_{t}(S_{t})\Big].\tag{7.7}
$$ 
(The increments for other states are of course $\Delta_{t}(s)~=~0$ , for all $s\neq S_{t}$ .) 

>  我们将 $\lambda$ 回报算法定义为以 $\lambda$ 回报为目标执行回溯更新的算法，在每一步 $t$，该算法根据 Eq 7.7 计算增量

As with $n$ -step TD methods, the updating can be either on-line or off-line. The upper row of Figure 7.6 shows the performance of the on-line and offline $\lambda$ -return algorithms on the 19-state random walk task (Example 7.1). The experiment was just as in the $n$ -step case (Figure 7.2) except that here we varied $\lambda$ instead of $n$ . Note that overall performance of the $\lambda$ -return algorithms is comparable to that of the $n$ -step algorithms. In both cases we get best performance with an intermediate value of the truncation parameter, $n$ or $\lambda$ . 
>  同样，更新可以是在线也可以是离线
>  $\lambda$ 回报算法的整体性能和 $n$ 步回报算法相当，两个算法一般都在截断参数 $\lambda$ 或 $n$ 取中间值时获得最佳性能

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/4343c691b163fe5ebd9e5e6ba6990fa067a9509c3a82ea44591ca657ca60ddfe.jpg) 

Figure 7.5: The forward or theoretical view. We decide how to update each state by looking forward to future rewards and states. 

The approach that we have been taking so far is what we call the theoretical, or forward, view of a learning algorithm. For each state visited, we look forward in time to all the future rewards and decide how best to combine them. We might imagine ourselves riding the stream of states, looking forward from each state to determine its update, as suggested by Figure 7.5. After looking forward from and updating one state, we move on to the next and never have to work with the preceding state again. Future states, on the other hand, are viewed and processed repeatedly, once from each vantage point preceding them. 
>  目前我们讨论的都是学习算法的前向视角，对于每个访问到的状态，我们会向前看未来的所有奖励，然后决定如何最好地结合它们
>  在从一个状态向前看并完成更新后，我们转向下一个状态，不再处理前面的状态，另一方面，未来的状态会被反复观察和处理

The $\lambda$ -return algorithm is the basis for the forward view of eligibility traces as used in the $\mathrm{TD}(\lambda)$ method. In fact, we show in a later section that, in the off-line case, the $\lambda$ -return algorithm is the TD( $\lambda$ ) algorithm. The $\lambda$ -return and $\mathrm{TD}(\lambda)$ methods use the $\lambda$ parameter to shift from one-step TD methods to Monte Carlo methods. The specific way this shift is done is interesting, but not obviously better or worse than the way it is done with simple $n$ -step methods by varying $n$ . Ultimately, the most compelling motivation for the $\lambda$ way of mixing $n$ -step backups is that there in a simple algorithm—TD(λ)—for achieving it. This is a mechanism issue rather than a theoretical one. In the next few sections we develop the mechanistic, or backward, view of eligibility traces as used in TD( $\lambda$ ). 
>  在离线更新的情况下，$\lambda$ 回报算法就是 TD($\lambda$) 算法，$\lambda$ 回报算法和 TD($\lambda$) 算法都使用 $\lambda$ 参数在一步 TD 方法和 MC 方法之间切换

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/ec0b5546b05e45c30fe0f8631cb7a5bfd9cc4373a1349c8fea8c630bca316737.jpg) 
Figure 7.6: Performance of all $\lambda$ -based algorithms on the 19-state random walk (Example 7.1). The $\lambda=0$ line is the same for all five on-line algorithms. 

## 7.3 The Backward View of TD($\lambda$) 
In the previous section we presented the forward or theoretical view of the tabular TD( $\lambda$ ) algorithm as a way of mixing backups that parametrically shifts from a TD method to a Monte Carlo method. 

In this section we instead define $\mathrm{TD}(\lambda)$ mechanistically and show that it can closely approximate the forward view. The mechanistic, or backward, view of $\mathrm{TD}(\lambda)$ is useful because it is simple conceptually and computationally. In particular, the forward view itself is not directly implementable because it is acausal, using at each step knowledge of what will happen many steps later. The backward view provides a causal, incremental mechanism for approximating the forward view and, in the off-line case, for achieving it exactly. 
>  前向视角的 TD($\lambda$) 在计算视角上是不可直接实现的，因为它是反因果的，它在每一步使用了多步之后才会发生的信息
>  后向视角则提供了一种因果性的，增量的机制来近似前向视角，且在离线情况下可以精确地符合前向视角 (因为离线情况下，整个 episode 的情况是全部已知的，就无所谓因果和反因果了)

In the backward view of $\mathrm{TD}(\lambda)$ , there is an additional memory variable associated with each state, its eligibility trace. The eligibility trace for state $s$ at time $t$ is a random variable denoted $E_{t}(s)\in\mathbb{R}^{+}$ . On each step, the eligibility traces of all non-visited states decay by $\gamma\lambda$ : 

$$
E_{t}(s)=\gamma\lambda E_{t-1}(s),\qquad\forall s\in\mathcal S,s\neq S_{t},\tag{7.8}
$$

where $\gamma$ is the discount rate and $\lambda$ is the parameter introduced in the previous section. Henceforth we refer to $\lambda$ as the trace-decay parameter.

>  TD ($\lambda$) 的后向视角中，每个状态关联一个额外的记忆变量，称为资格迹
>  状态 $s$ 在 $t$ 时刻的资格迹是一个随机变量，记作 $E_t(s)\in \mathbb R^+$，每经过一个时间步，所有未被访问的状态的资格迹按照 $\gamma \lambda$ 衰减，如 Eq 7.8 所示
>  $\lambda$ 也被称作迹衰减参数

What about the trace for $S_{t}$ , the one state visited at time $t$ ? The classical eligibility trace for $S_{t}$ decays just like for any state, but is then incremented by 1: 

$$
E_{t}(S_{t})=\gamma\lambda E_{t-1}(S_{t})+1.\tag{7.9}
$$  
>  对于在 $t$ 时刻访问到的状态 $S_t$，经典的方式是先进行衰减，然后加 1

This kind of eligibility trace is called an accumulating trace because it accumulates each time the state is visited, then fades away gradually when the state is not visited, as illustrated as illustrated below. 
>  这类资格迹称为累积迹，因为它在每次状态被访问到的时候累积，然后在状态不被访问的时候逐渐消失

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/208ca742d813f022921cf0b64012c257f220f5091917460b6e4ff1fed954eea8.jpg) 


Eligibility traces keep a simple record of which states have recently been visited, where “recently” is defined in terms of $\gamma\lambda$ . The traces are said to indicate the degree to which each state is eligible for undergoing learning changes should a reinforcing event occur. The reinforcing events we are concerned with are the moment-by-moment one-step TD errors.
>  资格迹记录了最近访问的状态，“最近” 是根据 $\gamma\lambda$ 定义的
>  状态的资格迹表示了该状态在强化事件中在多大程度上有资格经历学习变化，我们关注的强化事件是逐时刻根据一步 TD 误差优化
>  (直观来看，使用了资格迹之后，在每一个时间步，我们不仅更新当前访问状态的价值，还一定程度上更新之前访问过的状态的价值，可以认为当前时刻的 TD 误差不仅仅只和当前状态相关，和之前访问过的状态也有一定关联，故也可以用于更新它们的价值)

For example, the TD error for state-value prediction is 

$$
\delta_{t}=R_{t+1}+\gamma V_{t}(S_{t+1})-V_{t}(S_{t}).\tag{7.10}
$$ 
In the backward view of $\mathrm{TD}(\lambda)$ , the global TD error signal triggers proportional updates to all recently visited states, as signaled by their nonzero traces: 

$$
\Delta V_{t}(s)=\alpha\delta_{t}E_{t}(s),\qquad{\mathrm{for~all~}}s\in{\mathcal{S}}.\tag{7.11}
$$

>  例如，状态价值预测的一步 TD 误差如 Eq 7.10 所示
>  在 TD ($\lambda$) 的后向视角中，每一步的 TD 误差会用于对所有最近访问过的状态的价值更新，也就是说，各个状态 $s$ 在 $t$ 时刻的价值增量定义如 Eq 7.11 所示，它正比于 $t$ 时刻的 TD 误差乘上 $s$ 的资格迹
>  根据 Eq 7.11，如果 $s$ 最近被访问过，则该增量就不会是零，如果 $s$ 很久未被访问过，则该增量基本就是零

As always, these increments could be done on each step to form an on-line algorithm, or saved until the end of the episode to produce an off-line algorithm. In either case, equations (7.8–7.11) provide the mechanistic definition of the $\mathrm{TD}(\lambda)$ algorithm. A complete algorithm for on-line TD ( $\lambda$ ) is given in Figure 7.7. 
>  这一增量可以立刻在当前时间步用于价值更新，即在线更新，或者保存下来，知道回合结束进行离线更新
>  Eq 7.8 - Eq 7.11 为 TD($\lambda$) 算法提供了机械的定义，完整的在线 TD($\lambda$) 见 Figure 7.7

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/da98bd72fad1ae2ea165ee67608e38bf7a3d77de2c2357c31f0dfc538c596b22.jpg) 

Figure 7.7: On-line tabular $\mathrm{TD}(\lambda)$ . 

The backward view of TD( $\lambda$ ) is oriented backward in time. At each moment we look at the current TD error and assign it backward to each prior state according to the state’s eligibility trace at that time. We might imagine ourselves riding along the stream of states, computing TD errors, and shouting them back to the previously visited states, as suggested by Figure 7.8. Where the TD error and traces come together, we get the update given by (7.11). 
>  TD ($\lambda$) 的反向视角是在时间上反向，在每一时刻，我们观察当前的 TD 误差，然后根据所有状态在当前时刻的资格迹反向将给误差分配给之前访问过的所有状态 (如果没有访问过的话，则其资格迹还是初始值 0)


![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/97f79aae0872280dc210797bcd585d677eb960cc9ea6a65160b234e88b5b3652.jpg) 

Figure 7.8: The backward or mechanistic view. Each update depends on the current TD error combined with eligibility traces of past events. 

To better understand the backward view, consider what happens at various values of $\lambda$ . If $\lambda=0$ , then by (7.9) all traces are zero at $t$ except for the trace corresponding to $S_{t}$ . Thus the $\mathrm{TD}(\lambda)$ update (7.11) reduces to the simple TD rule (6.2), which we henceforth call TD(0). In terms of Figure 7.8, TD(0) is the case in which only the one state preceding the current one is changed by the TD error. For larger values of $\lambda$ , but still $\lambda<1$ , more of the preceding states are changed, but each more temporally distant state is changed less because its eligibility trace is smaller, as suggested in the figure. We say that the earlier states are given less credit for the TD error. 
>  考虑不同 $\lambda$ 值的情况，当 $\lambda = 0$，根据 Eq 7.9，在 $t$ 时刻，所有的迹都是 0，除了状态 $S_t$ 的迹是 1，此时 Eq 7.11 描述的 TD ($\lambda$) 更新等价于简单的 TD 规则，即 TD (0)，也就是只更新当前状态的价值
>  如果 $\lambda$ 较大，则会有更多的前序状态被改变，当然越早的状态的资格迹一般越小，故获得的增量越小

If $\lambda=1$ , then the credit given to earlier states falls only by $\gamma$ per step. This turns out to be just the right thing to do to achieve Monte Carlo behavior. For example, remember that the TD error, $\delta_{t}$ , includes an undiscounted term of $R_{t+1}$ . In passing this back $k$ steps it needs to be discounted, like any reward in a return, by $\gamma^{k}$ , which is just what the falling eligibility trace achieves. If $\lambda=1$ and $\gamma=1$ , then the eligibility traces do not decay at all with time. In this case the method behaves like a Monte Carlo method for an undiscounted, episodic task. If $\lambda=1$ , the algorithm is also known as TD(1). 
>  如果 $\lambda = 1$，则分配给之前状态的误差仅以 $\gamma$ 每步衰减，该算法称为 TD(1)
>  可以证明 TD(1) 正好可以实现 MC 行为，注意 TD 误差 $\delta_t$ 中包含了一个未折扣的奖励 $R_{t+1}$，该奖励相对于 $k$ 步之前的状态应该是要打折扣的，即乘上 $\gamma^k$，这也正是资格迹所实现的效果

TD(1) is a way of implementing Monte Carlo algorithms that is more general than those presented earlier and that significantly increases their range of applicability. Whereas the earlier Monte Carlo methods were limited to episodic tasks, TD(1) can be applied to discounted continuing tasks as well. Moreover, TD(1) can be performed incrementally and on-line. One disadvantage of Monte Carlo methods is that they learn nothing from an episode until it is over. For example, if a Monte Carlo control method does something that produces a very poor reward but does not end the episode, then the agent’s tendency to do that will be undiminished during the episode. On-line TD(1), on the other hand, learns in an $n$ -step TD way from the incomplete ongoing episode, where the $n$ steps are all the way up to the current step. If something unusually good or bad happens during an episode, control methods based on TD(1) can learn immediately and alter their behavior on that same episode. 
>  TD(1) 是一种实现蒙特卡洛算法的方法，比之前介绍的方法更为通用，并且显著扩展了这些方法的应用范围。与之前的蒙特卡洛方法仅限于离散任务不同，TD(1) 可以应用于具有折扣的持续任务。此外，TD(1) 可以增量地和在线地执行。
>  蒙特卡洛方法的一个缺点是它们在结束一个回合之前不会从该回合中学习任何东西。例如，如果一个蒙特卡洛控制方法采取了一个导致非常差的奖励但没有结束回合的行为，那么在这整个回合期间，智能体继续这样做的趋势不会减少。然而，在线 TD(1) 方法则可以在不完整且正在进行的回合中以 n 步时序差分的方式进行学习，其中 n 步一直延伸到当前步骤。如果在回合过程中发生了特别好或特别坏的事情，基于 TD(1) 的控制方法可以立即学习并在这同一回合中改变其行为。

It is revealing to revisit the 19-state random walk example (Example 7.1) to see how well the backward-view $\mathrm{TD}(\lambda)$ algorithm does in approximating the ideal of the forward-view $\lambda$ -return algorithm. The performances of off-line and on-line $\mathrm{TD}(\lambda)$ with accumulating traces are shown in the upper-right and middle-right panels of Figure 7.6. In the off-line case it has been proven that the $\lambda$ -return algorithm and $\mathrm{TD}(\lambda)$ are identical in their overall updates at the end of the episode. Thus, the one set of results in the upper-right panel is sufficient for both of these algorithms. However, recall that the off-line case is not our main focus, as all of its performance levels are generally lower and obtained over a narrower range of parameter values than can be obtained with on-line methods, as we saw earlier for $n$ -step methods in Figure 7.2 and for $\lambda$ -return methods in the upper two panels of Figure 7.6. 
>  在离线情况下，$\lambda$ 回报算法和 TD($\lambda$) 算法是等价的，但离线算法不是我们的主要关注点，因为离线算法的表现一般差于在线算法

In the on-line case, the performances of $\mathrm{TD}(\lambda)$ with accumulating traces (middle-right panel) are indeed much better and closer to that of the on-line $\lambda$ -return algorithm (upper-left panel). If $\lambda=0$ , then in fact it is the identical algorithm at all $\alpha$ , and if $\alpha$ is small, then for all $\lambda$ it is a close approximation to the $\lambda$ -return algorithm by the end of each episode. However, if both parameters are larger, for example $\lambda>0.9$ and $\alpha>0.5$ , then the algorithms perform substantially differently: the $\lambda$ -return algorithm performs a little less well whereas $\mathrm{TD}(\lambda)$ is likely to be unstable. This is not a terrible problem, as these parameter values are higher than one would want to use anyway, but it is a weakness of the method. 
>  在线情况下，使用累积迹的 TD ($\lambda$) 算法和 $\lambda$ 回报算法的性能十分接近，如果 $\lambda = 0$，二者等价；如果 $\alpha$ 很小，则对于所有的 $\lambda$ 值，在每个 episode 结束时它都是对 $\lambda$ -回报算法的一个很好的近似。
>  然而，如果两个参数都较大，例如 $\lambda>0.9$ 和 $\alpha>0.5$，那么这些算法的表现会有显著差异：$\lambda$ -回报算法的表现略差一些，而 $\mathrm{TD}(\lambda)$ 则可能不稳定。这并不是一个可怕的问题，因为这些参数值本来就高于实际想要使用的值，但它确实是该方法的一个弱点。

Two alternative variations of eligibility traces have been proposed to address these limitations of the accumulating trace. On each step, all three trace types decay the traces of the non-visited states in the same way, that is, according to (7.8), but they differ in how the visited state is incremented. The first trace variation is the replacing trace. Suppose a state is visited and then revisited before the trace due to the first visit has fully decayed to zero. With accumulating traces the revisit causes a further increment in the trace (7.9), driving it greater than 1, whereas, with replacing traces, the trace is simply reset to 1: 

$$
E_{t}(S_{t})=1.
$$ 
In the special case of $\lambda=1$ , $\mathrm{TD}(\lambda)$ with replacing traces is closely related to first-visit Monte Carlo methods

>  两种替代的 eligibility traces 变体被提出以解决累积迹的这些局限性。在每一步中，所有三种迹类型以相同的方式衰减未访问状态的迹，即根据公式（7.8），但它们在如何增加访问状态方面有所不同。
>  第一种迹变化称为替换迹。假设一个状态被访问，然后在其第一次访问的迹尚未完全衰减到零之前再次被访问。使用累积迹时，再次访问会导致迹进一步增加（公式7.9），使其大于1；而使用替换迹时，迹直接重置为1
>  在 $\lambda=1$ 的特殊情况下，带有替换迹的 $\mathrm{TD}(\lambda)$ 方法与首次访问蒙特卡洛方法密切相关。

. The second trace variation, called the dutch trace, is sort of intermediate between accumulating and replacing traces, depending on the step-size parameter $\alpha$ . Dutch traces are defined by 

$$
E_{t}(S_{t})=(1-\alpha)\gamma\lambda E_{t-1}(S_{t})+1.
$$ 
that as $\alpha$ approaches zero, the dutch trace becomes the accumulating trace, and, if $\alpha=1$ , the dutch trace becomes the replacing trace. 

>  第二种迹变化称为荷兰痕迹，它介于累积迹和替换迹之间，具体取决于步长参数 $\alpha$。荷兰迹定义如上
>  当 $\alpha$ 接近零时，Dutch 迹变为累积迹，如果 $\alpha=1$，Dutch 迹则变为替换迹。

Figure 7.9 contrasts the three kinds of traces, showing the behavior of the dutch trace for $\alpha=1/2$ . The performances of $\mathrm{TD}(\lambda)$ with these two kinds of traces are shown as additional panels in (7.6). In both cases, performance is more robust to the parameter values than it is with accumulating traces. The performance with dutch traces in particular achieves our goal of an on-line causal algorithm that closely approximates the $\lambda$ -return algorithm. 
>  图 7.9 对比了三种不同类型的迹，展示了当 $\alpha=1/2$ 时 Dutch 迹的行为。


![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/6eada10d694c1136c70661d3b0a499d48038bb3e71ab0ba8cfad79f6092d4bad.jpg) 

Figure 7.9: The three different kinds of traces. Accumulating traces add up each time a state is visited, whereas replacing traces are reset to one, and dutch traces do something inbetween, depending on $\alpha$ (here we show them for $\alpha=0.5$ ). In all cases the traces decay at a rate of $\gamma\lambda$ per step; here we show $\gamma\lambda=0.8$ such that the traces have a time constant of approximately 5 steps (the last four visits are on successive steps). 

## 7.4 Equivalences of Forward and Backward Views 
It is sometimes possible to prove that two learning methods originating in different ways are in fact equivalent in the strong sense that the value functions they produce are exactly the same on every time step. 

A simple case of this is that one-step methods and all $\lambda$ -based methods are equivalent if $\lambda=0$ . This follows immediately from the fact that their backup targets are all the same. Another easy-to-see example is the equivalence at $\lambda=1$ of off-line $\mathrm{TD}(\lambda)$ and the constant- $\alpha$ MC methods, as noted in the previous section. 

Of particular interest are equivalences between forward-view algorithms, which are often more intuitive and clearer conceptually, and backward-view algorithms that are efficient and causal. The best example of this that we have encountered so far is the equivalence at all $\lambda$ of the off-line $\lambda$ -return algorithm (forward view) and off-line $\mathrm{TD}(\lambda)$ with accumulating traces (backward view). That was an equivalence of value functions at the end of episodes and, because they are offline methods which don’t change values within an episode, it is a step-by-step equivalence as well. This equivalence was proved formally in the first edition of this book, and was verified empirically here on the 19-state random-walk example in producing the upper-left panel of Figure 7.6. 

For on-line methods (and $\lambda>0$ ) the first edition of this book established only approximate episode-by-episode equivalences between the $\lambda$ -return algorithm and $\mathrm{TD}(\lambda)$ . In the random-walk problem, at the end of episodes, TD( $\lambda$ ) with accumulating traces is almost the same as the $\lambda$ -return algorithm, but only for small $\alpha$ and $\lambda$ . With dutch traces the approximation is closer, but it is still not exact even on an episode-by-episode basis (compare the upper-left and middle left panels of Figure 7.6). Only recently has an interesting exact equivalence been established between a $\lambda$ -based forward view and an efficient backward-view implementation, in particular, between a “real-time” $\lambda$ -return algorithm and the “true online $\mathrm{TD}(\lambda)^{\mathfrak{N}}$ algorithm (van Seijen and Sutton, 2014). This is a striking and revealing result, but a little technical. The best way to present it is using the notation of linear function approximation, which we develop in Chapter 9. We postpone development of the real-time $\lambda$ -return algorithm until then and present here only the backward-view algorithm. 

True online $T D(\lambda)$ is defined by the dutch trace (Eqs. 7.13 and 7.8) and the following value function update: 

$$
V_{t+1}(s)=V_{t}(s)+\alpha\big[\delta_{t}+V_{t}(S_{t})-V_{t-1}(S_{t})\big]E_{t}(s)-\alpha I_{s S_{t}}\big[V_{t}(S_{t})-V_{t-1}(S_{t})\big],
$$ 
for all $s\in{\mathcal{S}}$ , where $I_{x y}$ is an identity-indicator function, equal to $1$ if $x=y$ and 0 otherwise. An efficient implementation is given as a boxed algorithm in Figure 7.10. 

Results on the 19-state random-walk example for true online $\mathrm{TD}(\lambda)$ are given in the lower-left panel of Figure 7.6. We see that in this example true on-line TD( $\lambda$ ) appears to perform slightly better than the on-line $\lambda$ -return algorithm, but not necessarly better than TD( $\lambda$ ) with dutch traces; most of the performance improvement seems to come from the dutch traces rather than the slightly different or extra terms in the equations above. Of course, this is just one example; benefits of the exact equivalence may appear on other problems. One thing we can say in that these slight differences enable true on-line $\mathrm{TD}(\lambda)$ with $\lambda=1$ to be exactly equivalent by the end of the episode to the constant- $\alpha$ MC method, while making updates on-line and in real-time. The same cannot be said for any of the other methods. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/9ff23f3fd396704e5c3ecb8a175249cfeb63707ba6f1b97b2fdcef82e721db7e.jpg) 

Figure 7.10: Tabular true on-line $\mathrm{TD}(\lambda)$ . 

## 7.5 Sarsa($\lambda$) 
How can eligibility traces be used not just for prediction, as in $\mathrm{TD}(\lambda)$ , but for control? As usual, the main idea of one popular approach is simply to learn action values, $Q_{t}(s,a)$ , rather than state values, $V_{t}(s)$ . In this section we show how eligibility traces can be combined with Sarsa in a straightforward way to produce an on-policy TD control method. The eligibility trace version of Sarsa we call $S a r s a(\lambda)$ , and the original version presented in the previous chapter we henceforth call one-step Sarsa. 
The idea in $\mathrm{Sarsa}(\lambda)$ is to apply the TD( $\lambda$ ) prediction method to state– action pairs rather than to states. Obviously, then, we need a trace not just for each state, but for each state–action pair. Let $E_{t}(s,a)$ denote the trace for state–action pair $s,a$ . The traces can be any of the three types—accumulating, replace, or dutch—and are updated in essentially the same way as before except of course being triggered by visiting the state–action pair (here given using the identity-indicator notation): 
$\begin{array}{r l}&{E_{t}(s,a)=\gamma\lambda E_{t-1}(s,a)+I_{s S_{t}}I_{a A_{t}}}\ &{E_{t}(s,a)=(1-\alpha)\gamma\lambda E_{t-1}(s,a)+I_{s S_{t}}I_{a A_{t}}}\ &{E_{t}(s,a)=(1-I_{s S_{t}}I_{a A_{t}})\gamma\lambda E_{t-1}(s,a)+I_{s S_{t}}I_{a A_{t}}}\end{array}$ (accumulating) (dutch) (replacing) 
for all $s\in\mathcal{S},a\in\mathcal{A}$ . Otherwise $\mathrm{Sarsa}(\lambda)$ is just like $\mathrm{TD}(\lambda)$ , substituting state–action variables for state variables— $Q_{t}(s,a)$ for $V_{t}(s)$ and $E_{t}(s,a)$ for $E_{t}(s)$ : 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/9beb9253cf1c92b8e809abe584017a755eaa3205c1db9903b12e048ba38fa531.jpg) 
Figure 7.11: Sarsa $(\lambda)$ ’s backup diagram. 
$$
Q_{t+1}(s,a)=Q_{t}(s,a)+\alpha\delta_{t}E_{t}(s,a),\qquad{\mathrm{for~all~}}s,a
$$ 
where 
$$
\delta_{t}=R_{t+1}+\gamma Q_{t}(S_{t+1},A_{t+1})-Q_{t}(S_{t},A_{t}).
$$ 
Figure 7.11 shows the backup diagram for $\mathrm{Sarsa}(\lambda)$ . Notice the similarity to the diagram of the $\mathrm{TD}(\lambda)$ algorithm (Figure 7.3). The first backup looks ahead one full step, to the next state–action pair, the second looks ahead two steps, and so on. A final backup is based on the complete return. The weighting of each backup is just as in $\mathrm{TD}(\lambda)$ and the $\lambda$ -return algorithm. 
One-step Sarsa and $\mathrm{Sarsa}(\lambda)$ are on-policy algorithms, meaning that they approximate $q_{\pi}(s,a)$ , the action values for the current policy, $\pi$ , then improve the policy gradually based on the approximate values for the current policy. The policy improvement can be done in many different ways, as we have seen throughout this book. For example, the simplest approach is to use the $\boldsymbol{\varepsilon}$ - greedy policy with respect to the current action-value estimates. Figure 7.12 shows the complete Sarsa(λ) algorithm for this case. 
Example 7.2: Traces in Gridworld The use of eligibility traces can substantially increase the efficiency of control algorithms. The reason for this is illustrated by the gridworld example in Figure 7.13. The first panel shows the path taken by an agent in a single episode, ending at a location of high reward, marked by the $^*$ . In this example the values were all initially 0, and all rewards were zero except for a positive reward at the \* location. The arrows in the other two panels show which action values were strengthened as a result of this path by one-step Sarsa and $\mathrm{Sarsa}(\lambda)$ methods. The one-step method strengthens only the last action of the sequence of actions that led to the high reward, whereas the trace method strengthens many actions of the sequence. The degree of strengthening (indicated by the size of the arrows) falls off (according to $\gamma\lambda$ or $(1-\alpha)\gamma\lambda)$ with steps from the reward. In this example, the fall off is 0.9 per step.  

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/aad9670b2d205db7a8d4a90f9c332f8fc1a81e4276da6d7ec89e73e4083ef1e5.jpg) 
Figure 7.12: Tabular $\mathrm{Sarsa}(\lambda)$ . 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/d544f393e0b2ecbb1fa23ccd9512bc735ab3f6da836cbbc5c6c36857ff738d38.jpg) 
Figure 7.13: Gridworld example of the speedup of policy learning due to the use of eligibility traces. 

## 7.6 Watkins’s $Q(\lambda)$ 
When Chris Watkins (1989) first proposed Q-learning, he also proposed a simple way to combine it with eligibility traces. Recall that Q-learning is an off-policy method, meaning that the policy learned about need not be the same as the one used to select actions. In particular, Q-learning learns about the greedy policy while it typically follows a policy involving exploratory actions— occasional selections of actions that are suboptimal according to $Q$ . Because of this, special care is required when introducing eligibility traces. 
Suppose we are backing up the state–action pair $S_{t},A_{t}$ at time $t$ . Suppose that on the next two time steps the agent selects the greedy action, but on the third, at time $t+3$ , the agent selects an exploratory, nongreedy action. In learning about the value of the greedy policy at $S_{t},A_{t}$ we can use subsequent experience only as long as the greedy policy is being followed. Thus, we can use the one-step and two-step returns, but not, in this case, the three-step return. The $n$ -step returns for all $n\geq3$ no longer have any necessary relationship to the greedy policy. 
Thus, unlike $\mathrm{TD}(\lambda)$ or $\mathrm{Sarsa}(\lambda)$ , Watkins’s Q( $\lambda$ ) does not look ahead all the way to the end of the episode in its backup. It only looks ahead as far as the next exploratory action. Aside from this difference, however, Watkins’s $\mathrm{Q}(\lambda)$ is much like $\mathrm{TD}(\lambda)$ and Sarsa $(\lambda)$ . Their lookahead stops at episode’s end, whereas $\mathrm{Q}(\lambda)$ ’s lookahead stops at the first exploratory action, or at episode’s end if there are no exploratory actions before that. Actually, to be more precise, one-step Q-learning and Watkins’s $\mathrm{Q}(\lambda)$ both look one action past the first exploration, using their knowledge of the action values. For example, suppose the first action, $A_{t+1}$ , is exploratory. Watkins’s $\mathrm{Q}(\lambda)$ would still do the one-step update of $\boldsymbol{Q}_{t}(S_{t},A_{t})$ toward $R_{t+1}{+}\gamma\operatorname*{max}_{a}Q_{t}(S_{t+1},a)$ . In general, if $A_{t+n}$ is the first exploratory action, then the longest backup is toward 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/f26a993bc887a7a2f0c7d690fee3b04795b48ed1072d690a6665ab0c9c392c4f.jpg) 
Figure 7.14: The backup diagram for Watkins’s $\mathrm{Q}(\lambda)$ . The series of component backups ends either with the end of the episode or with the first nongreedy action, whichever comes first. 
$$
R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^{n}\operatorname*{max}_{a}Q_{t}(S_{t+n},a),
$$ 
where we assume off-line updating. The backup diagram in Figure 7.14 illustrates the forward view of Watkins’s $\mathrm{Q}(\lambda)$ , showing all the component backups. 
The mechanistic or backward view of Watkins’s $\mathrm{Q}(\lambda)$ is also very simple. Eligibility traces are used just as in $\mathrm{Sarsa}(\lambda)$ , except that they are set to zero whenever an exploratory (nongreedy) action is taken. The trace update is best thought of as occurring in two steps. First, the traces for all state–action pairs are either decayed by $\gamma\lambda$ or, if an exploratory action was taken, set to $0$ . Second, the trace corresponding to the current state and action is incremented by 1. The overall result is 
$$
\begin{array}{r}{E_{t}(s,a)=\left\{\begin{array}{l l}{\gamma\lambda E_{t-1}(s,a)+I_{s S_{t}}\cdot I_{a A_{t}}}&{\mathrm{if~}Q_{t-1}(S_{t},A_{t})=\operatorname*{max}_{a}Q_{t-1}(S_{t},a);}\ {I_{s S_{t}}\cdot I_{a A_{t}}}&{\mathrm{otherwise}.}\end{array}\right.}\end{array}
$$ 
One could also use analogous dutch or replacing traces here. The rest of the algorithm is defined by 
$$
Q_{t+1}(s,a)=Q_{t}(s,a)+\alpha\delta_{t}E_{t}(s,a),\quad\forall s\in\mathbb{S},a\in\mathcal{A}(s)
$$ 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/cbf5a311d07dfac7eab9429b2e7200ffed2dd641b592703a3e80a5ed96cb9302.jpg) 
Figure 7.15: Tabular version of Watkins’s $\mathrm{Q}(\lambda)$ algorithm. 
where 
$$
\delta_{t}=R_{t+1}+\gamma\operatorname*{max}_{a^{\prime}}Q_{t}(S_{t+1},a^{\prime})-Q_{t}(S_{t},A_{t}).
$$ 
Figure 7.15 shows the complete algorithm in pseudocode. 
Unfortunately, cutting off traces every time an exploratory action is taken loses much of the advantage of using eligibility traces. If exploratory actions are frequent, as they often are early in learning, then only rarely will backups of more than one or two steps be done, and learning may be little faster than one-step Q-learning. 
## 7.7 Off-policy Eligibility Traces using Importance Sampling 
The eligibility traces in Watkins’s $\mathrm{Q}(\lambda)$ are a crude way to deal with offpolicy training. First, they treat the off-policy aspect as binary; either the target policy is followed and traces continue normally, or it is deviated from and traces are cut off completely; there is nothing inbetween. But the target policy may take different actions with different positive probabilities, as may the behavior policy, in which case following and deviating will be a matter of degree. In Chapter 5 we saw how to use the ratio of the two probabilities of taking the action to more precisely assign credit to a single action, and the product of ratios to assign credit to a sequence. 
Second, Watkins’s $\mathrm{Q}(\lambda)$ confounds bootstrapping and off-policy deviation. Bootstrapping refers to the degree to which an algorithm builds its estimates from other estimates, like TD and DP, or does not, like MC methods. In $\mathrm{TD}(\lambda)$ and $\mathrm{Sarsa}(\lambda)$ , the $\lambda$ parameter controls the degree of bootstrapping, with the value $\lambda=1$ denoting no bootstrapping, turning these TD methods into MC methods. But the same cannot be said for $\mathrm{Q}(\lambda)$ . As soon as there is a deviation from the target policy $\mathrm{Q}(\lambda)$ cuts the trace and uses its value estimate rather than waiting for the actual rewards—it bootstraps even if $\lambda=1$ . Ideally we would like to totally de-couple bootstrapping from the off-policy aspect, to use $\lambda$ to specify the degree of bootstrapping while using importance sampling to correct independently for the degree of off-policy deviation. 
## 7.8 Implementation Issues 
It might at first appear that methods using eligibility traces are much more complex than one-step methods. A naive implementation would require every state (or state–action pair) to update both its value estimate and its eligibility trace on every time step. This would not be a problem for implementations on single-instruction, multiple-data parallel computers or in plausible neural implementations, but it is a problem for implementations on conventional serial computers. Fortunately, for typical values of $\lambda$ and $\gamma$ the eligibility traces of almost all states are almost always nearly zero; only those that have recently been visited will have traces significantly greater than zero. Only these few states really need to be updated because the updates at the others will have essentially no effect. 
In practice, then, implementations on conventional computers keep track of and update only the few states with nonzero traces. Using this trick, the computational expense of using traces is typically a few times that of a onestep method. The exact multiple of course depends on $\lambda$ and $\gamma$ and on the expense of the other computations. Cichosz (1995) has demonstrated a further implementation technique that further reduces complexity to a constant independent of $\lambda$ and $\gamma$ . Finally, it should be noted that the tabular case is in some sense a worst case for the computational complexity of traces. When function approximation is used (Chapter 9), the computational advantages of not using traces generally decrease. For example, if artificial neural networks and backpropagation are used, then traces generally cause only a doubling of the required memory and computation per step. 
## 7.9 Variable $\lambda$ \*
The $\lambda$ -return can be significantly generalized beyond what we have described so far by allowing $\lambda$ to vary from step to step, that is, by redefining the trace update as 
$$
E_{t}(s)={\left\{\begin{array}{l l}{\gamma\lambda_{t}E_{t-1}(s)}&{{\mathrm{if~}}s\neq S_{t};}\ {\gamma\lambda_{t}E_{t-1}(s)+1}&{{\mathrm{if~}}s=S_{t},}\end{array}\right.}
$$ 
where $\lambda_{t}$ denotes the value of $\lambda$ at time $t$ . This is an advanced topic because the added generality has never been used in practical applications, but it is interesting theoretically and may yet prove useful. For example, one idea is to vary $\lambda$ as a function of state: $\lambda_{t}=\lambda(S_{t})$ . If a state’s value estimate is believed to be known with high certainty, then it makes sense to use that estimate fully, ignoring whatever states and rewards are received after it. This corresponds to cutting off all the traces once this state has been reached, that is, to choosing the $\lambda$ for the certain state to be zero or very small. Similarly, states whose value estimates are highly uncertain, perhaps because even the state estimate is unreliable, can be given λs near 1. This causes their estimated values to have little effect on any updates. They are “skipped over” until a state that is known better is encountered. Some of these ideas were explored formally by Sutton and Singh (1994). 
The eligibility trace equation above is the backward view of variable $\lambda\mathrm{s}$ . The corresponding forward view is a more general definition of the $\lambda$ -return: 
$$
\begin{array}{r c l}{{G_{t}^{\lambda}}}&{{=}}&{{\displaystyle\sum_{n=1}^{\infty}G_{t}^{(n)}(1-\lambda_{t+n})\prod_{i=t+1}^{t+n-1}\lambda_{i}}}\ {{}}&{{=}}&{{\displaystyle\sum_{k=t+1}^{T-1}G_{t}^{(k-t)}(1-\lambda_{k})\prod_{i=t+1}^{k-1}\lambda_{i}+G_{t}\prod_{i=t+1}^{T-1}\lambda_{i}.}}\end{array}
$$ 
## 7.10 Conclusions 
Eligibility traces in conjunction with TD errors provide an efficient, incremental way of shifting and choosing between Monte Carlo and TD methods. Traces can be used without TD errors to achieve a similar effect, but only awkwardly. A method such as TD( $\lambda$ ) enables this to be done from partial experiences and with little memory and little nonmeaningful variation in predictions. 
As we mentioned in Chapter 5, Monte Carlo methods may have advantages in non-Markov tasks because they do not bootstrap. Because eligibility traces make TD methods more like Monte Carlo methods, they also can have advantages in these cases. If one wants to use TD methods because of their other advantages, but the task is at least partially non-Markov, then the use of an eligibility trace method is indicated. Eligibility traces are the first line of defense against both long-delayed rewards and non-Markov tasks. 
By adjusting $\lambda$ , we can place eligibility trace methods anywhere along a continuum from Monte Carlo to one-step TD methods. Where shall we place them? We do not yet have a good theoretical answer to this question, but a clear empirical answer appears to be emerging. On tasks with many steps per episode, or many steps within the half-life of discounting, it appears significantly better to use eligibility traces than not to (e.g., see Figure 9.12). On the other hand, if the traces are so long as to produce a pure Monte Carlo method, or nearly so, then performance degrades sharply. An intermediate mixture appears to be the best choice. Eligibility traces should be used to bring us toward Monte Carlo methods, but not all the way there. In the future it may be possible to vary the trade-off between TD and Monte Carlo methods more finely by using variable $\lambda$ , but at present it is not clear how this can be done reliably and usefully. 
Methods using eligibility traces require more computation than one-step methods, but in return they offer significantly faster learning, particularly when rewards are delayed by many steps. Thus it often makes sense to use eligibility traces when data are scarce and cannot be repeatedly processed, as is often the case in on-line applications. On the other hand, in off-line applications in which data can be generated cheaply, perhaps from an inexpensive simulation, then it often does not pay to use eligibility traces. In these cases the objective is not to get more out of a limited amount of data, but simply to process as much data as possible as quickly as possible. In these cases the speedup per datum due to traces is typically not worth their computational cost, and one-step methods are favored. 

# 8 Planning and Learning with Tabular Methods 
In this chapter we develop a unified view of methods that require a model of the environment, such as dynamic programming and heuristic search, and methods that can be used without a model, such as Monte Carlo and temporal difference methods. 
>  本章为 model-based 方法，例如动态规划和启发式搜索，以及 model-free 方法，例如 MC 和 TD，构建统一的视角

We think of the former as planning methods and of the latter as learning methods. Although there are real differences between these two kinds of methods, there are also great similarities. In particular, the heart of both kinds of methods is the computation of value functions. Moreover, all the methods are based on looking ahead to future events, computing a backed up value, and then using it to update an approximate value function. 
>  我们将 model-based 方法视作规划方法，将 model-free 方法视作学习方法
>  规划方法和学习方法的核心都是计算价值函数，并且都基于向前看未来事件，计算回溯更新值，然后更新近似价值函数

Earlier in this book we presented Monte Carlo and temporal-difference methods as distinct alternatives, then showed how they can be seamlessly integrated by using eligibility traces such as in TD ( $\lambda$ ). Our goal in this chapter is a similar integration of planning and learning methods. Having established these as distinct in earlier chapters, we now explore the extent to which they can be intermixed. 

## 8.1 Models and Planning 
By a model of the environment we mean anything that an agent can use to predict how the environment will respond to its actions. Given a state and an action, a model produces a prediction of the resultant next state and next reward. If the model is stochastic, then there are several possible next states and next rewards, each with some probability of occurring. 
>  任意的智能体可以用于预测环境如何回应其动作的模型都可以视作环境模型
>  给定一个状态和动作，模型输出对下一个状态以及奖励的预测，如果模型是随机性的，则会存在多种可能的下一个状态和奖励，各自具有对应的概率

Some models produce a description of all possibilities and their probabilities; these we call distribution models. Other models produce just one of the possibilities, sampled according to the probabilities; these we call sample models. For example, consider modeling the sum of a dozen dice. A distribution model would produce all possible sums and their probabilities of occurring, whereas a sample model would produce an individual sum drawn according to this probability distribution. The kind of model assumed in dynamic programming—estimates of the state transition probabilities and expected rewards, $p (s^{\prime}|s, a)$ and $r (s, a, s^{\prime})$ — is a distribution model. The kind of model used in the blackjack example in Chapter 5 is a sample model. Distribution models are stronger than sample models in that they can always be used to produce samples. However, in surprisingly many applications it is much easier to obtain sample models than distribution models. 
>  一些模型输出完整的分布，我们称为分布模型，其他模型输出基于分布采样的单个样本，我们称为样本模型
>  在动态规划方法中使用的模型是分布模型
>  一般获取样本模型比获取分布模型简单

Models can be used to mimic or simulate experience. Given a starting state and action, a sample model produces a possible transition, and a distribution model generates all possible transitions weighted by their probabilities of occurring. Given a starting state and a policy, a sample model could produce an entire episode, and a distribution model could generate all possible episodes and their probabilities. In either case, we say the model is used to simulate the environment and produce simulated experience. 
>  模型可以用于模拟环境并生成模拟的经验
>  给定起始状态和动作，样本模型可以生成一个可能的 transition，分布模型可以生成所有可能的 transition
>  给定起始状态和策略，样本模型可以生成一个可能的 episode，分布模型可以生成所有可能的 episodes
>  无论是哪种情况，我们都称模型被用于模拟环境并生成模拟的经验

The word planning is used in several different ways in different fields. We use the term to refer to any computational process that takes a model as input and produces or improves a policy for interacting with the modeled environment: 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/56f5a7b2e9f5125327e5e443016274bd04b09c1447f9f191026f5cf80590ed4d.jpg) 


>  “规划”指任意接收模型作为输入，通过与该模型建模的环境交互，输出改进的策略的过程

Within artificial intelligence, there are two distinct approaches to planning according to our definition. In state-space planning, which includes the approach we take in this book, planning is viewed primarily as a search through the state space for an optimal policy or path to a goal. Actions cause transitions from state to state, and value functions are computed over states. In what we call plan-space planning, planning is instead viewed as a search through the space of plans. Operators transform one plan into another, and value functions, if any, are defined over the space of plans. Plan-space planning includes evolutionary methods and partial-order planning, a popular kind of planning in artificial intelligence in which the ordering of steps is not completely determined at all stages of planning. Plan-space methods are difficult to apply efficiently to the stochastic optimal control problems that are the focus in reinforcement learning, and we do not consider them further (but see Section 15.6 for one application of reinforcement learning within plan-space planning). 
>  有两类不同的规划方法
>  在状态空间规划中，规划主要被视作在状态空间中搜索前往目标的最优策略或路径，动作将要给状态转化为另一个状态，价值函数基于状态计算
>  在规划空间规划中，规划主要被视作在规划空间的搜索，算子将一个规划转化为另一个规划，价值函数基于规划本身计算
>  规划空间规划包含了进化方法和偏序规划
>  规划空间方法难以有效利用与强化学习关注的随机最优控制问题，故我们不进一步考虑它们

The unified view we present in this chapter is that all state-space planning methods share a common structure, a structure that is also present in the learning methods presented in this book. It takes the rest of the chapter to develop this view, but there are two basic ideas: (1) all state-space planning methods involve computing value functions as a key intermediate step toward improving the policy, and (2) they compute their value functions by backup operations applied to simulated experience. This common structure can be diagrammed as follows: 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/24530d7413477ebfd98f0f9c0b4b0268875e0a81f276e11af165f3c43f423365.jpg) 

>  所有的状态空间规划方法都共享相同的结构
>  两大基本观点是 1. 所有的状态空间规划方法用于提升策略的关键中间步骤都是计算价值函数 2. 这些方法通过对模拟经验应用回溯更新操作计算价值函数
>  结构如图所示

Dynamic programming methods clearly fit this structure: they make sweeps through the space of states, generating for each state the distribution of possible transitions. Each distribution is then used to compute a backed-up value and update the state’s estimated value. In this chapter we argue that various other state-space planning methods also fit this structure, with individual methods differing only in the kinds of backups they do, the order in which they do them, and in how long the backed-up information is retained. 
>  动态规划方法显然符合该结构
>  它们遍历状态空间，为每个状态生成可能 transitions 的分布，每个分布被用于计算回溯更新值，然后一起加权更新状态的估计值
>  许多其他状态空间规划方法和符合该结构，这些方法各自的差异仅在于它们所做的回溯更新类型、顺序以及回溯更新信息会保持多久

Viewing planning methods in this way emphasizes their relationship to the learning methods that we have described in this book. The heart of both learning and planning methods is the estimation of value functions by backup operations. The difference is that whereas planning uses simulated experience generated by a model, learning methods use real experience generated by the environment. 
>  以这种方法看待规划方法也强调了规划方法和学习方法的关联
>  规划方法和学习方法的核心都是通过回溯更新操作估计价值函数，差异在于规划方法使用的是模型生成的模拟经验，而学习方法使用的是环境生成的真实经验

Of course this difference leads to a number of other differences, for example, in how performance is assessed and in how flexibly experience can be generated. But the common structure means that many ideas and algorithms can be transferred between planning and learning. In particular, in many cases a learning algorithm can be substituted for the key backup step of a planning method. Learning methods require only experience as input, and in many cases they can be applied to simulated experience just as well as to real experience. 
>  当然这一差异会导致许多其他差异，例如如何评估性能和如何灵活生成经验
>  但是，共同的结构意味着规划和学习的许多算法和思想可以相互迁移
>  特别地，在许多情况下，一个学习算法将其关键的回溯更新步骤替换，就可以成为一个规划算法
>  学习方法仅需要经验作为输入，而许多情况下，输入的可以是真实经验也可以是模拟经验

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/edc8f24444aa85df65dd4c02fc00eded31ac5e21e755697bdc39557945f8f0cf.jpg) 

Figure 8.1: Random-sample one-step tabular Q-planning 

Figure 8.1 shows a simple example of a planning method based on one-step tabular Q-learning and on random samples from a sample model. This method, which we call random-sample one-step tabular Q-planning, converges to the optimal policy for the model under the same conditions that one-step tabular Q-learning converges to the optimal policy for the real environment (each state–action pair must be selected an infinite number of times in Step $1$ , and $\alpha$ must decrease appropriately over time). 
>  Fig 8.1 展示了基于一步表格 Q-learning 的规划方法，该方法利用样本模型的随机样本优化表格 Q 函数
>  我们称该方法为随机样本一步表格 Q-planning，该方法收敛到环境模型的最优策略的条件和一步表格 Q-learning 收敛到真实环境的最优策略的条件一致 (每个状态-动作对必须被访问无限次，学习率 $\alpha$ 必须随着时间递减)

In addition to the unified view of planning and learning methods, a second theme in this chapter is the benefits of planning in small, incremental steps. This enables planning to be interrupted or redirected at any time with little wasted computation, which appears to be a key requirement for efficiently intermixing planning with acting and with learning of the model. More surprisingly, later in this chapter we present evidence that planning in very small steps may be the most efficient approach even on pure planning problems if the problem is too large to be solved exactly. 
>  除了主要讨论规划和学习方法的统一视角外，本章的另一个主题是以小的、增量的步骤进行规划的好处，这使得我们可以在任何时候中断或重新调整规划，而不会造成过多的计算浪费，这也是高效地将规划于行动和模型学习结合的关键要求
>  即便是在纯规划问题上，如果问题规模过大而无法精确求解，小步长规划也是非常高效的方法

## 8.2 Integrating Planning, Acting, and Learning 
When planning is done on-line, while interacting with the environment, a number of interesting issues arise. New information gained from the interaction may change the model and thereby interact with planning. It may be desirable to customize the planning process in some way to the states or decisions currently under consideration, or expected in the near future. 
>  在规划是在线形式执行的，在与环境交互时，会出现许多有趣的问题
>  从交互中获得的新信息可能会改变模型，并进而影响规划过程，可能有必要以某种方式定制化以使用当前考虑的 (或者在未来期望出现的) 状态或决策

If decision-making and model-learning are both computation-intensive processes, then the available computational resources may need to be divided between them. 
>  如果决策和模型学习二者都是计算密集的过程，则可用的计算资源需要在二者时间划分

To begin exploring these issues, in this section we present Dyna-Q, a simple architecture integrating the major functions needed in an on-line planning agent. Each function appears in Dyna-Q in a simple, almost trivial, form. In subsequent sections we elaborate some of the alternate ways of achieving each function and the trade-offs between them. For now, we seek merely to illustrate the ideas and stimulate your intuition. 
>  我们在本节介绍 Dyna-Q 框架，它集成了在线规划智能体所需要的主要功能，且 Dyna-Q 中的每个功能都以简单的形式呈现
>  之后的小节将详述实现各个功能的不同方式以及它们之间的权衡，本节仅仅先介绍其概念

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/0e7b9781829a9c15714ccf220a2228611f6e7b82fd2c544e74c7b8d0a71b90f8.jpg) 

Figure 8.2: Relationships among learning, planning, and acting. 

Within a planning agent, there are at least two roles for real experience: it can be used to improve the model (to make it more accurately match the real environment) and it can be used to directly improve the value function and policy using the kinds of reinforcement learning methods we have discussed in previous chapters. The former we call model-learning, and the latter we call direct reinforcement learning (direct RL). 
>  在一个规划智能体中，真实的经验至少起到两个作用: 它可以用于改进模型 (使模型更加匹配真实环境)、它可以用于直接改进价值函数和策略 (使用我们之前讨论的强化学习方法)
>  前者我们称为模型学习，后者我们称为直接强化学习

>  目前看来，规划智能体比之前讨论的智能体多了一个功能要求，即需要具备自行拟合环境模型的能力

The possible relationships between experience, model, values, and policy are summarized in Figure 8.2. Each arrow shows a relationship of influence and presumed improvement. Note how experience can improve value and policy functions either directly or indirectly via the model. It is the latter, which is sometimes called indirect reinforcement learning, that is involved in planning. 
>  经验、模型、价值和策略可能的关系总结于 Figure 8.2，其中箭头表示影响关系或预期的改进
>  经验可以直接用于改进价值函数和策略函数，也可以间接通过模型改进价值函数和策略函数
>  经验通过模型间接改进价值和策略被称为间接强化学习，间接强化学习也是规划所涉及的内容

Both direct and indirect methods have advantages and disadvantages. Indirect methods often make fuller use of a limited amount of experience and thus achieve a better policy with fewer environmental interactions. On the other hand, direct methods are much simpler and are not affected by biases in the design of the model. 
>  直接方法和间接方法都各有优劣
>  间接方法通常可以更完全地利用有限数量的经验，进而在更少的环境交互下获得更好的策略；直接方法则更简单，不会被模型设计的偏差 (即人为构造的环境模型和真实环境的差异) 影响

Some have argued that indirect methods are always superior to direct ones, while others have argued that direct methods are responsible for most human and animal learning. Related debates in psychology and AI concern the relative importance of cognition as opposed to trial-and-error learning, and of deliberative planning as opposed to reactive decision-making. Our view is that the contrast between the alternatives in all these debates has been exaggerated, that more insight can be gained by recognizing the similarities between these two sides than by opposing them. For example, in this book we have emphasized the deep similarities between dynamic programming and temporal-difference methods, even though one was designed for planning and the other for model-free learning. 
>  有些人认为间接方法总是优于直接方法，而另一些人则认为直接方法是人类和动物学习的主要方式。心理学和人工智能领域的相关争论集中在认知相对于试错学习的重要性，以及深思熟虑的规划相对于反应性决策的重要性。我们的观点是，这些争论中的对立双方被过分夸大了，通过认识到这两种方法之间的相似性而非对立性，可以获得更多洞见。例如，在本书中，我们强调了动态规划与时间差分方法之间深刻的相似性，尽管前者是为规划而设计的，后者则是为无模型学习而设计的。

Dyna-Q includes all of the processes shown in Figure 8.2—planning, acting, model-learning, and direct RL—all occurring continually. The planning method is the random-sample one-step tabular Q-planning method given in Figure 8.1. The direct RL method is one-step tabular Q-learning. The model- learning method is also table-based and assumes the world is deterministic. After each transition $S_{t}, A_{t}\sim R_{t+1}, S_{t+1}$ , the model records in its table entry for $S_{t}, A_{t}$ the prediction that $R_{t+1}, S_{t+1}$ will deterministically follow. Thus, if the model is queried with a state–action pair that has been experienced before, it simply returns the last-observed next state and next reward as its prediction. During planning, the Q-planning algorithm randomly samples only from state–action pairs that have previously been experienced (in Step 1), so the model is never queried with a pair about which it has no information. 
>  Dyna-Q 包含了 Fig 8.2 中的所有过程: 规划、行动、模型学习、直接 RL，这些过程持续不断地进行
>  其规划方法即 Fig 8.1 的一步表格 Q-planning，其直接 RL 方法即一步表格 Q-learning
>  其模型学习方法同样基于表格，并且假设世界是确定性的，每次转移 $S_t, A_t \sim R_{t+1}, S_{t+1}$ 后，模型在其表格项中记录这一转移，因此，如果查询模型之前已经经历过的状态-动作对，它会直接返回最后观测到的结果作为其预测
>  在规划过程中，Q-planning 算法直接从之前经历过的状态-动作对中随机选取样本，故模型也不会被 Q-planning 算法查询它没有经历过的转移

The overall architecture of Dyna agents, of which the Dyna-Q algorithm is one example, is shown in Figure 8.3. The central column represents the basic interaction between agent and environment, giving rise to a trajectory of real experience. The arrow on the left of the figure represents direct reinforcement learning operating on real experience to improve the value function and the policy. On the right are model-based processes. The model is learned from real experience and gives rise to simulated experience. We use the term search control to refer to the process that selects the starting states and actions for the simulated experiences generated by the model. Finally, planning is achieved by applying reinforcement learning methods to the simulated experiences just as if they had really happened. Typically, as in Dyna-Q, the same reinforcement learning method is used both for learning from real experience and for planning from simulated experience. The reinforcement learning method is thus the “final common path” for both learning and planning. Learning and planning are deeply integrated in the sense that they share almost all the same machinery, differing only in the source of their experience. 

![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/d00bd93e04d015fa8f20d0f8a084de7a74d70f2b16c3f838dac6d564e217a384.jpg) 
Figure 8.3: The general Dyna Architecture 
Conceptually, planning, acting, model-learning, and direct RL occur simultaneously and in parallel in Dyna agents. For concreteness and implementation on a serial computer, however, we fully specify the order in which they occur within a time step. In Dyna-Q, the acting, model-learning, and direct RL processes require little computation, and we assume they consume just a fraction of the time. The remaining time in each step can be devoted to the planning process, which is inherently computation-intensive. Let us assume that there is time in each step, after acting, model-learning, and direct RL, to complete $n$ iterations (Steps 1–3) of the Q-planning algorithm. Figure 8.4 shows the complete algorithm for Dyna-Q. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/0116537f2d1d011c29dfe691f6d9579063f67a7782ceb08f50f8085f51712f97.jpg) 
Figure 8.4: Dyna-Q Algorithm. $M o d e l (s, a)$ denotes the contents of the model (predicted next state and reward) for state–action pair $s, a$ . Direct reinforcement learning, model-learning, and planning are implemented by steps (d), (e), and (f), respectively. If (e) and (f) were omitted, the remaining algorithm would be one-step tabular Q-learning. 
Example 8.1: Dyna Maze Consider the simple maze shown inset in Figure 8.5. In each of the 47 states there are four actions, up, down, right, and left, which take the agent deterministically to the corresponding neighboring states, except when movement is blocked by an obstacle or the edge of the maze, in which case the agent remains where it is. Reward is zero on all transitions, except those into the goal state, on which it is $+1$ . After reaching the goal state (G), the agent returns to the start state (S) to begin a new episode. This is a discounted, episodic task with $\gamma=0.95$ . 
The main part of Figure 8.5 shows average learning curves from an experiment in which Dyna-Q agents were applied to the maze task. The initial action values were zero, the step-size parameter was $\alpha=0.1$ , and the exploration parameter was $\epsilon=0.1$ . When selecting greedily among actions, ties were broken randomly. The agents varied in the number of planning steps, $n$ , they performed per real step. For each $n$ , the curves show the number of steps taken by the agent in each episode, averaged over 30 repetitions of the experiment. In each repetition, the initial seed for the random number generator was held constant across algorithms. Because of this, the first episode was exactly the same (about 1700 steps) for all values of $n$ , and its data are not shown in the figure. After the first episode, performance improved for all values of $n$ , but much more rapidly for larger values. Recall that the $n=0$ agent is a nonplanning agent, utilizing only direct reinforcement learning (onestep tabular Q-learning). This was by far the slowest agent on this problem, despite the fact that the parameter values ( $\alpha$ and $\boldsymbol{\varepsilon}$ ) were optimized for it. The nonplanning agent took about 25 episodes to reach ( $\xi\cdot$ -) optimal performance, whereas the $n=5$ agent took about five episodes, and the $n=50$ agent took only three episodes. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/9aa1fc9df9a1c1ad03390b688fcf70eae86f093501e47e7c5ec896a8defcf5f6.jpg) 
Figure 8.5: A simple maze (inset) and the average learning curves for Dyna-Q agents varying in their number of planning steps ( $_{n}$ ) per real step. The task is to travel from $\mathsf{S}$ to $\mathsf{G}$ as quickly as possible. 
Figure 8.6 shows why the planning agents found the solution so much faster than the nonplanning agent. Shown are the policies found by the $n=0$ and $n=50$ agents halfway through the second episode. Without planning ( $n=0$ ), each episode adds only one additional step to the policy, and so only one step (the last) has been learned so far. With planning, again only one step is learned during the first episode, but here during the second episode an extensive policy has been developed that by the episode’s end will reach almost back to the start state. This policy is built by the planning process while the agent is still wandering near the start state. By the end of the third episode a complete optimal policy will have been found and perfect performance attained. ■ 
In Dyna-Q, learning and planning are accomplished by exactly the same algorithm, operating on real experience for learning and on simulated experience for planning. Because planning proceeds incrementally, it is trivial to intermix planning and acting. Both proceed as fast as they can. The agent is always reactive and always deliberative, responding instantly to the latest sensory information and yet always planning in the background. Also ongoing in the background is the model-learning process. As new information is gained, the model is updated to better match reality. As the model changes, the ongoing planning process will gradually compute a different way of behaving to match the new model. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/cf5e1909716665d798e1f6176e652bc90b9325d93191f0a0261b2d217a74eca2.jpg) 
Figure 8.6: Policies found by planning and nonplanning Dyna-Q agents halfway through the second episode. The arrows indicate the greedy action in each state; no arrow is shown for a state if all of its action values are equal. The black square indicates the location of the agent. 
## 8.3 When the Model Is Wrong 
In the maze example presented in the previous section, the changes in the model were relatively modest. The model started out empty, and was then filled only with exactly correct information. In general, we cannot expect to be so fortunate. Models may be incorrect because the environment is stochastic and only a limited number of samples have been observed, because the model was learned using function approximation that has generalized imperfectly, or simply because the environment has changed and its new behavior has not yet been observed. When the model is incorrect, the planning process will compute a suboptimal policy. 
In some cases, the suboptimal policy computed by planning quickly leads to the discovery and correction of the modeling error. This tends to happen when the model is optimistic in the sense of predicting greater reward or better state transitions than are actually possible. The planned policy attempts to exploit these opportunities and in doing so discovers that they do not exist. 
Example 8.2: Blocking Maze A maze example illustrating this relatively minor kind of modeling error and recovery from it is shown in Figure 8.7. Initially, there is a short path from start to goal, to the right of the barrier, as shown in the upper left of the figure. After 1000 time steps, the short path is “blocked,” and a longer path is opened up along the left-hand side of the barrier, as shown in upper right of the figure. The graph shows average cumulative reward for Dyna-Q and two other Dyna agents. The first part of the graph shows that all three Dyna agents found the short path within 1000 steps. When the environment changed, the graphs become flat, indicating a period during which the agents obtained no reward because they were wandering around behind the barrier. After a while, however, they were able to find the new opening and the new optimal behavior. ■ 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/e057d09cc160b3334b367dd6673448e1d91c72cd96d3d1fd13eb9f73f3ed9d16.jpg) 
Figure 8.7: Average performance of Dyna agents on a blocking task. The left environment was used for the first 1000 steps, the right environment for the rest. Dyna-Q+ is Dyna-Q with an exploration bonus that encourages exploration. Dyna-AC is a Dyna agent that uses an actor–critic learning method instead of Q-learning. 
Greater difficulties arise when the environment changes to become better than it was before, and yet the formerly correct policy does not reveal the improvement. In these cases the modeling error may not be detected for a long time, if ever, as we see in the next example. 
Example 8.3: Shortcut Maze The problem caused by this kind of environmental change is illustrated by the maze example shown in Figure 8.8. Initially, the optimal path is to go around the left side of the barrier (upper left). After 3000 steps, however, a shorter path is opened up along the right side, without disturbing the longer path (upper right). The graph shows that two of the three Dyna agents never switched to the shortcut. In fact, they never realized that it existed. Their models said that there was no shortcut, so the more they planned, the less likely they were to step to the right and discover it. Even with an $\boldsymbol{\varepsilon}$ -greedy policy, it is very unlikely that an agent will take so many exploratory actions as to discover the shortcut. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/639eabf87092b493f67ff4da3a4f861ace73e16d37e8b1700aa3b1d582ee02dd.jpg) 
Figure 8.8: Average performance of Dyna agents on a shortcut task. The left environment was used for the first 3000 steps, the right environment for the rest. 
The general problem here is another version of the conflict between exploration and exploitation. In a planning context, exploration means trying actions that improve the model, whereas exploitation means behaving in the optimal way given the current model. We want the agent to explore to find changes in the environment, but not so much that performance is greatly degraded. As in the earlier exploration/exploitation conflict, there probably is no solution that is both perfect and practical, but simple heuristics are often effective. 
The Dyna-Q+ agent that did solve the shortcut maze uses one such heuristic. This agent keeps track for each state–action pair of how many time steps have elapsed since the pair was last tried in a real interaction with the environment. The more time that has elapsed, the greater (we might presume) the chance that the dynamics of this pair has changed and that the model of it is incorrect. To encourage behavior that tests long-untried actions, a special “bonus reward” is given on simulated experiences involving these actions. In particular, if the modeled reward for a transition is $R$ , and the transition has not been tried in $\tau$ time steps, then planning backups are done as if that transition produced a reward of $R+\kappa{\sqrt{\tau}}$ , for some small $\kappa$ . This encourages the agent to keep testing all accessible state transitions and even to plan long sequences of actions in order to carry out such tests. 1 Of course all this testing has its cost, but in many cases, as in the shortcut maze, this kind of computational curiosity is well worth the extra exploration. 
## 8.4 Prioritized Sweeping 
In the Dyna agents presented in the preceding sections, simulated transitions are started in state–action pairs selected uniformly at random from all previously experienced pairs. But a uniform selection is usually not the best; planning can be much more efficient if simulated transitions and backups are focused on particular state–action pairs. For example, consider what happens during the second episode of the first maze task (Figure 8.6). At the beginning of the second episode, only the state–action pair leading directly into the goal has a positive value; the values of all other pairs are still zero. This means that it is pointless to back up along almost all transitions, because they take the agent from one zero-valued state to another, and thus the backups would have no effect. Only a backup along a transition into the state just prior to the goal, or from it into the goal, will change any values. If simulated transitions are generated uniformly, then many wasteful backups will be made before stumbling onto one of the two useful ones. As planning progresses, the region of useful backups grows, but planning is still far less efficient than it would be if focused where it would do the most good. In the much larger problems that are our real objective, the number of states is so large that an unfocused search would be extremely inefficient. 
This example suggests that search might be usefully focused by working backward from goal states. Of course, we do not really want to use any methods specific to the idea of “goal state.” We want methods that work for general reward functions. Goal states are just a special case, convenient for stimulating intuition. In general, we want to work back not just from goal states but from any state whose value has changed. Assume that the values are initially correct given the model, as they were in the maze example prior to discovering the goal. Suppose now that the agent discovers a change in the environment and changes its estimated value of one state. Typically, this will imply that the values of many other states should also be changed, but the only useful onestep backups are those of actions that lead directly into the one state whose value has already been changed. If the values of these actions are updated, then the values of the predecessor states may change in turn. If so, then actions leading into them need to be backed up, and then their predecessor states may have changed. In this way one can work backward from arbitrary states that have changed in value, either performing useful backups or terminating the propagation. 
As the frontier of useful backups propagates backward, it often grows rapidly, producing many state–action pairs that could usefully be backed up. But not all of these will be equally useful. The values of some states may have changed a lot, whereas others have changed little. The predecessor pairs of those that have changed a lot are more likely to also change a lot. In a stochastic environment, variations in estimated transition probabilities also contribute to variations in the sizes of changes and in the urgency with which pairs need to be backed up. It is natural to prioritize the backups according to a measure of their urgency, and perform them in order of priority. This is the idea behind prioritized sweeping. A queue is maintained of every state–action pair whose estimated value would change nontrivially if backed up, prioritized by the size of the change. When the top pair in the queue is backed up, the effect on each of its predecessor pairs is computed. If the effect is greater than some small threshold, then the pair is inserted in the queue with the new priority (if there is a previous entry of the pair in the queue, then insertion results in only the higher priority entry remaining in the queue). In this way the effects of changes are efficiently propagated backward until quiescence. The full algorithm for the case of deterministic environments is given in Figure 8.9. 
Example 8.4: Prioritized Sweeping on Mazes Prioritized sweeping has been found to dramatically increase the speed at which optimal solutions are found in maze tasks, often by a factor of 5 to 10. A typical example is shown in Figure 8.10. These data are for a sequence of maze tasks of exactly the same structure as the one shown in Figure 8.5, except that they vary in the grid resolution. Prioritized sweeping maintained a decisive advantage over unprioritized Dyna-Q. Both systems made at most $n=5$ backups per environmental interaction. 
Example 8.5: Rod Maneuvering The objective in this task is to maneuver a rod around some awkwardly placed obstacles to a goal position in the fewest number of steps (Figure 8.11). The rod can be translated along its long axis or perpendicular to that axis, or it can be rotated in either direction around its center. The distance of each movement is approximately $1/20$ of the work space, and the rotation increment is 10 degrees. Translations are deterministic and quantized to one of $20\times20$ positions. The figure shows the obstacles and the shortest solution from start to goal, found by prioritized sweeping. This problem is still deterministic, but has four actions and 14,400 potential states (some of these are unreachable because of the obstacles). This problem is probably too large to be solved with unprioritized methods. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/6bf0f8b22eaaab9f5c1b134b3b80636b05333d2bcfc30b08356e4bb400f398e4.jpg) 
Figure 8.9: The prioritized sweeping algorithm for a deterministic environment. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/fb428519eded5a2292a74886b1b9e07cba6c31aa0bccc6f72a79ceb72fd55f39.jpg) 
Figure 8.10: Prioritized sweeping significantly shortens learning time on the Dyna maze task for a wide range of grid resolutions. Reprinted from Peng and Williams (1993). 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/7bba9d240f736f449c4cec7121086d8ed8d532c9f23e96d2e35659da8947b637.jpg) 
Figure 8.11: A rod-maneuvering task and its solution by prioritized sweeping. Reprinted from Moore and Atkeson (1993). 
Prioritized sweeping is clearly a powerful idea, but the algorithms that have been developed so far appear not to extend easily to more interesting cases. The greatest problem is that the algorithms appear to rely on the assumption of discrete states. When a change occurs at one state, these methods perform a computation on all the predecessor states that may have been affected. If function approximation is used to learn the model or the value function, then a single backup could influence a great many other states. It is not apparent how these states could be identified or processed efficiently. On the other hand, the general idea of focusing search on the states believed to have changed in value, and then on their predecessors, seems intuitively to be valid in general. Additional research may produce more general versions of prioritized sweeping. 
Extensions of prioritized sweeping to stochastic environments are relatively straightforward. The model is maintained by keeping counts of the number of times each state–action pair has been experienced and of what the next states were. It is natural then to backup each pair not with a sample backup, as we have been using so far, but with a full backup, taking into account all possible next states and their probabilities of occurring. 
## 8.5 Full vs. Sample Backups 
The examples in the previous sections give some idea of the range of possibilities for combining methods of learning and planning. In the rest of this chapter, we analyze some of the component ideas involved, starting with the relative advantages of full and sample backups. 
Much of this book has been about different kinds of backups, and we have considered a great many varieties. Focusing for the moment on one-step backups, they vary primarily along three binary dimensions. The first two dimensions are whether they back up state values or action values and whether they estimate the value for the optimal policy or for an arbitrary given policy. These two dimensions give rise to four classes of backups for approximating the four value functions, $q_{*}$ , $v_{*}$ , $q_{\pi}$ , and $v_{\pi}$ . The other binary dimension is whether the backups are full backups, considering all possible events that might happen, or sample backups, considering a single sample of what might happen. These three binary dimensions give rise to eight cases, seven of which correspond to specific algorithms, as shown in Figure 8.12. (The eighth case does not seem to correspond to any useful backup.) Any of these one-step backups can be used in planning methods. The Dyna-Q agents discussed earlier use $q_{*}$ sample backups, but they could just as well use $q_{*}$ full backups, or either full or sample $q_{\pi}$ backups. The Dyna-AC system uses $v_{\pi}$ sample backups together with a learning policy structure. For stochastic problems, prioritized sweeping is always done using one of the full backups. 
When we introduced one-step sample backups in Chapter 6, we presented them as substitutes for full backups. In the absence of a distribution model, full backups are not possible, but sample backups can be done using sample transitions from the environment or a sample model. Implicit in that point of view is that full backups, if possible, are preferable to sample backups. But are they? Full backups certainly yield a better estimate because they are uncorrupted by sampling error, but they also require more computation, and computation is often the limiting resource in planning. To properly assess the relative merits of full and sample backups for planning we must control for their different computational requirements. 
For concreteness, consider the full and sample backups for approximating $q_{*}$ , and the special case of discrete states and actions, a table-lookup representation of the approximate value function, $Q$ , and a model in the form of estimated dynamics, ${\hat{p}}(s^{\prime}, r|s, a)$ . The full backup for a state–action pair, $s, a$ , is: 

$$
Q(s,a)\gets\sum_{s^{\prime},r}\hat{p}(s^{\prime},r|s,a)\Big[r+\gamma\operatorname*{max}_{a^{\prime}}Q(s^{\prime},a^{\prime})\Big].
$$ 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/e1ff630a821e724b0ebc47d85a076427df30e5d3f74d89878ba4e5a48e772986.jpg) 

Figure 8.12: The one-step backups. 

The corresponding sample backup for $s, a$ , given a sample next state and reward, $S^{\prime}$ and $R$ (from the model), is the Q-learning-like update: 

$$
Q(s,a)\gets Q(s,a)+\alpha\Big[R+\gamma\operatorname*{max}_{a^{\prime}}Q(S^{\prime},a^{\prime})-Q(s,a)\Big],
$$ 
where $\alpha$ is the usual positive step-size parameter. 

The difference between these full and sample backups is significant to the extent that the environment is stochastic, specifically, to the extent that, given a state and action, many possible next states may occur with various probabilities. If only one next state is possible, then the full and sample backups given above are identical (taking $\alpha=1$ ). If there are many possible next states, then there may be significant differences. In favor of the full backup is that it is an exact computation, resulting in a new $Q (s, a)$ whose correctness is limited only by the correctness of the $Q (s^{\prime}, a^{\prime})$ at successor states. The sample backup is in addition affected by sampling error. On the other hand, the sample backup is cheaper computationally because it considers only one next state, not all possible next states. In practice, the computation required by backup operations is usually dominated by the number of state–action pairs at which $Q$ is evaluated. For a particular starting pair, $s, a$ , let $b$ be the branching factor (i.e., the number of possible next states, $s^{\prime}$ , for which ${\hat{p}}(s^{\prime}|s, a)>0)$ ). Then a full backup of this pair requires roughly $b$ times as much computation as a sample backup. 
If there is enough time to complete a full backup, then the resulting estimate is generally better than that of $b$ sample backups because of the absence of sampling error. But if there is insufficient time to complete a full backup, then sample backups are always preferable because they at least make some improvement in the value estimate with fewer than $b$ backups. In a large problem with many state–action pairs, we are often in the latter situation. With so many state–action pairs, full backups of all of them would take a very long time. Before that we may be much better off with a few sample backups at many state–action pairs than with full backups at a few pairs. Given a unit of computational effort, is it better devoted to a few full backups or to $b$ times as many sample backups? 
Figure 8.13 shows the results of an analysis that suggests an answer to this question. It shows the estimation error as a function of computation time for full and sample backups for a variety of branching factors, $b$ . The case considered is that in which all $b$ successor states are equally likely and in which the error in the initial estimate is 1. The values at the next states are assumed correct, so the full backup reduces the error to zero upon its completion. In this case, sample backups reduce the error according to $\textstyle{\sqrt{\frac{b-1}{b t}}}$ where $t$ is the number of sample backups that have been performed (assuming sample averages, i.e., $\alpha=1/t$ ). The key observation is that for moderately large $b$ the error falls dramatically with a tiny fraction of $b$ backups. For these cases, many state–action pairs could have their values improved dramatically, to within a few percent of the effect of a full backup, in the same time that one state–action pair could be backed up fully. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/5faade4c0783890ac8ed6d906692b94fc989b788d3bbe146d525d1a7e21e7a73.jpg) 
Figure 8.13: Comparison of efficiency of full and sample backups. 
The advantage of sample backups shown in Figure 8.13 is probably an underestimate of the real effect. In a real problem, the values of the successor states would themselves be estimates updated by backups. By causing estimates to be more accurate sooner, sample backups will have a second advantage in that the values backed up from the successor states will be more accurate. These results suggest that sample backups are likely to be superior to full backups on problems with large stochastic branching factors and too many states to be solved exactly. 

## 8.6 Trajectory Sampling 
In this section we compare two ways of distributing backups. The classical approach, from dynamic programming, is to perform sweeps through the entire state (or state–action) space, backing up each state (or state–action pair) once per sweep. This is problematic on large tasks because there may not be time to complete even one sweep. In many tasks the vast majority of the states are irrelevant because they are visited only under very poor policies or with very low probability. Exhaustive sweeps implicitly devote equal time to all parts of the state space rather than focusing where it is needed. As we discussed in Chapter 4, exhaustive sweeps and the equal treatment of all states that they imply are not necessary properties of dynamic programming. In principle, backups can be distributed any way one likes (to assure convergence, all states or state–action pairs must be visited in the limit an infinite number of times), but in practice exhaustive sweeps are often used. 
The second approach is to sample from the state or state–action space according to some distribution. One could sample uniformly, as in the Dyna-Q agent, but this would suffer from some of the same problems as exhaustive sweeps. More appealing is to distribute backups according to the on-policy distribution, that is, according to the distribution observed when following the current policy. One advantage of this distribution is that it is easily generated; one simply interacts with the model, following the current policy. In an episodic task, one starts in the start state (or according to the starting-state distribution) and simulates until the terminal state. In a continuing task, one starts anywhere and just keeps simulating. In either case, sample state transitions and rewards are given by the model, and sample actions are given by the current policy. In other words, one simulates explicit individual trajectories and performs backups at the state or state–action pairs encountered along the way. We call this way of generating experience and backups trajectory sampling. 
It is hard to imagine any efficient way of distributing backups according to the on-policy distribution other than by trajectory sampling. If one had an explicit representation of the on-policy distribution, then one could sweep through all states, weighting the backup of each according to the on-policy distribution, but this leaves us again with all the computational costs of exhaustive sweeps. Possibly one could sample and update individual state–action pairs from the distribution, but even if this could be done efficiently, what benefit would this provide over simulating trajectories? Even knowing the on-policy distribution in an explicit form is unlikely. The distribution changes whenever the policy changes, and computing the distribution requires computation comparable to a complete policy evaluation. Consideration of such other possibilities makes trajectory sampling seem both efficient and elegant. 
Is the on-policy distribution of backups a good one? Intuitively it seems like a good choice, at least better than the uniform distribution. For example, if you are learning to play chess, you study positions that might arise in real games, not random positions of chess pieces. The latter may be valid states, but to be able to accurately value them is a different skill from evaluating positions in real games. We will also see in Chapter 9 that the on-policy distribution has significant advantages when function approximation is used. Whether or not function approximation is used, one might expect on-policy focusing to significantly improve the speed of planning. 
Focusing on the on-policy distribution could be beneficial because it causes vast, uninteresting parts of the space to be ignored, or it could be detrimental because it causes the same old parts of the space to be backed up over and over. We conducted a small experiment to assess the effect empirically. To isolate the effect of the backup distribution, we used entirely one-step full tabular backups, as defined by (8.1). In the uniform case, we cycled through all state–action pairs, backing up each in place, and in the on-policy case we simulated episodes, backing up each state–action pair that occurred under the current $\epsilon$ -greedy policy ( $\epsilon=0.1$ ). The tasks were undiscounted episodic tasks, generated randomly as follows. From each of the |S| states, two actions were possible, each of which resulted in one of $b$ next states, all equally likely, with a different random selection of $b$ states for each state–action pair. The branching factor, $b$ , was the same for all state–action pairs. In addition, on all transitions there was a 0.1 probability of transition to the terminal state, ending the episode. We used episodic tasks to get a clear measure of the quality of the current policy. At any point in the planning process one can stop and exhaustively compute $v_{\tilde{\pi}}(s_{0})$ , the true value of the start state under the greedy policy, $\tilde{\pi}$ , given the current action-value function $Q$ , as an indication of how well the agent would do on a new episode on which it acted greedily (all the while assuming the model is correct). 
The upper part of Figure 8.14 shows results averaged over 200 sample tasks with 1000 states and branching factors of 1, 3, and 10. The quality of the policies found is plotted as a function of the number of full backups completed. In all cases, sampling according to the on-policy distribution resulted in faster planning initially and retarded planning in the long run. The effect was stronger, and the initial period of faster planning was longer, at smaller branching factors. In other experiments, we found that these effects also became stronger as the number of states increased. For example, the lower part of Figure 8.14 shows results for a branching factor of 1 for tasks with 10,000 states. In this case the advantage of on-policy focusing is large and long-lasting. 
All of these results make sense. In the short term, sampling according to the on-policy distribution helps by focusing on states that are near descendants of the start state. If there are many states and a small branching factor, this effect will be large and long-lasting. In the long run, focusing on the on-policy distribution may hurt because the commonly occurring states all already have their correct values. Sampling them is useless, whereas sampling other states may actually perform some useful work. This presumably is why the exhaustive, unfocused approach does better in the long run, at least for small problems. These results are not conclusive because they are only for problems generated in a particular, random way, but they do suggest that sampling according to the on-policy distribution can be a great advantage for large problems, in particular for problems in which a small subset of the state–action space is visited under the on-policy distribution. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/c59b8593b113e30be3f0c822f2e991c1a766d8f398790fb60831f3e5e20ed09e.jpg) 
Figure 8.14: Relative efficiency of backups distributed uniformly across the state space versus focused on simulated on-policy trajectories. Results are for randomly generated tasks of two sizes and various branching factors, $b$ . 
## 8.7 Heuristic Search 
The predominant state-space planning methods in artificial intelligence are collectively known as heuristic search. Although superficially different from the planning methods we have discussed so far in this chapter, heuristic search and some of its component ideas can be combined with these methods in useful ways. Unlike these methods, heuristic search is not concerned with changing the approximate, or “heuristic,” value function, but only with making improved action selections given the current value function. In other words, heuristic search is planning as part of a policy computation. 
In heuristic search, for each state encountered, a large tree of possible continuations is considered. The approximate value function is applied to the leaf nodes and then backed up toward the current state at the root. The backing up within the search tree is just the same as in the max-backups (those for $v_{*}$ and $q_{*}$ ) discussed throughout this book. The backing up stops at the state–action nodes for the current state. Once the backed-up values of these nodes are computed, the best of them is chosen as the current action, and then all backed-up values are discarded. 
In conventional heuristic search no effort is made to save the backed-up values by changing the approximate value function. In fact, the value function is generally designed by people and never changed as a result of search. However, it is natural to consider allowing the value function to be improved over time, using either the backed-up values computed during heuristic search or any of the other methods presented throughout this book. In a sense we have taken this approach all along. Our greedy and $\boldsymbol{\varepsilon}$ -greedy action-selection methods are not unlike heuristic search, albeit on a smaller scale. For example, to compute the greedy action given a model and a state-value function, we must look ahead from each possible action to each possible next state, backup the rewards and estimated values, and then pick the best action. Just as in conventional heuristic search, this process computes backed-up values of the possible actions, but does not attempt to save them. Thus, heuristic search can be viewed as an extension of the idea of a greedy policy beyond a single step. 
The point of searching deeper than one step is to obtain better action selections. If one has a perfect model and an imperfect action-value function, then in fact deeper search will usually yield better policies. 2 Certainly, if the search is all the way to the end of the episode, then the effect of the imperfect value function is eliminated, and the action determined in this way must be optimal. If the search is of sufficient depth $k$ such that $\gamma^{k}$ is very small, then the actions will be correspondingly near optimal. On the other hand, the deeper the search, the more computation is required, usually resulting in a slower response time. A good example is provided by Tesauro’s grandmaster-level backgammon player, TD-Gammon (Section 15.1). This system used TD ( $\lambda$ ) to learn an afterstate value function through many games of self-play, using a form of heuristic search to make its moves. As a model, TD-Gammon used a priori knowledge of the probabilities of dice rolls and the assumption that the opponent always selected the actions that TD-Gammon rated as best for it. Tesauro found that the deeper the heuristic search, the better the moves made by TD-Gammon, but the longer it took to make each move. Backgammon has a large branching factor, yet moves must be made within a few seconds. It was only feasible to search ahead selectively a few steps, but even so the search resulted in significantly better action selections. 
So far we have emphasized heuristic search as an action-selection technique, but this may not be its most important aspect. Heuristic search also suggests ways of selectively distributing backups that may lead to better and faster approximation of the optimal value function. A great deal of research on heuristic search has been devoted to making the search as efficient as possible. The search tree is grown selectively, deeper along some lines and shallower along others. For example, the search tree is often deeper for the actions that seem most likely to be best, and shallower for those that the agent will probably not want to take anyway. Can we use a similar idea to improve the distribution of backups? Perhaps it can be done by preferentially updating state–action pairs whose values appear to be close to the maximum available from the state. To our knowledge, this and other possibilities for distributing backups based on ideas borrowed from heuristic search have not yet been explored. 
We should not overlook the most obvious way in which heuristic search focuses backups: on the current state. Much of the effectiveness of heuristic search is due to its search tree being tightly focused on the states and actions that might immediately follow the current state. You may spend more of your life playing chess than checkers, but when you play checkers, it pays to think about checkers and about your particular checkers position, your likely next moves, and successor positions. However you select actions, it is these states and actions that are of highest priority for backups and where you most urgently want your approximate value function to be accurate. Not only should your computation be preferentially devoted to imminent events, but so should your limited memory resources. In chess, for example, there are far too many possible positions to store distinct value estimates for each of them, but chess programs based on heuristic search can easily store distinct estimates for the millions of positions they encounter looking ahead from a single position. This great focusing of memory and computational resources on the current decision is presumably the reason why heuristic search can be so effective. 
![](https://cdn-mineru.openxlab.org.cn/extract/1f83486c-03b4-4bfd-9cdf-2c61c53bbf89/1f2c1ad3ba17568367ee1f6ce08e976f47583e780df4517f0f44c0e1622de077.jpg) 
Figure 8.15: The deep backups of heuristic search can be implemented as a sequence of one-step backups (shown here outlined). The ordering shown is for a selective depth-first search. 
The distribution of backups can be altered in similar ways to focus on the current state and its likely successors. As a limiting case we might use exactly the methods of heuristic search to construct a search tree, and then perform the individual, one-step backups from bottom up, as suggested by Figure 8.15. If the backups are ordered in this way and a table-lookup representation is used, then exactly the same backup would be achieved as in heuristic search. Any state-space search can be viewed in this way as the piecing together of a large number of individual one-step backups. Thus, the performance improvement observed with deeper searches is not due to the use of multistep backups as such. Instead, it is due to the focus and concentration of backups on states and actions immediately downstream from the current state. By devoting a large amount of computation specifically relevant to the candidate actions, a much better decision can be made than by relying on unfocused backups. 
## 8.8 Monte Carlo Tree Search 
## 8.9 Summary 
We have presented a perspective emphasizing the surprisingly close relationships between planning optimal behavior and learning optimal behavior. Both involve estimating the same value functions, and in both cases it is natural to update the estimates incrementally, in a long series of small backup operations. This makes it straightforward to integrate learning and planning processes simply by allowing both to update the same estimated value function. In addition, any of the learning methods can be converted into planning methods simply by applying them to simulated (model-generated) experience rather than to real experience. In this case learning and planning become even more similar; they are possibly identical algorithms operating on two different sources of experience. 
It is straightforward to integrate incremental planning methods with acting and model-learning. Planning, acting, and model-learning interact in a circular fashion (Figure 8.2), each producing what the other needs to improve; no other interaction among them is either required or prohibited. The most natural approach is for all processes to proceed asynchronously and in parallel. If the processes must share computational resources, then the division can be handled almost arbitrarily—by whatever organization is most convenient and efficient for the task at hand. 
In this chapter we have touched upon a number of dimensions of variation among state-space planning methods. One of the most important of these is the distribution of backups, that is, of the focus of search. Prioritized sweeping focuses on the predecessors of states whose values have recently changed. Heuristic search applied to reinforcement learning focuses, inter alia, on the successors of the current state. Trajectory sampling is a convenient way of focusing on the on-policy distribution. All of these approaches can significantly speed planning and are current topics of research. 
Another interesting dimension of variation is the size of backups. The smaller the backups, the more incremental the planning methods can be. Among the smallest backups are one-step sample backups. We presented one study suggesting that one-step sample backups may be preferable on very large problems. A related issue is the depth of backups. In many cases deep backups can be implemented as sequences of shallow backups. 
