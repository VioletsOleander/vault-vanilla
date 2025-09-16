Site: https://cursor.com/cn/blog/tab-rl#the-policy-gradient
Date: 2025-09-11

# Online RL for Cursor Tab
Our new Tab model has a 28% higher accept rate, while making 21% fewer suggestions.

Posted By Jacob, Phillip, Shomil

6 minutes read

---

At Cursor, our goal is to make developers an order of magnitude more productive. An important part of that goal is Cursor Tab, our system that predicts your next action across your codebase. Whenever you type a character or move your cursor within the editor, our Tab model tries to predict what you'll do next, and if it has sufficient confidence, we'll display its prediction as a suggestion that you can accept by pressing Tab.
>  Cursor Tab 能够预测我们在我们代码库中的下一个动作
>  无论我们是键入了一个字符或在编辑器中移动光标，Tab model 会预测我们将做什么，如果模型的预测输出有足够的置信度，就会将预测结果以建议的形式展示出来，只需按下 Tab 键就可以接收该建议

Our Tab model runs on every user action, handling over 400 million requests per day. As a result, we have a lot of data about which suggestions users accept and reject. This post describes how we use this data to improve Tab using online reinforcement learning.
>  Tabl model 会响应每个用户操作，每天处理超过 4 亿次请求
>  因此，我们积累了大量关于用户接收和拒绝的建议，本文介绍我们如何使用这些数据，经过在线 RL 来提升 Tab model

Our approach is unusual because it involves rolling out new models to users frequently throughout the day and using that data for training. Most other LLM providers train on static datasets or use paid labelers, and only roll out a new model to users as part of a named model release every few months.
>  我们的方法与众不同，因为它会在一天内频繁地向用户推送新的模型，并使用数据进行训练
>  大多数其他 LLM 供应商在静态数据集上训练，或者依赖标注人员，且通常每个月才发布一次新的模型版本

## The problem of noisy suggestions
We try to keep the accept rate of Tab suggestions high. If the accept rate is low, it means we're showing too many incorrect suggestions, which is distracting and disrupts the flow of coding.
>  我们的目标是让 Tab model 的建议接受率更高
>  如果接受率低，说明我们提供了太多错误的建议

Achieving a high accept rate isn't just about making the model smarter, but also knowing when to suggest and when not to. Sometimes there simply isn't enough information to know what action the user is going to take: even if the model had perfect knowledge and reasoning ability, it wouldn't know what the user will do. In these situations, we shouldn't suggest anything.
>  提高接受率不仅仅是让模型更聪明，还涉及什么时候提出建议，什么时候不应该提出建议
>  有时往往是缺乏足够预测用户将做什么的信息，即便模型已经有了很好的推理能力，它也不知道用户将做什么，此时，我们将不会进行建议

To increase the accept rate of the model's suggestions, one simple approach is to train a separate model to predict whether the suggestion will be accepted. In 2022, Parth Thakkar [found that GitHub Copilot used this approach](https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals), deriving a "contextual filter score" using a logistic regression model taking 11 features as inputs, including the programming language, whether the previous suggestion was accepted or rejected, the trailing characters before the user's cursor, and other features. It's unknown what signal this model was trained to predict, but our best guess is that it's predicting the likelihood that the user will accept a suggestion if one is shown. When the score is lower than 15%, the suggestion is skipped and nothing is shown.
>  为了提高模型建议的接受率，一个简单的方法是训练一个独立的模型来预测是否建议会被接收
>  GitHub Copilot 采用了这种方法，它通过一个逻辑回归模型计算出一个 “上下文过滤得分”，该模型接收 11 个特征作为输入，包括了编程语言、是否之前的建议被接收或拒绝、光标的尾随字符等等
>  目前尚不清楚该模型被训练为预测什么信号，我们的推测是: 它在预测 “如果向用户展示该建议，它被接收的概率是多少”，如果得分低于 15%，该建议会被跳过，不想用户展示任何内容

This solution is viable, but we wanted a more general mechanism that reused the powerful representation of the code learned by the Tab model. Instead of filtering out bad suggestions, we wanted to alter the Tab model to avoid producing bad suggestions in the first place. Therefore, we opted to use policy gradient methods instead.
>  这个方法是可行的，但我们希望一个更通用的机制，复用 Tab model 学习到的关于代码的 representation
>  故我们不使用一个单独的模型来过滤坏的建议，而是让 Tab model 自己避免提出坏的建议
>  因此我们倾向于使用策略梯度方法

## The policy gradient
Policy gradient methods are a general way to optimize a "policy" (in this case, the Tab model) to increase a "reward". The reward is a number that we assign to every action taken by the policy. By using a policy gradient algorithm, we can update the policy so that it gets a higher average reward in the future.
>  策略梯度方法是一种为了提高奖励而优化策略的通用方法
>  奖励是我们给策略赋予的它执行每个动作得到的一个标量值
>  使用策略梯度算法，我们可以不断更新策略，使得它在未来获得更高的平均奖励

These algorithms work by allowing the policy to behave randomly, observing which actions lead to high or low reward, and then positively reinforcing the actions that led to high reward, while negatively reinforcing the actions that led to low reward.
>  算法的工作方式是让策略随机执行，观察哪个动作带来高奖励和低奖励，然后正向强化导致高奖励的动作，负向强化导致低奖励的动作

To use policy gradient methods to improve Tab, we defined a reward that encourages accepted suggestions while discouraging showing suggestions to the user that aren't accepted. Let's say we want the model to show a suggestion if its chance of being accepted is at least 25%. Then we could assign a reward of 0.75 for accepted suggestions, a reward of -0.25 for rejected suggestions, and a reward of 0 if no suggestion is shown. If the accept chance is p, then the expected reward if the suggestion is shown is 0.75⋅p−0.25⋅(1−p), which is positive exactly when p>0.25. So a policy acting to maximize reward will suggest when it estimates the accept chance is at least 25% and show nothing otherwise.
>  为了使用策略梯度方法提升 Tab model
>  我们定义了一个奖励机制: 鼓励被接收的建议，抑制不被接收的建议
>  假设我们希望在模型的建议的被接收概率达到 25% 时才提供建议，我们可以:
>  - 为被接收的建议赋予奖励 0.75
>  - 为被拒绝的建议赋予奖励 -0.25
>  - 为没有展示的建议赋予奖励 0
>  如果该建议的接收概率是 $p$，那么该建议被展示后，得到的期望奖励值为 $0.75\cdot p - 0.25 \cdot (1-p)$，这个期望值在 $p>0.25$ 时为正数
>  因此要最大化奖励，策略将会在它估计其建议的被接受概率大于 0.25 时才展示建议，否则就不会展示

In practice, we use a more complicated reward that accounts for the size of the suggestion as well as the possibility of jumping to other locations in the code and showing more suggestions, but this illustrates the core idea: rather than explicitly modeling the accept rate, we learn a policy that targets a particular accept rate. Presumably, the model learns in its internal representations a model of the acceptance probability (or at least a model of whether it exceeds 25%), but we leave that up to the optimizer.
>  实践中，我们使用更复杂的奖励，考虑了建议的大小、跳到其他位置的概率、展示更多建议的概率
>  核心思想是: 与其显式地建模接收概率，我们直接学习一个策略，使其输出能够达到特定的接受率水平
>  可以推测，模型可以在它的内部表示中学习到一个对接收概率进行判断的子模型 (或者至少一个判断接收概率是否大于 25% 的二分类子模型)，我们将这个工作交给优化器处理

>  直观上看，就是让模型对自己的输出具有判断，动作: 坏输出不如动作: 不输出，通过训练，使得模型生成坏输出的概率低于不输出的概率

## The importance of on-policy data
To obtain the policy update, we rely on a remarkable fact called the Policy Gradient Theorem, which states that if a policy $\pi(a\mid s, \theta)$ specifies a distribution over actions $a$ in states (i.e. the state of the user's codebase) $s∼P(s)$ parameterized by $\theta$, and the reward is $J(\theta) = E_{s\sim P (s), a\sim \pi (a\mid s, \theta)}[R (s, a)]$, then the gradient of the reward is given by:

$$
\nabla_\theta J(\theta) = E_{s\sim P(s), a\sim \pi(a\mid s, \theta)}[\nabla_\theta \log \pi(a\mid s, \theta)\cdot R(s, a)]
$$

>  策略 $\pi (a\mid s, \theta)$ 指定了在状态 (用户的 codebase 的状态) 分布 $s\sim P(s)$ 下的动作分布，由 $\theta$ 参数化
>  期望奖励为 $J (\theta) = E_{s\sim P (s), a \sim \pi (a\mid s, \theta)}[R (s, a)]$，那么该奖励关于 $\theta$ 的梯度形式如上

This is useful because it's tractable to estimate the right-hand side: we can obtain samples of states and actions $s \sim P(s),a \sim \pi(a∣s,\theta)$ by sampling from the suggestions shown on user requests, we can compute $\nabla_\theta \log⁡ \pi(a∣s, \theta)$ using a framework like PyTorch, and we can compute $R(s,a)$ by looking at whether the user accepted the suggestion. So we can use this equation to get an unbiased estimate of $\nabla_\theta J(\theta)$, allowing us to improve the policy through stochastic gradient descent.
>  我们通过从为用户的请求所展示的建议中采样 $s\sim P (s), a \sim \pi (a\mid s, \theta)$ (当且策略遇到的状态和它执行的动作)，然后利用 PyTorch 的自动微分引擎计算 $\nabla_\theta \log \pi (a\mid s,\theta)$
>  同时，我们通过查看是否用户接收了建议，计算 $R (s, a)$，进而可以计算策略梯度的无偏估计，并通过 SGD 提升策略

>  这里的目标函数没有建模为策略的动作价值函数 $Q_\pi(s, a)$，直接使用了 $R (s, a)$
>  因此策略梯度的计算简单了很多，不涉及 critic 的更新
>  但劣势就是这样的更新不容易让得到的策略考虑未来的奖励/接收概率

However, this only works if the actions are sampled from the policy being optimized. Once we've updated the policy, we no longer have samples from the policy being optimized — we only have samples from the previous policy. To get fresh "on-policy" samples, we need to deploy the new model to users and see how it behaves. That meant we needed good infrastructure in order to quickly deploy a new checkpoint and minimize the time taken between a suggestion being shown to a user and that data making its way to the next step of our training process. 
>  当然，以上的优化要求动作是根据当前策略采样的
>  当策略更新之后，我们就需要收集新的样本，不能使用之前策略的采样
>  为了获得新的 on-policy 样本，我们需要将新模型部署给用户，这意味着我们需要强大的 infra，以便快速部署新的模型 checkpoint，并最小化从用户看到建议到该建议进入下一阶段训练的时间

Currently, it takes us 1.5 to 2 hours to roll out a checkpoint and collect the data for the next step. While this is fast relative to what is typical in the AI industry, there is still room to make it much faster.
>  目前，我们大约需要 1.5-2h 来发布一个检查点并收集用于下一步训练的数据

## A new Tab model
Using the methods described here, we've trained a new Tab model that is now the default in Cursor. This model makes **21% fewer suggestions** than the previous model while having a **28% higher accept rate** for the suggestions it makes. We hope this improves your coding experience and plan to develop these methods further in the future.
>  新的模型的建议减少了 21%，接受率提高了 28%

![Graph showing the improvement in Tab model performance](https://cursor.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fgraph.e630f96d.png&w=3840&q=75&dpl=dpl_6g7TCNhGi5FR5UfvMorRjAN2eJ9S)

Performance improvements of the new Tab model using online