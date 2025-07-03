> https://openreview.net/forum?id=HPuSIXJaa9

**Keywords:** reinforcement learning from human feedback, language models, RLHF, preferences

**TL;DR:** Fine-tuning with RLHF is complicated; we show that it doesn't need to be.

#### **Paper Decision**
**Decision:** Accept (oral)
**Comment:**
Reviewers unanimously agree that the paper addresses an important problem and timely gives an excellent alternative to RLHF to LLMs. Theoretical analysis and thorough experimental results (plus the additional experiments in the rebuttal period) make the paper even stronger. The paper potentially has large impact in the LLM community. Therefore I recommend acceptance as an oral.

#### Author Rebuttal by Authors
**Rebuttal:**
We appreciate the detailed feedback provided by the reviewers. We address a few common points in this response. All other questions are addressed in reviewer specific responses.

***Re: Generalization of PPO and DPO***
First, we note that all of the evaluations in the paper compute test win rates on unseen test prompts.

To further assess the performance of PPO and DPO under distribution shifts, we evaluated the PPO and DPO policies from our Reddit TL;DR summarization experiment on a different distribution, news articles in the test split of the CNN/DailyMail dataset, using the best sampling temperatures from TL;DR (0 and 0.25). We computed the GPT-4 win rate against the ground-truth summaries in the datasets, using the same GPT-4 (C) prompt we used for Reddit TL;DR, but replacing the words "forum post" with "news article". We found that for this new distribution, DPO continues to outperform the PPO policy by a significant margin:

**DPO-0 winrate vs ground truth: 0.359 (+/- 0.030)**
**DPO-0.25 winrate vs ground truth: 0.309 (+/- 0.029)**
**PPO-0 winrate vs ground truth: 0.258 (+/- 0.027)**
**PPO-0.25 winrate vs ground truth: 0.230 (0.026)**

While it is not a comprehensive evaluation of generalization of different RLHF algorithms, **this experiment suggests that DPO policies can generalize similarly well to PPO policies, even without training on the additional unlabeled Reddit TL;DR prompts that PPO uses**. We will include this experiment in our next revision, and defer a more extensive investigation to future work.

>  作者关于泛化性的回应就是: 反正试验上没有说明 DPO 的泛化性更差，剩余的分析我们就不管了

***Re: Baselines on Anthropic-HH dialogue***
Some reviewers asked about additional baselines for the dialogue experiment.

We have added a comparison of DPO to the best-of-N method (i.e., generate N samples, return the one with highest reward under the reward model) using a reward model learned on all the preference data; see Fig. 3 in the attached rebuttal figure pdf. **We observe that DPO performs comparably to the strong best-of-N baseline.** Note, this method is computationally infeasible as it requires generating N samples at test-time (for example, 128) and choosing the best one, but, this baseline is often used as an oracle proxy for PPO [1][2]. We use best-of-N because it is easy to implement, as strong as PPO, and public RLHF checkpoints that we found for Anthropic use undocumented training splits and/or skip the SFT phase (e.g., the reciprocate/ppo_hh_gpt-j checkpoint you mentioned) which is non-standard and makes fair comparison with DPO difficult.

[1] Scaling Laws for Reward Model Overoptimization. Gao, Schulman, Hilton.

[2] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback, Dubois, et. al.

***Re: Details about IMDb sentiment task***
We provide additional details for the IMDb sentiment task, and will revise Section 6.1 to include these details in the next revision of the paper. Following the Transformers Reinforcement Learning (TRL) library, the prompts are prefixes from the IMDB dataset of length 2-8 tokens. We use the pre-trained sentiment classifier "siebert/sentiment-roberta-large-english" as a ground-truth reward model and GPT2-large as a base model. We use these larger models as we found the default ones to generate low-quality text and rewards to be somewhat inaccurate. We first use supervised fine-tuning on a subset of the IMDB data for 1 epoch. We then use this model to sample 4 completions for 25000 prefixes and create 6 preference pairs for each prefix using the ground-truth reward model. The RLHF reward model is initialized from the GPT2-large model and trained for 3 epochs on the preference datasets, and we take the checkpoint with the highest validation set accuracy. The “TRL” run uses the same hyper-parameters as the TRL library implementation. Our implementation uses larger batch samples of 1024 per PPO step.

##### **Followup question**
***Official Comment by Reviewer FksS13*** 
**Comment:**
Thank you for the extra experiments and details.

To clarify, for IMDB, are you reporting reward from the ground truth reward model or your learned reward model? (I assume the former)

##### Replying to Followup question
***Official Comment by Authors***
**Comment:**
Yes, the evaluation is reported using the ground truth reward model.

#### Official Review of Submission14322 by Reviewer FksS
**Summary:**
RLHF is generally done by training a reward model on a dataset of preferences and then using on-policy RL to finetune a language model with the reward model. This work proposes to bypass explicit learning of a reward model and finetune the language model directly on the preference dataset. This is conceptually similar to offline RL but the authors derive a direct preference objective such that the training is supervised not RL-based.

Using a standard preference model, a supervised objective (DPO) is derived, theoretically justified, and compared to PPO. The authors test their method on three tasks: sentiment completion (IMDB), summarization (as in Stiennon et al), and single-turn dialogue with Anthropic's HH dataset. They find that DPO achieves higher reward while staying closer to the reference model (KL) in the IMDB task. Using GPT-4 as a proxy for human judgement on summarization, DPO can match PPO's performance while being less sensitive to sampling temperature. They validate GPT-4 with human annotators and find that DPO is even slightly more preferred over PPO. On dialogue, they find that DPO outperforms supervised and re-ranking (again using GPT-4 eval) but do not compare against PPO.

**Soundness:** 4 excellent
**Presentation:** 3 good
**Contribution:** 4 excellent

**Strengths:**
The overall idea is relatively simple but the justification and theorems provide an excellent basis. The paper is timely and gives an excellent alternative to RLHF, that should be notably more efficient. The paper is clearly written and has the potential to be very impactful if used at scale in modern methods. GPT-4 evaluations and the follow-up human evaluations are strong and demonstrate both improved performance and robustness which can be notably bad in RLHF. The authors include the minimal code for DPO in the appendix which is commendable and many GPT-4 evaluation details for excellent reproducibility. The presentation is also very clear and breaking down the DPO loss into its components with clear language is an excellent addition.

**Weaknesses:**
The main weakness is the unexplored main difference between DPO and PPO: generalization. The authors note it in limitations but since PPO is trained on-policy and DPO is trained offline, looking at generalization outside the dataset would be helpful. The work, as is, feels sufficient without those experiments but the difference does play into an inconsistency with some theorems. The other major issue are some incongruenties in the evaluation that the authors should respond to.

>  Reviewer 认为文中没有深入探讨 DPO 和 PPO 的泛化能力区别，作者也认为这是 limitation
>  PPO 是 on-policy 训练，而 DPO 是 off-line 训练，因此探讨二者的泛化性差异将非常有价值

The proofs and theorems contain a hidden assumption that is not explicitly stated. In the formulations for PPO, the optimization uses $y \sim \pi_\theta$ whereas in the formulations for DPO $y\sim D$ i.e. DPO's algorithm is offline and the model learned from its implicit reward has guarantees over the dataset. In contrast, PPO's algorithm is learned over a new set of datapoints sampled from the model during training. This means that Equation 3 is maximized over a different set of data points than Equation 7. So even though Theoerem 1 is correct, Lemma 2 does not fully encompass this situation. 

>  Reviewer 认为，论文中的证明和定理包含了没有显式说明的假设: 在 PPO 的公式中，优化使用的是 $y\sim \pi_\theta$ ，而在 DPO 的公式中，优化使用的是 $y\sim D$，这表明 DPO 是 off-line 算法，模型从其隐式奖励学习，并保证仅限于数据集上
>  相比之下，PPO 则从一组模型训练时采样的一组新数据点上学习，这意味着 Eq 3 是在一组不同于 Eq 7 的数据点上优化的，因此，尽管定理 1 是正确的，但引理 2 并未完全涵盖这种情况

With offline DPO vs on-policy PPO, equivalent reward models could induce different optimal policies since they are being optimized over a different set of points. The section seems to imply that PPO and DPO can induce the same optimal policies because of the ambiguity with $y$. Clarifying this would be helpful. This also means that section 5.2 and the substitution used to make Equation 10 is not exactly accurate as the optimal policy for PPO is not the same as the optimal policy for DPO. I still feel the point of section 5.2 is reasonable but the authors should explicitly note the issue with the substitution.
>  因为 DPO 是 offline 而 PPO 是 on-polilcy，故等价的奖励模型会导致不同的最优策略，因为它们是在不同的一组点上优化
>  文中似乎认为 DPO 和 PPO 能够导出相同的最优策略，因为对 $y$ 进行了模糊处理，故 Reviewer 要求澄清这一点
>  这也意味着用于构造 Eq 10 的代换不是完全正确的，因为 PPO 的最优策略不等于 DPO 的最优策略

There are issues with each of the different evaluations, even though the overall picture does suggest that the method is sound. I will increase my score if the authors simply clarify some questions about the evaluations (not even necessary to do changes / re-run experiments)

For controlled sentiment (IMDB), the authors do not give many details, seem to deviate from previous work on the task without explanation. The GRUE benchmark (RL4LMs) specifically uses GPT-2 base and finds that models should not be supervised-finetuned before RL training. In contrast, the authors use GPT-2 large (not even GPT-2 medium) and do supervised finetuning before PPO. No details of the baseline are given except for the graph legend which implies the PPO baseline comes from the `trl` library, although the authors only claim to use the `trlx` library. It is also unclear whether the evaluation is over the train set or the test set. Since this task is the only one where the authors show the tradeoff between reward achieved and KL, these details seem important.

For summarization, the issue is relatively minor. GPT-4 may be an unfair evaluator (see Large Language Models are not Fair Evaluators) and the work would benefit from evaluating samples in both positions (i.e. option A then B and option B then A)

For Anthropic's HH, the authors claim there is no SFT model available but there are many available pretrained models for this benchmark that the authors do not compare against. The most prominent example being a GPT-J 6B model `reciprocate/ppo_hh_gpt-j` hosted on huggingface which also has wandb runs and repro code available on the `trlx` library. The authors have chosen Pythia 2.8B when there exists a similar Pythia 1.6B PPO-trained on the HH dataset and it is unclear why the authors do not compare to this model or another PPO-trained model.

**Questions:**
What are the compute differences between DPO and PPO? It feels like DPO should be much more efficient and this information would only strengthen your work.

Why do you not compare against ILQL for sentiment? As an offline RL method, it seems the closest RL comparison to DPO

For reproducibility, what exact API are you using for the GPT-4 evaluation (e.g. GPT-0314)

Can you move Figure 2 and 3 to the next pages so it is closer to the text that references it?

**Limitations:**
The authors did quite well to address limitations in the paper, I believe they covered most of my concerns there.

**Flag For Ethics Review:** No ethics review needed.

**Rating:** 8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Code Of Conduct:** Yes

##### Rebuttal by Authors
**Rebuttal:**
Thanks for appreciating our work and providing a great summary for it!

***DPO vs PPO generalization.***
We performed a follow-up experiment to assess the performance of PPO and DPO under distribution shifts. We evaluate the summarization policies trained on Reddit TL;DR data on the CNN/DailyMail dataset, and evaluate the win rate of each policy's sample vs the ground truth summary. We use the sampling temperatures that performed best for TL;DR. For space reasons, please see the general comment for full experimental details.

**Overall, DPO shows superior performance to PPO for OOD inputs as well:**
**DPO-0 winrate vs ground truth: 0.359 (+/- 0.030)**
**DPO-0.25 winrate vs ground truth: 0.309 (+/- 0.029)**
**PPO-0 winrate vs ground truth: 0.258 (+/- 0.027)**
**PPO-0.25 winrate vs ground truth: 0.230 (0.026)**

***PPO $y\sim \pi_\theta$ vs DPO $y\sim D$***  
We note that Eq. 3 and Eq. 7 represent different stages of RLHF. Eq. 3 shows the objective of the policy optimization stage, while Eq. 7 shows the reward learning stage (using the DPO parameterization). Thus the interpretation of y is different in these equations. The DPO reward parameterization allows us to extract the optimal solution to the problem in Eq. 3 exactly, in closed form; this parameterization simply means that we don't have to perform any additional RL optimization (such as online learning with PPO), which conventional RLHF does.

***DPO vs PPO optimal policy for finite data***
In the limit of infinite data and perfect optimization, the policy found by PPO will be equivalent to the DPO policy, which is the optimal policy (wrt the learned DPO reward) in Eq. 4, which holds for all prompts & answers (this assumes the PPO reward model agrees with the implicit DPO reward model; Theorem 1 and our empirical findings show that the two reward model classes have equivalent expressiveness, and our empirical findings suggest that the two parameterizations do achieve very similar classification accuracy on held out preference pairs). However, in the case of finite data or imperfect optimization, PPO might not recover the theoretically optimal policy that DPO does (Eq. 4). Our new generalization experiment in the summarization setting further suggests that **the DPO policy generalizes at least as well**, if not better than, the PPO policy. Finally, Lemma 2 only states equivalence of the optimal policy, but suboptimal policies found from finite data or imperfect optimization may indeed differ for two reward functions in the same equivalence class.

***Substitution in Equation 10***
Thanks for pointing this out; this is a typo. The optimal policy is the one induced by Eq. 4, not Eq. 7. We apologize for the confusion and will correct the mistake in the final version of the paper. The result (closed-form optimal policy) of Eq 4 holds on all prompts and answers, given a reward function. We agree that the paper would benefit from this discussion, and the points raised about the difference in the PPO / DPO problem settings and the optimal policies being different. Specifically, we will revise Section 5.2 to note the discrepancy between optimal policies.

>  作者回避了这个问题，并且最终也没有改
>  Reviewer 提出的这个问题其实很合理，说明作者也无法说明清楚
>  但是毕竟试验证明了方法的有效性...

***IMDb details***
Thanks for raising the concern about the missing details. We provide the details of the controlled sentiment generation task in the common response, along with the details for the PPO baseline. We follow the TRL library which first does an SFT step before running PPO. This is necessary since the experiment uses shorter prompts and the model might deviate from the movie review task. The evaluation is done on the test set. We will add all the details to the paper in the next revision.

***GPT-4 evaluation sample ordering***
Our present evaluation protocol randomly flips options A and B for every evaluation prompt for this reason. We make a brief note of this in Appendix C.1, but will certainly clarify in the main text.

***Anthropic HH baselines***
We have added a comparison of DPO to the oracle best-of-N method using a reward model learned on all the preference data; see Fig. 3 in the attached rebuttal figure pdf. We observe that DPO performs comparably to the best-of-N method. Note, this method is computationally infeasible as it requires generating N samples at test-time (for example, 128) and choosing the best one, but, this baseline is often used as an oracle proxy for PPO [1][2]. We use best-of-N because it is easy to implement, as strong as PPO, and public RLHF checkpoints that we found for Anthropic use undocumented training splits and/or skip the SFT phase (e.g., the reciprocate/ppo_hh_gpt-j checkpoint you mentioned) which is non-standard and makes fair comparison with DPO difficult.

[1] Scaling Laws for Reward Model Overoptimization. Gao, Schulman, Hilton.

[2] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback, Dubois, et. al.

***PPO vs DPO compute***
Depending on design choices DPO can require as little as half as much accelerator memory than PPO, since it does not maintain separate value function and reward models. In terms of run times, we found that DPO training requires about 5-8 times less compute time than the standard RLHF pipeline, depending on the model and dataset.

***ILQL baseline***
It is possible to use many different RL algorithms to align a model with the learned reward function and in this work we focused on PPO due to its prevalence. Concurrent work (Appendix D3 in [1]) evaluated several offline RL algorithms on the Anthropic HH datasets using a learned reward model and found that they yielded little to no improvement. [1] Fine-Tuning Language Models with Advantage-Induced Policy Alignment, Zhu et. al.

***GPT-4 API details***
We used the `gpt-4-0314` for evaluation. We will add the details to the experiments section of the paper.

***Figure 2/3 positioning***
Thanks for the suggestion! We will reformat the paper so that the figures are closer to the text referencing it.

##### Replying to Rebuttal by Authors
***Re: rebuttal***
**Comment:**
I've read the rebuttal and the authors have addressed my concerns. I am increasing my score.

**Generalization** these are very strong result and do suggest that DPO has some of the generalization properties. I would encourage you to add this to the main paper if possible.

**Lemma 2** I agree that this is true in the limit of infinite data. I assumed that you are using the optimal policies given a finite dataset. If you can specify infinite data, then I have no issue with this.

**Anthropic best of n** This is a sufficiently strong baseline and I think it covers my minor issues with evaluation here.

**ILQL** I was not aware of this result, thank you for referencing it

#### Official Review of Submission14322 by Reviewer GfFq
**Summary:**
The authors propose DPO, a method to fine-tune a language model to human preferences that does not use reinforcement learning or learn a reward model, but rather, directly optimizes the language model with preference data.

Through a change of variables, the authors express the reward function in terms of the optimal policy $\pi^*$ and $\pi_{\text{ref}}$. The authors write the human preference distribution in terms of $\pi^*$ and $\pi_{\text{ref}}$, and thus a maximum likelihood objective can be written in terms of $\pi_\theta$ and $\pi_{\text{ref}}$. This formulation allows for a simple DPO gradient update.

The method is equivalent to fitting a reparameterized Bradley-Terry model. With mild assumptions, DPO does not constrain the class of learned reward models.

Experimental results are run on three different open-ended generation tasks. Baselines include PPO, zero-shot and few-shot prompting, a SFT model, and Preferred-Ft. DPO’s reward/KL tradeoff dominates that of PPO. Furthermore, win-rates of DPO compared to other models (or test-set labels) beat baselines.

**Soundness:** 3 good
**Presentation:** 2 fair
**Contribution:** 4 excellent

**Strengths:**
The paper is a significant contribution in a high impact area -- namely, fine-tuning language models to human preferences. DPO opens a new paradigm of learning human preferences without RL, and is a viable and performant alternative to the RLHF pipeline, reducing computational demands, and requires little tuning of hyperparameters

DPO is grounded theoretically, equivalent to fitting a reparameterized Bradley-Terry model. The DPO update is simple and interpretable. Theorem 1 shows we do not necessarily constrain ourselves in terms of the reward model after reparameterization.

Experimental results are good. DPO learns a better Reward / KL tradeoff compared to baselines in the sentiment generation task. DPO also has excellent win-rates compared to a number of baselines as evaluated with GPT-4 on the summarization and single-turn dialogue tasks. Human evaluators show as high correlations with each other as compared with the GPT-4 evaluation.

**Weaknesses:**
Single-turn dialogue experimental results did not compare DPO with a RLHF baseline tuned from the same base model, and it appears that helpful-rejection-sampled data from the Anthropic HH dataset was used, a relatively weaker baseline compared to the helpful-online dataset.

There are very few details on the experimental setup for the IMDB Sentiment Generation task.

Improvements to the clarity of the paper can be made, including more detailed descriptions of experimental setup in the Appendix.

>  这个 Reviewer 要求试验细节

**Questions:**
Please provide experimental details for the IMDB Sentiment Generation task.

Please clarify what dataset, (specifically, whether the “helpful-online” or “helpful-rejection-sampled” data) was used for the single-turn dialogue experiment?

Why does win-rate in the single-turn dialogue task increase with higher sampling temperature in Figure 3?

>  Reviewer 问为什么 single-turn dialogue 任务中 win-rate 随着 sampling temperature 提高

**Limitations:**
Assumes we use Plackett-Luce models to model preferences.

**Flag For Ethics Review:** No ethics review needed.

**Rating:** 8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.

**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Code Of Conduct:** Yes

##### Rebuttal by Authors
**Rebuttal:**
Thanks for assessing our paper to be a significant contribution, and for your feedback!

***Single-turn dialogue experimental results did not compare DPO with a RLHF baseline…***
Please, see Fig. 3 in the rebuttal pdf for an additional baseline to our dialogue experiments, the oracle best-of-N method that uses a reward model learned on all the preference data to pick the best of N samples from the SFT model. We observe that DPO performs comparably to the best-of-N method. Note that best-of-N is computationally infeasible as it requires generating N samples at test-time (for example, 128) and choosing the best one, but this baseline is often used as an oracle proxy for PPO [1] and achieves comparable performance (Table 2 in [2]). We use best-of-N because it is easy to implement, as strong as PPO, and public RLHF checkpoints that we found for Anthropic use undocumented training splits and/or skip the SFT phase, which is non-standard and makes fair comparison with DPO difficult.

[1] Scaling Laws for Reward Model Overoptimization. Gao, Schulman, Hilton.

[2] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback, Dubois, et. al.

***There are very few details on the experimental setup for the IMDB Sentiment Generation task…***
Thanks for the suggestion! Here are additional details about the IMDB task, which we will add to Section 6.1 of the next revision of the paper:

Following the trl library, the prompts are prefixes from the IMDB dataset of length 2-8 tokens. We use the pre-trained sentiment classifier "siebert/sentiment-roberta-large-english" as a ground-truth reward model and GPT2-large as a base model. We use these larger models as we found the default ones to generate low-quality text and rewards to be somewhat inaccurate. We first use supervised fine-tuning on a subset of the IMDB data for 1 epoch. We then use this model to sample 4 completions for 25000 prefixes and create 6 preference pairs for each prefix using the ground-truth reward model. The RLHF reward model is initialized from the GPT2-large model and trained for 3 epochs on the preference datasets, and we take the checkpoint with the highest validation set accuracy. The “TRL” run uses the same hyper-parameters as the trl library implementation. Our implementation uses larger batch samples of 1024 per PPO step.

***Please clarify what dataset, (specifically, whether the “helpful-online” or “helpful-rejection-sampled” data) was used for the single-turn dialogue experiment?***
We used the entire Anthropic/hh-rlhf available on HuggingFace. We will add the details to the paper.

***Why does win-rate in the single-turn dialogue task increase with higher sampling temperature in Figure 3?***
For low-temperature (0.25) evaluation in the dialogue task, we observe a significant number of responses that get stuck in repetition loops, a common failure mode of smaller LMs being sampled at low temperature. For the two higher-temperature (0.7, 1.0) evaluations, the win rates are within one standard error of each other. We will note this observation of repetition loops at low temperature in Section 6.2 of the revised paper.

>  采样温度较低时，模型会变得更加确定性，倾向于以非常该的概率选择下一个 token，如果最可能的下一个 token 会导向模型已经生成过的短语或模型，它就会陷入循环，一直重复相同的单词或短语，这是小语言模型中常见的问题，因为较小的模型的内部表示比较不多样化，且对上下文的理解有限
>  (例如比较笨的人在词穷的时候就会一直 “额额啊啊”)
>  较高的采样温度会引入更多的不确定性，利于模型探索，提高生成的多样性和连贯性
>  存在一个温度临界点，超过之后，模型的随机性过大，性能反而下降
>  因此，更高温度的作用主要是缓解了重复循环

##### Replying to Rebuttal by Authors
***Official Comment by Reviewer GfFq***
**Comment:**
I have read the response as well as the common points from the authors. I increase my score to 8.

#### Official Review of Submission14322 by Reviewer cpnK
**Summary:**
This paper presents a new policy optimization algorithm with human preferences, called Direct Preference Optimization (DPO), which eliminates the need for explicitly fitting a reward model. The authors formulate the process of RLHF into a single stage policy learning by leveraging a mapping between reward functions and optimal policies. The main contribution of this paper is that they theoretically derive a new objective function that is equivalent to existing RLHF methods but can be simply learned by a single objective, and experimentally shows that it has better performance.

**Soundness:** 3 good
**Presentation:** 3 good
**Contribution:** 4 excellent

**Strengths:**
- This paper presents a new algorithm that can simply learn the policy from a given human preference dataset without explicit reward learning.
- This paper not only experimentally shows better results than RLHF, but also provides theoretical connections with RLHF.

**Weaknesses:**
- Since the reward model is not explicitly learned and the policy is learned only with the given human preference dataset, there is a possibility that the generalization performance for unseen prompts is relatively low compared to the existing RLHF.
- Experiments are insufficient to show that DPO is more stable than conventional RLHF. In order to show that the DPO can be learned more stably, it seems that the learning curve should be provided together with the results shown in the experiment.
- A detailed description of the experiment is lacking. (ex. amount of human preference dataset used in the experiment)
- Depending on the amount of human preference dataset, DPO may have better or worse performance than RLHF, but there is no analysis or explanation about this.

>  该 Reviewer 认为 DPO 的缺点有:
>  - 没有显式学习奖励模型，策略仅在人类偏好数据集上学习，可能会导致策略对于未见过的 prompt 的泛化性能低于 RLHF
>  - 没有试验显示 DPO 比传统 RLHF 更稳定，应该在提供试验结果的同时提供学习曲线
>  - 没有详细描述试验 (例如试验中使用的人类偏好数据集数量)
>  - 没有分析在 scale up/down (即增大人类偏好/减小人类偏好数据集大小和数量) 的情况下，DPO 和 RLHF 的性能比较

**Questions:**
- In Figure 2 of the experiment results, why does DPO perform better than PPO-GT even though PPO-GT uses a true reward function?
- The values of the target KL used in PPO-GT are large (target KL ∈ {3, 6, 9, 12}). Is there a reason why the hyperparameter search range is set this way? In the previous paper (ex. [1]), hyperparameter search was done in a much wider range (target KL ∈ {0.02, 0.05, 0.1, inf}), and there were good results at low values, so I wonder if the results of the PPO-GT used in the paper were suboptimally learned.
- In the results according to the sampling temperature, why does DPO have robust results to the sampling temperature?
- How does the fluency (or naturalness) of the generated sentences from the learned policy according to training iteration change in the DPO experiment? In the case of RLHF, the fluency of the generated sentence shows a result that is sensitive to the coefficient of the KL divergence term (i.e. Fluency deteriorates as learning progresses in RLHF algorithm). I wonder if DPO does not have a similar problem.
- Also, it was mentioned in the paper that DPO learning is stable regardless of hyperparameter beta, but I wonder if learning according to various beta values is stable without degeneration issues (i.e. degrading the fluency).
- Unlikelihood learning (i.e. decreasing the likelihood) objectives are easy to degrade fluency (or naturalness), but how is it possible to stably learn while including unlikelihood learning objectives in DPO? It seems that the main difference between the equation for the gradient of the DPO and unlikelihood learning is that the weight term is multiplied. What exactly does the weight do to make learning stable?
- What does the star mean in the right plot of Figure 2?

>  该 Reviewer 提出的问题有:
>  - Figure 2 中，为什么 DPO 甚至比 PPO-GT 的表现还好
>  - PPO-GT 中使用的目标 KL 散度值比较大，而在之前的工作中，KL 散度值在较小的情况下的效果也很好，所以作者有没有探究 KL 散度值在较小的情况下，PPO-GT 的性能
>  - 为什么 DPO 在采样温度相关的结果下更健壮
>  - DPO 试验中，生成的句子的流畅程度是如何根据训练迭代数量变化的，在 RLHF 中，生成的句子的流畅程度对于 KL 散度项的系数很敏感，是否 DPO 也有这样的问题
>  - 文中提到 DPO 的学习是稳定的，无论超参数 $\beta$ 是多少，但该 Reviewer 质疑这一结论 (这一个问题和上一个问题重复了)
>  - 降低似然的目标容易降低生成的句子的流畅程度，但为什么 DPO 中可以用这样的目标稳定训练，似乎 DPO 的目标中和降低似然的目标中的差异就是乘上的权重项，这个权重项是如何让学习稳定的
>  - Figure 2 中右图的 star 的意思

>  这个 Reviewer 一直在质疑 DPO 的学习稳定性，以及在好奇为什么 DPO 不会降低生成具体的流畅程度

[1] Rajkumar Ramamurthy and Prithviraj Ammanabrolu et al, Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization, ICLR 2023

**Limitations:**
- Since the reward model is not explicitly learned and used, generalization to unseen prompts may not work well.
- A comparison of RLHF and DPO in various experimental settings is lacking.
- As mentioned in Questions and Weaknesses, it seems that there are still some parts that are not clearly explained or verified.
- Details of the experimental setup are omitted.

>  Reviewer 认为的限制有:
>  - 没有显示和使用奖励模型，对没见过的 prompt 的泛化性能可能不好
>  - 没有在多样的试验设定下比较 RLHF 和 DPO

**Flag For Ethics Review:** No ethics review needed.

**Rating:** 7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence:** 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

**Code Of Conduct:** Yes

##### Rebuttal by Authors
**Rebuttal:**
Thanks for your helpful feedback and questions!

***Generalization of DPO vs existing RLHF***
First, we note that all of the evaluations in the paper already compute test win rates on unseen test prompts.

We performed a follow-up experiment to assess the performance of PPO and DPO under distribution shifts. We evaluate the summarization policies trained on Reddit TL;DR data on the CNN/DailyMail dataset, and evaluate the win rate of each policy's sample vs the ground truth summary. We use the sampling temperatures that performed best for TL;DR. For space reasons, please see the general comment for full experimental details.

**Overall, DPO shows superior performance to PPO for OOD inputs as well:**
**DPO-0 winrate vs ground truth: 0.359 (+/- 0.030)**
**DPO-0.25 winrate vs ground truth: 0.309 (+/- 0.029)**
**PPO-0 winrate vs ground truth: 0.258 (+/- 0.027)**
**PPO-0.25 winrate vs ground truth: 0.230 (0.026)**

> 关于泛化性能:
> 作者提到所有对于未见过的 test prompt 的评估都已经在文中展现
> 作者还做了额外的试验，以评估 PPO 和 DPO 在分布偏移 (即面对没有见过的测试 prompt) 下的性能，结论是 DPO 优于 PPO

***DPO stability***
Thanks for the suggestion! To more clearly illustrate DPO's stability across training runs, we trained 4 DPO models using EleutherAI/Pythia-1b on the Anthropic-HH dataset. Please find the training loss curves in Fig. 2 in the rebuttal figure pdf and the classification accuracy of the implicit DPO reward function in Fig. 4. We find that these learning curves are very consistent across training runs. Further, we evaluate the final GPT-4 win rate of each policy against the chosen response for unseen prompts in the test set. The win rates for the final policy from each training run were 0.426, 0.410, 0.414, 0.438 (standard dev: 0.0110). In comparison, sampling from the first policy four times, changing only the random seed used for sampling, we observe a similar distribution of scores, with slightly higher standard deviation: 0.426, 0.412, 0.406, 0.441 (standard dev: 0.0135). These results suggest that the variance in performance of DPO policies across different training seeds is less significant than the variance due to simply using different random seeds for sampling at test time.

>  关于 DPO 稳定性
>  作者训练了多个 DPO 模型，并报告了 implicit DPO 奖励函数的分类准确性
>  结果是在不同的 training run 中，训练曲线非常一致，且最优学习到的策略的性能也非常一致，说明 DPO 学习到的策略对于随机种子是不敏感的

Additionally, we also experiment with how the $\beta$ hyperparameter controls the KL-divergence, we find that the $\beta$ hyperparameter very reliably controls the KL-divergence of the final policy, as shown in Fig. 1 in the rebuttal figure pdf. In contrast, PPO implementations typically require an adaptive beta (based on a target KL divergence) because the same fixed beta can lead to very different KLs for different training runs.
>  此外，作者还试验探究了 $\beta$ 如何控制 KL 散度，发现 $\beta$ 可以稳定地控制最终策略的 KL 散度，相较之下，PPO 实现通常需要基于目标 KL 散度适应性地调节 $\beta$，因为在不同的 training runs 下，相同的固定 $\beta$ 会导致非常不同的 KL 散度

***Size of human preference dataset***
Thanks for bringing up the missing details. We use publicly available preference datasets for summarization (HuggingFace dataset CarperAI/openai_summarize_comparisons) with 92.5k comparisons and Anthropic HH dialogue (HuggingFace dataset Anthropic/hh-rlhf) with 161k comparisons. We will include these and other details about the preference datasets in the camera ready version. In our experiments, DPO and PPO use the same preference datasets (and thus, the same amount of preference data). While our new experiment evaluating the OOD performance of PPO and DPO gives some reason for optimism about DPO's data efficiency, directly comparing the performance of DPO and PPO as the amount of preference data is varied is an important question for analysis, which we defer to future work. Thank you for this suggestion!

>  关于数据集大小
>  文中已经在 92.5k 个数据的数据集以及 161k 个数据的数据集上进行了试验，更大的数据集上的试验作者不想做了

***Why does DPO perform better than PPO-GT even though PPO-GT uses a true reward function?***
In the controlled sentiment generation problem, we find that the learned PPO reward can achieve accuracy well over 90%. Hence we believe the discrepancy between DPO and PPO is not due to issues with reward modeling, but optimization of the reward. PPO only approximately optimizes the KL-constrained reward problem, while DPO samples from the closed-form optimal policy (without approximation). We hypothesize that PPO is less efficient than DPO, even with the ground truth reward, due to this noisy optimization.

> 关于 DPO 为什么比 PPO-GT 好
> 作者认为性能差异在于对奖励的优化，PPO 仅近似地优化 KL 约束的奖励问题，DPO 则直接从闭式的最优策略中采样 (没有近似)

***KL target values scaling***
Figure 2 reports sequence-level KL, which adjusting for sequence length results in per-token KLs of about 0.05, 0.1, 0.15 and 0.2, comparable to those used in Ramamurthy et. al.

>  关于不同的 KL 散度目标值
>  作者补了试验，性能依旧不差

***Fluency during training***
DPO has a very stable relationship between the KL divergence of the final policy and the hyperparameter $\beta$, as shown in the attached Fig. 1. We do not observe any degeneration as training progresses, however, in the limit of $\beta$ going to zero, the KL constraint vanishes, and DPO would purely maximize the (implicit) learned reward. In this case, optimizing a reward function trained on a small preference dataset would likely show some deterioration.

>  关于是否会降低流畅度
>  训练过程中没有发现 degeneration
>  如果 $\beta \to 0$，KL 约束近似没有，DPO 仅优化隐式的奖励函数，此时，在较小的偏好数据集上优化可能导致 degeneration

***Unlikelihood stability***
Indeed, unlikelihood objectives can degrade fluency. As we note in Section 4, the weight scaling the unlikelihood updates in DPO in would be < 1 for all comparisons, but for comparisons where the preferred completion has a higher reward than the dispreferred completion, the weight can be substantially lower, effectively stopping learning on those examples. Thus, the adaptive weights only change the model whenever the preference pair is ordered incorrectly under the LM, which we hypothesize leads to the stable training regime. Further, as beta is larger, this weight goes to zero more rapidly as the example is learned, effectively stopping optimization more quickly. Nonetheless, we cannot rule out the possibility that carefully tuned unlikelihood methods can also be successfully used for fine-tuning LM on comparison pairs.

>  关于 unlikelihood 目标的稳定性
>  作者认为 unlikelihood 目标确实会降低流畅度，但作者指出文中的 section 4 已经提到，DPO 的 unlikelihood 更新权重总是小于 1，并且只要模型正确理解了偏好 (偏好回应比不偏好回应的概率大)，权重会显著降低，故模型会减慢或停止从这些数据中的学习
>  因此，这样的适应性权重只会在偏好对没有被模型正确认知时才改变模型，故模型可以稳定训练
>  此外，如果 $\beta$ 更大，权重降低到零的速度更快

>  不可能性目标旨在减少模型对于某些 (不期望) 输出的概率，使其 “不那么可能” 被生成
>  如果过度地惩罚某些输出，或者惩罚了本应是流畅文本组成部分的元素，就可能导致模型的文本生成不自然，导致流畅度降低

***Figure 2 star***
Stars represent the win rates as computed in human evaluation, and we include 3 stars corresponding to the win rates for DPO @ 0.25, SFT @ 0.25 and PPO @ 0.0, all compared against PPO @ 0.0. We will clarify this in the figure caption.

##### Replying to Rebuttal by Authors
***Thanks for authors' rebuttal!  by AC***
**Comment:**
Reviewer cpnK, did the authors address your concerns about generalization performance for unseen prompts, DPO stability, as well as other concerns? Thanks.

##### Replying to Rebuttal by Authors
***Response to the rebuttal***
**Comment:**
Thank you for the rebuttal and the additional experiments. Since the authors have addressed most of my main concerns (generalization performance for unseen prompts and DPO stability), I would like to raise my score. I thank the authors for their response.

#### Official Review of Submission14322 by Reviewer eRhN
**Summary:**
This paper tackles a very important problem in LLM training, i.e., how to simplify the complex process of reinforcement learning from human feedback? RLHF is different to use, it is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the LLM using RL to maximize this estimated reward. The authors propose DPO, a training paradigm for training language models from preferences without reinforcement learning. DPO identifies a mapping between language model policies and reward functions that enables training a language model to satisfy human preferences directly, with a cross-entropy loss, without reinforcement learning. The experimental results show that DPO can fine-tune LLMs as well or better than existing algorithms.

**Soundness:** 4 excellent
**Presentation:** 4 excellent
**Contribution:** 4 excellent

**Strengths:**
This article addresses a crucial issue in LLM training, i.e., how to perform alignment in a simpler way without using RL. The proposed DPO is remarkably simple yet effective, and it exhibits good theoretical properties. I believe DPO is highly significant for researchers in the field and will serve as a powerful tool for their work.

**Weaknesses:**
This is a very solid piece of work. The proposed method is simple yet effective. I don't have any particular concerns or issues with it.

**Questions:**
None

**Limitations:**
Yes

**Flag For Ethics Review:** No ethics review needed.

**Rating:** 8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.

**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Code Of Conduct:** Yes

##### Rebuttal by Authors
**Rebuttal:**
Thank you for your detailed summary of the paper, your feedback, and believing that our work is highly significant for researchers!

##### Replying to Rebuttal by Authors
***Official Comment by Reviewer eRhN***
**Comment:**
I have read the author's response, and I maintain my score and recommend accepting this paper.