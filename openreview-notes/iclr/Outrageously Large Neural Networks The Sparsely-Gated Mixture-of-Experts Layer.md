>  https://openreview.net/forum?id=B1ckMDqlg

[Noam Shazeer](https://openreview.net/profile?email=noam%40google.com), [*Azalia Mirhoseini](https://openreview.net/profile?email=azalia%40google.com), [*Krzysztof Maziarz](https://openreview.net/profile?email=krzysztof.maziarz%40student.uj.edu.pl), [Andy Davis](https://openreview.net/profile?email=andydavis%40google.com), [Quoc Le](https://openreview.net/profile?email=qvl%40google.com), [Geoffrey Hinton](https://openreview.net/profile?email=geoffhinton%40google.com), [Jeff Dean](https://openreview.net/profile?email=jeff%40google.com)

Keywords: Deep learning

Conflicts: google.com

## ICLR committee final decision 
Comment: The paper uses mixtures of experts to increase the capacity of deep networks, and describes the implementation of such a model on a cluster of GPUs. The proposed mixture model achieves strong performances in language modeling and machine translation.

Decision: Accept (Poster)

## Rebuttal 
Comment: We thank the reviewers for their instructive feedback. We made a significant effort to improve our work both in terms of presentation and content to reflect the reviewers' suggestions. We believe our latest draft clarifies in detail our contributions with respect to the previous state of the field (e.g., conditional computation). 

The main idea of our paper can be summarized as this: Massively increasing the capacity of deep networks by employing efficient, general-purpose conditional computation. 

This idea seems hugely promising and hugely obvious. At first glance, it is utterly shocking that no one had successfully implemented it prior to us. In practice, however, there are major challenges in achieving high performance and high quality. We enumerate these challenges in the introduction section of our new draft. 

Our paper discusses how other authors have attacked these challenges, as well as our particular solutions. While some of our particular solutions (e.g., noisy-top-k gating, the particular batching schemes, the load-balancing loss, even the mixture-of-experts formalism) may not withstand the test of time, our main contribution, which is larger than these particulars, is to **prove by example that efficient, general-purpose conditional computation in deep networks is possible and very beneficial.** 

As such, this is likely a seminal paper in the field. Our apologies for not making this clearer in our first drafts. Please re-read our paper with this in mind, and consider updating your reviews accordingly, if appropriate. In addition to the major changes described above, we have made several other improvements: 
- We added experimental tests of our balancing losses (see Appendix A). 
- We added computational efficiency metrics (TFLOPS/GPU) to our language modeling experiments (see Tables 1, 7 and 8). 
- We added a set of language modeling experiments on a 100 billion word corpus, using MoE models with up to 137 billion parameters. These demonstrate major quality improvements and good computational efficiency up to 68 billion parameters (see Section 5.2). 
- We added a set of experiments on learning a multilingual machine translation model, showing very large improvements over recently published results (see Table 5). - We moved some of the less important content to appendices, bringing the paper length down to 9 pages.

## public comment by George Philipp
***From an interested reader: I would give this at least a 9 rating***
Comment: I read the paper and I feel the ratings are too low. The authors introduce a general-purpose mechanism to scale up neural networks significantly beyond their current size using sparsity of activation, i.e. **by forcing the activation of most neurons in the net to be zero for any given training example.** 

Firstly, I believe the sheer size of the models successfully trained in this paper warrant an 8 rating all by themselves. 

Secondly, we know historically that sparsity of parameters is among the most important modelling principles in machine learning, being used with great success in e.g. Lasso with the l1 penalty, in SVM with the hinge loss and in ConvNets by setting connections outside the receptive field to zero. This paper, in addition to sparsity of parameters (neurons in different experts are not connected) employs sparsity of activation, where the computation path is customized for each training example. It is, as far as I can tell, the first paper to implement this in a practical, scalable and general way for neural networks. If sparsity of activation turns out to be even a small fraction as important as sparsity of parameters, this paper will have a major impact.

Thirdly, I love the computational efficiency of the model presented. The authors achieve extreme sparsity yet fully utilize their GPUs. In particular, the authors design the network in such a way that there are very few connections between active and non-active units. If we have, say, a sparsely activated fully-connected network, most computation would be wasted on network connections that start on active units and end on non-active units. 

Fourthly, the authors discuss and provide a practical and elegant strategy for large-scale cluster implementation, showcasing their technical sophistication. It is perhaps unfortunate that current baseline datasets may not even be able to fully utilize the power of MoE or other to-be-designed networks following similar principles, but models like the one presented here are bound to only become more prominent in the future. 

I would rate this paper at least 9.

### Thank you! 
Comment: I think you said it better than we did. 

We rewrote our paper, clarifying the fact that this is the first demonstration of efficient massively-sparse conditional computation in neural networks, and enumerating the main challenges we encountered. 

We also added language modeling experiments on a larger corpus (section 5.2), demonstrating computational efficiency and significant quality improvements out to 68 billion parameters :)

## official review by AnonReviewer1
Review: This paper proposes a method for significantly increasing the number of parameters in a single layer while keeping computation in par with (or even less than) current SOTA models. The idea is based on using a large mixture of experts (MoE) (i.e. small networks), where only a few of them are adaptively activated via a gating network. 

While the idea seems intuitive, the main novelty in the paper is in designing the gating network which is encouraged to achieve two objectives: **utilizing all available experts (aka importance), and distributing computation fairly across them (aka load).** 

Additionally, the paper introduces two techniques for increasing the batch-size passed to each expert, and hence maximizing parallelization in GPUs. 

Experiments applying the proposed approach on RNNs in language modelling task show that it can beat SOTA results with significantly less computation, which is a result of selectively using much more parameters. Results on machine translation show that a model with more than 30x number of parameters can beat SOTA while incurring half of the effective computation. 

I have the several comments on the paper: 
- I believe that the authors can do a better job in their presentation. The paper currently is at 11 pages (which is too long in my opinion), but I find that Section 3.2 (the crux of the paper) needs better motivation and intuitive explanation. For example, equation 8 deserves more description than currently devoted to it. Additional space can be easily regained by moving details in the experiments section (e.g. architecture and training details) to the appendix for the curious readers. Experiment section can be better organized by finishing on experiment completely before moving to the other one. There are also some glitches in the writing, e.g. the end of Section 3.1. 
- The paper is missing some important references in conditional computation (e.g. [https://arxiv.org/pdf/1308.3432.pdf](https://arxiv.org/pdf/1308.3432.pdf)) which deal with very similar issues in deep learning. 
- One very important lesson from the conditional computation literature is that while we can in theory incur much less computation, in practice (especially with the current GPU architectures) the actual time does not match the theory. This can be due to inefficient branching in GPUs. It would be nice if the paper includes a discussion of how their model (and perhaps implementation) deal with this problem, and why it scales well in practice. 
>  条件计算在理论上可以减少计算量，但实际上的时间消耗可能和理论不匹配，可能的原因是 GPU 上的低效分支

- Table 1 and Table 3 contain repetitive information, and I think they should be combined in one (maybe moving Table 3 to appendix). One thing I do not understand is how does the number of ops/timestep relate to the training time. This also related to the pervious comment.

Rating: 7: Good paper, accept

Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct

### addressing your comments 
Comment: Thank you for your helpful suggestions. We updated our paper to address them as follows: 

Discussion on how to deal with branching and why our approach performs well: We rewrote the Introduction (Section 1) and clearly outlined the key design considerations and challenges to making massively sparse conditional computation work in practice. In the following sections (2, 3, and 4), we discussed our solutions to address all of these design considerations. Finally in Section 5, we showed how applying our approach to massive data sets can best realize the true potential of conditional computing. 

Length: We moved most of the less essential content to appendices, reducing the paper to 9 pages. 

Better explanation of balancing losses: We added a lot of additional explanation and motivation here, including experimental tests and citations to other authors who have encountered the same problem. (See section 4 and appendix A). We also clarify in our new introduction that this issue is only one of several major challenges to implementing efficient conditional computation. 

Related work: We added citations to a range of works on conditional computing and discussed their relation to our work (for example, see Sections 1.1, 1.3, 2.1, and 4). Citations include: 
Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv: 1308.3432, 2013. 
Andrew Davis and Itamar Arel. Low-rank approximations for conditional feedforward computation in deep neural networks. arXiv preprint arXiv: 1312.4461, 2013. 
David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv: 1312.4314, 2013. 
K. Cho and Y. Bengio. Exponentially Increasing the Capacity-to-Computation Ratio for Conditional Computation in Deep Learning. ArXiv e-prints, June 2014. 
Patrick Gallinari Ludovic Denoyer. Deep sequential neural network. arXiv preprint arXiv: 1410.0510, 2014. 
Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. Conditional computation in neural networks for faster models. arXiv preprint arXiv: 1511.06297, 2015. 
A. Almahairi, N. Ballas, T. Cooijmans, Y. Zheng, H. Larochelle, and A. Courville. Dynamic Capacity Networks. ArXiv e-prints, November 2015. 
Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. CoRR, abs/1611.06194, 2016. URL [http://arxiv.org/abs/1611](http://arxiv.org/abs/1611). 06194. 

Experiment tables and figures modification: We removed the (former) Table 3 to appendix C, and modified the description of the figures and related experiments (see Section 5.1).

#### Thanks for addressing my comments
Comment: The paper is indeed improved. I will update my score accordingly. One thing to keep in mind is that most previous approaches where probably concerning single GPU case. It would be nice if you state clearly in the paper that your method is specifically designed for cluster of GPU. On that note, do you think you would be able to get it to perform well also on single GPU?

##### Of course it performs well on a single GPU. 
Comment: Efficient training on a single GPU is simpler than efficient training on multiple GPUs, since the network bandwidth issue goes away. As we mention in section 3.1, our method scales gracefully with the number of devices. This includes scaling down to a single device. 

The one caveat is that the number of parameters is limited to what will fit in memory (about 1B parameters). As can be seen in our experiments (sec. 5.1 and 5.2), increasing the number of parameters to 1 billion can dramatically improve quality. Any of the models in our experiments which contain 1B parameters or fewer could have been trained on a single GPU. Reducing the training cluster from n gpus to one GPU would have increased the training time by a factor of approximately n.

## official review by AnonReviewer3
***Elegant use of MoE for expanding model capacity, but it would be very nice to discuss MoE alternatives in terms of computational efficiency and other factors.***

Review: Paper Strengths: 

-- Elegant use of MoE for expanding model capacity and enabling training large models necessary for exploiting very large datasets in a computationally feasible manner 
-- The effective batch size for training the MoE drastically increased also 
-- Interesting experimental results on the effects of increasing the number of MoEs, which is expected. 

Paper Weaknesses: 
--- there are many different ways of increasing model capacity to enable the exploitation of very large datasets; it would be very nice to discuss the use of MoE and other alternatives in terms of computational efficiency and other factors.

Rating: 6: Marginally above acceptance threshold

Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct

### addressing your comments
Comment: Thank you for your helpful suggestions. We updated our paper to address them as follows: 

Discussion of relation to other work to effectively increase model capacity to enable the exploitation of very large datasets: 
We rewrote the Introduction (Section 1) to clearly outline the key challenges to making efficient conditional computation work in practice. We cited a range of works on conditional computation and discussed their relation to our work (for example see Sections 1.1, 1.3, 2.1, and 4). 

Some of the related works we cited are as follows: 
Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv: 1308.3432, 2013. 
Andrew Davis and Itamar Arel. Low-rank approximations for conditional feedforward computation in deep neural networks. arXiv preprint arXiv: 1312.4461, 2013. 
David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv: 1312.4314, 2013. 
K. Cho and Y. Bengio. Exponentially Increasing the Capacity-to-Computation Ratio for Conditional Computation in Deep Learning. ArXiv e-prints, June 2014. 
Patrick Gallinari Ludovic Denoyer. Deep sequential neural network. arXiv preprint arXiv: 1410.0510, 2014. 
Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. Conditional computation in neural networks for faster models. arXiv preprint arXiv: 1511.06297, 2015. 
A. Almahairi, N. Ballas, T. Cooijmans, Y. Zheng, H. Larochelle, and A. Courville. Dynamic Capacity Networks. ArXiv e-prints, November 2015. 
Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. CoRR, abs/1611.06194, 2016. URL [http://arxiv.org/abs/1611](http://arxiv.org/abs/1611). 06194. 

Discussion of computational efficiency: We added computational efficiency metrics (TFLOPS/GPU) to our language modeling experiments (see Tables 1, 7 and 8).

### Which alternatives? 
Comment: Thanks for the review. We'd love to add a comparison of MoE to alternative ways of increasing model capacity. We are aware of the following alternatives: 
1. Dense layers (conventional layers requiring O(1) computations per parameter per training example). Expanding the number of parameters in the dense layers proportionally increases training time. 
2. Embedding layers (such as the word-embedding layers in our language models). These can be hugely sparse. We can expand the capacity of an embedding layer by either increasing its dimensionality or its feature space. Our belief is that MoE layers are more powerful than embedding layers, as evidenced by the experiments in our paper, where adding an MoE layer improved perplexity even in the presence of a large embedding layer. We attribute this to the fact that MoE layers, similar to dense layers, can handle orders of magnitude more parameter-example-interactions than embedding layers. The embedding layers tend to be limited by network bandwidth, as the parameters need to be sent over the network, whereas the MoE layers are limited only by GPU compute power, which tends to be much greater. 

Other than these, which alternatives do you think are most important to discuss?

## official review by AnonReviewer2
***Nice use of MoE with good results***

Review: This paper describes a method for greatly expanding network model size (in terms of number of stored parameters) in the context of a recurrent net, by applying a Mixture of Experts between recurrent net layers that is shared between all time steps. By process features from all timesteps at the same time, the effective batch size to the MoE is increased by a factor of the number of steps in the model; thus even for sparsely assigned experts, each expert can be used on a large enough sub-batch of inputs to remain computationally efficient. 

Another second technique that redistributes elements within a distributed model is also described, further increasing per-expert batch sizes. 

Experiments are performed on language modeling and machine translation tasks, showing significant gains by increasing the number of experts, compared to both SoA as well as explicitly computationally-matched baseline systems. 

An area that falls a bit short is in presenting plots or statistics on the real computational load and system behavior. While two loss terms were employed to balance the use of experts, these are not explored in the experiments section. It would have been nice to see the effects of these more, along with the effects of increasing effective batch sizes, e.g. measurements of the losses over the course of training, compared to the counts/histogram distributions of per-expert batch sizes. 
>  提到了没有对额外损失项的消融实验 (后来作者应该是补上了)
>  提到没有关于 effective batch size/per-expert batch size 对损失影响，这里作者没有回应

Overall I think this is a well-described system that achieves good results, using a nifty placement for the MoE that can overcome what otherwise might be a disadvantage for sparse computation. 

Small comment: I like Fig 3, but it's not entirely clear whether datapoints coincide between left and right plots. The H-H line has 3 points on left but 5 on the right? Also would be nice if the colors matched between corresponding lines.

Rating: 7: Good paper, accept
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct

### addressing your comments 
Comment: Thank you for your helpful suggestions. We updated our paper to address them as follows: 

Real computational load and system behavior: 
We added concrete computational efficiency metrics (TFLOPS/GPU) to our language modeling experiments. (see Tables 1, 7 and 8). 

Effect of loss terms: We added more detailed discussions related to our balancing losses as well as experimental tests to show their effectiveness (see Section 4 and Appendix A). 

Figure clarity: We modified Figure 3 (which is now Figure 2). Section 5.1 provides full explanations.

## pre-review question by AnonReviewer1
***Difference between Importance and Load***
Question: Can you explain the difference between Importance and Load as per-expert utilization measures? Why you think we need both terms in the loss function? 

Can you motivate the definition of P(x,i) and give an intuitive explanation? I'm not sure I understand what does kth_excluding term do here.

### Explanation of P(x, i) 
Comment: P(x, i) measures the probability that expert i is used for example x, which happens if expert i is among the top k experts for example x. 

Let's denote the values that are being compared by F(x, i). They each consist of a signal term and a noise term. Expert i is used if F(x, i) is in the top k elements of {F(x, \*)}. 

F(x, i) = (x · W_g)\_i + StandardNormal() · Softplus((x · W_noise)\_i) 

Ideally, we would compute the probability P(x, i) relative to a new random choice of noise for all the experts. We can't find a closed form for this, so we fix the noise for the other experts at the chosen values, and compute the probability relative to a new random choice of noise on expert i. 

The new value of F(x, i) will be the top k elements of {F(x, \*)} if and only if it is greater than the k-th highest of the other elements. This is what we mean by kth_excluding. We are determining the threshold value that the new value of F(x, i) will need to exceed in order for expert i to be used.

### Motivation of Importance and Load 
Comment: Yes. I can explain why each of these measures are necessary, and give examples where each measure is balanced while the other is unbalanced. 

The "Load" measure is for performance reasons. It measures the number of examples sent to an expert. If we have a setup where the experts are evaluated in parallel on different GPUs, then performance is limited by the expert that gets the largest number of examples. If, for example, one expert gets 10x as many examples as the others, the other GPUs sit 90% idle during training time, and throughput drops by a factor of 10. 
>  "Load" 度量发送给每个 expert 的样本数量，针对的是性能问题 (负载均衡)

The "Importance" measure is for quality reasons. It measures the sum of the gate values for an expert, across a batch of examples. When we do not assign an "Importance" loss, we have noticed that the network falls into a local minimum where many of the experts always get low (or zero) weights. This is a self-perpetuating situation, since those experts receive little training, don't learn much, and so the gating network continues to avoid them. This happens even without the sparsity (when k=n). Assigning an "Importance" loss breaks this cycle and forces all of the experts to start learning something. 
>  "Importance" 针对的是质量，它衡量了跨一个 batch 内的样本中，每个 expert 的门控值的和
>  如果没有 Importance loss，会进入局部极小，导致仅有少数 expert 始终被激活，许多专家总是得到非常低，接近零的权重，这种情况会自我强化
>  即便没有稀疏性，这种情况也会发生，Importance loss 可以缓解这种现象，让每个专家都能学习

Here is a case where Importance is balanced and Load is not balanced: 
n=4; k=2; batch_size=3; gating network output = 
Example 0: (expert_0: 0.75, expert_3: 0.25) 
Example 1: (expert_1: 0.75, expert_3: 0.25) 
Example 2: (expert_2: 0.75, expert_3: 0.25) 
Importance: (expert_0: 0.75, expert_1: 0.75, expert_2: 0.75, expert_3: 0.75) Load = (expert_0: 1, expert_1: 1, expert_2: 1, expert_3: 3) 

>  Importance 平衡而 Load 不平衡就是某个专家具有大量权重低的样本，虽然权重低，但是还是在 top-k 内，因此这个专家得到的样本就多，而其他专家只有少数权重高的样本，得到的样本就少

Here is a case where Load is balanced and Importance is not balanced: 
n=4; k=2; batch_size=4; gating network output = 
Example 0: (expert_0: 0.99, expert_1: 0.01) 
Example 1: (expert_0: 0.99, expert_1: 0.01) 
Example 2: (expert_2: 0.99, expert_3: 0.01) 
Example 3: (expert_2: 0.99, expert_3: 0.01) 
Importance: (expert_0 : 1.98, expert_1: 0.02, expert_2: 1.98, expert_3: 0.02) Load = (expert_0: 2, expert_1: 2, expert_2: 2, expert_3: 2) 

>  Load 平衡而 Importance 不平衡类似，某些专家得到非常高权重，某些专家得到非常低权重，但仍处于 top-k 内，这样负载可以均衡，但重要性不均衡

We have noticed both of these failure modes, so we include both losses.

## pre-review question by AnonReviewer2
***Multilayer mapping net baseline?***
Question: An additional baseline comparison, which I don't see yet, may be to replace the MoE with a simple 2- or so layer network with similar number of executed parameters or connections as the MoE (but fewer total parameters). E.g., to use the k in the top-k experts (plus the gating net) to find the number of executed params/connections and create a slightly deeper, k layer net matching this number of connections (e.g., k layers each with the same 512x512 matrix size), that replaces the MoE layer. The total number of stored params is fewer here, but the number of executed params and connections should be about the same, and it is a deeper net doing the mapping.

### Done. 
Comment: Thank you for your suggestion. We re-ran our language modeling experiments and included among our baselines models where the MoE was replaced by feed-forward nets with one and four hidden layers. 

Our main reason for re-running the experiments was that we changed our gating network to allow for soft load balancing constraints. This scheme seems cleaner than the hard load-balancing scheme described in the appendix, in that it does not employ batchwise functions in the model - only in the losses. 

We also changed the architecture of the experts to have one hidden layer, similar to our translation experiments, and did a better job of tuning the dropout rate. 

We did not include a model with 30 billion parameters, as we did in the previous revision. Our experiments suggest that this number of parameters is unnecessary for a 1-billion-word corpus. Instead, we added additional models with 4 billion parameters and different amounts of computation. If we have time before publication, we may add experiments with a larger corpus and a larger number of parameters so as to test the hypothesis that adding parameters helps until the number of parameters is on the order of the size of the training corpus.

## pre-review question by AnonReviewer3 
***Use of huge Mixture-of-Experts***
Question: Can you justify the use of Mixture-of-Experts in LM and MT?

### MoE enables training large models that are necessary for training very large datasets in a computationally feasible manner.
Comment: Thank you for your question! For the very large datasets available in LM and MT, most models underfit because they do not have enough weights. The obvious ways of adding more weights makes the training too slow per example so the model cannot be trained on all the data. 

The main motivation for a MoE is that we can have hugely more weights without much increase in the computation time per example because nearly all of the weights are not touched on any particular example.
>  在不增加计算开销的情况下增加参数量