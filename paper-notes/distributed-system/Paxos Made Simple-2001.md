# Abstract 
The Paxos algorithm, when presented in plain English, is very simple. 

# 1 Introduction 
The Paxos algorithm for implementing a fault-tolerant distributed system has been regarded as difficult to understand, perhaps because the original presentation was Greek to many readers [5]. In fact, it is among the simplest and most obvious of distributed algorithms. 

At its heart is a consensus algorithm—the “synod” algorithm of [5]. The next section shows that this consensus algorithm follows almost unavoidably from the properties we want it to satisfy. 
>  Paxos 的核心是一个共识算法—— synod 算法

The last section explains the complete Paxos algorithm, which is obtained by the straightforward application of consensus to the state machine approach for building a distributed system—an approach that should be well-known, since it is the subject of what is probably the most often-cited article on the theory of distributed systems [4]. 

# 2 The Consensus Algorithm 
## 2.1 The Problem 
Assume a collection of processes that can propose values. A consensus algorithm ensures that a single one among the proposed values is chosen. If no value is proposed, then no value should be chosen. If a value has been chosen, then processes should be able to learn the chosen value. 
>  假设有一组可以处理 values 的进程，一个共识算法需要确保这些 proposed values 中只有一个被选择，如果没有 value 被 propose，则不选择任何值
>  如果某个 value 被选择，processes 应该能知道这一 value

The safety requirements for consensus are: 
- Only a value that has been proposed may be chosen, 
- Only a single value is chosen, and 
- A process never learns that a value has been chosen unless it actually has been. 

>  共识的安全性要求是：
>  - 只能选择已经 proposed 的值
>  - 只能选择一个值
>  - 除非每个值确实已被选定，否则进程将永远不会知道该值已被选定

We won’t try to specify precise liveness requirements. However, the goal is to ensure that some proposed value is eventually chosen and, if a value has been chosen, then a process can eventually learn the value. 
>  目标是确保最终会选出某个 proposed value，并且如果已经选出了一个 value，则进程最终会知道这一 value

We let the three roles in the consensus algorithm be performed by three classes of agents: proposers, acceptors, and learners. In an implementation, a single process may act as more than one agent, but the mapping from agents to processes does not concern us here. 
>  共识算法有三类角色：proposer, acceptor, learner
>  在实现中，一个进程可能扮演多个角色/agents，但某个角色由哪些进程扮演在这里并不重要

Assume that agents can communicate with one another by sending messages. We use the customary asynchronous, non-Byzantine model, in which: 
- Agents operate at arbitrary speed, may fail by stopping, and may restart. Since all agents may fail after a value is chosen and then restart, a solution is impossible unless some information can be remembered by an agent that has failed and restarted. 
- Messages can take arbitrarily long to be delivered, can be duplicated, and can be lost, but they are not corrupted. 

>  假设 agents 可以通过消息互相交流，我们采用传统的异步，非拜占庭模型，在该模型中
>  - agents 以任意速度运行，可能会因为停止而失效，可能重启。因为所有 agents 可能在某个 value 被选定后故障，进而重启，除非某个故障并重启的 agent 能够记住某些信息，否则不存在解决方案
>  - 消息传递的时间可以非常长，也可能被重复发送，也可能丢失，但它们不会被篡改

> [!info] 非拜占庭模型 vs 拜占庭模型
>  非拜占庭模型通常用于设计容错性较高的系统，它假设节点会失效或正常工作，但不会有恶意行为，如发送恶意信息。在该模型中，失效节点的行为是可预测的，比如它们一般就停止发送消息或者发送无效消息，消息可能会延迟、丢失、重复，但不会被篡改和伪造
>  拜占庭模型通常用于设计更加安全的系统，它假设了节点不仅会失效，还可能会成为拜占庭故障节点，这些节点会故意发送错误或不一致的消息，试图破化系统的共识

## 2.2 Choosing a Value 
The easiest way to choose a value is to have a single acceptor agent. A proposer sends a proposal to the acceptor, who chooses the first proposed value that it receives. 
>  选择一个 value 最简单的方法是只有一个 acceptor agent
>  一个 proposer 向该 acceptor 发送提案，accetpor 选择它收到的第一个 value

Although simple, this solution is unsatisfactory because the failure of the acceptor makes any further progress impossible. 
>  其问题在于 acceptor 的故障将导致系统无法执行进展

So, let’s try another way of choosing a value. Instead of a single acceptor, let’s use multiple acceptor agents. A proposer sends a proposed value to a set of acceptors. An acceptor may accept the proposed value. The value is chosen when a large enough set of acceptors have accepted it. How large is large enough? To ensure that only a single value is chosen, we can let a large enough set consist of any majority of the agents. Because any two majorities have at least one acceptor in common, this works if an acceptor can accept at most one value. (There is an obvious generalization of a majority that has been observed in numerous papers, apparently starting with [3].) 
>  考虑使用多个 acceptor agents
>  一个 proposer 向一组 acceptor 发送一个 proposed value, acceptor 可以接受该 value，**当有足够多的 acceptors 接受了该 value，该 value 就被选定**
>  “足够多” 可以表示为 a majority of agents，因为任意两个 majorities 至少会存在一个共同的 acceptor，并且一个 acceptor 只能最多接受一个 value，因此在以 majority 为决策基准的情况下，不可能出现同时选出多个 value 的情况 (否则一定存在两个 majority 共同的 acceptor 接受了多个 values，矛盾)

In the absence of failure or message loss, we want a value to be chosen even if only one value is proposed by a single proposer. This suggests the requirement: 
>  在没有故障或消息丢失的情况下，即便只有一个 proposer 提出了一个 value，我们也希望选择一个 value，这表明了以下要求：

P1. An acceptor must accept the first proposal that it receives. 
>  P1. 一个 acceptor 必须接受它收到的第一个 proposal  (否则，在这种情况下，acceptors 一旦忽视了它收到的 value，就没有剩余的 value 了，进而最后没有 value 被选定)

>  P1 的提出是希望在只有一个 proposer 提出一个 value 的情况下，该 value 会被选中而不是被忽略

But this requirement raises a problem. Several values could be proposed by different proposers at about the same time, leading to a situation in which every acceptor has accepted a value, but no single value is accepted by a majority of them. Even with just two proposed values, if each is accepted by about half the acceptors, failure of a single acceptor could make it impossible to learn which of the values was chosen. 
>  但这一要求引发了一个问题，多个 values 可能在同一时间由多个 proposers 提出，导致每个 acceptor 都接受了一个 value，但没有一个 value 是被大多数 acceptor 接受的 (接收者超过半数)
>  即便只有两个 proposed values，如果每个 value 都被大约一半的 acceptors 接受，那么单一 acceptor 的故障可能导致我们无法确定选择哪个 value (数量打平)

P1 and the requirement that a value is chosen only when it is accepted by a majority of acceptors imply that an acceptor must be allowed to accept more than one proposal. We keep track of the different proposals that an acceptor may accept by assigning a (natural) number to each proposal, so a proposal consists of a proposal number and a value. To prevent confusion, we require that different proposals have different numbers. How this is achieved depends on the implementation, so for now we just assume it. A value is chosen when a single proposal with that value has been accepted by a majority of the acceptors. In that case, we say that the proposal (as well as its value) has been chosen. 
>  根据 P1，以及只有一个 value 被大多数 acceptor 接受后才能被选择的要求，我们必须允许一个 acceptor 接受多个 values
>  我们通过为每个 proposal 分配一个自然数编号，来追踪 acceptor 可能接受的不同 proposals，因此，一个 proposal 包含了一个 proposal number 和一个 value
>  不同的 proposal 要求具有不同的编号，如何实现这一点取决于具体实现方式
>  当具有该 value 的单个 proposal 被大多数 acceptors 接受时，该 value 就被选定，在该情况下，我们称该 proposal (及其对应的 value) 被选定

We can allow multiple proposals to be chosen, but we must guarantee that all chosen proposals have the same value. By induction on the proposal number, it suffices to guarantee: 
>  **我们可以允许多个 proposals 被选定，但我们必须确保所有被选中的 proposals 具有相同的 value**，为此，通过归纳法对 proposal 编号进行论证，只需保证以下条件即可：

P2. If a proposal with value $v$ is chosen, then every higher-numbered proposal that is chosen has value $v$ . 
>  P2. 如果一个值为 $v$ 的 proposal 被选中，则所有编号更高的被选中的 proposal 的值也必须是 $v$

Since numbers are totally ordered, condition P2 guarantees the crucial safety property that only a single value is chosen. 
>  因为编号是完全有序的，条件 P2 保证了只有一个 value 被选中，这是关键的安全属性

To be chosen, a proposal must be accepted by at least one acceptor. So, we can satisfy P2 by satisfying: 
>  要被选中，proposal 必须至少被一个 acceptor 接受
>  因此，我们可以通过满足以下条件来满足 P2：

$\mathrm{P2}^{a}$ . If a proposal with value $v$ is chosen, then every higher-numbered proposal accepted by any acceptor has value $\upsilon$ . 
>  $\mathrm {P2}^a$. 如果一个值为 $v$ 的 proposal 被选中，则任何 acceptor 接受的编号更高的 proposal 值也需要为 $v$ 

>  $\mathrm{P2}^a$ 比 P2 的条件还要更强，P2 只要求之后被选中的 proposal 的值为 $v$，而 $\mathrm{P2}^a$ 直接要求之后 acceptor 只能接受值为 $v$ 的 proposal

We still maintain P1 to ensure that some proposal is chosen. Because communication is asynchronous, a proposal could be chosen with some particular acceptor $c$ never having received any proposal. Suppose a new proposer “wakes up” and issues a higher-numbered proposal with a different value. P1 requires $c$ to accept this proposal, violating $\mathrm{P2}^{a}$ . 
>  我们仍然维持 P1，以确保某个 proposal 被选中 (即 acceptor 必须接受它遇到的第一个 proposal)
>  因为通信是异步的，一个 proposal 可能被某个从未收到任何 proposal 的 acceptor $c$ 接受到，假设一个新的 proposer “醒来”，并发出一个更高编号具有不同 value 的 proposal，根据 P1，$c$ 必须接受这个 proposal，这违反了 $\mathrm{P2}^a$ 

Maintaining both P1 and $\mathrm{P2}^{a}$ requires strengthening $\mathrm{P2}^{a}$ to: 
>  为了同时维持 P1 和 $\mathrm{P2}^a$，需要将 $\mathrm{P2}^a$ 加强为：

${\mathrm{P2}}^{b}$ . If a proposal with value $v$ is chosen, then every higher-numbered proposal issued by any proposer has value $v$ . 
>  $\mathrm{P2}^b$. 如果一个带有 value $v$ 的 proposal 被选中，则任何其他 proposer 发出的更高编号的 proposal 也必须带有 value $v$

>  $\mathrm{P2}^b$ 进一步加强了 $\mathrm{P2}^a$，它不要求 acceptor 的接受，而是要求 proposer 之后发出的 proposal 都带有 $v$

Since a proposal must be issued by a proposer before it can be accepted by an acceptor, $\mathrm{P2}^{b}$ implies $\mathrm{P2}^{a}$ , which in turn implies $P2$ . 
>  因为 proposal 必须由 proposer 发出后才能被 acceptor 接受，因此 $\mathrm{P2}^b$ 蕴含了 $\mathrm{P2}^a$，进而蕴含了 $\mathrm{P2}$

To discover how to satisfy $\mathrm{P2}^{b}$ , let’s consider how we would prove that it holds. We would assume that some proposal with number $m$ and value $v$ is chosen and show that any proposal issued with number $n>m$ also has value $\upsilon$ . 
>  为了弄清楚如何满足 $\mathrm{P2}^b$，让我们考虑如何证明它成立
>  我们假设某个编号为 $m$，值为 $v$ 的 proposal 被选中，然后**证明任意编号 $n > m$ 的 proposal 都具有值 $v$**

We would make the proof easier by using induction on $n$ , so we can prove that proposal number $n$ has value $\upsilon$ under the additional assumption that every proposal issued with a number in $m\ldots(n-1)$ has value $\upsilon$ , where $i\ldots j$ denotes the set of numbers from $i$ through $j$ . For the proposal numbered $m$ to be chosen, there must be some set $C$ consisting of a majority of acceptors such that every acceptor in $C$ accepted it. 
>  我们对 $n$ 使用归纳法，假设所有编号在范围 $m\dots(n-1)$ 的 proposal 都具有值 $v$，然后证明编号为 $n$ 的 proposal 具有 value $v$
>  为了使编号为 $m$ 的 proposal 被选中，必须存在一个由多数 acceptors 组成的集合 $C$，其中每个 acceptor 都接受该 proposal

Combining this with the induction assumption, the hypothesis that $m$ is chosen implies: 
>  结合归纳假设和 $m$ 被选中的假设，蕴含了:

Every acceptor in $C$ has accepted a proposal with number in $m\ldots(n-1)$ , and every proposal with number in $m\ldots(n-1)$ accepted by any acceptor has value $v$ . 
>  $C$ 中的任意 acceptor 都接受了一个编号在 $m\dots(n-1)$ 内的 proposal (至少都接受了编号为 $m$ 的 proposal)，并且任何编号在 $m\dots (n-1)$ 内的，被任意 acceptor 接受的 proposal 的值都是 $v$ (归纳假设中，假设了 $m$ 到 $n-1$ 的 proposal 的值都是 $v$)

Since any set $S$ consisting of a majority of acceptors contains at least one member of $C$ , we can conclude that a proposal numbered $n$ has value $v$ by ensuring that the following invariant is maintained: 
>  因为任意包含了大多数 acceptors 的集合 $S$ 包含了至少一个 $C$ 的成员 ($C$ 中的成员至少都已经接受了编号为 $m$，值为 $v$ 的 proposal)，我们可以推断一个编号为 $n$，value 为 $v$ 的 proposal 需要保持以下不变式：

$\mathrm{P2^{c}}$ . For any $\upsilon$ and $n$ , if a proposal with value $v$ and number $n$ is issued, then there is a set $S$ consisting of a majority of acceptors such that either (a) no acceptor in $S$ has accepted any proposal numbered less than $n$ , or (b) $v$ is the value of the highest-numbered proposal among all proposals numbered less than $n$ accepted by the acceptors in $S$ . 
>  $\mathrm{P2}^c$. 对于任意 $v$ 和 $n$，如果一个编号为 $n$，value 为 $v$ 的 proposal 被发出，则会存在一个由多数 acceptors 组成的集合 $S$，使得以下两种情况之一成立：
>  (a) 集合 $S$ 中的任何 acceptor 都没有接受编号小于 $n$ 的任意 proposal (那么根据 P1，$S$ 中的 acceptors ，如果收到这个新的 proposal，都将接受它，那么这个 proposal 将被选中 )
>  (b) $v$ 是集合 $S$ 中所有编号小于 $n$ 的已接受 proposal 中编号最高的 proposal 的值

We can therefore satisfy ${\mathrm{P2}}^{b}$ by maintaining the invariance of $\mathrm{P2^{c}}$ . 
>  我们进而可以通过维护 $\mathrm{P2}^c$ 的不变式来满足 $\mathrm{P2}^b$
>  要维护 $\mathrm{P2}^c$，proposer 必须遵循一定的要求，初始时，proposer 知道大多数集合 $S$ 中的 acceptors 还没有接受任意的 proposal，则它发出的这个 proposal 会被这些 acceptors 接受，进而这个 proposal 会被选中；之后，proposer 要发出新的 proposal 时，需要确认大多数集合 $S$ 中的 acceptors 接受的上一个 proposal 的值是什么，如果它和自己当前要发出的 proposal 的值相同，则允许发出，否则，当前的 proposal 不能发出
>  通过遵循这样的要求，proposer 归纳式地满足了一旦一个值为 $v$ 的 proposal 被大多数 acceptors 接受进而被选中后，它之后发出的 proposal 的值都必须是 $v$

To maintain the invariance of $\mathrm{P2^{c}}$ , a proposer that wants to issue a proposal numbered $n$ must learn the highest-numbered proposal with number less than $n$ , if any, that has been or will be accepted by each acceptor in some majority of acceptors. Learning about proposals already accepted is easy enough; predicting future acceptances is hard. Instead of trying to predict the future, the proposer controls it by extracting a promise that there won’t be any such acceptances. 
>  为了维持 $\mathrm{P2}^c$ 的不变式，一个想要发出编号为 $n$ 的 proposer 必须了解在某组多数 acceptors 中，已经被其中每个 acceptor 接受的编号小于 $n$ 的编号最大的 proposal
>  了解已经被接受的 proposal 相对简单，预测未来的接受则困难，与其尝试预测未来，proposer 通过提取一个承诺来控制未来，即承诺不会存在这样的接受 (也就是 proposer 在发出编号为 $n$ ，值为 $v$ 的 proposal 之前，通过询问了解到了这组 acceptors 已经接受了某个编号为 $m$，值为 $v$ 的 proposal，了解到之后，proposer 会准备发出这个编号为 $n$，值为 $v$ 的 proposal，这是满足 $\mathrm{P2}^c$ 的而在这段时间内，这组 acceptors 应该不能接受其他的编号小于 $n$ 的，值可能不是 $v$ 的 proposal，如果值不是 $v$ 的话，那 proposer 一旦发出当前的这个 proposal，$\mathrm{P2}^c$ 就违反了，要注意此时我们没有对 acceptors 施加约束，因此需要 proposer 请求它们做出承诺)

In other words, the proposer requests that the acceptors not accept any more proposals numbered less than $n$ . This leads to the following algorithm for issuing proposals. 
>  换句话说，proposer 请求 acceptors **不接受任意编号小于 $n$ 的 proposals**，这引发了以下用于发布 proposal 的算法

1. A proposer chooses a new proposal number $n$ and sends a request to each member of some set of acceptors, asking it to respond with: 
    (a) A promise never again to accept a proposal numbered less than $n$ , and 
    (b) The proposal with the highest number less than $n$ that it has accepted, if any. 

I will call such a request a prepare request with number $n$ . 

>  1.一个 proposer 选择一个新的 proposal 编号 $n$，然后向某个 acceptors 集合中的每个成员发送一个请求，要求其响应以下内容：
>  (a) 承诺不再接受编号小于 $n$ 的 proposal，并且
>  (b) 如果有的话，返回它所接受的编号小于 $n$ 且最大的 proposal
>  这样的请求称为编号 $n$ 的 prepare 请求

2. If the proposer receives the requested responses from a majority of the acceptors, then it can issue a proposal with number $n$ and value $\upsilon$ , where $v$ is the value of the highest-numbered proposal among the responses, or is any value selected by the proposer if the responders reported no proposals. 

> 2.如果 proposer 从大多数 acceptors 那里收到了请求回应，则它可以发出编号为 $n$ 且值为 $v$ 的 proposal，其中 $v$ 是回应中编号最高的 proposal 的值 (约束了自己能发出的值)，或者，如果回应中没有包含 proposal (还没有 proposal 被选定)，则选择任意一个值

A proposer issues a proposal by sending, to some set of acceptors, a request that the proposal be accepted. (This need not be the same set of acceptors that responded to the initial requests.) Let’s call this an accept request. 
>  proposer 发出一个 proposal 的方式是向某一组 acceptors 发送一个请求，请求该 proposal 被接受 (proposer 发送的 acceptors 不必是回应其最初请求的那一组 acceptors)，我们称这个请求为 accept request

This describes a proposer’s algorithm. 
>  以上描述了一个 proposer 的算法

What about an acceptor? It can receive two kinds of requests from proposers: prepare requests and accept requests. An acceptor can ignore any request without compromising safety. So, we need to say only when it is allowed to respond to a request. It can always respond to a prepare request. It can respond to an accept request, accepting the proposal, iff it has not promised not to. In other words: 
>  进而考虑一个 acceptor 的算法，一个 acceptor 可以从 proposers 接受两种类型的 requests: prepare request, accept request
>  一个 acceptor 可以忽略任何请求而不影响安全性，因此，我们只需要说明如何允许它回应一个 request 即可：
>  一个 acceptor 总是可以回应一个 prepare 请求；一个 acceptor 当且仅当它没有做出过承诺不接受该 accept 请求时，可以回应一个 accept 请求以接受该 proposal

$\operatorname{P1}^{a}$ . An acceptor can accept a proposal numbered $n$ iff it has not responded to a prepare request having a number greater than $n$ . 
>  $\mathrm{P1}^a$. 一个 acceptor 当且仅当它没有回应任何编号大于 $n$ 的 prepare 请求时 (如果回应了，那么 acceptor 已经做出过承诺，只能接受那个 prepare 请求对应的 accept 请求，其他的需要忽视)，可以接受编号为 $n$ 的 proposal

Observe that $\operatorname{P1}^{a}$ subsumes P1. 
>  $\mathrm{P1}^a$ 包含了 P1 (如果 acceptor 没有承诺过，则当前的 proposal 是它收到的第一个 proposal，需要接受，如果承诺过，则之前承诺的对应的 proposal 才视作是它收到的第一个 proposal)

We now have a complete algorithm for choosing a value that satisfies the required safety properties—assuming unique proposal numbers. The final algorithm is obtained by making one small optimization. 
>  我们现在有了一个完整的算法，用以选择一个满足所需安全属性的值——假设提案编号是唯一的
>  再进行一个小优化，就可以得到最后的算法

Suppose an acceptor receives a prepare request numbered $n$ , but it has already responded to a prepare request numbered greater than $n$ , thereby promising not to accept any new proposal numbered $n$ . There is then no reason for the acceptor to respond to the new prepare request, since it will not accept the proposal numbered $n$ that the proposer wants to issue. So we have the acceptor ignore such a prepare request. We also have it ignore a prepare request for a proposal it has already accepted. 
>  假设一个 acceptor 接收到一个编号为 $n$ 的 prepare 请求，但它已经回应了一个编号大于 $n$ 的 prepare 请求，从而承诺不再接受任何编号为 $n$ 的 proposal
>  那么，acceptor 没有理由对这个请求做出回应，因为它不会接受 proposer 想要发布的编号为 $n$ 的 proposal
>  因此，我们让 acceptor 忽略这个 prepare 请求，同时，我们让它忽略针对它已经接受的 proposal 的 prepare 请求

With this optimization, an acceptor needs to remember only the highest-numbered proposal that it has ever accepted and the number of the highest-numbered prepare request to which it has responded. Because $\mathrm{P2^{c}}$ must be kept invariant regardless of failures, an acceptor must remember this information even if it fails and then restarts. Note that the proposer can always abandon a proposal and forget all about it—as long as it never tries to issue another proposal with the same number. 
>  通过这个优化，一个 acceptor 只需要记住它曾经接受过的编号最高的 proposal，以及它回应过的编号最高的 prepare 请求即可
>  由于无论发生何种故障，都必须保持 $\mathrm{P2}^c$ 的不变式，故一个 acceptor 必须记住这些信息，即便是发生故障并重新启动后
>  注意，proposer 可以随时放弃某个 proposal 并完全忘记它——只要它不再尝试**以相同的编号**发出另一个 proposal

Putting the actions of the proposer and acceptor together, we see that the algorithm operates in the following two phases. 
>  结合 proposer 和 acceptor 的动作，算法分为以下两个阶段：

Phase 1. (a) A proposer selects a proposal number $n$ and sends a prepare request with number $n$ to a majority of acceptors. (b) If an acceptor receives a prepare request with number $n$ greater than that of any prepare request to which it has already responded, then it responds to the request with a promise not to accept any more proposals numbered less than $n$ and with the highest-numbered proposal (if any) that it has accepted. 

>  阶段 1
>  (a) 一个 proposer 选择一个编号 $n$，然后向多数 acceptors 发送带有编号 $n$ 的 prepare 请求
>  (b) 如果一个 acceptor 接受到一个编号 $n$ 的 prepare 请求，并且 $n$ 大于它之前回应过的任何 prepare 请求的编号，则它会回应这个请求，承诺不再接受任何编号小于 $n$ 的 proposal，同时回应中还附上它已经接受的编号最高的 proposal

Phase 2. (a) If the proposer receives a response to its prepare requests (numbered $n$ ) from a majority of acceptors, then it sends an accept request to each of those acceptors for a proposal numbered $n$ with a value $v$ , where $\upsilon$ is the value of the highest-numbered proposal among the responses, or is any value if the responses reported no proposals. (b) If an acceptor receives an accept request for a proposal numbered $n$ , it accepts the proposal unless it has already responded to a prepare request having a number greater than $n$ . 

>  阶段 2
>  (a) 如果 proposer 从多数 acceptors 中接收到了对它编号为 $n$ 的 prepare 请求的回应，则它向这些 acceptors 发送一个编号为 $n$，值为 $v$ 的 accept 请求，其中 $v$ 是回应中编号最高的 proposal 的值，如果没有收到 proposal，则可以是任意值
>  (b) 如果一个 acceptor 接收到一个编号为 $n$ 的 accept 请求，它会接受它，除非它已经回应过一个编号大于 $n$ 的 prepare 请求

A proposer can make multiple proposals, so long as it follows the algorithm for each one. It can abandon a proposal in the middle of the protocol at any time. (Correctness is maintained, even though requests and/or responses for the proposal may arrive at their destinations long after the proposal was abandoned.) 
>  proposer 可以发出多个 proposals，只要它对于每个 proposal 的发出都遵循算法，它也可以在协议执行过程中随时放弃某个 proposal (这样做不会损害正确性，即使被放弃的 proposal 相关的请求或回应可能在被放弃后的时间到达目的地)

It is probably a good idea to abandon a proposal if some proposer has begun trying to issue a higher-numbered one. Therefore, if an acceptor ignores a prepare or accept request because it has already received a prepare request with a higher number, then it should probably inform the proposer, who should then abandon its proposal. This is a performance optimization that does not affect correctness. 
>  如果某些 proposer 已经开始尝试发出更高编号的 proposal，则放弃当前 proposal 就是一个明智的选择
>  因此，如果一个 acceptor 因为它已经收到过带有更高编号的 prepare 请求而忽略了一个 prepare 或 accept 请求时，它应该通知 proposer, proposer 应该随后放弃该 proposal
>  这是一种性能优化措施，不会影响正确性

## 2.3 Learning a Chosen Value 
To learn that a value has been chosen, a learner must find out that a proposal has been accepted by a majority of acceptors. The obvious algorithm is to have each acceptor, whenever it accepts a proposal, respond to all learners, sending them the proposal. This allows learners to find out about a chosen value as soon as possible, but it requires each acceptor to respond to each learner—a number of responses equal to the product of the number of acceptors and the number of learners. 
>  要知道一个 value 已经被选中，一个 learner 必须发现一个 proposal 已经被大多数 acceptors 接受
>  一个显然的算法是让每个 acceptor 在它接受一个 proposal 时向所有的 leaners 回应，发送它们接受的 proposal，这使得 leaners 尽可能快地确定一个值被选中
>  但如果每个 acceptor 向所有 leaners 回应，总的消息数量将是 `num of accetpors x num of leaners`

The assumption of non-Byzantine failures makes it easy for one learner to find out from another learner that a value has been accepted. We can have the acceptors respond with their acceptances to a distinguished learner, which in turn informs the other learners when a value has been chosen. This approach requires an extra round for all the learners to discover the chosen value. It is also less reliable, since the distinguished learner could fail. But it requires a number of responses equal only to the sum of the number of acceptors and the number of learners. 
>  非拜占庭故障的假设使得 learner 容易从另一个 learner 中确认一个值已被接受
>  因此，我们可以让 acceptors 仅向**一个**指定的 learner 回应 (全部的 acceptors 都只告诉一个 learner，这个 learner 再告诉其他 learner)，这个 learner 进而会在某个值被选定的时候告知其他的 leaners
>  该方法需要额外的一轮通信，以便所有的 learners 都能发现选定的 value

More generally, the acceptors could respond with their acceptances to some set of distinguished learners, each of which can then inform all the learners when a value has been chosen. Using a larger set of distinguished learners provides greater reliability at the cost of greater communication complexity. 
>  更一般地，acceptors 可以将它们的接受结果回应给一组 learners，其中每个 learner 随后通知所有其他 learners
>  使用较大的一组 learners 接受结果可以提高可靠性，但也会增大通信复杂性

Because of message loss, a value could be chosen with no learner ever finding out. The learner could ask the acceptors what proposals they have accepted, but failure of an acceptor could make it impossible to know whether or not a majority had accepted a particular proposal. In that case, learners will find out what value is chosen only when a new proposal is chosen. If a learner needs to know whether a value has been chosen, it can have a proposer issue a proposal, using the algorithm described above. 
>  因为信息可能丢失，可能会出现一个值被选定，但没有 learner 发现的情况
>  learner 可以向 acceptor 询问它们已经接受了哪些 proposals，但 acceptor 的故障可能使得此时也无法确定是否大多数 acceptors 已经都接受了某个特定的 proposal
>  此时，learners 只能在新的 proposal 被选定后才能得知哪个值被选中
>  如果一个 learner 想要知道某个值是否已经被选中，它可以让一个 proposer 按照之前的算法发出一个 proposal (如果该 proposal 的值 $v$ 已经被选中，则该 proposal 会被 acceptors 接受，acceptors 告诉 learner, learner 就知道了 $v$ 已经被选中过了，如果该 proposal 的值 $v$ 不是之前被选中的值，则 proposer 根本不会发出来这个 proposal)

## 2.4 Progress 
It’s easy to construct a scenario in which two proposers each keep issuing a sequence of proposals with increasing numbers, none of which are ever chosen. Proposer $p$ completes phase 1 for a proposal number $n_{1}$ . Another proposer $q$ then completes phase 1 for a proposal number $n_{2}>n_{1}$ . Proposer $p$ ’s phase 2 accept requests for a proposal numbered $n_{1}$ are ignored because the acceptors have all promised not to accept any new proposal numbered less than $n_{2}$ . So, proposer $p$ then begins and completes phase 1 for a new proposal number $n_{3}>n_{2}$ , causing the second phase 2 accept requests of proposer $q$ to be ignored. And so on. 
>  容易构造一个场景，其中两个 proposers 各自不断发出一连串编号递增的 proposals，但这些 proposals 中的任何一个都不会被选中
>  proposer $p$ 完成了编号为 $n_1$ 的 proposal 的 phase 1，另一个 proposer $q$ 完成了编号为 $n_2 > n_1$ 的 proposal 的 phase 1
>  由于 acceptors 都承诺了不再接受任何编号小于 $n_2$ 的 proposal，则 proposer $p$ 的 phase 2 的 accept 请求会被忽略
>  因此，proposer $p$ 进而开始并完成了编号为 $n_3 > n_2$ 的 proposal 的 phase 1，导致 proposer $q$ 的 phase 2 的 accept 请求也被忽略，如此反复

To guarantee progress, a distinguished proposer must be selected as the only one to try issuing proposals. If the distinguished proposer can communicate successfully with a majority of acceptors, and if it uses a proposal with number greater than any already used, then it will succeed in issuing a proposal that is accepted. By abandoning a proposal and trying again if it learns about some request with a higher proposal number, the distinguished proposer will eventually choose a high enough proposal number. 
>  为了确保进展，必须选出一个 proposer 作为唯一尝试发出 proposals 的 agent
>  如果该 proposer 可以成功和大多数 acceptors 通信，并且它使用的 proposal 编号大于任何已经使用过的编号，则它将成功发出一个被接受的 proposal
>  通过在得知有更高编号的 proposal 时放弃当前 proposal 并且再次尝试，该 proposer 会最终选择一个足够高的 proposal 编号

If enough of the system (proposer, acceptors, and communication network) is working properly, liveness can therefore be achieved by electing a single distinguished proposer. The famous result of Fischer, Lynch, and Patterson [1] implies that a reliable algorithm for electing a proposer must use either randomness or real time—for example, by using timeouts. However, safety is ensured regardless of the success or failure of the election. 
>  如果系统中的各个部分 (proposer, acceptors, 通信网络) 正常运行，那么通过选举出一个 distinguished proposer 就可以实现 liveness
>  Fischer, Lynch 和 Patterson 的著名结果 [1] 表明，用于选举 proposer 的可靠算法必须使用随机性或真实时间——例如，通过使用 timeouts。然而，无论选举是否成功，安全性都能得到保证
>  (也就是说，不需要选举 distinguished proposer, Paxos 算法本身就保证了安全性，但不保证 liveness，进一步选举出一个 distinguished proposer, Paxos 算法就进一步保证了 liveness)

## 2.5 The Implementation 
The Paxos algorithm [5] assumes a network of processes. In its consensus algorithm, each process plays the role of proposer, acceptor, and learner. The algorithm chooses a leader, which plays the roles of the distinguished proposer and the distinguished learner. 
>  Paxos 算法假设了一个由进程组成的网络，在其共识算法中，每个进程都扮演 proposer, acceptor, learner 的角色
>  算法选择一个 leader，它扮演 distinguished proposer 和 distinguished leaner 的角色

The Paxos consensus algorithm is precisely the one described above, where requests and responses are sent as ordinary messages. (Response messages are tagged with the corresponding proposal number to prevent confusion.) Stable storage, preserved during failures, is used to maintain the information that the acceptor must remember. An acceptor records its intended response in stable storage before actually sending the response. 
>  Paxos 共识算法正如上述描述的算法，其中请求和回应都作为普通消息进行发送 (回应消息会标记上对应的 proposal 编号以防止混淆)
>  稳定存储 (即在故障期间保存信息)，被用于维护 acceptor 必须记住的信息，在 acceptor 实际发送回应之前，会在其稳定存储中记录它将要发送的回应

All that remains is to describe the mechanism for guaranteeing that no two proposals are ever issued with the same number. Different proposers choose their numbers from disjoint sets of numbers, so two different proposers never issue a proposal with the same number. Each proposer remembers (in stable storage) the highest-numbered proposal it has tried to issue, and begins phase 1 with a higher proposal number than any it has already used. 
>  接下来，我们需要描述一种机制，以确保永远不会发出编号相同的两个 proposal
>  不同的 proposers 从不相交的数字集合选择它们的 proposal 编号，故两个不同的 proposers 永远不会发出编号相同的 proposal
>  每个 proposer 会记住 (在稳定存储中) 它尝试发出的最高编号的 proposal，并在 phase 1 中使用比之前任何编号都高的新编号开始

# 3 Implementing a State Machine 
A simple way to implement a distributed system is as a collection of clients that issue commands to a central server. The server can be described as a deterministic state machine that performs client commands in some sequence. 
>  实现分布式系统的一种简单方法是将其作为向中心 server 发出命令的 clients
>  中心 server 可以被描述为一个确定性的状态机，它以某种顺序执行 client 的命令

The state machine has a current state; it performs a step by taking as input a command and producing an output and a new state. For example, the clients of a distributed banking system might be tellers, and the state-machine state might consist of the account balances of all users. A withdrawal would be performed by executing a state machine command that decreases an account’s balance if and only if the balance is greater than the amount withdrawn, producing as output the old and new balances. 
>  该状态机具有当前状态，它的一个 step 是将当前命令作为输入，产生一个输出和一个新的状态
>  例如，分布式银行系统的 clients 可能是柜员，而状态机的状态可能由所有用户的账户余额组成。取款操作通过执行一个减少账户的余额的状态机命令实现，该命令当且仅当账户余额大于取款金额时才执行，并输出旧余额和新余额

An implementation that uses a single central server fails if that server fails. We therefore instead use a collection of servers, each one independently implementing the state machine. Because the state machine is deterministic, all the servers will produce the same sequences of states and outputs if they all execute the same sequence of commands. A client issuing a command can then use the output generated for it by any server. 
>  一个依赖于单一中心 server 的实现会在 server 故障时失效
>  我们进而使用一组 servers，其中每个都独立实现一个状态机，因为状态机是确定性的，如果所有的 servers 都执行相同的命令序列，它们都将产生相同的输出和状态序列
>  发出命令的 client 可以使用任意 server 生成的输出

To guarantee that all servers execute the same sequence of state machine commands, we implement a sequence of separate instances of the Paxos consensus algorithm, the value chosen by the $i^{\mathrm{th}}$ instance being the $i^{\mathrm{th}}$ state machine command in the sequence. Each server plays all the roles (proposer, acceptor, and learner) in each instance of the algorithm. For now, I assume that the set of servers is fixed, so all instances of the consensus algorithm  use the same sets of agents. 
>  为了确保所有的 servers 执行相同的状态机命令，我们实现了一组独立的 Paxos 共识算法实例，由第 i 个实例选择的 value 即为命令序列中的第 i 个状态机命令
>  每个 server 都在算法的每个实例中扮演 proposer, acceptor, learner 的角色
>  目前，我们假设 servers 集合是固定的，故共识算法的所有实例使用的是同一组 agents

In normal operation, a single server is elected to be the leader, which acts as the distinguished proposer (the only one that tries to issue proposals) in all instances of the consensus algorithm. Clients send commands to the leader, who decides where in the sequence each command should appear. If the leader decides that a certain client command should be the $135^{\mathrm{th}}$ command, it tries to have that command chosen as the value of the $135^{\mathrm{th}}$ instance of the consensus algorithm. It will usually succeed. It might fail because of failures, or because another server also believes itself to be the leader and has a different idea of what the $135^{\mathrm{th}}$ command should be. But the consensus algorithm ensures that at most one command can be chosen as the $135^{\mathrm{th}}$ one. 
>  在正常运行中，一个 server 会被选举出作为 leader，它在所有共识算法实例中扮演 distinguished proposer (唯一尝试发出 proposal 的 proposer)
>  clients 向该 leader server 发送命令，如果该 leader server 决定某条特定的客户端命令将是第 135 条命令，则它将尝试让该命令成为共识算法的第 135 个实例所选定的值，通常都会成功，但由于故障，或者由于另一个 server 也相信它是 leader，并且对第 135 条命令有不同的看法，也会出现失败
>  但共识算法确保最多只有一条命令可以成为第 135 条命令

Key to the efficiency of this approach is that, in the Paxos consensus algorithm, the value to be proposed is not chosen until phase 2. Recall that, after completing phase 1 of the proposer’s algorithm, either the value to be proposed is determined or else the proposer is free to propose any value. 
>  该方法的效率关键在于：Paxos 共识算法中，被 proposed 的 value 直到 phase 2 才被选定
>  回忆一下，在完成 proposer 算法的 phase 1 之后，proposer 要么已经确定了需要 propose 的值，要么可以自由选择任意值提出

I will now describe how the Paxos state machine implementation works during normal operation. Later, I will discuss what can go wrong. 
>  现在开始描述在正常运行期间，Paxos 状态机的工作原理
>  之后，会讨论可能出现的问题

I consider what happens when the previous leader has just failed and a new leader has been selected. (System startup is a special case in which no commands have yet been proposed.) 
>  我们考虑之前的 leader 刚刚故障并且新的 leader 已经被选出的情况 (系统启动是一个该情况的特例，此时还没有任何命令被提出)

The new leader, being a learner in all instances of the consensus algorithm, should know most of the commands that have already been chosen. Suppose it knows commands 1–134, 138, and 139—that is, the values chosen in instances 1–134, 138, and 139 of the consensus algorithm. (We will see later how such a gap in the command sequence could arise.) It then executes phase 1 of instances 135–137 and of all instances greater than 139. (I describe below how this is done.) Suppose that the outcome of these executions determine the value to be proposed in instances 135 and 140, but leaves the proposed value unconstrained in all other instances. The leader then executes phase 2 for instances 135 and 140, thereby choosing commands 135 and 140. 
>  新的 leader 在所有的共识算法实例中都是一个 learner，因此应该知道多数已经被选定的命令
>  假设它知道命令 1-134, 138, 139，也就是它知道了共识算法第 1-134, 138, 139 次选定的值，它进而执行 135-137 实例的 phase 1，以及 139 以后的实例的 phase 1
>  假设这些 phase 1 执行的结果决定了 135 和 140 实例中在 phase 2 要提议的值，但其他实例中的要提议的值没有约束，leader 进而会为实例 135, 140 执行 phase 2，进而选择出命令 135, 140 (第 135、第 140 条要执行的命令)

The leader, as well as any other server that learns all the commands the leader knows, can now execute commands 1–135. However, it can’t execute commands 138–140, which it also knows, because commands 136 and 137 have yet to be chosen. The leader could take the next two commands requested by clients to be commands 136 and 137. Instead, we let it fill the gap immediately by proposing, as commands 136 and 137, a special “no-op” command that leaves the state unchanged. (It does this by executing phase 2 of instances 136 and 137 of the consensus algorithm.) Once these no-op commands have been chosen, commands 138–140 can be executed. 
>  leader 和其他知道了 leader 所知道的所有命令的 servers 现在可以执行命令 1-135，但是不能执行命令 138-140，虽然它们知道这些命令 (已经通过共识算法确认了)，但命令 136-137 尚未被选定
>  leader 可以让 client 请求的下两条命令分别为命令 136, 137，但我们则让 leader 立即填补这一空白，让它提议命令 136, 137 为一个特殊的 “无操作” 命令来保持状态不变 (leader 通过执行一致性算法实例 136, 137 的第二阶段来实现这一点)。当这些 “无操作” 命令被选定后，命令 138-140 就可以执行了

Commands 1–140 have now been chosen. The leader has also completed phase 1 for all instances greater than 140 of the consensus algorithm, and it is free to propose any value in phase 2 of those instances. It assigns command number 141 to the next command requested by a client, proposing it as the value in phase 2 of instance 141 of the consensus algorithm. It proposes the next client command it receives as command 142, and so on. 
>  目前，命令 1-140 都已经被选定，leader 也完成了所有 140 之后的共识算法实例的 phase 1，并且可以在这些实例的 phase 2 可以自由提出任意值
>  它将 client 请求的下一个命令作为命令 141，在共识算法实例 141 的 phase 2 中将该命令作为值提出
>  它将 client 请求的下一个命令作为命令 142，在共识算法实例 142 的 phase 2 中将该命令作为值提出

The leader can propose command 142 before it learns that its proposed command 141 has been chosen. It’s possible for all the messages it sent in proposing command 141 to be lost, and for command 142 to be chosen before any other server has learned what the leader proposed as command 141. When the leader fails to receive the expected response to its phase 2 messages in instance 141, it will retransmit those messages. 
>  leader 可以在它得知它提议的命令 141 被选定之前提议出命令 142
>  有可能 leader 发送的所有的对命令 141 的 proposal 都会丢失，并且命令 142 在其他 servers 知道 leader 对命令 141 的提议是什么之前就被选定。当 leader 没有收到它在实例 141 的 phase 2 的消息的预期回应时，它将重新传输这些消息。

If all goes well, its proposed command will be chosen. However, it could fail first, leaving a gap in the sequence of chosen commands. In general, suppose a leader can get $\alpha$ commands ahead—that is, it can propose commands $i+1$ through $i+\alpha$ after commands 1 through $i$ are chosen. A gap of up to $\alpha-1$ commands could then arise. 
>  如果一切顺利，它提出的命令将被选定，但也可能首先失败，从而在已经选定的命令序列中留下一个缺口
>  一般来说，假设 leader 可以提前得到 $\alpha$ 条命令——即它可以在命令 $1.. i$ 被选定后，提出命令 $i+1.. i+\alpha$，那么实际中有可能会出现最多 $\alpha-1$ 条命令的缺口 (只有命令 $i+\alpha$ 被选定，其他的 proposal 消息都丢失了，进而没有被选定)

A newly chosen leader executes phase 1 for infinitely many instances of the consensus algorithm—in the scenario above, for instances 135–137 and all instances greater than 139. Using the same proposal number for all instances, it can do this by sending a single reasonably short message to the other servers. In phase 1, an acceptor responds with more than a simple OK only if it has already received a phase 2 message from some proposer. (In the scenario, this was the case only for instances 135 and 140.) Thus, a server (acting as acceptor) can respond for all instances with a single reasonably short message. Executing these infinitely many instances of phase 1 therefore poses no problem. 
>  一个新选出的 leader 会为无限多的共识算法实例执行 phase 1，在上个情景中，就是为实例 135-137, 和 139 以后的所有实例
>  通过为所有的算法实例使用相同的提案编号，它可以通过向其他 servers 发送仅一条较短的消息就可以完成这个操作
>  在 phase 1，一个 acceptor 只有在它已经从某个 proposer 收到过一条 phase 2 消息的情况下 (在上个情境中，算法实例 135-140 就是这个情况)，它的回应才会比较复杂 (包含 proposal 信息)，否则只是一条简单的 “OK” 消息
>  因此，作为 acceptor 的 server 可以以一条相对较短的消息作为所有的算法实例中的回应
>  因此，为无限多的共识算法实例执行 phase 1 不会造成任何问题

Since failure of the leader and election of a new one should be rare events, the effective cost of executing a state machine command—that is, of achieving consensus on the command/value—is the cost of executing only phase 2 of the consensus algorithm. It can be shown that phase 2 of the Paxos consensus algorithm has the minimum possible cost of any algorithm for reaching agreement in the presence of faults [2]. Hence, the Paxos algorithm is essentially optimal. 
>  因为 leader 故障和选举新的 leader 的情况应该是较少见的，故执行状态机命令的有效成本——即对命令 (值) 达成共识——仅仅是执行共识算法的 phase 2 的成本
>  可以证明，在存在故障的情况下，Paxos 共识算法的第二阶段是任何达成一致性的算法中最低的，因此，Paxos 算法在本质上是最优的

This discussion of the normal operation of the system assumes that there is always a single leader, except for a brief period between the failure of the current leader and the election of a new one. In abnormal circumstances, the leader election might fail. If no server is acting as leader, then no new commands will be proposed. If multiple servers think they are leaders, then they can all propose values in the same instance of the consensus algorithm, which could prevent any value from being chosen. However, safety is preserved—two different servers will never disagree on the value chosen as the $i^{\mathrm{th}}$ state machine command. Election of a single leader is needed only to ensure progress. 
>  以上这段关于系统正常运行的讨论假设了总是存在单个 leader，除了在当前 leader 的故障和新的 leader 的选举之间存在一个短暂时期
>  在异常情况下，leader 选举可能失败，如果没有 server 作为 leader，则没有命令会被提出，如果多个 servers 认为它们是 leaders，则它们都可以在同一个共识算法实例中提出 values，这可能导致任何值都无法被选择
>  但是，安全性仍然是保持的——两个不同的 servers 永远不会对已经选择的第 i 条状态机命令产生分歧，选举出单一的 leader 仅仅是为了保证系统可以 progress

If the set of servers can change, then there must be some way of determining what servers implement what instances of the consensus algorithm. The easiest way to do this is through the state machine itself. The current set of servers can be made part of the state and can be changed with ordinary state-machine commands. We can allow a leader to get $\alpha$ commands ahead by letting the set of servers that execute instance $i+\alpha$ of the consensus algorithm be specified by the state after execution of the $i^{\mathrm{th}}$ state machine command. This permits a simple implementation of an arbitrarily sophisticated reconfiguration algorithm. 
>  如果 servers 集合可能会变化，则必须有一种方法决定哪些 servers 实现了哪些共识算法的实例
>  最简单的方法是通过状态机本身实现这一点，当前的 servers 集合可以作为状态机状态的一部分，并且可以随着普通的状态机命令进行更改
>  我们可以通过让第 $i$ 次状态机命令执行后的下一个状态指定执行第 $i+\alpha$ 个算法实例的 servers 集合，使得 leader 可以提前获取 $\alpha$ 条命令，这种方法可以简单地实现任意复杂的重新配置算法
