# Abstract 
Raft is a consensus algorithm for managing a replicated log. It produces a result equivalent to (multi-)Paxos, and it is as efficient as Paxos, but its structure is different from Paxos; this makes Raft more understandable than Paxos and also provides a better foundation for building practical systems. In order to enhance understandability, Raft separates the key elements of consensus, such as leader election, log replication, and safety, and it enforces a stronger degree of coherency to reduce the number of states that must be considered. Results from a user study demonstrate that Raft is easier for students to learn than Paxos. Raft also includes a new mechanism for changing the cluster membership, which uses overlapping majorities to guarantee safety. 
>  Raft 是一个用于管理 replicated log 的共识算法，Raft 的结果等同于 (multi-) Paxos，且和 Paxos 一样高效，但其结构不同，更易于理解
>  为了增强可理解性，Raft 将共识的关键要素，例如 leader election, log replication, safety 分离，并且采用更强的一致性 (coherency) 约束以减少需要考虑的状态数量
>  用户研究显示，Raft 相较于 Paxos 更易于学习
>  Raft 也包括了一个新的集群成员变更机制，通过 overlap majorities 来确保安全性

# 1 Introduction 
Consensus algorithms allow a collection of machines to work as a coherent group that can survive the failures of some of its members. Because of this, they play a key role in building reliable large-scale software systems. Paxos [15, 16] has dominated the discussion of consensus algorithms over the last decade: most implementations of consensus are based on Paxos or influenced by it, and Paxos has become the primary vehicle used to teach students about consensus. 
>  共识算法使得一组机器可以作为一个协调一致的整体运行，即便某些成员可能会发生故障
>  故共识算法在构建可靠的大规模系统中起到关键作用
>  Paxos 是过去十年共识算法讨论的核心，多数共识算法的实现都基于 Paxos

Unfortunately, Paxos is quite difficult to understand, in spite of numerous attempts to make it more approachable. Furthermore, its architecture requires complex changes to support practical systems. As a result, both system builders and students struggle with Paxos. 
>  不幸的是，Paxos 难以理解，此外，其架构需要复杂的改动才能支持实际的系统

After struggling with Paxos ourselves, we set out to find a new consensus algorithm that could provide a better foundation for system building and education. Our approach was unusual in that our primary goal was understandability: could we define a consensus algorithm for practical systems and describe it in a way that is significantly easier to learn than Paxos? Furthermore, we wanted the algorithm to facilitate the development of intuitions that are essential for system builders. It was important not just for the algorithm to work, but for it to be obvious why it works. 
>  我们开始寻找一种新的共识算法，为系统构建和教学提供更好的基础
>  我们的方法与众不同，因为我们的主要目标是可理解性：我们能否为实用系统定义一个共识算法，并且比 Paxos 更易于学习
>  我们还希望该算法能够促进开发人员形成构建系统所必须的直觉，这不仅仅要求算法能够正常工作，更重要的是让人们清楚地直到为什么能够正常工作

The result of this work is a consensus algorithm called Raft. In designing Raft we applied specific techniques to improve understandability, including decomposition (Raft separates leader election, log replication, and safety) and state space reduction (relative to Paxos, Raft reduces the degree of nondeterminism and the ways servers can be inconsistent with each other). 
>  本工作的结果是 Raft 共识算法，在设计 Raft 时，我们应用了特定的技术以提高其可理解性，包括 decomposition (Raft 分离了 leader election, log replication, safety) 和 state space reduction (相较于 Paxos， Raft 减少了非确定性的程度以及 servers 之前可能存在的不一致性)

A user study with 43 students at two universities shows that Raft is significantly easier to understand than Paxos: after learning both algorithms, 33 of these students were able to answer questions about Raft better than questions about Paxos. 

Raft is similar in many ways to existing consensus algorithms (most notably, Oki and Liskov’s Viewstamped Replication [29, 22]), but it has several novel features: 

- Strong leader: Raft uses a stronger form of leadership than other consensus algorithms. For example, log entries only flow from the leader to other servers. This simplifies the management of the replicated log and makes Raft easier to understand. 
- Leader election: Raft uses randomized timers to elect leaders. This adds only a small amount of mechanism to the heartbeats already required for any consensus algorithm, while resolving conflicts simply and rapidly.
- Membership changes: Raft’s mechanism for changing the set of servers in the cluster uses a new joint consensus approach where the majorities of two different configurations overlap during transitions. This allows the cluster to continue operating normally during configuration changes. 

>  Raft 在许多方面和现有共识算法相似，但具有几个新特点
>  - Strong leader: Raft 采用比其他共识算法更强的领导机制，例如，日志条目仅从 leader 流向其他 servers。这简化了 replicated log 的管理，并让 Raft 更易于理解
>  - Leader election: Raft 使用随机计时器来选取 leader。这仅在其他共识算法使用的心跳机制上添加了少量额外机制，同时以简单快速的方法解决冲突
>  - Membership changes: Raft 用于更改集群中 server 集合的机制使用了新的联合共识方法，在过渡期间，两个不同配置的 majorities 会重叠，这使得集群在配置变更的过程中能够正常运行

We believe that Raft is superior to Paxos and other consensus algorithms, both for educational purposes and as a foundation for implementation. It is simpler and more understandable than other algorithms; it is described completely enough to meet the needs of a practical system; it has several open-source implementations and is used by several companies; its safety properties have been formally specified and proven; and its efficiency is comparable to other algorithms. 
>  我们认为 Raft 不仅在教育目的上还是用作实现的基础，都由于 Paxos 和其他共识算法，它比其他算法更简单而易于理解，它的描述详细，足以满足实现实际系统的需要
>  Raft 已经有了多个开源实现，并且被若干公司采用
>  其安全性质已经被形式上定义并且证明，并且其性能和其他算法相当

The remainder of the paper introduces the replicated state machine problem (Section 2), discusses the strengths and weaknesses of Paxos (Section 3), describes our general approach to understandability (Section 4), presents the Raft consensus algorithm (Sections 5–8), evaluates Raft (Section 9), and discusses related work (Section 10). 

# 2 Replicated state machines 
Consensus algorithms typically arise in the context of replicated state machines [37]. In this approach, state machines on a collection of servers compute identical copies of the same state and can continue operating even if some of the servers are down. 
>  共识算法通常出现在 replicated state machine 的背景下
>  在复制状态机方法中，一组服务器上的状态机计算相同状态的相同副本，并且可以在部分服务器出现故障的情况下继续运行

Replicated state machines are used to solve a variety of fault tolerance problems in distributed systems. For example, large-scale systems that have a single cluster leader, such as GFS [8], HDFS [38], and RAMCloud [33], typically use a separate replicated state machine to manage leader election and store configuration information that must survive leader crashes. Examples of replicated state machines include Chubby [2] and ZooKeeper [11]. 
>  复制状态机被用于解决一系列分布式系统中的容错问题，例如具有单个集群领导者的大规模系统，像 GFS, HDFS, RAMCloud，通常使用一个独立的复制状态机来管理 leader election，以及存储必须在 leader 崩溃仍然需要保持可用的配置信息

![[pics/Raft-Fig1.png]]

Replicated state machines are typically implemented using a replicated log, as shown in Figure 1. Each server stores a log containing a series of commands, which its state machine executes in order. Each log contains the same commands in the same order, so each state machine processes the same sequence of commands. Since the state machines are deterministic, each computes the same state and the same sequence of outputs. 
>  复制状态机通常使用一个复制日志实现，如 Figure 1
>  每个服务器存储一个包含一系列命令的日志，其状态机会顺序执行这些命令，每个服务器的日志应该包含相同顺序的相同命令，以使得每个状态机都执行相同的指令序列
>  因为状态机是确定性的，每个状态机将计算出相同的状态和相同的输出序列

Keeping the replicated log consistent is the job of the consensus algorithm. The consensus module on a server receives commands from clients and adds them to its log. It communicates with the consensus modules on other servers to ensure that every log eventually contains the same requests in the same order, even if some servers fail. Once commands are properly replicated, each server’s state machine processes them in log order, and the outputs are returned to clients. As a result, the servers appear to form a single, highly reliable state machine. 
>  共识算法的任务就是保持 replicated log 是一致的
>  服务器上的共识模块从 clients 接收命令，然后将它们加入它的 log 中。共识模块会与其他服务器的共识模块通信，以确保所有的 log 最终都包含相同的顺序的相同的命令，即便这个过程中会出现某些服务器崩溃的情况
>  当 log 是一致时，每个服务器的状态机执行该 log 后，返回给 clients 的输出将是一致的，此时所有的服务器看起来就像一个单一的、高度可靠的状态机器

Consensus algorithms for practical systems typically have the following properties: 
- They ensure safety (never returning an incorrect result) under all non-Byzantine conditions, including network delays, partitions, and packet loss, duplication, and reordering. 
- They are fully functional (available) as long as any majority of the servers are operational and can communicate with each other and with clients. Thus, a typical cluster of five servers can tolerate the failure of any two servers. Servers are assumed to fail by stopping; they may later recover from state on stable storage and rejoin the cluster. 
- They do not depend on timing to ensure the consistency of the logs: faulty clocks and extreme message delays can, at worst, cause availability problems. 
- In the common case, a command can complete as soon as a majority of the cluster has responded to a single round of remote procedure calls; a minority of slow servers need not impact overall system performance. 

>  用于实际系统的共识算法通常有以下性质
>  - 在所有的非拜占庭条件下 (包括网络延迟、划分、丢包、重复、变序) 下，确保安全性 (永远不会返回一个错误的结果)
>  - 只要任意  majority of servers 保持运行并且可以互相通信，以及与 clients 通信，系统就完全可用 (available)。例如，一个五 servers 的集群可以容忍任意两个 server 同时故障。算法都假设了 server 在故障时会直接停止 (不会再恢复)，实践中，servers 可以之后根据稳定存储中的状态恢复，并重新加入集群
>  - servers 不需要依赖于时间来保证 logs 的一致性，故错误的时钟以及极端的消息延迟最多导致可用性问题 (不会影响安全性)
>  - 常见情况下，只要集群中的多数 servers 可以回应一轮 RPC，一个命令的执行即可完成，少数的 slow servers 不会影响整体的系统性能

# 3 What’s wrong with Paxos? 
Over the last ten years, Leslie Lamport’s Paxos protocol [15] has become almost synonymous with consensus: it is the protocol most commonly taught in courses, and most implementations of consensus use it as a starting point. 

Paxos first defines a protocol capable of reaching agreement on a single decision, such as a single replicated log entry. We refer to this subset as single-decree Paxos. Paxos then combines multiple instances of this protocol to facilitate a series of decisions such as a log (multi-Paxos). Paxos ensures both safety and liveness, and it supports changes in cluster membership. Its correctness has been proven, and it is efficient in the normal case. 
>  Paxos 首先定义了一个能够在单一决策 (例如单个 replicated log entry) 上达成一致的协议，该协议称为单判决 Paxos
>  Paxos 结合了多个单判决 Paxos 协议的实例，以支持在一系列决策上 (例如整个 log) 达成一致，得到 multi-Paxos
>  Paxos 同时保证安全性和活性，并且支持集群成员组成的变更，Paxos 的正确性已被证明，并且在正常情况下，Paxos 是高效的

Unfortunately, Paxos has two significant drawbacks. The first drawback is that Paxos is exceptionally difficult to understand. The full explanation [15] is notoriously opaque; few people succeed in understanding it, and only with great effort. As a result, there have been several attempts to explain Paxos in simpler terms [16, 20, 21]. These explanations focus on the single-decree subset, yet they are still challenging. In an informal survey of attendees at NSDI 2012, we found few people who were comfortable with Paxos, even among seasoned researchers. We struggled with Paxos ourselves; we were not able to understand the complete protocol until after reading several simplified explanations and designing our own alternative protocol, a process that took almost a year. 
>  Paxos 有两大缺点
>  其一是非常难以理解，少有人真正理解它，也有若干对 Paxos 的解释，这些解释聚焦于单判决 Paxos

We hypothesize that Paxos’ opaqueness derives from its choice of the single-decree subset as its foundation. Single-decree Paxos is dense and subtle: it is divided into two stages that do not have simple intuitive explanations and cannot be understood independently. Because of this, it is difficult to develop intuitions about why the single-decree protocol works. The composition rules for multi-Paxos add significant additional complexity and subtlety. We believe that the overall problem of reaching consensus on multiple decisions (i.e., a log instead of a single entry) can be decomposed in other ways that are more direct and obvious. 
>  我们认为 Paxos 的晦涩性来自于它选择了单判决子集作为其基础
>  单判决 Paxos 被划分为两个阶段，这两个阶段没有直观的解释，且难以独立理解，因此难以对单判决 Paxos 为何有效形成直觉
>  multi-Paxos 的组合规则进一步增加了复杂性
>  我们认为，在多个决策上 (即整个日志而不是单个日志条目) 达成共识的整个问题可以按照更加直接和显而易见的方式分解

The second problem with Paxos is that it does not provide a good foundation for building practical implementations. One reason is that there is no widely agreed-upon algorithm for multi-Paxos. Lamport’s descriptions are mostly about single-decree Paxos; he sketched possible approaches to multi-Paxos, but many details are missing. There have been several attempts to flesh out and optimize Paxos, such as [26], [39], and [13], but these differ from each other and from Lamport’s sketches. Systems such as Chubby [4] have implemented Paxos-like algorithms, but in most cases their details have not been published. 
>  Paxos 的第二个问题是它没有为构建实用实现提供良好基础，一个原因是没有一个被广泛接收的 multi-Paxos 的算法描述/实现。Lamport 的描述主要聚焦于 single-decree Paxos，仅概述了 multi-Paxos 的可能方法

Furthermore, the Paxos architecture is a poor one for building practical systems; this is another consequence of the single-decree decomposition. For example, there is little benefit to choosing a collection of log entries independently and then melding them into a sequential log; this just adds complexity. It is simpler and more efficient to design a system around a log, where new entries are appended sequentially in a constrained order. Another problem is that Paxos uses a symmetric peer-to-peer approach at its core (though it eventually suggests a weak form of leadership as a performance optimization). This makes sense in a simplified world where only one decision will be made, but few practical systems use this approach. If a series of decisions must be made, it is simpler and faster to first elect a leader, then have the leader coordinate the decisions. 
>  并且，Paxos 架构并不适合构建实际系统，这也是 Paxos 采用的 single-decree 分解导致的 (multi-Paxos 由 single-decree Paxos 组合而成，故 Paxos 采用的是 single-decree 分解)
>  例如，独立地选择一组日志条目，然后再将它们组合为一个顺序的日志不会带来好处，仅增加复杂性。相比之下，直接围绕一个日志设计系统会更为简单高效，新的条目会以受限的顺序附加到日志中
>  另一个问题是 Paxos 的核心采用了对称的点对点方法，这仅在需要做出单一决策的理想场景中合理，但实际系统很少采用该方法。如果需要进行一系列决策，更加简单快速的方式是选举出一个 leader，让 leader 协调决策

As a result, practical systems bear little resemblance to Paxos. Each implementation begins with Paxos, discovers the difficulties in implementing it, and then develops a significantly different architecture. This is time-consuming and error-prone, and the difficulties of understanding Paxos exacerbate the problem. Paxos’ formulation may be a good one for proving theorems about its correctness, but real implementations are so different from Paxos that the proofs have little value. The following comment from the Chubby implementers is typical: 

There are significant gaps between the description of the Paxos algorithm and the needs of a real-world system. . . . the final system will be based on an unproven protocol [4]. 

>  因此，实际系统和原生 Paxos 几乎没有相似之处，每个实现都从 Paxos 出发，在实现时发现问题时，会开发出另一个显著不同的架构
>  Paxos 的设计适合于证明关于其正确性的定理，但实际实现和 Paxos 相差太大，故这些证明并没有太多实际价值

Because of these problems, we concluded that Paxos does not provide a good foundation either for system building or for education. Given the importance of consensus in large-scale software systems, we decided to see if we could design an alternative consensus algorithm with better properties than Paxos. Raft is the result of that experiment. 
>  故我们认为 Paxos 不适合作为系统构建和教学的基础
>  Raft 则是比 Paxos 性质更好的替代性共识算法

# 4 Designing for understandability 
We had several goals in designing Raft: it must provide a complete and practical foundation for system building, so that it significantly reduces the amount of design work required of developers; it must be safe under all conditions and available under typical operating conditions; and it must be efficient for common operations. But our most important goal—and most difficult challenge—was understandability. It must be possible for a large audience to understand the algorithm comfortably. In addition, it must be possible to develop intuitions about the algorithm, so that system builders can make the extensions that are inevitable in real-world implementations. 
>  Raft 的设计目标包括：Raft 需要为系统构建提供完整且实用的基础，以减少开发者的设计工作；Raft 需要在任意条件下保持安全性，并且在典型的操作条件下保持可用性；Raft 对于常见操作必须高效
>  最重要的且最有挑战性的目标是可理解性，Raft 需要能让大量受众理解该算法，以便系统构建者在实际实现中 (基于理解) 进行拓展

There were numerous points in the design of Raft where we had to choose among alternative approaches. In these situations we evaluated the alternatives based on understandability: how hard is it to explain each alternative (for example, how complex is its state space, and does it have subtle implications?), and how easy will it be for a reader to completely understand the approach and its implications? 
>  在设计 Raft 时，存在许多情况，需要从多个替代方案中进行选择，我们以可理解性作为评估度量：解释该替代方案有多困难？(例如，其状态空间有多复杂，以及是否存在微妙的含义)，以及读者要完全理解该方法及其影响有多容易？

We recognize that there is a high degree of subjectivity in such analysis; nonetheless, we used two techniques that are generally applicable. The first technique is the well-known approach of problem decomposition: wherever possible, we divided problems into separate pieces that could be solved, explained, and understood relatively independently. For example, in Raft we separated leader election, log replication, safety, and membership changes. 
>  这类分析中存在大量主观性，但我们也使用了两个通用的方法
>  其一是问题分解：只要可能，我们会将问题分解为可以单独解决、解释、理解的相对独立的多个部分
>  例如，Raft 中，我们分离了 leader election, log replication, safety, membership changes

Our second approach was to simplify the state space by reducing the number of states to consider, making the system more coherent and eliminating nondeterminism where possible. Specifically, logs are not allowed to have holes, and Raft limits the ways in which logs can become inconsistent with each other. Although in most cases we tried to eliminate nondeterminism, there are some situations where nondeterminism actually improves understandability. In particular, randomized approaches introduce nondeterminism, but they tend to reduce the state space by handling all possible choices in a similar fashion (“choose any; it doesn’t matter”). We used randomization to simplify the Raft leader election algorithm. 
>  其二是通过减少需要考虑的状态数量来简化状态空间、让系统更加逻辑连贯和一致、尽可能消除非确定性
>  具体地说，logs 不允许有空缺条目 (减少可能的状态数量)，Raft 也限制了 logs 之间会变得不一致的可能方式 (减少可能的状态数量)
>  在多数情况下，我们尝试消除不确定性，但一些情况下，不确定性可以提高可理解性。特别是，随机化方法引入了不确定性，但这些方法倾向于以类似的方式简化状态空间 ("choose any; it doesn't matter")，我们使用了随机化来简化 Raft 的 leader election 算法

# 5 The Raft consensus algorithm 
Raft is an algorithm for managing a replicated log of the form described in Section 2. Figure 2 summarizes the algorithm in condensed form for reference, and Figure 3 lists key properties of the algorithm; the elements of these figures are discussed piecewise over the rest of this section. 

![[pics/Raft-Fig2.png]]

Raft implements consensus by first electing a distinguished leader, then giving the leader complete responsibility for managing the replicated log. The leader accepts log entries from clients, replicates them on other servers, and tells servers when it is safe to apply log entries to their state machines. 
>  为了实现共识，Raft 首先选举出一个 leader，该 leader 完全负责管理 replicated log
>  leader 从 clients 接收 entry ，将它们拷贝到其他服务器上，并且告诉其他服务器什么时候可以安全地将 entry 应用于其状态机

Having a leader simplifies the management of the replicated log. For example, the leader can decide where to place new entries in the log without consulting other servers, and data flows in a simple fashion from the leader to other servers. A leader can fail or become disconnected from the other servers, in which case a new leader is elected. 
>  leader 的设计简化了 replicated log 的管理，例如，leader 可以在不询问其他 servers 的情况下决定在哪里放置新的 entry ，数据也直接从 leader 流向其他服务器
>  leader 可能出现故障或断联，此时，新的 leader 需要被选出

Given the leader approach, Raft decomposes the consensus problem into three relatively independent subproblems, which are discussed in the subsections that follow: 

- Leader election: a new leader must be chosen when an existing leader fails (Section 5.2).  
- Log replication: the leader must accept log entries 
- Safety: the key safety property for Raft is the State Machine Safety Property in Figure 3: if any server has applied a particular log entry to its state machine, then no other server may apply a different command for the same log index. Section 5.4 describes how Raft ensures this property; the solution involves an additional restriction on the election mechanism described in Section 5.2. 

>  在 leader 方法下，Raft 将共识问题分解为三个相对独立的子问题
>  - Leader election: 现存的 leader 故障时，需要选出新 leader
>  - Log replication: leader 必须接收 log entries
>  - Safety: Raft 的关键安全属性是状态机安全属性。该属性描述为：如果任意 server 对其状态机应用了某个 entry ，则任意其他 server 不能对相同的日志索引应用不同的命令 (任意其他 server 只能在这个索引下应用相同的 entry )，为了确保该属性的始终成立，Raft 的解决方案是对选举机制施加了额外限制

![[pics/Raft-Fig3.png]]

After presenting the consensus algorithm, this section discusses the issue of availability and the role of timing in the system. 

## 5.1 Raft basics 
A Raft cluster contains several servers; five is a typical number, which allows the system to tolerate two failures. At any given time each server is in one of three states: leader, follower, or candidate. In normal operation there is exactly one leader and all of the other servers are followers. Followers are passive: they issue no requests on their own but simply respond to requests from leaders and candidates. The leader handles all client requests (if a client contacts a follower, the follower redirects it to the leader). The third state, candidate, is used to elect a new leader as described in Section 5.2. Figure 4 shows the states and their transitions; the transitions are discussed below. 
>  一个 Raft 集群包含多个 servers，在任意给定时间，每个 server 一定处于以下三种状态: leader, follower, candidate
>  常规运行时，只有一个 leader，其他 servers 都是 followers, followers 是被动的: 它们本身不发出任何请求，仅回应 leader 和 candidates 的请求
>  leader 处理所有的客户端请求 (如果某个客户端联系了 follower, follower 会将请求重定向到 leader)
>  candidate 可能会被选举为新的 leader

![[pics/Raft-Fig4.png]]

![[pics/Raft-Fig5.png]]

Raft divides time into terms of arbitrary length, as shown in Figure 5. Terms are numbered with consecutive integers. Each term begins with an election, in which one or more candidates attempt to become leader as described in Section 5.2. If a candidate wins the election, then it serves as leader for the rest of the term. In some situations an election will result in a split vote. In this case the term will end with no leader; a new term (with a new election)  will begin shortly. Raft ensures that there is at most one leader in a given term. 
>  Raft 将时间划分为任意长的任期，任期用连续的整数编号
>  每个任期以选举开始，在选举中，一个或多个 candidates 会尝试成为 leader，如果某位 candidate 赢得了选举，它会在该任期内担任 leader
>  有些情况下，选举会出现分票，此时，该任期以没有 leader 的状态结束，很快会开始一个新的任期以及新的选举
>  Raft 确保在给定的任期内最多只有一个 leader

Different servers may observe the transitions between terms at different times, and in some situations a server may not observe an election or even entire terms. Terms act as a logical clock [14] in Raft, and they allow servers to detect obsolete information such as stale leaders. Each server stores a current term number, which increases monotonically over time. Current terms are exchanged whenever servers communicate; if one server’s current term is smaller than the other’s, then it updates its current term to the larger value. If a candidate or leader discovers that its term is out of date, it immediately reverts to follower state. If a server receives a request with a stale term number, it rejects the request. 
>  不同的 servers 可能会在不同时间内观察到任期之间的转换，以及某些情况下，server 可能不会观察到某个选举甚至整个任期
>  Raft 中，任期充当逻辑时钟，任期帮助 servers 检测过时的信息，例如 stale leaders
>  **每个 server 存储一个当前任期数**，它随着时间单调增长，当 servers 之间相互通讯时，它们会**交换**当前任期，如果某个 server 的当前任期小于另一个 server 的，则它会将自己的当前任期更新到更大的那个值。如果 leader 或 candidate 发现其任期是过期的，它会立即恢复为 follower 状态
>  **如果 server 接收到带有过期任期号的请求，它会拒绝该请求**

Raft servers communicate using remote procedure calls (RPCs), and the basic consensus algorithm requires only two types of RPCs. `RequestVote` RPCs are initiated by candidates during elections (Section 5.2), and `AppendEntries` RPCs are initiated by leaders to replicate log entries and to provide a form of heartbeat (Section 5.3). Section 7 adds a third RPC for transferring snapshots between servers. Servers retry RPCs if they do not receive a response in a timely manner, and they issue RPCs in parallel for best performance. 
>  Raft servers 通过 RPC 通信，且基本的共识算法仅需要两类 RPC: `RequestVote`, `AppendEntries`
>  `RequestVote` RPC 由 candidates 在选举时发起, `AppendEntries` RPC 由 leaders 在复制 entry 以及要提供 heartbeat 消息时发起
>  Section 7 添加了一个新的 RPC 用于在 servers 之间传输快照时使用
>  servers 会在它们没有及时收到一个回应时重新发起 RPCs，并且 servers 会并行地发起 RPCs 以提高性能

## 5.2 Leader election 
Raft uses a heartbeat mechanism to trigger leader election. When servers start up, they begin as followers. A server remains in follower state as long as it receives valid RPCs from a leader or candidate. Leaders send periodic heartbeats (`AppendEntries` RPCs that carry no log entries) to all followers in order to maintain their authority. If a follower receives no communication over a period of time called the election timeout, then it assumes there is no viable leader and begins an election to choose a new leader. 
>  Raft 使用 heartbeat 机制以触发 leader 选举
>  servers 启动时都是 followers，只要 server 从 leader 或者 candidate 收到有效的 RPC，它就保持为 followers
>  leader 会像所有的 followers 定期发送 heartbeat 消息 (即不带有任何 entry 的 `AppendEntries` RPC) 用以维护其地位
>  如果一个 follower 在一段时间内 (称为 election timeout) 没有收到通讯，它假设目前没有有效的 leader，并开始选举以选出新的 leader

To begin an election, a follower increments its current term and transitions to candidate state. It then votes for itself and issues `RequestVote` RPCs in parallel to each of the other servers in the cluster. A candidate continues in this state until one of three things happens: (a) it wins the election, (b) another server establishes itself as leader, or (c) a period of time goes by with no winner. These outcomes are discussed separately in the paragraphs below. 
>  要开启选举，一个 follower 会增加它当前的任期号，然后转移到 candidate 状态。之后，它会为自己投票，然后并行地向集群中所有其他 servers 发起 `RequestVote` RPCs
>  一个 candidate 会在以下三个事件发生之前保持 candidate 状态
>  - 它赢得选举，成为 leader
>  - 另一个 server 成为 leader
>  - 一段时间过去后，没有 leader 出现

A candidate wins an election if it receives votes from a majority of the servers in the full cluster for the same term. Each server will vote for at most one candidate in a given term, on a first-come-first-served basis (note: Section 5.4 adds an additional restriction on votes). The majority rule ensures that at most one candidate can win the election for a particular term (the Election Safety Property in Figure 3). Once a candidate wins an election, it becomes leader. It then sends heartbeat messages to all of the other servers to establish its authority and prevent new elections. 
>  (赢得选举)
>  一个 candidate 在同一任期内收到多数 servers 的投票后，就成为 leader
>  在给定任期内，每个 server 只能为一个 candidate 投票，并且遵循先来先服务的原则 (哪个 candidate 的 `RequestVote` 先到，就为它投票)
>  多数原则确保了在特定的任期内，最多只有一个 candidate 赢得选举 (即确保了 Election Safety Property 成立)
>  candidate 赢得选举后，成为 leader，然后向所有其他的 servers 发送 heartbeat 消息，确立其权威，并防止新的选举发生

While waiting for votes, a candidate may receive an `AppendEntries` RPC from another server claiming to be leader. If the leader’s term (included in its RPC) is at least as large as the candidate’s current term, then the candidate recognizes the leader as legitimate and returns to follower state. If the term in the RPC is smaller than the candidate’s current term, then the candidate rejects the RPC and continues in candidate state. 
>  (另一个 server 成为 leader)
>  candidate 在等待投票的过程中可能从其他声称是 leader 的 server 收到 `AppendEntries` RPC，如果该 leader 的任期 (包含在其 RPC 中) 至少和 candidate 的当前任期一样大，该 candidate 会认可该 leader，并且回退到 follower 状态
>  如果 RPC 中的任期小于 candidate 当前的任期，则 candidate 会拒绝该 RPC，并继续保持 candidate 状态

The third possible outcome is that a candidate neither wins nor loses the election: if many followers become candidates at the same time, votes could be split so that no candidate obtains a majority. When this happens, each candidate will time out and start a new election by incrementing its term and initiating another round of `RequestVote` RPCs. However, without extra measures split votes could repeat indefinitely. 
>  (没有 leader 出现)
>  如果 candidate 既没有赢得选举，也没有输去选举 (如果许多 follower 同时成为 candidates，就可能出现分票，即没有 candidate 得到多数认可)，此时，每个 candidate 会在超时后，增加任期，发起另一轮 `RequestVote` RPC 以重新开始选举
>  但没有额外的措施时，split vote 的情况可能会无限地重复

Raft uses randomized election timeouts to ensure that split votes are rare and that they are resolved quickly. 
>  Raft 使用随机的选举超时时间以确保 spilt vote 情况是少见的，并且可以快速被解决

To prevent split votes in the first place, election timeouts are chosen randomly from a fixed interval (e.g., 150–300ms). This spreads out the servers so that in most cases only a single server will time out; it wins the election and sends heartbeats before any other servers time out. 
>  为了防止 split votes 的发生, (每个 server 的) election timeout 会从某个固定的区间内随机选择 (例如 150-300ms)
>  这样的机制减少了多个 server 同时发起选举的概率，因此，多数情况下，只会有一个 server 超时 (最早超时的那个)，它 (会迅速发起下一次选举，进而) 赢得选举，继而在其他 servers 超时之前发送 heartbeat 消息 

The same mechanism is used to handle split votes. Each candidate restarts its randomized election timeout at the start of an election, and it waits for that timeout to elapse before starting the next election; this reduces the likelihood of another split vote in the new election. Section 9.3 shows that this approach elects a leader rapidly. 
>  为了处理 split votes 的情况 (已经发生了)，Raft 仍然采用相同的机制，split votes 发生后，candidates 会超时，继而开始下一次选举，而每个 candidates 在开启下一次选举前，会重新设置随机 election timeout，**然后会在开启下一次选举之前先等待该 timeout 超时**，这将减少新选举时出现 split vote 的可能性
>  (发生了 split votes 的情况说明即便在随机 election timeout 的情况下，还是出现了部分 servers 在上个任期同时发起了选举，也就是它们随机到了相同的 election timeout，那么显然下一个任期时，为了避免它们再碰一起，需要重设 election timeout，然后在发起下一次选举前等待各自的 election timeout，以将它们错开)

Elections are an example of how understandability guided our choice between design alternatives. Initially we planned to use a ranking system: each candidate was assigned a unique rank, which was used to select between competing candidates. If a candidate discovered another candidate with higher rank, it would return to follower state so that the higher ranking candidate could more easily win the next election. We found that this approach created subtle issues around availability (a lower-ranked server might need to time out and become a candidate again if a higher-ranked server fails, but if it does so too soon, it can reset progress towards electing a leader). We made adjustments to the algorithm several times, but after each adjustment new corner cases appeared. Eventually we concluded that the randomized retry approach is more obvious and understandable. 
>  最初，我们打算采用排名系统：每个 candidate 会被分配一个唯一的排名，用于在 candidates 之间进行选择，如果某个 candidate 发现另一个 candidate 有更高的排名，它将返回 follower 状态，以便更高排名的 candidate 能够更轻松赢得下次选举 (因为 candidates 数量减少了)
>  但我们发现该方法存在隐晦的可用性问题 (如果一个高排名的 server 故障，低排名的 server 可能需要超时，然后再成为 candidate，但如果它这样做得太早，可能会重置选举 leader 的进度)
>  随机重试方法更加直观且易于理解

## 5.3 Log replication 
Once a leader has been elected, it begins servicing client requests. Each client request contains a command to be executed by the replicated state machines. The leader appends the command to its log as a new entry, then issues `AppendEntries` RPCs in parallel to each of the other servers to replicate the entry. When the entry has been safely replicated (as described below), the leader applies the entry to its state machine and returns the result of that execution to the client. 
>  leader 被选出后，它就开始处理客户端请求，每个客户端请求都包含一条要被 replicated state machines 处理的命令
>  leader 会将该命令作为新的 entry 追加到它的 log 中，然后并行向其他 servers 发出 `AppendEntries` RPC，令它们拷贝这一 entry 。当该 entry 已经被安全拷贝后，leader 对其状态机应用这一 entry ，然后将执行结果返回给客户端

If followers crash or run slowly, or if network packets are lost, the leader retries ` AppendEntries ` RPCs indefinitely (even after it has responded to the client) until all followers eventually store all log entries. 
>  如果 followers 崩溃或运行缓慢，或者网络包丢失，leader 会无限次地重试 `AppendEntries` RPC (即便在它已经回应了客户端之后)，直到所有的 followers 最终存储了所有的 entry 

![[pics/Raft-Fig6.png]]

Logs are organized as shown in Figure 6. Each log entry stores a state machine command along with the term number when the entry was received by the leader. The term numbers in log entries are used to detect inconsistencies between logs and to ensure some of the properties in Figure 3. Each log entry also has an integer index identifying its position in the log. 
>  日志的组织方式如 Figure 6 所示，每个 entry 都存储了一条状态机命令，以及 leader 接收到该条目时的任期号
>  entry 中的任期号会用于检测各个 server 的日志之间的不一致性，并且会用于确保 Figure 3 的一些属性
>  每个 entry 还有一个整数索引，用于标识它在整个日志中的位置

The leader decides when it is safe to apply a log entry to the state machines; such an entry is called committed. Raft guarantees that committed entries are durable and will eventually be executed by all of the available state machines. 
>  leader 决定何时可以安全地将一个 entry 应用到状态机，这样的被应用的 entry 被称为已提交的，Raft 保证已提交的 entry 是持久化的，并且最终会被所有可用的状态机执行 (所以应用已提交的 entry 是安全的)

A log entry is committed once the leader that created the entry has replicated it on a majority of the servers (e.g., entry 7 in Figure 6). This also commits all preceding entries in the leader’s log, including entries created by previous leaders.
>  对于一个 entry ，一旦创建该 entry 的 leader 已经将它 replicate 到了多数 servers 上，该 entry 就认为是已提交的
>  leader 提交一个 entry 时，其日志中先前所有的 entry 也会被一并提交，包括了由前任 leader 创建的 entry (即一个 entry 被认为已经被提交时，它之前的所有 entry 都认为已经被提交)
 
Section 5.4 discusses some subtleties when applying this rule after leader changes, and it also shows that this definition of commitment is safe. 

The leader keeps track of the highest index it knows to be committed, and it includes that index in future `AppendEntries` RPCs (including heartbeats) so that the other servers eventually find out. Once a follower learns that a log entry is committed, it applies the entry to its local state machine (in log order). 
>  leader 会跟踪它已知的最高索引的已提交 entry，并且会在它未来的 `AppendEntries` RPC 中附加这一索引，便于其他 servers 最终得知
>  当 follower 了解到某个 entry 已经提交，它就会按照日志顺序，将该 entry 应用到其本地状态机 (已经被提交的 entry 说明 leader 已经将它应用，follower 应该跟上)
 
We designed the Raft log mechanism to maintain a high level of coherency between the logs on different servers. Not only does this simplify the system’s behavior and make it more predictable, but it is an important component of ensuring safety. 
>  我们设计了 Raft log 机制维护不同 servers 的 logs 之间的高层次一致性，这一机制不仅简化了系统的行为，使其更加可预测，还是确保安全性的重要组件

Raft maintains the following properties, which together constitute the Log Matching Property in Figure 3: 

- If two entries in different logs have the same index and term, then they store the same command. 
- If two entries in different logs have the same index and term, then the logs are identical in all preceding entries. 

>  Raft 维护以下两个属性，这两个属性一起构成了 Log Matching Property
>  - 如果不同的 logs 中的两个 entry 有相同的索引和任期，则它们存储了相同的命令
>  - 如果不同的 logs 中的两个 entry 有相同的索引和任期，则这两个 logs 的之前所有 entry 都相同
>  (Log Matching Property: 如果两个 logs 包含了一个具有相同索引和任期的 entry ，则该 entry 之前的所有 entry ，包括该 entry 都是一致的)

The first property follows from the fact that a leader creates at most one entry with a given log index in a given term, and log entries never change their position in the log. 
>  第一个属性源于这样的事实：在一个给定任期内，leader 最多为每个日志索引创建一个 entry，并且 entry 永远不会改变其在日志中的位置 

>  假设该 entry 的索引为 `I`，任期为 `T`，那么在整个系统持续时间内，有且仅有一个索引为 `I` ，任期为 `T` 的 entry，就是该 entry。因为 leader 在任期 `T` 内不会再创建索引为 `I` 的 entry，之后的任期也不可能再创建索引为 `I` 的 entry，因为 entry 被 leader 创建后就不会更改，就算被创建了，其任期也不同

>  那么，无论 leader 将该 entry replicate 到 follower 的 log 中成功与否，都能确保只要某个 log 中出现了索引为 `I` ，任期为 `T` 的 entry，它一定就是该 entry 本身

The second property is guaranteed by a simple consistency check performed by `AppendEntries`. When sending an `AppendEntries` RPC, the leader includes the index and term of the entry in its log that immediately precedes the new entries. If the follower does not find an entry in its log with the same index and term, then it refuses the new entries. The consistency check acts as an induction step: the initial empty state of the logs satisfies the Log Matching Property, and the consistency check preserves the Log Matching Property whenever logs are extended. As a result, whenever `AppendEntries` returns successfully, the leader knows that the follower’s log is identical to its own log up through the new entries. 
>  第二个属性由 `AppendEntries` RPC 执行的一致性检查保证，leader 发送 `AppendEntries` RPC 时，消息中不仅带有新 entry ，还有该 entry 之前的 entry 的索引和任期。如果 follower 在其日志中找不到带有相同索引和任期的 entry ，则会拒绝 append 新 entry 
>  `AppendEntries` 的一致性检查起到了归纳步骤的作用：初始的空日志满足 Log Matching Property，而之后的每次 append 操作执行的一致性检查都确保日志在被拓展后仍保持 Log Matching Property，因此，每当 `AppendEntries` 成功返回时，leader 就知道 follower 的日志到目前为止和它的日志是一致的

>  形式化地说，假设要为 follower 的 log 的索引 `I` 处附加 entry，归纳假设为 “该 log 和 leader 的 log 的 `[0, I-1]` 处完全一致"
>  初始条件下，logs 都为空，满足归纳假设的条件，之后，每次的 `AppendEntries` 执行的一致性检查显然都会在归纳假设满足的条件下，使得 Append 之后的 logs 在 `[0, I]` 处完全一致
>  By induction, 成功的 `AppendEntries` 将确保 logs 被拓展后仍然保持完全一致
>  因此，两个 logs 最后保持一致的地方就是对二者同时的 `AppendEntries` 最后成功的地方

>  也可以考虑反证法来直接形式化证明第二个属性的成立
>  假设两个 logs 在索引 `I+N` 处具有相同的 entry，但在之前的某个索引 `I`  处的 entries 存在不同
>  那么 `AppendEntries` 将在索引 `I+1` 处开始就会失败，那么 follower 的 log 将不会和 leader 的 log 一样长，更不可能在之后的某个索引 ` I+N` 处具有相同的 entry ，和事实矛盾
>  因此，`AppendEntries` 将确保两个 logs 在 `I+N` 处具有相同 entry 时，之前的所有 entries 都相同

During normal operation, the logs of the leader and followers stay consistent, so the `AppendEntries` consistency check never fails. However, leader crashes can leave the logs inconsistent (the old leader may not have fully replicated all of the entries in its log). These inconsistencies can compound over a series of leader and follower crashes. 
>  在正常运行的情况下，leader 和 followers 的日志将保持一致，故 `AppendEntries` 的一致性检查将永远不会失败
>  但是，leader 崩溃可能导致日志不一致 (旧的 leader 可能没有完全 replicate 其日志中的所有条目，导致一部分 follower 的日志有新条目，一部分没有)，这些不一致会随着一系列 leader 和 follower 的崩溃而累积 
>  (因为一旦 `AppendEntries` 的一致性检查失败而 Append 失败，如果不处理现存的不一致问题的话，在该任期内，之后的 `AppendEntries` 也都将失败)

Figure 7 illustrates the ways in which followers’ logs may differ from that of a new leader. A follower may be missing entries that are present on the leader, it may have extra entries that are not present on the leader, or both. Missing and extraneous entries in a log may span multiple terms. 
>  Figure 7 描述了 follower 的日志与新 leader 的日志可能存在差异的方式，follower 可能缺少 leader 中存在的 entry ，follower 可能会有 leader 中没有的额外 entry，或者两种情况同时存在
>  日志中的缺失和多余的 entry 可能跨越多个任期

![[pics/Raft-Fig7.png]]

In Raft, the leader handles inconsistencies by forcing the followers’ logs to duplicate its own. This means that conflicting entries in follower logs will be overwritten with entries from the leader’s log. Section 5.4 will show that this is safe when coupled with one more restriction. 
>  Raft 中，leader 通过强迫 follower 的日志与其自身的日志保持一致来解决不一致性
>  这意味着 follower 日志中与 leader 日志中冲突的 entry 将被 leader 日志中的 entry 覆盖，在增加一个额外的限制条件下，这样做就是安全的

To bring a follower’s log into consistency with its own, the leader must find the latest log entry where the two logs agree, delete any entries in the follower’s log after that point, and send the follower all of the leader’s entries after that point. All of these actions happen in response to the consistency check performed by `AppendEntries` RPCs. 
>  为了使 follower 的日志和自己的保持一致，leader 必须先找到 follower 日志中和自己日志一致的最新的 entry，之后让 follower 删除其日志中该 entry 后的所有 entries，然后向 follower 发送自己日志中该 entry 后的所有 entries
>  这些动作都是对 `AppendEntries` RPC 的一致性检查做出的响应 (在一致性检查失败时的处理方式)

The leader maintains a `nextIndex` for each follower, which is the index of the next log entry the leader will send to that follower. When a leader first comes to power, it initializes all `nextIndex` values to the index just after the last one in its log (11 in Figure 7). 
>  leader 会为每个 follower 维护一个 `nextIndex` ，`nextIndex` 代表了 leader 将发送给 follower 的下一个 entry 的索引
>  当 candidate 成为 leader 时 (新的任期开始)，它会为所有 follower 初始化 `nextIndex` 值，将其初始化为自己的日志中最新的 entry 的下一项的索引 (也就是 candidate 一开始假设所有 follower 的 logs 都和它一致)

If a follower’s log is inconsistent with the leader’s, the ` AppendEntries ` consistency check will fail in the next `AppendEntries ` RPC. After a rejection, the leader decrements `nextIndex` and retries the ` AppendEntries ` RPC. Eventually `nextIndex` will reach a point where the leader and follower logs match. When this happens, ` AppendEntries ` will succeed, which removes any conflicting entries in the follower’s log and appends entries from the leader’s log (if any). Once ` AppendEntries ` succeeds, the follower’s log is consistent with the leader’s, and it will remain that way for the rest of the term. 
>  如果某个 follower 的日志和 leader 的日志不一致，下一次 `AppendEntries` 的一致性检查将失败，leader 收到拒绝消息后，会减少该 follower 的 `nextIndex`，然后重新发送 `AppendEntries` ，最后直到 `nextIndex` 达到了 follower 的日志中和 leader 一致的那一项，此时 `AppendEntries` 将成功，并且这一成功的 `AppendEntries` 将移除 follower 日志中所有不一致的 entries，然后将 leader 日志中的 entries 添加
>  因此，一旦 `AppendEntries` 成功，leader 就可以确认 follower 的日志和它是一致的

If desired, the protocol can be optimized to reduce the number of rejected `AppendEntries` RPCs. For example, when rejecting an `AppendEntries` request, the follower can include the term of the conflicting entry and the first index it stores for that term. With this information, the leader can decrement `nextIndex` to bypass all of the conflicting entries in that term; one ` AppendEntries ` RPC will be required for each term with conflicting entries, rather than one RPC per entry. In practice, we doubt this optimization is necessary, since failures happen infrequently and it is unlikely that there will be many inconsistent entries. 
>  该协议可以进一步优化，以减少被拒绝的 `AppendEntries` 数量
>  例如，当拒绝一个 `AppendEntries` 请求时，follower 在拒绝信息中包含冲突 entry 的任期以及它的 log 中该任期的第一个 entry 的索引
>  leader 收到这些信息后，可以减少 `nextIndex` 以跳过该任期内的所有冲突 entries，每个带有冲突 entry 的任期只需要一个 `AppendEntries` RPC，而不是每个 entry 都发送一条 RPC (感觉上是不管任期内的其他 entry 有没有冲突，只要有一个 entry 存在冲突，就认为 follower 的这个任期内的所有 entries 都需要重写，这样还是会存在过度跳过的问题，但是安全性还是没问题的)
>  实践中，我们认为这种优化可能没有必要，因为故障发生的频率不高，不太可能出现大量不一致的 entries (然而在 MIT Lab 中这个优化有必要 qwq)

With this mechanism, a leader does not need to take any special actions to restore log consistency when it comes to power. It just begins normal operation, and the logs automatically converge in response to failures of the `AppendEntries` consistency check. A leader never overwrites or deletes entries in its own log (the Leader Append-Only Property in Figure 3). 
>  在该机制下，candidate 成为 leader 时，不需要执行任何特殊动作以恢复日志一致性 (照常发送 `AppendEntires` RPC，发现不一致再处理即可)
>  leader 只需要开始正常运行，日志会在 `AppendEntries` 的一致性检查失败时自动收敛
>  leader 将永远不会覆盖或者删除它自己的日志中的条目 (Leader Append-Only Property: leader 只会向其日志中附加条目，这意味着 leader 永远认为自己日志中已经写入的条目是对的，让其他 follower 和它协调)

This log replication mechanism exhibits the desirable consensus properties described in Section 2: Raft can accept, replicate, and apply new log entries as long as a majority of the servers are up; in the normal case a new entry can be replicated with a single round of RPCs to a majority of the cluster; and a single slow follower will not impact performance. 
>  该日志复制机制展示了期望的一致性属性：只要多数服务器正常运行，Raft 就能够接收、复制并应用新的 entry (故障的 followers 总会收到之后的某次 `AppendEntries` ，然后其不一致会被处理，保证多数是为了确保 leader 能够确认 entry 可以提交)
>  在正常情况下，可以通过向集群中的多数服务器发送一轮 RPC 以 replicate 新的 entry，并且单个响应缓慢的 follower 不会影响整体性能 

## 5.4 Safety 
The previous sections described how Raft elects leaders and replicates log entries. However, the mechanisms described so far are not quite sufficient to ensure that each state machine executes exactly the same commands in the same order. For example, a follower might be unavailable while the leader commits several log entries, then it could be elected leader and overwrite these entries with new ones; as a result, different state machines might execute different command sequences. 
>  之前的部分描述了 Raft 如何选举 leaders 以及复制目录 entry
>  但目前为止描述的机制还不足以确保每个状态机以相同的顺序执行完全相同的命令
>  例如，一个 follower 可能在 leader 提交多个日志 entries 时不可用，**然后它被选举为 leader，并用新的 entries 覆盖这些 entries** (leader 永远认为自己是对的)，这就导致了不同的状态机可能会执行不同的指令序列 (这些被覆盖的 entries 没有被这个新 leader 执行)

This section completes the Raft algorithm by adding a restriction on which servers may be elected leader. The restriction ensures that the leader for any given term contains all of the entries committed in previous terms (the Leader Completeness Property from Figure 3). 
>  本节通过添加关于哪些 servers 可以被选举为 leader 的限制来完成 Raft 算法
>  该限制确保任意给定任期的 leader 将包含之前所有任期提交的目录项 (即确保满足 Leader Completeness Property: 只要是 leader，其 log 中一定包含所有已经提交的 entries)

Given the election restriction, we then make the rules for commitment more precise. Finally, we present a proof sketch for the Leader Completeness Property and show how it leads to correct behavior of the replicated state machine. 

### 5.4.1 Election restriction 
In any leader-based consensus algorithm, the leader must eventually store all of the committed log entries. In some consensus algorithms, such as Viewstamped Replication [22], a leader can be elected even if it doesn’t initially contain all of the committed entries. These algorithms contain additional mechanisms to identify the missing entries and transmit them to the new leader, either during the election process or shortly afterwards. Unfortunately, this results in considerable additional mechanism and complexity. 
>  在任意基于 leader 的共识算法中，leader 必须最后存储所有已提交的 entry 
>  在一些共识算法中，例如 Viewstamped Replication, 即便 candidate 最初不包含所有的提交项也可以被选举为 leader。这些算法包含了额外的机制来识别缺失的 entry ，然后将它们传输给新 leader。这会导致更多额外的复杂性

Raft uses a simpler approach where it guarantees that all the committed entries from previous terms are present on each new leader from the moment of its election, without the need to transfer those entries to the leader. This means that log entries only flow in one direction, from leaders to followers, and leaders never overwrite existing entries in their logs. 
>  Raft 采用更简单的方法，该方法保证之前任期的所有提交的 entry 在选举开始时就存在于 leader 的日志中，不需要向 leader 传输额外 entries
>  这意味着 entries 将仅沿着一个方向流动：从 leaders 到 followers，并且 leaders 将永远不会覆盖写其日志中现存的 entries

Raft uses the voting process to prevent a candidate from winning an election unless its log contains all committed entries. 
>  Raft 使用投票过程来确保只有自身日志包含了所有已提交 entries 的 candidate 可以赢得选举

A candidate must contact a majority of the cluster in order to be elected, which means that every committed entry must be present in at least one of those servers. If the candidate’s log is at least as up-to-date as any other log in that majority (where “up-to-date” is defined precisely below), then it will hold all the committed entries.
>  candidate 要赢得选举，必须和集群中多数 servers 通信，这意味着每个已提交的 entry 必须至少存在于这些 servers 的其中一个上 (“已提交“ 的意思就是多数的 servers 的 log 中都有了这一 entry)。如果 candidate 的日志相较于多数服务器的日志**都是** up-to-date 的，那么可以确定它包含了所有已提交的 entries

>  leader 只有在成功将 entry replicate 到多数 followers 上，才会认为该 entry 已提交

>  当前 leader 的任期过后，选举新 leader 时，根据上述要求 candidate 必须收到多数 voters 的投票，则 candidate 的 logs 需要比多数的 voters 的 logs 都更新，对于每一个已经提交的 entry，在这 “多数” 的 voters 中，至少会有一个 voter 包含这个 entry，因此，candidates 在收获选票的过程中，其 log 中的 entries 一定会和所有已经提交的 entries 进行比较，如果成功，就说明 candidates 的 log 包含了所有已经提交的 entries

>  这就确保了只有自身 log 包含了所有已提交的 entries 的 candidate 能够成为 leader，进而确保了 leader 的 log 一定包含所有已提交的 entries

 The `RequestVote` RPC implements this restriction: the RPC includes information about the candidate’s log, and the voter denies its vote if its own log is more up-to-date than that of the candidate. 
>  `RequestVote` RPC 实现了这一限制：该 RPC 包含了 candidate 的日志信息，voter 收到该 RPC 后，会将自己的日志信息和 candidate 的日志信息比较，如果 voter 的日志比 candidate 的日志更新，则 voter 会拒绝投票
>  (故只有 candidate 的日志比多数服务器的日期都 up-to-date 时，candidate 才可能成为 leader，否则票数不够)

Raft determines which of two logs is more up-to-date by comparing the index and term of the last entries in the logs. If the logs have last entries with different terms, then the log with the later term is more up-to-date. If the logs end with the same term, then whichever log is longer is more up-to-date. 
>  Raft 通过比较两个日志的最后一个 entries 的索引和任期来确认那个日志更 up-to-date，如果两个日志的最后一个 entries 的任期不同，则任期更高的日志更新，如果任期相同，则更长的日志更新

>  论证更新的 log 将包含更旧的 log 中的所有已经提交的 entries

>  情况 1. log a 的最后一个 entry 和 log b 的最后一个 entry 任期相同，但 log a 比 log b 更长
>  `AppendEntries` 的一致性检查机制确保了 log 中，任期相同的 entries 中不会出现索引的跳跃，它们一定是连续的
>  因此，更长的 log a 一定包含了 log b 中的最后一个 entry，根据 Log Matching Property 的保证，此时 log b 一定是 log a 的子集

>  情况 2. log a 的最后一个 entry 的任期大于 log b 的最后一个 entry 的任期
>  假设 log a 最后一个 entry 的任期是 `T` ，那么 log a 一定是在任期 `T` 的 leader 执行 `AppendEntries` 操作时通过了一致性检查，log b 则没有通过
>  因为任期 `T` 的 leader 的 log 中一定包含了任期 `T` 之前所有已经提交的 entries, log a 通过了一致性检查，根据 Log Matching Property 的保证，它一定也包含了任期 `T` 之前所有已经提交的 entries, log b 中所包含的已经提交的 entries 是其子集，故 log a 一定包含了 log b 中可能包含的所有已经提交的 entries

>  情况 2 的论证看似存在循环论证，因为 "任期 `T` 的 leader 的 log 中一定包含了任期 `T` 之前所有已经提交的 entries" 本身利用了要证明的 Leader Completeness Property
>  要破除循环，我们需要从最开始考虑，系统最初时，所有 logs 都为空，Leader Completeness Property 已经成立，之后，按照上述算法，我们总是在选举 leader 之前就确保 Leader Completeness Property 能够保持，因此，上述算法可以视作在 Leader Completeness Property 已经成立的前提下，继续地保持它成立
>  这样，我们就通过归纳破除了循环论证

### 5.4.2 Committing entries from previous terms 
As described in Section 5.3, a leader knows that an entry from its current term is committed once that entry is stored on a majority of the servers. 
>  根据 5.3 节所述，如果一个 leader 确认某个来自于它的任期内的 entry 已经被多数服务器存储，它就认为该 entry 已经被提交 (该 entry 已经达成共识，不可能再被覆盖，之后的 leader 一定都会有这个 entry，故该 entry 最终一定会被所有 servers 执行)

If a leader crashes before committing an entry, future leaders will attempt to finish replicating the entry.  
>  如果该 leader 在提交该 entry 时崩溃 (指将该 entry 应用到其状态机)，未来的 leader 将尝试完成对该 entry 的复制 (该 leader 也保证会在未来某一时间将该 entry 应用到其状态机)

However, a leader cannot immediately conclude that an entry from a previous term is committed once it is stored on a majority of servers. Figure 8 illustrates a situation where an old log entry is stored on a majority of servers, yet can still be overwritten by a future leader.
>  但是，一个 leader 确认来自之前任期的某个 entry 已经被多数服务器存储时，它则不能认为该 entry 已被提交 (即该 entry 可能被覆盖，没有拥有该 entry 的 candidate 也可能在之后被选为 leader，故之后的 leader 可以没有该 entry)
>  Figure 8 描述了一个情况，其中旧的 entry 被多数服务器存储，但仍然被未来的 leader 覆盖写 (因为这一 entry 没有被这个新 leader 存储，而新 leader 以它的 log 为准，就会覆盖其他 followers 的 log，这一 entry 就丢失了)

>  以上两点描述了 Raft 的提交规则，即 leader 如何认定某个 entry 是被提交的
>  (这个提交认定规则不能算作因，应该算作果，即在上一节描述的 log 比较机制下，只有这个提交认定机制是符合的)
>  (感觉提交认定规则和 log 比较机制的制定也是鸡生蛋蛋生鸡的问题，不如直接就将提交认定机制认作 log 比较机制的果)

![[pics/Raft-Fig8.png]]

Raft incurs this extra complexity in the commitment rules because log entries retain their original term numbers when a leader replicates entries from previous terms. In other consensus algorithms, if a new leader re-replicates entries from prior “terms,” it must do so with its new “term number.”
>  Raft 在提交规则中引入这一额外复杂性的原因在于当 leader replicates 来自于之前任期的 entries 时，这些 entries 会保持它们原来的任期号 (而不是 leader 目前的任期号)。在其他共识算法中，如果新的 leader 需要 re-replicates 之前任期的条目时，它必须将其任期号改为新的任期号

 Raft’s approach makes it easier to reason about log entries, since they maintain the same term number over time and across logs. In addition, new leaders in Raft send fewer log entries from previous terms than in other algorithms (other algorithms must send redundant log entries to renumber them before they can be committed). 
>  Raft 的方法使得 entry 更易于理解，因为它们的任期号是永远保持的
>  此外，Raft 中，新的 leader 发送的来自之前任期的 entries 少于其他共识算法 (其他算法必须先发送冗余的 entries ，将它们重新编号，以使得它们提交)

### 5.4.3 Safety argument 
Given the complete Raft algorithm, we can now argue more precisely that the Leader Completeness Property holds (this argument is based on the safety proof; see Section 9.2). We assume that the Leader Completeness Property does not hold, then we prove a contradiction. Suppose the leader for term T ($\text{leader}_T$) commits a log entry from its term, but that log entry is not stored by the leader of some future term. 
>  阐述了完整的 Raft 算法后，我们详细论证 (在该算法下) Leader Completeness Property 将成立
>  我们使用反证法，假设 Leader Completeness Property 不成立，然后证明出矛盾
>  假设任期 T 的 leader $\text{leader}_T$ 提交了它任期内的一个 entry ，但该 entry 未被某些未来任期的 leaders 存储

![[pics/Raft-Fig9.png]]

Consider the smallest term $U>T$ whose leader ($\text{leader}_U$) does not store the entry. 

1. The committed entry must have been absent from $\text{leader}_U$’s log at the time of its election (leaders never delete or overwrite entries). 
2. $\text{leader}_T$ replicated the entry on a majority of the cluster, and $\text{leader}_U$ received votes from a majority of the cluster. Thus, at least one server (“the voter”) both accepted the entry from $\text{leader}_T$ and voted for $\text{leader}_U$, as shown in Figure 9. The voter is key to reaching a contradiction. 
3. The voter must have accepted the committed entry from $\text{leader}_T$ before voting for $\text{leader}_U$; otherwise it would have rejected the `AppendEntries` request from $\text{leader}_T$ (its current term would have been higher than T). 
4. The voter still stored the entry when it voted for $\text{leader}_U$, since every intervening leader contained the entry (by assumption), leaders never remove entries, and followers only remove entries if they conflict with the leader. 
5. The voter granted its vote to $\text{leader}_U$, so $\text{leader}_U$’s log must have been as up-to-date as the voter’s. This leads to one of two contradictions. 
6. First, if the voter and $\text{leader}_U$ shared the same last log term, then $\text{leader}_U$’s log must have been at least as long as the voter’s, so its log contained every entry in the voter’s log. This is a contradiction, since the voter contained the committed entry and $\text{leader}_U$ was assumed not to. 
7. Otherwise, $\text{leader}_U$’s last log term must have been larger than the voter’s. Moreover, it was larger than T, since the voter’s last log term was at least T (it contains the committed entry from term T). The earlier leader that created $\text{leader}_U$’s last log entry must have contained the committed entry in its log (by assumption). Then, by the Log Matching Property, $\text{leader}_U$’s log must also contain the committed entry, which is a contradiction. 
8. This completes the contradiction. Thus, the leaders of all terms greater than T must contain all entries from term T that are committed in term T. 
9. The Log Matching Property guarantees that future leaders will also contain entries that are committed indirectly, such as index 2 in Figure 8(d). 

>  考虑最小的任期 $U>T$，其 leader $\text{leader}_U$ 没有存储该 entry 
>  1. 首先可以确定，在 $\text{leader}_U$ 的选举开始时 (在它还是 candidate 时)，该 entry 就不在它的日志中 (因为 leader 从不删除或覆盖写其日志)
>  2. $\text{leader}_T$ 认为该 entry 被提交，说明它已经确认该 entry 已经被 replicated 到多数服务器中。$\text{leader}_U$ 成为了 leader，说明它从多数服务器中收到了投票。那么可以确定，至少有一个服务器 (voter)，它既拥有该 entry ，也为 $\text{leader}_U$ 投了票。(这个 voter 是触发矛盾的关键)
>  3. 该 voter 必须在为 $\text{leader}_U$ 投票之前，就接收了来自 $\text{leader}_T$ 的 entry ，否则来自 $\text{leader}_T$ 的 `AppendEntries`  将被拒绝，因为为 $\text{leader}_U$ 投票后，该 voter 的任期号将大于 $\text{leader}_T$ 的任期号 (voter 的任期号更大时，voter 会认为 $\text{leader}_T$ 是过期的 leader，来自过期任期号的请求都会被拒绝)
>  4. 该 voter 在它为 $\text{leader}_U$ 投票的时候仍然存储该 entry ，因为在假设中，任期 $T$ 到任期 $U$ 之间的 leader 都包含了该 entry ，leaders 从不移除 entry ，且 followers 仅在它们的 entry 和 leader 冲突时才移除 entry ，而该 entry 显然在之前没有和任何 leaders 冲突
>  5. 该 voter 为 $\text{leader}_U$ 投了票，说明 $\text{leader}_U$ 的日志至少要比 $\text{leader}_T$ 的日志更新。这将导致两类矛盾中的一个：
>  6. 其一，如果该 voter 和 $\text{leader}_U$ 的最后一个 entry 的任期相同，则 $\text{leader}_U$ 的日志需要比该 voter 的日志更长，故 $\text{leader}_U$ 的日志将包含该 voter 日志中的所有 entry ，和条件冲突
>  7. 其二，如果 $\text{leader}_U$ 的最后一个 entry 的任期更大，由于该 voter 的 entry 的任期将至少比 $T$ 大 (因为它包含了任期 $T$ 的 entry )，则 $\text{leader}_U$ 的最后一个 entry 的任期将至少大于 $T$。在假设中，之前创建了 $\text{leader}_U$ 的最后一个 entry 的 leader 的日志中包含了 the committed entry，根据 Log Matching Property，$\text{leader}_U$ 的最后一个 entry 之前的 entries 应该和上一个 leader 一致，因此 $\text{leader}_U$ 一定会包含 the committed entry，和条件冲突
>  8. 由此，可以证明所有任期大于 $T$ 的 leaders 必须包含任期 $T$ 内所有提交的 entries
>  9. Log Matching Property 也确保了之后的 leaders 也会包含间接提交的 entries (间接提交即该 entry 是在 leader 的之前的任期收到的，而在那个任期中，leader 尚没有确定该 entry 已经 replicate 到多数 followers 上，该 server 之后一段时间没有继续当 leader，但该 server 的 log 中的这一 entry 也没有被其他 leader 覆盖写，当该 server 再一次成为 leader 后，如果它在该任期确认到该 entry 已经 replicate 到多数 followers 上，也认定该 entry 是已提交的，即便该 entry 不是在它当期的任期提交的，但只要是在该 leader 的任期即可)

>  这样的间接形式提交也可以认定为提交的原因在于要认定这一过去的 entry 已经 replicate 到多数服务器上，leader 实际上是通过确认当前任期某个的 entry 的提交而确认了该 entry 之前的 entry 提交了的
>  故感觉间接提交也不仅限于该 leader 在之前任期创建的 entries，实际上就是 Log Matching Property 

Given the Leader Completeness Property, we can prove the State Machine Safety Property from Figure 3, which states that if a server has applied a log entry at a given index to its state machine, no other server will ever apply a different log entry for the same index. At the time a server applies a log entry to its state machine, its log must be identical to the leader’s log up through that entry and the entry must be committed. Now consider the lowest term in which any server applies a given log index; the Log Completeness Property guarantees that the leaders for all higher terms will store that same log entry, so servers that apply the index in later terms will apply the same value. Thus, the State Machine Safety Property holds. 
>  给定 Leader Completeness Property，我们可以证明 State Machine Safety Property (如果一个 server 将一个特定索引的 log entry 应用到其状态机，所有其他 servers 也将在该索引中对其状态机应用相同的 log entry)
>  首先，我们知道一个 server 对其状态机应用 log entry 时，该 entry 必须是已提交的，故其 log 之前的所有部分必须和 leader 的 log 一致
>  我们考虑任意 server 在最低的任期应用某个索引上的 log entry, 此时 Log Completeness Property 确保了所有更高任期的 leaders 将都含有这一 entry，因此在之后的任期应用该索引上的 log entry 的 servers 将应用同样的值，故 State Machine Safety Property 成立

Finally, Raft requires servers to apply entries in log index order. Combined with the State Machine Safety Property, this means that all servers will apply exactly the same set of log entries to their state machines, in the same order. 
>  最后，Raft 要求 servers 以 log 索引顺序应用 entries，结合 State Machine Safety Property，这意味着所有的 servers 最后将对其状态机以相同的顺序应用相同的一组 log entries

## 5.5 Follower and candidate crashes 
Until this point we have focused on leader failures. Follower and candidate crashes are much simpler to handle than leader crashes, and they are both handled in the same way. If a follower or candidate crashes, then future `RequestVote` and ` AppendEntries ` RPCs sent to it will fail. Raft handles these failures by retrying indefinitely; if the crashed server restarts, then the RPC will complete successfully. If a server crashes after completing an RPC but before responding, then it will receive the same RPC again after it restarts. Raft RPCs are idempotent, so this causes no harm. For example, if a follower receives an ` AppendEntries ` request that includes log entries already present in its log, it ignores those entries in the new request. 
>  我们之前关注的都是 leader 故障，实际上 follower 和 candidate 的故障的处理要比 leader 故障的处理简单
>  follower 和 candidate 的故障处理方式是相同的，如果某个 follower 或 candidate 故障，则之后发送给它的 `RequestVote` 和 `AppendEntries` RPC 将失败，Raft 的处理方式是无限次地重发
>  如果故障的 server 重启，则 RPC 会最终成功完成；如果 server 完成了 RPC，但在回应之前故障，则它会在它重启后再次收到相同的 RPC, Raft 的 RPC 是等幂的，故这不会导致问题
>  例如，follower 收到了一个 `AppendEntires` 请求时，该 entry 已经在其 log 中，它就会忽略该请求

## 5.6 Timing and availability 
One of our requirements for Raft is that safety must not depend on timing: the system must not produce incorrect results just because some event happens more quickly or slowly than expected. 
>  我们对 Raft 的一个要求是其安全性不能依赖于时间：系统不能仅仅因为某些事件发生得比预期更快或更慢就产生错误的结果

However, availability (the ability of the system to respond to clients in a timely manner) must inevitably depend on timing. For example, if message exchanges take longer than the typical time between server crashes, candidates will not stay up long enough to win an election; without a steady leader, Raft cannot make progress. 
>  但是，系统的可用性 (系统及时响应客户端的能力) 不可避免的依赖于时间
>  例如，如果 server 之间的消息交换时间超过了 server 崩溃的一般间隔时间 (server 的上一次崩溃和下一次崩溃之间的时间)，candidates 将无法维持足够长的时间来赢得选举 (进而导致没有 leader)，如果没有一个稳定的 leader, Raft 将无法 progress

Leader election is the aspect of Raft where timing is most critical. Raft will be able to elect and maintain a steady leader as long as the system satisfies the following timing requirement: 

$$
broadcastTime \ll electionTimeout \ll MTBF
$$

In this inequality $broadcastTime$ is the average time it takes a server to send RPCs in parallel to every server in the cluster and receive their responses; $electionTimeout$ is the election timeout described in Section 5.2; and $M T B F$ is the average time between failures for a single server. 

>  Raft 中对时间要求最关键的部分是 leader election
>  只要系统满足广播时间 $\ll$ 选举超时时间 $\ll$ 单台服务器两次故障之间的平均时间间隔，Raft 就能够选举并维持一个稳定的 leader
>  该时序要求中，广播时间指一台服务器向集群中所有其他服务器并行发送 RPC 并收到回复的平均时间；选举超时时间是指 follower 在这段时间内没有收到来自 leader 的 RPC 后，就会发起选举；平均无故障时间是指单台 server 两次故障之间的平均时间间隔

The broadcast time should be an order of magnitude less than the election timeout so that leaders can reliably send the heartbeat messages required to keep followers from starting elections; given the randomized approach used for election timeouts, this inequality also makes split votes unlikely. 
>  广播时间应该比 election timeout 小一个数量级，leader 发送的 heartbeat 消息才可以稳定地避免 followers 发起选举
>  并且，因为采用了随机 election timeout 机制，广播时间比 election timeout 小一个数量级时，在 leader 故障之后，最先超时的 follower 可以赶在其他 follower 超时之前完成其选举，避免了 split votes 的发生

The election timeout should be a few orders of magnitude less than MTBF so that the system makes steady progress. 
>  election timeout 应该比 MTBF 小几个数量级，以确保系统可以稳定 progress

When the leader crashes, the system will be unavailable for roughly the election timeout; we would like this to represent only a small fraction of overall time. 
>  leader 崩溃后，系统将大约有 election timeout 的不可用时间，这个时间应该仅占总时间的一小部分

The broadcast time and MTBF are properties of the underlying system, while the election timeout is something we must choose. Raft’s RPCs typically require the recipient to persist information to stable storage, so the broadcast time may range from $0.5\mathrm{ms}$ to $20\mathrm{ms}$ , depending on storage technology. As a result, the election timeout is likely to be somewhere between $10\mathrm{ms}$ and $500\mathrm{ms}$ . Typical server MTBFs are several months or more, which easily satisfies the timing requirement. 
>  广播时间和平均无故障时间是系统底层的属性，election timeout 则是我们需要选择的参数
>  Raft 的 RPC 通常要求接收方将信息持久化到稳定存储中，故广播时间可能在 0.5 ms 到 20 ms 之间，具体取决于存储技术
>  因此 election timeout 应该在 10 ms 到 500 ms 之间 (广播时间的 20 倍以上)
>  典型的平均无故障时间可以到达数个月或更长，完全满足要求

# 6 Cluster membership changes 
Up until now we have assumed that the cluster configuration (the set of servers participating in the consensus algorithm) is fixed. In practice, it will occasionally be necessary to change the configuration, for example to replace servers when they fail or to change the degree of replication. Although this can be done by taking the entire cluster off-line, updating configuration files, and then restarting the cluster, this would leave the cluster unavailable during the changeover. In addition, if there are any manual steps, they risk operator error. In order to avoid these issues, we decided to automate configuration changes and incorporate them into the Raft consensus algorithm. 
>  目前为止，我们都假设了集群配置 (参与共识算法的服务器集合) 是固定的
>  实践中，有时需要更改配置，例如在 servers 故障时将其替换，或者调整 replication 的程度
>  可以通过让整个集群下线，更新配置，然后重启集群来实现，但这段时间内整个集群都不可用，此外，涉及了任何手动步骤都可能出现操作错误
>  为此，Raft 共识算法整合了自动化配置更改

For the configuration change mechanism to be safe, there must be no point during the transition where it is possible for two leaders to be elected for the same term. 
>  为了让配置变更机制保持安全性，在配置转换过程中，需要确保不可能在同一任期内选出两个 leader (这是在配置转化过程中是有可能发生的，例如部分 servers 的下线会让某个信息更新及时的 candidate 认为它获取了多数的选票)

>  一旦在同一任期选出了两个 leader，且两个 leader 的 logs 存在不一致的话，整个集群的 servers 的 logs 将出现无法预测的不一致现象，就算两个 leader 的 logs 一开始一致，两个 leader 也不太可能总是同时接到相同的 client 请求

Unfortunately, any approach where servers switch directly from the old configuration to the new configuration is unsafe. It isn’t possible to atomically switch all of the servers at once, so the cluster can potentially split into two independent majorities during the transition (see Figure 10). 
>  不幸的是，任何 servers 直接从旧配置切换到新配置的方法都是不安全的
>  一次性原子性地切换所有 servers 是不可能的，故在配置转换过程中，集群就有可能分裂为两个独立的 majority (一个是旧配置下的 majority，一个是新配置下的 majority，这两个 majority 是存在不相交的可能的，故就可能同时选出两个独立的 leaders)

![[pics/Raft-Fig10.png]]

In order to ensure safety, configuration changes must use a two-phase approach. There are a variety of ways to implement the two phases. For example, some systems (e.g., [22]) use the first phase to disable the old configuration so it cannot process client requests; then the second phase enables the new configuration. 
>  为了确保安全 (确保在配置更改时，即便会出现两个独立的 majority，仍然只可能选出一个 leader)，配置更改必须使用两阶段方法
>  一些系统使用第一阶段禁用旧的配置，使其无法处理客户端请求，然后在第二阶段启用新的配置

In Raft the cluster first switches to a transitional configuration we call joint consensus; once the joint consensus has been committed, the system then transitions to the new configuration. The joint consensus combines both the old and new configurations: 

- Log entries are replicated to all servers in both configurations. 
- Any server from either configuration may serve as leader. 
- Agreement (for elections and entry commitment) requires separate majorities from both the old and new configurations. 

>  Raft 中，集群首先切换到我们称之为联合共识的过度配置，**当联合共识提交后** (即过渡配置本身需要达成共识)，系统可以转换到新配置
>  联合共识结合了新的配置和旧的配置：
>  - 联合共识下，entries 需要被 replicated 到新配置下和旧配置下的所有 servers
>  - 新配置下的 servers 或者旧配置下的 servers 都可以成为 leader
>  - 对于选举和 entry 提交的共识同时需要新配置和旧配置各自独立的多数都达成共识

The joint consensus allows individual servers to transition between configurations at different times without compromising safety. Furthermore, joint consensus allows the cluster to continue servicing client requests throughout the configuration change. 
>  在联合共识下，servers 可以在不同的时间内完成各自的配置转换，而不会影响安全性
>  联合共识下，集群可以在配置转换的同时继续服务客户端请求

![[pics/Raft-Fig11.png]]

Cluster configurations are stored and communicated using special entries in the replicated log; 
>  集群配置会通过存储在 replicated log 中的特殊 entries 进行传播，其本身也以特殊 entries 的形式存储

Figure 11 illustrates the configuration change process. When the leader receives a request to change the configuration from $C_{\mathrm{old}}$ to $C_{\mathrm{new}}$ , it stores the configuration for joint consensus $(C_{\mathrm{old,new}}$ in the figure) as a log entry and replicates that entry using the mechanisms described previously. Once a given server adds the new configuration entry to its log, it uses that configuration for all future decisions (a server always uses the latest configuration in its log, regardless of whether the entry is committed). This means that the leader will use the rules of $C_{\mathrm{old,new}}$ to determine when the log entry for $C_{\mathrm{old,new}}$ is committed.
>  Fig 11 中，当 leader 收到从 $C_{old}$ 切换到 $C_{new}$ 的请求后，它将联合共识的配置 $C_{old, new}$ 以 entry 形式存储，然后将其 replicate 到其他 servers 的 log 中
>  当一个 server 将该 entry 添加到其 log 时，它会基于该配置来做出所有未来的决策 (server 总是会使用其 log 中最新的 entry，无论该 entry 是否已提交)，这意味着 leader 将使用 $C_{old, new}$ 的规则来决定 entry $C_{old, new}$ 是否已提交

>  server 基于联合共识的配置所需要做出的决定包括：1. 它是 candidate 时，它需要从旧配置的 majority 和新配置的 majority 都收到 votes 才能成为 leader 2. 它是 leader 时，它需要将 log replicate 到旧配置和新配置的所有 servers 上，并决定是否 entries 已提交

>  因此，当 leader 收到配置切换请求后，它将 $C_{old, new}$ 加入其 log 后，它判断 $C_{old, new}$ 是否已提交的标准就变强了，它需要确认 $C_{old}$ 和 $C_{new}$ 的 majority 都收到该 entry，才可以认为该 entry 已提交
>  (这里，我们直接把 $C_{old}, C_{new}$ 认为它们指代了旧配置下和新配置下的集群的 servers 集合，这两个集合可以相交也可以不相交，但无论如何，一旦某个共识在 $C_{old, new}$ 下达成，就说明该共识在 $C_{old}, C_{new}$ 单独也达成了，即 $C_{old}$ 中的 majority 和 $C_{new}$ 中的 majority 都存储了达成共识的 entry)

If the leader crashes, a new leader may be chosen under either $C_{\mathrm{old}}$ or $C_{\mathrm{old,new}}$ , depending on whether the winning candidate has received $C_{\mathrm{old,new}}$ .
>  如果 leader 故障，新的 leader 可能在 $C_{old}$ 或 $C_{old, new}$ 下被选出，取决于赢得选举的 candidate 是否已经收到 $C_{old, new}$

 In any case, $C_{\mathrm{new}}$ cannot make unilateral decisions during this period. 
>  但无论何种情况下，在此期间， $C_{new}$ 无法单方面地做出决定 

>  即无法仅依赖于 $C_{new}$ 的 majority 就确认提交和执行选举，这是因为目前我们还没有将 $C_{new}$ 添加到任何一个 servers 的 log 中
>  这就避免了 $C_{old}$ 和 majority 和 $C_{new}$ 的 majority 做出不同选择的情况，此时的共识要么是在 $C_{old}$ 下达成，要么在 $C_{old}$ 下达成的同时，也在 $C_{new}$ 达成 (等价于在 $C_{old, new}$ 达成)，故尽管可能存在两个 majority，也只能达成一个共识

Once $C_{\mathrm{old,new}}$ has been committed, neither $C_{\mathrm{old}}$ nor $C_{\mathrm{new}}$ can make decisions without approval of the other, and the Leader Completeness Property ensures that only servers with the $C_{\mathrm{old,new}}$ log entry can be elected as leader.
>  当 $C_{old, new}$ 提交后，Leader Completeness Property 确保了只有包含了 $C_{old, new}$ entry 的 candidate 才可以被选为 leader，此时的决定需要同时在 $C_{old}, C_{new}$ 中达成 majority

>  此时，$C_{old}$  的 majority 中一定有某个 server 包含了 $C_{old, new}$ 这一 entry，这个 server 将阻止 $C_{old}$ 中的 majority 单独做出决定，即防止了某个仍处于 $C_{old}$ 的 candidate 仅在收获了 $C_{old}$ 的 majority 的 votes 时就成为 leader，因为这个 server 不会给这个 candidate 投票

It is now safe for the leader to create a log entry describing $C_{\mathrm{new}}$ and replicate it to the cluster. Again, this configuration will take effect on each server as soon as it is seen. When the new configuration has been committed under the rules of $C_{\mathrm{new}}$ , the old configuration is irrelevant and servers not in the new configuration can be shut down. 
>  此时，可以安全创建描述 $C_{new}$ 的 entry，然后将其 replicate 到集群中 (因为现在 $C_{old}$ 已经无法单独做出决策了)
>  $C_{new}$ 将在被添加到 server 的 log 后就生效 (leader 就不会向在 $C_{old}$ 但不在 $C_{new}$ 中的 servers 发送消息了)，当 entry $C_{new}$ 在 $C_{new}$ 配置下被提交，旧配置就无关了 (因为已经确保了 $C_{new}$ 下的 servers 最后一定都会收到 entry $C_{new}$，即新配置不会丢失了)，属于旧配置的 servers 可以被关闭

>  在此之前不能关闭的原因是 $C_{new}$ 在没有提交之前还是可能丢失的，那么系统实际上仍处于 $C_{old, new}$ 状态，如果此时关闭了 $C_{old}$ 的机器，系统将永远不可能再 progress，因为不可能再达成共识了
>  $C_{new}$ 在 $C_{new}$ 的机器中达成共识后，$C_{new}$ 的 servers 本身就可以正常 progress 下去，此时 $C_{old}$ 就不再需要了

As shown in Figure 11, there is no time when $C_{\mathrm{old}}$ and $C_{\mathrm{new}}$ can both make unilateral decisions; this guarantees safety. 
>  如 Figure 11 所示，不会存在 $C_{old}, C_{new}$ 可以同时各自做出决定的情况，这就保证了安全性

There are three more issues to address for reconfiguration. The first issue is that new servers may not initially store any log entries. If they are added to the cluster in this state, it could take quite a while for them to catch up, during which time it might not be possible to commit new log entries. In order to avoid availability gaps, Raft introduces an additional phase before the configuration change, in which the new servers join the cluster as non-voting members (the leader replicates log entries to them, but they are not considered for majorities). Once the new servers have caught up with the rest of the cluster, the reconfiguration can proceed as described above. 
>  要完成重新配置，还有三个问题需要解决：
>  第一个问题是新的 servers 可能最初不存储任何 entries，故它们被添加到集群后，可能需要很久才能追上进度，而在它们追上的这一过程中，可能无法提交新的 entry
>  为了避免这一可用性的缺口，Raft 在配置变更之前引入了额外的阶段，其中新加入的 servers 会以 non-voting member 的身份加入 (leader 会将 log entries replicate 给它们，但它们不会计入 majority)，当这些新 servers 和集群的其他节点同步完毕后，就可以按照上述方式进行配置转换

The second issue is that the cluster leader may not be part of the new configuration. In this case, the leader steps down (returns to follower state) once it has committed the $C_{\mathrm{new}}$ log entry. This means that there will be a period of time (while it is committing $C_{\mathrm{new,}}$ ) when the leader is managing a cluster that does not include itself; it replicates log entries but does not count itself in majorities.
>  第二个问题是当前的 leader 可能不属于新配置
>  这种情况下，leader 会在 $C_{new}$ entry 提交后降级为 follower 状态 (继而不再发送 RPC，过一段时间后，$C_{new}$ 中的 servers 就会自发启动选举)
>  这意味着在 $C_{new}$ 正在提交的这段时间内，leader 正在管理一个不包括它自身的集群，它会 replicate log entries，但不会将自己记作 majority

The leader transition occurs when $C_{\mathrm{new}}$ is committed because this is the first point when the new configuration can operate independently (it will always be possible to choose a leader from $C_{\mathrm{new.}}$ ). Before this point, it may be the case that only a server from $C_{\mathrm{old}}$ can be elected leader. 
>   leader 切换发生在 $C_{new}$ 提交后的原因在于 $C_{new}$ 提交后，新配置就可以独立运行 (之后，$C_{new}$ 中的 servers 总是可能自行选举出 leader)

The third issue is that removed servers (those not in $C_{\mathrm{new}})$ can disrupt the cluster. These servers will not receive heartbeats, so they will time out and start new elections. They will then send RequestVote RPCs with new term numbers, and this will cause the current leader to revert to follower state. A new leader will eventually be elected, but the removed servers will time out again and the process will repeat, resulting in poor availability. 
>  第三个问题是被移除的 servers (属于 $C_{old}$ 但不属于 $C_{new}$) 可能会干扰集群
>  这些 servers 将不会再收到 leader 发来的 heartbeats，故可能会超时并发起选举，进而发送带有新任期号的 `RequestVote` RPCs，这会导致当前 leader 切换为 follower
>  之后，新的 leader 最终会被选出，但被移除的 servers 会再次超时
>  这一过程会重复发生，导致系统可用性变差

To prevent this problem, servers disregard RequestVote RPCs when they believe a current leader exists. Specifically, if a server receives a RequestVote RPC within the minimum election timeout of hearing from a current leader, it does not update its term or grant its vote. This does not affect normal elections, where each server waits at least a minimum election timeout before starting an election. However, it helps avoid disruptions from removed servers: if a leader is able to get heartbeats to its cluster, then it will not be deposed by larger term numbers. 
>  为了解决该问题，servers 会在它们认为当前存在 leader 时忽略 `RequestVote` RPC
>  具体地说，如果一个 server 首先收到了当前 leader 的消息，然后在最小 election timeout 之内收到了一个 `RequestVote` RPC，它将不会更新其任期或者投票
>  这不会影响正常选举，因为在正常选举发起之前，每个 server 都至少要等待一个最小的 election timeout 时间
>  但它有助于避免被移除 servers 的干扰：如果 leader 能够向其集群发送 heartbeats，它将不会因为更大的任期编号被替换

>  考虑如果该 leader 确实是一个过期的 leader 的情况
>  首先，该 leader 不会影响集群，因为它的 RPCs 都会被忽略
>  如果目前存在当期的 leader，当期 leader 发送的 `AppendEntries` 可以正常让过期 leader 退回 follower
>  如果当期 leader 崩溃，不再发送其 heartbeats, election timeout 后，某个 candidate 开始发送 `RequestVote` ，因为 `RequestVote` 是在 election timeout 后才发出，故 `RequestVote` 也可以正常让过期 leader 退回 follower

>  因此，在这个机制下, removed servers 在 election timeout 之前发来的 `RequestVotes` 不会影响集群，在 election timeout 之后发来的 `RequestVotes` 也不会影响集群，因为 servers 会识别出发送方不属于集群，进而不会回应

# 7 Log compaction 
Raft’s log grows during normal operation to incorporate more client requests, but in a practical system, it cannot grow without bound. As the log grows longer, it occupies more space and takes more time to replay. This will eventually cause availability problems without some mechanism to discard obsolete information that has accumulated in the log. 
>  实践中，Raft 的 log 不可能无限制增长，log 越长，就占用越多空间，并且需要越多时间 replay，最终会导致可用性问题
>  因此需要存在丢弃 log 中过期信息的机制

Snapshotting is the simplest approach to compaction. In snapshotting, the entire current system state is written to a snapshot on stable storage, then the entire log up to that point is discarded. Snapshotting is used in Chubby and ZooKeeper, and the remainder of this section describes snapshotting in Raft. 
>  快照是最简单的压缩方法，执行快照时，整个系统状态会被写入稳定存储上的 snapshot，然后在此之前的 log entries 都可以丢弃

Incremental approaches to compaction, such as log cleaning [36] and log-structured merge trees [30, 5], are also possible. These operate on a fraction of the data at once, so they spread the load of compaction more evenly over time. They first select a region of data that has accumulated many deleted and overwritten objects, then they rewrite the live objects from that region more compactly and free the region. This requires significant additional mechanism and complexity compared to snapshotting, which simplifies the problem by always operating on the entire data set. While log cleaning would require modifications to Raft, state machines can implement LSM trees using the same interface as snapshotting. 
>  增量式的压缩方法，例如 log cleaning, log-structured merge trees 也是可行的
>  这些方法一次仅处理一部分数据，故能更均匀地将压缩工作量随时间分配
>  这些方法首先选出一个积累了大量已删除或被复写对象的区域，然后从该区域重写活跃对象，进而释放该区域
>  相较于 snapshotting，这些方法更加复杂，snapshotting 每次操作都针对整个数据集，故更加简化

![[pics/Raft-Fig12.png]]

Figure 12 shows the basic idea of snapshotting in Raft. Each server takes snapshots independently, covering just the committed entries in its log. Most of the work consists of the state machine writing its current state to the snapshot. Raft also includes a small amount of metadata in the snapshot: the last included index is the index of the last entry in the log that the snapshot replaces (the last entry the state machine had applied), and the last included term is the term of this entry. These are preserved to support the `AppendEntries` consistency check for the first log entry following the snapshot, since that entry needs a previous log index and term. 
>  Raft 中，每个 server 独立执行 snapshot 操作，snapshot 会覆盖其 log 中已经提交的 entries
>  snapshot 操作的大部分工作是状态机将其当前状态写入 snapshot
>  snapshot 中还包含少量元数据: last included index 指 snapshot 中最后一个 entry 的索引 (也是目前状态机所应用的最后一个 entry), last included term 即该 entry 的任期
>  这些元信息会用于 snapshot 后收到的第一个 `AppendEntries` 的一致性检查 (一致性检查需要前一个 entry 的 index, term)

To enable cluster membership changes (Section 6), the snapshot also includes the latest configuration in the log as of last included index. Once a server completes writing a snapshot, it may delete all log entries up through the last included index, as well as any prior snapshot. 
>  为了支持集群成员变更，snapshot 还会包括到 last included index 为止，集群的最新配置
>  当 server 完成了 snapshot 的写入后，它就可以删去 last included index 之前所有 entries 以及 snapshots

Although servers normally take snapshots independently, the leader must occasionally send snapshots to followers that lag behind. This happens when the leader has already discarded the next log entry that it needs to send to a follower. Fortunately, this situation is unlikely in normal operation: a follower that has kept up with the leader would already have this entry. However, an exceptionally slow follower or a new server joining the cluster (Section 6) would not. The way to bring such a follower up-to-date is for the leader to send it a snapshot over the network. 
>  servers 会独立执行 snapshotting，但 leader 也必须偶尔向落后的 followers 发送它的 snapshots
>  当 leader 执行了 snapshotting，丢弃了它尚未发送给某个 follower 的 entries 时，leader 就需要向 followers 发送它的 snapshot
>  在正常运行中，这种情况不太可能发生，能够跟得上 leader 的 follower 不太可能会没有 leader 因为 snapshotting 丢弃的 entries
>  但如果一个 follower 异常缓慢，或者有 follower 新加入集群，leader 需要通过发送 snapshot 让 follower 赶上进度

The leader uses a new RPC called InstallSnapshot to send snapshots to followers that are too far behind; see Figure 13. When a follower receives a snapshot with this RPC, it must decide what to do with its existing log entries. Usually the snapshot will contain new information not already in the recipient’s log. In this case, the follower discards its entire log; it is all superseded by the snapshot and may possibly have uncommitted entries that conflict with the snapshot. If instead the follower receives a snapshot that describes a prefix of its log (due to retransmission or by mistake), then log entries covered by the snapshot are deleted but entries following the snapshot are still valid and must be retained. 
>  leader 会使用 `InstallSnapshot` RPC 来向落后的 followers 发送 snapshot
>  follower 收到该 RPC 后，它根据情况，依照该 snapshot 对其 log 进行修改
>  通常，snapshot 会包含 follower 的 log 尚没有的信息，此时 follower 丢弃其整个 log (即便里面可能包含和 snapshot 冲突的未提交信息)，以 snapshot 为新起点 (无论如何，以 leader 为准)
>  如果 snapshot 仅描述了 follower log 的前缀信息 (由于重传或误传)，follower 就删除 snapshot 覆盖的 entries，但之后的 entries 仍然有效并保留

This snapshotting approach departs from Raft’s strong leader principle, since followers can take snapshots without the knowledge of the leader. However, we think this departure is justified. While having a leader helps avoid conflicting decisions in reaching consensus, consensus has already been reached when snapshotting, so no decisions conflict. Data still only flows from leaders to followers, just followers can now reorganize their data. 
>  snapshotting 方法偏离了 Raft 的强 leader 原则，因为 followers 可以在 leader 不知道的情况下 snapshotting
>  我们认为这种偏离是合理的，leader 的作用是避免在达成共识时产生冲突决策，而 followers 在执行 snapshotting 时共识是已经达成了的，因此 snapshotting 不会导致决策冲突。数据仍然只从 leader 流向 followers，只是现在 followers 可以重新组织它们的数据

We considered an alternative leader-based approach in which only the leader would create a snapshot, then it would send this snapshot to each of its followers. However, this has two disadvantages. First, sending the snapshot to each follower would waste network bandwidth and slow the snapshotting process. Each follower already has the information needed to produce its own snapshots, and it is typically much cheaper for a server to produce a snapshot from its local state than it is to send and receive one over the network. Second, the leader’s implementation would be more complex. For example, the leader would need to send snapshots to followers in parallel with replicating new log entries to them, so as not to block new client requests. 
>  我们考虑过一种基于 leader 的替代方法，其中只有 leader 能创建 snapshot，然后 leader 会将 snapshot 发送给 followers，该方法有两个缺点
>  其一，向每个 follower 发送 snapshot 将浪费网络带宽，减缓 snapshotting 过程。实际上每个 follower 已经有了生成自己的 snapshot 所需的信息，follower 从本地状态生成快照显然比 leader 通过网络发送 snapshot 以及 followers 通过网络接收快照要更加经济
>  其二，leader 的实现会更加复杂，例如，leader 需要在向 followers replicate entries 的时并行地发送 snapshots，以避免阻塞新的客户端请求

There are two more issues that impact snapshotting performance. First, servers must decide when to snapshot. If a server snapshots too often, it wastes disk bandwidth and energy; if it snapshots too infrequently, it risks exhausting its storage capacity, and it increases the time required to replay the log during restarts. One simple strategy is to take a snapshot when the log reaches a fixed size in bytes. If this size is set to be significantly larger than the expected size of a snapshot, then the disk bandwidth overhead for snapshotting will be small. 
>  还有两个问题会影响 snapshotting 性能
>  其一，server 需要决定什么时候 snapshotting，过于频繁的话会浪费磁盘带宽和能源，频率过低的话更容易耗尽存储容量，并在重启时增加了 replay 时间
>  一种简单的策略是 log 达到固定大小时就执行 snapshotting，如果将该大小设置为显著大于预期的 snapshot 大小，snapshotting 的磁盘带宽将很小

The second performance issue is that writing a snapshot can take a significant amount of time, and we do not want this to delay normal operations. The solution is to use copy-on-write techniques so that new updates can be accepted without impacting the snapshot being written. For example, state machines built with functional data structures naturally support this. Alternatively, the operating system’s copy-on-write support (e.g., fork on Linux) can be used to create an in-memory snapshot of the entire state machine (our implementation uses this approach). 
>  其二，写入 snapshot 可能需要花费很长时间，而我们不希望这影响到正常操作
>  解决方案是 copy-on-write 技术，这样可以接收新的更新而不影响正在写入的 snapshot
>  例如，基于函数式数据结构的状态机天然支持这种方式，或者可以用 OS 的 copy-on-write 功能 (Linux 中的 fork) 来创建整个状态机的 in-memory snapshot

# 8 Client interaction 
This section describes how clients interact with Raft, including how clients find the cluster leader and how Raft supports linearizable semantics [10]. These issues apply to all consensus-based systems, and Raft’s solutions are similar to other systems. 
>  本节描述 clients 如何与 Raft 交互，包括了 clients 如何找到 cluster leader 以及 Raft 如何支持 linearizable 语义
>  这些问题适用于所有基于共识的系统

Clients of Raft send all of their requests to the leader. When a client first starts up, it connects to a randomly chosen server. If the client’s first choice is not the leader, that server will reject the client’s request and supply information about the most recent leader it has heard from (`AppendEntries` requests include the network address of the leader). If the leader crashes, client requests will time out; clients then try again with randomly-chosen servers. 
>  Raft 的 clients 会将它们的全部请求发送给 leader
>  client 启动时，他会连接到一个随机选择的 server，如果该 server 不是 leader，它会拒绝 client 的请求，并提供关于 leader 的信息 (`AppendEntries` RPC 包含了 leader 的网络地址)
>  如果 leader 崩溃，client 的请求会超时，然后会重新向随机的 server 发起请求

Our goal for Raft is to implement linearizable semantics (each operation appears to execute instantaneously, exactly once, at some point between its invocation and its response). However, as described so far Raft can execute a command multiple times: for example, if the leader crashes after committing the log entry but before responding to the client, the client will retry the command with a new leader, causing it to be executed a second time. The solution is for clients to assign unique serial numbers to every command. Then, the state machine tracks the latest serial number processed for each client, along with the associated response. If it receives a command whose serial number has already been executed, it responds immediately without re-executing the request. 
>  Raft 的目标是 (面向 client) 实现线性化语义 (每个操作看起来是在其调用和响应之间的某个时刻以瞬时地被仅执行一次)
>  Raft 可能会执行一条命令多次，例如 leader 在提交 entry 之后并且在回应 client 之前崩溃，client 将让新的 leader 重试该命令，从而导致它再次被执行 (新的 leader 也提交该命令后，状态机就会执行两次该 entry)
>  解决方案是让 client 为每个命令分配唯一的序列号，状态机会跟踪为每个 client 处理的命令的最后的序列号和对应的响应，如果收到已经处理过的序列号，会立即返回响应，而不是重复执行

Read-only operations can be handled without writing anything into the log. However, with no additional measures, this would run the risk of returning stale data, since the leader responding to the request might have been superseded by a newer leader of which it is unaware. Linearizable reads must not return stale data, and Raft needs two extra precautions to guarantee this without using the log. 
>  只读操作可以在不向 log 写入任何内容的情况下处理
>  但没有额外措施时，有可能返回过期的数据，因为回应请求的 leader 可能不知道他已经被新的 leader 取代

First, a leader must have the latest information on which entries are committed. The Leader Completeness Property guarantees that a leader has all committed entries, but at the start of its term, it may not know which those are. To find out, it needs to commit an entry from its term. Raft handles this by having each leader commit a blank no-op entry into the log at the start of its term. 
>  要在不使用 log 的情况下预防这一点，Raft 采用了两个额外的预防措施
>  其一，leader 必须知道关于哪些 entries 是已经提交的最新信息，Leader Completeness Property 确保了 leader 具有全部已经提交的 entries，但在其任期开始时，它可能不知道它的 entries 中具体哪些是已经提交的
>  为此，leader 需要提交其任期内的一个 entry (这样 leader 的整个 log 就都是已提交的了)，Raft 让每个 leader 在其任期开始时提交一个空的 no-op entry
>  (第一点确保了 leader 的信息是最新的)

Second, a leader must check whether it has been deposed before processing a read-only request (its information may be stale if a more recent leader has been elected). Raft handles this by having the leader exchange heartbeat messages with a majority of the cluster before responding to read-only requests. Alternatively, the leader could rely on the heartbeat mechanism to provide a form of lease [9], but this would rely on timing for safety (it assumes bounded clock skew). 
>  其二，leader 必须在处理只读请求之前，检查它自己是否已被替代 (如果已经被替代，它的信息就可能是过时的)
>  Raft 让 leader 在回应只读请求之前，和 cluster 的 majority **交换 heartbeat 消息** (这里应该理解为交换它们收到的最新的 heartbeat 消息，看看发送者是否有更高的任期，如果是，则自己就是过期的 leader)
>  或者，leader 可以依赖于 heartbeat 机制提供某种形式的租约 (也就是 leader 维护对自己的租约，如果它自己一段时间没有收到 heartbeat 的回应，就会在租约过期后自己退回 follower)，但这将依赖于时间来确保安全性

# 9 Implementation and evaluation 
We have implemented Raft as part of a replicated state machine that stores configuration information for RAMCloud [33] and assists in failover of the RAMCloud coordinator. The Raft implementation contains roughly 2000 lines of $\mathrm{C}{+}{+}$ code, not including tests, comments, or blank lines. The source code is freely available [23]. There are also about 25 independent third-party open source implementations [34] of Raft in various stages of development, based on drafts of this paper. Also, various companies are deploying Raft-based systems [34]. 

The remainder of this section evaluates Raft using three criteria: understandability, correctness, and performance. 

## 9.1 Understandability 
To measure Raft’s understandability relative to Paxos, we conducted an experimental study using upper-level undergraduate and graduate students in an Advanced Operating Systems course at Stanford University and a Distributed Computing course at U.C. Berkeley. We recorded a video lecture of Raft and another of Paxos, and created corresponding quizzes. 

The Raft lecture covered the content of this paper except for log compaction; the Paxos lecture covered enough material to create an equivalent replicated state machine, including single-decree Paxos, multi-decree Paxos, reconfiguration, and a few optimizations needed in practice (such as leader election). The quizzes tested basic understanding of the algorithms and also required students to reason about corner cases. 

Each student watched one video, took the corresponding quiz, watched the second video, and took the second quiz. About half of the participants did the Paxos portion first and the other half did the Raft portion first in order to account for both individual differences in performance and experience gained from the first portion of the study. We compared participants’ scores on each quiz to determine whether participants showed a better understanding of Raft. 

![](https://cdn-mineru.openxlab.org.cn/extract/1cc4f6e8-ee4c-4554-b37d-72676916e07f/d49e3bfd4db8c237887d9e03e1ae73f285a3d3d8dcb892d80d456491efa3456e.jpg) 

Figure 14: A scatter plot comparing 43 participants’ performance on the Raft and Paxos quizzes. Points above the diagonal (33) represent participants who scored higher for Raft. 

We tried to make the comparison between Paxos and Raft as fair as possible. 

The experiment favored Paxos in two ways: 15 of the 43 participants reported having some prior experience with Paxos, and the Paxos video is $14\%$ longer than the Raft video. As summarized in Table 1, we have taken steps to mitigate potential sources of bias. All of our materials are available for review [28, 31]. 

On average, participants scored 4.9 points higher on the Raft quiz than on the Paxos quiz (out of a possible 60 points, the mean Raft score was 25.7 and the mean Paxos score was 20.8); Figure 14 shows their individual scores. 

A paired $t\cdot$ -test states that, with $95\%$ confidence, the true distribution of Raft scores has a mean at least 2.5 points larger than the true distribution of Paxos scores. 
>  假设检验，根据样本数据，推断真实分布存在某种性质的可能性

We also created a linear regression model that predicts a new student’s quiz scores based on three factors: which quiz they took, their degree of prior Paxos experience, and the order in which they learned the algorithms. The model predicts that the choice of quiz produces a 12.5-point difference in favor of Raft. This is significantly higher than the observed difference of 4.9 points, because many of the actual students had prior Paxos experience, which helped Paxos considerably, whereas it helped Raft slightly less. Curiously, the model also predicts scores 6.3 points lower on Raft for people that have already taken the Paxos quiz; although we don’t know why, this does appear to be statistically significant. 

![](https://cdn-mineru.openxlab.org.cn/extract/1cc4f6e8-ee4c-4554-b37d-72676916e07f/37be7e919b4969c6be4adabadbdd132e151f48ab3aa6011dfe5bf346504d72b6.jpg) 

Figure 15: Using a 5-point scale, participants were asked (left) which algorithm they felt would be easier to implement in a functioning, correct, and efficient system, and (right) which would be easier to explain to a CS graduate student. 

We also surveyed participants after their quizzes to see which algorithm they felt would be easier to implement or explain; these results are shown in Figure 15. An overwhelming majority of participants reported Raft would be easier to implement and explain (33 of 41 for each question). However, these self-reported feelings may be less reliable than participants’ quiz scores, and participants may have been biased by knowledge of our hypothesis that Raft is easier to understand. 

A detailed discussion of the Raft user study is available at [31]. 

## 9.2 Correctness 
We have developed a formal specification and a proof of safety for the consensus mechanism described in Section 5. The formal specification [31] makes the information summarized in Figure 2 completely precise using the $\mathrm{TLA}+$ specification language [17].  It is about 400 lines long and serves as the subject of the proof. It is also useful on its own for anyone implementing Raft. 
>  我们为 Section 5 描述的共识机制的安全性提供了形式化规范及其安全性证明
>  该形式化规范使用 TLA+ 规范语言，它将 Figure 2 总结的信息确切地描述，整个描述大约有 400 行，该形式化描述也是安全性证明的对象

We have mechanically proven the Log Completeness Property using the TLA proof system [7]. However, this proof relies on invariants that have not been mechanically checked (for example, we have not proven the type safety of the specification). 
>  我们已经使用 TLA 证明系统机械地证明了 Log Completeness Property

Furthermore, we have written an informal proof [31] of the State Machine Safety property which is complete (it relies on the specification alone) and relatively precise (it is about 3500 words long). 
>  此外，我们还编写了一份关于 State Machine Safety Property 的非正式证明，该证明是完整的 (仅依赖于规范)，且相对精简

## 9.3 Performance 
Raft’s performance is similar to other consensus algorithms such as Paxos. The most important case for performance is when an established leader is replicating new log entries. Raft achieves this using the minimal number of messages (a single round-trip from the leader to half the cluster). 
>  Raft 的性能和其他一致性算法类似
>  关于其性能，最重要的情况是一个已确立的 leader replicate 新的 log entries 时，Raft 通过最少的消息数量 (从 leader 到集群的一半成员的单次 round-trip) 完成这一操作

It is also possible to further improve Raft’s performance. For example, it easily supports batching and pipelining requests for higher throughput and lower latency. Various optimizations have been proposed in the literature for other algorithms; many of these could be applied to Raft, but we leave this to future work. 
>  Raft 的性能还可以进一步提高，例如，它可以轻松支持批量处理和流水线处理请求，以提高吞吐量并降低延迟

We used our Raft implementation to measure the performance of Raft’s leader election algorithm and answer two questions. First, does the election process converge quickly? Second, what is the minimum downtime that can be achieved after leader crashes? 
>  我们度量 Raft 的 leader election 算法的性能，并回答两个问题
>  首先，election 过程是否快速收敛
>  其次，在 leader 崩溃后，可以实现的最短停机时间是多少

![[pics/Raft-Fig16.png]]

To measure leader election, we repeatedly crashed the leader of a cluster of five servers and timed how long it took to detect the crash and elect a new leader (see Figure 16). To generate a worst-case scenario, the servers in each trial had different log lengths, so some candidates were not eligible to become leader. Furthermore, to encourage split votes, our test script triggered a synchronized broadcast of heartbeat RPCs from the leader before terminating its process (this approximates the behavior of the leader replicating a new log entry prior to crashing). The leader was crashed uniformly randomly within its heartbeat interval, which was half of the minimum election timeout for all tests. Thus, the smallest possible downtime was about half of the minimum election timeout. 
>  执行度量时，我们反复地让一个五 servers 集群中的 leader 崩溃，然后记录检测到崩溃并选出新 leader 的耗时
>  为了生成最坏情况下的场景，我们在每次试验中让 servers 的 logs 长度都各不相同，因此一些 candidate 是不具备成为 leader 的资格的
>  此外，为了鼓励出现 split votes，我们的测试脚本会在终止 leader 的进程之前同步地广播 heartbeat RPCs (这近似模拟了 leader 在崩溃前 replicate 新的 entry 的行为)，leader 会在其 heartbeat 间隔内的随机时间点被强制崩溃
>  heartbeat 间隔是所有测试中，最小 election timeout 的一半，因此，最短的可能停机时间为最小 election timeout 的一半左右 (leader 在接近发出下一次 heartbeat 前崩溃，系统不可用一半的 minimum election timeout 后，某个 follower 达到它的 timeout，发起选举，成为 leader，系统再次可用)

The top graph in Figure 16 shows that a small amount of randomization in the election timeout is enough to avoid split votes in elections. In the absence of randomness, leader election consistently took longer than 10 seconds in our tests due to many split votes. Adding just 5ms of randomness helps significantly, resulting in a median downtime of $287\mathrm{ms}$ . Using more randomness improves worst-case behavior: with $50\mathrm{ms}$ of randomness the worst-case completion time (over 1000 trials) was $513\mathrm{ms}$ . 
>  Figure 16 top 显示在 election timeout 中加入少量的随机性就足以避免 split votes，没有随机性时，由于多次出现 split votes, leader election 总是需要 10s 以上的时间
>  添加了 5ms 左右的随机性就显著改善了情况，其中位数的停机时间降低到 287ms
>  添加更多的随机性提高了最坏情况下的表现: 50ms 的随机性使得最坏情况下的完成时间为 513ms

The bottom graph in Figure 16 shows that downtime can be reduced by reducing the election timeout. With an election timeout of $12{-}24\mathrm{ms}$ , it takes only $35\mathrm{ms}$ on average to elect a leader (the longest trial took $152\mathrm{ms}$ ). However, lowering the timeouts beyond this point violates Raft’s timing requirement: leaders have difficulty broadcasting heartbeats before other servers start new elections. This can cause unnecessary leader changes and lower overall system availability. We recommend using a conservative election timeout such as $150{-}300\mathrm{ms}$ ; such timeouts are unlikely to cause unnecessary leader changes and will still provide good availability. 
>  Figure 16 bottom 展示了减少 election timeout (随机性仍然保留) 可以减少停机时间
>  但是，进一步减少 election timeout 会违反 Raft 的时间要求: election timeout 过短时，leader 没有充分的时间在其他 servers timeout 之前完成 heartbeat 的广播，这会导致不必要的 leader 变化，降低系统的可用性
>  建议使用保守的 election timeout 150-300ms，这将不太可能导致不必要的 leader 变更

# 10 Related work 
There have been numerous publications related to consensus algorithms, many of which fall into one of the following categories: 
- Lamport’s original description of Paxos [15], and attempts to explain it more clearly [16, 20, 21]. 
- Elaborations of Paxos, which fill in missing details and modify the algorithm to provide a better foundation for implementation [26, 39, 13]. 
- Systems that implement consensus algorithms, such as Chubby [2, 4], ZooKeeper [11, 12], and Spanner [6]. The algorithms for Chubby and Spanner have not been published in detail, though both claim to be based on Paxos. ZooKeeper’s algorithm has been published in more detail, but it is quite different from Paxos. 
- Performance optimizations that can be applied to Paxos [18, 19, 3, 25, 1, 27]. 
- Oki and Liskov’s Viewstamped Replication (VR), an alternative approach to consensus developed around the same time as Paxos. The original description [29] was intertwined with a protocol for distributed transactions, but the core consensus protocol has been separated in a recent update [22]. VR uses a leader-based approach with many similarities to Raft. 

The greatest difference between Raft and Paxos is Raft’s strong leadership: Raft uses leader election as an essential part of the consensus protocol, and it concentrates as much functionality as possible in the leader. This approach results in a simpler algorithm that is easier to understand. For example, in Paxos, leader election is orthogonal to the basic consensus protocol: it serves only as a performance optimization and is not required for achieving consensus. However, this results in additional mechanism: Paxos includes both a two-phase protocol for basic consensus and a separate mechanism for leader election. In contrast, Raft incorporates leader election directly into the consensus algorithm and uses it as the first of the two phases of consensus. This results in less mechanism than in Paxos. 
>  Raft 和 Paxos 最大的差异在于 Raft 的强领导模式
>  Raft 将 leader election 作为共识协议的核心部分，并尽可能将功能集中到 leader 上
>  Paxos 中 leader election 和基本的共识协议无关，仅作为性能优化手段，leader election 为 Paxos 引入了额外机制: Paxos 既包含了用于基本共识的两阶段协议，也包含了单独的 leader election 机制
>  Raft 直接将 leader election 整合进算法中，并将其作为两阶段共识协议的第一阶段

Like Raft, VR and ZooKeeper are leader-based and therefore share many of Raft’s advantages over Paxos. However, Raft has less mechanism that VR or ZooKeeper because it minimizes the functionality in non-leaders. For example, log entries in Raft flow in only one direction: outward from the leader in `AppendEntries` RPCs. In VR log entries flow in both directions (leaders can receive log entries during the election process); this results in additional mechanism and complexity. The published description of ZooKeeper also transfers log entries both to and from the leader, but the implementation is apparently more like Raft [35]. 
>  Raft 最小化了非 leader 的功能
>  Raft 中，log entries 仅在一个方向流动

Raft has fewer message types than any other algorithm for consensus-based log replication that we are aware of. For example, we counted the message types VR and ZooKeeper use for basic consensus and membership changes (excluding log compaction and client interaction, as these are nearly independent of the algorithms). VR and ZooKeeper each define 10 different message types, while Raft has only 4 message types (two RPC requests and their responses). Raft’s messages are a bit more dense than the other algorithms’, but they are simpler collectively. In addition, VR and ZooKeeper are described in terms of transmitting entire logs during leader changes; additional message types will be required to optimize these mechanisms so that they are practical. 
>  Raft 的消息类型更少，仅有四类 (两类 RPC 和它们的回应)

Raft’s strong leadership approach simplifies the algorithm, but it precludes some performance optimizations. For example, Egalitarian Paxos (EPaxos) can achieve higher performance under some conditions with a leaderless approach [27]. EPaxos exploits commutativity in state machine commands. Any server can commit a command with just one round of communication as long as other commands that are proposed concurrently commute with it. However, if commands that are proposed concurrently do not commute with each other, EPaxos requires an additional round of communication. Because any server may commit commands, EPaxos balances load well between servers and is able to achieve lower latency than Raft in WAN settings. However, it adds significant complexity to Paxos. 
>  Raft 的强领导模式简化了算法，但排除了一些性能优化的可能性。例如，在某些条件下，无主架构的平等 Paxos（EPaxos）可以实现更高的性能 [27]
>  EPaxos 利用了状态机命令中的可交换性。只要其他同时提出的命令与之可交换，任何服务器都可以通过一轮通信提交命令。然而，如果同时提出的命令之间不具有可交换性，EPaxos 需要额外的一轮通信。由于任何服务器都可能提交命令，EPaxos 能够在服务器之间很好地平衡负载，并且在广域网（WAN）环境中能够比 Raft 实现更低的延迟。然而，这也显著增加了 Paxos 的复杂性。

Several different approaches for cluster membership changes have been proposed or implemented in other work, including Lamport’s original proposal [15], VR [22], and SMART [24]. We chose the joint consensus approach for Raft because it leverages the rest of the consensus protocol, so that very little additional mechanism is required for membership changes. Lamport’s $\alpha$ -based approach was not an option for Raft because it assumes consensus can be reached without a leader. In comparison to VR and SMART, Raft’s reconfiguration algorithm has the advantage that membership changes can occur without limiting the processing of normal requests; in contrast, VR stops all normal processing during configuration changes, and SMART imposes an $\alpha$ -like limit on the number of outstanding requests. Raft’s approach also adds less mechanism than either VR or SMART. 
>  在其他相关工作中，已经提出了或实现了几种不同的集群成员变更方法，包括 Lamport 的原始提案[15]、VR[22]和 SMART[24]。我们选择为 Raft 采用联合共识（joint consensus）的方法，因为它利用了共识协议的其余部分，因此对于成员变更几乎不需要额外的机制。
>  基于 $\alpha$ 的 Lamport 方法对 Raft 来说不是一个选项，因为它假设可以在没有领导者的情况下达成共识。
>  与 VR 和 SMART 相比，Raft 的重新配置算法的优势在于，成员变更可以在不限制正常请求处理的情况下进行；相比之下，VR 在配置变更期间停止所有正常处理，而 SMART 对未完成请求的数量施加了类似 $\alpha$ 的限制。此外，Raft 的方法比 VR 或 SMART 添加的机制更少。

# 11 Conclusion 
Algorithms are often designed with correctness, efficiency, and/or conciseness as the primary goals. Although these are all worthy goals, we believe that understandability is just as important. None of the other goals can be achieved until developers render the algorithm into a practical implementation, which will inevitably deviate from and expand upon the published form. Unless developers have a deep understanding of the algorithm and can create intuitions about it, it will be difficult for them to retain its desirable properties in their implementation. 

In this paper we addressed the issue of distributed consensus, where a widely accepted but impenetrable algorithm, Paxos, has challenged students and developers for many years. We developed a new algorithm, Raft, which we have shown to be more understandable than Paxos. We also believe that Raft provides a better foundation for system building. Using understandability as the primary design goal changed the way we approached the design of Raft; as the design progressed we found ourselves reusing a few techniques repeatedly, such as decomposing the problem and simplifying the state space. These techniques not only improved the understandability of Raft but also made it easier to convince ourselves of its correctness. 
>  本文中，我们解决了分布式一致性问题
>  我们开发了 Raft，它为系统构建提供了良好的基础
>  我们基于可理解性作为主要的设计目标，在设计中，我们发现我们反复使用了一些技术，例如将问题分解以简化状态空间，这些技术提高了 Raft 的可理解性，同时也更方便解释其正确性
