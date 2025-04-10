# Abstract 
Chain replication is a new approach to coordinating clusters of fail-stop storage servers. The approach is intended for supporting large-scale storage services that exhibit high throughput and availability without sacrificing strong consistency guarantees. 
>  链式复制是用于协调失效停止存储服务器集群的新方法
>  该方法旨在支持大规模存储服务，在不牺牲强一致性保证的前提下提供高的吞吐和可用性

Besides outlining the chain replication protocols themselves, simulation experiments explore the performance characteristics of a prototype implementation. Throughput, availability, and several object-placement strategies (including schemes based on distributed hash table routing) are discussed. 
>  本文除了描述 chain replication protocol 本身外，还进行了模拟试验验证了该协议的原型实现的性能特征
>  本文讨论了该协议的吞吐、可用性、几种对象放置策略 (包括基于分布式哈希表路由的方案)

# 1 Introduction 
A storage system typically implements operations so that clients can store, retrieve, and/or change data. File systems and database systems are perhaps the best known examples. With a file system, operations (read and write) access a single file and are idempotent; with a database system, operations (transactions) may each access multiple objects and are serializable. 
>  存储系统通常会为客户端实现用于存储、检索、修改数据的操作，例如文件系统和数据库系统
>  文件系统提供了读写操作，读写操作用于访问单个文件，且是幂等的
>  数据库系统提供的操作称为事务，事务可以访问多个对象，且是可序列化的

This paper is concerned with storage systems that sit somewhere between file systems and database systems. In particular, we are concerned with storage systems, henceforth called storage services, that 
- store objects (of an unspecified nature), 
- support query operations to return a value derived from a single object, and 
- support update operations to atomically change the state of a single object according to some pre-programmed, possibly non-deterministic, computation involving the prior state of that object. 

>  本文关注介于文件系统和数据库系统之间的存储系统，具体地说，我们关注具有以下特点的存储系统 (称为存储服务)
>  - 存储对象 (对象的性质未指定)
>  - 支持查询操作，该操作返回从单个对象派生的值
>  - 支持更新操作，该操作原子地改变单个对象的状态，更新操作的修改会基于对象先前的状态，根据某些预先编程的，可能是非确定的计算得到

A file system write is thus a special case of our storage service update which, in turn, is a special case of a database transaction. 
>  根据以上定义，file system write 操作是 storage service update 操作的特例，而 storage service update 操作是 database transaction 的特例

Increasingly, we see on-line vendors (like Amazon.com), search engines (like Google’s and FAST’s), and a host of other information-intensive services provide value by connecting large-scale storage systems to networks. A storage service is the appropriate compromise for such applications, when a database system would be too expensive and a file system lacks rich enough semantics. 
>  我们发现越来越多的在线供应商、搜索引擎以及其他信息密集型服务将大规模的存储系统连接到网络来提供价值
>  对于这类应用而言，存储服务是合适的折中方案，因为数据库系统过于昂贵，而文件系统缺乏足够丰富的语义功能

One challenge when building a large-scale storage service is maintaining high availability and high throughput despite failures and concomitant changes to the storage service’s configuration, as faulty components are detected and replaced. 
>  构建大规模存储系统的一个挑战是在发生故障并伴随存储系统配置变化 (检查故障组件并将其替换) 的情况下，维持高可用性和高吞吐

Consistency guarantees also can be crucial. But even when they are not, the construction of an application that fronts a storage service is often simplified given strong consistency guarantees, which assert that (i) operations to query and update individual objects are executed in some sequential order and (ii) the effects of update operations are necessarily reflected in results returned by subsequent query operations. 
>  强一致性保证也至关重要，且即便强一致性保证不是必要时，提供强一致性保证也可以简化面向存储服务的应用的构建开发流程
>  强一致性保证包括
>  1. 查询和更新单个对象的操作会以某种串行顺序执行
>  2. 更新操作的效果必然反映在后续查询操作返回的结果中

Strong consistency guarantees are often thought to be in tension with achieving high throughput and high availability. So system designers, reluctant to sacrifice system throughput or availability, regularly decline to support strong consistency guarantees. The Google File System (GFS) illustrates this thinking [11]. In fact, strong consistency guarantees in a large-scale storage service are not incompatible with high throughput and availability. And the new chain replication approach to coordinating fail-stop servers, which is the subject of this paper, simultaneously supports high throughput, availability, and strong consistency. 
>  通常认为提供强一致性保证和提供高吞吐和高可用性存在冲突，故系统设计者为了避免牺牲吞吐或可用性，经常选择不支持强一致性，GFS 就是一个例子
>   事实上，在大规模存储服务中，强一致性保证并不与高吞吐和高可用性矛盾，本文讨论的用于协调 fail-stop servers 的 chain replication 方法可以同时支持高吞吐、高可用性和强一致性

We proceed as follows. The interface to a generic storage service is specified in §2. In §3, we explain how query and update operations are implemented using chain replication. Chain replication can be viewed as an instance of the primary/backup approach, so 4 compares them. Then, 5 summarizes experiments to analyze throughput and availability using our prototype implementation of chain replication and a simulated network. Some of these simulations compare chain replication with storage systems (like CFS [7] and PAST [19]) based on distributed hash table (DHT) routing; other simulations reveal surprising behaviors when a system employing chain replication recovers from server failures. Chain replication is compared in 6 to other work on scalable storage systems, trading consistency for availability, and replica placement. Concluding remarks appear in §7, followed by endnotes. 

# 2 A Storage Service Interface 
Clients of a storage service issue requests for query and update operations. While it would be possible to ensure that each request reaching the storage service is guaranteed to be performed, the end-to-end argument [20] suggests there is little point in doing so. Clients are better off if the storage service simply generates a reply for each request it receives and completes, because this allows lost requests and lost replies to be handled as well: a client re-issues a request if too much time has elapsed without receiving a reply. 
>  存储服务的客户端会发出执行 query 和 update 操作的请求
>  虽然有可能确保每个到达存储服务的请求都能得到执行，但端到端论点表明这样做并无太大意义
>  对于客户端来说，如果存储服务器的行为只是简单地为它接收到的每个请求生成回复，然后返回，则客户端可以更好地处理请求丢失和回复丢失的情况: 等待回复时间过长时，就重新发送请求

- The reply for `query(objId, opts)` is derived from the value of object `objId` ; options `opts` characterizes what parts of ` objId ` are returned. The value of ` objId ` remains unchanged. 
- The reply for `update(objId, newVal, opts)` depends on options opts and, in the general case, can be a value $V$ produced in some nondeterministic pre-programmed way involving the current value of `objId` and/or value `newVal`; $V$ then becomes the new value of `objId` . $^1$

>  - 对 `query(objID, opts)` 请求的回复基于 `objId` 对象的值构造，`opts` 指定 `objId` 哪些部分会被返回，`objId` 的值保持不变
>  - 对 `update(objID, newVal, opts)` 请求的回复通常是某种非确定的预编程的方式所计算的值 $V$，计算基于 `objId` 当前的值和/或 `newVal`，返回后，$V$ 会成为 `objId` 的新值

Query operations are idempotent, but update operations need not be. A client that re-issues a nonidempotent update request must therefore take precautions to ensure the update has not already been performed. The client might, for example, first issue a query to determine whether the current value of the object already reflects the update. 
>  query 操作是幂等的，update 操作则不一定s是
>  故重新发送非幂等的 update 请求的客户端必须执行预防措施，确保该 update 请求对应的更新尚未被执行
>  例如，客户端可以首先发送一个 query 请求以确定当前值是否已经反映了即将发送的 update 请求将达成的结果

A client request that is lost before reaching the storage service is indistinguishable to that client from one that is ignored by the storage service. This means that clients would not be exposed to a new failure mode when a storage server exhibits transient outages during which client requests are ignored. Of course, acceptable client performance likely would depend on limiting the frequency and duration of transient outages. 
>  对于客户端而言，其请求在到达存储服务之前丢失和其请求被存储服务忽视没有区别
>  因此，当存储服务出现临时中断 (在此期间客户端请求被忽略) 时，客户端不会被暴露于新的故障模式

With chain replication, the duration of each transient outage is far shorter than the time required to remove a faulty host or to add a new host. So, client request processing proceeds with minimal disruption in the face of failure, recovery, and other reconfiguration. Most other replica-management protocols either block some operations or sacrifice consistency guarantees following failures and during reconfigurations. 
>  通过 chain replication，每次临时中断的持续时间都会远远低于移除故障主机再添加新主机的所需时间
>  因此，客户端请求的处理可以在出现服务器故障、恢复和其他重配置时，以最小的干扰继续执行
>  大多数其他副本管理协议要么在发生故障或重配置时阻塞某些操作，要么牺牲一致性保证

![[pics/chain replication-Fig1.png]]

We specify the functionality of our storage service by giving the client view of an object’s state and of that object’s state transitions in response to query and update requests. Figure 1 uses pseudo-code to give such a specification for an object `objID` . 
>  我们通过描述客户端视角下，一个对象的状态和在 query, update 请求下该对象的状态转移，来描述我们存储服务的功能
>  描述见 Fig1

The figure defines the state of `objID` in terms of two variables: the sequence $^2$ $H i s t_{o b j I D}$ of updates that have been performed on `objID` and a set $P e n d i n g_{o b j I D}$ of unprocessed requests. 
>  从 Fig1 中，可以看到存储服务基于两个变量 $Hist_{objID}, Pending_{objID}$ 定义了对象 `objID` 的状态
>  $Hist_{objID}$ 表示已经对 `objID` 执行过的 update 操作序列，$Pending_{objID}$ 表示尚未对 `objID` 执行的请求集合

Then, the figure lists possible state transitions. Transition T1 asserts that an arriving client request is added to $P e n d i n g_{o b j I D}$ . That some pending requests are ignored is specified by transition T2—this transition is presumably not taken too frequently. Transition T3 gives a high-level view of request processing: the request $r$ is first removed from $P e n d i n g_{o b j I D}$ ; query then causes a suitable reply to be produced whereas update also appends $r$ (denoted by ·) to $H i s t_{o b j I D}$ . $^3$
>  Fig1 中可能的状态转换有三种
>  Transition T1 将新到达的客户端请求加入 $Pending_{objID}$
>  Transition T2 忽略特定的待处理请求——该状态转换不会频繁发生
>  Transition T3 处理请求: 请求 $r$ 首先从 $Pending_{objID}$ 中移除，如果 $r$ 是 query，则生成适当回复，如果 $r$ 是 update，生成回复的同时还会将 $r$ 追加到 $Hist_{objID}$ 中

---

$^1$ The case where $V=n e w V a l$ yields a semantics for update that is simply a file system write operation; the case where $V=F(n e w V a l,o b j I D)$ amounts to support for atomic read-modify-write operations on objects. Though powerful, this semantics falls short of supporting transactions, which would allow a request to query and/or update multiple objects indivisibly. 

>  当 $V$ = `newVal`，则 `update(objId, newVal, opts)` 的语义等价于 file system write 操作的语义，当 $V$ = `F(newVal, objId)`，则 `updaet(objId, newVal, opts)` 的语义等价于为对象执行了原子性的读-修改-写操作 (读指需要读 `objId` 的原来存储的值)，该语义无法支持事务，事务要求的语义是以不可分割的方式查询和/或更新多个对象 (该语义只能一次更改一个对象，故多个对象的更改是可分割的)

$^2$ An actual implementation would probably store the current value of the object rather than storing the sequence of updates that produces this current value. We employ a sequence of updates representation here because it simplifies the task of arguing that strong consistency guarantees hold.

>  $Hist_{objID}$ 的实际实现一般仅存储 `objID` 的当前值而不是产生当前值的一系列 update 操作
>  这里用 update 操作序列表示 $Hist_{objID}$ 是因为该表示可以简化对强一致性保证的论证

$^3$ If $H i s t_{o b j I D}$ stores the current value of `objID` rather than its entire history then “ $H i s t_{o b j I D}\cdot r^{;}$ should be interpreted to denote applying the update to the object. 

>  如果 $Hist_{objID}$ 仅存储 `objID` 当前的值，而不是其 update 操作的历史，则 $Hist_{objID}\cdot r$ 应该解释为对 `objID` 施加 update 操作

# 3 Chain Replication Protocol 
Servers are assumed to be fail-stop [21] : 
- each server halts in response to a failure rather than making erroneous state transitions, and 
- a server’s halted state can be detected by the environment. 

>  chain replication protocol 假设了服务器都遵循 fail-stop:
>  - 每个服务器在发生故障时会停止运行，而不是进行错误的状态转换
>  - 服务器的停止状态可以被环境检测到

With an object replicated on $t$ servers, as many as $t-1$ of the servers can fail without compromising the object’s availability. The object’s availability is thus increased to the probability that all servers hosting that object have failed; simulations in §5.4 explore this probability for typical storage systems. Henceforth, we assume that at most $t-1$ of the servers replicating an object fail concurrently. 
>  复制到 $t$ 台服务器上的对象可以容忍 $t-1$ 台服务器同时故障而不影响对象的可用性
>  因此，对象的丢失概率等于所有存有该对象的服务器同时故障的概率
>  从现在起，我们复制了某个对象的 $t$ 个服务器中最多只有 $t-1$ 台会同时发生故障

![[pics/chain replication-Fig2.png]]

In chain replication, the servers replicating a given object `objID` are linearly ordered to form a chain. (See Figure 2.) The first server in the chain is called the head, the last server is called the tail, and request processing is implemented by the servers roughly as follows: 

**Reply Generation.** The reply for every request is generated and sent by the tail. 

**Query Processing.** Each query request is directed to the tail of the chain and processed there atomically using the replica of `objID` stored at the tail. 

**Update Processing.** Each update request is directed to the head of the chain. The request is processed there atomically using replica of `objID` at the head, then state changes are forwarded along a reliable FIFO link to the next element of the chain (where it is handled and forwarded), and so on until the request is handled by the tail. 

>  chain replication 中，复制了给定对象 `objID` 的服务器会线性排序为一个链，链中的第一个服务器称为 head，链中的最后一个服务器称为 tail，请求会大致按照以下顺序由各个服务器处理:
>  Reply Generation - 由 tail 生成并发送对所有请求的回复
>  Query Processing - 每个 query 请求都会重定向到 tail，然后基于 tail 存储的 `objID` 对象的副本原子化地被处理
>  Update Processing - 每个 update 请求都会重定向到 head，然后基于 head 存储的 `objID` 对象的副本原子化地被处理，状态变更会沿着可靠的 FIFO 链路被发送到链中的下一个服务器，下一个服务器应用该状态变更，然后继续转发，以此类推，直到 tail 完成了状态变更，update 请求的处理才算完毕

Strong consistency thus follows because query requests and update requests are all processed serially at a single server (the tail). 
>  因为 query 请求和 update 都会在单个服务器上 (tail) 被顺序地处理，故 chain replication 确保了强一致性

Processing a query request involves only a single server, and that means query is a relatively cheap operation. But when an update request is processed, computation done at $t-1$ of the $t$ servers does not contribute to producing the reply and, arguably, is redundant. The redundant servers do increase the fault-tolerance, though. 
>  处理 query 请求仅需要一个服务器，因此 query 是相对经济的操作
>  而当一个 update 请求正在被处理时，$t$ 个服务器中的前 $t-1$ 个服务器中执行的计算并不有助于生成回复 (只有 tail 能生成回复)，因此可以说这些计算是冗余的
>  然而这些冗余的服务器确实提高了容错性

Note that some redundant computation associated with the $t-1$ servers is avoided in chain replication because the new value is computed once by the head and then forwarded down the chain, so each replica has only to perform a write. This forwarding of state changes also means update can be a non-deterministic operation—the non-deterministic choice is made once, by the head. 
>  注意，chain replication 避免了前 $t-1$ 个服务器中的冗余计算，因为更新后的值实际上仅在 head 计算一次，然后就沿着链转发，故之后的每个副本仅需要执行一次写入操作
>  因为转发的是状态变更，这也允许了 update 操作可以是非确定性的——只有 head 会执行一次非确定性的选择

## 3.1 Protocol Details 
Clients do not directly read or write variables $H i s t_{o b j I D}$ and $P e n d i n g_{o b j I D}$ of Figure 1, so we are free to implement them in any way that is convenient. When chain replication is used to implement the specification of Figure 1: 

-  $H i s t_{o b j I D}$ is defined to be $H i s t_{o b j I D}^{T}$ , the value of $H i s t_{o b j I D}$ stored by tail $T$ of the chain, and 
- $P e n d i n g_{o b j I D}$ is defined to be the set of client requests received by any server in the chain and not yet processed by the tail. 

>  客户端不会直接对变量 $Hist_{objID}, Pending_{objID}$ 进行读写，故其实现方式可以比较灵活
>  chain replication protocol 在实现 Fig1 中的规范时:
>  - $Hist_{objID}$ 被定义为 $Hist^T_{objID}$，即 $Hist_{objID}$ 的值仅由 tail $T$ 存储
>  - $Pending_{objID}$ 被定义为链中任意服务器接收到但尚未被 tail 处理的请求集合

The chain replication protocols for query processing and update processing are then shown to satisfy the specification of Figure 1 by demonstrating how each state transition made by any server in the chain is equivalent either to a no-op or to allowed transitions T1, T2, or T3. 
>  我们展示链中任意服务器进行的每次状态转换要么等价于一个空操作，要么等价于 Fig1 中的规范所允许的三种转换 T1, T2, T3 的其中之一，进而证明 chain replication protocol 的 query processing 和 update processing 过程满足 Fig1 中的规范

Given the descriptions above for how $H i s t_{o b j I D}$ and $P e n d i n g_{o b j I D}$ are implemented by a chain (and assuming for the moment that failures do not occur), we observe that the only server transitions affecting $H i s t_{o b j I D}$ and $P e n d i n g_{o b j I D}$ are: (i) a server in the chain receiving a request from a client (which affects $P e n d i n g_{o b j I D};$ ), and (ii) the tail processing a client request (which affects $H i s t_{o b j I D}$ ). Since other server transitions are equivalent to no-ops, it suffices to show that transitions (i) and (ii) are consistent with T1 through T3. 
>  我们基于上面关于 chain replication protocol 的描述，并且暂时假设不会发生故障，可以观察到能够影响状态变量 $Hist_{objID}, Pending_{objID}$ 的 server transitions 只有
>  (i) 链中的某个服务器从一个客户端接收到一个请求 (影响 $Pending_{objID}$)
>  (ii) tail 服务器处理一个客户端请求 (影响 $Hist_{objID}$)
>  故其他的 server transitions (例如其他的服务器处理客户端请求) 都等价于空操作，接下来我们只需证明 transition (i), (ii) 符合 T1 至 T3 即可

**Client Request Arrives at Chain.** Clients send requests to either the head (update) or the tail (query). Receipt of a request $r$ by either adds $r$ to the set of requests received by a server but not yet processed by the tail. Thus, receipt of $r$ by either adds $r$ to $P e n d i n g_{o b j I D}$ (as defined above for a chain), and this is consistent with T1. 
>  Client Request Arrives at Chain
>  客户端向 head 发送 update 请求，向 tail 发送 query 请求 (发送到其他服务器会被重定向)，无论是 head 还是 tail 接收到请求，都会将请求 $r$ 添加到请求集合中 (该集合包含了已经被接收到的，但尚未被 tail 处理的请求)
>  因此，请求 $r$ 的接收者都会将其添加到 $Pending_{objID}$ 中，故请求的接收和 Transition T1 一致

**Request Processed by Tail.** Execution causes the request to be removed from the set of requests received by any replica that have not yet been processed by the tail, and therefore it deletes the request from $P e n d i n g_{o b j I D}$ (as defined above for a chain)—the first step of T3. Moreover, the processing of that request by tail $T$ uses replica $H i s t_{o b j I D}^{T}$ which, as defined above, implements $H i s t_{o b j I D}$ —and this is exactly what the remaining steps of T3 specify. 
>  Request Processed by Tail
>  tail $T$ 对请求的执行会导致该请求从 $Pending_{objID}$ 中被移除，这符合 T3 的第一步，此外，tail $T$ 对请求的处理会利用其副本 $Hist^T_{objID}$，根据之前的定义，$Hist^T_{objID}$ 就是 $Hist_{objID}$ 的实现，这符合 T3 的后续步骤

## 3.2 Coping with Server Failures 
In response to detecting the failure of a server that is part of a chain (and, by the fail-stop assumption, all such failures are detected), the chain is reconfigured to eliminate the failed server. For this purpose, we employ a service, called the master, that 

- detects failures of servers, 
- informs each server in the chain of its new predecessor or new successor in the new chain obtained by deleting the failed server, 
- informs clients which server is the head and which is the tail of the chain. 

>  我们利用 master service 检测链中的服务器的故障 (根据故障停止假设，所有的故障都会被检测到)，检测到故障后，链会被重新配置，消除故障的服务器
>  master service 负责
>  - 检测服务器故障
>  - 告知链中各个服务器它们的在新配置中 (删除故障服务器) 的前驱和后继服务器
>  - 告知客户端新配置中，哪个服务器是 head，哪个是 tail

In what follows, we assume the master is a single process that never fails. This simplifies the exposition but is not a realistic assumption; our prototype implementation of chain replication actually replicates a master process on multiple hosts, using Paxos [16] to coordinate those replicas so they behave in aggregate like a single process that does not fail. 
>  我们假设 master 是一个永不故障的进程，这将简化描述
>  当然这不符合实际，在我们的原型实现中，我们将 master 进程复制到多个主机上，并且使用 Paxos 协调这些副本，从而使得它们表现得像一个永不故障的进程

The master distinguishes three cases: (i) failure of the head, (ii) failure of the tail, and (iii) failure of some other server in the chain. 
>  master 区分三种情况
>  (i) head 故障
>  (ii) tail 故障
>  (iii) 链中其他服务器故障

The handling of each, however, depends on the following insight about how updates are propagated in a chain. 

Let the server at the head of the chain be labeled $H$ , the next server be labeled $H+1$ , etc., through the tail, which is given label $T$ . Define 

$$
H i s t_{o b j I D}^{i}\preceq H i s t_{o b j I D}^{j}
$$ 
to hold if sequence $^4$ of requests  $Hist^i_{objID}$ at the server with label $i$ is a prefix of sequence $H i s t_{o b j I D}^{j}$ at the server with label $j$ . Because updates are sent between elements of a chain over reliable FIFO links, the sequence of updates received by each server is a prefix of those received by its successor. 

>  记 head 为 $H$，head 后的服务器为 $H+1$，以此类推，tail 记作 $T$
>  如果服务器 $i$ 中存储的请求序列 $Hist^i_{objID}$ 是服务器 $j$ 中存储的请求序列 $Hist^j_{objID}$ 的前缀，则称 $Hist^i_{objID} \preceq Hist^j_{objID}$
>  因为 update 请求在链中通过可靠的 FIFO 链路传递，故每个服务器收到的 update 请求序列是其后继服务器收到的 update 请求序列的前缀 (感觉应该是前驱，不是后继)

So we have: 

**Update Propagation Invariant.** For servers labeled $i$ and $j$ such that $i\leq j$ holds (i.e., $i$ is a predecessor of $j$ in the chain) then: 

$$
H i s t_{o b j I D}^{j}\preceq H i s t_{o b j I D}^{i}.
$$ 
>  Update Propagation Invariant
>  对于服务器 $i\le j$ ($i$ 是 $j$ 在链中的某个前驱)，满足 $Hist^j_{objID} \preceq Hist^i_{objID}$

**Failure of the Head.** This case is handled by the master removing $H$ from the chain and making the successor to $H$ the new head of the chain. Such a successor must exist if our assumption holds that at most $t-1$ servers are faulty. 
>  Failure of the Head
>  head 故障后，master 从链中移除 $H$，让 $H$ 的后继成为新的 head
>  因为我们假设了最多 $t-1$ 个服务器同时故障，故 $H$ 的后继一定至少存在一个没有故障的服务器可以成为新 head

Changing the chain by deleting $H$ is a transition and, as such, must be shown to be either a no-op or consistent with T1, T2, and/or T3 of Figure 1. This is easily done. Altering the set of servers in the chain could change the contents of $P e n d i n g_{o b j I D}$ —recall, $P e n d i n g_{o b j I D}$ is defined as the set of requests received by any server in the chain and not yet processed by the tail, so deleting server $H$ from the chain has the effect of removing from $P e n d i n g_{o b j I D}$ those requests received by $H$ but not yet forwarded to a successor. Removing a request from $P e n d i n g_{o b j I D}$ is consistent with transition T2, so deleting $H$ from the chain is consistent with the specification in Figure 1. 
>  通过删除 $H$ 改变链也是一个状态转换，故必须证明它要么是 no-op，要么和 T1, T2, 或 T3 一致
>  更改链中的服务器集合会更改 $Pending_{objID}$ 的内容 ($Pending_{objID}$ 被定义为被链中任意服务器接收到，且尚未被 tail 处理的请求集合)，故移除 $H$ 的影响是要从 $Pending_{objID}$ 中移除 $H$ 已经收到，但尚未转发给其后继的请求
>  从 $Pending_{objID}$ 中移除请求和 T2 一致，故从链中删除 $H$ 仍然符合 Fig1 中规范指定的状态转换语义

**Failure of the Tail.** This case is handled by removing tail $T$ from the chain and making predecessor $T^{-}$ of $T$ the new tail of the chain. As before, such a predecessor must exist given our assumption that at most $t-1$ server replicas are faulty. 
>  Failure of the Tail
>  tail 故障后，master 从链中移除 $T$，让其前驱 $T^-$ 成为新 tail
>  因为我们假设了最多 $t-1$ 个服务器同时故障，故 $T$ 的前驱一定至少存在一个没有故障的服务器可以成为新 tail

This change to the chain alters the values of both $P e n d i n g_{o b j I D}$ and $H i s t_{o b j I D}$ , but does so in a manner consistent with repeated T3 transitions: $P e n d i n g_{o b j I D}$ decreases in size because $H i s t_{o b j I D}^{T}\preceq$ $H i s t_{o b j I D}^{T^{-}}$ (due to the Update Propagation Invariant, since $T^{-}<T$ holds), so so changing the tail from $T$ to $T^-$ potentially increases the set of requests completed by the tail which, by definition, decreases the set of requests in $P e n d i n g_{o b j I D}$ . Moreover, as required by T3, those update requests completed by $T^{-}$ but not completed by $T$ do now appear in $H i s t_{o b j I D}$ because with $T^{-}$ now the tail, $H i s t_{o b j I D}$ is defined as $H i s t_{o b j I D}^{T^{-}}$ . 
>  删除 $T$ 会同时改变 $Pending_{objID}$ 和 $Hist_{objID}$ 的值，但这一转换仍符合 T3 的语义: $Pending_{objID}$ 的大小将减小，这是因为 $Hist^T_{obj} \preceq Hist^{T^-}_{objID}$，故将 tail 从 $T$ 改为 $T^-$ 可能会增大 tail 已经完成的请求集合，根据定义，这会减小 $Pending_{objID}$ 的大小；被 $T^-$ 完成的但未被 $T$ 完成的更新请求将出现在 $Hist_{objID}$ 中，因为此时 $T^-$ 为 tail，$Hist_{objID}$ 被定义为 $Hist^{T^-}_{objID}$

**Failure of Other Servers.** Failure of a server $S$ internal to the chain is handled by deleting $S$ from the chain. The master first informs $S$ ’s successor $S^{+}$ of the new chain configuration and then informs $S$ ’s predecessor $S^{-}$ . This, however, could cause the Update Propagation Invariant to be invalidated unless some means is employed to ensure update requests that $S$ received before failing will still be forwarded along the chain (since those update requests already do appear in $H i s t_{o b j I D}^{i}$ for any predecessor $i$ of $S$ ). The obvious candidate to perform this forwarding is $S^{-}$ , but some bookkeeping and coordination are now required. 
>  Failure of Other Servers
>  链内部某个服务器 $S$ 的故障后，master 从链中移除 $S$
>  master 会首先告知 $S$ 的后继 $S^+$ 新的链配置，然后告知 $S$ 的前驱 $S^-$ 新的链配置，而这可能会导致 Update Propagation Invariant 失效，除非采取某些方法确保 $S$ 在故障之前收到的 update 请求仍然可以沿着链转发 (因为这些 update 请求已经出现在了 $S$ 的任何前驱服务器 $i$ 中的 $Hist^i_{objID}$ 中)

Let $U$ be a set of requests and let $<U$ be a total ordering on requests in that set. Define a request sequence $\overline{r}$ to be consistent with $(U,\<_{U})$ if (i) all requests in $\overline{r}$ appear in $U$ and (ii) requests are arranged in $\overline{{r}}$ in ascending order according to $<U$ . Finally, for request sequences $\overline{{r}}$ and $\overline{{r^{\prime}}}$ consistent with $(U,\<_{U})$ , define $\overline{{r}}\oplus\overline{{r^{\prime}}}$ to be a sequence of all requests appearing in $\overline{r}$ or in $\overline{{r^{\prime}}}$ such that $\overline{{r}}\oplus\overline{{r^{\prime}}}$ is consistent with $(U,\<_{U})$ (and therefore requests in sequence $\overline{{r}}\oplus\overline{{r^{\prime}}}$ are ordered according to $<U$ ). 

The Update Propagation Invariant is preserved by requiring that the first thing a replica $S^{-}$ connecting to a new successor $S^{+}$ does is: send to $S^{+}$ (using the FIFO link that connects them) those requests in $H i s t_{o b j I D}^{S^{-}}$ v te hbaet emni gsehnt tn omt ahya reparcohceeds $S^{+}$ n; do nfloyr awfatredr $S^{-}$ requests that it receives subsequent to assuming its new chain position. 

Thus, the Update Propagation Invariant will be maintained if $S^{-}$ , upon receiving notification from the master that $S^{+}$ is its new successor, first forwards the sequence of requests in $S e n t_{S^{-}}$ to $S^{+}$ . Moreover, there is no need for $S^{-}$ to forward the prefix of $S e n t_{S^{-}}$ that already appears in $H i s t_{o b j I D}^{S^{+}}$ . 

The protocol whose execution is depicted in Figure 3 embodies this approach (including the optimization of not sending more of the prefix than necessary). Message 1 informs $S^{+}$ of its new role; message 2 acknowledges and informs the master what is the sequence number $_{s n}$ of the last update request $S^{+}$ has received; message 3 informs $S^{-}$ of its new role and of $_{s n}$ so $S^{-}$ can compute the suffix of $S e n t_{S^{-}}$ to send to $S^{+}$ ; and message 4 carries that suffix. 

To this end, each server $i$ maintains a list $S e n t_{i}$ of update requests that $i$ has forwarded to some successor but that might not have been processed by the tail. The rules for adding and deleting elements on this list are straightforward: Whenever server $i$ forwards an update request $r$ to its successor, server $i$ also appends $r$ to $S e n t_{i}$ . The tail sends an acknowledgement $a c k(r)$ to its predecessor when it completes the processing of update request $r$ . And upon receipt $a c k(r)$ , a server $i$ deletes $r$ from $S e n t_{i}$ and forwards $a c k(r)$ to its predecessor. 

A request received by the tail must have been received by all of its predecessors in the chain, so we can conclude: 

Inprocess Requests Invariant. If $i\leq j$ then 

$$
{\cal H}i s t_{o b j I D}^{i}={\cal H}i s t_{o b j I D}^{j}\oplus{\cal S}e n t_{i}.
$$ 
Extending a Chain. Failed servers are removed from chains. But shorter chains tolerate fewer failures, and object availability ultimately could be compromised if ever there are too many server failures. The solution is to add new servers when chains get short. Provided the rate at which servers fail is not too high and adding a new server does not take too long, then chain length can be kept close to the desired $t$ servers (so $t-1$ further failures are needed to compromise object availability). 

A new server could, in theory, be added anywhere in a chain. In practice, adding a server $T^{+}$ to the very end of a chain seems simplist. For a tail $T^{+}$ , the value of $S e n t_{T^{+}}$ is always the empty list, so initializing $S e n t_{T^{+}}$ is trivial. All that remains is to initialize local object replica $H i s t_{o b j I D}^{T^{+}}$ in a way that satisfies the Update Propagation Invariant. 

$H i s t_{o b j I D}^{T^{+}}$ plished by having the chain’s current tail $T$ forward the object replica $H i s t_{o b j I D}^{T}$ it stores to $T^{+}$ . The forwarding (which may take some time if the object is large) can be concurrent with $T$ ’s processing query requests from clients and processing updates appended to $S e n t_{T}$ . Since $H i s t_{o b j I D}^{T^{+}}\preceq H i s t_{o b j I D}^{T}$ holds throughout this forwarding, Update Propagation Invariant holds. Therefore, once 

$$
H i s t_{o b j I D}^{T}=H i s t_{o b j I D}^{T^{+}}\oplus S e n t_{T}
$$ 
holds, Inprocess Requests Invariant is established and $T^{+}$ can begin serving as the chain’s tail: 

• $T$ is notified that it no longer is the tail. $T$ is thereafter free to discard query requests it receives from clients, but a more sensible policy is for $T$ to forward such requests to new tail $T^{+}$ . • Requests in $S e n t_{T}$ are sent (in sequence) to $T^{+}$ . The master is notified that $T^{+}$ is the new tail. • Clients are notified that query requests should be directed to $T^{+}$ . 

---

$^4$ If $H i s t_{o b j I D}^{i}$ is the current state rather than a sequence of updates, then $\preceq$ is defined to be the “prior value” relation rather than the “prefix of” relation. 

>  如果 $Hist^i_{objID}$ 表示当前状态，而不是 update 操作的序列，则 $\preceq$ 定义的关系含义为 "先前值"，而不是 "前缀"

# 4 Primary/Backup Protocols 
Chain replication is a form of primary/backup approach [3], which itself is an instance of the state machine approach [22] to replica management. In the primary/backup approach, one server, designated the primary 
• imposes a sequencing on client requests (and thereby ensures strong consistency holds), 
• distributes (in sequence) to other servers, known as backups, the client requests or resulting updates, 
• awaits acknowledgements from all non-faulty backups, and 
after receiving those acknowledgements then sends a reply to the client. 
If the primary fails, one of the back-ups is promoted into that role. 
With chain replication, the primary’s role in sequencing requests is shared by two replicas. The head sequences update requests; the tail extends that sequence by interleaving query requests. This sharing of responsibility not only partitions the sequencing task but also enables lower-latency and lower-overhead processing for query requests, because only a single server (the tail) is involved in processing a query and that processing is never delayed by activity elsewhere in the chain. Compare that to the primary backup approach, where the primary, before responding to a query, must await acknowledgements from backups for prior updates. 
In both chain replication and in the primary/backup approach, update requests must be disseminated to all servers replicating an object or else the replicas will diverge. Chain replication does this dissemination serially, resulting in higher latency than the primary/backup approach where requests were distributed to backups in parallel. With parallel dissemination, the time needed to generate a reply is proportional to the maximum latency of any non-faulty backup; with serial dissemination, it is proportional to the sum of those latencies. 
Simulations reported in 5 quantify all of these performance differences, including variants of chain replication and the primary/backup approach in which query requests are sent to any server (with expectations of trading increased performance for the strong consistency guarantee). 
Simulations are not necessary for understanding the differences in how server failures are handled by the two approaches, though. The central concern here is the duration of any transient outage experienced by clients when the service reconfigures in response to a server failure; a second concern is the added latency that server failures introduce. 
The delay to detect a server failure is by far the dominant cost, and this cost is identical for both chain replication and the primary/backup approach. What follows, then, is an analysis of the recovery costs for each approach assuming that a server failure has been detected; message delays are presumed to be the dominant source of protocol latency. 
For chain replication, there are three cases to consider: failure of the head, failure of a middle server, and failure of the tail. 
• Head Failure. Query processing continues uninterrupted. Update processing is unavailable for 2 message delivery delays while the master broadcasts a message to the new head and its successor, and then it notifies all clients of the new head using a broadcast. • Middle Server Failure. Query processing continues uninterrupted. Update processing can be delayed but update requests are not lost, hence no transient outage is experienced, provided some server in a prefix of the chain that has received the request remains operating. 
Failure of a middle server can lead to a delay in processing an update request—the protocol of Figure 3 involves 4 message delivery delays. 
• Tail Failure. Query and update processing are both unavailable for 2 message delivery delays while the master sends a message to the new tail and then notifies all clients of the new tail using a broadcast. 
With the primary/backup approach, there are two cases to consider: failure of the primary and failure of a backup. Query and update requests are affected the same way for each. 
Primary Failure. A transient outage of 5 message delays is experienced, as follows. The master detects the failure and broadcasts a message to all backups, requesting the number of updates each has processed and telling them to suspend processing requests. Each backup replies to the master. The master then broadcasts the identity of the new primary to all backups. The new primary is the one having processed the largest number of updates, and it must then forward to the backups any updates that they are missing. Finally, the master broadcasts a message notifying all clients of the new primary. 
• Backup Failure. Query processing continues uninterrupted provided no update requests are in progress. If an update request is in progress then a transient outage of at most 1 message delay is experienced while the master sends a message to the primary indicating that acknowledgements will not be forthcoming from the faulty backup and requests should not subsequently be sent there. 
So the worst case outage for chain replication (tail failure) is never as long as the worst case outage for primary/backup (primary failure); and the best case for chain replication (middle server failure) is shorter than the best case outage for primary/backup (backup failure). Still, if duration of transient outage is the dominant consideration in designing a storage service then choosing between chain replication and the primary/backup approach requires information about the mix of request types and about the chances of various servers failing. 
# 5 Simulation Experiments 
To better understand throughput and availability for chain replication, we performed a series of experiments in a simulated network. These involve prototype implementations of chain replication as well as some of the alternatives. Because we are mostly interested in delays intrinsic to the processing and communications that chain replication entails, we simulated a network with infinite bandwidth but with latencies of 1 ms per message. 
## 5.1 Single Chain, No Failures 
First, we consider the simple case when there is only one chain, no failures, and replication factor $t$ is 2, 3, and 10. We compare throughput for four different replication management alternatives: 
chain: Chain replication. 
• p/b: Primary/backup. 
weak-chain: Chain replication modified so query requests go to any random server. 
• weak-p/b: Primary/backup modified so query requests go to any random server. 
Note, weak-chain and weak-p/b do not implement the strong consistency guarantees that chain and $\mathbf{p}/\mathbf{b}$ do. 
We fix the query latency at a server to be 5 ms and fix the update latency to be 50 ms. (These numbers are based on actual values for querying or updating a web search index.) We assume each update entails some initial processing involving a disk read, and that it is cheaper to forward object-differences for storage than to repeat the update processing anew at each replica; we expect that the latency for a replica to process an object-difference message would be 20 ms (corresponding to a couple of disk accesses and a modest computation). 
So, for example, if a chain comprises three servers, the total latency to perform an update is 94 ms: 1 ms for the message from the client to the head, 50 ms for an update latency at the head, 20 ms to process the object difference message at each of the two other servers, and three additional 1 ms forwarding latencies. Query latency is only 7 ms, however. 
In Figure 4 we graph total throughput as a function of the percentage of requests that are updates for $t=2$ , $t=3$ and $t=10$ . There are 25 clients, each doing a mix of requests split between queries and updates consistent with the given percentage. Each client submits one request at a time, delaying between requests only long enough to receive the response for the previous request. So the clients together can have as many as 25 concurrent requests outstanding. Throughput for weak-chain and weak- ${\bf{p}}/{\bf{b}}$ was found to be virtually identical, so Figure 4 has only a single curve—labeled weak— rather than separate curves for weak-chain and weak- ${\bf{p}}/{\bf{b}}$ . 
![](https://cdn-mineru.openxlab.org.cn/extract/release/243b957a-3b13-4a61-b10f-47098a74e44f/f4463583e52348a880ece76404a831a28274e27faea29fe10b04dacb12e1bc7b.jpg) 
Figure 4: Request throughput as a function of the percentage of updates for various replication management alternatives chain, $\mathbf{p}/\mathbf{b}$ , and weak (denoting weak-chain, and weak- ${\bf p}/{\bf b}$ ) and for replication factors $t$ . 
Observe that chain replication (chain) has equal or superior performance to primary-backup $(\mathbf{p}/\mathbf{b})$ for all percentages of updates and each replication factor investigated. This is consistent with our expectations, because the head and the tail in chain replication share a load that, with the primary/backup approach, is handled solely by the primary. 
The curves for the weak variant of chain replication are perhaps surprising, as these weak variants are seen to perform worse than chain replication (with its strong consistency) when there are more than $15\%$ update requests. Two factors are involved: 
The weak variants of chain replication and primary/backup outperform pure chain replication for query-heavy loads by distributing the query load over all servers, an advantage that increases with replication factor. • Once the percentage of update requests increases, ordinary chain replication outperforms its weak variant—since all updates are done at the head. In particular, under pure chain replication (i) queries are not delayed at the head awaiting completion of update requests (which are relatively time consuming) and (ii) there is more capacity available at the head for update request processing if query requests are not also being handled there. 
Since weak-chain and weak-p/b do not implement strong consistency guarantees, there would seem to be surprisingly few settings where these replication management schemes would be preferred. 
Finally, note that the throughput of both chain replication and primary backup is not affected by replication factor provided there are sufficient concurrent requests so that multiple requests can be pipelined. 
## 5.2 Multiple Chains, No Failures 
If each object is managed by a separate chain and objects are large, then adding a new replica could involve considerable delay because of the time required for transferring an object’s state to that new replica. If, on the other hand, objects are small, then a large storage service will involve many objects. Each processor in the system is now likely to host servers from multiple chains—the costs of multiplexing the processors and communications channels may become prohibitive. Moreover, the failure of a single processor now affects multiple chains. 
A set of objects can always be grouped into a single volume, itself something that could be considered an object for purposes of chain replication, so a designer has considerable latitude in deciding object size. 
For the next set of experiments, we assume 
• a constant number of volumes, 
a hash function maps each object to a volume, hence to a unique chain, and 
each chain comprises servers hosted by processors selected from among those implementing the storage service. 
![](https://cdn-mineru.openxlab.org.cn/extract/release/243b957a-3b13-4a61-b10f-47098a74e44f/c3b632cc373242585b1336c797570c198ee99aa1c44ef51a19cddddb346e8b42.jpg) 
Figure 5: Average request throughput per client as a function of the number of servers for various percentages of updates. 
Clients are assumed to send their requests to a dispatcher which (i) computes the hash to determine the volume, hence chain, storing the object of concern and then (ii) forwards that request to the corresponding chain. (The master sends configuration information for each volume to the dispatcher, avoiding the need for the master to communicate directly with clients. Interposing a dispatcher adds a 1ms delay to updates and queries, but doesn’t affect throughput.) The reply produced by the chain is sent directly to the client and not by way of the dispatcher. 
There are 25 clients in our experiments, each submitting queries and updates at random, uniformly distributed over the chains. The clients send requests as fast as they can, subject to the restriction that each client can have only one request outstanding at a time. 
To facilitate comparisons with the GFS experiments [11], we assume 5000 volumes each replicated three times, and we vary the number of servers. We found little or no difference among chain, $\mathbf{p}/\mathbf{b}$ , weak chain, and weak $\mathbf{p}/\mathbf{b}$ alternatives, so Figure 5 shows the average request throughput per client for one—chain replication—as a function of the number of servers, for varying percentages of update requests. 
## 5.3 Effects of Failures on Throughput 
With chain replication, each server failure causes a three-stage process to start: 
1. Some time (we conservatively assume 10 seconds in our experiments) elapses before the master detects the server failure. 
2. The offending server is then deleted from the chain. 
3. The master ultimately adds a new server to that chain and initiates a data recovery process, which takes time proportional to (i) how much data was being stored on the faulty server and (ii) the available network bandwidth. 
Delays in detecting a failure or in deleting a faulty server from a chain can increase request processing latency and can increase transient outage duration. The experiments in this section explore this. 
We assume a storage service characterized by the parameters in Table 1; these values are inspired by what is reported for GFS [11]. The assumption about network bandwidth is based on reserving for data recovery at most half the bandwidth in a 100 Mbit/second network; the time to copy the 150 Gigabytes stored on one server is now 6 hours and 40 minutes. 
In order to measure the effects of a failures on the storage service, we apply a load. The exact details of the load do not matter greatly. Our experiments use eleven clients. Each client repeatedly chooses a random object, performs an operation, and awaits a reply; a watchdog timer causes the client to start the next loop iteration if 3 seconds elapse and no reply has been received. Ten of the clients exclusively submit query operations; the eleventh client exclusively submits update operations. 
Table 1: Simulated Storage Service Characteristics. 
<html><body><table><tr><td>parameter</td><td></td></tr><tr><td>number of servers (N) 24</td><td></td></tr><tr><td>number of volumes</td><td>5000</td></tr><tr><td>chain length ()</td><td>3</td></tr><tr><td>data storedper server</td><td>150( Gigabytes</td></tr><tr><td>maximum network band- width devoted to data recovery to/from any server</td><td>6.25 Megabytes/sec</td></tr><tr><td>serverreboottime after a failure</td><td>10 minutes</td></tr></table></body></html> 
![](https://cdn-mineru.openxlab.org.cn/extract/release/243b957a-3b13-4a61-b10f-47098a74e44f/e2343b84d43d5ae066fffc86114de5efcde5c1a3048c9705460dc3ffd070ddd8.jpg) 
Figure 6: Query and update throughput with one or two failures at time 00:30. 
Each experiment described executes for 2 simulated hours. Thirty minutes into the experiment, the failure of one or two servers is simulated (as in the GFS experiments). The master detects that failure and deletes the failed server from all of the chains involving that server. For each chain that was shortened by the failure, the master then selects a new server to add. Data recovery to those servers is started. 
Figure 6(a) shows aggregate query and update throughputs as a function of time in the case a single server $F$ fails. Note the sudden drop in throughput when the simulated failure occurs 30 minutes into the experiment. The resolution of the $x$ -axis is too coarse to see that the throughput is actually zero for about 10 seconds after the failure, since the master requires a bit more than 10 seconds to detect the server failure and then delete the failed server from all chains. 
With the failed server deleted from all chains, processing now can proceed, albeit at a somewhat lower rate because fewer servers are operational (and the same request processing load must be shared among them) and because data recovery is consuming resources at various servers. Lower curves on the graph reflect this. After 10 minutes, failed server $F$ becomes operational again, and it becomes a possible target for data recovery. Every time data recovery of some volume successfully completes at $F$ , query throughput improves (as seen on the graph). 
This is because $F$ , now the tail for another chain, is handling a growing proportion of the query load. 
One might expect that after all data recovery concludes, the query throughput would be what it was at the start of the experiment. The reality is more subtle, because volumes are no longer uniformly distributed among the servers. In particular, server $F$ will now participate in fewer chains than other servers but will be the tail of every chain in which it does participate. So the load is no longer well balanced over the servers, and aggregate query throughput is lower. 
Update throughput decreases to 0 at the time of the server failure and then, once the master deletes the failed server from all chains, throughput is actually better than it was initially. This throughput improvement occurs because the server failure causes some chains to be length 2 (rather than 3), reducing the amount of work involved in performing an update. 
The GFS experiments [11] consider the case where two servers fail, too, so Figure 6(b) depicts this for our chain replication protocol. Recovery is still smooth, although it takes additional time. 
## 5.4 Large Scale Replication of Critical Data 
As the number of servers increases, so should the aggregate rate of server failures. If too many servers fail, then a volume might become unavailable. The probability of this depends on how volumes are placed on servers and, in particular, the extent to which parallelism is possible during data recovery. 
![](https://cdn-mineru.openxlab.org.cn/extract/release/243b957a-3b13-4a61-b10f-47098a74e44f/e42634b3e7ce27f2e8ba4445fe111a7d2c758c32b2d46660326c2bfd4ea91dbe.jpg) 
Figure 7: The MTBU and $99\%$ confidence intervals as a function of the number of servers and replication factor for three different placement strategies: (a) DHT-based placement with maximum possible parallel recovery; (b) random placement, but with parallel recovery limited to the same degree as is possible with DHTs; (c) random placement with maximum possible parallel recovery. 
We have investigated three volume placement strategies: 
• ring: Replicas of a volume are placed at consecutive servers on a ring, determined by a consistent hash of the volume identifier. This is the strategy used in CFS [7] and PAST [19]. The number of parallel data recoveries possible is limited by the chain length $t$ . 
• rndpar: Replicas of a volume are placed randomly on servers. This is essentially the strategy used in GFS.5 Notice that, given enough servers, there is no limit on the number of parallel data recoveries possible. 
• rndseq: Replicas of a volume are placed randomly on servers (as in rndpar), but the maximum number of parallel data recoveries is limited by $t$ (as in ring). This strategy is not used in any system known to us but is a useful benchmark for quantifying the impacts of placement and parallel recovery. 
To understand the advantages of parallel data recovery, consider a server $F$ that fails and was participating in chains $C_{1},C_{2},\ldots,C_{n}$ . For each chain $C_{i}$ , data recovery requires a source from which the volume data is fetched and a host that will become the new element of chain $C_{i}$ . Given enough processors and no constraints on the placement of volumes, it is easy to ensure that the new elements are all disjoint. And with random placement of volumes, it is likely that the sources will be disjoint as well. With disjoint sources and new elements, data recovery for chains $C_{1},C_{2},\ldots,C_{n}$ can occur in parallel. And a shorter interval for data recovery of $C_{1},C_{2},\ldots,C_{n}$ , implies that there is a shorter window of vulnerability during which a small number of concurrent failures would render some volume unavailable. 
We seek to quantify the mean time between unavailability (MTBU) of any object as a function of the number of servers and the placement strategy. Each server is assumed to exhibit exponentially distributed failures with a MTBF (Mean Time Between Failures) of 24 hours.6 As the number of servers in a storage system increases, so would the number of volumes (otherwise, why add servers). In our experiments, the number of volumes is defined to be 100 times the initial number of servers, with each server storing 100 volumes at time 0. 
We postulate that the time it takes to copy all the data from one server to another is four hours, which corresponds to copying 100 Gigabytes across a 100 Mbit/sec network restricted so that only half bandwidth can be used for data recovery. As in the GFS experiments, the maximum number of parallel data recoveries on the network is limited to $40\%$ of the servers, and the minimum transfer time is set to 10 seconds (the time it takes to copy an individual GFS object, which is 64 KBytes). 
Figure 7(a) shows that the MTBU for the ring strategy appears to have an approximately Zipfian distribution as a function of the number of servers. 
Thus, in order to maintain a particular MTBU, it is necessary to grow chain length $t$ when increasing the number of servers. From the graph, it seems as though chain length needs to be increased as the logarithm of the number of servers. 
Figure 7(b) shows the MTBU for rndseq. For $t>$ 1, rndseq has lower MTBU than ring. Compared to ring, random placement is inferior because with random placement there are more sets of $t$ servers that together store a copy of a chain, and therefore there is a higher probability of a chain getting lost due to failures. 
However, random placement makes additional opportunities for parallel recovery possible if there are enough servers. Figure 7(c) shows the MTBU for rndpar. For few servers, rndpar performs the same as rndseq, but the increasing opportunity for parallel recovery with the number of servers improves the MTBU, and eventually rndpar outperforms rndseq, and more importantly, it outperforms ring. 
# 6 Related Work 
Scalability. Chain replication is an example of what Jimenéz-Peris and Patino-Martinez [14] call a ROWAA (read one, write all available) approach. They report that ROWAA approaches provide superior scaling of availability to quorum techniques, claiming that availability of ROWAA approaches improves exponentially with the number of replicas. They also argue that non-ROWAA approaches to replication will necessarily be inferior. Because ROWAA approaches also exhibit better throughout than the best known quorum systems (except for nearly write-only applications) [14], ROWAA would seem to be the better choice for replication in most real settings. 
Many file services trade consistency for performance and scalability. Examples include Bayou [17], Ficus [13], Coda [15], and Sprite [5]. Typically, these systems allow continued operation when a network partitions by offering tools to fix inconsistencies semi-automatically. Our chain replication does not offer graceful handling of partitioned operation, trading that instead for supporting all three of: high performance, scalability, and strong consistency. 
Large-scale peer-to-peer reliable file systems are a relatively recent avenue of inquiry. OceanStore [6], FARSITE [2], and PAST [19] are examples. Of these, only OceanStore provides strong (in fact, transactional) consistency guarantees. 
Google’s File System (GFS) [11] is a large-scale cluster-based reliable file system intended for applications similar to those motivating the invention of chain replication. But in GFS, concurrent overwrites are not serialized and read operations are not synchronized with write operations. Consequently, different replicas can be left in different states, and content returned by read operations may appear to vanish spontaneously from GFS. Such weak semantics imposes a burden on programmers of applications that use GFS. 
Availability versus Consistency. Yu and Vahdat [25] explore the trade-off between consistency and availability. They argue that even in relaxed consistency models, it is important to stay as close to strong consistency as possible if availability is to be maintained in the long run. On the other hand, Gray et al. [12] argue that systems with strong consistency have unstable behavior when scaled-up, and they propose the tentative update transaction for circumventing these scalability problems. 
Amza et al. [4] present a one-copy serializable transaction protocol that is optimized for replication. As in chain replication, updates are sent to all replicas whereas queries are processed only by replicas known to store all completed updates. (In chain replication, the tail is the one replica known to store all completed updates.) The protocol of [4] performs as well as replication protocols that provide weak consistency, and it scales well in the number of replicas. No analysis is given for behavior in the face of failures. 
Replica Placement. Previous work on replica placement has focussed on achieving high throughput and/or low latency rather than on supporting high availability. Acharya and Zdonik [1] advocate locating replicas according to predictions of future accesses (basing those predictions on past accesses). In the Mariposa project [23], a set of rules allows users to specify where to create replicas, whether to move data to the query or the query to the data, where to cache data, and more. Consistency is transactional, but no consideration is given to availability. Wolfson et al. consider strategies to optimize database replica placement in order to optimize performance [24]. The OceanStore project also considers replica placement [10, 6] but from the CDN (Content Distribution Network, such as Akamai) perspective of creating as few replicas as possible while supporting certain quality of service guarantees. There is a significant body of work (e.g., [18]) concerned with placement of web page replicas as well, all from the perspective of reducing latency and network load. 
Douceur and Wattenhofer investigate how to maximize the worst-case availability of files in FARSITE [2], while spreading the storage load evenly across all servers [8, 9]. Servers are assumed to have varying availabilities. The algorithms they consider repeatedly swap files between machines if doing so improves file availability. The results are of a theoretical nature for simple scenarios; it is unclear how well these algorithms will work in a realistic storage system. 
# 7 Concluding Remarks 
Chain replication supports high throughput for query and update requests, high availability of data objects, and strong consistency guarantees. This is possible, in part, because storage services built using chain replication can and do exhibit transient outages but clients cannot distinguish such outages from lost messages. Thus, the transient outages that chain replication introduces do not expose clients to new failure modes—chain replication represents an interesting balance between what failures it hides from clients and what failures it doesn’t. 
When chain replication is employed, high availability of data objects comes from carefully selecting a strategy for placement of volume replicas on servers. Our experiments demonstrated that with DHT-based placement strategies, availability is unlikely to scale with increases in the numbers of servers; but we also demonstrated that random placement of volumes does permit availability to scale with the number of servers if this placement strategy is used in concert with parallel data recovery, as introduced for GFS. 
Our current prototype is intended primarily for use in relatively homogeneous LAN clusters. Were our prototype to be deployed in a heterogeneous wide-area setting, then uniform random placement of volume replicas would no longer make sense. Instead, replica placement would have to depend on access patterns, network proximity, and observed host reliability. Protocols to re-order the elements of a chain would likely become crucial in order to control load imbalances. 
Our prototype chain replication implementation consists of 1500 lines of Java code, plus another 2300 lines of Java code for a Paxos library. The chain replication protocols are structured as a library that makes upcalls to a storage service (or other application). The experiments in this paper assumed a “null service” on a simulated network. But the library also runs over the Java socket library, so it could be used to support a variety of storage servicelike applications. 
Acknowledgements. Thanks to our colleagues Hakon Brugard, Kjetil Jacobsen, and Knut Omang at FAST who first brought this problem to our attention. Discussion with Mark Linderman and Sarah Chung were helpful in revising an earlier version of this paper. We are also grateful for the comments of the OSDI reviewers and shepherd Margo Seltzer. A grant from the Research Council of Norway to FAST ASA is noted and acknowledged. 
Van Renesse and Schneider are supported, in part, by AFOSR grant F49620–03–1–0156 and DARPA/AFRLIFGA grant F30602–99–1–0532, although the views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of these organizations or the U.S. Government. 
# Notes 

5Actually, the placement strategy is not discussed in [11]. GFS does some load balancing that results in an approximately even load across the servers, and in our simulations we expect that random placement is a good approximation of this strategy. 
$^6$ An unrealistically short MTBF was selected here to facilitate running long-duration simulations. 
