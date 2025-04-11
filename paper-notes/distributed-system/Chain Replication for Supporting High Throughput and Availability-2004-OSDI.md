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
>  事实上，在大规模存储服务中，强一致性保证并不与高吞吐和高可用性矛盾，本文讨论的用于协调 fail-stop servers 的 chain replication 方法可以同时支持高吞吐、高可用性和强一致性

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
>  query 操作是幂等的，update 操作则不一定是
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
>  该转发工作由 $S$ 的前驱 $S^-$ 执行，且需要额外的记录和协调机制

Let $U$ be a set of requests and let $<_U$ be a total ordering on requests in that set. Define a request sequence $\overline{r}$ to be consistent with $(U,<_{U})$ if (i) all requests in $\overline{r}$ appear in $U$ and (ii) requests are arranged in $\overline{{r}}$ in ascending order according to $<_U$ . Finally, for request sequences $\overline{{r}}$ and $\overline{{r^{\prime}}}$ consistent with $(U,<_{U})$ , define $\overline{{r}}\oplus\overline{{r^{\prime}}}$ to be a sequence of all requests appearing in $\overline{r}$ or in $\overline{{r^{\prime}}}$ such that $\overline{{r}}\oplus\overline{{r^{\prime}}}$ is consistent with $(U,<_{U})$ (and therefore requests in sequence $\overline{{r}}\oplus\overline{{r^{\prime}}}$ are ordered according to $<_U$ ). 
>  令 $U$ 为一个请求集合，$<_U$ 表示该集合中请求的全序关系
>  定义一个请求序列 $\bar r$，如果 $\bar r$ 满足
>  (i) $\bar r$ 中的所有请求都出现在 $U$ 中
>  (ii) $\bar r$ 中的请求按照 $<_U$ 升序排列
>  我们就称 $\bar r$ 和 $(U, <_U)$ 一致
>  如果请求序列 $\bar r, \bar {r'}$ 都和 $(U, <_U)$ 一致，我们定义运算 $\bar r \oplus \bar {r'}$ 的结果也为一个请求序列，它包含了 $\bar r, \bar{r'}$ 中出现的所有请求，并且与 $(U, <_U)$ 一致 (即 $\bar r \oplus \bar {r'}$ 中的请求都按照 $<_U$ 升序排列)

The Update Propagation Invariant is preserved by requiring that the first thing a replica $S^{-}$ connecting to a new successor $S^{+}$ does is: send to $S^{+}$ (using the FIFO link that connects them) those requests in $H i s t_{o b j I D}^{S^{-}}$ that might have not reached $S^{+}$ ; only after those have been sent may $S^{-}$ process and forward requests that it receives subsequent to assuming its new chain position. 
>  为了保持 Update Propagation Invariant，$S^-$ 连接到它的新后继 $S^+$ 做的第一件事将其存储的 $Hist^{S^-}_{objID}$ 中那些可能尚未到达 $S^+$ 的 update 请求发送给 $S^+$ (通过二者之间的 FIFO 链路)
>  只有在这些请求都发送完之后，$S^-$ 才可以接收后续的它在新配置中接收到的后续请求

To this end, each server $i$ maintains a list $S e n t_{i}$ of update requests that $i$ has forwarded to some successor but that might not have been processed by the tail. The rules for adding and deleting elements on this list are straightforward: Whenever server $i$ forwards an update request $r$ to its successor, server $i$ also appends $r$ to $S e n t_{i}$ . The tail sends an acknowledgement $a c k(r)$ to its predecessor when it completes the processing of update request $r$ . And upon receipt $a c k(r)$ , a server $i$ deletes $r$ from $S e n t_{i}$ and forwards $a c k(r)$ to its predecessor. 
>  为此，每个服务器 $i$ 都需要维护一个列表 $Sent_i$，它包含了 $i$ 已经转发给其后继但尚未被 tail 处理的 update 请求
>  为 $Send_i$ 添加或删除元素的规则为: 当 $i$ 向其后继转发请求 $r$ 时，就将 $r$ 添加到 $Sent_i$ 中；tail 完成对 update 请求 $r$ 的处理后，向其前驱发送确认消息 $ack(r)$, $i$ 收到 $ack(r)$ 后，就从 $Sent_i$ 中删除 $r$

A request received by the tail must have been received by all of its predecessors in the chain, so we can conclude: 

**Inprocess Requests Invariant.** If $i\leq j$ then 

$$
{ H}i s t_{o b j I D}^{i}={ H}i s t_{o b j I D}^{j}\oplus{ S}e n t_{i}.
$$ 
>  被 tail 收到的 update 请求一定已经被链中所有的前驱收到过，故我们有 Inprocess Requests Invariant 成立: 
>  如果 $i \le j$，则 $Hist^i_{objID} = Hist^j_{objID} \oplus Sent_i$
>  ($Hist^i_{objID}$ 包含了 $i$ 已经处理过的 update 请求，故它们一定都发送给了后继，$Hist^j_{objID}$ 包含了 $j$ 已经处理过的 update 请求，因为 $j$ 在 $i$ 后，故 $Hist^j_{objID}$ 是 $Hist^i_{objID}$ 的前缀，即存在 $i$ 已经处理但 $j$ 尚未处理的 update 请求，而这些请求一定包含在 $Sent_i$ 中，故上述等式成立)

Thus, the Update Propagation Invariant will be maintained if $S^{-}$ , upon receiving notification from the master that $S^{+}$ is its new successor, first forwards the sequence of requests in $S e n t_{S^{-}}$ to $S^{+}$ . Moreover, there is no need for $S^{-}$ to forward the prefix of $S e n t_{S^{-}}$ that already appears in $H i s t_{o b j I D}^{S^{+}}$ . 
>  因此，当 $S^-$ 被 master 通知其新后继是 $S^+$ 时，它会先将 $Sent_{S^-}$ 中的请求 ($S^-$ 已经处理，但不确定后续节点已经处理的请求) 都发送给 $S^+$，以维护 Update Propagation Invariant
>  并且，$S^-$ 不需要将 $Sent_{S^-}$ 中已经出现在 $Hist^{S^+}_{objID}$ 中的前缀进行发送 (出现在 $Hist^{S+}_{objID}$ 说明 $S^+$ 已经处理过了)

>  上述维护 Update Propagation Invariant 的讨论的目的是防止服务器处理的请求序列出现断层
>  例如 $S^-$ 处理了 $1, 2, 3$，$S$ 处理了 $1, 2$，$S^+$ 处理了 $1$，当 $S$ 下线后，$S^-$ 不能直接将新收到的 $4$ 转发给 $S^+$，否则 $S^+$ 的处理列表将是 $1, 4$，即出现了断层/丢失，而是应该先将 $2, 3$ 转发，再继续转发新收到的 $4$

![[pics/chain replication-Fig3.png]]

The protocol whose execution is depicted in Figure 3 embodies this approach (including the optimization of not sending more of the prefix than necessary). Message 1 informs $S^{+}$ of its new role; message 2 acknowledges and informs the master what is the sequence number ${s n}$ of the last update request $S^{+}$ has received; message 3 informs $S^{-}$ of its new role and of ${s n}$ so $S^{-}$ can compute the suffix of $S e n t_{S^{-}}$ to send to $S^{+}$ ; and message 4 carries that suffix. 
>  Fig3 展示了该协议的执行过程 (包括了仅发送必要部分前缀的优化)
>  master 发送 message 1 告知 $S^+$ 其新配置，$S^+$ 回复 message 2 进行确认，并且告诉 master 其处理的最后一个 update 请求的序列号 $sn$
>  master 发送 message 3 告知 $S^-$ 其新配置以及 $sn$，$S^-$ 计算需要发送的 $Sent_{S-}$ 的后缀，通过 message 4 将其转发给 $S^+$

**Extending a Chain.** Failed servers are removed from chains. But shorter chains tolerate fewer failures, and object availability ultimately could be compromised if ever there are too many server failures. The solution is to add new servers when chains get short. Provided the rate at which servers fail is not too high and adding a new server does not take too long, then chain length can be kept close to the desired $t$ servers (so $t-1$ further failures are needed to compromise object availability). 
>  Extending a Chain
>  当链因为故障变短时，可能需要为链添加新的服务器，如果服务器故障概率不太高，且添加新服务器不需要太多时间，则最好将链的长度始终维护在 $t$ 左右 (以持续容忍 $t-1$ 次故障)

A new server could, in theory, be added anywhere in a chain. In practice, adding a server $T^{+}$ to the very end of a chain seems simplest. For a tail $T^{+}$ , the value of $S e n t_{T^{+}}$ is always the empty list, so initializing $S e n t_{T^{+}}$ is trivial. All that remains is to initialize local object replica $H i s t_{o b j I D}^{T^{+}}$ in a way that satisfies the Update Propagation Invariant. 
>  理论上新服务器可以加入到链中的任意位置
>  实际中最简单的方式是将新服务器 $T^+$ 加入到链尾
>  链尾服务器 $T^+$ 的 $Sent_{T^+}$ 总是为空列表，故将 $Sent_{T^+}$ 初始化为空即可，而 $Hist^{T+}_{objID}$ 的初始化需要确保满足 Update Propagation Invariant

The initialization of $Hist^{T^+}_{objID}$ can be accomplished by having the chain’s current tail $T$ forward the object replica $H i s t_{o b j I D}^{T}$ it stores to $T^{+}$ . The forwarding (which may take some time if the object is large) can be concurrent with $T$ ’s processing query requests from clients and processing updates appended to $S e n t_{T}$ . Since $H i s t_{o b j I D}^{T^{+}}\preceq H i s t_{o b j I D}^{T}$ holds throughout this forwarding, Update Propagation Invariant holds. 
>  可以令之前的链尾 $T$ 发送 $Hist^T_{objID}$ 给 $T^+$ 作为 $Hist^{T^+}_{objID}$ 的初始化值
>  $T$ 在处理 $Hist^T_{objID}$ 的转发操作时并发地处理 query 请求以及 update 请求 (并将其添加到 $Sent_T$)
>  因为在整个转发过程中，$Hist^{T^+}_{objID}\preceq Hist^T_{objID}$ 都保持成立，故 Update Propagation Invariant 成立

Therefore, once 

$$
H i s t_{o b j I D}^{T}=H i s t_{o b j I D}^{T^{+}}\oplus S e n t_{T}
$$ 
holds, Inprocess Requests Invariant is established and $T^{+}$ can begin serving as the chain’s tail: 

-  $T$ is notified that it no longer is the tail. $T$ is thereafter free to discard query requests it receives from clients, but a more sensible policy is for $T$ to forward such requests to new tail $T^{+}$ . 
- Requests in $S e n t_{T}$ are sent (in sequence) to $T^{+}$ . The master is notified that $T^{+}$ is the new tail. 
- Clients are notified that query requests should be directed to $T^{+}$ . 

>  当 $Hist^{T}_{objID}$ 的传输完毕后，$Hist^T_{objID} = Hist^{T^+}_{objID} \oplus Sent_T$ 将成立，即 Inprocess Requests Invariant 建立，此时 $T^+$ 的初始化完成，可以作为新 tail 进行服务
>  - $T$ 将被通知自己不再是 tail，进而之后将从客户端收到的 query 请求都转发给 $T^+$
>  - $Sent_T$ 中的请求将会被发送给 $T^+$，master 会被通知 $T^+$ 成为了新 tail
>  - 客户端会被通知 query 请求应发送给 $T^+$

---

$^4$ If $H i s t_{o b j I D}^{i}$ is the current state rather than a sequence of updates, then $\preceq$ is defined to be the “prior value” relation rather than the “prefix of” relation. 

>  如果 $Hist^i_{objID}$ 表示当前状态，而不是 update 操作的序列，则 $\preceq$ 定义的关系含义为 "先前值"，而不是 "前缀"

# 4 Primary/Backup Protocols 
Chain replication is a form of primary/backup approach [3], which itself is an instance of the state machine approach [22] to replica management. In the primary/backup approach, one server, designated the primary 

- imposes a sequencing on client requests (and thereby ensures strong consistency holds), 
- distributes (in sequence) to other servers, known as backups, the client requests or resulting updates, 
- awaits acknowledgements from all non-faulty backups, and 
- after receiving those acknowledgements then sends a reply to the client. 

>  chain replication 是 primary/backup 方法的一种形式，而 primary/backup 方法则属于用于管理副本的状态机方法
>  primary/backup 方法中，primary server 负责
>  - 对客户端请求排序 (进而确保强一致性保持)
>  - 将请求或更新结果发送给其他 backup servers
>  - 等待 backups 的确认回复
>  - 收到 backups 的确认回复后，再向客户端发送回复

If the primary fails, one of the back-ups is promoted into that role. 
With chain replication, the primary’s role in sequencing requests is shared by two replicas. The head sequences update requests; the tail extends that sequence by interleaving query requests. This sharing of responsibility not only partitions the sequencing task but also enables lower-latency and lower-overhead processing for query requests, because only a single server (the tail) is involved in processing a query and that processing is never delayed by activity elsewhere in the chain. Compare that to the primary backup approach, where the primary, before responding to a query, must await acknowledgements from backups for prior updates. 
>  如果 primary 故障，则某个 backup 会升为 primary
>  chain replication 中，primary 对客户端请求进行排序的工作划分给了 head 和 tail, head 排序 update 请求, tail 将 query 请求交织在 head 排好的 update 请求序列中，得到完整的请求序列
>  该设计在划分了任务的同时确保了 query 处理的低延迟和低开销，因为只有 tail 负责处理 query，故 query 处理永远不会被链中的其他活动延迟
>  而在 primary/backup 方法中，primary 回应 query 之前需要等待所有 backups 对之前 update 的确认回复

In both chain replication and in the primary/backup approach, update requests must be disseminated to all servers replicating an object or else the replicas will diverge. Chain replication does this dissemination serially, resulting in higher latency than the primary/backup approach where requests were distributed to backups in parallel. With parallel dissemination, the time needed to generate a reply is proportional to the maximum latency of any non-faulty backup; with serial dissemination, it is proportional to the sum of those latencies. 
>  chain replication 和 primary/backup 方法中，update 请求都必须传播到所有复制了要更新对象的服务器，否则副本将出现分歧
>  chain replication 串行地执行传播，primary/backup 方法并行地传播，故 chain replication 的延迟更高
>  在并行传播下，生成 (对客户端的) 回复所需的时间与任意非故障 backup 的最大延迟成比例，在串行传播下，则与这些延迟的总和成比例 

Simulations reported in $\S 5$ quantify all of these performance differences, including variants of chain replication and the primary/backup approach in which query requests are sent to any server (with expectations of trading increased performance for the strong consistency guarantee). 
>  $\S 5$ 中的模拟报告了上述的性能差异，包括了 chain replication 和 primary/backup 方法的变体
>  在这些变体中，query 请求可以发送给任意服务器 (预期是牺牲强一致性来换取性能提升)

Simulations are not necessary for understanding the differences in how server failures are handled by the two approaches, though. The central concern here is the duration of any transient outage experienced by clients when the service reconfigures in response to a server failure; a second concern is the added latency that server failures introduce. 
>  chain replication 和 primary/backup 在处理服务器故障时的性能也存在差异
>  这里关心的性能指标主要是当服务因为服务器故障而重新配置时，客户端经历的短暂中断的持续时间，其次是因为服务器故障引起的额外延迟

The delay to detect a server failure is by far the dominant cost, and this cost is identical for both chain replication and the primary/backup approach. What follows, then, is an analysis of the recovery costs for each approach assuming that a server failure has been detected; message delays are presumed to be the dominant source of protocol latency. 
>  检测服务器故障带来的延迟是目前为止最主要的开销，且 chain replication 和 primary/backup 中该开销是相同的
>  因此，我们将分析假设已经检测到某个服务器故障时，chain replication 和 primary/backup 的恢复成本 (完成重新配置需要的时间)
>  在分析中，我们认为消息延迟是协议延迟的主要来源

For chain replication, there are three cases to consider: failure of the head, failure of a middle server, and failure of the tail. 

- **Head Failure.** Query processing continues uninterrupted. Update processing is unavailable for 2 message delivery delays while the master broadcasts a message to the new head and its successor, and then it notifies all clients of the new head using a broadcast. 
- **Middle Server Failure.** Query processing continues uninterrupted. Update processing can be delayed but update requests are not lost, hence no transient outage is experienced, provided some server in a prefix of the chain that has received the request remains operating. Failure of a middle server can lead to a delay in processing an update request—the protocol of Figure 3 involves 4 message delivery delays. 
- **Tail Failure.** Query and update processing are both unavailable for 2 message delivery delays while the master sends a message to the new tail and then notifies all clients of the new tail using a broadcast. 

>  chain replication 中需要考虑三种情况: 
>  - Head Failure: 此时 query 处理不受干扰，继续进行，update 处理将在 master 向新 head 和其后继广播一条消息时，以及之后 master 会使用广播告知所有客户端新的 head 时不可用，故总延迟一般是两个消息传递延迟时间
>  - Middle Server Failure: 此时 query 处理不受干扰，update 处理可能会延迟，但 update 请求不会丢失，因此只要链前缀中的收到 update 的某个服务器保持运行，客户端就不会经历服务中断。update 的处理延迟一般是 Fig4 中包含的四个消息传递延迟时间
>  - Tail Failure: 在 master 向新 tail 发送消息，之后广播告知所有服务器新 tail 的身份前，query 和 update 处理都将不可用，故通常延迟两个消息传递延迟时间

With the primary/backup approach, there are two cases to consider: failure of the primary and failure of a backup. Query and update requests are affected the same way for each. 

- **Primary Failure.** A transient outage of 5 message delays is experienced, as follows. The master detects the failure and broadcasts a message to all backups, requesting the number of updates each has processed and telling them to suspend processing requests. Each backup replies to the master. The master then broadcasts the identity of the new primary to all backups. The new primary is the one having processed the largest number of updates, and it must then forward to the backups any updates that they are missing. Finally, the master broadcasts a message notifying all clients of the new primary. 
- **Backup Failure.** Query processing continues uninterrupted provided no update requests are in progress. If an update request is in progress then a transient outage of at most 1 message delay is experienced while the master sends a message to the primary indicating that acknowledgements will not be forthcoming from the faulty backup and requests should not subsequently be sent there. 

>  primary/backup 需要考虑两种情况，每种情况下，query 和 update 都受到相同的影响
>  - Primary Failure: 客户端将经历大约 5 个消息延迟的服务中断，包括了 1. master 检测到故障并向所有 backups 广播一条消息，询问每个 backup 已经处理的 update 请求的数量，并告诉它们暂停 update 的处理 2. 每个 backup 回复 master 3. master 将新 primary 的身份广播给所有 backups，新 primary 即目前处理了最多 update 的 backup，且新 primary 会将其他 backup 缺失的 update 发送给它们 4. master 广播一条消息，告知所有客户端新 primary 的身份
>  - Backup Failure: 如果故障时没有正在处理的 update 请求，则 query 请求可以继续处理，如果故障时有正在处理的 update 请求，则最多会经历 1 个消息延迟的服务中断，期间 master 向 primary 发送一条消息，告知它将不会从故障的 backup 处受到确认，并且随后不需要将 update 请求发送到该故障的 backup 处

So the worst case outage for chain replication (tail failure) is never as long as the worst case outage for primary/backup (primary failure); and the best case for chain replication (middle server failure) is shorter than the best case outage for primary/backup (backup failure). 
>  因此，chain replication 的最坏情况服务中断时间 (tail 故障时) 比 primary/backup 的最坏情况服务中断时间 (primary 故障时) 短，且 chain replication 的最好情况服务中断时间 (middle server 故障时) 比 primary/backup 的最好情况服务中断时间 (backup 故障时) 短

Still, if duration of transient outage is the dominant consideration in designing a storage service then choosing between chain replication and the primary/backup approach requires information about the mix of request types and about the chances of various servers failing. 

# 5 Simulation Experiments 
To better understand throughput and availability for chain replication, we performed a series of experiments in a simulated network. These involve prototype implementations of chain replication as well as some of the alternatives. Because we are mostly interested in delays intrinsic to the processing and communications that chain replication entails, we simulated a network with infinite bandwidth but with latencies of 1 ms per message. 
>  我们在模拟的网络中执行一系列试验
>  因为我们主要关注 chain replication 涉及的处理和通信的固有延迟，故我们模拟了一个无限带宽的网络，其中每条信息延迟为 1ms

## 5.1 Single Chain, No Failures 
First, we consider the simple case when there is only one chain, no failures, and replication factor $t$ is 2, 3, and 10. 
>  我们先考虑最简单的情况，只有一条链，没有故障，replication factor $t$ 分别为 2, 3, 10

We compare throughput for four different replication management alternatives: 

- **chain**: Chain replication. 
- **p/b**: Primary/backup. 
- **weak-chain**: Chain replication modified so query requests go to any random server. 
- **weak-p/b**: Primary/backup modified so query requests go to any random server. 

>  我们比较四种副本管理方法的吞吐:
>  - chain: chain replication
>  - p/b: primary/backup
>  - weak-chain: chain replication, 但 query 请求可以发送给任意服务器
>  - weak-p/b: primary/backup，但 query 请求可以发送给任意服务器

Note, **weak-chain** and **weak-p/b** do not implement the strong consistency guarantees that **chain** and **p/b** do. 
>  注意，weak-chain, weak-p/b 并不实现 chain 和 p/b 所具有的强一致性保证

We fix the query latency at a server to be 5 ms and fix the update latency to be 50 ms. (These numbers are based on actual values for querying or updating a web search index.) 
>  我们将服务器的 query 延迟固定在 5ms，将 update 延迟固定在 50ms

We assume each update entails some initial processing involving a disk read, and that it is cheaper to forward object-differences for storage than to repeat the update processing anew at each replica; we expect that the latency for a replica to process an object-difference message would be 20 ms (corresponding to a couple of disk accesses and a modest computation). 
>  我们假设每次 update 涉及一些初始处理，包括磁盘读取操作，并且转发对象差异比直接转发 update 请求，由各个副本本地处理更新要更经济
>  我们预计副本处理对象差异消息的时间延迟为 20ms (对应于几次磁盘访问和少量计算)

So, for example, if a chain comprises three servers, the total latency to perform an update is 94 ms: 1 ms for the message from the client to the head, 50 ms for an update latency at the head, 20 ms to process the object difference message at each of the two other servers, and three additional 1 ms forwarding latencies. 
>  因此，如果一个链包括三个服务器，处理一次 update 的总延迟为 94ms: 
>  消息从客户端到 head - 1ms
>  head 处理 update - 50ms
>  另外两个服务器处理对象差异消息 - 20+20=40ms
>  三次转发延迟 - 3ms

Query latency is only 7 ms, however. 
>  query 延迟仅有 7ms:
>  客户端到 tail - 1ms
>  query 处理 - 5ms
>  tail 到客户端 - 1ms

![[pics/chain replication-Fig4.png]]

In Figure 4 we graph total throughput as a function of the percentage of requests that are updates for $t=2$ , $t=3$ and $t=10$ . There are 25 clients, each doing a mix of requests split between queries and updates consistent with the given percentage. Each client submits one request at a time, delaying between requests only long enough to receive the response for the previous request. So the clients together can have as many as 25 concurrent requests outstanding. 
>  Fig4 中绘制了总吞吐量关于 update 请求所占百分比的函数
>  一共有 25 个客户端，每个客户端混合执行 query 和 update 请求，每个客户端每次仅提交一次请求，请求之间留出足够的时间来接收上一个请求的响应
>  因此所有客户端最多并发提交 25 个请求

Throughput for **weak-chain** and **weak-p/b** was found to be virtually identical, so Figure 4 has only a single curve—labeled weak— rather than separate curves for **weak-chain** and **weak-p/b** . 
>  weak-chain, weak-p/b 的吞吐基本一致

Observe that chain replication (**chain**) has equal or superior performance to primary-backup (**p/b**) for all percentages of updates and each replication factor investigated. This is consistent with our expectations, because the head and the tail in chain replication share a load that, with the primary/backup approach, is handled solely by the primary. 
> chain 的性能优于 p/b，这与我们的预期一致，因为在 chain replication 中，head 和 tail 共同分担负载，而在 primary/backup 模式下，负载完全由 primary 处理

The curves for the weak variant of chain replication are perhaps surprising, as these weak variants are seen to perform worse than chain replication (with its strong consistency) when there are more than $15\%$ update requests. Two factors are involved: 

- The weak variants of chain replication and primary/backup outperform pure chain replication for query-heavy loads by distributing the query load over all servers, an advantage that increases with replication factor. 
- Once the percentage of update requests increases, ordinary chain replication outperforms its weak variant—since all updates are done at the head. In particular, under pure chain replication (i) queries are not delayed at the head awaiting completion of update requests (which are relatively time consuming) and (ii) there is more capacity available at the head for update request processing if query requests are not also being handled there. 

>  当 update 比例超过 15% 时，chain-weak 比 chain replication 本身更差，这有两个原因:
>  - chain, p/b 的 weak 变体因为将 query 负载分布到所有的副本，故 query 比例越高，其表现越好，且该优势随着 replication factor 而增加
>  - 当 update 比例增加，chain 优于其 weak 变体，因为所有的 update 都在 head 处理，并且在 chain 中 (i) query 不会在 head 等待 update 请求 (相对耗时) 的完成 (ii) query 不在 head 处理，head 具有更多的容量处理 update 请求

Since **weak-chain** and **weak-p/b** do not implement strong consistency guarantees, there would seem to be surprisingly few settings where these replication management schemes would be preferred. 

Finally, note that the throughput of both chain replication and primary backup is not affected by replication factor provided there are sufficient concurrent requests so that multiple requests can be pipelined. 
>  注意，只要存在足够多的并发请求，使得请求需要被流水线化处理，chain replication 和 primary backup 的吞吐就都不会受到 replication factor 的影响

## 5.2 Multiple Chains, No Failures 
If each object is managed by a separate chain and objects are large, then adding a new replica could involve considerable delay because of the time required for transferring an object’s state to that new replica. If, on the other hand, objects are small, then a large storage service will involve many objects. Each processor in the system is now likely to host servers from multiple chains—the costs of multiplexing the processors and communications channels may become prohibitive. Moreover, the failure of a single processor now affects multiple chains. 
>  如果每个对象都由一个单独的链管理，且对象较大，则添加新副本可能带来较大的延迟，因为需要花费时间将对象的状态传输到新副本上；而如果对象较小，则大型存储服务将涉及大量对象；系统中的每个处理器都可能托管来自多条链的服务器，对处理器和通信信道进行复用的成本可能很高，此外，单个处理器的故障可能影响多条链

A set of objects can always be grouped into a single volume, itself something that could be considered an object for purposes of chain replication, so a designer has considerable latitude in deciding object size. 
>  可以将一组对象归为一个卷，将该卷本身视为单个对象，进行 chain replication，故对象的大小可以自由决定

For the next set of experiments, we assume 

- a constant number of volumes, 
- a hash function maps each object to a volume, hence to a unique chain, and 
- each chain comprises servers hosted by processors selected from among those implementing the storage service. 

>  之后的试验中，我们假设有
>  - 常数数量的卷
>  - 一个哈希函数，将每个对象映射到一个卷，进而映射到一条唯一的链
>  - 每条链都由实现了存储服务的处理器上运行的服务器构成

Clients are assumed to send their requests to a dispatcher which (i) computes the hash to determine the volume, hence chain, storing the object of concern and then (ii) forwards that request to the corresponding chain. (The master sends configuration information for each volume to the dispatcher, avoiding the need for the master to communicate directly with clients. Interposing a dispatcher adds a 1ms delay to updates and queries, but doesn’t affect throughput.) The reply produced by the chain is sent directly to the client and not by way of the dispatcher. 
>  客户端将其请求发送给调度器，该调度器
>  (i) 根据请求涉及的对象计算哈希值以确定卷，进而确定链
>  (ii) 将该请求转发给对应的链 (master 将每个卷的配置信息提前发送给了调度器，不需要和客户端直接交互)
>  插入调度器为 update 和 query 添加了 1ms 延迟，但不影响吞吐量，链生成的回复直接发送给客户端，不会经过调度器

There are 25 clients in our experiments, each submitting queries and updates at random, uniformly distributed over the chains. The clients send requests as fast as they can, subject to the restriction that each client can have only one request outstanding at a time. 
>  试验中有 25 个客户端，每个客户端均匀向各个链发送 query 和 update 请求，客户端尽可能快地发送请求，但每个客户端最多只能有一个未处理的请求

![[pics/chain replication-Fig5.png]]

To facilitate comparisons with the GFS experiments [11], we assume 5000 volumes each replicated three times, and we vary the number of servers. We found little or no difference among chain, p/b , weak-chain, and weak-p/b alternatives, so Figure 5 shows the average request throughput per client for one—chain replication—as a function of the number of servers, for varying percentages of update requests. 
>  我们假设有 5000 个卷，每个卷复制三次
>  四类协议的吞吐量表现基本一致，故 Fig5 仅显示了 chain replication 协议中，在不同的 update 请求百分比下，每个客户端的平均请求吞吐随服务器数量变化的函数

## 5.3 Effects of Failures on Throughput 
With chain replication, each server failure causes a three-stage process to start: 
1. Some time (we conservatively assume 10 seconds in our experiments) elapses before the master detects the server failure. 
2. The offending server is then deleted from the chain. 
3. The master ultimately adds a new server to that chain and initiates a data recovery process, which takes time proportional to (i) how much data was being stored on the faulty server and (ii) the available network bandwidth. 

>  chain replication 中，每次服务器故障将导致一个三阶段流程被启动
>  1. 在 master 检测到服务器故障之前，会经过一段时间 (我们假设为 10s)
>  2. 故障服务器从链中被删除
>  3. master 将新服务器加入，并启动数据恢复流程，该流程所需时间正比于 (i) 故障服务器中存储的数据量 (ii) 可用的网络带宽

Delays in detecting a failure or in deleting a faulty server from a chain can increase request processing latency and can increase transient outage duration. The experiments in this section explore this. 
>  检测故障或删除链中故障服务器的延迟将提高请求处理延迟和提高服务中断时间

![[pics/chain replication-Table1.png]]

We assume a storage service characterized by the parameters in Table 1; these values are inspired by what is reported for GFS [11]. The assumption about network bandwidth is based on reserving for data recovery at most half the bandwidth in a 100 Mbit/second network; the time to copy the 150 Gigabytes stored on one server is now 6 hours and 40 minutes. 
>  存储服务的配置参数如 Table 1
>  其中

In order to measure the effects of a failures on the storage service, we apply a load. The exact details of the load do not matter greatly. Our experiments use eleven clients. Each client repeatedly chooses a random object, performs an operation, and awaits a reply; a watchdog timer causes the client to start the next loop iteration if 3 seconds elapse and no reply has been received. Ten of the clients exclusively submit query operations; the eleventh client exclusively submits update operations. 
>  为了衡量故障对存储服务的影响，我们引发故障的同时施加负载
>  试验使用 11 个客户端，每个客户端反复选择一个随机对象，对其执行操作，等待回复，如果 3s 内没有回复，客户端将开始下一轮循环，
>  10 个客户端仅提交 query 操作，1 个客户端仅提交 update 操作

Each experiment described executes for 2 simulated hours. Thirty minutes into the experiment, the failure of one or two servers is simulated (as in the GFS experiments). The master detects that failure and deletes the failed server from all of the chains involving that server. For each chain that was shortened by the failure, the master then selects a new server to add. Data recovery to those servers is started. 
>  每次试验执行两个消失，试验进行 30min 后，一到两个服务器会故障，master 检测到故障后，会删除该服务器在其关联的所有链中的角色
>  master 为每个减短的链选择需要加入的新服务器，然后启动对这些服务器的数据恢复过程

![[pics/chain replication-Fig6.png]]

Figure 6(a) shows aggregate query and update throughputs as a function of time in the case a single server $F$ fails. Note the sudden drop in throughput when the simulated failure occurs 30 minutes into the experiment. The resolution of the $x$ -axis is too coarse to see that the throughput is actually zero for about 10 seconds after the failure, since the master requires a bit more than 10 seconds to detect the server failure and then delete the failed server from all chains. 
>  Fig6a 展示了单个服务器故障下，聚合 query/update 吞吐关于时间的函数
>  在故障后，实际上会出现 10s 吞吐为零，因为 master 需要 10s 的时间检测到故障服务器，并将其从所有链中删除

With the failed server deleted from all chains, processing now can proceed, albeit at a somewhat lower rate because fewer servers are operational (and the same request processing load must be shared among them) and because data recovery is consuming resources at various servers. Lower curves on the graph reflect this. 
>  当故障服务器从所有链中删除后，请求处理可以继续进行，但整体的吞吐率变得更低，因为此时服务器数量更少，每个服务器要处理更多数量的负载 (注意在 Table 1 的配置下，为了负载均衡，每个服务器基本上都会运行 200 个左右的 head 和 200 个左右的 tail，故一个服务器故障后，其他各个服务器都需要承担更多的 head 和 tail 任务)，并且此时的数据恢复过程也需要在各个服务器上消耗资源

After 10 minutes, failed server $F$ becomes operational again, and it becomes a possible target for data recovery. Every time data recovery of some volume successfully completes at $F$ , query throughput improves (as seen on the graph). This is because $F$ , now the tail for another chain, is handling a growing proportion of the query load. 
>  10min 后，故障服务器 $F$ 再次可以运行，该服务器可以作为数据恢复的对象，当某个卷的数据成功在 $F$ 上恢复后，吞吐就会提高，因为 $F$ 此时担任了该卷的链的 tail，可以处理对应的 query 负载

One might expect that after all data recovery concludes, the query throughput would be what it was at the start of the experiment. The reality is more subtle, because volumes are no longer uniformly distributed among the servers. In particular, server $F$ will now participate in fewer chains than other servers but will be the tail of every chain in which it does participate. So the load is no longer well balanced over the servers, and aggregate query throughput is lower. 
>  在数据完全恢复后，query 吞吐并不会期望地恢复到试验之前的水平，因为卷不再均匀地分布在各个服务器上
>  特别地，$F$ 相对于其他服务器，将参与更少的链，但它在其参与的每个链中都作为 tail，因此服务器之间的负载不再均衡，故聚合 query 带宽变得更低

Update throughput decreases to 0 at the time of the server failure and then, once the master deletes the failed server from all chains, throughput is actually better than it was initially. This throughput improvement occurs because the server failure causes some chains to be length 2 (rather than 3), reducing the amount of work involved in performing an update. 
>  在服务器故障的时刻，update 吞吐会降低至 0，而当 master 从所有链中删除了故障服务器后，update 吞吐实际比之前更高
>  这是因为一些链的长度变短了，故执行一次 update 的工作量变少了

The GFS experiments [11] consider the case where two servers fail, too, so Figure 6(b) depicts this for our chain replication protocol. Recovery is still smooth, although it takes additional time. 

## 5.4 Large Scale Replication of Critical Data 
As the number of servers increases, so should the aggregate rate of server failures. If too many servers fail, then a volume might become unavailable. The probability of this depends on how volumes are placed on servers and, in particular, the extent to which parallelism is possible during data recovery. 
>  随着服务器数量增加，服务器故障的总概率也会增加，如果过多的服务器故障，就会导致某个卷不可用
>  具体的概率取决于卷是如何分布在服务器中的，以及数据恢复时的并行程度是多少

We have investigated three volume placement strategies: 
- **ring**: Replicas of a volume are placed at consecutive servers on a ring, determined by a consistent hash of the volume identifier. This is the strategy used in CFS [7] and PAST [19]. The number of parallel data recoveries possible is limited by the chain length $t$ . 
- **rndpar**: Replicas of a volume are placed randomly on servers. This is essentially the strategy used in GFS. $^5$ Notice that, given enough servers, there is no limit on the number of parallel data recoveries possible. 
- **rndseq**: Replicas of a volume are placed randomly on servers (as in rndpar), but the maximum number of parallel data recoveries is limited by $t$ (as in ring). This strategy is not used in any system known to us but is a useful benchmark for quantifying the impacts of placement and parallel recovery. 

>  我们研究了三种卷放置策略
>  - 环形 ring: 卷的副本放置在环形的连续服务器上，并行数据恢复的数量受到链长 $t$ 的限制
>  - 随机并行 rndpar: 卷的副本随机放置在各个服务器上，如果有足够多的服务器，则并行数据恢复的数量没有限制
>  - 随机序列 rndseq: 卷的副本随机放置在各个服务器上，但最大数量的数据恢复受限于 $t$，任何系统都没有采用该策略，但该策略是常用的量化放置和并行恢复影响的基准

To understand the advantages of parallel data recovery, consider a server $F$ that fails and was participating in chains $C_{1},C_{2},\ldots,C_{n}$ . For each chain $C_{i}$ , data recovery requires a source from which the volume data is fetched and a host that will become the new element of chain $C_{i}$ . Given enough processors and no constraints on the placement of volumes, it is easy to ensure that the new elements are all disjoint. And with random placement of volumes, it is likely that the sources will be disjoint as well. With disjoint sources and new elements, data recovery for chains $C_{1},C_{2},\ldots,C_{n}$ can occur in parallel. And a shorter interval for data recovery of $C_{1},C_{2},\ldots,C_{n}$ , implies that there is a shorter window of vulnerability during which a small number of concurrent failures would render some volume unavailable. 
>  为了理解并行数据恢复的优势，我们考虑一个服务器 $F$，它参与链 $C_1, C_2,\dots, C_n$，并故障
>  此时，对于每条链 $C_i$，数据恢复需要一个数据源用于获取卷数据和一个主机成为链 $C_i$ 的新元素
>  假设具有足够的处理器，并且对卷的放置没有约束，则各个链 $C_i$ 的新元素可以没有交集 (不会由同一个处理器在多个链中承担角色)，并且在随机放置卷的情况下，数据源也可能不相交
>  如果数据源和新元素是不相交的，链 $C_1, C_2, \dots, C_n$ 的恢复就可以并行进行

We seek to quantify the mean time between unavailability (MTBU) of any object as a function of the number of servers and the placement strategy. Each server is assumed to exhibit exponentially distributed failures with a MTBF (Mean Time Between Failures) of 24 hours. $^6$  As the number of servers in a storage system increases, so would the number of volumes (otherwise, why add servers). In our experiments, the number of volumes is defined to be 100 times the initial number of servers, with each server storing 100 volumes at time 0. 
>  我们视图将任何对象的平均无故障时间表达为服务器数量和放置策略的函数
>  假设每个服务器的故障时间服从指数分布，其平均故障时间为 24 小时
>  随着存储系统中的服务器数量增加，卷的数量也会增加
>  我们的试验中，卷的数量定义为初始服务器数量的 100 倍，每台服务器在时刻 0 存储 100 个卷

We postulate that the time it takes to copy all the data from one server to another is four hours, which corresponds to copying 100 Gigabytes across a 100 Mbit/sec network restricted so that only half bandwidth can be used for data recovery. As in the GFS experiments, the maximum number of parallel data recoveries on the network is limited to $40\%$ of the servers, and the minimum transfer time is set to 10 seconds (the time it takes to copy an individual GFS object, which is 64 KBytes). 
>  我们假设将所有数据从一个服务器拷贝到另一个服务器的时间是 4 小时
>  网络中最大并行恢复的数据量限制为服务器总数的 40%，最小传输时间设定为 10s

![[pics/chain replication-Fig7.png]]

Figure 7(a) shows that the MTBU for the ring strategy appears to have an approximately Zipfian distribution as a function of the number of servers. 
>  图 7 (a) 显示，环形策略的平均无故障时间似乎随着服务器数量呈 Zipf 分布。

Thus, in order to maintain a particular MTBU, it is necessary to grow chain length $t$ when increasing the number of servers. From the graph, it seems as though chain length needs to be increased as the logarithm of the number of servers. 
>  因此，为了维持特定的 MTBU，在增加服务器数量时需要增长链长。从图表中可以看出，链长似乎需要以服务器数量的对数形式增长。

Figure 7(b) shows the MTBU for rndseq. For $t>$ 1, rndseq has lower MTBU than ring. Compared to ring, random placement is inferior because with random placement there are more sets of $t$ servers that together store a copy of a chain, and therefore there is a higher probability of a chain getting lost due to failures. 
>  图 7 (b) 展示了随机串行的 MTBU。对于 $t>1$ 的情况，随机串行的 MTBU 低于环形策略。与环形策略相比，随机放置表现较差，因为在随机放置的情况下，存在更多包含 $t$ 台服务器的集合，这些集合共同存储一条链的副本，因此由于故障导致链丢失的概率更高。

However, random placement makes additional opportunities for parallel recovery possible if there are enough servers. Figure 7(c) shows the MTBU for rndpar. For few servers, rndpar performs the same as rndseq, but the increasing opportunity for parallel recovery with the number of servers improves the MTBU, and eventually rndpar outperforms rndseq, and more importantly, it outperforms ring. 
>  然而，如果服务器数量足够多，随机放置可以提供更多并行恢复的机会。图 7 (c) 展示了随机并行的 MTBU。对于少量服务器，随机并行的表现与随机串行相同，但随着服务器数量的增加，并行恢复的机会增多，从而提高了 MTBU，最终随机并行的表现优于随机串行，更重要的是，它也优于环形策略。

---

$^5$ Actually, the placement strategy is not discussed in [11]. GFS does some load balancing that results in an approximately even load across the servers, and in our simulations we expect that random placement is a good approximation of this strategy. 

$^6$ An unrealistically short MTBF was selected here to facilitate running long-duration simulations. 

# 6 Related Work 
**Scalability.** Chain replication is an example of what Jimenéz-Peris and Patino-Martinez [14] call a ROWAA (read one, write all available) approach. They report that ROWAA approaches provide superior scaling of availability to quorum techniques, claiming that availability of ROWAA approaches improves exponentially with the number of replicas. They also argue that non-ROWAA approaches to replication will necessarily be inferior. Because ROWAA approaches also exhibit better throughout than the best known quorum systems (except for nearly write-only applications) [14], ROWAA would seem to be the better choice for replication in most real settings. 
>  **可扩展性** 
>  链式复制是 Jimenéz-Peris 和 Patino-Martinez [14] 所称的 ROWAA（读取一个，写入所有可用）方法的一个例子。他们报告说，与一致性投票技术相比，ROWAA 方法在可用性方面提供了更优越的扩展性，并声称 ROWAA 方法的可用性会随着副本数量的增加呈指数级提升。他们还指出，非 ROWAA 的复制方法必然表现较差。由于 ROWAA 方法在吞吐量上也优于已知的最佳一致性投票系统（除了几乎只写入的应用程序外）[14]，因此在大多数实际场景中，ROWAA 应该是复制的更好选择。

Many file services trade consistency for performance and scalability. Examples include Bayou [17], Ficus [13], Coda [15], and Sprite [5]. Typically, these systems allow continued operation when a network partitions by offering tools to fix inconsistencies semi-automatically. Our chain replication does not offer graceful handling of partitioned operation, trading that instead for supporting all three of: high performance, scalability, and strong consistency. 
>  许多文件服务以性能和可扩展性为代价换取一致性。例子包括 Bayou [17]、Ficus [13]、Coda [15] 和 Sprite [5]。通常，这些系统在网络分区时允许继续运行，通过提供半自动修复不一致性的工具来实现这一点。我们的链式复制并不提供对分区操作的优雅处理方式，而是以此为代价，支持以下三个方面：高性能、可扩展性和强一致性。

Large-scale peer-to-peer reliable file systems are a relatively recent avenue of inquiry. OceanStore [6], FARSITE [2], and PAST [19] are examples. Of these, only OceanStore provides strong (in fact, transactional) consistency guarantees. 
大规模的对等可靠文件系统是一个相对较新的研究领域。OceanStore [6]、FARSITE [2] 和 PAST [19] 就是其中的例子。在这之中，只有 OceanStore 提供了强一致性（实际上是事务性）保证。

Google’s File System (GFS) [11] is a large-scale cluster-based reliable file system intended for applications similar to those motivating the invention of chain replication. But in GFS, concurrent overwrites are not serialized and read operations are not synchronized with write operations. Consequently, different replicas can be left in different states, and content returned by read operations may appear to vanish spontaneously from GFS. Such weak semantics imposes a burden on programmers of applications that use GFS. 
>  谷歌文件系统（GFS）[11] 是一种基于集群的大规模可靠文件系统，旨在服务于与链式复制发明动机相似的应用程序。但在 GFS 中，并发覆盖写入操作未被序列化，读取操作也未与写入操作同步。因此，不同的副本可能会处于不同的状态，且由读取操作返回的内容可能看似在 GFS 中自发消失。这种弱语义给使用 GFS 的应用程序开发者带来了负担。

**Availability versus Consistency.** Yu and Vahdat [25] explore the trade-off between consistency and availability. They argue that even in relaxed consistency models, it is important to stay as close to strong consistency as possible if availability is to be maintained in the long run. On the other hand, Gray et al. [12] argue that systems with strong consistency have unstable behavior when scaled-up, and they propose the tentative update transaction for circumventing these scalability problems. 
>  **可用性与一致性。** 
>  俞和瓦赫达特 [25] 探讨了一致性和可用性之间的权衡。他们认为，即使在宽松的一致性模型中，如果要长期维持可用性，也应尽可能接近强一致性。另一方面，格雷等人 [12] 认为具有强一致性的系统在扩展时表现出不稳定的行为，并提出了暂定更新事务以规避这些可扩展性问题。

Amza et al. [4] present a one-copy serializable transaction protocol that is optimized for replication. As in chain replication, updates are sent to all replicas whereas queries are processed only by replicas known to store all completed updates. (In chain replication, the tail is the one replica known to store all completed updates.) The protocol of [4] performs as well as replication protocols that provide weak consistency, and it scales well in the number of replicas. No analysis is given for behavior in the face of failures. 
>  阿姆扎等 [4] 提出了一种针对复制优化的一次写入可序列化事务协议。与链式复制类似，更新被发送到所有副本，而查询仅由已知存储所有已完成更新的副本处理。（在链式复制中，尾部是已知存储所有已完成更新的副本。）[4]中的协议性能与提供弱一致性的复制协议相当，并且在副本数量增加时表现良好。未对面对故障时的行为进行分析。

**Replica Placement.** Previous work on replica placement has focussed on achieving high throughput and/or low latency rather than on supporting high availability. Acharya and Zdonik [1] advocate locating replicas according to predictions of future accesses (basing those predictions on past accesses). In the Mariposa project [23], a set of rules allows users to specify where to create replicas, whether to move data to the query or the query to the data, where to cache data, and more. Consistency is transactional, but no consideration is given to availability. Wolfson et al. consider strategies to optimize database replica placement in order to optimize performance [24]. The OceanStore project also considers replica placement [10, 6] but from the CDN (Content Distribution Network, such as Akamai) perspective of creating as few replicas as possible while supporting certain quality of service guarantees. There is a significant body of work (e.g., [18]) concerned with placement of web page replicas as well, all from the perspective of reducing latency and network load. 
>  **副本放置** 
>  之前关于副本放置的研究主要集中在实现高吞吐量和/或低延迟，而不是支持高可用性。Acharya 和 Zdonik [1] 提倡根据未来访问的预测来定位副本（基于过去的访问进行预测）。在 Mariposa 项目 [23] 中，一组规则允许用户指定在哪里创建副本、是否将数据移动到查询或查询移动到数据、在哪里缓存数据等。一致性是事务性的，但没有考虑可用性。Wolfson 等人研究了优化数据库副本放置的策略以优化性能 [24]。OceanStore 项目也考虑了副本放置 [10, 6]，但从减少副本数量和支持一定服务质量保证的 CDN（内容分发网络，如 Akamai）视角出发。还有大量工作（例如 [18]）关注网页副本的放置，目的是降低延迟和网络负载。

Douceur and Wattenhofer investigate how to maximize the worst-case availability of files in FARSITE [2], while spreading the storage load evenly across all servers [8, 9]. Servers are assumed to have varying availabilities. The algorithms they consider repeatedly swap files between machines if doing so improves file availability. The results are of a theoretical nature for simple scenarios; it is unclear how well these algorithms will work in a realistic storage system. 
>  Douceur 和 Wattenhofer 探讨了如何在 FARSITE [2] 中最大化文件的最坏情况下的可用性，同时在所有服务器之间均匀分配存储负载 [8, 9]。假设服务器具有不同的可用性。他们所考虑的算法如果这样做可以提高文件可用性，则会反复在机器之间交换文件。这些结果在简单场景下具有理论性质；在现实的存储系统中，这些算法的表现尚不清楚。

# 7 Concluding Remarks 
Chain replication supports high throughput for query and update requests, high availability of data objects, and strong consistency guarantees. This is possible, in part, because storage services built using chain replication can and do exhibit transient outages but clients cannot distinguish such outages from lost messages. Thus, the transient outages that chain replication introduces do not expose clients to new failure modes—chain replication represents an interesting balance between what failures it hides from clients and what failures it doesn’t. 
>  chain replication 支持高吞吐的 query, update 请求，高可用的数据对象，以及强一致性保证
>  这在一定程度上成为可能，因为使用 chain replication 构建的存储服务可能会并且确实会出现短暂的中断，但客户端无法区分服务中断与消息丢失，因此，chain replication 引入的短暂中断不会让客户端暴露于新的故障模式中——chain replication 在隐藏哪些故障和不隐藏哪些故障之间实现了有趣的平衡

When chain replication is employed, high availability of data objects comes from carefully selecting a strategy for placement of volume replicas on servers. Our experiments demonstrated that with DHT-based placement strategies, availability is unlikely to scale with increases in the numbers of servers; but we also demonstrated that random placement of volumes does permit availability to scale with the number of servers if this placement strategy is used in concert with parallel data recovery, as introduced for GFS. 
>  当采用链式复制时，数据对象的高可用性来自于精心选择在服务器上放置卷副本的策略
>  我们的实验表明，使用基于分布式哈希表（DHT）的放置策略时，随着服务器数量的增加，可用性不太可能随之扩展；但我们同时也证明了，如果将随机放置卷的策略与并行数据恢复相结合（如 GFS 中引入的那样），则可用性确实可以随着服务器数量的增加而扩展

Our current prototype is intended primarily for use in relatively homogeneous LAN clusters. Were our prototype to be deployed in a heterogeneous wide-area setting, then uniform random placement of volume replicas would no longer make sense. Instead, replica placement would have to depend on access patterns, network proximity, and observed host reliability. Protocols to re-order the elements of a chain would likely become crucial in order to control load imbalances. 
>  我们目前的原型主要用于相对同质的局域网集群
>  如果我们的原型被部署在异构的广域环境中，那么均匀随机放置卷副本将不再有意义，相反，副本的放置需要依赖访问模式、网络邻近性以及观察到的主机可靠性
>  为了控制负载不平衡，重新排列链元素的协议很可能变得至关重要

Our prototype chain replication implementation consists of 1500 lines of Java code, plus another 2300 lines of Java code for a Paxos library. The chain replication protocols are structured as a library that makes upcalls to a storage service (or other application). The experiments in this paper assumed a “null service” on a simulated network. But the library also runs over the Java socket library, so it could be used to support a variety of storage servicelike applications. 
>  我们的原型链式复制实现包含 1500 行 Java 代码，另外还有 2300 行 Java 代码用于 Paxos 库
>  链式复制协议被设计为一个库，可以向上调用存储服务（或其他应用程序）
>  本文的实验假设在模拟网络中有一个“空服务”，但是该库也可以运行在 Java 套接字库之上，因此它可以用于支持各种类似于存储服务的应用程序

