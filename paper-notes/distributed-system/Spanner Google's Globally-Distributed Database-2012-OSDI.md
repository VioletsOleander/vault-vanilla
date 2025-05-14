# Abstract 
Spanner is Google’s scalable, multi-version, globally-distributed, and synchronously-replicated database. It is the first system to distribute data at global scale and support externally-consistent distributed transactions. 
>  Spanner 是 Google 的可拓展，多版本，全球分布式且同步复制的数据库
>  Spanner 是第一个在全球范围内分布数据并支持外部一致的分布式事务的系统

This paper describes how Spanner is structured, its feature set, the rationale underlying various design decisions, and a novel time API that exposes clock uncertainty. This API and its implementation are critical to supporting external consistency and a variety of powerful features: nonblocking reads in the past, lock-free read-only transactions, and atomic schema changes, across all of Spanner. 
>  本文介绍 Spanner 的架构，其功能集，各种设计决策背后的原理，以及一个暴露了时钟不确定性的时间 API
>  该 API 及其实现对于支持外部一致性以及一系列功能至关重要: 跨整个 Spanner 系统的过去非阻塞式读取、无锁只读事务、原子模式更改

> [!info] External Consistency
> External consistency (外部一致性) 是一种分布式系统中的数据一致性模型，在这种模型下，系统对外部观察者 (用户或者其他系统) 展现出一种顺序或者行为，就好像所有对数据的更改都是按照一个全局的、预定义的顺序来执行的一样
> 对于外部观察者来说，他们看到的系统状态变化是符合某个一致的逻辑顺序的，就好像所有的操作都是在一个单一的系统中以顺序方式执行的，而不是在多个分布式节点上并行执行的
> 
> - 与内部一致性 (Internal consistency) 的区别:
>     内部一致性主要关注系统内部各组件之间的状态一致性，例如，在一个分布式数据库集群中，内部一致性可能涉及数据在不同副本节点之间的同步，确保副本节点之间的数据是一致的
>     而外部一致性是从外部观察者的角度来定义的，侧重于外部看到的系统行为的一致性，比如，当一个用户向一个分布式系统发送一个请求来更新数据，然后另一个用户紧接着读取这个数据，外部一致性强调第二个用户读取到的数据应该符合第一个用户更新后的逻辑结果，而不是关注系统内部的数据副本同步细节
> - 与顺序一致性 (Sequential consistency) 的关系:
>     顺序一致性要求多线程或者多进程访问共享变量时，每个操作看起来都是按照全局的顺序执行的，这个顺序与程序中操作的顺序一致
>     外部一致性在某种程度上可以看作是顺序一致性的扩展，它不仅涉及到程序内部的顺序，**还考虑了对外部观察者的呈现**
>     例如，在一个分布式消息队列系统中，消息的生产和消费过程需要满足外部一致性，对于外部消费者来说，消息的消费顺序应该符合消息生产者发送消息的逻辑顺序，这就类似于顺序一致性在外部系统中的体现

# 1 Introduction 
Spanner is a scalable, globally-distributed database designed, built, and deployed at Google. At the highest level of abstraction, it is a database that shards data across many sets of Paxos [21] state machines in datacenters spread all over the world. Replication is used for global availability and geographic locality; clients automatically failover between replicas. Spanner automatically reshards data across machines as the amount of data or the number of servers changes, and it automatically migrates data across machines (even across datacenters) to balance load and in response to failures. Spanner is designed to scale up to millions of machines across hundreds of datacenters and trillions of database rows. 
>  从最高的抽象层次出发，Spanner 是一个将数据分散到世界各地的数据中心中的多组 Paxos 状态机的数据库系统
>  Spanner 使用复制实现全球可用性和地理局部性，客户端会自动在副本之间进行故障切换
>  随着数据量或服务器数量变化，Spanner 会自动对数据进行重新分片，并且会自动在机器之间 (甚至跨数据中心) 迁移数据以均衡负载和应对故障
>  Spanner 的设计目标是能拓展到数百个数据中心的数百万台机器以及万亿级别的数据库行

Applications can use Spanner for high availability, even in the face of wide-area natural disasters, by replicating their data within or even across continents. Our initial customer was F1 [35], a rewrite of Google’s advertising backend. F1 uses five replicas spread across the United States. Most other applications will probably replicate their data across 3 to 5 datacenters in one geographic region, but with relatively independent failure modes. That is, most applications will choose lower latency over higher availability, as long as they can survive 1 or 2 datacenter failures. 
>  应用可以利用 Spanner 将数据在大陆内甚至跨大陆进行复制，以实现面临自然灾害也能维持的高可用性
>  Spanner 的首个客户是 F1, F1 是 Google 广告后台的重写版本, F1 在美国分布了 5 个副本
>  大多数其他应用一般会在一个地理区域内将其数据跨 3 到 5 个数据中心复制，但这些数据中心的故障模式相对独立，也就是说，只要能够承受 1 到 2 个数据中心的故障，大多数应用会选择低延迟而不是高可用性

Spanner’s main focus is managing cross-datacenter replicated data, but we have also spent a great deal of time in designing and implementing important database features on top of our distributed-systems infrastructure. 
>  Spanner 的主要关注点在于管理跨数据中心复制的数据，同时我们也基于我们的分布式系统基础设施上，为 Spanner 设计并实现了重要的数据库功能

Even though many projects happily use Bigtable [9], we have also consistently received complaints from users that Bigtable can be difficult to use for some kinds of applications: those that have complex, evolving schemas, or those that want strong consistency in the presence of wide-area replication. (Similar claims have been made by other authors [37].) Many applications at Google have chosen to use Megastore [5] because of its semi-relational data model and support for synchronous replication, despite its relatively poor write throughput. As a consequence, Spanner has evolved from a Bigtable-like versioned key-value store into a temporal multi-version database. Data is stored in schematized semi-relational tables; data is versioned, and each version is automatically timestamped with its commit time; old versions of data are subject to configurable garbage-collection policies; and applications can read data at old timestamps. Spanner supports general-purpose transactions, and provides a SQL-based query language. 
>  虽然我们的许多项目已经基于 Bigtable 实现，但 Bigtable 对于某些类型的应用也是不便于使用的: 那些具有复杂且不断演化的模型的应用，或者那些需要在广域复制的情况下也保持一致性的应用
>  为此，Google 的许多应用选择使用具有半关系型数据模型并支持同步复制的 Megastore，即便它的写吞吐较差
>  因此 (为了满足需求)，Spanner 从一个类 Bigtable 的版本化键值存储演变为时序多版本数据库，数据以模式化的半关系表的形式存储; 数据被版本化，其每个版本的时间戳为其提交时间; 旧版本的数据基于可配置的垃圾回收策略清理; 应用可以读取旧版本的数据
>  Spanner 支持通用事务，并提供基于 SQL 的查询语言

As a globally-distributed database, Spanner provides several interesting features. First, the replication configurations for data can be dynamically controlled at a fine grain by applications. Applications can specify constraints to control which datacenters contain which data, how far data is from its users (to control read latency), how far replicas are from each other (to control write latency), and how many replicas are maintained (to control durability, availability, and read performance). Data can also be dynamically and transparently moved between datacenters by the system to balance resource usage across datacenters. Second, Spanner has two features that are difficult to implement in a distributed database: it provides externally consistent [16] reads and writes, and globally-consistent reads across the database at a timestamp. These features enable Spanner to support consistent backups, consistent MapReduce executions [12], and atomic schema updates, all at global scale, and even in the presence of ongoing transactions. 
>  作为一个全球分布式数据库，Spanner 提供了多种功能
>  首先，应用可以细粒度地动态控制数据的复制配置。应用可以指定约束，以控制哪个数据中心存储哪些数据、数据距离用户有多远 (以控制读延迟)、副本之间的距离有多远 (以控制写延迟)、要维护多少个副本 (以控制持久性、可用性和读性能)。应用可以通过 Spanner 在数据中心之间动态且透明地移动数据，以平衡数据中心的资源使用
>  其次，Spanner 具有两个在分布式数据库中难以实现的功能: 它提供了外部一致的读写，以及在时间戳上的跨整个数据库的全局一致读取。这些功能使得 Spanner 可以在全球范围内支持一致备份、一致 MapReduce 执行和原子化模式更新，并且同时可以持续运行事务

These features are enabled by the fact that Spanner assigns globally-meaningful commit timestamps to transactions, even though transactions may be distributed. The timestamps reflect serialization order. In addition, the serialization order satisfies external consistency (or equivalently, linearizability [20]): if a transaction $T_{1}$ commits before another transaction $T_{2}$ starts, then $T_{1}$ ’s commit timestamp is smaller than $T_{2}$ ’s. Spanner is the first system to provide such guarantees at global scale. 
>  这些功能的实现得益于 Spanner 为事务分配了全局的提交时间戳，即便事务可能说分布式的
>  这些时间戳反映了序列化顺序，这一序列化顺序满足了外部一致性 (或者等价地说，线性一致性): 如果事务 $T_1$ 在另一个事务 $T_2$ 开始之前提交，则 $T_1$ 的提交时间戳将小于 $T_2$ 的提交时间戳
>  Spanner 是首个在全球规模提供了这样保证的系统

The key enabler of these properties is a new TrueTime API and its implementation. The API directly exposes clock uncertainty, and the guarantees on Spanner’s timestamps depend on the bounds that the implementation provides. If the uncertainty is large, Spanner slows down to wait out that uncertainty. Google’s cluster-management software provides an implementation of the TrueTime API. This implementation keeps uncertainty small (generally less than 10ms) by using multiple modern clock references (GPS and atomic clocks). 
>  这些性质的实现的关键在于 TrueTime API，该 API 直接暴露了时钟不确定性，而 Spanner 的时间戳保证依赖于该 API 实现所提供的界
>  如果不确定性较大，Spanner 会减速以等待不确定性消失
>  Google 的集群管理软件提供了 TrueTime API 的实现，该实现通过使用多个现代实现参考 (GPS 和原子钟)，将不确定性保持在较小范围内 (通常低于 10ms)

> [!info] Clock Uncertainty
> Clock uncertainty (时钟不确定性) 是指时钟信号在频率、相位或周期方面的变化，以及这些变化导致的问题。
> 
> **产生原因**
> 1.硬件因素
> 晶体振荡器的不稳定性：在电子设备中，晶体振荡器通常是时钟信号的来源。温度、老化等因素会导致它的振动频率产生微小的漂移，这种频率漂移会使时钟信号的周期发生变化，进而产生时钟不确定性。
> 电源噪声：电源的不稳定会引入噪声，影响时钟电路的工作。电源中的噪声信号可能会叠加到时钟信号上，改变时钟信号的波形和幅度。比如，在一个复杂的数字电路板上，多个芯片同时工作时，电源的电流和电压波动可能会使时钟信号的上升沿和下降沿出现抖动，增加时钟不确定性。
> 
> 2.信号传输因素
> 布线延迟：在电路板布线中，时钟信号的传输路径长度会影响时钟的延迟。如果不同的时钟信号路径长度不一致，会导致时钟信号到达各个接收端的时间不同。例如，在一个高性能的计算机主板上，时钟信号从时钟发生器传输到多个处理器核心的布线长度不同，就会使时钟信号到达每个核心的时间产生差异，形成时钟不确定性。
> 信号完整性问题：信号在传输过程中可能会受到反射、串扰等因素的干扰。反射可能是由于线路阻抗不匹配引起的，它会使时钟信号的波形失真。串扰则是由于相邻信号线之间的电磁耦合，导致时钟信号受到其他信号的干扰。这些都会增加时钟的不确定性。
> 
> **具体表现形式**
   1.时钟抖动 (Jitter)
> 时钟抖动是指时钟信号在时间上**相对于理想位置**的短期偏离。它可以分为周期抖动、相位抖动等多种类型。周期抖动是相邻两个时钟周期之间的差异，相位抖动是时钟信号在相位上的随机变化。例如，在一个高速通信系统中，时钟抖动会导致数据采样点的偏移。如果采样点的时钟信号存在抖动，可能会使数据接收端无法正确地恢复发送端的数据，造成数据传输错误。
> 
> 2.时钟偏差 (Skew)
>  时钟偏差是指在同一个时钟域内，**不同位置的时钟信号**之间的时间差异。在多核处理器中，各个核心可能会受到不同的时钟偏差影响。如果时钟偏差过大，会导致不同核心之间的数据同步出现问题。比如，在一个双核处理器中，两个核心的时钟信号存在偏差，当它们需要进行数据交换时，可能会因为时钟不同步而读取到错误的数据。
> 
> **影响**
> 1.数字电路方面
> 影响数据采样。在数字系统中，数据的采样通常是在时钟信号的上升沿或下降沿进行的。时钟不确定性会导致采样时刻的不准确，使采样的数据可能存在错误。例如，在一个模数转换器（ADC）中，时钟不确定性会影响采样的精度，降低数字化信号的质量。
> 影响电路的时序性能。时钟不确定性会使电路中的门电路等逻辑元件的传输延迟发生变化，可能导致建立时间（setup time）和保持时间（hold time）等时序参数无法满足要求。这会使电路的工作频率降低，甚至导致电路无法正常工作。
> 
> 2.通信系统方面
> 影响信号的同步。在通信系统中，发送端和接收端需要通过时钟信号进行同步。时钟不确定性会导致同步失败，使接收端无法正确地恢复发送端的信号。例如，在光纤通信系统中，时钟不确定性会影响信号的时隙同步，导致数据帧的误判和丢失。
> 
> **解决方法**
> 1.硬件层面
>  采用高精度的晶体振荡器。高精度的晶体振荡器通常具有更好的温度补偿和老化特性，能够减少时钟信号的频率漂移。例如，一些高端的通信设备会使用恒温晶体振荡器（OCXO），它将晶体振荡器置于一个恒温环境中，以提高其稳定性。
>  优化电源系统设计。通过使用低噪声的电源、滤波电路等来减少电源噪声对时钟电路的影响。例如，在电路板设计中，为时钟电路提供独立的电源线路，并在电源线上添加滤波电容，以滤除高频噪声。
> 
>  2.电路布线层面
>  合理进行布线。在电路板布线时，尽量使时钟信号的传输路径长度一致，减少布线延迟带来的时钟偏差。可以采用蛇形走线等方式来匹配时钟信号的长度。同时，要保证信号传输线路的阻抗匹配，减少反射引起的时钟抖动。
>  增加屏蔽措施。对于容易受到串扰影响的时钟信号线，可以采用屏蔽线或者在信号线周围布置地线来减少串扰，提高时钟信号的完整性。
> 
>  3.软件层面（对于一些可编程系统）
>  可以采用时钟同步算法。在分布式系统中，通过软件算法来**对各个节点的时钟进行同步**。例如，网络时间协议（NTP）就是一种用于同步计算机时钟的软件解决方案，它可以减少不同节点之间时钟偏差带来的问题。

Section 2 describes the structure of Spanner’s implementation, its feature set, and the engineering decisions that went into their design. Section 3 describes our new TrueTime API and sketches its implementation. Section 4 describes how Spanner uses TrueTime to implement externally-consistent distributed transactions, lockfree read-only transactions, and atomic schema updates. Section 5 provides some benchmarks on Spanner’s performance and TrueTime behavior, and discusses the experiences of F1. Sections 6, 7, and 8 describe related and future work, and summarize our conclusions. 

# 2 Implementation 
This section describes the structure of and rationale underlying Spanner’s implementation. It then describes the directory abstraction, which is used to manage replication and locality, and is the unit of data movement. Finally, it describes our data model, why Spanner looks like a relational database instead of a key-value store, and how applications can control data locality. 
>  本节描述 Spanner 的结构和其背后的理由，然后描述目录抽象，该抽象被用于管理数据复制和数据位置，并且是数据移动的基本单位
>  最后，本节描述 Spanner 的数据模型，解释为什么 Spanner 看起来更像关系型数据库而不是键值存储，以及应用如何控制数据位置

A Spanner deployment is called a universe. Given that Spanner manages data globally, there will be only a handful of running universes. We currently run a test/playground universe, a development/production universe, and a production-only universe. 
>  一个 Spanner 部署被称为 universe
>  因为 Spanner 可以管理全球的数据，故运行的 universe 数量不多，目前，我们运行三个 universe: 一个测试/试验 universe，一个开发/部署 universe，一个仅用于生产的 universe

Spanner is organized as a set of zones, where each zone is the rough analog of a deployment of Bigtable servers [9]. 
>  Spanner 被组织为一组 zones，每个 zone 大致相当于一个 Bigtable servers 的部署

Zones are the unit of administrative deployment. The set of zones is also the set of locations across which data can be replicated. Zones can be added to or removed from a running system as new datacenters are brought into service and old ones are turned off, respectively. Zones are also the unit of physical isolation: there may be one or more zones in a datacenter, for example, if different applications’ data must be partitioned across different sets of servers in the same datacenter. 
>  zones 是管理部署的基本单位，zones 的集合也是数据可以被复制到的位置的集合
>  zones 可以随着新数据中心投入使用或旧数据中心被关闭而被添加到运行中的系统或从中被删除
>  zones 也是物理隔离的单元: 一个数据中心可以有一个或多个 zones，例如不同应用的数据在同一数据中心的不同服务器组之间进行划分

![[pics/Spanner-Fig1.png]]

Figure 1 illustrates the servers in a Spanner universe. A zone has one zonemaster and between one hundred and several thousand spanservers. The former assigns data to spanservers; the latter serve data to clients. 
>  Spanner universe 的示意如 Fig1 所示
>  一个 zone 有一个 zonemaster，以及一百到数千个 spanservers
>  zonemaster 负责将数据分配给 spanservers, spanservers 则负责为客户端提供数据

The per-zone location proxies are used by clients to locate the spanservers assigned to serve their data. 
>  客户端使用每个 zone 的 location proxies 来定位负责向其提供数据的 spanservers

>  类似 DNS 服务，客户端只需要知道 spanservers 的 “域名”，location proxies 将 “域名” 转化为具体的 “IP 地址”

The universe master and the placement driver are currently singletons. The universe master is primarily a console that displays status information about all the zones for interactive debugging. The placement driver handles automated movement of data across zones on the timescale of minutes. The placement driver periodically communicates with the spanservers to find data that needs to be moved, either to meet updated replication constraints or to balance load. 
>  universe master 和 placement driver 目前是单例的
>  universe master 主要作为控制台，显示所有 zones 的状态信息，以供交互式调试
>  placement driver 在分钟级的时间尺度上处理跨 zones 的数据自动移动，placement driver 会定期与 spanservers 通信，以找到需要被移动的数据 (要么是为了满足 updated replication constraints，要么是为了平衡负载)

For space reasons, we will only describe the spanserver in any detail. 

## 2.1 Spanserver Software Stack 
This section focuses on the spanserver implementation to illustrate how replication and distributed transactions have been layered onto our Bigtable-based implementation. 
>  本节介绍 spanserver 的实现，解释如何基于 Bigtable 实现 replication 和分布式事务

![[pics/Spanner-Fig2.png]]

The software stack is shown in Figure 2. At the bottom, each spanserver is responsible for between 100 and 1000 instances of a data structure called a tablet. A tablet is similar to Bigtable’s tablet abstraction, in that it implements a bag of the following mappings: 

```
(key: string, timestamp: int64) -> string
```

>  spanserver 的软件栈如 Fig2 所示
>  在底层，每个 spanserver 都负责 100 到 1000 个称为 tablet 的数据结构的实例
>  这里的 tablet 类似于 Bigtable 的 tablet 抽象，它实现了一组形式如上的映射，将 `key` 和 `timestamp` 映射到 `string` 类型的 `value`

Unlike Bigtable, Spanner assigns timestamps to data, which is an important way in which Spanner is more like a multi-version database than a key-value store. A tablet’s state is stored in set of B-tree-like files and a write-ahead log, all on a distributed file system called Colossus (the successor to the Google File System [15]). 
>  和 Bigtable 不同，Spanner 会为数据分配时间戳，也因此 Spanner 更像一个多版本数据库而不是键值存储 (不妨说是将键变为了 `(key, timestamp)` tuple)
>  一个 tablet 的状态被存储在一组类似 B-tree 的文件以及一个预写日志中，所有这些文件都存储在 Colossus 分布式文件系统中 (GFS 的后继系统)

To support replication, each spanserver implements a single Paxos state machine on top of each tablet. (An early Spanner incarnation supported multiple Paxos state machines per tablet, which allowed for more flexible replication configurations. The complexity of that design led us to abandon it.) Each state machine stores its metadata and log in its corresponding tablet. 
>  为了支持 replication，每个 spanserver 都在每个 tablet 上实现了一个 Paxos 状态机 (早期的 Spanner 设想在每个 tablet 上支持多个 Paxos 状态机，允许了更灵活的 replication 配置，但其设计过于复杂，故被放弃)
>  tablet 上的 Paxos 状态机将其元数据和日志存储在其对应的 tablet 中

>  也就是每个 spanserver 会在其负责的 100 到 1000 个 tablets 上实现一个 Paxos group，其中每个 tablet 有自己的 Paxos 状态机，整个 Paxos group 会对写入操作达成共识，进而实现了这 100 到 1000 个 tablets 上的一致状态 (tablet 存储的 `(key, timestamp)->value` 映射情况)

Our Paxos implementation supports long-lived leaders with time-based leader leases, whose length defaults to 10 seconds. The current Spanner implementation logs every Paxos write twice: once in the tablet’s log, and once in the Paxos log. This choice was made out of expediency, and we are likely to remedy this eventually. 
>  我们的 Paxos 实现使用基于时间的 leader lease 以支持 long-lived leaders，租约的时间默认为 10s
>  目前的 Spanner 实现会将每个 Paxos 写操作记录两次: 一次记录到 tablet 的 log，一次记录到 Paxos log，这一设计是出于便利性考虑，我们最终可能会改进这点

Our implementation of Paxos is pipelined, so as to improve Spanner’s throughput in the presence of WAN latencies; but writes are applied by Paxos in order (a fact on which we will depend in Section 4). 
>  我们的 Paxos 实现采用了流水线，以在存在广域网延迟的情况下提高 Spanner 的吞吐，但写操作仍由 Paxos 按顺序应用

The Paxos state machines are used to implement a consistently replicated bag of mappings. The key-value mapping state of each replica is stored in its corresponding tablet. Writes must initiate the Paxos protocol at the leader; reads access state directly from the underlying tablet at any replica that is sufficiently up-to-date. The set of replicas is collectively a Paxos group. 
>  tablet 上的 Paxos 状态机被用于实现一致的复制映射集合
>  每个副本的 key-value 映射状态存储于它对应的 tablet 中，写操作必须在 leader 处启动 Paxos 协议，读操作则直接读取任意足够 update-to-date 的副本的 tablet 的状态即可
>  副本的集合构成一个 Paxos 组

At every replica that is a leader, each spanserver implements a lock table to implement concurrency control. The lock table contains the state for two-phase locking: it maps ranges of keys to lock states. (Note that having a long-lived Paxos leader is critical to efficiently managing the lock table.) 
>  在每个作为 leader 的副本上，spanserver 实现了一个 lock table 来进行并发控制
>  lock table 包含了 two-phase locking 的状态: 它将 keys 的范围映射到锁状态，即 lock table 可以用于判断特定范围内的 keys 是否被上锁 (注意，拥有 long-lived 的 Paxos leader 对于高效管理 lock table 很重要)

In both Bigtable and Spanner, we designed for long-lived transactions (for example, for report generation, which might take on the order of minutes), which perform poorly under optimistic concurrency control in the presence of conflicts. Operations that require synchronization, such as transactional reads, acquire locks in the lock table; other operations bypass the lock table. 
>  Bigtable 和 Spanner 都针对长时间的事务进行了设计 (例如 report generation 事务需要几分钟的时间才能完成)，在乐观的并发控制下，如果这些长时间事务存在冲突，就会导致非常差的性能 (因此采用悲观的并发控制)
>  Spanner 中，需要同步的操作，例如事务式读，都需要在 lock table 中获取锁 (即需要经过 leader，以实现 Paxos 组内的同步)，其他不需要同步的操作可以绕过 lock table

At every replica that is a leader, each spanserver also implements a transaction manager to support distributed transactions. The transaction manager is used to implement a participant leader; the other replicas in the group will be referred to as participant slaves. 
>  在每个作为 leader 的副本上，每个 spanserver 还实现了一个事务管理器以支持分布式事务 (涉及多个 Paxos 组的事务)
>  事务管理器会被用于实现一个 participant leader，组内的其他副本则称为 participant slaves

If a transaction involves only one Paxos group (as is the case for most transactions), it can bypass the transaction manager, since the lock table and Paxos together provide transactionality. If a transaction involves more than one Paxos group, those groups’ leaders coordinate to perform two-phase commit. One of the participant groups is chosen as the coordinator: the participant leader of that group will be referred to as the coordinator leader, and the slaves of that group as coordinator slaves. 
>  如果事务仅涉及了一个 Paxos group (大多数事务都是这一情况)，它可以绕过事务管理器，因为 lock table 和 Paxos 相结合已经提供了事务性
>  如果事务涉及了多个 Paxos groups，则这些 Paxos groups 的 leaders 会协调执行 two-phase commit，其中一个 participant group 会被选为 coordinator: 该 group 的 participant leader 将作为 coordinator leader，该 group 的 participant slaves 将作为 coordinator slaves

The state of each transaction manager is stored in the underlying Paxos group (and therefore is replicated). 
>  每个事务管理器的状态都存储在对应的 Paxos group 中 (因此也会被复制)

## 2.2 Directories and Placement 
On top of the bag of key-value mappings, the Spanner implementation supports a bucketing abstraction called a directory, which is a set of contiguous keys that share a common prefix. (The choice of the term directory is a historical accident; a better term might be bucket.) 
>  除了键值映射的结构之外，Spanner 还支持一种称为目录的分桶抽象
>  目录本质是一组共享相同前缀的连续的 keys (因此，相较于 “目录”，更好的描述词实际上是 “桶”)

We will explain the source of that prefix in Section 2.3. Supporting directories allows applications to control the locality of their data by choosing keys carefully. 
>  Spanner 的目录抽象使得应用可以通过选择 keys 来控制其数据的位置

![[pics/Spanner-Fig3.png]]

A directory is the unit of data placement. All data in a directory has the same replication configuration. When data is moved between Paxos groups, it is moved directory by directory, as shown in Figure 3. Spanner might move a directory to shed load from a Paxos group; to put directories that are frequently accessed together into the same group; or to move a directory into a group that is closer to its accessors. Directories can be moved while client operations are ongoing. One could expect that a 50MB directory can be moved in a few seconds. 
>  目录是数据放置的单位，一个目录中的所有数据都有**相同的 replication 配置**
>  数据在 Paxos groups 之间移动时，是以目录单位进行的，如 Fig3 所示
>  Spanner 移动目录的理由一般有: 减轻某个 Paxos group 的负载、将经常一起访问的目录放入同一个 Paxos group、将目录移动到离其访问者更近的 Paxos group
>  目录可以在客户端操作进行的同时被移动，50MB 大小的目录一般可以在几秒内被移动完成

>  注意，一个 Paxos group 等价于一个 spanserver

The fact that a Paxos group may contain multiple directories implies that a Spanner tablet is different from a Bigtable tablet: the former is not necessarily a single lexicographically contiguous partition of the row space. Instead, a Spanner tablet is a container that may encapsulate multiple partitions of the row space. We made this decision so that it would be possible to colocate multiple directories that are frequently accessed together. 
>  Spanner 中的 Paxos group 可以包含多个目录，这说明了 Spanner tablet 和 Bigtable tablet 是不同的: Spanner tablet 不必要是一个按照字典序的连续行空间分区，Spanner tablet 可以是一个封装了多个行空间分区 (多个目录) 的容器
>  这样的设计是为了能够将经常一起被访问的目录放置在一起

Movedir is the background task used to move directories between Paxos groups [14]. Movedir is also used to add or remove replicas to Paxos groups [25], because Spanner does not yet support in-Paxos configuration changes. 
>  Paxos groups 之间移动目录的后台任务是 `Movedir`
>  `Movedir` 还用于向 Paxos groups 添加或移除副本，因为 Spanner 当前尚不支持 Paxos 组内的配置更改 (因此需要由组间的移动语义来间接实现)

Movedir is not implemented as a single transaction, so as to avoid blocking ongoing reads and writes on a bulky data move. Instead, movedir registers the fact that it is starting to move data and moves the data in the background. When it has moved all but a nominal amount of the data, it uses a transaction to atomically move that nominal amount and update the metadata for the two Paxos groups. 
>  `Movedir` 并未以单个事务的形式实现，以避免在数据移动时阻塞读写操作
>  `Movedir` 会先注册它开始移动数据的事实，然后在后台移动数据，当它完成几乎所有的数据迁移后，它会使用一个事务将剩余的一小部分数据原子地迁移并为两个相关的 Paxos  groups **更新元数据** (即虽然大部分的数据已经完成了迁移，但在没有原子化地完成最后的一小部分迁移之前，元数据是不会更新的)

A directory is also the smallest unit whose geographic-replication properties (or placement, for short) can be specified by an application. The design of our placement-specification language separates responsibilities for managing replication configurations. Administrators control two dimensions: the number and types of replicas, and the geographic placement of those replicas. They create a menu of named options in these two dimensions (e.g., North America, replicated 5 ways with 1 witness). An application controls how data is replicated, by tagging each database and/or individual directories with a combination of those options. 
>  目录也是应用能够指定地理复制属性 (即放置位置) 的最小单元
>  我们的放置规范语言的设计分离了管理复制配置的责任，管理者控制两个维度: 副本的数量和类型、副本的地理放置位置
>  管理者在这两个维度上创建带有选项的菜单 (示例选项: 北美，以五种方式复制，并包含一个见证节点)，应用通过为每个数据库和/或单独的目录标记这些选项的组合，来控制这些数据如何被复制

For example, an application might store each end-user’s data in its own directory, which would enable user A’s data to have three replicas in Europe, and user $B$ ’s data to have five replicas in North America. 
>  例如，应用可以将每个用户的数据存储在单独的目录中，并分别配置，这可以让用户 A 的数据在欧洲有三个副本，而用户 B 的数据在北美有 5 个副本

For expository clarity we have over-simplified. In fact, Spanner will shard a directory into multiple fragments if it grows too large. Fragments may be served from different Paxos groups (and therefore different servers). Movedir actually moves fragments, and not whole directories, between groups. 
>  上述的模型实际上存在简化，如果一个目录太大，Spanner 会将其划分为多个 fragments, fragments 可以由不同的 Paxos groups 存储 (进而由不同的服务器组存储)
>  `Movedir` 实际上是在 groups 之间移动 fragments，而不是整个目录

## 2.3 Data Model 
Spanner exposes the following set of data features to applications: a data model based on schematized semi-relational tables, a query language, and general-purpose transactions. 
>  Spanner 为应用暴露了以下的数据功能: 一个基于模式化半关系表的数据模型、一个查询语言、通用目的的事务

The move towards supporting these features was driven by many factors. The need to support schematized semi-relational tables and synchronous replication is supported by the popularity of Megastore [5]. At least 300 applications within Google use Megastore (despite its relatively low performance) because its data model is simpler to manage than Bigtable’s, and because of its support for synchronous replication across datacenters. (Bigtable only supports eventually-consistent replication across datacenters.) Examples of well-known Google applications that use Megastore are Gmail, Picasa, Calendar, Android Market, and AppEngine. 
>  支持这些功能的原因是多方面的
>  对模式化半关系表以及同步复制的支持的理由是 Megastore 的流行
>  Google 中至少有 300 个应用使用 Megastore (即便其性能相对较低)，因为其数据模型相较于 Bigtable 的数据模型更易于管理，并且也因为它支持跨数据中心的同步复制 (Bigtable 仅支持最终一致的跨数据中心复制)
>  使用 Megastore 的 Google 应用包括 Gmail, Picasa, Calendar, Android Market, App Engine

The need to support a SQL-like query language in Spanner was also clear, given the popularity of Dremel [28] as an interactive data-analysis tool. 
>  对类 SQL 查询语言的支持的理由是 Dremel 的流行 (一个交互式数据分析工具)

Finally, the lack of cross-row transactions in Bigtable led to frequent complaints; Percolator [32] was in part built to address this failing. Some authors have claimed that general two-phase commit is too expensive to support, because of the performance or availability problems that it brings [9, 10, 19]. We believe it is better to have application programmers deal with performance problems due to overuse of transactions as bottlenecks arise, rather than always coding around the lack of transactions. Running two-phase commit over Paxos mitigates the availability problems. 
>  对通用目的的事务的支持的理由是 Bigtable 无法支持跨行事务，Percolator 的构建也一部分源于这一理由
>  一些人认为，由于性能或可用性问题，支持通用的两阶段提交的成本过高；我们认为，让应用编程者在性能瓶颈出现时再处理过度使用事务的问题，要好于围绕着缺乏事务的形式而进行编程，并且在 Paxos 上运行两阶段提交缓解了可用性问题

The application data model is layered on top of the directory-bucketed key-value mappings supported by the implementation. An application creates one or more databases in a universe. Each database can contain an unlimited number of schematized tables. Tables look like relational-database tables, with rows, columns, and versioned values. We will not go into detail about the query language for Spanner. It looks like SQL with some extensions to support protocol-buffer-valued fields. 
>  应用的数据模型建立在 Spanner 所支持的**基于目录的键值映射**上
>  应用**在 universe 中**创建一个或多个数据库，每个数据库可以包含无限数量的模式化表，这些模式化表看起来像关系型数据库表，具有行、列和版本化的值
>  Spanner 的查询语言类似 SQL，并有一些拓展来支持协议缓冲区字段

Spanner’s data model is not purely relational, in that rows must have names. 
>  Spanner 的数据模型并不是完全的关系型，Spanner 的数据模型中，行必须有名字

> [!info] 关系型数据模型
> 关系型数据模型用二维表来表示实体以及实体之间的关系
> 这种二维表由行和列组成，每张表都有一个唯一的名称，称为关系名
> 例如，在一个学校数据库中，可以有一个 “学生” ，其中包含学生的学号、姓名、性别、年龄、专业等属性
> 关系型数据模型通过主键 (Primary Key) 来唯一标识表中的每一行，主键是表中**一个或多个列的组合**，其值在表中是唯一的
> 
> 主要组成部分
>   1. 关系: 关系就是一张二维表
>       表中的**每一行称为一个元组**，对应现实世界中的一个实体
>       例如，在 “学生” 表中，每一行就代表一个学生
>       表中的**每一列称为一个属性**，对应实体的某个特征
>       例如， “姓名”“性别” 等列就是学生的属性。
>       关系具有一些性质，例如: 表中的列是无序的，列的顺序可以任意交换；行也是无序的，行的顺序可以任意交换；不允许有重复的行
> 
>   2. 域: 域是一组具有相同数据类型的值的集合
>       例如，在 “学号” 这个属性的域可能是字符串类型，用于存储学号的字符组合；而 “年龄” 属性的域是整数类型，范围可能是 16 - 30 等等
>       域为属性提供了可能的取值范围，是**数据类型和取值范围的抽象**
> 
>   3. 元组和属性的完整性约束
>      元组完整性约束是指关系中的元组必须满足的规则
>      例如，在 “学生” 表中，每个学生必须有一个唯一的学号，这就要求 “学号” 属性不能有重复值
>      属性完整性约束是指关系中是属性必须满足的规则
>      例如，“性别” 属性只能取 “男” 或 “女”
> 
> 优点
>   - 结构简单清晰: 通过 SQL (结构化查询语言)，可以方便地进行复杂的数据查询和操作
>   - 理论基础坚实: 关系型数据模型建立在集合代数和谓词逻辑等数学理论基础上，这使得数据库的操作具有严格的数学依据，可以方便地进行数据的查询、插入、删除和更新等操作，并且能够保证操作的正确性和一致性
>   - 数据独立性高: 数据的**物理存储和逻辑结构是分离的**，应用程序可以只关注数据的逻辑结构，而不需要关心数据是如何存储在物理存储设备上的。例如，当数据库管理员对数据存储位置进行调整或者对存储结构优化进行时，应用程序可以不受影响。
> 

More precisely, every table is required to have an ordered set of one or more primary-key columns. This requirement is where Spanner still looks like a key-value store: the primary keys form the name for a row, and each table defines a mapping from the primary-key columns to the non-primary-key columns. A row has existence only if some value (even if it is NULL) is defined for the row’s keys. Imposing this structure is useful because it lets applications control data locality through their choices of keys. 
>  更准确地说，每个表都需要有一个包含了一个或多个 primary-key column 的有序集合 (也就是主键)
>  这一要求使得 Spanner 某种程度上看起来仍然像一个键值存储: primary keys 构成了行的名称，每个表则定义了**从 primary-key columns 到 non-primary-key columns 的映射**
>  行仅在其 primary-keys 定义了值 (即便是 NULL) 时才存在
>  这一结构使得应用可以通过对 keys 的选择来控制数据位置

>  实际上，差异在于传统关系型模型中，主键列为 NULL 值是允许的 (这种设计允许在没有主键值的情况下插入行，只要数据库允许主键列的 NULL 值)，而 Spanner 要求只有主键列不为 NULL，行才允许存在
>  因此 Spanner 中，主键列不仅用于唯一标识行，还作为行存在的必要条件，这种机制就类似于键-值存储系统，键存在才能有对应的值

![[pics/Spanner-Fig4.png]]

Figure 4 contains an example Spanner schema for storing photo metadata on a per-user, per-album basis. The schema language is similar to Megastore’s, with the additional requirement that every Spanner database must be partitioned by clients into one or more hierarchies of tables. 
>  Fig4 展示了一个按逐用户、逐相册的方式存储照片元数据的 Spanner 模式
>  该模式语言类似于 Megastore 的模式语言，但额外要求了每个 Spanner 数据库必须被客户端划分为一个或多个表的层次结构

> [!info] Schema, Schematized
> **schema** 定义了数据库中数据的组织结构和逻辑关系，包括表、视图、索引、列等的定义。“数据库模式” (database schema) 描述了整个数据库的结构蓝图，包括各个表的字段、数据类型、主键、外键以及表之间的关系等。
> **schematized** 强调按照一定的模式或架构进行组织和规划。“结构化的数据模型”(schematized data model) 表示数据模型是依据特定的模式构建的，有明确的结构和规则。

Client applications declare the hierarchies in database schemas via the `INTERLEAVE IN` declarations. The table at the top of a hierarchy is a directory table. Each row in a directory table with key $K$ , together with all of the rows in descendant tables that start with $K$ in lexicographic order, forms a directory. `ON DELETE CASCADE` says that deleting a row in the directory table deletes any associated child rows. 
>  客户端应用通过 `INTERLEAVE IN` 在数据库模式中声明表的层次
>  层次顶端的表是一个目录表，目录表中，带有 key $K$ 的行，和所有后代表中，其 key 按照字典序从 $K$ 开始的行，构成了一个字典 (这里的 $K$ 不是字母 K，是表示一个变量 $K$)
>  声明中的 `ON DELETE CASCADE` 表示删除目录表中的某一行会删除所有相关的子行

The figure also illustrates the interleaved layout for the example database: for example, `Albums(2,1)` represents the row from the `Albums` table for `user_id 2`, `album_id 1`. This interleaving of tables to form directories is significant because it allows clients to describe the locality relationships that exist between multiple tables, which is necessary for good performance in a sharded, distributed database. Without it, Spanner would not know the most important locality relationships. 
>  Fig4 中展示了示例数据库的交错布局: `Albums(2,1)` 表示了 `Albums` 表中的 `user_id 2, album_id 1` 对应的行
>  这种使用表的交错构建目录的方式非常重要，因为它允许客户端描述多个表之间存在的局部关系，这对于分片的分布式数据库的良好性能是必要的
>  没有这一点，Spanner 将无法了解最重要的局部关系

# 3 TrueTime 
This section describes the TrueTime API and sketches its implementation. We leave most of the details for another paper: our goal is to demonstrate the power of having such an API. 
>  本节描述 TrueTime API 及其实现的概要

![[pics/Spanner-Table1.png]]

Table 1 lists the methods of the API. TrueTime explicitly represents time as a `TTinterval`, which is an interval with bounded time uncertainty (unlike standard time interfaces that give clients no notion of uncertainty). The endpoints of a `TTinterval` are of type `TTstamp`. 
>  TrueTime API 的方法如 Table1 所示
>  TrueTime API 将时间显式的表示为一个具有有限不确定性的时间区间，记作 `TTinterval` (和向客户端提供没有不确定性概念的标准时间接口不同)
>  `TTinterval` 的端点的类型是 `TTstamp`

The `TT.now()` method returns a `TTinterval` that is guaranteed to contain the absolute time during which `TT.now()` was invoked. The time epoch is analogous to UNIX time with leap-second smearing. Define the instantaneous error bound as $\epsilon$ , which is half of the interval’s width, and the average error bound as ϵ. 
>  `TT.now()` 方法返回一个 `TTinterval` ，该区间保证包含 `TT.now()` 被调用时的绝对时间
>  我们定义瞬时误差界限为 $\epsilon$，它是区间宽度的一半，平均误差界限因此也是 $\epsilon$

The `TT.after()` and `TT.before()` methods are convenience wrappers around `TT.now()` . 
>  `TT.after(), TT.before()` 是 `TT.now()` 方法的包装器

Denote the absolute time of an event $e$ by the function $t_{a b s}(e)$ . In more formal terms, TrueTime guarantees that for an invocation $t t=T T.n o w()$ , $tt.earliest \leq t_{a b s}(e_{n o w})\leq t t.latest$, where $e_{n o w}$ is the invocation event. 
>  我们用函数 $t_{abs}(e)$ 表示事件 $e$ 的绝对时间
>  形式化地说，TrueTime API 保证对于一次 `TT.now()` 的调用 $tt = TT. now()$，其返回值 $tt$ 满足 $tt. earliest \le t_{abs}(e_{now}) \le tt. latest$，其中 $e_{now}$ 表示调用事件

>  也就是 `TT.now()` 并不会返回一个表示调用时刻的时间点，而是返回一个包含了调用时刻的时间区间

The underlying time references used by TrueTime are GPS and atomic clocks. TrueTime uses two forms of time reference because they have different failure modes. GPS reference-source vulnerabilities include antenna and receiver failures, local radio interference, correlated failures (e.g., design faults such as incorrect leap-second handling and spoofing), and GPS system outages. Atomic clocks can fail in ways uncorrelated to GPS and each other, and over long periods of time can drift significantly due to frequency error. 
>  TrueTime API 所使用的时间参考是 GPS 和原子钟，之所以使用两种形式的时间参考是因为二者具有不同的故障模式
>  GPS 的漏洞包括天线和接收器故障、本地射频干扰、相关故障 (例如设计缺陷，如错误处理闰秒和欺骗)，以及 GPS 系统中断
>  原子钟的故障方式则与 GPS 无关，原子钟自身可能在长时间下会由于频率误差而显著漂移

TrueTime is implemented by a set of time master machines per datacenter and a timeslave daemon per machine. The majority of masters have GPS receivers with dedicated antennas; these masters are separated physically to reduce the effects of antenna failures, radio interference, and spoofing. 
>  TrueTime 在每个数据中心由一组 time master machines 和每台机器上的一个 timeslave daemon 实现
>  大多数的 master 都配有带有专用天线的 GPS 接收器，这些 masters 会在物理上被分开，以减少天线故障、射频干扰和欺骗的影响

The remaining masters (which we refer to as Armageddon masters) are equipped with atomic clocks. An atomic clock is not that expensive: the cost of an Armageddon master is of the same order as that of a GPS master. 
>   剩余的 masters (我们称为 Armageddon masters) 配有原子钟
>   原子钟并不昂贵: Armageddon master 的成本与 GPS master 的成本处于同一量级

All masters’ time references are regularly compared against each other. Each master also cross-checks the rate at which its reference advances time against its own local clock, and evicts itself if there is substantial divergence.  Between synchronizations, Armageddon masters advertise a slowly increasing time uncertainty that is derived from conservatively applied worst-case clock drift. GPS masters advertise uncertainty that is typically close to zero. 
>  所有 time masters 的参考时间会定期互相比较
>  每个 master 也会将其参考时间的推进速率和本地的时钟进行交叉验证，如果存在显著偏差，会自动排除自身
>  在同步时，Armageddon masters 会发布一个逐渐增加的时间不确定性，该不确定性基于保守应用的最坏情况时钟偏移计算得到，GPS masters 发布的不确定性通常接近于零

Every daemon polls a variety of masters [29] to reduce vulnerability to errors from any one master. Some are GPS masters chosen from nearby datacenters; the rest are GPS masters from farther datacenters, as well as some Armageddon masters. Daemons apply a variant of Marzullo’s algorithm [27] to detect and reject liars, and synchronize the local machine clocks to the nonliars. To protect against broken local clocks, machines that exhibit frequency excursions larger than the worst-case bound derived from component specifications and operating environment are evicted. 
>  每台 time master machine 上的 timeslave daemon 会轮询多个 masters，以减少错误
>  其中一些是来自附近数据中心的 GPS masters，剩余是来自较远数据中心的 GPS masters，也有 Armageddon masters
>  daemon 会应用 Marzullo's 算法的变体来检测和拒绝谎报者，然后将本地机器时钟和非谎报者进行同步
>  为了防止本地时钟损坏，那些频率波动超出根据组件规格和运行环境推断出的最坏情况界限的机器会被驱逐

Between synchronizations, a daemon advertises a slowly increasing time uncertainty. $\epsilon$ is derived from conservatively applied worst-case local clock drift. $\epsilon$ also depends on time-master uncertainty and communication delay to the time masters. 
>  在同步时，daemon 会发布一个逐渐增加的时间不确定性 $\epsilon$ (如上面所说的)
>  $\epsilon$ 是从保守应用的最坏情况下的本地时钟漂移推导出的，$\epsilon$ 还取决于 time masters 的不确定性和与 time masters 的通信延迟

In our production environment, $\epsilon$ is typically a sawtooth function of time, varying from about 1 to 7 ms over each poll interval. $\overline{\epsilon}$ is therefore 4 ms most of the time. The daemon’s poll interval is currently 30 seconds, and the current applied drift rate is set at 200 microseconds/second, which together account for the sawtooth bounds from 0 to $6~\mathrm{ms}$ . The remaining $1~\mathrm{ms}$ comes from the communication delay to the time masters. 
>  在生产环境中，$\epsilon$ 通常是时间的锯齿波函数，在每次轮询间隔内从 1m 到 7ms 变化，故 $\bar \epsilon$ 大多数情况下是 4ms
>  daemon 的轮询间隔目前是 30s，当前应用的偏移速率则设定为每秒 200 微妙，这两个因素共同导致了从 0ms 到 6ms 的锯齿波范围
>  剩下的 1ms 来自于和 time masters 的通信延迟

Excursions from this sawtooth are possible in the presence of failures. For example, occasional time-master unavailability can cause datacenter-wide increases in $\epsilon$ . Similarly, overloaded machines and network links can result in occasional localized $\epsilon$ spikes. 
>  在出现故障的情况下，可能会偏离这个锯齿波
>  例如，偶尔的 time master 不可用情况会导致整个数据中心的 $\epsilon$ 增加，类似地，过载的机器和网络链接也可能导致偶尔的局部 $\epsilon$ 尖峰

# 4 Concurrency Control 
This section describes how TrueTime is used to guarantee the correctness properties around concurrency control, and how those properties are used to implement features such as externally consistent transactions, lock-free read-only transactions, and non-blocking reads in the past. These features enable, for example, the guarantee that a whole-database audit read at a timestamp $t$ will see exactly the effects of every transaction that has committed as of $t$ . 
>  本节描述 TrueTime 如何用于保证并发控制的正确性质，以及这些性质如何用于实现例如外部一致事务、无锁只读事务、对过去的非阻塞读这些特性
>  这些特性可以用于确保在时间戳 $t$ 进行的整个数据库的审计读取将精确看到 $t$ 时刻所有已提交事务的效果

Going forward, it will be important to distinguish writes as seen by Paxos (which we will refer to as Paxos writes unless the context is clear) from Spanner client writes. For example, two-phase commit generates a Paxos write for the prepare phase that has no corresponding Spanner client write. 
>  我们需要对 Paxos 的写操作和 Spanner client 的写操作进行区分
>  例如，两阶段提交会在准备阶段生成一个 Paxos write，而这个 Paxos write 没有对应的 Spanner client write

## 4.1 Timestamp Management 

![[pics/Spanner-Table2.png]]

Table 2 lists the types of operations that Spanner supports. The Spanner implementation supports read-write transactions, read-only transactions (predeclared snapshot-isolation transactions), and snapshot reads. Standalone writes are implemented as read-write transactions; non-snapshot standalone reads are implemented as read-only transactions. Both are internally retried (clients need not write their own retry loops). 
>  Spanner 所支持的操作见 Table2
>  Spanner 支持读写事务、只读事务 (预先声明的快照隔离事务)、快照读
>  独立的写操作通过读写事务实现，非快照的独立读操作通过只读事务实现，这两个操作都会在内部进行重试 (client 无需编写自己的重试循环)

A read-only transaction is a kind of transaction that has the performance benefits of snapshot isolation [6]. A read-only transaction must be predeclared as not having any writes; it is not simply a read-write transaction without any writes. Reads in a read-only transaction execute at a system-chosen timestamp without locking, so that incoming writes are not blocked. The execution of the reads in a read-only transaction can proceed on any replica that is sufficiently up-to-date (Section 4.1.3). 
>  Spanner 支持的只读事务遵循快照隔离语义，以提供更好的性能
>  一个只读事务必须预先声明为不包含任何写入操作，只读事务并没有实现为没有写入操作的读写事务，只读事务中的读操作会在系统选择的时间戳上执行，不需要锁，因此并发的写操作不会被阻塞
>  只读事务的读操作的执行 (实际读取数据的操作) 可以在任意足够新的副本上进行

A snapshot read is a read in the past that executes without locking. A client can either specify a timestamp for a snapshot read, or provide an upper bound on the desired timestamp’s staleness and let Spanner choose a timestamp. In either case, the execution of a snapshot read proceeds at any replica that is sufficiently up-to-date. 
>  快照读是对**过去**的一次读操作，**不需要锁**
>  client 可以为快照都指定时间戳，或者提供一个时间戳上界，让 Spanner 自行选择
>  在两种情况下，快照读的执行 (实际读取数据的操作) 可以在任意足够新的副本上进行

For both read-only transactions and snapshot reads, commit is inevitable once a timestamp has been chosen, unless the data at that timestamp has been garbage-collected. As a result, clients can avoid buffering results inside a retry loop. When a server fails, clients can internally continue the query on a different server by repeating the timestamp and the current read position. 
>  对于只读事务和快照读，一旦 client 选定好了时间戳，该操作就一定会提交 (即一定会被执行)，除非在该时间戳上的数据被垃圾收集
>  因为操作一定会提交，client 就不需要在重试循环中缓存结果，当处理操作的 server 故障后，client 只需要换一个 server，重复当前的读取位置和时间戳即可继续进行读取 (而不需要自行缓冲部分读取到的结果数据)

### 4.1.1 Paxos Leader Leases 
Spanner’s Paxos implementation uses timed leases to make leadership long-lived (10 seconds by default). A potential leader sends requests for timed lease votes; upon receiving a quorum of lease votes the leader knows it has a lease. A replica extends its lease vote implicitly on a successful write, and the leader requests lease-vote extensions if they are near expiration. 
>  Spanner 的 Paxos 实现使用定时租约使得 leadership 长期有效 (默认为 10s)
>  潜在的 leader 会发送请求获得定时租约投票，收到多数租约投票的 leader 就会获取租约
>  副本在成功写入后可以隐式地延长其租约投票，leader 在接近到期时会请求延长它收到的租约投票

Define a leader’s lease interval as starting when it discovers it has a quorum of lease votes, and as ending when it no longer has a quorum of lease votes (because some have expired). 
>  leader 的租约间隔定义为从其发现自己获得多数租约投票开始，到不再具有多数租约投票 (因为某些租约投票已经过期) 时结束

Spanner depends on the following disjointness invariant: for each Paxos group, each Paxos leader’s lease interval is disjoint from every other leader’s. Appendix A describes how this invariant is enforced. 
>  Spanner 依赖于以下不相交不变式: 
>  对于每个 Paxos group，其每个 leader 的租约间隔和所有其他 leader 的租约间隔互不重叠 (不会同时有两个 leader)
>  该不变式的实现方式见 Appendix A

The Spanner implementation permits a Paxos leader to abdicate by releasing its slaves from their lease votes. To preserve the disjointness invariant, Spanner constrains when abdication is permissible. Define $s_{m a x}$ to be the maximum timestamp used by a leader. Subsequent sections will describe when $s_{m a x}$ is advanced. Before abdicating, a leader must wait until $TT.after \left(s_{m a x}\right)$ is true. 
>  Spanner 允许 Paxos leader 通过释放其从属节点的租约投票来放弃领导权
>  为了维护不相交不变式，Spanner 对放弃领导权的时间进行了限制
>  定义 $s_{max}$ 为 leader 所使用的最大时间戳，在放弃领导权之前，leader 必须等待到 $TT. after(s_{max})$ 为 true (等待到当前的时间点一定大于 $s_{max}$)

### 4.1.2 Assigning Timestamps to RW Transactions 
Transactional reads and writes use two-phase locking. As a result, they can be assigned timestamps at any time when all locks have been acquired, but before any locks have been released. For a given transaction, Spanner assigns it the timestamp that Paxos assigns to the Paxos write that represents the transaction commit. 
>  事务式读写使用两阶段锁，因此，它们可以在获取锁时随时被分配时间戳，但是该时间戳应该在任意锁被释放之前获取
>  对于一个事务，Spanner 为其分配的时间戳为 Paxos 对表示事务提交的 Paxos write 分配的时间戳

Spanner depends on the following monotonicity invariant: within each Paxos group, Spanner assigns timestamps to Paxos writes in monotonically increasing order, even across leaders. A single leader replica can trivially assign timestamps in monotonically increasing order. This invariant is enforced across leaders by making use of the disjointness invariant: a leader must only assign timestamps within the interval of its leader lease. Note that whenever a timestamp $s$ is assigned, $s_{m a x}$ is advanced to $s$ to preserve disjointness. 
>  Spanner 依赖于以下的单调性不变式: 在每个 Paxos group 内，Spanner 以严格单调递增的顺序为 Paxos writes 分配时间戳，即便是跨 leader (即 leader 变化)
>  单个 leader 副本可以轻易地按严格递增的顺序分配时间戳，而跨 leader 的情况下，该不变式的保持则利用了不相交不变式: leader 只能在其 leader lease 的时段下分配时间戳 (故因为 leader lease 时段一定不相交，并且新 leader 的 lease 时段的时刻一定大于旧 leader 的 lease 时段，时间戳的单调性就得到了保证)
>  注意当 leader 分配了一个时间戳 $s$ 后，其 $s_{max}$ 就会被更新为 $s$，以保持不相交不变式 ($s_{max}$ 在定义上就是 leader 所使用的最大时间戳)

Spanner also enforces the following external-consistency invariant: if the start of a transaction $T_{2}$ occurs after the commit of a transaction $T_{1}$ , then the commit timestamp of $T_{2}$ must be greater than the commit timestamp of $T_{1}$ . 
>  Spanner 还维护以下的外部一致不变式:
>  如果事务 $T_2$ 的开始时刻发生在 $T_1$ 的提交时刻之后，则 $T_2$ 的提交时间戳必须大于 $T_1$ 的提交时间戳

Define the start and commit events for a transaction $T_{i}$ by $e_{i}^{s t a r t}$ and $e_{i}^{c o m m i t}$ ; and the commit timestamp of a transaction $T_{i}$ by $s_{i}$ . The invariant becomes $t_{a b s}(e_{1}^{c o m m i t})<t_{a b s}(e_{2}^{s t a r t})\Rightarrow s_{1}<s_{2}$ . 
>  我们将事务 $T_i$ 的开始和提交事件定义为 $e_i^{start}$ 和 $e_i^{commit}$，其提交时间戳定义为 $s_i$，则该不变式可以描述为 $t_{abs}(e_1^{commit}) < t_{abs}(e_2^{start}) \Rightarrow s_1 < s_2$

The protocol for executing transactions and assigning timestamps obeys two rules, which together guarantee this invariant, as shown below. 
>  为了维持该不变式，执行事务以及分配时间戳的协议遵循两条规则，如下所示

Define the arrival event of the commit request at the coordinator leader for a write $T_{i}$ to be $e_{i}^{s e r\nu e r}$ . 
>  定义写事务 $T_i$ 的提交请求到达 leader 的事件为 $e_i^{server}$

**Start** The coordinator leader for a write $T_{i}$ assigns a commit timestamp $s_{i}$ no less than the value of ${{T T.n o w()}}.latest$, computed after $e_{i}^{s e r\nu e r}$ . Note that the participant leaders do not matter here; Section 4.2.1 describes how they are involved in the implementation of the next rule. 
>  Start
>  写事务 $T_i$ 的 coordinator leader 为该事务分配提交时间戳 $s_i$，它不小于 $TT. now(). latest$，该值 ($TT.now().latest$) 是在事件 $e_i^{server}$ 发生之后计算的
>  此处与 participant leader 并不相关

**Commit Wait** The coordinator leader ensures that clients cannot see any data committed by $T_{i}$ until $T T.a f t e r(s_{i})$ is true. Commit wait ensures that $s_{i}$ is less than the absolute commit time of $T_{i}$ , or $s_{i} < t_{a b s}(e_{i}^{c o m m i t})$ . The implementation of commit wait is described in Section 4.2.1. 
>  Commit Wait
>  coordinator leader 确保在 $TT. after(s_i)$ 为真之前，不会有任何客户端看到事务 $T_i$ 提交的数据 (保证在此时，提交事件一定已经发生，才开放数据访问)
>  Commit Wait 确保了 $s_i$ 小于 $T_i$ 的绝对提交时间，即 $s_i < t_{abs}(e_i^{commit})$  (绝对提交时间即提交结果对外界可见的时间点)

Proof: 

$$
\begin{align}
s_1 &< t_{abs}(e_1^{commit})&\text{(commit wait)}\\
t_{abs}(e_1^{commit}) &< t_{abs}(e_w^{start})&\text{(assumption)}\\
t_{abs}(e_2^{start}) &\le t_{abs}(e_2^{server})&\text{(causality)}\\
t_{abs}(e_2^{start}) &\le s_2&\text{(start)}\\
s_1 &< s_2&\text{(transivity)}
\end{align}
$$

### 4.1.3 Serving Reads at a Timestamp 
The monotonicity invariant described in Section 4.1.2 allows Spanner to correctly determine whether a replica’s state is sufficiently up-to-date to satisfy a read. Every replica tracks a value called safe time $t_{s a f e}$ which is the  maximum timestamp at which a replica is up-to-date. A replica can satisfy a read at a timestamp $t$ if $t<=t_{s a f e}$ . 
>  上一节描述的单调性不变式允许 Spanner 正确判断副本的状态是否足够新以满足特定的读操作
>  每个副本都会追踪一个称为安全时间 $t_{safe}$ 的值，该值是副本的最新状态的最大时间戳
>  如果 $t <= t_{safe}$，则副本可以满足在时间戳 $t$ 处的读操作

Define $t_{s a f e}~=~m i n(t_{s a f e}^{P a x o s},t_{s a f e}^{T M})$ , where each Paxos state machine has a safe time $t_{s a f e}^{P a x o s}$ and each transaction manager has a safe time $t_{s a f e}^{T M}$ . $t_{s a f e}^{P a x o s}$ is simpler: it is the timestamp of the highest-applied Paxos write. Because timestamps increase monotonically and writes are applied in order, writes will no longer occur at or below  $t_{safe}^{Paxos}$ with respect to Paxos. 
>  $t_{safe}$ 定义为 $t_{safe} = \min(t_{safe}^{Paxos}, t_{safe}^{TM})$，其中 $t_{safe}^{Paxos}$ 表示 Paxos 状态机的安全时间，$t_{safe}^{TM}$ 表示事务管理器的安全时间
>  $t_{safe}^{Paxos}$ 等于 Paxos 状态机应用的最高 Paxos write 的时间戳，由于时间戳单调递增且写操作按顺序应用，因此与 Paxos 相关的写操作一定会在高于 $t_{safe}^{Paxos}$ 的时间戳上再发生

$t_{s a f e}^{T M}$ is $\infty$ at a replica if there are zero prepared (but not committed) transactions—that is, transactions in between the two phases of two-phase commit. (For a participant slave, $t_{s a f e}^{T M}$ actually refers to the replica’s leader’s transaction manager, whose state the slave can infer through metadata passed on Paxos writes.) If there are any such transactions, then the state affected by those transactions is indeterminate: a participant replica does not know yet whether such transactions will commit. 
>  如果存在零个已准备但未提交的事务——即处于两阶段提交的两阶段之间的事务，$t_{safe}^{TM}$ 在副本处就为无穷大 
>  (对于 participant slave，$t_{safe}^{TM}$ 实际上指副本的 leader 的事务管理器，slave 可以通过 Paxos writes 传入的元数据推断该事务管理器的状态)
>  如果存在这样的事务，则这些事务所影响的状态是不确定的: participant replica 尚不知道这些事务是否会提交

As we discuss in Section 4.2.1, the commit protocol ensures that every participant knows a lower bound on a prepared transaction’s timestamp. Every participant leader (for a group $g$ ) for a transaction $T_{i}$ assigns a prepare timestamp $s_{i,g}^{p r e p a r e}$  to its prepare record. The coordinator leader ensures that the transaction’s commit timestamp $s_{i}>=s_{i,g}^{p r e p a r e}$ $g$ for every replica in a group $g$ , over all transactions $T_{i}$ prepared at $g$ ,  $\begin{array}{r}{t_{s a f e}^{T M}=m i n_{i}(s_{i,g}^{p r e p a r e})-1}\end{array}$ over all transactions prepared at $g$ . 
>  


### 4.1.4 Assigning Timestamps to RO Transactions 
A read-only transaction executes in two phases: assign a timestamp $s_{r e a d}$ [8], and then execute the transaction’s reads as snapshot reads at $s_{r e a d}$ . The snapshot reads can execute at any replicas that are sufficiently up-to-date. 
The simple assignment of $s_{r e a d}=T T.n o w()$ ).latest, at any time after a transaction starts, preserves external consistency by an argument analogous to that presented for writes in Section 4.1.2. However, such a timestamp may require the execution of the data reads at $s_{r e a d}$ to block if $t_{s a f e}$ has not advanced sufficiently. (In addition, note that choosing a value of $s_{r e a d}$ may also advance $s_{m a x}$ to preserve disjointness.) To reduce the chances of blocking, Spanner should assign the oldest timestamp that preserves external consistency. Section 4.2.2 explains how such a timestamp can be chosen. 

## 4.2 Details 
This section explains some of the practical details of read-write transactions and read-only transactions elided earlier, as well as the implementation of a special transaction type used to implement atomic schema changes. 

It then describes some refinements of the basic schemes as described. 

### 4.2.1 Read-Write Transactions 
Like Bigtable, writes that occur in a transaction are buffered at the client until commit. As a result, reads in a transaction do not see the effects of the transaction’s writes. This design works well in Spanner because a read returns the timestamps of any data read, and uncommitted writes have not yet been assigned timestamps. 

Reads within read-write transactions use woundwait [33] to avoid deadlocks. The client issues reads to the leader replica of the appropriate group, which acquires read locks and then reads the most recent data. While a client transaction remains open, it sends keepalive messages to prevent participant leaders from timing out its transaction. When a client has completed all reads and buffered all writes, it begins two-phase commit. The client chooses a coordinator group and sends a commit message to each participant’s leader with the identity of the coordinator and any buffered writes. Having the client drive two-phase commit avoids sending data twice across wide-area links. 

A non-coordinator-participant leader first acquires write locks. It then chooses a prepare timestamp that must be larger than any timestamps it has assigned to previous transactions (to preserve monotonicity), and logs a prepare record through Paxos. Each participant then notifies the coordinator of its prepare timestamp. 

The coordinator leader also first acquires write locks, but skips the prepare phase. It chooses a timestamp for the entire transaction after hearing from all other participant leaders. The commit timestamp $s$ must be greater or equal to all prepare timestamps (to satisfy the constraints discussed in Section 4.1.3), greater than TT.now().latest at the time the coordinator received its commit message, and greater than any timestamps the leader has assigned to previous transactions (again, to preserve monotonicity). The coordinator leader then logs a commit record through Paxos (or an abort if it timed out while waiting on the other participants). 

Before allowing any coordinator replica to apply the commit record, the coordinator leader waits until TT.after $(s)$ , so as to obey the commit-wait rule described in Section 4.1.2. Because the coordinator leader chose $s$ based on ${{T T.n o w()}}$ .latest, and now waits until that timestamp is guaranteed to be in the past, the expected wait is at least $2*\overline{{\epsilon}}$ . This wait is typically overlapped with Paxos communication. After commit wait, the coordinator sends the commit timestamp to the client and all other participant leaders. Each participant leader logs the transaction’s outcome through Paxos. All participants apply at the same timestamp and then release locks. 

### 4.2.2 Read-Only Transactions 
Assigning a timestamp requires a negotiation phase between all of the Paxos groups that are involved in the reads. As a result, Spanner requires a scope expression for every read-only transaction, which is an expression that summarizes the keys that will be read by the entire transaction. Spanner automatically infers the scope for standalone queries. 
If the scope’s values are served by a single Paxos group, then the client issues the read-only transaction to that group’s leader. (The current Spanner implementation only chooses a timestamp for a read-only transaction at a Paxos leader.) That leader assigns $s_{r e a d}$ and executes the read. For a single-site read, Spanner generally does better than TT.now().latest. Define $L a s t T S()$ to be the timestamp of the last committed write at a Paxos group. If there are no prepared transactions, the assignment $s_{r e a d}=L a s t T S()$ trivially satisfies external consistency: the transaction will see the result of the last write, and therefore be ordered after it. 
If the scope’s values are served by multiple Paxos groups, there are several options. The most complicated option is to do a round of communication with all of the groups’s leaders to negotiate $s_{r e a d}$ based on LastTS(). Spanner currently implements a simpler choice. The client avoids a negotiation round, and just has its reads execute at $s_{r e a d}=T T.n o w()$ .latest (which may wait for safe time to advance). All reads in the transaction can be sent to replicas that are sufficiently up-to-date. 

### 4.2.3 Schema-Change Transactions 
TrueTime enables Spanner to support atomic schema changes. It would be infeasible to use a standard transaction, because the number of participants (the number of groups in a database) could be in the millions. Bigtable supports atomic schema changes in one datacenter, but its schema changes block all operations. 

A Spanner schema-change transaction is a generally non-blocking variant of a standard transaction. First, it is explicitly assigned a timestamp in the future, which is registered in the prepare phase. As a result, schema changes across thousands of servers can complete with minimal disruption to other concurrent activity. Second, reads and writes, which implicitly depend on the schema, synchronize with any registered schema-change timestamp at time $t\mathrm{:}$ they may proceed if their timestamps precede $t$ , but they must block behind the schemachange transaction if their timestamps are after $t$ . Without TrueTime, defining the schema change to happen at $t$ would be meaningless. 

<html><body><table><tr><td rowspan="2">replicas</td><td colspan="3">latency (ms)</td><td colspan="3">throughput (Kops/sec)</td></tr><tr><td>write</td><td>read-only transaction</td><td>snapshotread</td><td>write</td><td>read-only transaction</td><td>snapshot read</td></tr><tr><td>1D</td><td>9.4±.6</td><td>一</td><td>一</td><td>4.0±.3</td><td>一</td><td>一</td></tr><tr><td>1</td><td>14.4±1.0</td><td>1.4±.1</td><td>1.3±.1</td><td>4.1±.05</td><td>10.9±.4</td><td>13.5±.1</td></tr><tr><td>3</td><td>13.9±.6</td><td>1.3±.1</td><td>1.2±.1</td><td>2.2±.5</td><td>13.8±3.2</td><td>38.5±.3</td></tr><tr><td>5</td><td>14.4±.4</td><td>1.4±.05</td><td>1.3±.04</td><td>2.8±.3</td><td>25.3±5.2</td><td>50.0±1.1</td></tr></table></body></html>
Table 3: Operation microbenchmarks. Mean and standard deviation over 10 runs. 1D means one replica with commit wait disabled. 

### 4.2.4 Refinements 
$t_{s a f e}^{T M}$ as defined above has a weakness, in that a single prepared transaction prevents $t_{s a f e}$ from advancing. As a result, no reads can occur at later timestamps, even if the reads do not conflict with the transaction. Such false conflicts can be removed by augmenting $t_{s a f e}^{T M}$ with a fine-grained mapping from key ranges to prepared-transaction timestamps. This information can be stored in the lock table, which already maps key ranges to lock metadata. When a read arrives, it only needs to be checked against the fine-grained safe time for key ranges with which the read conflicts. 
LastTS() as defined above has a similar weakness: if a transaction has just committed, a non-conflicting readonly transaction must still be assigned $s_{r e a d}$ so as to follow that transaction. As a result, the execution of the read could be delayed. This weakness can be remedied similarly by augmenting LastTS() with a fine-grained mapping from key ranges to commit timestamps in the lock table. (We have not yet implemented this optimization.) When a read-only transaction arrives, its timestamp can be assigned by taking the maximum value of LastTS() for the key ranges with which the transaction conflicts, unless there is a conflicting prepared transaction (which can be determined from fine-grained safe time). 
tsPafxeos as defined above has a weakness in that it cannot advance in the absence of Paxos writes. That is, a snapshot read at $t$ cannot execute at Paxos groups whose last write happened before $t$ . Spanner addresses this problem by taking advantage of the disjointness of leader-lease intervals. Each Paxos leader advances $t_{s a f e}^{P a x o s}$ by keeping a threshold above which future writes’ timestamps will occur: it maintains a mapping MinNextTS $(n)$ from Paxos sequence number $n$ to the minimum timestamp that may be assigned to Paxos sequence number $n+1$ . A replica can advance $t_{s a f e}^{P a x o s}$ to $M i n N e x t T S(n)-1$ when it has applied through $n$ . 
A single leader can enforce its MinNextTS() promises easily. Because the timestamps promised by MinNextTS() lie within a leader’s lease, the disjointness invariant enforces MinNextTS() promises across leaders. If a leader wishes to advance MinNextTS() beyond the end of its leader lease, it must first extend its lease. Note that $s_{m a x}$ is always advanced to the highest value in MinNextTS() to preserve disjointness. 
A leader by default advances MinNextTS() values every 8 seconds. Thus, in the absence of prepared transactions, healthy slaves in an idle Paxos group can serve reads at timestamps greater than 8 seconds old in the worst case. A leader may also advance MinNextTS() values on demand from slaves. 

# 5 Evaluation 
We first measure Spanner’s performance with respect to replication, transactions, and availability. We then provide some data on TrueTime behavior, and a case study of our first client, F1. 
# 5.1 Microbenchmarks 
Table 3 presents some microbenchmarks for Spanner. These measurements were taken on timeshared machines: each spanserver ran on scheduling units of 4GB RAM and 4 cores (AMD Barcelona 2200MHz). Clients were run on separate machines. Each zone contained one spanserver. Clients and zones were placed in a set of datacenters with network distance of less than 1ms. (Such a layout should be commonplace: most applications do not need to distribute all of their data worldwide.) The test database was created with 50 Paxos groups with 2500 directories. Operations were standalone reads and writes of 4KB. All reads were served out of memory after a compaction, so that we are only measuring the overhead of Spanner’s call stack. In addition, one unmeasured round of reads was done first to warm any location caches. 
For the latency experiments, clients issued sufficiently few operations so as to avoid queuing at the servers. From the 1-replica experiments, commit wait is about 5ms, and Paxos latency is about 9ms. As the number of replicas increases, the latency stays roughly constant with less standard deviation because Paxos executes in parallel at a group’s replicas. As the number of replicas increases, the latency to achieve a quorum becomes less sensitive to slowness at one slave replica. 
For the throughput experiments, clients issued sufficiently many operations so as to saturate the servers’ 
Table 4: Two-phase commit scalability. Mean and standard deviations over 10 runs. 
<html><body><table><tr><td rowspan="2">participants</td><td colspan="2">latency (ms)</td></tr><tr><td>mean</td><td>99th percentile</td></tr><tr><td>1</td><td>17.0 ±1.4</td><td>75.0 ±34.9</td></tr><tr><td>2</td><td>24.5±2.5</td><td>87.6 ±35.9</td></tr><tr><td>5</td><td>31.5±6.2</td><td>104.5±52.2</td></tr><tr><td>10</td><td>30.0±3.7</td><td>95.6±25.4</td></tr><tr><td>25</td><td>35.5±5.6</td><td>100.4 ±42.7</td></tr><tr><td>50</td><td>42.7±4.1</td><td>93.7 ±22.9</td></tr><tr><td>100</td><td>71.4±7.6</td><td>131.2 ±17.6</td></tr><tr><td>200</td><td>150.5 ±11.0</td><td>320.3 ±35.1</td></tr></table></body></html> 
CPUs. Snapshot reads can execute at any up-to-date replicas, so their throughput increases almost linearly with the number of replicas. Single-read read-only transactions only execute at leaders because timestamp assignment must happen at leaders. Read-only-transaction throughput increases with the number of replicas because the number of effective spanservers increases: in the experimental setup, the number of spanservers equaled the number of replicas, and leaders were randomly distributed among the zones. Write throughput benefits from the same experimental artifact (which explains the increase in throughput from 3 to 5 replicas), but that benefit is outweighed by the linear increase in the amount of work performed per write, as the number of replicas increases. 
Table 4 demonstrates that two-phase commit can scale to a reasonable number of participants: it summarizes a set of experiments run across 3 zones, each with 25 spanservers. Scaling up to 50 participants is reasonable in both mean and 99th-percentile, and latencies start to rise noticeably at 100 participants. 
# 5.2 Availability 
Figure 5 illustrates the availability benefits of running Spanner in multiple datacenters. It shows the results of three experiments on throughput in the presence of datacenter failure, all of which are overlaid onto the same time scale. The test universe consisted of 5 zones $Z_{i}$ , each of which had 25 spanservers. The test database was sharded into 1250 Paxos groups, and 100 test clients constantly issued non-snapshot reads at an aggregrate rate of 50K reads/second. All of the leaders were explicitly placed in $Z_{1}$ . Five seconds into each test, all of the servers in one zone were killed: non-leader kills $Z_{2}$ ; leader-hard kills $Z_{1}$ ; leader-soft kills $Z_{1}$ , but it gives notifications to all of the servers that they should handoff leadership first. 
Killing $Z_{2}$ has no effect on read throughput. Killing $Z_{1}$ while giving the leaders time to handoff leadership to a different zone has a minor effect: the throughput drop is not visible in the graph, but is around $3.4\%$ . On the other hand, killing $Z_{1}$ with no warning has a severe effect: the rate of completion drops almost to 0. As leaders get re-elected, though, the throughput of the system rises to approximately 100K reads/second because of two artifacts of our experiment: there is extra capacity in the system, and operations are queued while the leader is unavailable. As a result, the throughput of the system rises before leveling off again at its steady-state rate. 
![](https://cdn-mineru.openxlab.org.cn/extract/ac27a704-8e33-4ee8-9a01-aea194c17790/7904ee511e74bf458323bf365a22d1e6041be3c88494d805b6d9957d86b12f22.jpg) 
Figure 5: Effect of killing servers on throughput. 
We can also see the effect of the fact that Paxos leader leases are set to 10 seconds. When we kill the zone, the leader-lease expiration times for the groups should be evenly distributed over the next 10 seconds. Soon after each lease from a dead leader expires, a new leader is elected. Approximately 10 seconds after the kill time, all of the groups have leaders and throughput has recovered. Shorter lease times would reduce the effect of server deaths on availability, but would require greater amounts of lease-renewal network traffic. We are in the process of designing and implementing a mechanism that will cause slaves to release Paxos leader leases upon leader failure. 
# 5.3 TrueTime 
Two questions must be answered with respect to TrueTime: is $\epsilon$ truly a bound on clock uncertainty, and how bad does $\epsilon$ get? For the former, the most serious problem would be if a local clock’s drift were greater than 200us/sec: that would break assumptions made by TrueTime. Our machine statistics show that bad CPUs are 6 times more likely than bad clocks. That is, clock issues are extremely infrequent, relative to much more serious hardware problems. As a result, we believe that TrueTime’s implementation is as trustworthy as any other piece of software upon which Spanner depends. 
Figure 6 presents TrueTime data taken at several thousand spanserver machines across datacenters up to 2200 km apart. It plots the 90th, 99th, and 99.9th percentiles of $\epsilon$ , sampled at timeslave daemons immediately after polling the time masters. This sampling elides the sawtooth in $\epsilon$ due to local-clock uncertainty, and therefore measures time-master uncertainty (which is generally 0) plus communication delay to the time masters. 
![](https://cdn-mineru.openxlab.org.cn/extract/ac27a704-8e33-4ee8-9a01-aea194c17790/50ff92a35ff6d9ecb0fca566695af9ae27c28a3a7f4125b50b80bf3742f5e80b.jpg) 
Figure 6: Distribution of TrueTime $\epsilon$ values, sampled right after timeslave daemon polls the time masters. 90th, 99th, and $99.9\mathrm{th}$ percentiles are graphed. 
The data shows that these two factors in determining the base value of $\epsilon$ are generally not a problem. However, there can be significant tail-latency issues that cause higher values of $\epsilon$ . The reduction in tail latencies beginning on March 30 were due to networking improvements that reduced transient network-link congestion. The increase in $\epsilon$ on April 13, approximately one hour in duration, resulted from the shutdown of 2 time masters at a datacenter for routine maintenance. We continue to investigate and remove causes of TrueTime spikes. 
# 5.4 F1 
Spanner started being experimentally evaluated under production workloads in early 2011, as part of a rewrite of Google’s advertising backend called F1 [35]. This backend was originally based on a MySQL database that was manually sharded many ways. The uncompressed dataset is tens of terabytes, which is small compared to many NoSQL instances, but was large enough to cause difficulties with sharded MySQL. The MySQL sharding scheme assigned each customer and all related data to a fixed shard. This layout enabled the use of indexes and complex query processing on a per-customer basis, but required some knowledge of the sharding in application business logic. Resharding this revenue-critical database as it grew in the number of customers and their data was extremely costly. The last resharding took over two years of intense effort, and involved coordination and testing across dozens of teams to minimize risk. This operation was too complex to do regularly: as a result, the team had to limit growth on the MySQL database by storing some data in external Bigtables, which compromised transactional behavior and the ability to query across all data. 
Table 5: Distribution of directory-fragment counts in F1. 
<html><body><table><tr><td>#fragments</td><td># directories</td></tr><tr><td>1</td><td>>100M</td></tr><tr><td>2-4</td><td>341</td></tr><tr><td>5-9</td><td>5336</td></tr><tr><td>10-14</td><td>232</td></tr><tr><td>15-99</td><td>34</td></tr><tr><td>100-500</td><td>7</td></tr></table></body></html> 
The F1 team chose to use Spanner for several reasons. First, Spanner removes the need to manually reshard. Second, Spanner provides synchronous replication and automatic failover. With MySQL master-slave replication, failover was difficult, and risked data loss and downtime. Third, F1 requires strong transactional semantics, which made using other NoSQL systems impractical. Application semantics requires transactions across arbitrary data, and consistent reads. The F1 team also needed secondary indexes on their data (since Spanner does not yet provide automatic support for secondary indexes), and was able to implement their own consistent global indexes using Spanner transactions. 
All application writes are now by default sent through F1 to Spanner, instead of the MySQL-based application stack. F1 has 2 replicas on the west coast of the US, and 3 on the east coast. This choice of replica sites was made to cope with outages due to potential major natural disasters, and also the choice of their frontend sites. Anecdotally, Spanner’s automatic failover has been nearly invisible to them. Although there have been unplanned cluster failures in the last few months, the most that the F1 team has had to do is update their database’s schema to tell Spanner where to preferentially place Paxos leaders, so as to keep them close to where their frontends moved. 
Spanner’s timestamp semantics made it efficient for F1 to maintain in-memory data structures computed from the database state. F1 maintains a logical history log of all changes, which is written into Spanner itself as part of every transaction. F1 takes full snapshots of data at a timestamp to initialize its data structures, and then reads incremental changes to update them. 
Table 5 illustrates the distribution of the number of fragments per directory in F1. Each directory typically corresponds to a customer in the application stack above F1. The vast majority of directories (and therefore customers) consist of only 1 fragment, which means that reads and writes to those customers’ data are guaranteed to occur on only a single server. The directories with more than 100 fragments are all tables that contain F1 secondary indexes: writes to more than a few fragments of such tables are extremely uncommon. The F1 team has only seen such behavior when they do untuned bulk data loads as transactions. 
Table 6: F1-perceived operation latencies measured over the course of 24 hours. 
<html><body><table><tr><td rowspan="2">operation</td><td colspan="2">latency (ms)</td><td rowspan="2">count</td></tr><tr><td>mean</td><td>std dev</td></tr><tr><td>all reads</td><td>8.7</td><td>376.4</td><td>21.5B</td></tr><tr><td>single-site commit</td><td>72.3</td><td>112.8</td><td>31.2M</td></tr><tr><td>multi-site commit</td><td>103.0</td><td>52.2</td><td>32.1M</td></tr></table></body></html> 
Table 6 presents Spanner operation latencies as measured from F1 servers. Replicas in the east-coast data centers are given higher priority in choosing Paxos leaders. The data in the table is measured from F1 servers in those data centers. The large standard deviation in write latencies is caused by a pretty fat tail due to lock conflicts. The even larger standard deviation in read latencies is partially due to the fact that Paxos leaders are spread across two data centers, only one of which has machines with SSDs. In addition, the measurement includes every read in the system from two datacenters: the mean and standard deviation of the bytes read were roughly 1.6KB and 119KB, respectively. 
# 6 Related Work 
Consistent replication across datacenters as a storage service has been provided by Megastore [5] and DynamoDB [3]. DynamoDB presents a key-value interface, and only replicates within a region. Spanner follows Megastore in providing a semi-relational data model, and even a similar schema language. Megastore does not achieve high performance. It is layered on top of Bigtable, which imposes high communication costs. It also does not support long-lived leaders: multiple replicas may initiate writes. All writes from different replicas necessarily conflict in the Paxos protocol, even if they do not logically conflict: throughput collapses on a Paxos group at several writes per second. Spanner provides higher performance, general-purpose transactions, and external consistency. 
Pavlo et al. [31] have compared the performance of databases and MapReduce [12]. They point to several other efforts that have been made to explore database functionality layered on distributed key-value stores [1, 4, 7, 41] as evidence that the two worlds are converging. We agree with the conclusion, but demonstrate that integrating multiple layers has its advantages: integrating concurrency control with replication reduces the cost of commit wait in Spanner, for example. 
The notion of layering transactions on top of a replicated store dates at least as far back as Gifford’s dissertation [16]. Scatter [17] is a recent DHT-based key-value store that layers transactions on top of consistent replication. Spanner focuses on providing a higher-level interface than Scatter does. Gray and Lamport [18] describe a non-blocking commit protocol based on Paxos. Their protocol incurs more messaging costs than twophase commit, which would aggravate the cost of commit over widely distributed groups. Walter [36] provides a variant of snapshot isolation that works within, but not across datacenters. In contrast, our read-only transactions provide a more natural semantics, because we support external consistency over all operations. 
There has been a spate of recent work on reducing or eliminating locking overheads. Calvin [40] eliminates concurrency control: it pre-assigns timestamps and then executes the transactions in timestamp order. HStore [39] and Granola [11] each supported their own classification of transaction types, some of which could avoid locking. None of these systems provides external consistency. Spanner addresses the contention issue by providing support for snapshot isolation. 
VoltDB [42] is a sharded in-memory database that supports master-slave replication over the wide area for disaster recovery, but not more general replication configurations. It is an example of what has been called NewSQL, which is a marketplace push to support scalable SQL [38]. A number of commercial databases implement reads in the past, such as MarkLogic [26] and Oracle’s Total Recall [30]. Lomet and Li [24] describe an implementation strategy for such a temporal database. 
Farsite derived bounds on clock uncertainty (much looser than TrueTime’s) relative to a trusted clock reference [13]: server leases in Farsite were maintained in the same way that Spanner maintains Paxos leases. Loosely synchronized clocks have been used for concurrencycontrol purposes in prior work [2, 23]. We have shown that TrueTime lets one reason about global time across sets of Paxos state machines. 
# 7 Future Work 
We have spent most of the last year working with the F1 team to transition Google’s advertising backend from MySQL to Spanner. We are actively improving its monitoring and support tools, as well as tuning its performance. In addition, we have been working on improving the functionality and performance of our backup/restore system. We are currently implementing the Spanner schema language, automatic maintenance of secondary indices, and automatic load-based resharding. Longer term, there are a couple of features that we plan to investigate. Optimistically doing reads in parallel may be a valuable strategy to pursue, but initial experiments have indicated that the right implementation is non-trivial. In addition, we plan to eventually support direct changes of Paxos configurations [22, 34]. 
Given that we expect many applications to replicate their data across datacenters that are relatively close to each other, TrueTime $\epsilon$ may noticeably affect performance. We see no insurmountable obstacle to reducing $\epsilon$ below 1ms. Time-master-query intervals can be reduced, and better clock crystals are relatively cheap. Time-master query latency could be reduced with improved networking technology, or possibly even avoided through alternate time-distribution technology. 
Finally, there are obvious areas for improvement. Although Spanner is scalable in the number of nodes, the node-local data structures have relatively poor performance on complex SQL queries, because they were designed for simple key-value accesses. Algorithms and data structures from DB literature could improve singlenode performance a great deal. Second, moving data automatically between datacenters in response to changes in client load has long been a goal of ours, but to make that goal effective, we would also need the ability to move client-application processes between datacenters in an automated, coordinated fashion. Moving processes raises the even more difficult problem of managing resource acquisition and allocation between datacenters. 
# Acknowledgements 
Many people have helped to improve this paper: our shepherd Jon Howell, who went above and beyond his responsibilities; the anonymous referees; and many Googlers: Atul Adya, Fay Chang, Frank Dabek, Sean Dorward, Bob Gruber, David Held, Nick Kline, Alex Thomson, and Joel Wein. Our management has been very supportive of both our work and of publishing this paper: Aristotle Balogh, Bill Coughran, Urs Ho¨lzle, Doron Meyer, Cos Nicolaou, Kathy Polizzi, Sridhar Ramaswany, and Shivakumar Venkataraman. 
We have built upon the work of the Bigtable and Megastore teams. The F1 team, and Jeff Shute in particular, worked closely with us in developing our data model and helped immensely in tracking down performance and correctness bugs. The Platforms team, and Luiz Barroso and Bob Felderman in particular, helped to make TrueTime happen. Finally, a lot of Googlers used to be on our team: Ken Ashcraft, Paul Cychosz, Krzysztof Ostrowski, Amir Voskoboynik, Matthew Weaver, Theo Vassilakis, and Eric Veach; or have joined our team recently: Nathan Bales, Adam Beberg, Vadim Borisov, Ken Chen, Brian Cooper, Cian Cullinan, Robert-Jan Huijsman, Milind Joshi, Andrey Khorlin, Dawid Kuroczko, Laramie Leavitt, Eric Li, Mike Mammarella, Sunil Mushran, Simon Nielsen, Ovidiu Platon, Ananth Shrinivas, Vadim Suvorov, and Marcel van der Holst. 
# 8 Conclusions 
To summarize, Spanner combines and extends on ideas from two research communities: from the database community, a familiar, easy-to-use, semi-relational interface, transactions, and an SQL-based query language; from the systems community, scalability, automatic sharding, fault tolerance, consistent replication, external consistency, and wide-area distribution. Since Spanner’s inception, we have taken more than 5 years to iterate to the current design and implementation. Part of this long iteration phase was due to a slow realization that Spanner should do more than tackle the problem of a globallyreplicated namespace, and should also focus on database features that Bigtable was missing. 
One aspect of our design stands out: the linchpin of Spanner’s feature set is TrueTime. We have shown that reifying clock uncertainty in the time API makes it possible to build distributed systems with much stronger time semantics. In addition, as the underlying system enforces tighter bounds on clock uncertainty, the overhead of the stronger semantics decreases. As a community, we should no longer depend on loosely synchronized clocks and weak time APIs in designing distributed algorithms. 
# A Paxos Leader-Lease Management 
The simplest means to ensure the disjointness of Paxosleader-lease intervals would be for a leader to issue a synchronous Paxos write of the lease interval, whenever it would be extended. A subsequent leader would read the interval and wait until that interval has passed. 
TrueTime can be used to ensure disjointness without these extra log writes. The potential $i$ th leader keeps a lower bound on the start of a lease vote from replica $r$ as $v_{i,r}^{l e a d e r}=T T.n o w()$ .earliest, computed before $e_{i,r}^{s e n d}$ (defined as when the lease request is sent by the leader). Each replica $r$ grants a lease at lease $e_{i,r}^{g r a n t}$ , which happens after $e_{i,r}^{r e c e i\nu e}$ (when the replica receives a lease request); the lease ends at $t_{i,r}^{e n d}=T T.n o w().l a t e s t+10$ computed after eri,erceive. A replica r obeys the singlevote rule: it will not grant another lease vote until TT.after $(t_{i,r}^{e n d})$ is true. To enforce this rule across different incarnations of $r$ , Spanner logs a lease vote at the granting replica before granting the lease; this log write can be piggybacked upon existing Paxos-protocol log writes. 
When the $i$ th leader receives a quorum of votes (event $e_{i}^{q u o r u m})$ , it computes its lease interval as $\begin{array}{r c l}{l e a s e_{i}}&{=}&{\left[T T.n o w().l a t e s t,m i n_{r}(v_{i,r}^{l e a d e r})\ +\ 10\right]}\end{array}$ . The lease is deemed to have expired at the leader when TT.before $(m i n_{r}(v_{i,r}^{l e a d e r})+10)$ is false. To prove disjointness, we make use of the fact that the $i$ th and $(i+1)\mathrm{th}$ leaders must have one replica in common in their quorums. Call that replica $r0$ . Proof: 
$$
\begin{array}{r l r}{l e a s e e_{i}\mathrm{,~}e a d=m i n_{\mathrm{-}}(v_{i,\mathrm{r}}^{t a d e r})+10}&{\mathrm{(by~definition)}}\\ {m i n_{\mathrm{r}}(v_{i,\mathrm{r}}^{t a d e r})+10\leq v_{i,\mathrm{r},\mathrm{of~}}^{t a n}+10}&{\mathrm{(minit)}}\\ {v_{i,\mathrm{r}}^{t a d e r}+10\leq t_{a b}(v_{i,\mathrm{r}}^{e n d e r})+10}&{\mathrm{(by~definition)}}\\ {t_{a b}(e_{i,\mathrm{r}}^{e x})+10\leq t_{a b}(v_{i,\mathrm{r}}^{e x})+10}&{\mathrm{(causality)}}\\ {t_{a b}(e_{i,\mathrm{r}}^{e x e i\mathrm{,~}}v_{i})+10\leq t_{a b}^{e n d}}&{\mathrm{(by~definition)}}\\ {t_{a b}(e_{i,\mathrm{r}}^{t r a d e r})}&{+t_{a b}(e_{i,\mathrm{r}}^{e x a d e r})}&{\mathrm{(bingle-volitson)}}\\ {t_{a b,\mathrm{r}}^{t r a d e r}}&{\leq t_{a b}(e_{i+1}^{e x},v_{i})}&{\mathrm{(causlity)}}\\ {t_{a b}(e_{i+1,\mathrm{r}}^{e x a d e r})\leq t_{a b}(e_{i+1}^{e x a m e})}&{\mathrm{(causality)}}\\ {t_{a b}(e_{i+1}^{e x a d e r})\leq t_{a b}(e_{i+1}^{e x a d e r})}&{\mathrm{(by~definition)}}\end{array}
$$ 