# Abstract 
In this paper, we describe ZooKeeper, a service for coordinating processes of distributed applications. Since ZooKeeper is part of critical infrastructure, ZooKeeper aims to provide a simple and high performance kernel for building more complex coordination primitives at the client. 
>  ZooKeeper 是一个用于协调分布式应用程序进程的服务，它是关键基础设施的一部分，旨在为客户端构建更复杂的协调原语提供简单且高性能的 kernel

It incorporates elements from group messaging, shared registers, and distributed lock services in a replicated, centralized service. The interface exposed by ZooKeeper has the wait-free aspects of shared registers with an event-driven mechanism similar to cache invalidations of distributed file systems to provide a simple, yet powerful coordination service. 
>  ZooKeeper 将组消息传递、共享注册表、分布式锁服务整合入一个复制的、集中的服务中
>  ZooKeeper 提供的接口结合了共享注册表的无等待特性以及类似于分布式文件系统缓存失效机制的事件驱动机制，从而提供了一个简单但功能强大的协调服务

The ZooKeeper interface enables a high-performance service implementation. In addition to the wait-free property, ZooKeeper provides a per client guarantee of FIFO execution of requests and linearizability for all requests that change the ZooKeeper state. 
>  ZooKeeper 接口能够实现高性能的服务实现，除了无等待特性外，ZooKeeper 还为每个客户端提供了请求的 FIFO 执行保证，并对所有改变 ZooKeeper 状态的请求提供线性一致性

These design decisions enable the implementation of a high performance processing pipeline with read requests being satisfied by local servers. We show for the target workloads, 2:1 to 100:1 read to write ratio, that ZooKeeper can handle tens to hundreds of thousands of transactions per second. This performance allows ZooKeeper to be used extensively by client applications. 
>  这些设计决策助于实现一个高性能的处理流水线，其中读取请求可以通过本地服务器实现
>  对于读写比率从 2:1 到 100:1 的目标工作负载，ZooKeeper 可以每秒处理数万到数十万次事务，这种性能使得 ZooKeeper 能够给被客户端应用程序广泛使用

# 1 Introduction 
Large-scale distributed applications require different forms of coordination. Configuration is one of the most basic forms of coordination. In its simplest form, configuration is just a list of operational parameters for the system processes, whereas more sophisticated systems have dynamic configuration parameters. Group membership and leader election are also common in distributed systems: often processes need to know which other processes are alive and what those processes are in charge of. Locks constitute a powerful coordination primitive that implement mutually exclusive access to critical resources. 
>  大规模分布式应用需要不同形式的 coordination
>  configuration 是最为基本的 coordination 形式之一，在最简单的形式下，configuration 只是系统进程的操作参数列表，而更复杂的系统则具有动态 configuration 参数
>  group membership 和 leader election 在分布式系统中也很常见: 通常进程需要知道哪些其他进程是活跃的以及这些进程负责什么
>  锁是一类 coordination primitive，用于实现对关键资源的互斥访问

One approach to coordination is to develop services for each of the different coordination needs. For example, Amazon Simple Queue Service [3] focuses specifically on queuing. Other services have been developed specifically for leader election [25] and configuration [27]. Services that implement more powerful primitives can be used to implement less powerful ones. For example, Chubby [6] is a locking service with strong synchronization guarantees. Locks can then be used to implement leader election, group membership, etc. 
>  一种实现 coordination 的方法是为不同的协调需求开发相应服务
>  例如 Amazon Simple Queue Service 专门针对队列功能，以及其他针对 leader election, configuration 的专门服务
>  实现了更强大原语的服务可以用于实现较弱功能的服务，例如 Chubby 是一个具有强同步保证的锁服务，我们可以利用锁服务实现 leader election, group membership 等功能

When designing our coordination service, we moved away from implementing specific primitives on the server side, and instead we opted for exposing an API that enables application developers to implement their own primitives. Such a choice led to the implementation of a coordination kernel that enables new primitives without requiring changes to the service core. This approach enables multiple forms of coordination adapted to the requirements of applications, instead of constraining developers to a fixed set of primitives. 
>  在设计我们的 coordination service 时，我们放弃了在服务端实现特定原语的做法，而是选择提供一个 API，让应用开发者可以自行实现他们的原语
>  这驱使我们实现一个 coordination kernel，基于该 kernel，能够在不更改服务核心的情况下实现新的原语，这使得我们可以基于其开发适应应用程序需求的多种形式的 coordination，而不是限制开发者使用一组固定的原语

>  ZooKeeper 实现了 coordination kernel，并提供了 API，分布式应用可以基于 API，利用 kernel 实现特定的 coordination 服务

When designing the API of ZooKeeper, we moved away from blocking primitives, such as locks. Blocking primitives for a coordination service can cause, among other problems, slow or faulty clients to impact negatively the performance of faster clients. The implementation of the service itself becomes more complicated if processing requests depends on responses and failure detection of other clients. 
>  在设计 ZooKeeper 的 API 时，我们放弃了阻塞原语，例如锁
>  blocking primitives 对于一个 coordination service 导致的主要问题是慢的或故障的客户端会影响快的客户端的性能 (因为要等待)
>  (如果实现了 blocking primitives，则为了避免慢的客户端影响性能，需要对客户端进行检测，即 ZooKeeper 还应该需要提供这部分的 API，而) 如果对请求的处理依赖于其他客户端的响应或者对其他客户端的故障检测，coordination service 的实现本身会变得更加复杂 (因此，ZooKeeper 的 API 不提供阻塞原语)

Our system, Zookeeper, hence implements an API that manipulates simple wait-free data objects organized hierarchically as in file systems. In fact, the ZooKeeper API resembles the one of any other file system, and looking at just the API signatures, ZooKeeper seems to be Chubby without the lock methods, open, and close. Implementing wait-free data objects, however, differentiates ZooKeeper significantly from systems based on blocking primitives such as locks. 
>  因此 ZooKeeper 实现了一个管理简单的无等待数据对象的 API (ZooKeeper 提供的 API 是用于管理无等待数据对象的，对于这类数据对象的管理可以被抽象为协调服务的实现)，这些数据对象以类似于文件系统层次的形式组织
>  实际上 ZooKeeper API 和文件系统 API 是类似的，从其 API 签名 (函数声明) 来看，ZooKeeper 看起来就像没有 lock methods, open, close 的 Chubby
>  但因为实现了无等待数据对象，ZooKeeper 和其他基于阻塞原语 (如锁) 的系统是显著不同的

Although the wait-free property is important for performance and fault tolerance, it is not sufficient for coordination. We have also to provide order guarantees for operations. In particular, we have found that guaranteeing both FIFO client ordering of all operations and linearizable writes enables an efficient implementation of the service and it is sufficient to implement coordination primitives of interest to our applications. In fact, we can implement consensus for any number of processes with our API, and according to the hierarchy of Herlihy, ZooKeeper implements a universal object [14]. 
>  尽管无等待属性对于容错和性能非常重要，但仍不足以支持协调
>  故我们还需提供操作的顺序保证，特别地，我们发现同时保证所有操作的 FIFO 客户端顺序和线性可写性使得 coordination service 可以被高效实现，并且足以实现对于应用有效的 coordination primitives
>  (ZooKeeper 提供的无等待数据对象的管理 API，在所有操作的 FIFO 保证下，就可以用于实现许多协调服务)
>  实际上，我们可以使用我们的 API 为任意数量的进程实现共识

The ZooKeeper service comprises an ensemble of servers that use replication to achieve high availability and performance. Its high performance enables applications comprising a large number of processes to use such a coordination kernel to manage all aspects of coordination. 
>  ZooKeeper 服务由一组使用 replication 的服务器组成，以达到高可用性和高性能
>  其高性能使得包含了大量进程的应用也可以使用 ZooKeeper 的 coordination kernel 管理 coordination 的各种方面

We were able to implement ZooKeeper using a simple pipelined architecture that allows us to have hundreds or thousands of requests outstanding while still achieving low latency. Such a pipeline naturally enables the execution of operations from a single client in FIFO order. Guaranteeing FIFO client order enables clients to submit operations asynchronously. With asynchronous operations, a client is able to have multiple outstanding operations at a time. This feature is desirable, for example, when a new client becomes a leader and it has to manipulate metadata and update it accordingly. Without the possibility of multiple outstanding operations, the time of initialization can be of the order of seconds instead of sub-second. 
>  我们通过一个简单的流水线架构实现 ZooKeeper，该架构允许我们同时处理数百/数千个请求的同时保持低延迟
>  这种流水线架构自然支持以 FIFO 的顺序执行来自单个 client 的操作，FIFO 的保证使得 clients 可以异步地提交操作 (不需要等待上一个操作完成就提交下一个操作)，因此一个 client 可以在同一时间有多个未完成的操作
>  这是我们想要的特性，例如，当新的 client 成为 leader 并且需要操作元数据并相应地更新时，如果 clients 不能同时提交多个操作，其初始化时间可能会达到数秒
>  (client 提交的操作都是 ZooKeeper API 提供的对无等待对象的管理操作，ZooKeeper 确保它们是 FIFO 的，并且支持异步提交)

To guarantee that update operations satisfy linearizability, we implement a leader-based atomic broadcast protocol [23], called Zab [24]. A typical workload of a ZooKeeper application, however, is dominated by read operations and it becomes desirable to scale read throughput. In ZooKeeper, servers process read operations locally, and we do not use Zab to totally order them. 
>  为了确保更新操作满足线性一致性，我们实现了一个基于 leader 的原子广播协议，称为 Zab
>  然而，ZooKeeper 应用的典型工作负载主要由读取操作组成，因此提高读取吞吐量是必要的。在 ZooKeeper 中，服务器会本地处理读操作，故我们不使用 Zab 来对它们进行排序
>  (ZooKeeper 提供的 API 中，对无等待对象的操作有 update 也有 read，其中基于 leader 的原子广播协议 Zab 确保了 update 操作的一致性，而 read 操作不经过 leader，服务器本地直接处理)

>  也就是说，update 操作都需要交由 leader 处理，leader 会通过共识协议，确保 update 操作达成共识，故 update 操作是满足 linearizability，即新的 write 操作应该是在上一次 write 的结果上执行，故 update operations 具有全序关系
>  但是，read 操作不会由 leader 发起共识，故 read 操作可能不会读取到最新的 write 结果
>  容易看出，update 操作的 linearizability 是确保 coordination 的必要条件

Caching data on the client side is an important technique to increase the performance of reads. For example, it is useful for a process to cache the identifier of the current leader instead of probing ZooKeeper every time it needs to know the leader. ZooKeeper uses a watch mechanism to enable clients to cache data without managing the client cache directly. With this mechanism, a client can watch for an update to a given data object, and receive a notification upon an update. 
>  在 client side 缓存数据是提高读取性能的重要技术
>  例如进程可以缓存当前 leader 的标识符，进而不需要每次向 ZooKeeper 查询，ZooKeeper 使用观察机制，其中 client 能够缓存数据，而 ZooKeeper 不会直接管理 client 缓存
>  该机制下，cilent 可以观察某个给定数据对象的更新，在它出现更新时收到通知
>  (ZooKeeper 允许 client 缓存 ZooKeeper 保存的某个数据对象的数据，并且在该数据对象被更新时通知 client，避免缓存不一致性的发生)

Chubby manages the client cache directly. It blocks updates to invalidate the caches of all clients caching the data being changed. Under this design, if any of these clients is slow or faulty, the update is delayed. Chubby uses leases to prevent a faulty client from blocking the system indefinitely. Leases, however, only bound the impact of slow or faulty clients, whereas ZooKeeper watches avoid the problem altogether. 
>  Chubby 则直接管理 client 缓存，它在更新 client 缓存时会阻塞，故如果任何一个需要更新的 client 响应缓慢或出现故障，更新就被会延迟
>  Chubby 使用租约来防止故障客户端无限期地阻塞系统，但这也只能限制慢的或者故障的客户端的影响，而 ZooKeeper 的观察机制则完全避免了这个问题
>  (ZooKeeper 中，client 的缓存由 client 自行更新，ZooKeeper 只负责提醒 client 什么时候更新；Chubby 中，client 的缓存由系统负责更新，并使用租约系统避免慢的 client 阻塞系统)

In this paper we discuss our design and implementation of ZooKeeper. With ZooKeeper, we are able to implement all coordination primitives that our applications require, even though only writes are linearizable. To validate our approach we show how we implement some coordination primitives with ZooKeeper. 
>  在本文中，我们讨论了 ZooKeeper 的设计与实现，通过 ZooKeeper，我们能够实现我们的应用需要的所有 coordination primitives，即便只有写操作是 linearizable
>  为了验证我们的方法，我们展示了如何利用 ZooKeeper 来实现某些 coordination primitives

To summarize, in this paper our main contributions are: 

**Coordination kernel:** We propose a wait-free coordination service with relaxed consistency guarantees for use in distributed systems. In particular, we describe our design and implementation of a coordination kernel, which we have used in many critical applications to implement various coordination techniques. 

**Coordination recipes:** We show how ZooKeeper can be used to build higher level coordination primitives, even blocking and strongly consistent primitives, that are often used in distributed applications. 

**Experience with Coordination:** We share some of the ways that we use ZooKeeper and evaluate its performance. 

>  本文的主要贡献有
>  - Coordination Kernel: 我们提出一种 wait-free coordination service，它提供了松弛的一致性保证，我们描述了 coordination kernel 的设计与实现，基于该 kernel 可以实现多种 coordination 技术
>  - Coordination Recipes: ZooKeeper 可以用于构建更高级的 coordination primitives，包括了阻塞原语和强一致原语
>  - Experience with Coordination: 我们分享了使用 ZooKeeper 的方式，并评估了其性能

# 2 The ZooKeeper service 
Clients submit requests to ZooKeeper through a client API using a ZooKeeper client library. In addition to exposing the ZooKeeper service interface through the client API, the client library also manages the network connections between the client and ZooKeeper servers. 
>  客户端通过一个使用 ZooKeeper client library 的 client API 向 ZooKeeper 提交请求
>  为了通过 client API 暴露 ZooKeeper 服务接口，client library 还管理了客户端和 ZooKeeper servers 之间的网络连接
>  (client API 为 client 暴露了 ZooKeeper 的服务接口, client 通过 client API 使用 ZooKeeper 服务)

In this section, we first provide a high-level view of the ZooKeeper service. We then discuss the API that clients use to interact with ZooKeeper.

**Terminology.** In this paper, we use client to denote a user of the ZooKeeper service, server to denote a process providing the ZooKeeper service, and znode to denote an in-memory data node in the ZooKeeper data, which is organized in a hierarchical namespace referred to as the data tree. We also use the terms update and write to refer to any operation that modifies the state of the data tree. Clients establish a session when they connect to ZooKeeper and obtain a session handle through which they issue requests. 
>  我们使用 client 表示 ZooKeeper 服务的用户; server 表示提供 ZooKeeper 服务的进程; znode 表示 ZooKeeper 数据中的一个内存数据节点, znode 在一个称为 data tree 的层次命名空间中组织; update 和 write 指代任意修改 data tree 状态的操作; client 在连接到 ZooKeeper 时会建立一个 session，并通过 session handle 发送请求

## 2.1 Service overview 
ZooKeeper provides to its clients the abstraction of a set of data nodes (znodes), organized according to a hierarchical name space. The znodes in this hierarchy are data objects that clients manipulate through the ZooKeeper API. 
>  ZooKeeper 向其 clients 提供的是一组数据节点 (znodes) 的抽象，znodes 按照层次化的命名空间组织
>  每个 znode 表示 client 通过 ZooKeeper API 管理的数据对象

Hierarchical name spaces are commonly used in file systems. It is a desirable way of organizing data objects, since users are used to this abstraction and it enables better organization of application meta-data. To refer to a given znode, we use the standard UNIX notation for file system paths. For example, we use $/{A}/{B}/{C}$ to denote the path to znode C, where C has $\mathrm{B}$ as its parent and B has A as its parent. All znodes store data, and all znodes, except for ephemeral znodes, can have children. 
>  层次化命名空间常见于文件系统，故用户本身习惯于这一抽象，进而也是组织数据对象的理想方式，也更便于组织应用元数据
>  对特定 znode 的引用的形式为 `/A/B/C` ，表示 znode C 的路径，C 的父节点是 B，B 的父节点是 A
>  所有的 znodes 都存储数据，并且除了临时 znodes 以外，都有子节点

There are two types of znodes that a client can create: 

**Regular:** Clients manipulate regular znodes by creating and deleting them explicitly; 

**Ephemeral:** Clients create such znodes, and they either delete them explicitly, or let the system remove them automatically when the session that creates them terminates (deliberately or due to a failure). 

>  client 可以创建两类 znode
>  - Regular: client 通过显式创建、删除来管理 regular znodes
>  - Ephemeral: client 显式创建 ephemeral znodes，这类 nodes 可以被显式删除，也可以在创建它们的 session 终止后 (无论是正常终止还是因为故障) 自动被系统删除

Additionally, when creating a new znode, a client can set a sequential flag. Nodes created with the sequential flag set have the value of a monotonically increasing counter appended to its name. If $n$ is the new znode and $p$ is the parent znode, then the sequence value of $n$ is never smaller than the value in the name of any other sequential znode ever created under $\mathsf{p}$ . 
>  创建新 znode 时，client 可以设定一个 sequential flag，这类 znode 的名字后将有一个单调递增的计数器
>  如果 n 是新创建的 znode，而 p 是其父 znode，则 n 的序列值永远不会小于在 p 下曾经创建过的任何其他的 sequential znode 名称中的序列值

ZooKeeper implements watches to allow clients to receive timely notifications of changes without requiring polling. When a client issues a read operation with a watch flag set, the operation completes as normal except that the server promises to notify the client when the information returned has changed. 
>  ZooKeeper 实现了 watches，使得 clients 无需进行询问即可收到变更通知
>  当 client 发出了设定了 watch flag 的读请求时，该操作除了正常完成外，server 还会承诺如果返回的信息发生更改时 (新数据写入)，会及时通知 client

Watches are one-time triggers associated with a session; they are unregistered once triggered or the session closes. Watches indicate that a change has happened, but do not provide the change. For example, if a client issues a `getData(''foo'', true)`  before “/foo” is changed twice, the client will get one watch event telling the client that data for “/foo” has changed. Session events, such as connection loss events, are also sent to watch callbacks so that clients know that watch events may be delayed. 
>  watches 是一次性触发器，且与 session 关联，如果 session 关闭，或者 watches 被触发了一次，它们就会被注销
>  watches 仅表示发生了变更，但不提供具体内容
>  例如 client 在 `/foo` 发生两次变化前发出 `getData(''foo'', true)` ，client 只会收到一次 watch event，告知 `/foo` 的数据已变更
>  会话事件，例如连接丢失事件也会发送给 watch callbacks (也会告知 client)，以便 client 知道 watch event 可能会延迟

**Data model.** The data model of ZooKeeper is essentially a file system with a simplified API and only full data reads and writes, or a key/value table with hierarchical keys. The hierarchal namespace is useful for allocating subtrees for the namespace of different applications and for setting access rights to those subtrees. We also exploit the concept of directories on the client side to build higher level primitives as we will see in section 2.4. 
>  ZooKeeper 的数据模型本质是带有简化 API 的文件系统，仅支持完整的数据读写，或者说本质是一个具有分层键的键值表
>  分层命名空间方便了为不同应用的命名空间分配子树以及为这些子树设置访问权限，我们在 client side 也采用了目录的概念来构建更高级的原语

![[pics/ZooKeeper-Fig1.png]]

Unlike files in file systems, znodes are not designed for general data storage. Instead, znodes map to abstractions of the client application, typically corresponding to meta-data used for coordination purposes. To illustrate, in Figure 1 we have two subtrees, one for Application 1 (/app1) and another for Application 2 (/app2). The subtree for Application 1 implements a simple group membership protocol: each client process $p_{i}$ creates a znode $\mathrm{p}_{i}$ under /app1, which persists as long as the process is running. 
>  和文件系统的文件不同，znodes 并不用于一般数据存储
>  znodes 映射至 client 应用的抽象，**通常对应于用于 coordination 的元数据**
>  例如，Figure 1 中，子树 `/app1` 对应 app1，子树 `/app2` 对应 app2，app1 的子树实现了一个简单的 group membership protocol: 每个 client 进程 $p_i$ 在树中创建 znode $p_i$，只要进程在运行，该 znode 就持续存在

Although znodes have not been designed for general data storage, ZooKeeper does allow clients to store some information that can be used for meta-data or configuration in a distributed computation. For example, in a leader-based application, it is useful for an application server that is just starting to learn which other server is currently the leader. To accomplish this goal, we can have the current leader write this information in a known location in the znode space. Znodes also have associated meta-data with time stamps and version counters, which allow clients to track changes to znodes and execute conditional updates based on the version of the znode. 
>  虽然 znode 设计上并不用于一般数据存储，但 ZooKeeper 允许 clients 使用 znodes 存储元数据或配置信息
>  例如，在基于 leader 的应用中，刚启动的应用服务器需要知道哪个服务器是 leader，为此，可以让当前的 leader 将其信息写入 znode 空间中的特定位置
>  znode 也关联了时间戳和版本计数器元数据，client 可以借助这些元数据追踪 znodes 的变化，并根据 znode 的版本执行条件更新

**Sessions.** A client connects to ZooKeeper and initiates a session. Sessions have an associated timeout. ZooKeeper considers a client faulty if it does not receive anything from its session for more than that timeout. A session ends when clients explicitly close a session handle or ZooKeeper detects that a clients is faulty. Within a session, a client observes a succession of state changes that reflect the execution of its operations. Sessions enable a client to move transparently from one server to another within a ZooKeeper ensemble, and hence persist across ZooKeeper servers. 
>  client 连接到 ZooKeeper，并发起一个 session
>  session 有相关的 timeout，如果 ZooKeeper 在 session 中的 timeout 时段内没有从 client 收到任何信息，就认为 client 故障
>  session 在 client 显式关闭 session handle 或者 ZooKeeper 检测到 client 故障时结束
>  在 session 中，client 通过观察连续的状态变化得知其操作的执行结果
>  session 使得 client 能够在 ZooKeeper 集群中的不同服务器 (replicated state machines) 透明地切换 (即和 client 沟通的 server 换成另一个)，进而在 ZooKeeper servers 之间持久化 (一个 server 故障了，session 不会结束，而是切换另一个 server 为该 session 提供服务)

## 2.2 Client API 
We present below a relevant subset of the ZooKeeper API, and discuss the semantics of each request. 
>  我们展示一部分相关的 ZooKeeper API 

`create(path, data, flags)`: Creates a znode with path name `path`, stores `data[]` in it, and returns the name of the new znode. `flags` enables a client to select the type of znode: regular, ephemeral, and set the sequential flag; 

`delete(path, version)`: Deletes the znode path if that znode is at the expected version; 

`exists(path, watch)`: Returns true if the znode with path name `path` exists, and returns false otherwise. The `watch` flag enables a client to set a watch on the znode; 

>  `create(path, data, flags)` : 以给定路径名创建 znode，并存储 `data[]`，返回 znode 的名称，`flags` 用于选择 znode 的类型: regular, ephemeral，以及用于设定 sequential flag
>  `delete(path, version)`: 如果 znode 处于给定版本，删除指定 znode
>  `exists(path, watch)`: 如果 znode 存在，返回 true，否则 false，`watch` 用于给 znode 设定 watch

`getData(path, watch)`: Returns the data and meta-data, such as version information, associated with the znode. The `watch` flag works in the same way as it does for `exists()`, except that ZooKeeper does not set the watch if the znode does not exist; 

`setData(path, data, version)`: Writes `data[]` to znode path if the version number is the current version of the znode; 

>  `getData(path, watch)`: 返回 znode 的数据和元数据，`watch` 用于设定 watch，和 `exists(path, watch)` 中的差异在于，在这里 ZooKeeper 不会在 `path` 不存在时设定 watch (`exists()` 则会)
>  `setData(path, data, version)`: 如果 znode 的版本号匹配，写入 `data[]`

`getChildren(path, watch)`: Returns the set of names of the children of a znode; 

`sync(path)`: Waits for all updates pending at the start of the operation to propagate to the server that the client is connected to. The `path` is currently ignored. 

>  `getChildren(path, watch)`: 返回 znode 的子节点名称
>  `sync(path)`: 等待 `sync(path)` 操作前 client 提交的所有未完成的操作被 client 连接的 server 完成，`path` 暂时没有用

All methods have both a synchronous and an asynchronous version available through the API. An application uses the synchronous API when it needs to execute a single ZooKeeper operation and it has no concurrent tasks to execute, so it makes the necessary ZooKeeper call and blocks. The asynchronous API, however, enables an application to have both multiple outstanding ZooKeeper operations and other tasks executed in parallel. The ZooKeeper client guarantees that the corresponding callbacks for each operation are invoked in order. 
>  API 为所有方法都提供了同步和异步版本
>  同步 API 在 client 调用后会阻塞，异步 API 允许 client 同时提交多个任务
>  ZooKeeper client 保证每个操作的对应的 callbacks 会按照 (提交) 顺序执行

Note that ZooKeeper does not use handles to access znodes. Each request instead includes the full path of the znode being operated on. Not only does this choice simplifies the API (no open() or close() methods), but it also eliminates extra state that the server would need to maintain. 
>  ZooKeeper 不使用 handle 访问 znodes，每个请求都包含了目标 znode 的完整路径
>  这简化了 API (没有提供 `open(), close()` 方法，这类方法通常接收一个路径，返回一个 handle)，也避免 server 维护一些额外状态

Each of the update methods take an expected version number, which enables the implementation of conditional updates. If the actual version number of the znode does not match the expected version number the update fails with an unexpected version error. If the version number is $-1$ , it does not perform version checking. 
>  所有的 update 方法 (即修改数据的方法) 都接收期望的版本号，版本号的设计用于实现条件更新
>  如果 znode 的真实版本号不匹配，更新会失败，如果 znode 的版本是 -1，则不会执行版本检查 (直接更新)

## 2.3 ZooKeeper guarantees 
ZooKeeper has two basic ordering guarantees: 

**Linearizable writes:** all requests that update the state of ZooKeeper are serializable and respect precedence; 

**FIFO client order:** all requests from a given client are executed in the order that they were sent by the client. 

>  ZooKeeper 有两个基本顺序保证
>  Linearizable writes: 所有更新 ZooKeeper 状态的请求都会串行化，并且遵守优先级
>  FIFO client order: 来自给定 client 的所有请求都按照 client 发送它们的顺序执行

Note that our definition of linearizability is different from the one originally proposed by Herlihy [15], and we call it A-linearizability (asynchronous linearizability). In the original definition of linearizability by Herlihy, a client is only able to have one outstanding operation at a time (a client is one thread). In ours, we allow a client to have multiple outstanding operations, and consequently we can choose to guarantee no specific order for outstanding operations of the same client or to guarantee FIFO order. We choose the latter for our property. 
>  我们对 linearizability 的定义和 Herlihy 最初提出的定义不同，我们称其为 A-linearizability (asynchronous linearizability)
>  在 Herlihy 的最初的 linearizability 定义中，一个 client 一次只能执行一个未完成的操作 (一个 client 即一个 thread)；在我们的定义中，我们允许一个 client 同时有多个未完成的操作，并且这些操作的完成顺序可以没有保证，或者保证 FIFO 顺序 (我们选择了后者，即 FIFO 保证)

It is important to observe that all results that hold for linearizable objects also hold for A-linearizable objects because a system that satisfies A-linearizability also satisfies linearizability.  
>  所有对于 linearizable 对象成立的结果对于 A-linearizable 对象也成立，因为满足 A-linearizability 的系统也满足 linearizability (在 FIFO 保证下，A-linearizability 是比 linearizability 更强的性质，A-linearizability 是在 linearizability 的基础上进一步满足了异步提交的请求的线性一致性)

Because only update requests are A-linearizable, ZooKeeper processes read requests locally at each replica. This allows the service to scale linearly as servers are added to the system.
>  ZooKeeper 仅保证更新请求是 A-linearizable
>  ZooKeeper 在每个副本上本地处理读请求，这使得服务可以随着更多 servers 的加入而线性拓展

To see how these two guarantees interact, consider the following scenario. A system comprising a number of processes elects a leader to command worker processes. When a new leader takes charge of the system, it must change a large number of configuration parameters and notify the other processes once it finishes.
>  要了解 linearizable writes 和 FIFO client order 这两个保证如何相互作用，我们考虑以下场景: 一个由多个进程组成的系统选举出一个 leader 来指挥工作进程，当新的 leader 接管时，它需要更改大量配置参数，并在完成后通知其他进程

We then have two important requirements: 

- As the new leader starts making changes, we do not want other processes to start using the configuration that is being changed; 
- If the new leader dies before the configuration has been fully updated, we do not want the processes to use this partial configuration. 

>  我们有两个重要的要求
>  - 新 leader 执行更改时，我们不希望其他进程按照旧配置启动
>  - 如果新 leader 在配置完全更新之前故障，我们不希望其他进程使用目前部分被修改的配置

Observe that distributed locks, such as the locks provided by Chubby, would help with the first requirement but are insufficient for the second. 
>  分布式锁，例如 Chubby 提供的锁可以满足第一个要求，但不足以满足第二个

 With ZooKeeper, the new leader can designate a path as the ready znode; other processes will only use the configuration when that znode exists. The new leader makes the configuration change by deleting ready, updating the various configuration znodes, and creating ready. 
 >  ZooKeeper 中，新 leader 可以指定某个路径为 ready znode，其他进程只有在该 ready znode 存在时才使用配置 (ready znode 用于传递 “配置可用” 的信号)
 >  新的 leader 执行配置更改时，会删除 ready znode，更新其他的配置 znodes，然后再创建 ready znode
 
All of these changes can be pipelined and issued asynchronously to quickly update the configuration state. Although the latency of a change operation is of the order of 2 milliseconds, a new leader that must update 5000 different znodes will take 10 seconds if the requests are issued one after the other; by issuing the requests asynchronously the requests will take less than a second. Because of the ordering guarantees, if a process sees the ready znode, it must also see all the configuration changes made by the new leader. If the new leader dies before the ready znode is created, the other processes know that the configuration has not been finalized and do not use it. 
 > 所有这些操作都可以被流水线化，异步提交和执行，从而快速更新配置状态
 > 虽然一次配置更改操作的延迟大约为 2ms，但如果请求是一个接着一个提交的，新的 leader 就必须花费 10s 更新 5000 个不同 znodes，通过异步地发出请求，整个过程可以在不到 1s 内完成
 > 因为存在顺序性保证，如果一个进程看见 ready znode，就意味着它能够看到新 leader 做的所有配置更改
 > 如果新 leader 在 ready znode 创建之前故障，则其他进程会直到配置尚未最终确定，故不会使用它
 
>  新的 leader 执行配置更改的方式就是向 ZooKeeper server 发送修改配置相关的 znode 的请求
>  因为允许异步的请求提交，新的 leader 可以迅速提交所有的配置更改请求，而无需等待上一个请求完成再提交下一个
>  新的 leader 在执行配置修改时，推荐遵循的协议是先提交删除 ready znode 的请求，然后提交所有更改配置 znodes 的请求，然后提交创建 ready znode 的请求
>  因为 ZooKeeper 保证操作的执行是 FIFO 的，故当其他进程看到 ready znode 重新出现时，就意味着所有的配置更改都完成了

The above scheme still has a problem: what happens if a process sees that ready exists before the new leader starts to make a change and then starts reading the configuration while the change is in progress. This problem is solved by the ordering guarantee for the notifications: if a client is watching for a change, the client will see the notification event before it sees the new state of the system after the change is made. Consequently, if the process that reads the ready znode requests to be notified of changes to that znode, it will see a notification informing the client of the change before it can read any of the new configuration. 
>  上述方案仍然存在一个问题: 一个进程可能在新 leader 执行更改之前看到了 ready znode 存在，然后在新 leader 执行更改的时候开始读取配置
>  该问题通过通知的顺序保证解决: 如果 client 在 watching 某个更改，在该更改发生后，系统转移到新状态之前，client 就会收到关于该更改的通知事件
>  那么，如果一个进程的读 ready znode 的请求 watch 了 ready znode，并被通知了 ready znode 的更改，该进程将在它能够读取任何新配置之前收到该通知 (也就是说，ZooKeeper 先通知进程 ready znode 被删除了，确认进程知道了这一点，才转移到下一个状态，开始接收新 leader 的请求，更新配置 znodes)

Another problem can arise when clients have their own communication channels in addition to ZooKeeper. For example, consider two clients $A$ and $B$ that have a shared configuration in ZooKeeper and communicate through a shared communication channel. If $A$ changes the shared configuration in ZooKeeper and tells $B$ of the change through the shared communication channel, $B$ would expect to see the change when it re-reads the configuration. If $B$ ’s ZooKeeper replica is slightly behind $A$ ’s, it may not see the new configuration. 
>  如果 clients 除了使用 ZooKeeper 以外，还有自己的通信通道，就会出现另一个问题
>  例如，考虑两个 clients A, B，它们在 ZooKeeper 中共享一个配置，并通过一个共享的通信通道交流，如果 A 更改了 ZooKeeper 中的共享配置，然后通过它们的共享通信通道告知 B，那么 B 重新读取配置时，它将期望看到该更改，如果 B 的 ZooKeeper 副本略微落后于 A 的，则它就可能看不到该更改

Using the above guarantees $B$ can make sure that it sees the most up-to-date information by issuing a write before re-reading the configuration. 
>  但如果二者仅通过 ZooKeeper 交流，借助 ZooKeeper 提供的保证，B 可以在重新读配置之前，提交一个写操作 (写入一个空字符，因为写是状态更新操作，ZooKeeper 提供的 linearizable writes 保证在该写操作执行之前，对于 B 的配置副本其他更改操作要先被执行，故 A 的修改就得以应用)，以确保它看到最新的信息

To handle this scenario more efficiently ZooKeeper provides the `sync` request: when followed by a read, constitutes a slow read. `sync` causes a server to apply all pending write requests before processing the read without the overhead of a full write. This primitive is similar in idea to the `flush` primitive of ISIS [5]. 
>  为了更高效处理该情况，ZooKeeper 提供了 `sync` 请求: `sync` 请求后面跟上一个读操作时，就等价于一个慢速读操作
>  `sync` 会让服务器在处理后面的 (读) 操作之前应用所有待处理的 (写) 请求，这样就无需使用上面的方法，避免了一次完整的写操作开销
>  这种原语的思想类似于 ISIS 中的 `flush` 原语

ZooKeeper also has the following two liveness and durability guarantees: if a majority of ZooKeeper servers are active and communicating the service will be available; and if the ZooKeeper service responds successfully to a change request, that change persists across any number of failures as long as a quorum of servers is eventually able to recover. 
>  ZooKeeper 还有以下两个关于可用性和持久性的保证:
>  如果多数 ZooKeeper servers 活跃并保持通信，服务就可用，并且，如果 ZooKeeper 服务成功响应了某个更改请求，只要有一组服务器最终能够恢复，该更改就会在任意多次故障后仍然持久存在 (更改请求被成功相应说明更改已提交)

## 2.4 Examples of primitives 
In this section, we show how to use the ZooKeeper API to implement more powerful primitives. The ZooKeeper service knows nothing about these more powerful primitives since they are entirely implemented at the client using the ZooKeeper client API. Some common primitives such as group membership and configuration management are also wait-free. For others, such as rendezvous, clients need to wait for an event. Even though ZooKeeper is wait-free, we can implement efficient blocking primitives with ZooKeeper. 
>  我们展示如何用 ZooKeeper API 来实现更强大的原语
>  这些原语对于 ZooKeeper 服务是不可知的，因为 ZooKeeper 服务是在 client 完全通过 ZooKeeper client API 实现的 (ZooKeeper 服务仅仅是提供一层抽象，原语的实现是在这层抽象之上，ZooKeeper 服务并不需要了解基于它能实现什么原语)
>  一些常见的原语例如 group membership, configuration management 也是无等待的，对于其他的原语例如 rendezvous, client 则需要等待一个事件
>  尽管 ZooKeeper 本身是无等待的，我们可以用 ZooKeeper 实现高效的阻塞原语

ZooKeeper’s ordering guarantees allow efficient reasoning about system state, and watches allow for efficient waiting. 

**Configuration Management** ZooKeeper can be used to implement dynamic configuration in a distributed application. In its simplest form configuration is stored in a znode, $z_{c}$ . Processes start up with the full pathname of $z_{c}$ . Starting processes obtain their configuration by reading $z_{c}$ with the watch flag set to true. If the configuration in $z_{c}$ is ever updated, the processes are notified and read the new configuration, again setting the watch flag to true. 
>  Configuration Management
>  ZooKeeper 可以用于实现分布式应用的动态配置，最简单的形式是将配置存储在一个 znode $z_c$ 中，进程启动时会使用 $z_c$ 的完整路径名，读取配置，并设定 watch flag 为 true，如果配置 $z_c$ 更新，进程将收到通知并读取新配置，同时再次设置 watch flag 为 true

Note that in this scheme, as in most others that use watches, watches are used to make sure that a process has the most recent information. For example, if a process watching $z_{c}$ is notified of a change to $z_{c}$ and before it can issue a read for $z_{c}$ there are three more changes to $z_{c}$ , the process does not receive three more notification events. This does not affect the behavior of the process, since those three events would have simply notified the process of something it already knows: the information it has for $z_{c}$ is stale. 
>  像这类使用 watch 的方法中，watch 是用于确保进程具有**最新**的信息
>  例如，如果一个 watch $z_c$ 的进程收到了 $z_c$ 发生了变化的通知，并且在它能够对 $z_c$ 发起读操作之前，$z_c$ 又发生了三次变化，那么该进程并不会再收到额外的三个通知事件
>  这实际上并不影响进程的行为，因为额外的三个通知事件也只是通知该进程他已经知道的信息: 它所拥有的关于 $z_c$ 的信息已经过时了

**Rendezvous** Sometimes in distributed systems, it is not always clear a priori what the final system configuration will look like. For example, a client may want to start a master process and several worker processes, but the starting processes is done by a scheduler, so the client does not know ahead of time information such as addresses and ports that it can give the worker processes to connect to the master. 
>  Rendezvous
>  有时在分布式系统中，最终的系统配置是什么样子在事先是不清楚的
>  例如，client 可能想启动一个 master 进程和多个 worker 进程，但这些进程的启动工作是由调度器完成的，故 client 并不事先知道例如 master 进程的地址和端口等信息，故不能提供这些信息给 worker 进程使它们连接到 master

We handle this scenario with ZooKeeper using a rendezvous znode, $z_{r}$ , which is an node created by the client. The client passes the full pathname of $z_{r}$ as a startup parameter of the master and worker processes. When the master starts it fills in $z_{r}$ with information about addresses and ports it is using. When workers start, they read $z_{r}$ with watch set to true. If $z_{r}$ has not been filled in yet, the worker waits to be notified when $z_{r}$ is updated. If $z_{r}$ is an ephemeral node, master and worker processes can watch for $z_{r}$ to be deleted and clean themselves up when the client ends. 
>  在 ZooKeeper 中，我们使用一个 rendezvous znode $z_r$ 处理该情况，该 znode 由 client 创建，client 会将 $z_r$ 的完整路径名作为 master 和 worker 进程的启动参数传递，当 master 启动时，它会在 $z_r$ 中填入自己的地址和端口信息，worker 启动后，会读取 $z_r$，并设定 watch，如果 $z_r$ 尚未被填充，worker 就等待 watch 的通知
>  如果 $z_r$ 是一个 ephemeral node, master 和 worker 进程可以监视 $z_r$ 是否被删除 (被删除说明 client 结束了服务)，并在 $z_r$ 被删除后自行清理自己

**Group Membership** We take advantage of ephemeral nodes to implement group membership. Specifically, we use the fact that ephemeral nodes allow us to see the state of the session that created the node. We start by designating a znode, $z_{g}$ to represent the group. When a process member of the group starts, it creates an ephemeral child znode under $z_{g}$ . If each process has a unique name or identifier, then that name is used as the name of the child znode; otherwise, the process creates the znode with the SEQUENTIAL flag to obtain a unique name assignment. Processes may put process information in the data of the child znode, addresses and ports used by the process, for example. 
>  Group Membership
>  我们利用临时节点实现 group membership
>  我们利用了可以借助临时节点查看创建了该节点的 session 状态的特性，首先，我们指定一个 znode $z_g$ 表示 group，当该 group 的某个成员进程启动时，它需要在 $z_g$ 下创建临时节点
>  如果每个进程都由唯一的名称或标识符，则其名称会直接用作它创建的节点的名称; 否则，进程会通过设置 SEQUENTIAL flag 来创建唯一的 znode 名称
>  进程还可以在其 znode 中存储进程信息，例如它使用的地址和端口

After the child znode is created under $z_{g}$ the process starts normally. It does not need to do anything else. If the process fails or ends, the znode that represents it under $z_{g}$ is automatically removed. 
>  创建了子节点后，进程正常启动，如果进程故障或结束，该 znode 会被自动移除 (因为是 ephemeral node)

Processes can obtain group information by simply listing the children of $z_{g}$ . If a process wants to monitor changes in group membership, the process can set the watch flag to true and refresh the group information (always setting the watch flag to true) when change notifications are received. 
>  进程 (包括组内进程) 可以通过列出 $z_g$ 的子节点获取组信息，如果进程想监控 group membership 的变化，可以设置 watch，并在收到更新后，再刷新 watch to true

**Simple Locks** Although ZooKeeper is not a lock service, it can be used to implement locks. Applications using ZooKeeper usually use synchronization primitives tailored to their needs, such as those shown above. Here we show how to implement locks with ZooKeeper to show that it can implement a wide variety of general synchronization primitives. 
>  Simple Locks
>  ZooKeeper 本身不是锁服务，但可以用于实现锁
>  使用 ZooKeeper 的应用通常根据需要构建同步原语，例如上面列出的，我们开始讨论如何用 ZooKeeper 实现锁，以展示 ZooKeeper 可以实现许多通用的同步原语

The simplest lock implementation uses “lock files”. The lock is represented by a znode. To acquire a lock, a client tries to create the designated znode with the EPHEMERAL flag. If the create succeeds, the client holds the lock. Otherwise, the client can read the znode with the watch flag set to be notified if the current leader dies. A client releases the lock when it dies or explicitly deletes the znode. Other clients that are waiting for a lock try again to acquire a lock once they observe the znode being deleted. 
>  最简单的锁实现使用 "lock files"，锁由一个 znode 表示
>  要获取锁，client 尝试以 EPHEMERAL 创建一个指定的 znode，如果创建成功，client 就持有锁，否则 (znode 已经被创建)，client 可以读取该 znode，设定 watch
>  client 在其终止或显式删除 znode 时释放锁，其他正在等待锁的 client 会在观察到 znode 被删除后再次尝试获取锁

While this simple locking protocol works, it does have some problems. First, it suffers from the herd effect. If there are many clients waiting to acquire a lock, they will all vie for the lock when it is released even though only one client can acquire the lock. Second, it only implements exclusive locking.  
>  上述简单的协议存在一些问题，其一，它会受到 “群体效应” 的影响，如果多个 client 等待获取锁时，锁被释放后，它们都会争夺锁，即便最后只有一个 client 会获取锁；其二，它只实现了互斥锁

The following two primitives show how both of these problems can be overcome.
>  要解决这两个问题，可以使用以下两个原语

**Simple Locks without Herd Effect** We define a lock znode $l$ to implement such locks. Intuitively we line up all the clients requesting the lock and each client obtains the lock in order of request arrival. Thus, clients wishing to obtain the lock do the following: 
>  Simple Locks without Herd Effect
>  我们定义 lock znode $l$ 来实现没有群体效应的锁
>  直观地说，我们将所有请求锁的 clients 排队，并按请求达到的顺序依次分配锁
>  希望获取锁的 clients 执行以下操作

```
Lock
1 n = create(l + "/lock-", EPHEMERAL | SEQUENTIAL)
2 C = getChindren(l, false)
3 if n is lowest znode in C, exit
4 p = znode in C ordered just before n
5 if exists(p, true) wait for watch event
6 goto 2
```

>  每个 client 在 znode `l` 下创建子节点，子节点是临时节点
>  创建子节点后，循环等待该子节点的上一个兄弟节点被移除 (表示上一个使用锁的进程结束)，等到后就退出 (进而对上锁区域执行操作)

```
Unlock 
1 delete(n) 
```

The use of the SEQUENTIAL flag in line 1 of `Lock` orders the client’s attempt to acquire the lock with respect to all other attempts. If the client’s znode has the lowest sequence number at line 3, the client holds the lock. Otherwise, the client waits for deletion of the znode that either has the lock or will receive the lock before this client’s znode. 
>  line 1 的 SEQUENTIAL flag 将当前 client 获取锁的尝试和其他所有尝试进行排序，如果 client 的 znode 的序列号最低，则 client 获取锁，否则，client 等待比其 znode 序列号更小的 znode 的删除 (也就是等待当前持有锁或者会在 client 之前持有锁的进程)

By only watching the znode that precedes the client’s znode, we avoid the herd effect by only waking up one process when a lock is released or a lock request is abandoned. Once the znode being watched by the client goes away, the client must check if it now holds the lock. (The previous lock request may have been abandoned and there is a znode with a lower sequence number still waiting for or holding the lock.) 
>  通过仅 watch 在 client 的 znode 之前的 znode，我们避免了群体效应，即此时在锁被释放或者锁请求被放弃时仅唤醒一个进程
>  当 client 正在 watch 的 znode 消失，client 需要检查自己是否现在持有锁 (因为有可能其 znode 的前一个 znode 消失的原因是锁请求被放弃，此时仍存在一个序列号更低的 znode 正在等待或者持有锁，故正在 watch 的 znode 消失并不一定意味着当前 client 获取了锁，此时，如果 client 仍有持有锁的需求，需要重新设置 watch)

Releasing a lock is as simple as deleting the znode $n$ that represents the lock request. By using the EPHEMERAL flag on creation, processes that crash will automatically cleanup any lock requests or release any locks that they may have. 
>  释放锁即删除表示锁请求的 znode，因为使用了 EPHEMERAL，故进程崩溃也会自动释放进程的全部锁请求

In summary, this locking scheme has the following advantages: 
1. The removal of a znode only causes one client to wake up, since each znode is watched by exactly one other client, so we do not have the herd effect; 
2. There is no polling or timeouts; 
3. Because of the way we have implemented locking, we can see by browsing the ZooKeeper data the amount of lock contention, break locks, and debug locking problems. 

>  该锁机制具有以下优势
>  1. 删除一个 znode 只会导致一个 client 被唤醒，因为每个 znode 仅被另一个 client watch，故不会出现群体效应
>  2. 没有轮询或超时机制 (通过 watch 机制发送通知)
>  3. 通过浏览 ZooKeeper 的数据，可以查看锁的竞争程度，并可以破坏锁并调试锁相关的问题

**Read/Write Locks** To implement read/write locks we change the lock procedure slightly and have separate read lock and write lock procedures. The unlock procedure is the same as the global lock case. 
>  Read/Write Locks
>  为了实现读写锁，我们略微改变锁的过程，解锁的过程保持不变

```
Write Lock 
1 n = create(l + “/write-”, EPHEMERAL|SEQUENTIAL) 
2 C = getChildren(l, false) 
3 if n is lowest znode in C, exit 
4 p = znode in C ordered just before n 
5 if exists(p, true) wait for event 
6 goto 2 
```

>  写锁的过程和之前的过程一致，因为写锁本质也是互斥锁

```
# Read Lock 
1 n = create(l + “/read-”, EPHEMERAL|SEQUENTIAL) 
2 C = getChildren(l, false) 
3 if no write znodes lower than n in C, exit 
4 p = write znode in C ordered just before n 
5 if exists(p, true) wait for event 
6 goto 3 
```

>  读锁的检查机制略微有些不同，写锁是检查当前 znode 之前是否存在任意 znode，读锁则只检查当前 znode 之前是否存在 write znode，如果没有，就持有锁 (存在 read znode 不影响获取读锁)，否则等待

This lock procedure varies slightly from the previous locks. Write locks differ only in naming. Since read locks may be shared, lines 3 and 4 vary slightly because only earlier write lock znodes prevent the client from obtaining a read lock. 
>  写锁的实现方式和之前的区别仅在于锁的命名
>  而因为读锁可以被共享，故检查方式略微不同，只有较早的 write lock znodes 可以防止 client 获取读锁

It may appear that we have a “herd effect” when there are several clients waiting for a read lock and get notified when the “write-” znode with the lower sequence number is deleted; in fact, this is a desired behavior, all those read clients should be released since they may now have the lock. 
>  当有多个 clients 等待一个读锁，并且在上一个 write znode 被释放后得到通知，可能会看起来出现群体效应
>  但该行为就是期望的，因为读锁可以被同时获取

**Double Barrier** Double barriers enable clients to synchronize the beginning and the end of a computation. When enough processes, defined by the barrier threshold, have joined the barrier, processes start their computation and leave the barrier once they have finished. 
>  Double Barrier
>  双屏障允许 clients 在计算的开始和结束进行同步，当足够数量的进程加入屏障 (该数量由 barrier threshold 定义) 后，这些进程会开始它们的计算，并在全部计算完成后离开屏障

We represent a barrier in ZooKeeper with a znode, referred to as $b$ . Every process $p$ registers with b – by creating a znode as a child of $b-$ on entry, and unregisters – removes the child – when it is ready to leave. Processes can enter the barrier when the number of child znodes of $b$ exceeds the barrier threshold. Processes can leave the barrier when all of the processes have removed their children. 
>  ZooKeeper 中，barrier 用一个 znode 表示，记作 $b$
>  每个进程 $p$ 进入 barrier 时，在 $b$ 下创建一个子节点进行注册，离开屏障时，通过删除子节点进行注销
>  当 $b$ 的子节点数量超过阈值时，进程可以进入屏障 (开始计算)，在 $b$ 的子节点都被移除后，进程可以离开屏障 (因此，进程需要等待所有进程一起开始计算，同时等待所有进程全部完成计算)

We use watches to efficiently wait for enter and exit conditions to be satisfied. To enter, processes watch for the existence of a ready child of $b$ that will be created by the process that causes the number of children to exceed the barrier threshold. To leave, processes watch for a particular child to disappear and only check the exit condition once that znode has been removed. 
>  watches 用于高效等待 barrier 的进入和离开条件的满足
>  进程监视 $b$ 的某个 ready 子节点是否存在，当它被创建后，$b$ 的子节点数量就超过了阈值
>  进程监视 $b$ 的某个特定子节点是否消失，仅在它被移除后，才会检查自己的退出条件

# 3 ZooKeeper Applications 
We now describe some applications that use ZooKeeper, and explain briefly how they use it. We show the primitives of each example in bold. 
>  我们描述使用 ZooKeeper 的一些应用

**The Fetching Service** Crawling is an important part of a search engine, and Yahoo! crawls billions of Web documents. The Fetching Service (FS) is part of the Yahoo! crawler and it is currently in production. 
>  The Fetching Service
>  爬虫是搜索引擎的重要组成部分，而雅虎每天会爬取数十亿个网页文档，抓取服务 (Fetching Service，简称 FS) 是雅虎爬虫的一部分，并且已经投入生产使用

Essentially, it has master processes that command page-fetching processes. The master provides the fetchers with configuration, and the fetchers write back informing of their status and health. The main advantages of using ZooKeeper for FS are recovering from failures of masters, guaranteeing availability despite failures, and decoupling the clients from the servers, allowing them to direct their request to healthy servers by just reading their status from ZooKeeper. Thus, FS uses ZooKeeper mainly to manage configuration metadata, although it also uses ZooKeeper to elect masters (leader election). 
>  Fetching Service 有一个 master 进程，它指挥页面抓取进程执行任务，master 为 fetchers 提供配置，fetchers 则会回传其状态和健康状况
>  使用 ZooKeeper 实现 FS 的主要优势在于可以从 master 的故障中恢复，在发生故障时仍确保可用性，并且解耦了客户端和服务器，允许客户端通过读取 ZooKeeper 中的状态信息来选择向健康的服务器发送请求
>  因此 FS 主要用 ZooKeeper 来管理配置元数据，也用 ZooKeeper 实现 leader election


![[pics/ZooKeeper-Fig2.png]]

Figure 2 shows the read and write traffic for a ZooKeeper server used by FS through a period of three days. To generate this graph, we count the number of operations for every second during the period, and each point corresponds to the number of operations in that second. 
>  Figure 2 展示了 FS 使用的 ZooKeeper 服务器在三天内的读写流量
>  为了生成该图，我们统计了该时段内每秒钟的操作数量，图中的每个点都对应了了一秒的操作数量

We observe that the read traffic is much higher compared to the write traffic. During periods in which the rate is higher than 1, 000 operations per second, the read: write ratio varies between 10:1 and 100:1. The read operations in this workload are `getData()`, `getChildren()`, and `exists()`, in increasing order of prevalence. 
>  我们发现读取流量远高于写入流量

**Katta** Katta [17] is a distributed indexer that uses ZooKeeper for coordination, and it is an example of a non-Yahoo! application. Katta divides the work of indexing using shards. A master server assigns shards to slaves and tracks progress. Slaves can fail, so the master must redistribute load as slaves come and go. The master can also fail, so other servers must be ready to take over in case of failure. Katta uses ZooKeeper to track the status of slave servers and the master (**group membership**), and to handle master failover (**leader election**). Katta also uses ZooKeeper to track and propagate the assignments of shards to slaves (**configuration management**). 
>  Katta
>  Katta 是一个分布式索引器，使用 ZooKeeper 进行 coordination
>  Katta 将索引的工作划分为 shards, master 服务器将 shards 分配给 slaves 并追踪进度，master 可能会故障，故其他服务器需要准备好在发生故障时接管
>  Katta 使用 ZooKeeper 追踪 slave 服务器的状态 (通过 group membership 原语)，处理 master 的故障转移 (通过 leader election 原语)
>  Katta 还使用 ZooKeeper 来跟踪和传播 shards 到 slaves 服务器 (通过 configuration management 原语)

**Yahoo! Message Broker** Yahoo! Message Broker (YMB) is a distributed publish-subscribe system. The system manages thousands of topics that clients can publish messages to and receive messages from. The topics are distributed among a set of servers to provide scalability. Each topic is replicated using a primary-backup scheme that ensures messages are replicated to two machines to ensure reliable message delivery. The servers that makeup YMB use a shared-nothing distributed architecture which makes coordination essential for correct operation. YMB uses ZooKeeper to manage the distribution of topics (**configuration metadata**), deal with failures of machines in the system (**failure detection** and **group membership**), and control system operation. 
>  Yahoo! Message Broker
>  YMB 是一个分布式发布-订阅系统，该系统管理上千个主题，clients 可以向这些主题发送消息，并从中接收消息
>  主题分布在一组服务器上，以提供可拓展性，每个主题都通过 primary-backup 方案进行复制，确保消息会被复制到两台机器上
>  构成 YMB 的服务器使用无共享的分布式架构，故 coordination 对于正确运行至关重要，YMB 使用 ZooKeeper 管理主题的分布 (通过 configuration meta 原语)、处理系统中机器的故障 (通过 failure detection, group membership 原语)、控制系统的运行

![[pics/ZooKeeper-Fig3.png]]

Figure 3 shows part of the znode data layout for YMB. Each broker domain has a znode called `nodes` that has an ephemeral znode for each of the active servers that compose the YMB service. Each YMB server creates an ephemeral znode under `nodes` with load and status information providing both group membership and status information through ZooKeeper. 
>  YMB 中，每个 broker 域都由一个称为 `nodes` 的 znode，每个构成了 YMB 服务的活跃服务器都对应它的一个子 ephemeral znode
>  该 ephemeral znode 存储了对应 YMB 服务器的负载和状态信息

Nodes such as `shutdown` and `migration prohibited` are monitored by all of the servers that make up the service and allow centralized control of YMB. The `topics` directory has a child znode for each topic managed by YMB. These topic znodes have child znodes that indicate the primary and backup server for each topic along with the subscribers of that topic. 
>  像 `shutdown`, `migration prohibited` 这样的节点由构成该服务的所有服务器监控，从而实现对 YMB 的集中控制
>  `topics` 目录中，每个 YMB 管理的主题都有一个对应的子 znode，这些 znodes 也有各自的子 znodes，表示该主题的 primary server 和 backup server

The primary and backup server znodes not only allow servers to discover the servers in charge of a topic, but they also manage **leader election** and server crashes. 
>  每个主题的 znode 的 primary 和 backup server znode 记录了关于存储了该主题信息的 primary 和 backup server 的元信息，故 servers 和 clients 可以利用它们查找存储了该主题的对应 server
>  另外，它们还可以用于管理 leader election 和 server crashes

# 4 ZooKeeper Implementation 
ZooKeeper provides high availability by replicating the ZooKeeper data on each server that composes the service. We assume that servers fail by crashing, and such faulty servers may later recover.
>  ZooKeeper 通过将 ZooKeeper 数据复制到构成服务的各个服务器来提供高可用性
>  我们假设了服务器的 fail by crashing，即故障的服务器会在之后恢复

 Figure 4 shows the high-level components of the ZooKeeper service. Upon receiving a request, a server prepares it for execution (request processor). If such a request requires coordination among the servers (write requests), then they use an agreement protocol (an implementation of atomic broadcast), and finally servers commit changes to the ZooKeeper database fully replicated across all servers of the ensemble.
 >  Figure 4 展示了 ZooKeeper 服务的高级组件
 >  服务器从客户端收到一个请求时，服务器会准备该请求以供执行 (request processor)，如果该请求需要 servers 之间的协调 (例如 write requests)，则 servers 会使用共识协议 (该协议是原子性广播的一个实现)，最后，servers 将更改提交到完全复制在 ZooKeeper 所有服务器上的数据库中
 
In the case of read requests, a server simply reads the state of the local database and generates a response to the request. 
>  对于读取请求，server 仅需要读取其本地数据库的状态，生成对请求的响应即可
  

![[pics/ZooKeeper-Fig4.png]]

The replicated database is an in-memory database containing the entire data tree. Each znode in the tree stores a maximum of 1MB of data by default, but this maximum value is a configuration parameter that can be changed in specific cases. 
>  replicated database 是内存数据库，包含了整个数据树
>  树中的每个 znode 默认存储最大 1MB 的数据，可以通过配置参数调节

For recoverability, we efficiently log updates to disk, and we force writes to be on the disk media before they are applied to the in-memory database. In fact, as Chubby [8], we keep a replay log (a write-ahead log, in our case) of committed operations and generate periodic snapshots of the in-memory database. 
>  为了保障数据可恢复，我们将更新记录到磁盘中，并且在更新应用于内存数据库之前，确保写操作已经应用到磁盘数据
>  和 Chubby 类似，我们保留了已提交操作的 replay log (一个预写日志，即先写日志，再应用操作)，并且为内存数据库定期生成快照

Every ZooKeeper server services clients. Clients connect to exactly one server to submit its requests. As we noted earlier, read requests are serviced from the local replica of each server database. Requests that change the state of the service, write requests, are processed by an agreement protocol. 
>  每台 ZooKeeper server 都会为客户端提供服务
>  客户端会连接到某一台服务器，以提交它的请求，如我们之前所述，读请求会根据每台服务器本地的数据库副本处理，改变服务状态的请求 (即写入请求)，会通过共识协议处理

As part of the agreement protocol write requests are forwarded to a single server, called the leader. The rest of the ZooKeeper servers, called followers, receive message proposals consisting of state changes from the leader and agree upon state changes. 
>  在共识协议中，写请求会被转发给 leader 服务器，其余的 ZooKeeper 服务器称为 followers，它们从 leader 接收包含状态变化的消息提案，就状态变化达成一致

## 4.1 Request Processor 
Since the messaging layer is atomic, we guarantee that the local replicas never diverge, although at any point in time some servers may have applied more transactions than others. 
>  因为消息层是原子性的 (要么成功传递消息，要么消息传递完全失败，不存在消息传递一半的情况)，我们保证 local replicas 永远不会出现分歧，虽然在任意给定的时间点，某台服务器可能比其他服务器应用了更多的 transactions

Unlike the requests sent from clients, the transactions are idempotent. When the leader receives a write request, it calculates what the state of the system will be when the write is applied and transforms it into a transaction that captures this new state. The future state must be calculated because there may be outstanding transactions that have not yet been applied to the database. 
>  和 clients 发送的请求不同, transactions 是幂等的
>  当 leader 接收一个写请求时，它会先计算出应用写入之后，系统的状态是什么，然后将该请求转化为一个 transaction，该 transaction 包含了新状态的信息
>  leader 必须计算应用写入后的未来状态，因为目前可能还存在尚未应用到数据库中的其他 transactions

For example, if a client does a conditional `setData` and the version number in the request matches the future version number of the znode being updated, the service generates a `setDataTXN` that contains the new data, the new version number, and updated time stamps. If an error occurs, such as mismatched version numbers or the znode to be updated does not exist, an `errorTXN` is generated instead. 
>  例如，如果 client 请求了一个条件性的 `setData` ，并且请求中的版本号和**正在更新的 znode 的未来版本号**匹配，ZooKeeper 服务将生成一个包含了新数据、新版本号和更新时间戳的 `setDataTXN` (通过计算，将请求转化为 transaction，即便当前 znode 的版本号不匹配，但更新后能匹配，就生成 `setDataTXN`)
>  如果发生了错误，例如版本号不匹配或者要更新的 znode 不存在，则生成一个 `errorTXN` (将请求转化为 transaction)

## 4.2 Atomic Broadcast 
All requests that update ZooKeeper state are forwarded to the leader. The leader executes the request and broadcasts the change to the ZooKeeper state through Zab [24], an atomic broadcast protocol. The server that receives the client request responds to the client when it delivers the corresponding state change. 
>  所有更新 ZooKeeper 状态的请求都会发送给 leader
>  leader 执行该请求，并且通过 Zab (一种原子广播协议) 将状态变更广播到 ZooKeeper 状态中
>  然后，接收了 client 请求的 server 会**在它执行了对应状态变更后**，回应 client

Zab uses by default simple majority quorums to decide on a proposal, so Zab and thus ZooKeeper can only work if a majority of servers are correct (i.e., with $2f+1$ server we can tolerate $f$ failures). 
>  Zab 默认使用多数模式来决定一个提案，故 Zab 和 ZooKeeper 只能在多数服务器正常运行的情况下工作 (即有 2f+1 个服务器说明我们可以容忍 f 次故障)

To achieve high throughput, ZooKeeper tries to keep the request processing pipeline full. It may have thousands of requests in different parts of the processing pipeline. Because state changes depend on the application of previous state changes, Zab provides stronger order guarantees than regular atomic broadcast. More specifically, Zab guarantees that changes broadcast by a leader are delivered in the order they were sent and all changes from previous leaders are delivered to an established leader before it broadcasts its own changes. 
>  为了达成高吞吐，ZooKeeper 尽量保持请求处理流水线的满载状态，处理流水线上的不同部分可能有上千个请求
>  因为当前状态变化的应用依赖于之前的状态变化，Zab 提供了比普通原子广播更强的顺序保证，更具体地说，Zab 确保由 leader 广播的状态变化**按照它们发送的顺序交付**，并且所有来自之前 leader 的状态变化会在当前 leader 广播其自身的状态变化中**被传递给当前 leader** (如果 servers 都执行确定状态机，那么这些都是确保共识所必须的)

There are a few implementation details that simplify our implementation and give us excellent performance. We use TCP for our transport so message order is maintained by the network, which allows us to simplify our implementation. We use the leader chosen by Zab as the ZooKeeper leader, so that the same process that creates transactions also proposes them. We use the log to keep track of proposals as the write-ahead log for the in-memory database, so that we do not have to write messages twice to disk. 
>  一些实现细节简化了我们的实现，并提高了性能
>  我们使用 TCP 执行消息传输，故网络会维护消息的顺序；我们使用 Zab 选举出的 leader 作为 ZooKeeper 的 leader，故创建 transaction 的进程同时也是提出它们的进程；我们使用日志来跟踪 proposals，日志是内存数据库的预写日志

During normal operation Zab does deliver all messages in order and exactly once, but since Zab does not persistently record the id of every message delivered, Zab may redeliver a message during recovery. Because we use idempotent transactions, multiple delivery is acceptable as long as they are delivered in order. In fact, ZooKeeper requires Zab to redeliver at least all messages that were delivered after the start of the last snapshot. 
>  在正常运行中，Zab 会按照顺序传递消息，并且每个消息仅传递一次，但由于 Zab 不会持久化记录每个已传递消息的 id，故在 recovery 过程中，Zab 可能会重新传递某个消息
>  但我们使用的是幂等事务，故只要消息按序传递，重复的消息是可接受的
>  事实上，ZooKeeper 要求 Zab 在 recovery 时至少重新传递自上一次快照开始后传递的所有消息

## 4.3 Replicated Database 
Each replica has a copy in memory of the ZooKeeper state. When a ZooKeeper server recovers from a crash, it needs to recover this internal state. 
>  每个 replica/server 在内存中都存储了一个 ZooKeeper 状态的拷贝，当 server 从崩溃中恢复后，它需要复原其内部状态

Replaying all delivered messages to recover state would take prohibitively long after running the server for a while, so ZooKeeper uses periodic snapshots and only requires redelivery of messages since the start of the snapshot. 
>  通过 replay 所有已发送的消息来恢复状态将花费过长时间，故 ZooKeeper 使用定期快照，并仅要求重新发送自快照开始以来的消息

We call ZooKeeper snapshots fuzzy snapshots since we do not lock the ZooKeeper state to take the snapshot; instead, we do a depth first scan of the tree atomically reading each znode’s data and meta-data and writing them to disk. Since the resulting fuzzy snapshot may have applied some subset of the state changes delivered during the generation of the snapshot, the result may not correspond to the state of ZooKeeper at any point in time. However, since state changes are idempotent, we can apply them twice as long as we apply the state changes in order. 
>  ZooKeeper 的快照称为模糊快照，因为我们不会在 snapshotting 时锁定 ZooKeeper 状态，我们将以深度优先的方式原子化扫描数据树，读取每个 znode 的数据和元数据，然后将其写入磁盘
>  因为生成模糊快照的时候可能有更改正被部分应用，故快照结果可能不会对应于 ZooKeeper 在任何时间点的状态
>  但是，因为状态更改是幂等的，只要按照正常顺序应用状态更改，我们就可以多次应用它们

For example, assume that in a ZooKeeper data tree two nodes /foo and /goo have values f1 and g1 respectively and both are at version 1 when the fuzzy snapshot begins, and the following stream of state changes arrive having the form `<transactionType, path, value, new-version>`: 

```
<SetDataTXN, /foo, f2, 2>
<SetDataTXN, /goo, g2, 2>
<SetDataTXN, /foo, f3, 3> 
```

After processing these state changes, /foo and /goo have values f3 and g2 with versions 3 and 2 respectively. However, the fuzzy snapshot may have recorded that /foo and /goo have values f3 and g1 with versions 3 and 1 respectively, which was not a valid state of the ZooKeeper data tree. 

>  例如，假设某个 server 存储的 ZooKeeper 数据树中有两个 znode `/foo, /goo` ，它们分别存储了值 `f1, g1` ，且各自处于 version 1
>  当模糊快照开始时，有三个 transactions 到达该 server，如上所示，处理完这三个 transactions，`/foo, /goo` 的值应该是 `f3, g2` ，且 version 分别是 3, 2
>  但执行模糊快照的过程中，可能仅执行了部分 transactions，故快照存下来的值可能是 `f3, g1` ，且 version 分别是 3, 1，这并不是 ZooKeeper 数据树的一个有效状态

If the server crashes and recovers with this snapshot and Zab redelivers the state changes, the resulting state corresponds to the state of the service before the crash. 

## 4.4 Client-Server Interactions 
When a server processes a write request, it also sends out and clears notifications relative to any watch that corresponds to that update. Servers process writes in order and do not process other writes or reads concurrently. This ensures strict succession of notifications. 
>  当 server 处理写请求时，它会向 watchers 发送通知，并清除 watches
>  servers 会按照顺序处理写操作，并且不会并发处理其他写和读操作，这确保了通知的严格顺序性

Note that servers handle notifications locally. Only the server that a client is connected to tracks and triggers notifications for that client. 
>  注意，server 在本地处理通知，只有 client 所连接的 server 会为该 client 追踪和触发通知 (因此，有时虽然写操作已经提交，但如果 client 所连接的 server 暂未应用该操作，client 就不会收到通知)

Read requests are handled locally at each server. Each read request is processed and tagged with a zxid that corresponds to the last transaction seen by the server. This zxid defines the partial order of the read requests with respect to the write requests. By processing reads locally, we obtain excellent read performance because it is just an in-memory operation on the local server, and there is no disk activity or agreement protocol to run. This design choice is key to achieving our goal of excellent performance with read-dominant workloads. 
>  每个 server 在本地处理读请求，每个读请求会被处理并附加上一个 zxid，该 zxid 对应 server 处理该请求时最近的一次 transaction
>  该 zxid 定义了读请求相对于写请求的偏序关系 (client 知道读取的数据是在哪一次写操作之后的)
>  通过在本地处理读请求，我们获得了出色的读取性能，因为 server 只需要执行对内存的读操作，没有磁盘活动和共识协议需要运行
>  因为我们的工作负载以读取为主，故这个设计选择十分重要

One drawback of using fast reads is not guaranteeing precedence order for read operations. That is, a read operation may return a stale value, even though a more recent update to the same znode has been committed. 
>  这样设计的一个缺点是无法保证读操作的优先顺序，即读操作可能返回过期的值，即便对于其读取的 znode 的更新操作已经提交

Not all of our applications require precedence order, but for applications that do require it, we have implemented sync. This primitive executes asynchronously and is ordered by the leader after all pending writes to its local replica. 
>  对于需要确保优先顺序的应用，我们实现了 `sync` 原语
>  `sync` 原语是异步执行的 (server 和 leader 同步之后就可以退出)，leader 会在完成对其本地副本的所有待完成的写入操作后执行 sync

To guarantee that a given read operation returns the latest updated value, a client calls sync followed by the read operation. The FIFO order guarantee of client operations together with the global guarantee of sync enables the result of the read operation to reflect any changes that happened before the sync was issued. 
>  client 要确保读取到最近更新的值时，需要在调用 read 操作之前先调用 sync
>  ZooKeeper 的 client operations FIFO 保证和 sync 的全局保证将确保读操作的结果能够反映在 sync 发出之前的所有更改

In our implementation, we do not need to atomically broadcast sync as we use a leader-based algorithm, and we simply place the sync operation at the end of the queue of requests between the leader and the server executing the call to sync.
>  在我们的实现中，因为我们使用的是基于 leader 的算法，故不需要原子广播 sync (不需要就 sync 达成共识)，我们只需要将 sync 操作放置在 leader 和执行 sync 调用的 server 的请求队列末尾即可 (leader 执行到 sync 后，会完成所有写入，然后与该 server 进行同步)

In order for this to work, the follower must be sure that the leader is still the leader. If there are pending transactions that commit, then the server does not suspect the leader. If the pending queue is empty, the leader needs to issue a null transaction to commit and orders the sync after that transaction.  This has the nice property that when the leader is under load, no extra broadcast traffic is generated.
>  要确保该方法有效，follower 需要确定当前的 leader 仍然是 leader (不是过期的 leader)，如果 follower 对于 leader 的请求队列中，此时有写入请求被 leader 转化为 transactions (或者之前转换的 transactions) 在被提交 (说明该 leader 在工作)，则 server 不会怀疑 leader
>  如果待处理事务队列为空，leader 需要发出一个空事务进行提交，并且在该事务后执行 sync
>  这样设计的一个好处是在 leader 处于高负载状态时，不会生成额外的广播流量 (server 通过 leader 日常提交的 transaction 确认 leader，不需要 leader 广播heartbeat)

 In our implementation, timeouts are set such that leaders realize they are not leaders before followers abandon them, so we do not issue the null transaction. 
>  在我们的实现中，我们设置了 timeout，使得 (过期的) leaders 会在 followers 放弃它们之前就意识到自己不是 leader，故它们不会发出空 transaction

ZooKeeper servers process requests from clients in FIFO order. Responses include the zxid that the response is relative to. Even heartbeat messages during intervals of no activity include the last zxid seen by the server that the client is connected to. If the client connects to a new server, that new server ensures that its view of the ZooKeeper data is at least as recent as the view of the client by checking the last zxid of the client against its last zxid. 
>  ZooKeeper 服务器以 FIFO 顺序处理来自客户端的请求
>  服务器的响应包括了该响应相关的 zxid (server 最近一次处理的事务 id)，即便是在没有活动的间隔期间的 heartbeat 消息也会包含 zxid
>  如果客户端连接到一个新的服务器，新的服务器会比较自身的 zxid 和客户端的最新的 zxid，以确保它保存的 ZooKeeper 数据树至少和客户端之前看到的一样新

If the client has a more recent view than the server, the server does not reestablish the session with the client until the server has caught up. The client is guaranteed to be able to find another server that has a recent view of the system since the client only sees changes that have been replicated to a majority of the ZooKeeper servers. This behavior is important to guarantee durability. 
>  如果客户端的视图更新，server 在赶上客户端的视图之前，不会重建和客户端的 session
>  因为客户端仅能看到已经复制到多数 ZooKeeper servers 的更改 (更改只有提交后，执行结果才会返回给客户端)，故客户端保证会找到另一个具有系统最新视图的 server
>  这一行为确保了持久性 (客户端总是能和某个视图一致的 server 沟通)

To detect client session failures, ZooKeeper uses timeouts. The leader determines that there has been a failure if no other server receives anything from a client session within the session timeout. If the client sends requests frequently enough, then there is no need to send any other message. Otherwise, the client sends heartbeat messages during periods of low activity.
>  为了检查客户端会话故障，ZooKeeper 使用 timeouts
>  如果没有其他 server 在 session timeout 内从 client session 中受到任何信息，leader 会判断 client 故障
>  如果 client 足够频繁地发送请求，则无需发送其他消息，否则需要发送 heartbeat 维持 session

 If the client cannot communicate with a server to send a request or heartbeat, it connects to a different ZooKeeper server to re-establish its session. To prevent the session from timing out, the ZooKeeper client library sends a heartbeat after the session has been idle for $s/3$ ms and switch to a new server if it has not heard from a server for $2s/3\mathrm{ms}$ , where $s$ is the session timeout in milliseconds. 
 >  如果 client 无法与某个 server 通信以发送请求或 heartbeat，它会连接到另一个 ZooKeeper server，重建其 session
 >  为了防止 session timeout, ZooKeeper client library 会在 session 空闲的 $s/3$ ms 后发送 heartbeat，如果在 $2s/3$ ms 内没有收到 server 的回复，就切换到另一个 server
 >  其中 $s$ 就是 session timeout

# 5 Evaluation 
We performed all of our evaluation on a cluster of 50 servers. Each server has one Xeon dual-core $2.1\mathrm{GHz}$ processor, 4GB of RAM, gigabit ethernet, and two SATA hard drives. We split the following discussion into two parts: throughput and latency of requests. 

## 5.1 Throughput 
To evaluate our system, we benchmark throughput when the system is saturated and the changes in throughput for various injected failures. We varied the number of servers that make up the ZooKeeper service, but always kept the number of clients the same. To simulate a large number of clients, we used 35 machines to simulate 250 simultaneous clients. 
>  我们在系统饱和时对吞吐量进行测试，并针对各种注入的故障分析吞吐量的变化
>  ZooKeeper service 中的 servers 数量会变化，但 clients 数量不变

We have a Java implementation of the ZooKeeper server, and both Java and C clients. For these experiments, we used the Java server configured to log to one dedicated disk and take snapshots on another. Our benchmark client uses the asynchronous Java client API, and each client has at least 100 requests outstanding. 
>  ZooKeeper server 有 Java 实现, client 有 C 和 Java 实现
>  实验中，server 将日志记录在一个磁盘，将快照记录在另一个磁盘
>  client 使用异步 Java API，每个 client 至少有 100 个待完成的请求

Each request consists of a read or write of 1K of data. We do not show benchmarks for other operations since the performance of all the operations that modify state are approximately the same, and the performance of nonstate modifying operations, excluding sync, are approximately the same. (The performance of sync approximates that of a light-weight write, since the request must go to the leader, but does not get broadcast.) 
>  每个请求包含 1KB 的数据读取或写入，因为修改状态的操作性能都类似，不修改状态的操作性能都类似，故其他操作不展示 (sync 的性能近似一个轻量的写操作，因为请求需要先由 server 发送到 leader，但不会进行广播)

Clients send counts of the number of completed operations every $300m s$ and we sample every 6s. To prevent memory overflows, servers throttle the number of concurrent requests in the system. ZooKeeper uses request throttling to keep servers from being overwhelmed. For these experiments, we configured the ZooKeeper servers to have a maximum of 2, 000 total requests in process. 
>  client 每 300ms 发送已完成操作的数量统计，我们每 6s 保存一次
>  为了避免内存溢出，servers 会限制系统中并发请求的数量，ZooKeeper 使用请求限流防止 servers 过载
>  我们配置 servers 最多同时处理 2000 个请求

![[pics/ZooKeeper-Fig5.png]]

In Figure 5, we show throughput as we vary the ratio of read to write requests, and each curve corresponds to a different number of servers providing the ZooKeeper service. 

![[pics/ZooKeeper-Table1.png]]

Table 1 shows the numbers at the extremes of the read loads. Read throughput is higher than write throughput because reads do not use atomic broadcast. 
>  读吞吐远高于写吞吐，因为读不需要原子广播

The graph also shows that the number of servers also has a negative impact on the performance of the broadcast protocol. From these graphs, we observe that the number of servers in the system does not only impact the number of failures that the service can handle, but also the workload the service can handle. Note that the curve for three servers crosses the others around $60\%$ . This situation is not exclusive of the three-server configuration, and happens for all configurations due to the parallelism local reads enable. It is not observable for other configurations in the figure, however, because we have capped the maximum y-axis throughput for readability. 
>  servers 数量对广播协议的性能有负面影响 (需要共识的 servers 提高了)
>  我们还发现 servers 数量不仅影响系统可以处理的故障数量，还影响系统可以处理的工作负载
>  可以看到 3 server 系统的吞吐曲线大约在读操作为 60%时与其他曲线相交 (因为读是本地处理的，故 server 越多，能够处理的读请求就越多)

There are two reasons for write requests taking longer than read requests. First, write requests must go through atomic broadcast, which requires some extra processing and adds latency to requests. The other reason for longer processing of write requests is that servers must ensure that transactions are logged to non-volatile store before sending acknowledgments back to the leader. In principle, this requirement is excessive, but for our production systems we trade performance for reliability since ZooKeeper constitutes application ground truth. 
>  写请求比读请求更久的原因有两点
>  - 其一，写请求需要经过原子广播，这需要额外的处理步骤，进而为请求处理添加延迟
>  - 其二，在 server 向 leader 发送确认之前，必须确保 transactions 已经记录在非易失存储中。在原则上，该要求可能过于严格，但对于我们的生产系统而言，我们选择用性能换取可靠性，因为 ZooKeeper 是其他应用的基础

We use more servers to tolerate more faults. We increase write throughput by partitioning the ZooKeeper data into multiple ZooKeeper ensembles. This performance trade off between replication and partitioning has been previously observed by Gray et al. [12]. 
>  我们使用更多的 servers 以容忍更多故障
>  通过将 ZooKeeper 数据划分到多个 ZooKeeper 集群中，我们可以提高写入吞吐量，Gray 等人已经观察到了在复制和划分之间的性能权衡

![[pics/ZooKeeper-Fig6.png]]

ZooKeeper is able to achieve such high throughput by distributing load across the servers that makeup the service. We can distribute the load because of our relaxed consistency guarantees. Chubby clients instead direct all requests to the leader. 
>  ZooKeeper 实现如此高的读吞吐量的原因是它将负载分布在了多个服务器上，因为我们采用了较为宽松的一致性保证，故可以分布负载
>  相比之下，Chubby clients 会将所有请求 (包括读请求) 发送到 leader

Figure 6 shows what happens if we do not take advantage of this relaxation and forced the clients to only connect to the leader. As expected the throughput is much lower for read-dominant workloads, but even for write-dominant workloads the throughput is lower. The extra CPU and network load caused by servicing clients impacts the ability of the leader to coordinate the broadcast of the proposals, which in turn adversely impacts the overall write performance. 
>  Fig6 展示了如果我们将请求都发送到 leader 时的吞吐情况
>  可以看到，对于以读为主的负载，吞吐显著降低，即便是以写为主的负载，吞吐量也较低，这是因为处理客户端请求而增加的 CPU 和网络负载会影响 leader 协调 proposals 的广播的能力，进而影响了整体写入性能

![[pics/ZooKeeper-Fig7.png]]

The atomic broadcast protocol does most of the work of the system and thus limits the performance of ZooKeeper more than any other component. Figure 7 shows the throughput of the atomic broadcast component. To benchmark its performance we simulate clients by generating the transactions directly at the leader, so there is no client connections or client requests and replies. 
>  原子广播协议承担了系统的大部分工作，因此是限制 ZooKeeper 性能最大的组件
>  Fig7 展示了原子广播组件的吞吐，为了测量该性能，我们在 leader 直接生成 transactions 来模拟客户端请求，故测试时没有实际的客户端连接或客户端请求和回复

At maximum throughput the atomic broadcast component becomes CPU bound. In theory the performance of Figure 7 would match the performance of ZooKeeper with $100\%$ writes. However, the ZooKeeper client communication, ACL checks, and request to transaction conversions all require CPU. The contention for CPU lowers ZooKeeper throughput to substantially less than the atomic broadcast component in isolation. 
>  在最大吞吐量下，原子广播组件的性能受 CPU 限制，在理论上，Fig7 展示的性能应该与 ZooKeeper 在 100% 写入时的性能相匹配
>  但 ZooKeeper 客户端通信、访问控制列表检查以及 request 到 transaction 的转换都需要 CPU 处理，故对 CPU 的竞争会降低 ZooKeeper 的吞吐量，使得实际中的吞吐远低于单独运行原子广播组件时的表现

Because ZooKeeper is a critical production component, up to now our development focus for ZooKeeper has been correctness and robustness. There are plenty of opportunities for improving performance significantly by eliminating things like extra copies, multiple serializations of the same object, more efficient internal data structures, etc. 
>  因为 ZooKeeper 是一个关键的生产组件，目前为止，我们的开发中心一直在正确性和健壮性上
>  通过消除额外拷贝、同一对象的多次序列化、使用更高效的内部数据结构等多个方法都有机会提高性能

![[pics/ZooKeeper-Fig8.png]]

To show the behavior of the system over time as failures are injected we ran a ZooKeeper service made up of 5 machines. We ran the same saturation benchmark as before, but this time we kept the write percentage at a constant $30\%$ , which is a conservative ratio of our expected workloads. Periodically we killed some of the server processes. 
>  为了展示系统随着时间推移出现故障时的行为，我们运行了和之前相同的饱和测试，将写入比例保持在 30%，这是一个相对于期望较保守的工作负载比值，我们周期性杀死一些 server 进程

Figure 8 shows the system throughput as it changes over time. The events marked in the figure are the following: 
>  Fig 8 展示了系统吞吐量随时间的变化情况，图中的标记事件如下

1. Failure and recovery of a follower; 
2. Failure and recovery of a different follower; 
3. Failure of the leader; 
4. Failure of two followers (a, b) in the first two marks, and recovery at the third mark (c); 
5. Failure of the leader. 
6. Recovery of the leader. 

There are a few important observations from this graph. First, if followers fail and recover quickly, then ZooKeeper is able to sustain a high throughput despite the failure. The failure of a single follower does not prevent servers from forming a quorum, and only reduces throughput roughly by the share of read requests that the server was processing before failing. 
>  根据 Fig8
>  其一，如果 followers 故障并且快速回复，则 ZooKeeper 仍可以维持较高的吞吐量，单个 follower 的故障不会妨碍 ZooKeeper 达成一致，只会大致减少该 server 在故障前处理的读请求的比例

Second, our leader election algorithm is able to recover fast enough to prevent throughput from dropping substantially. In our observations, ZooKeeper takes less than $200m s$ to elect a new leader. Thus, although servers stop serving requests for a fraction of second, we do not observe a throughput of zero due to our sampling period, which is on the order of seconds. 
>  其二，我们的 leader election 算法可以迅速恢复，防止吞吐大幅下降
>  在图中，ZooKeeper 花费 200ms 选举出新 leader，因此，尽管 servers 在短时间内停止服务请求，但由于我们的采样周期是以秒为单位的，故没有观察到吞吐为零的情况

Third, even if followers take more time to recover, ZooKeeper is able to raise throughput again once they start processing requests. One reason that we do not recover to the full throughput level after events 1, 2, and 4 is that the clients only switch followers when their connection to the follower is broken. Thus, after event 4 the clients do not redistribute themselves until the leader fails at events 3 and 5. In practice such imbalances work themselves out over time as clients come and go. 
>  其三，即便 followers 花费更长时间恢复，一旦它们开始处理请求，ZooKeeper 就可以恢复吞吐量

## 5.2 Latency of requests 
To assess the latency of requests, we created a benchmark modeled after the Chubby benchmark [6]. We create a worker process that simply sends a create, waits for it to finish, sends an asynchronous delete of the new node, and then starts the next create. We vary the number of workers accordingly, and for each run, we have each worker create 50,000 nodes. We calculate the throughput by dividing the number of create requests completed by the total time it took for all the workers to complete. 
>  为了评估请求延迟，我们基于 Chubby benchmark 创建 benchmark
>  我们生成一个工作进程，该进程仅发送创建请求，等待其完成，然后异步删除新创建的节点，并开始下一个创建操作
>  我们根据需要调整工作进程的数量，在每次运行中，每个工作进程会创建 50,000 个节点
>  我们通过将所有工作进程完成创建请求的总数除以总耗时来计算吞吐量

![[pics/ZooKeeper-Table2.png]]

Table 2 show the results of our benchmark. The create requests include 1K of data, rather than 5 bytes in the Chubby benchmark, to better coincide with our expected use. Even with these larger requests, the throughput of ZooKeeper is more than 3 times higher than the published throughput of Chubby. 
>  表 2 显示了我们的基准测试结果
>  创建请求包含 1KB 的数据，而不是 Chubby benchmark 中的 5 字节，以更好地符合我们预期的使用场景
>  即使在这些较大的请求下，ZooKeeper 的吞吐量仍比 Chubby 发布的吞吐量高出 3 倍以上

The throughput of the single ZooKeeper worker benchmark indicates that the average request latency is $1.2\mathrm{ms}$ for three servers and $1.4\mathrm{ms}$ for 9 servers. 
>  单个 ZooKeeper worker benchmark 的吞吐量表明，对于三台服务器的集群，平均请求延迟为 $1.2\mathrm{ms}$ (1/776)，而对于九台服务器，则为 $1.4\mathrm{ms}$ (1/711)。

## 5.3 Performance of barriers 
In this experiment, we execute a number of barriers sequentially to assess the performance of primitives implemented with ZooKeeper. 
>  在这项实验中，我们依次执行多个barriers，以评估使用 ZooKeeper 实现的原语的性能

For a given number of barriers $b$ , each client first enters all $b$ barriers, and then it leaves all $b$ barriers in succession. As we use the double-barrier algorithm of Section 2.4, a client first waits for all other clients to execute the enter() procedure before moving to next call (similarly for $\mathsf{1e a v e()}$ ). 
>  对于给定数量的屏障 $b$，每个客户端首先进入所有 $b$ 个屏障，然后按顺序离开所有 $b$ 个屏障
>  由于我们使用的是第 2.4 节中的双屏障算法，客户端在执行下一个调用之前，必须等待其他所有客户端完成 `enter()` 操作，`leave()` 操作也类似

<html><body><table><tr><td></td><td colspan="3">#ofclients</td></tr><tr><td>#ofbarriers</td><td>50</td><td>100</td><td>200</td></tr><tr><td>200</td><td>9.4</td><td>19.8</td><td>41.0</td></tr><tr><td>400</td><td>16.4</td><td>34.1</td><td>62.0</td></tr><tr><td>800</td><td>28.9</td><td>55.9</td><td>112.1</td></tr><tr><td>1600</td><td>54.0</td><td>102.7</td><td>234.4</td></tr></table></body></html> 

Table 3: Barrier experiment with time in seconds. Each point is the average of the time for each client to finish over five runs. 

We report the results of our experiments in Table 3. In this experiment, we have 50, 100, and 200 clients entering a number $b$ of barriers in succession, $b\in$ $\{200,400,800,1600\}$ . 
>  我们在表3中报告了我们的实验结果
>  在这个实验中，我们有 50、100 和 200 个客户端依次进入数量为 $b$ 的屏障，其中 $b\in\{200,400,800,1600\}$ 

Although an application can have thousands of ZooKeeper clients, quite often a much smaller subset participates in each coordination operation as clients are often grouped according to the specifics of the application. 
>  尽管一个应用程序可能有成千上万个 ZooKeeper 客户端，但通常在每次协调操作中只有一小部分客户端参与，因为客户端通常是根据应用程序的具体情况分组的

Two interesting observations from this experiment are that the time to process all barriers increase roughly linearly with the number of barriers, showing that concurrent access to the same part of the data tree did not produce any unexpected delay, and that latency increases proportionally to the number of clients. This is a consequence of not saturating the ZooKeeper service. 
>  从这个实验中可以观察到两个有趣的现象
>  其一，处理所有屏障所需的时间随着屏障数量的增加大致呈线性增长，这表明对数据树同一部分的并发访问并未产生任何意外延迟 (回忆一下 barrier 是通过对数据树中的一个 znode 的并发访问实现的)，其二，延迟与客户端数量成正比，这是因为 ZooKeeper 服务尚未达到饱和状态

In fact, we observe that even with clients proceeding in lock-step, the throughput of barrier operations (enter and leave) is between 1,950 and 3,100 operations per second in all cases. In ZooKeeper operations, this corresponds to throughput values between 10,700 and 17,000 operations per second. 
> 实际上，我们观察到即使客户端以同步方式运行，屏障操作（进入和离开）的吞吐量在所有情况下介于每秒 1,950 到 3,100 次操作之间，在 ZooKeeper 操作中，这相当于每秒 10,700 到 17,000 次操作的吞吐量

As in our implementation we have a ratio of reads to writes of 4:1 ($80\%$ of read operations), the throughput our benchmark code uses is much lower compared to the raw throughput ZooKeeper can achieve (over 40,000 according to Figure 5). This is due to clients waiting on other clients. 
>  在我们的实现中，读写比例为4:1（即 80%的操作是读取），因此基准测试代码使用的吞吐量远低于 ZooKeeper 能够实现的原始吞吐量（根据图 5 显示超过 40,000），这是因为客户端需要等待其他客户端完成操作 (故我们减少了 benchmark 中使用的吞吐量)

# 6 Related work 
ZooKeeper has the goal of providing a service that mitigates the problem of coordinating processes in distributed applications. To achieve this goal, its design uses ideas from previous coordination services, fault tolerant systems, distributed algorithms, and file systems. 
>  ZooKeeper 的目的是提供缓解分布式应用的进程协调的问题
>  为了实现该目标，其设计使用了来自先前的协调服务、容错系统、分布式算法、文件系统的思想

We are not the first to propose a system for the coordination of distributed applications. Some early systems propose a distributed lock service for transactional applications [13], and for sharing information in clusters of computers [19]. More recently, Chubby proposes a system to manage advisory locks for distributed applications [6]. 
>  我们并不是第一个提出分布式应用程序协调系统的
>  一些早期的系统提出了为事务性应用提供分布式锁服务[13]，以及在计算机集群中共享信息[19]
>  最近来说，Chubby 提出了一个用于管理分布式应用程序建议锁的系统[6]

Chubby shares several of the goals of ZooKeeper. It also has a file-system-like interface, and it uses an agreement protocol to guarantee the consistency of the replicas. However, ZooKeeper is not a lock service. It can be used by clients to implement locks, but there are no lock operations in its API. 
>  Chubby 和 ZooKeeper 共享多个目标。它也有类似文件系统的接口，并使用共识协议来保证副本的一致性。但是，ZooKeeper 并不是一个锁服务。客户端可以利用它来实现锁，但其 API 中并没有锁操作。


Unlike Chubby, ZooKeeper allows clients to connect to any ZooKeeper server, not just the leader. ZooKeeper clients can use their local replicas to serve data and manage watches since its consistency model is much more relaxed than Chubby. This enables ZooKeeper to provide higher performance than Chubby, allowing applications to make more extensive use of ZooKeeper. 
>  与 Chubby 不同的是，ZooKeeper 允许客户端连接到任何 ZooKeeper 服务器，而不仅仅是领导者。
>  由于其一致性模型更加宽松，ZooKeeper 客户端 (应该是 servers) 可以使用本地副本来提供数据和服务并管理 watches。这使得 ZooKeeper 能够提供比 Chubby 更高的性能，从而允许应用程序更广泛地使用 ZooKeeper。

There have been fault-tolerant systems proposed in the literature with the goal of mitigating the problem of building fault-tolerant distributed applications. One early system is ISIS [5]. The ISIS system transforms abstract type specifications into fault-tolerant distributed objects, thus making fault-tolerance mechanisms transparent to users. Horus [30] and Ensemble [31] are systems that evolved from ISIS. ZooKeeper embraces the notion of virtual synchrony of ISIS. 
>  容错系统在之前已经被提出，旨在缓解构建容错分布式应用程序的问题
>  一个早期的系统是 ISIS [5]。ISIS 系统将抽象类型规范转换为容错的分布式对象，从而使容错机制对用户透明。Horus [30] 和 Ensemble [31] 是从 ISIS 发展而来的系统。ZooKeeper 接受了 ISIS 中的虚拟同步概念。

Finally, Totem guarantees total order of message delivery in an architecture that exploits hardware broadcasts of local area networks [22]. ZooKeeper works with a wide variety of network topologies which motivated us to rely on TCP connections between server processes and not assume any special topology or hardware features. We also do not expose any of the ensemble communication used internally in ZooKeeper. 
>  Totem 通过利用局域网的硬件广播，在架构上保证消息传递的全序性 [22]
>  ZooKeeper 支持多种网络拓扑结构，故依赖于服务器进程之间的 TCP 连接，并不假设任何特殊的拓扑或硬件特性。我们也没有暴露 ZooKeeper 内部使用的任何组通信。

One important technique for building fault-tolerant services is state-machine replication [26], and Paxos [20] is an algorithm that enables efficient implementations of replicated state-machines for asynchronous systems. We use an algorithm that shares some of the characteristics of Paxos, but that combines transaction logging needed for consensus with write-ahead logging needed for data tree recovery to enable an efficient implementation. 
>  构建容错服务的一项重要技术是状态机复制[26]，而 Paxos[20]是一种能够为异步系统实现高效的状态机复制的算法。
>  我们使用了一种与 Paxos 具有某些相似特性的算法，但结合了用于共识的事务日志记录和用于数据树恢复的预写日志记录，以实现高效的实现。

There have been proposals of protocols for practical implementations of Byzantine-tolerant replicated state-machines [7, 10, 18, 1, 28]. ZooKeeper does not assume that servers can be Byzantine, but we do employ mechanisms such as checksums and sanity checks to catch non-malicious Byzantine faults. Clement et al. discuss an approach to make ZooKeeper fully Byzantine fault-tolerant without modifying the current server code base [9]. To date, we have not observed faults in production that would have been prevented using a fully Byzantine fault-tolerant protocol. [29]. 
>  已经有人提出了实用的拜占庭容错状态机复制协议的方案[7, 10, 18, 1, 28]
>  ZooKeeper 并未假设 servers 可能成为拜占庭故障节点，但我们确实采用了诸如校验和和合理性检查等机制来捕获非恶意的拜占庭故障
>  Clement 等人讨论了一种在不修改现有服务器代码库的情况下使 ZooKeeper 完全具备拜占庭容错能力的方法[9]。迄今为止，我们尚未在生产环境中观察到可以通过完全拜占庭容错协议防止的故障[29]。

Boxwood [21] is a system that uses distributed lock servers. Boxwood provides higher-level abstractions to applications, and it relies upon a distributed lock service based on Paxos. Like Boxwood, ZooKeeper is a component used to build distributed systems. ZooKeeper, however, has high-performance requirements and is used more extensively in client applications. ZooKeeper exposes lower-level primitives that applications use to implement higher-level primitives. 
>  Boxwood [21] 是一个使用分布式锁服务器的系统
>  Boxwood 为应用程序提供了更高层次的抽象，并依赖于基于 Paxos 的分布式锁服务。与 Boxwood 类似，ZooKeeper 是用于构建分布式系统的组件。然而，ZooKeeper 对高性能有更高的要求，并在客户端应用程序中被更广泛地使用
>  ZooKeeper 暴露了较低级别的原语，供应用程序用来实现更高级别的原语。

ZooKeeper resembles a small file system, but it only provides a small subset of the file system operations and adds functionality not present in most file systems such as ordering guarantees and conditional writes. ZooKeeper watches, however, are similar in spirit to the cache callbacks of AFS [16]. 
>  ZooKeeper 类似于一个小文件系统，但它只提供了一小部分文件系统操作，并添加了大多数文件系统中不存在的功能，例如顺序保证和条件写入
>  然而，ZooKeeper 的 watcher 在思想上与 AFS [16] 的缓存回调类似。

Sinfonia [2] introduces mini-transactions, a new paradigm for building scalable distributed systems. Sinfonia has been designed to store application data, whereas ZooKeeper stores application metadata. ZooKeeper keeps its state fully replicated and in memory for high performance and consistent latency. Our use of file system like operations and ordering enables functionality similar to mini-transactions. The znode is a convenient abstraction upon which we add watches, a functionality missing in Sinfonia. 
>  Sinfonia [2] 引入了微型事务（mini-transactions），这是一种构建可扩展分布式系统的全新范式
>  Sinfonia 的设计目的是存储应用程序数据，而 ZooKeeper 则用于存储应用程序的元数据。ZooKeeper 将其状态完全复制并保留在内存中，以实现高性能和一致的延迟
>  我们通过使用类似文件系统操作和排序的方式，实现了类似于微型事务的功能。znode 是一个方便的抽象概念，在此基础上我们添加了 watches 功能，这是 Sinfonia 中缺失的一项功能。

Dynamo [11] allows clients to get and put relatively small (less than 1M) amounts of data in a distributed key-value store. Unlike ZooKeeper, the key space in Dynamo is not hierarchal. Dynamo also does not provide strong durability and consistency guarantees for writes, but instead resolves conflicts on reads. 
>  Dynamo [11] 允许客户端在分布式键值存储中获取和存储相对较小（小于 1MB）的数据量。与 ZooKeeper 不同，Dynamo 的键空间不是分层结构。此外，Dynamo 并未为写入操作提供强持久性和一致性保证，而是在读取时解决冲突

DepSpace [4] uses a tuple space to provide a Byzantine fault-tolerant service. Like ZooKeeper DepSpace uses a simple server interface to implement strong synchronization primitives at the client. While DepSpace’s performance is much lower than ZooKeeper, it provides stronger fault tolerance and confidentiality guarantees. 
>  DepSpace [4] 使用元组空间来提供拜占庭容错服务
>  与 ZooKeeper 类似，DepSpace 通过简单的服务器接口在客户端实现强大的同步原语。尽管 DepSpace 的性能远低于 ZooKeeper，但它提供了更强的容错性和保密性保证。

# 7 Conclusions 
ZooKeeper takes a wait-free approach to the problem of coordinating processes in distributed systems, by exposing wait-free objects to clients. 
>  ZooKeeper 通过向客户端暴露无等待（的对象，以一种无等待的方式解决了分布式系统中进程协调的问题

We have found ZooKeeper to be useful for several applications inside and outside Yahoo!. ZooKeeper achieves throughput values of hundreds of thousands of operations per second for read-dominant workloads by using fast reads with watches, both of which served by local replicas.
>  我们发现 ZooKeeper 在雅虎内外的多个应用场景中都非常有用
>  对于以读为主的负载，ZooKeeper 可以通过使用快速读取和 watches (二者都由本地副本提供服务)，实现每秒数十万次操作的吞吐量

 Although our consistency guarantees for reads and watches appear to be weak, we have shown with our use cases that this combination allows us to implement efficient and sophisticated coordination protocols at the client even though reads are not precedence-ordered and the implementation of data objects is wait-free. The wait-free property has proved to be essential for high performance. 
>  尽管我们为读取和 watches 提供的一致性保证看似较弱，但我们通过实际用例表明，即使读取操作不是按优先级顺序执行的，并且数据对象的实现是无等待的，对于二者的一致性保证的组合允许我们在客户端实现高效且复杂的协调协议
>  无等待的特性已被证明对高性能至关重要。

Although we have described only a few applications, there are many others using ZooKeeper. We believe such a success is due to its simple interface and the powerful abstractions that one can implement through this interface. Further, because of the high-throughput of ZooKeeper, applications can make extensive use of it, not only course-grained locking. 
>  尽管我们只描述了少数几个应用场景，但还有许多其他应用正在使用 ZooKeeper
>  我们认为这种成功归因于其简洁的接口以及通过该接口可以实现的强大抽象功能
>  此外，由于 ZooKeeper 具有高吞吐量的特点，应用程序不仅可以用它实现粗粒度分布式锁，还可以广泛地利用它
