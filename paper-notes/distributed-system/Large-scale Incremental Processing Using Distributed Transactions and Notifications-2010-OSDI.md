# Abstract 
Updating an index of the web as documents are crawled requires continuously transforming a large repository of existing documents as new documents arrive. This task is one example of a class of data processing tasks that transform a large repository of data via small, independent mutations. These tasks lie in a gap between the capabilities of existing infrastructure. Databases do not meet the storage or throughput requirements of these tasks: Google’s indexing system stores tens of petabytes of data and processes billions of updates per day on thousands of machines. MapReduce and other batch-processing systems cannot process small updates individually as they rely on creating large batches for efficiency. 
>  在爬取网页文档，更新网络索引时，随着新文档被获取，我们需要不断地对现有的大规模文档库进行转换
>  这种类型的数据处理任务的特点是通过小的、独立的变更来转换大规模存储库
>  目前，这类任务尚没有现存基础设施有能力完成，数据库系统不能满足这类任务的存储和吞吐需求 (Google 的索引系统存储了数十个 PB 的数据，并且每天在数千台机器上处理数亿次更新)；MapReduce 和其他批处理系统由于依赖创建大批次以提高效率，无法单独处理小的更新

We have built Percolator, a system for incrementally processing updates to a large data set, and deployed it to create the Google web search index. By replacing a batch-based indexing system with an indexing system based on incremental processing using Percolator, we process the same number of documents per day, while reducing the average age of documents in Google search results by $50\%$ . 
>  我们构建了 Percolator，一个用于增量式处理对大规模数据集更新的系统
>  Percolator 已经被部署，用于创建 Google 网络搜索索引
>  Google 用基于 Percolator 的增量处理构建的索引系统替换了基于批量的索引系统之后，在每天处理相同数量的文档的情况下，将 Google 搜索结果中文档的平均年龄下降了 50%

# 1 Introduction 
Consider the task of building an index of the web that can be used to answer search queries. The indexing system starts by crawling every page on the web and processing them while maintaining a set of invariants on the index. For example, if the same content is crawled under multiple URLs, only the URL with the highest PageRank [28] appears in the index. Each link is also inverted so that the anchor text from each outgoing link is attached to the page the link points to. Link inversion must work across duplicates: links to a duplicate of a page should be forwarded to the highest PageRank duplicate if necessary. 
>  考虑一个任务：构建一个网络索引，用以回答搜索请求
>  为此，在开始时，索引系统需要爬取网络中的每个页面并处理它们，同时维护索引的一组不变式 (构建索引的规则)
>  例如，如果相同的内容通过多个 URLs 被获取，则只有具有最高 PageRank 的 URL 应该出现在索引中；每个链接需要进行反向处理，即每个出站链接的锚文本需要被附加到该链接指向的页面上；链接的反向处理必须跨重复项进行：如果必要的话，指向某个页面副本的链接应修改为指向该页面的具有最高 PageRank 副本的链接

>  反向处理的原因在于锚文本是对目标页面的描述，因此认为锚文本的信息对关于目标页面的搜索是有益的，因此将锚文本附加到目标页面上有助于提高搜索性能

This is a bulk-processing task that can be expressed as a series of MapReduce [13] operations: one for clustering duplicates, one for link inversion, etc. It’s easy to maintain invariants since MapReduce limits the parallelism of the computation; all documents finish one processing step before starting the next. For example, when the indexing system is writing inverted links to the current highest-PageRank URL, we need not worry about its PageRank concurrently changing; a previous MapReduce step has already determined its PageRank. 
>  初始的索引构建任务可以视作由一系列 MapReduce 操作表达的批处理任务：一次 MapReduce 进行重复项聚类，一次 MapReduce 进行链接反转
>  因为 MapReduce 限制了计算的并行性 (所有的文档在开始下一步处理步骤之前都必须完成当前的处理步骤)，因此较为容易维护不变式
>  例如，当索引系统正在向当前具有最高 PageRank 的 URL 写入反向链接时，我们无需担心其 PageRank 同时在发生变化，因为先前的 MapReduce 步骤已经确定了它的 PageRank

Now, consider how to update that index after recrawling some small portion of the web. It’s not sufficient to run the MapReduces over just the new pages since, for example, there are links between the new pages and the rest of the web. The MapReduces must be run again over the entire repository, that is, over both the new pages and the old pages. Given enough computing resources, MapReduce’s scalability makes this approach feasible, and, in fact, Google’s web search index was produced in this way prior to the work described here. However, reprocessing the entire web discards the work done in earlier runs and makes latency proportional to the size of the repository, rather than the size of an update. 
>  完成了初始的索引构建后，我们开始考虑在重新抓取了一小部分网络中的新网页后，如何更新索引
>  仅仅对新页面运行 MapReduce 是不够的，因为新页面和旧页面之间可能会存在链接，因此必须对整个存储库 (包含了新页面和旧页面) 再次运行 MapReduce
>  如果有足够的计算资源，MapReduce 的可拓展性使得这一方法是可行的，并且实际上，在这一工作之前，Google 的网络搜索索引就是由这种方式生成的
>  显然，重新处理整个网络的资源就浪费了之前处理的结果，并且使得更新的延迟正比于整个存储库的大小，而不是更新的大小

The indexing system could store the repository in a DBMS and update individual documents while using transactions to maintain invariants. However, existing DBMSs can’t handle the sheer volume of data: Google’s indexing system stores tens of petabytes across thousands of machines [30]. Distributed storage systems like Bigtable [9] can scale to the size of our repository but don’t provide tools to help programmers maintain data invariants in the face of concurrent updates. 
>  一种办法是，索引系统将存储库存储在 DBMS 中，使用事务来更新单个文档，同时维护不变式
>  但是，现存的 DBMS 无法处理如此庞大的数据量：Google 的索引系统在上千台机器中存储了数十 PB 的数据
>  像 Bigtable 这样的分布式存储系统可以拓展到这样的规模，**但在面对并发更新时，并没有为程序员提供工具，以维护数据不变式**

An ideal data processing system for the task of maintaining the web search index would be optimized for incremental processing; that is, it would allow us to maintain a very large repository of documents and update it efficiently as each new document was crawled. Given that the system will be processing many small updates concurrently, an ideal system would also provide mechanisms for maintaining invariants despite concurrent updates and for keeping track of which updates have been processed. 
>  一个理想的维护网络搜索索引的数据处理系统应该针对**增量处理**进行优化，也就是说，它可以在维护非常大的文档库的同时，随着新文档的到来执行高效的更新
>  考虑到系统将需要并发处理多个小型更新，理想的系统应该还能够**提供一种在并发更新下维护不变式，并跟踪哪些更新已经被处理的机制**

The remainder of this paper describes a particular incremental processing system: Percolator. Percolator provides the user with random access to a multi-PB repository. Random access allows us to process documents individually, avoiding the global scans of the repository that MapReduce requires. To achieve high throughput, many threads on many machines need to transform the repository concurrently, so Percolator provides ACIDcompliant transactions to make it easier for programmers to reason about the state of the repository; we currently implement snapshot isolation semantics [5]. 
>  本文将描述 Percolator 增量更新系统
>  Percolator 为用户提供了对 PB 级存储库的随机访问，随机访问使得我们可以独立处理文档，避免了 MapReduce 所需的对存储库的全局扫描
>  为了实现高吞吐，多台机器上的多线程需要并发地对存储库进行转换，故 Percolator 提供了符合 ACID 的事务，方便程序员推断存储库的状态
>  目前，我们实现了快照隔离语义

In addition to reasoning about concurrency, programmers of an incremental system need to keep track of the state of the incremental computation. To assist them in this task, Percolator provides observers: pieces of code that are invoked by the system whenever a user-specified column changes. Percolator applications are structured as a series of observers; each observer completes a task and creates more work for “downstream” observers by writing to the table. An external process triggers the first observer in the chain by writing initial data into the table. 
>  增量更新系统的程序员除了要分析并发问题以外，还需要追踪增量计算的状态
>  为此，Percolator 提供了 observers, observers 是一些代码片段，每当用户指定的列发生变化时，observers 就会被调用
>  Percolator 应用被组织为一系列 observers，每个 observer 完成一个任务，并通过向表格中写入数据，来为 “下游” observers 创建更多工作
>  外部的应用通过向表格中写入初始数据，来触发 observers chain 中的第一个 observer

Percolator was built specifically for incremental processing and is not intended to supplant existing solutions for most data processing tasks. Computations where the result can’t be broken down into small updates (sorting a file, for example) are better handled by MapReduce. Also, the computation should have strong consistency requirements; otherwise, Bigtable is sufficient. Finally, the computation should be very large in some dimension (total data size, CPU required for transformation, etc.); smaller computations not suited to MapReduce or Bigtable can be handled by traditional DBMSs. 
>  Percolator 是专门为增量处理而构建的，并不打算取代大多数数据处理任务的现存解决方案
>  不能被分解为小型更新的计算任务 (例如对文件排序) 更适合用 MapReduce 处理
>  不具有强一致性要求的计算任务更适合用 Bigtable 处理
>  不适合 MapReduce 和 Bigtable 的较小的计算更适合用传统的 DBMS 来处理
>  因此，Percolator 针对的是在某一维度上 (如总数据量，转换所需求的 CPU 核心数等) 非常庞大、具有强一致性要求、能够分解为小型更新的计算任务

Within Google, the primary application of Percolator is preparing web pages for inclusion in the live web search index. By converting the indexing system to an incremental system, we are able to process individual documents as they are crawled. This reduced the average document processing latency by a factor of 100, and the average age of a document appearing in a search result dropped by nearly 50 percent (the age of a search result includes delays other than indexing such as the time between a document being changed and being crawled). The system has also been used to render pages into images; Percolator tracks the relationship between web pages and the resources they depend on, so pages can be reprocessed when any depended-upon resources change. 
>  在 Google 中，Precolator 的主要应用是纳入新的网页，实时更新网络索引
>  Percolator 将 Google 的索引系统转化为了增量系统，进而可以随着文档被爬取而逐个处理它们，这将平均的文档处理延迟降低了 100 倍，并将出现在搜索结果中的文档的平均年龄降低了近 50% (一个搜索结果的年龄除了包含索引延迟以外，还包含了文档被更改与被爬取之间的时间等其他延迟)
>  索引系统还被用于将页面渲染为图像，Percolator 追踪网页之间的关系，以及它们依赖的资源，因此当它们依赖的资源改变时，页面可以被重新处理

# 2 Design 
Percolator provides two main abstractions for performing incremental processing at large scale: ACID transactions over a random-access repository and observers, a way to organize an incremental computation. 
>  Percolator 为执行大规模增量处理提供了两个抽象
>  - 随机访问存储库上的 ACID 事务
>  - observers (一种组织增量计算的方式)

![[pics/Percolator-Fig1.png]]

A Percolator system consists of three binaries that run on every machine in the cluster: a Percolator worker, a Bigtable [9] tablet server, and a GFS [20] chunkserver. All observers are linked into the Percolator worker, which scans the Bigtable for changed columns (“notifications”) and invokes the corresponding observers as a function call in the worker process. The observers perform transactions by sending read/write RPCs to Bigtable tablet servers, which in turn send read/write RPCs to GFS chunkservers.
>  Percolator 系统由三个在集群中所有机器上运行的二进制文件构成: a Percolator worker, a Bigtable tablet server, a GFS chunkserver
>  所有的 observers 都链接到 Percolator worker, Percolator worker 扫描 Bigtable 中更改的列 ("notifications")，然后以函数调用的形式，调用对应的 observers
>  observers 通过向 Bigtable tablet server 发送读写 RPCs 来执行事务，Bigtable tablet server 接收到这些 RPCs，转而向 GFS chunkservers 发送读写 RPCs

The system also depends on two small services: the timestamp oracle and the lightweight lock service. The timestamp oracle provides strictly increasing timestamps: a property required for correct operation of the snapshot isolation protocol. Workers use the lightweight lock service to make the search for dirty notifications more efficient. 
>  Percolator 系统还依赖于两个小服务：timestamp oracle 和轻量锁服务
>  timestamp oracle 服务提供了严格递增的时间戳，这是 snapshot isolation 协议正确运行所需的性质
>  Percolator worker 使用轻量锁服务来提高 (在 Bigtable 中) 查找 dirty notification 的效率

From the programmer’s perspective, a Percolator repository consists of a small number of tables. Each table is a collection of “cells” indexed by row and column. Each cell contains a value: an uninterpreted array of bytes. (Internally, to support snapshot isolation, we represent each cell as a series of values indexed by timestamp.) 
>  程序员视角下，Percolator 存储库由少量的表组成，每张表都是由行和列索引的 “cell” 的集合
>  每个 cell 包含一个值：一个未解释的字节数组 (在内部，为了支持 snapshot isolation，我们将每个 cell 表示为一组按时间戳索引的值)

The design of Percolator was influenced by the requirement to run at massive scales and the lack of a requirement for extremely low latency. Relaxed latency requirements let us take, for example, a lazy approach to cleaning up locks left behind by transactions running on failed machines. This lazy, simple-to-implement approach potentially delays transaction commit by tens of seconds. This delay would not be acceptable in a DBMS running OLTP tasks, but it is tolerable in an incremental processing system building an index of the web. Percolator has no central location for transaction management; in particular, it lacks a global deadlock detector. This increases the latency of conflicting transactions but allows the system to scale to thousands of machines. 
>  Percolator 的设计受到了对大规模运行的需求和对极低延迟的需求不高的影响
>  松弛的延迟需求允许我们采取 lazy 方法来清理在故障机器上运行的事务所遗留下的锁，这种 lazy 的简单方法可能会将事务提交延迟数十秒，这样的延迟在运行 OLTP 任务的 DBMS 中是不可接收的，但对于构建网络索引的增量处理系统则可以容忍
>  Percolator 没有中心化的事务管理，并且没有全局死锁检测器，这增加了冲突事务将导致的延迟时间，但允许系统拓展到数千台机器

> [!info] OLTP 实时事务处理
> OLTP (Online Transaction Processing) 是一种用于管理事务处理的计算系统，旨在支持日常业务操作中的事务处理任务。OLTP 系统通常处理大量的短事务，强调数据的**实时性**和一致性

## 2.1 Bigtable overview 
Percolator is built on top of the Bigtable distributed storage system. Bigtable presents a multi-dimensional sorted map to users: keys are (row, column, timestamp) tuples. Bigtable provides lookup and update operations on each row, and Bigtable row transactions enable atomic read-modify-write operations on individual rows. Bigtable handles petabytes of data and runs reliably on large numbers of (unreliable) machines. 
>  Percolator 基于 Bigtable 分布式存储系统构建
>  Bigtable 为用户呈现了多维的有序映射，其中 keys 是 (row, column, timestamp) 数组 (row, column 定位 cell, timestamp 定位 cell 中的 value)
>  Bigtable 提供了对每一行的 lookup 和 update 操作，并且 Bigtable 行事务允许对单独的行进行原子化的 read-modify-write 操作
>  Bigtable 可以处理 PB 级的数据，并且可以在大量 (不可靠的) 机器上可靠地运行

A running Bigtable consists of a collection of tablet servers, each of which is responsible for serving several tablets (contiguous regions of the key space). A master coordinates the operation of tablet servers by, for example, directing them to load or unload tablets. 
>  运行中的 Bigtable 由一组 tablet servers 组成，每个 tablet server 负责管理多个 tablets (tablet 即 key 空间的连续区域)
>  master 通过指导 tablet server 加载或卸载 tablets 来协调 tablet servers 的操作

A tablet is stored as a collection of read-only files in the Google SSTable format. SSTables are stored in GFS; Bigtable relies on GFS to preserve data in the event of disk loss. Bigtable allows users to control the performance characteristics of the table by grouping a set of columns into a locality group. The columns in each locality group are stored in their own set of SSTables, which makes scanning them less expensive since the data in other columns need not be scanned. 
>  tablet 以一组 Google SSTable 格式的只读文件存储，SSTable 则存储在 GFS 中，Bigtable 依赖 GFS 在磁盘故障的情况下仍保持数据可用
>  Bigtable 允许用户将多个列组为一个 locality group 来控制表的性能特性，每个 locality group 的列都存储在它自己的一组 SSTables 中，这使得扫描这些列的成本更低，因为不需要扫描其他列的数据

>  即表中的多个列构成一个 locality group，扫描特定列时，先定位其 locality group，在找到对应的 SSTables，对其进行扫描

The decision to build on Bigtable defined the overall shape of Percolator. Percolator maintains the gist of Bigtable’s interface: data is organized into Bigtable rows and columns, with Percolator metadata stored alongside in special columns (see Figure 5). Percolator’s API closely resembles Bigtable’s API: the Percolator library largely consists of Bigtable operations wrapped in Percolator-specific computation. The challenge, then, in implementing Percolator is providing the features that Bigtable does not: multirow transactions and the observer framework. 
>  Percolator 基于 Bigtable 构建的决定定义了 Percolator 的整体结构
>  Percolator 保留了 Bigtable 接口的核心: 数据以 Bigtable 的行和列组织，Percolator 元数据则存储在特殊的列中
>  Percolator 的 API 和 Bigtable 的 API 非常相似: Percolator 库主要由封装了 Percolator 特定计算的 Bigtable 操作组成
>  因此，实现 Percolator 的挑战在于提供 Bigtable 并未提供的特性: 多行事务和 observer 框架

## 2.2 Transactions 
Percolator provides cross-row, cross-table transactions with ACID snapshot-isolation semantics. 
>  Percolator 提供了具有 ACID snapshot-isolation 语义的跨行、跨表事务

Percolator users write their transaction code in an imperative language (currently C++,) and mix calls to the Percolator API with their code. 
>  Percolator 用户使用命令式语言 (目前为 C++) 编写其事务代码，并调用 Percolator API

![[pics/Percolator-Fig2.png]]

Figure 2 shows a simplified version of clustering documents by a hash of their contents. 
>  Fig2 展示了通过内容哈希对文档进行聚类的示例

In this example, if Commit() returns false, the transaction has conflicted (in this case, because two URLs with the same content hash were processed simultaneously) and should be retried after a backoff. 
>  在示例中，如果 `Commit()` 返回 `false` ，说明事务发生了冲突 (本例中，冲突是因为同时处理了具有相同内容哈希的两个 URL)，应在退避后重试

Calls to Get() and Commit() are blocking; parallelism is achieved by running many transactions simultaneously in a thread pool. 
>  对 `Commit(), Get()` 的调用是阻塞式的，并行性通过在线程池中同时运行多个事务来实现

>  Fig2 的 `UpdateDocument` 函数的作用主要在于为文档的哈希确定规范化 URL
>  函数先初始化了一个事务 `Transaction t`，然后开始设定文档的 URL 相关信息
>  `t.Get()` 尝试获取文档哈希是否已经存在规范化 URL，如果存在，说明文档是重复的，不需要再写入，否则，将当前文档的 URL 设置为文档哈希的规范化 URL

While it is possible to incrementally process data without the benefit of strong transactions, transactions make it more tractable for the user to reason about the state of the system and to avoid the introduction of errors into a long-lived repository.
>  虽然在没有强事务的情况下也可以增量地处理数据，但定义事务可以让用户更容易推断系统的状态，并避免将错误引入长期维护的存储库中

For example, in a transactional web-indexing system the programmer can make assumptions like: the hash of the contents of a document is always consistent with the table that indexes duplicates. Without transactions, an ill-timed crash could result in a permanent error: an entry in the document table that corresponds to no URL in the duplicates table. 
>  例如，在一个事务化的网页索引系统中，程序员可以假设: 文档的哈希值始终与用于索引重复项的表中的哈希值保持一致
>  如果索引系统没有实现事务，则时机不当的崩溃可能导致永久性的错误: 文档表中存在一个与重复表中没有任何 URL 对应的条目

Transactions also make it easy to build index tables that are always up to date and consistent. 
>  事务也使得构建始终保持最新且一致的索引表更加容易

Note that both of these examples require transactions that span rows, rather than the single-row transactions that Bigtable already provides. 
>  注意，这两个示例都需要跨行的事务，而不是 Bigtable 已经提供的单行事务

Percolator stores multiple versions of each data item using Bigtable’s timestamp dimension. Multiple versions are required to provide snapshot isolation [5], which presents each transaction with the appearance of reading from a stable snapshot at some timestamp. Writes appear in a different, later, timestamp. 
>  Percolator 使用 Bigtable 的 timestamp 维度存储每个数据项的不同版本
>  为了提供 snapshot isolation，维护多个版本的数据项是必须的，
>  在 snapshot isolation 中，每个事务都可以视作先从某个 timestamp 上的稳定的 snapshot 读数据，再在之后一个不同的 timestamp 上执行写操作

Snapshot isolation protects against write-write conflicts: if transactions A and B, running concurrently, write to the same cell, at most one will commit. Snapshot isolation does not provide serializability; in particular, transactions running under snapshot isolation are subject to write skew [5]. 
>  snapshot isolation 防止了 write-write 冲突: 如果并发的事务 A, B 对同一个单元格写入，则最多只有一个事务能够提交
>  snapshot isolation 不提供可序列化保证，特别是，snapshot isolation 下的事务可能受到写偏斜的影响

![[pics/Percolator-Fig3.png]]

The main advantage of snapshot isolation over a serializable protocol is more efficient reads. 
>  snapshot isolation 相较于一个可序列化协议的主要优势在于其读操作更高效

Because any timestamp represents a consistent snapshot, reading a cell requires only performing a Bigtable lookup at the given timestamp; acquiring locks is not necessary. Figure 3 illustrates the relationship between transactions under snapshot isolation.
>  这是因为任意 timestamp 都代表一个一致的 snapshot，所以读取一个 cell 仅需要在给定的 timestamp 上执行一次 Bigtable lookup，无需获取锁
>  Fig3 描述了 snapshot isolation 下事务之间的关系

Because it is built as a client library accessing Bigtable, rather than controlling access to storage itself, Percolator faces a different set of challenges implementing distributed transactions than traditional PDBMSs. 
>  由于 Percolator 是作为访问 Bigtable 的客户端库构建的，而不是自己直接控制存储访问，故 Percolator 在实现分布式事务时面临与传统 PDBMS 不同的挑战

Other parallel databases integrate locking into the system component that manages access to the disk: since each node already mediates access to data on the disk it can grant locks on requests and deny accesses that violate locking requirements. 
>  其他的并行数据库将锁集成到管理磁盘访问的系统组件中: 由于每个节点已经负责管理对磁盘的访问，故它可以响应锁请求 (授予锁) 以及拒绝违反了锁要求的访问

By contrast, any node in Percolator can (and does) issue requests to directly modify state in Bigtable: there is no convenient place to intercept traffic and assign locks. As a result, Percolator must explicitly maintain locks. 
>  相较之下，Percolator 中的任意节点都可以 (并且确实会) 向 Bigtable 发出直接修改状态的请求: 不存在可以可以拦截流量并分配锁的情况
>  因此，Percolator 必须显式地维护锁 (保证 Bigtable 原子地修改多行状态)

>  也就是 Percolator 为了基于 Bigtable 定义多行事务，需要自己显式地维护锁

Locks must persist in the face of machine failure; if a lock could disappear between the two phases of commit, the system could mistakenly commit two transactions that should have conflicted. The lock service must provide high throughput; thousands of machines will be requesting locks simultaneously. The lock service should also be low-latency; each Get() operation requires reading locks in addition to data, and we prefer to minimize this latency. 
>  锁必须在机器故障的情况下被持续保存，如果在提交的两个阶段之间锁消失，系统可能会错误地提交两个本应冲突的事务
>  锁服务必须提供高吞吐，因为将会有数千台机器同时请求锁
>  锁服务还应具有低延迟，每次 `Get()` 操作除了读取数据外，还需要额外读取锁，故我们期望最小化这一延迟

Given these requirements, the lock server will need to be replicated (to survive failure), distributed and balanced (to handle load), and write to a persistent data store. 
>  考虑到这些要求，锁服务需要被复制 (以应对故障)、需要是分布式且进行负载均衡 (以处理负载)、需要写入持久化数据存储

Bigtable itself satisfies all of our requirements, and so Percolator stores its locks in special in-memory columns in the same Bigtable that stores data and reads or modifies the locks in a Bigtable row transaction when accessing data in that row. 
>  Bigtable 本身满足了上述所有要求，故 Percolator 将它的锁都存储在了存储了数据的 Bigtable 中的特定的存内列中，在访问该 Bigtable 中的行数据时，行事务将读取并修改对应的锁

>  Percolator 依然基于 Bigtable 定义自己的锁服务，将锁存储在 Bigtable 中的列上

>  Fig4 (important) ref to original pdf

![[pics/Percolator-Fig5.png]]

We’ll now consider the transaction protocol in more detail. 
>  我们开始更详细地考虑事务协议

Figure 6 shows the pseudocode for Percolator transactions, and Figure 4 shows the layout of Percolator data and metadata during the execution of a transaction. These various metadata columns used by the system are described in Figure 5. 
>  Fig6 展示了 Percolator 事务的伪代码，Fig4 展示了在事务执行过程中 Percolator 数据和元数据的布局，Fig5 描述了系统使用的各种元数据列

```cpp
1 class Transaction { 
2  struct Write { Row row; Column col; string value; }; 
3  vector<Write> writes_; // 事务涉及到的 Writes
4  int start_ts_; 
5 
6  Transaction() : start_ts_(oracle.GetTimestamp()) {} // 获取起始时间戳
7  void Set(Write w) { writes_.push_back(w); } // Set() 缓存 Writes 直到事务的提交
8  bool Get(Row row, Column c, string* value) { 
9   while (true) { 
10   bigtable::Txn T = bigtable::StartRowTransaction(row); 
11   // Check for locks that signal concurrent writes. 
12   if (T.Read(row, c+"lock", [0, start_ts ])) { 
13    // There is a pending lock; try to clean it and wait 
14    BackoffAndMaybeCleanupLock(row, c); 
15    continue; 
16   } 
17 
18   // Find the latest write below our start timestamp. 
19   latest_write = T.Read(row, c+"write", [0, start_ts ]); 
20   if (!latest write.found()) return false; // no data 
21   int data_ts = latest_write.start_timestamp(); 
22   *value = T.Read(row, c+"data", [data_ts, data_ts]); 
23   return true; 
24  } 
25 } 
26 // Prewrite tries to lock cell w, returning false in case of conflic 
27  bool Prewrite(Write w, Write primary) { 
28   Column c = w.col; 
29   bigtable::Txn T = bigtable::StartRowTransaction(w.row); 
30 
31   // Abort on writes after our start timestamp . 
32   if (T.Read(w.row, c+"write", [start_ts , \infty])) return false; 
33   // . . . or locks at any timestamp. 
34   if (T.Read(w.row, c+"lock", [0, \infty])) return false; 
35 
36   T.Write(w.row, c+"data", start_ts , w.value); 
37   T.Write(w.row, c+"lock", start_ts , 
38    {primary.row, primary.col}); // The primary’s location. 
39   return T.Commit(); 
40  }
41  bool Commit() { 
42   Write primary = writes_[0]; 
43   vector<Write> secondaries(writes_.begin()+1, writes_.end()); 
44   if (!Prewrite(primary, primary)) return false; 
45   for (Write w : secondaries) 
46    if (!Prewrite(w, primary)) return false; 
47 
48    int commit_ts = oracle_.GetTimestamp(); 
49 
50   // Commit primary first. 
51   Write p = primary; 
52   bigtable::Txn T = bigtable::StartRowTransaction(p.row); 
53   if (!T.Read(p.row, p.col+"lock", [start_ts , start_ts ])) 
54    return false; // aborted while working 
55   T.Write(p.row, p.col+"write", commit_ts, 
56    start_ts ); // Pointer to data written at start_ts . 
57   T.Erase(p.row, p.col+"lock", commit_ts); 
58   if (!T.Commit()) return false; // commit point 
59 
60   // Second phase: write out write records for secondary cells. 
61   for (Write w : secondaries) { 
62    bigtable::Write(w.row, w.col+"write", commit_ts, start_ts ); 
63    bigtable::Erase(w.row, w.col+"lock", commit_ts); 
64  } 
65  return true; 
66  } 
67 } // class Transaction 
```

The transaction’s constructor asks the timestamp oracle for a start timestamp (line 6), which determines the consistent snapshot seen by Get(). Calls to Set() are buffered (line 7) until commit time. The basic approach for committing buffered writes is two-phase commit, which is coordinated by the client. Transactions on different machines interact through row transactions on Bigtable tablet servers. 
>  事务的构造函数会请求 timestamp oracle，获取一个起始时间戳 (line 6)，起始时间戳决定了 `Get()` 方法 (读操作)所看到的一致性快照
>  对 `Set()` 的调用 (写操作) 会被缓存 (line 7) 直到提交时，提交缓存的写操作的基本方法是两阶段提交，由客户端协调
>  不同机器上的事务通过 Bigtable tablet servers 上的行事务交互

In the first phase of commit (“prewrite”), we try to lock all the cells being written. (To handle client failure, we designate one lock arbitrarily as the primary; we’ll discuss this mechanism below.) 
>  在事务提交的第一阶段 (prewrite)，我们尝试锁住所有需要被写入的 cells (为了处理客户端故障，我们任意指定一个锁为 primary)

>  锁住需要被写入的 cell 的方法就是在该 cell 的这一行中找到 `lock` 这一列，然后在当前事务的起始时间戳上写入锁

The transaction reads metadata to check for conflicts in each cell being written. There are two kinds of conflicting metadata: if the transaction sees another write record after its start timestamp, it aborts (line 32); this is the write-write conflict that snapshot isolation guards against. If the transaction sees another lock at any timestamp, it also aborts (line 34). It’s possible that the other transaction is just being slow to release its lock after having already committed below our start timestamp, but we consider this unlikely, so we abort. If there is no conflict, we write the lock and the data to each cell at the start timestamp (lines 36-38). 
>  事务会读取元数据，以检查各个 cell 是否存在写入冲突，有两类元数据冲突:
>  - 如果事务看到了在其开始时间戳后的写入记录 (line 32)，则说明存在 write-write 冲突，为了维护 snapshot isolation，事务会中止
>  - 如果事务看到了在任何时间戳上的另一个锁，事务也会中止，虽然这有可能是其他事务在低于当前事务的起始时间戳下的时间戳上提交后，比较缓慢地释放锁，但我们认为这种情况不太可能，因此安全起见还是需要中止
>  如果没有冲突，我们将锁和数据写入各个 cell 中，当前事务的起始时间戳上

>  prewrite 阶段 (提交的第一阶段) 负责获取锁并写入数据

If no cells conflict, the transaction may commit and proceeds to the second phase. At the beginning of the second phase, the client obtains the commit timestamp from the timestamp oracle (line 48). Then, at each cell (starting with the primary), the client releases its lock and make its write visible to readers by replacing the lock with a write record. The write record indicates to readers that committed data exists in this cell; it contains a pointer to the start timestamp where readers can find the actual data. Once the primary’s write is visible (line 58), the transaction must commit since it has made a write visible to readers. 
>  在 prewrite 阶段，如果没有冲突，事务就可以提交并进入第二阶段 (`Prewrite` 函数的 line 39，调用 `T.Commit()`)
>  在第二阶段开始时，客户端从 timestamp oracle (line 48) 获取提交时间戳，然后，客户端在每个 cell (从 primary 开始) 释放其锁，并将锁替换为写入记录，使得该写入开始对读者可见
>  写入记录向读者表明该 cell 存在已经提交的写入数据，写入记录包含了一个指向事务起始时间戳的指针，读者可以通过该指针 (获取时间戳以) 找到实际数据
>  当 primary cell 对读者可见后 (line 58)，事务**必须提交**，因为它已经让自己的写入对读者可见

>  对于每一行，释放锁的方法就是在其 `lock` 列的对应时间戳上删去锁，并在其 `write` 列的对应时间戳上添加写入记录
>  写入记录存储指向该事务起始时间戳的指针，也就是说，`write` 列中，数据的写入记录的时间戳一般会大于 `data` 列中，数据本身的时间戳
>  这是因为 `data` 列中，数据本身的时间戳一般就是事务的起始时间戳，事务在开始时就会写入数据，而 `write` 列中，写入记录的时间戳则是事务的提交时间戳，表示事务的结束
>  提交时间戳的语义在于从这个时刻开始，事务的修改开始对外部可见，而不意味着数据的实际写入时间

>  以上描述了事务的写入操作的流程，即 `Set()` 方法
>  从伪代码上看，`Set()` 方法会缓存对应的写入，实际的每一个写入通过 `Prewrite()` 和 `Commit()` 方法，按照两阶段完成

A Get() operation first checks for a lock in the timestamp range `[0, start timestamp]`, which is the range of timestamps visible in the transaction’s snapshot (line 12). If a lock is present, another transaction is concurrently writing this cell, so the reading transaction must wait until the lock is released. If no conflicting lock is found, Get() reads the latest write record in that timestamp range (line 19) and returns the data item corresponding to that write record (line 22). 
>  事务的 `Get()` 操作首先检查时间戳范围 `[0, start_timestamp]` 内是否存在锁 (line 12)，这个时间戳范围是该事务的 snapshot 可见的时间戳范围 (因为 snapshot 处于 `start_timestamp`)
>  如果存在锁，说明有另一个事务正在并发地向该 cell 写入，故对该 cell 内容的读取必须等待该锁被释放 (以读取新内容，因为存在锁就说明写入事务的提交时间戳小于 `start_timestamp`，故为了维护 snapshot isolation，必须读取该写入)
>  如果没有发现冲突锁，`Get()` 会读取该时间戳范围 (`[0, start_timestamp`]) 内的最新写入记录 (line 19)，并返回与该写入记录对应的数据项

>  如果 cell 的时间戳范围 `[0, start_timestamp]` 内存在锁，依据上面描述的写入锁机制，说明有其他的起始时间戳位于 `[0, start_timestamp]` 范围的事务正在对该 cell 进行写入，注意该事务写入完成后，该 cell 的 `data` 列将会有一个时间戳在 `[0, start_timestamp]` 范围内的新数据，根据 snapshot isolation 语义，这个数据必须要被当前起始时间戳为 `start_timestampe` 的事务读到
>  因此，当前事务必须等待

>  如果没有发现冲突，说明不存在事务对当前 cell 进行更新，故当前 cell 的最新写入记录一定都处于 `[0, start_timestamp]` 范围内，故当前事务获取写入记录，进而获取最新的数据即可

>  以上描述了事务的读取操作的流程，即 `Get()` 方法

>  以下将客户端故障纳入考虑，进一步细化 Percolator 的提交语义，确保服从事务的原子性

Transaction processing is complicated by the possibility of client failure (tablet server failure does not affect the system since Bigtable guarantees that written locks persist across tablet server failures). If a client fails while a transaction is being committed, locks will be left behind. Percolator must clean up those locks or they will cause future transactions to hang indefinitely. Percolator takes a lazy approach to cleanup: when a transaction A encounters a conflicting lock left behind by transaction B, A may determine that B has failed and erase its locks. 
>  上述讨论的事务处理流程没有考虑到客户端故障的可能性 (tablet server 故障不会影响系统，因为 Bigtable 保证了写入的锁会在 tablet server 故障后仍然存在)
>  如果客户端在事务提交过程中故障，锁可能被遗留 (没有被释放)，Percolator 必须清理这些锁，否则它们会导致未来的事务被无限期地挂起
>  Percolator 采用 lazy 方式来进行清理: 当事务 A 遇到由事务 B 遗留的锁而被挂起时，A 可以判断 B 已经故障，并清除它的锁

>  客户端就是指 Percolator worker

It is very difficult for A to be perfectly confident in its judgment that B is failed; as a result we must avoid a race between A cleaning up B’s transaction and a not-actually-failed B committing the same transaction. Percolator handles this by designating one cell in every transaction as a synchronizing point for any commit or cleanup operations. This cell’s lock is called the primary lock. Both A and B agree on which lock is primary (the location of the primary is written into the locks at all other cells). Performing either a cleanup or commit operation requires modifying the primary lock; since this modification is performed under a Bigtable row transaction, only one of the cleanup or commit operations will succeed. Specifically: before B commits, it must check that it still holds the primary lock and replace it with a write record. Before A erases B’s lock, A must check the primary to ensure that B has not committed; if the primary lock is still present, then it can safely erase the lock. 
>  A 要对 B 是否故障做出完全确定的判断是非常困难的，因此，我们必须避免出现以下的竞争情况: A 判断 B 故障，将其锁清理，但 B 实际并未故障，并进行了提交
>  Percolator 通过在每个事务指定一个 cell 作为任何清理或提交操作的同步点来解决这个问题
>  这一被指定的 cell 的锁被称为 primary lock, A 和 B 会就哪个锁是 primary 达成一致 (primary lock 的位置会被写入所有其他 cell 的锁中)
>  无论是执行 cleanup 还是 commit 操作，都需要修改 primary lock，因为这一修改是在 Bigtable 行事务中完成的，故 cleanup 和 commit 操作只有一个能成功
>  具体地说: 
>  - 在 B 提交之前，它必须检查它是否持有 primary lock，并用写记录替换它
>  - 在 A 清理 B 的锁之前，它必须检查 primary lock 是否存在，以确保 B 还没有提交，如果 primary lock 还存在 (说明 B 尚未提交)，则 A 可以安全地删除该锁

>  Percolator 的事务是会修改多个行的，因此就会涉及处于多个行的不同 cell
>  其中某一行的 cell 会被指定为 primary，对该 cell 上的锁 (或者说在该行的 `lock` 列中写入的锁) 就是 primary lock
>  事务会在所有其他行的 `lock` 列中写入锁的同时写入对 primary lock 的引用

>  cleanup 操作会清除掉所有的锁，故涉及到了 primary lock
>  commit 操作会将所有的锁清除，同时写下写入记录，也涉及到了 primary lock
>  我们将事务在 cleanup 时和 commit 时所清除的第一个锁都定义为 primary lock，事务在 cleanup 和 commit 提交之前都需要先确定 primary 是否存在
>  在 commit 之前，如果 primary 不存在，说明自己被其他事务 cleanup 了，此时事务需要中止自己
>  在 cleanup 之前，如果 primary 不存在，说明要清理的事务正在进行 commit，此时不应该清理它
>  因为 primary 仅涉及单行，Bigtable 保证了它的原子性，故 cleanup 和 commit 就不会冲突

When a client crashes during the second phase of commit, a transaction will be past the commit point (it has written at least one write record) but will still have locks outstanding. We must perform roll-forward on these transactions. A transaction that encounters a lock can distinguish between the two cases by inspecting the primary lock: if the primary lock has been replaced by a write record, the transaction which wrote the lock must have committed and the lock must be rolled forward, otherwise it should be rolled back (since we always commit the primary first, we can be sure that it is safe to roll back if the primary is not committed). To roll forward, the transaction performing the cleanup replaces the stranded lock with a write record as the original transaction would have done. 
>  当客户端在 commit 的第二阶段中崩溃，事务已经处于了提交点之后 (它至少写入了一个写记录)，但仍然持有未释放的锁，此时，我们必须对这些事务执行向前恢复操作
>  一个遇到了锁的 (其他) 事务可以通过检查 primary lock 来区分两种情况，如果 primary lock 已经被一条写记录替换，则写入了该锁的事务必须被前滚，否则，应该被回滚 (这是因为我们总是先提交 primary lock，故如果 primary 没有被提交，回滚就是安全的)
>  为了前滚恢复，执行 cleanup 的事务 (即遇到了锁的其他事务) 会用原本事务本应写入的写记录替换掉原本事务本应替换掉的悬挂的锁

Since cleanup is synchronized on the primary lock, it is safe to clean up locks held by live clients; however, this incurs a performance penalty since rollback forces the transaction to abort. So, a transaction will not clean up a lock unless it suspects that a lock belongs to a dead or stuck worker. Percolator uses simple mechanisms to determine the liveness of another transaction. Running workers write a token into the Chubby lockservice [8] to indicate they belong to the system; other workers can use the existence of this token as a sign that the worker is alive (the token is automatically deleted when the process exits). To handle a worker that is live, but not working, we additionally write the wall time into the lock; a lock that contains a too-old wall time will be cleaned up even if the worker’s liveness token is valid. To handle long-running commit operations, workers periodically update this wall time while committing. 
>  因为 cleanup 是在 primary lock 上同步的，因此清理活跃的客户端持有的锁是安全的 (就如上面证明的一样，这只会导致对应事务被中止，不会损害原子性)
>  但是，这会带来性能开销，因为 (清理带来的) 回滚会强制事务中止，因此，除非事务怀疑某个锁属于已死的或卡住的 worker，否则它不会清理锁
>  Percolator 使用简单的机制来判断其他事务的活跃性: 运行中的 workers 会向 Chubby 锁服务中写入一个 token，以表明它们属于系统，其他 worker 可以利用该 token 的存在情况来判断 worker 是否活跃 (当进程退出时，token 会被自动删除)
>  为了处理活跃但不工作的 worker，我们额外在锁中写入当前时间戳，如果一个锁包含了过旧的时间戳，则即便其 worker 的 token 存在，它也会被清理
>  需要长时间运行的提交操作的 worker 为了避免其锁被清理，会在提交时定期更新其锁的时间戳

## 2.3 Timestamps 
The timestamp oracle is a server that hands out timestamps in strictly increasing order. Since every transaction requires contacting the timestamp oracle twice, this service must scale well. The oracle periodically allocates a range of timestamps by writing the highest allocated timestamp to stable storage; given an allocated range of timestamps, the oracle can satisfy future requests strictly from memory. If the oracle restarts, the timestamps will jump forward to the maximum allocated timestamp (but will never go backwards). 
>  timestamp oracle 是一个以严格递增顺序分配时间戳的服务器
>  因为每个事务都需要联系 timestamp oracle 两次 (起始时间戳和提交时间戳)，故该服务必须能很好地扩展
>  timestamp oracle 定期将已分配的最高时间戳写入稳定存储，来确认分配一批时间戳，给定一个已经分配的时间戳范围，timestamp oracle 可以直接从内存中满足请求
>  如果 timestamp oracle 重启，时间戳会跳转到当前的最大分配时间戳 (永远不会回退)

To save RPC overhead (at the cost of increasing transaction latency) each Percolator worker batches timestamp requests across transactions by maintaining only one pending RPC to the oracle. As the oracle becomes more loaded, the batching naturally increases to compensate. Batching increases the scalability of the oracle but does not affect the timestamp guarantees. 
>  为了减少 RPC 开销 (以增加事务延迟为代价)，每个 Percolator worker 会维护一个对 timestamp oracle 的待处理 RPC，以批量请求多个事务的时间戳
>  随着 timestamp oracle 负载的增加，批量处理的大小会自然地增长，以减小其负载
>  批处理提高了 timestamp oracle 的可拓展性，且不影响时间戳的保证

Our oracle serves around 2 million timestamps per second from a single machine. 
>  我们的单机 timestamp oracle 服务器每秒可以提供约 200 万个时间戳

The transaction protocol uses strictly increasing timestamps to guarantee that Get() returns all committed writes before the transaction’s start timestamp. 
>  Percolator 的事务协议使用严格递增的时间戳来保证 `Get()` 返回所有在事务的开始时间戳之前所有已提交的写入

To see how it provides this guarantee, consider a transaction R reading at timestamp $T_{R}$ and a transaction W that committed at timestamp $T_{W}<T_{R}$ ; we will show that R sees W’s writes. Since $T_{W}~<~T_{R}$ , we know that the timestamp oracle gave out $T_{W}$ before or in the same batch as $T_{R}$ ; hence, W requested $T_{W}$ before $R$ received $T_{R}$ . We know that R can’t do reads before receiving its start timestamp $T_{R}$ and that W wrote locks before requesting its commit timestamp $T_{W}$ . Therefore, the above property guarantees that W must have at least written all its locks before R did any reads; R’s Get() will see either the fully-committed write record or the lock, in which case W will block until the lock is released. Either way, W’s write is visible to R’s Get(). 
>  为了说明它是如何提供这种保证的，考虑一个事务 $R$，它在 $T_R$ 进行读，事务 $W$ 在 $T_W$ 进行提交，$T_W<T_R$
>  我们可以证明 $R$ 能看见 $W$ 的写入:
>  因为 $T_W<T_R$，故 timestamp oracle 对于 $T_W$ 的分发要早于 $T_R$ 的分发，或者在同一批次，因此 $W$ 一定在 $R$ 接收到 $T_R$ 之前就请求了 $T_W$
>  $R$ 在接收到其开始时间戳 $T_R$ 之前不能进行读，而 $W$ 在请求其提交时间戳 $T_W$ 之前已经写入了锁，因此，上述的性质保证了 $W$ 一定在 $R$ 的任何读之前写入了它的所有锁，$R$ 的 `Get()` 将要么看到完全提交的写记录，要么看到锁，如果看到锁，$R$ 会等待 $W$ 释放其锁
>  无论如何 $W$ 的写操作对 $R$ 的 `Get()` 都是可见的

## 2.4 Notifications 
Transactions let the user mutate the table while maintaining invariants, but users also need a way to trigger and run the transactions. In Percolator, the user writes code (“observers”) to be triggered by changes to the table, and we link all the observers into a binary running alongside every tablet server in the system. Each observer registers a function and a set of columns with Percolator, and Percolator invokes the function after data is written to one of those columns in any row. 
>  事务机制允许用户在更改表的同时维持不变式 (原子性)，但用户仍然需要一种触发和执行事务的方法
>  Percolator 中，用户需要编写代码 (observers) ，它们会在表被变更时被调用，所有的 observers 会被连接到系统中随着每个 tablet server 运行的二进制程序中 (即 Percolator worker)
>  每个 observer 向 Percolator 注册一个函数和一组列，当这些列中任意一列的数据被写入，Percolator 就会调用对应函数

Percolator applications are structured as a series of observers; each observer completes a task and creates more work for “downstream” observers by writing to the table. In our indexing system, a MapReduce loads crawled documents into Percolator by running loader transactions, which trigger the document processor transaction to index the document (parse, extract links, etc.). The document processor transaction triggers further transactions like clustering. The clustering transaction, in turn, triggers transactions to export changed document clusters to the serving system. 
>  Percolator 应用被组织为一系列 observers，每个 observer 完成其任务，并通过写入表格，为 “下游” observers 创建更多任务
>  在我们的索引系统中，一次 MapReduce 会通过运行 loader 事务，将爬取到的文档导入 Percolator 中，而 loader 事务会触发 document processor 事务对文档进行索引 (包括解析，提取链接等工作)
>  document processor 事务会触发进一步的事务，例如 clustering 事务，而 clustering 事务会触发其他事务将被修改的文档簇导出到服务系统

Notifications are similar to database triggers or events in active databases [29], but unlike database triggers, they cannot be used to maintain database invariants. In particular, the triggered observer runs in a separate transaction from the triggering write, so the triggering write and the triggered observer’s writes are not atomic. Notifications are intended to help structure an incremental computation rather than to help maintain data integrity. 
>  Percolator 中的 notifications 类似于数据库中的 events 或 triggers，不同之处在于，triggers 不能用于维护数据库的不变式
>  特别地，被触发的 observers 运行的是独立的事务，故触发这些 observers 的事务和被触发的 observers 运行的事务是分离的，不是原子的
>  notifications 旨在帮助构建增量计算，而不是维护数据完整性

This difference in semantics and intent makes observer behavior much easier to understand than the complex semantics of overlapping triggers. Percolator applications consist of very few observers — the Google indexing system has roughly 10 observers. Each observer is explicitly constructed in the main() of the worker binary, so it is clear what observers are active. It is possible for several observers to observe the same column, but we avoid this feature so it is clear what observer will run when a particular column is written. 
>  因为存在上述的语义和意图的不同，observer 的行为相较于 overlapping triggers 的语义更容易理解
>  Percolator 应用由少量的 observers 组成——Google 索引系统大约有 10 个 observers，每个 observer 都在 worker 二进制可执行文件的 `main()` 中被现实构造，故可以清楚确定哪些 observers 是活跃的 (意思应该是指 observers 被动态链接到 Percolator worker 的二进制可执行文件上)
>  多个 observers 可以观察同一列，但我们避免使用此功能，以便明确当特定的列被写入时，哪个 observer 会被运行

Users do need to be wary about infinite cycles of notifications, but Percolator does nothing to prevent this; the user typically constructs a series of observers to avoid infinite cycles. 
>  Percolator 没有设定机制避免 notifications 的无限循环，故用户需要避免出现这种情况

We do provide one guarantee: at most one observer’s transaction will commit for each change of an observed column. The converse is not true, however: multiple writes to an observed column may cause the corresponding observer to be invoked only once. We call this feature message collapsing, since it helps avoid computation by amortizing the cost of responding to many notifications. For example, it is sufficient for http://google.com to be reprocessed periodically rather than every time we discover a new link pointing to it. 
>  Percolator 提供了一个保证: 对于每个被观察的列的每个更改，最多只有一个 observer 的事务会提交
>  反过来则不成立: 对单个被观察列的多次写入可能导致相应的 observer 只被调用一次
>  我们将该机制称为 message collapsing，因为它通过分摊相应多个 notifications 的成本来避免计算
>  例如，对于 http://google.com 来说，定期重新处理就够了，不需要每次发现指向它的新链接时都进行处理

To provide these semantics for notifications, each observed column has an accompanying “acknowledgment” column for each observer, containing the latest start timestamp at which the observer ran. When the observed column is written, Percolator starts a transaction to process the notification. The transaction reads the observed column and its corresponding acknowledgment column. If the observed column was written after its last acknowledgment, then we run the observer and set the acknowledgment column to our start timestamp. Otherwise, the observer has already been run, so we do not run it again. 
>  为了为 notifications 提供上述语义，每个被观察的列，对于它的每个 observer，都有一个与之对应的 “acknowledgement” 列，该列包含了观察该列的 observer 最后一次运行的起始时间戳
>  当被观察的列被写入后，Percolator 会启动一个事务来处理 notification，该事务读取被观察的列和其对应的 acknowledgement 列，如果被观察的列是在其最后一次确认后被写入的，则我们运行 observer，并将 acknowledgement 列的值设定为当前事务的起始时间戳
>  如果不是 (列被写入的时间戳小于等于 observer 运行的时间戳)，则说明 observer 已经被运行过了，故我们不需要再运行一次

>  因此，如果被观察的列被快速地写入了多次，则 observer 可能只需要调用一次 (看到了最后一次写入的 observer)，就确保了 acknowledgment 的时间戳大于最后一次写入的时间戳，看到了之前的多次写入的 observer 检查到这一点后，就不会启动事务

Note that if Percolator accidentally starts two transactions concurrently for a particular notification, they will both see the dirty notification and run the observer, but one will abort because they will conflict on the acknowledgment column. We promise that at most one observer will commit for each notification. 
>  注意，如果 Percolator 意外地为某个 notification 启动了两个并发事务，则这两个事务都会看到 dirty notification，并运行 observer，但其中一个会中止，因为它们会在 acknowledge 列上发生冲突 (先完成的 observer 先写入 acknowledge)
>  因此，我们保证了对于每个 notification ，只有一个 observer 会提交

To implement notifications, Percolator needs to efficiently find dirty cells with observers that need to be run. This search is complicated by the fact that notifications are rare: our table has trillions of cells, but, if the system is keeping up with applied load, there will only be millions of notifications. Additionally, observer code is run on a large number of client processes distributed across a collection of machines, meaning that this search for dirty cells must be distributed. 
>  为了实现 notification, Percolator 需要高效地找到 dirty cells
>  由于 notification 占总体的比例很小，这种搜索会很复杂: 我们的表格有数万亿个 cells，如果系统能跟上应用负载，那么一般只会有数百万个 notifications
>  此外，observer 代码是在大量的分布在多台机器上的客户端进程 (Percolator workers) 运行的，这意味着对 dirty cell 的搜索必须是分布式的

To identify dirty cells, Percolator maintains a special “notify” Bigtable column, containing an entry for each dirty cell. When a transaction writes an observed cell, it also sets the corresponding notify cell. The workers perform a distributed scan over the notify column to find dirty cells. After the observer is triggered and the transaction commits, we remove the notify cell. Since the notify column is just a Bigtable column, not a Percolator column, it has no transactional properties and serves only as a hint to the scanner to check the acknowledgment column to determine if the observer should be run. 
>  为了判断 dirty cells, Percolator 维护一个 "notify" Bigtable 列，它包含了每个 dirty cell 的条目
>  当一个事务写入一个被观察的 cell 时，它也会设置对应的 notify cell, workers 会在 notify cell 上执行分布式扫描以发现 dirty cells
>  在 observer 被启动并且事务提交后，我们就移除 notify cell
>  因为 notify 列是 Bigtable 列而不是 Percolator 列，故其并没有事务性的属性，仅作为 scanner 的提示，指示 scanner 检查对应的 acknowledgement 列以确定是否应该运行 observer

To make this scan efficient, Percolator stores the notify column in a separate Bigtable locality group so that scanning over the column requires reading only the millions of dirty cells rather than the trillions of total data cells. Each Percolator worker dedicates several threads to the scan. For each thread, the worker chooses a portion of the table to scan by first picking a random Bigtable tablet, then picking a random key in the tablet, and finally scanning the table from that position. 
>  为了让 scan 更高效，Percolator 将 notify 列存储在分离的 Bigtable locality group 中，故对 notify 列的 scan 仅需要读取数百万个 dirty cells，而不是数万亿个 data cells
>  每个 Percolator worker 会为 scan 分配多个线程，对于每个线程，worker 会先选择一个随机的 Bigtable tablet，然后选择该 tablet 中的一个随机 key，让该线程从该位置开始扫描表格 (以发现 dirty cells)

Since each worker is scanning a random region of the table, we worry about two workers running observers on the same row concurrently. While this behavior will not cause correctness problems due to the transactional nature of notifications, it is inefficient. To avoid this, each worker acquires a lock from a lightweight lock service before scanning the row. This lock server need not persist state since it is advisory and thus is very scalable. 
>  因为每个 worker 都在 scan 表格中的一个随机区域，两个 worker 可能会并发地运行同一行的 observers
>  虽然由于 notifications 的事务性质，这一行为不会导致正确性问题 (上面所介绍的)，但这是低效的
>  为了避免这一情况，每个 worker 在 scan 对应的行之前，会从轻量的锁服务中获取锁 (避免不同的 workers 扫描到同一行)，锁服务不需要持久化状态 (因为它仅提供指导)，故是可拓展的

The random-scanning approach requires one additional tweak: when it was first deployed we noticed that scanning threads would tend to clump together in a few regions of the table, effectively reducing the parallelism of the scan. 
>  上述介绍的随机扫描方法还需要一个额外的调整: 我们在部署时注意到扫描线程往往会聚集在表的少数几个区域中，降低了扫描的并行性

This phenomenon is commonly seen in public transportation systems where it is known as “platooning” or “bus clumping” and occurs when a bus is slowed down (perhaps by traffic or slow loading). Since the number of passengers at each stop grows with time, loading delays become even worse, further slowing the bus. Simultaneously, any bus behind the slow bus speeds up as it needs to load fewer passengers at each stop. The result is a clump of buses arriving simultaneously at a stop [19]. 
>  这种现象在公共交通系统中也很常见，被称为 "车队效应"，并且在一辆公交车慢下来时就会发生，随着每个站点的乘客数量随时间增加，公交车的上客延迟将更长，进一步拖慢公交车
>  同时，任何紧跟在慢车后面的公交车会加速，因为它需要加载的乘客更少，结果是一组公交车同时到达某个站点

Our scanning threads behaved analogously: a thread that was running observers slowed down while threads “behind” it quickly skipped past the now-clean rows to clump with the lead thread and failed to pass the lead thread because the clump of threads overloaded tablet servers. 
>  我们的扫描线程的表现和其类似: 运行 observers 的线程会减速，而它后面的线程会快速地跳过现在正在清理的行，与领头线程聚在一起，然后由于线程群集使 tablet servers 过载而无法超越领头线程

To solve this problem, we modified our system in a way that public transportation systems cannot: when a scanning thread discovers that it is scanning the same row as another thread, it chooses a new random location in the table to scan. To further the transportation analogy, the buses (scanner threads) in our city avoid clumping by teleporting themselves to a random stop (location in the table) if they get too close to the bus in front of them. 
>  为了解决该问题，我们对系统进行了修改: 当扫描线程发现它正在扫描与其他线程相同的行时，它会在表格中随机选择新的一行进行扫描，以避免扎堆

Finally, experience with notifications led us to introduce a lighter-weight but semantically weaker notification mechanism. We found that when many duplicates of the same page were processed concurrently, each transaction would conflict trying to trigger reprocessing of the same duplicate cluster. 
>  最后，我们引入了一个更轻量但语义上更弱的 notification 机制
>  我们发现当大量相同页面并发被处理时，每个事务都会尝试触发对同一个重复 cluster (页面被映射到 cluster，页面相同，cluster 就相同) 的重处理，进而冲突

This led us to devise a way to notify a cell without the possibility of transactional conflict. We implement this weak notification by writing only to the Bigtable “notify” column. 
>  为此，我们设计了一种在不产生事务冲突的情况下 notify 一个 cell 的方法，我们通过仅向 Bigtable "notify" 列写入数据来实现这一弱 notification 机制

To preserve the transactional semantics of the rest of Percolator, we restrict these weak notifications to a special type of column that cannot be written, only notified. 
>  为了维护 Percolator 其余部分的事务语义，我们将这一弱 notification 仅用于一种特殊类型的列上，这种列只能被通知而不能被写入 (也就是这个列不存储数据，但可以用于触发 observers，因此只要写入 notify 列就行)

>  注意，正常情况下，数据应该是要写入到 Percolator table 中的 `data` 列的，并且在写入的同时在 notify 列写入 cell 信息，以便于触发 observers 来检查当前 cell 的数据是否发生变化，进而触发事务，事务结束后，还需要在 `ack` 列写入信息，避免其他更早的 observers 被重复触发

The weaker semantics also mean that multiple observers may run and commit as a result of a single weak notification (though the system tries to minimize this occurrence). 
>  较弱的语义也意味着单个弱通知可能导致多个 observers 运行并提交 (尽管系统会尽量减少这一情况的发生)
>  (这是因为此时的弱 notification 仅仅维护 notify 列，不使用 acknowledgment 列来维护其语义)

This has become an important feature for managing conflicts; if an observer frequently conflicts on a hotspot, it often helps to break it into two observers connected by a non-transactional notification on the hotspot. 
>  这一弱通知机制已经成为管理冲突的重要手段，如果一个 observer 经常在某个热点区域 (例如某个被频繁修改的 cell) 发生冲突，通常可以通过将其拆分为两个通过在热点上的非事务性 notification 连接的 observers 来解决
>  (这样能确保 observer 被成功调用，但语义就不好保证，因为可能会被重复调用，且原来的一个原子性事务的工作被拆分为了两个事务)

## 2.5 Discussion 
One of the inefficiencies of Percolator relative to a MapReduce-based system is the number of RPCs sent per work-unit. While MapReduce does a single large read to GFS and obtains all of the data for 10s or 100s of web pages, Percolator performs around 50 individual Bigtable operations to process a single document. 
>  与基于 MapReduce 的系统相比，Percolator 一个效率低下的地方在于每个工作单元发送的 RPCs 的数量
>  MapReduce 对 GFS 执行一次大型读取，获取数十个或数百个网页的数据；Percolator 则需要执行大约 50 次独立的 Bigtable 操作来处理单个文档

One source of additional RPCs occurs during commit. When writing a lock, we must do a read-modify-write operation requiring two Bigtable RPCs: one to read for conflicting locks or writes and another to write the new lock. To reduce this overhead, we modified the Bigtable API by adding conditional mutations which implements the read-modify-write step in a single RPC. 
>  Percolator 的额外 RPCs 的来源之一出现在提交阶段
>  当写入一个锁时，我们必须执行一次 read-modify-write 操作，这需要两次 Bigtable RPCs: 一次用于读取冲突的锁 (或者说冲突的写入)，一次用于写入新的锁
>  为了减少这一开销，我们修改了 Bigtable API，添加了条件更改，条件更改通过单次 RPC 来实现 read-modify-write 操作

Many conditional mutations destined for the same tablet server can also be batched together into a single RPC to further reduce the total number of RPCs we send. We create batches by delaying lock operations for several seconds to collect them into batches. Because locks are acquired in parallel, this adds only a few seconds to the latency of each transaction; we compensate for the additional latency with greater parallelism. Batching also increases the time window in which conflicts may occur, but in our low-contention environment this has not proved to be a problem. 
>  多个发往同一 tablet server 的条件更改可以被批量合并为一个 RPC，以进一步减少发送的总 RPC 数量
>  我们通过延迟锁操作几秒钟来将多个 RPCs 合并为一个批次，由于锁是并行获得的，这只会给每个事务添加几秒钟的延迟，且我们通过更高的并行性弥补了这一额外的延迟
>  批处理增加了可能发生冲突的时间窗口，但在我们的低争用环境中，这并未成为一个问题

We also perform the same batching when reading from the table: every read operation is delayed to give it a chance to form a batch with other reads to the same tablet server. 
>  我们还在从表中读取数据时执行相同的批处理操作: 每次读操作会被延迟，以便有机会和其他对相同的 tablet server 的读操作组合为一个批次

This delays each read, potentially greatly increasing transaction latency. A final optimization mitigates this effect, however: prefetching. 
>  这一处理延迟了每次的读操作，可能导致事务延迟显著增加
>  然而，有一个最终的优化措施: 预取，可以缓解该影响

Prefetching takes advantage of the fact that reading two or more values in the same row is essentially the same cost as reading one value. In either case, Bigtable must read the entire SSTable block from the file system and decompress it. Percolator attempts to predict, each time a column is read, what other columns in a row will be read later in the transaction. This prediction is made based on past behavior. Prefetching, combined with a cache of items that have already been read, reduces the number of Bigtable reads the system would otherwise do by a factor of 10. 
>  预取利用了在同一行中读取两个或多个值的成本与读取一个值的成本几乎相同的事实，无论是哪种情况，Bigtable 都必须从文件系统中读取整个 SSTable 块，然后对其解压缩
>  Percolator 每次在读取某一列时，都会尝试预测该行在事务中后续可能会被读取的其他列，这一预测基于之前的行为
>  结合已经读取的项的缓存，预取将系统原本需要执行的 Bigtable 读次数减少了 10 倍

Early in the implementation of Percolator, we decided to make all API calls blocking and rely on running thousands of threads per machine to provide enough parallelism to maintain good CPU utilization. We chose this thread-per-request model mainly to make application code easier to write, compared to the event-driven model. Forcing users to bundle up their state each of the (many) times they fetched a data item from the table would have made application development much more difficult. Our experience with thread-per-request was, on the whole, positive: application code is simple, we achieve good utilization on many-core machines, and crash debugging is simplified by meaningful and complete stack traces. We encountered fewer race conditions in application code than we feared. The biggest drawbacks of the approach were scalability issues in the Linux kernel and Google infrastructure related to high thread counts. Our in-house kernel development team was able to deploy fixes to address the kernel issues. 
>  在 Percolator 的早期实现中，我们将所有 API 调用都定为阻塞式，并依赖于在每台机器上运行上千线程来提供足够的并行性，以维持良好的 CPU 利用率
>  我们选择这一 thread-per-request 模型的原因主要在于相较于事件驱动的模型，其应用代码更容易编写，迫使用户每次从表中获取数据项时都要捆绑其状态，会让应用开发非常困难
>  总体而言，我们的 thread-per-request 经验是积极的: 应用程序代码简单易懂，我们也在多核机器上实现了良好的利用率，且 crash debugging 也更加简单，因为 stack traces 更加完整且有意义
>  我们遇到的应用程序代码中的竞争条件比我们预期的要少
>  这种方法最大的缺点就是 Linux kernel 和 Google infrastructure 中，与高线程数量相关的可拓展性问题，我们的 kernel 开发团队能够部署修复方案来解决 kernel 相关的问题

# 3 Evaluation 
Percolator lies somewhere in the performance space between MapReduce and DBMSs. For example, because Percolator is a distributed system, it uses far more resources to process a fixed amount of data than a traditional DBMS would; this is the cost of its scalability. Compared to MapReduce, Percolator can process data with far lower latency, but again, at the cost of additional resources required to support random lookups. These are engineering tradeoffs which are difficult to quantify: how much of an efficiency loss is too much to pay for the ability to add capacity endlessly simply by purchasing more machines? Or: how does one trade off the reduction in development time provided by a layered system against the corresponding decrease in efficiency? 
>  Percolator 的性能位于 MapReduce 和 DBMS 之间
>  因为 Percolator 是一个分布式系统，相较于传统的 DBMS，它处理固定数量的数据需要的资源会更多，这是其可拓展性的代价
>  相较于 MapReduce, Percolator 可以以更低的延迟处理数据，但代价是需要额外的资源来支持随机查找
>  这些都是难以量化的工程权衡问题: 为了能够通过无限购买更多机器来无限拓展容量 (可拓展性)，付出多少效率损失是可以接收的？或者，分层系统的开发事件的较少如何与相应的效率降低进行权衡？

In this section we attempt to answer some of these questions by first comparing Percolator to batch processing systems via our experiences with converting a MapReduce-based indexing pipeline to use Percolator. We’ll also evaluate Percolator with microbenchmarks and a synthetic workload based on the well-known TPC-E benchmark [1]; this test will give us a chance to evaluate the scalability and efficiency of Percolator relative to Bigtable and DBMSs. 
>  本节中，我们比较 Percolator 与批处理系统 (将基于 MapReduce 的索引 pipeline 转换为使用 Percolator) 来尝试回答这些问题
>  我们还会使用微基准测试和基于 TPC-E 基准的合成工作负载来评估 Percolator，以比较 Percolator 相对于 Bigtable 和 DBMS 的可拓展性和效率

All of the experiments in this section are run on a subset of the servers in a Google data center. The servers run the Linux operating system on x86 processors; each machine is connected to several commodity SATA drives. 

## 3.1 Converting from MapReduce 
We built Percolator to create Google’s large “base” index, a task previously performed by MapReduce. In our previous system, each day we crawled several billion documents and fed them along with a repository of existing documents through a series of 100 MapReduces. The result was an index which answered user queries. Though not all 100 MapReduces were on the critical path for every document, the organization of the system as a series of MapReduces meant that each document spent 2-3 days being indexed before it could be returned as a search result. 
>  我们用 Percolator 构建 Google 的大型 “基础” 索引，这个任务之前由 MapReduce 完成
>  在我们之前的系统中，我们每天会抓取数十亿份文档，然后将这些文档与现有的文档库一起交给一系列 100 个 MapReduce 任务进行处理，最后的结果是一个能回答用户查询的索引
>  虽然每个文档的关键路径不会都包含所有 100 个 MapReduce 任务，但系统的组织方式是一系列 MapReduce 任务意味着每个文档需要花费 2-3 天进行索引才能够作为搜索结果返回

The Percolator-based indexing system (known as Caffeine [25]), crawls the same number of documents, but we feed each document through Percolator as it is crawled. The immediate advantage, and main design goal, of Caffeine is a reduction in latency: the median document moves through Caffeine over $100\mathrm{x}$ faster than the previous system. This latency improvement grows as the system becomes more complex: adding a new clustering phase to the Percolator-based system requires an extra lookup for each document rather an extra scan over the repository. Additional clustering phases can also be implemented in the same transaction rather than in another MapReduce; this simplification is one reason the number of observers in Caffeine (10) is far smaller than the number of MapReduces in the previous system (100). This organization also allows for the possibility of performing additional processing on only a subset of the repository without rescanning the entire repository. 
>  基于 Percolator 的索引系统 Caffeine 爬取相同数量的文档，但会在爬取过程中将文档传递给 Percolator
>  Caffeine 的主要优势和设计目标是减小延迟: 文档通过 Caffeine 的速度的中位数比之前的系统快了 100 倍以上
>  随着系统变得更复杂，延迟上的改进会更明显: 在基于 Percolator 的系统中添加新的 clustering phase 仅需要为每个文档执行一次额外的查找，而不是对整个仓库的一次额外扫描，此外，额外的 clustering phase 可以在同一个事务上实现，而不是在另一个 MapReduce 任务上实现
>  这种简化是 Caffeine 中的 observers 的数量 (10) 远小于之前系统中 MapReduce 任务的数量 (100) 的主要原因
>  这种组织方式也允许多整个存储库的一部分执行额外处理，而不需要重新扫描整个存储库

Adding additional clustering phases isn’t free in an incremental system: more resources are required to make sure the system keeps up with the input, but this is still an improvement over batch processing systems where no amount of resources can overcome delays introduced by stragglers in an additional pass over the repository. Caffeine is essentially immune to stragglers that were a serious problem in our batch-based indexing system because the bulk of the processing does not get held up by a few very slow operations. The radically-lower latency of the new system also enables us to remove the rigid distinctions between large, slow-to-update indexes and smaller, more rapidly updated indexes. Because Percolator frees us from needing to process the repository each time we index documents, we can also make it larger: Caffeine’s document collection is currently $3\mathrm{x}$ larger than the previous system’s and is limited only by available disk space. 
>  在增量式系统中，加入额外的 clustering phase 也需要代价: 需要更多的资源以确保系统可以跟上输入
>  但相较于批处理系统仍然有所改进，批处理系统中，无论投入多少资源，都无法克服对存储库进行额外遍历时，慢节点引入的延迟
>  Caffeine 几乎不受慢节点的影响 (而慢节点在我们的基于批处理的索引系统则是严重的问题)，因为大部分处理不会因为少量非常慢的操作而阻塞
>  新系统显著减低的延迟使得在之前系统中，大型、更新缓慢的索引和小型、更新快速的索引之间的严格区分不再存在
>  因为 Percolator 让我们不必每次索引文档时处理整个仓库，我们也可以将其做得更大: Caffeine 的文档集合是旧系统的 3x 大，并且仅受可用磁盘空间的限制

Compared to the system it replaced, Caffeine uses roughly twice as many resources to process the same crawl rate. However, Caffeine makes good use of the extra resources. If we were to run the old indexing system with twice as many resources, we could either increase the index size or reduce latency by at most a factor of two (but not do both). On the other hand, if Caffeine were run with half the resources, it would not be able to process as many documents per day as the old system (but the documents it did produce would have much lower latency). 
>  相较于旧系统，Caffeine 处理相同的爬取速率所需的资源大约是旧系统的两倍
>  但 Caffeine 充分利用了这些额外资源
>  如果我们用两倍的资源运行旧系统，我们要么增加索引规模，要么最多将延迟降低一倍 (两者不能同时做到)，而用同样的资源运行 Caffeine，它每天处理的文档数量会低于旧系统，但它产生的文档会有更低的延迟

The new system is also easier to operate. Caffeine has far fewer moving parts: we run tablet servers, Percolator workers, and chunkservers. In the old system, each of a hundred different MapReduces needed to be individually configured and could independently fail. Also, the “peaky” nature of the MapReduce workload made it hard to fully utilize the resources of a datacenter compared to Percolator’s much smoother resource usage. 
>  Caffeine 也更容易运行，它的构成部件少得多: tablet servers, Percolator workers, chunkservers
>  在旧系统中，每个 MapReduce 任务都需要独立配置，且会独立失败，此外，MapReduce 的工作负载的 “peaky” 特性使得旧系统相较于 Percolator 的更加平稳的资源使用相比，更难充分利用数据中心的资源

The simplicity of writing straight-line code and the ability to do random lookups into the repository makes developing new features for Percolator easy. Under MapReduce, random lookups are awkward and costly. On the other hand, Caffeine developers need to reason about concurrency where it did not exist in the MapReduce paradigm. Transactions help deal with this concurrency, but can’t fully eliminate the added complexity. 
>  为 Percolator 提供了对存储库进行随机查找的能力，故为 Percolator 开发新特性也更加容易，而在 MapReduce 中，随机查找则非常笨拙且昂贵
>  但另一方面，Caffeine 开发者需要考虑 MapReduce 范式中原本不需要考虑的并发问题，事务有助于处理这种并发性，但无法完全消除由此带来的额外复杂性

To quantify the benefits of moving from MapReduce to Percolator, we created a synthetic benchmark that clusters newly crawled documents against a billion-document repository to remove duplicates in much the same way Google’s indexing pipeline operates. 
>  为了量化从 MapReduce 迁移到 Percolator 的好处，我们创建了一个合成基准测试，该测试针对数十亿文档的存储库，对新抓取的文档进行聚类，以去除重复项，其工作方式与 Google indexing pipeline 类似

Documents are clustered by three clustering keys. In a real system, the clustering keys would be properties of the document like redirect target or content hash, but in this experiment we selected them uniformly at random from a collection of 750M possible keys.
>  文档通过三个 clustering keys 进行聚类
>  在真实系统中，这些 clustering keys 将是文档的属性，例如重定向目标或内容哈希，在本实验中，我们从 7.5 亿个可能的 keys 中均匀地随机选择它们

 The average cluster in our synthetic repository contains 3.3 documents, and $93\%$ of the documents are in a non-singleton cluster. This distribution of keys exercises the clustering logic, but does not expose it to the few extremely large clusters we have seen in practice. These clusters only affect the latency tail and not the results we present here. 
>  在基准测试中的合成仓库中，平均每个 cluster 包含 3.3 个文档，且 93% 的文档属于 non-singleton cluster，这样的 keys 的分布能测试 clustering 逻辑，但不能将 clustering 逻辑暴露给实践中可能遇到的少数的超大 cluster
>  不过这样的超大 cluster 仅影响 latency tail，不会影响我们在这里展示的结果

In the Percolator clustering implementation, each crawled document is immediately written to the repository to be clustered by an observer. The observer maintains an index table for each clustering key and compares the document against each index to determine if it is a duplicate (an elaboration of Figure 2). 
>  在 Percolator 的 clustering 实现中，每个爬取到的文档会被立即写入存储库，并且被一个 observer 执行聚类
>  该 observer 为每个 clustering key 维护一个索引表，然后将文档与索引表中的每个索引比较，以决定该文档是否是重复项

MapReduce implements clustering of continually arriving documents by repeatedly running a sequence of three clustering MapReduces (one for each clustering key). The sequence of three MapReduces processes the entire repository and any crawled documents that accumulated while the previous three were running. 
>  MapReduce 的 clustering 实现则通过反复运行三个 clustering MapReduce 任务 (每个 MapReduce 任务对应一个 clustering key) 来实现对不断到达的文档的聚类
>  三个 clustering MapReduce 任务会处理整个存储库 + 在前一组 (三个) MapReduce 任务运行时任意新爬取到的文档

This experiment simulates clustering documents crawled at a uniform rate. Whether MapReduce or Percolator performs better under this metric is a function of the how frequently documents are crawled (the crawl rate) and the repository size.
>  试验中，文档的爬取率是均匀的，在这一标准下，MapReduce 和 Percolator 哪个的表现更好取决于文档被爬取的频率和仓库本身的大小

We explore this space by fixing the size of the repository and varying the rate at which new documents arrive, expressed as a percentage of the repository crawled per hour. In a practical system, a very small percentage of the repository would be crawled per hour: there are over 1 trillion web pages on the web (and ideally in an indexing system’s repository), far too many to crawl a reasonable fraction of in a single day. When the new input is a small fraction of the repository (low crawl rate), we expect Percolator to outperform MapReduce since MapReduce must map over the (large) repository to cluster the (small) batch of new documents while Percolator does work proportional only to the small batch of newly arrived documents (a lookup in up to three index tables per document). At very large crawl rates where the number of newly crawled documents approaches the size of the repository, MapReduce will perform better than Percolator. This cross-over occurs because streaming data from disk is much cheaper, per byte, than performing random lookups. At the cross-over the total cost of the lookups required to cluster the new documents under Percolator equals the cost to stream the documents and the repository through MapReduce. At crawl rates higher than that, one is better off using MapReduce. 
>  我们固定存储库的大小，改变爬取速率 (用每小时爬取的文档量占存储库的百分比表示)，来探索 Percolator 相对于 MapReduce 的优势区间
>  在实际的系统中，每小时爬取的文档量占存储库的比例会非常小: 网络上有超过一万亿个网页 (理想情况下，索引系统的存储库中会存储全部的这些网页)，这个数量远远超过每天可以爬取到的网页数量
>  当新增的输入仅占存储库的很小的比例时 (低爬取率)，我们预计 Percolator 会优于 MapReduce，因为 MapReduce 必须对整个存储库进行 map，以对少量的新文档进行聚类，而 Percolator 则仅需要处理新到达的文档 (每个文档最多在三个索引表中进行查找)
>  而在爬取率非常高的情况下 (新爬取的文档数量接近存储库大小)，MapReduce 的表现将优于 Percolator，这一交叉点出现的原因是: 从磁盘流式读取数据的平均每字节成本远远低于随机查找的成本，在交叉点处，Percolator 为了聚类新文档所需要的查找成本等于 MapReduce 流式处理新文档和整个存储库的成本，如果爬取率高于该交叉点，MapReduce 的性能更好

![[pics/Percolator-Fig7.png]]

We ran this benchmark on 240 machines and measured the median delay between when a document is crawled and when it is clustered. Figure 7 plots the median latency of document processing for both implementations as a function of crawl rate. 
>  我们在 240 台机器上运行了测试，并衡量了一个文档从被爬取到被聚类的延迟时间的中位数，如 Fig7 所示

When the crawl rate is low, Percolator clusters documents faster than MapReduce as expected; this scenario is illustrated by the leftmost pair of points which correspond to crawling 1 percent of documents per hour. MapReduce requires approximately 20 minutes to cluster the documents because it takes 20 minutes just to process the repository through the three MapReduces (the effect of the few newly crawled documents on the runtime is negligible). This results in an average delay between crawling a document and clustering of around 30 minutes: a random document waits 10 minutes after being crawled for the previous sequence of MapReduces to finish and then spends 20 minutes being processed by the three MapReduces. Percolator, on the other hand, finds a newly loaded document and processes it in two seconds on average, or about $1000\mathrm{x}$ faster than MapReduce. The two seconds includes the time to find the dirty notification and run the transaction that performs the clustering. Note that this $1000\mathrm{x}$ latency improvement could be made arbitrarily large by increasing the size of the repository. 
>  爬取率低时，Percolator 对文档聚类的速率远快于 MapReduce
>  MapReduce 需要大约 20 分钟来聚类，这是因为它需要花费 20 分钟处理为整个存储库运行那三个 MapReduce 任务，新加入的文档对运行时间的影响实际上可以忽略不记，这导致从爬取文档到聚类之间的延迟大约为 30 分钟: 一个随机的文档在被爬取后等待 10 分钟，直到上一轮的 MapReduce 完成，然后历经 20 分钟的处理
>  Percolator 则平均只需要两秒钟就能找到并处理新的文档，大约比 MapReduce 快 1000 倍，这两秒包括了查找 dirty notifications 和运行执行聚类的事务所需要的时间
>  注意，随着存储库增大，Percolator 的优势将更明显

As the crawl rate increases, MapReduce’s processing time grows correspondingly. Ideally, it would be proportional to the combined size of the repository and the input which grows with the crawl rate. In practice, the running time of a small MapReduce like this is limited by stragglers, so the growth in processing time (and thus clustering latency) is only weakly correlated to crawl rate at low crawl rates. The 6 percent crawl rate, for example, only adds 150GB to a 1TB data set; the extra time to process 150GB is in the noise. The latency of Percolator is relatively unchanged as the crawl rate grows until it suddenly increases to effectively infinity at a crawl rate of $40\%$ per hour. At this point, Percolator saturates the resources of the test cluster, is no longer able to keep up with the crawl rate, and begins building an unbounded queue of unprocessed documents. The dotted asymptote at $40\%$ is an extrapolation of Percolator’s performance beyond this breaking point.
>  随着爬取速率增大，MapReduce 的处理时间也相应增加，理想情况下，这应该和存储库和输入的总大小成正比，在实践中，运行时间会受到慢节点的限制，故处理时间的增长与爬取速率之间的关联性较弱
>  Percolator 的延迟在爬取速率增长时的一段时间内相对保持不变，直到爬取速率增长到 40% 时，Percolator 的延迟时间突然增长到无穷大，因为在这一点上，Percolator 超过了测试集群的资源容量，无法再跟上爬取速率，并开始积累一个无限增长的未处理文档队列
>  40% 之后的虚线是 Percolator 在超过这一点之后的外推性能

MapReduce is subject to the same effect: eventually crawled documents accumulate faster than MapReduce is able to cluster them, and the batch size will grow without bound in subsequent runs. In this particular configuration, however, MapReduce can sustain crawl rates in excess of $100\%$ (the dotted line, again, extrapolates performance). 
>  MapReduce 实际上也面临相同的问题: 最终爬取到的文档数量的积累速度会超过 MapReduce 能够对其进行聚类处理的速度，此时批量大小将会无限制地增长
>  不过，在我们的测试的配置下，MapRduce 可以承受超过 100% 的爬取速率 (虚线同样表示性能的外推)

These results show that Percolator can process documents at orders of magnitude better latency than MapReduce in the regime where we expect real systems to operate (single-digit crawl rates). 
>  试验结果说明了，在我们预期的真实系统运行环境下 (爬取速率的百分比为个位数)，Percolator 的处理延迟将远低于 MapReduce



## 3.2 Microbenchmarks 
In this section, we determine the cost of the transactional semantics provided by Percolator. In these experiments, we compare Percolator to a “raw” Bigtable. We are only interested in the relative performance of Bigtable and Percolator since any improvement in Bigtable performance will translate directly into an improvement in Percolator performance.
>  我们测试 Percolator 提供的事务性语义的开销
>  我们将 Percolator 和 “原始” 的 Bigtable 比较，我们仅关注 Bigtable 和 Percolator 之间的相对性能，因为 Bigtable 本身的性能提升本质也会让 Percolator 的性能提升

Figure 8: The overhead of Percolator operations relative to Bigtable. Write overhead is due to additional operations Percolator needs to check for conflicts. 

<html><body><center><table><tr><td></td><td>Bigtable</td><td>Percolator</td><td>Relative</td></tr><tr><td>Read/s</td><td>15513</td><td>14590</td><td>0.94</td></tr><tr><td>Write/s</td><td>31003</td><td>7232</td><td>0.23</td></tr></table></center></body></html> 

Figure 8 shows the performance of Percolator and raw Bigtable running against a single tablet server. All data was in the tablet server’s cache during the experiments and Percolator’s batching optimizations were disabled. 
>  Percolator 和 raw Bigtable 运行单个 tablet server 的性能如上所示
>  测试时，所有的数据都位于 tablet server 的 cache，且 Percolator 的批量优化被禁止

As expected, Percolator introduces overhead relative to Bigtable. We first measure the number of random writes that the two systems can perform. In the case of Percolator, we execute transactions that write a single cell and then commit; this represents the worst case for Percolator overhead. When doing a write, Percolator incurs roughly a factor of four overhead on this benchmark. This is the result of the extra operations Percolator requires for commit beyond the single write that Bigtable issues: a read to check for locks, a write to add the lock, and a second write to remove the lock record. The read, in particular, is more expensive than a write and accounts for most of the overhead. In this test, the limiting factor was the performance of the tablet server, so the additional overhead of fetching timestamps is not measured. 
>  Percolator 相较于 Bigtable 引入了额外开销
>  我们首先测量了两个系统每秒可以执行的随机写入数量，我们让 Percolator 执行一个写入单个 cell 的事务并提交，这代表了 Percolator 开销的最坏情况 (一次事务仅涉及一个 cell，事务的开销没有被多个 cells 摊销)
>  在测试中，Percolator 的写入开销大约是 Bigtable 的四倍，这是因为 Percolator 为了提交，除了单次的 Bigtable 写入外，还需要额外操作: 一次读以检查锁，一次写以添加锁，一次写以移除锁，并且在这些额外开销中，读操作比写操作更昂贵，占据了大部分额外开销
>  在测试中，限制因素是 tablet server 的性能，故没有度量获取 timestamps 的额外开销

We also tested random reads: Percolator performs a single Bigtable operation per read, but that read operation is somewhat more complex than the raw Bigtable operation (the Percolator read looks at metadata columns in addition to data columns). 
>  我们也测量了随机读取: Percolator 的每次读取需要执行单次 Bigtable operation，但其读操作本身比 raw Bigtable operation 要复杂 (Percolator 读操作除了要查看数据列，还需要查看元数据列)，故会略慢于 raw Bigtable read

## 3.3 Synthetic Workload 
To evaluate Percolator on a more realistic workload, we implemented a synthetic benchmark based on TPC-E [1]. This isn’t the ideal benchmark for Percolator since TPC-E is designed for OLTP systems, and a number of Percolator’s tradeoffs impact desirable properties of OLTP systems (the latency of conflicting transactions, for example). TPC-E is a widely recognized and understood benchmark, however, and it allows us to understand the cost of our system against more traditional databases. 
>  我们基于 TPC-E 实现了合成的基准测试，这一测试并不是相对于 Percolator 理想的测试，因为 TPC-E 针对 OLTP 系统设计，而 Percolator tradeoff 了许多相对于 OLTP 系统理想的性质 (例如冲突事务的延迟)
>  但 TPC-E 是被广泛使用的 benchmark，故可以帮助我们理解 Percolator 相对于传统数据库的开销

TPC-E simulates a brokerage firm with customers who perform trades, market search, and account inquiries. The brokerage submits trade orders to a market exchange, which executes the trade and updates broker and customer state. The benchmark measures the number of trades executed. On average, each customer performs a trade once every 500 seconds, so the benchmark scales by adding customers and associated data. 
>  TPC-E 模拟了一家拥有客户的经纪公司，这些客户执行交易、市场搜索和账户查询
>  该公司将交易订单提交给市场交易所，市场交易所执行交易，并更新经纪人和客户的状态
>  该 benchmark 度量交易执行的数量
>  该 benchmark 中的每个客户平均 500s 执行一次交易，故该 benchmark 通过添加顾客和相关数据来扩展规模

TPC-E traditionally has three components – a customer emulator, a market emulator, and a DBMS running stored SQL procedures. Since Percolator is a client library running against Bigtable, our implementation is a combined customer/market emulator that calls into the Percolator library to perform operations against Bigtable. Percolator provides a low-level Get/Set/iterator API rather than a high-level SQL interface, so we created indexes and did all the ‘query planning’ by hand. 
>  TPC-E 通常有三个组件: 顾客模拟器、市场模拟器和一个运行 SQL 程序的 DBMS
>  因为 Percolator 是一个运行在 Bigtable 上的客户端库，故我们的 benchmark 仅结合了 TPC-E 的顾客模拟器和市场模拟器，它们调用 Percolator 库来对 Bigtable 执行操作
>  Percolator 提供的是低级的 Get/Set/iterator API 而不是高级的 SQL 接口，故我们手动创建索引并完成了所有 “查询规划” 工作

Since Percolator is an incremental processing system rather than an OLTP system, we don’t attempt to meet the TPC-E latency targets. Our average transaction latency is 2 to 5 seconds, but outliers can take several minutes. Outliers are caused by, for example, exponential backoff on conflicts and Bigtable tablet unavailability. 
>  因为 Percolator 是增量处理系统而不是 OLTP 系统，故我们不尝试满足 TPC-E 的延迟目标
>  Percolator 的平均事务延迟在 2-5s，但存在需要几分钟的离群值，这些离群值是由例如冲突时的指数回退和不可用的 Bigtable tablet 导致的

Finally, we made a small modification to the TPC-E transactions. In TPC-E, each trade result increases the broker’s commission and increments his trade count. Each broker services a hundred customers, so the average broker must be updated once every 5 seconds, which causes repeated write conflicts in Percolator. In Percolator, we would implement this feature by writing the increment to a side table and periodically aggregating each broker’s increments; for the benchmark, we choose to simply omit this write. 
>  我们对 TPC-E 事务进行了修改，TPC-E 中，每次交易会增加经纪人的佣金和交易计数，每个经纪人服务上百个顾客，故每个经纪人的状态平均隔 5s 更新一次，这会在 Percolator 中导致频繁的写入冲突
>  我们将事务的这一修改操作省略 (或者可以将这一增量写入一个 side table，并定期将 side table 的数据汇总)

![[pics/Percolator-Fig9.png]]

Figure 9 shows how the resource usage of Percolator scales as demand increases. We will measure resource usage in CPU cores since that is the limiting resource in our experimental environment. We were able to procure a small number of machines for testing, but our test Bigtable cell shares the disk resources of a much larger production cluster. As a result, disk bandwidth is not a factor in the system’s performance. In this experiment, we configured the benchmark with increasing numbers of customers and measured both the achieved performance and the number of cores used by all parts of the system including cores used for background maintenance such as Bigtable compactions. The relationship between performance and resource usage is essentially linear across several orders of magnitude, from 11 cores to 15,000 cores. 
>  Fig9 中，我们用使用的 CPU 核心数量度量资源使用数量
>  在试验中，我们不断增加客户数量，然后度量整个系统的实际性能和需要使用的核心数量 (包括了用于后台维护，例如 Bigtable 合并的核心数量)
>  性能和资源使用的关系在几个数量级上基本呈线性关系 (从 11 个核心到 15000 个核心)

This experiment also provides an opportunity to measure the overheads in Percolator relative to a DBMS. The fastest commercial TPC-E system today performs 3,183 tpsE using a single large shared-memory machine with 64 Intel Nehalem cores with 2 hyperthreads per core [33]. Our synthetic benchmark based on TPC-E performs 11,200 tps using 15,000 cores. This comparison is very rough: the Nehalem cores in the comparison machine are significantly faster than the cores in our test cell (small-scale testing on Nehalem processors shows that they are $20\mathrm{-}30\%$ faster per-thread compared to the cores in the test cluster). However, we estimate that Percolator uses roughly 30 times more CPU per transaction than the benchmark system. On a cost-per-transaction basis, the gap is likely much less than 30 since our test cluster uses cheaper, commodity hardware compared to the enterprise-class hardware in the reference machine. 
>  试验也提供了度量 Percolator 相对于 DBMS 的开销的机会
>  目前最快的 TPC-E 商业系统使用一台具有 64 个 Intel Nehalem 核心 (每个核心有两个超线程) 的单一大型共享内存机器，实现了 3,183 tpsE 的性能[33]
>  Percolator 则在我们的合成基准测试上，使用 15000 个核心达到了 11200 tps 的性能
>  这种比较非常粗略: 比较机器中的 Nehalem 核心比我们在测试环境中使用的核要快得多 (对 Nehalem 处理器的小规模测试表明，它们每线程的速度比测试集群中的核快 20%-30%)
>  我们估计 Percolator 每次事务锁使用的 CPU 核心数大约是 benchmark 系统的 30 倍，不过从 cost-per-transaction 的角度来看，差异实际上应该小于 30 倍，因为我们的测试集群核心更便宜 (商品级硬件 vs 企业级硬件)

The conventional wisdom on implementing databases is to “get close to the iron” and use hardware as directly as possible since even operating system structures like disk caches and schedulers make it hard to implement an efficient database [32]. In Percolator we not only interposed an operating system between our database and the hardware, but also several layers of software and network links. 
>  实现数据库的传统智慧是 “贴近硬件”，即尽可能直接使用硬件，因为即便是 OS 本身的结构 (例如磁盘缓存和调度器) 都会让实现高效数据库本身变得困难
>  Percolator 则不仅基于 OS 层，还基于数层软件和网络链接层之上

The conventional wisdom is correct: this arrangement has a cost. There are substantial overheads in preparing requests to go on the wire, sending them, and processing them on a remote machine. To illustrate these overheads in Percolator, consider the act of mutating the database. In a DBMS, this incurs a function call to store the data in memory and a system call to force the log to hardware controlled RAID array. In Percolator, a client performing a transaction commit sends multiple RPCs to Bigtable, which commits the mutation by logging it to 3 chunkservers, which make system calls to actually flush the data to disk. Later, that same data will be compacted into minor and major sstables, each of which will be again replicated to multiple chunkservers. 
>  传统智慧是对的: 基于更多的层是有代价的
>  从准备请求、发送请求，再到远程机器处理请求中，存在显著的开销
>  为了说明在 Percolator 中的这些开销，考虑一个修改数据库的操作，在 DBMS 中，这涉及了: 一个函数调用，将数据存储到内存中、一个系统调用，将日志写入 RADI 阵列控制的硬件中；在 Percolator 中，执行事务提交的客户端会向 Bigtable 发送多个 RPC, Bitable 通过向三个 chunkservers 写入日志来提交这一变更，chunkservers 本身会执行系统调用，将数据实际写入磁盘，之后，相同的数据会被压缩为 minor 和 major sstables，并且每个 sstable 都会被再次复制到多个 chunkservers 上

![[pics/Percolator-Fig10.png]]

The CPU inflation factor is the cost of our layering. In exchange, we get scalability (our fastest result, though not directly comparable to TPC-E, is more than 3x the current official record [33]), and we inherit the useful features of the systems we build upon, like resilience to failures. To demonstrate the latter, we ran the benchmark with 15 tablet servers and allowed the performance to stabilize. Figure 10 shows the performance of the system over time. The dip in performance at 17:09 corresponds to a failure event: we killed a third of the tablet servers. Performance drops immediately after the failure event but recovers as the tablets are reloaded by other tablet servers. We allowed the killed tablet servers to restart so performance eventually returns to the original level. 
>  这样的分层设计的代价就是 CPU inflation factor，但作为交换，我们获得了可拓展性，并且继承了底层系统的有用特性，例如对故障的容忍能力
>  为了展示这一点，我们在 15 个 tablet servers 上进行了 benchmark, Fig10 展示了系统随时间的性能表现，其中的性能下降对应于一次故障事件: 关闭了一台 tablet server，可以看到故障事件发生时，性能会迅速下降，但随着 tablets 被其他 tablet servers relaod，性能可以回升，重启了关闭的 tablet server 之后，性能最终恢复到原来的水平

# 4 Related Work 
Batch processing systems like MapReduce [13, 22, 24] are well suited for efficiently transforming or analyzing an entire corpus: these systems can simultaneously use a large number of machines to process huge amounts of data quickly. 
>  像 MapReduce 这样的批处理系统非常适合高效地转换或分析整个语料库：这些系统可以同时利用大量机器快速处理海量数据

Despite this scalability, re-running a MapReduce pipeline on each small batch of updates results in unacceptable latency and wasted work. Overlapping or pipelining the adjacent stages can reduce latency [10], but straggler shards still set the minimum time to complete the pipeline. 
>  尽管具有这种可扩展性，但对每次小批量更新重新运行 MapReduce pipeline 会导致不可接受的延迟和浪费的工作
>  重叠或流水线化相邻阶段可以减少延迟[10]，但 straggler shards 仍然决定了完成 pipeline 所需的最短时间

Percolator avoids the expense of repeated scans by, essentially, creating indexes on the keys used to cluster documents; one of criticisms leveled by Stonebraker and DeWitt in their initial critique of MapReduce [16] was that MapReduce did not support such indexes. 
>  Percolator 通过在用于聚类文档的键上创建索引来避免重复扫描的开销
>  Stonebraker 和 DeWitt 在其最初对 MapReduce 的批评中[16]提到的一个批评就是 MapReduce 不支持此类索引

Several proposed modifications to MapReduce [18, 26, 35] reduce the cost of processing changes to a repository by allowing workers to randomly read a base repository while mapping over only newly arrived work. To implement clustering in these systems, we would likely maintain a repository per clustering phase. Avoiding the need to re-map the entire repository would allow us to make batches smaller, reducing latency. 
>  一些对 MapReduce 的拟议修改[18, 26, 35]通过允许 workers 在 map 新到达的工作的同时随机读取基础存储库，从而降低了处理存储库更改的成本。
>  在这些系统中实现聚类，我们可能会为每个聚类阶段维护一个存储库
>  避免 re-map 整个存储库的需求将使我们可以创建更小的批次，从而减少延迟

DryadInc [31] attacks the same problem by reusing identical portions of the computation from previous runs and allowing the user to specify a merge function that combines new input with previous iterations’ outputs. These systems represent a middle-ground between mapping over the entire repository using MapReduce and processing a single document at a time with Percolator. 
>  DryadInc[31]通过重用先前运行中的相同计算部分并允许用户指定一个合并函数来结合新的输入与前几次迭代的输出，从而解决了相同的问题
>  这些系统代表了使用 MapReduce 处理整个存储库和使用 Percolator 逐个处理文档之间的中间立场

Databases satisfy many of the requirements of an incremental system: a RDBMS can make many independent and concurrent changes to a large corpus and provides a flexible language for expressing computation (SQL). In fact, Percolator presents the user with a database-like interface: it supports transactions, iterators, and secondary indexes. While Percolator provides distributed transactions, it is by no means a full-fledged DBMS: it lacks a query language, for example, as well as full relational operations such as join. Percolator is also designed to operate at much larger scales than existing parallel databases and to deal better with failed machines. Unlike Percolator, database systems tend to emphasize latency over throughput since a human is often waiting for the results of a database query. 
>  数据库满足许多对增量系统的要求: RDBMS 可以对大型的语料库执行许多独立且并发的更改，并提供了用于表示计算的灵活的语言 SQL
>  实际上，Percolator 为用户提供了一个类数据库的接口: 它支持事务，迭代器和次级索引
>  虽然 Percolator 提供了分布式事务，但它并不是完整的 DBMS: 它缺乏查询语言，例如缺少像 join 这样的完整关系操作
>  Percolator 的设计目标也是要在比现有的并行数据库都更大的规模上运行，并且需要更好地处理失败的机器
>  与 Percolator 不同，数据库系统更倾向于强调延迟而非吞吐量，因为人类往往需要等待数据库查询的结果

The organization of data in Percolator mirrors that of shared-nothing parallel databases [7, 15, 4]. Data is distributed across a number of commodity machines in shared-nothing fashion: the machines communicate only via explicit RPCs; no shared memory or shared disks are used. Data stored by Percolator is partitioned by Bigtable into tablets of contiguous rows which are distributed among machines; this mirrors the declustering performed by parallel databases. 
>  Percolator 中的数据组织方式与无共享并行数据库中的数据组织方式相同，数据以无共享的方式分布在多台普通机器上: 这些机器通过现实的 RPC 进行通信，不使用共享内存或共享磁盘
>  Percolator 存储的数据由 Bigtable 划分到行连续的 tablets 上，tablets 则分布式存储在多个机器上，这种方式与并行数据库执行的 declustering 相同

The transaction management of Percolator builds on a long line of work on distributed transactions for database systems. Percolator implements snapshot isolation [5] by extending multi-version timestamp ordering [6] across a distributed system using two-phase commit. 
>  Percolator 的事务管理建立在数据库系统对分布式事务的大量研究的基础上
>  Percolator 通过在分布式系统中，使用两阶段提交拓展了多版本时间戳排序，而实现了 snapshot isolation

An analogy can be drawn between the role of observers in Percolator to incrementally move the system towards a “clean” state and the incremental maintenance of materialized views in traditional databases (see Gupta and Mumick [21] for a survey of the field). In practice, while some indexing tasks like clustering documents by contents could be expressed in a form appropriate for incremental view maintenance it would likely be hard to express the transformation of a raw document into an indexed document in such a form. 
>  Percolator 中 observers 的角色可以类比于传统数据库系统中对 materialized vies 的增量维护 (增量式地将系统推向 “clean” state)
>  在实践中，虽然某些索引任务，例如对文档按内容聚类可以以 incremental view maintenance 的形式表达，但将原始文档转换为索引文档的过程，要用这一形式表达，则较为困难

The utility of parallel databases and, by extension, a system like Percolator, has been questioned several times [17] over their history. Hardware trends have, in the past, worked against parallel databases. CPUs have become so much faster than disks that a few CPUs in a shared-memory machine can drive enough disk heads to service required loads without the complexity of distributed transactions: the top TPC-E benchmark results today are achieved on large shared-memory machines connected to a SAN. This trend is beginning to reverse itself, however, as the enormous datasets like those Percolator is intended to process become far too large for a single shared-memory machine to handle. These datasets require a distributed solution that can scale to 1000s of machines, while existing parallel databases can utilize only 100s of machines [30]. Percolator provides a system that is scalable enough for Internet-sized datasets by sacrificing some (but not all) of the flexibility and low-latency of parallel databases. 
>  并行数据库的实用性，并由此延伸到类似 Percolator 系统这样的实用性，曾多次受到质疑
>  在过去，硬件趋势对并行数据库并不利，CPU 的速度比磁盘快得多，故在共享内存机器中，几个 CPU 就足以驱动足够的磁盘头来满足所需的工作负载，无需涉及分布式事务: 如今 TPC-E benchmark 的最好结果是在连接到 SAN (Storage Area Network) 的大型共享内存机器上达到的
>  但是，这一趋势正在逆转，因为像 Percolator 要处理的庞大数据量无法由单个共享内存的机器处理，这样的数据量需要能够拓展到数千台机器的解决方案，而现有的并行数据库只能利用数百台机器
>  Percolator 提供了一个足够可拓展的系统，可以处理互联网规模的数据集，但为此不得不牺牲一些 (但不是全部) 并行数据库系统的灵活性和低延迟特性

Distributed storage systems like Bigtable have the scalability and fault-tolerance properties of MapReduce but provide a more natural abstraction for storing a repository. Using a distributed storage system allows for low-latency updates since the system can change state by mutating the repository rather than rewriting it. However, Percolator is a data transformation system, not only a data storage system: it provides a way to structure computation to transform that data. In contrast, systems like Dynamo [14], Bigtable, and PNUTS [11] provide highly available data storage without the attendant mechanisms of transformation. These systems can also be grouped with the NoSQL databases (MongoDB [27], to name one of many): both offer higher performance and scale better than traditional databases, but provide weaker semantics. 
>  像 Bigtable 这样的分布式存储系统具备 MapReduce 这样的可拓展性和容错性质，且提供了更自然的存储库抽象
>  使用分布式存储系统可以实现低延迟更新，因为系统可以通过变更存储库来改变状态，而不需要重新写入存储库
>  但 Percolator 是一个数据转换系统，而不仅仅是一个数据存储系统: 它提供了一种结构化计算来转换数据的方式
>  相比之下，像 Dynamo, Bigtable, PNUTS 这样的系统提供了高度可用的数据存储服务，但没有提供附带的数据转换机制
>  以上讨论的这些系统都可以归类为 NoSQL 数据库 (例如 MongoDB) 的一部分: 比传统数据库的性能更高且可拓展性更好，但提供的语义较弱 (毕竟 No SQL)

Percolator extends Bigtable with multi-row, distributed transactions, and it provides the observer interface to allow applications to be structured around notifications of changed data. We considered building the new indexing system directly on Bigtable, but the complexity of reasoning about concurrent state modification without the aid of strong consistency was daunting. Percolator does not inherit all of Bigtable’s features: it has limited support for replication of tables across data centers, for example. Since Bigtable’s cross data center replication strategy is consistent only on a per-tablet basis, replication is likely to break invariants between writes in a distributed transaction. Unlike Dynamo and PNUTS which serve responses to users, Percolator is willing to accept the lower availability of a single data center in return for stricter consistency. 
>  Percolator 在 Bigtable 的基础上拓展了多行、分布式的事务，并为应用提供了 observer 接口，使得应用基于数据变更的通知进行构建
>  我们曾考虑直接在 Bigtable 上构建新的索引系统，但没有多行事务的强一致性保证，对并发状态修改进行推理会非常复杂
>  Percolator 并没有继承所有的 Bigtable 特性: 例如，它对跨数据中心的表的支持有限，因为 Bigtable 的跨数据中心复制策略仅在每个 tablet 的基础上保持一致性，故分布式事务中的写操作可能会破坏不变式
>  和 Dynamo, PNUTS 这种直接相应用户请求的服务不同，Percolator 可以接收仅在单个数据中心上运行 (较低的可用性)，以换取更强的一致性

Several research systems have, like Percolator, extended distributed storage systems to include strong consistency. Sinfonia [3] provides a transactional interface to a distributed repository. Earlier published versions of Sinfonia [2] also offered a notification mechanism similar to the Percolator’s observer model. Sinfonia and Percolator differ in their intended use: Sinfonia is designed to build distributed infrastructure while Percolator is intended to be used directly by applications (this probably explains why Sinfonia’s authors dropped its notification mechanism). Additionally, Sinfonia’s mini-transactions have limited semantics compared to the transactions provided by RDBMSs or Percolator: the user must specify a list of items to compare, read, and write prior to issuing the transaction. The mini-transactions are sufficient to create a wide variety of infrastructure but could be limiting for application builders. 
>  许多的研究系统，类似 Percolator，对分布式存储系统进行拓展，以实现强一致性
>  Sinfonia 为分布式存储库提供了一个事务化的接口，且其更早的版本也提供了类似 Percolator observer 模型的通知机制
>  Sinfonia 和 Percolator 的用途目的并不同: Sinfonia 的设计目的是构建分布式 infrastructure, Percolator 则直接由应用使用
>  此外，Sinfonia 的 mini 事务的语义相较于 RDBMS 或 Percolator 的事务更弱: 用户必须在发起事务之前，指定需要比较、读、写的 items，这样的 mini 事务足以构建 infrastructure，但对于应用构建是限制

CloudTPS [34], like Percolator, builds an ACID-compliant datastore on top of a distributed storage system (HBase [23] or Bigtable). Percolator and CloudTPS systems differ in design, however: the transaction management layer of CloudTPS is handled by an intermediate layer of servers called local transaction managers that cache mutations before they are persisted to the underlying distributed storage system. By contrast, Percolator uses clients, directly communicating with Bigtable, to coordinate transaction management. The focus of the systems is also different: CloudTPS is intended to be a backend for a website and, as such, has a stronger focus on latency and partition tolerance than Percolator. 
>  CloudTPS 基于分布式存储系统构建了一个符合 ACID 的数据存储服务
>  Percolator 和 CloudTPS 的差异在于设计: CloudTPS 的事务管理层由 servers 的中间层处理 (local transaction managers)，它们会在持久化存储到底层分布式存储系统之前，对变更进行缓存
>  相较之下，Percolator 使用 clients，直接和 Bigtable 通信，以协调事务管理
>  两个系统的关注点也不同: CloudTPS 用于网站后端，相较于 Percolator，更注重延迟和分区容忍

ElasTraS [12], a transactional data store, is architecturally similar to Percolator; the Owning Transaction Managers in ElasTraS are essentially tablet servers. Unlike Percolator, ElasTraS offers limited transactional semantics (Sinfonia-like mini-transactions) when dynamically partitioning the dataset and has no support for structuring computation. 
>  ElasTraS，一个事务性数据存储服务，结构上类似 Percolator，其中的 Owning Transaction Manager 本质是 tablet servers
>  和 Percolator 不同，ElasTraS 提供的是受限的事务语义 (类似 Sinfonia 的 mini-transactions)，且不支持结构化计算

# 5 Conclusion and Future Work 
We have built and deployed Percolator and it has been used to produce Google’s websearch index since April, 2010. The system achieved the goals we set for reducing the latency of indexing a single document with an acceptable increase in resource usage compared to the previous indexing system. 
>  Percolator 实现了我们的目标: 在适度增加资源使用量的情况下，降低索引单个文档的延迟

The TPC-E results suggest a promising direction for future investigation. We chose an architecture that scales linearly over many orders of magnitude on commodity machines, but we’ve seen that this costs a significant 30- fold overhead compared to traditional database architectures. We are very interested in exploring this tradeoff and characterizing the nature of this overhead: how much is fundamental to distributed storage systems, and how much can be optimized away? 
>  TPC-E 测试结果为未来的研究指明了一个有前景的方向
>  我们选择了一种在普通机器上能够线性扩展多个数量级的架构，但我们也发现，与传统的数据库架构相比，这带来了显著的 30 倍的额外开销
>  我们非常有兴趣探索这一权衡，并分析这种开销的本质：其中有多少是分布式存储系统固有的，又有多少是可以优化掉的？
