# Abstract 
Bigtable is a distributed storage system for managing structured data that is designed to scale to a very large size: petabytes of data across thousands of commodity servers. 
>  Bigtable 是一个管理结构化数据的分布式存储系统，其设计目标是在上千台商品机器上存储 PB 级别的数据

Many projects at Google store data in Bigtable, including web indexing, Google Earth, and Google Finance. These applications place very different demands on Bigtable, both in terms of data size (from URLs to web pages to satellite imagery) and latency requirements (from backend bulk processing to real-time data serving). Despite these varied demands, Bigtable has successfully provided a flexible, high-performance solution for all of these Google products.
>  基于 Bigtable 的不同应用对 Bigtable 在数据大小 (从 URL 到卫星图像) 和延迟 (从后端批处理到实时数据服务) 方面有不同的要求
>  即便要求多样，Bigtable 也为这些应用提供了灵活、高性能的解决方案

In this paper we describe the simple data model provided by Bigtable, which gives clients dynamic control over data layout and format, and we describe the design and implementation of Bigtable. 
>  本文描述 Bigtable 提供的数据模型 (为客户端提供对数据布局和格式的动态控制)，以及 Bigtable 的设计和实现

# 1 Introduction 
Over the last two and a half years we have designed, implemented, and deployed a distributed storage system for managing structured data at Google called Bigtable. Bigtable is designed to reliably scale to petabytes of data and thousands of machines. 

Bigtable has achieved several goals: wide applicability, scalability, high performance, and high availability. Bigtable is used by more than sixty Google products and projects, including Google Analytics, Google Finance, Orkut, Personalized Search, Writely, and Google Earth. These products use Bigtable for a variety of demanding workloads, which range from throughput-oriented batch-processing jobs to latency-sensitive serving of data to end users. The Bigtable clusters used by these products span a wide range of configurations, from a handful to thousands of servers, and store up to several hundred terabytes of data. 
>  Bigtable 实现了几大目标: 广泛的应用型、可拓展性、高性能、高可用性
>  基于 Bigtable 的产品具有多样的工作负载要求，从专注吞吐的批处理任务到延迟敏感的对终端用户的数据服务
>  这些产品使用的 Bigtable clusters 的配置大不相同，从几个到几千个服务器，最高存储几百个 TB 的数据

In many ways, Bigtable resembles a database: it shares many implementation strategies with databases. Parallel databases [14] and main-memory databases [13] have achieved scalability and high performance, but Bigtable provides a different interface than such systems. Bigtable does not support a full relational data model; instead, it provides clients with a simple data model that supports dynamic control over data layout and format, and allows clients to reason about the locality properties of the data represented in the underlying storage. 
>  Bigtable 在许多方面类似于数据库: 它采用了许多数据库的实现策略
>  并行数据库和主存数据库具有可拓展性和高性能，但 Bigtable 提供的接口和它们不同，Bigtable 不为客户端提供完全的关系数据模型，而是提供一个更简单的数据模型
>  该数据模型支持对数据布局和格式的动态控制，且允许客户端分析数据在底层存储的局部性 (客户端对数据的控制权限更高了)

Data is indexed using row and column names that can be arbitrary strings. Bigtable also treats data as uninterpreted strings, although clients often serialize various forms of structured and semi-structured data into these strings. Clients can control the locality of their data through careful choices in their schemas. Finally, Bigtable schema parameters let clients dynamically control whether to serve data out of memory or from disk. 
>  该数据模型中，数据通过行和列名索引，行和列名可以是任意字符串
>  Bigtable 将数据本身也视作未解释的字符串，客户端需要将各种形式的结构化或半结构化数据序列化为字符串
>  客户端通过选择 schemas 来控制其数据的局部性，通过 schema 参数动态控制是将数据从内存中提供还是从磁盘中提供

Section 2 describes the data model in more detail, and Section 3 provides an overview of the client API. Section 4 briefly describes the underlying Google infrastructure on which Bigtable depends. Section 5 describes the fundamentals of the Bigtable implementation, and Section 6 describes some of the refinements that we made to improve Bigtable’s performance. Section 7 provides measurements of Bigtable’s performance. We describe several examples of how Bigtable is used at Google in Section 8, and discuss some lessons we learned in designing and supporting Bigtable in Section 9. Finally, Section 10 describes related work, and Section 11 presents our conclusions. 

# 2 Data Model 
A Bigtable is a sparse, distributed, persistent multidimensional sorted map. The map is indexed by a row key, column key, and a timestamp; each value in the map is an uninterpreted array of bytes. 

```
(row: string, column: string, time:int64) -> string 
```

>  Bigtable 是一个稀疏的、分布式的、持久化的多维有序映射
>  该映射以 row key, column key, timestamp 为索引，映射中每个值都是未解释的字节序列

We settled on this data model after examining a variety of potential uses of a Bigtable-like system. 
>  我们在研究了一个类 Bigtable 系统的可能用途之后，选择了该数据模型

![[pics/Bigtable-Fig1.png]]

As one concrete example that drove some of our design decisions, suppose we want to keep a copy of a large collection of web pages and related information that could be used by many different projects; let us call this particular table the Webtable. In Webtable, we would use URLs as row keys, various aspects of web pages as column names, and store the contents of the web pages in the contents: column under the timestamps when they were fetched, as illustrated in Figure 1. 
>  例如，假设我们需要保存大量网页及其相关信息的副本，称这一 table 为 Webtable
>  在 Webtable 中，我们用 URL 作为 row key，网页的各种属性作为 column names，例如将网页内容存储在 `contents:` 列，且 cell 内还按时间戳索引

## Rows 
The row keys in a table are arbitrary strings (currently up to 64KB in size, although 10-100 bytes is a typical size for most of our users). Every read or write of data under a single row key is atomic (regardless of the number of different columns being read or written in the row), a design decision that makes it easier for clients to reason about the system’s behavior in the presence of concurrent updates to the same row. 
>  row keys 是任意字符串 (最大允许大小为 64KB，一般的大小为 10-100 bytes)
>  对单个 row key 数据下的读和写都是原子化的 (包括了该行的所有列)

Bigtable maintains data in lexicographic order by row key. The row range for a table is dynamically partitioned. Each row range is called a tablet, which is the unit of distribution and load balancing. 
>  Bigtable 以词表序排序 row keys
>  一个 table 的 row range 可以动态划分，每个 row range 称为一个 tablet, tablet 是分布和负载均衡的单位

As a result, reads of short row ranges are efficient and typically require communication with only a small number of machines. Clients can exploit this property by selecting their row keys so that they get good locality for their data accesses. For example, in Webtable, pages in the same domain are grouped together into contiguous rows by reversing the hostname components of the URLs. 
>  因此，对小的 row range 的读取仅涉及较少的 tablet，故仅需要和较少数量的机器通信，故较为高效
>  客户端可以通过对 row keys 的选择以利用该性质，获取更好的局部性 (让通常会被一起获取的相关数据具有接近的 row keys，使它们尽量被存储在同一个 tablet)

For example, we store data for maps.google.com/index.html under the key com.google.maps/index.html. Storing pages from the same domain near each other makes some host and domain analyses more efficient. 
>  例如，Webtable 中，相同域名的网页会置于连续的 rows (因为对 URL 进行了反转，例如 `maps.google.com/index.html -> com.google.maps/index.html`)，使得相同域名下的网页会在物理上被尽可能临近存储

>  tablet (row range) 是分布和负载均衡单元

## Column Families 
Column keys are grouped into sets called column families, which form the basic unit of access control. All data stored in a column family is usually of the same type (we compress data in the same column family together). A column family must be created before data can be stored under any column key in that family; after a family has been created, any column key within the family can be used. It is our intent that the number of distinct column families in a table be small (in the hundreds at most), and that families rarely change during operation. In contrast, a table may have an unbounded number of columns. 
>  Column keys 被分为多组 column families, column families 是基本的访问控制单元
>  存储在相同 column family 的数据通常是同一类型 (会被一起压缩)
>  在数据被存储 column family 的任意 column key 之前，必须先创建 column family 
>  我们的意图是让一个 table 中 column families 的数量尽量少 (最多几百个)，且 column families 很少变化
>  相反，一个 table 的 column 数量没有限制

>  column families 是访问控制单元

A column key is named using the following syntax: family: qualifier. Column family names must be printable, but qualifiers may be arbitrary strings. An example column family for the Webtable is language, which stores the language in which a web page was written. We use only one column key in the language family, and it stores each web page’s language ID. Another useful column family for this table is anchor; each column key in this family represents a single anchor, as shown in Figure 1. The qualifier is the name of the referring site; the cell contents is the link text. 
>  column key 的命名遵循格式 `family: qualifier`，其中 `family` 必须是可打印的，`qualifier` 则可以是任意字符串
>  Webtable 中，一个 column family 名为 `language`，它存储网页所写就的语言，`language` family 中仅有一个 column key，它存储每个网页的语言 ID
>  Webtable 中，另一个 column family 名为 `anchor`，其中每个 column key 表示一个单独的锚点，其 `qualifier` 是锚点所引用的站点的名称，cell 内容是链接文本

> [! info] Anchor
> 狭义上的锚点指 HTML 中的 `<a>` 标签，当 `<a>` 包含 `href` 属性，它就作为超链接的起点，指向其他网页的链接，例如 
> ```html
> <a href="https://www.example.com">点击访问示例网站</a>
> ```
>  锚点也可以指页面内的定位点，当其他标签包含 `id` 属性，它就作为页内的命名锚点，超链接可以通过 `#id` 指向该位置，例如
> ```html
> <h2 id="section1">第一部分</h2>
> <a href="#section1">跳转到第一部分</a>
> ```

Access control and both disk and memory accounting are performed at the column-family level. 
>  访问控制以及磁盘、内存记账都在 column family 级别进行 (用户、应用的访问权限按照 column family 为单位配置，磁盘和内存数据空间的占用情况按照 column family 为单位显示，不会更细粒度到各个 column keys)

In our Webtable example, these controls allow us to manage several different types of applications: some that add new base data, some that read the base data and create derived column families, and some that are only allowed to view existing data (and possibly not even to view all of the existing families for privacy reasons). 
>  Webtable 中，一些应用 (数据生产者) 拥有对 `content` 列族的写权限，负责添加新的基础数据；一些应用 (数据处理者) 拥有对 `content` 列族的读权限，和衍生 column families 的写权限，它们读取基础数据，创建衍生列族；一些应用 (数据消费者) 只有对部分列族的读权限

## Timestamps 
Each cell in a Bigtable can contain multiple versions of the same data; these versions are indexed by timestamp. Bigtable timestamps are 64-bit integers. They can be assigned by Bigtable, in which case they represent “real time” in microseconds, or be explicitly assigned by client applications. 
>  Bigtable 中每个 cell 可以存储多版本数据，各个版本数据通过时间戳索引
>  Bigtable 时间戳是 64 位整型，时间戳可以由 Bigtable 赋予 (表示以微秒为单位的 “真实时间”)，也可以由客户端应用程序显式分配

Applications that need to avoid collisions must generate unique timestamps themselves. Different versions of a cell are stored in decreasing timestamp order, so that the most recent versions can be read first. 
>  需要避免冲突的应用需要自行生成唯一的时间戳
>  cell 的不同版本数据按时间戳降序存储，故最近的版本 (时间戳最高) 最先被读取

To make the management of versioned data less onerous, we support two per-column-family settings that tell Bigtable to garbage-collect cell versions automatically. The client can specify either that only the last $n$ versions of a cell be kept, or that only new-enough versions be kept (e.g., only keep values that were written in the last seven days). 
>  Bigtable 支持自动垃圾回收 cell 的多版本数据，我们支持两种针对每列族的设置: 客户端可以指定仅保存 cell 的最后 $n$ 个版本，或者仅保存较新的版本 (例如保存过去 7 天内写入的值)

In our Webtable example, we set the timestamps of the crawled pages stored in the contents: column to the times at which these page versions were actually crawled. The garbage-collection mechanism described above lets us keep only the most recent three versions of every page. 
>  Webtable 中，我们 `contents:` 列中存储的网页内容的时间戳是该内容被实际爬取到的时间戳
>  上述的垃圾回收机制被用于保存网页内容的最近三个版本

Bigtable supports several other features that allow the user to manipulate data in more complex ways. First, Bigtable supports single-row transactions, which can be used to perform atomic read-modify-write sequences on data stored under a single row key. Bigtable does not currently support general transactions across row keys, although it provides an interface for batching writes across row keys at the clients. Second, Bigtable allows cells to be used as integer counters. Finally, Bigtable supports the execution of client-supplied scripts in the address spaces of the servers. The scripts are written in a language developed at Google for processing data called Sawzall [28]. At the moment, our Sawzall-based API does not allow client scripts to write back into Bigtable, but it does allow various forms of data transformation, filtering based on arbitrary expressions, and summarization via a variety of operators. 
>  Bigtable 还支持允许用户以更复杂方式管理数据的几个其他特性
>  其一，Bigtable 支持单行事务，即对单个 row key 下数据的读-修改-写都是原子化的，Bigtable 不支持多行事务，但也为客户端提供了跨多个行的批量写入接口 (即客户端将多个写操作请求打包发送，减少通信量，但注意批量写入没有原子性保证，例如可能某些行的写入成功，某些行的写入失败)
>  其二，Bigtable 允许 cells 被用于/解释为 (原子性的) 整数计数器 (即可以对 cell 执行原子性的 increment 和 decrement 操作，而无需 read-modify-write，这对于需要频繁更新计数，例如网站访问量，这样的场景非常有用，避免了 read-modify-write 带来的竞争条件和性能开销)
>  其三，Bigtable 支持在服务器地址空间执行客户端提供的脚本，脚本需要以 Sawzall 语言编写，目前的 API 不支持客户端脚本向 Bigtable 写回数据 (为了安全性和一致性)，但允许各种形式的数据转换，例如基于任意表达式的过滤 (比仅基于 keys 更灵活)，以及通过各种算子的总结 (计算平均值、总和等)。在服务器端执行脚本可以减少数据传输量，客户端只需要提供脚本，等待计算结果即可，不需要获取全部数据再本地计算

# 3 API 
The Bigtable API provides functions for creating and deleting tables and column families. It also provides functions for changing cluster, table, and column family metadata, such as access control rights. 
>  Bigtable API 支持创建和删除 tables 和 column families，支持更改 cluster, table, column family 元数据，例如访问控制权限

```cpp
// Open the table
Table *T = OpenOrDie("/bigtable/web/webtable");
// Write a new anchor and delete an old anchor
RowMutation r1(T, "com.cnn.www"); 
r1.Set("anchor:www.c-span.org", "CNN"); r1.Delete("anchor:www.abc.com"); 
Operation op;
Apply(&op, &r1);
```

Figure 2: Writing to Bigtable

Client applications can write or delete values in Bigtable, look up values from individual rows, or iterate over a subset of the data in a table. 
>  客户端应用可以在 Bigtable 中写、删除值，在一行中查找值，或对 table 的数据子集进行迭代

Figure 2 shows  C++ code that uses a `RowMutation` abstraction to perform a series of updates. (Irrelevant details were elided to keep the example short.) The call to `Apply` performs an atomic mutation to the Webtable: it adds one anchor to www.cnn.com and deletes a different anchor. 

```cpp
Scanner scanner(T);
ScanStream *stream;
stream = scanner.FetchColumnFamily("anchor"); 
stream->SetReturnAllVersions(); 
scanner.Lookup("com.cnn.www");
for (; !stream->Done(); stream->Next()) {
    printf("%s %s %lld %s\n", 
        scanner.RowName(), 
        stream->ColumnName(), 
        stream->MicroTimestamp(), 
        stream->Value());
}
```

Figure 3 shows C++ code that uses a `Scanner` abstraction to iterate over all anchors in a particular row. Clients can iterate over multiple column families, and there are several mechanisms for limiting the rows, columns, and timestamps produced by a scan. For example, we could restrict the scan above to only produce anchors whose columns match the regular expression `anchor:*.cnn.com`, or to only produce anchors whose timestamps fall within ten days of the current time. 

Bigtable can be used with MapReduce [12], a framework for running large-scale parallel computations developed at Google. We have written a set of wrappers that allow a Bigtable to be used both as an input source and as an output target for MapReduce jobs. 
>  Bigtable 可以结合 MapReduce 使用，我们编写了一些 wrappers，使得 Bigtable 可以作为 MapReduce 任务的输入源，也可以作为 MapReduce 任务的输出目标

# 4 Building Blocks 
Bigtable is built on several other pieces of Google infrastructure. Bigtable uses the distributed Google File System (GFS) [17] to store log and data files. A Bigtable cluster typically operates in a shared pool of machines that run a wide variety of other distributed applications, and Bigtable processes often share the same machines with processes from other applications. Bigtable depends on a cluster management system for scheduling jobs, managing resources on shared machines, dealing with machine failures, and monitoring machine status. 
>  Bigtable 使用 GFS 存储日志和数据文件
>  Bigtable 集群通常在共享的机器池中运行，该机器池同时还运行其他分布式应用
>  Bigtable 依赖于一个集群管理系统来调度任务、管理共享机器的资源、处理机器故障、监控机器状态

The Google SSTable file format is used internally to store Bigtable data. An SSTable provides a persistent, ordered immutable map from keys to values, where both keys and values are arbitrary byte strings. Operations are provided to look up the value associated with a specified key, and to iterate over all key/value pairs in a specified key range. 
>  Bigtable 数据以 Google SSTable  (Sorted String Table) 文件格式存储
>  SSTable 是一个持久化、有序、不可变的键值映射表，其中键和值都是任意字节串
>  SSTable 提供查找特定 key 的 value 的操作，以及迭代指定 key range 内所有 key-value pairs 的操作

Internally, each SSTable contains a sequence of blocks (typically each block is 64KB in size, but this is configurable). A block index (stored at the end of the SSTable) is used to locate blocks; the index is loaded into memory when the SSTable is opened. A lookup can be performed with a single disk seek: we first find the appropriate block by performing a binary search in the in-memory index, and then reading the appropriate block from disk. 
>  每个 SSTable 包含一系列 blocks (每个通常为 64KB)
>  SSTable 的末尾存储 block index，用于定位它关联的 blocks，当 SSTable 被 open 后，block index 会被载入内存 (后续就基于存内 block index 查找关联的 blocks，不需要再读盘)
>  一次查找操作只需要一次磁盘寻道: 首先在存内 block index 通过二分查找定位对应的 block (block index 按照 key 排序，故二分查找可以高效找到 key 具体在哪个 block)，然后从磁盘读取对应 block 即可

Optionally, an SSTable can be completely mapped into memory, which allows us to perform lookups and scans without touching disk. 
>  如果 SSTable 存储频繁被使用的数据，SSTable 可以被完整映射到内存中 (所有相关 blocks 都读到存内)，以高效查找和扫描

Bigtable relies on a highly-available and persistent distributed lock service called Chubby [8]. A Chubby service consists of five active replicas, one of which is elected to be the master and actively serve requests. The service is live when a majority of the replicas are running and can communicate with each other. Chubby uses the Paxos algorithm [9, 23] to keep its replicas consistent in the face of failure. Chubby provides a namespace that consists of directories and small files. Each directory or file can be used as a lock, and reads and writes to a file are atomic. The Chubby client library provides consistent caching of Chubby files. Each Chubby client maintains a session with a Chubby service. A client’s session expires if it is unable to renew its session lease within the lease expiration time. When a client’s session expires, it loses any locks and open handles. Chubby clients can also register callbacks on Chubby files and directories for notification of changes or session expiration. 
>  Bigtable 依赖于高可用和持久化的分布式锁服务: Chubby
>  Chubby 服务包含五个活跃副本，一个副本作为 master 来服务请求，Chubby 只要多数副本活跃且可通讯就可以运行
>  Chubby 基于 Paxos 保持副本的一致性
>  Chubby 提供了由小文件和目录组成的命名空间 (抽象)，其中每个目录和文件都可以用作锁 (或者说被上锁)，对文件的读写是原子化的
>  Chubby client library 提供了 Chubby 文件的一致性缓存 (客户端在本地有文件的缓存数据，且保证是一致的，即如果数据有更新，客户端会看到更新)
>  每个 Chubby client 维护一个和 Chubby 服务的会话，当 client 的会话的 lease 过期后，会话就终止，会话的所有锁和打开的句柄都会被释放
>  Chubby client 可以在 Chubby 文件和目录上注册 callbacks，在文件被更改或自身会话过期时通知 client

Bigtable uses Chubby for a variety of tasks: to ensure that there is at most one active master at any time; to store the bootstrap location of Bigtable data (see Section 5.1); to discover tablet servers and finalize tablet server deaths (see Section 5.2); to store Bigtable schema information (the column family information for each table); and to store access control lists. If Chubby becomes unavailable for an extended period of time, Bigtable becomes unavailable. 
>  Bigtable 使用 Chubby 执行许多任务: 
>  - 确保每个时刻只有一个活跃的 master (master 在 Chubby 获取锁，如果 master 崩溃，其会话过期后，锁会被释放)
>  - 存储 Bigtable 数据的数据的引导位置 (即 Bigtable 的元数据表: root tablet，的位置，root tablet 被获取后，再根据它获取其他 tablet 的位置
>  - 发现 tablet 以及确认 tablet server 的死亡 (tablet server 在 Chubby 中注册自己)
>  - 存储 Bigtable 模式信息 (每个 table 的 column family 信息)
>  - 存储访问控制列表
>  如果 Chubby 不可用，Bigtable 会随之不可用

We recently measured this effect in 14 Bigtable clusters spanning 11 Chubby instances. The average percentage of Bigtable server hours during which some data stored in Bigtable was not available due to Chubby unavailability (caused by either Chubby outages or network issues) was $0 . 0 0 4 7 \%$ . The percentage for the single cluster that was most affected by Chubby unavailability was $0 . 0 3 2 6 \%$ . 
>  我们在 14 个 Bigtable 集群 (分布在 11 个 Chubby 实例上，Google 的基础设施是高度共享的) 进行了度量
>  Bigtable 中存储的数据由于 Chubby 的不可用 (Chubby 崩溃或网络问题) 而不可用的平均时间占总运行时间的 $0.0047\%$
>  其中最受影响的集群的不可用时间占总运行时间的 $0.0326\%$

# 5 Implementation 
The Bigtable implementation has three major components: a library that is linked into every client, one master server, and many tablet servers. Tablet servers can be dynamically added (or removed) from a cluster to accommodate changes in workloads. 
>  Bigtable 实现包含三个主要成分: 一个链接到每个 client 的库、一个 master server、多个 tablet servers
>  tablet servers 可以根据工作负载动态的添加到集群或从集群移除

The master is responsible for assigning tablets to tablet servers, detecting the addition and expiration of tablet servers, balancing tablet-server load, and garbage collection of files in GFS. In addition, it handles schema changes such as table and column family creations. 
>  master server 负责将 tablets 分配给 tablet servers、检测 tablet servers 的添加和过期、平衡 tablet servers 的负载、垃圾回收 GFS 中的文件
>  master server 还负责处理模式变化，例如 table 和 column family 创建

Each tablet server manages a set of tablets (typically we have somewhere between ten to a thousand tablets per tablet server). The tablet server handles read and write requests to the tablets that it has loaded, and also splits tablets that have grown too large. 
>  每个 tablet server 管理一组 tablets (通常数量在 10 到 1000)，tablet server 需要处理对这些 tablets 的读和写请求，同时需要将拓展得过大的 tablets 进行划分

As with many single-master distributed storage systems [17, 21], client data does not move through the master: clients communicate directly with tablet servers for reads and writes. Because Bigtable clients do not rely on the master for tablet location information, most clients never communicate with the master. As a result, the master is lightly loaded in practice. 
>  和许多 single-master 的分布式存储系统一样，Bigtable 中，客户端数据不会通过 master 移动: 客户端直接和 tablet servers 沟通以进行读写
>  因为 Bigtable 客户端不依赖于 master 来获取 tablet 位置信息，许多客户端从不和 master 沟通，因此实际中 master 的负载很轻量

A Bigtable cluster stores a number of tables. Each table consists of a set of tablets, and each tablet contains all data associated with a row range. Initially, each table consists of just one tablet. As a table grows, it is automatically split into multiple tablets, each approximately 100-200MB in size by default. 
>  一个 Bigtable cluster 存储许多 tables，其中每个 table 由多个 tablets 组成，而每个 tablet 则包含对应 row range 内的所有数据
>  开始时，每个 table 仅有一个 tablet，随着 table 增大，该 tablet 会自动划分为多个 tablets，每个的大小默认大约在 100-200MB

## 5.1 Tablet Location 

![[pics/Bigtable-Fig4.png]]

We use a three-level hierarchy analogous to that of a $B^+$ - tree [10] to store tablet location information (Figure 4). 
>  tablet 的位置信息的存储采用三级架构，类似于 B+ 树

The first level is a file stored in Chubby that contains the location of the root tablet. The root tablet contains the location of all tablets in a special METADATA table. Each METADATA tablet contains the location of a set of user tablets. The root tablet is just the first tablet in the METADATA table, but is treated specially—it is never split—to ensure that the tablet location hierarchy has no more than three levels. 
>  第一级是一个存储在 Chubby 中的文件，它包含了 root tablet 的位置信息
>  root tablet 包含了 `METADATA` table 中的所有 tablets 的位置信息 (第二级)，其中每个 `METADATA` tablet 包含一组用户 tablets  的位置信息 (第三级)
>  root tablet 实际上就是 `METADATA` table 的第一个 tablet，但进行了特殊对待——永远不会被划分——以确保 tablet 位置信息层次不会超过三层

>  三级存储:
>  - 第一级 Chubby: 存储 root tablet 的位置
>  - 第二级 root tablet: 存储所有 `METADATA` tablets 的位置
>  - 第三级 `METADATA` tablet: 存储用户数据 tablets 的位置

The METADATA table stores the location of a tablet under a row key that is an encoding of the tablet’s table identifier and its end row. Each METADATA row stores approximately 1KB of data in memory. With a modest limit of 128 MB METADATA tablets, our three-level location scheme is sufficient to address $2 ^ { 3 4 }$ tablets (or $2 ^ { 6 1 }$ bytes in $1 2 8 \mathrm { M B }$ tablets). 
>  `METADATA` table 中，一个 tablet 的位置存储的 row 的 key 是该 tablet 所属的 table 的 id 和该 tablet 的结束行的编码
>  每个 `METADATA` row 在存内存储大约 1KB 的数据，`METADATA` tablets 的总量限制在 128MB 时，该方案可以寻址 $2^{34}$ 个 tablets，如果每个 `METADATA` tablet 的大小拓展到 128MB，该方案可以寻址 $2^{61}$ 个 tablets

>  核心思想是有限的元数据可以索引海量的用户数据

The client library caches tablet locations. If the client does not know the location of a tablet, or if it discovers that cached location information is incorrect, then it recursively moves up the tablet location hierarchy. If the client’s cache is empty, the location algorithm requires three network round-trips, including one read from Chubby. If the client’s cache is stale, the location algorithm could take up to six round-trips, because stale cache entries are only discovered upon misses (assuming that METADATA tablets do not move very frequently). Although tablet locations are stored in memory, so no GFS accesses are required, we further reduce this cost in the common case by having the client library prefetch tablet locations: it reads the metadata for more than one tablet whenever it reads the METADATA table. 
>  客户端库会缓存 tablet 位置
>  如果客户端不知道所需的 tablet 位置，或者发现其缓存的位置有误，它会自上述层次向上搜索
>  如果客户端缓存为空，则定位算法需要三次网络往返: 
>  - 第一次: 向 Chubby 发送请求，获取 root tablet 位置
>  - 第二次: 向 root tablet server 发送请求，获取对应 `METADATA` tablet 位置
>  - 第三次: 向 `METADATA` tablet server 发送请求，获取用户数据 tablet 位置
>  如果客户端缓存过期，则定位算法最多需要六次往返，因为客户端只有在查找失败的时候才会发现缓存过期 (假设 `METADATA` 不频繁移动)
>  root tablet server 和 `METADATA` tablet server 都将 tablet 存在内存中，不需要 GFS 访问即可获取位置信息，我们还在客户端库添加了预取策略: 它在读 `METADATA` tablet 时，会一次读多个 tablet 的位置信息

We also store secondary information in the METADATA table, including a log of all events pertaining to each tablet (such as when a server begins serving it). This information is helpful for debugging and performance analysis. 
>  我们还在 `METADATA` table 存储了二级信息，包括了与每个 tablet 相关的所有事件的日志，用于 debugging 和性能分析

## 5.2 Tablet Assignment 
Each tablet is assigned to one tablet server at a time. The master keeps track of the set of live tablet servers, and the current assignment of tablets to tablet servers, including which tablets are unassigned. When a tablet is unassigned, and a tablet server with sufficient room for the tablet is available, the master assigns the tablet by sending a tablet load request to the tablet server. 
>  master server 追踪所有活跃的 tablet servers，以及当前对各个 tablet server 的 tablet 分配情况，以及还有哪些 tablet 尚未被分配
>  如果存在尚未被分配的 tablet，且存在具有可用空间的 tablet server 时，master 就向该 tablet server 发送 tablet load requets，将该 tablet 分配给该 tablet server

Bigtable uses Chubby to keep track of tablet servers. When a tablet server starts, it creates, and acquires an exclusive lock on, a uniquely-named file in a specific Chubby directory. The master monitors this directory (the servers directory) to discover tablet servers. A tablet server stops serving its tablets if it loses its exclusive lock: e.g., due to a network partition that caused the server to lose its Chubby session. (Chubby provides an efficient mechanism that allows a tablet server to check whether it still holds its lock without incurring network traffic.) 
>  Bigtable 使用 Chubby 来追踪 tablet servers
>  一个 tablet server 启动后，会在特定的 Chubby 目录 (`servers` 目录) 下对一个名称唯一的文件获取互斥锁
>  master server 监控该目录，以判断哪些 tablet server 是活跃的
>  tablet server 如果发现它自己失去了其互斥锁 (例如因为网络导致 Chubby 会话断开，Chubby 提供了一个高效的机制，使得 tablet server 在不需要产生网络流量的情况下可以检查它是否仍然持有锁)，则 tablet server 会停止服务

A tablet server will attempt to reacquire an exclusive lock on its file as long as the file still exists. If the file no longer exists, then the tablet server will never be able to serve again, so it kills itself. 
>  如果对应的文件仍然存在，tablet server 会重新尝试获取互斥锁
>  如果文件不存在，则 tablet server 需要自行终止，不再提供服务

Whenever a tablet server terminates (e.g., because the cluster management system is removing the tablet server’s machine from the cluster), it attempts to release its lock so that the master will reassign its tablets more quickly. 
>  当 tablet server 因为某些原因需要终止时 (例如集群管理系统将 tablet server 所在的机器移出集群)，它会尝试释放自己持有的锁，便于 master 迅速将其 tablets 重新分配

The master is responsible for detecting when a tablet server is no longer serving its tablets, and for reassigning those tablets as soon as possible. To detect when a tablet server is no longer serving its tablets, the master periodically asks each tablet server for the status of its lock. If a tablet server reports that it has lost its lock, or if the master was unable to reach a server during its last several attempts, the master attempts to acquire an exclusive lock on the server’s file. If the master is able to acquire the lock, then Chubby is live and the tablet server is either dead or having trouble reaching Chubby, so the master ensures that the tablet server can never serve again by deleting its server file. Once a server’s file has been deleted, the master can move all the tablets that were previously assigned to that server into the set of unassigned tablets.
>  master 需要定期检查某个 tablet server 是否已经停止服务，然后将其负责的 tablets 尽快重新分配
>  故 master 会周期性检查各个 tablet server 的锁，如果发现该锁被释放，或者 master 在最近几次尝试中都无法联系 tablet server，则 master 会自己尝试获取该锁
>  如果 master 获取了该锁，说明要么 tablet server 故障，要么由于网络问题无法联系 Chubby 服务，此时 master 会删除该文件，确保 tablet server 不会再提供服务，然后将之前分配的 tablets 记作未分配

To ensure that a Bigtable cluster is not vulnerable to networking issues between the master and Chubby, the master kills itself if its Chubby session expires. However, as described above, master failures do not change the assignment of tablets to tablet servers. 
>  为了确保 Bigtable 集群不会由于 master 和 Chubby 之间的网络问题而出现故障，master 会在其 Chubby 会话过期后终止自己
>  master 的故障不会改变 tablets 的分配情况

When a master is started by the cluster management system, it needs to discover the current tablet assignments before it can change them. The master executes the following steps at startup. (1) The master grabs a unique master lock in Chubby, which prevents concurrent master instantiations. (2) The master scans the servers directory in Chubby to find the live servers. (3) The master communicates with every live tablet server to discover what tablets are already assigned to each server. (4) The master scans the METADATA table to learn the set of tablets. Whenever this scan encounters a tablet that is not already assigned, the master adds the tablet to the set of unassigned tablets, which makes the tablet eligible for tablet assignment. 
>  新的 master 被集群管理系统启动后，它需要先确认当前的 tablet 分配情况
>  故它会在启动时执行:
>  1. 在 Chubby 中获取唯一的 master lock，以避免其他 master 启动
>  2. 扫描 Chubby 的 `serveres` 目录，确认活跃的 tablet servers
>  3. 和活跃的 tablet server 通讯，确定 tablets 的分配情况
>  4. 扫描 `METADATA` table，确定总的 tablets 集合，当发现没有被分配的 tablet 时，它将该 tablet 加入未分配集合，使得该 tablet 可以在未来被分配

One complication is that the scan of the METADATA table cannot happen until the METADATA tablets have been assigned. Therefore, before starting this scan (step 4), the master adds the root tablet to the set of unassigned tablets if an assignment for the root tablet was not discovered during step 3. This addition ensures that the root tablet will be assigned. Because the root tablet contains the names of all METADATA tablets, the master knows about all of them after it has scanned the root tablet. 
>  一个问题是 `METADATA` table 只有在 `METADATA` tablets 都被分配后，才能被扫描
>  因此，在 step 4 之前，master 会先将 root tablet 加入未分配集合 (如果在 step 3 没有发现有 tablet server 当前被分配了 root tablet)
>  这确保了 root tablet 将会被分配，而因为 root tablet 包含了所有 `METADATA` tablets 的名字，故在 root tablet 被分配之后，master 就可以获取所有 `METADATA` tablets 的名字和位置
>  (此时，如果有未分配的 `METADATA` tablet，就先分配，然后再读，最终 master 可以知道总的 tablets 集合及其各自的位置情况，完成初始化)

The set of existing tablets only changes when a table is created or deleted, two existing tablets are merged to form one larger tablet, or an existing tablet is split into two smaller tablets. The master is able to keep track of these changes because it initiates all but the last. Tablet splits are treated specially since they are initiated by a tablet server. The tablet server commits the split by recording information for the new tablet in the METADATA table. When the split has committed, it notifies the master. 
>  现存的 tablets 集合仅在 table 被创建或被删除时才变化
>  两个 tablets 可以被合并为一个更大的 tablet，一个 tablet 也可以被划分为两个更小的 tablets
>  master 负责发起 table 创建、删除和 tablet 的合并
>  tablet server 负责自行发起 tablet 划分，tablet server 通过再 `METADATA` table 记录新的 tablet 信息来提交该划分操作，操作提交后，tablet server 会通知 master

In case the split notification is lost (either because the tablet server or the master died), the master detects the new tablet when it asks a tablet server to load the tablet that has now split. The tablet server will notify the master of the split, because the tablet entry it finds in the METADATA table will specify only a portion of the tablet that the master asked it to load. 
>  如果通知丢失，master 会在向 tablet server 请求已经被划分的 tablet 时，知道划分操作已经被执行 (tablet server 会在按照 mater 的请求从 `METADATA` table 中寻找 tablet entry 时，发现该 tablet entry 仅匹配 master 请求的 tablet 的一部分，故发现 tablet 已经被划分，进而通知 master)

## 5.3 Tablet Serving 
The persistent state of a tablet is stored in GFS, as illustrated in Figure 5. Updates are committed to a commit log that stores redo records. Of these updates, the recently committed ones are stored in memory in a sorted buffer called a memtable; the older updates are stored in a sequence of SSTables. To recover a tablet, a tablet server reads its metadata from the METADATA table. This metadata contains the list of SSTables that comprise a tablet and a set of a redo points, which are pointers into any commit logs that may contain data for the tablet. The server reads the indices of the SSTables into memory and reconstructs the memtable by applying all of the updates that have committed since the redo points. 

When a write operation arrives at a tablet server, the server checks that it is well-formed, and that the sender is authorized to perform the mutation. Authorization is performed by reading the list of permitted writers from a Chubby file (which is almost always a hit in the Chubby client cache). A valid mutation is written to the commit log. Group commit is used to improve the throughput of lots of small mutations [13, 16]. After the write has been committed, its contents are inserted into the memtable. 

When a read operation arrives at a tablet server, it is similarly checked for well-formedness and proper authorization. A valid read operation is executed on a merged view of the sequence of SSTables and the memtable. Since the SSTables and the memtable are lexicographically sorted data structures, the merged view can be formed efficiently. 

Incoming read and write operations can continue while tablets are split and merged. 

## 5.4 Compactions 
As write operations execute, the size of the memtable increases. When the memtable size reaches a threshold, the memtable is frozen, a new memtable is created, and the frozen memtable is converted to an SSTable and written to GFS. This minor compaction process has two goals: it shrinks the memory usage of the tablet server, and it reduces the amount of data that has to be read from the commit log during recovery if this server dies. Incoming read and write operations can continue while compactions occur. 
Every minor compaction creates a new SSTable. If this behavior continued unchecked, read operations might need to merge updates from an arbitrary number of SSTables. Instead, we bound the number of such files by periodically executing a merging compaction in the background. A merging compaction reads the contents of a few SSTables and the memtable, and writes out a new SSTable. The input SSTables and memtable can be discarded as soon as the compaction has finished. 
A merging compaction that rewrites all SSTables into exactly one SSTable is called a major compaction. SSTables produced by non-major compactions can contain special deletion entries that suppress deleted data in older SSTables that are still live. A major compaction, on the other hand, produces an SSTable that contains no deletion information or deleted data. Bigtable cycles through all of its tablets and regularly applies major compactions to them. These major compactions allow Bigtable to reclaim resources used by deleted data, and also allow it to ensure that deleted data disappears from the system in a timely fashion, which is important for services that store sensitive data. 
# 6 Refinements 
The implementation described in the previous section required a number of refinements to achieve the high performance, availability, and reliability required by our users. This section describes portions of the implementation in more detail in order to highlight these refinements. 
# Locality groups 
Clients can group multiple column families together into a locality group. A separate SSTable is generated for each locality group in each tablet. Segregating column families that are not typically accessed together into separate locality groups enables more efficient reads. For example, page metadata in Webtable (such as language and checksums) can be in one locality group, and the contents of the page can be in a different group: an application that wants to read the metadata does not need to read through all of the page contents. 
In addition, some useful tuning parameters can be specified on a per-locality group basis. For example, a locality group can be declared to be in-memory. SSTables for in-memory locality groups are loaded lazily into the memory of the tablet server. Once loaded, column families that belong to such locality groups can be read without accessing the disk. This feature is useful for small pieces of data that are accessed frequently: we use it internally for the location column family in the METADATA table. 
# Caching for read performance 
To improve read performance, tablet servers use two levels of caching. The Scan Cache is a higher-level cache that caches the key-value pairs returned by the SSTable interface to the tablet server code. The Block Cache is a lower-level cache that caches SSTables blocks that were read from GFS. The Scan Cache is most useful for applications that tend to read the same data repeatedly. The Block Cache is useful for applications that tend to read data that is close to the data they recently read (e.g., sequential reads, or random reads of different columns in the same locality group within a hot row). 
# Compression 
Clients can control whether or not the SSTables for a locality group are compressed, and if so, which compression format is used. The user-specified compression format is applied to each SSTable block (whose size is controllable via a locality group specific tuning parameter). Although we lose some space by compressing each block separately, we benefit in that small portions of an SSTable can be read without decompressing the entire file. Many clients use a two-pass custom compression scheme. The first pass uses Bentley and McIlroy’s scheme [6], which compresses long common strings across a large window. The second pass uses a fast compression algorithm that looks for repetitions in a small $1 6 ~ \mathrm { K B }$ window of the data. Both compression passes are very fast—they encode at $1 0 0 { - } 2 0 0 \mathbf { M B } / \mathrm { s }$ , and decode at $4 0 0 { - } 1 0 0 0 \mathrm { M B / s }$ on modern machines. 
Even though we emphasized speed instead of space reduction when choosing our compression algorithms, this two-pass compression scheme does surprisingly well. For example, in Webtable, we use this compression scheme to store Web page contents. In one experiment, we stored a large number of documents in a compressed locality group. For the purposes of the experiment, we limited ourselves to one version of each document instead of storing all versions available to us. The scheme achieved a 10-to-1 reduction in space. This is much better than typical Gzip reductions of 3-to-1 or 4-to-1 on HTML pages because of the way Webtable rows are laid out: all pages from a single host are stored close to each other. This allows the Bentley-McIlroy algorithm to identify large amounts of shared boilerplate in pages from the same host. Many applications, not just Webtable, choose their row names so that similar data ends up clustered, and therefore achieve very good compression ratios. Compression ratios get even better when we store multiple versions of the same value in Bigtable. 
# Bloom filters 
As described in Section 5.3, a read operation has to read from all SSTables that make up the state of a tablet. If these SSTables are not in memory, we may end up doing many disk accesses. We reduce the number of accesses by allowing clients to specify that Bloom filters [7] should be created for SSTables in a particular locality group. A Bloom filter allows us to ask whether an SSTable might contain any data for a specified row/column pair. For certain applications, a small amount of tablet server memory used for storing Bloom filters drastically reduces the number of disk seeks required for read operations. Our use of Bloom filters also implies that most lookups for non-existent rows or columns do not need to touch disk. 
# Commit-log implementation 
If we kept the commit log for each tablet in a separate log file, a very large number of files would be written concurrently in GFS. Depending on the underlying file system implementation on each GFS server, these writes could cause a large number of disk seeks to write to the different physical log files. In addition, having separate log files per tablet also reduces the effectiveness of the group commit optimization, since groups would tend to be smaller. To fix these issues, we append mutations to a single commit log per tablet server, co-mingling mutations for different tablets in the same physical log file [18, 20]. 
Using one log provides significant performance benefits during normal operation, but it complicates recovery. When a tablet server dies, the tablets that it served will be moved to a large number of other tablet servers: each server typically loads a small number of the original server’s tablets. To recover the state for a tablet, the new tablet server needs to reapply the mutations for that tablet from the commit log written by the original tablet server. However, the mutations for these tablets were co-mingled in the same physical log file. One approach would be for each new tablet server to read this full commit log file and apply just the entries needed for the tablets it needs to recover. However, under such a scheme, if 100 machines were each assigned a single tablet from a failed tablet server, then the log file would be read 100 times (once by each server). 
We avoid duplicating log reads by first sorting the commit log entries in order of the keys ⟨table, row name, log sequence number⟩. In the sorted output, all mutations for a particular tablet are contiguous and can therefore be read efficiently with one disk seek followed by a sequential read. To parallelize the sorting, we partition the log file into $6 4 ~ \mathrm { M B }$ segments, and sort each segment in parallel on different tablet servers. This sorting process is coordinated by the master and is initiated when a tablet server indicates that it needs to recover mutations from some commit log file. 
Writing commit logs to GFS sometimes causes performance hiccups for a variety of reasons (e.g., a GFS server machine involved in the write crashes, or the network paths traversed to reach the particular set of three GFS servers is suffering network congestion, or is heavily loaded). To protect mutations from GFS latency spikes, each tablet server actually has two log writing threads, each writing to its own log file; only one of these two threads is actively in use at a time. If writes to the active log file are performing poorly, the log file writing is switched to the other thread, and mutations that are in the commit log queue are written by the newly active log writing thread. Log entries contain sequence numbers to allow the recovery process to elide duplicated entries resulting from this log switching process. 
# Speeding up tablet recovery 
If the master moves a tablet from one tablet server to another, the source tablet server first does a minor compaction on that tablet. This compaction reduces recovery time by reducing the amount of uncompacted state in the tablet server’s commit log. After finishing this compaction, the tablet server stops serving the tablet. Before it actually unloads the tablet, the tablet server does another (usually very fast) minor compaction to eliminate any remaining uncompacted state in the tablet server’s log that arrived while the first minor compaction was being performed. After this second minor compaction is complete, the tablet can be loaded on another tablet server without requiring any recovery of log entries. 
# Exploiting immutability 
Besides the SSTable caches, various other parts of the Bigtable system have been simplified by the fact that all of the SSTables that we generate are immutable. For example, we do not need any synchronization of accesses to the file system when reading from SSTables. As a result, concurrency control over rows can be implemented very efficiently. The only mutable data structure that is accessed by both reads and writes is the memtable. To reduce contention during reads of the memtable, we make each memtable row copy-on-write and allow reads and writes to proceed in parallel. 
Since SSTables are immutable, the problem of permanently removing deleted data is transformed to garbage collecting obsolete SSTables. Each tablet’s SSTables are registered in the METADATA table. The master removes obsolete SSTables as a mark-and-sweep garbage collection [25] over the set of SSTables, where the METADATA table contains the set of roots. 
Finally, the immutability of SSTables enables us to split tablets quickly. Instead of generating a new set of SSTables for each child tablet, we let the child tablets share the SSTables of the parent tablet. 
# 7 Performance Evaluation 
We set up a Bigtable cluster with $N$ tablet servers to measure the performance and scalability of Bigtable as $N$ is varied. The tablet servers were configured to use 1 GB of memory and to write to a GFS cell consisting of 1786 machines with two 400 GB IDE hard drives each. $N$ client machines generated the Bigtable load used for these tests. (We used the same number of clients as tablet servers to ensure that clients were never a bottleneck.) Each machine had two dual-core Opteron 2 GHz chips, enough physical memory to hold the working set of all running processes, and a single gigabit Ethernet link. The machines were arranged in a two-level tree-shaped switched network with approximately 100-200 Gbps of aggregate bandwidth available at the root. All of the machines were in the same hosting facility and therefore the round-trip time between any pair of machines was less than a millisecond. 
The tablet servers and master, test clients, and GFS servers all ran on the same set of machines. Every machine ran a GFS server. Some of the machines also ran either a tablet server, or a client process, or processes from other jobs that were using the pool at the same time as these experiments. 
$R$ is the distinct number of Bigtable row keys involved in the test. $R$ was chosen so that each benchmark read or wrote approximately 1 GB of data per tablet server. 
The sequential write benchmark used row keys with names 0 to $R - 1$ . This space of row keys was partitioned into $1 0 N$ equal-sized ranges. These ranges were assigned to the $N$ clients by a central scheduler that assigned the next available range to a client as soon as the client finished processing the previous range assigned to it. This dynamic assignment helped mitigate the effects of performance variations caused by other processes running on the client machines. We wrote a single string under each row key. Each string was generated randomly and was therefore uncompressible. In addition, strings under different row key were distinct, so no cross-row compression was possible. The random write benchmark was similar except that the row key was hashed modulo $R$ immediately before writing so that the write load was spread roughly uniformly across the entire row space for the entire duration of the benchmark. 
<html><body><table><tr><td rowspan="2">Experiment</td><td colspan="4"># of Tablet Servers</td></tr><tr><td>1</td><td> 50</td><td>250</td><td> 500</td></tr><tr><td>random reads</td><td>1212</td><td>593</td><td>479</td><td>241</td></tr><tr><td>random reads (mem)</td><td>10811</td><td>8511</td><td>8000</td><td>6250</td></tr><tr><td>random writes.</td><td>8850</td><td>3745</td><td>3425</td><td>2000</td></tr><tr><td>sequential reads</td><td>4425</td><td>2463</td><td>2625</td><td>2469</td></tr><tr><td>sequential writess</td><td>8547</td><td>3623</td><td>2451</td><td>1905</td></tr><tr><td>scans</td><td>15385</td><td>10526</td><td>9524</td><td>7843</td></tr></table></body></html> 
![](https://cdn-mineru.openxlab.org.cn/extract/5cf648ed-fd33-4470-8985-1ac3b24bba26/aadb50a443a93c8d1dc8b7d7acef7350b99e9b238d50ac58495ff363cc5ab837.jpg) 
Figure 6: Number of 1000-byte values read/written per second. The table shows the rate per tablet server; the graph shows the aggregate rate. 
The sequential read benchmark generated row keys in exactly the same way as the sequential write benchmark, but instead of writing under the row key, it read the string stored under the row key (which was written by an earlier invocation of the sequential write benchmark). Similarly, the random read benchmark shadowed the operation of the random write benchmark. 
The scan benchmark is similar to the sequential read benchmark, but uses support provided by the Bigtable API for scanning over all values in a row range. Using a scan reduces the number of RPCs executed by the benchmark since a single RPC fetches a large sequence of values from a tablet server. 
The random reads (mem) benchmark is similar to the random read benchmark, but the locality group that contains the benchmark data is marked as in-memory, and therefore the reads are satisfied from the tablet server’s memory instead of requiring a GFS read. For just this benchmark, we reduced the amount of data per tablet server from 1 GB to $1 0 0 ~ \mathrm { M B }$ so that it would fit comfortably in the memory available to the tablet server. 
Figure 6 shows two views on the performance of our benchmarks when reading and writing 1000-byte values to Bigtable. The table shows the number of operations per second per tablet server; the graph shows the aggregate number of operations per second. 
# Single tablet-server performance 
Let us first consider performance with just one tablet server. Random reads are slower than all other operations by an order of magnitude or more. Each random read involves the transfer of a 64 KB SSTable block over the network from GFS to a tablet server, out of which only a single 1000-byte value is used. The tablet server executes approximately 1200 reads per second, which translates into approximately $7 5 \mathrm { { M B / s } }$ of data read from GFS. This bandwidth is enough to saturate the tablet server CPUs because of overheads in our networking stack, SSTable parsing, and Bigtable code, and is also almost enough to saturate the network links used in our system. Most Bigtable applications with this type of an access pattern reduce the block size to a smaller value, typically 8KB. 
Random reads from memory are much faster since each 1000-byte read is satisfied from the tablet server’s local memory without fetching a large $6 4 \mathrm { K B }$ block from GFS. 
Random and sequential writes perform better than random reads since each tablet server appends all incoming writes to a single commit log and uses group commit to stream these writes efficiently to GFS. There is no significant difference between the performance of random writes and sequential writes; in both cases, all writes to the tablet server are recorded in the same commit log. 
Sequential reads perform better than random reads since every 64 KB SSTable block that is fetched from GFS is stored into our block cache, where it is used to serve the next 64 read requests. 
Scans are even faster since the tablet server can return a large number of values in response to a single client RPC, and therefore RPC overhead is amortized over a large number of values. 
# Scaling 
Aggregate throughput increases dramatically, by over a factor of a hundred, as we increase the number of tablet servers in the system from 1 to 500. For example, the performance of random reads from memory increases by almost a factor of 300 as the number of tablet server increases by a factor of 500. This behavior occurs because the bottleneck on performance for this benchmark is the individual tablet server CPU. 
Table 1: Distribution of number of tablet servers in Bigtable clusters. 
<html><body><table><tr><td># of tablet servers</td><td># of clusters</td></tr><tr><td>0 19</td><td>259</td></tr><tr><td>20 . 49</td><td>47</td></tr><tr><td>50 .. 99</td><td>20</td></tr><tr><td>100 . 499</td><td>50</td></tr><tr><td>> 500</td><td>12</td></tr></table></body></html> 
However, performance does not increase linearly. For most benchmarks, there is a significant drop in per-server throughput when going from 1 to 50 tablet servers. This drop is caused by imbalance in load in multiple server configurations, often due to other processes contending for CPU and network. Our load balancing algorithm attempts to deal with this imbalance, but cannot do a perfect job for two main reasons: rebalancing is throttled to reduce the number of tablet movements (a tablet is unavailable for a short time, typically less than one second, when it is moved), and the load generated by our benchmarks shifts around as the benchmark progresses. 
The random read benchmark shows the worst scaling (an increase in aggregate throughput by only a factor of 100 for a 500-fold increase in number of servers). This behavior occurs because (as explained above) we transfer one large 64KB block over the network for every 1000- byte read. This transfer saturates various shared 1 Gigabit links in our network and as a result, the per-server throughput drops significantly as we increase the number of machines. 
# 8 Real Applications 
As of August 2006, there are 388 non-test Bigtable clusters running in various Google machine clusters, with a combined total of about 24,500 tablet servers. Table 1 shows a rough distribution of tablet servers per cluster. Many of these clusters are used for development purposes and therefore are idle for significant periods. One group of 14 busy clusters with 8069 total tablet servers saw an aggregate volume of more than 1.2 million requests per second, with incoming RPC traffic of about $7 4 1 \mathrm { M B / s }$ and outgoing RPC traffic of about $1 6 \mathrm { G B / s }$ . 
Table 2 provides some data about a few of the tables currently in use. Some tables store data that is served to users, whereas others store data for batch processing; the tables range widely in total size, average cell size, percentage of data served from memory, and complexity of the table schema. In the rest of this section, we briefly describe how three product teams use Bigtable. 
# 8.1 Google Analytics 
Google Analytics (analytics.google.com) is a service that helps webmasters analyze traffic patterns at their web sites. It provides aggregate statistics, such as the number of unique visitors per day and the page views per URL per day, as well as site-tracking reports, such as the percentage of users that made a purchase, given that they earlier viewed a specific page. 
To enable the service, webmasters embed a small JavaScript program in their web pages. This program is invoked whenever a page is visited. It records various information about the request in Google Analytics, such as a user identifier and information about the page being fetched. Google Analytics summarizes this data and makes it available to webmasters. 
We briefly describe two of the tables used by Google Analytics. The raw click table $( \sim 2 0 0 ~ \mathrm { T B } )$ maintains a row for each end-user session. The row name is a tuple containing the website’s name and the time at which the session was created. This schema ensures that sessions that visit the same web site are contiguous, and that they are sorted chronologically. This table compresses to $14 \%$ of its original size. 
The summary table $( \mathrm { \tilde { 2 } 0 \ T B } )$ contains various predefined summaries for each website. This table is generated from the raw click table by periodically scheduled MapReduce jobs. Each MapReduce job extracts recent session data from the raw click table. The overall system’s throughput is limited by the throughput of GFS. This table compresses to $2 9 \%$ of its original size. 
# 8.2 Google Earth 
Google operates a collection of services that provide users with access to high-resolution satellite imagery of the world’s surface, both through the web-based Google Maps interface (maps.google.com) and through the Google Earth (earth.google.com) custom client software. These products allow users to navigate across the world’s surface: they can pan, view, and annotate satellite imagery at many different levels of resolution. This system uses one table to preprocess data, and a different set of tables for serving client data. 
The preprocessing pipeline uses one table to store raw imagery. During preprocessing, the imagery is cleaned and consolidated into final serving data. This table contains approximately 70 terabytes of data and therefore is served from disk. The images are efficiently compressed already, so Bigtable compression is disabled. 
Table 2: Characteristics of a few tables in production use. Table size (measured before compression) and $\#$ Cells indicate approximate sizes. Compression ratio is not given for tables that have compression disabled. 
<html><body><table><tr><td>Project name</td><td>Table size (TB)</td><td>Compression ratio</td><td># Cells (billions)</td><td># Column Families</td><td># Locality Groups</td><td>% in memory</td><td>Latency- sensitive?</td></tr><tr><td>Crawl</td><td>800</td><td>11%</td><td>1000</td><td>16</td><td>8</td><td>0%</td><td>No</td></tr><tr><td>Crawl</td><td> 50</td><td>33%</td><td>200</td><td>2</td><td>2</td><td>0%</td><td>No</td></tr><tr><td>Google Analytics</td><td>20</td><td>29%</td><td>10</td><td>1</td><td>1</td><td>0%</td><td>Yes</td></tr><tr><td>Google Analytics</td><td>200</td><td>14%</td><td>80</td><td>1</td><td>1</td><td>0%</td><td>Yes</td></tr><tr><td>Google Base</td><td>2</td><td>31%</td><td>10</td><td>29</td><td>3</td><td>15%</td><td>Yes</td></tr><tr><td>Google Earth</td><td>0.5</td><td>64%</td><td>8</td><td>7</td><td>2</td><td>33%</td><td>Yes</td></tr><tr><td>Google Earthn</td><td>70</td><td>-</td><td>9</td><td>8</td><td>3</td><td>0%</td><td>No</td></tr><tr><td>Orkut</td><td>9</td><td>-</td><td>0.9</td><td>8</td><td>5</td><td>1%</td><td>Yes</td></tr><tr><td>Personalized Search</td><td>4</td><td>47%</td><td>6</td><td>93</td><td>11</td><td>5%</td><td>Yes</td></tr></table></body></html> 
Each row in the imagery table corresponds to a single geographic segment. Rows are named to ensure that adjacent geographic segments are stored near each other. The table contains a column family to keep track of the sources of data for each segment. This column family has a large number of columns: essentially one for each raw data image. Since each segment is only built from a few images, this column family is very sparse. 
The preprocessing pipeline relies heavily on MapReduce over Bigtable to transform data. The overall system processes over 1 MB/sec of data per tablet server during some of these MapReduce jobs. 
The serving system uses one table to index data stored in GFS. This table is relatively small $( { \bf \tilde { \omega } } 5 0 0 ~ \mathrm { G B } )$ , but it must serve tens of thousands of queries per second per datacenter with low latency. As a result, this table is hosted across hundreds of tablet servers and contains inmemory column families. 
# 8.3 Personalized Search 
Personalized Search (www.google.com/psearch) is an opt-in service that records user queries and clicks across a variety of Google properties such as web search, images, and news. Users can browse their search histories to revisit their old queries and clicks, and they can ask for personalized search results based on their historical Google usage patterns. 
Personalized Search stores each user’s data in Bigtable. Each user has a unique userid and is assigned a row named by that userid. All user actions are stored in a table. A separate column family is reserved for each type of action (for example, there is a column family that stores all web queries). Each data element uses as its Bigtable timestamp the time at which the corresponding user action occurred. Personalized Search generates user profiles using a MapReduce over Bigtable. These user profiles are used to personalize live search results. 
The Personalized Search data is replicated across several Bigtable clusters to increase availability and to reduce latency due to distance from clients. The Personalized Search team originally built a client-side replication mechanism on top of Bigtable that ensured eventual consistency of all replicas. The current system now uses a replication subsystem that is built into the servers. 
The design of the Personalized Search storage system allows other groups to add new per-user information in their own columns, and the system is now used by many other Google properties that need to store per-user configuration options and settings. Sharing a table amongst many groups resulted in an unusually large number of column families. To help support sharing, we added a simple quota mechanism to Bigtable to limit the storage consumption by any particular client in shared tables; this mechanism provides some isolation between the various product groups using this system for per-user information storage. 
# 9 Lessons 
In the process of designing, implementing, maintaining, and supporting Bigtable, we gained useful experience and learned several interesting lessons. 
One lesson we learned is that large distributed systems are vulnerable to many types of failures, not just the standard network partitions and fail-stop failures assumed in many distributed protocols. For example, we have seen problems due to all of the following causes: memory and network corruption, large clock skew, hung machines, extended and asymmetric network partitions, bugs in other systems that we are using (Chubby for example), overflow of GFS quotas, and planned and unplanned hardware maintenance. As we have gained more experience with these problems, we have addressed them by changing various protocols. For example, we added checksumming to our RPC mechanism. We also handled some problems by removing assumptions made by one part of the system about another part. For example, we stopped assuming a given Chubby operation could return only one of a fixed set of errors. 
Another lesson we learned is that it is important to delay adding new features until it is clear how the new features will be used. For example, we initially planned to support general-purpose transactions in our API. Because we did not have an immediate use for them, however, we did not implement them. Now that we have many real applications running on Bigtable, we have been able to examine their actual needs, and have discovered that most applications require only single-row transactions. Where people have requested distributed transactions, the most important use is for maintaining secondary indices, and we plan to add a specialized mechanism to satisfy this need. The new mechanism will be less general than distributed transactions, but will be more efficient (especially for updates that span hundreds of rows or more) and will also interact better with our scheme for optimistic cross-data-center replication. 
A practical lesson that we learned from supporting Bigtable is the importance of proper system-level monitoring (i.e., monitoring both Bigtable itself, as well as the client processes using Bigtable). For example, we extended our RPC system so that for a sample of the RPCs, it keeps a detailed trace of the important actions done on behalf of that RPC. This feature has allowed us to detect and fix many problems such as lock contention on tablet data structures, slow writes to GFS while committing Bigtable mutations, and stuck accesses to the METADATA table when METADATA tablets are unavailable. Another example of useful monitoring is that every Bigtable cluster is registered in Chubby. This allows us to track down all clusters, discover how big they are, see which versions of our software they are running, how much traffic they are receiving, and whether or not there are any problems such as unexpectedly large latencies. 
The most important lesson we learned is the value of simple designs. Given both the size of our system (about 100,000 lines of non-test code), as well as the fact that code evolves over time in unexpected ways, we have found that code and design clarity are of immense help in code maintenance and debugging. One example of this is our tablet-server membership protocol. Our first protocol was simple: the master periodically issued leases to tablet servers, and tablet servers killed themselves if their lease expired. Unfortunately, this protocol reduced availability significantly in the presence of network problems, and was also sensitive to master recovery time. We redesigned the protocol several times until we had a protocol that performed well. However, the resulting protocol was too complex and depended on the behavior of Chubby features that were seldom exercised by other applications. We discovered that we were spending an inordinate amount of time debugging obscure corner cases, not only in Bigtable code, but also in Chubby code. Eventually, we scrapped this protocol and moved to a newer simpler protocol that depends solely on widely-used Chubby features. 
# 10 Related Work 
The Boxwood project [24] has components that overlap in some ways with Chubby, GFS, and Bigtable, since it provides for distributed agreement, locking, distributed chunk storage, and distributed B-tree storage. In each case where there is overlap, it appears that the Boxwood’s component is targeted at a somewhat lower level than the corresponding Google service. The Boxwood project’s goal is to provide infrastructure for building higher-level services such as file systems or databases, while the goal of Bigtable is to directly support client applications that wish to store data. 
Many recent projects have tackled the problem of providing distributed storage or higher-level services over wide area networks, often at “Internet scale.” This includes work on distributed hash tables that began with projects such as CAN [29], Chord [32], Tapestry [37], and Pastry [30]. These systems address concerns that do not arise for Bigtable, such as highly variable bandwidth, untrusted participants, or frequent reconfiguration; decentralized control and Byzantine fault tolerance are not Bigtable goals. 
In terms of the distributed data storage model that one might provide to application developers, we believe the key-value pair model provided by distributed B-trees or distributed hash tables is too limiting. Key-value pairs are a useful building block, but they should not be the only building block one provides to developers. The model we chose is richer than simple key-value pairs, and supports sparse semi-structured data. Nonetheless, it is still simple enough that it lends itself to a very efficient flat-file representation, and it is transparent enough (via locality groups) to allow our users to tune important behaviors of the system. 
Several database vendors have developed parallel databases that can store large volumes of data. Oracle’s Real Application Cluster database [27] uses shared disks to store data (Bigtable uses GFS) and a distributed lock manager (Bigtable uses Chubby). IBM’s DB2 Parallel Edition [4] is based on a shared-nothing [33] architecture similar to Bigtable. Each DB2 server is responsible for a subset of the rows in a table which it stores in a local relational database. Both products provide a complete relational model with transactions. 
Bigtable locality groups realize similar compression and disk read performance benefits observed for other systems that organize data on disk using column-based rather than row-based storage, including C-Store [1, 34] and commercial products such as Sybase IQ [15, 36], SenSage [31], $\mathrm { K D B + }$ [22], and the ColumnBM storage layer in MonetDB/X100 [38]. Another system that does vertical and horizontal data partioning into flat files and achieves good data compression ratios is AT&T’s Daytona database [19]. Locality groups do not support CPUcache-level optimizations, such as those described by Ailamaki [2]. 
The manner in which Bigtable uses memtables and SSTables to store updates to tablets is analogous to the way that the Log-Structured Merge Tree [26] stores updates to index data. In both systems, sorted data is buffered in memory before being written to disk, and reads must merge data from memory and disk. 
C-Store and Bigtable share many characteristics: both systems use a shared-nothing architecture and have two different data structures, one for recent writes, and one for storing long-lived data, with a mechanism for moving data from one form to the other. The systems differ significantly in their API: C-Store behaves like a relational database, whereas Bigtable provides a lower level read and write interface and is designed to support many thousands of such operations per second per server. C-Store is also a “read-optimized relational DBMS”, whereas Bigtable provides good performance on both read-intensive and write-intensive applications. 
Bigtable’s load balancer has to solve some of the same kinds of load and memory balancing problems faced by shared-nothing databases (e.g., [11, 35]). Our problem is somewhat simpler: (1) we do not consider the possibility of multiple copies of the same data, possibly in alternate forms due to views or indices; (2) we let the user tell us what data belongs in memory and what data should stay on disk, rather than trying to determine this dynamically; (3) we have no complex queries to execute or optimize. 
Given the unusual interface to Bigtable, an interesting question is how difficult it has been for our users to adapt to using it. New users are sometimes uncertain of how to best use the Bigtable interface, particularly if they are accustomed to using relational databases that support general-purpose transactions. Nevertheless, the fact that many Google products successfully use Bigtable demonstrates that our design works well in practice. 
We are in the process of implementing several additional Bigtable features, such as support for secondary indices and infrastructure for building cross-data-center replicated Bigtables with multiple master replicas. We have also begun deploying Bigtable as a service to product groups, so that individual groups do not need to maintain their own clusters. As our service clusters scale, we will need to deal with more resource-sharing issues within Bigtable itself [3, 5]. 
Finally, we have found that there are significant advantages to building our own storage solution at Google. We have gotten a substantial amount of flexibility from designing our own data model for Bigtable. In addition, our control over Bigtable’s implementation, and the other Google infrastructure upon which Bigtable depends, means that we can remove bottlenecks and inefficiencies as they arise. 

# 11 Conclusions 
We have described Bigtable, a distributed system for storing structured data at Google. Bigtable clusters have been in production use since April 2005, and we spent roughly seven person-years on design and implementation before that date. As of August 2006, more than sixty projects are using Bigtable. Our users like the performance and high availability provided by the Bigtable implementation, and that they can scale the capacity of their clusters by simply adding more machines to the system as their resource demands change over time. 
