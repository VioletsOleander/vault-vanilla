# Abstract 
MapReduce is a programming model and an associated implementation for processing and generating large data sets. Users specify a map function that processes a key/value pair to generate a set of intermediate key/value pairs, and a reduce function that merges all intermediate values associated with the same intermediate key. Many real world tasks are expressible in this model, as shown in the paper. 
>  MapReduce 是一个编程模型和一个用于处理和生成大数据集的相关实现
>  MapReduce 中，用户指定一个映射函数，该函数处理一个键值对以生成一组中间键值对，用户还指定一个归约函数，该函数会归并所有与同一中间键关联的值
>  如本文所示，许多真实世界任务可以用该模型表示

Programs written in this functional style are automatically parallelized and executed on a large cluster of commodity machines. The run-time system takes care of the details of partitioning the input data, scheduling the program’s execution across a set of machines, handling machine failures, and managing the required inter-machine communication. This allows programmers without any experience with parallel and distributed systems to easily utilize the resources of a large distributed system. 
>  按照这种函数式风格写的程序会自动在大的商品机集群上并行并执行
>  运行时系统负责处理输入数据的划分，在一组机器中调度程序的执行，处理机器故障，以及管理所需的机器间通信
>  这使得没有任何并行和分布式系统经验的程序员都可以利用大型分布式系统的资源

Our implementation of MapReduce runs on a large cluster of commodity machines and is highly scalable: a typical MapReduce computation processes many terabytes of data on thousands of machines. Programmers find the system easy to use: hundreds of MapReduce programs have been implemented and upwards of one thousand MapReduce jobs are executed on Google’s clusters every day. 
>  MapReduce 的实现运行在大的商品机集群上，且高度可拓展：一个典型的 MapReduce 计算在数千个机器上处理数 TB (1TB = 1024GB) 的数据
>  Google 已经实现了数百个 MapReduce 程序，同时 Google 的集群上每天执行超过一千个 MapReduce 作业

# 1 Introduction 
Over the past five years, the authors and many others at Google have implemented hundreds of special-purpose computations that process large amounts of raw data, such as crawled documents, web request logs, etc., to compute various kinds of derived data, such as inverted indices, various representations of the graph structure of web documents, summaries of the number of pages crawled per host, the set of most frequent queries in a given day, etc. Most such computations are conceptually straightforward. However, the input data is usually large and the computations have to be distributed across hundreds or thousands of machines in order to finish in a reasonable amount of time. The issues of how to parallelize the computation, distribute the data, and handle failures conspire to obscure the original simple computation with large amounts of complex code to deal with these issues. 
>  过去五年，作者和 Google 的许多其他员工实现了数百个专用目的的计算，用于处理大规模的原始数据，例如爬取的文档、网页请求日志等，并且用于计算各种衍生数据，例如倒排索引、网页文档的图结构的各种表示、每个主机爬取的页面数量的摘要、给定日期中最频繁查询的集合，等
>  大多数这类计算在概念上直接，但输入数据规模过大，计算需要在数百到数千个机器上分布，关于如何并行化计算、分布数据、处理故障的代码复杂化了原本简单的代码

As a reaction to this complexity, we designed a new abstraction that allows us to express the simple computations we were trying to perform but hides the messy details of parallelization, fault-tolerance, data distribution and load balancing in a library. Our abstraction is inspired by the map and reduce primitives present in Lisp and many other functional languages. We realized that most of our computations involved applying a map operation to each logical “record” in our input in order to compute a set of intermediate key/value pairs, and then applying a reduce operation to all the values that shared the same key, in order to combine the derived data appropriately.
>  为了解决该复杂化问题，我们设计了一层抽象，它将并行化、容错、数据分布和负载均衡隐藏在库中，在该抽象上，我们只需要表达简单的计算
>  这一抽象灵感来源于 Lisp 语言和许多其他函数式语言中的 map 和 reduce 原语，我们意识到我们的大多数计算涉及到对输入中的每个逻辑“记录”应用 map 操作以计算一组中间键值对，然后对共享相同键的所有值应用一个 reduce 操作，以适当地组合派生数据

 Our use of a functional model with user-specified map and reduce operations allows us to parallelize large computations easily and to use re-execution as the primary mechanism for fault tolerance. 
>  用户通过指定 map 和 reduce 操作，基于该模型，可以轻松并行大规模计算，并且将重执行作为主要的容错机制

The major contributions of this work are a simple and powerful interface that enables automatic parallelization and distribution of large-scale computations, combined with an implementation of this interface that achieves high performance on large clusters of commodity PCs. 
>  该工作的主要贡献是一个自动并行化和分布大规模计算的接口，以及该接口的一个实际实现，该实现可以在大规模商用 PC 的集群上取得高性能

Section 2 describes the basic programming model and gives several examples. Section 3 describes an implementation of the MapReduce interface tailored towards our cluster-based computing environment. Section 4 describes several refinements of the programming model that we have found useful. Section 5 has performance measurements of our implementation for a variety of tasks. Section 6 explores the use of MapReduce within Google including our experiences in using it as the basis for a rewrite of our production indexing system. Section 7 discusses related and future work. 

# 2 Programming Model 
The computation takes a set of input key/value pairs, and produces a set of output key/value pairs. The user of the MapReduce library expresses the computation as two functions: Map and Reduce. 
>  MapReduce 计算接受一组输入键值对，然后产生一组输出键值对
>  MapReduce 库的用户通过两个函数：Map 和 Reduce 来表达计算

Map, written by the user, takes an input pair and produces a set of intermediate key/value pairs. The MapReduce library groups together all intermediate values associated with the same intermediate key $I$ and passes them to the Reduce function. 
>  Map 函数由用户编写，它接受一个输入对，产生一组中间键值对
>  MapReduce 库将所有具有相同中间键 $I$ 的中间键值对分为同一组，然后将它们传递给 Reduce 函数

The Reduce function, also written by the user, accepts an intermediate key $I$ and a set of values for that key. It merges together these values to form a possibly smaller set of values. Typically just zero or one output value is produced per Reduce invocation. The intermediate values are supplied to the user’s reduce function via an iterator. This allows us to handle lists of values that are too large to fit in memory. 
>  Reduce 函数也由用户编写，它接受一个中间键 $I$ 和该键的一组值，将这些值归并，形成一个可能更小的值集合，一般情况下，每次 Reduce 调用只产生零个或者一个输出值
>  中间值是通过一个迭代器传递给 Reduce 函数，便于处理难以放入内存中的过长的值列表

### 2.1 Example 
Consider the problem of counting the number of occurrences of each word in a large collection of documents. The user would write code similar to the following pseudo-code: 
>  考虑计算一个大文档集合中每个单词的出现次数，用户代码将类似以下形式

```
map(String key, String value):
    // key: document name
    // value: document contents
    for each word w in values:
        EmitIntermediate(w, "1")

reduce(String key, Iterator values):
    // key: a word
    // values: a list of counts
    int results = 0
    for each v in values:
        result += ParseInt(v)
    Emit(AsString(result));
```

The map function emits each word plus an associated count of occurrences (just ‘1’ in this simple example). The reduce function sums together all counts emitted for a particular word. 
>  `map` 函数发出每个单词和其出现次数 (本例中就是 1)，`reduce` 函数求和特定单词发出的所有计数

In addition, the user writes code to fill in a mapreduce specification object with the names of the input and output files, and optional tuning parameters. The user then invokes the MapReduce function, passing it the specification object. The user’s code is linked together with the MapReduce library (implemented in $\mathrm{C}{+}+$ ). Appendix A contains the full program text for this example. 
>  此外，用户需要编写代码，填入一个包含了输入和输出文件名、可选调优参数的 mapreduce 规范对象
>  完成这些后，用户调用 MapReduce 函数，并将规范对象传递给该函数，用户的代码将会和 MapReduce 库 (C++实现) 链接在一起

### 2.2 Types 
Even though the previous pseudo-code is written in terms of string inputs and outputs, conceptually the map and reduce functions supplied by the user have associated types: 
>  概念上，用户提供的 map 和 reduce 函数的输入输出类型形式如下

```
map (k1, v1) -> list(k2, v2)
reduce (k2, list(v2)) -> list(v2)
```

I.e., the input keys and values are drawn from a different domain than the output keys and values. Furthermore, the intermediate keys and values are from the same domain as the output keys and values. 
>  可以看到，输入的键和值和中间键值可以来自于不同的域 (类型不同)，输出的值则和中间的值来自同一域 (类型相同)

Our $\mathrm{C}{+}{+}$ implementation passes strings to and from the user-defined functions and leaves it to the user code to convert between strings and appropriate types. 
>  我们的 C++ 实现期待从用户定义的函数 `map` 中接受字符串类型，同时也会向用户定义的函数 `reduce` 中传入字符串类型，字符串类型和其他特定类型的转换需要由用户代码实现

### 2.3 More Examples 
Here are a few simple examples of interesting programs that can be easily expressed as MapReduce computations. 
>  这里提供几个可以由 MapReduce 计算轻松完成的程序

**Distributed Grep:** The map function emits a line if it matches a supplied pattern. The reduce function is an identity function that just copies the supplied intermediate data to the output. 
>  分布式 grep
>  `map` 函数发出匹配给定模式的行，`reduce` 函数为恒等函数，将给定的输入数据复制到输出

**Count of URL Access Frequency:** The map function processes logs of web page requests and outputs $\langle{\mathrm{URL}},1\rangle$ . The reduce function adds together all values for the same URL and emits a URL, total count pair. 
>  计数 URL 访问频率
>  `map` 函数处理网页请求的日志，输出 `<URL, 1>` ，`reduce` 函数将相同 URL 的值相加，发出 URL 和其总计数

**Reverse Web-Link Graph:** The map function outputs ⟨target, source⟩ pairs for each link to a target URL found in a page named source. The reduce function concatenates the list of all source URLs associated with a given target URL and emits the pair: ⟨target, list (source)⟩ 
>  反转网页-链接图
>  `map` 函数对于在 source 页面找到的每个指向 target URL 的链接输出 `<target, source>` 对，`reduce` 函数拼接给定 URL 的所有关联的 source，发出 `<target, list(source)>` 对

**Term-Vector per Host:** A term vector summarizes the most important words that occur in a document or a set of documents as a list of ⟨word, frequency⟩ pairs. The map function emits a hostname, term vector pair for each input document (where the hostname is extracted from the URL of the document). The reduce function is passed all per-document term vectors for a given host. It adds these term vectors together, throwing away infrequent terms, and then emits a final ⟨hostname, term vector⟩ pair. 
>  每个主机的术语向量
>  一个术语向量以一个 `<word, frequency>` 列表的形式总结了一篇文档或者一个文档列表中出现的最重要的词
>  `map` 函数对于每个输入文档发出一个 `<hostname, term vector>` 对 (主机名从文档的 URL 提取)，`reduce` 函数接受给定主机的所有文档术语向量，将这些术语向量相加，丢弃不频繁出现的词条，发出最终的 `<hostname, term vector>` 对

**Inverted Index:** The map function parses each document, and emits a sequence of ⟨word, document $\mathtt{I D}\rangle$ pairs. The reduce function accepts all pairs for a given word, sorts the corresponding document IDs and emits a $\langle\mathrm{word},l i s t(\mathrm{document~ID})\rangle$ pair. The set of all output pairs forms a simple inverted index. It is easy to augment this computation to keep track of word positions. 
>  倒排索引
>  `map` 函数解析每篇文档，发出一个 `<word, document ID>` 对，`reduce` 函数接受给定单词的所有 `<word, document ID>` 对，将对应文档 ID 排序，然后发出一个 `<word, list<document ID>>` 对
>  所有输出对的集合就形成了一个倒排索引，该计算也很容易拓展，以进一步跟踪单词的位置

**Distributed Sort:** The map function extracts the key from each record, and emits a ⟨key, record⟩ pair. The reduce function emits all pairs unchanged. This computation depends on the partitioning facilities described in Section 4.1 and the ordering properties described in Section 4.2. 
>  分布式排序
>  `map` 函数从每个记录中提取出键，发出 `<key, record>` 对，`reduce` 函数原样发出所有的对，该计算依赖于 partitioning facilities 和 ordering properties

# 3 Implementation 
Many different implementations of the MapReduce interface are possible. The right choice depends on the environment. For example, one implementation may be suitable for a small shared-memory machine, another for a large NUMA multi-processor, and yet another for an even larger collection of networked machines. 
>  MapReduce 接口可以有许多可行的实现，具体选择依赖于使用环境
>  例如，一种选择可能适合于小型共享内存机器、一种选择适合于大型的非统一内存访问 (NUMA) 多处理器系统、一种选择适合于更大规模的网络机器集合

This section describes an implementation targeted to the computing environment in wide use at Google: large clusters of commodity PCs connected together with switched Ethernet [4]. 
>  本节描述针对 Google 使用的计算环境的 MapReduce 实现，即针对由以太网连接的大规模商用 PC 集群

In our environment: 

(1) Machines are typically dual-processor $\mathrm{x}86$ processors running Linux, with 2-4 GB of memory per machine. 
(2) Commodity networking hardware is used – typically either 100 megabits/second or 1 gigabit/second at the machine level, but averaging considerably less in overall bisection bandwidth. 
(3) A cluster consists of hundreds or thousands of machines, and therefore machine failures are common. 
(4) Storage is provided by inexpensive IDE disks attached directly to individual machines. A distributed file system [8] developed in-house is used to manage the data stored on these disks. The file system uses replication to provide availability and reliability on top of unreliable hardware. 
(5) Users submit jobs to a scheduling system. Each job consists of a set of tasks, and is mapped by the scheduler to a set of available machines within a cluster. 

>  Google 的环境中：
>  1. 机器一般是双核 x86 处理器，运行 Linux 操作系统，每台机器内存在 2-4 GB
>  2. 使用普通网络硬件，在单机上速度通常在 100MB/s 或 1GB/s，但整个网络的有效二分带宽平均来看要低得多
>  3. 一个集群包含数百个到上千个机器，机器故障很常见
>  4. 存储由直接连接到各机器的廉价 IDE 磁盘提供，Google 使用了自主研发的分布式文件系统管理这些磁盘上的数据，该文件系统使用复制以基于不可靠的硬件上提供可用性和可靠性
>  5. 用户向调度系统提交任务，每个任务包含一组工作，由调度器映射到集群内一组可用的机器上

> [! info] 二分带宽
> 二分带宽 (bisection bandwidth) 是指一个网络/图被**任意**分成两等份时，两部分之间可能的最小带宽。也就是说，将网络划分成使得两个分区之间的带宽达到了最小，该带宽就是网络的二分带宽
> 
> 给定图 $G$，节点集合 $V$，边集合 $E$，边权重函数 $w$，$G$ 的二分带宽为
> 
> $$\text{bisection bandwidth}(G) = \min_{S\subset V:|S| = \frac 1 2|V|}\sum_{u\in S,v\not\in S}w(u,v)$$
> 
> 如果一个网络 $G$ 满足 $\text{bisection bandwidth}(G)\ge\frac 1 2|V|$，称该网络具有全二分带宽，直观上，具有全二分带宽的网络意味着将其中所有节点匹配为源-目的地对，这些对同时以速率 1 发送数据时，不会有二分瓶颈
> 因此，二分带宽实际上衡量了网络被二分时的带宽瓶颈情况

## 3.1 Execution Overview 
The Map invocations are distributed across multiple machines by automatically partitioning the input data into a set of $M$ splits. The input splits can be processed in parallel by different machines. Reduce invocations are distributed by partitioning the intermediate key space into $R$ pieces using a partitioning function (e.g., $h a s h(k e y)\bmod R)$ ). The number of partitions $(R)$ and the partitioning function are specified by the user. 
>  Map 执行时，数据会自动被划分为 $M$ 个切片，在多台机器上分布式调用 `map` ，因此，输入切片会被多台机器并行处理
>  Reduce 执行时，中间的键会通过一个划分函数，例如 `hash(key) mod R` ，被划分为 `R` 个部分，然后分布式调用 `reduce` ，分区数量 R 和具体的划分函数由用户指定

![[pics/MapReduce-Fig1.png]]

Figure 1 shows the overall flow of a MapReduce operation in our implementation. When the user program calls the MapReduce function, the following sequence of actions occurs (the numbered labels in Figure 1 correspond to the numbers in the list below): 
>  用户调用 `MapReduce` 函数时，以下操作按序执行

1. The MapReduce library in the user program first splits the input files into $M$ pieces of typically 16 megabytes to 64 megabytes (MB) per piece (controllable by the user via an optional parameter). It then starts up many copies of the program on a cluster of machines. 
2. One of the copies of the program is special – the master. The rest are workers that are assigned work by the master. There are $M$ map tasks and $R$ reduce tasks to assign. The master picks idle workers and assigns each one a map task or a reduce task. 

>  (1) 用户程序的 MapReduce 库将输入划分为 M 个片段，每个片段一般 16MB 到 64 MB (用户可通过参数控制)，然后在集群上启动多个程序副本
>  (2) 一个程序副本会作为主控程序，剩余为工作程序，其工作由 master 分配。一共有 M 个 map 任务和 R 个 reduce 任务需要分配，master 会空闲的工作者分配一个 map 任务或 reduce 任务

3. A worker who is assigned a map task reads the contents of the corresponding input split. It parses key/value pairs out of the input data and passes each pair to the user-defined Map function. The intermediate key/value pairs produced by the Map function are buffered in memory. 
4. Periodically, the buffered pairs are written to local disk, partitioned into $R$ regions by the partitioning function. The locations of these buffered pairs on the local disk are passed back to the master, who is responsible for forwarding these locations to the reduce workers. 

>  (3) 得到 map 任务的 worker 读取对应的输入片段，解析键值对，将每个键值对传递给用户定义的 `map` 函数，`map` 函数生成中间键值对，缓存在内存中
>  (4) 这些缓存的中间键值对会周期性写入本地磁盘，写入时会存入被划分函数划分为 R 个区域中的其中一个，这些缓存的键值对在本地磁盘上的地址会被传递给 master, master 之后负责将这些地址转发给执行 reduce 的 worker

5. When a reduce worker is notified by the master about these locations, it uses remote procedure calls to read the buffered data from the local disks of the map workers. When a reduce worker has read all intermediate data, it sorts it by the intermediate keys so that all occurrences of the same key are grouped together. The sorting is needed because typically many different keys map to the same reduce task. If the amount of intermediate data is too large to fit in memory, an external sort is used. 
6. The reduce worker iterates over the sorted intermediate data and for each unique intermediate key encountered, it passes the key and the corresponding set of intermediate values to the user’s Reduce function. The output of the Reduce function is appended to a final output file for this reduce partition. 

> (5) reduce worker 被 master 告知这些地址时，它使用远程过程调用从 map worker 中的磁盘中读取对应的数据
> 读取完全部的数据后，它按照中间键对它们进行排序，使得相同键的键值对被分组在一起，排序是必要的，因为通常一个 reduce 任务会涉及多个不同的键。如果中间数据太大而无法放入内存，还需要使用外部排序
> (6) reduce worker 迭代排序好的中间数据，对于每个唯一的中间键，将该键和对应的一组值传递给用户的 `reduce` 函数，`reduce` 函数的输出将附加到该 reduce 分区中的最终输出文件

7. When all map tasks and reduce tasks have been completed, the master wakes up the user program. At this point, the MapReduce call in the user program returns back to the user code. 

>  (7) 所有 map 任务和 reduce 任务完成后，master 唤醒用户程序，用户程序中的 `MapReduce` 调用返回

After successful completion, the output of the `mapreduce` execution is available in the $R$ output files (one per reduce task, with file names as specified by the user). Typically, users do not need to combine these $R$ output files into one file – they often pass these files as input to another MapReduce call, or use them from another distributed application that is able to deal with input that is partitioned into multiple files. 
>  成功执行后，`mapreduce` 的输出可以在 R 个输出文件中找到，每个输出文件对应一个 reduce 任务
>  一般用户不需要将这 R 个输出文件结合为 1 个输出文件，而是可以将这些文件传递给另一个 `MapReduce` 调用，或者用于其他分布式程序的输入

## 3.2 Master Data Structures 
The master keeps several data structures. For each map task and reduce task, it stores the state (idle, in-progress, or completed), and the identity of the worker machine (for non-idle tasks). 
>  master 中，对于每个 map 任务和 reduce 任务，它存储该任务的状态 (idle, in-progress, completed) 和执行该任务的 worker 机器的 ID

The master is the conduit through which the location of intermediate file regions is propagated from map tasks to reduce tasks. Therefore, for each completed map task, the master stores the locations and sizes of the $R$ intermediate file regions produced by the map task. Updates to this location and size information are received as map tasks are completed. The information is pushed incrementally to workers that have in-progress reduce tasks. 
>  master 还负责了传递 map 任务产生的中间文件区域的位置给 reduce 任务，对于每个完成的 map 任务，master 存储 map 任务生成的 R 个中间文件区域的位置和大小，随着更多 map 任务的完成，master 接受对应的位置信息和大小信息然后进行相应的更新
>  这些信息会被逐步推送给正在执行 reduce 任务的 worker

## 3.3 Fault Tolerance 
Since the MapReduce library is designed to help process very large amounts of data using hundreds or thousands of machines, the library must tolerate machine failures gracefully. 

### Worker Failure 
The master pings every worker periodically. If no response is received from a worker in a certain amount of time, the master marks the worker as failed. Any map tasks completed by the worker are reset back to their initial idle state, and therefore become eligible for scheduling on other workers. Similarly, any map task or reduce task in progress on a failed worker is also reset to idle and becomes eligible for rescheduling. 
>  master 会周期性 ping 每个 worker，如果一段时间没收到恢复，master 标记该 worker 故障
>  任意由该 worker 完成的 map 任务会重置回其初始的 idle 状态，进而能调度给其他 worker，类似地，任意执行中的 map 任务或 reduce 任务也会重置会 idle 状态，以重新调度

Completed map tasks are re-executed on a failure because their output is stored on the local disk (s) of the failed machine and is therefore inaccessible. Completed reduce tasks do not need to be re-executed since their output is stored in a global file system. 
>  完成的 map 任务需要重新执行的原因在于其输出存储在故障机器的本地磁盘上，因此不能再访问
>  完成的 reduce 任务不需要再执行，因为其输出已经存回了全局文件系统

When a map task is executed first by worker $A$ and then later executed by worker $B$ (because $A$ failed), all workers executing reduce tasks are notified of the re-execution. Any reduce task that has not already read the data from worker $A$ will read the data from worker $B$ . 
>  当一个 map 任务先由 worker A 执行，然后由 worker B 执行 (A 故障)，所有执行 reduce 任务的 worker 会被告知这一点，所有还没有从 A 中读取数据的 reduce worker 会从 B 中读取数据

MapReduce is resilient to large-scale worker failures. For example, during one MapReduce operation, network maintenance on a running cluster was causing groups of 80 machines at a time to become unreachable for several minutes. The MapReduce master simply re-executed the work done by the unreachable worker machines, and continued to make forward progress, eventually completing the MapReduce operation. 
>  MapReduce 能够承受大规模 worker 故障
>  例如，一次 MapReduce 操作中，一个运行中集群上的网络维护导致一组 80 个机器同时在数分钟无法通过网络连接，MapRecude master 则重新调度由这些无可访问的机器已经执行的工作，然后继续按照往常一样执行

### Master Failure 
It is easy to make the master write periodic checkpoints of the master data structures described above. If the master task dies, a new copy can be started from the last checkpointed state. However, given that there is only a single master, its failure is unlikely; therefore our current implementation aborts the MapReduce computation if the master fails. Clients can check for this condition and retry the MapReduce operation if they desire. 
>  master 会周期性将 master 中的数据结构写为 checkpoints，如果 master 任务故障，可以从最新的 checkpoint 启动一个新副本
>  因为 master 只有一个，其故障的可能性不高，因此当前的 MapReduce 实现在 master 故障时直接终止计算，客户端需要检查这一情况，然后重启 MapReduce 任务 (从最新的 checkpoint)

### Semantics in the Presence of Failures 
When the user-supplied map and reduce operators are deterministic functions of their input values, our distributed implementation produces the same output as would have been produced by a non-faulting sequential execution of the entire program. 
>  如果用户定义的 `map, reduce` 函数相对于其输入数据的运算都是确定性的，则 MapReduce 的分布式实现将和没有故障的顺序执行的输出相同

We rely on atomic commits of map and reduce task outputs to achieve this property. Each in-progress task writes its output to private temporary files. A reduce task produces one such file, and a map task produces $R$ such files (one per reduce task). When a map task completes, the worker sends a message to the master and includes the names of the $R$ temporary files in the message. If the master receives a completion message for an already completed map task, it ignores the message. Otherwise, it records the names of $R$ files in a master data structure. 
>  这一性质的实现依赖于 map 和 reduce 任务输出的原子提交
>  每个执行中的任务都会将其输出写入一个私有的临时文件，一个 reducer 任务生成一个这样的文件，一个 map 任务生成 R 个这样的文件 (对应 R 个 reduce 任务)。当一个 map 任务完成后，worker 向 master 发送消息，消息包含了这 R 个临时文件的名称，如果 master 收到的是完成消息，则忽略这条消息，否则在 master 的数据结构中记录这 R 个文件

When a reduce task completes, the reduce worker atomically renames its temporary output file to the final output file. If the same reduce task is executed on multiple machines, multiple rename calls will be executed for the same final output file. We rely on the atomic rename operation provided by the underlying file system to guarantee that the final file system state contains just the data produced by one execution of the reduce task. 
>  当一个 reduce 任务完成后，reduce worker 自动将其临时输出文件重命名为最终输出文件
>  如果相同的 reduce 任务在多个机器执行，则相同的最终输出文件会被多次重命名，依赖于文件系统提供的原子重命名操作，可以保证最终的文件系统状态仅包含 reduce 任务的一次执行生成的数据

The vast majority of our map and reduce operators are deterministic, and the fact that our semantics are equivalent to a sequential execution in this case makes it very easy for programmers to reason about their program’s behavior. 

When the map and/or reduce operators are nondeterministic, we provide weaker but still reasonable semantics. In the presence of non-deterministic operators, the output of a particular reduce task $R_{1}$ is equivalent to the output for $R_{1}$ produced by a sequential execution of the non-deterministic program. However, the output for a different reduce task $R_{2}$ may correspond to the output for $R_{2}$ produced by a different sequential execution of the non-deterministic program. 
>  如果 `map` 或 `reduce` 操作不是确定性的，我们仍提供更弱但合理的语义
>  对于非确定性的计算，特定 reduce 任务 $R_1$ 的输出等价于非确定性程序的某次顺序执行产生的输出中 $R_1$ 的部分，但另一个 reduce 任务 $R_2$ 的输出可能不对应于这次顺序执行产生的输出中的 $R_2$ 的部分，而可能对应于另一次顺序执行产生的输出中的 $R_2$ 的部分

Consider map task $M$ and reduce tasks $R_{1}$ and $R_{2}$ . Let $e(R_{i})$ be the execution of $R_{i}$ that committed (there is exactly one such execution). The weaker semantics arise because $e(R_{1})$ may have read the output produced by one execution of $M$ and $e(R_{2})$ may have read the output produced by a different execution of $M$ . 
>  考虑 map 任务 $M$ 和 reduce 任务 $R_1, R_2$，令 $e(R_i)$ 表示 $R_i$ 提交的执行，上述较弱的语义出现的原因是 $e(R_1)$ 可能读取了某个 $M$ 执行的输出，而 $e(R_2)$ 可能读取另一个 $M$ 执行的输出

## 3.4 Locality 
Network bandwidth is a relatively scarce resource in our computing environment. We conserve network bandwidth by taking advantage of the fact that the input data (managed by GFS [8]) is stored on the local disks of the machines that make up our cluster. GFS divides each file into $64\mathrm{MB}$ blocks, and stores several copies of each block (typically 3 copies) on different machines. The MapReduce master takes the location information of the input files into account and attempts to schedule a map task on a machine that contains a replica of the corresponding input data. Failing that, it attempts to schedule a map task near a replica of that task’s input data (e.g., on a worker machine that is on the same network switch as the machine containing the data). When running large MapReduce operations on a significant fraction of the workers in a cluster, most input data is read locally and consumes no network bandwidth. 
>  Google 的计算环境中，网络带宽是相对稀缺资源
>  我们通过让输入数据由 GFS 管理，存储在集群机器的本地磁盘上，来节省网络带宽。GFS 将每个文件划分为 64MB 的块，将每个块的多个拷贝 (一般 3 个) 存储在不同的机器上
>  MapReduce master 会考虑 MapReduce 任务的输入文件的位置信息，尝试在包含了对应输入数据的副本的机器上调度 map 任务。如果失败，则尝试在其临近的机器上调度 map 任务 (例如在同一网络交换机下)
>  当在集群中的大部分机器上运行大规模 MapReduce 任务时，大多数输入数据都是在本地读取的，不会消耗网络带宽

## 3.5 Task Granularity 
We subdivide the map phase into $M$ pieces and the reduce phase into $R$ pieces, as described above. Ideally, $M$ and $R$ should be much larger than the number of worker machines. Having each worker perform many different tasks improves dynamic load balancing, and also speeds up recovery when a worker fails: the many map tasks it has completed can be spread out across all the other worker machines. 
>  我们将 map 阶段划分为 M 部分，reduce 阶段划分为 R 部分，理想情况下，M 和 R 应该远大于工作机器的数量，每个工作机执行多个不同的任务可以改善动态负载均衡，并且在某个工作机器故障时也可以加快恢复速度：该机器故障时，它已经完成的 map 任务需要重新执行，这些任务可以分散给所有其他工作机器执行

There are practical bounds on how large $M$ and $R$ can be in our implementation, since the master must make $O(M+R)$ scheduling decisions and keeps $O(M*R)$ state in memory as described above. (The constant factors for memory usage are small however: the $O(M*R)$ piece of the state consists of approximately one byte of data per map task/reduce task pair.) 
>  实践中，M 和 R 的大小存在限制，因为 master 必须执行 $O(M+R)$ 的调度决策，并且在内存中保存 $O(M*R)$ 的状态信息 (不过内存用量的常数因子很小，大约每对 map 任务-reduce 任务对的状态信息仅占一字节)

Furthermore, $R$ is often constrained by users because the output of each reduce task ends up in a separate output file. In practice, we tend to choose $M$ so that each individual task is roughly $16\text{MB}$ to $64\mathrm{MB}$ of input data (so that the locality optimization described above is most effective), and we make $R$ a small multiple of the number of worker machines we expect to use. We often perform MapReduce computations with $M=200,000$ and $R=5,000$ , using 2,000 worker machines. 
>  另外，R 常常会由用户限制，因为每个 reduce 任务的输出最终会放在单独的输出文件中
>  实践中，通常 M 的选择标准是让每个单独的任务大约有 16MB 到 64MB 的输入数据，这也使得上面描述的局部性优化效果最佳 (16MB 到 64 MB 都刚好可以放在一个 GFS 块里)，R 的选择标准通常是期望使用的 worker 机器数量的一个小倍数
>  Google 常常使用的 MapReduce 参数是 $M=200000, R=5000$，使用 2000 台 worker 机器

## 3.6 Backup Tasks 
One of the common causes that lengthens the total time taken for a MapReduce operation is a “straggler”: a machine that takes an unusually long time to complete one of the last few map or reduce tasks in the computation. Stragglers can arise for a whole host of reasons. For example, a machine with a bad disk may experience frequent correctable errors that slow its read performance from $30~\mathrm{MB/s}$ to $1\mathrm{MB/s}$ . The cluster scheduling system may have scheduled other tasks on the machine, causing it to execute the MapReduce code more slowly due to competition for CPU, memory, local disk, or network bandwidth. 
>  常见的导致 MapReduce 操作总时间延长的一个原因是拖后腿者，即某个机器在计算最后几个 map 或 reduce 任务时花费了异常长的时间
>  这种情况出现的原因有很多，例如一台硬盘有问题的机器可能会频繁遇到可纠正错误，导致其读取性能从 30MB/s 降低到 1MB/s，又例如集群调度系统可能在该机器上调度了其他任务，导致该机器执行 MapReduce 代码的速度变慢

A recent problem we experienced was a bug in machine initialization code that caused processor caches to be disabled: computations on affected machines slowed down by over a factor of one hundred. 
>  我们最近遇到的一个问题是在机器初始化代码中的一个 bug 导致处理器缓存被禁用，使得受影响的机器上的计算速度降低超过了 100 倍

We have a general mechanism to alleviate the problem of stragglers. When a MapReduce operation is close to completion, the master schedules backup executions of the remaining in-progress tasks. The task is marked as completed whenever either the primary or the backup execution completes. We have tuned this mechanism so that it typically increases the computational resources used by the operation by no more than a few percent. We have found that this significantly reduces the time to complete large MapReduce operations. As an example, the sort program described in Section 5.3 takes $44\%$ longer to complete when the backup task mechanism is disabled. 
>  缓解拖后腿者问题有一个通用机制：当一个 MapReduce 操作接近完成时，master 调度剩余的正在执行中任务的备份执行，当这些剩余任务的主执行或者备份执行其中之一完成了，就标记该任务完成
>  我们调优了该机制，使得它通常只会使操作使用的计算资源增加几个百分点
>  我们发现该机制可以显著减少大型 MapReduce 操作需要完成的时间

# 4 Refinements 
Although the basic functionality provided by simply writing Map and Reduce functions is sufficient for most needs, we have found a few extensions useful. These are described in this section. 

## 4.1 Partitioning Function 
The users of MapReduce specify the number of reduce tasks/output files that they desire $(R)$ . Data gets partitioned across these tasks using a partitioning function on the intermediate key. A default partitioning function is provided that uses hashing (e.g. $\cdot h a s h(k e y)$ mod $\begin{array}{r}{R}\end{array}$ ). This tends to result in fairly well-balanced partitions. In some cases, however, it is useful to partition data by some other function of the key. For example, sometimes the output keys are URLs, and we want all entries for a single host to end up in the same output file. To support situations like this, the user of the MapReduce library can provide a special partitioning function. For example, using $hash(Hostname(urlkey))\  \mathrm{mod}\ R$ as the partitioning function causes all URLs from the same host to end up in the same output file. 
>  MapReduce 用户负责指定所需要的 reduce 任务/输出文件数量 R，数据会通过中间键上的划分函数被分配到这 R 个任务中
>  默认提供的划分函数使用哈希 (例如 $hash(key)\ \mathrm{mod}\ R$)，这通常会产生较为平衡的分区
>  一些情况下，可以考虑用其他的划分函数，例如有时输出键是 URL，并且我们希望单个主机的所有条目都位于相同的输出文件中，用户可以提供自定义的划分函数，例如 $hash(Hostname(urlkey))\ \mathrm{mod}\ R$，使得带有相同主机的 URL 键都被分配到相同的输出文件中

## 4.2 Ordering Guarantees 
We guarantee that within a given partition, the intermediate key/value pairs are processed in increasing key order. This ordering guarantee makes it easy to generate a sorted output file per partition, which is useful when the output file format needs to support efficient random access lookups by key, or users of the output find it convenient to have the data sorted. 
>  我们保证在给定分区内，中间键值对会按照键的增序处理
>  这一顺序保证使得每个分区可以轻松生成排好序的输出文件，按照键排好序的输出文件会让在该文件中通过键的随机访问查找非常高效

## 4.3 Combiner Function 
In some cases, there is significant repetition in the intermediate keys produced by each map task, and the user-specified Reduce function is commutative and associative. A good example of this is the word counting example in Section 2.1. Since word frequencies tend to follow a Zipf distribution, each map task will produce hundreds or thousands of records of the form `<the, 1>` . All of these counts will be sent over the network to a single reduce task and then added together by the Reduce function to produce one number. 
>  一些情况下，每个 map 任务会生成大量重复的中间键，并且用户定义的 `reduce` 函数是可交换并且可结合的
>  一个例子就是单词计数，因为单词频率倾向于遵循齐普夫分布，每个 map 任务会生成数百个或者数千个形式为 `<the, 1>` 的记录，所有的这些记录会经过网络被发送到单个 reduce 任务中，然后由 `reduce` 函数相加得到一个数字

> [! info] 齐普夫定律
> 齐普夫定律是一条经验性定律，它指出当按降序排列一组测量值时，序列中第 $n$ 个条目的值通常大约与 $n$ 成反比
> 
> 齐普夫定律最著名的应用实例是对文本或自然语言语料库中单词频率表的描述，即
> 
> $$\text{word frequency} \propto \frac {1}{\text{word rank}}$$
> 
> 这意味着在一个给定的文本中，一个单词出现的频率与其排名成反比。例如，最常见的单词出现的频率是第二常见单词的两倍，是第三常见单词的三倍，以此类推。这种关系不仅适用于语言学，在许多其他领域也有其对应的应用。

We allow the user to specify an optional Combiner function that does partial merging of this data before it is sent over the network. 
>  我们允许用于定义一个可选的合并函数，它在将数据发送到网络之前先进行部分合并

The Combiner function is executed on each machine that performs a map task. Typically the same code is used to implement both the combiner and the reduce functions. The only difference between a reduce function and a combiner function is how the MapReduce library handles the output of the function. The output of a reduce function is written to the final output file. The output of a combiner function is written to an intermediate file that will be sent to a reduce task. 
>  合并函数会在每个执行了 map 任务的机器上执行
>  通常 `combiner` 函数和 `reduce` 函数的代码是相同的，二者的差异仅在于 MapReduce 库是如何处理这两个函数的输出的：`reduce` 函数的输出会被写入最终输出文件，`combiner` 函数的输出会被写入一个中间文件，该中间文件会被发送给一个 reduce 任务

Partial combining significantly speeds up certain classes of MapReduce operations. Appendix A contains an example that uses a combiner. 
>  部分合并可以显著加快某些 MapReduce 操作

>  直观上看，`combiner` 就是由 map worker 自己先对自己输出的中间键值对进行了一次归约，如果指定的归约操作是可交换且可结合的，则归约顺序就不重要，那么这样做不仅节约了网络资源，也不会影响最终结果

### 4.4 Input and Output Types 
The MapReduce library provides support for reading input data in several different formats. For example, “text” mode input treats each line as a key/value pair: the key is the offset in the file and the value is the contents of the line. Another common supported format stores a sequence of key/value pairs sorted by key. 
>  MapReduce 库为读取多种不同格式的输入数据提供了支持
>  例如，"文本" 模式输入将每一行视作一个键值对，键是文件中的偏移量 (即行号)，值是该行的内容
>  另一种常见的支持格式将键值对序列按照键排序

Each input type implementation knows how to split itself into meaningful ranges for processing as separate map tasks (e.g. text mode’s range splitting ensures that range splits occur only at line boundaries). Users can add support for a new input type by providing an implementation of a simple reader interface, though most users just use one of a small number of predefined input types. 
>  MapReduce 库的每种输入类型实现都知道如何将输入划分为有意义的区间以作为单独的 map 任务进行处理 (例如文本模式下，其区间划分确保仅在行的边界处发生)
>  用户可以通过实现一个简单的 `reader` 接口为新的输入类型添加支持

A reader does not necessarily need to provide data read from a file. For example, it is easy to define a reader that reads records from a database, or from data structures mapped in memory. 
>  `reader` 并不一定需要提供从文件中读取到的数据，例如，可以定义一个 `reader` 从数据库中读取记录，或者从内存中映射的数据结构读取数据

In a similar fashion, we support a set of output types for producing data in different formats and it is easy for user code to add support for new output types. 
>  类似地，MapReduce 库支持一组输出类型，以生成不同格式的输出数据，用户也可以为新的输出类型添加支持

## 4.5 Side-effects 
In some cases, users of MapReduce have found it convenient to produce auxiliary files as additional outputs from their map and/or reduce operators. We rely on the application writer to make such side-effects atomic and idempotent. Typically the application writes to a temporary file and atomically renames this file once it has been fully generated. 
>  一些情况下，MapReduce 用户发现让其 `map` 或 `reduce` 函数生成辅助文件作为额外输出非常方便，但这类副作用需要由程序编写者确保其原子性和幂等性
>  常见的副作用就是写入一个临时文件，然后在文件完全生成后将其原子性地重命名

We do not provide support for atomic two-phase commits of multiple output files produced by a single task. Therefore, tasks that produce multiple output files with cross-file consistency requirements should be deterministic. This restriction has never been an issue in practice. 
>  我们不支持对单次任务生成的多个输出文件进行原子性两阶段提交，因此，产生多个输出文件并且要求跨文件一致性的任务应该是确定性的

## 4.6 Skipping Bad Records 
Sometimes there are bugs in user code that cause the Map or Reduce functions to crash deterministically on certain records. Such bugs prevent a MapReduce operation from completing. The usual course of action is to fix the bug, but sometimes this is not feasible; perhaps the bug is in a third-party library for which source code is unavailable. Also, sometimes it is acceptable to ignore a few records, for example when doing statistical analysis on a large data set. We provide an optional mode of execution where the MapReduce library detects which records cause deterministic crashes and skips these records in order to make forward progress. 
>  有时用户代码中的 bug 会导致 `map` 或 `reduce` 函数在处理特定的记录时确定性地崩溃，这样的 bug 组织 MapReduce 操作完成
>  通常的解决方法就是修复 bug，但有时这不可行，例如 bug 来自于某个没有源代码的第三方库中，另外，有时忽略一些记录也是可以接受的，例如对大数据集进行统计分析时
>  因此，我们提供了一个可选的选项，当 MapReduce 库在执行时检查到哪个记录会导致确定性的崩溃时，可以跳过处理这些记录

Each worker process installs a signal handler that catches segmentation violations and bus errors. Before invoking a user Map or Reduce operation, the MapReduce library stores the sequence number of the argument in a global variable. If the user code generates a signal, the signal handler sends a “last gasp” UDP packet that contains the sequence number to the MapReduce master. When the master has seen more than one failure on a particular record, it indicates that the record should be skipped when it issues the next re-execution of the corresponding Map or Reduce task. 
>  每个 worker 进程都会安装一个信号处理程序，该程序捕获段错误和总线错误 (发生错误时，OS 会发送信号，例如 `SIGINT`，信号处理程序会处理这些信号)
>  MapReduce 库在调用用户提供的 `map` 或 `reduce` 操作时，会将其参数 (即某个记录) 的序列号存储在一个全局变量上，如果用户代码生成了信号，信号处理程序会发送一个包含了该序列号的 "最后一搏" UDP 包给 master，该 UDP 包包含了参数的序列号
>  当 master 看到特定的记录发生了多次故障之后，它在下一次调度执行相应的 `map` 或 `reduce` 任务时，会指示该记录应该被跳过

## 4.7 Local Execution 
Debugging problems in Map or Reduce functions can be tricky, since the actual computation happens in a distributed system, often on several thousand machines, with work assignment decisions made dynamically by the master. To help facilitate debugging, profiling, and small-scale testing, we have developed an alternative implementation of the MapReduce library that sequentially executes all of the work for a MapReduce operation on the local machine. Controls are provided to the user so that the computation can be limited to particular map tasks. Users invoke their program with a special flag and can then easily use any debugging or testing tools they find useful (e.g. `gdb`). 
>  debug `map` 或 `reduce` 函数中的问题较为麻烦，因为实际的计算发生在分布式系统中，通常涉及数千台机器，由 master 动态做出工作分配决策
>  我们开发了 MapReduce 库的一个替代实现，它在本地机器上顺序执行 MapReduce 操作的所有工作，并为用户提供了控制功能，使得计算可以限制在特定的 map 任务，debug 时可以对该替代实现进行 debug

## 4.8 Status Information 
The master runs an internal HTTP server and exports a set of status pages for human consumption. The status pages show the progress of the computation, such as how many tasks have been completed, how many are in progress, bytes of input, bytes of intermediate data, bytes of output, processing rates, etc. The pages also contain links to the standard error and standard output files generated by each task. 
>  master 运行一个内部的 HTTP 服务器，并导出一系列供用户观察的状态页面，状态页面展示了计算的进度，例如已经完成的任务数量、正在进行的任务数量、输入的字节数、中间数据的字节数、输出的字节数、处理速率等，页面中还包含了指向每个任务生成的标准输出文件和标准错误文件的链接

The user can use this data to predict how long the computation will take, and whether or not more resources should be added to the computation. These pages can also be used to figure out when the computation is much slower than expected. 
>  用户可以基于这些数据判断计算需要多长时间，以及是否应该向计算中添加资源等

In addition, the top-level status page shows which workers have failed, and which map and reduce tasks they were processing when they failed. This information is useful when attempting to diagnose bugs in the user code. 
>  另外，顶级状态页面展示了哪些 workers 出现故障，以及它们故障时正在处理哪些 map 和 reduce 任务

## 4.9 Counters 
The MapReduce library provides a counter facility to count occurrences of various events. For example, user code may want to count total number of words processed or the number of German documents indexed, etc. 
>  MapReduce 库提供了一个计数器功能来统计各种事件发生的次数，例如，用户代码可能希望统计处理的总单词数量等等

To use this facility, user code creates a named counter object and then increments the counter appropriately in the Map and/or Reduce function. For example: 
>  要使用该功能，用户代码需要创建一个命名的计数器对象，然后在 `map` 或 `reduce` 函数中适当增加该计数器的值，例如：

```
Counter* uppercase;
uppercase = GetCounter("uppercase");

map(String name, String contents):
    for each word w in contents:
        if (IsCapitialized(w)):
            uppercase->Increment();
        EmitIntermediate(w, "1");
```

The counter values from individual worker machines are periodically propagated to the master (piggybacked on the ping response). The master aggregates the counter values from successful map and reduce tasks and returns them to the user code when the MapReduce operation is completed. The current counter values are also displayed on the master status page so that a human can watch the progress of the live computation. When aggregating counter values, the master eliminates the effects of duplicate executions of the same map or reduce task to avoid double counting. (Duplicate executions can arise from our use of backup tasks and from re-execution of tasks due to failures.) 
>  单个 worker 机器上的计数器值会周期性地传播到 master (附加在 ping 相应报文中)，master 从成功地 map 和 reduce 任务中汇总计数器值，然后在 MapReduce 操作结束后将其返回给用户代码
>  当前的计数器值也会在 master 状态页面上展示，便于用户监控
>  当聚合计数器值时，master 会消除相同的 map 或 reduce 任务重复执行的影响，以避免重复计数 (因为我们采用了备份任务和重复执行等手段，故可能会出现重复执行的情况)

Some counter values are automatically maintained by the MapReduce library, such as the number of input key/value pairs processed and the number of output key/value pairs produced. 
>  一些计数器值会自动由 MapReduce 库维护，例如处理的输入键值对的数量和产生的输出键值对的数量

Users have found the counter facility useful for sanity checking the behavior of MapReduce operations. For example, in some MapReduce operations, the user code may want to ensure that the number of output pairs produced exactly equals the number of input pairs processed, or that the fraction of German documents processed is within some tolerable fraction of the total number of documents processed. 
>  计数器功能对于验证 MapReduce 操作的行为很有用
>  例如，一些 MapReduce 操作中，用户代码希望确保生成的输出对的数量能准确等于处理的输出对的数量

# 5 Performance 
In this section we measure the performance of MapReduce on two computations running on a large cluster of machines. One computation searches through approximately one terabyte of data looking for a particular pattern. The other computation sorts approximately one terabyte of data. 
>  我们测量在大型机器集群上运行两类 MapReduce 计算的性能
>  一个计算是在大约 1TB (1024GB) 的数据中搜索特定模式
>  另一个计算是对大约 1TB 的数据进行排序

These two programs are representative of a large subset of the real programs written by users of MapReduce – one class of programs shuffles data from one representation to another, and another class extracts a small amount of interesting data from a large data set. 
>  这两个程序代表了大量由 MapReduce 用户编写的实际程序——一类程序将数据从一种表示形式转化为另一种表示形式，另一类程序从大数据集中提取出少量感兴趣的数据

## 5.1 Cluster Configuration 
All of the programs were executed on a cluster that consisted of approximately 1800 machines. Each machine had two 2GHz Intel Xeon processors with Hyper-Threading enabled, 4GB of memory, two 160GB IDE disks, and a gigabit Ethernet link. The machines were arranged in a two-level tree-shaped switched network with approximately 100-200 Gbps of aggregate bandwidth available at the root. All of the machines were in the same hosting facility and therefore the round-trip time between any pair of machines was less than a millisecond. 
>  所有的程序在大约包含 1800 个机器的集群上运行，每个机器包含两个 2GHz Intel Xeon 处理器，启用超线程，4GB 内存，两个 160GB IDE 磁盘和一个 GB 级别 (千兆) 以太网链路
>  这些机器安排在一个两层的树形结构交换机网络中，树的根部提供了大约 100-200Gbps 的聚合带宽
>  所有机器位于同一托管设施中，任意两台机器之间的往返时间小于 1ms

Out of the 4GB of memory, approximately 1-1.5GB was reserved by other tasks running on the cluster. The programs were executed on a weekend afternoon, when the CPUs, disks, and network were mostly idle. 
>  机器的 4GB 内存中，大约 1-1.5 GB 会保留用于运行集群的其他任务

## 5.2 Grep 
The grep program scans through $10^{10}$ 100-byte records, searching for a relatively rare three-character pattern (the pattern occurs in 92,337 records). The input is split into approximately 64MB pieces ( $M=15000;$ ), and the entire output is placed in one file $(R=1$ ). 
>  grep 程序扫描 $10^{10}$ 个 100 字节大小的记录，搜索一个相对少见的三字符模式
>  输入数据被划分为 $M=15000$ 个大约 64MB 的片段，输出放在 $R=1$ 个文件中

![[pics/MapReduce-Fig2.png]]

Figure 2 shows the progress of the computation over time. The Y-axis shows the rate at which the input data is scanned. The rate gradually picks up as more machines are assigned to this MapReduce computation, and peaks at over $30\mathrm{GB}/\mathrm{s}$ when 1764 workers have been assigned. As the map tasks finish, the rate starts dropping and hits zero about 80 seconds into the computation. 
>  图 2 展示了计算随时间的进展。Y 轴显示输入数据被扫描的速度
>  随着越来越多的机器被分配到这个 MapReduce 计算中，扫描速度逐渐提高，并在分配了 1764 个 worker 时达到峰值，超过 $30\mathrm{GB}/\mathrm{s}$。随着 map 任务的完成，扫描速度开始下降，并在计算进行到大约 80 秒时降至零。

The entire computation takes approximately 150 seconds from start to finish. This includes about a minute of startup overhead. The overhead is due to the propagation of the program to all worker machines, and delays interacting with GFS to open the set of 1000 input files and to get the information needed for the locality optimization. 
>  整个计算从开始到结束大约需要 150 秒。这包括了大约一分钟的启动开销，该开销包括了将程序传播到所有 worker 机器上，以及与 GFS 交互以打开 1000 个输入文件并获取局部性优化所需信息所导致的延迟

## 5.3 Sort 
The sort program sorts $10^{10}$ 100-byte records (approximately 1 terabyte of data). This program is modeled after the TeraSort benchmark [10]. 
>  sort 程序对 $10^{10}$ 个 100 字节的记录进行排序，该程序基于 TeraSort 基准建模

The sorting program consists of less than 50 lines of user code. A three-line Map function extracts a 10-byte sorting key from a text line and emits the key and the original text line as the intermediate key/value pair. We used a built-in Identity function as the Reduce operator. This functions passes the intermediate key/value pair unchanged as the output key/value pair. The final sorted output is written to a set of 2-way replicated GFS files (i.e., 2 terabytes are written as the output of the program). 
>  排序程序包含的用户代码行数少于 50 行，其中三行的 `map` 函数从文本行中提取 10 字节的排序键，然后将该键和原来的文本作为中间键值对发出，`reduce` 函数使用内建的 Identity 函数，该函数直接将接受的中间键值对返回为输出键值对
>  最终排序好的输出写入一组两路复制的 GFS 文件，即程序的输出为 2TB

As before, the input data is split into 64MB pieces ($M=15000$ ). We partition the sorted output into 4000 files ($\mathit{R}=4000$ ). The partitioning function uses the initial bytes of the key to segregate it into one of $R$ pieces. 
>  输入数据被划分为 $M=15000$ 个 64MB 的片段，排序好的输出被划分为 $R=4000$ 个文件，划分函数使用键的起始字节作为划分依据

Our partitioning function for this benchmark has builtin knowledge of the distribution of keys. In a general sorting program, we would add a pre-pass MapReduce operation that would collect a sample of the keys and use the distribution of the sampled keys to compute split-points for the final sorting pass. 
>  我们的使用的划分函数内建了对该基准测试的键分布的了解，在通用的排序程序中，我们会添加一个预处理 MapReduce 操作，该操作会收集键的样本，使用采样的键的分布以计算最终排序阶段的划分点

![[pics/MapReduce-Fig3.png]]

Figure 3 (a) shows the progress of a normal execution of the sort program. The top-left graph shows the rate at which input is read. The rate peaks at about $13\mathrm{GB/s}$ and dies off fairly quickly since all map tasks finish before 200 seconds have elapsed. Note that the input rate is less than for grep. This is because the sort map tasks spend about half their time and I/O bandwidth writing intermediate output to their local disks. The corresponding intermediate output for grep had negligible size. 
>  图 3 (a) 展示了排序程序正常执行的进度
>  左上角的图表显示了输入数据的读取速率。该速率峰值约为 $13\text{GB/s}$，并且很快下降，因为所有 map 任务在 200 秒内完成
>  注意到 sort 程序的输入速率低于之前 grep 的输入速率。这是因为排序 map 任务大约有一半的时间和 I/O 带宽用于将中间输出写入本地磁盘，而 grep 对应的中间输出大小可以忽略不计

The middle-left graph shows the rate at which data is sent over the network from the map tasks to the reduce tasks. This shuffling starts as soon as the first map task completes. The first hump in the graph is for the first batch of approximately 1700 reduce tasks (the entire MapReduce was assigned about 1700 machines, and each machine executes at most one reduce task at a time). Roughly 300 seconds into the computation, some of these first batch of reduce tasks finish and we start shuffling data for the remaining reduce tasks. All of the shuffling is done about 600 seconds into the computation. 
>  中间左方的图表显示了从 map 任务到 reduce 任务的在网络上的数据发送速率
>  一旦第一个 map 任务完成，这种数据混洗就开始了。图表中的第一个波峰对应于大约 1700 个 reduce 任务的第一批（整个 MapReduce 分配了大约 1700 台机器，每台机器一次最多执行一个减少任务）
>  计算进行到约 300 秒时，第一批 reduce 任务中的一部分完成了，我们开始为剩余的 reduce 任务进行数据混洗
>  所有混洗操作大约在计算进行到 600 秒时完成

The bottom-left graph shows the rate at which sorted data is written to the final output files by the reduce tasks. There is a delay between the end of the first shuffling period and the start of the writing period because the machines are busy sorting the intermediate data. The writes continue at a rate of about 2-4 GB/s for a while. All of the writes finish about 850 seconds into the computation. Including startup overhead, the entire computation takes 891 seconds. This is similar to the current best reported result of 1057 seconds for the TeraSort benchmark [18]. 
>  底左图显示了 reduce 任务向最终输出文件写入排序数据的速率
>  在第一次 shuffle 阶段结束和写入阶段开始之间有一段延迟，因为机器正在忙于对中间数据进行排序。写入速率持续保持在大约 2-4 GB/s 一段时间。所有的写入操作在计算开始约 850 秒后完成。
>  包括启动开销在内，整个计算过程耗时 891 秒。这与目前最佳报告结果 1057 秒用于 TeraSort 基准测试的结果相似[18]。

A few things to note: the input rate is higher than the shuffle rate and the output rate because of our locality optimization – most data is read from a local disk and bypasses our relatively bandwidth constrained network. The shuffle rate is higher than the output rate because the output phase writes two copies of the sorted data (we make two replicas of the output for reliability and availability reasons). We write two replicas because that is the mechanism for reliability and availability provided by our underlying file system. Network bandwidth requirements for writing data would be reduced if the underlying file system used erasure coding [14] rather than replication. 
>  输入速率高于 shuffle 速率和输出速率的原因是局部性优化——大多数数据从局部磁盘读取，绕过了带宽受限的网络
>  shuffle 速率高于输出速率的原因是输出阶段写出了两份排序好的数据，写两份副本的原因在于这是 GFS 提供的可靠性和可用性的机制，如果底层文件系统使用纠错码而不是复制来写入数据，写出的带宽需求会减少

## 5.4 Effect of Backup Tasks 
In Figure 3 (b), we show an execution of the sort program with backup tasks disabled. The execution flow is similar to that shown in Figure 3 (a), except that there is a very long tail where hardly any write activity occurs. After 960 seconds, all except 5 of the reduce tasks are completed. However these last few stragglers don’t finish until 300 seconds later. The entire computation takes 1283 seconds, an increase of $44\%$ in elapsed time. 
>  Figure 3b 展示了禁用备份任务时的执行情况，执行流和 Figure 3a 类似，差异在于存在一个非常长的尾部，在此期间几乎没有任何写操作
>  960 秒后，除了 5 个规约任务外，所有任务都已完成。然而，最后这几个缓慢的任务直到 300 秒后才完成。整个计算耗时 1283 秒，相比之前增加了 44%的时间

## 5.5 Machine Failures 
In Figure 3 (c), we show an execution of the sort program where we intentionally killed 200 out of 1746 worker processes several minutes into the computation. The underlying cluster scheduler immediately restarted new worker processes on these machines (since only the processes were killed, the machines were still functioning properly). 
>  在图 3 (c) 中，我们展示了在计算进行了几分钟后，故意终止了其中的 200 个 worker 进程中的 1746 个时 sort 的执行情况，可以发现底层的集群调度器立即在这台机器上重启了新的 worker 进程（因为只是进程被终止，机器本身仍然正常运行）

The worker deaths show up as a negative input rate since some previously completed map work disappears (since the corresponding map workers were killed) and needs to be redone. The re-execution of this map work happens relatively quickly. The entire computation finishes in 933 seconds including startup overhead (just an increase of $5\%$ over the normal execution time). 
>  worker 进程的终止表现为输入速率下降，因为一些先前已完成的 map 工作消失了（因为相应的映射工人进程被终止），需要重新执行。这部分 map 工作的重新执行相对较快
>  整个计算在包括启动开销的情况下在 933 秒内完成（比正常执行时间增加了 5%）

# 6 Experience 
We wrote the first version of the MapReduce library in February of 2003, and made significant enhancements to it in August of 2003, including the locality optimization, dynamic load balancing of task execution across worker machines, etc. Since that time, we have been pleasantly surprised at how broadly applicable the MapReduce library has been for the kinds of problems we work on. 

It has been used across a wide range of domains within Google, including: 

- large-scale machine learning problems, 
- clustering problems for the Google News and Froogle products, 
- extraction of data used to produce reports of popular queries (e.g. Google Zeitgeist), 
- extraction of properties of web pages for new experiments and products (e.g. extraction of geographical locations from a large corpus of web pages for localized search), and 
- large-scale graph computations. 

>  MapReduce 已被广泛用于谷歌的各个领域，包括：
> - 大规模机器学习问题，
> - 为 Google 新闻和 Froogle 产品进行的聚类问题，
> - 提取用于生成热门查询报告的数据（例如 Google Zeitgeist），
> - 从网页中提取属性以用于新的实验和产品（例如从大量网页中提取地理位置以实现本地化搜索），
> - 大规模图计算

![[pics/MapReduce-Fig4.png]]

Figure 4 shows the significant growth in the number of separate MapReduce programs checked into our primary source code management system over time, from 0 in early 2003 to almost 900 separate instances as of late September 2004. MapReduce has been so successful because it makes it possible to write a simple program and run it efficiently on a thousand machines in the course of half an hour, greatly speeding up the development and prototyping cycle. Furthermore, it allows programmers who have no experience with distributed and/or parallel systems to exploit large amounts of resources easily. 
>  图 4 显示了随着时间的推移，我们主要的源代码管理系统中单独的 MapReduce 程序的数量显著增长，从 2003 年初的 0 个增加到 2004 年 9 月底接近 900 个实例。MapReduce 之所以如此成功，是因为它使得编写一个简单的程序并在半小时内在一千台机器上高效运行成为可能，大大加快了开发和原型设计的周期。此外，它还让没有分布式和/或并行系统经验的程序员能够轻松利用大量的资源。

![[pics/MapReduce-Table1.png]]

At the end of each job, the MapReduce library logs statistics about the computational resources used by the job. In Table 1, we show some statistics for a subset of MapReduce jobs run at Google in August 2004. 
>  在每个任务结束时，MapReduce 库会记录该任务所使用的计算资源的统计信息。在表 1 中，我们展示了 2004 年 8 月在 Google 运行的部分 MapReduce 任务的一些统计信息。

## 6.1 Large-Scale Indexing 
One of our most significant uses of MapReduce to date has been a complete rewrite of the production indexing system that produces the data structures used for the Google web search service. The indexing system takes as input a large set of documents that have been retrieved by our crawling system, stored as a set of GFS files. The raw contents for these documents are more than 20 terabytes of data. The indexing process runs as a sequence of five to ten MapReduce operations. Using MapReduce (instead of the ad-hoc distributed passes in the prior version of the indexing system) has provided several benefits: 

- The indexing code is simpler, smaller, and easier to understand, because the code that deals with fault tolerance, distribution and parallelization is hidden within the MapReduce library. For example, the size of one phase of the computation dropped from approximately 3800 lines of $\mathrm{C}{+}{+}$ code to approximately 700 lines when expressed using MapReduce. 
- The performance of the MapReduce library is good enough that we can keep conceptually unrelated computations separate, instead of mixing them together to avoid extra passes over the data. This makes it easy to change the indexing process. For example, one change that took a few months to make in our old indexing system took only a few days to implement in the new system. 
- The indexing process has become much easier to operate, because most of the problems caused by machine failures, slow machines, and networking hiccups are dealt with automatically by the MapReduce library without operator intervention. Furthermore, it is easy to improve the performance of the indexing process by adding new machines to the indexing cluster. 

>  我们迄今为止最重要的 MapReduce 应用之一是对生产索引系统进行了全面重写，该系统用于生成用于 Google 网页搜索服务的数据结构。索引系统以我们的爬虫系统检索到的一大批文档作为输入，这些文档存储为一组 GFS 文件。这些文档的原始内容超过 20TB 的数据。索引过程运行时会经过五到十个 MapReduce 操作。使用 MapReduce（而不是旧版索引系统的临时分布式传递）带来了几个好处：
> 
> - 索引代码更简单、更小且更容易理解，因为处理容错、分布和并行化的代码被隐藏在 MapReduce 库中。例如，当使用 MapReduce 表达时，计算的一个阶段的代码从大约 3800 行 C++代码减少到了大约 700 行。
> - MapReduce 库的性能足够好，这促使我们将概念上不相关的计算分开，而不是为了减少数据的额外传递而将它们混合在一起。这使得改变索引过程变得容易。例如，在旧索引系统中需要几个月才能完成的更改，在新系统中仅需几天即可实现。
> - 索引过程的操作变得更加容易，因为大多数由机器故障、慢速机器和网络问题引起的麻烦都可以由 MapReduce 库自动处理，无需人工干预。此外，通过向索引集群添加新机器，可以轻松提高索引过程的性能。

# 7 Related Work 
Many systems have provided restricted programming models and used the restrictions to parallelize the computation automatically. For example, an associative function can be computed over all prefixes of an $N$ element array in $\log N$ time on $N$ processors using parallel prefix computations [6, 9, 13]. MapReduce can be considered a simplification and distillation of some of these models based on our experience with large real-world computations. More significantly, we provide a fault-tolerant implementation that scales to thousands of processors. In contrast, most of the parallel processing systems have only been implemented on smaller scales and leave the details of handling machine failures to the programmer. 
>  许多系统提供了受限的编程模型，并利用这些限制自动并行化计算。例如，通过并行前缀计算，可以在 $N$ 个处理器上用 $\log N$ 的时间计算出一个 $N$ 元素数组的所有前缀的关联函数[6, 9, 13]。
>  根据我们在大规模实际计算中的经验，MapReduce 可以被视为对这些模型的一种简化和提炼。更重要的是，我们提供了一种可扩展到数千个处理器的容错实现。相比之下，大多数并行处理系统仅在较小规模上实现，并将处理机器故障的细节留给程序员。

Bulk Synchronous Programming [17] and some MPI primitives [11] provide higher-level abstractions that make it easier for programmers to write parallel programs. A key difference between these systems and MapReduce is that MapReduce exploits a restricted programming model to parallelize the user program automatically and to provide transparent fault-tolerance. 
>  批量同步编程（Bulk Synchronous Programming, BSP）[17] 和一些 MPI 原语（primitives）[11] 提供了更高层次的抽象，使程序员更容易编写并行程序。这些系统与 MapReduce 的一个关键区别在于，MapReduce 利用受限的编程模型自动并行化用户程序，并提供透明的容错性。

Our locality optimization draws its inspiration from techniques such as active disks [12, 15], where computation is pushed into processing elements that are close to local disks, to reduce the amount of data sent across I/O subsystems or the network. We run on commodity processors to which a small number of disks are directly connected instead of running directly on disk controller processors, but the general approach is similar. 
>  我们的本地优化技术借鉴了诸如活动磁盘[12, 15]等技术的灵感，这些技术将计算推送到靠近本地磁盘的处理单元，以减少通过 I/O 子系统或网络传输的数据量。我们运行在普通的处理器上，这些处理器直接连接了少量的磁盘，而不是直接在磁盘控制器处理器上运行，但总体方法是相似的。

Our backup task mechanism is similar to the eager scheduling mechanism employed in the Charlotte System [3]. One of the shortcomings of simple eager scheduling is that if a given task causes repeated failures, the entire computation fails to complete. We fix some instances of this problem with our mechanism for skipping bad records. 
>  我们的备份任务机制类似于夏洛特系统中采用的急切调度机制[3]。简单急切调度的一个缺点是，如果某个任务导致反复失败，整个计算将无法完成。我们通过跳过不良记录的机制来解决这个问题的一些实例。

The MapReduce implementation relies on an in-house cluster management system that is responsible for distributing and running user tasks on a large collection of shared machines. Though not the focus of this paper, the cluster management system is similar in spirit to other systems such as Condor [16]. 
>  MapReduce 的实现依赖于一个内部的集群管理系统，该系统负责在大量共享机器上分配和运行用户任务。虽然这不是本文的重点，但该集群管理系统在理念上与其他系统（如 Condor [16]）类似。

The sorting facility that is a part of the MapReduce library is similar in operation to NOW-Sort [1]. Source machines (map workers) partition the data to be sorted and send it to one of $R$ reduce workers. Each reduce worker sorts its data locally (in memory if possible). Of course NOW-Sort does not have the user-definable Map and Reduce functions that make our library widely applicable.
>  排序功能作为 MapReduce 库的一部分，其操作方式与 NOW-Sort [1]类似。源机器（map workers）将要排序的数据进行分区，并将其发送到 $R$ 个 reduce workers 中的一个。每个 reduce worker 在其本地（如果可能的话，在内存中）对数据进行排序。当然，NOW-Sort 没有用户可定义的 Map 和 Reduce 函数，这使得我们的库具有广泛的应用性。 

River [2] provides a programming model where processes communicate with each other by sending data over distributed queues. Like MapReduce, the River system tries to provide good average case performance even in the presence of non-uniformities introduced by heterogeneous hardware or system perturbations. River achieves this by careful scheduling of disk and network transfers to achieve balanced completion times. MapReduce has a different approach. By restricting the programming model, the MapReduce framework is able to partition the problem into a large number of fine-grained tasks. These tasks are dynamically scheduled on available workers so that faster workers process more tasks. The restricted programming model also allows us to schedule redundant executions of tasks near the end of the job which greatly reduces completion time in the presence of non-uniformities (such as slow or stuck workers). 
>  River 系统提供了一种编程模型，其中进程通过在分布式队列中发送数据来相互通信。与 MapReduce 类似，River 系统试图即使在由异构硬件或系统扰动引入的非均匀性存在的情况下，也能提供良好的平均性能。River 系统通过仔细调度磁盘和网络传输以实现平衡的完成时间来实现这一点。
>  MapReduce 则采取了不同的方法。通过限制编程模型，MapReduce 框架能够将问题划分为大量细粒度的任务。这些任务被动态地调度到可用的工作节点上，以便更快的工作节点处理更多的任务。受限的编程模型还允许我们在作业接近尾声时安排任务的冗余执行，这在存在非均匀性（如缓慢或卡住的工作节点）的情况下可以大大减少完成时间。

BAD-FS [5] has a very different programming model from MapReduce, and unlike MapReduce, is targeted to the execution of jobs across a wide-area network. However, there are two fundamental similarities. (1) Both systems use redundant execution to recover from data loss caused by failures. (2) Both use locality-aware scheduling to reduce the amount of data sent across congested network links. 
> BAD-FS [5] 的编程模型与 MapReduce 有很大的不同，而且与 MapReduce 不同的是，它旨在跨广域网执行作业。然而，两者有两个基本的相似之处。(1) 两个系统都使用冗余执行来从由故障引起的数据丢失中恢复。(2) 两者都使用基于局部性的调度以减少通过拥堵网络链路传输的数据量。 

TACC [7] is a system designed to simplify construction of highly-available networked services. Like MapReduce, it relies on re-execution as a mechanism for implementing fault-tolerance. 
>  TACC [7] 是一个旨在简化高度可用的网络服务构建的系统。与 MapReduce 类似，它依赖于重新执行作为实现容错机制的方法。

# 8 Conclusions 
The MapReduce programming model has been successfully used at Google for many different purposes. We attribute this success to several reasons. First, the model is easy to use, even for programmers without experience with parallel and distributed systems, since it hides the details of parallelization, fault-tolerance, locality optimization, and load balancing. Second, a large variety of problems are easily expressible as MapReduce computations. For example, MapReduce is used for the generation of data for Google’s production web search service, for sorting, for data mining, for machine learning, and many other systems. Third, we have developed an implementation of MapReduce that scales to large clusters of machines comprising thousands of machines. The implementation makes efficient use of these machine resources and therefore is suitable for use on many of the large computational problems encountered at Google. 
>  MapReduce 编程模型已在谷歌成功应用于许多不同的目的。我们认为这一成功的几个原因是：首先，该模型易于使用，即使是那些没有并行和分布式系统经验的程序员也能轻松上手，因为它隐藏了并行化、容错、局部性优化和负载均衡的细节。其次，大量问题可以很容易地表达为 MapReduce 计算。例如，MapReduce 被用于生成谷歌生产网络搜索服务的数据、排序、数据挖掘、机器学习等众多系统。第三，我们开发了一种可扩展到由数千台机器组成的大型集群的 MapReduce 实现。该实现有效地利用了这些机器资源，因此适用于谷歌遇到的许多大型计算问题。

We have learned several things from this work. First, restricting the programming model makes it easy to parallelize and distribute computations and to make such computations fault-tolerant. Second, network bandwidth is a scarce resource. A number of optimizations in our system are therefore targeted at reducing the amount of data sent across the network: the locality optimization allows us to read data from local disks, and writing a single copy of the intermediate data to local disk saves network bandwidth. Third, redundant execution can be used to reduce the impact of slow machines, and to handle machine failures and data loss. 
>  从这项工作中，我们学到了几件事。首先，限制编程模型使得计算和分布计算以及使这些计算具有容错性的过程变得简单。其次，网络带宽是一种稀缺资源。因此，我们系统的许多优化都是针对减少跨网络传输的数据量：局部性优化允许我们从本地磁盘读取数据，并将中间数据写入本地磁盘以节省网络带宽。第三，冗余执行可用于减少慢速机器的影响，并处理机器故障和数据丢失。

# A Word Frequency 
This section contains a program that counts the number of occurrences of each unique word in a set of input files specified on the command line. 

```cpp
#include "mapreduce/mapreduce.h"

// User's map function
class WordCounter: public Mapper {
public:
    virtual void Map(const MapInput& input) {
        const string& text = input.value();
        const int n = text.size();
        for(int i = 0; i < n;) {
            // Skip past leading whitespace
            while((i < n) && isspace(text[i]))
                i++;
        }
        // Find word end
        int start = i;
        while((i < n) && !isspace(text[i]))
            i++;
        if(start < i)
            Emit(text.substr(start, i - start), "1");
    }
};
REGISTER_MAPPER(WordCounter);

// User's reduce function
class Adder: public Reducer {
    virtual void Reduce(RecudeInput* input) {
        // Iterate over all entries with the same key and add the values
        int64 value = 0;
        while(!input->done()) {
            value += StringToInt(input->value());
            input->NextValue();
        }
        
        // Emit sum for input->key()
        Emit(IntToSting(value));
    }
};
REGISTER_REDUCER(Adder);

int main(int argc, char** argv) {
    ParseCommandLineFlags(argc, argv);
    
    MapReduceSpecification spec;
    
    // Store list of input files into "spec"
    for(int i = 1; i < argc; i++) {
        MapReduceInput* input = spec.add_input();
        input->set_format("text");
        input->set_filepattern(argv[i]);
        input->set_mapper_class("WordCounter");
    }
    
    // Specify the output files:
    //    /gfs/test/freq-00000-of-00100
    //    /gfs/test/freq-00001-of-00100
    MapReduceOutput* out = spec.output();
    out->set_filebase("/gfs/test/freq");
    out->set_num_tasks(100):
    out->set_reducer_class("Adder");
    
    // Optional: do partial sums within map tasks to save network bandiwidth
    out->set_combiner_class("Adder");
    
    // Tuning parameters: use at most 2000 machines and 100MB of memory per task
    spec.set_machines(2000)
    spec.set_map_megabytes(100);
    spec.set_reduce_megabytes(100);
    
    // Now run it
    MapReduceResult result;
    if(!MapReduce(spec, &result)) abort();
    
    // Done: 'result' structure contain info about counters, time taken, number of machine used, etc.
    
    return 0;
}
```
